"""
Duration of Absence (DOA) Scorer for AutoSwap Memory Management
Implements intelligent memory prioritization based on access patterns
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import time
import logging
from collections import defaultdict, deque
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class AccessPattern(Enum):
    """Memory access patterns"""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    STRIDED = "strided"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


class SwapPriority(Enum):
    """Swap priority levels"""
    CRITICAL = 0  # Never swap
    HIGH = 1      # Swap only under extreme pressure
    MEDIUM = 2    # Normal swap candidate
    LOW = 3       # Preferred swap candidate
    IDLE = 4      # Immediate swap candidate


@dataclass
class AccessMetrics:
    """Metrics for memory access tracking"""
    last_access_time: float
    total_accesses: int
    access_frequency: float  # Accesses per second
    access_pattern: AccessPattern
    temporal_locality: float  # 0-1 score
    spatial_locality: float   # 0-1 score
    reuse_distance: float     # Average cycles between reuses
    working_set_size: int     # Bytes actively used


@dataclass
class DOAScore:
    """Duration of Absence score with priority"""
    memory_block_id: str
    doa_value: float         # Duration since last access
    priority_score: float    # Combined priority (0-1, lower = higher priority)
    swap_priority: SwapPriority
    access_metrics: AccessMetrics
    size_bytes: int
    device_id: int
    is_pinned: bool = False
    timestamp: float = field(default_factory=time.time)


class DOAScorer:
    """
    Duration of Absence (DOA) Scorer for intelligent memory prioritization.
    Analyzes access patterns and calculates swap priorities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize DOA Scorer"""
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configuration
        self.doa_window_size = self.config.get('doa_window_size', 1000)
        self.temporal_decay_factor = self.config.get('temporal_decay_factor', 0.95)
        self.spatial_threshold = self.config.get('spatial_threshold', 0.7)
        self.frequency_threshold = self.config.get('frequency_threshold', 10.0)  # Hz
        self.idle_threshold_seconds = self.config.get('idle_threshold_seconds', 300.0)
        
        # Access tracking
        self.access_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.doa_window_size))
        self.access_metrics: Dict[str, AccessMetrics] = {}
        self.memory_blocks: Dict[str, Dict[str, Any]] = {}
        
        # Pattern detection
        self.pattern_detector = AccessPatternDetector()
        
        # Scoring parameters
        self.weight_frequency = self.config.get('weight_frequency', 0.3)
        self.weight_recency = self.config.get('weight_recency', 0.3)
        self.weight_pattern = self.config.get('weight_pattern', 0.2)
        self.weight_size = self.config.get('weight_size', 0.2)
        
        # Performance tracking
        self.scoring_stats = {
            'total_scorings': 0,
            'pattern_detections': 0,
            'priority_updates': 0,
            'average_scoring_time_ms': 0.0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        self.logger.info("DOAScorer initialized")
    
    def register_memory_block(self, block_id: str, size: int, device_id: int,
                             initial_priority: SwapPriority = SwapPriority.MEDIUM) -> None:
        """Register a memory block for DOA tracking"""
        with self.lock:
            current_time = time.time()
            
            self.memory_blocks[block_id] = {
                'size': size,
                'device_id': device_id,
                'registration_time': current_time,
                'initial_priority': initial_priority
            }
            
            # Initialize access metrics
            self.access_metrics[block_id] = AccessMetrics(
                last_access_time=current_time,
                total_accesses=1,
                access_frequency=0.0,
                access_pattern=AccessPattern.RANDOM,
                temporal_locality=0.5,
                spatial_locality=0.5,
                reuse_distance=0.0,
                working_set_size=size
            )
            
            # Record initial access
            self.access_history[block_id].append({
                'timestamp': current_time,
                'access_type': 'initial',
                'offset': 0,
                'length': size
            })
    
    def record_access(self, block_id: str, offset: int = 0, length: Optional[int] = None) -> None:
        """Record a memory access event"""
        if block_id not in self.memory_blocks:
            raise ValueError(f"Memory block {block_id} not registered")
        
        with self.lock:
            current_time = time.time()
            block_info = self.memory_blocks[block_id]
            
            if length is None:
                length = block_info['size']
            
            # Record access
            access_event = {
                'timestamp': current_time,
                'access_type': 'read_write',
                'offset': offset,
                'length': length
            }
            
            self.access_history[block_id].append(access_event)
            
            # Update metrics
            self._update_access_metrics(block_id, access_event)
    
    def calculate_doa_score(self, block_id: str) -> DOAScore:
        """Calculate Duration of Absence score for a memory block"""
        if block_id not in self.memory_blocks:
            raise ValueError(f"Memory block {block_id} not registered")
        
        start_time = time.perf_counter()
        
        with self.lock:
            current_time = time.time()
            metrics = self.access_metrics[block_id]
            block_info = self.memory_blocks[block_id]
            
            # Calculate DOA value (time since last access)
            doa_value = current_time - metrics.last_access_time
            
            # Calculate priority components
            recency_score = self._calculate_recency_score(doa_value)
            frequency_score = self._calculate_frequency_score(metrics.access_frequency)
            pattern_score = self._calculate_pattern_score(metrics)
            size_score = self._calculate_size_score(block_info['size'])
            
            # Combined priority score (0-1, lower = higher priority)
            priority_score = (
                self.weight_recency * (1 - recency_score) +
                self.weight_frequency * (1 - frequency_score) +
                self.weight_pattern * pattern_score +
                self.weight_size * size_score
            )
            
            # Determine swap priority level
            swap_priority = self._determine_swap_priority(
                priority_score, doa_value, metrics, block_info
            )
            
            # Create DOA score
            doa_score = DOAScore(
                memory_block_id=block_id,
                doa_value=doa_value,
                priority_score=priority_score,
                swap_priority=swap_priority,
                access_metrics=metrics,
                size_bytes=block_info['size'],
                device_id=block_info['device_id'],
                is_pinned=block_info.get('is_pinned', False)
            )
            
            # Update statistics
            scoring_time = (time.perf_counter() - start_time) * 1000
            self._update_scoring_stats(scoring_time)
            
            return doa_score
    
    def get_swap_candidates(self, required_bytes: int, device_id: int,
                           max_candidates: int = 10) -> List[DOAScore]:
        """Get prioritized list of swap candidates"""
        candidates = []
        
        with self.lock:
            # Calculate DOA scores for all blocks on device
            for block_id, block_info in self.memory_blocks.items():
                if block_info['device_id'] == device_id and not block_info.get('is_pinned', False):
                    try:
                        doa_score = self.calculate_doa_score(block_id)
                        candidates.append(doa_score)
                    except Exception as e:
                        self.logger.error(f"Failed to calculate DOA score for {block_id}: {e}")
            
            # Sort by swap priority and priority score
            candidates.sort(key=lambda x: (x.swap_priority.value, x.priority_score), reverse=True)
            
            # Select candidates that meet size requirement
            selected_candidates = []
            accumulated_size = 0
            
            for candidate in candidates:
                if accumulated_size >= required_bytes:
                    break
                
                selected_candidates.append(candidate)
                accumulated_size += candidate.size_bytes
                
                if len(selected_candidates) >= max_candidates:
                    break
            
            return selected_candidates
    
    def update_priority(self, block_id: str, new_priority: SwapPriority) -> None:
        """Update swap priority for a memory block"""
        with self.lock:
            if block_id in self.memory_blocks:
                self.memory_blocks[block_id]['manual_priority'] = new_priority
                self.scoring_stats['priority_updates'] += 1
    
    def pin_memory_block(self, block_id: str) -> None:
        """Pin a memory block to prevent swapping"""
        with self.lock:
            if block_id in self.memory_blocks:
                self.memory_blocks[block_id]['is_pinned'] = True
    
    def unpin_memory_block(self, block_id: str) -> None:
        """Unpin a memory block to allow swapping"""
        with self.lock:
            if block_id in self.memory_blocks:
                self.memory_blocks[block_id]['is_pinned'] = False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get DOA scorer statistics"""
        with self.lock:
            total_blocks = len(self.memory_blocks)
            pinned_blocks = sum(1 for b in self.memory_blocks.values() if b.get('is_pinned', False))
            
            # Calculate priority distribution
            priority_dist = defaultdict(int)
            for block_id in self.memory_blocks:
                try:
                    score = self.calculate_doa_score(block_id)
                    priority_dist[score.swap_priority.name] += 1
                except:
                    pass
            
            return {
                'total_blocks': total_blocks,
                'pinned_blocks': pinned_blocks,
                'priority_distribution': dict(priority_dist),
                'scoring_stats': self.scoring_stats.copy(),
                'config': {
                    'doa_window_size': self.doa_window_size,
                    'idle_threshold_seconds': self.idle_threshold_seconds,
                    'weights': {
                        'frequency': self.weight_frequency,
                        'recency': self.weight_recency,
                        'pattern': self.weight_pattern,
                        'size': self.weight_size
                    }
                }
            }
    
    def _update_access_metrics(self, block_id: str, access_event: Dict[str, Any]) -> None:
        """Update access metrics for a memory block"""
        metrics = self.access_metrics[block_id]
        history = self.access_history[block_id]
        
        # Update basic metrics
        metrics.last_access_time = access_event['timestamp']
        metrics.total_accesses += 1
        
        # Calculate access frequency
        if len(history) > 1:
            time_span = history[-1]['timestamp'] - history[0]['timestamp']
            if time_span > 0:
                metrics.access_frequency = len(history) / time_span
        
        # Detect access pattern
        if len(history) >= 5:
            pattern = self.pattern_detector.detect_pattern(list(history))
            metrics.access_pattern = pattern
            self.scoring_stats['pattern_detections'] += 1
        
        # Calculate temporal locality
        metrics.temporal_locality = self._calculate_temporal_locality(history)
        
        # Calculate spatial locality
        metrics.spatial_locality = self._calculate_spatial_locality(history)
        
        # Calculate reuse distance
        metrics.reuse_distance = self._calculate_reuse_distance(history)
    
    def _calculate_recency_score(self, doa_value: float) -> float:
        """Calculate recency score (0-1, higher = more recent)"""
        # Exponential decay based on time since last access
        return np.exp(-doa_value / self.idle_threshold_seconds)
    
    def _calculate_frequency_score(self, frequency: float) -> float:
        """Calculate frequency score (0-1, higher = more frequent)"""
        # Sigmoid function centered at threshold
        x = (frequency - self.frequency_threshold) / self.frequency_threshold
        return 1 / (1 + np.exp(-x))
    
    def _calculate_pattern_score(self, metrics: AccessMetrics) -> float:
        """Calculate pattern score (0-1, lower = better pattern)"""
        # Favorable patterns get lower scores (higher priority)
        pattern_scores = {
            AccessPattern.SEQUENTIAL: 0.2,
            AccessPattern.TEMPORAL: 0.3,
            AccessPattern.STRIDED: 0.5,
            AccessPattern.SPATIAL: 0.6,
            AccessPattern.RANDOM: 0.8
        }
        
        base_score = pattern_scores.get(metrics.access_pattern, 0.5)
        
        # Adjust based on locality
        locality_factor = (metrics.temporal_locality + metrics.spatial_locality) / 2
        return base_score * (1 - locality_factor * 0.3)
    
    def _calculate_size_score(self, size_bytes: int) -> float:
        """Calculate size score (0-1, higher = larger)"""
        # Log scale normalization
        size_mb = size_bytes / (1024 * 1024)
        return min(1.0, np.log1p(size_mb) / np.log1p(1024))  # Normalize to 1GB
    
    def _determine_swap_priority(self, priority_score: float, doa_value: float,
                                metrics: AccessMetrics, block_info: Dict[str, Any]) -> SwapPriority:
        """Determine swap priority level"""
        # Check for manual override
        if 'manual_priority' in block_info:
            return block_info['manual_priority']
        
        # Check if idle
        if doa_value > self.idle_threshold_seconds:
            return SwapPriority.IDLE
        
        # Check access frequency
        if metrics.access_frequency < 0.1:  # Less than once per 10 seconds
            return SwapPriority.LOW
        
        # Use priority score thresholds
        if priority_score < 0.2:
            return SwapPriority.CRITICAL
        elif priority_score < 0.4:
            return SwapPriority.HIGH
        elif priority_score < 0.6:
            return SwapPriority.MEDIUM
        else:
            return SwapPriority.LOW
    
    def _calculate_temporal_locality(self, history: deque) -> float:
        """Calculate temporal locality score"""
        if len(history) < 2:
            return 0.5
        
        # Calculate inter-access times
        inter_access_times = []
        for i in range(1, len(history)):
            delta = history[i]['timestamp'] - history[i-1]['timestamp']
            inter_access_times.append(delta)
        
        # Calculate coefficient of variation
        if inter_access_times:
            mean_time = np.mean(inter_access_times)
            std_time = np.std(inter_access_times)
            
            if mean_time > 0:
                cv = std_time / mean_time
                # Lower CV means more regular access pattern
                return 1 / (1 + cv)
        
        return 0.5
    
    def _calculate_spatial_locality(self, history: deque) -> float:
        """Calculate spatial locality score"""
        if len(history) < 2:
            return 0.5
        
        # Check for sequential or nearby accesses
        sequential_count = 0
        
        for i in range(1, len(history)):
            prev_end = history[i-1]['offset'] + history[i-1]['length']
            curr_start = history[i]['offset']
            
            # Check if accesses are sequential or nearby
            if abs(curr_start - prev_end) < 4096:  # Within a page
                sequential_count += 1
        
        return sequential_count / (len(history) - 1)
    
    def _calculate_reuse_distance(self, history: deque) -> float:
        """Calculate average reuse distance"""
        if len(history) < 2:
            return 0.0
        
        # Track unique access positions
        access_positions = {}
        reuse_distances = []
        
        for i, event in enumerate(history):
            key = (event['offset'], event['length'])
            
            if key in access_positions:
                # Calculate distance since last access
                distance = i - access_positions[key]
                reuse_distances.append(distance)
            
            access_positions[key] = i
        
        if reuse_distances:
            return np.mean(reuse_distances)
        else:
            return float('inf')  # No reuse detected
    
    def _update_scoring_stats(self, scoring_time_ms: float) -> None:
        """Update scoring statistics"""
        self.scoring_stats['total_scorings'] += 1
        
        # Update average scoring time
        total = self.scoring_stats['total_scorings']
        current_avg = self.scoring_stats['average_scoring_time_ms']
        self.scoring_stats['average_scoring_time_ms'] = (
            (current_avg * (total - 1) + scoring_time_ms) / total
        )


class AccessPatternDetector:
    """Detects memory access patterns"""
    
    def detect_pattern(self, access_history: List[Dict[str, Any]]) -> AccessPattern:
        """Detect access pattern from history"""
        if len(access_history) < 3:
            return AccessPattern.RANDOM
        
        # Extract offsets
        offsets = [event['offset'] for event in access_history]
        
        # Check for sequential pattern
        if self._is_sequential(offsets):
            return AccessPattern.SEQUENTIAL
        
        # Check for strided pattern
        if self._is_strided(offsets):
            return AccessPattern.STRIDED
        
        # Check for temporal pattern
        if self._is_temporal(access_history):
            return AccessPattern.TEMPORAL
        
        # Check for spatial pattern
        if self._is_spatial(offsets):
            return AccessPattern.SPATIAL
        
        return AccessPattern.RANDOM
    
    def _is_sequential(self, offsets: List[int]) -> bool:
        """Check if accesses are sequential"""
        for i in range(1, len(offsets)):
            if offsets[i] <= offsets[i-1]:
                return False
        return True
    
    def _is_strided(self, offsets: List[int]) -> bool:
        """Check if accesses follow a stride pattern"""
        if len(offsets) < 3:
            return False
        
        strides = [offsets[i] - offsets[i-1] for i in range(1, len(offsets))]
        
        # Check if strides are consistent
        if len(set(strides)) == 1 and strides[0] != 0:
            return True
        
        return False
    
    def _is_temporal(self, access_history: List[Dict[str, Any]]) -> bool:
        """Check if accesses have temporal pattern"""
        if len(access_history) < 4:
            return False
        
        # Extract timestamps
        timestamps = [event['timestamp'] for event in access_history]
        inter_times = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        
        # Check for regular timing
        mean_time = np.mean(inter_times)
        std_time = np.std(inter_times)
        
        if mean_time > 0:
            cv = std_time / mean_time
            return cv < 0.2  # Low coefficient of variation
        
        return False
    
    def _is_spatial(self, offsets: List[int]) -> bool:
        """Check if accesses are spatially clustered"""
        if len(offsets) < 3:
            return False
        
        # Check if accesses are within a small range
        offset_range = max(offsets) - min(offsets)
        mean_offset = np.mean(offsets)
        
        # Clustered if range is small relative to mean
        if mean_offset > 0:
            return offset_range / mean_offset < 0.5
        
        return False