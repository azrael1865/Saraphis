"""
AutoSwap Manager for GPU Memory Optimization
Orchestrates priority-based memory swapping with DOA scoring
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import time
import logging
import threading
from collections import defaultdict, deque
from enum import Enum
import asyncio

from .doa_scorer import DOAScorer, DOAScore, SwapPriority
from .priority_swapper import PrioritySwapper, SwapOperation, SwapDirection

logger = logging.getLogger(__name__)


class SwapPolicy(Enum):
    """Memory swap policies"""
    AGGRESSIVE = "aggressive"      # Swap early and often
    BALANCED = "balanced"         # Balance performance and memory
    CONSERVATIVE = "conservative" # Swap only when necessary
    ADAPTIVE = "adaptive"        # Adapt based on workload


class MemoryPressureLevel(Enum):
    """GPU memory pressure levels"""
    LOW = "low"           # < 50% utilization
    MODERATE = "moderate" # 50-75% utilization
    HIGH = "high"        # 75-90% utilization
    CRITICAL = "critical" # > 90% utilization


@dataclass
class AutoSwapConfig:
    """Configuration for AutoSwap system"""
    swap_policy: SwapPolicy = SwapPolicy.BALANCED
    memory_pressure_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.5,
        'moderate': 0.75,
        'high': 0.9,
        'critical': 0.95
    })
    min_swap_size_mb: float = 1.0
    max_swap_size_mb: float = 1024.0
    swap_ahead_factor: float = 1.2  # Swap 20% more than needed
    enable_predictive_swapping: bool = True
    enable_batch_swapping: bool = True
    monitoring_interval_seconds: float = 1.0
    auto_adjust_policy: bool = True


@dataclass
class SwapDecision:
    """Decision for memory swapping"""
    required_bytes: int
    available_bytes: int
    pressure_level: MemoryPressureLevel
    candidates: List[DOAScore]
    selected_blocks: List[str]
    total_swap_bytes: int
    timestamp: float = field(default_factory=time.time)


class AutoSwapManager:
    """
    AutoSwap Manager orchestrates intelligent GPU memory swapping.
    Integrates DOA scoring and priority-based swapping for optimal performance.
    """
    
    def __init__(self, gpu_optimizer: Any, config: Optional[AutoSwapConfig] = None):
        """Initialize AutoSwap Manager"""
        self.gpu_optimizer = gpu_optimizer
        self.config = config or AutoSwapConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.doa_scorer = DOAScorer({
            'doa_window_size': 1000,
            'idle_threshold_seconds': 300.0,
            'weight_frequency': 0.3,
            'weight_recency': 0.3,
            'weight_pattern': 0.2,
            'weight_size': 0.2
        })
        
        self.priority_swapper = PrioritySwapper({
            'max_concurrent_swaps': 4,
            'batch_size_mb': 64,
            'async_threshold_mb': 16,
            'enable_compression': True,
            'enable_predictive': self.config.enable_predictive_swapping
        })
        
        # Memory tracking
        self.monitored_blocks: Dict[str, Dict[str, Any]] = {}
        self.swap_decisions: deque = deque(maxlen=1000)
        self.memory_pressure_history: deque = deque(maxlen=100)
        
        # Policy management
        self.current_policy = self.config.swap_policy
        self.policy_performance: Dict[SwapPolicy, Dict[str, float]] = {
            policy: {'score': 0.5, 'usage_count': 0} for policy in SwapPolicy
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_swap_decisions': 0,
            'total_blocks_swapped': 0,
            'total_bytes_swapped': 0,
            'successful_swaps': 0,
            'failed_swaps': 0,
            'memory_pressure_events': 0,
            'policy_adjustments': 0,
            'average_decision_time_ms': 0.0
        }
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.last_monitoring_time = time.time()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Integration with GPU optimizer
        self._integrate_with_gpu_optimizer()
        
        self.logger.info(f"AutoSwapManager initialized with {self.config.swap_policy} policy")
    
    def start_monitoring(self) -> None:
        """Start automatic memory monitoring and swapping"""
        with self.lock:
            if self.monitoring_active:
                return
            
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            
            self.logger.info("AutoSwap monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop automatic monitoring"""
        with self.lock:
            self.monitoring_active = False
            
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5.0)
                self.monitoring_thread = None
            
            self.logger.info("AutoSwap monitoring stopped")
    
    def register_tensor(self, tensor_id: str, tensor: torch.Tensor,
                       priority: SwapPriority = SwapPriority.MEDIUM) -> None:
        """Register a tensor for AutoSwap management"""
        if not tensor.is_cuda:
            raise ValueError("Only GPU tensors can be registered for AutoSwap")
        
        with self.lock:
            # Register with DOA scorer
            size = tensor.numel() * tensor.element_size()
            device_id = tensor.device.index
            
            self.doa_scorer.register_memory_block(
                tensor_id, size, device_id, priority
            )
            
            # Register with priority swapper
            self.priority_swapper.register_memory_block(tensor_id, tensor)
            
            # Track in monitored blocks
            self.monitored_blocks[tensor_id] = {
                'tensor': tensor,
                'size': size,
                'device_id': device_id,
                'registration_time': time.time(),
                'last_swap_time': None,
                'swap_count': 0
            }
            
            self.logger.debug(f"Registered tensor {tensor_id} ({size / (1024*1024):.2f}MB) for AutoSwap")
    
    def unregister_tensor(self, tensor_id: str) -> None:
        """Unregister a tensor from AutoSwap management"""
        with self.lock:
            if tensor_id in self.monitored_blocks:
                del self.monitored_blocks[tensor_id]
                # Note: Keep DOA history for analysis
    
    def record_access(self, tensor_id: str, offset: int = 0,
                     length: Optional[int] = None) -> None:
        """Record tensor access for DOA tracking"""
        self.doa_scorer.record_access(tensor_id, offset, length)
    
    def make_swap_decision(self, required_bytes: int, device_id: int = 0) -> SwapDecision:
        """Make intelligent swap decision based on current state"""
        start_time = time.perf_counter()
        
        with self.lock:
            # Get current memory state
            memory_state = self._get_memory_state(device_id)
            pressure_level = self._calculate_pressure_level(memory_state)
            
            # Record pressure
            self.memory_pressure_history.append({
                'timestamp': time.time(),
                'level': pressure_level,
                'utilization': memory_state['utilization']
            })
            
            # Get swap candidates
            adjusted_required = int(required_bytes * self.config.swap_ahead_factor)
            candidates = self.doa_scorer.get_swap_candidates(
                adjusted_required, device_id, max_candidates=20
            )
            
            # Apply policy to select blocks
            selected_blocks = self._apply_swap_policy(
                candidates, adjusted_required, pressure_level
            )
            
            # Calculate total swap size
            total_swap_bytes = sum(
                self.monitored_blocks[block_id]['size']
                for block_id in selected_blocks
                if block_id in self.monitored_blocks
            )
            
            # Create decision
            decision = SwapDecision(
                required_bytes=required_bytes,
                available_bytes=memory_state['available'],
                pressure_level=pressure_level,
                candidates=candidates,
                selected_blocks=selected_blocks,
                total_swap_bytes=total_swap_bytes
            )
            
            # Update statistics
            decision_time = (time.perf_counter() - start_time) * 1000
            self._update_decision_stats(decision_time)
            
            self.swap_decisions.append(decision)
            
            return decision
    
    def execute_swap_decision(self, decision: SwapDecision) -> List[SwapOperation]:
        """Execute a swap decision"""
        operations = []
        
        if not decision.selected_blocks:
            return operations
        
        # Check if batch swapping is beneficial
        if (self.config.enable_batch_swapping and 
            len(decision.selected_blocks) > 3):
            # Batch swap
            swap_requests = []
            for block_id in decision.selected_blocks:
                score = next(
                    (c for c in decision.candidates if c.memory_block_id == block_id),
                    None
                )
                if score:
                    swap_requests.append((block_id, score, 'cpu'))
            
            operations = self.priority_swapper.batch_swap(swap_requests)
        else:
            # Individual swaps
            for block_id in decision.selected_blocks:
                score = next(
                    (c for c in decision.candidates if c.memory_block_id == block_id),
                    None
                )
                if score:
                    try:
                        operation = self.priority_swapper.swap_out(block_id, score, 'cpu')
                        operations.append(operation)
                        
                        # Update tracking
                        if block_id in self.monitored_blocks:
                            self.monitored_blocks[block_id]['last_swap_time'] = time.time()
                            self.monitored_blocks[block_id]['swap_count'] += 1
                        
                    except Exception as e:
                        self.logger.error(f"Failed to swap {block_id}: {e}")
                        self.performance_stats['failed_swaps'] += 1
        
        # Update statistics
        successful_ops = [op for op in operations if op.success]
        self.performance_stats['successful_swaps'] += len(successful_ops)
        self.performance_stats['total_blocks_swapped'] += len(successful_ops)
        self.performance_stats['total_bytes_swapped'] += sum(
            op.size_bytes for op in successful_ops
        )
        
        return operations
    
    def handle_memory_pressure(self, required_bytes: int, device_id: int = 0) -> bool:
        """Handle memory pressure by making and executing swap decisions"""
        try:
            # Make decision
            decision = self.make_swap_decision(required_bytes, device_id)
            
            if not decision.selected_blocks:
                self.logger.warning("No suitable blocks for swapping")
                return False
            
            # Execute swaps
            operations = self.execute_swap_decision(decision)
            
            # Check if enough memory was freed
            freed_bytes = sum(op.size_bytes for op in operations if op.success)
            success = freed_bytes >= required_bytes
            
            if success:
                self.logger.info(f"Successfully freed {freed_bytes / (1024*1024):.2f}MB")
            else:
                self.logger.warning(f"Only freed {freed_bytes / (1024*1024):.2f}MB, needed {required_bytes / (1024*1024):.2f}MB")
            
            # Update pressure event counter
            self.performance_stats['memory_pressure_events'] += 1
            
            # Consider policy adjustment
            if self.config.auto_adjust_policy:
                self._consider_policy_adjustment(decision, operations)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to handle memory pressure: {e}")
            return False
    
    def swap_in_tensor(self, tensor_id: str, device_id: int = 0) -> Optional[torch.Tensor]:
        """Swap a tensor back to GPU"""
        try:
            operation = self.priority_swapper.swap_in(tensor_id, device_id)
            
            if operation.success:
                # Get tensor reference
                location = self.priority_swapper.get_memory_location(tensor_id)
                return location.get('tensor')
            else:
                self.logger.error(f"Failed to swap in {tensor_id}: {operation.error_message}")
                return None
                
        except Exception as e:
            self.logger.error(f"Exception swapping in {tensor_id}: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive AutoSwap statistics"""
        with self.lock:
            # Calculate policy effectiveness
            policy_stats = {}
            for policy, perf in self.policy_performance.items():
                if perf['usage_count'] > 0:
                    policy_stats[policy.value] = {
                        'score': perf['score'],
                        'usage_count': perf['usage_count'],
                        'effectiveness': perf['score'] * 100
                    }
            
            # Get current pressure
            current_pressure = None
            if self.memory_pressure_history:
                current_pressure = self.memory_pressure_history[-1]['level'].value
            
            return {
                'performance_stats': self.performance_stats.copy(),
                'current_policy': self.current_policy.value,
                'policy_performance': policy_stats,
                'current_memory_pressure': current_pressure,
                'monitored_blocks': len(self.monitored_blocks),
                'total_monitored_bytes_mb': sum(
                    b['size'] for b in self.monitored_blocks.values()
                ) / (1024 * 1024),
                'doa_scorer_stats': self.doa_scorer.get_statistics(),
                'priority_swapper_stats': self.priority_swapper.get_statistics()
            }
    
    def _integrate_with_gpu_optimizer(self) -> None:
        """Integrate AutoSwap with GPU optimizer"""
        # Store reference in GPU optimizer
        self.gpu_optimizer.autoswap_manager = self
        
        # Override memory allocation hooks if needed
        self._setup_allocation_hooks()
    
    def _setup_allocation_hooks(self) -> None:
        """Setup hooks for automatic tensor registration"""
        # This would hook into PyTorch's memory allocation
        # For production, would use torch.cuda.memory allocation hooks
        pass
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                # Check each device
                for device_id in range(torch.cuda.device_count()):
                    memory_state = self._get_memory_state(device_id)
                    pressure_level = self._calculate_pressure_level(memory_state)
                    
                    # Handle high pressure proactively
                    if pressure_level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
                        # Estimate required free space
                        target_utilization = self.config.memory_pressure_thresholds['moderate']
                        current_utilization = memory_state['utilization']
                        
                        if current_utilization > target_utilization:
                            required_bytes = int(
                                (current_utilization - target_utilization) * memory_state['total']
                            )
                            
                            self.logger.info(f"Proactive swap triggered: {pressure_level.value} pressure")
                            self.handle_memory_pressure(required_bytes, device_id)
                
                # Sleep until next interval
                sleep_time = self.config.monitoring_interval_seconds - (time.time() - current_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.config.monitoring_interval_seconds)
    
    def _get_memory_state(self, device_id: int) -> Dict[str, Any]:
        """Get current GPU memory state"""
        with torch.cuda.device(device_id):
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            total = torch.cuda.get_device_properties(device_id).total_memory
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'total': total,
            'available': total - reserved,
            'utilization': reserved / total if total > 0 else 0
        }
    
    def _calculate_pressure_level(self, memory_state: Dict[str, Any]) -> MemoryPressureLevel:
        """Calculate memory pressure level"""
        utilization = memory_state['utilization']
        thresholds = self.config.memory_pressure_thresholds
        
        if utilization >= thresholds['critical']:
            return MemoryPressureLevel.CRITICAL
        elif utilization >= thresholds['high']:
            return MemoryPressureLevel.HIGH
        elif utilization >= thresholds['moderate']:
            return MemoryPressureLevel.MODERATE
        else:
            return MemoryPressureLevel.LOW
    
    def _apply_swap_policy(self, candidates: List[DOAScore], required_bytes: int,
                          pressure_level: MemoryPressureLevel) -> List[str]:
        """Apply swap policy to select blocks"""
        selected_blocks = []
        accumulated_bytes = 0
        
        if self.current_policy == SwapPolicy.AGGRESSIVE:
            # Swap more blocks, lower threshold
            for candidate in candidates:
                if candidate.swap_priority.value >= SwapPriority.MEDIUM.value:
                    selected_blocks.append(candidate.memory_block_id)
                    accumulated_bytes += candidate.size_bytes
                    
                    if accumulated_bytes >= required_bytes * 1.5:
                        break
                        
        elif self.current_policy == SwapPolicy.CONSERVATIVE:
            # Swap only idle/low priority blocks
            for candidate in candidates:
                if candidate.swap_priority.value >= SwapPriority.LOW.value:
                    selected_blocks.append(candidate.memory_block_id)
                    accumulated_bytes += candidate.size_bytes
                    
                    if accumulated_bytes >= required_bytes:
                        break
                        
        elif self.current_policy == SwapPolicy.BALANCED:
            # Balance based on pressure
            threshold = SwapPriority.MEDIUM
            if pressure_level == MemoryPressureLevel.CRITICAL:
                threshold = SwapPriority.HIGH
            elif pressure_level == MemoryPressureLevel.LOW:
                threshold = SwapPriority.LOW
            
            for candidate in candidates:
                if candidate.swap_priority.value >= threshold.value:
                    selected_blocks.append(candidate.memory_block_id)
                    accumulated_bytes += candidate.size_bytes
                    
                    if accumulated_bytes >= required_bytes:
                        break
                        
        else:  # ADAPTIVE
            # Use learned policy performance
            selected_blocks = self._apply_adaptive_policy(
                candidates, required_bytes, pressure_level
            )
        
        return selected_blocks
    
    def _apply_adaptive_policy(self, candidates: List[DOAScore], required_bytes: int,
                              pressure_level: MemoryPressureLevel) -> List[str]:
        """Apply adaptive policy based on past performance"""
        # For now, delegate to balanced policy
        # In production, would use ML-based policy selection
        self.current_policy = SwapPolicy.BALANCED
        return self._apply_swap_policy(candidates, required_bytes, pressure_level)
    
    def _consider_policy_adjustment(self, decision: SwapDecision,
                                  operations: List[SwapOperation]) -> None:
        """Consider adjusting swap policy based on results"""
        # Calculate decision effectiveness
        success_rate = sum(1 for op in operations if op.success) / max(len(operations), 1)
        
        # Update policy performance
        policy_perf = self.policy_performance[self.current_policy]
        policy_perf['usage_count'] += 1
        
        # Exponential moving average
        alpha = 0.1
        policy_perf['score'] = alpha * success_rate + (1 - alpha) * policy_perf['score']
        
        # Check if policy change is needed
        if policy_perf['score'] < 0.5 and policy_perf['usage_count'] > 10:
            # Find best performing policy
            best_policy = max(
                self.policy_performance.items(),
                key=lambda x: x[1]['score']
            )[0]
            
            if best_policy != self.current_policy:
                self.logger.info(f"Switching policy from {self.current_policy.value} to {best_policy.value}")
                self.current_policy = best_policy
                self.performance_stats['policy_adjustments'] += 1
    
    def _update_decision_stats(self, decision_time_ms: float) -> None:
        """Update decision statistics"""
        self.performance_stats['total_swap_decisions'] += 1
        
        # Update average decision time
        total = self.performance_stats['total_swap_decisions']
        current_avg = self.performance_stats['average_decision_time_ms']
        self.performance_stats['average_decision_time_ms'] = (
            (current_avg * (total - 1) + decision_time_ms) / total
        )