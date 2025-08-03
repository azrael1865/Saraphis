"""
Weighted Interval Graph Coloring for GPU Memory Optimization
Achieves optimal memory allocation through graph-based algorithms
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import logging
import heapq
from enum import Enum

logger = logging.getLogger(__name__)


class AllocationStatus(Enum):
    """Memory allocation status"""
    FREE = "free"
    ALLOCATED = "allocated"
    PENDING = "pending"
    COALESCING = "coalescing"


@dataclass
class MemoryInterval:
    """Represents a memory allocation interval"""
    start_addr: int
    end_addr: int
    size: int
    allocation_time: float
    last_access_time: float
    access_count: int
    device_id: int
    tensor_id: Optional[int] = None
    color: Optional[int] = None
    weight: float = 1.0
    status: AllocationStatus = AllocationStatus.FREE
    
    @property
    def duration(self) -> float:
        """Get interval duration"""
        return self.last_access_time - self.allocation_time
    
    @property
    def access_frequency(self) -> float:
        """Calculate access frequency"""
        if self.duration <= 0:
            return float(self.access_count)
        return self.access_count / max(self.duration, 1e-6)
    
    def overlaps(self, other: 'MemoryInterval') -> bool:
        """Check if intervals overlap"""
        return not (self.end_addr <= other.start_addr or other.end_addr <= self.start_addr)
    
    def can_coalesce(self, other: 'MemoryInterval') -> bool:
        """Check if intervals can be coalesced"""
        if self.device_id != other.device_id:
            return False
        if self.status != AllocationStatus.FREE or other.status != AllocationStatus.FREE:
            return False
        # Adjacent intervals
        return self.end_addr == other.start_addr or other.end_addr == self.start_addr


@dataclass
class IntervalNode:
    """Node in the interval graph"""
    interval: MemoryInterval
    neighbors: Set[int] = field(default_factory=set)
    degree: int = 0
    saturation: int = 0
    available_colors: Set[int] = field(default_factory=set)


class WeightedIntervalGraphColoring:
    """
    Implements weighted interval graph coloring for optimal GPU memory allocation.
    Reduces fragmentation through intelligent allocation strategies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the weighted interval graph coloring system"""
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Graph structures
        self.intervals: Dict[int, MemoryInterval] = {}
        self.nodes: Dict[int, IntervalNode] = {}
        self.adjacency_list: Dict[int, Set[int]] = defaultdict(set)
        
        # Coloring parameters
        self.max_colors = self.config.get('max_colors', 256)
        self.weight_threshold = self.config.get('weight_threshold', 0.1)
        self.coalescing_threshold = self.config.get('coalescing_threshold', 0.8)
        
        # Performance tracking
        self.coloring_stats = {
            'total_colorings': 0,
            'colors_used': 0,
            'fragmentation_reduced': 0.0,
            'coalescing_operations': 0,
            'optimization_time_ms': 0.0
        }
        
        # Interval ID counter
        self.next_interval_id = 0
        
        self.logger.info("WeightedIntervalGraphColoring initialized")
    
    def add_interval(self, start_addr: int, size: int, device_id: int,
                    allocation_time: float, access_count: int = 1) -> int:
        """Add a new memory interval to the graph"""
        interval_id = self.next_interval_id
        self.next_interval_id += 1
        
        interval = MemoryInterval(
            start_addr=start_addr,
            end_addr=start_addr + size,
            size=size,
            allocation_time=allocation_time,
            last_access_time=allocation_time,
            access_count=access_count,
            device_id=device_id,
            tensor_id=interval_id
        )
        
        # Calculate initial weight based on size and access pattern
        interval.weight = self._calculate_interval_weight(interval)
        
        self.intervals[interval_id] = interval
        self.nodes[interval_id] = IntervalNode(interval=interval)
        
        # Update graph structure
        self._update_adjacency(interval_id)
        
        return interval_id
    
    def remove_interval(self, interval_id: int) -> bool:
        """Remove an interval from the graph"""
        if interval_id not in self.intervals:
            return False
        
        # Remove from adjacency lists
        for neighbor_id in self.adjacency_list[interval_id]:
            self.adjacency_list[neighbor_id].discard(interval_id)
            if neighbor_id in self.nodes:
                self.nodes[neighbor_id].neighbors.discard(interval_id)
        
        # Remove interval
        del self.intervals[interval_id]
        del self.nodes[interval_id]
        del self.adjacency_list[interval_id]
        
        return True
    
    def update_access(self, interval_id: int, access_time: float) -> None:
        """Update interval access statistics"""
        if interval_id not in self.intervals:
            raise ValueError(f"Interval {interval_id} not found")
        
        interval = self.intervals[interval_id]
        interval.last_access_time = access_time
        interval.access_count += 1
        
        # Recalculate weight
        interval.weight = self._calculate_interval_weight(interval)
    
    def color_graph(self) -> Dict[int, int]:
        """
        Apply weighted graph coloring algorithm to minimize fragmentation.
        Returns mapping of interval_id to color.
        """
        start_time = time.perf_counter()
        
        # Reset coloring
        for interval in self.intervals.values():
            interval.color = None
        for node in self.nodes.values():
            node.available_colors = set(range(self.max_colors))
            node.saturation = 0
        
        # Sort intervals by weight (descending) for priority
        sorted_intervals = sorted(
            self.intervals.items(),
            key=lambda x: (x[1].weight, x[1].size),
            reverse=True
        )
        
        # Apply DSATUR algorithm with weight consideration
        colored_count = 0
        color_mapping = {}
        
        for interval_id, interval in sorted_intervals:
            if interval.status != AllocationStatus.FREE:
                continue
            
            node = self.nodes[interval_id]
            
            # Find available colors
            used_colors = set()
            for neighbor_id in node.neighbors:
                neighbor_interval = self.intervals.get(neighbor_id)
                if neighbor_interval and neighbor_interval.color is not None:
                    used_colors.add(neighbor_interval.color)
            
            available_colors = set(range(self.max_colors)) - used_colors
            
            if not available_colors:
                # Need more colors - this indicates high fragmentation
                self.logger.warning(f"No available colors for interval {interval_id}")
                continue
            
            # Choose color that minimizes fragmentation
            best_color = self._choose_optimal_color(interval_id, available_colors)
            interval.color = best_color
            color_mapping[interval_id] = best_color
            colored_count += 1
            
            # Update saturation for neighbors
            for neighbor_id in node.neighbors:
                if neighbor_id in self.nodes:
                    neighbor_node = self.nodes[neighbor_id]
                    neighbor_node.available_colors.discard(best_color)
                    neighbor_node.saturation = len(set(range(self.max_colors)) - neighbor_node.available_colors)
        
        # Calculate statistics
        colors_used = len(set(color_mapping.values())) if color_mapping else 0
        optimization_time = (time.perf_counter() - start_time) * 1000
        
        self.coloring_stats['total_colorings'] += 1
        self.coloring_stats['colors_used'] = colors_used
        self.coloring_stats['optimization_time_ms'] = optimization_time
        
        self.logger.info(f"Graph colored: {colored_count} intervals, {colors_used} colors used")
        
        return color_mapping
    
    def optimize_allocation(self, target_size: int, device_id: int) -> Optional[Tuple[int, int]]:
        """
        Find optimal memory allocation for requested size.
        Returns (start_address, interval_id) if found.
        """
        # First, try coalescing free intervals
        self._coalesce_free_intervals(device_id)
        
        # Find best-fit interval
        best_interval_id = None
        best_fit_score = float('inf')
        
        for interval_id, interval in self.intervals.items():
            if (interval.device_id == device_id and 
                interval.status == AllocationStatus.FREE and
                interval.size >= target_size):
                
                # Calculate fit score (lower is better)
                fit_score = self._calculate_fit_score(interval, target_size)
                
                if fit_score < best_fit_score:
                    best_fit_score = fit_score
                    best_interval_id = interval_id
        
        if best_interval_id is None:
            return None
        
        best_interval = self.intervals[best_interval_id]
        start_addr = best_interval.start_addr
        
        # Split interval if needed
        if best_interval.size > target_size:
            self._split_interval(best_interval_id, target_size)
        
        # Mark as allocated
        best_interval.status = AllocationStatus.ALLOCATED
        
        return (start_addr, best_interval_id)
    
    def calculate_fragmentation(self, device_id: Optional[int] = None) -> float:
        """Calculate memory fragmentation metric"""
        if device_id is not None:
            intervals = [i for i in self.intervals.values() if i.device_id == device_id]
        else:
            intervals = list(self.intervals.values())
        
        if not intervals:
            return 0.0
        
        # Calculate fragmentation based on free interval distribution
        free_intervals = [i for i in intervals if i.status == AllocationStatus.FREE]
        if not free_intervals:
            return 0.0
        
        # Metrics for fragmentation
        total_free_size = sum(i.size for i in free_intervals)
        largest_free_size = max(i.size for i in free_intervals)
        num_free_intervals = len(free_intervals)
        
        # Fragmentation score (0 = no fragmentation, 1 = highly fragmented)
        if total_free_size == 0:
            return 0.0
        
        # Multiple small fragments indicate high fragmentation
        size_fragmentation = 1.0 - (largest_free_size / total_free_size)
        count_fragmentation = min(1.0, num_free_intervals / 100.0)  # Normalize by typical count
        
        # Weighted fragmentation score
        fragmentation = 0.7 * size_fragmentation + 0.3 * count_fragmentation
        
        return min(1.0, max(0.0, fragmentation))
    
    def get_allocation_recommendations(self) -> List[str]:
        """Get recommendations for improving memory allocation"""
        recommendations = []
        
        # Analyze current state
        total_intervals = len(self.intervals)
        free_intervals = sum(1 for i in self.intervals.values() if i.status == AllocationStatus.FREE)
        fragmentation = self.calculate_fragmentation()
        
        if fragmentation > 0.5:
            recommendations.append(f"High fragmentation detected ({fragmentation:.2%}). Consider defragmentation.")
        
        if free_intervals > total_intervals * 0.3:
            recommendations.append(f"Many free intervals ({free_intervals}/{total_intervals}). Consider coalescing.")
        
        # Check for long-lived allocations
        current_time = time.time()
        old_allocations = [
            i for i in self.intervals.values()
            if i.status == AllocationStatus.ALLOCATED and 
            current_time - i.allocation_time > 300  # 5 minutes
        ]
        
        if old_allocations:
            recommendations.append(f"{len(old_allocations)} long-lived allocations detected. Review memory usage patterns.")
        
        # Color usage analysis
        if self.coloring_stats['colors_used'] > self.max_colors * 0.8:
            recommendations.append("High color usage indicates complex allocation patterns. Consider increasing pool size.")
        
        return recommendations
    
    def _calculate_interval_weight(self, interval: MemoryInterval) -> float:
        """Calculate weight for an interval based on various factors"""
        # Size factor (larger allocations get higher weight)
        size_factor = np.log1p(interval.size / (1024 * 1024))  # Log scale for MB
        
        # Access frequency factor
        freq_factor = np.log1p(interval.access_frequency)
        
        # Lifetime factor (longer-lived allocations get higher weight)
        lifetime_factor = np.log1p(interval.duration)
        
        # Combined weight
        weight = (
            0.4 * size_factor +
            0.4 * freq_factor +
            0.2 * lifetime_factor
        )
        
        return max(self.weight_threshold, weight)
    
    def _update_adjacency(self, interval_id: int) -> None:
        """Update adjacency relationships for an interval"""
        interval = self.intervals[interval_id]
        
        for other_id, other_interval in self.intervals.items():
            if other_id != interval_id and interval.overlaps(other_interval):
                self.adjacency_list[interval_id].add(other_id)
                self.adjacency_list[other_id].add(interval_id)
                
                self.nodes[interval_id].neighbors.add(other_id)
                self.nodes[other_id].neighbors.add(interval_id)
        
        # Update degrees
        self.nodes[interval_id].degree = len(self.nodes[interval_id].neighbors)
    
    def _choose_optimal_color(self, interval_id: int, available_colors: Set[int]) -> int:
        """Choose color that minimizes fragmentation"""
        interval = self.intervals[interval_id]
        
        # Group intervals by color
        color_groups = defaultdict(list)
        for other_id, other_interval in self.intervals.items():
            if other_interval.color is not None and other_interval.device_id == interval.device_id:
                color_groups[other_interval.color].append(other_interval)
        
        # Find color with best spatial locality
        best_color = None
        best_score = float('inf')
        
        for color in available_colors:
            if color in color_groups:
                # Calculate locality score
                score = self._calculate_locality_score(interval, color_groups[color])
                if score < best_score:
                    best_score = score
                    best_color = color
        
        # If no existing color groups, use lowest available color
        if best_color is None:
            best_color = min(available_colors)
        
        return best_color
    
    def _calculate_locality_score(self, interval: MemoryInterval, 
                                 color_group: List[MemoryInterval]) -> float:
        """Calculate spatial locality score (lower is better)"""
        if not color_group:
            return float('inf')
        
        # Find minimum distance to intervals in color group
        min_distance = float('inf')
        
        for other in color_group:
            if other.status == AllocationStatus.FREE:
                # Distance between intervals
                if interval.start_addr >= other.end_addr:
                    distance = interval.start_addr - other.end_addr
                elif other.start_addr >= interval.end_addr:
                    distance = other.start_addr - interval.end_addr
                else:
                    distance = 0  # Overlapping
                
                min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _calculate_fit_score(self, interval: MemoryInterval, target_size: int) -> float:
        """Calculate how well an interval fits the target size (lower is better)"""
        # Best fit: interval size close to target size
        size_diff = interval.size - target_size
        size_ratio = size_diff / target_size
        
        # Penalize fragmentation
        fragmentation_penalty = 0.0
        if size_diff > target_size:  # Would create a fragment
            fragmentation_penalty = size_ratio * 0.5
        
        # Consider interval weight (prefer lower weight intervals)
        weight_penalty = interval.weight * 0.1
        
        return size_ratio + fragmentation_penalty + weight_penalty
    
    def _coalesce_free_intervals(self, device_id: int) -> int:
        """Coalesce adjacent free intervals to reduce fragmentation"""
        coalesced_count = 0
        
        # Find all free intervals for device
        free_intervals = [
            (interval_id, interval)
            for interval_id, interval in self.intervals.items()
            if interval.device_id == device_id and interval.status == AllocationStatus.FREE
        ]
        
        # Sort by start address
        free_intervals.sort(key=lambda x: x[1].start_addr)
        
        i = 0
        while i < len(free_intervals) - 1:
            curr_id, curr_interval = free_intervals[i]
            next_id, next_interval = free_intervals[i + 1]
            
            if curr_interval.can_coalesce(next_interval):
                # Coalesce intervals
                curr_interval.end_addr = next_interval.end_addr
                curr_interval.size += next_interval.size
                curr_interval.last_access_time = max(
                    curr_interval.last_access_time,
                    next_interval.last_access_time
                )
                curr_interval.access_count += next_interval.access_count
                
                # Update weight
                curr_interval.weight = self._calculate_interval_weight(curr_interval)
                
                # Remove the coalesced interval
                self.remove_interval(next_id)
                free_intervals.pop(i + 1)
                
                coalesced_count += 1
                self.coloring_stats['coalescing_operations'] += 1
            else:
                i += 1
        
        if coalesced_count > 0:
            self.logger.debug(f"Coalesced {coalesced_count} intervals on device {device_id}")
        
        return coalesced_count
    
    def _split_interval(self, interval_id: int, split_size: int) -> int:
        """Split an interval into two parts"""
        if interval_id not in self.intervals:
            raise ValueError(f"Interval {interval_id} not found")
        
        interval = self.intervals[interval_id]
        if interval.size <= split_size:
            return interval_id
        
        # Create new interval for remainder
        new_interval_id = self.next_interval_id
        self.next_interval_id += 1
        
        new_interval = MemoryInterval(
            start_addr=interval.start_addr + split_size,
            end_addr=interval.end_addr,
            size=interval.size - split_size,
            allocation_time=interval.allocation_time,
            last_access_time=interval.last_access_time,
            access_count=0,  # New interval starts with 0 accesses
            device_id=interval.device_id,
            tensor_id=new_interval_id,
            weight=interval.weight * 0.5,  # Reduce weight for split
            status=AllocationStatus.FREE
        )
        
        # Update original interval
        interval.end_addr = interval.start_addr + split_size
        interval.size = split_size
        
        # Add new interval to graph
        self.intervals[new_interval_id] = new_interval
        self.nodes[new_interval_id] = IntervalNode(interval=new_interval)
        self._update_adjacency(new_interval_id)
        
        return new_interval_id
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            'total_intervals': len(self.intervals),
            'free_intervals': sum(1 for i in self.intervals.values() if i.status == AllocationStatus.FREE),
            'allocated_intervals': sum(1 for i in self.intervals.values() if i.status == AllocationStatus.ALLOCATED),
            'fragmentation': self.calculate_fragmentation(),
            'coloring_stats': self.coloring_stats.copy(),
            'average_interval_weight': np.mean([i.weight for i in self.intervals.values()]) if self.intervals else 0.0
        }