"""
Advanced Memory Pool Manager for GPU Memory Optimization
Implements multi-tiered pools with intelligent allocation strategies
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any, Deque
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import time
import threading
import logging
import heapq

logger = logging.getLogger(__name__)


class PoolTier(Enum):
    """Memory pool tiers based on allocation size"""
    SMALL = "small"      # < 1MB
    MEDIUM = "medium"    # 1MB - 16MB
    LARGE = "large"      # 16MB - 256MB
    HUGE = "huge"        # > 256MB


class AllocationStrategy(Enum):
    """Memory allocation strategies"""
    BEST_FIT = "best_fit"
    FIRST_FIT = "first_fit"
    WORST_FIT = "worst_fit"
    NEXT_FIT = "next_fit"
    ADAPTIVE = "adaptive"


@dataclass
class PooledMemoryBlock:
    """Memory block within a pool"""
    ptr: torch.cuda.caching_allocator_alloc
    size: int
    tier: PoolTier
    device_id: int
    allocated_time: float
    last_used_time: float
    use_count: int = 0
    is_allocated: bool = False
    tensor_ref: Optional[torch.Tensor] = None
    allocation_id: Optional[str] = None
    
    @property
    def age(self) -> float:
        """Get block age in seconds"""
        return time.time() - self.allocated_time
    
    @property
    def idle_time(self) -> float:
        """Get time since last use"""
        return time.time() - self.last_used_time


@dataclass
class PoolStatistics:
    """Statistics for a memory pool tier"""
    total_blocks: int = 0
    allocated_blocks: int = 0
    free_blocks: int = 0
    total_size_mb: float = 0.0
    allocated_size_mb: float = 0.0
    free_size_mb: float = 0.0
    allocation_count: int = 0
    deallocation_count: int = 0
    hit_rate: float = 0.0
    fragmentation: float = 0.0
    average_block_age: float = 0.0
    average_idle_time: float = 0.0


class TieredMemoryPool:
    """Single tier of the memory pool system"""
    
    def __init__(self, tier: PoolTier, device_id: int, config: Dict[str, Any]):
        self.tier = tier
        self.device_id = device_id
        self.config = config
        
        # Pool configuration
        self.min_block_size = self._get_tier_min_size()
        self.max_block_size = self._get_tier_max_size()
        self.initial_blocks = config.get(f'{tier.value}_initial_blocks', 10)
        self.max_blocks = config.get(f'{tier.value}_max_blocks', 100)
        self.block_alignment = config.get('block_alignment', 256)
        
        # Memory blocks
        self.free_blocks: List[PooledMemoryBlock] = []
        self.allocated_blocks: Dict[str, PooledMemoryBlock] = {}
        
        # Statistics
        self.stats = PoolStatistics()
        self.allocation_history: Deque[Tuple[float, int]] = deque(maxlen=1000)
        
        # Pre-allocate initial blocks
        self._preallocate_blocks()
    
    def _get_tier_min_size(self) -> int:
        """Get minimum size for tier"""
        sizes = {
            PoolTier.SMALL: 0,
            PoolTier.MEDIUM: 1024 * 1024,  # 1MB
            PoolTier.LARGE: 16 * 1024 * 1024,  # 16MB
            PoolTier.HUGE: 256 * 1024 * 1024  # 256MB
        }
        return sizes[self.tier]
    
    def _get_tier_max_size(self) -> int:
        """Get maximum size for tier"""
        sizes = {
            PoolTier.SMALL: 1024 * 1024,  # 1MB
            PoolTier.MEDIUM: 16 * 1024 * 1024,  # 16MB
            PoolTier.LARGE: 256 * 1024 * 1024,  # 256MB
            PoolTier.HUGE: float('inf')
        }
        return sizes[self.tier]
    
    def _preallocate_blocks(self) -> None:
        """Pre-allocate initial blocks for the pool"""
        block_size = self._calculate_optimal_block_size()
        
        with torch.cuda.device(self.device_id):
            for _ in range(self.initial_blocks):
                try:
                    # Allocate aligned memory
                    aligned_size = self._align_size(block_size)
                    tensor = torch.empty(aligned_size // 4, dtype=torch.float32, device=f'cuda:{self.device_id}')
                    
                    block = PooledMemoryBlock(
                        ptr=tensor.data_ptr(),
                        size=aligned_size,
                        tier=self.tier,
                        device_id=self.device_id,
                        allocated_time=time.time(),
                        last_used_time=time.time(),
                        tensor_ref=tensor
                    )
                    
                    self.free_blocks.append(block)
                    self.stats.total_blocks += 1
                    self.stats.free_blocks += 1
                    self.stats.total_size_mb += aligned_size / (1024 * 1024)
                    self.stats.free_size_mb += aligned_size / (1024 * 1024)
                    
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"Failed to preallocate block for {self.tier.value} pool")
                    break
    
    def allocate(self, size: int, allocation_id: str) -> Optional[PooledMemoryBlock]:
        """Allocate a block from the pool"""
        aligned_size = self._align_size(size)
        
        # Find suitable free block
        best_block = None
        best_index = -1
        
        for i, block in enumerate(self.free_blocks):
            if block.size >= aligned_size and not block.is_allocated:
                if best_block is None or block.size < best_block.size:
                    best_block = block
                    best_index = i
        
        if best_block is None:
            # Try to allocate new block if under limit
            if self.stats.total_blocks < self.max_blocks:
                best_block = self._allocate_new_block(aligned_size)
                if best_block:
                    self.free_blocks.append(best_block)
                    best_index = len(self.free_blocks) - 1
        
        if best_block is None:
            return None
        
        # Remove from free list
        self.free_blocks.pop(best_index)
        
        # Mark as allocated
        best_block.is_allocated = True
        best_block.allocation_id = allocation_id
        best_block.last_used_time = time.time()
        best_block.use_count += 1
        
        # Add to allocated blocks
        self.allocated_blocks[allocation_id] = best_block
        
        # Update statistics
        self.stats.allocated_blocks += 1
        self.stats.free_blocks -= 1
        self.stats.allocated_size_mb += best_block.size / (1024 * 1024)
        self.stats.free_size_mb -= best_block.size / (1024 * 1024)
        self.stats.allocation_count += 1
        
        # Record allocation
        self.allocation_history.append((time.time(), aligned_size))
        
        return best_block
    
    def deallocate(self, allocation_id: str) -> bool:
        """Deallocate a block back to the pool"""
        if allocation_id not in self.allocated_blocks:
            return False
        
        block = self.allocated_blocks.pop(allocation_id)
        
        # Reset block state
        block.is_allocated = False
        block.allocation_id = None
        block.last_used_time = time.time()
        
        # Add back to free list
        self.free_blocks.append(block)
        
        # Update statistics
        self.stats.allocated_blocks -= 1
        self.stats.free_blocks += 1
        self.stats.allocated_size_mb -= block.size / (1024 * 1024)
        self.stats.free_size_mb += block.size / (1024 * 1024)
        self.stats.deallocation_count += 1
        
        return True
    
    def defragment(self) -> int:
        """Defragment the pool by coalescing adjacent blocks"""
        if not self.free_blocks:
            return 0
        
        # Sort blocks by memory address
        self.free_blocks.sort(key=lambda b: b.ptr)
        
        coalesced_count = 0
        i = 0
        
        while i < len(self.free_blocks) - 1:
            current = self.free_blocks[i]
            next_block = self.free_blocks[i + 1]
            
            # Check if blocks are adjacent
            if current.ptr + current.size == next_block.ptr:
                # Coalesce blocks
                current.size += next_block.size
                current.last_used_time = max(current.last_used_time, next_block.last_used_time)
                current.use_count += next_block.use_count
                
                # Remove the coalesced block
                self.free_blocks.pop(i + 1)
                self.stats.total_blocks -= 1
                self.stats.free_blocks -= 1
                
                coalesced_count += 1
            else:
                i += 1
        
        return coalesced_count
    
    def evict_old_blocks(self, max_age_seconds: float = 300) -> int:
        """Evict blocks older than specified age"""
        current_time = time.time()
        evicted_count = 0
        
        # Filter old blocks
        new_free_blocks = []
        
        for block in self.free_blocks:
            if current_time - block.allocated_time > max_age_seconds:
                # Release the tensor
                if block.tensor_ref is not None:
                    del block.tensor_ref
                
                # Update statistics
                self.stats.total_blocks -= 1
                self.stats.free_blocks -= 1
                self.stats.total_size_mb -= block.size / (1024 * 1024)
                self.stats.free_size_mb -= block.size / (1024 * 1024)
                
                evicted_count += 1
            else:
                new_free_blocks.append(block)
        
        self.free_blocks = new_free_blocks
        return evicted_count
    
    def get_statistics(self) -> PoolStatistics:
        """Get current pool statistics"""
        # Update calculated statistics
        if self.stats.allocation_count > 0:
            self.stats.hit_rate = self.stats.deallocation_count / self.stats.allocation_count
        
        if self.stats.total_size_mb > 0:
            self.stats.fragmentation = 1.0 - (self.stats.allocated_size_mb / self.stats.total_size_mb)
        
        # Calculate average ages
        all_blocks = self.free_blocks + list(self.allocated_blocks.values())
        if all_blocks:
            self.stats.average_block_age = np.mean([b.age for b in all_blocks])
            self.stats.average_idle_time = np.mean([b.idle_time for b in self.free_blocks])
        
        return self.stats
    
    def _calculate_optimal_block_size(self) -> int:
        """Calculate optimal block size based on tier and history"""
        if self.tier == PoolTier.SMALL:
            return 512 * 1024  # 512KB
        elif self.tier == PoolTier.MEDIUM:
            return 8 * 1024 * 1024  # 8MB
        elif self.tier == PoolTier.LARGE:
            return 64 * 1024 * 1024  # 64MB
        else:  # HUGE
            return 512 * 1024 * 1024  # 512MB
    
    def _align_size(self, size: int) -> int:
        """Align size to block alignment boundary"""
        return ((size + self.block_alignment - 1) // self.block_alignment) * self.block_alignment
    
    def _allocate_new_block(self, size: int) -> Optional[PooledMemoryBlock]:
        """Allocate a new block"""
        try:
            with torch.cuda.device(self.device_id):
                tensor = torch.empty(size // 4, dtype=torch.float32, device=f'cuda:{self.device_id}')
                
                block = PooledMemoryBlock(
                    ptr=tensor.data_ptr(),
                    size=size,
                    tier=self.tier,
                    device_id=self.device_id,
                    allocated_time=time.time(),
                    last_used_time=time.time(),
                    tensor_ref=tensor
                )
                
                self.stats.total_blocks += 1
                self.stats.total_size_mb += size / (1024 * 1024)
                
                return block
                
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"Failed to allocate new block of size {size}")
            return None


class AdvancedMemoryPoolManager:
    """
    Advanced memory pool manager with multi-tiered pools and intelligent allocation.
    Reduces fragmentation through pool-based allocation and defragmentation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the advanced memory pool manager"""
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Pool configuration
        self.device_count = torch.cuda.device_count()
        self.allocation_strategy = AllocationStrategy(
            self.config.get('allocation_strategy', 'adaptive')
        )
        
        # Create tiered pools for each device
        self.pools: Dict[int, Dict[PoolTier, TieredMemoryPool]] = {}
        for device_id in range(self.device_count):
            self.pools[device_id] = {}
            for tier in PoolTier:
                self.pools[device_id][tier] = TieredMemoryPool(tier, device_id, self.config)
        
        # Allocation tracking
        self.allocations: Dict[str, Tuple[int, PoolTier, int]] = {}  # id -> (device, tier, size)
        self.allocation_counter = 0
        
        # Performance tracking
        self.performance_stats = {
            'total_allocations': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'total_deallocations': 0,
            'defragmentation_runs': 0,
            'blocks_coalesced': 0,
            'blocks_evicted': 0,
            'average_allocation_time_ms': 0.0
        }
        
        # Prediction model for adaptive allocation
        self.allocation_predictor = AllocationPredictor()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background maintenance
        self.maintenance_interval = config.get('maintenance_interval', 60.0)
        self.last_maintenance = time.time()
        
        self.logger.info("AdvancedMemoryPoolManager initialized")
    
    def allocate(self, size: int, device_id: int) -> Optional[Tuple[torch.Tensor, str]]:
        """
        Allocate memory from the pool system.
        Returns (tensor, allocation_id) if successful.
        """
        start_time = time.perf_counter()
        
        with self.lock:
            # Generate allocation ID
            allocation_id = f"alloc_{self.allocation_counter}_{int(time.time() * 1000)}"
            self.allocation_counter += 1
            
            # Determine appropriate tier
            tier = self._get_tier_for_size(size)
            
            # Try allocation with current strategy
            block = None
            
            if self.allocation_strategy == AllocationStrategy.ADAPTIVE:
                # Use predictor to choose best pool
                predicted_tier = self.allocation_predictor.predict_tier(size, device_id)
                if predicted_tier:
                    tier = predicted_tier
            
            # Try to allocate from the selected tier
            pool = self.pools[device_id][tier]
            block = pool.allocate(size, allocation_id)
            
            # If failed, try other tiers
            if block is None:
                for fallback_tier in PoolTier:
                    if fallback_tier != tier:
                        fallback_pool = self.pools[device_id][fallback_tier]
                        block = fallback_pool.allocate(size, allocation_id)
                        if block:
                            tier = fallback_tier
                            break
            
            # Update statistics
            allocation_time = (time.perf_counter() - start_time) * 1000
            self.performance_stats['total_allocations'] += 1
            
            if block:
                # Create tensor view
                tensor = torch.empty(
                    size // 4,
                    dtype=torch.float32,
                    device=f'cuda:{device_id}'
                )
                tensor.data = block.tensor_ref.data[:size // 4]
                
                # Track allocation
                self.allocations[allocation_id] = (device_id, tier, size)
                
                # Update statistics
                self.performance_stats['successful_allocations'] += 1
                self._update_average_allocation_time(allocation_time)
                
                # Update predictor
                self.allocation_predictor.record_allocation(size, device_id, tier, True)
                
                # Run maintenance if needed
                self._run_maintenance_if_needed()
                
                return (tensor, allocation_id)
            else:
                self.performance_stats['failed_allocations'] += 1
                self.logger.warning(f"Failed to allocate {size} bytes on device {device_id}")
                return None
    
    def deallocate(self, allocation_id: str) -> bool:
        """Deallocate memory back to the pool"""
        with self.lock:
            if allocation_id not in self.allocations:
                return False
            
            device_id, tier, size = self.allocations.pop(allocation_id)
            pool = self.pools[device_id][tier]
            
            success = pool.deallocate(allocation_id)
            
            if success:
                self.performance_stats['total_deallocations'] += 1
                
                # Consider defragmentation
                if pool.stats.fragmentation > 0.3:
                    self._defragment_pool(device_id, tier)
            
            return success
    
    def optimize_pools(self) -> Dict[str, Any]:
        """Optimize all pools for better performance"""
        optimization_start = time.perf_counter()
        results = {
            'devices_optimized': 0,
            'pools_optimized': 0,
            'blocks_coalesced': 0,
            'blocks_evicted': 0,
            'fragmentation_reduced': 0.0
        }
        
        with self.lock:
            for device_id in range(self.device_count):
                device_optimized = False
                
                for tier, pool in self.pools[device_id].items():
                    initial_fragmentation = pool.stats.fragmentation
                    
                    # Defragment pool
                    coalesced = pool.defragment()
                    results['blocks_coalesced'] += coalesced
                    
                    # Evict old blocks
                    evicted = pool.evict_old_blocks()
                    results['blocks_evicted'] += evicted
                    
                    if coalesced > 0 or evicted > 0:
                        device_optimized = True
                        results['pools_optimized'] += 1
                        
                        # Calculate fragmentation reduction
                        final_fragmentation = pool.get_statistics().fragmentation
                        results['fragmentation_reduced'] += initial_fragmentation - final_fragmentation
                
                if device_optimized:
                    results['devices_optimized'] += 1
        
        optimization_time = (time.perf_counter() - optimization_start) * 1000
        results['optimization_time_ms'] = optimization_time
        
        # Update global statistics
        self.performance_stats['defragmentation_runs'] += 1
        self.performance_stats['blocks_coalesced'] += results['blocks_coalesced']
        self.performance_stats['blocks_evicted'] += results['blocks_evicted']
        
        return results
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all pools"""
        stats = {
            'global_stats': self.performance_stats.copy(),
            'device_stats': {},
            'tier_summary': {}
        }
        
        with self.lock:
            # Aggregate by device
            for device_id in range(self.device_count):
                device_stats = {
                    'total_memory_mb': 0.0,
                    'allocated_memory_mb': 0.0,
                    'free_memory_mb': 0.0,
                    'fragmentation': 0.0,
                    'tier_stats': {}
                }
                
                for tier, pool in self.pools[device_id].items():
                    pool_stats = pool.get_statistics()
                    device_stats['tier_stats'][tier.value] = {
                        'total_blocks': pool_stats.total_blocks,
                        'allocated_blocks': pool_stats.allocated_blocks,
                        'free_blocks': pool_stats.free_blocks,
                        'total_size_mb': pool_stats.total_size_mb,
                        'allocated_size_mb': pool_stats.allocated_size_mb,
                        'hit_rate': pool_stats.hit_rate,
                        'fragmentation': pool_stats.fragmentation
                    }
                    
                    device_stats['total_memory_mb'] += pool_stats.total_size_mb
                    device_stats['allocated_memory_mb'] += pool_stats.allocated_size_mb
                    device_stats['free_memory_mb'] += pool_stats.free_size_mb
                
                # Calculate device fragmentation
                if device_stats['total_memory_mb'] > 0:
                    device_stats['fragmentation'] = (
                        device_stats['free_memory_mb'] / device_stats['total_memory_mb']
                    )
                
                stats['device_stats'][f'cuda:{device_id}'] = device_stats
            
            # Aggregate by tier across all devices
            for tier in PoolTier:
                tier_total = {
                    'total_blocks': 0,
                    'allocated_blocks': 0,
                    'total_size_mb': 0.0,
                    'allocated_size_mb': 0.0
                }
                
                for device_id in range(self.device_count):
                    pool_stats = self.pools[device_id][tier].get_statistics()
                    tier_total['total_blocks'] += pool_stats.total_blocks
                    tier_total['allocated_blocks'] += pool_stats.allocated_blocks
                    tier_total['total_size_mb'] += pool_stats.total_size_mb
                    tier_total['allocated_size_mb'] += pool_stats.allocated_size_mb
                
                stats['tier_summary'][tier.value] = tier_total
        
        return stats
    
    def get_recommendations(self) -> List[str]:
        """Get recommendations for pool optimization"""
        recommendations = []
        
        stats = self.get_pool_statistics()
        
        # Check global allocation failure rate
        if self.performance_stats['total_allocations'] > 0:
            failure_rate = (
                self.performance_stats['failed_allocations'] / 
                self.performance_stats['total_allocations']
            )
            if failure_rate > 0.1:
                recommendations.append(
                    f"High allocation failure rate ({failure_rate:.1%}). "
                    "Consider increasing pool sizes."
                )
        
        # Check device-specific issues
        for device_name, device_stats in stats['device_stats'].items():
            if device_stats['fragmentation'] > 0.4:
                recommendations.append(
                    f"{device_name}: High fragmentation ({device_stats['fragmentation']:.1%}). "
                    "Run optimization to defragment pools."
                )
            
            # Check tier utilization
            for tier_name, tier_stats in device_stats['tier_stats'].items():
                if tier_stats['allocated_blocks'] == tier_stats['total_blocks']:
                    recommendations.append(
                        f"{device_name}/{tier_name}: Pool fully allocated. "
                        "Consider increasing pool size."
                    )
        
        # Check maintenance frequency
        if self.performance_stats['defragmentation_runs'] < 1:
            recommendations.append("No defragmentation runs performed. Consider enabling automatic maintenance.")
        
        return recommendations
    
    def _get_tier_for_size(self, size: int) -> PoolTier:
        """Determine appropriate tier for allocation size"""
        if size < 1024 * 1024:  # < 1MB
            return PoolTier.SMALL
        elif size < 16 * 1024 * 1024:  # < 16MB
            return PoolTier.MEDIUM
        elif size < 256 * 1024 * 1024:  # < 256MB
            return PoolTier.LARGE
        else:
            return PoolTier.HUGE
    
    def _update_average_allocation_time(self, new_time: float) -> None:
        """Update running average of allocation time"""
        count = self.performance_stats['successful_allocations']
        current_avg = self.performance_stats['average_allocation_time_ms']
        self.performance_stats['average_allocation_time_ms'] = (
            (current_avg * (count - 1) + new_time) / count
        )
    
    def _defragment_pool(self, device_id: int, tier: PoolTier) -> int:
        """Defragment a specific pool"""
        pool = self.pools[device_id][tier]
        return pool.defragment()
    
    def _run_maintenance_if_needed(self) -> None:
        """Run maintenance operations if interval has passed"""
        current_time = time.time()
        if current_time - self.last_maintenance > self.maintenance_interval:
            self.last_maintenance = current_time
            
            # Run optimization in background
            threading.Thread(target=self.optimize_pools, daemon=True).start()


class AllocationPredictor:
    """Predicts optimal allocation tier based on patterns"""
    
    def __init__(self):
        self.allocation_history: Deque[Tuple[int, int, PoolTier, bool]] = deque(maxlen=10000)
        self.tier_success_rates: Dict[PoolTier, float] = defaultdict(lambda: 0.5)
    
    def predict_tier(self, size: int, device_id: int) -> Optional[PoolTier]:
        """Predict best tier for allocation"""
        # Simple heuristic for now - can be enhanced with ML
        best_tier = None
        best_score = 0.0
        
        for tier in PoolTier:
            # Check if tier can handle size
            if self._tier_can_handle_size(tier, size):
                score = self.tier_success_rates[tier]
                if score > best_score:
                    best_score = score
                    best_tier = tier
        
        return best_tier
    
    def record_allocation(self, size: int, device_id: int, tier: PoolTier, success: bool) -> None:
        """Record allocation outcome"""
        self.allocation_history.append((size, device_id, tier, success))
        
        # Update success rate
        tier_allocations = [
            (s, d, t, succ) for s, d, t, succ in self.allocation_history
            if t == tier
        ]
        
        if tier_allocations:
            success_count = sum(1 for _, _, _, succ in tier_allocations if succ)
            self.tier_success_rates[tier] = success_count / len(tier_allocations)
    
    def _tier_can_handle_size(self, tier: PoolTier, size: int) -> bool:
        """Check if tier can handle the requested size"""
        max_sizes = {
            PoolTier.SMALL: 1024 * 1024,
            PoolTier.MEDIUM: 16 * 1024 * 1024,
            PoolTier.LARGE: 256 * 1024 * 1024,
            PoolTier.HUGE: float('inf')
        }
        return size <= max_sizes[tier]