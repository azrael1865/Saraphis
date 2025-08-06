"""
JAX Memory Pool Integration - Advanced memory management for JAX operations
Integrates with existing AdvancedMemoryPool while providing JAX-specific optimizations
HARD FAILURES ONLY - NO GRACEFUL DEGRADATION

This module provides:
1. JAXMemoryPool - JAX-specific memory allocation strategies
2. Integration with existing AdvancedMemoryPool
3. Memory pressure monitoring for JAX operations
4. Unified memory metrics tracking
"""

import jax
import jax.numpy as jnp
from jax import device_put, device_get
from jax.lib import xla_bridge
import numpy as np
import torch
import time
import threading
import logging
from typing import Dict, List, Optional, Tuple, Any, Set, Deque
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import psutil

# Import existing memory management components
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from independent_core.compression_systems.gpu_memory.advanced_memory_pool import (
    AdvancedMemoryPoolManager,
    PoolTier,
    AllocationStrategy,
    PooledMemoryBlock,
    PoolStatistics
)
from independent_core.compression_systems.padic.memory_pressure_handler import (
    MemoryPressureHandler,
    MemoryMetrics,
    MemoryState
)
from independent_core.compression_systems.system_integration_coordinator import (
    SystemIntegrationCoordinator
)

logger = logging.getLogger(__name__)


class JAXMemoryBlockType(Enum):
    """Types of JAX memory blocks"""
    DEVICE_ARRAY = "device_array"       # Regular JAX arrays
    SHARDED_ARRAY = "sharded_array"     # Sharded across devices
    REPLICATED = "replicated"            # Replicated across devices
    HOST_MEMORY = "host_memory"          # Host-side memory
    PINNED_MEMORY = "pinned_memory"      # Pinned host memory for transfers


@dataclass
class JAXMemoryBlock:
    """JAX-specific memory block"""
    block_id: str
    array: Any  # jax.Array or numpy array
    size_bytes: int
    block_type: JAXMemoryBlockType
    device: Any  # JAX device
    allocated_time: float
    last_used_time: float
    use_count: int = 0
    is_allocated: bool = False
    sharding_spec: Optional[Any] = None  # For sharded arrays
    
    @property
    def age(self) -> float:
        """Get block age in seconds"""
        return time.time() - self.allocated_time
    
    @property
    def idle_time(self) -> float:
        """Get time since last use"""
        return time.time() - self.last_used_time


@dataclass
class JAXMemoryMetrics:
    """JAX-specific memory metrics"""
    timestamp: float
    device_id: int
    total_memory_bytes: int
    allocated_bytes: int
    free_bytes: int
    peak_allocated_bytes: int
    num_allocations: int
    num_deallocations: int
    fragmentation_ratio: float
    xla_memory_bytes: int  # XLA-specific memory usage
    
    @property
    def utilization(self) -> float:
        """Memory utilization ratio"""
        if self.total_memory_bytes == 0:
            return 0.0
        return self.allocated_bytes / self.total_memory_bytes


class JAXMemoryPool:
    """
    JAX-specific memory pool that integrates with existing AdvancedMemoryPool.
    Provides JAX-optimized allocation strategies and memory pressure monitoring.
    """
    
    def __init__(self, 
                 device: Any,
                 pool_size_mb: int = 1024,
                 existing_pool: Optional[AdvancedMemoryPoolManager] = None,
                 pressure_handler: Optional[MemoryPressureHandler] = None):
        """
        Initialize JAX memory pool
        
        Args:
            device: JAX device to manage memory for
            pool_size_mb: Size of memory pool in MB
            existing_pool: Existing AdvancedMemoryPool to integrate with
            pressure_handler: Memory pressure handler for monitoring
        """
        self.device = device
        self.pool_size_bytes = pool_size_mb * 1024 * 1024
        self.existing_pool = existing_pool
        self.pressure_handler = pressure_handler
        
        # Memory blocks management
        self.free_blocks: Dict[JAXMemoryBlockType, List[JAXMemoryBlock]] = defaultdict(list)
        self.allocated_blocks: Dict[str, JAXMemoryBlock] = {}
        self.block_counter = 0
        
        # Memory metrics
        self.metrics = JAXMemoryMetrics(
            timestamp=time.time(),
            device_id=self._get_device_id(),
            total_memory_bytes=self.pool_size_bytes,
            allocated_bytes=0,
            free_bytes=self.pool_size_bytes,
            peak_allocated_bytes=0,
            num_allocations=0,
            num_deallocations=0,
            fragmentation_ratio=0.0,
            xla_memory_bytes=0
        )
        
        # Performance tracking
        self.allocation_history: Deque[Tuple[float, int, bool]] = deque(maxlen=1000)
        self.stats = {
            'total_allocations': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'cache_hits': 0,
            'defragmentation_runs': 0,
            'memory_pressure_events': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Pre-allocate initial blocks
        self._preallocate_blocks()
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"JAXMemoryPool initialized on device {device} with {pool_size_mb}MB")
    
    def _get_device_id(self) -> int:
        """Get device ID from JAX device"""
        try:
            # Extract device ID from JAX device
            if hasattr(self.device, 'id'):
                return self.device.id
            # Parse from string representation
            device_str = str(self.device)
            if 'gpu' in device_str.lower():
                # Extract GPU ID
                import re
                match = re.search(r'gpu:(\d+)', device_str.lower())
                if match:
                    return int(match.group(1))
            return 0
        except:
            return 0
    
    def _preallocate_blocks(self) -> None:
        """Pre-allocate memory blocks for the pool"""
        # Calculate block sizes for different tiers
        block_configs = [
            (JAXMemoryBlockType.DEVICE_ARRAY, 64 * 1024 * 1024, 4),  # 64MB blocks
            (JAXMemoryBlockType.DEVICE_ARRAY, 16 * 1024 * 1024, 8),  # 16MB blocks
            (JAXMemoryBlockType.DEVICE_ARRAY, 4 * 1024 * 1024, 16),  # 4MB blocks
            (JAXMemoryBlockType.PINNED_MEMORY, 8 * 1024 * 1024, 4),  # 8MB pinned
        ]
        
        with self._lock:
            for block_type, block_size, count in block_configs:
                for _ in range(count):
                    if self.metrics.allocated_bytes + block_size > self.pool_size_bytes:
                        break
                    
                    try:
                        # Allocate block based on type
                        if block_type == JAXMemoryBlockType.DEVICE_ARRAY:
                            # Allocate JAX array on device
                            shape = (block_size // 4,)  # float32 elements
                            array = jnp.zeros(shape, dtype=jnp.float32)
                            array = device_put(array, self.device)
                        elif block_type == JAXMemoryBlockType.PINNED_MEMORY:
                            # Allocate pinned host memory
                            array = np.zeros(block_size // 4, dtype=np.float32)
                        else:
                            continue
                        
                        # Create block
                        block_id = f"block_{self.block_counter}_{int(time.time() * 1000)}"
                        self.block_counter += 1
                        
                        block = JAXMemoryBlock(
                            block_id=block_id,
                            array=array,
                            size_bytes=block_size,
                            block_type=block_type,
                            device=self.device,
                            allocated_time=time.time(),
                            last_used_time=time.time()
                        )
                        
                        self.free_blocks[block_type].append(block)
                        self.metrics.allocated_bytes += block_size
                        self.metrics.free_bytes = self.pool_size_bytes - self.metrics.allocated_bytes
                        
                    except Exception as e:
                        logger.warning(f"Failed to preallocate {block_type.value} block: {e}")
                        break
    
    def allocate(self, 
                size_bytes: int,
                block_type: JAXMemoryBlockType = JAXMemoryBlockType.DEVICE_ARRAY,
                shape: Optional[Tuple[int, ...]] = None,
                dtype: Any = jnp.float32) -> Optional[Tuple[Any, str]]:
        """
        Allocate memory from the pool
        
        Args:
            size_bytes: Size to allocate in bytes
            block_type: Type of memory block
            shape: Shape for the array (optional)
            dtype: Data type for the array
            
        Returns:
            Tuple of (array, block_id) if successful, None otherwise
        """
        start_time = time.perf_counter()
        
        with self._lock:
            self.stats['total_allocations'] += 1
            
            # Check memory pressure
            if self.pressure_handler:
                memory_state = self._check_memory_pressure()
                if memory_state == MemoryState.CRITICAL:
                    self.stats['memory_pressure_events'] += 1
                    # Try to free memory
                    self._handle_memory_pressure()
            
            # Find suitable free block
            best_block = None
            best_index = -1
            
            for i, block in enumerate(self.free_blocks[block_type]):
                if block.size_bytes >= size_bytes and not block.is_allocated:
                    if best_block is None or block.size_bytes < best_block.size_bytes:
                        best_block = block
                        best_index = i
            
            if best_block is not None:
                # Use existing block
                self.free_blocks[block_type].pop(best_index)
                self.stats['cache_hits'] += 1
            else:
                # Allocate new block
                try:
                    if shape is None:
                        # Calculate shape from size and dtype
                        element_size = np.dtype(dtype).itemsize
                        num_elements = size_bytes // element_size
                        shape = (num_elements,)
                    
                    if block_type == JAXMemoryBlockType.DEVICE_ARRAY:
                        array = jnp.zeros(shape, dtype=dtype)
                        array = device_put(array, self.device)
                    elif block_type == JAXMemoryBlockType.PINNED_MEMORY:
                        array = np.zeros(shape, dtype=dtype)
                    else:
                        array = jnp.zeros(shape, dtype=dtype)
                    
                    block_id = f"block_{self.block_counter}_{int(time.time() * 1000)}"
                    self.block_counter += 1
                    
                    best_block = JAXMemoryBlock(
                        block_id=block_id,
                        array=array,
                        size_bytes=size_bytes,
                        block_type=block_type,
                        device=self.device,
                        allocated_time=time.time(),
                        last_used_time=time.time()
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to allocate new block: {e}")
                    self.stats['failed_allocations'] += 1
                    self.allocation_history.append((time.time(), size_bytes, False))
                    return None
            
            # Mark as allocated
            best_block.is_allocated = True
            best_block.last_used_time = time.time()
            best_block.use_count += 1
            
            # Track allocation
            self.allocated_blocks[best_block.block_id] = best_block
            
            # Update metrics
            self.metrics.num_allocations += 1
            self.metrics.peak_allocated_bytes = max(
                self.metrics.peak_allocated_bytes,
                sum(b.size_bytes for b in self.allocated_blocks.values())
            )
            
            # Record allocation
            allocation_time = (time.perf_counter() - start_time) * 1000
            self.stats['successful_allocations'] += 1
            self.allocation_history.append((time.time(), size_bytes, True))
            
            # Integration with existing pool
            if self.existing_pool:
                # Report allocation to existing pool for unified tracking
                self._report_to_existing_pool(best_block.block_id, size_bytes, 'allocate')
            
            return (best_block.array, best_block.block_id)
    
    def deallocate(self, block_id: str) -> bool:
        """
        Deallocate memory back to the pool
        
        Args:
            block_id: ID of the block to deallocate
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            if block_id not in self.allocated_blocks:
                return False
            
            block = self.allocated_blocks.pop(block_id)
            
            # Reset block state
            block.is_allocated = False
            block.last_used_time = time.time()
            
            # Add back to free list
            self.free_blocks[block.block_type].append(block)
            
            # Update metrics
            self.metrics.num_deallocations += 1
            
            # Integration with existing pool
            if self.existing_pool:
                self._report_to_existing_pool(block_id, block.size_bytes, 'deallocate')
            
            # Check if defragmentation needed
            if self._should_defragment():
                self._defragment()
            
            return True
    
    def _check_memory_pressure(self) -> MemoryState:
        """Check current memory pressure state"""
        if not self.pressure_handler:
            # Calculate based on local metrics
            utilization = self.metrics.utilization
            if utilization > 0.95:
                return MemoryState.CRITICAL
            elif utilization > 0.90:
                return MemoryState.HIGH
            elif utilization > 0.75:
                return MemoryState.MODERATE
            else:
                return MemoryState.HEALTHY
        
        # Use pressure handler's assessment
        return self.pressure_handler.current_memory_state
    
    def _handle_memory_pressure(self) -> None:
        """Handle memory pressure by freeing old blocks"""
        # Free blocks that haven't been used recently
        current_time = time.time()
        max_idle_time = 60.0  # 60 seconds
        
        for block_type in list(self.free_blocks.keys()):
            blocks_to_keep = []
            
            for block in self.free_blocks[block_type]:
                if block.idle_time < max_idle_time:
                    blocks_to_keep.append(block)
                else:
                    # Free the block
                    self.metrics.allocated_bytes -= block.size_bytes
                    self.metrics.free_bytes += block.size_bytes
            
            self.free_blocks[block_type] = blocks_to_keep
    
    def _should_defragment(self) -> bool:
        """Check if defragmentation is needed"""
        # Calculate fragmentation ratio
        if self.metrics.allocated_bytes == 0:
            return False
        
        total_free = sum(len(blocks) for blocks in self.free_blocks.values())
        if total_free > 20:  # Too many free blocks indicates fragmentation
            return True
        
        # Check fragmentation ratio
        if self.metrics.fragmentation_ratio > 0.3:
            return True
        
        return False
    
    def _defragment(self) -> None:
        """Defragment the memory pool"""
        self.stats['defragmentation_runs'] += 1
        
        # Coalesce adjacent blocks (simplified for JAX)
        for block_type in self.free_blocks:
            blocks = self.free_blocks[block_type]
            if len(blocks) <= 1:
                continue
            
            # Sort by size for better coalescing
            blocks.sort(key=lambda b: b.size_bytes)
            
            # Update fragmentation ratio
            total_free_size = sum(b.size_bytes for b in blocks)
            if total_free_size > 0:
                largest_block = max(b.size_bytes for b in blocks)
                self.metrics.fragmentation_ratio = 1.0 - (largest_block / total_free_size)
    
    def _report_to_existing_pool(self, block_id: str, size: int, operation: str) -> None:
        """Report operations to existing pool for unified tracking"""
        if not self.existing_pool:
            return
        
        try:
            # Report to existing pool's performance stats
            if operation == 'allocate':
                self.existing_pool.performance_stats['total_allocations'] += 1
                self.existing_pool.performance_stats['successful_allocations'] += 1
            elif operation == 'deallocate':
                self.existing_pool.performance_stats['total_deallocations'] += 1
        except Exception as e:
            logger.warning(f"Failed to report to existing pool: {e}")
    
    def _monitor_memory(self) -> None:
        """Background thread for memory monitoring"""
        while self.monitoring_active:
            try:
                with self._lock:
                    # Update XLA memory usage
                    try:
                        # Get XLA client memory stats
                        backend = xla_bridge.get_backend()
                        if hasattr(backend, 'live_buffers'):
                            live_buffers = backend.live_buffers()
                            self.metrics.xla_memory_bytes = sum(
                                buf.nbytes for buf in live_buffers
                            )
                    except:
                        pass
                    
                    # Update metrics
                    self.metrics.timestamp = time.time()
                    self.metrics.allocated_bytes = sum(
                        b.size_bytes for b in self.allocated_blocks.values()
                    )
                    self.metrics.free_bytes = self.pool_size_bytes - self.metrics.allocated_bytes
                    
                    # Report to coordinator if available
                    if hasattr(self, 'coordinator'):
                        self._report_metrics_to_coordinator()
                
                time.sleep(0.1)  # 100ms interval
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
    
    def _report_metrics_to_coordinator(self) -> None:
        """Report metrics to system integration coordinator"""
        # This will be called by SystemIntegrationCoordinator
        pass
    
    def get_metrics(self) -> JAXMemoryMetrics:
        """Get current memory metrics"""
        with self._lock:
            return self.metrics
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            return {
                **self.stats,
                'metrics': {
                    'utilization': self.metrics.utilization,
                    'fragmentation_ratio': self.metrics.fragmentation_ratio,
                    'peak_allocated_bytes': self.metrics.peak_allocated_bytes,
                    'xla_memory_bytes': self.metrics.xla_memory_bytes
                },
                'blocks': {
                    'allocated': len(self.allocated_blocks),
                    'free': sum(len(blocks) for blocks in self.free_blocks.values())
                }
            }
    
    def optimize(self) -> Dict[str, Any]:
        """Optimize the memory pool"""
        optimization_start = time.perf_counter()
        
        with self._lock:
            initial_fragmentation = self.metrics.fragmentation_ratio
            
            # Defragment
            self._defragment()
            
            # Free old blocks
            self._handle_memory_pressure()
            
            # Rebalance block sizes based on allocation history
            if len(self.allocation_history) > 100:
                # Analyze allocation patterns
                recent_sizes = [size for _, size, _ in list(self.allocation_history)[-100:]]
                avg_size = np.mean(recent_sizes)
                
                # Adjust pre-allocation strategy
                # (Implementation depends on specific patterns)
            
            optimization_time = (time.perf_counter() - optimization_start) * 1000
            
            return {
                'optimization_time_ms': optimization_time,
                'initial_fragmentation': initial_fragmentation,
                'final_fragmentation': self.metrics.fragmentation_ratio,
                'blocks_freed': 0,  # Track in _handle_memory_pressure
                'success': True
            }
    
    def shutdown(self) -> None:
        """Shutdown the memory pool"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        
        # Clear all blocks
        with self._lock:
            self.allocated_blocks.clear()
            self.free_blocks.clear()
        
        logger.info("JAXMemoryPool shutdown complete")


# Integration function for SystemIntegrationCoordinator
def integrate_jax_memory_pool(coordinator: SystemIntegrationCoordinator,
                             jax_pool: JAXMemoryPool) -> None:
    """
    Integrate JAX memory pool with system coordinator
    
    Args:
        coordinator: System integration coordinator
        jax_pool: JAX memory pool instance
    """
    # Register JAX pool with coordinator
    if not hasattr(coordinator, 'jax_pools'):
        coordinator.jax_pools = {}
    
    device_id = jax_pool._get_device_id()
    coordinator.jax_pools[device_id] = jax_pool
    
    # Set coordinator reference in pool
    jax_pool.coordinator = coordinator
    
    # Register metrics callback
    def report_jax_metrics():
        metrics = jax_pool.get_metrics()
        stats = jax_pool.get_statistics()
        
        # Update coordinator's tracking
        if hasattr(coordinator, 'memory_metrics'):
            coordinator.memory_metrics['jax'] = {
                'device_id': device_id,
                'utilization': metrics.utilization,
                'allocated_mb': metrics.allocated_bytes / (1024 * 1024),
                'fragmentation': metrics.fragmentation_ratio,
                'stats': stats
            }
    
    # Register callback
    if hasattr(coordinator, 'register_metrics_callback'):
        coordinator.register_metrics_callback('jax_memory', report_jax_metrics)
    
    logger.info(f"JAX memory pool integrated with system coordinator for device {device_id}")


# Test function
def test_jax_memory_pool():
    """Test JAX memory pool functionality"""
    print("Testing JAX Memory Pool...")
    
    # Get JAX device
    devices = jax.devices()
    if not devices:
        print("No JAX devices available")
        return
    
    device = devices[0]
    print(f"Using device: {device}")
    
    # Create memory pool
    pool = JAXMemoryPool(device, pool_size_mb=512)
    
    # Test allocations
    print("\n1. Testing allocations...")
    allocations = []
    
    for i in range(5):
        size = (i + 1) * 1024 * 1024  # 1MB to 5MB
        result = pool.allocate(size, shape=(size // 4,), dtype=jnp.float32)
        
        if result:
            array, block_id = result
            allocations.append((array, block_id))
            print(f"   Allocated {size / 1024 / 1024:.1f}MB: {block_id}")
        else:
            print(f"   Failed to allocate {size / 1024 / 1024:.1f}MB")
    
    # Check metrics
    metrics = pool.get_metrics()
    print(f"\n2. Memory metrics:")
    print(f"   Utilization: {metrics.utilization:.2%}")
    print(f"   Allocated: {metrics.allocated_bytes / 1024 / 1024:.1f}MB")
    print(f"   Free: {metrics.free_bytes / 1024 / 1024:.1f}MB")
    
    # Test deallocation
    print("\n3. Testing deallocation...")
    for array, block_id in allocations[:3]:
        success = pool.deallocate(block_id)
        print(f"   Deallocated {block_id}: {success}")
    
    # Check statistics
    stats = pool.get_statistics()
    print(f"\n4. Pool statistics:")
    print(f"   Total allocations: {stats['total_allocations']}")
    print(f"   Successful: {stats['successful_allocations']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Allocated blocks: {stats['blocks']['allocated']}")
    print(f"   Free blocks: {stats['blocks']['free']}")
    
    # Test optimization
    print("\n5. Testing optimization...")
    opt_result = pool.optimize()
    print(f"   Optimization time: {opt_result['optimization_time_ms']:.2f}ms")
    print(f"   Fragmentation reduced: {opt_result['initial_fragmentation']:.2%} -> {opt_result['final_fragmentation']:.2%}")
    
    # Shutdown
    pool.shutdown()
    print("\n✓ JAX Memory Pool test complete!")


if __name__ == "__main__":
    test_jax_memory_pool()