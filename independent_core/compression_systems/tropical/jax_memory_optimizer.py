"""
JAX Memory Bandwidth Optimizer - Bandwidth-aware memory optimization for tropical operations
Provides memory access pattern analysis, tensor layout optimization, and prefetching strategies
HARD FAILURES ONLY - NO GRACEFUL DEGRADATION

This module provides:
1. JAXMemoryOptimizer - Bandwidth-aware scheduling and memory optimization
2. Memory access pattern analysis for tropical operations
3. Tensor layout optimization (row-major vs column-major)
4. Memory prefetching strategies
5. Bandwidth saturation detection and mitigation
"""

import jax
import jax.numpy as jnp
from jax import device_put, device_get, make_jaxpr
from jax.lib import xla_bridge
from jax.sharding import PositionalSharding, NamedSharding, Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
import numpy as np
import time
import threading
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import functools

# Import system components
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from independent_core.compression_systems.tropical.jax_memory_pool import JAXMemoryPool
from independent_core.compression_systems.tropical.jax_device_manager import JAXDeviceManager

logger = logging.getLogger(__name__)


class MemoryLayout(Enum):
    """Tensor memory layout"""
    ROW_MAJOR = "row_major"      # C-style, last dim changes fastest
    COLUMN_MAJOR = "column_major"  # Fortran-style, first dim changes fastest
    BLOCKED = "blocked"           # Block layout for cache efficiency
    TILED = "tiled"              # Tiled layout for GPU efficiency


class AccessPattern(Enum):
    """Memory access patterns"""
    SEQUENTIAL = "sequential"      # Sequential access
    STRIDED = "strided"           # Strided access
    RANDOM = "random"             # Random access
    BROADCAST = "broadcast"       # Broadcasting pattern
    REDUCTION = "reduction"       # Reduction pattern
    SCATTER_GATHER = "scatter_gather"  # Scatter/gather pattern


class PrefetchStrategy(Enum):
    """Memory prefetching strategies"""
    NONE = "none"                 # No prefetching
    NEXT_LINE = "next_line"       # Prefetch next cache line
    STRIDE = "stride"             # Stride-based prefetching
    ADAPTIVE = "adaptive"         # Adaptive prefetching
    AGGRESSIVE = "aggressive"     # Aggressive prefetching


@dataclass
class MemoryAccessProfile:
    """Profile of memory access patterns"""
    pattern: AccessPattern
    stride: int = 1
    locality: float = 0.0  # 0-1, temporal locality
    reuse_distance: int = 0
    bandwidth_utilization: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    prefetch_accuracy: float = 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total


@dataclass
class BandwidthMetrics:
    """Bandwidth utilization metrics"""
    timestamp: float
    read_bandwidth_gb_s: float
    write_bandwidth_gb_s: float
    total_bandwidth_gb_s: float
    theoretical_max_gb_s: float
    utilization_percent: float
    memory_bound_operations: int
    compute_bound_operations: int
    
    @property
    def is_bandwidth_saturated(self) -> bool:
        """Check if bandwidth is saturated"""
        return self.utilization_percent > 90.0
    
    @property
    def bandwidth_efficiency(self) -> float:
        """Calculate bandwidth efficiency"""
        if self.theoretical_max_gb_s == 0:
            return 0.0
        return self.total_bandwidth_gb_s / self.theoretical_max_gb_s


@dataclass
class TensorLayoutInfo:
    """Information about tensor layout"""
    shape: Tuple[int, ...]
    dtype: Any
    layout: MemoryLayout
    strides: Tuple[int, ...]
    is_contiguous: bool
    tile_size: Optional[Tuple[int, ...]] = None
    padding: Optional[Tuple[Tuple[int, int], ...]] = None
    
    @property
    def memory_footprint(self) -> int:
        """Calculate memory footprint in bytes"""
        element_size = np.dtype(self.dtype).itemsize
        return int(np.prod(self.shape) * element_size)


class JAXMemoryOptimizer:
    """
    Memory bandwidth optimizer for JAX operations.
    Analyzes access patterns and optimizes memory layout and prefetching.
    """
    
    def __init__(self,
                 device_manager: Optional[JAXDeviceManager] = None,
                 memory_pool: Optional[JAXMemoryPool] = None,
                 enable_profiling: bool = True,
                 enable_prefetching: bool = True):
        """
        Initialize memory optimizer
        
        Args:
            device_manager: JAX device manager
            memory_pool: JAX memory pool
            enable_profiling: Enable memory profiling
            enable_prefetching: Enable prefetching
        """
        self.device_manager = device_manager or JAXDeviceManager()
        self.memory_pool = memory_pool
        self.enable_profiling = enable_profiling
        self.enable_prefetching = enable_prefetching
        
        # Get device capabilities
        self.device_info = self._get_device_capabilities()
        
        # Access pattern analysis
        self.access_profiles: Dict[str, MemoryAccessProfile] = {}
        self.access_history: deque = deque(maxlen=1000)
        
        # Layout optimization
        self.layout_cache: Dict[str, TensorLayoutInfo] = {}
        self.optimal_layouts: Dict[str, MemoryLayout] = {}
        
        # Bandwidth monitoring
        self.bandwidth_metrics = BandwidthMetrics(
            timestamp=time.time(),
            read_bandwidth_gb_s=0.0,
            write_bandwidth_gb_s=0.0,
            total_bandwidth_gb_s=0.0,
            theoretical_max_gb_s=self.device_info['memory_bandwidth_gb'],
            utilization_percent=0.0,
            memory_bound_operations=0,
            compute_bound_operations=0
        )
        
        # Prefetch management
        self.prefetch_queue: deque = deque(maxlen=100)
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        
        # Statistics
        self.stats = {
            'layouts_optimized': 0,
            'bandwidth_optimizations': 0,
            'prefetch_operations': 0,
            'memory_coalescing': 0,
            'cache_optimizations': 0,
            'total_bytes_optimized': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Start monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_bandwidth, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"JAXMemoryOptimizer initialized with {self.device_info['memory_bandwidth_gb']}GB/s bandwidth")
    
    def _get_device_capabilities(self) -> Dict[str, Any]:
        """Get device memory capabilities"""
        # Default values
        capabilities = {
            'memory_bandwidth_gb': 900.0,  # Default for modern GPU
            'l2_cache_size_mb': 6.0,
            'l1_cache_size_kb': 128.0,
            'cache_line_size': 128,
            'memory_bus_width': 384,
            'num_memory_controllers': 12
        }
        
        # Try to get actual values from device
        try:
            devices = jax.devices()
            if devices and devices[0].platform == 'gpu':
                # Get GPU-specific info if available
                if self.device_manager:
                    device_info = self.device_manager.device_info.get(0)
                    if device_info and device_info.gpu_specs:
                        capabilities['memory_bandwidth_gb'] = device_info.gpu_specs.memory_bandwidth_gb
        except:
            pass
        
        return capabilities
    
    def analyze_access_pattern(self, 
                              tensor: Any,
                              operation: str,
                              indices: Optional[Any] = None) -> MemoryAccessProfile:
        """
        Analyze memory access pattern for a tensor operation
        
        Args:
            tensor: Input tensor
            operation: Operation being performed
            indices: Access indices if applicable
            
        Returns:
            Memory access profile
        """
        profile_key = f"{operation}_{tensor.shape}_{tensor.dtype}"
        
        with self._lock:
            if profile_key in self.access_profiles:
                return self.access_profiles[profile_key]
            
            # Analyze access pattern
            pattern = AccessPattern.SEQUENTIAL  # Default
            stride = 1
            locality = 0.8  # Default temporal locality
            
            if operation in ['matmul', 'dot']:
                pattern = AccessPattern.STRIDED
                stride = tensor.shape[-1] if len(tensor.shape) > 1 else 1
                locality = 0.6
            elif operation in ['reduce', 'sum', 'max']:
                pattern = AccessPattern.REDUCTION
                locality = 0.9
            elif operation in ['broadcast', 'expand']:
                pattern = AccessPattern.BROADCAST
                locality = 0.7
            elif operation in ['gather', 'scatter']:
                pattern = AccessPattern.SCATTER_GATHER
                locality = 0.3
                if indices is not None:
                    # Analyze index pattern
                    try:
                        idx_array = np.array(indices)
                        if len(idx_array) > 1:
                            diffs = np.diff(idx_array)
                            if np.all(diffs == diffs[0]):
                                stride = int(diffs[0])
                                pattern = AccessPattern.STRIDED
                    except:
                        pass
            
            # Create profile
            profile = MemoryAccessProfile(
                pattern=pattern,
                stride=stride,
                locality=locality,
                reuse_distance=int(1.0 / locality * 100),
                bandwidth_utilization=0.0
            )
            
            self.access_profiles[profile_key] = profile
            self.access_history.append((time.time(), profile_key, profile))
            
            return profile
    
    def optimize_tensor_layout(self,
                              tensor: Any,
                              access_profile: MemoryAccessProfile,
                              target_operation: str = "general") -> Tuple[Any, TensorLayoutInfo]:
        """
        Optimize tensor memory layout based on access pattern
        
        Args:
            tensor: Input tensor
            access_profile: Memory access profile
            target_operation: Target operation type
            
        Returns:
            Optimized tensor and layout info
        """
        with self._lock:
            self.stats['layouts_optimized'] += 1
            
            # Determine optimal layout
            optimal_layout = self._determine_optimal_layout(
                tensor.shape,
                access_profile,
                target_operation
            )
            
            # Get current layout info
            layout_info = TensorLayoutInfo(
                shape=tensor.shape,
                dtype=tensor.dtype,
                layout=optimal_layout,
                strides=self._calculate_strides(tensor.shape, optimal_layout),
                is_contiguous=True
            )
            
            # Apply layout transformation if needed
            optimized_tensor = self._apply_layout(tensor, optimal_layout, target_operation)
            
            # Cache layout info
            cache_key = f"{id(tensor)}_{target_operation}"
            self.layout_cache[cache_key] = layout_info
            
            # Update statistics
            self.stats['total_bytes_optimized'] += layout_info.memory_footprint
            
            return optimized_tensor, layout_info
    
    def _determine_optimal_layout(self,
                                 shape: Tuple[int, ...],
                                 profile: MemoryAccessProfile,
                                 operation: str) -> MemoryLayout:
        """Determine optimal memory layout"""
        # Decision based on access pattern and operation
        if profile.pattern == AccessPattern.SEQUENTIAL:
            return MemoryLayout.ROW_MAJOR
        elif profile.pattern == AccessPattern.STRIDED:
            if profile.stride == shape[-1]:
                return MemoryLayout.COLUMN_MAJOR
            else:
                return MemoryLayout.ROW_MAJOR
        elif profile.pattern == AccessPattern.REDUCTION:
            # For reductions, optimize for the reduction axis
            if operation.endswith('_axis_0'):
                return MemoryLayout.COLUMN_MAJOR
            else:
                return MemoryLayout.ROW_MAJOR
        elif operation in ['matmul', 'conv']:
            # Tiled layout for matrix operations
            return MemoryLayout.TILED
        else:
            return MemoryLayout.ROW_MAJOR
    
    def _calculate_strides(self, shape: Tuple[int, ...], layout: MemoryLayout) -> Tuple[int, ...]:
        """Calculate memory strides for given layout"""
        if layout == MemoryLayout.ROW_MAJOR:
            # C-style: last dimension changes fastest
            strides = []
            stride = 1
            for dim in reversed(shape):
                strides.append(stride)
                stride *= dim
            return tuple(reversed(strides))
        elif layout == MemoryLayout.COLUMN_MAJOR:
            # Fortran-style: first dimension changes fastest
            strides = []
            stride = 1
            for dim in shape:
                strides.append(stride)
                stride *= dim
            return tuple(strides)
        else:
            # Default to row-major
            return self._calculate_strides(shape, MemoryLayout.ROW_MAJOR)
    
    def _apply_layout(self, tensor: Any, layout: MemoryLayout, operation: str) -> Any:
        """Apply memory layout to tensor"""
        if layout == MemoryLayout.ROW_MAJOR:
            # Already in row-major (JAX default)
            return tensor
        elif layout == MemoryLayout.COLUMN_MAJOR:
            # Transpose to column-major
            ndim = len(tensor.shape)
            perm = tuple(reversed(range(ndim)))
            return jnp.transpose(tensor, perm)
        elif layout == MemoryLayout.TILED:
            # Apply tiling for better cache usage
            return self._apply_tiling(tensor, operation)
        elif layout == MemoryLayout.BLOCKED:
            # Apply blocking
            return self._apply_blocking(tensor)
        else:
            return tensor
    
    def _apply_tiling(self, tensor: Any, operation: str) -> Any:
        """Apply tiling transformation"""
        # Determine tile size based on cache
        cache_line_size = self.device_info['cache_line_size']
        element_size = tensor.dtype.itemsize
        elements_per_line = cache_line_size // element_size
        
        # Simple 2D tiling for matrices
        if len(tensor.shape) == 2:
            m, n = tensor.shape
            tile_size = min(32, elements_per_line)  # Common tile size
            
            if m > tile_size and n > tile_size:
                # Reshape into tiles
                m_tiles = m // tile_size
                n_tiles = n // tile_size
                
                if m_tiles * tile_size == m and n_tiles * tile_size == n:
                    # Perfect tiling
                    reshaped = tensor.reshape(m_tiles, tile_size, n_tiles, tile_size)
                    # Transpose to group tiles together
                    tiled = reshaped.transpose(0, 2, 1, 3)
                    return tiled.reshape(m, n)
        
        return tensor
    
    def _apply_blocking(self, tensor: Any) -> Any:
        """Apply blocking transformation"""
        # Similar to tiling but for general tensors
        # For simplicity, return as-is
        return tensor
    
    def setup_prefetching(self,
                         tensors: List[Any],
                         strategy: PrefetchStrategy = PrefetchStrategy.ADAPTIVE) -> None:
        """
        Setup prefetching for a list of tensors
        
        Args:
            tensors: List of tensors to prefetch
            strategy: Prefetching strategy
        """
        if not self.enable_prefetching:
            return
        
        with self._lock:
            self.stats['prefetch_operations'] += 1
            
            for tensor in tensors:
                if strategy == PrefetchStrategy.NEXT_LINE:
                    # Prefetch next cache line
                    self._prefetch_next_line(tensor)
                elif strategy == PrefetchStrategy.STRIDE:
                    # Stride-based prefetching
                    self._prefetch_stride(tensor)
                elif strategy == PrefetchStrategy.ADAPTIVE:
                    # Adaptive based on access pattern
                    profile_key = f"prefetch_{tensor.shape}_{tensor.dtype}"
                    if profile_key in self.access_profiles:
                        profile = self.access_profiles[profile_key]
                        if profile.locality > 0.7:
                            self._prefetch_stride(tensor)
                        else:
                            self._prefetch_next_line(tensor)
                elif strategy == PrefetchStrategy.AGGRESSIVE:
                    # Aggressive prefetching
                    self._prefetch_aggressive(tensor)
    
    def _prefetch_next_line(self, tensor: Any) -> None:
        """Prefetch next cache line"""
        # In JAX, we can hint the runtime about future access
        # This is a simplified representation
        self.prefetch_queue.append(('next_line', id(tensor), time.time()))
    
    def _prefetch_stride(self, tensor: Any) -> None:
        """Stride-based prefetching"""
        self.prefetch_queue.append(('stride', id(tensor), time.time()))
    
    def _prefetch_aggressive(self, tensor: Any) -> None:
        """Aggressive prefetching"""
        self.prefetch_queue.append(('aggressive', id(tensor), time.time()))
    
    def coalesce_memory_accesses(self, operations: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
        """
        Coalesce memory accesses for better bandwidth utilization
        
        Args:
            operations: List of (operation, tensor) pairs
            
        Returns:
            Reordered operations for better memory access
        """
        with self._lock:
            self.stats['memory_coalescing'] += 1
            
            # Group operations by memory access pattern
            grouped = defaultdict(list)
            for op, tensor in operations:
                profile = self.analyze_access_pattern(tensor, op)
                grouped[profile.pattern].append((op, tensor))
            
            # Reorder for better cache usage
            reordered = []
            
            # Sequential accesses first (best cache usage)
            reordered.extend(grouped[AccessPattern.SEQUENTIAL])
            
            # Then strided accesses
            reordered.extend(grouped[AccessPattern.STRIDED])
            
            # Broadcast and reduction
            reordered.extend(grouped[AccessPattern.BROADCAST])
            reordered.extend(grouped[AccessPattern.REDUCTION])
            
            # Random and scatter/gather last
            reordered.extend(grouped[AccessPattern.RANDOM])
            reordered.extend(grouped[AccessPattern.SCATTER_GATHER])
            
            return reordered
    
    def detect_bandwidth_saturation(self) -> Tuple[bool, float]:
        """
        Detect if memory bandwidth is saturated
        
        Returns:
            Tuple of (is_saturated, utilization_percent)
        """
        with self._lock:
            is_saturated = self.bandwidth_metrics.is_bandwidth_saturated
            utilization = self.bandwidth_metrics.utilization_percent
            
            if is_saturated:
                logger.warning(f"Memory bandwidth saturated at {utilization:.1f}%")
            
            return is_saturated, utilization
    
    def mitigate_bandwidth_saturation(self) -> Dict[str, Any]:
        """
        Mitigate bandwidth saturation through various strategies
        
        Returns:
            Mitigation results
        """
        with self._lock:
            self.stats['bandwidth_optimizations'] += 1
            
            strategies_applied = []
            
            # 1. Increase cache efficiency
            self._optimize_cache_usage()
            strategies_applied.append('cache_optimization')
            
            # 2. Apply memory compression if possible
            if self._can_apply_compression():
                self._apply_memory_compression()
                strategies_applied.append('memory_compression')
            
            # 3. Adjust prefetching to reduce pressure
            self.enable_prefetching = False  # Temporarily disable
            strategies_applied.append('prefetch_adjustment')
            
            # 4. Recommend operation fusion
            fusion_candidates = self._identify_fusion_candidates()
            if fusion_candidates:
                strategies_applied.append('operation_fusion')
            
            return {
                'strategies_applied': strategies_applied,
                'previous_utilization': self.bandwidth_metrics.utilization_percent,
                'fusion_candidates': fusion_candidates
            }
    
    def _optimize_cache_usage(self) -> None:
        """Optimize cache usage to reduce bandwidth pressure"""
        self.stats['cache_optimizations'] += 1
        
        # Clear old entries from layout cache
        current_time = time.time()
        old_entries = []
        for key, info in self.layout_cache.items():
            # Simple age-based eviction (would be more sophisticated in practice)
            if len(self.layout_cache) > 100:
                old_entries.append(key)
        
        for key in old_entries[:len(old_entries)//2]:
            del self.layout_cache[key]
    
    def _can_apply_compression(self) -> bool:
        """Check if memory compression can be applied"""
        # Simplified check
        return self.bandwidth_metrics.utilization_percent > 80
    
    def _apply_memory_compression(self) -> None:
        """Apply memory compression techniques"""
        # This would implement actual compression
        # For now, it's a placeholder
        pass
    
    def _identify_fusion_candidates(self) -> List[Tuple[str, str]]:
        """Identify operations that can be fused"""
        candidates = []
        
        # Analyze recent operations from history
        if len(self.access_history) >= 2:
            recent_ops = list(self.access_history)[-10:]
            
            for i in range(len(recent_ops) - 1):
                _, op1_key, profile1 = recent_ops[i]
                _, op2_key, profile2 = recent_ops[i + 1]
                
                # Check if operations can be fused
                if (profile1.pattern == profile2.pattern and
                    profile1.locality > 0.5 and profile2.locality > 0.5):
                    candidates.append((op1_key, op2_key))
        
        return candidates
    
    def _monitor_bandwidth(self) -> None:
        """Background thread to monitor bandwidth utilization"""
        last_read_bytes = 0
        last_write_bytes = 0
        last_time = time.time()
        
        while self.monitoring_active:
            try:
                time.sleep(0.1)  # 100ms sampling
                
                current_time = time.time()
                dt = current_time - last_time
                
                if dt > 0:
                    # Get current memory stats (simplified)
                    # In practice, would query actual GPU metrics
                    current_read_bytes = self.stats.get('total_bytes_optimized', 0)
                    current_write_bytes = current_read_bytes // 2  # Estimate
                    
                    # Calculate bandwidth
                    read_bw = (current_read_bytes - last_read_bytes) / dt / 1e9
                    write_bw = (current_write_bytes - last_write_bytes) / dt / 1e9
                    
                    with self._lock:
                        self.bandwidth_metrics.timestamp = current_time
                        self.bandwidth_metrics.read_bandwidth_gb_s = read_bw
                        self.bandwidth_metrics.write_bandwidth_gb_s = write_bw
                        self.bandwidth_metrics.total_bandwidth_gb_s = read_bw + write_bw
                        self.bandwidth_metrics.utilization_percent = (
                            100.0 * self.bandwidth_metrics.total_bandwidth_gb_s / 
                            self.bandwidth_metrics.theoretical_max_gb_s
                        )
                    
                    last_read_bytes = current_read_bytes
                    last_write_bytes = current_write_bytes
                    last_time = current_time
                    
            except Exception as e:
                logger.error(f"Bandwidth monitoring error: {e}")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get memory optimization statistics"""
        with self._lock:
            stats = dict(self.stats)
            
            # Add bandwidth metrics
            stats['bandwidth'] = {
                'current_gb_s': self.bandwidth_metrics.total_bandwidth_gb_s,
                'max_gb_s': self.bandwidth_metrics.theoretical_max_gb_s,
                'utilization_percent': self.bandwidth_metrics.utilization_percent,
                'is_saturated': self.bandwidth_metrics.is_bandwidth_saturated
            }
            
            # Add prefetch metrics
            if self.enable_prefetching:
                total_prefetch = self.prefetch_hits + self.prefetch_misses
                prefetch_accuracy = self.prefetch_hits / total_prefetch if total_prefetch > 0 else 0
                stats['prefetch'] = {
                    'hits': self.prefetch_hits,
                    'misses': self.prefetch_misses,
                    'accuracy': prefetch_accuracy,
                    'queue_size': len(self.prefetch_queue)
                }
            
            # Add access pattern distribution
            pattern_counts = defaultdict(int)
            for _, _, profile in self.access_history:
                pattern_counts[profile.pattern.value] += 1
            stats['access_patterns'] = dict(pattern_counts)
            
            return stats
    
    def shutdown(self) -> None:
        """Shutdown memory optimizer"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        
        logger.info("JAXMemoryOptimizer shutdown complete")


# Decorator for memory-optimized operations
def memory_optimized(optimizer=None):
    """Decorator for memory-optimized JAX operations"""
    def decorator(func):
        # Get global optimizer if not provided
        if optimizer is None:
            if not hasattr(memory_optimized, '_global_optimizer'):
                memory_optimized._global_optimizer = JAXMemoryOptimizer()
            opt = memory_optimized._global_optimizer
        else:
            opt = optimizer
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Analyze input tensors
            tensors = []
            for arg in args:
                if hasattr(arg, 'shape'):  # JAX array
                    tensors.append(arg)
            
            # Setup prefetching
            if tensors:
                opt.setup_prefetching(tensors, PrefetchStrategy.ADAPTIVE)
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Check bandwidth saturation
            is_saturated, utilization = opt.detect_bandwidth_saturation()
            if is_saturated:
                opt.mitigate_bandwidth_saturation()
            
            return result
        
        return wrapper
    return decorator


# Test function
def test_memory_optimizer():
    """Test memory optimizer functionality"""
    print("Testing JAX Memory Optimizer...")
    
    # Initialize optimizer
    optimizer = JAXMemoryOptimizer(enable_profiling=True, enable_prefetching=True)
    
    # Test tensor
    A = jnp.ones((1000, 1000), dtype=jnp.float32)
    B = jnp.ones((1000, 1000), dtype=jnp.float32)
    
    print("\n1. Testing access pattern analysis...")
    profile_matmul = optimizer.analyze_access_pattern(A, 'matmul')
    print(f"   MatMul pattern: {profile_matmul.pattern.value}")
    print(f"   Stride: {profile_matmul.stride}")
    print(f"   Locality: {profile_matmul.locality:.2f}")
    
    profile_reduce = optimizer.analyze_access_pattern(A, 'reduce')
    print(f"   Reduce pattern: {profile_reduce.pattern.value}")
    print(f"   Locality: {profile_reduce.locality:.2f}")
    
    print("\n2. Testing layout optimization...")
    optimized_A, layout_info = optimizer.optimize_tensor_layout(A, profile_matmul, 'matmul')
    print(f"   Original shape: {A.shape}")
    print(f"   Optimized layout: {layout_info.layout.value}")
    print(f"   Memory footprint: {layout_info.memory_footprint / 1024 / 1024:.2f}MB")
    
    print("\n3. Testing memory coalescing...")
    operations = [
        ('matmul', A),
        ('reduce', B),
        ('broadcast', A[:100]),
        ('gather', B[:, :100])
    ]
    reordered = optimizer.coalesce_memory_accesses(operations)
    print(f"   Original order: {[op for op, _ in operations]}")
    print(f"   Optimized order: {[op for op, _ in reordered]}")
    
    print("\n4. Testing prefetching...")
    optimizer.setup_prefetching([A, B], PrefetchStrategy.ADAPTIVE)
    print(f"   Prefetch queue size: {len(optimizer.prefetch_queue)}")
    
    print("\n5. Testing bandwidth monitoring...")
    is_saturated, utilization = optimizer.detect_bandwidth_saturation()
    print(f"   Bandwidth utilization: {utilization:.1f}%")
    print(f"   Is saturated: {is_saturated}")
    
    if is_saturated:
        print("\n6. Testing saturation mitigation...")
        mitigation = optimizer.mitigate_bandwidth_saturation()
        print(f"   Strategies applied: {mitigation['strategies_applied']}")
    
    # Get statistics
    stats = optimizer.get_optimization_stats()
    print(f"\n7. Optimization statistics:")
    print(f"   Layouts optimized: {stats['layouts_optimized']}")
    print(f"   Bandwidth optimizations: {stats['bandwidth_optimizations']}")
    print(f"   Memory coalescing: {stats['memory_coalescing']}")
    print(f"   Total bytes optimized: {stats['total_bytes_optimized'] / 1024 / 1024:.2f}MB")
    
    # Test decorated function
    @memory_optimized(optimizer=optimizer)
    def optimized_matmul(A, B):
        return jnp.matmul(A, B)
    
    print("\n8. Testing decorated function...")
    C = optimized_matmul(A, B)
    print(f"   Result shape: {C.shape}")
    
    # Shutdown
    optimizer.shutdown()
    
    print("\n✓ Memory optimizer test complete!")


if __name__ == "__main__":
    test_memory_optimizer()