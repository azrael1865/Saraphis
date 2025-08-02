"""
General GPU memory optimization for Saraphis Brain system.
Provides system-wide GPU memory management separate from compression-specific GPU optimization.
NO FALLBACKS - HARD FAILURES ONLY
"""

import asyncio
import gc
import logging
import threading
import time
import warnings
from collections import defaultdict, deque
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Set, Union, Callable
import numpy as np

try:
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GPUOptimizationStrategy(Enum):
    """GPU memory optimization strategies"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MEMORY_FIRST = "memory_first"
    PERFORMANCE_FIRST = "performance_first"


class GPUMemoryState(Enum):
    """GPU memory allocation states"""
    ALLOCATED = "allocated"
    CACHED = "cached"
    FREE = "free"
    RESERVED = "reserved"


@dataclass
class GPUMemorySnapshot:
    """Snapshot of GPU memory state"""
    timestamp: float
    device_id: int
    total_memory: int
    allocated_memory: int
    cached_memory: int
    free_memory: int
    reserved_memory: int
    memory_utilization: float
    active_streams: int
    pending_operations: int


@dataclass
class GPUAllocationRecord:
    """Record of GPU memory allocation"""
    allocation_id: str
    tensor_name: str
    device_id: int
    size_bytes: int
    dtype: str
    shape: Tuple[int, ...]
    allocated_at: float
    last_accessed: float
    access_count: int = 0
    stream_id: Optional[int] = None
    is_pinned: bool = False
    allocation_stack: List[str] = field(default_factory=list)


@dataclass
class StreamOptimizationResult:
    """Result of CUDA stream optimization"""
    stream_id: int
    operations_optimized: int
    memory_freed: int
    synchronization_reduced: int
    performance_improvement: float
    optimization_time: float


@dataclass
class GPUFragmentationReport:
    """Report on GPU memory fragmentation"""
    device_id: int
    total_fragmentation_mb: float
    fragmentation_percentage: float
    largest_free_block_mb: float
    free_block_count: int
    recommended_defragmentation: bool
    estimated_recovery_mb: float


class GeneralGPUMemoryOptimizer:
    """
    General GPU memory optimization for system-wide GPU resource management.
    Separate from compression-specific GPU optimization to avoid conflicts.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize general GPU memory optimizer"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, dict):
            raise TypeError(f"Config must be dict or None, got {type(config)}")
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GPU memory optimization")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for GPU memory optimization")
        
        self.config = config or {}
        self.logger = logging.getLogger('GeneralGPUMemoryOptimizer')
        
        # Configuration parameters
        self.optimization_strategy = GPUOptimizationStrategy(
            self.config.get('optimization_strategy', 'balanced')
        )
        self.enable_monitoring = self.config.get('enable_monitoring', True)
        self.enable_auto_optimization = self.config.get('enable_auto_optimization', True)
        self.memory_threshold_percent = self.config.get('memory_threshold_percent', 80.0)
        self.fragmentation_threshold = self.config.get('fragmentation_threshold', 0.3)
        self.snapshot_interval = self.config.get('snapshot_interval', 30.0)
        self.stream_pool_size = self.config.get('stream_pool_size', 8)
        self.enable_memory_pooling = self.config.get('enable_memory_pooling', True)
        self.max_cached_memory_mb = self.config.get('max_cached_memory_mb', 2048)
        
        # Validate configuration
        self._validate_config()
        
        # GPU device management
        self.device_count = torch.cuda.device_count()
        self.current_device = torch.cuda.current_device()
        self.device_properties = {}
        self.device_capabilities = {}
        
        # Initialize device information
        self._initialize_device_info()
        
        # Memory tracking
        self.memory_snapshots: Dict[int, deque] = {
            device_id: deque(maxlen=1000) 
            for device_id in range(self.device_count)
        }
        self.allocation_records: Dict[str, GPUAllocationRecord] = {}
        self.fragmentation_reports: Dict[int, GPUFragmentationReport] = {}
        
        # CUDA stream management
        self.stream_pools: Dict[int, List[torch.cuda.Stream]] = {}
        self.active_streams: Dict[int, Set[torch.cuda.Stream]] = {}
        self.stream_usage_stats: Dict[int, Dict[int, int]] = {}
        
        # Initialize stream pools
        self._initialize_stream_pools()
        
        # Memory pool management
        self.memory_pools: Dict[int, Dict[str, torch.Tensor]] = {}
        self.pool_usage_stats: Dict[int, Dict[str, int]] = {}
        
        if self.enable_memory_pooling:
            self._initialize_memory_pools()
        
        # Thread safety
        self._optimization_lock = threading.RLock()
        self._allocation_lock = threading.RLock()
        self._stream_lock = threading.RLock()
        
        # Background monitoring
        self._monitoring_enabled = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Performance metrics
        self.optimization_metrics = {
            'total_allocations': 0,
            'total_deallocations': 0,
            'memory_optimizations': 0,
            'stream_optimizations': 0,
            'fragmentation_reductions': 0,
            'total_memory_saved_mb': 0.0,
            'average_utilization': 0.0,
            'peak_memory_usage_mb': 0.0,
            'optimization_success_rate': 0.0
        }
        
        self.is_initialized = True
        self.logger.info(f"GeneralGPUMemoryOptimizer initialized with {self.device_count} GPU devices")
    
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        if not isinstance(self.memory_threshold_percent, (int, float)):
            raise TypeError(f"memory_threshold_percent must be numeric")
        if not (0.0 <= self.memory_threshold_percent <= 100.0):
            raise ValueError(f"memory_threshold_percent must be between 0 and 100")
        
        if not isinstance(self.fragmentation_threshold, (int, float)):
            raise TypeError(f"fragmentation_threshold must be numeric")
        if not (0.0 <= self.fragmentation_threshold <= 1.0):
            raise ValueError(f"fragmentation_threshold must be between 0.0 and 1.0")
        
        if not isinstance(self.snapshot_interval, (int, float)) or self.snapshot_interval <= 0:
            raise ValueError(f"snapshot_interval must be positive number")
        
        if not isinstance(self.stream_pool_size, int) or self.stream_pool_size <= 0:
            raise ValueError(f"stream_pool_size must be positive int")
        if self.stream_pool_size > 64:
            raise ValueError(f"stream_pool_size cannot exceed 64")
        
        if not isinstance(self.max_cached_memory_mb, (int, float)) or self.max_cached_memory_mb <= 0:
            raise ValueError(f"max_cached_memory_mb must be positive number")
    
    def _initialize_device_info(self) -> None:
        """Initialize GPU device information"""
        for device_id in range(self.device_count):
            props = torch.cuda.get_device_properties(device_id)
            self.device_properties[device_id] = {
                'name': props.name,
                'total_memory': props.total_memory,
                'major': props.major,
                'minor': props.minor,
                'multi_processor_count': props.multi_processor_count,
                'max_threads_per_block': props.max_threads_per_block,
                'max_threads_per_multiprocessor': props.max_threads_per_multiprocessor
            }
            
            # Calculate device capabilities
            compute_capability = props.major * 10 + props.minor
            self.device_capabilities[device_id] = {
                'compute_capability': compute_capability,
                'supports_tensor_cores': compute_capability >= 70,
                'supports_mixed_precision': compute_capability >= 70,
                'memory_bandwidth_gbps': self._estimate_memory_bandwidth(props),
                'theoretical_performance_tflops': self._estimate_peak_performance(props)
            }
        
        self.logger.info(f"Initialized {self.device_count} GPU devices")
    
    def _estimate_memory_bandwidth(self, props) -> float:
        """Estimate memory bandwidth based on device properties"""
        # Simplified estimation based on compute capability
        if props.major >= 8:  # Ampere and newer
            return 1000.0  # ~1TB/s for high-end cards
        elif props.major >= 7:  # Turing/Volta
            return 600.0   # ~600GB/s
        elif props.major >= 6:  # Pascal
            return 400.0   # ~400GB/s
        else:
            return 200.0   # Older architectures
    
    def _estimate_peak_performance(self, props) -> float:
        """Estimate peak performance in TFLOPS"""
        # Very simplified estimation
        base_performance = props.multi_processor_count * 1.0  # Base TFLOPS per SM
        if props.major >= 8:
            return base_performance * 2.0  # Ampere improvements
        elif props.major >= 7:
            return base_performance * 1.5  # Turing/Volta improvements
        else:
            return base_performance
    
    def _initialize_stream_pools(self) -> None:
        """Initialize CUDA stream pools for each device"""
        for device_id in range(self.device_count):
            with torch.cuda.device(device_id):
                streams = [torch.cuda.Stream() for _ in range(self.stream_pool_size)]
                self.stream_pools[device_id] = streams
                self.active_streams[device_id] = set()
                self.stream_usage_stats[device_id] = {i: 0 for i in range(self.stream_pool_size)}
        
        self.logger.info(f"Initialized stream pools with {self.stream_pool_size} streams per device")
    
    def _initialize_memory_pools(self) -> None:
        """Initialize memory pools for each device"""
        for device_id in range(self.device_count):
            self.memory_pools[device_id] = {}
            self.pool_usage_stats[device_id] = {}
        
        self.logger.info("Memory pooling enabled for all devices")
    
    def allocate_gpu_memory(self, shape: Tuple[int, ...], dtype: torch.dtype, 
                           name: str, device_id: Optional[int] = None) -> torch.Tensor:
        """
        Allocate GPU memory with tracking and optimization.
        
        Args:
            shape: Tensor shape to allocate
            dtype: Tensor data type
            name: Name identifier for the allocation
            device_id: Target device ID (uses current device if None)
            
        Returns:
            Allocated tensor
            
        Raises:
            ValueError: If shape or name is invalid
            RuntimeError: If allocation fails
        """
        # NO FALLBACKS - HARD FAILURES ONLY
        if not isinstance(shape, (tuple, list)):
            raise TypeError(f"shape must be tuple or list, got {type(shape)}")
        if not shape:
            raise ValueError("shape cannot be empty")
        if not all(isinstance(dim, int) and dim > 0 for dim in shape):
            raise ValueError("All shape dimensions must be positive integers")
        
        if not isinstance(name, str):
            raise TypeError(f"name must be str, got {type(name)}")
        if not name.strip():
            raise ValueError("name cannot be empty")
        
        if device_id is not None:
            if not isinstance(device_id, int):
                raise TypeError(f"device_id must be int or None, got {type(device_id)}")
            if not (0 <= device_id < self.device_count):
                raise ValueError(f"device_id {device_id} out of range [0, {self.device_count})")
        else:
            device_id = self.current_device
        
        # Calculate allocation size
        element_size = torch.tensor([], dtype=dtype).element_size()
        total_elements = np.prod(shape)
        size_bytes = total_elements * element_size
        
        # Check if allocation would exceed memory limits
        self._check_allocation_feasibility(device_id, size_bytes)
        
        allocation_start = time.time()
        
        try:
            # Try memory pool first if enabled
            if self.enable_memory_pooling:
                tensor = self._try_pool_allocation(device_id, shape, dtype, name)
                if tensor is not None:
                    self.logger.debug(f"Allocated '{name}' from memory pool")
                    return tensor
            
            # Direct allocation
            with torch.cuda.device(device_id):
                tensor = torch.empty(shape, dtype=dtype, device=f'cuda:{device_id}')
            
            # Record allocation
            allocation_id = f"{name}_{device_id}_{time.time()}"
            with self._allocation_lock:
                self.allocation_records[allocation_id] = GPUAllocationRecord(
                    allocation_id=allocation_id,
                    tensor_name=name,
                    device_id=device_id,
                    size_bytes=size_bytes,
                    dtype=str(dtype),
                    shape=shape,
                    allocated_at=time.time(),
                    last_accessed=time.time(),
                    allocation_stack=self._get_allocation_stack()
                )
                
                self.optimization_metrics['total_allocations'] += 1
            
            allocation_time = time.time() - allocation_start
            
            self.logger.debug(
                f"Allocated GPU memory '{name}': "
                f"{size_bytes / 1024 / 1024:.2f}MB on device {device_id} "
                f"in {allocation_time:.4f}s"
            )
            
            return tensor
            
        except torch.cuda.OutOfMemoryError as e:
            # Try optimization and retry once
            self.logger.warning(f"GPU OOM during allocation, attempting optimization")
            self.optimize_gpu_memory()
            
            try:
                with torch.cuda.device(device_id):
                    tensor = torch.empty(shape, dtype=dtype, device=f'cuda:{device_id}')
                
                self.logger.info(f"Allocation succeeded after optimization")
                return tensor
                
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError(
                    f"Failed to allocate {size_bytes / 1024 / 1024:.2f}MB on GPU {device_id} "
                    f"even after optimization: {e}"
                )
        
        except Exception as e:
            raise RuntimeError(f"Unexpected error during GPU allocation: {e}")
    
    def free_gpu_memory(self, tensor_name: str) -> bool:
        """
        Free GPU memory allocation by name.
        
        Args:
            tensor_name: Name of tensor allocation to free
            
        Returns:
            True if memory was freed, False if allocation not found
        """
        if not isinstance(tensor_name, str):
            raise TypeError(f"tensor_name must be str, got {type(tensor_name)}")
        
        freed = False
        
        with self._allocation_lock:
            # Find allocation records matching the name
            records_to_remove = []
            for allocation_id, record in self.allocation_records.items():
                if record.tensor_name == tensor_name:
                    records_to_remove.append(allocation_id)
            
            # Remove records and update metrics
            for allocation_id in records_to_remove:
                del self.allocation_records[allocation_id]
                freed = True
                self.optimization_metrics['total_deallocations'] += 1
        
        if freed:
            # Force garbage collection to actually free memory
            gc.collect()
            torch.cuda.empty_cache()
            
            self.logger.debug(f"Freed GPU memory allocation '{tensor_name}'")
        
        return freed
    
    def _try_pool_allocation(self, device_id: int, shape: Tuple[int, ...], 
                           dtype: torch.dtype, name: str) -> Optional[torch.Tensor]:
        """Try to allocate from memory pool"""
        pool_key = f"{shape}_{dtype}"
        
        if device_id in self.memory_pools:
            pool = self.memory_pools[device_id]
            if pool_key in pool:
                tensor = pool.pop(pool_key)
                
                # Update pool usage stats
                if device_id in self.pool_usage_stats:
                    self.pool_usage_stats[device_id][pool_key] = \
                        self.pool_usage_stats[device_id].get(pool_key, 0) + 1
                
                return tensor
        
        return None
    
    def _check_allocation_feasibility(self, device_id: int, size_bytes: int) -> None:
        """Check if allocation is feasible on target device"""
        with torch.cuda.device(device_id):
            free_memory = torch.cuda.get_device_properties(device_id).total_memory - torch.cuda.memory_allocated()
            
            if size_bytes > free_memory:
                # Try to free some memory first
                self.optimize_gpu_memory()
                
                # Check again
                free_memory = torch.cuda.get_device_properties(device_id).total_memory - torch.cuda.memory_allocated()
                if size_bytes > free_memory:
                    raise RuntimeError(
                        f"Allocation of {size_bytes / 1024 / 1024:.2f}MB would exceed "
                        f"available memory ({free_memory / 1024 / 1024:.2f}MB) on device {device_id}"
                    )
    
    def optimize_gpu_memory(self) -> Dict[str, Any]:
        """
        Optimize GPU memory usage across all devices.
        
        Returns:
            Dictionary containing optimization results
        """
        optimization_start = time.time()
        optimizations_applied = []
        total_memory_freed = 0
        
        with self._optimization_lock:
            for device_id in range(self.device_count):
                device_optimizations = self._optimize_device_memory(device_id)
                optimizations_applied.extend(device_optimizations)
                
                # Calculate memory freed for this device
                device_memory_freed = sum(
                    opt.get('memory_freed', 0) for opt in device_optimizations
                )
                total_memory_freed += device_memory_freed
        
        # Optimize CUDA streams
        stream_optimizations = self._optimize_cuda_streams()
        optimizations_applied.extend(stream_optimizations)
        
        # Update metrics
        self.optimization_metrics['memory_optimizations'] += 1
        self.optimization_metrics['total_memory_saved_mb'] += total_memory_freed / 1024 / 1024
        
        optimization_time = time.time() - optimization_start
        
        result = {
            'status': 'optimization_completed',
            'devices_optimized': self.device_count,
            'optimizations_applied': len(optimizations_applied),
            'total_memory_freed_mb': total_memory_freed / 1024 / 1024,
            'optimization_time': optimization_time,
            'optimization_details': optimizations_applied
        }
        
        self.logger.info(
            f"GPU memory optimization completed: freed {total_memory_freed / 1024 / 1024:.1f}MB "
            f"across {self.device_count} devices in {optimization_time:.2f}s"
        )
        
        return result
    
    def _optimize_device_memory(self, device_id: int) -> List[Dict[str, Any]]:
        """Optimize memory usage for a specific device"""
        optimizations = []
        
        with torch.cuda.device(device_id):
            # Get current memory state
            initial_allocated = torch.cuda.memory_allocated()
            initial_cached = torch.cuda.memory_reserved()
            
            # Clear PyTorch cache
            torch.cuda.empty_cache()
            
            after_cache_clear = torch.cuda.memory_allocated()
            cache_freed = initial_cached - torch.cuda.memory_reserved()
            
            if cache_freed > 0:
                optimizations.append({
                    'type': 'cache_clear',
                    'device_id': device_id,
                    'memory_freed': cache_freed,
                    'description': f"Cleared {cache_freed / 1024 / 1024:.1f}MB from PyTorch cache"
                })
            
            # Analyze memory fragmentation
            fragmentation_report = self._analyze_memory_fragmentation(device_id)
            if fragmentation_report.recommended_defragmentation:
                defrag_result = self._defragment_device_memory(device_id)
                if defrag_result['memory_freed'] > 0:
                    optimizations.append(defrag_result)
            
            # Clean up old allocations based on strategy
            cleanup_result = self._cleanup_stale_allocations(device_id)
            if cleanup_result['allocations_freed'] > 0:
                optimizations.append(cleanup_result)
            
            # Memory pool optimization
            if self.enable_memory_pooling:
                pool_result = self._optimize_memory_pools(device_id)
                if pool_result['pools_optimized'] > 0:
                    optimizations.append(pool_result)
        
        return optimizations
    
    def _analyze_memory_fragmentation(self, device_id: int) -> GPUFragmentationReport:
        """Analyze memory fragmentation for a device"""
        with torch.cuda.device(device_id):
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            cached_memory = torch.cuda.memory_reserved()
            free_memory = total_memory - allocated_memory
            
            # Simplified fragmentation analysis
            # In a real implementation, this would require more sophisticated analysis
            fragmentation_estimate = min(0.5, (cached_memory - allocated_memory) / total_memory)
            
            report = GPUFragmentationReport(
                device_id=device_id,
                total_fragmentation_mb=fragmentation_estimate * total_memory / 1024 / 1024,
                fragmentation_percentage=fragmentation_estimate * 100,
                largest_free_block_mb=free_memory / 1024 / 1024,
                free_block_count=1,  # Simplified
                recommended_defragmentation=fragmentation_estimate > self.fragmentation_threshold,
                estimated_recovery_mb=fragmentation_estimate * cached_memory / 1024 / 1024
            )
            
            self.fragmentation_reports[device_id] = report
            return report
    
    def _defragment_device_memory(self, device_id: int) -> Dict[str, Any]:
        """Attempt to defragment memory on a device"""
        with torch.cuda.device(device_id):
            initial_cached = torch.cuda.memory_reserved()
            
            # Force memory defragmentation by clearing cache and triggering GC
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            
            final_cached = torch.cuda.memory_reserved()
            memory_freed = initial_cached - final_cached
            
            return {
                'type': 'defragmentation',
                'device_id': device_id,
                'memory_freed': memory_freed,
                'description': f"Defragmented {memory_freed / 1024 / 1024:.1f}MB on device {device_id}"
            }
    
    def _cleanup_stale_allocations(self, device_id: int) -> Dict[str, Any]:
        """Clean up stale memory allocations"""
        current_time = time.time()
        stale_threshold = 3600  # 1 hour
        
        allocations_freed = 0
        memory_freed = 0
        
        with self._allocation_lock:
            stale_allocations = []
            for allocation_id, record in self.allocation_records.items():
                if (record.device_id == device_id and 
                    current_time - record.last_accessed > stale_threshold):
                    stale_allocations.append(allocation_id)
            
            for allocation_id in stale_allocations:
                record = self.allocation_records[allocation_id]
                memory_freed += record.size_bytes
                del self.allocation_records[allocation_id]
                allocations_freed += 1
        
        return {
            'type': 'stale_cleanup',
            'device_id': device_id,
            'allocations_freed': allocations_freed,
            'memory_freed': memory_freed,
            'description': f"Cleaned up {allocations_freed} stale allocations ({memory_freed / 1024 / 1024:.1f}MB)"
        }
    
    def _optimize_memory_pools(self, device_id: int) -> Dict[str, Any]:
        """Optimize memory pools for a device"""
        pools_optimized = 0
        memory_freed = 0
        
        if device_id in self.memory_pools:
            pool = self.memory_pools[device_id]
            usage_stats = self.pool_usage_stats.get(device_id, {})
            
            # Remove unused pool entries
            unused_keys = []
            for pool_key, tensor in pool.items():
                if usage_stats.get(pool_key, 0) == 0:
                    unused_keys.append(pool_key)
                    memory_freed += tensor.numel() * tensor.element_size()
            
            for key in unused_keys:
                del pool[key]
                pools_optimized += 1
        
        return {
            'type': 'pool_optimization',
            'device_id': device_id,
            'pools_optimized': pools_optimized,
            'memory_freed': memory_freed,
            'description': f"Optimized {pools_optimized} memory pools"
        }
    
    def _optimize_cuda_streams(self) -> List[Dict[str, Any]]:
        """Optimize CUDA stream usage"""
        optimizations = []
        
        with self._stream_lock:
            for device_id in range(self.device_count):
                if device_id in self.stream_pools:
                    # Synchronize unused streams
                    unused_streams = []
                    for i, stream in enumerate(self.stream_pools[device_id]):
                        if stream not in self.active_streams[device_id]:
                            stream.synchronize()
                            unused_streams.append(i)
                    
                    if unused_streams:
                        optimizations.append({
                            'type': 'stream_synchronization',
                            'device_id': device_id,
                            'streams_synchronized': len(unused_streams),
                            'description': f"Synchronized {len(unused_streams)} unused streams on device {device_id}"
                        })
        
        return optimizations
    
    def get_gpu_memory_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive GPU memory usage report.
        
        Returns:
            Dictionary containing detailed GPU memory analysis
        """
        device_reports = {}
        
        for device_id in range(self.device_count):
            with torch.cuda.device(device_id):
                # Current memory state
                props = torch.cuda.get_device_properties(device_id)
                allocated = torch.cuda.memory_allocated()
                cached = torch.cuda.memory_reserved()
                total = props.total_memory
                free = total - allocated
                
                # Allocation summary
                device_allocations = []
                total_tracked_memory = 0
                
                with self._allocation_lock:
                    for record in self.allocation_records.values():
                        if record.device_id == device_id:
                            device_allocations.append({
                                'name': record.tensor_name,
                                'size_mb': record.size_bytes / 1024 / 1024,
                                'shape': record.shape,
                                'dtype': record.dtype,
                                'age_hours': (time.time() - record.allocated_at) / 3600,
                                'access_count': record.access_count
                            })
                            total_tracked_memory += record.size_bytes
                
                # Stream usage
                stream_info = {
                    'total_streams': len(self.stream_pools.get(device_id, [])),
                    'active_streams': len(self.active_streams.get(device_id, set())),
                    'stream_usage_stats': self.stream_usage_stats.get(device_id, {})
                }
                
                # Memory pool info
                pool_info = {}
                if self.enable_memory_pooling and device_id in self.memory_pools:
                    pool = self.memory_pools[device_id]
                    pool_usage = self.pool_usage_stats.get(device_id, {})
                    
                    pool_info = {
                        'total_pools': len(pool),
                        'pool_usage_stats': pool_usage,
                        'total_pooled_memory_mb': sum(
                            tensor.numel() * tensor.element_size() 
                            for tensor in pool.values()
                        ) / 1024 / 1024
                    }
                
                device_reports[device_id] = {
                    'device_name': props.name,
                    'memory_state': {
                        'total_mb': total / 1024 / 1024,
                        'allocated_mb': allocated / 1024 / 1024,
                        'cached_mb': cached / 1024 / 1024,
                        'free_mb': free / 1024 / 1024,
                        'utilization_percent': (allocated / total) * 100
                    },
                    'tracked_allocations': {
                        'count': len(device_allocations),
                        'total_mb': total_tracked_memory / 1024 / 1024,
                        'allocations': device_allocations[:10]  # Top 10
                    },
                    'stream_info': stream_info,
                    'memory_pools': pool_info,
                    'fragmentation_report': (
                        self.fragmentation_reports[device_id].__dict__ 
                        if device_id in self.fragmentation_reports 
                        else None
                    )
                }
        
        # System-wide statistics
        total_memory = sum(
            torch.cuda.get_device_properties(i).total_memory 
            for i in range(self.device_count)
        )
        total_allocated = sum(
            torch.cuda.memory_allocated(i) 
            for i in range(self.device_count)
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_summary': {
                'total_devices': self.device_count,
                'total_memory_gb': total_memory / 1024 / 1024 / 1024,
                'total_allocated_gb': total_allocated / 1024 / 1024 / 1024,
                'overall_utilization_percent': (total_allocated / total_memory) * 100,
                'optimization_strategy': self.optimization_strategy.value
            },
            'device_reports': device_reports,
            'optimization_metrics': self.optimization_metrics.copy(),
            'configuration': {
                'memory_threshold_percent': self.memory_threshold_percent,
                'fragmentation_threshold': self.fragmentation_threshold,
                'stream_pool_size': self.stream_pool_size,
                'memory_pooling_enabled': self.enable_memory_pooling,
                'auto_optimization_enabled': self.enable_auto_optimization
            }
        }
    
    def manage_cuda_streams(self, device_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Manage CUDA streams for optimal GPU utilization.
        
        Args:
            device_id: Target device ID (manages all devices if None)
            
        Returns:
            Dictionary containing stream management results
        """
        if device_id is not None:
            if not isinstance(device_id, int):
                raise TypeError(f"device_id must be int or None, got {type(device_id)}")
            if not (0 <= device_id < self.device_count):
                raise ValueError(f"device_id {device_id} out of range")
            devices = [device_id]
        else:
            devices = list(range(self.device_count))
        
        management_results = {}
        
        for dev_id in devices:
            with self._stream_lock:
                streams = self.stream_pools.get(dev_id, [])
                active = self.active_streams.get(dev_id, set())
                
                # Synchronize all streams
                sync_count = 0
                for stream in streams:
                    if stream not in active:
                        stream.synchronize()
                        sync_count += 1
                
                # Reset usage statistics
                if dev_id in self.stream_usage_stats:
                    self.stream_usage_stats[dev_id] = {i: 0 for i in range(len(streams))}
                
                management_results[dev_id] = {
                    'total_streams': len(streams),
                    'active_streams': len(active),
                    'synchronized_streams': sync_count,
                    'optimization_time': time.time()
                }
        
        self.logger.info(f"Managed CUDA streams for {len(devices)} devices")
        
        return {
            'devices_managed': len(devices),
            'device_results': management_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_allocation_stack(self) -> List[str]:
        """Get current allocation stack trace"""
        try:
            import traceback
            stack = traceback.extract_stack()
            return [f"{frame.filename}:{frame.lineno} in {frame.name}" for frame in stack[-5:]]
        except Exception:
            return []
    
    def start_monitoring(self) -> None:
        """Start background GPU memory monitoring"""
        if self._monitoring_enabled:
            return
        
        self._monitoring_enabled = True
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="GPUMemoryMonitor",
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info("Background GPU memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background GPU memory monitoring"""
        if not self._monitoring_enabled:
            return
        
        self._monitoring_enabled = False
        self._stop_monitoring.set()
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Background GPU memory monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while not self._stop_monitoring.wait(self.snapshot_interval):
            try:
                # Take memory snapshots for all devices
                for device_id in range(self.device_count):
                    snapshot = self._take_memory_snapshot(device_id)
                    self.memory_snapshots[device_id].append(snapshot)
                
                # Check for critical memory conditions
                self._check_critical_gpu_memory()
                
                # Auto-optimization if enabled
                if self.enable_auto_optimization:
                    self._check_auto_optimization_triggers()
                
            except Exception as e:
                self.logger.error(f"Error in GPU memory monitoring loop: {e}")
    
    def _take_memory_snapshot(self, device_id: int) -> GPUMemorySnapshot:
        """Take memory snapshot for a device"""
        with torch.cuda.device(device_id):
            props = torch.cuda.get_device_properties(device_id)
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            total = props.total_memory
            
            return GPUMemorySnapshot(
                timestamp=time.time(),
                device_id=device_id,
                total_memory=total,
                allocated_memory=allocated,
                cached_memory=cached,
                free_memory=total - allocated,
                reserved_memory=cached,
                memory_utilization=(allocated / total) * 100,
                active_streams=len(self.active_streams.get(device_id, set())),
                pending_operations=0  # Would require deeper GPU state analysis
            )
    
    def _check_critical_gpu_memory(self) -> None:
        """Check for critical GPU memory conditions"""
        for device_id in range(self.device_count):
            with torch.cuda.device(device_id):
                props = torch.cuda.get_device_properties(device_id)
                allocated = torch.cuda.memory_allocated() 
                utilization = (allocated / props.total_memory) * 100
                
                if utilization > 95:
                    self.logger.critical(
                        f"Critical GPU memory usage on device {device_id}: {utilization:.1f}%"
                    )
                elif utilization > self.memory_threshold_percent:
                    self.logger.warning(
                        f"High GPU memory usage on device {device_id}: {utilization:.1f}%"
                    )
    
    def _check_auto_optimization_triggers(self) -> None:
        """Check if auto-optimization should be triggered"""
        for device_id in range(self.device_count):
            with torch.cuda.device(device_id):
                props = torch.cuda.get_device_properties(device_id)
                allocated = torch.cuda.memory_allocated()
                utilization = (allocated / props.total_memory) * 100
                
                if utilization > self.memory_threshold_percent:
                    self.logger.info(f"Auto-optimization triggered for device {device_id}")
                    self._optimize_device_memory(device_id)