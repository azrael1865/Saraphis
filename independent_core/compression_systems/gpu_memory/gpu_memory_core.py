"""
GPU Memory Optimizer - Comprehensive GPU memory optimization and CUDA stream management.
Handles GPU memory allocation, stream optimization, kernel management, and coordinated resource optimization.
"""

import torch
import torch.cuda
import asyncio
import threading
import time
import logging
import queue
import weakref
from typing import Dict, Any, Optional, List, Tuple, Set, Union, Callable
from collections import deque, defaultdict, OrderedDict
from dataclasses import dataclass, field
from contextlib import contextmanager, asynccontextmanager
import functools
import warnings
import numpy as np
import gc


@dataclass
class CUDAStream:
    """CUDA stream wrapper with metadata"""
    stream: torch.cuda.Stream
    name: str
    priority: int
    created_time: float
    last_used: float
    usage_count: int = 0
    is_active: bool = True
    associated_tensors: Set[int] = field(default_factory=set)


@dataclass
class GPUMemoryBlock:
    """GPU memory block for allocation tracking"""
    ptr: int
    size: int
    dtype: torch.dtype
    device: torch.device
    allocated_time: float
    last_accessed: float
    access_count: int = 0
    is_free: bool = False
    stream_id: Optional[int] = None
    # AutoSwap fields
    priority_score: float = 0.0
    doa_score: float = 0.0
    swap_candidate: bool = False
    tensor_id: Optional[str] = None


@dataclass
class GPUOptimizationResult:
    """Result of a GPU optimization operation"""
    success: bool
    memory_freed_mb: float
    streams_optimized: int
    fragmentation_reduced: float
    optimization_time_ms: float
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class KernelConfig:
    """Kernel configuration for optimal GPU execution"""
    block_size: Tuple[int, int, int]
    grid_size: Tuple[int, int, int]
    shared_memory: int
    registers_per_thread: int
    occupancy: float
    recommended_stream: Optional[torch.cuda.Stream] = None


class GPUMemoryOptimizer:
    """
    Comprehensive GPU memory optimization and CUDA stream management.
    Handles memory allocation, stream optimization, kernel management, and coordinated resource optimization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the GPU memory optimizer.
        
        Args:
            config: Configuration dictionary for GPU optimization
        """
        if config is None:
            config = {}
        
        self.config = config
        self.logger = logging.getLogger('GPUMemoryOptimizer')
        
        # Check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available - cannot initialize GPU memory optimizer")
        
        # Get device properties
        self.device_count = torch.cuda.device_count()
        self.current_device = torch.cuda.current_device()
        self.device_properties = {}
        for i in range(self.device_count):
            self.device_properties[i] = torch.cuda.get_device_properties(i)
        
        # Configuration parameters
        self.max_streams_per_device = config.get('max_streams_per_device', 32)
        self.history_size = config.get('history_size', 10000)
        self.optimization_interval = config.get('optimization_interval', 60.0)
        self.memory_threshold = config.get('memory_threshold', 0.85)
        self.fragmentation_threshold = config.get('fragmentation_threshold', 0.3)
        self.metrics_size = config.get('metrics_size', 5000)
        
        # Stream management
        self.streams: Dict[int, Dict[str, CUDAStream]] = defaultdict(dict)
        self.stream_pools: Dict[int, List[torch.cuda.Stream]] = defaultdict(list)
        self.stream_lock = threading.RLock()
        
        # Memory management
        self.memory_blocks: Dict[int, List[GPUMemoryBlock]] = defaultdict(list)
        self.memory_pools: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.memory_lock = threading.RLock()
        
        # Kernel optimization
        self.kernel_configs: Dict[str, KernelConfig] = {}
        self.kernel_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Performance tracking
        self.allocation_history = deque(maxlen=self.history_size)
        self.optimization_history = deque(maxlen=self.history_size)
        self.performance_metrics = {
            'total_allocations': 0,
            'total_deallocations': 0,
            'peak_memory_mb': 0.0,
            'current_memory_mb': 0.0,
            'optimization_count': 0,
            'stream_creations': 0,
            'kernel_optimizations': 0,
            'memory_saved_mb': 0.0,
            'fragmentation_events': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Component references for integration
        self.brain_core = None
        self.training_manager = None
        self.compression_systems = {}
        self.smart_pool = None  # SmartPool integration
        self.autoswap_manager = None  # AutoSwap integration
        
        # Background optimization
        self.optimization_active = True
        
        # Initialize SmartPool if enabled
        if config.get('enable_smart_pool', True):
            self._initialize_smart_pool()
            
        # Initialize AutoSwap if enabled
        if config.get('enable_autoswap', True):
            self._initialize_autoswap()
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        
        self.logger.info(f"GPU Memory Optimizer initialized with {self.device_count} devices")
    
    def set_component_references(self, references: Dict[str, Any]) -> None:
        """Set references to other system components for coordinated optimization."""
        self.brain_core = references.get('brain_core')
        self.training_manager = references.get('training_manager')
        self.compression_systems.update(references.get('compression_systems', {}))
        self.logger.debug("Component references updated")
    
    @contextmanager
    def optimize_memory_context(self):
        """
        Context manager for optimized memory operations.
        Automatically optimizes memory allocation and cleanup.
        """
        # Pre-optimization
        start_time = time.perf_counter()
        initial_memory = self._get_current_memory_usage()
        
        try:
            # Optimize memory before operation
            self._optimize_memory_pre_operation()
            yield
        finally:
            # Post-optimization cleanup
            self._optimize_memory_post_operation()
            
            # Track optimization effectiveness
            final_memory = self._get_current_memory_usage()
            optimization_time = (time.perf_counter() - start_time) * 1000
            memory_saved = max(0, initial_memory - final_memory)
            
            self.performance_metrics['memory_saved_mb'] += memory_saved
            self.optimization_history.append({
                'timestamp': time.time(),
                'memory_saved_mb': memory_saved,
                'optimization_time_ms': optimization_time,
                'context': 'memory_context'
            })
    
    def _optimize_memory_pre_operation(self) -> None:
        """Optimize memory before operation."""
        # Clear unused cached memory
        torch.cuda.empty_cache()
        
        # Synchronize streams to complete pending operations
        for device_id in range(self.device_count):
            with torch.cuda.device(device_id):
                torch.cuda.synchronize()
        
        # Optimize stream allocation
        self._optimize_stream_allocation()
    
    def _optimize_memory_post_operation(self) -> None:
        """Cleanup and optimize memory after operation."""
        # Clear cache again
        torch.cuda.empty_cache()
        
        # Clean up unused streams
        self._cleanup_unused_streams()
        
        # Update memory statistics
        self._update_memory_statistics()
    
    async def optimize_memory_allocation(self) -> GPUOptimizationResult:
        """
        Asynchronously optimize memory allocation across all devices.
        
        Returns:
            Optimization result with details
        """
        start_time = time.perf_counter()
        total_memory_freed = 0.0
        total_streams_optimized = 0
        total_fragmentation_reduced = 0.0
        recommendations = []
        warnings = []
        
        try:
            # Use SmartPool optimization if available
            if self.smart_pool is not None:
                smartpool_result = self.smart_pool.optimize_memory()
                total_memory_freed += smartpool_result.memory_freed_mb
                total_streams_optimized += smartpool_result.streams_optimized
                total_fragmentation_reduced += smartpool_result.fragmentation_reduced
                recommendations.extend(smartpool_result.recommendations)
                warnings.extend(smartpool_result.warnings)
                
                # Check if 13.3% target achieved
                if self.smart_pool.statistics.target_achieved:
                    recommendations.append("✓ SmartPool achieved 13.3% fragmentation reduction target")
            else:
                # Fallback to standard optimization
                # Optimize each device
                for device_id in range(self.device_count):
                    device_result = await self._optimize_device_memory_async(device_id)
                    total_memory_freed += device_result.memory_freed_mb
                    total_streams_optimized += device_result.streams_optimized
                    total_fragmentation_reduced += device_result.fragmentation_reduced
                    recommendations.extend(device_result.recommendations)
                    warnings.extend(device_result.warnings)
            
            # Global optimization
            await self._global_memory_optimization()
            
            optimization_time = (time.perf_counter() - start_time) * 1000
            
            result = GPUOptimizationResult(
                success=True,
                memory_freed_mb=total_memory_freed,
                streams_optimized=total_streams_optimized,
                fragmentation_reduced=total_fragmentation_reduced,
                optimization_time_ms=optimization_time,
                recommendations=recommendations,
                warnings=warnings
            )
            
            # Update metrics
            self.performance_metrics['optimization_count'] += 1
            self.performance_metrics['memory_saved_mb'] += total_memory_freed
            
            return result
            
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return GPUOptimizationResult(
                success=False,
                memory_freed_mb=0.0,
                streams_optimized=0,
                fragmentation_reduced=0.0,
                optimization_time_ms=(time.perf_counter() - start_time) * 1000,
                warnings=[f"Optimization failed: {str(e)}"]
            )
    
    async def _optimize_device_memory_async(self, device_id: int) -> GPUOptimizationResult:
        """Asynchronously optimize memory for a specific device."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._optimize_device_memory, device_id)
    
    async def _global_memory_optimization(self) -> None:
        """Perform global memory optimization across all devices."""
        # Balance memory usage across devices
        memory_usage = []
        for device_id in range(self.device_count):
            with torch.cuda.device(device_id):
                allocated = torch.cuda.memory_allocated(device_id)
                reserved = torch.cuda.memory_reserved(device_id)
                memory_usage.append((device_id, allocated, reserved))
        
        # Sort by memory usage
        memory_usage.sort(key=lambda x: x[1], reverse=True)
        
        # Log memory distribution
        self.logger.debug(f"Memory distribution: {memory_usage}")
    
    async def cleanup_streams(self) -> None:
        """Asynchronously cleanup and optimize CUDA streams."""
        try:
            # Synchronize all streams
            for device_id in range(self.device_count):
                await self._synchronize_device_streams_async(device_id)
            
            # Clean up unused streams
            with self.stream_lock:
                for device_id in self.streams:
                    streams_to_remove = []
                    for stream_name, cuda_stream in self.streams[device_id].items():
                        if (time.time() - cuda_stream.last_used) > 300:  # 5 minutes
                            streams_to_remove.append(stream_name)
                    
                    for stream_name in streams_to_remove:
                        del self.streams[device_id][stream_name]
            
            self.logger.debug("Stream cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Stream cleanup failed: {e}")
    
    async def _synchronize_device_streams_async(self, device_id: int) -> None:
        """Asynchronously synchronize streams for a specific device."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.synchronize_streams, device_id)
    
    def prepare_for_computation(self, estimated_memory_mb: float = 100.0, 
                               computation_type: str = 'general') -> Dict[str, Any]:
        """
        Prepare GPU resources for computation.
        
        Args:
            estimated_memory_mb: Estimated memory requirement in MB
            computation_type: Type of computation for optimization
            
        Returns:
            Preparation result with optimization details
        """
        try:
            start_time = time.perf_counter()
            
            # Check memory availability
            available_memory = self._get_available_memory()
            if estimated_memory_mb > available_memory:
                self.logger.warning(f"Requested memory ({estimated_memory_mb}MB) exceeds available ({available_memory}MB)")
            
            # Use SmartPool if available for optimized allocation
            if self.smart_pool is not None:
                from .smart_pool import AllocationRequest
                request = AllocationRequest(
                    size=int(estimated_memory_mb * 1024 * 1024),  # Convert to bytes
                    device_id=0,  # Default to first device
                    hint_lifetime=300.0 if computation_type == 'training' else 60.0,
                    hint_access_pattern=computation_type
                )
                allocation_result = self.smart_pool.allocate_memory(request)
                if allocation_result:
                    self.logger.debug(f"SmartPool allocation successful for {estimated_memory_mb}MB")
            
            # Pre-allocate if needed
            if computation_type == 'training':
                self._prepare_for_training_computation(estimated_memory_mb)
            elif computation_type == 'inference':
                self._prepare_for_inference_computation(estimated_memory_mb)
            else:
                self._prepare_for_general_computation(estimated_memory_mb)
            
            preparation_time = (time.perf_counter() - start_time) * 1000
            
            return {
                'success': True,
                'preparation_time_ms': preparation_time,
                'available_memory_mb': available_memory,
                'estimated_memory_mb': estimated_memory_mb,
                'computation_type': computation_type
            }
            
        except Exception as e:
            self.logger.error(f"Computation preparation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'available_memory_mb': 0.0
            }
    
    def cleanup_after_computation(self, computation_id: str = 'unknown') -> Dict[str, Any]:
        """
        Cleanup GPU resources after computation.
        
        Args:
            computation_id: Identifier for the computation
            
        Returns:
            Cleanup result with details
        """
        try:
            start_time = time.perf_counter()
            initial_memory = self._get_current_memory_usage()
            
            # Synchronize all operations
            torch.cuda.synchronize()
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Clean up temporary allocations
            self._cleanup_temporary_allocations()
            
            final_memory = self._get_current_memory_usage()
            cleanup_time = (time.perf_counter() - start_time) * 1000
            memory_freed = max(0, initial_memory - final_memory)
            
            return {
                'success': True,
                'cleanup_time_ms': cleanup_time,
                'memory_freed_mb': memory_freed,
                'computation_id': computation_id
            }
            
        except Exception as e:
            self.logger.error(f"Computation cleanup failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'computation_id': computation_id
            }
    
    def optimize_for_operation(self, operation_type: str, expected_memory_mb: float = 50.0,
                              priority: str = 'normal') -> Dict[str, Any]:
        """
        Optimize GPU resources for a specific operation.
        
        Args:
            operation_type: Type of operation (e.g., 'compression', 'neural_inference')
            expected_memory_mb: Expected memory usage in MB
            priority: Operation priority ('high', 'normal', 'low')
            
        Returns:
            Optimization result
        """
        try:
            start_time = time.perf_counter()
            
            # Get optimal configuration for operation
            optimal_config = self._get_optimal_config_for_operation(operation_type, expected_memory_mb)
            
            # Apply optimization based on priority
            if priority == 'high':
                self._apply_high_priority_optimization(optimal_config)
            elif priority == 'low':
                self._apply_low_priority_optimization(optimal_config)
            else:
                self._apply_normal_priority_optimization(optimal_config)
            
            optimization_time = (time.perf_counter() - start_time) * 1000
            
            return {
                'success': True,
                'operation_type': operation_type,
                'optimization_time_ms': optimization_time,
                'priority': priority,
                'config_applied': optimal_config
            }
            
        except Exception as e:
            self.logger.error(f"Operation optimization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'operation_type': operation_type
            }
    
    def prepare_for_training(self, estimated_memory_mb: float = 500.0, 
                           batch_size: int = 32, model_size_mb: float = 100.0) -> Dict[str, Any]:
        """
        Prepare GPU resources specifically for training.
        
        Args:
            estimated_memory_mb: Estimated memory for training
            batch_size: Training batch size
            model_size_mb: Model size in MB
            
        Returns:
            Training preparation result
        """
        try:
            start_time = time.perf_counter()
            
            # Calculate total memory requirement
            total_memory_needed = estimated_memory_mb + model_size_mb * 3  # Model + gradients + optimizer states
            
            # Check if we have enough memory
            available_memory = self._get_available_memory()
            if total_memory_needed > available_memory * 0.9:  # Leave 10% buffer
                self.logger.warning(f"Training memory requirement ({total_memory_needed}MB) near limit")
            
            # Optimize for training workload
            self._optimize_for_training_workload(batch_size, model_size_mb)
            
            # Pre-allocate training streams
            training_streams = self._allocate_training_streams()
            
            preparation_time = (time.perf_counter() - start_time) * 1000
            
            return {
                'success': True,
                'preparation_time_ms': preparation_time,
                'total_memory_needed_mb': total_memory_needed,
                'available_memory_mb': available_memory,
                'training_streams': len(training_streams),
                'batch_size': batch_size
            }
            
        except Exception as e:
            self.logger.error(f"Training preparation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def cleanup_after_training(self, session_id: str) -> Dict[str, Any]:
        """
        Cleanup GPU resources after training session.
        
        Args:
            session_id: Training session identifier
            
        Returns:
            Cleanup result
        """
        try:
            start_time = time.perf_counter()
            initial_memory = self._get_current_memory_usage()
            
            # Synchronize all training streams
            self._synchronize_training_streams()
            
            # Clear gradients and optimizer states
            torch.cuda.empty_cache()
            
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            
            # Clean up training-specific allocations
            self._cleanup_training_allocations(session_id)
            
            final_memory = self._get_current_memory_usage()
            cleanup_time = (time.perf_counter() - start_time) * 1000
            memory_freed = max(0, initial_memory - final_memory)
            
            return {
                'success': True,
                'cleanup_time_ms': cleanup_time,
                'memory_freed_mb': memory_freed,
                'session_id': session_id
            }
            
        except Exception as e:
            self.logger.error(f"Training cleanup failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id
            }
    
    @contextmanager
    def get_stream(self, device_id: Optional[int] = None, 
                   priority: str = 'normal') -> torch.cuda.Stream:
        """
        Get a CUDA stream from the pool.
        
        Args:
            device_id: Device ID (None for current device)
            priority: Stream priority ('high', 'normal', 'low')
            
        Yields:
            CUDA stream
        """
        if device_id is None:
            device_id = self.current_device
        
        if device_id < 0 or device_id >= self.device_count:
            raise ValueError(f"Invalid device ID: {device_id}")
        
        if priority not in ['high', 'normal', 'low']:
            priority = 'normal'
        
        priority_map = {'high': -1, 'normal': 0, 'low': 1}
        priority_int = priority_map[priority]
        
        with self.stream_lock:
            # Try to get existing stream
            device_streams = self.streams[device_id]
            
            # Find available stream with matching priority
            available_stream = None
            for stream_name, cuda_stream in device_streams.items():
                if cuda_stream.priority == priority_int and cuda_stream.is_active:
                    available_stream = cuda_stream
                    break
            
            # Create new stream if needed
            if available_stream is None and len(device_streams) < self.max_streams_per_device:
                stream_name = f"stream_{device_id}_{len(device_streams)}_{priority}"
                torch_stream = torch.cuda.Stream(device=device_id, priority=priority_int)
                
                available_stream = CUDAStream(
                    stream=torch_stream,
                    name=stream_name,
                    priority=priority_int,
                    created_time=time.time(),
                    last_used=time.time()
                )
                
                device_streams[stream_name] = available_stream
                self.performance_metrics['stream_creations'] += 1
            
            if available_stream is None:
                # Use default stream if no streams available
                available_stream = CUDAStream(
                    stream=torch.cuda.default_stream(device_id),
                    name="default",
                    priority=priority_int,
                    created_time=time.time(),
                    last_used=time.time()
                )
        
        try:
            # Update usage statistics
            available_stream.last_used = time.time()
            available_stream.usage_count += 1
            
            yield available_stream.stream
            
        finally:
            # Stream cleanup happens in background thread
            pass
    
    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                       device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """
        Allocate a tensor with optimal memory management.
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
            device: Target device
            
        Returns:
            Allocated tensor
        """
        if device is None:
            device = torch.device(f'cuda:{self.current_device}')
        elif isinstance(device, str):
            device = torch.device(device)
        
        try:
            # Calculate memory requirement
            element_size = torch.tensor([], dtype=dtype).element_size()
            total_elements = np.prod(shape)
            memory_mb = (total_elements * element_size) / (1024 * 1024)
            
            # Check memory availability
            if memory_mb > 100:  # Large allocation
                available_memory = self._get_available_memory()
                if memory_mb > available_memory * 0.8:
                    self.logger.warning(f"Large allocation ({memory_mb:.1f}MB) may cause memory pressure")
            
            # Allocate tensor
            tensor = torch.empty(shape, dtype=dtype, device=device)
            
            # Track allocation
            self._track_allocation(tensor, memory_mb)
            
            return tensor
            
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(f"GPU out of memory during tensor allocation: {e}")
            # Try to free some memory and retry
            torch.cuda.empty_cache()
            try:
                tensor = torch.empty(shape, dtype=dtype, device=device)
                self._track_allocation(tensor, memory_mb)
                return tensor
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError(f"Cannot allocate tensor of size {shape} ({memory_mb:.1f}MB)")
    
    def deallocate_tensor(self, tensor: torch.Tensor) -> None:
        """
        Deallocate a tensor and update tracking.
        
        Args:
            tensor: Tensor to deallocate
        """
        try:
            tensor_id = id(tensor)
            
            # Update tracking
            self._track_deallocation(tensor_id)
            
            # Delete tensor
            del tensor
            
        except Exception as e:
            self.logger.warning(f"Error during tensor deallocation: {e}")
    
    def optimize_allocation(self, memory_usage: float) -> Dict[str, Any]:
        """
        Optimize memory allocation based on current usage.
        
        Args:
            memory_usage: Current memory usage ratio (0.0 to 1.0)
            
        Returns:
            Optimization result
        """
        try:
            start_time = time.perf_counter()
            optimizations_applied = []
            
            if memory_usage > self.memory_threshold:
                # High memory pressure - aggressive optimization
                torch.cuda.empty_cache()
                optimizations_applied.append("cache_clear")
                
                # Clean up old allocations
                self._cleanup_old_allocations()
                optimizations_applied.append("allocation_cleanup")
                
                # Defragment memory if possible
                self._attempt_memory_defragmentation()
                optimizations_applied.append("defragmentation")
                
            elif memory_usage > 0.6:
                # Moderate memory pressure - gentle optimization
                torch.cuda.empty_cache()
                optimizations_applied.append("cache_clear")
                
            # Check for fragmentation
            fragmentation = self._calculate_memory_fragmentation()
            if fragmentation > self.fragmentation_threshold:
                self._attempt_memory_defragmentation()
                optimizations_applied.append("fragmentation_fix")
                self.performance_metrics['fragmentation_events'] += 1
            
            optimization_time = (time.perf_counter() - start_time) * 1000
            
            return {
                'success': True,
                'optimizations_applied': optimizations_applied,
                'memory_usage': memory_usage,
                'fragmentation': fragmentation,
                'optimization_time_ms': optimization_time
            }
            
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'memory_usage': memory_usage
            }
    
    def optimize_streams(self) -> Dict[str, Any]:
        """Optimize CUDA stream usage across all devices."""
        try:
            start_time = time.perf_counter()
            optimized_streams = 0
            
            with self.stream_lock:
                for device_id in range(self.device_count):
                    device_streams = self.streams[device_id]
                    
                    # Remove unused streams
                    current_time = time.time()
                    streams_to_remove = []
                    
                    for stream_name, cuda_stream in device_streams.items():
                        if (current_time - cuda_stream.last_used) > 600:  # 10 minutes
                            streams_to_remove.append(stream_name)
                    
                    for stream_name in streams_to_remove:
                        del device_streams[stream_name]
                        optimized_streams += 1
                    
                    # Rebalance stream priorities
                    self._rebalance_stream_priorities(device_id)
            
            optimization_time = (time.perf_counter() - start_time) * 1000
            
            return {
                'success': True,
                'streams_optimized': optimized_streams,
                'optimization_time_ms': optimization_time
            }
            
        except Exception as e:
            self.logger.error(f"Stream optimization failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def synchronize_streams(self, device_id: Optional[int] = None) -> None:
        """
        Synchronize CUDA streams for a device.
        
        Args:
            device_id: Device ID (None for all devices)
        """
        try:
            if device_id is not None:
                devices = [device_id]
            else:
                devices = list(range(self.device_count))
            
            for dev_id in devices:
                with torch.cuda.device(dev_id):
                    # Synchronize all streams for this device
                    device_streams = self.streams[dev_id]
                    for cuda_stream in device_streams.values():
                        if cuda_stream.is_active:
                            cuda_stream.stream.synchronize()
                    
                    # Also synchronize default stream
                    torch.cuda.synchronize(dev_id)
            
        except Exception as e:
            self.logger.error(f"Stream synchronization failed: {e}")
    
    def get_memory_info(self, device_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get comprehensive memory information.
        
        Args:
            device_id: Device ID (None for current device)
            
        Returns:
            Memory information dictionary
        """
        if device_id is None:
            device_id = self.current_device
        
        try:
            with torch.cuda.device(device_id):
                allocated = torch.cuda.memory_allocated(device_id)
                reserved = torch.cuda.memory_reserved(device_id)
                max_allocated = torch.cuda.max_memory_allocated(device_id)
                max_reserved = torch.cuda.max_memory_reserved(device_id)
                
                total_memory = self.device_properties[device_id].total_memory
                free_memory = total_memory - reserved
                
                return {
                    'device_id': device_id,
                    'allocated_mb': allocated / (1024 * 1024),
                    'reserved_mb': reserved / (1024 * 1024),
                    'free_mb': free_memory / (1024 * 1024),
                    'total_mb': total_memory / (1024 * 1024),
                    'max_allocated_mb': max_allocated / (1024 * 1024),
                    'max_reserved_mb': max_reserved / (1024 * 1024),
                    'utilization': reserved / total_memory,
                    'fragmentation': self._calculate_memory_fragmentation(device_id)
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get memory info: {e}")
            return {
                'device_id': device_id,
                'error': str(e)
            }
    
    def _optimize_device_memory(self, device_id: int) -> GPUOptimizationResult:
        """Optimize memory for a specific device."""
        try:
            start_time = time.perf_counter()
            initial_memory = self._get_device_memory_usage(device_id)
            
            with torch.cuda.device(device_id):
                # Clear cache
                torch.cuda.empty_cache()
                
                # Synchronize device
                torch.cuda.synchronize(device_id)
                
                # Clean up streams
                streams_cleaned = self._cleanup_device_streams(device_id)
                
                # Attempt defragmentation
                fragmentation_before = self._calculate_memory_fragmentation(device_id)
                self._attempt_memory_defragmentation(device_id)
                fragmentation_after = self._calculate_memory_fragmentation(device_id)
                
            final_memory = self._get_device_memory_usage(device_id)
            optimization_time = (time.perf_counter() - start_time) * 1000
            
            return GPUOptimizationResult(
                success=True,
                memory_freed_mb=max(0, initial_memory - final_memory),
                streams_optimized=streams_cleaned,
                fragmentation_reduced=max(0, fragmentation_before - fragmentation_after),
                optimization_time_ms=optimization_time,
                recommendations=[f"Device {device_id} optimized successfully"]
            )
            
        except Exception as e:
            return GPUOptimizationResult(
                success=False,
                memory_freed_mb=0.0,
                streams_optimized=0,
                fragmentation_reduced=0.0,
                optimization_time_ms=0.0,
                warnings=[f"Device {device_id} optimization failed: {str(e)}"]
            )
    
    def _optimize_stream_allocation(self) -> None:
        """Optimize stream allocation across devices."""
        with self.stream_lock:
            for device_id in range(self.device_count):
                device_streams = self.streams[device_id]
                
                # Count streams by priority
                priority_counts = defaultdict(int)
                for cuda_stream in device_streams.values():
                    priority_counts[cuda_stream.priority] += 1
                
                # Rebalance if needed
                total_streams = len(device_streams)
                if total_streams > self.max_streams_per_device * 0.8:
                    self._rebalance_device_streams(device_id)
    
    def _cleanup_unused_streams(self) -> None:
        """Clean up unused streams across all devices."""
        current_time = time.time()
        cleanup_threshold = 300  # 5 minutes
        
        with self.stream_lock:
            for device_id in self.streams:
                streams_to_remove = []
                for stream_name, cuda_stream in self.streams[device_id].items():
                    if (current_time - cuda_stream.last_used) > cleanup_threshold:
                        streams_to_remove.append(stream_name)
                
                for stream_name in streams_to_remove:
                    del self.streams[device_id][stream_name]
    
    def _update_memory_statistics(self) -> None:
        """Update memory usage statistics."""
        total_allocated = 0.0
        peak_memory = 0.0
        
        for device_id in range(self.device_count):
            try:
                allocated = torch.cuda.memory_allocated(device_id) / (1024 * 1024)
                max_allocated = torch.cuda.max_memory_allocated(device_id) / (1024 * 1024)
                
                total_allocated += allocated
                peak_memory = max(peak_memory, max_allocated)
                
            except Exception:
                continue
        
        self.performance_metrics['current_memory_mb'] = total_allocated
        self.performance_metrics['peak_memory_mb'] = max(
            self.performance_metrics['peak_memory_mb'], 
            peak_memory
        )
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB across all devices."""
        total_memory = 0.0
        for device_id in range(self.device_count):
            try:
                allocated = torch.cuda.memory_allocated(device_id)
                total_memory += allocated / (1024 * 1024)
            except Exception:
                continue
        return total_memory
    
    def _get_device_memory_usage(self, device_id: int) -> float:
        """Get memory usage for a specific device in MB."""
        try:
            allocated = torch.cuda.memory_allocated(device_id)
            return allocated / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _get_available_memory(self) -> float:
        """Get available memory in MB across all devices."""
        total_available = 0.0
        for device_id in range(self.device_count):
            try:
                total = self.device_properties[device_id].total_memory
                reserved = torch.cuda.memory_reserved(device_id)
                available = (total - reserved) / (1024 * 1024)
                total_available += available
            except Exception:
                continue
        return total_available
    
    def _calculate_memory_fragmentation(self, device_id: Optional[int] = None) -> float:
        """Calculate memory fragmentation for a device."""
        try:
            if device_id is None:
                # Calculate average fragmentation across all devices
                total_fragmentation = 0.0
                device_count = 0
                
                for dev_id in range(self.device_count):
                    frag = self._calculate_memory_fragmentation(dev_id)
                    if frag >= 0:
                        total_fragmentation += frag
                        device_count += 1
                
                return total_fragmentation / max(device_count, 1)
            
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            
            if reserved == 0:
                return 0.0
            
            # Simple fragmentation metric: ratio of unused reserved memory
            fragmentation = (reserved - allocated) / reserved
            return max(0.0, min(1.0, fragmentation))
            
        except Exception:
            return 0.0
    
    def _attempt_memory_defragmentation(self, device_id: Optional[int] = None) -> None:
        """Attempt to defragment GPU memory."""
        try:
            if device_id is not None:
                devices = [device_id]
            else:
                devices = list(range(self.device_count))
            
            for dev_id in devices:
                with torch.cuda.device(dev_id):
                    # Synchronize to complete all operations
                    torch.cuda.synchronize(dev_id)
                    
                    # Clear cache multiple times to force cleanup
                    for _ in range(3):
                        torch.cuda.empty_cache()
                        time.sleep(0.01)  # Small delay
                    
                    # Reset memory stats to get fresh start
                    torch.cuda.reset_peak_memory_stats(dev_id)
            
        except Exception as e:
            self.logger.warning(f"Memory defragmentation failed: {e}")
    
    def _cleanup_old_allocations(self) -> None:
        """Clean up old memory allocations."""
        current_time = time.time()
        cleanup_threshold = 3600  # 1 hour
        
        with self.memory_lock:
            for device_id in self.memory_blocks:
                blocks_to_remove = []
                for i, block in enumerate(self.memory_blocks[device_id]):
                    if (current_time - block.last_accessed) > cleanup_threshold and block.is_free:
                        blocks_to_remove.append(i)
                
                # Remove old blocks (in reverse order to maintain indices)
                for i in reversed(blocks_to_remove):
                    del self.memory_blocks[device_id][i]
    
    def register_tensor_for_autoswap(self, tensor: torch.Tensor, tensor_id: Optional[str] = None,
                                    priority: Optional[str] = None) -> str:
        """Register a tensor for AutoSwap management"""
        if not self.autoswap_manager:
            raise RuntimeError("AutoSwap not initialized")
        
        if tensor_id is None:
            tensor_id = f"tensor_{id(tensor)}_{int(time.time() * 1000)}"
        
        # Import SwapPriority
        from .doa_scorer import SwapPriority
        
        # Map priority string to enum
        priority_map = {
            'critical': SwapPriority.CRITICAL,
            'high': SwapPriority.HIGH,
            'medium': SwapPriority.MEDIUM,
            'low': SwapPriority.LOW,
            'idle': SwapPriority.IDLE
        }
        swap_priority = priority_map.get(priority, SwapPriority.MEDIUM)
        
        # Register with AutoSwap
        self.autoswap_manager.register_tensor(tensor_id, tensor, swap_priority)
        
        # Track in memory blocks
        with self.memory_lock:
            device_id = tensor.device.index if tensor.is_cuda else -1
            if device_id >= 0:
                block = GPUMemoryBlock(
                    ptr=tensor.data_ptr(),
                    size=tensor.numel() * tensor.element_size(),
                    dtype=tensor.dtype,
                    device=tensor.device,
                    allocated_time=time.time(),
                    last_accessed=time.time(),
                    tensor_id=tensor_id,
                    priority_score=swap_priority.value / 4.0,  # Normalize to 0-1
                    swap_candidate=swap_priority.value >= SwapPriority.MEDIUM.value
                )
                self.memory_blocks[device_id].append(block)
        
        return tensor_id
    
    def record_tensor_access(self, tensor_id: str, offset: int = 0, length: Optional[int] = None) -> None:
        """Record tensor access for DOA tracking"""
        if self.autoswap_manager:
            self.autoswap_manager.record_access(tensor_id, offset, length)
            
            # Update memory block access time
            with self.memory_lock:
                for device_blocks in self.memory_blocks.values():
                    for block in device_blocks:
                        if block.tensor_id == tensor_id:
                            block.last_accessed = time.time()
                            block.access_count += 1
                            break
    
    def handle_memory_pressure(self, required_bytes: int, device_id: int = 0) -> bool:
        """Handle memory pressure using AutoSwap"""
        if not self.autoswap_manager:
            self.logger.warning("AutoSwap not available for memory pressure handling")
            return False
        
        return self.autoswap_manager.handle_memory_pressure(required_bytes, device_id)
    
    def swap_in_tensor(self, tensor_id: str, device_id: int = 0) -> Optional[torch.Tensor]:
        """Swap a tensor back to GPU"""
        if not self.autoswap_manager:
            raise RuntimeError("AutoSwap not initialized")
        
        return self.autoswap_manager.swap_in_tensor(tensor_id, device_id)
    
    def _track_allocation(self, tensor: torch.Tensor, memory_mb: float) -> None:
        """Track tensor allocation."""
        try:
            self.performance_metrics['total_allocations'] += 1
            
            # Add to allocation history
            self.allocation_history.append({
                'timestamp': time.time(),
                'tensor_id': id(tensor),
                'memory_mb': memory_mb,
                'device': str(tensor.device),
                'shape': tuple(tensor.shape)
            })
            
        except Exception as e:
            self.logger.warning(f"Allocation tracking failed: {e}")
    
    def _track_deallocation(self, tensor_id: int) -> None:
        """Track tensor deallocation."""
        try:
            self.performance_metrics['total_deallocations'] += 1
            
        except Exception as e:
            self.logger.warning(f"Deallocation tracking failed: {e}")
    
    def _cleanup_device_streams(self, device_id: int) -> int:
        """Clean up streams for a specific device."""
        streams_cleaned = 0
        current_time = time.time()
        
        with self.stream_lock:
            if device_id in self.streams:
                streams_to_remove = []
                for stream_name, cuda_stream in self.streams[device_id].items():
                    if (current_time - cuda_stream.last_used) > 300:  # 5 minutes
                        streams_to_remove.append(stream_name)
                
                for stream_name in streams_to_remove:
                    del self.streams[device_id][stream_name]
                    streams_cleaned += 1
        
        return streams_cleaned
    
    def _rebalance_stream_priorities(self, device_id: int) -> None:
        """Rebalance stream priorities for a device."""
        with self.stream_lock:
            if device_id not in self.streams:
                return
            
            device_streams = self.streams[device_id]
            
            # Sort streams by usage count
            sorted_streams = sorted(device_streams.values(), key=lambda s: s.usage_count, reverse=True)
            
            # Reassign priorities based on usage
            for i, cuda_stream in enumerate(sorted_streams):
                if i < len(sorted_streams) // 3:
                    cuda_stream.priority = -1  # High priority
                elif i < 2 * len(sorted_streams) // 3:
                    cuda_stream.priority = 0   # Normal priority
                else:
                    cuda_stream.priority = 1   # Low priority
    
    def _rebalance_device_streams(self, device_id: int) -> None:
        """Rebalance streams for a device when approaching limits."""
        with self.stream_lock:
            device_streams = self.streams[device_id]
            
            # Remove least used streams if over limit
            if len(device_streams) > self.max_streams_per_device:
                sorted_streams = sorted(
                    device_streams.items(),
                    key=lambda x: (x[1].last_used, x[1].usage_count)
                )
                
                streams_to_remove = len(device_streams) - self.max_streams_per_device
                for i in range(streams_to_remove):
                    stream_name = sorted_streams[i][0]
                    del device_streams[stream_name]
    
    def _prepare_for_training_computation(self, estimated_memory_mb: float) -> None:
        """Prepare GPU for training computation."""
        # Clear cache aggressively
        torch.cuda.empty_cache()
        
        # Ensure we have enough streams for training
        self._ensure_training_streams()
        
        # Pre-warm memory allocator if needed
        if estimated_memory_mb > 1000:  # Large training job
            self._prewarm_memory_allocator(estimated_memory_mb)
    
    def _prepare_for_inference_computation(self, estimated_memory_mb: float) -> None:
        """Prepare GPU for inference computation."""
        # Light cleanup for inference
        torch.cuda.empty_cache()
        
        # Ensure inference streams are available
        self._ensure_inference_streams()
    
    def _prepare_for_general_computation(self, estimated_memory_mb: float) -> None:
        """Prepare GPU for general computation."""
        # Standard preparation
        torch.cuda.empty_cache()
        
        # Ensure basic streams are available
        self._ensure_basic_streams()
    
    def _cleanup_temporary_allocations(self) -> None:
        """Clean up temporary GPU allocations."""
        # This would clean up any temporary allocations we're tracking
        # For now, just clear the cache
        torch.cuda.empty_cache()
    
    def _get_optimal_config_for_operation(self, operation_type: str, memory_mb: float) -> Dict[str, Any]:
        """Get optimal configuration for operation type."""
        base_config = {
            'memory_mb': memory_mb,
            'streams_needed': 1,
            'priority': 'normal'
        }
        
        if operation_type == 'compression':
            base_config.update({
                'streams_needed': 2,
                'priority': 'high',
                'cache_strategy': 'aggressive'
            })
        elif operation_type == 'neural_inference':
            base_config.update({
                'streams_needed': 1,
                'priority': 'normal',
                'cache_strategy': 'moderate'
            })
        elif operation_type == 'training':
            base_config.update({
                'streams_needed': 4,
                'priority': 'high',
                'cache_strategy': 'conservative'
            })
        
        return base_config
    
    def _apply_high_priority_optimization(self, config: Dict[str, Any]) -> None:
        """Apply high priority optimization."""
        # Aggressive optimization for high priority operations
        torch.cuda.empty_cache()
        self._ensure_streams(config['streams_needed'], priority='high')
    
    def _apply_normal_priority_optimization(self, config: Dict[str, Any]) -> None:
        """Apply normal priority optimization."""
        # Standard optimization
        self._ensure_streams(config['streams_needed'], priority='normal')
    
    def _apply_low_priority_optimization(self, config: Dict[str, Any]) -> None:
        """Apply low priority optimization."""
        # Minimal optimization for low priority operations
        self._ensure_streams(config['streams_needed'], priority='low')
    
    def _optimize_for_training_workload(self, batch_size: int, model_size_mb: float) -> None:
        """Optimize GPU for training workload."""
        # Calculate optimal stream count based on workload
        stream_count = min(8, max(2, batch_size // 8))
        
        # Ensure training streams
        self._ensure_streams(stream_count, priority='high')
        
        # Configure memory management for training
        if model_size_mb > 500:  # Large model
            torch.cuda.set_per_process_memory_fraction(0.95)
    
    def _allocate_training_streams(self) -> List[torch.cuda.Stream]:
        """Allocate streams specifically for training."""
        training_streams = []
        
        for device_id in range(self.device_count):
            with torch.cuda.device(device_id):
                # Create dedicated training streams
                for i in range(4):  # 4 streams per device for training
                    stream = torch.cuda.Stream(device=device_id, priority=-1)  # High priority
                    training_streams.append(stream)
        
        return training_streams
    
    def _synchronize_training_streams(self) -> None:
        """Synchronize all training streams."""
        for device_id in range(self.device_count):
            with torch.cuda.device(device_id):
                # Synchronize all streams on this device
                device_streams = self.streams[device_id]
                for cuda_stream in device_streams.values():
                    if cuda_stream.priority == -1:  # High priority (training) streams
                        cuda_stream.stream.synchronize()
    
    def _cleanup_training_allocations(self, session_id: str) -> None:
        """Clean up training-specific allocations."""
        # Clean up any session-specific allocations
        # For now, just clear cache
        torch.cuda.empty_cache()
    
    def _ensure_training_streams(self) -> None:
        """Ensure training streams are available."""
        self._ensure_streams(4, priority='high')
    
    def _ensure_inference_streams(self) -> None:
        """Ensure inference streams are available."""
        self._ensure_streams(2, priority='normal')
    
    def _ensure_basic_streams(self) -> None:
        """Ensure basic streams are available."""
        self._ensure_streams(1, priority='normal')
    
    def _ensure_streams(self, count: int, priority: str = 'normal') -> None:
        """Ensure a minimum number of streams are available."""
        priority_map = {'high': -1, 'normal': 0, 'low': 1}
        priority_int = priority_map.get(priority, 0)
        
        with self.stream_lock:
            for device_id in range(self.device_count):
                device_streams = self.streams[device_id]
                
                # Count existing streams with this priority
                priority_streams = sum(
                    1 for s in device_streams.values() 
                    if s.priority == priority_int and s.is_active
                )
                
                # Create additional streams if needed
                streams_needed = max(0, count - priority_streams)
                for i in range(streams_needed):
                    stream_name = f"auto_{device_id}_{priority}_{i}_{time.time()}"
                    torch_stream = torch.cuda.Stream(device=device_id, priority=priority_int)
                    
                    cuda_stream = CUDAStream(
                        stream=torch_stream,
                        name=stream_name,
                        priority=priority_int,
                        created_time=time.time(),
                        last_used=time.time()
                    )
                    
                    device_streams[stream_name] = cuda_stream
    
    def _prewarm_memory_allocator(self, memory_mb: float) -> None:
        """Pre-warm the memory allocator for large allocations."""
        try:
            # Allocate and immediately free some memory to pre-warm allocator
            warmup_size = min(100, memory_mb // 10)  # 10% of requested, max 100MB
            warmup_tensor = torch.empty(
                (int(warmup_size * 1024 * 1024 // 4),),  # 4 bytes per float32
                dtype=torch.float32,
                device=f'cuda:{self.current_device}'
            )
            del warmup_tensor
            torch.cuda.empty_cache()
            
        except Exception as e:
            self.logger.warning(f"Memory allocator pre-warming failed: {e}")
    
    def _optimization_loop(self) -> None:
        """Background optimization loop."""
        while self.optimization_active:
            try:
                time.sleep(self.optimization_interval)
                
                if not self.optimization_active:
                    break
                
                # Perform background optimization
                self._background_optimization()
                
            except Exception as e:
                self.logger.error(f"Background optimization error: {e}")
    
    def _background_optimization(self) -> None:
        """Perform background optimization tasks."""
        try:
            # Check memory usage
            current_usage = self._get_current_memory_usage()
            total_memory = sum(
                self.device_properties[i].total_memory / (1024 * 1024)
                for i in range(self.device_count)
            )
            
            usage_ratio = current_usage / max(total_memory, 1)
            
            # Optimize if memory usage is high
            if usage_ratio > self.memory_threshold:
                self.optimize_allocation(usage_ratio)
            
            # Optimize streams periodically
            self.optimize_streams()
            
            # Update metrics
            self._update_memory_statistics()
            
        except Exception as e:
            self.logger.warning(f"Background optimization failed: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            **self.performance_metrics,
            'devices': self.device_count,
            'active_streams': sum(len(streams) for streams in self.streams.values()),
            'memory_info': {
                device_id: self.get_memory_info(device_id)
                for device_id in range(self.device_count)
            },
            'optimization_interval': self.optimization_interval,
            'memory_threshold': self.memory_threshold,
            'fragmentation_threshold': self.fragmentation_threshold
        }
    
    def reset_device(self, device_id: Optional[int] = None) -> None:
        """Reset and clean up a GPU device."""
        if device_id is None:
            device_id = self.current_device
        
        if device_id < 0 or device_id >= self.device_count:
            raise ValueError(f"Invalid device ID: {device_id}")
        
        with torch.cuda.device(device_id):
            # Synchronize all streams
            self.synchronize_streams(device_id)
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats(device_id)
            torch.cuda.reset_accumulated_memory_stats(device_id)
            
            # Clear pools
            self._cleanup_device_pools(device_id)
        
        self.logger.info(f"Reset device {device_id}")
    
    def _cleanup_device_pools(self, device_id: int) -> None:
        """Clean up memory and stream pools for a device."""
        with self.stream_lock:
            if device_id in self.streams:
                self.streams[device_id].clear()
            if device_id in self.stream_pools:
                self.stream_pools[device_id].clear()
        
        with self.memory_lock:
            if device_id in self.memory_blocks:
                self.memory_blocks[device_id].clear()
            if device_id in self.memory_pools:
                self.memory_pools[device_id].clear()
    
    def _initialize_smart_pool(self) -> None:
        """Initialize SmartPool memory management system"""
        try:
            from .smart_pool import integrate_smartpool_with_gpu_optimizer
            self.smart_pool = integrate_smartpool_with_gpu_optimizer(self)
            self.logger.info("SmartPool memory management initialized - targeting 13.3% fragmentation reduction")
        except ImportError as e:
            self.logger.warning(f"SmartPool not available: {e}")
            self.smart_pool = None
        except Exception as e:
            self.logger.error(f"Failed to initialize SmartPool: {e}")
            self.smart_pool = None
    
    def _initialize_autoswap(self) -> None:
        """Initialize AutoSwap priority-based memory swapping system"""
        try:
            from .auto_swap_manager import AutoSwapManager, AutoSwapConfig, SwapPolicy
            
            # Create AutoSwap configuration
            autoswap_config = AutoSwapConfig(
                swap_policy=SwapPolicy(self.config.get('swap_policy', 'balanced')),
                memory_pressure_thresholds={
                    'low': self.config.get('swap_threshold_low', 0.5),
                    'moderate': self.config.get('swap_threshold_moderate', 0.75),
                    'high': self.config.get('swap_threshold_high', 0.9),
                    'critical': self.config.get('swap_threshold_critical', 0.95)
                },
                min_swap_size_mb=self.config.get('min_swap_size_mb', 1.0),
                max_swap_size_mb=self.config.get('max_swap_size_mb', 1024.0),
                enable_predictive_swapping=self.config.get('enable_predictive_swapping', True),
                enable_batch_swapping=self.config.get('enable_batch_swapping', True),
                monitoring_interval_seconds=self.config.get('swap_monitoring_interval', 1.0)
            )
            
            # Initialize AutoSwap
            self.autoswap_manager = AutoSwapManager(self, autoswap_config)
            
            # Start monitoring if requested
            if self.config.get('autoswap_monitoring', True):
                self.autoswap_manager.start_monitoring()
            
            self.logger.info("AutoSwap priority-based memory swapping initialized")
            
        except ImportError as e:
            self.logger.warning(f"AutoSwap not available: {e}")
            self.autoswap_manager = None
        except Exception as e:
            self.logger.error(f"Failed to initialize AutoSwap: {e}")
            self.autoswap_manager = None
    
    def shutdown(self) -> None:
        """Shutdown the GPU memory optimizer and clean up resources."""
        self.logger.info("Shutting down GPU memory optimizer")
        
        # Stop AutoSwap monitoring
        if self.autoswap_manager:
            self.autoswap_manager.stop_monitoring()
        
        # Stop optimization thread
        self.optimization_active = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5.0)
        
        # Synchronize all devices
        for device_id in range(self.device_count):
            try:
                self.synchronize_streams(device_id)
                self.reset_device(device_id)
            except Exception as e:
                self.logger.error(f"Error resetting device {device_id}: {e}")
        
        # Clear all data structures
        with self.stream_lock:
            self.streams.clear()
            self.stream_pools.clear()
        
        with self.memory_lock:
            self.memory_blocks.clear()
            self.memory_pools.clear()
        
        self.kernel_configs.clear()
        self.kernel_performance.clear()
        
        # Clear references
        self.brain_core = None
        self.training_manager = None
        self.compression_systems.clear()
        
        self.logger.info("GPU memory optimizer shutdown complete")


# Legacy compatibility aliases
GPUMemoryManager = GPUMemoryOptimizer
MemoryPool = GPUMemoryOptimizer  # For basic pooling operations
StreamManager = GPUMemoryOptimizer  # For stream management operations
MemoryOptimizer = GPUMemoryOptimizer  # For optimization operations

# Legacy data structures for compatibility
MemoryState = type('MemoryState', (), {
    'FREE': 'free',
    'ALLOCATED': 'allocated', 
    'RESERVED': 'reserved',
    'DEFRAG_PENDING': 'defrag_pending'
})

MemoryBlock = GPUMemoryBlock


# Integration hooks for existing systems
class GPUMemoryIntegration:
    """Integration utilities for existing systems"""
    
    @staticmethod
    def register_with_brain_core(brain_core: Any, gpu_manager: GPUMemoryOptimizer) -> None:
        """Register GPU memory optimizer with BrainCore"""
        if hasattr(brain_core, 'register_gpu_memory_manager'):
            brain_core.register_gpu_memory_manager(gpu_manager)
        brain_core.gpu_memory_optimizer = gpu_manager
    
    @staticmethod
    def integrate_with_training_manager(training_manager: Any, gpu_manager: GPUMemoryOptimizer) -> None:
        """Integrate with TrainingManager"""
        training_manager.gpu_memory_optimizer = gpu_manager
        training_manager.gpu_allocate = lambda shape, device=None: gpu_manager.allocate_tensor(shape, device=device)
        training_manager.gpu_deallocate = gpu_manager.deallocate_tensor
        training_manager.get_gpu_stream = gpu_manager.get_stream
        training_manager.synchronize_gpu = gpu_manager.synchronize_streams
    
    @staticmethod
    def integrate_with_compression_systems(compression_system: Any, gpu_manager: GPUMemoryOptimizer) -> None:
        """Integrate with compression systems"""
        compression_system.gpu_memory_optimizer = gpu_manager
        compression_system.gpu_allocate = lambda shape, device=None: gpu_manager.allocate_tensor(shape, device=device)
        compression_system.gpu_get_stream = gpu_manager.get_stream
        compression_system.gpu_optimize = gpu_manager.optimize_allocation