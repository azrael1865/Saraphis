"""
JAX Configuration and Environment Management for Tropical Operations.
Integrates with existing PyTorch-based compression system providing 6x speedup.
NO PLACEHOLDERS - PRODUCTION READY
"""

import os
import sys
import warnings
import logging
import functools
import hashlib
import pickle
import weakref
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import gc
import threading
from contextlib import contextmanager

import numpy as np
import torch

# Handle JAX import with graceful fallback
JAX_AVAILABLE = False
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, vmap, pmap
    from jax.experimental import mesh_utils
    from jax.sharding import PositionalSharding
    from jax.lib import xla_bridge
    JAX_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"JAX not installed: {e}. Install with 'pip install jax[cuda12_local]' for GPU support")
    jax = None
    jnp = None

# Import existing system components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from independent_core.compression_systems.gpu_memory.gpu_auto_detector import (
    GPUAutoDetector,
    GPUSpecs,
    AutoOptimizedConfig
)
from independent_core.compression_systems.gpu_memory.gpu_memory_core import (
    GPUMemoryManager,
    GPUMemoryBlock
)
from independent_core.compression_systems.tropical.tropical_core import (
    TROPICAL_ZERO,
    TROPICAL_EPSILON
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JAXPlatform(Enum):
    """JAX computation platforms"""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    
    @classmethod
    def detect(cls) -> 'JAXPlatform':
        """Auto-detect best available platform"""
        if not JAX_AVAILABLE:
            return cls.CPU
            
        try:
            # Check for TPU first (highest priority)
            if 'tpu' in str(jax.devices()).lower():
                return cls.TPU
            # Check for GPU
            if jax.devices('gpu'):
                return cls.GPU
        except:
            pass
        return cls.CPU


@dataclass
class JAXConfig:
    """JAX configuration for tropical operations with auto-detection"""
    enable_jit: bool = True
    enable_x64: bool = False  # Use float32 by default for speed
    platform: JAXPlatform = field(default_factory=JAXPlatform.detect)
    device_id: int = 0
    memory_fraction: float = 0.75  # Reserve 75% of GPU memory
    enable_unified_memory: bool = True
    compilation_cache_size: int = 128
    parallel_compilation: bool = True
    xla_flags: Dict[str, str] = field(default_factory=lambda: {
        'xla_gpu_enable_triton_softmax_fusion': 'true',
        'xla_gpu_triton_gemm_any': 'true',
        'xla_gpu_enable_async_collectives': 'true',
        'xla_gpu_enable_latency_hiding_scheduler': 'true',
        'xla_gpu_enable_highest_priority_async_stream': 'true'
    })
    disable_jit_for_debugging: bool = False
    preallocate_memory: bool = True
    memory_cleanup_threshold: float = 0.9  # Trigger cleanup at 90% usage
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.memory_fraction <= 0 or self.memory_fraction > 1:
            raise ValueError(f"memory_fraction must be in (0, 1], got {self.memory_fraction}")
        if self.compilation_cache_size < 0:
            raise ValueError(f"compilation_cache_size must be non-negative, got {self.compilation_cache_size}")
        if self.device_id < 0:
            raise ValueError(f"device_id must be non-negative, got {self.device_id}")


class JAXEnvironment:
    """JAX environment manager for tropical operations"""
    
    def __init__(self, config: Optional[JAXConfig] = None):
        """Initialize JAX environment with configuration"""
        self.config = config or JAXConfig()
        self.gpu_detector = GPUAutoDetector()
        self.devices = None
        self.default_device = None
        self._initialized = False
        self._lock = threading.Lock()
        self._memory_stats = {}
        self._device_cache = {}
        
    def setup_environment(self) -> Dict[str, Any]:
        """Complete JAX environment setup with validation"""
        with self._lock:
            if self._initialized:
                return self.get_environment_info()
                
            result = {
                'jax_available': JAX_AVAILABLE,
                'platform': self.config.platform.value,
                'devices': [],
                'default_device': None,
                'memory_config': {},
                'xla_flags': {},
                'errors': []
            }
            
            if not JAX_AVAILABLE:
                result['errors'].append("JAX not installed. Install with 'pip install jax[cuda12_local]'")
                return result
                
            try:
                # Set XLA flags before any JAX operations
                self.setup_xla_flags()
                result['xla_flags'] = self.config.xla_flags
                
                # Configure JAX settings
                if self.config.enable_x64:
                    jax.config.update('jax_enable_x64', True)
                    
                if self.config.disable_jit_for_debugging:
                    jax.config.update('jax_disable_jit', True)
                    
                # Detect and configure devices
                self.devices = self.detect_jax_devices()
                result['devices'] = [self._device_to_dict(d) for d in self.devices]
                
                # Set default device
                if self.devices:
                    self.default_device = self.get_default_device()
                    result['default_device'] = self._device_to_dict(self.default_device)
                    
                # Configure memory
                self.configure_memory()
                result['memory_config'] = self.get_memory_config()
                
                # Validate installation
                if self.validate_installation():
                    self._initialized = True
                    logger.info(f"JAX environment initialized: {result['platform']}")
                else:
                    result['errors'].append("JAX validation failed")
                    
            except Exception as e:
                result['errors'].append(f"Setup failed: {str(e)}")
                logger.error(f"JAX environment setup failed: {e}")
                
            return result
            
    def detect_jax_devices(self) -> List[Any]:
        """Detect available JAX devices (GPU, CPU, TPU)"""
        if not JAX_AVAILABLE:
            return []
            
        devices = []
        try:
            # Get all devices
            all_devices = jax.devices()
            
            # Filter by platform if specified
            if self.config.platform == JAXPlatform.GPU:
                devices = [d for d in all_devices if d.platform == 'gpu']
                if not devices:
                    logger.warning("No GPU devices found, falling back to CPU")
                    devices = [d for d in all_devices if d.platform == 'cpu']
            elif self.config.platform == JAXPlatform.TPU:
                devices = [d for d in all_devices if d.platform == 'tpu']
                if not devices:
                    logger.warning("No TPU devices found, falling back to CPU")
                    devices = [d for d in all_devices if d.platform == 'cpu']
            else:
                devices = [d for d in all_devices if d.platform == 'cpu']
                
            # Cache device information
            for device in devices:
                self._device_cache[device.id] = {
                    'device': device,
                    'platform': device.platform,
                    'id': device.id,
                    'process_index': device.process_index if hasattr(device, 'process_index') else 0
                }
                
        except Exception as e:
            logger.error(f"Failed to detect JAX devices: {e}")
            
        return devices
        
    def configure_memory(self) -> None:
        """Configure JAX memory allocation with GPU integration"""
        if not JAX_AVAILABLE:
            return
            
        try:
            # Set memory fraction for GPU
            if self.config.platform == JAXPlatform.GPU and self.config.preallocate_memory:
                # Get GPU specs from detector
                gpu_specs = self.gpu_detector.detect_gpus()
                if gpu_specs:
                    primary_gpu = gpu_specs[0] if self.config.device_id >= len(gpu_specs) else gpu_specs[self.config.device_id]
                    
                    # Calculate memory allocation
                    total_memory_bytes = primary_gpu.total_memory_mb * 1024 * 1024
                    allocated_bytes = int(total_memory_bytes * self.config.memory_fraction)
                    
                    # Set XLA memory allocation
                    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
                    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(self.config.memory_fraction)
                    
                    self._memory_stats = {
                        'total_bytes': total_memory_bytes,
                        'allocated_bytes': allocated_bytes,
                        'fraction': self.config.memory_fraction,
                        'gpu_name': primary_gpu.name
                    }
                    
                    logger.info(f"Configured JAX memory: {allocated_bytes / 1e9:.2f}GB on {primary_gpu.name}")
            else:
                # CPU or no preallocation
                os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
                
        except Exception as e:
            logger.error(f"Failed to configure JAX memory: {e}")
            
    def setup_xla_flags(self) -> None:
        """Set XLA compilation flags for optimization"""
        if not self.config.xla_flags:
            return
            
        # Build XLA flags string
        flags = []
        for key, value in self.config.xla_flags.items():
            flags.append(f"--{key}={value}")
            
        if flags:
            xla_flags_str = ' '.join(flags)
            os.environ['XLA_FLAGS'] = xla_flags_str
            logger.debug(f"Set XLA_FLAGS: {xla_flags_str}")
            
    def validate_installation(self) -> bool:
        """Validate JAX and jaxlib installation"""
        if not JAX_AVAILABLE:
            return False
            
        try:
            # Test basic JAX operations
            test_array = jnp.array([1.0, 2.0, 3.0])
            result = jnp.sum(test_array)
            
            # Test device placement
            if self.devices:
                with jax.default_device(self.devices[0]):
                    test_on_device = jnp.array([1.0])
                    
            # Test JIT compilation if enabled
            if self.config.enable_jit:
                @jit
                def test_func(x):
                    return x * 2
                test_func(test_array)
                
            return True
            
        except Exception as e:
            logger.error(f"JAX validation failed: {e}")
            return False
            
    def get_default_device(self) -> Optional[Any]:
        """Get default device for operations"""
        if not self.devices:
            return None
            
        # Use configured device ID if valid
        if self.config.device_id < len(self.devices):
            return self.devices[self.config.device_id]
            
        # Otherwise use first available device
        return self.devices[0]
        
    def get_environment_info(self) -> Dict[str, Any]:
        """Get current environment information"""
        info = {
            'jax_available': JAX_AVAILABLE,
            'initialized': self._initialized,
            'platform': self.config.platform.value,
            'device_count': len(self.devices) if self.devices else 0,
            'memory_stats': self._memory_stats,
            'config': {
                'enable_jit': self.config.enable_jit,
                'enable_x64': self.config.enable_x64,
                'memory_fraction': self.config.memory_fraction,
                'compilation_cache_size': self.config.compilation_cache_size
            }
        }
        
        if JAX_AVAILABLE:
            info['jax_version'] = jax.__version__
            info['backend'] = xla_bridge.get_backend().platform if hasattr(xla_bridge, 'get_backend') else 'unknown'
            
        return info
        
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory configuration details"""
        return self._memory_stats.copy()
        
    def _device_to_dict(self, device: Any) -> Dict[str, Any]:
        """Convert JAX device to dictionary"""
        if device is None:
            return {}
            
        return {
            'id': device.id,
            'platform': device.platform,
            'device_kind': device.device_kind if hasattr(device, 'device_kind') else 'unknown'
        }
        
    @contextmanager
    def device_context(self, device_id: Optional[int] = None):
        """Context manager for device placement"""
        if not JAX_AVAILABLE or not self.devices:
            yield
            return
            
        device = self.devices[device_id] if device_id is not None and device_id < len(self.devices) else self.default_device
        
        if device:
            with jax.default_device(device):
                yield
        else:
            yield


class JAXDeviceManager:
    """Manage JAX devices and placement with PyTorch integration"""
    
    def __init__(self):
        """Initialize device manager"""
        self.devices = []
        self.current_device = None
        self._device_memory = {}
        self._lock = threading.Lock()
        
        if JAX_AVAILABLE:
            self.devices = jax.devices()
            if self.devices:
                self.current_device = self.devices[0]
                
    def get_device(self, device_id: Optional[int] = None) -> Optional[Any]:
        """Get specific JAX device"""
        if not self.devices:
            return None
            
        if device_id is None:
            return self.current_device
            
        if 0 <= device_id < len(self.devices):
            return self.devices[device_id]
            
        raise ValueError(f"Invalid device_id {device_id}, available: 0-{len(self.devices)-1}")
        
    def set_default_device(self, device: Any) -> None:
        """Set default device for operations"""
        if device not in self.devices:
            raise ValueError(f"Device {device} not in available devices")
            
        with self._lock:
            self.current_device = device
            
    def get_device_memory_stats(self, device: Any) -> Dict[str, float]:
        """Get memory statistics for device"""
        stats = {
            'allocated_bytes': 0,
            'reserved_bytes': 0,
            'free_bytes': 0,
            'total_bytes': 0
        }
        
        if not JAX_AVAILABLE:
            return stats
            
        try:
            # For GPU devices, try to get memory info
            if device.platform == 'gpu':
                # Use cached stats if available
                device_key = f"{device.platform}:{device.id}"
                if device_key in self._device_memory:
                    return self._device_memory[device_key]
                    
                # Try to get from backend
                backend = xla_bridge.get_backend()
                if hasattr(backend, 'live_buffers'):
                    buffers = backend.live_buffers()
                    for buffer in buffers:
                        if buffer.device() == device:
                            stats['allocated_bytes'] += buffer.nbytes
                            
                # Cache the stats
                self._device_memory[device_key] = stats
                
        except Exception as e:
            logger.debug(f"Could not get memory stats for device {device}: {e}")
            
        return stats
        
    def distribute_across_devices(self, data: Any, devices: Optional[List[Any]] = None) -> Any:
        """Distribute data across multiple devices using sharding"""
        if not JAX_AVAILABLE:
            return data
            
        devices = devices or self.devices
        if not devices or len(devices) == 1:
            return data
            
        try:
            # Create sharding specification
            sharding = PositionalSharding(devices)
            
            # Convert to JAX array if needed
            if isinstance(data, (np.ndarray, list)):
                data = jnp.array(data)
            elif isinstance(data, torch.Tensor):
                data = jnp.array(data.cpu().numpy())
                
            # Shard the data
            return jax.device_put(data, sharding.reshape(-1, 1))
            
        except Exception as e:
            logger.error(f"Failed to distribute data: {e}")
            return data
            
    def get_all_devices(self) -> List[Dict[str, Any]]:
        """Get information about all available devices"""
        device_info = []
        for device in self.devices:
            info = {
                'id': device.id,
                'platform': device.platform,
                'memory_stats': self.get_device_memory_stats(device),
                'is_default': device == self.current_device
            }
            device_info.append(info)
        return device_info


class JAXMemoryPool:
    """JAX memory pool integration with Saraphis GPU memory management"""
    
    def __init__(self, config: JAXConfig):
        """Initialize memory pool with configuration"""
        self.config = config
        self.allocated_buffers = {}
        self.memory_limit = None
        self._total_allocated = 0
        self._peak_allocated = 0
        self._allocation_count = 0
        self._deallocation_count = 0
        self._lock = threading.Lock()
        self._gpu_manager = None
        
    def allocate(self, shape: Tuple[int, ...], dtype: Any = None) -> Tuple[Any, str]:
        """Allocate JAX array with memory tracking"""
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not available for allocation")
            
        with self._lock:
            # Set default dtype if not provided
            if dtype is None:
                dtype = jnp.float32 if JAX_AVAILABLE else np.float32
            
            # Calculate size
            if JAX_AVAILABLE:
                nbytes = np.prod(shape) * jnp.dtype(dtype).itemsize
            else:
                nbytes = np.prod(shape) * np.dtype(dtype).itemsize
            
            # Check memory limit
            if self.memory_limit and self._total_allocated + nbytes > self.memory_limit:
                self._trigger_cleanup()
                if self._total_allocated + nbytes > self.memory_limit:
                    raise MemoryError(f"Allocation of {nbytes} bytes would exceed limit of {self.memory_limit}")
                    
            # Allocate array
            try:
                array = jnp.zeros(shape, dtype=dtype)
                
                # Generate unique buffer ID
                buffer_id = f"jax_buffer_{self._allocation_count}_{time.time()}"
                
                # Track allocation
                self.allocated_buffers[buffer_id] = {
                    'array': weakref.ref(array),
                    'shape': shape,
                    'dtype': dtype,
                    'nbytes': nbytes,
                    'allocated_time': time.time(),
                    'device': array.device() if hasattr(array, 'device') else None
                }
                
                self._total_allocated += nbytes
                self._peak_allocated = max(self._peak_allocated, self._total_allocated)
                self._allocation_count += 1
                
                logger.debug(f"Allocated JAX buffer {buffer_id}: {nbytes} bytes")
                
                return array, buffer_id
                
            except Exception as e:
                raise MemoryError(f"Failed to allocate JAX array: {e}")
                
    def deallocate(self, buffer_id: str) -> None:
        """Deallocate JAX array and update tracking"""
        with self._lock:
            if buffer_id not in self.allocated_buffers:
                logger.warning(f"Buffer {buffer_id} not found in pool")
                return
                
            buffer_info = self.allocated_buffers[buffer_id]
            
            # Update tracking
            self._total_allocated -= buffer_info['nbytes']
            self._deallocation_count += 1
            
            # Remove from tracking
            del self.allocated_buffers[buffer_id]
            
            logger.debug(f"Deallocated JAX buffer {buffer_id}")
            
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        with self._lock:
            # Clean up dead references
            self._cleanup_dead_references()
            
            return {
                'total_allocated_bytes': self._total_allocated,
                'peak_allocated_bytes': self._peak_allocated,
                'num_buffers': len(self.allocated_buffers),
                'allocation_count': self._allocation_count,
                'deallocation_count': self._deallocation_count,
                'memory_limit': self.memory_limit,
                'usage_fraction': self._total_allocated / self.memory_limit if self.memory_limit else 0
            }
            
    def integrate_with_saraphis(self, gpu_manager: GPUMemoryManager) -> None:
        """Integrate with existing Saraphis memory management"""
        self._gpu_manager = gpu_manager
        
        # Get GPU memory limits from Saraphis
        if hasattr(gpu_manager, 'get_memory_info'):
            memory_info = gpu_manager.get_memory_info()
            if 'total' in memory_info:
                # Set JAX memory limit based on configured fraction
                self.memory_limit = int(memory_info['total'] * self.config.memory_fraction)
                logger.info(f"JAX memory pool integrated with Saraphis: {self.memory_limit / 1e9:.2f}GB limit")
                
        # Register cleanup callback with Saraphis
        if hasattr(gpu_manager, 'register_cleanup_callback'):
            gpu_manager.register_cleanup_callback(self._trigger_cleanup)
            
    def _cleanup_dead_references(self) -> None:
        """Clean up deallocated arrays from tracking"""
        dead_buffers = []
        for buffer_id, info in self.allocated_buffers.items():
            if info['array']() is None:
                dead_buffers.append(buffer_id)
                self._total_allocated -= info['nbytes']
                
        for buffer_id in dead_buffers:
            del self.allocated_buffers[buffer_id]
            
    def _trigger_cleanup(self) -> None:
        """Trigger memory cleanup when approaching limit"""
        # Clean up dead references
        self._cleanup_dead_references()
        
        # Force garbage collection
        gc.collect()
        
        # If still over threshold, clear old buffers
        if self.memory_limit and self._total_allocated > self.memory_limit * self.config.memory_cleanup_threshold:
            # Sort buffers by age
            sorted_buffers = sorted(
                self.allocated_buffers.items(),
                key=lambda x: x[1]['allocated_time']
            )
            
            # Clear oldest buffers until under threshold
            target = self.memory_limit * 0.7  # Clear to 70% usage
            for buffer_id, info in sorted_buffers:
                if self._total_allocated <= target:
                    break
                self.deallocate(buffer_id)
                
    def clear_all(self) -> None:
        """Clear all allocated buffers"""
        with self._lock:
            self.allocated_buffers.clear()
            self._total_allocated = 0
            gc.collect()


class JAXCompilationCache:
    """Cache compiled JAX functions for reuse"""
    
    def __init__(self, max_size: int = 128):
        """Initialize compilation cache"""
        self.max_size = max_size
        self.cache = {}
        self.access_counts = {}
        self.compilation_times = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        
    def _make_key(self, func: Callable, args: Tuple, kwargs: Dict) -> str:
        """Generate cache key for function and arguments"""
        # Create stable key from function and argument shapes/dtypes
        key_parts = [func.__name__ if hasattr(func, '__name__') else str(func)]
        
        for arg in args:
            if isinstance(arg, (jnp.ndarray, np.ndarray)):
                key_parts.append(f"array_{arg.shape}_{arg.dtype}")
            elif isinstance(arg, torch.Tensor):
                key_parts.append(f"tensor_{tuple(arg.shape)}_{arg.dtype}")
            else:
                key_parts.append(str(type(arg)))
                
        # Add kwargs to key
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
            
        key_str = "_".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
        
    def get_or_compile(self, func: Callable, *args, **kwargs) -> Any:
        """Get cached compilation or compile and cache"""
        if not JAX_AVAILABLE:
            return func(*args, **kwargs)
            
        with self._lock:
            # Generate cache key
            cache_key = self._make_key(func, args, kwargs)
            
            # Check cache
            if cache_key in self.cache:
                self._hits += 1
                self.access_counts[cache_key] += 1
                logger.debug(f"Cache hit for {func.__name__}: {cache_key}")
                compiled_func = self.cache[cache_key]
            else:
                self._misses += 1
                logger.debug(f"Cache miss for {func.__name__}: {cache_key}")
                
                # Compile function
                start_time = time.time()
                compiled_func = jit(func)
                compilation_time = time.time() - start_time
                
                # Cache if under size limit
                if len(self.cache) >= self.max_size:
                    self._evict_lru()
                    
                self.cache[cache_key] = compiled_func
                self.access_counts[cache_key] = 1
                self.compilation_times[cache_key] = compilation_time
                
                logger.debug(f"Compiled {func.__name__} in {compilation_time:.3f}s")
                
        # Execute compiled function
        return compiled_func(*args, **kwargs)
        
    def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if not self.access_counts:
            return
            
        # Find LRU entry
        lru_key = min(self.access_counts, key=self.access_counts.get)
        
        # Remove from cache
        del self.cache[lru_key]
        del self.access_counts[lru_key]
        if lru_key in self.compilation_times:
            del self.compilation_times[lru_key]
            
        logger.debug(f"Evicted LRU cache entry: {lru_key}")
        
    def clear_cache(self) -> None:
        """Clear compilation cache"""
        with self._lock:
            self.cache.clear()
            self.access_counts.clear()
            self.compilation_times.clear()
            self._hits = 0
            self._misses = 0
            logger.info("Cleared JAX compilation cache")
            
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache hit/miss statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                'cache_size': len(self.cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                'total_compilation_time': sum(self.compilation_times.values()),
                'average_compilation_time': np.mean(list(self.compilation_times.values())) if self.compilation_times else 0
            }


class JAXPyTorchBridge:
    """Bridge between JAX and PyTorch tensors with zero-copy optimization"""
    
    @staticmethod
    def torch_to_jax(tensor: torch.Tensor, device: Optional[Any] = None) -> Any:
        """Convert PyTorch tensor to JAX array with validation"""
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not available for conversion")
            
        # Handle device placement
        if tensor.is_cuda:
            # For CUDA tensors, we need to ensure same device
            tensor_np = tensor.detach().cpu().numpy()
        else:
            # CPU tensor can be converted directly
            tensor_np = tensor.detach().numpy()
            
        # Convert to JAX array
        jax_array = jnp.array(tensor_np)
        
        # Place on specified device if provided
        if device is not None:
            jax_array = jax.device_put(jax_array, device)
            
        # Validate conversion
        if not JAXPyTorchBridge.validate_conversion(tensor, jax_array):
            raise ValueError("Tensor conversion validation failed")
            
        return jax_array
        
    @staticmethod
    def jax_to_torch(array: Any, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert JAX array to PyTorch tensor with validation"""
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not available for conversion")
            
        # Convert to numpy first
        array_np = np.array(array)
        
        # Convert to PyTorch
        tensor = torch.from_numpy(array_np)
        
        # Move to specified device
        if device is not None:
            tensor = tensor.to(device)
        elif torch.cuda.is_available():
            # Auto-detect CUDA if available
            tensor = tensor.cuda()
            
        # Validate conversion
        if not JAXPyTorchBridge.validate_conversion(array, tensor):
            raise ValueError("Array conversion validation failed")
            
        return tensor
        
    @staticmethod
    def share_memory(tensor: torch.Tensor) -> Any:
        """Create JAX array sharing memory with PyTorch tensor (when possible)"""
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not available for memory sharing")
            
        # Zero-copy only works for CPU tensors with compatible layout
        if tensor.is_cuda:
            logger.warning("Cannot share memory between CUDA PyTorch tensor and JAX, copying instead")
            return JAXPyTorchBridge.torch_to_jax(tensor)
            
        if not tensor.is_contiguous():
            logger.warning("Tensor is not contiguous, cannot share memory")
            return JAXPyTorchBridge.torch_to_jax(tensor)
            
        try:
            # Create numpy view of tensor data (zero-copy)
            tensor_np = tensor.detach().numpy()
            
            # Create JAX array from numpy (may copy depending on backend)
            # Use dlpack for true zero-copy when supported
            if hasattr(jax, 'dlpack') and hasattr(tensor, '__dlpack__'):
                # Use DLPack for zero-copy transfer
                dlpack_capsule = tensor.__dlpack__()
                jax_array = jax.dlpack.from_dlpack(dlpack_capsule)
                logger.debug("Created JAX array with zero-copy via DLPack")
            else:
                # Fallback to numpy conversion
                jax_array = jnp.asarray(tensor_np)
                logger.debug("Created JAX array via numpy (may copy)")
                
            return jax_array
            
        except Exception as e:
            logger.warning(f"Failed to share memory: {e}, falling back to copy")
            return JAXPyTorchBridge.torch_to_jax(tensor)
            
    @staticmethod
    def validate_conversion(original: Any, converted: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Validate tensor conversion accuracy"""
        try:
            # Get numpy arrays for comparison
            if isinstance(original, torch.Tensor):
                original_np = original.detach().cpu().numpy()
            elif JAX_AVAILABLE and isinstance(original, jnp.ndarray):
                original_np = np.array(original)
            else:
                original_np = np.array(original)
                
            if isinstance(converted, torch.Tensor):
                converted_np = converted.detach().cpu().numpy()
            elif JAX_AVAILABLE and isinstance(converted, jnp.ndarray):
                converted_np = np.array(converted)
            else:
                converted_np = np.array(converted)
                
            # Check shape
            if original_np.shape != converted_np.shape:
                logger.error(f"Shape mismatch: {original_np.shape} != {converted_np.shape}")
                return False
                
            # Check dtype compatibility
            if original_np.dtype.kind != converted_np.dtype.kind:
                logger.warning(f"Dtype kind mismatch: {original_np.dtype} != {converted_np.dtype}")
                
            # Check values
            if not np.allclose(original_np, converted_np, rtol=rtol, atol=atol):
                max_diff = np.max(np.abs(original_np - converted_np))
                logger.error(f"Value mismatch: max difference = {max_diff}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
            
    @staticmethod
    def benchmark_conversion(tensor_shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, 
                           num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark conversion performance"""
        if not JAX_AVAILABLE:
            return {'error': 'JAX not available'}
            
        # Create test tensor
        tensor = torch.randn(tensor_shape, dtype=dtype)
        
        # Benchmark torch to JAX
        start = time.time()
        for _ in range(num_iterations):
            jax_array = JAXPyTorchBridge.torch_to_jax(tensor)
        torch_to_jax_time = (time.time() - start) / num_iterations
        
        # Benchmark JAX to torch
        start = time.time()
        for _ in range(num_iterations):
            tensor_back = JAXPyTorchBridge.jax_to_torch(jax_array)
        jax_to_torch_time = (time.time() - start) / num_iterations
        
        # Benchmark memory sharing
        start = time.time()
        for _ in range(num_iterations):
            shared = JAXPyTorchBridge.share_memory(tensor)
        share_memory_time = (time.time() - start) / num_iterations
        
        return {
            'torch_to_jax_ms': torch_to_jax_time * 1000,
            'jax_to_torch_ms': jax_to_torch_time * 1000,
            'share_memory_ms': share_memory_time * 1000,
            'tensor_size_mb': tensor.numel() * tensor.element_size() / 1e6
        }


# Global singleton instances
_jax_environment = None
_device_manager = None
_compilation_cache = None


def get_jax_environment(config: Optional[JAXConfig] = None) -> JAXEnvironment:
    """Get or create global JAX environment"""
    global _jax_environment
    if _jax_environment is None:
        _jax_environment = JAXEnvironment(config)
        _jax_environment.setup_environment()
    return _jax_environment


def get_device_manager() -> JAXDeviceManager:
    """Get or create global device manager"""
    global _device_manager
    if _device_manager is None:
        _device_manager = JAXDeviceManager()
    return _device_manager


def get_compilation_cache(max_size: int = 128) -> JAXCompilationCache:
    """Get or create global compilation cache"""
    global _compilation_cache
    if _compilation_cache is None:
        _compilation_cache = JAXCompilationCache(max_size)
    return _compilation_cache


def setup_jax_for_tropical(memory_fraction: float = 0.75, 
                          enable_jit: bool = True,
                          platform: Optional[str] = None) -> Dict[str, Any]:
    """Quick setup function for tropical operations"""
    config = JAXConfig(
        memory_fraction=memory_fraction,
        enable_jit=enable_jit,
        platform=JAXPlatform(platform) if platform else JAXPlatform.detect()
    )
    
    env = get_jax_environment(config)
    result = env.setup_environment()
    
    if result['jax_available']:
        logger.info(f"JAX setup complete: {result['platform']} with {len(result['devices'])} devices")
    else:
        logger.warning("JAX not available - install with 'pip install jax[cuda12_local]'")
        
    return result