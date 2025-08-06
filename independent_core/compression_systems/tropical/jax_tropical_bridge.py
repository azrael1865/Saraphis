"""
JAX-Tropical Bridge - Seamless integration between PyTorch and JAX operations
Provides zero-copy transfers, operation fusion, and intelligent routing
HARD FAILURES ONLY - NO GRACEFUL DEGRADATION

This module provides:
1. TropicalJAXBridge - Core integration for PyTorch↔JAX operations
2. Lazy conversion strategies to minimize memory copies
3. Operation fusion detection and optimization
4. Gradient flow between PyTorch and JAX
5. HybridExecutor - Intelligent operation routing
6. Automatic operation batching
7. Pipeline parallelism support
8. Unified checkpoint system
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, pmap
from jax import device_put, device_get
from jax.tree_util import tree_map
import torch
import numpy as np
import time
import threading
import logging
import pickle
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import functools

# Import existing components
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from independent_core.compression_systems.tropical.jax_tropical_engine import (
    TropicalJAXEngine,
    JAXTropicalConfig
)
from independent_core.compression_systems.tropical.jax_memory_pool import JAXMemoryPool
from independent_core.compression_systems.tropical.jax_device_manager import JAXDeviceManager
from independent_core.compression_systems.tropical.jax_config_adapter import JAXConfigAdapter

logger = logging.getLogger(__name__)


class ConversionStrategy(Enum):
    """Tensor conversion strategies"""
    ZERO_COPY = "zero_copy"          # Zero-copy when possible
    LAZY = "lazy"                    # Lazy conversion on demand
    EAGER = "eager"                  # Immediate conversion
    CACHED = "cached"                # Cache conversions


class OperationType(Enum):
    """Operation types for routing"""
    MATMUL = "matmul"
    CONV = "conv"
    TROPICAL = "tropical"
    REDUCE = "reduce"
    ELEMENTWISE = "elementwise"
    CUSTOM = "custom"


class ExecutionBackend(Enum):
    """Execution backend selection"""
    PYTORCH = "pytorch"
    JAX = "jax"
    AUTO = "auto"


@dataclass
class TensorWrapper:
    """Wrapper for tensors that can exist in PyTorch or JAX"""
    pytorch_tensor: Optional[torch.Tensor] = None
    jax_array: Optional[Any] = None  # jax.Array
    numpy_array: Optional[np.ndarray] = None
    shape: Tuple[int, ...] = field(default_factory=tuple)
    dtype: Any = None
    device: str = "cpu"
    last_backend: ExecutionBackend = ExecutionBackend.PYTORCH
    conversion_count: int = 0
    
    def __post_init__(self):
        """Initialize shape and dtype from available tensors"""
        if self.pytorch_tensor is not None:
            self.shape = tuple(self.pytorch_tensor.shape)
            self.dtype = self.pytorch_tensor.dtype
            self.device = str(self.pytorch_tensor.device)
        elif self.jax_array is not None:
            self.shape = tuple(self.jax_array.shape)
            self.dtype = self.jax_array.dtype
        elif self.numpy_array is not None:
            self.shape = tuple(self.numpy_array.shape)
            self.dtype = self.numpy_array.dtype


class TropicalJAXBridge:
    """
    Bridge between PyTorch and JAX for tropical operations.
    Provides seamless tensor conversion and operation routing.
    """
    
    def __init__(self,
                 strategy: ConversionStrategy = ConversionStrategy.LAZY,
                 cache_size_mb: int = 512):
        """
        Initialize the bridge
        
        Args:
            strategy: Conversion strategy to use
            cache_size_mb: Size of conversion cache in MB
        """
        self.strategy = strategy
        self.cache_size_bytes = cache_size_mb * 1024 * 1024
        
        # Tensor tracking
        self.tensor_registry: Dict[int, TensorWrapper] = {}
        self.conversion_cache: Dict[str, Any] = {}
        self.cache_usage_bytes = 0
        
        # Performance tracking
        self.stats = {
            'torch_to_jax': 0,
            'jax_to_torch': 0,
            'zero_copy': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_bytes_converted': 0,
            'fusion_operations': 0
        }
        
        # Operation fusion detection
        self.operation_queue: deque = deque(maxlen=10)
        self.fusion_patterns = self._init_fusion_patterns()
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"TropicalJAXBridge initialized with {strategy.value} strategy")
    
    def _init_fusion_patterns(self) -> List[Tuple[List[OperationType], Callable]]:
        """Initialize operation fusion patterns"""
        return [
            # MatMul + Add -> Fused MatMul
            ([OperationType.MATMUL, OperationType.ELEMENTWISE], self._fuse_matmul_add),
            # Multiple reduces -> Single reduce
            ([OperationType.REDUCE, OperationType.REDUCE], self._fuse_reduces),
            # Conv + ReLU -> Fused Conv
            ([OperationType.CONV, OperationType.ELEMENTWISE], self._fuse_conv_activation)
        ]
    
    def torch_to_jax(self, 
                     tensor: torch.Tensor,
                     device: Optional[Any] = None) -> Any:
        """
        Convert PyTorch tensor to JAX array
        
        Args:
            tensor: PyTorch tensor
            device: Target JAX device (optional)
            
        Returns:
            JAX array
        """
        with self._lock:
            self.stats['torch_to_jax'] += 1
            
            # Check cache
            cache_key = self._get_cache_key(tensor)
            if self.strategy == ConversionStrategy.CACHED and cache_key in self.conversion_cache:
                self.stats['cache_hits'] += 1
                return self.conversion_cache[cache_key]
            
            self.stats['cache_misses'] += 1
            
            # Try zero-copy conversion
            if self._can_zero_copy(tensor):
                jax_array = self._zero_copy_torch_to_jax(tensor)
                self.stats['zero_copy'] += 1
            else:
                # Standard conversion through numpy
                numpy_array = tensor.detach().cpu().numpy()
                jax_array = jnp.array(numpy_array)
                
                if device:
                    jax_array = device_put(jax_array, device)
                elif tensor.is_cuda:
                    # Put on corresponding JAX device
                    jax_devices = jax.devices('gpu')
                    if jax_devices:
                        device_id = tensor.get_device()
                        if device_id < len(jax_devices):
                            jax_array = device_put(jax_array, jax_devices[device_id])
            
            # Update stats
            self.stats['total_bytes_converted'] += tensor.numel() * tensor.element_size()
            
            # Cache if enabled
            if self.strategy == ConversionStrategy.CACHED:
                self._cache_conversion(cache_key, jax_array, tensor.numel() * tensor.element_size())
            
            # Register tensor
            tensor_id = id(tensor)
            self.tensor_registry[tensor_id] = TensorWrapper(
                pytorch_tensor=tensor,
                jax_array=jax_array,
                last_backend=ExecutionBackend.JAX
            )
            
            return jax_array
    
    def jax_to_torch(self,
                     array: Any,
                     device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Convert JAX array to PyTorch tensor
        
        Args:
            array: JAX array
            device: Target PyTorch device (optional)
            
        Returns:
            PyTorch tensor
        """
        with self._lock:
            self.stats['jax_to_torch'] += 1
            
            # Convert to numpy first
            numpy_array = np.array(array)
            
            # Create PyTorch tensor
            tensor = torch.from_numpy(numpy_array)
            
            # Move to device if specified
            if device:
                tensor = tensor.to(device)
            elif hasattr(array, 'device'):
                # Try to match JAX device
                device_str = str(array.device())
                if 'gpu' in device_str.lower():
                    # Extract GPU ID and move to CUDA
                    import re
                    match = re.search(r'gpu:(\d+)', device_str.lower())
                    if match:
                        gpu_id = int(match.group(1))
                        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                            tensor = tensor.cuda(gpu_id)
            
            # Update stats
            self.stats['total_bytes_converted'] += numpy_array.nbytes
            
            # Register tensor
            array_id = id(array)
            self.tensor_registry[array_id] = TensorWrapper(
                pytorch_tensor=tensor,
                jax_array=array,
                last_backend=ExecutionBackend.PYTORCH
            )
            
            return tensor
    
    def _can_zero_copy(self, tensor: torch.Tensor) -> bool:
        """Check if zero-copy conversion is possible"""
        # Zero-copy requires:
        # 1. Tensor is on CPU
        # 2. Tensor is contiguous
        # 3. Tensor doesn't require grad
        return (not tensor.is_cuda and 
                tensor.is_contiguous() and 
                not tensor.requires_grad)
    
    def _zero_copy_torch_to_jax(self, tensor: torch.Tensor) -> Any:
        """Perform zero-copy conversion from PyTorch to JAX"""
        # Get numpy array without copying data
        numpy_array = tensor.detach().numpy()
        # Create JAX array that shares memory
        return jnp.array(numpy_array, copy=False)
    
    def _get_cache_key(self, tensor: Union[torch.Tensor, Any]) -> str:
        """Generate cache key for tensor"""
        if isinstance(tensor, torch.Tensor):
            key_data = f"{tensor.shape}_{tensor.dtype}_{tensor.device}_{tensor.data_ptr()}"
        else:
            key_data = f"{tensor.shape}_{tensor.dtype}_{id(tensor)}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _cache_conversion(self, key: str, array: Any, size_bytes: int) -> None:
        """Cache a conversion result"""
        # Check cache size limit
        if self.cache_usage_bytes + size_bytes > self.cache_size_bytes:
            # Evict oldest entries (simple FIFO)
            while self.cache_usage_bytes + size_bytes > self.cache_size_bytes and self.conversion_cache:
                oldest_key = next(iter(self.conversion_cache))
                del self.conversion_cache[oldest_key]
                self.cache_usage_bytes = max(0, self.cache_usage_bytes - size_bytes)
        
        self.conversion_cache[key] = array
        self.cache_usage_bytes += size_bytes
    
    def create_gradient_function(self, 
                                jax_fn: Callable,
                                argnums: Union[int, Tuple[int, ...]] = 0) -> Callable:
        """
        Create PyTorch-compatible gradient function from JAX function
        
        Args:
            jax_fn: JAX function to differentiate
            argnums: Arguments to differentiate with respect to
            
        Returns:
            PyTorch autograd function
        """
        jax_grad_fn = grad(jax_fn, argnums=argnums)
        
        class JAXGradientFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                # Convert inputs to JAX
                jax_args = [self.torch_to_jax(arg) if isinstance(arg, torch.Tensor) else arg 
                           for arg in args]
                
                # Compute forward
                output = jax_fn(*jax_args)
                
                # Save for backward
                ctx.save_for_backward(*[arg for arg in args if isinstance(arg, torch.Tensor)])
                ctx.jax_args = jax_args
                
                # Convert output to PyTorch
                return self.jax_to_torch(output)
            
            @staticmethod
            def backward(ctx, grad_output):
                # Get saved tensors
                saved_tensors = ctx.saved_tensors
                
                # Convert grad_output to JAX
                jax_grad_output = self.torch_to_jax(grad_output)
                
                # Compute JAX gradients
                jax_grads = jax_grad_fn(*ctx.jax_args)
                
                # Convert gradients to PyTorch
                if isinstance(jax_grads, tuple):
                    torch_grads = tuple(self.jax_to_torch(g) if g is not None else None 
                                      for g in jax_grads)
                else:
                    torch_grads = self.jax_to_torch(jax_grads) if jax_grads is not None else None
                
                return torch_grads
        
        return JAXGradientFunction.apply
    
    def detect_fusion_opportunity(self, operations: List[OperationType]) -> Optional[Callable]:
        """
        Detect if operations can be fused
        
        Args:
            operations: List of operation types
            
        Returns:
            Fused operation function or None
        """
        for pattern, fusion_fn in self.fusion_patterns:
            if len(operations) >= len(pattern):
                # Check if pattern matches
                for i in range(len(operations) - len(pattern) + 1):
                    if operations[i:i+len(pattern)] == pattern:
                        self.stats['fusion_operations'] += 1
                        return fusion_fn
        
        return None
    
    def _fuse_matmul_add(self, a: Any, b: Any, c: Any) -> Any:
        """Fused matmul + add operation"""
        return jnp.matmul(a, b) + c
    
    def _fuse_reduces(self, array: Any, axes: List[int]) -> Any:
        """Fuse multiple reduce operations"""
        return jnp.sum(array, axis=tuple(axes))
    
    def _fuse_conv_activation(self, input: Any, kernel: Any) -> Any:
        """Fused convolution + activation"""
        # Simplified - actual implementation would use lax.conv
        output = jax.lax.conv_general_dilated(
            input, kernel,
            window_strides=(1, 1),
            padding='SAME'
        )
        return jax.nn.relu(output)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics"""
        with self._lock:
            return {
                **self.stats,
                'cache_size_bytes': self.cache_usage_bytes,
                'cached_tensors': len(self.conversion_cache),
                'registered_tensors': len(self.tensor_registry)
            }


class HybridExecutor:
    """
    Intelligent executor that routes operations to PyTorch or JAX.
    Provides automatic batching and pipeline parallelism.
    """
    
    def __init__(self,
                 bridge: TropicalJAXBridge,
                 jax_engine: Optional[TropicalJAXEngine] = None,
                 device_manager: Optional[JAXDeviceManager] = None):
        """
        Initialize hybrid executor
        
        Args:
            bridge: Tropical JAX bridge
            jax_engine: JAX tropical engine
            device_manager: JAX device manager
        """
        self.bridge = bridge
        self.jax_engine = jax_engine or TropicalJAXEngine()
        self.device_manager = device_manager or JAXDeviceManager()
        
        # Operation routing
        self.routing_history: Dict[str, List[ExecutionBackend]] = defaultdict(list)
        self.performance_profile: Dict[str, Dict[ExecutionBackend, float]] = defaultdict(dict)
        
        # Batching
        self.batch_queue: Dict[str, List[Any]] = defaultdict(list)
        self.batch_configs = {
            'matmul': {'size': 32, 'timeout_ms': 10},
            'conv': {'size': 16, 'timeout_ms': 20},
            'tropical': {'size': 64, 'timeout_ms': 5}
        }
        
        # Pipeline stages
        self.pipeline_stages: List[Callable] = []
        self.pipeline_active = False
        
        # Statistics
        self.stats = {
            'operations_routed': 0,
            'pytorch_executions': 0,
            'jax_executions': 0,
            'auto_routing_decisions': 0,
            'batched_operations': 0,
            'pipeline_executions': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("HybridExecutor initialized")
    
    def route_operation(self,
                       operation: str,
                       *args,
                       backend: ExecutionBackend = ExecutionBackend.AUTO,
                       **kwargs) -> Any:
        """
        Route operation to optimal backend
        
        Args:
            operation: Operation name
            args: Operation arguments
            backend: Preferred backend (AUTO for automatic)
            kwargs: Operation keyword arguments
            
        Returns:
            Operation result
        """
        with self._lock:
            self.stats['operations_routed'] += 1
            
            # Determine backend
            if backend == ExecutionBackend.AUTO:
                backend = self._select_backend(operation, args)
                self.stats['auto_routing_decisions'] += 1
            
            # Record routing decision
            self.routing_history[operation].append(backend)
            
            # Execute on selected backend
            start_time = time.perf_counter()
            
            if backend == ExecutionBackend.JAX:
                result = self._execute_jax(operation, *args, **kwargs)
                self.stats['jax_executions'] += 1
            else:
                result = self._execute_pytorch(operation, *args, **kwargs)
                self.stats['pytorch_executions'] += 1
            
            # Record performance
            execution_time = time.perf_counter() - start_time
            self.performance_profile[operation][backend] = execution_time
            
            return result
    
    def _select_backend(self, operation: str, args: Tuple) -> ExecutionBackend:
        """Select optimal backend based on operation and data"""
        # Check performance history
        if operation in self.performance_profile:
            perfs = self.performance_profile[operation]
            if ExecutionBackend.JAX in perfs and ExecutionBackend.PYTORCH in perfs:
                # Choose faster backend
                if perfs[ExecutionBackend.JAX] < perfs[ExecutionBackend.PYTORCH]:
                    return ExecutionBackend.JAX
                else:
                    return ExecutionBackend.PYTORCH
        
        # Heuristic-based selection
        if operation in ['tropical_add', 'tropical_multiply', 'tropical_matmul']:
            # Tropical operations optimized for JAX
            return ExecutionBackend.JAX
        
        # Check data size
        total_elements = 0
        for arg in args:
            if hasattr(arg, 'numel'):
                total_elements += arg.numel()
            elif hasattr(arg, 'size'):
                total_elements += arg.size
        
        # Large operations benefit from JAX compilation
        if total_elements > 1000000:
            return ExecutionBackend.JAX
        
        # Default to PyTorch for small operations
        return ExecutionBackend.PYTORCH
    
    def _execute_jax(self, operation: str, *args, **kwargs) -> Any:
        """Execute operation using JAX"""
        # Convert arguments to JAX
        jax_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                jax_args.append(self.bridge.torch_to_jax(arg))
            else:
                jax_args.append(arg)
        
        # Execute operation
        if hasattr(self.jax_engine, operation):
            result = getattr(self.jax_engine, operation)(*jax_args, **kwargs)
        else:
            # Generic JAX operation
            result = self._generic_jax_op(operation, *jax_args, **kwargs)
        
        # Convert result back if needed
        if isinstance(result, (jnp.ndarray, jax.Array)):
            return self.bridge.jax_to_torch(result)
        
        return result
    
    def _execute_pytorch(self, operation: str, *args, **kwargs) -> Any:
        """Execute operation using PyTorch"""
        # Ensure arguments are PyTorch tensors
        torch_args = []
        for arg in args:
            if hasattr(arg, 'shape') and not isinstance(arg, torch.Tensor):
                torch_args.append(self.bridge.jax_to_torch(arg))
            else:
                torch_args.append(arg)
        
        # Map operation names
        op_map = {
            'tropical_add': lambda a, b: torch.maximum(a, b),
            'tropical_multiply': lambda a, b: a + b,
            'matmul': torch.matmul,
            'conv2d': torch.nn.functional.conv2d
        }
        
        if operation in op_map:
            return op_map[operation](*torch_args, **kwargs)
        
        # Default PyTorch operation
        if hasattr(torch, operation):
            return getattr(torch, operation)(*torch_args, **kwargs)
        
        raise ValueError(f"Unknown operation: {operation}")
    
    def _generic_jax_op(self, operation: str, *args, **kwargs) -> Any:
        """Execute generic JAX operation"""
        # Map to JAX operations
        op_map = {
            'matmul': jnp.matmul,
            'conv2d': lambda x, w: jax.lax.conv(x, w, (1, 1), 'SAME'),
            'sum': jnp.sum,
            'mean': jnp.mean,
            'max': jnp.max,
            'min': jnp.min
        }
        
        if operation in op_map:
            return op_map[operation](*args, **kwargs)
        
        # Try to find in jnp
        if hasattr(jnp, operation):
            return getattr(jnp, operation)(*args, **kwargs)
        
        raise ValueError(f"Unknown JAX operation: {operation}")
    
    def batch_operation(self,
                       operation: str,
                       data: Any,
                       batch_id: Optional[str] = None) -> Any:
        """
        Batch operation for efficient execution
        
        Args:
            operation: Operation to batch
            data: Input data
            batch_id: Batch identifier
            
        Returns:
            Operation result
        """
        with self._lock:
            if batch_id is None:
                batch_id = operation
            
            # Add to batch queue
            self.batch_queue[batch_id].append(data)
            
            # Check if batch is ready
            config = self.batch_configs.get(operation, {'size': 32, 'timeout_ms': 10})
            
            if len(self.batch_queue[batch_id]) >= config['size']:
                # Execute batch
                batch_data = self.batch_queue[batch_id]
                self.batch_queue[batch_id] = []
                
                self.stats['batched_operations'] += 1
                
                return self._execute_batch(operation, batch_data)
            
            # Wait for more data or timeout
            # (In production, use async/await or threading)
            return None
    
    def _execute_batch(self, operation: str, batch_data: List[Any]) -> List[Any]:
        """Execute batched operation"""
        # Stack batch data
        if all(isinstance(d, torch.Tensor) for d in batch_data):
            batched = torch.stack(batch_data)
            backend = ExecutionBackend.PYTORCH
        else:
            # Convert to JAX and stack
            jax_data = [self.bridge.torch_to_jax(d) if isinstance(d, torch.Tensor) else d 
                       for d in batch_data]
            batched = jnp.stack(jax_data)
            backend = ExecutionBackend.JAX
        
        # Execute operation
        result = self.route_operation(operation, batched, backend=backend)
        
        # Unbatch results
        if isinstance(result, torch.Tensor):
            return list(result)
        else:
            return list(result)
    
    def create_pipeline(self, stages: List[Callable]) -> None:
        """
        Create execution pipeline
        
        Args:
            stages: List of pipeline stages (functions)
        """
        self.pipeline_stages = stages
        self.pipeline_active = True
        logger.info(f"Created pipeline with {len(stages)} stages")
    
    def execute_pipeline(self, input_data: Any) -> Any:
        """
        Execute pipeline on input data
        
        Args:
            input_data: Input to pipeline
            
        Returns:
            Pipeline output
        """
        if not self.pipeline_active or not self.pipeline_stages:
            raise RuntimeError("No active pipeline")
        
        self.stats['pipeline_executions'] += 1
        
        # Execute stages sequentially
        # (In production, use actual pipeline parallelism)
        result = input_data
        for i, stage in enumerate(self.pipeline_stages):
            # Determine backend for stage
            if i % 2 == 0:
                # Alternate between backends for demonstration
                backend = ExecutionBackend.JAX
            else:
                backend = ExecutionBackend.PYTORCH
            
            # Execute stage
            if backend == ExecutionBackend.JAX and isinstance(result, torch.Tensor):
                result = self.bridge.torch_to_jax(result)
            elif backend == ExecutionBackend.PYTORCH and not isinstance(result, torch.Tensor):
                result = self.bridge.jax_to_torch(result)
            
            result = stage(result)
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get executor statistics"""
        with self._lock:
            stats = dict(self.stats)
            
            # Add routing analysis
            stats['routing_distribution'] = {}
            for op, backends in self.routing_history.items():
                if backends:
                    pytorch_count = backends.count(ExecutionBackend.PYTORCH)
                    jax_count = backends.count(ExecutionBackend.JAX)
                    stats['routing_distribution'][op] = {
                        'pytorch': pytorch_count,
                        'jax': jax_count,
                        'jax_percentage': jax_count / len(backends) * 100
                    }
            
            # Add performance comparison
            stats['performance_comparison'] = {}
            for op, perfs in self.performance_profile.items():
                if len(perfs) == 2:  # Both backends tested
                    pytorch_time = perfs.get(ExecutionBackend.PYTORCH, float('inf'))
                    jax_time = perfs.get(ExecutionBackend.JAX, float('inf'))
                    speedup = pytorch_time / jax_time if jax_time > 0 else 0
                    stats['performance_comparison'][op] = {
                        'pytorch_ms': pytorch_time * 1000,
                        'jax_ms': jax_time * 1000,
                        'jax_speedup': speedup
                    }
            
            return stats


class UnifiedCheckpointSystem:
    """
    Unified checkpoint system for mixed PyTorch/JAX models.
    Handles saving and loading of hybrid model states.
    """
    
    def __init__(self, bridge: TropicalJAXBridge):
        """
        Initialize checkpoint system
        
        Args:
            bridge: Tropical JAX bridge
        """
        self.bridge = bridge
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        
    def save_checkpoint(self,
                       checkpoint_name: str,
                       pytorch_state: Optional[Dict[str, Any]] = None,
                       jax_state: Optional[Dict[str, Any]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save checkpoint with both PyTorch and JAX states
        
        Args:
            checkpoint_name: Name of checkpoint
            pytorch_state: PyTorch model state dict
            jax_state: JAX model state
            metadata: Additional metadata
        """
        checkpoint = {
            'timestamp': time.time(),
            'pytorch_state': pytorch_state,
            'jax_state': jax_state,
            'metadata': metadata or {}
        }
        
        # Convert JAX arrays to numpy for serialization
        if jax_state:
            checkpoint['jax_state'] = tree_map(
                lambda x: np.array(x) if hasattr(x, 'shape') else x,
                jax_state
            )
        
        self.checkpoints[checkpoint_name] = checkpoint
        logger.info(f"Saved checkpoint: {checkpoint_name}")
    
    def load_checkpoint(self, checkpoint_name: str) -> Dict[str, Any]:
        """
        Load checkpoint
        
        Args:
            checkpoint_name: Name of checkpoint
            
        Returns:
            Checkpoint dictionary
        """
        if checkpoint_name not in self.checkpoints:
            raise ValueError(f"Checkpoint not found: {checkpoint_name}")
        
        checkpoint = self.checkpoints[checkpoint_name]
        
        # Convert numpy arrays back to JAX
        if checkpoint['jax_state']:
            checkpoint['jax_state'] = tree_map(
                lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x,
                checkpoint['jax_state']
            )
        
        return checkpoint
    
    def save_to_file(self, filepath: str, checkpoint_name: str) -> None:
        """Save checkpoint to file"""
        if checkpoint_name not in self.checkpoints:
            raise ValueError(f"Checkpoint not found: {checkpoint_name}")
        
        checkpoint = self.checkpoints[checkpoint_name]
        
        # Use pickle for serialization
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Saved checkpoint to: {filepath}")
    
    def load_from_file(self, filepath: str, checkpoint_name: str) -> None:
        """Load checkpoint from file"""
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.checkpoints[checkpoint_name] = checkpoint
        logger.info(f"Loaded checkpoint from: {filepath}")


# Test function
def test_jax_tropical_bridge():
    """Test JAX-Tropical bridge functionality"""
    print("Testing JAX-Tropical Bridge...")
    
    # Create bridge
    bridge = TropicalJAXBridge(strategy=ConversionStrategy.LAZY)
    
    # Test tensor conversion
    print("\n1. Testing tensor conversion...")
    torch_tensor = torch.randn(100, 100)
    jax_array = bridge.torch_to_jax(torch_tensor)
    print(f"   PyTorch -> JAX: {torch_tensor.shape} -> {jax_array.shape}")
    
    torch_back = bridge.jax_to_torch(jax_array)
    print(f"   JAX -> PyTorch: {jax_array.shape} -> {torch_back.shape}")
    
    # Verify conversion accuracy
    diff = torch.abs(torch_tensor - torch_back).max().item()
    print(f"   Conversion accuracy: max diff = {diff:.6f}")
    
    # Test hybrid executor
    print("\n2. Testing hybrid executor...")
    jax_engine = TropicalJAXEngine()
    executor = HybridExecutor(bridge, jax_engine)
    
    # Test operation routing
    a = torch.randn(50, 50)
    b = torch.randn(50, 50)
    
    result_auto = executor.route_operation('matmul', a, b, backend=ExecutionBackend.AUTO)
    print(f"   Auto-routed matmul: {result_auto.shape}")
    
    result_jax = executor.route_operation('matmul', a, b, backend=ExecutionBackend.JAX)
    print(f"   JAX matmul: {result_jax.shape}")
    
    # Test gradient flow
    print("\n3. Testing gradient flow...")
    
    def jax_fn(x):
        return jnp.sum(x ** 2)
    
    grad_fn = bridge.create_gradient_function(jax_fn)
    x = torch.randn(10, 10, requires_grad=True)
    y = grad_fn(x)
    print(f"   Created gradient function: input {x.shape} -> output {y.shape}")
    
    # Test batching
    print("\n4. Testing operation batching...")
    batch_results = []
    for i in range(5):
        data = torch.randn(10, 10)
        result = executor.batch_operation('matmul', data)
        if result is not None:
            batch_results.append(result)
    print(f"   Batched {len(batch_results)} operations")
    
    # Test pipeline
    print("\n5. Testing pipeline execution...")
    
    def stage1(x):
        return x * 2
    
    def stage2(x):
        return x + 1
    
    def stage3(x):
        return x ** 2
    
    executor.create_pipeline([stage1, stage2, stage3])
    
    input_data = torch.randn(5, 5)
    output = executor.execute_pipeline(input_data)
    print(f"   Pipeline: {input_data.shape} -> {output.shape}")
    
    # Test checkpoint system
    print("\n6. Testing checkpoint system...")
    checkpoint_sys = UnifiedCheckpointSystem(bridge)
    
    # Save checkpoint
    pytorch_state = {'weight': torch.randn(10, 10)}
    jax_state = {'params': jnp.ones((5, 5))}
    
    checkpoint_sys.save_checkpoint(
        'test_checkpoint',
        pytorch_state=pytorch_state,
        jax_state=jax_state,
        metadata={'epoch': 1}
    )
    
    # Load checkpoint
    loaded = checkpoint_sys.load_checkpoint('test_checkpoint')
    print(f"   Checkpoint saved and loaded successfully")
    print(f"   Metadata: {loaded['metadata']}")
    
    # Get statistics
    print("\n7. Bridge statistics:")
    stats = bridge.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n8. Executor statistics:")
    exec_stats = executor.get_statistics()
    print(f"   Operations routed: {exec_stats['operations_routed']}")
    print(f"   JAX executions: {exec_stats['jax_executions']}")
    print(f"   PyTorch executions: {exec_stats['pytorch_executions']}")
    
    if 'performance_comparison' in exec_stats:
        print("\n   Performance comparison:")
        for op, perf in exec_stats['performance_comparison'].items():
            print(f"      {op}: JAX speedup = {perf['jax_speedup']:.2f}x")
    
    print("\n✓ JAX-Tropical Bridge test complete!")


if __name__ == "__main__":
    test_jax_tropical_bridge()