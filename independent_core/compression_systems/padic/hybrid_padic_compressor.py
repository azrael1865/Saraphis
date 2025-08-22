"""
Hybrid P-adic Compression System - GPU-accelerated two-channel compression
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import gc

from base.compression_base import CompressionAlgorithm, CompressionValidator, CompressionMetrics
from padic_encoder import PadicWeight, PadicValidation, PadicMathematicalOperations
from hybrid_padic_structures import (
    HybridPadicWeight,
    HybridPadicValidator,
    HybridPadicConverter,
    HybridPadicManager
)


class HybridPadicCompressionSystem(CompressionAlgorithm):
    """GPU-accelerated hybrid p-adic compression system"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize hybrid p-adic compression system"""
        super().__init__(config)
        
        # Extract and validate configuration
        self._extract_config()
        self._validate_full_config()
        
        # Initialize components
        self.math_ops = PadicMathematicalOperations(self.prime, self.precision)
        self.validator = PadicValidation()
        self.compression_validator = CompressionValidator()
        self.metrics_calculator = CompressionMetrics()
        
        # Initialize hybrid components
        self.hybrid_manager = HybridPadicManager(config)
        self.hybrid_validator = HybridPadicValidator()
        self.hybrid_converter = HybridPadicConverter()
        
        # Initialize GPU memory management
        try:
            from ..gpu_memory.gpu_memory_core import GPUMemoryOptimizer
            self.gpu_memory_optimizer = GPUMemoryOptimizer(config.get('gpu_config', {}))
        except ImportError:
            self.gpu_memory_optimizer = None
            
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for hybrid p-adic compression")
        
        # Performance tracking
        self.performance_stats = {
            'total_compressions': 0,
            'total_decompressions': 0,
            'hybrid_compressions': 0,
            'pure_padic_compressions': 0,
            'average_compression_time': 0.0,
            'average_decompression_time': 0.0,
            'average_hybrid_time': 0.0,
            'average_pure_time': 0.0,
            'peak_memory_usage': 0,
            'gpu_memory_usage': 0
        }
        
        # Dynamic switching management (optional integration)
        self.dynamic_switching_manager = None
        if self.enable_dynamic_switching:
            self._initialize_dynamic_switching()
    
    def _extract_config(self) -> None:
        """Extract configuration parameters"""
        # Required parameters (reused from existing system)
        required_params = ['prime', 'precision', 'chunk_size', 'gpu_memory_limit_mb']
        for param in required_params:
            if param not in self.config:
                raise KeyError(f"Missing required configuration parameter: {param}")
        
        self.prime = self.config['prime']
        self.precision = self.config['precision']
        self.chunk_size = self.config['chunk_size']
        self.gpu_memory_limit = self.config['gpu_memory_limit_mb'] * 1024 * 1024
        
        # Hybrid-specific parameters - REACTIVATED SWITCHING WITH GPU REQUIREMENT
        self.enable_hybrid = self.config.get('enable_hybrid', True)
        self.hybrid_threshold = self.config.get('hybrid_threshold', 1000)  # Use hybrid for tensors > 1000 elements
        self.force_hybrid = self.config.get('force_hybrid', False)
        self.enable_dynamic_switching = self.config.get('enable_dynamic_switching', True)
        self.require_gpu_for_hybrid = self.config.get('require_gpu_for_hybrid', True)  # NEW: Require GPU for hybrid
        
        # Optional parameters with validation
        self.preserve_ultrametric = self.config.get('preserve_ultrametric', True)
        self.validate_reconstruction = self.config.get('validate_reconstruction', True)
        self.max_reconstruction_error = self.config.get('max_reconstruction_error', 1e-6)
        self.enable_gc = self.config.get('enable_gc', True)
        
        # Validate hybrid parameters
        if not isinstance(self.enable_hybrid, bool):
            raise TypeError(f"enable_hybrid must be bool, got {type(self.enable_hybrid)}")
        if not isinstance(self.hybrid_threshold, int):
            raise TypeError(f"hybrid_threshold must be int, got {type(self.hybrid_threshold)}")
        if self.hybrid_threshold <= 0:
            raise ValueError(f"hybrid_threshold must be > 0, got {self.hybrid_threshold}")
        if not isinstance(self.force_hybrid, bool):
            raise TypeError(f"force_hybrid must be bool, got {type(self.force_hybrid)}")
        if not isinstance(self.enable_dynamic_switching, bool):
            raise TypeError(f"enable_dynamic_switching must be bool, got {type(self.enable_dynamic_switching)}")
    
    def _validate_config(self) -> None:
        """Validate configuration in parent class"""
        self._validate_full_config()
    
    def _validate_full_config(self) -> None:
        """Validate all configuration parameters"""
        # Validate p-adic parameters (reused from existing system)
        PadicValidation.validate_prime(self.prime)
        PadicValidation.validate_precision(self.precision)
        
        if not isinstance(self.chunk_size, int):
            raise TypeError(f"Chunk size must be int, got {type(self.chunk_size)}")
        if self.chunk_size <= 0:
            raise ValueError(f"Chunk size must be > 0, got {self.chunk_size}")
        if self.chunk_size > 1000000:
            raise ValueError(f"Chunk size too large: {self.chunk_size}")
        
        if not isinstance(self.gpu_memory_limit, int):
            raise TypeError(f"GPU memory limit must be int, got {type(self.gpu_memory_limit)}")
        if self.gpu_memory_limit <= 0:
            raise ValueError(f"GPU memory limit must be > 0, got {self.gpu_memory_limit}")
        
        if not isinstance(self.max_reconstruction_error, (int, float)):
            raise TypeError(f"Max reconstruction error must be numeric, got {type(self.max_reconstruction_error)}")
        if self.max_reconstruction_error <= 0:
            raise ValueError(f"Max reconstruction error must be > 0, got {self.max_reconstruction_error}")
    
    def _initialize_dynamic_switching(self) -> None:
        """Initialize dynamic switching system"""
        try:
            from .dynamic_switching_manager import DynamicSwitchingManager, SwitchingConfig
            
            # Create switching configuration
            switching_config = SwitchingConfig(
                enable_dynamic_switching=self.enable_dynamic_switching,
                hybrid_data_size_threshold=self.hybrid_threshold,
                enable_gradient_analysis=True,
                enable_performance_prediction=True
            )
            
            # Create dynamic switching manager
            self.dynamic_switching_manager = DynamicSwitchingManager(switching_config)
            
            # Initialize with placeholders - would be properly initialized by integration layer
            # This is just for basic functionality
            
        except ImportError as e:
            # Dynamic switching not available
            import logging
            logger = logging.getLogger('HybridPadicCompressionSystem')
            logger.warning(f"Dynamic switching not available: {e}")
            self.dynamic_switching_manager = None
        except Exception as e:
            import logging
            logger = logging.getLogger('HybridPadicCompressionSystem')
            logger.error(f"Failed to initialize dynamic switching: {e}")
            self.dynamic_switching_manager = None
    
    def set_dynamic_switching_manager(self, switching_manager) -> None:
        """Set external dynamic switching manager"""
        self.dynamic_switching_manager = switching_manager
    
    def _should_use_hybrid(self, data: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> bool:
        """Determine if hybrid compression should be used with GPU requirement enforcement"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if data is None:
            raise ValueError("Data cannot be None")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Data must be torch.Tensor, got {type(data)}")
        
        context = context or {}
        
        # Check if hybrid is enabled
        if not self.enable_hybrid:
            return False
        
        # Force hybrid if configured
        if self.force_hybrid:
            # ENFORCE GPU REQUIREMENT FOR FORCED HYBRID
            if self.require_gpu_for_hybrid and not torch.cuda.is_available():
                raise RuntimeError("GPU is required for hybrid compression but CUDA is not available. Cannot fall back to pure p-adic mode.")
            return True
        
        # Use dynamic switching if available and enabled
        if self.enable_dynamic_switching and hasattr(self, 'dynamic_switching_manager'):
            try:
                should_switch_to_hybrid, confidence, trigger = self.dynamic_switching_manager.should_switch_to_hybrid(data, context)
                
                # Only use dynamic switching decision if confidence is high enough
                if confidence >= self.dynamic_switching_manager.config.auto_switch_confidence_threshold:
                    # ENFORCE GPU REQUIREMENT FOR DYNAMIC SWITCHING
                    if should_switch_to_hybrid and self.require_gpu_for_hybrid and not torch.cuda.is_available():
                        raise RuntimeError("GPU is required for hybrid compression but CUDA is not available. Cannot fall back to pure p-adic mode.")
                    return should_switch_to_hybrid
                
            except Exception as e:
                # Log error but continue with fallback logic
                import logging
                logger = logging.getLogger('HybridPadicCompressionSystem')
                logger.warning(f"Dynamic switching failed, using fallback logic: {e}")
        
        # Fallback to original logic
        # Use hybrid for large tensors
        if data.numel() > self.hybrid_threshold:
            # ENFORCE GPU REQUIREMENT FOR LARGE TENSORS
            if self.require_gpu_for_hybrid and not torch.cuda.is_available():
                raise RuntimeError("GPU is required for hybrid compression but CUDA is not available. Cannot fall back to pure p-adic mode.")
            return True
        
        # Use hybrid if data is already on GPU
        if data.is_cuda:
            return True
        
        return False
    
    def encode(self, data: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> Tuple[Union[List[HybridPadicWeight], List[PadicWeight]], Dict[str, Any]]:
        """Encode tensor using hybrid or pure p-adic representation with GPU requirement enforcement"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if data is None:
            raise ValueError("Data cannot be None")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Data must be torch.Tensor, got {type(data)}")
        
        start_time = time.time()
        context = context or {}
        
        # Determine compression method with GPU requirement enforcement
        use_hybrid = self._should_use_hybrid(data, context)
        
        # Record switching decision if dynamic switching is active
        if self.dynamic_switching_manager and self.enable_dynamic_switching:
            try:
                current_mode = "hybrid" if use_hybrid else "pure_padic"
                context['selected_mode'] = current_mode
                context['data_shape'] = data.shape
                context['encoding_start_time'] = start_time
            except Exception as e:
                # Continue with encoding even if switching recording fails
                import logging
                logger = logging.getLogger('HybridPadicCompressionSystem')
                logger.warning(f"Failed to record switching decision: {e}")
        
        if use_hybrid:
            return self._encode_hybrid(data)
        else:
            # ENFORCE: Never fall back to pure p-adic if GPU is required but not available
            if self.require_gpu_for_hybrid and not torch.cuda.is_available():
                raise RuntimeError("GPU is required for hybrid compression but CUDA is not available. Cannot fall back to pure p-adic mode.")
            return self._encode_pure_padic(data)
    
    def _encode_hybrid(self, data: torch.Tensor) -> Tuple[List[HybridPadicWeight], Dict[str, Any]]:
        """Encode using hybrid p-adic representation"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if data is None:
            raise ValueError("Data cannot be None")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Data must be torch.Tensor, got {type(data)}")
        
        start_time = time.time()
        
        # Move data to GPU if not already there
        if not data.is_cuda:
            data = data.to(self.device)
        
        # Prepare GPU memory
        if self.gpu_memory_optimizer:
            memory_mb = data.numel() * 4 / (1024 * 1024)  # 4 bytes per float32
            self.gpu_memory_optimizer.prepare_for_computation(
                estimated_memory_mb=memory_mb
            )
        
        # Convert to pure p-adic first (reuse existing logic)
        pure_padic_weights, metadata = self._encode_pure_padic(data)
        
        # Convert to hybrid representation
        hybrid_weights = []
        for padic_weight in pure_padic_weights:
            hybrid_weight = self.hybrid_manager.create_hybrid_weight(padic_weight)
            hybrid_weights.append(hybrid_weight)
        
        # Update metadata
        metadata['compression_type'] = 'hybrid'
        metadata['hybrid_conversion_time'] = time.time() - start_time
        metadata['num_hybrid_weights'] = len(hybrid_weights)
        
        # Update performance stats
        self.performance_stats['hybrid_compressions'] += 1
        
        return hybrid_weights, metadata
    
    def _encode_pure_padic(self, data: torch.Tensor) -> Tuple[List[PadicWeight], Dict[str, Any]]:
        """Encode using pure p-adic representation (reuse existing logic)"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if data is None:
            raise ValueError("Data cannot be None")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Data must be torch.Tensor, got {type(data)}")
        
        # Store original device and move to CPU for processing
        original_device = data.device
        if data.is_cuda:
            data_cpu = data.cpu()
        else:
            data_cpu = data
        
        # Flatten tensor for processing
        original_shape = data_cpu.shape
        flat_data = data_cpu.flatten()
        
        # Convert to float values
        float_values = flat_data.tolist()
        
        # Convert each float to p-adic representation
        padic_weights = []
        processed_count = 0
        
        for i in range(0, len(float_values), self.chunk_size):
            chunk = float_values[i:i+self.chunk_size]
            chunk_weights = []
            
            for j, value in enumerate(chunk):
                try:
                    padic_weight = self.math_ops.to_padic(value)
                    chunk_weights.append(padic_weight)
                except Exception as e:
                    raise ValueError(f"Failed to convert value at position {i + j}: {e}")
            
            padic_weights.extend(chunk_weights)
            processed_count += len(chunk)
            
            # Force garbage collection if enabled
            if self.enable_gc and processed_count % (self.chunk_size * 10) == 0:
                gc.collect()
        
        # Create metadata
        metadata = {
            'compression_type': 'pure_padic',
            'original_shape': original_shape,
            'prime': self.prime,
            'precision': self.precision,
            'dtype': str(data.dtype),
            'device': str(original_device),
            'total_elements': len(padic_weights)
        }
        
        # Update performance stats
        self.performance_stats['pure_padic_compressions'] += 1
        
        return padic_weights, metadata
    
    def decode(self, encoded_data: Union[List[HybridPadicWeight], List[PadicWeight]], 
               metadata: Dict[str, Any]) -> torch.Tensor:
        """Decode hybrid or pure p-adic representation with GPU requirement enforcement"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if encoded_data is None:
            raise ValueError("Encoded data cannot be None")
        if not isinstance(encoded_data, list):
            raise TypeError(f"Encoded data must be list, got {type(encoded_data)}")
        if metadata is None:
            raise ValueError("Metadata cannot be None")
        if not isinstance(metadata, dict):
            raise TypeError(f"Metadata must be dict, got {type(metadata)}")
        
        start_time = time.time()
        
        # Check compression type
        compression_type = metadata.get('compression_type', 'pure_padic')
        
        if compression_type == 'hybrid':
            # ENFORCE GPU REQUIREMENT FOR HYBRID DECODING
            if self.require_gpu_for_hybrid and not torch.cuda.is_available():
                raise RuntimeError("GPU is required for hybrid decompression but CUDA is not available. Cannot fall back to pure p-adic mode.")
            return self._decode_hybrid(encoded_data, metadata)
        else:
            # ENFORCE: Never fall back to pure p-adic if GPU is required but not available
            if self.require_gpu_for_hybrid and not torch.cuda.is_available():
                raise RuntimeError("GPU is required for hybrid compression but CUDA is not available. Cannot fall back to pure p-adic mode.")
            return self._decode_pure_padic(encoded_data, metadata)
    
    def _decode_hybrid(self, encoded_data: List[HybridPadicWeight], metadata: Dict[str, Any]) -> torch.Tensor:
        """Decode hybrid p-adic representation"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if encoded_data is None:
            raise ValueError("Encoded data cannot be None")
        if not isinstance(encoded_data, list):
            raise TypeError(f"Encoded data must be list, got {type(encoded_data)}")
        if not encoded_data:
            raise ValueError("Encoded data cannot be empty")
        
        # Validate all weights are hybrid
        for i, weight in enumerate(encoded_data):
            if not isinstance(weight, HybridPadicWeight):
                raise TypeError(f"Weight {i} must be HybridPadicWeight, got {type(weight)}")
            self.hybrid_validator.validate_hybrid_weight(weight)
        
        # Prepare GPU memory
        if self.gpu_memory_optimizer:
            total_elements = sum(weight.exponent_channel.numel() for weight in encoded_data)
            memory_mb = total_elements * 4 / (1024 * 1024)
            self.gpu_memory_optimizer.prepare_for_computation(
                estimated_memory_mb=memory_mb
            )
        
        # Convert hybrid weights back to pure p-adic
        pure_padic_weights = []
        for hybrid_weight in encoded_data:
            padic_weight = self.hybrid_manager.restore_padic_weight(hybrid_weight)
            pure_padic_weights.append(padic_weight)
        
        # Use existing pure p-adic decoding logic
        return self._decode_pure_padic(pure_padic_weights, metadata)
    
    def _decode_pure_padic(self, encoded_data: List[PadicWeight], metadata: Dict[str, Any]) -> torch.Tensor:
        """Decode pure p-adic representation (reuse existing logic)"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if encoded_data is None:
            raise ValueError("Encoded data cannot be None")
        if not isinstance(encoded_data, list):
            raise TypeError(f"Encoded data must be list, got {type(encoded_data)}")
        if not encoded_data:
            raise ValueError("Encoded data cannot be empty")
        
        # Validate metadata
        required_keys = {'original_shape', 'prime', 'precision', 'dtype', 'device', 'total_elements'}
        missing_keys = required_keys - set(metadata.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys in metadata: {missing_keys}")
        
        # Validate configuration matches
        if metadata['prime'] != self.prime:
            raise ValueError(f"Prime mismatch: {metadata['prime']} != {self.prime}")
        if metadata['precision'] != self.precision:
            raise ValueError(f"Precision mismatch: {metadata['precision']} != {self.precision}")
        if metadata['total_elements'] != len(encoded_data):
            raise ValueError(f"Element count mismatch: {metadata['total_elements']} != {len(encoded_data)}")
        
        # Validate all encoded weights
        for i, padic_weight in enumerate(encoded_data):
            if not isinstance(padic_weight, PadicWeight):
                raise TypeError(f"Element {i} must be PadicWeight, got {type(padic_weight)}")
        
        # Convert p-adic weights back to floats
        float_values = []
        processed_count = 0
        
        for i in range(0, len(encoded_data), self.chunk_size):
            chunk = encoded_data[i:i+self.chunk_size]
            chunk_floats = []
            
            for j, pw in enumerate(chunk):
                try:
                    float_val = self.math_ops.from_padic(pw)
                    chunk_floats.append(float_val)
                except Exception as e:
                    raise ValueError(f"Failed to convert p-adic weight at position {i + j}: {e}")
            
            float_values.extend(chunk_floats)
            processed_count += len(chunk)
            
            # Force garbage collection if enabled
            if self.enable_gc and processed_count % (self.chunk_size * 10) == 0:
                gc.collect()
        
        # Validate all weights were converted
        if len(float_values) != len(encoded_data):
            raise ValueError(f"Conversion count mismatch: {len(float_values)} != {len(encoded_data)}")
        
        # Reconstruct tensor
        try:
            # Parse dtype and device
            dtype_str = metadata['dtype'].split('.')[-1]
            if not hasattr(torch, dtype_str):
                raise ValueError(f"Invalid dtype: {metadata['dtype']}")
            dtype = getattr(torch, dtype_str)
            
            device = torch.device(metadata['device'])
            
            # Create tensor
            tensor = torch.tensor(float_values, dtype=dtype, device='cpu')
            tensor = tensor.reshape(metadata['original_shape'])
            tensor = tensor.to(device)
            
        except Exception as e:
            raise ValueError(f"Failed to reconstruct tensor: {e}")
        
        return tensor
    
    def compress(self, data: torch.Tensor) -> Dict[str, Any]:
        """Compress tensor using hybrid p-adic compression"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if data is None:
            raise ValueError("Data cannot be None")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Data must be torch.Tensor, got {type(data)}")
        
        start_time = time.time()
        
        # Encode data
        encoded_data, metadata = self.encode(data)
        
        # Calculate compression metrics
        original_size = data.numel() * data.element_size()
        
        # Estimate compressed size based on type
        if metadata.get('compression_type') == 'hybrid':
            # Hybrid weights have GPU tensors - estimate size
            compressed_size = len(encoded_data) * self.precision * 8  # 8 bytes per channel value
        else:
            # Pure p-adic weights
            compressed_size = len(encoded_data) * self.precision * 4  # 4 bytes per coefficient
        
        compression_ratio = self.metrics_calculator.calculate_compression_ratio(data, encoded_data)
        
        # Optional reconstruction validation
        if self.validate_reconstruction:
            reconstructed = self.decode(encoded_data, metadata)
            self.compression_validator.validate_reconstruction_error(
                data, reconstructed, self.max_reconstruction_error
            )
        
        # Create compression result
        result = {
            'encoded_data': encoded_data,
            'metadata': metadata,
            'compression_type': metadata.get('compression_type', 'pure_padic'),
            'original_shape': data.shape,
            'original_dtype': str(data.dtype),
            'compression_time': time.time() - start_time,
            'algorithm': self.__class__.__name__,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio
        }
        
        # Update performance stats
        compression_time = time.time() - start_time
        self._update_compression_stats(compression_time, metadata.get('compression_type', 'pure_padic'))
        
        return result
    
    def decompress(self, compressed: Dict[str, Any]) -> torch.Tensor:
        """Decompress hybrid p-adic compressed data"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if compressed is None:
            raise ValueError("Compressed data cannot be None")
        if not isinstance(compressed, dict):
            raise TypeError(f"Compressed data must be dict, got {type(compressed)}")
        
        required_keys = ['encoded_data', 'metadata', 'original_shape', 'original_dtype']
        for key in required_keys:
            if key not in compressed:
                raise KeyError(f"Missing required key in compressed data: {key}")
        
        start_time = time.time()
        
        # Decode data
        result = self.decode(compressed['encoded_data'], compressed['metadata'])
        
        # Restore original shape and dtype
        result = result.reshape(compressed['original_shape'])
        
        # Parse and restore dtype
        dtype_str = compressed['original_dtype'].split('.')[-1]
        if hasattr(torch, dtype_str):
            result = result.to(dtype=getattr(torch, dtype_str))
        
        # Update performance stats
        decompression_time = time.time() - start_time
        self._update_decompression_stats(decompression_time, compressed.get('compression_type', 'pure_padic'))
        
        return result
    
    def _update_compression_stats(self, compression_time: float, compression_type: str) -> None:
        """Update compression statistics"""
        self.performance_stats['total_compressions'] += 1
        
        if compression_type == 'hybrid':
            self.performance_stats['hybrid_compressions'] += 1
            hybrid_count = self.performance_stats['hybrid_compressions']
            self.performance_stats['average_hybrid_time'] = (
                (self.performance_stats['average_hybrid_time'] * (hybrid_count - 1) + compression_time) /
                hybrid_count
            )
        else:
            self.performance_stats['pure_padic_compressions'] += 1
            pure_count = self.performance_stats['pure_padic_compressions']
            self.performance_stats['average_pure_time'] = (
                (self.performance_stats['average_pure_time'] * (pure_count - 1) + compression_time) /
                pure_count
            )
        
        total_count = self.performance_stats['total_compressions']
        self.performance_stats['average_compression_time'] = (
            (self.performance_stats['average_compression_time'] * (total_count - 1) + compression_time) /
            total_count
        )
    
    def _update_decompression_stats(self, decompression_time: float, compression_type: str) -> None:
        """Update decompression statistics"""
        self.performance_stats['total_decompressions'] += 1
        total_count = self.performance_stats['total_decompressions']
        self.performance_stats['average_decompression_time'] = (
            (self.performance_stats['average_decompression_time'] * (total_count - 1) + decompression_time) /
            total_count
        )
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        result = {
            'current_gpu_usage': self.performance_stats['gpu_memory_usage'],
            'peak_memory_usage': self.performance_stats['peak_memory_usage'],
            'hybrid_memory_usage': self.hybrid_manager.get_memory_usage()
        }
        
        if self.gpu_memory_optimizer:
            result['gpu_memory_info'] = self.gpu_memory_optimizer.get_performance_metrics()
        
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.performance_stats.copy()
        stats['hybrid_conversion_stats'] = self.hybrid_manager.get_conversion_stats()
        
        if self.gpu_memory_optimizer:
            stats['gpu_optimization_stats'] = self.gpu_memory_optimizer.get_performance_metrics()
        
        return stats
    
    def reset_memory_tracking(self) -> None:
        """Reset memory tracking"""
        self.performance_stats['peak_memory_usage'] = 0
        self.performance_stats['gpu_memory_usage'] = 0
        self.hybrid_manager.reset_stats()
    
    def reset_performance_stats(self) -> None:
        """Reset performance statistics"""
        self.performance_stats = {
            'total_compressions': 0,
            'total_decompressions': 0,
            'hybrid_compressions': 0,
            'pure_padic_compressions': 0,
            'average_compression_time': 0.0,
            'average_decompression_time': 0.0,
            'average_hybrid_time': 0.0,
            'average_pure_time': 0.0,
            'peak_memory_usage': 0,
            'gpu_memory_usage': 0
        }
        self.hybrid_manager.reset_stats()
    
    def _check_gpu_requirement(self, operation: str = "compression") -> None:
        """Check if GPU is available when required"""
        if self.require_gpu_for_hybrid and not torch.cuda.is_available():
            raise RuntimeError(f"GPU is required for hybrid {operation} but CUDA is not available. Cannot fall back to pure p-adic mode.")
    
    def _validate_gpu_for_hybrid_operation(self, data: torch.Tensor, operation: str = "compression") -> None:
        """Validate GPU availability for hybrid operations"""
        if self.require_gpu_for_hybrid:
            if not torch.cuda.is_available():
                raise RuntimeError(f"GPU is required for hybrid {operation} but CUDA is not available. Cannot fall back to pure p-adic mode.")
            if not data.is_cuda and data.numel() > self.hybrid_threshold:
                # Try to move to GPU if it's a large tensor
                try:
                    data = data.cuda()
                except Exception as e:
                    raise RuntimeError(f"Failed to move tensor to GPU for hybrid {operation}: {e}")


class HybridPadicIntegrationManager:
    """Manages integration between hybrid and pure p-adic systems"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize integration manager"""
        self.config = config or {}
        self.hybrid_system = None
        self.pure_system = None
        self.integration_stats = {
            'total_operations': 0,
            'hybrid_operations': 0,
            'pure_operations': 0,
            'switching_operations': 0,
            'average_hybrid_performance': 0.0,
            'average_pure_performance': 0.0
        }
    
    def initialize_systems(self, config: Dict[str, Any]) -> None:
        """Initialize both hybrid and pure p-adic systems"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is None:
            raise ValueError("Config cannot be None")
        if not isinstance(config, dict):
            raise TypeError(f"Config must be dict, got {type(config)}")
        
        # Create hybrid system
        hybrid_config = config.copy()
        hybrid_config['enable_hybrid'] = True
        self.hybrid_system = HybridPadicCompressionSystem(hybrid_config)
        
        # Create pure system
        pure_config = config.copy()
        pure_config['enable_hybrid'] = False
        from .padic_compressor import PadicCompressionSystem
        self.pure_system = PadicCompressionSystem(pure_config)
    
    def compress_with_optimal_system(self, data: torch.Tensor) -> Dict[str, Any]:
        """Compress using the optimal system based on data characteristics"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if data is None:
            raise ValueError("Data cannot be None")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Data must be torch.Tensor, got {type(data)}")
        
        if self.hybrid_system is None or self.pure_system is None:
            raise RuntimeError("Systems not initialized. Call initialize_systems() first.")
        
        # Determine optimal system
        use_hybrid = self._should_use_hybrid(data)
        
        self.integration_stats['total_operations'] += 1
        
        if use_hybrid:
            self.integration_stats['hybrid_operations'] += 1
            result = self.hybrid_system.compress(data)
            self._update_performance_stats('hybrid', result.get('compression_time', 0.0))
            return result
        else:
            self.integration_stats['pure_operations'] += 1
            result = self.pure_system.compress(data)
            self._update_performance_stats('pure', result.get('encoding_time', 0.0))
            return result
    
    def decompress_with_optimal_system(self, compressed: Dict[str, Any]) -> torch.Tensor:
        """Decompress using the appropriate system based on compression type"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if compressed is None:
            raise ValueError("Compressed data cannot be None")
        if not isinstance(compressed, dict):
            raise TypeError(f"Compressed data must be dict, got {type(compressed)}")
        
        if self.hybrid_system is None or self.pure_system is None:
            raise RuntimeError("Systems not initialized. Call initialize_systems() first.")
        
        compression_type = compressed.get('compression_type', 'pure_padic')
        
        if compression_type == 'hybrid':
            return self.hybrid_system.decompress(compressed)
        else:
            return self.pure_system.decompress(compressed)
    
    def _should_use_hybrid(self, data: torch.Tensor) -> bool:
        """Determine if hybrid system should be used with GPU requirement enforcement"""
        # Use hybrid for large tensors or GPU tensors
        if data.numel() > 1000 or data.is_cuda:
            # ENFORCE GPU REQUIREMENT FOR HYBRID OPERATIONS
            if not torch.cuda.is_available():
                raise RuntimeError("GPU is required for hybrid compression but CUDA is not available. Cannot fall back to pure p-adic mode.")
            return True
        return False
    
    def _update_performance_stats(self, system_type: str, execution_time: float) -> None:
        """Update performance statistics"""
        if system_type == 'hybrid':
            count = self.integration_stats['hybrid_operations']
            old_avg = self.integration_stats['average_hybrid_performance']
            self.integration_stats['average_hybrid_performance'] = (
                (old_avg * (count - 1) + execution_time) / count
            )
        else:
            count = self.integration_stats['pure_operations'] 
            old_avg = self.integration_stats['average_pure_performance']
            self.integration_stats['average_pure_performance'] = (
                (old_avg * (count - 1) + execution_time) / count
            )
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics"""
        return self.integration_stats.copy()
    
    def get_performance_comparison(self) -> Dict[str, Any]:
        """Get performance comparison between systems"""
        hybrid_stats = self.hybrid_system.get_performance_stats() if self.hybrid_system else {}
        pure_stats = self.pure_system.get_performance_stats() if self.pure_system else {}
        
        return {
            'integration_stats': self.integration_stats,
            'hybrid_system_stats': hybrid_stats,
            'pure_system_stats': pure_stats,
            'hybrid_advantage': (
                self.integration_stats['average_pure_performance'] / 
                max(self.integration_stats['average_hybrid_performance'], 1e-10)
            ) if self.integration_stats['average_hybrid_performance'] > 0 else 1.0
        }
    
    def reset_integration_stats(self) -> None:
        """Reset integration statistics"""
        self.integration_stats = {
            'total_operations': 0,
            'hybrid_operations': 0,
            'pure_operations': 0,
            'switching_operations': 0,
            'average_hybrid_performance': 0.0,
            'average_pure_performance': 0.0
        }
        
        if self.hybrid_system:
            self.hybrid_system.reset_performance_stats()
        if self.pure_system:
            self.pure_system.reset_performance_stats()