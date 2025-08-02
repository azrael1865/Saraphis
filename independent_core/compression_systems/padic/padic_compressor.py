"""
P-adic compression system for neural network weights.
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import time
import gc

from ..base.compression_base import CompressionAlgorithm, CompressionValidator, CompressionMetrics
from .padic_encoder import PadicWeight, PadicValidation, PadicMathematicalOperations


class PadicCompressionSystem(CompressionAlgorithm):
    """P-adic compression system for neural network weights"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize P-adic compression system"""
        super().__init__(config)
        
        # Extract and validate configuration
        self._extract_config()
        self._validate_full_config()
        
        # Initialize components
        self.math_ops = PadicMathematicalOperations(self.prime, self.precision)
        self.validator = PadicValidation()
        self.compression_validator = CompressionValidator()
        self.metrics_calculator = CompressionMetrics()
        
        # GPU memory management
        self.current_gpu_usage = 0
        self._compression_count = 0
        self._decompression_count = 0
        
        # Performance tracking
        self.performance_stats = {
            'total_compressions': 0,
            'total_decompressions': 0,
            'average_compression_time': 0.0,
            'average_decompression_time': 0.0,
            'peak_memory_usage': 0
        }
    
    def _extract_config(self) -> None:
        """Extract configuration parameters"""
        # Required parameters
        required_params = ['prime', 'precision', 'chunk_size', 'gpu_memory_limit_mb']
        for param in required_params:
            if param not in self.config:
                raise KeyError(f"Missing required configuration parameter: {param}")
        
        self.prime = self.config['prime']
        self.precision = self.config['precision']
        self.chunk_size = self.config['chunk_size']
        self.gpu_memory_limit = self.config['gpu_memory_limit_mb'] * 1024 * 1024
        
        # Optional parameters with validation
        self.preserve_ultrametric = self.config.get('preserve_ultrametric', True)
        self.validate_reconstruction = self.config.get('validate_reconstruction', True)
        self.max_reconstruction_error = self.config.get('max_reconstruction_error', 1e-6)
        self.enable_gc = self.config.get('enable_gc', True)
        
        if not isinstance(self.preserve_ultrametric, bool):
            raise TypeError(f"preserve_ultrametric must be bool, got {type(self.preserve_ultrametric)}")
        if not isinstance(self.validate_reconstruction, bool):
            raise TypeError(f"validate_reconstruction must be bool, got {type(self.validate_reconstruction)}")
    
    def _validate_config(self) -> None:
        """Validate configuration in parent class"""
        # This method is required by parent class
        self._validate_full_config()
    
    def _validate_full_config(self) -> None:
        """Validate all configuration parameters"""
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
    
    def encode(self, data: torch.Tensor) -> Tuple[List[PadicWeight], Dict[str, Any]]:
        """Encode tensor using p-adic representation"""
        # Validate input
        self.validator.validate_tensor(data)
        
        # Check GPU memory before processing
        tensor_size = data.element_size() * data.numel()
        self._check_gpu_memory(tensor_size)
        
        # Flatten tensor for processing
        original_shape = data.shape
        flat_data = data.flatten()
        
        # Validate chunk size
        PadicValidation.validate_chunk_size(self.chunk_size, flat_data.numel())
        
        # Convert to CPU for p-adic processing
        cpu_data = flat_data.cpu().numpy()
        
        # Process in chunks to manage memory
        padic_weights = []
        processed_elements = 0
        
        for i in range(0, len(cpu_data), self.chunk_size):
            chunk = cpu_data[i:i+self.chunk_size]
            chunk_padic = []
            
            for val in chunk:
                try:
                    padic_weight = self.math_ops.to_padic(float(val))
                    chunk_padic.append(padic_weight)
                except Exception as e:
                    raise ValueError(f"Failed to convert value {val} at position {i + len(chunk_padic)}: {e}")
            
            padic_weights.extend(chunk_padic)
            processed_elements += len(chunk)
            
            # Force garbage collection if enabled
            if self.enable_gc and processed_elements % (self.chunk_size * 10) == 0:
                gc.collect()
        
        # Validate all weights were processed
        if len(padic_weights) != flat_data.size:
            raise ValueError(f"Mismatch in processed weights: {len(padic_weights)} != {flat_data.size}")
        
        # Validate ultrametric property if required
        if self.preserve_ultrametric:
            self._validate_ultrametric_property(padic_weights)
        
        # Update memory tracking
        self.current_gpu_usage += tensor_size
        self.performance_stats['peak_memory_usage'] = max(
            self.performance_stats['peak_memory_usage'], 
            self.current_gpu_usage
        )
        
        # Create metadata
        metadata = {
            'original_shape': original_shape,
            'prime': self.prime,
            'precision': self.precision,
            'dtype': str(data.dtype),
            'device': str(data.device),
            'chunk_size': self.chunk_size,
            'total_elements': len(padic_weights),
            'compression_timestamp': time.time()
        }
        
        return padic_weights, metadata
    
    def decode(self, encoded_data: List[PadicWeight], metadata: Dict[str, Any]) -> torch.Tensor:
        """Decode p-adic representation back to tensor"""
        # Validate inputs
        if not isinstance(encoded_data, list):
            raise TypeError(f"Encoded data must be list, got {type(encoded_data)}")
        if not encoded_data:
            raise ValueError("Encoded data cannot be empty")
        if not isinstance(metadata, dict):
            raise TypeError(f"Metadata must be dict, got {type(metadata)}")
        
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
        
        # Update memory tracking
        tensor_size = tensor.element_size() * tensor.numel()
        if self.current_gpu_usage >= tensor_size:
            self.current_gpu_usage -= tensor_size
        else:
            # This shouldn't happen but handle gracefully by resetting
            self.current_gpu_usage = 0
        
        return tensor
    
    def _check_gpu_memory(self, required_bytes: int) -> None:
        """Check if GPU memory can accommodate new tensor"""
        if self.current_gpu_usage + required_bytes > self.gpu_memory_limit:
            raise RuntimeError(
                f"GPU memory limit exceeded: "
                f"current={self.current_gpu_usage}, required={required_bytes}, "
                f"total={self.current_gpu_usage + required_bytes}, limit={self.gpu_memory_limit}"
            )
    
    def _validate_ultrametric_property(self, padic_weights: List[PadicWeight]) -> None:
        """Validate ultrametric property for p-adic weights"""
        if len(padic_weights) < 3:
            return  # Need at least 3 points to check ultrametric
        
        # Sample random triplets to check
        num_checks = min(100, len(padic_weights) // 3)
        if num_checks == 0:
            return
        
        try:
            rng = np.random.RandomState(42)  # Fixed seed for reproducibility
            for _ in range(num_checks):
                # Select random triplet
                indices = rng.choice(len(padic_weights), 3, replace=False)
                x, y, z = [padic_weights[i] for i in indices]
                
                # Validate ultrametric property
                self.math_ops.validate_ultrametric_property(x, y, z)
                
        except Exception as e:
            raise ValueError(f"Ultrametric property validation failed: {e}")
    
    def compress(self, data: torch.Tensor) -> Dict[str, Any]:
        """Full compression pipeline with validation and metrics"""
        start_time = time.time()
        
        try:
            # Perform encoding
            encoded_data, metadata = self.encode(data)
            
            # Calculate compression metrics
            original_size = data.numel() * data.element_size()
            compressed_size = len(encoded_data) * self.precision * 4  # Approximate size
            
            compression_ratio = self.metrics_calculator.calculate_compression_ratio(data, encoded_data)
            
            # Validate compression ratio
            min_ratio = self.config.get('min_compression_ratio', 1.1)
            self.compression_validator.validate_compression_ratio(
                original_size, compressed_size, min_ratio
            )
            
            # Optional reconstruction validation
            if self.validate_reconstruction:
                reconstructed = self.decode(encoded_data, metadata)
                self.compression_validator.validate_reconstruction_error(
                    data, reconstructed, self.max_reconstruction_error
                )
            
            # Update statistics
            compression_time = time.time() - start_time
            self._update_compression_stats(compression_time)
            
            return {
                'encoded_data': encoded_data,
                'metadata': metadata,
                'encoding_time': compression_time,
                'algorithm': self.__class__.__name__,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio
            }
            
        except Exception as e:
            # Clean up memory on failure
            if hasattr(self, 'current_gpu_usage'):
                self.current_gpu_usage = max(0, self.current_gpu_usage - data.element_size() * data.numel())
            raise  # Re-raise original exception
    
    def decompress(self, compressed: Dict[str, Any]) -> torch.Tensor:
        """Full decompression pipeline"""
        start_time = time.time()
        
        try:
            # Validate compressed data structure
            required_keys = ['encoded_data', 'metadata']
            for key in required_keys:
                if key not in compressed:
                    raise KeyError(f"Missing required key in compressed data: {key}")
            
            # Perform decoding
            result = self.decode(compressed['encoded_data'], compressed['metadata'])
            
            # Update statistics
            decompression_time = time.time() - start_time
            self._update_decompression_stats(decompression_time)
            
            return result
            
        except Exception as e:
            raise  # Re-raise original exception
    
    def _update_compression_stats(self, compression_time: float) -> None:
        """Update compression performance statistics"""
        self.performance_stats['total_compressions'] += 1
        
        # Update rolling average
        total = self.performance_stats['total_compressions']
        current_avg = self.performance_stats['average_compression_time']
        self.performance_stats['average_compression_time'] = (
            (current_avg * (total - 1) + compression_time) / total
        )
    
    def _update_decompression_stats(self, decompression_time: float) -> None:
        """Update decompression performance statistics"""
        self.performance_stats['total_decompressions'] += 1
        
        # Update rolling average
        total = self.performance_stats['total_decompressions']
        current_avg = self.performance_stats['average_decompression_time']
        self.performance_stats['average_decompression_time'] = (
            (current_avg * (total - 1) + decompression_time) / total
        )
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current GPU memory usage"""
        return {
            'current_usage_bytes': self.current_gpu_usage,
            'limit_bytes': self.gpu_memory_limit,
            'usage_percentage': (self.current_gpu_usage / self.gpu_memory_limit) * 100,
            'peak_usage_bytes': self.performance_stats['peak_memory_usage']
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return dict(self.performance_stats)
    
    def reset_memory_tracking(self) -> None:
        """Reset GPU memory tracking"""
        self.current_gpu_usage = 0
        self.performance_stats['peak_memory_usage'] = 0
    
    def reset_performance_stats(self) -> None:
        """Reset all performance statistics"""
        self.performance_stats = {
            'total_compressions': 0,
            'total_decompressions': 0,
            'average_compression_time': 0.0,
            'average_decompression_time': 0.0,
            'peak_memory_usage': 0
        }