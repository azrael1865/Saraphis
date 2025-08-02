"""
Base compression algorithm interfaces and abstract classes.
All methods must be implemented - no default implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union
import torch
import time


class CompressionAlgorithm(ABC):
    """Base interface for all compression algorithms. Fail-loud design."""
    
    def __init__(self, config: Dict[str, Any]):
        if not isinstance(config, dict):
            raise TypeError(f"Config must be dict, got {type(config)}")
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate configuration. Must throw on invalid config."""
        pass
    
    @abstractmethod
    def encode(self, data: torch.Tensor) -> Tuple[Any, Dict[str, Any]]:
        """Encode data. Must throw on any error."""
        pass
    
    @abstractmethod
    def decode(self, encoded_data: Any, metadata: Dict[str, Any]) -> torch.Tensor:
        """Decode data. Must throw on any error."""
        pass
    
    def compress(self, data: torch.Tensor) -> Dict[str, Any]:
        """Full compression pipeline. No error handling."""
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Data must be torch.Tensor, got {type(data)}")
        
        start_time = time.time()
        encoded_data, metadata = self.encode(data)
        encoding_time = time.time() - start_time
        
        return {
            'encoded_data': encoded_data,
            'metadata': metadata,
            'encoding_time': encoding_time,
            'algorithm': self.__class__.__name__,
            'original_size': data.numel() * data.element_size()
        }
    
    def decompress(self, compressed: Dict[str, Any]) -> torch.Tensor:
        """Full decompression pipeline. No error handling."""
        required_keys = ['encoded_data', 'metadata']
        for key in required_keys:
            if key not in compressed:
                raise KeyError(f"Missing required key: {key}")
        
        return self.decode(compressed['encoded_data'], compressed['metadata'])


class AdaptiveCompressor(CompressionAlgorithm):
    """Base for adaptive compression algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._adaptation_history = []
        if 'max_history' not in config:
            raise KeyError("AdaptiveCompressor requires 'max_history' in config")
        self.max_history = config['max_history']
    
    @abstractmethod
    def adapt_parameters(self, performance_feedback: Dict[str, float]) -> None:
        """Adapt compression parameters. Must throw on invalid feedback."""
        pass
    
    def add_performance_feedback(self, feedback: Dict[str, float]) -> None:
        """Add performance feedback for adaptation."""
        required_metrics = ['compression_ratio', 'reconstruction_error', 'processing_time']
        for metric in required_metrics:
            if metric not in feedback:
                raise KeyError(f"Missing required performance metric: {metric}")
        
        self._adaptation_history.append(feedback)
        if len(self._adaptation_history) > self.max_history:
            self._adaptation_history.pop(0)
        
        self.adapt_parameters(feedback)


class CompressionValidator:
    """Validates compression operations. Strict validation only."""
    
    @staticmethod
    def validate_tensor_input(data: torch.Tensor) -> None:
        """Validate tensor input. Throws on any issue."""
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(data)}")
        
        if data.numel() == 0:
            raise ValueError("Empty tensor not allowed")
        
        if torch.isnan(data).any():
            raise ValueError("NaN values in input tensor")
        
        if torch.isinf(data).any():
            raise ValueError("Infinite values in input tensor")
    
    @staticmethod
    def validate_compression_ratio(original_size: int, compressed_size: int, 
                                  min_ratio: float = 1.1) -> None:
        """Validate compression achieved minimum ratio."""
        if compressed_size >= original_size:
            raise ValueError(f"No compression achieved: {compressed_size} >= {original_size}")
        
        ratio = original_size / compressed_size
        if ratio < min_ratio:
            raise ValueError(f"Compression ratio {ratio:.2f} below minimum {min_ratio}")
    
    @staticmethod
    def validate_reconstruction_error(original: torch.Tensor, reconstructed: torch.Tensor,
                                    max_error: float = 0.01) -> None:
        """Validate reconstruction error within bounds."""
        if original.shape != reconstructed.shape:
            raise ValueError(f"Shape mismatch: {original.shape} != {reconstructed.shape}")
        
        mse = torch.nn.functional.mse_loss(original, reconstructed).item()
        if mse > max_error:
            raise ValueError(f"Reconstruction error {mse:.6f} exceeds maximum {max_error}")


class CompressionMetrics:
    """Calculate compression metrics. No approximations."""
    
    @staticmethod
    def calculate_compression_ratio(original: torch.Tensor, compressed: Any) -> float:
        """Calculate exact compression ratio."""
        original_bytes = original.numel() * original.element_size()
        
        if isinstance(compressed, torch.Tensor):
            compressed_bytes = compressed.numel() * compressed.element_size()
        elif isinstance(compressed, dict):
            compressed_bytes = sum(
                t.numel() * t.element_size() if isinstance(t, torch.Tensor) else len(str(t))
                for t in compressed.values()
            )
        else:
            compressed_bytes = len(str(compressed))
        
        if compressed_bytes == 0:
            raise ValueError("Compressed size cannot be zero")
        
        return original_bytes / compressed_bytes
    
    @staticmethod
    def calculate_reconstruction_error(original: torch.Tensor, 
                                     reconstructed: torch.Tensor) -> Dict[str, float]:
        """Calculate multiple error metrics."""
        if original.shape != reconstructed.shape:
            raise ValueError("Tensor shapes must match for error calculation")
        
        mse = torch.nn.functional.mse_loss(original, reconstructed).item()
        mae = torch.nn.functional.l1_loss(original, reconstructed).item()
        
        # Relative error
        original_norm = torch.norm(original).item()
        if original_norm == 0:
            raise ValueError("Original tensor has zero norm")
        
        error_norm = torch.norm(original - reconstructed).item()
        relative_error = error_norm / original_norm
        
        return {
            'mse': mse,
            'mae': mae,
            'relative_error': relative_error,
            'max_absolute_error': torch.max(torch.abs(original - reconstructed)).item()
        }