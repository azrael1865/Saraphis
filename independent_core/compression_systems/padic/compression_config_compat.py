"""
Compatibility layer for CompressionConfig
Provides backwards compatibility with test interface
"""

from dataclasses import dataclass
from typing import Optional

@dataclass 
class CompressionConfig:
    """Compatibility CompressionConfig for integration tests"""
    # Core P-adic parameters
    prime: int = 257
    base_precision: int = 4
    min_precision: int = 2
    max_precision: int = 4
    
    # Adaptive precision
    target_error: float = 1e-6
    importance_threshold: float = 0.1
    compression_priority: float = 0.5
    
    # Pattern detection
    min_pattern_length: int = 4
    max_pattern_length: int = 32
    min_pattern_frequency: int = 3
    pattern_hash_prime: int = 31
    
    # Sparse encoding
    sparsity_threshold: float = 1e-6
    target_sparsity: float = 0.95
    optimize_patterns: bool = True
    
    # Entropy coding
    huffman_threshold: float = 2.0
    arithmetic_threshold: float = 6.0
    enable_hybrid_entropy: bool = True
    
    # Memory and performance
    chunk_size: int = 10000
    max_tensor_size: int = 1_000_000
    enable_gpu: bool = True
    gpu_memory_limit_mb: int = 1024
    enable_memory_monitoring: bool = True
    
    # Device configuration (for integration test compatibility)
    compression_device: str = "cpu"
    decompression_device: str = "cpu"
    enable_device_fallback: bool = True
    
    # Test-specific parameters
    enable_parallel: bool = False
    
    # Validation and safety
    validate_reconstruction: bool = True
    max_reconstruction_error: float = 1e-5
    enable_logging: bool = True
    raise_on_error: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.prime <= 1:
            raise ValueError(f"Prime must be > 1, got {self.prime}")
        if self.base_precision <= 0:
            raise ValueError(f"Base precision must be > 0, got {self.base_precision}")
        if not (0 < self.target_error < 1):
            raise ValueError(f"Target error must be in (0, 1), got {self.target_error}")
        if not (1 <= self.min_precision <= self.base_precision <= self.max_precision):
            raise ValueError(
                f"Invalid precision bounds: min={self.min_precision}, "
                f"base={self.base_precision}, max={self.max_precision}"
            )
        if not (0 <= self.compression_priority <= 1):
            raise ValueError(f"Compression priority must be in [0, 1], got {self.compression_priority}")
        if self.chunk_size <= 0:
            raise ValueError(f"Chunk size must be positive, got {self.chunk_size}")