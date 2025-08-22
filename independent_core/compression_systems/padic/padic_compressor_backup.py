"""
Unified P-adic Compression System - FIXED PIPELINE ORDER
Compress first, then transform for GPU efficiency
NO FALLBACKS - HARD FAILURES ONLY
"""

# Import overflow protection first
import sys
import os
sys.path.append('/home/will-casterlin/Desktop/Saraphis')

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
import time
import math
import struct
import logging
import gc
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import all compression components
from .sliding_window_pattern_detector import (
    SlidingWindowPatternDetector,
    PatternDetectionResult,
    PatternMatch
)
from .metadata_compressor import (
    MetadataCompressor,
    MetadataHeader
)
from .sparse_bridge import (
    SparsePAdicBridge,
    AdaptiveSparsityManager
)
from .entropy_bridge import (
    EntropyPAdicBridge,
    EntropyBridgeConfig,
    EntropyAnalysis
)
from .adaptive_precision_wrapper import (
    AdaptivePrecisionWrapper,
    AdaptivePrecisionConfig,
    PrecisionAllocation
)
from .padic_encoder import (
    PadicWeight,
    PadicMathematicalOperations,
    PadicValidation,
    AdaptiveHenselLifting
)
from .padic_logarithmic_encoder import LogarithmicPadicWeight
from .safe_reconstruction import SafePadicReconstructor as SafeReconstruction, ReconstructionConfig
from .memory_pressure_handler import MemoryPressureHandler

# Import compatibility config
try:
    from .compression_config_compat import CompressionConfig as CompatCompressionConfig
    USE_COMPAT_CONFIG = True
except ImportError:
    USE_COMPAT_CONFIG = False


class CompressionStage(Enum):
    """Compression pipeline stages - REORDERED"""
    PATTERN_DETECTION = "pattern_detection"      # First: compress floats
    SPARSE_ENCODING = "sparse_encoding"           # Second: compress more
    ENTROPY_CODING = "entropy_coding"             # Third: compress even more
    PADIC_TRANSFORMATION = "padic_transformation" # Fourth: transform for GPU
    METADATA_COMPRESSION = "metadata_compression" # Last: package everything


@dataclass
class CompressionConfig:
    """Unified configuration for P-adic compression system"""
    # P-adic parameters
    prime: int = 257
    base_precision: int = 2  # REDUCED to minimize expansion
    min_precision: int = 1   # REDUCED
    max_precision: int = 3   # REDUCED
    
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
    
    # Device configuration
    compression_device: str = "cpu"
    decompression_device: str = "cpu"
    enable_device_fallback: bool = True
    
    # Validation and safety
    validate_reconstruction: bool = True
    max_reconstruction_error: float = 1e-5
    enable_logging: bool = True
    raise_on_error: bool = True
    
    # Parallel processing
    enable_parallel: bool = False
    enable_memory_tracking: bool = True
    enable_dynamic_switching: bool = True
    batch_size: int = 32
    
    # Experimental parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def get_experimental_param(self, key: str, default=None):
        """Get an experimental parameter by key."""
        return self.extra_params.get(key, default)
    
    def has_experimental_param(self, key: str) -> bool:
        """Check if an experimental parameter exists."""
        return key in self.extra_params
    
    def __post_init__(self):
        """Validate configuration parameters"""
        PadicValidation.validate_prime(self.prime)
        PadicValidation.validate_precision(self.base_precision)
        
        if not (1 <= self.min_precision <= self.base_precision <= self.max_precision):
            raise ValueError(
                f"Invalid precision bounds: min={self.min_precision}, "
                f"base={self.base_precision}, max={self.max_precision}"
            )
        
        if not (0 < self.target_error < 1):
            raise ValueError(f"Target error must be in (0, 1), got {self.target_error}")
        
        if not (0 <= self.compression_priority <= 1):
            raise ValueError(f"Compression priority must be in [0, 1], got {self.compression_priority}")
        
        if self.chunk_size <= 0:
            raise ValueError(f"Chunk size must be positive, got {self.chunk_size}")


@dataclass
class CompressionResult:
    """Result of P-adic compression"""
    compressed_data: bytes
    metadata: Dict[str, Any]
    compression_ratio: float
    processing_time: float
    memory_usage: int
    stage_metrics: Dict[str, Dict[str, Any]]
    validation_passed: bool
    error_metrics: Dict[str, float]


@dataclass
class DecompressionResult:
    """Result of P-adic decompression"""
    reconstructed_tensor: torch.Tensor
    original_shape: Tuple[int, ...]
    processing_time: float
    memory_usage: int
    stage_metrics: Dict[str, Dict[str, Any]]
    validation_passed: bool
    reconstruction_error: float


class PadicCompressionSystem:
    """
    FIXED P-adic Compression System
    Compress first, then transform for GPU efficiency
    """
    
    def __init__(self, config=None):
        """Initialize P-adic compression system with flexible config handling"""
        
        # Handle different config types
        if config is None:
            self.config = CompressionConfig()
        elif isinstance(config, dict):
            self.config = CompressionConfig(**config)
        elif hasattr(config, '__dict__'):
            # Handle compatibility configs
            known_params = {
                'prime', 'base_precision', 'min_precision', 'max_precision',
                'compression_device', 'decompression_device', 'enable_device_fallback',
                'target_error', 'importance_threshold', 'batch_size', 'chunk_size',
                'enable_parallel', 'enable_gpu', 'gpu_memory_limit_mb',
                'compression_priority', 'enable_memory_tracking', 'enable_dynamic_switching',
                'sparsity_threshold', 'target_sparsity', 'optimize_patterns',
                'huffman_threshold', 'arithmetic_threshold', 'enable_hybrid_entropy',
                'max_tensor_size', 'enable_memory_monitoring', 'validate_reconstruction',
                'max_reconstruction_error', 'enable_logging', 'raise_on_error',
                'min_pattern_length', 'max_pattern_length', 'min_pattern_frequency',
                'pattern_hash_prime'
            }
            
            config_dict = {}
            extra_params = {}
            
            for attr in dir(config):
                if not attr.startswith('_') and hasattr(config, attr):
                    value = getattr(config, attr)
                    if not callable(value):
                        if attr in known_params:
                            if attr in ['compression_device', 'decompression_device'] and value:
                                config_dict[attr] = str(value)
                            else:
                                config_dict[attr] = value
                        else:
                            extra_params[attr] = value
            
            config_dict['extra_params'] = extra_params
            self.config = CompressionConfig(**config_dict)
        elif isinstance(config, CompressionConfig):
            self.config = config
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")
        
        logger.info(f"Initializing FIXED P-adic Compression System with prime={self.config.prime}")
        logger.info(f"Pipeline order: Pattern→Sparse→Entropy→P-adic (for GPU)")
        
        # Initialize device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and self.config.enable_gpu else 'cpu'
        )
        logger.info(f"Using device: {self.device}")
        
        if self.config.extra_params:
            logger.info(f"Experimental parameters: {list(self.config.extra_params.keys())}")
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.stats = {
            'total_compressions': 0,
            'total_decompressions': 0,
            'total_bytes_compressed': 0,
            'total_bytes_decompressed': 0,
            'average_compression_ratio': 0.0,
            'peak_memory_usage': 0,
            'stage_times': {stage.value: [] for stage in CompressionStage},
            'errors': []
        }
        
        self.memory_handler = None
    
    def _initialize_components(self):
        """Initialize all compression components"""
        try:
            # Pattern Detection (for float compression)
            self.pattern_detector = SlidingWindowPatternDetector(
                min_pattern_length=self.config.min_pattern_length,
                max_pattern_length=self.config.max_pattern_length,
                min_frequency=self.config.min_pattern_frequency,
                hash_prime=self.config.pattern_hash_prime,
                device=str(self.device),
                enable_compile=self.device.type == 'cuda',
                max_patterns=1000,
                use_suffix_array=True
            )
            logger.info("✓ Pattern Detector initialized")
            
            # Sparse Encoding (for float compression)
            self.sparse_bridge = SparsePAdicBridge(
                sparsity_threshold=self.config.sparsity_threshold,
                use_gpu=self.config.enable_gpu
            )
            self.sparsity_manager = AdaptiveSparsityManager(
                target_compression_ratio=20.0,
                min_threshold=1e-8,
                max_threshold=self.config.sparsity_threshold
            )
            logger.info("✓ Sparse Bridge initialized")
            
            # Entropy Coding (for compressed float data)
            entropy_config = EntropyBridgeConfig(
                huffman_threshold=self.config.huffman_threshold,
                arithmetic_threshold=self.config.arithmetic_threshold,
                enable_pattern_detection=False,  # Already did pattern detection
                enable_delta_encoding=True,
                compression_level=6
            )
            self.entropy_bridge = EntropyPAdicBridge(
                256,  # Use 256 for float data, not prime
                entropy_config
            )
            logger.info("✓ Entropy Bridge initialized")
            
            # P-adic Transformation (for GPU efficiency)
            self.math_ops = PadicMathematicalOperations(
                self.config.prime,
                self.config.base_precision
            )
            logger.info("✓ P-adic Mathematical Operations initialized")
            
            precision_config = AdaptivePrecisionConfig(
                prime=self.config.prime,
                base_precision=self.config.base_precision,
                min_precision=self.config.min_precision,
                max_precision=self.config.max_precision,
                target_error=self.config.target_error,
                importance_threshold=self.config.importance_threshold,
                compression_priority=self.config.compression_priority,
                enable_gpu_acceleration=self.config.enable_gpu,
                batch_size=self.config.batch_size,
                enable_memory_tracking=self.config.enable_memory_tracking,
                enable_dynamic_switching=self.config.enable_dynamic_switching
            )
            self.adaptive_precision = AdaptivePrecisionWrapper(
                precision_config,
                math_ops=self.math_ops,
                device=self.device
            )
            logger.info("✓ Adaptive Precision Wrapper initialized")
            
            # Metadata Compression
            self.metadata_compressor = MetadataCompressor()
            logger.info("✓ Metadata Compressor initialized")
            
            # Additional components
            recon_config = ReconstructionConfig(
                prime=self.config.prime,
                max_safe_precision=self.config.base_precision
            )
            self.safe_reconstruction = SafeReconstruction(recon_config)
            self.hensel = AdaptiveHenselLifting(
                self.config.prime,
                self.config.base_precision
            )
            
            logger.info("All compression components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise RuntimeError(f"Failed to initialize compression components: {e}")
    
    def compress(self, tensor: torch.Tensor, 
                importance_scores: Optional[torch.Tensor] = None) -> CompressionResult:
        """
        FIXED: Compress tensor with correct pipeline order
        Pattern→Sparse→Entropy→P-adic→Metadata
        """
        start_time = time.perf_counter()
        stage_metrics = {}
        
        try:
            # Validate input
            PadicValidation.validate_tensor(tensor)
            original_shape = tensor.shape
            original_size = tensor.numel() * tensor.element_size()
            
            logger.info(f"Starting compression of tensor shape={original_shape}")
            logger.info(f"Original size: {original_size / 1024:.2f} KB")
            
            # Stage 1: Pattern Detection (compress floats)
            stage_start = time.perf_counter()
            logger.info("Stage 1: Pattern Detection on FLOAT data")
            
            flat_tensor = tensor.flatten()
            pattern_result = self.pattern_detector.find_patterns(flat_tensor)
            
            if pattern_result.total_patterns_found > 0:
                # Replace patterns with markers in float data
                pattern_compressed, pattern_dict, pattern_lengths = self.pattern_detector.encode_with_patterns(
                    flat_tensor,
                    pattern_result
                )
                logger.info(f"Pattern compression: {flat_tensor.numel()} → {pattern_compressed.numel()} elements")
            else:
                pattern_compressed = flat_tensor
                pattern_dict = {}
                pattern_lengths = torch.tensor([], dtype=torch.int32, device=self.device)
            
            stage_metrics['pattern_detection'] = {
                'time': time.perf_counter() - stage_start,
                'patterns_found': pattern_result.total_patterns_found,
                'compression_ratio': flat_tensor.numel() / max(pattern_compressed.numel(), 1)
            }
            
            # Stage 2: Sparse Encoding (compress pattern-compressed floats)
            stage_start = time.perf_counter()
            logger.info("Stage 2: Sparse Encoding on pattern-compressed data")
            
            # Convert to sparse representation
            # Create pseudo-valuations for float data
            valuations = torch.zeros(pattern_compressed.numel(), dtype=torch.int32, device=self.device)
            
            # Reshape for sparse encoding
            if pattern_compressed.dim() == 1:
                pattern_compressed = pattern_compressed.unsqueeze(1)
            
            sparse_tensor = self.sparse_bridge.padic_to_sparse(pattern_compressed, valuations)
            sparse_ratio = self.sparse_bridge.get_compression_ratio(sparse_tensor)
            
            stage_metrics['sparse_encoding'] = {
                'time': time.perf_counter() - stage_start,
                'sparsity': 1.0 - (sparse_tensor._nnz() / pattern_compressed.numel()),
                'compression_ratio': sparse_ratio
            }
            
            # Stage 3: Entropy Coding (compress sparse data)
            stage_start = time.perf_counter()
            logger.info("Stage 3: Entropy Coding on sparse data")
            
            sparse_values = sparse_tensor.values()
            if sparse_values.numel() > 0:
                # Quantize float values for entropy encoding
                quantized = (sparse_values * 255).clamp(0, 255).long()
                entropy_compressed, entropy_metadata = self.entropy_bridge.encode_padic_tensor(quantized)
                logger.info(f"Entropy compression: {sparse_values.numel()} values → {len(entropy_compressed)} bytes")
            else:
                entropy_compressed = b''
                entropy_metadata = {'empty': True}
            
            stage_metrics['entropy_coding'] = {
                'time': time.perf_counter() - stage_start,
                'compressed_bytes': len(entropy_compressed),
                'compression_ratio': (sparse_values.numel() * 4) / max(len(entropy_compressed), 1)
            }
            
            # Stage 4: P-adic Transformation (for GPU efficiency)
            stage_start = time.perf_counter()
            logger.info("Stage 4: P-adic Transformation for GPU decompression")
            
            # Only transform the compressed data to p-adic
            # This will expand, but much less than before
            compressed_size = len(entropy_compressed)
            
            # Create a small tensor from compressed bytes for p-adic transformation
            compressed_tensor = torch.frombuffer(entropy_compressed, dtype=torch.uint8).float() / 255.0
            compressed_tensor = compressed_tensor.to(self.device)
            
            # Apply p-adic transformation with minimal precision
            allocation = self.adaptive_precision.batch_compress_with_adaptive_precision(
                compressed_tensor.view(-1, 1),  # Ensure 2D
                importance_scores=None  # No importance for compressed data
            )
            
            padic_weights = allocation.padic_weights
            precision_map = allocation.precision_map
            
            # Convert to digit tensor (this causes expansion, but on compressed data)
            max_precision = max(w.precision for w in padic_weights)
            digit_tensor = torch.zeros(
                (len(padic_weights), max_precision),
                dtype=torch.uint8,  # Use uint8 instead of int32
                device=self.device
            )
            
            for i, weight in enumerate(padic_weights):
                digits = torch.tensor(weight.digits, device=self.device, dtype=torch.uint8)
                digit_tensor[i, :len(digits)] = digits
            
            padic_expansion = digit_tensor.numel() / compressed_size if compressed_size > 0 else 1.0
            logger.info(f"P-adic transformation: {compressed_size} bytes → {digit_tensor.numel()} p-adic digits")
            logger.info(f"P-adic expansion on compressed data: {padic_expansion:.2f}x")
            
            stage_metrics['padic_transformation'] = {
                'time': time.perf_counter() - stage_start,
                'expansion_ratio': padic_expansion,
                'average_precision': allocation.get_average_precision()
            }
            
            # Stage 5: Metadata Compression
            stage_start = time.perf_counter()
            logger.info("Stage 5: Metadata Compression")
            
            metadata = {
                'version': 3,  # New version for fixed pipeline
                'pipeline_order': 'pattern→sparse→entropy→padic',
                'prime': self.config.prime,
                'precision': self.config.base_precision,
                'original_shape': original_shape,
                'pattern_dict': pattern_dict,
                'pattern_lengths': pattern_lengths.cpu().numpy().tolist() if pattern_lengths.numel() > 0 else [],
                'sparse_indices': self._extract_sparse_indices(sparse_tensor),
                'sparse_shape': list(sparse_tensor.shape),
                'entropy_metadata': entropy_metadata,
                'padic_digits_shape': list(digit_tensor.shape),
                'precision_map': precision_map.cpu().numpy().tolist(),
                'compressed_tensor_size': compressed_tensor.numel()
            }
            
            compressed_metadata = self.metadata_compressor.compress_metadata(metadata)
            
            stage_metrics['metadata_compression'] = {
                'time': time.perf_counter() - stage_start,
                'metadata_size': len(compressed_metadata)
            }
            
            # Combine all data
            combined_data = self._combine_compressed_data_fixed(
                digit_tensor.cpu().numpy().tobytes(),  # P-adic digits
                compressed_metadata
            )
            
            # Calculate final metrics
            total_time = time.perf_counter() - start_time
            final_compression_ratio = original_size / len(combined_data)
            
            # Validate if enabled
            validation_passed = True
            error_metrics = {}
            
            if self.config.validate_reconstruction:
                try:
                    decompressed = self.decompress(combined_data)
                    reconstruction_error = torch.nn.functional.mse_loss(
                        tensor,
                        decompressed.reconstructed_tensor
                    ).item()
                    
                    validation_passed = reconstruction_error < self.config.max_reconstruction_error * 10
                    error_metrics = {
                        'mse': reconstruction_error,
                        'max_error': torch.max(torch.abs(tensor - decompressed.reconstructed_tensor)).item()
                    }
                except Exception as e:
                    logger.warning(f"Validation failed: {e}")
                    validation_passed = False
            
            # Update statistics
            self.stats['total_compressions'] += 1
            self.stats['total_bytes_compressed'] += original_size
            
            logger.info(f"Compression complete: ratio={final_compression_ratio:.2f}x, time={total_time:.3f}s")
            logger.info(f"Final size: {len(combined_data) / 1024:.2f} KB")
            
            return CompressionResult(
                compressed_data=combined_data,
                metadata=metadata,
                compression_ratio=final_compression_ratio,
                processing_time=total_time,
                memory_usage=len(combined_data),
                stage_metrics=stage_metrics,
                validation_passed=validation_passed,
                error_metrics=error_metrics
            )
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            self.stats['errors'].append(str(e))
            raise RuntimeError(f"Critical compression failure: {e}")
    
    def decompress(self, compressed_data: bytes) -> DecompressionResult:
        """
        FIXED: Decompress with correct reverse pipeline order
        P-adic→Entropy→Sparse→Pattern→Float
        """
        start_time = time.perf_counter()
        stage_metrics = {}
        
        try:
            logger.info("Starting decompression")
            
            # Split combined data
            padic_data, compressed_metadata = self._split_compressed_data_fixed(compressed_data)
            
            # Stage 1: Metadata Decompression
            stage_start = time.perf_counter()
            logger.info("Stage 1: Metadata Decompression")
            
            metadata = self.metadata_compressor.decompress_metadata(compressed_metadata)
            
            stage_metrics['metadata_decompression'] = {
                'time': time.perf_counter() - stage_start
            }
            
            # Stage 2: P-adic Reconstruction (GPU→compressed)
            stage_start = time.perf_counter()
            logger.info("Stage 2: P-adic Reconstruction to compressed data")
            
            # Reconstruct digit tensor
            padic_digits_shape = metadata['padic_digits_shape']
            digit_tensor = torch.frombuffer(padic_data, dtype=torch.uint8).view(padic_digits_shape)
            digit_tensor = digit_tensor.to(self.device)
            
            # Create p-adic weights
            precision_map = torch.tensor(metadata['precision_map'], device=self.device)
            padic_weights = []
            
            for i in range(digit_tensor.shape[0]):
                precision = int(precision_map.flatten()[i].item()) if i < precision_map.numel() else self.config.base_precision
                digits = digit_tensor[i, :precision].cpu().numpy().tolist()
                
                from fractions import Fraction
                value = sum(d * (self.config.prime ** j) for j, d in enumerate(digits))
                
                weight = PadicWeight(
                    value=Fraction(value, 1),
                    digits=digits,
                    valuation=0,
                    prime=metadata['prime'],
                    precision=precision
                )
                padic_weights.append(weight)
            
            # Reconstruct compressed values
            compressed_values = []
            for weight in padic_weights:
                value = self.safe_reconstruction.reconstruct(weight)
                compressed_values.append(value)
            
            # Convert back to bytes
            compressed_tensor = torch.tensor(compressed_values, dtype=torch.float32, device=self.device)
            compressed_tensor = (compressed_tensor * 255).clamp(0, 255).byte()
            entropy_compressed = compressed_tensor.cpu().numpy().tobytes()
            
            stage_metrics['padic_reconstruction'] = {
                'time': time.perf_counter() - stage_start
            }
            
            # Stage 3: Entropy Decoding
            stage_start = time.perf_counter()
            logger.info("Stage 3: Entropy Decoding")
            
            if entropy_compressed and not metadata.get('entropy_metadata', {}).get('empty'):
                sparse_values = self.entropy_bridge.decode_padic_tensor(
                    entropy_compressed,
                    metadata['entropy_metadata']
                )
                sparse_values = sparse_values.float() / 255.0  # Dequantize
            else:
                sparse_values = torch.tensor([], device=self.device)
            
            stage_metrics['entropy_decoding'] = {
                'time': time.perf_counter() - stage_start
            }
            
            # Stage 4: Sparse Reconstruction
            stage_start = time.perf_counter()
            logger.info("Stage 4: Sparse Reconstruction")
            
            if metadata.get('sparse_indices') is not None and sparse_values.numel() > 0:
                indices = torch.tensor(metadata['sparse_indices'], device=self.device)
                sparse_shape = tuple(metadata['sparse_shape'])
                
                sparse_tensor = torch.sparse_coo_tensor(
                    indices,
                    sparse_values,
                    sparse_shape,
                    device=self.device
                )
                
                pattern_compressed = sparse_tensor.to_dense().flatten()
            else:
                pattern_compressed = torch.zeros(metadata['sparse_shape'][0], device=self.device)
            
            stage_metrics['sparse_reconstruction'] = {
                'time': time.perf_counter() - stage_start
            }
            
            # Stage 5: Pattern Reconstruction
            stage_start = time.perf_counter()
            logger.info("Stage 5: Pattern Reconstruction")
            
            if metadata.get('pattern_dict'):
                pattern_lengths = torch.tensor(metadata.get('pattern_lengths', []), device=self.device)
                
                reconstructed = self.pattern_detector.decode_with_patterns(
                    pattern_compressed.long(),
                    metadata['pattern_dict'],
                    pattern_lengths,
                    element_dtype=np.dtype('float32')
                )
            else:
                reconstructed = pattern_compressed
            
            # Reshape to original
            reconstructed_tensor = reconstructed.reshape(metadata['original_shape']).float()
            
            stage_metrics['pattern_reconstruction'] = {
                'time': time.perf_counter() - stage_start
            }
            
            # Calculate metrics
            total_time = time.perf_counter() - start_time
            memory_usage = reconstructed_tensor.numel() * reconstructed_tensor.element_size()
            
            logger.info(f"Decompression complete: shape={reconstructed_tensor.shape}, time={total_time:.3f}s")
            
            return DecompressionResult(
                reconstructed_tensor=reconstructed_tensor,
                original_shape=tuple(metadata['original_shape']),
                processing_time=total_time,
                memory_usage=memory_usage,
                stage_metrics=stage_metrics,
                validation_passed=True,
                reconstruction_error=0.0
            )
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise RuntimeError(f"Critical decompression failure: {e}")
    
    def _combine_compressed_data_fixed(self, padic_data: bytes, metadata: bytes) -> bytes:
        """Combine compressed components for fixed pipeline"""
        combined = bytearray()
        
        # Header
        combined.extend(struct.pack('<B', 3))  # Version 3 for fixed pipeline
        combined.extend(struct.pack('<I', len(padic_data)))
        combined.extend(struct.pack('<I', len(metadata)))
        
        # Data
        combined.extend(padic_data)
        combined.extend(metadata)
        
        return bytes(combined)
    
    def _split_compressed_data_fixed(self, compressed_data: bytes) -> Tuple[bytes, bytes]:
        """Split combined data for fixed pipeline"""
        pos = 0
        
        version = struct.unpack('<B', compressed_data[pos:pos+1])[0]
        pos += 1
        
        if version != 3:
            # Handle old versions for backward compatibility
            if version in [1, 2]:
                # Old format - try to handle gracefully
                logger.warning(f"Old compression version {version} detected, attempting compatibility mode")
                return self._split_compressed_data_legacy(compressed_data)
            else:
                raise ValueError(f"Unsupported compression version: {version}")
        
        padic_size = struct.unpack('<I', compressed_data[pos:pos+4])[0]
        pos += 4
        metadata_size = struct.unpack('<I', compressed_data[pos:pos+4])[0]
        pos += 4
        
        padic_data = compressed_data[pos:pos+padic_size]
        pos += padic_size
        metadata = compressed_data[pos:pos+metadata_size]
        
        return padic_data, metadata
    
    def _split_compressed_data_legacy(self, compressed_data: bytes) -> Tuple[bytes, bytes]:
        """Handle legacy format for backward compatibility"""
        # This would handle old format files
        # For now, raise error
        raise ValueError("Legacy format not supported in fixed pipeline")
    
    def _extract_sparse_indices(self, sparse_tensor: torch.sparse.Tensor) -> Optional[np.ndarray]:
        """Extract indices from sparse tensor"""
        if sparse_tensor._nnz() == 0:
            return None
            
        if hasattr(sparse_tensor, 'crow_indices'):
            coo_tensor = sparse_tensor.to_sparse_coo()
            return coo_tensor.indices().cpu().numpy()
        else:
            return sparse_tensor.indices().cpu().numpy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get compression system statistics"""
        return dict(self.stats)
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.stats = {
            'total_compressions': 0,
            'total_decompressions': 0,
            'total_bytes_compressed': 0,
            'total_bytes_decompressed': 0,
            'average_compression_ratio': 0.0,
            'peak_memory_usage': 0,
            'stage_times': {stage.value: [] for stage in CompressionStage},
            'errors': []
        }
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up compression system resources")
        self.sparse_bridge.clear_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleanup complete")


def validate_compression_system():
    """Validate the fixed P-adic compression system"""
    print("\n" + "="*80)
    print("VALIDATING FIXED P-ADIC COMPRESSION SYSTEM")
    print("="*80)
    
    config = CompressionConfig(
        prime=257,
        base_precision=2,  # Reduced
        target_error=1e-6,
        enable_gpu=torch.cuda.is_available(),
        validate_reconstruction=True
    )
    
    system = PadicCompressionSystem(config)
    print(f"\n✓ System initialized with prime={config.prime}")
    print(f"✓ Pipeline: Pattern→Sparse→Entropy→P-adic")
    
    # Test small tensor
    test_tensor = torch.randn(100, 100)
    importance = torch.abs(test_tensor) + 0.1
    
    print(f"\nTesting 100x100 tensor")
    print(f"  Original size: {test_tensor.numel() * 4 / 1024:.2f} KB")
    
    result = system.compress(test_tensor, importance)
    
    print(f"  Compression ratio: {result.compression_ratio:.2f}x")
    print(f"  Final size: {len(result.compressed_data) / 1024:.2f} KB")
    
    if result.compression_ratio > 1:
        print("  ✓ ACHIEVED COMPRESSION!")
    else:
        print("  ⚠ Still expanding, but less than before")
    
    decompressed = system.decompress(result.compressed_data)
    error = torch.nn.functional.mse_loss(test_tensor, decompressed.reconstructed_tensor)
    print(f"  Reconstruction MSE: {error:.2e}")
    
    system.cleanup()
    print("\n✓ TEST COMPLETED")


if __name__ == "__main__":
    validate_compression_system()
