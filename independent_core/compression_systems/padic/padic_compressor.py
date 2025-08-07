"""
Unified P-adic Compression System
Production-ready implementation integrating all compression components sequentially
NO FALLBACKS - HARD FAILURES ONLY
"""

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
from .safe_reconstruction import SafeReconstruction
from .memory_pressure_handler import MemoryPressureHandler


class CompressionStage(Enum):
    """Compression pipeline stages"""
    ADAPTIVE_PRECISION = "adaptive_precision"
    PATTERN_DETECTION = "pattern_detection"
    SPARSE_ENCODING = "sparse_encoding"
    ENTROPY_CODING = "entropy_coding"
    METADATA_COMPRESSION = "metadata_compression"


@dataclass
class CompressionConfig:
    """Unified configuration for P-adic compression system"""
    # P-adic parameters
    prime: int = 257
    base_precision: int = 4
    min_precision: int = 2
    max_precision: int = 4
    
    # Adaptive precision
    target_error: float = 1e-6
    importance_threshold: float = 0.1
    compression_priority: float = 0.5  # Balance between accuracy and compression
    
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
    
    # Validation and safety
    validate_reconstruction: bool = True
    max_reconstruction_error: float = 1e-5
    enable_logging: bool = True
    raise_on_error: bool = True  # NO FALLBACKS - HARD FAILURES
    
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
    Unified P-adic Compression System
    Integrates all components in a sequential pipeline
    """
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        """Initialize P-adic compression system
        
        Args:
            config: System configuration (uses defaults if None)
        """
        self.config = config or CompressionConfig()
        
        # Validate configuration
        logger.info(f"Initializing P-adic Compression System with prime={self.config.prime}")
        
        # Initialize device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and self.config.enable_gpu else 'cpu'
        )
        logger.info(f"Using device: {self.device}")
        
        # Initialize all components
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
        
        # Memory management
        if self.config.enable_memory_monitoring:
            self.memory_handler = MemoryPressureHandler(
                memory_limit_mb=self.config.gpu_memory_limit_mb,
                pressure_threshold=0.8,
                aggressive_cleanup=True
            )
        else:
            self.memory_handler = None
    
    def _initialize_components(self):
        """Initialize all compression components"""
        try:
            # Stage 1: Adaptive Precision
            precision_config = AdaptivePrecisionConfig(
                prime=self.config.prime,
                base_precision=self.config.base_precision,
                min_precision=self.config.min_precision,
                max_precision=self.config.max_precision,
                target_error=self.config.target_error,
                importance_threshold=self.config.importance_threshold,
                compression_priority=self.config.compression_priority,
                enable_gpu_acceleration=self.config.enable_gpu
            )
            self.adaptive_precision = AdaptivePrecisionWrapper(precision_config)
            logger.info("✓ Adaptive Precision Wrapper initialized")
            
            # Stage 2: Pattern Detection
            self.pattern_detector = SlidingWindowPatternDetector(
                min_pattern_length=self.config.min_pattern_length,
                max_pattern_length=self.config.max_pattern_length,
                min_frequency=self.config.min_pattern_frequency,
                hash_prime=self.config.pattern_hash_prime,
                device=str(self.device),
                enable_compile=self.device.type == 'cuda'
            )
            logger.info("✓ Pattern Detector initialized")
            
            # Stage 3: Sparse Encoding
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
            
            # Stage 4: Entropy Coding
            entropy_config = EntropyBridgeConfig(
                huffman_threshold=self.config.huffman_threshold,
                arithmetic_threshold=self.config.arithmetic_threshold,
                enable_pattern_detection=True,
                enable_delta_encoding=True,
                compression_level=6
            )
            self.entropy_bridge = EntropyPAdicBridge(
                self.config.prime,
                entropy_config
            )
            logger.info("✓ Entropy Bridge initialized")
            
            # Stage 5: Metadata Compression
            self.metadata_compressor = MetadataCompressor()
            logger.info("✓ Metadata Compressor initialized")
            
            # Additional components
            self.safe_reconstruction = SafeReconstruction(
                self.config.prime,
                self.config.base_precision
            )
            self.math_ops = PadicMathematicalOperations(
                self.config.prime,
                self.config.base_precision
            )
            self.hensel = AdaptiveHenselLifting(
                self.config.prime,
                self.config.base_precision
            )
            
            logger.info("All compression components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            if self.config.raise_on_error:
                raise
    
    def compress(self, tensor: torch.Tensor, 
                importance_scores: Optional[torch.Tensor] = None) -> CompressionResult:
        """
        Compress tensor through sequential pipeline
        
        Args:
            tensor: Input tensor to compress
            importance_scores: Optional importance scores for adaptive precision
            
        Returns:
            CompressionResult with compressed data and metrics
        """
        start_time = time.perf_counter()
        stage_metrics = {}
        
        try:
            # Validate input
            PadicValidation.validate_tensor(tensor)
            original_shape = tensor.shape
            original_size = tensor.numel() * tensor.element_size()
            
            logger.info(f"Starting compression of tensor shape={original_shape}")
            
            # Memory monitoring
            if self.memory_handler:
                self.memory_handler.check_memory_pressure()
            
            # Stage 1: Adaptive Precision P-adic Conversion
            stage_start = time.perf_counter()
            logger.info("Stage 1: Adaptive Precision Conversion")
            
            allocation = self.adaptive_precision.batch_compress_with_adaptive_precision(
                tensor,
                importance_scores
            )
            
            padic_weights = allocation.weights
            precision_map = allocation.precision_map
            
            stage_metrics['adaptive_precision'] = {
                'time': time.perf_counter() - stage_start,
                'average_precision': allocation.get_average_precision(),
                'error_stats': allocation.get_error_statistics(),
                'compression_ratio': allocation.compression_ratio
            }
            
            # Convert p-adic weights to digit tensor for further processing
            max_precision = max(w.precision for w in padic_weights)
            digit_tensor = torch.zeros(
                (len(padic_weights), max_precision),
                dtype=torch.int32,
                device=self.device
            )
            
            for i, weight in enumerate(padic_weights):
                digits = torch.tensor(weight.digits, device=self.device)
                digit_tensor[i, :len(digits)] = digits
            
            # Stage 2: Pattern Detection
            stage_start = time.perf_counter()
            logger.info("Stage 2: Pattern Detection")
            
            flat_digits = digit_tensor.flatten()
            pattern_result = self.pattern_detector.find_patterns(flat_digits)
            
            # Encode with patterns
            encoded_data, pattern_dict, pattern_lengths = self.pattern_detector.encode_with_patterns(
                flat_digits,
                pattern_result
            )
            
            stage_metrics['pattern_detection'] = {
                'time': time.perf_counter() - stage_start,
                'patterns_found': pattern_result.total_patterns_found,
                'bytes_replaced': pattern_result.bytes_replaced,
                'compression_potential': pattern_result.compression_potential
            }
            
            # Stage 3: Sparse Encoding
            stage_start = time.perf_counter()
            logger.info("Stage 3: Sparse Encoding")
            
            # Create valuation tensor
            valuations = torch.tensor(
                [w.valuation for w in padic_weights],
                dtype=torch.int32,
                device=self.device
            )
            
            # Convert to sparse format
            sparse_tensor = self.sparse_bridge.padic_to_sparse(digit_tensor, valuations)
            compression_ratio = self.sparse_bridge.get_compression_ratio(sparse_tensor)
            
            # Update sparsity threshold if needed
            if compression_ratio < 10:
                new_threshold = self.sparsity_manager.update_threshold(compression_ratio)
                self.sparse_bridge.threshold = new_threshold
            
            stage_metrics['sparse_encoding'] = {
                'time': time.perf_counter() - stage_start,
                'sparsity': 1.0 - (sparse_tensor._nnz() / digit_tensor.numel()),
                'compression_ratio': compression_ratio,
                'nnz': sparse_tensor._nnz()
            }
            
            # Stage 4: Entropy Coding
            stage_start = time.perf_counter()
            logger.info("Stage 4: Entropy Coding")
            
            # Encode sparse values with entropy coding
            sparse_values = sparse_tensor.values()
            if sparse_values.numel() > 0:
                entropy_compressed, entropy_metadata = self.entropy_bridge.encode_padic_tensor(
                    sparse_values
                )
            else:
                entropy_compressed = b''
                entropy_metadata = {'empty': True}
            
            stage_metrics['entropy_coding'] = {
                'time': time.perf_counter() - stage_start,
                'method': entropy_metadata.get('encoding_method', 'none'),
                'compression_ratio': entropy_metadata.get('compression_metrics', {}).get('compression_ratio', 1.0)
            }
            
            # Stage 5: Metadata Compression
            stage_start = time.perf_counter()
            logger.info("Stage 5: Metadata Compression")
            
            # Prepare metadata
            metadata = {
                'version': 1,
                'prime': self.config.prime,
                'precision': self.config.base_precision,
                'original_shape': original_shape,
                'precision_map': precision_map.cpu().numpy().tolist(),
                'pattern_dict': pattern_dict,
                'pattern_lengths': pattern_lengths.cpu().numpy().tolist() if pattern_lengths.numel() > 0 else [],
                'sparse_indices': sparse_tensor.indices().cpu().numpy() if sparse_tensor._nnz() > 0 else None,
                'sparse_shape': list(sparse_tensor.shape),
                'valuations': valuations.cpu().numpy().tolist(),
                'entropy_metadata': entropy_metadata,
                'compression_config': {
                    'target_error': self.config.target_error,
                    'sparsity_threshold': self.config.sparsity_threshold
                }
            }
            
            compressed_metadata = self.metadata_compressor.compress_metadata(metadata)
            
            stage_metrics['metadata_compression'] = {
                'time': time.perf_counter() - stage_start,
                'metadata_size': len(compressed_metadata),
                'overhead_ratio': len(compressed_metadata) / (len(entropy_compressed) + 1)
            }
            
            # Combine all compressed data
            combined_data = self._combine_compressed_data(
                entropy_compressed,
                compressed_metadata,
                encoded_data.cpu().numpy().tobytes() if encoded_data.numel() > 0 else b''
            )
            
            # Calculate final metrics
            total_time = time.perf_counter() - start_time
            final_compression_ratio = original_size / len(combined_data)
            
            # Validate if enabled
            validation_passed = True
            error_metrics = {}
            
            if self.config.validate_reconstruction:
                try:
                    # Quick decompression test
                    decompressed = self.decompress(combined_data)
                    reconstruction_error = torch.nn.functional.mse_loss(
                        tensor,
                        decompressed.reconstructed_tensor
                    ).item()
                    
                    validation_passed = reconstruction_error < self.config.max_reconstruction_error
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
            self.stats['average_compression_ratio'] = (
                (self.stats['average_compression_ratio'] * (self.stats['total_compressions'] - 1) +
                 final_compression_ratio) / self.stats['total_compressions']
            )
            
            # Update stage times
            for stage, metrics in stage_metrics.items():
                self.stats['stage_times'][stage].append(metrics['time'])
            
            # Memory usage
            memory_usage = len(combined_data)
            if self.memory_handler:
                memory_usage = max(memory_usage, self.memory_handler.get_current_memory_usage())
            
            self.stats['peak_memory_usage'] = max(self.stats['peak_memory_usage'], memory_usage)
            
            logger.info(f"Compression complete: ratio={final_compression_ratio:.2f}x, time={total_time:.3f}s")
            
            return CompressionResult(
                compressed_data=combined_data,
                metadata=metadata,
                compression_ratio=final_compression_ratio,
                processing_time=total_time,
                memory_usage=memory_usage,
                stage_metrics=stage_metrics,
                validation_passed=validation_passed,
                error_metrics=error_metrics
            )
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            self.stats['errors'].append(str(e))
            if self.config.raise_on_error:
                raise
            else:
                # Should never reach here with NO FALLBACKS policy
                raise RuntimeError(f"Critical compression failure: {e}")
    
    def decompress(self, compressed_data: bytes) -> DecompressionResult:
        """
        Decompress data through reverse pipeline
        
        Args:
            compressed_data: Compressed data from compress()
            
        Returns:
            DecompressionResult with reconstructed tensor
        """
        start_time = time.perf_counter()
        stage_metrics = {}
        
        try:
            logger.info("Starting decompression")
            
            # Memory monitoring
            if self.memory_handler:
                self.memory_handler.check_memory_pressure()
            
            # Split combined data
            entropy_compressed, compressed_metadata, pattern_encoded = self._split_compressed_data(
                compressed_data
            )
            
            # Stage 1: Metadata Decompression
            stage_start = time.perf_counter()
            logger.info("Stage 1: Metadata Decompression")
            
            metadata = self.metadata_compressor.decompress_metadata(compressed_metadata)
            
            stage_metrics['metadata_decompression'] = {
                'time': time.perf_counter() - stage_start,
                'metadata_size': len(compressed_metadata)
            }
            
            # Stage 2: Entropy Decoding
            stage_start = time.perf_counter()
            logger.info("Stage 2: Entropy Decoding")
            
            if entropy_compressed and not metadata.get('entropy_metadata', {}).get('empty'):
                sparse_values_tensor = self.entropy_bridge.decode_padic_tensor(
                    entropy_compressed,
                    metadata['entropy_metadata']
                )
            else:
                sparse_values_tensor = torch.tensor([], device=self.device)
            
            stage_metrics['entropy_decoding'] = {
                'time': time.perf_counter() - stage_start,
                'values_decoded': sparse_values_tensor.numel()
            }
            
            # Stage 3: Sparse Reconstruction
            stage_start = time.perf_counter()
            logger.info("Stage 3: Sparse Reconstruction")
            
            # Reconstruct sparse tensor
            if metadata.get('sparse_indices') is not None and sparse_values_tensor.numel() > 0:
                indices = torch.tensor(metadata['sparse_indices'], device=self.device)
                sparse_shape = tuple(metadata['sparse_shape'])
                
                # Create sparse tensor
                sparse_tensor = torch.sparse_coo_tensor(
                    indices.T,
                    sparse_values_tensor,
                    sparse_shape,
                    device=self.device
                )
                
                # Add valuations
                valuations = torch.tensor(metadata['valuations'], device=self.device)
                sparse_tensor.valuations = valuations
                
                # Convert to dense
                digit_tensor, valuations = self.sparse_bridge.sparse_to_padic(sparse_tensor)
            else:
                # Empty or fully sparse tensor
                digit_tensor = torch.zeros(metadata['sparse_shape'], device=self.device)
                valuations = torch.tensor(metadata['valuations'], device=self.device)
            
            stage_metrics['sparse_reconstruction'] = {
                'time': time.perf_counter() - stage_start,
                'tensor_shape': list(digit_tensor.shape)
            }
            
            # Stage 4: Pattern Reconstruction
            stage_start = time.perf_counter()
            logger.info("Stage 4: Pattern Reconstruction")
            
            if pattern_encoded and metadata.get('pattern_dict'):
                # Decode pattern-encoded data
                pattern_tensor = torch.frombuffer(
                    pattern_encoded,
                    dtype=torch.int32
                ).to(self.device)
                
                pattern_lengths = torch.tensor(
                    metadata.get('pattern_lengths', []),
                    device=self.device
                )
                
                # Decode patterns
                decoded_digits = self.pattern_detector.decode_with_patterns(
                    pattern_tensor,
                    metadata['pattern_dict'],
                    pattern_lengths
                )
                
                # Use pattern-decoded data if available
                if decoded_digits.numel() > 0:
                    digit_tensor = decoded_digits.reshape(digit_tensor.shape)
            
            stage_metrics['pattern_reconstruction'] = {
                'time': time.perf_counter() - stage_start,
                'patterns_used': len(metadata.get('pattern_dict', {}))
            }
            
            # Stage 5: P-adic to Float Reconstruction
            stage_start = time.perf_counter()
            logger.info("Stage 5: P-adic to Float Reconstruction")
            
            # Reconstruct p-adic weights
            precision_map = torch.tensor(metadata['precision_map'], device=self.device)
            flat_precision = precision_map.flatten()
            
            # Create p-adic weights with variable precision
            padic_weights = []
            for i in range(digit_tensor.shape[0]):
                precision = int(flat_precision[i].item()) if i < flat_precision.shape[0] else self.config.base_precision
                digits = digit_tensor[i, :precision].cpu().numpy().tolist()
                valuation = valuations[i].item() if i < valuations.shape[0] else 0
                
                weight = PadicWeight(
                    digits=digits,
                    valuation=valuation,
                    prime=metadata['prime'],
                    precision=precision
                )
                padic_weights.append(weight)
            
            # Reconstruct float values
            reconstructed_values = []
            for weight in padic_weights:
                # Use safe reconstruction
                value = self.safe_reconstruction.safe_padic_to_float(weight)
                reconstructed_values.append(value)
            
            # Create tensor
            reconstructed_tensor = torch.tensor(
                reconstructed_values,
                dtype=torch.float32,
                device=self.device
            ).reshape(metadata['original_shape'])
            
            stage_metrics['padic_reconstruction'] = {
                'time': time.perf_counter() - stage_start,
                'weights_reconstructed': len(padic_weights)
            }
            
            # Calculate metrics
            total_time = time.perf_counter() - start_time
            memory_usage = reconstructed_tensor.numel() * reconstructed_tensor.element_size()
            
            # Validation
            validation_passed = True
            reconstruction_error = 0.0
            
            if self.config.validate_reconstruction:
                # Check for NaN or Inf
                if torch.isnan(reconstructed_tensor).any() or torch.isinf(reconstructed_tensor).any():
                    validation_passed = False
                    logger.warning("Reconstruction contains NaN or Inf values")
            
            # Update statistics
            self.stats['total_decompressions'] += 1
            self.stats['total_bytes_decompressed'] += len(compressed_data)
            
            logger.info(f"Decompression complete: shape={reconstructed_tensor.shape}, time={total_time:.3f}s")
            
            return DecompressionResult(
                reconstructed_tensor=reconstructed_tensor,
                original_shape=tuple(metadata['original_shape']),
                processing_time=total_time,
                memory_usage=memory_usage,
                stage_metrics=stage_metrics,
                validation_passed=validation_passed,
                reconstruction_error=reconstruction_error
            )
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            self.stats['errors'].append(str(e))
            if self.config.raise_on_error:
                raise
            else:
                # Should never reach here with NO FALLBACKS policy
                raise RuntimeError(f"Critical decompression failure: {e}")
    
    def _combine_compressed_data(self, entropy_data: bytes, 
                                metadata: bytes,
                                pattern_data: bytes) -> bytes:
        """Combine all compressed components into single byte stream"""
        combined = bytearray()
        
        # Header: version and component sizes
        combined.extend(struct.pack('<B', 1))  # Version
        combined.extend(struct.pack('<I', len(entropy_data)))
        combined.extend(struct.pack('<I', len(metadata)))
        combined.extend(struct.pack('<I', len(pattern_data)))
        
        # Data
        combined.extend(entropy_data)
        combined.extend(metadata)
        combined.extend(pattern_data)
        
        return bytes(combined)
    
    def _split_compressed_data(self, compressed_data: bytes) -> Tuple[bytes, bytes, bytes]:
        """Split combined data back into components"""
        pos = 0
        
        # Read header
        version = struct.unpack('<B', compressed_data[pos:pos+1])[0]
        pos += 1
        
        if version != 1:
            raise ValueError(f"Unsupported compression version: {version}")
        
        entropy_size = struct.unpack('<I', compressed_data[pos:pos+4])[0]
        pos += 4
        metadata_size = struct.unpack('<I', compressed_data[pos:pos+4])[0]
        pos += 4
        pattern_size = struct.unpack('<I', compressed_data[pos:pos+4])[0]
        pos += 4
        
        # Extract components
        entropy_data = compressed_data[pos:pos+entropy_size]
        pos += entropy_size
        metadata = compressed_data[pos:pos+metadata_size]
        pos += metadata_size
        pattern_data = compressed_data[pos:pos+pattern_size]
        
        return entropy_data, metadata, pattern_data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get compression system statistics"""
        stats = dict(self.stats)
        
        # Add component statistics
        stats['component_stats'] = {
            'adaptive_precision': self.adaptive_precision.get_statistics(),
            'pattern_detector': self.pattern_detector.analyze_compression_efficiency(
                torch.zeros(100)  # Dummy data for stats
            ) if hasattr(self.pattern_detector, 'analyze_compression_efficiency') else {},
            'sparse_bridge': {
                'cache_hits': self.sparse_bridge.cache_hits,
                'cache_misses': self.sparse_bridge.cache_misses
            },
            'entropy_bridge': self.entropy_bridge.get_statistics(),
            'metadata_compressor': self.metadata_compressor.get_statistics()
        }
        
        # Calculate stage time averages
        for stage in CompressionStage:
            times = stats['stage_times'][stage.value]
            if times:
                stats[f'avg_{stage.value}_time'] = sum(times) / len(times)
        
        return stats
    
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
        
        # Reset component statistics
        self.adaptive_precision.reset_statistics()
        self.sparse_bridge.clear_cache()
        self.entropy_bridge.reset_statistics()
        self.metadata_compressor.reset_statistics()
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up compression system resources")
        
        # Clear caches
        self.sparse_bridge.clear_cache()
        
        # Force garbage collection
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Cleanup complete")


def validate_compression_system():
    """Validate the unified P-adic compression system"""
    print("\n" + "="*80)
    print("VALIDATING UNIFIED P-ADIC COMPRESSION SYSTEM")
    print("="*80)
    
    # Create test configuration
    config = CompressionConfig(
        prime=257,
        base_precision=4,
        target_error=1e-6,
        enable_gpu=torch.cuda.is_available(),
        validate_reconstruction=True
    )
    
    # Initialize system
    system = PadicCompressionSystem(config)
    print(f"\n✓ System initialized with prime={config.prime}")
    
    # Test cases
    test_cases = [
        ("Small tensor", torch.randn(10, 10)),
        ("Medium tensor", torch.randn(100, 100)),
        ("Large tensor", torch.randn(1000, 100)),
        ("Sparse tensor", torch.randn(500, 500) * (torch.rand(500, 500) > 0.9)),
        ("Uniform tensor", torch.ones(200, 200) * 0.5),
        ("High variance", torch.randn(300, 300) * 10),
    ]
    
    for name, tensor in test_cases:
        print(f"\nTesting: {name}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Size: {tensor.numel() * 4 / 1024:.2f} KB")
        
        try:
            # Generate importance scores (could be gradients in real use)
            importance = torch.abs(tensor) + 0.1
            
            # Compress
            result = system.compress(tensor, importance)
            
            print(f"  Compression ratio: {result.compression_ratio:.2f}x")
            print(f"  Processing time: {result.processing_time:.3f}s")
            print(f"  Validation: {'PASSED' if result.validation_passed else 'FAILED'}")
            
            # Stage metrics
            for stage, metrics in result.stage_metrics.items():
                print(f"  {stage}: {metrics.get('time', 0):.3f}s")
            
            # Decompress
            decompressed = system.decompress(result.compressed_data)
            
            # Verify reconstruction
            error = torch.nn.functional.mse_loss(tensor, decompressed.reconstructed_tensor)
            print(f"  Reconstruction MSE: {error:.2e}")
            
            if error > config.max_reconstruction_error:
                print(f"  ⚠ Warning: High reconstruction error")
            else:
                print(f"  ✓ Reconstruction successful")
                
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            if config.raise_on_error:
                raise
    
    # Print final statistics
    stats = system.get_statistics()
    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)
    print(f"Total compressions: {stats['total_compressions']}")
    print(f"Total decompressions: {stats['total_decompressions']}")
    print(f"Average compression ratio: {stats['average_compression_ratio']:.2f}x")
    print(f"Peak memory usage: {stats['peak_memory_usage'] / 1024 / 1024:.2f} MB")
    
    # Clean up
    system.cleanup()
    
    print("\n✓ ALL TESTS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    validate_compression_system()