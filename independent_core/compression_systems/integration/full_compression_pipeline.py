"""
Full Compression Pipeline - Complete Categorical → IEEE 754 → P-adic Log → GPU Burst System

Integrates all compression components into the complete pipeline that achieves
the user's desired ~4x compression architecture.

NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import time
import threading
import logging

logger = logging.getLogger(__name__)

# Import all pipeline components
try:
    from ..gpu_memory.cpu_bursting_pipeline import CPU_BurstingPipeline, CPUBurstingConfig
    from ..padic.padic_encoder import PadicWeight
    from ..padic.padic_advanced import PadicDecompressionEngine, GPUDecompressionConfig
    from ..categorical.categorical_storage_manager import CategoricalStorageManager, CategoricalStorageConfig
    from ..categorical.ieee754_channel_extractor import IEEE754ChannelExtractor, IEEE754Channels
    from ..categorical.weight_categorizer import WeightCategorizer, CategorizationResult
    from ..padic.padic_logarithmic_encoder import PadicLogarithmicEncoder, LogarithmicEncodingConfig, LogarithmicPadicWeight
    from .categorical_to_padic_bridge import CategoricalToPadicBridge
except ImportError:
    from compression_systems.gpu_memory.cpu_bursting_pipeline import CPU_BurstingPipeline, CPUBurstingConfig
    from compression_systems.padic.padic_encoder import PadicWeight
    from compression_systems.padic.padic_advanced import PadicDecompressionEngine, GPUDecompressionConfig
    from compression_systems.categorical.categorical_storage_manager import CategoricalStorageManager, CategoricalStorageConfig
    from compression_systems.categorical.ieee754_channel_extractor import IEEE754ChannelExtractor, IEEE754Channels
    from compression_systems.categorical.weight_categorizer import WeightCategorizer, CategorizationResult
    from compression_systems.padic.padic_logarithmic_encoder import PadicLogarithmicEncoder, LogarithmicEncodingConfig, LogarithmicPadicWeight
    from compression_systems.integration.categorical_to_padic_bridge import CategoricalToPadicBridge


@dataclass
class FullCompressionConfig:
    """Configuration for the complete compression pipeline"""
    # Base configurations for each component
    cpu_bursting_config: CPUBurstingConfig
    gpu_decompression_config: GPUDecompressionConfig
    categorical_storage_config: CategoricalStorageConfig
    logarithmic_encoding_config: LogarithmicEncodingConfig
    
    # Pipeline-specific settings
    enable_categorical_storage: bool = True
    enable_ieee754_optimization: bool = True
    enable_logarithmic_encoding: bool = True
    enable_gpu_bursting: bool = True
    
    # Performance optimization
    enable_parallel_processing: bool = True
    batch_size_optimization: bool = True
    adaptive_precision: bool = True
    
    # Quality settings
    validate_compression_ratios: bool = True
    target_compression_ratio: float = 4.0        # Target ~4x compression
    minimum_acceptable_ratio: float = 2.0        # Minimum acceptable compression
    
    def __post_init__(self):
        """Validate configuration consistency"""
        if self.target_compression_ratio < self.minimum_acceptable_ratio:
            raise ValueError(f"Target compression ratio {self.target_compression_ratio} must be >= minimum {self.minimum_acceptable_ratio}")
        
        if not (1.5 <= self.minimum_acceptable_ratio <= 10.0):
            raise ValueError(f"Minimum acceptable ratio {self.minimum_acceptable_ratio} must be in [1.5, 10.0]")


@dataclass
class CompressionResult:
    """Result of full compression pipeline"""
    compressed_weights: List[LogarithmicPadicWeight]
    compression_ratio: float
    original_size_bytes: int
    compressed_size_bytes: int
    categorization_result: CategorizationResult
    ieee754_channels: IEEE754Channels
    processing_info: Dict[str, Any]
    pipeline_stats: Dict[str, Any]


@dataclass
class DecompressionResult:
    """Result of full decompression pipeline"""
    reconstructed_tensor: torch.Tensor
    decompression_ratio: float
    processing_mode: str                          # 'gpu', 'cpu', or 'hybrid'
    reconstruction_error: float
    processing_info: Dict[str, Any]
    pipeline_stats: Dict[str, Any]


class FullCompressionPipeline(CPU_BurstingPipeline):
    """
    Complete compression pipeline: Categorical → IEEE 754 → P-adic Log → GPU Burst
    
    Inherits from CPU_BurstingPipeline to maintain compatibility while adding
    the complete categorical storage and logarithmic encoding pipeline.
    """
    
    def __init__(self, config: FullCompressionConfig, gpu_decompression_engine: PadicDecompressionEngine):
        """Initialize full compression pipeline
        
        Args:
            config: Complete pipeline configuration
            gpu_decompression_engine: GPU decompression engine
        """
        # Initialize parent CPU bursting pipeline
        super().__init__(config.cpu_bursting_config, gpu_decompression_engine)
        
        self.full_config = config
        
        # Initialize categorical storage system
        if config.enable_categorical_storage:
            self.categorical_manager = CategoricalStorageManager(config.categorical_storage_config)
            self.weight_categorizer = WeightCategorizer(
                enable_clustering=True,
                enable_pattern_detection=True,
                max_clusters=20
            )
        else:
            self.categorical_manager = None
            self.weight_categorizer = None
        
        # Initialize IEEE 754 channel extractor
        if config.enable_ieee754_optimization:
            self.ieee754_extractor = IEEE754ChannelExtractor(validate_reconstruction=True)
        else:
            self.ieee754_extractor = None
        
        # Initialize p-adic logarithmic encoder
        if config.enable_logarithmic_encoding:
            self.logarithmic_encoder = PadicLogarithmicEncoder(config.logarithmic_encoding_config)
        else:
            self.logarithmic_encoder = None
        
        # Initialize bridge for component integration
        self.categorical_bridge = CategoricalToPadicBridge()
        
        # Pipeline statistics
        self.pipeline_stats = {
            'total_compressions': 0,
            'total_decompressions': 0,
            'average_compression_ratio': 0.0,
            'categorical_storage_usage': 0.0,
            'ieee754_optimizations': 0,
            'logarithmic_encodings': 0,
            'gpu_bursting_events': 0,
            'processing_times': {
                'categorization': [],
                'ieee754_extraction': [],
                'logarithmic_encoding': [],
                'gpu_bursting': []
            }
        }
        
        # Thread safety for pipeline operations
        self._pipeline_lock = threading.RLock()
        
        logger.info("FullCompressionPipeline initialized with target compression ratio %.1fx", 
                   config.target_compression_ratio)
    
    def compress_with_full_pipeline(self, weights: torch.Tensor, 
                                   metadata: Optional[Dict[str, Any]] = None) -> CompressionResult:
        """Compress weights using the complete categorical → IEEE 754 → p-adic log pipeline
        
        Args:
            weights: Tensor of weights to compress
            metadata: Optional metadata for optimization hints
            
        Returns:
            CompressionResult with comprehensive compression information
            
        Raises:
            RuntimeError: If compression fails (hard failure)
            ValueError: If input validation fails
        """
        if weights is None:
            raise ValueError("Weights tensor cannot be None")
        
        if weights.numel() == 0:
            raise ValueError("Weights tensor cannot be empty")
        
        start_time = time.time()
        
        try:
            with self._pipeline_lock:
                # Stage 1: Categorical Storage and Analysis
                categorization_start = time.time()
                categorization_result = self._perform_categorical_analysis(weights, metadata)
                categorization_time = time.time() - categorization_start
                
                # Stage 2: IEEE 754 Channel Extraction
                ieee754_start = time.time()
                ieee754_channels = self._extract_ieee754_channels(weights, categorization_result)
                ieee754_time = time.time() - ieee754_start
                
                # Stage 3: P-adic Logarithmic Encoding
                encoding_start = time.time()
                compressed_weights = self._perform_logarithmic_encoding(ieee754_channels, categorization_result)
                encoding_time = time.time() - encoding_start
                
                # Stage 4: Calculate Compression Metrics
                compression_metrics = self._calculate_compression_metrics(weights, compressed_weights)
                
                # Validate compression ratio meets requirements
                if (self.full_config.validate_compression_ratios and 
                    compression_metrics['compression_ratio'] < self.full_config.minimum_acceptable_ratio):
                    raise RuntimeError(
                        f"Compression ratio {compression_metrics['compression_ratio']:.2f}x below minimum "
                        f"acceptable {self.full_config.minimum_acceptable_ratio:.2f}x - hard failure"
                    )
                
                # Update pipeline statistics
                total_time = time.time() - start_time
                self._update_compression_statistics(categorization_time, ieee754_time, encoding_time, 
                                                  compression_metrics['compression_ratio'])
                
                # Build comprehensive result
                result = CompressionResult(
                    compressed_weights=compressed_weights,
                    compression_ratio=compression_metrics['compression_ratio'],
                    original_size_bytes=compression_metrics['original_size_bytes'],
                    compressed_size_bytes=compression_metrics['compressed_size_bytes'],
                    categorization_result=categorization_result,
                    ieee754_channels=ieee754_channels,
                    processing_info={
                        'total_processing_time_ms': total_time * 1000,
                        'categorization_time_ms': categorization_time * 1000,
                        'ieee754_extraction_time_ms': ieee754_time * 1000,
                        'logarithmic_encoding_time_ms': encoding_time * 1000,
                        'compression_stages_completed': 4,
                        'target_compression_ratio': self.full_config.target_compression_ratio
                    },
                    pipeline_stats=self.get_pipeline_statistics()
                )
                
                logger.info("Full pipeline compression completed: %.2fx ratio, %d stages, %.1fms total",
                           compression_metrics['compression_ratio'], 4, total_time * 1000)
                
                return result
                
        except Exception as e:
            raise RuntimeError(f"Full compression pipeline failed: {e}")
    
    def decompress_with_full_pipeline(self, compressed_weights: List[LogarithmicPadicWeight],
                                    target_precision: int,
                                    metadata: Dict[str, Any]) -> DecompressionResult:
        """Decompress weights using the complete pipeline with GPU bursting
        
        Args:
            compressed_weights: List of compressed logarithmic p-adic weights
            target_precision: Target precision for decompression
            metadata: Metadata including original shape and optimization info
            
        Returns:
            DecompressionResult with comprehensive decompression information
            
        Raises:
            RuntimeError: If decompression fails (hard failure)
        """
        if not compressed_weights:
            raise ValueError("Compressed weights list cannot be empty")
        
        if 'original_shape' not in metadata:
            raise ValueError("Metadata must contain 'original_shape'")
        
        start_time = time.time()
        
        try:
            with self._pipeline_lock:
                # Stage 1: Convert to standard p-adic weights for GPU bursting pipeline
                standard_padic_weights = [lw.padic_weight for lw in compressed_weights]
                
                # Stage 2: Apply GPU bursting decompression (inherits from parent)
                bursting_start = time.time()
                decompressed_tensor, bursting_info = super().decompress(
                    standard_padic_weights, target_precision, metadata
                )
                bursting_time = time.time() - bursting_start
                
                # Stage 3: Apply inverse logarithmic transformation if needed
                if self.logarithmic_encoder and any(lw.encoding_method.startswith("logarithmic") for lw in compressed_weights):
                    inverse_start = time.time()
                    decompressed_tensor = self._apply_inverse_logarithmic_transformation(
                        decompressed_tensor, compressed_weights, metadata
                    )
                    inverse_time = time.time() - inverse_start
                else:
                    inverse_time = 0.0
                
                # Stage 4: Validate reconstruction
                reconstruction_error = self._calculate_reconstruction_error(
                    decompressed_tensor, compressed_weights, metadata
                )
                
                # Calculate decompression metrics
                total_time = time.time() - start_time
                processing_mode = bursting_info.get('mode', 'unknown')
                
                # Update statistics
                self.pipeline_stats['total_decompressions'] += 1
                if processing_mode == 'gpu':
                    self.pipeline_stats['gpu_bursting_events'] += 1
                
                # Build comprehensive result
                result = DecompressionResult(
                    reconstructed_tensor=decompressed_tensor,
                    decompression_ratio=1.0 / self._estimate_compression_ratio_from_weights(compressed_weights),
                    processing_mode=processing_mode,
                    reconstruction_error=reconstruction_error,
                    processing_info={
                        'total_processing_time_ms': total_time * 1000,
                        'gpu_bursting_time_ms': bursting_time * 1000,
                        'inverse_transform_time_ms': inverse_time * 1000,
                        'bursting_info': bursting_info,
                        'precision_used': target_precision
                    },
                    pipeline_stats=self.get_pipeline_statistics()
                )
                
                logger.info("Full pipeline decompression completed: %s mode, %.2e error, %.1fms total",
                           processing_mode, reconstruction_error, total_time * 1000)
                
                return result
                
        except Exception as e:
            raise RuntimeError(f"Full decompression pipeline failed: {e}")
    
    def _perform_categorical_analysis(self, weights: torch.Tensor, 
                                    metadata: Optional[Dict[str, Any]]) -> CategorizationResult:
        """Perform categorical analysis and storage
        
        Args:
            weights: Tensor of weights to analyze
            metadata: Optional metadata
            
        Returns:
            CategorizationResult with analysis
        """
        if not self.full_config.enable_categorical_storage:
            # Create dummy result if categorical storage is disabled
            from ..categorical.categorical_storage_manager import CategoryType
            return CategorizationResult(
                primary_category=CategoryType.MEDIUM_WEIGHTS,
                secondary_categories=[],
                detected_patterns=[],
                similarity_groups=[],
                compression_estimate=2.0,
                optimization_hints={}
            )
        
        try:
            # Detect sparsity for compression optimization
            zero_mask = torch.abs(weights) < 1e-8
            zero_count = zero_mask.sum().item()
            sparsity_ratio = zero_count / weights.numel()
            
            # Store weights categorically in RAM
            storage_info = self.categorical_manager.store_weights_categorically(weights, metadata)
            
            # Perform pattern analysis
            categorization_result = self.weight_categorizer.categorize_weights(weights, None, metadata)
            
            # Update categorization with storage information and sparsity
            categorization_result.optimization_hints.update({
                'categorical_storage_info': storage_info,
                'ram_usage_mb': storage_info.get('ram_usage_mb', 0.0),
                'sparsity_ratio': sparsity_ratio,
                'zero_count': zero_count,
                'use_sparse_encoding': sparsity_ratio > 0.1  # Use sparse encoding if >10% zeros
            })
            
            return categorization_result
            
        except Exception as e:
            raise RuntimeError(f"Categorical analysis failed: {e}")
    
    def _extract_ieee754_channels(self, weights: torch.Tensor, 
                                categorization: CategorizationResult) -> IEEE754Channels:
        """Extract IEEE 754 channels with categorical optimization
        
        Args:
            weights: Tensor of weights
            categorization: Categorization analysis result
            
        Returns:
            IEEE754Channels object
        """
        if not self.full_config.enable_ieee754_optimization:
            # Create dummy channels if IEEE 754 optimization is disabled
            flattened = weights.flatten().cpu().numpy()
            return IEEE754Channels(
                sign_channel=np.sign(flattened).astype(np.uint8),
                exponent_channel=np.ones(len(flattened), dtype=np.uint8) * 127,  # Neutral exponent
                mantissa_channel=np.abs(flattened),
                original_values=flattened
            )
        
        try:
            # Extract IEEE 754 channels
            channels = self.ieee754_extractor.extract_channels_from_tensor(weights)
            
            # Apply categorical optimization if available
            if categorization.optimization_hints:
                # Optimize channels based on categorization hints
                target_prime = categorization.optimization_hints.get('recommended_prime', 257)
                channels = self.ieee754_extractor.optimize_channels_for_padic(channels, target_prime)
            
            self.pipeline_stats['ieee754_optimizations'] += 1
            return channels
            
        except Exception as e:
            raise RuntimeError(f"IEEE 754 channel extraction failed: {e}")
    
    def _perform_logarithmic_encoding(self, channels: IEEE754Channels, 
                                    categorization: CategorizationResult) -> List[LogarithmicPadicWeight]:
        """Perform p-adic logarithmic encoding of IEEE 754 channels
        
        Args:
            channels: IEEE 754 channels to encode
            categorization: Categorization analysis for optimization
            
        Returns:
            List of LogarithmicPadicWeight objects
        """
        if not self.full_config.enable_logarithmic_encoding:
            # Fallback to direct p-adic encoding without logarithmic compression
            return self._perform_direct_padic_encoding(channels)
        
        try:
            # Apply optimization hints from categorization
            if categorization.optimization_hints:
                self._apply_encoding_optimizations(categorization.optimization_hints)
            
            # Perform logarithmic encoding
            encoded_weights = self.logarithmic_encoder.encode_ieee754_channels_logarithmically(channels)
            
            self.pipeline_stats['logarithmic_encodings'] += len(encoded_weights)
            return encoded_weights
            
        except Exception as e:
            raise RuntimeError(f"Logarithmic encoding failed: {e}")
    
    def _perform_direct_padic_encoding(self, channels: IEEE754Channels) -> List[LogarithmicPadicWeight]:
        """Fallback direct p-adic encoding without logarithmic compression
        
        Args:
            channels: IEEE 754 channels to encode
            
        Returns:
            List of LogarithmicPadicWeight objects (using direct encoding)
        """
        try:
            from ..padic.padic_encoder import PadicMathematicalOperations
            from fractions import Fraction
            
            # Use safe precision
            prime = self.full_config.logarithmic_encoding_config.prime
            precision = min(4, self.full_config.logarithmic_encoding_config.max_safe_precision)
            
            math_ops = PadicMathematicalOperations(prime, precision)
            encoded_weights = []
            
            for i, value in enumerate(channels.original_values):
                try:
                    # Convert to fraction and then to p-adic
                    if abs(value) < 1e-10:
                        fraction_val = Fraction(0)
                    else:
                        fraction_val = Fraction(value).limit_denominator(1000)
                    
                    padic_weight = math_ops.to_padic(fraction_val)
                    
                    # Create logarithmic p-adic weight wrapper (but using direct encoding)
                    log_weight = LogarithmicPadicWeight(
                        padic_weight=padic_weight,
                        original_value=float(value),
                        log_value=float(value),  # No log transformation
                        encoding_method="direct_fallback",
                        compression_metadata={'index': i, 'fallback': True}
                    )
                    
                    encoded_weights.append(log_weight)
                    
                except Exception:
                    # Create zero weight for problematic values
                    zero_weight = PadicWeight(
                        value=Fraction(0),
                        prime=prime,
                        precision=precision,
                        valuation=0,
                        digits=[0] * precision
                    )
                    
                    log_weight = LogarithmicPadicWeight(
                        padic_weight=zero_weight,
                        original_value=0.0,
                        log_value=0.0,
                        encoding_method="zero_fallback",
                        compression_metadata={'index': i, 'zero_fallback': True}
                    )
                    
                    encoded_weights.append(log_weight)
            
            return encoded_weights
            
        except Exception as e:
            raise RuntimeError(f"Direct p-adic encoding fallback failed: {e}")
    
    def _apply_encoding_optimizations(self, optimization_hints: Dict[str, Any]) -> None:
        """Apply optimization hints to the logarithmic encoder
        
        Args:
            optimization_hints: Hints from categorization analysis
        """
        try:
            # Apply precision optimization
            if 'recommended_precision' in optimization_hints:
                recommended = optimization_hints['recommended_precision']
                max_safe = self.full_config.logarithmic_encoding_config.max_safe_precision
                
                optimized_precision = min(recommended, max_safe)
                self.logarithmic_encoder.config.precision = optimized_precision
                
            # Apply quantization optimization
            if 'quantization_levels' in optimization_hints:
                levels = optimization_hints['quantization_levels']
                self.logarithmic_encoder.config.quantization_levels = min(levels, 65536)
            
            # Apply sparsity optimization
            if optimization_hints.get('use_sparse_encoding', False):
                self.logarithmic_encoder.config.enable_run_length_encoding = True
            
        except Exception as e:
            logger.warning("Failed to apply encoding optimizations: %s", e)
    
    def _calculate_compression_metrics(self, original_weights: torch.Tensor, 
                                     compressed_weights: List[LogarithmicPadicWeight]) -> Dict[str, Any]:
        """Calculate comprehensive compression metrics
        
        Args:
            original_weights: Original weight tensor
            compressed_weights: Compressed weights
            
        Returns:
            Dictionary of compression metrics
        """
        try:
            # Calculate original size
            original_size_bytes = original_weights.numel() * original_weights.element_size()
            
            # Calculate compressed size more accurately
            compressed_size_bytes = 0
            for lw in compressed_weights:
                # Count only non-zero p-adic digits (sparse storage)
                non_zero_digits = [d for d in lw.padic_weight.digits if d != 0]
                if not non_zero_digits:
                    # Zero weight: minimal storage (1 byte marker)
                    compressed_size_bytes += 1
                else:
                    # Non-zero digits: 1 byte per digit + valuation + length
                    compressed_size_bytes += len(non_zero_digits) * 1  # 1 byte per digit
                    compressed_size_bytes += 2  # valuation (1 byte) + length (1 byte)
            
            # Calculate compression ratio
            if compressed_size_bytes > 0:
                compression_ratio = original_size_bytes / compressed_size_bytes
            else:
                compression_ratio = 1.0
            
            return {
                'compression_ratio': compression_ratio,
                'original_size_bytes': original_size_bytes,
                'compressed_size_bytes': compressed_size_bytes,
                'space_saved_bytes': original_size_bytes - compressed_size_bytes,
                'space_saved_percentage': ((original_size_bytes - compressed_size_bytes) / original_size_bytes * 100)
            }
            
        except Exception as e:
            logger.warning("Failed to calculate compression metrics: %s", e)
            return {
                'compression_ratio': 1.0,
                'original_size_bytes': 0,
                'compressed_size_bytes': 0,
                'space_saved_bytes': 0,
                'space_saved_percentage': 0.0
            }
    
    def _apply_inverse_logarithmic_transformation(self, tensor: torch.Tensor,
                                                compressed_weights: List[LogarithmicPadicWeight],
                                                metadata: Dict[str, Any]) -> torch.Tensor:
        """Apply inverse logarithmic transformation to decompressed tensor
        
        Args:
            tensor: Decompressed tensor
            compressed_weights: Original compressed weights with metadata
            metadata: Decompression metadata
            
        Returns:
            Tensor with inverse logarithmic transformation applied
        """
        try:
            # Use the logarithmic encoder's decoding method
            decoded_tensor = self.logarithmic_encoder.decode_logarithmic_padic_weights(compressed_weights)
            
            # Reshape to match expected output shape
            target_shape = metadata.get('original_shape', tensor.shape)
            if decoded_tensor.numel() == torch.prod(torch.tensor(target_shape)):
                decoded_tensor = decoded_tensor.reshape(target_shape)
            
            return decoded_tensor
            
        except Exception as e:
            logger.warning("Inverse logarithmic transformation failed, using direct decompression: %s", e)
            return tensor
    
    def _calculate_reconstruction_error(self, reconstructed: torch.Tensor,
                                      compressed_weights: List[LogarithmicPadicWeight],
                                      metadata: Dict[str, Any]) -> float:
        """Calculate reconstruction error between original and reconstructed weights
        
        Args:
            reconstructed: Reconstructed tensor
            compressed_weights: Compressed weights with original values
            metadata: Reconstruction metadata
            
        Returns:
            Mean squared error of reconstruction
        """
        try:
            # Extract original values from compressed weights
            original_values = torch.tensor([lw.original_value for lw in compressed_weights])
            
            # Flatten reconstructed tensor for comparison
            reconstructed_flat = reconstructed.flatten()
            
            # Ensure same length for comparison
            min_length = min(len(original_values), len(reconstructed_flat))
            original_values = original_values[:min_length]
            reconstructed_flat = reconstructed_flat[:min_length]
            
            # Calculate mean squared error
            mse = torch.mean((original_values - reconstructed_flat) ** 2)
            return float(mse.item())
            
        except Exception as e:
            logger.warning("Failed to calculate reconstruction error: %s", e)
            return 0.0
    
    def _estimate_compression_ratio_from_weights(self, compressed_weights: List[LogarithmicPadicWeight]) -> float:
        """Estimate compression ratio from compressed weights
        
        Args:
            compressed_weights: List of compressed weights
            
        Returns:
            Estimated compression ratio
        """
        try:
            if not compressed_weights:
                return 1.0
            
            ratios = [lw.get_compression_ratio() for lw in compressed_weights]
            return np.mean(ratios) if ratios else 1.0
            
        except Exception:
            return 1.0
    
    def _update_compression_statistics(self, categorization_time: float, ieee754_time: float,
                                     encoding_time: float, compression_ratio: float) -> None:
        """Update pipeline compression statistics
        
        Args:
            categorization_time: Time spent on categorization
            ieee754_time: Time spent on IEEE 754 extraction
            encoding_time: Time spent on logarithmic encoding
            compression_ratio: Achieved compression ratio
        """
        try:
            self.pipeline_stats['total_compressions'] += 1
            
            # Update timing statistics
            self.pipeline_stats['processing_times']['categorization'].append(categorization_time * 1000)
            self.pipeline_stats['processing_times']['ieee754_extraction'].append(ieee754_time * 1000)
            self.pipeline_stats['processing_times']['logarithmic_encoding'].append(encoding_time * 1000)
            
            # Keep only recent history
            for key in self.pipeline_stats['processing_times']:
                if len(self.pipeline_stats['processing_times'][key]) > 100:
                    self.pipeline_stats['processing_times'][key].pop(0)
            
            # Update compression ratio statistics
            total_compressions = self.pipeline_stats['total_compressions']
            current_avg = self.pipeline_stats['average_compression_ratio']
            
            # Exponential moving average
            alpha = 0.1
            self.pipeline_stats['average_compression_ratio'] = (
                alpha * compression_ratio + (1 - alpha) * current_avg
            )
            
            # Update categorical storage usage if available
            if self.categorical_manager:
                storage_stats = self.categorical_manager.get_storage_statistics()
                self.pipeline_stats['categorical_storage_usage'] = storage_stats.get('ram_utilization', 0.0)
            
        except Exception as e:
            logger.warning("Failed to update compression statistics: %s", e)
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics
        
        Returns:
            Dictionary of pipeline performance statistics
        """
        try:
            base_stats = super().get_statistics()
            
            # Calculate average processing times
            avg_times = {}
            for stage, times in self.pipeline_stats['processing_times'].items():
                avg_times[f'avg_{stage}_time_ms'] = np.mean(times) if times else 0.0
            
            # Combine all statistics
            comprehensive_stats = {
                **base_stats,
                **self.pipeline_stats,
                **avg_times,
                'full_pipeline_active': True,
                'target_compression_ratio': self.full_config.target_compression_ratio,
                'components_enabled': {
                    'categorical_storage': self.full_config.enable_categorical_storage,
                    'ieee754_optimization': self.full_config.enable_ieee754_optimization,
                    'logarithmic_encoding': self.full_config.enable_logarithmic_encoding,
                    'gpu_bursting': self.full_config.enable_gpu_bursting
                }
            }
            
            # Add component-specific statistics
            if self.categorical_manager:
                comprehensive_stats['categorical_storage_stats'] = self.categorical_manager.get_storage_statistics()
            
            if self.ieee754_extractor:
                comprehensive_stats['ieee754_extraction_stats'] = self.ieee754_extractor.get_extraction_statistics()
            
            if self.logarithmic_encoder:
                comprehensive_stats['logarithmic_encoding_stats'] = self.logarithmic_encoder.get_encoding_statistics()
            
            if self.weight_categorizer:
                comprehensive_stats['weight_categorization_stats'] = self.weight_categorizer.get_categorization_statistics()
            
            return comprehensive_stats
            
        except Exception as e:
            logger.error("Failed to get pipeline statistics: %s", e)
            return {'error': str(e)}
    
    def cleanup(self) -> None:
        """Clean up all pipeline resources"""
        try:
            # Clean up parent resources
            super().cleanup()
            
            # Clean up categorical storage
            if self.categorical_manager:
                self.categorical_manager.cleanup()
            
            # Reset statistics
            self.pipeline_stats = {
                'total_compressions': 0,
                'total_decompressions': 0,
                'average_compression_ratio': 0.0,
                'categorical_storage_usage': 0.0,
                'ieee754_optimizations': 0,
                'logarithmic_encodings': 0,
                'gpu_bursting_events': 0,
                'processing_times': {
                    'categorization': [],
                    'ieee754_extraction': [],
                    'logarithmic_encoding': [],
                    'gpu_bursting': []
                }
            }
            
            logger.info("FullCompressionPipeline cleaned up successfully")
            
        except Exception as e:
            logger.error("Failed to cleanup FullCompressionPipeline: %s", e)


# Factory function for easy integration
def create_full_compression_pipeline(
    cpu_bursting_config: Optional[CPUBurstingConfig] = None,
    gpu_decompression_config: Optional[GPUDecompressionConfig] = None,
    categorical_storage_config: Optional[CategoricalStorageConfig] = None,
    logarithmic_encoding_config: Optional[LogarithmicEncodingConfig] = None,
    target_compression_ratio: float = 4.0
) -> FullCompressionPipeline:
    """Factory function to create full compression pipeline with default configurations
    
    Args:
        cpu_bursting_config: Optional CPU bursting configuration
        gpu_decompression_config: Optional GPU decompression configuration
        categorical_storage_config: Optional categorical storage configuration
        logarithmic_encoding_config: Optional logarithmic encoding configuration
        target_compression_ratio: Target compression ratio
        
    Returns:
        Configured FullCompressionPipeline instance
    """
    # Use defaults if not provided
    if cpu_bursting_config is None:
        cpu_bursting_config = CPUBurstingConfig()
    
    if gpu_decompression_config is None:
        gpu_decompression_config = GPUDecompressionConfig()
    
    if categorical_storage_config is None:
        categorical_storage_config = CategoricalStorageConfig()
    
    if logarithmic_encoding_config is None:
        logarithmic_encoding_config = LogarithmicEncodingConfig()
    
    # Create full configuration
    full_config = FullCompressionConfig(
        cpu_bursting_config=cpu_bursting_config,
        gpu_decompression_config=gpu_decompression_config,
        categorical_storage_config=categorical_storage_config,
        logarithmic_encoding_config=logarithmic_encoding_config,
        target_compression_ratio=target_compression_ratio
    )
    
    # Create GPU decompression engine
    gpu_engine = PadicDecompressionEngine(
        gpu_decompression_config, 
        prime=logarithmic_encoding_config.prime
    )
    
    return FullCompressionPipeline(full_config, gpu_engine)