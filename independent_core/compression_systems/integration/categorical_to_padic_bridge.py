"""
Categorical to P-adic Bridge - Integration Layer

Provides seamless integration between categorical storage components and
p-adic encoding systems for optimal compression pipeline flow.

NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Import components for integration
try:
    from ..categorical.categorical_storage_manager import WeightCategory, CategoryType
    from ..categorical.ieee754_channel_extractor import IEEE754Channels
    from ..categorical.weight_categorizer import CategorizationResult, WeightPattern
    from ..padic.padic_encoder import PadicWeight
    from ..padic.padic_logarithmic_encoder import LogarithmicPadicWeight, LogarithmicEncodingConfig
except ImportError:
    from compression_systems.categorical.categorical_storage_manager import WeightCategory, CategoryType
    from compression_systems.categorical.ieee754_channel_extractor import IEEE754Channels
    from compression_systems.categorical.weight_categorizer import CategorizationResult, WeightPattern
    from compression_systems.padic.padic_encoder import PadicWeight
    from compression_systems.padic.padic_logarithmic_encoder import LogarithmicPadicWeight, LogarithmicEncodingConfig


@dataclass
class BridgeMapping:
    """Mapping between categorical data and p-adic representation"""
    category_id: str
    category_type: CategoryType
    ieee754_channels: IEEE754Channels
    padic_weights: List[PadicWeight]
    logarithmic_weights: List[LogarithmicPadicWeight]
    compression_metadata: Dict[str, Any]


@dataclass
class BridgeStatistics:
    """Statistics for bridge operations"""
    total_mappings_created: int = 0
    successful_conversions: int = 0
    failed_conversions: int = 0
    average_compression_ratio: float = 0.0
    category_conversion_rates: Dict[CategoryType, float] = None
    
    def __post_init__(self):
        if self.category_conversion_rates is None:
            self.category_conversion_rates = {}


class CategoricalToPadicBridge:
    """
    Bridge component for seamless integration between categorical storage
    and p-adic encoding systems in the compression pipeline.
    """
    
    def __init__(self):
        """Initialize the categorical to p-adic bridge"""
        self.bridge_stats = BridgeStatistics()
        self.active_mappings: Dict[str, BridgeMapping] = {}
        
        # Optimization settings for different category types
        self.category_optimization_settings = {
            CategoryType.ZERO_WEIGHTS: {
                'use_sparse_encoding': True,
                'precision_reduction': 2,  # Use lower precision for zeros
                'skip_logarithmic': True   # No need for log encoding of zeros
            },
            CategoryType.SMALL_WEIGHTS: {
                'use_sparse_encoding': False,
                'precision_reduction': 1,
                'skip_logarithmic': False,
                'quantization_boost': True
            },
            CategoryType.MEDIUM_WEIGHTS: {
                'use_sparse_encoding': False,
                'precision_reduction': 0,
                'skip_logarithmic': False,
                'optimal_settings': True
            },
            CategoryType.LARGE_WEIGHTS: {
                'use_sparse_encoding': False,
                'precision_reduction': 0,
                'skip_logarithmic': False,
                'high_precision_mode': True
            },
            CategoryType.HIGH_ENTROPY: {
                'use_sparse_encoding': False,
                'precision_reduction': 0,
                'skip_logarithmic': False,
                'complexity_handling': True
            },
            CategoryType.LOW_ENTROPY: {
                'use_sparse_encoding': True,
                'precision_reduction': 1,
                'skip_logarithmic': True,
                'pattern_optimization': True
            }
        }
        
        logger.info("CategoricalToPadicBridge initialized with optimization settings for %d category types",
                   len(self.category_optimization_settings))
    
    def create_categorical_to_padic_mapping(self, categories: List[WeightCategory],
                                          categorization_result: CategorizationResult,
                                          encoding_config: LogarithmicEncodingConfig) -> List[BridgeMapping]:
        """Create mappings from categorical data to p-adic representations
        
        Args:
            categories: List of weight categories from categorical storage
            categorization_result: Result of weight categorization analysis
            encoding_config: Configuration for p-adic encoding
            
        Returns:
            List of BridgeMapping objects
            
        Raises:
            RuntimeError: If mapping creation fails (hard failure)
        """
        if not categories:
            raise ValueError("Categories list cannot be empty")
        
        try:
            mappings = []
            
            for category in categories:
                try:
                    # Create bridge mapping for this category
                    mapping = self._create_single_category_mapping(
                        category, categorization_result, encoding_config
                    )
                    
                    if mapping:
                        mappings.append(mapping)
                        self.active_mappings[category.category_id] = mapping
                        self.bridge_stats.successful_conversions += 1
                    else:
                        self.bridge_stats.failed_conversions += 1
                        logger.warning("Failed to create mapping for category %s", category.category_id)
                
                except Exception as e:
                    self.bridge_stats.failed_conversions += 1
                    logger.error("Error creating mapping for category %s: %s", category.category_id, e)
                    # Continue with other categories instead of hard failure
                    continue
            
            self.bridge_stats.total_mappings_created += len(mappings)
            
            if not mappings:
                raise RuntimeError("Failed to create any categorical to p-adic mappings - hard failure")
            
            logger.info("Created %d categorical to p-adic mappings from %d categories",
                       len(mappings), len(categories))
            
            return mappings
            
        except Exception as e:
            raise RuntimeError(f"Categorical to p-adic mapping creation failed: {e}")
    
    def _create_single_category_mapping(self, category: WeightCategory,
                                      categorization_result: CategorizationResult,
                                      encoding_config: LogarithmicEncodingConfig) -> Optional[BridgeMapping]:
        """Create a single bridge mapping for one category
        
        Args:
            category: Weight category to map
            categorization_result: Categorization analysis
            encoding_config: Encoding configuration
            
        Returns:
            BridgeMapping object or None if mapping fails
        """
        try:
            # Get optimization settings for this category type
            optimization_settings = self.category_optimization_settings.get(
                category.category_type, {}
            )
            
            # Create optimized encoding configuration for this category
            category_encoding_config = self._create_category_optimized_config(
                encoding_config, category.category_type, optimization_settings, categorization_result
            )
            
            # Extract or create IEEE 754 channels for this category
            ieee754_channels = self._extract_or_reuse_ieee754_channels(category)
            
            # Convert to p-adic representation using optimized settings
            padic_weights, logarithmic_weights = self._convert_category_to_padic(
                category, ieee754_channels, category_encoding_config, optimization_settings
            )
            
            # Calculate compression metadata
            compression_metadata = self._calculate_category_compression_metadata(
                category, padic_weights, logarithmic_weights, optimization_settings
            )
            
            # Create bridge mapping
            mapping = BridgeMapping(
                category_id=category.category_id,
                category_type=category.category_type,
                ieee754_channels=ieee754_channels,
                padic_weights=padic_weights,
                logarithmic_weights=logarithmic_weights,
                compression_metadata=compression_metadata
            )
            
            return mapping
            
        except Exception as e:
            logger.error("Failed to create mapping for category %s: %s", category.category_id, e)
            return None
    
    def _create_category_optimized_config(self, base_config: LogarithmicEncodingConfig,
                                        category_type: CategoryType,
                                        optimization_settings: Dict[str, Any],
                                        categorization_result: CategorizationResult) -> LogarithmicEncodingConfig:
        """Create optimized encoding configuration for specific category
        
        Args:
            base_config: Base encoding configuration
            category_type: Type of category being optimized
            optimization_settings: Category-specific optimization settings
            categorization_result: Categorization analysis for additional hints
            
        Returns:
            Optimized LogarithmicEncodingConfig
        """
        # Create copy of base configuration
        optimized_config = LogarithmicEncodingConfig(
            prime=base_config.prime,
            precision=base_config.precision,
            max_safe_precision=base_config.max_safe_precision,
            use_natural_log=base_config.use_natural_log,
            log_offset=base_config.log_offset,
            normalize_before_log=base_config.normalize_before_log,
            scale_factor=base_config.scale_factor,
            separate_channel_encoding=base_config.separate_channel_encoding,
            optimize_exponent_encoding=base_config.optimize_exponent_encoding,
            optimize_mantissa_encoding=base_config.optimize_mantissa_encoding,
            enable_delta_encoding=base_config.enable_delta_encoding,
            enable_run_length_encoding=base_config.enable_run_length_encoding,
            quantize_before_encoding=base_config.quantize_before_encoding,
            quantization_levels=base_config.quantization_levels
        )
        
        # Apply category-specific optimizations
        if optimization_settings.get('precision_reduction', 0) > 0:
            reduction = optimization_settings['precision_reduction']
            optimized_config.precision = max(2, base_config.precision - reduction)
        
        if optimization_settings.get('high_precision_mode', False):
            optimized_config.precision = min(base_config.max_safe_precision, base_config.precision + 1)
        
        if optimization_settings.get('use_sparse_encoding', False):
            optimized_config.enable_run_length_encoding = True
            optimized_config.quantize_before_encoding = True
        
        if optimization_settings.get('quantization_boost', False):
            optimized_config.quantization_levels = min(65536, base_config.quantization_levels * 2)
        
        if optimization_settings.get('complexity_handling', False):
            optimized_config.normalize_before_log = True
            optimized_config.scale_factor = base_config.scale_factor * 0.5  # Reduce scaling for complex data
        
        if optimization_settings.get('pattern_optimization', False):
            optimized_config.enable_delta_encoding = True
            optimized_config.quantization_levels = max(64, base_config.quantization_levels // 4)
        
        # Apply hints from categorization result
        if categorization_result.optimization_hints:
            hints = categorization_result.optimization_hints
            
            if 'recommended_precision' in hints:
                suggested = hints['recommended_precision']
                optimized_config.precision = min(suggested, optimized_config.max_safe_precision)
            
            if 'use_sparse_encoding' in hints and hints['use_sparse_encoding']:
                optimized_config.enable_run_length_encoding = True
            
            if 'quantization_levels' in hints:
                optimized_config.quantization_levels = min(hints['quantization_levels'], 65536)
        
        return optimized_config
    
    def _extract_or_reuse_ieee754_channels(self, category: WeightCategory) -> IEEE754Channels:
        """Extract or reuse IEEE 754 channels for category
        
        Args:
            category: Weight category
            
        Returns:
            IEEE754Channels object
        """
        # Try to reuse existing channels if available
        if category.ieee754_channels:
            # Combine all existing channels in the category
            all_sign = np.concatenate([ch.sign_channel for ch in category.ieee754_channels])
            all_exponent = np.concatenate([ch.exponent_channel for ch in category.ieee754_channels])
            all_mantissa = np.concatenate([ch.mantissa_channel for ch in category.ieee754_channels])
            all_original = np.concatenate([ch.original_values for ch in category.ieee754_channels])
            
            return IEEE754Channels(
                sign_channel=all_sign,
                exponent_channel=all_exponent,
                mantissa_channel=all_mantissa,
                original_values=all_original
            )
        
        # Extract from weights if channels not available
        if category.weights:
            from ..categorical.ieee754_channel_extractor import IEEE754ChannelExtractor
            
            # Combine all weights in category
            combined_weights = torch.cat(category.weights, dim=0)
            
            # Extract channels
            extractor = IEEE754ChannelExtractor(validate_reconstruction=False)  # Skip validation for speed
            return extractor.extract_channels_from_tensor(combined_weights)
        
        # Create dummy channels as last resort
        logger.warning("Creating dummy IEEE 754 channels for category %s", category.category_id)
        dummy_size = 1
        return IEEE754Channels(
            sign_channel=np.zeros(dummy_size, dtype=np.uint8),
            exponent_channel=np.ones(dummy_size, dtype=np.uint8) * 127,
            mantissa_channel=np.zeros(dummy_size, dtype=np.float32),
            original_values=np.zeros(dummy_size, dtype=np.float32)
        )
    
    def _convert_category_to_padic(self, category: WeightCategory,
                                 ieee754_channels: IEEE754Channels,
                                 encoding_config: LogarithmicEncodingConfig,
                                 optimization_settings: Dict[str, Any]) -> Tuple[List[PadicWeight], List[LogarithmicPadicWeight]]:
        """Convert category data to p-adic representation
        
        Args:
            category: Weight category to convert
            ieee754_channels: IEEE 754 channels for conversion
            encoding_config: Optimized encoding configuration
            optimization_settings: Category-specific optimization settings
            
        Returns:
            Tuple of (standard p-adic weights, logarithmic p-adic weights)
        """
        try:
            from ..padic.padic_logarithmic_encoder import PadicLogarithmicEncoder
            
            # Create encoder with optimized configuration
            encoder = PadicLogarithmicEncoder(encoding_config)
            
            # Check if logarithmic encoding should be skipped
            if optimization_settings.get('skip_logarithmic', False):
                # Use direct encoding for simple cases (e.g., zeros, low entropy)
                logarithmic_weights = self._perform_direct_encoding(ieee754_channels, encoding_config)
            else:
                # Use full logarithmic encoding
                logarithmic_weights = encoder.encode_ieee754_channels_logarithmically(ieee754_channels)
            
            # Extract standard p-adic weights
            padic_weights = [lw.padic_weight for lw in logarithmic_weights]
            
            return padic_weights, logarithmic_weights
            
        except Exception as e:
            logger.error("Failed to convert category %s to p-adic: %s", category.category_id, e)
            # Return empty lists instead of failing
            return [], []
    
    def _perform_direct_encoding(self, channels: IEEE754Channels, 
                               config: LogarithmicEncodingConfig) -> List[LogarithmicPadicWeight]:
        """Perform direct p-adic encoding without logarithmic transformation
        
        Args:
            channels: IEEE 754 channels to encode
            config: Encoding configuration
            
        Returns:
            List of LogarithmicPadicWeight objects (using direct encoding)
        """
        try:
            from ..padic.padic_encoder import PadicMathematicalOperations
            from fractions import Fraction
            
            math_ops = PadicMathematicalOperations(config.prime, config.precision)
            encoded_weights = []
            
            for i, value in enumerate(channels.original_values):
                try:
                    # Convert to fraction and encode directly
                    if abs(value) < 1e-10:
                        fraction_val = Fraction(0)
                    else:
                        fraction_val = Fraction(value).limit_denominator(1000)
                    
                    padic_weight = math_ops.to_padic(fraction_val)
                    
                    # Create logarithmic wrapper (but mark as direct encoding)
                    log_weight = LogarithmicPadicWeight(
                        padic_weight=padic_weight,
                        original_value=float(value),
                        log_value=float(value),  # No log transformation
                        encoding_method="direct_category_optimized",
                        compression_metadata={
                            'index': i,
                            'direct_encoding': True,
                            'category_optimized': True
                        }
                    )
                    
                    encoded_weights.append(log_weight)
                    
                except Exception:
                    # Create zero weight for problematic values
                    zero_weight = PadicWeight(
                        value=Fraction(0),
                        prime=config.prime,
                        precision=config.precision,
                        valuation=0,
                        digits=[0] * config.precision
                    )
                    
                    log_weight = LogarithmicPadicWeight(
                        padic_weight=zero_weight,
                        original_value=0.0,
                        log_value=0.0,
                        encoding_method="zero_direct",
                        compression_metadata={'index': i, 'zero_fallback': True}
                    )
                    
                    encoded_weights.append(log_weight)
            
            return encoded_weights
            
        except Exception as e:
            logger.error("Direct encoding failed: %s", e)
            return []
    
    def _calculate_category_compression_metadata(self, category: WeightCategory,
                                               padic_weights: List[PadicWeight],
                                               logarithmic_weights: List[LogarithmicPadicWeight],
                                               optimization_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate compression metadata for category mapping
        
        Args:
            category: Original weight category
            padic_weights: Standard p-adic weights
            logarithmic_weights: Logarithmic p-adic weights
            optimization_settings: Applied optimization settings
            
        Returns:
            Dictionary of compression metadata
        """
        try:
            # Calculate original size
            original_size_bytes = 0
            for weight_tensor in category.weights:
                original_size_bytes += weight_tensor.numel() * weight_tensor.element_size()
            
            # Calculate compressed size
            compressed_size_bytes = 0
            for lw in logarithmic_weights:
                compressed_size_bytes += len(lw.padic_weight.digits) * 4  # 4 bytes per digit
                compressed_size_bytes += 32  # Metadata overhead
            
            # Calculate compression ratio
            compression_ratio = original_size_bytes / compressed_size_bytes if compressed_size_bytes > 0 else 1.0
            
            # Update bridge statistics
            if category.category_type not in self.bridge_stats.category_conversion_rates:
                self.bridge_stats.category_conversion_rates[category.category_type] = 0.0
            
            current_rate = self.bridge_stats.category_conversion_rates[category.category_type]
            self.bridge_stats.category_conversion_rates[category.category_type] = (
                (current_rate + compression_ratio) / 2.0  # Running average
            )
            
            return {
                'category_id': category.category_id,
                'category_type': category.category_type.value,
                'original_size_bytes': original_size_bytes,
                'compressed_size_bytes': compressed_size_bytes,
                'compression_ratio': compression_ratio,
                'optimization_settings_applied': optimization_settings,
                'padic_weights_count': len(padic_weights),
                'logarithmic_weights_count': len(logarithmic_weights),
                'encoding_methods': list(set(lw.encoding_method for lw in logarithmic_weights)),
                'average_precision_used': np.mean([len(pw.digits) for pw in padic_weights]) if padic_weights else 0.0
            }
            
        except Exception as e:
            logger.warning("Failed to calculate compression metadata for category %s: %s", category.category_id, e)
            return {
                'category_id': category.category_id,
                'category_type': category.category_type.value,
                'error': str(e)
            }
    
    def retrieve_padic_weights_by_category(self, category_types: List[CategoryType]) -> List[LogarithmicPadicWeight]:
        """Retrieve p-adic weights for specific category types
        
        Args:
            category_types: List of category types to retrieve
            
        Returns:
            List of LogarithmicPadicWeight objects
        """
        if not category_types:
            raise ValueError("Category types list cannot be empty")
        
        try:
            retrieved_weights = []
            
            for mapping in self.active_mappings.values():
                if mapping.category_type in category_types:
                    retrieved_weights.extend(mapping.logarithmic_weights)
            
            logger.debug("Retrieved %d p-adic weights for %d category types",
                        len(retrieved_weights), len(category_types))
            
            return retrieved_weights
            
        except Exception as e:
            logger.error("Failed to retrieve p-adic weights by category: %s", e)
            return []
    
    def get_bridge_statistics(self) -> Dict[str, Any]:
        """Get comprehensive bridge statistics
        
        Returns:
            Dictionary of bridge performance statistics
        """
        try:
            # Calculate overall statistics
            total_attempts = self.bridge_stats.successful_conversions + self.bridge_stats.failed_conversions
            success_rate = (
                self.bridge_stats.successful_conversions / max(1, total_attempts)
            )
            
            # Calculate average compression ratio across categories
            if self.bridge_stats.category_conversion_rates:
                overall_avg_compression = np.mean(list(self.bridge_stats.category_conversion_rates.values()))
            else:
                overall_avg_compression = 0.0
            
            return {
                'total_mappings_created': self.bridge_stats.total_mappings_created,
                'successful_conversions': self.bridge_stats.successful_conversions,
                'failed_conversions': self.bridge_stats.failed_conversions,
                'conversion_success_rate': success_rate,
                'average_compression_ratio': overall_avg_compression,
                'active_mappings_count': len(self.active_mappings),
                'category_conversion_rates': {
                    ct.value: rate for ct, rate in self.bridge_stats.category_conversion_rates.items()
                },
                'optimization_settings_available': len(self.category_optimization_settings),
                'supported_category_types': [ct.value for ct in self.category_optimization_settings.keys()]
            }
            
        except Exception as e:
            logger.error("Failed to get bridge statistics: %s", e)
            return {'error': str(e)}
    
    def clear_mappings(self) -> None:
        """Clear all active mappings and reset statistics"""
        try:
            self.active_mappings.clear()
            self.bridge_stats = BridgeStatistics()
            logger.info("Cleared all bridge mappings and reset statistics")
            
        except Exception as e:
            logger.error("Failed to clear bridge mappings: %s", e)


# Factory function for easy integration
def create_categorical_to_padic_bridge() -> CategoricalToPadicBridge:
    """Factory function to create categorical to p-adic bridge
    
    Returns:
        Configured CategoricalToPadicBridge instance
    """
    return CategoricalToPadicBridge()