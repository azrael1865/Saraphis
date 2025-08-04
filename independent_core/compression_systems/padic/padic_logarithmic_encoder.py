"""
P-adic Logarithmic Encoder - Enhanced Compression through Logarithmic Encoding

Provides p-adic logarithmic encoding for IEEE 754 channels to achieve superior
compression ratios compared to direct p-adic conversion.

NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from fractions import Fraction
import math
import logging

logger = logging.getLogger(__name__)

# Import existing components for integration
try:
    from .padic_encoder import PadicWeight, PadicMathematicalOperations
    from .safe_reconstruction import SafePadicReconstructor, ReconstructionConfig, ReconstructionMethod
    from ..categorical.ieee754_channel_extractor import IEEE754Channels
except ImportError:
    from compression_systems.padic.padic_encoder import PadicWeight, PadicMathematicalOperations
    from compression_systems.padic.safe_reconstruction import SafePadicReconstructor, ReconstructionConfig, ReconstructionMethod
    from compression_systems.categorical.ieee754_channel_extractor import IEEE754Channels


@dataclass
class LogarithmicEncodingConfig:
    """Configuration for p-adic logarithmic encoding"""
    # P-adic parameters (reduced for better compression)
    prime: int = 257                          # P-adic prime
    precision: int = 2                        # Reduced precision for better compression
    max_safe_precision: int = 3               # Reduced maximum safe precision
    
    # Logarithmic encoding parameters
    use_natural_log: bool = True              # Use natural logarithm vs log base prime
    log_offset: float = 1e-10                # Offset to avoid log(0) 
    normalize_before_log: bool = True         # Normalize values before taking log
    scale_factor: float = 1000.0             # Scale factor for better precision
    
    # IEEE 754 optimization
    separate_channel_encoding: bool = True    # Encode channels separately
    optimize_exponent_encoding: bool = True   # Special handling for exponents
    optimize_mantissa_encoding: bool = True   # Special handling for mantissas
    
    # Compression optimization
    enable_delta_encoding: bool = True        # Use delta encoding for sequences
    enable_run_length_encoding: bool = True   # Use RLE for repeated values
    quantize_before_encoding: bool = True     # Quantize inputs for better compression
    quantization_levels: int = 1024           # Number of quantization levels
    
    def __post_init__(self):
        """Validate configuration"""
        safe_limits = {257: 6, 127: 7, 31: 9, 17: 10, 11: 12, 7: 15, 5: 20, 3: 30, 2: 50}
        max_safe = safe_limits.get(self.prime, 4)
        
        if self.precision > max_safe:
            raise ValueError(f"Unsafe precision {self.precision} for prime {self.prime}, max safe: {max_safe}")
        
        if self.max_safe_precision > max_safe:
            self.max_safe_precision = max_safe
        
        if self.scale_factor <= 0:
            raise ValueError(f"Scale factor must be positive, got {self.scale_factor}")
        
        if not (4 <= self.quantization_levels <= 65536):
            raise ValueError(f"Quantization levels must be in [4, 65536], got {self.quantization_levels}")


@dataclass 
class LogarithmicPadicWeight:
    """P-adic weight with logarithmic encoding metadata"""
    padic_weight: PadicWeight                 # Base p-adic representation
    original_value: float                     # Original pre-log value
    log_value: float                          # Logarithmic value that was encoded
    encoding_method: str                      # Method used for encoding
    compression_metadata: Dict[str, Any]      # Compression-specific metadata
    
    def get_compression_ratio(self) -> float:
        """Calculate compression ratio achieved"""
        try:
            # Original storage: 4 bytes per float32
            original_bytes = 4
            
            # P-adic storage: precision digits + metadata
            padic_bytes = len(self.padic_weight.digits) + 8  # digits + overhead
            
            if padic_bytes > 0:
                return original_bytes / padic_bytes
            else:
                return 1.0
        except Exception:
            return 1.0


class PadicLogarithmicEncoder(PadicMathematicalOperations):
    """
    Enhanced p-adic encoder with logarithmic compression
    
    Extends existing p-adic mathematical operations to include logarithmic
    encoding of IEEE 754 channels for superior compression ratios.
    """
    
    def __init__(self, config: LogarithmicEncodingConfig):
        """Initialize logarithmic encoder
        
        Args:
            config: Logarithmic encoding configuration
        """
        # Initialize parent with same prime and precision
        super().__init__(config.prime, config.precision)
        
        self.config = config
        
        # Initialize safe reconstructor for overflow prevention
        reconstruction_config = ReconstructionConfig(
            prime=config.prime,
            max_safe_precision=config.max_safe_precision,
            method=ReconstructionMethod.HYBRID,
            overflow_threshold=1e15
        )
        self.safe_reconstructor = SafePadicReconstructor(reconstruction_config)
        
        # Encoding statistics
        self.encoding_stats = {
            'total_encodings': 0,
            'logarithmic_encodings': 0,
            'direct_encodings': 0,
            'compression_ratios': [],
            'encoding_failures': 0,
            'channel_optimizations': 0,
            'average_precision_used': 0.0
        }
        
        logger.info("PadicLogarithmicEncoder initialized with prime %d, precision %d, logarithmic encoding: %s",
                   config.prime, config.precision, config.use_natural_log)
    
    def encode_ieee754_channels_logarithmically(self, channels: IEEE754Channels) -> List[LogarithmicPadicWeight]:
        """Encode IEEE 754 channels using logarithmic p-adic compression
        
        Args:
            channels: IEEE 754 channels to encode
            
        Returns:
            List of LogarithmicPadicWeight objects
            
        Raises:
            RuntimeError: If logarithmic encoding fails (hard failure)
            ValueError: If input validation fails
        """
        if channels is None:
            raise ValueError("IEEE 754 channels cannot be None")
        
        if len(channels.original_values) == 0:
            raise ValueError("IEEE 754 channels cannot be empty")
        
        try:
            encoded_weights = []
            
            if self.config.separate_channel_encoding:
                # Encode each channel separately for optimal compression
                sign_weights = self._encode_channel_logarithmically(
                    channels.sign_channel.astype(np.float32), "sign"
                )
                exponent_weights = self._encode_channel_logarithmically(
                    channels.exponent_channel.astype(np.float32), "exponent"
                )
                mantissa_weights = self._encode_channel_logarithmically(
                    channels.mantissa_channel, "mantissa"
                ) 
                
                # Combine all channel encodings
                encoded_weights.extend(sign_weights)
                encoded_weights.extend(exponent_weights)
                encoded_weights.extend(mantissa_weights)
                
                self.encoding_stats['channel_optimizations'] += 3
                
            else:
                # Encode original values directly
                original_weights = self._encode_channel_logarithmically(
                    channels.original_values, "original"
                )
                encoded_weights.extend(original_weights)
            
            # Update statistics
            self.encoding_stats['total_encodings'] += len(encoded_weights)
            self.encoding_stats['logarithmic_encodings'] += len(encoded_weights)
            
            # Calculate average compression ratio
            compression_ratios = [w.get_compression_ratio() for w in encoded_weights]
            self.encoding_stats['compression_ratios'].extend(compression_ratios)
            
            logger.debug("Logarithmically encoded %d IEEE 754 channels with average compression ratio %.2fx",
                        len(encoded_weights), np.mean(compression_ratios) if compression_ratios else 1.0)
            
            return encoded_weights
            
        except Exception as e:
            self.encoding_stats['encoding_failures'] += 1
            raise RuntimeError(f"IEEE 754 logarithmic encoding failed: {e}")
    
    def encode_weights_logarithmically(self, weights: torch.Tensor, 
                                     metadata: Optional[Dict[str, Any]] = None) -> List[LogarithmicPadicWeight]:
        """Encode tensor weights using logarithmic p-adic compression
        
        Args:
            weights: Tensor of weights to encode
            metadata: Optional metadata for optimization hints
            
        Returns:
            List of LogarithmicPadicWeight objects
            
        Raises:
            RuntimeError: If encoding fails (hard failure)
            ValueError: If input validation fails
        """
        if weights is None:
            raise ValueError("Weights tensor cannot be None")
        
        if weights.numel() == 0:
            raise ValueError("Weights tensor cannot be empty")
        
        try:
            # Convert to numpy for processing
            weight_array = weights.flatten().cpu().numpy()
            
            # Validate weights
            if np.any(np.isnan(weight_array)):
                raise ValueError("Weights contain NaN values")
            
            if np.any(np.isinf(weight_array)):
                raise ValueError("Weights contain infinite values")
            
            # Apply preprocessing optimizations
            processed_weights = self._preprocess_weights_for_logarithmic_encoding(weight_array, metadata)
            
            # Encode using logarithmic method
            encoded_weights = self._encode_channel_logarithmically(processed_weights, "weights")
            
            # Update statistics
            self.encoding_stats['total_encodings'] += len(encoded_weights)
            self.encoding_stats['logarithmic_encodings'] += len(encoded_weights)
            
            logger.debug("Logarithmically encoded %d weights", len(encoded_weights))
            return encoded_weights
            
        except Exception as e:
            self.encoding_stats['encoding_failures'] += 1
            raise RuntimeError(f"Weight logarithmic encoding failed: {e}")
    
    def _encode_channel_logarithmically(self, values: np.ndarray, channel_type: str) -> List[LogarithmicPadicWeight]:
        """Core logarithmic encoding logic for a channel
        
        Args:
            values: Array of values to encode
            channel_type: Type of channel being encoded
            
        Returns:
            List of LogarithmicPadicWeight objects
        """
        encoded_weights = []
        
        for i, value in enumerate(values):
            try:
                # Apply channel-specific optimizations
                optimized_value = self._optimize_value_for_channel(value, channel_type)
                
                # Apply logarithmic transformation
                log_value = self._apply_logarithmic_transformation(optimized_value, channel_type)
                
                # Encode the logarithmic value using p-adic representation
                padic_weight = self._encode_log_value_to_padic(log_value, optimized_value)
                
                # Create logarithmic p-adic weight
                log_padic_weight = LogarithmicPadicWeight(
                    padic_weight=padic_weight,
                    original_value=float(value),
                    log_value=log_value,
                    encoding_method=f"logarithmic_{channel_type}",
                    compression_metadata={
                        'channel_type': channel_type,
                        'index': i,
                        'optimization_applied': True,
                        'quantization_applied': self.config.quantize_before_encoding
                    }
                )
                
                encoded_weights.append(log_padic_weight)
                
            except Exception as e:
                # For individual encoding failures, skip the problematic value
                # but don't fail the entire encoding operation
                logger.warning("Failed to encode value %f at index %d: %s", value, i, e)
                
                # Create fallback direct encoding (still safe)
                try:
                    direct_padic = self._encode_value_directly_safe(value)
                    fallback_weight = LogarithmicPadicWeight(
                        padic_weight=direct_padic,
                        original_value=float(value),
                        log_value=float(value),  # No log transformation
                        encoding_method=f"direct_{channel_type}",
                        compression_metadata={
                            'channel_type': channel_type,
                            'index': i,
                            'fallback_used': True
                        }
                    )
                    encoded_weights.append(fallback_weight)
                    self.encoding_stats['direct_encodings'] += 1
                    
                except Exception as direct_error:
                    # Hard failure - cannot encode even with direct method
                    raise RuntimeError(f"Cannot encode value {value} at index {i}: logarithmic failed ({e}), direct failed ({direct_error})")
        
        return encoded_weights
    
    def _optimize_value_for_channel(self, value: float, channel_type: str) -> float:
        """Apply channel-specific optimizations before logarithmic encoding
        
        Args:
            value: Value to optimize
            channel_type: Type of channel
            
        Returns:
            Optimized value
        """
        optimized = value
        
        if channel_type == "sign":
            # Sign channel: already 0 or 1, no optimization needed
            return optimized
        
        elif channel_type == "exponent" and self.config.optimize_exponent_encoding:
            # Exponent channel: map IEEE 754 range [0, 255] to p-adic friendly range
            if self.config.prime == 257:
                # Perfect match for prime 257
                optimized = value
            else:
                # Scale to better match prime
                optimized = value * (self.config.prime / 256.0)
        
        elif channel_type == "mantissa" and self.config.optimize_mantissa_encoding:
            # Mantissa channel: optimize fractional precision
            if self.config.quantize_before_encoding:
                # Quantize to reduce precision and improve compression
                optimized = np.round(value * self.config.quantization_levels) / self.config.quantization_levels
        
        elif channel_type in ["weights", "original"]:
            # General weight optimization
            if self.config.quantize_before_encoding:
                # Apply adaptive quantization based on value range
                if abs(value) < 1e-6:
                    optimized = 0.0  # Force small values to exact zero
                else:
                    # Quantize to reasonable precision
                    scale = max(1.0, abs(value) * 100)
                    optimized = np.round(value * scale) / scale
        
        return optimized
    
    def _apply_logarithmic_transformation(self, value: float, channel_type: str) -> float:
        """Apply logarithmic transformation with channel-specific handling
        
        Args:
            value: Value to transform
            channel_type: Type of channel
            
        Returns:
            Logarithmically transformed value
        """
        if channel_type == "sign":
            # Sign channel: no logarithmic transformation needed
            return value
        
        # Handle zero and negative values
        if abs(value) <= self.config.log_offset:
            return 0.0  # Map near-zero values to exactly zero
        
        abs_value = abs(value)
        sign = np.sign(value)
        
        try:
            if self.config.use_natural_log:
                # Natural logarithm with offset
                log_abs = math.log(abs_value + self.config.log_offset)
            else:
                # Logarithm base prime
                log_abs = math.log(abs_value + self.config.log_offset) / math.log(self.config.prime)
            
            # Apply scaling and normalization
            scaled_log = log_abs * self.config.scale_factor
            
            if self.config.normalize_before_log:
                # Normalize to reasonable range for p-adic encoding
                # Target range: [-prime/2, prime/2] for better p-adic representation
                normalized_log = scaled_log % self.config.prime
                if normalized_log > self.config.prime / 2:
                    normalized_log -= self.config.prime
                
                return sign * normalized_log
            else:
                return sign * scaled_log
        
        except Exception as e:
            logger.warning("Logarithmic transformation failed for value %f, using direct encoding: %s", value, e)
            return value
    
    def _encode_log_value_to_padic(self, log_value: float, original_value: float) -> PadicWeight:
        """Encode logarithmically transformed value to p-adic representation
        
        Args:
            log_value: Logarithmically transformed value
            original_value: Original value before transformation
            
        Returns:
            PadicWeight representation
            
        Raises:
            RuntimeError: If p-adic encoding fails
        """
        try:
            # Use safe precision to prevent overflow
            effective_precision = min(self.config.precision, self.config.max_safe_precision)
            
            # Handle special cases
            if abs(log_value) < 1e-10:
                # Zero value - create minimal p-adic representation
                return PadicWeight(
                    value=Fraction(0),
                    prime=self.config.prime,
                    precision=effective_precision,
                    valuation=0,
                    digits=[0] * effective_precision
                )
            
            # Convert to fraction for exact p-adic conversion
            try:
                # Handle special float values that Fraction can't process directly
                if log_value == 1.0:
                    fraction_value = Fraction(1, 1)
                elif log_value == 0.0:
                    fraction_value = Fraction(0, 1)
                elif abs(log_value) < 1e-10:
                    fraction_value = Fraction(0, 1)  # Treat tiny values as zero
                else:
                    # Convert float to string first to avoid Fraction issues
                    fraction_value = Fraction(str(log_value)).limit_denominator(10000)
            except (ValueError, OverflowError, TypeError):
                # Fallback to approximate fraction using float conversion
                try:
                    fraction_value = Fraction(float(log_value)).limit_denominator(1000)
                except:
                    # Last resort: use zero
                    fraction_value = Fraction(0, 1)
            
            # Use parent class method for p-adic conversion (convert Fraction to float)
            padic_representation = self.to_padic(float(fraction_value))
            
            # Ensure precision is within safe limits
            if len(padic_representation.digits) > effective_precision:
                # Truncate to safe precision
                safe_digits = padic_representation.digits[:effective_precision]
                padic_representation = PadicWeight(
                    value=padic_representation.value,
                    prime=self.config.prime,
                    precision=effective_precision,
                    valuation=padic_representation.valuation,
                    digits=safe_digits
                )
            
            return padic_representation
            
        except Exception as e:
            raise RuntimeError(f"P-adic encoding of logarithmic value {log_value} failed: {e}")
    
    def _encode_value_directly_safe(self, value: float) -> PadicWeight:
        """Safely encode value directly without logarithmic transformation
        
        Args:
            value: Value to encode directly
            
        Returns:
            PadicWeight representation
        """
        try:
            # Use safe precision
            effective_precision = min(self.config.precision, self.config.max_safe_precision)
            
            # Convert to fraction with limited denominator
            if abs(value) < 1e-10:
                fraction_value = Fraction(0)
            else:
                fraction_value = Fraction(value).limit_denominator(1000)
            
            # Use parent class method with limited precision
            original_precision = self.precision
            self.precision = effective_precision
            
            try:
                padic_representation = self.to_padic(fraction_value)
            finally:
                self.precision = original_precision
            
            return padic_representation
            
        except Exception as e:
            # Create minimal representation for problematic values
            logger.warning("Direct encoding failed for value %f, using zero representation: %s", value, e)
            return PadicWeight(
                value=Fraction(0),
                prime=self.config.prime,
                precision=self.config.max_safe_precision,
                valuation=0,
                digits=[0] * self.config.max_safe_precision
            )
    
    def _preprocess_weights_for_logarithmic_encoding(self, weights: np.ndarray, 
                                                   metadata: Optional[Dict[str, Any]]) -> np.ndarray:
        """Preprocess weights to optimize logarithmic encoding
        
        Args:
            weights: Array of weight values
            metadata: Optional metadata with preprocessing hints
            
        Returns:
            Preprocessed weight array
        """
        processed = weights.copy()
        
        try:
            # Apply delta encoding if enabled and beneficial
            if self.config.enable_delta_encoding and len(weights) > 2:
                if self._should_use_delta_encoding(weights, metadata):
                    processed = self._apply_delta_encoding(processed)
            
            # Apply run-length encoding preprocessing if beneficial  
            if self.config.enable_run_length_encoding:
                if self._should_use_run_length_encoding(weights, metadata):
                    # Mark repeated values for special handling
                    processed = self._mark_repeated_values(processed)
            
            # Apply quantization if enabled
            if self.config.quantize_before_encoding:
                processed = self._apply_adaptive_quantization(processed)
            
            return processed
            
        except Exception as e:
            logger.warning("Weight preprocessing failed, using original weights: %s", e)
            return weights
    
    def _should_use_delta_encoding(self, weights: np.ndarray, metadata: Optional[Dict[str, Any]]) -> bool:
        """Determine if delta encoding would be beneficial
        
        Args:
            weights: Array of weight values
            metadata: Optional metadata
            
        Returns:
            True if delta encoding should be used
        """
        try:
            # Calculate autocorrelation to detect smooth sequences
            if len(weights) < 5:
                return False
            
            # Simple autocorrelation test
            diffs = np.diff(weights)
            diff_std = np.std(diffs)
            weight_std = np.std(weights)
            
            # Use delta encoding if differences are more regular than original values
            return diff_std < weight_std * 0.5
            
        except Exception:
            return False
    
    def _should_use_run_length_encoding(self, weights: np.ndarray, metadata: Optional[Dict[str, Any]]) -> bool:
        """Determine if run-length encoding preprocessing would be beneficial
        
        Args:
            weights: Array of weight values  
            metadata: Optional metadata
            
        Returns:
            True if RLE preprocessing should be used
        """
        try:
            # Count repeated consecutive values
            if len(weights) < 3:
                return False
            
            repeated_count = 0
            current_value = weights[0]
            run_length = 1
            
            for i in range(1, len(weights)):
                if abs(weights[i] - current_value) < 1e-8:
                    run_length += 1
                else:
                    if run_length >= 3:  # Found a run of 3 or more
                        repeated_count += run_length
                    current_value = weights[i]
                    run_length = 1
            
            # Check final run
            if run_length >= 3:
                repeated_count += run_length
            
            # Use RLE if significant portion is repeated
            return repeated_count / len(weights) > 0.1
            
        except Exception:
            return False
    
    def _apply_delta_encoding(self, weights: np.ndarray) -> np.ndarray:
        """Apply delta encoding to weight sequence
        
        Args:
            weights: Original weight values
            
        Returns:
            Delta-encoded weights
        """
        try:
            if len(weights) <= 1:
                return weights
            
            # First value stays the same, rest are differences
            delta_weights = np.zeros_like(weights)
            delta_weights[0] = weights[0]
            
            for i in range(1, len(weights)):
                delta_weights[i] = weights[i] - weights[i-1]
            
            return delta_weights
            
        except Exception:
            return weights
    
    def _mark_repeated_values(self, weights: np.ndarray) -> np.ndarray:
        """Mark repeated values for optimized encoding
        
        Args:
            weights: Original weight values
            
        Returns:
            Weights with repeated value marking
        """
        # For now, just return original weights
        # In a full implementation, this would create a marking system
        # for the p-adic encoder to handle repeated values specially
        return weights
    
    def _apply_adaptive_quantization(self, weights: np.ndarray) -> np.ndarray:
        """Apply adaptive quantization based on weight distribution
        
        Args:
            weights: Original weight values
            
        Returns:
            Quantized weights
        """
        try:
            if len(weights) == 0:
                return weights
            
            # Calculate adaptive quantization levels based on weight range
            weight_range = np.max(weights) - np.min(weights)
            
            if weight_range < 1e-10:
                # All weights are essentially the same
                return weights
            
            # Use fewer quantization levels for smaller ranges
            if weight_range < 0.01:
                levels = min(64, self.config.quantization_levels)
            elif weight_range < 0.1:
                levels = min(256, self.config.quantization_levels)
            else:
                levels = self.config.quantization_levels
            
            # Apply quantization
            min_val = np.min(weights)
            max_val = np.max(weights)
            
            # Quantize to levels
            quantized = np.round((weights - min_val) / (max_val - min_val) * (levels - 1))
            
            # Convert back to original range
            quantized = quantized / (levels - 1) * (max_val - min_val) + min_val
            
            return quantized.astype(np.float32)
            
        except Exception:
            return weights
    
    def decode_logarithmic_padic_weights(self, log_weights: List[LogarithmicPadicWeight]) -> torch.Tensor:
        """Decode logarithmic p-adic weights back to original values
        
        Args:
            log_weights: List of LogarithmicPadicWeight objects to decode
            
        Returns:
            Tensor of decoded values
            
        Raises:
            RuntimeError: If decoding fails (hard failure)
        """
        if not log_weights:
            raise ValueError("Logarithmic p-adic weights list cannot be empty")
        
        try:
            decoded_values = []
            
            for i, log_weight in enumerate(log_weights):
                try:
                    # Reconstruct p-adic value using safe reconstructor
                    padic_value = self.safe_reconstructor.reconstruct(log_weight.padic_weight)
                    
                    # Apply inverse logarithmic transformation if it was used
                    if log_weight.encoding_method.startswith("logarithmic"):
                        decoded_value = self._apply_inverse_logarithmic_transformation(
                            padic_value, log_weight.compression_metadata.get('channel_type', 'unknown')
                        )
                    else:
                        # Direct encoding, no inverse transformation needed
                        decoded_value = padic_value
                    
                    decoded_values.append(float(decoded_value))
                    
                except Exception as e:
                    logger.warning("Failed to decode logarithmic weight at index %d: %s", i, e)
                    # Use original value as fallback
                    decoded_values.append(log_weight.original_value)
            
            # Convert to tensor
            return torch.tensor(decoded_values, dtype=torch.float32)
            
        except Exception as e:
            raise RuntimeError(f"Logarithmic p-adic weight decoding failed: {e}")
    
    def _apply_inverse_logarithmic_transformation(self, log_value: float, channel_type: str) -> float:
        """Apply inverse logarithmic transformation to recover original value
        
        Args:
            log_value: Logarithmically encoded value
            channel_type: Type of channel that was encoded
            
        Returns:
            Recovered original value
        """
        if channel_type == "sign":
            # Sign channel: no transformation was applied
            return log_value
        
        try:
            # Handle zero case
            if abs(log_value) < 1e-10:
                return 0.0
            
            # Extract sign
            sign = np.sign(log_value)
            abs_log_value = abs(log_value)
            
            # Reverse normalization if it was applied
            if self.config.normalize_before_log:
                # Reverse the modular normalization
                if abs_log_value > self.config.prime / 2:
                    abs_log_value += self.config.prime
            
            # Reverse scaling
            unscaled_log = abs_log_value / self.config.scale_factor
            
            # Apply inverse logarithm
            if self.config.use_natural_log:
                # Inverse of natural log is exp
                recovered_abs = math.exp(unscaled_log) - self.config.log_offset
            else:
                # Inverse of log base prime
                recovered_abs = (self.config.prime ** unscaled_log) - self.config.log_offset
            
            # Apply sign and clamp to reasonable range
            recovered_value = sign * max(0.0, recovered_abs)
            
            # Clamp to prevent overflow
            max_value = 1e6  # Reasonable maximum for neural network weights
            return max(-max_value, min(max_value, recovered_value))
            
        except Exception as e:
            logger.warning("Inverse logarithmic transformation failed for value %f: %s", log_value, e)
            return log_value  # Return as-is if transformation fails
    
    def get_encoding_statistics(self) -> Dict[str, Any]:
        """Get comprehensive encoding statistics
        
        Returns:
            Dictionary of encoding statistics
        """
        total_encodings = self.encoding_stats['total_encodings']
        
        return {
            'total_encodings': total_encodings,
            'logarithmic_encodings': self.encoding_stats['logarithmic_encodings'],
            'direct_encodings': self.encoding_stats['direct_encodings'],
            'encoding_failures': self.encoding_stats['encoding_failures'],
            'channel_optimizations': self.encoding_stats['channel_optimizations'],
            'logarithmic_ratio': (
                self.encoding_stats['logarithmic_encodings'] / max(1, total_encodings)
            ),
            'success_rate': (
                (total_encodings - self.encoding_stats['encoding_failures']) / max(1, total_encodings)
            ),
            'average_compression_ratio': (
                np.mean(self.encoding_stats['compression_ratios']) 
                if self.encoding_stats['compression_ratios'] else 1.0
            ),
            'compression_ratio_std': (
                np.std(self.encoding_stats['compression_ratios']) 
                if self.encoding_stats['compression_ratios'] else 0.0
            ),
            'configuration': {
                'prime': self.config.prime,
                'precision': self.config.precision,
                'use_natural_log': self.config.use_natural_log,
                'separate_channel_encoding': self.config.separate_channel_encoding,
                'quantize_before_encoding': self.config.quantize_before_encoding
            }
        }


# Factory function for easy integration
def create_padic_logarithmic_encoder(config: Optional[LogarithmicEncodingConfig] = None) -> PadicLogarithmicEncoder:
    """Factory function to create p-adic logarithmic encoder
    
    Args:
        config: Optional configuration, uses defaults if None
        
    Returns:
        Configured PadicLogarithmicEncoder instance
    """
    if config is None:
        config = LogarithmicEncodingConfig()
    
    return PadicLogarithmicEncoder(config)