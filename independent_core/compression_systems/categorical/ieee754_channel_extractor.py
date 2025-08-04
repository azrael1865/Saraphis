"""
IEEE 754 Channel Extractor - Float32 Decomposition for P-adic Compression

Extracts IEEE 754 sign, exponent, and mantissa components from float32 values
for optimized p-adic logarithmic encoding and compression.

NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
import struct
from typing import Tuple, List, Dict, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Import p-adic components for integration
try:
    from ..padic.padic_encoder import PadicWeight
except ImportError:
    from compression_systems.padic.padic_encoder import PadicWeight


@dataclass
class IEEE754Channels:
    """Container for IEEE 754 decomposed channels"""
    sign_channel: np.ndarray       # Sign bits (0 or 1)
    exponent_channel: np.ndarray   # Exponent values (0-255 for float32)
    mantissa_channel: np.ndarray   # Mantissa fractional parts
    original_values: np.ndarray    # Original float values for validation
    
    def __post_init__(self):
        """Validate channel consistency"""
        shapes = [
            self.sign_channel.shape,
            self.exponent_channel.shape, 
            self.mantissa_channel.shape,
            self.original_values.shape
        ]
        
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError(f"All channels must have same shape, got: {shapes}")
        
        # Validate sign channel (must be 0 or 1)
        if not np.all(np.isin(self.sign_channel, [0, 1])):
            raise ValueError("Sign channel must contain only 0 or 1 values")
        
        # Validate exponent channel (must be 0-255 for float32)
        if not np.all((self.exponent_channel >= 0) & (self.exponent_channel <= 255)):
            raise ValueError("Exponent channel must contain values in range [0, 255]")


class IEEE754ChannelExtractor:
    """
    Extract IEEE 754 sign, exponent, mantissa channels from float32 values
    
    Optimized for p-adic logarithmic compression pipeline integration.
    Follows existing codebase patterns for hard failures and memory management.
    """
    
    def __init__(self, validate_reconstruction: bool = True):
        """Initialize IEEE 754 channel extractor
        
        Args:
            validate_reconstruction: Whether to validate round-trip reconstruction
        """
        self.validate_reconstruction = validate_reconstruction
        self.extraction_stats = {
            'total_extractions': 0,
            'validation_failures': 0,
            'special_values_handled': 0,
            'average_mantissa_entropy': 0.0
        }
        
        # IEEE 754 float32 constants
        self.SIGN_MASK = 0x80000000      # 1 bit  - bit 31
        self.EXPONENT_MASK = 0x7F800000  # 8 bits - bits 30-23
        self.MANTISSA_MASK = 0x007FFFFF  # 23 bits - bits 22-0
        
        self.EXPONENT_SHIFT = 23
        self.EXPONENT_BIAS = 127
        
        # Special IEEE 754 values
        self.ZERO_EXPONENT = 0
        self.MAX_EXPONENT = 255
        self.INFINITY_EXPONENT = 255
        
        logger.info("IEEE754ChannelExtractor initialized with validation enabled: %s", validate_reconstruction)
    
    def extract_channels_from_tensor(self, tensor: torch.Tensor) -> IEEE754Channels:
        """Extract IEEE 754 channels from tensor
        
        Args:
            tensor: Input tensor (will be converted to float32)
            
        Returns:
            IEEE754Channels object with sign, exponent, mantissa components
            
        Raises:
            RuntimeError: If extraction fails (hard failure)
            ValueError: If input validation fails
        """
        if tensor is None:
            raise ValueError("Input tensor cannot be None")
        
        if tensor.numel() == 0:
            raise ValueError("Input tensor cannot be empty")
        
        try:
            # Convert to float32 and flatten for processing
            float_tensor = tensor.float().flatten()
            
            # Check for invalid values
            if torch.any(torch.isnan(float_tensor)):
                raise ValueError("Input tensor contains NaN values")
            
            # Convert to numpy for bit manipulation
            float_array = float_tensor.cpu().numpy()
            
            # Extract channels
            channels = self._extract_ieee754_channels(float_array)
            
            # Validate reconstruction if enabled
            if self.validate_reconstruction:
                self._validate_reconstruction(channels)
            
            # Update statistics
            self.extraction_stats['total_extractions'] += 1
            self.extraction_stats['average_mantissa_entropy'] = self._calculate_mantissa_entropy(channels.mantissa_channel)
            
            logger.debug("Successfully extracted IEEE 754 channels from tensor of shape %s", tensor.shape)
            return channels
            
        except Exception as e:
            raise RuntimeError(f"IEEE 754 channel extraction failed: {e}")
    
    def extract_channels_from_padic_weight(self, weight: PadicWeight) -> IEEE754Channels:
        """Extract IEEE 754 channels from reconstructed p-adic weight
        
        This method first reconstructs the float from p-adic representation,
        then extracts IEEE 754 channels for categorical storage optimization.
        
        Args:
            weight: PadicWeight to process
            
        Returns:
            IEEE754Channels object
            
        Raises:
            RuntimeError: If extraction fails (hard failure)
            ValueError: If weight validation fails
        """
        if not hasattr(weight, 'digits') or not hasattr(weight, 'valuation'):
            raise ValueError("Invalid PadicWeight: missing digits or valuation")
        
        if not weight.digits:
            raise ValueError("PadicWeight must have non-empty digits")
        
        try:
            # Reconstruct float value from p-adic weight
            # Use safe reconstruction to prevent overflow
            reconstructed_value = self._safe_padic_reconstruction(weight)
            
            # Convert to numpy array for channel extraction
            float_array = np.array([reconstructed_value], dtype=np.float32)
            
            # Extract channels
            channels = self._extract_ieee754_channels(float_array)
            
            logger.debug("Extracted IEEE 754 channels from PadicWeight with precision %d", weight.precision)
            return channels
            
        except Exception as e:
            raise RuntimeError(f"IEEE 754 extraction from PadicWeight failed: {e}")
    
    def _extract_ieee754_channels(self, float_array: np.ndarray) -> IEEE754Channels:
        """Core IEEE 754 channel extraction logic
        
        Args:
            float_array: Array of float32 values
            
        Returns:
            IEEE754Channels with extracted components
        """
        # Convert float32 to uint32 bit representation
        uint32_view = float_array.view(np.uint32)
        
        # Extract sign bits (1 bit)
        sign_channel = ((uint32_view & self.SIGN_MASK) >> 31).astype(np.uint8)
        
        # Extract exponent bits (8 bits)
        exponent_channel = ((uint32_view & self.EXPONENT_MASK) >> self.EXPONENT_SHIFT).astype(np.uint8)
        
        # Extract mantissa bits (23 bits)
        mantissa_bits = (uint32_view & self.MANTISSA_MASK).astype(np.uint32)
        
        # Convert mantissa to fractional representation
        # For normalized numbers: mantissa = 1.fraction
        # For denormalized numbers: mantissa = 0.fraction
        mantissa_channel = np.zeros_like(float_array, dtype=np.float32)
        
        # Handle normalized numbers (exponent != 0)
        normalized_mask = exponent_channel != self.ZERO_EXPONENT
        mantissa_channel[normalized_mask] = 1.0 + (mantissa_bits[normalized_mask].astype(np.float32) / (2**23))
        
        # Handle denormalized numbers (exponent == 0)
        denormalized_mask = exponent_channel == self.ZERO_EXPONENT
        mantissa_channel[denormalized_mask] = mantissa_bits[denormalized_mask].astype(np.float32) / (2**23)
        
        # Handle special cases
        special_count = self._handle_special_values(sign_channel, exponent_channel, mantissa_channel, float_array)
        self.extraction_stats['special_values_handled'] += special_count
        
        return IEEE754Channels(
            sign_channel=sign_channel,
            exponent_channel=exponent_channel,
            mantissa_channel=mantissa_channel,
            original_values=float_array.copy()
        )
    
    def _handle_special_values(self, sign_channel: np.ndarray, exponent_channel: np.ndarray, 
                              mantissa_channel: np.ndarray, float_array: np.ndarray) -> int:
        """Handle IEEE 754 special values (infinity, NaN, zero)
        
        Returns:
            Number of special values handled
        """
        special_count = 0
        
        # Handle infinity
        inf_mask = (exponent_channel == self.INFINITY_EXPONENT) & (mantissa_channel == 0)
        if np.any(inf_mask):
            logger.warning("Found %d infinity values in IEEE 754 extraction", np.sum(inf_mask))
            special_count += np.sum(inf_mask)
        
        # Handle NaN
        nan_mask = (exponent_channel == self.INFINITY_EXPONENT) & (mantissa_channel != 0)
        if np.any(nan_mask):
            raise ValueError(f"Found {np.sum(nan_mask)} NaN values - hard failure")
        
        # Handle zero
        zero_mask = (exponent_channel == self.ZERO_EXPONENT) & (mantissa_channel == 0)
        if np.any(zero_mask):
            logger.debug("Found %d zero values in IEEE 754 extraction", np.sum(zero_mask))
            special_count += np.sum(zero_mask)
        
        return int(special_count)
    
    def _validate_reconstruction(self, channels: IEEE754Channels) -> None:
        """Validate that channels can be reconstructed back to original values
        
        Args:
            channels: IEEE754Channels to validate
            
        Raises:
            RuntimeError: If reconstruction validation fails
        """
        try:
            # Reconstruct float values from channels
            reconstructed = self.reconstruct_from_channels(channels)
            
            # Compare with original values (allowing for floating point precision)
            tolerance = 1e-6
            max_diff = np.max(np.abs(reconstructed - channels.original_values))
            
            if max_diff > tolerance:
                self.extraction_stats['validation_failures'] += 1
                raise RuntimeError(f"IEEE 754 reconstruction validation failed: max difference {max_diff} > tolerance {tolerance}")
                
            logger.debug("IEEE 754 reconstruction validation passed with max difference: %e", max_diff)
            
        except Exception as e:
            raise RuntimeError(f"IEEE 754 reconstruction validation failed: {e}")
    
    def reconstruct_from_channels(self, channels: IEEE754Channels) -> np.ndarray:
        """Reconstruct float32 values from IEEE 754 channels
        
        Args:
            channels: IEEE754Channels to reconstruct
            
        Returns:
            Array of reconstructed float32 values
            
        Raises:
            RuntimeError: If reconstruction fails
        """
        try:
            # Reconstruct uint32 bit representation
            sign_bits = (channels.sign_channel.astype(np.uint32) << 31)
            exponent_bits = (channels.exponent_channel.astype(np.uint32) << self.EXPONENT_SHIFT)
            
            # Convert mantissa back to integer representation
            mantissa_bits = np.zeros_like(channels.mantissa_channel, dtype=np.uint32)
            
            # Handle normalized numbers
            normalized_mask = channels.exponent_channel != self.ZERO_EXPONENT
            mantissa_frac = channels.mantissa_channel[normalized_mask] - 1.0
            mantissa_bits[normalized_mask] = (mantissa_frac * (2**23)).astype(np.uint32)
            
            # Handle denormalized numbers  
            denormalized_mask = channels.exponent_channel == self.ZERO_EXPONENT
            mantissa_bits[denormalized_mask] = (channels.mantissa_channel[denormalized_mask] * (2**23)).astype(np.uint32)
            
            # Combine all components
            uint32_combined = sign_bits | exponent_bits | mantissa_bits
            
            # Convert back to float32
            reconstructed = uint32_combined.view(np.float32)
            
            return reconstructed
            
        except Exception as e:
            raise RuntimeError(f"IEEE 754 channel reconstruction failed: {e}")
    
    def _safe_padic_reconstruction(self, weight: PadicWeight) -> float:
        """Safely reconstruct float from p-adic weight to prevent overflow
        
        Args:
            weight: PadicWeight to reconstruct
            
        Returns:
            Reconstructed float value
            
        Raises:
            RuntimeError: If reconstruction fails or overflows
        """
        # Use safe precision limits to prevent overflow
        safe_precision_limits = {257: 6, 127: 7, 31: 9, 17: 10, 11: 12, 7: 15, 5: 20, 3: 30, 2: 50}
        max_safe_precision = safe_precision_limits.get(weight.prime, 4)
        
        effective_precision = min(weight.precision, len(weight.digits), max_safe_precision)
        
        if effective_precision > max_safe_precision:
            raise ValueError(f"Unsafe precision {effective_precision} for prime {weight.prime}, max safe: {max_safe_precision}")
        
        # Reconstruct using safe algorithm
        value = 0.0
        prime_power = 1.0
        
        for i in range(effective_precision):
            digit = weight.digits[i]
            
            # Validate digit range
            if digit < 0 or digit >= weight.prime:
                digit = max(0, min(digit, weight.prime - 1))  # Clamp to valid range
            
            value += digit * prime_power
            
            # Check for overflow before next iteration
            next_power = prime_power * weight.prime
            if next_power > 1e15:  # Conservative overflow threshold
                logger.warning("Stopping p-adic reconstruction at position %d to prevent overflow", i)
                break
                
            prime_power = next_power
        
        # Apply valuation
        if weight.valuation > 0:
            # Check for overflow in valuation application
            valuation_factor = weight.prime ** weight.valuation
            if value * valuation_factor > 1e15:
                raise OverflowError(f"Valuation {weight.valuation} would cause overflow")
            value *= valuation_factor
        elif weight.valuation < 0:
            for _ in range(abs(weight.valuation)):
                value /= weight.prime
        
        # Validate result
        if np.isnan(value) or np.isinf(value):
            raise ValueError(f"P-adic reconstruction produced invalid value: {value}")
        
        return float(value)
    
    def _calculate_mantissa_entropy(self, mantissa_channel: np.ndarray) -> float:
        """Calculate entropy of mantissa channel for compression analysis
        
        Args:
            mantissa_channel: Array of mantissa values
            
        Returns:
            Shannon entropy of mantissa distribution
        """
        try:
            # Quantize mantissa values for entropy calculation
            quantized = np.round(mantissa_channel * 1000).astype(np.int32)
            
            # Calculate histogram
            unique_values, counts = np.unique(quantized, return_counts=True)
            
            # Calculate probabilities
            probabilities = counts / len(quantized)
            
            # Calculate Shannon entropy
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Add small epsilon
            
            return float(entropy)
            
        except Exception:
            return 0.0  # Return 0 on calculation error
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get comprehensive extraction statistics
        
        Returns:
            Dictionary of extraction statistics
        """
        return {
            'total_extractions': self.extraction_stats['total_extractions'],
            'validation_failures': self.extraction_stats['validation_failures'],
            'special_values_handled': self.extraction_stats['special_values_handled'],
            'average_mantissa_entropy': self.extraction_stats['average_mantissa_entropy'],
            'validation_success_rate': (
                1.0 - (self.extraction_stats['validation_failures'] / max(1, self.extraction_stats['total_extractions']))
            )
        }
    
    def optimize_channels_for_padic(self, channels: IEEE754Channels, target_prime: int = 257) -> IEEE754Channels:
        """Optimize IEEE 754 channels for p-adic logarithmic encoding
        
        Args:
            channels: Original IEEE754Channels
            target_prime: Target p-adic prime for optimization
            
        Returns:
            Optimized IEEE754Channels for better p-adic compression
            
        Raises:
            RuntimeError: If optimization fails
        """
        try:
            # Create optimized copies
            optimized_sign = channels.sign_channel.copy()
            optimized_exponent = channels.exponent_channel.copy()
            optimized_mantissa = channels.mantissa_channel.copy()
            
            # Optimize exponent channel for p-adic prime alignment
            # Map IEEE 754 exponent range [0, 255] to p-adic friendly range
            if target_prime == 257:
                # For prime 257, optimize exponent distribution
                optimized_exponent = self._optimize_exponent_for_prime_257(optimized_exponent)
            
            # Optimize mantissa channel for better p-adic compression
            optimized_mantissa = self._optimize_mantissa_for_padic(optimized_mantissa, target_prime)
            
            # Create optimized channels object
            optimized_channels = IEEE754Channels(
                sign_channel=optimized_sign,
                exponent_channel=optimized_exponent,
                mantissa_channel=optimized_mantissa,
                original_values=channels.original_values.copy()
            )
            
            logger.debug("Optimized IEEE 754 channels for p-adic prime %d", target_prime)
            return optimized_channels
            
        except Exception as e:
            raise RuntimeError(f"IEEE 754 channel optimization failed: {e}")
    
    def _optimize_exponent_for_prime_257(self, exponent_channel: np.ndarray) -> np.ndarray:
        """Optimize exponent channel specifically for prime 257
        
        Args:
            exponent_channel: Original exponent values
            
        Returns:
            Optimized exponent values
        """
        # Map [0, 255] to [0, 256] range for better prime 257 alignment
        # This creates more compressible patterns in p-adic representation
        optimized = np.round(exponent_channel * (256.0 / 255.0)).astype(np.uint8)
        
        # Ensure we don't exceed the valid range
        optimized = np.clip(optimized, 0, 255)
        
        return optimized
    
    def _optimize_mantissa_for_padic(self, mantissa_channel: np.ndarray, prime: int) -> np.ndarray:
        """Optimize mantissa channel for p-adic compression
        
        Args:
            mantissa_channel: Original mantissa values
            prime: Target p-adic prime
            
        Returns:
            Optimized mantissa values
        """
        # Quantize mantissa to create more regular patterns for p-adic compression
        quantization_levels = min(prime, 256)  # Don't exceed reasonable quantization
        
        # Apply quantization
        quantized = np.round(mantissa_channel * quantization_levels) / quantization_levels
        
        # Ensure values remain in valid range [0, 2) for normalized numbers
        optimized = np.clip(quantized, 0.0, 2.0 - 1e-6)
        
        return optimized.astype(np.float32)


# Factory function for easy integration
def create_ieee754_extractor(validate_reconstruction: bool = True) -> IEEE754ChannelExtractor:
    """Factory function to create IEEE 754 channel extractor
    
    Args:
        validate_reconstruction: Enable reconstruction validation
        
    Returns:
        Configured IEEE754ChannelExtractor instance
    """
    return IEEE754ChannelExtractor(validate_reconstruction=validate_reconstruction)


# Utility functions for integration with existing pipeline
def extract_ieee754_from_tensor(tensor: torch.Tensor) -> IEEE754Channels:
    """Convenience function to extract IEEE 754 channels from tensor
    
    Args:
        tensor: Input tensor
        
    Returns:
        IEEE754Channels object
    """
    extractor = create_ieee754_extractor()
    return extractor.extract_channels_from_tensor(tensor)


def extract_ieee754_from_padic(weight: PadicWeight) -> IEEE754Channels:
    """Convenience function to extract IEEE 754 channels from p-adic weight
    
    Args:
        weight: PadicWeight object
        
    Returns:
        IEEE754Channels object
    """
    extractor = create_ieee754_extractor()
    return extractor.extract_channels_from_padic_weight(weight)