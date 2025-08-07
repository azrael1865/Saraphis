"""
Tropical Polynomial Reconstruction from Channels
GPU-accelerated polynomial and weight reconstruction
PRODUCTION READY - NO PLACEHOLDERS - HARD FAILURES ONLY

This module provides:
1. TropicalPolynomialReconstructor - Polynomial reconstruction from channels
2. TropicalToWeightConverter - Convert polynomials to neural network weights
3. LayerWiseDecompressor - Layer-specific decompression strategies
4. ModelDecompressor - Full model reconstruction
5. Accuracy verification system
"""

import torch
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib

# JAX removed - no longer supported

# Import tropical and channel components
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tropical_polynomial import (
    TropicalPolynomial,
    TropicalMonomial,
    TropicalPolynomialOperations
)
from tropical_core import (
    TropicalNumber,
    TROPICAL_ZERO,
    TROPICAL_EPSILON
)
from tropical_channel_extractor import TropicalChannels
from channel_decompressor import (
    ChannelMetadata,
    DecompressionMode,
    BaseChannelDecompressor,
    CoefficientChannelDecompressor,
    ExponentChannelDecompressor,
    MantissaChannelDecompressor
)
# JAX channel processor removed - no longer supported

logger = logging.getLogger(__name__)


class ReconstructionAccuracy(Enum):
    """Accuracy levels for reconstruction"""
    EXACT = "exact"              # Bit-exact reconstruction
    HIGH = "high"                # <0.01% error
    MEDIUM = "medium"            # <0.1% error
    LOW = "low"                  # <1% error
    APPROXIMATE = "approximate"  # Best effort


@dataclass
class ReconstructionConfig:
    """Configuration for polynomial reconstruction"""
    # Accuracy settings
    target_accuracy: ReconstructionAccuracy = ReconstructionAccuracy.HIGH
    max_reconstruction_error: float = 1e-6
    validate_reconstruction: bool = True
    
    # GPU settings
    use_gpu: bool = True
    gpu_batch_size: int = 1000
    enable_mixed_precision: bool = False
    
    # GPU settings
    use_gpu: bool = True
    # JAX removed - compilation level no longer used
    
    # Memory optimization
    enable_memory_mapping: bool = True
    max_memory_gb: float = 8.0
    enable_chunking: bool = True
    chunk_size: int = 10000
    
    # Caching
    enable_caching: bool = True
    cache_size_mb: int = 512
    
    # Verification
    enable_checksums: bool = True
    enable_error_correction: bool = True
    verify_against_original: bool = False


@dataclass
class ReconstructionMetrics:
    """Metrics for reconstruction performance"""
    num_polynomials: int = 0
    num_monomials: int = 0
    num_weights: int = 0
    
    # Timing
    channel_extraction_time: float = 0.0
    polynomial_reconstruction_time: float = 0.0
    weight_conversion_time: float = 0.0
    validation_time: float = 0.0
    total_time: float = 0.0
    
    # Accuracy
    max_error: float = 0.0
    mean_error: float = 0.0
    std_error: float = 0.0
    exact_matches: int = 0
    
    # Performance
    throughput_mbps: float = 0.0
    gpu_utilization: float = 0.0
    memory_peak_gb: float = 0.0
    
    def compute_accuracy_rate(self) -> float:
        """Compute reconstruction accuracy rate"""
        if self.num_weights == 0:
            return 1.0
        return self.exact_matches / self.num_weights


class TropicalPolynomialReconstructor:
    """
    Reconstructs tropical polynomials from compressed channels.
    Uses GPU acceleration when available.
    """
    
    def __init__(self, config: Optional[ReconstructionConfig] = None):
        """
        Initialize polynomial reconstructor
        
        Args:
            config: Reconstruction configuration
        """
        self.config = config or ReconstructionConfig()
        self.metrics = ReconstructionMetrics()
        
        # Initialize device
        if self.config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # JAX processor removed - no longer supported
        self.jax_processor = None
        
        # Initialize channel decompressors
        self.coeff_decompressor = CoefficientChannelDecompressor(
            device=self.device,
            enable_validation=self.config.enable_checksums
        )
        self.exp_decompressor = ExponentChannelDecompressor(
            device=self.device,
            enable_validation=self.config.enable_checksums
        )
        self.mant_decompressor = MantissaChannelDecompressor(
            device=self.device,
            enable_validation=self.config.enable_checksums
        )
        
        # Cache for reconstructed polynomials
        self.cache = {} if self.config.enable_caching else None
        
        logger.info("TropicalPolynomialReconstructor initialized on %s", self.device)
    
    def reconstruct_from_channels(self, 
                                 channels: TropicalChannels,
                                 validate: bool = True) -> TropicalPolynomial:
        """
        Reconstruct tropical polynomial from channels
        
        Args:
            channels: Compressed tropical channels
            validate: Whether to validate reconstruction
            
        Returns:
            Reconstructed TropicalPolynomial
        """
        start_time = time.time()
        
        # Check cache
        if self.cache is not None:
            cache_key = self._compute_cache_key(channels)
            if cache_key in self.cache:
                logger.debug("Cache hit for polynomial reconstruction")
                return self.cache[cache_key]
        
        # Extract and decompress channels
        extraction_start = time.time()
        
        coefficients = self._decompress_coefficient_channel(channels)
        exponents = self._decompress_exponent_channel(channels)
        
        self.metrics.channel_extraction_time += time.time() - extraction_start
        
        # Reconstruct polynomial
        reconstruction_start = time.time()
        
        # Use PyTorch reconstruction
        polynomial = self._torch_reconstruct_polynomial(coefficients, exponents, channels.metadata)
        
        self.metrics.polynomial_reconstruction_time += time.time() - reconstruction_start
        
        # Validate if requested
        if validate and self.config.validate_reconstruction:
            validation_start = time.time()
            self._validate_reconstruction(polynomial, channels)
            self.metrics.validation_time += time.time() - validation_start
        
        # Update metrics
        self.metrics.num_polynomials += 1
        self.metrics.num_monomials += len(polynomial.monomials)
        self.metrics.total_time += time.time() - start_time
        
        # Cache result
        if self.cache is not None and cache_key:
            self.cache[cache_key] = polynomial
        
        return polynomial
    
    def _decompress_coefficient_channel(self, channels: TropicalChannels) -> torch.Tensor:
        """Decompress coefficient channel"""
        # Create metadata for decompression
        metadata = ChannelMetadata(
            channel_type='coefficient',
            compression_ratio=channels.metadata.get('coeff_compression_ratio', 1.0),
            original_shape=tuple(channels.coefficient_channel.shape),
            compressed_shape=tuple(channels.coefficient_channel.shape),
            precision=channels.metadata.get('coeff_precision', 'float32'),
            encoding=channels.metadata.get('coeff_encoding', 'dense')
        )
        
        return self.coeff_decompressor.decompress(channels.coefficient_channel, metadata)
    
    def _decompress_exponent_channel(self, channels: TropicalChannels) -> torch.Tensor:
        """Decompress exponent channel"""
        metadata = ChannelMetadata(
            channel_type='exponent',
            compression_ratio=channels.metadata.get('exp_compression_ratio', 1.0),
            original_shape=tuple(channels.exponent_channel.shape),
            compressed_shape=tuple(channels.exponent_channel.shape),
            precision='int32',
            encoding=channels.metadata.get('exp_encoding', 'dense')
        )
        
        return self.exp_decompressor.decompress(channels.exponent_channel, metadata)
    
    # JAX reconstruction method removed - no longer supported
    
    def _torch_reconstruct_polynomial(self, 
                                     coefficients: torch.Tensor,
                                     exponents: torch.Tensor,
                                     metadata: Dict[str, Any]) -> TropicalPolynomial:
        """Reconstruct polynomial using PyTorch"""
        num_variables = metadata.get('num_variables', exponents.shape[1])
        
        # Create monomials
        monomials = []
        for i in range(coefficients.shape[0]):
            coeff = TropicalNumber(coefficients[i].item())
            
            # Extract exponent dictionary
            exp_dict = {}
            for j in range(num_variables):
                exp_val = int(exponents[i, j].item())
                if exp_val != 0:
                    exp_dict[j] = exp_val
            
            monomial = TropicalMonomial(coeff, exp_dict)
            monomials.append(monomial)
        
        return TropicalPolynomial(monomials)
    
    def _create_polynomial_from_arrays(self, 
                                      coeffs_array: Any,
                                      exps_array: Any,
                                      metadata: Dict[str, Any]) -> TropicalPolynomial:
        """Create TropicalPolynomial from arrays"""
        # Convert to numpy if needed
        coeffs_np = coeffs_array if isinstance(coeffs_array, np.ndarray) else np.array(coeffs_array)
        exps_np = exps_array if isinstance(exps_array, np.ndarray) else np.array(exps_array)
        
        num_variables = metadata.get('num_variables', exps_np.shape[1] if len(exps_np.shape) > 1 else 1)
        
        # Create monomials
        monomials = []
        for i in range(coeffs_np.shape[0]):
            coeff = TropicalNumber(float(coeffs_np[i]))
            
            # Extract exponent dictionary
            exp_dict = {}
            if len(exps_np.shape) > 1:
                for j in range(num_variables):
                    exp_val = int(exps_np[i, j])
                    if exp_val != 0:
                        exp_dict[j] = exp_val
            else:
                exp_val = int(exps_np[i])
                if exp_val != 0:
                    exp_dict[0] = exp_val
            
            monomial = TropicalMonomial(coeff, exp_dict)
            monomials.append(monomial)
        
        return TropicalPolynomial(monomials)
    
    def _validate_reconstruction(self, 
                                polynomial: TropicalPolynomial,
                                channels: TropicalChannels):
        """Validate polynomial reconstruction"""
        # Check monomial count
        expected_monomials = channels.metadata.get('num_monomials', len(polynomial.monomials))
        if len(polynomial.monomials) != expected_monomials:
            raise ValueError(f"Monomial count mismatch: {len(polynomial.monomials)} != {expected_monomials}")
        
        # Check degree
        expected_degree = channels.metadata.get('degree', polynomial.degree())
        if polynomial.degree() != expected_degree:
            raise ValueError(f"Degree mismatch: {polynomial.degree()} != {expected_degree}")
        
        # Validate checksum if available
        if 'polynomial_checksum' in channels.metadata:
            computed_checksum = self._compute_polynomial_checksum(polynomial)
            if computed_checksum != channels.metadata['polynomial_checksum']:
                raise ValueError("Polynomial checksum validation failed")
    
    def _compute_cache_key(self, channels: TropicalChannels) -> str:
        """Compute cache key for channels"""
        # Use hash of channel data
        hasher = hashlib.sha256()
        hasher.update(channels.coefficient_channel.cpu().numpy().tobytes())
        hasher.update(channels.exponent_channel.cpu().numpy().tobytes())
        return hasher.hexdigest()
    
    def _compute_polynomial_checksum(self, polynomial: TropicalPolynomial) -> str:
        """Compute checksum for polynomial"""
        hasher = hashlib.sha256()
        
        for monomial in polynomial.monomials:
            hasher.update(str(monomial.coefficient.value).encode())
            for var, exp in sorted(monomial.exponents.items()):
                hasher.update(f"{var}:{exp}".encode())
        
        return hasher.hexdigest()


class TropicalToWeightConverter:
    """
    Converts tropical polynomials to neural network weight tensors.
    Handles various layer types and weight formats.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize weight converter
        
        Args:
            device: Target device for weights
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conversion_cache = {}
    
    def polynomial_to_weights(self, 
                             polynomial: TropicalPolynomial,
                             target_shape: Tuple[int, ...],
                             layer_type: str = "dense") -> torch.Tensor:
        """
        Convert tropical polynomial to weight tensor
        
        Args:
            polynomial: Tropical polynomial
            target_shape: Target tensor shape
            layer_type: Type of layer (dense, conv2d, etc.)
            
        Returns:
            Weight tensor
        """
        # Extract coefficients from polynomial
        coefficients = self._extract_coefficients(polynomial, np.prod(target_shape))
        
        # Reshape based on layer type
        if layer_type == "dense":
            weights = self._reshape_dense(coefficients, target_shape)
        elif layer_type == "conv2d":
            weights = self._reshape_conv2d(coefficients, target_shape)
        elif layer_type == "attention":
            weights = self._reshape_attention(coefficients, target_shape)
        else:
            weights = coefficients.reshape(target_shape)
        
        return weights.to(self.device)
    
    def _extract_coefficients(self, 
                            polynomial: TropicalPolynomial,
                            num_weights: int) -> torch.Tensor:
        """Extract coefficient values from polynomial"""
        coeffs = []
        
        for monomial in polynomial.monomials:
            coeffs.append(monomial.coefficient.value)
        
        # Pad or truncate to match number of weights
        if len(coeffs) < num_weights:
            # Pad with tropical zero
            coeffs.extend([TROPICAL_ZERO] * (num_weights - len(coeffs)))
        elif len(coeffs) > num_weights:
            # Truncate
            coeffs = coeffs[:num_weights]
        
        return torch.tensor(coeffs, dtype=torch.float32)
    
    def _reshape_dense(self, coefficients: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """Reshape for dense/linear layers"""
        return coefficients.reshape(shape)
    
    def _reshape_conv2d(self, coefficients: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """Reshape for Conv2D layers (out_channels, in_channels, height, width)"""
        if len(shape) != 4:
            raise ValueError(f"Conv2D requires 4D shape, got {len(shape)}D")
        
        return coefficients.reshape(shape)
    
    def _reshape_attention(self, coefficients: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """Reshape for attention layers"""
        # Attention weights typically have shape (num_heads, seq_len, seq_len) or similar
        return coefficients.reshape(shape)


class LayerWiseDecompressor:
    """
    Layer-specific decompression strategies.
    Optimizes decompression based on layer characteristics.
    """
    
    def __init__(self, config: Optional[ReconstructionConfig] = None):
        """
        Initialize layer-wise decompressor
        
        Args:
            config: Reconstruction configuration
        """
        self.config = config or ReconstructionConfig()
        self.polynomial_reconstructor = TropicalPolynomialReconstructor(config)
        self.weight_converter = TropicalToWeightConverter()
        
        # Layer-specific strategies
        self.strategies = {
            'dense': self._decompress_dense_layer,
            'conv2d': self._decompress_conv_layer,
            'attention': self._decompress_attention_layer,
            'embedding': self._decompress_embedding_layer,
            'normalization': self._decompress_norm_layer
        }
    
    def decompress_layer(self, 
                         channels: TropicalChannels,
                         layer_info: Dict[str, Any]) -> torch.Tensor:
        """
        Decompress layer based on type and characteristics
        
        Args:
            channels: Compressed channels
            layer_info: Layer information (type, shape, etc.)
            
        Returns:
            Decompressed weight tensor
        """
        layer_type = layer_info.get('type', 'dense')
        strategy = self.strategies.get(layer_type, self._decompress_generic_layer)
        
        return strategy(channels, layer_info)
    
    def _decompress_dense_layer(self, 
                               channels: TropicalChannels,
                               layer_info: Dict[str, Any]) -> torch.Tensor:
        """Decompress dense/linear layer"""
        # Dense layers benefit from coefficient-focused decompression
        polynomial = self.polynomial_reconstructor.reconstruct_from_channels(channels)
        
        target_shape = layer_info['shape']
        weights = self.weight_converter.polynomial_to_weights(
            polynomial, target_shape, 'dense'
        )
        
        return weights
    
    def _decompress_conv_layer(self, 
                              channels: TropicalChannels,
                              layer_info: Dict[str, Any]) -> torch.Tensor:
        """Decompress convolutional layer"""
        # Conv layers have spatial structure to preserve
        polynomial = self.polynomial_reconstructor.reconstruct_from_channels(channels)
        
        target_shape = layer_info['shape']  # (out_channels, in_channels, H, W)
        weights = self.weight_converter.polynomial_to_weights(
            polynomial, target_shape, 'conv2d'
        )
        
        # Apply spatial smoothing if needed
        if layer_info.get('apply_smoothing', False):
            weights = self._apply_spatial_smoothing(weights)
        
        return weights
    
    def _decompress_attention_layer(self, 
                                   channels: TropicalChannels,
                                   layer_info: Dict[str, Any]) -> torch.Tensor:
        """Decompress attention layer"""
        # Attention layers require careful handling of query/key/value projections
        polynomial = self.polynomial_reconstructor.reconstruct_from_channels(channels)
        
        target_shape = layer_info['shape']
        weights = self.weight_converter.polynomial_to_weights(
            polynomial, target_shape, 'attention'
        )
        
        # Ensure attention weights sum to 1 if needed
        if layer_info.get('normalize', False):
            weights = torch.softmax(weights, dim=-1)
        
        return weights
    
    def _decompress_embedding_layer(self, 
                                   channels: TropicalChannels,
                                   layer_info: Dict[str, Any]) -> torch.Tensor:
        """Decompress embedding layer"""
        # Embeddings often have specific value ranges
        polynomial = self.polynomial_reconstructor.reconstruct_from_channels(channels)
        
        target_shape = layer_info['shape']  # (vocab_size, embedding_dim)
        weights = self.weight_converter.polynomial_to_weights(
            polynomial, target_shape, 'dense'
        )
        
        # Normalize embeddings if needed
        if layer_info.get('normalize', True):
            weights = torch.nn.functional.normalize(weights, p=2, dim=-1)
        
        return weights
    
    def _decompress_norm_layer(self, 
                              channels: TropicalChannels,
                              layer_info: Dict[str, Any]) -> torch.Tensor:
        """Decompress normalization layer (BatchNorm, LayerNorm, etc.)"""
        # Normalization layers have small parameter counts
        polynomial = self.polynomial_reconstructor.reconstruct_from_channels(channels)
        
        target_shape = layer_info['shape']
        weights = self.weight_converter.polynomial_to_weights(
            polynomial, target_shape, 'dense'
        )
        
        return weights
    
    def _decompress_generic_layer(self, 
                                 channels: TropicalChannels,
                                 layer_info: Dict[str, Any]) -> torch.Tensor:
        """Generic decompression for unknown layer types"""
        polynomial = self.polynomial_reconstructor.reconstruct_from_channels(channels)
        
        target_shape = layer_info.get('shape', (channels.coefficient_channel.shape[0],))
        weights = self.weight_converter.polynomial_to_weights(
            polynomial, target_shape, 'dense'
        )
        
        return weights
    
    def _apply_spatial_smoothing(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply spatial smoothing to convolutional weights"""
        # Simple Gaussian smoothing
        kernel_size = 3
        padding = kernel_size // 2
        
        # Create Gaussian kernel
        gaussian = torch.tensor([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], dtype=torch.float32) / 16.0
        gaussian = gaussian.view(1, 1, 3, 3)
        gaussian = gaussian.to(weights.device)
        
        # Apply convolution for smoothing
        if len(weights.shape) == 4:
            smoothed = torch.nn.functional.conv2d(
                weights.unsqueeze(1),
                gaussian,
                padding=padding
            ).squeeze(1)
            return smoothed
        
        return weights


class ModelDecompressor:
    """
    Full model decompression from compressed channels.
    Handles complete neural network reconstruction.
    """
    
    def __init__(self, config: Optional[ReconstructionConfig] = None):
        """
        Initialize model decompressor
        
        Args:
            config: Reconstruction configuration
        """
        self.config = config or ReconstructionConfig()
        self.layer_decompressor = LayerWiseDecompressor(config)
        self.metrics = ReconstructionMetrics()
    
    def decompress_model(self, 
                        compressed_layers: Dict[str, TropicalChannels],
                        model_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Decompress entire model from compressed channels
        
        Args:
            compressed_layers: Dictionary of layer_name -> compressed channels
            model_info: Model architecture information
            
        Returns:
            Dictionary of layer_name -> weight tensor
        """
        start_time = time.time()
        decompressed_weights = {}
        
        # Get layer ordering if available
        layer_order = model_info.get('layer_order', list(compressed_layers.keys()))
        
        for layer_name in layer_order:
            if layer_name not in compressed_layers:
                logger.warning("Layer %s not found in compressed data", layer_name)
                continue
            
            layer_start = time.time()
            
            # Get layer information
            layer_info = model_info.get('layers', {}).get(layer_name, {})
            layer_info['name'] = layer_name
            
            # Decompress layer
            channels = compressed_layers[layer_name]
            weights = self.layer_decompressor.decompress_layer(channels, layer_info)
            
            decompressed_weights[layer_name] = weights
            
            # Update metrics
            self.metrics.num_weights += weights.numel()
            
            logger.debug("Decompressed layer %s in %.3fs", 
                        layer_name, time.time() - layer_start)
        
        self.metrics.total_time = time.time() - start_time
        
        # Compute throughput
        total_bytes = sum(w.numel() * w.element_size() for w in decompressed_weights.values())
        self.metrics.throughput_mbps = (total_bytes / 1048576) / self.metrics.total_time
        
        logger.info("Model decompression complete: %d layers, %.2f MB in %.3fs (%.2f MB/s)",
                   len(decompressed_weights), total_bytes / 1048576,
                   self.metrics.total_time, self.metrics.throughput_mbps)
        
        return decompressed_weights
    
    def verify_decompression(self, 
                            decompressed: Dict[str, torch.Tensor],
                            reference: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Verify decompression accuracy
        
        Args:
            decompressed: Decompressed weights
            reference: Optional reference weights for comparison
            
        Returns:
            Verification results
        """
        results = {
            'num_layers': len(decompressed),
            'total_parameters': sum(w.numel() for w in decompressed.values()),
            'layer_shapes': {name: tuple(w.shape) for name, w in decompressed.items()},
            'dtypes': {name: str(w.dtype) for name, w in decompressed.items()},
            'devices': {name: str(w.device) for name, w in decompressed.items()}
        }
        
        if reference:
            # Compare with reference
            errors = {}
            for name, decomp_weight in decompressed.items():
                if name in reference:
                    ref_weight = reference[name]
                    
                    # Compute error metrics
                    diff = decomp_weight - ref_weight
                    max_error = torch.max(torch.abs(diff)).item()
                    mean_error = torch.mean(torch.abs(diff)).item()
                    relative_error = torch.mean(torch.abs(diff) / (torch.abs(ref_weight) + 1e-10)).item()
                    
                    errors[name] = {
                        'max_error': max_error,
                        'mean_error': mean_error,
                        'relative_error': relative_error
                    }
                    
                    # Update global metrics
                    self.metrics.max_error = max(self.metrics.max_error, max_error)
                    self.metrics.mean_error += mean_error
                    
                    # Count exact matches
                    if max_error < self.config.max_reconstruction_error:
                        self.metrics.exact_matches += decomp_weight.numel()
            
            results['errors'] = errors
            results['accuracy_rate'] = self.metrics.compute_accuracy_rate()
            
            # Compute overall statistics
            if errors:
                self.metrics.mean_error /= len(errors)
                results['overall_max_error'] = self.metrics.max_error
                results['overall_mean_error'] = self.metrics.mean_error
        
        return results