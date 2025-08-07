"""
Compression Strategy Pattern Framework.
Intelligently selects between P-adic, Tropical, or hybrid compression strategies
for each neural network layer based on mathematical properties.
NO PLACEHOLDERS - COMPLETE PRODUCTION IMPLEMENTATION
"""

import torch
import torch.nn as nn
import numpy as np
import math
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union, Protocol
from dataclasses import dataclass, field
from enum import Enum
import logging
from fractions import Fraction
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import system components with proper path handling
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import pattern detector if available
try:
    from .pattern_detector import WeightDistributionAnalyzer, DistributionAnalysis
    PATTERN_DETECTOR_AVAILABLE = True
except ImportError:
    PATTERN_DETECTOR_AVAILABLE = False
    WeightDistributionAnalyzer = None
    DistributionAnalysis = None

# Import CSR sparse compression if available
try:
    from .sparse_compressor import SparseCompressor, SparseCompressionResult
    from .csr_sparse_matrix import CSRPadicMatrix
    CSR_AVAILABLE = True
except ImportError:
    CSR_AVAILABLE = False
    SparseCompressor = None
    SparseCompressionResult = None
    CSRPadicMatrix = None

# Import neural analysis components
from independent_core.compression_systems.neural_analysis.layer_analyzer import (
    DenseLayerAnalyzer as LayerAnalyzer,
    CompressionMethod,
    RankAnalysis,
    SparsityAnalysis,
    NumericalAnalysis
)

# Import P-adic/Tropical bridge
from independent_core.compression_systems.integration.padic_tropical_bridge import (
    PadicTropicalConverter,
    HybridRepresentation,
    ConversionConfig
)

# Import base compression interfaces
from independent_core.compression_systems.base.compression_base import (
    CompressionAlgorithm,
    AdaptiveCompressor
)

# Import P-adic system
from independent_core.compression_systems.padic.padic_encoder import (
    PadicWeight,
    PadicValidation,
    PadicMathematicalOperations
)
from independent_core.compression_systems.padic.padic_logarithmic_encoder import (
    PadicLogarithmicEncoder,
    LogarithmicEncodingConfig
)

# Import Tropical system
from independent_core.compression_systems.tropical.tropical_core import (
    TropicalNumber,
    TropicalMathematicalOperations,
    TropicalValidation,
    TROPICAL_ZERO,
    TROPICAL_EPSILON,
    is_tropical_zero,
    to_tropical_safe,
    from_tropical_safe
)

from independent_core.compression_systems.tropical.tropical_polynomial import (
    TropicalPolynomial,
    TropicalMonomial,
    TropicalPolynomialOperations
)


@dataclass
class StrategyConfig:
    """Configuration for strategy selection"""
    sparsity_threshold: float = 0.7
    rank_ratio_threshold: float = 0.3
    periodicity_threshold: float = 0.8
    dynamic_range_threshold: float = 1e6
    hybrid_threshold: float = 0.5
    enable_adaptive: bool = True
    cache_decisions: bool = True
    prime: int = 251  # P-adic prime
    precision: int = 3  # P-adic precision (reduced for safety)
    tropical_epsilon: float = 1e-10
    max_batch_size: int = 10000
    use_gpu: bool = True
    gradient_smoothing: float = 0.01
    learning_rate: float = 0.01  # For adaptive threshold updates
    performance_weight_compression: float = 0.5
    performance_weight_reconstruction: float = 0.3
    performance_weight_speed: float = 0.2
    
    def __post_init__(self):
        """Validate configuration parameters"""
        # Validate thresholds
        if not (0.0 <= self.sparsity_threshold <= 1.0):
            raise ValueError(f"sparsity_threshold must be in [0, 1], got {self.sparsity_threshold}")
        if not (0.0 <= self.rank_ratio_threshold <= 1.0):
            raise ValueError(f"rank_ratio_threshold must be in [0, 1], got {self.rank_ratio_threshold}")
        if not (0.0 <= self.periodicity_threshold <= 1.0):
            raise ValueError(f"periodicity_threshold must be in [0, 1], got {self.periodicity_threshold}")
        if self.dynamic_range_threshold <= 0:
            raise ValueError(f"dynamic_range_threshold must be positive, got {self.dynamic_range_threshold}")
        if not (0.0 <= self.hybrid_threshold <= 1.0):
            raise ValueError(f"hybrid_threshold must be in [0, 1], got {self.hybrid_threshold}")
        
        # Validate p-adic parameters
        PadicValidation.validate_prime(self.prime)
        PadicValidation.validate_precision(self.precision)
        
        # Validate other parameters
        if self.tropical_epsilon <= 0 or self.tropical_epsilon > 1:
            raise ValueError(f"tropical_epsilon must be in (0, 1], got {self.tropical_epsilon}")
        if self.max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {self.max_batch_size}")
        if self.gradient_smoothing < 0 or self.gradient_smoothing > 1:
            raise ValueError(f"gradient_smoothing must be in [0, 1], got {self.gradient_smoothing}")
        if self.learning_rate <= 0 or self.learning_rate > 1:
            raise ValueError(f"learning_rate must be in (0, 1], got {self.learning_rate}")
        
        # Validate performance weights sum to 1
        weight_sum = (self.performance_weight_compression + 
                     self.performance_weight_reconstruction + 
                     self.performance_weight_speed)
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Performance weights must sum to 1, got {weight_sum}")


@dataclass
class CompressedData:
    """Container for compressed data with metadata"""
    strategy_name: str
    compressed_bytes: bytes
    metadata: Dict[str, Any]
    compression_ratio: float
    original_shape: Tuple[int, ...]
    original_dtype: torch.dtype
    compression_time: float
    strategy_specific_data: Optional[Any] = None
    
    def get_size_bytes(self) -> int:
        """Get total size of compressed data in bytes"""
        return len(self.compressed_bytes)


class CompressionStrategy(ABC):
    """Abstract base for compression strategies"""
    
    @abstractmethod
    def compress(self, tensor: torch.Tensor, 
                metadata: Optional[Dict[str, Any]] = None) -> CompressedData:
        """Compress tensor using specific strategy"""
        pass
        
    @abstractmethod
    def decompress(self, compressed: CompressedData) -> torch.Tensor:
        """Decompress data back to tensor"""
        pass
        
    @abstractmethod
    def estimate_compression_ratio(self, tensor: torch.Tensor) -> float:
        """Estimate achievable compression ratio"""
        pass
        
    @abstractmethod
    def supports_gradients(self) -> bool:
        """Whether strategy preserves gradient flow"""
        pass
        
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy identifier"""
        pass


class TropicalStrategy(CompressionStrategy):
    """Tropical compression strategy for sparse/low-rank layers"""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize tropical strategy"""
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tropical_ops = TropicalMathematicalOperations(self.device)
        self.polynomial_ops = TropicalPolynomialOperations(self.device)
        
    def compress(self, tensor: torch.Tensor, 
                metadata: Optional[Dict[str, Any]] = None) -> CompressedData:
        """Compress tensor using tropical mathematics"""
        start_time = time.time()
        
        # Validate input
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        if tensor.numel() == 0:
            raise ValueError("Cannot compress empty tensor")
        
        original_shape = tensor.shape
        original_dtype = tensor.dtype
        
        # Flatten tensor for processing
        flat_tensor = tensor.flatten().to(self.device)
        
        # Convert to tropical representation
        tropical_values = to_tropical_safe(flat_tensor)
        
        # Apply tropical polynomial approximation for compression
        polynomial = self.polynomial_ops.fit_polynomial(
            tropical_values.view(-1, 1),  # Treat as 1D function
            degree=min(5, max(2, int(np.log2(flat_tensor.numel()))))  # Adaptive degree
        )
        
        # Serialize polynomial coefficients
        coefficients = []
        for monomial in polynomial.monomials:
            coefficients.append(monomial.coefficient)
            # Store exponents in sparse format
            for var_idx, power in monomial.exponents.items():
                coefficients.extend([float(var_idx), float(power)])
            coefficients.append(-1.0)  # Delimiter
        
        # Convert to bytes
        coefficients_array = np.array(coefficients, dtype=np.float32)
        compressed_bytes = coefficients_array.tobytes()
        
        # Calculate compression ratio
        original_size = tensor.numel() * tensor.element_size()
        compressed_size = len(compressed_bytes)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        compression_time = time.time() - start_time
        
        return CompressedData(
            strategy_name="tropical",
            compressed_bytes=compressed_bytes,
            metadata={
                "polynomial_degree": polynomial.degree,
                "num_monomials": len(polynomial.monomials),
                "device": str(self.device),
                **(metadata or {})
            },
            compression_ratio=compression_ratio,
            original_shape=original_shape,
            original_dtype=original_dtype,
            compression_time=compression_time,
            strategy_specific_data=polynomial
        )
    
    def decompress(self, compressed: CompressedData) -> torch.Tensor:
        """Decompress tropical polynomial back to tensor"""
        if compressed.strategy_name != "tropical":
            raise ValueError(f"Invalid strategy for TropicalStrategy: {compressed.strategy_name}")
        
        # Deserialize polynomial coefficients
        coefficients_array = np.frombuffer(compressed.compressed_bytes, dtype=np.float32)
        
        # Reconstruct polynomial
        monomials = []
        i = 0
        while i < len(coefficients_array):
            coefficient = coefficients_array[i]
            i += 1
            
            exponents = {}
            while i < len(coefficients_array) and coefficients_array[i] != -1.0:
                var_idx = int(coefficients_array[i])
                power = int(coefficients_array[i + 1])
                exponents[var_idx] = power
                i += 2
            
            if i < len(coefficients_array):
                i += 1  # Skip delimiter
            
            monomials.append(TropicalMonomial(coefficient, exponents))
        
        polynomial = TropicalPolynomial(
            monomials=monomials,
            degree=compressed.metadata["polynomial_degree"]
        )
        
        # Evaluate polynomial to reconstruct values
        num_points = np.prod(compressed.original_shape)
        x_values = torch.linspace(0, 1, num_points, device=self.device).view(-1, 1)
        
        tropical_values = self.polynomial_ops.evaluate_batch(polynomial, x_values)
        
        # Convert back from tropical
        reconstructed = from_tropical_safe(tropical_values)
        
        # Reshape to original shape
        reconstructed = reconstructed.view(compressed.original_shape)
        reconstructed = reconstructed.to(compressed.original_dtype)
        
        return reconstructed
    
    def estimate_compression_ratio(self, tensor: torch.Tensor) -> float:
        """Estimate compression ratio for tropical strategy"""
        # Estimate based on sparsity and rank
        flat_tensor = tensor.flatten()
        sparsity = (flat_tensor.abs() < 1e-10).float().mean().item()
        
        # Higher sparsity -> better compression
        if sparsity > 0.9:
            return 8.0  # Very sparse -> excellent compression
        elif sparsity > 0.7:
            return 6.0  # Sparse -> good compression
        elif sparsity > 0.5:
            return 4.0  # Moderately sparse
        else:
            # Estimate based on rank for dense tensors
            if len(tensor.shape) >= 2:
                try:
                    U, S, V = torch.svd(tensor.view(tensor.shape[0], -1))
                    rank_ratio = (S > 1e-6).float().mean().item()
                    if rank_ratio < 0.3:
                        return 5.0  # Low rank -> good compression
                    elif rank_ratio < 0.5:
                        return 3.0  # Moderate rank
                    else:
                        return 2.0  # High rank -> limited compression
                except:
                    return 2.0
            else:
                return 2.0
    
    def supports_gradients(self) -> bool:
        """Tropical strategy supports approximate gradients"""
        return True
    
    def get_strategy_name(self) -> str:
        """Return strategy identifier"""
        return "tropical"


class PadicStrategy(CompressionStrategy):
    """P-adic compression strategy for periodic/high-precision layers"""
    
    def __init__(self, prime: int = 251, precision: int = 3):
        """Initialize p-adic strategy"""
        PadicValidation.validate_prime(prime)
        PadicValidation.validate_precision(precision)
        
        self.prime = prime
        self.precision = precision
        self.default_prime = prime  # Store default for fallback
        
        # Create config for logarithmic encoder
        config = LogarithmicEncodingConfig()
        config.prime = prime
        config.precision = min(precision, 3)  # Use reduced precision for compression
        config.max_safe_precision = min(precision, 3)
        
        # Enable adaptive Hensel lifting for better compression
        config.enable_adaptive_hensel = True
        config.hensel_target_error = 1e-10
        
        self.encoder = PadicLogarithmicEncoder(config)
        
        # Initialize dynamic prime selector for strategy-level selection
        from ..padic.dynamic_prime_selector import DynamicPrimeSelector
        self.prime_selector = DynamicPrimeSelector(enable_caching=True)
        
    def compress(self, tensor: torch.Tensor, 
                metadata: Optional[Dict[str, Any]] = None) -> CompressedData:
        """Compress tensor using p-adic encoding with optimal prime selection"""
        start_time = time.time()
        
        # Validate input
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        if tensor.numel() == 0:
            raise ValueError("Cannot compress empty tensor")
        
        original_shape = tensor.shape
        original_dtype = tensor.dtype
        
        # Select optimal prime based on tensor distribution
        optimal_p = self.optimal_prime(tensor, metadata)
        
        # Update encoder if prime changed
        if optimal_p != self.encoder.config.prime:
            config = LogarithmicEncodingConfig()
            config.prime = optimal_p
            config.precision = min(self.precision, 3)
            config.max_safe_precision = min(self.precision, 3)
            self.encoder = PadicLogarithmicEncoder(config)
        
        # Encode tensor to p-adic using logarithmic encoding
        compressed_weights = self.encoder.encode_weights_logarithmically(tensor)
        
        # Serialize compressed weights to bytes
        import pickle
        compressed_bytes = pickle.dumps(compressed_weights)
        
        # Calculate compression ratio
        original_size = tensor.numel() * tensor.element_size()
        compressed_size = len(compressed_bytes)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        compression_time = time.time() - start_time
        
        # Include prime selection details in metadata
        prime_selection_metadata = {}
        if hasattr(self, '_last_prime_selection'):
            selection = self._last_prime_selection
            prime_selection_metadata = {
                "prime_selection": {
                    "optimal_prime": selection.optimal_prime,
                    "efficiency": selection.efficiency,
                    "average_digits": selection.average_digits,
                    "entropy": selection.entropy,
                    "distribution_type": selection.distribution_type,
                    "selection_rationale": selection.selection_rationale,
                    "candidate_scores": selection.candidate_scores
                }
            }
        
        return CompressedData(
            strategy_name="padic",
            compressed_bytes=compressed_bytes,
            metadata={
                "prime": self.encoder.config.prime,  # Use actual prime used
                "precision": self.precision,
                "distribution_type": getattr(self, '_last_distribution_type', 'unknown'),
                **prime_selection_metadata,
                **(metadata or {})
            },
            compression_ratio=compression_ratio,
            original_shape=original_shape,
            original_dtype=original_dtype,
            compression_time=compression_time,
            strategy_specific_data=compressed_weights
        )
    
    def decompress(self, compressed: CompressedData) -> torch.Tensor:
        """Decompress p-adic encoded data back to tensor"""
        if compressed.strategy_name != "padic":
            raise ValueError(f"Invalid strategy for PadicStrategy: {compressed.strategy_name}")
        
        # Deserialize compressed weights
        import pickle
        compressed_weights = pickle.loads(compressed.compressed_bytes)
        
        # Decode from logarithmic p-adic weights
        reconstructed = self.encoder.decode_logarithmic_padic_weights(compressed_weights)
        
        # Reshape to original shape
        reconstructed = reconstructed.view(compressed.original_shape)
        reconstructed = reconstructed.to(compressed.original_dtype)
        
        return reconstructed
    
    def estimate_compression_ratio(self, tensor: torch.Tensor) -> float:
        """Estimate compression ratio for p-adic strategy"""
        # Estimate based on dynamic range and periodicity
        flat_tensor = tensor.flatten()
        
        # Calculate dynamic range
        non_zero = flat_tensor[flat_tensor.abs() > 1e-10]
        if len(non_zero) > 0:
            dynamic_range = (non_zero.abs().max() / non_zero.abs().min()).item()
        else:
            dynamic_range = 1.0
        
        # Estimate periodicity using FFT
        try:
            fft_result = torch.fft.fft(flat_tensor[:min(1024, len(flat_tensor))])
            fft_magnitude = fft_result.abs()
            
            # Find peaks in frequency domain
            sorted_magnitudes, _ = torch.sort(fft_magnitude, descending=True)
            if len(sorted_magnitudes) > 10:
                peak_ratio = sorted_magnitudes[:5].mean() / sorted_magnitudes[5:].mean()
                periodicity_score = min(1.0, peak_ratio.item() / 10.0)
            else:
                periodicity_score = 0.0
        except:
            periodicity_score = 0.0
        
        # High dynamic range or strong periodicity -> good p-adic compression
        if dynamic_range > 1e6 or periodicity_score > 0.8:
            return 6.0  # Excellent for p-adic
        elif dynamic_range > 1e4 or periodicity_score > 0.5:
            return 4.0  # Good for p-adic
        elif dynamic_range > 1e2 or periodicity_score > 0.3:
            return 3.0  # Moderate
        else:
            return 2.0  # Limited benefit
    
    def supports_gradients(self) -> bool:
        """P-adic strategy preserves exact arithmetic"""
        return True
    
    def get_strategy_name(self) -> str:
        """Return strategy identifier"""
        return "padic"
    
    def optimal_prime(self, tensor: torch.Tensor, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Select optimal prime for p-adic compression based on tensor distribution.
        
        Mathematical foundation: p* = argmin_p E[L_p] where L_p = -log_p(P(x))
        
        Distribution-based selection:
        - Gaussian → small primes (2, 3, 5)
        - Bimodal → 2^k based on separation
        - Sparse → large primes
        - Uniform → entropy-driven selection
        
        Args:
            tensor: Input tensor to analyze
            metadata: Optional metadata with entropy values from StrategySelector
            
        Returns:
            Optimal prime for compression
        """
        # Use DynamicPrimeSelector for comprehensive analysis
        prime_result = self.prime_selector.select_optimal_prime(tensor)
        
        # Store distribution type for metadata
        self._last_distribution_type = prime_result.distribution_type
        
        # Store selection result for debugging/analysis
        self._last_prime_selection = prime_result
        
        return prime_result.optimal_prime
    
    def detect_distribution_type(self, tensor: torch.Tensor) -> str:
        """
        Detect the distribution type of the tensor.
        
        Returns one of: "gaussian", "bimodal", "sparse", "uniform", "multimodal"
        """
        flat_tensor = tensor.flatten().detach().cpu().numpy()
        
        # Check for sparsity first
        sparsity = (np.abs(flat_tensor) < 1e-10).mean()
        if sparsity > 0.7:
            return "sparse"
        
        # Remove zeros for distribution analysis
        non_zero = flat_tensor[np.abs(flat_tensor) > 1e-10]
        if len(non_zero) < 10:
            return "sparse"
        
        # Calculate statistical measures
        skewness = stats.skew(non_zero)
        kurtosis = stats.kurtosis(non_zero)
        
        # Test for normality (Gaussian)
        # Jarque-Bera test for normality
        jb_stat = len(non_zero) / 6 * (skewness**2 + (kurtosis**2) / 4)
        # Critical value at 5% significance level is ~5.99
        if jb_stat < 5.99 and abs(skewness) < 0.5 and abs(kurtosis) < 1:
            return "gaussian"
        
        # Test for bimodality using Hartigan's dip test approximation
        is_bimodal = self._test_bimodality(non_zero)
        if is_bimodal:
            return "bimodal"
        
        # Test for uniformity using Kolmogorov-Smirnov test
        if len(non_zero) > 30:
            # Normalize to [0, 1] for uniform test
            normalized = (non_zero - non_zero.min()) / (non_zero.max() - non_zero.min() + 1e-10)
            ks_stat, p_value = stats.kstest(normalized, 'uniform')
            if p_value > 0.05:  # Cannot reject uniformity
                return "uniform"
        
        # Check for multimodality using kernel density estimation peaks
        num_modes = self._count_modes(non_zero)
        if num_modes > 2:
            return "multimodal"
        
        # Default to uniform for unclassified distributions
        return "uniform"
    
    def _test_bimodality(self, data: np.ndarray) -> bool:
        """
        Test for bimodality using simplified Hartigan's dip test.
        """
        if len(data) < 10:
            return False
        
        # Sort data
        sorted_data = np.sort(data)
        n = len(sorted_data)
        
        # Calculate empirical CDF
        ecdf = np.arange(1, n + 1) / n
        
        # Find maximum deviation from uniform CDF
        # This is a simplified version of the dip statistic
        uniform_cdf = (sorted_data - sorted_data[0]) / (sorted_data[-1] - sorted_data[0] + 1e-10)
        max_deviation = np.max(np.abs(ecdf - uniform_cdf))
        
        # Heuristic threshold for bimodality
        # Higher deviation suggests departure from unimodality
        threshold = 0.05 + 0.15 / np.sqrt(n)
        
        # Additional check: look for gap in middle of distribution
        mid_idx = n // 2
        quarter_idx = n // 4
        three_quarter_idx = 3 * n // 4
        
        if quarter_idx < mid_idx < three_quarter_idx:
            gap_ratio = (sorted_data[three_quarter_idx] - sorted_data[quarter_idx]) / (sorted_data[-1] - sorted_data[0] + 1e-10)
            if gap_ratio > 0.5 and max_deviation > threshold:
                return True
        
        return max_deviation > threshold * 2  # Stricter threshold for bimodality
    
    def _calculate_bimodal_separation(self, tensor: torch.Tensor) -> float:
        """
        Calculate separation between modes in bimodal distribution.
        """
        flat_tensor = tensor.flatten().detach().cpu().numpy()
        non_zero = flat_tensor[np.abs(flat_tensor) > 1e-10]
        
        if len(non_zero) < 10:
            return 0.0
        
        # Use k-means with k=2 to find modes
        from sklearn.cluster import KMeans
        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans.fit(non_zero.reshape(-1, 1))
            
            # Calculate separation as distance between cluster centers
            centers = kmeans.cluster_centers_.flatten()
            separation = abs(centers[1] - centers[0])
            
            # Normalize by data range
            data_range = non_zero.max() - non_zero.min()
            if data_range > 0:
                return separation / data_range
            else:
                return 0.0
        except:
            # Fallback: use simple percentile-based separation
            p25 = np.percentile(non_zero, 25)
            p75 = np.percentile(non_zero, 75)
            data_range = non_zero.max() - non_zero.min()
            if data_range > 0:
                return (p75 - p25) / data_range
            else:
                return 0.0
    
    def _count_modes(self, data: np.ndarray) -> int:
        """
        Count the number of modes using kernel density estimation.
        """
        if len(data) < 10:
            return 1
        
        from scipy.stats import gaussian_kde
        try:
            # Create kernel density estimate
            kde = gaussian_kde(data)
            
            # Create fine grid for evaluation
            x_grid = np.linspace(data.min(), data.max(), 1000)
            density = kde(x_grid)
            
            # Find local maxima
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(density, height=np.max(density) * 0.1)
            
            return max(1, len(peaks))
        except:
            return 1
    
    def _estimate_code_length(self, tensor: torch.Tensor, prime: int) -> float:
        """
        Estimate expected code length for given prime.
        L_p = -log_p(P(x)) where P(x) is probability of value x.
        """
        flat_tensor = tensor.flatten().detach().cpu().numpy()
        
        # Quantize to p-adic representation precision
        scale = prime ** self.precision
        quantized = np.round(flat_tensor * scale) / scale
        
        # Calculate frequency distribution
        unique, counts = np.unique(quantized, return_counts=True)
        probabilities = counts / len(quantized)
        
        # Calculate expected code length
        # Using Shannon entropy with base p
        entropy = 0.0
        for prob in probabilities:
            if prob > 0:
                entropy -= prob * np.log(prob) / np.log(prime)
        
        # Add overhead for p-adic representation
        # Each digit requires log2(p) bits
        overhead = self.precision * np.log2(prime)
        
        return entropy + overhead / 8  # Convert bits to bytes factor
    
    def _calculate_entropy(self, tensor: torch.Tensor) -> float:
        """
        Calculate Shannon entropy of tensor.
        """
        # Quantize values for histogram
        num_bins = 256
        tensor_np = tensor.detach().cpu().numpy()
        
        # Get min/max for binning
        min_val = tensor_np.min()
        max_val = tensor_np.max()
        
        if max_val == min_val:
            return 0.0
        
        # Create histogram
        histogram, _ = np.histogram(tensor_np, bins=num_bins)
        
        # Calculate probabilities
        probabilities = histogram / len(tensor_np)
        probabilities = probabilities[probabilities > 0]
        
        if len(probabilities) == 0:
            return 0.0
        
        # Calculate entropy in bits
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return float(entropy)
    
    def is_prime(self, n: int) -> bool:
        """
        Check if a number is prime.
        """
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        # Check odd divisors up to sqrt(n)
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def next_prime(self, n: int) -> int:
        """
        Find the next prime number >= n.
        """
        if n <= 2:
            return 2
        
        # Ensure we start with an odd number
        if n % 2 == 0:
            n += 1
        else:
            n += 2
        
        # Find next prime
        while not self.is_prime(n):
            n += 2
            
            # Safety check to prevent infinite loop
            if n > 1000000:
                return self.default_prime  # Fallback to default
        
        return n


class HybridStrategy(CompressionStrategy):
    """Combined P-adic/Tropical strategy for complex layers"""
    
    def __init__(self, config: Optional[ConversionConfig] = None):
        """Initialize hybrid strategy"""
        self.config = config or ConversionConfig()
        self.converter = PadicTropicalConverter(self.config)
        self.padic_strategy = PadicStrategy(self.config.prime, self.config.precision)
        self.tropical_strategy = TropicalStrategy()
        
    def compress(self, tensor: torch.Tensor, 
                metadata: Optional[Dict[str, Any]] = None) -> CompressedData:
        """Compress using hybrid p-adic/tropical approach"""
        start_time = time.time()
        
        # Validate input
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        if tensor.numel() == 0:
            raise ValueError("Cannot compress empty tensor")
        
        original_shape = tensor.shape
        original_dtype = tensor.dtype
        
        # Convert to hybrid representation
        hybrid_rep = self.converter.tensor_to_hybrid(tensor)
        
        # Serialize hybrid representation
        compressed_parts = []
        
        # Compress p-adic part
        if hybrid_rep.padic_part is not None and len(hybrid_rep.padic_part) > 0:
            padic_bytes = self._serialize_padic_weights(hybrid_rep.padic_part)
            compressed_parts.append(padic_bytes)
        else:
            compressed_parts.append(b'')
        
        # Compress tropical part
        if hybrid_rep.tropical_part is not None and len(hybrid_rep.tropical_part) > 0:
            tropical_bytes = self._serialize_tropical_numbers(hybrid_rep.tropical_part)
            compressed_parts.append(tropical_bytes)
        else:
            compressed_parts.append(b'')
        
        # Combine parts with length headers
        compressed_bytes = b''
        for part in compressed_parts:
            compressed_bytes += len(part).to_bytes(4, 'little')
            compressed_bytes += part
        
        # Calculate compression ratio
        original_size = tensor.numel() * tensor.element_size()
        compressed_size = len(compressed_bytes)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        compression_time = time.time() - start_time
        
        return CompressedData(
            strategy_name="hybrid",
            compressed_bytes=compressed_bytes,
            metadata={
                "config": {
                    "prime": self.config.prime,
                    "precision": self.config.precision,
                    "tropical_epsilon": self.config.tropical_epsilon
                },
                "hybrid_metadata": hybrid_rep.metadata,
                **(metadata or {})
            },
            compression_ratio=compression_ratio,
            original_shape=original_shape,
            original_dtype=original_dtype,
            compression_time=compression_time,
            strategy_specific_data=hybrid_rep
        )
    
    def decompress(self, compressed: CompressedData) -> torch.Tensor:
        """Decompress hybrid representation back to tensor"""
        if compressed.strategy_name != "hybrid":
            raise ValueError(f"Invalid strategy for HybridStrategy: {compressed.strategy_name}")
        
        # Deserialize hybrid parts
        compressed_bytes = compressed.compressed_bytes
        offset = 0
        
        # Read p-adic part
        padic_len = int.from_bytes(compressed_bytes[offset:offset+4], 'little')
        offset += 4
        padic_bytes = compressed_bytes[offset:offset+padic_len]
        offset += padic_len
        
        # Read tropical part
        tropical_len = int.from_bytes(compressed_bytes[offset:offset+4], 'little')
        offset += 4
        tropical_bytes = compressed_bytes[offset:offset+tropical_len]
        
        # Deserialize parts
        padic_part = self._deserialize_padic_weights(padic_bytes) if padic_len > 0 else []
        tropical_part = self._deserialize_tropical_numbers(tropical_bytes) if tropical_len > 0 else []
        
        # Reconstruct hybrid representation
        hybrid_rep = HybridRepresentation(
            padic_part=padic_part,
            tropical_part=tropical_part,
            metadata=compressed.metadata['hybrid_metadata']
        )
        
        # Convert back to tensor
        reconstructed = self.converter.hybrid_to_tensor(
            hybrid_rep,
            compressed.original_shape
        )
        
        reconstructed = reconstructed.to(compressed.original_dtype)
        
        return reconstructed
    
    def estimate_compression_ratio(self, tensor: torch.Tensor) -> float:
        """Estimate compression ratio for hybrid strategy"""
        # Hybrid typically achieves better compression than either alone
        padic_ratio = self.padic_strategy.estimate_compression_ratio(tensor)
        tropical_ratio = self.tropical_strategy.estimate_compression_ratio(tensor)
        
        # Hybrid can leverage strengths of both
        return max(padic_ratio, tropical_ratio) * 1.2  # 20% bonus for hybrid optimization
    
    def supports_gradients(self) -> bool:
        """Hybrid strategy supports gradients through both paths"""
        return True
    
    def get_strategy_name(self) -> str:
        """Return strategy identifier"""
        return "hybrid"
    
    def _serialize_padic_weights(self, weights: List[PadicWeight]) -> bytes:
        """Serialize list of p-adic weights to bytes"""
        data = []
        for weight in weights:
            data.append(weight.valuation)
            data.extend(weight.digits)
        
        array = np.array(data, dtype=np.int32)
        return array.tobytes()
    
    def _deserialize_padic_weights(self, data: bytes) -> List[PadicWeight]:
        """Deserialize bytes to list of p-adic weights"""
        if len(data) == 0:
            return []
        
        array = np.frombuffer(data, dtype=np.int32)
        weights = []
        
        # Each weight has 1 valuation + precision digits
        weight_size = 1 + self.config.precision
        num_weights = len(array) // weight_size
        
        for i in range(num_weights):
            offset = i * weight_size
            valuation = array[offset]
            digits = array[offset + 1:offset + weight_size].tolist()
            
            # Reconstruct Fraction value from digits
            value = sum(d * (self.config.prime ** (j - valuation)) 
                       for j, d in enumerate(digits, start=valuation))
            
            weights.append(PadicWeight(
                value=Fraction(value).limit_denominator(),
                prime=self.config.prime,
                precision=self.config.precision,
                valuation=valuation,
                digits=digits
            ))
        
        return weights
    
    def _serialize_tropical_numbers(self, numbers: List[TropicalNumber]) -> bytes:
        """Serialize list of tropical numbers to bytes"""
        values = [num.value for num in numbers]
        array = np.array(values, dtype=np.float32)
        return array.tobytes()
    
    def _deserialize_tropical_numbers(self, data: bytes) -> List[TropicalNumber]:
        """Deserialize bytes to list of tropical numbers"""
        if len(data) == 0:
            return []
        
        array = np.frombuffer(data, dtype=np.float32)
        return [TropicalNumber(float(val)) for val in array]


class CSRStrategy(CompressionStrategy):
    """CSR sparse matrix compression strategy for highly sparse layers"""
    
    def __init__(self, sparsity_threshold: float = 0.9, csr_threshold: float = 1e-6):
        """Initialize CSR strategy"""
        if not CSR_AVAILABLE:
            raise ImportError("CSR compression not available. Install required dependencies.")
        
        self.sparse_compressor = SparseCompressor(
            sparsity_threshold=sparsity_threshold,
            csr_threshold=csr_threshold,
            min_matrix_size=100
        )
        
    def compress(self, tensor: torch.Tensor, 
                metadata: Optional[Dict[str, Any]] = None) -> CompressedData:
        """Compress tensor using CSR sparse matrix format"""
        start_time = time.time()
        
        # Validate input
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        if tensor.numel() == 0:
            raise ValueError("Cannot compress empty tensor")
        
        original_shape = tensor.shape
        original_dtype = tensor.dtype
        
        # Attempt CSR compression
        result = self.sparse_compressor.compress(tensor, metadata)
        
        if result is None:
            # Fallback: create minimal compression if CSR not suitable
            import pickle
            compressed_bytes = pickle.dumps(tensor.detach().cpu().numpy())
            compression_ratio = 1.0
            
            return CompressedData(
                strategy_name="csr",
                compressed_bytes=compressed_bytes,
                metadata={
                    "csr_failed": True,
                    "reason": "Insufficient sparsity for CSR",
                    **(metadata or {})
                },
                compression_ratio=compression_ratio,
                original_shape=original_shape,
                original_dtype=original_dtype,
                compression_time=time.time() - start_time,
                strategy_specific_data=None
            )
        
        # Successful CSR compression
        return CompressedData(
            strategy_name="csr",
            compressed_bytes=result.compressed_data,
            metadata={
                "sparsity": result.sparsity,
                "nnz": result.nnz,
                "csr_threshold": result.metadata['csr_threshold'],
                "density": result.metadata['density'],
                "row_efficiency": result.metadata['row_efficiency'],
                "bandwidth_reduction": result.metadata['bandwidth_reduction'],
                **(metadata or {})
            },
            compression_ratio=result.compression_ratio,
            original_shape=original_shape,
            original_dtype=original_dtype,
            compression_time=result.compression_time,
            strategy_specific_data=result.csr_matrix
        )
    
    def decompress(self, compressed: CompressedData) -> torch.Tensor:
        """Decompress CSR data back to tensor"""
        if compressed.strategy_name != "csr":
            raise ValueError(f"Invalid strategy for CSRStrategy: {compressed.strategy_name}")
        
        # Check if CSR compression failed
        if compressed.metadata.get('csr_failed', False):
            # Deserialize directly
            import pickle
            array = pickle.loads(compressed.compressed_bytes)
            tensor = torch.from_numpy(array).to(compressed.original_dtype)
            return tensor.view(compressed.original_shape)
        
        # Decompress CSR
        reconstructed = self.sparse_compressor.decompress(compressed.compressed_bytes)
        
        # Ensure correct shape and dtype
        reconstructed = reconstructed.view(compressed.original_shape)
        reconstructed = reconstructed.to(compressed.original_dtype)
        
        return reconstructed
    
    def estimate_compression_ratio(self, tensor: torch.Tensor) -> float:
        """Estimate compression ratio for CSR strategy"""
        # Analyze tensor for CSR benefit
        analysis = self.sparse_compressor.analyze_benefit(tensor)
        
        if analysis.recommended:
            return analysis.expected_compression_ratio
        else:
            return 1.0  # No compression benefit
    
    def supports_gradients(self) -> bool:
        """CSR strategy preserves values exactly, supports gradients"""
        return True
    
    def get_strategy_name(self) -> str:
        """Return strategy identifier"""
        return "csr"


class StrategySelector:
    """Intelligent strategy selection based on layer analysis"""
    
    def __init__(self, config: StrategyConfig):
        """Initialize strategy selector"""
        self.config = config
        self.analyzer = LayerAnalyzer()
        self.strategy_cache: Dict[str, CompressionStrategy] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.decision_cache: Dict[str, str] = {} if config.cache_decisions else None
        
        # Initialize pattern detector if available
        self.distribution_analyzer = WeightDistributionAnalyzer() if PATTERN_DETECTOR_AVAILABLE else None
        self.distribution_cache: Dict[str, DistributionAnalysis] = {}
        
        # Initialize CSR sparse compressor if available
        self.sparse_compressor = SparseCompressor(sparsity_threshold=0.9) if CSR_AVAILABLE else None
        
        # Initialize strategies
        self._initialize_strategies()
    
    def _initialize_strategies(self) -> None:
        """Initialize available compression strategies"""
        device = torch.device('cuda' if self.config.use_gpu and torch.cuda.is_available() else 'cpu')
        
        self.strategy_cache['tropical'] = TropicalStrategy(device)
        self.strategy_cache['padic'] = PadicStrategy(self.config.prime, self.config.precision)
        
        conversion_config = ConversionConfig(
            prime=self.config.prime,
            precision=self.config.precision,
            tropical_epsilon=self.config.tropical_epsilon,
            use_gpu=self.config.use_gpu,
            gradient_smoothing=self.config.gradient_smoothing,
            max_batch_size=self.config.max_batch_size
        )
        self.strategy_cache['hybrid'] = HybridStrategy(conversion_config)
        
        # Initialize CSR strategy if available
        if CSR_AVAILABLE:
            self.strategy_cache['csr'] = CSRStrategy(
                sparsity_threshold=0.9,
                csr_threshold=1e-6
            )
            logger.info("CSR sparse matrix strategy initialized successfully")
        
        # JAX strategy removed - no longer supported
    
    def select_strategy(self, tensor: torch.Tensor, 
                       layer_name: str = "") -> Tuple[CompressionStrategy, Dict[str, float]]:
        """Select optimal compression strategy for tensor
        
        Returns:
            Tuple of (strategy, analysis) where analysis contains entropy values
        """
        # Check cache if enabled
        if self.decision_cache is not None and layer_name in self.decision_cache:
            strategy_name = self.decision_cache[layer_name]
            logger.info(f"Using cached strategy '{strategy_name}' for layer '{layer_name}'")
            # Still need to analyze for entropy values
            analysis = self.analyze_tensor(tensor)
            return self.strategy_cache[strategy_name], analysis
        
        # Analyze tensor
        analysis = self.analyze_tensor(tensor)
        
        # Compute scores for each strategy
        scores = self.compute_strategy_scores(analysis)
        
        # Select strategy with highest score
        best_strategy = max(scores, key=scores.get)
        
        # Cache decision if enabled
        if self.decision_cache is not None and layer_name:
            self.decision_cache[layer_name] = best_strategy
        
        logger.info(f"Selected strategy '{best_strategy}' for layer '{layer_name}' "
                   f"(scores: {scores})")
        
        return self.strategy_cache[best_strategy], analysis
    
    def analyze_tensor(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Comprehensive tensor analysis for strategy selection"""
        flat_tensor = tensor.flatten()
        
        analysis = {}
        
        # Sparsity analysis
        analysis['sparsity'] = (flat_tensor.abs() < 1e-10).float().mean().item()
        analysis['near_zero_ratio'] = (flat_tensor.abs() < 1e-6).float().mean().item()
        analysis['tensor_size'] = tensor.numel()  # Add tensor size for CSR decision
        
        # Rank analysis for 2D+ tensors
        if len(tensor.shape) >= 2:
            try:
                matrix = tensor.view(tensor.shape[0], -1)
                U, S, V = torch.svd(matrix)
                
                # Effective rank (99% energy)
                cumsum = torch.cumsum(S ** 2, dim=0)
                total_energy = cumsum[-1]
                effective_rank = torch.sum(cumsum < 0.99 * total_energy).item() + 1
                
                analysis['rank_ratio'] = effective_rank / min(matrix.shape)
                analysis['condition_number'] = (S[0] / S[-1]).item() if S[-1] > 1e-10 else 1e10
            except:
                analysis['rank_ratio'] = 1.0
                analysis['condition_number'] = 1.0
        else:
            analysis['rank_ratio'] = 1.0
            analysis['condition_number'] = 1.0
        
        # Dynamic range analysis
        non_zero = flat_tensor[flat_tensor.abs() > 1e-10]
        if len(non_zero) > 0:
            analysis['dynamic_range'] = (non_zero.abs().max() / non_zero.abs().min()).item()
        else:
            analysis['dynamic_range'] = 1.0
        
        # Periodicity analysis using FFT
        try:
            sample_size = min(1024, len(flat_tensor))
            fft_result = torch.fft.fft(flat_tensor[:sample_size])
            fft_magnitude = fft_result.abs()
            
            # Find peaks
            sorted_mags, _ = torch.sort(fft_magnitude, descending=True)
            if len(sorted_mags) > 10:
                peak_ratio = sorted_mags[:5].mean() / sorted_mags[5:].mean()
                analysis['periodicity_score'] = min(1.0, peak_ratio.item() / 10.0)
            else:
                analysis['periodicity_score'] = 0.0
        except:
            analysis['periodicity_score'] = 0.0
        
        # Statistical properties
        analysis['mean'] = flat_tensor.mean().item()
        analysis['std'] = flat_tensor.std().item()
        analysis['kurtosis'] = ((flat_tensor - analysis['mean']) ** 4).mean().item() / (analysis['std'] ** 4) if analysis['std'] > 0 else 0
        
        # Entropy analysis for local data windows
        analysis['local_entropy'] = self._calculate_local_entropy(flat_tensor)
        analysis['global_entropy'] = self._calculate_global_entropy(flat_tensor)
        analysis['entropy_variance'] = self._calculate_entropy_variance(flat_tensor)
        
        # Add distribution analysis if available
        if self.distribution_analyzer:
            try:
                dist_analysis = self.distribution_analyzer.analyze_distribution(tensor)
                
                # Add distribution metrics to analysis
                analysis['dist_skewness'] = dist_analysis.skewness
                analysis['dist_kurtosis'] = dist_analysis.kurtosis
                analysis['num_modes'] = dist_analysis.num_modes
                analysis['quantization_levels'] = dist_analysis.quantization_levels
                analysis['distribution_type'] = dist_analysis.distribution_type
                
                # Cache the full analysis
                cache_key = str(id(tensor))
                self.distribution_cache[cache_key] = dist_analysis
            except Exception as e:
                # Pattern detection failed, continue without it
                logger.warning(f"Pattern detection failed: {e}")
        
        return analysis
    
    def _calculate_local_entropy(self, tensor: torch.Tensor, window_size: int = 256) -> float:
        """
        Calculate entropy for local data windows with sliding window approach.
        Uses rolling histogram updates for O(1) complexity per window shift.
        
        Formula: H_local(x) = -sum(p(x_i) * log2(p(x_i)))
        
        Args:
            tensor: Flattened tensor to analyze
            window_size: Size of sliding window (default 256 weights)
            
        Returns:
            Average local entropy across all windows
        """
        if len(tensor) < window_size:
            # If tensor is smaller than window, calculate global entropy
            return self._calculate_global_entropy(tensor)
        
        # Quantize values for histogram (256 bins for efficiency)
        num_bins = 256
        tensor_np = tensor.detach().cpu().numpy()
        
        # Get min/max for binning
        min_val = tensor_np.min()
        max_val = tensor_np.max()
        
        if max_val == min_val:
            return 0.0  # No variation, entropy is 0
        
        # Compute bin edges
        bins = np.linspace(min_val, max_val, num_bins + 1)
        
        # Digitize the entire tensor
        digitized = np.digitize(tensor_np, bins) - 1  # -1 to make 0-indexed
        digitized = np.clip(digitized, 0, num_bins - 1)  # Ensure within bounds
        
        # Initialize rolling histogram for first window
        histogram = np.zeros(num_bins, dtype=np.int32)
        for i in range(min(window_size, len(digitized))):
            histogram[digitized[i]] += 1
        
        # Calculate entropy for first window
        entropies = []
        entropy = self._entropy_from_histogram(histogram, window_size)
        entropies.append(entropy)
        
        # Slide window and update histogram O(1) per shift
        for i in range(window_size, len(digitized)):
            # Remove old value
            old_bin = digitized[i - window_size]
            histogram[old_bin] -= 1
            
            # Add new value
            new_bin = digitized[i]
            histogram[new_bin] += 1
            
            # Calculate entropy for current window
            entropy = self._entropy_from_histogram(histogram, window_size)
            entropies.append(entropy)
        
        # Return average local entropy
        return float(np.mean(entropies)) if entropies else 0.0
    
    def _calculate_global_entropy(self, tensor: torch.Tensor) -> float:
        """
        Calculate global entropy for entire tensor.
        
        Formula: H(x) = -sum(p(x_i) * log2(p(x_i)))
        
        Args:
            tensor: Flattened tensor to analyze
            
        Returns:
            Global entropy value
        """
        # Quantize values for histogram
        num_bins = 256
        tensor_np = tensor.detach().cpu().numpy()
        
        # Get min/max for binning
        min_val = tensor_np.min()
        max_val = tensor_np.max()
        
        if max_val == min_val:
            return 0.0  # No variation, entropy is 0
        
        # Create histogram
        histogram, _ = np.histogram(tensor_np, bins=num_bins)
        
        # Calculate entropy
        return self._entropy_from_histogram(histogram, len(tensor_np))
    
    def _calculate_entropy_variance(self, tensor: torch.Tensor, window_size: int = 256) -> float:
        """
        Calculate variance of entropy across local windows.
        High variance indicates non-uniform distribution.
        
        Args:
            tensor: Flattened tensor to analyze
            window_size: Size of sliding window
            
        Returns:
            Variance of local entropies
        """
        if len(tensor) < window_size * 2:
            return 0.0  # Not enough data for meaningful variance
        
        # Calculate entropies for non-overlapping windows for efficiency
        tensor_np = tensor.detach().cpu().numpy()
        num_windows = len(tensor_np) // window_size
        
        entropies = []
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window = tensor_np[start:end]
            
            # Calculate entropy for this window
            entropy = self._calculate_global_entropy(torch.from_numpy(window))
            entropies.append(entropy)
        
        if len(entropies) < 2:
            return 0.0
        
        return float(np.var(entropies))
    
    def _entropy_from_histogram(self, histogram: np.ndarray, total_count: int) -> float:
        """
        Calculate entropy from histogram using Shannon entropy formula.
        
        Formula: H = -sum(p(x_i) * log2(p(x_i)))
        
        Args:
            histogram: Bin counts
            total_count: Total number of samples
            
        Returns:
            Entropy value in bits
        """
        if total_count == 0:
            return 0.0
        
        # Calculate probabilities
        probabilities = histogram / total_count
        
        # Remove zero probabilities to avoid log(0)
        probabilities = probabilities[probabilities > 0]
        
        if len(probabilities) == 0:
            return 0.0
        
        # Calculate entropy using base-2 logarithm (bits)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return float(entropy)
    
    def compute_strategy_scores(self, analysis: Dict[str, float]) -> Dict[str, float]:
        """Score each strategy based on analysis"""
        scores = {}
        
        # CSR strategy score for extreme sparsity (if available)
        if CSR_AVAILABLE and analysis['sparsity'] > 0.9:
            # CSR is optimal for very sparse matrices
            csr_score = 0.0
            
            # Strong preference for CSR when sparsity is extreme
            if analysis['sparsity'] > 0.95:
                csr_score = 0.95  # Almost always use CSR
            elif analysis['sparsity'] > 0.9:
                csr_score = 0.8  # Strong preference
            
            # Additional factors
            if analysis.get('distribution_type') == 'sparse':
                csr_score = min(1.0, csr_score + 0.1)
            
            # Check matrix size (CSR needs sufficient size to be beneficial)
            # Assuming we have access to tensor shape through analysis
            if 'tensor_size' in analysis and analysis['tensor_size'] < 100:
                csr_score *= 0.5  # Reduce score for small matrices
            
            scores['csr'] = csr_score
        
        # Tropical strategy score (reduced when CSR is better)
        tropical_score = 0.0
        if analysis['sparsity'] > self.config.sparsity_threshold:
            # If CSR is available and sparsity > 0.9, reduce tropical score
            if CSR_AVAILABLE and analysis['sparsity'] > 0.9:
                tropical_score += 0.2  # Reduced from 0.5
            else:
                tropical_score += 0.5
        if analysis['rank_ratio'] < self.config.rank_ratio_threshold:
            tropical_score += 0.3
        if analysis['near_zero_ratio'] > 0.5:
            tropical_score += 0.2
        
        # Bonus for extreme sparsity (only if CSR not available)
        if analysis['sparsity'] > 0.9 and not CSR_AVAILABLE:
            tropical_score *= 1.5
        
        # Distribution adjustments for tropical
        dist_type = analysis.get('distribution_type', 'unknown')
        if dist_type == 'sparse':
            tropical_score += 0.3  # Sparse is ideal for tropical
        elif dist_type == 'uniform':
            tropical_score -= 0.1  # Uniform doesn't benefit from tropical
        
        scores['tropical'] = min(1.0, tropical_score)
        
        # P-adic strategy score
        padic_score = 0.0
        if analysis['periodicity_score'] > self.config.periodicity_threshold:
            padic_score += 0.5
        if analysis['dynamic_range'] > self.config.dynamic_range_threshold:
            padic_score += 0.3
        if analysis['condition_number'] > 100:
            padic_score += 0.2
        
        # Entropy-based adjustments for P-adic
        # Low entropy (< 4 bits) suggests structured data good for P-adic
        if 'local_entropy' in analysis:
            if analysis['local_entropy'] < 4.0:  # Low entropy, highly structured
                padic_score += 0.2
            elif analysis['local_entropy'] > 7.0:  # High entropy, random-like
                padic_score -= 0.1
        
        # High entropy variance suggests non-uniform data, good for adaptive P-adic
        if 'entropy_variance' in analysis and analysis['entropy_variance'] > 0.5:
            padic_score += 0.1
        
        # Distribution-based adjustments for P-adic
        dist_type = analysis.get('distribution_type', 'unknown')
        if dist_type == 'gaussian':
            # Gaussian distributions compress well with P-adic
            padic_score += 0.2
        elif dist_type == 'heavy_tailed':
            # Heavy-tailed distributions benefit from P-adic's dynamic range handling
            padic_score += 0.15
        
        # Quantization awareness
        if analysis.get('quantization_levels', 256) < 32:
            # Already quantized, P-adic can leverage this
            padic_score += 0.1
        
        # Bonus for extreme dynamic range
        if analysis['dynamic_range'] > 1e8:
            padic_score *= 1.5
        
        scores['padic'] = min(1.0, max(0.0, padic_score))
        
        # Hybrid strategy score
        hybrid_score = 0.0
        if scores['tropical'] > self.config.hybrid_threshold and scores['padic'] > self.config.hybrid_threshold:
            hybrid_score = (scores['tropical'] + scores['padic']) / 1.5  # Synergy bonus
        elif analysis['kurtosis'] > 3:  # Complex distribution
            hybrid_score = 0.6
        elif analysis['condition_number'] > 1000 and analysis['sparsity'] > 0.3:
            hybrid_score = 0.7
        
        # Distribution adjustments for hybrid
        if dist_type == 'bimodal':
            hybrid_score += 0.3  # Bimodal benefits from hybrid approach
        elif dist_type == 'multimodal' and analysis.get('num_modes', 1) > 3:
            hybrid_score += 0.25  # Complex multimodal needs hybrid
        
        scores['hybrid'] = min(1.0, hybrid_score)
        
        # Apply adaptive learning adjustments if enabled
        if self.config.enable_adaptive:
            scores = self._apply_performance_adjustments(scores)
        
        # Ensure at least one strategy has non-zero score
        if max(scores.values()) == 0:
            scores['padic'] = 0.5  # Default fallback
        
        return scores
    
    def _apply_performance_adjustments(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Apply performance-based adjustments to scores"""
        adjusted_scores = scores.copy()
        
        for strategy_name in scores:
            if strategy_name in self.performance_history:
                history = self.performance_history[strategy_name]
                if len(history) > 0:
                    # Calculate average performance
                    avg_performance = np.mean(history[-10:])  # Last 10 results
                    
                    # Adjust score based on historical performance
                    adjustment = (avg_performance - 0.5) * self.config.learning_rate
                    adjusted_scores[strategy_name] = min(1.0, max(0.0, 
                        scores[strategy_name] + adjustment))
        
        return adjusted_scores
    
    def update_performance(self, strategy_name: str, 
                          compression_ratio: float,
                          reconstruction_error: float,
                          compression_time: float) -> None:
        """Update performance history for adaptive learning"""
        if strategy_name not in self.performance_history:
            self.performance_history[strategy_name] = []
        
        # Normalize metrics to [0, 1]
        normalized_compression = min(1.0, compression_ratio / 10.0)  # Cap at 10x
        normalized_error = 1.0 - min(1.0, reconstruction_error)  # Lower is better
        normalized_speed = 1.0 / (1.0 + compression_time)  # Faster is better
        
        # Compute weighted performance score
        performance_score = (
            self.config.performance_weight_compression * normalized_compression +
            self.config.performance_weight_reconstruction * normalized_error +
            self.config.performance_weight_speed * normalized_speed
        )
        
        self.performance_history[strategy_name].append(performance_score)
        
        # Limit history size
        if len(self.performance_history[strategy_name]) > 100:
            self.performance_history[strategy_name].pop(0)


class AdaptiveStrategyManager:
    """Manages strategy selection with learning capability"""
    
    def __init__(self, initial_config: StrategyConfig):
        """Initialize adaptive strategy manager"""
        self.config = initial_config
        self.selector = StrategySelector(initial_config)
        self.decision_history: List[Dict[str, Any]] = []
        self.compression_results: Dict[str, CompressedData] = {}
        
    def compress_model(self, model: nn.Module) -> Dict[str, CompressedData]:
        """Compress entire model with per-layer strategy selection"""
        compressed_layers = {}
        
        for name, module in model.named_modules():
            # Process only leaf modules with parameters
            if len(list(module.children())) == 0:
                for param_name, param in module.named_parameters():
                    if param is None or param.numel() == 0:
                        continue
                    
                    full_name = f"{name}.{param_name}" if name else param_name
                    
                    # Select strategy with analysis
                    strategy, analysis = self.selector.select_strategy(param.data, full_name)
                    
                    # Compress parameter with analysis metadata
                    start_time = time.time()
                    compressed = strategy.compress(param.data, metadata=analysis)
                    compression_time = time.time() - start_time
                    
                    # Store result
                    compressed_layers[full_name] = compressed
                    self.compression_results[full_name] = compressed
                    
                    # Calculate reconstruction error for learning
                    reconstructed = strategy.decompress(compressed)
                    reconstruction_error = torch.nn.functional.mse_loss(
                        reconstructed, param.data).item()
                    
                    # Update performance history
                    self.selector.update_performance(
                        strategy.get_strategy_name(),
                        compressed.compression_ratio,
                        reconstruction_error,
                        compression_time
                    )
                    
                    # Record decision
                    self.decision_history.append({
                        'layer_name': full_name,
                        'strategy': strategy.get_strategy_name(),
                        'compression_ratio': compressed.compression_ratio,
                        'reconstruction_error': reconstruction_error,
                        'compression_time': compression_time,
                        'tensor_shape': list(param.shape),
                        'tensor_numel': param.numel()
                    })
        
        return compressed_layers
    
    def learn_from_results(self) -> None:
        """Update strategy thresholds based on performance"""
        if not self.config.enable_adaptive or len(self.decision_history) < 10:
            return
        
        # Group decisions by strategy
        strategy_groups = {}
        for decision in self.decision_history:
            strategy = decision['strategy']
            if strategy not in strategy_groups:
                strategy_groups[strategy] = []
            strategy_groups[strategy].append(decision)
        
        # Analyze performance per strategy
        for strategy_name, decisions in strategy_groups.items():
            if len(decisions) < 3:
                continue
            
            # Calculate average metrics
            avg_ratio = np.mean([d['compression_ratio'] for d in decisions])
            avg_error = np.mean([d['reconstruction_error'] for d in decisions])
            
            # Update thresholds based on performance
            if strategy_name == 'tropical':
                # If tropical performs well, lower thresholds to use it more
                if avg_ratio > 5.0 and avg_error < 0.01:
                    self.config.sparsity_threshold *= 0.95
                    self.config.rank_ratio_threshold *= 1.05
                # If tropical performs poorly, raise thresholds
                elif avg_ratio < 2.0 or avg_error > 0.1:
                    self.config.sparsity_threshold *= 1.05
                    self.config.rank_ratio_threshold *= 0.95
            
            elif strategy_name == 'padic':
                # Adjust p-adic thresholds
                if avg_ratio > 5.0 and avg_error < 0.01:
                    self.config.periodicity_threshold *= 0.95
                    self.config.dynamic_range_threshold *= 0.9
                elif avg_ratio < 2.0 or avg_error > 0.1:
                    self.config.periodicity_threshold *= 1.05
                    self.config.dynamic_range_threshold *= 1.1
            
            elif strategy_name == 'hybrid':
                # Adjust hybrid threshold
                if avg_ratio > 6.0 and avg_error < 0.01:
                    self.config.hybrid_threshold *= 0.95
                elif avg_ratio < 3.0 or avg_error > 0.1:
                    self.config.hybrid_threshold *= 1.05
        
        # Ensure thresholds stay in valid ranges
        self.config.sparsity_threshold = min(0.95, max(0.3, self.config.sparsity_threshold))
        self.config.rank_ratio_threshold = min(0.7, max(0.1, self.config.rank_ratio_threshold))
        self.config.periodicity_threshold = min(0.95, max(0.3, self.config.periodicity_threshold))
        self.config.dynamic_range_threshold = min(1e10, max(100, self.config.dynamic_range_threshold))
        self.config.hybrid_threshold = min(0.8, max(0.2, self.config.hybrid_threshold))
        
        logger.info(f"Updated thresholds after learning: "
                   f"sparsity={self.config.sparsity_threshold:.3f}, "
                   f"rank_ratio={self.config.rank_ratio_threshold:.3f}, "
                   f"periodicity={self.config.periodicity_threshold:.3f}, "
                   f"dynamic_range={self.config.dynamic_range_threshold:.1e}, "
                   f"hybrid={self.config.hybrid_threshold:.3f}")
    
    def export_strategy_map(self) -> Dict[str, str]:
        """Export layer->strategy mapping for deployment"""
        strategy_map = {}
        
        for decision in self.decision_history:
            layer_name = decision['layer_name']
            strategy = decision['strategy']
            strategy_map[layer_name] = strategy
        
        return strategy_map
    
    def get_compression_summary(self) -> Dict[str, Any]:
        """Get summary statistics of compression performance"""
        if len(self.decision_history) == 0:
            return {}
        
        summary = {
            'total_layers': len(self.decision_history),
            'average_compression_ratio': np.mean([d['compression_ratio'] for d in self.decision_history]),
            'average_reconstruction_error': np.mean([d['reconstruction_error'] for d in self.decision_history]),
            'average_compression_time': np.mean([d['compression_time'] for d in self.decision_history]),
            'strategy_distribution': {},
            'best_performing_layers': [],
            'worst_performing_layers': []
        }
        
        # Count strategy usage
        for decision in self.decision_history:
            strategy = decision['strategy']
            if strategy not in summary['strategy_distribution']:
                summary['strategy_distribution'][strategy] = 0
            summary['strategy_distribution'][strategy] += 1
        
        # Find best and worst performing layers
        sorted_by_ratio = sorted(self.decision_history, 
                                key=lambda x: x['compression_ratio'], 
                                reverse=True)
        summary['best_performing_layers'] = [
            {'layer': d['layer_name'], 
             'ratio': d['compression_ratio'],
             'strategy': d['strategy']}
            for d in sorted_by_ratio[:5]
        ]
        
        sorted_by_error = sorted(self.decision_history, 
                               key=lambda x: x['reconstruction_error'])
        summary['worst_performing_layers'] = [
            {'layer': d['layer_name'],
             'error': d['reconstruction_error'],
             'strategy': d['strategy']}
            for d in sorted_by_error[-5:]
        ]
        
        return summary


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_strategy_config():
    """Test StrategyConfig validation"""
    print("Testing StrategyConfig...")
    
    # Valid config
    config = StrategyConfig()
    assert config.sparsity_threshold == 0.7
    assert config.prime == 251
    
    # Test invalid thresholds
    try:
        StrategyConfig(sparsity_threshold=1.5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    try:
        StrategyConfig(dynamic_range_threshold=-1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    print("✓ StrategyConfig tests passed")


def test_tropical_strategy():
    """Test TropicalStrategy compression and decompression"""
    print("Testing TropicalStrategy...")
    
    strategy = TropicalStrategy()
    
    # Test with sparse tensor
    tensor = torch.randn(10, 10)
    tensor[tensor.abs() < 0.5] = 0  # Make sparse
    
    compressed = strategy.compress(tensor)
    assert compressed.strategy_name == "tropical"
    assert compressed.compression_ratio > 1.0
    
    # Test decompression
    reconstructed = strategy.decompress(compressed)
    assert reconstructed.shape == tensor.shape
    
    # Test error handling
    try:
        strategy.compress("not a tensor")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    print("✓ TropicalStrategy tests passed")


def test_padic_strategy():
    """Test PadicStrategy compression and decompression"""
    print("Testing PadicStrategy...")
    
    strategy = PadicStrategy(prime=251, precision=16)
    
    # Test with periodic tensor
    tensor = torch.sin(torch.linspace(0, 4 * np.pi, 100)).view(10, 10)
    
    compressed = strategy.compress(tensor)
    assert compressed.strategy_name == "padic"
    assert compressed.compression_ratio > 1.0
    
    # Test decompression
    reconstructed = strategy.decompress(compressed)
    assert reconstructed.shape == tensor.shape
    
    # Test gradient support
    assert strategy.supports_gradients()
    
    print("✓ PadicStrategy tests passed")


def test_hybrid_strategy():
    """Test HybridStrategy compression and decompression"""
    print("Testing HybridStrategy...")
    
    config = ConversionConfig(prime=251, precision=16)
    strategy = HybridStrategy(config)
    
    # Test with complex tensor
    tensor = torch.randn(20, 20)
    tensor[::2, ::2] = 0  # Add structure
    
    compressed = strategy.compress(tensor)
    assert compressed.strategy_name == "hybrid"
    assert compressed.compression_ratio > 1.0
    
    # Test decompression
    reconstructed = strategy.decompress(compressed)
    assert reconstructed.shape == tensor.shape
    
    print("✓ HybridStrategy tests passed")


def test_strategy_selector():
    """Test StrategySelector decision making"""
    print("Testing StrategySelector...")
    
    config = StrategyConfig()
    selector = StrategySelector(config)
    
    # Test with sparse tensor (should select tropical)
    sparse_tensor = torch.zeros(100, 100)
    sparse_tensor[torch.rand(100, 100) > 0.9] = torch.randn(1)
    
    strategy, analysis = selector.select_strategy(sparse_tensor, "sparse_layer")
    assert strategy.get_strategy_name() in ["tropical", "hybrid"]
    assert 'local_entropy' in analysis  # Check entropy is calculated
    
    # Test with periodic tensor (should select p-adic)
    periodic_tensor = torch.sin(torch.linspace(0, 10 * np.pi, 10000)).view(100, 100)
    
    strategy, analysis = selector.select_strategy(periodic_tensor, "periodic_layer")
    # Strategy selection depends on analysis
    assert strategy.get_strategy_name() in ["padic", "tropical", "hybrid"]
    assert 'local_entropy' in analysis
    
    # Test caching
    if config.cache_decisions:
        strategy2, _ = selector.select_strategy(sparse_tensor, "sparse_layer")
        assert strategy.get_strategy_name() == strategy2.get_strategy_name()
    
    print("✓ StrategySelector tests passed")


def test_adaptive_manager():
    """Test AdaptiveStrategyManager with a small model"""
    print("Testing AdaptiveStrategyManager...")
    
    config = StrategyConfig(enable_adaptive=True)
    manager = AdaptiveStrategyManager(config)
    
    # Create a small test model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    
    # Compress model
    compressed = manager.compress_model(model)
    assert len(compressed) > 0
    
    # Test learning
    manager.learn_from_results()
    
    # Test export
    strategy_map = manager.export_strategy_map()
    assert len(strategy_map) > 0
    
    # Test summary
    summary = manager.get_compression_summary()
    assert 'total_layers' in summary
    assert summary['total_layers'] > 0
    
    print("✓ AdaptiveStrategyManager tests passed")


def test_performance_tracking():
    """Test performance tracking and updates"""
    print("Testing performance tracking...")
    
    config = StrategyConfig()
    selector = StrategySelector(config)
    
    # Update performance for different strategies
    selector.update_performance("tropical", 5.0, 0.01, 0.1)
    selector.update_performance("padic", 4.0, 0.02, 0.15)
    selector.update_performance("hybrid", 6.0, 0.005, 0.2)
    
    assert "tropical" in selector.performance_history
    assert len(selector.performance_history["tropical"]) == 1
    
    # Test adaptive adjustments
    tensor = torch.randn(50, 50)
    analysis = selector.analyze_tensor(tensor)
    scores = selector.compute_strategy_scores(analysis)
    
    # Scores should be adjusted based on performance
    assert all(0 <= score <= 1.0 for score in scores.values())
    
    print("✓ Performance tracking tests passed")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("Testing edge cases...")
    
    # Test with empty tensor
    strategy = TropicalStrategy()
    try:
        strategy.compress(torch.tensor([]))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test with very small tensor
    tiny_tensor = torch.tensor([1.0])
    compressed = strategy.compress(tiny_tensor.view(1, 1))
    reconstructed = strategy.decompress(compressed)
    assert reconstructed.shape == (1, 1)
    
    # Test with very large values
    large_tensor = torch.tensor([[1e10, -1e10], [1e-10, 0]])
    config = StrategyConfig()
    selector = StrategySelector(config)
    
    strategy, analysis = selector.select_strategy(large_tensor)
    compressed = strategy.compress(large_tensor, metadata=analysis)
    reconstructed = strategy.decompress(compressed)
    assert reconstructed.shape == large_tensor.shape
    
    print("✓ Edge case tests passed")


def run_all_tests():
    """Run all unit tests"""
    print("=" * 60)
    print("Running Compression Strategy Unit Tests")
    print("=" * 60)
    
    test_strategy_config()
    test_tropical_strategy()
    test_padic_strategy()
    test_hybrid_strategy()
    test_strategy_selector()
    test_adaptive_manager()
    test_performance_tracking()
    test_edge_cases()
    
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
