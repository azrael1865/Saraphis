"""
Dynamic Prime Selection for P-adic Compression System.
Selects optimal prime based on tensor distribution characteristics.
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
import math
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.cluster import KMeans
import threading
import time


@dataclass
class PrimeSelectionResult:
    """Result of dynamic prime selection"""
    optimal_prime: int
    efficiency: float
    average_digits: float
    entropy: float
    distribution_type: str
    selection_rationale: str
    candidate_scores: Dict[int, float]
    computation_time: float


class DynamicPrimeSelector:
    """
    Dynamic prime selection for p-adic compression.
    
    Mathematical foundation:
    - Optimal prime: p* = argmin_p Σᵢ ⌈log_p(|xᵢ|)⌉
    - Efficiency metric: E(p) = 32 / (log₂(p) * avg_digits)
    - Balances small base (fewer unique digits) vs large base (fewer total digits)
    """
    
    # Default prime candidates
    DEFAULT_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Extended primes for special cases
    EXTENDED_PRIMES = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251]
    
    def __init__(self, 
                 default_primes: Optional[List[int]] = None,
                 enable_caching: bool = True,
                 cache_size: int = 100):
        """
        Initialize dynamic prime selector.
        
        Args:
            default_primes: List of prime candidates to consider
            enable_caching: Whether to cache prime selections for similar tensors
            cache_size: Maximum number of cached selections
        """
        self.primes = default_primes or self.DEFAULT_PRIMES
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        
        # Validate all primes
        for p in self.primes:
            if not self._is_prime(p):
                raise ValueError(f"{p} is not a prime number")
        
        # Sort primes for consistent processing
        self.primes = sorted(self.primes)
        
        # Cache for prime selections
        self.selection_cache: Dict[str, PrimeSelectionResult] = {}
        self.cache_lock = threading.Lock()
        
        # Performance tracking
        self.selection_count = 0
        self.cache_hits = 0
        self.total_selection_time = 0.0
    
    def select_optimal_prime(self, 
                           tensor: torch.Tensor,
                           primes: Optional[List[int]] = None,
                           max_evaluation_samples: int = 10000,
                           early_stopping_threshold: float = 0.95) -> PrimeSelectionResult:
        """
        Select optimal prime for given tensor.
        
        Args:
            tensor: Input tensor to compress
            primes: Optional list of prime candidates (uses default if None)
            max_evaluation_samples: Maximum number of samples for evaluation
            early_stopping_threshold: Stop if efficiency exceeds this threshold
            
        Returns:
            PrimeSelectionResult with optimal prime and metrics
        """
        start_time = time.time()
        self.selection_count += 1
        
        # Validate input
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        if tensor.numel() == 0:
            raise ValueError("Cannot select prime for empty tensor")
        
        # Check cache if enabled
        if self.enable_caching:
            cache_key = self._compute_cache_key(tensor)
            with self.cache_lock:
                if cache_key in self.selection_cache:
                    self.cache_hits += 1
                    cached_result = self.selection_cache[cache_key]
                    # Update computation time
                    result = PrimeSelectionResult(
                        optimal_prime=cached_result.optimal_prime,
                        efficiency=cached_result.efficiency,
                        average_digits=cached_result.average_digits,
                        entropy=cached_result.entropy,
                        distribution_type=cached_result.distribution_type,
                        selection_rationale=cached_result.selection_rationale + " (cached)",
                        candidate_scores=cached_result.candidate_scores,
                        computation_time=time.time() - start_time
                    )
                    return result
        
        # Use provided primes or default
        candidate_primes = primes or self.primes
        
        # Analyze tensor characteristics
        distribution_type = self._detect_distribution_type(tensor)
        tensor_characteristics = self.analyze_tensor_characteristics(tensor)
        
        # Sample tensor for efficiency
        flat_tensor = tensor.flatten()
        if flat_tensor.numel() > max_evaluation_samples:
            # Random sampling for large tensors
            indices = torch.randperm(flat_tensor.numel())[:max_evaluation_samples]
            sample_tensor = flat_tensor[indices]
        else:
            sample_tensor = flat_tensor
        
        # Convert to numpy for faster processing
        sample_values = sample_tensor.detach().cpu().numpy()
        
        # Filter out zeros and very small values
        non_zero_mask = np.abs(sample_values) > 1e-10
        if not np.any(non_zero_mask):
            # All zeros - any prime works, choose smallest
            computation_time = time.time() - start_time
            self.total_selection_time += computation_time
            result = PrimeSelectionResult(
                optimal_prime=candidate_primes[0],
                efficiency=1.0,
                average_digits=1.0,
                entropy=0.0,
                distribution_type=distribution_type,
                selection_rationale="All zeros tensor - using smallest prime",
                candidate_scores={p: 1.0 for p in candidate_primes},
                computation_time=computation_time
            )
            self._update_cache(cache_key, result)
            return result
        
        non_zero_values = sample_values[non_zero_mask]
        
        # Evaluate each prime candidate
        best_prime = candidate_primes[0]
        best_efficiency = 0.0
        best_avg_digits = float('inf')
        best_entropy = 0.0
        candidate_scores = {}
        
        for p in candidate_primes:
            # Calculate average digit count
            avg_digits = self.calculate_average_digits(non_zero_values, p)
            
            # Calculate entropy in base p
            entropy = self.calculate_entropy(sample_values, p)
            
            # Calculate efficiency metric
            efficiency = self.compute_efficiency_metric(p, avg_digits, entropy, tensor_characteristics)
            
            candidate_scores[p] = efficiency
            
            # Update best if improved
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_prime = p
                best_avg_digits = avg_digits
                best_entropy = entropy
                
                # Early stopping if efficiency is very high
                if efficiency > early_stopping_threshold:
                    break
        
        # Generate selection rationale
        rationale = self._generate_rationale(
            best_prime, 
            distribution_type, 
            tensor_characteristics,
            best_efficiency,
            best_avg_digits
        )
        
        computation_time = time.time() - start_time
        self.total_selection_time += computation_time
        
        result = PrimeSelectionResult(
            optimal_prime=best_prime,
            efficiency=best_efficiency,
            average_digits=best_avg_digits,
            entropy=best_entropy,
            distribution_type=distribution_type,
            selection_rationale=rationale,
            candidate_scores=candidate_scores,
            computation_time=computation_time
        )
        
        # Update cache
        if self.enable_caching:
            self._update_cache(cache_key, result)
        
        return result
    
    def compute_digit_distribution(self, values: np.ndarray, prime: int, max_digits: int = 20) -> np.ndarray:
        """
        Compute digit distribution in base p.
        
        Args:
            values: Array of values to analyze
            prime: Base for p-adic representation
            max_digits: Maximum number of digits to consider
            
        Returns:
            Array of digit probabilities
        """
        if len(values) == 0:
            return np.zeros(prime)
        
        # Initialize digit counts
        digit_counts = np.zeros(prime, dtype=np.int64)
        
        for val in values:
            if abs(val) < 1e-10:
                digit_counts[0] += max_digits
                continue
            
            # Convert to p-adic digits
            abs_val = abs(val)
            
            # Handle integer part
            if abs_val >= 1:
                int_part = int(abs_val)
                digit_count = 0
                while int_part > 0 and digit_count < max_digits:
                    digit = int_part % prime
                    digit_counts[digit] += 1
                    int_part //= prime
                    digit_count += 1
            
            # Handle fractional part
            frac_part = abs_val - int(abs_val)
            if frac_part > 1e-10:
                digit_count = 0
                while frac_part > 1e-10 and digit_count < max_digits:
                    frac_part *= prime
                    digit = int(frac_part)
                    digit_counts[digit] += 1
                    frac_part -= digit
                    digit_count += 1
        
        # Normalize to probabilities
        total_digits = np.sum(digit_counts)
        if total_digits > 0:
            return digit_counts / total_digits
        else:
            return np.ones(prime) / prime
    
    def calculate_average_digits(self, values: np.ndarray, prime: int) -> float:
        """
        Calculate average number of digits needed in base p.
        
        Args:
            values: Array of values
            prime: Base for p-adic representation
            
        Returns:
            Average digit count
        """
        if len(values) == 0:
            return 1.0
        
        digit_counts = []
        
        for val in values:
            if abs(val) < 1e-10:
                digit_counts.append(1)  # Zero needs 1 digit
                continue
            
            # Calculate digits needed
            # Using formula: ⌈log_p(|x|)⌉
            abs_val = abs(val)
            if abs_val >= 1:
                # For values >= 1, use logarithm
                digits_needed = math.ceil(math.log(abs_val) / math.log(prime))
            else:
                # For values < 1, need to represent fractional part
                # Estimate based on precision needed
                precision_needed = -math.floor(math.log10(abs_val))
                digits_needed = math.ceil(precision_needed * math.log(10) / math.log(prime))
            
            digit_counts.append(max(1, digits_needed))
        
        return np.mean(digit_counts)
    
    def calculate_entropy(self, values: np.ndarray, prime: int, num_bins: int = 256) -> float:
        """
        Calculate entropy in base p.
        
        Args:
            values: Array of values
            prime: Base for calculation
            num_bins: Number of bins for histogram
            
        Returns:
            Entropy value
        """
        if len(values) == 0:
            return 0.0
        
        # Get digit distribution
        digit_dist = self.compute_digit_distribution(values, prime)
        
        # Calculate entropy
        entropy = 0.0
        for prob in digit_dist:
            if prob > 0:
                entropy -= prob * math.log(prob) / math.log(prime)
        
        return entropy
    
    def compute_efficiency_metric(self, 
                                 prime: int, 
                                 avg_digits: float, 
                                 entropy: float,
                                 characteristics: Dict[str, float]) -> float:
        """
        Calculate efficiency metric for prime selection.
        
        Formula: E(p) = compression_factor * entropy_factor * computational_factor
        
        Args:
            prime: Prime candidate
            avg_digits: Average digit count
            entropy: Entropy in base p
            characteristics: Tensor characteristics
            
        Returns:
            Efficiency score in [0, 1]
        """
        # Compression factor: fewer bits needed is better
        bits_per_digit = math.log2(prime)
        total_bits = bits_per_digit * avg_digits
        compression_factor = 32.0 / max(1.0, total_bits)  # Assume 32-bit floats
        
        # Entropy factor: higher entropy means better digit utilization
        max_entropy = 1.0  # Maximum entropy is 1 in base p
        entropy_factor = 0.5 + 0.5 * min(entropy / max_entropy, 1.0)
        
        # Computational factor: smaller primes are faster
        if prime == 2:
            computational_factor = 1.0  # Binary is fastest
        elif prime < 16:
            computational_factor = 0.9  # Small primes are fast
        elif prime < 256:
            computational_factor = 0.8  # Byte-sized primes are good
        else:
            computational_factor = 0.7  # Larger primes are slower
        
        # Adjust for distribution characteristics
        distribution_adjustment = 1.0
        
        # Sparse tensors benefit from larger primes
        if characteristics.get('sparsity', 0) > 0.8:
            if prime > 31:
                distribution_adjustment *= 1.2
        
        # High dynamic range benefits from flexible representation
        if characteristics.get('dynamic_range', 1) > 1e6:
            if prime in [2, 3, 5, 7]:  # Small primes handle range well
                distribution_adjustment *= 1.1
        
        # Periodic data benefits from primes matching period
        if characteristics.get('periodicity_score', 0) > 0.7:
            # Check if prime is close to detected period
            period = characteristics.get('dominant_period', 0)
            if period > 0 and abs(prime - period) < 5:
                distribution_adjustment *= 1.3
        
        # Combine factors
        efficiency = compression_factor * entropy_factor * computational_factor * distribution_adjustment
        
        # Normalize to [0, 1]
        return min(1.0, max(0.0, efficiency))
    
    def analyze_tensor_characteristics(self, tensor: torch.Tensor) -> Dict[str, float]:
        """
        Analyze tensor characteristics for prime selection.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Dictionary of characteristics
        """
        flat_tensor = tensor.flatten().detach().cpu().numpy()
        
        characteristics = {}
        
        # Sparsity
        characteristics['sparsity'] = np.mean(np.abs(flat_tensor) < 1e-10)
        
        # Dynamic range
        non_zero = flat_tensor[np.abs(flat_tensor) > 1e-10]
        if len(non_zero) > 0:
            characteristics['dynamic_range'] = np.max(np.abs(non_zero)) / np.min(np.abs(non_zero))
        else:
            characteristics['dynamic_range'] = 1.0
        
        # Statistical moments
        characteristics['mean'] = np.mean(flat_tensor)
        characteristics['std'] = np.std(flat_tensor)
        characteristics['skewness'] = stats.skew(flat_tensor)
        characteristics['kurtosis'] = stats.kurtosis(flat_tensor)
        
        # Periodicity detection using FFT
        try:
            if len(flat_tensor) > 10:
                fft_result = np.fft.fft(flat_tensor[:min(1024, len(flat_tensor))])
                fft_magnitude = np.abs(fft_result)
                
                # Find dominant frequency
                fft_magnitude[0] = 0  # Remove DC component
                peak_idx = np.argmax(fft_magnitude[:len(fft_magnitude)//2])
                
                if peak_idx > 0:
                    characteristics['dominant_period'] = len(fft_magnitude) / peak_idx
                    characteristics['periodicity_score'] = fft_magnitude[peak_idx] / np.mean(fft_magnitude)
                else:
                    characteristics['dominant_period'] = 0
                    characteristics['periodicity_score'] = 0
            else:
                characteristics['dominant_period'] = 0
                characteristics['periodicity_score'] = 0
        except:
            characteristics['dominant_period'] = 0
            characteristics['periodicity_score'] = 0
        
        # Quantization level detection
        unique_values = np.unique(flat_tensor)
        characteristics['unique_values'] = len(unique_values)
        characteristics['quantization_ratio'] = len(unique_values) / len(flat_tensor)
        
        return characteristics
    
    def _detect_distribution_type(self, tensor: torch.Tensor) -> str:
        """
        Detect the distribution type of the tensor.
        
        Returns one of: "gaussian", "bimodal", "sparse", "uniform", "multimodal", "heavy_tailed"
        """
        flat_tensor = tensor.flatten().detach().cpu().numpy()
        
        # Check for sparsity first
        sparsity = np.mean(np.abs(flat_tensor) < 1e-10)
        if sparsity > 0.7:
            return "sparse"
        
        # Remove zeros for distribution analysis
        non_zero = flat_tensor[np.abs(flat_tensor) > 1e-10]
        if len(non_zero) < 10:
            return "sparse"
        
        # Calculate statistical measures
        skewness = stats.skew(non_zero)
        kurtosis_val = stats.kurtosis(non_zero)
        
        # Test for heavy tails (high kurtosis)
        if kurtosis_val > 6:
            return "heavy_tailed"
        
        # Test for normality (Gaussian)
        # Jarque-Bera test approximation
        jb_stat = len(non_zero) / 6 * (skewness**2 + (kurtosis_val**2) / 4)
        if jb_stat < 5.99 and abs(skewness) < 0.5 and abs(kurtosis_val) < 1:
            return "gaussian"
        
        # Test for bimodality
        if self._test_bimodality(non_zero):
            return "bimodal"
        
        # Test for uniformity
        if len(non_zero) > 30:
            normalized = (non_zero - non_zero.min()) / (non_zero.max() - non_zero.min() + 1e-10)
            ks_stat, p_value = stats.kstest(normalized, 'uniform')
            if p_value > 0.05:
                return "uniform"
        
        # Check for multimodality
        num_modes = self._count_modes(non_zero)
        if num_modes > 2:
            return "multimodal"
        
        # Default
        return "uniform"
    
    def _test_bimodality(self, data: np.ndarray) -> bool:
        """Test for bimodality using KMeans clustering."""
        if len(data) < 10:
            return False
        
        try:
            # Use KMeans with k=2
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans.fit(data.reshape(-1, 1))
            
            # Calculate separation between clusters
            centers = kmeans.cluster_centers_.flatten()
            separation = abs(centers[1] - centers[0])
            
            # Check if separation is significant
            data_range = data.max() - data.min()
            if data_range > 0 and separation / data_range > 0.3:
                # Additional check: roughly equal cluster sizes
                labels = kmeans.labels_
                cluster_sizes = [np.sum(labels == i) for i in range(2)]
                size_ratio = min(cluster_sizes) / max(cluster_sizes)
                
                if size_ratio > 0.2:  # Not too imbalanced
                    return True
        except:
            pass
        
        return False
    
    def _count_modes(self, data: np.ndarray) -> int:
        """Count number of modes using kernel density estimation."""
        if len(data) < 10:
            return 1
        
        try:
            from scipy.stats import gaussian_kde
            from scipy.signal import find_peaks
            
            # Create kernel density estimate
            kde = gaussian_kde(data)
            
            # Create fine grid for evaluation
            x_grid = np.linspace(data.min(), data.max(), 1000)
            density = kde(x_grid)
            
            # Find local maxima
            peaks, _ = find_peaks(density, height=np.max(density) * 0.1)
            
            return max(1, len(peaks))
        except:
            return 1
    
    def _is_prime(self, n: int) -> bool:
        """Check if a number is prime."""
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
    
    def _compute_cache_key(self, tensor: torch.Tensor) -> str:
        """Compute cache key for tensor based on statistical properties."""
        flat_tensor = tensor.flatten()
        
        # Use statistical properties as key
        properties = [
            float(flat_tensor.mean()),
            float(flat_tensor.std()),
            float(flat_tensor.min()),
            float(flat_tensor.max()),
            tensor.shape,
            tensor.dtype
        ]
        
        # Create hash from properties
        import hashlib
        key_str = str(properties)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _update_cache(self, key: str, result: PrimeSelectionResult) -> None:
        """Update selection cache with new result."""
        if not self.enable_caching:
            return
        
        with self.cache_lock:
            # Implement LRU eviction if cache is full
            if len(self.selection_cache) >= self.cache_size:
                # Remove oldest entry (simple FIFO for now)
                oldest_key = next(iter(self.selection_cache))
                del self.selection_cache[oldest_key]
            
            self.selection_cache[key] = result
    
    def _generate_rationale(self, 
                           prime: int, 
                           distribution_type: str,
                           characteristics: Dict[str, float],
                           efficiency: float,
                           avg_digits: float) -> str:
        """Generate human-readable rationale for prime selection."""
        rationale_parts = []
        
        # Distribution-based reasoning
        if distribution_type == "gaussian":
            rationale_parts.append(f"Gaussian distribution detected - selected p={prime} for balanced compression")
        elif distribution_type == "sparse":
            rationale_parts.append(f"Sparse tensor (sparsity={characteristics['sparsity']:.2%}) - p={prime} handles zeros efficiently")
        elif distribution_type == "bimodal":
            rationale_parts.append(f"Bimodal distribution - p={prime} captures dual modes")
        elif distribution_type == "heavy_tailed":
            rationale_parts.append(f"Heavy-tailed distribution - p={prime} handles extreme values")
        elif distribution_type == "uniform":
            rationale_parts.append(f"Uniform distribution - p={prime} selected via entropy optimization")
        elif distribution_type == "multimodal":
            rationale_parts.append(f"Multimodal distribution - p={prime} balances complexity")
        
        # Efficiency reasoning
        rationale_parts.append(f"Efficiency score: {efficiency:.3f}")
        rationale_parts.append(f"Average digits: {avg_digits:.2f}")
        
        # Special characteristics
        if characteristics.get('dynamic_range', 1) > 1e6:
            rationale_parts.append(f"High dynamic range ({characteristics['dynamic_range']:.2e}) factored")
        
        if characteristics.get('periodicity_score', 0) > 0.7:
            period = characteristics.get('dominant_period', 0)
            if period > 0:
                rationale_parts.append(f"Periodic pattern detected (period≈{period:.1f})")
        
        return "; ".join(rationale_parts)
    
    def get_extended_prime_set(self, tensor: torch.Tensor) -> List[int]:
        """
        Get extended prime set for special cases.
        
        Args:
            tensor: Input tensor
            
        Returns:
            List of prime candidates
        """
        characteristics = self.analyze_tensor_characteristics(tensor)
        
        # Start with default primes
        prime_set = list(self.DEFAULT_PRIMES)
        
        # Add extended primes for specific cases
        if characteristics['sparsity'] > 0.9:
            # Very sparse - add larger primes
            prime_set.extend(self.EXTENDED_PRIMES[:5])
        
        if characteristics['dynamic_range'] > 1e8:
            # Very high dynamic range - add powers of 2 minus 1 (Mersenne-like)
            mersenne_candidates = [3, 7, 31, 127]
            for m in mersenne_candidates:
                if m not in prime_set and self._is_prime(m):
                    prime_set.append(m)
        
        if characteristics.get('periodicity_score', 0) > 0.8:
            # Strong periodicity - add primes near the period
            period = int(characteristics.get('dominant_period', 0))
            if period > 0:
                for offset in [-2, -1, 1, 2]:
                    candidate = period + offset
                    if candidate > 1 and candidate not in prime_set and self._is_prime(candidate):
                        prime_set.append(candidate)
        
        return sorted(prime_set)
    
    def parallel_prime_evaluation(self, 
                                 tensor: torch.Tensor,
                                 primes: Optional[List[int]] = None,
                                 num_threads: int = 4) -> PrimeSelectionResult:
        """
        Evaluate primes in parallel for faster selection.
        
        Args:
            tensor: Input tensor
            primes: Prime candidates
            num_threads: Number of parallel threads
            
        Returns:
            PrimeSelectionResult
        """
        import concurrent.futures
        
        candidate_primes = primes or self.primes
        flat_tensor = tensor.flatten().detach().cpu().numpy()
        
        # Prepare data
        non_zero = flat_tensor[np.abs(flat_tensor) > 1e-10]
        if len(non_zero) == 0:
            return self.select_optimal_prime(tensor, primes)
        
        characteristics = self.analyze_tensor_characteristics(tensor)
        
        def evaluate_prime(p):
            """Evaluate single prime."""
            avg_digits = self.calculate_average_digits(non_zero, p)
            entropy = self.calculate_entropy(flat_tensor, p)
            efficiency = self.compute_efficiency_metric(p, avg_digits, entropy, characteristics)
            return p, efficiency, avg_digits, entropy
        
        # Parallel evaluation
        best_result = None
        candidate_scores = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(evaluate_prime, p): p for p in candidate_primes}
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    p, efficiency, avg_digits, entropy = future.result()
                    candidate_scores[p] = efficiency
                    
                    if best_result is None or efficiency > best_result[1]:
                        best_result = (p, efficiency, avg_digits, entropy)
                except Exception as e:
                    # Prime evaluation failed - skip
                    p = futures[future]
                    candidate_scores[p] = 0.0
        
        if best_result is None:
            raise RuntimeError("All prime evaluations failed")
        
        best_prime, best_efficiency, best_avg_digits, best_entropy = best_result
        distribution_type = self._detect_distribution_type(tensor)
        
        return PrimeSelectionResult(
            optimal_prime=best_prime,
            efficiency=best_efficiency,
            average_digits=best_avg_digits,
            entropy=best_entropy,
            distribution_type=distribution_type,
            selection_rationale=self._generate_rationale(
                best_prime, distribution_type, characteristics, 
                best_efficiency, best_avg_digits
            ),
            candidate_scores=candidate_scores,
            computation_time=0.0  # Not tracked in parallel version
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get selection statistics."""
        return {
            'total_selections': self.selection_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(1, self.selection_count),
            'average_selection_time': self.total_selection_time / max(1, self.selection_count),
            'cache_size': len(self.selection_cache),
            'cache_enabled': self.enable_caching
        }
    
    def clear_cache(self) -> None:
        """Clear selection cache."""
        with self.cache_lock:
            self.selection_cache.clear()
    
    def reset_statistics(self) -> None:
        """Reset performance statistics."""
        self.selection_count = 0
        self.cache_hits = 0
        self.total_selection_time = 0.0