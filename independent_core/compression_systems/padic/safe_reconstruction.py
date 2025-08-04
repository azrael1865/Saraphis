"""
P-adic Reconstruction Solution with Overflow Prevention

This module provides production-ready solutions for p-adic reconstruction
that avoid mathematical overflow while maintaining precision.
"""

import numpy as np
import torch
from typing import Tuple, List, Optional, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum
import math
from decimal import Decimal, getcontext


class ReconstructionMethod(Enum):
    """Available reconstruction methods for different precision requirements."""
    DIRECT = "direct"  # Traditional sum(digit_i * prime^i)
    LOGARITHMIC = "logarithmic"  # Log-space computation
    MODULAR = "modular"  # Modular arithmetic approach
    HYBRID = "hybrid"  # Automatic method selection
    CHUNKED = "chunked"  # Process in chunks to avoid overflow
    DECIMAL_PRECISION = "decimal_precision"  # Use Python's decimal module


@dataclass
class ReconstructionConfig:
    """Configuration for p-adic reconstruction."""
    prime: int = 257
    max_safe_precision: int = 6  # Conservative default
    method: ReconstructionMethod = ReconstructionMethod.HYBRID
    overflow_threshold: float = 1e15  # INCREASED from 1e12 to 1e15 for better handling
    use_gpu: bool = True
    chunk_size: int = 4  # For chunked reconstruction
    decimal_precision: int = 50  # For decimal module


@dataclass
class PadicWeight:
    """P-adic weight representation."""
    digits: List[int]
    valuation: int
    precision: int
    prime: int


class SafePadicReconstructor:
    """
    Production-ready p-adic reconstruction with overflow prevention.
    
    Implements multiple reconstruction strategies to handle different
    precision requirements while avoiding numerical overflow.
    """
    
    def __init__(self, config: Optional[ReconstructionConfig] = None):
        self.config = config or ReconstructionConfig()
        self._validate_config()
        self._compute_safe_limits()
        
        # Pre-compute prime powers for efficiency
        self._precompute_prime_powers()
        
        # Set decimal precision for high-precision operations
        getcontext().prec = self.config.decimal_precision
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config.prime < 2:
            raise ValueError(f"Prime must be >= 2, got {self.config.prime}")
        
        if self.config.max_safe_precision < 1:
            raise ValueError(f"Max safe precision must be >= 1, got {self.config.max_safe_precision}")
        
        if self.config.overflow_threshold <= 0:
            raise ValueError(f"Overflow threshold must be positive, got {self.config.overflow_threshold}")
    
    def _compute_safe_limits(self):
        """Compute safe precision limits for the given prime."""
        # Calculate maximum safe precision based on float64 limits
        max_float64 = np.finfo(np.float64).max
        safe_precision = 0
        
        test_val = 1.0
        while test_val * self.config.prime < max_float64 / self.config.prime:
            test_val *= self.config.prime
            safe_precision += 1
        
        # Update config with computed safe limit
        self.config.max_safe_precision = min(
            self.config.max_safe_precision,
            safe_precision - 2  # Conservative buffer
        )
        
        # Store prime-specific thresholds
        self.safe_thresholds = {
            2: 53,    # 2^53 is exact in float64
            3: 33,    # 3^33 < 2^53
            5: 22,    # 5^22 < 2^53
            7: 19,    # 7^19 < 2^53
            11: 15,   # 11^15 < 2^53
            13: 14,   # 13^14 < 2^53
            17: 13,   # 17^13 < 2^53
            257: 6,   # 257^6 < 2^53
        }
    
    def _precompute_prime_powers(self):
        """Pre-compute prime powers for efficiency."""
        max_precompute = min(20, self.config.max_safe_precision + 5)
        self.prime_powers = np.zeros(max_precompute, dtype=np.float64)
        self.prime_powers[0] = 1.0
        
        for i in range(1, max_precompute):
            self.prime_powers[i] = self.prime_powers[i-1] * self.config.prime
            if self.prime_powers[i] > self.config.overflow_threshold:
                self.prime_powers = self.prime_powers[:i]
                break
    
    def reconstruct(self, weight: PadicWeight, 
                   target_precision: Optional[int] = None) -> float:
        """
        Reconstruct float value from p-adic weight using appropriate method.
        
        Args:
            weight: P-adic weight to reconstruct
            target_precision: Optional target precision (defaults to weight precision)
            
        Returns:
            Reconstructed float value
        """
        if self.config.method == ReconstructionMethod.HYBRID:
            return self._reconstruct_hybrid(weight, target_precision)
        
        method_map = {
            ReconstructionMethod.DIRECT: self._reconstruct_direct,
            ReconstructionMethod.LOGARITHMIC: self._reconstruct_logarithmic,
            ReconstructionMethod.MODULAR: self._reconstruct_modular,
            ReconstructionMethod.CHUNKED: self._reconstruct_chunked,
            ReconstructionMethod.DECIMAL_PRECISION: self._reconstruct_decimal,
        }
        
        return method_map[self.config.method](weight, target_precision)
    
    def _reconstruct_hybrid(self, weight: PadicWeight, 
                           target_precision: Optional[int] = None) -> float:
        """
        Automatically select best reconstruction method based on parameters.
        
        This is the recommended approach for production use.
        """
        effective_precision = self._get_effective_precision(weight, target_precision)
        
        # Decision tree for method selection
        if effective_precision <= self.config.max_safe_precision:
            # Safe for direct reconstruction
            return self._reconstruct_direct(weight, target_precision)
        
        elif effective_precision <= 10 and self.config.prime <= 10:
            # Use logarithmic for small primes with moderate precision
            return self._reconstruct_logarithmic(weight, target_precision)
        
        elif effective_precision <= 20:
            # Use chunked reconstruction for moderate precision
            return self._reconstruct_chunked(weight, target_precision)
        
        else:
            # Use decimal precision for very high precision
            return self._reconstruct_decimal(weight, target_precision)
    
    def _get_effective_precision(self, weight: PadicWeight, 
                                target_precision: Optional[int] = None) -> int:
        """Calculate effective precision for reconstruction."""
        available_precision = len(weight.digits)
        requested_precision = target_precision or weight.precision
        
        return min(available_precision, requested_precision)
    
    def _reconstruct_direct(self, weight: PadicWeight, 
                           target_precision: Optional[int] = None) -> float:
        """
        Direct reconstruction with overflow protection.
        
        Only used when precision is guaranteed safe.
        """
        effective_precision = self._get_effective_precision(weight, target_precision)
        
        # Safety check
        if effective_precision > self.config.max_safe_precision:
            raise ValueError(
                f"Direct reconstruction unsafe for precision {effective_precision} "
                f"(max safe: {self.config.max_safe_precision})"
            )
        
        value = 0.0
        for i in range(effective_precision):
            digit = self._validate_digit(weight.digits[i])
            
            if i < len(self.prime_powers):
                value += digit * self.prime_powers[i]
            else:
                value += digit * (self.config.prime ** i)
            
            # Overflow check
            if abs(value) > self.config.overflow_threshold:
                raise OverflowError(
                    f"Overflow detected at position {i}: value={value:.2e}"
                )
        
        # Apply valuation
        return self._apply_valuation(value, weight.valuation)
    
    def _reconstruct_logarithmic(self, weight: PadicWeight, 
                                target_precision: Optional[int] = None) -> float:
        """
        Reconstruction using logarithmic computation to avoid overflow.
        
        Works by computing log(sum) = log(sum(exp(log(terms))))
        """
        effective_precision = self._get_effective_precision(weight, target_precision)
        
        # Collect non-zero terms
        log_terms = []
        signs = []
        
        for i in range(effective_precision):
            digit = self._validate_digit(weight.digits[i])
            if digit != 0:
                # log(digit * prime^i) = log(digit) + i * log(prime)
                log_term = np.log(abs(digit)) + i * np.log(self.config.prime)
                log_terms.append(log_term)
                signs.append(np.sign(digit))
        
        if not log_terms:
            return 0.0
        
        # Use log-sum-exp trick for numerical stability
        log_terms = np.array(log_terms)
        signs = np.array(signs)
        
        max_log = np.max(log_terms)
        scaled_terms = signs * np.exp(log_terms - max_log)
        sum_scaled = np.sum(scaled_terms)
        
        if sum_scaled == 0:
            return 0.0
        
        result = np.sign(sum_scaled) * np.exp(max_log + np.log(abs(sum_scaled)))
        
        return self._apply_valuation(result, weight.valuation)
    
    def _reconstruct_modular(self, weight: PadicWeight, 
                            target_precision: Optional[int] = None) -> float:
        """
        Reconstruction using modular arithmetic for exact computation.
        
        Computes value modulo a large prime, then converts to float.
        """
        effective_precision = self._get_effective_precision(weight, target_precision)
        
        # Use a large prime for modular arithmetic
        modulus = 2**61 - 1  # Mersenne prime
        
        value_mod = 0
        prime_power_mod = 1
        
        for i in range(effective_precision):
            digit = self._validate_digit(weight.digits[i])
            value_mod = (value_mod + digit * prime_power_mod) % modulus
            prime_power_mod = (prime_power_mod * self.config.prime) % modulus
        
        # Convert back to float (may lose precision for very large values)
        if value_mod > modulus // 2:
            # Handle negative values in modular arithmetic
            value = float(value_mod - modulus)
        else:
            value = float(value_mod)
        
        return self._apply_valuation(value, weight.valuation)
    
    def _reconstruct_chunked(self, weight: PadicWeight, 
                            target_precision: Optional[int] = None) -> float:
        """
        Reconstruction in chunks to avoid overflow.
        
        Processes digits in groups, combining results carefully.
        """
        effective_precision = self._get_effective_precision(weight, target_precision)
        chunk_size = self.config.chunk_size
        
        chunks = []
        chunk_scales = []
        
        for chunk_start in range(0, effective_precision, chunk_size):
            chunk_end = min(chunk_start + chunk_size, effective_precision)
            
            # Reconstruct chunk
            chunk_value = 0.0
            for i in range(chunk_start, chunk_end):
                digit = self._validate_digit(weight.digits[i])
                local_power = i - chunk_start
                chunk_value += digit * (self.config.prime ** local_power)
            
            chunks.append(chunk_value)
            chunk_scales.append(chunk_start)
        
        # Combine chunks using Horner's method in reverse
        result = 0.0
        for chunk, scale in zip(reversed(chunks), reversed(chunk_scales)):
            if scale > 0:
                # Use logarithmic scaling for large powers
                log_scale = scale * np.log(self.config.prime)
                if log_scale > 100:  # Arbitrary threshold
                    # Too large to compute directly
                    if chunk != 0:
                        raise OverflowError(
                            f"Cannot safely compute prime^{scale} for non-zero chunk"
                        )
                else:
                    result = result * (self.config.prime ** scale) + chunk
            else:
                result += chunk
        
        return self._apply_valuation(result, weight.valuation)
    
    def _reconstruct_decimal(self, weight: PadicWeight, 
                            target_precision: Optional[int] = None) -> float:
        """
        High-precision reconstruction using Python's decimal module.
        
        Provides arbitrary precision at the cost of performance.
        """
        effective_precision = self._get_effective_precision(weight, target_precision)
        
        value = Decimal(0)
        prime_decimal = Decimal(self.config.prime)
        prime_power = Decimal(1)
        
        for i in range(effective_precision):
            digit = self._validate_digit(weight.digits[i])
            value += Decimal(digit) * prime_power
            prime_power *= prime_decimal
        
        # Apply valuation in decimal
        if weight.valuation > 0:
            for _ in range(weight.valuation):
                value *= prime_decimal
        elif weight.valuation < 0:
            for _ in range(abs(weight.valuation)):
                value /= prime_decimal
        
        # Convert to float (may lose precision)
        return float(value)
    
    def _validate_digit(self, digit: int) -> int:
        """Validate and clamp digit to valid range."""
        if digit < 0 or digit >= self.config.prime:
            return max(0, min(digit, self.config.prime - 1))
        return digit
    
    def _apply_valuation(self, value: float, valuation: int) -> float:
        """Apply p-adic valuation to reconstructed value."""
        if valuation == 0:
            return value
        
        if valuation > 0:
            # Check for overflow before multiplication
            max_valuation = int(np.log(self.config.overflow_threshold / abs(value)) / 
                              np.log(self.config.prime))
            if valuation > max_valuation:
                raise OverflowError(
                    f"Valuation {valuation} would cause overflow"
                )
            
            for _ in range(valuation):
                value *= self.config.prime
        else:
            # Division is safe from overflow
            for _ in range(abs(valuation)):
                value /= self.config.prime
        
        return value
    
    def reconstruct_batch_cpu(self, weights: List[PadicWeight], 
                             target_precision: Optional[int] = None) -> np.ndarray:
        """
        Batch reconstruction on CPU with parallel processing.
        
        Args:
            weights: List of p-adic weights
            target_precision: Optional target precision
            
        Returns:
            Array of reconstructed values
        """
        results = np.zeros(len(weights), dtype=np.float32)
        
        for i, weight in enumerate(weights):
            try:
                results[i] = self.reconstruct(weight, target_precision)
            except (OverflowError, ValueError) as e:
                # NO FALLBACKS - HARD FAILURE
                raise RuntimeError(f"Reconstruction failed for weight {i}: {e}")
        
        return results
    
    def reconstruct_batch_gpu(self, weights: List[PadicWeight], 
                             target_precision: Optional[int] = None) -> torch.Tensor:
        """
        Batch reconstruction on GPU with overflow detection.
        
        Args:
            weights: List of p-adic weights
            target_precision: Optional target precision
            
        Returns:
            Tensor of reconstructed values
        """
        if not self.config.use_gpu or not torch.cuda.is_available():
            # NO FALLBACKS - GPU REQUIRED
            raise RuntimeError("GPU reconstruction requested but GPU not available or disabled")
        
        # Prepare data for GPU processing
        batch_size = len(weights)
        max_precision = max(self._get_effective_precision(w, target_precision) 
                          for w in weights)
        
        # Ensure we don't exceed safe limits
        max_precision = min(max_precision, self.config.max_safe_precision)
        
        # Create GPU tensors
        digits_tensor = torch.zeros((batch_size, max_precision), 
                                  dtype=torch.int32, device='cuda')
        valuations = torch.zeros(batch_size, dtype=torch.int32, device='cuda')
        
        # Fill tensors
        for i, weight in enumerate(weights):
            eff_prec = min(max_precision, len(weight.digits))
            digits_tensor[i, :eff_prec] = torch.tensor(
                weight.digits[:eff_prec], dtype=torch.int32
            )
            valuations[i] = weight.valuation
        
        # Reconstruct on GPU
        results = self._gpu_reconstruct_kernel(
            digits_tensor, valuations, max_precision
        )
        
        # Validate results
        max_val = torch.max(torch.abs(results))
        if max_val > self.config.overflow_threshold:
            # NO FALLBACKS - HARD FAILURE
            raise OverflowError(f"GPU reconstruction produced overflow values: max={max_val:.2e}")
        
        return results
    
    def _gpu_reconstruct_kernel(self, digits: torch.Tensor, 
                               valuations: torch.Tensor,
                               precision: int) -> torch.Tensor:
        """
        GPU kernel for batch reconstruction.
        
        Uses safe precision limits and overflow detection.
        """
        batch_size = digits.shape[0]
        results = torch.zeros(batch_size, dtype=torch.float32, device='cuda')
        
        # Pre-compute prime powers on GPU
        prime_powers = torch.pow(
            self.config.prime, 
            torch.arange(precision, dtype=torch.float32, device='cuda')
        )
        
        # Batch matrix multiplication for efficiency
        # results = sum(digits * prime_powers) for each batch
        digits_float = digits.float()
        results = torch.sum(digits_float * prime_powers.unsqueeze(0), dim=1)
        
        # Apply valuations
        valuation_factors = torch.pow(
            float(self.config.prime), 
            valuations.float()
        )
        results *= valuation_factors
        
        return results


class PadicCompressionOptimizer:
    """
    Optimizer for p-adic compression parameters to avoid overflow.
    
    Automatically selects optimal prime and precision combinations.
    """
    
    def __init__(self):
        self.safe_configurations = self._build_safe_configurations()
    
    def _build_safe_configurations(self) -> List[Dict[str, Any]]:
        """Build list of safe prime/precision configurations."""
        configs = []
        
        # Small primes allow higher precision
        for prime in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            max_precision = int(52 / np.log2(prime))  # Based on float64 precision
            configs.append({
                'prime': prime,
                'max_precision': max_precision,
                'bits_per_digit': np.log2(prime),
                'efficiency': max_precision * np.log2(prime) / 8  # Bits per byte
            })
        
        # Larger primes for specific use cases
        for prime in [127, 251, 257]:
            max_precision = int(52 / np.log2(prime))
            configs.append({
                'prime': prime,
                'max_precision': max_precision,
                'bits_per_digit': np.log2(prime),
                'efficiency': max_precision * np.log2(prime) / 8
            })
        
        return sorted(configs, key=lambda x: x['efficiency'], reverse=True)
    
    def recommend_configuration(self, target_precision: int, 
                               target_bits: Optional[int] = None) -> Dict[str, Any]:
        """
        Recommend optimal configuration for given requirements.
        
        Args:
            target_precision: Desired number of p-adic digits
            target_bits: Optional target bit width
            
        Returns:
            Recommended configuration dictionary
        """
        valid_configs = [
            c for c in self.safe_configurations 
            if c['max_precision'] >= target_precision
        ]
        
        if not valid_configs:
            # Fallback to decimal precision method
            return {
                'prime': 10,
                'max_precision': target_precision,
                'method': ReconstructionMethod.DECIMAL_PRECISION,
                'warning': 'High precision requires decimal arithmetic'
            }
        
        if target_bits:
            # Find config closest to target bit width
            for config in valid_configs:
                total_bits = config['max_precision'] * config['bits_per_digit']
                if total_bits >= target_bits:
                    return config
        
        # Return most efficient valid configuration
        return valid_configs[0]


def create_safe_reconstruction_pipeline(cpu_pipeline, gpu_pipeline):
    """
    Factory function to create safe reconstruction pipeline.
    
    Replaces existing reconstruction methods with overflow-safe versions.
    """
    # Determine optimal configuration based on usage
    optimizer = PadicCompressionOptimizer()
    config_recommendation = optimizer.recommend_configuration(
        target_precision=6,  # Safe default
        target_bits=32
    )
    
    # Create safe reconstructor
    config = ReconstructionConfig(
        prime=config_recommendation['prime'],
        max_safe_precision=config_recommendation['max_precision'],
        method=ReconstructionMethod.HYBRID
    )
    
    reconstructor = SafePadicReconstructor(config)
    
    # Monkey-patch existing pipelines
    if cpu_pipeline:
        cpu_pipeline._reconstruct_float = lambda m, e, p, w: reconstructor.reconstruct(w, p)
        cpu_pipeline._safe_reconstructor = reconstructor
    
    if gpu_pipeline:
        gpu_pipeline._process_gpu_data_fixed = lambda g, b, p: reconstructor.reconstruct_batch_gpu(b, p)
        gpu_pipeline._safe_reconstructor = reconstructor
    
    return reconstructor


# Example usage for testing
if __name__ == "__main__":
    # Create test weight with SAFE precision for prime=257
    test_weight = PadicWeight(
        digits=[1, 2, 3, 4, 5, 6],  # Only 6 digits for safe precision
        valuation=0,
        precision=6,  # SAFE: precision=6 for prime=257
        prime=257
    )
    
    # Test different reconstruction methods
    config = ReconstructionConfig(prime=257, max_safe_precision=6)
    reconstructor = SafePadicReconstructor(config)
    
    try:
        # This should use hybrid method and avoid overflow
        result = reconstructor.reconstruct(test_weight)
        print(f"Reconstructed value: {result}")
    except Exception as e:
        print(f"Reconstruction failed: {e}")
    
    # Get configuration recommendation
    optimizer = PadicCompressionOptimizer()
    recommendation = optimizer.recommend_configuration(
        target_precision=6,  # SAFE: reduced from 10 to 6
        target_bits=32       # SAFE: reduced from 64 to 32
    )
    print(f"Recommended configuration: {recommendation}")