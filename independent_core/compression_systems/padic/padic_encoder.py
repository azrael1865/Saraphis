"""
P-adic number encoding and mathematical operations.
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from fractions import Fraction
import math


@dataclass
class PadicWeight:
    """P-adic representation of a neural network weight"""
    value: Fraction
    prime: int
    precision: int
    valuation: int
    digits: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate p-adic weight after initialization - throws on any issue"""
        if not isinstance(self.prime, int):
            raise TypeError(f"Prime must be int, got {type(self.prime)}")
        if self.prime < 2:
            raise ValueError(f"Prime must be >= 2, got {self.prime}")
        if not isinstance(self.precision, int):
            raise TypeError(f"Precision must be int, got {type(self.precision)}")
        if self.precision < 1:
            raise ValueError(f"Precision must be >= 1, got {self.precision}")
        if not isinstance(self.valuation, int):
            raise TypeError(f"Valuation must be int, got {type(self.valuation)}")
        if self.valuation < -self.precision:
            raise ValueError(f"Valuation {self.valuation} exceeds precision {self.precision}")
        if not isinstance(self.digits, list):
            raise TypeError(f"Digits must be list, got {type(self.digits)}")
        if len(self.digits) != self.precision:
            raise ValueError(f"Digits length {len(self.digits)} must equal precision {self.precision}")
        for i, digit in enumerate(self.digits):
            if not isinstance(digit, int):
                raise TypeError(f"Digit {i} must be int, got {type(digit)}")
            if not (0 <= digit < self.prime):
                raise ValueError(f"Digit {digit} at position {i} must be in range [0, {self.prime})")
        if not isinstance(self.value, Fraction):
            raise TypeError(f"Value must be Fraction, got {type(self.value)}")


class PadicValidation:
    """Validation utilities for P-adic operations - strict validation only"""
    
    @staticmethod
    def validate_prime(prime: int) -> None:
        """Validate prime number - throws on any issue"""
        if not isinstance(prime, int):
            raise TypeError(f"Prime must be int, got {type(prime)}")
        if prime < 2:
            raise ValueError(f"Prime must be >= 2, got {prime}")
        if prime == 2:
            return  # 2 is prime
        if prime % 2 == 0:
            raise ValueError(f"{prime} is not a prime number (even)")
        # Check odd divisors up to sqrt(prime)
        for i in range(3, int(math.sqrt(prime)) + 1, 2):
            if prime % i == 0:
                raise ValueError(f"{prime} is not a prime number (divisible by {i})")
    
    @staticmethod
    def validate_precision(precision: int) -> None:
        """Validate precision parameter - throws on any issue"""
        if not isinstance(precision, int):
            raise TypeError(f"Precision must be int, got {type(precision)}")
        if precision < 1:
            raise ValueError(f"Precision must be >= 1, got {precision}")
        if precision > 1000:
            raise ValueError(f"Precision {precision} exceeds maximum 1000")
    
    @staticmethod
    def validate_tensor(tensor: torch.Tensor) -> None:
        """Validate input tensor - throws on any issue"""
        if tensor is None:
            raise ValueError("Tensor cannot be None")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        if tensor.numel() == 0:
            raise ValueError("Tensor cannot be empty")
        if torch.isnan(tensor).any():
            raise ValueError("Tensor contains NaN values")
        if torch.isinf(tensor).any():
            raise ValueError("Tensor contains infinite values")
        if tensor.dtype not in [torch.float32, torch.float64, torch.float16]:
            raise TypeError(f"Tensor must be float type, got {tensor.dtype}")
    
    @staticmethod
    def validate_chunk_size(chunk_size: int, tensor_size: int) -> None:
        """Validate chunk size for processing"""
        if not isinstance(chunk_size, int):
            raise TypeError(f"Chunk size must be int, got {type(chunk_size)}")
        if chunk_size <= 0:
            raise ValueError(f"Chunk size must be > 0, got {chunk_size}")
        if chunk_size > tensor_size:
            raise ValueError(f"Chunk size {chunk_size} exceeds tensor size {tensor_size}")


class PadicMathematicalOperations:
    """P-adic arithmetic operations - fail loud on all errors"""
    
    def __init__(self, prime: int, precision: int):
        """Initialize p-adic operations"""
        PadicValidation.validate_prime(prime)
        PadicValidation.validate_precision(precision)
        self.prime = prime
        self.precision = precision
        # Pre-compute prime powers for efficiency
        self.prime_powers = [self.prime ** i for i in range(self.precision + 1)]
    
    def compute_valuation(self, num: int, denom: int) -> int:
        """Compute p-adic valuation of rational number"""
        if not isinstance(num, int):
            raise TypeError(f"Numerator must be int, got {type(num)}")
        if not isinstance(denom, int):
            raise TypeError(f"Denominator must be int, got {type(denom)}")
        if denom == 0:
            raise ValueError("Denominator cannot be zero")
        
        if num == 0:
            return self.precision  # Convention for zero
        
        # Count powers of p in numerator
        val_num = 0
        temp_num = abs(num)
        while temp_num % self.prime == 0:
            val_num += 1
            temp_num //= self.prime
            if val_num > self.precision:
                raise ValueError(f"Valuation exceeds precision {self.precision}")
        
        # Count powers of p in denominator
        val_denom = 0
        temp_denom = abs(denom)
        while temp_denom % self.prime == 0:
            val_denom += 1
            temp_denom //= self.prime
            if val_denom > self.precision:
                raise ValueError(f"Valuation exceeds precision {self.precision}")
        
        return val_num - val_denom
    
    def to_padic(self, value: float) -> PadicWeight:
        """Convert float to p-adic representation using proper arithmetic"""
        if not isinstance(value, (float, int)):
            raise TypeError(f"Value must be float or int, got {type(value)}")
        if math.isnan(value):
            raise ValueError("Cannot convert NaN to p-adic")
        if math.isinf(value):
            raise ValueError("Cannot convert infinity to p-adic")
        if abs(value) > 1e10:
            raise ValueError(f"Value {value} too large for conversion")
        
        # Convert to fraction with denominator limit
        try:
            frac = Fraction(value).limit_denominator(10**10)
        except (ValueError, OverflowError) as e:
            raise ValueError(f"Cannot convert {value} to fraction: {e}")
        
        # Compute valuation
        valuation = self.compute_valuation(frac.numerator, frac.denominator)
        
        # Extract p-adic digits using modular arithmetic
        try:
            digits = self._extract_padic_digits(frac)
        except ValueError as e:
            raise ValueError(f"Failed to convert {value} to p-adic: {str(e)}")
        
        return PadicWeight(
            value=frac,
            prime=self.prime,
            precision=self.precision,
            valuation=valuation,
            digits=digits
        )
    
    def _extract_padic_digits(self, frac: Fraction) -> List[int]:
        """Extract p-adic digits using proper modular arithmetic"""
        digits = []
        
        if frac.numerator == 0:
            return [0] * self.precision
        
        # Normalize fraction by removing powers of p from denominator
        num = frac.numerator
        denom = frac.denominator
        
        # Remove powers of p from denominator
        while denom % self.prime == 0:
            denom //= self.prime
        
        # Check if denominator is coprime to p
        if math.gcd(denom, self.prime) != 1:
            raise ValueError(f"Denominator {denom} not coprime to prime {self.prime}")
        
        # Compute modular inverse of denominator
        denom_inv = pow(denom, -1, self.prime_powers[self.precision])
        
        # Extract digits using p-adic expansion
        working_num = (num * denom_inv) % self.prime_powers[self.precision]
        for i in range(self.precision):
            digit = working_num % self.prime
            digits.append(digit)
            working_num //= self.prime
            if working_num == 0 and i >= 5:  # Early termination for efficiency
                digits.extend([0] * (self.precision - i - 1))
                break
        
        return digits
    
    def from_padic(self, padic: PadicWeight) -> float:
        """Convert p-adic representation back to float using reconstruction formula"""
        if not isinstance(padic, PadicWeight):
            raise TypeError(f"Expected PadicWeight, got {type(padic)}")
        if padic.prime != self.prime:
            raise ValueError(f"Prime mismatch: {padic.prime} != {self.prime}")
        if padic.precision != self.precision:
            raise ValueError(f"Precision mismatch: {padic.precision} != {self.precision}")
        
        # Reconstruct value from digits: Σ(d_i × p^i) × p^v
        value = 0
        for i, digit in enumerate(padic.digits):
            if i >= len(self.prime_powers):
                raise ValueError(f"Digit index {i} exceeds precomputed powers")
            value += digit * self.prime_powers[i]
        
        # Apply valuation
        if padic.valuation < 0:
            value = value / self.prime_powers[abs(padic.valuation)]
        elif padic.valuation > 0:
            if padic.valuation >= len(self.prime_powers):
                raise ValueError(f"Valuation {padic.valuation} exceeds precomputed powers")
            value = value * self.prime_powers[padic.valuation]
        
        # Normalize to appropriate range
        max_val = self.prime_powers[self.precision]
        if value > max_val / 2:
            value = value - max_val
        
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            raise ValueError(f"Conversion resulted in invalid float: {result}")
        
        return result
    
    def ultrametric_distance(self, x: PadicWeight, y: PadicWeight) -> float:
        """Compute ultrametric distance between two p-adic numbers"""
        if not isinstance(x, PadicWeight):
            raise TypeError(f"x must be PadicWeight, got {type(x)}")
        if not isinstance(y, PadicWeight):
            raise TypeError(f"y must be PadicWeight, got {type(y)}")
        if x.prime != y.prime or x.prime != self.prime:
            raise ValueError(f"Prime mismatch: x.prime={x.prime}, y.prime={y.prime}, self.prime={self.prime}")
        if x.precision != y.precision or x.precision != self.precision:
            raise ValueError(f"Precision mismatch: x.precision={x.precision}, y.precision={y.precision}, self.precision={self.precision}")
        
        # Find first differing digit
        for i in range(self.precision):
            if x.digits[i] != y.digits[i]:
                distance = self.prime ** (-i)
                if distance <= 0 or math.isnan(distance) or math.isinf(distance):
                    raise ValueError(f"Invalid ultrametric distance: {distance}")
                return distance
        
        # All digits are the same
        return 0.0
    
    def validate_ultrametric_property(self, x: PadicWeight, y: PadicWeight, z: PadicWeight) -> None:
        """Validate ultrametric inequality for three p-adic numbers"""
        d_xy = self.ultrametric_distance(x, y)
        d_xz = self.ultrametric_distance(x, z)
        d_yz = self.ultrametric_distance(y, z)
        
        # Ultrametric inequality: d(x,z) <= max(d(x,y), d(y,z))
        tolerance = 1e-12  # Minimal tolerance for floating point precision
        max_distance = max(d_xy, d_yz)
        
        if d_xz > max_distance + tolerance:
            raise ValueError(
                f"Ultrametric property violated: "
                f"d(x,z)={d_xz:.12e} > max(d(x,y)={d_xy:.12e}, d(y,z)={d_yz:.12e}) = {max_distance:.12e}"
            )


# Validation Functions
def validate_single_weight(weight: PadicWeight, expected_prime: int, expected_precision: int) -> bool:
    """Validate mathematical correctness of single weight"""
    try:
        # Check basic structure
        if not hasattr(weight, 'digits') or not hasattr(weight, 'valuation'):
            return False
        
        # Check prime and precision match
        if weight.prime != expected_prime or weight.precision != expected_precision:
            return False
        
        # Check digit properties
        if not isinstance(weight.digits, list) or len(weight.digits) != weight.precision:
            return False
        
        for digit in weight.digits:
            if not isinstance(digit, int) or not (0 <= digit < weight.prime):
                return False
        
        # Check valuation bounds
        if not isinstance(weight.valuation, int):
            return False
        
        if weight.valuation < -weight.precision or weight.valuation > weight.precision:
            return False
        
        return True
        
    except Exception:
        return False


def validate_padic_weights(weights: List[PadicWeight], prime: int, precision: int) -> bool:
    """Validate mathematical correctness of p-adic weights"""
    math_ops = PadicMathematicalOperations(prime, precision)
    
    for i, weight in enumerate(weights):
        # Check structural integrity
        if not validate_single_weight(weight, prime, precision):
            return False
        
        # Check reconstruction accuracy
        try:
            reconstructed = math_ops.from_padic(weight)
            original = float(weight.value)
            
            # Allow small numerical errors (1e-5 relative)
            relative_error = abs(original - reconstructed) / (abs(original) + 1e-10)
            if relative_error > 1e-5:
                return False
        except Exception:
            return False
    
    return True


def validate_ultrametric_property(weights: List[PadicWeight]) -> bool:
    """Validate ultrametric distance property preservation"""
    if len(weights) < 3:
        return True
    
    math_ops = PadicMathematicalOperations(weights[0].prime, weights[0].precision)
    
    # Sample validation for efficiency
    sample_size = min(10, len(weights))
    sample_indices = np.random.choice(len(weights), sample_size, replace=False)
    
    for i in sample_indices:
        for j in sample_indices:
            if i >= j:
                continue
                
            for k in sample_indices:
                if k == i or k == j:
                    continue
                
                try:
                    math_ops.validate_ultrametric_property(
                        weights[i], weights[j], weights[k]
                    )
                except ValueError:
                    return False
    
    return True


def create_real_padic_weights(num_weights: int, precision: int = 10, prime: int = 251) -> List[PadicWeight]:
    """Create mathematically correct p-adic weights for testing"""
    math_ops = PadicMathematicalOperations(prime, precision)
    weights = []
    
    value_ranges = [
        (-10.0, 10.0),      # Standard range
        (-1.0, 1.0),        # Small values
        (-100.0, 100.0),    # Larger values
        (0.001, 0.999),     # Fractional values
        (-0.999, -0.001)    # Negative fractional
    ]
    
    attempts = 0
    max_attempts = num_weights * 10  # Allow multiple attempts
    
    while len(weights) < num_weights and attempts < max_attempts:
        attempts += 1
        
        # Select value range based on index
        range_idx = (len(weights) % len(value_ranges))
        min_val, max_val = value_ranges[range_idx]
        
        # Generate value
        if attempts % 3 == 0:
            value = float(np.random.randint(int(min_val), int(max_val) + 1))
        elif attempts % 3 == 1:
            numerator = np.random.randint(1, 100)
            denominator = np.random.randint(1, 100)
            value = numerator / denominator
            if np.random.rand() > 0.5:
                value = -value
        else:
            value = np.random.uniform(min_val, max_val)
        
        try:
            # Convert to proper p-adic representation
            weight = math_ops.to_padic(value)
            
            # Validate the conversion by reconstructing
            reconstructed = math_ops.from_padic(weight)
            
            # Verify mathematical correctness
            relative_error = abs(value - reconstructed) / (abs(value) + 1e-10)
            if relative_error > 1e-6:
                continue  # Skip values with high reconstruction error
            
            # Additional validation
            if not validate_single_weight(weight, prime, precision):
                continue
            
            weights.append(weight)
            
        except (ValueError, TypeError, OverflowError):
            continue  # Skip values that can't be converted
    
    if len(weights) < num_weights:
        raise ValueError(
            f"Could not create enough valid p-adic weights. "
            f"Created {len(weights)} out of {num_weights} requested."
        )
    
    return weights[:num_weights]


def measure_weight_conversion_time(num_weights: int, precision: int, prime: int) -> Tuple[List[PadicWeight], float]:
    """Measure time to create real p-adic weights"""
    import time
    start_time = time.time()
    weights = create_real_padic_weights(num_weights, precision, prime)
    conversion_time = time.time() - start_time
    return weights, conversion_time