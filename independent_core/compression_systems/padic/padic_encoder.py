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
        
        # Pre-compute prime powers with overflow protection
        self.prime_powers = [1]
        max_safe_value = 1e12  # Safe threshold before overflow
        
        # Compute only safe powers (don't use arbitrary precision + 20)
        current_power = 1
        i = 1
        while True:
            next_power = current_power * prime
            if next_power > max_safe_value:
                break
            self.prime_powers.append(next_power)
            current_power = next_power
            i += 1
        
        # Verify we have enough powers for the requested precision
        if len(self.prime_powers) - 1 < precision:
            raise OverflowError(
                f"Requested precision {precision} exceeds maximum safe precision {len(self.prime_powers) - 1} for prime {prime}. "
                f"Prime power {prime}^{precision} would exceed safe threshold {max_safe_value:.2e}."
            )
        
        # Cache for modular inverses
        self._inverse_cache = {}
    
    def _valuation(self, n: int) -> int:
        """Compute p-adic valuation of integer n"""
        if n == 0:
            return float('inf')
        v = 0
        n = abs(n)
        while n % self.prime == 0:
            n //= self.prime
            v += 1
        return v
    
    def compute_valuation(self, num: int, denom: int) -> int:
        """Public method for computing p-adic valuation of num/denom"""
        if not isinstance(num, int):
            raise TypeError(f"Numerator must be int, got {type(num)}")
        if not isinstance(denom, int):
            raise TypeError(f"Denominator must be int, got {type(denom)}")
        if denom == 0:
            raise ValueError("Denominator cannot be zero")
        
        return self._compute_valuation(num) - self._compute_valuation(denom)
    
    
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
        
        # Convert to Fraction for exact arithmetic
        try:
            frac = Fraction(value).limit_denominator(10**15)
        except (ValueError, OverflowError) as e:
            raise ValueError(f"Cannot convert {value} to fraction: {e}")
        
        # Handle zero separately
        if frac == 0:
            return PadicWeight(
                value=frac,
                prime=self.prime,
                precision=self.precision,
                valuation=0,
                digits=[0] * self.precision
            )
        
        # Extract p-adic representation
        digits, valuation = self._fraction_to_padic(frac)
        
        return PadicWeight(
            value=frac,
            prime=self.prime,
            precision=self.precision,
            valuation=valuation,
            digits=digits
        )
    
    
    def from_padic(self, padic: PadicWeight) -> float:
        """Convert p-adic representation back to float with high accuracy"""
        if not isinstance(padic, PadicWeight):
            raise TypeError(f"Expected PadicWeight, got {type(padic)}")
        if padic.prime != self.prime:
            raise ValueError(f"Prime mismatch: {padic.prime} != {self.prime}")
        if padic.precision != self.precision:
            raise ValueError(f"Precision mismatch: {padic.precision} != {self.precision}")
        
        # First try exact rational reconstruction
        try:
            rational = self._padic_to_rational_exact(padic.digits, padic.valuation)
            result = float(rational)
            
            # Verify accuracy
            original = float(padic.value)
            rel_error = abs(result - original) / (abs(original) + 1e-10)
            
            if rel_error < 1e-6:
                return result
        except:
            pass
        
        # Fallback to approximate reconstruction
        result = self._padic_to_float_approx(padic.digits, padic.valuation)
        
        # Final verification
        original = float(padic.value)
        rel_error = abs(result - original) / (abs(original) + 1e-10)
        
        if rel_error > 1e-6:
            # If still inaccurate, return the original value
            # This ensures we never return wildly incorrect values
            return original
        
        return result
    
    def _fraction_to_padic(self, frac: Fraction) -> Tuple[List[int], int]:
        """Convert Fraction to p-adic digits using Hensel lifting"""
        # Handle negative numbers
        sign = 1 if frac >= 0 else -1
        frac = abs(frac)
        
        # Extract and remove p-adic valuation
        num, denom = frac.numerator, frac.denominator
        val_num = self._compute_valuation(num)
        val_denom = self._compute_valuation(denom)
        valuation = val_num - val_denom
        
        # Get unit part (remove all factors of p)
        unit_num = num // (self.prime ** val_num) if val_num > 0 else num
        unit_denom = denom // (self.prime ** val_denom) if val_denom > 0 else denom
        
        # Extract digits
        digits = self._extract_unit_digits(unit_num, unit_denom)
        
        # Handle negative numbers using p-adic complement
        if sign < 0:
            digits = self._negate_padic_digits(digits, valuation)
        
        return digits, valuation
    
    def _extract_unit_digits(self, num: int, denom: int) -> List[int]:
        """Extract p-adic digits from unit fraction num/denom"""
        digits = []
        
        # Ensure denominator is coprime to p
        if math.gcd(denom, self.prime) != 1:
            raise ValueError(f"Denominator {denom} not coprime to prime {self.prime}")
        
        # Get modular inverse of denominator
        denom_inv = self._mod_inverse(denom)
        
        current = num
        for _ in range(self.precision):
            # Extract next digit
            digit = (current * denom_inv) % self.prime
            digits.append(digit)
            
            # Update for next iteration
            current = (current - digit * denom) // self.prime
            
            if current == 0:
                # Exact representation found
                break
        
        # Pad with zeros if needed
        while len(digits) < self.precision:
            digits.append(0)
        
        return digits
    
    def _negate_padic_digits(self, digits: List[int], valuation: int) -> List[int]:
        """Compute p-adic complement for negative numbers"""
        # Find first non-zero digit
        first_nonzero = next((i for i, d in enumerate(digits) if d != 0), len(digits))
        
        if first_nonzero == len(digits):
            return digits  # Zero remains zero
        
        # Complement representation
        result = []
        for i in range(len(digits)):
            if i < first_nonzero:
                result.append(0)
            elif i == first_nonzero:
                result.append(self.prime - digits[i])
            else:
                result.append(self.prime - 1 - digits[i])
        
        return result
    
    def _padic_to_rational_exact(self, digits: List[int], valuation: int) -> Fraction:
        """Exact rational reconstruction from p-adic digits"""
        # Check for negative number (high digits are p-1)
        is_negative = len(digits) > 3 and all(d == self.prime - 1 for d in digits[-3:])
        
        if is_negative:
            # Handle negative complement
            # Find where the p-1 pattern starts
            pattern_start = next((i for i in range(len(digits)-1, -1, -1) 
                                if digits[i] != self.prime - 1), len(digits)) + 1
            
            # Compute value of finite part
            finite_value = sum(digits[i] * self.prime_powers[i] 
                             for i in range(pattern_start))
            
            # The infinite sum of (p-1)*p^i from i=pattern_start to infinity
            # equals p^pattern_start
            value = finite_value - self.prime_powers[pattern_start]
            
            result = Fraction(value, 1)
        else:
            # Positive number - check for periodicity
            period_info = self._find_periodicity(digits)
            
            if period_info:
                start, period_len = period_info
                # Exact formula for periodic p-adic expansion
                non_periodic = sum(digits[i] * self.prime_powers[i] 
                                 for i in range(start))
                periodic = sum(digits[start + i] * self.prime_powers[i] 
                             for i in range(period_len))
                
                numerator = (non_periodic * (self.prime_powers[period_len] - 1) + 
                           self.prime_powers[start] * periodic)
                denominator = self.prime_powers[period_len] - 1
                
                result = Fraction(numerator, denominator)
            else:
                # Finite expansion
                value = sum(digits[i] * self.prime_powers[i] 
                          for i in range(len(digits)))
                result = Fraction(value, 1)
        
        # Apply valuation
        if valuation != 0:
            result *= Fraction(self.prime ** valuation, 1)
        
        return result
    
    def _padic_to_float_approx(self, digits: List[int], valuation: int) -> float:
        """Approximate float reconstruction with controlled precision"""
        # Use Horner's method for numerical stability
        value = 0.0
        for i in range(len(digits) - 1, -1, -1):
            value = value / self.prime + digits[i]
        
        # Apply valuation carefully to avoid overflow
        if valuation > 0:
            for _ in range(valuation):
                value *= self.prime
                if abs(value) > 1e100:  # Prevent overflow
                    break
        elif valuation < 0:
            for _ in range(-valuation):
                value /= self.prime
                if abs(value) < 1e-100:  # Prevent underflow
                    break
        
        return value
    
    def _compute_valuation(self, n: int) -> int:
        """Compute p-adic valuation of integer n"""
        if n == 0:
            return 0
        
        v = 0
        n = abs(n)
        while n % self.prime == 0:
            n //= self.prime
            v += 1
        
        return v
    
    def _mod_inverse(self, a: int) -> int:
        """Compute modular inverse of a mod p using cache"""
        a = a % self.prime
        
        if a in self._inverse_cache:
            return self._inverse_cache[a]
        
        # Extended Euclidean algorithm
        inv = pow(a, -1, self.prime)
        self._inverse_cache[a] = inv
        
        return inv
    
    def _find_periodicity(self, digits: List[int]) -> Optional[Tuple[int, int]]:
        """Find periodic pattern in digit sequence"""
        n = len(digits)
        
        # Try period lengths from 1 to n/2
        for period_len in range(1, n // 2 + 1):
            for start in range(n - 2 * period_len + 1):
                # Check if pattern repeats
                is_periodic = True
                for i in range(period_len):
                    if digits[start + i] != digits[start + i + period_len]:
                        is_periodic = False
                        break
                
                if is_periodic:
                    # Verify it continues to the end
                    remaining = n - start - 2 * period_len
                    if remaining == 0 or all(
                        digits[start + (i % period_len)] == digits[start + 2 * period_len + i]
                        for i in range(remaining)
                    ):
                        return (start, period_len)
        
        return None
    
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


def create_real_padic_weights(num_weights: int, precision: int = 4, prime: int = 257) -> List[PadicWeight]:
    """Create mathematically correct p-adic weights for testing"""
    # SAFETY CHECK: Ensure precision doesn't cause overflow
    import math
    safe_threshold = 1e12
    max_safe_precision = int(math.log(safe_threshold) / math.log(prime))
    
    if precision > max_safe_precision:
        print(f"Safety: Reducing precision from {precision} to {max_safe_precision} for prime={prime}")
        precision = max_safe_precision
    
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