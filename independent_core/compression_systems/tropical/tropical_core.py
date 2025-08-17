"""
Tropical semiring operations for neural network compression.
Implements max-plus algebra: addition is max(x,y), multiplication is x+y.
NO PLACEHOLDERS - PRODUCTION READY
"""

import torch
import math
from typing import Union, List, Tuple, Optional, Any, Dict
from dataclasses import dataclass
import numpy as np


# Tropical zero constant - represents -∞ in tropical semiring
# Using -1e38 for GPU compatibility (avoids actual infinity)
TROPICAL_ZERO = -1e38
TROPICAL_EPSILON = 1e-10  # For numerical stability checks


@dataclass(frozen=True)
class TropicalNumber:
    """
    Immutable tropical number representation.
    In tropical semiring: addition is max, multiplication is standard addition.
    """
    value: float
    
    def __post_init__(self):
        """Validate tropical number on creation"""
        if not isinstance(self.value, (int, float)):
            raise TypeError(f"Tropical value must be numeric, got {type(self.value)}")
        if math.isnan(self.value):
            raise ValueError("Tropical value cannot be NaN")
        if math.isinf(self.value) and self.value > 0:
            raise ValueError("Tropical value cannot be positive infinity")
        # Clamp to TROPICAL_ZERO if too negative
        if self.value <= TROPICAL_ZERO:
            object.__setattr__(self, 'value', TROPICAL_ZERO)
    
    def is_zero(self) -> bool:
        """Check if this is tropical zero (-∞)"""
        return self.value <= TROPICAL_ZERO
    
    def __add__(self, other: 'TropicalNumber') -> 'TropicalNumber':
        """
        Tropical multiplication: a ⊗ b = a + b
        Using __add__ for ⊗ to maintain mathematical convention
        """
        if not isinstance(other, TropicalNumber):
            raise TypeError(f"Cannot multiply TropicalNumber with {type(other)}")
        
        # Handle tropical zero cases
        if self.is_zero() or other.is_zero():
            return TropicalNumber(TROPICAL_ZERO)
        
        result = self.value + other.value
        if result > 1e38:  # Overflow protection
            raise OverflowError(f"Tropical multiplication overflow: {self.value} + {other.value}")
        
        return TropicalNumber(result)
    
    def __or__(self, other: 'TropicalNumber') -> 'TropicalNumber':
        """
        Tropical addition: a ⊕ b = max(a, b)
        Using __or__ (|) for ⊕ operator
        """
        if not isinstance(other, TropicalNumber):
            raise TypeError(f"Cannot add TropicalNumber with {type(other)}")
        
        # max(-∞, x) = x
        if self.is_zero():
            return other
        if other.is_zero():
            return self
        
        return TropicalNumber(max(self.value, other.value))
    
    def __mul__(self, scalar: Union[int, float]) -> 'TropicalNumber':
        """
        Scalar multiplication in tropical semiring: n ⊗ a = n * a
        """
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Scalar must be numeric, got {type(scalar)}")
        if math.isnan(scalar):
            raise ValueError("Scalar cannot be NaN")
        if math.isinf(scalar):
            raise ValueError("Scalar cannot be infinity")
        
        if self.is_zero():
            return TropicalNumber(TROPICAL_ZERO)
        
        result = scalar * self.value
        if abs(result) > 1e38:
            raise OverflowError(f"Scalar multiplication overflow: {scalar} * {self.value}")
        
        return TropicalNumber(result)
    
    def __rmul__(self, scalar: Union[int, float]) -> 'TropicalNumber':
        """Right multiplication"""
        return self.__mul__(scalar)
    
    def __str__(self) -> str:
        """String representation for debugging"""
        if self.is_zero():
            return "T(-∞)"
        return f"T({self.value:.6f})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return f"TropicalNumber(value={self.value})"
    
    def __eq__(self, other: Any) -> bool:
        """Equality comparison"""
        if not isinstance(other, TropicalNumber):
            return False
        # Both are tropical zero
        if self.is_zero() and other.is_zero():
            return True
        # Close enough for numerical purposes
        return abs(self.value - other.value) < TROPICAL_EPSILON
    
    def __lt__(self, other: 'TropicalNumber') -> bool:
        """Less than comparison"""
        if not isinstance(other, TropicalNumber):
            raise TypeError(f"Cannot compare TropicalNumber with {type(other)}")
        return self.value < other.value
    
    def __le__(self, other: 'TropicalNumber') -> bool:
        """Less than or equal comparison"""
        return self == other or self < other
    
    def __gt__(self, other: 'TropicalNumber') -> bool:
        """Greater than comparison"""
        if not isinstance(other, TropicalNumber):
            raise TypeError(f"Cannot compare TropicalNumber with {type(other)}")
        return self.value > other.value
    
    def __ge__(self, other: 'TropicalNumber') -> bool:
        """Greater than or equal comparison"""
        return self == other or self > other
    
    def __hash__(self) -> int:
        """Hash for use in sets/dicts"""
        if self.is_zero():
            return hash(TROPICAL_ZERO)
        return hash(self.value)


class TropicalValidation:
    """Validation utilities for tropical operations - strict validation only"""
    
    @staticmethod
    def validate_tropical_value(value: Union[float, int]) -> None:
        """Validate a single tropical value"""
        if not isinstance(value, (int, float)):
            raise TypeError(f"Tropical value must be numeric, got {type(value)}")
        if math.isnan(value):
            raise ValueError("Tropical value cannot be NaN")
        if math.isinf(value) and value > 0:
            raise ValueError("Tropical value cannot be positive infinity")
        if value > 1e38:
            raise ValueError(f"Tropical value {value} exceeds maximum safe value")
    
    @staticmethod
    def validate_tropical_tensor(tensor: torch.Tensor) -> None:
        """Validate tensor for tropical operations"""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        if tensor.numel() == 0:
            raise ValueError("Empty tensor not allowed in tropical operations")
        if torch.isnan(tensor).any():
            raise ValueError("Tensor contains NaN values")
        if (tensor == float('inf')).any():
            raise ValueError("Tensor contains positive infinity values")
        if (tensor > 1e38).any():
            raise ValueError("Tensor contains values exceeding safe tropical range")
        if tensor.dtype not in [torch.float32, torch.float64]:
            raise TypeError(f"Tensor must be float type for tropical ops, got {tensor.dtype}")
    
    @staticmethod
    def validate_operation_args(a: Any, b: Any) -> None:
        """Validate arguments for binary tropical operation"""
        # For tensors
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            TropicalValidation.validate_tropical_tensor(a)
            TropicalValidation.validate_tropical_tensor(b)
            if a.shape != b.shape:
                raise ValueError(f"Tensor shape mismatch: {a.shape} vs {b.shape}")
        # For scalars
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            TropicalValidation.validate_tropical_value(a)
            TropicalValidation.validate_tropical_value(b)
        # For TropicalNumbers
        elif isinstance(a, TropicalNumber) and isinstance(b, TropicalNumber):
            pass  # Already validated in construction
        else:
            raise TypeError(f"Incompatible types for tropical operation: {type(a)} and {type(b)}")


class TropicalMathematicalOperations:
    """Core tropical mathematical operations - fail loud on all errors"""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize tropical operations with optional GPU device"""
        self.device = device or torch.device('cpu')
        if not isinstance(self.device, torch.device):
            raise TypeError(f"Device must be torch.device, got {type(self.device)}")
    
    def tropical_add(self, a: Union[float, torch.Tensor], 
                     b: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """
        Tropical addition: a ⊕ b = max(a, b)
        Works with scalars and tensors.
        """
        TropicalValidation.validate_operation_args(a, b)
        
        # Tensor operations
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            # Move to correct device if needed
            if a.device != self.device:
                a = a.to(self.device)
            if b.device != self.device:
                b = b.to(self.device)
            
            # Handle tropical zeros with masking
            a_is_zero = a <= TROPICAL_ZERO
            b_is_zero = b <= TROPICAL_ZERO
            
            # Compute max with proper zero handling
            result = torch.where(
                a_is_zero,
                b,
                torch.where(b_is_zero, a, torch.maximum(a, b))
            )
            return result
        
        # Scalar operations
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if a <= TROPICAL_ZERO:
                return b
            if b <= TROPICAL_ZERO:
                return a
            return max(a, b)
        
        else:
            raise TypeError(f"Unsupported types for tropical_add: {type(a)}, {type(b)}")
    
    def tropical_multiply(self, a: Union[float, torch.Tensor], 
                          b: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """
        Tropical multiplication: a ⊗ b = a + b
        Works with scalars and tensors.
        """
        TropicalValidation.validate_operation_args(a, b)
        
        # Tensor operations
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            # Move to correct device if needed
            if a.device != self.device:
                a = a.to(self.device)
            if b.device != self.device:
                b = b.to(self.device)
            
            # Handle tropical zeros
            a_is_zero = a <= TROPICAL_ZERO
            b_is_zero = b <= TROPICAL_ZERO
            either_zero = a_is_zero | b_is_zero
            
            # Compute sum with overflow protection
            result = torch.where(either_zero, torch.tensor(TROPICAL_ZERO, device=a.device, dtype=a.dtype), a + b)
            
            # Check for overflow
            if (result > 1e38).any():
                raise OverflowError("Tropical multiplication overflow detected")
            
            return result
        
        # Scalar operations
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if a <= TROPICAL_ZERO or b <= TROPICAL_ZERO:
                return TROPICAL_ZERO
            
            result = a + b
            if result > 1e38:
                raise OverflowError(f"Tropical multiplication overflow: {a} + {b}")
            
            return result
        
        else:
            raise TypeError(f"Unsupported types for tropical_multiply: {type(a)}, {type(b)}")
    
    def tropical_power(self, base: Union[float, torch.Tensor], 
                       exponent: Union[int, float]) -> Union[float, torch.Tensor]:
        """
        Tropical power: base ⊗^n = n * base (repeated tropical multiplication)
        """
        if not isinstance(exponent, (int, float)):
            raise TypeError(f"Exponent must be numeric, got {type(exponent)}")
        if math.isnan(exponent):
            raise ValueError("Exponent cannot be NaN")
        if math.isinf(exponent):
            raise ValueError("Exponent cannot be infinity")
        
        # Tensor operations
        if isinstance(base, torch.Tensor):
            TropicalValidation.validate_tropical_tensor(base)
            
            # Move to correct device if needed
            if base.device != self.device:
                base = base.to(self.device)
            
            # Handle tropical zeros
            base_is_zero = base <= TROPICAL_ZERO
            
            # Compute power
            result = torch.where(
                base_is_zero,
                torch.tensor(TROPICAL_ZERO, device=base.device, dtype=base.dtype),
                exponent * base
            )
            
            # Check for overflow
            if (torch.abs(result) > 1e38).any():
                raise OverflowError("Tropical power overflow detected")
            
            return result
        
        # Scalar operations
        elif isinstance(base, (int, float)):
            TropicalValidation.validate_tropical_value(base)
            
            if base <= TROPICAL_ZERO:
                return TROPICAL_ZERO
            
            result = exponent * base
            if abs(result) > 1e38:
                raise OverflowError(f"Tropical power overflow: {exponent} * {base}")
            
            return result
        
        else:
            raise TypeError(f"Unsupported base type for tropical_power: {type(base)}")
    
    def tropical_sum(self, values: Union[List[float], torch.Tensor]) -> Union[float, torch.Tensor]:
        """
        Tropical sum: ⊕_i values[i] = max(values)
        Computes the maximum over a list or tensor.
        """
        # Handle list of scalars
        if isinstance(values, list):
            if not values:
                raise ValueError("Cannot compute tropical sum of empty list")
            
            for i, val in enumerate(values):
                if not isinstance(val, (int, float)):
                    raise TypeError(f"Value at index {i} must be numeric, got {type(val)}")
                TropicalValidation.validate_tropical_value(val)
            
            # Filter out tropical zeros
            non_zero_values = [v for v in values if v > TROPICAL_ZERO]
            
            if not non_zero_values:
                return TROPICAL_ZERO
            
            return max(non_zero_values)
        
        # Handle tensor
        elif isinstance(values, torch.Tensor):
            TropicalValidation.validate_tropical_tensor(values)
            
            if values.numel() == 0:
                raise ValueError("Cannot compute tropical sum of empty tensor")
            
            # Move to correct device if needed
            if values.device != self.device:
                values = values.to(self.device)
            
            # Mask out tropical zeros
            non_zero_mask = values > TROPICAL_ZERO
            
            if not non_zero_mask.any():
                return torch.tensor(TROPICAL_ZERO, device=values.device, dtype=values.dtype)
            
            # Compute max only over non-zero values
            return values[non_zero_mask].max()
        
        else:
            raise TypeError(f"Values must be list or tensor, got {type(values)}")
    
    def tropical_product(self, values: Union[List[float], torch.Tensor]) -> Union[float, torch.Tensor]:
        """
        Tropical product: ⊗_i values[i] = sum(values)
        Computes the sum over a list or tensor.
        """
        # Handle list of scalars
        if isinstance(values, list):
            if not values:
                raise ValueError("Cannot compute tropical product of empty list")
            
            result = 0.0
            for i, val in enumerate(values):
                if not isinstance(val, (int, float)):
                    raise TypeError(f"Value at index {i} must be numeric, got {type(val)}")
                TropicalValidation.validate_tropical_value(val)
                
                if val <= TROPICAL_ZERO:
                    return TROPICAL_ZERO
                
                result += val
                if result > 1e38:
                    raise OverflowError(f"Tropical product overflow at index {i}")
            
            return result
        
        # Handle tensor
        elif isinstance(values, torch.Tensor):
            TropicalValidation.validate_tropical_tensor(values)
            
            if values.numel() == 0:
                raise ValueError("Cannot compute tropical product of empty tensor")
            
            # Move to correct device if needed
            if values.device != self.device:
                values = values.to(self.device)
            
            # Check for tropical zeros
            if (values <= TROPICAL_ZERO).any():
                return torch.tensor(TROPICAL_ZERO, device=self.device)
            
            # Compute sum
            result = values.sum()
            
            if result > 1e38:
                raise OverflowError("Tropical product overflow detected")
            
            return result
        
        else:
            raise TypeError(f"Values must be list or tensor, got {type(values)}")
    
    def tropical_matrix_multiply(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Tropical matrix multiplication: (A ⊗ B)_ij = max_k(A_ik + B_kj)
        """
        if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
            raise TypeError("Both arguments must be torch.Tensor")
        
        TropicalValidation.validate_tropical_tensor(A)
        TropicalValidation.validate_tropical_tensor(B)
        
        if A.dim() != 2 or B.dim() != 2:
            raise ValueError(f"Expected 2D matrices, got shapes {A.shape} and {B.shape}")
        
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Matrix dimensions incompatible: {A.shape} x {B.shape}")
        
        # Move to correct device
        if A.device != self.device:
            A = A.to(self.device)
        if B.device != self.device:
            B = B.to(self.device)
        
        m, k = A.shape
        k2, n = B.shape
        
        # Initialize result with tropical zeros
        result = torch.full((m, n), TROPICAL_ZERO, device=self.device)
        
        # Compute tropical matrix multiplication
        # (A ⊗ B)_ij = max_k(A_ik + B_kj)
        for i in range(m):
            for j in range(n):
                # Compute all products A_ik + B_kj for k
                products = A[i, :] + B[:, j]
                
                # Handle tropical zeros
                non_zero_mask = products > TROPICAL_ZERO
                
                if non_zero_mask.any():
                    result[i, j] = products[non_zero_mask].max()
        
        return result


# Utility functions

def is_tropical_zero(value: Union[float, torch.Tensor, TropicalNumber]) -> Union[bool, torch.Tensor]:
    """Check if value represents tropical zero (-∞)"""
    if isinstance(value, TropicalNumber):
        return value.is_zero()
    elif isinstance(value, torch.Tensor):
        return value <= TROPICAL_ZERO
    elif isinstance(value, (int, float)):
        return value <= TROPICAL_ZERO
    else:
        raise TypeError(f"Unsupported type for tropical zero check: {type(value)}")


def to_tropical_safe(value: Union[float, int, torch.Tensor]) -> Union[TropicalNumber, torch.Tensor]:
    """Convert value to tropical representation with validation"""
    if isinstance(value, TropicalNumber):
        return value
    elif isinstance(value, torch.Tensor):
        TropicalValidation.validate_tropical_tensor(value)
        # Clamp to tropical zero if needed
        return torch.where(value <= TROPICAL_ZERO, torch.tensor(TROPICAL_ZERO), value)
    elif isinstance(value, (int, float)):
        TropicalValidation.validate_tropical_value(value)
        return TropicalNumber(value)
    else:
        raise TypeError(f"Cannot convert {type(value)} to tropical representation")


def from_tropical_safe(value: Union[TropicalNumber, torch.Tensor]) -> Union[float, torch.Tensor]:
    """Convert from tropical representation with validation"""
    if isinstance(value, TropicalNumber):
        return value.value
    elif isinstance(value, torch.Tensor):
        return value
    elif isinstance(value, (int, float)):
        return value
    else:
        raise TypeError(f"Cannot convert {type(value)} from tropical representation")


def tropical_distance(a: Union[float, torch.Tensor], b: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
    """
    Compute tropical distance: d(a,b) = |a - b| in standard arithmetic
    This is a metric on the tropical semiring.
    """
    TropicalValidation.validate_operation_args(a, b)
    
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        # Both are tropical zero
        both_zero = (a <= TROPICAL_ZERO) & (b <= TROPICAL_ZERO)
        # One is tropical zero
        a_zero = a <= TROPICAL_ZERO
        b_zero = b <= TROPICAL_ZERO
        
        # Distance is 0 if both are zero, infinity if only one is zero
        return torch.where(
            both_zero,
            torch.tensor(0.0),
            torch.where(
                a_zero | b_zero,
                torch.tensor(float('inf')),
                torch.abs(a - b)
            )
        )
    else:
        # Scalar case
        if is_tropical_zero(a) and is_tropical_zero(b):
            return 0.0
        elif is_tropical_zero(a) or is_tropical_zero(b):
            return float('inf')
        else:
            return abs(a - b)


class TropicalGradientTracker:
    """
    Track gradients in tropical arithmetic for optimization.
    Uses subgradient methods since max is not differentiable.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize gradient tracker"""
        self.device = device or torch.device('cpu')
        self.ops = TropicalMathematicalOperations(device=self.device)
    
    def tropical_add_backward(self, grad_output: torch.Tensor, 
                             a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass for tropical addition (max operation).
        Returns subgradients for both inputs.
        """
        if not isinstance(grad_output, torch.Tensor):
            raise TypeError(f"grad_output must be tensor, got {type(grad_output)}")
        
        TropicalValidation.validate_tropical_tensor(a)
        TropicalValidation.validate_tropical_tensor(b)
        
        # Move to device
        if a.device != self.device:
            a = a.to(self.device)
        if b.device != self.device:
            b = b.to(self.device)
        if grad_output.device != self.device:
            grad_output = grad_output.to(self.device)
        
        # Subgradient of max: 1 where input equals max, 0 elsewhere
        # Handle ties by splitting gradient
        max_val = self.ops.tropical_add(a, b)
        
        a_is_max = torch.abs(a - max_val) < TROPICAL_EPSILON
        b_is_max = torch.abs(b - max_val) < TROPICAL_EPSILON
        
        # Handle tie-breaking
        both_max = a_is_max & b_is_max
        
        grad_a = torch.where(
            both_max,
            0.5 * grad_output,
            torch.where(a_is_max, grad_output, torch.zeros_like(grad_output))
        )
        
        grad_b = torch.where(
            both_max,
            0.5 * grad_output,
            torch.where(b_is_max, grad_output, torch.zeros_like(grad_output))
        )
        
        return grad_a, grad_b
    
    def tropical_multiply_backward(self, grad_output: torch.Tensor,
                                  a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass for tropical multiplication (addition).
        Returns gradients for both inputs.
        """
        if not isinstance(grad_output, torch.Tensor):
            raise TypeError(f"grad_output must be tensor, got {type(grad_output)}")
        
        TropicalValidation.validate_tropical_tensor(a)
        TropicalValidation.validate_tropical_tensor(b)
        
        # Move to device
        if grad_output.device != self.device:
            grad_output = grad_output.to(self.device)
        
        # Gradient of addition is straightforward
        # But need to handle tropical zeros
        a_is_zero = a <= TROPICAL_ZERO
        b_is_zero = b <= TROPICAL_ZERO
        
        grad_a = torch.where(a_is_zero | b_is_zero, torch.zeros_like(grad_output), grad_output)
        grad_b = torch.where(a_is_zero | b_is_zero, torch.zeros_like(grad_output), grad_output)
        
        return grad_a, grad_b