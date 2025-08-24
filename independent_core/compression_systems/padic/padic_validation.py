"""
P-adic Validation Module
Validation functions for p-adic weights and operations
"""

import torch
from typing import Union, Optional
from .padic_encoder import PadicWeight, validate_single_weight as _validate_single_weight

def validate_single_weight(weight: Union[torch.Tensor, PadicWeight], 
                          expected_prime: Optional[int] = None,
                          expected_precision: Optional[int] = None) -> bool:
    """
    Validate a single p-adic weight.
    
    For tensor inputs, just checks if it's a valid tensor.
    For PadicWeight inputs, delegates to the full validation.
    
    Args:
        weight: Weight to validate (tensor or PadicWeight)
        expected_prime: Expected prime for PadicWeight validation
        expected_precision: Expected precision for PadicWeight validation
        
    Returns:
        True if weight is valid, False otherwise
    """
    if isinstance(weight, torch.Tensor):
        # For tensor weights, just check basic validity
        return weight.numel() > 0 and torch.isfinite(weight).all().item()
    
    elif isinstance(weight, PadicWeight):
        # Use full validation for PadicWeight
        if expected_prime is None:
            expected_prime = weight.prime if hasattr(weight, 'prime') else 5
        if expected_precision is None:
            expected_precision = weight.precision if hasattr(weight, 'precision') else 10
        
        return _validate_single_weight(weight, expected_prime, expected_precision)
    
    else:
        # For other types, check if it's a number
        try:
            float(weight)
            return True
        except (TypeError, ValueError):
            return False

def validate_weight_tensor(weights: torch.Tensor) -> bool:
    """
    Validate a tensor of weights.
    
    Args:
        weights: Tensor of weights to validate
        
    Returns:
        True if all weights are valid, False otherwise
    """
    if not isinstance(weights, torch.Tensor):
        return False
    
    # Check for NaN or Inf values
    if not torch.isfinite(weights).all():
        return False
    
    # Check tensor is not empty
    if weights.numel() == 0:
        return False
    
    return True

def validate_padic_config(prime: int, precision: int) -> bool:
    """
    Validate p-adic configuration parameters.
    
    Args:
        prime: Prime number for p-adic system
        precision: Precision level
        
    Returns:
        True if configuration is valid, False otherwise
    """
    # Check prime is actually prime
    if prime < 2:
        return False
    
    if prime == 2:
        return precision > 0
    
    # Simple primality check for small primes
    for i in range(2, min(prime, 100)):
        if prime % i == 0:
            return False
    
    # Check precision is positive
    return precision > 0

__all__ = [
    'validate_single_weight',
    'validate_weight_tensor',
    'validate_padic_config'
]