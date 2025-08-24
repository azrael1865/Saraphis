"""
P-adic Mathematical Operations
Core mathematical functions for p-adic number operations
"""

import torch
import math
from typing import Union, Tuple, Optional

def padic_norm(x: Union[torch.Tensor, float], p: int) -> Union[torch.Tensor, float]:
    """
    Compute the p-adic norm of a number.
    
    The p-adic norm of x is p^(-v_p(x)) where v_p(x) is the p-adic valuation.
    For x = 0, the norm is 0.
    For x != 0, find the highest power of p that divides x.
    
    Args:
        x: Input number (tensor or scalar)
        p: Prime number for p-adic system
        
    Returns:
        P-adic norm of x
    """
    if isinstance(x, torch.Tensor):
        result = torch.zeros_like(x, dtype=torch.float32)
        
        # Handle zero values
        zero_mask = (x == 0)
        result[zero_mask] = 0.0
        
        # For non-zero values
        non_zero_mask = ~zero_mask
        if non_zero_mask.any():
            x_nonzero = x[non_zero_mask].abs()
            
            # Compute p-adic valuation
            valuation = torch.zeros_like(x_nonzero)
            temp = x_nonzero.clone()
            
            # Count how many times p divides x
            while True:
                divisible = (temp % p == 0)
                if not divisible.any():
                    break
                valuation[divisible] += 1
                temp[divisible] = temp[divisible] / p
            
            # p-adic norm is p^(-valuation)
            result[non_zero_mask] = p ** (-valuation.float())
        
        return result
    else:
        if x == 0:
            return 0.0
        
        x_abs = abs(x)
        valuation = 0
        
        # Count how many times p divides x
        while x_abs % p == 0:
            valuation += 1
            x_abs = x_abs // p
        
        return float(p ** (-valuation))

def padic_distance(x: Union[torch.Tensor, float], y: Union[torch.Tensor, float], p: int) -> Union[torch.Tensor, float]:
    """
    Compute the p-adic distance between two numbers.
    
    The p-adic distance is the p-adic norm of the difference: d_p(x,y) = |x-y|_p
    
    Args:
        x: First number (tensor or scalar)
        y: Second number (tensor or scalar)
        p: Prime number for p-adic system
        
    Returns:
        P-adic distance between x and y
    """
    return padic_norm(x - y, p)

def hensel_lift(f: callable, df: callable, x0: float, p: int, 
                precision: int = 10, modulus: Optional[int] = None) -> Tuple[float, int]:
    """
    Hensel's lifting lemma for finding roots modulo p^n.
    
    Given a polynomial f and an approximate root x0 modulo p,
    lift it to a root modulo p^n using Newton's method in p-adic numbers.
    
    Args:
        f: Function representing the polynomial
        df: Derivative of f
        x0: Initial approximation (root modulo p)
        p: Prime number
        precision: Target precision (find root modulo p^precision)
        modulus: Current modulus (default: p)
        
    Returns:
        Tuple of (lifted_root, final_modulus)
    """
    if modulus is None:
        modulus = p
    
    x = x0
    current_mod = modulus
    
    for i in range(precision):
        # Evaluate f(x) and f'(x)
        fx = f(x) % current_mod
        dfx = df(x) % current_mod
        
        # Check if derivative is invertible modulo p
        if dfx % p == 0:
            raise ValueError(f"Derivative is not invertible modulo {p}")
        
        # Find inverse of df(x) modulo current_mod
        # Using extended Euclidean algorithm
        dfx_inv = pow(int(dfx), -1, current_mod)
        
        # Newton step: x = x - f(x) * (f'(x))^(-1)
        x = (x - fx * dfx_inv) % current_mod
        
        # Increase precision
        if i < precision - 1:
            current_mod *= p
    
    return x, current_mod

def padic_valuation(n: int, p: int) -> int:
    """
    Compute the p-adic valuation of an integer n.
    
    The p-adic valuation v_p(n) is the highest power of p that divides n.
    
    Args:
        n: Integer to compute valuation for
        p: Prime number
        
    Returns:
        P-adic valuation of n
    """
    if n == 0:
        return float('inf')
    
    valuation = 0
    n = abs(n)
    
    while n % p == 0:
        valuation += 1
        n = n // p
    
    return valuation

def padic_expansion(n: int, p: int, precision: int = 10) -> list:
    """
    Compute the p-adic expansion of an integer.
    
    Represents n as sum of a_i * p^i where 0 <= a_i < p.
    
    Args:
        n: Integer to expand
        p: Prime base
        precision: Number of digits to compute
        
    Returns:
        List of p-adic digits [a_0, a_1, ..., a_{precision-1}]
    """
    if n == 0:
        return [0] * precision
    
    digits = []
    remainder = abs(n)
    
    for _ in range(precision):
        digit = remainder % p
        digits.append(digit)
        remainder = remainder // p
        
        if remainder == 0:
            # Pad with zeros if needed
            while len(digits) < precision:
                digits.append(0)
            break
    
    # Handle negative numbers
    if n < 0:
        # In p-adic representation, use complement
        carry = 1
        for i in range(len(digits)):
            digits[i] = (p - 1 - digits[i] + carry) % p
            carry = (p - 1 - digits[i] + carry) // p
    
    return digits

__all__ = [
    'padic_norm',
    'padic_distance', 
    'hensel_lift',
    'padic_valuation',
    'padic_expansion'
]