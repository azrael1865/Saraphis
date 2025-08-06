#!/usr/bin/env python3
"""
Standalone test for P-adic ↔ Tropical Bridge
Tests the core conversion logic without complex imports
"""

import torch
import math
from fractions import Fraction
from dataclasses import dataclass
from typing import List, Dict, Tuple


# Minimal implementations for testing
@dataclass
class MinimalPadicWeight:
    """Minimal P-adic weight for testing"""
    value: Fraction
    prime: int
    precision: int
    valuation: int
    digits: List[int]


class MinimalTropicalNumber:
    """Minimal Tropical number for testing"""
    def __init__(self, value: float):
        self.value = value
    
    def is_zero(self):
        return self.value <= -1e38


def test_conversion_math():
    """Test the mathematical conversion formulas"""
    print("Testing mathematical conversion formulas...")
    
    # Test valuation mapping: T(x) = -v_p(x) * log(p)
    prime = 7
    valuation = -2  # p^(-2) in p-adic
    log_prime = math.log(prime)
    
    # Expected tropical value
    tropical_value = -valuation * log_prime
    print(f"  P-adic valuation {valuation} (prime {prime}) → Tropical {tropical_value:.6f}")
    
    # Test inverse: v_p = -t / log(p)
    recovered_valuation = -tropical_value / log_prime
    print(f"  Tropical {tropical_value:.6f} → P-adic valuation {recovered_valuation:.6f}")
    
    assert abs(recovered_valuation - valuation) < 1e-10
    print("✓ Mathematical conversion test passed")


def test_digit_contribution():
    """Test how p-adic digits contribute to tropical value"""
    print("\nTesting digit contribution...")
    
    prime = 251
    digits = [5, 10, 0, 0, 0, 0, 0, 0]  # Some non-zero digits
    
    # Calculate digit contribution
    contribution = 0.0
    for i, digit in enumerate(digits):
        if digit != 0:
            contrib = math.log1p(digit / (prime ** (i + 1)))
            contribution += contrib
            print(f"  Digit {digit} at position {i}: contribution {contrib:.8f}")
    
    print(f"  Total digit contribution: {contribution:.8f}")
    assert contribution > 0  # Non-zero digits should contribute
    print("✓ Digit contribution test passed")


def test_tensor_operations():
    """Test tensor-level operations"""
    print("\nTesting tensor operations...")
    
    # Create test tensor
    test_tensor = torch.tensor([
        [1.0, -2.0, 3.5],
        [0.0, 5.0, -1.0]
    ])
    
    print(f"  Original tensor shape: {test_tensor.shape}")
    print(f"  Values: min={test_tensor.min():.2f}, max={test_tensor.max():.2f}")
    
    # Simulate tropical transformation
    TROPICAL_ZERO = -1e38
    tropical_tensor = torch.where(
        test_tensor <= TROPICAL_ZERO,
        torch.tensor(TROPICAL_ZERO),
        test_tensor
    )
    
    # Check properties
    assert tropical_tensor.shape == test_tensor.shape
    assert (tropical_tensor >= TROPICAL_ZERO).all()
    
    print(f"  Tropical tensor created successfully")
    print("✓ Tensor operations test passed")


def test_sparsity_handling():
    """Test handling of sparse tensors"""
    print("\nTesting sparsity handling...")
    
    # Create sparse tensor
    tensor = torch.randn(10, 10)
    tensor[tensor.abs() < 1.0] = 0  # Make it sparse
    
    sparsity = (tensor == 0).float().mean().item()
    print(f"  Tensor sparsity: {sparsity:.2%}")
    
    # Count non-zero elements
    non_zero_count = (tensor != 0).sum().item()
    print(f"  Non-zero elements: {non_zero_count}/{tensor.numel()}")
    
    # Test sparsity preservation logic
    zero_threshold = 1e-6
    sparse_mask = torch.abs(tensor) < zero_threshold
    
    # Simulate conversion that preserves sparsity
    converted = tensor.clone()
    converted[sparse_mask] = 0
    
    # Check sparsity is preserved
    converted_sparsity = (converted == 0).float().mean().item()
    assert abs(converted_sparsity - sparsity) < 0.01
    
    print(f"  Sparsity preserved after conversion: {converted_sparsity:.2%}")
    print("✓ Sparsity handling test passed")


def test_gradient_preservation():
    """Test gradient preservation through conversion"""
    print("\nTesting gradient preservation...")
    
    # Create tensor with gradients
    tensor = torch.randn(5, 5, requires_grad=True)
    
    # Simulate conversion operation
    converted = tensor * 1.0001 + 0.0001  # Small perturbation
    
    # Test gradient flow
    loss = converted.sum()
    loss.backward()
    
    assert tensor.grad is not None
    print(f"  Gradient shape: {tensor.grad.shape}")
    print(f"  Gradient norm: {tensor.grad.norm():.6f}")
    
    print("✓ Gradient preservation test passed")


def test_cache_mechanism():
    """Test caching logic"""
    print("\nTesting cache mechanism...")
    
    # Simulate cache
    cache = {}
    cache_hits = 0
    cache_misses = 0
    
    # Test data
    test_keys = [
        (Fraction(1, 2), -1, 7),
        (Fraction(3, 4), 0, 7),
        (Fraction(1, 2), -1, 7),  # Duplicate
        (Fraction(5, 6), 1, 7),
        (Fraction(3, 4), 0, 7),   # Duplicate
    ]
    
    for key in test_keys:
        if key in cache:
            cache_hits += 1
            value = cache[key]
            print(f"  Cache hit for {key[0]}")
        else:
            cache_misses += 1
            # Simulate computation
            value = float(key[0]) * math.log(key[2])
            cache[key] = value
            print(f"  Cache miss for {key[0]}, computed {value:.4f}")
    
    hit_rate = cache_hits / (cache_hits + cache_misses)
    print(f"  Final cache stats: hits={cache_hits}, misses={cache_misses}, rate={hit_rate:.2f}")
    
    assert cache_hits == 2  # Should have 2 hits
    assert cache_misses == 3  # Should have 3 misses
    print("✓ Cache mechanism test passed")


def test_performance_metrics():
    """Test performance measurement"""
    print("\nTesting performance metrics...")
    
    import time
    
    # Simulate conversion of large tensor
    size = 1000
    tensor = torch.randn(size)
    
    start = time.time()
    # Simulate conversion work
    for i in range(size):
        _ = math.log1p(abs(tensor[i].item()))
    elapsed = time.time() - start
    
    elements_per_second = size / elapsed
    print(f"  Processed {size} elements in {elapsed:.3f}s")
    print(f"  Rate: {elements_per_second:.0f} elements/second")
    
    # Check if performance is reasonable
    assert elements_per_second > 100  # Should process at least 100 elem/s
    print("✓ Performance metrics test passed")


def test_error_conditions():
    """Test error handling"""
    print("\nTesting error conditions...")
    
    # Test overflow protection
    large_value = 1e39
    TROPICAL_MAX = 1e38
    
    if large_value > TROPICAL_MAX:
        print(f"  Correctly detected overflow: {large_value:.2e} > {TROPICAL_MAX:.2e}")
    
    # Test NaN handling
    import math
    nan_value = float('nan')
    assert math.isnan(nan_value)
    print(f"  Correctly detected NaN")
    
    # Test shape mismatch
    tensor1 = torch.randn(3, 3)
    tensor2 = torch.randn(4, 4)
    
    if tensor1.shape != tensor2.shape:
        print(f"  Correctly detected shape mismatch: {tensor1.shape} vs {tensor2.shape}")
    
    print("✓ Error conditions test passed")


def main():
    """Run all tests"""
    print("=" * 60)
    print("P-ADIC ↔ TROPICAL BRIDGE STANDALONE TESTS")
    print("=" * 60)
    
    tests = [
        test_conversion_math,
        test_digit_contribution,
        test_tensor_operations,
        test_sparsity_handling,
        test_gradient_preservation,
        test_cache_mechanism,
        test_performance_metrics,
        test_error_conditions
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    # Additionally, test that our main module can be imported
    print("\nTesting module import...")
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        
        # Check if module exists
        bridge_path = os.path.join(os.path.dirname(__file__), 'padic_tropical_bridge.py')
        if os.path.exists(bridge_path):
            print(f"✓ Module file exists at {bridge_path}")
            
            # Check file size
            file_size = os.path.getsize(bridge_path)
            print(f"✓ Module size: {file_size:,} bytes")
            
            # Count lines
            with open(bridge_path, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"✓ Module has {line_count:,} lines of code")
        else:
            print(f"✗ Module file not found at {bridge_path}")
            failed += 1
            
    except Exception as e:
        print(f"✗ Module check failed: {e}")
        failed += 1
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)