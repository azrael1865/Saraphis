#!/usr/bin/env python3
"""
Test script for P-adic ↔ Tropical Bridge
"""

import sys
import os
import torch
from fractions import Fraction

# Add parent directories to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Direct imports to avoid __init__ issues
from padic.padic_encoder import PadicWeight, PadicValidation, PadicMathematicalOperations
from tropical.tropical_core import (
    TropicalNumber, TropicalMathematicalOperations, TropicalValidation,
    TROPICAL_ZERO, TROPICAL_EPSILON
)
from tropical.tropical_polynomial import TropicalPolynomial, TropicalMonomial

# Import our bridge module
from padic_tropical_bridge import (
    ConversionConfig, PadicTropicalConverter, HybridRepresentation,
    ConversionValidator
)


def test_basic_conversion():
    """Test basic p-adic to tropical conversion"""
    print("Testing basic P-adic ↔ Tropical conversion...")
    
    config = ConversionConfig(prime=7, precision=8)
    converter = PadicTropicalConverter(config)
    
    # Create a p-adic weight
    padic_weight = PadicWeight(
        value=Fraction(5, 7),
        prime=7,
        precision=8,
        valuation=-1,
        digits=[5, 0, 0, 0, 0, 0, 0, 0]
    )
    
    # Convert to tropical
    tropical = converter.padic_to_tropical(padic_weight)
    assert isinstance(tropical, TropicalNumber)
    assert not tropical.is_zero()
    print(f"  P-adic weight with valuation {padic_weight.valuation} → Tropical value {tropical.value:.6f}")
    
    # Convert back to p-adic
    padic_recovered = converter.tropical_to_padic(tropical)
    assert isinstance(padic_recovered, PadicWeight)
    assert padic_recovered.prime == 7
    assert padic_recovered.precision == 8
    print(f"  Tropical {tropical.value:.6f} → P-adic with valuation {padic_recovered.valuation}")
    
    print("✓ Basic conversion test passed")
    return True


def test_tensor_conversion():
    """Test tensor conversion between representations"""
    print("\nTesting tensor conversion...")
    
    config = ConversionConfig(prime=251, precision=16, use_gpu=False)
    converter = PadicTropicalConverter(config)
    
    # Create test tensor
    test_tensor = torch.randn(4, 4) * 10
    print(f"  Original tensor shape: {test_tensor.shape}, mean: {test_tensor.mean():.3f}")
    
    # Create p-adic representation
    padic_ops = PadicMathematicalOperations(prime=251, precision=16)
    padic_weights = []
    for value in test_tensor.flatten().numpy():
        padic_weights.append(padic_ops.to_padic(float(value)))
    print(f"  Created {len(padic_weights)} p-adic weights")
    
    # Convert to tropical tensor
    tropical_tensor = converter.tensor_padic_to_tropical(padic_weights, test_tensor.shape)
    assert tropical_tensor.shape == test_tensor.shape
    print(f"  Tropical tensor shape: {tropical_tensor.shape}, mean: {tropical_tensor.mean():.3f}")
    
    # Convert back to p-adic
    padic_recovered = converter.tensor_tropical_to_padic(tropical_tensor)
    assert len(padic_recovered) == test_tensor.numel()
    print(f"  Recovered {len(padic_recovered)} p-adic weights")
    
    print("✓ Tensor conversion test passed")
    return True


def test_validation():
    """Test conversion validation"""
    print("\nTesting conversion validation...")
    
    validator = ConversionValidator(tolerance=1e-3)
    
    # Create test tensors
    original = torch.randn(5, 5)
    
    # Test norm preservation
    converted_good = original + torch.randn_like(original) * 0.0001
    norm_preserved = validator.validate_norm_preservation(original, converted_good)
    print(f"  Norm preservation (small change): {norm_preserved}")
    assert norm_preserved
    
    converted_bad = original * 2
    norm_not_preserved = not validator.validate_norm_preservation(original, converted_bad)
    print(f"  Norm not preserved (2x scaling): {norm_not_preserved}")
    assert norm_not_preserved
    
    # Test conversion loss
    loss = validator.compute_conversion_loss(original, converted_good)
    print(f"  Conversion loss (small change): {loss:.6f}")
    assert loss < 0.01
    
    print("✓ Validation test passed")
    return True


def test_hybrid_representation():
    """Test hybrid representation functionality"""
    print("\nTesting hybrid representation...")
    
    # Create test tensor with specific characteristics
    test_tensor = torch.randn(8, 8) * 5
    test_tensor[test_tensor.abs() < 2] = 0  # Add sparsity
    
    print(f"  Test tensor: shape {test_tensor.shape}, sparsity {(test_tensor == 0).float().mean():.2f}")
    
    # Create hybrid representation
    hybrid = HybridRepresentation(test_tensor)
    
    # Compute p-adic representation
    padic_ops = PadicMathematicalOperations(prime=251, precision=16)
    hybrid.compute_padic(padic_ops)
    assert hybrid.padic_components is not None
    print(f"  P-adic: {len(hybrid.padic_components)} components, size ~{hybrid.padic_size} bytes")
    
    # Compute tropical representation  
    tropical_ops = TropicalMathematicalOperations(device=torch.device('cpu'))
    hybrid.compute_tropical(tropical_ops)
    assert hybrid.tropical_components is not None
    print(f"  Tropical: shape {hybrid.tropical_components.shape}, size {hybrid.tropical_size} bytes")
    
    # Get optimal representation
    optimal = hybrid.get_optimal_representation()
    assert optimal in ["padic", "tropical", "hybrid"]
    print(f"  Optimal representation: {optimal}")
    
    # Get compression ratio
    if optimal == "tropical":
        hybrid.active_mode = "tropical"
        ratio = hybrid.get_compression_ratio()
        print(f"  Compression ratio: {ratio:.2f}x")
    
    print("✓ Hybrid representation test passed")
    return True


def test_cache_performance():
    """Test caching mechanism"""
    print("\nTesting cache performance...")
    
    config = ConversionConfig(prime=7, precision=8)
    converter = PadicTropicalConverter(config)
    
    # Create test p-adic weight
    padic_weight = PadicWeight(
        value=Fraction(3, 7),
        prime=7,
        precision=8,
        valuation=0,
        digits=[3, 0, 0, 0, 0, 0, 0, 0]
    )
    
    # First conversion (cache miss)
    _ = converter.padic_to_tropical(padic_weight)
    print(f"  After first conversion: hits={converter._cache_hits}, misses={converter._cache_misses}")
    assert converter._cache_misses == 1
    assert converter._cache_hits == 0
    
    # Second conversion (cache hit)
    _ = converter.padic_to_tropical(padic_weight)
    print(f"  After second conversion: hits={converter._cache_hits}, misses={converter._cache_misses}")
    assert converter._cache_misses == 1
    assert converter._cache_hits == 1
    
    # Get cache stats
    stats = converter.get_cache_stats()
    print(f"  Cache stats: hit_rate={stats['hit_rate']:.2f}, size={stats['cache_size']}")
    assert stats['hit_rate'] == 0.5
    
    print("✓ Cache performance test passed")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("P-ADIC ↔ TROPICAL BRIDGE TESTS")
    print("=" * 60)
    
    tests = [
        test_basic_conversion,
        test_tensor_conversion,
        test_validation,
        test_hybrid_representation,
        test_cache_performance
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test_func.__name__} returned False")
        except Exception as e:
            print(f"✗ {test_func.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)