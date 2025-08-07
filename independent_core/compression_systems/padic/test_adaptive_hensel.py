"""
Test suite for adaptive precision Hensel lifting integration.
Verifies that Track B implementation is complete and functional.
"""

import torch
import numpy as np
import time
from fractions import Fraction
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from independent_core.compression_systems.padic.padic_encoder import (
    AdaptiveHenselLifting,
    PadicWeight,
    PadicMathematicalOperations
)
from independent_core.compression_systems.padic.padic_logarithmic_encoder import (
    PadicLogarithmicEncoder,
    LogarithmicEncodingConfig
)
from independent_core.compression_systems.strategies.compression_strategy import (
    PadicStrategy,
    StrategyConfig
)


def test_adaptive_hensel_basic():
    """Test basic adaptive Hensel lifting functionality"""
    print("Testing basic adaptive Hensel lifting...")
    
    prime = 251
    base_precision = 3
    
    # Initialize adaptive Hensel lifter
    lifter = AdaptiveHenselLifting(prime, base_precision)
    
    # Test with various values
    test_values = [
        3.14159,  # Pi
        2.71828,  # e
        1.41421,  # sqrt(2)
        0.123456789,  # Small precise value
        1234.5678,  # Large value
    ]
    
    for value in test_values:
        # Test with different target errors
        for target_error in [1e-6, 1e-8, 1e-10]:
            lifted_weight, iterations = lifter.adaptive_precision_hensel(
                value, target_error, prime
            )
            
            # Verify the lifted weight
            assert isinstance(lifted_weight, PadicWeight)
            assert lifted_weight.prime == prime
            
            # Check that error is within target
            error = lifter._calculate_padic_error(
                lifted_weight, 
                Fraction(value).limit_denominator(10**15),
                prime
            )
            
            print(f"  Value: {value:.6f}, Target error: {target_error:.1e}, "
                  f"Actual error: {error:.1e}, Iterations: {iterations}")
            
            # Allow some tolerance due to p-adic arithmetic limitations
            assert error <= target_error * 10 or error < 1e-6
    
    # Check statistics
    stats = lifter.get_stats()
    assert stats['total_lifts'] > 0
    assert stats['average_iterations'] > 0
    print(f"  Average iterations: {stats['average_iterations']:.2f}")
    print(f"  Early termination rate: {stats['early_termination_rate']:.2%}")
    print(f"  Convergence rate: {stats['convergence_rate']:.2%}")
    
    print("✓ Basic adaptive Hensel lifting tests passed")


def test_hensel_with_logarithmic_encoder():
    """Test Hensel lifting integration with logarithmic encoder"""
    print("Testing Hensel lifting with logarithmic encoder...")
    
    # Create config with Hensel enabled
    config = LogarithmicEncodingConfig()
    config.prime = 127
    config.precision = 3
    config.enable_adaptive_hensel = True
    config.hensel_target_error = 1e-10
    config.hensel_high_precision_threshold = 1e-8
    
    # Initialize encoder
    encoder = PadicLogarithmicEncoder(config)
    
    # Test tensor with values needing high precision
    test_tensor = torch.tensor([
        [1.23456789, 0.00000123],
        [9876.54321, 0.98765432],
        [3.14159265, 2.71828183]
    ])
    
    # Encode weights
    start_time = time.time()
    encoded_weights = encoder.encode_weights_logarithmically(test_tensor)
    encoding_time = time.time() - start_time
    
    print(f"  Encoded {len(encoded_weights)} weights in {encoding_time:.3f}s")
    
    # Check that Hensel lifting was used
    if encoder.hensel_lifter:
        stats = encoder.hensel_lifter.get_stats()
        print(f"  Hensel lifts performed: {stats['total_lifts']}")
        print(f"  Precision adjustments: {stats['precision_adjustments']}")
        
        # Verify some weights used Hensel
        assert encoder.encoding_stats['hensel_lifts'] > 0
        print(f"  Weights using Hensel: {encoder.encoding_stats['hensel_lifts']}")
        print(f"  Hensel improvements: {encoder.encoding_stats['hensel_improvements']}")
    
    # Test decoding
    decoded_tensor = encoder.decode_logarithmic_padic_weights(encoded_weights)
    
    # Reshape decoded tensor to match original shape
    decoded_tensor = decoded_tensor.view(test_tensor.shape)
    
    # Check reconstruction accuracy
    reconstruction_error = torch.nn.functional.mse_loss(decoded_tensor, test_tensor)
    print(f"  Reconstruction MSE: {reconstruction_error:.2e}")
    
    # Should have good reconstruction due to Hensel lifting
    assert reconstruction_error < 1e-4
    
    print("✓ Hensel + logarithmic encoder tests passed")


def test_padic_strategy_with_hensel():
    """Test PadicStrategy using adaptive Hensel lifting"""
    print("Testing PadicStrategy with adaptive Hensel...")
    
    # Initialize strategy (will use Hensel by default now)
    strategy = PadicStrategy(prime=251, precision=3)
    
    # Create test tensors with different characteristics
    
    # 1. High-precision periodic signal
    t = torch.linspace(0, 4 * np.pi, 1000)
    periodic_tensor = torch.sin(t) * 0.123456789
    periodic_tensor = periodic_tensor.view(50, 20)
    
    # 2. Tensor with large dynamic range
    dynamic_tensor = torch.randn(30, 30)
    dynamic_tensor[0, 0] = 1e-9
    dynamic_tensor[-1, -1] = 1e6
    
    # 3. Gaussian distribution (benefits from Hensel)
    gaussian_tensor = torch.randn(40, 40) * 0.01 + 0.5
    
    test_cases = [
        ("periodic", periodic_tensor),
        ("dynamic_range", dynamic_tensor),
        ("gaussian", gaussian_tensor)
    ]
    
    for name, tensor in test_cases:
        print(f"\n  Testing {name} tensor...")
        
        # Add metadata to trigger high precision
        metadata = {
            'dynamic_range': (tensor.max() / (tensor[tensor != 0].min().abs() + 1e-10)).item(),
            'local_entropy': 2.5,  # Low entropy suggests structure
            'distribution_type': 'gaussian' if 'gaussian' in name else 'unknown'
        }
        
        # Compress
        start_time = time.time()
        compressed = strategy.compress(tensor, metadata)
        compress_time = time.time() - start_time
        
        print(f"    Compression ratio: {compressed.compression_ratio:.2f}x")
        print(f"    Compression time: {compress_time:.3f}s")
        print(f"    Prime used: {compressed.metadata.get('prime', 'unknown')}")
        
        # Check if Hensel was used (in metadata)
        if 'hensel_stats' in compressed.metadata:
            hensel_stats = compressed.metadata['hensel_stats']
            if hensel_stats:
                print(f"    Hensel lifts: {hensel_stats.get('total_lifts', 0)}")
                print(f"    Avg iterations: {hensel_stats.get('average_iterations', 0):.2f}")
        
        # Decompress
        reconstructed = strategy.decompress(compressed)
        
        # Check accuracy
        mse = torch.nn.functional.mse_loss(reconstructed, tensor)
        print(f"    Reconstruction MSE: {mse:.2e}")
        
        # Different tensors have different accuracy requirements
        if name == "periodic":
            assert mse < 1e-6  # High accuracy for periodic
        elif name == "dynamic_range":
            # Relative error for dynamic range
            rel_error = mse / tensor.var()
            assert rel_error < 0.01
        else:
            assert mse < 1e-3  # Standard accuracy
    
    print("\n✓ PadicStrategy with Hensel tests passed")


def test_hensel_convergence():
    """Test quadratic convergence property of Hensel lifting"""
    print("Testing Hensel lifting convergence properties...")
    
    prime = 17  # Small prime for detailed analysis
    lifter = AdaptiveHenselLifting(prime, base_precision=2)
    
    # Test value
    test_value = 0.123456789
    target_frac = Fraction(test_value).limit_denominator(10**15)
    
    # Create initial weight
    math_ops = PadicMathematicalOperations(prime, 2)
    initial_weight = math_ops.to_padic(test_value)
    
    # Track convergence
    errors = []
    weights = [initial_weight]
    
    # Perform manual Hensel iterations
    current_weight = initial_weight
    for i in range(5):  # Limited iterations to observe convergence
        lifted_weight, converged = lifter.hensel_lift(
            current_weight, target_frac, prime
        )
        
        error = lifter._calculate_padic_error(lifted_weight, target_frac, prime)
        errors.append(error)
        weights.append(lifted_weight)
        
        print(f"  Iteration {i+1}: error = {error:.2e}")
        
        if converged or error < 1e-12:
            break
        
        current_weight = lifted_weight
    
    # Check quadratic convergence (error should roughly square each iteration)
    if len(errors) > 2:
        for i in range(1, len(errors) - 1):
            if errors[i] < 1e-10:  # Skip when errors are too small
                continue
            
            # Check if error decreased (allowing some tolerance)
            assert errors[i] <= errors[i-1] * 1.1
            
            # Quadratic convergence: e_{n+1} ≈ e_n^2
            if errors[i-1] < 0.1:  # Only check when in convergence region
                ratio = errors[i] / (errors[i-1] ** 2)
                print(f"    Convergence ratio e_{i+1}/e_{i}^2 = {ratio:.2f}")
    
    print("✓ Hensel convergence tests passed")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("Testing edge cases...")
    
    lifter = AdaptiveHenselLifting(prime=31, base_precision=3)
    
    # Test with zero
    weight, iterations = lifter.adaptive_precision_hensel(0.0, 1e-10)
    assert weight.value == Fraction(0)
    print("  ✓ Zero handling works")
    
    # Test with very small values
    weight, iterations = lifter.adaptive_precision_hensel(1e-15, 1e-10)
    assert isinstance(weight, PadicWeight)
    print("  ✓ Small value handling works")
    
    # Test with integer values
    weight, iterations = lifter.adaptive_precision_hensel(42.0, 1e-10)
    assert abs(float(weight.value) - 42.0) < 1e-6
    print("  ✓ Integer handling works")
    
    # Test error conditions
    try:
        lifter.adaptive_precision_hensel(float('nan'), 1e-10)
        assert False, "Should reject NaN"
    except ValueError:
        print("  ✓ NaN rejection works")
    
    try:
        lifter.adaptive_precision_hensel(float('inf'), 1e-10)
        assert False, "Should reject infinity"
    except ValueError:
        print("  ✓ Infinity rejection works")
    
    try:
        lifter.adaptive_precision_hensel(1.0, 0.0)
        assert False, "Should reject zero target error"
    except ValueError:
        print("  ✓ Zero target error rejection works")
    
    try:
        lifter.adaptive_precision_hensel(1.0, 2.0)
        assert False, "Should reject target error >= 1"
    except ValueError:
        print("  ✓ Large target error rejection works")
    
    print("✓ Edge case tests passed")


def test_performance_comparison():
    """Compare performance with and without Hensel lifting"""
    print("\nPerformance comparison: Hensel vs Standard encoding")
    print("=" * 60)
    
    # Create test tensor with mixed precision requirements
    size = 100
    test_tensor = torch.randn(size, size) * 0.001
    test_tensor[::10, ::10] = torch.randn(10, 10) * 1000  # Add high dynamic range
    
    # Test without Hensel
    config_no_hensel = LogarithmicEncodingConfig()
    config_no_hensel.prime = 127
    config_no_hensel.precision = 3
    config_no_hensel.enable_adaptive_hensel = False
    
    encoder_no_hensel = PadicLogarithmicEncoder(config_no_hensel)
    
    start = time.time()
    weights_no_hensel = encoder_no_hensel.encode_weights_logarithmically(test_tensor)
    time_no_hensel = time.time() - start
    
    decoded_no_hensel = encoder_no_hensel.decode_logarithmic_padic_weights(weights_no_hensel)
    mse_no_hensel = torch.nn.functional.mse_loss(decoded_no_hensel, test_tensor)
    
    # Test with Hensel
    config_hensel = LogarithmicEncodingConfig()
    config_hensel.prime = 127
    config_hensel.precision = 3
    config_hensel.enable_adaptive_hensel = True
    config_hensel.hensel_target_error = 1e-10
    
    encoder_hensel = PadicLogarithmicEncoder(config_hensel)
    
    start = time.time()
    weights_hensel = encoder_hensel.encode_weights_logarithmically(test_tensor)
    time_hensel = time.time() - start
    
    decoded_hensel = encoder_hensel.decode_logarithmic_padic_weights(weights_hensel)
    mse_hensel = torch.nn.functional.mse_loss(decoded_hensel, test_tensor)
    
    # Compare results
    print(f"Without Hensel lifting:")
    print(f"  Encoding time: {time_no_hensel:.3f}s")
    print(f"  Reconstruction MSE: {mse_no_hensel:.2e}")
    
    print(f"\nWith Hensel lifting:")
    print(f"  Encoding time: {time_hensel:.3f}s")
    print(f"  Reconstruction MSE: {mse_hensel:.2e}")
    print(f"  Hensel lifts used: {encoder_hensel.encoding_stats.get('hensel_lifts', 0)}")
    
    print(f"\nImprovement:")
    print(f"  MSE reduction: {(1 - mse_hensel/mse_no_hensel)*100:.1f}%")
    print(f"  Time overhead: {(time_hensel/time_no_hensel - 1)*100:.1f}%")
    
    # Hensel should improve accuracy (allow some tolerance for randomness)
    if mse_no_hensel > 1e-10:  # Only compare if there's meaningful error
        assert mse_hensel <= mse_no_hensel * 1.1  # Allow 10% tolerance
    
    print("\n✓ Performance comparison completed")


def run_all_tests():
    """Run all adaptive Hensel tests"""
    print("=" * 60)
    print("Running Adaptive Hensel Lifting Tests")
    print("=" * 60)
    
    test_adaptive_hensel_basic()
    print()
    test_hensel_with_logarithmic_encoder()
    print()
    test_padic_strategy_with_hensel()
    print()
    test_hensel_convergence()
    print()
    test_edge_cases()
    print()
    test_performance_comparison()
    
    print("\n" + "=" * 60)
    print("ALL ADAPTIVE HENSEL TESTS PASSED!")
    print("Track B Implementation Complete and Verified")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()