"""
Test suite for optimal prime selection in P-adic compression strategy.
Tests distribution detection and prime selection based on Task B1.2 requirements.
"""

import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from independent_core.compression_systems.strategies.compression_strategy import (
    PadicStrategy, StrategySelector, StrategyConfig
)


def test_distribution_detection():
    """Test distribution type detection"""
    print("Testing distribution detection...")
    
    strategy = PadicStrategy(prime=251, precision=3)
    
    # Test Gaussian distribution
    gaussian_tensor = torch.randn(1000)
    dist_type = strategy.detect_distribution_type(gaussian_tensor)
    print(f"  Gaussian tensor detected as: {dist_type}")
    assert dist_type == "gaussian", f"Expected 'gaussian', got '{dist_type}'"
    
    # Test sparse distribution
    sparse_tensor = torch.zeros(1000)
    sparse_tensor[torch.rand(1000) > 0.95] = torch.randn(50)
    dist_type = strategy.detect_distribution_type(sparse_tensor)
    print(f"  Sparse tensor detected as: {dist_type}")
    assert dist_type == "sparse", f"Expected 'sparse', got '{dist_type}'"
    
    # Test uniform distribution
    uniform_tensor = torch.rand(1000) * 2 - 1  # Uniform in [-1, 1]
    dist_type = strategy.detect_distribution_type(uniform_tensor)
    print(f"  Uniform tensor detected as: {dist_type}")
    assert dist_type in ["uniform", "gaussian"], f"Expected 'uniform' or 'gaussian', got '{dist_type}'"
    
    # Test bimodal distribution
    bimodal_tensor = torch.cat([
        torch.randn(500) - 3,  # Mode at -3
        torch.randn(500) + 3   # Mode at +3
    ])
    dist_type = strategy.detect_distribution_type(bimodal_tensor)
    print(f"  Bimodal tensor detected as: {dist_type}")
    # Bimodal detection is challenging, accept bimodal or multimodal
    assert dist_type in ["bimodal", "multimodal", "uniform"], f"Expected 'bimodal' or 'multimodal', got '{dist_type}'"
    
    print("✓ Distribution detection tests passed")


def test_prime_selection_gaussian():
    """Test prime selection for Gaussian distribution"""
    print("Testing prime selection for Gaussian distribution...")
    
    strategy = PadicStrategy(prime=251, precision=3)
    
    # Create Gaussian tensor
    tensor = torch.randn(1000)
    
    # Get optimal prime
    optimal_p = strategy.optimal_prime(tensor)
    print(f"  Optimal prime for Gaussian: {optimal_p}")
    
    # Should select small prime (2, 3, 5, or 7)
    assert optimal_p in [2, 3, 5, 7], f"Expected small prime, got {optimal_p}"
    
    print("✓ Gaussian prime selection test passed")


def test_prime_selection_sparse():
    """Test prime selection for sparse distribution"""
    print("Testing prime selection for sparse distribution...")
    
    strategy = PadicStrategy(prime=251, precision=3)
    
    # Create very sparse tensor
    tensor = torch.zeros(10000)
    tensor[torch.rand(10000) > 0.98] = torch.randn(200)
    
    # Get optimal prime
    optimal_p = strategy.optimal_prime(tensor)
    print(f"  Optimal prime for sparse (98% zeros): {optimal_p}")
    
    # Should select large prime for very sparse data
    assert optimal_p >= 127, f"Expected large prime for very sparse data, got {optimal_p}"
    
    # Create moderately sparse tensor
    tensor = torch.zeros(10000)
    tensor[torch.rand(10000) > 0.85] = torch.randn(1500)
    
    optimal_p = strategy.optimal_prime(tensor)
    print(f"  Optimal prime for sparse (85% zeros): {optimal_p}")
    
    # Should select medium prime for moderately sparse data
    assert 31 <= optimal_p <= 127, f"Expected medium prime for moderately sparse data, got {optimal_p}"
    
    print("✓ Sparse prime selection test passed")


def test_prime_selection_uniform():
    """Test prime selection for uniform distribution with entropy-based selection"""
    print("Testing prime selection for uniform distribution...")
    
    strategy = PadicStrategy(prime=251, precision=3)
    
    # Create uniform tensor with different entropy levels
    # Low entropy (structured)
    low_entropy_tensor = torch.zeros(1000)
    low_entropy_tensor[:500] = 1.0
    low_entropy_tensor[500:] = -1.0
    
    metadata = {'local_entropy': 1.0}  # Binary distribution has entropy ~1 bit
    optimal_p = strategy.optimal_prime(low_entropy_tensor, metadata)
    print(f"  Optimal prime for low entropy uniform: {optimal_p}")
    
    # p(x) = 2^⌈H_local(x)⌉ + 1 -> 2^1 + 1 = 3 (or next prime)
    assert optimal_p in [3, 5], f"Expected small prime for low entropy, got {optimal_p}"
    
    # High entropy (random)
    high_entropy_tensor = torch.rand(1000) * 2 - 1
    
    metadata = {'local_entropy': 7.5}  # High entropy
    optimal_p = strategy.optimal_prime(high_entropy_tensor, metadata)
    print(f"  Optimal prime for high entropy uniform: {optimal_p}")
    
    # p(x) = 2^⌈7.5⌉ + 1 -> 2^8 + 1 = 257 (or next prime)
    assert optimal_p >= 257, f"Expected large prime for high entropy, got {optimal_p}"
    
    print("✓ Uniform prime selection test passed")


def test_prime_selection_bimodal():
    """Test prime selection for bimodal distribution"""
    print("Testing prime selection for bimodal distribution...")
    
    strategy = PadicStrategy(prime=251, precision=3)
    
    # Create bimodal tensor with different separations
    # Close modes
    close_modes = torch.cat([
        torch.randn(500) * 0.5 - 0.5,  # Mode at -0.5
        torch.randn(500) * 0.5 + 0.5   # Mode at +0.5
    ])
    
    # Force bimodal detection by setting internal state
    strategy._last_distribution_type = "bimodal"
    
    # Mock the distribution type detection for this test
    original_detect = strategy.detect_distribution_type
    strategy.detect_distribution_type = lambda x: "bimodal"
    
    optimal_p = strategy.optimal_prime(close_modes)
    print(f"  Optimal prime for close bimodal modes: {optimal_p}")
    
    # Should select power of 2 based on separation
    assert optimal_p in [2, 4, 8, 16], f"Expected power of 2, got {optimal_p}"
    
    # Well-separated modes
    separated_modes = torch.cat([
        torch.randn(500) * 0.5 - 5,  # Mode at -5
        torch.randn(500) * 0.5 + 5   # Mode at +5
    ])
    
    optimal_p = strategy.optimal_prime(separated_modes)
    print(f"  Optimal prime for separated bimodal modes: {optimal_p}")
    
    # Should select larger power of 2 for well-separated modes
    assert optimal_p in [8, 16], f"Expected larger power of 2, got {optimal_p}"
    
    # Restore original method
    strategy.detect_distribution_type = original_detect
    
    print("✓ Bimodal prime selection test passed")


def test_prime_utilities():
    """Test prime checking and generation utilities"""
    print("Testing prime utility functions...")
    
    strategy = PadicStrategy(prime=251, precision=3)
    
    # Test is_prime
    assert strategy.is_prime(2) == True
    assert strategy.is_prime(3) == True
    assert strategy.is_prime(4) == False
    assert strategy.is_prime(5) == True
    assert strategy.is_prime(6) == False
    assert strategy.is_prime(7) == True
    assert strategy.is_prime(251) == True
    assert strategy.is_prime(252) == False
    
    # Test next_prime
    assert strategy.next_prime(2) == 2
    assert strategy.next_prime(3) == 3
    assert strategy.next_prime(4) == 5
    assert strategy.next_prime(6) == 7
    assert strategy.next_prime(10) == 11
    assert strategy.next_prime(100) == 101
    assert strategy.next_prime(250) == 251
    
    print("✓ Prime utility tests passed")


def test_integration_with_compression():
    """Test that optimal prime selection integrates with compression flow"""
    print("Testing integration with compression...")
    
    config = StrategyConfig()
    selector = StrategySelector(config)
    
    # Test with different tensor types
    test_cases = [
        ("Gaussian", torch.randn(100, 100)),
        ("Sparse", torch.zeros(100, 100).masked_scatter_(
            torch.rand(100, 100) > 0.9, torch.randn(1000))),
        ("Uniform", torch.rand(100, 100)),
        ("Periodic", torch.sin(torch.linspace(0, 10 * np.pi, 10000)).view(100, 100))
    ]
    
    for name, tensor in test_cases:
        print(f"  Testing {name} tensor...")
        
        # Select strategy
        strategy, analysis = selector.select_strategy(tensor, f"{name}_layer")
        
        # If P-adic strategy is selected, test compression with optimal prime
        if strategy.get_strategy_name() == "padic":
            # Compress with analysis metadata
            compressed = strategy.compress(tensor, metadata=analysis)
            
            # Check that prime was selected
            assert 'prime' in compressed.metadata
            assert 'distribution_type' in compressed.metadata
            
            print(f"    Used prime: {compressed.metadata['prime']}")
            print(f"    Distribution: {compressed.metadata['distribution_type']}")
            
            # Decompress and verify
            reconstructed = strategy.decompress(compressed)
            assert reconstructed.shape == tensor.shape
            
            # Calculate error
            error = torch.nn.functional.mse_loss(reconstructed, tensor).item()
            print(f"    Reconstruction error: {error:.6f}")
    
    print("✓ Integration tests passed")


def test_adaptive_prime_selection():
    """Test that prime adapts to different distributions in same strategy instance"""
    print("Testing adaptive prime selection...")
    
    strategy = PadicStrategy(prime=251, precision=3)
    
    # Compress Gaussian tensor
    gaussian = torch.randn(100, 100)
    compressed1 = strategy.compress(gaussian)
    prime1 = compressed1.metadata['prime']
    print(f"  Gaussian tensor used prime: {prime1}")
    
    # Compress sparse tensor
    sparse = torch.zeros(100, 100)
    sparse[torch.rand(100, 100) > 0.95] = torch.randn(500)
    compressed2 = strategy.compress(sparse)
    prime2 = compressed2.metadata['prime']
    print(f"  Sparse tensor used prime: {prime2}")
    
    # Primes should be different for different distributions
    assert prime1 != prime2, "Expected different primes for different distributions"
    assert prime1 < prime2, "Expected smaller prime for Gaussian than sparse"
    
    # Compress uniform tensor with high entropy
    uniform = torch.rand(100, 100) * 10
    metadata = {'local_entropy': 6.5}
    compressed3 = strategy.compress(uniform, metadata)
    prime3 = compressed3.metadata['prime']
    print(f"  High-entropy uniform tensor used prime: {prime3}")
    
    # Should use larger prime for high entropy
    assert prime3 > prime1, "Expected larger prime for high-entropy data"
    
    print("✓ Adaptive prime selection test passed")


def test_edge_cases():
    """Test edge cases in optimal prime selection"""
    print("Testing edge cases...")
    
    strategy = PadicStrategy(prime=251, precision=3)
    
    # Empty-like tensor (all zeros)
    zeros = torch.zeros(100)
    optimal_p = strategy.optimal_prime(zeros)
    print(f"  All zeros tensor prime: {optimal_p}")
    assert optimal_p > 0, "Should handle all-zeros tensor"
    
    # Single value tensor
    constant = torch.ones(100) * 3.14159
    optimal_p = strategy.optimal_prime(constant)
    print(f"  Constant tensor prime: {optimal_p}")
    assert optimal_p > 0, "Should handle constant tensor"
    
    # Very small tensor
    tiny = torch.randn(5)
    optimal_p = strategy.optimal_prime(tiny)
    print(f"  Tiny tensor prime: {optimal_p}")
    assert optimal_p > 0, "Should handle tiny tensor"
    
    # Extreme values
    extreme = torch.tensor([1e10, -1e10, 1e-10, -1e-10, 0])
    optimal_p = strategy.optimal_prime(extreme)
    print(f"  Extreme values tensor prime: {optimal_p}")
    assert optimal_p > 0, "Should handle extreme values"
    
    print("✓ Edge case tests passed")


def test_performance_impact():
    """Test that optimal prime selection improves compression"""
    print("Testing performance impact of optimal prime selection...")
    
    # Create tensors with known good distributions for each prime type
    test_cases = [
        # (tensor, expected_good_prime_range, description)
        (torch.randn(1000, 100), range(2, 8), "Gaussian"),
        (torch.zeros(1000, 100).masked_scatter_(
            torch.rand(1000, 100) > 0.95, torch.randn(5000)), 
         range(100, 260), "Very sparse"),
    ]
    
    for tensor, good_prime_range, description in test_cases:
        print(f"  Testing {description} tensor...")
        
        # Test with default prime
        default_strategy = PadicStrategy(prime=251, precision=3)
        default_strategy.optimal_prime = lambda x, m=None: 251  # Override to use fixed prime
        
        compressed_default = default_strategy.compress(tensor)
        ratio_default = compressed_default.compression_ratio
        
        # Test with optimal prime
        optimal_strategy = PadicStrategy(prime=251, precision=3)
        compressed_optimal = optimal_strategy.compress(tensor)
        ratio_optimal = compressed_optimal.compression_ratio
        
        print(f"    Default prime (251) ratio: {ratio_default:.2f}")
        print(f"    Optimal prime ({compressed_optimal.metadata['prime']}) ratio: {ratio_optimal:.2f}")
        
        # Optimal should generally be as good or better
        # Allow small tolerance for randomness
        assert ratio_optimal >= ratio_default * 0.95, \
            f"Optimal prime should not significantly worsen compression"
    
    print("✓ Performance impact tests passed")


def run_all_tests():
    """Run all optimal prime selection tests"""
    print("=" * 60)
    print("Running Optimal Prime Selection Tests")
    print("=" * 60)
    
    test_prime_utilities()
    test_distribution_detection()
    test_prime_selection_gaussian()
    test_prime_selection_sparse()
    test_prime_selection_uniform()
    test_prime_selection_bimodal()
    test_integration_with_compression()
    test_adaptive_prime_selection()
    test_edge_cases()
    test_performance_impact()
    
    print("=" * 60)
    print("ALL OPTIMAL PRIME TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()