"""
Test suite for Adaptive Precision Wrapper
Validates the implementation of Task E
"""

import torch
import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from independent_core.compression_systems.padic.adaptive_precision_wrapper import (
    AdaptivePrecisionWrapper,
    AdaptivePrecisionConfig,
    PrecisionAllocation
)


def test_basic_functionality():
    """Test basic adaptive precision computation"""
    print("Testing basic functionality...")
    
    config = AdaptivePrecisionConfig(
        prime=257,
        base_precision=3,
        min_precision=2,
        max_precision=4,
        target_error=1e-5
    )
    
    wrapper = AdaptivePrecisionWrapper(config)
    
    # Test with simple tensor
    tensor = torch.tensor([1.0, 2.0, 3.0, 0.5, -1.5, 10.0])
    weights, precision_map = wrapper.compute_adaptive_precision(tensor)
    
    assert len(weights) == tensor.numel()
    assert precision_map.shape == tensor.shape
    assert all(config.min_precision <= p <= config.max_precision for p in precision_map)
    
    print("✓ Basic functionality test passed")


def test_importance_based_allocation():
    """Test precision allocation based on importance scores"""
    print("Testing importance-based allocation...")
    
    config = AdaptivePrecisionConfig(
        prime=257,
        base_precision=3,
        min_precision=2,
        max_precision=4
    )
    
    wrapper = AdaptivePrecisionWrapper(config)
    
    # Create tensor with varying importance
    tensor = torch.randn(10, 10)
    importance_scores = torch.abs(tensor) * torch.rand(10, 10)
    
    # Allocate with limited bit budget
    total_bits = 500  # Forcing constraint
    precision_allocation = wrapper.allocate_precision_by_importance(
        tensor, importance_scores, total_bits
    )
    
    assert precision_allocation.shape == tensor.shape
    assert all(config.min_precision <= p <= config.max_precision 
              for p in precision_allocation.flatten())
    
    # Check that higher importance gets more precision
    high_importance_idx = importance_scores.argmax()
    low_importance_idx = importance_scores.argmin()
    assert precision_allocation.flatten()[high_importance_idx] >= \
           precision_allocation.flatten()[low_importance_idx]
    
    print("✓ Importance-based allocation test passed")


def test_batch_compression():
    """Test batch compression with full pipeline"""
    print("Testing batch compression...")
    
    config = AdaptivePrecisionConfig(
        prime=257,
        base_precision=3,
        min_precision=2,
        max_precision=4,
        compression_priority=0.7
    )
    
    wrapper = AdaptivePrecisionWrapper(config)
    
    # Test batch processing
    batch = torch.randn(32, 64) * 5.0
    importance = torch.abs(batch) + 0.1
    
    allocation = wrapper.batch_compress_with_adaptive_precision(
        batch, importance, total_bits=32*64*16
    )
    
    assert isinstance(allocation, PrecisionAllocation)
    assert len(allocation.weights) == batch.numel()
    assert allocation.precision_map.shape == batch.shape
    assert allocation.error_map.shape == batch.shape
    assert allocation.compression_ratio > 0
    
    # Check error bounds
    max_error = allocation.error_map.max().item()
    assert max_error < 1.0, f"Max error {max_error} exceeds bounds"
    
    print(f"✓ Batch compression test passed (ratio: {allocation.compression_ratio:.2f}x)")


def test_logarithmic_weights():
    """Test creation of logarithmic p-adic weights"""
    print("Testing logarithmic weight creation...")
    
    config = AdaptivePrecisionConfig(prime=257, base_precision=3)
    wrapper = AdaptivePrecisionWrapper(config)
    
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    precision_allocation = torch.tensor([[2, 3], [3, 4]], dtype=torch.int32)
    
    log_weights = wrapper.create_logarithmic_padic_weights(tensor, precision_allocation)
    
    assert len(log_weights) == tensor.numel()
    
    for i, weight in enumerate(log_weights):
        assert weight.padic_weight is not None
        assert weight.original_value == tensor.flatten()[i].item()
        assert 'precision' in weight.compression_metadata
        assert weight.compression_metadata['precision'] == precision_allocation.flatten()[i].item()
    
    print("✓ Logarithmic weight creation test passed")


def test_efficiency_monitoring():
    """Test efficiency monitoring and analysis"""
    print("Testing efficiency monitoring...")
    
    config = AdaptivePrecisionConfig(
        prime=257,
        base_precision=3,
        compression_priority=0.5
    )
    
    wrapper = AdaptivePrecisionWrapper(config)
    
    # Create test allocation
    tensor = torch.randn(20, 20)
    allocation = wrapper.batch_compress_with_adaptive_precision(tensor)
    
    # Monitor efficiency
    metrics = wrapper.monitor_precision_efficiency(allocation)
    
    assert 'average_precision' in metrics
    assert 'compression_ratio' in metrics
    assert 'error_stats' in metrics
    assert 'efficiency_score' in metrics
    assert 0 <= metrics['efficiency_score'] <= 1
    
    print(f"✓ Efficiency monitoring test passed (score: {metrics['efficiency_score']:.3f})")


def test_memory_efficiency():
    """Test memory efficiency of compression"""
    print("Testing memory efficiency...")
    
    config = AdaptivePrecisionConfig(
        prime=257,
        base_precision=3,
        enable_memory_tracking=True
    )
    
    wrapper = AdaptivePrecisionWrapper(config)
    
    # Large tensor to test memory
    large_tensor = torch.randn(100, 100)
    
    # Measure memory before
    original_size = large_tensor.element_size() * large_tensor.numel()
    
    # Compress
    allocation = wrapper.batch_compress_with_adaptive_precision(large_tensor)
    
    # Estimate compressed size
    bits_per_digit = np.log2(config.prime)
    compressed_size = allocation.total_bits / 8  # Convert to bytes
    
    memory_saved = original_size - compressed_size
    savings_ratio = memory_saved / original_size
    
    print(f"✓ Memory efficiency test passed")
    print(f"  Original size: {original_size} bytes")
    print(f"  Compressed size: {compressed_size:.0f} bytes") 
    print(f"  Memory saved: {savings_ratio:.1%}")


def test_hensel_convergence():
    """Test Hensel lifting convergence properties"""
    print("Testing Hensel lifting convergence...")
    
    config = AdaptivePrecisionConfig(
        prime=257,
        base_precision=4,
        target_error=1e-8
    )
    
    wrapper = AdaptivePrecisionWrapper(config)
    
    # Test values that should converge quickly
    easy_values = torch.tensor([1.0, 2.0, 0.5, 0.25])
    
    # Test values that need more iterations
    hard_values = torch.tensor([np.pi, np.e, np.sqrt(2), np.sqrt(3)])
    
    easy_weights, easy_precision = wrapper.compute_adaptive_precision(easy_values)
    hard_weights, hard_precision = wrapper.compute_adaptive_precision(hard_values)
    
    # Check convergence statistics
    stats = wrapper.hensel.get_stats()
    assert stats['total_lifts'] > 0
    assert 0 <= stats['convergence_rate'] <= 1
    
    print(f"✓ Hensel convergence test passed")
    print(f"  Convergence rate: {stats['convergence_rate']:.1%}")
    print(f"  Average iterations: {stats.get('average_iterations', 0):.1f}")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("Testing edge cases...")
    
    config = AdaptivePrecisionConfig(prime=257, base_precision=3)
    wrapper = AdaptivePrecisionWrapper(config)
    
    # Test with zeros
    zeros = torch.zeros(5)
    weights, precision = wrapper.compute_adaptive_precision(zeros)
    assert len(weights) == 5
    
    # Test with very small values
    tiny = torch.tensor([1e-10, 1e-15, 1e-20])
    weights, precision = wrapper.compute_adaptive_precision(tiny)
    assert len(weights) == 3
    
    # Test with very large values
    large = torch.tensor([1e5, 1e6, 1e7])
    weights, precision = wrapper.compute_adaptive_precision(large)
    assert len(weights) == 3
    
    # Test with negative values
    negative = torch.tensor([-1.0, -2.0, -3.0])
    weights, precision = wrapper.compute_adaptive_precision(negative)
    assert len(weights) == 3
    
    print("✓ Edge cases test passed")


def run_performance_benchmark():
    """Run performance benchmark"""
    print("\nRunning performance benchmark...")
    
    config = AdaptivePrecisionConfig(
        prime=257,
        base_precision=3,
        enable_gpu_acceleration=torch.cuda.is_available()
    )
    
    wrapper = AdaptivePrecisionWrapper(config)
    
    sizes = [100, 500, 1000, 5000]
    times = []
    ratios = []
    
    for size in sizes:
        tensor = torch.randn(size)
        
        start = time.perf_counter()
        allocation = wrapper.batch_compress_with_adaptive_precision(tensor)
        elapsed = time.perf_counter() - start
        
        times.append(elapsed)
        ratios.append(allocation.compression_ratio)
        
        print(f"  Size {size}: {elapsed:.3f}s, ratio: {allocation.compression_ratio:.2f}x")
    
    print(f"\nAverage compression ratio: {np.mean(ratios):.2f}x")
    print(f"Total processing time: {sum(times):.3f}s")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Adaptive Precision Wrapper Test Suite")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_importance_based_allocation()
        test_batch_compression()
        test_logarithmic_weights()
        test_efficiency_monitoring()
        test_memory_efficiency()
        test_hensel_convergence()
        test_edge_cases()
        
        print("\n✅ All tests passed!")
        
        run_performance_benchmark()
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()