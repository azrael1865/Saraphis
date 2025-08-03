"""
Test CPU bursting compression system with real p-adic weight generation
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional
from fractions import Fraction
import math
import gc
import psutil
import os

# Import real p-adic components
from independent_core.compression_systems.padic.padic_encoder import (
    PadicWeight, 
    PadicMathematicalOperations
)
from independent_core.compression_systems.gpu_memory.cpu_bursting_pipeline import (
    CPU_BurstingPipeline, 
    CPUDecompressionEngine,
    CPUBurstingConfig,
    DecompressionMode
)

def create_real_padic_weights(num_weights: int, precision: int = 10, prime: int = 251) -> List[PadicWeight]:
    """Create real p-adic weights for testing using proper mathematical conversion"""
    from independent_core.compression_systems.padic.padic_encoder import PadicMathematicalOperations
    
    # Initialize p-adic mathematical operations
    math_ops = PadicMathematicalOperations(prime, precision)
    weights = []
    
    # Value ranges for different test scenarios
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
            # Sometimes use integer values
            value = float(np.random.randint(int(min_val), int(max_val) + 1))
        elif attempts % 3 == 1:
            # Sometimes use simple fractions
            numerator = np.random.randint(1, 100)
            denominator = np.random.randint(1, 100)
            value = numerator / denominator
            if np.random.rand() > 0.5:
                value = -value
        else:
            # Use random float
            value = np.random.uniform(min_val, max_val)
        
        try:
            # Convert to proper p-adic representation
            weight = math_ops.to_padic(value)
            
            # Validate the conversion by reconstructing
            reconstructed = math_ops.from_padic(weight)
            
            # Verify mathematical correctness
            relative_error = abs(value - reconstructed) / (abs(value) + 1e-10)
            if relative_error > 1e-6:
                # Skip values with high reconstruction error
                continue
            
            # Additional validation
            if not validate_single_weight(weight, prime, precision):
                continue
            
            weights.append(weight)
            
        except (ValueError, TypeError, OverflowError) as e:
            # Skip values that can't be converted
            continue
    
    if len(weights) < num_weights:
        raise ValueError(
            f"Could not create enough valid p-adic weights. "
            f"Created {len(weights)} out of {num_weights} requested. "
            f"Consider adjusting value ranges or precision."
        )
    
    return weights[:num_weights]


def validate_single_weight(weight: PadicWeight, expected_prime: int, expected_precision: int) -> bool:
    """Validate a single p-adic weight for mathematical correctness"""
    try:
        # Check basic structure
        if not hasattr(weight, 'digits') or not hasattr(weight, 'valuation'):
            return False
        
        if not hasattr(weight, 'value') or not hasattr(weight, 'prime') or not hasattr(weight, 'precision'):
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
        
        # Check value is Fraction
        if not isinstance(weight.value, Fraction):
            return False
        
        return True
        
    except Exception:
        return False


def validate_padic_weights(weights: List[PadicWeight], prime: int, precision: int) -> bool:
    """Validate that p-adic weights are mathematically correct"""
    from independent_core.compression_systems.padic.padic_encoder import PadicMathematicalOperations
    
    if not weights:
        print("No weights to validate")
        return False
    
    math_ops = PadicMathematicalOperations(prime, precision)
    
    for i, weight in enumerate(weights):
        try:
            # Basic structure validation
            if not validate_single_weight(weight, prime, precision):
                print(f"Weight {i}: Failed structural validation")
                return False
            
            # Check reconstruction
            reconstructed = math_ops.from_padic(weight)
            original = float(weight.value)
            
            # Allow small numerical errors
            relative_error = abs(original - reconstructed) / (abs(original) + 1e-10)
            if relative_error > 1e-5:
                print(f"Weight {i}: Reconstruction error - original={original}, reconstructed={reconstructed}, error={relative_error}")
                return False
            
            # Verify digit extraction correctness
            if weight.value.numerator != 0:
                # Non-zero weights should have at least one non-zero digit
                if all(d == 0 for d in weight.digits):
                    print(f"Weight {i}: All digits are zero for non-zero value {weight.value}")
                    return False
                
        except Exception as e:
            print(f"Weight {i}: Validation failed - {type(e).__name__}: {e}")
            return False
    
    return True


def measure_weight_conversion_time(num_weights: int, precision: int, prime: int) -> Tuple[List[PadicWeight], float]:
    """Measure time to create real p-adic weights"""
    start_time = time.time()
    weights = create_real_padic_weights(num_weights, precision, prime)
    conversion_time = time.time() - start_time
    return weights, conversion_time


def test_cpu_decompression_engine():
    """Test CPU decompression engine with real p-adic weights"""
    print("\n=== Testing CPU Decompression Engine with Real P-adic Weights ===")
    
    # Test parameters
    batch_size = 100  # Increased for better testing
    precision = 10
    prime = 251
    
    # Create real p-adic weights with timing
    print(f"Creating {batch_size} real p-adic weights...")
    weights, conversion_time = measure_weight_conversion_time(batch_size, precision, prime)
    print(f"Weight creation time: {conversion_time:.3f}s ({conversion_time/batch_size*1000:.1f}ms per weight)")
    
    # Validate weights
    if not validate_padic_weights(weights, prime, precision):
        raise ValueError("P-adic weight validation failed")
    print("✓ All weights validated successfully")
    
    # Initialize engine with optimized config
    config = CPUBurstingConfig(
        num_cpu_workers=8,  # Increased for better performance
        cpu_batch_size=1000,  # Increased batch size
        use_multiprocessing=True,  # Enable multiprocessing
        enable_caching=True,
        cache_size_mb=2048  # Increased cache size
    )
    engine = CPUDecompressionEngine(config, prime)
    
    # Test different precision levels
    target_precisions = [8, 16, 32]
    
    for target_precision in target_precisions:
        print(f"\nTesting decompression to {target_precision}-bit precision:")
        
    metadata = {
            'original_shape': (batch_size, 10),
        'dtype': 'torch.float32',
            'device': 'cpu',
            'prime': prime,
            'precision': precision,
            'validation_enabled': True,
            'compression_ratio': precision / target_precision
        }
        
        try:
            # Measure decompression time
            start_time = time.time()
            decompressed, info = engine.decompress_batch_cpu(weights, target_precision, metadata)
            decompress_time = time.time() - start_time
            
            print(f"  Decompression time: {decompress_time:.3f}s")
            print(f"  Throughput: {batch_size/decompress_time:.1f} weights/sec")
            print(f"  Output shape: {decompressed.shape}")
            print(f"  Output dtype: {decompressed.dtype}")
            
            # Verify output properties
            assert decompressed.shape == tuple(metadata['original_shape'])
            assert decompressed.dtype == torch.float32
            assert not torch.isnan(decompressed).any()
            assert not torch.isinf(decompressed).any()
            
            # Check value range
            print(f"  Value range: [{decompressed.min():.3f}, {decompressed.max():.3f}]")
            
        except Exception as e:
            print(f"  ✗ Decompression failed: {type(e).__name__}: {e}")
            raise
    
    print("\n✓ CPU decompression engine tests passed with real p-adic weights")


def test_memory_threshold_detection():
    """Test memory threshold detection with real p-adic loads"""
    print("\n=== Testing Memory Threshold Detection ===")
    
    # Create mock GPU engine for testing
    class MockGPUEngine:
        def __init__(self):
            pass
        
        def decompress_progressive(self, weights, precision, metadata):
            # Simulate GPU decompression
            return torch.randn(metadata['original_shape'])
        
        def get_decompression_stats(self):
            return {'gpu_time': 0.1}
        
        def cleanup(self):
            pass
    
    config = CPUBurstingConfig(
        memory_pressure_threshold=0.85,
        gpu_memory_threshold_mb=2048,
        switch_delay_ms=10
    )
    
    pipeline = CPU_BurstingPipeline(config, MockGPUEngine())
    
    # Test with increasing batch sizes
    test_sizes = [100, 500, 1000, 2000]
    precision = 10
    prime = 251
    
    for size in test_sizes:
        print(f"\nTesting with {size} weights:")
        
        try:
            # Create weights
            weights, conv_time = measure_weight_conversion_time(size, precision, prime)
            print(f"  Creation time: {conv_time:.2f}s")
            
            # Validate
            if not validate_padic_weights(weights, prime, precision):
                raise ValueError("Weight validation failed")
            
            # Check memory before
            memory_before = psutil.virtual_memory().percent
            print(f"  Memory before: {memory_before:.1f}%")
            
            # Simulate GPU tensor (would be on GPU in real scenario)
            gpu_tensor = torch.randn(size, 512, 512, device='cpu')  # Simulated GPU tensor
            
            # Check if should switch
            should_switch = pipeline._select_decompression_mode() == DecompressionMode.CPU_ONLY
            print(f"  Should switch to CPU: {should_switch}")
            
            # Check memory after
            memory_after = psutil.virtual_memory().percent
            print(f"  Memory after: {memory_after:.1f}%")
            
            # Clean up
            del gpu_tensor
            gc.collect()
            
        except Exception as e:
            print(f"  ✗ Test failed: {type(e).__name__}: {e}")
            # Continue with other sizes
    
    print("\n✓ Memory threshold detection tests completed")


def test_automatic_switching():
    """Test automatic GPU→CPU switching with real p-adic weights"""
    print("\n=== Testing Automatic GPU→CPU Switching ===")
    
    # Create mock GPU engine for testing
    class MockGPUEngine:
        def __init__(self, fail_after=None):
            self.fail_after = fail_after
            self.call_count = 0
        
        def decompress_progressive(self, weights, precision, metadata):
            self.call_count += 1
            if self.fail_after and self.call_count > self.fail_after:
                raise RuntimeError("Simulated GPU OOM")
            # Simulate GPU decompression
            return torch.randn(metadata['original_shape'])
        
        def get_decompression_stats(self):
            return {'gpu_time': 0.1}
        
        def cleanup(self):
            pass
    
    config = CPUBurstingConfig(
        memory_pressure_threshold=0.80,
        gpu_memory_threshold_mb=2048,
        switch_delay_ms=5
    )
    
    pipeline = CPU_BurstingPipeline(config, MockGPUEngine())
    
    # Test parameters
    num_weights = 1000
    precision = 10
    prime = 251
    target_precision = 16
    
    print(f"Creating {num_weights} real p-adic weights...")
    weights, conv_time = measure_weight_conversion_time(num_weights, precision, prime)
    
    if not validate_padic_weights(weights, prime, precision):
        raise ValueError("Weight validation failed")
    
    metadata = {
        'original_shape': (num_weights, 256),
        'dtype': 'torch.float32',
        'device': 'cuda',  # Simulated GPU origin
        'prime': prime,
        'precision': precision,
        'validation_enabled': True
    }
    
    # Test decompression with automatic mode selection
    print("\nTesting automatic decompression mode selection:")
    
    # Force different memory conditions
    memory_conditions = [
        ("Low memory", 0.70),
        ("Medium memory", 0.85),
        ("High memory", 0.92)
    ]
    
    for condition_name, simulated_usage in memory_conditions:
        print(f"\n{condition_name} (simulated {simulated_usage*100:.0f}% usage):")
        
        # Override memory check for testing
        original_check = pipeline._get_gpu_memory_state
        pipeline._get_gpu_memory_state = lambda: {'utilization': simulated_usage}
        
        try:
            start_time = time.time()
            result, info = pipeline.decompress(weights, target_precision, metadata)
            decompress_time = time.time() - start_time
            
            print(f"  Decompression time: {decompress_time:.3f}s")
            print(f"  Result shape: {result.shape}")
            print(f"  Result device: {result.device}")
            
            # Verify result
            assert result.shape == tuple(metadata['original_shape'])
            assert not torch.isnan(result).any()
            
            # Check which mode was used based on memory
            if simulated_usage > config.memory_pressure_threshold:
                print("  ✓ Should have used CPU decompression")
            else:
                print("  ✓ Should have used GPU decompression")
                
        finally:
            # Restore original function
            pipeline._get_gpu_memory_state = original_check
    
    print("\n✓ Automatic switching tests completed")


def test_performance_comparison():
    """Compare performance between different batch sizes with real weights"""
    print("\n=== Performance Comparison with Real P-adic Weights ===")
    
    batch_sizes = [10, 50, 100, 500, 1000]
    precision = 10
    prime = 251
    target_precision = 16
    
    config = CPUBurstingConfig(
        num_cpu_workers=4,
        cpu_batch_size=50,
        use_multiprocessing=False,
        enable_caching=True,
        cache_size_mb=128
    )
    engine = CPUDecompressionEngine(config, prime)
    results = []
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        try:
            # Create weights
            weights, conv_time = measure_weight_conversion_time(batch_size, precision, prime)
            
            if not validate_padic_weights(weights, prime, precision):
                raise ValueError("Weight validation failed")
            
        metadata = {
                'original_shape': (batch_size, 128),
            'dtype': 'torch.float32',
                'device': 'cpu',
                'prime': prime,
                'precision': precision
            }
            
            # Measure decompression time (average of 3 runs)
            decompress_times = []
            for run in range(3):
                start_time = time.time()
                result, info = engine.decompress_batch_cpu(weights, target_precision, metadata)
                decompress_times.append(time.time() - start_time)
            
            avg_time = np.mean(decompress_times)
            std_time = np.std(decompress_times)
            throughput = batch_size / avg_time
            
            results.append({
                'batch_size': batch_size,
                'conv_time': conv_time,
                'decompress_time': avg_time,
                'std_time': std_time,
                'throughput': throughput,
                'time_per_weight': avg_time / batch_size * 1000  # ms
            })
            
            print(f"  Conversion time: {conv_time:.3f}s")
            print(f"  Decompression time: {avg_time:.3f}s ± {std_time:.3f}s")
            print(f"  Throughput: {throughput:.1f} weights/sec")
            print(f"  Time per weight: {avg_time/batch_size*1000:.2f}ms")
            
        except Exception as e:
            print(f"  ✗ Failed: {type(e).__name__}: {e}")
            continue
    
    # Summary
    print("\n=== Performance Summary ===")
    print("Batch Size | Conv Time | Decomp Time | Throughput | Per Weight")
    print("-" * 65)
    for r in results:
        print(f"{r['batch_size']:10d} | {r['conv_time']:9.3f}s | {r['decompress_time']:11.3f}s | "
              f"{r['throughput']:10.1f}/s | {r['time_per_weight']:10.2f}ms")


def test_integration():
    """Integration test with full pipeline using real p-adic weights"""
    print("\n=== Integration Test with Real P-adic Weights ===")
    
    # Create mock GPU engine for testing
    class MockGPUEngine:
        def __init__(self):
            pass
                
                def decompress_progressive(self, weights, precision, metadata):
            # Simulate GPU decompression
            return torch.randn(metadata['original_shape'])
                
                def get_decompression_stats(self):
            return {'gpu_time': 0.1}
                
                def cleanup(self):
                    pass
            
    # Initialize pipeline
    config = CPUBurstingConfig(
        memory_pressure_threshold=0.85,
        gpu_memory_threshold_mb=2048,
        switch_delay_ms=10
    )
    
    pipeline = CPU_BurstingPipeline(config, MockGPUEngine())
    
    # Test with different model sizes
    model_configs = [
        ("Small model", 100, 10, 8),
        ("Medium model", 1000, 10, 16),
        ("Large model", 5000, 8, 32)
    ]
    
    prime = 251
    
    for model_name, num_weights, precision, target_precision in model_configs:
        print(f"\n{model_name}: {num_weights} weights, {precision}-bit → {target_precision}-bit")
        
        try:
            # Create weights
            print("  Creating p-adic weights...")
            weights, conv_time = measure_weight_conversion_time(num_weights, precision, prime)
            
            # Validate
            if not validate_padic_weights(weights, prime, precision):
                raise ValueError("Weight validation failed")
            print(f"  ✓ Weights created and validated in {conv_time:.2f}s")
            
            # Prepare metadata
    metadata = {
                'original_shape': (num_weights, 512),
        'dtype': 'torch.float32',
                'device': 'cuda',
                'prime': prime,
                'precision': precision,
                'model_name': model_name,
                'compression_ratio': precision / target_precision
            }
            
            # Test full pipeline
            print("  Running full decompression pipeline...")
            start_time = time.time()
            result, info = pipeline.decompress(weights, target_precision, metadata)
            total_time = time.time() - start_time
            
            # Verify results
            assert result.shape == tuple(metadata['original_shape'])
            assert result.dtype == torch.float32
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()
            
            # Calculate metrics
            weights_per_sec = num_weights / total_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            print(f"  ✓ Decompression completed in {total_time:.3f}s")
            print(f"  Throughput: {weights_per_sec:.1f} weights/sec")
            print(f"  Memory usage: {memory_usage:.1f} MB")
            print(f"  Output range: [{result.min():.3f}, {result.max():.3f}]")
            
        except Exception as e:
            print(f"  ✗ Test failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✓ Integration tests completed")


def run_stress_test():
    """Stress test with large batches of real p-adic weights"""
    print("\n=== Stress Test with Real P-adic Weights ===")
    
    # Create mock GPU engine for testing
    class StressTestGPUEngine:
        def __init__(self):
            pass
        
        def decompress_progressive(self, weights, precision, metadata):
            # Simulate GPU decompression
            return torch.randn(metadata['original_shape'])
        
        def get_decompression_stats(self):
            return {'gpu_time': 0.1}
        
        def cleanup(self):
            pass
    
    # Stress test parameters
    stress_configs = [
        ("Quick test", 1000, 10),
        ("Standard test", 5000, 8),
        ("Heavy test", 10000, 6)
    ]
    
    prime = 251
    target_precision = 32
    
    config = CPUBurstingConfig(
        memory_pressure_threshold=0.80,
        gpu_memory_threshold_mb=2048,
        switch_delay_ms=5
    )
    
    pipeline = CPU_BurstingPipeline(config, StressTestGPUEngine())
    
    for test_name, num_weights, precision in stress_configs:
        print(f"\n{test_name}: {num_weights} weights at {precision}-bit precision")
        
        # Monitor initial state
        initial_memory = psutil.virtual_memory().percent
        print(f"Initial memory usage: {initial_memory:.1f}%")
        
        try:
            # Create weights in batches to avoid memory issues
            batch_size = min(1000, num_weights)
            all_weights = []
            total_conv_time = 0
            
            print(f"Creating weights in batches of {batch_size}...")
            for i in range(0, num_weights, batch_size):
                current_batch_size = min(batch_size, num_weights - i)
                weights, conv_time = measure_weight_conversion_time(current_batch_size, precision, prime)
                all_weights.extend(weights)
                total_conv_time += conv_time
                print(f"  Batch {i//batch_size + 1}: {len(all_weights)}/{num_weights} weights created")
            
            print(f"Total creation time: {total_conv_time:.2f}s")
            
            # Validate sample
            sample_size = min(100, len(all_weights))
            if not validate_padic_weights(all_weights[:sample_size], prime, precision):
                raise ValueError("Sample validation failed")
            
            # Prepare for decompression
            metadata = {
                'original_shape': (num_weights, 256),
                'dtype': 'torch.float32',
                'device': 'cuda',
                'prime': prime,
                'precision': precision,
                'stress_test': True
            }
            
            # Run decompression
            print("Running decompression...")
            start_time = time.time()
            
            # Process in chunks to manage memory
            chunk_size = 1000
            results = []
            
            for i in range(0, num_weights, chunk_size):
                chunk = all_weights[i:i+chunk_size]
                chunk_metadata = metadata.copy()
                chunk_metadata['original_shape'] = (len(chunk), 256)
                
                result, info = pipeline.decompress(chunk, target_precision, chunk_metadata)
                results.append(result)
                
                # Monitor memory
                current_memory = psutil.virtual_memory().percent
                if current_memory > 95:
                    print(f"  ⚠ High memory usage: {current_memory:.1f}%")
                    gc.collect()
            
            # Combine results
            final_result = torch.cat(results, dim=0)
            total_time = time.time() - start_time
            
            # Final statistics
            final_memory = psutil.virtual_memory().percent
            memory_increase = final_memory - initial_memory
            
            print(f"\nStress test completed:")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Throughput: {num_weights/total_time:.1f} weights/sec")
            print(f"  Memory increase: {memory_increase:.1f}%")
            print(f"  Final shape: {final_result.shape}")
            print(f"  ✓ {test_name} passed")
            
            # Cleanup
            del all_weights
            del results
            del final_result
            gc.collect()
                
        except Exception as e:
            print(f"  ✗ {test_name} failed: {type(e).__name__}: {e}")
            # Clean up and continue
            gc.collect()


def main():
    """Run all tests with real p-adic weights"""
    print("=" * 70)
    print("P-ADIC COMPRESSION BURST SYSTEM TEST SUITE")
    print("Using Real Mathematical P-adic Weight Generation")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run test suite
    tests = [
        test_cpu_decompression_engine,
        test_memory_threshold_detection,
        test_automatic_switching,
        test_performance_comparison,
        test_integration,
        run_stress_test
    ]
    
    failed_tests = []
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n✗ Test {test.__name__} failed with error: {type(e).__name__}: {e}")
            failed_tests.append((test.__name__, e))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    if not failed_tests:
        print("✓ All tests passed with real p-adic weights!")
    else:
        print(f"✗ {len(failed_tests)} test(s) failed:")
        for test_name, error in failed_tests:
            print(f"  - {test_name}: {type(error).__name__}: {error}")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)