#!/usr/bin/env python3
"""
Integration test for tropical channel extraction with polynomial system.
Validates end-to-end functionality and GPU acceleration.
"""

import torch
import time
from tropical_polynomial import TropicalPolynomial, TropicalMonomial, TropicalPolynomialOperations
from tropical_channel_extractor import (
    TropicalChannelManager,
    TropicalCoefficientExtractor,
    TropicalExponentExtractor,
    GPUChannelProcessor,
    TropicalChannels
)


def test_polynomial_channel_roundtrip():
    """Test converting polynomials to channels and back"""
    print("Testing polynomial ↔ channel conversion...")
    
    # Create complex polynomial
    monomials = [
        TropicalMonomial(1.5, {0: 3, 1: 2, 2: 1}),
        TropicalMonomial(2.7, {0: 1, 3: 4}),
        TropicalMonomial(0.8, {2: 2, 4: 1}),
        TropicalMonomial(5.2, {}),  # Constant
        TropicalMonomial(3.1, {1: 5}),
    ]
    original_poly = TropicalPolynomial(monomials, num_variables=5)
    
    # Convert to channels
    manager = TropicalChannelManager()
    channels = manager.polynomial_to_channels(original_poly)
    
    # Verify channel structure
    assert channels.coefficient_channel.shape[0] == 5
    assert channels.exponent_channel.shape == (5, 5)
    assert channels.metadata['num_variables'] == 5
    
    # Convert back
    reconstructed_poly = manager.channels_to_polynomial(channels)
    
    # Verify reconstruction
    assert reconstructed_poly.num_variables == original_poly.num_variables
    assert len(reconstructed_poly.monomials) == len(original_poly.monomials)
    
    # Test evaluation equivalence
    test_point = torch.tensor([1.0, 2.0, 0.5, 1.5, 3.0])
    orig_val = original_poly.evaluate(test_point)
    recon_val = reconstructed_poly.evaluate(test_point)
    assert abs(orig_val - recon_val) < 1e-6
    
    print("✅ Polynomial ↔ channel conversion successful")


def test_channel_compression():
    """Test coefficient and exponent compression"""
    print("\nTesting channel compression...")
    
    # Create polynomial with patterns
    # Arithmetic progression in coefficients
    monomials = [
        TropicalMonomial(2.0 + i * 3.0, {i: 2})
        for i in range(10)
    ]
    poly = TropicalPolynomial(monomials, num_variables=10)
    
    coeff_extractor = TropicalCoefficientExtractor()
    exp_extractor = TropicalExponentExtractor()
    
    # Test coefficient compression
    coeffs, coeff_meta = coeff_extractor.extract_with_compression(poly, compression_level=1)
    assert coeff_meta['compressed']
    if 'has_arithmetic_progression' in coeff_meta:
        assert coeff_meta['has_arithmetic_progression']
        print(f"  Detected arithmetic progression: start={coeff_meta.get('ap_start')}, step={coeff_meta.get('ap_step')}")
    
    # Test exponent compression
    exponents = exp_extractor.extract_exponents(poly)
    compressed_exp, exp_meta = exp_extractor.compress_exponents(exponents)
    assert exp_meta['sparsity'] > 0.8  # Should be very sparse
    print(f"  Exponent sparsity: {exp_meta['sparsity']:.2%}")
    
    print("✅ Channel compression working correctly")


def test_batch_operations():
    """Test batch processing of multiple polynomials"""
    print("\nTesting batch operations...")
    
    # Create multiple polynomials
    polynomials = []
    for i in range(100):
        monomials = [
            TropicalMonomial(float(j + i), {j % 3: (j + i) % 4})
            for j in range(5 + i % 10)
        ]
        polynomials.append(TropicalPolynomial(monomials, num_variables=5))
    
    manager = TropicalChannelManager()
    coeff_extractor = TropicalCoefficientExtractor()
    
    # Time batch extraction
    start = time.time()
    batch_coeffs = coeff_extractor.batch_extract(polynomials)
    batch_time = (time.time() - start) * 1000
    
    print(f"  Batch extracted {len(polynomials)} polynomials in {batch_time:.2f}ms")
    assert batch_coeffs.shape[0] == 100
    
    # Test channel merging
    channels_list = [manager.polynomial_to_channels(p) for p in polynomials[:10]]
    merged = manager.merge_channels(channels_list)
    assert merged.metadata['num_polynomials'] == 10
    
    print("✅ Batch operations successful")


def test_gpu_acceleration():
    """Test GPU acceleration if available"""
    if not torch.cuda.is_available():
        print("\n⚠️  GPU not available, skipping GPU tests")
        return
    
    print("\nTesting GPU acceleration...")
    
    device = torch.device('cuda')
    manager = TropicalChannelManager(device)
    processor = GPUChannelProcessor(device)
    
    # Create large polynomial
    monomials = [
        TropicalMonomial(float(i) * 0.1, {i % 20: (i % 5) + 1})
        for i in range(1000)
    ]
    poly = TropicalPolynomial(monomials, num_variables=20)
    
    # Convert to GPU channels
    channels = manager.polynomial_to_channels(poly)
    assert channels.device == device
    
    # Test GPU operations
    # 1. Coefficient normalization
    normalized = processor.process_coefficient_channel(channels.coefficient_channel, "normalize")
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0
    
    # 2. Exponent reduction
    reduced = processor.process_exponent_channel(channels.exponent_channel, "reduce")
    assert reduced.shape == channels.exponent_channel.shape
    
    # 3. Channel multiplication
    start = time.time()
    product = processor.parallel_channel_multiply(channels, channels)
    gpu_mult_time = (time.time() - start) * 1000
    print(f"  GPU multiplication of 1000×1000 monomials: {gpu_mult_time:.2f}ms")
    
    # 4. Channel addition
    start = time.time()
    sum_result = processor.parallel_channel_add(channels, channels)
    gpu_add_time = (time.time() - start) * 1000
    print(f"  GPU addition (concatenation): {gpu_add_time:.2f}ms")
    
    print("✅ GPU acceleration working correctly")


def test_memory_optimization():
    """Test memory layout optimization"""
    print("\nTesting memory optimization...")
    
    manager = TropicalChannelManager()
    
    # Create polynomial with varied coefficient magnitudes
    monomials = [
        TropicalMonomial(0.001, {0: 1}),
        TropicalMonomial(1000.0, {1: 2}),
        TropicalMonomial(0.1, {2: 1}),
        TropicalMonomial(50.0, {0: 1, 1: 1}),
        TropicalMonomial(0.5, {2: 2}),
    ]
    poly = TropicalPolynomial(monomials, num_variables=3)
    
    # Convert to channels
    channels = manager.polynomial_to_channels(poly)
    original_order = channels.coefficient_channel.clone()
    
    # Optimize layout
    optimized = manager.optimize_channel_layout(channels)
    
    # Verify optimization reordered by magnitude
    expected_order = torch.tensor([1000.0, 50.0, 0.5, 0.1, 0.001])
    assert torch.allclose(optimized.coefficient_channel, expected_order, rtol=1e-5)
    
    # Verify reconstruction still works
    reconstructed = manager.channels_to_polynomial(optimized)
    assert len(reconstructed.monomials) == len(poly.monomials)
    
    print("✅ Memory optimization successful")


def test_sparse_operations():
    """Test sparse exponent handling"""
    print("\nTesting sparse operations...")
    
    exp_extractor = TropicalExponentExtractor()
    
    # Create very sparse polynomial
    monomials = [
        TropicalMonomial(1.0, {0: 5}),
        TropicalMonomial(2.0, {50: 3}),
        TropicalMonomial(3.0, {99: 1}),
    ]
    poly = TropicalPolynomial(monomials, num_variables=100)
    
    # Extract sparse exponents
    indices, values = exp_extractor.extract_sparse_exponents(poly)
    
    # Should have only 3 non-zero entries
    assert values.shape[0] == 3
    assert indices.shape == (2, 3)
    
    # Test compression
    dense_exps = exp_extractor.extract_exponents(poly)
    compressed, metadata = exp_extractor.compress_exponents(dense_exps)
    
    assert metadata['sparsity'] > 0.95  # Should be extremely sparse
    assert metadata['format'] in ['coo_sparse', 'int8_dense', 'int16_dense']
    
    print(f"  Sparsity: {metadata['sparsity']:.2%}")
    print(f"  Compression format: {metadata['format']}")
    print("✅ Sparse operations successful")


def run_performance_benchmark():
    """Run performance benchmarks"""
    print("\n" + "="*50)
    print("Performance Benchmarks")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manager = TropicalChannelManager(device)
    
    # Benchmark different polynomial sizes
    sizes = [100, 500, 1000, 5000]
    
    for size in sizes:
        # Create polynomial
        monomials = [
            TropicalMonomial(float(i) * 0.01, {i % 50: (i % 5) + 1})
            for i in range(size)
        ]
        poly = TropicalPolynomial(monomials, num_variables=50)
        
        # Time conversion
        start = time.time()
        channels = manager.polynomial_to_channels(poly)
        conv_time = (time.time() - start) * 1000
        
        # Time reconstruction
        start = time.time()
        reconstructed = manager.channels_to_polynomial(channels)
        recon_time = (time.time() - start) * 1000
        
        print(f"\n{size} monomials:")
        print(f"  To channels: {conv_time:.2f}ms")
        print(f"  From channels: {recon_time:.2f}ms")
        print(f"  Total roundtrip: {conv_time + recon_time:.2f}ms")
        
        # Memory usage
        coeff_size = channels.coefficient_channel.element_size() * channels.coefficient_channel.nelement()
        exp_size = channels.exponent_channel.element_size() * channels.exponent_channel.nelement()
        total_size = (coeff_size + exp_size) / 1024  # KB
        print(f"  Channel memory: {total_size:.2f} KB")


def main():
    """Run all integration tests"""
    print("="*50)
    print("Tropical Channel Extraction Integration Tests")
    print("="*50)
    
    test_polynomial_channel_roundtrip()
    test_channel_compression()
    test_batch_operations()
    test_gpu_acceleration()
    test_memory_optimization()
    test_sparse_operations()
    run_performance_benchmark()
    
    print("\n" + "="*50)
    print("✅ All integration tests passed!")
    print("="*50)


if __name__ == "__main__":
    main()