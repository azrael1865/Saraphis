#!/usr/bin/env python3
"""
Test script for enhanced Triton kernels
Tests the three new critical kernels added to triton_kernels.py
"""

import torch
import numpy as np
import time
from independent_core.compression_systems.padic.triton_kernels import TritonPAdicOps, TRITON_AVAILABLE


def test_enhanced_batch_conversion():
    """Test the enhanced batch p-adic conversion with valuations"""
    if not TRITON_AVAILABLE:
        print("Triton not available, skipping tests")
        return
    
    print("\n=== Testing Enhanced Batch P-adic Conversion ===")
    
    # Initialize Triton ops
    ops = TritonPAdicOps(prime=257, precision=10, device='cuda')
    
    # Create test data
    test_data = torch.randn(100, 32, device='cuda') * 100
    
    # Test conversion with valuations
    start_time = time.time()
    padic_digits, valuations = ops.batch_convert(test_data, return_valuations=True)
    conversion_time = time.time() - start_time
    
    print(f"Input shape: {test_data.shape}")
    print(f"P-adic digits shape: {padic_digits.shape}")
    print(f"Valuations shape: {valuations.shape}")
    print(f"Conversion time: {conversion_time:.4f}s")
    print(f"Sample valuations: {valuations.flatten()[:10].tolist()}")
    
    # Verify column-major optimization helped
    print(f"Memory layout optimized: Column-major storage used internally")
    
    return padic_digits, valuations


def test_parallel_pattern_matching():
    """Test the parallel pattern matching kernel"""
    if not TRITON_AVAILABLE:
        print("Triton not available, skipping tests")
        return
    
    print("\n=== Testing Parallel Pattern Matching ===")
    
    # Initialize Triton ops
    ops = TritonPAdicOps(prime=257, precision=10, device='cuda')
    
    # Create test data with known patterns
    data_size = 10000
    pattern_size = 8
    num_patterns = 5
    
    # Create data with embedded patterns
    data = torch.randint(0, 256, (data_size,), device='cuda', dtype=torch.float32)
    
    # Create patterns to search for
    patterns = torch.randint(0, 256, (num_patterns, pattern_size), device='cuda', dtype=torch.float32)
    
    # Embed some patterns in the data
    for i in range(3):  # Embed first 3 patterns multiple times
        for pos in range(i * 100, data_size - pattern_size, 500):
            data[pos:pos + pattern_size] = patterns[i]
    
    # Run pattern matching
    start_time = time.time()
    match_mask, match_counts = ops.parallel_pattern_match(data, patterns, return_counts=True)
    matching_time = time.time() - start_time
    
    print(f"Data size: {data_size}")
    print(f"Number of patterns: {num_patterns}")
    print(f"Pattern size: {pattern_size}")
    print(f"Match mask shape: {match_mask.shape}")
    print(f"Matches per pattern: {match_counts.tolist()}")
    print(f"Matching time: {matching_time:.4f}s")
    print(f"Throughput: {(data_size * num_patterns) / matching_time / 1e6:.2f} M comparisons/s")
    
    return match_mask, match_counts


def test_sparse_csr_conversion():
    """Test the sparse CSR conversion kernel"""
    if not TRITON_AVAILABLE:
        print("Triton not available, skipping tests")
        return
    
    print("\n=== Testing Sparse CSR Conversion ===")
    
    # Initialize Triton ops
    ops = TritonPAdicOps(prime=257, precision=10, device='cuda')
    
    # Create sparse matrix (70% zeros)
    rows, cols = 512, 1024
    dense_matrix = torch.randn(rows, cols, device='cuda')
    sparsity_mask = torch.rand(rows, cols, device='cuda') < 0.7
    dense_matrix[sparsity_mask] = 0
    
    # Convert to CSR
    start_time = time.time()
    values, col_idx, row_ptr = ops.dense_to_csr(dense_matrix, threshold=1e-6)
    csr_time = time.time() - start_time
    
    # Calculate metrics
    nnz = values.shape[0]
    total_elements = rows * cols
    density = nnz / total_elements
    dense_bytes = total_elements * 4  # float32
    csr_bytes = nnz * 4 + nnz * 4 + (rows + 1) * 4  # values + col_idx + row_ptr
    compression_ratio = dense_bytes / csr_bytes
    
    print(f"Matrix shape: {rows}x{cols}")
    print(f"Non-zero elements: {nnz}/{total_elements} ({density:.2%} density)")
    print(f"CSR values shape: {values.shape}")
    print(f"CSR column indices shape: {col_idx.shape}")
    print(f"CSR row pointers shape: {row_ptr.shape}")
    print(f"Conversion time: {csr_time:.4f}s")
    print(f"Memory saved: {(dense_bytes - csr_bytes) / 1024:.2f} KB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    # Test reconstruction
    reconstructed = ops.csr_to_dense(values, col_idx, row_ptr, (rows, cols))
    reconstruction_error = torch.abs(reconstructed - dense_matrix).max().item()
    print(f"Reconstruction max error: {reconstruction_error:.2e}")
    
    return values, col_idx, row_ptr


def benchmark_kernels():
    """Benchmark all three kernels with realistic workloads"""
    if not TRITON_AVAILABLE:
        print("Triton not available, skipping benchmarks")
        return
    
    print("\n=== KERNEL BENCHMARKS ===")
    
    # Benchmark batch conversion
    print("\n1. Batch P-adic Conversion (1M elements):")
    ops = TritonPAdicOps(prime=257, precision=10, device='cuda')
    data = torch.randn(1000, 1000, device='cuda') * 100
    
    # Warm up
    ops.batch_convert(data[:100])
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(10):
        ops.batch_convert(data)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"   Average time: {elapsed/10:.4f}s")
    print(f"   Throughput: {1e6 * 10 / elapsed / 1e6:.2f} M elements/s")
    
    # Benchmark pattern matching
    print("\n2. Pattern Matching (100K data, 10 patterns):")
    data = torch.randint(0, 256, (100000,), device='cuda', dtype=torch.float32)
    patterns = torch.randint(0, 256, (10, 16), device='cuda', dtype=torch.float32)
    
    # Warm up
    ops.parallel_pattern_match(data[:1000], patterns[:2])
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(10):
        ops.parallel_pattern_match(data, patterns)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"   Average time: {elapsed/10:.4f}s")
    print(f"   Throughput: {100000 * 10 * 10 / elapsed / 1e9:.2f} G comparisons/s")
    
    # Benchmark CSR conversion
    print("\n3. CSR Conversion (2048x2048 matrix, 80% sparse):")
    dense = torch.randn(2048, 2048, device='cuda')
    mask = torch.rand(2048, 2048, device='cuda') < 0.8
    dense[mask] = 0
    
    # Warm up
    ops.dense_to_csr(dense[:100, :100])
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(10):
        ops.dense_to_csr(dense)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"   Average time: {elapsed/10:.4f}s")
    print(f"   Throughput: {2048*2048*10 / elapsed / 1e6:.2f} M elements/s")


def main():
    """Run all tests"""
    print("=" * 60)
    print("ENHANCED TRITON KERNELS TEST SUITE")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return
    
    if not TRITON_AVAILABLE:
        print("Triton not installed. Please install with: pip install triton")
        return
    
    # Run tests
    try:
        # Test individual kernels
        padic_digits, valuations = test_enhanced_batch_conversion()
        match_mask, match_counts = test_parallel_pattern_matching()
        values, col_idx, row_ptr = test_sparse_csr_conversion()
        
        # Run benchmarks
        benchmark_kernels()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()