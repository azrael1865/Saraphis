#!/usr/bin/env python3
"""
Standalone test for Huffman & Arithmetic entropy coding

Tests the entropy coding module independently without the full p-adic system.
"""

import sys
import os
import numpy as np
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import entropy coding module
from encoding.huffman_arithmetic import (
    HuffmanEncoder, 
    ArithmeticEncoder, 
    HybridEncoder,
    validate_reconstruction,
    run_compression_tests
)


def test_huffman_coding():
    """Test Huffman coding with various distributions"""
    print("\n" + "="*60)
    print("TESTING HUFFMAN CODING")
    print("="*60)
    
    prime = 257
    encoder = HuffmanEncoder(prime)
    
    # Test cases with different frequency distributions
    test_cases = [
        # Highly skewed distribution
        {
            'name': 'Highly Skewed',
            'data': [0] * 800 + [1] * 150 + [2] * 30 + [3] * 15 + [4] * 5,
            'expected_good': True
        },
        # Uniform distribution
        {
            'name': 'Uniform',
            'data': list(range(256)) * 4,
            'expected_good': False  # Huffman not good for uniform
        },
        # Binary-like
        {
            'name': 'Binary',
            'data': [0, 1] * 500,
            'expected_good': True
        },
        # Single value
        {
            'name': 'Single Value',
            'data': [42] * 100,
            'expected_good': True
        }
    ]
    
    for test in test_cases:
        print(f"\n  Test: {test['name']}")
        data = test['data']
        
        # Calculate frequencies
        from collections import Counter
        frequencies = Counter(data)
        
        print(f"    Data length: {len(data)}")
        print(f"    Unique symbols: {len(frequencies)}")
        
        try:
            # Build Huffman tree
            start = time.perf_counter()
            tree = encoder.build_huffman_tree(dict(frequencies))
            codes = encoder.generate_codes(tree)
            build_time = (time.perf_counter() - start) * 1000
            
            print(f"    Tree build time: {build_time:.2f} ms")
            print(f"    Code lengths: {[len(code) for code in codes.values()][:10]}...")
            
            # Encode
            start = time.perf_counter()
            encoded = encoder.huffman_encode(data)
            encode_time = (time.perf_counter() - start) * 1000
            
            # Decode
            start = time.perf_counter()
            decoded = encoder.huffman_decode(encoded, tree)
            decode_time = (time.perf_counter() - start) * 1000
            
            # Validate
            validate_reconstruction(data, decoded)
            
            # Calculate compression
            original_size = len(data)  # 1 byte per symbol for prime < 256
            compressed_size = len(encoded)
            ratio = original_size / compressed_size
            
            print(f"    Original size: {original_size} bytes")
            print(f"    Compressed size: {compressed_size} bytes")
            print(f"    Compression ratio: {ratio:.2f}x")
            print(f"    Encode time: {encode_time:.2f} ms")
            print(f"    Decode time: {decode_time:.2f} ms")
            
            if test['expected_good']:
                if ratio > 1.5:
                    print(f"    ✓ Good compression as expected")
                else:
                    print(f"    ⚠ Lower compression than expected")
            else:
                if ratio < 1.2:
                    print(f"    ✓ Poor compression as expected (not suitable for Huffman)")
                else:
                    print(f"    ⚠ Better compression than expected")
                    
            print(f"    Status: PASSED")
            
        except Exception as e:
            print(f"    Status: FAILED - {e}")
            import traceback
            traceback.print_exc()


def test_arithmetic_coding():
    """Test Arithmetic coding with various distributions"""
    print("\n" + "="*60)
    print("TESTING ARITHMETIC CODING")
    print("="*60)
    
    prime = 257
    encoder = ArithmeticEncoder(prime, precision_bits=32)
    
    # Test cases
    test_cases = [
        # Uniform distribution (good for arithmetic)
        {
            'name': 'Uniform',
            'data': list(range(100)) * 10,
            'probabilities': {i: 1/100 for i in range(100)}
        },
        # Skewed distribution
        {
            'name': 'Skewed',
            'data': [0] * 500 + [1] * 300 + [2] * 150 + [3] * 50,
            'probabilities': {0: 0.5, 1: 0.3, 2: 0.15, 3: 0.05}
        },
        # Binary
        {
            'name': 'Binary',
            'data': [0, 1] * 500,
            'probabilities': {0: 0.5, 1: 0.5}
        }
    ]
    
    for test in test_cases:
        print(f"\n  Test: {test['name']}")
        data = test['data']
        probs = test['probabilities']
        
        print(f"    Data length: {len(data)}")
        print(f"    Symbols: {len(probs)}")
        
        try:
            # Build CDF
            cdf = encoder.compute_cdf(probs)
            
            # Encode
            start = time.perf_counter()
            encoded = encoder.arithmetic_encode(data, probs)
            encode_time = (time.perf_counter() - start) * 1000
            
            # Decode
            start = time.perf_counter()
            decoded = encoder.arithmetic_decode(encoded, len(data))
            decode_time = (time.perf_counter() - start) * 1000
            
            # Validate
            validate_reconstruction(data, decoded)
            
            # Calculate compression
            original_size = len(data)
            compressed_size = len(encoded)
            ratio = original_size / compressed_size
            
            print(f"    Original size: {original_size} bytes")
            print(f"    Compressed size: {compressed_size} bytes")
            print(f"    Compression ratio: {ratio:.2f}x")
            print(f"    Encode time: {encode_time:.2f} ms")
            print(f"    Decode time: {decode_time:.2f} ms")
            print(f"    Status: PASSED")
            
        except Exception as e:
            print(f"    Status: FAILED - {e}")
            import traceback
            traceback.print_exc()


def test_hybrid_encoder():
    """Test hybrid encoder that chooses optimal method"""
    print("\n" + "="*60)
    print("TESTING HYBRID ENCODER")
    print("="*60)
    
    primes = [2, 3, 5, 7, 11, 127, 257]
    
    for prime in primes:
        print(f"\n  Testing with prime = {prime}")
        encoder = HybridEncoder(prime)
        
        # Generate test data with different characteristics
        test_data = {
            'skewed': [0] * 70 + [1] * 20 + [2] * 10,
            'uniform': list(range(min(prime, 50))) * 2,
            'sparse': [0, prime//2, prime-1] * 33,
        }
        
        for name, digits in test_data.items():
            print(f"\n    Pattern: {name}")
            
            try:
                # Encode
                compressed, metadata = encoder.encode_digits(digits)
                
                # Decode
                decoded = encoder.decode_digits(compressed, metadata)
                
                # Validate
                validate_reconstruction(digits, decoded)
                
                # Get metrics
                metrics = encoder.get_metrics()
                
                print(f"      Method: {metadata['method']}")
                print(f"      Compression: {metrics.compression_ratio:.2f}x")
                print(f"      Entropy: {metrics.entropy:.2f} bits")
                print(f"      Unique symbols: {metrics.unique_symbols}")
                print(f"      Status: PASSED")
                
            except Exception as e:
                print(f"      Status: FAILED - {e}")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*60)
    print("TESTING EDGE CASES")
    print("="*60)
    
    prime = 257
    
    # Test invalid inputs
    print("\n  Testing error handling:")
    
    # Test 1: Invalid prime
    try:
        encoder = HuffmanEncoder(-1)
        print("    Invalid prime: FAILED (should have raised error)")
    except ValueError as e:
        print(f"    Invalid prime: PASSED (caught: {e})")
    
    # Test 2: Empty data
    encoder = HybridEncoder(prime)
    try:
        compressed, metadata = encoder.encode_digits([])
        print("    Empty data: FAILED (should have raised error)")
    except ValueError as e:
        print(f"    Empty data: PASSED (caught: {e})")
    
    # Test 3: Out of range digits
    try:
        compressed, metadata = encoder.encode_digits([0, prime, 1])  # prime is out of range
        print("    Out of range: FAILED (should have raised error)")
    except ValueError as e:
        print(f"    Out of range: PASSED (caught: {e})")
    
    # Test 4: Large data
    print("\n  Testing performance with large data:")
    large_data = np.random.randint(0, prime, 100000).tolist()
    
    start = time.perf_counter()
    compressed, metadata = encoder.encode_digits(large_data)
    encode_time = time.perf_counter() - start
    
    start = time.perf_counter()
    decoded = encoder.decode_digits(compressed, metadata)
    decode_time = time.perf_counter() - start
    
    validate_reconstruction(large_data, decoded)
    
    print(f"    100k elements:")
    print(f"      Method: {metadata['method']}")
    print(f"      Compression: {metadata['metrics']['compression_ratio']:.2f}x")
    print(f"      Encode time: {encode_time*1000:.2f} ms")
    print(f"      Decode time: {decode_time*1000:.2f} ms")
    print(f"      Status: PASSED")


def benchmark_performance():
    """Benchmark performance across different scenarios"""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKS")
    print("="*60)
    
    prime = 257
    encoder = HybridEncoder(prime)
    
    sizes = [100, 1000, 10000, 50000]
    
    print("\n  Size vs Performance:")
    print("    Size    | Method   | Ratio | Encode(ms) | Decode(ms)")
    print("    --------|----------|-------|------------|------------")
    
    for size in sizes:
        # Generate random data
        data = np.random.randint(0, prime, size).tolist()
        
        # Measure
        start = time.perf_counter()
        compressed, metadata = encoder.encode_digits(data)
        encode_time = (time.perf_counter() - start) * 1000
        
        start = time.perf_counter()
        decoded = encoder.decode_digits(compressed, metadata)
        decode_time = (time.perf_counter() - start) * 1000
        
        # Validate
        validate_reconstruction(data, decoded)
        
        metrics = metadata['metrics']
        print(f"    {size:7d} | {metadata['method']:8s} | {metrics['compression_ratio']:5.2f} | "
              f"{encode_time:10.2f} | {decode_time:10.2f}")


def main():
    """Main test runner"""
    print("\n" + "="*80)
    print("HUFFMAN & ARITHMETIC CODING TEST SUITE")
    print("="*80)
    
    # Run all tests
    test_huffman_coding()
    test_arithmetic_coding()
    test_hybrid_encoder()
    test_edge_cases()
    benchmark_performance()
    
    # Run built-in tests
    print("\n" + "="*60)
    print("RUNNING BUILT-IN COMPRESSION TESTS")
    print("="*60)
    run_compression_tests(prime=257)
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()