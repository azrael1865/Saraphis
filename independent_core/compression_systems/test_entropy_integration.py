"""
Test script for entropy coding integration with p-adic compression system

Demonstrates the complete compression pipeline with optional entropy coding.
"""

import torch
import numpy as np
from typing import Dict, Any
import time

# Add path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import p-adic compression system
from padic.padic_compressor import PadicCompressionSystem

# Import entropy coding components
from encoding.huffman_arithmetic import HybridEncoder, validate_reconstruction


def test_entropy_coding_integration():
    """Test the integration of entropy coding with p-adic compression"""
    
    print("\n" + "="*80)
    print("TESTING ENTROPY CODING INTEGRATION WITH P-ADIC COMPRESSION")
    print("="*80)
    
    # Create test data - neural network weights
    torch.manual_seed(42)
    test_tensors = {
        'small_uniform': torch.randn(100, 10),
        'large_sparse': torch.randn(1000, 100) * 0.1,  # Sparse weights
        'structured': torch.cat([torch.zeros(500), torch.ones(500)]),  # Structured pattern
    }
    
    # Test with different prime values
    primes = [2, 3, 5, 7, 257]
    
    for prime in primes:
        print(f"\n{'='*60}")
        print(f"Testing with prime = {prime}")
        print(f"{'='*60}")
        
        # Test WITHOUT entropy coding
        config_no_entropy = {
            'prime': prime,
            'precision': 3,
            'chunk_size': 100,
            'gpu_memory_limit_mb': 1024,
            'enable_entropy_coding': False,
            'validate_reconstruction': True,
            'max_reconstruction_error': 1e-5
        }
        
        compressor_no_entropy = PadicCompressionSystem(config_no_entropy)
        
        # Test WITH entropy coding
        config_with_entropy = config_no_entropy.copy()
        config_with_entropy['enable_entropy_coding'] = True
        
        compressor_with_entropy = PadicCompressionSystem(config_with_entropy)
        
        for name, tensor in test_tensors.items():
            print(f"\nTesting tensor: {name}")
            print(f"  Shape: {tensor.shape}")
            print(f"  Size: {tensor.numel()} elements")
            
            try:
                # Compress without entropy coding
                start = time.perf_counter()
                result_no_entropy = compressor_no_entropy.compress(tensor)
                time_no_entropy = time.perf_counter() - start
                
                # Compress with entropy coding
                start = time.perf_counter()
                result_with_entropy = compressor_with_entropy.compress(tensor)
                time_with_entropy = time.perf_counter() - start
                
                # Compare results
                print(f"\n  Without Entropy Coding:")
                print(f"    Compressed size: {result_no_entropy['compressed_size']} bytes")
                print(f"    Compression ratio: {result_no_entropy['compression_ratio']:.2f}x")
                print(f"    Compression time: {time_no_entropy*1000:.2f} ms")
                
                print(f"\n  With Entropy Coding:")
                print(f"    Compressed size: {result_with_entropy['compressed_size']} bytes")
                print(f"    Compression ratio: {result_with_entropy['compression_ratio']:.2f}x")
                print(f"    Compression time: {time_with_entropy*1000:.2f} ms")
                
                if result_with_entropy['metadata']['entropy_coding']['enabled']:
                    entropy_info = result_with_entropy['metadata']['entropy_coding']
                    print(f"    Entropy method: {entropy_info['method']}")
                    print(f"    Original digits: {entropy_info['original_digits']}")
                    print(f"    Entropy compressed: {entropy_info['compressed_bytes']} bytes")
                    
                    metrics = entropy_info['entropy_metrics']
                    print(f"    Entropy: {metrics['entropy']:.2f} bits")
                    print(f"    Unique symbols: {metrics['unique_symbols']}")
                    print(f"    Entropy compression ratio: {metrics['compression_ratio']:.2f}x")
                
                # Test decompression
                print(f"\n  Testing decompression...")
                
                # Decompress without entropy
                start = time.perf_counter()
                reconstructed_no_entropy = compressor_no_entropy.decompress(result_no_entropy)
                decomp_time_no_entropy = time.perf_counter() - start
                
                # Decompress with entropy
                start = time.perf_counter()
                reconstructed_with_entropy = compressor_with_entropy.decompress(result_with_entropy)
                decomp_time_with_entropy = time.perf_counter() - start
                
                # Validate reconstruction
                error_no_entropy = torch.mean(torch.abs(tensor - reconstructed_no_entropy)).item()
                error_with_entropy = torch.mean(torch.abs(tensor - reconstructed_with_entropy)).item()
                
                print(f"    Without entropy - Error: {error_no_entropy:.2e}, Time: {decomp_time_no_entropy*1000:.2f} ms")
                print(f"    With entropy - Error: {error_with_entropy:.2e}, Time: {decomp_time_with_entropy*1000:.2f} ms")
                
                # Check if errors are acceptable
                max_error = 1e-4
                if error_no_entropy > max_error:
                    print(f"    WARNING: Reconstruction error without entropy too high!")
                if error_with_entropy > max_error:
                    print(f"    WARNING: Reconstruction error with entropy too high!")
                
                # Calculate improvement
                if result_with_entropy['compressed_size'] < result_no_entropy['compressed_size']:
                    improvement = (1 - result_with_entropy['compressed_size'] / result_no_entropy['compressed_size']) * 100
                    print(f"\n  IMPROVEMENT: Entropy coding reduced size by {improvement:.1f}%")
                else:
                    print(f"\n  NOTE: Entropy coding did not improve compression for this data")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()


def test_standalone_entropy_coding():
    """Test entropy coding module standalone"""
    
    print("\n" + "="*80)
    print("TESTING STANDALONE ENTROPY CODING")
    print("="*80)
    
    # Test with different p-adic primes
    primes = [2, 3, 5, 7, 11, 257]
    
    for prime in primes:
        print(f"\n{'='*60}")
        print(f"Testing standalone entropy coding with prime = {prime}")
        print(f"{'='*60}")
        
        encoder = HybridEncoder(prime)
        
        # Generate test digit sequences
        test_sequences = {
            'uniform': np.random.randint(0, prime, 1000).tolist(),
            'skewed': [0] * 500 + [1] * 300 + list(range(2, min(10, prime))) * 25,
            'sparse': [0, prime//2, prime-1] * 333,
        }
        
        for name, digits in test_sequences.items():
            print(f"\n  Testing sequence: {name}")
            print(f"    Length: {len(digits)}")
            print(f"    Unique values: {len(set(digits))}")
            
            try:
                # Encode
                compressed, metadata = encoder.encode_digits(digits)
                
                # Decode
                reconstructed = encoder.decode_digits(compressed, metadata)
                
                # Validate
                validate_reconstruction(digits, reconstructed)
                
                # Report metrics
                metrics = metadata['metrics']
                print(f"    Method chosen: {metadata['method']}")
                print(f"    Original size: {metrics['original_size']} bytes")
                print(f"    Compressed size: {metrics['compressed_size']} bytes")
                print(f"    Compression ratio: {metrics['compression_ratio']:.2f}x")
                print(f"    Entropy: {metrics['entropy']:.2f} bits")
                print(f"    Encoding time: {metrics['encoding_time_ms']:.2f} ms")
                print(f"    Status: PASSED")
                
            except Exception as e:
                print(f"    Status: FAILED - {e}")


def main():
    """Main test function"""
    
    # Test standalone entropy coding first
    test_standalone_entropy_coding()
    
    # Then test full integration
    test_entropy_coding_integration()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()