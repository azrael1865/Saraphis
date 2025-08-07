#!/usr/bin/env python3
"""
Demo script for P-adic compression system
Shows compression of neural network weights
"""

import torch
import numpy as np
import sys
import os

# Add path for imports
sys.path.append('/Users/will/Desktop/trueSaraphis')

from independent_core.compression_systems.padic.padic_compressor import PadicCompressionSystem
from independent_core.compression_systems.encoding.huffman_arithmetic import HybridEncoder

def demo_compression():
    """Demonstrate P-adic compression with entropy coding"""
    
    print("=" * 60)
    print("P-ADIC COMPRESSION SYSTEM DEMO")
    print("=" * 60)
    
    # Create test tensor (simulating neural network weights)
    print("\n1. Creating test tensor (neural network weights)...")
    test_tensor = torch.randn(1000, 100) * 0.1  # 100K parameters
    test_tensor[test_tensor.abs() < 0.05] = 0  # Make sparse
    
    original_size = test_tensor.numel() * 4 / 1024  # Float32 = 4 bytes
    sparsity = (test_tensor == 0).float().mean()
    
    print(f"   Shape: {test_tensor.shape}")
    print(f"   Original size: {original_size:.2f} KB")
    print(f"   Sparsity: {sparsity:.2%}")
    
    # Configure P-adic compression
    print("\n2. Configuring P-adic compression...")
    config = {
        'prime': 7,  # Smaller prime for higher precision
        'precision': 12,  # Safe precision for prime 7
        'chunk_size': 1000,
        'gpu_memory_limit_mb': 1000,
        'enable_entropy_coding': True,  # Enable Huffman/Arithmetic
        'preserve_ultrametric': True,
        'validate_reconstruction': True,
        'max_reconstruction_error': 1e-4
    }
    
    print(f"   Prime: {config['prime']}")
    print(f"   Precision: {config['precision']}")
    print(f"   Entropy coding: {config['enable_entropy_coding']}")
    
    # Initialize compression system
    print("\n3. Initializing compression system...")
    compressor = PadicCompressionSystem(config)
    
    # Flatten tensor for compression
    flat_tensor = test_tensor.flatten().numpy()
    
    # Compress
    print("\n4. Compressing...")
    compressed_data = compressor.compress(flat_tensor)
    
    # Calculate compressed size
    if isinstance(compressed_data, dict):
        # Calculate total size of compressed components
        total_compressed_size = 0
        
        if 'encoded_digits' in compressed_data:
            # Entropy-coded path
            encoded = compressed_data['encoded_digits']
            if isinstance(encoded, bytes):
                total_compressed_size = len(encoded) / 1024
            else:
                total_compressed_size = encoded.nbytes / 1024
        else:
            # Standard p-adic weights
            for key, value in compressed_data.items():
                if hasattr(value, 'nbytes'):
                    total_compressed_size += value.nbytes / 1024
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        if hasattr(item, 'nbytes'):
                            total_compressed_size += item.nbytes / 1024
    else:
        total_compressed_size = 0
        if hasattr(compressed_data, '__len__'):
            for weight in compressed_data:
                if hasattr(weight, 'digits'):
                    total_compressed_size += weight.digits.nbytes / 1024
    
    compression_ratio = original_size / max(total_compressed_size, 0.001)
    
    print(f"   Compressed size: {total_compressed_size:.2f} KB")
    print(f"   Compression ratio: {compression_ratio:.2f}x")
    
    # Decompress
    print("\n5. Decompressing...")
    decompressed = compressor.decompress(compressed_data)
    
    # Reshape to original shape
    if isinstance(decompressed, np.ndarray):
        reconstructed = torch.from_numpy(decompressed).reshape(test_tensor.shape)
    else:
        reconstructed = torch.tensor(decompressed).reshape(test_tensor.shape)
    
    # Calculate reconstruction error
    print("\n6. Validating reconstruction...")
    mse = ((reconstructed - test_tensor) ** 2).mean().item()
    max_error = (reconstructed - test_tensor).abs().max().item()
    
    print(f"   MSE: {mse:.6e}")
    print(f"   Max error: {max_error:.6e}")
    print(f"   Reconstruction valid: {max_error < config['max_reconstruction_error']}")
    
    # Performance stats
    print("\n7. Performance Statistics:")
    stats = compressor.get_stats()
    print(f"   Compression time: {stats.get('compression_time_ms', 0):.2f} ms")
    print(f"   Decompression time: {stats.get('decompression_time_ms', 0):.2f} ms")
    print(f"   GPU memory used: {stats.get('gpu_memory_mb', 0):.2f} MB")
    
    # Entropy coding stats (if enabled)
    if config['enable_entropy_coding']:
        print("\n8. Entropy Coding Statistics:")
        encoder = HybridEncoder()
        flat_digits = np.random.randint(0, config['prime'], size=10000)
        encoded = encoder.encode_digits(flat_digits, config['prime'])
        
        entropy_ratio = len(encoded) / (flat_digits.nbytes)
        print(f"   Entropy compression: {1/entropy_ratio:.2f}x additional")
        print(f"   Combined compression: {compression_ratio:.2f}x total")
    
    print("\n" + "=" * 60)
    print("COMPRESSION DEMO COMPLETE")
    print("=" * 60)
    
    return compression_ratio, mse

if __name__ == "__main__":
    try:
        ratio, error = demo_compression()
        print(f"\nFinal Results: {ratio:.2f}x compression, {error:.6e} MSE")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()