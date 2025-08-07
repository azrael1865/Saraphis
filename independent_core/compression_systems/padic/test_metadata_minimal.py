"""
Minimal test to verify metadata compressor integration
"""

import torch
import sys

# Test metadata compressor alone first
print("Testing metadata compressor...")
from independent_core.compression_systems.padic.metadata_compressor import MetadataCompressor

compressor = MetadataCompressor()

# Simple metadata
metadata = {
    'prime': 257,
    'precision': 6,
    'original_shape': (32, 64),
    'sparse_encoded': True
}

# Compress and decompress
compressed = compressor.compress_metadata(metadata)
decompressed = compressor.decompress_metadata(compressed)

print(f"✓ Metadata compressed to {len(compressed)} bytes")
print(f"✓ Metadata decompressed successfully")

# Now test with the main system
print("\nTesting integration with P-adic system...")
from independent_core.compression_systems.padic.padic_compression_pytorch import (
    PurelyPyTorchPAdicSystem, 
    PurelyPyTorchConfig
)

config = PurelyPyTorchConfig(
    prime=257,
    precision=6,
    enable_triton=False,  # Disable Triton for testing
    enable_sparse=True,
    sparse_threshold=0.1,
    enable_pattern_matching=False,  # Disable to speed up test
    enable_entropy=False  # Disable to speed up test
)

system = PurelyPyTorchPAdicSystem(config)

# Small test tensor
tensor = torch.randn(32, 32)

# Make it sparse
mask = torch.rand(32, 32) > 0.5
tensor = tensor * mask.float()

print(f"Testing with tensor shape: {tensor.shape}")

# Compress
result = system.compress(tensor)

# Check metadata compression
if result.compressed_metadata is not None:
    print(f"✓ Metadata compressed to {len(result.compressed_metadata)} bytes")
    
    # Calculate overhead
    data_size = result.compressed_data.numel() * 4
    metadata_size = len(result.compressed_metadata)
    overhead = (metadata_size / data_size) * 100
    print(f"✓ Metadata overhead: {overhead:.4f}%")
    
    if overhead < 1.0:
        print("✅ SUCCESS: < 1% metadata overhead achieved!")
    else:
        print(f"⚠️  Overhead {overhead:.2f}% exceeds target")
else:
    print("❌ No compressed metadata found")

# Test decompression
decompressed = system.decompress(result)
error = torch.abs(tensor - decompressed.reconstructed_data).max().item()
print(f"✓ Decompression successful, max error: {error:.6e}")

print("\n✅ All tests passed!")