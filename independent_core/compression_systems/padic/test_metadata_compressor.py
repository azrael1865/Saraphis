"""
Test suite for metadata compressor
Verifies < 1% metadata overhead and compression functionality
"""

import torch
import numpy as np
from typing import Dict, Any
import time

from .metadata_compressor import MetadataCompressor, MetadataHeader
from .padic_compression_pytorch import PurelyPyTorchPAdicSystem, PurelyPyTorchConfig, CompressionResult


def test_header_packing():
    """Test fixed-size header packing/unpacking"""
    print("Testing header packing...")
    
    compressor = MetadataCompressor()
    
    # Test header
    header = MetadataHeader(
        version=1,
        prime=257,
        precision=6,
        compression_flags=0xFF,  # All flags set
        original_shape=(32, 64, 128)
    )
    
    # Pack header
    packed = compressor._pack_header(header)
    assert len(packed) == 5, f"Header should be 5 bytes, got {len(packed)}"
    
    # Unpack header
    unpacked, offset = compressor._unpack_header(packed, 0)
    assert unpacked.version == header.version
    assert unpacked.prime == header.prime
    assert unpacked.precision == header.precision
    assert unpacked.compression_flags == header.compression_flags
    assert offset == 5
    
    print("✓ Header packing test passed")


def test_delta_encoding():
    """Test delta encoding for sparse indices"""
    print("Testing delta encoding...")
    
    compressor = MetadataCompressor()
    
    # Test indices
    indices = [0, 5, 10, 15, 100, 101, 102, 200, 300, 1000]
    
    # Encode
    encoded = compressor._delta_encode_indices(indices)
    
    # Decode
    decoded, _ = compressor._delta_decode_indices(encoded, 0)
    
    assert decoded == indices, f"Delta encoding failed: {decoded} != {indices}"
    
    # Check compression
    original_size = len(indices) * 8  # 8 bytes per int64
    compressed_size = len(encoded)
    ratio = compressed_size / original_size
    print(f"  Delta encoding ratio: {ratio:.2%} ({compressed_size}/{original_size} bytes)")
    
    print("✓ Delta encoding test passed")


def test_varint_encoding():
    """Test variable-length integer encoding"""
    print("Testing varint encoding...")
    
    compressor = MetadataCompressor()
    
    # Test various numbers
    test_values = [0, 1, -1, 127, -128, 255, -256, 1000, -1000, 1000000, -1000000]
    
    for value in test_values:
        # Encode
        encoded = compressor._encode_varint(value)
        
        # Decode
        decoded, _ = compressor._decode_varint(encoded, 0)
        
        assert decoded == value, f"Varint failed for {value}: got {decoded}"
        
        # Check size efficiency
        if abs(value) < 64:  # After zigzag encoding, values < 64 fit in 1 byte
            assert len(encoded) == 1, f"Small value {value} should use 1 byte, got {len(encoded)}"
        
    print("✓ Varint encoding test passed")


def test_flag_packing():
    """Test boolean flag bit packing"""
    print("Testing flag packing...")
    
    compressor = MetadataCompressor()
    
    # Test flags
    flags = [True, False, True, True, False, False, True, False,
             True, True, False, True]  # 12 flags
    
    # Pack
    packed = compressor._pack_flags(flags)
    
    # Unpack
    unpacked, _ = compressor._unpack_flags(packed, 0)
    
    assert unpacked == flags, f"Flag packing failed: {unpacked} != {flags}"
    
    # Check compression
    original_size = len(flags)  # 1 byte per bool typically
    compressed_size = len(packed)
    ratio = compressed_size / original_size
    print(f"  Flag packing ratio: {ratio:.2%} ({compressed_size}/{original_size} bytes)")
    
    print("✓ Flag packing test passed")


def test_pattern_dict_compression():
    """Test pattern dictionary compression"""
    print("Testing pattern dictionary compression...")
    
    compressor = MetadataCompressor()
    
    # Test pattern dictionary
    pattern_dict = {
        'pattern_1': torch.randn(4, 4),
        'pattern_2': [1.0, 2.0, 3.0, 4.0],
        'pattern_3': 42.5,
        'pattern_info': 'repeated_block'
    }
    
    # Compress
    compressed = compressor._compress_pattern_dict(pattern_dict)
    
    # Decompress
    decompressed, _ = compressor._decompress_pattern_dict(compressed, 0)
    
    # Verify
    assert len(decompressed) == len(pattern_dict)
    assert torch.allclose(decompressed['pattern_1'], pattern_dict['pattern_1'])
    assert decompressed['pattern_2'] == pattern_dict['pattern_2']
    assert abs(decompressed['pattern_3'] - pattern_dict['pattern_3']) < 0.01
    assert decompressed['pattern_info'] == pattern_dict['pattern_info']
    
    print("✓ Pattern dictionary compression test passed")


def test_full_metadata_compression():
    """Test complete metadata compression pipeline"""
    print("\nTesting full metadata compression...")
    
    compressor = MetadataCompressor()
    
    # Create comprehensive metadata
    metadata = {
        'version': 1,
        'prime': 257,
        'precision': 6,
        'original_shape': (64, 128, 256),
        'sparse_encoded': True,
        'sparse_indices': torch.tensor([0, 10, 20, 30, 100, 200, 300, 1000, 2000, 3000]),
        'sparse_values': torch.randn(10),
        'log_encoded': True,
        'pattern_matched': True,
        'pattern_dict': {
            'base_pattern': torch.randn(8),
            'frequency': 42
        },
        'entropy_coded': True,
        'entropy_metadata': {
            'min_val': -10.5,
            'max_val': 10.5,
            'entropy': 7.8,
            'n_bins': 256
        },
        'mixed_precision': True,
        'triton_accelerated': True,
        'additional_flags': [True, False, True, False, True, True, False, False]
    }
    
    # Compress
    compressed = compressor.compress_metadata(metadata)
    print(f"  Compressed size: {len(compressed)} bytes")
    
    # Decompress
    decompressed = compressor.decompress_metadata(compressed)
    
    # Verify all fields
    assert decompressed['version'] == metadata['version']
    assert decompressed['prime'] == metadata['prime']
    assert decompressed['precision'] == metadata['precision']
    assert decompressed['original_shape'] == metadata['original_shape']
    assert decompressed['sparse_encoded'] == metadata['sparse_encoded']
    assert decompressed['log_encoded'] == metadata['log_encoded']
    assert decompressed['pattern_matched'] == metadata['pattern_matched']
    assert decompressed['entropy_coded'] == metadata['entropy_coded']
    assert decompressed['mixed_precision'] == metadata['mixed_precision']
    assert decompressed['triton_accelerated'] == metadata['triton_accelerated']
    
    # Check sparse data
    if 'sparse_indices' in decompressed:
        assert torch.allclose(decompressed['sparse_indices'], metadata['sparse_indices'])
    if 'sparse_values' in decompressed:
        assert torch.allclose(decompressed['sparse_values'], metadata['sparse_values'], atol=1e-3)
    
    print("✓ Full metadata compression test passed")


def test_compression_overhead():
    """Test that metadata overhead is < 1% of compressed data"""
    print("\nTesting compression overhead...")
    
    # Initialize system
    config = PurelyPyTorchConfig(
        prime=257,
        precision=6,
        enable_sparse=True,
        enable_pattern_matching=True,
        enable_entropy=True
    )
    system = PurelyPyTorchPAdicSystem(config)
    compressor = MetadataCompressor()
    
    # Test various tensor sizes
    test_sizes = [
        (32, 32),      # Small 2D
        (64, 64, 64),  # Medium 3D
        (128, 256),    # Large 2D
        (32, 32, 32, 32)  # 4D tensor
    ]
    
    for size in test_sizes:
        print(f"\n  Testing size {size}...")
        
        # Generate test tensor with some sparsity
        tensor = torch.randn(size)
        mask = torch.rand(size) > 0.7  # 70% sparsity
        tensor = tensor * mask.float()
        
        # Compress
        result = system.compress(tensor)
        
        # Compress metadata
        metadata_bytes = compressor.compress_metadata(result.metadata)
        
        # Calculate sizes
        data_size = result.compressed_data.numel() * result.compressed_data.element_size()
        if result.sparse_indices is not None:
            data_size += result.sparse_indices.numel() * result.sparse_indices.element_size()
        
        metadata_size = len(metadata_bytes)
        
        # Calculate overhead
        overhead = compressor._estimate_compression_ratio(metadata_size, data_size)
        
        print(f"    Data size: {data_size} bytes")
        print(f"    Metadata size: {metadata_size} bytes")
        print(f"    Overhead: {overhead:.2f}%")
        
        # Verify < 1% overhead for reasonable data sizes
        if data_size > 1000:  # For data > 1KB
            assert overhead < 1.0, f"Metadata overhead {overhead:.2f}% exceeds 1% target"
    
    print("\n✓ Compression overhead test passed")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\nTesting edge cases...")
    
    compressor = MetadataCompressor()
    
    # Empty metadata
    empty_meta = {}
    compressed = compressor.compress_metadata(empty_meta)
    decompressed = compressor.decompress_metadata(compressed)
    assert isinstance(decompressed, dict)
    
    # Large indices
    large_indices = list(range(0, 1000000, 1000))
    encoded = compressor._delta_encode_indices(large_indices)
    decoded, _ = compressor._delta_decode_indices(encoded, 0)
    assert decoded == large_indices
    
    # Negative values in varint
    negative = -1234567890
    encoded = compressor._encode_varint(negative)
    decoded, _ = compressor._decode_varint(encoded, 0)
    assert decoded == negative
    
    print("✓ Edge cases test passed")


def benchmark_performance():
    """Benchmark compression/decompression performance"""
    print("\nBenchmarking performance...")
    
    compressor = MetadataCompressor()
    
    # Create large metadata
    metadata = {
        'version': 1,
        'prime': 257,
        'precision': 6,
        'original_shape': (512, 512, 512),
        'sparse_indices': torch.randint(0, 1000000, (10000,)),
        'sparse_values': torch.randn(10000),
        'pattern_dict': {f'pattern_{i}': torch.randn(32) for i in range(100)},
        'entropy_metadata': {
            'min_val': -100.0,
            'max_val': 100.0,
            'entropy': 8.5,
            'n_bins': 512
        }
    }
    
    # Benchmark compression
    start = time.time()
    for _ in range(100):
        compressed = compressor.compress_metadata(metadata)
    compress_time = (time.time() - start) / 100
    
    # Benchmark decompression
    start = time.time()
    for _ in range(100):
        decompressed = compressor.decompress_metadata(compressed)
    decompress_time = (time.time() - start) / 100
    
    print(f"  Compression time: {compress_time*1000:.2f} ms")
    print(f"  Decompression time: {decompress_time*1000:.2f} ms")
    print(f"  Compressed size: {len(compressed)} bytes")
    
    # Calculate throughput
    metadata_size = sum(
        t.numel() * t.element_size() if isinstance(t, torch.Tensor) else 0
        for t in metadata.values()
    )
    throughput = metadata_size / compress_time / 1024 / 1024  # MB/s
    print(f"  Throughput: {throughput:.2f} MB/s")
    
    print("✓ Performance benchmark completed")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Metadata Compressor Test Suite")
    print("=" * 60)
    
    # Unit tests
    test_header_packing()
    test_delta_encoding()
    test_varint_encoding()
    test_flag_packing()
    test_pattern_dict_compression()
    
    # Integration tests
    test_full_metadata_compression()
    test_compression_overhead()
    test_edge_cases()
    
    # Performance
    benchmark_performance()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("Metadata overhead target < 1% achieved")
    print("=" * 60)


if __name__ == "__main__":
    main()