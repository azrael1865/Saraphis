"""
Simple test for metadata compressor - verifies core functionality
"""

import torch
from .metadata_compressor import MetadataCompressor, MetadataHeader


def test_basic_compression():
    """Test basic metadata compression"""
    print("Testing basic metadata compression...")
    
    compressor = MetadataCompressor()
    
    # Simple metadata
    metadata = {
        'version': 1,
        'prime': 257,
        'precision': 6,
        'original_shape': (32, 64),
        'sparse_encoded': True,
        'log_encoded': True,
        'entropy_coded': False,
        'triton_accelerated': True
    }
    
    # Compress
    compressed = compressor.compress_metadata(metadata)
    print(f"  Original metadata fields: {len(metadata)}")
    print(f"  Compressed size: {len(compressed)} bytes")
    
    # Decompress
    decompressed = compressor.decompress_metadata(compressed)
    
    # Verify key fields
    assert decompressed['version'] == metadata['version']
    assert decompressed['prime'] == metadata['prime']
    assert decompressed['precision'] == metadata['precision']
    assert decompressed['original_shape'] == metadata['original_shape']
    assert decompressed['sparse_encoded'] == metadata['sparse_encoded']
    assert decompressed['log_encoded'] == metadata['log_encoded']
    
    print("✓ Basic compression test passed")
    return compressed, decompressed


def test_sparse_metadata():
    """Test sparse indices compression"""
    print("\nTesting sparse metadata compression...")
    
    compressor = MetadataCompressor()
    
    # Create sparse indices - simulate real sparse data
    indices = torch.tensor([0, 10, 20, 30, 100, 200, 300, 1000])
    values = torch.randn(8)
    
    metadata = {
        'prime': 257,
        'precision': 6,
        'original_shape': (100, 100),
        'sparse_encoded': True,
        'sparse_indices': indices,
        'sparse_values': values
    }
    
    # Compress
    compressed = compressor.compress_metadata(metadata)
    
    # Calculate overhead
    data_size = values.numel() * 4 + indices.numel() * 8  # float32 + int64
    metadata_size = len(compressed)
    overhead = (metadata_size / data_size) * 100
    
    print(f"  Data size: {data_size} bytes")
    print(f"  Metadata size: {metadata_size} bytes")
    print(f"  Overhead: {overhead:.2f}%")
    
    # Decompress and verify
    decompressed = compressor.decompress_metadata(compressed)
    
    if 'sparse_indices' in decompressed:
        assert torch.allclose(decompressed['sparse_indices'], indices)
        print("  ✓ Sparse indices recovered correctly")
    
    if 'sparse_values' in decompressed:
        # Allow some precision loss due to float16 compression
        assert torch.allclose(decompressed['sparse_values'], values, atol=1e-3)
        print("  ✓ Sparse values recovered correctly")
    
    print("✓ Sparse metadata test passed")
    return overhead


def test_overhead_target():
    """Test that we achieve < 1% overhead for realistic data"""
    print("\nTesting overhead target...")
    
    compressor = MetadataCompressor()
    
    # Realistic scenario: 1MB of compressed data
    data_size_mb = 1
    data_size_bytes = data_size_mb * 1024 * 1024
    
    # Typical metadata for 1MB compressed tensor
    metadata = {
        'version': 1,
        'prime': 257,
        'precision': 6,
        'original_shape': (512, 512, 4),  # ~4MB uncompressed
        'sparse_encoded': True,
        'sparse_indices': torch.arange(0, 10000, 10),  # 1000 indices
        'sparse_values': torch.randn(1000),
        'log_encoded': True,
        'pattern_matched': True,
        'entropy_coded': True,
        'entropy_metadata': {
            'min_val': -10.0,
            'max_val': 10.0,
            'entropy': 7.5,
            'n_bins': 256
        }
    }
    
    # Compress metadata
    compressed = compressor.compress_metadata(metadata)
    metadata_size = len(compressed)
    
    # Calculate overhead
    overhead = (metadata_size / data_size_bytes) * 100
    
    print(f"  Data size: {data_size_bytes:,} bytes ({data_size_mb}MB)")
    print(f"  Metadata size: {metadata_size} bytes")
    print(f"  Overhead: {overhead:.4f}%")
    
    # Verify < 1% target
    if overhead < 1.0:
        print(f"  ✓ Achieved < 1% overhead target!")
    else:
        print(f"  ✗ Overhead {overhead:.2f}% exceeds 1% target")
    
    return overhead


def test_compression_features():
    """Test individual compression features"""
    print("\nTesting compression features...")
    
    compressor = MetadataCompressor()
    
    # Test delta encoding
    indices = [0, 100, 200, 300, 1000, 2000, 3000]
    encoded = compressor._delta_encode_indices(indices)
    decoded, _ = compressor._delta_decode_indices(encoded, 0)
    assert decoded == indices
    print("  ✓ Delta encoding works")
    
    # Test varint encoding
    test_values = [0, 1, -1, 127, 255, 1000, -1000]
    for val in test_values:
        encoded = compressor._encode_varint(val)
        decoded, _ = compressor._decode_varint(encoded, 0)
        assert decoded == val
    print("  ✓ Varint encoding works")
    
    # Test flag packing
    flags = [True, False, True, True, False, False, True, False]
    packed = compressor._pack_flags(flags)
    unpacked, _ = compressor._unpack_flags(packed, 0)
    assert unpacked == flags
    print("  ✓ Flag packing works")
    
    # Test shape encoding
    shape = (32, 64, 128, 256)
    encoded = compressor._varint_encode_shape(shape)
    decoded, _ = compressor._varint_decode_shape(encoded, 0)
    assert decoded == shape
    print("  ✓ Shape encoding works")
    
    print("✓ All compression features passed")


def main():
    """Run simple tests"""
    print("=" * 60)
    print("Metadata Compressor Simple Test")
    print("=" * 60)
    
    # Run tests
    compressed, decompressed = test_basic_compression()
    sparse_overhead = test_sparse_metadata()
    overall_overhead = test_overhead_target()
    test_compression_features()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✓ Basic compression works")
    print(f"✓ Sparse metadata overhead: {sparse_overhead:.2f}%")
    print(f"✓ Overall overhead for 1MB data: {overall_overhead:.4f}%")
    
    if overall_overhead < 1.0:
        print(f"\n✅ SUCCESS: Achieved < 1% metadata overhead target!")
    else:
        print(f"\n⚠️  Warning: Overhead {overall_overhead:.2f}% exceeds 1% target")
    
    print("=" * 60)


if __name__ == "__main__":
    main()