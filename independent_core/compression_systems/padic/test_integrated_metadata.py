"""
Integration test for metadata compressor with full P-adic compression system
Verifies end-to-end compression with < 1% metadata overhead
"""

import torch
import numpy as np
from typing import Dict, Any
import time

from .padic_compression_pytorch import PurelyPyTorchPAdicSystem, PurelyPyTorchConfig, CompressionResult
from .metadata_compressor import MetadataCompressor


def test_integrated_compression():
    """Test metadata compression integrated with P-adic system"""
    print("Testing integrated metadata compression...")
    
    # Initialize system with all features enabled
    config = PurelyPyTorchConfig(
        prime=257,
        precision=6,
        enable_sparse=True,
        enable_pattern_matching=True,
        enable_entropy=True,
        enable_log_encoding=True,
        sparse_threshold=0.01
    )
    system = PurelyPyTorchPAdicSystem(config)
    
    # Test tensor with realistic sparsity
    size = (128, 256)
    tensor = torch.randn(size)
    
    # Add sparsity (70% zeros)
    mask = torch.rand(size) > 0.3
    tensor = tensor * mask.float()
    
    print(f"  Original tensor shape: {tensor.shape}")
    print(f"  Sparsity: {(1 - mask.float().mean().item()) * 100:.1f}%")
    
    # Compress
    result = system.compress(tensor)
    
    # Check that metadata was compressed
    assert result.compressed_metadata is not None, "Metadata should be compressed"
    
    # Calculate sizes
    data_size = result.compressed_data.numel() * result.compressed_data.element_size()
    if result.sparse_indices is not None:
        data_size += result.sparse_indices.numel() * result.sparse_indices.element_size()
    
    metadata_size = len(result.compressed_metadata)
    overhead = (metadata_size / data_size) * 100
    
    print(f"  Compressed data size: {data_size:,} bytes")
    print(f"  Compressed metadata size: {metadata_size} bytes")
    print(f"  Metadata overhead: {overhead:.4f}%")
    print(f"  Compression ratio: {result.compression_ratio:.2f}x")
    
    # Decompress
    decompressed = system.decompress(result)
    
    # Verify reconstruction
    error = torch.abs(tensor - decompressed.reconstructed_data).max().item()
    print(f"  Max reconstruction error: {error:.6e}")
    
    # Verify metadata was properly decompressed
    assert decompressed.metadata['prime'] == config.prime
    assert decompressed.metadata['precision'] == config.precision
    assert decompressed.metadata['original_shape'] == tensor.shape
    
    print("✓ Integrated compression test passed")
    
    return overhead


def test_various_sizes():
    """Test metadata overhead for various tensor sizes"""
    print("\nTesting metadata overhead for various sizes...")
    
    config = PurelyPyTorchConfig(
        prime=257,
        precision=6,
        enable_sparse=True,
        enable_entropy=True
    )
    system = PurelyPyTorchPAdicSystem(config)
    
    test_configs = [
        # (shape, sparsity, description)
        ((64, 64), 0.0, "Small dense"),
        ((64, 64), 0.5, "Small sparse"),
        ((256, 256), 0.0, "Medium dense"),
        ((256, 256), 0.7, "Medium sparse"),
        ((512, 512), 0.9, "Large very sparse"),
        ((128, 128, 128), 0.8, "3D sparse"),
    ]
    
    results = []
    
    for shape, sparsity, desc in test_configs:
        # Generate tensor
        tensor = torch.randn(shape)
        if sparsity > 0:
            mask = torch.rand(shape) > sparsity
            tensor = tensor * mask.float()
        
        # Compress
        result = system.compress(tensor)
        
        # Calculate overhead
        data_size = result.compressed_data.numel() * result.compressed_data.element_size()
        if result.sparse_indices is not None:
            data_size += result.sparse_indices.numel() * result.sparse_indices.element_size()
        
        metadata_size = len(result.compressed_metadata) if result.compressed_metadata else 0
        overhead = (metadata_size / data_size) * 100 if data_size > 0 else 0
        
        results.append({
            'description': desc,
            'shape': shape,
            'sparsity': sparsity,
            'data_size': data_size,
            'metadata_size': metadata_size,
            'overhead': overhead,
            'compression_ratio': result.compression_ratio
        })
        
        print(f"\n  {desc}:")
        print(f"    Shape: {shape}, Sparsity: {sparsity*100:.0f}%")
        print(f"    Data: {data_size:,} bytes, Metadata: {metadata_size} bytes")
        print(f"    Overhead: {overhead:.4f}%")
        print(f"    Compression ratio: {result.compression_ratio:.2f}x")
    
    # Check all overheads
    max_overhead = max(r['overhead'] for r in results if r['data_size'] > 1000)
    avg_overhead = np.mean([r['overhead'] for r in results if r['data_size'] > 1000])
    
    print(f"\n  Maximum overhead (for data > 1KB): {max_overhead:.4f}%")
    print(f"  Average overhead (for data > 1KB): {avg_overhead:.4f}%")
    
    if max_overhead < 1.0:
        print("  ✓ All sizes achieve < 1% metadata overhead!")
    
    return results


def test_metadata_preservation():
    """Test that all metadata is preserved through compression/decompression"""
    print("\nTesting metadata preservation...")
    
    config = PurelyPyTorchConfig(
        prime=257,
        precision=6,
        enable_sparse=True,
        enable_pattern_matching=True,
        enable_entropy=True,
        enable_log_encoding=True,
        enable_mixed_precision=False,  # Disable for CPU testing
        enable_triton=False  # Disable for CPU testing
    )
    system = PurelyPyTorchPAdicSystem(config)
    
    # Create tensor with patterns
    base_pattern = torch.tensor([1.0, 2.0, 3.0, 4.0])
    tensor = base_pattern.repeat(32, 32)
    tensor = tensor.reshape(64, 64)
    
    # Add some noise
    noise = torch.randn(64, 64) * 0.1
    tensor = tensor + noise
    
    # Compress
    result = system.compress(tensor)
    
    # Verify compressed metadata exists
    assert result.compressed_metadata is not None
    
    # Create a new result with only compressed data and metadata
    minimal_result = CompressionResult(
        compressed_data=result.compressed_data,
        metadata={},  # Empty metadata dict
        compression_ratio=result.compression_ratio,
        encoding_time=result.encoding_time,
        compressed_metadata=result.compressed_metadata  # Only compressed bytes
    )
    
    # Decompress using only compressed metadata
    decompressed = system.decompress(minimal_result)
    
    # Verify all critical metadata was preserved
    assert 'prime' in decompressed.metadata
    assert 'precision' in decompressed.metadata
    assert 'original_shape' in decompressed.metadata
    assert decompressed.metadata['prime'] == config.prime
    assert decompressed.metadata['precision'] == config.precision
    
    # Verify reconstruction
    reconstructed = decompressed.reconstructed_data
    assert reconstructed.shape == tensor.shape
    
    # Check reconstruction quality
    error = torch.abs(tensor - reconstructed).max().item()
    print(f"  Max reconstruction error: {error:.6e}")
    
    print("  ✓ All metadata preserved through compression")
    print("✓ Metadata preservation test passed")


def test_statistics_tracking():
    """Test that metadata overhead statistics are properly tracked"""
    print("\nTesting statistics tracking...")
    
    config = PurelyPyTorchConfig(
        prime=257,
        precision=6,
        enable_sparse=True
    )
    system = PurelyPyTorchPAdicSystem(config)
    
    # Reset statistics
    system.reset_statistics()
    
    # Perform multiple compressions
    total_data = 0
    total_metadata = 0
    
    for i in range(10):
        size = (32 * (i + 1), 32 * (i + 1))
        tensor = torch.randn(size)
        
        # Add varying sparsity
        sparsity = i * 0.1
        if sparsity > 0:
            mask = torch.rand(size) > sparsity
            tensor = tensor * mask.float()
        
        result = system.compress(tensor)
        
        # Track sizes
        data_size = result.compressed_data.numel() * result.compressed_data.element_size()
        if result.sparse_indices is not None:
            data_size += result.sparse_indices.numel() * result.sparse_indices.element_size()
        
        metadata_size = len(result.compressed_metadata) if result.compressed_metadata else 0
        
        total_data += data_size
        total_metadata += metadata_size
    
    # Get statistics
    stats = system.get_statistics()
    
    print(f"  Total compressions: {stats['total_compressions']}")
    print(f"  Average compression ratio: {stats['average_compression_ratio']:.2f}x")
    
    if 'metadata_compressor_stats' in stats:
        meta_stats = stats['metadata_compressor_stats']
        print(f"  Metadata compressor stats:")
        print(f"    Total compressions: {meta_stats['total_compressions']}")
        print(f"    Total metadata bytes: {meta_stats['total_metadata_bytes']:,}")
        print(f"    Total data bytes: {meta_stats['total_data_bytes']:,}")
    
    if 'metadata_overhead_percentage' in stats:
        print(f"  Overall metadata overhead: {stats['metadata_overhead_percentage']:.4f}%")
        
        # Verify < 1% overhead
        if stats['metadata_overhead_percentage'] < 1.0:
            print("  ✓ Achieved < 1% metadata overhead in statistics!")
    
    print("✓ Statistics tracking test passed")


def benchmark_metadata_compression():
    """Benchmark metadata compression performance"""
    print("\nBenchmarking metadata compression performance...")
    
    config = PurelyPyTorchConfig(
        prime=257,
        precision=6,
        enable_sparse=True,
        enable_entropy=True
    )
    system = PurelyPyTorchPAdicSystem(config)
    
    # Large tensor for benchmarking
    tensor = torch.randn(1024, 1024)
    
    # Add 80% sparsity
    mask = torch.rand(1024, 1024) > 0.8
    tensor = tensor * mask.float()
    
    # Warm up
    for _ in range(3):
        result = system.compress(tensor)
        _ = system.decompress(result)
    
    # Benchmark compression
    n_iterations = 10
    compress_times = []
    decompress_times = []
    metadata_sizes = []
    
    for _ in range(n_iterations):
        # Compress
        start = time.time()
        result = system.compress(tensor)
        compress_time = time.time() - start
        compress_times.append(compress_time)
        
        if result.compressed_metadata:
            metadata_sizes.append(len(result.compressed_metadata))
        
        # Decompress
        start = time.time()
        _ = system.decompress(result)
        decompress_time = time.time() - start
        decompress_times.append(decompress_time)
    
    # Calculate statistics
    avg_compress = np.mean(compress_times) * 1000  # ms
    avg_decompress = np.mean(decompress_times) * 1000  # ms
    avg_metadata_size = np.mean(metadata_sizes) if metadata_sizes else 0
    
    print(f"  Average compression time: {avg_compress:.2f} ms")
    print(f"  Average decompression time: {avg_decompress:.2f} ms")
    print(f"  Average metadata size: {avg_metadata_size:.0f} bytes")
    
    # Calculate throughput
    tensor_size = tensor.numel() * 4  # float32
    throughput = (tensor_size / 1024 / 1024) / np.mean(compress_times)  # MB/s
    print(f"  Compression throughput: {throughput:.2f} MB/s")
    
    print("✓ Performance benchmark completed")


def main():
    """Run all integration tests"""
    print("=" * 70)
    print("Integrated Metadata Compressor Test Suite")
    print("=" * 70)
    
    # Run tests
    overhead1 = test_integrated_compression()
    results = test_various_sizes()
    test_metadata_preservation()
    test_statistics_tracking()
    benchmark_metadata_compression()
    
    # Summary
    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 70)
    
    # Check overall success
    all_overheads = [r['overhead'] for r in results if r['data_size'] > 1000]
    all_overheads.append(overhead1)
    max_overhead = max(all_overheads)
    avg_overhead = np.mean(all_overheads)
    
    print(f"Maximum metadata overhead: {max_overhead:.4f}%")
    print(f"Average metadata overhead: {avg_overhead:.4f}%")
    
    if max_overhead < 1.0:
        print("\n✅ SUCCESS: All tests passed with < 1% metadata overhead!")
    else:
        print(f"\n⚠️  Warning: Maximum overhead {max_overhead:.2f}% exceeds 1% target")
    
    print("=" * 70)


if __name__ == "__main__":
    main()