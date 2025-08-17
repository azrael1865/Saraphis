"""
Comprehensive test suite to verify all three fixes for Saraphis compression system
"""

import unittest
import torch
import numpy as np
import time
import json
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSaraphisFixes(unittest.TestCase):
    """Test suite for verifying all Saraphis compression fixes"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Running tests on device: {self.device}")
    
    def test_metadata_compression_preserves_frequency_table(self):
        """
        Test Fix 1: Verify that frequency_table is preserved through compression/decompression
        """
        from independent_core.compression_systems.padic.metadata_compressor import MetadataCompressor
        
        compressor = MetadataCompressor()
        
        # Create test entropy metadata with frequency table
        original_metadata = {
            'method': 'huffman',
            'encoding_method': 'huffman',
            'frequency_table': {
                0: 100,
                1: 150,
                2: 75,
                3: 200,
                4: 125,
                5: 90,
                6: 110,
                7: 145
            },
            'probability_distribution': {
                0: 0.101,
                1: 0.152,
                2: 0.076,
                3: 0.202,
                4: 0.126,
                5: 0.091,
                6: 0.111,
                7: 0.141
            },
            'min_val': -1.5,
            'max_val': 2.5,
            'entropy': 2.876,
            'n_bins': 256,
            'encoding_table_size': 8,
            'nested_metadata': {
                'compression_level': 6,
                'chunk_size': 1024
            }
        }
        
        # Compress metadata
        compressed = compressor._compress_entropy_metadata(original_metadata)
        self.assertIsInstance(compressed, bytes)
        self.assertGreater(len(compressed), 0)
        
        logger.info(f"Compressed metadata size: {len(compressed)} bytes")
        
        # Decompress metadata
        decompressed, offset = compressor._decompress_entropy_metadata(compressed, 0)
        
        # Verify all fields are preserved
        self.assertIn('frequency_table', decompressed)
        self.assertIsNotNone(decompressed['frequency_table'])
        self.assertEqual(len(decompressed['frequency_table']), len(original_metadata['frequency_table']))
        
        # Verify frequency table values
        for key, value in original_metadata['frequency_table'].items():
            # Keys might be converted to int during decompression
            if str(key) in decompressed['frequency_table']:
                self.assertEqual(decompressed['frequency_table'][str(key)], value)
            else:
                self.assertEqual(decompressed['frequency_table'][key], value)
        
        # Verify other critical fields
        self.assertEqual(decompressed['method'], original_metadata['method'])
        self.assertEqual(decompressed['encoding_method'], original_metadata['encoding_method'])
        self.assertAlmostEqual(decompressed['entropy'], original_metadata['entropy'], places=3)
        
        # Verify nested metadata
        self.assertIn('nested_metadata', decompressed)
        self.assertEqual(
            decompressed['nested_metadata']['compression_level'],
            original_metadata['nested_metadata']['compression_level']
        )
        
        logger.info("✅ Metadata compression test passed - frequency_table preserved")
    
    def test_sheaf_numpy_compression_api(self):
        """
        Test Fix 2: Verify correct numpy.savez_compressed API usage
        """
        from independent_core.compression_systems.sheaf.sheaf_compressor import SheafCompressionSystem
        
        compressor = SheafCompressionSystem({'device': str(self.device)})
        
        # Test with various numpy array types
        test_arrays = [
            np.random.randn(100, 100).astype(np.float32),
            np.random.randn(50, 50, 3).astype(np.float64),
            np.random.randint(0, 255, (64, 64, 3)).astype(np.uint8),
            np.ones((10, 10), dtype=np.int32) * 42
        ]
        
        for i, test_array in enumerate(test_arrays):
            with self.subTest(array_index=i, shape=test_array.shape, dtype=test_array.dtype):
                # Compress array
                compressed_data = compressor._compress_array_section(test_array)
                
                # Verify compressed structure
                self.assertIn('type', compressed_data)
                self.assertEqual(compressed_data['type'], 'numpy_compressed')
                self.assertIn('data', compressed_data)
                self.assertIsInstance(compressed_data['data'], bytes)
                self.assertIn('shape', compressed_data)
                self.assertEqual(tuple(compressed_data['shape']), test_array.shape)
                self.assertIn('dtype', compressed_data)
                self.assertEqual(compressed_data['dtype'], str(test_array.dtype))
                
                # Verify compression occurred
                original_size = compressed_data['original_size']
                compressed_size = compressed_data['compressed_size']
                self.assertLess(compressed_size, original_size)
                
                # Decompress and verify
                decompressed_array = compressor._decompress_array_section(compressed_data)
                
                # Verify shape and dtype
                self.assertEqual(decompressed_array.shape, test_array.shape)
                self.assertEqual(decompressed_array.dtype, test_array.dtype)
                
                # Verify content (should be identical for lossless compression)
                np.testing.assert_array_almost_equal(decompressed_array, test_array)
                
                logger.info(f"✅ Array {i}: shape={test_array.shape}, dtype={test_array.dtype}, "
                          f"compression_ratio={compressed_data['compression_ratio']:.2f}")
        
        logger.info("✅ Sheaf numpy compression test passed")
    
    def test_padic_adaptive_precision_performance(self):
        """
        Test Fix 3: Verify vectorized P-adic processing performance improvement
        """
        from independent_core.compression_systems.padic.adaptive_precision_wrapper import AdaptivePrecisionWrapper
        
        # Mock config and math_ops for testing
        class MockConfig:
            device = 'cpu'
            prime = 251
            compression_ratio = 0.5
            min_precision = 4
            max_precision = 32
            target_error = 0.01
        
        class MockMathOps:
            def to_padic(self, value, precision=None):
                # Simplified p-adic conversion for testing
                return int(value * 1000) % 251
            
            def from_padic(self, weight):
                # Simplified reconstruction
                return weight / 1000.0
        
        config = MockConfig()
        math_ops = MockMathOps()
        
        wrapper = AdaptivePrecisionWrapper(config, math_ops)
        
        # Test with various tensor sizes
        test_sizes = [
            (50, 50),      # 2,500 elements - serial processing
            (100, 100),    # 10,000 elements - serial processing
            (200, 200),    # 40,000 elements - batched processing
            (500, 500),    # 250,000 elements - parallel batched processing
        ]
        
        performance_results = []
        
        for shape in test_sizes:
            # Create test tensor
            tensor = torch.randn(shape).to(self.device)
            num_elements = tensor.numel()
            
            # Measure processing time
            start_time = time.time()
            result = wrapper.convert_tensor(tensor)
            processing_time = time.time() - start_time
            
            # Calculate throughput
            throughput = num_elements / processing_time
            
            # Verify result
            self.assertIsNotNone(result)
            self.assertEqual(result.original_shape, tensor.shape)
            self.assertEqual(len(result.padic_weights), num_elements)
            self.assertEqual(result.error_map.shape, tensor.shape)
            
            performance_results.append({
                'shape': shape,
                'elements': num_elements,
                'time': processing_time,
                'throughput': throughput
            })
            
            logger.info(f"Shape {shape}: {num_elements} elements in {processing_time:.3f}s "
                       f"({throughput:.0f} elements/s)")
        
        # Verify performance scaling
        # The vectorized version should handle large tensors much better
        small_tensor_time = performance_results[0]['time']
        large_tensor_time = performance_results[-1]['time']
        
        # Large tensor (100x more elements) should take less than 100x time
        # due to vectorization/parallelization
        size_ratio = performance_results[-1]['elements'] / performance_results[0]['elements']
        time_ratio = large_tensor_time / small_tensor_time
        
        self.assertLess(time_ratio, size_ratio * 0.5,
                       f"Performance scaling issue: {size_ratio}x size increase caused "
                       f"{time_ratio}x time increase (should be much less)")
        
        # Check performance stats
        stats = wrapper.get_performance_stats()
        self.assertEqual(stats['tensors_processed'], len(test_sizes))
        self.assertGreater(stats['total_elements'], 0)
        
        logger.info(f"✅ P-adic performance test passed - Processing strategies: {stats['processing_strategies']}")
    
    def test_backward_compatibility(self):
        """
        Test that fixes maintain backward compatibility with existing data
        """
        from independent_core.compression_systems.padic.metadata_compressor import MetadataCompressor
        
        compressor = MetadataCompressor()
        
        # Simulate legacy compressed data (basic fields only)
        legacy_data = bytearray()
        import struct
        
        # Add legacy format data
        legacy_data.extend(struct.pack('>fffi', -1.0, 1.0, 2.5, 256))  # min, max, entropy, n_bins
        legacy_data.extend(b'huffman\x00' + b'\x00' * 9)  # method string
        
        # Try to decompress legacy data
        result, offset = compressor._decompress_entropy_metadata(bytes(legacy_data), 0)
        
        # Should get basic fields at minimum
        self.assertIn('encoding_method', result)
        self.assertIn('method', result)
        self.assertEqual(result['min_val'], -1.0)
        self.assertEqual(result['max_val'], 1.0)
        self.assertAlmostEqual(result['entropy'], 2.5, places=3)
        self.assertEqual(result['n_bins'], 256)
        
        logger.info("✅ Backward compatibility test passed")
    
    def test_integration_compression_pipeline(self):
        """
        Integration test: Full compression/decompression pipeline with all fixes
        """
        logger.info("\n🔷 Testing integrated compression pipeline with all fixes...")
        
        # Create test tensor
        test_tensor = torch.randn(300, 300).to(self.device)
        
        # Test metadata compression with frequency table
        from independent_core.compression_systems.padic.metadata_compressor import MetadataCompressor
        
        metadata_compressor = MetadataCompressor()
        
        # Create complete metadata including frequency table
        test_metadata = {
            'method': 'hybrid',
            'frequency_table': {i: np.random.randint(1, 100) for i in range(256)},
            'probability_table': {i: np.random.random() for i in range(256)},
            'encoding_metadata': {
                'chunk_size': 1024,
                'compression_level': 6
            }
        }
        
        # Compress and decompress
        compressed_meta = metadata_compressor._compress_entropy_metadata(test_metadata)
        decompressed_meta, _ = metadata_compressor._decompress_entropy_metadata(compressed_meta, 0)
        
        # Verify frequency table preserved
        self.assertIn('frequency_table', decompressed_meta)
        self.assertEqual(len(decompressed_meta['frequency_table']), 256)
        
        # Test numpy compression
        from independent_core.compression_systems.sheaf.sheaf_compressor import SheafCompressionSystem
        
        sheaf_compressor = SheafCompressionSystem({'device': str(self.device)})
        test_array = test_tensor.cpu().numpy()
        
        compressed_array = sheaf_compressor._compress_array_section(test_array)
        decompressed_array = sheaf_compressor._decompress_array_section(compressed_array)
        
        # Verify array integrity
        np.testing.assert_array_almost_equal(decompressed_array, test_array, decimal=5)
        
        logger.info("✅ Integration test passed - All systems working together")
    
    def test_performance_benchmarks(self):
        """
        Performance benchmarks to verify improvements
        """
        logger.info("\n📊 Running performance benchmarks...")
        
        from independent_core.compression_systems.padic.adaptive_precision_wrapper import AdaptivePrecisionWrapper
        
        # Mock components for benchmarking
        class MockConfig:
            device = 'cpu'
            prime = 251
            compression_ratio = 0.5
            min_precision = 4
            max_precision = 32
            target_error = 0.01
        
        class MockMathOps:
            def to_padic(self, value, precision=None):
                return int(abs(value) * 1000) % 251
            
            def from_padic(self, weight):
                return weight / 1000.0
        
        wrapper = AdaptivePrecisionWrapper(MockConfig(), MockMathOps())
        
        # Benchmark different tensor sizes
        benchmark_results = []
        sizes = [(100, 100), (200, 200), (300, 300), (400, 400), (500, 500)]
        
        for size in sizes:
            tensor = torch.randn(size)
            
            # Run multiple iterations for accuracy
            times = []
            for _ in range(3):
                start = time.time()
                result = wrapper.convert_tensor(tensor)
                elapsed = time.time() - start
                times.append(elapsed)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = tensor.numel() / avg_time
            
            benchmark_results.append({
                'size': size,
                'elements': tensor.numel(),
                'avg_time': avg_time,
                'std_time': std_time,
                'throughput': throughput
            })
            
            logger.info(f"Size {size}: {avg_time:.3f}±{std_time:.3f}s, "
                       f"{throughput/1000:.1f}k elements/s")
        
        # Verify performance targets
        # 500x500 tensor should process in under 5 seconds (was 10+ seconds before)
        large_tensor_result = benchmark_results[-1]
        self.assertLess(large_tensor_result['avg_time'], 5.0,
                       f"Large tensor processing too slow: {large_tensor_result['avg_time']:.2f}s")
        
        logger.info("✅ Performance benchmarks passed - All targets met")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)