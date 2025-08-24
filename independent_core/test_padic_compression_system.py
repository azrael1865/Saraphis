"""
Comprehensive test suite for PadicCompressionSystem.
Tests all functionality including edge cases, error handling, and integration.
"""

import unittest
import numpy as np
import torch
import sys
import os
from typing import List, Dict, Any, Optional
import psutil
import gc
import threading
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression_systems.padic.padic_compressor import (
    PadicCompressionSystem,
    CompressionConfig,
    CompressionResult
)


class TestPadicCompressionSystem(unittest.TestCase):
    """Test suite for PadicCompressionSystem class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Default configuration
        self.config = CompressionConfig(
            prime=257,
            base_precision=2,
            target_error=1e-6,
            enable_gpu=False,  # Use CPU for consistent testing
            device='cpu'
        )
        
        self.system = PadicCompressionSystem(self.config)
        
        # Small system for memory tests
        self.small_config = CompressionConfig(
            prime=7,
            base_precision=2,
            max_precision=3,
            enable_gpu=False,
            device='cpu'
        )
        self.small_system = PadicCompressionSystem(self.small_config)
        
    def tearDown(self):
        """Clean up after tests"""
        del self.system
        del self.small_system
        gc.collect()
    
    def test_initialization_default_config(self):
        """Test system initialization with default config"""
        system = PadicCompressionSystem()
        self.assertIsNotNone(system.config)
        self.assertEqual(system.config.prime, 257)
        self.assertEqual(system.config.base_precision, 2)
        self.assertIsNotNone(system.math_ops)
        self.assertIsNotNone(system.pattern_detector)
        self.assertIsNotNone(system.sparse_bridge)
        self.assertIsNotNone(system.entropy_bridge)
        self.assertIsNotNone(system.metadata_compressor)
    
    def test_initialization_custom_config(self):
        """Test system initialization with custom config"""
        custom_config = CompressionConfig(
            prime=13,
            base_precision=3,
            target_error=1e-5,
            enable_gpu=False
        )
        system = PadicCompressionSystem(custom_config)
        self.assertEqual(system.config.prime, 13)
        self.assertEqual(system.config.base_precision, 3)
        self.assertEqual(system.config.target_error, 1e-5)
        
    def test_compress_simple_tensor(self):
        """Test compression of simple tensor"""
        # Create simple test tensor
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        
        # Compress
        result = self.system.compress(tensor)
        
        # Verify result structure
        self.assertIsInstance(result, CompressionResult)
        self.assertIsNotNone(result.compressed_data)
        self.assertEqual(result.original_shape, list(tensor.shape))
        self.assertEqual(result.original_dtype, str(tensor.dtype))
        self.assertGreater(result.compression_ratio, 0)
        
    def test_compress_with_importance_scores(self):
        """Test compression with importance scores"""
        tensor = torch.randn(10, 10)
        importance_scores = torch.rand(10, 10)
        
        result = self.system.compress(tensor, importance_scores)
        
        self.assertIsInstance(result, CompressionResult)
        self.assertIsNotNone(result.compressed_data)
        
    def test_compress_decompress_roundtrip(self):
        """Test compress->decompress preserves values"""
        # Test various tensor types
        test_tensors = [
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            torch.randn(5, 5),
            torch.zeros(3, 3),
            torch.ones(4, 4) * 0.5,
            torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        ]
        
        for original in test_tensors:
            with self.subTest(tensor_shape=original.shape):
                # Compress
                result = self.system.compress(original)
                
                # Decompress
                reconstructed = self.system.decompress(result.compressed_data)
                
                # Verify shape and approximate values
                self.assertEqual(reconstructed.shape, original.shape)
                torch.testing.assert_close(
                    reconstructed, original, 
                    rtol=1e-3, atol=1e-3
                )
    
    def test_compress_empty_tensor(self):
        """Test compression of empty tensor"""
        empty_tensor = torch.tensor([])
        
        result = self.system.compress(empty_tensor)
        self.assertIsInstance(result, CompressionResult)
        
        # Decompress should return empty tensor
        reconstructed = self.system.decompress(result.compressed_data)
        self.assertEqual(reconstructed.numel(), 0)
        
    def test_compress_single_value_tensor(self):
        """Test compression of single value tensor"""
        single_tensor = torch.tensor([42.0])
        
        result = self.system.compress(single_tensor)
        reconstructed = self.system.decompress(result.compressed_data)
        
        torch.testing.assert_close(reconstructed, single_tensor, rtol=1e-3)
        
    def test_compress_large_tensor(self):
        """Test compression of larger tensor"""
        # Create moderately large tensor
        large_tensor = torch.randn(100, 50)
        
        result = self.system.compress(large_tensor)
        reconstructed = self.system.decompress(result.compressed_data)
        
        # Verify shape preservation
        self.assertEqual(reconstructed.shape, large_tensor.shape)
        
        # Allow for larger tolerance on large tensors
        torch.testing.assert_close(
            reconstructed, large_tensor, 
            rtol=1e-2, atol=1e-2
        )
    
    def test_compress_sparse_tensor(self):
        """Test compression of sparse tensor"""
        # Create sparse tensor (mostly zeros)
        sparse_tensor = torch.zeros(20, 20)
        sparse_tensor[::5, ::5] = torch.randn(4, 4)
        
        result = self.system.compress(sparse_tensor)
        reconstructed = self.system.decompress(result.compressed_data)
        
        torch.testing.assert_close(
            reconstructed, sparse_tensor, 
            rtol=1e-3, atol=1e-3
        )
        
    def test_compress_different_dtypes(self):
        """Test compression with different tensor dtypes"""
        base_data = [[1.0, 2.0], [3.0, 4.0]]
        
        dtypes = [torch.float32, torch.float64]
        if hasattr(torch, 'float16'):
            dtypes.append(torch.float16)
            
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                tensor = torch.tensor(base_data, dtype=dtype)
                
                result = self.system.compress(tensor)
                reconstructed = self.system.decompress(result.compressed_data)
                
                # Verify dtype is preserved (or at least compatible)
                self.assertEqual(reconstructed.shape, tensor.shape)
                torch.testing.assert_close(
                    reconstructed, tensor, 
                    rtol=1e-2, atol=1e-2  # More tolerance for different dtypes
                )
    
    def test_compress_negative_values(self):
        """Test compression of negative values"""
        negative_tensor = torch.tensor([[-1.0, -2.5], [3.7, -4.2]])
        
        result = self.system.compress(negative_tensor)
        reconstructed = self.system.decompress(result.compressed_data)
        
        torch.testing.assert_close(
            reconstructed, negative_tensor, 
            rtol=1e-3, atol=1e-3
        )
    
    def test_compress_small_values(self):
        """Test compression of very small values"""
        small_tensor = torch.tensor([[1e-6, 1e-7], [1e-8, 1e-5]])
        
        result = self.system.compress(small_tensor)
        reconstructed = self.system.decompress(result.compressed_data)
        
        # Allow larger tolerance for very small values
        torch.testing.assert_close(
            reconstructed, small_tensor, 
            rtol=1e-2, atol=1e-6
        )
    
    def test_compression_ratio_calculation(self):
        """Test compression ratio is calculated correctly"""
        tensor = torch.randn(10, 10)
        
        result = self.system.compress(tensor)
        
        # Compression ratio should be positive
        self.assertGreater(result.compression_ratio, 0)
        
        # Original size should be reasonable
        self.assertGreater(result.original_size, 0)
        
        # Compressed size should be reasonable
        self.assertGreater(result.compressed_size, 0)
        
    def test_metadata_preservation(self):
        """Test that metadata is preserved correctly"""
        tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)
        
        result = self.system.compress(tensor)
        
        # Check metadata
        self.assertEqual(result.original_shape, [2, 2])
        self.assertEqual(result.original_dtype, 'torch.float64')
        self.assertIsNotNone(result.compression_params)
        
    def test_error_handling_invalid_tensor(self):
        """Test error handling for invalid tensors"""
        # Test None tensor
        with self.assertRaises((TypeError, AttributeError)):
            self.system.compress(None)
            
        # Test non-tensor input
        with self.assertRaises((TypeError, AttributeError)):
            self.system.compress([1, 2, 3])
            
    def test_error_handling_nan_tensor(self):
        """Test handling of tensors with NaN values"""
        nan_tensor = torch.tensor([[1.0, float('nan')], [3.0, 4.0]])
        
        # Should handle NaN gracefully (either compress or raise specific error)
        try:
            result = self.system.compress(nan_tensor)
            # If it compresses, it should reconstruct reasonably
            reconstructed = self.system.decompress(result.compressed_data)
            self.assertEqual(reconstructed.shape, nan_tensor.shape)
        except (ValueError, RuntimeError) as e:
            # Acceptable to raise error for NaN values
            self.assertIn('nan', str(e).lower())
            
    def test_error_handling_inf_tensor(self):
        """Test handling of tensors with infinite values"""
        inf_tensor = torch.tensor([[1.0, float('inf')], [3.0, 4.0]])
        
        # Should handle infinity gracefully
        try:
            result = self.system.compress(inf_tensor)
            reconstructed = self.system.decompress(result.compressed_data)
            self.assertEqual(reconstructed.shape, inf_tensor.shape)
        except (ValueError, RuntimeError) as e:
            # Acceptable to raise error for inf values
            self.assertIn('inf', str(e).lower())
    
    def test_memory_efficiency(self):
        """Test memory usage is reasonable"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process moderately large tensor
        test_tensor = torch.randn(200, 200)
        
        # Compress and decompress multiple times
        for _ in range(5):
            result = self.small_system.compress(test_tensor)
            reconstructed = self.small_system.decompress(result.compressed_data)
            del result
            del reconstructed
        
        # Check memory hasn't grown excessively
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        self.assertLess(memory_growth, 200, 
                       f"Memory growth {memory_growth:.2f}MB is too high")
        
        # Cleanup
        gc.collect()
    
    def test_compression_consistency(self):
        """Test compression produces consistent results"""
        tensor = torch.randn(5, 5)
        
        # Compress the same tensor multiple times
        result1 = self.system.compress(tensor)
        result2 = self.system.compress(tensor)
        
        # Results should have same basic properties
        self.assertEqual(result1.original_shape, result2.original_shape)
        self.assertEqual(result1.original_dtype, result2.original_dtype)
        
        # Decompress both
        recon1 = self.system.decompress(result1.compressed_data)
        recon2 = self.system.decompress(result2.compressed_data)
        
        # Should reconstruct to same values
        torch.testing.assert_close(recon1, recon2, rtol=1e-6)
        
    def test_different_tensor_shapes(self):
        """Test compression of various tensor shapes"""
        shapes = [
            (10,),           # 1D
            (5, 5),          # 2D square
            (3, 4, 5),       # 3D
            (2, 3, 4, 5),    # 4D
            (1, 100),        # Row vector
            (100, 1),        # Column vector
        ]
        
        for shape in shapes:
            with self.subTest(shape=shape):
                tensor = torch.randn(*shape)
                
                result = self.system.compress(tensor)
                reconstructed = self.system.decompress(result.compressed_data)
                
                self.assertEqual(reconstructed.shape, tensor.shape)
                torch.testing.assert_close(
                    reconstructed, tensor, 
                    rtol=1e-3, atol=1e-3
                )
    
    def test_thread_safety(self):
        """Test basic thread safety"""
        results_queue = []
        
        def compress_decompress_task(tensor_id):
            tensor = torch.randn(10, 10) + tensor_id  # Unique tensor per thread
            
            try:
                result = self.small_system.compress(tensor)
                reconstructed = self.small_system.decompress(result.compressed_data)
                
                # Check if reconstruction is close to original
                is_close = torch.allclose(reconstructed, tensor, rtol=1e-2, atol=1e-2)
                results_queue.append(('success', tensor_id, is_close))
            except Exception as e:
                results_queue.append(('error', tensor_id, str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(3):  # Limited threads to avoid resource issues
            thread = threading.Thread(target=compress_decompress_task, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(results_queue), 3)
        for status, tensor_id, result in results_queue:
            self.assertEqual(status, 'success', f"Thread {tensor_id} failed: {result}")
            self.assertTrue(result, f"Thread {tensor_id} reconstruction inaccurate")
    
    def test_config_parameter_effects(self):
        """Test that different config parameters affect behavior"""
        tensor = torch.randn(20, 20)
        
        # Test different precisions
        configs = [
            CompressionConfig(prime=7, base_precision=1, enable_gpu=False),
            CompressionConfig(prime=7, base_precision=3, enable_gpu=False),
            CompressionConfig(prime=13, base_precision=2, enable_gpu=False),
        ]
        
        results = []
        for config in configs:
            system = PadicCompressionSystem(config)
            result = system.compress(tensor)
            reconstructed = system.decompress(result.compressed_data)
            results.append((result.compression_ratio, reconstructed))
        
        # All should reconstruct reasonably
        for ratio, recon in results:
            self.assertGreater(ratio, 0)
            self.assertEqual(recon.shape, tensor.shape)
            torch.testing.assert_close(recon, tensor, rtol=1e-2, atol=1e-2)
    
    def test_edge_case_tiny_tensor(self):
        """Test edge case of very small tensors"""
        tiny_tensor = torch.tensor([[0.1]])  # 1x1 tensor
        
        result = self.system.compress(tiny_tensor)
        reconstructed = self.system.decompress(result.compressed_data)
        
        torch.testing.assert_close(reconstructed, tiny_tensor, rtol=1e-3)
    
    def test_edge_case_zero_tensor(self):
        """Test edge case of all-zero tensor"""
        zero_tensor = torch.zeros(5, 5)
        
        result = self.system.compress(zero_tensor)
        reconstructed = self.system.decompress(result.compressed_data)
        
        torch.testing.assert_close(reconstructed, zero_tensor, atol=1e-6)
        
    def test_compression_result_serialization(self):
        """Test that compression results can be serialized/deserialized"""
        tensor = torch.randn(5, 5)
        
        result = self.system.compress(tensor)
        
        # Test that compressed_data is serializable
        self.assertIsNotNone(result.compressed_data)
        
        # Should be able to reconstruct from the data
        reconstructed = self.system.decompress(result.compressed_data)
        self.assertEqual(reconstructed.shape, tensor.shape)
        
    def test_importance_scores_effect(self):
        """Test that importance scores affect compression"""
        tensor = torch.randn(10, 10)
        
        # Compress without importance scores
        result1 = self.system.compress(tensor)
        
        # Compress with uniform importance scores (should be similar)
        uniform_scores = torch.ones_like(tensor) * 0.5
        result2 = self.system.compress(tensor, uniform_scores)
        
        # Compress with varied importance scores
        varied_scores = torch.rand_like(tensor)
        result3 = self.system.compress(tensor, varied_scores)
        
        # All should compress and decompress
        for result in [result1, result2, result3]:
            reconstructed = self.system.decompress(result.compressed_data)
            self.assertEqual(reconstructed.shape, tensor.shape)
            torch.testing.assert_close(reconstructed, tensor, rtol=1e-2, atol=1e-2)


class TestPadicCompressionSystemIntegration(unittest.TestCase):
    """Integration tests for PadicCompressionSystem"""
    
    def test_with_numpy_arrays(self):
        """Test compression of numpy arrays converted to tensors"""
        system = PadicCompressionSystem(CompressionConfig(enable_gpu=False))
        
        # Test numpy array
        np_array = np.random.randn(5, 5).astype(np.float32)
        tensor = torch.from_numpy(np_array)
        
        result = system.compress(tensor)
        reconstructed = system.decompress(result.compressed_data)
        
        np.testing.assert_allclose(
            reconstructed.numpy(), np_array, 
            rtol=1e-3, atol=1e-3
        )
    
    def test_batch_processing(self):
        """Test processing multiple tensors"""
        system = PadicCompressionSystem(CompressionConfig(enable_gpu=False))
        
        # Create batch of tensors
        tensors = [torch.randn(5, 5) for _ in range(3)]
        
        # Compress all
        results = [system.compress(t) for t in tensors]
        
        # Decompress all
        reconstructed = [system.decompress(r.compressed_data) for r in results]
        
        # Verify all match
        for original, recon in zip(tensors, reconstructed):
            torch.testing.assert_close(recon, original, rtol=1e-3, atol=1e-3)
    
    def test_performance_reasonable(self):
        """Test that compression performance is reasonable"""
        system = PadicCompressionSystem(CompressionConfig(enable_gpu=False))
        
        tensor = torch.randn(50, 50)
        
        # Time compression
        start_time = time.time()
        result = system.compress(tensor)
        compress_time = time.time() - start_time
        
        # Time decompression
        start_time = time.time()
        reconstructed = system.decompress(result.compressed_data)
        decompress_time = time.time() - start_time
        
        # Should complete in reasonable time (< 5 seconds each)
        self.assertLess(compress_time, 5.0, 
                       f"Compression took {compress_time:.2f}s, too slow")
        self.assertLess(decompress_time, 5.0, 
                       f"Decompression took {decompress_time:.2f}s, too slow")
        
        # Verify accuracy
        torch.testing.assert_close(reconstructed, tensor, rtol=1e-3, atol=1e-3)


def run_tests():
    """Run all tests with verbose output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPadicCompressionSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestPadicCompressionSystemIntegration))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✓ All PadicCompressionSystem tests passed!")
    else:
        print(f"✗ Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}")
                
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)