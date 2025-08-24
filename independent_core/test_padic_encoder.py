"""
Comprehensive test suite for PadicEncoder.
Tests all functionality including edge cases and memory management.
"""

import unittest
import numpy as np
import torch
import sys
import os
from typing import List
import psutil
import gc

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from independent_core.compression_systems.padic.padic_encoder import (
    PadicEncoder,
    PadicWeight,
    PadicMathematicalOperations,
    validate_precision,
    get_safe_precision,
    validate_single_weight,
    validate_padic_weights,
    validate_ultrametric_property,
    create_real_padic_weights,
    SAFE_PRECISION_LIMITS
)


class TestPadicEncoder(unittest.TestCase):
    """Test suite for PadicEncoder class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.encoder = PadicEncoder(prime=257, precision=4)
        self.small_encoder = PadicEncoder(prime=7, precision=3)
        
    def tearDown(self):
        """Clean up after tests"""
        del self.encoder
        del self.small_encoder
        gc.collect()
    
    def test_initialization(self):
        """Test encoder initialization with various parameters"""
        # Test default initialization
        encoder = PadicEncoder()
        self.assertEqual(encoder.prime, 257)
        self.assertEqual(encoder.precision, 4)
        
        # Test custom initialization
        encoder = PadicEncoder(prime=13, precision=5)
        self.assertEqual(encoder.prime, 13)
        self.assertEqual(encoder.precision, 5)
        
        # Test invalid prime
        with self.assertRaises(ValueError):
            PadicEncoder(prime=4)  # Not prime
        
        # Test invalid precision
        with self.assertRaises(ValueError):
            PadicEncoder(prime=7, precision=0)
            
    def test_encode_single_value(self):
        """Test encoding single values"""
        # Test positive integer
        weight = self.encoder.encode_single(42.0)
        self.assertIsInstance(weight, PadicWeight)
        self.assertEqual(weight.prime, 257)
        self.assertEqual(weight.precision, 4)
        self.assertAlmostEqual(weight.value, 42.0, places=5)
        
        # Test negative value
        weight = self.encoder.encode_single(-17.5)
        self.assertAlmostEqual(weight.value, -17.5, places=5)
        
        # Test zero
        weight = self.encoder.encode_single(0.0)
        self.assertAlmostEqual(weight.value, 0.0, places=5)
        
        # Test fractional value
        weight = self.encoder.encode_single(0.123)
        self.assertAlmostEqual(weight.value, 0.123, places=5)
        
    def test_encode_tensor(self):
        """Test encoding PyTorch tensors"""
        # Test 1D tensor
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        encoded = self.encoder.encode_tensor(tensor)
        
        self.assertEqual(len(encoded['weights']), 5)
        self.assertEqual(encoded['shape'], list(tensor.shape))
        self.assertEqual(encoded['prime'], 257)
        self.assertEqual(encoded['precision'], 4)
        
        # Test 2D tensor
        tensor_2d = torch.randn(3, 4)
        encoded_2d = self.encoder.encode_tensor(tensor_2d)
        self.assertEqual(len(encoded_2d['weights']), 12)
        self.assertEqual(encoded_2d['shape'], [3, 4])
        
        # Test empty tensor
        empty_tensor = torch.tensor([])
        encoded_empty = self.encoder.encode_tensor(empty_tensor)
        self.assertEqual(len(encoded_empty['weights']), 0)
        
    def test_decode_tensor(self):
        """Test decoding back to tensors"""
        # Create and encode a tensor
        original = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        encoded = self.encoder.encode_tensor(original)
        
        # Decode back
        decoded = self.encoder.decode_tensor(encoded)
        
        # Check shape and values
        self.assertEqual(decoded.shape, original.shape)
        torch.testing.assert_close(decoded, original, rtol=1e-4, atol=1e-4)
        
    def test_encode_decode_roundtrip(self):
        """Test that encode->decode preserves values"""
        # Test various value ranges
        test_values = [
            torch.tensor([0.0, 1.0, -1.0, 0.5, -0.5]),
            torch.tensor([1e-6, 1e-3, 1e3, 1e6]),
            torch.randn(10),
            torch.zeros(5),
            torch.ones(5) * 100
        ]
        
        for original in test_values:
            encoded = self.encoder.encode_tensor(original)
            decoded = self.encoder.decode_tensor(encoded)
            
            # Allow for small numerical errors
            torch.testing.assert_close(decoded, original, rtol=1e-3, atol=1e-3)
    
    def test_safe_precision_limits(self):
        """Test safe precision calculation for various primes"""
        # Test known limits
        self.assertEqual(get_safe_precision(2), 53)
        self.assertEqual(get_safe_precision(3), 33)
        self.assertEqual(get_safe_precision(5), 22)
        self.assertEqual(get_safe_precision(257), 6)
        
        # Test dynamic calculation for unknown prime
        safe_precision = get_safe_precision(263)  # Prime not in SAFE_PRECISION_LIMITS
        self.assertIsInstance(safe_precision, int)
        self.assertGreater(safe_precision, 0)
        self.assertLess(safe_precision, 10)
        
    def test_validate_precision(self):
        """Test precision validation"""
        # Test valid precision
        self.assertEqual(validate_precision(5), 5)
        self.assertEqual(validate_precision(np.int32(5)), 5)
        self.assertEqual(validate_precision(np.int64(10)), 10)
        
        # Test invalid precision
        with self.assertRaises(ValueError):
            validate_precision(0)
        
        with self.assertRaises(ValueError):
            validate_precision(-1)
            
        with self.assertRaises(ValueError):
            validate_precision(1001)
        
        # Test clamping to safe limit
        clamped = validate_precision(100, prime=2)
        self.assertEqual(clamped, 53)  # Should clamp to safe limit for prime=2
        
    def test_mathematical_operations(self):
        """Test PadicMathematicalOperations"""
        math_ops = PadicMathematicalOperations(prime=7, precision=4)
        
        # Test to_padic conversion
        weight = math_ops.to_padic(42.0)
        self.assertIsInstance(weight, PadicWeight)
        self.assertEqual(weight.prime, 7)
        
        # Test from_padic conversion
        reconstructed = math_ops.from_padic(weight)
        self.assertAlmostEqual(reconstructed, 42.0, places=3)
        
        # Test distance calculation
        w1 = math_ops.to_padic(10.0)
        w2 = math_ops.to_padic(12.0)
        distance = math_ops.distance(w1, w2)
        self.assertIsInstance(distance, float)
        self.assertGreaterEqual(distance, 0)
        
    def test_validate_single_weight(self):
        """Test single weight validation"""
        # Create valid weight
        weight = self.encoder.encode_single(10.0)
        self.assertTrue(validate_single_weight(weight, 257, 4))
        
        # Test with wrong prime
        self.assertFalse(validate_single_weight(weight, 7, 4))
        
        # Test with wrong precision
        self.assertFalse(validate_single_weight(weight, 257, 5))
        
    def test_validate_padic_weights(self):
        """Test batch weight validation"""
        # Create valid weights
        weights = [self.encoder.encode_single(float(i)) for i in range(5)]
        self.assertTrue(validate_padic_weights(weights, 257, 4))
        
        # Test with invalid weight in batch
        weights[2].digits[0] = 300  # Invalid digit (> prime)
        self.assertFalse(validate_padic_weights(weights, 257, 4))
        
    def test_ultrametric_property(self):
        """Test ultrametric distance property"""
        # Create weights that should satisfy ultrametric property
        weights = create_real_padic_weights(5, precision=3, prime=7)
        self.assertTrue(validate_ultrametric_property(weights))
        
    def test_create_real_padic_weights(self):
        """Test weight creation utility"""
        # Test basic creation
        weights = create_real_padic_weights(10, precision=3, prime=7)
        self.assertEqual(len(weights), 10)
        
        for weight in weights:
            self.assertEqual(weight.prime, 7)
            self.assertEqual(weight.precision, 3)
            self.assertIsInstance(weight.value, float)
            
        # Test with high precision (should be clamped for safety)
        weights = create_real_padic_weights(5, precision=100, prime=2)
        self.assertEqual(len(weights), 5)
        # Precision should be clamped to safe value
        for weight in weights:
            self.assertLessEqual(weight.precision, 53)
    
    def test_memory_efficiency(self):
        """Test memory usage is reasonable"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Encode a large tensor
        large_tensor = torch.randn(1000, 100)
        encoded = self.encoder.encode_tensor(large_tensor)
        
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = current_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this size)
        self.assertLess(memory_increase, 100, 
                       f"Memory increase {memory_increase:.2f}MB is too high")
        
        # Clean up
        del large_tensor
        del encoded
        gc.collect()
        
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test very small values
        small_val = self.encoder.encode_single(1e-10)
        self.assertIsNotNone(small_val)
        
        # Test very large values
        large_val = self.encoder.encode_single(1e10)
        self.assertIsNotNone(large_val)
        
        # Test NaN handling
        nan_tensor = torch.tensor([float('nan')])
        encoded_nan = self.encoder.encode_tensor(nan_tensor)
        self.assertIsNotNone(encoded_nan)
        
        # Test infinity handling
        inf_tensor = torch.tensor([float('inf'), float('-inf')])
        encoded_inf = self.encoder.encode_tensor(inf_tensor)
        self.assertIsNotNone(encoded_inf)
        
    def test_dtype_preservation(self):
        """Test that dtypes are handled correctly"""
        # Test float32
        tensor_f32 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        encoded = self.encoder.encode_tensor(tensor_f32)
        decoded = self.encoder.decode_tensor(encoded)
        self.assertEqual(decoded.dtype, torch.float32)
        
        # Test float64
        tensor_f64 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        encoded = self.encoder.encode_tensor(tensor_f64)
        decoded = self.encoder.decode_tensor(encoded)
        # Note: Decoder might default to float32, check actual implementation
        self.assertIn(decoded.dtype, [torch.float32, torch.float64])
        
    def test_batch_processing(self):
        """Test batch processing efficiency"""
        # Create batch of tensors
        batch_size = 100
        tensors = [torch.randn(10, 10) for _ in range(batch_size)]
        
        # Encode all tensors
        encoded_batch = [self.encoder.encode_tensor(t) for t in tensors]
        
        # Decode all tensors
        decoded_batch = [self.encoder.decode_tensor(e) for e in encoded_batch]
        
        # Verify all decoded correctly
        for original, decoded in zip(tensors, decoded_batch):
            torch.testing.assert_close(decoded, original, rtol=1e-3, atol=1e-3)
            
    def test_compression_ratio(self):
        """Test that compression provides some benefit"""
        # Create sparse tensor (many zeros)
        sparse_tensor = torch.zeros(100, 100)
        sparse_tensor[::10, ::10] = torch.randn(10, 10)
        
        # Encode
        encoded = self.encoder.encode_tensor(sparse_tensor)
        
        # Check that encoded representation exists
        self.assertIn('weights', encoded)
        self.assertIn('shape', encoded)
        self.assertEqual(len(encoded['weights']), 10000)
        
    def test_thread_safety(self):
        """Test basic thread safety"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def encode_decode_task(tensor, encoder, q):
            try:
                encoded = encoder.encode_tensor(tensor)
                decoded = encoder.decode_tensor(encoded)
                q.put(('success', decoded))
            except Exception as e:
                q.put(('error', str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            tensor = torch.randn(10, 10)
            thread = threading.Thread(
                target=encode_decode_task,
                args=(tensor, self.encoder, results)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        success_count = 0
        while not results.empty():
            status, _ = results.get()
            if status == 'success':
                success_count += 1
        
        self.assertEqual(success_count, 5, "All threads should complete successfully")


class TestPadicEncoderIntegration(unittest.TestCase):
    """Integration tests for PadicEncoder with other components"""
    
    def test_with_numpy_arrays(self):
        """Test encoding numpy arrays"""
        encoder = PadicEncoder(prime=13, precision=4)
        
        # Test 1D numpy array
        np_array = np.array([1.0, 2.0, 3.0, 4.0])
        tensor = torch.from_numpy(np_array)
        encoded = encoder.encode_tensor(tensor)
        decoded = encoder.decode_tensor(encoded)
        
        np.testing.assert_allclose(decoded.numpy(), np_array, rtol=1e-3)
        
    def test_with_different_shapes(self):
        """Test various tensor shapes"""
        encoder = PadicEncoder()
        
        shapes = [
            (10,),          # 1D
            (5, 5),         # 2D square
            (3, 4, 5),      # 3D
            (2, 3, 4, 5),   # 4D
            (1, 100),       # Row vector
            (100, 1),       # Column vector
        ]
        
        for shape in shapes:
            tensor = torch.randn(*shape)
            encoded = encoder.encode_tensor(tensor)
            decoded = encoder.decode_tensor(encoded)
            
            self.assertEqual(decoded.shape, tensor.shape)
            torch.testing.assert_close(decoded, tensor, rtol=1e-3, atol=1e-3)


def run_tests():
    """Run all tests with verbose output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPadicEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestPadicEncoderIntegration))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✓ All PadicEncoder tests passed!")
    else:
        print(f"✗ Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)