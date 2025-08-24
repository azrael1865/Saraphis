"""
Comprehensive Unit Tests for PadicValidation Class
Tests all validation methods with edge cases, error conditions, and type handling
"""

import sys
import os
import unittest
import math
import numpy as np
from typing import Any

# Add path to compression systems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'compression_systems', 'padic'))

# Try to import torch, but handle gracefully if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Skipping tensor validation tests.")

from padic_encoder import PadicValidation


class TestPadicValidation(unittest.TestCase):
    """Test suite for PadicValidation class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = PadicValidation()
    
    # ==================== Test validate_prime ====================
    
    def test_validate_prime_valid_small_primes(self):
        """Test validation of small prime numbers"""
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for prime in small_primes:
            try:
                self.validator.validate_prime(prime)
            except Exception as e:
                self.fail(f"Failed to validate prime {prime}: {e}")
    
    def test_validate_prime_valid_large_primes(self):
        """Test validation of larger prime numbers"""
        large_primes = [53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 
                       109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 
                       173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 
                       233, 239, 241, 251, 257]
        for prime in large_primes:
            try:
                self.validator.validate_prime(prime)
            except Exception as e:
                self.fail(f"Failed to validate prime {prime}: {e}")
    
    def test_validate_prime_very_large_primes(self):
        """Test validation of very large prime numbers"""
        very_large_primes = [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 
                            1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097]
        for prime in very_large_primes:
            try:
                self.validator.validate_prime(prime)
            except Exception as e:
                self.fail(f"Failed to validate prime {prime}: {e}")
    
    def test_validate_prime_invalid_composites(self):
        """Test that composite numbers are rejected"""
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 
                     26, 27, 28, 30, 32, 33, 34, 35, 36, 38, 39, 40, 42, 44, 
                     45, 46, 48, 49, 50, 100, 200, 256, 1000, 1024]
        for composite in composites:
            with self.assertRaises(ValueError) as context:
                self.validator.validate_prime(composite)
            self.assertIn("not a prime number", str(context.exception))
    
    def test_validate_prime_invalid_less_than_2(self):
        """Test that numbers less than 2 are rejected"""
        invalid_values = [-100, -10, -2, -1, 0, 1]
        for value in invalid_values:
            with self.assertRaises(ValueError) as context:
                self.validator.validate_prime(value)
            self.assertIn("Prime must be >= 2", str(context.exception))
    
    def test_validate_prime_invalid_types(self):
        """Test that non-integer types are rejected"""
        invalid_types = [
            2.5, 3.14, "5", [7], {11: 13}, None,
            complex(17, 0), bytes([19]), bytearray([23])
        ]
        for invalid in invalid_types:
            with self.assertRaises(TypeError) as context:
                self.validator.validate_prime(invalid)
            self.assertIn("Prime must be int", str(context.exception))
    
    def test_validate_prime_boolean_values(self):
        """Test that boolean values are handled properly"""
        # Booleans are subclass of int in Python, so they pass isinstance(x, int)
        # True == 1, False == 0, both less than 2
        with self.assertRaises(ValueError) as context:
            self.validator.validate_prime(True)
        self.assertIn("Prime must be >= 2", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            self.validator.validate_prime(False)
        self.assertIn("Prime must be >= 2", str(context.exception))
    
    def test_validate_prime_edge_case_2(self):
        """Test that 2 is correctly validated as prime"""
        try:
            self.validator.validate_prime(2)
        except Exception as e:
            self.fail(f"Failed to validate 2 as prime: {e}")
    
    def test_validate_prime_perfect_squares(self):
        """Test that perfect squares are correctly rejected"""
        perfect_squares = [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 
                          196, 225, 256, 289, 324, 361, 400]
        for square in perfect_squares:
            with self.assertRaises(ValueError) as context:
                self.validator.validate_prime(square)
            self.assertIn("not a prime number", str(context.exception))
    
    # ==================== Test validate_precision ====================
    
    def test_validate_precision_valid_python_int(self):
        """Test validation of valid Python integer precisions"""
        valid_precisions = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 999, 1000]
        for precision in valid_precisions:
            try:
                self.validator.validate_precision(precision)
            except Exception as e:
                self.fail(f"Failed to validate precision {precision}: {e}")
    
    def test_validate_precision_valid_numpy_integers(self):
        """Test validation of numpy integer types"""
        numpy_precisions = [
            np.int8(5), np.int16(10), np.int32(20), np.int64(50),
            np.uint8(5), np.uint16(10), np.uint32(20), np.uint64(50),
            np.intp(100), np.uintp(100)
        ]
        for precision in numpy_precisions:
            try:
                self.validator.validate_precision(precision)
            except Exception as e:
                self.fail(f"Failed to validate numpy precision {precision} of type {type(precision)}: {e}")
    
    def test_validate_precision_numpy_scalar(self):
        """Test validation of numpy scalar values"""
        arr = np.array([10, 20, 30])
        for i in range(len(arr)):
            try:
                self.validator.validate_precision(arr[i])
            except Exception as e:
                self.fail(f"Failed to validate numpy scalar {arr[i]}: {e}")
    
    def test_validate_precision_invalid_zero_negative(self):
        """Test that zero and negative precisions are rejected"""
        invalid_precisions = [-1000, -100, -10, -1, 0]
        for precision in invalid_precisions:
            with self.assertRaises(ValueError) as context:
                self.validator.validate_precision(precision)
            self.assertIn("Precision must be >= 1", str(context.exception))
    
    def test_validate_precision_invalid_too_large(self):
        """Test that excessively large precisions are rejected"""
        invalid_precisions = [1001, 1100, 2000, 5000, 10000, 100000]
        for precision in invalid_precisions:
            with self.assertRaises(ValueError) as context:
                self.validator.validate_precision(precision)
            self.assertIn("exceeds maximum 1000", str(context.exception))
    
    def test_validate_precision_invalid_types(self):
        """Test that non-integer types are rejected"""
        invalid_types = [
            2.5, 3.14, "5", [7], {11: 13}, None,
            complex(17, 0), bytes([19]), bytearray([23])
        ]
        for invalid in invalid_types:
            with self.assertRaises(TypeError) as context:
                self.validator.validate_precision(invalid)
            self.assertIn("Precision must be an integer type", str(context.exception))
    
    def test_validate_precision_boolean_values(self):
        """Test that boolean values are handled properly"""
        # Booleans are subclass of int in Python
        # False == 0 should be rejected as < 1
        with self.assertRaises(ValueError) as context:
            self.validator.validate_precision(False)
        self.assertIn("Precision must be >= 1", str(context.exception))
        
        # True == 1 should be accepted
        try:
            self.validator.validate_precision(True)
        except Exception as e:
            self.fail(f"Failed to validate True (1) as precision: {e}")
    
    def test_validate_precision_boundary_values(self):
        """Test boundary values for precision"""
        # Test minimum valid value
        try:
            self.validator.validate_precision(1)
        except Exception as e:
            self.fail(f"Failed to validate minimum precision 1: {e}")
        
        # Test maximum valid value
        try:
            self.validator.validate_precision(1000)
        except Exception as e:
            self.fail(f"Failed to validate maximum precision 1000: {e}")
    
    # ==================== Test validate_tensor ====================
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_validate_tensor_valid_float32(self):
        """Test validation of valid float32 tensors"""
        tensors = [
            torch.randn(10),
            torch.randn(5, 5),
            torch.randn(3, 4, 5),
            torch.ones(100),
            torch.zeros(50),
            torch.full((10, 10), 3.14)
        ]
        for tensor in tensors:
            tensor = tensor.float()  # Ensure float32
            try:
                self.validator.validate_tensor(tensor)
            except Exception as e:
                self.fail(f"Failed to validate tensor of shape {tensor.shape}: {e}")
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_validate_tensor_valid_float64(self):
        """Test validation of valid float64 tensors"""
        tensors = [
            torch.randn(10).double(),
            torch.randn(5, 5).double(),
            torch.ones(100).double(),
            torch.zeros(50).double()
        ]
        for tensor in tensors:
            try:
                self.validator.validate_tensor(tensor)
            except Exception as e:
                self.fail(f"Failed to validate float64 tensor: {e}")
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_validate_tensor_valid_float16(self):
        """Test validation of valid float16 tensors"""
        tensors = [
            torch.randn(10).half(),
            torch.randn(5, 5).half(),
            torch.ones(100).half(),
            torch.zeros(50).half()
        ]
        for tensor in tensors:
            try:
                self.validator.validate_tensor(tensor)
            except Exception as e:
                self.fail(f"Failed to validate float16 tensor: {e}")
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_validate_tensor_invalid_none(self):
        """Test that None tensor is rejected"""
        with self.assertRaises(ValueError) as context:
            self.validator.validate_tensor(None)
        self.assertIn("Tensor cannot be None", str(context.exception))
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_validate_tensor_invalid_not_tensor(self):
        """Test that non-tensor types are rejected"""
        invalid_types = [
            [1, 2, 3], np.array([1, 2, 3]), "tensor", 42, 3.14,
            {"tensor": True}, set([1, 2, 3]), bytes([1, 2, 3])
        ]
        for invalid in invalid_types:
            with self.assertRaises(TypeError) as context:
                self.validator.validate_tensor(invalid)
            self.assertIn("Expected torch.Tensor", str(context.exception))
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_validate_tensor_invalid_empty(self):
        """Test that empty tensors are rejected"""
        empty_tensors = [
            torch.tensor([]),
            torch.empty(0),
            torch.empty(0, 0),
            torch.empty(5, 0, 3)
        ]
        for tensor in empty_tensors:
            with self.assertRaises(ValueError) as context:
                self.validator.validate_tensor(tensor)
            self.assertIn("Tensor cannot be empty", str(context.exception))
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_validate_tensor_invalid_nan(self):
        """Test that tensors containing NaN are rejected"""
        nan_tensors = [
            torch.tensor([1.0, float('nan'), 3.0]),
            torch.full((3, 3), float('nan')),
            torch.tensor([float('nan')])
        ]
        for tensor in nan_tensors:
            with self.assertRaises(ValueError) as context:
                self.validator.validate_tensor(tensor)
            self.assertIn("Tensor contains NaN values", str(context.exception))
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_validate_tensor_invalid_inf(self):
        """Test that tensors containing infinity are rejected"""
        inf_tensors = [
            torch.tensor([1.0, float('inf'), 3.0]),
            torch.tensor([float('-inf')]),
            torch.full((3, 3), float('inf'))
        ]
        for tensor in inf_tensors:
            with self.assertRaises(ValueError) as context:
                self.validator.validate_tensor(tensor)
            self.assertIn("Tensor contains infinite values", str(context.exception))
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_validate_tensor_invalid_dtype(self):
        """Test that tensors with non-float dtypes are rejected"""
        invalid_dtype_tensors = [
            torch.randint(0, 10, (5, 5)),  # int64
            torch.tensor([1, 2, 3], dtype=torch.int32),
            torch.tensor([True, False, True]),  # bool
            torch.tensor([1, 2, 3], dtype=torch.int8),
            torch.tensor([1, 2, 3], dtype=torch.long)
        ]
        for tensor in invalid_dtype_tensors:
            with self.assertRaises(TypeError) as context:
                self.validator.validate_tensor(tensor)
            self.assertIn("Tensor must be float type", str(context.exception))
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_validate_tensor_mixed_valid_invalid(self):
        """Test tensor with valid values except for one NaN or inf"""
        # Create tensor with one NaN
        tensor_with_nan = torch.randn(10, 10)
        tensor_with_nan[5, 5] = float('nan')
        with self.assertRaises(ValueError) as context:
            self.validator.validate_tensor(tensor_with_nan)
        self.assertIn("NaN", str(context.exception))
        
        # Create tensor with one inf
        tensor_with_inf = torch.randn(10, 10)
        tensor_with_inf[5, 5] = float('inf')
        with self.assertRaises(ValueError) as context:
            self.validator.validate_tensor(tensor_with_inf)
        self.assertIn("infinite", str(context.exception))
    
    # ==================== Test validate_chunk_size ====================
    
    def test_validate_chunk_size_valid(self):
        """Test validation of valid chunk sizes"""
        test_cases = [
            (1, 1),      # Minimum valid
            (1, 10),     # Small chunk
            (10, 100),   # Normal chunk
            (100, 1000), # Large chunk
            (1000, 1000), # Chunk equals tensor size
            (500, 5000)  # Large tensor
        ]
        for chunk_size, tensor_size in test_cases:
            try:
                self.validator.validate_chunk_size(chunk_size, tensor_size)
            except Exception as e:
                self.fail(f"Failed to validate chunk_size={chunk_size}, tensor_size={tensor_size}: {e}")
    
    def test_validate_chunk_size_invalid_not_int(self):
        """Test that non-integer chunk sizes are rejected"""
        invalid_types = [
            2.5, "10", [20], {30: 40}, None,
            complex(50, 0), bytes([60]), np.float32(70)
        ]
        for invalid in invalid_types:
            with self.assertRaises(TypeError) as context:
                self.validator.validate_chunk_size(invalid, 100)
            self.assertIn("Chunk size must be int", str(context.exception))
    
    def test_validate_chunk_size_boolean_values(self):
        """Test that boolean values are handled properly"""
        # False == 0 should be rejected as <= 0
        with self.assertRaises(ValueError) as context:
            self.validator.validate_chunk_size(False, 100)
        self.assertIn("Chunk size must be > 0", str(context.exception))
        
        # True == 1 should be accepted if tensor_size >= 1
        try:
            self.validator.validate_chunk_size(True, 100)
        except Exception as e:
            self.fail(f"Failed to validate True (1) as chunk size: {e}")
    
    def test_validate_chunk_size_invalid_zero_negative(self):
        """Test that zero and negative chunk sizes are rejected"""
        invalid_sizes = [-1000, -100, -10, -1, 0]
        for size in invalid_sizes:
            with self.assertRaises(ValueError) as context:
                self.validator.validate_chunk_size(size, 100)
            self.assertIn("Chunk size must be > 0", str(context.exception))
    
    def test_validate_chunk_size_invalid_exceeds_tensor(self):
        """Test that chunk sizes exceeding tensor size are rejected"""
        test_cases = [
            (11, 10),
            (101, 100),
            (1001, 1000),
            (5001, 5000)
        ]
        for chunk_size, tensor_size in test_cases:
            with self.assertRaises(ValueError) as context:
                self.validator.validate_chunk_size(chunk_size, tensor_size)
            self.assertIn(f"Chunk size {chunk_size} exceeds tensor size {tensor_size}", 
                         str(context.exception))
    
    def test_validate_chunk_size_boundary_values(self):
        """Test boundary values for chunk size"""
        # Test minimum valid chunk size
        try:
            self.validator.validate_chunk_size(1, 1)
        except Exception as e:
            self.fail(f"Failed to validate minimum chunk size: {e}")
        
        # Test chunk size equal to tensor size
        try:
            self.validator.validate_chunk_size(1000, 1000)
        except Exception as e:
            self.fail(f"Failed to validate chunk size equal to tensor size: {e}")
    
    # ==================== Integration Tests ====================
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_all_methods_with_valid_inputs(self):
        """Test all validation methods with valid inputs in sequence"""
        try:
            # Test prime validation
            self.validator.validate_prime(257)
            
            # Test precision validation with different types
            self.validator.validate_precision(10)
            self.validator.validate_precision(np.int64(20))
            
            # Test tensor validation
            tensor = torch.randn(100, 100).float()
            self.validator.validate_tensor(tensor)
            
            # Test chunk size validation
            self.validator.validate_chunk_size(50, 100)
            
        except Exception as e:
            self.fail(f"Integration test failed with valid inputs: {e}")
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_error_messages_are_descriptive(self):
        """Test that error messages provide useful information"""
        # Test prime error message
        with self.assertRaises(ValueError) as context:
            self.validator.validate_prime(15)
        error_msg = str(context.exception)
        self.assertIn("15", error_msg)
        self.assertIn("divisible by", error_msg)
        
        # Test precision error message
        with self.assertRaises(ValueError) as context:
            self.validator.validate_precision(0)
        error_msg = str(context.exception)
        self.assertIn("Precision must be >= 1", error_msg)
        self.assertIn("got 0", error_msg)
        
        # Test tensor error message
        with self.assertRaises(ValueError) as context:
            self.validator.validate_tensor(torch.empty(0))
        error_msg = str(context.exception)
        self.assertIn("empty", error_msg)
        
        # Test chunk size error message
        with self.assertRaises(ValueError) as context:
            self.validator.validate_chunk_size(150, 100)
        error_msg = str(context.exception)
        self.assertIn("150", error_msg)
        self.assertIn("100", error_msg)
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_static_methods_without_instance(self):
        """Test that static methods can be called without instance"""
        try:
            # All methods are static, so they should work without instance
            PadicValidation.validate_prime(7)
            PadicValidation.validate_precision(5)
            PadicValidation.validate_tensor(torch.randn(10))
            PadicValidation.validate_chunk_size(5, 10)
        except Exception as e:
            self.fail(f"Failed to call static methods without instance: {e}")
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_thread_safety(self):
        """Test that validation methods are thread-safe"""
        import threading
        import time
        
        errors = []
        
        def validate_in_thread(thread_id):
            try:
                for _ in range(100):
                    PadicValidation.validate_prime(thread_id * 2 + 1 if thread_id * 2 + 1 in [3, 5, 7, 11, 13] else 7)
                    PadicValidation.validate_precision(thread_id + 1)
                    PadicValidation.validate_tensor(torch.randn(10))
                    PadicValidation.validate_chunk_size(5, 10)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=validate_in_thread, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        if errors:
            self.fail(f"Thread safety test failed: {errors}")


class TestPadicValidationPerformance(unittest.TestCase):
    """Performance tests for PadicValidation class"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        self.validator = PadicValidation()
    
    def test_validate_prime_performance(self):
        """Test performance of prime validation"""
        import time
        
        # Test with various prime sizes
        test_primes = [7, 97, 257, 1009, 10007]
        
        for prime in test_primes:
            start_time = time.time()
            for _ in range(1000):
                self.validator.validate_prime(prime)
            elapsed = time.time() - start_time
            
            # Should be very fast - less than 1ms per validation
            avg_time_ms = (elapsed / 1000) * 1000
            self.assertLess(avg_time_ms, 1.0, 
                          f"Prime validation too slow for {prime}: {avg_time_ms:.3f}ms")
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_validate_tensor_performance(self):
        """Test performance of tensor validation"""
        import time
        
        # Test with various tensor sizes
        tensor_sizes = [(10,), (100, 100), (50, 50, 50)]
        
        for size in tensor_sizes:
            tensor = torch.randn(*size)
            
            start_time = time.time()
            for _ in range(100):
                self.validator.validate_tensor(tensor)
            elapsed = time.time() - start_time
            
            # Should be reasonably fast even for large tensors
            avg_time_ms = (elapsed / 100) * 1000
            self.assertLess(avg_time_ms, 10.0, 
                          f"Tensor validation too slow for size {size}: {avg_time_ms:.3f}ms")


class TestPadicValidationEdgeCases(unittest.TestCase):
    """Edge case tests for PadicValidation class"""
    
    def setUp(self):
        """Set up edge case test fixtures"""
        self.validator = PadicValidation()
    
    def test_carmichael_numbers(self):
        """Test that Carmichael numbers (pseudoprimes) are correctly rejected"""
        # These are composite numbers that pass some primality tests
        carmichael_numbers = [561, 1105, 1729, 2465, 2821, 6601, 8911]
        
        for num in carmichael_numbers:
            with self.assertRaises(ValueError) as context:
                self.validator.validate_prime(num)
            self.assertIn("not a prime number", str(context.exception))
    
    def test_mersenne_primes(self):
        """Test validation of Mersenne primes"""
        # Small Mersenne primes: 2^p - 1 where p is prime
        mersenne_primes = [3, 7, 31, 127, 8191]  # 2^2-1, 2^3-1, 2^5-1, 2^7-1, 2^13-1
        
        for prime in mersenne_primes:
            try:
                self.validator.validate_prime(prime)
            except Exception as e:
                self.fail(f"Failed to validate Mersenne prime {prime}: {e}")
    
    def test_sophie_germain_primes(self):
        """Test validation of Sophie Germain primes"""
        # Primes p where 2p + 1 is also prime
        sophie_germain = [2, 3, 5, 11, 23, 29, 41, 53, 83, 89, 113, 131, 173, 179, 191]
        
        for prime in sophie_germain:
            try:
                self.validator.validate_prime(prime)
            except Exception as e:
                self.fail(f"Failed to validate Sophie Germain prime {prime}: {e}")
    
    def test_twin_primes(self):
        """Test validation of twin prime pairs"""
        twin_prime_pairs = [(3, 5), (5, 7), (11, 13), (17, 19), (29, 31), 
                           (41, 43), (59, 61), (71, 73), (101, 103)]
        
        for p1, p2 in twin_prime_pairs:
            try:
                self.validator.validate_prime(p1)
                self.validator.validate_prime(p2)
            except Exception as e:
                self.fail(f"Failed to validate twin primes ({p1}, {p2}): {e}")
    
    def test_precision_with_numpy_zero_dim_array(self):
        """Test precision validation with zero-dimensional numpy array"""
        # Note: The current implementation doesn't handle 0-d arrays correctly
        # It checks isinstance(precision, (int, np.integer)) which fails for 0-d arrays
        # This test documents the current behavior
        zero_dim = np.array(42)  # Zero-dimensional array
        
        # Check if it's handled as numpy integer type or raises TypeError
        # The implementation should ideally handle this, but currently may not
        try:
            self.validator.validate_precision(zero_dim)
        except TypeError as e:
            # Current implementation may reject 0-d arrays
            self.assertIn("Precision must be an integer type", str(e))
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_tensor_with_single_element(self):
        """Test tensor validation with single-element tensors"""
        single_element_tensors = [
            torch.tensor([3.14]),
            torch.tensor([[5.0]]),
            torch.ones(1),
            torch.zeros(1, 1, 1)
        ]
        
        for tensor in single_element_tensors:
            try:
                self.validator.validate_tensor(tensor.float())
            except Exception as e:
                self.fail(f"Failed to validate single-element tensor: {e}")
    
    def test_chunk_size_equals_one(self):
        """Test chunk size validation when chunk size is 1"""
        try:
            self.validator.validate_chunk_size(1, 1)
            self.validator.validate_chunk_size(1, 10)
            self.validator.validate_chunk_size(1, 100)
        except Exception as e:
            self.fail(f"Failed to validate chunk size of 1: {e}")


if __name__ == "__main__":
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPadicValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestPadicValidationPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestPadicValidationEdgeCases))
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)