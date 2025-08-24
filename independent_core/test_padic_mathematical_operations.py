"""
Comprehensive unit tests for PadicMathematicalOperations component.
Tests all mathematical operations, conversions, and edge cases.
"""

import unittest
import torch
import numpy as np
import math
from fractions import Fraction
from typing import List
import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compression_systems.padic.padic_encoder import (
    PadicMathematicalOperations,
    PadicWeight,
    PadicValidation,
    get_safe_precision,
    validate_precision,
    validate_single_weight,
    validate_padic_weights,
    validate_ultrametric_property,
    create_real_padic_weights
)


class TestPadicMathematicalOperations(unittest.TestCase):
    """Test suite for PadicMathematicalOperations class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.default_prime = 257
        self.default_precision = 4
        self.math_ops = PadicMathematicalOperations(self.default_prime, self.default_precision)
        
        # Common test values
        self.test_values = [0, 1, -1, 0.5, -0.5, 2.5, -2.5, 10, -10, 0.1, -0.1, 100, -100]
        self.test_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 127, 257]
        self.test_precisions = [1, 2, 3, 4, 5, 6, 7, 8]
    
    # ============= Initialization and Validation Tests =============
    
    def test_initialization_valid_parameters(self):
        """Test initialization with valid parameters"""
        for prime in self.test_primes:
            safe_precision = get_safe_precision(prime)
            for precision in range(1, min(safe_precision + 1, 10)):
                ops = PadicMathematicalOperations(prime, precision)
                self.assertEqual(ops.prime, prime)
                self.assertEqual(ops.precision, precision)
                self.assertIsNotNone(ops.prime_powers)
                self.assertTrue(len(ops.prime_powers) >= 1)
    
    def test_initialization_invalid_prime(self):
        """Test initialization with invalid prime values"""
        invalid_primes = [0, 1, -1, 4, 6, 8, 9, 10, 1.5, "prime", None]
        for invalid_prime in invalid_primes:
            with self.assertRaises((TypeError, ValueError)):
                PadicMathematicalOperations(invalid_prime, 4)
    
    def test_initialization_invalid_precision(self):
        """Test initialization with invalid precision values"""
        invalid_precisions = [0, -1, 1001, 1.5, "precision", None]
        for invalid_precision in invalid_precisions:
            with self.assertRaises((TypeError, ValueError)):
                PadicMathematicalOperations(257, invalid_precision)
    
    def test_initialization_overflow_protection(self):
        """Test that initialization prevents overflow for large precision"""
        prime = 257
        # Request precision that would cause overflow
        unsafe_precision = 100  # Way beyond safe limit
        
        with self.assertRaises(OverflowError) as context:
            PadicMathematicalOperations(prime, unsafe_precision)
        
        self.assertIn("exceeds maximum safe precision", str(context.exception))
    
    def test_get_safe_precision(self):
        """Test safe precision calculation for various primes"""
        # Test known primes
        self.assertEqual(get_safe_precision(2), 53)
        self.assertEqual(get_safe_precision(3), 33)
        self.assertEqual(get_safe_precision(5), 22)
        self.assertEqual(get_safe_precision(257), 6)
        
        # Test unknown prime (should calculate dynamically)
        large_prime = 1009  # Not in the predefined list
        safe_precision = get_safe_precision(large_prime)
        self.assertGreater(safe_precision, 0)
        self.assertLess(safe_precision, 20)  # Should be reasonable
    
    def test_validate_precision(self):
        """Test precision validation function"""
        # Valid precisions
        self.assertEqual(validate_precision(1), 1)
        self.assertEqual(validate_precision(10), 10)
        self.assertEqual(validate_precision(100), 100)
        
        # Numpy integer types
        self.assertEqual(validate_precision(np.int32(5)), 5)
        self.assertEqual(validate_precision(np.int64(10)), 10)
        
        # With prime checking
        result = validate_precision(10, prime=257)
        self.assertLessEqual(result, get_safe_precision(257))
        
        # Invalid precisions
        with self.assertRaises(ValueError):
            validate_precision(0)
        with self.assertRaises(ValueError):
            validate_precision(-1)
        with self.assertRaises(ValueError):
            validate_precision(1001)
        with self.assertRaises(TypeError):
            validate_precision("5")
    
    # ============= P-adic Conversion Tests =============
    
    def test_to_padic_basic_values(self):
        """Test basic p-adic conversion for common values"""
        for value in self.test_values:
            weight = self.math_ops.to_padic(value)
            
            # Validate structure
            self.assertIsInstance(weight, PadicWeight)
            self.assertEqual(weight.prime, self.default_prime)
            self.assertEqual(weight.precision, self.default_precision)
            self.assertEqual(len(weight.digits), self.default_precision)
            
            # Validate digits are in correct range
            for digit in weight.digits:
                self.assertGreaterEqual(digit, 0)
                self.assertLess(digit, self.default_prime)
    
    def test_to_padic_numpy_types(self):
        """Test p-adic conversion with numpy numeric types"""
        numpy_values = [
            np.float32(1.5),
            np.float64(2.5),
            np.int32(10),
            np.int64(-5),
            np.array(3.14).item(),  # numpy scalar
        ]
        
        for value in numpy_values:
            weight = self.math_ops.to_padic(value)
            self.assertIsInstance(weight, PadicWeight)
            self.assertEqual(weight.prime, self.default_prime)
    
    def test_to_padic_special_cases(self):
        """Test p-adic conversion for special cases"""
        # Zero
        weight_zero = self.math_ops.to_padic(0)
        self.assertEqual(weight_zero.digits, [0] * self.default_precision)
        self.assertEqual(weight_zero.valuation, 0)
        
        # One
        weight_one = self.math_ops.to_padic(1)
        self.assertEqual(weight_one.digits[0], 1)
        
        # Very small values
        small_value = 1e-10
        weight_small = self.math_ops.to_padic(small_value)
        self.assertIsInstance(weight_small, PadicWeight)
        
        # Large values (within safe range)
        large_value = 1000
        weight_large = self.math_ops.to_padic(large_value)
        self.assertIsInstance(weight_large, PadicWeight)
    
    def test_to_padic_invalid_inputs(self):
        """Test p-adic conversion with invalid inputs"""
        # NaN
        with self.assertRaises(ValueError):
            self.math_ops.to_padic(float('nan'))
        
        # Infinity
        with self.assertRaises(ValueError):
            self.math_ops.to_padic(float('inf'))
        with self.assertRaises(ValueError):
            self.math_ops.to_padic(float('-inf'))
        
        # Too large values
        with self.assertRaises(ValueError):
            self.math_ops.to_padic(1e11)
        
        # Invalid types
        with self.assertRaises(TypeError):
            self.math_ops.to_padic("not a number")
        with self.assertRaises(TypeError):
            self.math_ops.to_padic([1, 2, 3])
    
    def test_from_padic_reconstruction_accuracy(self):
        """Test accuracy of p-adic to float reconstruction"""
        for value in self.test_values:
            # Skip zero for relative error calculation
            if value == 0:
                continue
                
            weight = self.math_ops.to_padic(value)
            reconstructed = self.math_ops.from_padic(weight)
            
            # Check relative error
            rel_error = abs(reconstructed - value) / abs(value)
            self.assertLess(rel_error, 1e-5, 
                          f"Failed for value {value}: reconstructed {reconstructed}, rel_error {rel_error}")
    
    def test_from_padic_invalid_inputs(self):
        """Test from_padic with invalid inputs"""
        # Wrong type
        with self.assertRaises(TypeError):
            self.math_ops.from_padic("not a weight")
        
        # Create weight with different prime
        other_ops = PadicMathematicalOperations(127, 4)
        other_weight = other_ops.to_padic(1.0)
        
        with self.assertRaises(ValueError):
            self.math_ops.from_padic(other_weight)
        
        # Create weight with different precision
        other_ops2 = PadicMathematicalOperations(257, 3)
        other_weight2 = other_ops2.to_padic(1.0)
        
        with self.assertRaises(ValueError):
            self.math_ops.from_padic(other_weight2)
    
    def test_round_trip_conversion(self):
        """Test that to_padic followed by from_padic preserves values"""
        test_values = [0.1, 0.5, 1.0, 2.0, 3.14159, -1.5, -10.0, 100.0]
        
        for value in test_values:
            weight = self.math_ops.to_padic(value)
            reconstructed = self.math_ops.from_padic(weight)
            
            # Allow small numerical error
            self.assertAlmostEqual(value, reconstructed, places=5,
                                  msg=f"Round trip failed for {value}")
    
    # ============= Batch Conversion Tests =============
    
    def test_batch_to_padic_list_input(self):
        """Test batch conversion with list input"""
        values = [1.0, 2.0, 3.0, -1.0, 0.5]
        weights = self.math_ops.batch_to_padic(values)
        
        self.assertEqual(len(weights), len(values))
        for weight in weights:
            self.assertIsInstance(weight, PadicWeight)
            self.assertEqual(weight.prime, self.default_prime)
            self.assertEqual(weight.precision, self.default_precision)
    
    def test_batch_to_padic_numpy_input(self):
        """Test batch conversion with numpy array input"""
        values = np.array([1.0, 2.0, 3.0, -1.0, 0.5])
        weights = self.math_ops.batch_to_padic(values)
        
        self.assertEqual(len(weights), len(values))
        for i, weight in enumerate(weights):
            self.assertIsInstance(weight, PadicWeight)
            reconstructed = self.math_ops.from_padic(weight)
            self.assertAlmostEqual(reconstructed, values[i], places=5)
    
    def test_batch_to_padic_torch_input(self):
        """Test batch conversion with torch tensor input"""
        values = torch.tensor([1.0, 2.0, 3.0, -1.0, 0.5])
        weights = self.math_ops.batch_to_padic(values)
        
        self.assertEqual(len(weights), values.numel())
        for i, weight in enumerate(weights):
            self.assertIsInstance(weight, PadicWeight)
            reconstructed = self.math_ops.from_padic(weight)
            self.assertAlmostEqual(reconstructed, values[i].item(), places=5)
    
    def test_batch_to_padic_with_invalid_element(self):
        """Test batch conversion with an invalid element"""
        values = [1.0, float('nan'), 3.0]
        
        with self.assertRaises(ValueError) as context:
            self.math_ops.batch_to_padic(values)
        
        # Should indicate which element failed
        self.assertIn("element 1", str(context.exception))
    
    # ============= Ultrametric Distance Tests =============
    
    def test_ultrametric_distance_identical_weights(self):
        """Test ultrametric distance between identical weights"""
        weight = self.math_ops.to_padic(1.5)
        distance = self.math_ops.ultrametric_distance(weight, weight)
        self.assertEqual(distance, 0.0)
    
    def test_ultrametric_distance_different_weights(self):
        """Test ultrametric distance between different weights"""
        weight1 = self.math_ops.to_padic(1.0)
        weight2 = self.math_ops.to_padic(2.0)
        
        distance = self.math_ops.ultrametric_distance(weight1, weight2)
        self.assertGreater(distance, 0)
        self.assertLessEqual(distance, 1.0)
    
    def test_ultrametric_property(self):
        """Test that ultrametric inequality holds"""
        weight1 = self.math_ops.to_padic(1.0)
        weight2 = self.math_ops.to_padic(2.0)
        weight3 = self.math_ops.to_padic(3.0)
        
        # Ultrametric inequality: d(x,z) <= max(d(x,y), d(y,z))
        d_12 = self.math_ops.ultrametric_distance(weight1, weight2)
        d_23 = self.math_ops.ultrametric_distance(weight2, weight3)
        d_13 = self.math_ops.ultrametric_distance(weight1, weight3)
        
        max_distance = max(d_12, d_23)
        self.assertLessEqual(d_13, max_distance + 1e-12)
    
    def test_validate_ultrametric_property(self):
        """Test ultrametric validation function"""
        weight1 = self.math_ops.to_padic(1.0)
        weight2 = self.math_ops.to_padic(2.0)
        weight3 = self.math_ops.to_padic(3.0)
        
        # Should not raise any exception
        self.math_ops.validate_ultrametric_property(weight1, weight2, weight3)
    
    def test_ultrametric_distance_invalid_inputs(self):
        """Test ultrametric distance with invalid inputs"""
        weight = self.math_ops.to_padic(1.0)
        
        # Wrong type
        with self.assertRaises(TypeError):
            self.math_ops.ultrametric_distance(weight, "not a weight")
        
        # Different prime
        other_ops = PadicMathematicalOperations(127, 4)
        other_weight = other_ops.to_padic(1.0)
        
        with self.assertRaises(ValueError):
            self.math_ops.ultrametric_distance(weight, other_weight)
    
    # ============= Dynamic Prime Switching Tests =============
    
    def test_switch_prime_dynamically(self):
        """Test dynamic prime switching"""
        original_prime = self.math_ops.prime
        new_prime = 127
        
        # Convert a value with original prime
        weight_before = self.math_ops.to_padic(1.5)
        
        # Switch prime
        self.math_ops.switch_prime_dynamically(new_prime)
        self.assertEqual(self.math_ops.prime, new_prime)
        
        # Convert a value with new prime
        weight_after = self.math_ops.to_padic(1.5)
        self.assertEqual(weight_after.prime, new_prime)
        
        # Switch back
        self.math_ops.switch_prime_dynamically(original_prime)
        self.assertEqual(self.math_ops.prime, original_prime)
    
    def test_switch_prime_no_change(self):
        """Test switching to the same prime (no-op)"""
        original_prime = self.math_ops.prime
        cache_size_before = len(self.math_ops._inverse_cache)
        
        self.math_ops.switch_prime_dynamically(original_prime)
        
        # Should remain unchanged
        self.assertEqual(self.math_ops.prime, original_prime)
        self.assertEqual(len(self.math_ops._inverse_cache), cache_size_before)
    
    def test_switch_prime_invalid(self):
        """Test switching to invalid prime"""
        with self.assertRaises(ValueError):
            self.math_ops.switch_prime_dynamically(4)  # Not prime
        
        with self.assertRaises(TypeError):
            self.math_ops.switch_prime_dynamically("prime")
    
    def test_switch_prime_overflow_protection(self):
        """Test that prime switching prevents overflow"""
        # Create with safe precision first
        ops = PadicMathematicalOperations(257, 4)
        
        # Switching to prime 2 with precision 4 should work
        ops.switch_prime_dynamically(2)
        self.assertEqual(ops.prime, 2)
        
        # Switch back to 257
        ops.switch_prime_dynamically(257)
        self.assertEqual(ops.prime, 257)
        
        # Now test overflow protection with high precision
        ops.precision = 100  # Set artificially high precision
        
        # This should trigger overflow protection
        with self.assertRaises(OverflowError):
            ops.switch_prime_dynamically(2)
        
        # Original prime should be preserved
        self.assertEqual(ops.prime, 257)
    
    # ============= Precision Factory Method Tests =============
    
    def test_create_with_precision(self):
        """Test factory method for creating ops with different precision"""
        new_precision = 6
        new_ops = self.math_ops.create_with_precision(new_precision)
        
        self.assertEqual(new_ops.prime, self.math_ops.prime)
        self.assertEqual(new_ops.precision, new_precision)
        self.assertIsNot(new_ops, self.math_ops)  # Different instance
    
    def test_create_with_precision_numpy_input(self):
        """Test factory method with numpy integer input"""
        new_precision = np.int32(5)
        new_ops = self.math_ops.create_with_precision(new_precision)
        
        self.assertEqual(new_ops.precision, 5)
    
    def test_create_with_precision_invalid(self):
        """Test factory method with invalid precision"""
        with self.assertRaises(ValueError):
            self.math_ops.create_with_precision(0)
        
        with self.assertRaises(ValueError):
            self.math_ops.create_with_precision(-1)
        
        with self.assertRaises(TypeError):
            self.math_ops.create_with_precision("precision")
    
    def test_create_with_precision_safety_clamping(self):
        """Test that factory method clamps to safe precision"""
        # Request very high precision
        requested_precision = 100
        
        # This should work without error and clamp to safe limit
        import logging
        logger = logging.getLogger()
        with self.assertLogs(logger=logger, level='WARNING'):
            new_ops = self.math_ops.create_with_precision(requested_precision)
        
        # Should be clamped to safe limit
        safe_limit = get_safe_precision(self.math_ops.prime)
        self.assertLessEqual(new_ops.precision, safe_limit)
    
    # ============= Edge Cases and Error Handling Tests =============
    
    def test_fractional_values(self):
        """Test handling of fractional values"""
        fractions = [1/3, 2/7, 5/11, 22/7, -3/5]
        
        for frac in fractions:
            weight = self.math_ops.to_padic(frac)
            reconstructed = self.math_ops.from_padic(weight)
            
            # Check relative accuracy
            if frac != 0:
                rel_error = abs(reconstructed - frac) / abs(frac)
                self.assertLess(rel_error, 1e-4)
    
    def test_negative_values(self):
        """Test handling of negative values"""
        negative_values = [-1, -2, -0.5, -10.5, -100]
        
        for value in negative_values:
            weight = self.math_ops.to_padic(value)
            self.assertIsInstance(weight, PadicWeight)
            
            # Verify we can reconstruct the negative value
            reconstructed = self.math_ops.from_padic(weight)
            rel_error = abs(reconstructed - value) / (abs(value) + 1e-10)
            self.assertLess(rel_error, 1e-4,
                          f"Negative value {value} not properly encoded/decoded")
    
    def test_very_small_values(self):
        """Test handling of very small values near zero"""
        small_values = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
        
        for value in small_values:
            weight = self.math_ops.to_padic(value)
            reconstructed = self.math_ops.from_padic(weight)
            
            # Very small values might be treated as zero
            self.assertTrue(abs(reconstructed) < 1e-5)
    
    def test_thread_safety(self):
        """Test thread safety of operations"""
        import threading
        import queue
        
        results = queue.Queue()
        errors = queue.Queue()
        
        def convert_values(thread_id):
            try:
                for i in range(10):
                    value = thread_id + i * 0.1
                    weight = self.math_ops.to_padic(value)
                    reconstructed = self.math_ops.from_padic(weight)
                    results.put((thread_id, value, reconstructed))
            except Exception as e:
                errors.put((thread_id, str(e)))
        
        # Create and start threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=convert_values, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Check no errors occurred
        self.assertEqual(errors.qsize(), 0)
        
        # Check all results are valid
        while not results.empty():
            thread_id, original, reconstructed = results.get()
            rel_error = abs(reconstructed - original) / (abs(original) + 1e-10)
            self.assertLess(rel_error, 1e-4)
    
    def test_modular_inverse_caching(self):
        """Test that modular inverse caching works correctly"""
        # Clear cache
        self.math_ops._inverse_cache.clear()
        self.assertEqual(len(self.math_ops._inverse_cache), 0)
        
        # Compute inverse (should cache it)
        inv1 = self.math_ops._mod_inverse(5)
        self.assertEqual(len(self.math_ops._inverse_cache), 1)
        
        # Compute same inverse (should use cache)
        inv2 = self.math_ops._mod_inverse(5)
        self.assertEqual(inv1, inv2)
        self.assertEqual(len(self.math_ops._inverse_cache), 1)
        
        # Compute different inverse
        inv3 = self.math_ops._mod_inverse(7)
        self.assertEqual(len(self.math_ops._inverse_cache), 2)
    
    def test_valuation_computation(self):
        """Test p-adic valuation computation"""
        # Test public compute_valuation method
        val = self.math_ops.compute_valuation(
            self.default_prime * self.default_prime, 1
        )
        self.assertEqual(val, 2)
        
        val = self.math_ops.compute_valuation(1, self.default_prime)
        self.assertEqual(val, -1)
        
        # Test with invalid inputs
        with self.assertRaises(TypeError):
            self.math_ops.compute_valuation("not int", 1)
        
        with self.assertRaises(ValueError):
            self.math_ops.compute_valuation(1, 0)
    
    # ============= Helper Function Tests =============
    
    def test_validate_single_weight(self):
        """Test single weight validation function"""
        weight = self.math_ops.to_padic(1.5)
        
        # Valid weight
        self.assertTrue(validate_single_weight(
            weight, self.default_prime, self.default_precision
        ))
        
        # Wrong prime
        self.assertFalse(validate_single_weight(
            weight, 127, self.default_precision
        ))
        
        # Wrong precision
        self.assertFalse(validate_single_weight(
            weight, self.default_prime, 5
        ))
    
    def test_validate_padic_weights(self):
        """Test batch weight validation function"""
        weights = [
            self.math_ops.to_padic(1.0),
            self.math_ops.to_padic(2.0),
            self.math_ops.to_padic(3.0)
        ]
        
        # All valid
        self.assertTrue(validate_padic_weights(
            weights, self.default_prime, self.default_precision
        ))
        
        # Create an invalid weight (with wrong precision for validation)
        # This weight is valid on its own but invalid for our validation parameters
        other_ops = PadicMathematicalOperations(self.default_prime, 3)  # Different precision
        invalid_weight = other_ops.to_padic(1.0)
        weights.append(invalid_weight)
        
        # Should fail because precision doesn't match
        self.assertFalse(validate_padic_weights(
            weights, self.default_prime, self.default_precision
        ))
    
    def test_create_real_padic_weights(self):
        """Test creation of real p-adic weights for testing"""
        num_weights = 10
        weights = create_real_padic_weights(
            num_weights, 
            precision=self.default_precision,
            prime=self.default_prime
        )
        
        self.assertEqual(len(weights), num_weights)
        
        for weight in weights:
            self.assertIsInstance(weight, PadicWeight)
            self.assertEqual(weight.prime, self.default_prime)
            self.assertEqual(weight.precision, self.default_precision)
            
            # Validate weight
            self.assertTrue(validate_single_weight(
                weight, self.default_prime, self.default_precision
            ))
    
    def test_padic_weight_post_init_validation(self):
        """Test PadicWeight __post_init__ validation"""
        # Valid weight
        weight = PadicWeight(
            value=Fraction(1, 2),
            prime=257,
            precision=4,
            valuation=0,
            digits=[128, 0, 0, 0]
        )
        self.assertIsInstance(weight, PadicWeight)
        
        # Invalid prime
        with self.assertRaises(ValueError):
            PadicWeight(
                value=Fraction(1),
                prime=1,
                precision=4,
                valuation=0,
                digits=[0, 0, 0, 0]
            )
        
        # Invalid precision
        with self.assertRaises(ValueError):
            PadicWeight(
                value=Fraction(1),
                prime=257,
                precision=0,
                valuation=0,
                digits=[]
            )
        
        # Invalid digit
        with self.assertRaises(ValueError):
            PadicWeight(
                value=Fraction(1),
                prime=257,
                precision=4,
                valuation=0,
                digits=[257, 0, 0, 0]  # Digit >= prime
            )
        
        # Wrong number of digits
        with self.assertRaises(ValueError):
            PadicWeight(
                value=Fraction(1),
                prime=257,
                precision=4,
                valuation=0,
                digits=[0, 0, 0]  # Only 3 digits, needs 4
            )


class TestPadicValidation(unittest.TestCase):
    """Test suite for PadicValidation static methods"""
    
    def test_validate_prime(self):
        """Test prime validation"""
        # Valid primes
        valid_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 127, 257]
        for prime in valid_primes:
            PadicValidation.validate_prime(prime)  # Should not raise
        
        # Invalid primes
        invalid_primes = [0, 1, 4, 6, 8, 9, 10, 15, 20, 100]
        for invalid in invalid_primes:
            with self.assertRaises(ValueError):
                PadicValidation.validate_prime(invalid)
        
        # Invalid types
        with self.assertRaises(TypeError):
            PadicValidation.validate_prime(2.5)
        with self.assertRaises(TypeError):
            PadicValidation.validate_prime("prime")
    
    def test_validate_precision(self):
        """Test precision validation"""
        # Valid precisions
        PadicValidation.validate_precision(1)
        PadicValidation.validate_precision(10)
        PadicValidation.validate_precision(100)
        
        # Numpy integers
        PadicValidation.validate_precision(np.int32(5))
        PadicValidation.validate_precision(np.int64(10))
        
        # Invalid precisions
        with self.assertRaises(ValueError):
            PadicValidation.validate_precision(0)
        with self.assertRaises(ValueError):
            PadicValidation.validate_precision(-1)
        with self.assertRaises(ValueError):
            PadicValidation.validate_precision(1001)
        
        # Invalid types
        with self.assertRaises(TypeError):
            PadicValidation.validate_precision(1.5)
        with self.assertRaises(TypeError):
            PadicValidation.validate_precision("5")
    
    def test_validate_tensor(self):
        """Test tensor validation"""
        # Valid tensors
        valid_tensor = torch.tensor([1.0, 2.0, 3.0])
        PadicValidation.validate_tensor(valid_tensor)
        
        # Different dtypes
        PadicValidation.validate_tensor(torch.tensor([1.0], dtype=torch.float32))
        PadicValidation.validate_tensor(torch.tensor([1.0], dtype=torch.float64))
        PadicValidation.validate_tensor(torch.tensor([1.0], dtype=torch.float16))
        
        # Invalid: None
        with self.assertRaises(ValueError):
            PadicValidation.validate_tensor(None)
        
        # Invalid: Not a tensor
        with self.assertRaises(TypeError):
            PadicValidation.validate_tensor([1, 2, 3])
        
        # Invalid: Empty tensor
        with self.assertRaises(ValueError):
            PadicValidation.validate_tensor(torch.tensor([]))
        
        # Invalid: Contains NaN
        with self.assertRaises(ValueError):
            PadicValidation.validate_tensor(torch.tensor([1.0, float('nan'), 3.0]))
        
        # Invalid: Contains inf
        with self.assertRaises(ValueError):
            PadicValidation.validate_tensor(torch.tensor([1.0, float('inf'), 3.0]))
        
        # Invalid: Wrong dtype
        with self.assertRaises(TypeError):
            PadicValidation.validate_tensor(torch.tensor([1, 2, 3], dtype=torch.int32))
    
    def test_validate_chunk_size(self):
        """Test chunk size validation"""
        # Valid chunk sizes
        PadicValidation.validate_chunk_size(10, 100)
        PadicValidation.validate_chunk_size(1, 1)
        PadicValidation.validate_chunk_size(50, 50)
        
        # Invalid: Not integer
        with self.assertRaises(TypeError):
            PadicValidation.validate_chunk_size(10.5, 100)
        
        # Invalid: Zero or negative
        with self.assertRaises(ValueError):
            PadicValidation.validate_chunk_size(0, 100)
        with self.assertRaises(ValueError):
            PadicValidation.validate_chunk_size(-1, 100)
        
        # Invalid: Exceeds tensor size
        with self.assertRaises(ValueError):
            PadicValidation.validate_chunk_size(101, 100)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)