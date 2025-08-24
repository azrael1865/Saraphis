"""
Comprehensive Unit Tests for PadicWeight Class
Tests dataclass validation, initialization, and all edge cases
"""

import sys
import os
import unittest
from fractions import Fraction
from typing import List, Any

# Add path to compression systems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'compression_systems', 'padic'))

from padic_encoder import PadicWeight


class TestPadicWeight(unittest.TestCase):
    """Test suite for PadicWeight dataclass"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Valid default values for creating PadicWeight instances
        self.valid_value = Fraction(3, 4)
        self.valid_prime = 7
        self.valid_precision = 4
        self.valid_valuation = 0
        self.valid_digits = [3, 2, 1, 0]  # 4 digits for precision=4, all < prime=7
    
    # ==================== Test Valid Initialization ====================
    
    def test_valid_initialization_basic(self):
        """Test creating a valid PadicWeight instance"""
        weight = PadicWeight(
            value=self.valid_value,
            prime=self.valid_prime,
            precision=self.valid_precision,
            valuation=self.valid_valuation,
            digits=self.valid_digits
        )
        
        self.assertEqual(weight.value, Fraction(3, 4))
        self.assertEqual(weight.prime, 7)
        self.assertEqual(weight.precision, 4)
        self.assertEqual(weight.valuation, 0)
        self.assertEqual(weight.digits, [3, 2, 1, 0])
    
    def test_valid_initialization_various_primes(self):
        """Test initialization with various valid primes"""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        
        for prime in primes:
            precision = 3
            digits = [0, 1, min(2, prime-1)]  # Ensure digits are valid for the prime
            
            weight = PadicWeight(
                value=Fraction(1, 2),
                prime=prime,
                precision=precision,
                valuation=0,
                digits=digits
            )
            
            self.assertEqual(weight.prime, prime)
            self.assertEqual(weight.precision, precision)
    
    def test_valid_initialization_various_precisions(self):
        """Test initialization with various valid precisions"""
        for precision in [1, 2, 3, 4, 5, 10, 20, 50, 100]:
            digits = [0] * precision  # All zeros is valid
            
            weight = PadicWeight(
                value=Fraction(0, 1),
                prime=5,
                precision=precision,
                valuation=0,
                digits=digits
            )
            
            self.assertEqual(weight.precision, precision)
            self.assertEqual(len(weight.digits), precision)
    
    def test_valid_initialization_various_valuations(self):
        """Test initialization with various valid valuations"""
        precision = 5
        valuations = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 10, 100]
        
        for valuation in valuations:
            weight = PadicWeight(
                value=Fraction(1, 3),
                prime=7,
                precision=precision,
                valuation=valuation,
                digits=[0] * precision
            )
            
            self.assertEqual(weight.valuation, valuation)
    
    def test_valid_initialization_edge_digits(self):
        """Test initialization with edge case digit values"""
        prime = 11
        precision = 4
        
        # All zeros
        weight1 = PadicWeight(
            value=Fraction(0, 1),
            prime=prime,
            precision=precision,
            valuation=0,
            digits=[0, 0, 0, 0]
        )
        self.assertEqual(weight1.digits, [0, 0, 0, 0])
        
        # All maximum (prime-1)
        weight2 = PadicWeight(
            value=Fraction(-1, 1),
            prime=prime,
            precision=precision,
            valuation=0,
            digits=[10, 10, 10, 10]
        )
        self.assertEqual(weight2.digits, [10, 10, 10, 10])
        
        # Mixed values
        weight3 = PadicWeight(
            value=Fraction(123, 456),
            prime=prime,
            precision=precision,
            valuation=0,
            digits=[0, 5, 10, 3]
        )
        self.assertEqual(weight3.digits, [0, 5, 10, 3])
    
    # ==================== Test Invalid Prime ====================
    
    def test_invalid_prime_not_int(self):
        """Test that non-integer primes are rejected"""
        invalid_primes = [2.5, "7", [11], {13: 17}, None, complex(19, 0)]
        
        for invalid_prime in invalid_primes:
            with self.assertRaises(TypeError) as context:
                PadicWeight(
                    value=self.valid_value,
                    prime=invalid_prime,
                    precision=self.valid_precision,
                    valuation=self.valid_valuation,
                    digits=self.valid_digits
                )
            self.assertIn("Prime must be int", str(context.exception))
    
    def test_invalid_prime_less_than_2(self):
        """Test that primes less than 2 are rejected"""
        invalid_primes = [-100, -10, -1, 0, 1]
        
        for invalid_prime in invalid_primes:
            with self.assertRaises(ValueError) as context:
                PadicWeight(
                    value=self.valid_value,
                    prime=invalid_prime,
                    precision=self.valid_precision,
                    valuation=self.valid_valuation,
                    digits=self.valid_digits
                )
            self.assertIn("Prime must be >= 2", str(context.exception))
    
    # ==================== Test Invalid Precision ====================
    
    def test_invalid_precision_not_int(self):
        """Test that non-integer precisions are rejected"""
        invalid_precisions = [4.5, "4", [4], {4: 4}, None, complex(4, 0)]
        
        for invalid_precision in invalid_precisions:
            with self.assertRaises(TypeError) as context:
                PadicWeight(
                    value=self.valid_value,
                    prime=self.valid_prime,
                    precision=invalid_precision,
                    valuation=self.valid_valuation,
                    digits=self.valid_digits
                )
            self.assertIn("Precision must be int", str(context.exception))
    
    def test_invalid_precision_less_than_1(self):
        """Test that precisions less than 1 are rejected"""
        invalid_precisions = [-100, -10, -1, 0]
        
        for invalid_precision in invalid_precisions:
            # Need to adjust digits to match
            with self.assertRaises(ValueError) as context:
                PadicWeight(
                    value=self.valid_value,
                    prime=self.valid_prime,
                    precision=invalid_precision,
                    valuation=self.valid_valuation,
                    digits=[]  # Empty for precision 0
                )
            self.assertIn("Precision must be >= 1", str(context.exception))
    
    # ==================== Test Invalid Valuation ====================
    
    def test_invalid_valuation_not_int(self):
        """Test that non-integer valuations are rejected"""
        invalid_valuations = [0.5, "0", [0], {0: 0}, None, complex(0, 0)]
        
        for invalid_valuation in invalid_valuations:
            with self.assertRaises(TypeError) as context:
                PadicWeight(
                    value=self.valid_value,
                    prime=self.valid_prime,
                    precision=self.valid_precision,
                    valuation=invalid_valuation,
                    digits=self.valid_digits
                )
            self.assertIn("Valuation must be int", str(context.exception))
    
    def test_invalid_valuation_exceeds_precision(self):
        """Test that valuations exceeding -precision are rejected"""
        precision = 5
        invalid_valuations = [-6, -7, -10, -100]
        
        for invalid_valuation in invalid_valuations:
            with self.assertRaises(ValueError) as context:
                PadicWeight(
                    value=self.valid_value,
                    prime=self.valid_prime,
                    precision=precision,
                    valuation=invalid_valuation,
                    digits=[0] * precision
                )
            self.assertIn(f"Valuation {invalid_valuation} exceeds precision {precision}", 
                         str(context.exception))
    
    # ==================== Test Invalid Digits ====================
    
    def test_invalid_digits_not_list(self):
        """Test that non-list digits are rejected"""
        invalid_digits = [
            (3, 2, 1, 0),  # tuple
            {0: 3, 1: 2, 2: 1, 3: 0},  # dict
            "3210",  # string
            3210,  # int
            None
        ]
        
        for invalid in invalid_digits:
            with self.assertRaises(TypeError) as context:
                PadicWeight(
                    value=self.valid_value,
                    prime=self.valid_prime,
                    precision=self.valid_precision,
                    valuation=self.valid_valuation,
                    digits=invalid
                )
            self.assertIn("Digits must be list", str(context.exception))
    
    def test_invalid_digits_wrong_length(self):
        """Test that digits with wrong length are rejected"""
        precision = 4
        wrong_length_digits = [
            [],  # Too short
            [1],  # Too short
            [1, 2],  # Too short
            [1, 2, 3],  # Too short
            [1, 2, 3, 4, 5],  # Too long
            [1, 2, 3, 4, 5, 6],  # Too long
        ]
        
        for digits in wrong_length_digits:
            with self.assertRaises(ValueError) as context:
                PadicWeight(
                    value=self.valid_value,
                    prime=self.valid_prime,
                    precision=precision,
                    valuation=self.valid_valuation,
                    digits=digits
                )
            self.assertIn(f"Digits length {len(digits)} must equal precision {precision}", 
                         str(context.exception))
    
    def test_invalid_digits_non_int_elements(self):
        """Test that non-integer digit elements are rejected"""
        invalid_digit_lists = [
            [1.5, 2, 3, 4],  # float
            ["1", 2, 3, 4],  # string
            [1, None, 3, 4],  # None
            [1, [2], 3, 4],  # nested list
            [1, 2, {3}, 4],  # set
        ]
        
        for digits in invalid_digit_lists:
            with self.assertRaises(TypeError) as context:
                PadicWeight(
                    value=self.valid_value,
                    prime=self.valid_prime,
                    precision=self.valid_precision,
                    valuation=self.valid_valuation,
                    digits=digits
                )
            self.assertIn("must be int", str(context.exception))
    
    def test_invalid_digits_out_of_range(self):
        """Test that digits outside [0, prime) are rejected"""
        prime = 5
        precision = 4
        
        # Negative digits
        invalid_digits_lists = [
            [-1, 0, 1, 2],
            [0, -1, 1, 2],
            [0, 1, -1, 2],
            [0, 1, 2, -1],
            [-5, -4, -3, -2],
        ]
        
        for digits in invalid_digits_lists:
            with self.assertRaises(ValueError) as context:
                PadicWeight(
                    value=self.valid_value,
                    prime=prime,
                    precision=precision,
                    valuation=self.valid_valuation,
                    digits=digits
                )
            self.assertIn("must be in range [0,", str(context.exception))
        
        # Digits >= prime
        invalid_digits_lists = [
            [5, 0, 1, 2],  # 5 >= prime(5)
            [0, 5, 1, 2],
            [0, 1, 5, 2],
            [0, 1, 2, 5],
            [6, 7, 8, 9],  # All >= prime(5)
        ]
        
        for digits in invalid_digits_lists:
            with self.assertRaises(ValueError) as context:
                PadicWeight(
                    value=self.valid_value,
                    prime=prime,
                    precision=precision,
                    valuation=self.valid_valuation,
                    digits=digits
                )
            self.assertIn("must be in range [0,", str(context.exception))
    
    # ==================== Test Invalid Value ====================
    
    def test_invalid_value_not_fraction(self):
        """Test that non-Fraction values are rejected"""
        invalid_values = [
            0.75,  # float
            3,  # int
            "3/4",  # string
            [3, 4],  # list
            {3: 4},  # dict
            None,
            complex(3, 4)
        ]
        
        for invalid_value in invalid_values:
            with self.assertRaises(TypeError) as context:
                PadicWeight(
                    value=invalid_value,
                    prime=self.valid_prime,
                    precision=self.valid_precision,
                    valuation=self.valid_valuation,
                    digits=self.valid_digits
                )
            self.assertIn("Value must be Fraction", str(context.exception))
    
    # ==================== Test Edge Cases ====================
    
    def test_edge_case_minimum_prime(self):
        """Test with minimum valid prime (2)"""
        weight = PadicWeight(
            value=Fraction(1, 2),
            prime=2,
            precision=8,
            valuation=0,
            digits=[0, 1, 0, 1, 0, 1, 0, 1]  # Binary digits
        )
        self.assertEqual(weight.prime, 2)
        for digit in weight.digits:
            self.assertIn(digit, [0, 1])
    
    def test_edge_case_minimum_precision(self):
        """Test with minimum valid precision (1)"""
        weight = PadicWeight(
            value=Fraction(1, 3),
            prime=7,
            precision=1,
            valuation=0,
            digits=[3]  # Single digit
        )
        self.assertEqual(weight.precision, 1)
        self.assertEqual(len(weight.digits), 1)
    
    def test_edge_case_zero_fraction(self):
        """Test with zero as the value"""
        weight = PadicWeight(
            value=Fraction(0, 1),
            prime=5,
            precision=3,
            valuation=0,
            digits=[0, 0, 0]
        )
        self.assertEqual(weight.value, Fraction(0, 1))
        self.assertEqual(weight.digits, [0, 0, 0])
    
    def test_edge_case_negative_fraction(self):
        """Test with negative fraction value"""
        weight = PadicWeight(
            value=Fraction(-3, 4),
            prime=7,
            precision=4,
            valuation=0,
            digits=[4, 5, 6, 6]  # p-adic representation of negative
        )
        self.assertEqual(weight.value, Fraction(-3, 4))
    
    def test_edge_case_large_valuation(self):
        """Test with large positive valuation"""
        weight = PadicWeight(
            value=Fraction(1, 1000),
            prime=5,
            precision=3,
            valuation=1000,  # Large positive valuation
            digits=[1, 2, 3]
        )
        self.assertEqual(weight.valuation, 1000)
    
    def test_edge_case_boundary_valuation(self):
        """Test with valuation at the boundary (-precision)"""
        precision = 10
        weight = PadicWeight(
            value=Fraction(1, 7),
            prime=3,
            precision=precision,
            valuation=-precision,  # Exactly at boundary
            digits=[0] * precision
        )
        self.assertEqual(weight.valuation, -precision)
    
    # ==================== Test Boolean Edge Cases ====================
    
    def test_boolean_as_prime(self):
        """Test that boolean values as prime are handled correctly"""
        # True = 1, False = 0, both invalid as primes (< 2)
        with self.assertRaises(ValueError) as context:
            PadicWeight(
                value=self.valid_value,
                prime=True,  # True = 1
                precision=self.valid_precision,
                valuation=self.valid_valuation,
                digits=self.valid_digits
            )
        self.assertIn("Prime must be >= 2", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            PadicWeight(
                value=self.valid_value,
                prime=False,  # False = 0
                precision=self.valid_precision,
                valuation=self.valid_valuation,
                digits=self.valid_digits
            )
        self.assertIn("Prime must be >= 2", str(context.exception))
    
    def test_boolean_as_precision(self):
        """Test that boolean values as precision are handled correctly"""
        # True = 1 (valid), False = 0 (invalid)
        weight = PadicWeight(
            value=self.valid_value,
            prime=self.valid_prime,
            precision=True,  # True = 1
            valuation=self.valid_valuation,
            digits=[3]  # Single digit for precision=1
        )
        self.assertEqual(weight.precision, 1)
        
        with self.assertRaises(ValueError) as context:
            PadicWeight(
                value=self.valid_value,
                prime=self.valid_prime,
                precision=False,  # False = 0
                valuation=self.valid_valuation,
                digits=[]
            )
        self.assertIn("Precision must be >= 1", str(context.exception))
    
    def test_boolean_as_valuation(self):
        """Test that boolean values as valuation are handled correctly"""
        # Both True=1 and False=0 are valid valuations
        weight1 = PadicWeight(
            value=self.valid_value,
            prime=self.valid_prime,
            precision=self.valid_precision,
            valuation=True,  # True = 1
            digits=self.valid_digits
        )
        self.assertEqual(weight1.valuation, 1)
        
        weight2 = PadicWeight(
            value=self.valid_value,
            prime=self.valid_prime,
            precision=self.valid_precision,
            valuation=False,  # False = 0
            digits=self.valid_digits
        )
        self.assertEqual(weight2.valuation, 0)
    
    def test_boolean_in_digits(self):
        """Test that boolean values in digits are handled correctly"""
        # True=1, False=0, both valid as digits if < prime
        weight = PadicWeight(
            value=self.valid_value,
            prime=7,
            precision=4,
            valuation=self.valid_valuation,
            digits=[True, False, True, False]  # [1, 0, 1, 0]
        )
        self.assertEqual(weight.digits, [1, 0, 1, 0])


class TestPadicWeightConsistency(unittest.TestCase):
    """Test consistency and relationships between PadicWeight fields"""
    
    def test_digits_prime_consistency(self):
        """Test that all digits are always less than prime"""
        test_cases = [
            (2, [0, 1, 0, 1]),
            (3, [0, 1, 2, 0]),
            (5, [0, 1, 2, 3, 4]),
            (7, [6, 5, 4, 3, 2, 1, 0]),
        ]
        
        for prime, digits in test_cases:
            weight = PadicWeight(
                value=Fraction(1, 2),
                prime=prime,
                precision=len(digits),
                valuation=0,
                digits=digits
            )
            
            for digit in weight.digits:
                self.assertLess(digit, prime)
                self.assertGreaterEqual(digit, 0)
    
    def test_precision_digits_consistency(self):
        """Test that digits length always equals precision"""
        for precision in range(1, 20):
            digits = list(range(min(precision, 7)))  # Use valid digits
            # Pad or truncate to match precision
            if len(digits) < precision:
                digits += [0] * (precision - len(digits))
            else:
                digits = digits[:precision]
            
            weight = PadicWeight(
                value=Fraction(1, 3),
                prime=11,  # Large enough for all test digits
                precision=precision,
                valuation=0,
                digits=digits
            )
            
            self.assertEqual(len(weight.digits), precision)
            self.assertEqual(weight.precision, precision)
    
    def test_valuation_bounds(self):
        """Test that valuation respects bounds relative to precision"""
        precision = 10
        
        # Test various valid valuations
        for valuation in range(-precision, 100):
            weight = PadicWeight(
                value=Fraction(1, 7),
                prime=5,
                precision=precision,
                valuation=valuation,
                digits=[0] * precision
            )
            
            self.assertGreaterEqual(weight.valuation, -precision)
    
    def test_immutability_after_creation(self):
        """Test that PadicWeight fields maintain their values after creation"""
        original_value = Fraction(3, 4)
        original_prime = 7
        original_precision = 4
        original_valuation = 2
        original_digits = [1, 2, 3, 4]
        
        weight = PadicWeight(
            value=original_value,
            prime=original_prime,
            precision=original_precision,
            valuation=original_valuation,
            digits=original_digits.copy()
        )
        
        # Verify all fields match originals
        self.assertEqual(weight.value, original_value)
        self.assertEqual(weight.prime, original_prime)
        self.assertEqual(weight.precision, original_precision)
        self.assertEqual(weight.valuation, original_valuation)
        self.assertEqual(weight.digits, original_digits)
        
        # Modifying the original list shouldn't affect the weight
        original_digits[0] = 999
        self.assertEqual(weight.digits[0], 1)  # Should still be 1


class TestPadicWeightSpecialCases(unittest.TestCase):
    """Test special mathematical cases for PadicWeight"""
    
    def test_unit_fractions(self):
        """Test various unit fractions as values"""
        unit_fractions = [
            Fraction(1, 1),
            Fraction(1, 2),
            Fraction(1, 3),
            Fraction(1, 4),
            Fraction(1, 5),
            Fraction(1, 10),
            Fraction(1, 100),
        ]
        
        for frac in unit_fractions:
            weight = PadicWeight(
                value=frac,
                prime=7,
                precision=5,
                valuation=0,
                digits=[0, 1, 2, 3, 4]
            )
            self.assertEqual(weight.value, frac)
    
    def test_integer_fractions(self):
        """Test integer values as fractions"""
        for i in range(-10, 11):
            weight = PadicWeight(
                value=Fraction(i, 1),
                prime=11,
                precision=3,
                valuation=0,
                digits=[0, 1, 2]
            )
            self.assertEqual(weight.value.numerator, i)
            self.assertEqual(weight.value.denominator, 1)
    
    def test_large_prime_values(self):
        """Test with larger prime values"""
        large_primes = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
                       151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
                       199, 211, 223, 227, 229, 233, 239, 241, 251, 257]
        
        for prime in large_primes:
            # Create valid digits for this prime
            precision = 2
            digits = [0, min(10, prime-1)]  # Ensure valid digits
            
            weight = PadicWeight(
                value=Fraction(1, 2),
                prime=prime,
                precision=precision,
                valuation=0,
                digits=digits
            )
            self.assertEqual(weight.prime, prime)
    
    def test_repeating_decimal_fractions(self):
        """Test fractions that produce repeating decimals"""
        repeating_fractions = [
            Fraction(1, 3),   # 0.333...
            Fraction(1, 6),   # 0.1666...
            Fraction(1, 7),   # 0.142857...
            Fraction(2, 3),   # 0.666...
            Fraction(5, 6),   # 0.8333...
            Fraction(1, 9),   # 0.111...
            Fraction(1, 11),  # 0.0909...
        ]
        
        for frac in repeating_fractions:
            weight = PadicWeight(
                value=frac,
                prime=5,
                precision=6,
                valuation=0,
                digits=[0, 1, 2, 3, 4, 0]
            )
            self.assertEqual(weight.value, frac)
    
    def test_maximum_digit_values(self):
        """Test with maximum possible digit values for each prime"""
        test_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        for prime in test_primes:
            precision = 3
            max_digit = prime - 1
            digits = [max_digit] * precision  # All digits at maximum
            
            weight = PadicWeight(
                value=Fraction(-1, 1),  # Often represented with max digits
                prime=prime,
                precision=precision,
                valuation=0,
                digits=digits
            )
            
            for digit in weight.digits:
                self.assertEqual(digit, max_digit)


if __name__ == "__main__":
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPadicWeight))
    suite.addTests(loader.loadTestsFromTestCase(TestPadicWeightConsistency))
    suite.addTests(loader.loadTestsFromTestCase(TestPadicWeightSpecialCases))
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("PADICWEIGHT TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)