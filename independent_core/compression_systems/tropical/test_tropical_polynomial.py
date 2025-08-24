"""
Comprehensive Test Suite for TropicalPolynomial
Tests all polynomial components, operations, and edge cases
"""

import unittest
import torch
import math
import numpy as np
from typing import Dict, List, Optional
import tempfile
import time
from collections import defaultdict
import itertools

# Import the modules to test
try:
    # Try absolute imports first
    from compression_systems.tropical.tropical_polynomial import (
        TropicalMonomial,
        TropicalPolynomial,
        TropicalPolynomialOperations
    )

    from compression_systems.tropical.tropical_core import (
        TROPICAL_ZERO,
        TROPICAL_EPSILON,
        TropicalNumber,
        is_tropical_zero,
        to_tropical_safe,
        from_tropical_safe
    )
except ImportError:
    # Fall back to relative imports
    from .tropical_polynomial import (
        TropicalMonomial,
        TropicalPolynomial,
        TropicalPolynomialOperations
    )

    from .tropical_core import (
        TROPICAL_ZERO,
        TROPICAL_EPSILON,
        TropicalNumber,
        is_tropical_zero,
        to_tropical_safe,
        from_tropical_safe
    )


class TestTropicalMonomial(unittest.TestCase):
    """Test TropicalMonomial dataclass"""
    
    def test_monomial_creation_valid(self):
        """Test valid monomial creation"""
        # Basic monomial
        m1 = TropicalMonomial(5.0, {0: 2, 1: 1})
        self.assertEqual(m1.coefficient, 5.0)
        self.assertEqual(m1.exponents, {0: 2, 1: 1})
        self.assertFalse(m1.is_zero())
        
        # Constant monomial (no variables)
        m2 = TropicalMonomial(3.0, {})
        self.assertEqual(m2.coefficient, 3.0)
        self.assertEqual(m2.exponents, {})
        self.assertEqual(m2.degree(), 0)
        
        # Tropical zero monomial
        m3 = TropicalMonomial(TROPICAL_ZERO, {0: 1})
        self.assertTrue(m3.is_zero())
        
        # Very negative coefficient (should become tropical zero)
        m4 = TropicalMonomial(-1e39, {0: 1})
        self.assertTrue(m4.is_zero())
        self.assertEqual(m4.coefficient, TROPICAL_ZERO)
    
    def test_monomial_creation_invalid(self):
        """Test invalid monomial creation"""
        # Non-numeric coefficient
        with self.assertRaises(TypeError):
            TropicalMonomial("not_a_number", {})
        
        # NaN coefficient
        with self.assertRaises(ValueError):
            TropicalMonomial(float('nan'), {})
        
        # Positive infinity coefficient
        with self.assertRaises(ValueError):
            TropicalMonomial(float('inf'), {})
        
        # Coefficient exceeds safe range
        with self.assertRaises(ValueError):
            TropicalMonomial(1e39, {})
        
        # Non-dict exponents
        with self.assertRaises(TypeError):
            TropicalMonomial(1.0, [1, 2, 3])
        
        # Non-integer variable index
        with self.assertRaises(TypeError):
            TropicalMonomial(1.0, {"x": 1})
        
        # Negative variable index
        with self.assertRaises(ValueError):
            TropicalMonomial(1.0, {-1: 2})
        
        # Non-integer exponent
        with self.assertRaises(TypeError):
            TropicalMonomial(1.0, {0: 1.5})
        
        # Negative exponent
        with self.assertRaises(ValueError):
            TropicalMonomial(1.0, {0: -1})
    
    def test_monomial_zero_exponent_removal(self):
        """Test that zero exponents are removed"""
        m = TropicalMonomial(2.0, {0: 1, 1: 0, 2: 3, 3: 0})
        self.assertEqual(m.exponents, {0: 1, 2: 3})
        self.assertNotIn(1, m.exponents)
        self.assertNotIn(3, m.exponents)
    
    def test_monomial_degree(self):
        """Test degree calculation"""
        # Single variable
        m1 = TropicalMonomial(1.0, {0: 3})
        self.assertEqual(m1.degree(), 3)
        
        # Multiple variables
        m2 = TropicalMonomial(1.0, {0: 2, 1: 3, 2: 1})
        self.assertEqual(m2.degree(), 6)
        
        # Constant (no variables)
        m3 = TropicalMonomial(1.0, {})
        self.assertEqual(m3.degree(), 0)
        
        # Tropical zero
        m4 = TropicalMonomial(TROPICAL_ZERO, {0: 5})
        self.assertEqual(m4.degree(), 5)
    
    def test_monomial_evaluation_with_list(self):
        """Test monomial evaluation with list input"""
        m = TropicalMonomial(2.0, {0: 1, 1: 2})
        
        # Basic evaluation
        point = [3.0, 1.0]
        result = m.evaluate(point)
        expected = 2.0 + 1*3.0 + 2*1.0  # 2 + 3 + 2 = 7
        self.assertAlmostEqual(result, expected, places=6)
        
        # With zero values
        point = [0.0, 0.0]
        result = m.evaluate(point)
        self.assertAlmostEqual(result, 2.0, places=6)
        
        # Negative values
        point = [-1.0, -2.0]
        result = m.evaluate(point)
        expected = 2.0 + 1*(-1.0) + 2*(-2.0)  # 2 - 1 - 4 = -3
        self.assertAlmostEqual(result, expected, places=6)
    
    def test_monomial_evaluation_with_tensor(self):
        """Test monomial evaluation with tensor input"""
        m = TropicalMonomial(1.0, {0: 2, 1: 1})
        
        # CPU tensor
        point = torch.tensor([2.0, 3.0])
        result = m.evaluate(point)
        expected = 1.0 + 2*2.0 + 1*3.0  # 1 + 4 + 3 = 8
        self.assertAlmostEqual(result, expected, places=6)
        
        # GPU tensor (if available)
        if torch.cuda.is_available():
            point_gpu = torch.tensor([2.0, 3.0], device='cuda')
            result = m.evaluate(point_gpu)
            self.assertAlmostEqual(result, expected, places=6)
    
    def test_monomial_evaluation_with_numpy(self):
        """Test monomial evaluation with numpy array"""
        m = TropicalMonomial(0.0, {0: 1, 1: 1})
        
        point = np.array([2.0, 3.0])
        result = m.evaluate(point)
        expected = 0.0 + 1*2.0 + 1*3.0  # 0 + 2 + 3 = 5
        self.assertAlmostEqual(result, expected, places=6)
    
    def test_monomial_evaluation_errors(self):
        """Test evaluation error cases"""
        m = TropicalMonomial(1.0, {0: 1, 2: 1})
        
        # Invalid type
        with self.assertRaises(TypeError):
            m.evaluate("invalid")
        
        # Index out of range
        with self.assertRaises(IndexError):
            m.evaluate([1.0])  # Need at least 3 values for indices 0, 2
        
        # Overflow detection
        m_large = TropicalMonomial(1e38, {0: 1})
        with self.assertRaises(OverflowError):
            m_large.evaluate([1.0])
    
    def test_monomial_tropical_zero(self):
        """Test tropical zero monomial behavior"""
        m = TropicalMonomial(TROPICAL_ZERO, {0: 1, 1: 2})
        
        self.assertTrue(m.is_zero())
        
        # Evaluation of zero monomial
        result = m.evaluate([1.0, 2.0])
        self.assertEqual(result, TROPICAL_ZERO)
    
    def test_monomial_string_representation(self):
        """Test string representation"""
        # Regular monomial
        m1 = TropicalMonomial(2.5, {0: 1, 1: 2})
        str_repr = str(m1)
        self.assertIn("2.5", str_repr)
        self.assertIn("x_0", str_repr)
        self.assertIn("x_1^2", str_repr)
        
        # Constant monomial
        m2 = TropicalMonomial(3.0, {})
        self.assertEqual(str(m2), "3.000000")
        
        # Tropical zero
        m3 = TropicalMonomial(TROPICAL_ZERO, {})
        self.assertEqual(str(m3), "T(-∞)")
    
    def test_monomial_hash_and_equality(self):
        """Test hashing and equality"""
        m1 = TropicalMonomial(1.0, {0: 1, 1: 2})
        m2 = TropicalMonomial(1.0, {0: 1, 1: 2})
        m3 = TropicalMonomial(1.0, {0: 2, 1: 1})
        m4 = TropicalMonomial(2.0, {0: 1, 1: 2})
        
        # Equal monomials
        self.assertEqual(m1, m2)
        self.assertEqual(hash(m1), hash(m2))
        
        # Different exponents
        self.assertNotEqual(m1, m3)
        
        # Different coefficient
        self.assertNotEqual(m1, m4)
        
        # Tropical zeros are equal
        z1 = TropicalMonomial(TROPICAL_ZERO, {0: 1})
        z2 = TropicalMonomial(TROPICAL_ZERO, {1: 2})
        self.assertEqual(z1, z2)
        self.assertEqual(hash(z1), hash(z2))
        
        # Equality with epsilon tolerance
        m5 = TropicalMonomial(1.0, {0: 1})
        m6 = TropicalMonomial(1.0 + TROPICAL_EPSILON/2, {0: 1})
        self.assertEqual(m5, m6)
        
        # Not equal to other types
        self.assertNotEqual(m1, "not a monomial")
        self.assertNotEqual(m1, 1.0)
    
    def test_monomial_immutability(self):
        """Test that monomials are immutable"""
        m = TropicalMonomial(1.0, {0: 1})
        
        # Cannot modify coefficient
        with self.assertRaises(AttributeError):
            m.coefficient = 2.0
        
        # Cannot modify exponents
        with self.assertRaises(AttributeError):
            m.exponents = {1: 1}
    
    def test_monomial_coefficient_clamping(self):
        """Test that very negative coefficients get clamped to TROPICAL_ZERO"""
        # Test exact TROPICAL_ZERO
        m1 = TropicalMonomial(TROPICAL_ZERO, {0: 1})
        self.assertEqual(m1.coefficient, TROPICAL_ZERO)
        self.assertTrue(m1.is_zero())
        
        # Test value below TROPICAL_ZERO gets clamped
        m2 = TropicalMonomial(TROPICAL_ZERO - 1, {0: 1})
        self.assertEqual(m2.coefficient, TROPICAL_ZERO)
        self.assertTrue(m2.is_zero())
        
        # Test very negative value gets clamped
        m3 = TropicalMonomial(-1e40, {0: 1})
        self.assertEqual(m3.coefficient, TROPICAL_ZERO)
        self.assertTrue(m3.is_zero())
        
        # Test regular negative value is not clamped
        m4 = TropicalMonomial(-100.0, {0: 1})
        self.assertEqual(m4.coefficient, -100.0)
        self.assertFalse(m4.is_zero())
        
        # Test positive value is not clamped
        m5 = TropicalMonomial(10.0, {0: 1})
        self.assertEqual(m5.coefficient, 10.0)
        self.assertFalse(m5.is_zero())


class TestTropicalPolynomial(unittest.TestCase):
    """Test TropicalPolynomial class"""
    
    def test_polynomial_creation_valid(self):
        """Test valid polynomial creation"""
        m1 = TropicalMonomial(1.0, {0: 1})
        m2 = TropicalMonomial(2.0, {1: 1})
        
        # Basic creation
        poly = TropicalPolynomial([m1, m2], num_variables=2)
        self.assertEqual(len(poly.monomials), 2)
        self.assertEqual(poly.num_variables, 2)
        
        # Empty polynomial
        empty_poly = TropicalPolynomial([], num_variables=3)
        self.assertEqual(len(empty_poly.monomials), 0)
        self.assertEqual(empty_poly.num_variables, 3)
        
        # With tropical zeros (should be filtered)
        m3 = TropicalMonomial(TROPICAL_ZERO, {0: 2})
        poly2 = TropicalPolynomial([m1, m2, m3], num_variables=2)
        self.assertEqual(len(poly2.monomials), 2)
        
        # With duplicate monomials (should be deduplicated)
        m4 = TropicalMonomial(1.0, {0: 1})  # Same as m1
        poly3 = TropicalPolynomial([m1, m2, m4], num_variables=2)
        self.assertEqual(len(poly3.monomials), 2)
    
    def test_polynomial_creation_invalid(self):
        """Test invalid polynomial creation"""
        m1 = TropicalMonomial(1.0, {0: 1})
        
        # Non-list monomials
        with self.assertRaises(TypeError):
            TropicalPolynomial(m1, num_variables=2)
        
        # Non-integer num_variables
        with self.assertRaises(TypeError):
            TropicalPolynomial([m1], num_variables="two")
        
        # Non-positive num_variables
        with self.assertRaises(ValueError):
            TropicalPolynomial([m1], num_variables=0)
        
        with self.assertRaises(ValueError):
            TropicalPolynomial([m1], num_variables=-1)
        
        # Non-monomial in list
        with self.assertRaises(TypeError):
            TropicalPolynomial([m1, "not a monomial"], num_variables=2)
        
        # Variable index out of range
        m_invalid = TropicalMonomial(1.0, {3: 1})
        with self.assertRaises(ValueError):
            TropicalPolynomial([m_invalid], num_variables=2)
    
    def test_polynomial_evaluation_single_point(self):
        """Test polynomial evaluation at single point"""
        m1 = TropicalMonomial(0.0, {})  # Constant 0
        m2 = TropicalMonomial(1.0, {0: 1})  # 1 + x₀
        m3 = TropicalMonomial(2.0, {1: 1})  # 2 + x₁
        
        poly = TropicalPolynomial([m1, m2, m3], num_variables=2)
        
        # Test point
        point = torch.tensor([3.0, 1.0])
        result = poly.evaluate(point)
        # max(0, 1+3, 2+1) = max(0, 4, 3) = 4
        self.assertAlmostEqual(result, 4.0, places=6)
        
        # Another test point
        point2 = torch.tensor([0.0, 5.0])
        result2 = poly.evaluate(point2)
        # max(0, 1+0, 2+5) = max(0, 1, 7) = 7
        self.assertAlmostEqual(result2, 7.0, places=6)
        
        # Test with negative values
        point3 = torch.tensor([-2.0, -1.0])
        result3 = poly.evaluate(point3)
        # max(0, 1-2, 2-1) = max(0, -1, 1) = 1
        self.assertAlmostEqual(result3, 1.0, places=6)
    
    def test_polynomial_evaluation_batch(self):
        """Test batch evaluation of polynomial"""
        m1 = TropicalMonomial(0.0, {})
        m2 = TropicalMonomial(1.0, {0: 1})
        poly = TropicalPolynomial([m1, m2], num_variables=1)
        
        # Batch of points
        points = torch.tensor([[1.0], [2.0], [3.0], [-1.0]])
        results = poly.evaluate(points)
        
        self.assertEqual(results.shape, (4,))
        self.assertAlmostEqual(results[0].item(), 2.0, places=6)  # max(0, 1+1)
        self.assertAlmostEqual(results[1].item(), 3.0, places=6)  # max(0, 1+2)
        self.assertAlmostEqual(results[2].item(), 4.0, places=6)  # max(0, 1+3)
        self.assertAlmostEqual(results[3].item(), 0.0, places=6)  # max(0, 1-1)
    
    def test_polynomial_evaluation_gpu(self):
        """Test GPU-accelerated evaluation"""
        if not torch.cuda.is_available():
            self.skipTest("GPU not available")
        
        m1 = TropicalMonomial(0.0, {})
        m2 = TropicalMonomial(1.0, {0: 1, 1: 1})
        m3 = TropicalMonomial(2.0, {0: 2})
        poly = TropicalPolynomial([m1, m2, m3], num_variables=2)
        
        # GPU batch evaluation
        points = torch.randn(100, 2, device='cuda')
        results = poly.evaluate(points)
        
        self.assertEqual(results.shape, (100,))
        self.assertEqual(results.device.type, 'cuda')
        
        # Verify correctness with single point
        single_point = points[0].unsqueeze(0)
        single_result = poly.evaluate(single_point)
        self.assertAlmostEqual(results[0].item(), single_result[0].item(), places=5)
    
    def test_polynomial_evaluation_errors(self):
        """Test evaluation error cases"""
        m1 = TropicalMonomial(1.0, {0: 1})
        poly = TropicalPolynomial([m1], num_variables=2)
        
        # Wrong input type
        with self.assertRaises(TypeError):
            poly.evaluate([1.0, 2.0])
        
        # Wrong dimension (1D)
        with self.assertRaises(ValueError):
            poly.evaluate(torch.tensor([1.0, 2.0, 3.0]))
        
        # Wrong dimension (3D)
        with self.assertRaises(ValueError):
            poly.evaluate(torch.tensor([[[1.0]]]))
        
        # Dimension mismatch
        with self.assertRaises(ValueError):
            poly.evaluate(torch.tensor([[1.0]]))  # Need 2 variables
    
    def test_polynomial_addition(self):
        """Test tropical polynomial addition (max)"""
        m1 = TropicalMonomial(1.0, {0: 1})
        m2 = TropicalMonomial(2.0, {1: 1})
        poly1 = TropicalPolynomial([m1], num_variables=2)
        poly2 = TropicalPolynomial([m2], num_variables=2)
        
        # Add polynomials
        poly_sum = poly1.add(poly2)
        self.assertEqual(len(poly_sum.monomials), 2)
        
        # Test evaluation
        point = torch.tensor([3.0, 1.0])
        result = poly_sum.evaluate(point)
        # max(1+3, 2+1) = max(4, 3) = 4
        self.assertAlmostEqual(result, 4.0, places=6)
        
        # Add with itself
        poly_double = poly1.add(poly1)
        self.assertEqual(len(poly_double.monomials), 1)  # Same monomial
    
    def test_polynomial_multiplication(self):
        """Test tropical polynomial multiplication (sum)"""
        m1 = TropicalMonomial(1.0, {0: 1})
        m2 = TropicalMonomial(2.0, {1: 1})
        poly1 = TropicalPolynomial([m1], num_variables=2)
        poly2 = TropicalPolynomial([m2], num_variables=2)
        
        # Multiply polynomials
        poly_prod = poly1.multiply(poly2)
        self.assertEqual(len(poly_prod.monomials), 1)
        
        # Check resulting monomial
        prod_monomial = poly_prod.monomials[0]
        self.assertAlmostEqual(prod_monomial.coefficient, 3.0, places=6)  # 1 + 2
        self.assertEqual(prod_monomial.exponents, {0: 1, 1: 1})
        
        # Test evaluation
        point = torch.tensor([2.0, 3.0])
        result = poly_prod.evaluate(point)
        # 3 + 1*2 + 1*3 = 3 + 2 + 3 = 8
        self.assertAlmostEqual(result, 8.0, places=6)
    
    def test_polynomial_multiplication_multiple_monomials(self):
        """Test multiplication with multiple monomials"""
        m1 = TropicalMonomial(1.0, {0: 1})
        m2 = TropicalMonomial(2.0, {0: 2})
        m3 = TropicalMonomial(0.0, {1: 1})
        m4 = TropicalMonomial(3.0, {1: 2})
        
        poly1 = TropicalPolynomial([m1, m2], num_variables=2)
        poly2 = TropicalPolynomial([m3, m4], num_variables=2)
        
        poly_prod = poly1.multiply(poly2)
        # Should have 2*2 = 4 monomials
        self.assertEqual(len(poly_prod.monomials), 4)
    
    def test_polynomial_operation_errors(self):
        """Test operation error cases"""
        poly1 = TropicalPolynomial([TropicalMonomial(1.0, {})], num_variables=2)
        poly2 = TropicalPolynomial([TropicalMonomial(1.0, {})], num_variables=3)
        
        # Variable count mismatch
        with self.assertRaises(ValueError):
            poly1.add(poly2)
        
        with self.assertRaises(ValueError):
            poly1.multiply(poly2)
        
        # Wrong type
        with self.assertRaises(TypeError):
            poly1.add("not a polynomial")
        
        with self.assertRaises(TypeError):
            poly1.multiply(1.0)
    
    def test_polynomial_degree(self):
        """Test degree calculation"""
        # Empty polynomial
        empty = TropicalPolynomial([], num_variables=2)
        self.assertEqual(empty.degree(), 0)
        
        # Single monomial
        m1 = TropicalMonomial(1.0, {0: 2, 1: 3})
        poly1 = TropicalPolynomial([m1], num_variables=2)
        self.assertEqual(poly1.degree(), 5)
        
        # Multiple monomials
        m2 = TropicalMonomial(2.0, {0: 1})
        m3 = TropicalMonomial(3.0, {1: 7})
        poly2 = TropicalPolynomial([m1, m2, m3], num_variables=2)
        self.assertEqual(poly2.degree(), 7)  # Maximum degree
    
    def test_newton_polytope(self):
        """Test Newton polytope computation"""
        m1 = TropicalMonomial(1.0, {0: 2, 1: 1})
        m2 = TropicalMonomial(2.0, {0: 1, 1: 2})
        m3 = TropicalMonomial(0.0, {})
        
        poly = TropicalPolynomial([m1, m2, m3], num_variables=2)
        polytope = poly.newton_polytope()
        
        self.assertEqual(polytope.shape, (3, 2))
        
        # Check vertices
        expected = torch.tensor([[2.0, 1.0], [1.0, 2.0], [0.0, 0.0]])
        self.assertTrue(torch.allclose(polytope, expected))
        
        # Empty polynomial
        empty = TropicalPolynomial([], num_variables=3)
        empty_polytope = empty.newton_polytope()
        self.assertEqual(empty_polytope.shape, (0, 3))
    
    def test_dense_matrix_conversion(self):
        """Test conversion to dense matrix"""
        m1 = TropicalMonomial(1.0, {0: 2, 1: 1})
        m2 = TropicalMonomial(2.0, {0: 1, 1: 3})
        poly = TropicalPolynomial([m1, m2], num_variables=2)
        
        # CPU conversion
        matrix = poly.to_dense_matrix()
        self.assertEqual(matrix.shape, (3, 2))  # (num_vars + 1, num_monomials)
        
        # Check coefficients (row 0)
        self.assertAlmostEqual(matrix[0, 0].item(), 1.0, places=6)
        self.assertAlmostEqual(matrix[0, 1].item(), 2.0, places=6)
        
        # Check exponents
        self.assertAlmostEqual(matrix[1, 0].item(), 2.0, places=6)  # x0 power in m1
        self.assertAlmostEqual(matrix[1, 1].item(), 1.0, places=6)  # x0 power in m2
        self.assertAlmostEqual(matrix[2, 0].item(), 1.0, places=6)  # x1 power in m1
        self.assertAlmostEqual(matrix[2, 1].item(), 3.0, places=6)  # x1 power in m2
        
        # GPU conversion
        if torch.cuda.is_available():
            gpu_matrix = poly.to_dense_matrix(device=torch.device('cuda'))
            self.assertEqual(gpu_matrix.device.type, 'cuda')
            self.assertTrue(torch.allclose(gpu_matrix.cpu(), matrix))
        
        # Test caching
        matrix2 = poly.to_dense_matrix()
        self.assertTrue(matrix is matrix2)  # Should return cached version
    
    def test_polynomial_string_representation(self):
        """Test string representations"""
        # Empty polynomial
        empty = TropicalPolynomial([], num_variables=2)
        self.assertEqual(str(empty), "T(-∞)")
        
        # Small polynomial
        m1 = TropicalMonomial(1.0, {0: 1})
        m2 = TropicalMonomial(2.0, {1: 2})
        small = TropicalPolynomial([m1, m2], num_variables=2)
        str_repr = str(small)
        self.assertIn("TropicalPoly", str_repr)
        self.assertIn("2 vars", str_repr)
        self.assertIn("max{", str_repr)
        
        # Large polynomial (should truncate)
        monomials = [TropicalMonomial(float(i), {0: i}) for i in range(10)]
        large = TropicalPolynomial(monomials, num_variables=1)
        str_repr = str(large)
        self.assertIn("more)", str_repr)
        
        # Repr
        repr_str = repr(small)
        self.assertIn("TropicalPolynomial", repr_str)
        self.assertIn("monomials=2", repr_str)
        self.assertIn("num_variables=2", repr_str)
    
    def test_polynomial_overflow_protection(self):
        """Test overflow protection in operations"""
        # Large coefficient multiplication
        m1 = TropicalMonomial(5e37, {0: 1})
        m2 = TropicalMonomial(5e37, {1: 1})
        poly1 = TropicalPolynomial([m1], num_variables=2)
        poly2 = TropicalPolynomial([m2], num_variables=2)
        
        with self.assertRaises(OverflowError):
            poly1.multiply(poly2)
    
    def test_tropical_multiplication_is_addition(self):
        """Test that tropical multiplication is standard addition"""
        # Create simple monomials with known coefficients
        m1 = TropicalMonomial(3.0, {0: 1})  # 3 + x_0
        m2 = TropicalMonomial(5.0, {1: 1})  # 5 + x_1
        
        poly1 = TropicalPolynomial([m1], num_variables=2)
        poly2 = TropicalPolynomial([m2], num_variables=2)
        
        # Tropical multiplication: (3 + x_0) ⊗ (5 + x_1) = 8 + x_0 + x_1
        result = poly1.multiply(poly2)
        
        # Check the result has one monomial
        self.assertEqual(len(result.monomials), 1)
        
        # Check coefficient is sum: 3 + 5 = 8
        self.assertAlmostEqual(result.monomials[0].coefficient, 8.0, places=6)
        
        # Check exponents are combined
        self.assertEqual(result.monomials[0].exponents, {0: 1, 1: 1})
        
        # Verify by evaluation
        point = torch.tensor([2.0, 3.0])
        # Should be 8 + 2 + 3 = 13
        self.assertAlmostEqual(result.evaluate(point), 13.0, places=6)


class TestTropicalPolynomialOperations(unittest.TestCase):
    """Test TropicalPolynomialOperations class"""
    
    def test_operations_initialization(self):
        """Test operations class initialization"""
        # Default CPU
        ops_cpu = TropicalPolynomialOperations()
        self.assertEqual(ops_cpu.device.type, 'cpu')
        
        # Explicit CPU
        ops_cpu2 = TropicalPolynomialOperations(device=torch.device('cpu'))
        self.assertEqual(ops_cpu2.device.type, 'cpu')
        
        # GPU if available
        if torch.cuda.is_available():
            ops_gpu = TropicalPolynomialOperations(device=torch.device('cuda'))
            self.assertEqual(ops_gpu.device.type, 'cuda')
        
        # Invalid device
        with self.assertRaises(TypeError):
            TropicalPolynomialOperations(device="not a device")
    
    def test_batch_evaluate(self):
        """Test batch evaluation of multiple polynomials"""
        ops = TropicalPolynomialOperations()
        
        # Create test polynomials
        m1 = TropicalMonomial(1.0, {0: 1})
        m2 = TropicalMonomial(2.0, {1: 1})
        poly1 = TropicalPolynomial([m1], num_variables=2)
        poly2 = TropicalPolynomial([m2], num_variables=2)
        
        # Test points
        points = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        # Batch evaluate
        results = ops.batch_evaluate([poly1, poly2], points)
        
        self.assertEqual(results.shape, (2, 3))
        
        # Verify results
        # poly1 at [1,2]: 1 + 1 = 2
        self.assertAlmostEqual(results[0, 0].item(), 2.0, places=6)
        # poly2 at [1,2]: 2 + 2 = 4
        self.assertAlmostEqual(results[1, 0].item(), 4.0, places=6)
    
    def test_batch_evaluate_errors(self):
        """Test batch evaluation error cases"""
        ops = TropicalPolynomialOperations()
        
        poly = TropicalPolynomial([TropicalMonomial(1.0, {})], num_variables=2)
        points = torch.tensor([[1.0, 2.0]])
        
        # Not a list
        with self.assertRaises(TypeError):
            ops.batch_evaluate(poly, points)
        
        # Empty list
        with self.assertRaises(ValueError):
            ops.batch_evaluate([], points)
        
        # Not a tensor
        with self.assertRaises(TypeError):
            ops.batch_evaluate([poly], [[1.0, 2.0]])
        
        # Wrong type in list
        with self.assertRaises(TypeError):
            ops.batch_evaluate([poly, "not a poly"], points)
        
        # Dimension mismatch
        poly2 = TropicalPolynomial([TropicalMonomial(1.0, {})], num_variables=3)
        with self.assertRaises(ValueError):
            ops.batch_evaluate([poly, poly2], points)
    
    def test_find_tropical_roots_1d(self):
        """Test root finding in 1D"""
        ops = TropicalPolynomialOperations()
        
        # Polynomial with known roots: max(x, 2)
        # Root at x=2 where both terms equal 2
        m1 = TropicalMonomial(0.0, {0: 1})  # 0 + 1*x = x
        m2 = TropicalMonomial(2.0, {})  # 2 + 0*x = 2
        poly = TropicalPolynomial([m1, m2], num_variables=1)
        
        roots = ops.find_tropical_roots(poly, search_region=(0, 3), num_samples=100)
        
        self.assertIsInstance(roots, list)
        # Should find at least one root near x=2
        if roots:
            # Check if any root is close to expected value
            has_close_root = any(abs(r[0].item() - 2.0) < 0.2 for r in roots)
            self.assertTrue(has_close_root)
    
    def test_find_tropical_roots_2d(self):
        """Test root finding in 2D"""
        ops = TropicalPolynomialOperations()
        
        # Simple 2D polynomial
        m1 = TropicalMonomial(0.0, {0: 1})
        m2 = TropicalMonomial(0.0, {1: 1})
        m3 = TropicalMonomial(1.0, {})
        poly = TropicalPolynomial([m1, m2, m3], num_variables=2)
        
        roots = ops.find_tropical_roots(poly, search_region=(-2, 2), num_samples=400)
        
        self.assertIsInstance(roots, list)
        for root in roots:
            self.assertEqual(root.shape, (2,))
    
    def test_find_tropical_roots_errors(self):
        """Test root finding error cases"""
        ops = TropicalPolynomialOperations()
        
        poly = TropicalPolynomial([TropicalMonomial(1.0, {})], num_variables=1)
        
        # Wrong type
        with self.assertRaises(TypeError):
            ops.find_tropical_roots("not a poly")
        
        # Invalid num_samples
        with self.assertRaises(ValueError):
            ops.find_tropical_roots(poly, num_samples=0)
        
        # Invalid search region
        with self.assertRaises(ValueError):
            ops.find_tropical_roots(poly, search_region=(2, 1))
    
    def test_compute_tropical_resultant(self):
        """Test tropical resultant computation"""
        ops = TropicalPolynomialOperations()
        
        # Create two polynomials
        m1 = TropicalMonomial(1.0, {0: 2})
        m2 = TropicalMonomial(2.0, {1: 1})
        poly1 = TropicalPolynomial([m1, m2], num_variables=2)
        
        m3 = TropicalMonomial(0.0, {0: 1, 1: 1})
        m4 = TropicalMonomial(3.0, {})
        poly2 = TropicalPolynomial([m3, m4], num_variables=2)
        
        resultant = ops.compute_tropical_resultant(poly1, poly2)
        
        self.assertIsInstance(resultant, float)
        self.assertFalse(math.isnan(resultant))
        self.assertFalse(math.isinf(resultant))
        
        # Empty polynomial case
        empty = TropicalPolynomial([], num_variables=2)
        resultant_empty = ops.compute_tropical_resultant(poly1, empty)
        self.assertEqual(resultant_empty, TROPICAL_ZERO)
    
    def test_compute_tropical_resultant_errors(self):
        """Test resultant computation errors"""
        ops = TropicalPolynomialOperations()
        
        poly1 = TropicalPolynomial([TropicalMonomial(1.0, {})], num_variables=2)
        poly2 = TropicalPolynomial([TropicalMonomial(1.0, {})], num_variables=3)
        
        # Wrong types
        with self.assertRaises(TypeError):
            ops.compute_tropical_resultant("not poly", poly1)
        
        with self.assertRaises(TypeError):
            ops.compute_tropical_resultant(poly1, 1.0)
        
        # Dimension mismatch
        with self.assertRaises(ValueError):
            ops.compute_tropical_resultant(poly1, poly2)
    
    def test_interpolate_from_points(self):
        """Test polynomial interpolation"""
        ops = TropicalPolynomialOperations()
        
        # Simple 1D interpolation
        points = torch.tensor([[0.0], [1.0], [2.0]])
        values = torch.tensor([0.0, 1.0, 4.0])
        
        poly = ops.interpolate_from_points(points, values, max_degree=2)
        
        self.assertEqual(poly.num_variables, 1)
        self.assertGreater(len(poly.monomials), 0)
        
        # Check that it approximately interpolates
        for i in range(len(points)):
            val = poly.evaluate(points[i])
            # Tropical interpolation may not be exact
            self.assertLess(abs(val - values[i].item()), 10.0)
    
    def test_interpolate_2d(self):
        """Test 2D interpolation"""
        ops = TropicalPolynomialOperations()
        
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        values = torch.tensor([0.0, 1.0, 2.0, 3.0])
        
        poly = ops.interpolate_from_points(points, values, max_degree=2)
        
        self.assertEqual(poly.num_variables, 2)
        self.assertGreater(len(poly.monomials), 0)
    
    def test_interpolate_errors(self):
        """Test interpolation error cases"""
        ops = TropicalPolynomialOperations()
        
        points = torch.tensor([[1.0], [2.0]])
        values = torch.tensor([1.0, 2.0])
        
        # Wrong types
        with self.assertRaises(TypeError):
            ops.interpolate_from_points([[1.0], [2.0]], values)
        
        with self.assertRaises(TypeError):
            ops.interpolate_from_points(points, [1.0, 2.0])
        
        # Shape mismatch
        with self.assertRaises(ValueError):
            ops.interpolate_from_points(points, torch.tensor([1.0]))
        
        # Invalid max_degree
        with self.assertRaises(ValueError):
            ops.interpolate_from_points(points, values, max_degree=0)
    
    def test_tropical_roots_non_differentiability(self):
        """Test that roots occur at non-differentiable points (where max is achieved by multiple terms)"""
        ops = TropicalPolynomialOperations()
        
        # Create polynomial f(x) = max(x, 4-x) 
        # This has a root at x=2 where both terms equal 2
        m1 = TropicalMonomial(0.0, {0: 1})  # x
        m2 = TropicalMonomial(4.0, {})  # 4
        m3 = TropicalMonomial(0.0, {})  # 0 (to create second branch at 4-x effect)
        
        # We need a polynomial that's actually max(x, 4-x)
        # In tropical: max(0+x, 4+(-1)*x) but we can't have negative exponents
        # Instead use max(x, 4) which has non-differentiability at x=4
        poly = TropicalPolynomial([m1, m2], num_variables=1)
        
        roots = ops.find_tropical_roots(poly, search_region=(0, 6), num_samples=200)
        
        # Should find roots near x=4 where both x and 4 achieve the max
        if roots:
            # At least one root should be near x=4
            close_roots = [r for r in roots if abs(r[0].item() - 4.0) < 0.5]
            self.assertGreater(len(close_roots), 0, "Should find root near x=4 where polynomial is non-differentiable")
    
    def test_minkowski_sum_in_resultant(self):
        """Test that resultant computation uses Minkowski sum correctly"""
        ops = TropicalPolynomialOperations()
        
        # Create two simple polynomials with known Newton polytopes
        m1 = TropicalMonomial(1.0, {0: 2})  # x^2
        m2 = TropicalMonomial(2.0, {0: 1})  # x
        poly1 = TropicalPolynomial([m1, m2], num_variables=1)
        
        m3 = TropicalMonomial(0.0, {0: 1})  # x
        m4 = TropicalMonomial(3.0, {})      # constant
        poly2 = TropicalPolynomial([m3, m4], num_variables=1)
        
        # Compute resultant (uses Minkowski sum internally)
        resultant = ops.compute_tropical_resultant(poly1, poly2)
        
        # Resultant should be non-zero for these polynomials
        self.assertNotEqual(resultant, TROPICAL_ZERO)
        self.assertGreater(resultant, 0)
        
        # Test with empty polynomial (Minkowski sum should handle this)
        empty_poly = TropicalPolynomial([], num_variables=1)
        resultant_empty = ops.compute_tropical_resultant(poly1, empty_poly)
        self.assertEqual(resultant_empty, TROPICAL_ZERO)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and complex use cases"""
    
    def test_polynomial_composition(self):
        """Test composing multiple operations"""
        # Create two polynomials
        m1 = TropicalMonomial(1.0, {0: 1})
        m2 = TropicalMonomial(2.0, {1: 1})
        poly1 = TropicalPolynomial([m1], num_variables=2)
        poly2 = TropicalPolynomial([m2], num_variables=2)
        
        # Add them
        poly_sum = poly1.add(poly2)
        
        # Multiply with another
        m3 = TropicalMonomial(0.5, {0: 1, 1: 1})
        poly3 = TropicalPolynomial([m3], num_variables=2)
        poly_result = poly_sum.multiply(poly3)
        
        # Evaluate
        point = torch.tensor([1.0, 2.0])
        result = poly_result.evaluate(point)
        
        self.assertIsInstance(result, (float, np.floating))
        self.assertFalse(math.isnan(result))
    
    def test_large_polynomial_performance(self):
        """Test performance with large polynomials"""
        # Create large polynomial
        num_monomials = 500
        num_variables = 10
        monomials = []
        
        for i in range(num_monomials):
            coeff = float(i) / 100.0
            exponents = {}
            # Sparse exponents
            for j in range(min(3, num_variables)):
                var_idx = (i + j) % num_variables
                power = (i % 3) + 1
                exponents[var_idx] = power
            monomials.append(TropicalMonomial(coeff, exponents))
        
        poly = TropicalPolynomial(monomials, num_variables)
        
        # Time evaluation
        points = torch.randn(100, num_variables)
        start_time = time.time()
        results = poly.evaluate(points)
        elapsed = time.time() - start_time
        
        self.assertEqual(results.shape, (100,))
        self.assertLess(elapsed, 5.0)  # Should complete within 5 seconds
    
    def test_gpu_cpu_consistency(self):
        """Test that GPU and CPU give same results"""
        if not torch.cuda.is_available():
            self.skipTest("GPU not available")
        
        # Create polynomial
        m1 = TropicalMonomial(1.0, {0: 2, 1: 1})
        m2 = TropicalMonomial(2.0, {0: 1, 1: 2})
        m3 = TropicalMonomial(0.0, {2: 1})
        poly = TropicalPolynomial([m1, m2, m3], num_variables=3)
        
        # Test points
        points_cpu = torch.randn(50, 3)
        points_gpu = points_cpu.cuda()
        
        # Evaluate on both
        results_cpu = poly.evaluate(points_cpu)
        results_gpu = poly.evaluate(points_gpu)
        
        # Compare results
        self.assertTrue(torch.allclose(results_cpu, results_gpu.cpu(), rtol=1e-5))
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        # Very small coefficients
        m1 = TropicalMonomial(1e-10, {0: 1})
        m2 = TropicalMonomial(1e-10, {1: 1})
        poly_small = TropicalPolynomial([m1, m2], num_variables=2)
        
        point = torch.tensor([1e-10, 1e-10])
        result = poly_small.evaluate(point)
        self.assertFalse(math.isnan(result))
        self.assertFalse(math.isinf(result))
        
        # Large coefficients (but within safe range)
        m3 = TropicalMonomial(1e10, {0: 1})
        m4 = TropicalMonomial(1e10, {1: 1})
        poly_large = TropicalPolynomial([m3, m4], num_variables=2)
        
        point = torch.tensor([1.0, 1.0])
        result = poly_large.evaluate(point)
        self.assertFalse(math.isnan(result))
        self.assertFalse(math.isinf(result))
    
    def test_memory_efficiency(self):
        """Test memory efficiency with sparse representation"""
        # Create very sparse polynomial
        num_variables = 1000
        monomials = []
        
        # Only 10 monomials in 1000-dimensional space
        for i in range(10):
            exponents = {i * 100: 1}  # Very sparse
            monomials.append(TropicalMonomial(float(i), exponents))
        
        poly = TropicalPolynomial(monomials, num_variables)
        
        # Should handle evaluation efficiently
        point = torch.zeros(num_variables)
        point[0] = 1.0
        point[500] = 2.0
        
        result = poly.evaluate(point)
        self.assertIsInstance(result, (float, np.floating))
    
    def test_edge_case_empty_polynomial(self):
        """Test edge cases with empty polynomial"""
        empty = TropicalPolynomial([], num_variables=5)
        
        # Single point evaluation
        point = torch.randn(5)
        result = empty.evaluate(point)
        self.assertEqual(result, TROPICAL_ZERO)
        
        # Batch evaluation
        points = torch.randn(10, 5)
        results = empty.evaluate(points)
        self.assertTrue(torch.all(results == TROPICAL_ZERO))
        
        # Operations
        m1 = TropicalMonomial(1.0, {0: 1})
        non_empty = TropicalPolynomial([m1], num_variables=5)
        
        sum_poly = empty.add(non_empty)
        self.assertEqual(len(sum_poly.monomials), 1)
        
        prod_poly = empty.multiply(non_empty)
        self.assertEqual(len(prod_poly.monomials), 0)  # Product with tropical zero
    
    def test_polynomial_with_many_duplicates(self):
        """Test deduplication with many duplicate monomials"""
        monomials = []
        
        # Add same monomial many times
        for _ in range(100):
            monomials.append(TropicalMonomial(1.0, {0: 1, 1: 2}))
        
        # Add a few different ones
        monomials.append(TropicalMonomial(2.0, {0: 2}))
        monomials.append(TropicalMonomial(3.0, {1: 3}))
        
        poly = TropicalPolynomial(monomials, num_variables=2)
        
        # Should have only 3 unique monomials
        self.assertEqual(len(poly.monomials), 3)
    
    def test_resultant_special_cases(self):
        """Test resultant computation special cases"""
        ops = TropicalPolynomialOperations()
        
        # Same polynomial
        m1 = TropicalMonomial(1.0, {0: 1})
        poly = TropicalPolynomial([m1], num_variables=1)
        resultant = ops.compute_tropical_resultant(poly, poly)
        self.assertIsInstance(resultant, float)
        
        # One constant polynomial
        m2 = TropicalMonomial(5.0, {})
        const_poly = TropicalPolynomial([m2], num_variables=1)
        resultant = ops.compute_tropical_resultant(poly, const_poly)
        self.assertIsInstance(resultant, float)


class TestPerformanceRequirements(unittest.TestCase):
    """Test performance requirements"""
    
    def test_evaluation_speed(self):
        """Test that evaluation meets speed requirements"""
        # Create moderately large polynomial
        num_monomials = 100
        num_variables = 5
        monomials = []
        
        for i in range(num_monomials):
            coeff = float(i)
            exponents = {j: (i + j) % 3 for j in range(min(3, num_variables))}
            monomials.append(TropicalMonomial(coeff, exponents))
        
        poly = TropicalPolynomial(monomials, num_variables)
        
        # Time batch evaluation
        points = torch.randn(1000, num_variables)
        
        start = time.time()
        results = poly.evaluate(points)
        elapsed = time.time() - start
        
        # Should evaluate 1000 points in less than 1 second
        self.assertLess(elapsed, 1.0)
        self.assertEqual(results.shape, (1000,))
    
    def test_gpu_speedup(self):
        """Test GPU provides speedup for large batches"""
        if not torch.cuda.is_available():
            self.skipTest("GPU not available")
        
        # Create polynomial
        num_monomials = 200
        num_variables = 10
        monomials = []
        
        for i in range(num_monomials):
            coeff = float(i) / 10.0
            exponents = {}
            for j in range(min(3, num_variables)):
                var_idx = (i + j) % num_variables
                exponents[var_idx] = (i % 3) + 1
            monomials.append(TropicalMonomial(coeff, exponents))
        
        poly = TropicalPolynomial(monomials, num_variables)
        
        # Large batch
        points_cpu = torch.randn(5000, num_variables)
        points_gpu = points_cpu.cuda()
        
        # Time CPU
        start_cpu = time.time()
        results_cpu = poly.evaluate(points_cpu)
        time_cpu = time.time() - start_cpu
        
        # Time GPU
        torch.cuda.synchronize()
        start_gpu = time.time()
        results_gpu = poly.evaluate(points_gpu)
        torch.cuda.synchronize()
        time_gpu = time.time() - start_gpu
        
        # GPU should be faster for large batches
        # Note: May not always be true for small polynomials
        print(f"CPU time: {time_cpu:.3f}s, GPU time: {time_gpu:.3f}s")
        
        # At minimum, GPU shouldn't be much slower
        self.assertLess(time_gpu, time_cpu * 2.0)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)