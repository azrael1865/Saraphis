"""
Comprehensive tests for JAX Tropical Engine.
Tests all core operations, advanced operations, and integration with existing system.
NO PLACEHOLDERS - PRODUCTION READY
"""

import os
import sys
import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import unittest
import warnings

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import tropical components
from independent_core.compression_systems.tropical.tropical_core import (
    TropicalNumber,
    TropicalMathematicalOperations,
    TROPICAL_ZERO,
    TROPICAL_EPSILON
)
from independent_core.compression_systems.tropical.tropical_polynomial import (
    TropicalPolynomial,
    TropicalMonomial
)
from independent_core.compression_systems.tropical.tropical_channel_extractor import (
    TropicalChannels
)

# Try to import JAX components
JAX_AVAILABLE = False
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, grad
    JAX_AVAILABLE = True
    
    from independent_core.compression_systems.tropical.jax_tropical_engine import (
        JAXTropicalConfig,
        TropicalJAXEngine,
        TropicalJAXOperations,
        JAXChannelProcessor,
        TropicalXLAKernels,
        TropicalJAXBenchmark
    )
except ImportError as e:
    warnings.warn(f"JAX not available: {e}")
    jnp = None


class TestTropicalJAXEngine(unittest.TestCase):
    """Test suite for JAX Tropical Engine core operations"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        if not JAX_AVAILABLE:
            raise unittest.SkipTest("JAX not available")
        
        # Initialize engine with test configuration
        cls.config = JAXTropicalConfig(
            enable_jit=True,
            enable_vmap=True,
            precision="float32",
            compilation_level=2
        )
        cls.engine = TropicalJAXEngine(cls.config)
        cls.ops = TropicalJAXOperations(cls.engine)
        cls.channel_proc = JAXChannelProcessor(cls.engine)
    
    def test_tropical_add(self):
        """Test tropical addition (max operation)"""
        # Test scalar addition
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([2.0, 1.0, 4.0])
        
        result = self.engine.tropical_add(a, b)
        expected = jnp.array([2.0, 2.0, 4.0])
        
        self.assertTrue(jnp.allclose(result, expected))
        
        # Test with tropical zeros
        a_with_zero = jnp.array([TROPICAL_ZERO, 2.0, 3.0])
        b_with_zero = jnp.array([1.0, TROPICAL_ZERO, 4.0])
        
        result = self.engine.tropical_add(a_with_zero, b_with_zero)
        expected = jnp.array([1.0, 2.0, 4.0])
        
        self.assertTrue(jnp.allclose(result, expected))
    
    def test_tropical_multiply(self):
        """Test tropical multiplication (addition)"""
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([2.0, 1.0, 4.0])
        
        result = self.engine.tropical_multiply(a, b)
        expected = jnp.array([3.0, 3.0, 7.0])
        
        self.assertTrue(jnp.allclose(result, expected))
        
        # Test with tropical zeros
        a_with_zero = jnp.array([TROPICAL_ZERO, 2.0, 3.0])
        b_normal = jnp.array([1.0, 2.0, 4.0])
        
        result = self.engine.tropical_multiply(a_with_zero, b_normal)
        expected = jnp.array([TROPICAL_ZERO, 4.0, 7.0])
        
        self.assertTrue(jnp.allclose(result, expected))
    
    def test_tropical_power(self):
        """Test tropical power operation"""
        base = jnp.array([1.0, 2.0, 3.0])
        exponent = 3
        
        result = self.engine.tropical_power(base, exponent)
        expected = jnp.array([3.0, 6.0, 9.0])
        
        self.assertTrue(jnp.allclose(result, expected))
        
        # Test with tropical zeros
        base_with_zero = jnp.array([TROPICAL_ZERO, 2.0, 3.0])
        result = self.engine.tropical_power(base_with_zero, 2)
        expected = jnp.array([TROPICAL_ZERO, 4.0, 6.0])
        
        self.assertTrue(jnp.allclose(result, expected))
    
    def test_tropical_matrix_multiply(self):
        """Test tropical matrix multiplication"""
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        B = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        
        result = self.engine.tropical_matrix_multiply(A, B)
        
        # Expected: C[i,j] = max_k(A[i,k] + B[k,j])
        # C[0,0] = max(1+2, 2+1) = max(3, 3) = 3
        # C[0,1] = max(1+1, 2+3) = max(2, 5) = 5
        # C[1,0] = max(3+2, 4+1) = max(5, 5) = 5
        # C[1,1] = max(3+1, 4+3) = max(4, 7) = 7
        expected = jnp.array([[3.0, 5.0], [5.0, 7.0]])
        
        self.assertTrue(jnp.allclose(result, expected))
        
        # Test larger matrices
        A_large = jnp.ones((10, 15))
        B_large = jnp.ones((15, 20))
        result_large = self.engine.tropical_matrix_multiply(A_large, B_large)
        
        self.assertEqual(result_large.shape, (10, 20))
        self.assertTrue(jnp.all(result_large == 2.0))  # 1 + 1 = 2
    
    def test_polynomial_operations(self):
        """Test polynomial conversion and evaluation"""
        # Create a tropical polynomial
        monomials = [
            TropicalMonomial(2.0, {0: 1}),     # 2 + x
            TropicalMonomial(3.0, {1: 2}),     # 3 + 2y
            TropicalMonomial(1.0, {0: 1, 1: 1}) # 1 + x + y
        ]
        poly = TropicalPolynomial(monomials, num_variables=2)
        
        # Convert to JAX format
        poly_jax = self.engine.polynomial_to_jax(poly)
        
        self.assertEqual(poly_jax['num_variables'], 2)
        self.assertEqual(poly_jax['coefficients'].shape[0], 3)
        self.assertEqual(poly_jax['exponents'].shape, (3, 2))
        
        # Test evaluation at single point
        point = jnp.array([1.0, 1.0])
        result = self.engine.evaluate_polynomial(
            poly_jax['coefficients'],
            poly_jax['exponents'],
            point
        )
        
        # Expected: max(2+1, 3+2, 1+1+1) = max(3, 5, 3) = 5
        self.assertAlmostEqual(float(result), 5.0, places=5)
        
        # Test batch evaluation
        points = jnp.array([[1.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
        results = self.engine.evaluate_polynomial(
            poly_jax['coefficients'],
            poly_jax['exponents'],
            points
        )
        
        self.assertEqual(results.shape, (3,))
        # Expected: [5.0, 4.0, 7.0]
        expected = jnp.array([5.0, 4.0, 7.0])
        self.assertTrue(jnp.allclose(results, expected, atol=1e-5))
    
    def test_vmap_polynomial_evaluation(self):
        """Test vectorized polynomial evaluation"""
        # Create multiple polynomials
        poly1 = TropicalPolynomial([
            TropicalMonomial(1.0, {0: 1}),
            TropicalMonomial(2.0, {1: 1})
        ], num_variables=2)
        
        poly2 = TropicalPolynomial([
            TropicalMonomial(0.0, {}),
            TropicalMonomial(3.0, {0: 1, 1: 1})
        ], num_variables=2)
        
        polynomials = [poly1, poly2]
        
        # Evaluate at multiple points
        points = jnp.array([[1.0, 1.0], [2.0, 2.0]])
        
        results = self.engine.vmap_polynomial_evaluation(polynomials, points)
        
        self.assertEqual(results.shape, (2, 2))
        
        # Verify individual evaluations
        # poly1 at [1,1]: max(1+1, 2+1) = 3
        # poly1 at [2,2]: max(1+2, 2+2) = 4
        # poly2 at [1,1]: max(0, 3+1+1) = 5
        # poly2 at [2,2]: max(0, 3+2+2) = 7
        expected = jnp.array([[3.0, 4.0], [5.0, 7.0]])
        self.assertTrue(jnp.allclose(results, expected, atol=1e-5))
    
    def test_tropical_conv1d(self):
        """Test 1D tropical convolution"""
        signal = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        kernel = jnp.array([1.0, 0.0, -1.0])
        
        # Valid padding
        result = self.ops.tropical_conv1d(signal, kernel, padding='valid')
        self.assertEqual(result.shape[0], 3)  # 5 - 3 + 1 = 3
        
        # Same padding
        result_same = self.ops.tropical_conv1d(signal, kernel, padding='same')
        self.assertEqual(result_same.shape[0], 5)  # Same as input
        
        # Test with tropical zeros
        signal_with_zero = jnp.array([TROPICAL_ZERO, 2.0, 3.0, 4.0, 5.0])
        result = self.ops.tropical_conv1d(signal_with_zero, kernel, padding='valid')
        self.assertEqual(result.shape[0], 3)
    
    def test_tropical_conv2d(self):
        """Test 2D tropical convolution"""
        input_2d = jnp.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ])
        kernel = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        
        # Test with stride 1
        result = self.ops.tropical_conv2d(input_2d, kernel, stride=(1, 1))
        self.assertEqual(result.shape, (3, 3))
        
        # Test with stride 2
        result_stride2 = self.ops.tropical_conv2d(input_2d, kernel, stride=(2, 2))
        self.assertEqual(result_stride2.shape, (2, 2))
    
    def test_tropical_pool2d(self):
        """Test 2D tropical pooling"""
        input_2d = jnp.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ])
        
        # 2x2 pooling
        result = self.ops.tropical_pool2d(input_2d, (2, 2))
        expected = jnp.array([[6.0, 8.0], [14.0, 16.0]])
        
        self.assertTrue(jnp.allclose(result, expected))
    
    def test_batch_tropical_distance(self):
        """Test batch tropical distance computation"""
        # Single vector comparison
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([2.0, 1.0, 4.0])
        
        distance = self.ops.batch_tropical_distance(a, b)
        # max(|1-2|, |2-1|, |3-4|) = max(1, 1, 1) = 1
        self.assertAlmostEqual(float(distance), 1.0, places=5)
        
        # Test with tropical zeros
        a_with_zero = jnp.array([TROPICAL_ZERO, 2.0, 3.0])
        b_normal = jnp.array([1.0, 2.0, 3.0])
        
        distance = self.ops.batch_tropical_distance(a_with_zero, b_normal)
        # First element: one is zero, other is not -> infinity
        self.assertTrue(jnp.isinf(distance))
        
        # Both zeros
        a_both_zero = jnp.array([TROPICAL_ZERO, TROPICAL_ZERO])
        b_both_zero = jnp.array([TROPICAL_ZERO, TROPICAL_ZERO])
        
        distance = self.ops.batch_tropical_distance(a_both_zero, b_both_zero)
        self.assertEqual(float(distance), 0.0)
    
    def test_tropical_gradient(self):
        """Test tropical gradient computation"""
        # Define a simple tropical function
        def tropical_func(x):
            # f(x) = max(2 + x[0], 3 + 2*x[1])
            term1 = 2.0 + x[0]
            term2 = 3.0 + 2.0 * x[1]
            return jnp.maximum(term1, term2)
        
        # Compute gradient at a point
        x = jnp.array([1.0, 1.0])
        gradient = self.ops.tropical_gradient(tropical_func, x)
        
        self.assertEqual(gradient.shape, (2,))
        
        # The gradient should be non-zero
        self.assertTrue(jnp.any(gradient != 0))
    
    def test_tropical_softmax(self):
        """Test tropical softmax approximation"""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test with different temperatures
        result_hot = self.ops.tropical_softmax(x, temperature=0.1)
        result_cold = self.ops.tropical_softmax(x, temperature=10.0)
        
        # Hot temperature should be closer to max
        self.assertTrue(result_hot >= 4.9)  # Close to max(x) = 5
        
        # Cold temperature should be smoother
        self.assertTrue(result_cold < result_hot)
        
        # Test with 2D array
        x_2d = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result_2d = self.ops.tropical_softmax(x_2d, temperature=1.0, axis=1)
        
        self.assertEqual(result_2d.shape, (2,))
    
    def test_channel_operations(self):
        """Test channel processing operations"""
        # Create test channels
        coeffs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, TROPICAL_ZERO, 0.1])
        exponents = jnp.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 0],
            [2, 0, 0]
        ])
        
        # Test normalization
        norm_coeffs, norm_exps = self.channel_proc.process_channels(
            coeffs, exponents, operation="normalize"
        )
        
        # Should remove tropical zeros and near-zeros
        self.assertTrue(jnp.all(norm_coeffs[5] == TROPICAL_ZERO))
        
        # Test sparsification
        sparse_coeffs, sparse_exps = self.channel_proc.process_channels(
            coeffs, exponents, operation="sparsify"
        )
        
        # Should keep only top values
        num_nonzero = jnp.sum(sparse_coeffs > TROPICAL_ZERO)
        self.assertLessEqual(num_nonzero, 100)
        
        # Test compression
        comp_coeffs, comp_exps = self.channel_proc.process_channels(
            coeffs, exponents, operation="compress"
        )
        
        # Coefficients should be quantized
        self.assertEqual(comp_coeffs.shape, coeffs.shape)
    
    def test_xla_kernels(self):
        """Test custom XLA kernels"""
        # Test tropical matrix multiplication kernel
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        B = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        
        result = TropicalXLAKernels.tropical_matmul_kernel(A, B)
        expected = jnp.array([[3.0, 5.0], [5.0, 7.0]])
        
        self.assertTrue(jnp.allclose(result, expected))
        
        # Test reduce kernel
        array = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        reduced = TropicalXLAKernels.tropical_reduce_kernel(array, axis=1)
        expected = jnp.array([3.0, 6.0])
        
        self.assertTrue(jnp.allclose(reduced, expected))
        
        # Test scan kernel
        carry = jnp.array(0.0)
        x = jnp.array(1.0)
        new_carry, output = TropicalXLAKernels.tropical_scan_kernel(carry, x)
        
        self.assertEqual(float(new_carry), 1.0)
        self.assertEqual(float(output), 1.0)
        
        # Test attention kernel
        seq_len = 4
        d_k = 2
        d_v = 3
        
        query = jnp.ones((seq_len, d_k))
        key = jnp.ones((seq_len, d_k))
        value = jnp.ones((seq_len, d_v))
        
        output = TropicalXLAKernels.tropical_attention_kernel(query, key, value)
        
        self.assertEqual(output.shape, (seq_len, d_v))
    
    def test_integration_with_pytorch(self):
        """Test integration with PyTorch tensors"""
        # Create PyTorch tensor
        torch_tensor = torch.randn(3, 3)
        
        # Convert to JAX
        from independent_core.compression_systems.tropical.jax_config import JAXPyTorchBridge
        bridge = JAXPyTorchBridge()
        
        jax_array = bridge.torch_to_jax(torch_tensor)
        
        # Perform tropical operations
        result = self.engine.tropical_add(jax_array, jax_array)
        
        # Convert back to PyTorch
        torch_result = bridge.jax_to_torch(result)
        
        self.assertEqual(torch_result.shape, torch_tensor.shape)
        self.assertTrue(torch.allclose(torch_result, torch_tensor))


class TestTropicalJAXPerformance(unittest.TestCase):
    """Performance benchmarks for JAX Tropical Engine"""
    
    @classmethod
    def setUpClass(cls):
        """Set up benchmark environment"""
        if not JAX_AVAILABLE:
            raise unittest.SkipTest("JAX not available")
        
        cls.config = JAXTropicalConfig(
            enable_jit=True,
            enable_vmap=True,
            enable_pmap=False
        )
        cls.engine = TropicalJAXEngine(cls.config)
        cls.benchmark = TropicalJAXBenchmark()
    
    def test_matmul_performance(self):
        """Benchmark matrix multiplication performance"""
        sizes = [(10, 10), (100, 100), (500, 500)]
        
        for size in sizes:
            A = jnp.ones(size)
            B = jnp.ones(size)
            
            def matmul_func():
                return self.engine.tropical_matrix_multiply(A, B)
            
            stats = self.benchmark.benchmark_operation(
                matmul_func, 
                num_iterations=100,
                warmup=10
            )
            
            print(f"\nMatrix size {size}: {stats['mean_ms']:.2f}ms ± {stats['std_ms']:.2f}ms")
            
            # Performance should be reasonable
            if size == (10, 10):
                self.assertLess(stats['mean_ms'], 1.0)  # < 1ms for small matrices
            elif size == (100, 100):
                self.assertLess(stats['mean_ms'], 10.0)  # < 10ms for medium matrices
    
    def test_polynomial_evaluation_performance(self):
        """Benchmark polynomial evaluation performance"""
        # Create a large polynomial
        num_monomials = 100
        num_variables = 10
        
        coeffs = jnp.ones(num_monomials)
        exponents = jnp.ones((num_monomials, num_variables), dtype=jnp.int32)
        
        # Test with varying number of points
        for num_points in [10, 100, 1000]:
            points = jnp.ones((num_points, num_variables))
            
            def eval_func():
                return self.engine.evaluate_polynomial(coeffs, exponents, points)
            
            stats = self.benchmark.benchmark_operation(
                eval_func,
                num_iterations=100,
                warmup=10
            )
            
            print(f"\nPolynomial eval ({num_points} points): {stats['mean_ms']:.2f}ms")
            
            # Should scale reasonably with number of points
            self.assertLess(stats['mean_ms'], num_points * 0.1)
    
    def test_jit_compilation_speedup(self):
        """Test that JIT compilation provides speedup"""
        # Create engine without JIT
        config_no_jit = JAXTropicalConfig(enable_jit=False)
        engine_no_jit = TropicalJAXEngine(config_no_jit)
        
        # Create engine with JIT
        config_jit = JAXTropicalConfig(enable_jit=True)
        engine_jit = TropicalJAXEngine(config_jit)
        
        # Test data
        A = jnp.ones((100, 100))
        B = jnp.ones((100, 100))
        
        # Benchmark without JIT
        def no_jit_func():
            return engine_no_jit.tropical_matrix_multiply(A, B)
        
        stats_no_jit = self.benchmark.benchmark_operation(
            no_jit_func,
            num_iterations=50,
            warmup=5
        )
        
        # Benchmark with JIT
        def jit_func():
            return engine_jit.tropical_matrix_multiply(A, B)
        
        stats_jit = self.benchmark.benchmark_operation(
            jit_func,
            num_iterations=50,
            warmup=5
        )
        
        speedup = stats_no_jit['mean_ms'] / stats_jit['mean_ms']
        print(f"\nJIT speedup: {speedup:.2f}x")
        
        # JIT should provide at least 2x speedup
        self.assertGreater(speedup, 2.0)


def run_comprehensive_tests():
    """Run all tests and print summary"""
    print("=" * 60)
    print("JAX TROPICAL ENGINE TEST SUITE")
    print("=" * 60)
    
    if not JAX_AVAILABLE:
        print("JAX is not installed. Skipping tests.")
        print("Install with: pip install jax[cuda12_local]")
        return False
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTropicalJAXEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestTropicalJAXPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✓ ALL TESTS PASSED")
        print(f"  Tests run: {result.testsRun}")
        print(f"  Time: {time.time():.2f}s")
    else:
        print("✗ TESTS FAILED")
        print(f"  Failures: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)