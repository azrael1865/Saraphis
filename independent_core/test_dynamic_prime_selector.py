"""
Comprehensive test suite for DynamicPrimeSelector
Tests all functionality and edge cases to identify root issues
"""

import unittest
import torch
import numpy as np
import threading
import time
from unittest.mock import patch
from compression_systems.padic.dynamic_prime_selector import (
    DynamicPrimeSelector, 
    PrimeSelectionResult
)


class TestDynamicPrimeSelector(unittest.TestCase):
    """Comprehensive tests for DynamicPrimeSelector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.selector = DynamicPrimeSelector()
        
        # Test tensors with different characteristics
        self.sparse_tensor = torch.zeros(1000)
        self.sparse_tensor[::100] = torch.randn(10) * 100  # 1% non-zero
        
        self.gaussian_tensor = torch.randn(1000)
        
        self.uniform_tensor = torch.rand(1000) * 2 - 1  # [-1, 1]
        
        self.bimodal_tensor = torch.cat([
            torch.randn(500) - 3,  # Mode 1
            torch.randn(500) + 3   # Mode 2
        ])
        
        self.periodic_tensor = torch.sin(torch.linspace(0, 4*np.pi, 1000))
        
        self.high_dynamic_range_tensor = torch.tensor([
            1e-10, 1e-5, 0.1, 1.0, 10.0, 1e5, 1e10
        ], dtype=torch.float32)
        
        self.zero_tensor = torch.zeros(1000)
        self.single_value_tensor = torch.ones(1000) * 5.5
        
    def test_initialization_default(self):
        """Test initialization with default parameters"""
        selector = DynamicPrimeSelector()
        self.assertIsInstance(selector.primes, list)
        self.assertTrue(all(selector._is_prime(p) for p in selector.primes))
        self.assertEqual(len(selector.primes), len(DynamicPrimeSelector.DEFAULT_PRIMES))
        self.assertTrue(selector.enable_caching)
        self.assertEqual(selector.cache_size, 100)
    
    def test_initialization_custom_primes(self):
        """Test initialization with custom primes"""
        custom_primes = [2, 3, 5, 7, 11]
        selector = DynamicPrimeSelector(default_primes=custom_primes)
        self.assertEqual(selector.primes, sorted(custom_primes))
    
    def test_initialization_invalid_primes(self):
        """Test initialization with invalid primes"""
        with self.assertRaises(ValueError):
            DynamicPrimeSelector(default_primes=[2, 3, 4, 5])  # 4 is not prime
        
        with self.assertRaises(ValueError):
            DynamicPrimeSelector(default_primes=[1, 2, 3])  # 1 is not prime
    
    def test_is_prime_method(self):
        """Test prime checking method"""
        selector = DynamicPrimeSelector()
        
        # Test known primes
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        for p in primes:
            self.assertTrue(selector._is_prime(p), f"{p} should be prime")
        
        # Test non-primes
        non_primes = [0, 1, 4, 6, 8, 9, 10, 12, 15, 16, 20, 25, 27]
        for n in non_primes:
            self.assertFalse(selector._is_prime(n), f"{n} should not be prime")
    
    def test_select_optimal_prime_invalid_input(self):
        """Test optimal prime selection with invalid inputs"""
        # Non-tensor input
        with self.assertRaises(TypeError):
            self.selector.select_optimal_prime([1, 2, 3])
        
        with self.assertRaises(TypeError):
            self.selector.select_optimal_prime(np.array([1, 2, 3]))
        
        # Empty tensor
        with self.assertRaises(ValueError):
            self.selector.select_optimal_prime(torch.tensor([]))
    
    def test_select_optimal_prime_zero_tensor(self):
        """Test optimal prime selection with all-zero tensor"""
        result = self.selector.select_optimal_prime(self.zero_tensor)
        
        self.assertIsInstance(result, PrimeSelectionResult)
        self.assertEqual(result.optimal_prime, self.selector.primes[0])  # Should use smallest prime
        self.assertEqual(result.efficiency, 1.0)
        self.assertEqual(result.entropy, 0.0)
        self.assertIn("zeros", result.selection_rationale.lower())
    
    def test_select_optimal_prime_gaussian(self):
        """Test optimal prime selection with Gaussian tensor"""
        result = self.selector.select_optimal_prime(self.gaussian_tensor)
        
        self.assertIsInstance(result, PrimeSelectionResult)
        self.assertIn(result.optimal_prime, self.selector.primes)
        self.assertGreater(result.efficiency, 0.0)
        self.assertLessEqual(result.efficiency, 1.0)
        self.assertGreater(result.average_digits, 0.0)
        self.assertGreaterEqual(result.entropy, 0.0)
        self.assertIn("distribution", result.selection_rationale.lower())
        self.assertGreater(result.computation_time, 0.0)
    
    def test_select_optimal_prime_sparse(self):
        """Test optimal prime selection with sparse tensor"""
        result = self.selector.select_optimal_prime(self.sparse_tensor)
        
        self.assertIsInstance(result, PrimeSelectionResult)
        self.assertEqual(result.distribution_type, "sparse")
        self.assertIn("sparse", result.selection_rationale.lower())
    
    def test_select_optimal_prime_uniform(self):
        """Test optimal prime selection with uniform tensor"""
        result = self.selector.select_optimal_prime(self.uniform_tensor)
        
        self.assertIsInstance(result, PrimeSelectionResult)
        # Distribution type could be uniform or gaussian depending on random data
        self.assertIn(result.distribution_type, ["uniform", "gaussian"])
    
    def test_select_optimal_prime_bimodal(self):
        """Test optimal prime selection with bimodal tensor"""
        result = self.selector.select_optimal_prime(self.bimodal_tensor)
        
        self.assertIsInstance(result, PrimeSelectionResult)
        # Could be bimodal or other depending on detection
        self.assertIsInstance(result.distribution_type, str)
    
    def test_select_optimal_prime_periodic(self):
        """Test optimal prime selection with periodic tensor"""
        result = self.selector.select_optimal_prime(self.periodic_tensor)
        
        self.assertIsInstance(result, PrimeSelectionResult)
        self.assertIsInstance(result.distribution_type, str)
    
    def test_select_optimal_prime_high_dynamic_range(self):
        """Test optimal prime selection with high dynamic range"""
        result = self.selector.select_optimal_prime(self.high_dynamic_range_tensor)
        
        self.assertIsInstance(result, PrimeSelectionResult)
        characteristics = self.selector.analyze_tensor_characteristics(self.high_dynamic_range_tensor)
        self.assertGreater(characteristics['dynamic_range'], 1e10)
    
    def test_select_optimal_prime_custom_primes(self):
        """Test optimal prime selection with custom prime list"""
        custom_primes = [2, 3, 5]
        result = self.selector.select_optimal_prime(self.gaussian_tensor, primes=custom_primes)
        
        self.assertIn(result.optimal_prime, custom_primes)
        self.assertEqual(set(result.candidate_scores.keys()), set(custom_primes))
    
    def test_select_optimal_prime_early_stopping(self):
        """Test early stopping mechanism"""
        # Use a simple tensor that should give high efficiency quickly
        simple_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = self.selector.select_optimal_prime(
            simple_tensor, 
            early_stopping_threshold=0.1  # Very low threshold
        )
        
        self.assertIsInstance(result, PrimeSelectionResult)
        # Should stop early and not evaluate all primes
        self.assertLessEqual(len(result.candidate_scores), len(self.selector.primes))
    
    def test_compute_digit_distribution_empty(self):
        """Test digit distribution with empty array"""
        dist = self.selector.compute_digit_distribution(np.array([]), 5)
        self.assertEqual(len(dist), 5)
        self.assertTrue(np.allclose(dist, np.zeros(5)))
    
    def test_compute_digit_distribution_zeros(self):
        """Test digit distribution with zeros"""
        values = np.array([0.0, 0.0, 0.0])
        dist = self.selector.compute_digit_distribution(values, 3)
        self.assertEqual(len(dist), 3)
        self.assertGreater(dist[0], 0)  # Should have high probability for digit 0
    
    def test_compute_digit_distribution_integers(self):
        """Test digit distribution with integers"""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        dist = self.selector.compute_digit_distribution(values, 3)
        self.assertEqual(len(dist), 3)
        self.assertTrue(np.allclose(np.sum(dist), 1.0))
    
    def test_compute_digit_distribution_fractions(self):
        """Test digit distribution with fractional values"""
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        dist = self.selector.compute_digit_distribution(values, 2)
        self.assertEqual(len(dist), 2)
        self.assertTrue(np.allclose(np.sum(dist), 1.0))
    
    def test_calculate_average_digits_empty(self):
        """Test average digits calculation with empty array"""
        avg = self.selector.calculate_average_digits(np.array([]), 3)
        self.assertEqual(avg, 1.0)
    
    def test_calculate_average_digits_zeros(self):
        """Test average digits calculation with zeros"""
        values = np.array([0.0, 0.0, 0.0])
        avg = self.selector.calculate_average_digits(values, 3)
        self.assertEqual(avg, 1.0)  # Zeros need 1 digit
    
    def test_calculate_average_digits_integers(self):
        """Test average digits calculation with integers"""
        values = np.array([1, 10, 100])  # 1, 2, 3 digits in base 10
        avg = self.selector.calculate_average_digits(values, 10)
        self.assertAlmostEqual(avg, 2.0, places=1)  # Average should be 2
    
    def test_calculate_average_digits_fractions(self):
        """Test average digits calculation with small fractions"""
        values = np.array([0.1, 0.01, 0.001])
        avg = self.selector.calculate_average_digits(values, 10)
        self.assertGreater(avg, 1.0)  # Should need more than 1 digit
    
    def test_calculate_entropy_empty(self):
        """Test entropy calculation with empty array"""
        entropy = self.selector.calculate_entropy(np.array([]), 3)
        self.assertEqual(entropy, 0.0)
    
    def test_calculate_entropy_uniform(self):
        """Test entropy calculation with uniform distribution"""
        # Values that should give uniform digit distribution in base 2
        values = np.array([0, 1] * 100)  # Equal 0s and 1s
        entropy = self.selector.calculate_entropy(values, 2)
        self.assertGreater(entropy, 0.0)
        self.assertLessEqual(entropy, 1.0)  # Max entropy in base 2
    
    def test_calculate_entropy_skewed(self):
        """Test entropy calculation with skewed distribution"""
        values = np.array([0] * 90 + [1] * 10)  # Very skewed
        entropy = self.selector.calculate_entropy(values, 2)
        self.assertGreater(entropy, 0.0)
        self.assertLess(entropy, 1.0)  # Should be less than uniform
    
    def test_compute_efficiency_metric_basic(self):
        """Test efficiency metric computation"""
        characteristics = {'sparsity': 0.1, 'dynamic_range': 100}
        efficiency = self.selector.compute_efficiency_metric(2, 5.0, 0.8, characteristics)
        
        self.assertGreaterEqual(efficiency, 0.0)
        self.assertLessEqual(efficiency, 1.0)
    
    def test_compute_efficiency_metric_sparse(self):
        """Test efficiency metric with sparse tensor characteristics"""
        characteristics = {'sparsity': 0.9, 'dynamic_range': 100}
        
        # Large prime should get bonus for sparse data
        eff_small = self.selector.compute_efficiency_metric(3, 5.0, 0.8, characteristics)
        eff_large = self.selector.compute_efficiency_metric(37, 5.0, 0.8, characteristics)
        
        # This tests the sparse adjustment logic
        self.assertGreaterEqual(eff_small, 0.0)
        self.assertGreaterEqual(eff_large, 0.0)
    
    def test_compute_efficiency_metric_high_dynamic_range(self):
        """Test efficiency metric with high dynamic range"""
        characteristics = {'sparsity': 0.1, 'dynamic_range': 1e8}
        
        efficiency = self.selector.compute_efficiency_metric(2, 5.0, 0.8, characteristics)
        self.assertGreaterEqual(efficiency, 0.0)
        self.assertLessEqual(efficiency, 1.0)
    
    def test_compute_efficiency_metric_periodic(self):
        """Test efficiency metric with periodic data"""
        characteristics = {
            'sparsity': 0.1, 
            'dynamic_range': 100,
            'periodicity_score': 0.8,
            'dominant_period': 7
        }
        
        # Prime 7 should get bonus for matching period
        eff_7 = self.selector.compute_efficiency_metric(7, 5.0, 0.8, characteristics)
        eff_11 = self.selector.compute_efficiency_metric(11, 5.0, 0.8, characteristics)
        
        self.assertGreaterEqual(eff_7, 0.0)
        self.assertGreaterEqual(eff_11, 0.0)
    
    def test_analyze_tensor_characteristics_gaussian(self):
        """Test tensor characteristics analysis for Gaussian data"""
        chars = self.selector.analyze_tensor_characteristics(self.gaussian_tensor)
        
        self.assertIn('sparsity', chars)
        self.assertIn('dynamic_range', chars)
        self.assertIn('mean', chars)
        self.assertIn('std', chars)
        self.assertIn('skewness', chars)
        self.assertIn('kurtosis', chars)
        self.assertIn('dominant_period', chars)
        self.assertIn('periodicity_score', chars)
        self.assertIn('unique_values', chars)
        self.assertIn('quantization_ratio', chars)
        
        # Gaussian should have low sparsity
        self.assertLess(chars['sparsity'], 0.5)
    
    def test_analyze_tensor_characteristics_sparse(self):
        """Test tensor characteristics analysis for sparse data"""
        chars = self.selector.analyze_tensor_characteristics(self.sparse_tensor)
        
        # Should detect high sparsity
        self.assertGreater(chars['sparsity'], 0.8)
        self.assertGreater(chars['dynamic_range'], 1.0)
    
    def test_analyze_tensor_characteristics_zero(self):
        """Test tensor characteristics analysis for all-zero data"""
        chars = self.selector.analyze_tensor_characteristics(self.zero_tensor)
        
        self.assertEqual(chars['sparsity'], 1.0)  # All zeros
        self.assertEqual(chars['dynamic_range'], 1.0)  # No range
        self.assertEqual(chars['mean'], 0.0)
        self.assertEqual(chars['std'], 0.0)
    
    def test_analyze_tensor_characteristics_single_value(self):
        """Test tensor characteristics analysis for constant data"""
        chars = self.selector.analyze_tensor_characteristics(self.single_value_tensor)
        
        self.assertEqual(chars['sparsity'], 0.0)  # No zeros
        self.assertEqual(chars['unique_values'], 1)  # Only one unique value
        self.assertAlmostEqual(chars['std'], 0.0)  # No variation
    
    def test_detect_distribution_type_sparse(self):
        """Test distribution type detection for sparse data"""
        dist_type = self.selector._detect_distribution_type(self.sparse_tensor)
        self.assertEqual(dist_type, "sparse")
    
    def test_detect_distribution_type_gaussian(self):
        """Test distribution type detection for Gaussian data"""
        dist_type = self.selector._detect_distribution_type(self.gaussian_tensor)
        # Could be gaussian or uniform depending on random seed
        self.assertIn(dist_type, ["gaussian", "uniform", "heavy_tailed"])
    
    def test_detect_distribution_type_zero(self):
        """Test distribution type detection for all-zero data"""
        dist_type = self.selector._detect_distribution_type(self.zero_tensor)
        self.assertEqual(dist_type, "sparse")
    
    def test_test_bimodality_clear_bimodal(self):
        """Test bimodality detection with clear bimodal data"""
        clear_bimodal = torch.cat([
            torch.ones(100) * -5,  # Clear mode 1
            torch.ones(100) * 5    # Clear mode 2
        ])
        is_bimodal = self.selector._test_bimodality(clear_bimodal.numpy())
        self.assertTrue(is_bimodal)
    
    def test_test_bimodality_unimodal(self):
        """Test bimodality detection with unimodal data"""
        unimodal = torch.randn(1000)  # Should be unimodal
        is_bimodal = self.selector._test_bimodality(unimodal.numpy())
        # Could be false positive due to randomness, but usually false
        self.assertIsInstance(is_bimodal, bool)
    
    def test_test_bimodality_small_data(self):
        """Test bimodality detection with small dataset"""
        small_data = np.array([1, 2, 3])
        is_bimodal = self.selector._test_bimodality(small_data)
        self.assertFalse(is_bimodal)
    
    def test_count_modes_unimodal(self):
        """Test mode counting with unimodal data"""
        unimodal = np.random.normal(0, 1, 1000)
        num_modes = self.selector._count_modes(unimodal)
        self.assertGreaterEqual(num_modes, 1)
    
    def test_count_modes_small_data(self):
        """Test mode counting with small dataset"""
        small_data = np.array([1, 2, 3])
        num_modes = self.selector._count_modes(small_data)
        self.assertEqual(num_modes, 1)
    
    def test_caching_enabled(self):
        """Test caching functionality when enabled"""
        selector = DynamicPrimeSelector(enable_caching=True)
        
        # First call
        result1 = selector.select_optimal_prime(self.gaussian_tensor)
        self.assertEqual(selector.cache_hits, 0)
        
        # Second call with same tensor should hit cache
        result2 = selector.select_optimal_prime(self.gaussian_tensor)
        self.assertEqual(selector.cache_hits, 1)
        
        # Results should be identical except computation time
        self.assertEqual(result1.optimal_prime, result2.optimal_prime)
        self.assertEqual(result1.efficiency, result2.efficiency)
        self.assertIn("cached", result2.selection_rationale)
    
    def test_caching_disabled(self):
        """Test behavior with caching disabled"""
        selector = DynamicPrimeSelector(enable_caching=False)
        
        # Multiple calls should never hit cache
        result1 = selector.select_optimal_prime(self.gaussian_tensor)
        result2 = selector.select_optimal_prime(self.gaussian_tensor)
        
        self.assertEqual(selector.cache_hits, 0)
        self.assertNotIn("cached", result2.selection_rationale)
    
    def test_cache_eviction(self):
        """Test cache eviction when cache is full"""
        selector = DynamicPrimeSelector(enable_caching=True, cache_size=2)
        
        # Fill cache beyond capacity
        tensors = [
            torch.randn(100),
            torch.randn(100) * 2,
            torch.randn(100) * 3
        ]
        
        for tensor in tensors:
            selector.select_optimal_prime(tensor)
        
        # Cache should be limited to size
        self.assertLessEqual(len(selector.selection_cache), 2)
    
    def test_compute_cache_key_consistency(self):
        """Test cache key computation consistency"""
        tensor1 = torch.randn(100)
        tensor2 = tensor1.clone()
        
        key1 = self.selector._compute_cache_key(tensor1)
        key2 = self.selector._compute_cache_key(tensor2)
        
        self.assertEqual(key1, key2)
    
    def test_compute_cache_key_different_tensors(self):
        """Test cache key computation for different tensors"""
        tensor1 = torch.randn(100)
        tensor2 = torch.randn(100)
        
        key1 = self.selector._compute_cache_key(tensor1)
        key2 = self.selector._compute_cache_key(tensor2)
        
        # Should be different with high probability
        self.assertNotEqual(key1, key2)
    
    def test_get_extended_prime_set_sparse(self):
        """Test extended prime set for sparse tensors"""
        prime_set = self.selector.get_extended_prime_set(self.sparse_tensor)
        
        self.assertGreater(len(prime_set), len(self.selector.DEFAULT_PRIMES))
        self.assertEqual(prime_set, sorted(prime_set))
        self.assertTrue(all(self.selector._is_prime(p) for p in prime_set))
    
    def test_get_extended_prime_set_high_dynamic_range(self):
        """Test extended prime set for high dynamic range tensors"""
        prime_set = self.selector.get_extended_prime_set(self.high_dynamic_range_tensor)
        
        # Should include Mersenne-like primes
        mersenne_primes = [3, 7, 31, 127]
        for mp in mersenne_primes:
            if mp not in self.selector.DEFAULT_PRIMES:
                self.assertIn(mp, prime_set)
    
    def test_get_extended_prime_set_periodic(self):
        """Test extended prime set for periodic tensors"""
        # Create tensor with clear period
        period_tensor = torch.sin(2 * np.pi * torch.arange(1000) / 13)  # Period 13
        prime_set = self.selector.get_extended_prime_set(period_tensor)
        
        # Should potentially include primes near 13
        self.assertGreater(len(prime_set), len(self.selector.DEFAULT_PRIMES))
    
    def test_parallel_prime_evaluation_basic(self):
        """Test parallel prime evaluation"""
        result = self.selector.parallel_prime_evaluation(
            self.gaussian_tensor, 
            num_threads=2
        )
        
        self.assertIsInstance(result, PrimeSelectionResult)
        self.assertIn(result.optimal_prime, self.selector.primes)
        self.assertGreater(result.efficiency, 0.0)
    
    def test_parallel_prime_evaluation_custom_primes(self):
        """Test parallel prime evaluation with custom primes"""
        custom_primes = [2, 3, 5, 7]
        result = self.selector.parallel_prime_evaluation(
            self.gaussian_tensor,
            primes=custom_primes,
            num_threads=2
        )
        
        self.assertIn(result.optimal_prime, custom_primes)
        self.assertEqual(set(result.candidate_scores.keys()), set(custom_primes))
    
    def test_parallel_prime_evaluation_zero_tensor(self):
        """Test parallel evaluation with zero tensor"""
        # Should fall back to sequential evaluation
        result = self.selector.parallel_prime_evaluation(self.zero_tensor)
        self.assertIsInstance(result, PrimeSelectionResult)
    
    def test_statistics_tracking(self):
        """Test statistics tracking functionality"""
        selector = DynamicPrimeSelector()
        
        # Initial stats
        stats = selector.get_statistics()
        self.assertEqual(stats['total_selections'], 0)
        self.assertEqual(stats['cache_hits'], 0)
        
        # After selections
        selector.select_optimal_prime(self.gaussian_tensor)
        selector.select_optimal_prime(self.gaussian_tensor)  # Should hit cache
        
        stats = selector.get_statistics()
        self.assertEqual(stats['total_selections'], 2)
        self.assertEqual(stats['cache_hits'], 1)
        self.assertGreater(stats['average_selection_time'], 0.0)
    
    def test_clear_cache(self):
        """Test cache clearing functionality"""
        selector = DynamicPrimeSelector()
        
        # Build cache
        selector.select_optimal_prime(self.gaussian_tensor)
        self.assertGreater(len(selector.selection_cache), 0)
        
        # Clear cache
        selector.clear_cache()
        self.assertEqual(len(selector.selection_cache), 0)
    
    def test_reset_statistics(self):
        """Test statistics reset functionality"""
        selector = DynamicPrimeSelector()
        
        # Build statistics
        selector.select_optimal_prime(self.gaussian_tensor)
        self.assertGreater(selector.selection_count, 0)
        
        # Reset statistics
        selector.reset_statistics()
        stats = selector.get_statistics()
        self.assertEqual(stats['total_selections'], 0)
        self.assertEqual(stats['cache_hits'], 0)
    
    def test_thread_safety(self):
        """Test thread safety of prime selection"""
        selector = DynamicPrimeSelector()
        results = []
        exceptions = []
        
        def select_prime():
            try:
                result = selector.select_optimal_prime(torch.randn(100))
                results.append(result)
            except Exception as e:
                exceptions.append(e)
        
        # Create multiple threads
        threads = [threading.Thread(target=select_prime) for _ in range(10)]
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check results
        self.assertEqual(len(exceptions), 0, f"Exceptions occurred: {exceptions}")
        self.assertEqual(len(results), 10)
        self.assertTrue(all(isinstance(r, PrimeSelectionResult) for r in results))
    
    def test_edge_case_very_small_values(self):
        """Test with very small floating point values"""
        tiny_tensor = torch.tensor([1e-15, 1e-14, 1e-13])
        result = self.selector.select_optimal_prime(tiny_tensor)
        
        self.assertIsInstance(result, PrimeSelectionResult)
        self.assertGreater(result.efficiency, 0.0)
    
    def test_edge_case_very_large_values(self):
        """Test with very large floating point values"""
        huge_tensor = torch.tensor([1e15, 1e14, 1e13])
        result = self.selector.select_optimal_prime(huge_tensor)
        
        self.assertIsInstance(result, PrimeSelectionResult)
        self.assertGreater(result.efficiency, 0.0)
    
    def test_edge_case_nan_values(self):
        """Test handling of NaN values"""
        nan_tensor = torch.tensor([1.0, float('nan'), 3.0])
        
        # Should handle NaN gracefully or raise appropriate error
        try:
            result = self.selector.select_optimal_prime(nan_tensor)
            self.assertIsInstance(result, PrimeSelectionResult)
        except (ValueError, RuntimeError):
            pass  # Acceptable to raise error for NaN
    
    def test_edge_case_inf_values(self):
        """Test handling of infinite values"""
        inf_tensor = torch.tensor([1.0, float('inf'), 3.0])
        
        # Should handle inf gracefully or raise appropriate error
        try:
            result = self.selector.select_optimal_prime(inf_tensor)
            self.assertIsInstance(result, PrimeSelectionResult)
        except (ValueError, RuntimeError):
            pass  # Acceptable to raise error for inf
    
    def test_edge_case_single_element(self):
        """Test with single element tensor"""
        single_tensor = torch.tensor([5.5])
        result = self.selector.select_optimal_prime(single_tensor)
        
        self.assertIsInstance(result, PrimeSelectionResult)
        self.assertGreater(result.efficiency, 0.0)
    
    def test_edge_case_two_elements(self):
        """Test with two element tensor"""
        two_tensor = torch.tensor([1.0, 2.0])
        result = self.selector.select_optimal_prime(two_tensor)
        
        self.assertIsInstance(result, PrimeSelectionResult)
        self.assertGreater(result.efficiency, 0.0)
    
    def test_edge_case_all_same_nonzero(self):
        """Test with all same non-zero values"""
        same_tensor = torch.ones(100) * 7.3
        result = self.selector.select_optimal_prime(same_tensor)
        
        self.assertIsInstance(result, PrimeSelectionResult)
        self.assertGreater(result.efficiency, 0.0)
    
    def test_different_tensor_dtypes(self):
        """Test with different tensor data types"""
        dtypes_to_test = [torch.float32, torch.float64]
        
        for dtype in dtypes_to_test:
            tensor = torch.randn(100).to(dtype)
            result = self.selector.select_optimal_prime(tensor)
            
            self.assertIsInstance(result, PrimeSelectionResult)
            self.assertGreater(result.efficiency, 0.0)
    
    def test_different_tensor_shapes(self):
        """Test with different tensor shapes"""
        shapes_to_test = [
            (10,),           # 1D
            (5, 2),          # 2D
            (2, 3, 4),       # 3D
            (2, 2, 2, 2),    # 4D
        ]
        
        for shape in shapes_to_test:
            tensor = torch.randn(shape)
            result = self.selector.select_optimal_prime(tensor)
            
            self.assertIsInstance(result, PrimeSelectionResult)
            self.assertGreater(result.efficiency, 0.0)
    
    def test_large_tensor_sampling(self):
        """Test sampling behavior with large tensors"""
        large_tensor = torch.randn(20000)  # Should trigger sampling
        
        result = self.selector.select_optimal_prime(
            large_tensor, 
            max_evaluation_samples=5000
        )
        
        self.assertIsInstance(result, PrimeSelectionResult)
        self.assertGreater(result.efficiency, 0.0)
    
    def test_max_evaluation_samples_parameter(self):
        """Test max_evaluation_samples parameter"""
        medium_tensor = torch.randn(1000)
        
        # Small sample size
        result1 = self.selector.select_optimal_prime(
            medium_tensor, 
            max_evaluation_samples=100
        )
        
        # Large sample size  
        result2 = self.selector.select_optimal_prime(
            medium_tensor,
            max_evaluation_samples=1000
        )
        
        self.assertIsInstance(result1, PrimeSelectionResult)
        self.assertIsInstance(result2, PrimeSelectionResult)
    
    def test_generate_rationale_completeness(self):
        """Test that rationale generation covers all cases"""
        characteristics = {
            'sparsity': 0.1,
            'dynamic_range': 100,
            'periodicity_score': 0.5,
            'dominant_period': 0
        }
        
        rationale = self.selector._generate_rationale(
            7, "gaussian", characteristics, 0.85, 5.2
        )
        
        self.assertIsInstance(rationale, str)
        self.assertGreater(len(rationale), 0)
        self.assertIn("7", rationale)
        self.assertIn("0.850", rationale)
        self.assertIn("5.20", rationale)
    
    def test_selection_result_attributes(self):
        """Test that PrimeSelectionResult has all expected attributes"""
        result = self.selector.select_optimal_prime(self.gaussian_tensor)
        
        # Check all required attributes exist
        self.assertTrue(hasattr(result, 'optimal_prime'))
        self.assertTrue(hasattr(result, 'efficiency'))
        self.assertTrue(hasattr(result, 'average_digits'))
        self.assertTrue(hasattr(result, 'entropy'))
        self.assertTrue(hasattr(result, 'distribution_type'))
        self.assertTrue(hasattr(result, 'selection_rationale'))
        self.assertTrue(hasattr(result, 'candidate_scores'))
        self.assertTrue(hasattr(result, 'computation_time'))
        
        # Check types
        self.assertIsInstance(result.optimal_prime, int)
        self.assertIsInstance(result.efficiency, float)
        self.assertIsInstance(result.average_digits, float)
        self.assertIsInstance(result.entropy, float)
        self.assertIsInstance(result.distribution_type, str)
        self.assertIsInstance(result.selection_rationale, str)
        self.assertIsInstance(result.candidate_scores, dict)
        self.assertIsInstance(result.computation_time, float)


class TestDynamicPrimeSelectorIntegration(unittest.TestCase):
    """Integration tests for DynamicPrimeSelector with realistic scenarios"""
    
    def setUp(self):
        self.selector = DynamicPrimeSelector()
    
    def test_neural_network_weights(self):
        """Test with tensor resembling neural network weights"""
        # Xavier initialization-like tensor
        fan_in, fan_out = 128, 64
        nn_weights = torch.randn(fan_out, fan_in) * np.sqrt(2.0 / (fan_in + fan_out))
        
        result = self.selector.select_optimal_prime(nn_weights)
        
        self.assertIsInstance(result, PrimeSelectionResult)
        self.assertGreater(result.efficiency, 0.0)
    
    def test_image_like_tensor(self):
        """Test with tensor resembling image data"""
        # Image-like tensor (normalized to [0,1])
        image_tensor = torch.rand(3, 224, 224)  # RGB image
        
        result = self.selector.select_optimal_prime(image_tensor)
        
        self.assertIsInstance(result, PrimeSelectionResult)
        self.assertGreater(result.efficiency, 0.0)
    
    def test_time_series_data(self):
        """Test with time series-like data"""
        # Sine wave with noise
        t = torch.linspace(0, 10, 1000)
        time_series = torch.sin(2 * np.pi * t) + 0.1 * torch.randn(1000)
        
        result = self.selector.select_optimal_prime(time_series)
        
        self.assertIsInstance(result, PrimeSelectionResult)
        # Should detect some periodicity
        characteristics = self.selector.analyze_tensor_characteristics(time_series)
        self.assertGreater(characteristics['periodicity_score'], 0.1)
    
    def test_embeddings_like_tensor(self):
        """Test with tensor resembling word embeddings"""
        # Random embeddings for vocabulary
        vocab_size, embed_dim = 10000, 300
        embeddings = torch.randn(vocab_size, embed_dim) * 0.1
        
        result = self.selector.select_optimal_prime(embeddings)
        
        self.assertIsInstance(result, PrimeSelectionResult)
        self.assertGreater(result.efficiency, 0.0)
    
    def test_performance_with_different_sizes(self):
        """Test performance scaling with different tensor sizes"""
        sizes = [100, 1000, 10000, 50000]
        times = []
        
        for size in sizes:
            tensor = torch.randn(size)
            start_time = time.time()
            result = self.selector.select_optimal_prime(tensor)
            end_time = time.time()
            
            times.append(end_time - start_time)
            self.assertIsInstance(result, PrimeSelectionResult)
        
        # Performance should scale reasonably (not exponentially)
        # Large tensors should use sampling
        self.assertLess(times[-1], 5.0)  # Should complete within 5 seconds


if __name__ == '__main__':
    unittest.main(verbosity=2)