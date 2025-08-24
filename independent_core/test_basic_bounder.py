"""
Comprehensive Unit Tests for BasicGradientBounder
Tests gradient bounding, explosion/vanishing handling, and configuration
"""

import unittest
import torch
import numpy as np
from typing import Dict, Any
import sys
import os

# Add path to gac_system
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gac_system.basic_bounder import (
    BasicGradientBounder,
    BoundingResult,
    BasicGradientBoundingError
)


class TestBasicGradientBounderInit(unittest.TestCase):
    """Test BasicGradientBounder initialization"""
    
    def test_default_initialization(self):
        """Test initialization with default configuration"""
        bounder = BasicGradientBounder()
        
        self.assertEqual(bounder.max_norm, 1.0)
        self.assertEqual(bounder.min_norm, 1e-8)
        self.assertEqual(bounder.clip_value, 10.0)
        self.assertTrue(bounder.enable_adaptive_scaling)
        self.assertEqual(bounder.scaling_factor, 1.0)
        self.assertEqual(bounder.norm_type, 2)
        self.assertIsNone(bounder.norm_dim)
        self.assertEqual(bounder.explosion_threshold, 100.0)
        self.assertEqual(bounder.vanishing_threshold, 1e-10)
    
    def test_custom_initialization(self):
        """Test initialization with custom configuration"""
        config = {
            'max_norm': 5.0,
            'min_norm': 1e-6,
            'clip_value': 20.0,
            'enable_adaptive_scaling': False,
            'scaling_factor': 0.5,
            'norm_type': 1,
            'explosion_threshold': 50.0,
            'vanishing_threshold': 1e-8
        }
        
        bounder = BasicGradientBounder(config)
        
        self.assertEqual(bounder.max_norm, 5.0)
        self.assertEqual(bounder.min_norm, 1e-6)
        self.assertEqual(bounder.clip_value, 20.0)
        self.assertFalse(bounder.enable_adaptive_scaling)
        self.assertEqual(bounder.scaling_factor, 0.5)
        self.assertEqual(bounder.norm_type, 1)
        self.assertEqual(bounder.explosion_threshold, 50.0)
        self.assertEqual(bounder.vanishing_threshold, 1e-8)
    
    def test_statistics_initialization(self):
        """Test that statistics are properly initialized"""
        bounder = BasicGradientBounder()
        
        self.assertEqual(bounder.total_bounds, 0)
        self.assertEqual(bounder.norm_clips, 0)
        self.assertEqual(bounder.value_clips, 0)
        self.assertEqual(bounder.scaling_applications, 0)
        self.assertEqual(bounder.explosion_detections, 0)
        self.assertEqual(bounder.vanishing_detections, 0)
        self.assertEqual(bounder.last_bound_time, 0.0)
        self.assertEqual(bounder.total_bound_time, 0.0)


class TestBoundGradients(unittest.TestCase):
    """Test gradient bounding functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.bounder = BasicGradientBounder()
    
    def test_bound_normal_gradients(self):
        """Test bounding normal gradients"""
        gradients = torch.randn(10, 10) * 0.1
        result = self.bounder.bound_gradients(gradients)
        
        self.assertIsInstance(result, BoundingResult)
        self.assertIsInstance(result.bounded_gradients, torch.Tensor)
        self.assertEqual(result.bounded_gradients.shape, gradients.shape)
        self.assertEqual(result.bounding_type, "basic")
        self.assertIsInstance(result.applied_factor, float)
        self.assertIsInstance(result.metadata, dict)
    
    def test_bound_none_gradients(self):
        """Test that None gradients raise error"""
        with self.assertRaises(BasicGradientBoundingError) as context:
            self.bounder.bound_gradients(None)
        self.assertIn("cannot be None", str(context.exception))
    
    def test_bound_non_tensor_gradients(self):
        """Test that non-tensor gradients raise TypeError"""
        with self.assertRaises(TypeError) as context:
            self.bounder.bound_gradients([1, 2, 3])
        self.assertIn("must be torch.Tensor", str(context.exception))
    
    def test_bound_empty_gradients(self):
        """Test that empty gradients raise error"""
        empty_tensor = torch.tensor([])
        with self.assertRaises(BasicGradientBoundingError) as context:
            self.bounder.bound_gradients(empty_tensor)
        self.assertIn("cannot be empty", str(context.exception))
    
    def test_bound_with_context(self):
        """Test bounding with context information"""
        gradients = torch.randn(5, 5)
        context = {
            'learning_rate': 0.1,
            'iteration': 50,
            'loss': 2.5
        }
        
        result = self.bounder.bound_gradients(gradients, context)
        
        self.assertIsInstance(result, BoundingResult)
        self.assertIn('context', result.metadata)
        self.assertEqual(result.metadata['context'], context)
    
    def test_gradient_not_modified_in_place(self):
        """Test that original gradients are not modified"""
        gradients = torch.randn(5, 5)
        original = gradients.clone()
        
        result = self.bounder.bound_gradients(gradients)
        
        self.assertTrue(torch.allclose(gradients, original))
        self.assertIsNot(result.bounded_gradients, gradients)


class TestGradientExplosion(unittest.TestCase):
    """Test gradient explosion handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.bounder = BasicGradientBounder({'explosion_threshold': 10.0})
    
    def test_detect_gradient_explosion(self):
        """Test detection of gradient explosion"""
        # Create exploding gradients
        gradients = torch.randn(5, 5) * 100
        
        result = self.bounder.bound_gradients(gradients)
        
        self.assertIn('explosion_handling', result.metadata['applied_operations'])
        self.assertIn('explosion', result.metadata['bounding_details'])
        self.assertTrue(result.metadata['bounding_details']['explosion']['detected'])
        self.assertEqual(self.bounder.explosion_detections, 1)
    
    def test_handle_gradient_explosion(self):
        """Test that exploding gradients are properly bounded"""
        gradients = torch.ones(5, 5) * 1000
        
        result = self.bounder.bound_gradients(gradients)
        
        # Check that gradients are clipped
        max_value = torch.max(torch.abs(result.bounded_gradients)).item()
        self.assertLess(max_value, self.bounder.explosion_threshold)
        
        # Check that norm is reduced
        original_norm = torch.norm(gradients).item()
        bounded_norm = torch.norm(result.bounded_gradients).item()
        self.assertLess(bounded_norm, original_norm)
    
    def test_no_explosion_for_normal_gradients(self):
        """Test that normal gradients don't trigger explosion handling"""
        gradients = torch.randn(5, 5) * 0.1
        
        result = self.bounder.bound_gradients(gradients)
        
        self.assertNotIn('explosion_handling', result.metadata['applied_operations'])
        self.assertEqual(self.bounder.explosion_detections, 0)


class TestVanishingGradients(unittest.TestCase):
    """Test vanishing gradient handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.bounder = BasicGradientBounder({
            'vanishing_threshold': 1e-6,
            'min_norm': 1e-4
        })
    
    def test_detect_vanishing_gradients(self):
        """Test detection of vanishing gradients"""
        # Create vanishing gradients
        gradients = torch.randn(5, 5) * 1e-8
        
        result = self.bounder.bound_gradients(gradients)
        
        self.assertIn('vanishing_handling', result.metadata['applied_operations'])
        self.assertIn('vanishing', result.metadata['bounding_details'])
        self.assertTrue(result.metadata['bounding_details']['vanishing']['detected'])
        self.assertEqual(self.bounder.vanishing_detections, 1)
    
    def test_handle_vanishing_gradients(self):
        """Test that vanishing gradients are properly scaled"""
        gradients = torch.ones(5, 5) * 1e-10
        
        result = self.bounder.bound_gradients(gradients)
        
        # Check that gradients are scaled up
        original_norm = torch.norm(gradients).item()
        bounded_norm = torch.norm(result.bounded_gradients).item()
        self.assertGreater(bounded_norm, original_norm)
        self.assertGreaterEqual(bounded_norm, self.bounder.min_norm * 0.9)  # Allow small tolerance
    
    def test_no_vanishing_for_normal_gradients(self):
        """Test that normal gradients don't trigger vanishing handling"""
        gradients = torch.randn(5, 5) * 0.1
        
        result = self.bounder.bound_gradients(gradients)
        
        self.assertNotIn('vanishing_handling', result.metadata['applied_operations'])
        self.assertEqual(self.bounder.vanishing_detections, 0)


class TestNormClipping(unittest.TestCase):
    """Test gradient norm clipping"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.bounder = BasicGradientBounder({'max_norm': 1.0})
    
    def test_clip_large_norm(self):
        """Test clipping of gradients with large norm"""
        gradients = torch.randn(5, 5) * 10
        
        result = self.bounder.bound_gradients(gradients)
        
        # Check that norm is clipped
        bounded_norm = torch.norm(result.bounded_gradients).item()
        self.assertLessEqual(bounded_norm, self.bounder.max_norm * 1.01)  # Small tolerance
        
        self.assertIn('norm_clipping', result.metadata['applied_operations'])
        self.assertEqual(self.bounder.norm_clips, 1)
    
    def test_no_clip_small_norm(self):
        """Test that small norm gradients are not clipped"""
        gradients = torch.randn(5, 5) * 0.1
        
        result = self.bounder.bound_gradients(gradients)
        
        # Check that gradients are essentially unchanged
        original_norm = torch.norm(gradients).item()
        bounded_norm = torch.norm(result.bounded_gradients).item()
        
        if original_norm < self.bounder.max_norm:
            self.assertAlmostEqual(bounded_norm, original_norm, places=5)
    
    def test_per_dimension_norm_clipping(self):
        """Test per-dimension norm clipping"""
        bounder = BasicGradientBounder({
            'max_norm': 1.0,
            'norm_dim': 1
        })
        
        gradients = torch.randn(5, 10) * 10
        result = bounder.bound_gradients(gradients)
        
        # Check that each row norm is bounded
        for i in range(result.bounded_gradients.shape[0]):
            row_norm = torch.norm(result.bounded_gradients[i]).item()
            self.assertLessEqual(row_norm, bounder.max_norm * 1.1)  # Small tolerance


class TestValueClipping(unittest.TestCase):
    """Test gradient value clipping"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.bounder = BasicGradientBounder({'clip_value': 5.0})
    
    def test_clip_large_values(self):
        """Test clipping of large gradient values"""
        gradients = torch.tensor([[10.0, -15.0], [20.0, -25.0]])
        
        result = self.bounder.bound_gradients(gradients)
        
        # Check all values are within bounds
        max_val = torch.max(torch.abs(result.bounded_gradients)).item()
        self.assertLessEqual(max_val, self.bounder.clip_value)
        
        self.assertIn('value_clipping', result.metadata['applied_operations'])
        self.assertEqual(self.bounder.value_clips, 1)
    
    def test_no_clip_small_values(self):
        """Test that small values are not clipped"""
        gradients = torch.randn(5, 5) * 0.5
        
        result = self.bounder.bound_gradients(gradients)
        
        # Check that gradients are unchanged if all within bounds
        if torch.max(torch.abs(gradients)).item() < self.bounder.clip_value:
            self.assertTrue(torch.allclose(result.bounded_gradients, gradients))


class TestAdaptiveScaling(unittest.TestCase):
    """Test adaptive scaling functionality"""
    
    def test_scaling_with_high_learning_rate(self):
        """Test scaling adjustment for high learning rate"""
        bounder = BasicGradientBounder({'enable_adaptive_scaling': True})
        gradients = torch.randn(5, 5)
        context = {'learning_rate': 0.1}
        
        result = bounder.bound_gradients(gradients, context)
        
        if 'adaptive_scaling' in result.metadata['applied_operations']:
            scaling_details = result.metadata['bounding_details']['adaptive_scaling']
            self.assertLessEqual(scaling_details['scale_factor'], 1.0)
            self.assertEqual(scaling_details['scaling_reason'], 'high_learning_rate')
    
    def test_scaling_with_iteration_warmup(self):
        """Test scaling adjustment during warmup"""
        bounder = BasicGradientBounder({'enable_adaptive_scaling': True})
        gradients = torch.randn(5, 5)
        context = {'iteration': 50}
        
        result = bounder.bound_gradients(gradients, context)
        
        if 'adaptive_scaling' in result.metadata['applied_operations']:
            scaling_details = result.metadata['bounding_details']['adaptive_scaling']
            self.assertIn('iteration_warmup', scaling_details['scaling_reason'])
    
    def test_scaling_with_high_loss(self):
        """Test scaling adjustment for high loss"""
        bounder = BasicGradientBounder({'enable_adaptive_scaling': True})
        gradients = torch.randn(5, 5)
        context = {'loss': 100.0}
        
        result = bounder.bound_gradients(gradients, context)
        
        if 'adaptive_scaling' in result.metadata['applied_operations']:
            scaling_details = result.metadata['bounding_details']['adaptive_scaling']
            self.assertLessEqual(scaling_details['scale_factor'], 1.0)
            self.assertEqual(scaling_details['scaling_reason'], 'high_loss')
    
    def test_no_scaling_when_disabled(self):
        """Test that scaling is not applied when disabled"""
        bounder = BasicGradientBounder({'enable_adaptive_scaling': False})
        gradients = torch.randn(5, 5)
        context = {'learning_rate': 0.1, 'iteration': 50, 'loss': 100.0}
        
        result = bounder.bound_gradients(gradients, context)
        
        self.assertNotIn('adaptive_scaling', result.metadata['applied_operations'])
        self.assertEqual(bounder.scaling_applications, 0)


class TestStatistics(unittest.TestCase):
    """Test statistics tracking"""
    
    def test_statistics_tracking(self):
        """Test that statistics are properly tracked"""
        bounder = BasicGradientBounder({'max_norm': 0.1})
        
        # Perform multiple operations
        for _ in range(5):
            gradients = torch.randn(5, 5) * 10  # Will trigger norm clipping
            bounder.bound_gradients(gradients)
        
        stats = bounder.get_bounding_statistics()
        
        self.assertEqual(stats['operation_counts']['total_bounds'], 5)
        self.assertGreater(stats['operation_counts']['norm_clips'], 0)
        self.assertGreater(stats['performance']['total_time'], 0)
        self.assertGreater(stats['performance']['average_time'], 0)
    
    def test_reset_statistics(self):
        """Test resetting statistics"""
        bounder = BasicGradientBounder()
        
        # Perform some operations
        bounder.bound_gradients(torch.randn(5, 5))
        bounder.bound_gradients(torch.randn(5, 5))
        
        # Reset statistics
        bounder.reset_statistics()
        
        self.assertEqual(bounder.total_bounds, 0)
        self.assertEqual(bounder.norm_clips, 0)
        self.assertEqual(bounder.value_clips, 0)
        self.assertEqual(bounder.total_bound_time, 0.0)
    
    def test_empty_statistics(self):
        """Test statistics when no operations performed"""
        bounder = BasicGradientBounder()
        stats = bounder.get_bounding_statistics()
        
        self.assertEqual(stats['status'], 'no_operations')


class TestConfigurationUpdate(unittest.TestCase):
    """Test configuration update functionality"""
    
    def test_update_valid_config(self):
        """Test updating valid configuration parameters"""
        bounder = BasicGradientBounder()
        
        new_config = {
            'max_norm': 2.0,
            'clip_value': 15.0,
            'enable_adaptive_scaling': False
        }
        
        bounder.update_config(new_config)
        
        self.assertEqual(bounder.max_norm, 2.0)
        self.assertEqual(bounder.clip_value, 15.0)
        self.assertFalse(bounder.enable_adaptive_scaling)
    
    def test_update_invalid_config(self):
        """Test that invalid config parameters are ignored"""
        bounder = BasicGradientBounder()
        original_max_norm = bounder.max_norm
        
        new_config = {
            'invalid_param': 123,
            'max_norm': 2.0
        }
        
        bounder.update_config(new_config)
        
        # Valid param should be updated
        self.assertEqual(bounder.max_norm, 2.0)
        # Invalid param should not create new attribute
        self.assertFalse(hasattr(bounder, 'invalid_param'))


class TestHealthCheck(unittest.TestCase):
    """Test gradient health checking"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.bounder = BasicGradientBounder()
    
    def test_healthy_gradients(self):
        """Test that normal gradients are considered healthy"""
        gradients = torch.randn(5, 5) * 0.1
        self.assertTrue(self.bounder.is_gradient_healthy(gradients))
    
    def test_unhealthy_none_gradients(self):
        """Test that None gradients are unhealthy"""
        self.assertFalse(self.bounder.is_gradient_healthy(None))
    
    def test_unhealthy_empty_gradients(self):
        """Test that empty gradients are unhealthy"""
        empty_tensor = torch.tensor([])
        self.assertFalse(self.bounder.is_gradient_healthy(empty_tensor))
    
    def test_unhealthy_nan_gradients(self):
        """Test that NaN gradients are unhealthy"""
        gradients = torch.tensor([[float('nan'), 1.0], [2.0, 3.0]])
        self.assertFalse(self.bounder.is_gradient_healthy(gradients))
    
    def test_unhealthy_inf_gradients(self):
        """Test that infinite gradients are unhealthy"""
        gradients = torch.tensor([[float('inf'), 1.0], [2.0, 3.0]])
        self.assertFalse(self.bounder.is_gradient_healthy(gradients))
    
    def test_unhealthy_exploding_gradients(self):
        """Test that exploding gradients are unhealthy"""
        gradients = torch.ones(5, 5) * 1000
        self.assertFalse(self.bounder.is_gradient_healthy(gradients))
    
    def test_unhealthy_vanishing_gradients(self):
        """Test that vanishing gradients are unhealthy"""
        gradients = torch.ones(5, 5) * 1e-12
        self.assertFalse(self.bounder.is_gradient_healthy(gradients))


class TestRecommendedBounds(unittest.TestCase):
    """Test recommended bounds calculation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.bounder = BasicGradientBounder()
    
    def test_recommended_bounds_normal_gradients(self):
        """Test recommended bounds for normal gradients"""
        gradients = torch.randn(5, 5)
        recommendations = self.bounder.get_recommended_bounds(gradients)
        
        self.assertIn('max_norm', recommendations)
        self.assertIn('clip_value', recommendations)
        self.assertIn('scaling_factor', recommendations)
        self.assertIn('current_norm', recommendations)
        self.assertIn('current_max', recommendations)
        self.assertIn('current_std', recommendations)
        
        # Check that recommendations are reasonable
        self.assertGreater(recommendations['max_norm'], 0)
        self.assertGreater(recommendations['clip_value'], 0)
        self.assertGreater(recommendations['scaling_factor'], 0)
    
    def test_recommended_bounds_empty_gradients(self):
        """Test recommended bounds for empty gradients"""
        empty_tensor = torch.tensor([])
        recommendations = self.bounder.get_recommended_bounds(empty_tensor)
        
        self.assertEqual(recommendations, {})
    
    def test_recommended_bounds_large_gradients(self):
        """Test recommended bounds for large gradients"""
        gradients = torch.randn(5, 5) * 100
        recommendations = self.bounder.get_recommended_bounds(gradients)
        
        # Should recommend scaling down
        self.assertLessEqual(recommendations['scaling_factor'], 1.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios"""
    
    def test_single_element_gradient(self):
        """Test bounding single element gradient"""
        bounder = BasicGradientBounder()
        gradients = torch.tensor([5.0])
        
        result = bounder.bound_gradients(gradients)
        
        self.assertEqual(result.bounded_gradients.shape, gradients.shape)
        self.assertIsInstance(result.applied_factor, float)
    
    def test_zero_gradients(self):
        """Test bounding zero gradients"""
        bounder = BasicGradientBounder()
        gradients = torch.zeros(5, 5)
        
        result = bounder.bound_gradients(gradients)
        
        # Zero gradients might trigger vanishing gradient handling
        self.assertTrue(torch.allclose(result.bounded_gradients, gradients) or
                       torch.norm(result.bounded_gradients).item() >= bounder.min_norm * 0.9)
    
    def test_very_high_dimensional_gradients(self):
        """Test bounding high-dimensional gradients"""
        bounder = BasicGradientBounder()
        gradients = torch.randn(100, 100, 10)
        
        result = bounder.bound_gradients(gradients)
        
        self.assertEqual(result.bounded_gradients.shape, gradients.shape)
        self.assertIsInstance(result, BoundingResult)
    
    def test_different_norm_types(self):
        """Test different norm types (L1, L2, Linf)"""
        for norm_type in [1, 2, float('inf')]:
            bounder = BasicGradientBounder({'norm_type': norm_type})
            gradients = torch.randn(5, 5)
            
            result = bounder.bound_gradients(gradients)
            
            self.assertIsInstance(result, BoundingResult)
            self.assertEqual(result.bounded_gradients.shape, gradients.shape)


if __name__ == "__main__":
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBasicGradientBounderInit))
    suite.addTests(loader.loadTestsFromTestCase(TestBoundGradients))
    suite.addTests(loader.loadTestsFromTestCase(TestGradientExplosion))
    suite.addTests(loader.loadTestsFromTestCase(TestVanishingGradients))
    suite.addTests(loader.loadTestsFromTestCase(TestNormClipping))
    suite.addTests(loader.loadTestsFromTestCase(TestValueClipping))
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveScaling))
    suite.addTests(loader.loadTestsFromTestCase(TestStatistics))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigurationUpdate))
    suite.addTests(loader.loadTestsFromTestCase(TestHealthCheck))
    suite.addTests(loader.loadTestsFromTestCase(TestRecommendedBounds))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("BASICBOUNDER TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)