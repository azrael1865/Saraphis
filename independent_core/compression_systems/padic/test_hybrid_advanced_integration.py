"""
Comprehensive Tests for Hybrid Advanced Features Integration
Tests for HybridAdvancedIntegration, HybridHenselLifting, HybridHierarchicalClustering, and HybridOptimizationManager
NO FALLBACKS - HARD FAILURES ONLY
"""

import unittest
import torch
import time
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

# Import advanced integration components
from .hybrid_advanced_integration import (
    HybridAdvancedIntegration, AdvancedIntegrationConfig, 
    AdvancedFeatureType, IntegrationStatus
)
from .hybrid_hensel_lifting import (
    HybridHenselLifting, HybridLiftingResult, HybridLiftingStats
)
from .hybrid_clustering import (
    HybridHierarchicalClustering, HybridClusteringResult, 
    HybridClusterNode, HybridClusteringStats
)
from .hybrid_optimization import (
    HybridOptimizationManager, HybridOptimizerConfig, 
    HybridOptimizerState, HybridOptimizationStats
)

# Import existing advanced features for integration testing
from .padic_advanced import (
    HenselLiftingProcessor, HenselLiftingConfig,
    HierarchicalClusteringManager, ClusteringConfig,
    PadicOptimizationManager
)

# Import hybrid structures
from .hybrid_padic_structures import HybridPadicWeight
from ...performance_optimizer import PerformanceOptimizer


class TestAdvancedIntegrationConfig(unittest.TestCase):
    """Test cases for AdvancedIntegrationConfig"""
    
    def test_valid_config_creation(self):
        """Test valid configuration creation"""
        config = AdvancedIntegrationConfig(
            enable_hensel_lifting=True,
            enable_hierarchical_clustering=True,
            enable_padic_optimization=True,
            hensel_lifting_threshold=500,
            gpu_memory_limit_mb=2048
        )
        
        self.assertTrue(config.enable_hensel_lifting)
        self.assertTrue(config.enable_hierarchical_clustering)
        self.assertTrue(config.enable_padic_optimization)
        self.assertEqual(config.hensel_lifting_threshold, 500)
        self.assertEqual(config.gpu_memory_limit_mb, 2048)
    
    def test_invalid_config_values(self):
        """Test invalid configuration values"""
        # Test invalid thresholds
        with self.assertRaises(ValueError):
            AdvancedIntegrationConfig(hensel_lifting_threshold=-1)
        
        with self.assertRaises(ValueError):
            AdvancedIntegrationConfig(clustering_threshold=0)
        
        with self.assertRaises(ValueError):
            AdvancedIntegrationConfig(gpu_memory_limit_mb=-100)
        
        with self.assertRaises(ValueError):
            AdvancedIntegrationConfig(max_concurrent_operations=0)
    
    def test_default_config_values(self):
        """Test default configuration values"""
        config = AdvancedIntegrationConfig()
        
        self.assertTrue(config.enable_hensel_lifting)
        self.assertTrue(config.enable_hierarchical_clustering)
        self.assertTrue(config.enable_padic_optimization)
        self.assertGreater(config.hensel_lifting_threshold, 0)
        self.assertGreater(config.gpu_memory_limit_mb, 0)


class TestHybridAdvancedIntegration(unittest.TestCase):
    """Test cases for HybridAdvancedIntegration"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = AdvancedIntegrationConfig(
            enable_hensel_lifting=True,
            enable_hierarchical_clustering=True,
            enable_padic_optimization=True
        )
        self.integration = HybridAdvancedIntegration(self.config)
    
    def test_integration_initialization(self):
        """Test integration initialization"""
        self.assertFalse(self.integration.is_initialized)
        self.assertEqual(self.integration.integration_status, IntegrationStatus.NOT_INITIALIZED)
        
        # Test initialization
        hensel_config = HenselLiftingConfig(max_iterations=20)
        clustering_config = ClusteringConfig(max_cluster_size=500)
        
        self.integration.initialize_advanced_features(
            hensel_config=hensel_config,
            clustering_config=clustering_config,
            prime=7,
            precision=10
        )
        
        self.assertTrue(self.integration.is_initialized)
        self.assertEqual(self.integration.integration_status, IntegrationStatus.INITIALIZED)
        self.assertIsNotNone(self.integration.hensel_lifting_processor)
        self.assertIsNotNone(self.integration.clustering_manager)
        self.assertIsNotNone(self.integration.optimization_manager)
    
    def test_feature_integration(self):
        """Test feature integration methods"""
        # Initialize first
        self.integration.initialize_advanced_features(prime=7, precision=10)
        
        # Test Hensel lifting integration
        hensel_success = self.integration.integrate_hensel_lifting_with_hybrid()
        self.assertTrue(hensel_success)
        self.assertEqual(
            self.integration.feature_status[AdvancedFeatureType.HENSEL_LIFTING],
            IntegrationStatus.INTEGRATED
        )
        
        # Test clustering integration
        clustering_success = self.integration.integrate_clustering_with_hybrid()
        self.assertTrue(clustering_success)
        self.assertEqual(
            self.integration.feature_status[AdvancedFeatureType.HIERARCHICAL_CLUSTERING],
            IntegrationStatus.INTEGRATED
        )
        
        # Test optimization integration
        optimization_success = self.integration.integrate_optimization_with_hybrid()
        self.assertTrue(optimization_success)
        self.assertEqual(
            self.integration.feature_status[AdvancedFeatureType.PADIC_OPTIMIZATION],
            IntegrationStatus.INTEGRATED
        )
    
    def test_compatibility_validation(self):
        """Test advanced feature compatibility validation"""
        self.integration.initialize_advanced_features(prime=7, precision=10)
        
        # Test large data (should be compatible with all features)
        large_data = torch.randn(2000, dtype=torch.float32)
        
        self.assertTrue(self.integration.validate_advanced_compatibility(
            large_data, AdvancedFeatureType.HENSEL_LIFTING
        ))
        self.assertTrue(self.integration.validate_advanced_compatibility(
            large_data, AdvancedFeatureType.HIERARCHICAL_CLUSTERING
        ))
        self.assertTrue(self.integration.validate_advanced_compatibility(
            large_data, AdvancedFeatureType.PADIC_OPTIMIZATION
        ))
        
        # Test small data (should not be compatible)
        small_data = torch.randn(10, dtype=torch.float32)
        
        self.assertFalse(self.integration.validate_advanced_compatibility(
            small_data, AdvancedFeatureType.HENSEL_LIFTING
        ))
    
    def test_integration_status(self):
        """Test integration status reporting"""
        self.integration.initialize_advanced_features(prime=7, precision=10)
        
        status = self.integration.get_advanced_integration_status()
        
        self.assertIn('overall_status', status)
        self.assertIn('is_initialized', status)
        self.assertIn('feature_status', status)
        self.assertIn('component_availability', status)
        self.assertIn('configuration', status)
        self.assertIn('metrics', status)
        
        self.assertTrue(status['is_initialized'])
        self.assertEqual(status['overall_status'], IntegrationStatus.INITIALIZED.value)
    
    def test_integration_metrics(self):
        """Test integration metrics"""
        self.integration.initialize_advanced_features(prime=7, precision=10)
        
        # Update some metrics
        self.integration.update_operation_metrics(
            "hybrid", 0.8, AdvancedFeatureType.HENSEL_LIFTING
        )
        
        metrics = self.integration.get_integration_metrics()
        
        self.assertIn('overall_metrics', metrics)
        self.assertIn('feature_metrics', metrics)
        self.assertIn('performance_history', metrics)
        self.assertIn('integration_health', metrics)
        
        self.assertEqual(metrics['overall_metrics']['total_operations'], 1)
        self.assertEqual(metrics['overall_metrics']['hybrid_operations'], 1)
    
    def test_invalid_inputs(self):
        """Test invalid input handling"""
        # Test invalid config type
        with self.assertRaises(TypeError):
            HybridAdvancedIntegration("invalid_config")
        
        # Test operations without initialization
        with self.assertRaises(RuntimeError):
            self.integration.integrate_hensel_lifting_with_hybrid()
        
        # Test invalid data for compatibility
        with self.assertRaises(TypeError):
            self.integration.validate_advanced_compatibility(
                "invalid_data", AdvancedFeatureType.HENSEL_LIFTING
            )
    
    def tearDown(self):
        """Clean up test environment"""
        if self.integration and self.integration.is_initialized:
            self.integration.shutdown()


class TestHybridHenselLifting(unittest.TestCase):
    """Test cases for HybridHenselLifting"""
    
    def setUp(self):
        """Set up test environment"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for hybrid Hensel lifting tests")
        
        self.config = HenselLiftingConfig(max_iterations=20, convergence_tolerance=1e-8)
        self.hensel_lifting = HybridHenselLifting(self.config, prime=7, base_precision=10)
    
    def test_hensel_lifting_initialization(self):
        """Test Hensel lifting initialization"""
        self.assertEqual(self.hensel_lifting.prime, 7)
        self.assertEqual(self.hensel_lifting.base_precision, 10)
        self.assertEqual(self.hensel_lifting.current_precision, 10)
        self.assertIsNotNone(self.hensel_lifting.math_ops)
        self.assertIsNotNone(self.hensel_lifting.validator)
    
    def test_hybrid_weight_validation(self):
        """Test hybrid weight validation"""
        # Create valid hybrid weight
        valid_weight = HybridPadicWeight(
            exponent_channel=torch.tensor([1.0, 2.0], device='cuda'),
            mantissa_channel=torch.tensor([0.5, 0.3], device='cuda'),
            prime=7,
            precision=10,
            valuation=0,
            device=torch.device('cuda'),
            dtype=torch.float32
        )
        
        self.assertTrue(self.hensel_lifting.validate_hybrid_weight(valid_weight))
        
        # Test invalid weight
        self.assertFalse(self.hensel_lifting.validate_hybrid_weight("invalid"))
    
    def test_precision_lifting(self):
        """Test precision lifting operation"""
        # Create test hybrid weight
        test_weight = HybridPadicWeight(
            exponent_channel=torch.tensor([1.0, 2.0], device='cuda'),
            mantissa_channel=torch.tensor([0.5, 0.3], device='cuda'),
            prime=7,
            precision=10,
            valuation=0,
            device=torch.device('cuda'),
            dtype=torch.float32
        )
        
        # Test lifting to higher precision
        result = self.hensel_lifting.lift_hybrid_to_precision(test_weight, 15)
        
        self.assertIsInstance(result, HybridLiftingResult)
        self.assertEqual(result.lifted_weight.precision, 15)
        self.assertGreater(result.lifting_time_ms, 0)
        self.assertGreaterEqual(result.iterations_used, 1)
    
    def test_lifting_validation(self):
        """Test lifting result validation"""
        # Create test weights
        original = HybridPadicWeight(
            exponent_channel=torch.tensor([1.0], device='cuda'),
            mantissa_channel=torch.tensor([0.5], device='cuda'),
            prime=7,
            precision=10,
            valuation=0,
            device=torch.device('cuda'),
            dtype=torch.float32
        )
        
        lifted = HybridPadicWeight(
            exponent_channel=torch.tensor([1.1], device='cuda'),
            mantissa_channel=torch.tensor([0.55], device='cuda'),
            prime=7,
            precision=15,
            valuation=0,
            device=torch.device('cuda'),
            dtype=torch.float32
        )
        
        # Test validation
        is_valid = self.hensel_lifting.validate_hybrid_lifting(original, lifted)
        self.assertIsInstance(is_valid, bool)
    
    def test_lifting_statistics(self):
        """Test lifting statistics"""
        stats = self.hensel_lifting.get_hybrid_lifting_stats()
        
        self.assertIn('overall_stats', stats)
        self.assertIn('precision_stats', stats)
        self.assertIn('performance_stats', stats)
        self.assertIn('configuration', stats)
        
        self.assertEqual(stats['precision_stats']['base_precision'], 10)
        self.assertEqual(stats['configuration']['prime'], 7)
    
    def test_performance_optimization(self):
        """Test performance optimization"""
        optimization_result = self.hensel_lifting.optimize_hybrid_lifting_performance()
        
        self.assertIn('status', optimization_result)
        # Should indicate insufficient data for new instance
        self.assertEqual(optimization_result['status'], 'insufficient_data')
    
    def test_invalid_inputs(self):
        """Test invalid input handling"""
        # Test invalid config
        with self.assertRaises(TypeError):
            HybridHenselLifting("invalid_config", 7, 10)
        
        # Test invalid prime
        with self.assertRaises(ValueError):
            HybridHenselLifting(self.config, -1, 10)
        
        # Test invalid precision
        with self.assertRaises(ValueError):
            HybridHenselLifting(self.config, 7, 0)
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'hensel_lifting'):
            self.hensel_lifting.shutdown()


class TestHybridHierarchicalClustering(unittest.TestCase):
    """Test cases for HybridHierarchicalClustering"""
    
    def setUp(self):
        """Set up test environment"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for hybrid clustering tests")
        
        self.config = ClusteringConfig(max_cluster_size=100, min_cluster_size=2)
        self.clustering = HybridHierarchicalClustering(self.config, prime=7)
    
    def test_clustering_initialization(self):
        """Test clustering initialization"""
        self.assertEqual(self.clustering.prime, 7)
        self.assertIsNotNone(self.clustering.validator)
        self.assertEqual(self.clustering.device, torch.device('cuda'))
    
    def test_ultrametric_distance_computation(self):
        """Test ultrametric distance computation"""
        # Create test weights
        weight1 = HybridPadicWeight(
            exponent_channel=torch.tensor([1.0], device='cuda'),
            mantissa_channel=torch.tensor([0.5], device='cuda'),
            prime=7,
            precision=10,
            valuation=0,
            device=torch.device('cuda'),
            dtype=torch.float32
        )
        
        weight2 = HybridPadicWeight(
            exponent_channel=torch.tensor([2.0], device='cuda'),
            mantissa_channel=torch.tensor([0.3], device='cuda'),
            prime=7,
            precision=10,
            valuation=0,
            device=torch.device('cuda'),
            dtype=torch.float32
        )
        
        # Test distance computation
        distance = self.clustering.compute_hybrid_ultrametric_distance(weight1, weight2)
        
        self.assertIsInstance(distance, float)
        self.assertGreaterEqual(distance, 0.0)
        
        # Test distance to self (should be 0)
        self_distance = self.clustering.compute_hybrid_ultrametric_distance(weight1, weight1)
        self.assertEqual(self_distance, 0.0)
    
    def test_hierarchical_clustering(self):
        """Test hierarchical clustering operation"""
        # Create test weights
        test_weights = []
        for i in range(5):
            weight = HybridPadicWeight(
                exponent_channel=torch.tensor([float(i)], device='cuda'),
                mantissa_channel=torch.tensor([0.1 * i], device='cuda'),
                prime=7,
                precision=10,
                valuation=0,
                device=torch.device('cuda'),
                dtype=torch.float32
            )
            test_weights.append(weight)
        
        # Test clustering
        result = self.clustering.build_hybrid_hierarchical_clustering(test_weights)
        
        self.assertIsInstance(result, HybridClusteringResult)
        self.assertIsInstance(result.root_node, HybridClusterNode)
        self.assertGreater(result.total_clusters, 0)
        self.assertGreater(result.clustering_time_ms, 0)
        self.assertGreaterEqual(result.clustering_quality_score, 0.0)
        self.assertLessEqual(result.clustering_quality_score, 1.0)
    
    def test_cluster_tree_validation(self):
        """Test cluster tree validation"""
        # Create simple test weights
        test_weights = [
            HybridPadicWeight(
                exponent_channel=torch.tensor([1.0], device='cuda'),
                mantissa_channel=torch.tensor([0.5], device='cuda'),
                prime=7,
                precision=10,
                valuation=0,
                device=torch.device('cuda'),
                dtype=torch.float32
            ),
            HybridPadicWeight(
                exponent_channel=torch.tensor([2.0], device='cuda'),
                mantissa_channel=torch.tensor([0.3], device='cuda'),
                prime=7,
                precision=10,
                valuation=0,
                device=torch.device('cuda'),
                dtype=torch.float32
            )
        ]
        
        # Build clustering
        result = self.clustering.build_hybrid_hierarchical_clustering(test_weights)
        
        # Validate tree
        is_valid = self.clustering.validate_hybrid_cluster_tree(result.root_node)
        self.assertIsInstance(is_valid, bool)
    
    def test_clustering_statistics(self):
        """Test clustering statistics"""
        stats = self.clustering.get_hybrid_clustering_stats()
        
        self.assertIn('overall_stats', stats)
        self.assertIn('performance_stats', stats)
        self.assertIn('clustering_patterns', stats)
        self.assertIn('configuration', stats)
        
        self.assertEqual(stats['configuration']['prime'], 7)
        self.assertEqual(stats['configuration']['max_cluster_size'], 100)
    
    def test_invalid_inputs(self):
        """Test invalid input handling"""
        # Test invalid config
        with self.assertRaises(TypeError):
            HybridHierarchicalClustering("invalid_config", 7)
        
        # Test invalid prime
        with self.assertRaises(ValueError):
            HybridHierarchicalClustering(self.config, 0)
        
        # Test empty weights list
        with self.assertRaises(ValueError):
            self.clustering.build_hybrid_hierarchical_clustering([])
        
        # Test invalid weight type
        with self.assertRaises(TypeError):
            self.clustering.compute_hybrid_ultrametric_distance("invalid", "invalid")
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'clustering'):
            self.clustering.shutdown()


class TestHybridOptimizationManager(unittest.TestCase):
    """Test cases for HybridOptimizationManager"""
    
    def setUp(self):
        """Set up test environment"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for hybrid optimization tests")
        
        self.optimization_manager = HybridOptimizationManager(prime=7)
        
        # Create test parameters
        self.test_params = [
            HybridPadicWeight(
                exponent_channel=torch.tensor([1.0, 2.0], device='cuda'),
                mantissa_channel=torch.tensor([0.5, 0.3], device='cuda'),
                prime=7,
                precision=10,
                valuation=0,
                device=torch.device('cuda'),
                dtype=torch.float32
            ),
            HybridPadicWeight(
                exponent_channel=torch.tensor([0.5, 1.5], device='cuda'),
                mantissa_channel=torch.tensor([0.2, 0.8], device='cuda'),
                prime=7,
                precision=10,
                valuation=0,
                device=torch.device('cuda'),
                dtype=torch.float32
            )
        ]
    
    def test_optimization_manager_initialization(self):
        """Test optimization manager initialization"""
        self.assertEqual(self.optimization_manager.prime, 7)
        self.assertEqual(self.optimization_manager.device, torch.device('cuda'))
        self.assertEqual(len(self.optimization_manager.optimizers), 0)
    
    def test_sgd_optimizer_creation(self):
        """Test SGD optimizer creation"""
        optimizer_id = self.optimization_manager.create_hybrid_sgd_optimizer(
            self.test_params, lr=0.01, momentum=0.9
        )
        
        self.assertIsInstance(optimizer_id, str)
        self.assertIn(optimizer_id, self.optimization_manager.optimizers)
        
        optimizer_state = self.optimization_manager.optimizers[optimizer_id]
        self.assertEqual(optimizer_state.optimizer_type, "sgd")
        self.assertEqual(len(optimizer_state.parameters), 2)
        self.assertIsNotNone(optimizer_state.momentum_buffer_exp)
        self.assertIsNotNone(optimizer_state.momentum_buffer_man)
    
    def test_adam_optimizer_creation(self):
        """Test Adam optimizer creation"""
        optimizer_id = self.optimization_manager.create_hybrid_adam_optimizer(
            self.test_params, lr=0.001, betas=(0.9, 0.999)
        )
        
        self.assertIsInstance(optimizer_id, str)
        self.assertIn(optimizer_id, self.optimization_manager.optimizers)
        
        optimizer_state = self.optimization_manager.optimizers[optimizer_id]
        self.assertEqual(optimizer_state.optimizer_type, "adam")
        self.assertIsNotNone(optimizer_state.exp_avg_exp)
        self.assertIsNotNone(optimizer_state.exp_avg_man)
        self.assertIsNotNone(optimizer_state.exp_avg_sq_exp)
        self.assertIsNotNone(optimizer_state.exp_avg_sq_man)
    
    def test_rmsprop_optimizer_creation(self):
        """Test RMSprop optimizer creation"""
        optimizer_id = self.optimization_manager.create_hybrid_rmsprop_optimizer(
            self.test_params, lr=0.01, alpha=0.99
        )
        
        self.assertIsInstance(optimizer_id, str)
        self.assertIn(optimizer_id, self.optimization_manager.optimizers)
        
        optimizer_state = self.optimization_manager.optimizers[optimizer_id]
        self.assertEqual(optimizer_state.optimizer_type, "rmsprop")
        self.assertIsNotNone(optimizer_state.square_avg_exp)
        self.assertIsNotNone(optimizer_state.square_avg_man)
    
    def test_optimization_step(self):
        """Test optimization step"""
        # Create optimizer
        optimizer_id = self.optimization_manager.create_hybrid_sgd_optimizer(
            self.test_params, lr=0.01
        )
        
        # Create test gradients
        gradients = [
            (torch.randn(2, device='cuda'), torch.randn(2, device='cuda')),
            (torch.randn(2, device='cuda'), torch.randn(2, device='cuda'))
        ]
        
        # Perform optimization step
        success = self.optimization_manager.step_hybrid_optimizer(optimizer_id, gradients)
        
        self.assertTrue(success)
        
        # Check that step count increased
        optimizer_state = self.optimization_manager.optimizers[optimizer_id]
        self.assertEqual(optimizer_state.step_count, 1)
    
    def test_optimization_statistics(self):
        """Test optimization statistics"""
        # Create and use optimizer
        optimizer_id = self.optimization_manager.create_hybrid_sgd_optimizer(
            self.test_params, lr=0.01
        )
        
        stats = self.optimization_manager.get_hybrid_optimization_stats()
        
        self.assertIn('overall_stats', stats)
        self.assertIn('optimizer_type_stats', stats)
        self.assertIn('performance_stats', stats)
        self.assertIn('parameter_stats', stats)
        self.assertIn('optimizer_details', stats)
        self.assertIn('configuration', stats)
        
        self.assertEqual(stats['overall_stats']['total_optimizers_created'], 1)
        self.assertEqual(stats['overall_stats']['active_optimizers'], 1)
    
    def test_optimizer_info(self):
        """Test optimizer information retrieval"""
        optimizer_id = self.optimization_manager.create_hybrid_sgd_optimizer(
            self.test_params, lr=0.01, momentum=0.9
        )
        
        info = self.optimization_manager.get_optimizer_info(optimizer_id)
        
        self.assertIn('optimizer_id', info)
        self.assertIn('optimizer_type', info)
        self.assertIn('step_count', info)
        self.assertIn('parameter_count', info)
        self.assertIn('configuration', info)
        self.assertIn('performance', info)
        
        self.assertEqual(info['optimizer_type'], 'sgd')
        self.assertEqual(info['parameter_count'], 2)
        self.assertEqual(info['configuration']['learning_rate'], 0.01)
    
    def test_optimizer_removal(self):
        """Test optimizer removal"""
        optimizer_id = self.optimization_manager.create_hybrid_sgd_optimizer(
            self.test_params, lr=0.01
        )
        
        self.assertIn(optimizer_id, self.optimization_manager.optimizers)
        
        success = self.optimization_manager.remove_optimizer(optimizer_id)
        
        self.assertTrue(success)
        self.assertNotIn(optimizer_id, self.optimization_manager.optimizers)
    
    def test_invalid_inputs(self):
        """Test invalid input handling"""
        # Test invalid prime
        with self.assertRaises(ValueError):
            HybridOptimizationManager(prime=0)
        
        # Test empty parameters list
        with self.assertRaises(ValueError):
            self.optimization_manager.create_hybrid_sgd_optimizer([])
        
        # Test invalid parameter type
        with self.assertRaises(TypeError):
            self.optimization_manager.create_hybrid_sgd_optimizer(["invalid"])
        
        # Test invalid gradients
        optimizer_id = self.optimization_manager.create_hybrid_sgd_optimizer(
            self.test_params, lr=0.01
        )
        
        with self.assertRaises(ValueError):
            self.optimization_manager.step_hybrid_optimizer(optimizer_id, [])
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'optimization_manager'):
            self.optimization_manager.shutdown()


class TestCompleteIntegration(unittest.TestCase):
    """Test cases for complete advanced features integration"""
    
    def setUp(self):
        """Set up test environment"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for complete integration tests")
        
        self.config = AdvancedIntegrationConfig(
            enable_hensel_lifting=True,
            enable_hierarchical_clustering=True,
            enable_padic_optimization=True
        )
        self.integration = HybridAdvancedIntegration(self.config)
    
    def test_complete_initialization_and_integration(self):
        """Test complete initialization and integration workflow"""
        # Initialize advanced features
        self.integration.initialize_advanced_features(prime=7, precision=10)
        self.assertTrue(self.integration.is_initialized)
        
        # Integrate all features
        hensel_success = self.integration.integrate_hensel_lifting_with_hybrid()
        clustering_success = self.integration.integrate_clustering_with_hybrid()
        optimization_success = self.integration.integrate_optimization_with_hybrid()
        
        self.assertTrue(hensel_success)
        self.assertTrue(clustering_success)
        self.assertTrue(optimization_success)
        
        # Check integration status
        status = self.integration.get_advanced_integration_status()
        
        integrated_features = [
            status['feature_status']['hensel_lifting'],
            status['feature_status']['hierarchical_clustering'],
            status['feature_status']['padic_optimization']
        ]
        
        self.assertTrue(all(s == 'integrated' for s in integrated_features))
    
    def test_feature_switching_decision(self):
        """Test feature switching decision logic"""
        self.integration.initialize_advanced_features(prime=7, precision=10)
        
        # Test with large data (should use hybrid)
        large_data = torch.randn(2000, dtype=torch.float32)
        
        should_use_hybrid_hensel = self.integration.should_use_hybrid_feature(
            large_data, AdvancedFeatureType.HENSEL_LIFTING
        )
        should_use_hybrid_clustering = self.integration.should_use_hybrid_feature(
            large_data, AdvancedFeatureType.HIERARCHICAL_CLUSTERING
        )
        should_use_hybrid_optimization = self.integration.should_use_hybrid_feature(
            large_data, AdvancedFeatureType.PADIC_OPTIMIZATION
        )
        
        self.assertTrue(should_use_hybrid_hensel)
        self.assertTrue(should_use_hybrid_clustering)
        self.assertTrue(should_use_hybrid_optimization)
        
        # Test with small data (should use pure)
        small_data = torch.randn(10, dtype=torch.float32)
        
        should_use_hybrid_small = self.integration.should_use_hybrid_feature(
            small_data, AdvancedFeatureType.HENSEL_LIFTING
        )
        
        self.assertFalse(should_use_hybrid_small)
    
    def test_performance_monitoring(self):
        """Test performance monitoring integration"""
        self.integration.initialize_advanced_features(prime=7, precision=10)
        
        # Update metrics for different features
        self.integration.update_operation_metrics(
            "hybrid", 0.9, AdvancedFeatureType.HENSEL_LIFTING
        )
        self.integration.update_operation_metrics(
            "pure", 0.7, AdvancedFeatureType.HIERARCHICAL_CLUSTERING
        )
        self.integration.update_operation_metrics(
            "hybrid", 0.8, AdvancedFeatureType.PADIC_OPTIMIZATION
        )
        
        # Check metrics
        metrics = self.integration.get_integration_metrics()
        
        self.assertEqual(metrics['overall_metrics']['total_operations'], 3)
        self.assertEqual(metrics['overall_metrics']['hybrid_operations'], 2)
        self.assertEqual(metrics['overall_metrics']['pure_operations'], 1)
        
        # Check feature-specific metrics
        self.assertEqual(metrics['feature_metrics']['hensel_lifting_operations'], 1)
        self.assertEqual(metrics['feature_metrics']['clustering_operations'], 1)
        self.assertEqual(metrics['feature_metrics']['optimization_operations'], 1)
    
    def test_concurrent_operations(self):
        """Test concurrent advanced feature operations"""
        self.integration.initialize_advanced_features(prime=7, precision=10)
        
        def update_metrics_worker(feature_type, operation_count):
            for _ in range(operation_count):
                self.integration.update_operation_metrics(
                    "hybrid", 0.8, feature_type
                )
                time.sleep(0.01)  # Small delay to simulate work
        
        # Create multiple threads for different features
        threads = [
            threading.Thread(target=update_metrics_worker, 
                           args=(AdvancedFeatureType.HENSEL_LIFTING, 5)),
            threading.Thread(target=update_metrics_worker, 
                           args=(AdvancedFeatureType.HIERARCHICAL_CLUSTERING, 3)),
            threading.Thread(target=update_metrics_worker, 
                           args=(AdvancedFeatureType.PADIC_OPTIMIZATION, 4))
        ]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check final metrics
        metrics = self.integration.get_integration_metrics()
        self.assertEqual(metrics['overall_metrics']['total_operations'], 12)  # 5+3+4
        self.assertEqual(metrics['feature_metrics']['hensel_lifting_operations'], 5)
        self.assertEqual(metrics['feature_metrics']['clustering_operations'], 3)
        self.assertEqual(metrics['feature_metrics']['optimization_operations'], 4)
    
    def tearDown(self):
        """Clean up test environment"""
        if self.integration and self.integration.is_initialized:
            self.integration.shutdown()


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling in advanced integration"""
    
    def test_initialization_without_cuda(self):
        """Test initialization behavior without CUDA"""
        # This test would need to mock CUDA availability
        pass
    
    def test_invalid_configuration_handling(self):
        """Test handling of invalid configurations"""
        # Test invalid integration config
        with self.assertRaises(ValueError):
            AdvancedIntegrationConfig(hensel_lifting_threshold=-1)
        
        # Test invalid optimizer config
        with self.assertRaises(ValueError):
            HybridOptimizerConfig(learning_rate=-0.01)
    
    def test_integration_failure_handling(self):
        """Test handling of integration failures"""
        integration = HybridAdvancedIntegration()
        
        # Test operations without initialization
        with self.assertRaises(RuntimeError):
            integration.integrate_hensel_lifting_with_hybrid()
        
        with self.assertRaises(RuntimeError):
            integration.integrate_clustering_with_hybrid()
        
        with self.assertRaises(RuntimeError):
            integration.integrate_optimization_with_hybrid()


def run_hybrid_advanced_integration_tests():
    """Run all hybrid advanced integration test suites"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestAdvancedIntegrationConfig,
        TestHybridAdvancedIntegration,
        TestHybridHenselLifting,
        TestHybridHierarchicalClustering,
        TestHybridOptimizationManager,
        TestCompleteIntegration,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    # Run hybrid advanced integration tests
    success = run_hybrid_advanced_integration_tests()
    
    print(f"\nHybrid advanced integration tests {'PASSED' if success else 'FAILED'}")
    exit(0 if success else 1)