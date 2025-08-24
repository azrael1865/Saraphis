"""
Comprehensive test suite for TrainingPerformanceOptimizer.
Tests all functionality including edge cases, error handling, and integration.
"""

import unittest
import torch
import numpy as np
import time
import psutil
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from collections import deque
import json
import hashlib

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_performance_optimizer import (
    TrainingPerformanceOptimizer,
    TrainingPerformanceOptimizerConfig,
    TrainingPerformanceMetrics,
    OptimizationResult,
    TrainingAnalytics,
    OptimizationStrategy,
    OptimizationLevel,
    PerformanceRegression
)

class TestTrainingPerformanceOptimizer(unittest.TestCase):
    """Test suite for TrainingPerformanceOptimizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Default configuration
        self.config = TrainingPerformanceOptimizerConfig(
            optimization_strategy=OptimizationStrategy.ADAPTIVE,
            optimization_level=OptimizationLevel.MODERATE,
            enable_gpu_optimization=torch.cuda.is_available(),
            enable_memory_optimization=True,
            enable_algorithm_optimization=True,
            enable_regression_detection=True,
            enable_real_time_monitoring=False,  # Disable for testing
            enable_predictive_optimization=False,  # Disable for testing
            optimization_interval_seconds=1,  # Faster for testing
            monitoring_interval_seconds=1,  # Faster for testing
        )
        
        self.optimizer = TrainingPerformanceOptimizer(self.config)
        
        # Mock training manager and compression system
        self.mock_training_manager = Mock()
        self.mock_compression = Mock()
        
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'optimizer'):
            self.optimizer.shutdown()
    
    def test_initialization_default_config(self):
        """Test optimizer initialization with default config"""
        optimizer = TrainingPerformanceOptimizer()
        self.assertIsNotNone(optimizer.config)
        self.assertEqual(optimizer.config.optimization_strategy, OptimizationStrategy.ADAPTIVE)
        self.assertEqual(optimizer.config.optimization_level, OptimizationLevel.MODERATE)
        self.assertIsInstance(optimizer.performance_metrics, deque)
        self.assertIsInstance(optimizer.optimization_analytics, TrainingAnalytics)
        optimizer.shutdown()
    
    def test_initialization_custom_config(self):
        """Test optimizer initialization with custom config"""
        custom_config = TrainingPerformanceOptimizerConfig(
            optimization_strategy=OptimizationStrategy.PERFORMANCE_FIRST,
            optimization_level=OptimizationLevel.AGGRESSIVE,
            enable_gpu_optimization=False,
            performance_regression_threshold=0.2
        )
        optimizer = TrainingPerformanceOptimizer(custom_config)
        self.assertEqual(optimizer.config.optimization_strategy, OptimizationStrategy.PERFORMANCE_FIRST)
        self.assertEqual(optimizer.config.optimization_level, OptimizationLevel.AGGRESSIVE)
        self.assertFalse(optimizer.config.enable_gpu_optimization)
        self.assertEqual(optimizer.config.performance_regression_threshold, 0.2)
        optimizer.shutdown()
    
    def test_initialization_invalid_config(self):
        """Test initialization with invalid config"""
        with self.assertRaises(TypeError):
            TrainingPerformanceOptimizer("invalid_config")
    
    def test_initialize_optimizer(self):
        """Test optimizer initialization with systems"""
        self.optimizer.initialize_optimizer(
            training_manager=self.mock_training_manager,
            hybrid_compression=self.mock_compression
        )
        self.assertEqual(self.optimizer.training_manager, self.mock_training_manager)
        self.assertEqual(self.optimizer.hybrid_compression, self.mock_compression)
        # monitoring_active should be False since enable_real_time_monitoring is False in config
        self.assertFalse(self.optimizer.monitoring_active)
        self.assertTrue(self.optimizer.optimization_active)
    
    def test_profile_training_performance(self):
        """Test training performance profiling"""
        session_id = "test_session_001"
        
        def mock_training_operation():
            time.sleep(0.01)  # Simulate training
            return {"loss": 0.5, "accuracy": 0.85}
        
        metrics = self.optimizer.profile_training_performance(
            session_id, mock_training_operation
        )
        
        self.assertIsInstance(metrics, TrainingPerformanceMetrics)
        self.assertEqual(metrics.session_id, session_id)
        self.assertGreater(metrics.iteration_time_ms, 0)
        self.assertGreaterEqual(metrics.memory_usage_gb, 0)
        self.assertGreaterEqual(metrics.cpu_utilization_percent, 0)
    
    def test_optimize_training_gpu_memory(self):
        """Test GPU memory optimization"""
        session_id = "test_session_002"
        
        # Initialize with mock systems
        self.optimizer.initialize_optimizer(
            training_manager=self.mock_training_manager,
            hybrid_compression=self.mock_compression
        )
        
        # Mock GPU availability
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.memory_allocated', return_value=1e9):  # 1GB
                with patch('torch.cuda.max_memory_allocated', return_value=2e9):  # 2GB
                    result = self.optimizer.optimize_training_gpu_memory(session_id)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.session_id, session_id)
        self.assertEqual(result.optimization_type, "gpu_memory")
        self.assertIsInstance(result.optimizations_applied, list)
    
    def test_optimize_training_algorithms(self):
        """Test training algorithm optimization"""
        session_id = "test_session_003"
        
        # Initialize with mock systems
        self.optimizer.initialize_optimizer(
            training_manager=self.mock_training_manager,
            hybrid_compression=self.mock_compression
        )
        
        # Set up mock training manager responses
        self.mock_training_manager.get_batch_size.return_value = 32
        self.mock_training_manager.get_learning_rate.return_value = 0.001
        
        result = self.optimizer.optimize_training_algorithms(session_id)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.session_id, session_id)
        self.assertEqual(result.optimization_type, "algorithms")
        self.assertIsInstance(result.optimizations_applied, list)
    
    def test_detect_training_performance_regression(self):
        """Test performance regression detection"""
        session_id = "test_session_004"
        
        # Add baseline metrics
        baseline_metrics = TrainingPerformanceMetrics(
            session_id=session_id,
            timestamp=datetime.utcnow(),
            iteration_time_ms=100.0,
            memory_usage_gb=4.0,
            gpu_utilization_percent=80.0,
            cpu_utilization_percent=60.0
        )
        self.optimizer.performance_metrics.append(baseline_metrics)
        self.optimizer.session_baselines[session_id] = {
            'iteration_time_ms': 100.0,
            'memory_usage_gb': 4.0
        }
        
        # Add degraded metrics
        for i in range(5):
            degraded_metrics = TrainingPerformanceMetrics(
                session_id=session_id,
                timestamp=datetime.utcnow(),
                iteration_time_ms=150.0,  # 50% slower
                memory_usage_gb=6.0,  # 50% more memory
                gpu_utilization_percent=40.0,
                cpu_utilization_percent=90.0
            )
            self.optimizer.performance_metrics.append(degraded_metrics)
        
        result = self.optimizer.detect_training_performance_regression(session_id)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result['regression_detected'])
        self.assertNotEqual(result['regression_severity'], 'none')
        self.assertIn('iteration_time_ms', result['regressions'])
    
    def test_monitor_training_performance(self):
        """Test training performance monitoring"""
        session_id = "test_session_005"
        
        # Add some metrics
        metrics = TrainingPerformanceMetrics(
            session_id=session_id,
            timestamp=datetime.utcnow(),
            iteration_time_ms=100.0,
            memory_usage_gb=4.0,
            gpu_utilization_percent=75.0,
            cpu_utilization_percent=55.0,
            training_loss=0.3,
            training_accuracy=0.9
        )
        self.optimizer.performance_metrics.append(metrics)
        
        result = self.optimizer.monitor_training_performance(session_id)
        
        self.assertIsInstance(result, dict)
        self.assertIn('current_metrics', result)
        self.assertIn('performance_trends', result)
        self.assertIn('alerts', result)
        self.assertIn('recommendations', result)
    
    def test_optimize_training_for_performance(self):
        """Test comprehensive performance optimization"""
        session_id = "test_session_006"
        
        # Initialize with mock systems
        self.optimizer.initialize_optimizer(
            training_manager=self.mock_training_manager,
            hybrid_compression=self.mock_compression
        )
        
        performance_data = {
            'memory_pressure': True,
            'gpu_underutilized': False,
            'cpu_bottleneck': False,
            'compression_inefficient': False
        }
        
        result = self.optimizer.optimize_training_for_performance(
            session_id, performance_data
        )
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.session_id, session_id)
        self.assertEqual(result.optimization_strategy, OptimizationStrategy.MEMORY_FIRST)
        self.assertIsInstance(result.optimizations_applied, list)
    
    def test_get_optimization_analytics(self):
        """Test analytics retrieval"""
        # Add some optimization results
        result1 = OptimizationResult(
            optimization_id="opt_001",
            session_id="session_001",
            optimization_type="gpu_memory",
            optimization_strategy=OptimizationStrategy.ADAPTIVE,
            optimization_level=OptimizationLevel.MODERATE,
            optimization_successful=True,
            performance_improvement_percent=15.0,
            memory_improvement_gb=2.0
        )
        result2 = OptimizationResult(
            optimization_id="opt_002",
            session_id="session_002",
            optimization_type="algorithms",
            optimization_strategy=OptimizationStrategy.PERFORMANCE_FIRST,
            optimization_level=OptimizationLevel.AGGRESSIVE,
            optimization_successful=True,
            performance_improvement_percent=20.0
        )
        
        self.optimizer.optimization_history.append(result1)
        self.optimizer.optimization_history.append(result2)
        self.optimizer._update_optimization_analytics(result1)
        self.optimizer._update_optimization_analytics(result2)
        
        # Get global analytics
        analytics = self.optimizer.get_optimization_analytics()
        
        self.assertIsInstance(analytics, dict)
        self.assertIn('global_analytics', analytics)
        self.assertIn('system_performance', analytics)
        self.assertIn('optimization_history', analytics)
        
        # Get session-specific analytics
        session_analytics = self.optimizer.get_optimization_analytics("session_001")
        self.assertIn('session_analytics', session_analytics)
        self.assertEqual(session_analytics['session_analytics']['session_id'], 'session_001')
    
    def test_system_metrics_capture(self):
        """Test system metrics capture"""
        metrics = self.optimizer._capture_system_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('cpu_percent', metrics)
        self.assertIn('memory_gb', metrics)
        self.assertIn('available_memory_gb', metrics)
        self.assertGreaterEqual(metrics['cpu_percent'], 0)
        self.assertGreaterEqual(metrics['memory_gb'], 0)
    
    def test_performance_baseline_calculation(self):
        """Test performance baseline calculation"""
        session_id = "test_session_007"
        
        # Add multiple metrics
        for i in range(10):
            metrics = TrainingPerformanceMetrics(
                session_id=session_id,
                timestamp=datetime.utcnow(),
                iteration_time_ms=100.0 + i,
                memory_usage_gb=4.0 + i * 0.1,
                gpu_utilization_percent=75.0,
                cpu_utilization_percent=55.0
            )
            self.optimizer.performance_metrics.append(metrics)
        
        avg_performance = self.optimizer._calculate_average_performance(
            list(self.optimizer.performance_metrics)
        )
        
        self.assertIsInstance(avg_performance, dict)
        self.assertIn('iteration_time_ms', avg_performance)
        self.assertIn('memory_usage_gb', avg_performance)
        self.assertAlmostEqual(avg_performance['iteration_time_ms'], 104.5, places=1)
    
    def test_regression_severity_classification(self):
        """Test regression severity classification"""
        self.assertEqual(
            self.optimizer._classify_regression_severity(0.05),
            PerformanceRegression.MINOR
        )
        self.assertEqual(
            self.optimizer._classify_regression_severity(0.15),
            PerformanceRegression.MODERATE
        )
        self.assertEqual(
            self.optimizer._classify_regression_severity(0.35),
            PerformanceRegression.SEVERE
        )
        self.assertEqual(
            self.optimizer._classify_regression_severity(0.55),
            PerformanceRegression.CRITICAL
        )
    
    def test_optimization_id_generation(self):
        """Test optimization ID generation"""
        opt_id1 = self.optimizer._generate_optimization_id("test_type")
        opt_id2 = self.optimizer._generate_optimization_id("test_type")
        
        self.assertIsInstance(opt_id1, str)
        self.assertIsInstance(opt_id2, str)
        self.assertNotEqual(opt_id1, opt_id2)  # Should be unique
        self.assertTrue(opt_id1.startswith("test_type_"))
    
    def test_thread_safety(self):
        """Test thread safety of optimizer operations"""
        session_id = "test_session_thread"
        results = []
        errors = []
        
        def optimization_task():
            try:
                result = self.optimizer.monitor_training_performance(session_id)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=optimization_task)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join(timeout=5)
        
        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")
        self.assertEqual(len(results), 5)
    
    def test_memory_optimization_edge_cases(self):
        """Test memory optimization edge cases"""
        session_id = "test_session_memory"
        
        # Test with zero memory
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(available=0, total=8e9, used=8e9, percent=100.0)
            metrics = self.optimizer._capture_system_metrics()
            self.assertEqual(metrics['available_memory_gb'], 0)
        
        # Test with very high memory usage
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(available=1e6, total=8e9, used=7.999e9, percent=99.9)
            metrics = self.optimizer._capture_system_metrics()
            self.assertLess(metrics['available_memory_gb'], 0.001)
    
    def test_gpu_optimization_without_cuda(self):
        """Test GPU optimization when CUDA is not available"""
        session_id = "test_session_no_gpu"
        
        with patch('torch.cuda.is_available', return_value=False):
            result = self.optimizer.optimize_training_gpu_memory(session_id)
            
        self.assertIsInstance(result, OptimizationResult)
        self.assertFalse(result.optimization_successful)
        self.assertIn("not available", result.error_message.lower())
    
    def test_optimizer_status(self):
        """Test optimizer status retrieval"""
        status = self.optimizer.get_optimizer_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('monitoring_active', status)
        self.assertIn('optimization_active', status)
        self.assertIn('total_optimizations', status)
        self.assertIn('active_optimizations', status)
    
    def test_shutdown(self):
        """Test optimizer shutdown"""
        self.optimizer.monitoring_active = True
        self.optimizer.optimization_active = True
        self.optimizer.regression_detection_active = True
        
        self.optimizer.shutdown()
        
        self.assertFalse(self.optimizer.monitoring_active)
        self.assertFalse(self.optimizer.optimization_active)
        self.assertFalse(self.optimizer.regression_detection_active)
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Invalid optimization strategy type
        with self.assertRaises(TypeError):
            TrainingPerformanceOptimizerConfig(
                optimization_strategy="invalid"
            )
        
        # Invalid optimization level type
        with self.assertRaises(TypeError):
            TrainingPerformanceOptimizerConfig(
                optimization_level="invalid"
            )
        
        # Invalid regression threshold
        with self.assertRaises(ValueError):
            TrainingPerformanceOptimizerConfig(
                performance_regression_threshold=1.5
            )
        
        # Invalid memory threshold
        with self.assertRaises(ValueError):
            TrainingPerformanceOptimizerConfig(
                memory_optimization_threshold_gb=-1
            )
    
    def test_metrics_validation(self):
        """Test metrics validation"""
        # Empty session ID
        with self.assertRaises(ValueError):
            TrainingPerformanceMetrics(
                session_id="",
                timestamp=datetime.utcnow()
            )
        
        # Negative iteration time
        with self.assertRaises(ValueError):
            TrainingPerformanceMetrics(
                session_id="test",
                timestamp=datetime.utcnow(),
                iteration_time_ms=-1
            )
    
    def test_optimization_result_validation(self):
        """Test optimization result validation"""
        # Empty IDs
        with self.assertRaises(ValueError):
            OptimizationResult(
                optimization_id="",
                session_id="",
                optimization_type="test",
                optimization_strategy=OptimizationStrategy.ADAPTIVE,
                optimization_level=OptimizationLevel.MODERATE
            )
        
        # Invalid strategy type
        with self.assertRaises(TypeError):
            OptimizationResult(
                optimization_id="opt_001",
                session_id="session_001",
                optimization_type="test",
                optimization_strategy="invalid",
                optimization_level=OptimizationLevel.MODERATE
            )
    
    def test_performance_trend_analysis(self):
        """Test performance trend analysis"""
        session_id = "test_session_trends"
        
        # Add trending metrics
        for i in range(20):
            metrics = TrainingPerformanceMetrics(
                session_id=session_id,
                timestamp=datetime.utcnow() + timedelta(seconds=i),
                iteration_time_ms=100.0 + i * 2,  # Degrading
                memory_usage_gb=4.0 - i * 0.05,  # Improving
                gpu_utilization_percent=75.0,  # Stable
                cpu_utilization_percent=55.0 + i * 0.5
            )
            self.optimizer.performance_metrics.append(metrics)
        
        trends = self.optimizer._analyze_performance_trends(session_id)
        
        self.assertIsInstance(trends, dict)
        self.assertEqual(trends['iteration_time'], 'degrading')
        self.assertEqual(trends['memory_usage'], 'improving')
        self.assertEqual(trends['gpu_utilization'], 'stable')


class TestTrainingPerformanceOptimizerIntegration(unittest.TestCase):
    """Integration tests for TrainingPerformanceOptimizer"""
    
    def test_full_optimization_cycle(self):
        """Test complete optimization cycle"""
        config = TrainingPerformanceOptimizerConfig(
            optimization_strategy=OptimizationStrategy.ADAPTIVE,
            optimization_level=OptimizationLevel.AGGRESSIVE,
            enable_regression_detection=True
        )
        optimizer = TrainingPerformanceOptimizer(config)
        
        # Mock systems
        mock_training_manager = Mock()
        mock_compression = Mock()
        
        # Initialize
        optimizer.initialize_optimizer(
            training_manager=mock_training_manager,
            hybrid_compression=mock_compression
        )
        
        session_id = "integration_test"
        
        # Profile performance
        def training_op():
            time.sleep(0.001)
            return {"loss": 0.5}
        
        metrics = optimizer.profile_training_performance(session_id, training_op)
        self.assertIsNotNone(metrics)
        
        # Monitor performance
        monitoring_result = optimizer.monitor_training_performance(session_id)
        self.assertIsNotNone(monitoring_result)
        
        # Detect regression
        regression_result = optimizer.detect_training_performance_regression(session_id)
        self.assertIsNotNone(regression_result)
        
        # Optimize
        optimization_result = optimizer.optimize_training_for_performance(
            session_id,
            {'memory_pressure': False, 'gpu_underutilized': True}
        )
        self.assertIsNotNone(optimization_result)
        
        # Get analytics
        analytics = optimizer.get_optimization_analytics(session_id)
        self.assertIsNotNone(analytics)
        
        # Shutdown
        optimizer.shutdown()
    
    def test_continuous_monitoring(self):
        """Test continuous monitoring over time"""
        optimizer = TrainingPerformanceOptimizer()
        session_id = "continuous_test"
        
        # Simulate monitoring over time
        for i in range(10):
            metrics = TrainingPerformanceMetrics(
                session_id=session_id,
                timestamp=datetime.utcnow(),
                iteration_time_ms=100.0 + np.random.normal(0, 10),
                memory_usage_gb=4.0 + np.random.normal(0, 0.5),
                gpu_utilization_percent=75.0 + np.random.normal(0, 5),
                cpu_utilization_percent=55.0 + np.random.normal(0, 5),
                training_loss=0.5 - i * 0.01,
                training_accuracy=0.7 + i * 0.02
            )
            optimizer.performance_metrics.append(metrics)
            
            if i % 3 == 0:  # Monitor every 3 iterations
                result = optimizer.monitor_training_performance(session_id)
                self.assertIsNotNone(result)
        
        # Final analytics
        analytics = optimizer.get_optimization_analytics(session_id)
        self.assertIn('session_analytics', analytics)
        
        optimizer.shutdown()
    
    def test_adaptive_optimization(self):
        """Test adaptive optimization based on conditions"""
        config = TrainingPerformanceOptimizerConfig(
            optimization_strategy=OptimizationStrategy.ADAPTIVE
        )
        optimizer = TrainingPerformanceOptimizer(config)
        
        session_id = "adaptive_test"
        
        # Test different scenarios
        scenarios = [
            {'memory_pressure': True, 'gpu_underutilized': False},
            {'memory_pressure': False, 'gpu_underutilized': True},
            {'memory_pressure': False, 'cpu_bottleneck': True},
            {'compression_inefficient': True}
        ]
        
        for scenario in scenarios:
            result = optimizer.optimize_training_for_performance(
                session_id, scenario
            )
            self.assertIsNotNone(result)
            # Verify appropriate strategy was selected
            if scenario.get('memory_pressure'):
                self.assertEqual(result.optimization_strategy, OptimizationStrategy.MEMORY_FIRST)
            elif scenario.get('gpu_underutilized'):
                self.assertEqual(result.optimization_strategy, OptimizationStrategy.GPU_OPTIMIZED)
        
        optimizer.shutdown()


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingPerformanceOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingPerformanceOptimizerIntegration))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✓ All TrainingPerformanceOptimizer tests passed!")
    else:
        print(f"✗ Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
        for failure in result.failures:
            print(f"\nFAILURE: {failure[0]}")
            print(failure[1])
        for error in result.errors:
            print(f"\nERROR: {error[0]}")
            print(error[1])
    
    return result


if __name__ == "__main__":
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)