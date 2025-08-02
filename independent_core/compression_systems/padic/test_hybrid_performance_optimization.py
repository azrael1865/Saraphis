"""
Comprehensive Tests for Hybrid Performance Optimization
Tests for HybridPerformanceOptimizer, HybridPerformanceMonitor, HybridPerformanceAnalyzer, and HybridPerformanceTuner
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

# Import hybrid performance optimization components
from .hybrid_performance_optimizer import (
    HybridPerformanceOptimizer, HybridOptimizationConfig,
    HybridPerformanceCategory, HybridOperationProfile
)
from .hybrid_performance_monitor import (
    HybridPerformanceMonitor, HybridMonitoringConfig,
    MonitoringLevel, HybridOperationMonitorData
)
from .hybrid_performance_analyzer import (
    HybridPerformanceAnalyzer, HybridAnalysisConfig,
    BottleneckType, BottleneckAnalysisResult
)
from .hybrid_performance_tuner import (
    HybridPerformanceTuner, HybridTuningConfig,
    TuningStrategy, TuningParameter
)

# Import hybrid system components
from .hybrid_padic_structures import HybridPadicWeight, HybridPadicManager
from .dynamic_switching_manager import DynamicSwitchingManager
from .hybrid_padic_compressor import HybridPadicCompressionSystem
from ...performance_optimizer import PerformanceOptimizer


class TestHybridOptimizationConfig(unittest.TestCase):
    """Test cases for HybridOptimizationConfig"""
    
    def test_valid_config_creation(self):
        """Test valid configuration creation"""
        config = HybridOptimizationConfig(
            enable_gpu_optimization=True,
            enable_memory_optimization=True,
            gpu_memory_threshold_mb=1024,
            performance_regression_threshold=0.1
        )
        
        self.assertTrue(config.enable_gpu_optimization)
        self.assertTrue(config.enable_memory_optimization)
        self.assertEqual(config.gpu_memory_threshold_mb, 1024)
        self.assertEqual(config.performance_regression_threshold, 0.1)
    
    def test_invalid_config_values(self):
        """Test invalid configuration values"""
        with self.assertRaises(ValueError):
            HybridOptimizationConfig(gpu_memory_threshold_mb=-100)
        
        with self.assertRaises(ValueError):
            HybridOptimizationConfig(performance_regression_threshold=-0.1)
        
        with self.assertRaises(ValueError):
            HybridOptimizationConfig(learning_rate=1.5)
    
    def test_default_config_values(self):
        """Test default configuration values"""
        config = HybridOptimizationConfig()
        
        self.assertTrue(config.enable_gpu_optimization)
        self.assertTrue(config.enable_memory_optimization)
        self.assertGreater(config.gpu_memory_threshold_mb, 0)
        self.assertGreater(config.performance_regression_threshold, 0)


class TestHybridPerformanceOptimizer(unittest.TestCase):
    """Test cases for HybridPerformanceOptimizer"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = HybridOptimizationConfig()
        self.optimizer = HybridPerformanceOptimizer(self.config)
        
        # Create mock components
        self.mock_base_optimizer = Mock(spec=PerformanceOptimizer)
        self.mock_hybrid_manager = Mock(spec=HybridPadicManager)
        self.mock_switching_manager = Mock(spec=DynamicSwitchingManager)
        self.mock_compression_system = Mock(spec=HybridPadicCompressionSystem)
        
        # Mock compression system stats
        self.mock_compression_system.performance_stats = {
            'total_compressions': 100,
            'total_decompressions': 100,
            'average_compression_time': 50.0,
            'average_decompression_time': 30.0
        }
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        self.assertFalse(self.optimizer.is_initialized)
        
        # Test initialization
        self.optimizer.initialize_hybrid_optimization(
            self.mock_base_optimizer,
            self.mock_hybrid_manager,
            self.mock_switching_manager,
            self.mock_compression_system
        )
        
        self.assertTrue(self.optimizer.is_initialized)
        self.assertIsNotNone(self.optimizer.base_performance_optimizer)
        self.assertIsNotNone(self.optimizer.hybrid_padic_manager)
    
    def test_invalid_initialization_parameters(self):
        """Test invalid initialization parameters"""
        with self.assertRaises(TypeError):
            self.optimizer.initialize_hybrid_optimization(
                "invalid_optimizer",
                self.mock_hybrid_manager,
                self.mock_switching_manager,
                self.mock_compression_system
            )
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=1024*1024*512)  # 512MB
    def test_operation_profiling(self, mock_memory, mock_cuda):
        """Test hybrid operation profiling"""
        self.optimizer.initialize_hybrid_optimization(
            self.mock_base_optimizer,
            self.mock_hybrid_manager,
            self.mock_switching_manager,
            self.mock_compression_system
        )
        
        def test_operation():
            time.sleep(0.01)  # Simulate work
            return "test_result"
        
        # Configure mock base optimizer
        self.mock_base_optimizer.profile_operation.return_value = "test_result"
        
        result = self.optimizer.profile_hybrid_operation(
            "test_operation",
            test_operation,
            HybridPerformanceCategory.COMPRESSION
        )
        
        self.assertEqual(result, "test_result")
        self.mock_base_optimizer.profile_operation.assert_called_once()
        
        # Check that operation was recorded
        self.assertGreater(len(self.optimizer.operation_profiles), 0)
    
    def test_compression_optimization(self):
        """Test compression performance optimization"""
        self.optimizer.initialize_hybrid_optimization(
            self.mock_base_optimizer,
            self.mock_hybrid_manager,
            self.mock_switching_manager,
            self.mock_compression_system
        )
        
        # Add some mock operation profiles
        for i in range(20):
            profile = HybridOperationProfile(
                operation_name=f"compression_{i}",
                operation_category=HybridPerformanceCategory.COMPRESSION,
                execution_time_ms=100.0 + i,
                gpu_memory_used_mb=512.0,
                cpu_usage_percent=50.0,
                memory_usage_mb=1024.0,
                switching_overhead_ms=5.0,
                compression_ratio=2.5,
                success=True,
                hybrid_mode_used=True
            )
            self.optimizer.operation_profiles.append(profile)
        
        result = self.optimizer.optimize_hybrid_compression_performance()
        
        self.assertIn('optimizations_applied', result)
        self.assertIn('performance_improvements', result)
        self.assertIsInstance(result['optimizations_applied'], list)
    
    def test_performance_regression_detection(self):
        """Test performance regression detection"""
        self.optimizer.initialize_hybrid_optimization(
            self.mock_base_optimizer,
            self.mock_hybrid_manager,
            self.mock_switching_manager,
            self.mock_compression_system
        )
        
        # Set up baseline
        self.optimizer.baseline_performance = {
            'compression_time_ms': 50.0,
            'compression_success_rate': 0.95
        }
        
        # Add profiles that show regression
        for i in range(25):
            profile = HybridOperationProfile(
                operation_name=f"operation_{i}",
                operation_category=HybridPerformanceCategory.COMPRESSION,
                execution_time_ms=80.0,  # Worse than baseline
                gpu_memory_used_mb=512.0,
                cpu_usage_percent=50.0,
                memory_usage_mb=1024.0,
                switching_overhead_ms=5.0,
                compression_ratio=2.5,
                success=True,
                hybrid_mode_used=True
            )
            self.optimizer.operation_profiles.append(profile)
        
        regressions = self.optimizer.detect_hybrid_performance_regression()
        
        # Should detect regression in execution time
        self.assertGreater(len(regressions), 0)
        regression_types = [r['type'] for r in regressions]
        self.assertIn('execution_time_regression', regression_types)
    
    def test_performance_report_generation(self):
        """Test performance report generation"""
        self.optimizer.initialize_hybrid_optimization(
            self.mock_base_optimizer,
            self.mock_hybrid_manager,
            self.mock_switching_manager,
            self.mock_compression_system
        )
        
        # Add some operation data
        profile = HybridOperationProfile(
            operation_name="test_operation",
            operation_category=HybridPerformanceCategory.COMPRESSION,
            execution_time_ms=100.0,
            gpu_memory_used_mb=512.0,
            cpu_usage_percent=50.0,
            memory_usage_mb=1024.0,
            switching_overhead_ms=5.0,
            compression_ratio=2.5,
            success=True,
            hybrid_mode_used=True
        )
        self.optimizer.operation_profiles.append(profile)
        self.optimizer.performance_metrics.update_with_profile(profile)
        
        report = self.optimizer.get_hybrid_performance_report()
        
        self.assertIn('overview', report)
        self.assertIn('timing_metrics', report)
        self.assertIn('resource_metrics', report)
        self.assertIn('category_performance', report)
        self.assertIn('optimization_status', report)
        
        # Check specific metrics
        self.assertEqual(report['overview']['total_operations'], 1)
        self.assertEqual(report['overview']['hybrid_operations'], 1)
    
    def tearDown(self):
        """Clean up test environment"""
        if self.optimizer and self.optimizer.is_initialized:
            self.optimizer.shutdown()


class TestHybridPerformanceMonitor(unittest.TestCase):
    """Test cases for HybridPerformanceMonitor"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = HybridMonitoringConfig(
            monitoring_level=MonitoringLevel.DETAILED,
            monitoring_interval_seconds=0.1  # Fast for testing
        )
        self.monitor = HybridPerformanceMonitor(self.config)
        
        # Create mock components
        self.mock_hybrid_manager = Mock(spec=HybridPadicManager)
        self.mock_switching_manager = Mock(spec=DynamicSwitchingManager)
        self.mock_compression_system = Mock(spec=HybridPadicCompressionSystem)
    
    def test_monitor_initialization(self):
        """Test monitor initialization"""
        self.assertFalse(self.monitor.is_monitoring)
        
        # Test initialization
        self.monitor.start_hybrid_monitoring(
            self.mock_hybrid_manager,
            self.mock_switching_manager,
            self.mock_compression_system
        )
        
        self.assertTrue(self.monitor.is_monitoring)
        self.assertIsNotNone(self.monitor.hybrid_padic_manager)
    
    def test_operation_monitoring(self):
        """Test operation monitoring"""
        self.monitor.start_hybrid_monitoring(self.mock_hybrid_manager)
        
        # Start monitoring an operation
        operation_id = self.monitor.monitor_hybrid_operation(
            "test_operation",
            {"hybrid_mode": True, "data_size": 1000}
        )
        
        self.assertIsInstance(operation_id, str)
        self.assertIn(operation_id, self.monitor.active_operations)
        
        # Complete the operation
        self.monitor.complete_operation_monitoring(operation_id, success=True)
        
        self.assertNotIn(operation_id, self.monitor.active_operations)
        self.assertGreater(len(self.monitor.operation_history), 0)
    
    def test_performance_metrics_collection(self):
        """Test performance metrics collection"""
        self.monitor.start_hybrid_monitoring(self.mock_hybrid_manager)
        
        # Add some operation data
        for i in range(10):
            op_id = self.monitor.monitor_hybrid_operation(
                f"operation_{i}",
                {"hybrid_mode": i % 2 == 0, "data_size": 1000 + i}
            )
            self.monitor.complete_operation_monitoring(op_id, success=True)
        
        metrics = self.monitor.get_hybrid_performance_metrics()
        
        self.assertIn('monitoring_status', metrics)
        self.assertIn('operation_metrics', metrics)
        self.assertIn('performance_metrics', metrics)
        self.assertIn('resource_metrics', metrics)
        
        # Check operation counts
        self.assertEqual(metrics['operation_metrics']['total_operations'], 10)
        self.assertEqual(metrics['operation_metrics']['hybrid_operations'], 5)
        self.assertEqual(metrics['operation_metrics']['pure_operations'], 5)
    
    @patch('torch.cuda.is_available', return_value=True)
    def test_performance_trends_analysis(self, mock_cuda):
        """Test performance trends analysis"""
        self.monitor.start_hybrid_monitoring(self.mock_hybrid_manager)
        
        # Add trending data (improving performance over time)
        for i in range(15):
            op_data = HybridOperationMonitorData(
                operation_id=f"op_{i}",
                operation_name=f"operation_{i}",
                start_time=datetime.utcnow(),
                execution_time_ms=100.0 - i,  # Improving performance
                hybrid_mode_used=True,
                success=True
            )
            op_data.calculate_performance_score()
            self.monitor.operation_history.append(op_data)
        
        trends = self.monitor.analyze_hybrid_performance_trends()
        
        self.assertIn('performance_trend', trends)
        self.assertIn('execution_time_trend', trends)
        
        # Should detect improving trend
        perf_trend = trends['performance_trend']
        if 'direction' in perf_trend:
            self.assertIn(perf_trend['direction'], ['increasing', 'stable'])
    
    def test_alert_generation(self):
        """Test performance alert generation"""
        self.monitor.start_hybrid_monitoring(self.mock_hybrid_manager)
        
        # Set up metrics that should trigger alerts
        self.monitor.monitoring_metrics.average_execution_time_ms = 6000.0  # Above threshold
        self.monitor.monitoring_metrics.peak_gpu_memory_usage_mb = 1500.0   # Above threshold
        
        alerts = self.monitor.generate_hybrid_performance_alerts()
        
        self.assertGreater(len(alerts), 0)
        
        # Check alert types
        alert_types = [alert.alert_type for alert in alerts]
        self.assertIn('execution_time_high', alert_types)
        self.assertIn('gpu_memory_high', alert_types)
    
    def test_concurrent_monitoring(self):
        """Test concurrent operation monitoring"""
        self.monitor.start_hybrid_monitoring(self.mock_hybrid_manager)
        
        def monitor_operations(start_index, count):
            for i in range(start_index, start_index + count):
                op_id = self.monitor.monitor_hybrid_operation(
                    f"concurrent_op_{i}",
                    {"hybrid_mode": True, "data_size": 1000}
                )
                time.sleep(0.01)  # Simulate work
                self.monitor.complete_operation_monitoring(op_id, success=True)
        
        # Start multiple threads
        threads = [
            threading.Thread(target=monitor_operations, args=(0, 5)),
            threading.Thread(target=monitor_operations, args=(5, 5)),
            threading.Thread(target=monitor_operations, args=(10, 5))
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Check that all operations were monitored
        self.assertEqual(len(self.monitor.operation_history), 15)
    
    def tearDown(self):
        """Clean up test environment"""
        if self.monitor and self.monitor.is_monitoring:
            self.monitor.stop_hybrid_monitoring()


class TestHybridPerformanceAnalyzer(unittest.TestCase):
    """Test cases for HybridPerformanceAnalyzer"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = HybridAnalysisConfig()
        self.analyzer = HybridPerformanceAnalyzer(self.config)
        
        # Create mock components
        self.mock_hybrid_manager = Mock(spec=HybridPadicManager)
        self.mock_switching_manager = Mock(spec=DynamicSwitchingManager)
        self.mock_compression_system = Mock(spec=HybridPadicCompressionSystem)
        self.mock_performance_monitor = Mock(spec=HybridPerformanceMonitor)
        
        # Mock operation history
        self.mock_operation_data = []
        for i in range(60):  # Enough for analysis
            op_data = HybridOperationMonitorData(
                operation_id=f"op_{i}",
                operation_name=f"operation_{i}",
                start_time=datetime.utcnow(),
                execution_time_ms=100.0 + i % 20,
                gpu_memory_peak_mb=512.0 + i % 100,
                cpu_usage_percent=50.0 + i % 30,
                memory_usage_mb=1024.0 + i % 200,
                hybrid_mode_used=i % 2 == 0,
                success=True
            )
            op_data.calculate_performance_score()
            self.mock_operation_data.append(op_data)
        
        self.mock_performance_monitor.operation_history = self.mock_operation_data
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        self.assertFalse(self.analyzer.is_initialized)
        
        # Test initialization
        self.analyzer.initialize_analyzer(
            self.mock_hybrid_manager,
            self.mock_switching_manager,
            self.mock_compression_system,
            self.mock_performance_monitor
        )
        
        self.assertTrue(self.analyzer.is_initialized)
        self.assertIsNotNone(self.analyzer.performance_monitor)
    
    def test_bottleneck_analysis(self):
        """Test bottleneck analysis"""
        self.analyzer.initialize_analyzer(
            self.mock_hybrid_manager,
            self.mock_switching_manager,
            self.mock_compression_system,
            self.mock_performance_monitor
        )
        
        # Override the _get_operation_data method to return our mock data
        self.analyzer._get_operation_data = lambda: self.mock_operation_data
        
        bottlenecks = self.analyzer.analyze_hybrid_bottlenecks()
        
        self.assertIsInstance(bottlenecks, list)
        
        # Check bottleneck structure if any found
        for bottleneck in bottlenecks:
            self.assertIsInstance(bottleneck, BottleneckAnalysisResult)
            self.assertIsInstance(bottleneck.bottleneck_type, BottleneckType)
            self.assertGreaterEqual(bottleneck.severity, 0.0)
            self.assertLessEqual(bottleneck.severity, 1.0)
    
    def test_performance_comparison(self):
        """Test hybrid vs pure performance comparison"""
        self.analyzer.initialize_analyzer(
            self.mock_hybrid_manager,
            self.mock_switching_manager,
            self.mock_compression_system,
            self.mock_performance_monitor
        )
        
        # Override the _get_operation_data method
        self.analyzer._get_operation_data = lambda: self.mock_operation_data
        
        comparison = self.analyzer.compare_hybrid_vs_pure_performance()
        
        self.assertIn('hybrid_performance', comparison.__dict__)
        self.assertIn('pure_performance', comparison.__dict__)
        self.assertIn('performance_improvement', comparison.__dict__)
        self.assertIn('hybrid_advantages', comparison.__dict__)
        self.assertIn('pure_advantages', comparison.__dict__)
        
        # Check that we have metrics for both modes
        self.assertGreater(len(comparison.hybrid_performance), 0)
        self.assertGreater(len(comparison.pure_performance), 0)
    
    def test_resource_analysis(self):
        """Test resource usage analysis"""
        self.analyzer.initialize_analyzer(
            self.mock_hybrid_manager,
            self.mock_switching_manager,
            self.mock_compression_system,
            self.mock_performance_monitor
        )
        
        # Override the _get_operation_data method
        self.analyzer._get_operation_data = lambda: self.mock_operation_data
        
        resource_analysis = self.analyzer.analyze_gpu_memory_usage()
        
        self.assertIn('gpu_memory_analysis', resource_analysis.__dict__)
        self.assertIn('cpu_usage_analysis', resource_analysis.__dict__)
        self.assertIn('memory_usage_analysis', resource_analysis.__dict__)
        self.assertIn('resource_efficiency', resource_analysis.__dict__)
        self.assertIn('optimization_opportunities', resource_analysis.__dict__)
    
    def test_optimization_recommendations(self):
        """Test optimization recommendation generation"""
        self.analyzer.initialize_analyzer(
            self.mock_hybrid_manager,
            self.mock_switching_manager,
            self.mock_compression_system,
            self.mock_performance_monitor
        )
        
        # Override methods to return mock data
        self.analyzer._get_operation_data = lambda: self.mock_operation_data
        
        # Run analysis to populate caches
        self.analyzer.analyze_hybrid_bottlenecks()
        self.analyzer.compare_hybrid_vs_pure_performance()
        
        recommendations = self.analyzer.generate_optimization_recommendations()
        
        self.assertIsInstance(recommendations, list)
        
        # Check recommendation structure
        for rec in recommendations:
            self.assertIn('recommendation_id', rec.__dict__)
            self.assertIn('title', rec.__dict__)
            self.assertIn('expected_improvement', rec.__dict__)
            self.assertGreaterEqual(rec.expected_improvement, 0.0)
            self.assertLessEqual(rec.expected_improvement, 1.0)
    
    def test_performance_prediction(self):
        """Test performance impact prediction"""
        self.analyzer.initialize_analyzer(
            self.mock_hybrid_manager,
            self.mock_switching_manager,
            self.mock_compression_system,
            self.mock_performance_monitor
        )
        
        # Override the _get_operation_data method
        self.analyzer._get_operation_data = lambda: self.mock_operation_data
        
        changes = {
            'gpu_memory_threshold': 1024.0,
            'switching_frequency': 0.5
        }
        
        prediction = self.analyzer.predict_performance_impact(changes)
        
        self.assertIn('predicted_metrics', prediction.__dict__)
        self.assertIn('confidence_intervals', prediction.__dict__)
        self.assertIn('prediction_accuracy', prediction.__dict__)
        self.assertIn('recommendations', prediction.__dict__)
        
        self.assertGreaterEqual(prediction.prediction_accuracy, 0.0)
        self.assertLessEqual(prediction.prediction_accuracy, 1.0)
    
    def tearDown(self):
        """Clean up test environment"""
        if self.analyzer and self.analyzer.is_initialized:
            self.analyzer.shutdown()


class TestHybridPerformanceTuner(unittest.TestCase):
    """Test cases for HybridPerformanceTuner"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = HybridTuningConfig(
            max_tuning_iterations=10,  # Small for testing
            patience=3
        )
        self.tuner = HybridPerformanceTuner(self.config)
        
        # Create mock components
        self.mock_hybrid_manager = Mock(spec=HybridPadicManager)
        self.mock_switching_manager = Mock(spec=DynamicSwitchingManager)
        self.mock_compression_system = Mock(spec=HybridPadicCompressionSystem)
        self.mock_performance_monitor = Mock(spec=HybridPerformanceMonitor)
        self.mock_performance_analyzer = Mock(spec=HybridPerformanceAnalyzer)
        
        # Mock performance metrics
        self.mock_performance_monitor.get_hybrid_performance_metrics.return_value = {
            'operation_metrics': {'total_operations': 100, 'success_rate': 0.95},
            'performance_metrics': {'average_execution_time_ms': 100.0},
            'resource_metrics': {'average_gpu_memory_mb': 512.0, 'average_cpu_usage': 50.0},
            'switching_metrics': {'average_switching_overhead_ms': 10.0}
        }
    
    def test_tuner_initialization(self):
        """Test tuner initialization"""
        self.assertFalse(self.tuner.is_initialized)
        
        # Test initialization
        self.tuner.initialize_tuner(
            self.mock_hybrid_manager,
            self.mock_switching_manager,
            self.mock_compression_system,
            self.mock_performance_monitor,
            self.mock_performance_analyzer
        )
        
        self.assertTrue(self.tuner.is_initialized)
        self.assertGreater(len(self.tuner.tuning_parameters), 0)
    
    def test_tuning_parameter_creation(self):
        """Test tuning parameter creation and validation"""
        param = TuningParameter(
            name="test_param",
            current_value=50.0,
            min_value=10.0,
            max_value=100.0,
            step_size=5.0,
            parameter_type="continuous"
        )
        
        self.assertEqual(param.name, "test_param")
        self.assertEqual(param.current_value, 50.0)
        
        # Test value proposal
        new_value = param.propose_new_value(TuningStrategy.BALANCED)
        self.assertGreaterEqual(new_value, param.min_value)
        self.assertLessEqual(new_value, param.max_value)
    
    def test_invalid_tuning_parameter(self):
        """Test invalid tuning parameter creation"""
        with self.assertRaises(ValueError):
            TuningParameter(
                name="",  # Empty name
                current_value=50.0,
                min_value=10.0,
                max_value=100.0,
                step_size=5.0,
                parameter_type="continuous"
            )
        
        with self.assertRaises(ValueError):
            TuningParameter(
                name="test",
                current_value=150.0,  # Outside range
                min_value=10.0,
                max_value=100.0,
                step_size=5.0,
                parameter_type="continuous"
            )
    
    def test_switching_threshold_optimization(self):
        """Test switching threshold optimization"""
        self.tuner.initialize_tuner(
            self.mock_hybrid_manager,
            self.mock_switching_manager,
            self.mock_compression_system,
            self.mock_performance_monitor,
            self.mock_performance_analyzer
        )
        
        result = self.tuner.optimize_switching_thresholds()
        
        self.assertIn('optimized_parameters', result)
        self.assertIn('performance_improvement', result)
        self.assertIn('optimization_score', result)
        
        self.assertIsInstance(result['optimized_parameters'], dict)
        self.assertGreaterEqual(result['optimization_score'], 0.0)
    
    def test_gpu_memory_tuning(self):
        """Test GPU memory allocation tuning"""
        self.tuner.initialize_tuner(
            self.mock_hybrid_manager,
            self.mock_switching_manager,
            self.mock_compression_system,
            self.mock_performance_monitor,
            self.mock_performance_analyzer
        )
        
        result = self.tuner.tune_gpu_memory_allocation()
        
        self.assertIn('best_strategy', result)
        self.assertIn('performance_score', result)
        self.assertIn('memory_efficiency_improvement', result)
        
        if result['best_strategy']:
            self.assertIn('strategy', result['best_strategy'])
            self.assertIn('base_allocation', result['best_strategy'])
    
    def test_tuning_recommendations(self):
        """Test tuning recommendation generation"""
        self.tuner.initialize_tuner(
            self.mock_hybrid_manager,
            self.mock_switching_manager,
            self.mock_compression_system,
            self.mock_performance_monitor,
            self.mock_performance_analyzer
        )
        
        recommendations = self.tuner.get_tuning_recommendations()
        
        self.assertIsInstance(recommendations, list)
        
        # Should have some recommendations based on mock metrics
        self.assertGreater(len(recommendations), 0)
        
        for rec in recommendations:
            self.assertIsInstance(rec, str)
            self.assertGreater(len(rec), 0)
    
    def tearDown(self):
        """Clean up test environment"""
        if self.tuner and self.tuner.is_initialized:
            self.tuner.shutdown()


class TestIntegratedPerformanceOptimization(unittest.TestCase):
    """Test cases for integrated performance optimization workflow"""
    
    def setUp(self):
        """Set up integrated test environment"""
        # Create configurations
        self.opt_config = HybridOptimizationConfig()
        self.mon_config = HybridMonitoringConfig(monitoring_interval_seconds=0.1)
        self.ana_config = HybridAnalysisConfig()
        self.tun_config = HybridTuningConfig(max_tuning_iterations=5)
        
        # Create components
        self.optimizer = HybridPerformanceOptimizer(self.opt_config)
        self.monitor = HybridPerformanceMonitor(self.mon_config)
        self.analyzer = HybridPerformanceAnalyzer(self.ana_config)
        self.tuner = HybridPerformanceTuner(self.tun_config)
        
        # Create mock system components
        self.mock_base_optimizer = Mock(spec=PerformanceOptimizer)
        self.mock_hybrid_manager = Mock(spec=HybridPadicManager)
        self.mock_switching_manager = Mock(spec=DynamicSwitchingManager)
        self.mock_compression_system = Mock(spec=HybridPadicCompressionSystem)
        
        # Mock compression system stats
        self.mock_compression_system.performance_stats = {
            'total_compressions': 100,
            'average_compression_time': 50.0
        }
    
    def test_integrated_workflow(self):
        """Test complete integrated performance optimization workflow"""
        # Initialize all components
        self.optimizer.initialize_hybrid_optimization(
            self.mock_base_optimizer,
            self.mock_hybrid_manager,
            self.mock_switching_manager,
            self.mock_compression_system
        )
        
        self.monitor.start_hybrid_monitoring(
            self.mock_hybrid_manager,
            self.mock_switching_manager,
            self.mock_compression_system
        )
        
        self.analyzer.initialize_analyzer(
            self.mock_hybrid_manager,
            self.mock_switching_manager,
            self.mock_compression_system,
            self.monitor
        )
        
        self.tuner.initialize_tuner(
            self.mock_hybrid_manager,
            self.mock_switching_manager,
            self.mock_compression_system,
            self.monitor,
            self.analyzer
        )
        
        # Verify all components are initialized
        self.assertTrue(self.optimizer.is_initialized)
        self.assertTrue(self.monitor.is_monitoring)
        self.assertTrue(self.analyzer.is_initialized)
        self.assertTrue(self.tuner.is_initialized)
        
        # Test workflow steps
        
        # 1. Monitor some operations
        for i in range(10):
            op_id = self.monitor.monitor_hybrid_operation(
                f"workflow_op_{i}",
                {"hybrid_mode": i % 2 == 0, "data_size": 1000 + i}
            )
            time.sleep(0.01)  # Simulate work
            self.monitor.complete_operation_monitoring(op_id, success=True)
        
        # 2. Get performance metrics
        metrics = self.monitor.get_hybrid_performance_metrics()
        self.assertEqual(metrics['operation_metrics']['total_operations'], 10)
        
        # 3. Perform analysis (with mock data)
        self.analyzer._get_operation_data = lambda: list(self.monitor.operation_history)
        
        # 4. Get tuning recommendations
        recommendations = self.tuner.get_tuning_recommendations()
        self.assertIsInstance(recommendations, list)
        
        # 5. Optimize specific components
        switching_result = self.tuner.optimize_switching_thresholds()
        self.assertIn('optimized_parameters', switching_result)
        
        memory_result = self.tuner.tune_gpu_memory_allocation()
        self.assertIn('best_strategy', memory_result)
    
    def test_component_interaction(self):
        """Test interaction between performance optimization components"""
        # Initialize components with cross-references
        self.optimizer.initialize_hybrid_optimization(
            self.mock_base_optimizer,
            self.mock_hybrid_manager,
            self.mock_switching_manager,
            self.mock_compression_system
        )
        
        self.monitor.start_hybrid_monitoring(self.mock_hybrid_manager)
        
        # Test that optimizer can use monitor data
        self.optimizer.performance_monitor = self.monitor
        
        # Add operation to monitor
        op_id = self.monitor.monitor_hybrid_operation(
            "interaction_test",
            {"hybrid_mode": True, "data_size": 1000}
        )
        self.monitor.complete_operation_monitoring(op_id, success=True)
        
        # Verify optimizer can access monitor data
        self.assertGreater(len(self.monitor.operation_history), 0)
        
        # Test optimizer using monitor data for optimization
        result = self.optimizer.optimize_hybrid_compression_performance()
        self.assertIn('status', result)
    
    def test_error_handling_in_integration(self):
        """Test error handling in integrated workflow"""
        # Test uninitialized component access
        with self.assertRaises(RuntimeError):
            self.analyzer.analyze_hybrid_bottlenecks()
        
        with self.assertRaises(RuntimeError):
            self.tuner.tune_hybrid_parameters()
        
        # Test invalid operation monitoring
        self.monitor.start_hybrid_monitoring(self.mock_hybrid_manager)
        
        with self.assertRaises(ValueError):
            self.monitor.monitor_hybrid_operation("", {})  # Empty name
        
        with self.assertRaises(TypeError):
            self.monitor.monitor_hybrid_operation("test", "invalid_data")  # Invalid data type
    
    def tearDown(self):
        """Clean up integrated test environment"""
        if self.optimizer.is_initialized:
            self.optimizer.shutdown()
        if self.monitor.is_monitoring:
            self.monitor.stop_hybrid_monitoring()
        if self.analyzer.is_initialized:
            self.analyzer.shutdown()
        if self.tuner.is_initialized:
            self.tuner.shutdown()


def run_hybrid_performance_optimization_tests():
    """Run all hybrid performance optimization test suites"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestHybridOptimizationConfig,
        TestHybridPerformanceOptimizer,
        TestHybridPerformanceMonitor,
        TestHybridPerformanceAnalyzer,
        TestHybridPerformanceTuner,
        TestIntegratedPerformanceOptimization
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
    
    # Run hybrid performance optimization tests
    success = run_hybrid_performance_optimization_tests()
    
    print(f"\nHybrid performance optimization tests {'PASSED' if success else 'FAILED'}")
    exit(0 if success else 1)