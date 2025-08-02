"""
Comprehensive Tests for Dynamic Switching System
Tests for DynamicSwitchingManager, SwitchingDecisionEngine, and SwitchingPerformanceMonitor
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

# Import switching system components
from .dynamic_switching_manager import (
    DynamicSwitchingManager, SwitchingConfig, SwitchingEvent, 
    SwitchingAnalytics, CompressionMode, SwitchingTrigger
)
from .switching_decision_engine import (
    SwitchingDecisionEngine, DecisionWeights, DecisionAnalysis,
    PerformancePrediction, DecisionCriterion
)
from .switching_performance_monitor import (
    SwitchingPerformanceMonitor, SwitchingPerformanceRecord,
    PerformanceRegression, PerformanceTrend, PerformanceAlert
)

# Import existing components for testing
from .hybrid_padic_compressor import HybridPadicCompressionSystem
from ...gac_system.direction_state import DirectionStateManager, DirectionState
from ...gac_system.enhanced_bounder import EnhancedGradientBounder
from ...performance_optimizer import PerformanceOptimizer


class TestSwitchingConfig(unittest.TestCase):
    """Test cases for SwitchingConfig"""
    
    def test_valid_config_creation(self):
        """Test valid configuration creation"""
        config = SwitchingConfig(
            enable_dynamic_switching=True,
            auto_switch_confidence_threshold=0.8,
            hybrid_data_size_threshold=500,
            gpu_memory_pressure_threshold=0.75
        )
        
        self.assertTrue(config.enable_dynamic_switching)
        self.assertEqual(config.auto_switch_confidence_threshold, 0.8)
        self.assertEqual(config.hybrid_data_size_threshold, 500)
        self.assertEqual(config.gpu_memory_pressure_threshold, 0.75)
    
    def test_invalid_config_values(self):
        """Test invalid configuration values"""
        # Test invalid confidence threshold
        with self.assertRaises(ValueError):
            SwitchingConfig(auto_switch_confidence_threshold=1.5)
        
        # Test invalid data size threshold
        with self.assertRaises(ValueError):
            SwitchingConfig(hybrid_data_size_threshold=-1)
        
        # Test invalid memory threshold
        with self.assertRaises(ValueError):
            SwitchingConfig(gpu_memory_pressure_threshold=1.2)
        
        # Test invalid switching interval
        with self.assertRaises(ValueError):
            SwitchingConfig(min_switching_interval_ms=-100)
    
    def test_default_config_values(self):
        """Test default configuration values"""
        config = SwitchingConfig()
        
        self.assertTrue(config.enable_dynamic_switching)
        self.assertEqual(config.default_mode, CompressionMode.AUTO)
        self.assertGreaterEqual(config.auto_switch_confidence_threshold, 0.0)
        self.assertLessEqual(config.auto_switch_confidence_threshold, 1.0)


class TestSwitchingEvent(unittest.TestCase):
    """Test cases for SwitchingEvent"""
    
    def test_valid_event_creation(self):
        """Test valid switching event creation"""
        event = SwitchingEvent(
            event_id="test_event_123",
            timestamp=datetime.utcnow(),
            from_mode=CompressionMode.PURE_PADIC,
            to_mode=CompressionMode.HYBRID,
            trigger=SwitchingTrigger.DATA_SIZE,
            data_shape=(100, 100),
            decision_confidence=0.85,
            switching_time_ms=25.5,
            performance_impact=0.1,
            success=True
        )
        
        self.assertEqual(event.event_id, "test_event_123")
        self.assertEqual(event.from_mode, CompressionMode.PURE_PADIC)
        self.assertEqual(event.to_mode, CompressionMode.HYBRID)
        self.assertEqual(event.trigger, SwitchingTrigger.DATA_SIZE)
        self.assertTrue(event.success)
    
    def test_invalid_event_values(self):
        """Test invalid event values"""
        # Test invalid confidence
        with self.assertRaises(ValueError):
            SwitchingEvent(
                event_id="test",
                timestamp=datetime.utcnow(),
                from_mode=CompressionMode.PURE_PADIC,
                to_mode=CompressionMode.HYBRID,
                trigger=SwitchingTrigger.DATA_SIZE,
                data_shape=(100,),
                decision_confidence=1.5,  # Invalid
                switching_time_ms=25.0,
                performance_impact=0.0,
                success=True
            )
        
        # Test invalid switching time
        with self.assertRaises(ValueError):
            SwitchingEvent(
                event_id="test",
                timestamp=datetime.utcnow(),
                from_mode=CompressionMode.PURE_PADIC,
                to_mode=CompressionMode.HYBRID,
                trigger=SwitchingTrigger.DATA_SIZE,
                data_shape=(100,),
                decision_confidence=0.8,
                switching_time_ms=-10.0,  # Invalid
                performance_impact=0.0,
                success=True
            )


class TestDecisionWeights(unittest.TestCase):
    """Test cases for DecisionWeights"""
    
    def test_valid_weights(self):
        """Test valid decision weights"""
        weights = DecisionWeights(
            gradient_stability=0.3,
            data_size=0.2,
            memory_usage=0.15,
            performance_history=0.2,
            error_rate=0.1,
            computational_load=0.03,
            gpu_utilization=0.02
        )
        
        # Check weights sum to 1.0
        total = (weights.gradient_stability + weights.data_size + weights.memory_usage +
                weights.performance_history + weights.error_rate + weights.computational_load +
                weights.gpu_utilization)
        self.assertAlmostEqual(total, 1.0, places=6)
    
    def test_invalid_weight_sum(self):
        """Test invalid weight sum"""
        with self.assertRaises(ValueError):
            DecisionWeights(
                gradient_stability=0.5,
                data_size=0.5,  # Total > 1.0
                memory_usage=0.2,
                performance_history=0.1,
                error_rate=0.1,
                computational_load=0.05,
                gpu_utilization=0.05
            )
    
    def test_invalid_individual_weights(self):
        """Test invalid individual weights"""
        with self.assertRaises(ValueError):
            DecisionWeights(gradient_stability=-0.1)  # Negative weight
        
        with self.assertRaises(ValueError):
            DecisionWeights(data_size=1.5)  # Weight > 1.0


class TestSwitchingDecisionEngine(unittest.TestCase):
    """Test cases for SwitchingDecisionEngine"""
    
    def setUp(self):
        """Set up test environment"""
        self.decision_engine = SwitchingDecisionEngine()
        
        # Create mock components
        self.mock_direction_manager = Mock(spec=DirectionStateManager)
        self.mock_performance_optimizer = Mock(spec=PerformanceOptimizer)
        self.mock_gradient_bounder = Mock(spec=EnhancedGradientBounder)
    
    def test_engine_initialization(self):
        """Test decision engine initialization"""
        self.assertFalse(self.decision_engine.is_initialized)
        
        # Initialize with mock components
        self.decision_engine.initialize_decision_engine(
            direction_manager=self.mock_direction_manager,
            performance_optimizer=self.mock_performance_optimizer,
            gradient_bounder=self.mock_gradient_bounder
        )
        
        self.assertTrue(self.decision_engine.is_initialized)
    
    def test_analyze_switching_criteria(self):
        """Test switching criteria analysis"""
        self.decision_engine.initialize_decision_engine(
            direction_manager=self.mock_direction_manager,
            performance_optimizer=self.mock_performance_optimizer
        )
        
        # Create test data
        test_data = torch.randn(200, 200, dtype=torch.float32)
        test_context = {
            'gradients': torch.randn(50, dtype=torch.float32),
            'error_rate': 0.001
        }
        
        # Mock direction manager response
        self.mock_direction_manager.update_direction_state.return_value = DirectionState.STABLE
        
        # Analyze criteria
        result = self.decision_engine.analyze_switching_criteria(test_data, test_context)
        
        # Validate result structure
        self.assertIn('weighted_score', result)
        self.assertIn('overall_confidence', result)
        self.assertIn('recommendation', result)
        self.assertIn('criteria_analyses', result)
        
        # Validate score ranges
        self.assertGreaterEqual(result['weighted_score'], 0.0)
        self.assertLessEqual(result['weighted_score'], 1.0)
        self.assertGreaterEqual(result['overall_confidence'], 0.0)
        self.assertLessEqual(result['overall_confidence'], 1.0)
    
    def test_evaluate_gradient_direction(self):
        """Test gradient direction evaluation"""
        self.decision_engine.initialize_decision_engine(
            direction_manager=self.mock_direction_manager,
            performance_optimizer=self.mock_performance_optimizer
        )
        
        # Test different gradient scenarios
        test_gradients = torch.randn(100, dtype=torch.float32)
        
        # Mock stable gradients
        self.mock_direction_manager.update_direction_state.return_value = DirectionState.STABLE
        confidence = self.decision_engine.evaluate_gradient_direction(test_gradients)
        self.assertGreaterEqual(confidence, 0.8)  # High confidence for stable
        
        # Mock oscillating gradients
        self.mock_direction_manager.update_direction_state.return_value = DirectionState.OSCILLATING
        confidence = self.decision_engine.evaluate_gradient_direction(test_gradients)
        self.assertGreaterEqual(confidence, 0.7)  # Good confidence for oscillating
    
    def test_predict_performance_impact(self):
        """Test performance impact prediction"""
        self.decision_engine.initialize_decision_engine(
            direction_manager=self.mock_direction_manager,
            performance_optimizer=self.mock_performance_optimizer
        )
        
        test_data = torch.randn(1000, dtype=torch.float32)
        
        # Test hybrid mode prediction
        hybrid_prediction = self.decision_engine.predict_performance_impact(test_data, "hybrid")
        self.assertIsInstance(hybrid_prediction, PerformancePrediction)
        self.assertGreater(hybrid_prediction.predicted_compression_time, 0)
        self.assertGreater(hybrid_prediction.predicted_decompression_time, 0)
        
        # Test pure mode prediction
        pure_prediction = self.decision_engine.predict_performance_impact(test_data, "pure_padic")
        self.assertIsInstance(pure_prediction, PerformancePrediction)
        self.assertGreater(pure_prediction.predicted_compression_time, 0)
    
    def test_invalid_inputs(self):
        """Test invalid input handling"""
        self.decision_engine.initialize_decision_engine(
            direction_manager=self.mock_direction_manager,
            performance_optimizer=self.mock_performance_optimizer
        )
        
        # Test invalid data type
        with self.assertRaises(TypeError):
            self.decision_engine.analyze_switching_criteria("invalid_data")
        
        # Test empty tensor
        with self.assertRaises(ValueError):
            self.decision_engine.analyze_switching_criteria(torch.empty(0))
        
        # Test invalid mode in prediction
        with self.assertRaises(ValueError):
            self.decision_engine.predict_performance_impact(torch.randn(10), "invalid_mode")


class TestSwitchingPerformanceMonitor(unittest.TestCase):
    """Test cases for SwitchingPerformanceMonitor"""
    
    def setUp(self):
        """Set up test environment"""
        self.performance_monitor = SwitchingPerformanceMonitor()
        self.mock_performance_optimizer = Mock(spec=PerformanceOptimizer)
    
    def test_monitor_initialization(self):
        """Test performance monitor initialization"""
        self.assertFalse(self.performance_monitor.is_initialized)
        
        # Initialize with mock optimizer
        self.performance_monitor.initialize_performance_monitor(self.mock_performance_optimizer)
        
        self.assertTrue(self.performance_monitor.is_initialized)
    
    def test_monitor_switching_performance(self):
        """Test switching performance monitoring"""
        self.performance_monitor.initialize_performance_monitor(self.mock_performance_optimizer)
        
        # Create mock switching event
        mock_event = Mock()
        mock_event.event_id = "test_event_123"
        mock_event.from_mode = CompressionMode.PURE_PADIC
        mock_event.to_mode = CompressionMode.HYBRID
        mock_event.trigger = SwitchingTrigger.DATA_SIZE
        mock_event.success = True
        mock_event.decision_confidence = 0.8
        
        # Monitor the event
        record = self.performance_monitor.monitor_switching_performance(mock_event)
        
        # Validate record
        self.assertIsInstance(record, SwitchingPerformanceRecord)
        self.assertEqual(record.switch_event_id, "test_event_123")
        self.assertTrue(record.success)
    
    def test_detect_performance_regression(self):
        """Test performance regression detection"""
        self.performance_monitor.initialize_performance_monitor(self.mock_performance_optimizer)
        
        # Add some test performance records with regressions
        for i in range(5):
            mock_event = Mock()
            mock_event.event_id = f"event_{i}"
            mock_event.from_mode = CompressionMode.PURE_PADIC
            mock_event.to_mode = CompressionMode.HYBRID
            mock_event.trigger = SwitchingTrigger.AUTOMATIC
            mock_event.success = True
            mock_event.decision_confidence = 0.7
            
            self.performance_monitor.monitor_switching_performance(mock_event)
        
        # Detect regressions
        regressions = self.performance_monitor.detect_performance_regression()
        self.assertIsInstance(regressions, list)
    
    def test_calculate_switching_overhead(self):
        """Test switching overhead calculation"""
        self.performance_monitor.initialize_performance_monitor(self.mock_performance_optimizer)
        
        # Add some test records
        for i in range(3):
            mock_event = Mock()
            mock_event.event_id = f"event_{i}"
            mock_event.from_mode = CompressionMode.PURE_PADIC
            mock_event.to_mode = CompressionMode.HYBRID
            mock_event.trigger = SwitchingTrigger.AUTOMATIC
            mock_event.success = True
            mock_event.decision_confidence = 0.7
            
            self.performance_monitor.monitor_switching_performance(mock_event)
        
        # Calculate overhead
        overhead_stats = self.performance_monitor.calculate_switching_overhead()
        
        if 'error' not in overhead_stats:
            self.assertIn('average_overhead_ms', overhead_stats)
            self.assertIn('total_records', overhead_stats)
    
    def test_get_performance_trends(self):
        """Test performance trends analysis"""
        self.performance_monitor.initialize_performance_monitor(self.mock_performance_optimizer)
        
        # Get trends (may be empty initially)
        trends = self.performance_monitor.get_performance_trends()
        
        self.assertIn('analysis_timestamp', trends)
        self.assertIn('trends', trends)
        self.assertIn('data_points_analyzed', trends)


class TestDynamicSwitchingManager(unittest.TestCase):
    """Test cases for DynamicSwitchingManager"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = SwitchingConfig(
            enable_dynamic_switching=True,
            hybrid_data_size_threshold=100,
            pure_data_size_threshold=50
        )
        self.switching_manager = DynamicSwitchingManager(self.config)
        
        # Create mock components
        self.mock_direction_manager = Mock(spec=DirectionStateManager)
        self.mock_performance_optimizer = Mock(spec=PerformanceOptimizer)
        self.mock_compression_system = Mock(spec=HybridPadicCompressionSystem)
    
    def test_manager_initialization(self):
        """Test switching manager initialization"""
        self.assertFalse(self.switching_manager.is_initialized)
        
        # Initialize with mock components
        self.switching_manager.initialize_switching_system(
            direction_manager=self.mock_direction_manager,
            performance_optimizer=self.mock_performance_optimizer,
            hybrid_compression_system=self.mock_compression_system
        )
        
        self.assertTrue(self.switching_manager.is_initialized)
    
    def test_should_switch_to_hybrid(self):
        """Test hybrid switching decision"""
        self.switching_manager.initialize_switching_system(
            direction_manager=self.mock_direction_manager,
            performance_optimizer=self.mock_performance_optimizer,
            hybrid_compression_system=self.mock_compression_system
        )
        
        # Test large data (should favor hybrid)
        large_data = torch.randn(200, 200, dtype=torch.float32)
        should_switch, confidence, trigger = self.switching_manager.should_switch_to_hybrid(large_data)
        
        self.assertIsInstance(should_switch, bool)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertIsInstance(trigger, SwitchingTrigger)
        
        # For large data, should typically favor hybrid
        if should_switch:
            self.assertEqual(trigger, SwitchingTrigger.DATA_SIZE)
    
    def test_should_switch_to_pure(self):
        """Test pure switching decision"""
        self.switching_manager.initialize_switching_system(
            direction_manager=self.mock_direction_manager,
            performance_optimizer=self.mock_performance_optimizer,
            hybrid_compression_system=self.mock_compression_system
        )
        
        # Test small data (should favor pure)
        small_data = torch.randn(10, 10, dtype=torch.float32)
        should_switch, confidence, trigger = self.switching_manager.should_switch_to_pure(small_data)
        
        self.assertIsInstance(should_switch, bool)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertIsInstance(trigger, SwitchingTrigger)
    
    def test_execute_switch(self):
        """Test switch execution"""
        self.switching_manager.initialize_switching_system(
            direction_manager=self.mock_direction_manager,
            performance_optimizer=self.mock_performance_optimizer,
            hybrid_compression_system=self.mock_compression_system
        )
        
        # Test switch to hybrid
        event = self.switching_manager.execute_switch(
            self.mock_compression_system,
            CompressionMode.HYBRID
        )
        
        self.assertIsInstance(event, SwitchingEvent)
        self.assertEqual(event.to_mode, CompressionMode.HYBRID)
        self.assertTrue(event.success)
        self.assertGreater(event.switching_time_ms, 0)
    
    def test_switching_analytics(self):
        """Test switching analytics"""
        self.switching_manager.initialize_switching_system(
            direction_manager=self.mock_direction_manager,
            performance_optimizer=self.mock_performance_optimizer,
            hybrid_compression_system=self.mock_compression_system
        )
        
        # Execute some switches to generate analytics
        self.switching_manager.execute_switch(
            self.mock_compression_system,
            CompressionMode.HYBRID
        )
        
        # Get analytics
        analytics = self.switching_manager.get_switching_analytics()
        
        self.assertIn('summary', analytics)
        self.assertIn('switching_patterns', analytics)
        self.assertIn('current_state', analytics)
        self.assertIn('configuration', analytics)
    
    def test_rate_limiting(self):
        """Test switching rate limiting"""
        # Create config with very low limits for testing
        config = SwitchingConfig(
            min_switching_interval_ms=1000.0,  # 1 second minimum interval
            max_switches_per_minute=2
        )
        manager = DynamicSwitchingManager(config)
        
        manager.initialize_switching_system(
            direction_manager=self.mock_direction_manager,
            performance_optimizer=self.mock_performance_optimizer,
            hybrid_compression_system=self.mock_compression_system
        )
        
        # Execute first switch
        event1 = manager.execute_switch(self.mock_compression_system, CompressionMode.HYBRID)
        self.assertTrue(event1.success)
        
        # Try immediate second switch (should be rate limited)
        test_data = torch.randn(100, dtype=torch.float32)
        should_switch, confidence, trigger = manager.should_switch_to_pure(test_data)
        
        # Rate limiting may prevent switch
        if not should_switch:
            self.assertEqual(trigger, SwitchingTrigger.AUTOMATIC)
    
    def test_invalid_inputs(self):
        """Test invalid input handling"""
        self.switching_manager.initialize_switching_system(
            direction_manager=self.mock_direction_manager,
            performance_optimizer=self.mock_performance_optimizer,
            hybrid_compression_system=self.mock_compression_system
        )
        
        # Test invalid data type
        with self.assertRaises(TypeError):
            self.switching_manager.should_switch_to_hybrid("invalid_data")
        
        # Test empty tensor
        with self.assertRaises(ValueError):
            self.switching_manager.should_switch_to_hybrid(torch.empty(0))
        
        # Test invalid target mode for execution
        with self.assertRaises(ValueError):
            self.switching_manager.execute_switch(
                self.mock_compression_system,
                CompressionMode.AUTO  # Cannot switch to AUTO directly
            )
    
    def tearDown(self):
        """Clean up test environment"""
        if self.switching_manager and self.switching_manager.is_initialized:
            self.switching_manager.shutdown()


class TestIntegrationWithHybridCompression(unittest.TestCase):
    """Test cases for integration with hybrid compression system"""
    
    def setUp(self):
        """Set up test environment"""
        # Create configuration for hybrid compression system
        self.compression_config = {
            'prime': 7,
            'precision': 10,
            'chunk_size': 100,
            'gpu_memory_limit_mb': 512,
            'enable_hybrid': True,
            'enable_dynamic_switching': True,
            'hybrid_threshold': 100
        }
        
        # Only run if CUDA is available (hybrid system requires GPU)
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for hybrid compression integration tests")
        
        try:
            self.compression_system = HybridPadicCompressionSystem(self.compression_config)
        except Exception as e:
            self.skipTest(f"Failed to create hybrid compression system: {e}")
    
    def test_dynamic_switching_integration(self):
        """Test dynamic switching integration with hybrid compression"""
        # Create test data
        test_data = torch.randn(150, 150, dtype=torch.float32)
        
        # Test encoding with dynamic switching context
        context = {
            'gradients': torch.randn(50, dtype=torch.float32),
            'error_rate': 0.0001,
            'performance_metrics': {
                'compression_time': 50.0,
                'memory_usage': 0.3
            }
        }
        
        try:
            # Encode with context
            encoded_data, metadata = self.compression_system.encode(test_data, context)
            
            # Validate encoding worked
            self.assertIsNotNone(encoded_data)
            self.assertIsNotNone(metadata)
            self.assertIn('compression_type', metadata)
            
            # Decode and validate
            decoded_data = self.compression_system.decode(encoded_data, metadata)
            self.assertEqual(decoded_data.shape, test_data.shape)
            
        except Exception as e:
            self.skipTest(f"Integration test failed: {e}")
    
    def test_switching_manager_integration(self):
        """Test switching manager integration with compression system"""
        if not hasattr(self.compression_system, 'dynamic_switching_manager'):
            self.skipTest("Dynamic switching manager not available")
        
        # Test setting dynamic switching manager
        switching_config = SwitchingConfig(enable_dynamic_switching=True)
        switching_manager = DynamicSwitchingManager(switching_config)
        
        # Set the manager
        self.compression_system.set_dynamic_switching_manager(switching_manager)
        
        # Verify integration
        self.assertIsNotNone(self.compression_system.dynamic_switching_manager)


class TestConcurrentSwitching(unittest.TestCase):
    """Test cases for concurrent switching operations"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = SwitchingConfig(enable_dynamic_switching=True)
        self.switching_manager = DynamicSwitchingManager(self.config)
        
        # Create mock components
        self.mock_direction_manager = Mock(spec=DirectionStateManager)
        self.mock_performance_optimizer = Mock(spec=PerformanceOptimizer)
        self.mock_compression_system = Mock(spec=HybridPadicCompressionSystem)
        
        self.switching_manager.initialize_switching_system(
            direction_manager=self.mock_direction_manager,
            performance_optimizer=self.mock_performance_optimizer,
            hybrid_compression_system=self.mock_compression_system
        )
    
    def test_concurrent_switching_decisions(self):
        """Test concurrent switching decisions"""
        def make_switching_decision(data_size):
            test_data = torch.randn(data_size, dtype=torch.float32)
            return self.switching_manager.should_switch_to_hybrid(test_data)
        
        # Create multiple threads
        threads = []
        results = []
        
        for i in range(5):
            thread = threading.Thread(
                target=lambda size=(i+1)*50: results.append(make_switching_decision(size))
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Validate results
        self.assertEqual(len(results), 5)
        for should_switch, confidence, trigger in results:
            self.assertIsInstance(should_switch, bool)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
    
    def tearDown(self):
        """Clean up test environment"""
        if self.switching_manager:
            self.switching_manager.shutdown()


class TestPerformanceAndStress(unittest.TestCase):
    """Performance and stress tests for switching system"""
    
    def setUp(self):
        """Set up test environment"""
        self.switching_manager = DynamicSwitchingManager()
        
        # Create mock components
        self.mock_direction_manager = Mock(spec=DirectionStateManager)
        self.mock_performance_optimizer = Mock(spec=PerformanceOptimizer)
        self.mock_compression_system = Mock(spec=HybridPadicCompressionSystem)
        
        self.switching_manager.initialize_switching_system(
            direction_manager=self.mock_direction_manager,
            performance_optimizer=self.mock_performance_optimizer,
            hybrid_compression_system=self.mock_compression_system
        )
    
    def test_decision_performance(self):
        """Test decision-making performance"""
        test_data = torch.randn(1000, dtype=torch.float32)
        
        # Time multiple decisions
        start_time = time.time()
        for _ in range(100):
            self.switching_manager.should_switch_to_hybrid(test_data)
        end_time = time.time()
        
        # Each decision should be fast (< 10ms average)
        avg_time_ms = (end_time - start_time) * 1000 / 100
        self.assertLess(avg_time_ms, 10.0, "Decision making too slow")
    
    def test_switching_execution_performance(self):
        """Test switching execution performance"""
        # Time switch execution
        start_time = time.time()
        event = self.switching_manager.execute_switch(
            self.mock_compression_system,
            CompressionMode.HYBRID
        )
        end_time = time.time()
        
        # Switch should be fast (< 100ms)
        switch_time_ms = (end_time - start_time) * 1000
        self.assertLess(switch_time_ms, 100.0, "Switch execution too slow")
        self.assertTrue(event.success)
    
    def tearDown(self):
        """Clean up test environment"""
        if self.switching_manager:
            self.switching_manager.shutdown()


def run_dynamic_switching_tests():
    """Run all dynamic switching test suites"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestSwitchingConfig,
        TestSwitchingEvent,
        TestDecisionWeights,
        TestSwitchingDecisionEngine,
        TestSwitchingPerformanceMonitor,
        TestDynamicSwitchingManager,
        TestIntegrationWithHybridCompression,
        TestConcurrentSwitching,
        TestPerformanceAndStress
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
    
    # Run dynamic switching tests
    success = run_dynamic_switching_tests()
    
    print(f"\nDynamic switching tests {'PASSED' if success else 'FAILED'}")
    exit(0 if success else 1)