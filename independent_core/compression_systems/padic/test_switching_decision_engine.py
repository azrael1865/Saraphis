#!/usr/bin/env python3
"""
Comprehensive Test Suite for SwitchingDecisionEngine
Tests all aspects of the intelligent switching decision system
"""

import unittest
import torch
import numpy as np
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, patch

# Import the module to test
from switching_decision_engine import (
    SwitchingDecisionEngine,
    DecisionCriterion,
    DecisionWeights,
    DecisionAnalysis,
    PerformancePrediction,
    DirectionState,
    DirectionStateManager,
    EnhancedGradientBounder,
    PerformanceOptimizer
)


class TestDecisionWeights(unittest.TestCase):
    """Test DecisionWeights dataclass"""
    
    def test_valid_weights_creation(self):
        """Test creating valid decision weights"""
        weights = DecisionWeights(
            gradient_stability=0.25,
            data_size=0.20,
            memory_usage=0.15,
            performance_history=0.20,
            error_rate=0.10,
            computational_load=0.05,
            gpu_utilization=0.05
        )
        
        # Check all weights are set correctly
        self.assertEqual(weights.gradient_stability, 0.25)
        self.assertEqual(weights.data_size, 0.20)
        self.assertEqual(weights.memory_usage, 0.15)
        self.assertEqual(weights.performance_history, 0.20)
        self.assertEqual(weights.error_rate, 0.10)
        self.assertEqual(weights.computational_load, 0.05)
        self.assertEqual(weights.gpu_utilization, 0.05)
    
    def test_weights_must_sum_to_one(self):
        """Test that weights must sum to 1.0"""
        with self.assertRaises(ValueError) as ctx:
            DecisionWeights(
                gradient_stability=0.3,
                data_size=0.3,
                memory_usage=0.3,
                performance_history=0.2,
                error_rate=0.1,
                computational_load=0.1,
                gpu_utilization=0.1
            )
        self.assertIn("must sum to 1.0", str(ctx.exception))
    
    def test_individual_weight_validation(self):
        """Test individual weight validation"""
        with self.assertRaises(ValueError) as ctx:
            DecisionWeights(
                gradient_stability=-0.1,  # Invalid negative weight
                data_size=0.3,
                memory_usage=0.2,
                performance_history=0.3,
                error_rate=0.2,
                computational_load=0.05,
                gpu_utilization=0.05
            )
        self.assertIn("must be in [0, 1]", str(ctx.exception))
        
        with self.assertRaises(ValueError) as ctx:
            DecisionWeights(
                gradient_stability=1.5,  # Invalid > 1 weight
                data_size=0.0,
                memory_usage=0.0,
                performance_history=0.0,
                error_rate=0.0,
                computational_load=0.0,
                gpu_utilization=-0.5
            )
        self.assertIn("must be in [0, 1]", str(ctx.exception))


class TestDecisionAnalysis(unittest.TestCase):
    """Test DecisionAnalysis dataclass"""
    
    def test_valid_analysis_creation(self):
        """Test creating valid decision analysis"""
        analysis = DecisionAnalysis(
            criterion=DecisionCriterion.GRADIENT_STABILITY,
            score=0.8,
            confidence=0.9,
            reasoning="Test reasoning",
            metadata={'test': 'data'}
        )
        
        self.assertEqual(analysis.criterion, DecisionCriterion.GRADIENT_STABILITY)
        self.assertEqual(analysis.score, 0.8)
        self.assertEqual(analysis.confidence, 0.9)
        self.assertEqual(analysis.reasoning, "Test reasoning")
        self.assertEqual(analysis.metadata, {'test': 'data'})
    
    def test_score_validation(self):
        """Test score validation"""
        with self.assertRaises(ValueError) as ctx:
            DecisionAnalysis(
                criterion=DecisionCriterion.DATA_SIZE,
                score=1.5,  # Invalid score > 1
                confidence=0.9,
                reasoning="Test"
            )
        self.assertIn("Score must be in [0, 1]", str(ctx.exception))
        
        with self.assertRaises(ValueError) as ctx:
            DecisionAnalysis(
                criterion=DecisionCriterion.DATA_SIZE,
                score=-0.1,  # Invalid score < 0
                confidence=0.9,
                reasoning="Test"
            )
        self.assertIn("Score must be in [0, 1]", str(ctx.exception))
    
    def test_confidence_validation(self):
        """Test confidence validation"""
        with self.assertRaises(ValueError) as ctx:
            DecisionAnalysis(
                criterion=DecisionCriterion.MEMORY_USAGE,
                score=0.5,
                confidence=2.0,  # Invalid confidence > 1
                reasoning="Test"
            )
        self.assertIn("Confidence must be in [0, 1]", str(ctx.exception))
    
    def test_type_validation(self):
        """Test type validation"""
        with self.assertRaises(TypeError):
            DecisionAnalysis(
                criterion="invalid",  # Wrong type
                score=0.5,
                confidence=0.5,
                reasoning="Test"
            )
        
        with self.assertRaises(TypeError):
            DecisionAnalysis(
                criterion=DecisionCriterion.ERROR_RATE,
                score=0.5,
                confidence=0.5,
                reasoning=123  # Wrong type for reasoning
            )


class TestPerformancePrediction(unittest.TestCase):
    """Test PerformancePrediction dataclass"""
    
    def test_valid_prediction_creation(self):
        """Test creating valid performance prediction"""
        prediction = PerformancePrediction(
            predicted_compression_time=10.5,
            predicted_decompression_time=5.2,
            predicted_memory_usage=1024,
            predicted_compression_ratio=0.3,
            predicted_error_rate=0.001,
            confidence=0.8,
            prediction_basis="test_model"
        )
        
        self.assertEqual(prediction.predicted_compression_time, 10.5)
        self.assertEqual(prediction.predicted_decompression_time, 5.2)
        self.assertEqual(prediction.predicted_memory_usage, 1024)
        self.assertEqual(prediction.predicted_compression_ratio, 0.3)
        self.assertEqual(prediction.predicted_error_rate, 0.001)
        self.assertEqual(prediction.confidence, 0.8)
        self.assertEqual(prediction.prediction_basis, "test_model")
    
    def test_negative_value_validation(self):
        """Test that negative values are rejected"""
        with self.assertRaises(ValueError) as ctx:
            PerformancePrediction(
                predicted_compression_time=-1.0,  # Invalid negative
                predicted_decompression_time=5.2,
                predicted_memory_usage=1024,
                predicted_compression_ratio=0.3,
                predicted_error_rate=0.001,
                confidence=0.8,
                prediction_basis="test"
            )
        self.assertIn("must be non-negative", str(ctx.exception))
    
    def test_confidence_validation(self):
        """Test confidence validation"""
        with self.assertRaises(ValueError) as ctx:
            PerformancePrediction(
                predicted_compression_time=10.5,
                predicted_decompression_time=5.2,
                predicted_memory_usage=1024,
                predicted_compression_ratio=0.3,
                predicted_error_rate=0.001,
                confidence=1.5,  # Invalid confidence > 1
                prediction_basis="test"
            )
        self.assertIn("Confidence must be in [0, 1]", str(ctx.exception))


class TestSwitchingDecisionEngine(unittest.TestCase):
    """Test SwitchingDecisionEngine main functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'gradient_stability_weight': 0.25,
            'data_size_weight': 0.20,
            'memory_usage_weight': 0.15,
            'performance_history_weight': 0.20,
            'error_rate_weight': 0.10,
            'computational_load_weight': 0.05,
            'gpu_utilization_weight': 0.05,
            'hybrid_threshold': 0.7,
            'pure_threshold': 0.3,
            'confidence_threshold': 0.6,
            'enable_adaptive_thresholds': True,
            'threshold_adaptation_rate': 0.1
        }
        self.engine = SwitchingDecisionEngine(self.config)
    
    def test_initialization(self):
        """Test engine initialization"""
        # Test with config
        self.assertIsNotNone(self.engine)
        self.assertFalse(self.engine.is_initialized)
        self.assertEqual(self.engine.hybrid_threshold, 0.7)
        self.assertEqual(self.engine.pure_threshold, 0.3)
        
        # Test without config
        engine = SwitchingDecisionEngine()
        self.assertIsNotNone(engine)
        self.assertFalse(engine.is_initialized)
        
        # Test with invalid config
        with self.assertRaises(TypeError):
            SwitchingDecisionEngine("invalid_config")
    
    def test_initialize_decision_engine(self):
        """Test initializing decision engine with components"""
        direction_manager = DirectionStateManager()
        performance_optimizer = PerformanceOptimizer()
        gradient_bounder = EnhancedGradientBounder()
        
        # Initialize engine
        self.engine.initialize_decision_engine(
            direction_manager,
            performance_optimizer,
            gradient_bounder
        )
        
        self.assertTrue(self.engine.is_initialized)
        self.assertIsNotNone(self.engine.direction_state_manager)
        self.assertIsNotNone(self.engine.performance_optimizer)
        self.assertIsNotNone(self.engine.gradient_bounder)
        
        # Test re-initialization (should be no-op)
        self.engine.initialize_decision_engine(
            direction_manager,
            performance_optimizer,
            gradient_bounder
        )
        self.assertTrue(self.engine.is_initialized)
    
    def test_initialize_with_invalid_components(self):
        """Test initialization with invalid components"""
        with self.assertRaises(RuntimeError) as ctx:
            self.engine.initialize_decision_engine(
                "invalid",  # Wrong type
                PerformanceOptimizer(),
                None
            )
        self.assertIn("initialization failed", str(ctx.exception))
    
    def test_analyze_switching_criteria_not_initialized(self):
        """Test analyzing criteria when not initialized"""
        data = torch.randn(10, 10)
        
        with self.assertRaises(RuntimeError) as ctx:
            self.engine.analyze_switching_criteria(data)
        self.assertIn("not initialized", str(ctx.exception))
    
    def test_analyze_switching_criteria_invalid_data(self):
        """Test analyzing criteria with invalid data"""
        # Initialize engine first
        self.engine.initialize_decision_engine(
            DirectionStateManager(),
            PerformanceOptimizer(),
            None
        )
        
        # Test with non-tensor data
        with self.assertRaises(TypeError):
            self.engine.analyze_switching_criteria("invalid_data")
        
        # Test with empty tensor
        with self.assertRaises(ValueError):
            self.engine.analyze_switching_criteria(torch.tensor([]))
    
    def test_analyze_switching_criteria_basic(self):
        """Test basic switching criteria analysis"""
        # Initialize engine
        self.engine.initialize_decision_engine(
            DirectionStateManager(),
            PerformanceOptimizer(),
            None
        )
        
        # Test with simple data
        data = torch.randn(100, 100)
        result = self.engine.analyze_switching_criteria(data)
        
        # Check result structure
        self.assertIn('weighted_score', result)
        self.assertIn('overall_confidence', result)
        self.assertIn('recommendation', result)
        self.assertIn('criteria_analyses', result)
        self.assertIn('performance_confidence', result)
        self.assertIn('processing_time_ms', result)
        
        # Check score and confidence ranges
        self.assertGreaterEqual(result['weighted_score'], 0.0)
        self.assertLessEqual(result['weighted_score'], 1.0)
        self.assertGreaterEqual(result['overall_confidence'], 0.0)
        self.assertLessEqual(result['overall_confidence'], 1.0)
        
        # Check recommendation is valid
        self.assertIn(result['recommendation'], [
            'switch_to_hybrid',
            'switch_to_pure',
            'maintain_current',
            'insufficient_confidence'
        ])
    
    def test_analyze_switching_criteria_with_context(self):
        """Test switching criteria analysis with context"""
        # Initialize engine
        self.engine.initialize_decision_engine(
            DirectionStateManager(),
            PerformanceOptimizer(),
            None
        )
        
        # Test with context including gradients
        data = torch.randn(50, 50)
        gradients = torch.randn(50, 50)
        context = {
            'gradients': gradients,
            'error_rate': 0.005,
            'mode': 'hybrid'
        }
        
        result = self.engine.analyze_switching_criteria(data, context)
        
        # Check result
        self.assertIsNotNone(result)
        self.assertIn('criteria_analyses', result)
        
        # Check that gradient analysis was performed
        gradient_analyses = [
            a for a in result['criteria_analyses']
            if a['criterion'] == DecisionCriterion.GRADIENT_STABILITY.value
        ]
        self.assertEqual(len(gradient_analyses), 1)
    
    def test_evaluate_gradient_direction(self):
        """Test gradient direction evaluation"""
        # Initialize engine
        self.engine.initialize_decision_engine(
            DirectionStateManager(),
            PerformanceOptimizer(),
            None
        )
        
        # Test with valid gradients
        gradients = torch.randn(100)
        confidence = self.engine.evaluate_gradient_direction(gradients)
        
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Test with invalid gradients
        with self.assertRaises(TypeError):
            self.engine.evaluate_gradient_direction("invalid")
        
        with self.assertRaises(ValueError):
            self.engine.evaluate_gradient_direction(torch.tensor([]))
    
    def test_evaluate_gradient_direction_with_manager(self):
        """Test gradient direction evaluation with direction state manager"""
        # Create a mock direction manager
        mock_manager = MagicMock(spec=DirectionStateManager)
        mock_manager.update_direction_state.return_value = DirectionState.STABLE
        
        # Initialize engine with mock
        self.engine.initialize_decision_engine(
            mock_manager,
            PerformanceOptimizer(),
            None
        )
        
        # Test stable state
        gradients = torch.randn(100)
        confidence = self.engine.evaluate_gradient_direction(gradients)
        self.assertEqual(confidence, 0.9)  # Stable should return 0.9
        
        # Test oscillating state
        mock_manager.update_direction_state.return_value = DirectionState.OSCILLATING
        confidence = self.engine.evaluate_gradient_direction(gradients)
        self.assertEqual(confidence, 0.8)  # Oscillating should return 0.8
        
        # Test ascending state
        mock_manager.update_direction_state.return_value = DirectionState.ASCENDING
        confidence = self.engine.evaluate_gradient_direction(gradients)
        self.assertEqual(confidence, 0.4)  # Ascending should return 0.4
        
        # Test descending state
        mock_manager.update_direction_state.return_value = DirectionState.DESCENDING
        confidence = self.engine.evaluate_gradient_direction(gradients)
        self.assertEqual(confidence, 0.3)  # Descending should return 0.3
        
        # Test unknown state
        mock_manager.update_direction_state.return_value = DirectionState.UNKNOWN
        confidence = self.engine.evaluate_gradient_direction(gradients)
        self.assertEqual(confidence, 0.5)  # Unknown should return 0.5
    
    def test_predict_performance_impact_hybrid(self):
        """Test performance impact prediction for hybrid mode"""
        # Initialize engine
        self.engine.initialize_decision_engine(
            DirectionStateManager(),
            PerformanceOptimizer(),
            None
        )
        
        # Test with various data sizes
        for size in [10, 100, 1000, 10000]:
            data = torch.randn(size)
            prediction = self.engine.predict_performance_impact(data, 'hybrid')
            
            self.assertIsInstance(prediction, PerformancePrediction)
            self.assertGreater(prediction.predicted_compression_time, 0)
            self.assertGreater(prediction.predicted_decompression_time, 0)
            self.assertGreater(prediction.predicted_memory_usage, 0)
            self.assertGreater(prediction.predicted_compression_ratio, 0)
            self.assertGreaterEqual(prediction.predicted_error_rate, 0)
            self.assertGreaterEqual(prediction.confidence, 0)
            self.assertLessEqual(prediction.confidence, 1)
            self.assertIsNotNone(prediction.prediction_basis)
    
    def test_predict_performance_impact_pure(self):
        """Test performance impact prediction for pure p-adic mode"""
        # Initialize engine
        self.engine.initialize_decision_engine(
            DirectionStateManager(),
            PerformanceOptimizer(),
            None
        )
        
        # Test with various data sizes
        for size in [10, 100, 1000, 10000]:
            data = torch.randn(size)
            prediction = self.engine.predict_performance_impact(data, 'pure_padic')
            
            self.assertIsInstance(prediction, PerformancePrediction)
            self.assertGreater(prediction.predicted_compression_time, 0)
            self.assertGreater(prediction.predicted_decompression_time, 0)
            self.assertGreater(prediction.predicted_memory_usage, 0)
            self.assertGreater(prediction.predicted_compression_ratio, 0)
            self.assertGreaterEqual(prediction.predicted_error_rate, 0)
            self.assertGreaterEqual(prediction.confidence, 0)
            self.assertLessEqual(prediction.confidence, 1)
    
    def test_predict_performance_impact_invalid(self):
        """Test performance impact prediction with invalid parameters"""
        # Initialize engine
        self.engine.initialize_decision_engine(
            DirectionStateManager(),
            PerformanceOptimizer(),
            None
        )
        
        # Test with invalid data
        with self.assertRaises(TypeError):
            self.engine.predict_performance_impact("invalid", 'hybrid')
        
        # Test with invalid mode
        data = torch.randn(100)
        with self.assertRaises(ValueError):
            self.engine.predict_performance_impact(data, 'invalid_mode')
    
    def test_calculate_switching_confidence(self):
        """Test switching confidence calculation"""
        # Initialize engine
        self.engine.initialize_decision_engine(
            DirectionStateManager(),
            PerformanceOptimizer(),
            None
        )
        
        # Test with various data
        data = torch.randn(100, 100)
        confidence = self.engine.calculate_switching_confidence(data)
        
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Test with context
        context = {'error_rate': 0.01, 'mode': 'hybrid'}
        confidence_with_context = self.engine.calculate_switching_confidence(data, context)
        
        self.assertGreaterEqual(confidence_with_context, 0.0)
        self.assertLessEqual(confidence_with_context, 1.0)
        
        # Test with invalid data
        with self.assertRaises(TypeError):
            self.engine.calculate_switching_confidence([1, 2, 3])
    
    def test_get_optimal_switching_threshold(self):
        """Test getting optimal switching thresholds"""
        # Initialize engine
        self.engine.initialize_decision_engine(
            DirectionStateManager(),
            PerformanceOptimizer(),
            None
        )
        
        # Test without history (should return defaults)
        hybrid_thresh, pure_thresh = self.engine.get_optimal_switching_threshold()
        self.assertEqual(hybrid_thresh, 0.7)
        self.assertEqual(pure_thresh, 0.3)
        
        # Add some decision history
        for _ in range(15):
            self.engine.decision_history.append({
                'timestamp': datetime.now(timezone.utc),
                'data_shape': (100, 100),
                'weighted_score': 0.6,
                'overall_confidence': 0.7,
                'recommendation': 'switch_to_hybrid',
                'analyses': [],
                'processing_time_ms': 10.0
            })
        
        # Test with history
        hybrid_thresh, pure_thresh = self.engine.get_optimal_switching_threshold()
        
        # Should be adapted but within valid range
        self.assertGreaterEqual(hybrid_thresh, 0.5)
        self.assertLessEqual(hybrid_thresh, 0.9)
        self.assertGreaterEqual(pure_thresh, 0.1)
        self.assertLessEqual(pure_thresh, 0.5)
    
    def test_update_decision_weights(self):
        """Test updating decision weights"""
        # Initialize engine
        self.engine.initialize_decision_engine(
            DirectionStateManager(),
            PerformanceOptimizer(),
            None
        )
        
        # Test with valid performance data
        performance_data = {
            'performance_score': 0.8,
            'compression_ratio': 0.3,
            'error_rate': 0.001,
            'mode': 'hybrid'
        }
        
        self.engine.update_decision_weights(performance_data)
        
        # Check that data was stored
        self.assertEqual(len(self.engine.performance_history), 1)
        self.assertEqual(self.engine.performance_history[-1]['data'], performance_data)
        
        # Test with invalid data
        with self.assertRaises(TypeError):
            self.engine.update_decision_weights("invalid")
    
    def test_update_configuration(self):
        """Test updating engine configuration"""
        # Initialize engine
        self.engine.initialize_decision_engine(
            DirectionStateManager(),
            PerformanceOptimizer(),
            None
        )
        
        # Update configuration
        new_config = {
            'hybrid_threshold': 0.8,
            'pure_threshold': 0.2,
            'confidence_threshold': 0.7
        }
        
        self.engine.update_configuration(new_config)
        
        self.assertEqual(self.engine.hybrid_threshold, 0.8)
        self.assertEqual(self.engine.pure_threshold, 0.2)
        self.assertEqual(self.engine.confidence_threshold, 0.7)
        
        # Test with invalid config
        with self.assertRaises(TypeError):
            self.engine.update_configuration("invalid")
    
    def test_shutdown(self):
        """Test engine shutdown"""
        # Initialize engine
        self.engine.initialize_decision_engine(
            DirectionStateManager(),
            PerformanceOptimizer(),
            None
        )
        
        # Add some data
        self.engine.performance_history.append({'test': 'data'})
        self.engine.decision_history.append({'test': 'decision'})
        self.engine.gradient_analysis_cache['test'] = 'cache'
        
        # Shutdown
        self.engine.shutdown()
        
        # Check that everything is cleared
        self.assertFalse(self.engine.is_initialized)
        self.assertEqual(len(self.engine.performance_history), 0)
        self.assertEqual(len(self.engine.decision_history), 0)
        self.assertEqual(len(self.engine.gradient_analysis_cache), 0)


class TestAnalysisMethods(unittest.TestCase):
    """Test individual analysis methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = SwitchingDecisionEngine()
        self.engine.initialize_decision_engine(
            DirectionStateManager(),
            PerformanceOptimizer(),
            None
        )
    
    def test_analyze_data_size(self):
        """Test data size analysis"""
        # Test large data
        large_data = torch.randn(3000)
        analysis = self.engine._analyze_data_size(large_data)
        self.assertEqual(analysis.criterion, DecisionCriterion.DATA_SIZE)
        self.assertGreater(analysis.score, 0.8)  # Should favor hybrid
        self.assertEqual(analysis.confidence, 0.9)
        
        # Test medium data
        medium_data = torch.randn(500)
        analysis = self.engine._analyze_data_size(medium_data)
        self.assertAlmostEqual(analysis.score, 0.5, delta=0.2)  # Should be neutral
        
        # Test small data
        small_data = torch.randn(50)
        analysis = self.engine._analyze_data_size(small_data)
        self.assertLess(analysis.score, 0.3)  # Should favor pure
    
    @patch('torch.cuda.is_available')
    def test_analyze_memory_usage_no_gpu(self, mock_cuda):
        """Test memory usage analysis without GPU"""
        mock_cuda.return_value = False
        
        data = torch.randn(100)
        analysis = self.engine._analyze_memory_usage(data)
        
        self.assertEqual(analysis.criterion, DecisionCriterion.MEMORY_USAGE)
        self.assertEqual(analysis.score, 0.4)  # No GPU should favor pure
        self.assertEqual(analysis.confidence, 0.6)
        self.assertEqual(analysis.metadata['gpu_available'], False)
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    def test_analyze_memory_usage_with_gpu(self, mock_allocated, mock_properties, mock_cuda):
        """Test memory usage analysis with GPU"""
        mock_cuda.return_value = True
        mock_properties.return_value = MagicMock(total_memory=1024*1024*1024)  # 1GB
        mock_allocated.return_value = 512*1024*1024  # 512MB (50% usage)
        
        data = torch.randn(100)
        analysis = self.engine._analyze_memory_usage(data)
        
        self.assertEqual(analysis.criterion, DecisionCriterion.MEMORY_USAGE)
        # 50% usage is < 0.6, so it should return 0.7 (low usage allows hybrid)
        self.assertEqual(analysis.score, 0.7)  # Low usage allows hybrid
        self.assertEqual(analysis.confidence, 0.8)
        self.assertIn('gpu_memory_usage_ratio', analysis.metadata)
    
    def test_analyze_performance_history_empty(self):
        """Test performance history analysis with no history"""
        analysis = self.engine._analyze_performance_history()
        
        self.assertEqual(analysis.criterion, DecisionCriterion.PERFORMANCE_HISTORY)
        self.assertEqual(analysis.score, 0.5)  # Neutral with no history
        self.assertEqual(analysis.confidence, 0.1)  # Low confidence
        self.assertEqual(analysis.metadata['history_length'], 0)
    
    def test_analyze_performance_history_with_data(self):
        """Test performance history analysis with data"""
        # Add good performance history
        for _ in range(10):
            self.engine.performance_history.append({
                'timestamp': datetime.now(timezone.utc),
                'data': {'performance_score': 0.9}
            })
        
        analysis = self.engine._analyze_performance_history()
        
        self.assertEqual(analysis.criterion, DecisionCriterion.PERFORMANCE_HISTORY)
        self.assertLess(analysis.score, 0.4)  # Good performance should not switch
        self.assertGreater(analysis.confidence, 0.7)
        self.assertGreater(analysis.metadata['avg_performance'], 0.8)
        
        # Clear and add poor performance history
        self.engine.performance_history.clear()
        for _ in range(10):
            self.engine.performance_history.append({
                'timestamp': datetime.now(timezone.utc),
                'data': {'performance_score': 0.3}
            })
        
        analysis = self.engine._analyze_performance_history()
        self.assertGreater(analysis.score, 0.7)  # Poor performance should switch
    
    def test_analyze_error_rate(self):
        """Test error rate analysis"""
        # Test high error rate
        context = {'error_rate': 0.02}
        analysis = self.engine._analyze_error_rate(context)
        self.assertEqual(analysis.criterion, DecisionCriterion.ERROR_RATE)
        self.assertGreater(analysis.score, 0.7)  # High error should switch
        
        # Test moderate error rate
        context = {'error_rate': 0.005}
        analysis = self.engine._analyze_error_rate(context)
        self.assertAlmostEqual(analysis.score, 0.6, delta=0.1)
        
        # Test low error rate
        context = {'error_rate': 0.0001}
        analysis = self.engine._analyze_error_rate(context)
        self.assertLess(analysis.score, 0.4)  # Low error should not switch
        
        # Test missing error rate
        context = {}
        analysis = self.engine._analyze_error_rate(context)
        self.assertEqual(analysis.score, 0.3)  # Default to low error
    
    def test_analyze_computational_load(self):
        """Test computational load analysis (placeholder)"""
        analysis = self.engine._analyze_computational_load()
        
        self.assertEqual(analysis.criterion, DecisionCriterion.COMPUTATIONAL_LOAD)
        self.assertEqual(analysis.score, 0.5)  # Placeholder returns neutral
        self.assertEqual(analysis.confidence, 0.3)
        self.assertTrue(analysis.metadata.get('placeholder', False))
    
    @patch('torch.cuda.is_available')
    def test_analyze_gpu_utilization_no_gpu(self, mock_cuda):
        """Test GPU utilization analysis without GPU"""
        mock_cuda.return_value = False
        
        analysis = self.engine._analyze_gpu_utilization()
        
        self.assertEqual(analysis.criterion, DecisionCriterion.GPU_UTILIZATION)
        self.assertEqual(analysis.score, 0.2)  # No GPU favors pure
        self.assertEqual(analysis.confidence, 0.8)
        self.assertEqual(analysis.metadata['gpu_available'], False)
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.current_device')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.get_device_properties')
    def test_analyze_gpu_utilization_with_gpu(self, mock_properties, mock_allocated,
                                              mock_current, mock_count, mock_cuda):
        """Test GPU utilization analysis with GPU"""
        mock_cuda.return_value = True
        mock_count.return_value = 1
        mock_current.return_value = 0
        mock_properties.return_value = MagicMock(total_memory=1024*1024*1024)  # 1GB
        mock_allocated.return_value = 200*1024*1024  # 200MB (low utilization)
        
        analysis = self.engine._analyze_gpu_utilization()
        
        self.assertEqual(analysis.criterion, DecisionCriterion.GPU_UTILIZATION)
        self.assertGreater(analysis.score, 0.7)  # Low utilization allows hybrid
        self.assertEqual(analysis.confidence, 0.6)
        self.assertIn('utilization_proxy', analysis.metadata)


class TestWeightedScoring(unittest.TestCase):
    """Test weighted scoring and recommendation logic"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = SwitchingDecisionEngine()
        self.engine.initialize_decision_engine(
            DirectionStateManager(),
            PerformanceOptimizer(),
            None
        )
    
    def test_calculate_weighted_score(self):
        """Test weighted score calculation"""
        analyses = [
            DecisionAnalysis(
                criterion=DecisionCriterion.GRADIENT_STABILITY,
                score=0.8,
                confidence=0.9,
                reasoning="Test"
            ),
            DecisionAnalysis(
                criterion=DecisionCriterion.DATA_SIZE,
                score=0.6,
                confidence=0.8,
                reasoning="Test"
            ),
            DecisionAnalysis(
                criterion=DecisionCriterion.MEMORY_USAGE,
                score=0.4,
                confidence=0.7,
                reasoning="Test"
            )
        ]
        
        weighted_score = self.engine._calculate_weighted_score(analyses)
        
        self.assertGreaterEqual(weighted_score, 0.0)
        self.assertLessEqual(weighted_score, 1.0)
    
    def test_calculate_overall_confidence(self):
        """Test overall confidence calculation"""
        analyses = [
            DecisionAnalysis(
                criterion=DecisionCriterion.GRADIENT_STABILITY,
                score=0.8,
                confidence=0.9,
                reasoning="Test"
            ),
            DecisionAnalysis(
                criterion=DecisionCriterion.DATA_SIZE,
                score=0.6,
                confidence=0.7,
                reasoning="Test"
            ),
            DecisionAnalysis(
                criterion=DecisionCriterion.MEMORY_USAGE,
                score=0.4,
                confidence=0.5,
                reasoning="Test"
            )
        ]
        
        overall_confidence = self.engine._calculate_overall_confidence(analyses)
        
        # Should be average of confidences
        expected = (0.9 + 0.7 + 0.5) / 3
        self.assertAlmostEqual(overall_confidence, expected, places=5)
        
        # Test with empty list
        self.assertEqual(self.engine._calculate_overall_confidence([]), 0.0)
    
    def test_make_switching_recommendation(self):
        """Test switching recommendation logic"""
        # Test insufficient confidence
        recommendation = self.engine._make_switching_recommendation(0.8, 0.5)
        self.assertEqual(recommendation, "insufficient_confidence")
        
        # Test switch to hybrid (high score)
        recommendation = self.engine._make_switching_recommendation(0.8, 0.8)
        self.assertEqual(recommendation, "switch_to_hybrid")
        
        # Test switch to pure (low score)
        recommendation = self.engine._make_switching_recommendation(0.2, 0.8)
        self.assertEqual(recommendation, "switch_to_pure")
        
        # Test maintain current (neutral score)
        recommendation = self.engine._make_switching_recommendation(0.5, 0.8)
        self.assertEqual(recommendation, "maintain_current")


class TestHelperMethods(unittest.TestCase):
    """Test helper methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = SwitchingDecisionEngine()
    
    def test_calculate_data_complexity(self):
        """Test data complexity calculation"""
        # Test with uniform data (low complexity)
        uniform_data = torch.ones(100)
        complexity = self.engine._calculate_data_complexity(uniform_data)
        self.assertEqual(complexity, 0.0)  # Constant data has zero complexity
        
        # Test with random data (higher complexity)
        random_data = torch.randn(100)
        complexity = self.engine._calculate_data_complexity(random_data)
        self.assertGreaterEqual(complexity, 0.0)
        self.assertLessEqual(complexity, 1.0)
        
        # Test with structured data
        structured_data = torch.arange(100).float()
        complexity = self.engine._calculate_data_complexity(structured_data)
        self.assertGreaterEqual(complexity, 0.0)
        self.assertLessEqual(complexity, 1.0)
    
    def test_get_historical_performance_adjustment(self):
        """Test historical performance adjustment calculation"""
        # Test with no history
        adjustment = self.engine._get_historical_performance_adjustment('hybrid')
        self.assertEqual(adjustment, 1.0)  # No adjustment
        
        # Add history for hybrid mode
        for _ in range(5):
            self.engine.performance_history.append({
                'timestamp': datetime.now(timezone.utc),
                'data': {'mode': 'hybrid', 'performance_score': 0.8}
            })
        
        adjustment = self.engine._get_historical_performance_adjustment('hybrid')
        # Better performance should give lower adjustment (faster)
        self.assertAlmostEqual(adjustment, 1.2, delta=0.1)
        
        # Test with no matching mode in history
        adjustment = self.engine._get_historical_performance_adjustment('pure_padic')
        self.assertEqual(adjustment, 1.0)  # No adjustment
    
    def test_find_optimal_threshold(self):
        """Test optimal threshold finding"""
        # Create some decision history
        decisions = []
        for i in range(10):
            decisions.append({
                'timestamp': datetime.now(timezone.utc),
                'data_shape': (100, 100),
                'weighted_score': 0.6 + i * 0.01,
                'overall_confidence': 0.7,
                'recommendation': 'switch_to_hybrid' if i % 2 == 0 else 'maintain_current',
                'analyses': [],
                'processing_time_ms': 10.0
            })
        
        # Find optimal threshold for hybrid
        optimal = self.engine._find_optimal_threshold('hybrid', decisions)
        
        # Should be close to current threshold
        self.assertGreaterEqual(optimal, self.engine.hybrid_threshold - 0.2)
        self.assertLessEqual(optimal, self.engine.hybrid_threshold + 0.2)
        
        # Find optimal threshold for pure
        optimal = self.engine._find_optimal_threshold('pure', decisions)
        
        # Should be close to current threshold
        self.assertGreaterEqual(optimal, self.engine.pure_threshold - 0.2)
        self.assertLessEqual(optimal, self.engine.pure_threshold + 0.2)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = SwitchingDecisionEngine()
    
    def test_zero_gradients(self):
        """Test handling of zero gradients"""
        self.engine.initialize_decision_engine(
            DirectionStateManager(),
            PerformanceOptimizer(),
            None
        )
        
        zero_gradients = torch.zeros(100)
        confidence = self.engine.evaluate_gradient_direction(zero_gradients)
        
        # Should return neutral confidence for zero gradients
        self.assertEqual(confidence, 0.5)
    
    def test_nan_in_data(self):
        """Test handling of NaN values in data"""
        self.engine.initialize_decision_engine(
            DirectionStateManager(),
            PerformanceOptimizer(),
            None
        )
        
        # Create data with NaN
        data = torch.randn(100)
        data[50] = float('nan')
        
        # Should handle gracefully
        try:
            result = self.engine.analyze_switching_criteria(data)
            # If it doesn't raise an error, check result is valid
            self.assertIsNotNone(result)
        except RuntimeError:
            # If it does raise, that's also acceptable error handling
            pass
    
    def test_inf_in_data(self):
        """Test handling of Inf values in data"""
        self.engine.initialize_decision_engine(
            DirectionStateManager(),
            PerformanceOptimizer(),
            None
        )
        
        # Create data with Inf
        data = torch.randn(100)
        data[25] = float('inf')
        data[75] = float('-inf')
        
        # Should handle gracefully
        try:
            result = self.engine.analyze_switching_criteria(data)
            self.assertIsNotNone(result)
        except RuntimeError:
            # If it does raise, that's also acceptable error handling
            pass
    
    def test_very_large_data(self):
        """Test with very large data tensors"""
        self.engine.initialize_decision_engine(
            DirectionStateManager(),
            PerformanceOptimizer(),
            None
        )
        
        # Create large tensor
        large_data = torch.randn(10000)
        
        # Should handle without issues
        result = self.engine.analyze_switching_criteria(large_data)
        self.assertIsNotNone(result)
        
        # Check that large data favors hybrid
        self.assertIn('criteria_analyses', result)
        data_size_analyses = [
            a for a in result['criteria_analyses']
            if a['criterion'] == DecisionCriterion.DATA_SIZE.value
        ]
        if data_size_analyses:
            self.assertGreater(data_size_analyses[0]['score'], 0.8)
    
    def test_concurrent_analysis(self):
        """Test concurrent analysis calls"""
        self.engine.initialize_decision_engine(
            DirectionStateManager(),
            PerformanceOptimizer(),
            None
        )
        
        # Multiple rapid calls should not interfere
        data1 = torch.randn(100)
        data2 = torch.randn(200)
        data3 = torch.randn(300)
        
        result1 = self.engine.analyze_switching_criteria(data1)
        result2 = self.engine.analyze_switching_criteria(data2)
        result3 = self.engine.analyze_switching_criteria(data3)
        
        # All should complete successfully
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        self.assertIsNotNone(result3)
        
        # Decision history should have all three
        self.assertGreaterEqual(len(self.engine.decision_history), 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)