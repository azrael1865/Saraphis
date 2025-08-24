"""
Comprehensive test suite for MLBasedProofEngine
Tests all aspects of ML-based proof generation, gradient analysis, and prediction validation
"""

import pytest
import numpy as np
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock
import threading

from proof_system.ml_based_engine import MLBasedProofEngine


class TestMLBasedProofEngineInitialization:
    """Test MLBasedProofEngine initialization"""
    
    def test_initialization_default(self):
        """Test engine initialization with default settings"""
        engine = MLBasedProofEngine()
        
        assert engine.logger is not None
        assert engine.prediction_history == []
        assert hasattr(engine, 'generate_ml_proof')
        assert hasattr(engine, 'analyze_gradients')
    
    def test_multiple_instances(self):
        """Test that multiple instances maintain separate state"""
        engine1 = MLBasedProofEngine()
        engine2 = MLBasedProofEngine()
        
        # Add prediction to engine1
        engine1.prediction_history.append({'prediction': 0.5, 'confidence': 0.8, 'timestamp': time.time()})
        
        # engine2 should not have this prediction
        assert len(engine1.prediction_history) == 1
        assert len(engine2.prediction_history) == 0


class TestGenerateMLProof:
    """Test generate_ml_proof method"""
    
    def test_generate_proof_basic(self):
        """Test basic proof generation"""
        engine = MLBasedProofEngine()
        
        transaction = {
            'features': [0.1, 0.2, 0.3],
            'model_prediction': 0.7,
            'model_confidence': 0.85
        }
        
        model_state = {}
        
        proof = engine.generate_ml_proof(transaction, model_state)
        
        assert 'model_prediction' in proof
        assert proof['model_prediction'] == 0.7
        assert 'model_confidence' in proof
        assert proof['model_confidence'] == 0.85
        assert 'confidence_score' in proof
        assert 'prediction_stability' in proof
        assert 'feature_importance' in proof
        assert 'gradient_analysis' in proof
        assert 'uncertainty_analysis' in proof
        assert 'generation_time_ms' in proof
        assert 'timestamp' in proof
    
    def test_generate_proof_with_numpy_features(self):
        """Test proof generation with numpy array features"""
        engine = MLBasedProofEngine()
        
        transaction = {
            'features': np.array([0.1, 0.2, 0.3]),
            'model_prediction': 0.6,
            'model_confidence': 0.75
        }
        
        model_state = {}
        
        proof = engine.generate_ml_proof(transaction, model_state)
        
        assert proof['model_prediction'] == 0.6
        assert proof['model_confidence'] == 0.75
        assert isinstance(proof['feature_importance'], dict)
    
    def test_generate_proof_with_gradients(self):
        """Test proof generation with gradient analysis"""
        engine = MLBasedProofEngine()
        
        transaction = {
            'features': [0.1, 0.2, 0.3],
            'model_prediction': 0.8,
            'model_confidence': 0.9
        }
        
        model_state = {
            'gradients': np.array([0.01, -0.02, 0.03, 0.04])
        }
        
        proof = engine.generate_ml_proof(transaction, model_state)
        
        assert 'gradient_analysis' in proof
        assert proof['gradient_analysis'] != {}
        assert 'gradient_norm' in proof['gradient_analysis']
        assert 'gradient_stability' in proof['gradient_analysis']
    
    def test_generate_proof_without_features(self):
        """Test proof generation without features"""
        engine = MLBasedProofEngine()
        
        transaction = {
            'model_prediction': 0.5,
            'model_confidence': 0.5
        }
        
        model_state = {}
        
        proof = engine.generate_ml_proof(transaction, model_state)
        
        assert proof['model_prediction'] == 0.5
        assert proof['feature_importance'] == {}
    
    def test_generate_proof_missing_prediction(self):
        """Test proof generation with missing prediction (defaults)"""
        engine = MLBasedProofEngine()
        
        transaction = {
            'features': [0.1, 0.2]
        }
        
        model_state = {}
        
        proof = engine.generate_ml_proof(transaction, model_state)
        
        # Should use default values
        assert proof['model_prediction'] == 0.5
        assert proof['model_confidence'] == 0.5
    
    def test_prediction_history_management(self):
        """Test that prediction history is properly managed"""
        engine = MLBasedProofEngine()
        
        # Generate multiple proofs
        for i in range(5):
            transaction = {
                'model_prediction': 0.5 + i * 0.1,
                'model_confidence': 0.8
            }
            engine.generate_ml_proof(transaction, {})
        
        assert len(engine.prediction_history) == 5
        
        # Check that history items have correct structure
        for item in engine.prediction_history:
            assert 'prediction' in item
            assert 'confidence' in item
            assert 'timestamp' in item
    
    def test_prediction_history_limit(self):
        """Test that prediction history is limited to 1000 entries"""
        engine = MLBasedProofEngine()
        
        # Add 1005 predictions
        for i in range(1005):
            engine.prediction_history.append({
                'prediction': 0.5,
                'confidence': 0.8,
                'timestamp': time.time()
            })
        
        transaction = {'model_prediction': 0.6, 'model_confidence': 0.9}
        engine.generate_ml_proof(transaction, {})
        
        # Should keep only last 1000
        assert len(engine.prediction_history) == 1000
        
        # Most recent should be the last one
        assert engine.prediction_history[-1]['prediction'] == 0.6


class TestAnalyzeGradients:
    """Test analyze_gradients method"""
    
    def test_analyze_gradients_normal(self):
        """Test gradient analysis with normal gradients"""
        engine = MLBasedProofEngine()
        
        gradients = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        
        analysis = engine.analyze_gradients(gradients)
        
        assert 'gradient_norm' in analysis
        assert 'gradient_mean' in analysis
        assert 'gradient_std' in analysis
        assert 'gradient_max' in analysis
        assert 'has_nan' in analysis
        assert 'has_inf' in analysis
        assert 'is_exploding' in analysis
        assert 'is_vanishing' in analysis
        assert 'gradient_stability' in analysis
        assert 'convergence_indicator' in analysis
        
        assert analysis['has_nan'] == False
        assert analysis['has_inf'] == False
        assert analysis['gradient_stability'] == 'stable'
    
    def test_analyze_gradients_exploding(self):
        """Test gradient analysis with exploding gradients"""
        engine = MLBasedProofEngine()
        
        gradients = np.array([100.0, 200.0, -150.0, 180.0])
        
        analysis = engine.analyze_gradients(gradients)
        
        assert analysis['is_exploding'] == True
        assert analysis['gradient_stability'] == 'unstable'
    
    def test_analyze_gradients_vanishing(self):
        """Test gradient analysis with vanishing gradients"""
        engine = MLBasedProofEngine()
        
        gradients = np.array([1e-8, -1e-9, 1e-8, -1e-8])
        
        analysis = engine.analyze_gradients(gradients)
        
        assert analysis['is_vanishing'] == True
        assert analysis['gradient_stability'] == 'unstable'
    
    def test_analyze_gradients_with_nan(self):
        """Test gradient analysis with NaN values"""
        engine = MLBasedProofEngine()
        
        gradients = np.array([0.01, np.nan, 0.03, -0.01])
        
        analysis = engine.analyze_gradients(gradients)
        
        assert analysis['has_nan'] == True
        assert analysis['gradient_stability'] == 'unstable'
    
    def test_analyze_gradients_with_inf(self):
        """Test gradient analysis with infinite values"""
        engine = MLBasedProofEngine()
        
        gradients = np.array([0.01, np.inf, 0.03, -0.01])
        
        analysis = engine.analyze_gradients(gradients)
        
        assert analysis['has_inf'] == True
        assert analysis['gradient_stability'] == 'unstable'
    
    def test_analyze_gradients_empty(self):
        """Test gradient analysis with empty gradients"""
        engine = MLBasedProofEngine()
        
        gradients = np.array([])
        
        analysis = engine.analyze_gradients(gradients)
        
        assert 'error' in analysis
        assert analysis['error'] == 'No gradients provided'
    
    def test_analyze_gradients_none(self):
        """Test gradient analysis with None"""
        engine = MLBasedProofEngine()
        
        analysis = engine.analyze_gradients(None)
        
        assert 'error' in analysis
        assert analysis['error'] == 'No gradients provided'
    
    def test_analyze_gradients_list_input(self):
        """Test gradient analysis with list input"""
        engine = MLBasedProofEngine()
        
        gradients = [0.01, -0.02, 0.03, -0.01]
        
        analysis = engine.analyze_gradients(gradients)
        
        assert 'gradient_norm' in analysis
        assert analysis['gradient_stability'] == 'stable'
    
    def test_gradient_convergence_assessment(self):
        """Test gradient convergence assessment"""
        engine = MLBasedProofEngine()
        
        # Test converged gradients
        converged = np.array([1e-7, -1e-7, 1e-8])
        analysis = engine.analyze_gradients(converged)
        assert analysis['convergence_indicator']['status'] == 'converged'
        
        # Test converging gradients
        converging = np.array([1e-4, -1e-4, 1e-4])
        analysis = engine.analyze_gradients(converging)
        assert analysis['convergence_indicator']['status'] == 'converging'
        
        # Test slow convergence
        slow = np.array([0.05, -0.03, 0.04])
        analysis = engine.analyze_gradients(slow)
        assert analysis['convergence_indicator']['status'] == 'slow_convergence'
        
        # Test not converging
        not_converging = np.array([1.0, -0.8, 0.9])
        analysis = engine.analyze_gradients(not_converging)
        assert analysis['convergence_indicator']['status'] == 'not_converging'


class TestPredictionStability:
    """Test prediction stability analysis"""
    
    def test_stability_insufficient_data(self):
        """Test stability analysis with insufficient data"""
        engine = MLBasedProofEngine()
        
        stability = engine._analyze_prediction_stability(0.6)
        
        assert stability['stability_score'] == 0.5
        assert stability['variance'] == 0.0
        assert stability['trend'] == 'insufficient_data'
    
    def test_stability_with_history(self):
        """Test stability analysis with prediction history"""
        engine = MLBasedProofEngine()
        
        # Add some history
        for i in range(5):
            engine.prediction_history.append({
                'prediction': 0.5 + i * 0.01,
                'confidence': 0.8,
                'timestamp': time.time()
            })
        
        stability = engine._analyze_prediction_stability(0.55)
        
        assert 'stability_score' in stability
        assert 'variance' in stability
        assert 'mean_prediction' in stability
        assert 'trend' in stability
        assert 'sample_size' in stability
        
        assert stability['sample_size'] == 6  # 5 history + 1 current
    
    def test_stability_high_variance(self):
        """Test stability with high variance predictions"""
        engine = MLBasedProofEngine()
        
        # Add high variance history
        predictions = [0.1, 0.9, 0.2, 0.8, 0.3]
        for pred in predictions:
            engine.prediction_history.append({
                'prediction': pred,
                'confidence': 0.8,
                'timestamp': time.time()
            })
        
        stability = engine._analyze_prediction_stability(0.7)
        
        # High variance should result in low stability score
        assert stability['stability_score'] < 0.5
        assert stability['variance'] > 0.1
    
    def test_stability_trend_detection(self):
        """Test trend detection in predictions"""
        engine = MLBasedProofEngine()
        
        # Add increasing trend
        for i in range(10):
            engine.prediction_history.append({
                'prediction': 0.3 + i * 0.05,
                'confidence': 0.8,
                'timestamp': time.time()
            })
        
        stability = engine._analyze_prediction_stability(0.85)
        
        assert stability['trend'] == 'increasing'
        
        # Test decreasing trend
        engine.prediction_history = []
        for i in range(10):
            engine.prediction_history.append({
                'prediction': 0.8 - i * 0.05,
                'confidence': 0.8,
                'timestamp': time.time()
            })
        
        stability = engine._analyze_prediction_stability(0.25)
        
        assert stability['trend'] == 'decreasing'
        
        # Test stable trend
        engine.prediction_history = []
        for i in range(10):
            engine.prediction_history.append({
                'prediction': 0.5 + np.random.normal(0, 0.001),
                'confidence': 0.8,
                'timestamp': time.time()
            })
        
        stability = engine._analyze_prediction_stability(0.5)
        
        assert stability['trend'] == 'stable'


class TestFeatureImportance:
    """Test feature importance calculation"""
    
    def test_feature_importance_basic(self):
        """Test basic feature importance calculation"""
        engine = MLBasedProofEngine()
        
        features = [0.1, 0.5, 0.2, 0.8]
        model_state = {}
        
        importance = engine._calculate_feature_importance(features, model_state)
        
        assert len(importance) == 4
        assert all(f'feature_{i}' in importance for i in range(4))
        
        # Should sum to 1 (normalized)
        total = sum(importance.values())
        assert abs(total - 1.0) < 1e-6
        
        # Feature 3 (0.8) should have highest importance
        assert importance['feature_3'] == max(importance.values())
    
    def test_feature_importance_empty(self):
        """Test feature importance with empty features"""
        engine = MLBasedProofEngine()
        
        features = []
        model_state = {}
        
        importance = engine._calculate_feature_importance(features, model_state)
        
        assert importance == {}
    
    def test_feature_importance_zero_features(self):
        """Test feature importance with all zero features"""
        engine = MLBasedProofEngine()
        
        features = [0.0, 0.0, 0.0]
        model_state = {}
        
        importance = engine._calculate_feature_importance(features, model_state)
        
        # All features should have equal importance when all are zero
        assert len(importance) == 3
        for value in importance.values():
            assert abs(value - 1/3) < 1e-6
    
    def test_feature_importance_negative_features(self):
        """Test feature importance with negative features"""
        engine = MLBasedProofEngine()
        
        features = [-0.5, 0.3, -0.8, 0.2]
        model_state = {}
        
        importance = engine._calculate_feature_importance(features, model_state)
        
        # Should use absolute values
        assert importance['feature_2'] == max(importance.values())  # |-0.8| = 0.8 is largest
        
        # Should still sum to 1
        total = sum(importance.values())
        assert abs(total - 1.0) < 1e-6


class TestUncertaintyQuantification:
    """Test uncertainty quantification"""
    
    def test_uncertainty_near_boundary(self):
        """Test uncertainty near decision boundary"""
        engine = MLBasedProofEngine()
        
        # Prediction near 0.5 boundary
        uncertainty = engine._quantify_uncertainty(0.51, 0.9)
        
        assert uncertainty['boundary_uncertainty'] > 0.9
        assert uncertainty['uncertainty_reason'] == 'near_decision_boundary'
    
    def test_uncertainty_low_confidence(self):
        """Test uncertainty with low model confidence"""
        engine = MLBasedProofEngine()
        
        uncertainty = engine._quantify_uncertainty(0.8, 0.4)
        
        assert uncertainty['confidence_uncertainty'] == 0.6
        assert uncertainty['uncertainty_reason'] == 'low_model_confidence'
    
    def test_uncertainty_high_overall(self):
        """Test high overall uncertainty"""
        engine = MLBasedProofEngine()
        
        # Both near boundary and low confidence
        uncertainty = engine._quantify_uncertainty(0.52, 0.3)
        
        assert uncertainty['combined_uncertainty'] > 0.7
        assert uncertainty['uncertainty_reason'] == 'high_overall_uncertainty'
    
    def test_uncertainty_acceptable(self):
        """Test acceptable uncertainty"""
        engine = MLBasedProofEngine()
        
        # Far from boundary and high confidence
        uncertainty = engine._quantify_uncertainty(0.9, 0.95)
        
        assert uncertainty['combined_uncertainty'] < 0.3
        assert uncertainty['uncertainty_reason'] == 'acceptable_uncertainty'
    
    def test_uncertainty_extreme_predictions(self):
        """Test uncertainty with extreme predictions"""
        engine = MLBasedProofEngine()
        
        # Very confident positive prediction
        uncertainty = engine._quantify_uncertainty(0.99, 0.99)
        
        assert uncertainty['boundary_uncertainty'] < 0.1
        assert uncertainty['confidence_uncertainty'] < 0.1
        assert uncertainty['prediction_certainty'] > 0.9
        
        # Very confident negative prediction
        uncertainty = engine._quantify_uncertainty(0.01, 0.99)
        
        assert uncertainty['boundary_uncertainty'] < 0.1
        assert uncertainty['confidence_uncertainty'] < 0.1
        assert uncertainty['prediction_certainty'] > 0.9


class TestMLConfidenceCalculation:
    """Test ML confidence score calculation"""
    
    def test_confidence_calculation_balanced(self):
        """Test confidence calculation with balanced inputs"""
        engine = MLBasedProofEngine()
        
        stability = {'stability_score': 0.8}
        uncertainty = {'prediction_certainty': 0.7}
        
        confidence = engine._calculate_ml_confidence(0.75, stability, uncertainty)
        
        # Should be weighted average
        expected = 0.75 * 0.4 + 0.8 * 0.3 + 0.7 * 0.3
        assert abs(confidence - expected) < 1e-6
    
    def test_confidence_calculation_clipping(self):
        """Test confidence calculation with clipping"""
        engine = MLBasedProofEngine()
        
        # Test upper bound
        stability = {'stability_score': 1.5}  # Invalid but should be clipped
        uncertainty = {'prediction_certainty': 1.2}
        
        confidence = engine._calculate_ml_confidence(1.1, stability, uncertainty)
        
        assert confidence <= 1.0
        
        # Test lower bound
        stability = {'stability_score': -0.5}
        uncertainty = {'prediction_certainty': -0.2}
        
        confidence = engine._calculate_ml_confidence(-0.3, stability, uncertainty)
        
        assert confidence >= 0.0
    
    def test_confidence_calculation_missing_values(self):
        """Test confidence calculation with missing values"""
        engine = MLBasedProofEngine()
        
        stability = {}  # Missing stability_score
        uncertainty = {}  # Missing prediction_certainty
        
        confidence = engine._calculate_ml_confidence(0.6, stability, uncertainty)
        
        # Should use defaults (0.5)
        expected = 0.6 * 0.4 + 0.5 * 0.3 + 0.5 * 0.3
        assert abs(confidence - expected) < 1e-6


class TestEnsemblePredictions:
    """Test ensemble prediction aggregation"""
    
    def test_aggregate_ensemble_basic(self):
        """Test basic ensemble aggregation"""
        engine = MLBasedProofEngine()
        
        predictions = {
            'model1': 0.7,
            'model2': 0.75,
            'model3': 0.72,
            'model4': 0.68
        }
        
        result = engine.aggregate_ensemble_predictions(predictions)
        
        assert 'ensemble_mean' in result
        assert 'ensemble_std' in result
        assert 'ensemble_median' in result
        assert 'agreement_score' in result
        assert 'outlier_models' in result
        assert 'model_count' in result
        
        assert result['model_count'] == 4
        assert abs(result['ensemble_mean'] - 0.7125) < 1e-6
    
    def test_aggregate_ensemble_with_outliers(self):
        """Test ensemble aggregation with outliers"""
        engine = MLBasedProofEngine()
        
        predictions = {
            'model1': 0.5,
            'model2': 0.52,
            'model3': 0.49,
            'model4': 0.9  # Outlier
        }
        
        result = engine.aggregate_ensemble_predictions(predictions)
        
        assert 'model4' in result['outlier_models']
        assert result['agreement_score'] < 0.5  # Low agreement due to outlier
    
    def test_aggregate_ensemble_empty(self):
        """Test ensemble aggregation with empty predictions"""
        engine = MLBasedProofEngine()
        
        predictions = {}
        
        result = engine.aggregate_ensemble_predictions(predictions)
        
        assert 'error' in result
        assert result['error'] == 'No predictions provided'
    
    def test_aggregate_ensemble_single_model(self):
        """Test ensemble aggregation with single model"""
        engine = MLBasedProofEngine()
        
        predictions = {'model1': 0.6}
        
        result = engine.aggregate_ensemble_predictions(predictions)
        
        assert result['ensemble_mean'] == 0.6
        assert result['ensemble_std'] == 0.0
        assert result['ensemble_median'] == 0.6
        assert result['agreement_score'] == 1.0  # Perfect agreement (only one model)
        assert len(result['outlier_models']) == 0
    
    def test_aggregate_ensemble_high_agreement(self):
        """Test ensemble with high agreement"""
        engine = MLBasedProofEngine()
        
        predictions = {
            'model1': 0.80,
            'model2': 0.81,
            'model3': 0.79,
            'model4': 0.80,
            'model5': 0.81
        }
        
        result = engine.aggregate_ensemble_predictions(predictions)
        
        assert result['agreement_score'] > 0.9
        assert len(result['outlier_models']) == 0


class TestTemporalConsistency:
    """Test temporal consistency checking"""
    
    def test_temporal_consistency_normal(self):
        """Test temporal consistency with normal prediction"""
        engine = MLBasedProofEngine()
        
        historical = [0.5, 0.52, 0.48, 0.51, 0.49, 0.50]
        current = 0.52
        
        result = engine.check_temporal_consistency(current, historical)
        
        assert result['consistency_score'] > 0.8
        assert result['anomaly_detected'] == False
        assert result['z_score'] < 2.0
    
    def test_temporal_consistency_anomaly(self):
        """Test temporal consistency with anomalous prediction"""
        engine = MLBasedProofEngine()
        
        historical = [0.5, 0.52, 0.48, 0.51, 0.49, 0.50]
        current = 0.9  # Anomaly
        
        result = engine.check_temporal_consistency(current, historical)
        
        assert result['consistency_score'] < 0.5
        assert result['anomaly_detected'] == True
        assert result['z_score'] > 2.0
    
    def test_temporal_consistency_no_history(self):
        """Test temporal consistency with no historical data"""
        engine = MLBasedProofEngine()
        
        historical = []
        current = 0.6
        
        result = engine.check_temporal_consistency(current, historical)
        
        assert result['consistency_score'] == 0.5
        assert result['anomaly_detected'] == False
        assert result['message'] == 'No historical data'
    
    def test_temporal_consistency_constant_history(self):
        """Test temporal consistency with constant historical values"""
        engine = MLBasedProofEngine()
        
        historical = [0.5, 0.5, 0.5, 0.5, 0.5]
        current = 0.5
        
        result = engine.check_temporal_consistency(current, historical)
        
        # Perfect consistency
        assert result['consistency_score'] == 1.0
        assert result['anomaly_detected'] == False
        assert result['z_score'] == 0.0
        assert result['historical_std'] == 0.0
    
    def test_temporal_consistency_edge_case(self):
        """Test temporal consistency at 2 std dev boundary"""
        engine = MLBasedProofEngine()
        
        historical = [0.5, 0.5, 0.5, 0.5, 0.5]
        # Add variation to get non-zero std
        historical = [0.48, 0.50, 0.52, 0.49, 0.51]
        
        mean = np.mean(historical)
        std = np.std(historical)
        current = mean + 2 * std  # Exactly at boundary
        
        result = engine.check_temporal_consistency(current, historical)
        
        assert abs(result['z_score'] - 2.0) < 0.1
        # Should not be anomaly if exactly at 2 std
        # (implementation uses > 2.0, not >= 2.0)


class TestGenerateProof:
    """Test the main generate_proof method"""
    
    def test_generate_proof_complete(self):
        """Test complete proof generation"""
        engine = MLBasedProofEngine()
        
        transaction = {
            'transaction_id': 'tx_123',
            'features': [0.1, 0.2, 0.3],
            'model_prediction': 0.75,
            'model_confidence': 0.88
        }
        
        model_state = {
            'iteration': 1000,
            'gradients': np.array([0.01, -0.02, 0.015])
        }
        
        proof = engine.generate_proof(transaction, model_state)
        
        assert proof['engine_type'] == 'ml_based'
        assert proof['transaction_id'] == 'tx_123'
        assert 'ml_analysis' in proof
        assert 'proof_metadata' in proof
        
        # Check ml_analysis contents
        ml_analysis = proof['ml_analysis']
        assert ml_analysis['model_prediction'] == 0.75
        assert ml_analysis['model_confidence'] == 0.88
        
        # Check metadata
        metadata = proof['proof_metadata']
        assert metadata['engine_version'] == '1.0.0'
        assert metadata['model_iteration'] == 1000
        assert 'generation_timestamp' in metadata
    
    def test_generate_proof_missing_transaction_id(self):
        """Test proof generation with missing transaction ID"""
        engine = MLBasedProofEngine()
        
        transaction = {
            'features': [0.1, 0.2],
            'model_prediction': 0.6
        }
        
        model_state = {}
        
        proof = engine.generate_proof(transaction, model_state)
        
        assert proof['transaction_id'] == 'unknown'
    
    def test_generate_proof_missing_iteration(self):
        """Test proof generation with missing model iteration"""
        engine = MLBasedProofEngine()
        
        transaction = {
            'transaction_id': 'tx_456',
            'model_prediction': 0.7
        }
        
        model_state = {}  # No iteration
        
        proof = engine.generate_proof(transaction, model_state)
        
        assert proof['proof_metadata']['model_iteration'] == 0


class TestThreadSafety:
    """Test thread safety of the engine"""
    
    def test_concurrent_proof_generation(self):
        """Test concurrent proof generation from multiple threads"""
        engine = MLBasedProofEngine()
        results = []
        errors = []
        
        def generate_proof_thread(thread_id):
            try:
                transaction = {
                    'transaction_id': f'tx_{thread_id}',
                    'features': np.random.rand(5).tolist(),
                    'model_prediction': np.random.rand(),
                    'model_confidence': np.random.rand()
                }
                
                model_state = {
                    'iteration': thread_id * 100,
                    'gradients': np.random.randn(5)
                }
                
                proof = engine.generate_proof(transaction, model_state)
                results.append(proof)
            except Exception as e:
                errors.append(e)
        
        # Create and start threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=generate_proof_thread, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 10
        
        # Each proof should have unique transaction_id
        transaction_ids = [p['transaction_id'] for p in results]
        assert len(set(transaction_ids)) == 10
    
    def test_concurrent_history_management(self):
        """Test concurrent updates to prediction history"""
        engine = MLBasedProofEngine()
        
        def add_predictions(thread_id):
            for i in range(100):
                transaction = {
                    'model_prediction': np.random.rand(),
                    'model_confidence': np.random.rand()
                }
                engine.generate_ml_proof(transaction, {})
        
        # Create and start threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=add_predictions, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Should have 500 predictions total
        assert len(engine.prediction_history) == 500


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_extreme_predictions(self):
        """Test handling of extreme prediction values"""
        engine = MLBasedProofEngine()
        
        # Test with prediction > 1
        transaction = {
            'model_prediction': 1.5,
            'model_confidence': 0.9
        }
        
        proof = engine.generate_ml_proof(transaction, {})
        assert proof['model_prediction'] == 1.5  # Should handle gracefully
        
        # Test with negative prediction
        transaction = {
            'model_prediction': -0.5,
            'model_confidence': 0.9
        }
        
        proof = engine.generate_ml_proof(transaction, {})
        assert proof['model_prediction'] == -0.5  # Should handle gracefully
    
    def test_extreme_confidence(self):
        """Test handling of extreme confidence values"""
        engine = MLBasedProofEngine()
        
        # Test with confidence > 1
        transaction = {
            'model_prediction': 0.5,
            'model_confidence': 1.5
        }
        
        proof = engine.generate_ml_proof(transaction, {})
        
        # Confidence score should be clipped
        assert 0.0 <= proof['confidence_score'] <= 1.0
    
    def test_very_large_features(self):
        """Test handling of very large feature arrays"""
        engine = MLBasedProofEngine()
        
        # Create large feature array
        features = np.random.randn(10000).tolist()
        
        transaction = {
            'features': features,
            'model_prediction': 0.6,
            'model_confidence': 0.8
        }
        
        proof = engine.generate_ml_proof(transaction, {})
        
        assert len(proof['feature_importance']) == 10000
        
        # Should still sum to 1
        total = sum(proof['feature_importance'].values())
        assert abs(total - 1.0) < 1e-5
    
    def test_very_large_gradients(self):
        """Test handling of very large gradient arrays"""
        engine = MLBasedProofEngine()
        
        # Create large gradient array
        gradients = np.random.randn(10000)
        
        analysis = engine.analyze_gradients(gradients)
        
        assert 'gradient_norm' in analysis
        assert 'convergence_indicator' in analysis
    
    def test_special_float_values(self):
        """Test handling of special float values"""
        engine = MLBasedProofEngine()
        
        # Test with infinity in features
        transaction = {
            'features': [1.0, np.inf, 2.0],
            'model_prediction': 0.5
        }
        
        proof = engine.generate_ml_proof(transaction, {})
        # Should handle infinity in feature importance calculation
        assert 'feature_importance' in proof
        
        # Test with NaN in gradients
        model_state = {
            'gradients': np.array([1.0, np.nan, 2.0])
        }
        
        transaction = {'model_prediction': 0.5}
        proof = engine.generate_ml_proof(transaction, model_state)
        
        assert proof['gradient_analysis']['has_nan'] == True


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_complete_ml_workflow(self):
        """Test complete ML proof generation workflow"""
        engine = MLBasedProofEngine()
        
        # Simulate multiple transactions over time
        for i in range(20):
            transaction = {
                'transaction_id': f'tx_{i}',
                'features': np.random.randn(10).tolist(),
                'model_prediction': 0.5 + np.random.randn() * 0.1,  # Around 0.5
                'model_confidence': 0.7 + np.random.rand() * 0.2  # 0.7-0.9
            }
            
            model_state = {
                'iteration': i * 100,
                'gradients': np.random.randn(10) * np.exp(-i/10)  # Decreasing gradients
            }
            
            proof = engine.generate_proof(transaction, model_state)
            
            # Verify proof structure
            assert proof['engine_type'] == 'ml_based'
            assert proof['transaction_id'] == f'tx_{i}'
            
            # Later proofs should show convergence
            if i > 10:
                gradient_analysis = proof['ml_analysis']['gradient_analysis']
                if 'convergence_indicator' in gradient_analysis:
                    # Gradients are decreasing, so should show convergence
                    assert gradient_analysis['convergence_indicator']['status'] in [
                        'converged', 'converging', 'slow_convergence'
                    ]
        
        # Check that history was properly maintained
        assert len(engine.prediction_history) == 20
        
        # Verify stability improves over time (as predictions stabilize)
        final_transaction = {
            'model_prediction': 0.5,
            'model_confidence': 0.85
        }
        
        final_proof = engine.generate_ml_proof(final_transaction, {})
        stability = final_proof['prediction_stability']
        
        # Should have sufficient data for trend analysis
        assert stability['trend'] in ['stable', 'increasing', 'decreasing']
        assert stability['sample_size'] >= 10
    
    def test_ensemble_workflow(self):
        """Test ensemble prediction workflow"""
        engine = MLBasedProofEngine()
        
        # Generate predictions from multiple models
        model_predictions = {
            'model_rf': 0.72,
            'model_xgb': 0.75,
            'model_nn': 0.71,
            'model_svm': 0.73,
            'model_lr': 0.30  # Outlier model
        }
        
        # Aggregate ensemble
        ensemble_result = engine.aggregate_ensemble_predictions(model_predictions)
        
        assert ensemble_result['model_count'] == 5
        assert 'model_lr' in ensemble_result['outlier_models']
        
        # Use ensemble result in proof generation
        transaction = {
            'transaction_id': 'tx_ensemble',
            'features': [0.1, 0.2, 0.3],
            'model_prediction': ensemble_result['ensemble_mean'],
            'model_confidence': ensemble_result['agreement_score']
        }
        
        proof = engine.generate_proof(transaction, {})
        
        assert proof['ml_analysis']['model_prediction'] == ensemble_result['ensemble_mean']
        assert proof['ml_analysis']['model_confidence'] == ensemble_result['agreement_score']
    
    def test_temporal_validation_workflow(self):
        """Test temporal validation workflow"""
        engine = MLBasedProofEngine()
        
        # Build historical predictions
        historical_predictions = []
        for i in range(50):
            pred = 0.6 + np.random.randn() * 0.05  # Normal range around 0.6
            
            transaction = {
                'model_prediction': pred,
                'model_confidence': 0.8
            }
            
            engine.generate_ml_proof(transaction, {})
            historical_predictions.append(pred)
        
        # Test normal prediction
        normal_pred = 0.62
        consistency_normal = engine.check_temporal_consistency(
            normal_pred, 
            historical_predictions[-20:]
        )
        
        assert consistency_normal['anomaly_detected'] == False
        assert consistency_normal['consistency_score'] > 0.7
        
        # Test anomalous prediction
        anomaly_pred = 0.95
        consistency_anomaly = engine.check_temporal_consistency(
            anomaly_pred,
            historical_predictions[-20:]
        )
        
        assert consistency_anomaly['anomaly_detected'] == True
        assert consistency_anomaly['consistency_score'] < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])