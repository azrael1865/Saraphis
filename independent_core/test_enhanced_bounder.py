"""
Comprehensive Unit Tests for EnhancedBounder
Tests the enhanced gradient bounding with direction awareness and adaptive clipping
"""

import pytest
import torch
import numpy as np
import time
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, Optional
import logging

# Import the modules we're testing
from gac_system.enhanced_bounder import (
    EnhancedGradientBounder,
    EnhancedBoundingResult,
    EnhancedGradientBoundingError
)
from gac_system.gac_types import DirectionType, DirectionState
from gac_system.direction_state import DirectionStateManager, DirectionHistory
from gac_system.direction_validator import DirectionValidator, ValidationResult
from gac_system.basic_bounder import BasicGradientBounder, BoundingResult


class TestEnhancedBoundingResult:
    """Test the EnhancedBoundingResult dataclass"""
    
    def test_enhanced_result_creation(self):
        """Test creating an enhanced bounding result"""
        gradients = torch.randn(10, 10)
        direction_state = DirectionState(
            direction=DirectionType.ASCENT,
            confidence=0.9,
            magnitude=1.5,
            timestamp=time.time(),
            metadata={}
        )
        
        result = EnhancedBoundingResult(
            bounded_gradients=gradients,
            applied_factor=0.8,
            bounding_type="enhanced",
            direction_state=direction_state,
            direction_confidence=0.9,
            adaptive_factors={'test': 1.0},
            direction_based_adjustments={'adjusted': True},
            metadata={}
        )
        
        assert result.bounded_gradients is gradients
        assert result.applied_factor == 0.8
        assert result.direction_state == direction_state
        assert result.direction_confidence == 0.9
        assert 'test' in result.adaptive_factors
    
    def test_enhanced_result_inherits_from_basic(self):
        """Test that EnhancedBoundingResult inherits from BoundingResult"""
        result = EnhancedBoundingResult(
            bounded_gradients=torch.zeros(5, 5),
            applied_factor=1.0,
            bounding_type="test",
            direction_state=None,
            direction_confidence=0.5,
            adaptive_factors={},
            direction_based_adjustments={},
            metadata={}
        )
        
        # Should have basic result fields
        assert hasattr(result, 'bounded_gradients')
        assert hasattr(result, 'applied_factor')
        assert hasattr(result, 'bounding_type')
        assert hasattr(result, 'metadata')


class TestEnhancedGradientBounderInit:
    """Test EnhancedGradientBounder initialization"""
    
    def test_default_initialization(self):
        """Test initialization with default config"""
        bounder = EnhancedGradientBounder()
        
        assert bounder.direction_sensitivity == 0.8
        assert bounder.confidence_threshold == 0.7
        assert bounder.adaptive_scaling_strength == 1.0
        assert bounder.enable_momentum_adjustment == True
        assert bounder.enable_predictive_bounding == True
        assert bounder.enable_confidence_weighting == True
        assert bounder.enable_transition_smoothing == True
        
        # Check direction bounds are initialized
        assert DirectionType.ASCENT in bounder.direction_bounds
        assert DirectionType.DESCENT in bounder.direction_bounds
        assert DirectionType.STABLE in bounder.direction_bounds
        assert DirectionType.OSCILLATING in bounder.direction_bounds
    
    def test_custom_configuration(self):
        """Test initialization with custom config"""
        config = {
            'direction_sensitivity': 0.5,
            'confidence_threshold': 0.6,
            'adaptive_scaling_strength': 0.8,
            'ascent_max_norm': 3.0,
            'descent_max_norm': 2.5,
            'enable_momentum_adjustment': False,
            'enable_predictive_bounding': False
        }
        
        bounder = EnhancedGradientBounder(config)
        
        assert bounder.direction_sensitivity == 0.5
        assert bounder.confidence_threshold == 0.6
        assert bounder.adaptive_scaling_strength == 0.8
        assert bounder.direction_bounds[DirectionType.ASCENT]['max_norm'] == 3.0
        assert bounder.direction_bounds[DirectionType.DESCENT]['max_norm'] == 2.5
        assert bounder.enable_momentum_adjustment == False
        assert bounder.enable_predictive_bounding == False
    
    def test_direction_bounds_structure(self):
        """Test that direction bounds have correct structure"""
        bounder = EnhancedGradientBounder()
        
        for direction_type in DirectionType:
            bounds = bounder.direction_bounds[direction_type]
            assert 'max_norm' in bounds
            assert 'clip_value' in bounds
            assert 'scaling_factor' in bounds
            assert isinstance(bounds['max_norm'], (int, float))
            assert isinstance(bounds['clip_value'], (int, float))
            assert isinstance(bounds['scaling_factor'], (int, float))


class TestEnhancedGradientBounderComponents:
    """Test component management"""
    
    def test_set_direction_components(self):
        """Test setting direction components"""
        bounder = EnhancedGradientBounder()
        state_manager = Mock(spec=DirectionStateManager)
        validator = Mock(spec=DirectionValidator)
        
        bounder.set_direction_components(state_manager, validator)
        
        assert bounder.direction_state_manager == state_manager
        assert bounder.direction_validator == validator
        validator.set_direction_state_manager.assert_called_once_with(state_manager)
    
    def test_set_direction_components_none_error(self):
        """Test error when setting None components"""
        bounder = EnhancedGradientBounder()
        
        with pytest.raises(ValueError, match="Direction components cannot be None"):
            bounder.set_direction_components(None, Mock())
        
        with pytest.raises(ValueError, match="Direction components cannot be None"):
            bounder.set_direction_components(Mock(), None)


class TestEnhancedGradientBounderBounding:
    """Test gradient bounding functionality"""
    
    def test_bound_gradients_basic(self):
        """Test basic gradient bounding without direction components"""
        bounder = EnhancedGradientBounder()
        gradients = torch.randn(10, 10)
        
        result = bounder.bound_gradients(gradients)
        
        assert isinstance(result, EnhancedBoundingResult)
        assert result.bounded_gradients.shape == gradients.shape
        assert result.bounding_type == "enhanced_direction_aware"
        assert result.direction_state is None  # No direction manager set
        assert result.direction_confidence == 1.0  # Default confidence
    
    def test_bound_gradients_with_direction_components(self):
        """Test gradient bounding with direction components"""
        bounder = EnhancedGradientBounder()
        
        # Mock direction components
        state_manager = Mock(spec=DirectionStateManager)
        direction_state = DirectionState(
            direction=DirectionType.ASCENT,
            confidence=0.85,
            magnitude=1.2,
            timestamp=time.time(),
            metadata={}
        )
        state_manager.update_direction_state.return_value = direction_state
        state_manager.direction_history = []
        
        validator = Mock(spec=DirectionValidator)
        validation_result = Mock(spec=ValidationResult)
        validation_result.confidence = 0.85
        validation_result.metadata = {}
        validator.validate_direction_state.return_value = validation_result
        validator.set_direction_state_manager = Mock()
        
        bounder.set_direction_components(state_manager, validator)
        
        gradients = torch.randn(10, 10)
        result = bounder.bound_gradients(gradients)
        
        assert isinstance(result, EnhancedBoundingResult)
        assert result.direction_state == direction_state
        assert result.direction_confidence == 0.85
        state_manager.update_direction_state.assert_called_once()
        validator.validate_direction_state.assert_called_once_with(direction_state, {})
    
    def test_bound_gradients_none_error(self):
        """Test error when gradients are None"""
        bounder = EnhancedGradientBounder()
        
        with pytest.raises(EnhancedGradientBoundingError, match="Gradients cannot be None"):
            bounder.bound_gradients(None)
    
    def test_bound_gradients_wrong_type_error(self):
        """Test error when gradients are wrong type"""
        bounder = EnhancedGradientBounder()
        
        with pytest.raises(TypeError, match="Gradients must be torch.Tensor"):
            bounder.bound_gradients(np.array([1, 2, 3]))
    
    def test_bound_gradients_empty_error(self):
        """Test error when gradients are empty"""
        bounder = EnhancedGradientBounder()
        
        with pytest.raises(EnhancedGradientBoundingError, match="Gradients tensor cannot be empty"):
            bounder.bound_gradients(torch.tensor([]))
    
    def test_bound_gradients_with_context(self):
        """Test gradient bounding with context"""
        bounder = EnhancedGradientBounder()
        gradients = torch.randn(5, 5)
        context = {'iteration': 100, 'loss': 0.5}
        
        result = bounder.bound_gradients(gradients, context)
        
        assert isinstance(result, EnhancedBoundingResult)
        assert 'context' in result.metadata
        assert result.metadata['context'] == context


class TestDirectionSpecificBounding:
    """Test direction-specific bounding functionality"""
    
    def test_apply_direction_specific_bounding_ascent(self):
        """Test direction-specific bounding for ascent"""
        bounder = EnhancedGradientBounder()
        gradients = torch.randn(10, 10) * 5  # Large gradients
        
        direction_state = DirectionState(
            direction=DirectionType.ASCENT,
            confidence=0.9,
            magnitude=2.0,
            timestamp=time.time(),
            metadata={}
        )
        
        result = bounder._apply_direction_specific_bounding(gradients, direction_state, 0.9)
        
        assert 'gradients' in result
        assert 'factors' in result
        assert 'direction' in result
        assert result['direction'] == 'ascent'
        assert torch.all(torch.isfinite(result['gradients']))
    
    def test_apply_direction_specific_bounding_stable(self):
        """Test direction-specific bounding for stable state"""
        bounder = EnhancedGradientBounder()
        gradients = torch.randn(10, 10)
        
        direction_state = DirectionState(
            direction=DirectionType.STABLE,
            confidence=0.95,
            magnitude=0.5,
            timestamp=time.time(),
            metadata={}
        )
        
        result = bounder._apply_direction_specific_bounding(gradients, direction_state, 0.95)
        
        # Stable direction should have more conservative bounding
        assert result['factors']['direction_scaling'] == 0.8  # Default stable scaling
        assert result['direction'] == 'stable'
    
    def test_apply_direction_specific_bounding_low_confidence(self):
        """Test bounding with low confidence"""
        bounder = EnhancedGradientBounder()
        gradients = torch.randn(10, 10)
        
        direction_state = DirectionState(
            direction=DirectionType.OSCILLATING,
            confidence=0.3,  # Low confidence
            magnitude=1.0,
            timestamp=time.time(),
            metadata={}
        )
        
        result = bounder._apply_direction_specific_bounding(gradients, direction_state, 0.3)
        
        # Low confidence should result in more conservative bounding
        assert result['factors']['confidence_multiplier'] == 0.5  # Minimum multiplier
        assert result['direction'] == 'oscillating'


class TestConfidenceWeightedBounding:
    """Test confidence-weighted bounding"""
    
    def test_apply_confidence_weighted_bounding_high_confidence(self):
        """Test confidence weighting with high confidence"""
        bounder = EnhancedGradientBounder()
        gradients = torch.randn(10, 10)
        
        direction_state = DirectionState(
            direction=DirectionType.ASCENT,
            confidence=0.9,
            magnitude=1.0,
            timestamp=time.time(),
            metadata={}
        )
        
        result = bounder._apply_confidence_weighted_bounding(gradients, direction_state, 0.9)
        
        assert 'gradients' in result
        assert 'factors' in result
        assert result['confidence'] == 0.9
        assert result['applied'] == False  # High confidence, no adjustment needed
    
    def test_apply_confidence_weighted_bounding_low_confidence(self):
        """Test confidence weighting with low confidence"""
        bounder = EnhancedGradientBounder()
        gradients = torch.randn(10, 10)
        original_norm = torch.norm(gradients)
        
        direction_state = DirectionState(
            direction=DirectionType.ASCENT,
            confidence=0.3,  # Low confidence
            magnitude=1.0,
            timestamp=time.time(),
            metadata={}
        )
        
        result = bounder._apply_confidence_weighted_bounding(gradients, direction_state, 0.3)
        
        assert result['applied'] == True  # Low confidence, adjustment applied
        assert result['confidence'] == 0.3
        
        # Gradients should be scaled down
        new_norm = torch.norm(result['gradients'])
        assert new_norm < original_norm


class TestTransitionSmoothing:
    """Test transition smoothing functionality"""
    
    def test_apply_transition_smoothing_no_history(self):
        """Test transition smoothing without sufficient history"""
        bounder = EnhancedGradientBounder()
        gradients = torch.randn(10, 10)
        
        direction_state = DirectionState(
            direction=DirectionType.ASCENT,
            confidence=0.8,
            magnitude=1.0,
            timestamp=time.time(),
            metadata={}
        )
        
        result = bounder._apply_transition_smoothing(gradients, direction_state)
        
        assert result['applied'] == False
        assert result['reason'] == 'insufficient_history'
        assert torch.equal(result['gradients'], gradients)
    
    def test_apply_transition_smoothing_with_transition(self):
        """Test transition smoothing with direction transition"""
        bounder = EnhancedGradientBounder()
        
        # Create mock state manager with history
        state_manager = Mock(spec=DirectionStateManager)
        history = [
            DirectionHistory(DirectionType.ASCENT, 0.9, 1.0, time.time(), 1.0, 0.1),
            DirectionHistory(DirectionType.DESCENT, 0.8, 0.9, time.time(), 0.9, 0.1),
            DirectionHistory(DirectionType.ASCENT, 0.85, 0.95, time.time(), 0.95, 0.1)
        ]
        state_manager.direction_history = history
        
        bounder.direction_state_manager = state_manager
        
        gradients = torch.randn(10, 10)
        original_norm = torch.norm(gradients)
        
        direction_state = DirectionState(
            direction=DirectionType.ASCENT,
            confidence=0.8,
            magnitude=1.0,
            timestamp=time.time(),
            metadata={}
        )
        
        result = bounder._apply_transition_smoothing(gradients, direction_state)
        
        assert result['applied'] == True
        assert result['transition_detected'] == True
        
        # Gradients should be smoothed (reduced)
        new_norm = torch.norm(result['gradients'])
        assert new_norm < original_norm


class TestMomentumAdjustment:
    """Test momentum-based adjustments"""
    
    def test_apply_momentum_adjustment_first_call(self):
        """Test momentum adjustment on first call"""
        bounder = EnhancedGradientBounder()
        gradients = torch.randn(10, 10)
        
        result = bounder._apply_momentum_adjustment(gradients, {})
        
        assert result['applied'] == False  # No momentum yet
        assert result['momentum_available'] == True
        assert torch.equal(result['gradients'], gradients)  # No adjustment on first call
        assert bounder.momentum_gradients is not None
    
    def test_apply_momentum_adjustment_aligned(self):
        """Test momentum adjustment with aligned gradients"""
        bounder = EnhancedGradientBounder()
        
        # Set up initial momentum
        initial_gradients = torch.ones(10, 10)
        bounder.momentum_gradients = initial_gradients.clone()
        
        # Aligned gradients (similar direction)
        aligned_gradients = torch.ones(10, 10) * 1.1
        
        result = bounder._apply_momentum_adjustment(aligned_gradients, {})
        
        # Should have high cosine similarity
        assert result['factors']['cosine_similarity'] > 0.9
        assert result['factors']['momentum_factor'] > 1.0  # Should boost gradients
        assert result['applied'] == True
    
    def test_apply_momentum_adjustment_opposed(self):
        """Test momentum adjustment with opposed gradients"""
        bounder = EnhancedGradientBounder()
        
        # Set up initial momentum
        initial_gradients = torch.ones(10, 10)
        bounder.momentum_gradients = initial_gradients.clone()
        
        # Opposed gradients (opposite direction)
        opposed_gradients = -torch.ones(10, 10)
        
        result = bounder._apply_momentum_adjustment(opposed_gradients, {})
        
        # Should have negative cosine similarity
        assert result['factors']['cosine_similarity'] < -0.9
        assert result['factors']['momentum_factor'] < 1.0  # Should reduce gradients
        assert result['applied'] == True
    
    def test_apply_momentum_adjustment_zero_norm_handling(self):
        """Test momentum adjustment with zero norm gradients"""
        bounder = EnhancedGradientBounder()
        
        # Set up momentum
        bounder.momentum_gradients = torch.ones(10, 10)
        
        # Zero gradients
        zero_gradients = torch.zeros(10, 10)
        
        result = bounder._apply_momentum_adjustment(zero_gradients, {})
        
        # Should handle gracefully
        assert result['factors']['momentum_factor'] == 1.0  # No adjustment
        assert result['applied'] == False


class TestPredictiveBounding:
    """Test predictive bounding functionality"""
    
    def test_apply_predictive_bounding_insufficient_history(self):
        """Test predictive bounding without sufficient history"""
        bounder = EnhancedGradientBounder()
        gradients = torch.randn(10, 10)
        
        direction_state = DirectionState(
            direction=DirectionType.ASCENT,
            confidence=0.8,
            magnitude=1.0,
            timestamp=time.time(),
            metadata={}
        )
        
        result = bounder._apply_predictive_bounding(gradients, direction_state, {})
        
        assert result['applied'] == False
        assert result['reason'] == 'insufficient_history'
        assert torch.equal(result['gradients'], gradients)
    
    def test_apply_predictive_bounding_increasing_magnitude(self):
        """Test predictive bounding with increasing magnitude trend"""
        bounder = EnhancedGradientBounder()
        
        # Create mock state manager with increasing magnitude history
        state_manager = Mock(spec=DirectionStateManager)
        history = [
            DirectionHistory(DirectionType.ASCENT, 0.9, 0.5, time.time(), 1.0, 0.1),
            DirectionHistory(DirectionType.ASCENT, 0.9, 1.0, time.time(), 1.0, 0.1),
            DirectionHistory(DirectionType.ASCENT, 0.9, 1.5, time.time(), 1.0, 0.1),
            DirectionHistory(DirectionType.ASCENT, 0.9, 2.0, time.time(), 1.0, 0.1),
            DirectionHistory(DirectionType.ASCENT, 0.9, 2.5, time.time(), 1.0, 0.1)
        ]
        state_manager.direction_history = history
        
        bounder.direction_state_manager = state_manager
        
        gradients = torch.randn(10, 10)
        original_norm = torch.norm(gradients)
        
        direction_state = DirectionState(
            direction=DirectionType.ASCENT,
            confidence=0.9,
            magnitude=3.0,
            timestamp=time.time(),
            metadata={}
        )
        
        result = bounder._apply_predictive_bounding(gradients, direction_state, {})
        
        assert result['applied'] == True
        assert result['factors']['magnitude_trend'] > 0  # Positive trend
        assert result['factors']['predictive_factor'] < 1.0  # Should scale down
        
        # Gradients should be reduced
        new_norm = torch.norm(result['gradients'])
        assert new_norm < original_norm
    
    def test_apply_predictive_bounding_oscillating(self):
        """Test predictive bounding for oscillating direction"""
        bounder = EnhancedGradientBounder()
        
        # Create mock state manager with history
        state_manager = Mock(spec=DirectionStateManager)
        history = [
            DirectionHistory(DirectionType.OSCILLATING, 0.7, 1.0, time.time(), 1.0, 0.1)
            for _ in range(5)
        ]
        state_manager.direction_history = history
        
        bounder.direction_state_manager = state_manager
        
        gradients = torch.randn(10, 10)
        
        direction_state = DirectionState(
            direction=DirectionType.OSCILLATING,
            confidence=0.7,
            magnitude=1.0,
            timestamp=time.time(),
            metadata={}
        )
        
        result = bounder._apply_predictive_bounding(gradients, direction_state, {})
        
        assert result['applied'] == True
        assert result['factors']['predictive_factor'] == 0.85  # Oscillation damping
        assert result['predictions']['direction'] == 'oscillating'


class TestStatisticsAndMonitoring:
    """Test statistics and monitoring functionality"""
    
    def test_get_enhanced_statistics(self):
        """Test getting enhanced statistics"""
        bounder = EnhancedGradientBounder()
        
        # Perform some operations
        gradients = torch.randn(10, 10)
        for _ in range(5):
            bounder.bound_gradients(gradients)
        
        stats = bounder.get_enhanced_statistics()
        
        assert 'basic_statistics' in stats
        assert 'enhanced_operations' in stats
        assert 'direction_specific_bounds' in stats
        assert 'enhancement_performance' in stats
        assert 'feature_status' in stats
        assert 'component_status' in stats
        
        assert stats['enhanced_operations']['total_enhanced_bounds'] >= 0
        assert stats['feature_status']['momentum_adjustment'] == True
        assert stats['component_status']['direction_state_manager'] == False  # Not set
    
    def test_reset_statistics(self):
        """Test resetting statistics"""
        bounder = EnhancedGradientBounder()
        
        # Perform operations
        gradients = torch.randn(10, 10)
        bounder.bound_gradients(gradients)
        bounder.confidence_adjustments = 5
        bounder.predictive_bounds = 3
        
        bounder.reset_statistics()
        
        assert bounder.confidence_adjustments == 0
        assert bounder.predictive_bounds == 0
        assert bounder.transition_smoothings == 0
        assert all(count == 0 for count in bounder.direction_specific_bounds.values())


class TestDirectionBoundsManagement:
    """Test direction bounds management"""
    
    def test_update_direction_bounds(self):
        """Test updating direction-specific bounds"""
        bounder = EnhancedGradientBounder()
        
        new_bounds = {
            'max_norm': 3.0,
            'clip_value': 20.0,
            'scaling_factor': 1.5
        }
        
        bounder.update_direction_bounds(DirectionType.ASCENT, new_bounds)
        
        assert bounder.direction_bounds[DirectionType.ASCENT]['max_norm'] == 3.0
        assert bounder.direction_bounds[DirectionType.ASCENT]['clip_value'] == 20.0
        assert bounder.direction_bounds[DirectionType.ASCENT]['scaling_factor'] == 1.5
    
    def test_get_direction_bounds(self):
        """Test getting direction-specific bounds"""
        bounder = EnhancedGradientBounder()
        
        bounds = bounder.get_direction_bounds(DirectionType.DESCENT)
        
        assert 'max_norm' in bounds
        assert 'clip_value' in bounds
        assert 'scaling_factor' in bounds
        assert bounds['max_norm'] == 1.5  # Default descent max_norm
    
    def test_get_direction_bounds_returns_copy(self):
        """Test that get_direction_bounds returns a copy"""
        bounder = EnhancedGradientBounder()
        
        bounds1 = bounder.get_direction_bounds(DirectionType.STABLE)
        bounds1['max_norm'] = 999
        
        bounds2 = bounder.get_direction_bounds(DirectionType.STABLE)
        
        assert bounds2['max_norm'] != 999  # Should not be modified


class TestHealthAndRecommendations:
    """Test health checking and recommendations"""
    
    def test_is_enhancement_healthy_no_components(self):
        """Test health check without components"""
        bounder = EnhancedGradientBounder()
        
        assert bounder.is_enhancement_healthy() == False
    
    def test_is_enhancement_healthy_with_components(self):
        """Test health check with components"""
        bounder = EnhancedGradientBounder()
        
        state_manager = Mock(spec=DirectionStateManager)
        validator = Mock(spec=DirectionValidator)
        validator.set_direction_state_manager = Mock()
        
        bounder.set_direction_components(state_manager, validator)
        
        assert bounder.is_enhancement_healthy() == True
    
    def test_is_enhancement_healthy_high_overhead(self):
        """Test health check with high overhead"""
        bounder = EnhancedGradientBounder()
        
        state_manager = Mock(spec=DirectionStateManager)
        validator = Mock(spec=DirectionValidator)
        validator.set_direction_state_manager = Mock()
        
        bounder.set_direction_components(state_manager, validator)
        
        # Simulate high overhead
        bounder.enhancement_overhead = 10.0
        bounder.direction_specific_bounds[DirectionType.ASCENT] = 50
        
        assert bounder.is_enhancement_healthy() == False  # Too much overhead
    
    def test_get_optimal_bounds_recommendation_no_manager(self):
        """Test recommendations without state manager"""
        bounder = EnhancedGradientBounder()
        
        recommendations = bounder.get_optimal_bounds_recommendation()
        
        assert recommendations['status'] == 'no_direction_manager'
    
    def test_get_optimal_bounds_recommendation_with_data(self):
        """Test recommendations with sufficient data"""
        bounder = EnhancedGradientBounder()
        
        state_manager = Mock(spec=DirectionStateManager)
        state_manager.get_direction_summary.return_value = {}
        bounder.direction_state_manager = state_manager
        
        # Simulate usage
        bounder.direction_specific_bounds[DirectionType.ASCENT] = 150
        bounder.direction_specific_bounds[DirectionType.DESCENT] = 50
        
        recommendations = bounder.get_optimal_bounds_recommendation()
        
        assert 'recommendations' in recommendations
        assert 'ascent' in recommendations['recommendations']
        assert 'descent' in recommendations['recommendations']
        
        # High usage should increase bounds
        ascent_rec = recommendations['recommendations']['ascent']
        assert ascent_rec['max_norm'] > bounder.direction_bounds[DirectionType.ASCENT]['max_norm']


class TestEdgeCasesAndIntegration:
    """Test edge cases and integration scenarios"""
    
    def test_bound_gradients_full_pipeline(self):
        """Test full bounding pipeline with all features enabled"""
        config = {
            'enable_momentum_adjustment': True,
            'enable_predictive_bounding': True,
            'enable_confidence_weighting': True,
            'enable_transition_smoothing': True,
            'confidence_threshold': 0.6
        }
        
        bounder = EnhancedGradientBounder(config)
        
        # Set up mocked components
        state_manager = Mock(spec=DirectionStateManager)
        direction_state = DirectionState(
            direction=DirectionType.ASCENT,
            confidence=0.5,  # Low confidence to trigger weighting
            magnitude=1.5,
            timestamp=time.time(),
            metadata={}
        )
        state_manager.update_direction_state.return_value = direction_state
        state_manager.direction_history = [
            DirectionHistory(DirectionType.ASCENT, 0.8, 1.0, time.time(), 1.0, 0.1)
            for _ in range(10)
        ]
        state_manager.get_direction_summary.return_value = {}
        
        validator = Mock(spec=DirectionValidator)
        validation_result = Mock(spec=ValidationResult)
        validation_result.confidence = 0.5
        validation_result.metadata = {}
        validator.validate_direction_state.return_value = validation_result
        validator.set_direction_state_manager = Mock()
        
        bounder.set_direction_components(state_manager, validator)
        
        # Initialize momentum
        initial_gradients = torch.randn(10, 10)
        bounder.bound_gradients(initial_gradients)
        
        # Test full pipeline
        gradients = torch.randn(10, 10) * 10  # Large gradients
        context = {'iteration': 100}
        
        result = bounder.bound_gradients(gradients, context)
        
        assert isinstance(result, EnhancedBoundingResult)
        assert result.direction_state == direction_state
        assert result.direction_confidence == 0.5
        assert 'direction_specific' in result.direction_based_adjustments
        assert 'confidence_weighting' in result.direction_based_adjustments
        assert 'momentum' in result.direction_based_adjustments
        assert bounder.confidence_adjustments > 0
    
    def test_bound_gradients_zero_gradients(self):
        """Test handling of zero gradients"""
        bounder = EnhancedGradientBounder()
        gradients = torch.zeros(10, 10)
        
        result = bounder.bound_gradients(gradients)
        
        assert torch.all(result.bounded_gradients == 0)
        assert result.applied_factor == 0.0 or torch.isnan(torch.tensor(result.applied_factor))
    
    def test_bound_gradients_inf_gradients(self):
        """Test handling of infinite gradients"""
        bounder = EnhancedGradientBounder()
        gradients = torch.full((10, 10), float('inf'))
        
        result = bounder.bound_gradients(gradients)
        
        # Should bound infinite values
        assert torch.all(torch.isfinite(result.bounded_gradients))
    
    def test_bound_gradients_nan_gradients(self):
        """Test handling of NaN gradients"""
        bounder = EnhancedGradientBounder()
        gradients = torch.full((10, 10), float('nan'))
        
        # This might raise an error or handle gracefully depending on implementation
        # For now, just test it doesn't crash
        try:
            result = bounder.bound_gradients(gradients)
            # If it succeeds, check result is valid
            assert result.bounded_gradients.shape == gradients.shape
        except (EnhancedGradientBoundingError, RuntimeError):
            # Expected for NaN inputs
            pass
    
    def test_bound_gradients_very_large_gradients(self):
        """Test handling of very large gradients"""
        bounder = EnhancedGradientBounder()
        gradients = torch.randn(10, 10) * 1e10
        
        result = bounder.bound_gradients(gradients)
        
        # Should be bounded
        assert torch.all(torch.abs(result.bounded_gradients) < 1e10)
        assert result.applied_factor < 1.0
    
    def test_thread_safety_considerations(self):
        """Test that multiple boundings don't interfere"""
        bounder = EnhancedGradientBounder()
        
        # Perform multiple boundings
        results = []
        for i in range(10):
            gradients = torch.randn(5, 5) * (i + 1)
            result = bounder.bound_gradients(gradients)
            results.append(result)
        
        # Each should be independent
        assert len(results) == 10
        assert all(isinstance(r, EnhancedBoundingResult) for r in results)
    
    def test_momentum_decay_over_time(self):
        """Test that momentum decays properly"""
        bounder = EnhancedGradientBounder({'momentum_decay_rate': 0.5})
        
        # Initialize momentum
        initial_gradients = torch.ones(10, 10) * 10
        bounder.bound_gradients(initial_gradients)
        initial_momentum_norm = torch.norm(bounder.momentum_gradients).item()
        
        # Apply several times with zero gradients
        for _ in range(5):
            bounder.bound_gradients(torch.zeros(10, 10))
        
        final_momentum_norm = torch.norm(bounder.momentum_gradients).item()
        
        # Momentum should decay
        assert final_momentum_norm < initial_momentum_norm


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])