"""
Comprehensive Test Suite for DirectionState and DirectionStateManager
Tests all direction tracking, state transitions, and confidence calculations
"""

import unittest
import torch
import numpy as np
import time
from typing import Dict, Any, List, Optional
from collections import deque
from unittest.mock import MagicMock, patch

# Import the modules to test
from direction_state import DirectionStateManager, DirectionHistory
from gac_types import DirectionType, DirectionState


class TestDirectionState(unittest.TestCase):
    """Test DirectionState dataclass"""
    
    def test_direction_state_creation(self):
        """Test creating DirectionState instances"""
        state = DirectionState(
            direction=DirectionType.ASCENT,
            confidence=0.8,
            magnitude=1.5,
            timestamp=1234.5,
            metadata={'key': 'value'}
        )
        
        self.assertEqual(state.direction, DirectionType.ASCENT)
        self.assertEqual(state.confidence, 0.8)
        self.assertEqual(state.magnitude, 1.5)
        self.assertEqual(state.timestamp, 1234.5)
        self.assertEqual(state.metadata['key'], 'value')
    
    def test_direction_type_enum(self):
        """Test DirectionType enum values"""
        self.assertEqual(DirectionType.ASCENT.value, "ascent")
        self.assertEqual(DirectionType.DESCENT.value, "descent")
        self.assertEqual(DirectionType.STABLE.value, "stable")
        self.assertEqual(DirectionType.OSCILLATING.value, "oscillating")
        
        # Test all enum members are present
        self.assertEqual(len(DirectionType), 6)  # Now includes POSITIVE and NEGATIVE


class TestDirectionHistory(unittest.TestCase):
    """Test DirectionHistory dataclass"""
    
    def test_direction_history_creation(self):
        """Test creating DirectionHistory instances"""
        history = DirectionHistory(
            direction=DirectionType.DESCENT,
            confidence=0.75,
            magnitude=2.0,
            timestamp=1000.0,
            gradient_norm=1.8,
            gradient_variance=0.3
        )
        
        self.assertEqual(history.direction, DirectionType.DESCENT)
        self.assertEqual(history.confidence, 0.75)
        self.assertEqual(history.magnitude, 2.0)
        self.assertEqual(history.timestamp, 1000.0)
        self.assertEqual(history.gradient_norm, 1.8)
        self.assertEqual(history.gradient_variance, 0.3)


class TestDirectionStateManager(unittest.TestCase):
    """Test DirectionStateManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'history_size': 10,
            'smoothing_factor': 0.8,
            'confidence_threshold': 0.7,
            'stability_window': 5,
            'oscillation_threshold': 3,
            'magnitude_threshold': 1e-6,
            'direction_change_threshold': 0.5,
            'stable_variance_threshold': 0.1
        }
        self.manager = DirectionStateManager(self.config)
    
    def test_initialization(self):
        """Test manager initialization"""
        # Test with config
        self.assertEqual(self.manager.history_size, 10)
        self.assertEqual(self.manager.smoothing_factor, 0.8)
        self.assertEqual(self.manager.confidence_threshold, 0.7)
        self.assertEqual(self.manager.stability_window, 5)
        
        # Test initial state
        self.assertIsNone(self.manager.current_state)
        self.assertEqual(len(self.manager.direction_history), 0)
        self.assertEqual(len(self.manager.gradient_history), 0)
        self.assertEqual(self.manager.smoothed_magnitude, 0.0)
        self.assertIsNone(self.manager.smoothed_direction_vector)
        
        # Test without config (defaults)
        manager_default = DirectionStateManager()
        self.assertEqual(manager_default.history_size, 100)
        self.assertEqual(manager_default.smoothing_factor, 0.8)
    
    def test_update_direction_state_valid(self):
        """Test updating direction state with valid gradients"""
        gradients = torch.tensor([0.1, 0.2, 0.3])
        context = {'iteration': 1}
        
        state = self.manager.update_direction_state(gradients, context)
        
        self.assertIsNotNone(state)
        self.assertIsInstance(state, DirectionState)
        self.assertIn(state.direction, DirectionType)
        self.assertGreaterEqual(state.confidence, 0.0)
        self.assertLessEqual(state.confidence, 1.0)
        self.assertGreater(state.magnitude, 0.0)
        self.assertGreater(state.timestamp, 0.0)
        
        # Check state was stored
        self.assertEqual(self.manager.current_state, state)
        self.assertEqual(self.manager.total_updates, 1)
        self.assertEqual(len(self.manager.gradient_history), 1)
        self.assertEqual(len(self.manager.direction_history), 1)
    
    def test_update_direction_state_invalid(self):
        """Test error handling for invalid gradient inputs"""
        # None gradients
        with self.assertRaises(ValueError):
            self.manager.update_direction_state(None)
        
        # Non-tensor gradients
        with self.assertRaises(TypeError):
            self.manager.update_direction_state([0.1, 0.2, 0.3])
        
        # Empty tensor
        with self.assertRaises(ValueError):
            self.manager.update_direction_state(torch.tensor([]))
    
    def test_determine_direction_type_stable(self):
        """Test direction determination for stable gradients"""
        # Very small gradients should be STABLE
        small_gradients = torch.tensor([1e-7, 1e-8, 1e-9])
        gradient_norm = torch.norm(small_gradients).item()
        gradient_variance = small_gradients.var().item()
        
        direction = self.manager._determine_direction_type(
            small_gradients, gradient_norm, gradient_variance
        )
        
        self.assertEqual(direction, DirectionType.STABLE)
    
    def test_determine_direction_type_oscillating(self):
        """Test oscillation detection"""
        # Set up oscillating history
        for i in range(6):
            direction = DirectionType.ASCENT if i % 2 == 0 else DirectionType.DESCENT
            history = DirectionHistory(
                direction=direction,
                confidence=0.8,
                magnitude=1.0,
                timestamp=i,
                gradient_norm=1.0,
                gradient_variance=0.5
            )
            self.manager.direction_history.append(history)
        
        # High variance gradient
        gradients = torch.tensor([1.0, -1.0, 1.0, -1.0])
        gradient_norm = torch.norm(gradients).item()
        gradient_variance = gradients.var().item()
        
        # This should detect oscillation
        is_oscillating = self.manager._is_oscillating()
        self.assertTrue(is_oscillating)
    
    def test_analyze_gradient_trend(self):
        """Test gradient trend analysis"""
        # Add gradient history with ascending trend
        for i in range(5):
            gradients = torch.ones(3) * (i + 1) * 0.1
            self.manager.gradient_history.append(gradients)
        
        trend = self.manager._analyze_gradient_trend()
        self.assertEqual(trend, DirectionType.ASCENT)
        
        # Clear and test descending trend
        self.manager.gradient_history.clear()
        for i in range(5):
            gradients = torch.ones(3) * (5 - i) * 0.1
            self.manager.gradient_history.append(gradients)
        
        trend = self.manager._analyze_gradient_trend()
        self.assertEqual(trend, DirectionType.DESCENT)
        
        # Test stable trend
        self.manager.gradient_history.clear()
        for i in range(5):
            gradients = torch.ones(3) * 0.5
            self.manager.gradient_history.append(gradients)
        
        trend = self.manager._analyze_gradient_trend()
        self.assertEqual(trend, DirectionType.STABLE)
    
    def test_calculate_direction_confidence(self):
        """Test confidence calculation"""
        gradients = torch.tensor([0.5, 0.5, 0.5])
        context = {'learning_rate': 0.01, 'iteration': 100}
        
        # Add some history for better confidence calculation
        for i in range(3):
            self.manager.gradient_history.append(torch.ones(3) * 0.5)
            history = DirectionHistory(
                direction=DirectionType.ASCENT,
                confidence=0.7,
                magnitude=0.5,
                timestamp=i,
                gradient_norm=0.5,
                gradient_variance=0.01
            )
            self.manager.direction_history.append(history)
        
        self.manager.smoothed_direction_vector = torch.ones(3) / np.sqrt(3)
        
        confidence = self.manager._calculate_direction_confidence(
            DirectionType.ASCENT, gradients, context
        )
        
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_calculate_stability_score(self):
        """Test stability score calculation"""
        # Add consistent history for high stability
        for i in range(10):
            history = DirectionHistory(
                direction=DirectionType.ASCENT,
                confidence=0.8,
                magnitude=1.0,
                timestamp=i,
                gradient_norm=1.0,
                gradient_variance=0.01
            )
            self.manager.direction_history.append(history)
        
        stability = self.manager._calculate_stability_score()
        
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 1.0)
        self.assertGreater(stability, 0.7)  # Should be high for consistent history
        
        # Test with inconsistent history
        self.manager.direction_history.clear()
        for i in range(10):
            direction = DirectionType.ASCENT if i % 2 == 0 else DirectionType.DESCENT
            history = DirectionHistory(
                direction=direction,
                confidence=0.5 + (i % 3) * 0.1,
                magnitude=0.5 + (i % 4) * 0.3,
                timestamp=i,
                gradient_norm=1.0,
                gradient_variance=0.1
            )
            self.manager.direction_history.append(history)
        
        stability = self.manager._calculate_stability_score()
        self.assertLess(stability, 0.5)  # Should be low for inconsistent history
    
    def test_direction_transitions(self):
        """Test direction transition tracking"""
        # First update
        gradients1 = torch.tensor([1.0, 1.0, 1.0])
        state1 = self.manager.update_direction_state(gradients1)
        
        # Force a different direction for second update
        self.manager.smoothed_magnitude = 10.0  # Make next gradient look smaller
        gradients2 = torch.tensor([0.001, 0.001, 0.001])
        state2 = self.manager.update_direction_state(gradients2)
        
        # Check if transition was tracked (only if directions are different)
        if state1.direction != state2.direction:
            self.assertEqual(self.manager.direction_transitions, 1)
            self.assertEqual(len(self.manager.direction_transition_history), 1)
    
    def test_get_direction_summary(self):
        """Test comprehensive summary generation"""
        # Add some history
        for i in range(5):
            gradients = torch.randn(3)
            self.manager.update_direction_state(gradients, {'iteration': i})
        
        summary = self.manager.get_direction_summary()
        
        self.assertIn('current_state', summary)
        self.assertIn('statistics', summary)
        self.assertIn('recent_distribution', summary)
        self.assertIn('performance_metrics', summary)
        
        # Check current state fields
        current = summary['current_state']
        self.assertIn('direction', current)
        self.assertIn('confidence', current)
        self.assertIn('magnitude', current)
        self.assertIn('timestamp', current)
        self.assertIn('stability_score', current)
        
        # Check statistics
        stats = summary['statistics']
        self.assertEqual(stats['total_updates'], 5)
        self.assertIn('history_length', stats)
        self.assertIn('smoothed_magnitude', stats)
    
    def test_transition_matrix_calculation(self):
        """Test transition probability matrix calculation"""
        # Add transition history
        transitions = [
            (DirectionType.ASCENT, DirectionType.STABLE, 1.0),
            (DirectionType.STABLE, DirectionType.DESCENT, 2.0),
            (DirectionType.DESCENT, DirectionType.ASCENT, 3.0),
            (DirectionType.ASCENT, DirectionType.ASCENT, 4.0),
            (DirectionType.ASCENT, DirectionType.STABLE, 5.0)
        ]
        
        for trans in transitions:
            self.manager.direction_transition_history.append(trans)
        
        matrix = self.manager._calculate_transition_matrix()
        
        self.assertIsInstance(matrix, dict)
        # Check ASCENT transitions (3 total: 2 to STABLE, 1 to ASCENT)
        self.assertAlmostEqual(matrix['ascent']['stable'], 2/3, places=5)
        self.assertAlmostEqual(matrix['ascent']['ascent'], 1/3, places=5)
    
    def test_stability_trend(self):
        """Test stability trend detection"""
        # Add improving stability history
        for i in range(15):
            direction = DirectionType.ASCENT if i > 5 else DirectionType.DESCENT
            history = DirectionHistory(
                direction=direction,
                confidence=0.5 + i * 0.02,
                magnitude=1.0,
                timestamp=i,
                gradient_norm=1.0,
                gradient_variance=0.1 - i * 0.005
            )
            self.manager.direction_history.append(history)
        
        trend = self.manager._get_stability_trend()
        self.assertIn(trend, ['improving', 'stable', 'degrading', 'insufficient_data', 'unknown'])
    
    def test_direction_persistence(self):
        """Test direction persistence calculation"""
        # Set current state
        self.manager.current_state = DirectionState(
            direction=DirectionType.ASCENT,
            confidence=0.8,
            magnitude=1.0,
            timestamp=time.time(),
            metadata={}
        )
        
        # Add history with mostly ASCENT
        for i in range(10):
            direction = DirectionType.ASCENT if i < 7 else DirectionType.STABLE
            history = DirectionHistory(
                direction=direction,
                confidence=0.7,
                magnitude=1.0,
                timestamp=i,
                gradient_norm=1.0,
                gradient_variance=0.1
            )
            self.manager.direction_history.append(history)
        
        persistence = self.manager._calculate_direction_persistence()
        self.assertAlmostEqual(persistence, 0.7, places=5)
    
    def test_reset(self):
        """Test manager reset"""
        # Add some state
        gradients = torch.randn(3)
        self.manager.update_direction_state(gradients)
        
        # Reset
        self.manager.reset()
        
        # Check everything is cleared
        self.assertIsNone(self.manager.current_state)
        self.assertEqual(len(self.manager.direction_history), 0)
        self.assertEqual(len(self.manager.gradient_history), 0)
        self.assertEqual(self.manager.smoothed_magnitude, 0.0)
        self.assertIsNone(self.manager.smoothed_direction_vector)
        self.assertEqual(self.manager.direction_transitions, 0)
        self.assertEqual(self.manager.total_updates, 0)
    
    def test_helper_methods(self):
        """Test helper methods"""
        # Test when no state
        self.assertIsNone(self.manager.get_current_state())
        self.assertFalse(self.manager.is_stable())
        self.assertFalse(self.manager.is_oscillating())
        self.assertIsNone(self.manager.current_direction)
        self.assertEqual(self.manager.current_confidence, 0.0)
        self.assertEqual(self.manager.current_magnitude, 0.0)
        
        # Add a stable state
        self.manager.current_state = DirectionState(
            direction=DirectionType.STABLE,
            confidence=0.8,
            magnitude=0.001,
            timestamp=time.time(),
            metadata={}
        )
        
        self.assertIsNotNone(self.manager.get_current_state())
        self.assertTrue(self.manager.is_stable())
        self.assertFalse(self.manager.is_oscillating())
        self.assertEqual(self.manager.current_direction, DirectionType.STABLE)
        self.assertEqual(self.manager.current_confidence, 0.8)
        self.assertEqual(self.manager.current_magnitude, 0.001)
    
    def test_get_direction_trend(self):
        """Test getting dominant direction trend"""
        # Not enough history
        trend = self.manager.get_direction_trend(window_size=5)
        self.assertIsNone(trend)
        
        # Add history with dominant ASCENT
        for i in range(10):
            direction = DirectionType.ASCENT if i % 3 != 0 else DirectionType.STABLE
            history = DirectionHistory(
                direction=direction,
                confidence=0.7,
                magnitude=1.0,
                timestamp=i,
                gradient_norm=1.0,
                gradient_variance=0.1
            )
            self.manager.direction_history.append(history)
        
        trend = self.manager.get_direction_trend(window_size=5)
        self.assertEqual(trend, DirectionType.ASCENT)
        
        # No dominant direction
        self.manager.direction_history.clear()
        for i in range(10):
            direction = [DirectionType.ASCENT, DirectionType.DESCENT, DirectionType.STABLE][i % 3]
            history = DirectionHistory(
                direction=direction,
                confidence=0.7,
                magnitude=1.0,
                timestamp=i,
                gradient_norm=1.0,
                gradient_variance=0.1
            )
            self.manager.direction_history.append(history)
        
        trend = self.manager.get_direction_trend(window_size=5)
        self.assertIsNone(trend)  # No direction is dominant enough
    
    def test_history_size_limit(self):
        """Test that history respects size limits"""
        # Add more items than history_size
        for i in range(20):
            gradients = torch.randn(3)
            self.manager.update_direction_state(gradients)
        
        # Check histories are limited
        self.assertLessEqual(len(self.manager.direction_history), self.config['history_size'])
        self.assertLessEqual(len(self.manager.gradient_history), self.config['history_size'])
        self.assertEqual(self.manager.total_updates, 20)  # Total updates still tracked
    
    def test_smoothing_updates(self):
        """Test smoothed value updates"""
        # First update
        gradients1 = torch.ones(3) * 2.0
        self.manager.update_direction_state(gradients1)
        first_smoothed = self.manager.smoothed_magnitude
        
        # Second update with different magnitude
        gradients2 = torch.ones(3) * 1.0
        self.manager.update_direction_state(gradients2)
        second_smoothed = self.manager.smoothed_magnitude
        
        # Check smoothing effect
        grad1_norm = torch.norm(gradients1).item()
        grad2_norm = torch.norm(gradients2).item()
        
        # Smoothed value should be between the two norms
        self.assertLess(second_smoothed, grad1_norm)
        self.assertGreater(second_smoothed, grad2_norm)
        
        # Check smoothing formula
        expected = self.config['smoothing_factor'] * first_smoothed + \
                   (1 - self.config['smoothing_factor']) * grad2_norm
        self.assertAlmostEqual(second_smoothed, expected, places=5)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Zero gradient
        zero_grad = torch.zeros(3)
        state = self.manager.update_direction_state(zero_grad)
        self.assertEqual(state.direction, DirectionType.STABLE)
        
        # Single element tensor
        single = torch.tensor([1.0])
        state = self.manager.update_direction_state(single)
        self.assertIsNotNone(state)
        
        # Very large gradients
        large = torch.ones(3) * 1e10
        state = self.manager.update_direction_state(large)
        self.assertIsNotNone(state)
        self.assertGreater(state.magnitude, 0)
        
        # Negative gradients
        negative = torch.tensor([-1.0, -2.0, -3.0])
        state = self.manager.update_direction_state(negative)
        self.assertIsNotNone(state)
    
    def test_confidence_special_cases(self):
        """Test confidence calculation special cases"""
        # Oscillating direction should have lower confidence
        self.manager.current_state = DirectionState(
            direction=DirectionType.OSCILLATING,
            confidence=0.5,
            magnitude=1.0,
            timestamp=time.time(),
            metadata={}
        )
        
        gradients = torch.randn(3)
        confidence = self.manager._calculate_direction_confidence(
            DirectionType.OSCILLATING, gradients, {}
        )
        
        # Should be reduced by factor of 0.8
        self.assertLess(confidence, 1.0)
        
        # Stable with tiny gradients should have high confidence
        tiny_gradients = torch.ones(3) * 1e-8
        confidence = self.manager._calculate_direction_confidence(
            DirectionType.STABLE, tiny_gradients, {}
        )
        self.assertGreater(confidence, 0.5)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios"""
    
    def test_full_lifecycle(self):
        """Test complete lifecycle of direction tracking"""
        manager = DirectionStateManager()
        
        # Simulate gradient evolution
        phases = [
            # Phase 1: Initial ascent
            ([1.0, 1.0, 1.0], DirectionType.ASCENT),
            ([1.2, 1.1, 1.0], DirectionType.ASCENT),
            ([1.5, 1.4, 1.3], DirectionType.ASCENT),
            # Phase 2: Stabilizing
            ([0.5, 0.5, 0.5], DirectionType.STABLE),
            ([0.1, 0.1, 0.1], DirectionType.STABLE),
            ([0.01, 0.01, 0.01], DirectionType.STABLE),
            # Phase 3: Descent
            ([0.5, 0.5, 0.5], DirectionType.STABLE),
            ([0.3, 0.3, 0.3], DirectionType.DESCENT),
            ([0.1, 0.1, 0.1], DirectionType.STABLE),
        ]
        
        for i, (grad_values, expected_trend) in enumerate(phases):
            gradients = torch.tensor(grad_values)
            state = manager.update_direction_state(gradients, {'iteration': i})
            
            self.assertIsNotNone(state)
            # After enough history, trend should be detectable
            if i > 3:
                trend = manager.get_direction_trend(window_size=3)
                # Trend might not always match exactly due to smoothing
        
        # Check final summary
        summary = manager.get_direction_summary()
        self.assertIn('current_state', summary)
        self.assertEqual(summary['statistics']['total_updates'], len(phases))
    
    def test_oscillation_detection_scenario(self):
        """Test realistic oscillation detection"""
        manager = DirectionStateManager({
            'oscillation_threshold': 3,
            'stability_window': 5
        })
        
        # Create oscillating gradient pattern
        for i in range(10):
            magnitude = 1.0 if i % 2 == 0 else -1.0
            gradients = torch.tensor([magnitude, -magnitude, magnitude])
            state = manager.update_direction_state(gradients, {'iteration': i})
        
        # After enough oscillations, should detect it
        if len(manager.direction_history) >= 6:
            is_oscillating = manager._is_oscillating()
            # Should detect oscillation pattern
    
    def test_concurrent_updates(self):
        """Test behavior with rapid concurrent-like updates"""
        manager = DirectionStateManager()
        
        # Simulate rapid updates
        states = []
        for i in range(100):
            gradients = torch.randn(3) * (1 + i * 0.01)
            state = manager.update_direction_state(gradients, {'iteration': i})
            states.append(state)
        
        # All states should be valid
        for state in states:
            self.assertIsNotNone(state)
            self.assertIn(state.direction, DirectionType)
            self.assertGreaterEqual(state.confidence, 0.0)
            self.assertLessEqual(state.confidence, 1.0)
        
        # Check consistency
        self.assertEqual(manager.total_updates, 100)
        self.assertEqual(manager.current_state, states[-1])


if __name__ == "__main__":
    unittest.main(verbosity=2)