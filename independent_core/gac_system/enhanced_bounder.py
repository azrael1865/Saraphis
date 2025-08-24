"""
GAC Enhanced Direction-Aware Gradient Bounder
Enhanced gradient bounding with direction awareness and adaptive clipping
"""

import torch
import numpy as np
import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import logging

from .gac_types import DirectionType, DirectionState
from .direction_state import DirectionStateManager
from .direction_validator import DirectionValidator
from .basic_bounder import BasicGradientBounder, BoundingResult

logger = logging.getLogger(__name__)


@dataclass
class EnhancedBoundingResult(BoundingResult):
    """Enhanced result with direction-aware information"""
    direction_state: Optional[DirectionState]
    direction_confidence: float
    adaptive_factors: Dict[str, float]
    direction_based_adjustments: Dict[str, Any]


class EnhancedGradientBoundingError(Exception):
    """Raised when enhanced gradient bounding fails"""
    pass


class EnhancedGradientBounder:
    """Enhanced direction-aware gradient bounding with adaptive clipping strategies"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
            
        # Initialize basic bounder
        self.basic_bounder = BasicGradientBounder(config.get('basic_config', {}))
        
        # Direction-aware configuration
        self.direction_sensitivity = config.get('direction_sensitivity', 0.8)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.adaptive_scaling_strength = config.get('adaptive_scaling_strength', 1.0)
        
        # Direction-specific bounding parameters
        self.direction_bounds = {
            DirectionType.ASCENT: {
                'max_norm': config.get('ascent_max_norm', 2.0),
                'clip_value': config.get('ascent_clip_value', 15.0),
                'scaling_factor': config.get('ascent_scaling', 1.1)
            },
            DirectionType.DESCENT: {
                'max_norm': config.get('descent_max_norm', 1.5),
                'clip_value': config.get('descent_clip_value', 12.0),
                'scaling_factor': config.get('descent_scaling', 0.9)
            },
            DirectionType.STABLE: {
                'max_norm': config.get('stable_max_norm', 0.5),
                'clip_value': config.get('stable_clip_value', 5.0),
                'scaling_factor': config.get('stable_scaling', 0.8)
            },
            DirectionType.OSCILLATING: {
                'max_norm': config.get('oscillating_max_norm', 1.0),
                'clip_value': config.get('oscillating_clip_value', 8.0),
                'scaling_factor': config.get('oscillating_scaling', 0.7)
            },
            DirectionType.POSITIVE: {
                'max_norm': config.get('positive_max_norm', 1.8),
                'clip_value': config.get('positive_clip_value', 10.0),
                'scaling_factor': config.get('positive_scaling', 1.0)
            },
            DirectionType.NEGATIVE: {
                'max_norm': config.get('negative_max_norm', 1.8),
                'clip_value': config.get('negative_clip_value', 10.0),
                'scaling_factor': config.get('negative_scaling', 1.0)
            }
        }
        
        # Momentum and history tracking
        self.enable_momentum_adjustment = config.get('enable_momentum_adjustment', True)
        self.momentum_factor = config.get('momentum_factor', 0.9)
        self.history_window = config.get('history_window', 10)
        
        # Advanced features
        self.enable_predictive_bounding = config.get('enable_predictive_bounding', True)
        self.enable_confidence_weighting = config.get('enable_confidence_weighting', True)
        self.enable_transition_smoothing = config.get('enable_transition_smoothing', True)
        
        # Component references
        self.direction_state_manager: Optional[DirectionStateManager] = None
        self.direction_validator: Optional[DirectionValidator] = None
        
        # Enhanced statistics
        self.direction_specific_bounds = {dt: 0 for dt in DirectionType}
        self.confidence_adjustments = 0
        self.predictive_bounds = 0
        self.transition_smoothings = 0
        
        # Performance tracking
        self.enhancement_overhead = 0.0
        self.last_enhancement_time = 0.0
        
        # Momentum tracking
        self.momentum_gradients: Optional[torch.Tensor] = None
        self.momentum_decay_rate = config.get('momentum_decay_rate', 0.1)
        
    def set_direction_components(self, state_manager: DirectionStateManager, validator: DirectionValidator):
        """Set references to direction components"""
        if state_manager is None or validator is None:
            raise ValueError("Direction components cannot be None")
        
        self.direction_state_manager = state_manager
        self.direction_validator = validator
        validator.set_direction_state_manager(state_manager)
        
    def bound_gradients(self, gradients: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> EnhancedBoundingResult:
        """Apply enhanced direction-aware gradient bounding"""
        if gradients is None:
            raise EnhancedGradientBoundingError("Gradients cannot be None")
        if not isinstance(gradients, torch.Tensor):
            raise TypeError("Gradients must be torch.Tensor")
        if gradients.numel() == 0:
            raise EnhancedGradientBoundingError("Gradients tensor cannot be empty")
            
        if context is None:
            context = {}
            
        enhancement_start = time.time()
        
        # Step 1: Update direction state if manager available
        direction_state = None
        if self.direction_state_manager:
            direction_state = self.direction_state_manager.update_direction_state(gradients, context)
        
        # Step 2: Validate direction state if validator available
        direction_confidence = 1.0
        validation_result = None
        if self.direction_validator and direction_state:
            validation_result = self.direction_validator.validate_direction_state(direction_state, context)
            direction_confidence = validation_result.confidence
        
        # Step 3: Apply basic bounding first
        basic_result = self.basic_bounder.bound_gradients(gradients, context)
        bounded_gradients = basic_result.bounded_gradients
        
        # Step 4: Apply direction-aware enhancements
        enhancement_metadata = {}
        adaptive_factors = {}
        
        if direction_state:
            # Direction-specific parameter adjustment
            direction_adjustments = self._apply_direction_specific_bounding(
                bounded_gradients, direction_state, direction_confidence
            )
            bounded_gradients = direction_adjustments['gradients']
            enhancement_metadata['direction_specific'] = direction_adjustments
            adaptive_factors.update(direction_adjustments['factors'])
            
            # Confidence-weighted adjustments
            if self.enable_confidence_weighting and direction_confidence < self.confidence_threshold:
                confidence_adjustments = self._apply_confidence_weighted_bounding(
                    bounded_gradients, direction_state, direction_confidence
                )
                bounded_gradients = confidence_adjustments['gradients']
                enhancement_metadata['confidence_weighting'] = confidence_adjustments
                adaptive_factors.update(confidence_adjustments['factors'])
                self.confidence_adjustments += 1
            
            # Transition smoothing
            if self.enable_transition_smoothing:
                smoothing_result = self._apply_transition_smoothing(
                    bounded_gradients, direction_state
                )
                bounded_gradients = smoothing_result['gradients']
                enhancement_metadata['transition_smoothing'] = smoothing_result
                adaptive_factors.update(smoothing_result['factors'])
                if smoothing_result['applied']:
                    self.transition_smoothings += 1
        
        # Step 5: Momentum-based adjustments
        if self.enable_momentum_adjustment:
            momentum_result = self._apply_momentum_adjustment(bounded_gradients, context)
            bounded_gradients = momentum_result['gradients']
            enhancement_metadata['momentum'] = momentum_result
            adaptive_factors.update(momentum_result['factors'])
        
        # Step 6: Predictive bounding
        if self.enable_predictive_bounding and direction_state:
            predictive_result = self._apply_predictive_bounding(
                bounded_gradients, direction_state, context
            )
            bounded_gradients = predictive_result['gradients']
            enhancement_metadata['predictive'] = predictive_result
            adaptive_factors.update(predictive_result['factors'])
            if predictive_result['applied']:
                self.predictive_bounds += 1
        
        # Calculate final metrics
        original_norm = torch.norm(gradients, p=2).item()
        final_norm = torch.norm(bounded_gradients, p=2).item()
        total_applied_factor = final_norm / (original_norm + 1e-12)
        
        # Record timing
        enhancement_end = time.time()
        enhancement_time = enhancement_end - enhancement_start
        self.last_enhancement_time = enhancement_time
        self.enhancement_overhead += enhancement_time
        
        # Update direction-specific statistics
        if direction_state:
            self.direction_specific_bounds[direction_state.direction] += 1
        
        # Create enhanced result
        result = EnhancedBoundingResult(
            bounded_gradients=bounded_gradients,
            applied_factor=total_applied_factor,
            bounding_type="enhanced_direction_aware",
            direction_state=direction_state,
            direction_confidence=direction_confidence,
            adaptive_factors=adaptive_factors,
            direction_based_adjustments=enhancement_metadata,
            metadata={
                'original_norm': original_norm,
                'final_norm': final_norm,
                'basic_result': basic_result.metadata,
                'enhancement_metadata': enhancement_metadata,
                'adaptive_factors': adaptive_factors,
                'direction_confidence': direction_confidence,
                'validation_result': validation_result.metadata if validation_result else None,
                'enhancement_time': enhancement_time,
                'context': context
            }
        )
        
        return result
    
    def _apply_direction_specific_bounding(self, gradients: torch.Tensor, direction_state: DirectionState, confidence: float) -> Dict[str, Any]:
        """Apply direction-specific bounding parameters"""
        direction = direction_state.direction
        bounds = self.direction_bounds[direction]
        
        # Get direction-specific parameters
        max_norm = bounds['max_norm']
        clip_value = bounds['clip_value']
        scaling_factor = bounds['scaling_factor']
        
        # Adjust based on confidence
        confidence_multiplier = confidence if confidence > 0.5 else 0.5
        adjusted_max_norm = max_norm * confidence_multiplier
        adjusted_clip_value = clip_value * confidence_multiplier
        
        # Apply norm clipping with direction-specific bounds
        current_norm = torch.norm(gradients, p=2)
        norm_clipped = False
        if current_norm > adjusted_max_norm:
            scale_factor = adjusted_max_norm / current_norm
            gradients = gradients * scale_factor
            norm_clipped = True
        
        # Apply value clipping
        value_clipped = False
        original_max = torch.max(torch.abs(gradients)).item()
        if original_max > adjusted_clip_value:
            gradients = torch.clamp(gradients, -adjusted_clip_value, adjusted_clip_value)
            value_clipped = True
        
        # Apply direction-specific scaling
        gradients = gradients * scaling_factor
        
        return {
            'gradients': gradients,
            'factors': {
                'direction_max_norm': adjusted_max_norm,
                'direction_clip_value': adjusted_clip_value,
                'direction_scaling': scaling_factor,
                'confidence_multiplier': confidence_multiplier
            },
            'applied_operations': {
                'norm_clipped': norm_clipped,
                'value_clipped': value_clipped,
                'scaling_applied': True
            },
            'direction': direction.value,
            'original_bounds': bounds
        }
    
    def _apply_confidence_weighted_bounding(self, gradients: torch.Tensor, direction_state: DirectionState, confidence: float) -> Dict[str, Any]:
        """Apply confidence-weighted bounding adjustments"""
        # Lower confidence = more conservative bounding
        confidence_factor = max(0.1, confidence)
        conservative_multiplier = confidence_factor ** 2
        
        # Scale down gradients when confidence is low
        uncertainty_penalty = 1.0 - (1.0 - confidence) * self.direction_sensitivity
        adjusted_gradients = gradients * uncertainty_penalty
        
        # Additional norm constraint for low confidence
        if confidence < 0.5:
            max_uncertain_norm = 0.5 * confidence_factor
            current_norm = torch.norm(adjusted_gradients, p=2)
            if current_norm > max_uncertain_norm:
                scale_factor = max_uncertain_norm / current_norm
                adjusted_gradients = adjusted_gradients * scale_factor
        
        return {
            'gradients': adjusted_gradients,
            'factors': {
                'confidence_factor': confidence_factor,
                'uncertainty_penalty': uncertainty_penalty,
                'conservative_multiplier': conservative_multiplier
            },
            'confidence': confidence,
            'applied': confidence < self.confidence_threshold
        }
    
    def _apply_transition_smoothing(self, gradients: torch.Tensor, direction_state: DirectionState) -> Dict[str, Any]:
        """Apply smoothing during direction transitions"""
        if not self.direction_state_manager or len(self.direction_state_manager.direction_history) < 2:
            return {
                'gradients': gradients,
                'factors': {},
                'applied': False,
                'reason': 'insufficient_history'
            }
        
        # Check for recent direction transition
        history = list(self.direction_state_manager.direction_history)
        recent_directions = [h.direction for h in history[-3:]]
        
        # Detect transition
        transition_detected = len(set(recent_directions)) > 1
        
        if not transition_detected:
            return {
                'gradients': gradients,
                'factors': {},
                'applied': False,
                'reason': 'no_transition'
            }
        
        # Apply transition smoothing
        smoothing_factor = 0.7  # Reduce gradient magnitude during transitions
        transition_penalty = direction_state.confidence * 0.5 + 0.5  # Confidence-based penalty
        
        smoothed_gradients = gradients * smoothing_factor * transition_penalty
        
        return {
            'gradients': smoothed_gradients,
            'factors': {
                'smoothing_factor': smoothing_factor,
                'transition_penalty': transition_penalty,
                'total_smoothing': smoothing_factor * transition_penalty
            },
            'applied': True,
            'recent_directions': [d.value for d in recent_directions],
            'transition_detected': transition_detected
        }
    
    def _apply_momentum_adjustment(self, gradients: torch.Tensor, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply momentum-based gradient adjustments"""
        momentum_applied = False
        momentum_factor = 1.0
        previous_momentum = self.momentum_gradients is not None
        
        # Update momentum gradients
        if self.momentum_gradients is None:
            self.momentum_gradients = gradients.clone()
            # On first call, no momentum adjustment is applied yet
            adjusted_gradients = gradients
        else:
            # Exponential moving average
            self.momentum_gradients = (
                self.momentum_factor * self.momentum_gradients + 
                (1 - self.momentum_factor) * gradients
            )
            
            # Apply momentum decay
            self.momentum_gradients *= (1 - self.momentum_decay_rate)
            
            # Calculate momentum influence
            # Cosine similarity between current and momentum gradients
            grad_norm = torch.norm(gradients)
            momentum_norm = torch.norm(self.momentum_gradients)
            
            if grad_norm > 1e-8 and momentum_norm > 1e-8:
                cosine_sim = torch.dot(
                    gradients.flatten() / grad_norm,
                    self.momentum_gradients.flatten() / momentum_norm
                ).item()
                
                # Adjust based on alignment with momentum
                if cosine_sim > 0.5:  # Aligned with momentum
                    momentum_factor = 1.0 + 0.1 * cosine_sim
                elif cosine_sim < -0.5:  # Opposed to momentum
                    momentum_factor = 1.0 - 0.2 * abs(cosine_sim)
                
                momentum_applied = abs(momentum_factor - 1.0) > 0.01
            
            adjusted_gradients = gradients * momentum_factor
        
        return {
            'gradients': adjusted_gradients,
            'factors': {
                'momentum_factor': momentum_factor,
                'momentum_norm': torch.norm(self.momentum_gradients).item() if self.momentum_gradients is not None else 0.0,
                'cosine_similarity': cosine_sim if 'cosine_sim' in locals() else 0.0
            },
            'applied': momentum_applied and previous_momentum,  # Only applied if there was previous momentum
            'momentum_available': self.momentum_gradients is not None
        }
    
    def _apply_predictive_bounding(self, gradients: torch.Tensor, direction_state: DirectionState, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply predictive bounding based on direction trends"""
        if not self.direction_state_manager or len(self.direction_state_manager.direction_history) < 5:
            return {
                'gradients': gradients,
                'factors': {},
                'applied': False,
                'reason': 'insufficient_history'
            }
        
        # Analyze recent trends
        history = list(self.direction_state_manager.direction_history)
        recent_magnitudes = [h.magnitude for h in history[-5:]]
        recent_confidences = [h.confidence for h in history[-5:]]
        
        # Predict future behavior
        try:
            magnitude_trend = np.polyfit(range(len(recent_magnitudes)), recent_magnitudes, 1)[0]
            confidence_trend = np.polyfit(range(len(recent_confidences)), recent_confidences, 1)[0]
        except (np.linalg.LinAlgError, ValueError):
            # Handle edge cases where polyfit might fail
            magnitude_trend = 0.0
            confidence_trend = 0.0
        
        # Apply predictive adjustments
        predictive_factor = 1.0
        prediction_applied = False
        
        # If magnitude is increasing rapidly, apply preventive scaling
        if magnitude_trend > 1.0:
            predictive_factor *= 0.8  # Preemptive scaling down
            prediction_applied = True
        
        # If confidence is declining, be more conservative
        if confidence_trend < -0.1:
            predictive_factor *= 0.9
            prediction_applied = True
        
        # Direction-specific predictions
        if direction_state.direction == DirectionType.OSCILLATING:
            # Oscillations tend to amplify, so dampen proactively
            predictive_factor *= 0.85
            prediction_applied = True
        elif direction_state.direction == DirectionType.ASCENT and magnitude_trend > 0.3:  # Lower threshold for ascending trends
            # Ascending gradients with increasing magnitude
            predictive_factor *= 0.9
            prediction_applied = True
        
        adjusted_gradients = gradients * predictive_factor
        
        return {
            'gradients': adjusted_gradients,
            'factors': {
                'predictive_factor': predictive_factor,
                'magnitude_trend': magnitude_trend,
                'confidence_trend': confidence_trend
            },
            'applied': prediction_applied,
            'predictions': {
                'magnitude_increasing': magnitude_trend > 0.5,
                'confidence_declining': confidence_trend < -0.1,
                'direction': direction_state.direction.value
            }
        }
    
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get comprehensive enhanced bounding statistics"""
        basic_stats = self.basic_bounder.get_bounding_statistics()
        
        total_bounds = sum(self.direction_specific_bounds.values())
        
        enhanced_stats = {
            "basic_statistics": basic_stats,
            "enhanced_operations": {
                "total_enhanced_bounds": total_bounds,
                "confidence_adjustments": self.confidence_adjustments,
                "predictive_bounds": self.predictive_bounds,
                "transition_smoothings": self.transition_smoothings
            },
            "direction_specific_bounds": {
                dt.value: count for dt, count in self.direction_specific_bounds.items()
            },
            "enhancement_performance": {
                "total_enhancement_overhead": self.enhancement_overhead,
                "average_enhancement_time": self.enhancement_overhead / max(1, total_bounds),
                "last_enhancement_time": self.last_enhancement_time
            },
            "feature_status": {
                "momentum_adjustment": self.enable_momentum_adjustment,
                "predictive_bounding": self.enable_predictive_bounding,
                "confidence_weighting": self.enable_confidence_weighting,
                "transition_smoothing": self.enable_transition_smoothing
            },
            "component_status": {
                "direction_state_manager": self.direction_state_manager is not None,
                "direction_validator": self.direction_validator is not None
            }
        }
        
        return enhanced_stats
    
    def reset_statistics(self):
        """Reset all statistics including basic bounder"""
        self.basic_bounder.reset_statistics()
        self.direction_specific_bounds = {dt: 0 for dt in DirectionType}
        self.confidence_adjustments = 0
        self.predictive_bounds = 0
        self.transition_smoothings = 0
        self.enhancement_overhead = 0.0
        self.last_enhancement_time = 0.0
    
    def update_direction_bounds(self, direction: DirectionType, new_bounds: Dict[str, float]):
        """Update direction-specific bounding parameters"""
        if direction in self.direction_bounds:
            self.direction_bounds[direction].update(new_bounds)
            logger.info(f"Updated bounds for {direction.value}: {new_bounds}")
        else:
            logger.warning(f"Unknown direction type: {direction}")
    
    def get_direction_bounds(self, direction: DirectionType) -> Dict[str, float]:
        """Get current bounds for a specific direction"""
        return self.direction_bounds.get(direction, {}).copy()
    
    def is_enhancement_healthy(self) -> bool:
        """Check if enhancement system is functioning properly"""
        # Check component availability
        if not self.direction_state_manager or not self.direction_validator:
            return False
        
        # Check performance overhead
        total_bounds = sum(self.direction_specific_bounds.values())
        if total_bounds > 0:
            avg_overhead = self.enhancement_overhead / total_bounds
            if avg_overhead > 0.1:  # More than 100ms per operation
                return False
        
        return True
    
    def get_optimal_bounds_recommendation(self) -> Dict[str, Any]:
        """Get recommendations for optimal bounding parameters"""
        if not self.direction_state_manager:
            return {"status": "no_direction_manager"}
        
        stats = self.direction_state_manager.get_direction_summary()
        recommendations = {}
        
        # Analyze direction-specific performance
        for direction in DirectionType:
            direction_name = direction.value
            bound_count = self.direction_specific_bounds[direction]
            
            if bound_count > 10:  # Sufficient data
                current_bounds = self.direction_bounds[direction]
                
                # Basic recommendations based on usage patterns
                if bound_count > 100:  # High usage
                    rec_max_norm = current_bounds['max_norm'] * 1.1
                    rec_clip_value = current_bounds['clip_value'] * 1.1
                else:  # Low usage, be more conservative
                    rec_max_norm = current_bounds['max_norm'] * 0.9
                    rec_clip_value = current_bounds['clip_value'] * 0.9
                
                recommendations[direction_name] = {
                    "max_norm": rec_max_norm,
                    "clip_value": rec_clip_value,
                    "scaling_factor": current_bounds['scaling_factor'],
                    "usage_count": bound_count
                }
        
        return {
            "recommendations": recommendations,
            "analysis_basis": "usage_patterns",
            "total_operations": sum(self.direction_specific_bounds.values())
        }