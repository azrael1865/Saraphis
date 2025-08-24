"""
GAC Direction State Management
Manages gradient direction tracking and state transitions for the GAC system
"""

import torch
import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple, Deque
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime
import logging

try:
    from .gac_types import DirectionType, DirectionState
except ImportError:
    from gac_types import DirectionType, DirectionState

logger = logging.getLogger(__name__)


@dataclass
class DirectionHistory:
    """Historical direction information"""
    direction: DirectionType
    confidence: float
    magnitude: float
    timestamp: float
    gradient_norm: float
    gradient_variance: float


class DirectionStateManager:
    """Manages gradient direction state tracking and transitions"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
            
        # Configuration
        self.history_size = config.get('history_size', 100)
        self.smoothing_factor = config.get('smoothing_factor', 0.8)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.stability_window = config.get('stability_window', 10)
        self.oscillation_threshold = config.get('oscillation_threshold', 3)
        
        # Direction detection thresholds
        self.magnitude_threshold = config.get('magnitude_threshold', 1e-6)
        self.direction_change_threshold = config.get('direction_change_threshold', 0.5)
        self.stable_variance_threshold = config.get('stable_variance_threshold', 0.1)
        
        # State tracking
        self.current_state: Optional[DirectionState] = None
        self.direction_history: Deque[DirectionHistory] = deque(maxlen=self.history_size)
        self.gradient_history: Deque[torch.Tensor] = deque(maxlen=self.history_size)
        
        # Smoothed metrics
        self.smoothed_magnitude = 0.0
        self.smoothed_direction_vector = None
        self.stability_score = 0.0
        
        # Statistics
        self.direction_transitions = 0
        self.total_updates = 0
        self.last_update_time = 0.0
        
        # Direction transition tracking
        self.direction_transition_history: Deque[Tuple[DirectionType, DirectionType, float]] = deque(maxlen=50)
        
    def update_direction_state(self, gradients: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> DirectionState:
        """Update direction state based on new gradients"""
        if gradients is None:
            raise ValueError("Gradients cannot be None")
        if not isinstance(gradients, torch.Tensor):
            raise TypeError("Gradients must be torch.Tensor")
        if gradients.numel() == 0:
            raise ValueError("Gradients tensor cannot be empty")
        
        if context is None:
            context = {}
            
        current_time = time.time()
        
        # Store gradient for history
        self.gradient_history.append(gradients.clone().detach())
        
        # Calculate gradient metrics
        gradient_norm = torch.norm(gradients).item()
        gradient_mean = gradients.mean().item()
        gradient_variance = gradients.var().item()
        
        # Update smoothed magnitude (initialize on first update)
        if self.smoothed_magnitude == 0.0 and self.total_updates == 0:
            self.smoothed_magnitude = gradient_norm
        else:
            self.smoothed_magnitude = (
                self.smoothing_factor * self.smoothed_magnitude + 
                (1 - self.smoothing_factor) * gradient_norm
            )
        
        # Update smoothed direction vector
        if gradient_norm > self.magnitude_threshold:
            normalized_gradients = gradients / gradient_norm
            if self.smoothed_direction_vector is None:
                self.smoothed_direction_vector = normalized_gradients.clone()
            else:
                self.smoothed_direction_vector = (
                    self.smoothing_factor * self.smoothed_direction_vector +
                    (1 - self.smoothing_factor) * normalized_gradients
                )
        
        # Determine current direction
        direction_type = self._determine_direction_type(gradients, gradient_norm, gradient_variance)
        
        # Calculate confidence
        confidence = self._calculate_direction_confidence(direction_type, gradients, context)
        
        # Update stability score
        self.stability_score = self._calculate_stability_score()
        
        # Create direction state
        new_state = DirectionState(
            direction=direction_type,
            confidence=confidence,
            magnitude=gradient_norm,
            timestamp=current_time,
            metadata={
                'gradient_mean': gradient_mean,
                'gradient_variance': gradient_variance,
                'smoothed_magnitude': self.smoothed_magnitude,
                'stability_score': self.stability_score,
                'history_length': len(self.direction_history),
                'context': context
            }
        )
        
        # Check for direction transition
        if self.current_state and self.current_state.direction != direction_type:
            self.direction_transitions += 1
            self.direction_transition_history.append((
                self.current_state.direction,
                direction_type,
                current_time
            ))
            logger.debug(f"Direction transition: {self.current_state.direction.value} -> {direction_type.value}")
        
        # Store in history
        history_entry = DirectionHistory(
            direction=direction_type,
            confidence=confidence,
            magnitude=gradient_norm,
            timestamp=current_time,
            gradient_norm=gradient_norm,
            gradient_variance=gradient_variance
        )
        self.direction_history.append(history_entry)
        
        # Update state
        self.current_state = new_state
        self.total_updates += 1
        self.last_update_time = current_time
        
        return new_state
    
    def _determine_direction_type(self, gradients: torch.Tensor, gradient_norm: float, gradient_variance: float) -> DirectionType:
        """Determine the current gradient direction type"""
        
        # Check for near-zero gradients (stable)
        if gradient_norm < self.magnitude_threshold:
            return DirectionType.STABLE
        
        # Check for high variance (potentially oscillating)
        if gradient_variance > self.stable_variance_threshold * gradient_norm:
            # Further check for oscillation pattern
            if self._is_oscillating():
                return DirectionType.OSCILLATING
        
        # Determine ascent vs descent based on gradient history
        if len(self.gradient_history) < 2:
            # Not enough history, classify based on magnitude
            return DirectionType.ASCENT if gradient_norm > self.smoothed_magnitude else DirectionType.STABLE
        
        # Compare with recent gradients to determine trend
        recent_direction = self._analyze_gradient_trend()
        
        return recent_direction
    
    def _analyze_gradient_trend(self) -> DirectionType:
        """Analyze gradient trend from recent history"""
        if len(self.gradient_history) < 3:
            return DirectionType.STABLE
        
        # Get recent gradients
        recent_gradients = list(self.gradient_history)[-min(5, len(self.gradient_history)):]
        recent_norms = [torch.norm(g).item() for g in recent_gradients]
        
        # Calculate trend
        if len(recent_norms) >= 3:
            # Simple linear trend analysis
            x = np.arange(len(recent_norms))
            coeffs = np.polyfit(x, recent_norms, 1)
            slope = coeffs[0]
            
            # Classify based on slope relative to mean magnitude
            mean_norm = np.mean(recent_norms)
            relative_slope = slope / (mean_norm + 1e-8)
            
            # Use a more sensitive threshold for trend detection
            # A relative slope of 0.2 means 20% increase per step
            trend_threshold = self.direction_change_threshold * 0.4  # More sensitive
            
            if relative_slope > trend_threshold:
                return DirectionType.ASCENT
            elif relative_slope < -trend_threshold:
                return DirectionType.DESCENT
            else:
                return DirectionType.STABLE
        
        # Fallback: compare first and last
        if recent_norms[-1] > recent_norms[0] * (1 + self.direction_change_threshold):
            return DirectionType.ASCENT
        elif recent_norms[-1] < recent_norms[0] * (1 - self.direction_change_threshold):
            return DirectionType.DESCENT
        else:
            return DirectionType.STABLE
    
    def _is_oscillating(self) -> bool:
        """Check if gradients are oscillating"""
        if len(self.direction_history) < self.oscillation_threshold * 2:
            return False
        
        # Count direction changes in recent history
        recent_history = list(self.direction_history)[-self.oscillation_threshold * 2:]
        direction_changes = 0
        
        for i in range(1, len(recent_history)):
            if recent_history[i].direction != recent_history[i-1].direction:
                direction_changes += 1
        
        # If we have many direction changes, it's oscillating
        return direction_changes >= self.oscillation_threshold
    
    def _calculate_direction_confidence(self, direction_type: DirectionType, gradients: torch.Tensor, context: Dict[str, Any]) -> float:
        """Calculate confidence in the direction determination"""
        confidence_factors = []
        
        # Factor 1: Gradient magnitude consistency
        if len(self.gradient_history) >= 3:
            recent_norms = [torch.norm(g).item() for g in list(self.gradient_history)[-3:]]
            norm_consistency = 1.0 - (np.std(recent_norms) / (np.mean(recent_norms) + 1e-8))
            confidence_factors.append(max(0.0, min(1.0, norm_consistency)))
        else:
            confidence_factors.append(0.5)  # Neutral confidence
        
        # Factor 2: Direction vector consistency
        if self.smoothed_direction_vector is not None and torch.norm(gradients) > self.magnitude_threshold:
            normalized_gradients = gradients / torch.norm(gradients)
            cosine_similarity = torch.dot(
                normalized_gradients.flatten(),
                self.smoothed_direction_vector.flatten()
            ).item()
            direction_consistency = (cosine_similarity + 1.0) / 2.0  # Map [-1,1] to [0,1]
            confidence_factors.append(direction_consistency)
        else:
            confidence_factors.append(0.5)
        
        # Factor 3: Historical direction consistency
        if len(self.direction_history) >= 3:
            recent_directions = [h.direction for h in list(self.direction_history)[-3:]]
            same_direction_count = sum(1 for d in recent_directions if d == direction_type)
            historical_consistency = same_direction_count / len(recent_directions)
            confidence_factors.append(historical_consistency)
        else:
            confidence_factors.append(0.5)
        
        # Factor 4: Stability score
        confidence_factors.append(self.stability_score)
        
        # Factor 5: Context-based factors
        if 'learning_rate' in context:
            lr = context['learning_rate']
            # Higher learning rates can lead to less stable directions
            lr_factor = 1.0 / (1.0 + 10 * lr)
            confidence_factors.append(lr_factor)
        
        if 'iteration' in context:
            iteration = context['iteration']
            # Early iterations have lower confidence
            iteration_factor = min(1.0, iteration / 1000.0)
            confidence_factors.append(iteration_factor)
        
        # Special case adjustments
        if direction_type == DirectionType.OSCILLATING:
            # Lower confidence for oscillating detection
            confidence_factors = [f * 0.8 for f in confidence_factors]
        elif direction_type == DirectionType.STABLE and torch.norm(gradients).item() < self.magnitude_threshold:
            # High confidence for genuinely small gradients - override other factors
            # When gradients are very small, we're confident it's stable
            return 0.9
        
        # Combine factors (weighted average)
        weights = [0.3, 0.3, 0.2, 0.1, 0.05, 0.05][:len(confidence_factors)]
        weights = weights + [0.0] * (len(confidence_factors) - len(weights))
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        confidence = sum(f * w for f, w in zip(confidence_factors, weights))
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_stability_score(self) -> float:
        """Calculate current stability score"""
        if len(self.direction_history) < self.stability_window:
            return 0.5  # Neutral stability
        
        recent_history = list(self.direction_history)[-self.stability_window:]
        
        # Factor 1: Direction consistency (penalize oscillation)
        directions = [h.direction for h in recent_history]
        most_common_direction = max(set(directions), key=directions.count)
        direction_consistency = directions.count(most_common_direction) / len(directions)
        
        # Penalize direction changes
        direction_changes = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i-1])
        change_penalty = direction_changes / (len(directions) - 1) if len(directions) > 1 else 0
        direction_consistency *= (1.0 - change_penalty)
        
        # Factor 2: Magnitude consistency
        magnitudes = [h.magnitude for h in recent_history]
        magnitude_cv = np.std(magnitudes) / (np.mean(magnitudes) + 1e-8)  # Coefficient of variation
        magnitude_consistency = 1.0 / (1.0 + magnitude_cv)
        
        # Factor 3: Confidence consistency
        confidences = [h.confidence for h in recent_history]
        confidence_consistency = 1.0 - np.std(confidences)
        
        # Combine factors
        stability = (
            0.5 * direction_consistency +
            0.3 * magnitude_consistency +
            0.2 * max(0.0, confidence_consistency)
        )
        
        return max(0.0, min(1.0, stability))
    
    def get_direction_summary(self) -> Dict[str, Any]:
        """Get comprehensive direction state summary"""
        if not self.current_state:
            return {"status": "no_state"}
        
        # Recent direction distribution
        recent_directions = [h.direction for h in list(self.direction_history)[-20:]]
        direction_counts = {dt: recent_directions.count(dt) for dt in DirectionType}
        
        # Transition analysis
        transition_matrix = self._calculate_transition_matrix()
        
        return {
            "current_state": {
                "direction": self.current_state.direction.value,
                "confidence": self.current_state.confidence,
                "magnitude": self.current_state.magnitude,
                "timestamp": self.current_state.timestamp,
                "stability_score": self.stability_score
            },
            "statistics": {
                "total_updates": self.total_updates,
                "direction_transitions": self.direction_transitions,
                "history_length": len(self.direction_history),
                "smoothed_magnitude": self.smoothed_magnitude
            },
            "recent_distribution": {
                dt.value: count for dt, count in direction_counts.items()
            },
            "transition_matrix": transition_matrix,
            "performance_metrics": {
                "average_confidence": np.mean([h.confidence for h in self.direction_history]) if self.direction_history else 0.0,
                "stability_trend": self._get_stability_trend(),
                "direction_persistence": self._calculate_direction_persistence()
            }
        }
    
    def _calculate_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Calculate direction transition probability matrix"""
        if len(self.direction_transition_history) < 2:
            return {}
        
        # Count transitions
        transition_counts = defaultdict(lambda: defaultdict(int))
        total_transitions = defaultdict(int)
        
        for from_dir, to_dir, _ in self.direction_transition_history:
            transition_counts[from_dir][to_dir] += 1
            total_transitions[from_dir] += 1
        
        # Convert to probabilities
        transition_matrix = {}
        for from_dir in DirectionType:
            transition_matrix[from_dir.value] = {}
            for to_dir in DirectionType:
                if total_transitions[from_dir] > 0:
                    prob = transition_counts[from_dir][to_dir] / total_transitions[from_dir]
                else:
                    prob = 0.0
                transition_matrix[from_dir.value][to_dir.value] = prob
        
        return transition_matrix
    
    def _get_stability_trend(self) -> str:
        """Get stability trend description"""
        if len(self.direction_history) < 10:
            return "insufficient_data"
        
        recent_stability_scores = []
        window_size = 5
        
        for i in range(len(self.direction_history) - window_size + 1):
            window_history = list(self.direction_history)[i:i + window_size]
            directions = [h.direction for h in window_history]
            consistency = len(set(directions)) == 1
            recent_stability_scores.append(1.0 if consistency else 0.0)
        
        if len(recent_stability_scores) >= 3:
            trend = recent_stability_scores[-1] - recent_stability_scores[0]
            if trend > 0.2:
                return "improving"
            elif trend < -0.2:
                return "degrading"
            else:
                return "stable"
        
        return "unknown"
    
    def _calculate_direction_persistence(self) -> float:
        """Calculate how persistent current direction is"""
        if not self.current_state or len(self.direction_history) < 3:
            return 0.0
        
        current_dir = self.current_state.direction
        recent_history = list(self.direction_history)[-10:]
        
        persistence_count = sum(1 for h in recent_history if h.direction == current_dir)
        return persistence_count / len(recent_history)
    
    def reset(self):
        """Reset direction state manager"""
        self.current_state = None
        self.direction_history.clear()
        self.gradient_history.clear()
        self.direction_transition_history.clear()
        self.smoothed_magnitude = 0.0
        self.smoothed_direction_vector = None
        self.stability_score = 0.0
        self.direction_transitions = 0
        self.total_updates = 0
        self.last_update_time = 0.0
    
    def get_current_state(self) -> Optional[DirectionState]:
        """Get current direction state"""
        return self.current_state
    
    def is_stable(self) -> bool:
        """Check if current direction is stable"""
        return (self.current_state is not None and 
                self.current_state.direction == DirectionType.STABLE and
                self.current_state.confidence >= self.confidence_threshold)
    
    def is_oscillating(self) -> bool:
        """Check if current direction is oscillating"""
        return (self.current_state is not None and 
                self.current_state.direction == DirectionType.OSCILLATING and
                self.current_state.confidence >= self.confidence_threshold)
    
    def get_direction_trend(self, window_size: int = 5) -> Optional[DirectionType]:
        """Get dominant direction trend over recent history"""
        if len(self.direction_history) < window_size:
            return None
        
        recent_directions = [h.direction for h in list(self.direction_history)[-window_size:]]
        most_common = max(set(recent_directions), key=recent_directions.count)
        
        # Only return if it's clearly dominant
        if recent_directions.count(most_common) >= window_size * 0.6:
            return most_common
        
        return None
    
    @property
    def current_direction(self) -> Optional[DirectionType]:
        """Get current direction type"""
        return self.current_state.direction if self.current_state else None
    
    @property
    def current_confidence(self) -> float:
        """Get current confidence"""
        return self.current_state.confidence if self.current_state else 0.0
    
    @property
    def current_magnitude(self) -> float:
        """Get current magnitude"""
        return self.current_state.magnitude if self.current_state else 0.0