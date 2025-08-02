"""
GAC Direction Validator
Validates gradient direction consistency and detects anomalies in direction patterns
"""

import torch
import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from datetime import datetime
import logging

from .gac_types import DirectionType, DirectionState
from .direction_state import DirectionStateManager

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of direction validation"""
    is_valid: bool
    confidence: float
    validation_type: str
    issues: List[str]
    metadata: Dict[str, Any]


@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    anomaly_detected: bool
    anomaly_type: str
    severity: float  # 0.0 to 1.0
    description: str
    timestamp: float
    affected_metrics: List[str]


class DirectionValidationError(Exception):
    """Raised when direction validation fails critically"""
    pass


class DirectionValidator:
    """Validates gradient direction consistency and detects anomalies"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
            
        # Configuration
        self.consistency_threshold = config.get('consistency_threshold', 0.8)
        self.anomaly_threshold = config.get('anomaly_threshold', 0.7)
        self.validation_window = config.get('validation_window', 10)
        self.max_direction_changes = config.get('max_direction_changes', 5)
        self.magnitude_deviation_threshold = config.get('magnitude_deviation_threshold', 3.0)
        
        # Validation thresholds
        self.min_confidence_threshold = config.get('min_confidence_threshold', 0.3)
        self.stability_threshold = config.get('stability_threshold', 0.5)
        self.oscillation_frequency_threshold = config.get('oscillation_frequency_threshold', 0.3)
        
        # Pattern detection
        self.pattern_memory_size = config.get('pattern_memory_size', 50)
        self.enable_anomaly_detection = config.get('enable_anomaly_detection', True)
        self.enable_pattern_validation = config.get('enable_pattern_validation', True)
        
        # Validation history
        self.validation_history: deque = deque(maxlen=100)
        self.anomaly_history: deque = deque(maxlen=50)
        self.pattern_violations: deque = deque(maxlen=30)
        
        # Statistical tracking
        self.direction_statistics: Dict[DirectionType, Dict[str, float]] = {
            dt: {'count': 0, 'total_confidence': 0.0, 'avg_magnitude': 0.0, 'violations': 0}
            for dt in DirectionType
        }
        
        # Validation performance
        self.total_validations = 0
        self.failed_validations = 0
        self.anomalies_detected = 0
        
        # State reference
        self.direction_state_manager: Optional[DirectionStateManager] = None
        
    def set_direction_state_manager(self, manager: DirectionStateManager):
        """Set reference to direction state manager"""
        if manager is None:
            raise ValueError("Direction state manager cannot be None")
        self.direction_state_manager = manager
        
    def validate_direction_state(self, direction_state: DirectionState, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate a direction state for consistency and correctness"""
        if direction_state is None:
            raise DirectionValidationError("Direction state cannot be None")
        
        if context is None:
            context = {}
            
        self.total_validations += 1
        issues = []
        validation_metadata = {}
        
        # Validation 1: Basic state validation
        basic_result = self._validate_basic_state(direction_state)
        if not basic_result.is_valid:
            issues.extend(basic_result.issues)
        validation_metadata.update(basic_result.metadata)
        
        # Validation 2: Confidence validation
        confidence_result = self._validate_confidence(direction_state, context)
        if not confidence_result.is_valid:
            issues.extend(confidence_result.issues)
        validation_metadata.update(confidence_result.metadata)
        
        # Validation 3: Historical consistency validation
        if self.direction_state_manager:
            consistency_result = self._validate_historical_consistency(direction_state)
            if not consistency_result.is_valid:
                issues.extend(consistency_result.issues)
            validation_metadata.update(consistency_result.metadata)
        
        # Validation 4: Pattern validation
        if self.enable_pattern_validation:
            pattern_result = self._validate_direction_patterns(direction_state)
            if not pattern_result.is_valid:
                issues.extend(pattern_result.issues)
            validation_metadata.update(pattern_result.metadata)
        
        # Validation 5: Anomaly detection
        if self.enable_anomaly_detection:
            anomaly_result = self._detect_anomalies(direction_state, context)
            if anomaly_result.anomaly_detected:
                issues.append(f"Anomaly detected: {anomaly_result.description}")
                validation_metadata['anomaly'] = {
                    'type': anomaly_result.anomaly_type,
                    'severity': anomaly_result.severity,
                    'affected_metrics': anomaly_result.affected_metrics
                }
        
        # Calculate overall validation confidence
        individual_confidences = [
            basic_result.confidence,
            confidence_result.confidence,
            consistency_result.confidence if self.direction_state_manager else 1.0,
            pattern_result.confidence if self.enable_pattern_validation else 1.0
        ]
        overall_confidence = np.mean(individual_confidences)
        
        # Determine overall validity
        is_valid = len(issues) == 0 and overall_confidence >= self.consistency_threshold
        
        # Create result
        result = ValidationResult(
            is_valid=is_valid,
            confidence=overall_confidence,
            validation_type="comprehensive",
            issues=issues,
            metadata=validation_metadata
        )
        
        # Update statistics
        self._update_validation_statistics(direction_state, result)
        
        # Store in history
        self.validation_history.append({
            'timestamp': time.time(),
            'direction': direction_state.direction.value,
            'is_valid': is_valid,
            'confidence': overall_confidence,
            'issues_count': len(issues)
        })
        
        if not is_valid:
            self.failed_validations += 1
            logger.warning(f"Direction validation failed for {direction_state.direction.value}: {issues}")
        
        return result
        
    def _validate_basic_state(self, direction_state: DirectionState) -> ValidationResult:
        """Validate basic state properties"""
        issues = []
        metadata = {}
        
        # Check confidence range
        if not (0.0 <= direction_state.confidence <= 1.0):
            issues.append(f"Invalid confidence value: {direction_state.confidence}")
        
        # Check magnitude
        if direction_state.magnitude < 0:
            issues.append(f"Invalid negative magnitude: {direction_state.magnitude}")
        
        # Check timestamp
        current_time = time.time()
        if direction_state.timestamp > current_time + 1.0:  # Allow small future tolerance
            issues.append(f"Invalid future timestamp: {direction_state.timestamp}")
        
        # Check for extremely old timestamps
        if current_time - direction_state.timestamp > 3600:  # 1 hour
            issues.append(f"Timestamp too old: {current_time - direction_state.timestamp:.2f} seconds")
        
        # Direction-specific validations
        if direction_state.direction == DirectionType.STABLE:
            if direction_state.magnitude > 10.0:  # Stable should have small magnitude
                issues.append(f"Stable direction with high magnitude: {direction_state.magnitude}")
        elif direction_state.direction == DirectionType.OSCILLATING:
            if direction_state.confidence > 0.9:  # Oscillating should have some uncertainty
                issues.append(f"Oscillating direction with very high confidence: {direction_state.confidence}")
        
        # Check metadata
        if not isinstance(direction_state.metadata, dict):
            issues.append("Invalid metadata type")
        
        metadata['basic_checks'] = {
            'confidence_valid': 0.0 <= direction_state.confidence <= 1.0,
            'magnitude_valid': direction_state.magnitude >= 0,
            'timestamp_valid': 0 <= current_time - direction_state.timestamp <= 3600,
            'metadata_valid': isinstance(direction_state.metadata, dict)
        }
        
        confidence = 1.0 - (len(issues) / 6.0)  # 6 possible issues
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence=max(0.0, confidence),
            validation_type="basic",
            issues=issues,
            metadata=metadata
        )
        
    def _validate_confidence(self, direction_state: DirectionState, context: Dict[str, Any]) -> ValidationResult:
        """Validate confidence levels"""
        issues = []
        metadata = {}
        
        confidence = direction_state.confidence
        
        # Check minimum confidence threshold
        if confidence < self.min_confidence_threshold:
            issues.append(f"Confidence below minimum threshold: {confidence:.3f} < {self.min_confidence_threshold}")
        
        # Context-specific confidence validation
        if 'iteration' in context:
            iteration = context['iteration']
            # Early iterations should have lower confidence expectations
            if iteration < 100 and confidence > 0.95:
                issues.append(f"Unrealistically high confidence in early training: {confidence:.3f}")
        
        if 'learning_rate' in context:
            lr = context['learning_rate']
            # High learning rates should generally have lower confidence
            if lr > 0.01 and confidence > 0.9:
                issues.append(f"High confidence with high learning rate: conf={confidence:.3f}, lr={lr}")
        
        # Direction-specific confidence validation
        if direction_state.direction == DirectionType.OSCILLATING:
            if confidence > 0.8:
                issues.append(f"Unexpectedly high confidence for oscillating direction: {confidence:.3f}")
        elif direction_state.direction == DirectionType.STABLE:
            if direction_state.magnitude < 1e-6 and confidence < 0.7:
                issues.append(f"Low confidence for clearly stable gradients: {confidence:.3f}")
        
        # Confidence consistency check
        metadata_confidence = direction_state.metadata.get('confidence_factors', {})
        if metadata_confidence:
            factor_mean = np.mean(list(metadata_confidence.values()))
            if abs(confidence - factor_mean) > 0.3:
                issues.append(f"Confidence inconsistent with factors: {confidence:.3f} vs {factor_mean:.3f}")
        
        metadata['confidence_checks'] = {
            'meets_minimum': confidence >= self.min_confidence_threshold,
            'contextually_appropriate': len(issues) == 0,
            'confidence_value': confidence
        }
        
        confidence_score = max(0.0, 1.0 - len(issues) * 0.2)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence=confidence_score,
            validation_type="confidence",
            issues=issues,
            metadata=metadata
        )
        
    def _validate_historical_consistency(self, direction_state: DirectionState) -> ValidationResult:
        """Validate consistency with historical patterns"""
        issues = []
        metadata = {}
        
        if not self.direction_state_manager or len(self.direction_state_manager.direction_history) < 3:
            return ValidationResult(
                is_valid=True,
                confidence=1.0,
                validation_type="historical",
                issues=[],
                metadata={'insufficient_history': True}
            )
        
        history = list(self.direction_state_manager.direction_history)
        current_direction = direction_state.direction
        
        # Check for rapid direction changes
        recent_directions = [h.direction for h in history[-self.validation_window:]]
        direction_changes = sum(1 for i in range(1, len(recent_directions)) 
                              if recent_directions[i] != recent_directions[i-1])
        
        if direction_changes > self.max_direction_changes:
            issues.append(f"Excessive direction changes: {direction_changes} in {len(recent_directions)} steps")
        
        # Check magnitude consistency
        recent_magnitudes = [h.magnitude for h in history[-5:]]
        if len(recent_magnitudes) >= 3:
            magnitude_std = np.std(recent_magnitudes)
            magnitude_mean = np.mean(recent_magnitudes)
            
            if magnitude_mean > 0 and magnitude_std / magnitude_mean > self.magnitude_deviation_threshold:
                issues.append(f"Inconsistent magnitude pattern: CV={magnitude_std/magnitude_mean:.3f}")
        
        # Check confidence consistency
        recent_confidences = [h.confidence for h in history[-5:]]
        if len(recent_confidences) >= 3:
            confidence_std = np.std(recent_confidences)
            if confidence_std > 0.3:
                issues.append(f"Highly variable confidence: std={confidence_std:.3f}")
        
        # Check for impossible transitions
        if len(history) >= 2:
            prev_direction = history[-1].direction
            prev_magnitude = history[-1].magnitude
            
            # Stable -> High magnitude direction without gradual transition
            if (prev_direction == DirectionType.STABLE and 
                current_direction != DirectionType.STABLE and
                prev_magnitude < 1e-6 and direction_state.magnitude > 10.0):
                issues.append("Implausible transition from stable to high magnitude")
        
        # Oscillation pattern validation
        if current_direction == DirectionType.OSCILLATING:
            oscillation_count = sum(1 for h in history[-10:] if h.direction == DirectionType.OSCILLATING)
            if oscillation_count < 3:
                issues.append("Insufficient oscillation pattern for oscillating classification")
        
        metadata['historical_checks'] = {
            'direction_changes': direction_changes,
            'magnitude_consistency': magnitude_std / magnitude_mean if 'magnitude_mean' in locals() and magnitude_mean > 0 else 0,
            'confidence_variability': confidence_std if 'confidence_std' in locals() else 0,
            'history_length': len(history)
        }
        
        consistency_score = max(0.0, 1.0 - len(issues) * 0.15)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence=consistency_score,
            validation_type="historical",
            issues=issues,
            metadata=metadata
        )
        
    def _validate_direction_patterns(self, direction_state: DirectionState) -> ValidationResult:
        """Validate direction against known patterns"""
        issues = []
        metadata = {}
        
        # Pattern 1: Stable periods should be genuinely stable
        if direction_state.direction == DirectionType.STABLE:
            if direction_state.magnitude > 1.0:
                issues.append(f"Stable direction with non-stable magnitude: {direction_state.magnitude}")
            
            stability_score = direction_state.metadata.get('stability_score', 0.5)
            if stability_score < self.stability_threshold:
                issues.append(f"Low stability score for stable direction: {stability_score}")
        
        # Pattern 2: Ascent/Descent should have clear trends
        if direction_state.direction in [DirectionType.ASCENT, DirectionType.DESCENT]:
            if direction_state.magnitude < 1e-6:
                issues.append(f"Directional movement with negligible magnitude: {direction_state.magnitude}")
        
        # Pattern 3: Oscillations should have reasonable frequency
        if direction_state.direction == DirectionType.OSCILLATING:
            # Check if we have enough direction changes to justify oscillation
            if self.direction_state_manager:
                transition_history = self.direction_state_manager.direction_transition_history
                recent_transitions = [t for t in transition_history if time.time() - t[2] < 100]  # Last 100 time units
                
                if len(recent_transitions) < 2:
                    issues.append("Insufficient transitions for oscillating classification")
                else:
                    # Check transition frequency
                    time_span = recent_transitions[-1][2] - recent_transitions[0][2]
                    if time_span > 0:
                        frequency = len(recent_transitions) / time_span
                        if frequency < self.oscillation_frequency_threshold:
                            issues.append(f"Low oscillation frequency: {frequency:.3f}")
        
        # Pattern 4: Confidence should correlate with pattern clarity
        expected_confidence = self._calculate_expected_confidence(direction_state)
        confidence_diff = abs(direction_state.confidence - expected_confidence)
        if confidence_diff > 0.3:
            issues.append(f"Confidence mismatch with pattern: expected {expected_confidence:.3f}, got {direction_state.confidence:.3f}")
        
        metadata['pattern_checks'] = {
            'direction_appropriate': direction_state.direction.value,
            'magnitude_appropriate': direction_state.magnitude > 1e-6 if direction_state.direction != DirectionType.STABLE else direction_state.magnitude <= 1.0,
            'confidence_appropriate': confidence_diff <= 0.3,
            'expected_confidence': expected_confidence
        }
        
        pattern_score = max(0.0, 1.0 - len(issues) * 0.2)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence=pattern_score,
            validation_type="pattern",
            issues=issues,
            metadata=metadata
        )
        
    def _calculate_expected_confidence(self, direction_state: DirectionState) -> float:
        """Calculate expected confidence based on direction characteristics"""
        base_confidence = 0.7
        
        # Adjust based on direction type
        if direction_state.direction == DirectionType.STABLE:
            if direction_state.magnitude < 1e-6:
                base_confidence = 0.9  # Very stable
            else:
                base_confidence = 0.6  # Moderately stable
        elif direction_state.direction == DirectionType.OSCILLATING:
            base_confidence = 0.5  # Inherently uncertain
        else:  # ASCENT or DESCENT
            base_confidence = 0.8  # Generally clear
        
        # Adjust based on magnitude
        if direction_state.magnitude > 10.0:
            base_confidence += 0.1  # High magnitude = clearer signal
        elif direction_state.magnitude < 0.1:
            base_confidence -= 0.1  # Low magnitude = less clear
        
        return max(0.0, min(1.0, base_confidence))
        
    def _detect_anomalies(self, direction_state: DirectionState, context: Dict[str, Any]) -> AnomalyDetection:
        """Detect anomalies in direction state"""
        current_time = time.time()
        
        # Anomaly 1: Confidence-magnitude mismatch
        if direction_state.magnitude > 5.0 and direction_state.confidence < 0.3:
            return AnomalyDetection(
                anomaly_detected=True,
                anomaly_type="confidence_magnitude_mismatch",
                severity=0.8,
                description="High magnitude with very low confidence",
                timestamp=current_time,
                affected_metrics=['confidence', 'magnitude']
            )
        
        # Anomaly 2: Impossible gradient magnitude
        if direction_state.magnitude > 1000.0:
            return AnomalyDetection(
                anomaly_detected=True,
                anomaly_type="extreme_magnitude",
                severity=1.0,
                description=f"Extreme gradient magnitude: {direction_state.magnitude}",
                timestamp=current_time,
                affected_metrics=['magnitude']
            )
        
        # Anomaly 3: Temporal anomaly
        if 'iteration' in context and context['iteration'] > 1000:
            # Late in training, should not have very high magnitudes unless something is wrong
            if direction_state.magnitude > 50.0:
                return AnomalyDetection(
                    anomaly_detected=True,
                    anomaly_type="late_training_instability",
                    severity=0.9,
                    description="High gradient magnitude late in training",
                    timestamp=current_time,
                    affected_metrics=['magnitude', 'training_stability']
                )
        
        # Anomaly 4: Confidence anomaly
        if direction_state.confidence > 0.95 and direction_state.direction == DirectionType.OSCILLATING:
            return AnomalyDetection(
                anomaly_detected=True,
                anomaly_type="overconfident_oscillation",
                severity=0.7,
                description="Unrealistically high confidence for oscillating gradients",
                timestamp=current_time,
                affected_metrics=['confidence']
            )
        
        # Anomaly 5: Statistical anomaly based on history
        if self.direction_state_manager and len(self.direction_state_manager.direction_history) >= 10:
            direction_stats = self.direction_statistics.get(direction_state.direction)
            if direction_stats and direction_stats['count'] > 0:
                avg_magnitude = direction_stats['avg_magnitude']
                if avg_magnitude > 0 and abs(direction_state.magnitude - avg_magnitude) > 3 * avg_magnitude:
                    return AnomalyDetection(
                        anomaly_detected=True,
                        anomaly_type="statistical_outlier",
                        severity=0.6,
                        description=f"Magnitude deviates significantly from historical average",
                        timestamp=current_time,
                        affected_metrics=['magnitude']
                    )
        
        # No anomaly detected
        return AnomalyDetection(
            anomaly_detected=False,
            anomaly_type="none",
            severity=0.0,
            description="No anomalies detected",
            timestamp=current_time,
            affected_metrics=[]
        )
        
    def _update_validation_statistics(self, direction_state: DirectionState, result: ValidationResult):
        """Update validation statistics"""
        direction = direction_state.direction
        stats = self.direction_statistics[direction]
        
        # Update counts
        stats['count'] += 1
        stats['total_confidence'] += direction_state.confidence
        
        # Update average magnitude
        if stats['count'] == 1:
            stats['avg_magnitude'] = direction_state.magnitude
        else:
            stats['avg_magnitude'] = (
                (stats['avg_magnitude'] * (stats['count'] - 1) + direction_state.magnitude) / 
                stats['count']
            )
        
        # Update violations
        if not result.is_valid:
            stats['violations'] += 1
            
    def validate_direction_transition(self, from_state: DirectionState, to_state: DirectionState) -> ValidationResult:
        """Validate a direction state transition"""
        if from_state is None or to_state is None:
            raise DirectionValidationError("Both states required for transition validation")
        
        issues = []
        metadata = {}
        
        # Check temporal consistency
        if to_state.timestamp <= from_state.timestamp:
            issues.append("Invalid timestamp order in transition")
        
        # Check transition validity
        time_diff = to_state.timestamp - from_state.timestamp
        magnitude_change = abs(to_state.magnitude - from_state.magnitude)
        
        # Validate rapid changes
        if time_diff < 1.0 and magnitude_change > 10.0:
            issues.append(f"Implausible rapid magnitude change: {magnitude_change:.3f} in {time_diff:.3f}s")
        
        # Direction transition logic
        if from_state.direction == DirectionType.STABLE:
            if to_state.direction != DirectionType.STABLE and to_state.magnitude < from_state.magnitude:
                issues.append("Transition from stable should increase magnitude")
        
        if from_state.direction == DirectionType.OSCILLATING:
            if to_state.direction == DirectionType.STABLE and time_diff < 5.0:
                issues.append("Too rapid transition from oscillating to stable")
        
        metadata['transition_analysis'] = {
            'time_diff': time_diff,
            'magnitude_change': magnitude_change,
            'from_direction': from_state.direction.value,
            'to_direction': to_state.direction.value,
            'confidence_change': to_state.confidence - from_state.confidence
        }
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence=1.0 - len(issues) * 0.2,
            validation_type="transition",
            issues=issues,
            metadata=metadata
        )
        
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics"""
        if self.total_validations == 0:
            return {"status": "no_validations"}
        
        success_rate = 1.0 - (self.failed_validations / self.total_validations)
        
        # Direction-specific statistics
        direction_stats = {}
        for direction, stats in self.direction_statistics.items():
            if stats['count'] > 0:
                direction_stats[direction.value] = {
                    'count': stats['count'],
                    'avg_confidence': stats['total_confidence'] / stats['count'],
                    'avg_magnitude': stats['avg_magnitude'],
                    'violation_rate': stats['violations'] / stats['count']
                }
        
        # Recent validation trend
        recent_validations = list(self.validation_history)[-10:]
        recent_success_rate = sum(1 for v in recent_validations if v['is_valid']) / len(recent_validations) if recent_validations else 0.0
        
        return {
            "overall_statistics": {
                "total_validations": self.total_validations,
                "failed_validations": self.failed_validations,
                "success_rate": success_rate,
                "anomalies_detected": self.anomalies_detected
            },
            "direction_statistics": direction_stats,
            "recent_performance": {
                "recent_success_rate": recent_success_rate,
                "trend": "improving" if recent_success_rate > success_rate else "declining" if recent_success_rate < success_rate else "stable"
            },
            "validation_history_length": len(self.validation_history),
            "configuration": {
                "consistency_threshold": self.consistency_threshold,
                "anomaly_threshold": self.anomaly_threshold,
                "validation_window": self.validation_window
            }
        }
        
    def reset(self):
        """Reset validator state"""
        self.validation_history.clear()
        self.anomaly_history.clear()
        self.pattern_violations.clear()
        self.direction_statistics = {
            dt: {'count': 0, 'total_confidence': 0.0, 'avg_magnitude': 0.0, 'violations': 0}
            for dt in DirectionType
        }
        self.total_validations = 0
        self.failed_validations = 0
        self.anomalies_detected = 0
        
    def is_direction_valid(self, direction_state: DirectionState) -> bool:
        """Quick validation check"""
        try:
            result = self.validate_direction_state(direction_state)
            return result.is_valid
        except Exception:
            return False
            
    def get_validation_confidence(self, direction_state: DirectionState) -> float:
        """Get validation confidence without full validation"""
        try:
            result = self.validate_direction_state(direction_state)
            return result.confidence
        except Exception:
            return 0.0