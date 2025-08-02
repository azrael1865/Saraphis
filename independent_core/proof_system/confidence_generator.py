"""
Confidence Generator
Generates confidence scores by aggregating different proof sources
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class ConfidenceGenerator:
    """Generates confidence scores from multiple proof sources"""
    
    def __init__(self):
        """Initialize confidence generator"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.default_weights = {
            'rule_based': 0.3,
            'ml_based': 0.5,
            'cryptographic': 0.2
        }
        
    def generate_confidence(self, rule_score: Optional[float] = None, 
                          ml_probability: Optional[float] = None,
                          crypto_valid: Optional[bool] = None,
                          weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Generate confidence score from multiple sources"""
        start_time = time.time()
        
        # Use provided weights or defaults
        if weights is None:
            weights = self.default_weights.copy()
            
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            weights = self.default_weights.copy()
            
        # Track missing components
        missing_components = []
        available_components = {}
        
        # Process rule-based score
        if rule_score is not None:
            available_components['rule_based'] = min(max(rule_score, 0.0), 1.0)
        else:
            missing_components.append('rule_based')
            
        # Process ML probability
        if ml_probability is not None:
            available_components['ml_based'] = min(max(ml_probability, 0.0), 1.0)
        else:
            missing_components.append('ml_based')
            
        # Process cryptographic validation
        if crypto_valid is not None:
            available_components['cryptographic'] = 1.0 if crypto_valid else 0.0
        else:
            missing_components.append('cryptographic')
            
        # Calculate weighted confidence score
        confidence_score = 0.0
        total_available_weight = 0.0
        
        for component, score in available_components.items():
            weight = weights.get(component, 0.0)
            confidence_score += score * weight
            total_available_weight += weight
            
        # Adjust for missing components
        if total_available_weight > 0 and total_available_weight < 1.0:
            # Redistribute weights among available components
            confidence_score = confidence_score / total_available_weight
            
        # Apply penalty for invalid cryptographic proof
        crypto_penalty = 0.0
        if crypto_valid is False:
            crypto_penalty = 0.5  # Significant penalty for invalid crypto proof
            confidence_score = max(0.0, confidence_score - crypto_penalty)
            
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            confidence_score, available_components, missing_components
        )
        
        generation_time = time.time() - start_time
        
        result = {
            'score': min(max(confidence_score, 0.0), 1.0),
            'confidence_interval': confidence_interval,
            'components': {
                'rule_based': available_components.get('rule_based'),
                'ml_based': available_components.get('ml_based'),
                'cryptographic': available_components.get('cryptographic'),
                'crypto_penalty': crypto_penalty if crypto_penalty > 0 else None
            },
            'weights_used': weights,
            'missing_components': missing_components,
            'generation_time_ms': generation_time * 1000,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
        
    def _calculate_confidence_interval(self, score: float, available: Dict[str, float], 
                                     missing: List[str]) -> Tuple[float, float]:
        """Calculate confidence interval for the score"""
        # Base interval width depends on number of available components
        base_width = 0.1  # 10% base uncertainty
        
        # Increase uncertainty for missing components
        missing_penalty = len(missing) * 0.05
        
        # Increase uncertainty for scores near decision boundary (0.5)
        boundary_penalty = max(0, 0.1 - abs(score - 0.5) * 0.2)
        
        # Calculate total uncertainty
        total_uncertainty = base_width + missing_penalty + boundary_penalty
        
        # Calculate interval bounds
        lower_bound = max(0.0, score - total_uncertainty)
        upper_bound = min(1.0, score + total_uncertainty)
        
        return (lower_bound, upper_bound)
        
    def generate_temporal_confidence(self, current_scores: Dict[str, float],
                                   historical_scores: List[Dict[str, float]]) -> Dict[str, Any]:
        """Generate confidence considering temporal evolution"""
        if not historical_scores:
            return self.generate_confidence(**current_scores)
            
        # Calculate stability metrics
        stability_metrics = self._calculate_temporal_stability(historical_scores)
        
        # Generate current confidence
        current_confidence = self.generate_confidence(**current_scores)
        
        # Adjust confidence based on temporal stability
        stability_factor = stability_metrics['stability_score']
        adjusted_score = current_confidence['score'] * (0.7 + 0.3 * stability_factor)
        
        return {
            'score': min(max(adjusted_score, 0.0), 1.0),
            'base_confidence': current_confidence,
            'temporal_stability': stability_metrics,
            'adjustment_factor': stability_factor,
            'timestamp': datetime.now().isoformat()
        }
        
    def _calculate_temporal_stability(self, historical_scores: List[Dict[str, float]]) -> Dict[str, Any]:
        """Calculate temporal stability metrics"""
        if len(historical_scores) < 2:
            return {
                'stability_score': 0.5,
                'trend': 'insufficient_data',
                'variance': 0.0
            }
            
        # Extract scores over time
        scores_over_time = [
            self.generate_confidence(**scores)['score'] 
            for scores in historical_scores
        ]
        
        # Calculate variance (lower variance = higher stability)
        variance = np.var(scores_over_time)
        stability_score = max(0.0, 1.0 - variance * 5)  # Scale variance to stability
        
        # Calculate trend
        if len(scores_over_time) >= 3:
            trend_slope = np.polyfit(range(len(scores_over_time)), scores_over_time, 1)[0]
            if abs(trend_slope) < 0.01:
                trend = 'stable'
            elif trend_slope > 0:
                trend = 'improving'
            else:
                trend = 'declining'
        else:
            trend = 'insufficient_data'
            
        return {
            'stability_score': float(stability_score),
            'variance': float(variance),
            'trend': trend,
            'sample_size': len(scores_over_time),
            'mean_score': float(np.mean(scores_over_time)),
            'std_score': float(np.std(scores_over_time))
        }
        
    def aggregate_ensemble_confidence(self, individual_confidences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate confidence from ensemble of sources"""
        if not individual_confidences:
            return {
                'error': 'No confidence scores provided',
                'score': 0.0
            }
            
        # Extract scores
        scores = [conf['score'] for conf in individual_confidences if 'score' in conf]
        
        if not scores:
            return {
                'error': 'No valid confidence scores found',
                'score': 0.0
            }
            
        # Calculate ensemble statistics
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        # Calculate agreement level
        agreement_score = max(0.0, 1.0 - std_score * 2)
        
        # Choose aggregation method based on agreement
        if agreement_score > 0.8:
            # High agreement - use mean
            final_score = mean_score
            aggregation_method = 'mean'
        elif agreement_score > 0.5:
            # Medium agreement - use median
            final_score = median_score
            aggregation_method = 'median'
        else:
            # Low agreement - use conservative estimate
            final_score = min_score
            aggregation_method = 'conservative'
            
        return {
            'score': float(final_score),
            'ensemble_statistics': {
                'mean': float(mean_score),
                'median': float(median_score),
                'std': float(std_score),
                'min': float(min_score),
                'max': float(max_score),
                'count': len(scores)
            },
            'agreement_score': float(agreement_score),
            'aggregation_method': aggregation_method,
            'timestamp': datetime.now().isoformat()
        }
        
    def calibrate_confidence(self, predicted_confidence: float, 
                           actual_outcome: bool, 
                           historical_calibration: Optional[List[Tuple[float, bool]]] = None) -> Dict[str, Any]:
        """Calibrate confidence based on historical accuracy"""
        # Add current observation to calibration data
        calibration_data = historical_calibration or []
        calibration_data.append((predicted_confidence, actual_outcome))
        
        if len(calibration_data) < 10:
            return {
                'calibrated_confidence': predicted_confidence,
                'calibration_quality': 'insufficient_data',
                'reliability_score': 0.5
            }
            
        # Group predictions into bins
        bins = np.linspace(0, 1, 11)  # 10 bins
        bin_indices = np.digitize([conf for conf, _ in calibration_data], bins) - 1
        
        # Calculate calibration metrics for each bin
        bin_metrics = {}
        for i in range(len(bins) - 1):
            bin_data = [(conf, outcome) for j, (conf, outcome) in enumerate(calibration_data) 
                       if bin_indices[j] == i]
            
            if bin_data:
                bin_confidence = np.mean([conf for conf, _ in bin_data])
                bin_accuracy = np.mean([outcome for _, outcome in bin_data])
                bin_metrics[i] = {
                    'confidence': bin_confidence,
                    'accuracy': bin_accuracy,
                    'count': len(bin_data)
                }
                
        # Find appropriate bin for current prediction
        current_bin = min(max(int(predicted_confidence * 10), 0), 9)
        
        if current_bin in bin_metrics and bin_metrics[current_bin]['count'] >= 3:
            # Use calibrated confidence from historical data
            calibrated_confidence = bin_metrics[current_bin]['accuracy']
            calibration_quality = 'calibrated'
        else:
            # Not enough data for calibration
            calibrated_confidence = predicted_confidence
            calibration_quality = 'uncalibrated'
            
        # Calculate overall reliability score
        reliability_scores = []
        for metrics in bin_metrics.values():
            if metrics['count'] >= 3:
                # Reliability is inverse of calibration error
                calibration_error = abs(metrics['confidence'] - metrics['accuracy'])
                reliability = 1.0 - calibration_error
                reliability_scores.append(reliability)
                
        overall_reliability = np.mean(reliability_scores) if reliability_scores else 0.5
        
        return {
            'calibrated_confidence': float(calibrated_confidence),
            'original_confidence': float(predicted_confidence),
            'calibration_quality': calibration_quality,
            'reliability_score': float(overall_reliability),
            'bin_metrics': bin_metrics,
            'calibration_data_size': len(calibration_data)
        }
        
    def generate_explanation(self, confidence_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate human-readable explanation of confidence score"""
        score = confidence_result['score']
        components = confidence_result.get('components', {})
        missing = confidence_result.get('missing_components', [])
        
        # Determine confidence level
        if score >= 0.9:
            level = 'very_high'
            description = 'Very high confidence in the assessment'
        elif score >= 0.7:
            level = 'high'
            description = 'High confidence in the assessment'
        elif score >= 0.5:
            level = 'medium'
            description = 'Medium confidence in the assessment'
        elif score >= 0.3:
            level = 'low'
            description = 'Low confidence in the assessment'
        else:
            level = 'very_low'
            description = 'Very low confidence in the assessment'
            
        # Generate component explanations
        component_explanations = []
        
        if components.get('rule_based') is not None:
            rule_score = components['rule_based']
            if rule_score > 0.7:
                component_explanations.append('Business rules strongly indicate fraud risk')
            elif rule_score > 0.3:
                component_explanations.append('Business rules show moderate fraud indicators')
            else:
                component_explanations.append('Business rules show low fraud risk')
                
        if components.get('ml_based') is not None:
            ml_score = components['ml_based']
            if ml_score > 0.7:
                component_explanations.append('Machine learning model predicts high fraud probability')
            elif ml_score > 0.3:
                component_explanations.append('Machine learning model shows moderate fraud probability')
            else:
                component_explanations.append('Machine learning model predicts low fraud probability')
                
        if components.get('cryptographic') is not None:
            crypto_valid = components['cryptographic']
            if crypto_valid:
                component_explanations.append('Data integrity verified cryptographically')
            else:
                component_explanations.append('Data integrity could not be verified')
                
        # Generate warnings
        warnings = []
        if missing:
            warnings.append(f"Missing assessment from: {', '.join(missing)}")
            
        if components.get('crypto_penalty'):
            warnings.append('Confidence reduced due to failed cryptographic validation')
            
        return {
            'confidence_level': level,
            'description': description,
            'component_explanations': component_explanations,
            'warnings': warnings,
            'score': score,
            'interpretation': self._interpret_score(score)
        }
        
    def _interpret_score(self, score: float) -> str:
        """Interpret confidence score for business users"""
        if score >= 0.9:
            return 'Extremely reliable assessment - proceed with high confidence'
        elif score >= 0.7:
            return 'Reliable assessment - minimal additional verification needed'
        elif score >= 0.5:
            return 'Moderately reliable - consider additional verification'
        elif score >= 0.3:
            return 'Lower reliability - additional verification recommended'
        else:
            return 'Low reliability - manual review strongly recommended'