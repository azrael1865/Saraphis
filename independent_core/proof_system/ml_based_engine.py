"""
ML-Based Proof Engine
Generates proofs based on machine learning model predictions and analysis
"""

import logging
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class MLBasedProofEngine:
    """ML-based proof engine for model prediction validation"""
    
    def __init__(self):
        """Initialize ML-based engine"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.prediction_history = []
        
    def generate_ml_proof(self, transaction: Dict[str, Any], model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ML-based proof for transaction prediction"""
        start_time = time.time()
        
        features = transaction.get('features', [])
        if isinstance(features, np.ndarray):
            features = features.tolist()
            
        model_prediction = transaction.get('model_prediction', 0.5)
        model_confidence = transaction.get('model_confidence', 0.5)
        
        # Analyze prediction stability
        stability_analysis = self._analyze_prediction_stability(model_prediction)
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(features, model_state)
        
        # Analyze gradients if available
        gradient_analysis = {}
        if 'gradients' in model_state:
            gradient_analysis = self.analyze_gradients(model_state['gradients'])
            
        # Calculate uncertainty
        uncertainty_analysis = self._quantify_uncertainty(model_prediction, model_confidence)
        
        # Generate overall confidence score
        confidence_score = self._calculate_ml_confidence(
            model_confidence, stability_analysis, uncertainty_analysis
        )
        
        proof_time = time.time() - start_time
        
        proof = {
            'model_prediction': model_prediction,
            'model_confidence': model_confidence,
            'confidence_score': confidence_score,
            'prediction_stability': stability_analysis,
            'feature_importance': feature_importance,
            'gradient_analysis': gradient_analysis,
            'uncertainty_analysis': uncertainty_analysis,
            'generation_time_ms': proof_time * 1000,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store prediction for future stability analysis
        self.prediction_history.append({
            'prediction': model_prediction,
            'confidence': model_confidence,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
            
        return proof
        
    def analyze_gradients(self, gradients: np.ndarray) -> Dict[str, Any]:
        """Analyze gradient properties for proof validation"""
        if gradients is None or len(gradients) == 0:
            return {'error': 'No gradients provided'}
            
        if isinstance(gradients, list):
            gradients = np.array(gradients)
            
        # Calculate gradient statistics
        gradient_norm = np.linalg.norm(gradients)
        gradient_mean = np.mean(gradients)
        gradient_std = np.std(gradients)
        gradient_max = np.max(np.abs(gradients))
        
        # Check for problematic gradients
        has_nan = np.isnan(gradients).any()
        has_inf = np.isinf(gradients).any()
        
        # Stability indicators
        is_exploding = gradient_norm > 100.0
        is_vanishing = gradient_norm < 1e-7
        
        # Convergence indicator
        convergence_indicator = self._assess_gradient_convergence(gradients)
        
        return {
            'gradient_norm': float(gradient_norm),
            'gradient_mean': float(gradient_mean),
            'gradient_std': float(gradient_std),
            'gradient_max': float(gradient_max),
            'has_nan': has_nan,
            'has_inf': has_inf,
            'is_exploding': is_exploding,
            'is_vanishing': is_vanishing,
            'gradient_stability': 'stable' if not (is_exploding or is_vanishing or has_nan or has_inf) else 'unstable',
            'convergence_indicator': convergence_indicator
        }
        
    def _analyze_prediction_stability(self, current_prediction: float) -> Dict[str, Any]:
        """Analyze prediction stability over time using industry-standard metrics"""
        if len(self.prediction_history) < 2:
            return {
                'stability_score': 0.5,
                'variance': 0.0,
                'trend': 'insufficient_data'
            }
            
        recent_predictions = [p['prediction'] for p in self.prediction_history[-10:]]
        recent_predictions.append(current_prediction)
        
        variance = np.var(recent_predictions)
        std_dev = np.std(recent_predictions)
        mean_prediction = np.mean(recent_predictions)
        
        # Industry-standard stability score using Coefficient of Variation (CV)
        # CV = std_dev / mean, lower CV means more stable
        # For predictions near 0, use absolute stability
        if abs(mean_prediction) > 0.1:
            cv = std_dev / abs(mean_prediction)
            # Convert CV to stability score (0-1 scale, higher is more stable)
            # CV < 0.1 is very stable, CV > 0.5 is very unstable
            stability_score = max(0.0, min(1.0, 1.0 - cv * 2))
        else:
            # For values near zero, use absolute variance
            stability_score = max(0.0, 1.0 - std_dev * 10)
        
        # Determine trend using linear regression
        if len(recent_predictions) >= 3:
            trend_slope = np.polyfit(range(len(recent_predictions)), recent_predictions, 1)[0]
            # Use statistical significance for trend detection
            # Slope threshold based on standard error
            slope_threshold = std_dev / np.sqrt(len(recent_predictions))
            if abs(trend_slope) < slope_threshold:
                trend = 'stable'
            elif trend_slope > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
        else:
            trend = 'insufficient_data'
            
        return {
            'stability_score': float(stability_score),
            'variance': float(variance),
            'mean_prediction': float(mean_prediction),
            'trend': trend,
            'sample_size': len(recent_predictions)
        }
        
    def _calculate_feature_importance(self, features: List[float], model_state: Dict[str, Any]) -> Dict[str, float]:
        """Calculate feature importance for the current prediction"""
        if not features:
            return {}
            
        # Simple importance calculation based on feature magnitude and model weights
        importance = {}
        
        for i, feature_value in enumerate(features):
            # Use feature magnitude as proxy for importance
            feature_importance = abs(feature_value)
            
            # Normalize by max feature value to get relative importance
            max_feature = max(abs(f) for f in features) if features else 1.0
            if max_feature > 0:
                feature_importance = feature_importance / max_feature
            else:
                # All features are zero, assign equal importance
                feature_importance = 1.0 / len(features) if features else 0.0
                
            importance[f'feature_{i}'] = feature_importance
            
        # Normalize to sum to 1
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
            
        return importance
        
    def _quantify_uncertainty(self, prediction: float, confidence: float) -> Dict[str, Any]:
        """Quantify prediction uncertainty"""
        # Uncertainty increases near decision boundary (0.5)
        boundary_distance = abs(prediction - 0.5)
        boundary_uncertainty = 1.0 - (boundary_distance * 2)
        
        # Uncertainty from model confidence
        confidence_uncertainty = 1.0 - confidence
        
        # Combined uncertainty
        combined_uncertainty = (boundary_uncertainty + confidence_uncertainty) / 2
        
        # Determine uncertainty reason
        if combined_uncertainty > 0.7:
            uncertainty_reason = 'high_overall_uncertainty'
        elif boundary_distance < 0.1:
            uncertainty_reason = 'near_decision_boundary'
        elif confidence < 0.6:
            uncertainty_reason = 'low_model_confidence'
        else:
            uncertainty_reason = 'acceptable_uncertainty'
            
        return {
            'boundary_uncertainty': float(boundary_uncertainty),
            'confidence_uncertainty': float(confidence_uncertainty),
            'combined_uncertainty': float(combined_uncertainty),
            'uncertainty_reason': uncertainty_reason,
            'prediction_certainty': 1.0 - combined_uncertainty
        }
        
    def _calculate_ml_confidence(self, model_confidence: float, stability: Dict[str, Any], 
                                uncertainty: Dict[str, Any]) -> float:
        """Calculate overall ML confidence score"""
        # Weight different factors
        confidence_weight = 0.4
        stability_weight = 0.3
        certainty_weight = 0.3
        
        stability_score = stability.get('stability_score', 0.5)
        certainty_score = uncertainty.get('prediction_certainty', 0.5)
        
        overall_confidence = (
            model_confidence * confidence_weight +
            stability_score * stability_weight +
            certainty_score * certainty_weight
        )
        
        return min(max(overall_confidence, 0.0), 1.0)
        
    def _assess_gradient_convergence(self, gradients: np.ndarray) -> Dict[str, Any]:
        """Assess gradient convergence properties"""
        # Simple convergence assessment based on gradient magnitude
        gradient_norm = np.linalg.norm(gradients)
        
        if gradient_norm < 1e-6:
            convergence_status = 'converged'
            convergence_rate = 1.0
        elif gradient_norm < 1e-3:
            convergence_status = 'converging'
            convergence_rate = 0.7
        elif gradient_norm < 1e-1:
            convergence_status = 'slow_convergence'
            convergence_rate = 0.4
        else:
            convergence_status = 'not_converging'
            convergence_rate = 0.1
            
        return {
            'status': convergence_status,
            'rate': convergence_rate,
            'gradient_magnitude': float(gradient_norm)
        }
        
    def aggregate_ensemble_predictions(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """Aggregate predictions from ensemble using industry-standard methods"""
        if not predictions:
            return {'error': 'No predictions provided'}
            
        pred_values = list(predictions.values())
        
        ensemble_mean = np.mean(pred_values)
        ensemble_std = np.std(pred_values)
        ensemble_median = np.median(pred_values)
        
        # Industry-standard agreement score using Inter-Rater Reliability
        # For continuous values, use Intraclass Correlation Coefficient approximation
        # Simplified: agreement based on coefficient of variation
        if abs(ensemble_mean) > 0.01:
            cv = ensemble_std / abs(ensemble_mean)
            # Low CV = high agreement
            agreement_score = max(0.0, min(1.0, 1.0 - cv))
        else:
            # For values near zero, use absolute standard deviation
            agreement_score = max(0.0, 1.0 - ensemble_std * 10)
        
        # Industry-standard outlier detection using Modified Z-Score (MAD method)
        outliers = []
        
        if len(pred_values) >= 3:
            # Use Median Absolute Deviation (MAD) for robust outlier detection
            median = np.median(pred_values)
            mad = np.median(np.abs(pred_values - median))
            
            # Modified Z-score threshold (typically 3.5 for outliers)
            # More sensitive threshold of 2.5 for ML predictions
            threshold = 2.5
            
            if mad > 0:
                for model, pred in predictions.items():
                    modified_z_score = 0.6745 * (pred - median) / mad
                    if abs(modified_z_score) > threshold:
                        outliers.append(model)
            else:
                # If MAD is 0, use IQR method as fallback
                q1 = np.percentile(pred_values, 25)
                q3 = np.percentile(pred_values, 75)
                iqr = q3 - q1
                
                if iqr > 0:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    for model, pred in predictions.items():
                        if pred < lower_bound or pred > upper_bound:
                            outliers.append(model)
                else:
                    # All values are identical, check for any different values
                    for model, pred in predictions.items():
                        if abs(pred - median) > 1e-10:
                            outliers.append(model)
        else:
            # For very small ensembles, use simple threshold
            # Any prediction more than 2 std away is an outlier
            if ensemble_std > 0:
                for model, pred in predictions.items():
                    z_score = abs(pred - ensemble_mean) / ensemble_std
                    if z_score > 2.0:
                        outliers.append(model)
                
        return {
            'ensemble_mean': float(ensemble_mean),
            'ensemble_std': float(ensemble_std),
            'ensemble_median': float(ensemble_median),
            'agreement_score': float(agreement_score),
            'outlier_models': outliers,
            'model_count': len(predictions)
        }
        
    def check_temporal_consistency(self, current: float, historical: List[float]) -> Dict[str, Any]:
        """Check temporal consistency using Statistical Process Control (SPC) methods"""
        if not historical:
            return {
                'consistency_score': 0.5,
                'anomaly_detected': False,
                'message': 'No historical data'
            }
            
        hist_mean = np.mean(historical)
        hist_std = np.std(historical)
        
        # Industry-standard SPC approach using control limits
        # Upper Control Limit (UCL) and Lower Control Limit (LCL)
        # Using 3-sigma limits for anomaly detection (99.7% confidence)
        
        if hist_std > 0:
            # Calculate z-score
            z_score = (current - hist_mean) / hist_std
            
            # Anomaly detection using 3-sigma rule
            anomaly_detected = abs(z_score) > 3.0
            
            # Consistency score using exponential decay function
            # Score decreases exponentially as z-score increases
            # Score = exp(-|z|^2 / 2), similar to Gaussian distribution
            consistency_score = np.exp(-z_score**2 / 2)
            
            # Alternative linear scoring for interpretability
            # Within 1 sigma: high consistency (>0.68)
            # Within 2 sigma: moderate consistency (>0.32)
            # Within 3 sigma: low consistency (>0.05)
            # Beyond 3 sigma: very low consistency
            if abs(z_score) <= 1.0:
                consistency_score = 0.9 - 0.2 * abs(z_score)
            elif abs(z_score) <= 2.0:
                consistency_score = 0.7 - 0.15 * (abs(z_score) - 1.0)
            elif abs(z_score) <= 3.0:
                consistency_score = 0.55 - 0.15 * (abs(z_score) - 2.0)
            else:
                consistency_score = max(0.0, 0.4 - 0.1 * (abs(z_score) - 3.0))
                
            z_score = float(abs(z_score))
        else:
            # No variation in historical data
            if abs(current - hist_mean) < 1e-10:
                consistency_score = 1.0
                anomaly_detected = False
                z_score = 0.0
            else:
                # Any deviation from constant historical value is anomalous
                consistency_score = 0.0
                anomaly_detected = True
                z_score = float('inf')
        
        return {
            'consistency_score': float(consistency_score),
            'anomaly_detected': anomaly_detected,
            'z_score': z_score,
            'historical_mean': float(hist_mean),
            'historical_std': float(hist_std)
        }
        
    def generate_proof(self, transaction: Dict[str, Any], model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive ML-based proof"""
        ml_proof = self.generate_ml_proof(transaction, model_state)
        
        return {
            'engine_type': 'ml_based',
            'transaction_id': transaction.get('transaction_id', 'unknown'),
            'ml_analysis': ml_proof,
            'proof_metadata': {
                'engine_version': '1.0.0',
                'generation_timestamp': datetime.now().isoformat(),
                'model_iteration': model_state.get('iteration', 0)
            }
        }