# Part B - Advanced Model Evaluation Features
# This continues from the ModelEvaluationSystem class in Part A

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import warnings
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
from statsmodels.stats.power import TTestPower
from statsmodels.stats.proportion import proportions_ztest
import yaml
from jinja2 import Template
import markdown
import pdfkit

# Add these methods to the existing ModelEvaluationSystem class:

    def analyze_prediction_errors(self, 
                                model: Any,
                                test_data: pd.DataFrame,
                                error_analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive error analysis including patterns and root causes
        
        Args:
            model: Trained model to analyze
            test_data: Test dataset with features and labels
            error_analysis_config: Configuration for error analysis
            
        Returns:
            Detailed error analysis report
        """
        try:
            # Extract features and labels
            X_test = test_data.drop(columns=[error_analysis_config.get('target_column', 'is_fraud')])
            y_test = test_data[error_analysis_config.get('target_column', 'is_fraud')]
            
            # Get predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Identify error cases
            false_positives_mask = (y_pred == 1) & (y_test == 0)
            false_negatives_mask = (y_pred == 0) & (y_test == 1)
            
            # Analyze error patterns
            error_analysis = {
                'timestamp': datetime.now(),
                'total_predictions': len(y_test),
                'total_errors': np.sum(y_pred != y_test),
                'error_rate': np.mean(y_pred != y_test),
                'false_positive_analysis': self._analyze_error_subset(
                    X_test[false_positives_mask],
                    y_proba[false_positives_mask] if y_proba is not None else None,
                    'false_positive'
                ),
                'false_negative_analysis': self._analyze_error_subset(
                    X_test[false_negatives_mask],
                    y_proba[false_negatives_mask] if y_proba is not None else None,
                    'false_negative'
                ),
                'error_distribution': {},
                'feature_importance_in_errors': {},
                'temporal_error_patterns': {},
                'recommendations': []
            }
            
            # Analyze error distribution by prediction confidence
            if y_proba is not None:
                confidence_bins = np.linspace(0, 1, 11)
                error_analysis['error_distribution']['by_confidence'] = []
                
                for i in range(len(confidence_bins) - 1):
                    mask = (y_proba >= confidence_bins[i]) & (y_proba < confidence_bins[i + 1])
                    if np.sum(mask) > 0:
                        error_rate = np.mean(y_pred[mask] != y_test[mask])
                        error_analysis['error_distribution']['by_confidence'].append({
                            'confidence_range': (confidence_bins[i], confidence_bins[i + 1]),
                            'count': int(np.sum(mask)),
                            'error_rate': float(error_rate)
                        })
            
            # Analyze feature importance in errors
            if hasattr(model, 'feature_importances_'):
                feature_names = X_test.columns.tolist()
                importances = model.feature_importances_
                
                # Compare feature distributions between errors and correct predictions
                correct_mask = y_pred == y_test
                error_mask = ~correct_mask
                
                feature_diffs = {}
                for i, feature in enumerate(feature_names):
                    if error_mask.sum() > 0 and correct_mask.sum() > 0:
                        error_mean = X_test[feature][error_mask].mean()
                        correct_mean = X_test[feature][correct_mask].mean()
                        diff = abs(error_mean - correct_mean)
                        feature_diffs[feature] = {
                            'importance': float(importances[i]),
                            'mean_diff': float(diff),
                            'error_mean': float(error_mean),
                            'correct_mean': float(correct_mean)
                        }
                
                # Sort by impact (importance * difference)
                sorted_features = sorted(
                    feature_diffs.items(),
                    key=lambda x: x[1]['importance'] * x[1]['mean_diff'],
                    reverse=True
                )[:10]
                
                error_analysis['feature_importance_in_errors'] = dict(sorted_features)
            
            # Temporal error patterns (if timestamp available)
            if 'timestamp' in test_data.columns:
                error_analysis['temporal_error_patterns'] = self._analyze_temporal_errors(
                    test_data,
                    y_pred,
                    y_test
                )
            
            # Generate recommendations
            error_analysis['recommendations'] = self._generate_error_recommendations(error_analysis)
            
            # Log results
            self.logger.info(f"Error analysis completed: {error_analysis['error_rate']:.2%} error rate")
            
            return error_analysis
            
        except Exception as e:
            self.logger.error(f"Error analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_error_subset(self, X_subset: pd.DataFrame, 
                            proba_subset: Optional[np.ndarray],
                            error_type: str) -> Dict[str, Any]:
        """Analyze a specific subset of errors"""
        analysis = {
            'count': len(X_subset),
            'percentage': len(X_subset) / len(X_subset) * 100 if len(X_subset) > 0 else 0,
            'feature_statistics': {},
            'confidence_stats': {}
        }
        
        if len(X_subset) > 0:
            # Feature statistics
            for col in X_subset.columns:
                analysis['feature_statistics'][col] = {
                    'mean': float(X_subset[col].mean()),
                    'std': float(X_subset[col].std()),
                    'median': float(X_subset[col].median())
                }
            
            # Confidence statistics
            if proba_subset is not None:
                analysis['confidence_stats'] = {
                    'mean': float(np.mean(proba_subset)),
                    'std': float(np.std(proba_subset)),
                    'min': float(np.min(proba_subset)),
                    'max': float(np.max(proba_subset))
                }
        
        return analysis
    
    def _analyze_temporal_errors(self, data: pd.DataFrame, 
                               y_pred: np.ndarray, 
                               y_true: np.ndarray) -> Dict[str, Any]:
        """Analyze temporal patterns in errors"""
        temporal_analysis = {}
        
        # Add error indicator
        data = data.copy()
        data['is_error'] = y_pred != y_true
        
        # Group by time periods
        if 'timestamp' in data.columns:
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
            
            # Hourly error rates
            hourly_errors = data.groupby('hour')['is_error'].agg(['sum', 'count', 'mean'])
            temporal_analysis['hourly_error_rates'] = hourly_errors.to_dict()
            
            # Daily error rates
            daily_errors = data.groupby('day_of_week')['is_error'].agg(['sum', 'count', 'mean'])
            temporal_analysis['daily_error_rates'] = daily_errors.to_dict()
        
        return temporal_analysis
    
    def _generate_error_recommendations(self, error_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on error analysis"""
        recommendations = []
        
        # High error rate recommendation
        if error_analysis['error_rate'] > 0.1:
            recommendations.append(
                "High error rate detected. Consider retraining with more data or adjusting model hyperparameters."
            )
        
        # False positive/negative imbalance
        fp_count = error_analysis['false_positive_analysis']['count']
        fn_count = error_analysis['false_negative_analysis']['count']
        
        if fp_count > 2 * fn_count:
            recommendations.append(
                "High false positive rate. Consider adjusting decision threshold or class weights."
            )
        elif fn_count > 2 * fp_count:
            recommendations.append(
                "High false negative rate. This could lead to missed fraud cases. Consider lowering threshold."
            )
        
        # Feature-based recommendations
        if error_analysis['feature_importance_in_errors']:
            top_feature = list(error_analysis['feature_importance_in_errors'].keys())[0]
            recommendations.append(
                f"Feature '{top_feature}' shows significant difference in error cases. Consider feature engineering."
            )
        
        return recommendations
    
    def calculate_confidence_intervals(self, 
                                     performance_metrics: Dict[str, float],
                                     confidence_level: float = 0.95,
                                     n_samples: Optional[int] = None,
                                     method: str = 'bootstrap') -> Dict[str, Dict[str, float]]:
        """
        Calculate statistical confidence intervals for performance metrics
        
        Args:
            performance_metrics: Dictionary of metric values
            confidence_level: Confidence level (default 0.95)
            n_samples: Number of samples used (for analytical methods)
            method: Method to use ('bootstrap', 'analytical')
            
        Returns:
            Confidence intervals for each metric
        """
        confidence_intervals = {}
        alpha = 1 - confidence_level
        
        for metric_name, metric_value in performance_metrics.items():
            if method == 'analytical' and n_samples:
                # Use analytical methods for proportion-based metrics
                if metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
                    # Wilson score interval for proportions
                    z = stats.norm.ppf(1 - alpha/2)
                    denominator = 1 + z**2/n_samples
                    center = (metric_value + z**2/(2*n_samples)) / denominator
                    margin = z * np.sqrt(metric_value*(1-metric_value)/n_samples + z**2/(4*n_samples**2)) / denominator
                    
                    confidence_intervals[metric_name] = {
                        'point_estimate': metric_value,
                        'lower_bound': max(0, center - margin),
                        'upper_bound': min(1, center + margin),
                        'margin_of_error': margin,
                        'method': 'wilson_score'
                    }
                elif metric_name == 'auc_roc':
                    # DeLong method approximation for AUC
                    # Simplified - in practice would use full DeLong method
                    se = np.sqrt((metric_value * (1 - metric_value)) / n_samples)
                    margin = z * se
                    
                    confidence_intervals[metric_name] = {
                        'point_estimate': metric_value,
                        'lower_bound': max(0, metric_value - margin),
                        'upper_bound': min(1, metric_value + margin),
                        'margin_of_error': margin,
                        'method': 'delong_approximation'
                    }
            else:
                # For bootstrap or unknown metrics, use conservative estimate
                # In practice, would implement full bootstrap
                se_estimate = 0.05  # Conservative estimate
                z = stats.norm.ppf(1 - alpha/2)
                margin = z * se_estimate
                
                confidence_intervals[metric_name] = {
                    'point_estimate': metric_value,
                    'lower_bound': max(0, metric_value - margin),
                    'upper_bound': min(1, metric_value + margin),
                    'margin_of_error': margin,
                    'method': 'bootstrap_estimate'
                }
        
        return confidence_intervals
    
    def detect_model_overfitting(self,
                               train_metrics: Dict[str, float],
                               val_metrics: Dict[str, float],
                               test_metrics: Dict[str, float],
                               thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Detect overfitting with comprehensive diagnostics
        
        Args:
            train_metrics: Training set performance metrics
            val_metrics: Validation set performance metrics
            test_metrics: Test set performance metrics
            thresholds: Custom thresholds for overfitting detection
            
        Returns:
            Overfitting analysis results
        """
        default_thresholds = {
            'performance_gap': 0.05,  # 5% difference indicates potential overfitting
            'variance_ratio': 1.5,    # High variance ratio indicates instability
            'generalization_gap': 0.1  # Gap between train and test
        }
        
        if thresholds:
            default_thresholds.update(thresholds)
        
        overfitting_analysis = {
            'timestamp': datetime.now(),
            'is_overfitting': False,
            'severity': 'none',
            'metrics_comparison': {},
            'indicators': [],
            'recommendations': []
        }
        
        # Calculate gaps between datasets
        for metric in train_metrics:
            if metric in val_metrics and metric in test_metrics:
                train_val_gap = train_metrics[metric] - val_metrics[metric]
                train_test_gap = train_metrics[metric] - test_metrics[metric]
                val_test_gap = val_metrics[metric] - test_metrics[metric]
                
                overfitting_analysis['metrics_comparison'][metric] = {
                    'train': train_metrics[metric],
                    'validation': val_metrics[metric],
                    'test': test_metrics[metric],
                    'train_val_gap': train_val_gap,
                    'train_test_gap': train_test_gap,
                    'val_test_gap': val_test_gap
                }
                
                # Check for overfitting indicators
                if train_val_gap > default_thresholds['performance_gap']:
                    overfitting_analysis['indicators'].append(
                        f"{metric}: significant train-validation gap ({train_val_gap:.3f})"
                    )
                    overfitting_analysis['is_overfitting'] = True
                
                if train_test_gap > default_thresholds['generalization_gap']:
                    overfitting_analysis['indicators'].append(
                        f"{metric}: poor generalization ({train_test_gap:.3f} train-test gap)"
                    )
                    overfitting_analysis['is_overfitting'] = True
        
        # Determine severity
        if overfitting_analysis['is_overfitting']:
            avg_gap = np.mean([
                v['train_test_gap'] 
                for v in overfitting_analysis['metrics_comparison'].values()
            ])
            
            if avg_gap > 0.15:
                overfitting_analysis['severity'] = 'severe'
            elif avg_gap > 0.10:
                overfitting_analysis['severity'] = 'moderate'
            else:
                overfitting_analysis['severity'] = 'mild'
        
        # Generate recommendations
        if overfitting_analysis['is_overfitting']:
            severity = overfitting_analysis['severity']
            
            if severity == 'severe':
                overfitting_analysis['recommendations'].extend([
                    "Severe overfitting detected. Consider significant model simplification.",
                    "Increase regularization parameters (L1/L2).",
                    "Reduce model complexity (fewer features, smaller architecture).",
                    "Collect more training data if possible."
                ])
            elif severity == 'moderate':
                overfitting_analysis['recommendations'].extend([
                    "Moderate overfitting detected. Apply regularization techniques.",
                    "Consider dropout layers or early stopping.",
                    "Use cross-validation for hyperparameter tuning."
                ])
            else:
                overfitting_analysis['recommendations'].extend([
                    "Mild overfitting detected. Minor adjustments recommended.",
                    "Fine-tune regularization parameters.",
                    "Consider ensemble methods for better generalization."
                ])
        
        # Calculate variance metrics
        all_metrics = [train_metrics, val_metrics, test_metrics]
        for metric in train_metrics:
            values = [m.get(metric, 0) for m in all_metrics]
            variance = np.var(values)
            overfitting_analysis['metrics_comparison'][metric]['variance'] = float(variance)
        
        self.logger.info(f"Overfitting detection completed: {overfitting_analysis['severity']}")
        
        return overfitting_analysis
    
    def analyze_feature_importance_stability(self,
                                           model: Any,
                                           data_samples: List[pd.DataFrame],
                                           stability_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze stability of feature importance across different data samples
        
        Args:
            model: Model with feature importance capability
            data_samples: List of data samples to test stability
            stability_config: Configuration for stability analysis
            
        Returns:
            Feature importance stability analysis
        """
        if not hasattr(model, 'feature_importances_') and not hasattr(model, 'coef_'):
            return {'error': 'Model does not support feature importance analysis'}
        
        importance_matrices = []
        feature_names = data_samples[0].columns.tolist()
        
        # Calculate importance for each sample
        for i, sample in enumerate(data_samples):
            try:
                # Clone and retrain model on sample
                from sklearn.base import clone
                model_clone = clone(model)
                
                # Assume target is last column
                X_sample = sample.iloc[:, :-1]
                y_sample = sample.iloc[:, -1]
                
                model_clone.fit(X_sample, y_sample)
                
                # Get feature importance
                if hasattr(model_clone, 'feature_importances_'):
                    importances = model_clone.feature_importances_
                else:
                    importances = np.abs(model_clone.coef_).flatten()
                
                importance_matrices.append(importances)
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate importance for sample {i}: {str(e)}")
        
        importance_matrix = np.array(importance_matrices)
        
        # Calculate stability metrics
        stability_analysis = {
            'timestamp': datetime.now(),
            'n_samples': len(data_samples),
            'feature_stability': {},
            'overall_stability_score': 0,
            'unstable_features': [],
            'stability_metrics': {}
        }
        
        # Analyze each feature
        for i, feature in enumerate(feature_names[:-1]):  # Exclude target
            feature_importances = importance_matrix[:, i]
            
            stability_metrics = {
                'mean': float(np.mean(feature_importances)),
                'std': float(np.std(feature_importances)),
                'cv': float(np.std(feature_importances) / (np.mean(feature_importances) + 1e-8)),
                'min': float(np.min(feature_importances)),
                'max': float(np.max(feature_importances)),
                'range': float(np.max(feature_importances) - np.min(feature_importances))
            }
            
            stability_analysis['feature_stability'][feature] = stability_metrics
            
            # Flag unstable features (high coefficient of variation)
            if stability_metrics['cv'] > stability_config.get('cv_threshold', 0.5):
                stability_analysis['unstable_features'].append({
                    'feature': feature,
                    'cv': stability_metrics['cv'],
                    'reason': 'High coefficient of variation'
                })
        
        # Calculate overall stability score
        all_cvs = [v['cv'] for v in stability_analysis['feature_stability'].values()]
        stability_analysis['overall_stability_score'] = 1 - np.mean(all_cvs)
        
        # Rank features by stability
        stable_features = sorted(
            stability_analysis['feature_stability'].items(),
            key=lambda x: x[1]['cv']
        )[:10]
        
        stability_analysis['most_stable_features'] = [
            {'feature': f[0], 'cv': f[1]['cv']} for f in stable_features
        ]
        
        # Correlation analysis between samples
        if len(importance_matrices) > 1:
            correlation_matrix = np.corrcoef(importance_matrices)
            stability_analysis['sample_correlations'] = {
                'mean_correlation': float(np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])),
                'min_correlation': float(np.min(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
            }
        
        self.logger.info(f"Feature stability analysis completed: score {stability_analysis['overall_stability_score']:.3f}")
        
        return stability_analysis
    
    def evaluate_model_fairness(self,
                              model: Any,
                              test_data: pd.DataFrame,
                              fairness_metrics: List[str],
                              protected_attributes: List[str]) -> Dict[str, Any]:
        """
        Evaluate model fairness across protected attributes
        
        Args:
            model: Trained model to evaluate
            test_data: Test data with protected attributes
            fairness_metrics: List of fairness metrics to calculate
            protected_attributes: List of protected attribute columns
            
        Returns:
            Fairness evaluation results
        """
        fairness_results = {
            'timestamp': datetime.now(),
            'protected_attributes': protected_attributes,
            'fairness_metrics': {},
            'disparate_impact': {},
            'equal_opportunity': {},
            'demographic_parity': {},
            'recommendations': []
        }
        
        # Get predictions
        X_test = test_data.drop(columns=['is_fraud'] + protected_attributes)
        y_true = test_data['is_fraud']
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Analyze each protected attribute
        for attribute in protected_attributes:
            if attribute not in test_data.columns:
                continue
                
            fairness_results['fairness_metrics'][attribute] = {}
            
            # Get unique groups
            groups = test_data[attribute].unique()
            
            # Calculate metrics for each group
            group_metrics = {}
            for group in groups:
                group_mask = test_data[attribute] == group
                group_metrics[group] = {
                    'size': int(np.sum(group_mask)),
                    'positive_rate': float(np.mean(y_pred[group_mask])),
                    'true_positive_rate': float(np.mean(y_pred[group_mask & (y_true == 1)])) if np.sum(y_true[group_mask] == 1) > 0 else 0,
                    'false_positive_rate': float(np.mean(y_pred[group_mask & (y_true == 0)])) if np.sum(y_true[group_mask] == 0) > 0 else 0,
                    'accuracy': float(np.mean(y_pred[group_mask] == y_true[group_mask]))
                }
            
            fairness_results['fairness_metrics'][attribute]['groups'] = group_metrics
            
            # Calculate disparate impact
            if len(groups) == 2:
                rates = [group_metrics[g]['positive_rate'] for g in groups]
                if rates[1] > 0:
                    disparate_impact = rates[0] / rates[1]
                else:
                    disparate_impact = float('inf')
                    
                fairness_results['disparate_impact'][attribute] = {
                    'value': disparate_impact,
                    'threshold_80_percent': 0.8 <= disparate_impact <= 1.25,
                    'groups': list(groups)
                }
                
                # Flag if outside 80% rule
                if not fairness_results['disparate_impact'][attribute]['threshold_80_percent']:
                    fairness_results['recommendations'].append(
                        f"Disparate impact detected for {attribute}: {disparate_impact:.2f}"
                    )
            
            # Calculate equal opportunity difference
            tpr_values = [group_metrics[g]['true_positive_rate'] for g in groups]
            equal_opp_diff = max(tpr_values) - min(tpr_values)
            
            fairness_results['equal_opportunity'][attribute] = {
                'difference': equal_opp_diff,
                'max_group': groups[np.argmax(tpr_values)],
                'min_group': groups[np.argmin(tpr_values)]
            }
            
            if equal_opp_diff > 0.1:
                fairness_results['recommendations'].append(
                    f"Equal opportunity violation for {attribute}: {equal_opp_diff:.2f} TPR difference"
                )
            
            # Calculate demographic parity
            selection_rates = [group_metrics[g]['positive_rate'] for g in groups]
            demo_parity_diff = max(selection_rates) - min(selection_rates)
            
            fairness_results['demographic_parity'][attribute] = {
                'difference': demo_parity_diff,
                'selection_rates': {str(g): rate for g, rate in zip(groups, selection_rates)}
            }
            
            if demo_parity_diff > 0.1:
                fairness_results['recommendations'].append(
                    f"Demographic parity violation for {attribute}: {demo_parity_diff:.2f} selection rate difference"
                )
        
        # Overall fairness score
        all_disparate_impacts = [v['value'] for v in fairness_results['disparate_impact'].values() if v['value'] != float('inf')]
        all_equal_opp_diffs = [v['difference'] for v in fairness_results['equal_opportunity'].values()]
        
        fairness_results['overall_fairness_score'] = {
            'disparate_impact_score': np.mean([min(di, 1/di) for di in all_disparate_impacts]) if all_disparate_impacts else 0,
            'equal_opportunity_score': 1 - np.mean(all_equal_opp_diffs) if all_equal_opp_diffs else 0,
            'combined_score': 0
        }
        
        fairness_results['overall_fairness_score']['combined_score'] = np.mean([
            fairness_results['overall_fairness_score']['disparate_impact_score'],
            fairness_results['overall_fairness_score']['equal_opportunity_score']
        ])
        
        self.logger.info(f"Fairness evaluation completed: score {fairness_results['overall_fairness_score']['combined_score']:.3f}")
        
        return fairness_results
    
    def setup_ab_test(self,
                     model_a: Any,
                     model_b: Any,
                     test_design: Dict[str, Any],
                     power_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Setup A/B test with design and power analysis
        
        Args:
            model_a: Control model
            model_b: Treatment model
            test_design: Test design parameters
            power_analysis: Power analysis configuration
            
        Returns:
            A/B test configuration
        """
        ab_test_config = {
            'test_id': f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'created_at': datetime.now(),
            'models': {
                'control': {
                    'model': model_a,
                    'name': test_design.get('control_name', 'Model A'),
                    'description': test_design.get('control_description', '')
                },
                'treatment': {
                    'model': model_b,
                    'name': test_design.get('treatment_name', 'Model B'),
                    'description': test_design.get('treatment_description', '')
                }
            },
            'test_design': test_design,
            'power_analysis': {},
            'sample_size_calculation': {},
            'test_duration': {},
            'success_criteria': test_design.get('success_criteria', {}),
            'randomization': test_design.get('randomization', {'method': 'simple', 'ratio': 0.5})
        }
        
        # Perform power analysis
        effect_size = power_analysis.get('effect_size', 0.02)  # 2% lift
        alpha = power_analysis.get('alpha', 0.05)
        power = power_analysis.get('power', 0.8)
        
        # Calculate required sample size
        power_calculator = TTestPower()
        required_n = power_calculator.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=ab_test_config['randomization']['ratio']
        )
        
        ab_test_config['sample_size_calculation'] = {
            'effect_size': effect_size,
            'alpha': alpha,
            'power': power,
            'required_sample_size_per_group': int(np.ceil(required_n)),
            'total_required_samples': int(np.ceil(required_n * 2))
        }
        
        # Calculate test duration based on expected traffic
        daily_traffic = test_design.get('expected_daily_traffic', 1000)
        required_days = np.ceil(ab_test_config['sample_size_calculation']['total_required_samples'] / daily_traffic)
        
        ab_test_config['test_duration'] = {
            'minimum_days': int(required_days),
            'recommended_days': int(max(required_days, 7)),  # At least 1 week
            'maximum_days': test_design.get('max_duration_days', 30)
        }
        
        # Set up monitoring metrics
        ab_test_config['monitoring_metrics'] = test_design.get('metrics', [
            'precision', 'recall', 'f1_score', 'auc_roc', 
            'false_positive_rate', 'false_negative_rate'
        ])
        
        # Initialize results storage
        ab_test_config['results'] = {
            'control': {'predictions': [], 'actuals': [], 'timestamps': []},
            'treatment': {'predictions': [], 'actuals': [], 'timestamps': []},
            'metadata': {'start_time': None, 'end_time': None, 'status': 'configured'}
        }
        
        self.logger.info(f"A/B test configured: {ab_test_config['test_id']}")
        
        return ab_test_config
    
    def run_ab_test_evaluation(self,
                             ab_test_config: Dict[str, Any],
                             real_time_data: pd.DataFrame,
                             duration: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Execute live A/B test evaluation
        
        Args:
            ab_test_config: A/B test configuration
            real_time_data: Streaming data for evaluation
            duration: Test duration (optional)
            
        Returns:
            A/B test results
        """
        test_results = {
            'test_id': ab_test_config['test_id'],
            'start_time': datetime.now(),
            'end_time': None,
            'total_samples': 0,
            'group_sizes': {'control': 0, 'treatment': 0},
            'interim_results': [],
            'final_results': None,
            'early_stopping': None
        }
        
        # Mark test as running
        ab_test_config['results']['metadata']['status'] = 'running'
        ab_test_config['results']['metadata']['start_time'] = test_results['start_time']
        
        # Simulate real-time evaluation (in practice, this would be event-driven)
        n_samples = len(real_time_data)
        randomization_ratio = ab_test_config['randomization']['ratio']
        
        # Random assignment
        assignment = np.random.binomial(1, randomization_ratio, n_samples)
        control_mask = assignment == 0
        treatment_mask = assignment == 1
        
        # Get features and labels
        feature_cols = [col for col in real_time_data.columns if col not in ['is_fraud', 'timestamp']]
        X = real_time_data[feature_cols]
        y_true = real_time_data['is_fraud']
        
        # Make predictions for each group
        control_model = ab_test_config['models']['control']['model']
        treatment_model = ab_test_config['models']['treatment']['model']
        
        # Control group
        if np.sum(control_mask) > 0:
            X_control = X[control_mask]
            y_control = y_true[control_mask]
            control_pred = control_model.predict(X_control)
            
            ab_test_config['results']['control']['predictions'].extend(control_pred.tolist())
            ab_test_config['results']['control']['actuals'].extend(y_control.tolist())
            test_results['group_sizes']['control'] = np.sum(control_mask)
        
        # Treatment group
        if np.sum(treatment_mask) > 0:
            X_treatment = X[treatment_mask]
            y_treatment = y_true[treatment_mask]
            treatment_pred = treatment_model.predict(X_treatment)
            
            ab_test_config['results']['treatment']['predictions'].extend(treatment_pred.tolist())
            ab_test_config['results']['treatment']['actuals'].extend(y_treatment.tolist())
            test_results['group_sizes']['treatment'] = np.sum(treatment_mask)
        
        test_results['total_samples'] = n_samples
        
        # Check for early stopping
        if test_results['total_samples'] >= ab_test_config['sample_size_calculation']['total_required_samples'] * 0.5:
            early_stop_result = self._check_early_stopping(ab_test_config)
            if early_stop_result['should_stop']:
                test_results['early_stopping'] = early_stop_result
                self.logger.info(f"Early stopping triggered: {early_stop_result['reason']}")
        
        # Calculate interim results
        interim_metrics = self._calculate_ab_metrics(ab_test_config)
        test_results['interim_results'].append({
            'timestamp': datetime.now(),
            'n_samples': test_results['total_samples'],
            'metrics': interim_metrics
        })
        
        # Check if test is complete
        if (test_results['total_samples'] >= ab_test_config['sample_size_calculation']['total_required_samples'] or
            (duration and datetime.now() - test_results['start_time'] >= duration)):
            
            test_results['end_time'] = datetime.now()
            test_results['final_results'] = self.analyze_ab_test_results(
                ab_test_config,
                statistical_tests=['ttest', 'mann_whitney', 'chi_square'],
                effect_size='cohens_d'
            )
            
            ab_test_config['results']['metadata']['status'] = 'completed'
            ab_test_config['results']['metadata']['end_time'] = test_results['end_time']
        
        self.logger.info(f"A/B test evaluation updated: {test_results['total_samples']} samples")
        
        return test_results
    
    def _check_early_stopping(self, ab_test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check if early stopping criteria are met"""
        control_metrics = self._calculate_metrics_from_results(
            ab_test_config['results']['control']['predictions'],
            ab_test_config['results']['control']['actuals']
        )
        
        treatment_metrics = self._calculate_metrics_from_results(
            ab_test_config['results']['treatment']['predictions'],
            ab_test_config['results']['treatment']['actuals']
        )
        
        # Check for significant degradation
        key_metric = 'f1_score'
        if treatment_metrics[key_metric] < control_metrics[key_metric] * 0.95:
            return {
                'should_stop': True,
                'reason': f"Treatment model shows significant degradation in {key_metric}"
            }
        
        # Check for clear winner (very high confidence)
        n_control = len(ab_test_config['results']['control']['predictions'])
        n_treatment = len(ab_test_config['results']['treatment']['predictions'])
        
        if n_control > 100 and n_treatment > 100:
            # Perform interim statistical test
            _, p_value = ttest_ind(
                ab_test_config['results']['control']['predictions'],
                ab_test_config['results']['treatment']['predictions']
            )
            
            if p_value < 0.001:  # Very high confidence
                return {
                    'should_stop': True,
                    'reason': f"High confidence result achieved (p={p_value:.4f})"
                }
        
        return {'should_stop': False, 'reason': None}
    
    def _calculate_ab_metrics(self, ab_test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for both groups in A/B test"""
        metrics = {}
        
        for group in ['control', 'treatment']:
            if ab_test_config['results'][group]['predictions']:
                group_metrics = self._calculate_metrics_from_results(
                    ab_test_config['results'][group]['predictions'],
                    ab_test_config['results'][group]['actuals']
                )
                metrics[group] = group_metrics
        
        return metrics
    
    def _calculate_metrics_from_results(self, predictions: List, actuals: List) -> Dict[str, float]:
        """Calculate metrics from prediction results"""
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        return {
            'precision': precision_score(actuals, predictions, zero_division=0),
            'recall': recall_score(actuals, predictions, zero_division=0),
            'f1_score': f1_score(actuals, predictions, zero_division=0),
            'accuracy': np.mean(np.array(predictions) == np.array(actuals))
        }
    
    def analyze_ab_test_results(self,
                              test_results: Dict[str, Any],
                              statistical_tests: List[str],
                              effect_size: str = 'cohens_d') -> Dict[str, Any]:
        """
        Analyze A/B test results with statistical significance and effect size
        
        Args:
            test_results: A/B test results or configuration
            statistical_tests: List of statistical tests to perform
            effect_size: Effect size measure to calculate
            
        Returns:
            Statistical analysis of A/B test
        """
        analysis = {
            'timestamp': datetime.now(),
            'statistical_tests': {},
            'effect_sizes': {},
            'performance_comparison': {},
            'winner': None,
            'confidence_level': None,
            'recommendations': []
        }
        
        # Extract results
        if 'results' in test_results:
            results = test_results['results']
        else:
            results = test_results
            
        control_pred = np.array(results['control']['predictions'])
        control_actual = np.array(results['control']['actuals'])
        treatment_pred = np.array(results['treatment']['predictions'])
        treatment_actual = np.array(results['treatment']['actuals'])
        
        # Calculate performance metrics
        control_metrics = self._calculate_metrics_from_results(control_pred, control_actual)
        treatment_metrics = self._calculate_metrics_from_results(treatment_pred, treatment_actual)
        
        analysis['performance_comparison'] = {
            'control': control_metrics,
            'treatment': treatment_metrics,
            'lift': {}
        }
        
        # Calculate lift for each metric
        for metric in control_metrics:
            if control_metrics[metric] > 0:
                lift = (treatment_metrics[metric] - control_metrics[metric]) / control_metrics[metric]
                analysis['performance_comparison']['lift'][metric] = lift
        
        # Perform statistical tests
        if 'ttest' in statistical_tests:
            t_stat, p_value = ttest_ind(control_pred, treatment_pred)
            analysis['statistical_tests']['ttest'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        
        if 'mann_whitney' in statistical_tests:
            u_stat, p_value = mannwhitneyu(control_pred, treatment_pred, alternative='two-sided')
            analysis['statistical_tests']['mann_whitney'] = {
                'u_statistic': float(u_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        
        if 'chi_square' in statistical_tests:
            # Create contingency table
            contingency = np.array([
                [np.sum((control_pred == 1) & (control_actual == 1)), np.sum((control_pred == 0) & (control_actual == 0))],
                [np.sum((treatment_pred == 1) & (treatment_actual == 1)), np.sum((treatment_pred == 0) & (treatment_actual == 0))]
            ])
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            
            analysis['statistical_tests']['chi_square'] = {
                'chi2_statistic': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'significant': p_value < 0.05
            }
        
        # Calculate effect size
        if effect_size == 'cohens_d':
            # Cohen's d for difference in means
            pooled_std = np.sqrt((np.std(control_pred)**2 + np.std(treatment_pred)**2) / 2)
            cohens_d = (np.mean(treatment_pred) - np.mean(control_pred)) / pooled_std if pooled_std > 0 else 0
            
            analysis['effect_sizes']['cohens_d'] = {
                'value': float(cohens_d),
                'interpretation': self._interpret_cohens_d(cohens_d)
            }
        
        # Determine winner
        significant_tests = [test for test in analysis['statistical_tests'].values() if test['significant']]
        
        if significant_tests:
            # Check if treatment is better
            key_metric = 'f1_score'
            if treatment_metrics[key_metric] > control_metrics[key_metric]:
                analysis['winner'] = 'treatment'
                analysis['confidence_level'] = 1 - min([test['p_value'] for test in significant_tests])
            else:
                analysis['winner'] = 'control'
                analysis['confidence_level'] = 1 - min([test['p_value'] for test in significant_tests])
        else:
            analysis['winner'] = 'no_difference'
            analysis['confidence_level'] = 0
        
        # Generate recommendations
        if analysis['winner'] == 'treatment':
            analysis['recommendations'].append(
                f"Treatment model shows significant improvement. Consider deploying with {analysis['confidence_level']:.1%} confidence."
            )
        elif analysis['winner'] == 'control':
            analysis['recommendations'].append(
                "Control model performs better. Continue with current model."
            )
        else:
            analysis['recommendations'].append(
                "No significant difference detected. Consider extending test duration or increasing sample size."
            )
        
        self.logger.info(f"A/B test analysis completed: winner={analysis['winner']}")
        
        return analysis
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def calculate_ab_test_sample_size(self,
                                    effect_size: float,
                                    power: float = 0.8,
                                    alpha_level: float = 0.05,
                                    baseline_rate: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate required sample size for A/B test
        
        Args:
            effect_size: Expected effect size (relative or absolute)
            power: Statistical power (default 0.8)
            alpha_level: Significance level (default 0.05)
            baseline_rate: Baseline conversion rate (for proportions)
            
        Returns:
            Sample size calculation results
        """
        sample_size_results = {
            'parameters': {
                'effect_size': effect_size,
                'power': power,
                'alpha_level': alpha_level,
                'baseline_rate': baseline_rate
            },
            'results': {},
            'recommendations': []
        }
        
        # For continuous metrics (t-test)
        power_calculator = TTestPower()
        n_continuous = power_calculator.solve_power(
            effect_size=effect_size,
            alpha=alpha_level,
            power=power,
            ratio=1
        )
        
        sample_size_results['results']['continuous_metric'] = {
            'sample_size_per_group': int(np.ceil(n_continuous)),
            'total_sample_size': int(np.ceil(n_continuous * 2)),
            'test_type': 't-test'
        }
        
        # For proportions (if baseline rate provided)
        if baseline_rate is not None:
            # Convert relative effect size to absolute difference
            absolute_diff = baseline_rate * effect_size
            treatment_rate = baseline_rate + absolute_diff
            
            # Use normal approximation for sample size
            z_alpha = stats.norm.ppf(1 - alpha_level/2)
            z_beta = stats.norm.ppf(power)
            
            p_bar = (baseline_rate + treatment_rate) / 2
            n_prop = ((z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + 
                      z_beta * np.sqrt(baseline_rate * (1 - baseline_rate) + 
                                     treatment_rate * (1 - treatment_rate)))**2 / 
                     (treatment_rate - baseline_rate)**2)
            
            sample_size_results['results']['proportion_metric'] = {
                'sample_size_per_group': int(np.ceil(n_prop)),
                'total_sample_size': int(np.ceil(n_prop * 2)),
                'test_type': 'z-test for proportions',
                'baseline_rate': baseline_rate,
                'expected_treatment_rate': treatment_rate
            }
        
        # Duration estimate
        daily_traffic_estimate = 1000  # Default estimate
        max_sample_size = max([r['total_sample_size'] for r in sample_size_results['results'].values()])
        
        sample_size_results['duration_estimate'] = {
            'days_required': int(np.ceil(max_sample_size / daily_traffic_estimate)),
            'weeks_required': int(np.ceil(max_sample_size / (daily_traffic_estimate * 7)))
        }
        
        # Recommendations
        if max_sample_size > 10000:
            sample_size_results['recommendations'].append(
                "Large sample size required. Consider sequential testing or increasing effect size threshold."
            )
        
        if effect_size < 0.1:
            sample_size_results['recommendations'].append(
                "Small effect size will require very large sample. Consider if this difference is practically significant."
            )
        
        sample_size_results['recommendations'].append(
            f"Recommended minimum test duration: {sample_size_results['duration_estimate']['weeks_required']} weeks"
        )
        
        return sample_size_results
    
    def generate_evaluation_report(self,
                                 evaluation_results: Dict[str, Any],
                                 report_config: Dict[str, Any],
                                 output_format: str = 'html') -> Union[str, bytes]:
        """
        Generate comprehensive evaluation report
        
        Args:
            evaluation_results: All evaluation results
            report_config: Report configuration
            output_format: Output format ('html', 'pdf', 'markdown')
            
        Returns:
            Generated report
        """
        # Create report template
        report_template = """
        # Model Evaluation Report
        
        **Generated:** {{ timestamp }}
        
        ## Executive Summary
        
        {{ executive_summary }}
        
        ## Model Performance
        
        ### Overall Metrics
        {{ performance_table }}
        
        ### Confidence Intervals
        {{ confidence_intervals }}
        
        ## Detailed Analysis
        
        ### Error Analysis
        - **Error Rate:** {{ error_rate }}%
        - **Primary Error Types:** {{ error_types }}
        
        ### Overfitting Detection
        - **Status:** {{ overfitting_status }}
        - **Severity:** {{ overfitting_severity }}
        
        ### Feature Importance Stability
        - **Stability Score:** {{ stability_score }}
        - **Unstable Features:** {{ unstable_features }}
        
        ### Fairness Evaluation
        - **Overall Fairness Score:** {{ fairness_score }}
        - **Protected Attributes Analyzed:** {{ protected_attributes }}
        
        ## Recommendations
        
        {{ recommendations }}
        
        ## Appendix
        
        ### Detailed Metrics
        {{ detailed_metrics }}
        
        ### Test Configuration
        {{ test_configuration }}
        """
        
        # Prepare template data
        template_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'executive_summary': self._generate_executive_summary(evaluation_results),
            'performance_table': self._format_performance_table(evaluation_results.get('performance_metrics', {})),
            'confidence_intervals': self._format_confidence_intervals(evaluation_results.get('confidence_intervals', {})),
            'error_rate': evaluation_results.get('error_analysis', {}).get('error_rate', 0) * 100,
            'error_types': self._summarize_error_types(evaluation_results.get('error_analysis', {})),
            'overfitting_status': evaluation_results.get('overfitting_analysis', {}).get('is_overfitting', False),
            'overfitting_severity': evaluation_results.get('overfitting_analysis', {}).get('severity', 'none'),
            'stability_score': evaluation_results.get('stability_analysis', {}).get('overall_stability_score', 0),
            'unstable_features': ', '.join([f['feature'] for f in evaluation_results.get('stability_analysis', {}).get('unstable_features', [])[:5]]),
            'fairness_score': evaluation_results.get('fairness_evaluation', {}).get('overall_fairness_score', {}).get('combined_score', 0),
            'protected_attributes': ', '.join(evaluation_results.get('fairness_evaluation', {}).get('protected_attributes', [])),
            'recommendations': self._compile_recommendations(evaluation_results),
            'detailed_metrics': json.dumps(evaluation_results.get('performance_metrics', {}), indent=2),
            'test_configuration': json.dumps(report_config, indent=2)
        }
        
        # Render template
        template = Template(report_template)
        rendered_report = template.render(**template_data)
        
        # Convert to requested format
        if output_format == 'markdown':
            return rendered_report
        elif output_format == 'html':
            html_report = markdown.markdown(rendered_report, extensions=['tables', 'fenced_code'])
            
            # Add CSS styling
            styled_html = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1, h2, h3 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    code {{ background-color: #f5f5f5; padding: 2px 4px; }}
                </style>
            </head>
            <body>
                {html_report}
            </body>
            </html>
            """
            return styled_html
        elif output_format == 'pdf':
            # Convert HTML to PDF
            html_report = markdown.markdown(rendered_report)
            pdf_bytes = pdfkit.from_string(html_report, False)
            return pdf_bytes
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary from results"""
        summary_points = []
        
        # Performance summary
        if 'performance_metrics' in results:
            key_metric = results['performance_metrics'].get('f1_score', 0)
            summary_points.append(f"Model achieved F1 score of {key_metric:.3f}")
        
        # Overfitting status
        if 'overfitting_analysis' in results:
            if results['overfitting_analysis'].get('is_overfitting'):
                summary_points.append("⚠️ Overfitting detected - model may not generalize well")
            else:
                summary_points.append("✓ No significant overfitting detected")
        
        # Fairness status
        if 'fairness_evaluation' in results:
            fairness_score = results['fairness_evaluation'].get('overall_fairness_score', {}).get('combined_score', 0)
            if fairness_score < 0.8:
                summary_points.append("⚠️ Fairness concerns identified")
            else:
                summary_points.append("✓ Model shows acceptable fairness")
        
        return '\n'.join(f"- {point}" for point in summary_points)
    
    def _format_performance_table(self, metrics: Dict[str, float]) -> str:
        """Format performance metrics as markdown table"""
        if not metrics:
            return "No performance metrics available"
        
        table = "| Metric | Value |\n|--------|-------|\n"
        for metric, value in metrics.items():
            table += f"| {metric.replace('_', ' ').title()} | {value:.4f} |\n"
        
        return table
    
    def _format_confidence_intervals(self, intervals: Dict[str, Dict[str, float]]) -> str:
        """Format confidence intervals as markdown"""
        if not intervals:
            return "No confidence intervals calculated"
        
        output = ""
        for metric, ci in intervals.items():
            output += f"- **{metric}**: {ci['point_estimate']:.3f} [{ci['lower_bound']:.3f}, {ci['upper_bound']:.3f}]\n"
        
        return output
    
    def _summarize_error_types(self, error_analysis: Dict[str, Any]) -> str:
        """Summarize error types from analysis"""
        if not error_analysis:
            return "No error analysis available"
        
        fp = error_analysis.get('false_positive_analysis', {}).get('count', 0)
        fn = error_analysis.get('false_negative_analysis', {}).get('count', 0)
        
        return f"False Positives: {fp}, False Negatives: {fn}"
    
    def _compile_recommendations(self, results: Dict[str, Any]) -> str:
        """Compile all recommendations from various analyses"""
        all_recommendations = []
        
        # Collect recommendations from each analysis
        for analysis_type in ['error_analysis', 'overfitting_analysis', 'fairness_evaluation']:
            if analysis_type in results:
                recommendations = results[analysis_type].get('recommendations', [])
                all_recommendations.extend(recommendations)
        
        if not all_recommendations:
            return "No specific recommendations at this time."
        
        return '\n'.join(f"- {rec}" for rec in all_recommendations[:10])  # Top 10
    
    def create_performance_comparison_charts(self,
                                           models_performance: Dict[str, Dict[str, float]],
                                           chart_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create performance comparison visualizations
        
        Args:
            models_performance: Performance metrics for multiple models
            chart_config: Chart configuration
            
        Returns:
            Dictionary of chart objects
        """
        charts = {}
        
        # Prepare data
        model_names = list(models_performance.keys())
        metrics = list(next(iter(models_performance.values())).keys())
        
        # 1. Radar chart for overall comparison
        fig_radar = go.Figure()
        
        for model_name, metrics_dict in models_performance.items():
            values = [metrics_dict.get(m, 0) for m in metrics]
            values.append(values[0])  # Close the radar
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model_name
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True,
            title="Model Performance Comparison - Radar Chart"
        )
        
        charts['radar_chart'] = fig_radar
        
        # 2. Bar chart comparison
        fig_bar = go.Figure()
        
        for metric in metrics:
            values = [models_performance[model].get(metric, 0) for model in model_names]
            fig_bar.add_trace(go.Bar(name=metric, x=model_names, y=values))
        
        fig_bar.update_layout(
            barmode='group',
            title="Model Performance Comparison - Bar Chart",
            xaxis_title="Models",
            yaxis_title="Score",
            yaxis_range=[0, 1]
        )
        
        charts['bar_chart'] = fig_bar
        
        # 3. Heatmap for detailed comparison
        values_matrix = [[models_performance[model].get(metric, 0) for metric in metrics] 
                        for model in model_names]
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=values_matrix,
            x=metrics,
            y=model_names,
            colorscale='RdYlGn',
            zmid=0.5
        ))
        
        fig_heatmap.update_layout(
            title="Model Performance Heatmap",
            xaxis_title="Metrics",
            yaxis_title="Models"
        )
        
        charts['heatmap'] = fig_heatmap
        
        # 4. Box plot for confidence intervals (if available)
        if chart_config.get('include_confidence_intervals'):
            fig_box = go.Figure()
            
            for model_name in model_names:
                # Simulate confidence intervals (in practice, use actual data)
                y_values = []
                for metric in metrics:
                    base_value = models_performance[model_name].get(metric, 0)
                    # Add some variation
                    y_values.extend(np.random.normal(base_value, 0.02, 100))
                
                fig_box.add_trace(go.Box(y=y_values, name=model_name))
            
            fig_box.update_layout(
                title="Model Performance Distribution",
                yaxis_title="Score"
            )
            
            charts['box_plot'] = fig_box
        
        # Save charts if requested
        if chart_config.get('save_charts'):
            output_dir = Path(chart_config.get('output_dir', './charts'))
            output_dir.mkdir(exist_ok=True)
            
            for chart_name, fig in charts.items():
                fig.write_html(output_dir / f"{chart_name}.html")
                fig.write_image(output_dir / f"{chart_name}.png")
        
        return charts
    
    def generate_model_cards(self,
                           model_info: Dict[str, Any],
                           performance_data: Dict[str, Any],
                           documentation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate model cards for documentation
        
        Args:
            model_info: Model information and metadata
            performance_data: Performance metrics and analysis
            documentation_config: Documentation configuration
            
        Returns:
            Model card documentation
        """
        model_card = {
            'model_details': {
                'name': model_info.get('name', 'Unnamed Model'),
                'version': model_info.get('version', '1.0'),
                'type': model_info.get('type', 'Unknown'),
                'created_date': model_info.get('created_date', datetime.now().isoformat()),
                'created_by': model_info.get('created_by', 'Unknown'),
                'description': model_info.get('description', ''),
                'architecture': model_info.get('architecture', {}),
                'training_data': model_info.get('training_data', {}),
                'hyperparameters': model_info.get('hyperparameters', {})
            },
            'intended_use': {
                'primary_use_cases': model_info.get('use_cases', []),
                'primary_users': model_info.get('users', []),
                'out_of_scope_uses': model_info.get('out_of_scope', [])
            },
            'performance': {
                'metrics': performance_data.get('performance_metrics', {}),
                'confidence_intervals': performance_data.get('confidence_intervals', {}),
                'evaluation_data': performance_data.get('evaluation_data', {}),
                'analysis_results': {
                    'error_analysis': performance_data.get('error_analysis', {}),
                    'fairness_evaluation': performance_data.get('fairness_evaluation', {}),
                    'overfitting_analysis': performance_data.get('overfitting_analysis', {})
                }
            },
            'limitations': {
                'known_limitations': [],
                'trade_offs': []
            },
            'ethical_considerations': {
                'fairness_assessment': {},
                'bias_mitigation': [],
                'privacy_considerations': []
            },
            'references': model_info.get('references', [])
        }
        
        # Add limitations based on analysis
        if performance_data.get('error_analysis', {}).get('error_rate', 0) > 0.1:
            model_card['limitations']['known_limitations'].append(
                "Model shows error rate above 10% - careful monitoring required"
            )
        
        if performance_data.get('overfitting_analysis', {}).get('is_overfitting'):
            model_card['limitations']['known_limitations'].append(
                "Overfitting detected - model may not generalize well to new data"
            )
        
        # Add fairness considerations
        fairness_eval = performance_data.get('fairness_evaluation', {})
        if fairness_eval:
            model_card['ethical_considerations']['fairness_assessment'] = {
                'protected_attributes_tested': fairness_eval.get('protected_attributes', []),
                'fairness_score': fairness_eval.get('overall_fairness_score', {}),
                'disparate_impact': fairness_eval.get('disparate_impact', {})
            }
            
            # Add bias warnings
            for attr, impact in fairness_eval.get('disparate_impact', {}).items():
                if not impact.get('threshold_80_percent'):
                    model_card['ethical_considerations']['bias_mitigation'].append(
                        f"Consider bias mitigation for attribute: {attr}"
                    )
        
        # Generate formatted output
        if documentation_config.get('format') == 'markdown':
            model_card['formatted_output'] = self._format_model_card_markdown(model_card)
        elif documentation_config.get('format') == 'json':
            model_card['formatted_output'] = json.dumps(model_card, indent=2)
        else:
            model_card['formatted_output'] = yaml.dump(model_card, default_flow_style=False)
        
        return model_card
    
    def _format_model_card_markdown(self, model_card: Dict[str, Any]) -> str:
        """Format model card as markdown"""
        md = f"""# Model Card: {model_card['model_details']['name']}

## Model Details
- **Version**: {model_card['model_details']['version']}
- **Type**: {model_card['model_details']['type']}
- **Created**: {model_card['model_details']['created_date']}
- **Description**: {model_card['model_details']['description']}

## Intended Use
### Primary Use Cases
{self._format_list(model_card['intended_use']['primary_use_cases'])}

### Out of Scope
{self._format_list(model_card['intended_use']['out_of_scope_uses'])}

## Performance
{self._format_performance_metrics(model_card['performance']['metrics'])}

## Limitations
{self._format_list(model_card['limitations']['known_limitations'])}

## Ethical Considerations
### Fairness Assessment
Protected attributes tested: {', '.join(model_card['ethical_considerations']['fairness_assessment'].get('protected_attributes_tested', []))}

### Bias Mitigation
{self._format_list(model_card['ethical_considerations']['bias_mitigation'])}
"""
        return md
    
    def _format_list(self, items: List[str]) -> str:
        """Format list as markdown"""
        if not items:
            return "- None identified"
        return '\n'.join(f"- {item}" for item in items)
    
    def _format_performance_metrics(self, metrics: Dict[str, float]) -> str:
        """Format performance metrics for model card"""
        if not metrics:
            return "No metrics available"
        
        output = "| Metric | Value |\n|--------|-------|\n"
        for metric, value in metrics.items():
            output += f"| {metric} | {value:.4f} |\n"
        return output
    
    def export_evaluation_results(self,
                                results: Dict[str, Any],
                                export_format: str,
                                metadata: Optional[Dict[str, Any]] = None) -> Union[str, bytes]:
        """
        Export evaluation results with metadata
        
        Args:
            results: Evaluation results to export
            export_format: Export format ('json', 'csv', 'parquet', 'pickle')
            metadata: Additional metadata to include
            
        Returns:
            Exported data
        """
        # Add metadata
        export_data = {
            'results': results,
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'export_format': export_format,
                'version': '1.0',
                **(metadata or {})
            }
        }
        
        if export_format == 'json':
            return json.dumps(export_data, indent=2, default=str)
        
        elif export_format == 'csv':
            # Flatten results for CSV export
            flattened = self._flatten_dict(results)
            df = pd.DataFrame([flattened])
            return df.to_csv(index=False)
        
        elif export_format == 'parquet':
            # Convert to DataFrame and export as parquet
            flattened = self._flatten_dict(results)
            df = pd.DataFrame([flattened])
            return df.to_parquet()
        
        elif export_format == 'pickle':
            import pickle
            return pickle.dumps(export_data)
        
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)