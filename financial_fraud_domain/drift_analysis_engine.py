"""
Drift Analysis Engine for Model Performance Impact Assessment
Advanced drift quantification and impact analysis using statistical metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
import time
import threading
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Statistical imports with fallbacks
try:
    from scipy.spatial.distance import jensenshannon
    JS_AVAILABLE = True
except ImportError:
    JS_AVAILABLE = False
    warnings.warn("Jensen-Shannon distance not available. Install scipy>=1.7.0 for JS divergence support.")


class DriftAnalysisEngine:
    """
    Engine for performing comprehensive drift impact analysis.
    Supports PSI, KL divergence, Wasserstein distance, correlation analysis, and impact assessment.
    """
    
    def __init__(self, logger: logging.Logger = None):
        """Initialize the drift analysis engine."""
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._lock = threading.Lock()
    
    def analyze_drift_impact(self, drift_data: Dict[str, Any], 
                            accuracy_data: Dict[str, Any], 
                            impact_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive drift impact analysis"""
        with self._lock:
            # Prepare data for analysis
            prepared_drift = self._prepare_drift_data(drift_data)
            prepared_accuracy = self._prepare_accuracy_data(accuracy_data)
            
            # Initialize results structure
            analysis_results = {
                "drift_metrics": {},
                "impact_correlation": {},
                "threshold_analysis": {},
                "trend_analysis": {},
                "feature_impact": {},
                "overall_impact": {},
                "data_points": 0
            }
            
            # Calculate drift metrics
            if impact_metrics.get("psi", True):
                psi_results = self._calculate_psi_impact(prepared_drift, prepared_accuracy)
                analysis_results["drift_metrics"]["psi"] = psi_results
            
            if impact_metrics.get("kl_divergence", True):
                kl_results = self._calculate_kl_divergence_impact(prepared_drift, prepared_accuracy)
                analysis_results["drift_metrics"]["kl_divergence"] = kl_results
            
            if impact_metrics.get("wasserstein_distance", True):
                wasserstein_results = self._calculate_wasserstein_impact(prepared_drift, prepared_accuracy)
                analysis_results["drift_metrics"]["wasserstein_distance"] = wasserstein_results
            
            if impact_metrics.get("js_divergence", False):
                js_results = self._calculate_js_divergence_impact(prepared_drift, prepared_accuracy)
                analysis_results["drift_metrics"]["js_divergence"] = js_results
            
            # Calculate accuracy correlation
            if impact_metrics.get("accuracy_correlation", True):
                correlation_results = self._calculate_drift_accuracy_correlation(
                    prepared_drift, 
                    prepared_accuracy,
                    analysis_results["drift_metrics"]
                )
                analysis_results["impact_correlation"] = correlation_results
                analysis_results["data_points"] = correlation_results.get("data_points", 0)
            
            # Analyze feature-level impact
            feature_impact = self._analyze_feature_drift_impact(prepared_drift, prepared_accuracy)
            analysis_results["feature_impact"] = feature_impact
            
            # Threshold analysis
            threshold_results = self._perform_threshold_analysis(
                analysis_results["drift_metrics"],
                impact_metrics.get("thresholds", {})
            )
            analysis_results["threshold_analysis"] = threshold_results
            
            # Trend analysis
            trend_results = self._analyze_drift_impact_trends(
                analysis_results["drift_metrics"],
                analysis_results["impact_correlation"]
            )
            analysis_results["trend_analysis"] = trend_results
            
            # Calculate overall impact score
            overall_impact = self._calculate_overall_drift_impact(analysis_results)
            analysis_results["overall_impact"] = overall_impact
            
            return analysis_results
    
    def _prepare_drift_data(self, drift_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare drift data for impact analysis"""
        prepared = {
            "features": {},
            "overall_drift": {},
            "timestamps": [],
            "total_features": 0
        }
        
        # Handle different drift data formats
        if "features" in drift_data:
            for feature, feature_drift in drift_data["features"].items():
                if isinstance(feature_drift, dict):
                    prepared["features"][feature] = {
                        "drift_scores": feature_drift.get("scores", []),
                        "drift_detected": feature_drift.get("detected", False),
                        "drift_type": feature_drift.get("type", "unknown"),
                        "distribution_changes": feature_drift.get("distribution_changes", {})
                    }
                    prepared["total_features"] += 1
        
        # Extract overall drift metrics
        if "overall_metrics" in drift_data:
            prepared["overall_drift"] = drift_data["overall_metrics"]
        
        # Extract timestamps
        if "timestamps" in drift_data:
            prepared["timestamps"] = drift_data["timestamps"]
        
        return prepared
    
    def _prepare_accuracy_data(self, accuracy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare accuracy data for drift correlation analysis"""
        prepared = {
            "accuracy_values": [],
            "timestamps": [],
            "metrics": {}
        }
        
        # Extract accuracy time series
        if "history" in accuracy_data:
            for entry in accuracy_data["history"]:
                if "accuracy" in entry and "timestamp" in entry:
                    prepared["accuracy_values"].append(entry["accuracy"])
                    prepared["timestamps"].append(entry["timestamp"])
        
        elif "accuracy" in accuracy_data:
            # Direct accuracy values
            if isinstance(accuracy_data["accuracy"], list):
                prepared["accuracy_values"] = accuracy_data["accuracy"]
            else:
                prepared["accuracy_values"] = [accuracy_data["accuracy"]]
        
        # Extract other metrics if available
        for metric in ["precision", "recall", "f1_score"]:
            if metric in accuracy_data:
                if isinstance(accuracy_data[metric], list):
                    prepared["metrics"][metric] = accuracy_data[metric]
                else:
                    prepared["metrics"][metric] = [accuracy_data[metric]]
        
        return prepared
    
    def _calculate_psi_impact(self, drift_data: Dict[str, Any], 
                             accuracy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Population Stability Index (PSI) impact"""
        try:
            psi_scores = []
            feature_psi = {}
            
            # Calculate PSI for each feature
            for feature, feature_drift in drift_data.get("features", {}).items():
                if "distribution_changes" in feature_drift:
                    dist_changes = feature_drift["distribution_changes"]
                    
                    # PSI calculation
                    if "expected" in dist_changes and "actual" in dist_changes:
                        expected = np.array(dist_changes["expected"])
                        actual = np.array(dist_changes["actual"])
                        
                        # Ensure same length and normalize
                        min_len = min(len(expected), len(actual))
                        expected = expected[:min_len]
                        actual = actual[:min_len]
                        
                        # Normalize to probabilities
                        expected = expected / np.sum(expected) if np.sum(expected) > 0 else expected
                        actual = actual / np.sum(actual) if np.sum(actual) > 0 else actual
                        
                        # Calculate PSI
                        psi = 0
                        for i in range(len(expected)):
                            if expected[i] > 0 and actual[i] > 0:
                                psi += (actual[i] - expected[i]) * np.log(actual[i] / expected[i])
                        
                        psi_scores.append(psi)
                        feature_psi[feature] = {
                            "psi_value": float(psi),
                            "stability_level": self._interpret_psi_score(psi)
                        }
            
            # Overall PSI
            if psi_scores:
                overall_psi = np.mean(psi_scores)
                max_psi = np.max(psi_scores)
                
                return {
                    "overall_psi": float(overall_psi),
                    "max_psi": float(max_psi),
                    "feature_psi": feature_psi,
                    "stability_assessment": self._interpret_psi_score(overall_psi),
                    "num_features": len(psi_scores)
                }
            
            return {"status": "no_psi_data"}
            
        except Exception as e:
            self.logger.error(f"Error calculating PSI impact: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_kl_divergence_impact(self, drift_data: Dict[str, Any], 
                                       accuracy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Kullback-Leibler divergence impact"""
        try:
            kl_scores = []
            feature_kl = {}
            
            # Calculate KL divergence for each feature
            for feature, feature_drift in drift_data.get("features", {}).items():
                if "distribution_changes" in feature_drift:
                    dist_changes = feature_drift["distribution_changes"]
                    
                    if "expected" in dist_changes and "actual" in dist_changes:
                        expected = np.array(dist_changes["expected"])
                        actual = np.array(dist_changes["actual"])
                        
                        # Ensure same length and normalize
                        min_len = min(len(expected), len(actual))
                        expected = expected[:min_len]
                        actual = actual[:min_len]
                        
                        # Add small epsilon to avoid log(0)
                        epsilon = 1e-10
                        expected = expected + epsilon
                        actual = actual + epsilon
                        
                        # Normalize
                        expected = expected / np.sum(expected)
                        actual = actual / np.sum(actual)
                        
                        # Calculate KL divergence
                        kl_div = np.sum(actual * np.log(actual / expected))
                        
                        kl_scores.append(kl_div)
                        feature_kl[feature] = {
                            "kl_divergence": float(kl_div),
                            "divergence_level": self._interpret_kl_divergence(kl_div)
                        }
            
            # Overall KL divergence
            if kl_scores:
                overall_kl = np.mean(kl_scores)
                max_kl = np.max(kl_scores)
                
                return {
                    "overall_kl_divergence": float(overall_kl),
                    "max_kl_divergence": float(max_kl),
                    "feature_kl_divergence": feature_kl,
                    "divergence_assessment": self._interpret_kl_divergence(overall_kl),
                    "num_features": len(kl_scores)
                }
            
            return {"status": "no_kl_data"}
            
        except Exception as e:
            self.logger.error(f"Error calculating KL divergence impact: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_wasserstein_impact(self, drift_data: Dict[str, Any], 
                                     accuracy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Wasserstein distance impact"""
        try:
            wasserstein_scores = []
            feature_wasserstein = {}
            
            # Calculate Wasserstein distance for each feature
            for feature, feature_drift in drift_data.get("features", {}).items():
                if "distribution_changes" in feature_drift:
                    dist_changes = feature_drift["distribution_changes"]
                    
                    if "expected_values" in dist_changes and "actual_values" in dist_changes:
                        expected = np.array(dist_changes["expected_values"])
                        actual = np.array(dist_changes["actual_values"])
                        
                        # Calculate Wasserstein distance
                        w_dist = stats.wasserstein_distance(expected, actual)
                        
                        wasserstein_scores.append(w_dist)
                        feature_wasserstein[feature] = {
                            "wasserstein_distance": float(w_dist),
                            "distance_level": self._interpret_wasserstein_distance(w_dist)
                        }
            
            # Overall Wasserstein distance
            if wasserstein_scores:
                overall_wasserstein = np.mean(wasserstein_scores)
                max_wasserstein = np.max(wasserstein_scores)
                
                return {
                    "overall_wasserstein": float(overall_wasserstein),
                    "max_wasserstein": float(max_wasserstein),
                    "feature_wasserstein": feature_wasserstein,
                    "distance_assessment": self._interpret_wasserstein_distance(overall_wasserstein),
                    "num_features": len(wasserstein_scores)
                }
            
            return {"status": "no_wasserstein_data"}
            
        except Exception as e:
            self.logger.error(f"Error calculating Wasserstein distance impact: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_js_divergence_impact(self, drift_data: Dict[str, Any], 
                                       accuracy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Jensen-Shannon divergence impact"""
        if not JS_AVAILABLE:
            return {"status": "js_not_available"}
        
        try:
            js_scores = []
            feature_js = {}
            
            # Calculate JS divergence for each feature
            for feature, feature_drift in drift_data.get("features", {}).items():
                if "distribution_changes" in feature_drift:
                    dist_changes = feature_drift["distribution_changes"]
                    
                    if "expected" in dist_changes and "actual" in dist_changes:
                        expected = np.array(dist_changes["expected"])
                        actual = np.array(dist_changes["actual"])
                        
                        # Ensure same length and normalize
                        min_len = min(len(expected), len(actual))
                        expected = expected[:min_len]
                        actual = actual[:min_len]
                        
                        # Normalize to probabilities
                        expected = expected / np.sum(expected) if np.sum(expected) > 0 else expected
                        actual = actual / np.sum(actual) if np.sum(actual) > 0 else actual
                        
                        # Calculate JS divergence
                        js_div = jensenshannon(expected, actual) ** 2  # Squared for JS divergence
                        
                        js_scores.append(js_div)
                        feature_js[feature] = {
                            "js_divergence": float(js_div),
                            "divergence_level": self._interpret_js_divergence(js_div)
                        }
            
            # Overall JS divergence
            if js_scores:
                overall_js = np.mean(js_scores)
                max_js = np.max(js_scores)
                
                return {
                    "overall_js_divergence": float(overall_js),
                    "max_js_divergence": float(max_js),
                    "feature_js_divergence": feature_js,
                    "divergence_assessment": self._interpret_js_divergence(overall_js),
                    "num_features": len(js_scores)
                }
            
            return {"status": "no_js_data"}
            
        except Exception as e:
            self.logger.error(f"Error calculating JS divergence impact: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_drift_accuracy_correlation(self, drift_data: Dict[str, Any], 
                                            accuracy_data: Dict[str, Any],
                                            drift_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate correlation between drift and accuracy degradation"""
        try:
            correlation_results = {
                "drift_accuracy_correlation": {},
                "lag_correlation": {},
                "causality_analysis": {},
                "impact_timeline": {},
                "data_points": 0
            }
            
            # Get accuracy time series
            accuracy_values = accuracy_data.get("accuracy_values", [])
            accuracy_timestamps = accuracy_data.get("timestamps", [])
            
            if not accuracy_values:
                return {"status": "no_accuracy_data"}
            
            # Aggregate drift scores over time
            drift_scores_timeline = self._aggregate_drift_scores_timeline(drift_data, drift_metrics)
            
            if not drift_scores_timeline:
                return {"status": "no_drift_timeline"}
            
            # Align time series
            aligned_drift, aligned_accuracy = self._align_time_series(
                drift_scores_timeline,
                list(zip(accuracy_timestamps, accuracy_values)) if accuracy_timestamps else enumerate(accuracy_values)
            )
            
            if len(aligned_drift) < 10:  # Minimum points for meaningful correlation
                return {"status": "insufficient_data", "data_points": len(aligned_drift)}
            
            correlation_results["data_points"] = len(aligned_drift)
            
            # Calculate direct correlation
            if len(aligned_drift) > 0 and len(aligned_accuracy) > 0:
                pearson_corr, pearson_p = stats.pearsonr(aligned_drift, aligned_accuracy)
                spearman_corr, spearman_p = stats.spearmanr(aligned_drift, aligned_accuracy)
                
                correlation_results["drift_accuracy_correlation"] = {
                    "pearson": {
                        "coefficient": float(pearson_corr),
                        "p_value": float(pearson_p),
                        "significant": pearson_p < 0.05
                    },
                    "spearman": {
                        "coefficient": float(spearman_corr),
                        "p_value": float(spearman_p),
                        "significant": spearman_p < 0.05
                    },
                    "correlation_strength": self._interpret_correlation(abs(pearson_corr))
                }
            
            # Calculate lag correlation
            lag_correlations = self._calculate_lag_correlations(aligned_drift, aligned_accuracy)
            correlation_results["lag_correlation"] = lag_correlations
            
            # Causality analysis
            causality = self._analyze_drift_causality(aligned_drift, aligned_accuracy)
            correlation_results["causality_analysis"] = causality
            
            # Impact timeline
            impact_timeline = self._create_impact_timeline(
                aligned_drift, 
                aligned_accuracy,
                accuracy_timestamps[:len(aligned_accuracy)] if accuracy_timestamps else None
            )
            correlation_results["impact_timeline"] = impact_timeline
            
            return correlation_results
            
        except Exception as e:
            self.logger.error(f"Error calculating drift-accuracy correlation: {e}")
            return {"status": "error", "error": str(e)}
    
    def _aggregate_drift_scores_timeline(self, drift_data: Dict[str, Any], 
                                        drift_metrics: Dict[str, Any]) -> List[Tuple[Any, float]]:
        """Aggregate drift scores into timeline"""
        timeline = []
        
        # Extract overall drift scores if available
        if "overall_drift" in drift_data:
            overall = drift_data["overall_drift"]
            if "timeline" in overall:
                for entry in overall["timeline"]:
                    if "timestamp" in entry and "score" in entry:
                        timeline.append((entry["timestamp"], entry["score"]))
            elif "scores" in overall:
                # Use indices as timestamps if no timestamps available
                for i, score in enumerate(overall["scores"]):
                    timeline.append((i, score))
        
        # If no overall drift, aggregate from drift metrics
        if not timeline and drift_metrics:
            # Combine different drift metrics
            combined_scores = []
            
            for metric_name, metric_data in drift_metrics.items():
                if isinstance(metric_data, dict) and "overall" in metric_data:
                    score = metric_data["overall"]
                    # Normalize different metrics to [0, 1] range
                    if metric_name == "psi":
                        normalized = min(score / 0.5, 1.0)  # PSI > 0.5 is very high
                    elif metric_name == "kl_divergence":
                        normalized = min(score / 1.0, 1.0)  # KL > 1.0 is very high
                    elif metric_name == "wasserstein_distance":
                        normalized = min(score / 0.2, 1.0)  # Wasserstein > 0.2 is high
                    else:
                        normalized = score
                    
                    combined_scores.append(normalized)
            
            if combined_scores:
                # Create single timeline point with average score
                avg_score = np.mean(combined_scores)
                timeline.append((0, avg_score))
        
        return sorted(timeline, key=lambda x: x[0])
    
    def _align_time_series(self, series1: List[Tuple[Any, float]], 
                          series2: List[Tuple[Any, float]]) -> Tuple[List[float], List[float]]:
        """Align two time series for correlation analysis"""
        # Convert to dictionaries for easier lookup
        dict1 = {t: v for t, v in series1}
        dict2 = {t: v for t, v in series2}
        
        # Find common timestamps
        common_times = sorted(set(dict1.keys()) & set(dict2.keys()))
        
        if not common_times:
            # If no common timestamps, try to interpolate
            # For now, just use the shorter series length
            min_len = min(len(series1), len(series2))
            values1 = [v for _, v in series1[:min_len]]
            values2 = [v for _, v in series2[:min_len]]
            return values1, values2
        
        # Extract aligned values
        aligned1 = [dict1[t] for t in common_times]
        aligned2 = [dict2[t] for t in common_times]
        
        return aligned1, aligned2
    
    def _calculate_lag_correlations(self, drift_series: List[float], 
                                   accuracy_series: List[float], 
                                   max_lag: int = 10) -> Dict[str, Any]:
        """Calculate correlation at different time lags"""
        lag_results = {
            "optimal_lag": 0,
            "max_correlation": 0,
            "lag_correlations": {}
        }
        
        max_lag = min(max_lag, len(drift_series) // 2)  # Ensure enough data
        
        for lag in range(0, max_lag + 1):
            if lag == 0:
                # No lag
                corr, p_value = stats.pearsonr(drift_series, accuracy_series)
            else:
                # Lag drift series
                if len(drift_series[:-lag]) > 10 and len(accuracy_series[lag:]) > 10:
                    corr, p_value = stats.pearsonr(drift_series[:-lag], accuracy_series[lag:])
                else:
                    continue
            
            lag_results["lag_correlations"][lag] = {
                "correlation": float(corr),
                "p_value": float(p_value),
                "significant": p_value < 0.05
            }
            
            # Track optimal lag (highest absolute correlation)
            if abs(corr) > abs(lag_results["max_correlation"]):
                lag_results["max_correlation"] = float(corr)
                lag_results["optimal_lag"] = lag
        
        # Interpretation
        if lag_results["optimal_lag"] > 0:
            lag_results["interpretation"] = (
                f"Drift changes lead accuracy changes by {lag_results['optimal_lag']} time periods"
            )
        else:
            lag_results["interpretation"] = "Drift and accuracy changes occur simultaneously"
        
        return lag_results
    
    def _analyze_drift_causality(self, drift_series: List[float], 
                                accuracy_series: List[float]) -> Dict[str, Any]:
        """Analyze potential causal relationship between drift and accuracy"""
        try:
            # Simple causality analysis using Granger causality concept
            # (simplified version without full statistical test)
            
            causality_results = {
                "drift_causes_accuracy_change": {},
                "accuracy_causes_drift_change": {},
                "bidirectional_relationship": False,
                "relationship_strength": "none"
            }
            
            if len(drift_series) < 20:
                return {"status": "insufficient_data_for_causality"}
            
            # Check if drift predicts accuracy
            # Simple approach: compare prediction with and without drift
            from sklearn.metrics import r2_score
            
            # Prepare lagged data
            X_drift = np.array(drift_series[:-1]).reshape(-1, 1)
            y_accuracy = np.array(accuracy_series[1:])
            
            # Model 1: Predict accuracy from previous accuracy only
            X_accuracy_only = np.array(accuracy_series[:-1]).reshape(-1, 1)
            model1 = LinearRegression()
            model1.fit(X_accuracy_only, y_accuracy)
            pred1 = model1.predict(X_accuracy_only)
            r2_1 = r2_score(y_accuracy, pred1)
            
            # Model 2: Predict accuracy from previous accuracy and drift
            X_combined = np.column_stack([X_accuracy_only.flatten(), X_drift.flatten()])
            model2 = LinearRegression()
            model2.fit(X_combined, y_accuracy)
            pred2 = model2.predict(X_combined)
            r2_2 = r2_score(y_accuracy, pred2)
            
            # Improvement in prediction
            improvement = r2_2 - r2_1
            
            causality_results["drift_causes_accuracy_change"] = {
                "r2_without_drift": float(r2_1),
                "r2_with_drift": float(r2_2),
                "improvement": float(improvement),
                "significant": improvement > 0.05,
                "drift_coefficient": float(model2.coef_[1])
            }
            
            # Determine relationship strength
            if improvement > 0.2:
                causality_results["relationship_strength"] = "strong"
            elif improvement > 0.1:
                causality_results["relationship_strength"] = "moderate"
            elif improvement > 0.05:
                causality_results["relationship_strength"] = "weak"
            else:
                causality_results["relationship_strength"] = "negligible"
            
            return causality_results
            
        except Exception as e:
            self.logger.error(f"Error in causality analysis: {e}")
            return {"status": "error", "error": str(e)}
    
    def _create_impact_timeline(self, drift_series: List[float], 
                               accuracy_series: List[float],
                               timestamps: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Create timeline showing drift impact on accuracy"""
        timeline = {
            "critical_periods": [],
            "impact_events": [],
            "trend": "stable"
        }
        
        if len(drift_series) < 5:
            return timeline
        
        # Identify critical periods (high drift + low accuracy)
        drift_threshold = np.percentile(drift_series, 75)  # Top 25% drift
        accuracy_threshold = np.percentile(accuracy_series, 25)  # Bottom 25% accuracy
        
        for i in range(len(drift_series)):
            if drift_series[i] > drift_threshold and accuracy_series[i] < accuracy_threshold:
                period = {
                    "index": i,
                    "drift_score": float(drift_series[i]),
                    "accuracy": float(accuracy_series[i]),
                    "severity": "high"
                }
                
                if timestamps and i < len(timestamps):
                    period["timestamp"] = str(timestamps[i])
                
                timeline["critical_periods"].append(period)
        
        # Identify impact events (sudden changes)
        if len(drift_series) > 1:
            drift_changes = np.diff(drift_series)
            accuracy_changes = np.diff(accuracy_series)
            
            for i in range(1, len(drift_changes)):
                # Significant drift increase followed by accuracy decrease
                if (drift_changes[i] > np.std(drift_changes) * 2 and 
                    accuracy_changes[i] < -np.std(accuracy_changes)):
                    
                    event = {
                        "index": i,
                        "event_type": "drift_induced_degradation",
                        "drift_change": float(drift_changes[i]),
                        "accuracy_change": float(accuracy_changes[i])
                    }
                    
                    if timestamps and i < len(timestamps):
                        event["timestamp"] = str(timestamps[i])
                    
                    timeline["impact_events"].append(event)
        
        # Determine overall trend
        if len(drift_series) > 10:
            drift_trend = np.polyfit(range(len(drift_series)), drift_series, 1)[0]
            accuracy_trend = np.polyfit(range(len(accuracy_series)), accuracy_series, 1)[0]
            
            if drift_trend > 0.01 and accuracy_trend < -0.01:
                timeline["trend"] = "deteriorating"
            elif drift_trend < -0.01 and accuracy_trend > 0.01:
                timeline["trend"] = "improving"
            else:
                timeline["trend"] = "stable"
        
        return timeline
    
    def _analyze_feature_drift_impact(self, drift_data: Dict[str, Any], 
                                     accuracy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze impact of individual feature drift on accuracy"""
        feature_impact = {
            "high_impact_features": [],
            "feature_rankings": {},
            "interaction_effects": {}
        }
        
        feature_scores = {}
        
        # Analyze each feature's drift
        for feature, feature_drift in drift_data.get("features", {}).items():
            if feature_drift.get("drift_detected", False):
                # Calculate impact score based on drift magnitude and type
                impact_score = 0
                
                # Drift magnitude contribution
                if "drift_scores" in feature_drift:
                    avg_drift = np.mean(feature_drift["drift_scores"])
                    impact_score += avg_drift * 0.5
                
                # Drift type contribution
                drift_type = feature_drift.get("drift_type", "unknown")
                if drift_type == "sudden":
                    impact_score *= 1.5
                elif drift_type == "gradual":
                    impact_score *= 1.0
                
                feature_scores[feature] = {
                    "impact_score": float(impact_score),
                    "drift_type": drift_type,
                    "drift_detected": True
                }
                
                # High impact features
                if impact_score > 0.7:
                    feature_impact["high_impact_features"].append({
                        "feature": feature,
                        "impact_score": float(impact_score),
                        "drift_type": drift_type
                    })
        
        # Rank features by impact
        ranked_features = sorted(feature_scores.items(), 
                               key=lambda x: x[1]["impact_score"], 
                               reverse=True)
        
        for rank, (feature, scores) in enumerate(ranked_features):
            feature_impact["feature_rankings"][feature] = {
                "rank": rank + 1,
                "impact_score": scores["impact_score"],
                "impact_level": self._interpret_feature_impact(scores["impact_score"])
            }
        
        # Analyze feature interactions (simplified)
        if len(feature_scores) >= 2:
            # Check for features that drift together
            high_impact_pairs = []
            features_list = list(feature_scores.keys())
            
            for i in range(len(features_list)):
                for j in range(i + 1, len(features_list)):
                    feat1, feat2 = features_list[i], features_list[j]
                    
                    # Both features have high impact
                    if (feature_scores[feat1]["impact_score"] > 0.5 and 
                        feature_scores[feat2]["impact_score"] > 0.5):
                        
                        interaction_score = (feature_scores[feat1]["impact_score"] + 
                                           feature_scores[feat2]["impact_score"]) / 2
                        
                        high_impact_pairs.append({
                            "features": [feat1, feat2],
                            "interaction_score": float(interaction_score),
                            "interaction_type": "compound_drift"
                        })
            
            feature_impact["interaction_effects"]["high_impact_pairs"] = high_impact_pairs[:5]  # Top 5
        
        return feature_impact
    
    def _perform_threshold_analysis(self, drift_metrics: Dict[str, Any], 
                                   thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze drift metrics against configured thresholds"""
        # Default thresholds if not provided
        default_thresholds = {
            "psi": {"warning": 0.1, "critical": 0.25},
            "kl_divergence": {"warning": 0.5, "critical": 1.0},
            "wasserstein_distance": {"warning": 0.1, "critical": 0.2},
            "js_divergence": {"warning": 0.3, "critical": 0.6}
        }
        
        # Merge with provided thresholds
        for metric, levels in default_thresholds.items():
            if metric not in thresholds:
                thresholds[metric] = levels
        
        threshold_results = {
            "violations": [],
            "status_by_metric": {},
            "overall_status": "normal"
        }
        
        severity_scores = {"normal": 0, "warning": 1, "critical": 2}
        max_severity = "normal"
        
        # Check each metric against thresholds
        for metric_name, metric_data in drift_metrics.items():
            if metric_name in thresholds and isinstance(metric_data, dict):
                metric_thresholds = thresholds[metric_name]
                
                # Get the overall metric value
                metric_value = None
                if f"overall_{metric_name}" in metric_data:
                    metric_value = metric_data[f"overall_{metric_name}"]
                elif "overall" in metric_data:
                    metric_value = metric_data["overall"]
                
                if metric_value is not None:
                    status = "normal"
                    
                    if metric_value >= metric_thresholds.get("critical", float('inf')):
                        status = "critical"
                        threshold_results["violations"].append({
                            "metric": metric_name,
                            "value": float(metric_value),
                            "threshold": metric_thresholds["critical"],
                            "severity": "critical"
                        })
                    elif metric_value >= metric_thresholds.get("warning", float('inf')):
                        status = "warning"
                        threshold_results["violations"].append({
                            "metric": metric_name,
                            "value": float(metric_value),
                            "threshold": metric_thresholds["warning"],
                            "severity": "warning"
                        })
                    
                    threshold_results["status_by_metric"][metric_name] = {
                        "status": status,
                        "value": float(metric_value),
                        "thresholds": metric_thresholds
                    }
                    
                    # Update overall status
                    if severity_scores[status] > severity_scores[max_severity]:
                        max_severity = status
        
        threshold_results["overall_status"] = max_severity
        threshold_results["num_violations"] = len(threshold_results["violations"])
        
        return threshold_results
    
    def _analyze_drift_impact_trends(self, drift_metrics: Dict[str, Any], 
                                    correlation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in drift impact over time"""
        trend_analysis = {
            "drift_trend": "stable",
            "impact_trend": "stable",
            "acceleration": "none",
            "forecast": {}
        }
        
        # Analyze drift metric trends
        drift_values = []
        for metric_data in drift_metrics.values():
            if isinstance(metric_data, dict) and "overall" in metric_data:
                drift_values.append(metric_data["overall"])
        
        if drift_values and len(drift_values) > 1:
            # Simple trend detection
            drift_change = drift_values[-1] - drift_values[0]
            
            if drift_change > 0.1:
                trend_analysis["drift_trend"] = "increasing"
            elif drift_change < -0.1:
                trend_analysis["drift_trend"] = "decreasing"
            
            # Acceleration (is the rate of change increasing?)
            if len(drift_values) >= 3:
                first_half_change = drift_values[len(drift_values)//2] - drift_values[0]
                second_half_change = drift_values[-1] - drift_values[len(drift_values)//2]
                
                if second_half_change > first_half_change * 1.5:
                    trend_analysis["acceleration"] = "accelerating"
                elif second_half_change < first_half_change * 0.5:
                    trend_analysis["acceleration"] = "decelerating"
        
        # Analyze impact trend from correlation data
        if correlation_data and "impact_timeline" in correlation_data:
            timeline = correlation_data["impact_timeline"]
            
            if timeline.get("trend"):
                if timeline["trend"] == "deteriorating":
                    trend_analysis["impact_trend"] = "worsening"
                elif timeline["trend"] == "improving":
                    trend_analysis["impact_trend"] = "improving"
        
        # Simple forecast
        if trend_analysis["drift_trend"] == "increasing" and trend_analysis["impact_trend"] == "worsening":
            trend_analysis["forecast"] = {
                "outlook": "negative",
                "recommendation": "Immediate intervention required to address increasing drift"
            }
        elif trend_analysis["drift_trend"] == "decreasing" and trend_analysis["impact_trend"] == "improving":
            trend_analysis["forecast"] = {
                "outlook": "positive",
                "recommendation": "Continue current drift mitigation strategies"
            }
        else:
            trend_analysis["forecast"] = {
                "outlook": "neutral",
                "recommendation": "Monitor drift metrics closely for changes"
            }
        
        return trend_analysis
    
    def _calculate_overall_drift_impact(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall drift impact score and assessment"""
        overall_impact = {
            "impact_score": 0.0,
            "impact_level": "low",
            "confidence": 0.0,
            "contributing_factors": []
        }
        
        impact_components = []
        weights = {
            "drift_magnitude": 0.3,
            "accuracy_correlation": 0.3,
            "threshold_violations": 0.2,
            "feature_impact": 0.2
        }
        
        # Drift magnitude component
        if "drift_metrics" in analysis:
            drift_scores = []
            for metric_data in analysis["drift_metrics"].values():
                if isinstance(metric_data, dict) and "overall" in metric_data:
                    # Normalize different metrics
                    metric_name = next((k for k in metric_data.keys() if k.startswith("overall_")), "")
                    
                    if "psi" in metric_name:
                        normalized = min(metric_data["overall"] / 0.5, 1.0)
                    elif "kl" in metric_name:
                        normalized = min(metric_data["overall"] / 1.0, 1.0)
                    elif "wasserstein" in metric_name:
                        normalized = min(metric_data["overall"] / 0.2, 1.0)
                    else:
                        normalized = metric_data["overall"]
                    
                    drift_scores.append(normalized)
            
            if drift_scores:
                drift_magnitude = np.mean(drift_scores)
                impact_components.append(("drift_magnitude", drift_magnitude, weights["drift_magnitude"]))
                
                if drift_magnitude > 0.7:
                    overall_impact["contributing_factors"].append("High drift magnitude detected")
        
        # Accuracy correlation component
        if "impact_correlation" in analysis and "drift_accuracy_correlation" in analysis["impact_correlation"]:
            corr_data = analysis["impact_correlation"]["drift_accuracy_correlation"]
            if "pearson" in corr_data:
                correlation_strength = abs(corr_data["pearson"]["coefficient"])
                impact_components.append(("accuracy_correlation", correlation_strength, weights["accuracy_correlation"]))
                
                if correlation_strength > 0.5:
                    overall_impact["contributing_factors"].append("Strong drift-accuracy correlation")
        
        # Threshold violations component
        if "threshold_analysis" in analysis:
            threshold_data = analysis["threshold_analysis"]
            if threshold_data.get("overall_status") == "critical":
                violation_score = 1.0
            elif threshold_data.get("overall_status") == "warning":
                violation_score = 0.5
            else:
                violation_score = 0.0
            
            impact_components.append(("threshold_violations", violation_score, weights["threshold_violations"]))
            
            if violation_score > 0:
                overall_impact["contributing_factors"].append(
                    f"{threshold_data.get('num_violations', 0)} threshold violations detected"
                )
        
        # Feature impact component
        if "feature_impact" in analysis:
            feature_data = analysis["feature_impact"]
            high_impact_count = len(feature_data.get("high_impact_features", []))
            total_features = len(feature_data.get("feature_rankings", {}))
            
            if total_features > 0:
                feature_impact_score = min(high_impact_count / max(total_features * 0.2, 1), 1.0)
                impact_components.append(("feature_impact", feature_impact_score, weights["feature_impact"]))
                
                if high_impact_count > 0:
                    overall_impact["contributing_factors"].append(
                        f"{high_impact_count} high-impact features identified"
                    )
        
        # Calculate weighted impact score
        if impact_components:
            total_weight = sum(weight for _, _, weight in impact_components)
            overall_impact["impact_score"] = sum(
                score * weight for _, score, weight in impact_components
            ) / total_weight
            
            # Confidence based on available components
            overall_impact["confidence"] = total_weight
        
        # Determine impact level
        score = overall_impact["impact_score"]
        if score >= 0.8:
            overall_impact["impact_level"] = "critical"
        elif score >= 0.6:
            overall_impact["impact_level"] = "high"
        elif score >= 0.4:
            overall_impact["impact_level"] = "moderate"
        elif score >= 0.2:
            overall_impact["impact_level"] = "low"
        else:
            overall_impact["impact_level"] = "minimal"
        
        return overall_impact
    
    # Helper interpretation methods
    
    def _interpret_psi_score(self, psi: float) -> str:
        """Interpret PSI score"""
        if psi < 0.1:
            return "stable"
        elif psi < 0.2:
            return "slight_change"
        elif psi < 0.5:
            return "moderate_change"
        else:
            return "significant_change"
    
    def _interpret_kl_divergence(self, kl: float) -> str:
        """Interpret KL divergence"""
        if kl < 0.1:
            return "negligible"
        elif kl < 0.5:
            return "low"
        elif kl < 1.0:
            return "moderate"
        else:
            return "high"
    
    def _interpret_wasserstein_distance(self, distance: float) -> str:
        """Interpret Wasserstein distance"""
        if distance < 0.05:
            return "very_similar"
        elif distance < 0.1:
            return "similar"
        elif distance < 0.2:
            return "moderate_difference"
        else:
            return "significant_difference"
    
    def _interpret_js_divergence(self, js: float) -> str:
        """Interpret JS divergence"""
        if js < 0.1:
            return "negligible"
        elif js < 0.3:
            return "low"
        elif js < 0.6:
            return "moderate"
        else:
            return "high"
    
    def _interpret_feature_impact(self, score: float) -> str:
        """Interpret feature impact score"""
        if score >= 0.8:
            return "critical"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "moderate"
        elif score >= 0.2:
            return "low"
        else:
            return "minimal"
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation strength"""
        if correlation >= 0.8:
            return "very_strong"
        elif correlation >= 0.6:
            return "strong"
        elif correlation >= 0.4:
            return "moderate"
        elif correlation >= 0.2:
            return "weak"
        else:
            return "very_weak"