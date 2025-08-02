"""
Stability Analysis Engine for Model Performance Assessment
Specialized algorithms for drift detection, pattern analysis, and stability scoring
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import time
import logging


class StabilityAnalysisEngine:
    """
    Engine for performing complex stability analysis algorithms.
    Handles drift detection, pattern recognition, and stability scoring.
    """
    
    def __init__(self, logger: logging.Logger = None):
        """Initialize the stability analysis engine."""
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    def analyze_variance_patterns(self, values: np.ndarray, metric_name: str) -> Dict[str, Any]:
        """Analyze variance patterns in performance data"""
        # Basic statistics
        mean_value = np.mean(values)
        std_value = np.std(values)
        cv = std_value / mean_value if mean_value != 0 else 0
        
        # Rolling variance analysis
        window_size = min(10, len(values) // 3)
        rolling_vars = pd.Series(values).rolling(window=window_size).var()
        
        # Variance trend
        variance_trend = "stable"
        if len(rolling_vars.dropna()) > 2:
            x = np.arange(len(rolling_vars.dropna()))
            slope, _, r_value, p_value, _ = stats.linregress(x, rolling_vars.dropna())
            if p_value < 0.05:
                variance_trend = "increasing" if slope > 0 else "decreasing"
        
        # Variance stability index (VSI)
        vsi = 1 - cv if cv < 1 else 0
        
        # Homoscedasticity test (Breusch-Pagan test approximation)
        homoscedastic = True
        if len(values) > 20:
            residuals = values - np.mean(values)
            x = np.arange(len(values)).reshape(-1, 1)
            
            # Simple linear regression on squared residuals
            lr = LinearRegression()
            lr.fit(x, residuals**2)
            r_squared = lr.score(x, residuals**2)
            
            # If R² is significant, variance is not constant
            homoscedastic = r_squared < 0.1
        
        return {
            "mean": float(mean_value),
            "std": float(std_value),
            "coefficient_of_variation": float(cv),
            "variance_stability_index": float(vsi),
            "variance_trend": variance_trend,
            "homoscedastic": homoscedastic,
            "rolling_variance": {
                "values": rolling_vars.dropna().tolist(),
                "window_size": window_size,
                "trend": variance_trend
            },
            "interpretation": self._interpret_variance_analysis(cv, variance_trend, homoscedastic)
        }
    
    def detect_performance_drift(self, values: np.ndarray, timestamps=None) -> Dict[str, Any]:
        """Detect drift in model performance"""
        # Split data into reference and test periods
        split_point = len(values) // 2
        reference_data = values[:split_point]
        test_data = values[split_point:]
        
        drift_scores = {}
        
        # 1. Population Stability Index (PSI)
        psi_score = self._calculate_psi(reference_data, test_data)
        drift_scores["psi"] = {
            "score": float(psi_score),
            "threshold": 0.1,
            "drift_detected": psi_score > 0.1,
            "severity": self._interpret_psi(psi_score)
        }
        
        # 2. Kullback-Leibler Divergence
        kl_divergence = self._calculate_kl_divergence(reference_data, test_data)
        drift_scores["kl_divergence"] = {
            "score": float(kl_divergence),
            "threshold": 0.1,
            "drift_detected": kl_divergence > 0.1,
            "severity": self._interpret_kl_divergence(kl_divergence)
        }
        
        # 3. Wasserstein Distance
        wasserstein_dist = stats.wasserstein_distance(reference_data, test_data)
        drift_scores["wasserstein_distance"] = {
            "score": float(wasserstein_dist),
            "threshold": 0.05,
            "drift_detected": wasserstein_dist > 0.05,
            "severity": self._interpret_wasserstein(wasserstein_dist)
        }
        
        # 4. Statistical tests
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.ks_2samp(reference_data, test_data)
        drift_scores["ks_test"] = {
            "statistic": float(ks_stat),
            "p_value": float(ks_p),
            "drift_detected": ks_p < 0.05,
            "significance_level": 0.05
        }
        
        # 5. Sliding window drift detection
        window_drift = self._detect_sliding_window_drift(values)
        
        # Overall drift assessment
        drift_detected = any([
            drift_scores["psi"]["drift_detected"],
            drift_scores["kl_divergence"]["drift_detected"],
            drift_scores["wasserstein_distance"]["drift_detected"],
            drift_scores["ks_test"]["drift_detected"]
        ])
        
        return {
            "drift_scores": drift_scores,
            "sliding_window_analysis": window_drift,
            "overall_drift_detected": drift_detected,
            "drift_severity": self._calculate_overall_drift_severity(drift_scores),
            "reference_period": {
                "start_index": 0,
                "end_index": split_point,
                "mean": float(np.mean(reference_data)),
                "std": float(np.std(reference_data))
            },
            "test_period": {
                "start_index": split_point,
                "end_index": len(values),
                "mean": float(np.mean(test_data)),
                "std": float(np.std(test_data))
            }
        }
    
    def analyze_consistency(self, values: np.ndarray) -> Dict[str, Any]:
        """Analyze performance consistency"""
        # Consistency metrics
        consistency_metrics = {}
        
        # 1. Inter-quartile range consistency
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        iqr_ratio = iqr / np.median(values) if np.median(values) != 0 else 0
        consistency_metrics["iqr_consistency"] = 1 - min(iqr_ratio, 1)
        
        # 2. Successive difference consistency
        diffs = np.diff(values)
        diff_std = np.std(diffs)
        diff_mean = np.mean(np.abs(diffs))
        consistency_metrics["successive_diff_consistency"] = 1 - min(diff_std / (np.mean(values) + 1e-10), 1)
        
        # 3. Autocorrelation consistency
        if len(values) > 20:
            autocorr = pd.Series(values).autocorr(lag=1)
            consistency_metrics["autocorrelation"] = abs(autocorr)
        else:
            consistency_metrics["autocorrelation"] = 0
        
        # 4. Runs test for randomness
        runs_test_result = self._perform_runs_test(values)
        consistency_metrics["runs_test"] = runs_test_result
        
        # 5. Consistency index (composite score)
        consistency_index = np.mean([
            consistency_metrics["iqr_consistency"],
            consistency_metrics["successive_diff_consistency"],
            1 - consistency_metrics["autocorrelation"]  # Lower autocorr = more consistent
        ])
        
        return {
            "consistency_index": float(consistency_index),
            "metrics": consistency_metrics,
            "interpretation": self._interpret_consistency(consistency_index),
            "volatility": {
                "daily": float(diff_std),
                "normalized": float(diff_std / (np.mean(values) + 1e-10))
            }
        }
    
    def analyze_degradation_patterns(self, values: np.ndarray, timestamps=None) -> Dict[str, Any]:
        """Analyze performance degradation patterns"""
        # Linear degradation analysis
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Calculate degradation rate
        degradation_rate = -slope / np.mean(values) if np.mean(values) != 0 else 0
        
        # Segmented degradation analysis
        segments = self._analyze_segmented_degradation(values)
        
        # Acceleration analysis
        if len(values) > 10:
            # Fit polynomial to detect acceleration
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(x.reshape(-1, 1))
            poly_model = LinearRegression()
            poly_model.fit(X_poly, values)
            
            # Second derivative approximation
            acceleration = poly_model.coef_[2] * 2 if len(poly_model.coef_) > 2 else 0
        else:
            acceleration = 0
        
        # Time to threshold analysis
        if slope < 0:
            current_value = values[-1]
            threshold = 0.8 * values[0]  # 80% of initial performance
            if slope != 0:
                time_to_threshold = max(0, (threshold - current_value) / (-slope))
            else:
                time_to_threshold = float('inf')
        else:
            time_to_threshold = float('inf')
        
        return {
            "degradation_rate": float(degradation_rate),
            "linear_trend": {
                "slope": float(slope),
                "r_squared": float(r_value**2),
                "p_value": float(p_value),
                "significant": p_value < 0.05
            },
            "acceleration": float(acceleration),
            "is_accelerating": acceleration < -0.0001,
            "segments": segments,
            "time_to_threshold": float(time_to_threshold) if time_to_threshold != float('inf') else None,
            "degradation_type": self._classify_degradation_pattern(slope, acceleration, segments),
            "severity": self._assess_degradation_severity(degradation_rate, acceleration)
        }
    
    def identify_stability_patterns(self, values: np.ndarray, variance_result: Dict[str, Any],
                                   drift_result: Dict[str, Any], degradation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Identify specific stability patterns"""
        patterns = {
            "identified_patterns": [],
            "pattern_details": {},
            "stability_classification": "unknown"
        }
        
        # 1. Check for sudden drops
        sudden_drops = self._detect_sudden_drops(values)
        if sudden_drops["detected"]:
            patterns["identified_patterns"].append("sudden_drops")
            patterns["pattern_details"]["sudden_drops"] = sudden_drops
        
        # 2. Check for cyclic instability
        cyclic_pattern = self._detect_cyclic_instability(values)
        if cyclic_pattern["detected"]:
            patterns["identified_patterns"].append("cyclic_instability")
            patterns["pattern_details"]["cyclic_instability"] = cyclic_pattern
        
        # 3. Check for gradual degradation
        if degradation_result["degradation_rate"] > 0.001 and degradation_result["linear_trend"]["significant"]:
            patterns["identified_patterns"].append("gradual_degradation")
            patterns["pattern_details"]["gradual_degradation"] = {
                "rate": degradation_result["degradation_rate"],
                "confidence": degradation_result["linear_trend"]["r_squared"]
            }
        
        # 4. Check for increasing variance
        if variance_result["variance_trend"] == "increasing":
            patterns["identified_patterns"].append("increasing_variance")
            patterns["pattern_details"]["increasing_variance"] = {
                "trend": variance_result["variance_trend"],
                "cv": variance_result["coefficient_of_variation"]
            }
        
        # 5. Check for distribution shift
        if drift_result["overall_drift_detected"]:
            patterns["identified_patterns"].append("distribution_shift")
            patterns["pattern_details"]["distribution_shift"] = {
                "severity": drift_result["drift_severity"],
                "primary_metric": max(drift_result["drift_scores"].items(), 
                                    key=lambda x: x[1].get("score", 0))[0]
            }
        
        # Classify overall stability
        patterns["stability_classification"] = self._classify_stability(patterns["identified_patterns"])
        
        return patterns
    
    def calculate_stability_score(self, variance_result: Dict[str, Any], drift_result: Dict[str, Any],
                                 consistency_result: Dict[str, Any], degradation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive stability score"""
        # Component scores (0-1, higher is more stable)
        scores = {
            "variance_score": variance_result["variance_stability_index"],
            "drift_score": 1 - min(drift_result["drift_severity"], 1),
            "consistency_score": consistency_result["consistency_index"],
            "degradation_score": 1 - min(abs(degradation_result["degradation_rate"]) * 10, 1)
        }
        
        # Weights for different components
        weights = {
            "variance_score": 0.25,
            "drift_score": 0.3,
            "consistency_score": 0.25,
            "degradation_score": 0.2
        }
        
        # Calculate weighted score
        overall_score = sum(scores[k] * weights[k] for k in scores)
        
        # Determine stability level
        if overall_score >= 0.8:
            stability_level = "high"
        elif overall_score >= 0.6:
            stability_level = "moderate"
        elif overall_score >= 0.4:
            stability_level = "low"
        else:
            stability_level = "critical"
        
        return {
            "overall_score": float(overall_score),
            "component_scores": {k: float(v) for k, v in scores.items()},
            "weights": weights,
            "stability_level": stability_level,
            "confidence": self._calculate_stability_confidence(variance_result, consistency_result)
        }
    
    # Private helper methods
    
    def _calculate_psi(self, reference: np.ndarray, test: np.ndarray) -> float:
        """Calculate Population Stability Index"""
        # Create bins based on reference data
        min_val = min(reference.min(), test.min())
        max_val = max(reference.max(), test.max())
        bins = np.linspace(min_val, max_val, 11)
        
        # Calculate distributions
        ref_hist, _ = np.histogram(reference, bins=bins)
        test_hist, _ = np.histogram(test, bins=bins)
        
        # Normalize
        ref_hist = ref_hist / len(reference)
        test_hist = test_hist / len(test)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        ref_hist = ref_hist + epsilon
        test_hist = test_hist + epsilon
        
        # Calculate PSI
        psi = np.sum((test_hist - ref_hist) * np.log(test_hist / ref_hist))
        
        return psi
    
    def _calculate_kl_divergence(self, reference: np.ndarray, test: np.ndarray) -> float:
        """Calculate Kullback-Leibler divergence"""
        # Create probability distributions
        min_val = min(reference.min(), test.min())
        max_val = max(reference.max(), test.max())
        bins = np.linspace(min_val, max_val, 20)
        
        ref_hist, _ = np.histogram(reference, bins=bins, density=True)
        test_hist, _ = np.histogram(test, bins=bins, density=True)
        
        # Normalize to probabilities
        ref_hist = ref_hist / ref_hist.sum()
        test_hist = test_hist / test_hist.sum()
        
        # Add epsilon to avoid log(0)
        epsilon = 1e-10
        ref_hist = ref_hist + epsilon
        test_hist = test_hist + epsilon
        
        # Calculate KL divergence
        kl_div = stats.entropy(ref_hist, test_hist)
        
        return kl_div
    
    def _detect_sliding_window_drift(self, values: np.ndarray) -> Dict[str, Any]:
        """Detect drift using sliding window approach"""
        window_size = max(10, len(values) // 10)
        stride = max(1, window_size // 2)
        
        drift_points = []
        drift_scores = []
        
        for i in range(0, len(values) - 2 * window_size, stride):
            window1 = values[i:i + window_size]
            window2 = values[i + window_size:i + 2 * window_size]
            
            # Calculate drift score
            ks_stat, ks_p = stats.ks_2samp(window1, window2)
            
            if ks_p < 0.05:
                drift_points.append(i + window_size)
                drift_scores.append(ks_stat)
        
        return {
            "window_size": window_size,
            "drift_points": drift_points,
            "drift_scores": drift_scores,
            "num_drift_events": len(drift_points),
            "drift_frequency": len(drift_points) / (len(values) / window_size) if len(values) > 0 else 0
        }
    
    def _perform_runs_test(self, values: np.ndarray) -> Dict[str, Any]:
        """Perform runs test for randomness"""
        median = np.median(values)
        binary = values > median
        
        # Count runs
        runs = 1
        for i in range(1, len(binary)):
            if binary[i] != binary[i-1]:
                runs += 1
        
        # Expected runs and variance
        n1 = np.sum(binary)
        n2 = len(binary) - n1
        
        if n1 == 0 or n2 == 0:
            return {"randomness": 1.0, "runs": runs, "expected_runs": 0}
        
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))
        
        if variance > 0:
            z_score = (runs - expected_runs) / np.sqrt(variance)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            p_value = 1.0
        
        return {
            "runs": runs,
            "expected_runs": float(expected_runs),
            "p_value": float(p_value),
            "randomness": float(p_value),  # Higher p-value = more random = less predictable pattern
            "is_random": p_value > 0.05
        }
    
    def _analyze_segmented_degradation(self, values: np.ndarray) -> Dict[str, Any]:
        """Analyze degradation in segments"""
        num_segments = min(5, len(values) // 10)
        segment_size = len(values) // num_segments
        
        segments = []
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < num_segments - 1 else len(values)
            
            segment_values = values[start_idx:end_idx]
            x = np.arange(len(segment_values))
            
            if len(segment_values) > 2:
                slope, _, r_value, p_value, _ = stats.linregress(x, segment_values)
                
                segments.append({
                    "segment": i,
                    "start_index": start_idx,
                    "end_index": end_idx,
                    "mean": float(np.mean(segment_values)),
                    "slope": float(slope),
                    "degradation_rate": float(-slope / np.mean(segment_values)) if np.mean(segment_values) != 0 else 0,
                    "significant": p_value < 0.05
                })
        
        # Analyze degradation acceleration across segments
        if len(segments) > 1:
            degradation_rates = [s["degradation_rate"] for s in segments]
            acceleration_trend = "constant"
            
            if len(degradation_rates) > 2:
                x = np.arange(len(degradation_rates))
                slope, _, _, p_value, _ = stats.linregress(x, degradation_rates)
                
                if p_value < 0.05:
                    acceleration_trend = "accelerating" if slope > 0 else "decelerating"
        else:
            acceleration_trend = "unknown"
        
        return {
            "segments": segments,
            "num_segments": num_segments,
            "acceleration_trend": acceleration_trend
        }
    
    def _detect_sudden_drops(self, values: np.ndarray) -> Dict[str, Any]:
        """Detect sudden performance drops"""
        # Calculate differences
        diffs = np.diff(values)
        
        # Detect significant drops (more than 2 std from mean)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        threshold = mean_diff - 2 * std_diff
        
        drops = []
        for i, diff in enumerate(diffs):
            if diff < threshold:
                drops.append({
                    "index": i + 1,
                    "magnitude": float(diff),
                    "from_value": float(values[i]),
                    "to_value": float(values[i + 1]),
                    "percentage_drop": float((values[i] - values[i + 1]) / values[i] * 100) if values[i] != 0 else 0
                })
        
        return {
            "detected": len(drops) > 0,
            "num_drops": len(drops),
            "drops": drops,
            "max_drop": min(diffs) if len(diffs) > 0 else 0,
            "threshold_used": float(threshold)
        }
    
    def _detect_cyclic_instability(self, values: np.ndarray) -> Dict[str, Any]:
        """Detect cyclic patterns in stability"""
        if len(values) < 20:
            return {"detected": False, "message": "Insufficient data for cyclic analysis"}
        
        # Perform FFT to detect dominant frequencies
        fft_values = np.fft.fft(values - np.mean(values))
        frequencies = np.fft.fftfreq(len(values))
        
        # Get power spectrum
        power = np.abs(fft_values)**2
        
        # Find dominant frequencies (excluding DC component)
        positive_freq_idx = frequencies > 0
        positive_freqs = frequencies[positive_freq_idx]
        positive_power = power[positive_freq_idx]
        
        if len(positive_power) == 0:
            return {"detected": False, "message": "No positive frequencies found"}
        
        # Find peaks in power spectrum
        peak_threshold = np.mean(positive_power) + 2 * np.std(positive_power)
        peak_indices = positive_power > peak_threshold
        
        if np.any(peak_indices):
            dominant_freq_idx = np.argmax(positive_power)
            dominant_freq = positive_freqs[dominant_freq_idx]
            dominant_period = 1 / dominant_freq if dominant_freq > 0 else len(values)
            
            # Calculate cyclic strength
            cyclic_strength = positive_power[dominant_freq_idx] / np.sum(positive_power)
            
            return {
                "detected": True,
                "dominant_period": float(dominant_period),
                "cyclic_strength": float(cyclic_strength),
                "num_cycles": int(len(values) / dominant_period),
                "interpretation": self._interpret_cyclic_pattern(dominant_period, cyclic_strength)
            }
        
        return {"detected": False, "message": "No significant cyclic patterns found"}
    
    # Interpretation helper methods
    
    def _interpret_variance_analysis(self, cv: float, trend: str, homoscedastic: bool) -> str:
        """Interpret variance analysis results"""
        interpretations = []
        
        if cv < 0.1:
            interpretations.append("very low variance")
        elif cv < 0.2:
            interpretations.append("low variance")
        elif cv < 0.3:
            interpretations.append("moderate variance")
        else:
            interpretations.append("high variance")
        
        if trend != "stable":
            interpretations.append(f"{trend} variance trend")
        
        if not homoscedastic:
            interpretations.append("heteroscedastic (non-constant variance)")
        
        return "; ".join(interpretations)
    
    def _interpret_psi(self, psi: float) -> str:
        """Interpret PSI score"""
        if psi < 0.1:
            return "negligible"
        elif psi < 0.2:
            return "moderate"
        else:
            return "significant"
    
    def _interpret_kl_divergence(self, kl: float) -> str:
        """Interpret KL divergence"""
        if kl < 0.05:
            return "negligible"
        elif kl < 0.1:
            return "low"
        elif kl < 0.2:
            return "moderate"
        else:
            return "high"
    
    def _interpret_wasserstein(self, distance: float) -> str:
        """Interpret Wasserstein distance"""
        if distance < 0.01:
            return "negligible"
        elif distance < 0.05:
            return "low"
        elif distance < 0.1:
            return "moderate"
        else:
            return "high"
    
    def _interpret_consistency(self, consistency_index: float) -> str:
        """Interpret consistency index"""
        if consistency_index >= 0.8:
            return "highly consistent"
        elif consistency_index >= 0.6:
            return "moderately consistent"
        elif consistency_index >= 0.4:
            return "somewhat inconsistent"
        else:
            return "highly inconsistent"
    
    def _interpret_cyclic_pattern(self, period: float, strength: float) -> str:
        """Interpret cyclic pattern characteristics"""
        period_desc = ""
        if period < 7:
            period_desc = "short-term"
        elif period < 30:
            period_desc = "weekly"
        elif period < 90:
            period_desc = "monthly"
        else:
            period_desc = "long-term"
        
        strength_desc = ""
        if strength < 0.1:
            strength_desc = "weak"
        elif strength < 0.3:
            strength_desc = "moderate"
        else:
            strength_desc = "strong"
        
        return f"{strength_desc} {period_desc} cyclic pattern (period: {period:.1f})"
    
    def _classify_degradation_pattern(self, slope: float, acceleration: float, segments: Dict[str, Any]) -> str:
        """Classify the type of degradation pattern"""
        if abs(slope) < 0.0001:
            return "stable"
        elif slope > 0:
            return "improving"
        elif acceleration < -0.0001:
            return "accelerating_degradation"
        elif segments["acceleration_trend"] == "accelerating":
            return "progressive_degradation"
        else:
            return "linear_degradation"
    
    def _assess_degradation_severity(self, degradation_rate: float, acceleration: float) -> str:
        """Assess severity of degradation"""
        if degradation_rate < 0.001:
            return "negligible"
        elif degradation_rate < 0.005:
            return "low"
        elif degradation_rate < 0.01:
            return "moderate"
        elif degradation_rate < 0.05 or acceleration < -0.0001:
            return "high"
        else:
            return "critical"
    
    def _classify_stability(self, patterns: List[str]) -> str:
        """Classify overall stability based on identified patterns"""
        if not patterns:
            return "stable"
        elif "sudden_drops" in patterns or "distribution_shift" in patterns:
            return "unstable"
        elif "gradual_degradation" in patterns or "increasing_variance" in patterns:
            return "deteriorating"
        elif "cyclic_instability" in patterns:
            return "cyclic"
        else:
            return "marginal"
    
    def _calculate_overall_drift_severity(self, drift_scores: Dict[str, Any]) -> float:
        """Calculate overall drift severity from multiple metrics"""
        severities = []
        
        # Map severity strings to numeric values
        severity_map = {"negligible": 0.1, "low": 0.3, "moderate": 0.6, "significant": 0.8, "high": 0.9}
        
        for metric, result in drift_scores.items():
            if "severity" in result:
                severities.append(severity_map.get(result["severity"], 0.5))
            elif "score" in result:
                # Normalize scores to 0-1 range
                if metric == "psi":
                    severities.append(min(result["score"] / 0.3, 1))
                elif metric == "kl_divergence":
                    severities.append(min(result["score"] / 0.3, 1))
                elif metric == "wasserstein_distance":
                    severities.append(min(result["score"] / 0.2, 1))
        
        return np.mean(severities) if severities else 0.0
    
    def _calculate_stability_confidence(self, variance_result: Dict[str, Any], 
                                      consistency_result: Dict[str, Any]) -> float:
        """Calculate confidence in stability assessment"""
        # Factors affecting confidence:
        # 1. Low variance indicates reliable measurements
        # 2. High consistency indicates predictable behavior
        # 3. Homoscedasticity indicates stable variance
        
        confidence_factors = []
        
        # Variance factor
        cv = variance_result.get("coefficient_of_variation", 1)
        variance_confidence = 1 - min(cv, 1)
        confidence_factors.append(variance_confidence)
        
        # Consistency factor
        consistency_index = consistency_result.get("consistency_index", 0)
        confidence_factors.append(consistency_index)
        
        # Homoscedasticity factor
        if variance_result.get("homoscedastic", False):
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        return float(np.mean(confidence_factors))