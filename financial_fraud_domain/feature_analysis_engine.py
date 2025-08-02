"""
Feature Analysis Engine for Accuracy Impact Assessment
Advanced feature impact analysis using SHAP, permutation importance, and correlation methods
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
import time
import threading
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Feature analysis imports with fallbacks
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Feature importance analysis will use fallback methods.")

try:
    import pingouin
    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False
    warnings.warn("Pingouin not available. Partial correlation analysis will be skipped.")

from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.cluster import AgglomerativeClustering


class FeatureAnalysisEngine:
    """
    Engine for performing comprehensive feature impact analysis.
    Supports SHAP, permutation importance, correlation, and interaction analysis.
    """
    
    def __init__(self, logger: logging.Logger = None):
        """Initialize the feature analysis engine."""
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._lock = threading.Lock()
    
    def analyze_feature_impact(self, feature_data: Dict[str, Any], 
                              accuracy_data: Dict[str, Any], 
                              impact_analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive feature impact analysis"""
        with self._lock:
            # Prepare data
            prepared_data = self._prepare_feature_impact_data(feature_data, accuracy_data)
            
            # Get requested methods
            methods = impact_analysis_config.get("methods", ["shap", "permutation_importance", "correlation", "mutual_information"])
            
            # Initialize results structure
            analysis_results = {
                "shap_analysis": None,
                "permutation_importance": None,
                "correlation_analysis": None,
                "mutual_information": None,
                "interaction_effects": None
            }
            
            # Perform each analysis
            if "shap" in methods:
                try:
                    analysis_results["shap_analysis"] = self._perform_shap_analysis(prepared_data, impact_analysis_config)
                except Exception as e:
                    self.logger.warning(f"SHAP analysis failed: {e}")
                    analysis_results["shap_analysis"] = {"status": "failed", "error": str(e)}
            
            if "permutation_importance" in methods:
                try:
                    analysis_results["permutation_importance"] = self._perform_permutation_importance(prepared_data, impact_analysis_config)
                except Exception as e:
                    self.logger.warning(f"Permutation importance failed: {e}")
                    analysis_results["permutation_importance"] = {"status": "failed", "error": str(e)}
            
            if "correlation" in methods:
                try:
                    analysis_results["correlation_analysis"] = self._perform_feature_correlation_analysis(prepared_data)
                except Exception as e:
                    self.logger.warning(f"Correlation analysis failed: {e}")
                    analysis_results["correlation_analysis"] = {"status": "failed", "error": str(e)}
            
            if "mutual_information" in methods:
                try:
                    analysis_results["mutual_information"] = self._perform_mutual_information_analysis(prepared_data)
                except Exception as e:
                    self.logger.warning(f"Mutual information analysis failed: {e}")
                    analysis_results["mutual_information"] = {"status": "failed", "error": str(e)}
            
            # Feature interaction analysis
            if impact_analysis_config.get("analyze_interactions", True):
                try:
                    analysis_results["interaction_effects"] = self._analyze_feature_interactions(prepared_data, analysis_results)
                except Exception as e:
                    self.logger.warning(f"Interaction analysis failed: {e}")
                    analysis_results["interaction_effects"] = {"status": "failed", "error": str(e)}
            
            return {
                "analysis_results": analysis_results,
                "prepared_data_info": {
                    "n_samples": prepared_data["n_samples"],
                    "n_features": prepared_data["n_features"],
                    "feature_names": prepared_data["feature_names"]
                }
            }
    
    def _prepare_feature_impact_data(self, feature_data: Dict[str, Any], 
                                    accuracy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for feature impact analysis"""
        prepared = {
            "feature_matrix": None,
            "feature_names": [],
            "accuracy_values": None,
            "n_samples": 0,
            "n_features": 0,
            "model_predictions": None,
            "models": {}
        }
        
        # Handle different feature data formats
        if "feature_matrix" in feature_data:
            prepared["feature_matrix"] = np.array(feature_data["feature_matrix"])
            prepared["feature_names"] = feature_data.get("feature_names", 
                                                         [f"feature_{i}" for i in range(prepared["feature_matrix"].shape[1])])
        elif "features" in feature_data:
            features_dict = feature_data["features"]
            feature_names = list(features_dict.keys())
            
            n_samples = len(next(iter(features_dict.values())))
            feature_matrix = np.zeros((n_samples, len(feature_names)))
            
            for i, (name, values) in enumerate(features_dict.items()):
                feature_matrix[:, i] = np.array(values)
            
            prepared["feature_matrix"] = feature_matrix
            prepared["feature_names"] = feature_names
        
        # Handle accuracy data
        if "values" in accuracy_data:
            prepared["accuracy_values"] = np.array(accuracy_data["values"])
        elif "metrics" in accuracy_data:
            if "accuracy" in accuracy_data["metrics"]:
                prepared["accuracy_values"] = np.array(accuracy_data["metrics"]["accuracy"])
            else:
                first_metric = next(iter(accuracy_data["metrics"].values()))
                prepared["accuracy_values"] = np.array(first_metric)
        
        # Ensure data alignment
        if prepared["feature_matrix"] is not None and prepared["accuracy_values"] is not None:
            n_samples_features = prepared["feature_matrix"].shape[0]
            n_samples_accuracy = len(prepared["accuracy_values"])
            
            if n_samples_features != n_samples_accuracy:
                min_samples = min(n_samples_features, n_samples_accuracy)
                prepared["feature_matrix"] = prepared["feature_matrix"][:min_samples]
                prepared["accuracy_values"] = prepared["accuracy_values"][:min_samples]
        
        # Update metadata
        prepared["n_samples"] = prepared["feature_matrix"].shape[0] if prepared["feature_matrix"] is not None else 0
        prepared["n_features"] = prepared["feature_matrix"].shape[1] if prepared["feature_matrix"] is not None else 0
        
        # Add model information
        if "models" in feature_data:
            prepared["models"] = feature_data["models"]
        elif "model" in feature_data:
            prepared["models"]["default"] = feature_data["model"]
        
        if "predictions" in accuracy_data:
            prepared["model_predictions"] = np.array(accuracy_data["predictions"])
        
        return prepared
    
    def _perform_shap_analysis(self, prepared_data: Dict[str, Any], 
                              config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform SHAP analysis"""
        if not SHAP_AVAILABLE:
            return self._fallback_feature_importance(prepared_data)
        
        shap_results = {
            "method": "shap",
            "feature_importance": {},
            "interaction_values": {},
            "summary_plots_data": {},
            "models_analyzed": 0
        }
        
        X = prepared_data["feature_matrix"]
        y = prepared_data["accuracy_values"]
        feature_names = prepared_data["feature_names"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use provided models or train default
        models_to_analyze = prepared_data.get("models", {})
        
        if not models_to_analyze:
            default_model = RandomForestRegressor(n_estimators=100, random_state=42)
            default_model.fit(X_train, y_train)
            models_to_analyze["default_rf"] = default_model
        
        # Analyze each model
        for model_name, model in models_to_analyze.items():
            try:
                if hasattr(model, "predict"):
                    # Create appropriate explainer
                    if hasattr(model, "n_estimators"):  # Tree-based
                        explainer = shap.TreeExplainer(model)
                    else:
                        explainer = shap.KernelExplainer(model.predict, 
                                                        shap.sample(X_train, min(100, len(X_train))))
                    
                    # Calculate SHAP values
                    shap_values = explainer.shap_values(X_test[:min(1000, len(X_test))])
                    
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]
                    
                    # Calculate feature importance
                    feature_importance = np.abs(shap_values).mean(axis=0)
                    feature_importance_normalized = feature_importance / feature_importance.sum()
                    
                    # Create importance dictionary
                    importance_dict = {
                        feature_names[i]: {
                            "importance": float(feature_importance[i]),
                            "importance_normalized": float(feature_importance_normalized[i]),
                            "rank": 0
                        }
                        for i in range(len(feature_names))
                    }
                    
                    # Add rankings
                    sorted_features = sorted(importance_dict.items(), 
                                           key=lambda x: x[1]["importance"], 
                                           reverse=True)
                    for rank, (fname, _) in enumerate(sorted_features):
                        importance_dict[fname]["rank"] = rank + 1
                    
                    shap_results["feature_importance"][model_name] = importance_dict
                    
                    # Store summary data
                    shap_results["summary_plots_data"][model_name] = {
                        "shap_values_sample": shap_values[:100].tolist(),
                        "feature_values_sample": X_test[:100].tolist(),
                        "base_value": float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0
                    }
                    
                    shap_results["models_analyzed"] += 1
                    
            except Exception as e:
                self.logger.warning(f"SHAP analysis failed for model {model_name}: {e}")
                shap_results["feature_importance"][model_name] = {"error": str(e)}
        
        # Calculate consensus if multiple models
        if len(shap_results["feature_importance"]) > 1:
            consensus_importance = self._calculate_consensus_importance(shap_results["feature_importance"])
            shap_results["consensus_importance"] = consensus_importance
        
        return shap_results
    
    def _perform_permutation_importance(self, prepared_data: Dict[str, Any], 
                                       config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform permutation importance analysis"""
        perm_results = {
            "method": "permutation_importance",
            "feature_importance": {},
            "confidence_intervals": {},
            "n_repeats": config.get("n_repeats", 10)
        }
        
        X = prepared_data["feature_matrix"]
        y = prepared_data["accuracy_values"]
        feature_names = prepared_data["feature_names"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = prepared_data.get("models", {})
        
        if not models:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            models["default_rf"] = model
        
        # Calculate permutation importance for each model
        for model_name, model in models.items():
            try:
                if hasattr(model, "predict"):
                    result = permutation_importance(
                        model, X_test, y_test,
                        n_repeats=perm_results["n_repeats"],
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    importances = result.importances_mean
                    importances_std = result.importances_std
                    
                    # Create importance dictionary
                    importance_dict = {}
                    for i, fname in enumerate(feature_names):
                        importance_dict[fname] = {
                            "importance": float(importances[i]),
                            "std": float(importances_std[i]),
                            "confidence_interval": {
                                "lower": float(importances[i] - 2 * importances_std[i]),
                                "upper": float(importances[i] + 2 * importances_std[i])
                            },
                            "rank": 0
                        }
                    
                    # Add rankings
                    sorted_features = sorted(importance_dict.items(), 
                                           key=lambda x: x[1]["importance"], 
                                           reverse=True)
                    for rank, (fname, _) in enumerate(sorted_features):
                        importance_dict[fname]["rank"] = rank + 1
                    
                    perm_results["feature_importance"][model_name] = importance_dict
                    
                    # Store confidence intervals
                    perm_results["confidence_intervals"][model_name] = {
                        fname: {
                            "raw_scores": result.importances[i].tolist(),
                            "mean": float(importances[i]),
                            "std": float(importances_std[i]),
                            "cv": float(importances_std[i] / importances[i]) if importances[i] != 0 else float('inf')
                        }
                        for i, fname in enumerate(feature_names)
                    }
                    
            except Exception as e:
                self.logger.warning(f"Permutation importance failed for model {model_name}: {e}")
                perm_results["feature_importance"][model_name] = {"error": str(e)}
        
        return perm_results
    
    def _perform_feature_correlation_analysis(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive correlation analysis"""
        corr_results = {
            "method": "correlation_analysis",
            "pearson_correlation": {},
            "spearman_correlation": {},
            "partial_correlation": {},
            "correlation_clusters": {}
        }
        
        X = prepared_data["feature_matrix"]
        y = prepared_data["accuracy_values"]
        feature_names = prepared_data["feature_names"]
        
        df = pd.DataFrame(X, columns=feature_names)
        df['accuracy'] = y
        
        # Pearson correlation
        pearson_corr = df.corr(method='pearson')
        accuracy_corr_pearson = pearson_corr['accuracy'].drop('accuracy')
        
        corr_results["pearson_correlation"] = {
            fname: {
                "correlation": float(accuracy_corr_pearson[fname]),
                "abs_correlation": float(abs(accuracy_corr_pearson[fname])),
                "p_value": float(stats.pearsonr(df[fname], df['accuracy'])[1]),
                "significant": stats.pearsonr(df[fname], df['accuracy'])[1] < 0.05
            }
            for fname in feature_names
        }
        
        # Spearman correlation
        spearman_corr = df.corr(method='spearman')
        accuracy_corr_spearman = spearman_corr['accuracy'].drop('accuracy')
        
        corr_results["spearman_correlation"] = {
            fname: {
                "correlation": float(accuracy_corr_spearman[fname]),
                "abs_correlation": float(abs(accuracy_corr_spearman[fname])),
                "p_value": float(stats.spearmanr(df[fname], df['accuracy'])[1]),
                "significant": stats.spearmanr(df[fname], df['accuracy'])[1] < 0.05
            }
            for fname in feature_names
        }
        
        # Partial correlation if available
        if PINGOUIN_AVAILABLE:
            try:
                import pingouin as pg
                
                partial_correlations = {}
                for fname in feature_names:
                    covariates = [f for f in feature_names if f != fname]
                    
                    if len(covariates) > 0:
                        partial_result = pg.partial_corr(
                            data=df,
                            x=fname,
                            y='accuracy',
                            covar=covariates[:min(5, len(covariates))]
                        )
                        
                        if not partial_result.empty:
                            partial_correlations[fname] = {
                                "correlation": float(partial_result['r'].iloc[0]),
                                "p_value": float(partial_result['p-val'].iloc[0]),
                                "significant": partial_result['p-val'].iloc[0] < 0.05
                            }
                
                corr_results["partial_correlation"] = partial_correlations
                
            except Exception as e:
                self.logger.warning(f"Partial correlation failed: {e}")
        
        # Feature correlation clusters
        try:
            feature_corr_matrix = pearson_corr.drop('accuracy', axis=0).drop('accuracy', axis=1)
            distance_matrix = 1 - abs(feature_corr_matrix)
            
            n_clusters = min(5, len(feature_names) // 3)
            if n_clusters >= 2:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric='precomputed',
                    linkage='average'
                )
                cluster_labels = clustering.fit_predict(distance_matrix)
                
                clusters = {}
                for i, label in enumerate(cluster_labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(feature_names[i])
                
                corr_results["correlation_clusters"] = {
                    f"cluster_{i}": {
                        "features": features,
                        "size": len(features),
                        "avg_internal_correlation": float(
                            feature_corr_matrix.loc[features, features].abs().mean().mean()
                        )
                    }
                    for i, features in clusters.items()
                }
                
        except Exception as e:
            self.logger.warning(f"Correlation clustering failed: {e}")
        
        return corr_results
    
    def _perform_mutual_information_analysis(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform mutual information analysis"""
        mi_results = {
            "method": "mutual_information",
            "feature_scores": {},
            "normalized_scores": {},
            "binning_strategy": "adaptive"
        }
        
        X = prepared_data["feature_matrix"]
        y = prepared_data["accuracy_values"]
        feature_names = prepared_data["feature_names"]
        
        # Standard mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        max_score = mi_scores.max() if mi_scores.max() > 0 else 1
        normalized_scores = mi_scores / max_score
        
        for i, fname in enumerate(feature_names):
            mi_results["feature_scores"][fname] = {
                "mutual_info": float(mi_scores[i]),
                "normalized": float(normalized_scores[i]),
                "rank": 0
            }
        
        # Add rankings
        sorted_features = sorted(mi_results["feature_scores"].items(), 
                               key=lambda x: x[1]["mutual_info"], 
                               reverse=True)
        for rank, (fname, _) in enumerate(sorted_features):
            mi_results["feature_scores"][fname]["rank"] = rank + 1
        
        # Discretized mutual information
        try:
            discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
            X_discretized = discretizer.fit_transform(X)
            
            mi_scores_discretized = mutual_info_regression(X_discretized, y, random_state=42)
            
            mi_results["discretized_scores"] = {
                fname: {
                    "mutual_info": float(mi_scores_discretized[i]),
                    "normalized": float(mi_scores_discretized[i] / mi_scores_discretized.max()) 
                        if mi_scores_discretized.max() > 0 else 0
                }
                for i, fname in enumerate(feature_names)
            }
            
        except Exception as e:
            self.logger.warning(f"Discretized MI calculation failed: {e}")
        
        return mi_results
    
    def _analyze_feature_interactions(self, prepared_data: Dict[str, Any], 
                                     previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feature interactions"""
        interaction_results = {
            "method": "interaction_analysis",
            "pairwise_interactions": {},
            "interaction_strength": {},
            "synergistic_features": [],
            "antagonistic_features": []
        }
        
        X = prepared_data["feature_matrix"]
        y = prepared_data["accuracy_values"]
        feature_names = prepared_data["feature_names"]
        
        # Limit to top features for efficiency
        top_n = min(10, len(feature_names))
        top_features_indices = self._get_top_feature_indices(previous_results, feature_names, top_n)
        
        # Pairwise interaction analysis
        interaction_scores = {}
        
        for i, idx1 in enumerate(top_features_indices):
            for j, idx2 in enumerate(top_features_indices[i+1:], i+1):
                feature1, feature2 = feature_names[idx1], feature_names[idx2]
                
                try:
                    # Model with individual features
                    X_individual = X[:, [idx1, idx2]]
                    model_individual = Ridge(alpha=1.0)
                    score_individual = cross_val_score(model_individual, X_individual, y, cv=5).mean()
                    
                    # Model with interaction
                    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                    X_interaction = poly.fit_transform(X_individual)
                    model_interaction = Ridge(alpha=1.0)
                    score_interaction = cross_val_score(model_interaction, X_interaction, y, cv=5).mean()
                    
                    # Calculate interaction effect
                    interaction_effect = score_interaction - score_individual
                    
                    interaction_key = f"{feature1}_x_{feature2}"
                    interaction_scores[interaction_key] = {
                        "feature1": feature1,
                        "feature2": feature2,
                        "individual_score": float(score_individual),
                        "interaction_score": float(score_interaction),
                        "interaction_effect": float(interaction_effect),
                        "effect_type": "synergistic" if interaction_effect > 0.01 else 
                                      "antagonistic" if interaction_effect < -0.01 else "neutral"
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Interaction analysis failed for {feature1} x {feature2}: {e}")
        
        interaction_results["pairwise_interactions"] = interaction_scores
        
        # Identify synergistic and antagonistic features
        for interaction, scores in interaction_scores.items():
            if scores["effect_type"] == "synergistic":
                interaction_results["synergistic_features"].append({
                    "features": [scores["feature1"], scores["feature2"]],
                    "effect": scores["interaction_effect"]
                })
            elif scores["effect_type"] == "antagonistic":
                interaction_results["antagonistic_features"].append({
                    "features": [scores["feature1"], scores["feature2"]],
                    "effect": scores["interaction_effect"]
                })
        
        # Sort by effect magnitude
        interaction_results["synergistic_features"].sort(key=lambda x: abs(x["effect"]), reverse=True)
        interaction_results["antagonistic_features"].sort(key=lambda x: abs(x["effect"]), reverse=True)
        
        return interaction_results
    
    def _get_top_feature_indices(self, previous_results: Dict[str, Any], 
                                feature_names: List[str], top_n: int) -> List[int]:
        """Get indices of top features from previous analyses"""
        feature_scores = {}
        
        # Aggregate scores from different methods
        if "shap_analysis" in previous_results and previous_results["shap_analysis"]:
            shap_data = previous_results["shap_analysis"]
            if "consensus_importance" in shap_data:
                for fname, importance in shap_data["consensus_importance"].items():
                    if fname not in feature_scores:
                        feature_scores[fname] = []
                    feature_scores[fname].append(importance["importance"])
        
        if "permutation_importance" in previous_results and previous_results["permutation_importance"]:
            perm_data = previous_results["permutation_importance"]
            for model_results in perm_data.get("feature_importance", {}).values():
                if isinstance(model_results, dict) and "error" not in model_results:
                    for fname, importance in model_results.items():
                        if isinstance(importance, dict) and fname in feature_names:
                            if fname not in feature_scores:
                                feature_scores[fname] = []
                            feature_scores[fname].append(importance.get("importance", 0))
        
        # Calculate average scores
        avg_scores = {}
        for fname, scores in feature_scores.items():
            avg_scores[fname] = np.mean(scores)
        
        if not avg_scores:
            return list(range(min(top_n, len(feature_names))))
        
        # Get top feature names
        sorted_features = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        top_feature_names = [fname for fname, _ in sorted_features[:top_n]]
        
        # Convert to indices
        indices = []
        for fname in top_feature_names:
            if fname in feature_names:
                indices.append(feature_names.index(fname))
        
        return indices
    
    def _fallback_feature_importance(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback feature importance when SHAP unavailable"""
        X = prepared_data["feature_matrix"]
        y = prepared_data["accuracy_values"]
        feature_names = prepared_data["feature_names"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        importances = model.feature_importances_
        
        feature_importance = {}
        for i, fname in enumerate(feature_names):
            feature_importance[fname] = {
                "importance": float(importances[i]),
                "importance_normalized": float(importances[i] / importances.sum()),
                "rank": 0
            }
        
        # Add rankings
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1]["importance"], 
                               reverse=True)
        for rank, (fname, _) in enumerate(sorted_features):
            feature_importance[fname]["rank"] = rank + 1
        
        return {
            "method": "random_forest_importance",
            "feature_importance": {"fallback_model": feature_importance},
            "models_analyzed": 1
        }
    
    def _calculate_consensus_importance(self, model_importances: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate consensus feature importance across models"""
        consensus = {}
        feature_scores = {}
        
        for model_name, importance_dict in model_importances.items():
            if isinstance(importance_dict, dict) and "error" not in importance_dict:
                for fname, scores in importance_dict.items():
                    if isinstance(scores, dict):
                        if fname not in feature_scores:
                            feature_scores[fname] = []
                        
                        score = scores.get("importance_normalized", scores.get("importance", 0))
                        feature_scores[fname].append(score)
        
        for fname, scores in feature_scores.items():
            consensus[fname] = {
                "importance": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "agreement": float(1 - np.std(scores) / (np.mean(scores) + 1e-10)),
                "num_models": len(scores)
            }
        
        # Add rankings
        sorted_features = sorted(consensus.items(), 
                               key=lambda x: x[1]["importance"], 
                               reverse=True)
        for rank, (fname, _) in enumerate(sorted_features):
            consensus[fname]["rank"] = rank + 1
        
        return consensus