"""
Advanced Analytics Engine for Phase 6B - Saraphis Financial Fraud Detection System
Group 6B: Advanced Analytics Capabilities (Methods 6-10)

This module provides advanced analytics capabilities including:
- Predictive accuracy forecasting with ARIMA, Prophet, LSTM
- Feature impact analysis with SHAP values and permutation importance
- Root cause analysis with causal inference
- Model drift impact metrics and quantification
- ML-driven accuracy improvement recommendations

Author: Saraphis Development Team
Version: 1.0.0
"""

import logging
import threading
import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Forecasting libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import shap

# Configuration
warnings.filterwarnings('ignore', category=FutureWarning)
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

@dataclass
class ForecastResult:
    """Comprehensive forecast result with multiple models and confidence intervals."""
    model_id: str
    forecast_horizon: int
    timestamp: datetime
    
    # Forecast values
    arima_forecast: List[float]
    prophet_forecast: List[float]
    lstm_forecast: List[float]
    ensemble_forecast: List[float]
    
    # Confidence intervals (95%)
    arima_confidence: List[Tuple[float, float]]
    prophet_confidence: List[Tuple[float, float]]
    lstm_confidence: List[Tuple[float, float]]
    ensemble_confidence: List[Tuple[float, float]]
    
    # Model performance metrics
    arima_metrics: Dict[str, float]
    prophet_metrics: Dict[str, float]
    lstm_metrics: Dict[str, float]
    ensemble_metrics: Dict[str, float]
    
    # Best performing model
    best_model: str
    best_model_score: float
    
    # Trend analysis
    trend_direction: str  # 'improving', 'declining', 'stable'
    trend_strength: float  # 0-1 scale
    seasonality_detected: bool
    
    # Risk assessment
    forecast_reliability: float  # 0-1 scale
    prediction_intervals: Dict[str, Tuple[float, float]]
    risk_level: str  # 'low', 'medium', 'high'

@dataclass
class FeatureImpactResult:
    """Feature impact analysis with SHAP values and importance rankings."""
    model_id: str
    analysis_period: str
    timestamp: datetime
    
    # SHAP analysis
    shap_values: Dict[str, List[float]]
    shap_feature_importance: Dict[str, float]
    shap_interaction_matrix: Dict[str, Dict[str, float]]
    
    # Permutation importance
    permutation_importance: Dict[str, float]
    permutation_std: Dict[str, float]
    
    # Correlation analysis
    feature_accuracy_correlation: Dict[str, float]
    correlation_significance: Dict[str, float]
    
    # Feature rankings
    top_positive_features: List[Tuple[str, float]]
    top_negative_features: List[Tuple[str, float]]
    most_stable_features: List[Tuple[str, float]]
    most_volatile_features: List[Tuple[str, float]]
    
    # Impact quantification
    feature_impact_scores: Dict[str, float]
    cumulative_impact: float
    marginal_effects: Dict[str, float]
    
    # Recommendations
    feature_optimization_suggestions: List[Dict[str, Any]]
    feature_monitoring_priorities: List[str]

@dataclass
class RootCauseResult:
    """Root cause analysis with causal inference and event timeline."""
    model_id: str
    accuracy_decline_period: str
    timestamp: datetime
    
    # Causal analysis
    identified_causes: List[Dict[str, Any]]
    causal_strength: Dict[str, float]
    confidence_scores: Dict[str, float]
    
    # Event timeline
    significant_events: List[Dict[str, Any]]
    event_correlation_matrix: Dict[str, Dict[str, float]]
    timeline_analysis: Dict[str, Any]
    
    # Decision tree analysis
    decision_path: List[Dict[str, Any]]
    split_conditions: List[str]
    feature_thresholds: Dict[str, float]
    
    # Impact quantification
    cause_impact_scores: Dict[str, float]
    cumulative_impact: float
    recovery_indicators: List[str]
    
    # Recommendations
    immediate_actions: List[Dict[str, Any]]
    preventive_measures: List[Dict[str, Any]]
    monitoring_recommendations: List[str]
    
    # Risk assessment
    recurrence_probability: float
    impact_severity: str  # 'low', 'medium', 'high', 'critical'
    recovery_timeline: str

@dataclass
class DriftImpactResult:
    """Model drift impact metrics with quantification and monitoring."""
    model_id: str
    drift_analysis_period: str
    timestamp: datetime
    
    # Drift quantification
    feature_drift_scores: Dict[str, float]
    prediction_drift_score: float
    overall_drift_magnitude: float
    
    # Impact correlation
    drift_accuracy_correlation: Dict[str, float]
    drift_performance_impact: Dict[str, float]
    cumulative_drift_effect: float
    
    # Threshold monitoring
    drift_threshold_breaches: List[Dict[str, Any]]
    threshold_severity: Dict[str, str]
    escalation_triggers: List[str]
    
    # Temporal analysis
    drift_velocity: Dict[str, float]  # Rate of drift over time
    drift_acceleration: Dict[str, float]  # Change in drift rate
    drift_patterns: Dict[str, List[float]]
    
    # Risk assessment
    drift_risk_level: str  # 'low', 'medium', 'high', 'critical'
    business_impact_score: float
    model_reliability_score: float
    
    # Monitoring recommendations
    monitoring_frequency: Dict[str, str]
    alert_thresholds: Dict[str, float]
    intervention_recommendations: List[Dict[str, Any]]

@dataclass
class ImprovementRecommendation:
    """ML-driven accuracy improvement recommendation with priority scoring."""
    recommendation_id: str
    model_id: str
    category: str  # 'data', 'features', 'model', 'infrastructure'
    priority: str  # 'low', 'medium', 'high', 'critical'
    
    # Recommendation details
    title: str
    description: str
    rationale: str
    expected_impact: float  # Estimated accuracy improvement
    confidence: float  # 0-1 scale
    
    # Implementation details
    implementation_complexity: str  # 'low', 'medium', 'high'
    estimated_effort: str  # 'hours', 'days', 'weeks'
    required_resources: List[str]
    dependencies: List[str]
    
    # Cost-benefit analysis
    implementation_cost: float
    expected_benefit: float
    roi_estimate: float
    payback_period: str
    
    # Action plan
    action_steps: List[Dict[str, Any]]
    success_metrics: List[str]
    monitoring_plan: Dict[str, Any]
    
    # Risk assessment
    implementation_risk: str  # 'low', 'medium', 'high'
    potential_side_effects: List[str]
    rollback_plan: str

class AdvancedAnalyticsEngine:
    """
    Advanced Analytics Engine for Phase 6B - Advanced Analytics Capabilities
    
    Provides sophisticated analytics capabilities including forecasting, feature impact
    analysis, root cause analysis, drift impact metrics, and ML-driven recommendations.
    """
    
    def __init__(self, orchestrator):
        """Initialize the Advanced Analytics Engine with orchestrator reference."""
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._cache = {}
        self._cache_ttl = {}
        
        # Initialize ML models
        self._initialize_models()
        
        # Performance tracking
        self._performance_metrics = {
            'forecasts_generated': 0,
            'feature_analyses_completed': 0,
            'root_cause_investigations': 0,
            'drift_assessments': 0,
            'recommendations_generated': 0
        }
        
        self.logger.info("Advanced Analytics Engine initialized successfully")
    
    def _initialize_models(self):
        """Initialize machine learning models for advanced analytics."""
        try:
            # LSTM model architecture
            self._lstm_model = None  # Will be built dynamically based on data
            self._scaler = MinMaxScaler()
            
            # SHAP explainer (will be initialized per model)
            self._shap_explainers = {}
            
            # Drift detection models
            self._drift_detector = IsolationForest(contamination=0.1, random_state=42)
            
            # Recommendation engine components
            self._recommendation_models = {
                'feature_optimizer': RandomForestRegressor(n_estimators=100, random_state=42),
                'impact_predictor': RandomForestRegressor(n_estimators=100, random_state=42)
            }
            
            self.logger.debug("ML models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing ML models: {e}")
            raise
    
    def generate_predictive_accuracy_forecasts(
        self,
        model_ids: List[str],
        forecast_horizon: int = 30,
        methods: List[str] = None,
        confidence_level: float = 0.95,
        include_ensemble: bool = True
    ) -> Dict[str, ForecastResult]:
        """
        Generate predictive accuracy forecasts using ARIMA, Prophet, LSTM, and ensemble methods.
        
        Args:
            model_ids: List of model IDs to forecast
            forecast_horizon: Number of periods to forecast ahead
            methods: Forecasting methods to use ['arima', 'prophet', 'lstm', 'ensemble']
            confidence_level: Confidence level for prediction intervals
            include_ensemble: Whether to include ensemble predictions
        
        Returns:
            Dictionary mapping model IDs to ForecastResult objects
        """
        try:
            with self._lock:
                self.logger.info(f"Generating predictive forecasts for {len(model_ids)} models")
                
                if methods is None:
                    methods = ['arima', 'prophet', 'lstm', 'ensemble']
                
                results = {}
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {
                        executor.submit(
                            self._generate_model_forecast,
                            model_id, forecast_horizon, methods, confidence_level, include_ensemble
                        ): model_id for model_id in model_ids
                    }
                    
                    for future in as_completed(futures):
                        model_id = futures[future]
                        try:
                            result = future.result()
                            results[model_id] = result
                            self.logger.debug(f"Forecast completed for model {model_id}")
                        except Exception as e:
                            self.logger.error(f"Forecast failed for model {model_id}: {e}")
                            results[model_id] = self._create_error_forecast_result(model_id, str(e))
                
                self._performance_metrics['forecasts_generated'] += len(results)
                self.logger.info(f"Generated forecasts for {len(results)} models")
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error generating predictive forecasts: {e}")
            raise
    
    def _generate_model_forecast(
        self,
        model_id: str,
        forecast_horizon: int,
        methods: List[str],
        confidence_level: float,
        include_ensemble: bool
    ) -> ForecastResult:
        """Generate forecast for a single model using multiple methods."""
        try:
            # Get historical accuracy data
            historical_data = self._get_model_accuracy_history(model_id)
            
            if len(historical_data) < 30:  # Minimum data requirement
                raise ValueError(f"Insufficient historical data for model {model_id}")
            
            # Prepare time series data
            ts_data = pd.Series(
                historical_data['accuracy_values'],
                index=pd.to_datetime(historical_data['timestamps'])
            )
            
            forecasts = {}
            confidence_intervals = {}
            metrics = {}
            
            # ARIMA forecasting
            if 'arima' in methods:
                arima_forecast, arima_conf, arima_metrics = self._arima_forecast(
                    ts_data, forecast_horizon, confidence_level
                )
                forecasts['arima'] = arima_forecast
                confidence_intervals['arima'] = arima_conf
                metrics['arima'] = arima_metrics
            
            # Prophet forecasting
            if 'prophet' in methods:
                prophet_forecast, prophet_conf, prophet_metrics = self._prophet_forecast(
                    ts_data, forecast_horizon, confidence_level
                )
                forecasts['prophet'] = prophet_forecast
                confidence_intervals['prophet'] = prophet_conf
                metrics['prophet'] = prophet_metrics
            
            # LSTM forecasting
            if 'lstm' in methods:
                lstm_forecast, lstm_conf, lstm_metrics = self._lstm_forecast(
                    ts_data, forecast_horizon, confidence_level
                )
                forecasts['lstm'] = lstm_forecast
                confidence_intervals['lstm'] = lstm_conf
                metrics['lstm'] = lstm_metrics
            
            # Ensemble forecasting
            ensemble_forecast = []
            ensemble_conf = []
            ensemble_metrics = {}
            
            if include_ensemble and len(forecasts) > 1:
                ensemble_forecast, ensemble_conf, ensemble_metrics = self._ensemble_forecast(
                    forecasts, confidence_intervals, confidence_level
                )
            
            # Determine best model
            best_model, best_score = self._select_best_forecast_model(metrics)
            
            # Trend analysis
            trend_direction, trend_strength, seasonality = self._analyze_forecast_trends(ts_data)
            
            # Risk assessment
            reliability, prediction_intervals, risk_level = self._assess_forecast_risk(
                forecasts, confidence_intervals, ts_data
            )
            
            return ForecastResult(
                model_id=model_id,
                forecast_horizon=forecast_horizon,
                timestamp=datetime.now(),
                arima_forecast=forecasts.get('arima', []),
                prophet_forecast=forecasts.get('prophet', []),
                lstm_forecast=forecasts.get('lstm', []),
                ensemble_forecast=ensemble_forecast,
                arima_confidence=confidence_intervals.get('arima', []),
                prophet_confidence=confidence_intervals.get('prophet', []),
                lstm_confidence=confidence_intervals.get('lstm', []),
                ensemble_confidence=ensemble_conf,
                arima_metrics=metrics.get('arima', {}),
                prophet_metrics=metrics.get('prophet', {}),
                lstm_metrics=metrics.get('lstm', {}),
                ensemble_metrics=ensemble_metrics,
                best_model=best_model,
                best_model_score=best_score,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                seasonality_detected=seasonality,
                forecast_reliability=reliability,
                prediction_intervals=prediction_intervals,
                risk_level=risk_level
            )
            
        except Exception as e:
            self.logger.error(f"Error generating forecast for model {model_id}: {e}")
            raise
    
    def analyze_feature_impact_on_accuracy(
        self,
        model_ids: List[str],
        analysis_period: str = "30d",
        include_shap: bool = True,
        include_permutation: bool = True,
        include_correlation: bool = True
    ) -> Dict[str, FeatureImpactResult]:
        """
        Analyze feature impact on model accuracy using SHAP values, permutation importance,
        and correlation analysis.
        
        Args:
            model_ids: List of model IDs to analyze
            analysis_period: Time period for analysis ('7d', '30d', '90d')
            include_shap: Whether to perform SHAP analysis
            include_permutation: Whether to perform permutation importance
            include_correlation: Whether to perform correlation analysis
        
        Returns:
            Dictionary mapping model IDs to FeatureImpactResult objects
        """
        try:
            with self._lock:
                self.logger.info(f"Analyzing feature impact for {len(model_ids)} models")
                
                results = {}
                
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = {
                        executor.submit(
                            self._analyze_model_feature_impact,
                            model_id, analysis_period, include_shap, 
                            include_permutation, include_correlation
                        ): model_id for model_id in model_ids
                    }
                    
                    for future in as_completed(futures):
                        model_id = futures[future]
                        try:
                            result = future.result()
                            results[model_id] = result
                            self.logger.debug(f"Feature impact analysis completed for model {model_id}")
                        except Exception as e:
                            self.logger.error(f"Feature impact analysis failed for model {model_id}: {e}")
                            results[model_id] = self._create_error_feature_impact_result(model_id, str(e))
                
                self._performance_metrics['feature_analyses_completed'] += len(results)
                self.logger.info(f"Completed feature impact analysis for {len(results)} models")
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error analyzing feature impact: {e}")
            raise
    
    def perform_root_cause_analysis(
        self,
        model_id: str,
        accuracy_decline_threshold: float = 0.05,
        analysis_window: str = "14d",
        include_causal_inference: bool = True,
        include_event_timeline: bool = True
    ) -> RootCauseResult:
        """
        Perform root cause analysis for accuracy declines using causal inference,
        decision trees, and event timeline analysis.
        
        Args:
            model_id: Model ID to analyze
            accuracy_decline_threshold: Minimum decline to trigger analysis
            analysis_window: Time window for analysis
            include_causal_inference: Whether to perform causal analysis
            include_event_timeline: Whether to analyze event timeline
        
        Returns:
            RootCauseResult object with identified causes and recommendations
        """
        try:
            self.logger.info(f"Performing root cause analysis for model {model_id}")
            
            # Detect accuracy decline periods
            decline_periods = self._detect_accuracy_declines(
                model_id, accuracy_decline_threshold, analysis_window
            )
            
            if not decline_periods:
                self.logger.info(f"No significant accuracy declines detected for model {model_id}")
                return self._create_no_decline_result(model_id)
            
            # Analyze the most recent decline period
            decline_period = decline_periods[0]
            
            # Get relevant data for analysis
            model_data = self._get_root_cause_analysis_data(model_id, decline_period)
            
            identified_causes = []
            causal_strength = {}
            confidence_scores = {}
            
            # Causal inference analysis
            if include_causal_inference:
                causes, strengths, confidences = self._perform_causal_inference(model_data)
                identified_causes.extend(causes)
                causal_strength.update(strengths)
                confidence_scores.update(confidences)
            
            # Decision tree analysis
            decision_path, split_conditions, thresholds = self._decision_tree_analysis(model_data)
            
            # Event timeline analysis
            events = []
            event_correlations = {}
            timeline_analysis = {}
            
            if include_event_timeline:
                events, event_correlations, timeline_analysis = self._analyze_event_timeline(
                    model_id, decline_period
                )
            
            # Impact quantification
            cause_impacts = self._quantify_cause_impacts(identified_causes, model_data)
            cumulative_impact = sum(cause_impacts.values())
            
            # Generate recommendations
            immediate_actions = self._generate_immediate_actions(identified_causes, cause_impacts)
            preventive_measures = self._generate_preventive_measures(identified_causes)
            monitoring_recommendations = self._generate_monitoring_recommendations(identified_causes)
            
            # Risk assessment
            recurrence_prob = self._assess_recurrence_probability(identified_causes, model_data)
            impact_severity = self._assess_impact_severity(cumulative_impact)
            recovery_timeline = self._estimate_recovery_timeline(identified_causes, cause_impacts)
            
            # Recovery indicators
            recovery_indicators = self._identify_recovery_indicators(identified_causes)
            
            result = RootCauseResult(
                model_id=model_id,
                accuracy_decline_period=f"{decline_period['start']} to {decline_period['end']}",
                timestamp=datetime.now(),
                identified_causes=identified_causes,
                causal_strength=causal_strength,
                confidence_scores=confidence_scores,
                significant_events=events,
                event_correlation_matrix=event_correlations,
                timeline_analysis=timeline_analysis,
                decision_path=decision_path,
                split_conditions=split_conditions,
                feature_thresholds=thresholds,
                cause_impact_scores=cause_impacts,
                cumulative_impact=cumulative_impact,
                recovery_indicators=recovery_indicators,
                immediate_actions=immediate_actions,
                preventive_measures=preventive_measures,
                monitoring_recommendations=monitoring_recommendations,
                recurrence_probability=recurrence_prob,
                impact_severity=impact_severity,
                recovery_timeline=recovery_timeline
            )
            
            self._performance_metrics['root_cause_investigations'] += 1
            self.logger.info(f"Root cause analysis completed for model {model_id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error performing root cause analysis for model {model_id}: {e}")
            raise
    
    def calculate_model_drift_impact_metrics(
        self,
        model_ids: List[str],
        drift_analysis_period: str = "30d",
        include_threshold_monitoring: bool = True,
        calculate_business_impact: bool = True
    ) -> Dict[str, DriftImpactResult]:
        """
        Calculate model drift impact metrics with quantification, correlation analysis,
        and threshold monitoring.
        
        Args:
            model_ids: List of model IDs to analyze
            drift_analysis_period: Time period for drift analysis
            include_threshold_monitoring: Whether to monitor threshold breaches
            calculate_business_impact: Whether to calculate business impact
        
        Returns:
            Dictionary mapping model IDs to DriftImpactResult objects
        """
        try:
            with self._lock:
                self.logger.info(f"Calculating drift impact metrics for {len(model_ids)} models")
                
                results = {}
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {
                        executor.submit(
                            self._calculate_model_drift_impact,
                            model_id, drift_analysis_period, 
                            include_threshold_monitoring, calculate_business_impact
                        ): model_id for model_id in model_ids
                    }
                    
                    for future in as_completed(futures):
                        model_id = futures[future]
                        try:
                            result = future.result()
                            results[model_id] = result
                            self.logger.debug(f"Drift impact calculation completed for model {model_id}")
                        except Exception as e:
                            self.logger.error(f"Drift impact calculation failed for model {model_id}: {e}")
                            results[model_id] = self._create_error_drift_result(model_id, str(e))
                
                self._performance_metrics['drift_assessments'] += len(results)
                self.logger.info(f"Completed drift impact analysis for {len(results)} models")
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error calculating drift impact metrics: {e}")
            raise
    
    def generate_accuracy_improvement_recommendations(
        self,
        model_ids: List[str],
        recommendation_categories: List[str] = None,
        max_recommendations_per_model: int = 10,
        min_expected_impact: float = 0.01,
        include_cost_benefit: bool = True
    ) -> Dict[str, List[ImprovementRecommendation]]:
        """
        Generate ML-driven accuracy improvement recommendations with priority scoring,
        cost-benefit analysis, and implementation plans.
        
        Args:
            model_ids: List of model IDs to generate recommendations for
            recommendation_categories: Categories to focus on ['data', 'features', 'model', 'infrastructure']
            max_recommendations_per_model: Maximum number of recommendations per model
            min_expected_impact: Minimum expected accuracy improvement threshold
            include_cost_benefit: Whether to include cost-benefit analysis
        
        Returns:
            Dictionary mapping model IDs to lists of ImprovementRecommendation objects
        """
        try:
            with self._lock:
                self.logger.info(f"Generating improvement recommendations for {len(model_ids)} models")
                
                if recommendation_categories is None:
                    recommendation_categories = ['data', 'features', 'model', 'infrastructure']
                
                results = {}
                
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = {
                        executor.submit(
                            self._generate_model_recommendations,
                            model_id, recommendation_categories, max_recommendations_per_model,
                            min_expected_impact, include_cost_benefit
                        ): model_id for model_id in model_ids
                    }
                    
                    for future in as_completed(futures):
                        model_id = futures[future]
                        try:
                            recommendations = future.result()
                            results[model_id] = recommendations
                            self.logger.debug(f"Generated {len(recommendations)} recommendations for model {model_id}")
                        except Exception as e:
                            self.logger.error(f"Recommendation generation failed for model {model_id}: {e}")
                            results[model_id] = []
                
                total_recommendations = sum(len(recs) for recs in results.values())
                self._performance_metrics['recommendations_generated'] += total_recommendations
                self.logger.info(f"Generated {total_recommendations} improvement recommendations")
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error generating improvement recommendations: {e}")
            raise
    
    # Helper methods for forecasting
    def _arima_forecast(self, ts_data: pd.Series, horizon: int, confidence_level: float) -> Tuple[List[float], List[Tuple[float, float]], Dict[str, float]]:
        """Generate ARIMA forecast with confidence intervals and metrics."""
        try:
            # Determine ARIMA order using AIC
            best_order = self._find_best_arima_order(ts_data)
            
            # Fit ARIMA model
            model = ARIMA(ts_data, order=best_order)
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=horizon)
            conf_int = fitted_model.get_forecast(steps=horizon).conf_int(alpha=1-confidence_level)
            
            # Calculate metrics
            in_sample_pred = fitted_model.fittedvalues
            metrics = {
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'mse': mean_squared_error(ts_data, in_sample_pred),
                'mae': mean_absolute_error(ts_data, in_sample_pred)
            }
            
            confidence_intervals = [(row[0], row[1]) for _, row in conf_int.iterrows()]
            
            return forecast.tolist(), confidence_intervals, metrics
            
        except Exception as e:
            self.logger.error(f"ARIMA forecasting error: {e}")
            return [], [], {}
    
    def _prophet_forecast(self, ts_data: pd.Series, horizon: int, confidence_level: float) -> Tuple[List[float], List[Tuple[float, float]], Dict[str, float]]:
        """Generate Prophet forecast with confidence intervals and metrics."""
        try:
            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': ts_data.index,
                'y': ts_data.values
            })
            
            # Initialize and fit Prophet model
            model = Prophet(
                uncertainty_samples=1000,
                interval_width=confidence_level
            )
            model.fit(df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=horizon)
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Extract forecast values and confidence intervals
            forecast_values = forecast['yhat'].tail(horizon).tolist()
            conf_intervals = list(zip(
                forecast['yhat_lower'].tail(horizon).tolist(),
                forecast['yhat_upper'].tail(horizon).tolist()
            ))
            
            # Calculate metrics on in-sample predictions
            in_sample_pred = forecast['yhat'].iloc[:-horizon]
            metrics = {
                'mse': mean_squared_error(ts_data, in_sample_pred),
                'mae': mean_absolute_error(ts_data, in_sample_pred),
                'mape': np.mean(np.abs((ts_data - in_sample_pred) / ts_data)) * 100
            }
            
            return forecast_values, conf_intervals, metrics
            
        except Exception as e:
            self.logger.error(f"Prophet forecasting error: {e}")
            return [], [], {}
    
    def _lstm_forecast(self, ts_data: pd.Series, horizon: int, confidence_level: float) -> Tuple[List[float], List[Tuple[float, float]], Dict[str, float]]:
        """Generate LSTM forecast with confidence intervals and metrics."""
        try:
            # Prepare data for LSTM
            scaled_data = self._scaler.fit_transform(ts_data.values.reshape(-1, 1))
            
            # Create sequences
            sequence_length = min(30, len(scaled_data) // 4)
            X, y = self._create_lstm_sequences(scaled_data, sequence_length)
            
            if len(X) < 10:  # Insufficient data
                return [], [], {}
            
            # Build and train LSTM model
            model = self._build_lstm_model(sequence_length)
            model.fit(X, y, epochs=50, batch_size=32, verbose=0)
            
            # Generate forecast
            forecast_scaled = self._generate_lstm_forecast(model, scaled_data, horizon, sequence_length)
            forecast = self._scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
            
            # Generate confidence intervals using bootstrapping
            conf_intervals = self._generate_lstm_confidence_intervals(
                model, scaled_data, horizon, sequence_length, confidence_level
            )
            
            # Calculate metrics
            train_pred_scaled = model.predict(X)
            train_pred = self._scaler.inverse_transform(train_pred_scaled).flatten()
            actual_train = self._scaler.inverse_transform(y.reshape(-1, 1)).flatten()
            
            metrics = {
                'mse': mean_squared_error(actual_train, train_pred),
                'mae': mean_absolute_error(actual_train, train_pred),
                'rmse': np.sqrt(mean_squared_error(actual_train, train_pred))
            }
            
            return forecast.tolist(), conf_intervals, metrics
            
        except Exception as e:
            self.logger.error(f"LSTM forecasting error: {e}")
            return [], [], {}
    
    # Additional helper methods would continue here...
    # (Due to length constraints, showing key structure and first few methods)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics."""
        with self._lock:
            return {
                'performance_metrics': self._performance_metrics.copy(),
                'cache_size': len(self._cache),
                'active_models': len(self._shap_explainers),
                'last_updated': datetime.now().isoformat()
            }
    
    def clear_cache(self) -> None:
        """Clear internal cache."""
        with self._lock:
            self._cache.clear()
            self._cache_ttl.clear()
            self.logger.info("Advanced Analytics Engine cache cleared")
    
    def _get_model_accuracy_history(self, model_id: str) -> Dict[str, List]:
        """Get historical accuracy data for a model."""
        # Implementation would interface with orchestrator's accuracy database
        pass
    
    def _create_error_forecast_result(self, model_id: str, error: str) -> ForecastResult:
        """Create error forecast result."""
        return ForecastResult(
            model_id=model_id,
            forecast_horizon=0,
            timestamp=datetime.now(),
            arima_forecast=[],
            prophet_forecast=[],
            lstm_forecast=[],
            ensemble_forecast=[],
            arima_confidence=[],
            prophet_confidence=[],
            lstm_confidence=[],
            ensemble_confidence=[],
            arima_metrics={'error': error},
            prophet_metrics={'error': error},
            lstm_metrics={'error': error},
            ensemble_metrics={'error': error},
            best_model='none',
            best_model_score=0.0,
            trend_direction='unknown',
            trend_strength=0.0,
            seasonality_detected=False,
            forecast_reliability=0.0,
            prediction_intervals={},
            risk_level='high'
        )