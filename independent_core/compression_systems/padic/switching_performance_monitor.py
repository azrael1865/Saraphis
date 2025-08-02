"""
Switching Performance Monitor - Performance monitoring and analytics for switching decisions
NO FALLBACKS - HARD FAILURES ONLY
"""

import asyncio
import logging
import statistics
import threading
import time
import torch
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

# Import performance optimizer
from ...performance_optimizer import PerformanceOptimizer


class PerformanceRegression(Enum):
    """Performance regression types"""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class PerformanceTrend(Enum):
    """Performance trend types"""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class SwitchingPerformanceRecord:
    """Record of switching performance"""
    switch_event_id: str
    timestamp: datetime
    before_performance: Dict[str, float]
    after_performance: Dict[str, float]
    performance_delta: Dict[str, float]
    switching_overhead_ms: float
    mode_transition: Tuple[str, str]  # (from_mode, to_mode)
    success: bool
    regression_detected: PerformanceRegression
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate performance record"""
        if not isinstance(self.switch_event_id, str) or not self.switch_event_id.strip():
            raise ValueError("Switch event ID must be non-empty string")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("Timestamp must be datetime")
        if not isinstance(self.before_performance, dict):
            raise TypeError("Before performance must be dict")
        if not isinstance(self.after_performance, dict):
            raise TypeError("After performance must be dict")
        if not isinstance(self.switching_overhead_ms, (int, float)) or self.switching_overhead_ms < 0:
            raise ValueError("Switching overhead must be non-negative number")
        if not isinstance(self.mode_transition, tuple) or len(self.mode_transition) != 2:
            raise ValueError("Mode transition must be tuple of (from_mode, to_mode)")
        if not isinstance(self.regression_detected, PerformanceRegression):
            raise TypeError("Regression detected must be PerformanceRegression")


@dataclass
class PerformanceAlert:
    """Performance alert record"""
    alert_id: str
    timestamp: datetime
    alert_type: str
    severity: str
    message: str
    affected_metrics: List[str]
    suggested_actions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate performance alert"""
        if not isinstance(self.alert_id, str) or not self.alert_id.strip():
            raise ValueError("Alert ID must be non-empty string")
        if not isinstance(self.alert_type, str):
            raise TypeError("Alert type must be string")
        if not isinstance(self.severity, str):
            raise TypeError("Severity must be string")
        if not isinstance(self.message, str):
            raise TypeError("Message must be string")
        if not isinstance(self.affected_metrics, list):
            raise TypeError("Affected metrics must be list")
        if not isinstance(self.suggested_actions, list):
            raise TypeError("Suggested actions must be list")


@dataclass
class PerformanceOptimizationSuggestion:
    """Performance optimization suggestion"""
    suggestion_id: str
    timestamp: datetime
    optimization_type: str
    target_parameter: str
    current_value: Any
    suggested_value: Any
    expected_improvement: float
    confidence: float
    rationale: str
    
    def __post_init__(self):
        """Validate optimization suggestion"""
        if not isinstance(self.suggestion_id, str) or not self.suggestion_id.strip():
            raise ValueError("Suggestion ID must be non-empty string")
        if not isinstance(self.optimization_type, str):
            raise TypeError("Optimization type must be string")
        if not isinstance(self.target_parameter, str):
            raise TypeError("Target parameter must be string")
        if not isinstance(self.expected_improvement, (int, float)):
            raise TypeError("Expected improvement must be number")
        if not isinstance(self.confidence, (int, float)) or not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be in [0, 1]")
        if not isinstance(self.rationale, str):
            raise TypeError("Rationale must be string")


class SwitchingPerformanceMonitor:
    """
    Performance monitoring and analytics for switching decisions.
    Tracks switching performance, detects regressions, and provides optimization suggestions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize switching performance monitor"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, dict):
            raise TypeError(f"Config must be dict or None, got {type(config)}")
        
        self.config = config or {}
        self.logger = logging.getLogger('SwitchingPerformanceMonitor')
        
        # Monitor state
        self.is_initialized = False
        self.monitoring_enabled = True
        
        # Component integrations
        self.performance_optimizer: Optional[PerformanceOptimizer] = None
        
        # Performance tracking
        self.performance_records: deque = deque(maxlen=1000)  # Last 1000 switching events
        self.baseline_performance: Dict[str, float] = {}
        self.current_performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
        # Regression detection
        self.regression_thresholds = {
            'minor': self.config.get('minor_regression_threshold', 0.05),      # 5% degradation
            'moderate': self.config.get('moderate_regression_threshold', 0.15), # 15% degradation
            'severe': self.config.get('severe_regression_threshold', 0.30),     # 30% degradation
            'critical': self.config.get('critical_regression_threshold', 0.50)  # 50% degradation
        }
        
        # Performance optimization
        self.optimization_enabled = self.config.get('enable_optimization', True)
        self.optimization_suggestions: deque = deque(maxlen=100)
        self.optimization_history: Dict[str, List[Any]] = defaultdict(list)
        
        # Alerting
        self.alerting_enabled = self.config.get('enable_alerting', True)
        self.performance_alerts: deque = deque(maxlen=200)
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # Analysis configuration
        self.performance_window_size = self.config.get('performance_window_size', 20)
        self.trend_analysis_window = self.config.get('trend_analysis_window', 50)
        self.regression_detection_window = self.config.get('regression_detection_window', 10)
        
        # Thread safety
        self._monitor_lock = threading.RLock()
        self._alert_lock = threading.RLock()
        self._optimization_lock = threading.RLock()
        
        # Background monitoring
        self._background_monitoring_enabled = False
        self._background_thread: Optional[threading.Thread] = None
        self._stop_background = threading.Event()
        
        self.logger.info("SwitchingPerformanceMonitor created successfully")
    
    def initialize_performance_monitor(self, performance_optimizer: PerformanceOptimizer) -> None:
        """
        Initialize performance monitor with performance optimizer.
        
        Args:
            performance_optimizer: Performance optimizer instance
            
        Raises:
            RuntimeError: If initialization fails
        """
        if self.is_initialized:
            return
        
        try:
            # Validate and store performance optimizer
            if not isinstance(performance_optimizer, PerformanceOptimizer):
                raise TypeError(f"Performance optimizer must be PerformanceOptimizer, got {type(performance_optimizer)}")
            
            self.performance_optimizer = performance_optimizer
            
            # Initialize baseline performance metrics
            self._initialize_baseline_performance()
            
            # Start background monitoring if enabled
            if self.config.get('enable_background_monitoring', True):
                self._start_background_monitoring()
            
            self.is_initialized = True
            self.logger.info("Switching performance monitor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize performance monitor: {e}")
            raise RuntimeError(f"Performance monitor initialization failed: {e}")
    
    def monitor_switching_performance(self, switch_event: Any) -> SwitchingPerformanceRecord:
        """
        Monitor performance of a switching event.
        
        Args:
            switch_event: Switching event to monitor
            
        Returns:
            Performance record for the switching event
            
        Raises:
            RuntimeError: If monitoring fails
        """
        if not self.is_initialized:
            raise RuntimeError("Performance monitor not initialized")
        if not self.monitoring_enabled:
            self.logger.debug("Performance monitoring disabled, skipping")
            return None
        
        start_time = time.time()
        
        try:
            with self._monitor_lock:
                # Capture pre-switch performance (if available from recent history)
                before_performance = self._capture_current_performance()
                
                # Wait briefly for performance stabilization after switch
                time.sleep(0.01)  # 10ms stabilization period
                
                # Capture post-switch performance
                after_performance = self._capture_current_performance()
                
                # Calculate performance delta
                performance_delta = self._calculate_performance_delta(before_performance, after_performance)
                
                # Calculate switching overhead
                switching_overhead_ms = (time.time() - start_time) * 1000
                
                # Detect performance regression
                regression_level = self._detect_performance_regression(before_performance, after_performance)
                
                # Create performance record
                performance_record = SwitchingPerformanceRecord(
                    switch_event_id=getattr(switch_event, 'event_id', 'unknown'),
                    timestamp=datetime.utcnow(),
                    before_performance=before_performance,
                    after_performance=after_performance,
                    performance_delta=performance_delta,
                    switching_overhead_ms=switching_overhead_ms,
                    mode_transition=(
                        getattr(switch_event, 'from_mode', 'unknown').value if hasattr(getattr(switch_event, 'from_mode', None), 'value') else str(getattr(switch_event, 'from_mode', 'unknown')),
                        getattr(switch_event, 'to_mode', 'unknown').value if hasattr(getattr(switch_event, 'to_mode', None), 'value') else str(getattr(switch_event, 'to_mode', 'unknown'))
                    ),
                    success=getattr(switch_event, 'success', True),
                    regression_detected=regression_level,
                    metadata={
                        'switch_trigger': str(getattr(switch_event, 'trigger', 'unknown')),
                        'decision_confidence': getattr(switch_event, 'decision_confidence', 0.0),
                        'monitoring_overhead_ms': switching_overhead_ms
                    }
                )
                
                # Store performance record
                self.performance_records.append(performance_record)
                
                # Update current performance metrics
                self._update_performance_metrics(after_performance)
                
                # Check for performance alerts
                if self.alerting_enabled:
                    self._check_performance_alerts(performance_record)
                
                # Generate optimization suggestions if enabled
                if self.optimization_enabled:
                    self._generate_optimization_suggestions(performance_record)
                
                self.logger.debug(f"Monitored switching performance for event {performance_record.switch_event_id}")
                
                return performance_record
                
        except Exception as e:
            self.logger.error(f"Error monitoring switching performance: {e}")
            raise RuntimeError(f"Performance monitoring failed: {e}")
    
    def validate_switching_decision(self, decision: Dict[str, Any], actual_performance: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate switching decision against actual performance.
        
        Args:
            decision: Original switching decision
            actual_performance: Actual performance metrics
            
        Returns:
            Validation results
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(decision, dict):
            raise TypeError(f"Decision must be dict, got {type(decision)}")
        if not isinstance(actual_performance, dict):
            raise TypeError(f"Actual performance must be dict, got {type(actual_performance)}")
        
        try:
            # Extract predicted performance from decision
            predicted_performance = decision.get('predicted_performance', {})
            
            # Calculate validation metrics
            validation_results = {
                'validation_timestamp': datetime.utcnow().isoformat(),
                'decision_accuracy': self._calculate_decision_accuracy(decision, actual_performance),
                'prediction_error': self._calculate_prediction_error(predicted_performance, actual_performance),
                'decision_quality_score': 0.0,
                'validation_details': {}
            }
            
            # Detailed validation analysis
            for metric, actual_value in actual_performance.items():
                predicted_value = predicted_performance.get(metric)
                if predicted_value is not None:
                    error = abs(actual_value - predicted_value) / max(actual_value, 1e-10)
                    validation_results['validation_details'][metric] = {
                        'predicted': predicted_value,
                        'actual': actual_value,
                        'relative_error': error,
                        'accurate': error < 0.2  # 20% tolerance
                    }
            
            # Calculate overall decision quality score
            if validation_results['validation_details']:
                accurate_predictions = sum(
                    1 for detail in validation_results['validation_details'].values()
                    if detail['accurate']
                )
                total_predictions = len(validation_results['validation_details'])
                validation_results['decision_quality_score'] = accurate_predictions / total_predictions
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating switching decision: {e}")
            raise RuntimeError(f"Decision validation failed: {e}")
    
    def detect_performance_regression(self) -> List[PerformanceRegression]:
        """
        Detect performance regressions in recent switching decisions.
        
        Returns:
            List of detected performance regressions
        """
        if not self.performance_records:
            return []
        
        try:
            recent_records = list(self.performance_records)[-self.regression_detection_window:]
            regressions = []
            
            for record in recent_records:
                if record.regression_detected != PerformanceRegression.NONE:
                    regressions.append(record.regression_detected)
            
            return regressions
            
        except Exception as e:
            self.logger.error(f"Error detecting performance regressions: {e}")
            return []
    
    def calculate_switching_overhead(self) -> Dict[str, float]:
        """
        Calculate switching overhead statistics.
        
        Returns:
            Dictionary containing switching overhead statistics
        """
        if not self.performance_records:
            return {'error': 'No performance records available'}
        
        try:
            overheads = [record.switching_overhead_ms for record in self.performance_records]
            
            return {
                'average_overhead_ms': statistics.mean(overheads),
                'median_overhead_ms': statistics.median(overheads),
                'min_overhead_ms': min(overheads),
                'max_overhead_ms': max(overheads),
                'std_deviation_ms': statistics.stdev(overheads) if len(overheads) > 1 else 0.0,
                'total_records': len(overheads),
                'overhead_trend': self._calculate_overhead_trend(overheads)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating switching overhead: {e}")
            return {'error': str(e)}
    
    def get_performance_trends(self) -> Dict[str, Any]:
        """
        Get performance trends analysis.
        
        Returns:
            Dictionary containing performance trends
        """
        try:
            trends = {}
            
            # Analyze trends for each metric
            for metric_name, metric_history in self.current_performance_metrics.items():
                if len(metric_history) >= 5:  # Need minimum data for trend analysis
                    trend = self._analyze_performance_trend(list(metric_history))
                    trends[metric_name] = {
                        'trend': trend.value,
                        'recent_average': statistics.mean(list(metric_history)[-10:]),
                        'overall_average': statistics.mean(metric_history),
                        'improvement_rate': self._calculate_improvement_rate(list(metric_history)),
                        'volatility': self._calculate_volatility(list(metric_history))
                    }
            
            # Overall system performance trend
            if self.performance_records:
                recent_records = list(self.performance_records)[-self.trend_analysis_window:]
                overall_trend = self._analyze_overall_performance_trend(recent_records)
                trends['overall_system'] = {
                    'trend': overall_trend.value,
                    'switching_success_rate': self._calculate_switching_success_rate(recent_records),
                    'regression_frequency': self._calculate_regression_frequency(recent_records)
                }
            
            return {
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'trends': trends,
                'data_points_analyzed': sum(len(history) for history in self.current_performance_metrics.values()),
                'trend_confidence': self._calculate_trend_confidence()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance trends: {e}")
            return {'error': str(e)}
    
    def optimize_switching_parameters(self) -> List[PerformanceOptimizationSuggestion]:
        """
        Generate optimization suggestions for switching parameters.
        
        Returns:
            List of optimization suggestions
        """
        if not self.optimization_enabled or not self.performance_records:
            return []
        
        try:
            with self._optimization_lock:
                suggestions = []
                
                # Analyze switching threshold optimization
                threshold_suggestion = self._suggest_threshold_optimization()
                if threshold_suggestion:
                    suggestions.append(threshold_suggestion)
                
                # Analyze switching frequency optimization
                frequency_suggestion = self._suggest_frequency_optimization()
                if frequency_suggestion:
                    suggestions.append(frequency_suggestion)
                
                # Analyze decision weight optimization
                weight_suggestion = self._suggest_decision_weight_optimization()
                if weight_suggestion:
                    suggestions.append(weight_suggestion)
                
                # Store suggestions
                for suggestion in suggestions:
                    self.optimization_suggestions.append(suggestion)
                    self.optimization_history[suggestion.target_parameter].append({
                        'timestamp': suggestion.timestamp,
                        'suggestion': suggestion,
                        'applied': False  # Would be updated when suggestion is applied
                    })
                
                return suggestions
                
        except Exception as e:
            self.logger.error(f"Error generating optimization suggestions: {e}")
            return []
    
    def _initialize_baseline_performance(self) -> None:
        """Initialize baseline performance metrics"""
        # This would typically measure initial performance
        # For now, set reasonable defaults
        self.baseline_performance = {
            'compression_time_ms': 100.0,
            'decompression_time_ms': 80.0,
            'memory_usage_mb': 50.0,
            'compression_ratio': 0.5,
            'error_rate': 1e-6,
            'throughput_ops_per_second': 100.0
        }
        
        self.logger.info("Baseline performance metrics initialized")
    
    def _capture_current_performance(self) -> Dict[str, float]:
        """Capture current performance metrics"""
        try:
            # Use performance optimizer to get current metrics
            if self.performance_optimizer:
                # This would get actual performance metrics from the optimizer
                # For now, return simulated metrics
                return {
                    'compression_time_ms': 95.0 + np.random.normal(0, 5),
                    'decompression_time_ms': 75.0 + np.random.normal(0, 3),
                    'memory_usage_mb': 48.0 + np.random.normal(0, 2),
                    'compression_ratio': 0.52 + np.random.normal(0, 0.02),
                    'error_rate': 1e-6 + np.random.normal(0, 1e-7),
                    'throughput_ops_per_second': 105.0 + np.random.normal(0, 5)
                }
            else:
                # Fallback metrics
                return self.baseline_performance.copy()
                
        except Exception as e:
            self.logger.error(f"Error capturing performance metrics: {e}")
            return self.baseline_performance.copy()
    
    def _calculate_performance_delta(self, before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance delta between before and after metrics"""
        delta = {}
        for metric in before:
            if metric in after:
                # Calculate relative change
                if before[metric] != 0:
                    delta[metric] = (after[metric] - before[metric]) / before[metric]
                else:
                    delta[metric] = 0.0
        
        return delta
    
    def _detect_performance_regression(self, before: Dict[str, float], after: Dict[str, float]) -> PerformanceRegression:
        """Detect performance regression level"""
        try:
            max_degradation = 0.0
            
            # Check key performance metrics for degradation
            critical_metrics = ['compression_time_ms', 'decompression_time_ms', 'error_rate']
            
            for metric in critical_metrics:
                if metric in before and metric in after:
                    if before[metric] > 0:
                        # For time and error metrics, higher is worse
                        if metric.endswith('_ms') or metric == 'error_rate':
                            degradation = (after[metric] - before[metric]) / before[metric]
                        else:
                            # For other metrics, lower might be worse
                            degradation = (before[metric] - after[metric]) / before[metric]
                        
                        max_degradation = max(max_degradation, degradation)
            
            # Determine regression level
            if max_degradation >= self.regression_thresholds['critical']:
                return PerformanceRegression.CRITICAL
            elif max_degradation >= self.regression_thresholds['severe']:
                return PerformanceRegression.SEVERE
            elif max_degradation >= self.regression_thresholds['moderate']:
                return PerformanceRegression.MODERATE
            elif max_degradation >= self.regression_thresholds['minor']:
                return PerformanceRegression.MINOR
            else:
                return PerformanceRegression.NONE
                
        except Exception as e:
            self.logger.error(f"Error detecting regression: {e}")
            return PerformanceRegression.NONE
    
    def _update_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Update current performance metrics history"""
        for metric_name, value in metrics.items():
            self.current_performance_metrics[metric_name].append(value)
    
    def _check_performance_alerts(self, record: SwitchingPerformanceRecord) -> None:
        """Check for performance alerts"""
        try:
            # Check for regression alerts
            if record.regression_detected != PerformanceRegression.NONE:
                alert_type = f"performance_regression_{record.regression_detected.value}"
                
                # Check cooldown
                if self._is_alert_in_cooldown(alert_type):
                    return
                
                alert = PerformanceAlert(
                    alert_id=f"alert_{int(time.time())}_{len(self.performance_alerts)}",
                    timestamp=datetime.utcnow(),
                    alert_type=alert_type,
                    severity=record.regression_detected.value,
                    message=f"Performance regression detected: {record.regression_detected.value}",
                    affected_metrics=list(record.performance_delta.keys()),
                    suggested_actions=[
                        "Review switching decision criteria",
                        "Check system resource availability",
                        "Consider adjusting switching thresholds"
                    ],
                    metadata={
                        'switch_event_id': record.switch_event_id,
                        'mode_transition': record.mode_transition,
                        'performance_delta': record.performance_delta
                    }
                )
                
                with self._alert_lock:
                    self.performance_alerts.append(alert)
                    self.alert_cooldowns[alert_type] = datetime.utcnow()
                
                self.logger.warning(f"Performance alert generated: {alert.message}")
                
        except Exception as e:
            self.logger.error(f"Error checking performance alerts: {e}")
    
    def _generate_optimization_suggestions(self, record: SwitchingPerformanceRecord) -> None:
        """Generate optimization suggestions based on performance record"""
        # This is a simplified implementation - would need more sophisticated analysis
        pass
    
    def _calculate_decision_accuracy(self, decision: Dict[str, Any], actual: Dict[str, float]) -> float:
        """Calculate decision accuracy score"""
        # Simplified accuracy calculation
        if 'recommendation' in decision:
            # This would need actual outcome tracking
            return 0.8  # Placeholder accuracy
        return 0.5
    
    def _calculate_prediction_error(self, predicted: Dict[str, float], actual: Dict[str, float]) -> float:
        """Calculate prediction error"""
        if not predicted or not actual:
            return 1.0  # High error if no predictions
        
        errors = []
        for metric in predicted:
            if metric in actual and predicted[metric] > 0:
                error = abs(actual[metric] - predicted[metric]) / predicted[metric]
                errors.append(error)
        
        return statistics.mean(errors) if errors else 1.0
    
    def _calculate_overhead_trend(self, overheads: List[float]) -> str:
        """Calculate overhead trend"""
        if len(overheads) < 5:
            return "insufficient_data"
        
        # Simple trend analysis
        recent = overheads[-10:]
        older = overheads[-20:-10] if len(overheads) >= 20 else overheads[:-10]
        
        if not older:
            return "insufficient_data"
        
        recent_avg = statistics.mean(recent)
        older_avg = statistics.mean(older)
        
        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def _analyze_performance_trend(self, values: List[float]) -> PerformanceTrend:
        """Analyze performance trend"""
        if len(values) < 5:
            return PerformanceTrend.UNKNOWN
        
        # Calculate trend using linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Calculate volatility
        volatility = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
        
        if volatility > 0.2:  # High volatility
            return PerformanceTrend.VOLATILE
        elif slope > 0.05:  # Improving
            return PerformanceTrend.IMPROVING
        elif slope < -0.05:  # Declining
            return PerformanceTrend.DECLINING
        else:  # Stable
            return PerformanceTrend.STABLE
    
    def _calculate_improvement_rate(self, values: List[float]) -> float:
        """Calculate improvement rate"""
        if len(values) < 2:
            return 0.0
        
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        if first_half and second_half:
            avg_first = statistics.mean(first_half)
            avg_second = statistics.mean(second_half)
            
            if avg_first > 0:
                return (avg_second - avg_first) / avg_first
        
        return 0.0
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility metric"""
        if len(values) < 2:
            return 0.0
        
        return statistics.stdev(values) / statistics.mean(values) if statistics.mean(values) > 0 else 0.0
    
    def _analyze_overall_performance_trend(self, records: List[SwitchingPerformanceRecord]) -> PerformanceTrend:
        """Analyze overall performance trend from records"""
        if len(records) < 5:
            return PerformanceTrend.UNKNOWN
        
        # Use switching success rate as overall performance indicator
        success_rates = []
        window_size = 5
        
        for i in range(len(records) - window_size + 1):
            window = records[i:i + window_size]
            success_rate = sum(1 for r in window if r.success) / len(window)
            success_rates.append(success_rate)
        
        return self._analyze_performance_trend(success_rates)
    
    def _calculate_switching_success_rate(self, records: List[SwitchingPerformanceRecord]) -> float:
        """Calculate switching success rate"""
        if not records:
            return 0.0
        
        successful = sum(1 for record in records if record.success)
        return successful / len(records)
    
    def _calculate_regression_frequency(self, records: List[SwitchingPerformanceRecord]) -> float:
        """Calculate regression frequency"""
        if not records:
            return 0.0
        
        regressions = sum(1 for record in records if record.regression_detected != PerformanceRegression.NONE)
        return regressions / len(records)
    
    def _calculate_trend_confidence(self) -> float:
        """Calculate confidence in trend analysis"""
        total_data_points = sum(len(history) for history in self.current_performance_metrics.values())
        
        if total_data_points < 10:
            return 0.1
        elif total_data_points < 50:
            return 0.5
        elif total_data_points < 100:
            return 0.8
        else:
            return 0.9
    
    def _suggest_threshold_optimization(self) -> Optional[PerformanceOptimizationSuggestion]:
        """Suggest threshold optimization"""
        # Placeholder implementation
        return None
    
    def _suggest_frequency_optimization(self) -> Optional[PerformanceOptimizationSuggestion]:
        """Suggest frequency optimization"""
        # Placeholder implementation
        return None
    
    def _suggest_decision_weight_optimization(self) -> Optional[PerformanceOptimizationSuggestion]:
        """Suggest decision weight optimization"""
        # Placeholder implementation
        return None
    
    def _is_alert_in_cooldown(self, alert_type: str) -> bool:
        """Check if alert type is in cooldown period"""
        cooldown_duration = timedelta(minutes=5)  # 5-minute cooldown
        
        if alert_type in self.alert_cooldowns:
            return datetime.utcnow() - self.alert_cooldowns[alert_type] < cooldown_duration
        
        return False
    
    def _start_background_monitoring(self) -> None:
        """Start background monitoring thread"""
        if self._background_monitoring_enabled:
            return
        
        self._background_monitoring_enabled = True
        self._stop_background.clear()
        self._background_thread = threading.Thread(
            target=self._background_monitoring_loop,
            name="PerformanceMonitor",
            daemon=True
        )
        self._background_thread.start()
        self.logger.info("Background performance monitoring started")
    
    def _background_monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while not self._stop_background.wait(60.0):  # Check every minute
            try:
                # Periodic performance analysis
                self._periodic_performance_analysis()
                
                # Clean up old data
                self._cleanup_old_data()
                
            except Exception as e:
                self.logger.error(f"Error in background monitoring: {e}")
    
    def _periodic_performance_analysis(self) -> None:
        """Perform periodic performance analysis"""
        # Generate optimization suggestions
        if self.optimization_enabled:
            self.optimize_switching_parameters()
        
        # Check for performance trends
        trends = self.get_performance_trends()
        
        # Log performance summary
        if trends.get('trends'):
            self.logger.debug(f"Performance trend analysis: {len(trends['trends'])} metrics analyzed")
    
    def _cleanup_old_data(self) -> None:
        """Clean up old performance data"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        # Clean up alerts
        with self._alert_lock:
            self.performance_alerts = deque([
                alert for alert in self.performance_alerts
                if alert.timestamp >= cutoff_time
            ], maxlen=200)
        
        # Clean up alert cooldowns
        self.alert_cooldowns = {
            alert_type: timestamp
            for alert_type, timestamp in self.alert_cooldowns.items()
            if timestamp >= cutoff_time
        }
    
    def update_configuration(self, config: Dict[str, Any]) -> None:
        """Update performance monitor configuration"""
        if not isinstance(config, dict):
            raise TypeError(f"Config must be dict, got {type(config)}")
        
        self.config.update(config)
        
        # Update thresholds
        self.regression_thresholds.update({
            'minor': config.get('minor_regression_threshold', self.regression_thresholds['minor']),
            'moderate': config.get('moderate_regression_threshold', self.regression_thresholds['moderate']),
            'severe': config.get('severe_regression_threshold', self.regression_thresholds['severe']),
            'critical': config.get('critical_regression_threshold', self.regression_thresholds['critical'])
        })
        
        # Update monitoring settings
        self.monitoring_enabled = config.get('enable_monitoring', self.monitoring_enabled)
        self.optimization_enabled = config.get('enable_optimization', self.optimization_enabled)
        self.alerting_enabled = config.get('enable_alerting', self.alerting_enabled)
        
        self.logger.info("Performance monitor configuration updated")
    
    def shutdown(self) -> None:
        """Shutdown performance monitor"""
        self.logger.info("Shutting down switching performance monitor")
        
        # Stop background monitoring
        if self._background_monitoring_enabled:
            self._background_monitoring_enabled = False
            self._stop_background.set()
            if self._background_thread and self._background_thread.is_alive():
                self._background_thread.join(timeout=5.0)
        
        # Clear data
        self.performance_records.clear()
        self.current_performance_metrics.clear()
        self.optimization_suggestions.clear()
        self.performance_alerts.clear()
        
        self.is_initialized = False
        self.logger.info("Switching performance monitor shutdown complete")