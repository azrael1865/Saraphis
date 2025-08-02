"""
Strategy Recorder - Records and tracks proof strategy executions
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import threading
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class StrategyExecution:
    """Record of a single strategy execution"""
    strategy_name: str
    execution_id: str
    start_time: float
    end_time: float
    success: bool
    confidence: float
    proof_data: Dict[str, Any]
    context: Dict[str, Any]
    error_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # NO FALLBACKS - HARD FAILURES ONLY
        if not self.strategy_name or not isinstance(self.strategy_name, str):
            raise ValueError("Strategy name must be non-empty string")
        if not self.execution_id or not isinstance(self.execution_id, str):
            raise ValueError("Execution ID must be non-empty string")
        if self.start_time <= 0:
            raise ValueError("Start time must be positive")
        if self.end_time <= 0:
            raise ValueError("End time must be positive")
        if self.end_time < self.start_time:
            raise ValueError("End time must be after start time")
        if not isinstance(self.success, bool):
            raise TypeError("Success must be boolean")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not isinstance(self.proof_data, dict):
            raise TypeError("Proof data must be dict")
        if not isinstance(self.context, dict):
            raise TypeError("Context must be dict")
    
    @property
    def execution_time(self) -> float:
        """Get execution time in seconds"""
        return self.end_time - self.start_time


@dataclass
class StrategyMetrics:
    """Performance metrics for a strategy"""
    strategy_name: str
    total_executions: int = 0
    successful_executions: int = 0
    total_execution_time: float = 0.0
    total_confidence: float = 0.0
    error_counts: Dict[str, int] = field(default_factory=dict)
    confidence_history: List[float] = field(default_factory=list)
    execution_time_history: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        # NO FALLBACKS - HARD FAILURES ONLY
        if not self.strategy_name or not isinstance(self.strategy_name, str):
            raise ValueError("Strategy name must be non-empty string")
        if self.total_executions < 0:
            raise ValueError("Total executions must be non-negative")
        if self.successful_executions < 0:
            raise ValueError("Successful executions must be non-negative")
        if self.successful_executions > self.total_executions:
            raise ValueError("Successful executions cannot exceed total executions")
        if self.total_execution_time < 0:
            raise ValueError("Total execution time must be non-negative")
        if self.total_confidence < 0:
            raise ValueError("Total confidence must be non-negative")
    
    @property
    def success_rate(self) -> float:
        """Get success rate as percentage"""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions
    
    @property
    def average_confidence(self) -> float:
        """Get average confidence score"""
        if self.total_executions == 0:
            return 0.0
        return self.total_confidence / self.total_executions
    
    @property
    def average_execution_time(self) -> float:
        """Get average execution time"""
        if self.total_executions == 0:
            return 0.0
        return self.total_execution_time / self.total_executions
    
    def update_from_execution(self, execution: StrategyExecution):
        """Update metrics from a new execution"""
        if execution.strategy_name != self.strategy_name:
            raise ValueError(f"Strategy name mismatch: {execution.strategy_name} != {self.strategy_name}")
        
        self.total_executions += 1
        if execution.success:
            self.successful_executions += 1
        
        self.total_execution_time += execution.execution_time
        self.total_confidence += execution.confidence
        
        # Update error counts
        if execution.error_info:
            error_type = execution.error_info.get('type', 'unknown')
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Update histories
        self.confidence_history.append(execution.confidence)
        self.execution_time_history.append(execution.execution_time)
        
        # Limit history size
        max_history = 1000
        if len(self.confidence_history) > max_history:
            self.confidence_history = self.confidence_history[-max_history:]
        if len(self.execution_time_history) > max_history:
            self.execution_time_history = self.execution_time_history[-max_history:]


class TrendAnalyzer:
    """Analyzes trends in strategy performance"""
    
    def __init__(self, window_size: int = 50):
        if window_size <= 0:
            raise ValueError("Window size must be positive")
        self.window_size = window_size
    
    def analyze_confidence_trend(self, confidence_history: List[float]) -> Dict[str, Any]:
        """Analyze confidence trend"""
        if not confidence_history:
            return {'trend': 'unknown', 'slope': 0.0, 'confidence': 0.0}
        
        if len(confidence_history) < 2:
            return {'trend': 'stable', 'slope': 0.0, 'confidence': confidence_history[0]}
        
        # Use recent window
        recent_data = confidence_history[-self.window_size:]
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_data))
        y = np.array(recent_data)
        
        # Linear regression
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            slope = 0.0
        else:
            slope = numerator / denominator
        
        # Determine trend
        if abs(slope) < 0.001:
            trend = 'stable'
        elif slope > 0:
            trend = 'improving'
        else:
            trend = 'declining'
        
        # Calculate R-squared for trend confidence
        y_pred = x_mean + slope * (x - x_mean)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        
        if ss_tot == 0:
            r_squared = 1.0
        else:
            r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'trend': trend,
            'slope': slope,
            'confidence': max(0.0, r_squared),
            'recent_mean': y_mean,
            'data_points': len(recent_data)
        }
    
    def analyze_execution_time_trend(self, time_history: List[float]) -> Dict[str, Any]:
        """Analyze execution time trend"""
        if not time_history:
            return {'trend': 'unknown', 'slope': 0.0, 'mean_time': 0.0}
        
        if len(time_history) < 2:
            return {'trend': 'stable', 'slope': 0.0, 'mean_time': time_history[0]}
        
        # Use recent window
        recent_data = time_history[-self.window_size:]
        
        # Calculate trend
        x = np.arange(len(recent_data))
        y = np.array(recent_data)
        
        # Linear regression
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            slope = 0.0
        else:
            slope = numerator / denominator
        
        # Determine trend
        if abs(slope) < 0.01:  # 10ms threshold
            trend = 'stable'
        elif slope > 0:
            trend = 'slowing'
        else:
            trend = 'accelerating'
        
        return {
            'trend': trend,
            'slope': slope,
            'mean_time': y_mean,
            'std_time': np.std(recent_data),
            'data_points': len(recent_data)
        }


class AnomalyDetector:
    """Detects anomalous strategy executions"""
    
    def __init__(self, z_score_threshold: float = 3.0):
        if z_score_threshold <= 0:
            raise ValueError("Z-score threshold must be positive")
        self.z_score_threshold = z_score_threshold
    
    def detect_execution_anomalies(self, executions: List[StrategyExecution]) -> List[Dict[str, Any]]:
        """Detect anomalous executions"""
        if len(executions) < 10:  # Need enough data
            return []
        
        anomalies = []
        
        # Extract metrics
        execution_times = [e.execution_time for e in executions]
        confidences = [e.confidence for e in executions]
        
        # Calculate statistics
        time_mean = np.mean(execution_times)
        time_std = np.std(execution_times)
        conf_mean = np.mean(confidences)
        conf_std = np.std(confidences)
        
        # Detect anomalies
        for execution in executions:
            anomaly_reasons = []
            
            # Execution time anomaly
            if time_std > 0:
                time_z_score = abs(execution.execution_time - time_mean) / time_std
                if time_z_score > self.z_score_threshold:
                    anomaly_reasons.append(f"execution_time_outlier (z-score: {time_z_score:.2f})")
            
            # Confidence anomaly
            if conf_std > 0:
                conf_z_score = abs(execution.confidence - conf_mean) / conf_std
                if conf_z_score > self.z_score_threshold:
                    anomaly_reasons.append(f"confidence_outlier (z-score: {conf_z_score:.2f})")
            
            # Success/failure pattern anomaly
            if not execution.success and execution.confidence > 0.8:
                anomaly_reasons.append("high_confidence_failure")
            
            if execution.success and execution.confidence < 0.3:
                anomaly_reasons.append("low_confidence_success")
            
            if anomaly_reasons:
                anomalies.append({
                    'execution_id': execution.execution_id,
                    'strategy_name': execution.strategy_name,
                    'timestamp': execution.start_time,
                    'reasons': anomaly_reasons,
                    'execution_time': execution.execution_time,
                    'confidence': execution.confidence,
                    'success': execution.success
                })
        
        return anomalies


class PatternAnalyzer:
    """Analyzes patterns in strategy executions"""
    
    def __init__(self):
        pass
    
    def find_temporal_patterns(self, executions: List[StrategyExecution]) -> Dict[str, Any]:
        """Find temporal patterns in executions"""
        if not executions:
            return {'patterns': [], 'analysis': 'insufficient_data'}
        
        # Group by hour of day
        hourly_performance = defaultdict(list)
        for execution in executions:
            hour = int((execution.start_time % 86400) // 3600)  # Hour of day
            hourly_performance[hour].append(execution.success)
        
        # Calculate hourly success rates
        hourly_success_rates = {}
        for hour, successes in hourly_performance.items():
            if successes:
                hourly_success_rates[hour] = sum(successes) / len(successes)
        
        # Find peak and trough hours
        if hourly_success_rates:
            best_hour = max(hourly_success_rates.items(), key=lambda x: x[1])
            worst_hour = min(hourly_success_rates.items(), key=lambda x: x[1])
            
            patterns = []
            if best_hour[1] - worst_hour[1] > 0.1:  # Significant difference
                patterns.append({
                    'type': 'temporal_performance_variation',
                    'best_hour': best_hour[0],
                    'best_success_rate': best_hour[1],
                    'worst_hour': worst_hour[0],
                    'worst_success_rate': worst_hour[1]
                })
            
            return {
                'patterns': patterns,
                'hourly_success_rates': hourly_success_rates,
                'analysis': 'completed'
            }
        
        return {'patterns': [], 'analysis': 'no_patterns_found'}
    
    def find_failure_patterns(self, executions: List[StrategyExecution]) -> Dict[str, Any]:
        """Find patterns in failures"""
        failures = [e for e in executions if not e.success]
        
        if not failures:
            return {'patterns': [], 'analysis': 'no_failures'}
        
        patterns = []
        
        # Error type clustering
        error_types = defaultdict(int)
        for failure in failures:
            if failure.error_info:
                error_type = failure.error_info.get('type', 'unknown')
                error_types[error_type] += 1
        
        if error_types:
            most_common_error = max(error_types.items(), key=lambda x: x[1])
            if most_common_error[1] > len(failures) * 0.3:  # More than 30% of failures
                patterns.append({
                    'type': 'dominant_error_type',
                    'error_type': most_common_error[0],
                    'frequency': most_common_error[1],
                    'percentage': most_common_error[1] / len(failures)
                })
        
        # Context-based failure patterns
        context_failures = defaultdict(int)
        for failure in failures:
            context_keys = list(failure.context.keys())
            context_signature = tuple(sorted(context_keys))
            context_failures[context_signature] += 1
        
        for context_sig, count in context_failures.items():
            if count > len(failures) * 0.2:  # More than 20% of failures
                patterns.append({
                    'type': 'context_failure_pattern',
                    'context_signature': context_sig,
                    'frequency': count,
                    'percentage': count / len(failures)
                })
        
        return {
            'patterns': patterns,
            'total_failures': len(failures),
            'error_distribution': dict(error_types),
            'analysis': 'completed'
        }
    
    def find_performance_patterns(self, executions: List[StrategyExecution]) -> Dict[str, Any]:
        """Find performance patterns"""
        if len(executions) < 10:
            return {'patterns': [], 'analysis': 'insufficient_data'}
        
        patterns = []
        
        # Confidence vs execution time correlation
        confidences = [e.confidence for e in executions]
        exec_times = [e.execution_time for e in executions]
        
        # Calculate correlation
        if len(confidences) > 1 and len(exec_times) > 1:
            correlation = np.corrcoef(confidences, exec_times)[0, 1]
            
            if abs(correlation) > 0.5:  # Strong correlation
                patterns.append({
                    'type': 'confidence_time_correlation',
                    'correlation': correlation,
                    'interpretation': 'positive' if correlation > 0 else 'negative',
                    'strength': 'strong' if abs(correlation) > 0.7 else 'moderate'
                })
        
        # Success streaks
        success_sequence = [e.success for e in executions]
        current_streak = 0
        max_success_streak = 0
        max_failure_streak = 0
        current_success_streak = 0
        current_failure_streak = 0
        
        for success in success_sequence:
            if success:
                current_success_streak += 1
                current_failure_streak = 0
                max_success_streak = max(max_success_streak, current_success_streak)
            else:
                current_failure_streak += 1
                current_success_streak = 0
                max_failure_streak = max(max_failure_streak, current_failure_streak)
        
        if max_success_streak > 5 or max_failure_streak > 3:
            patterns.append({
                'type': 'execution_streaks',
                'max_success_streak': max_success_streak,
                'max_failure_streak': max_failure_streak,
                'current_success_streak': current_success_streak,
                'current_failure_streak': current_failure_streak
            })
        
        return {
            'patterns': patterns,
            'analysis': 'completed',
            'total_executions': len(executions)
        }


class StrategyRecorder:
    """Records and tracks proof strategy executions"""
    
    def __init__(self, max_executions: int = 10000):
        # NO FALLBACKS - HARD FAILURES ONLY
        if max_executions <= 0:
            raise ValueError("Max executions must be positive")
        
        self.max_executions = max_executions
        self.executions: deque[StrategyExecution] = deque(maxlen=max_executions)
        self.strategy_metrics: Dict[str, StrategyMetrics] = {}
        self.execution_lock = threading.RLock()
        
        # Analyzers
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.pattern_analyzer = PatternAnalyzer()
        
        # Performance tracking
        self.total_recordings = 0
        self.last_cleanup_time = time.time()
    
    def record_execution(self, execution: StrategyExecution) -> None:
        """Record a strategy execution"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if execution is None:
            raise ValueError("Execution cannot be None")
        if not isinstance(execution, StrategyExecution):
            raise TypeError("Execution must be StrategyExecution instance")
        
        with self.execution_lock:
            # Add to executions
            self.executions.append(execution)
            
            # Update strategy metrics
            if execution.strategy_name not in self.strategy_metrics:
                self.strategy_metrics[execution.strategy_name] = StrategyMetrics(execution.strategy_name)
            
            self.strategy_metrics[execution.strategy_name].update_from_execution(execution)
            
            # Update counters
            self.total_recordings += 1
            
            # Periodic cleanup
            current_time = time.time()
            if current_time - self.last_cleanup_time > 3600:  # Every hour
                self._cleanup_old_data()
                self.last_cleanup_time = current_time
    
    def get_strategy_metrics(self, strategy_name: str) -> Optional[StrategyMetrics]:
        """Get metrics for a specific strategy"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not strategy_name or not isinstance(strategy_name, str):
            raise ValueError("Strategy name must be non-empty string")
        
        with self.execution_lock:
            return self.strategy_metrics.get(strategy_name)
    
    def get_all_strategy_metrics(self) -> Dict[str, StrategyMetrics]:
        """Get metrics for all strategies"""
        with self.execution_lock:
            return self.strategy_metrics.copy()
    
    def get_executions_by_strategy(self, strategy_name: str) -> List[StrategyExecution]:
        """Get all executions for a specific strategy"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not strategy_name or not isinstance(strategy_name, str):
            raise ValueError("Strategy name must be non-empty string")
        
        with self.execution_lock:
            return [e for e in self.executions if e.strategy_name == strategy_name]
    
    def get_recent_executions(self, count: int = 100) -> List[StrategyExecution]:
        """Get recent executions"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if count <= 0:
            raise ValueError("Count must be positive")
        
        with self.execution_lock:
            return list(self.executions)[-count:]
    
    def analyze_strategy_trends(self, strategy_name: str) -> Dict[str, Any]:
        """Analyze trends for a specific strategy"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not strategy_name or not isinstance(strategy_name, str):
            raise ValueError("Strategy name must be non-empty string")
        
        executions = self.get_executions_by_strategy(strategy_name)
        if not executions:
            raise ValueError(f"No executions found for strategy {strategy_name}")
        
        metrics = self.strategy_metrics[strategy_name]
        
        # Analyze trends
        confidence_trend = self.trend_analyzer.analyze_confidence_trend(metrics.confidence_history)
        time_trend = self.trend_analyzer.analyze_execution_time_trend(metrics.execution_time_history)
        
        return {
            'strategy_name': strategy_name,
            'confidence_trend': confidence_trend,
            'execution_time_trend': time_trend,
            'current_metrics': {
                'success_rate': metrics.success_rate,
                'average_confidence': metrics.average_confidence,
                'average_execution_time': metrics.average_execution_time,
                'total_executions': metrics.total_executions
            }
        }
    
    def detect_anomalies(self, strategy_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect anomalous executions"""
        with self.execution_lock:
            if strategy_name:
                if not isinstance(strategy_name, str):
                    raise TypeError("Strategy name must be string")
                executions = self.get_executions_by_strategy(strategy_name)
            else:
                executions = list(self.executions)
        
        return self.anomaly_detector.detect_execution_anomalies(executions)
    
    def analyze_patterns(self, strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """Analyze execution patterns"""
        with self.execution_lock:
            if strategy_name:
                if not isinstance(strategy_name, str):
                    raise TypeError("Strategy name must be string")
                executions = self.get_executions_by_strategy(strategy_name)
            else:
                executions = list(self.executions)
        
        temporal_patterns = self.pattern_analyzer.find_temporal_patterns(executions)
        failure_patterns = self.pattern_analyzer.find_failure_patterns(executions)
        performance_patterns = self.pattern_analyzer.find_performance_patterns(executions)
        
        return {
            'temporal_patterns': temporal_patterns,
            'failure_patterns': failure_patterns,
            'performance_patterns': performance_patterns,
            'total_executions_analyzed': len(executions)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        with self.execution_lock:
            if not self.strategy_metrics:
                return {
                    'total_strategies': 0,
                    'total_executions': 0,
                    'overall_success_rate': 0.0,
                    'strategies': {}
                }
            
            total_executions = sum(m.total_executions for m in self.strategy_metrics.values())
            total_successes = sum(m.successful_executions for m in self.strategy_metrics.values())
            overall_success_rate = total_successes / total_executions if total_executions > 0 else 0.0
            
            strategy_summaries = {}
            for name, metrics in self.strategy_metrics.items():
                strategy_summaries[name] = {
                    'success_rate': metrics.success_rate,
                    'average_confidence': metrics.average_confidence,
                    'average_execution_time': metrics.average_execution_time,
                    'total_executions': metrics.total_executions,
                    'error_types': list(metrics.error_counts.keys())
                }
            
            return {
                'total_strategies': len(self.strategy_metrics),
                'total_executions': total_executions,
                'overall_success_rate': overall_success_rate,
                'total_recordings': self.total_recordings,
                'strategies': strategy_summaries
            }
    
    def _cleanup_old_data(self) -> None:
        """Clean up old data to maintain performance"""
        # This is called automatically during record_execution
        # The deque automatically handles size limits
        
        # Clean up metrics history if too large
        for metrics in self.strategy_metrics.values():
            max_history = 1000
            if len(metrics.confidence_history) > max_history:
                metrics.confidence_history = metrics.confidence_history[-max_history:]
            if len(metrics.execution_time_history) > max_history:
                metrics.execution_time_history = metrics.execution_time_history[-max_history:]
    
    def reset_metrics(self, strategy_name: Optional[str] = None) -> None:
        """Reset metrics for a strategy or all strategies"""
        with self.execution_lock:
            if strategy_name:
                if not isinstance(strategy_name, str):
                    raise TypeError("Strategy name must be string")
                if strategy_name in self.strategy_metrics:
                    self.strategy_metrics[strategy_name] = StrategyMetrics(strategy_name)
                else:
                    raise ValueError(f"Strategy {strategy_name} not found")
            else:
                # Reset all
                for name in self.strategy_metrics:
                    self.strategy_metrics[name] = StrategyMetrics(name)
                self.executions.clear()
                self.total_recordings = 0
    
    def export_data(self, strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """Export recorded data"""
        with self.execution_lock:
            if strategy_name:
                if not isinstance(strategy_name, str):
                    raise TypeError("Strategy name must be string")
                executions = self.get_executions_by_strategy(strategy_name)
                metrics = self.strategy_metrics.get(strategy_name)
            else:
                executions = list(self.executions)
                metrics = self.strategy_metrics
            
            # Convert executions to serializable format
            execution_data = []
            for execution in executions:
                execution_data.append({
                    'strategy_name': execution.strategy_name,
                    'execution_id': execution.execution_id,
                    'start_time': execution.start_time,
                    'end_time': execution.end_time,
                    'execution_time': execution.execution_time,
                    'success': execution.success,
                    'confidence': execution.confidence,
                    'proof_data': execution.proof_data,
                    'context': execution.context,
                    'error_info': execution.error_info,
                    'metadata': execution.metadata
                })
            
            return {
                'executions': execution_data,
                'metrics': metrics if strategy_name else {name: vars(m) for name, m in metrics.items()},
                'export_timestamp': time.time(),
                'total_executions': len(executions)
            }