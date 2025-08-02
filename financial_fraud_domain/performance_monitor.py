"""
Performance Monitor for Financial Fraud Detection
Monitoring and metrics for fraud detection system
"""

import logging
import time
import json
import threading
import queue
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import statistics
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import psutil
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PerformanceStatus(Enum):
    """System performance status"""
    OPTIMAL = "optimal"
    NORMAL = "normal"
    DEGRADED = "degraded"
    CRITICAL = "critical"


@dataclass
class MetricData:
    """Individual metric data point"""
    name: str
    value: float
    timestamp: datetime
    metric_type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert"""
    alert_id: str
    severity: AlertSeverity
    metric_name: str
    message: str
    timestamp: datetime
    threshold_value: float
    actual_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Performance report"""
    report_id: str
    start_time: datetime
    end_time: datetime
    metrics_summary: Dict[str, Dict[str, float]]
    alerts: List[PerformanceAlert]
    recommendations: List[str]
    status: PerformanceStatus
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricAggregator:
    """Aggregates metrics over time windows"""
    
    def __init__(self, window_size: int = 60):
        """
        Initialize metric aggregator.
        
        Args:
            window_size: Window size in seconds
        """
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.RLock()
    
    def add_metric(self, metric: MetricData):
        """Add metric to aggregator"""
        with self._lock:
            key = self._get_metric_key(metric)
            self.metrics[key].append((metric.timestamp, metric.value))
    
    def get_aggregated(self, 
                      metric_name: str,
                      aggregation_type: str = 'mean',
                      window: Optional[int] = None) -> Optional[float]:
        """
        Get aggregated metric value.
        
        Args:
            metric_name: Name of metric
            aggregation_type: Type of aggregation (mean, sum, max, min, p50, p95, p99)
            window: Time window in seconds
            
        Returns:
            Aggregated value or None
        """
        window = window or self.window_size
        cutoff_time = datetime.now() - timedelta(seconds=window)
        
        with self._lock:
            values = []
            for key, data_points in self.metrics.items():
                if metric_name in key:
                    for timestamp, value in data_points:
                        if timestamp >= cutoff_time:
                            values.append(value)
            
            if not values:
                return None
            
            if aggregation_type == 'mean':
                return statistics.mean(values)
            elif aggregation_type == 'sum':
                return sum(values)
            elif aggregation_type == 'max':
                return max(values)
            elif aggregation_type == 'min':
                return min(values)
            elif aggregation_type == 'p50':
                return np.percentile(values, 50)
            elif aggregation_type == 'p95':
                return np.percentile(values, 95)
            elif aggregation_type == 'p99':
                return np.percentile(values, 99)
            else:
                return statistics.mean(values)
    
    def _get_metric_key(self, metric: MetricData) -> str:
        """Get unique key for metric"""
        tags_str = '_'.join(f"{k}={v}" for k, v in sorted(metric.tags.items()))
        return f"{metric.name}_{metric.metric_type.value}_{tags_str}"


class FinancialPerformanceMonitor:
    """
    Performance monitoring for Financial Fraud Detection system.
    
    Tracks metrics, generates reports, and provides performance insights.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize performance monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config = self._load_config(config)
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.aggregator = MetricAggregator(
            window_size=self.config.get('aggregation_window', 60)
        )
        
        # Performance tracking
        self.start_time = time.time()
        self.metric_counters = defaultdict(int)
        self.last_reset = datetime.now()
        
        # Alert management
        self.alerts = deque(maxlen=self.config.get('max_alerts', 1000))
        self.alert_handlers = []
        self.alert_thresholds = self._load_alert_thresholds()
        
        # Reporting
        self.reports = deque(maxlen=self.config.get('max_reports', 100))
        self.report_generators = self._initialize_report_generators()
        
        # System monitoring
        self.system_monitor = SystemResourceMonitor()
        
        # Background monitoring
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._monitoring_queue = queue.Queue()
        
        # Storage
        self.storage_path = Path(self.config.get('storage_path', 'performance_data'))
        self.storage_path.mkdir(exist_ok=True)
        
        # Start background monitoring if enabled
        if self.config.get('enable_background_monitoring', True):
            self._start_background_monitoring()
        
        self.logger.info("FinancialPerformanceMonitor initialized successfully")
    
    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            'aggregation_window': 60,  # seconds
            'max_alerts': 1000,
            'max_reports': 100,
            'enable_background_monitoring': True,
            'monitoring_interval': 10,  # seconds
            'storage_path': 'performance_data',
            'alert_cooldown': 300,  # seconds
            'metric_retention_days': 7,
            'enable_system_monitoring': True,
            'enable_ml_metrics': True,
            'enable_api_metrics': True
        }
        
        if config:
            default_config.update(config)
        
        return default_config
    
    def _load_alert_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Load alert thresholds"""
        return {
            # Processing performance
            'avg_processing_time_ms': {
                'warning': 100,
                'error': 500,
                'critical': 1000,
                'aggregation': 'p95'
            },
            'fraud_detection_latency_ms': {
                'warning': 200,
                'error': 1000,
                'critical': 5000,
                'aggregation': 'p99'
            },
            
            # Accuracy metrics
            'false_positive_rate': {
                'warning': 0.05,
                'error': 0.10,
                'critical': 0.20,
                'aggregation': 'mean'
            },
            'false_negative_rate': {
                'warning': 0.02,
                'error': 0.05,
                'critical': 0.10,
                'aggregation': 'mean'
            },
            
            # System resources
            'cpu_usage_percent': {
                'warning': 70,
                'error': 85,
                'critical': 95,
                'aggregation': 'mean'
            },
            'memory_usage_percent': {
                'warning': 70,
                'error': 85,
                'critical': 95,
                'aggregation': 'mean'
            },
            
            # Error rates
            'error_rate': {
                'warning': 0.01,
                'error': 0.05,
                'critical': 0.10,
                'aggregation': 'mean'
            },
            'api_timeout_rate': {
                'warning': 0.01,
                'error': 0.05,
                'critical': 0.10,
                'aggregation': 'mean'
            }
        }
    
    def _initialize_report_generators(self) -> Dict[str, Callable]:
        """Initialize report generation functions"""
        return {
            'performance_summary': self._generate_performance_summary,
            'accuracy_report': self._generate_accuracy_report,
            'resource_utilization': self._generate_resource_report,
            'alert_summary': self._generate_alert_summary
        }
    
    # Main monitoring methods
    
    def monitor_performance(self, 
                          operation_name: str,
                          duration_ms: float,
                          success: bool = True,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Monitor performance of an operation.
        
        Args:
            operation_name: Name of the operation
            duration_ms: Duration in milliseconds
            success: Whether operation succeeded
            metadata: Additional metadata
        """
        try:
            # Record timing metric
            self.record_metric(
                f"{operation_name}_duration_ms",
                duration_ms,
                MetricType.TIMER,
                tags={'success': str(success).lower()},
                metadata=metadata
            )
            
            # Record success/failure
            if success:
                self.record_metric(f"{operation_name}_success", 1, MetricType.COUNTER)
            else:
                self.record_metric(f"{operation_name}_failure", 1, MetricType.COUNTER)
            
            # Check for performance degradation
            if duration_ms > self._get_expected_duration(operation_name) * 2:
                self.logger.warning(
                    f"Performance degradation detected for {operation_name}: "
                    f"{duration_ms:.2f}ms (expected < {self._get_expected_duration(operation_name):.2f}ms)"
                )
            
        except Exception as e:
            self.logger.error(f"Error monitoring performance: {e}", exc_info=True)
    
    def track_metrics(self, metrics_batch: List[Dict[str, Any]]) -> None:
        """
        Track a batch of metrics.
        
        Args:
            metrics_batch: List of metric dictionaries
        """
        try:
            for metric_data in metrics_batch:
                self.record_metric(
                    metric_data['name'],
                    metric_data['value'],
                    MetricType(metric_data.get('type', 'gauge')),
                    tags=metric_data.get('tags', {}),
                    metadata=metric_data.get('metadata', {})
                )
            
            self.logger.debug(f"Tracked {len(metrics_batch)} metrics")
            
        except Exception as e:
            self.logger.error(f"Error tracking metrics batch: {e}", exc_info=True)
    
    def record_metric(self,
                     name: str,
                     value: float,
                     metric_type: MetricType = MetricType.GAUGE,
                     tags: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            tags: Metric tags
            metadata: Additional metadata
        """
        try:
            metric = MetricData(
                name=name,
                value=value,
                timestamp=datetime.now(),
                metric_type=metric_type,
                tags=tags or {},
                metadata=metadata or {}
            )
            
            # Store metric
            self.metrics[name].append(metric)
            self.aggregator.add_metric(metric)
            self.metric_counters[name] += 1
            
            # Queue for background processing
            if self._monitoring_queue:
                self._monitoring_queue.put(('metric', metric))
            
            # Check thresholds
            self._check_alert_thresholds(metric)
            
            self.logger.debug(f"Recorded metric {name}: {value}")
            
        except Exception as e:
            self.logger.error(f"Error recording metric: {e}", exc_info=True)
    
    def generate_reports(self, 
                        report_types: Optional[List[str]] = None,
                        time_range: Optional[Tuple[datetime, datetime]] = None) -> List[PerformanceReport]:
        """
        Generate performance reports.
        
        Args:
            report_types: Types of reports to generate
            time_range: Time range for reports
            
        Returns:
            List of performance reports
        """
        try:
            if not report_types:
                report_types = list(self.report_generators.keys())
            
            if not time_range:
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=1)
                time_range = (start_time, end_time)
            
            reports = []
            
            for report_type in report_types:
                if report_type in self.report_generators:
                    report = self.report_generators[report_type](time_range)
                    reports.append(report)
                    self.reports.append(report)
                    self.logger.info(f"Generated {report_type} report")
            
            return reports
            
        except Exception as e:
            self.logger.error(f"Error generating reports: {e}", exc_info=True)
            return []
    
    # Alert management
    
    def add_alert_handler(self, handler: Callable[[PerformanceAlert], None]) -> None:
        """Add alert handler"""
        self.alert_handlers.append(handler)
        self.logger.info(f"Added alert handler: {handler.__name__}")
    
    def _check_alert_thresholds(self, metric: MetricData) -> None:
        """Check if metric exceeds alert thresholds"""
        if metric.name not in self.alert_thresholds:
            return
        
        thresholds = self.alert_thresholds[metric.name]
        aggregation = thresholds.get('aggregation', 'mean')
        
        # Get aggregated value
        current_value = self.aggregator.get_aggregated(
            metric.name,
            aggregation,
            window=60
        )
        
        if current_value is None:
            return
        
        # Check thresholds
        severity = None
        threshold_value = None
        
        if current_value >= thresholds.get('critical', float('inf')):
            severity = AlertSeverity.CRITICAL
            threshold_value = thresholds['critical']
        elif current_value >= thresholds.get('error', float('inf')):
            severity = AlertSeverity.ERROR
            threshold_value = thresholds['error']
        elif current_value >= thresholds.get('warning', float('inf')):
            severity = AlertSeverity.WARNING
            threshold_value = thresholds['warning']
        
        if severity:
            self._create_alert(
                metric.name,
                severity,
                threshold_value,
                current_value,
                f"{metric.name} exceeded {severity.value} threshold"
            )
    
    def _create_alert(self,
                     metric_name: str,
                     severity: AlertSeverity,
                     threshold_value: float,
                     actual_value: float,
                     message: str) -> None:
        """Create and process alert"""
        # Check cooldown
        if self._is_alert_in_cooldown(metric_name, severity):
            return
        
        alert = PerformanceAlert(
            alert_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{metric_name}",
            severity=severity,
            metric_name=metric_name,
            message=message,
            timestamp=datetime.now(),
            threshold_value=threshold_value,
            actual_value=actual_value
        )
        
        self.alerts.append(alert)
        
        # Process alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}", exc_info=True)
        
        self.logger.warning(
            f"Alert created: {severity.value} - {metric_name}: "
            f"{actual_value:.2f} (threshold: {threshold_value:.2f})"
        )
    
    def _is_alert_in_cooldown(self, metric_name: str, severity: AlertSeverity) -> bool:
        """Check if alert is in cooldown period"""
        cooldown = timedelta(seconds=self.config.get('alert_cooldown', 300))
        cutoff_time = datetime.now() - cooldown
        
        for alert in reversed(self.alerts):
            if (alert.metric_name == metric_name and 
                alert.severity == severity and
                alert.timestamp > cutoff_time):
                return True
        
        return False
    
    # Report generation methods
    
    def _generate_performance_summary(self, 
                                    time_range: Tuple[datetime, datetime]) -> PerformanceReport:
        """Generate performance summary report"""
        metrics_summary = {
            'processing_times': {
                'avg_ms': self.aggregator.get_aggregated('fraud_detection_duration_ms', 'mean') or 0,
                'p95_ms': self.aggregator.get_aggregated('fraud_detection_duration_ms', 'p95') or 0,
                'p99_ms': self.aggregator.get_aggregated('fraud_detection_duration_ms', 'p99') or 0
            },
            'throughput': {
                'requests_per_second': self._calculate_throughput('fraud_detection_success'),
                'transactions_per_second': self._calculate_throughput('transaction_processed')
            },
            'success_rates': {
                'overall': self._calculate_success_rate('fraud_detection'),
                'api_requests': self._calculate_success_rate('api_request')
            }
        }
        
        # Get recent alerts
        recent_alerts = [
            alert for alert in self.alerts
            if time_range[0] <= alert.timestamp <= time_range[1]
        ]
        
        # Determine status
        status = self._determine_system_status(metrics_summary, recent_alerts)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics_summary, recent_alerts)
        
        return PerformanceReport(
            report_id=f"perf_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=time_range[0],
            end_time=time_range[1],
            metrics_summary=metrics_summary,
            alerts=recent_alerts,
            recommendations=recommendations,
            status=status
        )
    
    def _generate_accuracy_report(self, 
                                time_range: Tuple[datetime, datetime]) -> PerformanceReport:
        """Generate accuracy metrics report"""
        metrics_summary = {
            'detection_accuracy': {
                'true_positive_rate': self._calculate_detection_rate('true_positive'),
                'false_positive_rate': self._calculate_detection_rate('false_positive'),
                'true_negative_rate': self._calculate_detection_rate('true_negative'),
                'false_negative_rate': self._calculate_detection_rate('false_negative')
            },
            'model_performance': {
                'precision': self._calculate_precision(),
                'recall': self._calculate_recall(),
                'f1_score': self._calculate_f1_score()
            },
            'confidence_metrics': {
                'avg_confidence': self.aggregator.get_aggregated('model_confidence', 'mean') or 0,
                'low_confidence_rate': self._calculate_low_confidence_rate()
            }
        }
        
        recent_alerts = [
            alert for alert in self.alerts
            if time_range[0] <= alert.timestamp <= time_range[1] and
            'accuracy' in alert.metric_name.lower()
        ]
        
        recommendations = []
        
        # Check accuracy metrics
        if metrics_summary['detection_accuracy']['false_positive_rate'] > 0.05:
            recommendations.append("High false positive rate detected. Consider adjusting detection thresholds.")
        
        if metrics_summary['model_performance']['precision'] < 0.90:
            recommendations.append("Model precision below target. Review training data and feature engineering.")
        
        return PerformanceReport(
            report_id=f"accuracy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=time_range[0],
            end_time=time_range[1],
            metrics_summary=metrics_summary,
            alerts=recent_alerts,
            recommendations=recommendations,
            status=self._determine_accuracy_status(metrics_summary)
        )
    
    def _generate_resource_report(self, 
                                time_range: Tuple[datetime, datetime]) -> PerformanceReport:
        """Generate resource utilization report"""
        # Get system metrics
        system_metrics = self.system_monitor.get_current_metrics()
        
        metrics_summary = {
            'cpu_utilization': {
                'current_percent': system_metrics.get('cpu_percent', 0),
                'avg_percent': self.aggregator.get_aggregated('cpu_usage_percent', 'mean') or 0,
                'peak_percent': self.aggregator.get_aggregated('cpu_usage_percent', 'max') or 0
            },
            'memory_utilization': {
                'current_percent': system_metrics.get('memory_percent', 0),
                'avg_percent': self.aggregator.get_aggregated('memory_usage_percent', 'mean') or 0,
                'peak_percent': self.aggregator.get_aggregated('memory_usage_percent', 'max') or 0,
                'used_mb': system_metrics.get('memory_used_mb', 0)
            },
            'disk_utilization': {
                'current_percent': system_metrics.get('disk_percent', 0),
                'read_mb_s': system_metrics.get('disk_read_mb_s', 0),
                'write_mb_s': system_metrics.get('disk_write_mb_s', 0)
            }
        }
        
        recent_alerts = [
            alert for alert in self.alerts
            if time_range[0] <= alert.timestamp <= time_range[1] and
            any(resource in alert.metric_name for resource in ['cpu', 'memory', 'disk'])
        ]
        
        recommendations = self._generate_resource_recommendations(metrics_summary)
        
        return PerformanceReport(
            report_id=f"resource_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=time_range[0],
            end_time=time_range[1],
            metrics_summary=metrics_summary,
            alerts=recent_alerts,
            recommendations=recommendations,
            status=self._determine_resource_status(metrics_summary)
        )
    
    def _generate_alert_summary(self, 
                              time_range: Tuple[datetime, datetime]) -> PerformanceReport:
        """Generate alert summary report"""
        recent_alerts = [
            alert for alert in self.alerts
            if time_range[0] <= alert.timestamp <= time_range[1]
        ]
        
        # Group alerts by severity
        alerts_by_severity = defaultdict(list)
        for alert in recent_alerts:
            alerts_by_severity[alert.severity.value].append(alert)
        
        # Group alerts by metric
        alerts_by_metric = defaultdict(list)
        for alert in recent_alerts:
            alerts_by_metric[alert.metric_name].append(alert)
        
        metrics_summary = {
            'alert_counts': {
                'total': len(recent_alerts),
                'critical': len(alerts_by_severity['critical']),
                'error': len(alerts_by_severity['error']),
                'warning': len(alerts_by_severity['warning']),
                'info': len(alerts_by_severity['info'])
            },
            'top_alerting_metrics': {
                metric: len(alerts) 
                for metric, alerts in sorted(
                    alerts_by_metric.items(), 
                    key=lambda x: len(x[1]), 
                    reverse=True
                )[:5]
            },
            'alert_rate': {
                'alerts_per_hour': len(recent_alerts) / max(
                    (time_range[1] - time_range[0]).total_seconds() / 3600, 1
                )
            }
        }
        
        recommendations = []
        
        if metrics_summary['alert_counts']['critical'] > 0:
            recommendations.append("Critical alerts detected. Immediate investigation required.")
        
        if metrics_summary['alert_rate']['alerts_per_hour'] > 10:
            recommendations.append("High alert rate detected. Review alert thresholds and system health.")
        
        return PerformanceReport(
            report_id=f"alert_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=time_range[0],
            end_time=time_range[1],
            metrics_summary=metrics_summary,
            alerts=recent_alerts[:20],  # Limit to recent 20
            recommendations=recommendations,
            status=self._determine_alert_status(metrics_summary)
        )
    
    # Helper methods
    
    def _get_expected_duration(self, operation_name: str) -> float:
        """Get expected duration for operation in ms"""
        expected_durations = {
            'fraud_detection': 50,
            'batch_processing': 200,
            'model_inference': 20,
            'api_request': 100,
            'database_query': 30
        }
        
        return expected_durations.get(operation_name, 100)
    
    def _calculate_throughput(self, metric_base: str) -> float:
        """Calculate throughput in operations per second"""
        count = self.aggregator.get_aggregated(f"{metric_base}_count", 'sum', window=60)
        if count:
            return count / 60.0
        return 0.0
    
    def _calculate_success_rate(self, operation_base: str) -> float:
        """Calculate success rate for operation"""
        success_count = self.aggregator.get_aggregated(f"{operation_base}_success", 'sum', window=300) or 0
        failure_count = self.aggregator.get_aggregated(f"{operation_base}_failure", 'sum', window=300) or 0
        
        total = success_count + failure_count
        if total > 0:
            return success_count / total
        return 1.0
    
    def _calculate_detection_rate(self, detection_type: str) -> float:
        """Calculate detection rate"""
        detection_count = self.aggregator.get_aggregated(f"{detection_type}_count", 'sum', window=3600) or 0
        total_count = self.aggregator.get_aggregated('total_detections', 'sum', window=3600) or 1
        
        return detection_count / total_count
    
    def _calculate_precision(self) -> float:
        """Calculate precision metric"""
        tp = self.aggregator.get_aggregated('true_positive_count', 'sum', window=3600) or 0
        fp = self.aggregator.get_aggregated('false_positive_count', 'sum', window=3600) or 0
        
        if tp + fp > 0:
            return tp / (tp + fp)
        return 0.0
    
    def _calculate_recall(self) -> float:
        """Calculate recall metric"""
        tp = self.aggregator.get_aggregated('true_positive_count', 'sum', window=3600) or 0
        fn = self.aggregator.get_aggregated('false_negative_count', 'sum', window=3600) or 0
        
        if tp + fn > 0:
            return tp / (tp + fn)
        return 0.0
    
    def _calculate_f1_score(self) -> float:
        """Calculate F1 score"""
        precision = self._calculate_precision()
        recall = self._calculate_recall()
        
        if precision + recall > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0.0
    
    def _calculate_low_confidence_rate(self) -> float:
        """Calculate rate of low confidence predictions"""
        low_conf_count = self.aggregator.get_aggregated('low_confidence_count', 'sum', window=3600) or 0
        total_count = self.aggregator.get_aggregated('total_predictions', 'sum', window=3600) or 1
        
        return low_conf_count / total_count
    
    def _determine_system_status(self, 
                               metrics_summary: Dict[str, Any],
                               alerts: List[PerformanceAlert]) -> PerformanceStatus:
        """Determine overall system status"""
        critical_alerts = sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)
        error_alerts = sum(1 for a in alerts if a.severity == AlertSeverity.ERROR)
        
        if critical_alerts > 0:
            return PerformanceStatus.CRITICAL
        elif error_alerts > 2:
            return PerformanceStatus.DEGRADED
        elif error_alerts > 0:
            return PerformanceStatus.NORMAL
        else:
            return PerformanceStatus.OPTIMAL
    
    def _determine_accuracy_status(self, metrics_summary: Dict[str, Any]) -> PerformanceStatus:
        """Determine accuracy status"""
        f1_score = metrics_summary['model_performance']['f1_score']
        
        if f1_score >= 0.95:
            return PerformanceStatus.OPTIMAL
        elif f1_score >= 0.90:
            return PerformanceStatus.NORMAL
        elif f1_score >= 0.80:
            return PerformanceStatus.DEGRADED
        else:
            return PerformanceStatus.CRITICAL
    
    def _determine_resource_status(self, metrics_summary: Dict[str, Any]) -> PerformanceStatus:
        """Determine resource utilization status"""
        cpu_avg = metrics_summary['cpu_utilization']['avg_percent']
        memory_avg = metrics_summary['memory_utilization']['avg_percent']
        
        if cpu_avg > 90 or memory_avg > 90:
            return PerformanceStatus.CRITICAL
        elif cpu_avg > 75 or memory_avg > 75:
            return PerformanceStatus.DEGRADED
        elif cpu_avg > 50 or memory_avg > 50:
            return PerformanceStatus.NORMAL
        else:
            return PerformanceStatus.OPTIMAL
    
    def _determine_alert_status(self, metrics_summary: Dict[str, Any]) -> PerformanceStatus:
        """Determine alert status"""
        critical_count = metrics_summary['alert_counts']['critical']
        error_count = metrics_summary['alert_counts']['error']
        
        if critical_count > 0:
            return PerformanceStatus.CRITICAL
        elif error_count > 5:
            return PerformanceStatus.DEGRADED
        elif error_count > 0:
            return PerformanceStatus.NORMAL
        else:
            return PerformanceStatus.OPTIMAL
    
    def _generate_recommendations(self, 
                                metrics_summary: Dict[str, Any],
                                alerts: List[PerformanceAlert]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Check processing times
        p99_time = metrics_summary['processing_times']['p99_ms']
        if p99_time > 1000:
            recommendations.append(
                f"High P99 latency ({p99_time:.0f}ms). Consider optimizing slow operations."
            )
        
        # Check success rates
        overall_success = metrics_summary['success_rates']['overall']
        if overall_success < 0.99:
            recommendations.append(
                f"Success rate ({overall_success:.1%}) below target. Investigate failures."
            )
        
        # Check for repeated alerts
        alert_counts = defaultdict(int)
        for alert in alerts:
            alert_counts[alert.metric_name] += 1
        
        for metric, count in alert_counts.items():
            if count > 5:
                recommendations.append(
                    f"Frequent alerts for {metric}. Consider adjusting thresholds or investigating root cause."
                )
        
        return recommendations[:5]  # Limit to top 5
    
    def _generate_resource_recommendations(self, metrics_summary: Dict[str, Any]) -> List[str]:
        """Generate resource utilization recommendations"""
        recommendations = []
        
        cpu_avg = metrics_summary['cpu_utilization']['avg_percent']
        if cpu_avg > 80:
            recommendations.append("High CPU utilization. Consider scaling horizontally or optimizing CPU-intensive operations.")
        
        memory_avg = metrics_summary['memory_utilization']['avg_percent']
        if memory_avg > 80:
            recommendations.append("High memory utilization. Review memory usage patterns and consider increasing resources.")
        
        return recommendations
    
    # Background monitoring
    
    def _start_background_monitoring(self):
        """Start background monitoring thread"""
        self._monitoring_thread = threading.Thread(
            target=self._background_monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info("Background monitoring started")
    
    def _background_monitoring_loop(self):
        """Background monitoring loop"""
        while not self._stop_monitoring.is_set():
            try:
                # Process queued metrics
                while not self._monitoring_queue.empty():
                    try:
                        item_type, item = self._monitoring_queue.get_nowait()
                        # Process item based on type
                        if item_type == 'metric':
                            # Metric already processed, just for background tasks
                            pass
                    except queue.Empty:
                        break
                
                # Collect system metrics
                if self.config.get('enable_system_monitoring', True):
                    self._collect_system_metrics()
                
                # Clean old metrics
                self._clean_old_metrics()
                
                # Sleep
                time.sleep(self.config.get('monitoring_interval', 10))
                
            except Exception as e:
                self.logger.error(f"Background monitoring error: {e}", exc_info=True)
    
    def _collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            metrics = self.system_monitor.get_current_metrics()
            
            self.record_metric('cpu_usage_percent', metrics['cpu_percent'])
            self.record_metric('memory_usage_percent', metrics['memory_percent'])
            self.record_metric('memory_used_mb', metrics['memory_used_mb'])
            self.record_metric('disk_usage_percent', metrics['disk_percent'])
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _clean_old_metrics(self):
        """Clean old metrics data"""
        retention_days = self.config.get('metric_retention_days', 7)
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        
        for metric_name, metric_list in list(self.metrics.items()):
            # Keep only recent metrics
            self.metrics[metric_name] = [
                m for m in metric_list if m.timestamp > cutoff_time
            ]
            
            # Remove empty lists
            if not self.metrics[metric_name]:
                del self.metrics[metric_name]
    
    # Persistence methods
    
    def persist_metrics(self) -> bool:
        """Persist metrics to storage"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save metrics
            metrics_file = self.storage_path / f'metrics_{timestamp}.pkl'
            with open(metrics_file, 'wb') as f:
                pickle.dump({
                    'metrics': dict(self.metrics),
                    'counters': dict(self.metric_counters),
                    'timestamp': datetime.now()
                }, f)
            
            # Save reports
            reports_file = self.storage_path / f'reports_{timestamp}.json'
            reports_data = [asdict(r) for r in self.reports]
            with open(reports_file, 'w') as f:
                json.dump(reports_data, f, indent=2, default=str)
            
            self.logger.info(f"Persisted metrics and reports to {self.storage_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to persist metrics: {e}", exc_info=True)
            return False
    
    def load_metrics(self, filename: str) -> bool:
        """Load metrics from storage"""
        try:
            filepath = self.storage_path / filename
            
            if filepath.suffix == '.pkl':
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.metrics.update(data['metrics'])
                    self.metric_counters.update(data['counters'])
            
            self.logger.info(f"Loaded metrics from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load metrics: {e}", exc_info=True)
            return False
    
    # Optimization suggestions
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get performance optimization suggestions"""
        suggestions = []
        
        # Analyze recent performance
        recent_reports = list(self.reports)[-5:]
        
        if recent_reports:
            # Check for consistent issues
            common_issues = defaultdict(int)
            
            for report in recent_reports:
                if report.status == PerformanceStatus.DEGRADED:
                    common_issues['degraded_performance'] += 1
                if report.status == PerformanceStatus.CRITICAL:
                    common_issues['critical_issues'] += 1
                
                for rec in report.recommendations:
                    if 'latency' in rec.lower():
                        common_issues['latency'] += 1
                    if 'accuracy' in rec.lower():
                        common_issues['accuracy'] += 1
                    if 'resource' in rec.lower():
                        common_issues['resources'] += 1
            
            # Generate suggestions based on issues
            if common_issues['latency'] > 2:
                suggestions.append("Consider implementing caching for frequently accessed data")
                suggestions.append("Review database query optimization")
                suggestions.append("Evaluate async processing for non-critical operations")
            
            if common_issues['accuracy'] > 2:
                suggestions.append("Review and update fraud detection models")
                suggestions.append("Analyze false positive/negative patterns")
                suggestions.append("Consider ensemble methods for improved accuracy")
            
            if common_issues['resources'] > 2:
                suggestions.append("Implement auto-scaling for peak loads")
                suggestions.append("Optimize memory usage in data processing")
                suggestions.append("Consider distributed processing for large batches")
        
        return suggestions
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        return {
            'total_metrics': sum(len(metrics) for metrics in self.metrics.values()),
            'unique_metrics': len(self.metrics),
            'metric_counts': dict(self.metric_counters),
            'active_alerts': len([a for a in self.alerts if a.timestamp > datetime.now() - timedelta(hours=1)]),
            'recent_reports': len(self.reports),
            'uptime_seconds': time.time() - self.start_time,
            'last_reset': self.last_reset.isoformat()
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        self.metrics.clear()
        self.metric_counters.clear()
        self.alerts.clear()
        self.reports.clear()
        self.last_reset = datetime.now()
        
        self.logger.info("All metrics reset")
    
    def stop(self):
        """Stop performance monitor"""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        
        # Persist final metrics
        self.persist_metrics()
        
        self.logger.info("Performance monitor stopped")
    
    def __repr__(self) -> str:
        metrics_count = sum(len(metrics) for metrics in self.metrics.values())
        return (f"FinancialPerformanceMonitor(metrics={metrics_count}, "
               f"alerts={len(self.alerts)}, reports={len(self.reports)})")


class SystemResourceMonitor:
    """Monitor system resource utilization"""
    
    def __init__(self):
        """Initialize system monitor"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Suppress psutil warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning, module='psutil')
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Optional: disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                disk_read_mb_s = disk_io.read_bytes / (1024 * 1024)
                disk_write_mb_s = disk_io.write_bytes / (1024 * 1024)
            else:
                disk_read_mb_s = 0
                disk_write_mb_s = 0
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_used_mb': memory_used_mb,
                'disk_percent': disk_percent,
                'disk_read_mb_s': disk_read_mb_s,
                'disk_write_mb_s': disk_write_mb_s
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {
                'cpu_percent': 0,
                'memory_percent': 0,
                'memory_used_mb': 0,
                'disk_percent': 0,
                'disk_read_mb_s': 0,
                'disk_write_mb_s': 0
            }


# Maintain backward compatibility
PerformanceMonitor = FinancialPerformanceMonitor


if __name__ == "__main__":
    # Example usage and testing
    import random
    
    # Initialize monitor
    monitor = FinancialPerformanceMonitor()
    
    print("=== Financial Performance Monitor ===\n")
    
    # Test 1: Record various metrics
    print("Test 1: Recording metrics")
    
    # Simulate fraud detection operations
    for i in range(50):
        operation_time = random.uniform(10, 100)
        success = random.random() > 0.05
        
        monitor.monitor_performance(
            'fraud_detection',
            operation_time,
            success=success,
            metadata={'batch_size': random.randint(1, 100)}
        )
        
        # Record accuracy metrics
        if success and random.random() > 0.1:
            monitor.record_metric('true_positive_count', 1, MetricType.COUNTER)
        elif not success:
            monitor.record_metric('false_positive_count', 1, MetricType.COUNTER)
        
        monitor.record_metric('model_confidence', random.uniform(0.5, 1.0))
        
        time.sleep(0.1)
    
    print("Recorded 50 operations\n")
    
    # Test 2: Generate reports
    print("Test 2: Generating reports")
    reports = monitor.generate_reports()
    
    for report in reports:
        print(f"\nReport: {report.report_id}")
        print(f"Status: {report.status.value}")
        print(f"Alerts: {len(report.alerts)}")
        print(f"Recommendations: {len(report.recommendations)}")
        if report.recommendations:
            print("- " + "\n- ".join(report.recommendations[:3]))
    
    # Test 3: Get optimization suggestions
    print("\nTest 3: Optimization suggestions")
    suggestions = monitor.get_optimization_suggestions()
    if suggestions:
        print("Suggested optimizations:")
        for suggestion in suggestions:
            print(f"- {suggestion}")
    else:
        print("No optimization suggestions at this time")
    
    # Test 4: Alert handling
    print("\nTest 4: Alert handling")
    
    def alert_handler(alert: PerformanceAlert):
        print(f"ALERT: [{alert.severity.value}] {alert.message}")
    
    monitor.add_alert_handler(alert_handler)
    
    # Trigger some alerts
    monitor.record_metric('cpu_usage_percent', 95)
    monitor.record_metric('error_rate', 0.15)
    
    # Test 5: Metrics summary
    print("\nTest 5: Metrics summary")
    metrics_summary = monitor.get_metrics()
    print(f"Total metrics: {metrics_summary['total_metrics']}")
    print(f"Unique metrics: {metrics_summary['unique_metrics']}")
    print(f"Active alerts: {metrics_summary['active_alerts']}")
    print(f"Uptime: {metrics_summary['uptime_seconds']:.1f} seconds")
    
    # Test 6: Persistence
    print("\nTest 6: Persistence")
    if monitor.persist_metrics():
        print("Metrics persisted successfully")
    
    # Stop monitor
    monitor.stop()
    print("\nPerformance monitor stopped")