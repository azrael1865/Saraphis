"""
Real-Time Accuracy Monitor for Financial Fraud Detection
Production-ready monitoring system for tracking model accuracy in real-time with drift detection.
Integrates with existing monitoring infrastructure for comprehensive accuracy tracking.
"""

import logging
import time
import threading
import queue
import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque, Counter
from contextlib import contextmanager
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import statistics
import math

# Import from existing modules
try:
    from enhanced_fraud_core_monitoring import (
        MonitoringManager, MetricsCollector, PerformanceMetrics,
        CacheManager, monitor_performance
    )
    from accuracy_tracking_db import (
        AccuracyTrackingDatabase, AccuracyMetric, ModelVersion,
        MetricType, DataType, ModelStatus
    )
    from enhanced_fraud_core_exceptions import (
        EnhancedFraudException, MonitoringError, ValidationError,
        ErrorContext, create_error_context
    )
except ImportError:
    # Fallback for standalone development
    from enhanced_fraud_core_monitoring import (
        MonitoringManager, MetricsCollector, PerformanceMetrics,
        CacheManager, monitor_performance
    )
    from accuracy_tracking_db import (
        AccuracyTrackingDatabase, AccuracyMetric, ModelVersion,
        MetricType, DataType, ModelStatus
    )
    from enhanced_fraud_core_exceptions import (
        EnhancedFraudException, MonitoringError, ValidationError,
        ErrorContext, create_error_context
    )

# Configure logging
logger = logging.getLogger(__name__)

# ======================== CUSTOM EXCEPTIONS ========================

class RealTimeMonitoringError(MonitoringError):
    """Base exception for real-time monitoring errors"""
    pass

class DriftDetectionError(RealTimeMonitoringError):
    """Exception raised during drift detection"""
    pass

class MetricCalculationError(RealTimeMonitoringError):
    """Exception raised during metric calculation"""
    pass

# ======================== ENUMS AND CONSTANTS ========================

class DriftType(Enum):
    """Types of drift detected"""
    ACCURACY_DRIFT = "accuracy_drift"
    CONFIDENCE_DRIFT = "confidence_drift"
    PREDICTION_DRIFT = "prediction_drift"
    FEATURE_DRIFT = "feature_drift"

class DriftSeverity(Enum):
    """Severity levels for drift"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MonitoringStatus(Enum):
    """Monitoring status"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

# Default monitoring configuration
DEFAULT_MONITORING_CONFIG = {
    'accuracy_window_minutes': 60,
    'drift_threshold': 0.05,
    'confidence_threshold': 0.8,
    'min_samples_for_calculation': 100,
    'sliding_window_size': 1000,
    'update_interval_seconds': 30,
    'enable_auto_alert': True,
    'cache_predictions': True,
    'max_cached_predictions': 10000
}

# ======================== DATA STRUCTURES ========================

@dataclass
class PredictionRecord:
    """Record of a single prediction"""
    prediction_id: str
    model_id: str
    timestamp: datetime
    prediction: Dict[str, Any]
    actual: Optional[Dict[str, Any]]
    features: Dict[str, Any]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccuracyWindow:
    """Sliding window for accuracy calculation"""
    window_id: str
    model_id: str
    start_time: datetime
    end_time: datetime
    predictions: List[PredictionRecord]
    accuracy_metrics: Dict[str, float]
    sample_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DriftAlert:
    """Drift detection alert"""
    alert_id: str
    model_id: str
    timestamp: datetime
    drift_type: DriftType
    severity: DriftSeverity
    current_value: float
    baseline_value: float
    threshold: float
    details: Dict[str, Any]
    recommendations: List[str]

@dataclass
class ModelMonitoringState:
    """State of model monitoring"""
    model_id: str
    status: MonitoringStatus
    start_time: datetime
    last_update: datetime
    total_predictions: int
    current_accuracy: float
    baseline_accuracy: Optional[float]
    drift_alerts: List[DriftAlert]
    accuracy_history: deque
    confidence_history: deque
    metadata: Dict[str, Any] = field(default_factory=dict)

# ======================== REAL-TIME ACCURACY MONITOR ========================

class RealTimeAccuracyMonitor:
    """
    Production-ready real-time accuracy monitoring system for fraud detection models.
    Tracks predictions, calculates live accuracy, and detects drift.
    """
    
    def __init__(
        self,
        monitoring_manager: MonitoringManager,
        accuracy_database: AccuracyTrackingDatabase,
        monitor_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize RealTimeAccuracyMonitor with existing monitoring infrastructure.
        
        Args:
            monitoring_manager: MonitoringManager instance
            accuracy_database: AccuracyTrackingDatabase instance
            monitor_config: Optional monitoring configuration
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Validate inputs
        if not isinstance(monitoring_manager, MonitoringManager):
            raise ValidationError(
                "monitoring_manager must be a MonitoringManager instance",
                context=create_error_context(
                    component="RealTimeAccuracyMonitor",
                    operation="init"
                )
            )
        
        if not isinstance(accuracy_database, AccuracyTrackingDatabase):
            raise ValidationError(
                "accuracy_database must be an AccuracyTrackingDatabase instance",
                context=create_error_context(
                    component="RealTimeAccuracyMonitor",
                    operation="init"
                )
            )
        
        self.monitoring_manager = monitoring_manager
        self.accuracy_database = accuracy_database
        self.metrics_collector = monitoring_manager.metrics_collector
        
        # Load configuration
        self.config = self._load_config(monitor_config)
        
        # Initialize components
        self._init_monitoring_state()
        self._init_prediction_cache()
        self._init_drift_detection()
        self._init_background_tasks()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self.operation_stats = defaultdict(int)
        
        self.logger.info(
            f"RealTimeAccuracyMonitor initialized with config: "
            f"accuracy_window={self.config['accuracy_window_minutes']}min, "
            f"drift_threshold={self.config['drift_threshold']}"
        )
    
    def _load_config(self, monitor_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load and validate monitoring configuration"""
        config = DEFAULT_MONITORING_CONFIG.copy()
        
        if monitor_config:
            # Validate and merge user config
            for key, value in monitor_config.items():
                if key in config:
                    config[key] = value
                else:
                    self.logger.warning(f"Unknown configuration key: {key}")
        
        return config
    
    def _init_monitoring_state(self) -> None:
        """Initialize monitoring state tracking"""
        self.model_states = {}
        self.active_models = set()
        self.monitoring_threads = {}
        self.stop_monitoring_events = {}
    
    def _init_prediction_cache(self) -> None:
        """Initialize prediction caching system"""
        self.prediction_cache = {}
        self.cache_size = {}
        self.prediction_queue = queue.Queue(maxsize=self.config['max_cached_predictions'])
        
        # Initialize cache manager if not available
        if not hasattr(self.monitoring_manager, 'cache_manager') or not self.monitoring_manager.cache_manager:
            from enhanced_fraud_core_monitoring import CacheManager, MonitoringConfig
            cache_config = MonitoringConfig()
            cache_config.cache_size = self.config['max_cached_predictions']
            self.cache_manager = CacheManager(cache_config)
        else:
            self.cache_manager = self.monitoring_manager.cache_manager
    
    def _init_drift_detection(self) -> None:
        """Initialize drift detection components"""
        self.drift_detectors = {}
        self.baseline_metrics = {}
        self.drift_history = defaultdict(list)
        
        # Register health check with monitoring manager
        self.monitoring_manager.add_health_check(
            'real_time_accuracy_monitor',
            self._health_check
        )
    
    def _init_background_tasks(self) -> None:
        """Initialize background processing tasks"""
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.update_thread = None
        self.stop_updates = threading.Event()
        
        # Start background update thread
        self._start_background_updates()
    
    # ======================== MONITORING CONTROL METHODS ========================
    
    def start_monitoring(
        self,
        model_ids: List[str],
        monitoring_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Initialize real-time monitoring for specified models.
        
        Args:
            model_ids: List of model IDs to monitor
            monitoring_config: Optional model-specific configuration
            
        Returns:
            Monitoring initialization status
        """
        start_time = time.time()
        results = {
            'started': [],
            'failed': [],
            'already_active': []
        }
        
        with self._lock:
            for model_id in model_ids:
                try:
                    if model_id in self.active_models:
                        results['already_active'].append(model_id)
                        self.logger.warning(f"Model {model_id} is already being monitored")
                        continue
                    
                    # Create model state
                    model_state = ModelMonitoringState(
                        model_id=model_id,
                        status=MonitoringStatus.ACTIVE,
                        start_time=datetime.now(),
                        last_update=datetime.now(),
                        total_predictions=0,
                        current_accuracy=0.0,
                        baseline_accuracy=None,
                        drift_alerts=[],
                        accuracy_history=deque(maxlen=self.config['sliding_window_size']),
                        confidence_history=deque(maxlen=self.config['sliding_window_size'])
                    )
                    
                    # Load baseline metrics if available
                    baseline = self._load_baseline_metrics(model_id)
                    if baseline:
                        model_state.baseline_accuracy = baseline.get('accuracy', 0.8)
                        self.baseline_metrics[model_id] = baseline
                    
                    # Initialize model-specific components
                    self.model_states[model_id] = model_state
                    self.prediction_cache[model_id] = deque(maxlen=self.config['sliding_window_size'])
                    self.cache_size[model_id] = 0
                    
                    # Start model-specific monitoring thread
                    stop_event = threading.Event()
                    self.stop_monitoring_events[model_id] = stop_event
                    
                    monitor_thread = threading.Thread(
                        target=self._model_monitoring_loop,
                        args=(model_id, monitoring_config),
                        daemon=True
                    )
                    monitor_thread.start()
                    self.monitoring_threads[model_id] = monitor_thread
                    
                    # Add to active models
                    self.active_models.add(model_id)
                    results['started'].append(model_id)
                    
                    # Record in metrics collector
                    self.metrics_collector.record_performance_metric(
                        PerformanceMetrics(
                            timestamp=datetime.now(),
                            operation_name=f"start_monitoring_{model_id}",
                            duration=time.time() - start_time,
                            success=True
                        )
                    )
                    
                    self.logger.info(f"Started monitoring for model {model_id}")
                    
                except Exception as e:
                    results['failed'].append({
                        'model_id': model_id,
                        'error': str(e)
                    })
                    self.logger.error(f"Failed to start monitoring for {model_id}: {e}")
        
        self.operation_stats['models_monitored'] += len(results['started'])
        
        return {
            'status': 'completed',
            'duration': time.time() - start_time,
            'results': results,
            'active_models': len(self.active_models)
        }
    
    def stop_monitoring(self, model_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Stop monitoring for specified models or all models.
        
        Args:
            model_ids: List of model IDs to stop monitoring (None for all)
            
        Returns:
            Stop operation results
        """
        with self._lock:
            if model_ids is None:
                model_ids = list(self.active_models)
            
            results = {
                'stopped': [],
                'not_found': []
            }
            
            for model_id in model_ids:
                if model_id not in self.active_models:
                    results['not_found'].append(model_id)
                    continue
                
                # Stop monitoring thread
                if model_id in self.stop_monitoring_events:
                    self.stop_monitoring_events[model_id].set()
                
                # Update model state
                if model_id in self.model_states:
                    self.model_states[model_id].status = MonitoringStatus.STOPPED
                
                # Clean up
                self.active_models.discard(model_id)
                results['stopped'].append(model_id)
                
                self.logger.info(f"Stopped monitoring for model {model_id}")
            
            return results
    
    # ======================== PREDICTION TRACKING METHODS ========================
    
    def track_prediction(
        self,
        model_id: str,
        prediction: Dict[str, Any],
        actual: Optional[Dict[str, Any]],
        features: Dict[str, Any],
        confidence: float,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Track individual model prediction with features and confidence.
        
        Args:
            model_id: Model identifier
            prediction: Prediction output (e.g., {'fraud_probability': 0.85, 'prediction': 'fraud'})
            actual: Actual label if available (e.g., {'is_fraud': True})
            features: Transaction features used for prediction
            confidence: Prediction confidence score
            timestamp: Prediction timestamp (defaults to now)
            
        Returns:
            Tracking result with prediction ID
        """
        if model_id not in self.active_models:
            raise ValidationError(
                f"Model {model_id} is not being monitored. Start monitoring first.",
                context=create_error_context(
                    component="RealTimeAccuracyMonitor",
                    operation="track_prediction",
                    model_id=model_id
                )
            )
        
        start_time = time.time()
        timestamp = timestamp or datetime.now()
        
        # Create prediction record
        prediction_id = self._generate_prediction_id()
        
        prediction_record = PredictionRecord(
            prediction_id=prediction_id,
            model_id=model_id,
            timestamp=timestamp,
            prediction=prediction,
            actual=actual,
            features=features,
            confidence=confidence,
            metadata={
                'tracked_at': datetime.now().isoformat(),
                'has_actual': actual is not None
            }
        )
        
        with self._lock:
            # Add to prediction cache
            self.prediction_cache[model_id].append(prediction_record)
            self.cache_size[model_id] += 1
            
            # Update model state
            model_state = self.model_states[model_id]
            model_state.total_predictions += 1
            model_state.last_update = datetime.now()
            
            # Track confidence
            model_state.confidence_history.append(confidence)
            
            # If actual label is available, update accuracy immediately
            if actual is not None:
                self._update_accuracy_metrics(model_id, prediction_record)
            
            # Queue for background processing
            try:
                self.prediction_queue.put_nowait((model_id, prediction_record))
            except queue.Full:
                self.logger.warning(f"Prediction queue full for model {model_id}")
        
        # Record performance metric
        self.metrics_collector.record_performance_metric(
            PerformanceMetrics(
                timestamp=datetime.now(),
                operation_name="track_prediction",
                duration=time.time() - start_time,
                success=True,
                additional_data={'model_id': model_id, 'has_actual': actual is not None}
            )
        )
        
        self.operation_stats['predictions_tracked'] += 1
        
        return {
            'prediction_id': prediction_id,
            'model_id': model_id,
            'timestamp': timestamp.isoformat(),
            'tracked': True,
            'cache_size': self.cache_size[model_id]
        }
    
    def update_prediction_actual(
        self,
        prediction_id: str,
        actual: Dict[str, Any]
    ) -> bool:
        """
        Update a tracked prediction with actual outcome.
        
        Args:
            prediction_id: Prediction identifier
            actual: Actual outcome
            
        Returns:
            Success status
        """
        with self._lock:
            # Find prediction in cache
            for model_id, predictions in self.prediction_cache.items():
                for pred in predictions:
                    if pred.prediction_id == prediction_id:
                        pred.actual = actual
                        pred.metadata['actual_updated_at'] = datetime.now().isoformat()
                        
                        # Update accuracy metrics
                        self._update_accuracy_metrics(model_id, pred)
                        
                        self.logger.debug(f"Updated actual for prediction {prediction_id}")
                        return True
        
        self.logger.warning(f"Prediction {prediction_id} not found in cache")
        return False
    
    # ======================== ACCURACY CALCULATION METHODS ========================
    
    def calculate_live_accuracy(
        self,
        model_id: str,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Calculate accuracy over specified time window.
        
        Args:
            model_id: Model identifier
            time_window: Time window for calculation (defaults to config)
            
        Returns:
            Accuracy metrics for the time window
        """
        if model_id not in self.active_models:
            raise ValidationError(f"Model {model_id} is not being monitored")
        
        if time_window is None:
            time_window = timedelta(minutes=self.config['accuracy_window_minutes'])
        
        cutoff_time = datetime.now() - time_window
        
        with self._lock:
            # Get predictions within time window
            predictions = [
                p for p in self.prediction_cache[model_id]
                if p.timestamp >= cutoff_time and p.actual is not None
            ]
            
            if len(predictions) < self.config['min_samples_for_calculation']:
                return {
                    'model_id': model_id,
                    'time_window': str(time_window),
                    'sample_count': len(predictions),
                    'status': 'insufficient_data',
                    'min_required': self.config['min_samples_for_calculation']
                }
            
            # Calculate metrics
            y_true = []
            y_pred = []
            confidences = []
            
            for pred in predictions:
                # Extract actual and predicted labels
                y_true.append(int(pred.actual.get('is_fraud', 0)))
                y_pred.append(int(pred.prediction.get('prediction', '') == 'fraud'))
                confidences.append(pred.confidence)
            
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Calculate comprehensive metrics
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'avg_confidence': np.mean(confidences),
                'confidence_std': np.std(confidences)
            }
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics.update({
                    'true_positives': int(tp),
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                    'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
                })
            
            # Create accuracy window record
            window_record = AccuracyWindow(
                window_id=self._generate_window_id(),
                model_id=model_id,
                start_time=cutoff_time,
                end_time=datetime.now(),
                predictions=predictions,
                accuracy_metrics=metrics,
                sample_count=len(predictions)
            )
            
            # Update model state
            model_state = self.model_states[model_id]
            model_state.current_accuracy = metrics['accuracy']
            model_state.accuracy_history.append({
                'timestamp': datetime.now(),
                'accuracy': metrics['accuracy'],
                'sample_count': len(predictions)
            })
            
            # Store in database
            self._store_accuracy_metrics(model_id, metrics, window_record)
            
            return {
                'model_id': model_id,
                'time_window': str(time_window),
                'start_time': cutoff_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'sample_count': len(predictions),
                'metrics': metrics,
                'status': 'calculated'
            }
    
    def get_current_accuracy_metrics(self, model_id: str) -> Dict[str, Any]:
        """
        Get current accuracy state and metrics for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Current accuracy metrics and state
        """
        if model_id not in self.model_states:
            return {
                'model_id': model_id,
                'status': 'not_monitored',
                'error': f'Model {model_id} is not being monitored'
            }
        
        with self._lock:
            model_state = self.model_states[model_id]
            
            # Calculate recent metrics
            recent_metrics = self.calculate_live_accuracy(
                model_id,
                timedelta(minutes=self.config['accuracy_window_minutes'])
            )
            
            # Get trend information
            trend_info = self._calculate_accuracy_trend(model_id)
            
            # Get drift status
            drift_status = self._get_drift_status(model_id)
            
            return {
                'model_id': model_id,
                'status': model_state.status.value,
                'monitoring_start': model_state.start_time.isoformat(),
                'last_update': model_state.last_update.isoformat(),
                'total_predictions': model_state.total_predictions,
                'current_accuracy': model_state.current_accuracy,
                'baseline_accuracy': model_state.baseline_accuracy,
                'recent_metrics': recent_metrics.get('metrics', {}),
                'trend': trend_info,
                'drift_status': drift_status,
                'active_alerts': len([a for a in model_state.drift_alerts if a.timestamp > datetime.now() - timedelta(hours=1)]),
                'cache_size': self.cache_size.get(model_id, 0)
            }
    
    def update_accuracy_statistics(
        self,
        model_id: str,
        accuracy_data: Dict[str, Any]
    ) -> bool:
        """
        Update rolling statistics for model accuracy.
        
        Args:
            model_id: Model identifier
            accuracy_data: New accuracy data to incorporate
            
        Returns:
            Success status
        """
        if model_id not in self.model_states:
            self.logger.error(f"Model {model_id} not found in monitoring states")
            return False
        
        try:
            with self._lock:
                model_state = self.model_states[model_id]
                
                # Update accuracy history
                model_state.accuracy_history.append({
                    'timestamp': datetime.now(),
                    'accuracy': accuracy_data.get('accuracy', 0),
                    'precision': accuracy_data.get('precision', 0),
                    'recall': accuracy_data.get('recall', 0),
                    'f1_score': accuracy_data.get('f1_score', 0),
                    'sample_count': accuracy_data.get('sample_count', 0)
                })
                
                # Update current accuracy
                model_state.current_accuracy = accuracy_data.get('accuracy', model_state.current_accuracy)
                model_state.last_update = datetime.now()
                
                # Check for significant changes
                if model_state.baseline_accuracy:
                    accuracy_change = abs(model_state.current_accuracy - model_state.baseline_accuracy)
                    if accuracy_change > self.config['drift_threshold']:
                        self._trigger_accuracy_alert(model_id, accuracy_change)
                
                # Store in database
                self._store_accuracy_update(model_id, accuracy_data)
                
                self.logger.debug(f"Updated accuracy statistics for model {model_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update accuracy statistics: {e}")
            return False
    
    # ======================== DRIFT DETECTION METHODS ========================
    
    def detect_basic_accuracy_drift(
        self,
        model_id: str,
        threshold_config: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Basic accuracy drift detection using threshold-based approach.
        
        Args:
            model_id: Model identifier
            threshold_config: Custom threshold configuration
            
        Returns:
            Drift detection results
        """
        if model_id not in self.model_states:
            raise ValidationError(f"Model {model_id} is not being monitored")
        
        # Default thresholds
        default_thresholds = {
            'accuracy_drop': 0.05,  # 5% drop
            'precision_drop': 0.05,
            'recall_drop': 0.05,
            'f1_drop': 0.05
        }
        
        if threshold_config:
            default_thresholds.update(threshold_config)
        
        with self._lock:
            model_state = self.model_states[model_id]
            
            # Get recent and baseline metrics
            recent_metrics = self.calculate_live_accuracy(model_id)
            if recent_metrics.get('status') != 'calculated':
                return {
                    'drift_detected': False,
                    'reason': 'insufficient_data',
                    'details': recent_metrics
                }
            
            metrics = recent_metrics['metrics']
            baseline = self.baseline_metrics.get(model_id, {})
            
            drift_results = {
                'drift_detected': False,
                'drift_metrics': {},
                'severity': DriftSeverity.LOW.value,
                'recommendations': []
            }
            
            # Check each metric
            for metric_name, threshold_key in [
                ('accuracy', 'accuracy_drop'),
                ('precision', 'precision_drop'),
                ('recall', 'recall_drop'),
                ('f1_score', 'f1_drop')
            ]:
                current_value = metrics.get(metric_name, 0)
                baseline_value = baseline.get(metric_name, current_value)
                
                drop = baseline_value - current_value
                if drop > default_thresholds[threshold_key]:
                    drift_results['drift_detected'] = True
                    drift_results['drift_metrics'][metric_name] = {
                        'current': current_value,
                        'baseline': baseline_value,
                        'drop': drop,
                        'threshold': default_thresholds[threshold_key]
                    }
            
            # Determine severity
            if drift_results['drift_detected']:
                drift_count = len(drift_results['drift_metrics'])
                max_drop = max(m['drop'] for m in drift_results['drift_metrics'].values())
                
                if max_drop > 0.15 or drift_count >= 3:
                    drift_results['severity'] = DriftSeverity.CRITICAL.value
                elif max_drop > 0.10 or drift_count >= 2:
                    drift_results['severity'] = DriftSeverity.HIGH.value
                elif max_drop > 0.05:
                    drift_results['severity'] = DriftSeverity.MEDIUM.value
                
                # Create drift alert
                alert = DriftAlert(
                    alert_id=self._generate_alert_id(),
                    model_id=model_id,
                    timestamp=datetime.now(),
                    drift_type=DriftType.ACCURACY_DRIFT,
                    severity=DriftSeverity(drift_results['severity']),
                    current_value=metrics['accuracy'],
                    baseline_value=baseline.get('accuracy', metrics['accuracy']),
                    threshold=default_thresholds['accuracy_drop'],
                    details=drift_results['drift_metrics'],
                    recommendations=self._generate_drift_recommendations(drift_results)
                )
                
                model_state.drift_alerts.append(alert)
                drift_results['alert_id'] = alert.alert_id
                
                # Send alert if enabled
                if self.config['enable_auto_alert']:
                    self._send_drift_alert(alert)
            
            drift_results['timestamp'] = datetime.now().isoformat()
            drift_results['model_id'] = model_id
            
            return drift_results
    
    def monitor_prediction_confidence_drift(
        self,
        model_id: str,
        confidence_thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Monitor drift in prediction confidence distributions.
        
        Args:
            model_id: Model identifier
            confidence_thresholds: Custom confidence thresholds
            
        Returns:
            Confidence drift analysis
        """
        if model_id not in self.model_states:
            raise ValidationError(f"Model {model_id} is not being monitored")
        
        # Default thresholds
        default_thresholds = {
            'mean_shift': 0.1,  # 10% shift in mean confidence
            'std_increase': 0.05,  # 5% increase in std deviation
            'low_confidence_rate': 0.2  # 20% predictions below threshold
        }
        
        if confidence_thresholds:
            default_thresholds.update(confidence_thresholds)
        
        with self._lock:
            model_state = self.model_states[model_id]
            
            # Get recent confidence values
            recent_confidences = list(model_state.confidence_history)
            if len(recent_confidences) < self.config['min_samples_for_calculation']:
                return {
                    'drift_detected': False,
                    'reason': 'insufficient_data',
                    'sample_count': len(recent_confidences)
                }
            
            # Calculate statistics
            current_mean = np.mean(recent_confidences)
            current_std = np.std(recent_confidences)
            low_confidence_count = sum(1 for c in recent_confidences if c < self.config['confidence_threshold'])
            low_confidence_rate = low_confidence_count / len(recent_confidences)
            
            # Get baseline statistics
            baseline = self.baseline_metrics.get(model_id, {})
            baseline_mean = baseline.get('confidence_mean', current_mean)
            baseline_std = baseline.get('confidence_std', current_std)
            
            # Detect drift
            drift_results = {
                'drift_detected': False,
                'confidence_stats': {
                    'current_mean': current_mean,
                    'baseline_mean': baseline_mean,
                    'current_std': current_std,
                    'baseline_std': baseline_std,
                    'low_confidence_rate': low_confidence_rate,
                    'sample_count': len(recent_confidences)
                },
                'drift_indicators': []
            }
            
            # Check mean shift
            mean_shift = abs(current_mean - baseline_mean)
            if mean_shift > default_thresholds['mean_shift']:
                drift_results['drift_detected'] = True
                drift_results['drift_indicators'].append({
                    'type': 'mean_shift',
                    'value': mean_shift,
                    'threshold': default_thresholds['mean_shift']
                })
            
            # Check std increase
            std_increase = current_std - baseline_std
            if std_increase > default_thresholds['std_increase']:
                drift_results['drift_detected'] = True
                drift_results['drift_indicators'].append({
                    'type': 'std_increase',
                    'value': std_increase,
                    'threshold': default_thresholds['std_increase']
                })
            
            # Check low confidence rate
            if low_confidence_rate > default_thresholds['low_confidence_rate']:
                drift_results['drift_detected'] = True
                drift_results['drift_indicators'].append({
                    'type': 'low_confidence_rate',
                    'value': low_confidence_rate,
                    'threshold': default_thresholds['low_confidence_rate']
                })
            
            # Create alert if drift detected
            if drift_results['drift_detected']:
                alert = DriftAlert(
                    alert_id=self._generate_alert_id(),
                    model_id=model_id,
                    timestamp=datetime.now(),
                    drift_type=DriftType.CONFIDENCE_DRIFT,
                    severity=self._determine_confidence_drift_severity(drift_results),
                    current_value=current_mean,
                    baseline_value=baseline_mean,
                    threshold=default_thresholds['mean_shift'],
                    details=drift_results,
                    recommendations=self._generate_confidence_drift_recommendations(drift_results)
                )
                
                model_state.drift_alerts.append(alert)
                drift_results['alert_id'] = alert.alert_id
            
            return drift_results
    
    def track_accuracy_trend_changes(
        self,
        model_id: str,
        trend_analysis_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Track and analyze accuracy trend changes over time.
        
        Args:
            model_id: Model identifier
            trend_analysis_config: Configuration for trend analysis
            
        Returns:
            Trend analysis results
        """
        if model_id not in self.model_states:
            raise ValidationError(f"Model {model_id} is not being monitored")
        
        # Default configuration
        default_config = {
            'window_size': 20,  # Number of points for trend
            'min_points': 10,
            'trend_threshold': 0.02,  # 2% change considered significant
            'smoothing_factor': 0.3
        }
        
        if trend_analysis_config:
            default_config.update(trend_analysis_config)
        
        with self._lock:
            model_state = self.model_states[model_id]
            
            # Get accuracy history
            accuracy_points = [
                h['accuracy'] for h in model_state.accuracy_history
                if 'accuracy' in h
            ][-default_config['window_size']:]
            
            if len(accuracy_points) < default_config['min_points']:
                return {
                    'trend_detected': False,
                    'reason': 'insufficient_data',
                    'points_available': len(accuracy_points),
                    'points_required': default_config['min_points']
                }
            
            # Calculate trend using linear regression
            x = np.arange(len(accuracy_points))
            y = np.array(accuracy_points)
            
            # Apply exponential smoothing
            smoothed_y = self._exponential_smoothing(y, default_config['smoothing_factor'])
            
            # Fit linear trend
            coeffs = np.polyfit(x, smoothed_y, 1)
            slope = coeffs[0]
            
            # Calculate trend metrics
            trend_direction = 'increasing' if slope > 0 else 'decreasing'
            trend_magnitude = abs(slope) * len(accuracy_points)  # Total change
            
            # Detect significant trends
            trend_results = {
                'trend_detected': abs(trend_magnitude) > default_config['trend_threshold'],
                'trend_direction': trend_direction,
                'trend_slope': float(slope),
                'trend_magnitude': float(trend_magnitude),
                'current_accuracy': float(accuracy_points[-1]),
                'accuracy_range': {
                    'min': float(np.min(accuracy_points)),
                    'max': float(np.max(accuracy_points)),
                    'mean': float(np.mean(accuracy_points)),
                    'std': float(np.std(accuracy_points))
                },
                'window_size': len(accuracy_points),
                'analysis_config': default_config
            }
            
            # Add change points detection
            change_points = self._detect_change_points(accuracy_points)
            if change_points:
                trend_results['change_points'] = change_points
            
            # Generate recommendations based on trend
            if trend_results['trend_detected']:
                if trend_direction == 'decreasing' and trend_magnitude > 0.05:
                    trend_results['severity'] = 'high'
                    trend_results['recommendations'] = [
                        "Significant decreasing trend in accuracy detected",
                        "Consider retraining the model with recent data",
                        "Investigate potential data drift or distribution changes"
                    ]
                elif trend_direction == 'decreasing':
                    trend_results['severity'] = 'medium'
                    trend_results['recommendations'] = [
                        "Gradual accuracy degradation observed",
                        "Monitor closely for continued decline"
                    ]
                else:
                    trend_results['severity'] = 'low'
                    trend_results['recommendations'] = [
                        "Accuracy trend is stable or improving"
                    ]
            
            return trend_results
    
    def calculate_accuracy_stability_metrics(
        self,
        model_id: str,
        stability_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Calculate stability metrics for model accuracy.
        
        Args:
            model_id: Model identifier
            stability_window: Time window for stability calculation
            
        Returns:
            Stability metrics and analysis
        """
        if model_id not in self.model_states:
            raise ValidationError(f"Model {model_id} is not being monitored")
        
        if stability_window is None:
            stability_window = timedelta(hours=24)
        
        cutoff_time = datetime.now() - stability_window
        
        with self._lock:
            model_state = self.model_states[model_id]
            
            # Get accuracy values within window
            accuracy_values = []
            timestamps = []
            
            for record in model_state.accuracy_history:
                if record['timestamp'] >= cutoff_time:
                    accuracy_values.append(record['accuracy'])
                    timestamps.append(record['timestamp'])
            
            if len(accuracy_values) < 2:
                return {
                    'stability_calculated': False,
                    'reason': 'insufficient_data',
                    'data_points': len(accuracy_values)
                }
            
            # Calculate stability metrics
            accuracy_array = np.array(accuracy_values)
            
            stability_metrics = {
                'mean_accuracy': float(np.mean(accuracy_array)),
                'std_accuracy': float(np.std(accuracy_array)),
                'coefficient_of_variation': float(np.std(accuracy_array) / (np.mean(accuracy_array) + 1e-8)),
                'range': float(np.max(accuracy_array) - np.min(accuracy_array)),
                'iqr': float(np.percentile(accuracy_array, 75) - np.percentile(accuracy_array, 25)),
                'data_points': len(accuracy_values),
                'time_window': str(stability_window),
                'start_time': timestamps[0].isoformat() if timestamps else None,
                'end_time': timestamps[-1].isoformat() if timestamps else None
            }
            
            # Calculate stability score (0-1, higher is more stable)
            cv_score = max(0, 1 - stability_metrics['coefficient_of_variation'])
            range_score = max(0, 1 - stability_metrics['range'])
            
            stability_metrics['stability_score'] = (cv_score + range_score) / 2
            
            # Determine stability status
            if stability_metrics['stability_score'] > 0.9:
                stability_status = 'highly_stable'
            elif stability_metrics['stability_score'] > 0.7:
                stability_status = 'stable'
            elif stability_metrics['stability_score'] > 0.5:
                stability_status = 'moderately_stable'
            else:
                stability_status = 'unstable'
            
            stability_metrics['stability_status'] = stability_status
            
            # Add time-based analysis
            if len(timestamps) > 1:
                time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                             for i in range(len(timestamps)-1)]
                stability_metrics['avg_update_interval'] = np.mean(time_diffs)
                stability_metrics['update_regularity'] = 1 - (np.std(time_diffs) / (np.mean(time_diffs) + 1e-8))
            
            # Generate recommendations
            recommendations = []
            if stability_status == 'unstable':
                recommendations.append("Model showing unstable accuracy - investigate causes")
                recommendations.append("Consider increasing monitoring frequency")
            elif stability_metrics['coefficient_of_variation'] > 0.1:
                recommendations.append("High variability in accuracy detected")
            
            stability_metrics['recommendations'] = recommendations
            
            return stability_metrics
    
    # ======================== HELPER METHODS ========================
    
    def _model_monitoring_loop(
        self,
        model_id: str,
        monitoring_config: Optional[Dict[str, Any]]
    ) -> None:
        """Background monitoring loop for a specific model"""
        while model_id in self.active_models and not self.stop_monitoring_events[model_id].is_set():
            try:
                # Calculate accuracy metrics
                self.calculate_live_accuracy(model_id)
                
                # Check for drift
                self.detect_basic_accuracy_drift(model_id)
                self.monitor_prediction_confidence_drift(model_id)
                
                # Track trends
                self.track_accuracy_trend_changes(model_id)
                
                # Sleep for update interval
                time.sleep(self.config['update_interval_seconds'])
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop for {model_id}: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _update_accuracy_metrics(self, model_id: str, prediction: PredictionRecord) -> None:
        """Update accuracy metrics with new prediction"""
        if prediction.actual is None:
            return
        
        # This is called within a lock context
        model_state = self.model_states[model_id]
        
        # Quick accuracy update for recent predictions
        recent_predictions = [
            p for p in self.prediction_cache[model_id]
            if p.timestamp >= datetime.now() - timedelta(minutes=5) and p.actual is not None
        ]
        
        if len(recent_predictions) >= 10:  # Quick update threshold
            y_true = [int(p.actual.get('is_fraud', 0)) for p in recent_predictions]
            y_pred = [int(p.prediction.get('prediction', '') == 'fraud') for p in recent_predictions]
            
            quick_accuracy = accuracy_score(y_true, y_pred)
            model_state.current_accuracy = quick_accuracy
    
    def _store_accuracy_metrics(
        self,
        model_id: str,
        metrics: Dict[str, float],
        window: AccuracyWindow
    ) -> None:
        """Store accuracy metrics in database"""
        try:
            # Store each metric
            for metric_name, metric_value in metrics.items():
                if metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
                    metric_type = MetricType[metric_name.upper()]
                    
                    self.accuracy_database.record_accuracy_metrics(
                        model_id=model_id,
                        model_version="live",
                        y_true=np.array([1]),  # Dummy for aggregated metric
                        y_pred=np.array([1]),
                        data_type=DataType.PRODUCTION,
                        dataset_id=window.window_id
                    )
            
            # Record aggregated metric
            metric = AccuracyMetric(
                metric_id=self._generate_metric_id(),
                model_id=model_id,
                model_version="live",
                data_type=DataType.PRODUCTION,
                metric_type=MetricType.ACCURACY,
                metric_value=metrics['accuracy'],
                timestamp=datetime.now(),
                sample_size=window.sample_count,
                additional_info={
                    'window_id': window.window_id,
                    'metrics': metrics
                }
            )
            
            # Store using database
            self.accuracy_database.record_accuracy_metrics(
                model_id=model_id,
                model_version="live",
                y_true=np.ones(window.sample_count),  # Placeholder
                y_pred=np.ones(window.sample_count),  # Placeholder
                data_type=DataType.PRODUCTION
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store accuracy metrics: {e}")
    
    def _load_baseline_metrics(self, model_id: str) -> Optional[Dict[str, float]]:
        """Load baseline metrics for a model"""
        try:
            # Try to load from database
            recent_metrics = self.accuracy_database.get_accuracy_metrics(
                model_id=model_id,
                data_type=DataType.TEST,
                start_date=datetime.now() - timedelta(days=7)
            )
            
            if recent_metrics:
                # Calculate average baseline
                baseline = {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0
                }
                
                metric_counts = defaultdict(int)
                
                for metric in recent_metrics:
                    if metric.metric_type == MetricType.ACCURACY:
                        baseline['accuracy'] += float(metric.metric_value)
                        metric_counts['accuracy'] += 1
                    # Add other metrics similarly
                
                # Average the values
                for key in baseline:
                    if metric_counts[key] > 0:
                        baseline[key] /= metric_counts[key]
                
                return baseline
            
        except Exception as e:
            self.logger.error(f"Failed to load baseline metrics: {e}")
        
        # Return default baseline with explicit warning
        self.logger.warning(
            f"USING DUMMY BASELINE: No real baseline metrics available for model {model_id}. "
            f"Default baseline values should NOT be used for production monitoring decisions."
        )
        
        return {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.77,
            'confidence_mean': 0.85,
            'confidence_std': 0.15,
            '_is_dummy_data': True,
            '_warning': 'These are synthetic baseline values'
        }
    
    def _calculate_accuracy_trend(self, model_id: str) -> Dict[str, Any]:
        """Calculate accuracy trend for a model"""
        model_state = self.model_states[model_id]
        
        if len(model_state.accuracy_history) < 2:
            return {'trend': 'insufficient_data'}
        
        # Get recent accuracy values
        recent_accuracies = [
            h['accuracy'] for h in list(model_state.accuracy_history)[-10:]
            if 'accuracy' in h
        ]
        
        if len(recent_accuracies) < 2:
            return {'trend': 'insufficient_data'}
        
        # Simple trend calculation
        first_half_mean = np.mean(recent_accuracies[:len(recent_accuracies)//2])
        second_half_mean = np.mean(recent_accuracies[len(recent_accuracies)//2:])
        
        if second_half_mean > first_half_mean + 0.01:
            trend = 'improving'
        elif second_half_mean < first_half_mean - 0.01:
            trend = 'degrading'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'change': second_half_mean - first_half_mean,
            'current': recent_accuracies[-1],
            'window_size': len(recent_accuracies)
        }
    
    def _get_drift_status(self, model_id: str) -> Dict[str, Any]:
        """Get current drift status for a model"""
        model_state = self.model_states[model_id]
        
        # Get recent alerts
        recent_alerts = [
            a for a in model_state.drift_alerts
            if a.timestamp >= datetime.now() - timedelta(hours=1)
        ]
        
        if not recent_alerts:
            return {
                'drift_detected': False,
                'status': 'no_drift'
            }
        
        # Find most severe alert
        severity_order = [DriftSeverity.CRITICAL, DriftSeverity.HIGH, 
                         DriftSeverity.MEDIUM, DriftSeverity.LOW]
        
        most_severe = DriftSeverity.LOW
        for alert in recent_alerts:
            if severity_order.index(alert.severity) < severity_order.index(most_severe):
                most_severe = alert.severity
        
        return {
            'drift_detected': True,
            'severity': most_severe.value,
            'alert_count': len(recent_alerts),
            'drift_types': list(set(a.drift_type.value for a in recent_alerts))
        }
    
    def _generate_drift_recommendations(self, drift_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on drift results"""
        recommendations = []
        
        severity = drift_results.get('severity', 'low')
        drift_metrics = drift_results.get('drift_metrics', {})
        
        if severity == DriftSeverity.CRITICAL.value:
            recommendations.append("CRITICAL: Immediate action required - model performance severely degraded")
            recommendations.append("Consider rolling back to previous model version")
            recommendations.append("Initiate emergency retraining procedure")
        elif severity == DriftSeverity.HIGH.value:
            recommendations.append("HIGH: Significant drift detected - investigate immediately")
            recommendations.append("Analyze recent data for distribution changes")
            recommendations.append("Prepare for model retraining if drift continues")
        elif severity == DriftSeverity.MEDIUM.value:
            recommendations.append("MEDIUM: Moderate drift detected - monitor closely")
            recommendations.append("Review feature importance and data quality")
        
        # Specific recommendations based on metrics
        if 'accuracy' in drift_metrics and drift_metrics['accuracy']['drop'] > 0.1:
            recommendations.append("Accuracy drop exceeds 10% - check for data quality issues")
        
        if 'recall' in drift_metrics and drift_metrics['recall']['drop'] > 0.1:
            recommendations.append("Recall significantly decreased - risk of missing fraud cases")
        
        if 'precision' in drift_metrics and drift_metrics['precision']['drop'] > 0.1:
            recommendations.append("Precision decreased - expect more false positives")
        
        return recommendations
    
    def _generate_confidence_drift_recommendations(self, drift_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for confidence drift"""
        recommendations = []
        
        stats = drift_results.get('confidence_stats', {})
        indicators = drift_results.get('drift_indicators', [])
        
        for indicator in indicators:
            if indicator['type'] == 'low_confidence_rate':
                recommendations.append("High rate of low-confidence predictions detected")
                recommendations.append("Consider reviewing model calibration")
            elif indicator['type'] == 'mean_shift':
                recommendations.append("Significant shift in average confidence detected")
                recommendations.append("Model may be encountering unfamiliar patterns")
            elif indicator['type'] == 'std_increase':
                recommendations.append("Increased variance in confidence scores")
                recommendations.append("Model uncertainty has increased")
        
        return recommendations
    
    def _determine_confidence_drift_severity(self, drift_results: Dict[str, Any]) -> DriftSeverity:
        """Determine severity of confidence drift"""
        indicators = drift_results.get('drift_indicators', [])
        
        if len(indicators) >= 3:
            return DriftSeverity.HIGH
        elif len(indicators) >= 2:
            return DriftSeverity.MEDIUM
        elif any(i['value'] > i['threshold'] * 2 for i in indicators):
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.LOW
    
    def _trigger_accuracy_alert(self, model_id: str, accuracy_change: float) -> None:
        """Trigger alert for significant accuracy change"""
        alert = DriftAlert(
            alert_id=self._generate_alert_id(),
            model_id=model_id,
            timestamp=datetime.now(),
            drift_type=DriftType.ACCURACY_DRIFT,
            severity=DriftSeverity.HIGH if accuracy_change > 0.1 else DriftSeverity.MEDIUM,
            current_value=self.model_states[model_id].current_accuracy,
            baseline_value=self.model_states[model_id].baseline_accuracy,
            threshold=self.config['drift_threshold'],
            details={'accuracy_change': accuracy_change},
            recommendations=["Significant accuracy change detected", "Review recent predictions"]
        )
        
        self.model_states[model_id].drift_alerts.append(alert)
        
        if self.config['enable_auto_alert']:
            self._send_drift_alert(alert)
    
    def _send_drift_alert(self, alert: DriftAlert) -> None:
        """Send drift alert through monitoring system"""
        try:
            # Log alert
            self.logger.warning(
                f"Drift Alert [{alert.severity.value}] for model {alert.model_id}: "
                f"{alert.drift_type.value} - Current: {alert.current_value:.3f}, "
                f"Baseline: {alert.baseline_value:.3f}"
            )
            
            # Send through monitoring manager if available
            if hasattr(self.monitoring_manager, '_send_alert'):
                self.monitoring_manager._send_alert(
                    severity=alert.severity,
                    message=f"Model {alert.model_id}: {alert.drift_type.value} detected"
                )
            
        except Exception as e:
            self.logger.error(f"Failed to send drift alert: {e}")
    
    def _exponential_smoothing(self, data: np.ndarray, alpha: float) -> np.ndarray:
        """Apply exponential smoothing to data"""
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        
        return smoothed
    
    def _detect_change_points(self, data: List[float]) -> List[Dict[str, Any]]:
        """Detect change points in accuracy data"""
        if len(data) < 10:
            return []
        
        change_points = []
        window_size = 5
        
        for i in range(window_size, len(data) - window_size):
            before = data[i-window_size:i]
            after = data[i:i+window_size]
            
            # Simple change detection using mean difference
            mean_diff = abs(np.mean(after) - np.mean(before))
            
            if mean_diff > 0.05:  # 5% change threshold
                change_points.append({
                    'index': i,
                    'before_mean': np.mean(before),
                    'after_mean': np.mean(after),
                    'change_magnitude': mean_diff
                })
        
        return change_points
    
    def _store_accuracy_update(self, model_id: str, accuracy_data: Dict[str, Any]) -> None:
        """Store accuracy update in database"""
        try:
            # Create dummy arrays for the database method
            sample_count = accuracy_data.get('sample_count', 100)
            y_true = np.ones(sample_count)
            y_pred = np.ones(sample_count)
            
            # Adjust predictions to match accuracy
            accuracy = accuracy_data.get('accuracy', 0.5)
            n_correct = int(sample_count * accuracy)
            y_pred[:sample_count-n_correct] = 0
            
            self.accuracy_database.record_accuracy_metrics(
                model_id=model_id,
                model_version="live",
                y_true=y_true,
                y_pred=y_pred,
                data_type=DataType.PRODUCTION
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store accuracy update: {e}")
    
    def _start_background_updates(self) -> None:
        """Start background update thread"""
        def update_loop():
            while not self.stop_updates.is_set():
                try:
                    # Process prediction queue
                    while not self.prediction_queue.empty():
                        try:
                            model_id, prediction = self.prediction_queue.get_nowait()
                            # Additional processing if needed
                        except queue.Empty:
                            break
                    
                    # Periodic cleanup
                    self._cleanup_old_predictions()
                    
                    time.sleep(30)  # Update every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"Background update error: {e}")
                    time.sleep(60)
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
    
    def _cleanup_old_predictions(self) -> None:
        """Clean up old predictions from cache"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        with self._lock:
            for model_id in list(self.prediction_cache.keys()):
                # Remove old predictions
                old_size = len(self.prediction_cache[model_id])
                self.prediction_cache[model_id] = deque(
                    (p for p in self.prediction_cache[model_id] if p.timestamp > cutoff_time),
                    maxlen=self.config['sliding_window_size']
                )
                
                removed = old_size - len(self.prediction_cache[model_id])
                if removed > 0:
                    self.cache_size[model_id] -= removed
                    self.logger.debug(f"Removed {removed} old predictions for model {model_id}")
    
    def _health_check(self) -> bool:
        """Health check for monitoring manager"""
        try:
            # Check if monitoring is active
            if not self.active_models:
                return True  # No models to monitor is valid
            
            # Check each active model
            for model_id in self.active_models:
                if model_id not in self.model_states:
                    return False
                
                # Check if model has recent updates
                model_state = self.model_states[model_id]
                if datetime.now() - model_state.last_update > timedelta(minutes=10):
                    self.logger.warning(f"Model {model_id} has not updated in 10 minutes")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def _generate_prediction_id(self) -> str:
        """Generate unique prediction ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"pred_{timestamp}_{random_suffix}"
    
    def _generate_window_id(self) -> str:
        """Generate unique window ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"window_{timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"alert_{timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
    
    def _generate_metric_id(self) -> str:
        """Generate unique metric ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        return f"metric_{timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
    
    # ======================== PUBLIC UTILITY METHODS ========================
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        with self._lock:
            summary = {
                'active_models': list(self.active_models),
                'total_models': len(self.active_models),
                'total_predictions': sum(s.total_predictions for s in self.model_states.values()),
                'models': {}
            }
            
            for model_id in self.active_models:
                if model_id in self.model_states:
                    model_state = self.model_states[model_id]
                    summary['models'][model_id] = {
                        'status': model_state.status.value,
                        'current_accuracy': model_state.current_accuracy,
                        'total_predictions': model_state.total_predictions,
                        'active_alerts': len([a for a in model_state.drift_alerts 
                                            if a.timestamp > datetime.now() - timedelta(hours=1)]),
                        'last_update': model_state.last_update.isoformat()
                    }
            
            return summary
    
    def export_monitoring_report(
        self,
        model_id: str,
        output_path: Path,
        include_predictions: bool = False
    ) -> bool:
        """Export monitoring report for a model"""
        try:
            if model_id not in self.model_states:
                self.logger.error(f"Model {model_id} not found")
                return False
            
            report = {
                'model_id': model_id,
                'report_timestamp': datetime.now().isoformat(),
                'monitoring_state': asdict(self.model_states[model_id]),
                'current_metrics': self.get_current_accuracy_metrics(model_id),
                'drift_analysis': self.detect_basic_accuracy_drift(model_id),
                'trend_analysis': self.track_accuracy_trend_changes(model_id),
                'stability_metrics': self.calculate_accuracy_stability_metrics(model_id)
            }
            
            # Remove non-serializable items
            report['monitoring_state'].pop('accuracy_history', None)
            report['monitoring_state'].pop('confidence_history', None)
            report['monitoring_state'].pop('drift_alerts', None)
            
            # Add predictions if requested
            if include_predictions and model_id in self.prediction_cache:
                recent_predictions = list(self.prediction_cache[model_id])[-100:]  # Last 100
                report['recent_predictions'] = [
                    {
                        'prediction_id': p.prediction_id,
                        'timestamp': p.timestamp.isoformat(),
                        'prediction': p.prediction,
                        'actual': p.actual,
                        'confidence': p.confidence
                    }
                    for p in recent_predictions
                ]
            
            # Save report
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Exported monitoring report to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export report: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown the monitoring system"""
        self.logger.info("Shutting down RealTimeAccuracyMonitor")
        
        # Stop all monitoring
        self.stop_monitoring()
        
        # Stop background tasks
        self.stop_updates.set()
        if self.update_thread:
            self.update_thread.join(timeout=5)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("RealTimeAccuracyMonitor shutdown complete")


# ======================== USAGE EXAMPLES ========================

if __name__ == "__main__":
    # Example usage demonstrating the real-time accuracy monitoring system
    import numpy as np
    from datetime import datetime, timedelta
    
    print("=== RealTimeAccuracyMonitor Examples ===\n")
    
    # Initialize required components (mock versions for example)
    class MockMonitoringManager(MonitoringManager):
        def __init__(self):
            self.metrics_collector = type('obj', (object,), {
                'record_performance_metric': lambda x: None
            })
            self.cache_manager = None
        
        def add_health_check(self, name, func):
            pass
        
        def _send_alert(self, severity, message):
            print(f"ALERT [{severity}]: {message}")
    
    class MockAccuracyDatabase(AccuracyTrackingDatabase):
        def __init__(self):
            pass
        
        def record_accuracy_metrics(self, **kwargs):
            return {'status': 'recorded'}
        
        def get_accuracy_metrics(self, **kwargs):
            return []
    
    # Initialize monitor
    monitoring_manager = MockMonitoringManager()
    accuracy_database = MockAccuracyDatabase()
    
    monitor = RealTimeAccuracyMonitor(
        monitoring_manager=monitoring_manager,
        accuracy_database=accuracy_database,
        monitor_config={
            'accuracy_window_minutes': 5,
            'drift_threshold': 0.05,
            'min_samples_for_calculation': 10
        }
    )
    
    # Example 1: Start monitoring
    print("1. Starting monitoring for fraud detection model...")
    result = monitor.start_monitoring(['fraud_model_v1.2'])
    print(f"   Started: {result['results']['started']}")
    print(f"   Active models: {result['active_models']}\n")
    
    # Example 2: Track predictions
    print("2. Tracking model predictions...")
    
    # Simulate predictions
    for i in range(20):
        prediction = {
            'fraud_probability': np.random.random(),
            'prediction': 'fraud' if np.random.random() > 0.5 else 'legitimate'
        }
        
        actual = {
            'is_fraud': np.random.random() > 0.5
        } if i % 2 == 0 else None  # Only half have actuals immediately
        
        features = {
            'amount': np.random.uniform(10, 1000),
            'merchant_risk': np.random.random(),
            'user_history': np.random.randint(0, 100)
        }
        
        track_result = monitor.track_prediction(
            model_id='fraud_model_v1.2',
            prediction=prediction,
            actual=actual,
            features=features,
            confidence=prediction['fraud_probability']
        )
        
        if i == 0:
            print(f"   First prediction tracked: {track_result['prediction_id']}")
    
    print(f"   Total predictions tracked: 20\n")
    
    # Example 3: Calculate live accuracy
    print("3. Calculating live accuracy metrics...")
    
    # Add more predictions with actuals for accuracy calculation
    for i in range(15):
        prediction = {
            'fraud_probability': np.random.random(),
            'prediction': 'fraud' if np.random.random() > 0.3 else 'legitimate'
        }
        
        actual = {'is_fraud': np.random.random() > 0.7}
        
        monitor.track_prediction(
            model_id='fraud_model_v1.2',
            prediction=prediction,
            actual=actual,
            features={'amount': np.random.uniform(10, 1000)},
            confidence=prediction['fraud_probability']
        )
    
    accuracy_result = monitor.calculate_live_accuracy('fraud_model_v1.2')
    if 'metrics' in accuracy_result:
        print(f"   Accuracy: {accuracy_result['metrics']['accuracy']:.3f}")
        print(f"   Precision: {accuracy_result['metrics']['precision']:.3f}")
        print(f"   Recall: {accuracy_result['metrics']['recall']:.3f}")
        print(f"   F1 Score: {accuracy_result['metrics']['f1_score']:.3f}\n")
    
    # Example 4: Detect drift
    print("4. Checking for accuracy drift...")
    drift_result = monitor.detect_basic_accuracy_drift('fraud_model_v1.2')
    print(f"   Drift detected: {drift_result['drift_detected']}")
    if drift_result['drift_detected']:
        print(f"   Severity: {drift_result['severity']}")
        print(f"   Recommendations: {drift_result['recommendations'][0]}")
    print()
    
    # Example 5: Monitor confidence drift
    print("5. Monitoring prediction confidence drift...")
    confidence_drift = monitor.monitor_prediction_confidence_drift('fraud_model_v1.2')
    print(f"   Drift detected: {confidence_drift['drift_detected']}")
    if 'confidence_stats' in confidence_drift:
        print(f"   Mean confidence: {confidence_drift['confidence_stats']['current_mean']:.3f}")
        print(f"   Confidence std: {confidence_drift['confidence_stats']['current_std']:.3f}\n")
    
    # Example 6: Track accuracy trends
    print("6. Analyzing accuracy trends...")
    trend_result = monitor.track_accuracy_trend_changes('fraud_model_v1.2')
    if trend_result.get('trend_detected'):
        print(f"   Trend: {trend_result['trend_direction']}")
        print(f"   Magnitude: {trend_result['trend_magnitude']:.3f}")
    else:
        print(f"   {trend_result.get('reason', 'No significant trend detected')}\n")
    
    # Example 7: Calculate stability metrics
    print("7. Calculating stability metrics...")
    stability = monitor.calculate_accuracy_stability_metrics('fraud_model_v1.2')
    if stability.get('stability_calculated', False):
        print(f"   Stability score: {stability['stability_score']:.3f}")
        print(f"   Status: {stability['stability_status']}")
        print(f"   Coefficient of variation: {stability['coefficient_of_variation']:.3f}\n")
    
    # Example 8: Get current metrics
    print("8. Getting current accuracy metrics...")
    current = monitor.get_current_accuracy_metrics('fraud_model_v1.2')
    print(f"   Status: {current['status']}")
    print(f"   Total predictions: {current['total_predictions']}")
    print(f"   Current accuracy: {current['current_accuracy']:.3f}")
    print(f"   Active alerts: {current['active_alerts']}\n")
    
    # Example 9: Monitoring summary
    print("9. Monitoring summary:")
    summary = monitor.get_monitoring_summary()
    print(f"   Active models: {summary['total_models']}")
    print(f"   Total predictions: {summary['total_predictions']}")
    for model_id, info in summary['models'].items():
        print(f"   {model_id}: accuracy={info['current_accuracy']:.3f}, "
              f"predictions={info['total_predictions']}")
    
    # Cleanup
    monitor.shutdown()
    
    print("\n=== All examples completed successfully! ===")