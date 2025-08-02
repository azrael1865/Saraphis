#!/usr/bin/env python3
"""
Real-time Progress Tracking System for Saraphis AI Training.
Provides comprehensive monitoring, alerting, and visualization capabilities.
"""

import time
import threading
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import warnings

# Optional imports for advanced features
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    # Fallback for Figure type hint
    Figure = None

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics to track."""
    LOSS = "loss"
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    LEARNING_RATE = "learning_rate"
    GRADIENT_NORM = "gradient_norm"
    MEMORY_USAGE = "memory_usage"
    GPU_UTILIZATION = "gpu_utilization"
    THROUGHPUT = "throughput"
    CUSTOM = "custom"


@dataclass
class ProgressMetrics:
    """Container for training progress metrics."""
    epoch: int
    batch: int
    total_batches: int
    samples_processed: int
    total_samples: int
    
    # Timing metrics
    epoch_start_time: float
    batch_start_time: float
    samples_per_second: float
    time_per_batch: float
    estimated_time_remaining: float
    
    # Performance metrics
    train_loss: float
    val_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: float = 0.001
    gradient_norm: Optional[float] = None
    
    # Resource metrics
    memory_usage_mb: float = 0.0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    
    # Additional metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def get_progress_percentage(self) -> float:
        """Get overall progress percentage."""
        if self.total_samples > 0:
            return (self.samples_processed / self.total_samples) * 100
        return 0.0


@dataclass
class TrainingAlert:
    """Training alert information."""
    alert_id: str
    severity: AlertSeverity
    alert_type: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['severity'] = self.severity.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.resolution_time:
            data['resolution_time'] = self.resolution_time.isoformat()
        return data


class ProgressTracker:
    """
    Comprehensive progress tracking system for training sessions.
    Provides real-time monitoring, alerting, and visualization.
    """
    
    def __init__(self, session_id: str, config: Dict[str, Any] = None):
        """
        Initialize progress tracker.
        
        Args:
            session_id: Training session identifier
            config: Progress tracking configuration
        """
        self.session_id = session_id
        self.config = config or {}
        
        # Core tracking state
        self._current_epoch = 0
        self._current_batch = 0
        self._total_epochs = self.config.get('total_epochs', 100)
        self._total_batches_per_epoch = self.config.get('batches_per_epoch', 100)
        self._samples_per_batch = self.config.get('batch_size', 32)
        
        # Timing tracking
        self._training_start_time = None
        self._epoch_start_time = None
        self._batch_start_time = None
        self._batch_times = deque(maxlen=100)
        self._epoch_times = deque(maxlen=10)
        
        # Metrics history
        self._metrics_history = []
        self._metric_buffers = defaultdict(lambda: deque(maxlen=1000))
        self._best_metrics = {}
        self._metric_trends = {}
        
        # Alert system
        self._alerts = []
        self._alert_thresholds = self._initialize_alert_thresholds()
        self._alert_callbacks = defaultdict(list)
        self._consecutive_stagnation_count = 0
        self._last_improvement_epoch = 0
        
        # Progress callbacks
        self._progress_callbacks = []
        self._visualization_callbacks = []
        
        # Resource monitoring
        self._resource_monitor = ResourceMonitor()
        self._resource_history = deque(maxlen=1000)
        
        # Visualization state
        self._visualization_enabled = self.config.get('enable_visualization', True)
        self._visualization_interval = self.config.get('visualization_interval', 10)
        self._plot_data = defaultdict(list)
        
        # Thread safety
        self._lock = threading.RLock()
        self._stop_monitoring = threading.Event()
        
        # Logging
        self.logger = self._setup_logging()
        
        # Start background monitoring if enabled
        if self.config.get('enable_monitoring', True):
            self._start_background_monitoring()
        
        self.logger.info(f"ProgressTracker initialized for session {session_id}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup progress tracker logging."""
        logger = logging.getLogger(f"ProgressTracker.{self.session_id}")
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.setLevel(logging.INFO)
        return logger
    
    def _initialize_alert_thresholds(self) -> Dict[str, Any]:
        """Initialize alert thresholds."""
        return {
            'loss_stagnation_epochs': self.config.get('loss_stagnation_epochs', 5),
            'loss_explosion_threshold': self.config.get('loss_explosion_threshold', 10.0),
            'gradient_explosion_threshold': self.config.get('gradient_explosion_threshold', 100.0),
            'memory_warning_mb': self.config.get('memory_warning_mb', 6000),
            'memory_critical_mb': self.config.get('memory_critical_mb', 7500),
            'gpu_warning_percent': self.config.get('gpu_warning_percent', 85),
            'gpu_critical_percent': self.config.get('gpu_critical_percent', 95),
            'accuracy_drop_threshold': self.config.get('accuracy_drop_threshold', 0.1),
            'throughput_warning_threshold': self.config.get('throughput_warning_threshold', 10),
            'nan_loss_tolerance': self.config.get('nan_loss_tolerance', 0)
        }
    
    def start_training(self, total_epochs: int, batches_per_epoch: int, 
                      samples_per_batch: int) -> None:
        """
        Start tracking a training session.
        
        Args:
            total_epochs: Total number of epochs
            batches_per_epoch: Number of batches per epoch
            samples_per_batch: Number of samples per batch
        """
        with self._lock:
            self._training_start_time = time.time()
            self._total_epochs = total_epochs
            self._total_batches_per_epoch = batches_per_epoch
            self._samples_per_batch = samples_per_batch
            self._current_epoch = 0
            self._current_batch = 0
            
            self.logger.info(
                f"Training started: {total_epochs} epochs, "
                f"{batches_per_epoch} batches/epoch, "
                f"{samples_per_batch} samples/batch"
            )
            
            # Notify callbacks
            self._trigger_callbacks('training_start', {
                'total_epochs': total_epochs,
                'total_batches': total_epochs * batches_per_epoch,
                'total_samples': total_epochs * batches_per_epoch * samples_per_batch
            })
    
    def start_epoch(self, epoch: int) -> None:
        """Start tracking a new epoch."""
        with self._lock:
            self._current_epoch = epoch
            self._current_batch = 0
            self._epoch_start_time = time.time()
            
            # Calculate epoch progress
            epoch_progress = (epoch - 1) / self._total_epochs * 100
            
            self.logger.debug(f"Epoch {epoch}/{self._total_epochs} started ({epoch_progress:.1f}% complete)")
            
            # Notify callbacks
            self._trigger_callbacks('epoch_start', {
                'epoch': epoch,
                'total_epochs': self._total_epochs,
                'progress_percent': epoch_progress
            })
    
    def start_batch(self, batch: int) -> None:
        """Start tracking a new batch."""
        with self._lock:
            self._current_batch = batch
            self._batch_start_time = time.time()
    
    def update_batch(self, metrics: Dict[str, float]) -> None:
        """
        Update metrics for current batch.
        
        Args:
            metrics: Dictionary of metric values
        """
        with self._lock:
            if self._batch_start_time is None:
                self._batch_start_time = time.time()
            
            batch_time = time.time() - self._batch_start_time
            self._batch_times.append(batch_time)
            
            # Calculate samples processed
            samples_processed = (
                (self._current_epoch - 1) * self._total_batches_per_epoch * self._samples_per_batch +
                self._current_batch * self._samples_per_batch
            )
            total_samples = self._total_epochs * self._total_batches_per_epoch * self._samples_per_batch
            
            # Calculate throughput
            avg_batch_time = np.mean(list(self._batch_times)) if self._batch_times else batch_time
            samples_per_second = self._samples_per_batch / avg_batch_time if avg_batch_time > 0 else 0
            
            # Estimate time remaining
            batches_remaining = (
                (self._total_epochs - self._current_epoch) * self._total_batches_per_epoch +
                (self._total_batches_per_epoch - self._current_batch)
            )
            estimated_time_remaining = batches_remaining * avg_batch_time
            
            # Get resource metrics
            resource_metrics = self._resource_monitor.get_current_metrics()
            
            # Create progress metrics
            progress_metrics = ProgressMetrics(
                epoch=self._current_epoch,
                batch=self._current_batch,
                total_batches=self._total_batches_per_epoch,
                samples_processed=samples_processed,
                total_samples=total_samples,
                epoch_start_time=self._epoch_start_time or time.time(),
                batch_start_time=self._batch_start_time,
                samples_per_second=samples_per_second,
                time_per_batch=batch_time,
                estimated_time_remaining=estimated_time_remaining,
                train_loss=metrics.get('loss', 0.0),
                train_accuracy=metrics.get('accuracy'),
                learning_rate=metrics.get('learning_rate', 0.001),
                gradient_norm=metrics.get('gradient_norm'),
                memory_usage_mb=resource_metrics.get('memory_mb', 0),
                gpu_utilization=resource_metrics.get('gpu_percent', 0),
                cpu_utilization=resource_metrics.get('cpu_percent', 0),
                custom_metrics={k: v for k, v in metrics.items() 
                              if k not in ['loss', 'accuracy', 'learning_rate', 'gradient_norm']}
            )
            
            # Update metric buffers
            for metric_name, value in metrics.items():
                self._metric_buffers[metric_name].append(value)
            
            # Check for alerts
            self._check_alerts(progress_metrics, metrics)
            
            # Update visualization data
            if self._visualization_enabled:
                self._update_visualization_data(progress_metrics)
            
            # Trigger progress callbacks
            self._trigger_progress_callbacks(progress_metrics)
            
            # Log progress periodically
            if self._current_batch % self.config.get('log_interval', 10) == 0:
                self._log_progress(progress_metrics)
    
    def end_batch(self) -> None:
        """End tracking for current batch."""
        with self._lock:
            if self._current_batch >= self._total_batches_per_epoch:
                self.end_epoch()
    
    def end_epoch(self, validation_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        End tracking for current epoch.
        
        Args:
            validation_metrics: Optional validation metrics
        """
        with self._lock:
            if self._epoch_start_time:
                epoch_time = time.time() - self._epoch_start_time
                self._epoch_times.append(epoch_time)
            
            # Update validation metrics
            if validation_metrics:
                for metric_name, value in validation_metrics.items():
                    self._metric_buffers[f'val_{metric_name}'].append(value)
                
                # Check for best metrics
                self._update_best_metrics(validation_metrics)
                
                # Check for improvement
                self._check_improvement(validation_metrics)
            
            # Calculate epoch statistics
            epoch_stats = self._calculate_epoch_statistics()
            
            # Store metrics history
            self._metrics_history.append({
                'epoch': self._current_epoch,
                'timestamp': datetime.now().isoformat(),
                'duration': epoch_time if 'epoch_time' in locals() else 0,
                'metrics': epoch_stats,
                'validation_metrics': validation_metrics
            })
            
            # Trigger epoch callbacks
            self._trigger_callbacks('epoch_end', {
                'epoch': self._current_epoch,
                'metrics': epoch_stats,
                'validation_metrics': validation_metrics
            })
            
            self.logger.info(f"Epoch {self._current_epoch}/{self._total_epochs} completed in {epoch_time:.2f}s")
    
    def _check_alerts(self, progress_metrics: ProgressMetrics, batch_metrics: Dict[str, float]) -> None:
        """Check for training issues and generate alerts."""
        # Check for NaN loss
        if np.isnan(progress_metrics.train_loss):
            self._generate_alert(
                AlertSeverity.CRITICAL,
                'nan_loss',
                'NaN loss detected',
                {'epoch': progress_metrics.epoch, 'batch': progress_metrics.batch}
            )
        
        # Check for loss explosion
        if progress_metrics.train_loss > self._alert_thresholds['loss_explosion_threshold']:
            self._generate_alert(
                AlertSeverity.ERROR,
                'loss_explosion',
                f'Loss exploded to {progress_metrics.train_loss:.4f}',
                {'loss': progress_metrics.train_loss, 'threshold': self._alert_thresholds['loss_explosion_threshold']}
            )
        
        # Check for gradient explosion
        if progress_metrics.gradient_norm and progress_metrics.gradient_norm > self._alert_thresholds['gradient_explosion_threshold']:
            self._generate_alert(
                AlertSeverity.WARNING,
                'gradient_explosion',
                f'Large gradient norm: {progress_metrics.gradient_norm:.2f}',
                {'gradient_norm': progress_metrics.gradient_norm, 'threshold': self._alert_thresholds['gradient_explosion_threshold']}
            )
        
        # Check memory usage
        if progress_metrics.memory_usage_mb > self._alert_thresholds['memory_critical_mb']:
            self._generate_alert(
                AlertSeverity.CRITICAL,
                'memory_critical',
                f'Critical memory usage: {progress_metrics.memory_usage_mb:.0f}MB',
                {'memory_mb': progress_metrics.memory_usage_mb, 'threshold': self._alert_thresholds['memory_critical_mb']}
            )
        elif progress_metrics.memory_usage_mb > self._alert_thresholds['memory_warning_mb']:
            self._generate_alert(
                AlertSeverity.WARNING,
                'memory_warning',
                f'High memory usage: {progress_metrics.memory_usage_mb:.0f}MB',
                {'memory_mb': progress_metrics.memory_usage_mb, 'threshold': self._alert_thresholds['memory_warning_mb']}
            )
        
        # Check GPU utilization
        if progress_metrics.gpu_utilization > self._alert_thresholds['gpu_critical_percent']:
            self._generate_alert(
                AlertSeverity.ERROR,
                'gpu_critical',
                f'Critical GPU utilization: {progress_metrics.gpu_utilization:.1f}%',
                {'gpu_percent': progress_metrics.gpu_utilization, 'threshold': self._alert_thresholds['gpu_critical_percent']}
            )
        
        # Check throughput
        if progress_metrics.samples_per_second < self._alert_thresholds['throughput_warning_threshold']:
            self._generate_alert(
                AlertSeverity.WARNING,
                'low_throughput',
                f'Low training throughput: {progress_metrics.samples_per_second:.1f} samples/sec',
                {'throughput': progress_metrics.samples_per_second, 'threshold': self._alert_thresholds['throughput_warning_threshold']}
            )
    
    def _generate_alert(self, severity: AlertSeverity, alert_type: str, 
                       message: str, details: Dict[str, Any]) -> None:
        """Generate a training alert."""
        alert_id = f"{self.session_id}_{alert_type}_{int(time.time() * 1000)}"
        
        alert = TrainingAlert(
            alert_id=alert_id,
            severity=severity,
            alert_type=alert_type,
            message=message,
            details=details
        )
        
        self._alerts.append(alert)
        
        # Log alert
        log_method = getattr(self.logger, severity.value, self.logger.info)
        log_method(f"Alert: {message}")
        
        # Trigger alert callbacks
        for callback in self._alert_callbacks[severity]:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    def _check_improvement(self, validation_metrics: Dict[str, float]) -> None:
        """Check if model is improving."""
        val_loss = validation_metrics.get('loss', float('inf'))
        
        # Check if this is the best loss so far
        best_loss = self._best_metrics.get('loss', float('inf'))
        if val_loss < best_loss:
            self._last_improvement_epoch = self._current_epoch
            self._consecutive_stagnation_count = 0
        else:
            self._consecutive_stagnation_count += 1
        
        # Check for stagnation
        if self._consecutive_stagnation_count >= self._alert_thresholds['loss_stagnation_epochs']:
            self._generate_alert(
                AlertSeverity.WARNING,
                'training_stagnation',
                f'No improvement for {self._consecutive_stagnation_count} epochs',
                {
                    'epochs_without_improvement': self._consecutive_stagnation_count,
                    'last_improvement_epoch': self._last_improvement_epoch,
                    'current_loss': val_loss,
                    'best_loss': best_loss
                }
            )
    
    def _update_best_metrics(self, metrics: Dict[str, float]) -> None:
        """Update best metrics tracking."""
        for metric_name, value in metrics.items():
            if metric_name.endswith('loss') or metric_name.endswith('error'):
                # Lower is better
                if metric_name not in self._best_metrics or value < self._best_metrics[metric_name]:
                    self._best_metrics[metric_name] = value
                    self._best_metrics[f'{metric_name}_epoch'] = self._current_epoch
            else:
                # Higher is better (accuracy, etc.)
                if metric_name not in self._best_metrics or value > self._best_metrics[metric_name]:
                    self._best_metrics[metric_name] = value
                    self._best_metrics[f'{metric_name}_epoch'] = self._current_epoch
    
    def _calculate_epoch_statistics(self) -> Dict[str, Any]:
        """Calculate statistics for the current epoch."""
        stats = {}
        
        # Calculate averages for each metric
        for metric_name, values in self._metric_buffers.items():
            if values:
                recent_values = list(values)[-self._total_batches_per_epoch:]
                stats[f'{metric_name}_mean'] = np.mean(recent_values)
                stats[f'{metric_name}_std'] = np.std(recent_values)
                stats[f'{metric_name}_min'] = np.min(recent_values)
                stats[f'{metric_name}_max'] = np.max(recent_values)
                
                # Calculate trend
                if len(recent_values) > 1:
                    trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                    stats[f'{metric_name}_trend'] = trend
        
        return stats
    
    def _update_visualization_data(self, metrics: ProgressMetrics) -> None:
        """Update data for visualization."""
        step = (metrics.epoch - 1) * self._total_batches_per_epoch + metrics.batch
        
        self._plot_data['step'].append(step)
        self._plot_data['loss'].append(metrics.train_loss)
        
        if metrics.train_accuracy is not None:
            self._plot_data['accuracy'].append(metrics.train_accuracy)
        
        if metrics.learning_rate is not None:
            self._plot_data['learning_rate'].append(metrics.learning_rate)
        
        self._plot_data['throughput'].append(metrics.samples_per_second)
        self._plot_data['memory'].append(metrics.memory_usage_mb)
        
        # Trigger visualization update
        if step % self._visualization_interval == 0:
            self._trigger_visualization_callbacks()
    
    def _trigger_progress_callbacks(self, metrics: ProgressMetrics) -> None:
        """Trigger registered progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                self.logger.error(f"Progress callback error: {e}")
    
    def _trigger_callbacks(self, event: str, data: Dict[str, Any]) -> None:
        """Trigger event callbacks."""
        for callback in self._progress_callbacks:
            try:
                if hasattr(callback, f'on_{event}'):
                    getattr(callback, f'on_{event}')(data)
            except Exception as e:
                self.logger.error(f"Callback error for event '{event}': {e}")
    
    def _trigger_visualization_callbacks(self) -> None:
        """Trigger visualization callbacks."""
        for callback in self._visualization_callbacks:
            try:
                callback(self._plot_data)
            except Exception as e:
                self.logger.error(f"Visualization callback error: {e}")
    
    def _log_progress(self, metrics: ProgressMetrics) -> None:
        """Log training progress."""
        progress_percent = metrics.get_progress_percentage()
        eta_str = self._format_time(metrics.estimated_time_remaining)
        
        log_message = (
            f"[Epoch {metrics.epoch}/{self._total_epochs}] "
            f"[Batch {metrics.batch}/{metrics.total_batches}] "
            f"Progress: {progress_percent:.1f}% | "
            f"Loss: {metrics.train_loss:.4f} | "
            f"Speed: {metrics.samples_per_second:.1f} samples/s | "
            f"ETA: {eta_str}"
        )
        
        if metrics.train_accuracy is not None:
            log_message += f" | Acc: {metrics.train_accuracy:.4f}"
        
        self.logger.info(log_message)
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m {seconds%60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def _start_background_monitoring(self) -> None:
        """Start background monitoring thread."""
        self._monitor_thread = threading.Thread(
            target=self._background_monitor,
            daemon=True
        )
        self._monitor_thread.start()
    
    def _background_monitor(self) -> None:
        """Background monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                # Update resource metrics
                metrics = self._resource_monitor.get_current_metrics()
                self._resource_history.append({
                    'timestamp': time.time(),
                    'metrics': metrics
                })
                
                # Sleep for monitoring interval
                time.sleep(self.config.get('monitoring_interval', 5))
                
            except Exception as e:
                self.logger.error(f"Background monitoring error: {e}")
    
    def register_progress_callback(self, callback: Callable) -> None:
        """Register a progress callback."""
        self._progress_callbacks.append(callback)
    
    def register_alert_callback(self, severity: AlertSeverity, callback: Callable) -> None:
        """Register an alert callback for specific severity."""
        self._alert_callbacks[severity].append(callback)
    
    def register_visualization_callback(self, callback: Callable) -> None:
        """Register a visualization callback."""
        self._visualization_callbacks.append(callback)
    
    def get_current_progress(self) -> Dict[str, Any]:
        """Get current training progress."""
        with self._lock:
            total_batches = self._total_epochs * self._total_batches_per_epoch
            current_batch = (self._current_epoch - 1) * self._total_batches_per_epoch + self._current_batch
            
            progress = {
                'epoch': self._current_epoch,
                'batch': self._current_batch,
                'total_epochs': self._total_epochs,
                'total_batches': total_batches,
                'current_batch_overall': current_batch,
                'progress_percent': (current_batch / total_batches * 100) if total_batches > 0 else 0,
                'training_time': time.time() - self._training_start_time if self._training_start_time else 0,
                'best_metrics': self._best_metrics.copy(),
                'active_alerts': len([a for a in self._alerts if not a.resolved]),
                'total_alerts': len(self._alerts)
            }
            
            # Add latest metrics
            if self._metrics_history:
                progress['latest_metrics'] = self._metrics_history[-1]
            
            # Add ETA
            if self._batch_times:
                avg_batch_time = np.mean(list(self._batch_times))
                batches_remaining = total_batches - current_batch
                progress['eta_seconds'] = batches_remaining * avg_batch_time
                progress['eta_formatted'] = self._format_time(progress['eta_seconds'])
            
            return progress
    
    def get_metrics_history(self, metric_name: Optional[str] = None) -> Dict[str, List[float]]:
        """Get metrics history."""
        with self._lock:
            if metric_name:
                return list(self._metric_buffers.get(metric_name, []))
            else:
                return {name: list(values) for name, values in self._metric_buffers.items()}
    
    def get_alerts(self, severity: Optional[AlertSeverity] = None, 
                   unresolved_only: bool = False) -> List[Dict[str, Any]]:
        """Get training alerts."""
        with self._lock:
            alerts = self._alerts
            
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            if unresolved_only:
                alerts = [a for a in alerts if not a.resolved]
            
            return [a.to_dict() for a in alerts]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.resolution_time = datetime.now()
                    return True
            return False
    
    def create_progress_bar(self) -> Optional['ProgressBar']:
        """Create a progress bar for training."""
        if TQDM_AVAILABLE:
            return ProgressBar(self)
        else:
            self.logger.warning("tqdm not available, progress bar disabled")
            return None
    
    def plot_metrics(self, metrics: Optional[List[str]] = None, 
                    save_path: Optional[Path] = None) -> Optional[Figure]:
        """
        Plot training metrics.
        
        Args:
            metrics: List of metrics to plot (default: all)
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure or None if not available
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available, plotting disabled")
            return None
        
        with self._lock:
            if not self._plot_data.get('step'):
                self.logger.warning("No data to plot")
                return None
            
            # Select metrics to plot
            if metrics is None:
                metrics = ['loss', 'accuracy', 'learning_rate', 'throughput']
            
            available_metrics = [m for m in metrics if m in self._plot_data and self._plot_data[m]]
            
            if not available_metrics:
                self.logger.warning("No available metrics to plot")
                return None
            
            # Create figure
            fig, axes = plt.subplots(len(available_metrics), 1, 
                                   figsize=(10, 4 * len(available_metrics)),
                                   squeeze=False)
            
            steps = self._plot_data['step']
            
            for idx, metric in enumerate(available_metrics):
                ax = axes[idx, 0]
                values = self._plot_data[metric]
                
                # Plot metric
                ax.plot(steps[:len(values)], values, label=metric)
                ax.set_xlabel('Training Step')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} Over Time')
                ax.grid(True, alpha=0.3)
                
                # Add trend line if enough data
                if len(values) > 10:
                    z = np.polyfit(steps[:len(values)], values, 1)
                    p = np.poly1d(z)
                    ax.plot(steps[:len(values)], p(steps[:len(values)]), 
                           "--", alpha=0.5, label='trend')
                
                ax.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Metrics plot saved to {save_path}")
            
            return fig
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        with self._lock:
            total_time = time.time() - self._training_start_time if self._training_start_time else 0
            
            report = {
                'session_id': self.session_id,
                'status': 'completed' if self._current_epoch >= self._total_epochs else 'in_progress',
                'progress': self.get_current_progress(),
                'duration': {
                    'total_seconds': total_time,
                    'formatted': self._format_time(total_time)
                },
                'performance': {
                    'best_metrics': self._best_metrics.copy(),
                    'final_metrics': self._metrics_history[-1] if self._metrics_history else None,
                    'average_throughput': np.mean(list(self._metric_buffers.get('throughput', [0])))
                },
                'resource_usage': {
                    'peak_memory_mb': max(self._metric_buffers.get('memory_usage_mb', [0])),
                    'average_memory_mb': np.mean(list(self._metric_buffers.get('memory_usage_mb', [0]))),
                    'peak_gpu_percent': max(self._metric_buffers.get('gpu_utilization', [0])),
                    'average_gpu_percent': np.mean(list(self._metric_buffers.get('gpu_utilization', [0])))
                },
                'alerts_summary': {
                    'total_alerts': len(self._alerts),
                    'critical_alerts': len([a for a in self._alerts if a.severity == AlertSeverity.CRITICAL]),
                    'error_alerts': len([a for a in self._alerts if a.severity == AlertSeverity.ERROR]),
                    'warning_alerts': len([a for a in self._alerts if a.severity == AlertSeverity.WARNING]),
                    'unresolved_alerts': len([a for a in self._alerts if not a.resolved])
                }
            }
            
            return report
    
    def save_progress(self, save_path: Path) -> None:
        """Save progress data to file."""
        with self._lock:
            progress_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'configuration': self.config,
                'current_state': {
                    'epoch': self._current_epoch,
                    'batch': self._current_batch,
                    'total_epochs': self._total_epochs,
                    'total_batches_per_epoch': self._total_batches_per_epoch
                },
                'metrics_history': self._metrics_history,
                'best_metrics': self._best_metrics,
                'alerts': [a.to_dict() for a in self._alerts],
                'resource_history': list(self._resource_history)
            }
            
            with open(save_path, 'w') as f:
                json.dump(progress_data, f, indent=2)
            
            self.logger.info(f"Progress saved to {save_path}")
    
    def stop(self) -> None:
        """Stop progress tracking."""
        self._stop_monitoring.set()
        
        # Generate final report
        report = self.generate_report()
        self.logger.info(f"Training stopped. Final report: {json.dumps(report, indent=2)}")


class ResourceMonitor:
    """Monitor system resource usage."""
    
    def __init__(self):
        self._last_update = time.time()
        self._metrics = {
            'memory_mb': 0,
            'cpu_percent': 0,
            'gpu_percent': 0,
            'gpu_memory_mb': 0
        }
        
        # Try to import resource monitoring libraries
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            self.psutil = None
        
        try:
            import GPUtil
            self.gputil = GPUtil
        except ImportError:
            self.gputil = None
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current resource metrics."""
        current_time = time.time()
        
        # Update metrics if enough time has passed
        if current_time - self._last_update > 1.0:  # Update every second
            self._update_metrics()
            self._last_update = current_time
        
        return self._metrics.copy()
    
    def _update_metrics(self) -> None:
        """Update resource metrics."""
        # CPU and Memory metrics
        if self.psutil:
            try:
                process = self.psutil.Process()
                self._metrics['memory_mb'] = process.memory_info().rss / 1024 / 1024
                self._metrics['cpu_percent'] = process.cpu_percent(interval=0.1)
            except Exception:
                pass
        
        # GPU metrics
        if self.gputil:
            try:
                gpus = self.gputil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    self._metrics['gpu_percent'] = gpu.load * 100
                    self._metrics['gpu_memory_mb'] = gpu.memoryUsed
            except Exception:
                pass


class ProgressBar:
    """Progress bar wrapper for training visualization."""
    
    def __init__(self, tracker: ProgressTracker):
        self.tracker = tracker
        self._pbar = None
        self._epoch_pbar = None
    
    def __enter__(self):
        """Context manager entry."""
        if TQDM_AVAILABLE:
            # Create nested progress bars
            progress = self.tracker.get_current_progress()
            
            # Epoch progress bar
            self._epoch_pbar = tqdm(
                total=progress['total_epochs'],
                desc="Epochs",
                position=0,
                leave=True
            )
            
            # Batch progress bar
            self._pbar = tqdm(
                total=progress['total_batches'],
                desc="Batches",
                position=1,
                leave=False
            )
            
            # Register callback to update progress bars
            self.tracker.register_progress_callback(self._update_progress)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._pbar:
            self._pbar.close()
        if self._epoch_pbar:
            self._epoch_pbar.close()
    
    def _update_progress(self, metrics: ProgressMetrics) -> None:
        """Update progress bars."""
        if self._pbar:
            # Update batch progress
            current_batch = (metrics.epoch - 1) * metrics.total_batches + metrics.batch
            self._pbar.n = current_batch
            self._pbar.refresh()
            
            # Update postfix with metrics
            postfix = {
                'loss': f"{metrics.train_loss:.4f}",
                'acc': f"{metrics.train_accuracy:.4f}" if metrics.train_accuracy else "N/A",
                'speed': f"{metrics.samples_per_second:.1f}samples/s"
            }
            self._pbar.set_postfix(postfix)
        
        if self._epoch_pbar:
            # Update epoch progress
            self._epoch_pbar.n = metrics.epoch
            self._epoch_pbar.refresh()


def create_console_dashboard(tracker: ProgressTracker) -> 'ConsoleDashboard':
    """Create a console dashboard for training monitoring."""
    return ConsoleDashboard(tracker)


class ConsoleDashboard:
    """Console-based training dashboard."""
    
    def __init__(self, tracker: ProgressTracker):
        self.tracker = tracker
        self._running = False
        self._update_thread = None
    
    def start(self, update_interval: float = 1.0) -> None:
        """Start the dashboard."""
        self._running = True
        self._update_thread = threading.Thread(
            target=self._update_loop,
            args=(update_interval,),
            daemon=True
        )
        self._update_thread.start()
    
    def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=2.0)
    
    def _update_loop(self, interval: float) -> None:
        """Dashboard update loop."""
        while self._running:
            try:
                self._render_dashboard()
                time.sleep(interval)
            except Exception as e:
                print(f"Dashboard error: {e}")
    
    def _render_dashboard(self) -> None:
        """Render the dashboard."""
        # Clear screen (platform-specific)
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Get current progress
        progress = self.tracker.get_current_progress()
        metrics_history = self.tracker.get_metrics_history()
        alerts = self.tracker.get_alerts(unresolved_only=True)
        
        # Header
        print("=" * 80)
        print(f"Training Dashboard - Session: {self.tracker.session_id}")
        print("=" * 80)
        
        # Progress
        print(f"\nProgress: Epoch {progress['epoch']}/{progress['total_epochs']} | "
              f"Batch {progress['batch']}/{progress['total_batches']} | "
              f"{progress['progress_percent']:.1f}% Complete")
        
        # Progress bar
        bar_length = 50
        filled = int(bar_length * progress['progress_percent'] / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"[{bar}] ETA: {progress.get('eta_formatted', 'N/A')}")
        
        # Metrics
        print("\nCurrent Metrics:")
        if 'latest_metrics' in progress:
            latest = progress['latest_metrics']
            if 'metrics' in latest:
                for key, value in latest['metrics'].items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
        
        # Best metrics
        print("\nBest Metrics:")
        for key, value in progress['best_metrics'].items():
            if not key.endswith('_epoch') and isinstance(value, (int, float)):
                epoch = progress['best_metrics'].get(f'{key}_epoch', 'N/A')
                print(f"  {key}: {value:.4f} (epoch {epoch})")
        
        # Alerts
        if alerts:
            print(f"\nActive Alerts ({len(alerts)}):")
            for alert in alerts[:5]:  # Show top 5
                severity_symbol = {
                    'critical': '🔴',
                    'error': '🟠',
                    'warning': '🟡',
                    'info': '🔵'
                }.get(alert['severity'], '⚪')
                print(f"  {severity_symbol} [{alert['severity'].upper()}] {alert['message']}")
        
        # Resource usage
        if 'memory_usage_mb' in metrics_history:
            memory_values = metrics_history['memory_usage_mb']
            if memory_values:
                current_memory = memory_values[-1]
                peak_memory = max(memory_values)
                print(f"\nResource Usage:")
                print(f"  Memory: {current_memory:.0f}MB (peak: {peak_memory:.0f}MB)")
        
        print("\n" + "=" * 80)


# Context manager for null context
class nullcontext:
    """Null context manager for when progress bar is not available."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass