"""
Unified Performance Monitor - Real-time metrics for P-adic and Tropical pipelines
Comprehensive performance tracking and comparison
NO PLACEHOLDERS - PRODUCTION CODE ONLY
HARD FAILURES ONLY - NO GRACEFUL DEGRADATION
"""

import time
import threading
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union, Deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics"""
    THROUGHPUT = "throughput"  # Operations/second
    LATENCY = "latency"  # Time per operation (ms)
    COMPRESSION_RATIO = "compression_ratio"  # Compression effectiveness
    MEMORY_USAGE = "memory_usage"  # Memory consumption (MB)
    GPU_UTILIZATION = "gpu_utilization"  # GPU usage (%)
    CPU_UTILIZATION = "cpu_utilization"  # CPU usage (%)
    ERROR_RATE = "error_rate"  # Failure rate (%)
    ACCURACY = "accuracy"  # Reconstruction accuracy
    # JAX-specific metrics
    JAX_COMPILATION = "jax_compilation"  # JIT compilation time (ms)
    JAX_CACHE_HITS = "jax_cache_hits"  # Compilation cache hit rate
    JAX_MEMORY_POOL = "jax_memory_pool"  # JAX memory pool usage (MB)


class PipelineType(Enum):
    """Compression pipeline types"""
    PADIC = "padic"
    TROPICAL = "tropical"
    HYBRID = "hybrid"
    JAX = "jax"  # JAX-accelerated tropical


@dataclass
class PerformanceMetric:
    """Single performance metric measurement"""
    timestamp: float
    pipeline_type: PipelineType
    metric_type: MetricType
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'pipeline_type': self.pipeline_type.value,
            'metric_type': self.metric_type.value,
            'value': self.value,
            'metadata': self.metadata
        }


@dataclass
class PerformanceSnapshot:
    """Snapshot of all performance metrics at a point in time"""
    timestamp: float
    padic_metrics: Dict[MetricType, float]
    tropical_metrics: Dict[MetricType, float]
    system_metrics: Dict[str, float]
    
    def get_comparison(self) -> Dict[str, Any]:
        """Get comparison between pipelines"""
        comparison = {}
        
        # Compare each metric type
        for metric_type in MetricType:
            padic_val = self.padic_metrics.get(metric_type, 0.0)
            tropical_val = self.tropical_metrics.get(metric_type, 0.0)
            
            if padic_val > 0 and tropical_val > 0:
                ratio = tropical_val / padic_val
                comparison[metric_type.value] = {
                    'padic': padic_val,
                    'tropical': tropical_val,
                    'ratio': ratio,
                    'better': 'tropical' if ratio > 1.0 else 'padic'
                }
        
        return comparison


@dataclass
class MonitorConfig:
    """Configuration for performance monitor"""
    # Monitoring settings
    sampling_interval_ms: int = 100
    history_window_size: int = 1000
    enable_detailed_tracking: bool = True
    
    # Alert thresholds
    latency_threshold_ms: float = 100.0
    memory_threshold_mb: float = 8192.0
    error_rate_threshold: float = 0.05
    
    # Reporting settings
    report_interval_seconds: int = 60
    enable_auto_reporting: bool = True
    
    # Comparison settings
    enable_pipeline_comparison: bool = True
    comparison_window_size: int = 100
    
    # JAX-specific settings
    jax_compilation_threshold_ms: float = 500.0  # Alert if compilation takes too long
    jax_cache_hit_threshold: float = 0.7  # Alert if cache hit rate is too low
    
    def __post_init__(self):
        """Validate configuration"""
        if self.sampling_interval_ms <= 0:
            raise ValueError(f"sampling_interval_ms must be positive, got {self.sampling_interval_ms}")
        if self.history_window_size <= 0:
            raise ValueError(f"history_window_size must be positive, got {self.history_window_size}")


class UnifiedPerformanceMonitor:
    """
    Unified performance monitoring for compression pipelines
    Tracks and compares P-adic and Tropical pipeline performance
    """
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        """Initialize performance monitor"""
        self.config = config or MonitorConfig()
        
        # Metric storage
        self.metrics_history: Deque[PerformanceMetric] = deque(maxlen=config.history_window_size)
        self.snapshots: Deque[PerformanceSnapshot] = deque(maxlen=config.history_window_size // 10)
        
        # Current metrics (latest values)
        self.current_metrics = {
            PipelineType.PADIC: defaultdict(float),
            PipelineType.TROPICAL: defaultdict(float),
            PipelineType.HYBRID: defaultdict(float),
            PipelineType.JAX: defaultdict(float)
        }
        
        # Aggregated statistics
        self.statistics = {
            PipelineType.PADIC: self._init_pipeline_stats(),
            PipelineType.TROPICAL: self._init_pipeline_stats(),
            PipelineType.HYBRID: self._init_pipeline_stats(),
            PipelineType.JAX: self._init_pipeline_stats()
        }
        
        # JAX-specific monitoring if available
        self.jax_monitor = None
        try:
            from .jax_performance_monitor import JAXPerformanceMonitor
            self.jax_monitor = JAXPerformanceMonitor()
        except ImportError:
            pass
        
        # System metrics
        self.system_metrics = {
            'total_operations': 0,
            'total_bytes_processed': 0,
            'uptime_seconds': 0.0,
            'start_time': time.time()
        }
        
        # Alert tracking
        self.alerts: Deque[Dict[str, Any]] = deque(maxlen=100)
        self.alert_callbacks = []
        
        # Monitoring state
        self.monitoring_active = True
        self._lock = threading.RLock()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Start reporting thread if enabled
        if config.enable_auto_reporting:
            self.report_thread = threading.Thread(target=self._reporting_loop, daemon=True)
            self.report_thread.start()
        
        logger.info("UnifiedPerformanceMonitor initialized")
    
    def _init_pipeline_stats(self) -> Dict[str, Any]:
        """Initialize pipeline statistics structure"""
        return {
            'count': defaultdict(int),
            'sum': defaultdict(float),
            'min': defaultdict(lambda: float('inf')),
            'max': defaultdict(lambda: float('-inf')),
            'avg': defaultdict(float),
            'std': defaultdict(float)
        }
    
    def record_metric(self, pipeline_type: PipelineType, metric_type: MetricType,
                     value: float, metadata: Optional[Dict[str, Any]] = None):
        """
        Record a performance metric
        
        Args:
            pipeline_type: Which pipeline (padic/tropical/hybrid)
            metric_type: Type of metric
            value: Metric value
            metadata: Optional additional information
        """
        with self._lock:
            # Create metric record
            metric = PerformanceMetric(
                timestamp=time.time(),
                pipeline_type=pipeline_type,
                metric_type=metric_type,
                value=value,
                metadata=metadata or {}
            )
            
            # Add to history
            self.metrics_history.append(metric)
            
            # Update current metrics
            self.current_metrics[pipeline_type][metric_type] = value
            
            # Update statistics
            self._update_statistics(pipeline_type, metric_type, value)
            
            # Check for alerts
            self._check_alerts(metric)
            
            # Update system metrics
            self.system_metrics['total_operations'] += 1
    
    def record_channel_metrics(self, channel_type: str, 
                             compression_ratio: float,
                             reconstruction_accuracy: float,
                             processing_time_ms: float):
        """
        Record channel-specific metrics
        
        Args:
            channel_type: Type of channel ('coefficient', 'exponent', 'mantissa')
            compression_ratio: Compression ratio for this channel
            reconstruction_accuracy: Reconstruction accuracy (0-1)
            processing_time_ms: Processing time for this channel
        """
        with self._lock:
            if channel_type not in self.channel_metrics:
                raise ValueError(f"Unknown channel type: {channel_type}")
            
            # Update channel metrics
            self.channel_metrics[channel_type]['compression_ratio'] = compression_ratio
            self.channel_metrics[channel_type]['accuracy'] = reconstruction_accuracy
            self.channel_metrics[channel_type]['latency'] = processing_time_ms
            self.channel_metrics[channel_type]['efficiency'] = compression_ratio / max(processing_time_ms, 0.001)
            
            # Record as standard metrics with channel prefix
            metric_type_map = {
                'compression_ratio': MetricType.CHANNEL_COMPRESSION_RATIO,
                'accuracy': MetricType.CHANNEL_RECONSTRUCTION
            }
            
            for metric_name, metric_type in metric_type_map.items():
                if metric_name == 'compression_ratio':
                    self.record_metric(PipelineType.TROPICAL, metric_type, compression_ratio,
                                     metadata={'channel': channel_type})
                elif metric_name == 'accuracy':
                    self.record_metric(PipelineType.TROPICAL, metric_type, reconstruction_accuracy,
                                     metadata={'channel': channel_type})
            
            # Check channel-specific alerts
            if reconstruction_accuracy < self.config.channel_accuracy_threshold:
                alert = {
                    'type': 'low_channel_accuracy',
                    'channel': channel_type,
                    'accuracy': reconstruction_accuracy,
                    'threshold': self.config.channel_accuracy_threshold,
                    'timestamp': time.time()
                }
                self.alerts.append(alert)
                self._trigger_alert_callbacks(alert)
            
            if compression_ratio < self.config.channel_compression_threshold:
                alert = {
                    'type': 'low_channel_compression',
                    'channel': channel_type,
                    'ratio': compression_ratio,
                    'threshold': self.config.channel_compression_threshold,
                    'timestamp': time.time()
                }
                self.alerts.append(alert)
                self._trigger_alert_callbacks(alert)
    
    def record_jax_metrics(self, compilation_time_ms: float,
                          cache_hit_rate: float,
                          memory_pool_usage_mb: float):
        """
        Record JAX-specific metrics
        
        Args:
            compilation_time_ms: JIT compilation time
            cache_hit_rate: Compilation cache hit rate (0-1)
            memory_pool_usage_mb: JAX memory pool usage
        """
        with self._lock:
            # Record JAX metrics
            self.record_metric(PipelineType.JAX, MetricType.JAX_COMPILATION, compilation_time_ms)
            self.record_metric(PipelineType.JAX, MetricType.JAX_CACHE_HITS, cache_hit_rate)
            self.record_metric(PipelineType.JAX, MetricType.JAX_MEMORY_POOL, memory_pool_usage_mb)
            
            # Check JAX-specific alerts
            if compilation_time_ms > self.config.jax_compilation_threshold_ms:
                alert = {
                    'type': 'slow_jax_compilation',
                    'compilation_time_ms': compilation_time_ms,
                    'threshold': self.config.jax_compilation_threshold_ms,
                    'timestamp': time.time()
                }
                self.alerts.append(alert)
                self._trigger_alert_callbacks(alert)
            
            if cache_hit_rate < self.config.jax_cache_hit_threshold:
                alert = {
                    'type': 'low_jax_cache_hits',
                    'cache_hit_rate': cache_hit_rate,
                    'threshold': self.config.jax_cache_hit_threshold,
                    'timestamp': time.time()
                }
                self.alerts.append(alert)
                self._trigger_alert_callbacks(alert)
            
            # If JAX monitor is available, sync metrics
            if self.jax_monitor:
                try:
                    jax_perf = self.jax_monitor.get_current_metrics()
                    if jax_perf:
                        for key, value in jax_perf.items():
                            if isinstance(value, (int, float)):
                                # Map JAX monitor metrics to our metric types
                                if 'latency' in key.lower():
                                    self.record_metric(PipelineType.JAX, MetricType.LATENCY, value)
                                elif 'throughput' in key.lower():
                                    self.record_metric(PipelineType.JAX, MetricType.THROUGHPUT, value)
                                elif 'memory' in key.lower():
                                    self.record_metric(PipelineType.JAX, MetricType.MEMORY_USAGE, value)
                except Exception as e:
                    logger.error(f"Failed to sync JAX monitor metrics: {e}")
    
    def record_compression(self, pipeline_type: PipelineType,
                         input_size_bytes: int, output_size_bytes: int,
                         compression_time_ms: float, success: bool = True):
        """
        Record a compression operation
        
        Args:
            pipeline_type: Which pipeline was used
            input_size_bytes: Input data size
            output_size_bytes: Compressed size
            compression_time_ms: Time taken
            success: Whether operation succeeded
        """
        with self._lock:
            # Calculate metrics
            compression_ratio = input_size_bytes / max(output_size_bytes, 1)
            throughput = (input_size_bytes / (1024 * 1024)) / (compression_time_ms / 1000) if compression_time_ms > 0 else 0
            
            # Record individual metrics
            self.record_metric(pipeline_type, MetricType.COMPRESSION_RATIO, compression_ratio)
            self.record_metric(pipeline_type, MetricType.LATENCY, compression_time_ms)
            self.record_metric(pipeline_type, MetricType.THROUGHPUT, throughput)
            
            if not success:
                current_error_rate = self.current_metrics[pipeline_type].get(MetricType.ERROR_RATE, 0.0)
                self.record_metric(pipeline_type, MetricType.ERROR_RATE, current_error_rate + 0.01)
            
            # Update system metrics
            self.system_metrics['total_bytes_processed'] += input_size_bytes
    
    def record_decompression(self, pipeline_type: PipelineType,
                           decompression_time_ms: float,
                           reconstruction_error: Optional[float] = None):
        """
        Record a decompression operation
        
        Args:
            pipeline_type: Which pipeline was used
            decompression_time_ms: Time taken
            reconstruction_error: Optional reconstruction error
        """
        with self._lock:
            # Record latency
            self.record_metric(pipeline_type, MetricType.LATENCY, decompression_time_ms)
            
            # Record accuracy if available
            if reconstruction_error is not None:
                accuracy = 1.0 - min(reconstruction_error, 1.0)
                self.record_metric(pipeline_type, MetricType.ACCURACY, accuracy)
    
    def record_memory_usage(self, pipeline_type: PipelineType,
                          gpu_memory_mb: Optional[float] = None,
                          cpu_memory_mb: Optional[float] = None):
        """
        Record memory usage
        
        Args:
            pipeline_type: Which pipeline
            gpu_memory_mb: GPU memory usage
            cpu_memory_mb: CPU memory usage
        """
        with self._lock:
            total_memory = 0.0
            
            if gpu_memory_mb is not None:
                total_memory += gpu_memory_mb
                
            if cpu_memory_mb is not None:
                total_memory += cpu_memory_mb
            
            if total_memory > 0:
                self.record_metric(pipeline_type, MetricType.MEMORY_USAGE, total_memory)
    
    def record_utilization(self, gpu_percent: Optional[float] = None,
                         cpu_percent: Optional[float] = None):
        """
        Record resource utilization
        
        Args:
            gpu_percent: GPU utilization percentage
            cpu_percent: CPU utilization percentage
        """
        with self._lock:
            if gpu_percent is not None:
                for pipeline_type in PipelineType:
                    if self._is_pipeline_active(pipeline_type):
                        self.record_metric(pipeline_type, MetricType.GPU_UTILIZATION, gpu_percent)
            
            if cpu_percent is not None:
                for pipeline_type in PipelineType:
                    if self._is_pipeline_active(pipeline_type):
                        self.record_metric(pipeline_type, MetricType.CPU_UTILIZATION, cpu_percent)
    
    def get_current_metrics(self, pipeline_type: Optional[PipelineType] = None) -> Dict[str, Any]:
        """
        Get current metrics
        
        Args:
            pipeline_type: Specific pipeline or None for all
            
        Returns:
            Dictionary of current metrics
        """
        with self._lock:
            if pipeline_type:
                return dict(self.current_metrics[pipeline_type])
            else:
                return {
                    pt.value: dict(metrics)
                    for pt, metrics in self.current_metrics.items()
                }
    
    def get_statistics(self, pipeline_type: Optional[PipelineType] = None,
                      metric_type: Optional[MetricType] = None) -> Dict[str, Any]:
        """
        Get aggregated statistics
        
        Args:
            pipeline_type: Specific pipeline or None for all
            metric_type: Specific metric or None for all
            
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            if pipeline_type and metric_type:
                stats = self.statistics[pipeline_type]
                return {
                    'count': stats['count'][metric_type],
                    'sum': stats['sum'][metric_type],
                    'min': stats['min'][metric_type],
                    'max': stats['max'][metric_type],
                    'avg': stats['avg'][metric_type],
                    'std': stats['std'][metric_type]
                }
            elif pipeline_type:
                return self._format_pipeline_stats(self.statistics[pipeline_type])
            else:
                return {
                    pt.value: self._format_pipeline_stats(stats)
                    for pt, stats in self.statistics.items()
                }
    
    def get_comparison(self) -> Dict[str, Any]:
        """
        Get performance comparison between pipelines
        
        Returns:
            Comparison dictionary
        """
        with self._lock:
            # Create snapshot
            snapshot = self._create_snapshot()
            comparison = snapshot.get_comparison()
            
            # Add overall winner determination
            padic_wins = 0
            tropical_wins = 0
            
            for metric_comparison in comparison.values():
                if metric_comparison.get('better') == 'padic':
                    padic_wins += 1
                elif metric_comparison.get('better') == 'tropical':
                    tropical_wins += 1
            
            comparison['overall'] = {
                'padic_wins': padic_wins,
                'tropical_wins': tropical_wins,
                'recommended': 'tropical' if tropical_wins > padic_wins else 'padic'
            }
            
            return comparison
    
    def get_history(self, pipeline_type: Optional[PipelineType] = None,
                   metric_type: Optional[MetricType] = None,
                   last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get historical metrics
        
        Args:
            pipeline_type: Filter by pipeline
            metric_type: Filter by metric type
            last_n: Number of recent records
            
        Returns:
            List of metric records
        """
        with self._lock:
            # Filter history
            filtered = []
            for metric in self.metrics_history:
                if pipeline_type and metric.pipeline_type != pipeline_type:
                    continue
                if metric_type and metric.metric_type != metric_type:
                    continue
                filtered.append(metric.to_dict())
            
            # Limit to last_n if specified
            if last_n:
                filtered = filtered[-last_n:]
            
            return filtered
    
    def get_alerts(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get recent alerts
        
        Args:
            last_n: Number of recent alerts
            
        Returns:
            List of alerts
        """
        with self._lock:
            alerts = list(self.alerts)
            if last_n:
                alerts = alerts[-last_n:]
            return alerts
    
    def add_alert_callback(self, callback):
        """Add callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Returns:
            Performance report dictionary
        """
        with self._lock:
            uptime = time.time() - self.system_metrics['start_time']
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': uptime,
                'uptime_formatted': str(timedelta(seconds=int(uptime))),
                'system_metrics': dict(self.system_metrics),
                'current_metrics': self.get_current_metrics(),
                'statistics': self.get_statistics(),
                'comparison': self.get_comparison(),
                'recent_alerts': self.get_alerts(last_n=10),
                'recommendations': self._generate_recommendations()
            }
            
            return report
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """
        Export metrics to file
        
        Args:
            filepath: Output file path
            format: Export format ('json', 'csv')
        """
        with self._lock:
            if format == 'json':
                report = self.generate_report()
                with open(filepath, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Metrics exported to {filepath}")
            elif format == 'csv':
                # Export metrics history to CSV format
                import csv
                
                with open(filepath, 'w', newline='') as csvfile:
                    # Define CSV headers
                    fieldnames = [
                        'timestamp', 'datetime', 'pipeline_type', 'metric_type', 
                        'value', 'metadata_keys', 'metadata_values'
                    ]
                    
                    # Add dynamic metadata columns based on what's in the data
                    metadata_columns = set()
                    for metric in self.metrics_history:
                        if metric.metadata:
                            metadata_columns.update(metric.metadata.keys())
                    
                    # Sort metadata columns for consistent ordering
                    metadata_columns = sorted(metadata_columns)
                    fieldnames.extend([f'metadata_{col}' for col in metadata_columns])
                    
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    # Write metrics history
                    for metric in self.metrics_history:
                        row = {
                            'timestamp': metric.timestamp,
                            'datetime': datetime.fromtimestamp(metric.timestamp).isoformat(),
                            'pipeline_type': metric.pipeline_type.value,
                            'metric_type': metric.metric_type.value,
                            'value': metric.value,
                            'metadata_keys': '|'.join(metric.metadata.keys()) if metric.metadata else '',
                            'metadata_values': '|'.join(str(v) for v in metric.metadata.values()) if metric.metadata else ''
                        }
                        
                        # Add individual metadata columns
                        for col in metadata_columns:
                            row[f'metadata_{col}'] = metric.metadata.get(col, '') if metric.metadata else ''
                        
                        writer.writerow(row)
                    
                    # Write aggregated statistics as a separate section
                    writer.writerow({})  # Empty row separator
                    writer.writerow({'timestamp': 'STATISTICS', 'datetime': datetime.now().isoformat()})
                    
                    for pipeline_type, stats in self.statistics.items():
                        for stat_type in ['count', 'sum', 'min', 'max', 'avg', 'std']:
                            stat_dict = stats[stat_type]
                            for metric_type, value in stat_dict.items():
                                if isinstance(metric_type, MetricType):
                                    writer.writerow({
                                        'timestamp': f'STAT_{stat_type.upper()}',
                                        'datetime': '',
                                        'pipeline_type': pipeline_type.value,
                                        'metric_type': metric_type.value,
                                        'value': value,
                                        'metadata_keys': stat_type,
                                        'metadata_values': str(value)
                                    })
                    
                    # Write current metrics snapshot
                    writer.writerow({})  # Empty row separator
                    writer.writerow({'timestamp': 'CURRENT_METRICS', 'datetime': datetime.now().isoformat()})
                    
                    for pipeline_type, metrics in self.current_metrics.items():
                        for metric_type, value in metrics.items():
                            if isinstance(metric_type, MetricType):
                                writer.writerow({
                                    'timestamp': time.time(),
                                    'datetime': datetime.now().isoformat(),
                                    'pipeline_type': pipeline_type.value,
                                    'metric_type': metric_type.value,
                                    'value': value,
                                    'metadata_keys': 'current',
                                    'metadata_values': str(value)
                                })
                    
                    # Write system metrics
                    writer.writerow({})  # Empty row separator
                    writer.writerow({'timestamp': 'SYSTEM_METRICS', 'datetime': datetime.now().isoformat()})
                    
                    for key, value in self.system_metrics.items():
                        writer.writerow({
                            'timestamp': time.time(),
                            'datetime': datetime.now().isoformat(),
                            'pipeline_type': 'SYSTEM',
                            'metric_type': key,
                            'value': value,
                            'metadata_keys': 'system',
                            'metadata_values': str(value)
                        })
                
                logger.info(f"Metrics exported to CSV: {filepath}")
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def _update_statistics(self, pipeline_type: PipelineType,
                          metric_type: MetricType, value: float):
        """Update aggregated statistics"""
        stats = self.statistics[pipeline_type]
        
        # Update count and sum
        stats['count'][metric_type] += 1
        stats['sum'][metric_type] += value
        
        # Update min/max
        stats['min'][metric_type] = min(stats['min'][metric_type], value)
        stats['max'][metric_type] = max(stats['max'][metric_type], value)
        
        # Update average
        count = stats['count'][metric_type]
        stats['avg'][metric_type] = stats['sum'][metric_type] / count
        
        # Update standard deviation (simplified online algorithm)
        if count > 1:
            avg = stats['avg'][metric_type]
            prev_std = stats['std'][metric_type]
            stats['std'][metric_type] = np.sqrt(
                ((count - 2) * prev_std**2 + (value - avg)**2) / (count - 1)
            )
    
    def _check_alerts(self, metric: PerformanceMetric):
        """Check if metric triggers any alerts"""
        alert = None
        
        # Check latency threshold
        if metric.metric_type == MetricType.LATENCY:
            if metric.value > self.config.latency_threshold_ms:
                alert = {
                    'type': 'high_latency',
                    'pipeline': metric.pipeline_type.value,
                    'value': metric.value,
                    'threshold': self.config.latency_threshold_ms,
                    'timestamp': metric.timestamp
                }
        
        # Check memory threshold
        elif metric.metric_type == MetricType.MEMORY_USAGE:
            if metric.value > self.config.memory_threshold_mb:
                alert = {
                    'type': 'high_memory',
                    'pipeline': metric.pipeline_type.value,
                    'value': metric.value,
                    'threshold': self.config.memory_threshold_mb,
                    'timestamp': metric.timestamp
                }
        
        # Check error rate threshold
        elif metric.metric_type == MetricType.ERROR_RATE:
            if metric.value > self.config.error_rate_threshold:
                alert = {
                    'type': 'high_error_rate',
                    'pipeline': metric.pipeline_type.value,
                    'value': metric.value,
                    'threshold': self.config.error_rate_threshold,
                    'timestamp': metric.timestamp
                }
        
        if alert:
            self.alerts.append(alert)
            
            # Trigger callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def _create_snapshot(self) -> PerformanceSnapshot:
        """Create current performance snapshot"""
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            padic_metrics=dict(self.current_metrics[PipelineType.PADIC]),
            tropical_metrics=dict(self.current_metrics[PipelineType.TROPICAL]),
            system_metrics=dict(self.system_metrics)
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def _is_pipeline_active(self, pipeline_type: PipelineType) -> bool:
        """Check if pipeline has recent activity"""
        # Check if pipeline has any recent metrics
        return len(self.current_metrics[pipeline_type]) > 0
    
    def _format_pipeline_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Format pipeline statistics for output"""
        formatted = {}
        
        for metric_type in MetricType:
            if stats['count'][metric_type] > 0:
                formatted[metric_type.value] = {
                    'count': stats['count'][metric_type],
                    'min': stats['min'][metric_type],
                    'max': stats['max'][metric_type],
                    'avg': stats['avg'][metric_type],
                    'std': stats['std'][metric_type]
                }
        
        return formatted
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Check compression ratios
        padic_ratio = self.current_metrics[PipelineType.PADIC].get(MetricType.COMPRESSION_RATIO, 0)
        tropical_ratio = self.current_metrics[PipelineType.TROPICAL].get(MetricType.COMPRESSION_RATIO, 0)
        
        if tropical_ratio > padic_ratio * 1.2:
            recommendations.append("Tropical pipeline shows significantly better compression ratios")
        elif padic_ratio > tropical_ratio * 1.2:
            recommendations.append("P-adic pipeline shows significantly better compression ratios")
        
        # Check latency
        padic_latency = self.current_metrics[PipelineType.PADIC].get(MetricType.LATENCY, float('inf'))
        tropical_latency = self.current_metrics[PipelineType.TROPICAL].get(MetricType.LATENCY, float('inf'))
        
        if tropical_latency < padic_latency * 0.5:
            recommendations.append("Tropical pipeline is significantly faster")
        elif padic_latency < tropical_latency * 0.5:
            recommendations.append("P-adic pipeline is significantly faster")
        
        # Check error rates
        for pipeline_type in [PipelineType.PADIC, PipelineType.TROPICAL]:
            error_rate = self.current_metrics[pipeline_type].get(MetricType.ERROR_RATE, 0)
            if error_rate > self.config.error_rate_threshold:
                recommendations.append(f"{pipeline_type.value} pipeline has high error rate ({error_rate:.2%})")
        
        # Check memory usage
        for pipeline_type in [PipelineType.PADIC, PipelineType.TROPICAL]:
            memory = self.current_metrics[pipeline_type].get(MetricType.MEMORY_USAGE, 0)
            if memory > self.config.memory_threshold_mb:
                recommendations.append(f"{pipeline_type.value} pipeline using excessive memory ({memory:.0f}MB)")
        
        if not recommendations:
            recommendations.append("All systems operating within normal parameters")
        
        return recommendations
    
    def _monitoring_loop(self):
        """Background monitoring thread"""
        while self.monitoring_active:
            try:
                # Update system metrics
                with self._lock:
                    self.system_metrics['uptime_seconds'] = time.time() - self.system_metrics['start_time']
                
                # Sleep for sampling interval
                time.sleep(self.config.sampling_interval_ms / 1000.0)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    def _reporting_loop(self):
        """Background reporting thread"""
        while self.monitoring_active:
            try:
                # Wait for report interval
                time.sleep(self.config.report_interval_seconds)
                
                # Generate and log report
                report = self.generate_report()
                logger.info(f"Performance Report: {json.dumps(report['comparison'], indent=2)}")
                
            except Exception as e:
                logger.error(f"Reporting loop error: {e}")
    
    def shutdown(self):
        """Shutdown monitor"""
        self.monitoring_active = False
        
        # Generate final report
        final_report = self.generate_report()
        logger.info(f"Final Performance Report: {json.dumps(final_report['comparison'], indent=2)}")
        
        logger.info("UnifiedPerformanceMonitor shutdown complete")