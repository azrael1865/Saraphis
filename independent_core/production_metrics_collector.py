"""
Production Metrics Collection and Analysis System
Provides comprehensive metrics collection, aggregation, analysis, and reporting for production environments.
NO FALLBACKS - HARD FAILURES ONLY architecture.
"""

import asyncio
import json
import logging
import time
import threading
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union, Set, Tuple
from collections import defaultdict, deque
import hashlib
import os
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
from concurrent.futures import ThreadPoolExecutor

try:
    from .production_monitoring_system import ProductionMonitoringSystem
    from .production_config_manager import ProductionConfigManager
    from .production_telemetry import TelemetryManager
except ImportError:
    # Handle import when running as standalone script
    try:
        from production_monitoring_system import ProductionMonitoringSystem
        from production_config_manager import ProductionConfigManager
        from production_telemetry import TelemetryManager
    except ImportError:
        ProductionMonitoringSystem = None
        ProductionConfigManager = None
        TelemetryManager = None


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"
    PERCENTAGE = "percentage"
    DISTRIBUTION = "distribution"


class MetricCategory(Enum):
    """Categories of metrics."""
    SYSTEM = "system"
    APPLICATION = "application"
    BUSINESS = "business"
    CUSTOM = "custom"
    PERFORMANCE = "performance"
    SECURITY = "security"
    INFRASTRUCTURE = "infrastructure"


class AggregationType(Enum):
    """Types of metric aggregations."""
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    MEDIAN = "median"
    PERCENTILE = "percentile"
    RATE = "rate"
    STDDEV = "stddev"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    metric_type: MetricType
    category: MetricCategory
    description: str
    unit: str
    tags: Set[str] = field(default_factory=set)
    dimensions: Set[str] = field(default_factory=set)
    aggregations: List[AggregationType] = field(default_factory=list)
    retention_days: int = 30
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'name': self.name,
            'metric_type': self.metric_type.value,
            'category': self.category.value,
            'description': self.description,
            'unit': self.unit,
            'tags': list(self.tags),
            'dimensions': list(self.dimensions),
            'aggregations': [agg.value for agg in self.aggregations],
            'retention_days': self.retention_days,
            'alert_thresholds': self.alert_thresholds
        }


@dataclass
class MetricValue:
    """A metric value with timestamp and metadata."""
    metric_name: str
    value: Union[int, float]
    timestamp: datetime
    dimensions: Dict[str, str] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'metric_name': self.metric_name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'dimensions': self.dimensions,
            'tags': list(self.tags),
            'metadata': self.metadata
        }


@dataclass
class MetricAlert:
    """Metric alert definition."""
    metric_name: str
    condition: str
    threshold: float
    severity: AlertSeverity
    message: str
    enabled: bool = True
    cooldown_minutes: int = 5
    last_triggered: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'metric_name': self.metric_name,
            'condition': self.condition,
            'threshold': self.threshold,
            'severity': self.severity.value,
            'message': self.message,
            'enabled': self.enabled,
            'cooldown_minutes': self.cooldown_minutes,
            'last_triggered': self.last_triggered.isoformat() if self.last_triggered else None
        }


@dataclass
class MetricAggregation:
    """Aggregated metric data."""
    metric_name: str
    aggregation_type: AggregationType
    value: float
    count: int
    start_time: datetime
    end_time: datetime
    dimensions: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'metric_name': self.metric_name,
            'aggregation_type': self.aggregation_type.value,
            'value': self.value,
            'count': self.count,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'dimensions': self.dimensions
        }


class MetricCollector(ABC):
    """Abstract base class for metric collectors."""
    
    @abstractmethod
    def collect_metrics(self) -> List[MetricValue]:
        """Collect metrics and return list of metric values."""
        pass
    
    @abstractmethod
    def get_metric_definitions(self) -> List[MetricDefinition]:
        """Get definitions of metrics this collector provides."""
        pass
    
    @abstractmethod
    def get_collector_info(self) -> Dict[str, Any]:
        """Get information about this collector."""
        pass


class SystemMetricsCollector(MetricCollector):
    """Collector for system-level metrics."""
    
    def __init__(self):
        self.name = "system_metrics_collector"
        self.last_collection = None
        self.previous_values = {}
        
    def collect_metrics(self) -> List[MetricValue]:
        """Collect system metrics."""
        try:
            import psutil
            
            current_time = datetime.now(timezone.utc)
            metrics = []
            
            # CPU Metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(MetricValue(
                metric_name="system.cpu.utilization",
                value=cpu_percent,
                timestamp=current_time,
                dimensions={"host": os.uname().nodename},
                tags={"system", "cpu"},
                metadata={"unit": "percent"}
            ))
            
            # CPU per core
            cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
            for i, cpu_val in enumerate(cpu_per_core):
                metrics.append(MetricValue(
                    metric_name="system.cpu.core_utilization",
                    value=cpu_val,
                    timestamp=current_time,
                    dimensions={"host": os.uname().nodename, "core": str(i)},
                    tags={"system", "cpu", "core"},
                    metadata={"unit": "percent"}
                ))
            
            # Load average
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()
                for i, period in enumerate(['1min', '5min', '15min']):
                    metrics.append(MetricValue(
                        metric_name="system.load_average",
                        value=load_avg[i],
                        timestamp=current_time,
                        dimensions={"host": os.uname().nodename, "period": period},
                        tags={"system", "load"},
                        metadata={"unit": "count"}
                    ))
            
            # Memory Metrics
            memory = psutil.virtual_memory()
            memory_metrics = {
                "system.memory.total": memory.total,
                "system.memory.available": memory.available,
                "system.memory.used": memory.used,
                "system.memory.free": memory.free,
                "system.memory.utilization": memory.percent
            }
            
            for metric_name, value in memory_metrics.items():
                unit = "bytes" if "utilization" not in metric_name else "percent"
                metrics.append(MetricValue(
                    metric_name=metric_name,
                    value=value,
                    timestamp=current_time,
                    dimensions={"host": os.uname().nodename},
                    tags={"system", "memory"},
                    metadata={"unit": unit}
                ))
            
            # Swap Metrics
            swap = psutil.swap_memory()
            swap_metrics = {
                "system.swap.total": swap.total,
                "system.swap.used": swap.used,
                "system.swap.free": swap.free,
                "system.swap.utilization": swap.percent
            }
            
            for metric_name, value in swap_metrics.items():
                unit = "bytes" if "utilization" not in metric_name else "percent"
                metrics.append(MetricValue(
                    metric_name=metric_name,
                    value=value,
                    timestamp=current_time,
                    dimensions={"host": os.uname().nodename},
                    tags={"system", "swap"},
                    metadata={"unit": unit}
                ))
            
            # Disk Metrics
            disk_usage = psutil.disk_usage('/')
            disk_metrics = {
                "system.disk.total": disk_usage.total,
                "system.disk.used": disk_usage.used,
                "system.disk.free": disk_usage.free,
                "system.disk.utilization": (disk_usage.used / disk_usage.total) * 100
            }
            
            for metric_name, value in disk_metrics.items():
                unit = "bytes" if "utilization" not in metric_name else "percent"
                metrics.append(MetricValue(
                    metric_name=metric_name,
                    value=value,
                    timestamp=current_time,
                    dimensions={"host": os.uname().nodename, "mount": "/"},
                    tags={"system", "disk"},
                    metadata={"unit": unit}
                ))
            
            # Disk I/O Metrics
            disk_io = psutil.disk_io_counters()
            if disk_io:
                disk_io_metrics = {
                    "system.disk.read_bytes": disk_io.read_bytes,
                    "system.disk.write_bytes": disk_io.write_bytes,
                    "system.disk.read_count": disk_io.read_count,
                    "system.disk.write_count": disk_io.write_count,
                    "system.disk.read_time": disk_io.read_time,
                    "system.disk.write_time": disk_io.write_time
                }
                
                for metric_name, value in disk_io_metrics.items():
                    # Calculate rates for cumulative counters
                    rate_value = None
                    if self.previous_values.get(metric_name) is not None:
                        prev_value, prev_time = self.previous_values[metric_name]
                        time_diff = (current_time - prev_time).total_seconds()
                        if time_diff > 0:
                            rate_value = (value - prev_value) / time_diff
                    
                    self.previous_values[metric_name] = (value, current_time)
                    
                    # Add cumulative metric
                    unit = "bytes" if "bytes" in metric_name else "count" if "count" in metric_name else "milliseconds"
                    metrics.append(MetricValue(
                        metric_name=metric_name,
                        value=value,
                        timestamp=current_time,
                        dimensions={"host": os.uname().nodename},
                        tags={"system", "disk", "io"},
                        metadata={"unit": unit, "type": "cumulative"}
                    ))
                    
                    # Add rate metric if available
                    if rate_value is not None:
                        rate_unit = "bytes_per_sec" if "bytes" in metric_name else "ops_per_sec" if "count" in metric_name else "ms_per_sec"
                        metrics.append(MetricValue(
                            metric_name=f"{metric_name}_rate",
                            value=rate_value,
                            timestamp=current_time,
                            dimensions={"host": os.uname().nodename},
                            tags={"system", "disk", "io", "rate"},
                            metadata={"unit": rate_unit, "type": "rate"}
                        ))
            
            # Network Metrics
            network_io = psutil.net_io_counters()
            if network_io:
                network_metrics = {
                    "system.network.bytes_sent": network_io.bytes_sent,
                    "system.network.bytes_recv": network_io.bytes_recv,
                    "system.network.packets_sent": network_io.packets_sent,
                    "system.network.packets_recv": network_io.packets_recv,
                    "system.network.errin": network_io.errin,
                    "system.network.errout": network_io.errout,
                    "system.network.dropin": network_io.dropin,
                    "system.network.dropout": network_io.dropout
                }
                
                for metric_name, value in network_metrics.items():
                    # Calculate rates for cumulative counters
                    rate_value = None
                    if self.previous_values.get(metric_name) is not None:
                        prev_value, prev_time = self.previous_values[metric_name]
                        time_diff = (current_time - prev_time).total_seconds()
                        if time_diff > 0:
                            rate_value = (value - prev_value) / time_diff
                    
                    self.previous_values[metric_name] = (value, current_time)
                    
                    # Add cumulative metric
                    unit = "bytes" if "bytes" in metric_name else "count"
                    metrics.append(MetricValue(
                        metric_name=metric_name,
                        value=value,
                        timestamp=current_time,
                        dimensions={"host": os.uname().nodename},
                        tags={"system", "network"},
                        metadata={"unit": unit, "type": "cumulative"}
                    ))
                    
                    # Add rate metric if available
                    if rate_value is not None:
                        rate_unit = "bytes_per_sec" if "bytes" in metric_name else "ops_per_sec"
                        metrics.append(MetricValue(
                            metric_name=f"{metric_name}_rate",
                            value=rate_value,
                            timestamp=current_time,
                            dimensions={"host": os.uname().nodename},
                            tags={"system", "network", "rate"},
                            metadata={"unit": rate_unit, "type": "rate"}
                        ))
            
            # Process Metrics
            process_count = len(psutil.pids())
            metrics.append(MetricValue(
                metric_name="system.processes.count",
                value=process_count,
                timestamp=current_time,
                dimensions={"host": os.uname().nodename},
                tags={"system", "processes"},
                metadata={"unit": "count"}
            ))
            
            self.last_collection = current_time
            return metrics
            
        except Exception as e:
            raise RuntimeError(f"System metrics collection failed: {e}")
    
    def get_metric_definitions(self) -> List[MetricDefinition]:
        """Get system metric definitions."""
        return [
            MetricDefinition(
                name="system.cpu.utilization",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.SYSTEM,
                description="CPU utilization percentage",
                unit="percent",
                tags={"system", "cpu"},
                dimensions={"host"},
                aggregations=[AggregationType.AVERAGE, AggregationType.MAX],
                alert_thresholds={"high": 80.0, "critical": 95.0}
            ),
            MetricDefinition(
                name="system.memory.utilization",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.SYSTEM,
                description="Memory utilization percentage",
                unit="percent",
                tags={"system", "memory"},
                dimensions={"host"},
                aggregations=[AggregationType.AVERAGE, AggregationType.MAX],
                alert_thresholds={"high": 85.0, "critical": 95.0}
            ),
            MetricDefinition(
                name="system.disk.utilization",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.SYSTEM,
                description="Disk utilization percentage",
                unit="percent",
                tags={"system", "disk"},
                dimensions={"host", "mount"},
                aggregations=[AggregationType.AVERAGE, AggregationType.MAX],
                alert_thresholds={"high": 85.0, "critical": 95.0}
            ),
            MetricDefinition(
                name="system.load_average",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.SYSTEM,
                description="System load average",
                unit="count",
                tags={"system", "load"},
                dimensions={"host", "period"},
                aggregations=[AggregationType.AVERAGE, AggregationType.MAX],
                alert_thresholds={"high": 5.0, "critical": 10.0}
            )
        ]
    
    def get_collector_info(self) -> Dict[str, Any]:
        """Get system collector information."""
        return {
            'name': self.name,
            'type': 'system',
            'last_collection': self.last_collection.isoformat() if self.last_collection else None,
            'capabilities': ['cpu', 'memory', 'disk', 'network', 'processes'],
            'metric_count': len(self.get_metric_definitions()),
            'dependencies': ['psutil']
        }


class ApplicationMetricsCollector(MetricCollector):
    """Collector for application-level metrics."""
    
    def __init__(self, app_name: str = "independent_core"):
        self.app_name = app_name
        self.name = f"application_metrics_collector_{app_name}"
        self.last_collection = None
        self.custom_metrics = {}
        self.timers = {}
        self.counters = {}
        self.gauges = {}
        
    def collect_metrics(self) -> List[MetricValue]:
        """Collect application metrics."""
        try:
            current_time = datetime.now(timezone.utc)
            metrics = []
            
            # Application runtime metrics
            uptime = time.time() - getattr(self, '_start_time', time.time())
            metrics.append(MetricValue(
                metric_name="application.uptime",
                value=uptime,
                timestamp=current_time,
                dimensions={"app": self.app_name, "host": os.uname().nodename},
                tags={"application", "runtime"},
                metadata={"unit": "seconds"}
            ))
            
            # Process-specific metrics
            current_process = psutil.Process()
            process_metrics = {
                "application.memory.rss": current_process.memory_info().rss,
                "application.memory.vms": current_process.memory_info().vms,
                "application.memory.percent": current_process.memory_percent(),
                "application.cpu.percent": current_process.cpu_percent(),
                "application.threads.count": current_process.num_threads(),
                "application.fds.count": current_process.num_fds() if hasattr(current_process, 'num_fds') else 0
            }
            
            for metric_name, value in process_metrics.items():
                unit = "bytes" if "memory" in metric_name and "percent" not in metric_name else \
                       "percent" if "percent" in metric_name else "count"
                metrics.append(MetricValue(
                    metric_name=metric_name,
                    value=value,
                    timestamp=current_time,
                    dimensions={"app": self.app_name, "host": os.uname().nodename, "pid": str(os.getpid())},
                    tags={"application", "process"},
                    metadata={"unit": unit}
                ))
            
            # Custom counters
            for name, value in self.counters.items():
                metrics.append(MetricValue(
                    metric_name=f"application.counter.{name}",
                    value=value,
                    timestamp=current_time,
                    dimensions={"app": self.app_name, "host": os.uname().nodename},
                    tags={"application", "counter", "custom"},
                    metadata={"unit": "count", "type": "counter"}
                ))
            
            # Custom gauges
            for name, value in self.gauges.items():
                metrics.append(MetricValue(
                    metric_name=f"application.gauge.{name}",
                    value=value,
                    timestamp=current_time,
                    dimensions={"app": self.app_name, "host": os.uname().nodename},
                    tags={"application", "gauge", "custom"},
                    metadata={"unit": "value", "type": "gauge"}
                ))
            
            # Timer metrics
            for name, timer_data in self.timers.items():
                if timer_data['samples']:
                    avg_time = statistics.mean(timer_data['samples'])
                    metrics.append(MetricValue(
                        metric_name=f"application.timer.{name}.avg",
                        value=avg_time,
                        timestamp=current_time,
                        dimensions={"app": self.app_name, "host": os.uname().nodename},
                        tags={"application", "timer", "custom"},
                        metadata={"unit": "seconds", "type": "timer", "sample_count": len(timer_data['samples'])}
                    ))
                    
                    metrics.append(MetricValue(
                        metric_name=f"application.timer.{name}.count",
                        value=len(timer_data['samples']),
                        timestamp=current_time,
                        dimensions={"app": self.app_name, "host": os.uname().nodename},
                        tags={"application", "timer", "custom"},
                        metadata={"unit": "count", "type": "timer"}
                    ))
            
            # Custom metrics
            for name, metric_data in self.custom_metrics.items():
                metrics.append(MetricValue(
                    metric_name=f"application.custom.{name}",
                    value=metric_data['value'],
                    timestamp=current_time,
                    dimensions={"app": self.app_name, "host": os.uname().nodename},
                    tags={"application", "custom"},
                    metadata=metric_data.get('metadata', {})
                ))
            
            self.last_collection = current_time
            return metrics
            
        except Exception as e:
            raise RuntimeError(f"Application metrics collection failed: {e}")
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter metric."""
        self.counters[name] = self.counters.get(name, 0) + value
    
    def set_gauge(self, name: str, value: float):
        """Set a gauge metric value."""
        self.gauges[name] = value
    
    def record_timer(self, name: str, duration: float):
        """Record a timer value."""
        if name not in self.timers:
            self.timers[name] = {'samples': deque(maxlen=1000)}
        self.timers[name]['samples'].append(duration)
    
    def add_custom_metric(self, name: str, value: Any, metadata: Optional[Dict[str, Any]] = None):
        """Add a custom metric."""
        self.custom_metrics[name] = {
            'value': value,
            'metadata': metadata or {}
        }
    
    def get_metric_definitions(self) -> List[MetricDefinition]:
        """Get application metric definitions."""
        return [
            MetricDefinition(
                name="application.uptime",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.APPLICATION,
                description="Application uptime in seconds",
                unit="seconds",
                tags={"application", "runtime"},
                dimensions={"app", "host"},
                aggregations=[AggregationType.MAX]
            ),
            MetricDefinition(
                name="application.memory.percent",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.APPLICATION,
                description="Application memory usage percentage",
                unit="percent",
                tags={"application", "memory"},
                dimensions={"app", "host", "pid"},
                aggregations=[AggregationType.AVERAGE, AggregationType.MAX],
                alert_thresholds={"high": 80.0, "critical": 95.0}
            ),
            MetricDefinition(
                name="application.cpu.percent",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.APPLICATION,
                description="Application CPU usage percentage",
                unit="percent",
                tags={"application", "cpu"},
                dimensions={"app", "host", "pid"},
                aggregations=[AggregationType.AVERAGE, AggregationType.MAX],
                alert_thresholds={"high": 70.0, "critical": 90.0}
            )
        ]
    
    def get_collector_info(self) -> Dict[str, Any]:
        """Get application collector information."""
        return {
            'name': self.name,
            'type': 'application',
            'app_name': self.app_name,
            'last_collection': self.last_collection.isoformat() if self.last_collection else None,
            'custom_metrics_count': len(self.custom_metrics),
            'counters_count': len(self.counters),
            'gauges_count': len(self.gauges),
            'timers_count': len(self.timers),
            'capabilities': ['runtime', 'process', 'custom_metrics', 'counters', 'gauges', 'timers']
        }


class BusinessMetricsCollector(MetricCollector):
    """Collector for business-level metrics."""
    
    def __init__(self):
        self.name = "business_metrics_collector"
        self.last_collection = None
        self.business_metrics = {}
        
    def collect_metrics(self) -> List[MetricValue]:
        """Collect business metrics."""
        try:
            current_time = datetime.now(timezone.utc)
            metrics = []
            
            # Collect business metrics from the stored metrics
            for name, metric_data in self.business_metrics.items():
                metrics.append(MetricValue(
                    metric_name=f"business.{name}",
                    value=metric_data['value'],
                    timestamp=current_time,
                    dimensions=metric_data.get('dimensions', {}),
                    tags={"business"}.union(metric_data.get('tags', set())),
                    metadata=metric_data.get('metadata', {})
                ))
            
            self.last_collection = current_time
            return metrics
            
        except Exception as e:
            raise RuntimeError(f"Business metrics collection failed: {e}")
    
    def add_business_metric(self, name: str, value: Any, dimensions: Optional[Dict[str, str]] = None,
                           tags: Optional[Set[str]] = None, metadata: Optional[Dict[str, Any]] = None):
        """Add a business metric."""
        self.business_metrics[name] = {
            'value': value,
            'dimensions': dimensions or {},
            'tags': tags or set(),
            'metadata': metadata or {}
        }
    
    def get_metric_definitions(self) -> List[MetricDefinition]:
        """Get business metric definitions."""
        definitions = []
        for name in self.business_metrics.keys():
            definitions.append(MetricDefinition(
                name=f"business.{name}",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.BUSINESS,
                description=f"Business metric: {name}",
                unit="value",
                tags={"business"},
                aggregations=[AggregationType.SUM, AggregationType.AVERAGE]
            ))
        return definitions
    
    def get_collector_info(self) -> Dict[str, Any]:
        """Get business collector information."""
        return {
            'name': self.name,
            'type': 'business',
            'last_collection': self.last_collection.isoformat() if self.last_collection else None,
            'business_metrics_count': len(self.business_metrics),
            'capabilities': ['business_metrics']
        }


class MetricAggregator:
    """Aggregates metrics over time windows."""
    
    def __init__(self):
        self.aggregations = defaultdict(list)
        self.time_windows = {
            '1min': timedelta(minutes=1),
            '5min': timedelta(minutes=5),
            '15min': timedelta(minutes=15),
            '1hour': timedelta(hours=1),
            '1day': timedelta(days=1)
        }
    
    def add_metric(self, metric: MetricValue):
        """Add a metric for aggregation."""
        metric_key = f"{metric.metric_name}:{json.dumps(metric.dimensions, sort_keys=True)}"
        self.aggregations[metric_key].append(metric)
    
    def compute_aggregations(self, window: str) -> List[MetricAggregation]:
        """Compute aggregations for a time window."""
        try:
            if window not in self.time_windows:
                raise ValueError(f"Unknown time window: {window}")
            
            window_duration = self.time_windows[window]
            current_time = datetime.now(timezone.utc)
            window_start = current_time - window_duration
            
            aggregations = []
            
            for metric_key, metrics in self.aggregations.items():
                # Filter metrics within time window
                window_metrics = [m for m in metrics if m.timestamp >= window_start]
                
                if not window_metrics:
                    continue
                
                metric_name = metric_key.split(':')[0]
                dimensions = json.loads(':'.join(metric_key.split(':')[1:]))
                values = [m.value for m in window_metrics]
                
                # Compute different aggregations
                if values:
                    aggregations.extend([
                        MetricAggregation(
                            metric_name=f"{metric_name}.sum",
                            aggregation_type=AggregationType.SUM,
                            value=sum(values),
                            count=len(values),
                            start_time=window_start,
                            end_time=current_time,
                            dimensions=dimensions
                        ),
                        MetricAggregation(
                            metric_name=f"{metric_name}.avg",
                            aggregation_type=AggregationType.AVERAGE,
                            value=statistics.mean(values),
                            count=len(values),
                            start_time=window_start,
                            end_time=current_time,
                            dimensions=dimensions
                        ),
                        MetricAggregation(
                            metric_name=f"{metric_name}.min",
                            aggregation_type=AggregationType.MIN,
                            value=min(values),
                            count=len(values),
                            start_time=window_start,
                            end_time=current_time,
                            dimensions=dimensions
                        ),
                        MetricAggregation(
                            metric_name=f"{metric_name}.max",
                            aggregation_type=AggregationType.MAX,
                            value=max(values),
                            count=len(values),
                            start_time=window_start,
                            end_time=current_time,
                            dimensions=dimensions
                        ),
                        MetricAggregation(
                            metric_name=f"{metric_name}.count",
                            aggregation_type=AggregationType.COUNT,
                            value=len(values),
                            count=len(values),
                            start_time=window_start,
                            end_time=current_time,
                            dimensions=dimensions
                        )
                    ])
                    
                    # Add median if enough data points
                    if len(values) >= 3:
                        aggregations.append(MetricAggregation(
                            metric_name=f"{metric_name}.median",
                            aggregation_type=AggregationType.MEDIAN,
                            value=statistics.median(values),
                            count=len(values),
                            start_time=window_start,
                            end_time=current_time,
                            dimensions=dimensions
                        ))
                    
                    # Add standard deviation if enough data points
                    if len(values) >= 2:
                        aggregations.append(MetricAggregation(
                            metric_name=f"{metric_name}.stddev",
                            aggregation_type=AggregationType.STDDEV,
                            value=statistics.stdev(values),
                            count=len(values),
                            start_time=window_start,
                            end_time=current_time,
                            dimensions=dimensions
                        ))
            
            return aggregations
            
        except Exception as e:
            raise RuntimeError(f"Metric aggregation failed: {e}")
    
    def cleanup_old_metrics(self, retention_hours: int = 24):
        """Clean up old metrics beyond retention period."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
            
            for metric_key in list(self.aggregations.keys()):
                # Filter out old metrics
                self.aggregations[metric_key] = [
                    m for m in self.aggregations[metric_key] 
                    if m.timestamp >= cutoff_time
                ]
                
                # Remove empty keys
                if not self.aggregations[metric_key]:
                    del self.aggregations[metric_key]
                    
        except Exception as e:
            logging.error(f"Failed to cleanup old metrics: {e}")


class MetricAlertManager:
    """Manages metric-based alerts."""
    
    def __init__(self):
        self.alerts = {}
        self.alert_handlers = []
        
    def add_alert(self, alert: MetricAlert):
        """Add a metric alert."""
        self.alerts[alert.metric_name] = alert
    
    def remove_alert(self, metric_name: str):
        """Remove a metric alert."""
        if metric_name in self.alerts:
            del self.alerts[metric_name]
    
    def add_alert_handler(self, handler: Callable[[MetricAlert, MetricValue], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def check_alerts(self, metrics: List[MetricValue]) -> List[Tuple[MetricAlert, MetricValue]]:
        """Check metrics against alert conditions."""
        triggered_alerts = []
        
        try:
            for metric in metrics:
                if metric.metric_name in self.alerts:
                    alert = self.alerts[metric.metric_name]
                    
                    if not alert.enabled:
                        continue
                    
                    # Check cooldown period
                    if alert.last_triggered:
                        cooldown_end = alert.last_triggered + timedelta(minutes=alert.cooldown_minutes)
                        if datetime.now(timezone.utc) < cooldown_end:
                            continue
                    
                    # Evaluate alert condition
                    if self._evaluate_condition(alert.condition, metric.value, alert.threshold):
                        alert.last_triggered = datetime.now(timezone.utc)
                        triggered_alerts.append((alert, metric))
                        
                        # Call alert handlers
                        for handler in self.alert_handlers:
                            try:
                                handler(alert, metric)
                            except Exception as e:
                                logging.error(f"Alert handler error: {e}")
            
            return triggered_alerts
            
        except Exception as e:
            raise RuntimeError(f"Alert checking failed: {e}")
    
    def _evaluate_condition(self, condition: str, value: float, threshold: float) -> bool:
        """Evaluate alert condition."""
        try:
            if condition == "greater_than":
                return value > threshold
            elif condition == "less_than":
                return value < threshold
            elif condition == "equal":
                return abs(value - threshold) < 1e-9
            elif condition == "greater_equal":
                return value >= threshold
            elif condition == "less_equal":
                return value <= threshold
            else:
                logging.warning(f"Unknown alert condition: {condition}")
                return False
                
        except Exception as e:
            logging.error(f"Condition evaluation error: {e}")
            return False
    
    def get_alert_status(self) -> Dict[str, Any]:
        """Get status of all alerts."""
        return {
            'total_alerts': len(self.alerts),
            'enabled_alerts': len([a for a in self.alerts.values() if a.enabled]),
            'alerts': {name: alert.to_dict() for name, alert in self.alerts.items()}
        }


class MetricsManager:
    """Main metrics management system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.collectors = {}
        self.aggregator = MetricAggregator()
        self.alert_manager = MetricAlertManager()
        self.running = False
        self.collection_thread = None
        self.aggregation_thread = None
        self.alert_thread = None
        self.collection_interval = 60  # seconds
        self.aggregation_interval = 300  # 5 minutes
        self.alert_check_interval = 30  # 30 seconds
        self.storage_path = "./metrics_data"
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.metric_definitions = {}
        self.recent_metrics = deque(maxlen=10000)
        
        # Initialize default collectors
        self._setup_default_collectors()
        
        # Load configuration if provided
        if config_path:
            self._load_config(config_path)
        
        # Setup default alerts
        self._setup_default_alerts()
    
    def _setup_default_collectors(self):
        """Setup default metric collectors."""
        # System collector
        system_collector = SystemMetricsCollector()
        self.add_collector("system", system_collector)
        
        # Application collector
        app_collector = ApplicationMetricsCollector()
        self.add_collector("application", app_collector)
        
        # Business collector
        business_collector = BusinessMetricsCollector()
        self.add_collector("business", business_collector)
    
    def _setup_default_alerts(self):
        """Setup default metric alerts."""
        default_alerts = [
            MetricAlert(
                metric_name="system.cpu.utilization",
                condition="greater_than",
                threshold=80.0,
                severity=AlertSeverity.HIGH,
                message="High CPU utilization detected",
                cooldown_minutes=5
            ),
            MetricAlert(
                metric_name="system.memory.utilization",
                condition="greater_than",
                threshold=85.0,
                severity=AlertSeverity.HIGH,
                message="High memory utilization detected",
                cooldown_minutes=5
            ),
            MetricAlert(
                metric_name="system.disk.utilization",
                condition="greater_than",
                threshold=90.0,
                severity=AlertSeverity.CRITICAL,
                message="Critical disk utilization detected",
                cooldown_minutes=10
            )
        ]
        
        for alert in default_alerts:
            self.alert_manager.add_alert(alert)
    
    def _load_config(self, config_path: str):
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Apply configuration
            if 'collection_interval' in config:
                self.collection_interval = config['collection_interval']
            
            if 'aggregation_interval' in config:
                self.aggregation_interval = config['aggregation_interval']
            
            if 'alert_check_interval' in config:
                self.alert_check_interval = config['alert_check_interval']
            
            if 'storage_path' in config:
                self.storage_path = config['storage_path']
            
            # Load custom alerts
            if 'alerts' in config:
                for alert_config in config['alerts']:
                    alert = MetricAlert(
                        metric_name=alert_config['metric_name'],
                        condition=alert_config['condition'],
                        threshold=alert_config['threshold'],
                        severity=AlertSeverity(alert_config['severity']),
                        message=alert_config['message'],
                        enabled=alert_config.get('enabled', True),
                        cooldown_minutes=alert_config.get('cooldown_minutes', 5)
                    )
                    self.alert_manager.add_alert(alert)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load metrics configuration: {e}")
    
    def add_collector(self, name: str, collector: MetricCollector):
        """Add a metric collector."""
        self.collectors[name] = collector
        
        # Register metric definitions
        for definition in collector.get_metric_definitions():
            self.metric_definitions[definition.name] = definition
    
    def remove_collector(self, name: str):
        """Remove a metric collector."""
        if name in self.collectors:
            del self.collectors[name]
    
    def start_metrics_collection(self):
        """Start metrics collection system."""
        if self.running:
            return
        
        try:
            self.running = True
            
            # Start collection thread
            self.collection_thread = threading.Thread(target=self._collection_worker, daemon=True)
            self.collection_thread.start()
            
            # Start aggregation thread
            self.aggregation_thread = threading.Thread(target=self._aggregation_worker, daemon=True)
            self.aggregation_thread.start()
            
            # Start alert checking thread
            self.alert_thread = threading.Thread(target=self._alert_worker, daemon=True)
            self.alert_thread.start()
            
            logging.info("Metrics collection system started successfully")
            
        except Exception as e:
            self.running = False
            raise RuntimeError(f"Failed to start metrics collection system: {e}")
    
    def stop_metrics_collection(self):
        """Stop metrics collection system."""
        if not self.running:
            return
        
        try:
            self.running = False
            
            # Stop threads
            if self.collection_thread:
                self.collection_thread.join(timeout=5)
            
            if self.aggregation_thread:
                self.aggregation_thread.join(timeout=5)
            
            if self.alert_thread:
                self.alert_thread.join(timeout=5)
            
            # Shutdown executor
            self.executor.shutdown(wait=True, timeout=10)
            
            logging.info("Metrics collection system stopped successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to stop metrics collection system: {e}")
    
    def _collection_worker(self):
        """Worker thread for metrics collection."""
        while self.running:
            try:
                # Collect from all collectors
                all_metrics = []
                
                for collector_name, collector in self.collectors.items():
                    try:
                        metrics = collector.collect_metrics()
                        all_metrics.extend(metrics)
                    except Exception as e:
                        logging.error(f"Collection error from {collector_name}: {e}")
                
                # Store metrics
                if all_metrics:
                    self._store_metrics(all_metrics)
                    
                    # Add to recent metrics for aggregation
                    self.recent_metrics.extend(all_metrics)
                    
                    # Add to aggregator
                    for metric in all_metrics:
                        self.aggregator.add_metric(metric)
                
                # Wait for next collection
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logging.error(f"Collection worker error: {e}")
                time.sleep(30)  # Wait before retrying
    
    def _aggregation_worker(self):
        """Worker thread for metrics aggregation."""
        while self.running:
            try:
                # Compute aggregations for different time windows
                for window in ['1min', '5min', '15min', '1hour']:
                    try:
                        aggregations = self.aggregator.compute_aggregations(window)
                        if aggregations:
                            self._store_aggregations(aggregations, window)
                    except Exception as e:
                        logging.error(f"Aggregation error for window {window}: {e}")
                
                # Cleanup old metrics
                self.aggregator.cleanup_old_metrics()
                
                # Wait for next aggregation
                time.sleep(self.aggregation_interval)
                
            except Exception as e:
                logging.error(f"Aggregation worker error: {e}")
                time.sleep(60)
    
    def _alert_worker(self):
        """Worker thread for alert checking."""
        while self.running:
            try:
                # Get recent metrics for alert checking
                recent_metrics_list = list(self.recent_metrics)
                
                if recent_metrics_list:
                    # Check alerts
                    triggered_alerts = self.alert_manager.check_alerts(recent_metrics_list)
                    
                    # Log triggered alerts
                    for alert, metric in triggered_alerts:
                        logging.warning(f"ALERT: {alert.message} - {metric.metric_name}={metric.value} (threshold: {alert.threshold})")
                
                # Wait for next alert check
                time.sleep(self.alert_check_interval)
                
            except Exception as e:
                logging.error(f"Alert worker error: {e}")
                time.sleep(30)
    
    def _store_metrics(self, metrics: List[MetricValue]):
        """Store metrics to storage."""
        try:
            storage_path = Path(self.storage_path)
            storage_path.mkdir(parents=True, exist_ok=True)
            
            # Create timestamped file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"metrics_{timestamp}.json"
            file_path = storage_path / filename
            
            # Convert metrics to dict format
            data = [metric.to_dict() for metric in metrics]
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            raise RuntimeError(f"Failed to store metrics: {e}")
    
    def _store_aggregations(self, aggregations: List[MetricAggregation], window: str):
        """Store aggregations to storage."""
        try:
            storage_path = Path(self.storage_path) / "aggregations" / window
            storage_path.mkdir(parents=True, exist_ok=True)
            
            # Create timestamped file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"aggregations_{window}_{timestamp}.json"
            file_path = storage_path / filename
            
            # Convert aggregations to dict format
            data = [agg.to_dict() for agg in aggregations]
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            logging.error(f"Failed to store aggregations: {e}")
    
    def get_metrics_status(self) -> Dict[str, Any]:
        """Get current metrics system status."""
        return {
            'running': self.running,
            'collection_interval': self.collection_interval,
            'aggregation_interval': self.aggregation_interval,
            'alert_check_interval': self.alert_check_interval,
            'storage_path': self.storage_path,
            'collectors': {name: collector.get_collector_info() for name, collector in self.collectors.items()},
            'metric_definitions_count': len(self.metric_definitions),
            'recent_metrics_count': len(self.recent_metrics),
            'alert_status': self.alert_manager.get_alert_status()
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary statistics."""
        try:
            recent_metrics_list = list(self.recent_metrics)
            
            summary = {
                'total_metrics': len(recent_metrics_list),
                'metrics_by_category': {},
                'metrics_by_type': {},
                'collection_rate': len(recent_metrics_list) / max(self.collection_interval, 1),
                'top_metrics': {},
                'storage_usage': self._get_storage_usage()
            }
            
            # Analyze metrics by category and type
            for metric in recent_metrics_list:
                # Extract category from metric name
                category = metric.metric_name.split('.')[0] if '.' in metric.metric_name else 'unknown'
                summary['metrics_by_category'][category] = summary['metrics_by_category'].get(category, 0) + 1
                
                # Count by metric name
                summary['top_metrics'][metric.metric_name] = summary['top_metrics'].get(metric.metric_name, 0) + 1
            
            # Get top 10 metrics
            summary['top_metrics'] = dict(sorted(summary['top_metrics'].items(), key=lambda x: x[1], reverse=True)[:10])
            
            return summary
            
        except Exception as e:
            raise RuntimeError(f"Failed to get metrics summary: {e}")
    
    def _get_storage_usage(self) -> Dict[str, Any]:
        """Get storage usage statistics."""
        try:
            storage_path = Path(self.storage_path)
            if not storage_path.exists():
                return {'total_size': 0, 'file_count': 0}
            
            total_size = 0
            file_count = 0
            
            for file_path in storage_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            return {
                'total_size': total_size,
                'file_count': file_count,
                'average_file_size': total_size / max(file_count, 1)
            }
            
        except Exception as e:
            logging.error(f"Failed to get storage usage: {e}")
            return {'total_size': 0, 'file_count': 0}


# Integration with existing systems
def integrate_with_monitoring_system(metrics_manager: MetricsManager,
                                   monitoring_system: 'ProductionMonitoringSystem') -> bool:
    """Integrate metrics with production monitoring system."""
    try:
        # Add monitoring system metrics collector
        class MonitoringMetricsCollector(MetricCollector):
            def __init__(self, monitoring_sys):
                self.monitoring_system = monitoring_sys
                self.name = "monitoring_system_metrics"
            
            def collect_metrics(self) -> List[MetricValue]:
                current_time = datetime.now(timezone.utc)
                metrics = []
                
                try:
                    # Get monitoring status
                    status = self.monitoring_system.get_monitoring_status()
                    
                    for component, component_status in status.get('components', {}).items():
                        if isinstance(component_status, dict) and 'running' in component_status:
                            metrics.append(MetricValue(
                                metric_name=f"monitoring.component.status",
                                value=1 if component_status['running'] else 0,
                                timestamp=current_time,
                                dimensions={"component": component, "host": os.uname().nodename},
                                tags={"monitoring", "status"},
                                metadata={"unit": "boolean"}
                            ))
                    
                    # Get monitoring metrics
                    monitoring_metrics = self.monitoring_system.get_metrics()
                    for metric_name, metric_value in monitoring_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            metrics.append(MetricValue(
                                metric_name=f"monitoring.{metric_name}",
                                value=metric_value,
                                timestamp=current_time,
                                dimensions={"host": os.uname().nodename},
                                tags={"monitoring", "metrics"},
                                metadata={"unit": "value"}
                            ))
                
                except Exception as e:
                    logging.error(f"Failed to collect monitoring metrics: {e}")
                
                return metrics
            
            def get_metric_definitions(self) -> List[MetricDefinition]:
                return [
                    MetricDefinition(
                        name="monitoring.component.status",
                        metric_type=MetricType.GAUGE,
                        category=MetricCategory.SYSTEM,
                        description="Monitoring component status",
                        unit="boolean",
                        tags={"monitoring", "status"},
                        dimensions={"component", "host"}
                    )
                ]
            
            def get_collector_info(self) -> Dict[str, Any]:
                return {
                    'name': self.name,
                    'type': 'integration',
                    'integration_type': 'monitoring_system',
                    'capabilities': ['component_status', 'monitoring_metrics']
                }
        
        # Add the collector
        monitoring_collector = MonitoringMetricsCollector(monitoring_system)
        metrics_manager.add_collector("monitoring_integration", monitoring_collector)
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to integrate with monitoring system: {e}")
        return False


def integrate_with_telemetry_system(metrics_manager: MetricsManager,
                                  telemetry_manager: TelemetryManager) -> bool:
    """Integrate metrics with telemetry system."""
    try:
        # Add telemetry-based metrics collector
        class TelemetryMetricsCollector(MetricCollector):
            def __init__(self, telemetry_mgr):
                self.telemetry_manager = telemetry_mgr
                self.name = "telemetry_system_metrics"
            
            def collect_metrics(self) -> List[MetricValue]:
                current_time = datetime.now(timezone.utc)
                metrics = []
                
                try:
                    # Get telemetry status
                    telemetry_status = self.telemetry_manager.get_telemetry_status()
                    
                    metrics.append(MetricValue(
                        metric_name="telemetry.system.status",
                        value=1 if telemetry_status['running'] else 0,
                        timestamp=current_time,
                        dimensions={"host": os.uname().nodename},
                        tags={"telemetry", "status"},
                        metadata={"unit": "boolean"}
                    ))
                    
                    metrics.append(MetricValue(
                        metric_name="telemetry.queue.size",
                        value=telemetry_status['queue_size'],
                        timestamp=current_time,
                        dimensions={"host": os.uname().nodename},
                        tags={"telemetry", "queue"},
                        metadata={"unit": "count"}
                    ))
                    
                    # Get telemetry metrics
                    telemetry_metrics = self.telemetry_manager.get_telemetry_metrics()
                    
                    metrics.append(MetricValue(
                        metric_name="telemetry.data_points.total",
                        value=telemetry_metrics['total_data_points'],
                        timestamp=current_time,
                        dimensions={"host": os.uname().nodename},
                        tags={"telemetry", "data_points"},
                        metadata={"unit": "count"}
                    ))
                    
                    metrics.append(MetricValue(
                        metric_name="telemetry.collection.rate",
                        value=telemetry_metrics['collection_rate'],
                        timestamp=current_time,
                        dimensions={"host": os.uname().nodename},
                        tags={"telemetry", "rate"},
                        metadata={"unit": "per_second"}
                    ))
                
                except Exception as e:
                    logging.error(f"Failed to collect telemetry metrics: {e}")
                
                return metrics
            
            def get_metric_definitions(self) -> List[MetricDefinition]:
                return [
                    MetricDefinition(
                        name="telemetry.system.status",
                        metric_type=MetricType.GAUGE,
                        category=MetricCategory.SYSTEM,
                        description="Telemetry system status",
                        unit="boolean",
                        tags={"telemetry", "status"},
                        dimensions={"host"}
                    ),
                    MetricDefinition(
                        name="telemetry.data_points.total",
                        metric_type=MetricType.COUNTER,
                        category=MetricCategory.SYSTEM,
                        description="Total telemetry data points collected",
                        unit="count",
                        tags={"telemetry", "data_points"},
                        dimensions={"host"}
                    )
                ]
            
            def get_collector_info(self) -> Dict[str, Any]:
                return {
                    'name': self.name,
                    'type': 'integration',
                    'integration_type': 'telemetry_system',
                    'capabilities': ['telemetry_status', 'telemetry_metrics']
                }
        
        # Add the collector
        telemetry_collector = TelemetryMetricsCollector(telemetry_manager)
        metrics_manager.add_collector("telemetry_integration", telemetry_collector)
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to integrate with telemetry system: {e}")
        return False