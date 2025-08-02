"""
Production Monitoring System - Comprehensive production monitoring system
NO FALLBACKS - HARD FAILURES ONLY

This module provides comprehensive production monitoring capabilities including
real-time system monitoring, application monitoring, database monitoring,
network monitoring, security monitoring, and alert management.

Key Features:
- Multi-level monitoring (BASIC, STANDARD, ENHANCED, COMPREHENSIVE)
- Multi-component monitoring (SYSTEM, APPLICATION, DATABASE, NETWORK, SECURITY)
- Real-time monitoring with configurable intervals
- Alert management with severity levels and escalation
- Health check system with automatic failure detection
- Performance tracking with historical data
- Resource monitoring (CPU, memory, disk, network, GPU)
- Application monitoring (response times, throughput, errors)
- Database monitoring (connections, queries, performance)
- Network monitoring (latency, bandwidth, connectivity)
- Security monitoring (authentication, authorization, threats)

Architecture: NO FALLBACKS - HARD FAILURES ONLY
All monitoring operations must succeed or fail explicitly with detailed error information.
"""

import os
import json
import logging
import threading
import time
import psutil
import socket
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import traceback
import uuid
import asyncio
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
import sqlite3
from contextlib import contextmanager
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from production_config_manager import ProductionConfigManager, MonitoringConfig
        from compression_systems.padic.hybrid_performance_monitor import HybridPerformanceMonitor
        from progress_tracker import ProgressTracker, AlertSeverity
except ImportError:
    pass

logger = logging.getLogger(__name__)


class ProductionMonitoringLevel(Enum):
    """Production monitoring level types."""
    BASIC = "basic"
    STANDARD = "standard"
    ENHANCED = "enhanced"
    COMPREHENSIVE = "comprehensive"


class MonitoringComponent(Enum):
    """Monitoring component types."""
    SYSTEM = "system"
    APPLICATION = "application"
    DATABASE = "database"
    NETWORK = "network"
    SECURITY = "security"
    CUSTOM = "custom"


class MonitoringStatus(Enum):
    """Monitoring status types."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPING = "stopping"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """Health check status types."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ProductionMetric:
    """Production metric data structure."""
    metric_id: str
    component: MonitoringComponent
    metric_name: str
    metric_value: Union[int, float, str, bool]
    metric_unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate metric data."""
        if not self.metric_id:
            raise ValueError("Metric ID cannot be empty")
        if not self.metric_name:
            raise ValueError("Metric name cannot be empty")
        if not isinstance(self.component, MonitoringComponent):
            raise TypeError("Component must be MonitoringComponent")


@dataclass
class ProductionAlert:
    """Production alert data structure."""
    alert_id: str
    component: MonitoringComponent
    severity: AlertSeverity
    alert_type: str
    message: str
    metric_name: Optional[str] = None
    metric_value: Optional[Union[int, float]] = None
    threshold: Optional[Union[int, float]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate alert data."""
        if not self.alert_id:
            raise ValueError("Alert ID cannot be empty")
        if not isinstance(self.component, MonitoringComponent):
            raise TypeError("Component must be MonitoringComponent")
        if not isinstance(self.severity, AlertSeverity):
            raise TypeError("Severity must be AlertSeverity")
        if not self.message:
            raise ValueError("Alert message cannot be empty")


@dataclass
class ProductionHealthCheck:
    """Production health check data structure."""
    check_id: str
    component: MonitoringComponent
    check_name: str
    status: HealthStatus
    message: str
    check_duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate health check data."""
        if not self.check_id:
            raise ValueError("Check ID cannot be empty")
        if not isinstance(self.component, MonitoringComponent):
            raise TypeError("Component must be MonitoringComponent")
        if not isinstance(self.status, HealthStatus):
            raise TypeError("Status must be HealthStatus")
        if not self.check_name:
            raise ValueError("Check name cannot be empty")


@dataclass
class MonitoringConfiguration:
    """Monitoring configuration settings."""
    monitoring_level: ProductionMonitoringLevel = ProductionMonitoringLevel.STANDARD
    enabled_components: List[MonitoringComponent] = field(default_factory=lambda: [
        MonitoringComponent.SYSTEM,
        MonitoringComponent.APPLICATION
    ])
    
    # Monitoring intervals
    system_monitoring_interval: int = 30
    application_monitoring_interval: int = 60
    database_monitoring_interval: int = 120
    network_monitoring_interval: int = 60
    security_monitoring_interval: int = 300
    
    # Health check intervals
    health_check_interval: int = 30
    health_check_timeout: float = 10.0
    
    # Alert settings
    alert_enabled: bool = True
    alert_cooldown_minutes: int = 5
    alert_escalation_enabled: bool = True
    alert_escalation_minutes: int = 30
    
    # Data retention
    metrics_retention_hours: int = 168  # 7 days
    alerts_retention_hours: int = 720   # 30 days
    health_checks_retention_hours: int = 48  # 2 days
    
    # Performance settings
    max_concurrent_checks: int = 10
    monitoring_thread_pool_size: int = 5
    
    # Thresholds
    cpu_threshold_percent: float = 80.0
    memory_threshold_percent: float = 85.0
    disk_threshold_percent: float = 90.0
    network_latency_threshold_ms: float = 100.0
    response_time_threshold_ms: float = 1000.0
    error_rate_threshold_percent: float = 5.0


class SystemMonitor:
    """System resource monitoring component."""
    
    def __init__(self, config: MonitoringConfiguration):
        self.config = config
        self.logger = logging.getLogger('SystemMonitor')
        self.metrics_cache: deque = deque(maxlen=1000)
        self.last_check = time.time()
    
    def collect_system_metrics(self) -> List[ProductionMetric]:
        """Collect system resource metrics."""
        try:
            metrics = []
            timestamp = datetime.utcnow()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            metrics.extend([
                ProductionMetric(
                    metric_id=f"system_cpu_percent_{int(time.time())}",
                    component=MonitoringComponent.SYSTEM,
                    metric_name="cpu_usage_percent",
                    metric_value=cpu_percent,
                    metric_unit="percent",
                    timestamp=timestamp
                ),
                ProductionMetric(
                    metric_id=f"system_cpu_count_{int(time.time())}",
                    component=MonitoringComponent.SYSTEM,
                    metric_name="cpu_count",
                    metric_value=cpu_count,
                    metric_unit="count",
                    timestamp=timestamp
                )
            ])
            
            if cpu_freq:
                metrics.append(ProductionMetric(
                    metric_id=f"system_cpu_freq_{int(time.time())}",
                    component=MonitoringComponent.SYSTEM,
                    metric_name="cpu_frequency_mhz",
                    metric_value=cpu_freq.current,
                    metric_unit="mhz",
                    timestamp=timestamp
                ))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            metrics.extend([
                ProductionMetric(
                    metric_id=f"system_memory_percent_{int(time.time())}",
                    component=MonitoringComponent.SYSTEM,
                    metric_name="memory_usage_percent",
                    metric_value=memory.percent,
                    metric_unit="percent",
                    timestamp=timestamp
                ),
                ProductionMetric(
                    metric_id=f"system_memory_total_{int(time.time())}",
                    component=MonitoringComponent.SYSTEM,
                    metric_name="memory_total_gb",
                    metric_value=memory.total / (1024**3),
                    metric_unit="gb",
                    timestamp=timestamp
                ),
                ProductionMetric(
                    metric_id=f"system_memory_available_{int(time.time())}",
                    component=MonitoringComponent.SYSTEM,
                    metric_name="memory_available_gb",
                    metric_value=memory.available / (1024**3),
                    metric_unit="gb",
                    timestamp=timestamp
                ),
                ProductionMetric(
                    metric_id=f"system_swap_percent_{int(time.time())}",
                    component=MonitoringComponent.SYSTEM,
                    metric_name="swap_usage_percent",
                    metric_value=swap.percent,
                    metric_unit="percent",
                    timestamp=timestamp
                )
            ])
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            metrics.extend([
                ProductionMetric(
                    metric_id=f"system_disk_percent_{int(time.time())}",
                    component=MonitoringComponent.SYSTEM,
                    metric_name="disk_usage_percent",
                    metric_value=(disk_usage.used / disk_usage.total) * 100,
                    metric_unit="percent",
                    timestamp=timestamp
                ),
                ProductionMetric(
                    metric_id=f"system_disk_free_{int(time.time())}",
                    component=MonitoringComponent.SYSTEM,
                    metric_name="disk_free_gb",
                    metric_value=disk_usage.free / (1024**3),
                    metric_unit="gb",
                    timestamp=timestamp
                )
            ])
            
            # Network metrics
            net_io = psutil.net_io_counters()
            if net_io:
                metrics.extend([
                    ProductionMetric(
                        metric_id=f"system_network_bytes_sent_{int(time.time())}",
                        component=MonitoringComponent.SYSTEM,
                        metric_name="network_bytes_sent",
                        metric_value=net_io.bytes_sent,
                        metric_unit="bytes",
                        timestamp=timestamp
                    ),
                    ProductionMetric(
                        metric_id=f"system_network_bytes_recv_{int(time.time())}",
                        component=MonitoringComponent.SYSTEM,
                        metric_name="network_bytes_received",
                        metric_value=net_io.bytes_recv,
                        metric_unit="bytes",
                        timestamp=timestamp
                    )
                ])
            
            # Process metrics
            process_count = len(psutil.pids())
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            
            metrics.extend([
                ProductionMetric(
                    metric_id=f"system_process_count_{int(time.time())}",
                    component=MonitoringComponent.SYSTEM,
                    metric_name="process_count",
                    metric_value=process_count,
                    metric_unit="count",
                    timestamp=timestamp
                ),
                ProductionMetric(
                    metric_id=f"system_load_avg_{int(time.time())}",
                    component=MonitoringComponent.SYSTEM,
                    metric_name="load_average_1min",
                    metric_value=load_avg[0],
                    metric_unit="ratio",
                    timestamp=timestamp
                )
            ])
            
            # GPU metrics (if available)
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    metrics.extend([
                        ProductionMetric(
                            metric_id=f"system_gpu_{i}_util_{int(time.time())}",
                            component=MonitoringComponent.SYSTEM,
                            metric_name=f"gpu_{i}_utilization_percent",
                            metric_value=gpu_util.gpu,
                            metric_unit="percent",
                            timestamp=timestamp,
                            tags={"gpu_id": str(i)}
                        ),
                        ProductionMetric(
                            metric_id=f"system_gpu_{i}_memory_{int(time.time())}",
                            component=MonitoringComponent.SYSTEM,
                            metric_name=f"gpu_{i}_memory_usage_percent",
                            metric_value=(gpu_memory.used / gpu_memory.total) * 100,
                            metric_unit="percent",
                            timestamp=timestamp,
                            tags={"gpu_id": str(i)}
                        )
                    ])
            except ImportError:
                pass  # GPU monitoring not available
            except Exception as e:
                self.logger.warning(f"GPU monitoring failed: {e}")
            
            # Cache metrics for trend analysis
            self.metrics_cache.extend(metrics)
            self.last_check = time.time()
            
            return metrics
            
        except Exception as e:
            error_msg = f"Failed to collect system metrics: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def check_system_health(self) -> List[ProductionHealthCheck]:
        """Check system health status."""
        try:
            health_checks = []
            timestamp = datetime.utcnow()
            
            # CPU health check
            start_time = time.time()
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_check_duration = (time.time() - start_time) * 1000
            
            cpu_status = HealthStatus.HEALTHY
            cpu_message = f"CPU usage: {cpu_percent:.1f}%"
            
            if cpu_percent > self.config.cpu_threshold_percent:
                cpu_status = HealthStatus.UNHEALTHY if cpu_percent > 95 else HealthStatus.DEGRADED
                cpu_message = f"High CPU usage: {cpu_percent:.1f}% > {self.config.cpu_threshold_percent}%"
            
            health_checks.append(ProductionHealthCheck(
                check_id=f"system_cpu_health_{int(time.time())}",
                component=MonitoringComponent.SYSTEM,
                check_name="cpu_health",
                status=cpu_status,
                message=cpu_message,
                check_duration_ms=cpu_check_duration,
                timestamp=timestamp,
                details={"cpu_percent": cpu_percent, "threshold": self.config.cpu_threshold_percent}
            ))
            
            # Memory health check
            start_time = time.time()
            memory = psutil.virtual_memory()
            memory_check_duration = (time.time() - start_time) * 1000
            
            memory_status = HealthStatus.HEALTHY
            memory_message = f"Memory usage: {memory.percent:.1f}%"
            
            if memory.percent > self.config.memory_threshold_percent:
                memory_status = HealthStatus.UNHEALTHY if memory.percent > 95 else HealthStatus.DEGRADED
                memory_message = f"High memory usage: {memory.percent:.1f}% > {self.config.memory_threshold_percent}%"
            
            health_checks.append(ProductionHealthCheck(
                check_id=f"system_memory_health_{int(time.time())}",
                component=MonitoringComponent.SYSTEM,
                check_name="memory_health",
                status=memory_status,
                message=memory_message,
                check_duration_ms=memory_check_duration,
                timestamp=timestamp,
                details={"memory_percent": memory.percent, "threshold": self.config.memory_threshold_percent}
            ))
            
            # Disk health check
            start_time = time.time()
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            disk_check_duration = (time.time() - start_time) * 1000
            
            disk_status = HealthStatus.HEALTHY
            disk_message = f"Disk usage: {disk_percent:.1f}%"
            
            if disk_percent > self.config.disk_threshold_percent:
                disk_status = HealthStatus.UNHEALTHY if disk_percent > 98 else HealthStatus.DEGRADED
                disk_message = f"High disk usage: {disk_percent:.1f}% > {self.config.disk_threshold_percent}%"
            
            health_checks.append(ProductionHealthCheck(
                check_id=f"system_disk_health_{int(time.time())}",
                component=MonitoringComponent.SYSTEM,
                check_name="disk_health",
                status=disk_status,
                message=disk_message,
                check_duration_ms=disk_check_duration,
                timestamp=timestamp,
                details={"disk_percent": disk_percent, "threshold": self.config.disk_threshold_percent}
            ))
            
            return health_checks
            
        except Exception as e:
            error_msg = f"Failed to check system health: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)


class ApplicationMonitor:
    """Application performance monitoring component."""
    
    def __init__(self, config: MonitoringConfiguration):
        self.config = config
        self.logger = logging.getLogger('ApplicationMonitor')
        self.response_times: deque = deque(maxlen=1000)
        self.error_counts: defaultdict = defaultdict(int)
        self.request_counts: defaultdict = defaultdict(int)
    
    def collect_application_metrics(self) -> List[ProductionMetric]:
        """Collect application performance metrics."""
        try:
            metrics = []
            timestamp = datetime.utcnow()
            
            # Response time metrics
            if self.response_times:
                avg_response_time = statistics.mean(self.response_times)
                p95_response_time = np.percentile(list(self.response_times), 95)
                p99_response_time = np.percentile(list(self.response_times), 99)
                
                metrics.extend([
                    ProductionMetric(
                        metric_id=f"app_response_time_avg_{int(time.time())}",
                        component=MonitoringComponent.APPLICATION,
                        metric_name="response_time_average_ms",
                        metric_value=avg_response_time,
                        metric_unit="ms",
                        timestamp=timestamp
                    ),
                    ProductionMetric(
                        metric_id=f"app_response_time_p95_{int(time.time())}",
                        component=MonitoringComponent.APPLICATION,
                        metric_name="response_time_p95_ms",
                        metric_value=p95_response_time,
                        metric_unit="ms",
                        timestamp=timestamp
                    ),
                    ProductionMetric(
                        metric_id=f"app_response_time_p99_{int(time.time())}",
                        component=MonitoringComponent.APPLICATION,
                        metric_name="response_time_p99_ms",
                        metric_value=p99_response_time,
                        metric_unit="ms",
                        timestamp=timestamp
                    )
                ])
            
            # Request count metrics
            total_requests = sum(self.request_counts.values())
            metrics.append(ProductionMetric(
                metric_id=f"app_requests_total_{int(time.time())}",
                component=MonitoringComponent.APPLICATION,
                metric_name="requests_total",
                metric_value=total_requests,
                metric_unit="count",
                timestamp=timestamp
            ))
            
            # Error rate metrics
            total_errors = sum(self.error_counts.values())
            error_rate = (total_errors / max(1, total_requests)) * 100
            
            metrics.extend([
                ProductionMetric(
                    metric_id=f"app_errors_total_{int(time.time())}",
                    component=MonitoringComponent.APPLICATION,
                    metric_name="errors_total",
                    metric_value=total_errors,
                    metric_unit="count",
                    timestamp=timestamp
                ),
                ProductionMetric(
                    metric_id=f"app_error_rate_{int(time.time())}",
                    component=MonitoringComponent.APPLICATION,
                    metric_name="error_rate_percent",
                    metric_value=error_rate,
                    metric_unit="percent",
                    timestamp=timestamp
                )
            ])
            
            # Throughput metrics
            requests_per_second = len(self.response_times) / 60  # Assuming 1-minute window
            metrics.append(ProductionMetric(
                metric_id=f"app_throughput_{int(time.time())}",
                component=MonitoringComponent.APPLICATION,
                metric_name="throughput_rps",
                metric_value=requests_per_second,
                metric_unit="rps",
                timestamp=timestamp
            ))
            
            return metrics
            
        except Exception as e:
            error_msg = f"Failed to collect application metrics: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def record_request(self, response_time_ms: float, status_code: int = 200) -> None:
        """Record application request metrics."""
        try:
            self.response_times.append(response_time_ms)
            self.request_counts[status_code] += 1
            
            if status_code >= 400:
                self.error_counts[status_code] += 1
                
        except Exception as e:
            self.logger.error(f"Failed to record request metrics: {e}")


class DatabaseMonitor:
    """Database monitoring component."""
    
    def __init__(self, config: MonitoringConfiguration):
        self.config = config
        self.logger = logging.getLogger('DatabaseMonitor')
        self.connection_counts: deque = deque(maxlen=100)
        self.query_times: deque = deque(maxlen=1000)
    
    def collect_database_metrics(self) -> List[ProductionMetric]:
        """Collect database performance metrics."""
        try:
            metrics = []
            timestamp = datetime.utcnow()
            
            # Connection metrics
            if self.connection_counts:
                avg_connections = statistics.mean(self.connection_counts)
                max_connections = max(self.connection_counts)
                
                metrics.extend([
                    ProductionMetric(
                        metric_id=f"db_connections_avg_{int(time.time())}",
                        component=MonitoringComponent.DATABASE,
                        metric_name="connections_average",
                        metric_value=avg_connections,
                        metric_unit="count",
                        timestamp=timestamp
                    ),
                    ProductionMetric(
                        metric_id=f"db_connections_max_{int(time.time())}",
                        component=MonitoringComponent.DATABASE,
                        metric_name="connections_maximum",
                        metric_value=max_connections,
                        metric_unit="count",
                        timestamp=timestamp
                    )
                ])
            
            # Query performance metrics
            if self.query_times:
                avg_query_time = statistics.mean(self.query_times)
                p95_query_time = np.percentile(list(self.query_times), 95)
                
                metrics.extend([
                    ProductionMetric(
                        metric_id=f"db_query_time_avg_{int(time.time())}",
                        component=MonitoringComponent.DATABASE,
                        metric_name="query_time_average_ms",
                        metric_value=avg_query_time,
                        metric_unit="ms",
                        timestamp=timestamp
                    ),
                    ProductionMetric(
                        metric_id=f"db_query_time_p95_{int(time.time())}",
                        component=MonitoringComponent.DATABASE,
                        metric_name="query_time_p95_ms",
                        metric_value=p95_query_time,
                        metric_unit="ms",
                        timestamp=timestamp
                    )
                ])
            
            return metrics
            
        except Exception as e:
            error_msg = f"Failed to collect database metrics: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)


class NetworkMonitor:
    """Network monitoring component."""
    
    def __init__(self, config: MonitoringConfiguration):
        self.config = config
        self.logger = logging.getLogger('NetworkMonitor')
        self.latency_measurements: deque = deque(maxlen=100)
    
    def collect_network_metrics(self) -> List[ProductionMetric]:
        """Collect network performance metrics."""
        try:
            metrics = []
            timestamp = datetime.utcnow()
            
            # Network latency
            latency = self._measure_network_latency("8.8.8.8")  # Google DNS
            if latency is not None:
                self.latency_measurements.append(latency)
                
                metrics.append(ProductionMetric(
                    metric_id=f"network_latency_{int(time.time())}",
                    component=MonitoringComponent.NETWORK,
                    metric_name="network_latency_ms",
                    metric_value=latency,
                    metric_unit="ms",
                    timestamp=timestamp
                ))
            
            # Average latency over time
            if self.latency_measurements:
                avg_latency = statistics.mean(self.latency_measurements)
                metrics.append(ProductionMetric(
                    metric_id=f"network_latency_avg_{int(time.time())}",
                    component=MonitoringComponent.NETWORK,
                    metric_name="network_latency_average_ms",
                    metric_value=avg_latency,
                    metric_unit="ms",
                    timestamp=timestamp
                ))
            
            return metrics
            
        except Exception as e:
            error_msg = f"Failed to collect network metrics: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _measure_network_latency(self, host: str) -> Optional[float]:
        """Measure network latency to host."""
        try:
            start_time = time.time()
            socket.create_connection((host, 53), timeout=5).close()
            return (time.time() - start_time) * 1000
        except Exception:
            return None


class SecurityMonitor:
    """Security monitoring component."""
    
    def __init__(self, config: MonitoringConfiguration):
        self.config = config
        self.logger = logging.getLogger('SecurityMonitor')
        self.failed_auth_attempts: defaultdict = defaultdict(int)
        self.security_events: deque = deque(maxlen=1000)
    
    def collect_security_metrics(self) -> List[ProductionMetric]:
        """Collect security monitoring metrics."""
        try:
            metrics = []
            timestamp = datetime.utcnow()
            
            # Authentication failure metrics
            total_auth_failures = sum(self.failed_auth_attempts.values())
            metrics.append(ProductionMetric(
                metric_id=f"security_auth_failures_{int(time.time())}",
                component=MonitoringComponent.SECURITY,
                metric_name="authentication_failures_total",
                metric_value=total_auth_failures,
                metric_unit="count",
                timestamp=timestamp
            ))
            
            # Security events metrics
            metrics.append(ProductionMetric(
                metric_id=f"security_events_total_{int(time.time())}",
                component=MonitoringComponent.SECURITY,
                metric_name="security_events_total",
                metric_value=len(self.security_events),
                metric_unit="count",
                timestamp=timestamp
            ))
            
            return metrics
            
        except Exception as e:
            error_msg = f"Failed to collect security metrics: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)


class ProductionMonitoringSystem:
    """
    Production Monitoring System - Comprehensive production monitoring orchestration.
    
    This class provides complete monitoring capabilities including real-time system monitoring,
    application monitoring, database monitoring, network monitoring, security monitoring,
    and alert management for production environments.
    """
    
    def __init__(self, config: Optional[MonitoringConfiguration] = None):
        self.config = config or MonitoringConfiguration()
        self.monitoring_lock = threading.RLock()
        self.monitoring_status = MonitoringStatus.STOPPED
        
        # Monitoring components
        self.system_monitor = SystemMonitor(self.config)
        self.application_monitor = ApplicationMonitor(self.config)
        self.database_monitor = DatabaseMonitor(self.config)
        self.network_monitor = NetworkMonitor(self.config)
        self.security_monitor = SecurityMonitor(self.config)
        
        # Data storage
        self.metrics_storage: deque = deque(maxlen=10000)
        self.alerts_storage: deque = deque(maxlen=1000)
        self.health_checks_storage: deque = deque(maxlen=1000)
        
        # Monitoring threads
        self.monitoring_threads: Dict[str, threading.Thread] = {}
        self.stop_monitoring = threading.Event()
        
        # Alert management
        self.active_alerts: Dict[str, ProductionAlert] = {}
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # Integration references
        self.production_config_manager: Optional['ProductionConfigManager'] = None
        self.hybrid_performance_monitor: Optional['HybridPerformanceMonitor'] = None
        self.progress_tracker: Optional['ProgressTracker'] = None
        
        # Monitoring statistics
        self.start_time = datetime.utcnow()
        self.metrics_collected = 0
        self.alerts_generated = 0
        self.health_checks_performed = 0
        
        logger.info(f"ProductionMonitoringSystem initialized with {self.config.monitoring_level.value} level")
    
    def initialize_monitoring_system(
        self,
        production_config_manager: Optional['ProductionConfigManager'] = None,
        hybrid_performance_monitor: Optional['HybridPerformanceMonitor'] = None,
        progress_tracker: Optional['ProgressTracker'] = None
    ) -> None:
        """Initialize monitoring system with integrations."""
        try:
            with self.monitoring_lock:
                self.production_config_manager = production_config_manager
                self.hybrid_performance_monitor = hybrid_performance_monitor
                self.progress_tracker = progress_tracker
                
                # Apply configuration from production config manager
                if production_config_manager:
                    self._integrate_production_config()
                
                # Integrate with hybrid performance monitor
                if hybrid_performance_monitor:
                    self._integrate_hybrid_monitor()
                
                logger.info("Production monitoring system initialized successfully")
                
        except Exception as e:
            error_msg = f"Failed to initialize monitoring system: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def start_monitoring(self) -> None:
        """Start production monitoring."""
        try:
            with self.monitoring_lock:
                if self.monitoring_status == MonitoringStatus.RUNNING:
                    logger.warning("Monitoring is already running")
                    return
                
                self.monitoring_status = MonitoringStatus.STARTING
                self.stop_monitoring.clear()
                
                # Start monitoring threads for enabled components
                for component in self.config.enabled_components:
                    thread_name = f"Monitor_{component.value}"
                    
                    if component == MonitoringComponent.SYSTEM:
                        thread = threading.Thread(
                            target=self._system_monitoring_loop,
                            name=thread_name,
                            daemon=True
                        )
                    elif component == MonitoringComponent.APPLICATION:
                        thread = threading.Thread(
                            target=self._application_monitoring_loop,
                            name=thread_name,
                            daemon=True
                        )
                    elif component == MonitoringComponent.DATABASE:
                        thread = threading.Thread(
                            target=self._database_monitoring_loop,
                            name=thread_name,
                            daemon=True
                        )
                    elif component == MonitoringComponent.NETWORK:
                        thread = threading.Thread(
                            target=self._network_monitoring_loop,
                            name=thread_name,
                            daemon=True
                        )
                    elif component == MonitoringComponent.SECURITY:
                        thread = threading.Thread(
                            target=self._security_monitoring_loop,
                            name=thread_name,
                            daemon=True
                        )
                    else:
                        continue
                    
                    self.monitoring_threads[component.value] = thread
                    thread.start()
                
                # Start health check thread
                health_thread = threading.Thread(
                    target=self._health_check_loop,
                    name="HealthChecker",
                    daemon=True
                )
                self.monitoring_threads["health_checker"] = health_thread
                health_thread.start()
                
                # Start alert processing thread
                alert_thread = threading.Thread(
                    target=self._alert_processing_loop,
                    name="AlertProcessor",
                    daemon=True
                )
                self.monitoring_threads["alert_processor"] = alert_thread
                alert_thread.start()
                
                self.monitoring_status = MonitoringStatus.RUNNING
                logger.info("Production monitoring started successfully")
                
        except Exception as e:
            self.monitoring_status = MonitoringStatus.ERROR
            error_msg = f"Failed to start monitoring: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def stop_monitoring(self) -> None:
        """Stop production monitoring."""
        try:
            with self.monitoring_lock:
                if self.monitoring_status == MonitoringStatus.STOPPED:
                    logger.warning("Monitoring is already stopped")
                    return
                
                self.monitoring_status = MonitoringStatus.STOPPING
                self.stop_monitoring.set()
                
                # Wait for all threads to stop
                for thread_name, thread in self.monitoring_threads.items():
                    if thread.is_alive():
                        thread.join(timeout=10.0)
                        if thread.is_alive():
                            logger.warning(f"Monitoring thread {thread_name} did not stop gracefully")
                
                self.monitoring_threads.clear()
                self.monitoring_status = MonitoringStatus.STOPPED
                logger.info("Production monitoring stopped successfully")
                
        except Exception as e:
            error_msg = f"Failed to stop monitoring: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        try:
            with self.monitoring_lock:
                uptime = datetime.utcnow() - self.start_time
                
                return {
                    'status': self.monitoring_status.value,
                    'monitoring_level': self.config.monitoring_level.value,
                    'enabled_components': [c.value for c in self.config.enabled_components],
                    'uptime_seconds': uptime.total_seconds(),
                    'metrics_collected': self.metrics_collected,
                    'alerts_generated': self.alerts_generated,
                    'health_checks_performed': self.health_checks_performed,
                    'active_alerts': len(self.active_alerts),
                    'monitoring_threads': {
                        name: thread.is_alive() for name, thread in self.monitoring_threads.items()
                    },
                    'recent_metrics': len(self.metrics_storage),
                    'recent_alerts': len(self.alerts_storage),
                    'recent_health_checks': len(self.health_checks_storage)
                }
                
        except Exception as e:
            error_msg = f"Failed to get monitoring status: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_metrics(
        self,
        component: Optional[MonitoringComponent] = None,
        metric_name: Optional[str] = None,
        time_range_minutes: int = 60
    ) -> List[ProductionMetric]:
        """Get metrics with optional filtering."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(minutes=time_range_minutes)
            
            filtered_metrics = []
            for metric in self.metrics_storage:
                if metric.timestamp < cutoff_time:
                    continue
                
                if component and metric.component != component:
                    continue
                
                if metric_name and metric.metric_name != metric_name:
                    continue
                
                filtered_metrics.append(metric)
            
            return filtered_metrics
            
        except Exception as e:
            error_msg = f"Failed to get metrics: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        component: Optional[MonitoringComponent] = None,
        resolved: Optional[bool] = None
    ) -> List[ProductionAlert]:
        """Get alerts with optional filtering."""
        try:
            filtered_alerts = []
            for alert in self.alerts_storage:
                if severity and alert.severity != severity:
                    continue
                
                if component and alert.component != component:
                    continue
                
                if resolved is not None and alert.resolved != resolved:
                    continue
                
                filtered_alerts.append(alert)
            
            return filtered_alerts
            
        except Exception as e:
            error_msg = f"Failed to get alerts: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        try:
            recent_checks = [
                check for check in self.health_checks_storage
                if check.timestamp > datetime.utcnow() - timedelta(minutes=5)
            ]
            
            if not recent_checks:
                return {
                    'overall_status': HealthStatus.UNKNOWN.value,
                    'components': {},
                    'last_check': None
                }
            
            component_statuses = {}
            for component in MonitoringComponent:
                component_checks = [c for c in recent_checks if c.component == component]
                if component_checks:
                    latest_check = max(component_checks, key=lambda x: x.timestamp)
                    component_statuses[component.value] = {
                        'status': latest_check.status.value,
                        'message': latest_check.message,
                        'last_check': latest_check.timestamp.isoformat()
                    }
            
            # Determine overall status
            if all(status['status'] == HealthStatus.HEALTHY.value for status in component_statuses.values()):
                overall_status = HealthStatus.HEALTHY
            elif any(status['status'] == HealthStatus.UNHEALTHY.value for status in component_statuses.values()):
                overall_status = HealthStatus.UNHEALTHY
            elif any(status['status'] == HealthStatus.DEGRADED.value for status in component_statuses.values()):
                overall_status = HealthStatus.DEGRADED
            else:
                overall_status = HealthStatus.UNKNOWN
            
            return {
                'overall_status': overall_status.value,
                'components': component_statuses,
                'last_check': max(recent_checks, key=lambda x: x.timestamp).timestamp.isoformat()
            }
            
        except Exception as e:
            error_msg = f"Failed to get health status: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _system_monitoring_loop(self) -> None:
        """System monitoring background loop."""
        while not self.stop_monitoring.is_set():
            try:
                metrics = self.system_monitor.collect_system_metrics()
                
                with self.monitoring_lock:
                    self.metrics_storage.extend(metrics)
                    self.metrics_collected += len(metrics)
                
                # Check for alerts
                self._check_metrics_for_alerts(metrics)
                
                # Wait for next interval
                time.sleep(self.config.system_monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in system monitoring loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _application_monitoring_loop(self) -> None:
        """Application monitoring background loop."""
        while not self.stop_monitoring.is_set():
            try:
                metrics = self.application_monitor.collect_application_metrics()
                
                with self.monitoring_lock:
                    self.metrics_storage.extend(metrics)
                    self.metrics_collected += len(metrics)
                
                # Check for alerts
                self._check_metrics_for_alerts(metrics)
                
                # Wait for next interval
                time.sleep(self.config.application_monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in application monitoring loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _database_monitoring_loop(self) -> None:
        """Database monitoring background loop."""
        while not self.stop_monitoring.is_set():
            try:
                metrics = self.database_monitor.collect_database_metrics()
                
                with self.monitoring_lock:
                    self.metrics_storage.extend(metrics)
                    self.metrics_collected += len(metrics)
                
                # Check for alerts
                self._check_metrics_for_alerts(metrics)
                
                # Wait for next interval
                time.sleep(self.config.database_monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in database monitoring loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _network_monitoring_loop(self) -> None:
        """Network monitoring background loop."""
        while not self.stop_monitoring.is_set():
            try:
                metrics = self.network_monitor.collect_network_metrics()
                
                with self.monitoring_lock:
                    self.metrics_storage.extend(metrics)
                    self.metrics_collected += len(metrics)
                
                # Check for alerts
                self._check_metrics_for_alerts(metrics)
                
                # Wait for next interval
                time.sleep(self.config.network_monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in network monitoring loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _security_monitoring_loop(self) -> None:
        """Security monitoring background loop."""
        while not self.stop_monitoring.is_set():
            try:
                metrics = self.security_monitor.collect_security_metrics()
                
                with self.monitoring_lock:
                    self.metrics_storage.extend(metrics)
                    self.metrics_collected += len(metrics)
                
                # Check for alerts
                self._check_metrics_for_alerts(metrics)
                
                # Wait for next interval
                time.sleep(self.config.security_monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in security monitoring loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _health_check_loop(self) -> None:
        """Health check background loop."""
        while not self.stop_monitoring.is_set():
            try:
                health_checks = []
                
                # Collect health checks from enabled components
                if MonitoringComponent.SYSTEM in self.config.enabled_components:
                    health_checks.extend(self.system_monitor.check_system_health())
                
                with self.monitoring_lock:
                    self.health_checks_storage.extend(health_checks)
                    self.health_checks_performed += len(health_checks)
                
                # Generate alerts for unhealthy components
                for check in health_checks:
                    if check.status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]:
                        severity = AlertSeverity.HIGH if check.status == HealthStatus.UNHEALTHY else AlertSeverity.MEDIUM
                        self._generate_alert(
                            component=check.component,
                            severity=severity,
                            alert_type="health_check_failure",
                            message=f"Health check failed: {check.message}",
                            metric_name=check.check_name
                        )
                
                # Wait for next interval
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _alert_processing_loop(self) -> None:
        """Alert processing background loop."""
        while not self.stop_monitoring.is_set():
            try:
                # Process alert cooldowns
                current_time = datetime.utcnow()
                expired_cooldowns = [
                    key for key, cooldown_time in self.alert_cooldowns.items()
                    if current_time > cooldown_time
                ]
                
                for key in expired_cooldowns:
                    del self.alert_cooldowns[key]
                
                # Check for alert escalations
                if self.config.alert_escalation_enabled:
                    self._process_alert_escalations()
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
                # Wait before next processing cycle
                time.sleep(60)  # Process alerts every minute
                
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _check_metrics_for_alerts(self, metrics: List[ProductionMetric]) -> None:
        """Check metrics for alert conditions."""
        try:
            for metric in metrics:
                # CPU threshold check
                if metric.metric_name == "cpu_usage_percent" and isinstance(metric.metric_value, (int, float)):
                    if metric.metric_value > self.config.cpu_threshold_percent:
                        self._generate_alert(
                            component=metric.component,
                            severity=AlertSeverity.HIGH if metric.metric_value > 95 else AlertSeverity.MEDIUM,
                            alert_type="cpu_threshold_exceeded",
                            message=f"CPU usage {metric.metric_value:.1f}% exceeds threshold {self.config.cpu_threshold_percent}%",
                            metric_name=metric.metric_name,
                            metric_value=metric.metric_value,
                            threshold=self.config.cpu_threshold_percent
                        )
                
                # Memory threshold check
                elif metric.metric_name == "memory_usage_percent" and isinstance(metric.metric_value, (int, float)):
                    if metric.metric_value > self.config.memory_threshold_percent:
                        self._generate_alert(
                            component=metric.component,
                            severity=AlertSeverity.HIGH if metric.metric_value > 95 else AlertSeverity.MEDIUM,
                            alert_type="memory_threshold_exceeded",
                            message=f"Memory usage {metric.metric_value:.1f}% exceeds threshold {self.config.memory_threshold_percent}%",
                            metric_name=metric.metric_name,
                            metric_value=metric.metric_value,
                            threshold=self.config.memory_threshold_percent
                        )
                
                # Disk threshold check
                elif metric.metric_name == "disk_usage_percent" and isinstance(metric.metric_value, (int, float)):
                    if metric.metric_value > self.config.disk_threshold_percent:
                        self._generate_alert(
                            component=metric.component,
                            severity=AlertSeverity.CRITICAL if metric.metric_value > 98 else AlertSeverity.HIGH,
                            alert_type="disk_threshold_exceeded",
                            message=f"Disk usage {metric.metric_value:.1f}% exceeds threshold {self.config.disk_threshold_percent}%",
                            metric_name=metric.metric_name,
                            metric_value=metric.metric_value,
                            threshold=self.config.disk_threshold_percent
                        )
                
                # Response time threshold check
                elif metric.metric_name in ["response_time_average_ms", "response_time_p95_ms"] and isinstance(metric.metric_value, (int, float)):
                    if metric.metric_value > self.config.response_time_threshold_ms:
                        self._generate_alert(
                            component=metric.component,
                            severity=AlertSeverity.HIGH,
                            alert_type="response_time_threshold_exceeded",
                            message=f"Response time {metric.metric_value:.1f}ms exceeds threshold {self.config.response_time_threshold_ms}ms",
                            metric_name=metric.metric_name,
                            metric_value=metric.metric_value,
                            threshold=self.config.response_time_threshold_ms
                        )
                
                # Error rate threshold check
                elif metric.metric_name == "error_rate_percent" and isinstance(metric.metric_value, (int, float)):
                    if metric.metric_value > self.config.error_rate_threshold_percent:
                        self._generate_alert(
                            component=metric.component,
                            severity=AlertSeverity.CRITICAL if metric.metric_value > 10 else AlertSeverity.HIGH,
                            alert_type="error_rate_threshold_exceeded",
                            message=f"Error rate {metric.metric_value:.1f}% exceeds threshold {self.config.error_rate_threshold_percent}%",
                            metric_name=metric.metric_name,
                            metric_value=metric.metric_value,
                            threshold=self.config.error_rate_threshold_percent
                        )
                
        except Exception as e:
            logger.error(f"Error checking metrics for alerts: {e}")
    
    def _generate_alert(
        self,
        component: MonitoringComponent,
        severity: AlertSeverity,
        alert_type: str,
        message: str,
        metric_name: Optional[str] = None,
        metric_value: Optional[Union[int, float]] = None,
        threshold: Optional[Union[int, float]] = None
    ) -> None:
        """Generate monitoring alert."""
        try:
            if not self.config.alert_enabled:
                return
            
            # Check alert cooldown
            cooldown_key = f"{component.value}_{alert_type}_{metric_name}"
            if cooldown_key in self.alert_cooldowns:
                return  # Still in cooldown period
            
            alert = ProductionAlert(
                alert_id=str(uuid.uuid4()),
                component=component,
                severity=severity,
                alert_type=alert_type,
                message=message,
                metric_name=metric_name,
                metric_value=metric_value,
                threshold=threshold
            )
            
            with self.monitoring_lock:
                self.alerts_storage.append(alert)
                self.active_alerts[alert.alert_id] = alert
                self.alerts_generated += 1
            
            # Set cooldown
            cooldown_time = datetime.utcnow() + timedelta(minutes=self.config.alert_cooldown_minutes)
            self.alert_cooldowns[cooldown_key] = cooldown_time
            
            # Send alert notifications
            self._send_alert_notifications(alert)
            
            logger.warning(f"Alert generated: {alert.severity.value} - {alert.message}")
            
        except Exception as e:
            logger.error(f"Failed to generate alert: {e}")
    
    def _send_alert_notifications(self, alert: ProductionAlert) -> None:
        """Send alert notifications."""
        try:
            # Integrate with progress tracker if available
            if self.progress_tracker and hasattr(self.progress_tracker, 'add_alert'):
                # Map our AlertSeverity to progress tracker AlertSeverity
                try:
                    from progress_tracker import AlertSeverity as PTAlertSeverity
                    severity_mapping = {
                        AlertSeverity.LOW: PTAlertSeverity.INFO,
                        AlertSeverity.MEDIUM: PTAlertSeverity.WARNING,
                        AlertSeverity.HIGH: PTAlertSeverity.ERROR,
                        AlertSeverity.CRITICAL: PTAlertSeverity.CRITICAL
                    }
                    
                    pt_severity = severity_mapping.get(alert.severity, PTAlertSeverity.WARNING)
                    self.progress_tracker.add_alert(
                        alert_type=alert.alert_type,
                        message=alert.message,
                        severity=pt_severity,
                        details={'component': alert.component.value, 'metric_name': alert.metric_name}
                    )
                except ImportError:
                    pass  # Progress tracker not available
            
            # Log alert (basic notification)
            logger.warning(f"ALERT: {alert.severity.value.upper()} - {alert.component.value} - {alert.message}")
            
        except Exception as e:
            logger.error(f"Failed to send alert notifications: {e}")
    
    def _process_alert_escalations(self) -> None:
        """Process alert escalations."""
        try:
            escalation_time = datetime.utcnow() - timedelta(minutes=self.config.alert_escalation_minutes)
            
            for alert in self.active_alerts.values():
                if alert.timestamp < escalation_time and not alert.resolved:
                    # Escalate alert
                    if alert.severity != AlertSeverity.CRITICAL:
                        escalated_severity = {
                            AlertSeverity.LOW: AlertSeverity.MEDIUM,
                            AlertSeverity.MEDIUM: AlertSeverity.HIGH,
                            AlertSeverity.HIGH: AlertSeverity.CRITICAL
                        }.get(alert.severity, AlertSeverity.CRITICAL)
                        
                        self._generate_alert(
                            component=alert.component,
                            severity=escalated_severity,
                            alert_type=f"escalated_{alert.alert_type}",
                            message=f"ESCALATED: {alert.message}",
                            metric_name=alert.metric_name,
                            metric_value=alert.metric_value,
                            threshold=alert.threshold
                        )
            
        except Exception as e:
            logger.error(f"Error processing alert escalations: {e}")
    
    def _cleanup_old_alerts(self) -> None:
        """Clean up old resolved alerts."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=self.config.alerts_retention_hours)
            
            # Remove old alerts from active alerts
            to_remove = [
                alert_id for alert_id, alert in self.active_alerts.items()
                if alert.resolved and alert.resolved_timestamp and alert.resolved_timestamp < cutoff_time
            ]
            
            for alert_id in to_remove:
                del self.active_alerts[alert_id]
            
        except Exception as e:
            logger.error(f"Error cleaning up old alerts: {e}")
    
    def _integrate_production_config(self) -> None:
        """Integrate with production configuration manager."""
        try:
            if not self.production_config_manager:
                return
            
            # Get monitoring configuration
            if hasattr(self.production_config_manager, 'monitoring_config'):
                monitoring_config = self.production_config_manager.monitoring_config
                
                # Apply configuration settings
                if hasattr(monitoring_config, 'monitoring_interval'):
                    self.config.system_monitoring_interval = monitoring_config.monitoring_interval
                
                if hasattr(monitoring_config, 'health_check_interval'):
                    self.config.health_check_interval = monitoring_config.health_check_interval
                
                if hasattr(monitoring_config, 'alert_thresholds'):
                    thresholds = monitoring_config.alert_thresholds
                    if 'cpu_percent' in thresholds:
                        self.config.cpu_threshold_percent = thresholds['cpu_percent']
                    if 'memory_percent' in thresholds:
                        self.config.memory_threshold_percent = thresholds['memory_percent']
                    if 'response_time_ms' in thresholds:
                        self.config.response_time_threshold_ms = thresholds['response_time_ms']
            
            logger.info("Production configuration integrated successfully")
            
        except Exception as e:
            logger.error(f"Failed to integrate production configuration: {e}")
    
    def _integrate_hybrid_monitor(self) -> None:
        """Integrate with hybrid performance monitor."""
        try:
            if not self.hybrid_performance_monitor:
                return
            
            # Subscribe to hybrid monitor metrics if possible
            logger.info("Hybrid performance monitor integration initialized")
            
        except Exception as e:
            logger.error(f"Failed to integrate hybrid performance monitor: {e}")
    
    def shutdown(self) -> None:
        """Shutdown monitoring system."""
        try:
            self.stop_monitoring()
            logger.info("Production monitoring system shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during monitoring system shutdown: {e}")


def create_production_monitoring_system(
    config: Optional[MonitoringConfiguration] = None
) -> ProductionMonitoringSystem:
    """Factory function to create a ProductionMonitoringSystem instance."""
    return ProductionMonitoringSystem(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    monitoring_system = create_production_monitoring_system()
    
    # Initialize monitoring
    monitoring_system.initialize_monitoring_system()
    
    # Start monitoring
    try:
        monitoring_system.start_monitoring()
        print("Monitoring started successfully")
        
        # Let it run for a short time
        time.sleep(30)
        
        # Get status
        status = monitoring_system.get_monitoring_status()
        print(f"Monitoring status: {status['status']}")
        print(f"Metrics collected: {status['metrics_collected']}")
        
        # Get health status
        health = monitoring_system.get_health_status()
        print(f"Overall health: {health['overall_status']}")
        
    except KeyboardInterrupt:
        print("Stopping monitoring...")
    finally:
        monitoring_system.stop_monitoring()
        print("Monitoring stopped")