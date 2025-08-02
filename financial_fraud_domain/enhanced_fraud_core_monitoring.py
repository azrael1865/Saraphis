"""
Enhanced Fraud Detection Core - Chunk 4: Monitoring and Performance Optimization
Comprehensive monitoring, metrics collection, and performance optimization for fraud detection
"""

import logging
import time
import threading
import asyncio
import psutil
import json
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import weakref
from functools import wraps
import gc
import traceback
import resource
import sqlite3
import pickle
import gzip
import statistics
from contextlib import contextmanager
import cProfile
import pstats
import io

# Import core exceptions and enums
# Try absolute imports first (for direct module imports)
try:
    from enhanced_fraud_core_exceptions import (
        EnhancedFraudException, PerformanceError, MonitoringError, ResourceError,
        MonitoringLevel, AlertSeverity, ErrorContext, create_error_context
    )
except ImportError as e:
    # Fallback to relative imports (when imported as part of a package)
    try:
        from enhanced_fraud_core_exceptions import (
            EnhancedFraudException, PerformanceError, MonitoringError, ResourceError,
            MonitoringLevel, AlertSeverity, ErrorContext, create_error_context
        )
    except ImportError:
        raise ImportError(
            "Failed to import required modules for EnhancedFraudCoreMonitoring. "
            "Please ensure enhanced_fraud_core_exceptions module is available."
        ) from e

# Configure logging
logger = logging.getLogger(__name__)

# ======================== MONITORING CONFIGURATION ========================

@dataclass
class MonitoringConfig:
    """Configuration for monitoring system"""
    # Collection intervals
    metrics_collection_interval: float = 5.0  # seconds
    performance_monitoring_interval: float = 10.0
    resource_monitoring_interval: float = 15.0
    health_check_interval: float = 30.0
    
    # Storage settings
    metrics_retention_hours: int = 24
    detailed_metrics_retention_hours: int = 6
    performance_history_size: int = 1000
    
    # Alert thresholds
    cpu_threshold: float = 0.8
    memory_threshold: float = 0.8
    disk_threshold: float = 0.9
    response_time_threshold: float = 2.0  # seconds
    error_rate_threshold: float = 0.05  # 5%
    
    # Performance optimization
    enable_performance_profiling: bool = True
    enable_resource_monitoring: bool = True
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # 1 hour
    
    # Monitoring levels
    monitoring_level: MonitoringLevel = MonitoringLevel.DETAILED
    log_performance_metrics: bool = True
    log_resource_usage: bool = True
    
    # Database settings
    metrics_db_path: str = "fraud_metrics.db"
    compress_stored_metrics: bool = True

# ======================== PERFORMANCE METRICS ========================

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    operation_name: str
    duration: float
    success: bool
    error_type: Optional[str] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResourceMetrics:
    """Resource usage metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int] = field(default_factory=dict)
    disk_io: Dict[str, int] = field(default_factory=dict)
    open_files: int = 0
    thread_count: int = 0

@dataclass
class HealthMetrics:
    """Health check metrics"""
    timestamp: datetime
    component_name: str
    is_healthy: bool
    response_time: float
    error_message: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)

# ======================== METRICS COLLECTOR ========================

class MetricsCollector:
    """Collects and stores performance metrics"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.performance_metrics = deque(maxlen=config.performance_history_size)
        self.resource_metrics = deque(maxlen=config.performance_history_size)
        self.health_metrics = deque(maxlen=config.performance_history_size)
        
        # Real-time metrics
        self.current_metrics = {
            'requests_per_second': 0,
            'average_response_time': 0.0,
            'error_rate': 0.0,
            'active_connections': 0,
            'cache_hit_rate': 0.0
        }
        
        # Thread-safe operations
        self.lock = threading.Lock()
        
        # Database for persistent storage
        self.db_connection = None
        self._init_database()
        
        # Background monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.start_monitoring()
    
    def _init_database(self) -> None:
        """Initialize metrics database"""
        try:
            self.db_connection = sqlite3.connect(
                self.config.metrics_db_path,
                check_same_thread=False
            )
            
            # Create tables
            self.db_connection.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    operation_name TEXT NOT NULL,
                    duration REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_type TEXT,
                    memory_usage REAL,
                    cpu_usage REAL,
                    additional_data TEXT
                )
            ''')
            
            self.db_connection.execute('''
                CREATE TABLE IF NOT EXISTS resource_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_percent REAL NOT NULL,
                    memory_percent REAL NOT NULL,
                    disk_percent REAL NOT NULL,
                    network_io TEXT,
                    disk_io TEXT,
                    open_files INTEGER,
                    thread_count INTEGER
                )
            ''')
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics database: {e}")
            self.db_connection = None
    
    def record_performance_metric(self, metric: PerformanceMetrics) -> None:
        """Record a performance metric"""
        with self.lock:
            self.performance_metrics.append(metric)
            
            # Update real-time metrics
            self._update_realtime_metrics(metric)
            
            # Store in database
            if self.db_connection:
                self._store_performance_metric(metric)
    
    def record_resource_metric(self, metric: ResourceMetrics) -> None:
        """Record a resource metric"""
        with self.lock:
            self.resource_metrics.append(metric)
            
            # Store in database
            if self.db_connection:
                self._store_resource_metric(metric)
    
    def record_health_metric(self, metric: HealthMetrics) -> None:
        """Record a health metric"""
        with self.lock:
            self.health_metrics.append(metric)
    
    def _update_realtime_metrics(self, metric: PerformanceMetrics) -> None:
        """Update real-time metrics based on new performance data"""
        # Calculate requests per second (last minute)
        now = datetime.now()
        recent_metrics = [
            m for m in self.performance_metrics
            if (now - m.timestamp).total_seconds() <= 60
        ]
        
        if recent_metrics:
            self.current_metrics['requests_per_second'] = len(recent_metrics) / 60.0
            self.current_metrics['average_response_time'] = statistics.mean(
                [m.duration for m in recent_metrics]
            )
            
            error_count = sum(1 for m in recent_metrics if not m.success)
            self.current_metrics['error_rate'] = error_count / len(recent_metrics)
    
    def _store_performance_metric(self, metric: PerformanceMetrics) -> None:
        """Store performance metric in database"""
        try:
            additional_data = json.dumps(metric.additional_data) if metric.additional_data else None
            
            if self.config.compress_stored_metrics and additional_data:
                additional_data = gzip.compress(additional_data.encode('utf-8'))
            
            self.db_connection.execute('''
                INSERT INTO performance_metrics 
                (timestamp, operation_name, duration, success, error_type, memory_usage, cpu_usage, additional_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metric.timestamp.isoformat(),
                metric.operation_name,
                metric.duration,
                metric.success,
                metric.error_type,
                metric.memory_usage,
                metric.cpu_usage,
                additional_data
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Failed to store performance metric: {e}")
    
    def _store_resource_metric(self, metric: ResourceMetrics) -> None:
        """Store resource metric in database"""
        try:
            self.db_connection.execute('''
                INSERT INTO resource_metrics 
                (timestamp, cpu_percent, memory_percent, disk_percent, network_io, disk_io, open_files, thread_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metric.timestamp.isoformat(),
                metric.cpu_percent,
                metric.memory_percent,
                metric.disk_percent,
                json.dumps(metric.network_io),
                json.dumps(metric.disk_io),
                metric.open_files,
                metric.thread_count
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Failed to store resource metric: {e}")
    
    def start_monitoring(self) -> None:
        """Start background monitoring thread"""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("Monitoring thread started")
    
    def stop_monitoring_thread(self) -> None:
        """Stop background monitoring thread"""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Monitoring thread stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while not self.stop_monitoring.is_set():
            try:
                # Collect resource metrics
                if self.config.enable_resource_monitoring:
                    resource_metric = self._collect_resource_metrics()
                    self.record_resource_metric(resource_metric)
                
                # Cleanup old metrics
                self._cleanup_old_metrics()
                
                # Sleep until next collection
                time.sleep(self.config.resource_monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config.resource_monitoring_interval)
    
    def _collect_resource_metrics(self) -> ResourceMetrics:
        """Collect current resource usage metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network_io = psutil.net_io_counters()._asdict()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()._asdict()
            
            # Process info
            process = psutil.Process()
            open_files = len(process.open_files())
            thread_count = process.num_threads()
            
            return ResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                network_io=network_io,
                disk_io=disk_io,
                open_files=open_files,
                thread_count=thread_count
            )
            
        except Exception as e:
            logger.error(f"Failed to collect resource metrics: {e}")
            return ResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0
            )
    
    def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics from database"""
        if not self.db_connection:
            return
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.config.metrics_retention_hours)
            
            self.db_connection.execute(
                'DELETE FROM performance_metrics WHERE timestamp < ?',
                (cutoff_time.isoformat(),)
            )
            
            self.db_connection.execute(
                'DELETE FROM resource_metrics WHERE timestamp < ?',
                (cutoff_time.isoformat(),)
            )
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics"""
        with self.lock:
            return self.current_metrics.copy()
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_metrics = [
                m for m in self.performance_metrics
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {'message': 'No metrics available for the specified period'}
        
        # Calculate statistics
        durations = [m.duration for m in recent_metrics]
        success_count = sum(1 for m in recent_metrics if m.success)
        error_count = len(recent_metrics) - success_count
        
        return {
            'total_requests': len(recent_metrics),
            'successful_requests': success_count,
            'failed_requests': error_count,
            'success_rate': success_count / len(recent_metrics),
            'average_response_time': statistics.mean(durations),
            'median_response_time': statistics.median(durations),
            'p95_response_time': statistics.quantiles(durations, n=20)[18] if len(durations) > 1 else durations[0],
            'min_response_time': min(durations),
            'max_response_time': max(durations),
            'requests_per_second': len(recent_metrics) / (hours * 3600)
        }
    
    def get_resource_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get resource usage summary"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_metrics = [
                m for m in self.resource_metrics
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {'message': 'No resource metrics available'}
        
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        return {
            'cpu_usage': {
                'average': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory_usage': {
                'average': statistics.mean(memory_values),
                'max': max(memory_values),
                'min': min(memory_values)
            },
            'current_threads': recent_metrics[-1].thread_count if recent_metrics else 0,
            'current_open_files': recent_metrics[-1].open_files if recent_metrics else 0
        }
    
    def shutdown(self) -> None:
        """Shutdown metrics collector"""
        self.stop_monitoring_thread()
        if self.db_connection:
            self.db_connection.close()
        logger.info("Metrics collector shutdown")

# ======================== PERFORMANCE PROFILER ========================

class PerformanceProfiler:
    """Performance profiler for detailed analysis"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.profiling_enabled = config.enable_performance_profiling
        self.active_profiles = {}
        self.lock = threading.Lock()
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations"""
        if not self.profiling_enabled:
            yield
            return
        
        profiler = cProfile.Profile()
        start_time = time.time()
        
        try:
            profiler.enable()
            yield
        finally:
            profiler.disable()
            duration = time.time() - start_time
            
            # Store profile results
            with self.lock:
                self.active_profiles[operation_name] = {
                    'profiler': profiler,
                    'duration': duration,
                    'timestamp': datetime.now()
                }
    
    def get_profile_results(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get profiling results for an operation"""
        with self.lock:
            if operation_name not in self.active_profiles:
                return None
            
            profile_data = self.active_profiles[operation_name]
            profiler = profile_data['profiler']
            
            # Generate stats
            stats_buffer = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_buffer)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            
            return {
                'operation_name': operation_name,
                'duration': profile_data['duration'],
                'timestamp': profile_data['timestamp'].isoformat(),
                'profile_stats': stats_buffer.getvalue()
            }
    
    def clear_profiles(self) -> None:
        """Clear all stored profiles"""
        with self.lock:
            self.active_profiles.clear()

# ======================== CACHE MANAGER ========================

class CacheManager:
    """High-performance cache manager with monitoring"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.cache = {}
        self.cache_access_times = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            self.cache_stats['total_requests'] += 1
            
            if key in self.cache:
                # Check TTL
                if self._is_expired(key):
                    self._remove_item(key)
                    self.cache_stats['misses'] += 1
                    return None
                
                # Update access time
                self.cache_access_times[key] = time.time()
                self.cache_stats['hits'] += 1
                return self.cache[key]['value']
            else:
                self.cache_stats['misses'] += 1
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache"""
        with self.lock:
            # Check if cache is full
            if len(self.cache) >= self.config.cache_size and key not in self.cache:
                self._evict_lru()
            
            # Store item
            current_time = time.time()
            self.cache[key] = {
                'value': value,
                'created_at': current_time,
                'ttl': ttl or self.config.cache_ttl
            }
            self.cache_access_times[key] = current_time
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache item is expired"""
        if key not in self.cache:
            return True
        
        item = self.cache[key]
        return time.time() - item['created_at'] > item['ttl']
    
    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self.cache_access_times:
            return
        
        # Find least recently used key
        lru_key = min(self.cache_access_times, key=self.cache_access_times.get)
        self._remove_item(lru_key)
        self.cache_stats['evictions'] += 1
    
    def _remove_item(self, key: str) -> None:
        """Remove item from cache"""
        self.cache.pop(key, None)
        self.cache_access_times.pop(key, None)
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        with self.lock:
            total = self.cache_stats['total_requests']
            if total == 0:
                return 0.0
            return self.cache_stats['hits'] / total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'cache_size': len(self.cache),
                'max_size': self.config.cache_size,
                'hit_rate': self.get_hit_rate(),
                'stats': self.cache_stats.copy()
            }
    
    def clear(self) -> None:
        """Clear all cache items"""
        with self.lock:
            self.cache.clear()
            self.cache_access_times.clear()
            self.cache_stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'total_requests': 0
            }

# ======================== MONITORING DECORATORS ========================

def monitor_performance(metrics_collector: MetricsCollector, operation_name: str = None):
    """Decorator to monitor function performance"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            start_time = time.time()
            start_memory = psutil.virtual_memory().percent
            success = True
            error_type = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_type = type(e).__name__
                raise
            finally:
                # Record performance metric
                duration = time.time() - start_time
                end_memory = psutil.virtual_memory().percent
                
                metric = PerformanceMetrics(
                    timestamp=datetime.now(),
                    operation_name=op_name,
                    duration=duration,
                    success=success,
                    error_type=error_type,
                    memory_usage=end_memory - start_memory,
                    cpu_usage=psutil.cpu_percent(interval=None)
                )
                
                metrics_collector.record_performance_metric(metric)
        
        return wrapper
    return decorator

def with_caching(cache_manager: CacheManager, key_func: Callable = None, ttl: int = None):
    """Decorator to add caching to functions"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

def with_profiling(profiler: PerformanceProfiler, operation_name: str = None):
    """Decorator to add profiling to functions"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            with profiler.profile_operation(op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# ======================== MONITORING MANAGER ========================

class MonitoringManager:
    """Comprehensive monitoring manager"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_collector = MetricsCollector(config)
        self.performance_profiler = PerformanceProfiler(config)
        self.cache_manager = CacheManager(config)
        
        # Health checks
        self.health_checks = {}
        self.health_check_results = {}
        
        # Alerts
        self.alert_callbacks = []
        
        # Start health check monitoring
        self._start_health_monitoring()
    
    def add_health_check(self, name: str, check_func: Callable) -> None:
        """Add health check function"""
        self.health_checks[name] = check_func
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    def _start_health_monitoring(self) -> None:
        """Start health check monitoring"""
        def health_check_loop():
            while True:
                try:
                    self._run_health_checks()
                    time.sleep(self.config.health_check_interval)
                except Exception as e:
                    logger.error(f"Error in health check loop: {e}")
                    time.sleep(self.config.health_check_interval)
        
        thread = threading.Thread(target=health_check_loop)
        thread.daemon = True
        thread.start()
    
    def _run_health_checks(self) -> None:
        """Run all health checks"""
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                is_healthy = check_func()
                response_time = time.time() - start_time
                
                metric = HealthMetrics(
                    timestamp=datetime.now(),
                    component_name=name,
                    is_healthy=is_healthy,
                    response_time=response_time
                )
                
                self.metrics_collector.record_health_metric(metric)
                self.health_check_results[name] = metric
                
                # Check for alerts
                if not is_healthy:
                    self._send_alert(AlertSeverity.ERROR, f"Health check failed: {name}")
                
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                
                metric = HealthMetrics(
                    timestamp=datetime.now(),
                    component_name=name,
                    is_healthy=False,
                    response_time=0.0,
                    error_message=str(e)
                )
                
                self.metrics_collector.record_health_metric(metric)
                self.health_check_results[name] = metric
                self._send_alert(AlertSeverity.ERROR, f"Health check error: {name} - {e}")
    
    def _send_alert(self, severity: AlertSeverity, message: str) -> None:
        """Send alert to registered callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(severity, message)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'current_metrics': self.metrics_collector.get_current_metrics(),
            'performance_summary': self.metrics_collector.get_performance_summary(),
            'resource_summary': self.metrics_collector.get_resource_summary(),
            'cache_stats': self.cache_manager.get_stats(),
            'health_checks': {
                name: {
                    'is_healthy': result.is_healthy,
                    'response_time': result.response_time,
                    'last_check': result.timestamp.isoformat(),
                    'error_message': result.error_message
                }
                for name, result in self.health_check_results.items()
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def shutdown(self) -> None:
        """Shutdown monitoring manager"""
        self.metrics_collector.shutdown()
        self.performance_profiler.clear_profiles()
        self.cache_manager.clear()
        logger.info("Monitoring manager shutdown")

# ======================== MONITORING UTILITIES ========================

def create_default_monitoring_manager() -> MonitoringManager:
    """Create default monitoring manager"""
    config = MonitoringConfig()
    return MonitoringManager(config)

def setup_basic_health_checks(manager: MonitoringManager) -> None:
    """Setup basic health checks"""
    
    def memory_health_check():
        memory = psutil.virtual_memory()
        return memory.percent < 90.0
    
    def cpu_health_check():
        cpu_percent = psutil.cpu_percent(interval=1)
        return cpu_percent < 90.0
    
    def disk_health_check():
        disk = psutil.disk_usage('/')
        return disk.percent < 95.0
    
    manager.add_health_check('memory', memory_health_check)
    manager.add_health_check('cpu', cpu_health_check)
    manager.add_health_check('disk', disk_health_check)

def create_performance_dashboard_data(manager: MonitoringManager) -> Dict[str, Any]:
    """Create data for performance dashboard"""
    return {
        'system_status': manager.get_system_status(),
        'alerts': [],  # Would be populated with active alerts
        'trends': {
            'response_time_trend': [],  # Would be populated with historical data
            'error_rate_trend': [],
            'resource_usage_trend': []
        }
    }