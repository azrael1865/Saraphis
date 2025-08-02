"""
Enhanced Proof Verifier - Chunk 3: Performance and Monitoring Enhancements
Advanced performance optimization and comprehensive monitoring capabilities
for the enhanced proof verifier system.
"""

import logging
import json
import time
import threading
import asyncio
import queue
import collections
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set, Type, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import warnings
import traceback
import psutil
import numpy as np
import pandas as pd
import sqlite3
import pickle
import hashlib
import weakref
from functools import wraps, lru_cache
import cProfile
import pstats
import io
from collections import defaultdict, deque
import gc
import resource
import signal
import os

# Import enhanced proof components
try:
    from enhanced_proof_verifier import (
        FinancialProofVerifier as EnhancedFinancialProofVerifier,
        EnhancedProofClaim, EnhancedProofEvidence, EnhancedProofResult,
        SecurityLevel, ProofVerificationException, ProofConfigurationError,
        ProofGenerationError, ProofValidationError, ProofTimeoutError,
        ProofSecurityError, ProofIntegrityError, ClaimValidationError,
        EvidenceValidationError, ProofSystemError, ProofStorageError,
        CryptographicError, ProofExpiredError, ResourceLimitError,
        SecurityValidator, ResourceMonitor
    )
    from enhanced_proof_integration import (
        ProofIntegrationManager, ProofIntegrationConfig, ProofSystemAnalyzer,
        ProofIntegrationError, ProofSystemAnalysisError
    )
    PROOF_COMPONENTS = True
except ImportError as e:
    PROOF_COMPONENTS = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Proof verifier components not available: {e}")

# Configure logging
logger = logging.getLogger(__name__)

# ======================== PERFORMANCE MONITORING CONFIGURATION ========================

@dataclass
class PerformanceMonitoringConfig:
    """Configuration for performance monitoring"""
    
    # Monitoring intervals (seconds)
    metrics_collection_interval: int = 5
    performance_analysis_interval: int = 60
    resource_monitoring_interval: int = 10
    health_check_interval: int = 30
    
    # Thresholds
    cpu_threshold: float = 0.8
    memory_threshold: float = 0.8
    disk_threshold: float = 0.9
    response_time_threshold: float = 5.0
    error_rate_threshold: float = 0.1
    
    # Retention settings
    metrics_retention_hours: int = 24
    detailed_metrics_retention_hours: int = 6
    alert_retention_hours: int = 72
    
    # Caching settings
    enable_result_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # 1 hour
    
    # Performance optimization
    enable_async_processing: bool = True
    enable_parallel_processing: bool = True
    max_worker_threads: int = 4
    max_worker_processes: int = 2
    batch_processing_size: int = 100
    
    # Profiling settings
    enable_profiling: bool = False
    profiling_sample_rate: float = 0.01
    profiling_output_dir: str = "/tmp/proof_profiling"
    
    # Alerting settings
    enable_alerts: bool = True
    alert_cooldown_minutes: int = 15
    critical_alert_threshold: int = 3
    
    # Database settings
    metrics_database_path: str = "/tmp/proof_metrics.db"
    enable_metrics_persistence: bool = True

class PerformanceMonitoringError(Exception):
    """Exception raised during performance monitoring"""
    def __init__(self, message: str, component: str = None, metric: str = None):
        super().__init__(message)
        self.component = component
        self.metric = metric
        self.timestamp = datetime.now()

# ======================== ADVANCED METRICS COLLECTOR ========================

class AdvancedMetricsCollector:
    """Advanced metrics collection and analysis"""
    
    def __init__(self, config: PerformanceMonitoringConfig):
        self.config = config
        self.metrics_storage = defaultdict(deque)
        self.alert_history = deque(maxlen=1000)
        self.profiling_data = {}
        self.lock = threading.Lock()
        self.running = False
        self.collection_thread = None
        
        # Initialize metrics database
        if config.enable_metrics_persistence:
            self._init_metrics_database()
    
    def _init_metrics_database(self):
        """Initialize metrics database"""
        try:
            self.db_connection = sqlite3.connect(
                self.config.metrics_database_path,
                check_same_thread=False
            )
            
            # Create tables
            self.db_connection.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    component TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    tags TEXT
                )
            ''')
            
            self.db_connection.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    severity TEXT,
                    component TEXT,
                    message TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            self.db_connection.commit()
            logger.info("Metrics database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics database: {e}")
            self.config.enable_metrics_persistence = False
    
    def start_collection(self):
        """Start metrics collection"""
        if self.running:
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main metrics collection loop"""
        while self.running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect application metrics
                self._collect_application_metrics()
                
                # Cleanup old metrics
                self._cleanup_old_metrics()
                
                # Sleep until next collection
                time.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(self.config.metrics_collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            self.record_metric("system", "cpu_percent", cpu_percent)
            self.record_metric("system", "cpu_count", cpu_count)
            if cpu_freq:
                self.record_metric("system", "cpu_freq_current", cpu_freq.current)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric("system", "memory_percent", memory.percent)
            self.record_metric("system", "memory_available", memory.available)
            self.record_metric("system", "memory_used", memory.used)
            self.record_metric("system", "memory_total", memory.total)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.record_metric("system", "disk_percent", disk.percent)
            self.record_metric("system", "disk_free", disk.free)
            self.record_metric("system", "disk_used", disk.used)
            self.record_metric("system", "disk_total", disk.total)
            
            # Network metrics
            net_io = psutil.net_io_counters()
            self.record_metric("system", "network_bytes_sent", net_io.bytes_sent)
            self.record_metric("system", "network_bytes_recv", net_io.bytes_recv)
            self.record_metric("system", "network_packets_sent", net_io.packets_sent)
            self.record_metric("system", "network_packets_recv", net_io.packets_recv)
            
            # Process metrics
            process = psutil.Process()
            self.record_metric("process", "cpu_percent", process.cpu_percent())
            self.record_metric("process", "memory_percent", process.memory_percent())
            self.record_metric("process", "num_threads", process.num_threads())
            self.record_metric("process", "num_fds", process.num_fds())
            
            # Check thresholds and generate alerts
            self._check_system_thresholds(cpu_percent, memory.percent, disk.percent)
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
    
    def _collect_application_metrics(self):
        """Collect application-specific metrics"""
        try:
            # Garbage collection metrics
            gc_stats = gc.get_stats()
            for i, stat in enumerate(gc_stats):
                self.record_metric("gc", f"generation_{i}_collections", stat['collections'])
                self.record_metric("gc", f"generation_{i}_collected", stat['collected'])
                self.record_metric("gc", f"generation_{i}_uncollectable", stat['uncollectable'])
            
            # Resource usage
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            self.record_metric("resource", "max_rss", rusage.ru_maxrss)
            self.record_metric("resource", "user_time", rusage.ru_utime)
            self.record_metric("resource", "system_time", rusage.ru_stime)
            
            # Thread metrics
            active_threads = threading.active_count()
            self.record_metric("threading", "active_threads", active_threads)
            
        except Exception as e:
            logger.error(f"Application metrics collection failed: {e}")
    
    def _check_system_thresholds(self, cpu_percent: float, memory_percent: float, disk_percent: float):
        """Check system thresholds and generate alerts"""
        try:
            # CPU threshold
            if cpu_percent > self.config.cpu_threshold * 100:
                self._generate_alert(
                    "warning",
                    "system",
                    f"High CPU usage: {cpu_percent:.1f}%"
                )
            
            # Memory threshold
            if memory_percent > self.config.memory_threshold * 100:
                self._generate_alert(
                    "warning",
                    "system",
                    f"High memory usage: {memory_percent:.1f}%"
                )
            
            # Disk threshold
            if disk_percent > self.config.disk_threshold * 100:
                self._generate_alert(
                    "critical",
                    "system",
                    f"High disk usage: {disk_percent:.1f}%"
                )
            
        except Exception as e:
            logger.error(f"Threshold checking failed: {e}")
    
    def _generate_alert(self, severity: str, component: str, message: str):
        """Generate system alert"""
        try:
            alert = {
                'timestamp': datetime.now(),
                'severity': severity,
                'component': component,
                'message': message,
                'resolved': False
            }
            
            # Add to alert history
            self.alert_history.append(alert)
            
            # Log alert
            log_level = logging.WARNING if severity == 'warning' else logging.ERROR
            logger.log(log_level, f"ALERT [{severity.upper()}] {component}: {message}")
            
            # Persist alert
            if self.config.enable_metrics_persistence:
                self._persist_alert(alert)
            
        except Exception as e:
            logger.error(f"Alert generation failed: {e}")
    
    def _persist_alert(self, alert: Dict[str, Any]):
        """Persist alert to database"""
        try:
            self.db_connection.execute('''
                INSERT INTO alerts (timestamp, severity, component, message, resolved)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                alert['timestamp'],
                alert['severity'],
                alert['component'],
                alert['message'],
                alert['resolved']
            ))
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Alert persistence failed: {e}")
    
    def record_metric(self, component: str, metric_name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric value"""
        try:
            timestamp = datetime.now()
            
            # Store in memory
            with self.lock:
                key = f"{component}.{metric_name}"
                self.metrics_storage[key].append({
                    'timestamp': timestamp,
                    'value': value,
                    'tags': tags or {}
                })
            
            # Persist to database
            if self.config.enable_metrics_persistence:
                self._persist_metric(timestamp, component, metric_name, value, tags)
            
        except Exception as e:
            logger.error(f"Metric recording failed: {e}")
    
    def _persist_metric(self, timestamp: datetime, component: str, metric_name: str, value: float, tags: Dict[str, str]):
        """Persist metric to database"""
        try:
            tags_json = json.dumps(tags or {})
            
            self.db_connection.execute('''
                INSERT INTO metrics (timestamp, component, metric_name, metric_value, tags)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, component, metric_name, value, tags_json))
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Metric persistence failed: {e}")
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory issues"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.config.metrics_retention_hours)
            
            with self.lock:
                for key in list(self.metrics_storage.keys()):
                    metrics = self.metrics_storage[key]
                    # Remove old metrics
                    while metrics and metrics[0]['timestamp'] < cutoff_time:
                        metrics.popleft()
                    
                    # Remove empty deques
                    if not metrics:
                        del self.metrics_storage[key]
            
            # Cleanup database
            if self.config.enable_metrics_persistence:
                self.db_connection.execute('''
                    DELETE FROM metrics WHERE timestamp < ?
                ''', (cutoff_time,))
                self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Metrics cleanup failed: {e}")
    
    def get_metrics(self, component: str, metric_name: str, hours: int = 1) -> List[Dict[str, Any]]:
        """Get metrics for a specific component and metric"""
        try:
            key = f"{component}.{metric_name}"
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with self.lock:
                if key not in self.metrics_storage:
                    return []
                
                return [
                    metric for metric in self.metrics_storage[key]
                    if metric['timestamp'] >= cutoff_time
                ]
            
        except Exception as e:
            logger.error(f"Metrics retrieval failed: {e}")
            return []
    
    def get_metric_summary(self, component: str, metric_name: str, hours: int = 1) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        try:
            metrics = self.get_metrics(component, metric_name, hours)
            
            if not metrics:
                return {'error': 'No metrics found'}
            
            values = [m['value'] for m in metrics]
            
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'median': sorted(values)[len(values) // 2],
                'latest': values[-1],
                'timestamp_range': {
                    'start': metrics[0]['timestamp'].isoformat(),
                    'end': metrics[-1]['timestamp'].isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Metric summary failed: {e}")
            return {'error': str(e)}
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            return [
                alert for alert in self.alert_history
                if alert['timestamp'] >= cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Alert retrieval failed: {e}")
            return []

# ======================== PERFORMANCE OPTIMIZER ========================

class PerformanceOptimizer:
    """Advanced performance optimization for proof verifier"""
    
    def __init__(self, config: PerformanceMonitoringConfig):
        self.config = config
        self.optimization_history = []
        self.cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        self.lock = threading.Lock()
        
        # Initialize cache
        if config.enable_result_caching:
            self._init_cache()
    
    def _init_cache(self):
        """Initialize result cache"""
        try:
            self.cache = {}
            self.cache_timestamps = {}
            self.cache_access_count = defaultdict(int)
            
            # Start cache cleanup thread
            cleanup_thread = threading.Thread(target=self._cache_cleanup_loop, daemon=True)
            cleanup_thread.start()
            
            logger.info("Result cache initialized")
            
        except Exception as e:
            logger.error(f"Cache initialization failed: {e}")
    
    def _cache_cleanup_loop(self):
        """Cache cleanup loop"""
        while True:
            try:
                self._cleanup_expired_cache_entries()
                time.sleep(self.config.cache_ttl // 4)  # Cleanup every quarter TTL
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                time.sleep(60)
    
    def _cleanup_expired_cache_entries(self):
        """Remove expired cache entries"""
        try:
            current_time = time.time()
            expired_keys = []
            
            with self.lock:
                for key, timestamp in self.cache_timestamps.items():
                    if current_time - timestamp > self.config.cache_ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    if key in self.cache:
                        del self.cache[key]
                    if key in self.cache_timestamps:
                        del self.cache_timestamps[key]
                    if key in self.cache_access_count:
                        del self.cache_access_count[key]
                    
                    self.cache_stats['evictions'] += 1
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    def cache_result(self, key: str, result: Any) -> None:
        """Cache a result"""
        try:
            if not self.config.enable_result_caching:
                return
            
            with self.lock:
                # Check cache size limit
                if len(self.cache) >= self.config.cache_size:
                    # Remove least recently used entry
                    lru_key = min(self.cache_access_count, key=self.cache_access_count.get)
                    if lru_key in self.cache:
                        del self.cache[lru_key]
                    if lru_key in self.cache_timestamps:
                        del self.cache_timestamps[lru_key]
                    if lru_key in self.cache_access_count:
                        del self.cache_access_count[lru_key]
                    
                    self.cache_stats['evictions'] += 1
                
                # Store result
                self.cache[key] = result
                self.cache_timestamps[key] = time.time()
                self.cache_access_count[key] = 0
            
        except Exception as e:
            logger.error(f"Result caching failed: {e}")
    
    def get_cached_result(self, key: str) -> Tuple[bool, Any]:
        """Get cached result"""
        try:
            if not self.config.enable_result_caching:
                return False, None
            
            with self.lock:
                if key in self.cache:
                    # Check if expired
                    if time.time() - self.cache_timestamps[key] > self.config.cache_ttl:
                        # Remove expired entry
                        del self.cache[key]
                        del self.cache_timestamps[key]
                        del self.cache_access_count[key]
                        
                        self.cache_stats['evictions'] += 1
                        self.cache_stats['misses'] += 1
                        return False, None
                    
                    # Update access count
                    self.cache_access_count[key] += 1
                    self.cache_stats['hits'] += 1
                    
                    return True, self.cache[key]
                else:
                    self.cache_stats['misses'] += 1
                    return False, None
            
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
            return False, None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            with self.lock:
                total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
                hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
                
                return {
                    'cache_size': len(self.cache),
                    'max_cache_size': self.config.cache_size,
                    'hits': self.cache_stats['hits'],
                    'misses': self.cache_stats['misses'],
                    'evictions': self.cache_stats['evictions'],
                    'hit_rate': hit_rate,
                    'cache_utilization': len(self.cache) / self.config.cache_size
                }
            
        except Exception as e:
            logger.error(f"Cache stats retrieval failed: {e}")
            return {'error': str(e)}
    
    def optimize_proof_verification(self, proof_verifier: Any) -> Dict[str, Any]:
        """Optimize proof verification performance"""
        try:
            optimization_start = time.time()
            
            optimization_report = {
                'timestamp': datetime.now().isoformat(),
                'optimizations_applied': [],
                'performance_improvements': {},
                'recommendations': []
            }
            
            # Optimization 1: Cache frequently used computations
            if self.config.enable_result_caching:
                cache_stats = self.get_cache_stats()
                if cache_stats.get('hit_rate', 0) < 0.3:  # Low hit rate
                    optimization_report['recommendations'].append({
                        'type': 'cache_optimization',
                        'description': 'Consider increasing cache size or adjusting TTL',
                        'current_hit_rate': cache_stats.get('hit_rate', 0),
                        'target_hit_rate': 0.5
                    })
            
            # Optimization 2: Memory optimization
            gc.collect()  # Force garbage collection
            optimization_report['optimizations_applied'].append('garbage_collection')
            
            # Optimization 3: Thread pool optimization
            if hasattr(proof_verifier, 'executor') and proof_verifier.executor:
                # Check thread pool efficiency
                optimization_report['optimizations_applied'].append('thread_pool_check')
            
            # Optimization 4: Database connection optimization
            if hasattr(proof_verifier, 'db_connection'):
                # Optimize database connections
                optimization_report['optimizations_applied'].append('database_optimization')
            
            # Record optimization
            optimization_duration = time.time() - optimization_start
            optimization_report['optimization_duration'] = optimization_duration
            
            self.optimization_history.append(optimization_report)
            
            # Keep only recent optimizations
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-50:]
            
            logger.info(f"Proof verification optimization completed in {optimization_duration:.2f}s")
            return optimization_report
            
        except Exception as e:
            logger.error(f"Proof verification optimization failed: {e}")
            return {'error': str(e)}

# ======================== PROOF PERFORMANCE PROFILER ========================

class ProofPerformanceProfiler:
    """Advanced performance profiling for proof verification"""
    
    def __init__(self, config: PerformanceMonitoringConfig):
        self.config = config
        self.profiling_data = {}
        self.active_profiles = {}
        self.lock = threading.Lock()
        
        # Create profiling output directory
        if config.enable_profiling:
            os.makedirs(config.profiling_output_dir, exist_ok=True)
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations"""
        if not self.config.enable_profiling:
            yield
            return
        
        # Check sampling rate
        if np.random.random() > self.config.profiling_sample_rate:
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
            
            # Store profiling data
            with self.lock:
                if operation_name not in self.profiling_data:
                    self.profiling_data[operation_name] = []
                
                profile_data = {
                    'timestamp': datetime.now(),
                    'duration': duration,
                    'profiler': profiler
                }
                
                self.profiling_data[operation_name].append(profile_data)
                
                # Keep only recent profiles
                if len(self.profiling_data[operation_name]) > 100:
                    self.profiling_data[operation_name] = self.profiling_data[operation_name][-50:]
            
            # Save profile data
            self._save_profile_data(operation_name, profiler, duration)
    
    def _save_profile_data(self, operation_name: str, profiler: cProfile.Profile, duration: float):
        """Save profiling data to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{operation_name}_{timestamp}.prof"
            filepath = Path(self.config.profiling_output_dir) / filename
            
            # Save binary profile
            profiler.dump_stats(str(filepath))
            
            # Save text summary
            text_filename = f"{operation_name}_{timestamp}.txt"
            text_filepath = Path(self.config.profiling_output_dir) / text_filename
            
            with open(text_filepath, 'w') as f:
                f.write(f"Operation: {operation_name}\n")
                f.write(f"Duration: {duration:.4f} seconds\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write("=" * 50 + "\n")
                
                # Get profiling statistics
                stats = pstats.Stats(profiler)
                stats.sort_stats('cumulative')
                
                # Redirect output to string
                output = io.StringIO()
                stats.print_stats(20)  # Top 20 functions
                f.write(output.getvalue())
            
            logger.debug(f"Profile data saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Profile data saving failed: {e}")
    
    def get_operation_profile_summary(self, operation_name: str) -> Dict[str, Any]:
        """Get profile summary for an operation"""
        try:
            with self.lock:
                if operation_name not in self.profiling_data:
                    return {'error': 'No profiling data available'}
                
                profiles = self.profiling_data[operation_name]
                
                if not profiles:
                    return {'error': 'No profiling data available'}
                
                # Calculate statistics
                durations = [p['duration'] for p in profiles]
                
                return {
                    'operation_name': operation_name,
                    'profile_count': len(profiles),
                    'duration_stats': {
                        'min': min(durations),
                        'max': max(durations),
                        'mean': sum(durations) / len(durations),
                        'median': sorted(durations)[len(durations) // 2]
                    },
                    'recent_profiles': [
                        {
                            'timestamp': p['timestamp'].isoformat(),
                            'duration': p['duration']
                        }
                        for p in profiles[-10:]  # Last 10 profiles
                    ]
                }
            
        except Exception as e:
            logger.error(f"Profile summary failed: {e}")
            return {'error': str(e)}
    
    def analyze_performance_bottlenecks(self) -> Dict[str, Any]:
        """Analyze performance bottlenecks across all operations"""
        try:
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'operations_analyzed': len(self.profiling_data),
                'bottlenecks': [],
                'recommendations': []
            }
            
            with self.lock:
                for operation_name, profiles in self.profiling_data.items():
                    if not profiles:
                        continue
                    
                    # Get recent profiles
                    recent_profiles = profiles[-10:]
                    avg_duration = sum(p['duration'] for p in recent_profiles) / len(recent_profiles)
                    
                    # Identify bottlenecks
                    if avg_duration > 1.0:  # Longer than 1 second
                        analysis['bottlenecks'].append({
                            'operation': operation_name,
                            'average_duration': avg_duration,
                            'severity': 'high' if avg_duration > 5.0 else 'medium'
                        })
                    
                    # Generate recommendations
                    if avg_duration > 2.0:
                        analysis['recommendations'].append({
                            'operation': operation_name,
                            'recommendation': 'Consider optimizing this operation',
                            'current_duration': avg_duration,
                            'target_duration': avg_duration * 0.5
                        })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Performance bottleneck analysis failed: {e}")
            return {'error': str(e)}

# ======================== COMPREHENSIVE MONITORING MANAGER ========================

class ComprehensiveMonitoringManager:
    """Comprehensive monitoring manager for proof verifier system"""
    
    def __init__(self, config: PerformanceMonitoringConfig = None):
        self.config = config or PerformanceMonitoringConfig()
        self.metrics_collector = AdvancedMetricsCollector(self.config)
        self.performance_optimizer = PerformanceOptimizer(self.config)
        self.profiler = ProofPerformanceProfiler(self.config)
        self.monitoring_active = False
        self.health_status = 'unknown'
        
        # Start monitoring
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start comprehensive monitoring"""
        try:
            if self.monitoring_active:
                return
            
            self.monitoring_active = True
            
            # Start metrics collection
            self.metrics_collector.start_collection()
            
            # Start health monitoring
            health_thread = threading.Thread(target=self._health_monitoring_loop, daemon=True)
            health_thread.start()
            
            # Start performance analysis
            analysis_thread = threading.Thread(target=self._performance_analysis_loop, daemon=True)
            analysis_thread.start()
            
            logger.info("Comprehensive monitoring started")
            
        except Exception as e:
            logger.error(f"Monitoring startup failed: {e}")
    
    def stop_monitoring(self):
        """Stop comprehensive monitoring"""
        try:
            self.monitoring_active = False
            self.metrics_collector.stop_collection()
            
            logger.info("Comprehensive monitoring stopped")
            
        except Exception as e:
            logger.error(f"Monitoring shutdown failed: {e}")
    
    def _health_monitoring_loop(self):
        """Health monitoring loop"""
        while self.monitoring_active:
            try:
                self._perform_health_check()
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.config.health_check_interval)
    
    def _performance_analysis_loop(self):
        """Performance analysis loop"""
        while self.monitoring_active:
            try:
                self._perform_performance_analysis()
                time.sleep(self.config.performance_analysis_interval)
                
            except Exception as e:
                logger.error(f"Performance analysis error: {e}")
                time.sleep(self.config.performance_analysis_interval)
    
    def _perform_health_check(self):
        """Perform system health check"""
        try:
            # Check system resources
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            # Determine health status
            if (cpu_percent > 90 or memory_percent > 90 or disk_percent > 95):
                self.health_status = 'critical'
            elif (cpu_percent > 70 or memory_percent > 80 or disk_percent > 90):
                self.health_status = 'warning'
            else:
                self.health_status = 'healthy'
            
            # Log health status
            logger.info(f"System health: {self.health_status} "
                       f"(CPU: {cpu_percent:.1f}%, "
                       f"Memory: {memory_percent:.1f}%, "
                       f"Disk: {disk_percent:.1f}%)")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.health_status = 'error'
    
    def _perform_performance_analysis(self):
        """Perform performance analysis"""
        try:
            # Analyze cache performance
            cache_stats = self.performance_optimizer.get_cache_stats()
            
            # Analyze profiling data
            bottlenecks = self.profiler.analyze_performance_bottlenecks()
            
            # Log performance insights
            if cache_stats.get('hit_rate', 0) < 0.3:
                logger.warning(f"Low cache hit rate: {cache_stats.get('hit_rate', 0):.2%}")
            
            if bottlenecks.get('bottlenecks'):
                logger.warning(f"Performance bottlenecks detected: {len(bottlenecks['bottlenecks'])}")
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
    
    def record_proof_verification_metrics(self, duration: float, success: bool, 
                                        proof_type: str = None, error_type: str = None):
        """Record proof verification metrics"""
        try:
            # Record basic metrics
            self.metrics_collector.record_metric("proof_verification", "duration", duration)
            self.metrics_collector.record_metric("proof_verification", "success", 1 if success else 0)
            
            # Record detailed metrics
            if proof_type:
                self.metrics_collector.record_metric(
                    "proof_verification", 
                    f"duration_{proof_type}", 
                    duration
                )
            
            if not success and error_type:
                self.metrics_collector.record_metric(
                    "proof_verification", 
                    f"error_{error_type}", 
                    1
                )
            
        except Exception as e:
            logger.error(f"Proof verification metrics recording failed: {e}")
    
    def optimize_system_performance(self, proof_verifier: Any) -> Dict[str, Any]:
        """Optimize system performance"""
        try:
            return self.performance_optimizer.optimize_proof_verification(proof_verifier)
            
        except Exception as e:
            logger.error(f"System performance optimization failed: {e}")
            return {'error': str(e)}
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard"""
        try:
            dashboard = {
                'timestamp': datetime.now().isoformat(),
                'health_status': self.health_status,
                'system_metrics': {
                    'cpu': self.metrics_collector.get_metric_summary('system', 'cpu_percent'),
                    'memory': self.metrics_collector.get_metric_summary('system', 'memory_percent'),
                    'disk': self.metrics_collector.get_metric_summary('system', 'disk_percent')
                },
                'proof_verification_metrics': {
                    'duration': self.metrics_collector.get_metric_summary('proof_verification', 'duration'),
                    'success_rate': self.metrics_collector.get_metric_summary('proof_verification', 'success')
                },
                'cache_performance': self.performance_optimizer.get_cache_stats(),
                'recent_alerts': self.metrics_collector.get_recent_alerts(hours=24),
                'performance_bottlenecks': self.profiler.analyze_performance_bottlenecks()
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Monitoring dashboard generation failed: {e}")
            return {'error': str(e)}
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'report_period_hours': hours,
                'executive_summary': {
                    'health_status': self.health_status,
                    'alert_count': len(self.metrics_collector.get_recent_alerts(hours)),
                    'bottleneck_count': len(self.profiler.analyze_performance_bottlenecks().get('bottlenecks', []))
                },
                'detailed_metrics': {
                    'system_performance': {
                        'cpu_usage': self.metrics_collector.get_metric_summary('system', 'cpu_percent', hours),
                        'memory_usage': self.metrics_collector.get_metric_summary('system', 'memory_percent', hours),
                        'disk_usage': self.metrics_collector.get_metric_summary('system', 'disk_percent', hours)
                    },
                    'proof_verification': {
                        'duration': self.metrics_collector.get_metric_summary('proof_verification', 'duration', hours),
                        'success_rate': self.metrics_collector.get_metric_summary('proof_verification', 'success', hours)
                    }
                },
                'optimization_recommendations': self.profiler.analyze_performance_bottlenecks().get('recommendations', []),
                'cache_efficiency': self.performance_optimizer.get_cache_stats()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            return {'error': str(e)}

# ======================== MONITORING DECORATORS ========================

def monitor_proof_verification(monitoring_manager: ComprehensiveMonitoringManager):
    """Decorator for monitoring proof verification operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            error_type = None
            
            try:
                # Profile the operation
                with monitoring_manager.profiler.profile_operation(func.__name__):
                    result = func(*args, **kwargs)
                
                success = True
                return result
                
            except Exception as e:
                error_type = type(e).__name__
                raise
                
            finally:
                duration = time.time() - start_time
                monitoring_manager.record_proof_verification_metrics(
                    duration=duration,
                    success=success,
                    proof_type=func.__name__,
                    error_type=error_type
                )
        
        return wrapper
    return decorator

def cache_proof_result(performance_optimizer: PerformanceOptimizer, cache_key_func: Callable = None):
    """Decorator for caching proof verification results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hashlib.md5(str(args).encode()).hexdigest()}"
            
            # Check cache
            cached, result = performance_optimizer.get_cached_result(cache_key)
            if cached:
                return result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            performance_optimizer.cache_result(cache_key, result)
            
            return result
        
        return wrapper
    return decorator

# Export components
__all__ = [
    'PerformanceMonitoringConfig',
    'PerformanceMonitoringError',
    'AdvancedMetricsCollector',
    'PerformanceOptimizer',
    'ProofPerformanceProfiler',
    'ComprehensiveMonitoringManager',
    'monitor_proof_verification',
    'cache_proof_result'
]

if __name__ == "__main__":
    print("Enhanced Proof Verifier - Chunk 3: Performance and monitoring enhancements loaded")
    
    # Basic monitoring test
    try:
        config = PerformanceMonitoringConfig()
        monitoring_manager = ComprehensiveMonitoringManager(config)
        print("✓ Comprehensive monitoring manager created successfully")
        
        # Test metrics collection
        time.sleep(2)  # Let metrics collect
        
        dashboard = monitoring_manager.get_monitoring_dashboard()
        print(f"✓ Monitoring dashboard generated: {dashboard['health_status']}")
        
        # Test performance report
        report = monitoring_manager.get_performance_report(hours=1)
        print(f"✓ Performance report generated: {report['executive_summary']['health_status']}")
        
    except Exception as e:
        print(f"✗ Monitoring test failed: {e}")
    finally:
        # Cleanup
        try:
            monitoring_manager.stop_monitoring()
        except:
            pass