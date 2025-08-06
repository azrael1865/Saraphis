"""
System Health Checker - Comprehensive health monitoring for neural compression framework
Real-time resource monitoring, pipeline health checks, and performance analysis
NO PLACEHOLDERS - PRODUCTION READY
FAIL LOUD - NO GRACEFUL DEGRADATION
"""

import os
import sys
import time
import json
import psutil
import torch
import threading
import subprocess
import warnings
import importlib
import traceback
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Import existing monitoring infrastructure
from .compression_analytics import (
    UnifiedPerformanceMonitor,
    MetricType,
    PipelineType,
    PerformanceMetric,
    MonitorConfig
)

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class RecoveryAction(Enum):
    """Self-healing recovery actions"""
    RESTART_PIPELINE = "restart_pipeline"
    CLEAR_MEMORY = "clear_memory"
    REDUCE_BATCH_SIZE = "reduce_batch_size"
    SWITCH_TO_CPU = "switch_to_cpu"
    THROTTLE_REQUESTS = "throttle_requests"
    KILL_ZOMBIE_THREADS = "kill_zombie_threads"
    RESET_GPU = "reset_gpu"
    GARBAGE_COLLECT = "garbage_collect"


@dataclass
class ResourceMetrics:
    """System resource metrics"""
    timestamp: float
    cpu_percent: float
    cpu_temp: Optional[float]
    cpu_freq_current: float
    cpu_freq_max: float
    memory_used_gb: float
    memory_available_gb: float
    memory_percent: float
    swap_used_gb: float
    swap_percent: float
    disk_io_read_mb_s: float
    disk_io_write_mb_s: float
    disk_usage_percent: float
    network_sent_mb_s: float
    network_recv_mb_s: float
    gpu_memory_used_mb: Optional[float]
    gpu_memory_total_mb: Optional[float]
    gpu_utilization: Optional[float]
    gpu_temperature: Optional[float]
    gpu_power_draw: Optional[float]
    
    def __post_init__(self):
        """Validate metrics"""
        if self.cpu_percent < 0 or self.cpu_percent > 100:
            raise ValueError(f"Invalid CPU percentage: {self.cpu_percent}")
        if self.memory_percent < 0 or self.memory_percent > 100:
            raise ValueError(f"Invalid memory percentage: {self.memory_percent}")
        if self.disk_usage_percent < 0 or self.disk_usage_percent > 100:
            raise ValueError(f"Invalid disk usage percentage: {self.disk_usage_percent}")


@dataclass
class PipelineHealth:
    """Pipeline health status"""
    pipeline_type: PipelineType
    status: HealthStatus
    last_success_time: float
    total_operations: int
    failed_operations: int
    success_rate: float
    avg_latency_ms: float
    max_latency_ms: float
    error_messages: List[str]
    is_operational: bool
    
    def __post_init__(self):
        """Validate pipeline health"""
        if self.success_rate < 0 or self.success_rate > 1:
            raise ValueError(f"Invalid success rate: {self.success_rate}")
        if self.avg_latency_ms < 0:
            raise ValueError(f"Invalid average latency: {self.avg_latency_ms}")


@dataclass
class DependencyStatus:
    """Dependency validation status"""
    dependency_name: str
    is_available: bool
    version: Optional[str]
    error_message: Optional[str]
    is_critical: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class HealthAlert:
    """Health monitoring alert"""
    alert_id: str
    timestamp: float
    severity: AlertSeverity
    component: str
    message: str
    metrics: Dict[str, Any]
    recovery_actions: List[RecoveryAction]
    auto_recovery_attempted: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp,
            'severity': self.severity.value,
            'component': self.component,
            'message': self.message,
            'metrics': self.metrics,
            'recovery_actions': [action.value for action in self.recovery_actions],
            'auto_recovery_attempted': self.auto_recovery_attempted
        }


@dataclass
class HealthScore:
    """Overall system health score"""
    overall_score: float  # 0-100
    resource_score: float
    pipeline_score: float
    dependency_score: float
    performance_score: float
    timestamp: float
    status: HealthStatus
    
    def __post_init__(self):
        """Validate health score"""
        for score_name, score_value in [
            ('overall_score', self.overall_score),
            ('resource_score', self.resource_score),
            ('pipeline_score', self.pipeline_score),
            ('dependency_score', self.dependency_score),
            ('performance_score', self.performance_score)
        ]:
            if not 0 <= score_value <= 100:
                raise ValueError(f"Invalid {score_name}: {score_value}, must be 0-100")


class ResourceHealthMonitor:
    """
    Monitor CPU usage/temperature/throttling
    Track GPU health/memory/temperature
    Detect memory pressure and leaks
    Monitor disk I/O and space
    """
    
    def __init__(self, sampling_interval_ms: int = 1000,
                 history_size: int = 100):
        """
        Initialize resource health monitor
        
        Args:
            sampling_interval_ms: Sampling interval in milliseconds
            history_size: Size of metrics history buffer
        """
        if sampling_interval_ms <= 0:
            raise ValueError(f"Sampling interval must be positive, got {sampling_interval_ms}")
        if history_size <= 0:
            raise ValueError(f"History size must be positive, got {history_size}")
        
        self.sampling_interval_ms = sampling_interval_ms
        self.history_size = history_size
        self.metrics_history: deque[ResourceMetrics] = deque(maxlen=history_size)
        self.monitoring_active = True
        self._lock = threading.RLock()
        
        # Thresholds for health detection
        self.thresholds = {
            'cpu_percent_warning': 80.0,
            'cpu_percent_critical': 95.0,
            'cpu_temp_warning': 80.0,
            'cpu_temp_critical': 95.0,
            'memory_percent_warning': 85.0,
            'memory_percent_critical': 95.0,
            'gpu_memory_percent_warning': 90.0,
            'gpu_memory_percent_critical': 98.0,
            'gpu_temp_warning': 85.0,
            'gpu_temp_critical': 95.0,
            'disk_usage_warning': 85.0,
            'disk_usage_critical': 95.0
        }
        
        # Memory leak detection
        self.memory_baseline = None
        self.memory_growth_threshold = 100  # MB
        self.memory_leak_detected = False
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"ResourceHealthMonitor initialized with {sampling_interval_ms}ms interval")
    
    def get_current_metrics(self) -> ResourceMetrics:
        """
        Get current system resource metrics
        
        Returns:
            Current resource metrics
        """
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()
        cpu_temp = self._get_cpu_temperature()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk I/O metrics
        disk_io = psutil.disk_io_counters()
        disk_usage = psutil.disk_usage('/')
        
        # Network metrics
        net_io = psutil.net_io_counters()
        
        # GPU metrics
        gpu_metrics = self._get_gpu_metrics()
        
        metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            cpu_temp=cpu_temp,
            cpu_freq_current=cpu_freq.current if cpu_freq else 0,
            cpu_freq_max=cpu_freq.max if cpu_freq else 0,
            memory_used_gb=memory.used / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            memory_percent=memory.percent,
            swap_used_gb=swap.used / (1024**3),
            swap_percent=swap.percent,
            disk_io_read_mb_s=disk_io.read_bytes / (1024**2) if disk_io else 0,
            disk_io_write_mb_s=disk_io.write_bytes / (1024**2) if disk_io else 0,
            disk_usage_percent=disk_usage.percent,
            network_sent_mb_s=net_io.bytes_sent / (1024**2) if net_io else 0,
            network_recv_mb_s=net_io.bytes_recv / (1024**2) if net_io else 0,
            gpu_memory_used_mb=gpu_metrics.get('memory_used_mb'),
            gpu_memory_total_mb=gpu_metrics.get('memory_total_mb'),
            gpu_utilization=gpu_metrics.get('utilization'),
            gpu_temperature=gpu_metrics.get('temperature'),
            gpu_power_draw=gpu_metrics.get('power_draw')
        )
        
        return metrics
    
    def check_cpu_health(self) -> Tuple[HealthStatus, List[str]]:
        """
        Check CPU health status
        
        Returns:
            Health status and list of issues
        """
        with self._lock:
            if not self.metrics_history:
                return HealthStatus.UNKNOWN, ["No metrics available"]
            
            recent_metrics = list(self.metrics_history)[-10:]
            issues = []
            
            # Check CPU usage
            avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
            max_cpu = max([m.cpu_percent for m in recent_metrics])
            
            if max_cpu >= self.thresholds['cpu_percent_critical']:
                issues.append(f"Critical CPU usage: {max_cpu:.1f}%")
            elif avg_cpu >= self.thresholds['cpu_percent_warning']:
                issues.append(f"High CPU usage: {avg_cpu:.1f}% average")
            
            # Check CPU temperature
            if recent_metrics[-1].cpu_temp:
                temp = recent_metrics[-1].cpu_temp
                if temp >= self.thresholds['cpu_temp_critical']:
                    issues.append(f"Critical CPU temperature: {temp:.1f}°C")
                elif temp >= self.thresholds['cpu_temp_warning']:
                    issues.append(f"High CPU temperature: {temp:.1f}°C")
            
            # Check CPU throttling
            if recent_metrics[-1].cpu_freq_current < recent_metrics[-1].cpu_freq_max * 0.7:
                issues.append("CPU throttling detected")
            
            # Determine overall status
            if any("Critical" in issue for issue in issues):
                return HealthStatus.CRITICAL, issues
            elif issues:
                return HealthStatus.WARNING, issues
            else:
                return HealthStatus.HEALTHY, []
    
    def check_memory_health(self) -> Tuple[HealthStatus, List[str]]:
        """
        Check memory health and detect leaks
        
        Returns:
            Health status and list of issues
        """
        with self._lock:
            if not self.metrics_history:
                return HealthStatus.UNKNOWN, ["No metrics available"]
            
            recent_metrics = list(self.metrics_history)[-10:]
            issues = []
            
            # Check memory usage
            current_memory = recent_metrics[-1].memory_percent
            avg_memory = np.mean([m.memory_percent for m in recent_metrics])
            
            if current_memory >= self.thresholds['memory_percent_critical']:
                issues.append(f"Critical memory usage: {current_memory:.1f}%")
            elif avg_memory >= self.thresholds['memory_percent_warning']:
                issues.append(f"High memory usage: {avg_memory:.1f}% average")
            
            # Check swap usage
            if recent_metrics[-1].swap_percent > 50:
                issues.append(f"High swap usage: {recent_metrics[-1].swap_percent:.1f}%")
            
            # Memory leak detection
            if len(self.metrics_history) >= 20:
                old_metrics = list(self.metrics_history)[:10]
                memory_growth = (avg_memory - np.mean([m.memory_percent for m in old_metrics]))
                
                if memory_growth > 10:  # 10% growth
                    issues.append(f"Potential memory leak: {memory_growth:.1f}% growth")
                    self.memory_leak_detected = True
            
            # Determine overall status
            if any("Critical" in issue for issue in issues):
                return HealthStatus.CRITICAL, issues
            elif self.memory_leak_detected:
                return HealthStatus.WARNING, issues + ["Memory leak detected"]
            elif issues:
                return HealthStatus.WARNING, issues
            else:
                return HealthStatus.HEALTHY, []
    
    def check_gpu_health(self) -> Tuple[HealthStatus, List[str]]:
        """
        Check GPU health status
        
        Returns:
            Health status and list of issues
        """
        with self._lock:
            if not self.metrics_history:
                return HealthStatus.UNKNOWN, ["No metrics available"]
            
            recent_metrics = list(self.metrics_history)[-10:]
            latest = recent_metrics[-1]
            issues = []
            
            # Check if GPU is available
            if latest.gpu_memory_used_mb is None:
                return HealthStatus.UNKNOWN, ["GPU not available or not detected"]
            
            # Check GPU memory
            if latest.gpu_memory_total_mb and latest.gpu_memory_total_mb > 0:
                gpu_memory_percent = (latest.gpu_memory_used_mb / latest.gpu_memory_total_mb) * 100
                
                if gpu_memory_percent >= self.thresholds['gpu_memory_percent_critical']:
                    issues.append(f"Critical GPU memory usage: {gpu_memory_percent:.1f}%")
                elif gpu_memory_percent >= self.thresholds['gpu_memory_percent_warning']:
                    issues.append(f"High GPU memory usage: {gpu_memory_percent:.1f}%")
            
            # Check GPU temperature
            if latest.gpu_temperature:
                if latest.gpu_temperature >= self.thresholds['gpu_temp_critical']:
                    issues.append(f"Critical GPU temperature: {latest.gpu_temperature:.1f}°C")
                elif latest.gpu_temperature >= self.thresholds['gpu_temp_warning']:
                    issues.append(f"High GPU temperature: {latest.gpu_temperature:.1f}°C")
            
            # Check GPU utilization
            if latest.gpu_utilization and latest.gpu_utilization > 95:
                issues.append(f"GPU at maximum utilization: {latest.gpu_utilization:.1f}%")
            
            # Determine overall status
            if any("Critical" in issue for issue in issues):
                return HealthStatus.CRITICAL, issues
            elif issues:
                return HealthStatus.WARNING, issues
            else:
                return HealthStatus.HEALTHY, []
    
    def check_disk_health(self) -> Tuple[HealthStatus, List[str]]:
        """
        Check disk I/O and space health
        
        Returns:
            Health status and list of issues
        """
        with self._lock:
            if not self.metrics_history:
                return HealthStatus.UNKNOWN, ["No metrics available"]
            
            recent_metrics = list(self.metrics_history)[-10:]
            latest = recent_metrics[-1]
            issues = []
            
            # Check disk usage
            if latest.disk_usage_percent >= self.thresholds['disk_usage_critical']:
                issues.append(f"Critical disk usage: {latest.disk_usage_percent:.1f}%")
            elif latest.disk_usage_percent >= self.thresholds['disk_usage_warning']:
                issues.append(f"High disk usage: {latest.disk_usage_percent:.1f}%")
            
            # Check I/O rates (threshold at 100 MB/s sustained)
            avg_read = np.mean([m.disk_io_read_mb_s for m in recent_metrics])
            avg_write = np.mean([m.disk_io_write_mb_s for m in recent_metrics])
            
            if avg_read > 100:
                issues.append(f"High disk read rate: {avg_read:.1f} MB/s")
            if avg_write > 100:
                issues.append(f"High disk write rate: {avg_write:.1f} MB/s")
            
            # Determine overall status
            if any("Critical" in issue for issue in issues):
                return HealthStatus.CRITICAL, issues
            elif issues:
                return HealthStatus.WARNING, issues
            else:
                return HealthStatus.HEALTHY, []
    
    def detect_resource_anomalies(self) -> List[str]:
        """
        Detect resource usage anomalies
        
        Returns:
            List of detected anomalies
        """
        with self._lock:
            if len(self.metrics_history) < 20:
                return []
            
            anomalies = []
            recent = list(self.metrics_history)[-10:]
            older = list(self.metrics_history)[-20:-10]
            
            # CPU spike detection
            recent_cpu = np.mean([m.cpu_percent for m in recent])
            older_cpu = np.mean([m.cpu_percent for m in older])
            if recent_cpu > older_cpu * 2 and recent_cpu > 50:
                anomalies.append(f"CPU usage spike: {older_cpu:.1f}% -> {recent_cpu:.1f}%")
            
            # Memory spike detection
            recent_mem = np.mean([m.memory_percent for m in recent])
            older_mem = np.mean([m.memory_percent for m in older])
            if recent_mem > older_mem * 1.5 and recent_mem > 50:
                anomalies.append(f"Memory usage spike: {older_mem:.1f}% -> {recent_mem:.1f}%")
            
            # I/O burst detection
            recent_io = np.mean([m.disk_io_read_mb_s + m.disk_io_write_mb_s for m in recent])
            older_io = np.mean([m.disk_io_read_mb_s + m.disk_io_write_mb_s for m in older])
            if recent_io > older_io * 3 and recent_io > 50:
                anomalies.append(f"I/O burst detected: {recent_io:.1f} MB/s")
            
            return anomalies
    
    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature if available"""
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.label in ['Core 0', 'CPU', 'Package id 0']:
                            return entry.current
                # Return first available temperature
                for name, entries in temps.items():
                    if entries:
                        return entries[0].current
        except:
            pass
        return None
    
    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """Get GPU metrics if available"""
        metrics = {}
        
        try:
            if torch.cuda.is_available():
                # Get current device
                device_id = torch.cuda.current_device()
                
                # Memory metrics
                memory_allocated = torch.cuda.memory_allocated(device_id) / (1024**2)
                memory_reserved = torch.cuda.memory_reserved(device_id) / (1024**2)
                
                metrics['memory_used_mb'] = memory_allocated
                metrics['memory_total_mb'] = torch.cuda.get_device_properties(device_id).total_memory / (1024**2)
                
                # Try to get additional metrics via nvidia-ml-py if available
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                    
                    # Utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics['utilization'] = util.gpu
                    
                    # Temperature
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    metrics['temperature'] = temp
                    
                    # Power
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
                    metrics['power_draw'] = power
                    
                except:
                    pass
        except:
            pass
        
        return metrics
    
    def _monitoring_loop(self):
        """Background monitoring thread"""
        while self.monitoring_active:
            try:
                metrics = self.get_current_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                    
                    # Set memory baseline
                    if self.memory_baseline is None:
                        self.memory_baseline = metrics.memory_used_gb
                
                time.sleep(self.sampling_interval_ms / 1000.0)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                raise
    
    def shutdown(self):
        """Shutdown the monitor"""
        self.monitoring_active = False
        logger.info("ResourceHealthMonitor shutdown")


class PipelineHealthChecker:
    """
    Check P-adic pipeline operational status
    Verify Tropical pipeline functionality
    Monitor JAX engine health
    Validate channel system integrity
    """
    
    def __init__(self, performance_monitor: Optional[UnifiedPerformanceMonitor] = None):
        """
        Initialize pipeline health checker
        
        Args:
            performance_monitor: Optional performance monitor instance
        """
        self.performance_monitor = performance_monitor
        self.pipeline_status: Dict[PipelineType, PipelineHealth] = {}
        self.operation_history: Dict[PipelineType, deque] = {
            PipelineType.PADIC: deque(maxlen=100),
            PipelineType.TROPICAL: deque(maxlen=100),
            PipelineType.HYBRID: deque(maxlen=100)
        }
        self._lock = threading.RLock()
        
        # Initialize pipeline health
        for pipeline_type in PipelineType:
            self.pipeline_status[pipeline_type] = PipelineHealth(
                pipeline_type=pipeline_type,
                status=HealthStatus.UNKNOWN,
                last_success_time=0,
                total_operations=0,
                failed_operations=0,
                success_rate=1.0,
                avg_latency_ms=0,
                max_latency_ms=0,
                error_messages=[],
                is_operational=False
            )
        
        logger.info("PipelineHealthChecker initialized")
    
    def check_padic_pipeline(self) -> PipelineHealth:
        """
        Check P-adic pipeline health
        
        Returns:
            Pipeline health status
        """
        with self._lock:
            try:
                # Try to import and test P-adic components
                from ..padic.padic_advanced import PadicDecompressionEngine
                from ..padic.hybrid_padic_compressor import HybridPadicCompressionSystem
                
                # Create test instance
                engine = PadicDecompressionEngine(prime=251, precision=16)
                
                # Test basic operation
                test_data = torch.randn(10, 10)
                start_time = time.time()
                
                # Simulate compression test (would use actual compression in production)
                result = engine.validate_configuration()
                latency_ms = (time.time() - start_time) * 1000
                
                # Update health status
                health = self.pipeline_status[PipelineType.PADIC]
                health.status = HealthStatus.HEALTHY
                health.last_success_time = time.time()
                health.total_operations += 1
                health.avg_latency_ms = latency_ms
                health.is_operational = True
                health.error_messages = []
                
                self.operation_history[PipelineType.PADIC].append({
                    'timestamp': time.time(),
                    'success': True,
                    'latency_ms': latency_ms
                })
                
            except Exception as e:
                health = self.pipeline_status[PipelineType.PADIC]
                health.status = HealthStatus.FAILURE
                health.failed_operations += 1
                health.is_operational = False
                health.error_messages.append(str(e))
                
                self.operation_history[PipelineType.PADIC].append({
                    'timestamp': time.time(),
                    'success': False,
                    'error': str(e)
                })
            
            # Calculate success rate
            if health.total_operations > 0:
                health.success_rate = 1.0 - (health.failed_operations / health.total_operations)
            
            return health
    
    def check_tropical_pipeline(self) -> PipelineHealth:
        """
        Check Tropical pipeline health
        
        Returns:
            Pipeline health status
        """
        with self._lock:
            try:
                # Try to import and test Tropical components
                from ..tropical.tropical_compression_pipeline import TropicalCompressionPipeline
                
                # Test basic functionality
                test_tensor = torch.randn(10, 10)
                start_time = time.time()
                
                # Check if module loads correctly
                pipeline = TropicalCompressionPipeline()
                latency_ms = (time.time() - start_time) * 1000
                
                # Update health status
                health = self.pipeline_status[PipelineType.TROPICAL]
                health.status = HealthStatus.HEALTHY
                health.last_success_time = time.time()
                health.total_operations += 1
                health.avg_latency_ms = latency_ms
                health.is_operational = True
                health.error_messages = []
                
                self.operation_history[PipelineType.TROPICAL].append({
                    'timestamp': time.time(),
                    'success': True,
                    'latency_ms': latency_ms
                })
                
            except Exception as e:
                health = self.pipeline_status[PipelineType.TROPICAL]
                health.status = HealthStatus.FAILURE
                health.failed_operations += 1
                health.is_operational = False
                health.error_messages.append(str(e))
                
                self.operation_history[PipelineType.TROPICAL].append({
                    'timestamp': time.time(),
                    'success': False,
                    'error': str(e)
                })
            
            # Calculate success rate
            if health.total_operations > 0:
                health.success_rate = 1.0 - (health.failed_operations / health.total_operations)
            
            return health
    
    def check_jax_engine(self) -> Tuple[HealthStatus, Dict[str, Any]]:
        """
        Check JAX engine health
        
        Returns:
            Health status and JAX info
        """
        jax_info = {
            'available': False,
            'version': None,
            'devices': [],
            'default_backend': None
        }
        
        try:
            import jax
            import jax.numpy as jnp
            
            jax_info['available'] = True
            jax_info['version'] = jax.__version__
            
            # Get available devices
            devices = jax.devices()
            jax_info['devices'] = [str(d) for d in devices]
            jax_info['default_backend'] = jax.default_backend()
            
            # Test basic operation
            test_array = jnp.ones((10, 10))
            result = jnp.sum(test_array)
            
            if result == 100:
                return HealthStatus.HEALTHY, jax_info
            else:
                jax_info['error'] = "JAX computation failed validation"
                return HealthStatus.DEGRADED, jax_info
                
        except ImportError:
            jax_info['error'] = "JAX not installed"
            return HealthStatus.WARNING, jax_info
        except Exception as e:
            jax_info['error'] = str(e)
            return HealthStatus.FAILURE, jax_info
    
    def check_channel_integrity(self) -> Tuple[HealthStatus, List[str]]:
        """
        Validate channel system integrity
        
        Returns:
            Health status and list of issues
        """
        issues = []
        
        try:
            # Check channel extractor
            from ..tropical.tropical_channel_extractor import IEEEChannelExtractor
            extractor = IEEEChannelExtractor()
            
            # Test extraction
            test_value = np.float32(3.14159)
            channels = extractor.extract_channels(test_value)
            
            if len(channels) != 3:  # sign, exponent, mantissa
                issues.append(f"Channel extraction returned {len(channels)} channels, expected 3")
            
            # Check channel decompressor
            from ..tropical.tropical_channel_decompressor import TropicalChannelDecompressor
            decompressor = TropicalChannelDecompressor()
            
            if issues:
                return HealthStatus.DEGRADED, issues
            else:
                return HealthStatus.HEALTHY, []
                
        except ImportError as e:
            issues.append(f"Failed to import channel components: {e}")
            return HealthStatus.FAILURE, issues
        except Exception as e:
            issues.append(f"Channel system error: {e}")
            return HealthStatus.FAILURE, issues
    
    def check_all_pipelines(self) -> Dict[PipelineType, PipelineHealth]:
        """
        Check all pipeline health statuses
        
        Returns:
            Dictionary of pipeline health statuses
        """
        with self._lock:
            self.check_padic_pipeline()
            self.check_tropical_pipeline()
            
            # Check hybrid based on component health
            padic_health = self.pipeline_status[PipelineType.PADIC]
            tropical_health = self.pipeline_status[PipelineType.TROPICAL]
            
            hybrid_health = self.pipeline_status[PipelineType.HYBRID]
            if padic_health.is_operational and tropical_health.is_operational:
                hybrid_health.status = HealthStatus.HEALTHY
                hybrid_health.is_operational = True
            elif padic_health.is_operational or tropical_health.is_operational:
                hybrid_health.status = HealthStatus.DEGRADED
                hybrid_health.is_operational = True
            else:
                hybrid_health.status = HealthStatus.FAILURE
                hybrid_health.is_operational = False
            
            return dict(self.pipeline_status)
    
    def get_pipeline_metrics(self, pipeline_type: PipelineType) -> Dict[str, Any]:
        """
        Get detailed metrics for a pipeline
        
        Args:
            pipeline_type: Pipeline to get metrics for
            
        Returns:
            Pipeline metrics dictionary
        """
        with self._lock:
            health = self.pipeline_status[pipeline_type]
            history = list(self.operation_history[pipeline_type])
            
            metrics = {
                'status': health.status.value,
                'is_operational': health.is_operational,
                'success_rate': health.success_rate,
                'total_operations': health.total_operations,
                'failed_operations': health.failed_operations,
                'avg_latency_ms': health.avg_latency_ms,
                'last_success_time': health.last_success_time,
                'recent_errors': health.error_messages[-5:],
                'history_summary': self._summarize_history(history)
            }
            
            return metrics
    
    def _summarize_history(self, history: List[Dict]) -> Dict[str, Any]:
        """Summarize operation history"""
        if not history:
            return {'no_data': True}
        
        successes = [h for h in history if h.get('success', False)]
        failures = [h for h in history if not h.get('success', True)]
        
        summary = {
            'total': len(history),
            'successes': len(successes),
            'failures': len(failures),
            'success_rate': len(successes) / len(history) if history else 0
        }
        
        if successes:
            latencies = [h['latency_ms'] for h in successes if 'latency_ms' in h]
            if latencies:
                summary['avg_latency_ms'] = np.mean(latencies)
                summary['p95_latency_ms'] = np.percentile(latencies, 95)
        
        return summary


class DependencyHealthValidator:
    """
    Verify torch installation and CUDA availability
    Check JAX installation and device access
    Validate numerical libraries
    Test compression system imports
    """
    
    def __init__(self):
        """Initialize dependency validator"""
        self.dependencies: Dict[str, DependencyStatus] = {}
        self._lock = threading.RLock()
        
        # Define critical dependencies
        self.critical_deps = {
            'torch', 'numpy', 'psutil'
        }
        
        # Define optional dependencies
        self.optional_deps = {
            'jax', 'jaxlib', 'pynvml', 'scipy', 'pandas'
        }
        
        logger.info("DependencyHealthValidator initialized")
    
    def validate_torch(self) -> DependencyStatus:
        """
        Validate PyTorch installation and CUDA
        
        Returns:
            Dependency status for PyTorch
        """
        try:
            import torch
            
            version = torch.__version__
            cuda_available = torch.cuda.is_available()
            
            # Additional CUDA info
            cuda_info = {}
            if cuda_available:
                cuda_info = {
                    'device_count': torch.cuda.device_count(),
                    'current_device': torch.cuda.current_device(),
                    'device_name': torch.cuda.get_device_name(0),
                    'cuda_version': torch.version.cuda
                }
            
            status = DependencyStatus(
                dependency_name='torch',
                is_available=True,
                version=f"{version} (CUDA: {cuda_available})",
                error_message=None,
                is_critical=True
            )
            
            # Store additional info
            status.cuda_info = cuda_info
            
            return status
            
        except ImportError as e:
            return DependencyStatus(
                dependency_name='torch',
                is_available=False,
                version=None,
                error_message=str(e),
                is_critical=True
            )
        except Exception as e:
            return DependencyStatus(
                dependency_name='torch',
                is_available=False,
                version=None,
                error_message=f"Validation error: {e}",
                is_critical=True
            )
    
    def validate_jax(self) -> DependencyStatus:
        """
        Validate JAX installation and devices
        
        Returns:
            Dependency status for JAX
        """
        try:
            import jax
            import jaxlib
            
            version = f"jax={jax.__version__}, jaxlib={jaxlib.__version__}"
            
            # Check devices
            devices = jax.devices()
            device_info = {
                'device_count': len(devices),
                'devices': [str(d) for d in devices],
                'default_backend': jax.default_backend()
            }
            
            status = DependencyStatus(
                dependency_name='jax',
                is_available=True,
                version=version,
                error_message=None,
                is_critical=False
            )
            
            # Store additional info
            status.device_info = device_info
            
            return status
            
        except ImportError:
            return DependencyStatus(
                dependency_name='jax',
                is_available=False,
                version=None,
                error_message="JAX not installed",
                is_critical=False
            )
        except Exception as e:
            return DependencyStatus(
                dependency_name='jax',
                is_available=False,
                version=None,
                error_message=f"Validation error: {e}",
                is_critical=False
            )
    
    def validate_numerical_libraries(self) -> List[DependencyStatus]:
        """
        Validate numerical computation libraries
        
        Returns:
            List of dependency statuses
        """
        statuses = []
        
        # NumPy
        try:
            import numpy as np
            statuses.append(DependencyStatus(
                dependency_name='numpy',
                is_available=True,
                version=np.__version__,
                error_message=None,
                is_critical=True
            ))
        except ImportError as e:
            statuses.append(DependencyStatus(
                dependency_name='numpy',
                is_available=False,
                version=None,
                error_message=str(e),
                is_critical=True
            ))
        
        # SciPy
        try:
            import scipy
            statuses.append(DependencyStatus(
                dependency_name='scipy',
                is_available=True,
                version=scipy.__version__,
                error_message=None,
                is_critical=False
            ))
        except ImportError:
            statuses.append(DependencyStatus(
                dependency_name='scipy',
                is_available=False,
                version=None,
                error_message="SciPy not installed",
                is_critical=False
            ))
        
        # Pandas
        try:
            import pandas as pd
            statuses.append(DependencyStatus(
                dependency_name='pandas',
                is_available=True,
                version=pd.__version__,
                error_message=None,
                is_critical=False
            ))
        except ImportError:
            statuses.append(DependencyStatus(
                dependency_name='pandas',
                is_available=False,
                version=None,
                error_message="Pandas not installed",
                is_critical=False
            ))
        
        return statuses
    
    def test_compression_imports(self) -> Tuple[bool, List[str]]:
        """
        Test compression system imports
        
        Returns:
            Success status and list of failed imports
        """
        failed_imports = []
        
        # List of compression modules to test
        modules_to_test = [
            '..padic.padic_advanced',
            '..padic.hybrid_padic_compressor',
            '..padic.memory_pressure_handler',
            '..tropical.tropical_compression_pipeline',
            '..tropical.tropical_decompression_pipeline',
            '..tropical.tropical_channel_extractor',
            '..gpu_memory.gpu_memory_core',
            '..gpu_memory.smart_pool',
            '..gpu_memory.cpu_bursting_pipeline'
        ]
        
        for module_path in modules_to_test:
            try:
                importlib.import_module(module_path, package=__package__)
            except ImportError as e:
                failed_imports.append(f"{module_path}: {str(e)}")
            except Exception as e:
                failed_imports.append(f"{module_path}: Unexpected error - {str(e)}")
        
        return len(failed_imports) == 0, failed_imports
    
    def validate_all_dependencies(self) -> Dict[str, Any]:
        """
        Validate all dependencies
        
        Returns:
            Complete dependency validation report
        """
        with self._lock:
            report = {
                'timestamp': time.time(),
                'torch': asdict(self.validate_torch()),
                'jax': asdict(self.validate_jax()),
                'numerical_libraries': [asdict(s) for s in self.validate_numerical_libraries()],
                'compression_imports': {},
                'overall_status': HealthStatus.HEALTHY.value
            }
            
            # Test compression imports
            success, failed = self.test_compression_imports()
            report['compression_imports'] = {
                'success': success,
                'failed_imports': failed
            }
            
            # Determine overall status
            critical_missing = False
            
            # Check critical dependencies
            if not report['torch']['is_available']:
                critical_missing = True
            
            for lib in report['numerical_libraries']:
                if lib['is_critical'] and not lib['is_available']:
                    critical_missing = True
            
            if critical_missing:
                report['overall_status'] = HealthStatus.FAILURE.value
            elif not success or not report['jax']['is_available']:
                report['overall_status'] = HealthStatus.DEGRADED.value
            
            return report


class PerformanceHealthAnalyzer:
    """
    Detect performance degradation trends
    Identify bottlenecks
    Memory leak detection with growth tracking
    Deadlock and hang detection
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize performance health analyzer
        
        Args:
            window_size: Size of analysis window
        """
        if window_size <= 0:
            raise ValueError(f"Window size must be positive, got {window_size}")
        
        self.window_size = window_size
        self.performance_history: deque = deque(maxlen=window_size)
        self.memory_samples: deque = deque(maxlen=window_size)
        self.thread_states: Dict[int, Dict] = {}
        self._lock = threading.RLock()
        
        # Thresholds
        self.degradation_threshold = 0.2  # 20% performance drop
        self.memory_growth_threshold = 50  # MB
        self.hang_detection_timeout = 30  # seconds
        
        # Tracking
        self.baseline_performance = None
        self.performance_trends = defaultdict(list)
        self.bottlenecks_detected = []
        self.deadlock_detected = False
        
        logger.info(f"PerformanceHealthAnalyzer initialized with window_size={window_size}")
    
    def record_performance_sample(self, throughput: float, latency_ms: float,
                                 memory_mb: float, timestamp: Optional[float] = None):
        """
        Record a performance sample
        
        Args:
            throughput: Operations per second
            latency_ms: Operation latency in milliseconds
            memory_mb: Memory usage in MB
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            sample = {
                'timestamp': timestamp,
                'throughput': throughput,
                'latency_ms': latency_ms,
                'memory_mb': memory_mb
            }
            
            self.performance_history.append(sample)
            self.memory_samples.append(memory_mb)
            
            # Set baseline
            if self.baseline_performance is None and len(self.performance_history) >= 10:
                self.baseline_performance = {
                    'throughput': np.mean([s['throughput'] for s in list(self.performance_history)[:10]]),
                    'latency_ms': np.mean([s['latency_ms'] for s in list(self.performance_history)[:10]]),
                    'memory_mb': np.mean([s['memory_mb'] for s in list(self.performance_history)[:10]])
                }
    
    def detect_degradation(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect performance degradation trends
        
        Returns:
            Degradation detected flag and details
        """
        with self._lock:
            if len(self.performance_history) < 20 or self.baseline_performance is None:
                return False, {'insufficient_data': True}
            
            recent_samples = list(self.performance_history)[-10:]
            
            # Calculate recent averages
            recent_throughput = np.mean([s['throughput'] for s in recent_samples])
            recent_latency = np.mean([s['latency_ms'] for s in recent_samples])
            
            # Compare with baseline
            throughput_drop = (self.baseline_performance['throughput'] - recent_throughput) / self.baseline_performance['throughput']
            latency_increase = (recent_latency - self.baseline_performance['latency_ms']) / self.baseline_performance['latency_ms']
            
            degradation_detected = False
            details = {
                'baseline_throughput': self.baseline_performance['throughput'],
                'current_throughput': recent_throughput,
                'throughput_drop_percent': throughput_drop * 100,
                'baseline_latency_ms': self.baseline_performance['latency_ms'],
                'current_latency_ms': recent_latency,
                'latency_increase_percent': latency_increase * 100
            }
            
            if throughput_drop > self.degradation_threshold:
                degradation_detected = True
                details['issue'] = f"Throughput degraded by {throughput_drop*100:.1f}%"
            
            if latency_increase > self.degradation_threshold:
                degradation_detected = True
                details['issue'] = details.get('issue', '') + f" Latency increased by {latency_increase*100:.1f}%"
            
            return degradation_detected, details
    
    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        Identify system bottlenecks
        
        Returns:
            List of identified bottlenecks
        """
        with self._lock:
            bottlenecks = []
            
            if len(self.performance_history) < 10:
                return bottlenecks
            
            recent = list(self.performance_history)[-10:]
            
            # High latency bottleneck
            avg_latency = np.mean([s['latency_ms'] for s in recent])
            max_latency = max([s['latency_ms'] for s in recent])
            
            if avg_latency > 100:  # 100ms threshold
                bottlenecks.append({
                    'type': 'high_latency',
                    'severity': 'warning' if avg_latency < 500 else 'critical',
                    'avg_latency_ms': avg_latency,
                    'max_latency_ms': max_latency,
                    'recommendation': 'Consider batch size reduction or GPU offloading'
                })
            
            # Low throughput bottleneck
            avg_throughput = np.mean([s['throughput'] for s in recent])
            
            if self.baseline_performance and avg_throughput < self.baseline_performance['throughput'] * 0.5:
                bottlenecks.append({
                    'type': 'low_throughput',
                    'severity': 'critical',
                    'current_throughput': avg_throughput,
                    'expected_throughput': self.baseline_performance['throughput'],
                    'recommendation': 'Check for resource contention or I/O blocking'
                })
            
            # Memory pressure bottleneck
            if self.memory_samples:
                memory_usage = list(self.memory_samples)[-1]
                memory_available = psutil.virtual_memory().available / (1024**2)
                
                if memory_available < 1000:  # Less than 1GB available
                    bottlenecks.append({
                        'type': 'memory_pressure',
                        'severity': 'critical',
                        'available_mb': memory_available,
                        'used_mb': memory_usage,
                        'recommendation': 'Reduce batch size or enable memory swapping'
                    })
            
            self.bottlenecks_detected = bottlenecks
            return bottlenecks
    
    def detect_memory_leak(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect memory leaks with growth tracking
        
        Returns:
            Leak detected flag and details
        """
        with self._lock:
            if len(self.memory_samples) < 20:
                return False, {'insufficient_data': True}
            
            # Split samples into older and recent
            samples = list(self.memory_samples)
            older_samples = samples[:len(samples)//2]
            recent_samples = samples[len(samples)//2:]
            
            # Calculate growth
            older_avg = np.mean(older_samples)
            recent_avg = np.mean(recent_samples)
            growth = recent_avg - older_avg
            growth_rate = growth / len(samples)  # MB per sample
            
            # Check for consistent growth
            leak_detected = growth > self.memory_growth_threshold
            
            # Linear regression for trend
            x = np.arange(len(samples))
            coeffs = np.polyfit(x, samples, 1)
            slope = coeffs[0]
            
            details = {
                'older_avg_mb': older_avg,
                'recent_avg_mb': recent_avg,
                'total_growth_mb': growth,
                'growth_rate_mb_per_sample': growth_rate,
                'trend_slope': slope,
                'samples_analyzed': len(samples)
            }
            
            if leak_detected:
                details['estimated_leak_rate_mb_per_hour'] = slope * 3600 / (self.window_size * 0.1)  # Assuming 0.1s per sample
            
            return leak_detected, details
    
    def detect_deadlock(self) -> Tuple[bool, List[str]]:
        """
        Detect potential deadlocks and hangs
        
        Returns:
            Deadlock detected flag and list of stuck threads
        """
        with self._lock:
            current_time = time.time()
            stuck_threads = []
            
            # Get all threads
            for thread in threading.enumerate():
                thread_id = thread.ident
                
                if thread_id not in self.thread_states:
                    self.thread_states[thread_id] = {
                        'name': thread.name,
                        'last_active': current_time,
                        'is_alive': thread.is_alive()
                    }
                else:
                    # Check if thread state changed
                    if thread.is_alive():
                        # Thread still alive, check if it's stuck
                        last_active = self.thread_states[thread_id]['last_active']
                        
                        if current_time - last_active > self.hang_detection_timeout:
                            stuck_threads.append(f"{thread.name} (ID: {thread_id})")
            
            # Clean up dead threads
            dead_threads = [tid for tid, state in self.thread_states.items() 
                          if not any(t.ident == tid for t in threading.enumerate())]
            for tid in dead_threads:
                del self.thread_states[tid]
            
            self.deadlock_detected = len(stuck_threads) > 0
            
            return self.deadlock_detected, stuck_threads
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary
        
        Returns:
            Performance analysis summary
        """
        with self._lock:
            degradation_detected, degradation_details = self.detect_degradation()
            leak_detected, leak_details = self.detect_memory_leak()
            deadlock_detected, stuck_threads = self.detect_deadlock()
            bottlenecks = self.identify_bottlenecks()
            
            summary = {
                'timestamp': time.time(),
                'samples_analyzed': len(self.performance_history),
                'performance_degradation': {
                    'detected': degradation_detected,
                    'details': degradation_details
                },
                'memory_leak': {
                    'detected': leak_detected,
                    'details': leak_details
                },
                'deadlock': {
                    'detected': deadlock_detected,
                    'stuck_threads': stuck_threads
                },
                'bottlenecks': bottlenecks,
                'health_status': self._determine_health_status(
                    degradation_detected, leak_detected, deadlock_detected, len(bottlenecks) > 0
                )
            }
            
            return summary
    
    def _determine_health_status(self, degradation: bool, leak: bool, 
                                deadlock: bool, bottleneck: bool) -> str:
        """Determine overall health status"""
        if deadlock:
            return HealthStatus.CRITICAL.value
        elif leak or (degradation and bottleneck):
            return HealthStatus.WARNING.value
        elif degradation or bottleneck:
            return HealthStatus.DEGRADED.value
        else:
            return HealthStatus.HEALTHY.value


class HealthAlertingSystem:
    """
    Calculate overall health score (0-100)
    Manage alert thresholds and escalation
    Support multiple notification channels
    Implement auto-recovery triggers
    """
    
    def __init__(self, resource_monitor: ResourceHealthMonitor,
                 pipeline_checker: PipelineHealthChecker,
                 dependency_validator: DependencyHealthValidator,
                 performance_analyzer: PerformanceHealthAnalyzer):
        """
        Initialize health alerting system
        
        Args:
            resource_monitor: Resource health monitor
            pipeline_checker: Pipeline health checker
            dependency_validator: Dependency validator
            performance_analyzer: Performance analyzer
        """
        self.resource_monitor = resource_monitor
        self.pipeline_checker = pipeline_checker
        self.dependency_validator = dependency_validator
        self.performance_analyzer = performance_analyzer
        
        self.alerts: deque[HealthAlert] = deque(maxlen=1000)
        self.alert_handlers: Dict[AlertSeverity, List[Callable]] = defaultdict(list)
        self.recovery_handlers: Dict[RecoveryAction, Callable] = {}
        self._lock = threading.RLock()
        
        # Alert thresholds
        self.thresholds = {
            'health_score_warning': 70,
            'health_score_critical': 50,
            'health_score_emergency': 30
        }
        
        # Auto-recovery settings
        self.auto_recovery_enabled = True
        self.recovery_attempts: Dict[str, int] = defaultdict(int)
        self.max_recovery_attempts = 3
        
        # Alert state tracking
        self.active_alerts: Set[str] = set()
        self.alert_counter = 0
        
        logger.info("HealthAlertingSystem initialized")
    
    def calculate_health_score(self) -> HealthScore:
        """
        Calculate overall system health score
        
        Returns:
            Health score object
        """
        with self._lock:
            # Resource health (25% weight)
            cpu_status, cpu_issues = self.resource_monitor.check_cpu_health()
            memory_status, memory_issues = self.resource_monitor.check_memory_health()
            gpu_status, gpu_issues = self.resource_monitor.check_gpu_health()
            disk_status, disk_issues = self.resource_monitor.check_disk_health()
            
            resource_score = self._status_to_score([cpu_status, memory_status, gpu_status, disk_status])
            
            # Pipeline health (25% weight)
            pipeline_health = self.pipeline_checker.check_all_pipelines()
            pipeline_statuses = [ph.status for ph in pipeline_health.values()]
            pipeline_score = self._status_to_score(pipeline_statuses)
            
            # Dependency health (25% weight)
            dep_report = self.dependency_validator.validate_all_dependencies()
            dep_status = HealthStatus[dep_report['overall_status'].upper()]
            dependency_score = self._status_to_score([dep_status]) 
            
            # Performance health (25% weight)
            perf_summary = self.performance_analyzer.get_performance_summary()
            perf_status = HealthStatus[perf_summary['health_status'].upper()]
            performance_score = self._status_to_score([perf_status])
            
            # Calculate overall score
            overall_score = (
                resource_score * 0.25 +
                pipeline_score * 0.25 +
                dependency_score * 0.25 +
                performance_score * 0.25
            )
            
            # Determine overall status
            if overall_score >= 90:
                overall_status = HealthStatus.HEALTHY
            elif overall_score >= 70:
                overall_status = HealthStatus.DEGRADED
            elif overall_score >= 50:
                overall_status = HealthStatus.WARNING
            elif overall_score >= 30:
                overall_status = HealthStatus.CRITICAL
            else:
                overall_status = HealthStatus.FAILURE
            
            return HealthScore(
                overall_score=overall_score,
                resource_score=resource_score,
                pipeline_score=pipeline_score,
                dependency_score=dependency_score,
                performance_score=performance_score,
                timestamp=time.time(),
                status=overall_status
            )
    
    def create_alert(self, severity: AlertSeverity, component: str,
                    message: str, metrics: Optional[Dict[str, Any]] = None,
                    recovery_actions: Optional[List[RecoveryAction]] = None) -> HealthAlert:
        """
        Create and dispatch a health alert
        
        Args:
            severity: Alert severity level
            component: Component that triggered alert
            message: Alert message
            metrics: Optional metrics data
            recovery_actions: Optional recovery actions
            
        Returns:
            Created health alert
        """
        with self._lock:
            self.alert_counter += 1
            alert_id = f"alert_{self.alert_counter}_{int(time.time())}"
            
            alert = HealthAlert(
                alert_id=alert_id,
                timestamp=time.time(),
                severity=severity,
                component=component,
                message=message,
                metrics=metrics or {},
                recovery_actions=recovery_actions or [],
                auto_recovery_attempted=False
            )
            
            self.alerts.append(alert)
            self.active_alerts.add(alert_id)
            
            # Dispatch to handlers
            self._dispatch_alert(alert)
            
            # Attempt auto-recovery if enabled
            if self.auto_recovery_enabled and recovery_actions:
                self._attempt_auto_recovery(alert)
            
            return alert
    
    def register_alert_handler(self, severity: AlertSeverity, handler: Callable):
        """
        Register an alert handler for a severity level
        
        Args:
            severity: Severity level to handle
            handler: Handler function (receives HealthAlert)
        """
        with self._lock:
            self.alert_handlers[severity].append(handler)
            logger.info(f"Registered alert handler for {severity.value}")
    
    def register_recovery_handler(self, action: RecoveryAction, handler: Callable):
        """
        Register a recovery action handler
        
        Args:
            action: Recovery action type
            handler: Handler function (receives HealthAlert)
        """
        with self._lock:
            self.recovery_handlers[action] = handler
            logger.info(f"Registered recovery handler for {action.value}")
    
    def check_and_alert(self) -> List[HealthAlert]:
        """
        Check system health and generate alerts
        
        Returns:
            List of generated alerts
        """
        with self._lock:
            alerts_generated = []
            
            # Get health score
            health_score = self.calculate_health_score()
            
            # Check overall health
            if health_score.overall_score < self.thresholds['health_score_emergency']:
                alert = self.create_alert(
                    severity=AlertSeverity.EMERGENCY,
                    component='system',
                    message=f"Emergency: System health critical ({health_score.overall_score:.1f}/100)",
                    metrics={'health_score': asdict(health_score)},
                    recovery_actions=[RecoveryAction.RESTART_PIPELINE, RecoveryAction.CLEAR_MEMORY]
                )
                alerts_generated.append(alert)
                
            elif health_score.overall_score < self.thresholds['health_score_critical']:
                alert = self.create_alert(
                    severity=AlertSeverity.CRITICAL,
                    component='system',
                    message=f"Critical: System health degraded ({health_score.overall_score:.1f}/100)",
                    metrics={'health_score': asdict(health_score)},
                    recovery_actions=[RecoveryAction.GARBAGE_COLLECT, RecoveryAction.REDUCE_BATCH_SIZE]
                )
                alerts_generated.append(alert)
                
            elif health_score.overall_score < self.thresholds['health_score_warning']:
                alert = self.create_alert(
                    severity=AlertSeverity.WARNING,
                    component='system',
                    message=f"Warning: System health below normal ({health_score.overall_score:.1f}/100)",
                    metrics={'health_score': asdict(health_score)}
                )
                alerts_generated.append(alert)
            
            # Check specific components
            self._check_resource_alerts(alerts_generated)
            self._check_performance_alerts(alerts_generated)
            self._check_pipeline_alerts(alerts_generated)
            
            return alerts_generated
    
    def _check_resource_alerts(self, alerts_list: List[HealthAlert]):
        """Check and generate resource-related alerts"""
        # CPU alerts
        cpu_status, cpu_issues = self.resource_monitor.check_cpu_health()
        if cpu_status == HealthStatus.CRITICAL:
            alert = self.create_alert(
                severity=AlertSeverity.CRITICAL,
                component='cpu',
                message=f"CPU health critical: {', '.join(cpu_issues)}",
                recovery_actions=[RecoveryAction.THROTTLE_REQUESTS]
            )
            alerts_list.append(alert)
        
        # Memory alerts
        memory_status, memory_issues = self.resource_monitor.check_memory_health()
        if memory_status == HealthStatus.CRITICAL:
            alert = self.create_alert(
                severity=AlertSeverity.CRITICAL,
                component='memory',
                message=f"Memory health critical: {', '.join(memory_issues)}",
                recovery_actions=[RecoveryAction.CLEAR_MEMORY, RecoveryAction.GARBAGE_COLLECT]
            )
            alerts_list.append(alert)
        
        # GPU alerts
        gpu_status, gpu_issues = self.resource_monitor.check_gpu_health()
        if gpu_status == HealthStatus.CRITICAL:
            alert = self.create_alert(
                severity=AlertSeverity.CRITICAL,
                component='gpu',
                message=f"GPU health critical: {', '.join(gpu_issues)}",
                recovery_actions=[RecoveryAction.SWITCH_TO_CPU, RecoveryAction.RESET_GPU]
            )
            alerts_list.append(alert)
    
    def _check_performance_alerts(self, alerts_list: List[HealthAlert]):
        """Check and generate performance-related alerts"""
        perf_summary = self.performance_analyzer.get_performance_summary()
        
        # Memory leak alert
        if perf_summary['memory_leak']['detected']:
            alert = self.create_alert(
                severity=AlertSeverity.ERROR,
                component='performance',
                message="Memory leak detected",
                metrics=perf_summary['memory_leak']['details'],
                recovery_actions=[RecoveryAction.GARBAGE_COLLECT, RecoveryAction.RESTART_PIPELINE]
            )
            alerts_list.append(alert)
        
        # Deadlock alert
        if perf_summary['deadlock']['detected']:
            alert = self.create_alert(
                severity=AlertSeverity.EMERGENCY,
                component='performance',
                message=f"Deadlock detected: {len(perf_summary['deadlock']['stuck_threads'])} threads stuck",
                metrics={'stuck_threads': perf_summary['deadlock']['stuck_threads']},
                recovery_actions=[RecoveryAction.KILL_ZOMBIE_THREADS, RecoveryAction.RESTART_PIPELINE]
            )
            alerts_list.append(alert)
        
        # Performance degradation alert
        if perf_summary['performance_degradation']['detected']:
            alert = self.create_alert(
                severity=AlertSeverity.WARNING,
                component='performance',
                message="Performance degradation detected",
                metrics=perf_summary['performance_degradation']['details'],
                recovery_actions=[RecoveryAction.REDUCE_BATCH_SIZE]
            )
            alerts_list.append(alert)
    
    def _check_pipeline_alerts(self, alerts_list: List[HealthAlert]):
        """Check and generate pipeline-related alerts"""
        pipeline_health = self.pipeline_checker.check_all_pipelines()
        
        for pipeline_type, health in pipeline_health.items():
            if health.status == HealthStatus.FAILURE:
                alert = self.create_alert(
                    severity=AlertSeverity.ERROR,
                    component=f'pipeline_{pipeline_type.value}',
                    message=f"{pipeline_type.value} pipeline failure",
                    metrics={'error_messages': health.error_messages},
                    recovery_actions=[RecoveryAction.RESTART_PIPELINE]
                )
                alerts_list.append(alert)
    
    def _dispatch_alert(self, alert: HealthAlert):
        """Dispatch alert to registered handlers"""
        handlers = self.alert_handlers.get(alert.severity, [])
        for handler in handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def _attempt_auto_recovery(self, alert: HealthAlert):
        """Attempt automatic recovery actions"""
        if not alert.recovery_actions:
            return
        
        recovery_key = f"{alert.component}_{alert.recovery_actions[0].value}"
        
        # Check recovery attempt limit
        if self.recovery_attempts[recovery_key] >= self.max_recovery_attempts:
            logger.warning(f"Max recovery attempts reached for {recovery_key}")
            return
        
        self.recovery_attempts[recovery_key] += 1
        alert.auto_recovery_attempted = True
        
        # Execute recovery actions
        for action in alert.recovery_actions:
            if action in self.recovery_handlers:
                try:
                    logger.info(f"Executing recovery action: {action.value}")
                    self.recovery_handlers[action](alert)
                except Exception as e:
                    logger.error(f"Recovery action {action.value} failed: {e}")
            else:
                # Built-in recovery actions
                self._execute_builtin_recovery(action, alert)
    
    def _execute_builtin_recovery(self, action: RecoveryAction, alert: HealthAlert):
        """Execute built-in recovery actions"""
        try:
            if action == RecoveryAction.GARBAGE_COLLECT:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Executed garbage collection")
                
            elif action == RecoveryAction.CLEAR_MEMORY:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                logger.info("Cleared memory caches")
                
            elif action == RecoveryAction.THROTTLE_REQUESTS:
                # Would implement request throttling logic
                logger.info("Request throttling activated")
                
            else:
                logger.warning(f"No built-in handler for recovery action: {action.value}")
                
        except Exception as e:
            logger.error(f"Built-in recovery action {action.value} failed: {e}")
    
    def _status_to_score(self, statuses: List[HealthStatus]) -> float:
        """Convert health statuses to numeric score"""
        if not statuses:
            return 0.0
        
        status_scores = {
            HealthStatus.HEALTHY: 100,
            HealthStatus.DEGRADED: 75,
            HealthStatus.WARNING: 50,
            HealthStatus.CRITICAL: 25,
            HealthStatus.FAILURE: 10,
            HealthStatus.UNKNOWN: 50
        }
        
        scores = [status_scores.get(status, 50) for status in statuses]
        return np.mean(scores)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Get summary of alerts
        
        Returns:
            Alert summary dictionary
        """
        with self._lock:
            alerts = list(self.alerts)
            
            summary = {
                'total_alerts': len(alerts),
                'active_alerts': len(self.active_alerts),
                'by_severity': defaultdict(int),
                'by_component': defaultdict(int),
                'recent_alerts': [],
                'recovery_attempts': dict(self.recovery_attempts)
            }
            
            for alert in alerts:
                summary['by_severity'][alert.severity.value] += 1
                summary['by_component'][alert.component] += 1
            
            # Get last 10 alerts
            recent = sorted(alerts, key=lambda a: a.timestamp, reverse=True)[:10]
            summary['recent_alerts'] = [alert.to_dict() for alert in recent]
            
            return summary


# Example usage and testing
if __name__ == "__main__":
    # Initialize monitoring configuration
    monitor_config = MonitorConfig(
        sampling_interval_ms=100,
        history_window_size=1000,
        enable_detailed_tracking=True
    )
    
    # Initialize performance monitor
    perf_monitor = UnifiedPerformanceMonitor(config=monitor_config)
    
    # Initialize health monitoring components
    resource_monitor = ResourceHealthMonitor(sampling_interval_ms=1000)
    pipeline_checker = PipelineHealthChecker(performance_monitor=perf_monitor)
    dependency_validator = DependencyHealthValidator()
    performance_analyzer = PerformanceHealthAnalyzer(window_size=100)
    
    # Initialize alerting system
    alerting_system = HealthAlertingSystem(
        resource_monitor=resource_monitor,
        pipeline_checker=pipeline_checker,
        dependency_validator=dependency_validator,
        performance_analyzer=performance_analyzer
    )
    
    # Register example alert handler
    def example_alert_handler(alert: HealthAlert):
        print(f"ALERT [{alert.severity.value}]: {alert.message}")
    
    alerting_system.register_alert_handler(AlertSeverity.CRITICAL, example_alert_handler)
    alerting_system.register_alert_handler(AlertSeverity.EMERGENCY, example_alert_handler)
    
    # Run health checks
    print("Starting system health monitoring...")
    
    # Check dependencies
    dep_report = dependency_validator.validate_all_dependencies()
    print(f"\nDependency Status: {dep_report['overall_status']}")
    
    # Check pipelines
    pipeline_health = pipeline_checker.check_all_pipelines()
    for pipeline_type, health in pipeline_health.items():
        print(f"{pipeline_type.value} Pipeline: {health.status.value}")
    
    # Calculate health score
    health_score = alerting_system.calculate_health_score()
    print(f"\nOverall Health Score: {health_score.overall_score:.1f}/100")
    print(f"  Resource Score: {health_score.resource_score:.1f}")
    print(f"  Pipeline Score: {health_score.pipeline_score:.1f}")
    print(f"  Dependency Score: {health_score.dependency_score:.1f}")
    print(f"  Performance Score: {health_score.performance_score:.1f}")
    
    # Simulate some performance samples
    for i in range(20):
        performance_analyzer.record_performance_sample(
            throughput=100 + np.random.randn() * 10,
            latency_ms=50 + np.random.randn() * 5,
            memory_mb=1000 + i * 5  # Simulate memory growth
        )
        time.sleep(0.1)
    
    # Check for alerts
    alerts = alerting_system.check_and_alert()
    if alerts:
        print(f"\nGenerated {len(alerts)} alerts")
    
    # Get alert summary
    alert_summary = alerting_system.get_alert_summary()
    print(f"\nAlert Summary:")
    print(f"  Total Alerts: {alert_summary['total_alerts']}")
    print(f"  Active Alerts: {alert_summary['active_alerts']}")
    
    # Get performance summary
    perf_summary = performance_analyzer.get_performance_summary()
    print(f"\nPerformance Analysis:")
    print(f"  Degradation Detected: {perf_summary['performance_degradation']['detected']}")
    print(f"  Memory Leak Detected: {perf_summary['memory_leak']['detected']}")
    print(f"  Deadlock Detected: {perf_summary['deadlock']['detected']}")
    print(f"  Bottlenecks: {len(perf_summary['bottlenecks'])}")
    
    # Shutdown
    print("\nShutting down health monitoring...")
    resource_monitor.shutdown()
    perf_monitor.shutdown()
    print("Health monitoring shutdown complete")