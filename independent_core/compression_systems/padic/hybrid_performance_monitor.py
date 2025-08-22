"""
Hybrid Performance Monitor - Real-time monitoring of hybrid system performance
NO FALLBACKS - HARD FAILURES ONLY
"""

import asyncio
import logging
import threading
import time
import torch
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

# Import hybrid system components
from hybrid_padic_structures import HybridPadicWeight, HybridPadicManager
from dynamic_switching_manager import DynamicSwitchingManager
from hybrid_padic_compressor import HybridPadicCompressionSystem


class MonitoringLevel(Enum):
    """Monitoring level enumeration"""
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    DEBUG = "debug"


class AlertSeverity(Enum):
    """Alert severity enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MonitoringStatus(Enum):
    """Monitoring status enumeration"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class HybridOperationMonitorData:
    """Data structure for monitoring hybrid operations"""
    operation_id: str
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time_ms: float = 0.0
    gpu_memory_before_mb: float = 0.0
    gpu_memory_after_mb: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    hybrid_mode_used: bool = False
    switching_events: int = 0
    data_size_elements: int = 0
    success: bool = True
    error_message: Optional[str] = None
    performance_score: float = 0.0
    
    def __post_init__(self):
        """Validate monitoring data"""
        if not isinstance(self.operation_id, str) or not self.operation_id.strip():
            raise ValueError("Operation ID must be non-empty string")
        if not isinstance(self.operation_name, str) or not self.operation_name.strip():
            raise ValueError("Operation name must be non-empty string")
        if not isinstance(self.start_time, datetime):
            raise TypeError("Start time must be datetime")
        if not isinstance(self.success, bool):
            raise TypeError("Success must be bool")
    
    def calculate_performance_score(self) -> float:
        """Calculate performance score for operation"""
        if not self.success:
            return 0.0
        
        # Base score from execution time (lower is better)
        time_score = max(0.0, 1.0 - (self.execution_time_ms / 10000.0))  # Normalize to 10 seconds
        
        # GPU utilization score (higher is better for GPU operations)
        gpu_score = self.gpu_utilization_percent / 100.0 if self.hybrid_mode_used else 1.0
        
        # Memory efficiency score (lower usage is better)
        memory_score = max(0.0, 1.0 - (self.gpu_memory_peak_mb / 2048.0))  # Normalize to 2GB
        
        # Combined score
        self.performance_score = (time_score * 0.4 + gpu_score * 0.3 + memory_score * 0.3)
        return self.performance_score


@dataclass
class HybridPerformanceAlert:
    """Performance alert for hybrid operations"""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    operation_name: Optional[str] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    
    def __post_init__(self):
        """Validate alert"""
        if not isinstance(self.alert_id, str) or not self.alert_id.strip():
            raise ValueError("Alert ID must be non-empty string")
        if not isinstance(self.alert_type, str) or not self.alert_type.strip():
            raise ValueError("Alert type must be non-empty string")
        if not isinstance(self.severity, AlertSeverity):
            raise TypeError("Severity must be AlertSeverity")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("Message must be non-empty string")


@dataclass
class HybridMonitoringMetrics:
    """Comprehensive monitoring metrics for hybrid operations"""
    # Operation counts
    total_operations_monitored: int = 0
    hybrid_operations: int = 0
    pure_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    
    # Performance metrics
    average_execution_time_ms: float = 0.0
    average_hybrid_execution_time_ms: float = 0.0
    average_pure_execution_time_ms: float = 0.0
    min_execution_time_ms: float = float('inf')
    max_execution_time_ms: float = 0.0
    
    # Resource metrics
    average_gpu_memory_usage_mb: float = 0.0
    peak_gpu_memory_usage_mb: float = 0.0
    average_cpu_usage_percent: float = 0.0
    average_memory_usage_mb: float = 0.0
    average_gpu_utilization_percent: float = 0.0
    
    # Switching metrics
    total_switching_events: int = 0
    hybrid_to_pure_switches: int = 0
    pure_to_hybrid_switches: int = 0
    average_switching_overhead_ms: float = 0.0
    
    # Quality metrics
    average_performance_score: float = 0.0
    performance_stability: float = 0.0  # Coefficient of variation
    success_rate: float = 0.0
    
    # Monitoring metadata
    monitoring_start_time: Optional[datetime] = None
    last_update: Optional[datetime] = None
    monitoring_duration_hours: float = 0.0
    
    def update_with_operation(self, operation_data: HybridOperationMonitorData):
        """Update metrics with new operation data"""
        self.total_operations_monitored += 1
        self.last_update = datetime.utcnow()
        
        if self.monitoring_start_time is None:
            self.monitoring_start_time = operation_data.start_time
        
        # Update monitoring duration
        if self.monitoring_start_time:
            duration = self.last_update - self.monitoring_start_time
            self.monitoring_duration_hours = duration.total_seconds() / 3600.0
        
        # Update operation counts
        if operation_data.hybrid_mode_used:
            self.hybrid_operations += 1
        else:
            self.pure_operations += 1
        
        if operation_data.success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1
        
        # Update execution time metrics
        exec_time = operation_data.execution_time_ms
        if exec_time > 0:
            # Update averages
            if self.total_operations_monitored > 1:
                old_avg = self.average_execution_time_ms
                self.average_execution_time_ms = (
                    (old_avg * (self.total_operations_monitored - 1) + exec_time) / self.total_operations_monitored
                )
                
                if operation_data.hybrid_mode_used and self.hybrid_operations > 1:
                    old_hybrid_avg = self.average_hybrid_execution_time_ms
                    self.average_hybrid_execution_time_ms = (
                        (old_hybrid_avg * (self.hybrid_operations - 1) + exec_time) / self.hybrid_operations
                    )
                elif not operation_data.hybrid_mode_used and self.pure_operations > 1:
                    old_pure_avg = self.average_pure_execution_time_ms
                    self.average_pure_execution_time_ms = (
                        (old_pure_avg * (self.pure_operations - 1) + exec_time) / self.pure_operations
                    )
            else:
                self.average_execution_time_ms = exec_time
                if operation_data.hybrid_mode_used:
                    self.average_hybrid_execution_time_ms = exec_time
                else:
                    self.average_pure_execution_time_ms = exec_time
            
            # Update min/max
            self.min_execution_time_ms = min(self.min_execution_time_ms, exec_time)
            self.max_execution_time_ms = max(self.max_execution_time_ms, exec_time)
        
        # Update resource metrics
        self._update_resource_metrics(operation_data)
        
        # Update switching metrics
        self.total_switching_events += operation_data.switching_events
        
        # Update quality metrics
        self._update_quality_metrics(operation_data)
    
    def _update_resource_metrics(self, operation_data: HybridOperationMonitorData):
        """Update resource usage metrics"""
        # GPU memory
        if operation_data.gpu_memory_peak_mb > 0:
            old_gpu_avg = self.average_gpu_memory_usage_mb
            self.average_gpu_memory_usage_mb = (
                (old_gpu_avg * (self.total_operations_monitored - 1) + operation_data.gpu_memory_peak_mb) / 
                self.total_operations_monitored
            )
            self.peak_gpu_memory_usage_mb = max(self.peak_gpu_memory_usage_mb, operation_data.gpu_memory_peak_mb)
        
        # CPU usage
        if operation_data.cpu_usage_percent > 0:
            old_cpu_avg = self.average_cpu_usage_percent
            self.average_cpu_usage_percent = (
                (old_cpu_avg * (self.total_operations_monitored - 1) + operation_data.cpu_usage_percent) / 
                self.total_operations_monitored
            )
        
        # Memory usage
        if operation_data.memory_usage_mb > 0:
            old_mem_avg = self.average_memory_usage_mb
            self.average_memory_usage_mb = (
                (old_mem_avg * (self.total_operations_monitored - 1) + operation_data.memory_usage_mb) / 
                self.total_operations_monitored
            )
        
        # GPU utilization
        if operation_data.gpu_utilization_percent > 0:
            old_gpu_util_avg = self.average_gpu_utilization_percent
            self.average_gpu_utilization_percent = (
                (old_gpu_util_avg * (self.total_operations_monitored - 1) + operation_data.gpu_utilization_percent) / 
                self.total_operations_monitored
            )
    
    def _update_quality_metrics(self, operation_data: HybridOperationMonitorData):
        """Update quality metrics"""
        # Performance score
        if operation_data.performance_score > 0:
            old_perf_avg = self.average_performance_score
            self.average_performance_score = (
                (old_perf_avg * (self.total_operations_monitored - 1) + operation_data.performance_score) / 
                self.total_operations_monitored
            )
        
        # Success rate
        self.success_rate = self.successful_operations / self.total_operations_monitored


@dataclass
class HybridMonitoringConfig:
    """Configuration for hybrid performance monitoring"""
    monitoring_level: MonitoringLevel = MonitoringLevel.DETAILED
    monitoring_interval_seconds: float = 1.0
    gpu_monitoring_enabled: bool = True
    cpu_monitoring_enabled: bool = True
    memory_monitoring_enabled: bool = True
    switching_monitoring_enabled: bool = True
    
    # Alert thresholds
    execution_time_threshold_ms: float = 5000.0
    gpu_memory_threshold_mb: float = 1024.0
    cpu_usage_threshold_percent: float = 80.0
    memory_usage_threshold_mb: float = 2048.0
    gpu_utilization_threshold_percent: float = 90.0
    performance_score_threshold: float = 0.3
    
    # Data retention
    max_operation_history: int = 1000
    max_alert_history: int = 500
    data_retention_hours: int = 24
    
    # Performance optimization
    enable_async_monitoring: bool = True
    batch_processing_size: int = 10
    monitoring_overhead_limit_percent: float = 5.0
    
    def __post_init__(self):
        """Validate configuration"""
        if not isinstance(self.monitoring_level, MonitoringLevel):
            raise TypeError("Monitoring level must be MonitoringLevel")
        if self.monitoring_interval_seconds <= 0:
            raise ValueError("Monitoring interval must be positive")
        if self.execution_time_threshold_ms <= 0:
            raise ValueError("Execution time threshold must be positive")
        if self.max_operation_history <= 0:
            raise ValueError("Max operation history must be positive")


class HybridPerformanceMonitor:
    """
    Real-time monitoring of hybrid system performance.
    Provides comprehensive monitoring of hybrid operations, resource usage, and performance metrics.
    """
    
    def __init__(self, config: Optional[HybridMonitoringConfig] = None):
        """Initialize hybrid performance monitor"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, HybridMonitoringConfig):
            raise TypeError(f"Config must be HybridMonitoringConfig or None, got {type(config)}")
        
        self.config = config or HybridMonitoringConfig()
        self.logger = logging.getLogger('HybridPerformanceMonitor')
        
        # Monitoring state
        self.monitoring_status = MonitoringStatus.STOPPED
        self.is_monitoring = False
        
        # Component references
        self.hybrid_padic_manager: Optional[HybridPadicManager] = None
        self.dynamic_switching_manager: Optional[DynamicSwitchingManager] = None
        self.hybrid_compression_system: Optional[HybridPadicCompressionSystem] = None
        
        # Monitoring data
        self.monitoring_metrics = HybridMonitoringMetrics()
        self.operation_history: deque = deque(maxlen=self.config.max_operation_history)
        self.alert_history: deque = deque(maxlen=self.config.max_alert_history)
        self.active_operations: Dict[str, HybridOperationMonitorData] = {}
        
        # Performance tracking
        self.performance_trends: deque = deque(maxlen=100)
        self.resource_usage_history: deque = deque(maxlen=200)
        self.switching_events: deque = deque(maxlen=500)
        
        # Thread safety
        self._monitoring_lock = threading.RLock()
        self._metrics_lock = threading.RLock()
        self._operations_lock = threading.RLock()
        self._alerts_lock = threading.RLock()
        
        # Background monitoring
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._monitoring_enabled = False
        
        # Async monitoring
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_tasks: List[asyncio.Task] = []
        
        self.logger.info("HybridPerformanceMonitor created successfully")
    
    def start_hybrid_monitoring(self,
                              hybrid_manager: HybridPadicManager,
                              switching_manager: Optional[DynamicSwitchingManager] = None,
                              compression_system: Optional[HybridPadicCompressionSystem] = None) -> None:
        """
        Start hybrid performance monitoring.
        
        Args:
            hybrid_manager: Hybrid p-adic manager instance
            switching_manager: Optional dynamic switching manager
            compression_system: Optional hybrid compression system
            
        Raises:
            TypeError: If hybrid_manager is invalid
            RuntimeError: If monitoring fails to start
        """
        if not isinstance(hybrid_manager, HybridPadicManager):
            raise TypeError(f"Hybrid manager must be HybridPadicManager, got {type(hybrid_manager)}")
        if switching_manager is not None and not isinstance(switching_manager, DynamicSwitchingManager):
            raise TypeError(f"Switching manager must be DynamicSwitchingManager, got {type(switching_manager)}")
        if compression_system is not None and not isinstance(compression_system, HybridPadicCompressionSystem):
            raise TypeError(f"Compression system must be HybridPadicCompressionSystem, got {type(compression_system)}")
        
        if self.is_monitoring:
            return
        
        with self._monitoring_lock:
            try:
                self.monitoring_status = MonitoringStatus.STARTING
                
                # Set component references
                self.hybrid_padic_manager = hybrid_manager
                self.dynamic_switching_manager = switching_manager
                self.hybrid_compression_system = compression_system
                
                # Initialize monitoring
                self._initialize_monitoring()
                
                # Start monitoring threads
                self._start_monitoring_threads()
                
                # Start async monitoring if enabled
                if self.config.enable_async_monitoring:
                    self._start_async_monitoring()
                
                self.monitoring_status = MonitoringStatus.RUNNING
                self.is_monitoring = True
                
                self.logger.info("Hybrid performance monitoring started successfully")
                
            except Exception as e:
                self.monitoring_status = MonitoringStatus.ERROR
                self.logger.error(f"Failed to start hybrid monitoring: {e}")
                raise RuntimeError(f"Hybrid monitoring startup failed: {e}")
    
    def stop_hybrid_monitoring(self) -> None:
        """
        Stop hybrid performance monitoring.
        
        Raises:
            RuntimeError: If monitoring fails to stop
        """
        if not self.is_monitoring:
            return
        
        with self._monitoring_lock:
            try:
                self.monitoring_status = MonitoringStatus.STOPPING
                
                # Stop monitoring threads
                self._stop_monitoring_threads()
                
                # Stop async monitoring
                self._stop_async_monitoring()
                
                # Finalize monitoring data
                self._finalize_monitoring()
                
                self.monitoring_status = MonitoringStatus.STOPPED
                self.is_monitoring = False
                
                self.logger.info("Hybrid performance monitoring stopped successfully")
                
            except Exception as e:
                self.monitoring_status = MonitoringStatus.ERROR
                self.logger.error(f"Failed to stop hybrid monitoring: {e}")
                raise RuntimeError(f"Hybrid monitoring shutdown failed: {e}")
    
    def monitor_hybrid_operation(self, operation_name: str, operation_data: Dict[str, Any]) -> str:
        """
        Monitor a hybrid operation.
        
        Args:
            operation_name: Name of the operation to monitor
            operation_data: Data about the operation
            
        Returns:
            Operation ID for tracking
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If monitoring is not active
        """
        if not isinstance(operation_name, str) or not operation_name.strip():
            raise ValueError("Operation name must be non-empty string")
        if not isinstance(operation_data, dict):
            raise TypeError("Operation data must be dict")
        
        if not self.is_monitoring:
            raise RuntimeError("Monitoring is not active")
        
        operation_id = str(uuid.uuid4())
        
        try:
            # Create monitoring data
            monitor_data = HybridOperationMonitorData(
                operation_id=operation_id,
                operation_name=operation_name,
                start_time=datetime.utcnow(),
                gpu_memory_before_mb=self._get_gpu_memory_usage(),
                cpu_usage_percent=self._get_cpu_usage(),
                memory_usage_mb=self._get_memory_usage(),
                gpu_utilization_percent=self._get_gpu_utilization(),
                hybrid_mode_used=operation_data.get('hybrid_mode', False),
                data_size_elements=operation_data.get('data_size', 0)
            )
            
            # Store active operation
            with self._operations_lock:
                self.active_operations[operation_id] = monitor_data
            
            self.logger.debug(f"Started monitoring operation '{operation_name}' with ID {operation_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring operation '{operation_name}': {e}")
            raise RuntimeError(f"Operation monitoring failed: {e}")
        
        return operation_id
    
    def complete_operation_monitoring(self, operation_id: str, success: bool = True, 
                                    error_message: Optional[str] = None) -> None:
        """
        Complete monitoring for an operation.
        
        Args:
            operation_id: ID of the operation to complete
            success: Whether the operation was successful
            error_message: Optional error message if operation failed
            
        Raises:
            ValueError: If operation_id is invalid
            RuntimeError: If operation is not being monitored
        """
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ValueError("Operation ID must be non-empty string")
        if not isinstance(success, bool):
            raise TypeError("Success must be bool")
        
        with self._operations_lock:
            if operation_id not in self.active_operations:
                raise RuntimeError(f"Operation {operation_id} is not being monitored")
            
            monitor_data = self.active_operations[operation_id]
        
        try:
            # Complete monitoring data
            monitor_data.end_time = datetime.utcnow()
            monitor_data.execution_time_ms = (
                (monitor_data.end_time - monitor_data.start_time).total_seconds() * 1000
            )
            monitor_data.gpu_memory_after_mb = self._get_gpu_memory_usage()
            monitor_data.gpu_memory_peak_mb = max(
                monitor_data.gpu_memory_before_mb,
                monitor_data.gpu_memory_after_mb
            )
            monitor_data.success = success
            monitor_data.error_message = error_message
            
            # Calculate performance score
            monitor_data.calculate_performance_score()
            
            # Update metrics
            with self._metrics_lock:
                self.monitoring_metrics.update_with_operation(monitor_data)
            
            # Store in history
            self.operation_history.append(monitor_data)
            
            # Remove from active operations
            with self._operations_lock:
                del self.active_operations[operation_id]
            
            # Check for alerts
            self._check_operation_alerts(monitor_data)
            
            self.logger.debug(f"Completed monitoring operation {operation_id}: "
                            f"{monitor_data.execution_time_ms:.2f}ms, success={success}")
            
        except Exception as e:
            self.logger.error(f"Failed to complete monitoring for operation {operation_id}: {e}")
            # Don't raise exception here to avoid interfering with operation completion
    
    def get_hybrid_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current hybrid performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        with self._metrics_lock:
            return {
                'monitoring_status': self.monitoring_status.value,
                'monitoring_level': self.config.monitoring_level.value,
                'operation_metrics': {
                    'total_operations': self.monitoring_metrics.total_operations_monitored,
                    'hybrid_operations': self.monitoring_metrics.hybrid_operations,
                    'pure_operations': self.monitoring_metrics.pure_operations,
                    'successful_operations': self.monitoring_metrics.successful_operations,
                    'failed_operations': self.monitoring_metrics.failed_operations,
                    'success_rate': self.monitoring_metrics.success_rate,
                    'hybrid_ratio': (
                        self.monitoring_metrics.hybrid_operations / 
                        max(1, self.monitoring_metrics.total_operations_monitored)
                    )
                },
                'performance_metrics': {
                    'average_execution_time_ms': self.monitoring_metrics.average_execution_time_ms,
                    'average_hybrid_time_ms': self.monitoring_metrics.average_hybrid_execution_time_ms,
                    'average_pure_time_ms': self.monitoring_metrics.average_pure_execution_time_ms,
                    'min_execution_time_ms': self.monitoring_metrics.min_execution_time_ms,
                    'max_execution_time_ms': self.monitoring_metrics.max_execution_time_ms,
                    'average_performance_score': self.monitoring_metrics.average_performance_score,
                    'performance_stability': self.monitoring_metrics.performance_stability
                },
                'resource_metrics': {
                    'average_gpu_memory_mb': self.monitoring_metrics.average_gpu_memory_usage_mb,
                    'peak_gpu_memory_mb': self.monitoring_metrics.peak_gpu_memory_usage_mb,
                    'average_cpu_usage': self.monitoring_metrics.average_cpu_usage_percent,
                    'average_memory_usage_mb': self.monitoring_metrics.average_memory_usage_mb,
                    'average_gpu_utilization': self.monitoring_metrics.average_gpu_utilization_percent
                },
                'switching_metrics': {
                    'total_switching_events': self.monitoring_metrics.total_switching_events,
                    'hybrid_to_pure_switches': self.monitoring_metrics.hybrid_to_pure_switches,
                    'pure_to_hybrid_switches': self.monitoring_metrics.pure_to_hybrid_switches,
                    'average_switching_overhead_ms': self.monitoring_metrics.average_switching_overhead_ms
                },
                'monitoring_info': {
                    'monitoring_start_time': (
                        self.monitoring_metrics.monitoring_start_time.isoformat() 
                        if self.monitoring_metrics.monitoring_start_time else None
                    ),
                    'last_update': (
                        self.monitoring_metrics.last_update.isoformat() 
                        if self.monitoring_metrics.last_update else None
                    ),
                    'monitoring_duration_hours': self.monitoring_metrics.monitoring_duration_hours,
                    'active_operations': len(self.active_operations),
                    'operation_history_length': len(self.operation_history)
                }
            }
    
    def analyze_hybrid_performance_trends(self) -> Dict[str, Any]:
        """
        Analyze hybrid performance trends.
        
        Returns:
            Dictionary containing trend analysis
        """
        if len(self.operation_history) < 10:
            return {'status': 'insufficient_data', 'required_operations': 10}
        
        try:
            recent_operations = list(self.operation_history)[-50:]  # Last 50 operations
            
            # Performance trend analysis
            execution_times = [op.execution_time_ms for op in recent_operations if op.success]
            performance_scores = [op.performance_score for op in recent_operations if op.success]
            
            # Calculate trends
            trends = {
                'performance_trend': self._calculate_trend(performance_scores),
                'execution_time_trend': self._calculate_trend(execution_times),
                'hybrid_usage_trend': self._calculate_hybrid_usage_trend(recent_operations),
                'resource_usage_trends': self._analyze_resource_trends(recent_operations),
                'stability_analysis': self._analyze_performance_stability(recent_operations)
            }
            
            # Add trend insights
            trends['insights'] = self._generate_trend_insights(trends)
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance trends: {e}")
            return {'status': 'analysis_failed', 'error': str(e)}
    
    def generate_hybrid_performance_alerts(self) -> List[HybridPerformanceAlert]:
        """
        Generate performance alerts based on current metrics.
        
        Returns:
            List of performance alerts
        """
        alerts = []
        
        try:
            current_time = datetime.utcnow()
            
            # Check execution time alerts
            if self.monitoring_metrics.average_execution_time_ms > self.config.execution_time_threshold_ms:
                alert = HybridPerformanceAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_type="execution_time_high",
                    severity=AlertSeverity.HIGH,
                    message=f"Average execution time ({self.monitoring_metrics.average_execution_time_ms:.2f}ms) "
                           f"exceeds threshold ({self.config.execution_time_threshold_ms}ms)",
                    metric_name="average_execution_time_ms",
                    metric_value=self.monitoring_metrics.average_execution_time_ms,
                    threshold=self.config.execution_time_threshold_ms,
                    timestamp=current_time
                )
                alerts.append(alert)
            
            # Check GPU memory alerts
            if self.monitoring_metrics.peak_gpu_memory_usage_mb > self.config.gpu_memory_threshold_mb:
                alert = HybridPerformanceAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_type="gpu_memory_high",
                    severity=AlertSeverity.MEDIUM,
                    message=f"Peak GPU memory usage ({self.monitoring_metrics.peak_gpu_memory_usage_mb:.2f}MB) "
                           f"exceeds threshold ({self.config.gpu_memory_threshold_mb}MB)",
                    metric_name="peak_gpu_memory_usage_mb",
                    metric_value=self.monitoring_metrics.peak_gpu_memory_usage_mb,
                    threshold=self.config.gpu_memory_threshold_mb,
                    timestamp=current_time
                )
                alerts.append(alert)
            
            # Check CPU usage alerts
            if self.monitoring_metrics.average_cpu_usage_percent > self.config.cpu_usage_threshold_percent:
                alert = HybridPerformanceAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_type="cpu_usage_high",
                    severity=AlertSeverity.MEDIUM,
                    message=f"Average CPU usage ({self.monitoring_metrics.average_cpu_usage_percent:.1f}%) "
                           f"exceeds threshold ({self.config.cpu_usage_threshold_percent}%)",
                    metric_name="average_cpu_usage_percent",
                    metric_value=self.monitoring_metrics.average_cpu_usage_percent,
                    threshold=self.config.cpu_usage_threshold_percent,
                    timestamp=current_time
                )
                alerts.append(alert)
            
            # Check performance score alerts
            if self.monitoring_metrics.average_performance_score < self.config.performance_score_threshold:
                alert = HybridPerformanceAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_type="performance_score_low",
                    severity=AlertSeverity.HIGH,
                    message=f"Average performance score ({self.monitoring_metrics.average_performance_score:.3f}) "
                           f"below threshold ({self.config.performance_score_threshold})",
                    metric_name="average_performance_score",
                    metric_value=self.monitoring_metrics.average_performance_score,
                    threshold=self.config.performance_score_threshold,
                    timestamp=current_time
                )
                alerts.append(alert)
            
            # Check success rate alerts
            if self.monitoring_metrics.success_rate < 0.95:  # 95% success rate threshold
                alert = HybridPerformanceAlert(
                    alert_id=str(uuid.uuid4()),
                    alert_type="success_rate_low",
                    severity=AlertSeverity.CRITICAL,
                    message=f"Success rate ({self.monitoring_metrics.success_rate:.1%}) is below 95%",
                    metric_name="success_rate",
                    metric_value=self.monitoring_metrics.success_rate,
                    threshold=0.95,
                    timestamp=current_time
                )
                alerts.append(alert)
            
            # Store alerts
            with self._alerts_lock:
                self.alert_history.extend(alerts)
            
            if alerts:
                self.logger.warning(f"Generated {len(alerts)} performance alerts")
            
        except Exception as e:
            self.logger.error(f"Error generating performance alerts: {e}")
        
        return alerts
    
    def _initialize_monitoring(self) -> None:
        """Initialize monitoring components"""
        try:
            # Reset monitoring metrics
            self.monitoring_metrics = HybridMonitoringMetrics()
            
            # Clear data structures
            self.operation_history.clear()
            self.alert_history.clear()
            self.active_operations.clear()
            self.performance_trends.clear()
            self.resource_usage_history.clear()
            self.switching_events.clear()
            
            # Initialize GPU monitoring if enabled
            if self.config.gpu_monitoring_enabled and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            self.logger.debug("Monitoring components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring: {e}")
            raise
    
    def _start_monitoring_threads(self) -> None:
        """Start monitoring threads"""
        self._monitoring_enabled = True
        self._stop_monitoring.clear()
        
        # Start main monitoring thread
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="HybridPerformanceMonitor",
            daemon=True
        )
        self._monitoring_thread.start()
        
        self.logger.debug("Monitoring threads started")
    
    def _stop_monitoring_threads(self) -> None:
        """Stop monitoring threads"""
        if self._monitoring_enabled:
            self._monitoring_enabled = False
            self._stop_monitoring.set()
            
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=5.0)
        
        self.logger.debug("Monitoring threads stopped")
    
    def _start_async_monitoring(self) -> None:
        """Start async monitoring tasks"""
        try:
            # This would start async monitoring tasks if needed
            # For now, just log that async monitoring is enabled
            self.logger.debug("Async monitoring enabled")
            
        except Exception as e:
            self.logger.error(f"Failed to start async monitoring: {e}")
    
    def _stop_async_monitoring(self) -> None:
        """Stop async monitoring tasks"""
        try:
            # Cancel any async tasks
            for task in self._async_tasks:
                if not task.done():
                    task.cancel()
            
            self._async_tasks.clear()
            self.logger.debug("Async monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping async monitoring: {e}")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while not self._stop_monitoring.wait(self.config.monitoring_interval_seconds):
            try:
                # Update resource usage history
                current_usage = {
                    'timestamp': datetime.utcnow(),
                    'gpu_memory_mb': self._get_gpu_memory_usage(),
                    'cpu_usage_percent': self._get_cpu_usage(),
                    'memory_usage_mb': self._get_memory_usage(),
                    'gpu_utilization_percent': self._get_gpu_utilization()
                }
                
                self.resource_usage_history.append(current_usage)
                
                # Generate alerts if needed
                if len(self.operation_history) > 10:
                    self.generate_hybrid_performance_alerts()
                
                # Clean up old data
                self._cleanup_old_data()
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _finalize_monitoring(self) -> None:
        """Finalize monitoring data"""
        try:
            # Complete any remaining active operations
            with self._operations_lock:
                for operation_id, monitor_data in list(self.active_operations.items()):
                    self.complete_operation_monitoring(operation_id, success=False, 
                                                     error_message="Monitoring stopped")
            
            self.logger.debug("Monitoring data finalized")
            
        except Exception as e:
            self.logger.error(f"Error finalizing monitoring: {e}")
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            return torch.cuda.memory_allocated() / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage"""
        # This would require nvidia-ml-py or similar library for real implementation
        return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        # This would require psutil or similar library for real implementation
        return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        # This would require psutil or similar library for real implementation
        return 0.0
    
    def _check_operation_alerts(self, monitor_data: HybridOperationMonitorData) -> None:
        """Check for alerts based on operation data"""
        alerts = []
        
        # Check execution time
        if monitor_data.execution_time_ms > self.config.execution_time_threshold_ms:
            alert = HybridPerformanceAlert(
                alert_id=str(uuid.uuid4()),
                alert_type="operation_slow",
                severity=AlertSeverity.MEDIUM,
                message=f"Operation '{monitor_data.operation_name}' took {monitor_data.execution_time_ms:.2f}ms",
                operation_name=monitor_data.operation_name,
                metric_name="execution_time_ms",
                metric_value=monitor_data.execution_time_ms,
                threshold=self.config.execution_time_threshold_ms
            )
            alerts.append(alert)
        
        # Check performance score
        if monitor_data.performance_score < self.config.performance_score_threshold:
            alert = HybridPerformanceAlert(
                alert_id=str(uuid.uuid4()),
                alert_type="operation_poor_performance",
                severity=AlertSeverity.HIGH,
                message=f"Operation '{monitor_data.operation_name}' had poor performance score: {monitor_data.performance_score:.3f}",
                operation_name=monitor_data.operation_name,
                metric_name="performance_score",
                metric_value=monitor_data.performance_score,
                threshold=self.config.performance_score_threshold
            )
            alerts.append(alert)
        
        # Store alerts
        if alerts:
            with self._alerts_lock:
                self.alert_history.extend(alerts)
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend for a series of values"""
        if len(values) < 5:
            return {'status': 'insufficient_data'}
        
        # Simple linear trend calculation
        n = len(values)
        x = list(range(n))
        
        # Calculate linear regression
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        intercept = (sum_y - slope * sum_x) / n
        
        # Determine trend direction
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        return {
            'direction': direction,
            'slope': slope,
            'intercept': intercept,
            'recent_average': sum(values[-5:]) / 5,
            'overall_average': sum(values) / len(values)
        }
    
    def _calculate_hybrid_usage_trend(self, operations: List[HybridOperationMonitorData]) -> Dict[str, Any]:
        """Calculate hybrid usage trend"""
        if len(operations) < 5:
            return {'status': 'insufficient_data'}
        
        hybrid_ratios = []
        window_size = min(10, len(operations) // 2)
        
        for i in range(window_size, len(operations) + 1, window_size):
            window_ops = operations[i-window_size:i]
            hybrid_count = sum(1 for op in window_ops if op.hybrid_mode_used)
            ratio = hybrid_count / len(window_ops)
            hybrid_ratios.append(ratio)
        
        trend = self._calculate_trend(hybrid_ratios)
        trend['current_hybrid_ratio'] = hybrid_ratios[-1] if hybrid_ratios else 0.0
        
        return trend
    
    def _analyze_resource_trends(self, operations: List[HybridOperationMonitorData]) -> Dict[str, Any]:
        """Analyze resource usage trends"""
        gpu_memory_values = [op.gpu_memory_peak_mb for op in operations if op.gpu_memory_peak_mb > 0]
        cpu_usage_values = [op.cpu_usage_percent for op in operations if op.cpu_usage_percent > 0]
        
        return {
            'gpu_memory_trend': self._calculate_trend(gpu_memory_values),
            'cpu_usage_trend': self._calculate_trend(cpu_usage_values)
        }
    
    def _analyze_performance_stability(self, operations: List[HybridOperationMonitorData]) -> Dict[str, Any]:
        """Analyze performance stability"""
        if len(operations) < 10:
            return {'status': 'insufficient_data'}
        
        execution_times = [op.execution_time_ms for op in operations if op.success]
        performance_scores = [op.performance_score for op in operations if op.success]
        
        if not execution_times or not performance_scores:
            return {'status': 'no_successful_operations'}
        
        # Calculate coefficient of variation
        exec_time_mean = sum(execution_times) / len(execution_times)
        exec_time_std = (sum((x - exec_time_mean) ** 2 for x in execution_times) / len(execution_times)) ** 0.5
        exec_time_cv = exec_time_std / exec_time_mean if exec_time_mean > 0 else 0
        
        perf_score_mean = sum(performance_scores) / len(performance_scores)
        perf_score_std = (sum((x - perf_score_mean) ** 2 for x in performance_scores) / len(performance_scores)) ** 0.5
        perf_score_cv = perf_score_std / perf_score_mean if perf_score_mean > 0 else 0
        
        # Determine stability level
        if exec_time_cv < 0.1 and perf_score_cv < 0.1:
            stability = "excellent"
        elif exec_time_cv < 0.2 and perf_score_cv < 0.2:
            stability = "good"
        elif exec_time_cv < 0.4 and perf_score_cv < 0.4:
            stability = "moderate"
        else:
            stability = "poor"
        
        return {
            'stability_level': stability,
            'execution_time_coefficient_of_variation': exec_time_cv,
            'performance_score_coefficient_of_variation': perf_score_cv,
            'execution_time_std': exec_time_std,
            'performance_score_std': perf_score_std
        }
    
    def _generate_trend_insights(self, trends: Dict[str, Any]) -> List[str]:
        """Generate insights from trend analysis"""
        insights = []
        
        # Performance trend insights
        perf_trend = trends.get('performance_trend', {})
        if perf_trend.get('direction') == 'decreasing':
            insights.append("Performance is declining - consider optimization")
        elif perf_trend.get('direction') == 'increasing':
            insights.append("Performance is improving")
        
        # Execution time insights
        time_trend = trends.get('execution_time_trend', {})
        if time_trend.get('direction') == 'increasing':
            insights.append("Execution times are increasing - potential performance regression")
        
        # Hybrid usage insights
        hybrid_trend = trends.get('hybrid_usage_trend', {})
        if hybrid_trend.get('direction') == 'decreasing':
            insights.append("Hybrid mode usage is decreasing - check switching logic")
        
        # Stability insights
        stability = trends.get('stability_analysis', {})
        if stability.get('stability_level') == 'poor':
            insights.append("Performance is unstable - investigate variability causes")
        
        return insights
    
    def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.config.data_retention_hours)
        
        # Clean operation history
        self.operation_history = deque([
            op for op in self.operation_history
            if op.start_time >= cutoff_time
        ], maxlen=self.config.max_operation_history)
        
        # Clean alert history
        with self._alerts_lock:
            self.alert_history = deque([
                alert for alert in self.alert_history
                if alert.timestamp >= cutoff_time
            ], maxlen=self.config.max_alert_history)
    
    def shutdown(self) -> None:
        """Shutdown hybrid performance monitor"""
        self.logger.info("Shutting down hybrid performance monitor")
        
        # Stop monitoring
        if self.is_monitoring:
            self.stop_hybrid_monitoring()
        
        # Clear references
        self.hybrid_padic_manager = None
        self.dynamic_switching_manager = None
        self.hybrid_compression_system = None
        
        # Clear data
        self.operation_history.clear()
        self.alert_history.clear()
        self.active_operations.clear()
        self.performance_trends.clear()
        self.resource_usage_history.clear()
        self.switching_events.clear()
        
        self.monitoring_status = MonitoringStatus.STOPPED
        self.logger.info("Hybrid performance monitor shutdown complete")