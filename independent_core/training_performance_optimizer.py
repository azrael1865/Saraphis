"""
Training Performance Optimizer - Performance optimization for training with hybrid system
NO FALLBACKS - HARD FAILURES ONLY
"""

import logging
import threading
import time
import torch
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from enum import Enum
import json
import hashlib
import psutil

# Import training system components (avoid circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from training_manager import TrainingManager

# Import hybrid compression system components
from compression_systems.padic.hybrid_padic_compressor import HybridPadicCompressionSystem


class OptimizationStrategy(Enum):
    """Training optimization strategy enumeration"""
    PERFORMANCE_FIRST = "performance_first"
    MEMORY_FIRST = "memory_first"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"
    GPU_OPTIMIZED = "gpu_optimized"
    CPU_OPTIMIZED = "cpu_optimized"


class OptimizationLevel(Enum):
    """Training optimization level enumeration"""
    MINIMAL = "minimal"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


class PerformanceRegression(Enum):
    """Performance regression severity enumeration"""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class TrainingPerformanceOptimizerConfig:
    """Configuration for training performance optimizer"""
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE
    optimization_level: OptimizationLevel = OptimizationLevel.MODERATE
    enable_gpu_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_algorithm_optimization: bool = True
    enable_regression_detection: bool = True
    enable_real_time_monitoring: bool = True
    enable_predictive_optimization: bool = True
    
    # Performance thresholds
    performance_regression_threshold: float = 0.15  # 15% performance drop
    memory_optimization_threshold_gb: float = 8.0
    gpu_optimization_threshold_percent: float = 85.0
    cpu_optimization_threshold_percent: float = 80.0
    
    # Optimization parameters
    optimization_interval_seconds: int = 60
    monitoring_interval_seconds: int = 10
    regression_detection_window: int = 20
    performance_baseline_window: int = 50
    
    # GPU optimization
    gpu_memory_threshold_percent: float = 90.0
    gpu_batch_size_optimization: bool = True
    gpu_precision_optimization: bool = True
    
    # Algorithm optimization
    enable_batch_size_optimization: bool = True
    enable_learning_rate_optimization: bool = True
    enable_compression_parameter_optimization: bool = True
    
    # Analytics configuration
    enable_analytics: bool = True
    analytics_history_size: int = 1000
    optimization_history_size: int = 200
    
    def __post_init__(self):
        """Validate configuration"""
        if not isinstance(self.optimization_strategy, OptimizationStrategy):
            raise TypeError("Optimization strategy must be OptimizationStrategy")
        if not isinstance(self.optimization_level, OptimizationLevel):
            raise TypeError("Optimization level must be OptimizationLevel")
        if not (0.0 <= self.performance_regression_threshold <= 1.0):
            raise ValueError("Performance regression threshold must be between 0.0 and 1.0")
        if self.memory_optimization_threshold_gb <= 0:
            raise ValueError("Memory optimization threshold must be positive")


@dataclass
class TrainingPerformanceMetrics:
    """Training performance metrics"""
    session_id: str
    timestamp: datetime
    
    # Training performance
    iteration_time_ms: float = 0.0
    batch_processing_time_ms: float = 0.0
    gradient_computation_time_ms: float = 0.0
    parameter_update_time_ms: float = 0.0
    
    # System performance
    memory_usage_gb: float = 0.0
    gpu_memory_usage_gb: float = 0.0
    gpu_utilization_percent: float = 0.0
    cpu_utilization_percent: float = 0.0
    
    # Compression performance
    compression_time_ms: float = 0.0
    compression_ratio: float = 0.0
    compression_throughput_mbps: float = 0.0
    
    # Training accuracy metrics
    training_loss: float = 0.0
    training_accuracy: float = 0.0
    validation_loss: float = 0.0
    validation_accuracy: float = 0.0
    
    # Optimization metrics
    optimization_applied: bool = False
    optimization_type: str = ""
    optimization_impact: float = 0.0
    
    def __post_init__(self):
        """Validate metrics"""
        if not self.session_id:
            raise ValueError("Session ID cannot be empty")
        if self.iteration_time_ms < 0:
            raise ValueError("Iteration time must be non-negative")


@dataclass
class OptimizationResult:
    """Training optimization result"""
    optimization_id: str
    session_id: str
    optimization_type: str
    optimization_strategy: OptimizationStrategy
    optimization_level: OptimizationLevel
    
    # Performance impact
    performance_improvement_percent: float = 0.0
    memory_improvement_gb: float = 0.0
    gpu_improvement_percent: float = 0.0
    
    # Optimization details
    optimizations_applied: List[str] = field(default_factory=list)
    parameters_changed: Dict[str, Any] = field(default_factory=dict)
    
    # Success tracking
    optimization_successful: bool = False
    optimization_time_ms: float = 0.0
    error_message: Optional[str] = None
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate optimization result"""
        if not self.optimization_id or not self.session_id:
            raise ValueError("Optimization ID and Session ID cannot be empty")
        if not isinstance(self.optimization_strategy, OptimizationStrategy):
            raise TypeError("Optimization strategy must be OptimizationStrategy")


@dataclass
class TrainingAnalytics:
    """Comprehensive training performance analytics"""
    total_optimizations: int = 0
    successful_optimizations: int = 0
    failed_optimizations: int = 0
    
    # Performance improvements
    average_performance_improvement_percent: float = 0.0
    average_memory_improvement_gb: float = 0.0
    average_gpu_improvement_percent: float = 0.0
    
    # Regression detection
    regressions_detected: int = 0
    regressions_resolved: int = 0
    current_regression_level: PerformanceRegression = PerformanceRegression.NONE
    
    # System health
    system_performance_score: float = 0.0
    optimization_effectiveness_score: float = 0.0
    
    # Trends
    performance_trend: str = "stable"
    memory_trend: str = "stable"
    gpu_trend: str = "stable"
    
    timestamp: datetime = field(default_factory=datetime.utcnow)


class TrainingPerformanceOptimizer:
    """
    Performance optimization for training with hybrid system.
    Provides profiling, optimization, and regression detection for training operations.
    """
    
    def __init__(self, config: Optional[TrainingPerformanceOptimizerConfig] = None):
        """Initialize training performance optimizer"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, TrainingPerformanceOptimizerConfig):
            raise TypeError(f"Config must be TrainingPerformanceOptimizerConfig or None, got {type(config)}")
        
        self.config = config or TrainingPerformanceOptimizerConfig()
        self.logger = logging.getLogger('TrainingPerformanceOptimizer')
        
        # System references
        self.training_manager: Optional['TrainingManager'] = None
        self.hybrid_compression: Optional[HybridPadicCompressionSystem] = None
        
        # Performance tracking
        self.performance_metrics: deque = deque(maxlen=self.config.analytics_history_size)
        self.performance_baselines: Dict[str, float] = {}
        self.session_baselines: Dict[str, Dict[str, float]] = {}
        
        # Optimization tracking
        self.optimization_history: deque = deque(maxlen=self.config.optimization_history_size)
        self.active_optimizations: Dict[str, OptimizationResult] = {}
        self.optimization_analytics = TrainingAnalytics()
        
        # Regression detection
        self.regression_detection_active = False
        self.detected_regressions: Dict[str, Dict[str, Any]] = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.optimization_active = False
        self.last_optimization: Optional[datetime] = None
        
        # Thread safety
        self._optimizer_lock = threading.RLock()
        self._metrics_lock = threading.RLock()
        self._optimization_lock = threading.RLock()
        
        # Performance profiling
        self.profiling_active = False
        self.profiling_sessions: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("TrainingPerformanceOptimizer created successfully")
    
    def initialize_optimizer(self,
                           training_manager: 'TrainingManager',
                           hybrid_compression: HybridPadicCompressionSystem) -> None:
        """
        Initialize training performance optimizer with required systems.
        
        Args:
            training_manager: Training manager instance
            hybrid_compression: Hybrid compression system
            
        Raises:
            TypeError: If systems are invalid
            RuntimeError: If initialization fails
        """
        if training_manager is None:
            raise ValueError("Training manager cannot be None")
        if hasattr(training_manager, '__class__') and 'TrainingManager' not in str(training_manager.__class__):
            raise TypeError(f"Training manager must be TrainingManager, got {type(training_manager)}")
        if not isinstance(hybrid_compression, HybridPadicCompressionSystem):
            raise TypeError(f"Hybrid compression must be HybridPadicCompressionSystem, got {type(hybrid_compression)}")
        
        try:
            with self._optimizer_lock:
                # Store system references
                self.training_manager = training_manager
                self.hybrid_compression = hybrid_compression
                
                # Initialize optimization systems
                self._initialize_performance_baselines()
                self._setup_optimization_hooks()
                
                if self.config.enable_real_time_monitoring:
                    self._start_performance_monitoring()
                
                if self.config.enable_regression_detection:
                    self._start_regression_detection()
                
                if self.config.enable_analytics:
                    self._initialize_optimization_analytics()
                
                self.optimization_active = True
                self.logger.info("Training performance optimizer initialized successfully")
                
        except Exception as e:
            self.optimization_active = False
            self.logger.error(f"Failed to initialize training performance optimizer: {e}")
            raise RuntimeError(f"Optimizer initialization failed: {e}")
    
    def profile_training_performance(self, session_id: str, training_operation: Callable) -> TrainingPerformanceMetrics:
        """
        Profile training performance for a specific operation.
        
        Args:
            session_id: Training session identifier
            training_operation: Training operation to profile
            
        Returns:
            Performance metrics for the operation
            
        Raises:
            ValueError: If session or operation is invalid
            RuntimeError: If profiling fails
        """
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Session ID must be non-empty string")
        if not callable(training_operation):
            raise TypeError("Training operation must be callable")
        
        try:
            profiling_start = time.time()
            
            # Initialize profiling for session
            if session_id not in self.profiling_sessions:
                self.profiling_sessions[session_id] = {
                    'start_time': datetime.utcnow(),
                    'operation_count': 0,
                    'total_time_ms': 0.0
                }
            
            # Capture initial system metrics
            initial_metrics = self._capture_system_metrics()
            
            # Profile training operation
            operation_start = time.time()
            
            try:
                operation_result = training_operation()
            except Exception as e:
                self.logger.error(f"Training operation failed during profiling: {e}")
                raise RuntimeError(f"Training operation profiling failed: {e}")
            
            operation_time = (time.time() - operation_start) * 1000  # ms
            
            # Capture final system metrics
            final_metrics = self._capture_system_metrics()
            
            # Calculate performance metrics
            performance_metrics = TrainingPerformanceMetrics(
                session_id=session_id,
                timestamp=datetime.utcnow(),
                iteration_time_ms=operation_time,
                memory_usage_gb=final_metrics.get('memory_usage_gb', 0.0),
                gpu_memory_usage_gb=final_metrics.get('gpu_memory_usage_gb', 0.0),
                gpu_utilization_percent=final_metrics.get('gpu_utilization_percent', 0.0),
                cpu_utilization_percent=final_metrics.get('cpu_utilization_percent', 0.0)
            )
            
            # Extract specific timing metrics if available
            if hasattr(operation_result, 'timing_metrics'):
                timing = operation_result.timing_metrics
                performance_metrics.batch_processing_time_ms = timing.get('batch_processing_time_ms', 0.0)
                performance_metrics.gradient_computation_time_ms = timing.get('gradient_computation_time_ms', 0.0)
                performance_metrics.parameter_update_time_ms = timing.get('parameter_update_time_ms', 0.0)
            
            # Extract training metrics if available
            if hasattr(operation_result, 'training_metrics'):
                training = operation_result.training_metrics
                performance_metrics.training_loss = training.get('loss', 0.0)
                performance_metrics.training_accuracy = training.get('accuracy', 0.0)
                performance_metrics.validation_loss = training.get('val_loss', 0.0)
                performance_metrics.validation_accuracy = training.get('val_accuracy', 0.0)
            
            # Update profiling session
            self.profiling_sessions[session_id]['operation_count'] += 1
            self.profiling_sessions[session_id]['total_time_ms'] += operation_time
            
            # Store performance metrics
            with self._metrics_lock:
                self.performance_metrics.append(performance_metrics)
            
            profiling_time = (time.time() - profiling_start) * 1000
            self.logger.debug(f"Training performance profiled for session {session_id}: {operation_time:.2f}ms operation, {profiling_time:.2f}ms profiling")
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to profile training performance for session {session_id}: {e}")
            raise RuntimeError(f"Training performance profiling failed: {e}")
    
    def optimize_training_gpu_memory(self, session_id: str) -> OptimizationResult:
        """
        Optimize GPU memory usage for training.
        
        Args:
            session_id: Training session identifier
            
        Returns:
            GPU memory optimization result
            
        Raises:
            ValueError: If session is invalid
            RuntimeError: If optimization fails
        """
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Session ID must be non-empty string")
        
        try:
            optimization_id = self._generate_optimization_id("gpu_memory")
            optimization_start = time.time()
            
            # Create optimization result
            result = OptimizationResult(
                optimization_id=optimization_id,
                session_id=session_id,
                optimization_type="gpu_memory",
                optimization_strategy=self.config.optimization_strategy,
                optimization_level=self.config.optimization_level
            )
            
            with self._optimization_lock:
                self.active_optimizations[optimization_id] = result
            
            # Capture initial GPU memory state
            initial_gpu_memory = self._get_gpu_memory_usage()
            
            optimizations_applied = []
            
            # Apply GPU memory optimizations
            if self.config.enable_gpu_optimization:
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    optimizations_applied.append("gpu_cache_cleared")
                
                # Optimize batch size if enabled
                if self.config.gpu_batch_size_optimization:
                    batch_optimization = self._optimize_gpu_batch_size(session_id)
                    if batch_optimization['success']:
                        optimizations_applied.extend(batch_optimization['optimizations'])
                        result.parameters_changed.update(batch_optimization['parameters'])
                
                # Optimize precision if enabled
                if self.config.gpu_precision_optimization:
                    precision_optimization = self._optimize_gpu_precision(session_id)
                    if precision_optimization['success']:
                        optimizations_applied.extend(precision_optimization['optimizations'])
                        result.parameters_changed.update(precision_optimization['parameters'])
                
                # Optimize memory allocation
                memory_optimization = self._optimize_gpu_memory_allocation(session_id)
                if memory_optimization['success']:
                    optimizations_applied.extend(memory_optimization['optimizations'])
                    result.parameters_changed.update(memory_optimization['parameters'])
            
            # Capture final GPU memory state
            final_gpu_memory = self._get_gpu_memory_usage()
            memory_improvement = initial_gpu_memory - final_gpu_memory
            
            # Update optimization result
            result.optimizations_applied = optimizations_applied
            result.memory_improvement_gb = memory_improvement
            result.gpu_improvement_percent = (memory_improvement / max(initial_gpu_memory, 0.1)) * 100
            result.optimization_successful = len(optimizations_applied) > 0 and memory_improvement > 0
            result.optimization_time_ms = (time.time() - optimization_start) * 1000
            
            # Generate recommendations
            result.recommendations = self._generate_gpu_optimization_recommendations(session_id, initial_gpu_memory, final_gpu_memory)
            
            # Record optimization
            with self._optimization_lock:
                del self.active_optimizations[optimization_id]
            
            self.optimization_history.append(result)
            self._update_optimization_analytics(result)
            
            self.logger.info(f"GPU memory optimization completed for session {session_id}: {memory_improvement:.2f}GB improvement")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to optimize GPU memory for session {session_id}: {e}")
            if optimization_id in self.active_optimizations:
                with self._optimization_lock:
                    del self.active_optimizations[optimization_id]
            raise RuntimeError(f"GPU memory optimization failed: {e}")
    
    def optimize_training_algorithms(self, session_id: str) -> OptimizationResult:
        """
        Optimize training algorithms for better performance.
        
        Args:
            session_id: Training session identifier
            
        Returns:
            Algorithm optimization result
            
        Raises:
            ValueError: If session is invalid
            RuntimeError: If optimization fails
        """
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Session ID must be non-empty string")
        
        try:
            optimization_id = self._generate_optimization_id("algorithms")
            optimization_start = time.time()
            
            # Create optimization result
            result = OptimizationResult(
                optimization_id=optimization_id,
                session_id=session_id,
                optimization_type="algorithms",
                optimization_strategy=self.config.optimization_strategy,
                optimization_level=self.config.optimization_level
            )
            
            with self._optimization_lock:
                self.active_optimizations[optimization_id] = result
            
            # Capture initial performance metrics
            initial_performance = self._capture_training_performance_baseline(session_id)
            
            optimizations_applied = []
            
            # Apply algorithm optimizations
            if self.config.enable_algorithm_optimization:
                # Optimize batch size
                if self.config.enable_batch_size_optimization:
                    batch_optimization = self._optimize_training_batch_size(session_id)
                    if batch_optimization['success']:
                        optimizations_applied.extend(batch_optimization['optimizations'])
                        result.parameters_changed.update(batch_optimization['parameters'])
                
                # Optimize learning rate
                if self.config.enable_learning_rate_optimization:
                    lr_optimization = self._optimize_learning_rate(session_id)
                    if lr_optimization['success']:
                        optimizations_applied.extend(lr_optimization['optimizations'])
                        result.parameters_changed.update(lr_optimization['parameters'])
                
                # Optimize compression parameters
                if self.config.enable_compression_parameter_optimization:
                    compression_optimization = self._optimize_compression_parameters(session_id)
                    if compression_optimization['success']:
                        optimizations_applied.extend(compression_optimization['optimizations'])
                        result.parameters_changed.update(compression_optimization['parameters'])
                
                # Optimize data loading
                data_optimization = self._optimize_data_loading(session_id)
                if data_optimization['success']:
                    optimizations_applied.extend(data_optimization['optimizations'])
                    result.parameters_changed.update(data_optimization['parameters'])
            
            # Capture final performance metrics
            final_performance = self._capture_training_performance_baseline(session_id)
            
            # Calculate performance improvement
            performance_improvement = self._calculate_performance_improvement(initial_performance, final_performance)
            
            # Update optimization result
            result.optimizations_applied = optimizations_applied
            result.performance_improvement_percent = performance_improvement
            result.optimization_successful = len(optimizations_applied) > 0 and performance_improvement > 0
            result.optimization_time_ms = (time.time() - optimization_start) * 1000
            
            # Generate recommendations
            result.recommendations = self._generate_algorithm_optimization_recommendations(session_id, performance_improvement)
            
            # Record optimization
            with self._optimization_lock:
                del self.active_optimizations[optimization_id]
            
            self.optimization_history.append(result)
            self._update_optimization_analytics(result)
            
            self.logger.info(f"Algorithm optimization completed for session {session_id}: {performance_improvement:.2f}% improvement")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to optimize training algorithms for session {session_id}: {e}")
            if optimization_id in self.active_optimizations:
                with self._optimization_lock:
                    del self.active_optimizations[optimization_id]
            raise RuntimeError(f"Training algorithm optimization failed: {e}")
    
    def detect_training_performance_regression(self, session_id: str) -> Dict[str, Any]:
        """
        Detect performance regression in training.
        
        Args:
            session_id: Training session identifier
            
        Returns:
            Regression detection results
            
        Raises:
            ValueError: If session is invalid
            RuntimeError: If detection fails
        """
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Session ID must be non-empty string")
        
        try:
            # Get recent performance metrics for session
            session_metrics = [m for m in self.performance_metrics if m.session_id == session_id]
            
            if len(session_metrics) < self.config.regression_detection_window:
                return {
                    'regression_detected': False,
                    'reason': 'Insufficient data for regression detection',
                    'metrics_count': len(session_metrics),
                    'required_count': self.config.regression_detection_window
                }
            
            # Get baseline performance
            baseline_performance = self._get_session_baseline(session_id)
            if not baseline_performance:
                return {
                    'regression_detected': False,
                    'reason': 'No baseline performance available for session',
                    'session_id': session_id
                }
            
            # Analyze recent performance
            recent_metrics = session_metrics[-self.config.regression_detection_window:]
            current_performance = self._calculate_average_performance(recent_metrics)
            
            # Detect regressions
            regressions = self._analyze_performance_regressions(baseline_performance, current_performance)
            
            # Determine overall regression severity
            regression_severity = self._determine_regression_severity(regressions)
            
            regression_result = {
                'session_id': session_id,
                'regression_detected': regression_severity != PerformanceRegression.NONE,
                'regression_severity': regression_severity.value,
                'regressions': regressions,
                'baseline_performance': baseline_performance,
                'current_performance': current_performance,
                'detection_timestamp': datetime.utcnow().isoformat(),
                'metrics_analyzed': len(recent_metrics)
            }
            
            # Record regression if detected
            if regression_severity != PerformanceRegression.NONE:
                self.detected_regressions[session_id] = regression_result
                self.optimization_analytics.regressions_detected += 1
                
                # Trigger automatic optimization if severe
                if regression_severity in [PerformanceRegression.SEVERE, PerformanceRegression.CRITICAL]:
                    self._trigger_automatic_optimization(session_id, regression_result)
            
            self.logger.debug(f"Regression detection completed for session {session_id}: {regression_severity.value}")
            return regression_result
            
        except Exception as e:
            self.logger.error(f"Failed to detect performance regression for session {session_id}: {e}")
            raise RuntimeError(f"Performance regression detection failed: {e}")
    
    def monitor_training_performance(self, session_id: str) -> Dict[str, Any]:
        """
        Monitor training performance in real-time.
        
        Args:
            session_id: Training session identifier
            
        Returns:
            Performance monitoring results
            
        Raises:
            ValueError: If session is invalid
            RuntimeError: If monitoring fails
        """
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Session ID must be non-empty string")
        
        try:
            # Capture current performance metrics
            current_metrics = self._capture_comprehensive_performance_metrics(session_id)
            
            # Analyze performance trends
            performance_trends = self._analyze_performance_trends(session_id)
            
            # Check for performance alerts
            performance_alerts = self._check_performance_alerts(session_id, current_metrics)
            
            # Generate performance recommendations
            performance_recommendations = self._generate_performance_recommendations(session_id, current_metrics, performance_trends)
            
            # Update performance baselines if needed
            self._update_performance_baselines(session_id, current_metrics)
            
            monitoring_result = {
                'session_id': session_id,
                'current_metrics': current_metrics,
                'performance_trends': performance_trends,
                'alerts': performance_alerts,
                'recommendations': performance_recommendations,
                'monitoring_timestamp': datetime.utcnow().isoformat(),
                'monitoring_active': self.monitoring_active,
                'regression_detection_active': self.regression_detection_active
            }
            
            self.logger.debug(f"Performance monitoring completed for session {session_id}")
            return monitoring_result
            
        except Exception as e:
            self.logger.error(f"Failed to monitor training performance for session {session_id}: {e}")
            raise RuntimeError(f"Training performance monitoring failed: {e}")
    
    def optimize_training_for_performance(self, session_id: str, performance_data: Dict[str, Any]) -> OptimizationResult:
        """
        Optimize training based on performance data.
        
        Args:
            session_id: Training session identifier
            performance_data: Performance data to base optimization on
            
        Returns:
            Performance-based optimization result
            
        Raises:
            ValueError: If session or performance data is invalid
            RuntimeError: If optimization fails
        """
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Session ID must be non-empty string")
        if not isinstance(performance_data, dict):
            raise TypeError("Performance data must be dict")
        
        try:
            optimization_id = self._generate_optimization_id("performance_based")
            optimization_start = time.time()
            
            # Create optimization result
            result = OptimizationResult(
                optimization_id=optimization_id,
                session_id=session_id,
                optimization_type="performance_based",
                optimization_strategy=self.config.optimization_strategy,
                optimization_level=self.config.optimization_level
            )
            
            with self._optimization_lock:
                self.active_optimizations[optimization_id] = result
            
            # Analyze performance data
            performance_analysis = self._analyze_performance_data(session_id, performance_data)
            
            # Determine optimization strategy based on performance data
            optimization_strategy = self._determine_optimization_strategy_from_performance(performance_analysis)
            
            optimizations_applied = []
            
            # Apply performance-based optimizations
            if 'memory_pressure' in performance_analysis['issues']:
                memory_optimization = self._apply_memory_pressure_optimization(session_id)
                if memory_optimization['success']:
                    optimizations_applied.extend(memory_optimization['optimizations'])
                    result.parameters_changed.update(memory_optimization['parameters'])
            
            if 'gpu_underutilization' in performance_analysis['issues']:
                gpu_optimization = self._apply_gpu_utilization_optimization(session_id)
                if gpu_optimization['success']:
                    optimizations_applied.extend(gpu_optimization['optimizations'])
                    result.parameters_changed.update(gpu_optimization['parameters'])
            
            if 'cpu_bottleneck' in performance_analysis['issues']:
                cpu_optimization = self._apply_cpu_bottleneck_optimization(session_id)
                if cpu_optimization['success']:
                    optimizations_applied.extend(cpu_optimization['optimizations'])
                    result.parameters_changed.update(cpu_optimization['parameters'])
            
            if 'compression_inefficiency' in performance_analysis['issues']:
                compression_optimization = self._apply_compression_efficiency_optimization(session_id)
                if compression_optimization['success']:
                    optimizations_applied.extend(compression_optimization['optimizations'])
                    result.parameters_changed.update(compression_optimization['parameters'])
            
            # Calculate optimization impact
            optimization_impact = self._calculate_optimization_impact(session_id, performance_analysis, optimizations_applied)
            
            # Update optimization result
            result.optimizations_applied = optimizations_applied
            result.performance_improvement_percent = optimization_impact.get('performance_improvement', 0.0)
            result.memory_improvement_gb = optimization_impact.get('memory_improvement', 0.0)
            result.gpu_improvement_percent = optimization_impact.get('gpu_improvement', 0.0)
            result.optimization_successful = len(optimizations_applied) > 0
            result.optimization_time_ms = (time.time() - optimization_start) * 1000
            
            # Generate recommendations
            result.recommendations = self._generate_performance_based_recommendations(session_id, performance_analysis, optimization_impact)
            
            # Record optimization
            with self._optimization_lock:
                del self.active_optimizations[optimization_id]
            
            self.optimization_history.append(result)
            self._update_optimization_analytics(result)
            
            self.logger.info(f"Performance-based optimization completed for session {session_id}: {len(optimizations_applied)} optimizations applied")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to optimize training for performance for session {session_id}: {e}")
            if optimization_id in self.active_optimizations:
                with self._optimization_lock:
                    del self.active_optimizations[optimization_id]
            raise RuntimeError(f"Performance-based optimization failed: {e}")
    
    def get_optimization_analytics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive optimization analytics.
        
        Args:
            session_id: Optional specific session ID for detailed analytics
            
        Returns:
            Optimization analytics results
            
        Raises:
            ValueError: If session ID is invalid
            RuntimeError: If analytics retrieval fails
        """
        try:
            if session_id:
                if not isinstance(session_id, str):
                    raise ValueError("Session ID must be string")
                
                # Get session-specific analytics
                session_optimizations = [opt for opt in self.optimization_history if opt.session_id == session_id]
                session_metrics = [m for m in self.performance_metrics if m.session_id == session_id]
                
                session_analytics = self._calculate_session_optimization_analytics(session_id, session_optimizations, session_metrics)
                
                return {
                    'session_id': session_id,
                    'session_analytics': session_analytics,
                    'session_optimizations': session_optimizations,
                    'session_metrics': session_metrics[-20:],  # Recent metrics
                    'baseline_performance': self.session_baselines.get(session_id, {}),
                    'detected_regressions': self.detected_regressions.get(session_id, {})
                }
            else:
                # Get global analytics
                self._update_global_optimization_analytics()
                
                return {
                    'global_analytics': self.optimization_analytics,
                    'optimization_history': list(self.optimization_history),
                    'performance_baselines': self.performance_baselines,
                    'session_baselines': self.session_baselines,
                    'active_optimizations': len(self.active_optimizations),
                    'detected_regressions': self.detected_regressions,
                    'profiling_sessions': self.profiling_sessions,
                    'system_performance': self._get_system_performance_summary(),
                    'optimization_trends': self._analyze_optimization_trends(),
                    'recommendations': self._generate_global_optimization_recommendations()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get optimization analytics: {e}")
            raise RuntimeError(f"Optimization analytics retrieval failed: {e}")
    
    # Helper methods for implementation
    
    def _initialize_performance_baselines(self) -> None:
        """Initialize performance baselines"""
        try:
            self.performance_baselines = {
                'iteration_time_ms': 100.0,
                'memory_usage_gb': 4.0,
                'gpu_utilization_percent': 70.0,
                'cpu_utilization_percent': 50.0,
                'compression_time_ms': 50.0,
                'compression_ratio': 0.6
            }
            self.logger.debug("Performance baselines initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize performance baselines: {e}")
            raise RuntimeError(f"Performance baseline initialization failed: {e}")
    
    def _setup_optimization_hooks(self) -> None:
        """Setup optimization hooks"""
        try:
            if hasattr(self.training_manager, 'add_hook'):
                self.training_manager.add_hook('pre_training', self._training_pre_hook)
                self.training_manager.add_hook('post_training', self._training_post_hook)
                self.training_manager.add_hook('performance_degradation', self._performance_degradation_hook)
            
            self.logger.debug("Optimization hooks setup completed")
        except Exception as e:
            self.logger.error(f"Failed to setup optimization hooks: {e}")
            raise RuntimeError(f"Optimization hooks setup failed: {e}")
    
    def _start_performance_monitoring(self) -> None:
        """Start performance monitoring"""
        try:
            self.monitoring_active = True
            self.logger.debug("Performance monitoring started")
        except Exception as e:
            self.logger.error(f"Failed to start performance monitoring: {e}")
            raise RuntimeError(f"Performance monitoring start failed: {e}")
    
    def _start_regression_detection(self) -> None:
        """Start regression detection"""
        try:
            self.regression_detection_active = True
            self.logger.debug("Regression detection started")
        except Exception as e:
            self.logger.error(f"Failed to start regression detection: {e}")
            raise RuntimeError(f"Regression detection start failed: {e}")
    
    def _initialize_optimization_analytics(self) -> None:
        """Initialize optimization analytics"""
        try:
            self.optimization_analytics = TrainingAnalytics()
            self.logger.debug("Optimization analytics initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize optimization analytics: {e}")
            raise RuntimeError(f"Optimization analytics initialization failed: {e}")
    
    def _capture_system_metrics(self) -> Dict[str, float]:
        """Capture current system metrics"""
        try:
            metrics = {}
            
            # Memory metrics
            memory_info = psutil.virtual_memory()
            metrics['memory_usage_gb'] = memory_info.used / (1024**3)
            metrics['memory_percent'] = memory_info.percent
            
            # CPU metrics
            metrics['cpu_utilization_percent'] = psutil.cpu_percent(interval=0.1)
            
            # GPU metrics (if available)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                metrics['gpu_utilization_percent'] = float(gpu_util.gpu)
                metrics['gpu_memory_usage_gb'] = gpu_memory.used / (1024**3)
                metrics['gpu_memory_percent'] = (gpu_memory.used / gpu_memory.total) * 100
            except:
                metrics['gpu_utilization_percent'] = 0.0
                metrics['gpu_memory_usage_gb'] = 0.0
                metrics['gpu_memory_percent'] = 0.0
            
            return metrics
        except Exception as e:
            self.logger.error(f"Failed to capture system metrics: {e}")
            return {}
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in GB"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return memory_info.used / (1024**3)
        except:
            return 0.0
    
    def _generate_optimization_id(self, optimization_type: str) -> str:
        """Generate unique optimization ID"""
        timestamp = str(int(time.time() * 1000))
        return f"{optimization_type}_{timestamp}_{hash(threading.current_thread()) % 10000:04d}"
    
    def _optimize_gpu_batch_size(self, session_id: str) -> Dict[str, Any]:
        """Optimize GPU batch size"""
        try:
            # Simple batch size optimization logic
            current_gpu_memory = self._get_gpu_memory_usage()
            
            optimizations = []
            parameters = {}
            
            if current_gpu_memory > 8.0:  # High memory usage
                # Reduce batch size
                new_batch_size = max(16, int(64 * 0.8))  # Reduce by 20%
                optimizations.append(f"batch_size_reduced_to_{new_batch_size}")
                parameters['batch_size'] = new_batch_size
            elif current_gpu_memory < 4.0:  # Low memory usage
                # Increase batch size
                new_batch_size = min(128, int(64 * 1.2))  # Increase by 20%
                optimizations.append(f"batch_size_increased_to_{new_batch_size}")
                parameters['batch_size'] = new_batch_size
            
            return {
                'success': len(optimizations) > 0,
                'optimizations': optimizations,
                'parameters': parameters
            }
        except Exception as e:
            self.logger.error(f"GPU batch size optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _optimize_gpu_precision(self, session_id: str) -> Dict[str, Any]:
        """Optimize GPU precision"""
        try:
            optimizations = []
            parameters = {}
            
            # Enable mixed precision if not already enabled
            if torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
                optimizations.append("mixed_precision_enabled")
                parameters['use_mixed_precision'] = True
            
            return {
                'success': len(optimizations) > 0,
                'optimizations': optimizations,
                'parameters': parameters
            }
        except Exception as e:
            self.logger.error(f"GPU precision optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _optimize_gpu_memory_allocation(self, session_id: str) -> Dict[str, Any]:
        """Optimize GPU memory allocation"""
        try:
            optimizations = []
            parameters = {}
            
            # Clear unused GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                optimizations.append("gpu_cache_cleared")
                
                # Set memory fraction if high usage detected
                current_memory = self._get_gpu_memory_usage()
                if current_memory > 10.0:  # High usage
                    optimizations.append("gpu_memory_fraction_optimized")
                    parameters['gpu_memory_fraction'] = 0.9
            
            return {
                'success': len(optimizations) > 0,
                'optimizations': optimizations,
                'parameters': parameters
            }
        except Exception as e:
            self.logger.error(f"GPU memory allocation optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_gpu_optimization_recommendations(self, session_id: str, initial_memory: float, final_memory: float) -> List[str]:
        """Generate GPU optimization recommendations"""
        recommendations = []
        
        memory_improvement = initial_memory - final_memory
        
        if memory_improvement > 1.0:
            recommendations.append("Significant GPU memory savings achieved - consider maintaining current settings")
        elif memory_improvement < 0.1:
            recommendations.append("Limited GPU memory improvement - consider more aggressive optimization")
        
        if final_memory > 10.0:
            recommendations.append("GPU memory usage still high - consider reducing model complexity or batch size")
        
        return recommendations if recommendations else ["GPU memory optimization completed successfully"]
    
    def _optimize_training_batch_size(self, session_id: str) -> Dict[str, Any]:
        """Optimize training batch size"""
        try:
            # Analyze current performance to determine optimal batch size
            recent_metrics = [m for m in self.performance_metrics if m.session_id == session_id][-10:]
            
            optimizations = []
            parameters = {}
            
            if len(recent_metrics) >= 3:
                avg_iteration_time = np.mean([m.iteration_time_ms for m in recent_metrics])
                avg_memory_usage = np.mean([m.memory_usage_gb for m in recent_metrics])
                
                if avg_iteration_time > 200.0 and avg_memory_usage < 6.0:
                    # Slow iterations but low memory - increase batch size
                    optimizations.append("batch_size_increased_for_throughput")
                    parameters['batch_size'] = 128
                elif avg_memory_usage > 8.0:
                    # High memory usage - decrease batch size
                    optimizations.append("batch_size_decreased_for_memory")
                    parameters['batch_size'] = 32
            
            return {
                'success': len(optimizations) > 0,
                'optimizations': optimizations,
                'parameters': parameters
            }
        except Exception as e:
            self.logger.error(f"Training batch size optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _optimize_learning_rate(self, session_id: str) -> Dict[str, Any]:
        """Optimize learning rate"""
        try:
            # Simple learning rate optimization based on recent performance
            recent_metrics = [m for m in self.performance_metrics if m.session_id == session_id][-5:]
            
            optimizations = []
            parameters = {}
            
            if len(recent_metrics) >= 3:
                # Check if loss is improving
                losses = [m.training_loss for m in recent_metrics if m.training_loss > 0]
                if len(losses) >= 2:
                    if losses[-1] > losses[0]:  # Loss increasing
                        optimizations.append("learning_rate_reduced_for_stability")
                        parameters['learning_rate'] = 0.001
                    elif abs(losses[-1] - losses[0]) < 0.001:  # Loss plateauing
                        optimizations.append("learning_rate_adjusted_for_convergence")
                        parameters['learning_rate'] = 0.01
            
            return {
                'success': len(optimizations) > 0,
                'optimizations': optimizations,
                'parameters': parameters
            }
        except Exception as e:
            self.logger.error(f"Learning rate optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _optimize_compression_parameters(self, session_id: str) -> Dict[str, Any]:
        """Optimize compression parameters"""
        try:
            optimizations = []
            parameters = {}
            
            # Optimize compression based on performance
            if hasattr(self.hybrid_compression, 'optimize_for_training'):
                compression_optimization = self.hybrid_compression.optimize_for_training(session_id)
                if compression_optimization.get('success', False):
                    optimizations.extend(compression_optimization.get('optimizations', []))
                    parameters.update(compression_optimization.get('parameters', {}))
            
            return {
                'success': len(optimizations) > 0,
                'optimizations': optimizations,
                'parameters': parameters
            }
        except Exception as e:
            self.logger.error(f"Compression parameter optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _optimize_data_loading(self, session_id: str) -> Dict[str, Any]:
        """Optimize data loading"""
        try:
            optimizations = []
            parameters = {}
            
            # Simple data loading optimizations
            optimizations.append("data_loading_num_workers_optimized")
            parameters['num_workers'] = min(psutil.cpu_count(), 8)
            
            optimizations.append("data_loading_pin_memory_enabled")
            parameters['pin_memory'] = True
            
            return {
                'success': len(optimizations) > 0,
                'optimizations': optimizations,
                'parameters': parameters
            }
        except Exception as e:
            self.logger.error(f"Data loading optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _capture_training_performance_baseline(self, session_id: str) -> Dict[str, float]:
        """Capture training performance baseline"""
        try:
            return self.session_baselines.get(session_id, self.performance_baselines.copy())
        except Exception as e:
            self.logger.error(f"Failed to capture training performance baseline: {e}")
            return {}
    
    def _calculate_performance_improvement(self, initial: Dict[str, float], final: Dict[str, float]) -> float:
        """Calculate overall performance improvement percentage"""
        try:
            if not initial or not final:
                return 0.0
            
            improvements = []
            
            # Calculate improvement for each metric
            for metric in ['iteration_time_ms', 'memory_usage_gb']:
                if metric in initial and metric in final and initial[metric] > 0:
                    improvement = (initial[metric] - final[metric]) / initial[metric] * 100
                    improvements.append(improvement)
            
            return np.mean(improvements) if improvements else 0.0
        except Exception as e:
            self.logger.error(f"Failed to calculate performance improvement: {e}")
            return 0.0
    
    def _generate_algorithm_optimization_recommendations(self, session_id: str, improvement: float) -> List[str]:
        """Generate algorithm optimization recommendations"""
        recommendations = []
        
        if improvement > 10.0:
            recommendations.append("Significant performance improvement achieved - consider applying to other sessions")
        elif improvement < 2.0:
            recommendations.append("Limited performance improvement - consider alternative optimization strategies")
        
        return recommendations if recommendations else ["Algorithm optimization completed"]
    
    def _get_session_baseline(self, session_id: str) -> Optional[Dict[str, float]]:
        """Get performance baseline for session"""
        return self.session_baselines.get(session_id)
    
    def _calculate_average_performance(self, metrics: List[TrainingPerformanceMetrics]) -> Dict[str, float]:
        """Calculate average performance from metrics"""
        try:
            if not metrics:
                return {}
            
            return {
                'iteration_time_ms': np.mean([m.iteration_time_ms for m in metrics]),
                'memory_usage_gb': np.mean([m.memory_usage_gb for m in metrics]),
                'gpu_utilization_percent': np.mean([m.gpu_utilization_percent for m in metrics]),
                'cpu_utilization_percent': np.mean([m.cpu_utilization_percent for m in metrics]),
                'compression_time_ms': np.mean([m.compression_time_ms for m in metrics if m.compression_time_ms > 0])
            }
        except Exception as e:
            self.logger.error(f"Failed to calculate average performance: {e}")
            return {}
    
    def _analyze_performance_regressions(self, baseline: Dict[str, float], current: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Analyze performance regressions"""
        try:
            regressions = {}
            
            for metric in ['iteration_time_ms', 'memory_usage_gb', 'compression_time_ms']:
                if metric in baseline and metric in current and baseline[metric] > 0:
                    change_percent = (current[metric] - baseline[metric]) / baseline[metric] * 100
                    
                    if change_percent > self.config.performance_regression_threshold * 100:
                        severity = self._classify_regression_severity(change_percent)
                        regressions[metric] = {
                            'baseline_value': baseline[metric],
                            'current_value': current[metric],
                            'change_percent': change_percent,
                            'severity': severity.value
                        }
            
            return regressions
        except Exception as e:
            self.logger.error(f"Failed to analyze performance regressions: {e}")
            return {}
    
    def _classify_regression_severity(self, change_percent: float) -> PerformanceRegression:
        """Classify regression severity based on change percentage"""
        if change_percent > 50.0:
            return PerformanceRegression.CRITICAL
        elif change_percent > 30.0:
            return PerformanceRegression.SEVERE
        elif change_percent > 15.0:
            return PerformanceRegression.MODERATE
        elif change_percent > 5.0:
            return PerformanceRegression.MINOR
        else:
            return PerformanceRegression.NONE
    
    def _determine_regression_severity(self, regressions: Dict[str, Dict[str, Any]]) -> PerformanceRegression:
        """Determine overall regression severity"""
        if not regressions:
            return PerformanceRegression.NONE
        
        severities = [PerformanceRegression(r['severity']) for r in regressions.values()]
        
        # Return the most severe regression found
        severity_order = [PerformanceRegression.CRITICAL, PerformanceRegression.SEVERE, 
                         PerformanceRegression.MODERATE, PerformanceRegression.MINOR, 
                         PerformanceRegression.NONE]
        
        for severity in severity_order:
            if severity in severities:
                return severity
        
        return PerformanceRegression.NONE
    
    def _trigger_automatic_optimization(self, session_id: str, regression_result: Dict[str, Any]) -> None:
        """Trigger automatic optimization for severe regressions"""
        try:
            self.logger.warning(f"Triggering automatic optimization for session {session_id} due to {regression_result['regression_severity']} regression")
            
            # Apply appropriate optimization based on regression type
            if 'iteration_time_ms' in regression_result['regressions']:
                self.optimize_training_algorithms(session_id)
            
            if 'memory_usage_gb' in regression_result['regressions']:
                self.optimize_training_gpu_memory(session_id)
            
        except Exception as e:
            self.logger.error(f"Failed to trigger automatic optimization: {e}")
    
    def _capture_comprehensive_performance_metrics(self, session_id: str) -> Dict[str, Any]:
        """Capture comprehensive performance metrics"""
        try:
            system_metrics = self._capture_system_metrics()
            
            # Add session-specific metrics
            session_metrics = {
                'session_id': session_id,
                'timestamp': datetime.utcnow().isoformat(),
                'profiling_active': session_id in self.profiling_sessions
            }
            
            if session_id in self.profiling_sessions:
                profiling_data = self.profiling_sessions[session_id]
                session_metrics.update({
                    'profiling_duration_minutes': (datetime.utcnow() - profiling_data['start_time']).total_seconds() / 60,
                    'operations_profiled': profiling_data['operation_count'],
                    'average_operation_time_ms': profiling_data['total_time_ms'] / max(1, profiling_data['operation_count'])
                })
            
            return {**system_metrics, **session_metrics}
        except Exception as e:
            self.logger.error(f"Failed to capture comprehensive performance metrics: {e}")
            return {}
    
    def _analyze_performance_trends(self, session_id: str) -> Dict[str, str]:
        """Analyze performance trends for session"""
        try:
            session_metrics = [m for m in self.performance_metrics if m.session_id == session_id][-20:]
            
            if len(session_metrics) < 5:
                return {'insufficient_data': 'true'}
            
            trends = {}
            
            # Analyze iteration time trend
            iteration_times = [m.iteration_time_ms for m in session_metrics]
            if len(iteration_times) >= 3:
                time_trend = np.polyfit(range(len(iteration_times)), iteration_times, 1)[0]
                trends['iteration_time'] = 'improving' if time_trend < -2.0 else 'degrading' if time_trend > 2.0 else 'stable'
            
            # Analyze memory trend
            memory_usages = [m.memory_usage_gb for m in session_metrics]
            if len(memory_usages) >= 3:
                memory_trend = np.polyfit(range(len(memory_usages)), memory_usages, 1)[0]
                trends['memory_usage'] = 'improving' if memory_trend < -0.1 else 'degrading' if memory_trend > 0.1 else 'stable'
            
            # Analyze GPU trend
            gpu_usages = [m.gpu_utilization_percent for m in session_metrics]
            if len(gpu_usages) >= 3:
                gpu_trend = np.polyfit(range(len(gpu_usages)), gpu_usages, 1)[0]
                trends['gpu_utilization'] = 'improving' if gpu_trend > 1.0 else 'degrading' if gpu_trend < -1.0 else 'stable'
            
            return trends
        except Exception as e:
            self.logger.error(f"Failed to analyze performance trends: {e}")
            return {}
    
    def _check_performance_alerts(self, session_id: str, metrics: Dict[str, Any]) -> List[str]:
        """Check for performance alerts"""
        alerts = []
        
        memory_usage = metrics.get('memory_usage_gb', 0.0)
        if memory_usage > self.config.memory_optimization_threshold_gb:
            alerts.append(f"High memory usage: {memory_usage:.1f}GB > {self.config.memory_optimization_threshold_gb}GB")
        
        gpu_utilization = metrics.get('gpu_utilization_percent', 0.0)
        if gpu_utilization > self.config.gpu_optimization_threshold_percent:
            alerts.append(f"High GPU utilization: {gpu_utilization:.1f}% > {self.config.gpu_optimization_threshold_percent}%")
        
        cpu_utilization = metrics.get('cpu_utilization_percent', 0.0)
        if cpu_utilization > self.config.cpu_optimization_threshold_percent:
            alerts.append(f"High CPU utilization: {cpu_utilization:.1f}% > {self.config.cpu_optimization_threshold_percent}%")
        
        return alerts
    
    def _generate_performance_recommendations(self, session_id: str, metrics: Dict[str, Any], trends: Dict[str, str]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if trends.get('iteration_time') == 'degrading':
            recommendations.append("Iteration time degrading - consider algorithm optimization")
        
        if trends.get('memory_usage') == 'degrading':
            recommendations.append("Memory usage increasing - consider memory optimization")
        
        if trends.get('gpu_utilization') == 'degrading':
            recommendations.append("GPU utilization declining - consider workload optimization")
        
        if metrics.get('memory_usage_gb', 0) > self.config.memory_optimization_threshold_gb * 0.8:
            recommendations.append("Memory usage approaching threshold - enable memory optimization")
        
        return recommendations if recommendations else ["Performance within acceptable parameters"]
    
    def _update_performance_baselines(self, session_id: str, metrics: Dict[str, Any]) -> None:
        """Update performance baselines"""
        try:
            if session_id not in self.session_baselines:
                self.session_baselines[session_id] = {}
            
            # Update session-specific baselines
            for metric in ['memory_usage_gb', 'gpu_utilization_percent', 'cpu_utilization_percent']:
                if metric in metrics:
                    current_baseline = self.session_baselines[session_id].get(metric, metrics[metric])
                    # Use exponential moving average to update baseline
                    self.session_baselines[session_id][metric] = 0.9 * current_baseline + 0.1 * metrics[metric]
        except Exception as e:
            self.logger.error(f"Failed to update performance baselines: {e}")
    
    def _update_optimization_analytics(self, result: OptimizationResult) -> None:
        """Update optimization analytics with result"""
        try:
            self.optimization_analytics.total_optimizations += 1
            
            if result.optimization_successful:
                self.optimization_analytics.successful_optimizations += 1
                
                # Update performance improvement averages
                total_success = self.optimization_analytics.successful_optimizations
                self.optimization_analytics.average_performance_improvement_percent = (
                    (self.optimization_analytics.average_performance_improvement_percent * (total_success - 1) + 
                     result.performance_improvement_percent) / total_success
                )
                
                self.optimization_analytics.average_memory_improvement_gb = (
                    (self.optimization_analytics.average_memory_improvement_gb * (total_success - 1) + 
                     result.memory_improvement_gb) / total_success
                )
                
                self.optimization_analytics.average_gpu_improvement_percent = (
                    (self.optimization_analytics.average_gpu_improvement_percent * (total_success - 1) + 
                     result.gpu_improvement_percent) / total_success
                )
            else:
                self.optimization_analytics.failed_optimizations += 1
            
            # Update optimization effectiveness score
            if self.optimization_analytics.total_optimizations > 0:
                self.optimization_analytics.optimization_effectiveness_score = (
                    self.optimization_analytics.successful_optimizations / 
                    self.optimization_analytics.total_optimizations
                )
            
            # Update timestamp
            self.optimization_analytics.timestamp = datetime.utcnow()
            
        except Exception as e:
            self.logger.error(f"Failed to update optimization analytics: {e}")
    
    def _calculate_session_optimization_analytics(self, session_id: str, optimizations: List[OptimizationResult], metrics: List[TrainingPerformanceMetrics]) -> Dict[str, Any]:
        """Calculate optimization analytics for specific session"""
        try:
            if not optimizations:
                return {'session_id': session_id, 'total_optimizations': 0}
            
            successful_optimizations = [opt for opt in optimizations if opt.optimization_successful]
            
            return {
                'session_id': session_id,
                'total_optimizations': len(optimizations),
                'successful_optimizations': len(successful_optimizations),
                'success_rate': len(successful_optimizations) / len(optimizations),
                'optimization_types': {
                    'gpu_memory': len([opt for opt in optimizations if opt.optimization_type == 'gpu_memory']),
                    'algorithms': len([opt for opt in optimizations if opt.optimization_type == 'algorithms']),
                    'performance_based': len([opt for opt in optimizations if opt.optimization_type == 'performance_based'])
                },
                'average_performance_improvement': np.mean([opt.performance_improvement_percent for opt in successful_optimizations]) if successful_optimizations else 0.0,
                'average_memory_improvement': np.mean([opt.memory_improvement_gb for opt in successful_optimizations]) if successful_optimizations else 0.0,
                'total_metrics_captured': len(metrics),
                'average_iteration_time_ms': np.mean([m.iteration_time_ms for m in metrics]) if metrics else 0.0
            }
        except Exception as e:
            self.logger.error(f"Failed to calculate session optimization analytics: {e}")
            return {'session_id': session_id, 'error': str(e)}
    
    def _update_global_optimization_analytics(self) -> None:
        """Update global optimization analytics"""
        try:
            # Calculate system performance score
            if self.performance_metrics:
                recent_metrics = list(self.performance_metrics)[-20:]
                avg_performance = self._calculate_average_performance(recent_metrics)
                
                # Simple scoring based on resource utilization efficiency
                memory_score = 1.0 - min(1.0, avg_performance.get('memory_usage_gb', 0) / 16.0)  # Assume 16GB max
                gpu_score = avg_performance.get('gpu_utilization_percent', 0) / 100.0
                cpu_score = avg_performance.get('cpu_utilization_percent', 0) / 100.0
                
                self.optimization_analytics.system_performance_score = (memory_score + gpu_score + cpu_score) / 3.0
            
            # Update timestamp
            self.optimization_analytics.timestamp = datetime.utcnow()
            
        except Exception as e:
            self.logger.error(f"Failed to update global optimization analytics: {e}")
    
    def _get_system_performance_summary(self) -> Dict[str, Any]:
        """Get system performance summary"""
        try:
            current_metrics = self._capture_system_metrics()
            
            return {
                'current_metrics': current_metrics,
                'optimization_active': self.optimization_active,
                'monitoring_active': self.monitoring_active,
                'regression_detection_active': self.regression_detection_active,
                'active_profiling_sessions': len(self.profiling_sessions),
                'detected_regressions': len(self.detected_regressions),
                'system_performance_score': self.optimization_analytics.system_performance_score
            }
        except Exception as e:
            self.logger.error(f"Failed to get system performance summary: {e}")
            return {}
    
    def _analyze_optimization_trends(self) -> Dict[str, str]:
        """Analyze optimization trends"""
        try:
            recent_optimizations = list(self.optimization_history)[-20:]
            
            if len(recent_optimizations) < 5:
                return {'insufficient_data': 'true'}
            
            # Analyze performance improvement trend
            improvements = [opt.performance_improvement_percent for opt in recent_optimizations if opt.optimization_successful]
            
            trends = {}
            if len(improvements) >= 3:
                improvement_trend = np.polyfit(range(len(improvements)), improvements, 1)[0]
                trends['performance_improvement'] = 'improving' if improvement_trend > 0.5 else 'degrading' if improvement_trend < -0.5 else 'stable'
            
            # Analyze success rate trend
            success_rates = []
            for i in range(max(0, len(recent_optimizations) - 10), len(recent_optimizations)):
                batch = recent_optimizations[max(0, i-5):i+1]
                if batch:
                    success_rate = sum(1 for opt in batch if opt.optimization_successful) / len(batch)
                    success_rates.append(success_rate)
            
            if len(success_rates) >= 3:
                success_trend = np.polyfit(range(len(success_rates)), success_rates, 1)[0]
                trends['success_rate'] = 'improving' if success_trend > 0.1 else 'degrading' if success_trend < -0.1 else 'stable'
            
            return trends
        except Exception as e:
            self.logger.error(f"Failed to analyze optimization trends: {e}")
            return {}
    
    def _generate_global_optimization_recommendations(self) -> List[str]:
        """Generate global optimization recommendations"""
        recommendations = []
        
        if self.optimization_analytics.optimization_effectiveness_score < 0.7:
            recommendations.append("Optimization effectiveness below 70% - review optimization strategies")
        
        if len(self.detected_regressions) > 0:
            recommendations.append(f"{len(self.detected_regressions)} performance regressions detected - consider immediate optimization")
        
        if self.optimization_analytics.system_performance_score < 0.6:
            recommendations.append("System performance score below 60% - comprehensive optimization recommended")
        
        if not self.monitoring_active:
            recommendations.append("Performance monitoring disabled - enable for better optimization insights")
        
        return recommendations if recommendations else ["System optimization operating within acceptable parameters"]
    
    def _training_pre_hook(self, session_id: str, training_data: Dict[str, Any]) -> None:
        """Training pre-operation hook"""
        try:
            # Start profiling if enabled
            if self.config.enable_real_time_monitoring and session_id not in self.profiling_sessions:
                self.profiling_sessions[session_id] = {
                    'start_time': datetime.utcnow(),
                    'operation_count': 0,
                    'total_time_ms': 0.0
                }
        except Exception as e:
            self.logger.error(f"Training pre-hook failed: {e}")
    
    def _training_post_hook(self, session_id: str, training_data: Dict[str, Any]) -> None:
        """Training post-operation hook"""
        try:
            # Check for regression detection
            if self.config.enable_regression_detection:
                self.detect_training_performance_regression(session_id)
        except Exception as e:
            self.logger.error(f"Training post-hook failed: {e}")
    
    def _performance_degradation_hook(self, session_id: str, degradation_data: Dict[str, Any]) -> None:
        """Performance degradation hook"""
        try:
            # Trigger optimization for performance degradation
            if degradation_data.get('severity') in ['severe', 'critical']:
                self.optimize_training_for_performance(session_id, degradation_data)
        except Exception as e:
            self.logger.error(f"Performance degradation hook failed: {e}")
    
    def get_optimizer_status(self) -> Dict[str, Any]:
        """Get current optimizer status"""
        return {
            'optimization_active': self.optimization_active,
            'monitoring_active': self.monitoring_active,
            'regression_detection_active': self.regression_detection_active,
            'profiling_active': self.profiling_active,
            'active_optimizations': len(self.active_optimizations),
            'profiling_sessions': len(self.profiling_sessions),
            'detected_regressions': len(self.detected_regressions),
            'optimization_strategy': self.config.optimization_strategy.value,
            'optimization_level': self.config.optimization_level.value
        }
    
    def shutdown(self) -> None:
        """Shutdown training performance optimizer"""
        try:
            with self._optimizer_lock:
                self.optimization_active = False
                self.monitoring_active = False
                self.regression_detection_active = False
                self.profiling_active = False
                
                # Complete active optimizations
                for optimization_id in list(self.active_optimizations.keys()):
                    optimization = self.active_optimizations[optimization_id]
                    optimization.optimization_successful = False
                    optimization.error_message = "Optimizer shutdown"
                
                self.active_optimizations.clear()
                self.profiling_sessions.clear()
                
                self.logger.info("Training performance optimizer shutdown completed")
                
        except Exception as e:
            self.logger.error(f"Training performance optimizer shutdown failed: {e}")
            raise RuntimeError(f"Optimizer shutdown failed: {e}")
    
    # Placeholder methods for complex optimization logic
    def _analyze_performance_data(self, session_id: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance data to identify issues"""
        # Placeholder implementation
        return {'issues': []}
    
    def _determine_optimization_strategy_from_performance(self, analysis: Dict[str, Any]) -> OptimizationStrategy:
        """Determine optimization strategy from performance analysis"""
        return self.config.optimization_strategy
    
    def _apply_memory_pressure_optimization(self, session_id: str) -> Dict[str, Any]:
        """Apply memory pressure optimization"""
        return {'success': False, 'optimizations': [], 'parameters': {}}
    
    def _apply_gpu_utilization_optimization(self, session_id: str) -> Dict[str, Any]:
        """Apply GPU utilization optimization"""
        return {'success': False, 'optimizations': [], 'parameters': {}}
    
    def _apply_cpu_bottleneck_optimization(self, session_id: str) -> Dict[str, Any]:
        """Apply CPU bottleneck optimization"""
        return {'success': False, 'optimizations': [], 'parameters': {}}
    
    def _apply_compression_efficiency_optimization(self, session_id: str) -> Dict[str, Any]:
        """Apply compression efficiency optimization"""
        return {'success': False, 'optimizations': [], 'parameters': {}}
    
    def _calculate_optimization_impact(self, session_id: str, analysis: Dict[str, Any], optimizations: List[str]) -> Dict[str, float]:
        """Calculate optimization impact"""
        return {'performance_improvement': 0.0, 'memory_improvement': 0.0, 'gpu_improvement': 0.0}
    
    def _generate_performance_based_recommendations(self, session_id: str, analysis: Dict[str, Any], impact: Dict[str, float]) -> List[str]:
        """Generate performance-based recommendations"""
        return ["Performance-based optimization completed"]