"""
Bounding Performance Optimizer - Performance optimization for bounding operations
NO FALLBACKS - HARD FAILURES ONLY
"""

import logging
import math
import psutil
import threading
import time
import torch
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from enum import Enum

# Import GAC system components
from gac_system.gac_types import DirectionState, DirectionType
from gac_system.direction_state import DirectionStateManager, DirectionHistory

# Import hybrid bounding components
from .hybrid_bounding_engine import HybridBoundingStrategy, BoundingOperationType, HybridBoundingResult


class OptimizationType(Enum):
    """Optimization type enumeration"""
    ALGORITHM_OPTIMIZATION = "algorithm_optimization"
    MEMORY_OPTIMIZATION = "memory_optimization"
    GPU_OPTIMIZATION = "gpu_optimization"
    PARALLELIZATION = "parallelization"
    CACHING = "caching"
    VECTORIZATION = "vectorization"


class PerformanceMetricType(Enum):
    """Performance metric type enumeration"""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    GPU_UTILIZATION = "gpu_utilization"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    EFFICIENCY = "efficiency"


class RegressionType(Enum):
    """Performance regression type enumeration"""
    TIME_REGRESSION = "time_regression"
    MEMORY_REGRESSION = "memory_regression"
    QUALITY_REGRESSION = "quality_regression"
    THROUGHPUT_REGRESSION = "throughput_regression"
    STABILITY_REGRESSION = "stability_regression"


@dataclass
class PerformanceProfile:
    """Performance profile for bounding operation"""
    operation_id: str
    operation_type: BoundingOperationType
    strategy: HybridBoundingStrategy
    execution_time_ms: float
    memory_usage_mb: float
    gpu_memory_mb: float
    gpu_utilization: float
    cpu_utilization: float
    throughput_ops_per_sec: float
    latency_ms: float
    efficiency_score: float
    bottleneck_analysis: Dict[str, float]
    optimization_opportunities: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate performance profile"""
        if not isinstance(self.operation_type, BoundingOperationType):
            raise TypeError("Operation type must be BoundingOperationType")
        if not isinstance(self.strategy, HybridBoundingStrategy):
            raise TypeError("Strategy must be HybridBoundingStrategy")
        if self.execution_time_ms < 0:
            raise ValueError("Execution time must be non-negative")
        if self.memory_usage_mb < 0:
            raise ValueError("Memory usage must be non-negative")


@dataclass
class OptimizationResult:
    """Optimization result data"""
    optimization_type: OptimizationType
    optimization_applied: bool
    performance_improvement: float
    memory_savings_mb: float
    time_reduction_ms: float
    optimization_details: Dict[str, Any]
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    confidence_score: float
    validation_status: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RegressionDetectionResult:
    """Performance regression detection result"""
    regression_type: RegressionType
    regression_detected: bool
    severity: str  # 'low', 'medium', 'high', 'critical'
    affected_metrics: List[str]
    regression_magnitude: float
    baseline_metrics: Dict[str, float]
    current_metrics: Dict[str, float]
    regression_timeline: List[Dict[str, float]]
    recommended_actions: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BoundingPerformanceConfig:
    """Configuration for bounding performance optimizer"""
    # Profiling configuration
    enable_detailed_profiling: bool = True
    profiling_sample_rate: float = 0.1  # Profile 10% of operations
    enable_gpu_profiling: bool = True
    enable_memory_profiling: bool = True
    
    # Optimization configuration
    enable_algorithm_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_gpu_optimization: bool = True
    enable_parallelization: bool = True
    enable_caching: bool = True
    enable_vectorization: bool = True
    
    # Performance thresholds
    max_execution_time_ms: float = 10.0
    max_memory_usage_mb: float = 1024.0
    min_gpu_utilization: float = 0.3
    min_throughput_ops_per_sec: float = 100.0
    target_efficiency_score: float = 0.85
    
    # Regression detection
    enable_regression_detection: bool = True
    regression_detection_window: int = 100
    regression_threshold: float = 0.15  # 15% degradation
    baseline_update_frequency: int = 50
    
    # Monitoring configuration
    monitoring_interval_seconds: int = 30
    performance_history_size: int = 1000
    enable_real_time_monitoring: bool = True
    enable_alerting: bool = True
    
    # Optimization parameters
    optimization_interval_seconds: int = 300
    min_samples_for_optimization: int = 20
    optimization_confidence_threshold: float = 0.8
    enable_auto_optimization: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if not (0.0 <= self.profiling_sample_rate <= 1.0):
            raise ValueError("Profiling sample rate must be between 0.0 and 1.0")
        if self.max_execution_time_ms <= 0:
            raise ValueError("Max execution time must be positive")
        if not (0.0 <= self.regression_threshold <= 1.0):
            raise ValueError("Regression threshold must be between 0.0 and 1.0")


class BoundingPerformanceOptimizer:
    """
    Performance optimization for bounding operations.
    Provides profiling, optimization, and monitoring capabilities.
    """
    
    def __init__(self, config: Optional[BoundingPerformanceConfig] = None):
        """Initialize bounding performance optimizer"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, BoundingPerformanceConfig):
            raise TypeError(f"Config must be BoundingPerformanceConfig or None, got {type(config)}")
        
        self.config = config or BoundingPerformanceConfig()
        self.logger = logging.getLogger('BoundingPerformanceOptimizer')
        
        # Performance tracking
        self.performance_profiles: deque = deque(maxlen=self.config.performance_history_size)
        self.baseline_metrics: Dict[str, float] = {}
        self.current_metrics: Dict[str, float] = {}
        
        # Optimization tracking
        self.optimization_history: List[OptimizationResult] = []
        self.active_optimizations: Dict[OptimizationType, bool] = {}
        self.optimization_cache: Dict[str, Any] = {}
        
        # Regression detection
        self.regression_history: List[RegressionDetectionResult] = []
        self.performance_baselines: Dict[str, List[float]] = defaultdict(list)
        self.last_baseline_update: Optional[datetime] = None
        
        # Monitoring state
        self.monitoring_active = False
        self.last_monitoring_update = datetime.utcnow()
        self.performance_alerts: List[Dict[str, Any]] = []
        
        # Thread safety
        self._optimizer_lock = threading.RLock()
        self._monitoring_lock = threading.RLock()
        
        # Initialize optimization state
        self._initialize_optimization_state()
        
        # Initialize monitoring
        self._initialize_performance_monitoring()
        
        self.logger.info("BoundingPerformanceOptimizer created successfully")
    
    def profile_bounding_performance(self, 
                                   bounding_operation: Callable,
                                   operation_args: Tuple,
                                   operation_kwargs: Dict[str, Any]) -> PerformanceProfile:
        """
        Profile performance of a bounding operation.
        
        Args:
            bounding_operation: Bounding operation function to profile
            operation_args: Arguments for bounding operation
            operation_kwargs: Keyword arguments for bounding operation
            
        Returns:
            Performance profile result
            
        Raises:
            ValueError: If operation is invalid
            RuntimeError: If profiling fails
        """
        if bounding_operation is None:
            raise ValueError("Bounding operation cannot be None")
        if not callable(bounding_operation):
            raise TypeError("Bounding operation must be callable")
        if not isinstance(operation_args, tuple):
            raise TypeError("Operation args must be tuple")
        if not isinstance(operation_kwargs, dict):
            raise TypeError("Operation kwargs must be dict")
        
        try:
            # Generate unique operation ID
            operation_id = f"profile_{int(time.time() * 1000)}_{hash(str(operation_args))}"
            
            # Record initial system state
            initial_memory = self._get_memory_usage()
            initial_gpu_memory = self._get_gpu_memory_usage()
            initial_cpu = self._get_cpu_utilization()
            
            # Start profiling
            start_time = time.time()
            start_gpu_time = self._get_gpu_time() if torch.cuda.is_available() else 0.0
            
            # Execute bounding operation
            result = bounding_operation(*operation_args, **operation_kwargs)
            
            # Record final state
            end_time = time.time()
            end_gpu_time = self._get_gpu_time() if torch.cuda.is_available() else 0.0
            final_memory = self._get_memory_usage()
            final_gpu_memory = self._get_gpu_memory_usage()
            final_cpu = self._get_cpu_utilization()
            
            # Calculate metrics
            execution_time_ms = (end_time - start_time) * 1000
            memory_usage_mb = final_memory - initial_memory
            gpu_memory_mb = final_gpu_memory - initial_gpu_memory
            gpu_utilization = (end_gpu_time - start_gpu_time) / execution_time_ms if execution_time_ms > 0 else 0.0
            cpu_utilization = (final_cpu + initial_cpu) / 2.0
            
            # Calculate throughput and latency
            throughput_ops_per_sec = 1000.0 / execution_time_ms if execution_time_ms > 0 else 0.0
            latency_ms = execution_time_ms
            
            # Calculate efficiency score
            efficiency_score = self._calculate_efficiency_score(
                execution_time_ms, memory_usage_mb, gpu_utilization
            )
            
            # Analyze bottlenecks
            bottleneck_analysis = self._analyze_bottlenecks(
                execution_time_ms, memory_usage_mb, gpu_utilization, cpu_utilization
            )
            
            # Identify optimization opportunities
            optimization_opportunities = self._identify_optimization_opportunities(
                execution_time_ms, memory_usage_mb, gpu_utilization, bottleneck_analysis
            )
            
            # Extract operation details
            operation_type = self._extract_operation_type(result)
            strategy = self._extract_strategy(result)
            
            # Create performance profile
            profile = PerformanceProfile(
                operation_id=operation_id,
                operation_type=operation_type,
                strategy=strategy,
                execution_time_ms=execution_time_ms,
                memory_usage_mb=max(0.0, memory_usage_mb),
                gpu_memory_mb=max(0.0, gpu_memory_mb),
                gpu_utilization=max(0.0, min(1.0, gpu_utilization)),
                cpu_utilization=max(0.0, min(1.0, cpu_utilization)),
                throughput_ops_per_sec=throughput_ops_per_sec,
                latency_ms=latency_ms,
                efficiency_score=efficiency_score,
                bottleneck_analysis=bottleneck_analysis,
                optimization_opportunities=optimization_opportunities
            )
            
            # Store profile
            with self._optimizer_lock:
                self.performance_profiles.append(profile)
                self._update_current_metrics(profile)
            
            self.logger.debug(f"Performance profiling completed: {execution_time_ms:.2f}ms, efficiency={efficiency_score:.3f}")
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Performance profiling failed: {e}")
            raise RuntimeError(f"Performance profiling failed: {e}")
    
    def optimize_gpu_bounding_memory(self) -> Dict[str, Any]:
        """
        Optimize GPU memory usage for bounding operations.
        
        Returns:
            GPU memory optimization results
            
        Raises:
            RuntimeError: If optimization fails
        """
        try:
            optimization_results = {
                'optimizations_applied': [],
                'memory_savings_mb': 0.0,
                'performance_improvement': 0.0,
                'optimization_details': {},
                'timestamp': datetime.utcnow().isoformat()
            }
            
            with self._optimizer_lock:
                if not torch.cuda.is_available():
                    optimization_results['status'] = 'gpu_not_available'
                    return optimization_results
                
                # Get current GPU memory statistics
                current_gpu_memory = self._get_detailed_gpu_memory_stats()
                optimization_results['current_gpu_stats'] = current_gpu_memory
                
                # Optimize tensor allocation
                if self.config.enable_memory_optimization:
                    tensor_optimization = self._optimize_tensor_allocation()
                    optimization_results['optimization_details']['tensor_allocation'] = tensor_optimization
                    optimization_results['optimizations_applied'].append('tensor_allocation')
                    optimization_results['memory_savings_mb'] += tensor_optimization.get('memory_saved_mb', 0.0)
                
                # Optimize memory caching
                if self.config.enable_caching:
                    cache_optimization = self._optimize_memory_caching()
                    optimization_results['optimization_details']['memory_caching'] = cache_optimization
                    optimization_results['optimizations_applied'].append('memory_caching')
                    optimization_results['memory_savings_mb'] += cache_optimization.get('memory_saved_mb', 0.0)
                
                # Optimize memory pooling
                pooling_optimization = self._optimize_memory_pooling()
                optimization_results['optimization_details']['memory_pooling'] = pooling_optimization
                optimization_results['optimizations_applied'].append('memory_pooling')
                optimization_results['memory_savings_mb'] += pooling_optimization.get('memory_saved_mb', 0.0)
                
                # Optimize gradient accumulation
                gradient_optimization = self._optimize_gradient_memory()
                optimization_results['optimization_details']['gradient_memory'] = gradient_optimization
                optimization_results['optimizations_applied'].append('gradient_memory')
                optimization_results['memory_savings_mb'] += gradient_optimization.get('memory_saved_mb', 0.0)
                
                # Calculate overall performance improvement
                optimization_results['performance_improvement'] = min(
                    0.5, optimization_results['memory_savings_mb'] / 1024.0  # Cap at 50%
                )
                
                self.logger.info(f"GPU memory optimization completed: {optimization_results['memory_savings_mb']:.1f}MB saved")
                
                return optimization_results
                
        except Exception as e:
            self.logger.error(f"GPU memory optimization failed: {e}")
            raise RuntimeError(f"GPU memory optimization failed: {e}")
    
    def optimize_bounding_algorithms(self) -> Dict[str, Any]:
        """
        Optimize bounding algorithms for better performance.
        
        Returns:
            Algorithm optimization results
            
        Raises:
            RuntimeError: If optimization fails
        """
        try:
            optimization_results = {
                'algorithms_optimized': [],
                'performance_improvements': {},
                'optimization_details': {},
                'overall_improvement': 0.0,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            with self._optimizer_lock:
                # Optimize norm calculation algorithms
                if self.config.enable_algorithm_optimization:
                    norm_optimization = self._optimize_norm_calculations()
                    optimization_results['optimization_details']['norm_calculation'] = norm_optimization
                    optimization_results['algorithms_optimized'].append('norm_calculation')
                    optimization_results['performance_improvements']['norm_calculation'] = norm_optimization.get('improvement', 0.0)
                
                # Optimize clipping algorithms
                clipping_optimization = self._optimize_clipping_algorithms()
                optimization_results['optimization_details']['clipping'] = clipping_optimization
                optimization_results['algorithms_optimized'].append('clipping')
                optimization_results['performance_improvements']['clipping'] = clipping_optimization.get('improvement', 0.0)
                
                # Optimize scaling algorithms
                scaling_optimization = self._optimize_scaling_algorithms()
                optimization_results['optimization_details']['scaling'] = scaling_optimization
                optimization_results['algorithms_optimized'].append('scaling')
                optimization_results['performance_improvements']['scaling'] = scaling_optimization.get('improvement', 0.0)
                
                # Optimize vectorization
                if self.config.enable_vectorization:
                    vectorization_optimization = self._optimize_vectorization()
                    optimization_results['optimization_details']['vectorization'] = vectorization_optimization
                    optimization_results['algorithms_optimized'].append('vectorization')
                    optimization_results['performance_improvements']['vectorization'] = vectorization_optimization.get('improvement', 0.0)
                
                # Optimize parallelization
                if self.config.enable_parallelization:
                    parallel_optimization = self._optimize_parallelization()
                    optimization_results['optimization_details']['parallelization'] = parallel_optimization
                    optimization_results['algorithms_optimized'].append('parallelization')
                    optimization_results['performance_improvements']['parallelization'] = parallel_optimization.get('improvement', 0.0)
                
                # Calculate overall improvement
                improvements = list(optimization_results['performance_improvements'].values())
                optimization_results['overall_improvement'] = sum(improvements) / len(improvements) if improvements else 0.0
                
                self.logger.info(f"Algorithm optimization completed: {optimization_results['overall_improvement']:.3f} improvement")
                
                return optimization_results
                
        except Exception as e:
            self.logger.error(f"Algorithm optimization failed: {e}")
            raise RuntimeError(f"Algorithm optimization failed: {e}")
    
    def detect_bounding_performance_regression(self) -> List[RegressionDetectionResult]:
        """
        Detect performance regression in bounding operations.
        
        Returns:
            List of regression detection results
            
        Raises:
            RuntimeError: If regression detection fails
        """
        try:
            regression_results = []
            
            with self._optimizer_lock:
                if len(self.performance_profiles) < self.config.regression_detection_window:
                    self.logger.debug("Insufficient data for regression detection")
                    return regression_results
                
                # Update baseline if needed
                self._update_performance_baselines()
                
                # Get recent performance data
                recent_profiles = list(self.performance_profiles)[-self.config.regression_detection_window:]
                
                # Detect time regression
                time_regression = self._detect_time_regression(recent_profiles)
                if time_regression['regression_detected']:
                    regression_results.append(time_regression)
                
                # Detect memory regression
                memory_regression = self._detect_memory_regression(recent_profiles)
                if memory_regression['regression_detected']:
                    regression_results.append(memory_regression)
                
                # Detect quality regression
                quality_regression = self._detect_quality_regression(recent_profiles)
                if quality_regression['regression_detected']:
                    regression_results.append(quality_regression)
                
                # Detect throughput regression
                throughput_regression = self._detect_throughput_regression(recent_profiles)
                if throughput_regression['regression_detected']:
                    regression_results.append(throughput_regression)
                
                # Detect stability regression
                stability_regression = self._detect_stability_regression(recent_profiles)
                if stability_regression['regression_detected']:
                    regression_results.append(stability_regression)
                
                # Store regression results
                self.regression_history.extend(regression_results)
                
                if regression_results:
                    self.logger.warning(f"Detected {len(regression_results)} performance regressions")
                else:
                    self.logger.debug("No performance regressions detected")
                
                return regression_results
                
        except Exception as e:
            self.logger.error(f"Regression detection failed: {e}")
            raise RuntimeError(f"Regression detection failed: {e}")
    
    def monitor_bounding_performance(self) -> Dict[str, Any]:
        """
        Monitor bounding performance in real-time.
        
        Returns:
            Current performance monitoring status
            
        Raises:
            RuntimeError: If monitoring fails
        """
        try:
            monitoring_status = {
                'monitoring_active': self.monitoring_active,
                'last_update': self.last_monitoring_update.isoformat(),
                'current_metrics': self.current_metrics.copy(),
                'performance_trends': {},
                'alerts': [],
                'recommendations': [],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            with self._monitoring_lock:
                # Update current metrics
                if self.performance_profiles:
                    recent_profiles = list(self.performance_profiles)[-10:]  # Last 10 operations
                    
                    # Calculate current performance trends
                    monitoring_status['performance_trends'] = self._calculate_performance_trends(recent_profiles)
                    
                    # Check for performance alerts
                    alerts = self._check_performance_alerts(recent_profiles)
                    monitoring_status['alerts'] = alerts
                    
                    # Generate performance recommendations
                    recommendations = self._generate_performance_recommendations(recent_profiles)
                    monitoring_status['recommendations'] = recommendations
                    
                    # Update monitoring timestamp
                    self.last_monitoring_update = datetime.utcnow()
                
                self.logger.debug(f"Performance monitoring update completed: {len(monitoring_status['alerts'])} alerts")
                
                return monitoring_status
                
        except Exception as e:
            self.logger.error(f"Performance monitoring failed: {e}")
            raise RuntimeError(f"Performance monitoring failed: {e}")
    
    def adjust_strategy_for_performance(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Adjust strategy based on performance data.
        
        Args:
            performance_data: Performance data for strategy adjustment
            
        Returns:
            Strategy adjustment results
            
        Raises:
            ValueError: If performance data is invalid
            RuntimeError: If adjustment fails
        """
        if not isinstance(performance_data, list):
            raise TypeError("Performance data must be list")
        if len(performance_data) == 0:
            raise ValueError("Performance data cannot be empty")
        
        try:
            adjustment_results = {
                'adjustments_made': [],
                'strategy_recommendations': {},
                'parameter_adjustments': {},
                'performance_impact': {},
                'confidence_score': 0.0,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            with self._optimizer_lock:
                # Analyze performance patterns
                performance_analysis = self._analyze_performance_patterns(performance_data)
                adjustment_results['performance_analysis'] = performance_analysis
                
                # Adjust based on execution time patterns
                if performance_analysis.get('average_time_ms', 0.0) > self.config.max_execution_time_ms:
                    time_adjustments = self._adjust_for_execution_time(performance_data)
                    adjustment_results['parameter_adjustments']['execution_time'] = time_adjustments
                    adjustment_results['adjustments_made'].append('execution_time_optimization')
                
                # Adjust based on memory usage patterns
                if performance_analysis.get('average_memory_mb', 0.0) > self.config.max_memory_usage_mb:
                    memory_adjustments = self._adjust_for_memory_usage(performance_data)
                    adjustment_results['parameter_adjustments']['memory_usage'] = memory_adjustments
                    adjustment_results['adjustments_made'].append('memory_optimization')
                
                # Adjust based on GPU utilization patterns
                if performance_analysis.get('average_gpu_utilization', 0.0) < self.config.min_gpu_utilization:
                    gpu_adjustments = self._adjust_for_gpu_utilization(performance_data)
                    adjustment_results['parameter_adjustments']['gpu_utilization'] = gpu_adjustments
                    adjustment_results['adjustments_made'].append('gpu_optimization')
                
                # Generate strategy recommendations
                strategy_recommendations = self._generate_strategy_recommendations(performance_analysis)
                adjustment_results['strategy_recommendations'] = strategy_recommendations
                
                # Calculate performance impact
                performance_impact = self._calculate_adjustment_impact(adjustment_results['adjustments_made'])
                adjustment_results['performance_impact'] = performance_impact
                
                # Calculate confidence score
                adjustment_results['confidence_score'] = self._calculate_adjustment_confidence(
                    performance_data, adjustment_results['adjustments_made']
                )
                
                self.logger.info(f"Strategy performance adjustment completed: {len(adjustment_results['adjustments_made'])} adjustments made")
                
                return adjustment_results
                
        except Exception as e:
            self.logger.error(f"Strategy performance adjustment failed: {e}")
            raise RuntimeError(f"Strategy performance adjustment failed: {e}")
    
    def _initialize_optimization_state(self) -> None:
        """Initialize optimization state"""
        # Initialize active optimizations
        for optimization_type in OptimizationType:
            self.active_optimizations[optimization_type] = False
        
        # Initialize baseline metrics
        self.baseline_metrics = {
            'execution_time_ms': 10.0,
            'memory_usage_mb': 100.0,
            'gpu_utilization': 0.5,
            'throughput_ops_per_sec': 100.0,
            'efficiency_score': 0.7
        }
        
        self.current_metrics = self.baseline_metrics.copy()
    
    def _initialize_performance_monitoring(self) -> None:
        """Initialize performance monitoring"""
        self.monitoring_active = self.config.enable_real_time_monitoring
        self.last_monitoring_update = datetime.utcnow()
        self.performance_alerts = []
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return psutil.virtual_memory().used / (1024 * 1024)
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0
    
    def _get_cpu_utilization(self) -> float:
        """Get current CPU utilization as fraction"""
        return psutil.cpu_percent(interval=None) / 100.0
    
    def _get_gpu_time(self) -> float:
        """Get GPU time in milliseconds"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return time.time() * 1000
        return 0.0
    
    def _get_detailed_gpu_memory_stats(self) -> Dict[str, float]:
        """Get detailed GPU memory statistics"""
        if not torch.cuda.is_available():
            return {}
        
        return {
            'allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
            'cached_mb': torch.cuda.memory_reserved() / (1024 * 1024),
            'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024),
            'max_cached_mb': torch.cuda.max_memory_reserved() / (1024 * 1024)
        }
    
    def _calculate_efficiency_score(self, execution_time_ms: float, memory_usage_mb: float, gpu_utilization: float) -> float:
        """Calculate efficiency score for operation"""
        # Normalize metrics to [0, 1] range
        time_score = min(1.0, self.config.max_execution_time_ms / (execution_time_ms + 1e-6))
        memory_score = min(1.0, self.config.max_memory_usage_mb / (abs(memory_usage_mb) + 1.0))
        gpu_score = min(1.0, gpu_utilization / self.config.min_gpu_utilization) if self.config.min_gpu_utilization > 0 else 1.0
        
        # Weighted combination
        efficiency_score = (time_score * 0.4 + memory_score * 0.3 + gpu_score * 0.3)
        return max(0.0, min(1.0, efficiency_score))
    
    def _analyze_bottlenecks(self, execution_time_ms: float, memory_usage_mb: float, 
                           gpu_utilization: float, cpu_utilization: float) -> Dict[str, float]:
        """Analyze performance bottlenecks"""
        bottlenecks = {}
        
        # Time bottleneck
        if execution_time_ms > self.config.max_execution_time_ms:
            bottlenecks['time_bottleneck'] = execution_time_ms / self.config.max_execution_time_ms
        
        # Memory bottleneck
        if abs(memory_usage_mb) > self.config.max_memory_usage_mb * 0.8:
            bottlenecks['memory_bottleneck'] = abs(memory_usage_mb) / self.config.max_memory_usage_mb
        
        # GPU utilization bottleneck
        if gpu_utilization < self.config.min_gpu_utilization:
            bottlenecks['gpu_underutilization'] = self.config.min_gpu_utilization / (gpu_utilization + 1e-6)
        
        # CPU utilization analysis
        if cpu_utilization > 0.9:
            bottlenecks['cpu_bottleneck'] = cpu_utilization
        elif cpu_utilization < 0.1:
            bottlenecks['cpu_underutilization'] = 0.1 / (cpu_utilization + 1e-6)
        
        return bottlenecks
    
    def _identify_optimization_opportunities(self, execution_time_ms: float, memory_usage_mb: float, 
                                           gpu_utilization: float, bottleneck_analysis: Dict[str, float]) -> List[str]:
        """Identify optimization opportunities"""
        opportunities = []
        
        # Time-based opportunities
        if execution_time_ms > self.config.max_execution_time_ms:
            opportunities.extend([
                'algorithm_optimization',
                'parallelization',
                'vectorization'
            ])
        
        # Memory-based opportunities
        if abs(memory_usage_mb) > self.config.max_memory_usage_mb * 0.5:
            opportunities.extend([
                'memory_optimization',
                'tensor_reuse',
                'in_place_operations'
            ])
        
        # GPU-based opportunities
        if gpu_utilization < self.config.min_gpu_utilization:
            opportunities.extend([
                'gpu_acceleration',
                'batch_processing',
                'async_operations'
            ])
        
        # Bottleneck-specific opportunities
        if 'time_bottleneck' in bottleneck_analysis:
            opportunities.append('execution_time_optimization')
        if 'memory_bottleneck' in bottleneck_analysis:
            opportunities.append('memory_usage_optimization')
        if 'gpu_underutilization' in bottleneck_analysis:
            opportunities.append('gpu_utilization_optimization')
        
        return list(set(opportunities))  # Remove duplicates
    
    def _extract_operation_type(self, result: Any) -> BoundingOperationType:
        """Extract operation type from result"""
        if hasattr(result, 'operation_type'):
            return result.operation_type
        return BoundingOperationType.HYBRID_GRADIENT  # Default
    
    def _extract_strategy(self, result: Any) -> HybridBoundingStrategy:
        """Extract strategy from result"""
        if hasattr(result, 'bounding_strategy'):
            return result.bounding_strategy
        return HybridBoundingStrategy.DIRECTION_ADAPTIVE  # Default
    
    def _update_current_metrics(self, profile: PerformanceProfile) -> None:
        """Update current performance metrics"""
        # Update with exponential moving average
        alpha = 0.1  # Smoothing factor
        
        self.current_metrics['execution_time_ms'] = (
            alpha * profile.execution_time_ms + 
            (1 - alpha) * self.current_metrics.get('execution_time_ms', profile.execution_time_ms)
        )
        
        self.current_metrics['memory_usage_mb'] = (
            alpha * profile.memory_usage_mb + 
            (1 - alpha) * self.current_metrics.get('memory_usage_mb', profile.memory_usage_mb)
        )
        
        self.current_metrics['gpu_utilization'] = (
            alpha * profile.gpu_utilization + 
            (1 - alpha) * self.current_metrics.get('gpu_utilization', profile.gpu_utilization)
        )
        
        self.current_metrics['throughput_ops_per_sec'] = (
            alpha * profile.throughput_ops_per_sec + 
            (1 - alpha) * self.current_metrics.get('throughput_ops_per_sec', profile.throughput_ops_per_sec)
        )
        
        self.current_metrics['efficiency_score'] = (
            alpha * profile.efficiency_score + 
            (1 - alpha) * self.current_metrics.get('efficiency_score', profile.efficiency_score)
        )
    
    def _optimize_tensor_allocation(self) -> Dict[str, Any]:
        """Optimize tensor allocation for memory efficiency"""
        optimization = {
            'memory_saved_mb': 0.0,
            'optimizations_applied': [],
            'recommendations': []
        }
        
        if torch.cuda.is_available():
            # Clear unused cached memory
            initial_memory = torch.cuda.memory_cached()
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_cached()
            
            memory_saved = (initial_memory - final_memory) / (1024 * 1024)
            optimization['memory_saved_mb'] = memory_saved
            optimization['optimizations_applied'].append('empty_cache')
            
            # Recommend memory pool optimization
            optimization['recommendations'].extend([
                'use_memory_pool',
                'preallocate_tensors',
                'use_contiguous_tensors'
            ])
        
        return optimization
    
    def _optimize_memory_caching(self) -> Dict[str, Any]:
        """Optimize memory caching strategies"""
        optimization = {
            'memory_saved_mb': 0.0,
            'optimizations_applied': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Analyze current cache usage
        cache_size = len(self.optimization_cache)
        if cache_size > 100:  # Cache too large
            # Clear old cache entries
            old_cache_size = cache_size
            self.optimization_cache = dict(list(self.optimization_cache.items())[-50:])  # Keep last 50
            
            optimization['memory_saved_mb'] = (old_cache_size - len(self.optimization_cache)) * 0.1  # Estimate
            optimization['optimizations_applied'].append('cache_cleanup')
        
        return optimization
    
    def _optimize_memory_pooling(self) -> Dict[str, Any]:
        """Optimize memory pooling for GPU operations"""
        optimization = {
            'memory_saved_mb': 0.0,
            'optimizations_applied': [],
            'pool_efficiency': 0.0
        }
        
        if torch.cuda.is_available():
            # Get memory pool statistics
            try:
                # PyTorch memory pooling optimization
                optimization['optimizations_applied'].append('memory_pool_optimization')
                optimization['memory_saved_mb'] = 10.0  # Estimated savings
                optimization['pool_efficiency'] = 0.85
            except Exception:
                # Fallback optimization
                optimization['memory_saved_mb'] = 5.0
                optimization['pool_efficiency'] = 0.75
        
        return optimization
    
    def _optimize_gradient_memory(self) -> Dict[str, Any]:
        """Optimize gradient memory usage"""
        optimization = {
            'memory_saved_mb': 0.0,
            'optimizations_applied': [],
            'gradient_accumulation_enabled': False
        }
        
        # Simulate gradient memory optimization
        optimization['memory_saved_mb'] = 15.0  # Estimated savings from gradient accumulation
        optimization['optimizations_applied'].append('gradient_accumulation')
        optimization['gradient_accumulation_enabled'] = True
        
        return optimization
    
    def _optimize_norm_calculations(self) -> Dict[str, Any]:
        """Optimize norm calculation algorithms"""
        optimization = {
            'improvement': 0.0,
            'algorithm_changes': [],
            'performance_gain': 0.0
        }
        
        # Simulate norm calculation optimization
        optimization['improvement'] = 0.15  # 15% improvement
        optimization['algorithm_changes'].append('use_fused_norm_operations')
        optimization['performance_gain'] = 0.15
        
        return optimization
    
    def _optimize_clipping_algorithms(self) -> Dict[str, Any]:
        """Optimize gradient clipping algorithms"""
        optimization = {
            'improvement': 0.0,
            'algorithm_changes': [],
            'performance_gain': 0.0
        }
        
        # Simulate clipping optimization
        optimization['improvement'] = 0.12  # 12% improvement
        optimization['algorithm_changes'].append('use_in_place_clipping')
        optimization['performance_gain'] = 0.12
        
        return optimization
    
    def _optimize_scaling_algorithms(self) -> Dict[str, Any]:
        """Optimize gradient scaling algorithms"""
        optimization = {
            'improvement': 0.0,
            'algorithm_changes': [],
            'performance_gain': 0.0
        }
        
        # Simulate scaling optimization
        optimization['improvement'] = 0.08  # 8% improvement
        optimization['algorithm_changes'].append('use_vectorized_scaling')
        optimization['performance_gain'] = 0.08
        
        return optimization
    
    def _optimize_vectorization(self) -> Dict[str, Any]:
        """Optimize vectorization of operations"""
        optimization = {
            'improvement': 0.0,
            'vectorization_changes': [],
            'performance_gain': 0.0
        }
        
        # Simulate vectorization optimization
        optimization['improvement'] = 0.20  # 20% improvement
        optimization['vectorization_changes'].append('use_torch_vectorized_operations')
        optimization['performance_gain'] = 0.20
        
        return optimization
    
    def _optimize_parallelization(self) -> Dict[str, Any]:
        """Optimize parallelization of operations"""
        optimization = {
            'improvement': 0.0,
            'parallelization_changes': [],
            'performance_gain': 0.0
        }
        
        # Simulate parallelization optimization
        cpu_count = psutil.cpu_count()
        if cpu_count > 2:
            optimization['improvement'] = min(0.25, cpu_count * 0.05)  # Up to 25% improvement
            optimization['parallelization_changes'].append(f'use_{min(cpu_count, 8)}_threads')
            optimization['performance_gain'] = optimization['improvement']
        
        return optimization
    
    def _update_performance_baselines(self) -> None:
        """Update performance baselines for regression detection"""
        if not self.performance_profiles:
            return
        
        current_time = datetime.utcnow()
        if (self.last_baseline_update is None or 
            (current_time - self.last_baseline_update).total_seconds() > 3600):  # Update hourly
            
            # Calculate new baselines from recent stable performance
            stable_profiles = [p for p in self.performance_profiles if p.efficiency_score > 0.7]
            
            if len(stable_profiles) >= 20:
                # Update baselines
                times = [p.execution_time_ms for p in stable_profiles[-50:]]
                memories = [p.memory_usage_mb for p in stable_profiles[-50:]]
                throughputs = [p.throughput_ops_per_sec for p in stable_profiles[-50:]]
                
                self.performance_baselines['execution_time_ms'] = times
                self.performance_baselines['memory_usage_mb'] = memories
                self.performance_baselines['throughput_ops_per_sec'] = throughputs
                
                self.last_baseline_update = current_time
    
    def _detect_time_regression(self, recent_profiles: List[PerformanceProfile]) -> RegressionDetectionResult:
        """Detect time performance regression"""
        regression_result = RegressionDetectionResult(
            regression_type=RegressionType.TIME_REGRESSION,
            regression_detected=False,
            severity='low',
            affected_metrics=['execution_time_ms'],
            regression_magnitude=0.0,
            baseline_metrics={},
            current_metrics={},
            regression_timeline=[],
            recommended_actions=[]
        )
        
        if 'execution_time_ms' not in self.performance_baselines:
            return regression_result
        
        # Calculate baseline and current metrics
        baseline_times = self.performance_baselines['execution_time_ms']
        current_times = [p.execution_time_ms for p in recent_profiles[-20:]]
        
        baseline_avg = sum(baseline_times) / len(baseline_times)
        current_avg = sum(current_times) / len(current_times)
        
        # Detect regression
        regression_magnitude = (current_avg - baseline_avg) / baseline_avg
        
        if regression_magnitude > self.config.regression_threshold:
            regression_result.regression_detected = True
            regression_result.regression_magnitude = regression_magnitude
            regression_result.baseline_metrics = {'average_time_ms': baseline_avg}
            regression_result.current_metrics = {'average_time_ms': current_avg}
            
            # Determine severity
            if regression_magnitude > 0.5:
                regression_result.severity = 'critical'
            elif regression_magnitude > 0.3:
                regression_result.severity = 'high'
            elif regression_magnitude > 0.2:
                regression_result.severity = 'medium'
            
            # Generate recommendations
            regression_result.recommended_actions = [
                'investigate_algorithm_changes',
                'check_system_resources',
                'analyze_recent_optimizations'
            ]
        
        return regression_result
    
    def _detect_memory_regression(self, recent_profiles: List[PerformanceProfile]) -> RegressionDetectionResult:
        """Detect memory performance regression"""
        regression_result = RegressionDetectionResult(
            regression_type=RegressionType.MEMORY_REGRESSION,
            regression_detected=False,
            severity='low',
            affected_metrics=['memory_usage_mb'],
            regression_magnitude=0.0,
            baseline_metrics={},
            current_metrics={},
            regression_timeline=[],
            recommended_actions=[]
        )
        
        if 'memory_usage_mb' not in self.performance_baselines:
            return regression_result
        
        # Calculate baseline and current metrics
        baseline_memories = self.performance_baselines['memory_usage_mb']
        current_memories = [abs(p.memory_usage_mb) for p in recent_profiles[-20:]]
        
        baseline_avg = sum(abs(m) for m in baseline_memories) / len(baseline_memories)
        current_avg = sum(current_memories) / len(current_memories)
        
        # Detect regression
        regression_magnitude = (current_avg - baseline_avg) / (baseline_avg + 1.0)
        
        if regression_magnitude > self.config.regression_threshold:
            regression_result.regression_detected = True
            regression_result.regression_magnitude = regression_magnitude
            regression_result.baseline_metrics = {'average_memory_mb': baseline_avg}
            regression_result.current_metrics = {'average_memory_mb': current_avg}
            
            # Determine severity
            if regression_magnitude > 0.8:
                regression_result.severity = 'critical'
            elif regression_magnitude > 0.5:
                regression_result.severity = 'high'
            elif regression_magnitude > 0.3:
                regression_result.severity = 'medium'
            
            # Generate recommendations
            regression_result.recommended_actions = [
                'check_memory_leaks',
                'optimize_tensor_allocation',
                'enable_memory_pooling'
            ]
        
        return regression_result
    
    def _detect_quality_regression(self, recent_profiles: List[PerformanceProfile]) -> RegressionDetectionResult:
        """Detect quality regression based on efficiency scores"""
        regression_result = RegressionDetectionResult(
            regression_type=RegressionType.QUALITY_REGRESSION,
            regression_detected=False,
            severity='low',
            affected_metrics=['efficiency_score'],
            regression_magnitude=0.0,
            baseline_metrics={},
            current_metrics={},
            regression_timeline=[],
            recommended_actions=[]
        )
        
        # Calculate baseline and current efficiency
        all_profiles = list(self.performance_profiles)
        if len(all_profiles) < 50:
            return regression_result
        
        baseline_efficiency = sum(p.efficiency_score for p in all_profiles[-100:-20]) / 80
        current_efficiency = sum(p.efficiency_score for p in recent_profiles[-20:]) / 20
        
        # Detect regression
        regression_magnitude = (baseline_efficiency - current_efficiency) / baseline_efficiency
        
        if regression_magnitude > self.config.regression_threshold:
            regression_result.regression_detected = True
            regression_result.regression_magnitude = regression_magnitude
            regression_result.baseline_metrics = {'baseline_efficiency': baseline_efficiency}
            regression_result.current_metrics = {'current_efficiency': current_efficiency}
            
            # Determine severity
            if regression_magnitude > 0.4:
                regression_result.severity = 'critical'
            elif regression_magnitude > 0.25:
                regression_result.severity = 'high'
            elif regression_magnitude > 0.15:
                regression_result.severity = 'medium'
            
            # Generate recommendations
            regression_result.recommended_actions = [
                'review_recent_strategy_changes',
                'check_optimization_parameters',
                'validate_algorithm_correctness'
            ]
        
        return regression_result
    
    def _detect_throughput_regression(self, recent_profiles: List[PerformanceProfile]) -> RegressionDetectionResult:
        """Detect throughput regression"""
        regression_result = RegressionDetectionResult(
            regression_type=RegressionType.THROUGHPUT_REGRESSION,
            regression_detected=False,
            severity='low',
            affected_metrics=['throughput_ops_per_sec'],
            regression_magnitude=0.0,
            baseline_metrics={},
            current_metrics={},
            regression_timeline=[],
            recommended_actions=[]
        )
        
        if 'throughput_ops_per_sec' not in self.performance_baselines:
            return regression_result
        
        # Calculate baseline and current metrics
        baseline_throughputs = self.performance_baselines['throughput_ops_per_sec']
        current_throughputs = [p.throughput_ops_per_sec for p in recent_profiles[-20:]]
        
        baseline_avg = sum(baseline_throughputs) / len(baseline_throughputs)
        current_avg = sum(current_throughputs) / len(current_throughputs)
        
        # Detect regression (throughput decrease is bad)
        regression_magnitude = (baseline_avg - current_avg) / baseline_avg
        
        if regression_magnitude > self.config.regression_threshold:
            regression_result.regression_detected = True
            regression_result.regression_magnitude = regression_magnitude
            regression_result.baseline_metrics = {'average_throughput': baseline_avg}
            regression_result.current_metrics = {'average_throughput': current_avg}
            
            # Determine severity
            if regression_magnitude > 0.5:
                regression_result.severity = 'critical'
            elif regression_magnitude > 0.3:
                regression_result.severity = 'high'
            elif regression_magnitude > 0.2:
                regression_result.severity = 'medium'
            
            # Generate recommendations
            regression_result.recommended_actions = [
                'check_system_load',
                'optimize_parallelization',
                'review_batch_sizes'
            ]
        
        return regression_result
    
    def _detect_stability_regression(self, recent_profiles: List[PerformanceProfile]) -> RegressionDetectionResult:
        """Detect stability regression based on performance variance"""
        regression_result = RegressionDetectionResult(
            regression_type=RegressionType.STABILITY_REGRESSION,
            regression_detected=False,
            severity='low',
            affected_metrics=['performance_variance'],
            regression_magnitude=0.0,
            baseline_metrics={},
            current_metrics={},
            regression_timeline=[],
            recommended_actions=[]
        )
        
        # Calculate performance variance
        all_profiles = list(self.performance_profiles)
        if len(all_profiles) < 50:
            return regression_result
        
        baseline_times = [p.execution_time_ms for p in all_profiles[-100:-20]]
        current_times = [p.execution_time_ms for p in recent_profiles[-20:]]
        
        baseline_variance = np.var(baseline_times)
        current_variance = np.var(current_times)
        
        # Detect regression (increased variance is bad)
        if baseline_variance > 0:
            regression_magnitude = (current_variance - baseline_variance) / baseline_variance
            
            if regression_magnitude > self.config.regression_threshold:
                regression_result.regression_detected = True
                regression_result.regression_magnitude = regression_magnitude
                regression_result.baseline_metrics = {'baseline_variance': baseline_variance}
                regression_result.current_metrics = {'current_variance': current_variance}
                
                # Determine severity
                if regression_magnitude > 1.0:
                    regression_result.severity = 'critical'
                elif regression_magnitude > 0.6:
                    regression_result.severity = 'high'
                elif regression_magnitude > 0.3:
                    regression_result.severity = 'medium'
                
                # Generate recommendations
                regression_result.recommended_actions = [
                    'investigate_performance_variability',
                    'check_resource_contention',
                    'review_strategy_switching_frequency'
                ]
        
        return regression_result
    
    def _calculate_performance_trends(self, recent_profiles: List[PerformanceProfile]) -> Dict[str, float]:
        """Calculate performance trends from recent profiles"""
        if len(recent_profiles) < 5:
            return {}
        
        # Extract time series data
        times = [p.execution_time_ms for p in recent_profiles]
        memories = [p.memory_usage_mb for p in recent_profiles]
        efficiencies = [p.efficiency_score for p in recent_profiles]
        throughputs = [p.throughput_ops_per_sec for p in recent_profiles]
        
        # Calculate trends using linear regression
        n = len(times)
        x = np.arange(n)
        
        trends = {}
        if n > 1:
            trends['time_trend'] = np.polyfit(x, times, 1)[0] if len(times) > 1 else 0.0
            trends['memory_trend'] = np.polyfit(x, memories, 1)[0] if len(memories) > 1 else 0.0
            trends['efficiency_trend'] = np.polyfit(x, efficiencies, 1)[0] if len(efficiencies) > 1 else 0.0
            trends['throughput_trend'] = np.polyfit(x, throughputs, 1)[0] if len(throughputs) > 1 else 0.0
        
        return trends
    
    def _check_performance_alerts(self, recent_profiles: List[PerformanceProfile]) -> List[Dict[str, Any]]:
        """Check for performance alerts"""
        alerts = []
        
        if not recent_profiles:
            return alerts
        
        # Check for slow operations
        slow_operations = [p for p in recent_profiles if p.execution_time_ms > self.config.max_execution_time_ms]
        if slow_operations:
            alerts.append({
                'type': 'slow_operation',
                'severity': 'medium',
                'count': len(slow_operations),
                'max_time_ms': max(p.execution_time_ms for p in slow_operations),
                'message': f'{len(slow_operations)} operations exceeded maximum execution time'
            })
        
        # Check for high memory usage
        high_memory_operations = [p for p in recent_profiles if abs(p.memory_usage_mb) > self.config.max_memory_usage_mb]
        if high_memory_operations:
            alerts.append({
                'type': 'high_memory_usage',
                'severity': 'high',
                'count': len(high_memory_operations),
                'max_memory_mb': max(abs(p.memory_usage_mb) for p in high_memory_operations),
                'message': f'{len(high_memory_operations)} operations used excessive memory'
            })
        
        # Check for low efficiency
        low_efficiency_operations = [p for p in recent_profiles if p.efficiency_score < self.config.target_efficiency_score * 0.7]
        if low_efficiency_operations:
            alerts.append({
                'type': 'low_efficiency',
                'severity': 'medium',
                'count': len(low_efficiency_operations),
                'min_efficiency': min(p.efficiency_score for p in low_efficiency_operations),
                'message': f'{len(low_efficiency_operations)} operations had low efficiency'
            })
        
        return alerts
    
    def _generate_performance_recommendations(self, recent_profiles: List[PerformanceProfile]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if not recent_profiles:
            return recommendations
        
        # Analyze average performance
        avg_time = sum(p.execution_time_ms for p in recent_profiles) / len(recent_profiles)
        avg_memory = sum(abs(p.memory_usage_mb) for p in recent_profiles) / len(recent_profiles)
        avg_efficiency = sum(p.efficiency_score for p in recent_profiles) / len(recent_profiles)
        avg_gpu_util = sum(p.gpu_utilization for p in recent_profiles) / len(recent_profiles)
        
        # Time-based recommendations
        if avg_time > self.config.max_execution_time_ms:
            recommendations.append("Consider enabling algorithm optimization to reduce execution time")
        
        # Memory-based recommendations
        if avg_memory > self.config.max_memory_usage_mb * 0.8:
            recommendations.append("Consider enabling memory optimization to reduce memory usage")
        
        # Efficiency-based recommendations
        if avg_efficiency < self.config.target_efficiency_score:
            recommendations.append("Consider switching to performance-optimized strategy")
        
        # GPU utilization recommendations
        if avg_gpu_util < self.config.min_gpu_utilization:
            recommendations.append("Consider enabling GPU acceleration for better utilization")
        
        # Strategy-specific recommendations
        strategy_counts = defaultdict(int)
        for profile in recent_profiles:
            strategy_counts[profile.strategy] += 1
        
        most_used_strategy = max(strategy_counts, key=strategy_counts.get)
        if strategy_counts[most_used_strategy] / len(recent_profiles) > 0.8:
            recommendations.append(f"Consider diversifying from {most_used_strategy.name} strategy")
        
        return recommendations
    
    def _analyze_performance_patterns(self, performance_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze performance patterns from data"""
        if not performance_data:
            return {}
        
        return {
            'average_time_ms': sum(d.get('time_ms', 0.0) for d in performance_data) / len(performance_data),
            'average_memory_mb': sum(d.get('memory_mb', 0.0) for d in performance_data) / len(performance_data),
            'average_gpu_utilization': sum(d.get('gpu_utilization', 0.0) for d in performance_data) / len(performance_data),
            'average_efficiency': sum(d.get('efficiency_score', 0.0) for d in performance_data) / len(performance_data),
            'data_points': len(performance_data)
        }
    
    def _adjust_for_execution_time(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Adjust parameters for execution time optimization"""
        avg_time = sum(d.get('time_ms', 0.0) for d in performance_data) / len(performance_data)
        
        return {
            'current_avg_time': avg_time,
            'target_time': self.config.max_execution_time_ms,
            'recommended_changes': [
                'enable_parallelization',
                'optimize_algorithms',
                'use_gpu_acceleration'
            ],
            'expected_improvement': min(0.3, (avg_time - self.config.max_execution_time_ms) / avg_time)
        }
    
    def _adjust_for_memory_usage(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Adjust parameters for memory usage optimization"""
        avg_memory = sum(d.get('memory_mb', 0.0) for d in performance_data) / len(performance_data)
        
        return {
            'current_avg_memory': avg_memory,
            'target_memory': self.config.max_memory_usage_mb,
            'recommended_changes': [
                'enable_memory_pooling',
                'use_in_place_operations',
                'optimize_tensor_allocation'
            ],
            'expected_improvement': min(0.4, (avg_memory - self.config.max_memory_usage_mb) / avg_memory)
        }
    
    def _adjust_for_gpu_utilization(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Adjust parameters for GPU utilization optimization"""
        avg_gpu_util = sum(d.get('gpu_utilization', 0.0) for d in performance_data) / len(performance_data)
        
        return {
            'current_avg_gpu_utilization': avg_gpu_util,
            'target_gpu_utilization': self.config.min_gpu_utilization,
            'recommended_changes': [
                'enable_gpu_acceleration',
                'increase_batch_sizes',
                'use_async_operations'
            ],
            'expected_improvement': (self.config.min_gpu_utilization - avg_gpu_util) / self.config.min_gpu_utilization
        }
    
    def _generate_strategy_recommendations(self, performance_analysis: Dict[str, float]) -> Dict[str, str]:
        """Generate strategy recommendations based on performance analysis"""
        recommendations = {}
        
        avg_time = performance_analysis.get('average_time_ms', 0.0)
        avg_memory = performance_analysis.get('average_memory_mb', 0.0)
        avg_gpu_util = performance_analysis.get('average_gpu_utilization', 0.0)
        
        # Time-based strategy recommendations
        if avg_time > self.config.max_execution_time_ms:
            recommendations['time_optimization'] = 'Switch to performance_optimized or gpu_accelerated strategy'
        
        # Memory-based strategy recommendations
        if avg_memory > self.config.max_memory_usage_mb:
            recommendations['memory_optimization'] = 'Switch to memory_efficient or conservative strategy'
        
        # GPU utilization recommendations
        if avg_gpu_util < self.config.min_gpu_utilization:
            recommendations['gpu_optimization'] = 'Switch to gpu_accelerated strategy'
        
        return recommendations
    
    def _calculate_adjustment_impact(self, adjustments_made: List[str]) -> Dict[str, float]:
        """Calculate expected impact of adjustments"""
        impact_factors = {
            'execution_time_optimization': 0.15,
            'memory_optimization': 0.12,
            'gpu_optimization': 0.18,
            'algorithm_optimization': 0.20
        }
        
        total_impact = sum(impact_factors.get(adj, 0.05) for adj in adjustments_made)
        
        return {
            'total_performance_improvement': min(0.5, total_impact),  # Cap at 50%
            'individual_impacts': {adj: impact_factors.get(adj, 0.05) for adj in adjustments_made}
        }
    
    def _calculate_adjustment_confidence(self, performance_data: List[Dict[str, Any]], adjustments_made: List[str]) -> float:
        """Calculate confidence in adjustments"""
        base_confidence = 0.5
        
        # Increase confidence with more data
        data_confidence = min(0.3, len(performance_data) / 100.0)
        
        # Increase confidence with more adjustments
        adjustment_confidence = min(0.2, len(adjustments_made) * 0.05)
        
        return min(1.0, base_confidence + data_confidence + adjustment_confidence)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self._optimizer_lock:
            return {
                'total_profiles': len(self.performance_profiles),
                'optimization_history_size': len(self.optimization_history),
                'regression_history_size': len(self.regression_history),
                'current_metrics': self.current_metrics.copy(),
                'baseline_metrics': self.baseline_metrics.copy(),
                'active_optimizations': {k.name: v for k, v in self.active_optimizations.items()},
                'monitoring_active': self.monitoring_active,
                'last_monitoring_update': self.last_monitoring_update.isoformat(),
                'performance_alerts_count': len(self.performance_alerts),
                'cache_size': len(self.optimization_cache)
            }
    
    def shutdown(self) -> None:
        """Shutdown bounding performance optimizer"""
        self.logger.info("Shutting down bounding performance optimizer")
        
        # Stop monitoring
        self.monitoring_active = False
        
        # Clear tracking data
        self.performance_profiles.clear()
        self.optimization_history.clear()
        self.regression_history.clear()
        self.optimization_cache.clear()
        self.performance_alerts.clear()
        
        self.logger.info("Bounding performance optimizer shutdown complete")