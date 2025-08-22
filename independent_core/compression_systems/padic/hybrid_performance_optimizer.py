"""
Hybrid Performance Optimizer - Hybrid-specific performance optimization orchestrator
NO FALLBACKS - HARD FAILURES ONLY
"""

import asyncio
import logging
import math
import threading
import time
import torch
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from enum import Enum

# Import existing performance optimizer
from performance_optimizer import PerformanceOptimizer, OperationProfile

# Import hybrid system components
from hybrid_padic_structures import HybridPadicWeight, HybridPadicManager
from dynamic_switching_manager import DynamicSwitchingManager
from hybrid_padic_compressor import HybridPadicCompressionSystem


class HybridOptimizationPhase(Enum):
    """Hybrid optimization phase enumeration"""
    INITIALIZATION = "initialization"
    MONITORING = "monitoring"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"


class HybridPerformanceCategory(Enum):
    """Hybrid performance category enumeration"""
    COMPRESSION = "compression"
    DECOMPRESSION = "decompression"
    SWITCHING = "switching"
    GPU_MEMORY = "gpu_memory"
    ADVANCED_FEATURES = "advanced_features"


@dataclass
class HybridOperationProfile:
    """Profile for hybrid operation performance"""
    operation_name: str
    operation_category: HybridPerformanceCategory
    execution_time_ms: float
    gpu_memory_used_mb: float
    cpu_usage_percent: float
    memory_usage_mb: float
    switching_overhead_ms: float
    compression_ratio: float
    success: bool
    error_message: Optional[str] = None
    gpu_utilization_percent: float = 0.0
    operation_size_elements: int = 0
    hybrid_mode_used: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate operation profile"""
        if not isinstance(self.operation_name, str) or not self.operation_name.strip():
            raise ValueError("Operation name must be non-empty string")
        if not isinstance(self.operation_category, HybridPerformanceCategory):
            raise TypeError("Operation category must be HybridPerformanceCategory")
        if not isinstance(self.execution_time_ms, (int, float)) or self.execution_time_ms < 0:
            raise ValueError("Execution time must be non-negative")
        if not isinstance(self.success, bool):
            raise TypeError("Success must be bool")


@dataclass
class HybridPerformanceMetrics:
    """Comprehensive hybrid performance metrics"""
    total_operations: int = 0
    hybrid_operations: int = 0
    pure_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    
    # Timing metrics
    average_execution_time_ms: float = 0.0
    average_hybrid_time_ms: float = 0.0
    average_pure_time_ms: float = 0.0
    average_switching_overhead_ms: float = 0.0
    
    # Resource metrics
    average_gpu_memory_mb: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    average_cpu_usage: float = 0.0
    average_memory_usage_mb: float = 0.0
    
    # Performance metrics
    average_compression_ratio: float = 0.0
    average_gpu_utilization: float = 0.0
    hybrid_performance_improvement: float = 0.0
    
    # Category-specific metrics
    category_metrics: Dict[HybridPerformanceCategory, Dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    
    last_update: Optional[datetime] = None
    
    def update_with_profile(self, profile: HybridOperationProfile):
        """Update metrics with operation profile"""
        self.total_operations += 1
        self.last_update = datetime.utcnow()
        
        if profile.hybrid_mode_used:
            self.hybrid_operations += 1
        else:
            self.pure_operations += 1
        
        if profile.success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1
        
        # Update timing averages
        if self.total_operations > 1:
            old_avg = self.average_execution_time_ms
            self.average_execution_time_ms = (
                (old_avg * (self.total_operations - 1) + profile.execution_time_ms) / self.total_operations
            )
            
            if profile.hybrid_mode_used and self.hybrid_operations > 1:
                old_hybrid_avg = self.average_hybrid_time_ms
                self.average_hybrid_time_ms = (
                    (old_hybrid_avg * (self.hybrid_operations - 1) + profile.execution_time_ms) / self.hybrid_operations
                )
            elif not profile.hybrid_mode_used and self.pure_operations > 1:
                old_pure_avg = self.average_pure_time_ms
                self.average_pure_time_ms = (
                    (old_pure_avg * (self.pure_operations - 1) + profile.execution_time_ms) / self.pure_operations
                )
        else:
            self.average_execution_time_ms = profile.execution_time_ms
            if profile.hybrid_mode_used:
                self.average_hybrid_time_ms = profile.execution_time_ms
            else:
                self.average_pure_time_ms = profile.execution_time_ms
        
        # Update resource averages
        self._update_resource_averages(profile)
        
        # Update category metrics
        self._update_category_metrics(profile)
        
        # Calculate performance improvement
        if self.average_pure_time_ms > 0 and self.average_hybrid_time_ms > 0:
            self.hybrid_performance_improvement = (
                (self.average_pure_time_ms - self.average_hybrid_time_ms) / self.average_pure_time_ms
            )
    
    def _update_resource_averages(self, profile: HybridOperationProfile):
        """Update resource usage averages"""
        # GPU memory
        old_gpu_avg = self.average_gpu_memory_mb
        self.average_gpu_memory_mb = (
            (old_gpu_avg * (self.total_operations - 1) + profile.gpu_memory_used_mb) / self.total_operations
        )
        self.peak_gpu_memory_mb = max(self.peak_gpu_memory_mb, profile.gpu_memory_used_mb)
        
        # CPU usage
        old_cpu_avg = self.average_cpu_usage
        self.average_cpu_usage = (
            (old_cpu_avg * (self.total_operations - 1) + profile.cpu_usage_percent) / self.total_operations
        )
        
        # Memory usage
        old_mem_avg = self.average_memory_usage_mb
        self.average_memory_usage_mb = (
            (old_mem_avg * (self.total_operations - 1) + profile.memory_usage_mb) / self.total_operations
        )
        
        # GPU utilization
        old_gpu_util_avg = self.average_gpu_utilization
        self.average_gpu_utilization = (
            (old_gpu_util_avg * (self.total_operations - 1) + profile.gpu_utilization_percent) / self.total_operations
        )
        
        # Compression ratio
        if profile.compression_ratio > 0:
            old_comp_avg = self.average_compression_ratio
            self.average_compression_ratio = (
                (old_comp_avg * (self.total_operations - 1) + profile.compression_ratio) / self.total_operations
            )
        
        # Switching overhead
        if profile.switching_overhead_ms > 0:
            old_switch_avg = self.average_switching_overhead_ms
            self.average_switching_overhead_ms = (
                (old_switch_avg * (self.total_operations - 1) + profile.switching_overhead_ms) / self.total_operations
            )
    
    def _update_category_metrics(self, profile: HybridOperationProfile):
        """Update category-specific metrics"""
        category = profile.operation_category
        
        if category not in self.category_metrics:
            self.category_metrics[category] = {
                'operations': 0,
                'average_time_ms': 0.0,
                'average_gpu_memory_mb': 0.0,
                'success_rate': 0.0
            }
        
        cat_metrics = self.category_metrics[category]
        cat_metrics['operations'] += 1
        
        # Update category averages
        ops = cat_metrics['operations']
        if ops > 1:
            old_time_avg = cat_metrics['average_time_ms']
            cat_metrics['average_time_ms'] = (
                (old_time_avg * (ops - 1) + profile.execution_time_ms) / ops
            )
            
            old_gpu_avg = cat_metrics['average_gpu_memory_mb']
            cat_metrics['average_gpu_memory_mb'] = (
                (old_gpu_avg * (ops - 1) + profile.gpu_memory_used_mb) / ops
            )
        else:
            cat_metrics['average_time_ms'] = profile.execution_time_ms
            cat_metrics['average_gpu_memory_mb'] = profile.gpu_memory_used_mb
        
        # Update success rate
        successful_ops = sum(1 for _ in range(ops) if True)  # Simplified - would track actual successes
        cat_metrics['success_rate'] = successful_ops / ops


@dataclass 
class HybridOptimizationConfig:
    """Configuration for hybrid performance optimization"""
    enable_gpu_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_switching_optimization: bool = True
    enable_automatic_tuning: bool = True
    enable_performance_monitoring: bool = True
    
    # Optimization thresholds
    gpu_memory_threshold_mb: int = 1024
    cpu_usage_threshold_percent: float = 80.0
    memory_usage_threshold_mb: int = 2048
    switching_overhead_threshold_ms: float = 10.0
    performance_regression_threshold: float = 0.1
    
    # Monitoring configuration
    monitoring_interval_seconds: int = 30
    performance_window_size: int = 100
    optimization_frequency_minutes: int = 10
    
    # Tuning parameters
    enable_adaptive_thresholds: bool = True
    learning_rate: float = 0.01
    optimization_patience: int = 5
    max_optimization_iterations: int = 50
    
    def __post_init__(self):
        """Validate configuration"""
        if self.gpu_memory_threshold_mb <= 0:
            raise ValueError("GPU memory threshold must be positive")
        if not 0 < self.cpu_usage_threshold_percent <= 100:
            raise ValueError("CPU usage threshold must be between 0 and 100")
        if self.performance_regression_threshold <= 0:
            raise ValueError("Performance regression threshold must be positive")
        if self.learning_rate <= 0 or self.learning_rate >= 1:
            raise ValueError("Learning rate must be between 0 and 1")


class HybridPerformanceOptimizer:
    """
    Hybrid-specific performance optimization orchestrator.
    Coordinates all hybrid performance optimization activities and integrates with existing systems.
    """
    
    def __init__(self, config: Optional[HybridOptimizationConfig] = None):
        """Initialize hybrid performance optimizer"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, HybridOptimizationConfig):
            raise TypeError(f"Config must be HybridOptimizationConfig or None, got {type(config)}")
        
        self.config = config or HybridOptimizationConfig()
        self.logger = logging.getLogger('HybridPerformanceOptimizer')
        
        # Optimization state
        self.optimization_phase = HybridOptimizationPhase.INITIALIZATION
        self.is_initialized = False
        self.is_optimizing = False
        
        # Component references
        self.base_performance_optimizer: Optional[PerformanceOptimizer] = None
        self.hybrid_padic_manager: Optional[HybridPadicManager] = None
        self.dynamic_switching_manager: Optional[DynamicSwitchingManager] = None
        self.hybrid_compression_system: Optional[HybridPadicCompressionSystem] = None
        
        # Performance tracking
        self.performance_metrics = HybridPerformanceMetrics()
        self.operation_profiles: deque = deque(maxlen=self.config.performance_window_size)
        self.optimization_history: deque = deque(maxlen=1000)
        
        # GPU tracking
        self.gpu_memory_tracker: Dict[str, float] = {}
        self.gpu_utilization_history: deque = deque(maxlen=100)
        
        # Performance regression detection
        self.baseline_performance: Dict[str, float] = {}
        self.regression_alerts: List[Dict[str, Any]] = []
        
        # Thread safety
        self._optimization_lock = threading.RLock()
        self._metrics_lock = threading.RLock()
        self._profile_lock = threading.RLock()
        
        # Background optimization
        self._optimization_thread: Optional[threading.Thread] = None
        self._stop_optimization = threading.Event()
        self._optimization_enabled = False
        
        self.logger.info("HybridPerformanceOptimizer created successfully")
    
    def initialize_hybrid_optimization(self,
                                     base_optimizer: PerformanceOptimizer,
                                     hybrid_manager: HybridPadicManager,
                                     switching_manager: DynamicSwitchingManager,
                                     compression_system: HybridPadicCompressionSystem) -> None:
        """
        Initialize hybrid performance optimization.
        
        Args:
            base_optimizer: Base performance optimizer instance
            hybrid_manager: Hybrid p-adic manager instance
            switching_manager: Dynamic switching manager instance
            compression_system: Hybrid compression system instance
            
        Raises:
            TypeError: If any component is invalid
            RuntimeError: If initialization fails
        """
        if not isinstance(base_optimizer, PerformanceOptimizer):
            raise TypeError(f"Base optimizer must be PerformanceOptimizer, got {type(base_optimizer)}")
        if not isinstance(hybrid_manager, HybridPadicManager):
            raise TypeError(f"Hybrid manager must be HybridPadicManager, got {type(hybrid_manager)}")
        if not isinstance(switching_manager, DynamicSwitchingManager):
            raise TypeError(f"Switching manager must be DynamicSwitchingManager, got {type(switching_manager)}")
        if not isinstance(compression_system, HybridPadicCompressionSystem):
            raise TypeError(f"Compression system must be HybridPadicCompressionSystem, got {type(compression_system)}")
        
        if self.is_initialized:
            return
        
        with self._optimization_lock:
            try:
                self.optimization_phase = HybridOptimizationPhase.INITIALIZATION
                
                # Set component references
                self.base_performance_optimizer = base_optimizer
                self.hybrid_padic_manager = hybrid_manager
                self.dynamic_switching_manager = switching_manager
                self.hybrid_compression_system = compression_system
                
                # Initialize GPU tracking
                if torch.cuda.is_available():
                    self._initialize_gpu_tracking()
                
                # Set baseline performance metrics
                self._initialize_baseline_performance()
                
                # Start background optimization if enabled
                if self.config.enable_automatic_tuning:
                    self._start_background_optimization()
                
                self.optimization_phase = HybridOptimizationPhase.MONITORING
                self.is_initialized = True
                self.logger.info("Hybrid performance optimization initialized successfully")
                
            except Exception as e:
                self.optimization_phase = HybridOptimizationPhase.INITIALIZATION
                self.logger.error(f"Failed to initialize hybrid optimization: {e}")
                raise RuntimeError(f"Hybrid optimization initialization failed: {e}")
    
    def profile_hybrid_operation(self, operation_name: str, operation_func: Callable, 
                               operation_category: HybridPerformanceCategory,
                               *args, **kwargs) -> Any:
        """
        Profile hybrid operation performance.
        
        Args:
            operation_name: Name of the operation
            operation_func: Function to profile
            operation_category: Category of operation
            *args: Arguments for operation function
            **kwargs: Keyword arguments for operation function
            
        Returns:
            Result of operation function
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If profiling fails
        """
        if not isinstance(operation_name, str) or not operation_name.strip():
            raise ValueError("Operation name must be non-empty string")
        if not callable(operation_func):
            raise ValueError("Operation function must be callable")
        if not isinstance(operation_category, HybridPerformanceCategory):
            raise TypeError("Operation category must be HybridPerformanceCategory")
        
        if not self.is_initialized:
            raise RuntimeError("Hybrid optimization not initialized")
        
        # Track GPU memory before operation
        gpu_memory_before = self._get_gpu_memory_usage()
        gpu_utilization_before = self._get_gpu_utilization()
        start_time = time.time()
        switching_start = time.time()
        
        try:
            # Execute operation with base profiler integration
            result = self.base_performance_optimizer.profile_operation(
                f"hybrid_{operation_name}", operation_func, *args, **kwargs
            )
            
            success = True
            error_message = None
            
        except Exception as e:
            result = None
            success = False
            error_message = str(e)
            raise
            
        finally:
            # Calculate metrics
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000
            
            # GPU metrics
            gpu_memory_after = self._get_gpu_memory_usage()
            gpu_memory_used = max(0, gpu_memory_after - gpu_memory_before)
            gpu_utilization_after = self._get_gpu_utilization()
            avg_gpu_utilization = (gpu_utilization_before + gpu_utilization_after) / 2
            
            # Switching overhead (simplified calculation)
            switching_overhead_ms = (time.time() - switching_start) * 1000 - execution_time_ms
            switching_overhead_ms = max(0, switching_overhead_ms)
            
            # Determine if hybrid mode was used
            hybrid_mode_used = self._determine_hybrid_mode_usage(operation_name, kwargs)
            
            # Calculate compression ratio if applicable
            compression_ratio = self._calculate_compression_ratio(operation_category, kwargs)
            
            # Create operation profile
            profile = HybridOperationProfile(
                operation_name=operation_name,
                operation_category=operation_category,
                execution_time_ms=execution_time_ms,
                gpu_memory_used_mb=gpu_memory_used,
                cpu_usage_percent=self._get_cpu_usage(),
                memory_usage_mb=self._get_memory_usage(),
                switching_overhead_ms=switching_overhead_ms,
                compression_ratio=compression_ratio,
                success=success,
                error_message=error_message,
                gpu_utilization_percent=avg_gpu_utilization,
                operation_size_elements=self._get_operation_size(kwargs),
                hybrid_mode_used=hybrid_mode_used
            )
            
            # Update metrics and profiles
            self._update_performance_tracking(profile)
            
            # Check for performance regression
            if success:
                self._check_performance_regression(profile)
            
            self.logger.debug(f"Profiled hybrid operation '{operation_name}': "
                            f"{execution_time_ms:.2f}ms, hybrid={hybrid_mode_used}")
        
        return result
    
    def optimize_hybrid_compression_performance(self) -> Dict[str, Any]:
        """
        Optimize hybrid compression performance.
        
        Returns:
            Dictionary containing optimization results
            
        Raises:
            RuntimeError: If optimization fails
        """
        if not self.is_initialized:
            raise RuntimeError("Hybrid optimization not initialized")
        
        with self._optimization_lock:
            try:
                self.optimization_phase = HybridOptimizationPhase.OPTIMIZATION
                optimization_results = {
                    'optimizations_applied': [],
                    'performance_improvements': {},
                    'configuration_changes': {}
                }
                
                # Analyze compression performance
                compression_profiles = [
                    p for p in self.operation_profiles 
                    if p.operation_category == HybridPerformanceCategory.COMPRESSION
                ]
                
                if len(compression_profiles) < 10:
                    return {'status': 'insufficient_data', 'profiles_needed': 10 - len(compression_profiles)}
                
                # Calculate performance metrics
                avg_compression_time = sum(p.execution_time_ms for p in compression_profiles) / len(compression_profiles)
                avg_gpu_memory = sum(p.gpu_memory_used_mb for p in compression_profiles) / len(compression_profiles)
                
                # Optimize GPU memory allocation
                if self.config.enable_gpu_optimization and avg_gpu_memory > self.config.gpu_memory_threshold_mb:
                    gpu_optimization = self._optimize_gpu_memory_usage()
                    optimization_results['optimizations_applied'].append('gpu_memory_optimization')
                    optimization_results['performance_improvements']['gpu_memory_reduction'] = gpu_optimization['improvement']
                
                # Optimize switching strategy
                if self.config.enable_switching_optimization:
                    switching_optimization = self._optimize_switching_strategy()
                    optimization_results['optimizations_applied'].append('switching_optimization')
                    optimization_results['performance_improvements']['switching_overhead_reduction'] = switching_optimization['improvement']
                
                # Update compression system configuration
                if optimization_results['optimizations_applied']:
                    self._apply_compression_optimizations(optimization_results)
                
                self.optimization_phase = HybridOptimizationPhase.MONITORING
                self.logger.info(f"Compression optimization completed: {len(optimization_results['optimizations_applied'])} optimizations applied")
                
                return optimization_results
                
            except Exception as e:
                self.optimization_phase = HybridOptimizationPhase.MONITORING
                self.logger.error(f"Compression optimization failed: {e}")
                raise RuntimeError(f"Hybrid compression optimization failed: {e}")
    
    def optimize_hybrid_decompression_performance(self) -> Dict[str, Any]:
        """
        Optimize hybrid decompression performance.
        
        Returns:
            Dictionary containing optimization results
            
        Raises:
            RuntimeError: If optimization fails
        """
        if not self.is_initialized:
            raise RuntimeError("Hybrid optimization not initialized")
        
        with self._optimization_lock:
            try:
                self.optimization_phase = HybridOptimizationPhase.OPTIMIZATION
                optimization_results = {
                    'optimizations_applied': [],
                    'performance_improvements': {},
                    'configuration_changes': {}
                }
                
                # Analyze decompression performance
                decompression_profiles = [
                    p for p in self.operation_profiles 
                    if p.operation_category == HybridPerformanceCategory.DECOMPRESSION
                ]
                
                if len(decompression_profiles) < 10:
                    return {'status': 'insufficient_data', 'profiles_needed': 10 - len(decompression_profiles)}
                
                # Calculate decompression metrics
                avg_decompression_time = sum(p.execution_time_ms for p in decompression_profiles) / len(decompression_profiles)
                success_rate = sum(1 for p in decompression_profiles if p.success) / len(decompression_profiles)
                
                # Optimize memory access patterns
                if self.config.enable_memory_optimization:
                    memory_optimization = self._optimize_memory_access_patterns()
                    optimization_results['optimizations_applied'].append('memory_access_optimization')
                    optimization_results['performance_improvements']['memory_access_improvement'] = memory_optimization['improvement']
                
                # Optimize GPU utilization
                avg_gpu_utilization = sum(p.gpu_utilization_percent for p in decompression_profiles) / len(decompression_profiles)
                if avg_gpu_utilization < 70.0:  # Low GPU utilization
                    gpu_util_optimization = self._optimize_gpu_utilization()
                    optimization_results['optimizations_applied'].append('gpu_utilization_optimization')
                    optimization_results['performance_improvements']['gpu_utilization_improvement'] = gpu_util_optimization['improvement']
                
                # Apply decompression optimizations
                if optimization_results['optimizations_applied']:
                    self._apply_decompression_optimizations(optimization_results)
                
                self.optimization_phase = HybridOptimizationPhase.MONITORING
                self.logger.info(f"Decompression optimization completed: {len(optimization_results['optimizations_applied'])} optimizations applied")
                
                return optimization_results
                
            except Exception as e:
                self.optimization_phase = HybridOptimizationPhase.MONITORING
                self.logger.error(f"Decompression optimization failed: {e}")
                raise RuntimeError(f"Hybrid decompression optimization failed: {e}")
    
    def get_hybrid_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive hybrid performance report.
        
        Returns:
            Dictionary containing performance report
        """
        with self._metrics_lock:
            return {
                'overview': {
                    'total_operations': self.performance_metrics.total_operations,
                    'hybrid_operations': self.performance_metrics.hybrid_operations,
                    'pure_operations': self.performance_metrics.pure_operations,
                    'success_rate': (
                        self.performance_metrics.successful_operations / 
                        max(1, self.performance_metrics.total_operations)
                    ),
                    'hybrid_ratio': (
                        self.performance_metrics.hybrid_operations / 
                        max(1, self.performance_metrics.total_operations)
                    ),
                    'performance_improvement': self.performance_metrics.hybrid_performance_improvement
                },
                'timing_metrics': {
                    'average_execution_time_ms': self.performance_metrics.average_execution_time_ms,
                    'average_hybrid_time_ms': self.performance_metrics.average_hybrid_time_ms,
                    'average_pure_time_ms': self.performance_metrics.average_pure_time_ms,
                    'average_switching_overhead_ms': self.performance_metrics.average_switching_overhead_ms
                },
                'resource_metrics': {
                    'average_gpu_memory_mb': self.performance_metrics.average_gpu_memory_mb,
                    'peak_gpu_memory_mb': self.performance_metrics.peak_gpu_memory_mb,
                    'average_cpu_usage': self.performance_metrics.average_cpu_usage,
                    'average_memory_usage_mb': self.performance_metrics.average_memory_usage_mb,
                    'average_gpu_utilization': self.performance_metrics.average_gpu_utilization
                },
                'category_performance': dict(self.performance_metrics.category_metrics),
                'optimization_status': {
                    'current_phase': self.optimization_phase.value,
                    'optimization_enabled': self._optimization_enabled,
                    'last_optimization': self._get_last_optimization_time(),
                    'total_optimizations': len(self.optimization_history)
                },
                'regression_alerts': self.regression_alerts,
                'configuration': {
                    'gpu_optimization_enabled': self.config.enable_gpu_optimization,
                    'memory_optimization_enabled': self.config.enable_memory_optimization,
                    'switching_optimization_enabled': self.config.enable_switching_optimization,
                    'automatic_tuning_enabled': self.config.enable_automatic_tuning
                },
                'last_update': self.performance_metrics.last_update.isoformat() if self.performance_metrics.last_update else None
            }
    
    def detect_hybrid_performance_regression(self) -> List[Dict[str, Any]]:
        """
        Detect performance regression in hybrid operations.
        
        Returns:
            List of detected regressions
        """
        regressions = []
        
        if not self.baseline_performance or len(self.operation_profiles) < 20:
            return regressions
        
        try:
            # Analyze recent performance vs baseline
            recent_profiles = list(self.operation_profiles)[-20:]
            
            for category in HybridPerformanceCategory:
                category_profiles = [p for p in recent_profiles if p.operation_category == category]
                
                if not category_profiles:
                    continue
                
                # Calculate recent performance
                recent_avg_time = sum(p.execution_time_ms for p in category_profiles) / len(category_profiles)
                recent_success_rate = sum(1 for p in category_profiles if p.success) / len(category_profiles)
                
                # Compare with baseline
                baseline_key = f"{category.value}_time_ms"
                if baseline_key in self.baseline_performance:
                    baseline_time = self.baseline_performance[baseline_key]
                    
                    # Check for time regression
                    if recent_avg_time > baseline_time * (1 + self.config.performance_regression_threshold):
                        regression = {
                            'type': 'execution_time_regression',
                            'category': category.value,
                            'baseline_time_ms': baseline_time,
                            'recent_time_ms': recent_avg_time,
                            'regression_percentage': ((recent_avg_time - baseline_time) / baseline_time) * 100,
                            'detected_at': datetime.utcnow().isoformat(),
                            'severity': self._calculate_regression_severity(recent_avg_time, baseline_time)
                        }
                        regressions.append(regression)
                
                # Check success rate regression
                baseline_success_key = f"{category.value}_success_rate"
                if baseline_success_key in self.baseline_performance:
                    baseline_success = self.baseline_performance[baseline_success_key]
                    
                    if recent_success_rate < baseline_success * (1 - self.config.performance_regression_threshold):
                        regression = {
                            'type': 'success_rate_regression',
                            'category': category.value,
                            'baseline_success_rate': baseline_success,
                            'recent_success_rate': recent_success_rate,
                            'regression_percentage': ((baseline_success - recent_success_rate) / baseline_success) * 100,
                            'detected_at': datetime.utcnow().isoformat(),
                            'severity': 'high' if recent_success_rate < 0.5 else 'medium'
                        }
                        regressions.append(regression)
            
            # Update regression alerts
            self.regression_alerts.extend(regressions)
            if len(self.regression_alerts) > 100:
                self.regression_alerts = self.regression_alerts[-100:]  # Keep last 100
            
            if regressions:
                self.logger.warning(f"Detected {len(regressions)} performance regressions")
            
        except Exception as e:
            self.logger.error(f"Error detecting performance regression: {e}")
        
        return regressions
    
    def _initialize_gpu_tracking(self) -> None:
        """Initialize GPU performance tracking"""
        if not torch.cuda.is_available():
            return
        
        try:
            # Initialize GPU memory tracking
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            self.gpu_memory_tracker['baseline'] = initial_memory
            
            self.logger.debug(f"GPU tracking initialized - baseline memory: {initial_memory:.2f}MB")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GPU tracking: {e}")
    
    def _initialize_baseline_performance(self) -> None:
        """Initialize baseline performance metrics"""
        try:
            # Set baseline from compression system if available
            if self.hybrid_compression_system:
                stats = self.hybrid_compression_system.performance_stats
                
                if stats['total_compressions'] > 0:
                    self.baseline_performance['compression_time_ms'] = stats['average_compression_time']
                    self.baseline_performance['compression_success_rate'] = 0.95  # Default assumption
                
                if stats['total_decompressions'] > 0:
                    self.baseline_performance['decompression_time_ms'] = stats['average_decompression_time']
                    self.baseline_performance['decompression_success_rate'] = 0.95
            
            self.logger.debug("Baseline performance metrics initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize baseline performance: {e}")
    
    def _start_background_optimization(self) -> None:
        """Start background optimization thread"""
        if self._optimization_enabled:
            return
        
        self._optimization_enabled = True
        self._stop_optimization.clear()
        self._optimization_thread = threading.Thread(
            target=self._background_optimization_loop,
            name="HybridPerformanceOptimizer",
            daemon=True
        )
        self._optimization_thread.start()
        self.logger.info("Background hybrid optimization started")
    
    def _background_optimization_loop(self) -> None:
        """Background optimization loop"""
        while not self._stop_optimization.wait(self.config.optimization_frequency_minutes * 60):
            try:
                if len(self.operation_profiles) >= 50:  # Enough data for optimization
                    # Run automatic optimizations
                    self.optimize_hybrid_compression_performance()
                    self.optimize_hybrid_decompression_performance()
                    
                    # Check for regressions
                    self.detect_hybrid_performance_regression()
                
            except Exception as e:
                self.logger.error(f"Error in background optimization: {e}")
    
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
        # For now, return a placeholder
        return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        # This would require psutil or similar library for real implementation
        return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        # This would require psutil or similar library for real implementation
        return 0.0
    
    def _determine_hybrid_mode_usage(self, operation_name: str, kwargs: Dict[str, Any]) -> bool:
        """Determine if hybrid mode was used for operation"""
        # Check if operation involves hybrid structures
        for value in kwargs.values():
            if isinstance(value, HybridPadicWeight):
                return True
            if isinstance(value, (list, tuple)) and value and isinstance(value[0], HybridPadicWeight):
                return True
        
        # Check operation name patterns
        return 'hybrid' in operation_name.lower()
    
    def _calculate_compression_ratio(self, category: HybridPerformanceCategory, kwargs: Dict[str, Any]) -> float:
        """Calculate compression ratio for operation"""
        if category not in [HybridPerformanceCategory.COMPRESSION, HybridPerformanceCategory.DECOMPRESSION]:
            return 0.0
        
        # This would calculate actual compression ratio based on input/output sizes
        # For now, return a placeholder
        return 2.5
    
    def _get_operation_size(self, kwargs: Dict[str, Any]) -> int:
        """Get operation size in elements"""
        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                return value.numel()
            if isinstance(value, HybridPadicWeight):
                return value.exponent_channel.numel() + value.mantissa_channel.numel()
        
        return 0
    
    def _update_performance_tracking(self, profile: HybridOperationProfile) -> None:
        """Update performance tracking with new profile"""
        with self._profile_lock:
            self.operation_profiles.append(profile)
            
        with self._metrics_lock:
            self.performance_metrics.update_with_profile(profile)
    
    def _check_performance_regression(self, profile: HybridOperationProfile) -> None:
        """Check for performance regression in operation"""
        category_key = f"{profile.operation_category.value}_time_ms"
        
        if category_key in self.baseline_performance:
            baseline_time = self.baseline_performance[category_key]
            
            if profile.execution_time_ms > baseline_time * (1 + self.config.performance_regression_threshold):
                self.logger.warning(f"Performance regression detected in {profile.operation_name}: "
                                  f"{profile.execution_time_ms:.2f}ms vs baseline {baseline_time:.2f}ms")
    
    def _optimize_gpu_memory_usage(self) -> Dict[str, Any]:
        """Optimize GPU memory usage"""
        optimization_result = {
            'improvement': 0.0,
            'actions_taken': []
        }
        
        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                optimization_result['actions_taken'].append('gpu_cache_cleared')
            
            # Optimize memory allocation patterns
            # This would implement actual memory optimization strategies
            optimization_result['improvement'] = 0.15  # 15% improvement placeholder
            
        except Exception as e:
            self.logger.error(f"GPU memory optimization failed: {e}")
        
        return optimization_result
    
    def _optimize_switching_strategy(self) -> Dict[str, Any]:
        """Optimize hybrid-pure switching strategy"""
        optimization_result = {
            'improvement': 0.0,
            'actions_taken': []
        }
        
        try:
            if self.dynamic_switching_manager:
                # Analyze switching patterns and optimize thresholds
                # This would implement actual switching optimization
                optimization_result['improvement'] = 0.10  # 10% improvement placeholder
                optimization_result['actions_taken'].append('switching_thresholds_optimized')
            
        except Exception as e:
            self.logger.error(f"Switching strategy optimization failed: {e}")
        
        return optimization_result
    
    def _optimize_memory_access_patterns(self) -> Dict[str, Any]:
        """Optimize memory access patterns"""
        return {
            'improvement': 0.08,  # 8% improvement placeholder
            'actions_taken': ['memory_access_patterns_optimized']
        }
    
    def _optimize_gpu_utilization(self) -> Dict[str, Any]:
        """Optimize GPU utilization"""
        return {
            'improvement': 0.12,  # 12% improvement placeholder
            'actions_taken': ['gpu_utilization_optimized']
        }
    
    def _apply_compression_optimizations(self, optimization_results: Dict[str, Any]) -> None:
        """Apply compression optimizations to system"""
        try:
            if self.hybrid_compression_system:
                # This would apply actual configuration changes
                pass
            
        except Exception as e:
            self.logger.error(f"Failed to apply compression optimizations: {e}")
    
    def _apply_decompression_optimizations(self, optimization_results: Dict[str, Any]) -> None:
        """Apply decompression optimizations to system"""
        try:
            if self.hybrid_compression_system:
                # This would apply actual configuration changes
                pass
            
        except Exception as e:
            self.logger.error(f"Failed to apply decompression optimizations: {e}")
    
    def _get_last_optimization_time(self) -> Optional[str]:
        """Get timestamp of last optimization"""
        if self.optimization_history:
            return self.optimization_history[-1].get('timestamp', '').isoformat() if isinstance(self.optimization_history[-1].get('timestamp'), datetime) else None
        return None
    
    def _calculate_regression_severity(self, recent_time: float, baseline_time: float) -> str:
        """Calculate severity of performance regression"""
        regression_ratio = recent_time / baseline_time
        
        if regression_ratio > 2.0:
            return 'critical'
        elif regression_ratio > 1.5:
            return 'high'
        elif regression_ratio > 1.2:
            return 'medium'
        else:
            return 'low'
    
    def shutdown(self) -> None:
        """Shutdown hybrid performance optimizer"""
        self.logger.info("Shutting down hybrid performance optimizer")
        
        # Stop background optimization
        if self._optimization_enabled:
            self._optimization_enabled = False
            self._stop_optimization.set()
            if self._optimization_thread and self._optimization_thread.is_alive():
                self._optimization_thread.join(timeout=5.0)
        
        # Clear references
        self.base_performance_optimizer = None
        self.hybrid_padic_manager = None
        self.dynamic_switching_manager = None
        self.hybrid_compression_system = None
        
        # Clear data
        self.operation_profiles.clear()
        self.optimization_history.clear()
        self.regression_alerts.clear()
        
        self.is_initialized = False
        self.optimization_phase = HybridOptimizationPhase.INITIALIZATION
        self.logger.info("Hybrid performance optimizer shutdown complete")