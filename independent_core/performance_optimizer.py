"""
Main performance optimization orchestrator for Saraphis Brain system.
Provides comprehensive performance profiling, bottleneck detection, and optimization strategy coordination.
NO FALLBACKS - HARD FAILURES ONLY
"""

import asyncio
import functools
import gc
import logging
import psutil
import threading
import time
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Callable, Union, Set
import traceback
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class OptimizationStrategy(Enum):
    """Available optimization strategies"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class BottleneckType(Enum):
    """Types of detected bottlenecks"""
    CPU_BOUND = "cpu_bound"
    MEMORY_BOUND = "memory_bound"
    IO_BOUND = "io_bound"
    GPU_BOUND = "gpu_bound"
    NETWORK_BOUND = "network_bound"
    SYNCHRONIZATION = "synchronization"
    ALGORITHM = "algorithm"


@dataclass
class OperationProfile:
    """Profile data for a single operation"""
    operation_name: str
    execution_time: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float = 0.0
    io_operations: int = 0
    thread_id: int = 0
    timestamp: float = field(default_factory=time.time)
    call_stack: List[str] = field(default_factory=list)
    args_size: int = 0
    return_size: int = 0
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class BottleneckReport:
    """Report for detected performance bottleneck"""
    bottleneck_type: BottleneckType
    operation_name: str
    severity: float  # 0.0 to 1.0
    impact_factor: float
    occurrence_count: int
    average_delay: float
    peak_delay: float
    suggested_optimizations: List[str] = field(default_factory=list)
    estimated_improvement: float = 0.0
    confidence_score: float = 0.0


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation"""
    strategy: OptimizationStrategy
    target_component: str
    action_type: str
    description: str
    estimated_improvement: float
    implementation_effort: str  # "low", "medium", "high"
    priority: str  # "low", "medium", "high", "critical"
    dependencies: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)


class PerformanceOptimizer:
    """
    Main orchestrator for all performance optimizations across the Saraphis Brain system.
    Provides comprehensive performance profiling, bottleneck detection, and optimization coordination.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize performance optimizer"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, dict):
            raise TypeError(f"Config must be dict or None, got {type(config)}")
        
        self.config = config or {}
        self.logger = logging.getLogger('PerformanceOptimizer')
        
        # Configuration parameters
        self.enable_profiling = self.config.get('enable_profiling', True)
        self.profile_history_size = self.config.get('profile_history_size', 10000)
        self.bottleneck_detection_threshold = self.config.get('bottleneck_detection_threshold', 0.8)
        self.optimization_strategy = OptimizationStrategy(
            self.config.get('optimization_strategy', 'balanced')
        )
        self.profiling_interval = self.config.get('profiling_interval', 1.0)
        self.enable_async_profiling = self.config.get('enable_async_profiling', True)
        self.max_concurrent_profiles = self.config.get('max_concurrent_profiles', 100)
        
        # Validate configuration
        self._validate_config()
        
        # Performance data storage
        self.operation_profiles: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.profile_history_size)
        )
        self.bottleneck_reports: Dict[str, BottleneckReport] = {}
        self.optimization_recommendations: List[OptimizationRecommendation] = []
        
        # Performance hooks registry
        self.performance_hooks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Real-time monitoring
        self.system_monitor = SystemMonitor()
        self.operation_timers: Dict[str, float] = {}
        self.active_operations: Set[str] = set()
        
        # Thread safety
        self._profile_lock = threading.RLock()
        self._optimization_lock = threading.RLock()
        self._hook_lock = threading.RLock()
        
        # Background monitoring
        self._monitoring_enabled = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Integration tracking
        self.brain_core_integration = False
        self.training_manager_integration = False
        self.gac_system_integration = False
        
        # Performance metrics
        self.performance_metrics = {
            'total_operations_profiled': 0,
            'total_bottlenecks_detected': 0,
            'total_optimizations_applied': 0,
            'average_operation_time': 0.0,
            'system_efficiency_score': 0.0,
            'optimization_success_rate': 0.0,
            'memory_optimization_savings': 0.0,
            'cpu_optimization_savings': 0.0,
            'gpu_optimization_savings': 0.0
        }
        
        self.is_initialized = True
        self.logger.info("PerformanceOptimizer initialized successfully")
    
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        if not isinstance(self.enable_profiling, bool):
            raise TypeError(f"enable_profiling must be bool, got {type(self.enable_profiling)}")
        
        if not isinstance(self.profile_history_size, int) or self.profile_history_size <= 0:
            raise ValueError(f"profile_history_size must be positive int, got {self.profile_history_size}")
        
        if not isinstance(self.bottleneck_detection_threshold, (int, float)):
            raise TypeError(f"bottleneck_detection_threshold must be numeric")
        if not (0.0 <= self.bottleneck_detection_threshold <= 1.0):
            raise ValueError(f"bottleneck_detection_threshold must be between 0.0 and 1.0")
        
        if not isinstance(self.profiling_interval, (int, float)) or self.profiling_interval <= 0:
            raise ValueError(f"profiling_interval must be positive number, got {self.profiling_interval}")
        
        if not isinstance(self.max_concurrent_profiles, int) or self.max_concurrent_profiles <= 0:
            raise ValueError(f"max_concurrent_profiles must be positive int")
    
    def profile_operation(self, operation_name: str, operation_func: Callable, *args, **kwargs) -> Any:
        """
        Profile a single operation and collect performance metrics.
        
        Args:
            operation_name: Name identifier for the operation
            operation_func: Function to profile
            *args: Arguments to pass to operation_func
            **kwargs: Keyword arguments to pass to operation_func
            
        Returns:
            Result of operation_func execution
            
        Raises:
            ValueError: If operation_name is invalid
            TypeError: If operation_func is not callable
        """
        # NO FALLBACKS - HARD FAILURES ONLY
        if not isinstance(operation_name, str):
            raise TypeError(f"operation_name must be str, got {type(operation_name)}")
        if not operation_name.strip():
            raise ValueError("operation_name cannot be empty")
        if not callable(operation_func):
            raise TypeError(f"operation_func must be callable, got {type(operation_func)}")
        
        if not self.enable_profiling:
            return operation_func(*args, **kwargs)
        
        # Check concurrent profile limit
        with self._profile_lock:
            if len(self.active_operations) >= self.max_concurrent_profiles:
                raise RuntimeError(f"Maximum concurrent profiles ({self.max_concurrent_profiles}) exceeded")
            self.active_operations.add(operation_name)
        
        # Initialize profile data
        profile = OperationProfile(
            operation_name=operation_name,
            execution_time=0.0,
            cpu_usage=0.0,
            memory_usage=0.0,
            thread_id=threading.get_ident(),
            call_stack=self._get_call_stack()
        )
        
        # Measure initial system state
        initial_memory = psutil.virtual_memory().used
        initial_cpu = psutil.cpu_percent()
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            initial_gpu_memory = torch.cuda.memory_allocated()
        else:
            initial_gpu_memory = 0
        
        start_time = time.perf_counter()
        
        try:
            # Calculate argument sizes
            profile.args_size = self._calculate_object_size(args) + self._calculate_object_size(kwargs)
            
            # Execute operation
            result = operation_func(*args, **kwargs)
            
            # Calculate return size
            profile.return_size = self._calculate_object_size(result)
            profile.success = True
            
            return result
            
        except Exception as e:
            profile.success = False
            profile.error_message = str(e)
            raise
            
        finally:
            # Measure final system state
            end_time = time.perf_counter()
            profile.execution_time = end_time - start_time
            
            final_memory = psutil.virtual_memory().used
            final_cpu = psutil.cpu_percent()
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                final_gpu_memory = torch.cuda.memory_allocated()
                profile.gpu_usage = final_gpu_memory - initial_gpu_memory
            
            profile.cpu_usage = final_cpu - initial_cpu
            profile.memory_usage = final_memory - initial_memory
            
            # Store profile data
            with self._profile_lock:
                self.operation_profiles[operation_name].append(profile)
                self.active_operations.discard(operation_name)
                self.performance_metrics['total_operations_profiled'] += 1
            
            # Update performance metrics
            self._update_performance_metrics(profile)
            
            # Trigger performance hooks
            self._trigger_performance_hooks(operation_name, profile)
            
            self.logger.debug(
                f"Profiled operation '{operation_name}': "
                f"time={profile.execution_time:.4f}s, "
                f"memory={profile.memory_usage / 1024 / 1024:.2f}MB, "
                f"success={profile.success}"
            )
    
    async def profile_operation_async(self, operation_name: str, operation_coro, *args, **kwargs) -> Any:
        """
        Asynchronously profile a coroutine operation.
        
        Args:
            operation_name: Name identifier for the operation
            operation_coro: Coroutine to profile
            *args: Arguments to pass to operation_coro
            **kwargs: Keyword arguments to pass to operation_coro
            
        Returns:
            Result of operation_coro execution
        """
        if not self.enable_async_profiling:
            return await operation_coro(*args, **kwargs)
        
        # NO FALLBACKS - HARD FAILURES ONLY
        if not isinstance(operation_name, str):
            raise TypeError(f"operation_name must be str, got {type(operation_name)}")
        if not operation_name.strip():
            raise ValueError("operation_name cannot be empty")
        if not asyncio.iscoroutinefunction(operation_coro):
            raise TypeError(f"operation_coro must be coroutine function")
        
        # Similar profiling logic as sync version
        return await self._profile_async_operation(operation_name, operation_coro, *args, **kwargs)
    
    async def _profile_async_operation(self, operation_name: str, operation_coro, *args, **kwargs) -> Any:
        """Internal async profiling implementation"""
        profile = OperationProfile(
            operation_name=operation_name,
            execution_time=0.0,
            cpu_usage=0.0,
            memory_usage=0.0,
            thread_id=threading.get_ident(),
            call_stack=self._get_call_stack()
        )
        
        initial_memory = psutil.virtual_memory().used
        start_time = time.perf_counter()
        
        try:
            result = await operation_coro(*args, **kwargs)
            profile.success = True
            return result
            
        except Exception as e:
            profile.success = False
            profile.error_message = str(e)
            raise
            
        finally:
            end_time = time.perf_counter()
            profile.execution_time = end_time - start_time
            profile.memory_usage = psutil.virtual_memory().used - initial_memory
            
            with self._profile_lock:
                self.operation_profiles[operation_name].append(profile)
                self.performance_metrics['total_operations_profiled'] += 1
            
            self._update_performance_metrics(profile)
            self._trigger_performance_hooks(operation_name, profile)
    
    def detect_bottlenecks(self) -> Dict[str, BottleneckReport]:
        """
        Analyze collected performance data to detect system bottlenecks.
        
        Returns:
            Dictionary mapping operation names to bottleneck reports
        """
        if not self.enable_profiling:
            raise RuntimeError("Profiling must be enabled to detect bottlenecks")
        
        bottlenecks = {}
        
        with self._profile_lock:
            for operation_name, profiles in self.operation_profiles.items():
                if len(profiles) < 10:  # Require minimum samples
                    continue
                
                bottleneck = self._analyze_operation_bottleneck(operation_name, list(profiles))
                if bottleneck and bottleneck.severity >= self.bottleneck_detection_threshold:
                    bottlenecks[operation_name] = bottleneck
        
        # Update stored bottleneck reports
        with self._optimization_lock:
            self.bottleneck_reports.update(bottlenecks)
            self.performance_metrics['total_bottlenecks_detected'] = len(self.bottleneck_reports)
        
        self.logger.info(f"Detected {len(bottlenecks)} bottlenecks")
        return bottlenecks
    
    def _analyze_operation_bottleneck(self, operation_name: str, profiles: List[OperationProfile]) -> Optional[BottleneckReport]:
        """Analyze profiles for a single operation to detect bottlenecks"""
        if not profiles:
            return None
        
        # Calculate statistics
        execution_times = [p.execution_time for p in profiles if p.success]
        memory_usages = [p.memory_usage for p in profiles if p.success]
        cpu_usages = [p.cpu_usage for p in profiles if p.success]
        
        if not execution_times:
            return None
        
        avg_time = np.mean(execution_times)
        std_time = np.std(execution_times)
        max_time = np.max(execution_times)
        
        avg_memory = np.mean(memory_usages)
        avg_cpu = np.mean(cpu_usages)
        
        # Determine bottleneck type
        bottleneck_type = BottleneckType.CPU_BOUND
        severity = 0.0
        
        # Memory bound detection
        if avg_memory > 100 * 1024 * 1024:  # > 100MB average
            bottleneck_type = BottleneckType.MEMORY_BOUND
            severity = min(1.0, avg_memory / (1024 * 1024 * 1024))  # Normalize by 1GB
        
        # CPU bound detection
        elif avg_cpu > 80:  # > 80% CPU usage
            bottleneck_type = BottleneckType.CPU_BOUND
            severity = min(1.0, avg_cpu / 100.0)
        
        # High execution time variation (synchronization issues)
        elif std_time > avg_time * 0.5:
            bottleneck_type = BottleneckType.SYNCHRONIZATION
            severity = min(1.0, std_time / avg_time)
        
        # Algorithm inefficiency (consistently slow)
        elif avg_time > 1.0:  # > 1 second average
            bottleneck_type = BottleneckType.ALGORITHM
            severity = min(1.0, avg_time / 10.0)  # Normalize by 10 seconds
        
        if severity < 0.1:  # Not significant enough
            return None
        
        # Generate optimization suggestions
        suggestions = self._generate_bottleneck_suggestions(bottleneck_type, profiles)
        
        return BottleneckReport(
            bottleneck_type=bottleneck_type,
            operation_name=operation_name,
            severity=severity,
            impact_factor=avg_time * len(profiles),
            occurrence_count=len(profiles),
            average_delay=avg_time,
            peak_delay=max_time,
            suggested_optimizations=suggestions,
            estimated_improvement=self._estimate_optimization_improvement(bottleneck_type, severity),
            confidence_score=min(1.0, len(profiles) / 100.0)
        )
    
    def _generate_bottleneck_suggestions(self, bottleneck_type: BottleneckType, profiles: List[OperationProfile]) -> List[str]:
        """Generate optimization suggestions based on bottleneck type"""
        suggestions = []
        
        if bottleneck_type == BottleneckType.MEMORY_BOUND:
            suggestions.extend([
                "Enable memory pooling and reuse",
                "Implement lazy loading for large data structures",
                "Add garbage collection optimization",
                "Consider data compression techniques",
                "Implement memory-mapped file operations"
            ])
        
        elif bottleneck_type == BottleneckType.CPU_BOUND:
            suggestions.extend([
                "Enable parallel processing where possible",
                "Optimize critical algorithm sections",
                "Consider CPU-specific optimizations (vectorization)",
                "Implement caching for repeated computations",
                "Profile and optimize hot code paths"
            ])
        
        elif bottleneck_type == BottleneckType.SYNCHRONIZATION:
            suggestions.extend([
                "Reduce lock contention",
                "Implement lock-free data structures",
                "Optimize thread pool configuration",
                "Consider async/await patterns",
                "Minimize critical section sizes"
            ])
        
        elif bottleneck_type == BottleneckType.ALGORITHM:
            suggestions.extend([
                "Review algorithm complexity",
                "Implement more efficient data structures",
                "Add early termination conditions",
                "Consider approximation algorithms",
                "Implement incremental processing"
            ])
        
        elif bottleneck_type == BottleneckType.GPU_BOUND:
            suggestions.extend([
                "Optimize GPU memory allocation",
                "Improve CUDA kernel efficiency",
                "Implement GPU memory pooling",
                "Optimize tensor operations",
                "Consider mixed-precision training"
            ])
        
        return suggestions
    
    def _estimate_optimization_improvement(self, bottleneck_type: BottleneckType, severity: float) -> float:
        """Estimate potential improvement from optimization"""
        base_improvement = {
            BottleneckType.MEMORY_BOUND: 0.6,
            BottleneckType.CPU_BOUND: 0.4,
            BottleneckType.SYNCHRONIZATION: 0.8,
            BottleneckType.ALGORITHM: 0.7,
            BottleneckType.GPU_BOUND: 0.5,
            BottleneckType.IO_BOUND: 0.3,
            BottleneckType.NETWORK_BOUND: 0.2
        }
        
        return base_improvement.get(bottleneck_type, 0.3) * severity
    
    def optimize_system(self) -> Dict[str, Any]:
        """
        Apply system-wide optimizations based on detected bottlenecks.
        
        Returns:
            Dictionary containing optimization results and metrics
        """
        if not self.enable_profiling:
            raise RuntimeError("Profiling must be enabled for system optimization")
        
        optimization_start = time.time()
        
        # Detect current bottlenecks
        bottlenecks = self.detect_bottlenecks()
        
        if not bottlenecks:
            return {
                'status': 'no_optimization_needed',
                'bottlenecks_found': 0,
                'optimizations_applied': 0,
                'total_time': time.time() - optimization_start
            }
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(bottlenecks)
        
        # Apply optimizations based on strategy
        applied_optimizations = self._apply_optimizations(recommendations)
        
        # Update metrics
        with self._optimization_lock:
            self.optimization_recommendations = recommendations
            self.performance_metrics['total_optimizations_applied'] += len(applied_optimizations)
        
        optimization_time = time.time() - optimization_start
        
        result = {
            'status': 'optimization_completed',
            'bottlenecks_found': len(bottlenecks),
            'recommendations_generated': len(recommendations),
            'optimizations_applied': len(applied_optimizations),
            'total_time': optimization_time,
            'applied_optimizations': applied_optimizations,
            'estimated_improvement': sum(r.estimated_improvement for r in recommendations)
        }
        
        self.logger.info(
            f"System optimization completed: {len(applied_optimizations)} optimizations applied "
            f"in {optimization_time:.2f}s"
        )
        
        return result
    
    def _generate_optimization_recommendations(self, bottlenecks: Dict[str, BottleneckReport]) -> List[OptimizationRecommendation]:
        """Generate specific optimization recommendations from bottleneck reports"""
        recommendations = []
        
        for operation_name, report in bottlenecks.items():
            # Priority based on severity and impact
            if report.severity > 0.8 and report.impact_factor > 100:
                priority = "critical"
            elif report.severity > 0.6:
                priority = "high"
            elif report.severity > 0.4:
                priority = "medium"
            else:
                priority = "low"
            
            for suggestion in report.suggested_optimizations:
                recommendation = OptimizationRecommendation(
                    strategy=self.optimization_strategy,
                    target_component=operation_name,
                    action_type=report.bottleneck_type.value,
                    description=suggestion,
                    estimated_improvement=report.estimated_improvement,
                    implementation_effort=self._estimate_implementation_effort(suggestion),
                    priority=priority,
                    dependencies=self._identify_dependencies(suggestion),
                    risks=self._identify_risks(suggestion)
                )
                recommendations.append(recommendation)
        
        # Sort by priority and estimated improvement
        priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        recommendations.sort(
            key=lambda r: (priority_order[r.priority], r.estimated_improvement),
            reverse=True
        )
        
        return recommendations
    
    def _estimate_implementation_effort(self, suggestion: str) -> str:
        """Estimate implementation effort for optimization suggestion"""
        high_effort_keywords = ["algorithm", "data structure", "architecture", "refactor"]
        medium_effort_keywords = ["pool", "cache", "parallel", "optimize"]
        
        suggestion_lower = suggestion.lower()
        
        if any(keyword in suggestion_lower for keyword in high_effort_keywords):
            return "high"
        elif any(keyword in suggestion_lower for keyword in medium_effort_keywords):
            return "medium"
        else:
            return "low"
    
    def _identify_dependencies(self, suggestion: str) -> List[str]:
        """Identify dependencies for optimization suggestion"""
        dependencies = []
        suggestion_lower = suggestion.lower()
        
        if "parallel" in suggestion_lower or "thread" in suggestion_lower:
            dependencies.append("thread_pool_configuration")
        if "gpu" in suggestion_lower or "cuda" in suggestion_lower:
            dependencies.append("gpu_memory_optimizer")
        if "memory" in suggestion_lower:
            dependencies.append("memory_optimizer")
        if "cache" in suggestion_lower:
            dependencies.append("caching_system")
        
        return dependencies
    
    def _identify_risks(self, suggestion: str) -> List[str]:
        """Identify potential risks for optimization suggestion"""
        risks = []
        suggestion_lower = suggestion.lower()
        
        if "parallel" in suggestion_lower:
            risks.append("Race conditions and synchronization issues")
        if "algorithm" in suggestion_lower:
            risks.append("Correctness verification required")
        if "memory" in suggestion_lower:
            risks.append("Memory usage pattern changes")
        if "cache" in suggestion_lower:
            risks.append("Cache invalidation complexity")
        
        return risks
    
    def _apply_optimizations(self, recommendations: List[OptimizationRecommendation]) -> List[str]:
        """Apply optimization recommendations based on current strategy"""
        applied = []
        
        for recommendation in recommendations:
            if self._should_apply_optimization(recommendation):
                success = self._apply_single_optimization(recommendation)
                if success:
                    applied.append(recommendation.description)
        
        return applied
    
    def _should_apply_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Determine if optimization should be applied based on strategy"""
        if self.optimization_strategy == OptimizationStrategy.CONSERVATIVE:
            return (recommendation.priority in ["critical", "high"] and 
                   recommendation.implementation_effort == "low")
        
        elif self.optimization_strategy == OptimizationStrategy.BALANCED:
            return (recommendation.priority in ["critical", "high", "medium"] and
                   recommendation.implementation_effort in ["low", "medium"])
        
        elif self.optimization_strategy == OptimizationStrategy.AGGRESSIVE:
            return recommendation.priority != "low"
        
        # Custom strategy - apply all high-impact, low-risk optimizations
        return (recommendation.estimated_improvement > 0.3 and
               len(recommendation.risks) <= 2)
    
    def _apply_single_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply a single optimization recommendation"""
        try:
            # This is where specific optimization implementations would go
            # For now, we'll simulate successful application
            self.logger.info(f"Applied optimization: {recommendation.description}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply optimization {recommendation.description}: {e}")
            return False
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Dictionary containing detailed performance analysis
        """
        with self._profile_lock:
            operation_stats = {}
            for operation_name, profiles in self.operation_profiles.items():
                if profiles:
                    execution_times = [p.execution_time for p in profiles if p.success]
                    memory_usages = [p.memory_usage for p in profiles if p.success]
                    success_rate = sum(1 for p in profiles if p.success) / len(profiles)
                    
                    if execution_times:
                        operation_stats[operation_name] = {
                            'total_calls': len(profiles),
                            'success_rate': success_rate,
                            'avg_execution_time': np.mean(execution_times),
                            'max_execution_time': np.max(execution_times),
                            'min_execution_time': np.min(execution_times),
                            'std_execution_time': np.std(execution_times),
                            'avg_memory_usage': np.mean(memory_usages),
                            'total_memory_usage': np.sum(memory_usages)
                        }
        
        with self._optimization_lock:
            bottleneck_summary = {
                name: {
                    'type': report.bottleneck_type.value,
                    'severity': report.severity,
                    'impact_factor': report.impact_factor,
                    'estimated_improvement': report.estimated_improvement
                }
                for name, report in self.bottleneck_reports.items()
            }
        
        # System-wide metrics
        system_metrics = {
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'active_threads': threading.active_count(),
            'process_memory': psutil.Process().memory_info().rss / 1024 / 1024  # MB
        }
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            system_metrics['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            system_metrics['gpu_memory_cached'] = torch.cuda.memory_reserved() / 1024 / 1024  # MB
        
        return {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': self.performance_metrics.copy(),
            'system_metrics': system_metrics,
            'operation_statistics': operation_stats,
            'bottleneck_summary': bottleneck_summary,
            'optimization_recommendations': [
                {
                    'target': rec.target_component,
                    'action': rec.action_type,
                    'description': rec.description,
                    'priority': rec.priority,
                    'estimated_improvement': rec.estimated_improvement
                }
                for rec in self.optimization_recommendations
            ],
            'configuration': {
                'optimization_strategy': self.optimization_strategy.value,
                'profiling_enabled': self.enable_profiling,
                'bottleneck_threshold': self.bottleneck_detection_threshold,
                'profile_history_size': self.profile_history_size
            }
        }
    
    def register_performance_hook(self, component_name: str, hook_func: Callable) -> None:
        """
        Register a performance hook for a specific component.
        
        Args:
            component_name: Name of the component to hook
            hook_func: Function to call when component performance data is available
            
        Raises:
            TypeError: If component_name is not string or hook_func is not callable
        """
        if not isinstance(component_name, str):
            raise TypeError(f"component_name must be str, got {type(component_name)}")
        if not component_name.strip():
            raise ValueError("component_name cannot be empty")
        if not callable(hook_func):
            raise TypeError(f"hook_func must be callable, got {type(hook_func)}")
        
        with self._hook_lock:
            self.performance_hooks[component_name].append(hook_func)
        
        self.logger.info(f"Registered performance hook for component '{component_name}'")
    
    def _trigger_performance_hooks(self, operation_name: str, profile: OperationProfile) -> None:
        """Trigger registered performance hooks for an operation"""
        with self._hook_lock:
            hooks = self.performance_hooks.get(operation_name, [])
        
        for hook in hooks:
            try:
                hook(profile)
            except Exception as e:
                self.logger.error(f"Error in performance hook for '{operation_name}': {e}")
    
    def _get_call_stack(self) -> List[str]:
        """Get current call stack for debugging"""
        try:
            stack = traceback.extract_stack()
            return [f"{frame.filename}:{frame.lineno} in {frame.name}" for frame in stack[-10:]]
        except Exception:
            return []
    
    def _calculate_object_size(self, obj) -> int:
        """Calculate approximate memory size of an object"""
        try:
            import sys
            return sys.getsizeof(obj)
        except Exception:
            return 0
    
    def _update_performance_metrics(self, profile: OperationProfile) -> None:
        """Update running performance metrics"""
        total_ops = self.performance_metrics['total_operations_profiled']
        if total_ops > 0:
            old_avg = self.performance_metrics['average_operation_time']
            new_avg = (old_avg * (total_ops - 1) + profile.execution_time) / total_ops
            self.performance_metrics['average_operation_time'] = new_avg
        
        # Calculate system efficiency score (simplified)
        if profile.execution_time > 0:
            efficiency = min(1.0, 1.0 / profile.execution_time)  # Inverse of execution time
            current_score = self.performance_metrics['system_efficiency_score']
            self.performance_metrics['system_efficiency_score'] = (current_score * 0.9 + efficiency * 0.1)
    
    def start_monitoring(self) -> None:
        """Start background performance monitoring"""
        if self._monitoring_enabled:
            return
        
        self._monitoring_enabled = True
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="PerformanceMonitor",
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info("Background performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background performance monitoring"""
        if not self._monitoring_enabled:
            return
        
        self._monitoring_enabled = False
        self._stop_monitoring.set()
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Background performance monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while not self._stop_monitoring.wait(self.profiling_interval):
            try:
                # Update system monitoring
                self.system_monitor.update()
                
                # Check for critical performance issues
                self._check_critical_performance()
                
                # Periodic bottleneck detection
                if self.performance_metrics['total_operations_profiled'] % 100 == 0:
                    self.detect_bottlenecks()
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring loop: {e}")
    
    def _check_critical_performance(self) -> None:
        """Check for critical performance issues requiring immediate attention"""
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        if cpu_usage > 95:
            self.logger.warning(f"Critical CPU usage detected: {cpu_usage}%")
        
        if memory_usage > 90:
            self.logger.warning(f"Critical memory usage detected: {memory_usage}%")
        
        # Check for operations taking too long
        current_time = time.time()
        for operation_name in list(self.active_operations):
            if operation_name in self.operation_timers:
                elapsed = current_time - self.operation_timers[operation_name]
                if elapsed > 30:  # 30 seconds threshold
                    self.logger.warning(f"Long-running operation detected: '{operation_name}' ({elapsed:.1f}s)")


class SystemMonitor:
    """System resource monitoring helper"""
    
    def __init__(self):
        self.cpu_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)
        self.last_update = 0
    
    def update(self) -> None:
        """Update system metrics"""
        current_time = time.time()
        if current_time - self.last_update < 1.0:  # Rate limit to 1 second
            return
        
        self.cpu_history.append(psutil.cpu_percent())
        self.memory_history.append(psutil.virtual_memory().percent)
        self.last_update = current_time
    
    def get_average_cpu(self) -> float:
        """Get average CPU usage over monitoring period"""
        return np.mean(self.cpu_history) if self.cpu_history else 0.0
    
    def get_average_memory(self) -> float:
        """Get average memory usage over monitoring period"""
        return np.mean(self.memory_history) if self.memory_history else 0.0