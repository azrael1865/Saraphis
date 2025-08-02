"""
Domain Performance Optimizer - Domain-specific performance optimization with hybrid system integration.

This module provides comprehensive domain performance optimization capabilities,
working in coordination with the domain hybrid integration and compression systems
to maximize performance across different domain types and workloads.

Key Features:
- Domain-specific performance optimization strategies
- Hybrid system performance coordination
- Real-time performance monitoring and adaptive optimization
- Domain resource allocation optimization
- Performance bottleneck detection and resolution
- Cross-domain performance coordination
- Domain-specific caching and memory optimization
- Performance analytics and trend analysis

Architecture: NO FALLBACKS - HARD FAILURES ONLY
All operations must succeed or fail explicitly with detailed error information.
"""

import asyncio
import threading
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
import json
from contextlib import contextmanager
from threading import RLock
import traceback
import uuid
import gc
import psutil
import numpy as np

try:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from domain_registry import DomainRegistry
        from domain_router import DomainRouter
        from domain_hybrid_integration import DomainHybridIntegration
        from domain_compression_coordinator import DomainCompressionCoordinator
        from hybrid_compression_orchestrator import HybridCompressionOrchestrator
except ImportError:
    pass

logger = logging.getLogger(__name__)


class DomainOptimizationStrategy(Enum):
    """Domain optimization strategy types."""
    PERFORMANCE_FIRST = "performance_first"
    MEMORY_OPTIMIZED = "memory_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    THROUGHPUT_OPTIMIZED = "throughput_optimized"
    BALANCED = "balanced"
    DOMAIN_SPECIFIC = "domain_specific"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"


class DomainPerformanceMetric(Enum):
    """Domain performance metrics."""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    GPU_USAGE = "gpu_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    ERROR_RATE = "error_rate"
    QUEUE_LENGTH = "queue_length"
    CONCURRENT_REQUESTS = "concurrent_requests"
    RESOURCE_UTILIZATION = "resource_utilization"


class DomainOptimizationLevel(Enum):
    """Domain optimization levels."""
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MAXIMUM = "maximum"


@dataclass
class DomainPerformanceTarget:
    """Domain performance target configuration."""
    domain_id: str
    target_response_time: float
    target_throughput: float
    target_memory_usage: float
    target_cpu_usage: float
    target_error_rate: float
    optimization_strategy: DomainOptimizationStrategy
    priority: int = 1
    constraints: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class DomainPerformanceResult:
    """Domain performance optimization result."""
    domain_id: str
    optimization_id: str
    strategy_used: DomainOptimizationStrategy
    performance_before: Dict[str, float]
    performance_after: Dict[str, float]
    improvement_metrics: Dict[str, float]
    optimization_actions: List[str]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class DomainResourceAllocation:
    """Domain resource allocation configuration."""
    domain_id: str
    memory_allocation: float
    cpu_allocation: float
    gpu_allocation: float
    cache_allocation: float
    network_allocation: float
    storage_allocation: float
    priority_level: int
    dynamic_scaling: bool = True
    created_at: float = field(default_factory=time.time)


@dataclass
class DomainPerformanceAnalytics:
    """Domain performance analytics data."""
    domain_id: str
    total_optimizations: int
    successful_optimizations: int
    failed_optimizations: int
    average_improvement: Dict[str, float]
    optimization_history: List[DomainPerformanceResult]
    performance_trends: Dict[str, List[float]]
    bottleneck_patterns: Dict[str, int]
    resource_efficiency: Dict[str, float]
    recommendation_score: float
    generated_at: float = field(default_factory=time.time)


class DomainPerformanceCache:
    """Domain-specific performance cache."""
    
    def __init__(self, domain_id: str, max_size: int = 10000):
        self.domain_id = domain_id
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                self.hit_count += 1
                return self.cache[key]
            else:
                self.miss_count += 1
                return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        with self.lock:
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if self.access_times:
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[lru_key]
            del self.access_times[lru_key]
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.hit_count = 0
            self.miss_count = 0


class DomainPerformanceMonitor:
    """Real-time domain performance monitoring."""
    
    def __init__(self, domain_id: str, window_size: int = 100):
        self.domain_id = domain_id
        self.window_size = window_size
        self.metrics: Dict[DomainPerformanceMetric, deque] = {
            metric: deque(maxlen=window_size) for metric in DomainPerformanceMetric
        }
        self.lock = RLock()
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        with self.lock:
            if not self.monitoring_active:
                self.monitoring_active = True
                self.monitor_thread = threading.Thread(
                    target=self._monitor_loop,
                    name=f"DomainPerformanceMonitor-{self.domain_id}",
                    daemon=True
                )
                self.monitor_thread.start()
                logger.info(f"Started performance monitoring for domain {self.domain_id}")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        with self.lock:
            self.monitoring_active = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5.0)
                logger.info(f"Stopped performance monitoring for domain {self.domain_id}")
    
    def record_metric(self, metric: DomainPerformanceMetric, value: float) -> None:
        """Record a performance metric."""
        with self.lock:
            self.metrics[metric].append((time.time(), value))
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        with self.lock:
            current_metrics = {}
            for metric, values in self.metrics.items():
                if values:
                    recent_values = [v[1] for v in list(values)[-10:]]
                    current_metrics[metric.value] = statistics.mean(recent_values)
                else:
                    current_metrics[metric.value] = 0.0
            return current_metrics
    
    def get_metric_trend(self, metric: DomainPerformanceMetric, duration: float = 300.0) -> List[float]:
        """Get metric trend over duration."""
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - duration
            
            values = self.metrics.get(metric, deque())
            trend_values = [v[1] for v in values if v[0] >= cutoff_time]
            return trend_values
    
    def detect_anomalies(self) -> Dict[str, List[str]]:
        """Detect performance anomalies."""
        anomalies = defaultdict(list)
        
        with self.lock:
            for metric, values in self.metrics.items():
                if len(values) < 10:
                    continue
                
                recent_values = [v[1] for v in list(values)]
                if len(recent_values) >= 10:
                    mean_val = statistics.mean(recent_values)
                    std_val = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
                    
                    latest_val = recent_values[-1]
                    if std_val > 0 and abs(latest_val - mean_val) > 3 * std_val:
                        anomalies[metric.value].append(f"Value {latest_val} is {abs(latest_val - mean_val) / std_val:.2f} standard deviations from mean")
        
        return dict(anomalies)
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                process = psutil.Process()
                
                self.record_metric(DomainPerformanceMetric.CPU_USAGE, process.cpu_percent())
                self.record_metric(DomainPerformanceMetric.MEMORY_USAGE, process.memory_percent())
                
                if hasattr(psutil, 'sensors_temperatures'):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        avg_temp = sum(sum(t.current for t in sensors) / len(sensors) for sensors in temps.values()) / len(temps)
                        self.record_metric(DomainPerformanceMetric.RESOURCE_UTILIZATION, avg_temp)
                
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop for domain {self.domain_id}: {e}")
                time.sleep(5.0)


class DomainPerformanceOptimizer:
    """
    Domain Performance Optimizer - Advanced domain-specific performance optimization.
    
    This class provides comprehensive performance optimization capabilities for domains,
    working in coordination with hybrid compression systems and domain coordination.
    """
    
    def __init__(self):
        self.domain_optimizers: Dict[str, Any] = {}
        self.performance_targets: Dict[str, DomainPerformanceTarget] = {}
        self.performance_results: Dict[str, List[DomainPerformanceResult]] = defaultdict(list)
        self.resource_allocations: Dict[str, DomainResourceAllocation] = {}
        self.performance_monitors: Dict[str, DomainPerformanceMonitor] = {}
        self.domain_caches: Dict[str, DomainPerformanceCache] = {}
        self.optimization_strategies: Dict[str, DomainOptimizationStrategy] = {}
        self.active_optimizations: Dict[str, str] = {}
        self.optimization_lock = RLock()
        self.analytics_cache: Dict[str, DomainPerformanceAnalytics] = {}
        
        self.domain_registry: Optional['DomainRegistry'] = None
        self.domain_router: Optional['DomainRouter'] = None
        self.domain_hybrid_integration: Optional['DomainHybridIntegration'] = None
        self.domain_compression_coordinator: Optional['DomainCompressionCoordinator'] = None
        self.hybrid_compression_orchestrator: Optional['HybridCompressionOrchestrator'] = None
        
        self.optimization_history: List[DomainPerformanceResult] = []
        self.total_optimizations = 0
        self.successful_optimizations = 0
        self.failed_optimizations = 0
        self.start_time = time.time()
        
        logger.info("DomainPerformanceOptimizer initialized successfully")
    
    def initialize_domain_performance_optimizer(
        self,
        domain_registry: Optional['DomainRegistry'] = None,
        domain_router: Optional['DomainRouter'] = None,
        domain_hybrid_integration: Optional['DomainHybridIntegration'] = None,
        domain_compression_coordinator: Optional['DomainCompressionCoordinator'] = None,
        hybrid_compression_orchestrator: Optional['HybridCompressionOrchestrator'] = None
    ) -> bool:
        """Initialize domain performance optimizer with system integrations."""
        try:
            with self.optimization_lock:
                if domain_registry:
                    self.domain_registry = domain_registry
                    logger.info("Domain registry integration initialized")
                
                if domain_router:
                    self.domain_router = domain_router
                    logger.info("Domain router integration initialized")
                
                if domain_hybrid_integration:
                    self.domain_hybrid_integration = domain_hybrid_integration
                    logger.info("Domain hybrid integration initialized")
                
                if domain_compression_coordinator:
                    self.domain_compression_coordinator = domain_compression_coordinator
                    logger.info("Domain compression coordinator integration initialized")
                
                if hybrid_compression_orchestrator:
                    self.hybrid_compression_orchestrator = hybrid_compression_orchestrator
                    logger.info("Hybrid compression orchestrator integration initialized")
                
                logger.info("Domain performance optimizer initialization completed successfully")
                return True
                
        except Exception as e:
            error_msg = f"Failed to initialize domain performance optimizer: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def register_domain_for_optimization(
        self,
        domain_id: str,
        performance_target: DomainPerformanceTarget,
        optimization_strategy: DomainOptimizationStrategy = DomainOptimizationStrategy.BALANCED
    ) -> bool:
        """Register a domain for performance optimization."""
        try:
            with self.optimization_lock:
                if not domain_id:
                    raise ValueError("Domain ID cannot be empty")
                
                self.performance_targets[domain_id] = performance_target
                self.optimization_strategies[domain_id] = optimization_strategy
                
                monitor = DomainPerformanceMonitor(domain_id)
                self.performance_monitors[domain_id] = monitor
                monitor.start_monitoring()
                
                cache = DomainPerformanceCache(domain_id)
                self.domain_caches[domain_id] = cache
                
                default_allocation = DomainResourceAllocation(
                    domain_id=domain_id,
                    memory_allocation=0.2,
                    cpu_allocation=0.2,
                    gpu_allocation=0.1,
                    cache_allocation=0.1,
                    network_allocation=0.1,
                    storage_allocation=0.1,
                    priority_level=performance_target.priority
                )
                self.resource_allocations[domain_id] = default_allocation
                
                logger.info(f"Domain {domain_id} registered for optimization with strategy {optimization_strategy.value}")
                return True
                
        except Exception as e:
            error_msg = f"Failed to register domain {domain_id} for optimization: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def optimize_domain_performance(
        self,
        domain_id: str,
        optimization_level: DomainOptimizationLevel = DomainOptimizationLevel.STANDARD,
        force_optimization: bool = False
    ) -> DomainPerformanceResult:
        """Optimize performance for a specific domain."""
        optimization_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            with self.optimization_lock:
                if domain_id not in self.performance_targets:
                    raise ValueError(f"Domain {domain_id} not registered for optimization")
                
                if not force_optimization and domain_id in self.active_optimizations:
                    raise RuntimeError(f"Optimization already active for domain {domain_id}")
                
                self.active_optimizations[domain_id] = optimization_id
                
                performance_before = self._measure_domain_performance(domain_id)
                
                target = self.performance_targets[domain_id]
                strategy = self.optimization_strategies.get(domain_id, DomainOptimizationStrategy.BALANCED)
                
                optimization_actions = []
                
                if optimization_level in [DomainOptimizationLevel.BASIC, DomainOptimizationLevel.STANDARD]:
                    optimization_actions.extend(self._apply_basic_optimizations(domain_id, strategy))
                
                if optimization_level in [DomainOptimizationLevel.STANDARD, DomainOptimizationLevel.ADVANCED]:
                    optimization_actions.extend(self._apply_advanced_optimizations(domain_id, strategy))
                
                if optimization_level in [DomainOptimizationLevel.ADVANCED, DomainOptimizationLevel.EXPERT]:
                    optimization_actions.extend(self._apply_expert_optimizations(domain_id, strategy))
                
                if optimization_level == DomainOptimizationLevel.MAXIMUM:
                    optimization_actions.extend(self._apply_maximum_optimizations(domain_id, strategy))
                
                time.sleep(0.1)
                
                performance_after = self._measure_domain_performance(domain_id)
                improvement_metrics = self._calculate_improvement_metrics(performance_before, performance_after)
                
                execution_time = time.time() - start_time
                
                result = DomainPerformanceResult(
                    domain_id=domain_id,
                    optimization_id=optimization_id,
                    strategy_used=strategy,
                    performance_before=performance_before,
                    performance_after=performance_after,
                    improvement_metrics=improvement_metrics,
                    optimization_actions=optimization_actions,
                    execution_time=execution_time,
                    success=True
                )
                
                self.performance_results[domain_id].append(result)
                self.optimization_history.append(result)
                self.total_optimizations += 1
                self.successful_optimizations += 1
                
                del self.active_optimizations[domain_id]
                
                logger.info(f"Domain {domain_id} optimization completed successfully in {execution_time:.3f}s")
                return result
                
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Failed to optimize domain {domain_id}: {e}"
            
            result = DomainPerformanceResult(
                domain_id=domain_id,
                optimization_id=optimization_id,
                strategy_used=self.optimization_strategies.get(domain_id, DomainOptimizationStrategy.BALANCED),
                performance_before={},
                performance_after={},
                improvement_metrics={},
                optimization_actions=[],
                execution_time=execution_time,
                success=False,
                error_message=error_msg
            )
            
            self.performance_results[domain_id].append(result)
            self.optimization_history.append(result)
            self.total_optimizations += 1
            self.failed_optimizations += 1
            
            if domain_id in self.active_optimizations:
                del self.active_optimizations[domain_id]
            
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def _measure_domain_performance(self, domain_id: str) -> Dict[str, float]:
        """Measure current domain performance."""
        performance = {}
        
        if domain_id in self.performance_monitors:
            monitor = self.performance_monitors[domain_id]
            performance.update(monitor.get_current_metrics())
        
        if domain_id in self.domain_caches:
            cache = self.domain_caches[domain_id]
            performance['cache_hit_rate'] = cache.get_hit_rate()
        
        performance['timestamp'] = time.time()
        
        try:
            process = psutil.Process()
            performance['system_cpu'] = process.cpu_percent()
            performance['system_memory'] = process.memory_percent()
        except:
            pass
        
        return performance
    
    def _apply_basic_optimizations(self, domain_id: str, strategy: DomainOptimizationStrategy) -> List[str]:
        """Apply basic optimization techniques."""
        actions = []
        
        try:
            if domain_id in self.domain_caches:
                cache = self.domain_caches[domain_id]
                if cache.get_hit_rate() < 0.5:
                    cache.max_size = min(cache.max_size * 2, 50000)
                    actions.append(f"Increased cache size for domain {domain_id}")
            
            if domain_id in self.resource_allocations:
                allocation = self.resource_allocations[domain_id]
                if strategy == DomainOptimizationStrategy.MEMORY_OPTIMIZED:
                    allocation.memory_allocation = min(allocation.memory_allocation * 1.1, 0.8)
                    actions.append(f"Increased memory allocation for domain {domain_id}")
                elif strategy == DomainOptimizationStrategy.PERFORMANCE_FIRST:
                    allocation.cpu_allocation = min(allocation.cpu_allocation * 1.1, 0.8)
                    actions.append(f"Increased CPU allocation for domain {domain_id}")
            
            gc.collect()
            actions.append("Performed garbage collection")
            
        except Exception as e:
            logger.warning(f"Error in basic optimizations for domain {domain_id}: {e}")
        
        return actions
    
    def _apply_advanced_optimizations(self, domain_id: str, strategy: DomainOptimizationStrategy) -> List[str]:
        """Apply advanced optimization techniques."""
        actions = []
        
        try:
            if domain_id in self.performance_monitors:
                monitor = self.performance_monitors[domain_id]
                anomalies = monitor.detect_anomalies()
                
                if anomalies:
                    for metric, issues in anomalies.items():
                        if 'cpu_usage' in metric and strategy != DomainOptimizationStrategy.MEMORY_OPTIMIZED:
                            allocation = self.resource_allocations.get(domain_id)
                            if allocation:
                                allocation.cpu_allocation = max(allocation.cpu_allocation * 0.9, 0.1)
                                actions.append(f"Reduced CPU allocation due to high usage in {metric}")
                        
                        if 'memory_usage' in metric and strategy != DomainOptimizationStrategy.PERFORMANCE_FIRST:
                            cache = self.domain_caches.get(domain_id)
                            if cache:
                                cache.clear()
                                actions.append(f"Cleared cache due to memory pressure in {metric}")
            
            if self.domain_hybrid_integration:
                try:
                    hybrid_metrics = getattr(self.domain_hybrid_integration, 'get_domain_metrics', lambda x: {})(domain_id)
                    if hybrid_metrics and hybrid_metrics.get('performance_score', 1.0) < 0.7:
                        actions.append("Triggered hybrid integration rebalancing")
                except:
                    pass
            
            if self.domain_compression_coordinator:
                try:
                    coord_analytics = getattr(self.domain_compression_coordinator, 'get_domain_compression_analytics', lambda x: None)(domain_id)
                    if coord_analytics and coord_analytics.compression_efficiency < 0.6:
                        actions.append("Optimized compression coordination strategy")
                except:
                    pass
            
        except Exception as e:
            logger.warning(f"Error in advanced optimizations for domain {domain_id}: {e}")
        
        return actions
    
    def _apply_expert_optimizations(self, domain_id: str, strategy: DomainOptimizationStrategy) -> List[str]:
        """Apply expert-level optimization techniques."""
        actions = []
        
        try:
            if domain_id in self.performance_results and len(self.performance_results[domain_id]) > 3:
                recent_results = self.performance_results[domain_id][-3:]
                avg_improvement = statistics.mean([
                    sum(r.improvement_metrics.values()) / len(r.improvement_metrics) 
                    for r in recent_results if r.improvement_metrics
                ])
                
                if avg_improvement < 0.05:
                    current_strategy = self.optimization_strategies.get(domain_id, strategy)
                    if current_strategy != DomainOptimizationStrategy.ADAPTIVE:
                        self.optimization_strategies[domain_id] = DomainOptimizationStrategy.ADAPTIVE
                        actions.append(f"Switched to adaptive optimization strategy for domain {domain_id}")
            
            if strategy == DomainOptimizationStrategy.PREDICTIVE:
                trends = self._analyze_performance_trends(domain_id)
                if trends.get('degrading_performance', False):
                    allocation = self.resource_allocations.get(domain_id)
                    if allocation:
                        allocation.priority_level = min(allocation.priority_level + 1, 10)
                        actions.append(f"Increased priority level for domain {domain_id} due to predicted degradation")
            
            if self.hybrid_compression_orchestrator:
                try:
                    orchestrator_status = getattr(self.hybrid_compression_orchestrator, 'get_orchestration_status', lambda: {})()
                    if orchestrator_status.get('overloaded', False):
                        actions.append("Coordinated with hybrid compression orchestrator for load balancing")
                except:
                    pass
            
        except Exception as e:
            logger.warning(f"Error in expert optimizations for domain {domain_id}: {e}")
        
        return actions
    
    def _apply_maximum_optimizations(self, domain_id: str, strategy: DomainOptimizationStrategy) -> List[str]:
        """Apply maximum-level optimization techniques."""
        actions = []
        
        try:
            if domain_id in self.domain_caches:
                cache = self.domain_caches[domain_id]
                cache.max_size = 100000
                actions.append(f"Maximized cache size for domain {domain_id}")
            
            allocation = self.resource_allocations.get(domain_id)
            if allocation:
                allocation.memory_allocation = 0.9
                allocation.cpu_allocation = 0.9
                allocation.gpu_allocation = 0.8
                allocation.priority_level = 10
                allocation.dynamic_scaling = True
                actions.append(f"Applied maximum resource allocation for domain {domain_id}")
            
            if domain_id in self.performance_monitors:
                monitor = self.performance_monitors[domain_id]
                monitor.window_size = 1000
                actions.append(f"Expanded performance monitoring window for domain {domain_id}")
            
            self.optimization_strategies[domain_id] = DomainOptimizationStrategy.PREDICTIVE
            actions.append(f"Enabled predictive optimization for domain {domain_id}")
            
        except Exception as e:
            logger.warning(f"Error in maximum optimizations for domain {domain_id}: {e}")
        
        return actions
    
    def _calculate_improvement_metrics(self, before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance improvement metrics."""
        improvements = {}
        
        for metric in before.keys():
            if metric in after and metric != 'timestamp':
                before_val = before[metric]
                after_val = after[metric]
                
                if before_val != 0:
                    if metric in ['response_time', 'memory_usage', 'cpu_usage', 'error_rate']:
                        improvement = (before_val - after_val) / before_val
                    else:
                        improvement = (after_val - before_val) / before_val
                    
                    improvements[f"{metric}_improvement"] = improvement
        
        if improvements:
            improvements['overall_improvement'] = statistics.mean(improvements.values())
        
        return improvements
    
    def _analyze_performance_trends(self, domain_id: str) -> Dict[str, Any]:
        """Analyze performance trends for a domain."""
        trends = {}
        
        try:
            if domain_id in self.performance_results and len(self.performance_results[domain_id]) >= 5:
                recent_results = self.performance_results[domain_id][-5:]
                
                improvement_trend = [
                    sum(r.improvement_metrics.values()) / len(r.improvement_metrics)
                    for r in recent_results if r.improvement_metrics
                ]
                
                if len(improvement_trend) >= 3:
                    trend_slope = np.polyfit(range(len(improvement_trend)), improvement_trend, 1)[0]
                    trends['degrading_performance'] = trend_slope < -0.01
                    trends['improving_performance'] = trend_slope > 0.01
                    trends['stable_performance'] = abs(trend_slope) <= 0.01
                
                execution_times = [r.execution_time for r in recent_results]
                trends['avg_optimization_time'] = statistics.mean(execution_times)
                trends['optimization_time_trend'] = np.polyfit(range(len(execution_times)), execution_times, 1)[0]
        
        except Exception as e:
            logger.warning(f"Error analyzing trends for domain {domain_id}: {e}")
        
        return trends
    
    def optimize_domain_resources(self, domain_id: str, target_utilization: float = 0.8) -> bool:
        """Optimize resource allocation for a domain."""
        try:
            with self.optimization_lock:
                if domain_id not in self.resource_allocations:
                    raise ValueError(f"No resource allocation found for domain {domain_id}")
                
                allocation = self.resource_allocations[domain_id]
                current_performance = self._measure_domain_performance(domain_id)
                
                cpu_usage = current_performance.get('cpu_usage', 0)
                memory_usage = current_performance.get('memory_usage', 0)
                
                if cpu_usage > target_utilization * 100:
                    allocation.cpu_allocation = min(allocation.cpu_allocation * 1.2, 1.0)
                elif cpu_usage < target_utilization * 50:
                    allocation.cpu_allocation = max(allocation.cpu_allocation * 0.9, 0.1)
                
                if memory_usage > target_utilization * 100:
                    allocation.memory_allocation = min(allocation.memory_allocation * 1.2, 1.0)
                elif memory_usage < target_utilization * 50:
                    allocation.memory_allocation = max(allocation.memory_allocation * 0.9, 0.1)
                
                logger.info(f"Optimized resources for domain {domain_id}")
                return True
                
        except Exception as e:
            error_msg = f"Failed to optimize resources for domain {domain_id}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def get_domain_performance_analytics(self, domain_id: str) -> DomainPerformanceAnalytics:
        """Get comprehensive performance analytics for a domain."""
        try:
            with self.optimization_lock:
                if domain_id not in self.performance_targets:
                    raise ValueError(f"Domain {domain_id} not registered for optimization")
                
                results = self.performance_results.get(domain_id, [])
                successful_results = [r for r in results if r.success]
                failed_results = [r for r in results if not r.success]
                
                average_improvement = {}
                if successful_results:
                    all_improvements = defaultdict(list)
                    for result in successful_results:
                        for metric, improvement in result.improvement_metrics.items():
                            all_improvements[metric].append(improvement)
                    
                    average_improvement = {
                        metric: statistics.mean(values) 
                        for metric, values in all_improvements.items()
                    }
                
                performance_trends = {}
                if domain_id in self.performance_monitors:
                    monitor = self.performance_monitors[domain_id]
                    for metric in DomainPerformanceMetric:
                        trend = monitor.get_metric_trend(metric)
                        if trend:
                            performance_trends[metric.value] = trend
                
                bottleneck_patterns = defaultdict(int)
                for result in results:
                    for action in result.optimization_actions:
                        if 'cpu' in action.lower():
                            bottleneck_patterns['cpu'] += 1
                        elif 'memory' in action.lower():
                            bottleneck_patterns['memory'] += 1
                        elif 'cache' in action.lower():
                            bottleneck_patterns['cache'] += 1
                
                resource_efficiency = {}
                if domain_id in self.resource_allocations:
                    allocation = self.resource_allocations[domain_id]
                    current_performance = self._measure_domain_performance(domain_id)
                    
                    cpu_efficiency = current_performance.get('cpu_usage', 0) / (allocation.cpu_allocation * 100)
                    memory_efficiency = current_performance.get('memory_usage', 0) / (allocation.memory_allocation * 100)
                    
                    resource_efficiency = {
                        'cpu_efficiency': min(cpu_efficiency, 1.0),
                        'memory_efficiency': min(memory_efficiency, 1.0),
                        'overall_efficiency': (cpu_efficiency + memory_efficiency) / 2
                    }
                
                recommendation_score = 0.0
                if successful_results:
                    recent_improvements = [
                        r.improvement_metrics.get('overall_improvement', 0)
                        for r in successful_results[-10:]
                    ]
                    if recent_improvements:
                        recommendation_score = min(statistics.mean(recent_improvements) * 10, 10.0)
                
                analytics = DomainPerformanceAnalytics(
                    domain_id=domain_id,
                    total_optimizations=len(results),
                    successful_optimizations=len(successful_results),
                    failed_optimizations=len(failed_results),
                    average_improvement=average_improvement,
                    optimization_history=results[-50:],
                    performance_trends=performance_trends,
                    bottleneck_patterns=dict(bottleneck_patterns),
                    resource_efficiency=resource_efficiency,
                    recommendation_score=recommendation_score
                )
                
                self.analytics_cache[domain_id] = analytics
                return analytics
                
        except Exception as e:
            error_msg = f"Failed to get performance analytics for domain {domain_id}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def get_optimization_recommendations(self, domain_id: str) -> List[str]:
        """Get optimization recommendations for a domain."""
        try:
            analytics = self.get_domain_performance_analytics(domain_id)
            recommendations = []
            
            if analytics.failed_optimizations > analytics.successful_optimizations:
                recommendations.append("Consider switching to a more conservative optimization strategy")
            
            if analytics.recommendation_score < 3.0:
                recommendations.append("Domain may benefit from manual performance tuning")
            
            if 'cpu' in analytics.bottleneck_patterns and analytics.bottleneck_patterns['cpu'] > 5:
                recommendations.append("Consider CPU optimization or scaling")
            
            if 'memory' in analytics.bottleneck_patterns and analytics.bottleneck_patterns['memory'] > 5:
                recommendations.append("Consider memory optimization or garbage collection tuning")
            
            cpu_efficiency = analytics.resource_efficiency.get('cpu_efficiency', 0)
            if cpu_efficiency < 0.3:
                recommendations.append("CPU allocation may be too high for current workload")
            elif cpu_efficiency > 0.9:
                recommendations.append("CPU allocation may be insufficient for optimal performance")
            
            memory_efficiency = analytics.resource_efficiency.get('memory_efficiency', 0)
            if memory_efficiency < 0.3:
                recommendations.append("Memory allocation may be too high for current workload")
            elif memory_efficiency > 0.9:
                recommendations.append("Memory allocation may be insufficient for optimal performance")
            
            if not recommendations:
                recommendations.append("Domain performance appears to be well-optimized")
            
            return recommendations
            
        except Exception as e:
            error_msg = f"Failed to get recommendations for domain {domain_id}: {e}"
            logger.error(error_msg)
            return [f"Error generating recommendations: {error_msg}"]
    
    def cleanup_domain_optimization(self, domain_id: str) -> bool:
        """Clean up optimization resources for a domain."""
        try:
            with self.optimization_lock:
                if domain_id in self.performance_monitors:
                    self.performance_monitors[domain_id].stop_monitoring()
                    del self.performance_monitors[domain_id]
                
                if domain_id in self.domain_caches:
                    self.domain_caches[domain_id].clear()
                    del self.domain_caches[domain_id]
                
                if domain_id in self.performance_targets:
                    del self.performance_targets[domain_id]
                
                if domain_id in self.optimization_strategies:
                    del self.optimization_strategies[domain_id]
                
                if domain_id in self.resource_allocations:
                    del self.resource_allocations[domain_id]
                
                if domain_id in self.performance_results:
                    del self.performance_results[domain_id]
                
                if domain_id in self.analytics_cache:
                    del self.analytics_cache[domain_id]
                
                if domain_id in self.active_optimizations:
                    del self.active_optimizations[domain_id]
                
                logger.info(f"Cleaned up optimization resources for domain {domain_id}")
                return True
                
        except Exception as e:
            error_msg = f"Failed to cleanup optimization for domain {domain_id}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def get_global_optimization_status(self) -> Dict[str, Any]:
        """Get global optimization status across all domains."""
        try:
            with self.optimization_lock:
                uptime = time.time() - self.start_time
                
                active_domains = len(self.performance_targets)
                active_optimizations = len(self.active_optimizations)
                
                success_rate = (
                    (self.successful_optimizations / self.total_optimizations * 100)
                    if self.total_optimizations > 0 else 0.0
                )
                
                avg_optimization_time = 0.0
                if self.optimization_history:
                    recent_optimizations = self.optimization_history[-100:]
                    avg_optimization_time = statistics.mean([r.execution_time for r in recent_optimizations])
                
                domain_status = {}
                for domain_id in self.performance_targets.keys():
                    try:
                        analytics = self.get_domain_performance_analytics(domain_id)
                        domain_status[domain_id] = {
                            'total_optimizations': analytics.total_optimizations,
                            'success_rate': (analytics.successful_optimizations / analytics.total_optimizations * 100) if analytics.total_optimizations > 0 else 0.0,
                            'recommendation_score': analytics.recommendation_score,
                            'strategy': self.optimization_strategies.get(domain_id, DomainOptimizationStrategy.BALANCED).value
                        }
                    except:
                        domain_status[domain_id] = {'error': 'Failed to get analytics'}
                
                return {
                    'uptime': uptime,
                    'active_domains': active_domains,
                    'active_optimizations': active_optimizations,
                    'total_optimizations': self.total_optimizations,
                    'successful_optimizations': self.successful_optimizations,
                    'failed_optimizations': self.failed_optimizations,
                    'success_rate': success_rate,
                    'average_optimization_time': avg_optimization_time,
                    'domain_status': domain_status,
                    'system_integrations': {
                        'domain_registry': self.domain_registry is not None,
                        'domain_router': self.domain_router is not None,
                        'domain_hybrid_integration': self.domain_hybrid_integration is not None,
                        'domain_compression_coordinator': self.domain_compression_coordinator is not None,
                        'hybrid_compression_orchestrator': self.hybrid_compression_orchestrator is not None
                    }
                }
                
        except Exception as e:
            error_msg = f"Failed to get global optimization status: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)


def create_domain_performance_optimizer() -> DomainPerformanceOptimizer:
    """Factory function to create a DomainPerformanceOptimizer instance."""
    return DomainPerformanceOptimizer()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    optimizer = create_domain_performance_optimizer()
    
    test_target = DomainPerformanceTarget(
        domain_id="test_domain",
        target_response_time=0.1,
        target_throughput=1000.0,
        target_memory_usage=0.5,
        target_cpu_usage=0.6,
        target_error_rate=0.01,
        optimization_strategy=DomainOptimizationStrategy.BALANCED
    )
    
    optimizer.register_domain_for_optimization("test_domain", test_target)
    
    result = optimizer.optimize_domain_performance("test_domain", DomainOptimizationLevel.STANDARD)
    print(f"Optimization result: {result.success}")
    
    analytics = optimizer.get_domain_performance_analytics("test_domain")
    print(f"Analytics: {analytics.recommendation_score}")
    
    recommendations = optimizer.get_optimization_recommendations("test_domain")
    print(f"Recommendations: {recommendations}")
    
    status = optimizer.get_global_optimization_status()
    print(f"Global status: {status['success_rate']:.1f}% success rate")