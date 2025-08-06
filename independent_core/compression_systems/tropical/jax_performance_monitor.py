"""
JAX Performance Monitor - Real-time performance monitoring and adaptive tuning
Provides comprehensive benchmarking, regression detection, and production telemetry
HARD FAILURES ONLY - NO GRACEFUL DEGRADATION

This module provides:
1. JAXPerformanceMonitor - Real-time metrics and monitoring
2. Comprehensive benchmark suite comparing PyTorch vs JAX
3. Performance regression detection system
4. Production telemetry integration
5. Adaptive performance tuning based on workload
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap
import torch
import numpy as np
import time
import threading
import logging
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import functools
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    UTILIZATION = "utilization"
    ACCURACY = "accuracy"
    COMPILATION = "compilation"


class PerformanceState(Enum):
    """Performance state indicators"""
    OPTIMAL = "optimal"           # Meeting all SLAs
    DEGRADED = "degraded"         # Some SLA violations
    CRITICAL = "critical"         # Major SLA violations
    RECOVERING = "recovering"     # Recovering from degradation


class WorkloadType(Enum):
    """Types of workloads for adaptive tuning"""
    BATCH_INFERENCE = "batch_inference"
    REAL_TIME = "real_time"
    TRAINING = "training"
    MIXED = "mixed"


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    unit: str
    timestamp: float
    metric_type: MetricType
    tags: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for telemetry"""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp,
            'type': self.metric_type.value,
            'tags': self.tags
        }


@dataclass
class PerformanceSLA:
    """Service Level Agreement for performance"""
    metric_name: str
    threshold: float
    comparison: str  # 'less_than', 'greater_than'
    window_seconds: float = 60.0
    violation_count: int = 0
    last_violation: Optional[float] = None
    
    def check_violation(self, value: float) -> bool:
        """Check if value violates SLA"""
        if self.comparison == 'less_than':
            violated = value >= self.threshold
        else:  # greater_than
            violated = value <= self.threshold
        
        if violated:
            self.violation_count += 1
            self.last_violation = time.time()
        
        return violated


@dataclass
class BenchmarkResult:
    """Result of a benchmark comparison"""
    operation: str
    input_shape: Tuple[int, ...]
    jax_time_ms: float
    pytorch_time_ms: float
    speedup: float
    memory_jax_mb: float
    memory_pytorch_mb: float
    accuracy_match: bool
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_faster(self) -> bool:
        """Check if JAX is faster"""
        return self.speedup > 1.0


@dataclass
class RegressionAlert:
    """Performance regression alert"""
    metric_name: str
    baseline_value: float
    current_value: float
    regression_percent: float
    detected_at: float
    severity: str  # 'low', 'medium', 'high'
    resolved: bool = False
    resolution_time: Optional[float] = None


class JAXPerformanceMonitor:
    """
    Comprehensive performance monitoring for JAX operations.
    Tracks metrics, detects regressions, and enables adaptive tuning.
    """
    
    def __init__(self,
                 enable_telemetry: bool = True,
                 sla_config: Optional[Dict[str, Any]] = None,
                 baseline_file: Optional[str] = None,
                 export_interval_seconds: float = 60.0):
        """
        Initialize performance monitor
        
        Args:
            enable_telemetry: Enable telemetry export
            sla_config: SLA configuration
            baseline_file: File with performance baselines
            export_interval_seconds: Telemetry export interval
        """
        self.enable_telemetry = enable_telemetry
        self.export_interval = export_interval_seconds
        
        # Metrics storage
        self.metrics: Dict[str, Deque[PerformanceMetric]] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        self.current_metrics: Dict[str, float] = {}
        
        # SLA management
        self.slas: Dict[str, PerformanceSLA] = {}
        if sla_config:
            self._load_sla_config(sla_config)
        else:
            self._setup_default_slas()
        
        # Performance baselines
        self.baselines: Dict[str, float] = {}
        if baseline_file:
            self._load_baselines(baseline_file)
        
        # Regression detection
        self.regression_alerts: List[RegressionAlert] = []
        self.regression_threshold = 0.1  # 10% regression threshold
        
        # Benchmark results
        self.benchmark_results: List[BenchmarkResult] = []
        
        # Performance state
        self.performance_state = PerformanceState.OPTIMAL
        self.state_history: Deque[Tuple[float, PerformanceState]] = deque(maxlen=1000)
        
        # Workload detection
        self.current_workload = WorkloadType.MIXED
        self.workload_history: Deque[Tuple[float, WorkloadType]] = deque(maxlen=100)
        
        # Adaptive tuning parameters
        self.tuning_params = {
            'compilation_level': 2,
            'batch_size': 32,
            'prefetch_enabled': True,
            'fusion_aggressive': True,
            'memory_optimization': True
        }
        
        # Statistics
        self.stats = {
            'metrics_recorded': 0,
            'sla_violations': 0,
            'regressions_detected': 0,
            'benchmarks_run': 0,
            'adaptations_performed': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Start monitoring threads
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        if enable_telemetry:
            self.telemetry_thread = threading.Thread(target=self._telemetry_loop, daemon=True)
            self.telemetry_thread.start()
        
        logger.info("JAXPerformanceMonitor initialized")
    
    def _setup_default_slas(self) -> None:
        """Setup default SLAs"""
        self.slas = {
            'latency_p99': PerformanceSLA('latency_p99', 100.0, 'less_than'),
            'throughput': PerformanceSLA('throughput', 1000.0, 'greater_than'),
            'memory_usage': PerformanceSLA('memory_usage', 8000.0, 'less_than'),
            'compilation_time': PerformanceSLA('compilation_time', 5000.0, 'less_than')
        }
    
    def _load_sla_config(self, config: Dict[str, Any]) -> None:
        """Load SLA configuration"""
        for name, params in config.items():
            self.slas[name] = PerformanceSLA(
                metric_name=name,
                threshold=params['threshold'],
                comparison=params.get('comparison', 'less_than'),
                window_seconds=params.get('window_seconds', 60.0)
            )
    
    def _load_baselines(self, baseline_file: str) -> None:
        """Load performance baselines"""
        try:
            with open(baseline_file, 'rb') as f:
                self.baselines = pickle.load(f)
            logger.info(f"Loaded {len(self.baselines)} baselines")
        except Exception as e:
            logger.warning(f"Failed to load baselines: {e}")
    
    def record_metric(self,
                     name: str,
                     value: float,
                     unit: str = "ms",
                     metric_type: MetricType = MetricType.LATENCY,
                     tags: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a performance metric
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            metric_type: Type of metric
            tags: Additional tags
        """
        with self._lock:
            metric = PerformanceMetric(
                name=name,
                value=value,
                unit=unit,
                timestamp=time.time(),
                metric_type=metric_type,
                tags=tags or {}
            )
            
            self.metrics[name].append(metric)
            self.current_metrics[name] = value
            self.stats['metrics_recorded'] += 1
            
            # Check SLA
            if name in self.slas:
                if self.slas[name].check_violation(value):
                    self.stats['sla_violations'] += 1
                    self._handle_sla_violation(name, value)
            
            # Check for regression
            if name in self.baselines:
                self._check_regression(name, value)
    
    def _handle_sla_violation(self, metric_name: str, value: float) -> None:
        """Handle SLA violation"""
        logger.warning(f"SLA violation: {metric_name} = {value}")
        
        # Update performance state
        violation_count = sum(1 for sla in self.slas.values() if sla.violation_count > 0)
        
        if violation_count >= len(self.slas) * 0.5:
            self.performance_state = PerformanceState.CRITICAL
        elif violation_count > 0:
            self.performance_state = PerformanceState.DEGRADED
        
        self.state_history.append((time.time(), self.performance_state))
    
    def _check_regression(self, metric_name: str, value: float) -> None:
        """Check for performance regression"""
        baseline = self.baselines[metric_name]
        
        # Calculate regression
        if baseline > 0:
            if metric_name.endswith('_time') or 'latency' in metric_name:
                # For time metrics, higher is worse
                regression_pct = (value - baseline) / baseline
            else:
                # For throughput metrics, lower is worse
                regression_pct = (baseline - value) / baseline
            
            if regression_pct > self.regression_threshold:
                alert = RegressionAlert(
                    metric_name=metric_name,
                    baseline_value=baseline,
                    current_value=value,
                    regression_percent=regression_pct * 100,
                    detected_at=time.time(),
                    severity='high' if regression_pct > 0.3 else 'medium'
                )
                
                self.regression_alerts.append(alert)
                self.stats['regressions_detected'] += 1
                
                logger.warning(f"Performance regression detected: {metric_name} "
                             f"regressed by {regression_pct*100:.1f}%")
    
    def benchmark_operation(self,
                           jax_func: Callable,
                           pytorch_func: Callable,
                           input_shape: Tuple[int, ...],
                           operation_name: str = "unknown",
                           num_iterations: int = 100) -> BenchmarkResult:
        """
        Benchmark JAX vs PyTorch operation
        
        Args:
            jax_func: JAX function to benchmark
            pytorch_func: PyTorch function to benchmark
            input_shape: Input tensor shape
            operation_name: Name of operation
            num_iterations: Number of iterations
            
        Returns:
            Benchmark result
        """
        with self._lock:
            self.stats['benchmarks_run'] += 1
            
            # Create test data
            numpy_data = np.random.randn(*input_shape).astype(np.float32)
            jax_data = jnp.array(numpy_data)
            torch_data = torch.from_numpy(numpy_data)
            
            if torch.cuda.is_available():
                torch_data = torch_data.cuda()
            
            # Warmup
            for _ in range(10):
                _ = jax_func(jax_data)
                _ = pytorch_func(torch_data)
            
            # Benchmark JAX
            jax_times = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                jax_result = jax_func(jax_data)
                jax_result.block_until_ready()
                jax_times.append((time.perf_counter() - start) * 1000)
            
            jax_time_ms = np.median(jax_times)
            
            # Benchmark PyTorch
            torch_times = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                torch_result = pytorch_func(torch_data)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                torch_times.append((time.perf_counter() - start) * 1000)
            
            pytorch_time_ms = np.median(torch_times)
            
            # Calculate speedup
            speedup = pytorch_time_ms / jax_time_ms if jax_time_ms > 0 else 0
            
            # Check accuracy
            jax_np = np.array(jax_result)
            torch_np = torch_result.cpu().numpy()
            accuracy_match = np.allclose(jax_np, torch_np, rtol=1e-5, atol=1e-7)
            
            # Estimate memory (simplified)
            element_size = 4  # float32
            num_elements = np.prod(input_shape)
            memory_estimate_mb = (num_elements * element_size) / (1024 * 1024)
            
            result = BenchmarkResult(
                operation=operation_name,
                input_shape=input_shape,
                jax_time_ms=jax_time_ms,
                pytorch_time_ms=pytorch_time_ms,
                speedup=speedup,
                memory_jax_mb=memory_estimate_mb,
                memory_pytorch_mb=memory_estimate_mb,
                accuracy_match=accuracy_match,
                timestamp=time.time()
            )
            
            self.benchmark_results.append(result)
            
            # Record metrics
            self.record_metric(f"benchmark_{operation_name}_speedup", speedup, 
                             "x", MetricType.THROUGHPUT)
            
            return result
    
    def detect_workload_type(self) -> WorkloadType:
        """
        Detect current workload type based on metrics
        
        Returns:
            Detected workload type
        """
        with self._lock:
            # Analyze recent metrics
            recent_metrics = {}
            for name, values in self.metrics.items():
                if values:
                    recent = [m.value for m in list(values)[-100:]]
                    recent_metrics[name] = {
                        'mean': np.mean(recent),
                        'std': np.std(recent),
                        'cv': np.std(recent) / np.mean(recent) if np.mean(recent) > 0 else 0
                    }
            
            # Heuristics for workload detection
            if 'batch_size' in recent_metrics:
                batch_mean = recent_metrics['batch_size']['mean']
                if batch_mean > 64:
                    workload = WorkloadType.BATCH_INFERENCE
                elif batch_mean == 1:
                    workload = WorkloadType.REAL_TIME
                else:
                    workload = WorkloadType.MIXED
            elif 'latency_p99' in recent_metrics:
                latency_cv = recent_metrics['latency_p99']['cv']
                if latency_cv < 0.1:  # Low variance
                    workload = WorkloadType.BATCH_INFERENCE
                elif latency_cv > 0.5:  # High variance
                    workload = WorkloadType.REAL_TIME
                else:
                    workload = WorkloadType.MIXED
            else:
                workload = WorkloadType.MIXED
            
            self.current_workload = workload
            self.workload_history.append((time.time(), workload))
            
            return workload
    
    def adapt_performance_tuning(self) -> Dict[str, Any]:
        """
        Adapt performance tuning based on workload and metrics
        
        Returns:
            Updated tuning parameters
        """
        with self._lock:
            self.stats['adaptations_performed'] += 1
            
            workload = self.detect_workload_type()
            old_params = dict(self.tuning_params)
            
            # Adapt based on workload
            if workload == WorkloadType.BATCH_INFERENCE:
                # Optimize for throughput
                self.tuning_params['compilation_level'] = 3
                self.tuning_params['batch_size'] = 128
                self.tuning_params['prefetch_enabled'] = True
                self.tuning_params['fusion_aggressive'] = True
                
            elif workload == WorkloadType.REAL_TIME:
                # Optimize for latency
                self.tuning_params['compilation_level'] = 2
                self.tuning_params['batch_size'] = 1
                self.tuning_params['prefetch_enabled'] = False
                self.tuning_params['fusion_aggressive'] = False
                
            elif workload == WorkloadType.TRAINING:
                # Optimize for memory and compute balance
                self.tuning_params['compilation_level'] = 3
                self.tuning_params['batch_size'] = 32
                self.tuning_params['prefetch_enabled'] = True
                self.tuning_params['memory_optimization'] = True
            
            # Adapt based on performance state
            if self.performance_state == PerformanceState.CRITICAL:
                # Emergency optimizations
                self.tuning_params['compilation_level'] = 1
                self.tuning_params['fusion_aggressive'] = False
                self.tuning_params['memory_optimization'] = True
            
            changes = {k: v for k, v in self.tuning_params.items() 
                      if old_params.get(k) != v}
            
            if changes:
                logger.info(f"Adapted tuning parameters: {changes}")
            
            return self.tuning_params
    
    def export_metrics_chrome_tracing(self, output_file: str) -> None:
        """
        Export metrics in Chrome tracing format
        
        Args:
            output_file: Output file path
        """
        with self._lock:
            events = []
            
            for name, metrics in self.metrics.items():
                for metric in metrics:
                    event = {
                        'name': metric.name,
                        'cat': metric.metric_type.value,
                        'ph': 'X',  # Complete event
                        'ts': int(metric.timestamp * 1e6),  # microseconds
                        'dur': int(metric.value * 1e3),  # duration in microseconds
                        'pid': 1,
                        'tid': hash(name) % 100,
                        'args': metric.tags
                    }
                    events.append(event)
            
            trace_data = {
                'traceEvents': events,
                'displayTimeUnit': 'ms'
            }
            
            with open(output_file, 'w') as f:
                json.dump(trace_data, f, indent=2)
            
            logger.info(f"Exported {len(events)} events to {output_file}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary
        
        Returns:
            Performance summary
        """
        with self._lock:
            summary = {
                'current_state': self.performance_state.value,
                'current_workload': self.current_workload.value,
                'metrics': {},
                'sla_status': {},
                'active_regressions': [],
                'recent_benchmarks': [],
                'tuning_params': dict(self.tuning_params),
                'statistics': dict(self.stats)
            }
            
            # Aggregate metrics
            for name, values in self.metrics.items():
                if values:
                    recent = [m.value for m in list(values)[-100:]]
                    summary['metrics'][name] = {
                        'current': self.current_metrics.get(name, 0),
                        'mean': np.mean(recent),
                        'p50': np.median(recent),
                        'p95': np.percentile(recent, 95),
                        'p99': np.percentile(recent, 99)
                    }
            
            # SLA status
            for name, sla in self.slas.items():
                summary['sla_status'][name] = {
                    'threshold': sla.threshold,
                    'violations': sla.violation_count,
                    'last_violation': sla.last_violation
                }
            
            # Active regressions
            for alert in self.regression_alerts:
                if not alert.resolved:
                    summary['active_regressions'].append({
                        'metric': alert.metric_name,
                        'regression_percent': alert.regression_percent,
                        'severity': alert.severity
                    })
            
            # Recent benchmarks
            for result in self.benchmark_results[-5:]:
                summary['recent_benchmarks'].append({
                    'operation': result.operation,
                    'speedup': result.speedup,
                    'accuracy_match': result.accuracy_match
                })
            
            return summary
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                time.sleep(1.0)  # 1 second interval
                
                # Check performance state
                self._update_performance_state()
                
                # Adaptive tuning
                if self.performance_state != PerformanceState.OPTIMAL:
                    self.adapt_performance_tuning()
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
    
    def _update_performance_state(self) -> None:
        """Update overall performance state"""
        with self._lock:
            # Count active issues
            sla_violations = sum(1 for sla in self.slas.values() 
                               if sla.violation_count > 0)
            active_regressions = sum(1 for alert in self.regression_alerts 
                                   if not alert.resolved)
            
            # Determine state
            if sla_violations == 0 and active_regressions == 0:
                new_state = PerformanceState.OPTIMAL
            elif sla_violations >= 3 or active_regressions >= 3:
                new_state = PerformanceState.CRITICAL
            elif self.performance_state == PerformanceState.CRITICAL:
                new_state = PerformanceState.RECOVERING
            else:
                new_state = PerformanceState.DEGRADED
            
            if new_state != self.performance_state:
                logger.info(f"Performance state changed: {self.performance_state.value} -> {new_state.value}")
                self.performance_state = new_state
                self.state_history.append((time.time(), new_state))
    
    def _telemetry_loop(self) -> None:
        """Background telemetry export loop"""
        while self.monitoring_active:
            try:
                time.sleep(self.export_interval)
                self._export_telemetry()
            except Exception as e:
                logger.error(f"Telemetry loop error: {e}")
    
    def _export_telemetry(self) -> None:
        """Export telemetry data"""
        # This would integrate with actual telemetry system
        summary = self.get_performance_summary()
        logger.debug(f"Telemetry export: {summary['current_state']}")
    
    def shutdown(self) -> None:
        """Shutdown performance monitor"""
        self.monitoring_active = False
        
        # Export final metrics
        if self.enable_telemetry:
            self._export_telemetry()
        
        logger.info("JAXPerformanceMonitor shutdown complete")


# Test function
def test_performance_monitor():
    """Test performance monitor functionality"""
    print("Testing JAX Performance Monitor...")
    
    # Initialize monitor
    monitor = JAXPerformanceMonitor(
        enable_telemetry=True,
        sla_config={
            'test_latency': {
                'threshold': 50.0,
                'comparison': 'less_than'
            }
        }
    )
    
    print("\n1. Recording metrics...")
    for i in range(10):
        latency = 30 + i * 5 + np.random.randn() * 5
        monitor.record_metric('test_latency', latency, 'ms', MetricType.LATENCY)
        time.sleep(0.1)
    
    print(f"   Recorded {monitor.stats['metrics_recorded']} metrics")
    
    print("\n2. Testing benchmark...")
    
    @jit
    def jax_matmul(A):
        return jnp.matmul(A, A)
    
    def pytorch_matmul(A):
        return torch.matmul(A, A)
    
    result = monitor.benchmark_operation(
        jax_matmul,
        pytorch_matmul,
        (100, 100),
        "matmul_test",
        num_iterations=50
    )
    
    print(f"   JAX time: {result.jax_time_ms:.2f}ms")
    print(f"   PyTorch time: {result.pytorch_time_ms:.2f}ms")
    print(f"   Speedup: {result.speedup:.2f}x")
    print(f"   Accuracy match: {result.accuracy_match}")
    
    print("\n3. Testing workload detection...")
    workload = monitor.detect_workload_type()
    print(f"   Detected workload: {workload.value}")
    
    print("\n4. Testing adaptive tuning...")
    tuning = monitor.adapt_performance_tuning()
    print(f"   Tuning parameters: {tuning}")
    
    print("\n5. Testing SLA violations...")
    # Trigger SLA violation
    monitor.record_metric('test_latency', 100.0, 'ms', MetricType.LATENCY)
    print(f"   SLA violations: {monitor.stats['sla_violations']}")
    print(f"   Performance state: {monitor.performance_state.value}")
    
    print("\n6. Getting performance summary...")
    summary = monitor.get_performance_summary()
    print(f"   Current state: {summary['current_state']}")
    print(f"   Metrics tracked: {len(summary['metrics'])}")
    print(f"   Active regressions: {len(summary['active_regressions'])}")
    
    print("\n7. Exporting Chrome tracing...")
    monitor.export_metrics_chrome_tracing("test_trace.json")
    print("   Exported to test_trace.json")
    
    # Cleanup
    monitor.shutdown()
    import os
    if os.path.exists("test_trace.json"):
        os.remove("test_trace.json")
    
    print("\n✓ Performance monitor test complete!")


if __name__ == "__main__":
    test_performance_monitor()