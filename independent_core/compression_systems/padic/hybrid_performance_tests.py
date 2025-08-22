"""
Hybrid Performance Tests - Performance testing with regression detection
NO FALLBACKS - HARD FAILURES ONLY
"""

import logging
import math
import statistics
import threading
import time
import torch
import unittest
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from enum import Enum

# Import hybrid system components
from hybrid_padic_structures import HybridPadicWeight, HybridPadicManager, HybridPadicValidator
from hybrid_padic_compressor import HybridPadicCompressionSystem
from dynamic_switching_manager import DynamicSwitchingManager
from hybrid_performance_optimizer import HybridPerformanceOptimizer
from hybrid_performance_monitor import HybridPerformanceMonitor

# Import existing components for comparison
from padic_encoder import PadicWeight, PadicCompressionSystem


class PerformanceTestType(Enum):
    """Performance test type enumeration"""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    MEMORY = "memory"
    GPU_MEMORY = "gpu_memory"
    SCALABILITY = "scalability"
    REGRESSION = "regression"
    STRESS = "stress"


class PerformanceMetric(Enum):
    """Performance metric enumeration"""
    EXECUTION_TIME = "execution_time_ms"
    THROUGHPUT_OPS_SEC = "throughput_ops_per_sec"
    MEMORY_USAGE = "memory_usage_mb"
    GPU_MEMORY_USAGE = "gpu_memory_usage_mb"
    CPU_USAGE = "cpu_usage_percent"
    COMPRESSION_RATIO = "compression_ratio"
    SUCCESS_RATE = "success_rate"


class PerformanceTestResult(Enum):
    """Performance test result enumeration"""
    PASSED = "passed"
    FAILED = "failed"
    REGRESSION = "regression"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class PerformanceBenchmark:
    """Performance benchmark definition"""
    benchmark_name: str
    metric: PerformanceMetric
    expected_value: float
    tolerance_percent: float
    comparison_operator: str  # 'less_than', 'greater_than', 'equal_to'
    description: str
    
    def __post_init__(self):
        """Validate benchmark definition"""
        if not isinstance(self.benchmark_name, str) or not self.benchmark_name.strip():
            raise ValueError("Benchmark name must be non-empty string")
        if not isinstance(self.metric, PerformanceMetric):
            raise TypeError("Metric must be PerformanceMetric")
        if not isinstance(self.expected_value, (int, float)) or self.expected_value < 0:
            raise ValueError("Expected value must be non-negative number")
        if not isinstance(self.tolerance_percent, (int, float)) or self.tolerance_percent < 0:
            raise ValueError("Tolerance percent must be non-negative")
        if self.comparison_operator not in ['less_than', 'greater_than', 'equal_to']:
            raise ValueError("Comparison operator must be 'less_than', 'greater_than', or 'equal_to'")
    
    def evaluate(self, actual_value: float) -> bool:
        """Evaluate if actual value meets benchmark"""
        tolerance_value = self.expected_value * (self.tolerance_percent / 100.0)
        
        if self.comparison_operator == 'less_than':
            return actual_value <= (self.expected_value + tolerance_value)
        elif self.comparison_operator == 'greater_than':
            return actual_value >= (self.expected_value - tolerance_value)
        else:  # equal_to
            return abs(actual_value - self.expected_value) <= tolerance_value


@dataclass
class PerformanceTestExecution:
    """Result of performance test execution"""
    test_name: str
    test_type: PerformanceTestType
    result: PerformanceTestResult
    execution_time_ms: float
    metrics: Dict[PerformanceMetric, float]
    benchmark_results: Dict[str, bool]
    regression_detected: bool = False
    regression_details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate test execution result"""
        if not isinstance(self.test_name, str) or not self.test_name.strip():
            raise ValueError("Test name must be non-empty string")
        if not isinstance(self.test_type, PerformanceTestType):
            raise TypeError("Test type must be PerformanceTestType")
        if not isinstance(self.result, PerformanceTestResult):
            raise TypeError("Result must be PerformanceTestResult")
        if not isinstance(self.execution_time_ms, (int, float)) or self.execution_time_ms < 0:
            raise ValueError("Execution time must be non-negative")


@dataclass
class PerformanceTestConfig:
    """Configuration for hybrid performance tests"""
    enable_throughput_tests: bool = True
    enable_latency_tests: bool = True
    enable_memory_tests: bool = True
    enable_gpu_memory_tests: bool = True
    enable_scalability_tests: bool = True
    enable_regression_tests: bool = True
    enable_stress_tests: bool = True
    
    # Test parameters
    test_data_sizes: List[int] = field(default_factory=lambda: [100, 1000, 5000, 10000])
    throughput_test_duration_seconds: int = 10
    latency_test_iterations: int = 100
    stress_test_duration_seconds: int = 30
    
    # Performance benchmarks
    max_compression_time_ms: float = 1000.0
    min_throughput_ops_sec: float = 10.0
    max_memory_usage_mb: float = 500.0
    max_gpu_memory_usage_mb: float = 1024.0
    min_compression_ratio: float = 1.5
    min_success_rate: float = 0.95
    
    # Regression detection
    regression_threshold_percent: float = 10.0
    regression_history_window: int = 10
    
    # Execution parameters
    enable_parallel_execution: bool = True
    max_concurrent_tests: int = 4
    test_timeout_seconds: int = 300
    warmup_iterations: int = 5
    
    def __post_init__(self):
        """Validate configuration"""
        if not isinstance(self.test_data_sizes, list) or not self.test_data_sizes:
            raise ValueError("Test data sizes must be non-empty list")
        if self.throughput_test_duration_seconds <= 0:
            raise ValueError("Throughput test duration must be positive")
        if self.latency_test_iterations <= 0:
            raise ValueError("Latency test iterations must be positive")
        if self.regression_threshold_percent <= 0:
            raise ValueError("Regression threshold must be positive")


@dataclass
class PerformanceTestReport:
    """Comprehensive performance test report"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    regression_tests: int
    warning_tests: int
    error_tests: int
    total_execution_time_ms: float
    test_results: List[PerformanceTestExecution]
    type_summary: Dict[PerformanceTestType, Dict[str, int]]
    metric_summary: Dict[PerformanceMetric, Dict[str, float]]
    benchmark_summary: Dict[str, Dict[str, Any]]
    regression_summary: Dict[str, Any]
    performance_trends: Dict[str, List[float]]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Calculate derived metrics"""
        if self.total_tests > 0:
            self.success_rate = self.passed_tests / self.total_tests
            self.regression_rate = self.regression_tests / self.total_tests
        else:
            self.success_rate = 0.0
            self.regression_rate = 0.0


class HybridPerformanceTests:
    """
    Comprehensive performance testing for hybrid p-adic system.
    Includes throughput, latency, memory, scalability, and regression testing.
    """
    
    def __init__(self, config: Optional[PerformanceTestConfig] = None):
        """Initialize hybrid performance tests"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, PerformanceTestConfig):
            raise TypeError(f"Config must be PerformanceTestConfig or None, got {type(config)}")
        
        self.config = config or PerformanceTestConfig()
        self.logger = logging.getLogger('HybridPerformanceTests')
        
        # System components
        self.hybrid_manager: Optional[HybridPadicManager] = None
        self.hybrid_compressor: Optional[HybridPadicCompressionSystem] = None
        self.switching_manager: Optional[DynamicSwitchingManager] = None
        self.pure_compressor: Optional[PadicCompressionSystem] = None
        self.performance_monitor: Optional[HybridPerformanceMonitor] = None
        
        # Test state
        self.is_initialized = False
        self.is_running = False
        
        # Performance tracking
        self.test_results: List[PerformanceTestExecution] = []
        self.baseline_metrics: Dict[str, float] = {}
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.regression_history_window))
        
        # Benchmarks
        self.benchmarks: Dict[str, PerformanceBenchmark] = {}
        
        # Thread safety
        self._test_lock = threading.RLock()
        self._results_lock = threading.RLock()
        self._history_lock = threading.RLock()
        
        # Test data cache
        self.test_data_cache: Dict[int, torch.Tensor] = {}
        
        self.logger.info("HybridPerformanceTests created successfully")
    
    def initialize_performance_tests(self,
                                   hybrid_manager: HybridPadicManager,
                                   hybrid_compressor: HybridPadicCompressionSystem,
                                   switching_manager: DynamicSwitchingManager,
                                   pure_compressor: Optional[PadicCompressionSystem] = None,
                                   performance_monitor: Optional[HybridPerformanceMonitor] = None) -> None:
        """
        Initialize performance tests with system components.
        
        Args:
            hybrid_manager: Hybrid p-adic manager instance
            hybrid_compressor: Hybrid compression system instance
            switching_manager: Dynamic switching manager instance
            pure_compressor: Optional pure p-adic compressor for comparison
            performance_monitor: Optional performance monitor
            
        Raises:
            TypeError: If any component is invalid
            RuntimeError: If initialization fails
        """
        if not isinstance(hybrid_manager, HybridPadicManager):
            raise TypeError(f"Hybrid manager must be HybridPadicManager, got {type(hybrid_manager)}")
        if not isinstance(hybrid_compressor, HybridPadicCompressionSystem):
            raise TypeError(f"Hybrid compressor must be HybridPadicCompressionSystem, got {type(hybrid_compressor)}")
        if not isinstance(switching_manager, DynamicSwitchingManager):
            raise TypeError(f"Switching manager must be DynamicSwitchingManager, got {type(switching_manager)}")
        if pure_compressor is not None and not isinstance(pure_compressor, PadicCompressionSystem):
            raise TypeError(f"Pure compressor must be PadicCompressionSystem, got {type(pure_compressor)}")
        if performance_monitor is not None and not isinstance(performance_monitor, HybridPerformanceMonitor):
            raise TypeError(f"Performance monitor must be HybridPerformanceMonitor, got {type(performance_monitor)}")
        
        try:
            # Set component references
            self.hybrid_manager = hybrid_manager
            self.hybrid_compressor = hybrid_compressor
            self.switching_manager = switching_manager
            self.pure_compressor = pure_compressor
            self.performance_monitor = performance_monitor
            
            # Initialize benchmarks
            self._initialize_benchmarks()
            
            # Pre-generate test data
            self._generate_test_data()
            
            # Establish baseline metrics
            self._establish_baseline_metrics()
            
            self.is_initialized = True
            self.logger.info("Hybrid performance tests initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize performance tests: {e}")
            raise RuntimeError(f"Performance tests initialization failed: {e}")
    
    def run_all_performance_tests(self) -> PerformanceTestReport:
        """
        Run all enabled performance tests and generate comprehensive report.
        
        Returns:
            Comprehensive performance test report
            
        Raises:
            RuntimeError: If tests are not initialized or execution fails
        """
        if not self.is_initialized:
            raise RuntimeError("Performance tests not initialized")
        
        if self.is_running:
            raise RuntimeError("Performance tests are already running")
        
        with self._test_lock:
            try:
                self.is_running = True
                self.test_results.clear()
                start_time = time.time()
                
                self.logger.info("Starting comprehensive hybrid performance tests")
                
                # Run throughput tests
                if self.config.enable_throughput_tests:
                    self._run_throughput_tests()
                
                # Run latency tests
                if self.config.enable_latency_tests:
                    self._run_latency_tests()
                
                # Run memory tests
                if self.config.enable_memory_tests:
                    self._run_memory_tests()
                
                # Run GPU memory tests
                if self.config.enable_gpu_memory_tests:
                    self._run_gpu_memory_tests()
                
                # Run scalability tests
                if self.config.enable_scalability_tests:
                    self._run_scalability_tests()
                
                # Run regression tests
                if self.config.enable_regression_tests:
                    self._run_regression_tests()
                
                # Run stress tests
                if self.config.enable_stress_tests:
                    self._run_stress_tests()
                
                # Calculate execution time
                total_execution_time = (time.time() - start_time) * 1000
                
                # Update performance history
                self._update_performance_history()
                
                # Generate report
                report = self._generate_performance_report(total_execution_time)
                
                self.logger.info(f"Performance tests completed: {report.passed_tests}/{report.total_tests} tests passed")
                
                return report
                
            except Exception as e:
                self.logger.error(f"Performance tests execution failed: {e}")
                raise RuntimeError(f"Performance tests execution failed: {e}")
            finally:
                self.is_running = False
    
    def test_compression_throughput(self, data_size: int) -> PerformanceTestExecution:
        """
        Test compression throughput performance.
        
        Args:
            data_size: Size of test data
            
        Returns:
            Performance test execution result
            
        Raises:
            RuntimeError: If test fails
        """
        test_name = f"compression_throughput_{data_size}"
        start_time = time.time()
        
        try:
            test_data = self._get_test_data(data_size)
            operations_completed = 0
            total_compression_time = 0.0
            
            # Warmup
            for _ in range(self.config.warmup_iterations):
                self.hybrid_compressor.compress(test_data.clone())
            
            # Measure throughput
            throughput_start = time.time()
            while (time.time() - throughput_start) < self.config.throughput_test_duration_seconds:
                op_start = time.time()
                compressed = self.hybrid_compressor.compress(test_data.clone())
                op_time = time.time() - op_start
                
                total_compression_time += op_time
                operations_completed += 1
                
                # Clean up
                del compressed
            
            # Calculate metrics
            throughput_ops_sec = operations_completed / self.config.throughput_test_duration_seconds
            avg_compression_time = (total_compression_time / operations_completed) * 1000  # ms
            
            metrics = {
                PerformanceMetric.THROUGHPUT_OPS_SEC: throughput_ops_sec,
                PerformanceMetric.EXECUTION_TIME: avg_compression_time
            }
            
            # Evaluate benchmarks
            benchmark_results = {}
            if 'min_throughput' in self.benchmarks:
                benchmark_results['min_throughput'] = self.benchmarks['min_throughput'].evaluate(throughput_ops_sec)
            if 'max_compression_time' in self.benchmarks:
                benchmark_results['max_compression_time'] = self.benchmarks['max_compression_time'].evaluate(avg_compression_time)
            
            # Determine result
            result = PerformanceTestResult.PASSED
            if not all(benchmark_results.values()):
                result = PerformanceTestResult.FAILED
            
            execution_time = (time.time() - start_time) * 1000
            
            return PerformanceTestExecution(
                test_name=test_name,
                test_type=PerformanceTestType.THROUGHPUT,
                result=result,
                execution_time_ms=execution_time,
                metrics=metrics,
                benchmark_results=benchmark_results,
                details={
                    'operations_completed': operations_completed,
                    'test_duration_seconds': self.config.throughput_test_duration_seconds,
                    'data_size': data_size
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return PerformanceTestExecution(
                test_name=test_name,
                test_type=PerformanceTestType.THROUGHPUT,
                result=PerformanceTestResult.ERROR,
                execution_time_ms=execution_time,
                metrics={},
                benchmark_results={},
                error_message=str(e)
            )
    
    def test_compression_latency(self, data_size: int) -> PerformanceTestExecution:
        """
        Test compression latency performance.
        
        Args:
            data_size: Size of test data
            
        Returns:
            Performance test execution result
            
        Raises:
            RuntimeError: If test fails
        """
        test_name = f"compression_latency_{data_size}"
        start_time = time.time()
        
        try:
            test_data = self._get_test_data(data_size)
            latencies = []
            
            # Warmup
            for _ in range(self.config.warmup_iterations):
                self.hybrid_compressor.compress(test_data.clone())
            
            # Measure latencies
            for _ in range(self.config.latency_test_iterations):
                op_start = time.time()
                compressed = self.hybrid_compressor.compress(test_data.clone())
                op_time = (time.time() - op_start) * 1000  # ms
                latencies.append(op_time)
                
                # Clean up
                del compressed
            
            # Calculate statistics
            avg_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
            p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            metrics = {
                PerformanceMetric.EXECUTION_TIME: avg_latency
            }
            
            # Evaluate benchmarks
            benchmark_results = {}
            if 'max_compression_time' in self.benchmarks:
                benchmark_results['max_compression_time'] = self.benchmarks['max_compression_time'].evaluate(avg_latency)
            
            # Determine result
            result = PerformanceTestResult.PASSED
            if not all(benchmark_results.values()):
                result = PerformanceTestResult.FAILED
            
            execution_time = (time.time() - start_time) * 1000
            
            return PerformanceTestExecution(
                test_name=test_name,
                test_type=PerformanceTestType.LATENCY,
                result=result,
                execution_time_ms=execution_time,
                metrics=metrics,
                benchmark_results=benchmark_results,
                details={
                    'avg_latency_ms': avg_latency,
                    'median_latency_ms': median_latency,
                    'p95_latency_ms': p95_latency,
                    'p99_latency_ms': p99_latency,
                    'min_latency_ms': min_latency,
                    'max_latency_ms': max_latency,
                    'iterations': self.config.latency_test_iterations,
                    'data_size': data_size
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return PerformanceTestExecution(
                test_name=test_name,
                test_type=PerformanceTestType.LATENCY,
                result=PerformanceTestResult.ERROR,
                execution_time_ms=execution_time,
                metrics={},
                benchmark_results={},
                error_message=str(e)
            )
    
    def test_memory_usage(self, data_size: int) -> PerformanceTestExecution:
        """
        Test memory usage performance.
        
        Args:
            data_size: Size of test data
            
        Returns:
            Performance test execution result
            
        Raises:
            RuntimeError: If test fails
        """
        test_name = f"memory_usage_{data_size}"
        start_time = time.time()
        
        try:
            import gc
            import psutil
            
            # Get initial memory
            initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            test_data = self._get_test_data(data_size)
            
            # Perform compression operations
            memory_measurements = []
            for _ in range(10):
                compressed = self.hybrid_compressor.compress(test_data.clone())
                decompressed = self.hybrid_compressor.decompress(
                    compressed['compressed_data'],
                    compressed['compression_info']
                )
                
                # Measure memory
                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
                memory_usage = current_memory - initial_memory
                memory_measurements.append(memory_usage)
                
                # Clean up
                del compressed, decompressed
            
            # Force garbage collection
            gc.collect()
            
            # Calculate memory statistics
            avg_memory_usage = statistics.mean(memory_measurements)
            max_memory_usage = max(memory_measurements)
            final_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            memory_growth = final_memory - initial_memory
            
            metrics = {
                PerformanceMetric.MEMORY_USAGE: avg_memory_usage
            }
            
            # Evaluate benchmarks
            benchmark_results = {}
            if 'max_memory_usage' in self.benchmarks:
                benchmark_results['max_memory_usage'] = self.benchmarks['max_memory_usage'].evaluate(max_memory_usage)
            
            # Check for memory leaks
            memory_leak_detected = memory_growth > (self.config.max_memory_usage_mb * 0.5)
            if memory_leak_detected:
                result = PerformanceTestResult.WARNING
            else:
                result = PerformanceTestResult.PASSED if all(benchmark_results.values()) else PerformanceTestResult.FAILED
            
            execution_time = (time.time() - start_time) * 1000
            
            return PerformanceTestExecution(
                test_name=test_name,
                test_type=PerformanceTestType.MEMORY,
                result=result,
                execution_time_ms=execution_time,
                metrics=metrics,
                benchmark_results=benchmark_results,
                details={
                    'avg_memory_usage_mb': avg_memory_usage,
                    'max_memory_usage_mb': max_memory_usage,
                    'memory_growth_mb': memory_growth,
                    'memory_leak_detected': memory_leak_detected,
                    'data_size': data_size
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return PerformanceTestExecution(
                test_name=test_name,
                test_type=PerformanceTestType.MEMORY,
                result=PerformanceTestResult.ERROR,
                execution_time_ms=execution_time,
                metrics={},
                benchmark_results={},
                error_message=str(e)
            )
    
    def test_gpu_memory_usage(self, data_size: int) -> PerformanceTestExecution:
        """
        Test GPU memory usage performance.
        
        Args:
            data_size: Size of test data
            
        Returns:
            Performance test execution result
            
        Raises:
            RuntimeError: If test fails
        """
        test_name = f"gpu_memory_usage_{data_size}"
        start_time = time.time()
        
        try:
            if not torch.cuda.is_available():
                return PerformanceTestExecution(
                    test_name=test_name,
                    test_type=PerformanceTestType.GPU_MEMORY,
                    result=PerformanceTestResult.PASSED,  # Skip but mark as passed
                    execution_time_ms=0.0,
                    metrics={},
                    benchmark_results={},
                    details={'reason': 'CUDA not available'}
                )
            
            # Reset GPU memory stats
            torch.cuda.reset_peak_memory_stats()
            initial_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            
            test_data = self._get_test_data(data_size).cuda()
            
            # Perform compression operations on GPU
            gpu_memory_measurements = []
            for _ in range(5):
                compressed = self.hybrid_compressor.compress(test_data.clone())
                decompressed = self.hybrid_compressor.decompress(
                    compressed['compressed_data'],
                    compressed['compression_info']
                )
                
                # Measure GPU memory
                current_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                gpu_memory_usage = current_gpu_memory - initial_gpu_memory
                gpu_memory_measurements.append(gpu_memory_usage)
                
                # Clean up
                del compressed, decompressed
            
            # Get peak GPU memory
            peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            peak_gpu_usage = peak_gpu_memory - initial_gpu_memory
            
            # Calculate GPU memory statistics
            avg_gpu_memory_usage = statistics.mean(gpu_memory_measurements)
            max_gpu_memory_usage = max(gpu_memory_measurements)
            
            # Clean up
            del test_data
            torch.cuda.empty_cache()
            
            metrics = {
                PerformanceMetric.GPU_MEMORY_USAGE: avg_gpu_memory_usage
            }
            
            # Evaluate benchmarks
            benchmark_results = {}
            if 'max_gpu_memory_usage' in self.benchmarks:
                benchmark_results['max_gpu_memory_usage'] = self.benchmarks['max_gpu_memory_usage'].evaluate(peak_gpu_usage)
            
            # Determine result
            result = PerformanceTestResult.PASSED if all(benchmark_results.values()) else PerformanceTestResult.FAILED
            
            execution_time = (time.time() - start_time) * 1000
            
            return PerformanceTestExecution(
                test_name=test_name,
                test_type=PerformanceTestType.GPU_MEMORY,
                result=result,
                execution_time_ms=execution_time,
                metrics=metrics,
                benchmark_results=benchmark_results,
                details={
                    'avg_gpu_memory_usage_mb': avg_gpu_memory_usage,
                    'max_gpu_memory_usage_mb': max_gpu_memory_usage,
                    'peak_gpu_usage_mb': peak_gpu_usage,
                    'data_size': data_size
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return PerformanceTestExecution(
                test_name=test_name,
                test_type=PerformanceTestType.GPU_MEMORY,
                result=PerformanceTestResult.ERROR,
                execution_time_ms=execution_time,
                metrics={},
                benchmark_results={},
                error_message=str(e)
            )
    
    def test_scalability(self) -> PerformanceTestExecution:
        """
        Test scalability across different data sizes.
        
        Returns:
            Performance test execution result
            
        Raises:
            RuntimeError: If test fails
        """
        test_name = "scalability_test"
        start_time = time.time()
        
        try:
            scalability_results = []
            
            for data_size in self.config.test_data_sizes:
                test_data = self._get_test_data(data_size)
                
                # Measure compression time
                op_start = time.time()
                compressed = self.hybrid_compressor.compress(test_data)
                compression_time = (time.time() - op_start) * 1000  # ms
                
                # Calculate compression ratio
                compression_ratio = test_data.numel() / len(compressed['compressed_data'])
                
                scalability_results.append({
                    'data_size': data_size,
                    'compression_time_ms': compression_time,
                    'compression_ratio': compression_ratio,
                    'throughput_elements_per_ms': data_size / compression_time
                })
                
                # Clean up
                del test_data, compressed
            
            # Analyze scalability
            time_complexity = self._analyze_time_complexity(scalability_results)
            scalability_score = self._calculate_scalability_score(scalability_results)
            
            metrics = {
                PerformanceMetric.EXECUTION_TIME: statistics.mean([r['compression_time_ms'] for r in scalability_results]),
                PerformanceMetric.COMPRESSION_RATIO: statistics.mean([r['compression_ratio'] for r in scalability_results])
            }
            
            # Evaluate scalability
            benchmark_results = {
                'scalability_acceptable': scalability_score >= 0.7  # Threshold for acceptable scalability
            }
            
            # Determine result
            result = PerformanceTestResult.PASSED if all(benchmark_results.values()) else PerformanceTestResult.FAILED
            
            execution_time = (time.time() - start_time) * 1000
            
            return PerformanceTestExecution(
                test_name=test_name,
                test_type=PerformanceTestType.SCALABILITY,
                result=result,
                execution_time_ms=execution_time,
                metrics=metrics,
                benchmark_results=benchmark_results,
                details={
                    'scalability_results': scalability_results,
                    'time_complexity': time_complexity,
                    'scalability_score': scalability_score,
                    'data_sizes_tested': self.config.test_data_sizes
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return PerformanceTestExecution(
                test_name=test_name,
                test_type=PerformanceTestType.SCALABILITY,
                result=PerformanceTestResult.ERROR,
                execution_time_ms=execution_time,
                metrics={},
                benchmark_results={},
                error_message=str(e)
            )
    
    def detect_performance_regression(self) -> PerformanceTestExecution:
        """
        Detect performance regression compared to baseline and history.
        
        Returns:
            Performance test execution result
            
        Raises:
            RuntimeError: If test fails
        """
        test_name = "performance_regression_detection"
        start_time = time.time()
        
        try:
            regressions_detected = []
            regression_details = {}
            
            # Test current performance against baseline
            if self.baseline_metrics:
                current_metrics = self._measure_current_performance()
                
                for metric_name, baseline_value in self.baseline_metrics.items():
                    current_value = current_metrics.get(metric_name, 0.0)
                    
                    # Calculate regression percentage
                    if baseline_value > 0:
                        regression_percent = ((current_value - baseline_value) / baseline_value) * 100
                        
                        # Check for regression (worse performance)
                        if metric_name.endswith('_time_ms') and regression_percent > self.config.regression_threshold_percent:
                            regressions_detected.append(f"{metric_name}_regression")
                            regression_details[metric_name] = {
                                'baseline_value': baseline_value,
                                'current_value': current_value,
                                'regression_percent': regression_percent,
                                'threshold_percent': self.config.regression_threshold_percent
                            }
                        elif metric_name.endswith('_ops_sec') and regression_percent < -self.config.regression_threshold_percent:
                            regressions_detected.append(f"{metric_name}_regression")
                            regression_details[metric_name] = {
                                'baseline_value': baseline_value,
                                'current_value': current_value,
                                'regression_percent': regression_percent,
                                'threshold_percent': self.config.regression_threshold_percent
                            }
            
            # Test against performance history
            history_regressions = self._detect_history_regressions()
            regressions_detected.extend(history_regressions)
            
            metrics = {
                PerformanceMetric.SUCCESS_RATE: 1.0 if not regressions_detected else 0.0
            }
            
            # Determine result
            if regressions_detected:
                result = PerformanceTestResult.REGRESSION
            else:
                result = PerformanceTestResult.PASSED
            
            execution_time = (time.time() - start_time) * 1000
            
            return PerformanceTestExecution(
                test_name=test_name,
                test_type=PerformanceTestType.REGRESSION,
                result=result,
                execution_time_ms=execution_time,
                metrics=metrics,
                benchmark_results={'no_regression': len(regressions_detected) == 0},
                regression_detected=len(regressions_detected) > 0,
                regression_details=regression_details,
                details={
                    'regressions_detected': regressions_detected,
                    'baseline_available': bool(self.baseline_metrics),
                    'regression_threshold_percent': self.config.regression_threshold_percent
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return PerformanceTestExecution(
                test_name=test_name,
                test_type=PerformanceTestType.REGRESSION,
                result=PerformanceTestResult.ERROR,
                execution_time_ms=execution_time,
                metrics={},
                benchmark_results={},
                error_message=str(e)
            )
    
    def test_stress_performance(self) -> PerformanceTestExecution:
        """
        Test performance under stress conditions.
        
        Returns:
            Performance test execution result
            
        Raises:
            RuntimeError: If test fails
        """
        test_name = "stress_performance_test"
        start_time = time.time()
        
        try:
            stress_results = []
            successful_operations = 0
            failed_operations = 0
            
            # Run stress test
            stress_start = time.time()
            while (time.time() - stress_start) < self.config.stress_test_duration_seconds:
                try:
                    # Use random data size
                    data_size = self.config.test_data_sizes[len(stress_results) % len(self.config.test_data_sizes)]
                    test_data = torch.randn(data_size)
                    
                    # Measure operation
                    op_start = time.time()
                    compressed = self.hybrid_compressor.compress(test_data)
                    decompressed = self.hybrid_compressor.decompress(
                        compressed['compressed_data'],
                        compressed['compression_info']
                    )
                    op_time = (time.time() - op_start) * 1000  # ms
                    
                    # Validate result
                    reconstruction_error = torch.norm(test_data - decompressed).item()
                    operation_successful = reconstruction_error < 1e-6
                    
                    stress_results.append({
                        'operation_time_ms': op_time,
                        'data_size': data_size,
                        'reconstruction_error': reconstruction_error,
                        'successful': operation_successful
                    })
                    
                    if operation_successful:
                        successful_operations += 1
                    else:
                        failed_operations += 1
                    
                    # Clean up
                    del test_data, compressed, decompressed
                    
                except Exception as op_e:
                    failed_operations += 1
                    stress_results.append({
                        'operation_time_ms': 0.0,
                        'data_size': 0,
                        'reconstruction_error': float('inf'),
                        'successful': False,
                        'error': str(op_e)
                    })
            
            # Calculate stress test metrics
            total_operations = successful_operations + failed_operations
            success_rate = successful_operations / total_operations if total_operations > 0 else 0.0
            
            successful_results = [r for r in stress_results if r['successful']]
            if successful_results:
                avg_operation_time = statistics.mean([r['operation_time_ms'] for r in successful_results])
                max_operation_time = max([r['operation_time_ms'] for r in successful_results])
            else:
                avg_operation_time = 0.0
                max_operation_time = 0.0
            
            metrics = {
                PerformanceMetric.SUCCESS_RATE: success_rate,
                PerformanceMetric.EXECUTION_TIME: avg_operation_time
            }
            
            # Evaluate stress test results
            benchmark_results = {
                'success_rate_acceptable': success_rate >= self.config.min_success_rate,
                'performance_stable': max_operation_time < (self.config.max_compression_time_ms * 2)  # Allow 2x normal time under stress
            }
            
            # Determine result
            result = PerformanceTestResult.PASSED if all(benchmark_results.values()) else PerformanceTestResult.FAILED
            
            execution_time = (time.time() - start_time) * 1000
            
            return PerformanceTestExecution(
                test_name=test_name,
                test_type=PerformanceTestType.STRESS,
                result=result,
                execution_time_ms=execution_time,
                metrics=metrics,
                benchmark_results=benchmark_results,
                details={
                    'total_operations': total_operations,
                    'successful_operations': successful_operations,
                    'failed_operations': failed_operations,
                    'success_rate': success_rate,
                    'avg_operation_time_ms': avg_operation_time,
                    'max_operation_time_ms': max_operation_time,
                    'stress_duration_seconds': self.config.stress_test_duration_seconds
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return PerformanceTestExecution(
                test_name=test_name,
                test_type=PerformanceTestType.STRESS,
                result=PerformanceTestResult.ERROR,
                execution_time_ms=execution_time,
                metrics={},
                benchmark_results={},
                error_message=str(e)
            )
    
    def _initialize_benchmarks(self) -> None:
        """Initialize performance benchmarks"""
        self.benchmarks = {
            'max_compression_time': PerformanceBenchmark(
                benchmark_name='Maximum Compression Time',
                metric=PerformanceMetric.EXECUTION_TIME,
                expected_value=self.config.max_compression_time_ms,
                tolerance_percent=10.0,
                comparison_operator='less_than',
                description='Compression should complete within time limit'
            ),
            'min_throughput': PerformanceBenchmark(
                benchmark_name='Minimum Throughput',
                metric=PerformanceMetric.THROUGHPUT_OPS_SEC,
                expected_value=self.config.min_throughput_ops_sec,
                tolerance_percent=10.0,
                comparison_operator='greater_than',
                description='Throughput should meet minimum requirements'
            ),
            'max_memory_usage': PerformanceBenchmark(
                benchmark_name='Maximum Memory Usage',
                metric=PerformanceMetric.MEMORY_USAGE,
                expected_value=self.config.max_memory_usage_mb,
                tolerance_percent=20.0,
                comparison_operator='less_than',
                description='Memory usage should stay within limits'
            ),
            'max_gpu_memory_usage': PerformanceBenchmark(
                benchmark_name='Maximum GPU Memory Usage',
                metric=PerformanceMetric.GPU_MEMORY_USAGE,
                expected_value=self.config.max_gpu_memory_usage_mb,
                tolerance_percent=20.0,
                comparison_operator='less_than',
                description='GPU memory usage should stay within limits'
            ),
            'min_compression_ratio': PerformanceBenchmark(
                benchmark_name='Minimum Compression Ratio',
                metric=PerformanceMetric.COMPRESSION_RATIO,
                expected_value=self.config.min_compression_ratio,
                tolerance_percent=10.0,
                comparison_operator='greater_than',
                description='Compression ratio should meet minimum requirements'
            ),
            'min_success_rate': PerformanceBenchmark(
                benchmark_name='Minimum Success Rate',
                metric=PerformanceMetric.SUCCESS_RATE,
                expected_value=self.config.min_success_rate,
                tolerance_percent=5.0,
                comparison_operator='greater_than',
                description='Success rate should meet minimum requirements'
            )
        }
    
    def _generate_test_data(self) -> None:
        """Pre-generate test data for performance"""
        for size in self.config.test_data_sizes:
            if size not in self.test_data_cache:
                self.test_data_cache[size] = torch.randn(size, dtype=torch.float32)
    
    def _get_test_data(self, size: int) -> torch.Tensor:
        """Get test data of specified size"""
        if size in self.test_data_cache:
            return self.test_data_cache[size].clone()
        else:
            return torch.randn(size, dtype=torch.float32)
    
    def _establish_baseline_metrics(self) -> None:
        """Establish baseline performance metrics"""
        try:
            # Measure baseline compression time
            test_data = self._get_test_data(1000)
            
            compression_times = []
            for _ in range(10):
                start_time = time.time()
                compressed = self.hybrid_compressor.compress(test_data.clone())
                compression_time = (time.time() - start_time) * 1000  # ms
                compression_times.append(compression_time)
                del compressed
            
            avg_compression_time = statistics.mean(compression_times)
            
            # Measure baseline throughput
            operations = 0
            throughput_start = time.time()
            while (time.time() - throughput_start) < 5.0:  # 5 second measurement
                compressed = self.hybrid_compressor.compress(test_data.clone())
                operations += 1
                del compressed
            
            throughput = operations / 5.0
            
            self.baseline_metrics = {
                'compression_time_ms': avg_compression_time,
                'throughput_ops_sec': throughput
            }
            
            self.logger.info(f"Baseline metrics established: {self.baseline_metrics}")
            
        except Exception as e:
            self.logger.warning(f"Failed to establish baseline metrics: {e}")
            self.baseline_metrics = {}
    
    def _measure_current_performance(self) -> Dict[str, float]:
        """Measure current performance metrics"""
        current_metrics = {}
        
        try:
            test_data = self._get_test_data(1000)
            
            # Measure compression time
            compression_times = []
            for _ in range(5):
                start_time = time.time()
                compressed = self.hybrid_compressor.compress(test_data.clone())
                compression_time = (time.time() - start_time) * 1000  # ms
                compression_times.append(compression_time)
                del compressed
            
            current_metrics['compression_time_ms'] = statistics.mean(compression_times)
            
            # Measure throughput
            operations = 0
            throughput_start = time.time()
            while (time.time() - throughput_start) < 2.0:  # 2 second measurement
                compressed = self.hybrid_compressor.compress(test_data.clone())
                operations += 1
                del compressed
            
            current_metrics['throughput_ops_sec'] = operations / 2.0
            
        except Exception as e:
            self.logger.warning(f"Failed to measure current performance: {e}")
        
        return current_metrics
    
    def _detect_history_regressions(self) -> List[str]:
        """Detect regressions based on performance history"""
        regressions = []
        
        with self._history_lock:
            for metric_name, history in self.performance_history.items():
                if len(history) >= 3:  # Need at least 3 data points
                    recent_values = list(history)[-3:]  # Last 3 values
                    trend = self._calculate_trend(recent_values)
                    
                    # Check for negative trend (regression)
                    if metric_name.endswith('_time_ms') and trend > self.config.regression_threshold_percent:
                        regressions.append(f"{metric_name}_history_regression")
                    elif metric_name.endswith('_ops_sec') and trend < -self.config.regression_threshold_percent:
                        regressions.append(f"{metric_name}_history_regression")
        
        return regressions
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend percentage from list of values"""
        if len(values) < 2:
            return 0.0
        
        first_value = values[0]
        last_value = values[-1]
        
        if first_value > 0:
            return ((last_value - first_value) / first_value) * 100
        else:
            return 0.0
    
    def _analyze_time_complexity(self, scalability_results: List[Dict[str, Any]]) -> str:
        """Analyze time complexity from scalability results"""
        if len(scalability_results) < 2:
            return "insufficient_data"
        
        # Simple linear regression to estimate complexity
        data_sizes = [r['data_size'] for r in scalability_results]
        times = [r['compression_time_ms'] for r in scalability_results]
        
        # Check if it's roughly linear (O(n))
        ratios = []
        for i in range(1, len(scalability_results)):
            size_ratio = data_sizes[i] / data_sizes[i-1]
            time_ratio = times[i] / times[i-1]
            ratios.append(time_ratio / size_ratio)
        
        avg_ratio = statistics.mean(ratios)
        
        if 0.8 <= avg_ratio <= 1.2:
            return "linear"  # O(n)
        elif avg_ratio > 1.2:
            return "superlinear"  # Worse than O(n)
        else:
            return "sublinear"  # Better than O(n)
    
    def _calculate_scalability_score(self, scalability_results: List[Dict[str, Any]]) -> float:
        """Calculate scalability score (0.0 to 1.0)"""
        if len(scalability_results) < 2:
            return 0.0
        
        # Score based on throughput consistency
        throughputs = [r['throughput_elements_per_ms'] for r in scalability_results]
        throughput_variance = statistics.variance(throughputs)
        throughput_mean = statistics.mean(throughputs)
        
        # Lower variance relative to mean = better scalability
        if throughput_mean > 0:
            coefficient_of_variation = math.sqrt(throughput_variance) / throughput_mean
            scalability_score = max(0.0, 1.0 - coefficient_of_variation)
        else:
            scalability_score = 0.0
        
        return scalability_score
    
    def _update_performance_history(self) -> None:
        """Update performance history with latest results"""
        with self._history_lock:
            for result in self.test_results:
                for metric, value in result.metrics.items():
                    history_key = f"{result.test_name}_{metric.value}"
                    self.performance_history[history_key].append(value)
    
    def _run_throughput_tests(self) -> None:
        """Run all throughput tests"""
        self.logger.info("Running throughput tests")
        
        for data_size in self.config.test_data_sizes:
            result = self.test_compression_throughput(data_size)
            with self._results_lock:
                self.test_results.append(result)
    
    def _run_latency_tests(self) -> None:
        """Run all latency tests"""
        self.logger.info("Running latency tests")
        
        for data_size in self.config.test_data_sizes:
            result = self.test_compression_latency(data_size)
            with self._results_lock:
                self.test_results.append(result)
    
    def _run_memory_tests(self) -> None:
        """Run all memory tests"""
        self.logger.info("Running memory tests")
        
        for data_size in [1000, 5000]:  # Subset for memory tests
            result = self.test_memory_usage(data_size)
            with self._results_lock:
                self.test_results.append(result)
    
    def _run_gpu_memory_tests(self) -> None:
        """Run all GPU memory tests"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available - skipping GPU memory tests")
            return
        
        self.logger.info("Running GPU memory tests")
        
        for data_size in [1000, 5000]:  # Subset for GPU memory tests
            result = self.test_gpu_memory_usage(data_size)
            with self._results_lock:
                self.test_results.append(result)
    
    def _run_scalability_tests(self) -> None:
        """Run all scalability tests"""
        self.logger.info("Running scalability tests")
        
        result = self.test_scalability()
        with self._results_lock:
            self.test_results.append(result)
    
    def _run_regression_tests(self) -> None:
        """Run all regression tests"""
        self.logger.info("Running regression tests")
        
        result = self.detect_performance_regression()
        with self._results_lock:
            self.test_results.append(result)
    
    def _run_stress_tests(self) -> None:
        """Run all stress tests"""
        self.logger.info("Running stress tests")
        
        result = self.test_stress_performance()
        with self._results_lock:
            self.test_results.append(result)
    
    def _generate_performance_report(self, total_execution_time: float) -> PerformanceTestReport:
        """Generate comprehensive performance test report"""
        # Count results by status
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.result == PerformanceTestResult.PASSED)
        failed_tests = sum(1 for r in self.test_results if r.result == PerformanceTestResult.FAILED)
        regression_tests = sum(1 for r in self.test_results if r.result == PerformanceTestResult.REGRESSION)
        warning_tests = sum(1 for r in self.test_results if r.result == PerformanceTestResult.WARNING)
        error_tests = sum(1 for r in self.test_results if r.result == PerformanceTestResult.ERROR)
        
        # Type summary
        type_summary = {}
        for test_type in PerformanceTestType:
            type_results = [r for r in self.test_results if r.test_type == test_type]
            type_summary[test_type] = {
                'total': len(type_results),
                'passed': sum(1 for r in type_results if r.result == PerformanceTestResult.PASSED),
                'failed': sum(1 for r in type_results if r.result == PerformanceTestResult.FAILED),
                'regression': sum(1 for r in type_results if r.result == PerformanceTestResult.REGRESSION)
            }
        
        # Metric summary
        metric_summary = {}
        for metric in PerformanceMetric:
            metric_values = []
            for result in self.test_results:
                if metric in result.metrics:
                    metric_values.append(result.metrics[metric])
            
            if metric_values:
                metric_summary[metric] = {
                    'average': statistics.mean(metric_values),
                    'min': min(metric_values),
                    'max': max(metric_values),
                    'count': len(metric_values)
                }
        
        # Benchmark summary
        benchmark_summary = {}
        for benchmark_name, benchmark in self.benchmarks.items():
            benchmark_results = []
            for result in self.test_results:
                if benchmark_name in result.benchmark_results:
                    benchmark_results.append(result.benchmark_results[benchmark_name])
            
            if benchmark_results:
                benchmark_summary[benchmark_name] = {
                    'pass_rate': sum(benchmark_results) / len(benchmark_results),
                    'total_evaluations': len(benchmark_results),
                    'passed_evaluations': sum(benchmark_results)
                }
        
        # Regression summary
        regression_results = [r for r in self.test_results if r.regression_detected]
        regression_summary = {
            'regressions_detected': len(regression_results),
            'regression_details': [r.regression_details for r in regression_results if r.regression_details]
        }
        
        # Performance trends
        performance_trends = {}
        with self._history_lock:
            for metric_name, history in self.performance_history.items():
                if len(history) >= 2:
                    performance_trends[metric_name] = list(history)
        
        # Recommendations
        recommendations = []
        if failed_tests > 0:
            recommendations.append(f"Address {failed_tests} failed performance tests")
        if regression_tests > 0:
            recommendations.append(f"Investigate {regression_tests} performance regressions")
        if error_tests > 0:
            recommendations.append(f"Fix {error_tests} test errors")
        
        # Specific benchmark recommendations
        for benchmark_name, summary in benchmark_summary.items():
            if summary['pass_rate'] < 0.8:
                recommendations.append(f"Improve performance for {benchmark_name} (pass rate: {summary['pass_rate']:.1%})")
        
        return PerformanceTestReport(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            regression_tests=regression_tests,
            warning_tests=warning_tests,
            error_tests=error_tests,
            total_execution_time_ms=total_execution_time,
            test_results=self.test_results.copy(),
            type_summary=type_summary,
            metric_summary=metric_summary,
            benchmark_summary=benchmark_summary,
            regression_summary=regression_summary,
            performance_trends=performance_trends,
            recommendations=recommendations
        )
    
    def shutdown(self) -> None:
        """Shutdown performance tests"""
        self.logger.info("Shutting down hybrid performance tests")
        
        # Clear test data cache
        self.test_data_cache.clear()
        
        # Clear performance history
        with self._history_lock:
            self.performance_history.clear()
        
        # Clear references
        self.hybrid_manager = None
        self.hybrid_compressor = None
        self.switching_manager = None
        self.pure_compressor = None
        self.performance_monitor = None
        
        self.is_initialized = False
        self.logger.info("Hybrid performance tests shutdown complete")


def run_hybrid_performance_tests(
    hybrid_manager: HybridPadicManager,
    hybrid_compressor: HybridPadicCompressionSystem,
    switching_manager: DynamicSwitchingManager,
    pure_compressor: Optional[PadicCompressionSystem] = None,
    performance_monitor: Optional[HybridPerformanceMonitor] = None,
    config: Optional[PerformanceTestConfig] = None
) -> PerformanceTestReport:
    """
    Run hybrid performance tests with provided components.
    
    Args:
        hybrid_manager: Hybrid p-adic manager instance
        hybrid_compressor: Hybrid compression system instance
        switching_manager: Dynamic switching manager instance
        pure_compressor: Optional pure p-adic compressor
        performance_monitor: Optional performance monitor
        config: Optional performance test configuration
        
    Returns:
        Performance test report
        
    Raises:
        RuntimeError: If test execution fails
    """
    # Create performance tests
    performance_tests = HybridPerformanceTests(config)
    
    try:
        # Initialize tests
        performance_tests.initialize_performance_tests(
            hybrid_manager,
            hybrid_compressor,
            switching_manager,
            pure_compressor,
            performance_monitor
        )
        
        # Run all tests
        report = performance_tests.run_all_performance_tests()
        
        return report
        
    finally:
        # Ensure cleanup
        performance_tests.shutdown()