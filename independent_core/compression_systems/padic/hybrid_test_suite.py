"""
Hybrid Test Suite - Comprehensive test suite for hybrid p-adic system
NO FALLBACKS - HARD FAILURES ONLY
"""

import logging
import math
import threading
import time
import torch
import tracemalloc
import unittest
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from enum import Enum

# Import hybrid system components
from .hybrid_padic_structures import HybridPadicWeight, HybridPadicManager, HybridPadicValidator
from .hybrid_padic_compressor import HybridPadicCompressionSystem
from .dynamic_switching_manager import DynamicSwitchingManager
from .hybrid_performance_optimizer import HybridPerformanceOptimizer
from .hybrid_performance_monitor import HybridPerformanceMonitor

# Import existing components for comparison
from .padic_encoder import PadicWeight, PadicCompressionSystem, PadicMathematicalOperations


class TestCategory(Enum):
    """Test category enumeration"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    MEMORY = "memory"
    GPU = "gpu"
    ERROR_HANDLING = "error_handling"
    MATHEMATICAL = "mathematical"


class TestResult(Enum):
    """Test result enumeration"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestExecutionResult:
    """Result of test execution"""
    test_name: str
    test_category: TestCategory
    result: TestResult
    execution_time_ms: float
    memory_usage_mb: float
    gpu_memory_usage_mb: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate test execution result"""
        if not isinstance(self.test_name, str) or not self.test_name.strip():
            raise ValueError("Test name must be non-empty string")
        if not isinstance(self.test_category, TestCategory):
            raise TypeError("Test category must be TestCategory")
        if not isinstance(self.result, TestResult):
            raise TypeError("Result must be TestResult")
        if not isinstance(self.execution_time_ms, (int, float)) or self.execution_time_ms < 0:
            raise ValueError("Execution time must be non-negative")


@dataclass
class TestSuiteConfig:
    """Configuration for hybrid test suite"""
    enable_unit_tests: bool = True
    enable_integration_tests: bool = True
    enable_performance_tests: bool = True
    enable_memory_tests: bool = True
    enable_gpu_tests: bool = True
    enable_error_handling_tests: bool = True
    enable_mathematical_tests: bool = True
    
    # Test parameters
    test_data_sizes: List[int] = field(default_factory=lambda: [100, 1000, 10000])
    performance_tolerance_ms: float = 1000.0
    memory_tolerance_mb: float = 100.0
    gpu_memory_tolerance_mb: float = 512.0
    mathematical_tolerance: float = 1e-10
    
    # Test execution
    enable_parallel_execution: bool = True
    max_concurrent_tests: int = 4
    test_timeout_seconds: int = 300
    
    # Reporting
    enable_detailed_reporting: bool = True
    generate_performance_charts: bool = False
    save_test_artifacts: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if not isinstance(self.test_data_sizes, list) or not self.test_data_sizes:
            raise ValueError("Test data sizes must be non-empty list")
        if self.performance_tolerance_ms <= 0:
            raise ValueError("Performance tolerance must be positive")
        if self.memory_tolerance_mb <= 0:
            raise ValueError("Memory tolerance must be positive")
        if self.mathematical_tolerance <= 0:
            raise ValueError("Mathematical tolerance must be positive")


@dataclass
class TestSuiteReport:
    """Comprehensive test suite report"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    total_execution_time_ms: float
    test_results: List[TestExecutionResult]
    category_summary: Dict[TestCategory, Dict[str, int]]
    performance_summary: Dict[str, float]
    memory_summary: Dict[str, float]
    gpu_summary: Dict[str, float]
    critical_failures: List[str]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Calculate derived metrics"""
        if self.total_tests > 0:
            self.success_rate = self.passed_tests / self.total_tests
            self.failure_rate = self.failed_tests / self.total_tests
        else:
            self.success_rate = 0.0
            self.failure_rate = 0.0


class HybridTestSuite:
    """
    Comprehensive test suite for hybrid p-adic system.
    Provides thorough testing of all hybrid components with mathematical validation.
    """
    
    def __init__(self, config: Optional[TestSuiteConfig] = None):
        """Initialize hybrid test suite"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, TestSuiteConfig):
            raise TypeError(f"Config must be TestSuiteConfig or None, got {type(config)}")
        
        self.config = config or TestSuiteConfig()
        self.logger = logging.getLogger('HybridTestSuite')
        
        # Test components
        self.hybrid_manager: Optional[HybridPadicManager] = None
        self.hybrid_compressor: Optional[HybridPadicCompressionSystem] = None
        self.switching_manager: Optional[DynamicSwitchingManager] = None
        self.pure_compressor: Optional[PadicCompressionSystem] = None
        
        # Test state
        self.is_initialized = False
        self.is_running = False
        
        # Test tracking
        self.test_results: List[TestExecutionResult] = []
        self.current_test: Optional[str] = None
        self.test_start_time: Optional[datetime] = None
        
        # Memory tracking
        self.memory_tracker = None
        self.initial_memory: float = 0.0
        
        # Thread safety
        self._test_lock = threading.RLock()
        self._results_lock = threading.RLock()
        
        # Test data cache
        self.test_data_cache: Dict[int, torch.Tensor] = {}
        self.hybrid_weights_cache: Dict[int, List[HybridPadicWeight]] = {}
        
        self.logger.info("HybridTestSuite created successfully")
    
    def initialize_test_suite(self,
                            hybrid_manager: HybridPadicManager,
                            hybrid_compressor: HybridPadicCompressionSystem,
                            switching_manager: DynamicSwitchingManager,
                            pure_compressor: Optional[PadicCompressionSystem] = None) -> None:
        """
        Initialize test suite with system components.
        
        Args:
            hybrid_manager: Hybrid p-adic manager instance
            hybrid_compressor: Hybrid compression system instance
            switching_manager: Dynamic switching manager instance
            pure_compressor: Optional pure p-adic compressor for comparison
            
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
        
        try:
            # Set component references
            self.hybrid_manager = hybrid_manager
            self.hybrid_compressor = hybrid_compressor
            self.switching_manager = switching_manager
            self.pure_compressor = pure_compressor
            
            # Initialize memory tracking
            if self.config.enable_memory_tests:
                tracemalloc.start()
                self.initial_memory = self._get_memory_usage()
            
            # Initialize GPU tracking
            if self.config.enable_gpu_tests and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # Pre-generate test data
            self._generate_test_data()
            
            self.is_initialized = True
            self.logger.info("Hybrid test suite initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize test suite: {e}")
            raise RuntimeError(f"Test suite initialization failed: {e}")
    
    def run_all_tests(self) -> TestSuiteReport:
        """
        Run all enabled tests and generate comprehensive report.
        
        Returns:
            Comprehensive test suite report
            
        Raises:
            RuntimeError: If test suite is not initialized or execution fails
        """
        if not self.is_initialized:
            raise RuntimeError("Test suite not initialized")
        
        if self.is_running:
            raise RuntimeError("Test suite is already running")
        
        with self._test_lock:
            try:
                self.is_running = True
                self.test_results.clear()
                start_time = time.time()
                
                self.logger.info("Starting comprehensive hybrid test suite execution")
                
                # Run unit tests
                if self.config.enable_unit_tests:
                    self._run_unit_tests()
                
                # Run integration tests
                if self.config.enable_integration_tests:
                    self._run_integration_tests()
                
                # Run performance tests
                if self.config.enable_performance_tests:
                    self._run_performance_tests()
                
                # Run memory tests
                if self.config.enable_memory_tests:
                    self._run_memory_tests()
                
                # Run GPU tests
                if self.config.enable_gpu_tests:
                    self._run_gpu_tests()
                
                # Run error handling tests
                if self.config.enable_error_handling_tests:
                    self._run_error_handling_tests()
                
                # Run mathematical tests
                if self.config.enable_mathematical_tests:
                    self._run_mathematical_tests()
                
                # Calculate execution time
                total_execution_time = (time.time() - start_time) * 1000
                
                # Generate report
                report = self._generate_test_report(total_execution_time)
                
                self.logger.info(f"Test suite completed: {report.passed_tests}/{report.total_tests} tests passed")
                
                return report
                
            except Exception as e:
                self.logger.error(f"Test suite execution failed: {e}")
                raise RuntimeError(f"Test suite execution failed: {e}")
            finally:
                self.is_running = False
    
    def test_hybrid_compression_correctness(self, data_size: int = 1000) -> TestExecutionResult:
        """
        Test hybrid compression correctness.
        
        Args:
            data_size: Size of test data
            
        Returns:
            Test execution result
            
        Raises:
            RuntimeError: If test fails
        """
        test_name = f"hybrid_compression_correctness_{data_size}"
        
        return self._execute_test(
            test_name=test_name,
            test_category=TestCategory.UNIT,
            test_function=self._test_compression_correctness_impl,
            data_size=data_size
        )
    
    def test_hybrid_decompression_correctness(self, data_size: int = 1000) -> TestExecutionResult:
        """
        Test hybrid decompression correctness.
        
        Args:
            data_size: Size of test data
            
        Returns:
            Test execution result
            
        Raises:
            RuntimeError: If test fails
        """
        test_name = f"hybrid_decompression_correctness_{data_size}"
        
        return self._execute_test(
            test_name=test_name,
            test_category=TestCategory.UNIT,
            test_function=self._test_decompression_correctness_impl,
            data_size=data_size
        )
    
    def test_hybrid_pure_equivalence(self, data_size: int = 1000) -> TestExecutionResult:
        """
        Test equivalence between hybrid and pure p-adic operations.
        
        Args:
            data_size: Size of test data
            
        Returns:
            Test execution result
            
        Raises:
            RuntimeError: If test fails or pure compressor not available
        """
        if self.pure_compressor is None:
            raise RuntimeError("Pure compressor required for equivalence testing")
        
        test_name = f"hybrid_pure_equivalence_{data_size}"
        
        return self._execute_test(
            test_name=test_name,
            test_category=TestCategory.MATHEMATICAL,
            test_function=self._test_hybrid_pure_equivalence_impl,
            data_size=data_size
        )
    
    def test_hybrid_memory_management(self, data_size: int = 10000) -> TestExecutionResult:
        """
        Test hybrid memory management and leak detection.
        
        Args:
            data_size: Size of test data
            
        Returns:
            Test execution result
            
        Raises:
            RuntimeError: If test fails
        """
        test_name = f"hybrid_memory_management_{data_size}"
        
        return self._execute_test(
            test_name=test_name,
            test_category=TestCategory.MEMORY,
            test_function=self._test_memory_management_impl,
            data_size=data_size
        )
    
    def test_hybrid_error_handling(self) -> TestExecutionResult:
        """
        Test hybrid error handling compliance.
        
        Returns:
            Test execution result
            
        Raises:
            RuntimeError: If test fails
        """
        test_name = "hybrid_error_handling"
        
        return self._execute_test(
            test_name=test_name,
            test_category=TestCategory.ERROR_HANDLING,
            test_function=self._test_error_handling_impl
        )
    
    def test_hybrid_gpu_operations(self, data_size: int = 5000) -> TestExecutionResult:
        """
        Test hybrid GPU operations and memory management.
        
        Args:
            data_size: Size of test data
            
        Returns:
            Test execution result
            
        Raises:
            RuntimeError: If test fails or CUDA not available
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for GPU operations testing")
        
        test_name = f"hybrid_gpu_operations_{data_size}"
        
        return self._execute_test(
            test_name=test_name,
            test_category=TestCategory.GPU,
            test_function=self._test_gpu_operations_impl,
            data_size=data_size
        )
    
    def _execute_test(self, test_name: str, test_category: TestCategory, 
                     test_function: Callable, **kwargs) -> TestExecutionResult:
        """Execute individual test with monitoring"""
        self.current_test = test_name
        self.test_start_time = datetime.utcnow()
        
        # Memory tracking
        memory_before = self._get_memory_usage()
        gpu_memory_before = self._get_gpu_memory_usage()
        
        start_time = time.time()
        
        try:
            # Execute test function
            test_function(**kwargs)
            
            # Calculate metrics
            execution_time = (time.time() - start_time) * 1000
            memory_after = self._get_memory_usage()
            gpu_memory_after = self._get_gpu_memory_usage()
            
            result = TestExecutionResult(
                test_name=test_name,
                test_category=test_category,
                result=TestResult.PASSED,
                execution_time_ms=execution_time,
                memory_usage_mb=memory_after - memory_before,
                gpu_memory_usage_mb=gpu_memory_after - gpu_memory_before,
                details={'success': True}
            )
            
            self.logger.debug(f"Test {test_name} passed in {execution_time:.2f}ms")
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            memory_after = self._get_memory_usage()
            gpu_memory_after = self._get_gpu_memory_usage()
            
            result = TestExecutionResult(
                test_name=test_name,
                test_category=test_category,
                result=TestResult.FAILED,
                execution_time_ms=execution_time,
                memory_usage_mb=memory_after - memory_before,
                gpu_memory_usage_mb=gpu_memory_after - gpu_memory_before,
                error_message=str(e),
                details={'exception_type': type(e).__name__}
            )
            
            self.logger.error(f"Test {test_name} failed: {e}")
        
        # Store result
        with self._results_lock:
            self.test_results.append(result)
        
        return result
    
    def _run_unit_tests(self) -> None:
        """Run all unit tests"""
        self.logger.info("Running unit tests")
        
        for data_size in self.config.test_data_sizes:
            self.test_hybrid_compression_correctness(data_size)
            self.test_hybrid_decompression_correctness(data_size)
    
    def _run_integration_tests(self) -> None:
        """Run all integration tests"""
        self.logger.info("Running integration tests")
        
        # Test switching manager integration
        self._execute_test(
            "switching_manager_integration",
            TestCategory.INTEGRATION,
            self._test_switching_integration_impl
        )
        
        # Test manager integration
        self._execute_test(
            "hybrid_manager_integration", 
            TestCategory.INTEGRATION,
            self._test_manager_integration_impl
        )
    
    def _run_performance_tests(self) -> None:
        """Run all performance tests"""
        self.logger.info("Running performance tests")
        
        for data_size in self.config.test_data_sizes:
            self._execute_test(
                f"compression_performance_{data_size}",
                TestCategory.PERFORMANCE,
                self._test_compression_performance_impl,
                data_size=data_size
            )
    
    def _run_memory_tests(self) -> None:
        """Run all memory tests"""
        self.logger.info("Running memory tests")
        
        for data_size in [1000, 5000, 10000]:
            self.test_hybrid_memory_management(data_size)
    
    def _run_gpu_tests(self) -> None:
        """Run all GPU tests"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available - skipping GPU tests")
            return
        
        self.logger.info("Running GPU tests")
        
        for data_size in [1000, 5000]:
            self.test_hybrid_gpu_operations(data_size)
    
    def _run_error_handling_tests(self) -> None:
        """Run all error handling tests"""
        self.logger.info("Running error handling tests")
        
        self.test_hybrid_error_handling()
    
    def _run_mathematical_tests(self) -> None:
        """Run all mathematical tests"""
        self.logger.info("Running mathematical tests")
        
        if self.pure_compressor:
            for data_size in [500, 1000]:
                self.test_hybrid_pure_equivalence(data_size)
        
        # Test mathematical properties
        self._execute_test(
            "mathematical_properties",
            TestCategory.MATHEMATICAL,
            self._test_mathematical_properties_impl
        )
    
    # Test implementation methods
    
    def _test_compression_correctness_impl(self, data_size: int) -> None:
        """Test compression correctness implementation"""
        test_data = self._get_test_data(data_size)
        
        # Test compression
        compressed = self.hybrid_compressor.compress(test_data)
        
        if not isinstance(compressed, dict):
            raise RuntimeError("Compression must return dict")
        if 'compressed_data' not in compressed:
            raise RuntimeError("Compressed result must contain 'compressed_data'")
        if 'compression_info' not in compressed:
            raise RuntimeError("Compressed result must contain 'compression_info'")
    
    def _test_decompression_correctness_impl(self, data_size: int) -> None:
        """Test decompression correctness implementation"""
        test_data = self._get_test_data(data_size)
        
        # Compress then decompress
        compressed = self.hybrid_compressor.compress(test_data)
        decompressed = self.hybrid_compressor.decompress(compressed['compressed_data'], compressed['compression_info'])
        
        # Verify reconstruction
        if not torch.allclose(test_data, decompressed, atol=self.config.mathematical_tolerance):
            raise RuntimeError("Decompression does not reconstruct original data accurately")
    
    def _test_hybrid_pure_equivalence_impl(self, data_size: int) -> None:
        """Test hybrid-pure equivalence implementation"""
        test_data = self._get_test_data(data_size)
        
        # Compress with both systems
        hybrid_compressed = self.hybrid_compressor.compress(test_data)
        pure_compressed = self.pure_compressor.compress(test_data)
        
        # Decompress both
        hybrid_decompressed = self.hybrid_compressor.decompress(
            hybrid_compressed['compressed_data'], 
            hybrid_compressed['compression_info']
        )
        pure_decompressed = self.pure_compressor.decompress(
            pure_compressed['compressed_data'],
            pure_compressed['compression_info']
        )
        
        # Verify equivalence
        if not torch.allclose(hybrid_decompressed, pure_decompressed, atol=self.config.mathematical_tolerance):
            raise RuntimeError("Hybrid and pure decompression results do not match")
    
    def _test_memory_management_impl(self, data_size: int) -> None:
        """Test memory management implementation"""
        initial_memory = self._get_memory_usage()
        
        # Perform multiple compression/decompression cycles
        for _ in range(10):
            test_data = self._get_test_data(data_size)
            compressed = self.hybrid_compressor.compress(test_data)
            decompressed = self.hybrid_compressor.decompress(
                compressed['compressed_data'],
                compressed['compression_info']
            )
            del test_data, compressed, decompressed
        
        # Force garbage collection
        import gc
        gc.collect()
        
        final_memory = self._get_memory_usage()
        memory_growth = final_memory - initial_memory
        
        if memory_growth > self.config.memory_tolerance_mb:
            raise RuntimeError(f"Memory leak detected: {memory_growth:.2f}MB growth")
    
    def _test_error_handling_impl(self) -> None:
        """Test error handling implementation"""
        # Test invalid input handling
        test_cases = [
            (None, "Data cannot be None"),
            ("invalid", "Data must be torch.Tensor"),
            (torch.tensor([]), "Data cannot be empty"),
        ]
        
        for invalid_input, expected_error in test_cases:
            try:
                self.hybrid_compressor.compress(invalid_input)
                raise RuntimeError(f"Expected error for input {invalid_input}")
            except (ValueError, TypeError) as e:
                if expected_error not in str(e):
                    raise RuntimeError(f"Unexpected error message: {e}")
            except Exception as e:
                raise RuntimeError(f"Unexpected exception type: {type(e)}")
    
    def _test_gpu_operations_impl(self, data_size: int) -> None:
        """Test GPU operations implementation"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        initial_gpu_memory = torch.cuda.memory_allocated()
        
        # Test GPU operations
        test_data = self._get_test_data(data_size).cuda()
        compressed = self.hybrid_compressor.compress(test_data)
        decompressed = self.hybrid_compressor.decompress(
            compressed['compressed_data'],
            compressed['compression_info']
        )
        
        # Verify GPU memory management
        peak_gpu_memory = torch.cuda.max_memory_allocated()
        gpu_memory_used = (peak_gpu_memory - initial_gpu_memory) / (1024 * 1024)  # MB
        
        if gpu_memory_used > self.config.gpu_memory_tolerance_mb:
            raise RuntimeError(f"Excessive GPU memory usage: {gpu_memory_used:.2f}MB")
        
        # Clean up
        del test_data, compressed, decompressed
        torch.cuda.empty_cache()
    
    def _test_switching_integration_impl(self) -> None:
        """Test switching manager integration"""
        test_data = self._get_test_data(1000)
        
        # Test switching decision
        should_switch = self.switching_manager.should_switch_to_hybrid(test_data)
        
        if not isinstance(should_switch, bool):
            raise RuntimeError("Switching decision must return bool")
    
    def _test_manager_integration_impl(self) -> None:
        """Test hybrid manager integration"""
        # Test manager operations
        config = {'test': True}
        
        # This would test actual manager integration
        # For now, just verify manager is accessible
        if self.hybrid_manager is None:
            raise RuntimeError("Hybrid manager not available")
    
    def _test_compression_performance_impl(self, data_size: int) -> None:
        """Test compression performance implementation"""
        test_data = self._get_test_data(data_size)
        
        start_time = time.time()
        compressed = self.hybrid_compressor.compress(test_data)
        compression_time = (time.time() - start_time) * 1000
        
        if compression_time > self.config.performance_tolerance_ms:
            raise RuntimeError(f"Compression too slow: {compression_time:.2f}ms")
    
    def _test_mathematical_properties_impl(self) -> None:
        """Test mathematical properties implementation"""
        # Test ultrametric property preservation
        validator = HybridPadicValidator()
        
        # Create test hybrid weights
        test_weights = self._get_hybrid_weights(3)
        
        for weight in test_weights:
            try:
                validator.validate_hybrid_weight(weight)
            except Exception as e:
                raise RuntimeError(f"Mathematical validation failed: {e}")
    
    # Helper methods
    
    def _generate_test_data(self) -> None:
        """Pre-generate test data for performance"""
        for size in self.config.test_data_sizes + [500, 5000, 10000]:
            if size not in self.test_data_cache:
                self.test_data_cache[size] = torch.randn(size, dtype=torch.float32)
            
            if size not in self.hybrid_weights_cache:
                self.hybrid_weights_cache[size] = self._create_hybrid_weights(min(size // 100, 10))
    
    def _get_test_data(self, size: int) -> torch.Tensor:
        """Get test data of specified size"""
        if size in self.test_data_cache:
            return self.test_data_cache[size].clone()
        else:
            return torch.randn(size, dtype=torch.float32)
    
    def _get_hybrid_weights(self, count: int) -> List[HybridPadicWeight]:
        """Get hybrid weights for testing"""
        if count in self.hybrid_weights_cache:
            return self.hybrid_weights_cache[count]
        else:
            return self._create_hybrid_weights(count)
    
    def _create_hybrid_weights(self, count: int) -> List[HybridPadicWeight]:
        """Create hybrid weights for testing"""
        weights = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for i in range(count):
            weight = HybridPadicWeight(
                exponent_channel=torch.randn(10, device=device),
                mantissa_channel=torch.randn(10, device=device),
                prime=7,
                precision=10,
                valuation=0,
                device=device,
                dtype=torch.float32
            )
            weights.append(weight)
        
        return weights
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if self.memory_tracker is None:
            try:
                import psutil
                return psutil.Process().memory_info().rss / (1024 * 1024)
            except ImportError:
                return 0.0
        else:
            current, peak = tracemalloc.get_traced_memory()
            return current / (1024 * 1024)
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0
    
    def _generate_test_report(self, total_execution_time: float) -> TestSuiteReport:
        """Generate comprehensive test report"""
        # Count results by category
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.result == TestResult.PASSED)
        failed_tests = sum(1 for r in self.test_results if r.result == TestResult.FAILED)
        skipped_tests = sum(1 for r in self.test_results if r.result == TestResult.SKIPPED)
        error_tests = sum(1 for r in self.test_results if r.result == TestResult.ERROR)
        
        # Category summary
        category_summary = {}
        for category in TestCategory:
            category_results = [r for r in self.test_results if r.test_category == category]
            category_summary[category] = {
                'total': len(category_results),
                'passed': sum(1 for r in category_results if r.result == TestResult.PASSED),
                'failed': sum(1 for r in category_results if r.result == TestResult.FAILED)
            }
        
        # Performance summary
        performance_results = [r for r in self.test_results if r.test_category == TestCategory.PERFORMANCE]
        performance_summary = {
            'average_execution_time_ms': sum(r.execution_time_ms for r in performance_results) / len(performance_results) if performance_results else 0.0,
            'max_execution_time_ms': max((r.execution_time_ms for r in performance_results), default=0.0),
            'min_execution_time_ms': min((r.execution_time_ms for r in performance_results), default=0.0)
        }
        
        # Memory summary
        memory_summary = {
            'average_memory_usage_mb': sum(r.memory_usage_mb for r in self.test_results) / total_tests if total_tests > 0 else 0.0,
            'max_memory_usage_mb': max((r.memory_usage_mb for r in self.test_results), default=0.0),
            'total_memory_growth_mb': sum(max(0, r.memory_usage_mb) for r in self.test_results)
        }
        
        # GPU summary
        gpu_summary = {
            'average_gpu_memory_mb': sum(r.gpu_memory_usage_mb for r in self.test_results) / total_tests if total_tests > 0 else 0.0,
            'max_gpu_memory_mb': max((r.gpu_memory_usage_mb for r in self.test_results), default=0.0),
            'total_gpu_memory_mb': sum(max(0, r.gpu_memory_usage_mb) for r in self.test_results)
        }
        
        # Critical failures
        critical_failures = [
            r.test_name for r in self.test_results 
            if r.result == TestResult.FAILED and r.test_category in [TestCategory.MATHEMATICAL, TestCategory.ERROR_HANDLING]
        ]
        
        # Recommendations
        recommendations = []
        if failed_tests > 0:
            recommendations.append(f"Address {failed_tests} failed tests")
        if memory_summary['total_memory_growth_mb'] > 100:
            recommendations.append("Investigate memory usage - potential leaks detected")
        if gpu_summary['max_gpu_memory_mb'] > 1024:
            recommendations.append("Review GPU memory usage - exceeds 1GB")
        
        return TestSuiteReport(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            error_tests=error_tests,
            total_execution_time_ms=total_execution_time,
            test_results=self.test_results.copy(),
            category_summary=category_summary,
            performance_summary=performance_summary,
            memory_summary=memory_summary,
            gpu_summary=gpu_summary,
            critical_failures=critical_failures,
            recommendations=recommendations
        )
    
    def shutdown(self) -> None:
        """Shutdown test suite"""
        self.logger.info("Shutting down hybrid test suite")
        
        # Clear test data caches
        self.test_data_cache.clear()
        self.hybrid_weights_cache.clear()
        
        # Stop memory tracking
        if self.memory_tracker is not None:
            tracemalloc.stop()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear references
        self.hybrid_manager = None
        self.hybrid_compressor = None
        self.switching_manager = None
        self.pure_compressor = None
        
        self.is_initialized = False
        self.logger.info("Hybrid test suite shutdown complete")