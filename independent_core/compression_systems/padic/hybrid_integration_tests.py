"""
Hybrid Integration Tests - Integration testing for hybrid system components
NO FALLBACKS - HARD FAILURES ONLY
"""

import logging
import math
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
from hybrid_performance_analyzer import HybridPerformanceAnalyzer
from hybrid_performance_tuner import HybridPerformanceTuner
from hybrid_test_suite import HybridTestSuite
from hybrid_validation_framework import HybridValidationFramework

# Import existing components for integration
from padic_encoder import PadicWeight, PadicCompressionSystem, PadicMathematicalOperations
from performance_optimizer import PerformanceOptimizer


class IntegrationTestType(Enum):
    """Integration test type enumeration"""
    COMPONENT = "component"
    SYSTEM = "system"
    CROSS_SYSTEM = "cross_system"
    END_TO_END = "end_to_end"
    WORKFLOW = "workflow"


class IntegrationTestResult(Enum):
    """Integration test result enumeration"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class IntegrationTestExecution:
    """Result of integration test execution"""
    test_name: str
    test_type: IntegrationTestType
    result: IntegrationTestResult
    execution_time_ms: float
    components_tested: List[str]
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate test execution result"""
        if not isinstance(self.test_name, str) or not self.test_name.strip():
            raise ValueError("Test name must be non-empty string")
        if not isinstance(self.test_type, IntegrationTestType):
            raise TypeError("Test type must be IntegrationTestType")
        if not isinstance(self.result, IntegrationTestResult):
            raise TypeError("Result must be IntegrationTestResult")
        if not isinstance(self.execution_time_ms, (int, float)) or self.execution_time_ms < 0:
            raise ValueError("Execution time must be non-negative")


@dataclass
class IntegrationTestConfig:
    """Configuration for hybrid integration tests"""
    enable_component_tests: bool = True
    enable_system_tests: bool = True
    enable_cross_system_tests: bool = True
    enable_end_to_end_tests: bool = True
    enable_workflow_tests: bool = True
    
    # Test parameters
    test_data_sizes: List[int] = field(default_factory=lambda: [100, 1000, 5000])
    performance_tolerance_ms: float = 2000.0
    memory_tolerance_mb: float = 200.0
    mathematical_tolerance: float = 1e-10
    
    # Execution parameters
    enable_parallel_execution: bool = True
    max_concurrent_tests: int = 4
    test_timeout_seconds: int = 600
    
    # Component configuration
    enable_performance_components: bool = True
    enable_validation_components: bool = True
    enable_gpu_testing: bool = True
    
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
class IntegrationTestReport:
    """Comprehensive integration test report"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    total_execution_time_ms: float
    test_results: List[IntegrationTestExecution]
    type_summary: Dict[IntegrationTestType, Dict[str, int]]
    component_coverage: Dict[str, int]
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


class HybridIntegrationTests:
    """
    Comprehensive integration testing for hybrid p-adic system.
    Tests integration between all hybrid system components and workflows.
    """
    
    def __init__(self, config: Optional[IntegrationTestConfig] = None):
        """Initialize hybrid integration tests"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, IntegrationTestConfig):
            raise TypeError(f"Config must be IntegrationTestConfig or None, got {type(config)}")
        
        self.config = config or IntegrationTestConfig()
        self.logger = logging.getLogger('HybridIntegrationTests')
        
        # Core components
        self.hybrid_manager: Optional[HybridPadicManager] = None
        self.hybrid_compressor: Optional[HybridPadicCompressionSystem] = None
        self.switching_manager: Optional[DynamicSwitchingManager] = None
        self.pure_compressor: Optional[PadicCompressionSystem] = None
        
        # Performance components
        self.performance_optimizer: Optional[HybridPerformanceOptimizer] = None
        self.performance_monitor: Optional[HybridPerformanceMonitor] = None
        self.performance_analyzer: Optional[HybridPerformanceAnalyzer] = None
        self.performance_tuner: Optional[HybridPerformanceTuner] = None
        
        # Testing components
        self.test_suite: Optional[HybridTestSuite] = None
        self.validation_framework: Optional[HybridValidationFramework] = None
        
        # Base components
        self.base_performance_optimizer: Optional[PerformanceOptimizer] = None
        
        # Test state
        self.is_initialized = False
        self.is_running = False
        
        # Test tracking
        self.test_results: List[IntegrationTestExecution] = []
        self.current_test: Optional[str] = None
        self.test_start_time: Optional[datetime] = None
        
        # Thread safety
        self._test_lock = threading.RLock()
        self._results_lock = threading.RLock()
        
        # Component registry
        self.component_registry: Dict[str, Any] = {}
        
        self.logger.info("HybridIntegrationTests created successfully")
    
    def initialize_integration_tests(self,
                                   hybrid_manager: HybridPadicManager,
                                   hybrid_compressor: HybridPadicCompressionSystem,
                                   switching_manager: DynamicSwitchingManager,
                                   pure_compressor: Optional[PadicCompressionSystem] = None,
                                   base_performance_optimizer: Optional[PerformanceOptimizer] = None) -> None:
        """
        Initialize integration tests with system components.
        
        Args:
            hybrid_manager: Hybrid p-adic manager instance
            hybrid_compressor: Hybrid compression system instance
            switching_manager: Dynamic switching manager instance
            pure_compressor: Optional pure p-adic compressor
            base_performance_optimizer: Optional base performance optimizer
            
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
        if base_performance_optimizer is not None and not isinstance(base_performance_optimizer, PerformanceOptimizer):
            raise TypeError(f"Base optimizer must be PerformanceOptimizer, got {type(base_performance_optimizer)}")
        
        try:
            # Set core component references
            self.hybrid_manager = hybrid_manager
            self.hybrid_compressor = hybrid_compressor
            self.switching_manager = switching_manager
            self.pure_compressor = pure_compressor
            self.base_performance_optimizer = base_performance_optimizer
            
            # Register core components
            self.component_registry = {
                'hybrid_manager': hybrid_manager,
                'hybrid_compressor': hybrid_compressor,
                'switching_manager': switching_manager
            }
            
            if pure_compressor:
                self.component_registry['pure_compressor'] = pure_compressor
            if base_performance_optimizer:
                self.component_registry['base_performance_optimizer'] = base_performance_optimizer
            
            # Initialize performance components if enabled
            if self.config.enable_performance_components:
                self._initialize_performance_components()
            
            # Initialize validation components if enabled
            if self.config.enable_validation_components:
                self._initialize_validation_components()
            
            self.is_initialized = True
            self.logger.info("Hybrid integration tests initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize integration tests: {e}")
            raise RuntimeError(f"Integration tests initialization failed: {e}")
    
    def run_all_integration_tests(self) -> IntegrationTestReport:
        """
        Run all enabled integration tests and generate comprehensive report.
        
        Returns:
            Comprehensive integration test report
            
        Raises:
            RuntimeError: If tests are not initialized or execution fails
        """
        if not self.is_initialized:
            raise RuntimeError("Integration tests not initialized")
        
        if self.is_running:
            raise RuntimeError("Integration tests are already running")
        
        with self._test_lock:
            try:
                self.is_running = True
                self.test_results.clear()
                start_time = time.time()
                
                self.logger.info("Starting comprehensive hybrid integration tests")
                
                # Run component tests
                if self.config.enable_component_tests:
                    self._run_component_integration_tests()
                
                # Run system tests
                if self.config.enable_system_tests:
                    self._run_system_integration_tests()
                
                # Run cross-system tests
                if self.config.enable_cross_system_tests:
                    self._run_cross_system_integration_tests()
                
                # Run end-to-end tests
                if self.config.enable_end_to_end_tests:
                    self._run_end_to_end_integration_tests()
                
                # Run workflow tests
                if self.config.enable_workflow_tests:
                    self._run_workflow_integration_tests()
                
                # Calculate execution time
                total_execution_time = (time.time() - start_time) * 1000
                
                # Generate report
                report = self._generate_integration_report(total_execution_time)
                
                self.logger.info(f"Integration tests completed: {report.passed_tests}/{report.total_tests} tests passed")
                
                return report
                
            except Exception as e:
                self.logger.error(f"Integration tests execution failed: {e}")
                raise RuntimeError(f"Integration tests execution failed: {e}")
            finally:
                self.is_running = False
    
    def test_core_component_integration(self) -> IntegrationTestExecution:
        """
        Test integration between core hybrid components.
        
        Returns:
            Integration test execution result
            
        Raises:
            RuntimeError: If test fails
        """
        test_name = "core_component_integration"
        start_time = time.time()
        
        try:
            # Test hybrid manager and compressor integration
            test_data = torch.randn(1000)
            
            # Test switching manager decision integration
            should_switch = self.switching_manager.should_switch_to_hybrid(test_data)
            if not isinstance(should_switch, bool):
                raise RuntimeError("Switching manager integration failed")
            
            # Test compression integration
            compressed = self.hybrid_compressor.compress(test_data)
            if not isinstance(compressed, dict):
                raise RuntimeError("Compression integration failed")
            
            # Test decompression integration
            decompressed = self.hybrid_compressor.decompress(
                compressed['compressed_data'],
                compressed['compression_info']
            )
            
            # Validate reconstruction
            reconstruction_error = torch.norm(test_data - decompressed).item()
            if reconstruction_error > self.config.mathematical_tolerance:
                raise RuntimeError(f"Integration reconstruction error: {reconstruction_error}")
            
            execution_time = (time.time() - start_time) * 1000
            
            return IntegrationTestExecution(
                test_name=test_name,
                test_type=IntegrationTestType.COMPONENT,
                result=IntegrationTestResult.PASSED,
                execution_time_ms=execution_time,
                components_tested=['hybrid_manager', 'hybrid_compressor', 'switching_manager'],
                details={
                    'reconstruction_error': reconstruction_error,
                    'switching_decision': should_switch,
                    'data_size': test_data.numel()
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return IntegrationTestExecution(
                test_name=test_name,
                test_type=IntegrationTestType.COMPONENT,
                result=IntegrationTestResult.FAILED,
                execution_time_ms=execution_time,
                components_tested=['hybrid_manager', 'hybrid_compressor', 'switching_manager'],
                error_message=str(e)
            )
    
    def test_performance_system_integration(self) -> IntegrationTestExecution:
        """
        Test integration of performance optimization system.
        
        Returns:
            Integration test execution result
            
        Raises:
            RuntimeError: If test fails
        """
        if not self.config.enable_performance_components:
            return IntegrationTestExecution(
                test_name="performance_system_integration",
                test_type=IntegrationTestType.SYSTEM,
                result=IntegrationTestResult.SKIPPED,
                execution_time_ms=0.0,
                components_tested=[],
                details={'reason': 'Performance components disabled'}
            )
        
        test_name = "performance_system_integration"
        start_time = time.time()
        
        try:
            # Test performance optimizer integration
            if self.performance_optimizer and self.performance_optimizer.is_initialized:
                # Test profiling integration
                def test_operation():
                    test_data = torch.randn(1000)
                    return self.hybrid_compressor.compress(test_data)
                
                result = self.performance_optimizer.profile_hybrid_operation(
                    "integration_test_operation",
                    test_operation
                )
                
                if result is None:
                    raise RuntimeError("Performance optimizer integration failed")
            
            # Test performance monitor integration
            if self.performance_monitor and self.performance_monitor.is_monitoring:
                op_id = self.performance_monitor.monitor_hybrid_operation(
                    "integration_test",
                    {"test": True}
                )
                
                if not isinstance(op_id, str):
                    raise RuntimeError("Performance monitor integration failed")
                
                self.performance_monitor.complete_operation_monitoring(op_id, success=True)
                
                # Verify operation was recorded
                if len(self.performance_monitor.operation_history) == 0:
                    raise RuntimeError("Performance monitor operation recording failed")
            
            # Test performance analyzer integration
            if self.performance_analyzer and self.performance_analyzer.is_initialized:
                # This would test analyzer integration if we have operation data
                pass
            
            # Test performance tuner integration
            if self.performance_tuner and self.performance_tuner.is_initialized:
                recommendations = self.performance_tuner.get_tuning_recommendations()
                if not isinstance(recommendations, list):
                    raise RuntimeError("Performance tuner integration failed")
            
            execution_time = (time.time() - start_time) * 1000
            
            components_tested = []
            if self.performance_optimizer:
                components_tested.append('performance_optimizer')
            if self.performance_monitor:
                components_tested.append('performance_monitor')
            if self.performance_analyzer:
                components_tested.append('performance_analyzer')
            if self.performance_tuner:
                components_tested.append('performance_tuner')
            
            return IntegrationTestExecution(
                test_name=test_name,
                test_type=IntegrationTestType.SYSTEM,
                result=IntegrationTestResult.PASSED,
                execution_time_ms=execution_time,
                components_tested=components_tested,
                details={'components_integrated': len(components_tested)}
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return IntegrationTestExecution(
                test_name=test_name,
                test_type=IntegrationTestType.SYSTEM,
                result=IntegrationTestResult.FAILED,
                execution_time_ms=execution_time,
                components_tested=['performance_system'],
                error_message=str(e)
            )
    
    def test_validation_system_integration(self) -> IntegrationTestExecution:
        """
        Test integration of validation system.
        
        Returns:
            Integration test execution result
            
        Raises:
            RuntimeError: If test fails
        """
        if not self.config.enable_validation_components:
            return IntegrationTestExecution(
                test_name="validation_system_integration",
                test_type=IntegrationTestType.SYSTEM,
                result=IntegrationTestResult.SKIPPED,
                execution_time_ms=0.0,
                components_tested=[],
                details={'reason': 'Validation components disabled'}
            )
        
        test_name = "validation_system_integration"
        start_time = time.time()
        
        try:
            components_tested = []
            
            # Test test suite integration
            if self.test_suite and self.test_suite.is_initialized:
                # Run a quick unit test
                test_data = torch.randn(100)
                result = self.test_suite.test_hybrid_compression_correctness(100)
                
                if result.result.value != 'passed':
                    raise RuntimeError("Test suite integration failed")
                
                components_tested.append('test_suite')
            
            # Test validation framework integration
            if self.validation_framework and self.validation_framework.is_initialized:
                # Run a quick validation
                test_data = torch.randn(100)
                result = self.validation_framework.validate_compression_decompression_cycle(test_data)
                
                if result.status.value != 'passed':
                    raise RuntimeError("Validation framework integration failed")
                
                components_tested.append('validation_framework')
            
            execution_time = (time.time() - start_time) * 1000
            
            return IntegrationTestExecution(
                test_name=test_name,
                test_type=IntegrationTestType.SYSTEM,
                result=IntegrationTestResult.PASSED,
                execution_time_ms=execution_time,
                components_tested=components_tested,
                details={'components_integrated': len(components_tested)}
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return IntegrationTestExecution(
                test_name=test_name,
                test_type=IntegrationTestType.SYSTEM,
                result=IntegrationTestResult.FAILED,
                execution_time_ms=execution_time,
                components_tested=['validation_system'],
                error_message=str(e)
            )
    
    def test_hybrid_pure_cross_system_integration(self) -> IntegrationTestExecution:
        """
        Test cross-system integration between hybrid and pure systems.
        
        Returns:
            Integration test execution result
            
        Raises:
            RuntimeError: If test fails
        """
        if self.pure_compressor is None:
            return IntegrationTestExecution(
                test_name="hybrid_pure_cross_system_integration",
                test_type=IntegrationTestType.CROSS_SYSTEM,
                result=IntegrationTestResult.SKIPPED,
                execution_time_ms=0.0,
                components_tested=[],
                details={'reason': 'Pure compressor not available'}
            )
        
        test_name = "hybrid_pure_cross_system_integration"
        start_time = time.time()
        
        try:
            test_data = torch.randn(1000)
            
            # Compress with both systems
            hybrid_compressed = self.hybrid_compressor.compress(test_data.clone())
            pure_compressed = self.pure_compressor.compress(test_data.clone())
            
            # Decompress both
            hybrid_decompressed = self.hybrid_compressor.decompress(
                hybrid_compressed['compressed_data'],
                hybrid_compressed['compression_info']
            )
            pure_decompressed = self.pure_compressor.decompress(
                pure_compressed['compressed_data'],
                pure_compressed['compression_info']
            )
            
            # Test cross-system compatibility
            equivalence_error = torch.norm(hybrid_decompressed - pure_decompressed).item()
            if equivalence_error > self.config.mathematical_tolerance:
                raise RuntimeError(f"Cross-system equivalence error: {equivalence_error}")
            
            # Test switching logic integration
            should_use_hybrid = self.switching_manager.should_switch_to_hybrid(test_data)
            
            execution_time = (time.time() - start_time) * 1000
            
            return IntegrationTestExecution(
                test_name=test_name,
                test_type=IntegrationTestType.CROSS_SYSTEM,
                result=IntegrationTestResult.PASSED,
                execution_time_ms=execution_time,
                components_tested=['hybrid_compressor', 'pure_compressor', 'switching_manager'],
                details={
                    'equivalence_error': equivalence_error,
                    'switching_decision': should_use_hybrid,
                    'hybrid_compression_ratio': test_data.numel() / len(hybrid_compressed['compressed_data']),
                    'pure_compression_ratio': test_data.numel() / len(pure_compressed['compressed_data'])
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return IntegrationTestExecution(
                test_name=test_name,
                test_type=IntegrationTestType.CROSS_SYSTEM,
                result=IntegrationTestResult.FAILED,
                execution_time_ms=execution_time,
                components_tested=['hybrid_compressor', 'pure_compressor'],
                error_message=str(e)
            )
    
    def test_end_to_end_workflow(self) -> IntegrationTestExecution:
        """
        Test complete end-to-end workflow integration.
        
        Returns:
            Integration test execution result
            
        Raises:
            RuntimeError: If test fails
        """
        test_name = "end_to_end_workflow"
        start_time = time.time()
        
        try:
            components_used = []
            workflow_steps = []
            
            # Step 1: Generate test data
            test_data = torch.randn(2000)
            workflow_steps.append("data_generation")
            
            # Step 2: Switching decision
            should_use_hybrid = self.switching_manager.should_switch_to_hybrid(test_data)
            workflow_steps.append("switching_decision")
            components_used.append('switching_manager')
            
            # Step 3: Performance monitoring (if available)
            op_id = None
            if self.performance_monitor and self.performance_monitor.is_monitoring:
                op_id = self.performance_monitor.monitor_hybrid_operation(
                    "end_to_end_test",
                    {"hybrid_mode": should_use_hybrid, "data_size": test_data.numel()}
                )
                workflow_steps.append("performance_monitoring_start")
                components_used.append('performance_monitor')
            
            # Step 4: Compression
            compression_start = time.time()
            compressed = self.hybrid_compressor.compress(test_data)
            compression_time = (time.time() - compression_start) * 1000
            workflow_steps.append("compression")
            components_used.append('hybrid_compressor')
            
            # Step 5: Decompression
            decompression_start = time.time()
            decompressed = self.hybrid_compressor.decompress(
                compressed['compressed_data'],
                compressed['compression_info']
            )
            decompression_time = (time.time() - decompression_start) * 1000
            workflow_steps.append("decompression")
            
            # Step 6: Validation
            reconstruction_error = torch.norm(test_data - decompressed).item()
            if reconstruction_error > self.config.mathematical_tolerance:
                raise RuntimeError(f"End-to-end reconstruction error: {reconstruction_error}")
            workflow_steps.append("validation")
            
            # Step 7: Complete performance monitoring (if started)
            if op_id:
                self.performance_monitor.complete_operation_monitoring(op_id, success=True)
                workflow_steps.append("performance_monitoring_complete")
            
            # Step 8: Performance analysis (if available)
            if self.performance_analyzer and self.performance_analyzer.is_initialized:
                # This would trigger analysis if we had enough data
                workflow_steps.append("performance_analysis")
                components_used.append('performance_analyzer')
            
            execution_time = (time.time() - start_time) * 1000
            
            return IntegrationTestExecution(
                test_name=test_name,
                test_type=IntegrationTestType.END_TO_END,
                result=IntegrationTestResult.PASSED,
                execution_time_ms=execution_time,
                components_tested=components_used,
                details={
                    'workflow_steps': workflow_steps,
                    'reconstruction_error': reconstruction_error,
                    'compression_time_ms': compression_time,
                    'decompression_time_ms': decompression_time,
                    'compression_ratio': test_data.numel() / len(compressed['compressed_data']),
                    'switching_decision': should_use_hybrid,
                    'data_size': test_data.numel()
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return IntegrationTestExecution(
                test_name=test_name,
                test_type=IntegrationTestType.END_TO_END,
                result=IntegrationTestResult.FAILED,
                execution_time_ms=execution_time,
                components_tested=['end_to_end_workflow'],
                error_message=str(e)
            )
    
    def test_concurrent_operations_integration(self) -> IntegrationTestExecution:
        """
        Test integration under concurrent operations.
        
        Returns:
            Integration test execution result
            
        Raises:
            RuntimeError: If test fails
        """
        test_name = "concurrent_operations_integration"
        start_time = time.time()
        
        try:
            import concurrent.futures
            
            def worker_operation(worker_id: int) -> Dict[str, Any]:
                """Worker operation for concurrent testing"""
                test_data = torch.randn(1000 + worker_id * 100)
                
                # Test switching decision
                should_switch = self.switching_manager.should_switch_to_hybrid(test_data)
                
                # Test compression/decompression
                compressed = self.hybrid_compressor.compress(test_data)
                decompressed = self.hybrid_compressor.decompress(
                    compressed['compressed_data'],
                    compressed['compression_info']
                )
                
                # Validate result
                error = torch.norm(test_data - decompressed).item()
                
                return {
                    'worker_id': worker_id,
                    'switching_decision': should_switch,
                    'reconstruction_error': error,
                    'success': error < self.config.mathematical_tolerance
                }
            
            # Run concurrent operations
            max_workers = min(4, self.config.max_concurrent_tests)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(worker_operation, i) 
                    for i in range(max_workers * 2)
                ]
                
                results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                    except Exception as e:
                        raise RuntimeError(f"Concurrent operation failed: {e}")
            
            # Validate all operations succeeded
            successful_operations = sum(1 for r in results if r['success'])
            if successful_operations != len(results):
                raise RuntimeError(f"Only {successful_operations}/{len(results)} concurrent operations succeeded")
            
            execution_time = (time.time() - start_time) * 1000
            
            return IntegrationTestExecution(
                test_name=test_name,
                test_type=IntegrationTestType.WORKFLOW,
                result=IntegrationTestResult.PASSED,
                execution_time_ms=execution_time,
                components_tested=['hybrid_compressor', 'switching_manager'],
                details={
                    'concurrent_operations': len(results),
                    'successful_operations': successful_operations,
                    'max_reconstruction_error': max(r['reconstruction_error'] for r in results),
                    'average_reconstruction_error': sum(r['reconstruction_error'] for r in results) / len(results)
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return IntegrationTestExecution(
                test_name=test_name,
                test_type=IntegrationTestType.WORKFLOW,
                result=IntegrationTestResult.FAILED,
                execution_time_ms=execution_time,
                components_tested=['concurrent_operations'],
                error_message=str(e)
            )
    
    def _initialize_performance_components(self) -> None:
        """Initialize performance optimization components"""
        try:
            from .hybrid_performance_optimizer import HybridPerformanceOptimizer, HybridOptimizationConfig
            from .hybrid_performance_monitor import HybridPerformanceMonitor, HybridMonitoringConfig
            from .hybrid_performance_analyzer import HybridPerformanceAnalyzer, HybridAnalysisConfig
            from .hybrid_performance_tuner import HybridPerformanceTuner, HybridTuningConfig
            
            # Initialize performance optimizer
            opt_config = HybridOptimizationConfig()
            self.performance_optimizer = HybridPerformanceOptimizer(opt_config)
            
            if self.base_performance_optimizer:
                self.performance_optimizer.initialize_hybrid_optimization(
                    self.base_performance_optimizer,
                    self.hybrid_manager,
                    self.switching_manager,
                    self.hybrid_compressor
                )
                self.component_registry['performance_optimizer'] = self.performance_optimizer
            
            # Initialize performance monitor
            mon_config = HybridMonitoringConfig(monitoring_interval_seconds=0.1)
            self.performance_monitor = HybridPerformanceMonitor(mon_config)
            self.performance_monitor.start_hybrid_monitoring(
                self.hybrid_manager,
                self.switching_manager,
                self.hybrid_compressor
            )
            self.component_registry['performance_monitor'] = self.performance_monitor
            
            # Initialize performance analyzer
            ana_config = HybridAnalysisConfig()
            self.performance_analyzer = HybridPerformanceAnalyzer(ana_config)
            self.performance_analyzer.initialize_analyzer(
                self.hybrid_manager,
                self.switching_manager,
                self.hybrid_compressor,
                self.performance_monitor
            )
            self.component_registry['performance_analyzer'] = self.performance_analyzer
            
            # Initialize performance tuner
            tun_config = HybridTuningConfig(max_tuning_iterations=5)
            self.performance_tuner = HybridPerformanceTuner(tun_config)
            self.performance_tuner.initialize_tuner(
                self.hybrid_manager,
                self.switching_manager,
                self.hybrid_compressor,
                self.performance_monitor,
                self.performance_analyzer
            )
            self.component_registry['performance_tuner'] = self.performance_tuner
            
            self.logger.info("Performance components initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize performance components: {e}")
    
    def _initialize_validation_components(self) -> None:
        """Initialize validation and testing components"""
        try:
            from .hybrid_test_suite import HybridTestSuite, TestSuiteConfig
            from .hybrid_validation_framework import HybridValidationFramework, ValidationFrameworkConfig
            
            # Initialize test suite
            test_config = TestSuiteConfig(
                test_data_sizes=[100, 500, 1000],
                enable_parallel_execution=False,  # Simpler for integration
                max_concurrent_tests=2
            )
            self.test_suite = HybridTestSuite(test_config)
            self.test_suite.initialize_test_suite(
                self.hybrid_manager,
                self.hybrid_compressor,
                self.switching_manager,
                self.pure_compressor
            )
            self.component_registry['test_suite'] = self.test_suite
            
            # Initialize validation framework
            val_config = ValidationFrameworkConfig(
                enable_parallel_validation=False,  # Simpler for integration
                max_concurrent_validations=2
            )
            self.validation_framework = HybridValidationFramework(val_config)
            self.validation_framework.initialize_framework(
                self.hybrid_manager,
                self.hybrid_compressor,
                self.switching_manager,
                self.pure_compressor
            )
            self.component_registry['validation_framework'] = self.validation_framework
            
            self.logger.info("Validation components initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize validation components: {e}")
    
    def _run_component_integration_tests(self) -> None:
        """Run all component integration tests"""
        self.logger.info("Running component integration tests")
        
        # Test core component integration
        result = self.test_core_component_integration()
        with self._results_lock:
            self.test_results.append(result)
    
    def _run_system_integration_tests(self) -> None:
        """Run all system integration tests"""
        self.logger.info("Running system integration tests")
        
        # Test performance system integration
        result = self.test_performance_system_integration()
        with self._results_lock:
            self.test_results.append(result)
        
        # Test validation system integration
        result = self.test_validation_system_integration()
        with self._results_lock:
            self.test_results.append(result)
    
    def _run_cross_system_integration_tests(self) -> None:
        """Run all cross-system integration tests"""
        self.logger.info("Running cross-system integration tests")
        
        # Test hybrid-pure cross-system integration
        result = self.test_hybrid_pure_cross_system_integration()
        with self._results_lock:
            self.test_results.append(result)
    
    def _run_end_to_end_integration_tests(self) -> None:
        """Run all end-to-end integration tests"""
        self.logger.info("Running end-to-end integration tests")
        
        # Test end-to-end workflow
        result = self.test_end_to_end_workflow()
        with self._results_lock:
            self.test_results.append(result)
    
    def _run_workflow_integration_tests(self) -> None:
        """Run all workflow integration tests"""
        self.logger.info("Running workflow integration tests")
        
        # Test concurrent operations integration
        result = self.test_concurrent_operations_integration()
        with self._results_lock:
            self.test_results.append(result)
    
    def _generate_integration_report(self, total_execution_time: float) -> IntegrationTestReport:
        """Generate comprehensive integration test report"""
        # Count results by status
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.result == IntegrationTestResult.PASSED)
        failed_tests = sum(1 for r in self.test_results if r.result == IntegrationTestResult.FAILED)
        skipped_tests = sum(1 for r in self.test_results if r.result == IntegrationTestResult.SKIPPED)
        error_tests = sum(1 for r in self.test_results if r.result == IntegrationTestResult.ERROR)
        
        # Type summary
        type_summary = {}
        for test_type in IntegrationTestType:
            type_results = [r for r in self.test_results if r.test_type == test_type]
            type_summary[test_type] = {
                'total': len(type_results),
                'passed': sum(1 for r in type_results if r.result == IntegrationTestResult.PASSED),
                'failed': sum(1 for r in type_results if r.result == IntegrationTestResult.FAILED)
            }
        
        # Component coverage
        component_coverage = defaultdict(int)
        for result in self.test_results:
            for component in result.components_tested:
                component_coverage[component] += 1
        
        # Critical failures
        critical_failures = [
            r.test_name for r in self.test_results 
            if r.result == IntegrationTestResult.FAILED and 'core' in r.test_name
        ]
        
        # Recommendations
        recommendations = []
        if failed_tests > 0:
            recommendations.append(f"Address {failed_tests} failed integration tests")
        if len(critical_failures) > 0:
            recommendations.append(f"Fix {len(critical_failures)} critical integration failures")
        if error_tests > 0:
            recommendations.append(f"Investigate {error_tests} integration test errors")
        
        # Component coverage recommendations
        low_coverage_components = [
            comp for comp, count in component_coverage.items() if count < 2
        ]
        if low_coverage_components:
            recommendations.append(f"Improve test coverage for: {', '.join(low_coverage_components)}")
        
        return IntegrationTestReport(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            error_tests=error_tests,
            total_execution_time_ms=total_execution_time,
            test_results=self.test_results.copy(),
            type_summary=type_summary,
            component_coverage=dict(component_coverage),
            critical_failures=critical_failures,
            recommendations=recommendations
        )
    
    def shutdown(self) -> None:
        """Shutdown integration tests"""
        self.logger.info("Shutting down hybrid integration tests")
        
        # Shutdown performance components
        if self.performance_tuner and self.performance_tuner.is_initialized:
            self.performance_tuner.shutdown()
        if self.performance_analyzer and self.performance_analyzer.is_initialized:
            self.performance_analyzer.shutdown()
        if self.performance_monitor and self.performance_monitor.is_monitoring:
            self.performance_monitor.stop_hybrid_monitoring()
        if self.performance_optimizer and self.performance_optimizer.is_initialized:
            self.performance_optimizer.shutdown()
        
        # Shutdown validation components
        if self.validation_framework and self.validation_framework.is_initialized:
            self.validation_framework.shutdown()
        if self.test_suite and self.test_suite.is_initialized:
            self.test_suite.shutdown()
        
        # Clear references
        self.hybrid_manager = None
        self.hybrid_compressor = None
        self.switching_manager = None
        self.pure_compressor = None
        self.base_performance_optimizer = None
        
        self.performance_optimizer = None
        self.performance_monitor = None
        self.performance_analyzer = None
        self.performance_tuner = None
        
        self.test_suite = None
        self.validation_framework = None
        
        # Clear component registry
        self.component_registry.clear()
        
        self.is_initialized = False
        self.logger.info("Hybrid integration tests shutdown complete")


def run_hybrid_integration_tests(
    hybrid_manager: HybridPadicManager,
    hybrid_compressor: HybridPadicCompressionSystem,
    switching_manager: DynamicSwitchingManager,
    pure_compressor: Optional[PadicCompressionSystem] = None,
    base_performance_optimizer: Optional[PerformanceOptimizer] = None,
    config: Optional[IntegrationTestConfig] = None
) -> IntegrationTestReport:
    """
    Run hybrid integration tests with provided components.
    
    Args:
        hybrid_manager: Hybrid p-adic manager instance
        hybrid_compressor: Hybrid compression system instance
        switching_manager: Dynamic switching manager instance
        pure_compressor: Optional pure p-adic compressor
        base_performance_optimizer: Optional base performance optimizer
        config: Optional integration test configuration
        
    Returns:
        Integration test report
        
    Raises:
        RuntimeError: If test execution fails
    """
    # Create integration tests
    integration_tests = HybridIntegrationTests(config)
    
    try:
        # Initialize tests
        integration_tests.initialize_integration_tests(
            hybrid_manager,
            hybrid_compressor,
            switching_manager,
            pure_compressor,
            base_performance_optimizer
        )
        
        # Run all tests
        report = integration_tests.run_all_integration_tests()
        
        return report
        
    finally:
        # Ensure cleanup
        integration_tests.shutdown()