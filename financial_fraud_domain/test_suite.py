"""
Comprehensive Test Suite for Financial Fraud Detection
Enterprise-grade testing framework for fraud detection domain
"""

import logging
import unittest
import time
import threading
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import statistics
import traceback
import sys
import os
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of tests"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    REGRESSION = "regression"
    LOAD = "load"
    STRESS = "stress"


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestPriority(Enum):
    """Test priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TestResult:
    """Individual test result"""
    test_id: str
    test_name: str
    test_type: TestType
    status: TestStatus
    execution_time_ms: float
    start_time: datetime
    end_time: datetime
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    assertions: int = 0
    passed_assertions: int = 0
    failed_assertions: int = 0
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuiteResult:
    """Test suite execution results"""
    suite_id: str
    start_time: datetime
    end_time: datetime
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0
    
    total_execution_time_ms: float = 0.0
    average_test_time_ms: float = 0.0
    
    test_results: List[TestResult] = field(default_factory=list)
    coverage_data: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    security_findings: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate test success rate"""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100.0


@dataclass
class TestConfiguration:
    """Test configuration settings"""
    parallel_execution: bool = True
    max_workers: int = 4
    timeout_seconds: int = 300
    retry_failed_tests: bool = True
    max_retries: int = 2
    collect_coverage: bool = True
    performance_benchmarks: bool = True
    security_scanning: bool = True
    generate_reports: bool = True
    report_format: str = "json"  # json, html, xml
    output_directory: str = "test_results"


class TestExecutor:
    """Handles test execution with various strategies"""
    
    def __init__(self, config: TestConfiguration):
        self.config = config
        self.results: List[TestResult] = []
        self.lock = threading.Lock()
        
    def execute_tests(self, test_cases: List[Callable]) -> List[TestResult]:
        """Execute test cases with configured strategy"""
        if self.config.parallel_execution:
            return self._execute_parallel(test_cases)
        else:
            return self._execute_sequential(test_cases)
    
    def _execute_parallel(self, test_cases: List[Callable]) -> List[TestResult]:
        """Execute tests in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_test = {
                executor.submit(self._execute_single_test, test_case): test_case 
                for test_case in test_cases
            }
            
            for future in as_completed(future_to_test, timeout=self.config.timeout_seconds):
                test_case = future_to_test[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Test {test_case.__name__} failed with exception: {e}")
                    # Create error result
                    error_result = TestResult(
                        test_id=f"test_{len(results)}",
                        test_name=test_case.__name__,
                        test_type=TestType.UNIT,
                        status=TestStatus.ERROR,
                        execution_time_ms=0.0,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error_message=str(e),
                        stack_trace=traceback.format_exc()
                    )
                    results.append(error_result)
        
        return results
    
    def _execute_sequential(self, test_cases: List[Callable]) -> List[TestResult]:
        """Execute tests sequentially"""
        results = []
        
        for test_case in test_cases:
            try:
                result = self._execute_single_test(test_case)
                results.append(result)
            except Exception as e:
                logger.error(f"Test {test_case.__name__} failed with exception: {e}")
                error_result = TestResult(
                    test_id=f"test_{len(results)}",
                    test_name=test_case.__name__,
                    test_type=TestType.UNIT,
                    status=TestStatus.ERROR,
                    execution_time_ms=0.0,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    error_message=str(e),
                    stack_trace=traceback.format_exc()
                )
                results.append(error_result)
        
        return results
    
    def _execute_single_test(self, test_case: Callable) -> TestResult:
        """Execute a single test case"""
        start_time = datetime.now()
        test_id = f"test_{int(time.time() * 1000)}"
        
        try:
            start_ms = time.time() * 1000
            
            # Execute the test
            test_case()
            
            end_ms = time.time() * 1000
            end_time = datetime.now()
            
            return TestResult(
                test_id=test_id,
                test_name=test_case.__name__,
                test_type=TestType.UNIT,
                status=TestStatus.PASSED,
                execution_time_ms=end_ms - start_ms,
                start_time=start_time,
                end_time=end_time,
                assertions=1,
                passed_assertions=1
            )
            
        except AssertionError as e:
            end_time = datetime.now()
            return TestResult(
                test_id=test_id,
                test_name=test_case.__name__,
                test_type=TestType.UNIT,
                status=TestStatus.FAILED,
                execution_time_ms=(end_time - start_time).total_seconds() * 1000,
                start_time=start_time,
                end_time=end_time,
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                assertions=1,
                failed_assertions=1
            )
        except Exception as e:
            end_time = datetime.now()
            return TestResult(
                test_id=test_id,
                test_name=test_case.__name__,
                test_type=TestType.UNIT,
                status=TestStatus.ERROR,
                execution_time_ms=(end_time - start_time).total_seconds() * 1000,
                start_time=start_time,
                end_time=end_time,
                error_message=str(e),
                stack_trace=traceback.format_exc()
            )


class PerformanceTester:
    """Performance testing capabilities"""
    
    def __init__(self):
        self.benchmarks: Dict[str, List[float]] = defaultdict(list)
        
    def benchmark_function(self, func: Callable, iterations: int = 100) -> Dict[str, float]:
        """Benchmark a function's performance"""
        execution_times = []
        
        for _ in range(iterations):
            start_time = time.time()
            func()
            end_time = time.time()
            execution_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'average_ms': statistics.mean(execution_times),
            'median_ms': statistics.median(execution_times),
            'min_ms': min(execution_times),
            'max_ms': max(execution_times),
            'std_dev_ms': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            'iterations': iterations
        }
    
    def stress_test(self, func: Callable, duration_seconds: int = 60) -> Dict[str, Any]:
        """Perform stress testing"""
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        execution_count = 0
        errors = 0
        execution_times = []
        
        while time.time() < end_time:
            try:
                func_start = time.time()
                func()
                func_end = time.time()
                execution_times.append((func_end - func_start) * 1000)
                execution_count += 1
            except Exception as e:
                errors += 1
                logger.error(f"Stress test error: {e}")
        
        actual_duration = time.time() - start_time
        
        return {
            'duration_seconds': actual_duration,
            'total_executions': execution_count,
            'executions_per_second': execution_count / actual_duration,
            'total_errors': errors,
            'error_rate': errors / execution_count if execution_count > 0 else 0,
            'average_execution_time_ms': statistics.mean(execution_times) if execution_times else 0,
            'max_execution_time_ms': max(execution_times) if execution_times else 0
        }


class SecurityTester:
    """Security testing capabilities"""
    
    def __init__(self):
        self.vulnerabilities: List[Dict[str, Any]] = []
        
    def scan_for_vulnerabilities(self, target_function: Callable) -> List[Dict[str, Any]]:
        """Basic security vulnerability scanning"""
        vulnerabilities = []
        
        # Test for SQL injection patterns
        sql_injection_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users --"
        ]
        
        for payload in sql_injection_payloads:
            try:
                result = target_function(payload)
                if "error" in str(result).lower() or "exception" in str(result).lower():
                    vulnerabilities.append({
                        'type': 'SQL_INJECTION',
                        'payload': payload,
                        'severity': 'HIGH',
                        'description': 'Potential SQL injection vulnerability detected'
                    })
            except Exception:
                pass
        
        # Test for XSS patterns
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>"
        ]
        
        for payload in xss_payloads:
            try:
                result = target_function(payload)
                if payload in str(result):
                    vulnerabilities.append({
                        'type': 'XSS',
                        'payload': payload,
                        'severity': 'MEDIUM',
                        'description': 'Potential XSS vulnerability detected'
                    })
            except Exception:
                pass
        
        return vulnerabilities
    
    def test_input_validation(self, func: Callable) -> List[Dict[str, Any]]:
        """Test input validation"""
        issues = []
        
        # Test with various invalid inputs
        invalid_inputs = [
            None,
            "",
            " " * 1000,  # Large whitespace
            "A" * 10000,  # Large string
            -1,
            float('inf'),
            float('nan'),
            {},
            [],
            {'malicious': 'payload'}
        ]
        
        for invalid_input in invalid_inputs:
            try:
                func(invalid_input)
                issues.append({
                    'type': 'INPUT_VALIDATION',
                    'input': str(invalid_input)[:100],
                    'severity': 'MEDIUM',
                    'description': 'Function did not properly validate input'
                })
            except (ValueError, TypeError, AttributeError):
                # Expected behavior - function properly rejected invalid input
                pass
            except Exception as e:
                issues.append({
                    'type': 'UNEXPECTED_ERROR',
                    'input': str(invalid_input)[:100],
                    'error': str(e),
                    'severity': 'HIGH',
                    'description': 'Unexpected error handling invalid input'
                })
        
        return issues


class CodeCoverageTracker:
    """Code coverage tracking"""
    
    def __init__(self):
        self.coverage_data: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
    def start_coverage(self):
        """Start coverage tracking"""
        # In a real implementation, this would integrate with coverage.py
        logger.info("Coverage tracking started")
        
    def stop_coverage(self) -> Dict[str, float]:
        """Stop coverage tracking and return results"""
        # Mock coverage data - in real implementation would use coverage.py
        mock_coverage = {
            'fraud_core.py': 85.5,
            'data_loader.py': 92.3,
            'ml_predictor.py': 78.9,
            'api_interface.py': 88.1,
            'performance_monitor.py': 75.6
        }
        
        logger.info("Coverage tracking stopped")
        return mock_coverage


class FinancialTestSuite:
    """
    Comprehensive test suite for Financial Fraud Detection domain.
    Provides enterprise-grade testing capabilities including unit, integration,
    performance, and security testing.
    """
    
    def __init__(self, config: Optional[TestConfiguration] = None):
        """Initialize the test suite"""
        self.config = config or TestConfiguration()
        self.executor = TestExecutor(self.config)
        self.performance_tester = PerformanceTester()
        self.security_tester = SecurityTester()
        self.coverage_tracker = CodeCoverageTracker()
        
        # Test registry
        self.unit_tests: List[Callable] = []
        self.integration_tests: List[Callable] = []
        self.performance_tests: List[Callable] = []
        self.security_tests: List[Callable] = []
        
        # Results storage
        self.results_history: List[TestSuiteResult] = []
        
        # Create output directory
        Path(self.config.output_directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("FinancialTestSuite initialized with comprehensive testing capabilities")
    
    def register_unit_test(self, test_func: Callable):
        """Register a unit test function"""
        self.unit_tests.append(test_func)
        
    def register_integration_test(self, test_func: Callable):
        """Register an integration test function"""
        self.integration_tests.append(test_func)
        
    def register_performance_test(self, test_func: Callable):
        """Register a performance test function"""
        self.performance_tests.append(test_func)
        
    def register_security_test(self, test_func: Callable):
        """Register a security test function"""
        self.security_tests.append(test_func)
    
    def run_all_tests(self) -> TestSuiteResult:
        """Run all registered tests"""
        logger.info("Starting comprehensive test suite execution")
        
        suite_start = datetime.now()
        suite_id = f"suite_{int(time.time())}"
        
        # Start coverage tracking
        if self.config.collect_coverage:
            self.coverage_tracker.start_coverage()
        
        all_results = []
        
        # Run different test types
        if self.unit_tests:
            logger.info(f"Running {len(self.unit_tests)} unit tests")
            unit_results = self.executor.execute_tests(self.unit_tests)
            all_results.extend(unit_results)
        
        if self.integration_tests:
            logger.info(f"Running {len(self.integration_tests)} integration tests")
            integration_results = self.executor.execute_tests(self.integration_tests)
            all_results.extend(integration_results)
        
        if self.performance_tests and self.config.performance_benchmarks:
            logger.info(f"Running {len(self.performance_tests)} performance tests")
            performance_results = self._run_performance_tests()
            all_results.extend(performance_results)
        
        if self.security_tests and self.config.security_scanning:
            logger.info(f"Running {len(self.security_tests)} security tests")
            security_results = self._run_security_tests()
            all_results.extend(security_results)
        
        suite_end = datetime.now()
        
        # Stop coverage tracking
        coverage_data = {}
        if self.config.collect_coverage:
            coverage_data = self.coverage_tracker.stop_coverage()
        
        # Calculate results
        suite_result = self._calculate_suite_results(
            suite_id, suite_start, suite_end, all_results, coverage_data
        )
        
        # Store results
        self.results_history.append(suite_result)
        
        # Generate reports
        if self.config.generate_reports:
            self._generate_reports(suite_result)
        
        logger.info(f"Test suite completed: {suite_result.passed_tests}/{suite_result.total_tests} tests passed")
        return suite_result
    
    def run_unit_tests(self) -> TestSuiteResult:
        """Run only unit tests"""
        logger.info("Running unit tests only")
        
        suite_start = datetime.now()
        suite_id = f"unit_suite_{int(time.time())}"
        
        results = self.executor.execute_tests(self.unit_tests)
        
        suite_end = datetime.now()
        
        suite_result = self._calculate_suite_results(
            suite_id, suite_start, suite_end, results, {}
        )
        
        return suite_result
    
    def run_integration_tests(self) -> TestSuiteResult:
        """Run only integration tests"""
        logger.info("Running integration tests only")
        
        suite_start = datetime.now()
        suite_id = f"integration_suite_{int(time.time())}"
        
        results = self.executor.execute_tests(self.integration_tests)
        
        suite_end = datetime.now()
        
        suite_result = self._calculate_suite_results(
            suite_id, suite_start, suite_end, results, {}
        )
        
        return suite_result
    
    def run_performance_tests(self) -> TestSuiteResult:
        """Run only performance tests"""
        logger.info("Running performance tests only")
        
        suite_start = datetime.now()
        suite_id = f"performance_suite_{int(time.time())}"
        
        results = self._run_performance_tests()
        
        suite_end = datetime.now()
        
        suite_result = self._calculate_suite_results(
            suite_id, suite_start, suite_end, results, {}
        )
        
        return suite_result
    
    def run_security_tests(self) -> TestSuiteResult:
        """Run only security tests"""
        logger.info("Running security tests only")
        
        suite_start = datetime.now()
        suite_id = f"security_suite_{int(time.time())}"
        
        results = self._run_security_tests()
        
        suite_end = datetime.now()
        
        suite_result = self._calculate_suite_results(
            suite_id, suite_start, suite_end, results, {}
        )
        
        return suite_result
    
    def _run_performance_tests(self) -> List[TestResult]:
        """Execute performance tests with benchmarking"""
        results = []
        
        for test_func in self.performance_tests:
            start_time = datetime.now()
            
            try:
                # Run benchmark
                benchmark_results = self.performance_tester.benchmark_function(test_func)
                
                # Run stress test
                stress_results = self.performance_tester.stress_test(test_func, duration_seconds=30)
                
                end_time = datetime.now()
                
                result = TestResult(
                    test_id=f"perf_{int(time.time() * 1000)}",
                    test_name=test_func.__name__,
                    test_type=TestType.PERFORMANCE,
                    status=TestStatus.PASSED,
                    execution_time_ms=(end_time - start_time).total_seconds() * 1000,
                    start_time=start_time,
                    end_time=end_time,
                    metadata={
                        'benchmark': benchmark_results,
                        'stress_test': stress_results
                    }
                )
                
                results.append(result)
                
            except Exception as e:
                end_time = datetime.now()
                result = TestResult(
                    test_id=f"perf_{int(time.time() * 1000)}",
                    test_name=test_func.__name__,
                    test_type=TestType.PERFORMANCE,
                    status=TestStatus.ERROR,
                    execution_time_ms=(end_time - start_time).total_seconds() * 1000,
                    start_time=start_time,
                    end_time=end_time,
                    error_message=str(e),
                    stack_trace=traceback.format_exc()
                )
                results.append(result)
        
        return results
    
    def _run_security_tests(self) -> List[TestResult]:
        """Execute security tests with vulnerability scanning"""
        results = []
        
        for test_func in self.security_tests:
            start_time = datetime.now()
            
            try:
                # Run vulnerability scan
                vulnerabilities = self.security_tester.scan_for_vulnerabilities(test_func)
                
                # Run input validation tests
                validation_issues = self.security_tester.test_input_validation(test_func)
                
                end_time = datetime.now()
                
                # Determine status based on findings
                status = TestStatus.PASSED
                if vulnerabilities or validation_issues:
                    status = TestStatus.FAILED
                
                result = TestResult(
                    test_id=f"sec_{int(time.time() * 1000)}",
                    test_name=test_func.__name__,
                    test_type=TestType.SECURITY,
                    status=status,
                    execution_time_ms=(end_time - start_time).total_seconds() * 1000,
                    start_time=start_time,
                    end_time=end_time,
                    metadata={
                        'vulnerabilities': vulnerabilities,
                        'validation_issues': validation_issues
                    }
                )
                
                results.append(result)
                
            except Exception as e:
                end_time = datetime.now()
                result = TestResult(
                    test_id=f"sec_{int(time.time() * 1000)}",
                    test_name=test_func.__name__,
                    test_type=TestType.SECURITY,
                    status=TestStatus.ERROR,
                    execution_time_ms=(end_time - start_time).total_seconds() * 1000,
                    start_time=start_time,
                    end_time=end_time,
                    error_message=str(e),
                    stack_trace=traceback.format_exc()
                )
                results.append(result)
        
        return results
    
    def _calculate_suite_results(
        self, 
        suite_id: str, 
        start_time: datetime, 
        end_time: datetime, 
        test_results: List[TestResult],
        coverage_data: Dict[str, float]
    ) -> TestSuiteResult:
        """Calculate comprehensive suite results"""
        
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.status == TestStatus.PASSED)
        failed_tests = sum(1 for r in test_results if r.status == TestStatus.FAILED)
        skipped_tests = sum(1 for r in test_results if r.status == TestStatus.SKIPPED)
        error_tests = sum(1 for r in test_results if r.status == TestStatus.ERROR)
        
        total_execution_time = sum(r.execution_time_ms for r in test_results)
        average_test_time = total_execution_time / total_tests if total_tests > 0 else 0
        
        # Extract performance metrics
        performance_metrics = {}
        for result in test_results:
            if result.test_type == TestType.PERFORMANCE and result.metadata:
                benchmark = result.metadata.get('benchmark', {})
                performance_metrics[result.test_name] = benchmark
        
        # Extract security findings
        security_findings = []
        for result in test_results:
            if result.test_type == TestType.SECURITY and result.metadata:
                vulnerabilities = result.metadata.get('vulnerabilities', [])
                validation_issues = result.metadata.get('validation_issues', [])
                security_findings.extend(vulnerabilities)
                security_findings.extend(validation_issues)
        
        return TestSuiteResult(
            suite_id=suite_id,
            start_time=start_time,
            end_time=end_time,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            error_tests=error_tests,
            total_execution_time_ms=total_execution_time,
            average_test_time_ms=average_test_time,
            test_results=test_results,
            coverage_data=coverage_data,
            performance_metrics=performance_metrics,
            security_findings=security_findings
        )
    
    def _generate_reports(self, suite_result: TestSuiteResult):
        """Generate test reports in various formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config.report_format == "json":
            self._generate_json_report(suite_result, timestamp)
        elif self.config.report_format == "html":
            self._generate_html_report(suite_result, timestamp)
        elif self.config.report_format == "xml":
            self._generate_xml_report(suite_result, timestamp)
    
    def _generate_json_report(self, suite_result: TestSuiteResult, timestamp: str):
        """Generate JSON test report"""
        report_file = Path(self.config.output_directory) / f"test_report_{timestamp}.json"
        
        # Convert to serializable format
        report_data = {
            'suite_id': suite_result.suite_id,
            'start_time': suite_result.start_time.isoformat(),
            'end_time': suite_result.end_time.isoformat(),
            'summary': {
                'total_tests': suite_result.total_tests,
                'passed_tests': suite_result.passed_tests,
                'failed_tests': suite_result.failed_tests,
                'skipped_tests': suite_result.skipped_tests,
                'error_tests': suite_result.error_tests,
                'success_rate': suite_result.success_rate,
                'total_execution_time_ms': suite_result.total_execution_time_ms,
                'average_test_time_ms': suite_result.average_test_time_ms
            },
            'test_results': [
                {
                    'test_id': r.test_id,
                    'test_name': r.test_name,
                    'test_type': r.test_type.value,
                    'status': r.status.value,
                    'execution_time_ms': r.execution_time_ms,
                    'start_time': r.start_time.isoformat(),
                    'end_time': r.end_time.isoformat(),
                    'error_message': r.error_message,
                    'assertions': r.assertions,
                    'passed_assertions': r.passed_assertions,
                    'failed_assertions': r.failed_assertions,
                    'warnings': r.warnings,
                    'metadata': r.metadata
                }
                for r in suite_result.test_results
            ],
            'coverage_data': suite_result.coverage_data,
            'performance_metrics': suite_result.performance_metrics,
            'security_findings': suite_result.security_findings
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"JSON report generated: {report_file}")
    
    def _generate_html_report(self, suite_result: TestSuiteResult, timestamp: str):
        """Generate HTML test report"""
        report_file = Path(self.config.output_directory) / f"test_report_{timestamp}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Financial Fraud Detection Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .error {{ color: orange; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Financial Fraud Detection Test Report</h1>
            
            <div class="summary">
                <h2>Test Summary</h2>
                <p><strong>Suite ID:</strong> {suite_result.suite_id}</p>
                <p><strong>Execution Time:</strong> {suite_result.start_time} - {suite_result.end_time}</p>
                <p><strong>Total Tests:</strong> {suite_result.total_tests}</p>
                <p><strong>Passed:</strong> <span class="passed">{suite_result.passed_tests}</span></p>
                <p><strong>Failed:</strong> <span class="failed">{suite_result.failed_tests}</span></p>
                <p><strong>Errors:</strong> <span class="error">{suite_result.error_tests}</span></p>
                <p><strong>Success Rate:</strong> {suite_result.success_rate:.1f}%</p>
            </div>
            
            <h2>Test Results</h2>
            <table>
                <tr>
                    <th>Test Name</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Duration (ms)</th>
                    <th>Error Message</th>
                </tr>
        """
        
        for result in suite_result.test_results:
            status_class = result.status.value.lower()
            html_content += f"""
                <tr>
                    <td>{result.test_name}</td>
                    <td>{result.test_type.value}</td>
                    <td class="{status_class}">{result.status.value}</td>
                    <td>{result.execution_time_ms:.2f}</td>
                    <td>{result.error_message or ''}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {report_file}")
    
    def _generate_xml_report(self, suite_result: TestSuiteResult, timestamp: str):
        """Generate XML test report (JUnit format)"""
        report_file = Path(self.config.output_directory) / f"test_report_{timestamp}.xml"
        
        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="FinancialFraudDetectionTests" 
           tests="{suite_result.total_tests}" 
           failures="{suite_result.failed_tests}" 
           errors="{suite_result.error_tests}" 
           time="{suite_result.total_execution_time_ms / 1000:.3f}"
           timestamp="{suite_result.start_time.isoformat()}">
"""
        
        for result in suite_result.test_results:
            xml_content += f"""
    <testcase name="{result.test_name}" 
              classname="{result.test_type.value}" 
              time="{result.execution_time_ms / 1000:.3f}">
"""
            
            if result.status == TestStatus.FAILED:
                xml_content += f"""
        <failure message="{result.error_message or 'Test failed'}">{result.stack_trace or ''}</failure>
"""
            elif result.status == TestStatus.ERROR:
                xml_content += f"""
        <error message="{result.error_message or 'Test error'}">{result.stack_trace or ''}</error>
"""
            
            xml_content += "    </testcase>\n"
        
        xml_content += "</testsuite>"
        
        with open(report_file, 'w') as f:
            f.write(xml_content)
        
        logger.info(f"XML report generated: {report_file}")
    
    def save_results(self, file_path: str):
        """Save test results to file"""
        with open(file_path, 'wb') as f:
            pickle.dump(self.results_history, f)
        logger.info(f"Test results saved to {file_path}")
    
    def load_results(self, file_path: str):
        """Load test results from file"""
        with open(file_path, 'rb') as f:
            self.results_history = pickle.load(f)
        logger.info(f"Test results loaded from {file_path}")
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of all test results"""
        if not self.results_history:
            return {'message': 'No test results available'}
        
        latest_result = self.results_history[-1]
        
        return {
            'latest_run': {
                'suite_id': latest_result.suite_id,
                'total_tests': latest_result.total_tests,
                'passed_tests': latest_result.passed_tests,
                'failed_tests': latest_result.failed_tests,
                'success_rate': latest_result.success_rate,
                'execution_time_ms': latest_result.total_execution_time_ms
            },
            'historical_data': {
                'total_runs': len(self.results_history),
                'average_success_rate': statistics.mean([r.success_rate for r in self.results_history]),
                'average_execution_time_ms': statistics.mean([r.total_execution_time_ms for r in self.results_history])
            }
        }


# Example test cases for demonstration
class FraudDetectionTests(unittest.TestCase):
    """Example fraud detection test cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = {
            'transaction_id': 'test_123',
            'amount': 100.0,
            'merchant': 'Test Merchant',
            'timestamp': datetime.now().isoformat()
        }
    
    def test_basic_fraud_detection(self):
        """Test basic fraud detection functionality"""
        # Mock fraud detection logic
        result = self._mock_fraud_detection(self.test_data)
        self.assertIsNotNone(result)
        self.assertIn('fraud_score', result)
        self.assertTrue(0 <= result['fraud_score'] <= 1)
    
    def test_high_amount_detection(self):
        """Test detection of high amount transactions"""
        high_amount_data = self.test_data.copy()
        high_amount_data['amount'] = 10000.0
        
        result = self._mock_fraud_detection(high_amount_data)
        self.assertGreater(result['fraud_score'], 0.5)
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs"""
        with self.assertRaises(ValueError):
            self._mock_fraud_detection(None)
        
        with self.assertRaises(ValueError):
            self._mock_fraud_detection({})
    
    def _mock_fraud_detection(self, data):
        """Mock fraud detection function for testing"""
        if data is None or not isinstance(data, dict):
            raise ValueError("Invalid input data")
        
        if 'amount' not in data:
            raise ValueError("Missing required field: amount")
        
        # Simple scoring logic for testing
        amount = data['amount']
        fraud_score = min(amount / 10000.0, 1.0)  # Higher amounts = higher score
        
        return {
            'fraud_score': fraud_score,
            'is_fraud': fraud_score > 0.5,
            'transaction_id': data.get('transaction_id', 'unknown')
        }


if __name__ == "__main__":
    # Example usage of the comprehensive test suite
    
    # Configure test suite
    config = TestConfiguration(
        parallel_execution=True,
        max_workers=4,
        collect_coverage=True,
        performance_benchmarks=True,
        security_scanning=True,
        generate_reports=True,
        report_format="json"
    )
    
    # Initialize test suite
    test_suite = FinancialTestSuite(config)
    
    # Register some example tests
    def example_unit_test():
        assert 1 + 1 == 2
    
    def example_performance_test():
        time.sleep(0.001)  # Simulate some work
        return "completed"
    
    def example_security_test(input_data="normal_input"):
        if "malicious" in str(input_data):
            return "error: malicious input detected"
        return "safe"
    
    test_suite.register_unit_test(example_unit_test)
    test_suite.register_performance_test(example_performance_test)
    test_suite.register_security_test(example_security_test)
    
    # Run all tests
    results = test_suite.run_all_tests()
    
    print(f"\nTest Results Summary:")
    print(f"Total Tests: {results.total_tests}")
    print(f"Passed: {results.passed_tests}")
    print(f"Failed: {results.failed_tests}")
    print(f"Success Rate: {results.success_rate:.1f}%")
    
    # Also run standard unittest if called directly
    unittest.main(argv=[''], exit=False)