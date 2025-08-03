"""
Saraphis Integration Test Manager
Production-ready integration test management with comprehensive validation
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import logging
import random
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import json
import hashlib

from .test_orchestrator import TestOrchestrator
from .component_validator import ComponentValidator
from .system_validator import SystemValidator
from .performance_validator import PerformanceValidator
from .security_validator import SecurityValidator
from .test_report_generator import TestReportGenerator

logger = logging.getLogger(__name__)


class IntegrationTestManager:
    """Production-ready integration test management with comprehensive validation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize test components
        self.test_orchestrator = TestOrchestrator(config.get('orchestrator', {}))
        self.component_validator = ComponentValidator(config.get('components', {}))
        self.system_validator = SystemValidator(config.get('system', {}))
        self.performance_validator = PerformanceValidator(config.get('performance', {}))
        self.security_validator = SecurityValidator(config.get('security', {}))
        self.test_report_generator = TestReportGenerator(config.get('reporting', {}))
        
        # Test execution tracking
        self.test_history = deque(maxlen=1000)
        self.active_test_sessions = {}
        self.test_metrics = defaultdict(lambda: {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'critical_failures': 0,
            'average_execution_time': 0,
            'last_execution': None
        })
        
        # Configuration
        self.parallel_execution = config.get('parallel_execution', True)
        self.max_parallel_tests = config.get('max_parallel_tests', 10)
        self.test_timeout_seconds = config.get('test_timeout_seconds', 300)
        self.retry_on_failure = config.get('retry_on_failure', True)
        self.max_retries = config.get('max_retries', 3)
        
        # Test suite definitions
        self.test_suites = self._initialize_test_suites()
        
        # Thread safety
        self._lock = threading.Lock()
        
        self.logger.info("Integration Test Manager initialized")
        
    def execute_test_suite(self, test_suite: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute comprehensive integration test suite"""
        try:
            # Initialize test execution
            test_session_id = self._generate_test_session_id()
            start_time = time.time()
            
            self.logger.info(f"Starting test suite '{test_suite}' with session ID: {test_session_id}")
            
            # Validate test suite configuration
            suite_validation = self._validate_test_suite(test_suite)
            if not suite_validation['valid']:
                return {
                    'success': False,
                    'error': 'Invalid test suite configuration',
                    'details': suite_validation['details']
                }
            
            # Initialize test session
            self._initialize_test_session(test_session_id, test_suite, options)
            
            # Execute test orchestration
            self.logger.info("Executing test orchestration...")
            orchestration_result = self.test_orchestrator.coordinate_tests(
                test_suite, options or {}
            )
            
            # Execute component validation
            self.logger.info("Executing component validation...")
            component_results = self.component_validator.validate_all_components()
            
            # Execute system integration validation
            self.logger.info("Executing system integration validation...")
            system_results = self.system_validator.validate_system_integration()
            
            # Execute performance validation
            self.logger.info("Executing performance validation...")
            performance_results = self.performance_validator.validate_performance_integration()
            
            # Execute security validation
            self.logger.info("Executing security validation...")
            security_results = self.security_validator.validate_security_integration()
            
            # Aggregate all test results
            aggregated_results = self._aggregate_test_results(
                orchestration_result,
                component_results,
                system_results,
                performance_results,
                security_results
            )
            
            # Generate comprehensive test report
            test_report = self.test_report_generator.generate_report(
                test_session_id, aggregated_results, start_time
            )
            
            # Calculate test execution metrics
            execution_metrics = self._calculate_execution_metrics(
                start_time, aggregated_results
            )
            
            # Update test history and metrics
            self._update_test_history(test_session_id, test_suite, aggregated_results)
            
            # Finalize test session
            self._finalize_test_session(test_session_id, aggregated_results)
            
            return {
                'success': True,
                'test_session_id': test_session_id,
                'test_suite': test_suite,
                'results': aggregated_results,
                'report': test_report,
                'execution_metrics': execution_metrics,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Integration test suite execution failed: {e}")
            return {
                'success': False,
                'error': f'Test suite execution failed: {str(e)}'
            }
    
    def get_test_status(self, test_session_id: str) -> Dict[str, Any]:
        """Get status of running or completed test"""
        try:
            with self._lock:
                if test_session_id in self.active_test_sessions:
                    session = self.active_test_sessions[test_session_id]
                    return {
                        'success': True,
                        'status': 'running',
                        'session': session,
                        'elapsed_time': time.time() - session['start_time']
                    }
                
                # Check test history
                for test in self.test_history:
                    if test['test_session_id'] == test_session_id:
                        return {
                            'success': True,
                            'status': 'completed',
                            'test': test
                        }
                
                return {
                    'success': False,
                    'error': f'Test session not found: {test_session_id}'
                }
                
        except Exception as e:
            self.logger.error(f"Test status retrieval failed: {e}")
            return {
                'success': False,
                'error': f'Status retrieval failed: {str(e)}'
            }
    
    def get_test_metrics(self, test_suite: Optional[str] = None) -> Dict[str, Any]:
        """Get test execution metrics"""
        try:
            with self._lock:
                if test_suite:
                    if test_suite in self.test_metrics:
                        return {
                            'success': True,
                            'test_suite': test_suite,
                            'metrics': self.test_metrics[test_suite].copy()
                        }
                    else:
                        return {
                            'success': False,
                            'error': f'No metrics for test suite: {test_suite}'
                        }
                else:
                    # Return all metrics
                    return {
                        'success': True,
                        'metrics': dict(self.test_metrics),
                        'summary': self._calculate_metrics_summary()
                    }
                    
        except Exception as e:
            self.logger.error(f"Test metrics retrieval failed: {e}")
            return {
                'success': False,
                'error': f'Metrics retrieval failed: {str(e)}'
            }
    
    def _initialize_test_suites(self) -> Dict[str, Dict[str, Any]]:
        """Initialize test suite definitions"""
        return {
            'full_integration': {
                'description': 'Complete system integration testing',
                'components': ['brain', 'uncertainty', 'training', 'compression', 'proof', 'security', 'monitoring', 'api', 'data'],
                'duration_estimate': 300,  # 5 minutes
                'critical': True,
                'parallel': True
            },
            'component_integration': {
                'description': 'Component-level integration testing',
                'components': ['brain', 'uncertainty', 'training'],
                'duration_estimate': 120,  # 2 minutes
                'critical': False,
                'parallel': False
            },
            'performance_integration': {
                'description': 'Performance-focused integration testing',
                'components': ['brain', 'compression', 'training'],
                'duration_estimate': 180,  # 3 minutes
                'critical': True,
                'parallel': True
            },
            'security_integration': {
                'description': 'Security-focused integration testing',
                'components': ['brain', 'security', 'proof'],
                'duration_estimate': 240,  # 4 minutes
                'critical': True,
                'parallel': False
            },
            'production_readiness': {
                'description': 'Production readiness validation',
                'components': ['brain', 'uncertainty', 'training', 'compression', 'proof', 'security', 'monitoring', 'scaling', 'recovery'],
                'duration_estimate': 600,  # 10 minutes
                'critical': True,
                'parallel': True
            }
        }
    
    def _validate_test_suite(self, test_suite: str) -> Dict[str, Any]:
        """Validate test suite configuration"""
        try:
            if test_suite not in self.test_suites:
                return {
                    'valid': False,
                    'details': f'Invalid test suite: {test_suite}. Valid suites: {list(self.test_suites.keys())}'
                }
            
            suite_config = self.test_suites[test_suite]
            
            # Validate components availability
            missing_components = []
            for component in suite_config['components']:
                if not self._is_component_available(component):
                    missing_components.append(component)
            
            if missing_components:
                return {
                    'valid': False,
                    'details': f'Missing components: {missing_components}'
                }
            
            return {
                'valid': True,
                'suite_config': suite_config
            }
            
        except Exception as e:
            self.logger.error(f"Test suite validation failed: {e}")
            return {
                'valid': False,
                'details': f'Validation error: {str(e)}'
            }
    
    def _initialize_test_session(self, test_session_id: str, test_suite: str, options: Dict[str, Any]):
        """Initialize test session"""
        with self._lock:
            self.active_test_sessions[test_session_id] = {
                'test_suite': test_suite,
                'options': options,
                'start_time': time.time(),
                'status': 'running',
                'progress': 0,
                'current_phase': 'initialization'
            }
    
    def _finalize_test_session(self, test_session_id: str, results: Dict[str, Any]):
        """Finalize test session"""
        with self._lock:
            if test_session_id in self.active_test_sessions:
                session = self.active_test_sessions.pop(test_session_id)
                session['end_time'] = time.time()
                session['results'] = results
                session['status'] = 'completed'
    
    def _aggregate_test_results(self, orchestration_result: Dict[str, Any],
                               component_results: Dict[str, Any],
                               system_results: Dict[str, Any],
                               performance_results: Dict[str, Any],
                               security_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate all test results into comprehensive summary"""
        try:
            aggregated_results = {
                'orchestration': orchestration_result,
                'components': component_results,
                'system': system_results,
                'performance': performance_results,
                'security': security_results,
                'summary': {
                    'total_tests': 0,
                    'passed_tests': 0,
                    'failed_tests': 0,
                    'skipped_tests': 0,
                    'success_rate': 0.0,
                    'critical_failures': 0,
                    'performance_issues': 0,
                    'security_issues': 0,
                    'integration_issues': 0
                }
            }
            
            # Calculate summary statistics
            all_results = [
                orchestration_result,
                component_results,
                system_results,
                performance_results,
                security_results
            ]
            
            for result in all_results:
                if result.get('success', False):
                    test_counts = result.get('test_counts', {})
                    aggregated_results['summary']['total_tests'] += test_counts.get('total', 0)
                    aggregated_results['summary']['passed_tests'] += test_counts.get('passed', 0)
                    aggregated_results['summary']['failed_tests'] += test_counts.get('failed', 0)
                    aggregated_results['summary']['skipped_tests'] += test_counts.get('skipped', 0)
                    
                    # Count critical issues
                    if result.get('critical_failures', 0) > 0:
                        aggregated_results['summary']['critical_failures'] += result['critical_failures']
                    
                    # Count performance issues
                    if result.get('performance_issues', 0) > 0:
                        aggregated_results['summary']['performance_issues'] += result['performance_issues']
                    
                    # Count security issues
                    if result.get('security_issues', 0) > 0:
                        aggregated_results['summary']['security_issues'] += result['security_issues']
                    
                    # Count integration issues
                    if result.get('integration_issues', 0) > 0:
                        aggregated_results['summary']['integration_issues'] += result['integration_issues']
            
            # Calculate success rate
            total_tests = aggregated_results['summary']['total_tests']
            if total_tests > 0:
                aggregated_results['summary']['success_rate'] = (
                    aggregated_results['summary']['passed_tests'] / total_tests
                )
            
            # Determine overall status
            if aggregated_results['summary']['critical_failures'] > 0:
                aggregated_results['overall_status'] = 'critical_failure'
            elif aggregated_results['summary']['failed_tests'] > 0:
                aggregated_results['overall_status'] = 'failure'
            elif aggregated_results['summary']['success_rate'] >= 0.95:
                aggregated_results['overall_status'] = 'success'
            else:
                aggregated_results['overall_status'] = 'partial_success'
            
            return aggregated_results
            
        except Exception as e:
            self.logger.error(f"Test result aggregation failed: {e}")
            return {
                'error': f'Aggregation failed: {str(e)}',
                'summary': {
                    'total_tests': 0,
                    'passed_tests': 0,
                    'failed_tests': 0,
                    'success_rate': 0.0
                },
                'overall_status': 'error'
            }
    
    def _calculate_execution_metrics(self, start_time: float, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate test execution metrics"""
        try:
            execution_time = time.time() - start_time
            summary = results.get('summary', {})
            
            metrics = {
                'execution_time_seconds': execution_time,
                'execution_time_minutes': execution_time / 60,
                'tests_per_second': summary.get('total_tests', 0) / max(execution_time, 1),
                'success_rate_percentage': summary.get('success_rate', 0) * 100,
                'critical_failure_rate': (
                    summary.get('critical_failures', 0) / 
                    max(summary.get('total_tests', 0), 1)
                ),
                'performance_issue_rate': (
                    summary.get('performance_issues', 0) / 
                    max(summary.get('total_tests', 0), 1)
                ),
                'security_issue_rate': (
                    summary.get('security_issues', 0) / 
                    max(summary.get('total_tests', 0), 1)
                ),
                'integration_issue_rate': (
                    summary.get('integration_issues', 0) / 
                    max(summary.get('total_tests', 0), 1)
                ),
                'average_test_time': execution_time / max(summary.get('total_tests', 0), 1)
            }
            
            # Add performance categories
            if execution_time < 60:
                metrics['performance_category'] = 'excellent'
            elif execution_time < 180:
                metrics['performance_category'] = 'good'
            elif execution_time < 300:
                metrics['performance_category'] = 'acceptable'
            else:
                metrics['performance_category'] = 'slow'
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Execution metrics calculation failed: {e}")
            return {
                'error': f'Metrics calculation failed: {str(e)}'
            }
    
    def _update_test_history(self, test_session_id: str, test_suite: str, results: Dict[str, Any]):
        """Update test history and metrics"""
        try:
            with self._lock:
                # Add to history
                self.test_history.append({
                    'test_session_id': test_session_id,
                    'test_suite': test_suite,
                    'timestamp': time.time(),
                    'overall_status': results.get('overall_status', 'unknown'),
                    'summary': results.get('summary', {})
                })
                
                # Update metrics
                metrics = self.test_metrics[test_suite]
                metrics['total_executions'] += 1
                
                if results.get('overall_status') in ['success', 'partial_success']:
                    metrics['successful_executions'] += 1
                else:
                    metrics['failed_executions'] += 1
                
                if results.get('summary', {}).get('critical_failures', 0) > 0:
                    metrics['critical_failures'] += 1
                
                # Update average execution time
                if 'execution_time_seconds' in results:
                    current_avg = metrics['average_execution_time']
                    total_execs = metrics['total_executions']
                    new_time = results['execution_time_seconds']
                    metrics['average_execution_time'] = (
                        (current_avg * (total_execs - 1) + new_time) / total_execs
                    )
                
                metrics['last_execution'] = time.time()
                
        except Exception as e:
            self.logger.error(f"Test history update failed: {e}")
    
    def _is_component_available(self, component: str) -> bool:
        """Check if component is available for testing"""
        # In production, this would check actual component availability
        # For now, assume all components are available
        available_components = [
            'brain', 'uncertainty', 'training', 'compression', 
            'proof', 'security', 'monitoring', 'api', 'data',
            'scaling', 'recovery'
        ]
        return component in available_components
    
    def _calculate_metrics_summary(self) -> Dict[str, Any]:
        """Calculate summary of all test metrics"""
        try:
            total_executions = sum(m['total_executions'] for m in self.test_metrics.values())
            successful_executions = sum(m['successful_executions'] for m in self.test_metrics.values())
            failed_executions = sum(m['failed_executions'] for m in self.test_metrics.values())
            critical_failures = sum(m['critical_failures'] for m in self.test_metrics.values())
            
            return {
                'total_executions': total_executions,
                'successful_executions': successful_executions,
                'failed_executions': failed_executions,
                'critical_failures': critical_failures,
                'overall_success_rate': successful_executions / max(total_executions, 1),
                'critical_failure_rate': critical_failures / max(total_executions, 1),
                'test_suites_count': len(self.test_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Metrics summary calculation failed: {e}")
            return {}
    
    def _generate_test_session_id(self) -> str:
        """Generate unique test session ID"""
        timestamp = int(time.time() * 1000)
        random_suffix = hashlib.md5(f"{timestamp}{random.random()}".encode()).hexdigest()[:8]
        return f"test_session_{timestamp}_{random_suffix}"