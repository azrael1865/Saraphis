#!/usr/bin/env python3
"""
Test script for Saraphis Production Integration Testing & Validation System
Tests the complete integration testing framework
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List

# Import the production testing components
from production_testing import (
    IntegrationTestManager,
    TestOrchestrator,
    ComponentValidator,
    SystemValidator,
    PerformanceValidator,
    SecurityValidator,
    TestReportGenerator
)


class Colors:
    """ANSI color codes for output"""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(title: str):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_section(title: str):
    """Print formatted section header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- {title} ---{Colors.END}")


def print_result(label: str, value: Any, indent: int = 0):
    """Print formatted result"""
    indent_str = "  " * indent
    if isinstance(value, bool):
        color = Colors.GREEN if value else Colors.RED
        print(f"{indent_str}{label}: {color}{value}{Colors.END}")
    elif isinstance(value, (int, float)):
        print(f"{indent_str}{label}: {Colors.YELLOW}{value}{Colors.END}")
    elif isinstance(value, str):
        print(f"{indent_str}{label}: {Colors.MAGENTA}{value}{Colors.END}")
    else:
        print(f"{indent_str}{label}: {value}")


def test_integration_test_manager():
    """Test the Integration Test Manager"""
    print_section("Testing Integration Test Manager")
    
    config = {
        'orchestrator': {
            'max_parallel_tests': 10,
            'test_timeout': 300
        },
        'components': {
            'max_retries': 3,
            'critical_components': ['brain_core', 'security_system']
        },
        'system': {
            'max_integration_latency': 200,
            'min_integration_throughput': 100
        },
        'performance': {
            'max_response_time': 1000,
            'min_throughput': 100
        },
        'security': {
            'gdpr_compliance': True,
            'max_auth_failure_rate': 0.01
        },
        'reporting': {
            'report_format': 'json',
            'include_detailed_results': True
        }
    }
    
    # Initialize manager
    manager = IntegrationTestManager(config)
    print_result("Manager initialized", True)
    
    # Execute test suite
    print("\nExecuting full integration test suite...")
    start_time = time.time()
    
    result = manager.execute_test_suite('full_integration', {
        'verbose': True,
        'fail_fast': False
    })
    
    execution_time = time.time() - start_time
    
    # Display results
    print_result("Execution completed", result['success'])
    print_result("Test session ID", result.get('test_session_id', 'N/A'))
    print_result("Execution time", f"{execution_time:.2f} seconds")
    
    if result['success']:
        summary = result['results']['summary']
        print("\nTest Summary:")
        print_result("Total tests", summary['total_tests'], 1)
        print_result("Passed tests", summary['passed_tests'], 1)
        print_result("Failed tests", summary['failed_tests'], 1)
        print_result("Success rate", f"{summary['success_rate']*100:.1f}%", 1)
        print_result("Critical failures", summary['critical_failures'], 1)
        
        # Check test status
        status = manager.get_test_status(result['test_session_id'])
        print_result("Test status", status.get('status', 'unknown'), 1)
    
    return result['success']


def test_test_orchestrator():
    """Test the Test Orchestrator"""
    print_section("Testing Test Orchestrator")
    
    config = {
        'max_parallel_tests': 5,
        'test_timeout': 60,
        'dependency_resolution': True
    }
    
    orchestrator = TestOrchestrator(config)
    print_result("Orchestrator initialized", True)
    
    # Define test plan
    test_plan = {
        'tests': [
            {'id': 'test1', 'dependencies': []},
            {'id': 'test2', 'dependencies': ['test1']},
            {'id': 'test3', 'dependencies': ['test1']},
            {'id': 'test4', 'dependencies': ['test2', 'test3']}
        ]
    }
    
    # Coordinate tests
    result = orchestrator.coordinate_tests('test_suite', test_plan)
    
    print_result("Coordination completed", result['success'])
    print_result("Execution order", result.get('execution_order', []))
    print_result("Total execution time", f"{result.get('total_execution_time', 0):.2f}s")
    
    return result['success']


def test_component_validator():
    """Test the Component Validator"""
    print_section("Testing Component Validator")
    
    config = {
        'validation_timeout': 30,
        'critical_components': ['brain_core', 'security_system']
    }
    
    validator = ComponentValidator(config)
    print_result("Component Validator initialized", True)
    
    # Validate all components
    result = validator.validate_all_components()
    
    print_result("Validation completed", result['success'])
    
    if result['success']:
        # Display component results
        print("\nComponent Validation Results:")
        for component, comp_result in result['component_results'].items():
            status = comp_result['overall_status']
            color = Colors.GREEN if status == 'passed' else Colors.RED
            print(f"  {component}: {color}{status}{Colors.END}")
    
    return result['success']


def test_system_validator():
    """Test the System Validator"""
    print_section("Testing System Validator")
    
    config = {
        'max_integration_latency': 200,
        'min_integration_throughput': 100,
        'max_integration_error_rate': 0.01
    }
    
    validator = SystemValidator(config)
    print_result("System Validator initialized", True)
    
    # Validate system integration
    result = validator.validate_system_integration()
    
    print_result("Validation completed", result['success'])
    
    if result['success']:
        aggregated = result['aggregated_results']
        print("\nSystem Integration Health:")
        print_result("Overall health score", f"{aggregated['overall_integration_health']*100:.1f}%", 1)
        print_result("Integration issues", aggregated['integration_issues'], 1)
    
    return result['success']


def test_performance_validator():
    """Test the Performance Validator"""
    print_section("Testing Performance Validator")
    
    config = {
        'max_response_time': 1000,
        'min_throughput': 100,
        'max_cpu_usage': 80,
        'max_memory_usage': 80
    }
    
    validator = PerformanceValidator(config)
    print_result("Performance Validator initialized", True)
    
    # Validate performance
    result = validator.validate_performance_integration()
    
    print_result("Validation completed", result['success'])
    
    if result['success']:
        aggregated = result['aggregated_results']
        print("\nPerformance Health:")
        print_result("Overall performance health", f"{aggregated['overall_performance_health']*100:.1f}%", 1)
        print_result("Performance violations", aggregated['performance_violations'], 1)
        print_result("Bottlenecks identified", aggregated['bottlenecks_identified'], 1)
    
    return result['success']


def test_security_validator():
    """Test the Security Validator"""
    print_section("Testing Security Validator")
    
    config = {
        'gdpr_compliance': True,
        'max_auth_failure_rate': 0.01,
        'min_password_entropy': 60,
        'min_key_length': 256
    }
    
    validator = SecurityValidator(config)
    print_result("Security Validator initialized", True)
    
    # Validate security
    result = validator.validate_security_integration()
    
    print_result("Validation completed", result['success'])
    
    if result['success']:
        aggregated = result['aggregated_results']
        print("\nSecurity Assessment:")
        print_result("Overall security score", f"{aggregated['overall_security_score']*100:.1f}%", 1)
        print_result("Vulnerabilities found", aggregated['vulnerabilities_found'], 1)
        print_result("Compliance violations", aggregated['compliance_violations'], 1)
    
    return result['success']


def test_report_generator():
    """Test the Report Generator"""
    print_section("Testing Report Generator")
    
    config = {
        'report_format': 'json',
        'include_detailed_results': True,
        'include_recommendations': True,
        'report_directory': '/tmp/saraphis_test_reports'
    }
    
    generator = TestReportGenerator(config)
    print_result("Report Generator initialized", True)
    
    # Generate sample test results
    test_results = {
        'summary': {
            'total_tests': 100,
            'passed_tests': 92,
            'failed_tests': 8,
            'skipped_tests': 0,
            'success_rate': 0.92,
            'critical_failures': 2,
            'performance_issues': 5,
            'security_issues': 1,
            'integration_issues': 3
        },
        'orchestration': {'success': True},
        'components': {'success': True},
        'system': {'success': True},
        'performance': {'success': True},
        'security': {'success': True}
    }
    
    # Generate report
    report = generator.generate_report('test_session_123', test_results, time.time() - 300)
    
    if 'error' not in report:
        print_result("Report generated", True)
        print_result("Report file", report.get('report_file', 'N/A'))
        
        # Display executive summary
        if 'executive_summary' in report:
            summary = report['executive_summary']
            print("\nExecutive Summary:")
            print_result("Overall status", summary['overall_status'], 1)
            print_result("Success rate", summary['test_statistics']['success_rate'], 1)
            print_result("Risk level", summary['risk_assessment']['risk_level'], 1)
        
        # Display recommendations
        if 'recommendations' in report and report['recommendations']:
            print("\nTop Recommendations:")
            for i, rec in enumerate(report['recommendations'][:3]):
                print(f"  {i+1}. [{rec['priority'].upper()}] {rec['title']}")
        
        return True
    else:
        print_result("Report generation failed", False)
        return False


def test_end_to_end_workflow():
    """Test complete end-to-end workflow"""
    print_section("Testing End-to-End Workflow")
    
    # Configuration for all components
    config = {
        'orchestrator': {'max_parallel_tests': 10},
        'components': {'critical_components': ['brain_core']},
        'system': {'max_integration_latency': 200},
        'performance': {'max_response_time': 1000},
        'security': {'gdpr_compliance': True},
        'reporting': {'include_detailed_results': True}
    }
    
    # Initialize manager
    manager = IntegrationTestManager(config)
    
    # Execute different test suites
    test_suites = ['component_integration', 'performance_integration', 'security_integration']
    
    all_results = []
    for suite in test_suites:
        print(f"\n  Executing {suite} test suite...")
        result = manager.execute_test_suite(suite)
        all_results.append(result)
        
        status_color = Colors.GREEN if result['success'] else Colors.RED
        print(f"  {suite}: {status_color}{'PASSED' if result['success'] else 'FAILED'}{Colors.END}")
    
    # Get overall metrics
    metrics = manager.get_test_metrics()
    
    if metrics['success']:
        print("\nOverall Test Metrics:")
        summary = metrics.get('summary', {})
        print_result("Total executions", summary.get('total_executions', 0), 1)
        print_result("Overall success rate", f"{summary.get('overall_success_rate', 0)*100:.1f}%", 1)
        print_result("Critical failure rate", f"{summary.get('critical_failure_rate', 0)*100:.1f}%", 1)
    
    return all(r['success'] for r in all_results)


def main():
    """Main test function"""
    print_header("Saraphis Production Integration Testing System Test")
    
    start_time = time.time()
    
    # Track test results
    test_results = {}
    
    # Run individual component tests
    tests = [
        ("Integration Test Manager", test_integration_test_manager),
        ("Test Orchestrator", test_test_orchestrator),
        ("Component Validator", test_component_validator),
        ("System Validator", test_system_validator),
        ("Performance Validator", test_performance_validator),
        ("Security Validator", test_security_validator),
        ("Report Generator", test_report_generator),
        ("End-to-End Workflow", test_end_to_end_workflow)
    ]
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            test_results[test_name] = success
        except Exception as e:
            print(f"{Colors.RED}Test failed with error: {e}{Colors.END}")
            test_results[test_name] = False
    
    # Summary
    print_header("Test Summary")
    
    total_tests = len(test_results)
    passed_tests = sum(1 for success in test_results.values() if success)
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {Colors.GREEN}{passed_tests}{Colors.END}")
    print(f"Failed: {Colors.RED}{failed_tests}{Colors.END}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    # Individual results
    print("\nIndividual Test Results:")
    for test_name, success in test_results.items():
        status_color = Colors.GREEN if success else Colors.RED
        status_text = "PASSED" if success else "FAILED"
        print(f"  {test_name}: {status_color}{status_text}{Colors.END}")
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    # Overall result
    all_passed = all(test_results.values())
    if all_passed:
        print(f"\n{Colors.GREEN}{Colors.BOLD}ALL TESTS PASSED!{Colors.END}")
        print("The Production Integration Testing System is working correctly.")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}SOME TESTS FAILED!{Colors.END}")
        print("Please check the failed tests and fix any issues.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())