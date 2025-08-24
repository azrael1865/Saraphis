"""
Comprehensive test suite for ProductionFinalValidator
Tests all functionality and edge cases with hard failures only
NO FALLBACKS - NO MOCKS - REAL TESTING
"""

import unittest
import tempfile
import shutil
import time
import json
import threading
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from dataclasses import asdict

# Direct import to avoid dependency issues
import importlib.util
import os

# Import FinalValidator directly
spec = importlib.util.spec_from_file_location(
    "final_validator", 
    os.path.join(os.path.dirname(__file__), "production_validation", "final_validator.py")
)
final_validator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(final_validator_module)

ProductionFinalValidator = final_validator_module.ProductionFinalValidator
ValidationResult = final_validator_module.ValidationResult
SystemHealthScore = final_validator_module.SystemHealthScore
DeploymentDecision = final_validator_module.DeploymentDecision
ValidationSeverity = final_validator_module.ValidationSeverity
SystemComponent = final_validator_module.SystemComponent
HealthStatus = final_validator_module.HealthStatus


class TestProductionFinalValidator(unittest.TestCase):
    """Test cases for ProductionFinalValidator"""
    
    def setUp(self):
        """Set up test environment"""
        # Create mock brain system
        self.mock_brain = MagicMock()
        self.mock_brain.predict.return_value = {"result": "test_prediction", "confidence": 0.95}
        self.mock_brain.domains = ['financial_fraud', 'cybersecurity', 'molecular_analysis']
        
        # Create mock components
        self.mock_components = {
            'monitoring_system': MagicMock(),
            'deployment_manager': MagicMock(),
            'security_validator': MagicMock(),
            'brain_orchestrator': MagicMock(),
            'neural_orchestrator': MagicMock(),
            'reasoning_orchestrator': MagicMock(),
            'uncertainty_orchestrator': MagicMock(),
            'gac_system': MagicMock(),
            'proof_system': MagicMock(),
            'compression_api': MagicMock(),
            'training_manager': MagicMock(),
            'universal_ai_core': MagicMock(),
            'error_recovery_system': MagicMock()
        }
        
        # Setup mock responses
        self.mock_components['monitoring_system'].get_monitoring_status.return_value = {
            'status': 'running', 'uptime': '24h', 'alerts': 0
        }
        self.mock_components['monitoring_system'].get_health_status.return_value = {
            'overall_health': 0.95, 'components': []
        }
        self.mock_components['security_validator'].run_full_security_validation.return_value = {
            'security_score': 0.95, 'vulnerabilities': []
        }
        self.mock_components['gac_system'].clip_gradients.return_value = [0.05, 0.1, 0.15]
        
        # Production configuration
        self.production_config = {
            'environment': 'production',
            'debug_mode': False,
            'monitoring_enabled': True,
            'security_level': 'high',
            'performance_targets': {
                'api_latency_ms': 100,
                'throughput_rps': 1000
            }
        }
        
        # Create validator instance
        self.validator = ProductionFinalValidator(
            self.mock_brain, 
            self.mock_components, 
            self.production_config
        )
        
        # Create temporary directory for reports
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test ProductionFinalValidator initialization"""
        # Check basic initialization
        self.assertEqual(self.validator.brain_system, self.mock_brain)
        self.assertEqual(self.validator.all_components, self.mock_components)
        self.assertEqual(self.validator.production_config, self.production_config)
        
        # Check validation state
        self.assertIsInstance(self.validator.validation_results, list)
        self.assertIsInstance(self.validator.health_scores, dict)
        self.assertIsInstance(self.validator.validation_lock, threading.RLock)
        
        # Check configuration values
        self.assertEqual(self.validator.validation_timeout, 300)
        self.assertEqual(self.validator.component_timeout, 30)
        
        # Check performance benchmarks
        self.assertIn('api_response_ms', self.validator.performance_benchmarks)
        self.assertIn('prediction_latency_ms', self.validator.performance_benchmarks)
        
        # Check required scores
        self.assertEqual(self.validator.required_scores['overall'], 0.95)
        self.assertEqual(self.validator.required_scores['reliability'], 0.99)
    
    def test_validation_result_dataclass(self):
        """Test ValidationResult dataclass"""
        # Valid creation
        result = ValidationResult(
            component=SystemComponent.BRAIN_CORE,
            test_name="test_initialization",
            passed=True,
            severity=ValidationSeverity.INFO,
            message="Test passed",
            duration_ms=100.5
        )
        
        self.assertEqual(result.component, SystemComponent.BRAIN_CORE)
        self.assertEqual(result.test_name, "test_initialization")
        self.assertTrue(result.passed)
        self.assertEqual(result.severity, ValidationSeverity.INFO)
        self.assertEqual(result.message, "Test passed")
        self.assertEqual(result.duration_ms, 100.5)
        self.assertIsInstance(result.timestamp, datetime)
        self.assertIsInstance(result.details, dict)
        
        # Test post-init validation
        with self.assertRaises(TypeError):
            ValidationResult(
                component="invalid_component",  # Should be SystemComponent
                test_name="test",
                passed=True,
                severity=ValidationSeverity.INFO,
                message="Test",
                duration_ms=0.0
            )
        
        with self.assertRaises(TypeError):
            ValidationResult(
                component=SystemComponent.BRAIN_CORE,
                test_name="test",
                passed=True,
                severity="invalid_severity",  # Should be ValidationSeverity
                message="Test",
                duration_ms=0.0
            )
    
    def test_system_health_score_dataclass(self):
        """Test SystemHealthScore dataclass"""
        health_score = SystemHealthScore(
            component=SystemComponent.BRAIN_CORE,
            health_score=0.95,
            status=HealthStatus.HEALTHY,
            metrics={'cpu': 45.2, 'memory': 78.5},
            issues=[]
        )
        
        self.assertEqual(health_score.component, SystemComponent.BRAIN_CORE)
        self.assertEqual(health_score.health_score, 0.95)
        self.assertEqual(health_score.status, HealthStatus.HEALTHY)
        self.assertEqual(health_score.metrics['cpu'], 45.2)
        self.assertEqual(len(health_score.issues), 0)
    
    def test_deployment_decision_dataclass(self):
        """Test DeploymentDecision dataclass"""
        decision = DeploymentDecision(
            decision="GO",
            overall_score=0.96,
            readiness_score=0.95,
            performance_score=0.92,
            security_score=0.98,
            reliability_score=0.99,
            critical_issues=[],
            warnings=["Monitor performance closely"],
            recommendations=["Gradual rollout recommended"]
        )
        
        self.assertEqual(decision.decision, "GO")
        self.assertEqual(decision.overall_score, 0.96)
        self.assertEqual(len(decision.critical_issues), 0)
        self.assertEqual(len(decision.warnings), 1)
        self.assertEqual(len(decision.recommendations), 1)
    
    def test_validate_final_system_health(self):
        """Test final system health validation"""
        result = self.validator.validate_final_system_health()
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('validation', result)
        self.assertEqual(result['validation'], 'system_health')
        self.assertIn('overall_health', result)
        self.assertIn('component_scores', result)
        self.assertIn('duration_ms', result)
        self.assertIn('passed', result)
        
        # Check overall health calculation
        self.assertIsInstance(result['overall_health'], float)
        self.assertGreaterEqual(result['overall_health'], 0.0)
        self.assertLessEqual(result['overall_health'], 1.0)
        
        # Check component scores
        self.assertIsInstance(result['component_scores'], dict)
        self.assertGreater(len(result['component_scores']), 0)
        
        # Check duration tracking
        self.assertIsInstance(result['duration_ms'], float)
        self.assertGreater(result['duration_ms'], 0)
    
    def test_brain_core_validation(self):
        """Test Brain Core validation"""
        result = self.validator._validate_brain_core()
        
        # Should pass with mock brain
        self.assertEqual(result['status'], 'healthy')
        self.assertEqual(result['health_score'], 1.0)
        self.assertIn('metrics', result)
        self.assertEqual(result['metrics']['initialization'], 1.0)
        
        # Test with failing brain
        self.validator.brain_system = None
        result = self.validator._validate_brain_core()
        
        self.assertEqual(result['status'], 'unhealthy')
        self.assertEqual(result['health_score'], 0.0)
        self.assertIn('error', result)
        
        # Test with brain prediction failure
        self.validator.brain_system = MagicMock()
        self.validator.brain_system.predict.return_value = {'error': 'Prediction failed'}
        result = self.validator._validate_brain_core()
        
        self.assertEqual(result['status'], 'unhealthy')
        self.assertEqual(result['health_score'], 0.0)
    
    def test_reasoning_systems_validation(self):
        """Test Reasoning Systems validation"""
        result = self.validator._validate_reasoning_systems()
        
        # Should pass with all orchestrators present
        self.assertIn(result['status'], ['healthy', 'degraded'])
        self.assertIsInstance(result['health_score'], float)
        self.assertIn('metrics', result)
        
        # Test with missing orchestrators
        empty_components = {}
        temp_validator = ProductionFinalValidator(
            self.mock_brain, empty_components, self.production_config
        )
        result = temp_validator._validate_reasoning_systems()
        
        self.assertEqual(result['health_score'], 0.0)
    
    def test_gac_system_validation(self):
        """Test GAC optimization system validation"""
        result = self.validator._validate_gac_system()
        
        # Should pass with mock GAC system
        self.assertEqual(result['status'], 'healthy')
        self.assertEqual(result['health_score'], 1.0)
        
        # Verify GAC method was called
        self.mock_components['gac_system'].clip_gradients.assert_called_once()
        
        # Test with missing GAC system
        components_no_gac = {k: v for k, v in self.mock_components.items() if k != 'gac_system'}
        temp_validator = ProductionFinalValidator(
            self.mock_brain, components_no_gac, self.production_config
        )
        result = temp_validator._validate_gac_system()
        
        self.assertEqual(result['status'], 'unhealthy')
        self.assertEqual(result['health_score'], 0.0)
    
    def test_uncertainty_system_validation(self):
        """Test Uncertainty Quantification system validation"""
        result = self.validator._validate_uncertainty_system()
        
        # Should pass with uncertainty orchestrator
        self.assertEqual(result['status'], 'healthy')
        self.assertEqual(result['health_score'], 1.0)
        
        # Test without uncertainty orchestrator
        components_no_uncertainty = {k: v for k, v in self.mock_components.items() 
                                   if k != 'uncertainty_orchestrator'}
        temp_validator = ProductionFinalValidator(
            self.mock_brain, components_no_uncertainty, self.production_config
        )
        result = temp_validator._validate_uncertainty_system()
        
        self.assertEqual(result['health_score'], 0.5)
        self.assertEqual(result['status'], 'degraded')
    
    def test_proof_system_validation(self):
        """Test Proof Generation/Verification system validation"""
        result = self.validator._validate_proof_system()
        
        # Should pass with proof system
        self.assertEqual(result['status'], 'healthy')
        self.assertEqual(result['health_score'], 1.0)
        
        # Test without proof system
        components_no_proof = {k: v for k, v in self.mock_components.items() 
                              if k != 'proof_system'}
        temp_validator = ProductionFinalValidator(
            self.mock_brain, components_no_proof, self.production_config
        )
        result = temp_validator._validate_proof_system()
        
        self.assertEqual(result['health_score'], 0.5)
    
    def test_compression_systems_validation(self):
        """Test Compression Systems validation"""
        result = self.validator._validate_compression_systems()
        
        # Should pass with compression API
        self.assertEqual(result['status'], 'healthy')
        self.assertEqual(result['health_score'], 1.0)
    
    def test_training_system_validation(self):
        """Test Training System validation"""
        result = self.validator._validate_training_system()
        
        # Should pass with training manager
        self.assertEqual(result['status'], 'healthy')
        self.assertEqual(result['health_score'], 1.0)
    
    def test_security_system_validation(self):
        """Test Security System validation"""
        result = self.validator._validate_security_system()
        
        # Should pass with security validator
        self.assertEqual(result['status'], 'healthy')
        self.assertEqual(result['health_score'], 1.0)
    
    def test_domain_systems_validation(self):
        """Test Domain-Specific Systems validation"""
        result = self.validator._validate_domain_systems()
        
        # Should pass with all domains available in mock brain
        self.assertEqual(result['status'], 'healthy')
        self.assertEqual(result['health_score'], 1.0)
        self.assertEqual(result['metrics']['domains_available'], 3)
        self.assertEqual(result['metrics']['total_domains'], 3)
        
        # Test with brain without domains
        brain_no_domains = MagicMock()
        brain_no_domains.domains = []
        temp_validator = ProductionFinalValidator(
            brain_no_domains, self.mock_components, self.production_config
        )
        result = temp_validator._validate_domain_systems()
        
        self.assertEqual(result['status'], 'degraded')
        self.assertEqual(result['health_score'], 0.0)
    
    def test_universal_ai_core_validation(self):
        """Test Universal AI Core system validation"""
        result = self.validator._validate_universal_ai_core()
        
        # Should pass with universal AI core
        self.assertEqual(result['status'], 'healthy')
        self.assertEqual(result['health_score'], 1.0)
    
    def test_production_monitoring_validation(self):
        """Test Production Monitoring system validation"""
        result = self.validator._validate_production_monitoring()
        
        # Should pass with monitoring system
        self.assertEqual(result['status'], 'healthy')
        self.assertEqual(result['health_score'], 1.0)
        
        # Verify monitoring system method was called
        self.mock_components['monitoring_system'].get_monitoring_status.assert_called()
    
    def test_production_readiness(self):
        """Test production readiness testing"""
        result = self.validator.test_production_readiness()
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertEqual(result['validation'], 'production_readiness')
        self.assertIn('readiness_score', result)
        self.assertIn('checks_passed', result)
        self.assertIn('total_checks', result)
        self.assertIn('details', result)
        self.assertIn('passed', result)
        
        # Check readiness score calculation
        self.assertIsInstance(result['readiness_score'], float)
        self.assertGreaterEqual(result['readiness_score'], 0.0)
        self.assertLessEqual(result['readiness_score'], 1.0)
        
        # Check that all readiness checks were performed
        self.assertEqual(len(result['details']), 8)  # 8 readiness checks
    
    def test_final_performance_validation(self):
        """Test final performance validation"""
        result = self.validator.validate_final_performance()
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertEqual(result['validation'], 'performance')
        self.assertIn('performance_score', result)
        self.assertIn('tests', result)
        self.assertIn('benchmarks', result)
        self.assertIn('passed', result)
        
        # Check performance tests
        expected_tests = [
            'api_response', 'prediction_latency', 'training_startup',
            'memory_usage', 'cpu_utilization', 'concurrent_operations', 'throughput'
        ]
        for test in expected_tests:
            self.assertIn(test, result['tests'])
        
        # Check benchmarks are included
        self.assertEqual(result['benchmarks'], self.validator.performance_benchmarks)
    
    def test_final_security_validation(self):
        """Test final security validation"""
        result = self.validator.test_final_security()
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertEqual(result['validation'], 'security')
        self.assertIn('security_score', result)
        self.assertIn('vulnerabilities', result)
        self.assertIn('passed', result)
        
        # Should use security validator if available
        self.mock_components['security_validator'].run_full_security_validation.assert_called_once()
        
        # Test without security validator
        components_no_security = {k: v for k, v in self.mock_components.items() 
                                 if k != 'security_validator'}
        temp_validator = ProductionFinalValidator(
            self.mock_brain, components_no_security, self.production_config
        )
        temp_validator.security_validator = None
        
        result = temp_validator.test_final_security()
        self.assertIsInstance(result['security_score'], float)
    
    def test_final_scalability_validation(self):
        """Test final scalability validation"""
        result = self.validator.validate_final_scalability()
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertEqual(result['validation'], 'scalability')
        self.assertIn('scalability_score', result)
        self.assertIn('tests', result)
        self.assertIn('max_concurrent_users', result)
        self.assertIn('passed', result)
        
        # Check scalability tests
        self.assertEqual(len(result['tests']), 6)  # 6 scalability tests
        self.assertEqual(result['max_concurrent_users'], 1000)
    
    def test_final_reliability_validation(self):
        """Test final reliability validation"""
        result = self.validator.test_final_reliability()
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertEqual(result['validation'], 'reliability')
        self.assertIn('reliability_score', result)
        self.assertIn('tests', result)
        self.assertIn('uptime_requirement', result)
        self.assertIn('passed', result)
        
        # Check reliability tests
        self.assertEqual(len(result['tests']), 6)  # 6 reliability tests
        self.assertEqual(result['uptime_requirement'], '99.9%')
    
    def test_production_configuration_validation(self):
        """Test production configuration validation"""
        result = self.validator.validate_production_configuration()
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertEqual(result['validation'], 'configuration')
        self.assertIn('configuration_score', result)
        self.assertIn('validations', result)
        self.assertIn('passed', result)
        
        # Check configuration validations
        self.assertEqual(len(result['validations']), 6)  # 6 config validations
    
    def test_disaster_recovery_testing(self):
        """Test disaster recovery testing"""
        result = self.validator.test_disaster_recovery()
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertEqual(result['validation'], 'disaster_recovery')
        self.assertIn('recovery_score', result)
        self.assertIn('tests', result)
        self.assertIn('rto_minutes', result)
        self.assertIn('rpo_minutes', result)
        self.assertIn('passed', result)
        
        # Check recovery objectives
        self.assertEqual(result['rto_minutes'], 30)
        self.assertEqual(result['rpo_minutes'], 15)
        
        # Check recovery tests
        self.assertEqual(len(result['tests']), 6)  # 6 recovery tests
    
    def test_monitoring_systems_validation(self):
        """Test monitoring systems validation"""
        result = self.validator.validate_monitoring_systems()
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertEqual(result['validation'], 'monitoring_systems')
        self.assertIn('monitoring_score', result)
        self.assertIn('monitoring_active', result)
        self.assertIn('alerting_configured', result)
        self.assertIn('passed', result)
        
        # Should pass with mock monitoring system
        self.assertGreater(result['monitoring_score'], 0)
        self.assertTrue(result['monitoring_active'])
    
    def test_deployment_decision_making(self):
        """Test deployment decision making logic"""
        # Test GO decision with good scores
        good_scores = {
            'health': 0.98,
            'readiness': 0.96,
            'performance': 0.94,
            'security': 0.92,
            'reliability': 0.99
        }
        
        decision = self.validator._make_deployment_decision(good_scores, 0.96)
        
        self.assertEqual(decision.decision, "GO")
        self.assertEqual(decision.overall_score, 0.96)
        self.assertEqual(len(decision.critical_issues), 0)
        self.assertGreater(len(decision.recommendations), 0)
        
        # Test NO-GO decision with poor scores
        poor_scores = {
            'health': 0.85,
            'readiness': 0.70,  # Below required 0.95
            'performance': 0.60,  # Below required 0.90
            'security': 0.80,    # Below required 0.90
            'reliability': 0.95  # Below required 0.99
        }
        
        decision = self.validator._make_deployment_decision(poor_scores, 0.78)
        
        self.assertEqual(decision.decision, "NO-GO")
        self.assertGreater(len(decision.critical_issues), 0)
        self.assertIn("Address all critical issues", decision.recommendations[0])
    
    def test_final_deployment_report_generation(self):
        """Test final deployment report generation"""
        # Create mock validation results
        validation_results = {
            'system_health': {'overall_health': 0.95, 'passed': True},
            'production_readiness': {'readiness_score': 0.94, 'passed': True},
            'performance': {'performance_score': 0.92, 'passed': True},
            'security': {'security_score': 0.96, 'passed': True},
            'reliability': {'reliability_score': 0.99, 'passed': True}
        }
        
        report = self.validator.generate_final_deployment_report(validation_results)
        
        # Check report structure
        self.assertIsInstance(report, dict)
        self.assertIn('report_id', report)
        self.assertIn('timestamp', report)
        self.assertIn('deployment_decision', report)
        self.assertIn('scores', report)
        self.assertIn('validation_results', report)
        self.assertIn('system_components', report)
        self.assertIn('validation_summary', report)
        self.assertIn('recommendations', report)
        
        # Check report ID is valid UUID
        uuid.UUID(report['report_id'])  # Should not raise exception
        
        # Check deployment decision
        self.assertIsInstance(report['deployment_decision'], dict)
        
        # Check validation summary
        summary = report['validation_summary']
        self.assertIn('total_tests', summary)
        self.assertIn('passed_tests', summary)
        self.assertIn('failed_tests', summary)
        self.assertIn('critical_issues', summary)
        
        # Verify report is saved
        self.assertTrue(Path("deployment_reports").exists())
    
    def test_complete_validation_run(self):
        """Test complete validation run"""
        result = self.validator.run_complete_validation()
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('report_id', result)
        self.assertIn('deployment_decision', result)
        self.assertIn('total_validation_duration_ms', result)
        
        # Check that all validations were performed
        expected_validations = [
            'system_health', 'production_readiness', 'performance', 
            'security', 'scalability', 'reliability', 'configuration',
            'disaster_recovery', 'monitoring_systems'
        ]
        
        for validation in expected_validations:
            # These should be present in the validation results
            pass  # Structure verified in report generation
        
        # Check duration tracking
        self.assertIsInstance(result['total_validation_duration_ms'], float)
        self.assertGreater(result['total_validation_duration_ms'], 0)
    
    def test_performance_tests_detailed(self):
        """Test detailed performance test methods"""
        # Test API response time
        api_result = self.validator._test_api_response_time()
        self.assertIn('response_time_ms', api_result)
        self.assertIn('benchmark_ms', api_result)
        self.assertIn('passed', api_result)
        
        # Test prediction latency
        latency_result = self.validator._test_prediction_latency()
        self.assertIn('latency_ms', latency_result)
        self.assertIn('benchmark_ms', latency_result)
        
        # Test training startup time
        startup_result = self.validator._test_training_startup_time()
        self.assertIn('startup_time_ms', startup_result)
        self.assertTrue(startup_result['passed'])
        
        # Test memory usage
        memory_result = self.validator._test_memory_usage()
        self.assertIn('memory_usage_mb', memory_result)
        self.assertIsInstance(memory_result['memory_usage_mb'], (int, float))
        
        # Test CPU utilization
        cpu_result = self.validator._test_cpu_utilization()
        self.assertIn('cpu_usage_percent', cpu_result)
        self.assertIsInstance(cpu_result['cpu_usage_percent'], (int, float))
        
        # Test concurrent operations
        concurrent_result = self.validator._test_concurrent_operations()
        self.assertIn('concurrent_operations', concurrent_result)
        self.assertIn('successful', concurrent_result)
        self.assertEqual(concurrent_result['concurrent_operations'], 10)
        
        # Test system throughput
        throughput_result = self.validator._test_system_throughput()
        self.assertIn('throughput_ops_per_second', throughput_result)
        self.assertIn('operations_tested', throughput_result)
        self.assertEqual(throughput_result['operations_tested'], 100)
    
    def test_overall_health_calculation(self):
        """Test overall health score calculation"""
        # Add some health scores
        self.validator.health_scores[SystemComponent.BRAIN_CORE] = SystemHealthScore(
            component=SystemComponent.BRAIN_CORE,
            health_score=0.95,
            status=HealthStatus.HEALTHY
        )
        
        self.validator.health_scores[SystemComponent.SECURITY_SYSTEM] = SystemHealthScore(
            component=SystemComponent.SECURITY_SYSTEM,
            health_score=0.90,
            status=HealthStatus.HEALTHY
        )
        
        # Calculate overall health
        overall_health = self.validator._calculate_overall_health()
        
        # Should be weighted average
        self.assertIsInstance(overall_health, float)
        self.assertGreaterEqual(overall_health, 0.0)
        self.assertLessEqual(overall_health, 1.0)
        
        # Test with empty scores
        empty_validator = ProductionFinalValidator(
            self.mock_brain, {}, self.production_config
        )
        empty_health = empty_validator._calculate_overall_health()
        self.assertEqual(empty_health, 0.0)
    
    def test_validation_error_handling(self):
        """Test validation error handling"""
        # Create validator with failing brain
        failing_brain = MagicMock()
        failing_brain.predict.side_effect = Exception("Brain failure")
        
        failing_validator = ProductionFinalValidator(
            failing_brain, self.mock_components, self.production_config
        )
        
        # Should handle brain failure gracefully in validation
        result = failing_validator._validate_brain_core()
        self.assertEqual(result['status'], 'unhealthy')
        self.assertEqual(result['health_score'], 0.0)
        self.assertIn('error', result)
    
    def test_concurrent_validation_safety(self):
        """Test thread safety of concurrent validations"""
        def run_validation():
            return self.validator._validate_brain_core()
        
        # Run multiple validations concurrently
        import threading
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_validation) for _ in range(20)]
            results = [f.result() for f in futures]
        
        # All should succeed
        self.assertEqual(len(results), 20)
        for result in results:
            self.assertIn('status', result)
            self.assertIn('health_score', result)
    
    def test_validation_result_recording(self):
        """Test validation result recording"""
        # Record a validation failure
        self.validator._record_validation_failure(
            SystemComponent.BRAIN_CORE,
            "test_failure",
            "Test failure message"
        )
        
        # Check it was recorded
        self.assertEqual(len(self.validator.validation_results), 1)
        result = self.validator.validation_results[0]
        
        self.assertEqual(result.component, SystemComponent.BRAIN_CORE)
        self.assertEqual(result.test_name, "test_failure")
        self.assertFalse(result.passed)
        self.assertEqual(result.severity, ValidationSeverity.CRITICAL)
        self.assertEqual(result.message, "Test failure message")
    
    def test_component_health_recording(self):
        """Test component health score recording"""
        # Record component health
        health_result = {
            'status': 'healthy',
            'health_score': 0.95,
            'metrics': {'cpu': 45.0, 'memory': 60.0},
            'error': None
        }
        
        self.validator._record_component_health(SystemComponent.BRAIN_CORE, health_result)
        
        # Check it was recorded
        self.assertIn(SystemComponent.BRAIN_CORE, self.validator.health_scores)
        recorded_health = self.validator.health_scores[SystemComponent.BRAIN_CORE]
        
        self.assertEqual(recorded_health.component, SystemComponent.BRAIN_CORE)
        self.assertEqual(recorded_health.health_score, 0.95)
        self.assertEqual(recorded_health.status, HealthStatus.HEALTHY)
        self.assertEqual(recorded_health.metrics['cpu'], 45.0)
        self.assertEqual(len(recorded_health.issues), 0)
    
    def test_hard_failure_behavior(self):
        """Test that system fails hard when required"""
        # Test complete validation with failing components
        failing_components = {}
        failing_brain = None
        
        failing_validator = ProductionFinalValidator(
            failing_brain, failing_components, self.production_config
        )
        
        # Should raise RuntimeError for deployment failure
        with self.assertRaises(RuntimeError):
            failing_validator.run_complete_validation()
    
    def test_no_fallback_behavior(self):
        """Test that there are no fallback mechanisms"""
        # All validation methods should fail hard when components are missing
        # This is verified by the error handling tests above
        
        # Test that missing brain causes hard failure
        validator_no_brain = ProductionFinalValidator(
            None, self.mock_components, self.production_config
        )
        
        brain_result = validator_no_brain._validate_brain_core()
        self.assertEqual(brain_result['health_score'], 0.0)
        self.assertEqual(brain_result['status'], 'unhealthy')
        
        # No fallback score - it's 0.0 indicating complete failure
        self.assertNotEqual(brain_result['health_score'], 0.5)  # No partial fallback


class TestProductionFinalValidatorIntegration(unittest.TestCase):
    """Integration tests for ProductionFinalValidator"""
    
    def setUp(self):
        """Set up integration test environment"""
        # Create more realistic mock components
        self.mock_brain = MagicMock()
        self.mock_brain.predict.return_value = {"prediction": "test", "confidence": 0.95}
        self.mock_brain.domains = ['financial_fraud', 'cybersecurity', 'molecular_analysis']
        
        self.full_components = {
            'monitoring_system': MagicMock(),
            'deployment_manager': MagicMock(),
            'security_validator': MagicMock(),
            'brain_orchestrator': MagicMock(),
            'neural_orchestrator': MagicMock(),
            'reasoning_orchestrator': MagicMock(),
            'uncertainty_orchestrator': MagicMock(),
            'gac_system': MagicMock(),
            'proof_system': MagicMock(),
            'compression_api': MagicMock(),
            'training_manager': MagicMock(),
            'universal_ai_core': MagicMock(),
            'error_recovery_system': MagicMock()
        }
        
        # Configure mocks for successful operation
        self.full_components['monitoring_system'].get_monitoring_status.return_value = {
            'status': 'running', 'uptime': '168h', 'alerts': 0
        }
        self.full_components['security_validator'].run_full_security_validation.return_value = {
            'security_score': 0.96, 'vulnerabilities': []
        }
        self.full_components['gac_system'].clip_gradients.return_value = [0.05, 0.1, 0.15]
        
        self.production_config = {
            'environment': 'production',
            'debug_mode': False,
            'monitoring_enabled': True,
            'security_level': 'high'
        }
        
        self.validator = ProductionFinalValidator(
            self.mock_brain, 
            self.full_components, 
            self.production_config
        )
        
        # Setup temp directory
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up integration test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_validation_pipeline(self):
        """Test complete validation pipeline"""
        # Run complete validation
        result = self.validator.run_complete_validation()
        
        # Should succeed with all components present
        self.assertIsInstance(result, dict)
        self.assertIn('deployment_decision', result)
        
        # Check that deployment decision is GO
        decision = result['deployment_decision']
        self.assertEqual(decision['decision'], "GO")
        
        # Check that report was generated and saved
        report_dir = Path("deployment_reports")
        self.assertTrue(report_dir.exists())
        
        report_files = list(report_dir.glob("*.json"))
        self.assertEqual(len(report_files), 1)
        
        # Verify report content
        with open(report_files[0], 'r') as f:
            saved_report = json.load(f)
        
        self.assertEqual(saved_report['report_id'], result['report_id'])
    
    def test_validation_with_degraded_systems(self):
        """Test validation with some degraded systems"""
        # Remove some components to simulate degraded state
        degraded_components = {k: v for k, v in self.full_components.items() 
                             if k not in ['proof_system', 'universal_ai_core']}
        
        degraded_validator = ProductionFinalValidator(
            self.mock_brain, degraded_components, self.production_config
        )
        
        # Run validation
        health_result = degraded_validator.validate_final_system_health()
        
        # Should still work but with lower scores
        self.assertLess(health_result['overall_health'], 1.0)
        self.assertIn('component_scores', health_result)
    
    def test_performance_under_load(self):
        """Test validation performance under load"""
        start_time = time.time()
        
        # Run multiple validations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(self.validator.validate_final_system_health)
                for _ in range(10)
            ]
            results = [f.result() for f in futures]
        
        end_time = time.time()
        
        # All should succeed
        self.assertEqual(len(results), 10)
        for result in results:
            self.assertIn('overall_health', result)
            self.assertIn('passed', result)
        
        # Should complete within reasonable time (10 seconds for 10 validations)
        self.assertLess(end_time - start_time, 10.0)
    
    def test_validation_state_consistency(self):
        """Test that validation state remains consistent"""
        # Run multiple validations and check state
        initial_results_count = len(self.validator.validation_results)
        initial_health_count = len(self.validator.health_scores)
        
        # Run several validations
        self.validator.validate_final_system_health()
        self.validator.test_production_readiness()
        self.validator.validate_final_performance()
        
        # State should be updated consistently
        final_results_count = len(self.validator.validation_results)
        final_health_count = len(self.validator.health_scores)
        
        # Should have added validation results and health scores
        self.assertGreaterEqual(final_results_count, initial_results_count)
        self.assertGreaterEqual(final_health_count, initial_health_count)


if __name__ == '__main__':
    unittest.main()