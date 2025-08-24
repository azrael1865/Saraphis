#!/usr/bin/env python3
"""
Comprehensive test for SecurityValidator to identify all issues
"""

import time
import traceback
from typing import Dict, Any, List, Tuple
import threading
import json
import tempfile
import shutil
from pathlib import Path


def test_component(name: str, test_func, *args, **kwargs) -> Tuple[bool, str]:
    """Test a component and return status"""
    try:
        test_func(*args, **kwargs)
        return True, ""
    except Exception as e:
        tb = traceback.format_exc()
        return False, f"Error: {str(e)}\nTraceback: {tb}"


class ComprehensiveSecurityValidatorTester:
    def __init__(self):
        self.results = []
        self.temp_dirs = []
    
    def cleanup(self):
        """Clean up temporary resources"""
        for temp_dir in self.temp_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except:
                pass
    
    def test_imports(self):
        """Test that all imports work"""
        from production_testing.security_validator import SecurityValidator
        assert SecurityValidator is not None
    
    def test_basic_initialization(self):
        """Test basic SecurityValidator initialization"""
        from production_testing.security_validator import SecurityValidator
        
        # Test with empty config
        validator = SecurityValidator({})
        assert validator is not None
        
        # Test with basic config
        config = {
            'max_auth_failure_rate': 0.05,
            'max_encryption_time': 200,
            'min_password_entropy': 50,
            'max_session_duration': 7200,
            'min_key_length': 128,
            'gdpr_compliance': True,
            'hipaa_compliance': True,
            'pci_dss_compliance': False,
            'sox_compliance': False,
            'max_parallel_tests': 5
        }
        validator = SecurityValidator(config)
        assert validator.config == config
        assert validator.security_thresholds['max_auth_failure_rate'] == 0.05
        assert validator.compliance_requirements['gdpr'] == True
    
    def test_security_integration_validation(self):
        """Test security integration validation"""
        from production_testing.security_validator import SecurityValidator
        
        validator = SecurityValidator({})
        results = validator.validate_security_integration()
        
        assert isinstance(results, dict)
        assert 'authentication_authorization' in results
        assert 'data_encryption' in results
        assert 'access_control' in results
        assert 'vulnerability_protection' in results
        assert 'compliance' in results
        assert 'audit_logging' in results
        assert 'summary' in results
    
    def test_authentication_authorization(self):
        """Test authentication and authorization validation"""
        from production_testing.security_validator import SecurityValidator
        
        validator = SecurityValidator({})
        
        # Test auth validation methods
        auth_results = validator._validate_authentication_authorization()
        assert isinstance(auth_results, dict)
        assert 'password_strength' in auth_results
        assert 'multi_factor_auth' in auth_results
        assert 'session_management' in auth_results
        assert 'token_security' in auth_results
    
    def test_data_encryption(self):
        """Test data encryption validation"""
        from production_testing.security_validator import SecurityValidator
        
        validator = SecurityValidator({})
        
        # Test encryption validation
        encryption_results = validator._validate_data_encryption()
        assert isinstance(encryption_results, dict)
        assert 'encryption_at_rest' in encryption_results
        assert 'encryption_in_transit' in encryption_results
        assert 'key_management' in encryption_results
        assert 'algorithm_strength' in encryption_results
    
    def test_access_control(self):
        """Test access control validation"""
        from production_testing.security_validator import SecurityValidator
        
        validator = SecurityValidator({})
        
        # Test access control validation
        access_results = validator._validate_access_control()
        assert isinstance(access_results, dict)
        assert 'rbac' in access_results
        assert 'principle_least_privilege' in access_results
        assert 'access_reviews' in access_results
        assert 'permission_boundaries' in access_results
    
    def test_vulnerability_protection(self):
        """Test vulnerability protection validation"""
        from production_testing.security_validator import SecurityValidator
        
        validator = SecurityValidator({})
        
        # Test vulnerability protection
        vuln_results = validator._validate_vulnerability_protection()
        assert isinstance(vuln_results, dict)
        assert 'sql_injection' in vuln_results
        assert 'xss_protection' in vuln_results
        assert 'csrf_protection' in vuln_results
        assert 'input_validation' in vuln_results
    
    def test_compliance_validation(self):
        """Test compliance validation"""
        from production_testing.security_validator import SecurityValidator
        
        config = {
            'gdpr_compliance': True,
            'hipaa_compliance': True,
            'pci_dss_compliance': True,
            'sox_compliance': True
        }
        validator = SecurityValidator(config)
        
        # Test compliance validation
        compliance_results = validator._validate_compliance()
        assert isinstance(compliance_results, dict)
        assert 'gdpr' in compliance_results
        assert 'hipaa' in compliance_results
        assert 'pci_dss' in compliance_results
        assert 'sox' in compliance_results
    
    def test_audit_logging(self):
        """Test audit logging validation"""
        from production_testing.security_validator import SecurityValidator
        
        validator = SecurityValidator({})
        
        # Test audit logging
        audit_results = validator._validate_audit_logging()
        assert isinstance(audit_results, dict)
        assert 'log_integrity' in audit_results
        assert 'log_retention' in audit_results
        assert 'log_monitoring' in audit_results
        assert 'log_analysis' in audit_results
    
    def test_security_metrics(self):
        """Test security metrics tracking"""
        from production_testing.security_validator import SecurityValidator
        
        validator = SecurityValidator({})
        
        # Run some validations to generate metrics
        validator.validate_security_integration()
        
        # Get security metrics
        metrics = validator.get_security_metrics()
        assert isinstance(metrics, dict)
        assert 'overall_score' in metrics
        assert 'test_results' in metrics
        assert 'vulnerabilities' in metrics
        assert 'compliance_status' in metrics
    
    def test_parallel_validation(self):
        """Test parallel security validation"""
        from production_testing.security_validator import SecurityValidator
        
        config = {'max_parallel_tests': 3}
        validator = SecurityValidator(config)
        
        # Run parallel tests
        results = validator.run_parallel_security_tests()
        assert isinstance(results, dict)
        assert 'execution_time' in results
        assert 'parallel_tests_run' in results
    
    def test_security_report(self):
        """Test security report generation"""
        from production_testing.security_validator import SecurityValidator
        
        validator = SecurityValidator({})
        
        # Generate security report
        report = validator.generate_security_report()
        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'security_score' in report
        assert 'recommendations' in report
        assert 'critical_issues' in report
    
    def test_thread_safety(self):
        """Test thread safety of SecurityValidator"""
        from production_testing.security_validator import SecurityValidator
        
        validator = SecurityValidator({})
        results = []
        
        def worker():
            result = validator.validate_security_integration()
            results.append(result)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Verify all threads completed
        assert len(results) == 5
        for result in results:
            assert isinstance(result, dict)
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        from production_testing.security_validator import SecurityValidator
        
        # Test with invalid config values
        config = {
            'max_auth_failure_rate': -1,  # Invalid negative value
            'min_password_entropy': 0,    # Too low
            'max_parallel_tests': 0        # Invalid zero
        }
        validator = SecurityValidator(config)
        
        # Should handle gracefully
        results = validator.validate_security_integration()
        assert isinstance(results, dict)
    
    def test_cleanup(self):
        """Test cleanup and resource management"""
        from production_testing.security_validator import SecurityValidator
        
        validator = SecurityValidator({'max_parallel_tests': 5})
        
        # Run validation
        validator.validate_security_integration()
        
        # Cleanup
        validator.cleanup()
        
        # Verify executor pool is shutdown
        assert validator.executor_pool._shutdown == True
    
    def run_all_tests(self) -> Tuple[int, int]:
        """Run all tests and return results"""
        test_methods = [
            ("Imports", self.test_imports),
            ("Basic Initialization", self.test_basic_initialization),
            ("Security Integration Validation", self.test_security_integration_validation),
            ("Authentication Authorization", self.test_authentication_authorization),
            ("Data Encryption", self.test_data_encryption),
            ("Access Control", self.test_access_control),
            ("Vulnerability Protection", self.test_vulnerability_protection),
            ("Compliance Validation", self.test_compliance_validation),
            ("Audit Logging", self.test_audit_logging),
            ("Security Metrics", self.test_security_metrics),
            ("Parallel Validation", self.test_parallel_validation),
            ("Security Report", self.test_security_report),
            ("Thread Safety", self.test_thread_safety),
            ("Edge Cases", self.test_edge_cases),
            ("Cleanup", self.test_cleanup)
        ]
        
        print("\n" + "="*80)
        print("COMPREHENSIVE SECURITY VALIDATOR TESTING")
        print("="*80)
        
        passed = 0
        failed = 0
        failed_tests = []
        
        for test_name, test_func in test_methods:
            success, error = test_component(test_name, test_func)
            
            if success:
                print(f"✅ {test_name:<35} - OK")
                passed += 1
            else:
                print(f"❌ {test_name:<35} - FAILED")
                failed += 1
                failed_tests.append((test_name, error))
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total Tests: {passed + failed}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        
        if failed_tests:
            print("\nFAILED TESTS:")
            for test_name, error in failed_tests:
                print(f"  - {test_name}: {error[:100]}")
        
        print("\n" + "="*80)
        print("ACTION ITEMS")
        print("="*80)
        
        if failed == 0:
            print("\n✅ All SecurityValidator tests passed!")
        else:
            print("\nIssues to fix (in order of priority):")
            for i, (test_name, error) in enumerate(failed_tests, 1):
                print(f"{i}. {test_name}: {error[:100]}")
        
        self.cleanup()
        return passed, failed


def main():
    tester = ComprehensiveSecurityValidatorTester()
    passed, failed = tester.run_all_tests()
    
    # Exit with appropriate code
    exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()