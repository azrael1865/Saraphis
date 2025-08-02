"""
Enhanced Proof Verifier - Chunk 2: Advanced Testing Framework
Comprehensive testing framework for the enhanced proof verifier system,
including unit tests, integration tests, performance tests, and security tests.
"""

import logging
import json
import time
import threading
import asyncio
import pytest
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set, Type, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import traceback
import psutil
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import random
import string
import hashlib
import secrets

# Import enhanced proof components
try:
    from enhanced_proof_verifier import (
        FinancialProofVerifier as EnhancedFinancialProofVerifier,
        EnhancedProofClaim, EnhancedProofEvidence, EnhancedProofResult,
        SecurityLevel, ProofVerificationException, ProofConfigurationError,
        ProofGenerationError, ProofValidationError, ProofTimeoutError,
        ProofSecurityError, ProofIntegrityError, ClaimValidationError,
        EvidenceValidationError, ProofSystemError, ProofStorageError,
        CryptographicError, ProofExpiredError, ResourceLimitError,
        SecurityValidator, ResourceMonitor
    )
    from proof_verifier import (
        ProofType, ProofStatus, ProofLevel, ProofClaim, ProofEvidence, ProofResult
    )
    from enhanced_proof_integration import (
        ProofIntegrationManager, ProofIntegrationConfig, ProofSystemAnalyzer,
        ProofIntegrationError, ProofSystemAnalysisError
    )
    PROOF_COMPONENTS = True
except ImportError as e:
    PROOF_COMPONENTS = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Proof verifier components not available: {e}")

# Configure logging
logger = logging.getLogger(__name__)

# ======================== TEST CONFIGURATION ========================

@dataclass
class ProofTestConfig:
    """Configuration for proof verifier testing"""
    
    # Test execution settings
    enable_unit_tests: bool = True
    enable_integration_tests: bool = True
    enable_performance_tests: bool = True
    enable_security_tests: bool = True
    enable_stress_tests: bool = True
    enable_concurrent_tests: bool = True
    
    # Test data settings
    test_data_size: int = 1000
    stress_test_size: int = 10000
    concurrent_test_threads: int = 10
    
    # Performance test settings
    performance_timeout: int = 30
    performance_threshold_ms: int = 1000
    throughput_threshold: float = 10.0  # proofs per second
    
    # Security test settings
    security_test_iterations: int = 100
    cryptographic_test_rounds: int = 50
    
    # Mock settings
    mock_external_services: bool = True
    mock_database_operations: bool = True
    mock_network_calls: bool = True
    
    # Test environment
    test_environment: str = "testing"
    cleanup_after_tests: bool = True
    preserve_test_data: bool = False

class ProofTestError(Exception):
    """Exception raised during proof testing"""
    def __init__(self, message: str, test_type: str = None, component: str = None):
        super().__init__(message)
        self.test_type = test_type
        self.component = component
        self.timestamp = datetime.now()

# ======================== TEST DATA GENERATORS ========================

class ProofTestDataGenerator:
    """Generates test data for proof verification testing"""
    
    def __init__(self, config: ProofTestConfig):
        self.config = config
        self.random_seed = 42
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
    
    def generate_transaction_data(self, count: int = 100, 
                                fraud_rate: float = 0.05) -> List[Dict[str, Any]]:
        """Generate synthetic transaction data for testing"""
        transactions = []
        
        for i in range(count):
            # Generate basic transaction
            transaction = {
                'transaction_id': f'TEST_TXN_{i:08d}',
                'user_id': f'TEST_USER_{random.randint(1, 1000):06d}',
                'amount': round(random.uniform(1.0, 10000.0), 2),
                'timestamp': datetime.now() - timedelta(
                    hours=random.randint(0, 72),
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59)
                ),
                'merchant_id': f'TEST_MERCHANT_{random.randint(1, 100):04d}',
                'payment_method': random.choice(['credit_card', 'debit_card', 'bank_transfer']),
                'currency': random.choice(['USD', 'EUR', 'GBP']),
                'country': random.choice(['US', 'UK', 'DE', 'FR', 'CA']),
                'category': random.choice(['groceries', 'gas', 'restaurants', 'shopping', 'entertainment']),
                'channel': random.choice(['online', 'pos', 'atm', 'mobile']),
                'ip_address': f'{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}',
                'device_id': f'DEVICE_{random.randint(1, 1000):06d}',
                'is_fraud': random.random() < fraud_rate
            }
            
            # Add fraud indicators for fraudulent transactions
            if transaction['is_fraud']:
                transaction.update(self._add_fraud_indicators(transaction))
            
            transactions.append(transaction)
        
        return transactions
    
    def _add_fraud_indicators(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Add fraud indicators to a transaction"""
        fraud_indicators = {}
        
        # Unusual amounts
        if random.random() < 0.3:
            fraud_indicators['unusual_amount'] = True
            fraud_indicators['amount'] = round(random.uniform(5000.0, 50000.0), 2)
        
        # Unusual times
        if random.random() < 0.2:
            fraud_indicators['unusual_time'] = True
            fraud_indicators['timestamp'] = datetime.now().replace(
                hour=random.randint(2, 5),  # Early morning
                minute=random.randint(0, 59)
            )
        
        # Multiple rapid transactions
        if random.random() < 0.4:
            fraud_indicators['rapid_transactions'] = True
            fraud_indicators['velocity_score'] = random.uniform(0.8, 1.0)
        
        # Geographic anomalies
        if random.random() < 0.3:
            fraud_indicators['geographic_anomaly'] = True
            fraud_indicators['country'] = random.choice(['XX', 'YY', 'ZZ'])  # Unusual countries
        
        return fraud_indicators
    
    def generate_proof_claims(self, count: int = 100) -> List[Dict[str, Any]]:
        """Generate test proof claims"""
        claims = []
        
        for i in range(count):
            claim = {
                'claim_id': f'TEST_CLAIM_{i:08d}',
                'claim_type': random.choice(list(ProofType)).value if PROOF_COMPONENTS else 'transaction_fraud',
                'transaction_id': f'TEST_TXN_{i:08d}',
                'timestamp': datetime.now() - timedelta(minutes=random.randint(0, 1440)),
                'fraud_probability': random.uniform(0.0, 1.0),
                'risk_score': random.uniform(0.0, 100.0),
                'evidence': self._generate_evidence_data(),
                'model_version': f'v{random.randint(1, 10)}.{random.randint(0, 9)}',
                'model_confidence': random.uniform(0.5, 1.0),
                'violated_rules': [f'rule_{random.randint(1, 20)}' for _ in range(random.randint(0, 3))]
            }
            
            claims.append(claim)
        
        return claims
    
    def _generate_evidence_data(self) -> Dict[str, Any]:
        """Generate evidence data for proof claims"""
        return {
            'transaction_features': {
                'amount_zscore': random.uniform(-3.0, 3.0),
                'time_since_last_transaction': random.randint(1, 86400),
                'merchant_risk_score': random.uniform(0.0, 1.0),
                'user_risk_score': random.uniform(0.0, 1.0)
            },
            'rule_evidence': {
                'amount_limit_exceeded': random.choice([True, False]),
                'velocity_limit_exceeded': random.choice([True, False]),
                'geographic_rule_violated': random.choice([True, False])
            },
            'ml_evidence': {
                'model_prediction': random.uniform(0.0, 1.0),
                'feature_importance': {
                    'amount': random.uniform(0.0, 1.0),
                    'time': random.uniform(0.0, 1.0),
                    'merchant': random.uniform(0.0, 1.0),
                    'user': random.uniform(0.0, 1.0)
                }
            }
        }
    
    def generate_corrupted_data(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate corrupted data for security testing"""
        corrupted_data = []
        
        corruption_types = [
            'sql_injection',
            'xss_payload',
            'buffer_overflow',
            'null_bytes',
            'unicode_attacks',
            'format_string',
            'path_traversal'
        ]
        
        for i in range(count):
            corruption_type = random.choice(corruption_types)
            
            if corruption_type == 'sql_injection':
                payload = "'; DROP TABLE users; --"
            elif corruption_type == 'xss_payload':
                payload = "<script>alert('XSS')</script>"
            elif corruption_type == 'buffer_overflow':
                payload = "A" * 10000
            elif corruption_type == 'null_bytes':
                payload = "test\x00malicious"
            elif corruption_type == 'unicode_attacks':
                payload = "test\u202e\u0041\u0042"
            elif corruption_type == 'format_string':
                payload = "%s%s%s%s%s%s%s%s"
            elif corruption_type == 'path_traversal':
                payload = "../../../../etc/passwd"
            else:
                payload = "malicious_payload"
            
            corrupted_transaction = {
                'transaction_id': f'CORRUPT_{i:08d}',
                'user_id': payload,
                'amount': payload if corruption_type != 'buffer_overflow' else -999999,
                'timestamp': payload if corruption_type != 'null_bytes' else datetime.now(),
                'merchant_id': payload,
                'payment_method': payload,
                'currency': payload,
                'country': payload,
                'corruption_type': corruption_type
            }
            
            corrupted_data.append(corrupted_transaction)
        
        return corrupted_data

# ======================== PROOF VERIFIER TEST SUITE ========================

class ProofVerifierTestSuite:
    """Comprehensive test suite for proof verifier system"""
    
    def __init__(self, config: ProofTestConfig = None):
        self.config = config or ProofTestConfig()
        self.test_data_generator = ProofTestDataGenerator(self.config)
        self.integration_manager = None
        self.test_results = {}
        self.setup_completed = False
        
    def setup_test_environment(self):
        """Set up test environment"""
        try:
            logger.info("Setting up proof verifier test environment")
            
            # Create integration manager with test configuration
            integration_config = ProofIntegrationConfig(
                enable_data_validation=True,
                enable_preprocessing=False,  # Disable for faster testing
                enable_ml_integration=False,  # Disable for isolated testing
                enable_performance_monitoring=True,
                max_worker_threads=2,
                proof_timeout=30,
                batch_size=50,
                enable_test_mode=True
            )
            
            if PROOF_COMPONENTS:
                self.integration_manager = ProofIntegrationManager(integration_config)
            
            self.setup_completed = True
            logger.info("Test environment setup completed")
            
        except Exception as e:
            logger.error(f"Test environment setup failed: {e}")
            raise ProofTestError(f"Test setup failed: {e}", test_type="setup")
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        try:
            if not self.setup_completed:
                self.setup_test_environment()
            
            test_results = {
                'test_summary': {
                    'start_time': datetime.now().isoformat(),
                    'test_environment': self.config.test_environment,
                    'total_tests': 0,
                    'passed_tests': 0,
                    'failed_tests': 0,
                    'skipped_tests': 0
                },
                'test_categories': {}
            }
            
            # Run test categories
            if self.config.enable_unit_tests:
                test_results['test_categories']['unit_tests'] = self._run_unit_tests()
            
            if self.config.enable_integration_tests:
                test_results['test_categories']['integration_tests'] = self._run_integration_tests()
            
            if self.config.enable_performance_tests:
                test_results['test_categories']['performance_tests'] = self._run_performance_tests()
            
            if self.config.enable_security_tests:
                test_results['test_categories']['security_tests'] = self._run_security_tests()
            
            if self.config.enable_stress_tests:
                test_results['test_categories']['stress_tests'] = self._run_stress_tests()
            
            if self.config.enable_concurrent_tests:
                test_results['test_categories']['concurrent_tests'] = self._run_concurrent_tests()
            
            # Calculate overall results
            total_tests = 0
            passed_tests = 0
            failed_tests = 0
            skipped_tests = 0
            
            for category, results in test_results['test_categories'].items():
                for test_name, test_result in results.items():
                    total_tests += 1
                    if test_result.get('passed', False):
                        passed_tests += 1
                    elif test_result.get('skipped', False):
                        skipped_tests += 1
                    else:
                        failed_tests += 1
            
            test_results['test_summary'].update({
                'end_time': datetime.now().isoformat(),
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'skipped_tests': skipped_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0
            })
            
            logger.info(f"Comprehensive tests completed: {passed_tests}/{total_tests} passed")
            return test_results
            
        except Exception as e:
            logger.error(f"Comprehensive test execution failed: {e}")
            raise ProofTestError(f"Test execution failed: {e}", test_type="comprehensive")
    
    def _run_unit_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run unit tests for proof verifier components"""
        unit_tests = {}
        
        # Test proof claim creation
        unit_tests['proof_claim_creation'] = self._run_test(
            "Proof claim creation",
            self._test_proof_claim_creation
        )
        
        # Test proof evidence validation
        unit_tests['proof_evidence_validation'] = self._run_test(
            "Proof evidence validation",
            self._test_proof_evidence_validation
        )
        
        # Test security validator
        unit_tests['security_validator'] = self._run_test(
            "Security validator",
            self._test_security_validator
        )
        
        # Test resource monitor
        unit_tests['resource_monitor'] = self._run_test(
            "Resource monitor",
            self._test_resource_monitor
        )
        
        # Test proof verification
        unit_tests['proof_verification'] = self._run_test(
            "Proof verification",
            self._test_proof_verification
        )
        
        # Test exception handling
        unit_tests['exception_handling'] = self._run_test(
            "Exception handling",
            self._test_exception_handling
        )
        
        return unit_tests
    
    def _run_integration_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run integration tests"""
        integration_tests = {}
        
        # Test full integration workflow
        integration_tests['full_integration_workflow'] = self._run_test(
            "Full integration workflow",
            self._test_full_integration_workflow
        )
        
        # Test component integration
        integration_tests['component_integration'] = self._run_test(
            "Component integration",
            self._test_component_integration
        )
        
        # Test data validation integration
        integration_tests['data_validation_integration'] = self._run_test(
            "Data validation integration",
            self._test_data_validation_integration
        )
        
        # Test error handling integration
        integration_tests['error_handling_integration'] = self._run_test(
            "Error handling integration",
            self._test_error_handling_integration
        )
        
        return integration_tests
    
    def _run_performance_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run performance tests"""
        performance_tests = {}
        
        # Test single proof verification speed
        performance_tests['single_proof_speed'] = self._run_test(
            "Single proof verification speed",
            self._test_single_proof_speed
        )
        
        # Test batch proof verification
        performance_tests['batch_proof_verification'] = self._run_test(
            "Batch proof verification",
            self._test_batch_proof_verification
        )
        
        # Test memory usage
        performance_tests['memory_usage'] = self._run_test(
            "Memory usage",
            self._test_memory_usage
        )
        
        # Test throughput
        performance_tests['throughput'] = self._run_test(
            "Throughput",
            self._test_throughput
        )
        
        return performance_tests
    
    def _run_security_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run security tests"""
        security_tests = {}
        
        # Test input sanitization
        security_tests['input_sanitization'] = self._run_test(
            "Input sanitization",
            self._test_input_sanitization
        )
        
        # Test injection attacks
        security_tests['injection_attacks'] = self._run_test(
            "Injection attacks",
            self._test_injection_attacks
        )
        
        # Test cryptographic security
        security_tests['cryptographic_security'] = self._run_test(
            "Cryptographic security",
            self._test_cryptographic_security
        )
        
        # Test access control
        security_tests['access_control'] = self._run_test(
            "Access control",
            self._test_access_control
        )
        
        return security_tests
    
    def _run_stress_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run stress tests"""
        stress_tests = {}
        
        # Test high volume processing
        stress_tests['high_volume_processing'] = self._run_test(
            "High volume processing",
            self._test_high_volume_processing
        )
        
        # Test resource exhaustion
        stress_tests['resource_exhaustion'] = self._run_test(
            "Resource exhaustion",
            self._test_resource_exhaustion
        )
        
        # Test timeout handling
        stress_tests['timeout_handling'] = self._run_test(
            "Timeout handling",
            self._test_timeout_handling
        )
        
        return stress_tests
    
    def _run_concurrent_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run concurrent tests"""
        concurrent_tests = {}
        
        # Test concurrent proof verification
        concurrent_tests['concurrent_proof_verification'] = self._run_test(
            "Concurrent proof verification",
            self._test_concurrent_proof_verification
        )
        
        # Test thread safety
        concurrent_tests['thread_safety'] = self._run_test(
            "Thread safety",
            self._test_thread_safety
        )
        
        # Test race conditions
        concurrent_tests['race_conditions'] = self._run_test(
            "Race conditions",
            self._test_race_conditions
        )
        
        return concurrent_tests
    
    def _run_test(self, test_name: str, test_func: Callable) -> Dict[str, Any]:
        """Run a single test with comprehensive error handling"""
        try:
            start_time = time.time()
            
            # Execute test
            test_func()
            
            duration = time.time() - start_time
            
            return {
                'passed': True,
                'duration': duration,
                'message': f"{test_name} passed",
                'timestamp': datetime.now().isoformat(),
                'details': {}
            }
            
        except Exception as e:
            duration = time.time() - start_time if 'start_time' in locals() else 0
            
            return {
                'passed': False,
                'duration': duration,
                'message': f"{test_name} failed: {str(e)}",
                'error': str(e),
                'error_type': type(e).__name__,
                'timestamp': datetime.now().isoformat(),
                'traceback': traceback.format_exc()
            }
    
    # ======================== UNIT TEST IMPLEMENTATIONS ========================
    
    def _test_proof_claim_creation(self):
        """Test proof claim creation"""
        if not PROOF_COMPONENTS:
            raise ProofTestError("Proof components not available", test_type="unit")
        
        # Test valid claim creation
        test_data = self.test_data_generator.generate_transaction_data(1)[0]
        
        claim = EnhancedProofClaim(
            claim_id="TEST_CLAIM_001",
            claim_type=ProofType.TRANSACTION_FRAUD,
            transaction_id=test_data['transaction_id'],
            timestamp=datetime.now(),
            fraud_probability=0.75,
            risk_score=85.0,
            evidence={'test': 'evidence'}
        )
        
        assert claim.claim_id == "TEST_CLAIM_001"
        assert claim.claim_type == ProofType.TRANSACTION_FRAUD
        assert claim.fraud_probability == 0.75
        assert claim.risk_score == 85.0
        
        # Test claim validation
        claim.validate()
    
    def _test_proof_evidence_validation(self):
        """Test proof evidence validation"""
        if not PROOF_COMPONENTS:
            raise ProofTestError("Proof components not available", test_type="unit")
        
        # Test valid evidence
        evidence = EnhancedProofEvidence(
            evidence_id="TEST_EVIDENCE_001",
            evidence_type="transaction_data",
            data={'amount': 100.0, 'currency': 'USD'},
            timestamp=datetime.now(),
            source="test_system"
        )
        
        assert evidence.evidence_id == "TEST_EVIDENCE_001"
        assert evidence.evidence_type == "transaction_data"
        
        # Test evidence validation
        evidence.validate()
    
    def _test_security_validator(self):
        """Test security validator"""
        if not PROOF_COMPONENTS:
            raise ProofTestError("Proof components not available", test_type="unit")
        
        validator = SecurityValidator()
        
        # Test valid input
        valid_input = "valid_transaction_id_123"
        assert validator.validate_input_safety(valid_input) == True
        
        # Test malicious input
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../../../etc/passwd",
            "\x00malicious",
            "A" * 10000
        ]
        
        for malicious_input in malicious_inputs:
            assert validator.validate_input_safety(malicious_input) == False
    
    def _test_resource_monitor(self):
        """Test resource monitor"""
        if not PROOF_COMPONENTS:
            raise ProofTestError("Proof components not available", test_type="unit")
        
        monitor = ResourceMonitor()
        
        # Test resource monitoring
        with monitor.monitor_operation("test_operation"):
            time.sleep(0.1)  # Simulate work
        
        # Check metrics were recorded
        assert "test_operation" in monitor.metrics
        assert monitor.metrics["test_operation"]["call_count"] > 0
        assert monitor.metrics["test_operation"]["total_duration"] > 0
    
    def _test_proof_verification(self):
        """Test proof verification"""
        if not PROOF_COMPONENTS:
            raise ProofTestError("Proof components not available", test_type="unit")
        
        # Create proof verifier
        verifier = EnhancedFinancialProofVerifier()
        
        # Generate test claim
        test_data = self.test_data_generator.generate_transaction_data(1)[0]
        
        claim = EnhancedProofClaim(
            claim_id="TEST_CLAIM_002",
            claim_type=ProofType.TRANSACTION_FRAUD,
            transaction_id=test_data['transaction_id'],
            timestamp=datetime.now(),
            fraud_probability=0.85,
            risk_score=90.0,
            evidence=test_data
        )
        
        # Verify proof
        result = verifier.verify_proof(claim)
        
        assert result is not None
        assert hasattr(result, 'claim_id')
        assert result.claim_id == "TEST_CLAIM_002"
    
    def _test_exception_handling(self):
        """Test exception handling"""
        if not PROOF_COMPONENTS:
            raise ProofTestError("Proof components not available", test_type="unit")
        
        # Test various exception types
        exceptions_to_test = [
            ProofConfigurationError,
            ProofGenerationError,
            ProofValidationError,
            ProofTimeoutError,
            ProofSecurityError,
            ProofIntegrityError
        ]
        
        for exc_class in exceptions_to_test:
            try:
                raise exc_class("Test exception")
            except exc_class as e:
                assert str(e) == "Test exception"
                assert isinstance(e, ProofVerificationException)
    
    # ======================== INTEGRATION TEST IMPLEMENTATIONS ========================
    
    def _test_full_integration_workflow(self):
        """Test full integration workflow"""
        if not self.integration_manager:
            raise ProofTestError("Integration manager not available", test_type="integration")
        
        # Generate test transaction
        test_data = self.test_data_generator.generate_transaction_data(1)[0]
        
        # Process with full integration
        result = self.integration_manager.process_proof_with_full_integration(
            test_data, 
            proof_type="transaction_fraud"
        )
        
        assert result is not None
        assert 'success' in result
        assert 'proof_result' in result or 'error' in result
    
    def _test_component_integration(self):
        """Test component integration"""
        if not self.integration_manager:
            raise ProofTestError("Integration manager not available", test_type="integration")
        
        # Test integration status
        status = self.integration_manager.get_integration_status()
        
        assert status is not None
        assert 'components' in status
        assert 'timestamp' in status
    
    def _test_data_validation_integration(self):
        """Test data validation integration"""
        if not self.integration_manager:
            raise ProofTestError("Integration manager not available", test_type="integration")
        
        # Test with valid data
        valid_data = self.test_data_generator.generate_transaction_data(1)[0]
        
        result = self.integration_manager.process_proof_with_full_integration(valid_data)
        
        # Should process successfully or with non-critical errors
        assert result is not None
    
    def _test_error_handling_integration(self):
        """Test error handling integration"""
        if not self.integration_manager:
            raise ProofTestError("Integration manager not available", test_type="integration")
        
        # Test with corrupted data
        corrupted_data = self.test_data_generator.generate_corrupted_data(1)[0]
        
        try:
            result = self.integration_manager.process_proof_with_full_integration(corrupted_data)
            # Should handle gracefully
            assert result is not None
        except Exception as e:
            # Should be a recoverable error
            assert isinstance(e, (ProofIntegrationError, ProofValidationError))
    
    # ======================== PERFORMANCE TEST IMPLEMENTATIONS ========================
    
    def _test_single_proof_speed(self):
        """Test single proof verification speed"""
        if not self.integration_manager:
            raise ProofTestError("Integration manager not available", test_type="performance")
        
        test_data = self.test_data_generator.generate_transaction_data(1)[0]
        
        start_time = time.time()
        result = self.integration_manager.process_proof_with_full_integration(test_data)
        duration = time.time() - start_time
        
        # Should complete within performance threshold
        assert duration < (self.config.performance_threshold_ms / 1000.0)
        assert result is not None
    
    def _test_batch_proof_verification(self):
        """Test batch proof verification"""
        if not self.integration_manager:
            raise ProofTestError("Integration manager not available", test_type="performance")
        
        batch_size = 10
        test_data = self.test_data_generator.generate_transaction_data(batch_size)
        
        start_time = time.time()
        results = []
        
        for data in test_data:
            result = self.integration_manager.process_proof_with_full_integration(data)
            results.append(result)
        
        duration = time.time() - start_time
        
        # Check batch processing efficiency
        assert len(results) == batch_size
        assert duration < (batch_size * self.config.performance_threshold_ms / 1000.0)
    
    def _test_memory_usage(self):
        """Test memory usage"""
        if not self.integration_manager:
            raise ProofTestError("Integration manager not available", test_type="performance")
        
        # Monitor memory usage
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        # Process test data
        test_data = self.test_data_generator.generate_transaction_data(100)
        
        for data in test_data:
            self.integration_manager.process_proof_with_full_integration(data)
        
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable (under 50MB)
        assert memory_increase < 50 * 1024 * 1024
    
    def _test_throughput(self):
        """Test throughput"""
        if not self.integration_manager:
            raise ProofTestError("Integration manager not available", test_type="performance")
        
        test_count = 50
        test_data = self.test_data_generator.generate_transaction_data(test_count)
        
        start_time = time.time()
        
        for data in test_data:
            self.integration_manager.process_proof_with_full_integration(data)
        
        duration = time.time() - start_time
        throughput = test_count / duration
        
        # Should meet throughput threshold
        assert throughput >= self.config.throughput_threshold
    
    # ======================== SECURITY TEST IMPLEMENTATIONS ========================
    
    def _test_input_sanitization(self):
        """Test input sanitization"""
        if not PROOF_COMPONENTS:
            raise ProofTestError("Proof components not available", test_type="security")
        
        validator = SecurityValidator()
        
        # Test various malicious inputs
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../../../etc/passwd",
            "\x00malicious",
            "A" * 10000,
            "%s%s%s%s%s%s%s%s",
            "test\u202e\u0041\u0042"
        ]
        
        for malicious_input in malicious_inputs:
            # Should be detected as unsafe
            assert validator.validate_input_safety(malicious_input) == False
    
    def _test_injection_attacks(self):
        """Test injection attacks"""
        if not self.integration_manager:
            raise ProofTestError("Integration manager not available", test_type="security")
        
        # Test SQL injection payloads
        sql_payloads = [
            "'; DROP TABLE transactions; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM users --",
            "'; UPDATE users SET password='hacked' --"
        ]
        
        for payload in sql_payloads:
            corrupted_data = {
                'transaction_id': payload,
                'user_id': 'TEST_USER',
                'amount': 100.0,
                'timestamp': datetime.now()
            }
            
            # Should handle safely without executing malicious code
            try:
                result = self.integration_manager.process_proof_with_full_integration(corrupted_data)
                # Should either reject or sanitize the input
                assert result is not None
            except (ProofValidationError, ProofSecurityError, ProofIntegrationError):
                # Expected security-related exceptions
                pass
    
    def _test_cryptographic_security(self):
        """Test cryptographic security"""
        if not PROOF_COMPONENTS:
            raise ProofTestError("Proof components not available", test_type="security")
        
        # Test hash consistency
        test_data = "test_data_for_hashing"
        hash1 = hashlib.sha256(test_data.encode()).hexdigest()
        hash2 = hashlib.sha256(test_data.encode()).hexdigest()
        
        assert hash1 == hash2
        
        # Test random token generation
        token1 = secrets.token_hex(32)
        token2 = secrets.token_hex(32)
        
        assert token1 != token2
        assert len(token1) == 64  # 32 bytes = 64 hex chars
    
    def _test_access_control(self):
        """Test access control"""
        if not PROOF_COMPONENTS:
            raise ProofTestError("Proof components not available", test_type="security")
        
        # Test security level enforcement
        security_levels = [SecurityLevel.LOW, SecurityLevel.MEDIUM, SecurityLevel.HIGH]
        
        for level in security_levels:
            # Should be able to create verifier with different security levels
            verifier = EnhancedFinancialProofVerifier(security_level=level)
            assert verifier is not None
    
    # ======================== STRESS TEST IMPLEMENTATIONS ========================
    
    def _test_high_volume_processing(self):
        """Test high volume processing"""
        if not self.integration_manager:
            raise ProofTestError("Integration manager not available", test_type="stress")
        
        # Generate large dataset
        test_data = self.test_data_generator.generate_transaction_data(self.config.stress_test_size)
        
        start_time = time.time()
        successful_proofs = 0
        failed_proofs = 0
        
        for data in test_data:
            try:
                result = self.integration_manager.process_proof_with_full_integration(data)
                if result and result.get('success', False):
                    successful_proofs += 1
                else:
                    failed_proofs += 1
            except Exception:
                failed_proofs += 1
        
        duration = time.time() - start_time
        
        # Should handle high volume with reasonable success rate
        success_rate = successful_proofs / len(test_data)
        assert success_rate >= 0.8  # 80% success rate minimum
        
        logger.info(f"High volume test: {successful_proofs}/{len(test_data)} successful in {duration:.2f}s")
    
    def _test_resource_exhaustion(self):
        """Test resource exhaustion handling"""
        if not self.integration_manager:
            raise ProofTestError("Integration manager not available", test_type="stress")
        
        # Test with large data objects
        large_data = {
            'transaction_id': 'LARGE_TXN_001',
            'user_id': 'TEST_USER',
            'amount': 100.0,
            'timestamp': datetime.now(),
            'large_field': 'A' * 1000000  # 1MB of data
        }
        
        try:
            result = self.integration_manager.process_proof_with_full_integration(large_data)
            # Should handle or reject gracefully
            assert result is not None
        except (ResourceLimitError, ProofValidationError):
            # Expected resource-related exceptions
            pass
    
    def _test_timeout_handling(self):
        """Test timeout handling"""
        if not self.integration_manager:
            raise ProofTestError("Integration manager not available", test_type="stress")
        
        # Test with data that might cause timeouts
        test_data = {
            'transaction_id': 'TIMEOUT_TXN_001',
            'user_id': 'TEST_USER',
            'amount': 100.0,
            'timestamp': datetime.now(),
            'timeout_simulation': True
        }
        
        try:
            result = self.integration_manager.process_proof_with_full_integration(test_data)
            # Should complete or timeout gracefully
            assert result is not None
        except (ProofTimeoutError, ProofIntegrationError):
            # Expected timeout-related exceptions
            pass
    
    # ======================== CONCURRENT TEST IMPLEMENTATIONS ========================
    
    def _test_concurrent_proof_verification(self):
        """Test concurrent proof verification"""
        if not self.integration_manager:
            raise ProofTestError("Integration manager not available", test_type="concurrent")
        
        def process_proof(data):
            return self.integration_manager.process_proof_with_full_integration(data)
        
        # Generate test data
        test_data = self.test_data_generator.generate_transaction_data(self.config.concurrent_test_threads)
        
        # Execute concurrently
        with ThreadPoolExecutor(max_workers=self.config.concurrent_test_threads) as executor:
            futures = [executor.submit(process_proof, data) for data in test_data]
            results = [future.result() for future in as_completed(futures)]
        
        # All should complete
        assert len(results) == len(test_data)
        successful_results = [r for r in results if r and r.get('success', False)]
        
        # Should have reasonable success rate
        success_rate = len(successful_results) / len(results)
        assert success_rate >= 0.7  # 70% success rate minimum for concurrent execution
    
    def _test_thread_safety(self):
        """Test thread safety"""
        if not self.integration_manager:
            raise ProofTestError("Integration manager not available", test_type="concurrent")
        
        shared_counter = {'value': 0}
        lock = threading.Lock()
        
        def increment_counter():
            for _ in range(100):
                with lock:
                    shared_counter['value'] += 1
                
                # Process proof during counter increment
                test_data = self.test_data_generator.generate_transaction_data(1)[0]
                self.integration_manager.process_proof_with_full_integration(test_data)
        
        # Run multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=increment_counter)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Counter should be correct (no race conditions)
        assert shared_counter['value'] == 500
    
    def _test_race_conditions(self):
        """Test race conditions"""
        if not self.integration_manager:
            raise ProofTestError("Integration manager not available", test_type="concurrent")
        
        # Test concurrent access to shared resources
        results = []
        
        def concurrent_access():
            for i in range(10):
                test_data = self.test_data_generator.generate_transaction_data(1)[0]
                test_data['thread_id'] = threading.current_thread().ident
                test_data['iteration'] = i
                
                result = self.integration_manager.process_proof_with_full_integration(test_data)
                results.append(result)
        
        # Run concurrent threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=concurrent_access)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have all results without corruption
        assert len(results) == 30  # 3 threads * 10 iterations
    
    def cleanup_test_environment(self):
        """Clean up test environment"""
        if self.config.cleanup_after_tests:
            try:
                if self.integration_manager:
                    self.integration_manager.shutdown()
                
                # Clean up test data if not preserving
                if not self.config.preserve_test_data:
                    # Clean up any temporary test files
                    pass
                
                logger.info("Test environment cleanup completed")
                
            except Exception as e:
                logger.error(f"Test environment cleanup failed: {e}")

# Export components
__all__ = [
    'ProofTestConfig',
    'ProofTestError',
    'ProofTestDataGenerator',
    'ProofVerifierTestSuite'
]

if __name__ == "__main__":
    print("Enhanced Proof Verifier - Chunk 2: Advanced testing framework loaded")
    
    # Basic test
    try:
        test_suite = ProofVerifierTestSuite()
        print("✓ Proof verifier test suite created successfully")
        
        # Run limited tests
        test_suite.config.stress_test_size = 100  # Reduced for quick test
        test_suite.config.concurrent_test_threads = 3
        
        results = test_suite.run_comprehensive_tests()
        
        summary = results['test_summary']
        print(f"✓ Tests completed: {summary['passed_tests']}/{summary['total_tests']} passed")
        print(f"✓ Success rate: {summary['success_rate']:.2%}")
        
    except Exception as e:
        print(f"✗ Proof verifier test failed: {e}")