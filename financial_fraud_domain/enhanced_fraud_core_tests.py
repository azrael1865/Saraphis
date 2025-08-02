"""
Enhanced Fraud Detection Core - Test Suite
Comprehensive test suite for the enhanced fraud detection system
"""

import unittest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os
from typing import Dict, Any, List

# Import all components to test
from enhanced_fraud_core_main import (
    EnhancedFraudDetectionCore, EnhancedFraudCoreConfig, FraudDetectionResult,
    create_default_fraud_core, create_production_fraud_core
)
from enhanced_fraud_core_integration import (
    IntegrationManager, IntegrationConfig, FraudDetectionTestFramework,
    setup_complete_fraud_detection_system
)
from enhanced_fraud_core_exceptions import (
    EnhancedFraudException, DetectionError, ValidationError, SecurityError,
    DetectionStrategy, ValidationLevel, SecurityLevel
)
from enhanced_fraud_core_validators import ValidationFramework, ValidationConfig
from enhanced_fraud_core_recovery import ErrorRecoveryManager, CircuitBreaker
from enhanced_fraud_core_monitoring import MonitoringManager
from enhanced_fraud_core_security import SecurityManager, SecurityContext

# ======================== BASE TEST CLASS ========================

class EnhancedFraudCoreTestCase(unittest.TestCase):
    """Base test case for enhanced fraud detection core"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_config = EnhancedFraudCoreConfig(
            detection_strategy=DetectionStrategy.HYBRID,
            enable_async_processing=False,  # Disable for testing
            max_worker_threads=2,
            fraud_probability_threshold=0.7,
            enable_audit_logging=False  # Disable for testing
        )
        
        self.fraud_core = EnhancedFraudDetectionCore(self.test_config)
        
        # Sample test transactions
        self.normal_transaction = {
            'transaction_id': 'test_normal_001',
            'user_id': 'user_123',
            'amount': 50.0,
            'merchant_id': 'merchant_abc',
            'timestamp': datetime.now().isoformat(),
            'type': 'purchase',
            'currency': 'USD',
            'description': 'Coffee purchase'
        }
        
        self.suspicious_transaction = {
            'transaction_id': 'test_fraud_001',
            'user_id': 'user_456',
            'amount': 15000.0,
            'merchant_id': 'suspicious_merchant_1',
            'timestamp': datetime.now().isoformat(),
            'type': 'purchase',
            'currency': 'USD',
            'description': 'Large purchase'
        }
        
        self.invalid_transaction = {
            'transaction_id': 'test_invalid_001',
            'user_id': 'user_789',
            'amount': -100.0,  # Invalid negative amount
            'merchant_id': 'merchant_def',
            'timestamp': 'invalid_timestamp',
            'type': 'purchase',
            'currency': 'USD'
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        if hasattr(self, 'fraud_core'):
            self.fraud_core.shutdown()

# ======================== CORE FUNCTIONALITY TESTS ========================

class TestFraudDetectionCore(EnhancedFraudCoreTestCase):
    """Test core fraud detection functionality"""
    
    def test_initialization(self):
        """Test fraud core initialization"""
        self.assertTrue(self.fraud_core.initialized)
        self.assertIsNotNone(self.fraud_core.validation_framework)
        self.assertIsNotNone(self.fraud_core.error_recovery_manager)
        self.assertIsNotNone(self.fraud_core.monitoring_manager)
        self.assertIsNotNone(self.fraud_core.security_manager)
    
    def test_normal_transaction_detection(self):
        """Test detection of normal transaction"""
        result = self.fraud_core.detect_fraud(self.normal_transaction)
        
        self.assertIsInstance(result, FraudDetectionResult)
        self.assertEqual(result.transaction_id, 'test_normal_001')
        self.assertFalse(result.fraud_detected)
        self.assertLess(result.fraud_probability, 0.7)
        self.assertTrue(result.validation_passed)
        self.assertGreater(result.detection_time, 0)
    
    def test_suspicious_transaction_detection(self):
        """Test detection of suspicious transaction"""
        result = self.fraud_core.detect_fraud(self.suspicious_transaction)
        
        self.assertIsInstance(result, FraudDetectionResult)
        self.assertEqual(result.transaction_id, 'test_fraud_001')
        self.assertTrue(result.fraud_detected)
        self.assertGreater(result.fraud_probability, 0.7)
        self.assertTrue(result.validation_passed)
        self.assertGreater(result.detection_time, 0)
    
    def test_invalid_transaction_validation(self):
        """Test validation of invalid transaction"""
        with self.assertRaises(ValidationError):
            self.fraud_core.detect_fraud(self.invalid_transaction)
    
    def test_batch_detection(self):
        """Test batch fraud detection"""
        transactions = [self.normal_transaction, self.suspicious_transaction]
        results = self.fraud_core.batch_detect_fraud(transactions)
        
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], FraudDetectionResult)
        self.assertIsInstance(results[1], FraudDetectionResult)
        
        # Check that results match expected patterns
        normal_result = next(r for r in results if r.transaction_id == 'test_normal_001')
        suspicious_result = next(r for r in results if r.transaction_id == 'test_fraud_001')
        
        self.assertFalse(normal_result.fraud_detected)
        self.assertTrue(suspicious_result.fraud_detected)
    
    def test_detection_strategies(self):
        """Test different detection strategies"""
        strategies = [
            DetectionStrategy.RULES_ONLY,
            DetectionStrategy.ML_ONLY,
            DetectionStrategy.HYBRID,
            DetectionStrategy.ENSEMBLE
        ]
        
        for strategy in strategies:
            with self.subTest(strategy=strategy):
                config = EnhancedFraudCoreConfig(
                    detection_strategy=strategy,
                    enable_async_processing=False,
                    enable_audit_logging=False
                )
                core = EnhancedFraudDetectionCore(config)
                
                try:
                    result = core.detect_fraud(self.normal_transaction)
                    self.assertEqual(result.detection_strategy, strategy)
                    self.assertIsInstance(result, FraudDetectionResult)
                finally:
                    core.shutdown()
    
    def test_system_status(self):
        """Test system status reporting"""
        status = self.fraud_core.get_system_status()
        
        self.assertIn('core_status', status)
        self.assertIn('validation_status', status)
        self.assertIn('monitoring_status', status)
        self.assertIn('security_status', status)
        self.assertIn('recovery_status', status)
        self.assertIn('component_health', status)
        
        self.assertTrue(status['core_status']['initialized'])

# ======================== VALIDATION FRAMEWORK TESTS ========================

class TestValidationFramework(EnhancedFraudCoreTestCase):
    """Test validation framework"""
    
    def setUp(self):
        super().setUp()
        self.validation_config = ValidationConfig(
            validation_level=ValidationLevel.STANDARD,
            enable_async_validation=False,
            validation_timeout=10.0
        )
        self.validation_framework = ValidationFramework(self.validation_config)
    
    def tearDown(self):
        super().tearDown()
        self.validation_framework.shutdown()
    
    def test_transaction_validation(self):
        """Test transaction validation"""
        result = self.validation_framework.validate_all(
            {'transaction': self.normal_transaction}
        )
        
        self.assertTrue(result['is_valid'])
        self.assertEqual(len(result['errors']), 0)
        self.assertIn('validation_results', result)
    
    def test_invalid_transaction_validation(self):
        """Test validation of invalid transaction"""
        result = self.validation_framework.validate_all(
            {'transaction': self.invalid_transaction}
        )
        
        self.assertFalse(result['is_valid'])
        self.assertGreater(len(result['errors']), 0)
    
    def test_validation_caching(self):
        """Test validation result caching"""
        # First validation
        start_time = time.time()
        result1 = self.validation_framework.validate_all(
            {'transaction': self.normal_transaction}
        )
        first_time = time.time() - start_time
        
        # Second validation (should be cached)
        start_time = time.time()
        result2 = self.validation_framework.validate_all(
            {'transaction': self.normal_transaction}
        )
        second_time = time.time() - start_time
        
        self.assertEqual(result1['is_valid'], result2['is_valid'])
        self.assertLess(second_time, first_time)  # Cached should be faster
    
    def test_validation_performance_metrics(self):
        """Test validation performance metrics"""
        self.validation_framework.validate_all(
            {'transaction': self.normal_transaction}
        )
        
        metrics = self.validation_framework.get_performance_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn('transaction', metrics)

# ======================== PREPROCESSING INTEGRATION TESTS ========================

class TestPreprocessingIntegration(EnhancedFraudCoreTestCase):
    """Test preprocessing integration with fraud detection core"""
    
    def setUp(self):
        super().setUp()
        # Enable preprocessing in test configuration
        self.preprocessing_config = {
            'feature_engineering': {
                'enable_time_features': True,
                'enable_amount_features': True,
                'enable_frequency_features': True,
                'enable_velocity_features': True,
                'enable_merchant_features': True,
                'enable_geographic_features': True
            },
            'data_quality': {
                'missing_value_threshold': 0.1,
                'outlier_method': 'iqr',
                'outlier_threshold': 3.0,
                'duplicate_threshold': 0.1
            },
            'feature_selection': {
                'method': 'mutual_info',
                'k_features': 50,
                'correlation_threshold': 0.9
            },
            'scaling': {
                'method': 'standard'
            }
        }
        
        # Update fraud core config to enable preprocessing
        self.test_config.enable_preprocessing = True
        self.test_config.preprocessing_config = self.preprocessing_config
        
        # Reinitialize fraud core with preprocessing
        self.fraud_core = EnhancedFraudDetectionCore(self.test_config)
    
    def test_preprocessing_manager_initialization(self):
        """Test preprocessing manager initialization"""
        try:
            from enhanced_fraud_core_main import CompletePreprocessingManager
            
            # Test direct initialization
            preprocessing_manager = CompletePreprocessingManager(self.preprocessing_config)
            self.assertIsNotNone(preprocessing_manager)
            self.assertEqual(preprocessing_manager.config, self.preprocessing_config)
            
        except ImportError:
            self.skipTest("CompletePreprocessingManager not available")
    
    def test_preprocessing_transaction_processing(self):
        """Test transaction preprocessing"""
        try:
            from enhanced_fraud_core_main import CompletePreprocessingManager
            
            preprocessing_manager = CompletePreprocessingManager(self.preprocessing_config)
            
            # Test comprehensive transaction preprocessing
            test_transaction = {
                'transaction_id': 'TEST_PREPROCESS_001',
                'user_id': 'USER_TEST_001',
                'amount': 275.50,
                'timestamp': '2024-01-15T15:30:00Z',
                'merchant_id': 'MERCHANT_TEST_001',
                'merchant_category': 'restaurant',
                'location': 'San Francisco, CA',
                'payment_method': 'debit_card',
                'currency': 'USD',
                'description': 'Restaurant payment'
            }
            
            # Preprocess transaction
            result = preprocessing_manager.preprocess_transaction(test_transaction)
            
            # Validate preprocessing result structure
            self.assertIn('processed_data', result)
            self.assertIn('metadata', result)
            
            processed_data = result['processed_data']
            metadata = result['metadata']
            
            # Validate core features are present
            self.assertIn('amount', processed_data)
            self.assertIn('transaction_id', processed_data)
            
            # Validate engineered features
            expected_time_features = ['hour_of_day', 'day_of_week', 'is_weekend', 'is_business_hours']
            for feature in expected_time_features:
                self.assertIn(feature, processed_data, f"Missing time feature: {feature}")
            
            expected_amount_features = ['amount_log', 'amount_zscore']
            for feature in expected_amount_features:
                self.assertIn(feature, processed_data, f"Missing amount feature: {feature}")
            
            # Validate metadata
            self.assertIn('feature_count', metadata)
            self.assertIn('processing_time', metadata)
            self.assertIn('data_quality', metadata)
            
            # Verify feature count meets expectations
            feature_count = metadata['feature_count']
            self.assertGreaterEqual(feature_count, 30, f"Expected at least 30 features, got {feature_count}")
            
        except ImportError:
            self.skipTest("CompletePreprocessingManager not available")
    
    def test_fraud_detection_with_preprocessing(self):
        """Test fraud detection with preprocessing enabled"""
        if not hasattr(self.fraud_core, 'enable_preprocessing') or not self.fraud_core.enable_preprocessing:
            self.skipTest("Preprocessing not enabled in fraud core")
        
        # Test normal transaction with preprocessing
        result = self.fraud_core.detect_fraud(self.normal_transaction)
        
        self.assertIsInstance(result, FraudDetectionResult)
        self.assertEqual(result.transaction_id, 'test_normal_001')
        
        # Check if preprocessing metadata is included
        if hasattr(result, 'additional_metadata') and result.additional_metadata:
            preprocessing_metadata = result.additional_metadata.get('preprocessing')
            if preprocessing_metadata:
                self.assertIn('feature_count', preprocessing_metadata)
                self.assertIn('processing_time', preprocessing_metadata)
        
        # Test suspicious transaction with preprocessing
        result = self.fraud_core.detect_fraud(self.suspicious_transaction)
        
        self.assertIsInstance(result, FraudDetectionResult)
        self.assertEqual(result.transaction_id, 'test_fraud_001')
    
    def test_preprocessing_validation_integration(self):
        """Test integration of preprocessing validation"""
        try:
            from enhanced_fraud_core_validators import PreprocessingValidator, ValidationConfig
            
            validation_config = ValidationConfig(
                enable_basic_validation=True,
                enable_advanced_validation=True,
                enable_preprocessing_validation=True,
                validation_timeout=30.0
            )
            
            preprocessing_validator = PreprocessingValidator(validation_config)
            
            # Create sample preprocessing result
            sample_result = {
                'processed_data': {
                    'transaction_id': 'TEST_VAL_001',
                    'amount': 150.0,
                    'hour_of_day': 14,
                    'day_of_week': 2,
                    'amount_log': 5.01,
                    'amount_zscore': 0.3,
                    'merchant_risk_score': 0.2
                },
                'metadata': {
                    'feature_count': 45,
                    'processing_time': 0.025,
                    'data_quality': {
                        'quality_score': 0.92,
                        'missing_values': 0,
                        'outliers_detected': 0,
                        'duplicates_found': 0
                    },
                    'feature_engineering': {
                        'time_features': 8,
                        'amount_features': 10,
                        'frequency_features': 12,
                        'velocity_features': 8,
                        'merchant_features': 7
                    }
                }
            }
            
            # Validate preprocessing result
            validation_result = preprocessing_validator.validate(sample_result)
            
            # Check validation structure
            self.assertIn('is_valid', validation_result)
            self.assertIn('validation_score', validation_result)
            self.assertIn('errors', validation_result)
            self.assertIn('warnings', validation_result)
            self.assertIn('details', validation_result)
            
            # Check validation score is reasonable
            validation_score = validation_result['validation_score']
            self.assertGreaterEqual(validation_score, 0.0)
            self.assertLessEqual(validation_score, 1.0)
            
        except ImportError:
            self.skipTest("Preprocessing validation components not available")
    
    def test_preprocessing_feature_engineering_quality(self):
        """Test quality of feature engineering"""
        try:
            from enhanced_fraud_core_main import CompletePreprocessingManager
            
            preprocessing_manager = CompletePreprocessingManager(self.preprocessing_config)
            
            # Create diverse test transactions
            test_transactions = [
                {
                    'transaction_id': f'TEST_FE_{i:03d}',
                    'user_id': f'USER_{i:03d}',
                    'amount': 50.0 + (i * 10),
                    'timestamp': f'2024-01-{15 + (i % 15):02d}T{10 + (i % 12):02d}:30:00Z',
                    'merchant_id': f'MERCHANT_{i % 10:03d}',
                    'merchant_category': ['grocery', 'restaurant', 'gas', 'retail', 'online'][i % 5],
                    'location': f'City_{i % 5}',
                    'payment_method': ['credit_card', 'debit_card', 'cash'][i % 3],
                    'currency': 'USD'
                }
                for i in range(10)
            ]
            
            feature_counts = []
            processing_times = []
            quality_scores = []
            
            for transaction in test_transactions:
                result = preprocessing_manager.preprocess_transaction(transaction)
                
                metadata = result['metadata']
                feature_counts.append(metadata['feature_count'])
                processing_times.append(metadata['processing_time'])
                
                quality_info = metadata.get('data_quality', {})
                quality_scores.append(quality_info.get('quality_score', 0.0))
            
            # Validate consistency
            self.assertTrue(all(count >= 30 for count in feature_counts), 
                           f"Some transactions had too few features: {feature_counts}")
            
            self.assertTrue(all(time < 1.0 for time in processing_times), 
                           f"Some preprocessing took too long: {processing_times}")
            
            self.assertTrue(all(score >= 0.7 for score in quality_scores), 
                           f"Some quality scores too low: {quality_scores}")
            
            # Check feature count consistency (should be similar across transactions)
            feature_count_std = (sum((x - sum(feature_counts)/len(feature_counts))**2 for x in feature_counts) / len(feature_counts))**0.5
            self.assertLess(feature_count_std, 5, f"Feature count variance too high: {feature_count_std}")
            
        except ImportError:
            self.skipTest("CompletePreprocessingManager not available")

# ======================== ERROR RECOVERY TESTS ========================

class TestErrorRecovery(EnhancedFraudCoreTestCase):
    """Test error recovery mechanisms"""
    
    def test_error_recovery_on_failure(self):
        """Test error recovery when fraud detection fails"""
        # Mock a failure in the fraud detection
        original_execute = self.fraud_core._execute_detection_strategy
        
        def mock_execute(*args, **kwargs):
            raise DetectionError("Simulated detection failure")
        
        self.fraud_core._execute_detection_strategy = mock_execute
        
        # Should recover and return fallback result
        result = self.fraud_core.detect_fraud(self.normal_transaction)
        
        self.assertIsInstance(result, FraudDetectionResult)
        self.assertFalse(result.fraud_detected)
        self.assertFalse(result.validation_passed)
        self.assertIn('Fallback result', result.explanation)
        
        # Restore original method
        self.fraud_core._execute_detection_strategy = original_execute
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker functionality"""
        from enhanced_fraud_core_recovery import CircuitBreaker, CircuitBreakerConfig
        
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=1,
            timeout=1.0
        )
        circuit = CircuitBreaker(config, "test_circuit")
        
        # Function that always fails
        def failing_function():
            raise Exception("Always fails")
        
        # Should fail and eventually open circuit
        for i in range(3):
            with self.assertRaises(Exception):
                circuit.call(failing_function)
        
        # Circuit should be open now
        from enhanced_fraud_core_exceptions import CircuitBreakerError
        with self.assertRaises(CircuitBreakerError):
            circuit.call(failing_function)

# ======================== MONITORING TESTS ========================

class TestMonitoring(EnhancedFraudCoreTestCase):
    """Test monitoring functionality"""
    
    def test_performance_monitoring(self):
        """Test performance monitoring"""
        # Perform some operations to generate metrics
        self.fraud_core.detect_fraud(self.normal_transaction)
        self.fraud_core.detect_fraud(self.suspicious_transaction)
        
        # Check monitoring status
        status = self.fraud_core.monitoring_manager.get_system_status()
        
        self.assertIn('current_metrics', status)
        self.assertIn('performance_summary', status)
        self.assertIn('resource_summary', status)
    
    def test_metrics_collection(self):
        """Test metrics collection"""
        collector = self.fraud_core.monitoring_manager.metrics_collector
        
        # Get initial metrics
        initial_metrics = collector.get_current_metrics()
        
        # Perform operation
        self.fraud_core.detect_fraud(self.normal_transaction)
        
        # Get updated metrics
        updated_metrics = collector.get_current_metrics()
        
        # Should have recorded some activity
        self.assertIsInstance(initial_metrics, dict)
        self.assertIsInstance(updated_metrics, dict)
    
    def test_cache_functionality(self):
        """Test cache functionality"""
        cache = self.fraud_core.monitoring_manager.cache_manager
        
        # Test cache operations
        cache.set('test_key', 'test_value')
        value = cache.get('test_key')
        
        self.assertEqual(value, 'test_value')
        
        # Test cache stats
        stats = cache.get_stats()
        self.assertIn('cache_size', stats)
        self.assertIn('hit_rate', stats)

# ======================== SECURITY TESTS ========================

class TestSecurity(EnhancedFraudCoreTestCase):
    """Test security functionality"""
    
    def test_security_context_creation(self):
        """Test security context creation"""
        security_context = SecurityContext(
            user_id='test_user',
            session_id='test_session',
            ip_address='127.0.0.1',
            user_agent='test_agent',
            permissions=['read', 'write'],
            authentication_method='password',
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1)
        )
        
        self.assertEqual(security_context.user_id, 'test_user')
        self.assertIn('read', security_context.permissions)
    
    def test_threat_detection(self):
        """Test threat detection"""
        threat_detector = self.fraud_core.security_manager.threat_detector
        
        # Test with malicious request
        malicious_request = {
            'input': '<script>alert("xss")</script>',
            'query': 'SELECT * FROM users DROP TABLE users'
        }
        
        threats = threat_detector.analyze_request(
            '127.0.0.1', 'test_user', malicious_request
        )
        
        self.assertGreater(len(threats), 0)
        self.assertTrue(any('xss' in threat.event_type for threat in threats))
    
    def test_audit_logging(self):
        """Test audit logging"""
        audit_logger = self.fraud_core.security_manager.audit_logger
        
        # Log an event
        audit_logger.log_event(
            event_type='test_event',
            user_id='test_user',
            resource='test_resource',
            action='test_action',
            result='success'
        )
        
        # Search for the event
        events = audit_logger.search_events(
            user_id='test_user',
            event_type='test_event',
            limit=10
        )
        
        self.assertGreater(len(events), 0)
        self.assertEqual(events[0].event_type, 'test_event')

# ======================== INTEGRATION TESTS ========================

class TestIntegration(unittest.TestCase):
    """Test system integration"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = IntegrationConfig(
            test_mode=True,
            database_url=f"sqlite:///{self.temp_dir}/test.db",
            enable_message_queue=False,
            log_level="ERROR"  # Reduce log noise in tests
        )
        self.integration_manager = IntegrationManager(self.config)
        self.integration_manager.initialize_fraud_core()
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        self.integration_manager.shutdown()
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_integration_initialization(self):
        """Test integration manager initialization"""
        self.assertTrue(self.integration_manager.initialized)
        self.assertIsNotNone(self.integration_manager.fraud_core)
        
        status = self.integration_manager.get_integration_status()
        self.assertTrue(status['initialized'])
        self.assertTrue(status['fraud_core_initialized'])
    
    def test_integrated_fraud_detection(self):
        """Test integrated fraud detection"""
        transaction = {
            'transaction_id': 'integration_test_001',
            'user_id': 'test_user',
            'amount': 100.0,
            'merchant_id': 'test_merchant',
            'timestamp': datetime.now().isoformat(),
            'type': 'purchase'
        }
        
        result = self.integration_manager.detect_fraud_with_integration(transaction)
        
        self.assertIsInstance(result, FraudDetectionResult)
        self.assertEqual(result.transaction_id, 'integration_test_001')
        self.assertIsNotNone(result.additional_metadata)
    
    def test_database_storage(self):
        """Test database storage integration"""
        transaction = {
            'transaction_id': 'db_test_001',
            'user_id': 'test_user',
            'amount': 200.0,
            'merchant_id': 'test_merchant',
            'timestamp': datetime.now().isoformat(),
            'type': 'purchase'
        }
        
        result = self.integration_manager.detect_fraud_with_integration(transaction)
        
        # Check database integration status
        if 'database' in self.integration_manager.integrations:
            db_status = self.integration_manager.integrations['database'].get_status()
            self.assertEqual(db_status['status'], 'healthy')
            self.assertGreater(db_status['result_count'], 0)

# ======================== PERFORMANCE TESTS ========================

class TestPerformance(EnhancedFraudCoreTestCase):
    """Test performance characteristics"""
    
    def test_single_transaction_performance(self):
        """Test single transaction detection performance"""
        start_time = time.time()
        result = self.fraud_core.detect_fraud(self.normal_transaction)
        end_time = time.time()
        
        detection_time = end_time - start_time
        
        self.assertLess(detection_time, 1.0)  # Should complete within 1 second
        self.assertGreater(result.detection_time, 0)
        self.assertLess(result.detection_time, detection_time)
    
    def test_batch_processing_performance(self):
        """Test batch processing performance"""
        transactions = []
        for i in range(100):
            transaction = {
                'transaction_id': f'perf_test_{i}',
                'user_id': f'user_{i}',
                'amount': 100.0 + i,
                'merchant_id': f'merchant_{i % 10}',
                'timestamp': datetime.now().isoformat(),
                'type': 'purchase'
            }
            transactions.append(transaction)
        
        start_time = time.time()
        results = self.fraud_core.batch_detect_fraud(transactions)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        self.assertEqual(len(results), 100)
        self.assertLess(total_time, 30.0)  # Should complete within 30 seconds
        
        # Calculate average detection time
        avg_detection_time = sum(r.detection_time for r in results) / len(results)
        self.assertLess(avg_detection_time, 1.0)

# ======================== COMPREHENSIVE TEST FRAMEWORK ========================

class TestFraudDetectionTestFramework(unittest.TestCase):
    """Test the fraud detection test framework itself"""
    
    def setUp(self):
        """Set up test framework test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = IntegrationConfig(
            test_mode=True,
            database_url=f"sqlite:///{self.temp_dir}/framework_test.db",
            enable_message_queue=False,
            log_level="ERROR"
        )
        self.integration_manager = IntegrationManager(self.config)
        self.integration_manager.initialize_fraud_core()
        self.test_framework = FraudDetectionTestFramework(self.integration_manager)
    
    def tearDown(self):
        """Clean up test framework test fixtures"""
        self.integration_manager.shutdown()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_test_execution(self):
        """Test basic test execution"""
        results = self.test_framework.run_basic_tests()
        
        self.assertIn('total_tests', results)
        self.assertIn('passed_tests', results)
        self.assertIn('failed_tests', results)
        self.assertIn('test_details', results)
        
        self.assertGreater(results['total_tests'], 0)
        self.assertIsInstance(results['test_details'], list)
    
    def test_performance_test_execution(self):
        """Test performance test execution"""
        results = self.test_framework.run_performance_tests(num_transactions=50)
        
        self.assertIn('total_transactions', results)
        self.assertIn('successful_transactions', results)
        self.assertIn('transactions_per_second', results)
        self.assertIn('average_detection_time', results)
        
        self.assertEqual(results['total_transactions'], 50)
        self.assertGreater(results['transactions_per_second'], 0)
    
    def test_stress_test_execution(self):
        """Test stress test execution"""
        results = self.test_framework.run_stress_tests(
            concurrent_requests=5,
            duration_seconds=10
        )
        
        self.assertIn('concurrent_requests', results)
        self.assertIn('duration_seconds', results)
        self.assertIn('total_requests', results)
        self.assertIn('requests_per_second', results)
        self.assertIn('error_rate', results)
        
        self.assertEqual(results['concurrent_requests'], 5)
        self.assertEqual(results['duration_seconds'], 10)
        self.assertGreater(results['total_requests'], 0)

# ======================== TEST SUITE RUNNER ========================

def create_test_suite():
    """Create comprehensive test suite"""
    suite = unittest.TestSuite()
    
    # Add all test cases
    test_cases = [
        TestFraudDetectionCore,
        TestValidationFramework,
        TestErrorRecovery,
        TestMonitoring,
        TestSecurity,
        TestIntegration,
        TestPerformance,
        TestFraudDetectionTestFramework
    ]
    
    for test_case in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_case)
        suite.addTests(tests)
    
    return suite

def run_all_tests():
    """Run all tests and return results"""
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return {
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
        'failure_details': result.failures,
        'error_details': result.errors
    }

def run_specific_test_category(category: str):
    """Run specific category of tests"""
    test_categories = {
        'core': TestFraudDetectionCore,
        'validation': TestValidationFramework,
        'recovery': TestErrorRecovery,
        'monitoring': TestMonitoring,
        'security': TestSecurity,
        'integration': TestIntegration,
        'performance': TestPerformance,
        'framework': TestFraudDetectionTestFramework
    }
    
    if category not in test_categories:
        raise ValueError(f"Invalid test category: {category}")
    
    suite = unittest.TestLoader().loadTestsFromTestCase(test_categories[category])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return {
        'category': category,
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
    }

if __name__ == '__main__':
    print("Running Enhanced Fraud Detection Core Test Suite...")
    print("=" * 60)
    
    results = run_all_tests()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total tests run: {results['tests_run']}")
    print(f"Failures: {results['failures']}")
    print(f"Errors: {results['errors']}")
    print(f"Success rate: {results['success_rate']:.2%}")
    
    if results['failures'] > 0:
        print("\nFAILURES:")
        for failure in results['failure_details']:
            print(f"- {failure[0]}: {failure[1]}")
    
    if results['errors'] > 0:
        print("\nERRORS:")
        for error in results['error_details']:
            print(f"- {error[0]}: {error[1]}")
    
    print("\n" + "=" * 60)