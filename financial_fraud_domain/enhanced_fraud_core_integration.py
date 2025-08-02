"""
Enhanced Fraud Detection Core - Chunk 7: Integration Utilities and Testing Framework
Integration utilities, testing framework, and system integration for the enhanced fraud detection core
"""

import logging
import time
import threading
import asyncio
import unittest
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import json
import uuid
from functools import wraps
import traceback
from contextlib import contextmanager
import tempfile
import os
import sqlite3
from pathlib import Path
import yaml
import configparser
from unittest.mock import Mock, patch, MagicMock

# Import all core components
from enhanced_fraud_core_main import (
    EnhancedFraudDetectionCore, EnhancedFraudCoreConfig, FraudDetectionResult,
    create_default_fraud_core, create_production_fraud_core
)
from enhanced_fraud_core_exceptions import (
    EnhancedFraudException, DetectionError, ValidationError, SecurityError,
    IntegrationError, DetectionStrategy, ValidationLevel, SecurityLevel,
    ErrorContext, create_error_context
)
from enhanced_fraud_core_security import SecurityContext, SecurityManager

# Configure logging
logger = logging.getLogger(__name__)

# ======================== INTEGRATION CONFIGURATION ========================

@dataclass
class IntegrationConfig:
    """Configuration for system integration"""
    
    # Database integration
    database_url: str = "sqlite:///fraud_detection.db"
    database_pool_size: int = 10
    database_timeout: int = 30
    
    # Message queue integration
    message_queue_url: str = "redis://localhost:6379"
    enable_message_queue: bool = False
    
    # API integration
    api_base_url: str = "http://localhost:8000"
    api_timeout: int = 30
    api_retries: int = 3
    
    # Webhook integration
    webhook_url: Optional[str] = None
    webhook_timeout: int = 10
    webhook_retries: int = 2
    
    # External service integration
    external_services: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Monitoring integration
    metrics_backend: str = "prometheus"
    metrics_port: int = 9090
    
    # Logging integration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    
    # Testing configuration
    test_mode: bool = False
    test_data_path: str = "test_data"
    mock_external_services: bool = False

# ======================== INTEGRATION MANAGER ========================

class IntegrationManager:
    """Manages integration with external systems"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.fraud_core = None
        self.integrations = {}
        self.initialized = False
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize integrations
        self._initialize_integrations()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper())
        logging.basicConfig(
            level=log_level,
            format=self.config.log_format,
            filename=self.config.log_file
        )
        
        logger.info("Logging initialized")
    
    def _initialize_integrations(self) -> None:
        """Initialize all integrations"""
        try:
            # Initialize database integration
            self._initialize_database()
            
            # Initialize message queue integration
            if self.config.enable_message_queue:
                self._initialize_message_queue()
            
            # Initialize API integration
            self._initialize_api()
            
            # Initialize webhook integration
            if self.config.webhook_url:
                self._initialize_webhook()
            
            # Initialize external services
            self._initialize_external_services()
            
            # Initialize monitoring
            self._initialize_monitoring()
            
            self.initialized = True
            logger.info("All integrations initialized successfully")
            
        except Exception as e:
            logger.error(f"Integration initialization failed: {e}")
            raise IntegrationError(f"Failed to initialize integrations: {e}")
    
    def _initialize_database(self) -> None:
        """Initialize database integration"""
        try:
            # Create database connection
            self.integrations['database'] = DatabaseIntegration(self.config.database_url)
            logger.info("Database integration initialized")
        except Exception as e:
            logger.error(f"Database integration failed: {e}")
            raise
    
    def _initialize_message_queue(self) -> None:
        """Initialize message queue integration"""
        try:
            self.integrations['message_queue'] = MessageQueueIntegration(self.config.message_queue_url)
            logger.info("Message queue integration initialized")
        except Exception as e:
            logger.warning(f"Message queue integration failed: {e}")
    
    def _initialize_api(self) -> None:
        """Initialize API integration"""
        try:
            self.integrations['api'] = APIIntegration(
                self.config.api_base_url,
                self.config.api_timeout,
                self.config.api_retries
            )
            logger.info("API integration initialized")
        except Exception as e:
            logger.warning(f"API integration failed: {e}")
    
    def _initialize_webhook(self) -> None:
        """Initialize webhook integration"""
        try:
            self.integrations['webhook'] = WebhookIntegration(
                self.config.webhook_url,
                self.config.webhook_timeout,
                self.config.webhook_retries
            )
            logger.info("Webhook integration initialized")
        except Exception as e:
            logger.warning(f"Webhook integration failed: {e}")
    
    def _initialize_external_services(self) -> None:
        """Initialize external service integrations"""
        for service_name, service_config in self.config.external_services.items():
            try:
                self.integrations[service_name] = ExternalServiceIntegration(
                    service_name, service_config
                )
                logger.info(f"External service '{service_name}' initialized")
            except Exception as e:
                logger.warning(f"External service '{service_name}' failed: {e}")
    
    def _initialize_monitoring(self) -> None:
        """Initialize monitoring integration"""
        try:
            self.integrations['monitoring'] = MonitoringIntegration(
                self.config.metrics_backend,
                self.config.metrics_port
            )
            logger.info("Monitoring integration initialized")
        except Exception as e:
            logger.warning(f"Monitoring integration failed: {e}")
    
    def initialize_fraud_core(self, core_config: Optional[EnhancedFraudCoreConfig] = None) -> None:
        """Initialize fraud detection core"""
        if not self.initialized:
            raise IntegrationError("Integration manager not initialized")
        
        if core_config is None:
            if self.config.test_mode:
                self.fraud_core = create_default_fraud_core()
            else:
                self.fraud_core = create_production_fraud_core()
        else:
            self.fraud_core = EnhancedFraudDetectionCore(core_config)
        
        logger.info("Fraud detection core initialized")
    
    def detect_fraud_with_integration(self, transaction: Dict[str, Any],
                                     security_context: Optional[SecurityContext] = None) -> FraudDetectionResult:
        """Detect fraud with full integration support"""
        if not self.fraud_core:
            raise IntegrationError("Fraud detection core not initialized")
        
        try:
            # Pre-processing integrations
            transaction = self._preprocess_transaction(transaction)
            
            # Detect fraud
            result = self.fraud_core.detect_fraud(transaction, security_context)
            
            # Post-processing integrations
            result = self._postprocess_result(result, transaction)
            
            # Store result
            self._store_result(result, transaction)
            
            # Send notifications
            self._send_notifications(result, transaction)
            
            return result
            
        except Exception as e:
            logger.error(f"Integrated fraud detection failed: {e}")
            raise
    
    def _preprocess_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess transaction with integrations"""
        # Enrich transaction data from external services
        if 'api' in self.integrations:
            enriched_data = self.integrations['api'].enrich_transaction(transaction)
            transaction.update(enriched_data)
        
        # Add timestamp if not present
        if 'timestamp' not in transaction:
            transaction['timestamp'] = datetime.now().isoformat()
        
        return transaction
    
    def _postprocess_result(self, result: FraudDetectionResult,
                           transaction: Dict[str, Any]) -> FraudDetectionResult:
        """Post-process result with integrations"""
        # Add integration metadata
        result.additional_metadata['integration_version'] = '1.0.0'
        result.additional_metadata['processed_at'] = datetime.now().isoformat()
        
        return result
    
    def _store_result(self, result: FraudDetectionResult, transaction: Dict[str, Any]) -> None:
        """Store result in database"""
        if 'database' in self.integrations:
            self.integrations['database'].store_result(result, transaction)
    
    def _send_notifications(self, result: FraudDetectionResult, transaction: Dict[str, Any]) -> None:
        """Send notifications based on result"""
        if result.fraud_detected:
            # Send webhook notification
            if 'webhook' in self.integrations:
                self.integrations['webhook'].send_fraud_alert(result, transaction)
            
            # Send message queue notification
            if 'message_queue' in self.integrations:
                self.integrations['message_queue'].send_fraud_alert(result, transaction)
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integrations"""
        status = {
            'initialized': self.initialized,
            'fraud_core_initialized': self.fraud_core is not None,
            'integrations': {}
        }
        
        for name, integration in self.integrations.items():
            try:
                status['integrations'][name] = integration.get_status()
            except Exception as e:
                status['integrations'][name] = {'status': 'error', 'error': str(e)}
        
        return status
    
    def shutdown(self) -> None:
        """Shutdown all integrations"""
        logger.info("Shutting down integrations...")
        
        # Shutdown fraud core
        if self.fraud_core:
            self.fraud_core.shutdown()
        
        # Shutdown integrations
        for name, integration in self.integrations.items():
            try:
                integration.shutdown()
                logger.info(f"Integration '{name}' shutdown")
            except Exception as e:
                logger.error(f"Failed to shutdown integration '{name}': {e}")
        
        self.initialized = False
        logger.info("All integrations shutdown")

# ======================== INTEGRATION COMPONENTS ========================

class DatabaseIntegration:
    """Database integration for fraud detection"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection = None
        self._initialize_connection()
        self._create_tables()
    
    def _initialize_connection(self) -> None:
        """Initialize database connection"""
        if self.database_url.startswith('sqlite:///'):
            db_path = self.database_url.replace('sqlite:///', '')
            self.connection = sqlite3.connect(db_path, check_same_thread=False)
        else:
            # For other databases, would use appropriate driver
            raise NotImplementedError("Only SQLite supported in this implementation")
    
    def _create_tables(self) -> None:
        """Create necessary database tables"""
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS fraud_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL,
                fraud_detected BOOLEAN NOT NULL,
                fraud_probability REAL NOT NULL,
                risk_score REAL NOT NULL,
                confidence REAL NOT NULL,
                detection_strategy TEXT NOT NULL,
                detection_time REAL NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TEXT NOT NULL,
                result_data TEXT NOT NULL
            )
        ''')
        
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL,
                amount REAL NOT NULL,
                merchant_id TEXT,
                timestamp TEXT NOT NULL,
                transaction_data TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        
        self.connection.commit()
    
    def store_result(self, result: FraudDetectionResult, transaction: Dict[str, Any]) -> None:
        """Store fraud detection result"""
        try:
            # Store transaction
            self.connection.execute('''
                INSERT OR REPLACE INTO transactions 
                (transaction_id, user_id, amount, merchant_id, timestamp, transaction_data, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                transaction.get('transaction_id'),
                transaction.get('user_id'),
                transaction.get('amount'),
                transaction.get('merchant_id'),
                transaction.get('timestamp'),
                json.dumps(transaction),
                datetime.now().isoformat()
            ))
            
            # Store result
            self.connection.execute('''
                INSERT INTO fraud_results 
                (transaction_id, fraud_detected, fraud_probability, risk_score, confidence,
                 detection_strategy, detection_time, timestamp, created_at, result_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.transaction_id,
                result.fraud_detected,
                result.fraud_probability,
                result.risk_score,
                result.confidence,
                result.detection_strategy.value,
                result.detection_time,
                result.timestamp.isoformat(),
                datetime.now().isoformat(),
                json.dumps(result.__dict__, default=str)
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Failed to store result: {e}")
            self.connection.rollback()
    
    def get_status(self) -> Dict[str, Any]:
        """Get database status"""
        try:
            cursor = self.connection.execute('SELECT COUNT(*) FROM fraud_results')
            result_count = cursor.fetchone()[0]
            
            cursor = self.connection.execute('SELECT COUNT(*) FROM transactions')
            transaction_count = cursor.fetchone()[0]
            
            return {
                'status': 'healthy',
                'result_count': result_count,
                'transaction_count': transaction_count
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def shutdown(self) -> None:
        """Shutdown database connection"""
        if self.connection:
            self.connection.close()

class MessageQueueIntegration:
    """Message queue integration for fraud detection"""
    
    def __init__(self, queue_url: str):
        self.queue_url = queue_url
        self.connection = None
        # In a real implementation, would initialize Redis/RabbitMQ connection
    
    def send_fraud_alert(self, result: FraudDetectionResult, transaction: Dict[str, Any]) -> None:
        """Send fraud alert to message queue"""
        # Placeholder for message queue implementation
        logger.info(f"Fraud alert sent to queue: {result.transaction_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get message queue status"""
        return {'status': 'healthy'}
    
    def shutdown(self) -> None:
        """Shutdown message queue connection"""
        pass

class APIIntegration:
    """API integration for fraud detection"""
    
    def __init__(self, base_url: str, timeout: int, retries: int):
        self.base_url = base_url
        self.timeout = timeout
        self.retries = retries
    
    def enrich_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich transaction with external API data"""
        # Placeholder for API enrichment
        return {
            'enriched_at': datetime.now().isoformat(),
            'enrichment_source': 'external_api'
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get API status"""
        return {'status': 'healthy'}
    
    def shutdown(self) -> None:
        """Shutdown API integration"""
        pass

class WebhookIntegration:
    """Webhook integration for fraud detection"""
    
    def __init__(self, webhook_url: str, timeout: int, retries: int):
        self.webhook_url = webhook_url
        self.timeout = timeout
        self.retries = retries
    
    def send_fraud_alert(self, result: FraudDetectionResult, transaction: Dict[str, Any]) -> None:
        """Send fraud alert via webhook"""
        # Placeholder for webhook implementation
        logger.info(f"Fraud alert sent via webhook: {result.transaction_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get webhook status"""
        return {'status': 'healthy'}
    
    def shutdown(self) -> None:
        """Shutdown webhook integration"""
        pass

class ExternalServiceIntegration:
    """External service integration"""
    
    def __init__(self, service_name: str, config: Dict[str, Any]):
        self.service_name = service_name
        self.config = config
    
    def get_status(self) -> Dict[str, Any]:
        """Get external service status"""
        return {'status': 'healthy'}
    
    def shutdown(self) -> None:
        """Shutdown external service integration"""
        pass

class MonitoringIntegration:
    """Monitoring integration for fraud detection"""
    
    def __init__(self, backend: str, port: int):
        self.backend = backend
        self.port = port
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring status"""
        return {'status': 'healthy'}
    
    def shutdown(self) -> None:
        """Shutdown monitoring integration"""
        pass

# ======================== TESTING FRAMEWORK ========================

class FraudDetectionTestFramework:
    """Testing framework for fraud detection system"""
    
    def __init__(self, integration_manager: IntegrationManager):
        self.integration_manager = integration_manager
        self.test_data = {}
        self.test_results = []
        self.load_test_data()
    
    def load_test_data(self) -> None:
        """Load test data for testing"""
        # Sample test transactions
        self.test_data = {
            'normal_transactions': [
                {
                    'transaction_id': 'tx_normal_1',
                    'user_id': 'user_123',
                    'amount': 50.0,
                    'merchant_id': 'merchant_abc',
                    'timestamp': datetime.now().isoformat(),
                    'type': 'purchase'
                },
                {
                    'transaction_id': 'tx_normal_2',
                    'user_id': 'user_456',
                    'amount': 25.0,
                    'merchant_id': 'merchant_def',
                    'timestamp': datetime.now().isoformat(),
                    'type': 'purchase'
                }
            ],
            'fraudulent_transactions': [
                {
                    'transaction_id': 'tx_fraud_1',
                    'user_id': 'user_789',
                    'amount': 15000.0,
                    'merchant_id': 'suspicious_merchant_1',
                    'timestamp': datetime.now().isoformat(),
                    'type': 'purchase'
                },
                {
                    'transaction_id': 'tx_fraud_2',
                    'user_id': 'user_999',
                    'amount': 8000.0,
                    'merchant_id': 'suspicious_merchant_2',
                    'timestamp': datetime.now().isoformat(),
                    'type': 'purchase'
                }
            ]
        }
    
    def run_basic_tests(self) -> Dict[str, Any]:
        """Run basic fraud detection tests"""
        test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
        
        # Test normal transactions
        for transaction in self.test_data['normal_transactions']:
            result = self._test_transaction(transaction, expected_fraud=False)
            test_results['test_details'].append(result)
            test_results['total_tests'] += 1
            if result['passed']:
                test_results['passed_tests'] += 1
            else:
                test_results['failed_tests'] += 1
        
        # Test fraudulent transactions
        for transaction in self.test_data['fraudulent_transactions']:
            result = self._test_transaction(transaction, expected_fraud=True)
            test_results['test_details'].append(result)
            test_results['total_tests'] += 1
            if result['passed']:
                test_results['passed_tests'] += 1
            else:
                test_results['failed_tests'] += 1
        
        return test_results
    
    def _test_transaction(self, transaction: Dict[str, Any], expected_fraud: bool) -> Dict[str, Any]:
        """Test individual transaction"""
        try:
            result = self.integration_manager.detect_fraud_with_integration(transaction)
            
            # Check if result matches expectation
            passed = result.fraud_detected == expected_fraud
            
            return {
                'transaction_id': transaction['transaction_id'],
                'expected_fraud': expected_fraud,
                'detected_fraud': result.fraud_detected,
                'fraud_probability': result.fraud_probability,
                'passed': passed,
                'detection_time': result.detection_time,
                'error': None
            }
            
        except Exception as e:
            return {
                'transaction_id': transaction['transaction_id'],
                'expected_fraud': expected_fraud,
                'detected_fraud': False,
                'fraud_probability': 0.0,
                'passed': False,
                'detection_time': 0.0,
                'error': str(e)
            }
    
    def run_performance_tests(self, num_transactions: int = 1000) -> Dict[str, Any]:
        """Run performance tests"""
        start_time = time.time()
        
        # Generate test transactions
        transactions = []
        for i in range(num_transactions):
            transaction = {
                'transaction_id': f'perf_tx_{i}',
                'user_id': f'user_{i % 100}',
                'amount': 100.0 + (i % 1000),
                'merchant_id': f'merchant_{i % 50}',
                'timestamp': datetime.now().isoformat(),
                'type': 'purchase'
            }
            transactions.append(transaction)
        
        # Process transactions
        results = []
        for transaction in transactions:
            try:
                result = self.integration_manager.detect_fraud_with_integration(transaction)
                results.append(result)
            except Exception as e:
                logger.error(f"Performance test failed for transaction {transaction['transaction_id']}: {e}")
        
        total_time = time.time() - start_time
        
        # Calculate performance metrics
        detection_times = [r.detection_time for r in results if r.detection_time > 0]
        avg_detection_time = sum(detection_times) / len(detection_times) if detection_times else 0
        
        return {
            'total_transactions': num_transactions,
            'successful_transactions': len(results),
            'total_time': total_time,
            'transactions_per_second': len(results) / total_time if total_time > 0 else 0,
            'average_detection_time': avg_detection_time,
            'max_detection_time': max(detection_times) if detection_times else 0,
            'min_detection_time': min(detection_times) if detection_times else 0
        }
    
    def run_stress_tests(self, concurrent_requests: int = 50, duration_seconds: int = 60) -> Dict[str, Any]:
        """Run stress tests with concurrent requests"""
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        results = []
        errors = []
        
        def stress_worker():
            while time.time() < end_time:
                try:
                    transaction = {
                        'transaction_id': f'stress_tx_{uuid.uuid4()}',
                        'user_id': f'stress_user_{threading.current_thread().ident}',
                        'amount': 100.0,
                        'merchant_id': 'stress_merchant',
                        'timestamp': datetime.now().isoformat(),
                        'type': 'purchase'
                    }
                    
                    result = self.integration_manager.detect_fraud_with_integration(transaction)
                    results.append(result)
                    
                except Exception as e:
                    errors.append(str(e))
                
                time.sleep(0.1)  # Small delay to prevent overwhelming
        
        # Start concurrent workers
        threads = []
        for _ in range(concurrent_requests):
            thread = threading.Thread(target=stress_worker)
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        return {
            'concurrent_requests': concurrent_requests,
            'duration_seconds': duration_seconds,
            'total_requests': len(results),
            'total_errors': len(errors),
            'requests_per_second': len(results) / total_time if total_time > 0 else 0,
            'error_rate': len(errors) / (len(results) + len(errors)) if (len(results) + len(errors)) > 0 else 0,
            'success_rate': len(results) / (len(results) + len(errors)) if (len(results) + len(errors)) > 0 else 0
        }

# ======================== CONFIGURATION UTILITIES ========================

def load_config_from_file(config_file: str) -> IntegrationConfig:
    """Load integration configuration from file"""
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    return IntegrationConfig(**config_data)

def save_config_to_file(config: IntegrationConfig, config_file: str) -> None:
    """Save integration configuration to file"""
    config_path = Path(config_file)
    
    config_dict = {
        'database_url': config.database_url,
        'database_pool_size': config.database_pool_size,
        'database_timeout': config.database_timeout,
        'message_queue_url': config.message_queue_url,
        'enable_message_queue': config.enable_message_queue,
        'api_base_url': config.api_base_url,
        'api_timeout': config.api_timeout,
        'api_retries': config.api_retries,
        'webhook_url': config.webhook_url,
        'webhook_timeout': config.webhook_timeout,
        'webhook_retries': config.webhook_retries,
        'external_services': config.external_services,
        'metrics_backend': config.metrics_backend,
        'metrics_port': config.metrics_port,
        'log_level': config.log_level,
        'log_format': config.log_format,
        'log_file': config.log_file,
        'test_mode': config.test_mode,
        'test_data_path': config.test_data_path,
        'mock_external_services': config.mock_external_services
    }
    
    if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

def create_default_integration_manager() -> IntegrationManager:
    """Create default integration manager"""
    config = IntegrationConfig()
    return IntegrationManager(config)

def create_test_integration_manager() -> IntegrationManager:
    """Create integration manager for testing"""
    config = IntegrationConfig(
        test_mode=True,
        mock_external_services=True,
        database_url="sqlite:///test_fraud_detection.db",
        enable_message_queue=False,
        log_level="DEBUG"
    )
    return IntegrationManager(config)

# ======================== MAIN INTEGRATION UTILITIES ========================

def setup_complete_fraud_detection_system(config_file: Optional[str] = None) -> IntegrationManager:
    """Setup complete fraud detection system with all integrations"""
    
    # Load configuration
    if config_file:
        config = load_config_from_file(config_file)
    else:
        config = IntegrationConfig()
    
    # Create integration manager
    manager = IntegrationManager(config)
    
    # Initialize fraud core
    manager.initialize_fraud_core()
    
    logger.info("Complete fraud detection system initialized")
    return manager

def run_system_health_check(manager: IntegrationManager) -> Dict[str, Any]:
    """Run comprehensive system health check"""
    
    health_status = {
        'overall_health': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'checks': {}
    }
    
    try:
        # Check integration status
        integration_status = manager.get_integration_status()
        health_status['checks']['integrations'] = integration_status
        
        # Check fraud core status
        if manager.fraud_core:
            core_status = manager.fraud_core.get_system_status()
            health_status['checks']['fraud_core'] = core_status
        
        # Run basic functionality test
        test_transaction = {
            'transaction_id': 'health_check_tx',
            'user_id': 'health_check_user',
            'amount': 100.0,
            'merchant_id': 'health_check_merchant',
            'timestamp': datetime.now().isoformat(),
            'type': 'purchase'
        }
        
        result = manager.detect_fraud_with_integration(test_transaction)
        health_status['checks']['fraud_detection'] = {
            'status': 'healthy',
            'test_result': {
                'fraud_detected': result.fraud_detected,
                'detection_time': result.detection_time
            }
        }
        
    except Exception as e:
        health_status['overall_health'] = 'unhealthy'
        health_status['checks']['error'] = str(e)
        logger.error(f"Health check failed: {e}")
    
    return health_status