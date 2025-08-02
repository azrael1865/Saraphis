"""
Enhanced ML Predictor - Chunk 4: Advanced Features and Integration Utilities
Comprehensive advanced features, testing framework, deployment utilities, and system integration
for the enhanced ML predictor system.
"""

import logging
import json
import pickle
import joblib
import pandas as pd
import numpy as np
import pytest
import unittest
import time
import threading
import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager
from unittest.mock import Mock, patch, MagicMock
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib
import sqlite3
import asyncio
from abc import ABC, abstractmethod

# ML and data science imports
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import seaborn as sns

# Import enhanced ML components
try:
    from enhanced_ml_framework import (
        BaseModel, ModelConfig, ModelType, ValidationLevel,
        MLPredictorError, ModelNotFittedError, InvalidInputError,
        ModelValidationError, ModelTrainingError, PredictionError,
        HyperparameterError, InputValidator,
        validate_input, handle_errors, monitor_performance
    )
    from enhanced_ml_models import (
        EnhancedRandomForestModel, EnhancedXGBoostModel, EnhancedLightGBMModel,
        EnhancedLogisticRegressionModel, EnhancedEnsembleModel
    )
    from enhanced_ml_predictor import (
        EnhancedFinancialMLPredictor, ModelMetrics, PredictionResult,
        ModelVersion, ModelStatus, ModelDriftDetector
    )
    ML_COMPONENTS = True
except ImportError as e:
    ML_COMPONENTS = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Enhanced ML components not available: {e}")

# Import existing system components
try:
    from enhanced_data_validator import EnhancedFinancialDataValidator
    from enhanced_transaction_validator import EnhancedTransactionFieldValidator
    from enhanced_data_loader import EnhancedFinancialDataLoader
    from enhanced_preprocessing_integration import PreprocessingIntegrationManager
    EXISTING_COMPONENTS = True
except ImportError as e:
    EXISTING_COMPONENTS = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import existing components: {e}")

# Configure logging
logger = logging.getLogger(__name__)

# ======================== ADVANCED ML INTEGRATION CONFIGURATION ========================

@dataclass
class MLIntegrationConfig:
    """Configuration for ML system integration"""
    enable_preprocessing: bool = True
    enable_validation: bool = True
    enable_monitoring: bool = True
    enable_drift_detection: bool = True
    enable_auto_retraining: bool = True
    enable_model_versioning: bool = True
    enable_parallel_processing: bool = True
    enable_async_processing: bool = False
    
    # Performance settings
    max_worker_threads: int = 4
    max_worker_processes: int = 2
    batch_size: int = 1000
    prediction_timeout: int = 30
    training_timeout: int = 3600
    
    # Model management
    max_model_versions: int = 10
    model_retention_days: int = 30
    auto_deployment_enabled: bool = False
    auto_deployment_threshold: float = 0.95
    
    # Monitoring settings
    drift_check_interval: int = 3600  # 1 hour
    performance_log_interval: int = 300  # 5 minutes
    model_backup_interval: int = 86400  # 24 hours
    
    # Testing settings
    enable_test_mode: bool = False
    test_data_size: int = 1000
    mock_external_services: bool = False
    
    # Security settings
    enable_encryption: bool = True
    audit_logging: bool = True
    access_control: bool = True

class MLIntegrationError(MLPredictorError):
    """Raised when ML integration fails"""
    def __init__(self, message: str, component: str = None, **kwargs):
        super().__init__(message, error_code="ML_INTEGRATION_ERROR", **kwargs)
        self.component = component
        self.recoverable = True

class MLDeploymentError(MLPredictorError):
    """Raised when ML deployment fails"""
    def __init__(self, message: str, deployment_stage: str = None, **kwargs):
        super().__init__(message, error_code="ML_DEPLOYMENT_ERROR", **kwargs)
        self.deployment_stage = deployment_stage
        self.recoverable = False

class MLTestError(MLPredictorError):
    """Raised when ML testing fails"""
    def __init__(self, message: str, test_type: str = None, **kwargs):
        super().__init__(message, error_code="ML_TEST_ERROR", **kwargs)
        self.test_type = test_type
        self.recoverable = True

# ======================== ADVANCED ML INTEGRATION MANAGER ========================

class MLIntegrationManager:
    """Manages integration between ML predictor and existing system components"""
    
    def __init__(self, config: MLIntegrationConfig):
        self.config = config
        self.predictor = None
        self.validator = None
        self.loader = None
        self.preprocessor = None
        self.integration_state = {}
        self.performance_metrics = {}
        self.lock = threading.Lock()
        self.executor = None
        self.async_loop = None
        
        # Initialize components
        self._initialize_components()
        
        # Start monitoring if enabled
        if self.config.enable_monitoring:
            self._start_monitoring()
    
    def _initialize_components(self):
        """Initialize all ML integration components"""
        try:
            # Initialize ML predictor
            if ML_COMPONENTS:
                model_config = ModelConfig(
                    model_type=ModelType.ENSEMBLE,
                    fraud_threshold=0.5,
                    validation_level=ValidationLevel.STRICT,
                    enable_monitoring=self.config.enable_monitoring
                )
                self.predictor = EnhancedFinancialMLPredictor(model_config)
                
                # Initialize supporting components
                if EXISTING_COMPONENTS:
                    if self.config.enable_validation:
                        self.validator = EnhancedFinancialDataValidator()
                    
                    if self.config.enable_preprocessing:
                        from enhanced_preprocessing_integration import get_integrated_preprocessor
                        self.preprocessor = get_integrated_preprocessor("production")
                    
                    self.loader = EnhancedFinancialDataLoader()
                
                # Initialize thread pool if parallel processing enabled
                if self.config.enable_parallel_processing:
                    self.executor = ThreadPoolExecutor(max_workers=self.config.max_worker_threads)
                
                # Initialize async event loop if async processing enabled
                if self.config.enable_async_processing:
                    self.async_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self.async_loop)
                
                logger.info("ML integration components initialized successfully")
            else:
                logger.warning("ML components not available")
                
        except Exception as e:
            logger.error(f"Failed to initialize ML components: {e}")
            raise MLIntegrationError(f"Component initialization failed: {e}", component="initialization")
    
    def _start_monitoring(self):
        """Start background monitoring tasks"""
        if self.config.enable_monitoring:
            monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            monitoring_thread.start()
            logger.info("ML monitoring started")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                # Check drift detection
                if self.config.enable_drift_detection and self.predictor:
                    self._check_model_drift()
                
                # Log performance metrics
                self._log_performance_metrics()
                
                # Check for auto-retraining
                if self.config.enable_auto_retraining:
                    self._check_auto_retraining()
                
                # Cleanup old models
                self._cleanup_old_models()
                
                # Sleep until next check
                time.sleep(self.config.drift_check_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait before retrying
    
    @contextmanager
    def ml_session(self, session_id: str = None):
        """Context manager for ML processing sessions"""
        session_id = session_id or f"ml_session_{int(time.time())}"
        
        try:
            with self.lock:
                self.integration_state[session_id] = {
                    'start_time': datetime.now(),
                    'status': 'active',
                    'operations': [],
                    'metrics': {}
                }
            
            logger.info(f"Starting ML session: {session_id}")
            yield session_id
            
        except Exception as e:
            logger.error(f"ML session {session_id} failed: {e}")
            if session_id in self.integration_state:
                self.integration_state[session_id]['status'] = 'failed'
                self.integration_state[session_id]['error'] = str(e)
            raise
            
        finally:
            if session_id in self.integration_state:
                self.integration_state[session_id]['end_time'] = datetime.now()
                self.integration_state[session_id]['status'] = 'completed'
                self._cleanup_session(session_id)
    
    def _cleanup_session(self, session_id: str):
        """Clean up resources for an ML session"""
        try:
            if session_id in self.integration_state:
                session_data = self.integration_state[session_id]
                
                # Log session summary
                duration = (session_data.get('end_time', datetime.now()) - 
                           session_data['start_time']).total_seconds()
                
                logger.info(f"ML session {session_id} completed in {duration:.2f}s")
                
                # Keep only last 10 sessions
                if len(self.integration_state) > 10:
                    oldest_sessions = sorted(
                        self.integration_state.keys(),
                        key=lambda x: self.integration_state[x]['start_time']
                    )[:-10]
                    
                    for old_session in oldest_sessions:
                        del self.integration_state[old_session]
                        
        except Exception as e:
            logger.error(f"Session cleanup failed for {session_id}: {e}")
    
    def process_with_full_integration(self, data: pd.DataFrame, **kwargs) -> List[PredictionResult]:
        """Process data with full system integration"""
        with self.ml_session() as session_id:
            try:
                # Step 1: Data validation
                if self.config.enable_validation and self.validator:
                    validation_result = self.validator.validate_transaction_data(data)
                    if not validation_result.is_valid:
                        logger.warning(f"Validation issues found: {len(validation_result.issues)}")
                        
                        # Filter out critical issues
                        critical_issues = [i for i in validation_result.issues 
                                         if i.severity.value >= 4]
                        if critical_issues:
                            raise MLIntegrationError(
                                f"Critical validation issues: {len(critical_issues)}", 
                                component="validation"
                            )
                
                # Step 2: Data preprocessing
                processed_data = data
                if self.config.enable_preprocessing and self.preprocessor:
                    processed_data = self.preprocessor.process_data_integrated(data)
                
                # Step 3: ML prediction
                if not self.predictor:
                    raise MLIntegrationError("ML predictor not initialized", component="predictor")
                
                # Process in batches if data is large
                if len(processed_data) > self.config.batch_size:
                    results = self._process_in_batches(processed_data)
                else:
                    results = self.predictor.predict(processed_data)
                
                # Step 4: Post-processing validation
                validated_results = self._validate_predictions(results)
                
                # Step 5: Update session metrics
                self.integration_state[session_id]['operations'].append({
                    'operation': 'full_integration_processing',
                    'timestamp': datetime.now(),
                    'input_size': len(data),
                    'output_size': len(validated_results),
                    'success': True
                })
                
                logger.info(f"Full integration processing completed: {len(data)} -> {len(validated_results)}")
                return validated_results
                
            except Exception as e:
                logger.error(f"Full integration processing failed: {e}")
                
                # Update session state
                self.integration_state[session_id]['operations'].append({
                    'operation': 'full_integration_processing',
                    'timestamp': datetime.now(),
                    'error': str(e),
                    'success': False
                })
                
                raise
    
    def _process_in_batches(self, data: pd.DataFrame) -> List[PredictionResult]:
        """Process data in batches for better performance"""
        results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i + batch_size]
            batch_results = self.predictor.predict(batch)
            results.extend(batch_results)
            
            # Log progress
            if i % (batch_size * 10) == 0:
                logger.info(f"Processed {i + len(batch)}/{len(data)} records")
        
        return results
    
    def _validate_predictions(self, results: List[PredictionResult]) -> List[PredictionResult]:
        """Validate prediction results"""
        validated_results = []
        
        for result in results:
            # Check probability bounds
            if not 0 <= result.fraud_probability <= 1:
                logger.warning(f"Invalid probability for {result.transaction_id}: {result.fraud_probability}")
                continue
            
            # Check risk score
            if result.risk_score < 0:
                logger.warning(f"Invalid risk score for {result.transaction_id}: {result.risk_score}")
                continue
            
            # Check model version
            if not result.model_version:
                logger.warning(f"Missing model version for {result.transaction_id}")
                result.model_version = "unknown"
            
            validated_results.append(result)
        
        return validated_results
    
    def _check_model_drift(self):
        """Check for model drift"""
        try:
            if hasattr(self.predictor, 'drift_detector'):
                drift_score = self.predictor.drift_detector.check_drift()
                if drift_score > 0.1:  # Threshold for drift
                    logger.warning(f"Model drift detected: {drift_score:.3f}")
                    
                    if self.config.enable_auto_retraining:
                        logger.info("Triggering auto-retraining due to drift")
                        self._trigger_auto_retraining()
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
    
    def _log_performance_metrics(self):
        """Log performance metrics"""
        try:
            if self.predictor:
                stats = self.predictor.get_statistics()
                
                # Log key metrics
                logger.info(f"ML Performance - Predictions: {stats.get('total_predictions', 0)}, "
                           f"Failures: {stats.get('failed_predictions', 0)}, "
                           f"Success Rate: {stats.get('success_rate', 0):.2%}")
                
                # Store metrics
                self.performance_metrics[datetime.now().isoformat()] = stats
                
                # Keep only recent metrics
                if len(self.performance_metrics) > 100:
                    old_keys = list(self.performance_metrics.keys())[:-50]
                    for key in old_keys:
                        del self.performance_metrics[key]
                        
        except Exception as e:
            logger.error(f"Performance logging failed: {e}")
    
    def _check_auto_retraining(self):
        """Check if auto-retraining should be triggered"""
        try:
            if self.predictor and hasattr(self.predictor, 'get_statistics'):
                stats = self.predictor.get_statistics()
                success_rate = stats.get('success_rate', 1.0)
                
                if success_rate < self.config.auto_deployment_threshold:
                    logger.info(f"Success rate {success_rate:.2%} below threshold, considering retraining")
                    # Implement auto-retraining logic here
                    
        except Exception as e:
            logger.error(f"Auto-retraining check failed: {e}")
    
    def _trigger_auto_retraining(self):
        """Trigger automatic model retraining"""
        try:
            logger.info("Starting automatic model retraining")
            
            # Load recent data for retraining
            if self.loader:
                recent_data = self.loader.load_recent_data(days=30)
                if len(recent_data) > 1000:  # Minimum data requirement
                    
                    # Preprocess data
                    if self.preprocessor:
                        processed_data = self.preprocessor.process_data_integrated(recent_data)
                    else:
                        processed_data = recent_data
                    
                    # Retrain model
                    if 'is_fraud' in processed_data.columns:
                        X = processed_data.drop('is_fraud', axis=1)
                        y = processed_data['is_fraud']
                        
                        # Train new model version
                        success = self.predictor.train(X, y)
                        if success:
                            logger.info("Auto-retraining completed successfully")
                        else:
                            logger.error("Auto-retraining failed")
                    else:
                        logger.error("Target variable 'is_fraud' not found in training data")
                else:
                    logger.warning(f"Insufficient data for retraining: {len(recent_data)} records")
            else:
                logger.error("Data loader not available for auto-retraining")
                
        except Exception as e:
            logger.error(f"Auto-retraining failed: {e}")
    
    def _cleanup_old_models(self):
        """Clean up old model versions"""
        try:
            if self.predictor and hasattr(self.predictor, 'model_versions'):
                cutoff_date = datetime.now() - timedelta(days=self.config.model_retention_days)
                
                versions_to_remove = []
                for version_id, version in self.predictor.model_versions.items():
                    if (version.created_at < cutoff_date and 
                        not version.is_active and 
                        len(self.predictor.model_versions) > self.config.max_model_versions):
                        versions_to_remove.append(version_id)
                
                for version_id in versions_to_remove:
                    del self.predictor.model_versions[version_id]
                    logger.info(f"Removed old model version: {version_id}")
                    
        except Exception as e:
            logger.error(f"Model cleanup failed: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        return {
            'components': {
                'predictor': self.predictor is not None,
                'validator': self.validator is not None,
                'loader': self.loader is not None,
                'preprocessor': self.preprocessor is not None
            },
            'config': asdict(self.config),
            'active_sessions': len([s for s in self.integration_state.values() 
                                  if s['status'] == 'active']),
            'total_sessions': len(self.integration_state),
            'performance_metrics': self.performance_metrics,
            'predictor_stats': self.predictor.get_statistics() if self.predictor else {}
        }
    
    def shutdown(self):
        """Shutdown integration manager and cleanup resources"""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            
            if self.async_loop:
                self.async_loop.close()
            
            logger.info("ML integration manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Shutdown failed: {e}")

# ======================== COMPREHENSIVE TESTING FRAMEWORK ========================

class MLTestSuite:
    """Comprehensive test suite for enhanced ML system"""
    
    def __init__(self, config: MLIntegrationConfig = None):
        self.config = config or MLIntegrationConfig(enable_test_mode=True)
        self.test_data_cache = {}
        self.test_results = {}
        self.integration_manager = None
        
    def setup_test_environment(self):
        """Set up test environment"""
        try:
            # Initialize integration manager with test config
            test_config = MLIntegrationConfig(
                enable_test_mode=True,
                enable_monitoring=False,
                enable_auto_retraining=False,
                mock_external_services=True,
                max_worker_threads=2,
                batch_size=100
            )
            
            self.integration_manager = MLIntegrationManager(test_config)
            logger.info("Test environment setup completed")
            
        except Exception as e:
            logger.error(f"Test environment setup failed: {e}")
            raise MLTestError(f"Test setup failed: {e}", test_type="setup")
    
    def generate_test_data(self, size: int = 1000, data_type: str = "normal") -> pd.DataFrame:
        """Generate test data for ML testing"""
        if f"{data_type}_{size}" in self.test_data_cache:
            return self.test_data_cache[f"{data_type}_{size}"]
        
        np.random.seed(42)  # For reproducibility
        
        if data_type == "normal":
            data = pd.DataFrame({
                'transaction_id': [f'TEST_TXN_{i:08d}' for i in range(size)],
                'user_id': [f'TEST_USER_{i%100:06d}' for i in range(size)],
                'amount': np.random.exponential(100, size),
                'timestamp': pd.date_range('2023-01-01', periods=size, freq='1H'),
                'merchant_id': [f'TEST_MERCHANT_{i%50:04d}' for i in range(size)],
                'payment_method': np.random.choice(['credit_card', 'debit_card', 'bank_transfer'], size),
                'currency': np.random.choice(['USD', 'EUR', 'GBP'], size),
                'country': np.random.choice(['US', 'UK', 'DE'], size),
                'is_fraud': np.random.choice([0, 1], size, p=[0.95, 0.05])
            })
        
        elif data_type == "anomalous":
            data = pd.DataFrame({
                'transaction_id': [f'ANOMALY_TXN_{i:08d}' for i in range(size)],
                'user_id': [f'ANOMALY_USER_{i%10:06d}' for i in range(size)],
                'amount': np.random.exponential(10000, size),  # Much higher amounts
                'timestamp': pd.date_range('2023-01-01', periods=size, freq='1min'),  # Rapid transactions
                'merchant_id': [f'ANOMALY_MERCHANT_{i%5:04d}' for i in range(size)],
                'payment_method': np.random.choice(['credit_card'], size),  # Single payment method
                'currency': np.random.choice(['USD'], size),
                'country': np.random.choice(['US'], size),
                'is_fraud': np.random.choice([0, 1], size, p=[0.5, 0.5])  # Higher fraud rate
            })
        
        elif data_type == "corrupted":
            data = pd.DataFrame({
                'transaction_id': ['INVALID_ID'] * size,
                'user_id': [None] * size,
                'amount': ['not_a_number'] * size,
                'timestamp': ['invalid_date'] * size,
                'merchant_id': [''] * size,
                'payment_method': ['INVALID'] * size,
                'currency': ['XXX'] * size,
                'country': ['INVALID'] * size,
                'is_fraud': [2] * size  # Invalid values
            })
        
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        self.test_data_cache[f"{data_type}_{size}"] = data
        return data
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        try:
            self.setup_test_environment()
            
            test_results = {
                'unit_tests': self._run_unit_tests(),
                'integration_tests': self._run_integration_tests(),
                'performance_tests': self._run_performance_tests(),
                'stress_tests': self._run_stress_tests(),
                'security_tests': self._run_security_tests()
            }
            
            # Calculate overall results
            total_tests = sum(len(category) for category in test_results.values())
            passed_tests = sum(
                sum(1 for test in category.values() if test.get('passed', False))
                for category in test_results.values()
            )
            
            test_results['summary'] = {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'timestamp': datetime.now().isoformat()
            }
            
            return test_results
            
        except Exception as e:
            logger.error(f"Comprehensive tests failed: {e}")
            raise MLTestError(f"Test execution failed: {e}", test_type="comprehensive")
    
    def _run_unit_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run unit tests for ML components"""
        tests = {}
        
        # Test ML framework components
        tests['ml_framework'] = self._run_test(
            "ML framework components",
            lambda: self._test_ml_framework()
        )
        
        # Test individual models
        tests['individual_models'] = self._run_test(
            "Individual ML models",
            lambda: self._test_individual_models()
        )
        
        # Test main predictor
        tests['main_predictor'] = self._run_test(
            "Main ML predictor",
            lambda: self._test_main_predictor()
        )
        
        # Test integration manager
        tests['integration_manager'] = self._run_test(
            "Integration manager",
            lambda: self._test_integration_manager()
        )
        
        return tests
    
    def _run_integration_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run integration tests"""
        tests = {}
        
        # Test full processing pipeline
        tests['full_pipeline'] = self._run_test(
            "Full processing pipeline",
            lambda: self._test_full_pipeline()
        )
        
        # Test component integration
        tests['component_integration'] = self._run_test(
            "Component integration",
            lambda: self._test_component_integration()
        )
        
        # Test error handling
        tests['error_handling'] = self._run_test(
            "Error handling",
            lambda: self._test_error_handling()
        )
        
        return tests
    
    def _run_performance_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run performance tests"""
        tests = {}
        
        # Test prediction speed
        tests['prediction_speed'] = self._run_test(
            "Prediction speed",
            lambda: self._test_prediction_speed()
        )
        
        # Test batch processing
        tests['batch_processing'] = self._run_test(
            "Batch processing",
            lambda: self._test_batch_processing()
        )
        
        # Test memory usage
        tests['memory_usage'] = self._run_test(
            "Memory usage",
            lambda: self._test_memory_usage()
        )
        
        return tests
    
    def _run_stress_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run stress tests"""
        tests = {}
        
        # Test high volume processing
        tests['high_volume'] = self._run_test(
            "High volume processing",
            lambda: self._test_high_volume_processing()
        )
        
        # Test concurrent processing
        tests['concurrent_processing'] = self._run_test(
            "Concurrent processing",
            lambda: self._test_concurrent_processing()
        )
        
        return tests
    
    def _run_security_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run security tests"""
        tests = {}
        
        # Test input validation
        tests['input_validation'] = self._run_test(
            "Input validation",
            lambda: self._test_input_validation()
        )
        
        # Test data sanitization
        tests['data_sanitization'] = self._run_test(
            "Data sanitization",
            lambda: self._test_data_sanitization()
        )
        
        return tests
    
    def _run_test(self, test_name: str, test_func: Callable) -> Dict[str, Any]:
        """Run a single test with error handling"""
        try:
            start_time = time.time()
            test_func()
            duration = time.time() - start_time
            
            return {
                'passed': True,
                'duration': duration,
                'message': f"{test_name} passed",
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'passed': False,
                'duration': time.time() - start_time if 'start_time' in locals() else 0,
                'message': f"{test_name} failed: {str(e)}",
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _test_ml_framework(self):
        """Test ML framework components"""
        if not ML_COMPONENTS:
            return
        
        # Test exception handling
        exc = MLPredictorError("Test error")
        assert exc.error_code == "ML_PREDICTOR_ERROR"
        
        # Test model configuration
        config = ModelConfig(model_type=ModelType.RANDOM_FOREST)
        assert config.model_type == ModelType.RANDOM_FOREST
        
        # Test input validator
        validator = InputValidator()
        test_data = self.generate_test_data(100)
        validator.validate_input(test_data)
    
    def _test_individual_models(self):
        """Test individual ML models"""
        if not ML_COMPONENTS:
            return
        
        # Test model creation
        config = ModelConfig(model_type=ModelType.RANDOM_FOREST)
        model = EnhancedRandomForestModel(config)
        
        # Test model training
        test_data = self.generate_test_data(100)
        if 'is_fraud' in test_data.columns:
            X = test_data.drop('is_fraud', axis=1).select_dtypes(include=[np.number])
            y = test_data['is_fraud']
            
            model.fit(X, y)
            assert model.fitted
            
            # Test prediction
            predictions = model.predict(X)
            assert len(predictions) == len(X)
    
    def _test_main_predictor(self):
        """Test main ML predictor"""
        if not ML_COMPONENTS:
            return
        
        # Test predictor initialization
        config = ModelConfig(model_type=ModelType.RANDOM_FOREST)
        predictor = EnhancedFinancialMLPredictor(config)
        
        # Test training
        test_data = self.generate_test_data(100)
        if 'is_fraud' in test_data.columns:
            X = test_data.drop('is_fraud', axis=1).select_dtypes(include=[np.number])
            y = test_data['is_fraud']
            
            success = predictor.train(X, y)
            assert success
            
            # Test prediction
            results = predictor.predict(X)
            assert len(results) == len(X)
            assert all(isinstance(r, PredictionResult) for r in results)
    
    def _test_integration_manager(self):
        """Test integration manager"""
        if not self.integration_manager:
            return
        
        # Test session management
        with self.integration_manager.ml_session("test_session") as session_id:
            assert session_id == "test_session"
            assert session_id in self.integration_manager.integration_state
        
        # Test status retrieval
        status = self.integration_manager.get_integration_status()
        assert 'components' in status
        assert 'config' in status
    
    def _test_full_pipeline(self):
        """Test full processing pipeline"""
        if not self.integration_manager:
            return
        
        # Test with normal data
        test_data = self.generate_test_data(50)
        results = self.integration_manager.process_with_full_integration(test_data)
        
        assert results is not None
        assert len(results) > 0
        assert all(isinstance(r, PredictionResult) for r in results)
    
    def _test_component_integration(self):
        """Test component integration"""
        if not self.integration_manager:
            return
        
        # Test individual component access
        assert self.integration_manager.predictor is not None
        
        # Test component interaction
        status = self.integration_manager.get_integration_status()
        components = status['components']
        
        # At least predictor should be available
        assert components['predictor'] is True
    
    def _test_error_handling(self):
        """Test error handling"""
        if not self.integration_manager:
            return
        
        # Test with corrupted data
        try:
            corrupted_data = self.generate_test_data(10, "corrupted")
            results = self.integration_manager.process_with_full_integration(corrupted_data)
            # Should either succeed with cleaned data or fail gracefully
        except MLIntegrationError:
            pass  # Expected for corrupted data
    
    def _test_prediction_speed(self):
        """Test prediction speed"""
        if not self.integration_manager:
            return
        
        test_data = self.generate_test_data(100)
        
        start_time = time.time()
        results = self.integration_manager.process_with_full_integration(test_data)
        duration = time.time() - start_time
        
        # Should process 100 records in under 10 seconds
        assert duration < 10.0
        assert len(results) == len(test_data)
    
    def _test_batch_processing(self):
        """Test batch processing"""
        if not self.integration_manager:
            return
        
        # Test with data larger than batch size
        test_data = self.generate_test_data(250)  # Larger than default batch size
        results = self.integration_manager.process_with_full_integration(test_data)
        
        assert len(results) == len(test_data)
    
    def _test_memory_usage(self):
        """Test memory usage"""
        if not self.integration_manager:
            return
        
        # Test with moderate data size
        test_data = self.generate_test_data(500)
        
        # Monitor memory usage during processing
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        results = self.integration_manager.process_with_full_integration(test_data)
        
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable (under 100MB)
        assert memory_increase < 100 * 1024 * 1024
    
    def _test_high_volume_processing(self):
        """Test high volume processing"""
        if not self.integration_manager:
            return
        
        # Test with large data
        test_data = self.generate_test_data(2000)
        
        start_time = time.time()
        results = self.integration_manager.process_with_full_integration(test_data)
        duration = time.time() - start_time
        
        # Should handle large volume efficiently
        assert len(results) == len(test_data)
        assert duration < 60.0  # Under 1 minute
    
    def _test_concurrent_processing(self):
        """Test concurrent processing"""
        if not self.integration_manager:
            return
        
        def process_data():
            test_data = self.generate_test_data(100)
            return self.integration_manager.process_with_full_integration(test_data)
        
        # Test concurrent execution
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_data) for _ in range(3)]
            results = [future.result() for future in futures]
        
        # All should succeed
        assert len(results) == 3
        assert all(len(r) == 100 for r in results)
    
    def _test_input_validation(self):
        """Test input validation"""
        if not self.integration_manager:
            return
        
        # Test with invalid data types
        invalid_data = pd.DataFrame({
            'invalid_column': [1, 2, 3]
        })
        
        try:
            results = self.integration_manager.process_with_full_integration(invalid_data)
            # Should handle gracefully
        except MLIntegrationError:
            pass  # Expected for invalid data
    
    def _test_data_sanitization(self):
        """Test data sanitization"""
        if not self.integration_manager:
            return
        
        # Test with data containing potential security issues
        test_data = self.generate_test_data(50)
        
        # Add some potentially malicious data
        test_data.loc[0, 'user_id'] = "'; DROP TABLE users; --"
        test_data.loc[1, 'merchant_id'] = "<script>alert('xss')</script>"
        
        # Should process without issues
        results = self.integration_manager.process_with_full_integration(test_data)
        assert len(results) > 0

# ======================== DEPLOYMENT UTILITIES ========================

class MLDeploymentManager:
    """Manages ML model deployment and lifecycle"""
    
    def __init__(self, integration_manager: MLIntegrationManager):
        self.integration_manager = integration_manager
        self.deployment_history = []
        self.deployment_lock = threading.Lock()
    
    def deploy_model(self, model_version: str, environment: str = "production") -> bool:
        """Deploy a model version to specified environment"""
        try:
            with self.deployment_lock:
                logger.info(f"Deploying model {model_version} to {environment}")
                
                # Validate model version
                if not self._validate_model_version(model_version):
                    raise MLDeploymentError(
                        f"Model version {model_version} validation failed",
                        deployment_stage="validation"
                    )
                
                # Deploy model
                success = self.integration_manager.predictor.deploy_model(model_version)
                
                if success:
                    # Record deployment
                    self.deployment_history.append({
                        'model_version': model_version,
                        'environment': environment,
                        'timestamp': datetime.now(),
                        'status': 'successful'
                    })
                    
                    logger.info(f"Model {model_version} deployed successfully to {environment}")
                    return True
                else:
                    raise MLDeploymentError(
                        f"Model deployment failed for {model_version}",
                        deployment_stage="deployment"
                    )
                    
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            
            # Record failed deployment
            self.deployment_history.append({
                'model_version': model_version,
                'environment': environment,
                'timestamp': datetime.now(),
                'status': 'failed',
                'error': str(e)
            })
            
            return False
    
    def _validate_model_version(self, model_version: str) -> bool:
        """Validate model version before deployment"""
        try:
            if not self.integration_manager.predictor:
                return False
            
            # Check if version exists
            if model_version not in self.integration_manager.predictor.model_versions:
                return False
            
            version_info = self.integration_manager.predictor.model_versions[model_version]
            
            # Check if version is validated
            if version_info.status != ModelStatus.VALIDATED:
                return False
            
            # Check model performance
            if hasattr(version_info, 'metrics'):
                if version_info.metrics.accuracy < 0.8:  # Minimum accuracy threshold
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    def rollback_deployment(self, target_version: str = None) -> bool:
        """Rollback to previous model version"""
        try:
            with self.deployment_lock:
                if target_version:
                    logger.info(f"Rolling back to model version {target_version}")
                    return self.deploy_model(target_version)
                else:
                    # Find previous successful deployment
                    successful_deployments = [
                        d for d in self.deployment_history 
                        if d['status'] == 'successful'
                    ]
                    
                    if len(successful_deployments) >= 2:
                        previous_version = successful_deployments[-2]['model_version']
                        logger.info(f"Rolling back to previous version {previous_version}")
                        return self.deploy_model(previous_version)
                    else:
                        logger.error("No previous successful deployment found")
                        return False
                        
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def get_deployment_history(self) -> List[Dict[str, Any]]:
        """Get deployment history"""
        return self.deployment_history.copy()

# ======================== FACTORY AND MAIN INTERFACE ========================

class MLIntegrationFactory:
    """Factory for creating ML integration systems"""
    
    @staticmethod
    def create_development_system() -> MLIntegrationManager:
        """Create ML system for development environment"""
        config = MLIntegrationConfig(
            enable_preprocessing=True,
            enable_validation=True,
            enable_monitoring=True,
            enable_drift_detection=True,
            enable_auto_retraining=False,
            enable_test_mode=True,
            max_worker_threads=2,
            batch_size=500
        )
        
        return MLIntegrationManager(config)
    
    @staticmethod
    def create_production_system() -> MLIntegrationManager:
        """Create ML system for production environment"""
        config = MLIntegrationConfig(
            enable_preprocessing=True,
            enable_validation=True,
            enable_monitoring=True,
            enable_drift_detection=True,
            enable_auto_retraining=True,
            enable_test_mode=False,
            max_worker_threads=4,
            batch_size=1000,
            prediction_timeout=30,
            training_timeout=3600,
            audit_logging=True
        )
        
        return MLIntegrationManager(config)
    
    @staticmethod
    def create_testing_system() -> MLIntegrationManager:
        """Create ML system for testing"""
        config = MLIntegrationConfig(
            enable_preprocessing=False,
            enable_validation=False,
            enable_monitoring=False,
            enable_drift_detection=False,
            enable_auto_retraining=False,
            enable_test_mode=True,
            max_worker_threads=1,
            batch_size=100,
            mock_external_services=True
        )
        
        return MLIntegrationManager(config)

# ======================== MAIN INTEGRATION INTERFACE ========================

def get_integrated_ml_system(environment: str = "development") -> MLIntegrationManager:
    """Get configured ML system for specified environment"""
    factory = MLIntegrationFactory()
    
    if environment == "development":
        return factory.create_development_system()
    elif environment == "production":
        return factory.create_production_system()
    elif environment == "testing":
        return factory.create_testing_system()
    else:
        raise ValueError(f"Unknown environment: {environment}")

def run_ml_tests() -> Dict[str, Any]:
    """Run comprehensive ML tests"""
    test_suite = MLTestSuite()
    return test_suite.run_comprehensive_tests()

# Export integration components
__all__ = [
    # Configuration
    'MLIntegrationConfig',
    
    # Core integration
    'MLIntegrationManager',
    
    # Testing
    'MLTestSuite',
    
    # Deployment
    'MLDeploymentManager',
    
    # Factory
    'MLIntegrationFactory',
    
    # Exceptions
    'MLIntegrationError',
    'MLDeploymentError',
    'MLTestError',
    
    # Main interface
    'get_integrated_ml_system',
    'run_ml_tests'
]

if __name__ == "__main__":
    print("Enhanced ML Predictor - Chunk 4: Advanced features and integration utilities loaded")
    
    # Run basic integration test
    try:
        ml_system = get_integrated_ml_system("development")
        print("✓ ML integration system created successfully")
        
        # Run tests
        test_results = run_ml_tests()
        print(f"✓ Tests completed: {test_results['summary']['passed_tests']}/{test_results['summary']['total_tests']} passed")
        
    except Exception as e:
        print(f"✗ ML integration test failed: {e}")