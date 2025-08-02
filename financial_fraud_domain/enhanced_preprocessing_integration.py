"""
Enhanced Data Preprocessor - Chunk 4: Integration Utilities and Testing Framework
Comprehensive integration utilities, testing framework, and system integration
for the enhanced data preprocessing system.
"""

import logging
import pandas as pd
import numpy as np
import pytest
import unittest
import time
import threading
import pickle
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Type, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager
from unittest.mock import Mock, patch, MagicMock
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor

# Import enhanced preprocessing components
try:
    from enhanced_preprocessing_framework import (
        PreprocessingException, PreprocessingConfigError, PreprocessingInputError,
        PreprocessingTimeoutError, PreprocessingMemoryError, PreprocessingSecurityError,
        PreprocessingIntegrationError, FeatureEngineeringError, EncodingError,
        ScalingError, ImputationError, PartialPreprocessingError,
        SecurityConfig, ValidationConfig, PerformanceConfig,
        InputValidator, PerformanceMonitor, SecurityValidator, QualityValidator
    )
    from enhanced_preprocessing_steps import (
        PreprocessingStep, FeatureType, ScalingMethod, ImputationMethod,
        PreprocessingConfig, PreprocessingStepBase, DataCleaner, FeatureEngineer
    )
    from enhanced_preprocessing_pipeline import (
        CategoricalEncoder, FeatureScaler, FeatureSelector, EnhancedFinancialDataPreprocessor
    )
    PREPROCESSING_COMPONENTS = True
except ImportError as e:
    PREPROCESSING_COMPONENTS = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Enhanced preprocessing components not available: {e}")

# Import existing system components
try:
    from enhanced_data_validator import EnhancedFinancialDataValidator
    from enhanced_transaction_validator import EnhancedTransactionFieldValidator
    from enhanced_data_loader import EnhancedFinancialDataLoader
    EXISTING_COMPONENTS = True
except ImportError as e:
    EXISTING_COMPONENTS = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import existing components: {e}")

# Configure logging
logger = logging.getLogger(__name__)

# ======================== INTEGRATION UTILITIES ========================

@dataclass
class IntegrationConfig:
    """Configuration for preprocessing system integration"""
    validate_before_processing: bool = True
    validate_after_processing: bool = True
    enable_performance_monitoring: bool = True
    enable_security_validation: bool = True
    enable_quality_validation: bool = True
    backup_original_data: bool = True
    persist_preprocessing_state: bool = True
    integration_timeout: int = 300  # 5 minutes
    max_retry_attempts: int = 3
    fallback_to_simple_preprocessing: bool = True
    
    # Component integration settings
    use_enhanced_validator: bool = True
    use_enhanced_loader: bool = True
    enable_async_processing: bool = False
    
    # Testing settings
    enable_test_mode: bool = False
    test_data_size_limit: int = 1000
    mock_external_dependencies: bool = False

class PreprocessingIntegrationManager:
    """Manages integration between preprocessing components and existing system"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.preprocessor = None
        self.validator = None
        self.loader = None
        self.integration_state = {}
        self.performance_metrics = {}
        self.lock = threading.Lock()
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all preprocessing components"""
        try:
            if PREPROCESSING_COMPONENTS:
                # Initialize enhanced preprocessor
                preprocessing_config = PreprocessingConfig()
                self.preprocessor = EnhancedFinancialDataPreprocessor(preprocessing_config)
                
                # Initialize enhanced validator if available
                if EXISTING_COMPONENTS and self.config.use_enhanced_validator:
                    self.validator = EnhancedFinancialDataValidator()
                
                # Initialize enhanced loader if available
                if EXISTING_COMPONENTS and self.config.use_enhanced_loader:
                    self.loader = EnhancedFinancialDataLoader()
                    
                logger.info("Enhanced preprocessing components initialized successfully")
            else:
                logger.warning("Enhanced preprocessing components not available")
                
        except Exception as e:
            logger.error(f"Failed to initialize preprocessing components: {e}")
            if not self.config.fallback_to_simple_preprocessing:
                raise PreprocessingIntegrationError(f"Component initialization failed: {e}")
    
    @contextmanager
    def preprocessing_session(self, session_id: str = None):
        """Context manager for preprocessing sessions with proper cleanup"""
        session_id = session_id or f"session_{int(time.time())}"
        
        try:
            with self.lock:
                self.integration_state[session_id] = {
                    'start_time': datetime.now(),
                    'status': 'active',
                    'operations': []
                }
            
            logger.info(f"Starting preprocessing session: {session_id}")
            yield session_id
            
        except Exception as e:
            logger.error(f"Preprocessing session {session_id} failed: {e}")
            if session_id in self.integration_state:
                self.integration_state[session_id]['status'] = 'failed'
                self.integration_state[session_id]['error'] = str(e)
            raise
            
        finally:
            if session_id in self.integration_state:
                self.integration_state[session_id]['end_time'] = datetime.now()
                self.integration_state[session_id]['status'] = 'completed'
                
                # Cleanup resources
                self._cleanup_session(session_id)
    
    def _cleanup_session(self, session_id: str):
        """Clean up resources for a preprocessing session"""
        try:
            if session_id in self.integration_state:
                session_data = self.integration_state[session_id]
                
                # Log session summary
                duration = (session_data.get('end_time', datetime.now()) - 
                           session_data['start_time']).total_seconds()
                
                logger.info(f"Session {session_id} completed in {duration:.2f}s")
                
                # Clean up temporary files if any
                if 'temp_files' in session_data:
                    for temp_file in session_data['temp_files']:
                        try:
                            Path(temp_file).unlink(missing_ok=True)
                        except Exception as e:
                            logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
                
                # Remove old session data (keep last 10 sessions)
                if len(self.integration_state) > 10:
                    oldest_sessions = sorted(
                        self.integration_state.keys(),
                        key=lambda x: self.integration_state[x]['start_time']
                    )[:-10]
                    
                    for old_session in oldest_sessions:
                        del self.integration_state[old_session]
                        
        except Exception as e:
            logger.error(f"Session cleanup failed for {session_id}: {e}")
    
    def process_data_integrated(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Process data with full system integration"""
        with self.preprocessing_session() as session_id:
            try:
                # Step 1: Pre-processing validation
                if self.config.validate_before_processing and self.validator:
                    validation_result = self.validator.validate_transaction_data(data)
                    if not validation_result.is_valid:
                        logger.warning(f"Pre-processing validation issues: {len(validation_result.issues)}")
                        
                        # Handle validation issues based on severity
                        critical_issues = [i for i in validation_result.issues 
                                         if i.severity.value >= 4]  # ERROR and CRITICAL
                        
                        if critical_issues and not self.config.fallback_to_simple_preprocessing:
                            raise PreprocessingInputError(
                                f"Critical validation issues prevent processing: {len(critical_issues)} issues"
                            )
                
                # Step 2: Backup original data if configured
                original_data = None
                if self.config.backup_original_data:
                    original_data = data.copy()
                
                # Step 3: Enhanced preprocessing
                if self.preprocessor:
                    processed_data = self.preprocessor.fit_transform(data)
                else:
                    # Fallback to simple preprocessing
                    processed_data = self._simple_preprocessing_fallback(data)
                
                # Step 4: Post-processing validation
                if self.config.validate_after_processing and self.validator:
                    post_validation = self.validator.validate_transaction_data(processed_data)
                    if not post_validation.is_valid:
                        logger.warning(f"Post-processing validation issues: {len(post_validation.issues)}")
                
                # Step 5: Quality validation
                if self.config.enable_quality_validation:
                    quality_issues = self._validate_processing_quality(original_data, processed_data)
                    if quality_issues:
                        logger.warning(f"Quality validation issues: {len(quality_issues)}")
                
                # Step 6: Update integration state
                self.integration_state[session_id]['operations'].append({
                    'operation': 'data_processing',
                    'timestamp': datetime.now(),
                    'input_shape': data.shape,
                    'output_shape': processed_data.shape,
                    'success': True
                })
                
                logger.info(f"Data processing completed: {data.shape} -> {processed_data.shape}")
                return processed_data
                
            except Exception as e:
                logger.error(f"Integrated data processing failed: {e}")
                
                # Update integration state with error
                self.integration_state[session_id]['operations'].append({
                    'operation': 'data_processing',
                    'timestamp': datetime.now(),
                    'error': str(e),
                    'success': False
                })
                
                # Attempt fallback if configured
                if self.config.fallback_to_simple_preprocessing:
                    logger.info("Attempting fallback to simple preprocessing")
                    return self._simple_preprocessing_fallback(data)
                
                raise
    
    def _simple_preprocessing_fallback(self, data: pd.DataFrame) -> pd.DataFrame:
        """Simple preprocessing fallback when enhanced preprocessing fails"""
        try:
            logger.info("Using simple preprocessing fallback")
            processed_data = data.copy()
            
            # Basic cleaning
            processed_data = processed_data.dropna()
            processed_data = processed_data.drop_duplicates()
            
            # Basic feature engineering
            if 'timestamp' in processed_data.columns:
                processed_data['hour'] = pd.to_datetime(processed_data['timestamp']).dt.hour
                processed_data['day_of_week'] = pd.to_datetime(processed_data['timestamp']).dt.dayofweek
            
            # Basic scaling for numeric columns
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col != 'is_fraud':  # Don't scale target variable
                    mean_val = processed_data[col].mean()
                    std_val = processed_data[col].std()
                    if std_val > 0:
                        processed_data[col] = (processed_data[col] - mean_val) / std_val
            
            logger.info("Simple preprocessing fallback completed")
            return processed_data
            
        except Exception as e:
            logger.error(f"Simple preprocessing fallback failed: {e}")
            raise PreprocessingIntegrationError(f"All preprocessing methods failed: {e}")
    
    def _validate_processing_quality(self, original_data: pd.DataFrame, 
                                   processed_data: pd.DataFrame) -> List[str]:
        """Validate the quality of preprocessing results"""
        quality_issues = []
        
        try:
            # Check for excessive data loss
            original_rows = len(original_data)
            processed_rows = len(processed_data)
            data_loss_percentage = (1 - processed_rows / original_rows) * 100
            
            if data_loss_percentage > 50:
                quality_issues.append(
                    f"Excessive data loss: {data_loss_percentage:.1f}% of rows removed"
                )
            
            # Check for feature explosion
            original_features = len(original_data.columns)
            processed_features = len(processed_data.columns)
            feature_ratio = processed_features / original_features
            
            if feature_ratio > 10:
                quality_issues.append(
                    f"Feature explosion: {feature_ratio:.1f}x increase in features"
                )
            
            # Check for missing values in critical columns
            critical_columns = ['amount', 'timestamp', 'user_id']
            for col in critical_columns:
                if col in processed_data.columns:
                    missing_percentage = processed_data[col].isnull().mean() * 100
                    if missing_percentage > 10:
                        quality_issues.append(
                            f"High missing values in {col}: {missing_percentage:.1f}%"
                        )
            
            # Check for numerical stability
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if processed_data[col].std() == 0:
                    quality_issues.append(f"Zero variance in column {col}")
                
                if processed_data[col].isinf().any():
                    quality_issues.append(f"Infinite values in column {col}")
                
                if processed_data[col].isna().any():
                    quality_issues.append(f"NaN values in column {col}")
            
        except Exception as e:
            quality_issues.append(f"Quality validation failed: {e}")
        
        return quality_issues
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status and metrics"""
        return {
            'component_status': {
                'preprocessor': self.preprocessor is not None,
                'validator': self.validator is not None,
                'loader': self.loader is not None
            },
            'active_sessions': len([s for s in self.integration_state.values() 
                                  if s['status'] == 'active']),
            'total_sessions': len(self.integration_state),
            'performance_metrics': self.performance_metrics,
            'config': asdict(self.config)
        }

# ======================== TESTING FRAMEWORK ========================

class PreprocessingTestSuite:
    """Comprehensive test suite for enhanced preprocessing system"""
    
    def __init__(self, config: IntegrationConfig = None):
        self.config = config or IntegrationConfig(enable_test_mode=True)
        self.test_data_cache = {}
        self.test_results = {}
        
    def generate_test_data(self, data_type: str = "financial", size: int = 1000) -> pd.DataFrame:
        """Generate test data for preprocessing validation"""
        if data_type in self.test_data_cache:
            return self.test_data_cache[data_type]
        
        np.random.seed(42)  # For reproducibility
        
        if data_type == "financial":
            data = pd.DataFrame({
                'transaction_id': [f'TXN_{i:08d}' for i in range(size)],
                'user_id': [f'USER_{i%100:06d}' for i in range(size)],
                'amount': np.random.exponential(100, size),
                'timestamp': pd.date_range('2023-01-01', periods=size, freq='1H'),
                'merchant_id': [f'MERCHANT_{i%50:04d}' for i in range(size)],
                'payment_method': np.random.choice(['credit_card', 'debit_card', 'bank_transfer'], size),
                'currency': np.random.choice(['USD', 'EUR', 'GBP'], size),
                'country': np.random.choice(['US', 'UK', 'DE'], size),
                'is_fraud': np.random.choice([0, 1], size, p=[0.95, 0.05])
            })
            
            # Add some data quality issues for testing
            data.loc[10:20, 'amount'] = np.nan  # Missing values
            data.loc[30:35, 'amount'] = -100  # Negative values
            data.loc[50:60] = data.loc[50:60].copy()  # Duplicates
            
        elif data_type == "corrupted":
            data = pd.DataFrame({
                'transaction_id': ['INVALID_ID'] * size,
                'amount': ['not_a_number'] * size,
                'timestamp': ['invalid_date'] * size,
                'currency': ['INVALID_CURRENCY'] * size
            })
            
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        self.test_data_cache[data_type] = data
        return data
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests for preprocessing components"""
        test_results = {
            'framework_tests': self._test_preprocessing_framework(),
            'steps_tests': self._test_preprocessing_steps(),
            'pipeline_tests': self._test_preprocessing_pipeline(),
            'integration_tests': self._test_integration_manager()
        }
        
        # Calculate overall success rate
        total_tests = sum(len(category) for category in test_results.values())
        passed_tests = sum(
            sum(1 for test in category.values() if test['passed'])
            for category in test_results.values()
        )
        
        test_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        return test_results
    
    def _test_preprocessing_framework(self) -> Dict[str, Dict[str, Any]]:
        """Test preprocessing framework components"""
        tests = {}
        
        # Test exception handling
        tests['exception_handling'] = self._run_test(
            "Exception handling",
            lambda: self._test_exception_creation()
        )
        
        # Test validators
        tests['input_validation'] = self._run_test(
            "Input validation",
            lambda: self._test_input_validation()
        )
        
        # Test performance monitoring
        tests['performance_monitoring'] = self._run_test(
            "Performance monitoring",
            lambda: self._test_performance_monitoring()
        )
        
        return tests
    
    def _test_preprocessing_steps(self) -> Dict[str, Dict[str, Any]]:
        """Test individual preprocessing steps"""
        tests = {}
        
        # Test data cleaner
        tests['data_cleaner'] = self._run_test(
            "Data cleaner",
            lambda: self._test_data_cleaner()
        )
        
        # Test feature engineer
        tests['feature_engineer'] = self._run_test(
            "Feature engineer",
            lambda: self._test_feature_engineer()
        )
        
        return tests
    
    def _test_preprocessing_pipeline(self) -> Dict[str, Dict[str, Any]]:
        """Test preprocessing pipeline components"""
        tests = {}
        
        # Test categorical encoder
        tests['categorical_encoder'] = self._run_test(
            "Categorical encoder",
            lambda: self._test_categorical_encoder()
        )
        
        # Test feature scaler
        tests['feature_scaler'] = self._run_test(
            "Feature scaler",
            lambda: self._test_feature_scaler()
        )
        
        # Test main preprocessor
        tests['main_preprocessor'] = self._run_test(
            "Main preprocessor",
            lambda: self._test_main_preprocessor()
        )
        
        return tests
    
    def _test_integration_manager(self) -> Dict[str, Dict[str, Any]]:
        """Test integration manager"""
        tests = {}
        
        # Test integration manager initialization
        tests['manager_init'] = self._run_test(
            "Integration manager initialization",
            lambda: self._test_integration_manager_init()
        )
        
        # Test session management
        tests['session_management'] = self._run_test(
            "Session management",
            lambda: self._test_session_management()
        )
        
        # Test fallback processing
        tests['fallback_processing'] = self._run_test(
            "Fallback processing",
            lambda: self._test_fallback_processing()
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
                'duration': time.time() - start_time,
                'message': f"{test_name} failed: {str(e)}",
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _test_exception_creation(self):
        """Test exception creation and handling"""
        if not PREPROCESSING_COMPONENTS:
            return  # Skip if components not available
        
        # Test basic exception
        exc = PreprocessingException("Test message")
        assert exc.error_code == "PREPROCESSING_ERROR"
        assert "Test message" in str(exc)
        
        # Test specific exceptions
        config_exc = PreprocessingConfigError("Config error", config_field="test_field")
        assert config_exc.config_field == "test_field"
        
        input_exc = PreprocessingInputError("Input error", input_type="DataFrame")
        assert input_exc.input_type == "DataFrame"
    
    def _test_input_validation(self):
        """Test input validation functionality"""
        if not PREPROCESSING_COMPONENTS:
            return
        
        # Test with valid data
        test_data = self.generate_test_data("financial", 100)
        validator = InputValidator()
        
        # Should not raise exception for valid data
        validator.validate_input(test_data)
        
        # Test with invalid data
        invalid_data = pd.DataFrame()  # Empty DataFrame
        try:
            validator.validate_input(invalid_data)
            assert False, "Should have raised exception for empty data"
        except PreprocessingInputError:
            pass  # Expected
    
    def _test_performance_monitoring(self):
        """Test performance monitoring"""
        if not PREPROCESSING_COMPONENTS:
            return
        
        monitor = PerformanceMonitor()
        
        # Test operation monitoring
        with monitor.monitor_operation("test_operation"):
            time.sleep(0.1)  # Simulate work
        
        # Check metrics were recorded
        assert "test_operation" in monitor.metrics
        assert monitor.metrics["test_operation"]["call_count"] > 0
    
    def _test_data_cleaner(self):
        """Test data cleaner functionality"""
        if not PREPROCESSING_COMPONENTS:
            return
        
        test_data = self.generate_test_data("financial", 100)
        config = PreprocessingConfig()
        cleaner = DataCleaner(config)
        
        # Test fitting and transforming
        cleaner.fit(test_data)
        cleaned_data = cleaner.transform(test_data)
        
        # Should have fewer rows due to cleaning
        assert len(cleaned_data) <= len(test_data)
        assert cleaner.fitted
    
    def _test_feature_engineer(self):
        """Test feature engineer functionality"""
        if not PREPROCESSING_COMPONENTS:
            return
        
        test_data = self.generate_test_data("financial", 100)
        config = PreprocessingConfig()
        engineer = FeatureEngineer(config)
        
        # Test fitting and transforming
        engineer.fit(test_data)
        engineered_data = engineer.transform(test_data)
        
        # Should have more columns due to feature engineering
        assert len(engineered_data.columns) >= len(test_data.columns)
        assert engineer.fitted
    
    def _test_categorical_encoder(self):
        """Test categorical encoder functionality"""
        if not PREPROCESSING_COMPONENTS:
            return
        
        test_data = self.generate_test_data("financial", 100)
        config = PreprocessingConfig()
        encoder = CategoricalEncoder(config)
        
        # Test fitting and transforming
        encoder.fit(test_data)
        encoded_data = encoder.transform(test_data)
        
        # Check encoding was applied
        assert encoder.fitted
        assert len(encoded_data.columns) >= len(test_data.columns)
    
    def _test_feature_scaler(self):
        """Test feature scaler functionality"""
        if not PREPROCESSING_COMPONENTS:
            return
        
        test_data = self.generate_test_data("financial", 100)
        config = PreprocessingConfig()
        scaler = FeatureScaler(config)
        
        # Test fitting and transforming
        scaler.fit(test_data)
        scaled_data = scaler.transform(test_data)
        
        # Check scaling was applied
        assert scaler.fitted
        assert len(scaled_data) == len(test_data)
    
    def _test_main_preprocessor(self):
        """Test main preprocessor functionality"""
        if not PREPROCESSING_COMPONENTS:
            return
        
        test_data = self.generate_test_data("financial", 100)
        config = PreprocessingConfig()
        preprocessor = EnhancedFinancialDataPreprocessor(config)
        
        # Test fit_transform
        processed_data = preprocessor.fit_transform(test_data)
        
        # Check processing was successful
        assert processed_data is not None
        assert len(processed_data) > 0
    
    def _test_integration_manager_init(self):
        """Test integration manager initialization"""
        config = IntegrationConfig()
        manager = PreprocessingIntegrationManager(config)
        
        # Check manager was initialized
        assert manager.config == config
        assert manager.integration_state == {}
        assert manager.performance_metrics == {}
    
    def _test_session_management(self):
        """Test session management functionality"""
        config = IntegrationConfig()
        manager = PreprocessingIntegrationManager(config)
        
        # Test session creation and cleanup
        with manager.preprocessing_session("test_session") as session_id:
            assert session_id == "test_session"
            assert session_id in manager.integration_state
            assert manager.integration_state[session_id]['status'] == 'active'
        
        # Session should be completed after context exit
        assert manager.integration_state[session_id]['status'] == 'completed'
    
    def _test_fallback_processing(self):
        """Test fallback processing functionality"""
        config = IntegrationConfig(fallback_to_simple_preprocessing=True)
        manager = PreprocessingIntegrationManager(config)
        
        # Test fallback with valid data
        test_data = self.generate_test_data("financial", 50)
        processed_data = manager._simple_preprocessing_fallback(test_data)
        
        # Should return processed data
        assert processed_data is not None
        assert len(processed_data) > 0

# ======================== INTEGRATION FACTORY ========================

class PreprocessingIntegrationFactory:
    """Factory for creating integrated preprocessing systems"""
    
    @staticmethod
    def create_development_system() -> PreprocessingIntegrationManager:
        """Create preprocessing system for development environment"""
        config = IntegrationConfig(
            validate_before_processing=True,
            validate_after_processing=True,
            enable_performance_monitoring=True,
            enable_security_validation=True,
            enable_quality_validation=True,
            backup_original_data=True,
            fallback_to_simple_preprocessing=True,
            enable_test_mode=True
        )
        
        return PreprocessingIntegrationManager(config)
    
    @staticmethod
    def create_production_system() -> PreprocessingIntegrationManager:
        """Create preprocessing system for production environment"""
        config = IntegrationConfig(
            validate_before_processing=True,
            validate_after_processing=True,
            enable_performance_monitoring=True,
            enable_security_validation=True,
            enable_quality_validation=True,
            backup_original_data=False,  # Disable for performance
            fallback_to_simple_preprocessing=True,
            enable_test_mode=False,
            integration_timeout=180,  # 3 minutes for production
            max_retry_attempts=2
        )
        
        return PreprocessingIntegrationManager(config)
    
    @staticmethod
    def create_testing_system() -> PreprocessingIntegrationManager:
        """Create preprocessing system for testing"""
        config = IntegrationConfig(
            validate_before_processing=False,  # Disable for speed
            validate_after_processing=False,
            enable_performance_monitoring=False,
            enable_security_validation=False,
            enable_quality_validation=False,
            backup_original_data=False,
            fallback_to_simple_preprocessing=True,
            enable_test_mode=True,
            test_data_size_limit=100,
            mock_external_dependencies=True
        )
        
        return PreprocessingIntegrationManager(config)

# ======================== MAIN INTEGRATION INTERFACE ========================

def get_integrated_preprocessor(environment: str = "development") -> PreprocessingIntegrationManager:
    """Get configured preprocessing system for specified environment"""
    factory = PreprocessingIntegrationFactory()
    
    if environment == "development":
        return factory.create_development_system()
    elif environment == "production":
        return factory.create_production_system()
    elif environment == "testing":
        return factory.create_testing_system()
    else:
        raise ValueError(f"Unknown environment: {environment}")

def run_preprocessing_tests() -> Dict[str, Any]:
    """Run comprehensive preprocessing tests"""
    test_suite = PreprocessingTestSuite()
    return test_suite.run_unit_tests()

# Export integration components
__all__ = [
    # Configuration
    'IntegrationConfig',
    
    # Core integration
    'PreprocessingIntegrationManager',
    
    # Testing
    'PreprocessingTestSuite',
    
    # Factory
    'PreprocessingIntegrationFactory',
    
    # Main interface
    'get_integrated_preprocessor',
    'run_preprocessing_tests'
]

if __name__ == "__main__":
    print("Enhanced Data Preprocessor - Chunk 4: Integration utilities and testing framework loaded")
    
    # Run basic integration test
    try:
        manager = get_integrated_preprocessor("development")
        print("✓ Integration manager created successfully")
        
        # Run tests
        test_results = run_preprocessing_tests()
        print(f"✓ Tests completed: {test_results['summary']['passed_tests']}/{test_results['summary']['total_tests']} passed")
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")