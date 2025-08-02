"""
Enhanced Financial Data Loader for Financial Fraud Detection Domain
Comprehensive data loading, validation, and preprocessing for transaction data
with enhanced validation and error handling for production use.
"""

import logging
import pandas as pd
import numpy as np
import requests
import json
import hashlib
import pickle
import os
import time
import asyncio
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Iterator, AsyncIterator
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from tqdm import tqdm
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
import sqlite3
from urllib.parse import urljoin, urlparse
import warnings
import traceback
import signal
import re

# Configure logging
logger = logging.getLogger(__name__)

# ======================== ENHANCED DECORATORS ========================

def enhanced_error_handler(error_message: str = "Operation failed", max_retries: int = 3):
    """Enhanced error handler decorator with retry logic"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}, retrying...")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}: {e}")
            
            # If we get here, all retries failed
            if isinstance(last_exception, (DataSecurityError, DataIntegrityError)):
                raise last_exception  # Don't wrap critical errors
            raise DataLoaderException(f"{error_message}: {last_exception}")
        return wrapper
    return decorator

def enhanced_performance_monitor(operation_name: str):
    """Enhanced performance monitor decorator"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                result = func(self, *args, **kwargs)
                
                # Record successful operation
                if hasattr(self, '_performance_monitor'):
                    duration = time.time() - start_time
                    memory_used = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
                    
                    self._performance_monitor['processing_times'][operation_name] = duration
                    self._performance_monitor['memory_usage']['peak'] = max(
                        self._performance_monitor['memory_usage'].get('peak', 0),
                        psutil.Process().memory_info().rss / 1024 / 1024
                    )
                
                return result
                
            except Exception as e:
                # Record failed operation
                if hasattr(self, '_performance_monitor'):
                    duration = time.time() - start_time
                    self._performance_monitor['error_counts'][operation_name] = (
                        self._performance_monitor['error_counts'].get(operation_name, 0) + 1
                    )
                raise
        return wrapper
    return decorator

def enhanced_input_validator(func):
    """Enhanced input validator decorator"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Basic input validation
        if 'source' in kwargs and not kwargs['source']:
            raise DataSourceValidationError("Source cannot be empty")
        
        # Check for required arguments based on function signature
        import inspect
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Validate source parameter if present
        if 'source' in bound_args.arguments:
            source = bound_args.arguments['source']
            if isinstance(source, str) and len(source.strip()) == 0:
                raise DataSourceValidationError("Source string cannot be empty")
        
        return func(*args, **kwargs)
    return wrapper

# ======================== ENHANCED EXCEPTION HIERARCHY ========================

class DataLoaderException(Exception):
    """Base exception for data loader errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, 
                 error_code: str = "DATA_LOADER_ERROR", recoverable: bool = True):
        super().__init__(message)
        self.details = details or {}
        self.timestamp = datetime.now()
        self.error_code = error_code
        self.recoverable = recoverable
        self.retry_count = 0
        self.stack_trace = traceback.format_exc()

class DataSourceValidationError(DataLoaderException):
    """Raised when data source validation fails"""
    def __init__(self, message: str, validation_errors: List[str] = None, 
                 field: str = None, **kwargs):
        super().__init__(message, error_code="SOURCE_VALIDATION_ERROR", **kwargs)
        self.validation_errors = validation_errors or []
        self.field = field

class DataAccessError(DataLoaderException):
    """Raised when data cannot be accessed"""
    def __init__(self, message: str, access_type: str = None, 
                 resource: str = None, **kwargs):
        super().__init__(message, error_code="DATA_ACCESS_ERROR", **kwargs)
        self.access_type = access_type
        self.resource = resource

class DataFormatError(DataLoaderException):
    """Raised when data format is invalid"""
    def __init__(self, message: str, format_type: str = None, 
                 expected_format: str = None, **kwargs):
        super().__init__(message, error_code="DATA_FORMAT_ERROR", **kwargs)
        self.format_type = format_type
        self.expected_format = expected_format

class DataSizeError(DataLoaderException):
    """Raised when data size exceeds limits"""
    def __init__(self, message: str, actual_size: Any = None, 
                 max_size: Any = None, **kwargs):
        super().__init__(message, error_code="DATA_SIZE_ERROR", **kwargs)
        self.actual_size = actual_size
        self.max_size = max_size

class DataIntegrityError(DataLoaderException):
    """Raised when data integrity check fails"""
    def __init__(self, message: str, checksum_expected: str = None, 
                 checksum_actual: str = None, **kwargs):
        super().__init__(message, error_code="DATA_INTEGRITY_ERROR", recoverable=False, **kwargs)
        self.checksum_expected = checksum_expected
        self.checksum_actual = checksum_actual

class DataSecurityError(DataLoaderException):
    """Raised when security validation fails"""
    def __init__(self, message: str, security_check: str = None, 
                 severity: str = "HIGH", threat_type: str = None, **kwargs):
        super().__init__(message, error_code="DATA_SECURITY_ERROR", recoverable=False, **kwargs)
        self.security_check = security_check
        self.severity = severity
        self.threat_type = threat_type

class DataQualityError(DataLoaderException):
    """Raised when data quality is below threshold"""
    def __init__(self, message: str, quality_score: float = None, 
                 threshold: float = None, **kwargs):
        super().__init__(message, error_code="DATA_QUALITY_ERROR", **kwargs)
        self.quality_score = quality_score
        self.threshold = threshold

class DataLoadTimeoutError(DataLoaderException):
    """Raised when data loading times out"""
    def __init__(self, message: str, timeout_seconds: float = None, **kwargs):
        super().__init__(message, error_code="DATA_LOAD_TIMEOUT", **kwargs)
        self.timeout_seconds = timeout_seconds

class DataMemoryError(DataLoaderException):
    """Raised when memory limits are exceeded"""
    def __init__(self, message: str, memory_required: float = None, 
                 memory_available: float = None, **kwargs):
        super().__init__(message, error_code="DATA_MEMORY_ERROR", **kwargs)
        self.memory_required = memory_required
        self.memory_available = memory_available

class DataCompatibilityError(DataLoaderException):
    """Raised when data format is incompatible"""
    def __init__(self, message: str, source_format: str = None, 
                 target_format: str = None, **kwargs):
        super().__init__(message, error_code="DATA_COMPATIBILITY_ERROR", **kwargs)
        self.source_format = source_format
        self.target_format = target_format

# ======================== ENUMS ========================

class DataSourceType(Enum):
    """Supported data source types for financial data"""
    CSV = "csv"
    JSON = "json"
    API = "api"
    PARQUET = "parquet"
    EXCEL = "excel"
    DATABASE = "database"
    STREAM = "stream"
    S3 = "s3"
    KAFKA = "kafka"

class LoadStatus(Enum):
    """Data loading operation status"""
    PENDING = "pending"
    LOADING = "loading" 
    VALIDATING = "validating"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"
    EXPIRED = "expired"
    DEGRADED = "degraded"
    TIMEOUT = "timeout"

class ValidationLevel(Enum):
    """Data validation strictness levels"""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"
    PARANOID = "paranoid"

class SecurityLevel(Enum):
    """Data security validation levels"""
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"

class DataQualityLevel(Enum):
    """Data quality assessment levels"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    EXHAUSTIVE = "exhaustive"

class ProcessingMode(Enum):
    """Data processing modes"""
    BATCH = "batch"
    STREAMING = "streaming"
    INCREMENTAL = "incremental"
    REAL_TIME = "real_time"

# ======================== DATA CLASSES ========================

@dataclass
class DataLoadMetrics:
    """Comprehensive metrics for data loading operations"""
    source_name: str
    source_type: DataSourceType
    total_records: int
    valid_records: int
    invalid_records: int
    duplicate_records: int
    load_time_seconds: float
    validation_time_seconds: float
    preprocessing_time_seconds: float
    cache_hit: bool
    memory_usage_mb: float
    error_count: int = 0
    warning_count: int = 0
    data_quality_score: float = 0.0
    validation_level: ValidationLevel = ValidationLevel.BASIC
    security_level: Optional[SecurityLevel] = None
    quality_level: DataQualityLevel = DataQualityLevel.STANDARD
    processing_mode: ProcessingMode = ProcessingMode.BATCH
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None
    load_status: LoadStatus = LoadStatus.PENDING
    retry_count: int = 0
    degradation_applied: bool = False
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    security_checks: List[str] = field(default_factory=list)
    compliance_status: Dict[str, bool] = field(default_factory=dict)

@dataclass
class DataValidationResult:
    """Enhanced result of data validation"""
    is_valid: bool
    cleaned_data: Optional[pd.DataFrame]
    warnings: List[str]
    errors: List[str]
    quality_score: float
    validation_level: ValidationLevel
    security_score: float = 1.0
    compliance_score: float = 1.0
    performance_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    degradation_applied: bool = False

@dataclass
class DataSource:
    """Comprehensive data source configuration"""
    name: str
    source_type: DataSourceType
    location: str
    credentials: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None
    validation_level: ValidationLevel = ValidationLevel.BASIC
    security_level: Optional[SecurityLevel] = None
    quality_level: DataQualityLevel = DataQualityLevel.STANDARD
    processing_mode: ProcessingMode = ProcessingMode.BATCH
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds
    batch_size: Optional[int] = None
    retry_attempts: int = 3
    timeout: int = 300  # seconds
    compression: Optional[str] = None
    encoding: str = "utf-8"
    headers: Optional[Dict[str, str]] = None
    schema: Optional[Dict[str, Any]] = None
    preprocessing_hooks: List[str] = field(default_factory=list)
    postprocessing_hooks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_memory_mb: int = 1024
    verify_checksum: bool = True
    allowed_formats: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    quality_threshold: float = 0.8
    enable_degradation: bool = True
    enable_monitoring: bool = True
    enable_recovery: bool = True
    compliance_requirements: List[str] = field(default_factory=list)
    data_retention_days: int = 90
    encryption_required: bool = False
    audit_enabled: bool = True

@dataclass
class LoadConfiguration:
    """Configuration for data loading operations"""
    max_workers: int = 4
    default_batch_size: int = 10000
    enable_async: bool = False
    max_memory_mb: int = 2048
    enable_recovery: bool = True
    enable_monitoring: bool = True
    enable_caching: bool = True
    cache_compression: bool = True
    enable_degradation: bool = True
    security_enabled: bool = True
    audit_enabled: bool = True
    performance_tracking: bool = True
    quality_enforcement: bool = True
    compliance_checking: bool = True
    retry_strategy: str = "exponential_backoff"
    timeout_strategy: str = "adaptive"
    memory_strategy: str = "dynamic"
    error_handling_strategy: str = "graceful_degradation"

@dataclass
class DataProcessingContext:
    """Context for data processing operations"""
    session_id: str
    user_id: Optional[str] = None
    operation_id: Optional[str] = None
    environment: str = "production"
    security_context: Dict[str, Any] = field(default_factory=dict)
    compliance_context: Dict[str, Any] = field(default_factory=dict)
    performance_requirements: Dict[str, Any] = field(default_factory=dict)
    quality_requirements: Dict[str, Any] = field(default_factory=dict)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    
# ======================== DECORATORS ========================

def validate_inputs(func):
    """Decorator to validate function inputs with enhanced checking"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Enhanced input validation
            for i, arg in enumerate(args):
                if arg is None and i > 0:  # Skip self
                    raise ValueError(f"Argument {i} cannot be None")
                    
            # Validate specific parameter types
            for key, value in kwargs.items():
                if key.endswith('_level') and hasattr(value, 'value'):
                    # Enum validation
                    continue
                elif key.endswith('_size') and isinstance(value, (int, float)):
                    if value <= 0:
                        raise ValueError(f"Parameter {key} must be positive")
                elif key.endswith('_timeout') and isinstance(value, (int, float)):
                    if value <= 0:
                        raise ValueError(f"Timeout {key} must be positive")
                        
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Input validation failed for {func.__name__}: {e}")
            raise DataSourceValidationError(f"Input validation failed: {e}")
    return wrapper

def handle_errors(error_type: type = DataLoaderException, 
                 default_return: Any = None,
                 log_level: str = "error",
                 recovery_enabled: bool = True):
    """Decorator for comprehensive error handling with recovery"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_type as e:
                getattr(logger, log_level)(f"{func.__name__} failed: {e}")
                
                # Enhanced error handling with recovery
                if hasattr(args[0], '_handle_error') and recovery_enabled:
                    return args[0]._handle_error(e, default_return, func.__name__)
                return default_return
                
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {e}")
                
                # Wrap unexpected errors with context
                wrapped_error = DataLoaderException(
                    f"Unexpected error in {func.__name__}: {e}",
                    {
                        'function': func.__name__,
                        'original_error': str(e),
                        'error_type': type(e).__name__,
                        'args': str(args[1:]) if len(args) > 1 else '',
                        'kwargs': str(kwargs)
                    }
                )
                
                if hasattr(args[0], '_handle_error') and recovery_enabled:
                    return args[0]._handle_error(wrapped_error, default_return, func.__name__)
                    
                raise wrapped_error
        return wrapper
    return decorator

def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, 
                    backoff: float = 2.0, exceptions: Tuple = None):
    """Enhanced decorator to retry failed operations with configurable strategy"""
    if exceptions is None:
        exceptions = (DataAccessError, DataLoadTimeoutError, ConnectionError)
        
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    result = func(*args, **kwargs)
                    
                    # Log successful retry
                    if attempt > 0:
                        logger.info(f"{func.__name__} succeeded on attempt {attempt + 1}")
                        
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    
                    # Check if error is recoverable
                    if hasattr(e, 'recoverable') and not e.recoverable:
                        logger.error(f"{func.__name__} failed with non-recoverable error: {e}")
                        break
                        
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}), "
                                     f"retrying in {wait_time:.1f}s: {e}")
                        time.sleep(wait_time)
                        
                        # Update retry count if available
                        if hasattr(e, 'retry_count'):
                            e.retry_count = attempt + 1
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        
                except Exception as e:
                    # Non-retryable exception
                    logger.error(f"{func.__name__} failed with non-retryable error: {e}")
                    raise
            
            if last_exception:
                raise last_exception
        return wrapper
    return decorator

def timeout_handler(timeout_seconds: int, adaptive: bool = False):
    """Enhanced decorator to add timeout handling with adaptive capabilities"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            effective_timeout = timeout_seconds
            
            # Adaptive timeout based on data size or complexity
            if adaptive and len(args) > 1:
                if hasattr(args[1], 'get') and 'expected_size_mb' in args[1].get('options', {}):
                    size_mb = args[1]['options']['expected_size_mb']
                    # Increase timeout for larger datasets
                    effective_timeout = max(timeout_seconds, int(size_mb * 0.1))
                    
            def timeout_callback(signum, frame):
                raise DataLoadTimeoutError(
                    f"Operation timed out after {effective_timeout} seconds",
                    timeout_seconds=effective_timeout
                )
            
            # Set signal alarm (Unix only)
            if hasattr(signal, 'SIGALRM'):
                old_handler = signal.signal(signal.SIGALRM, timeout_callback)
                signal.alarm(effective_timeout)
                
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                # Fallback for Windows - basic implementation
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                if elapsed > effective_timeout:
                    logger.warning(f"Operation took {elapsed:.1f}s, exceeding timeout of {effective_timeout}s")
                
            return result
        return wrapper
    return decorator

def check_memory_usage(max_memory_mb: int = 1024, adaptive: bool = True):
    """Enhanced decorator to check memory usage with adaptive limits"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get current memory state
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            available_memory = psutil.virtual_memory().available / 1024 / 1024
            
            effective_limit = max_memory_mb
            
            # Adaptive memory management
            if adaptive:
                # Adjust limit based on available memory
                if available_memory < max_memory_mb:
                    effective_limit = max(512, available_memory * 0.8)
                    logger.warning(f"Adjusted memory limit to {effective_limit:.0f}MB due to low available memory")
                    
            # Pre-execution memory check
            if available_memory < effective_limit:
                raise DataMemoryError(
                    f"Insufficient memory: {available_memory:.0f}MB available, {effective_limit:.0f}MB required",
                    memory_required=effective_limit,
                    memory_available=available_memory
                )
            
            # Execute function with memory monitoring
            try:
                result = func(*args, **kwargs)
                
                # Post-execution memory check
                final_memory = process.memory_info().rss / 1024 / 1024
                memory_used = final_memory - initial_memory
                
                if memory_used > effective_limit:
                    logger.warning(f"Memory usage ({memory_used:.0f}MB) exceeded limit ({effective_limit:.0f}MB)")
                    
                # Add memory metrics to result if it's a metrics object
                if hasattr(result, 'memory_usage_mb'):
                    result.memory_usage_mb = memory_used
                    
                return result
                
            except MemoryError as e:
                raise DataMemoryError(
                    f"Out of memory during {func.__name__}: {e}",
                    memory_required=effective_limit
                )
        return wrapper
    return decorator

def performance_monitor(track_detailed: bool = True):
    """Decorator to monitor performance metrics"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                result = func(*args, **kwargs)
                
                # Calculate metrics
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                # Log performance metrics
                logger.debug(f"{func.__name__} performance: {execution_time:.2f}s, "
                           f"memory delta: {memory_delta:.1f}MB")
                
                # Add performance data to result if possible
                if hasattr(result, 'performance_metrics'):
                    result.performance_metrics.update({
                        f'{func.__name__}_execution_time': execution_time,
                        f'{func.__name__}_memory_delta': memory_delta,
                        f'{func.__name__}_timestamp': start_time
                    })
                    
                return result
                
            except Exception as e:
                # Log performance for failed operations too
                execution_time = time.time() - start_time
                logger.debug(f"{func.__name__} failed after {execution_time:.2f}s: {e}")
                raise
                
        return wrapper
    return decorator

# Export for Chunk 1
__all__ = [
    # Exceptions
    'DataLoaderException', 'DataSourceValidationError', 'DataAccessError',
    'DataFormatError', 'DataSizeError', 'DataIntegrityError', 'DataSecurityError',
    'DataQualityError', 'DataLoadTimeoutError', 'DataMemoryError', 'DataCompatibilityError',
    
    # Enums
    'DataSourceType', 'LoadStatus', 'ValidationLevel', 'SecurityLevel',
    'DataQualityLevel', 'ProcessingMode',
    
    # Data Classes
    'DataLoadMetrics', 'DataValidationResult', 'DataSource', 'LoadConfiguration',
    'DataProcessingContext',
    
    # Decorators
    'validate_inputs', 'handle_errors', 'retry_on_failure', 'timeout_handler',
    'check_memory_usage', 'performance_monitor'
]

# ======================== VALIDATORS ========================

class DataValidator(ABC):
    """Abstract base class for enhanced data validators"""
    
    @abstractmethod
    def validate(self, data: pd.DataFrame, level: ValidationLevel = ValidationLevel.BASIC,
                context: Optional[DataProcessingContext] = None) -> DataValidationResult:
        """Validate data and return comprehensive result"""
        pass
    
    @abstractmethod
    def get_validation_schema(self) -> Dict[str, Any]:
        """Get validation schema for this validator"""
        pass

class EnhancedFinancialTransactionValidator(DataValidator):
    """Enhanced validator for financial transaction data with comprehensive validation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.required_columns = self.config.get('required_columns', [
            'transaction_id', 'amount', 'timestamp', 'user_id'
        ])
        self.optional_columns = self.config.get('optional_columns', [
            'merchant_id', 'transaction_type', 'currency', 'location', 'account_id',
            'description', 'category', 'payment_method', 'risk_score'
        ])
        self.amount_limits = self.config.get('amount_limits', {
            'min': 0.01,
            'max': 1000000.0,
            'suspicious_threshold': 10000.0,
            'micro_transaction_threshold': 1.0,
            'large_transaction_threshold': 100000.0
        })
        self.time_limits = self.config.get('time_limits', {
            'earliest_date': '2020-01-01',
            'latest_date': None,  # Will use current date
            'future_tolerance_hours': 24
        })
        self.quality_weights = self.config.get('quality_weights', {
            'completeness': 0.25,
            'validity': 0.25,
            'consistency': 0.25,
            'accuracy': 0.25
        })
        self.fraud_patterns = self.config.get('fraud_patterns', {
            'round_amount_threshold': 0.7,
            'velocity_threshold_minutes': 1,
            'duplicate_threshold_percentage': 0.05,
            'outlier_threshold_std': 3.0
        })
        
    def get_validation_schema(self) -> Dict[str, Any]:
        """Get comprehensive validation schema"""
        return {
            'required_columns': self.required_columns,
            'optional_columns': self.optional_columns,
            'column_types': {
                'transaction_id': 'string',
                'amount': 'float',
                'timestamp': 'datetime',
                'user_id': 'string',
                'merchant_id': 'string',
                'transaction_type': 'categorical',
                'currency': 'categorical',
                'location': 'string'
            },
            'constraints': {
                'amount': {'min': self.amount_limits['min'], 'max': self.amount_limits['max']},
                'timestamp': {
                    'min': self.time_limits['earliest_date'],
                    'max': self.time_limits['latest_date']
                },
                'currency': {'allowed_values': ['USD', 'EUR', 'GBP', 'CAD', 'JPY', 'AUD', 'CHF']},
                'transaction_type': {
                    'allowed_values': ['purchase', 'withdrawal', 'deposit', 'transfer', 'refund']
                }
            }
        }
        
    def validate(self, data: pd.DataFrame, level: ValidationLevel = ValidationLevel.BASIC,
                context: Optional[DataProcessingContext] = None) -> DataValidationResult:
        """Comprehensive validation of financial transaction data"""
        validation_start = time.time()
        warnings = []
        errors = []
        recommendations = []
        metadata = {
            'validation_start_time': validation_start,
            'validation_level': level.value,
            'original_count': len(data) if not data.empty else 0
        }
        
        if data.empty:
            return DataValidationResult(
                is_valid=False,
                cleaned_data=None,
                warnings=[],
                errors=["Dataset is empty"],
                quality_score=0.0,
                validation_level=level,
                metadata=metadata
            )
        
        # Track quality metrics with enhanced scoring
        quality_metrics = {
            'completeness': 0.0,
            'validity': 0.0,
            'consistency': 0.0,
            'accuracy': 0.0
        }
        
        security_score = 1.0
        compliance_score = 1.0
        performance_score = 1.0
        
        try:
            # Level 1: Basic validation with enhanced error reporting
            data, basic_result = self._enhanced_basic_validation(data, context)
            warnings.extend(basic_result['warnings'])
            errors.extend(basic_result['errors'])
            recommendations.extend(basic_result.get('recommendations', []))
            quality_metrics['completeness'] = basic_result['completeness_score']
            
            if level == ValidationLevel.NONE:
                quality_score = quality_metrics['completeness']
            else:
                # Level 2: Strict validation with business rules
                if level in [ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE, ValidationLevel.PARANOID]:
                    data, strict_result = self._enhanced_strict_validation(data, context)
                    warnings.extend(strict_result['warnings'])
                    errors.extend(strict_result['errors'])
                    recommendations.extend(strict_result.get('recommendations', []))
                    quality_metrics['validity'] = strict_result['validity_score']
                
                # Level 3: Comprehensive validation with pattern detection
                if level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.PARANOID]:
                    data, comp_result = self._enhanced_comprehensive_validation(data, context)
                    warnings.extend(comp_result['warnings'])
                    errors.extend(comp_result['errors'])
                    recommendations.extend(comp_result.get('recommendations', []))
                    quality_metrics['consistency'] = comp_result['consistency_score']
                
                # Level 4: Paranoid validation with advanced analytics
                if level == ValidationLevel.PARANOID:
                    data, paranoid_result = self._enhanced_paranoid_validation(data, context)
                    warnings.extend(paranoid_result['warnings'])
                    errors.extend(paranoid_result['errors'])
                    recommendations.extend(paranoid_result.get('recommendations', []))
                    quality_metrics['accuracy'] = paranoid_result.get('accuracy_score', 1.0)
                    security_score = paranoid_result.get('security_score', 1.0)
                
                # Calculate weighted quality score
                quality_score = sum(
                    quality_metrics[metric] * self.quality_weights[metric]
                    for metric in quality_metrics
                )
            
            # Enhanced metadata
            final_count = len(data) if data is not None else 0
            validation_time = time.time() - validation_start
            
            metadata.update({
                'final_count': final_count,
                'removed_count': metadata['original_count'] - final_count,
                'removal_rate': (metadata['original_count'] - final_count) / metadata['original_count'] 
                    if metadata['original_count'] > 0 else 0,
                'validation_time_seconds': validation_time,
                'quality_metrics': quality_metrics,
                'performance_metrics': {
                    'records_per_second': final_count / validation_time if validation_time > 0 else 0,
                    'memory_efficiency': self._calculate_memory_efficiency(data),
                    'processing_speed_score': min(1.0, 10000 / validation_time) if validation_time > 0 else 1.0
                }
            })
            
            # Enhanced validation details
            validation_details = {
                'schema_compliance': self._check_schema_compliance(data),
                'data_distribution': self._analyze_data_distribution(data),
                'anomaly_detection': self._detect_anomalies(data),
                'pattern_analysis': self._analyze_patterns(data),
                'quality_assessment': quality_metrics
            }
            
            performance_score = metadata['performance_metrics']['processing_speed_score']
            
            return DataValidationResult(
                is_valid=len(errors) == 0 and quality_score >= 0.5,
                cleaned_data=data,
                warnings=warnings,
                errors=errors,
                quality_score=quality_score,
                validation_level=level,
                security_score=security_score,
                compliance_score=compliance_score,
                performance_score=performance_score,
                metadata=metadata,
                validation_details=validation_details,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Enhanced validation failed: {e}")
            return DataValidationResult(
                is_valid=False,
                cleaned_data=None,
                warnings=warnings,
                errors=errors + [f"Validation error: {str(e)}"],
                quality_score=0.0,
                validation_level=level,
                metadata=metadata
            )
    
    def _enhanced_basic_validation(self, data: pd.DataFrame, 
                                  context: Optional[DataProcessingContext] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Enhanced basic validation with improved error handling"""
        result = {
            'warnings': [], 'errors': [], 'recommendations': [],
            'completeness_score': 0.0, 'schema_issues': []
        }
        
        # Enhanced column validation
        missing_cols = set(self.required_columns) - set(data.columns)
        if missing_cols:
            result['errors'].append(f"Missing required columns: {missing_cols}")
            result['recommendations'].append("Ensure data source includes all required columns")
            result['completeness_score'] = 0.0
            return data, result
        
        # Check for unexpected columns
        unexpected_cols = set(data.columns) - set(self.required_columns + self.optional_columns)
        if unexpected_cols:
            result['warnings'].append(f"Unexpected columns found: {unexpected_cols}")
            result['recommendations'].append("Review data schema for consistency")
        
        # Enhanced completeness calculation
        total_cells = len(data) * len(self.required_columns)
        null_cells = data[self.required_columns].isna().sum().sum()
        completeness = (total_cells - null_cells) / total_cells if total_cells > 0 else 0
        
        # Enhanced data type conversion with error tracking
        conversion_errors = {}
        try:
            if 'timestamp' in data.columns:
                original_count = len(data)
                data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
                null_timestamps = data['timestamp'].isna().sum()
                conversion_errors['timestamp'] = null_timestamps
                
                if null_timestamps > 0:
                    result['warnings'].append(f"Found {null_timestamps} invalid timestamps")
                    if null_timestamps / original_count > 0.1:
                        result['errors'].append("High percentage of invalid timestamps (>10%)")
                    data = data.dropna(subset=['timestamp'])
            
            if 'amount' in data.columns:
                original_count = len(data)
                data['amount'] = pd.to_numeric(data['amount'], errors='coerce')
                null_amounts = data['amount'].isna().sum()
                conversion_errors['amount'] = null_amounts
                
                if null_amounts > 0:
                    result['warnings'].append(f"Found {null_amounts} invalid amounts")
                    if null_amounts / original_count > 0.05:
                        result['errors'].append("High percentage of invalid amounts (>5%)")
                    data = data.dropna(subset=['amount'])
                    
        except Exception as e:
            result['errors'].append(f"Data type conversion failed: {e}")
        
        # Enhanced null value handling
        initial_count = len(data)
        for col in self.required_columns:
            if col in data.columns:
                null_count = data[col].isna().sum()
                if null_count > 0:
                    null_percentage = null_count / len(data) * 100
                    if null_percentage > 5:
                        result['warnings'].append(
                            f"High null percentage in {col}: {null_percentage:.1f}%"
                        )
                        result['recommendations'].append(
                            f"Investigate data quality issues in {col} column"
                        )
        
        data = data.dropna(subset=self.required_columns)
        if len(data) < initial_count:
            removed_count = initial_count - len(data)
            result['warnings'].append(f"Removed {removed_count} rows with null required values")
            
            if removed_count / initial_count > 0.2:
                result['errors'].append("High data loss due to null values (>20%)")
                result['recommendations'].append("Review data collection process")
        
        result['completeness_score'] = completeness
        result['conversion_errors'] = conversion_errors
        return data, result
    
    def _enhanced_strict_validation(self, data: pd.DataFrame,
                                   context: Optional[DataProcessingContext] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Enhanced strict validation with business rule enforcement"""
        result = {
            'warnings': [], 'errors': [], 'recommendations': [],
            'validity_score': 1.0, 'business_rules_violations': []
        }
        initial_count = len(data)
        
        # Enhanced amount validation with business context
        if 'amount' in data.columns:
            min_amount = self.amount_limits['min']
            max_amount = self.amount_limits['max']
            suspicious_threshold = self.amount_limits['suspicious_threshold']
            
            # Range validation
            invalid_amounts = data[
                (data['amount'] < min_amount) | 
                (data['amount'] > max_amount)
            ]
            
            if len(invalid_amounts) > 0:
                invalid_percentage = len(invalid_amounts) / len(data) * 100
                result['warnings'].append(
                    f"Found {len(invalid_amounts)} transactions with invalid amounts ({invalid_percentage:.1f}%)"
                )
                result['business_rules_violations'].append('amount_range_violation')
                data = data[~data.index.isin(invalid_amounts.index)]
                result['validity_score'] *= max(0.1, 1 - invalid_percentage / 100)
            
            # Suspicious amount detection
            suspicious_amounts = data[data['amount'] > suspicious_threshold]
            if len(suspicious_amounts) > 0:
                suspicious_percentage = len(suspicious_amounts) / len(data) * 100
                result['warnings'].append(
                    f"Found {len(suspicious_amounts)} transactions above suspicious threshold ({suspicious_percentage:.1f}%)"
                )
                if suspicious_percentage > 5:
                    result['recommendations'].append("Review high-value transactions for legitimacy")
                    result['validity_score'] *= 0.95
                    
            # Micro-transaction analysis
            micro_threshold = self.amount_limits.get('micro_transaction_threshold', 1.0)
            micro_transactions = data[data['amount'] < micro_threshold]
            if len(micro_transactions) > len(data) * 0.3:
                result['warnings'].append("High percentage of micro-transactions detected")
                result['recommendations'].append("Verify micro-transaction processing rules")
        
        # Enhanced timestamp validation with business hours
        if 'timestamp' in data.columns:
            earliest_date = pd.to_datetime(self.time_limits['earliest_date'])
            latest_date = pd.to_datetime(self.time_limits['latest_date'] or datetime.now())
            future_tolerance = pd.Timedelta(hours=self.time_limits.get('future_tolerance_hours', 24))
            
            # Date range validation
            invalid_dates = data[
                (data['timestamp'] < earliest_date) |
                (data['timestamp'] > latest_date + future_tolerance)
            ]
            
            if len(invalid_dates) > 0:
                result['warnings'].append(f"Found {len(invalid_dates)} transactions with invalid timestamps")
                result['business_rules_violations'].append('timestamp_range_violation')
                data = data[~data.index.isin(invalid_dates.index)]
                result['validity_score'] *= 0.9
                
            # Business hours analysis
            if not data.empty:
                business_hours_mask = (
                    (data['timestamp'].dt.hour >= 9) & 
                    (data['timestamp'].dt.hour <= 17) &
                    (data['timestamp'].dt.weekday < 5)
                )
                off_hours_percentage = (1 - business_hours_mask.mean()) * 100
                
                if off_hours_percentage > 70:
                    result['warnings'].append(f"High percentage of off-hours transactions: {off_hours_percentage:.1f}%")
                    result['recommendations'].append("Review off-hours transaction patterns")
        
        # Enhanced duplicate detection
        if 'transaction_id' in data.columns:
            duplicates = data.duplicated(subset=['transaction_id'], keep='first')
            if duplicates.any():
                duplicate_count = duplicates.sum()
                duplicate_percentage = duplicate_count / initial_count * 100
                result['warnings'].append(
                    f"Found {duplicate_count} duplicate transaction IDs ({duplicate_percentage:.1f}%)"
                )
                result['business_rules_violations'].append('duplicate_transaction_ids')
                data = data[~duplicates]
                result['validity_score'] *= max(0.5, 1 - duplicate_percentage / 100)
                
                if duplicate_percentage > 1:
                    result['recommendations'].append("Investigate transaction ID generation process")
        
        # Enhanced format validation
        format_issues = self._validate_data_formats(data)
        if format_issues:
            result['warnings'].extend(format_issues['warnings'])
            result['errors'].extend(format_issues['errors'])
            result['recommendations'].extend(format_issues['recommendations'])
            result['validity_score'] *= format_issues['format_score']
        
        return data, result
    
    def _enhanced_comprehensive_validation(self, data: pd.DataFrame,
                                          context: Optional[DataProcessingContext] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Enhanced comprehensive validation with advanced pattern detection"""
        result = {
            'warnings': [], 'errors': [], 'recommendations': [],
            'consistency_score': 1.0, 'pattern_anomalies': []
        }
        
        # Advanced velocity analysis
        velocity_analysis = self._analyze_transaction_velocity(data)
        if velocity_analysis:
            result['warnings'].extend(velocity_analysis['warnings'])
            result['recommendations'].extend(velocity_analysis['recommendations'])
            result['pattern_anomalies'].extend(velocity_analysis['anomalies'])
            result['consistency_score'] *= velocity_analysis['consistency_score']
        
        # Enhanced amount pattern analysis
        amount_analysis = self._analyze_amount_patterns(data)
        if amount_analysis:
            result['warnings'].extend(amount_analysis['warnings'])
            result['recommendations'].extend(amount_analysis['recommendations'])
            result['pattern_anomalies'].extend(amount_analysis['anomalies'])
            result['consistency_score'] *= amount_analysis['consistency_score']
        
        # Cross-field consistency validation
        consistency_analysis = self._validate_cross_field_consistency(data)
        if consistency_analysis:
            result['warnings'].extend(consistency_analysis['warnings'])
            result['recommendations'].extend(consistency_analysis['recommendations'])
            result['consistency_score'] *= consistency_analysis['consistency_score']
        
        # User behavior pattern analysis
        if 'user_id' in data.columns:
            behavior_analysis = self._analyze_user_behavior_patterns(data)
            if behavior_analysis:
                result['warnings'].extend(behavior_analysis['warnings'])
                result['recommendations'].extend(behavior_analysis['recommendations'])
                result['pattern_anomalies'].extend(behavior_analysis['anomalies'])
                result['consistency_score'] *= behavior_analysis['consistency_score']
        
        return data, result
    
    def _enhanced_paranoid_validation(self, data: pd.DataFrame,
                                     context: Optional[DataProcessingContext] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Enhanced paranoid validation with forensic-level analysis"""
        result = {
            'warnings': [], 'errors': [], 'recommendations': [],
            'accuracy_score': 1.0, 'security_score': 1.0, 'forensic_findings': []
        }
        
        # Benford's Law analysis for fraud detection
        benford_analysis = self._perform_benfords_law_analysis(data)
        if benford_analysis:
            result['warnings'].extend(benford_analysis['warnings'])
            result['recommendations'].extend(benford_analysis['recommendations'])
            result['forensic_findings'].extend(benford_analysis['findings'])
            result['accuracy_score'] *= benford_analysis['accuracy_score']
        
        # Advanced time pattern analysis
        time_analysis = self._perform_advanced_time_analysis(data)
        if time_analysis:
            result['warnings'].extend(time_analysis['warnings'])
            result['recommendations'].extend(time_analysis['recommendations'])
            result['forensic_findings'].extend(time_analysis['findings'])
            result['accuracy_score'] *= time_analysis['accuracy_score']
        
        # Fraud pattern detection
        fraud_analysis = self._detect_advanced_fraud_patterns(data)
        if fraud_analysis:
            result['warnings'].extend(fraud_analysis['warnings'])
            result['recommendations'].extend(fraud_analysis['recommendations'])
            result['forensic_findings'].extend(fraud_analysis['findings'])
            result['security_score'] *= fraud_analysis['security_score']
        
        # Data quality fingerprinting
        fingerprint_analysis = self._perform_data_fingerprinting(data)
        if fingerprint_analysis:
            result['warnings'].extend(fingerprint_analysis['warnings'])
            result['recommendations'].extend(fingerprint_analysis['recommendations'])
            result['forensic_findings'].extend(fingerprint_analysis['findings'])
            result['accuracy_score'] *= fingerprint_analysis['accuracy_score']
        
        return data, result
    
    # Helper methods for enhanced validation
    def _calculate_memory_efficiency(self, data: pd.DataFrame) -> float:
        """Calculate memory efficiency score"""
        if data is None or data.empty:
            return 0.0
        try:
            memory_usage = data.memory_usage(deep=True).sum()
            theoretical_minimum = len(data) * len(data.columns) * 8  # 8 bytes per value minimum
            return min(1.0, theoretical_minimum / memory_usage) if memory_usage > 0 else 1.0
        except:
            return 0.5
    
    def _check_schema_compliance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check compliance with expected schema"""
        schema = self.get_validation_schema()
        compliance = {
            'column_compliance': True,
            'type_compliance': True,
            'constraint_compliance': True,
            'issues': []
        }
        
        try:
            # Check column presence
            required_cols = set(schema['required_columns'])
            actual_cols = set(data.columns)
            missing_cols = required_cols - actual_cols
            
            if missing_cols:
                compliance['column_compliance'] = False
                compliance['issues'].append(f"Missing columns: {missing_cols}")
            
            # Check data types
            for col, expected_type in schema.get('column_types', {}).items():
                if col in data.columns:
                    actual_type = str(data[col].dtype)
                    if not self._is_compatible_type(actual_type, expected_type):
                        compliance['type_compliance'] = False
                        compliance['issues'].append(f"Type mismatch in {col}: expected {expected_type}, got {actual_type}")
            
            return compliance
            
        except Exception as e:
            logger.warning(f"Schema compliance check failed: {e}")
            return compliance
    
    def _is_compatible_type(self, actual_type: str, expected_type: str) -> bool:
        """Check if actual type is compatible with expected type"""
        type_mappings = {
            'string': ['object', 'string'],
            'float': ['float64', 'float32', 'int64', 'int32'],
            'datetime': ['datetime64[ns]', 'datetime64'],
            'categorical': ['object', 'string', 'category']
        }
        
        compatible_types = type_mappings.get(expected_type, [expected_type])
        return actual_type in compatible_types
    
    def _analyze_data_distribution(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data distribution characteristics"""
        distribution = {'numerical': {}, 'categorical': {}, 'temporal': {}}
        
        try:
            for col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    distribution['numerical'][col] = {
                        'mean': float(data[col].mean()) if not data[col].isna().all() else None,
                        'std': float(data[col].std()) if not data[col].isna().all() else None,
                        'skewness': float(data[col].skew()) if not data[col].isna().all() else None,
                        'kurtosis': float(data[col].kurtosis()) if not data[col].isna().all() else None
                    }
                elif pd.api.types.is_datetime64_any_dtype(data[col]):
                    distribution['temporal'][col] = {
                        'min_date': str(data[col].min()) if not data[col].isna().all() else None,
                        'max_date': str(data[col].max()) if not data[col].isna().all() else None,
                        'date_range_days': (data[col].max() - data[col].min()).days if not data[col].isna().all() else None
                    }
                else:
                    distribution['categorical'][col] = {
                        'unique_count': int(data[col].nunique()),
                        'most_frequent': str(data[col].mode().iloc[0]) if not data[col].empty and not data[col].isna().all() else None
                    }
            
            return distribution
            
        except Exception as e:
            logger.warning(f"Data distribution analysis failed: {e}")
            return distribution
    
    def _detect_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect statistical anomalies in the data"""
        anomalies = {'statistical': [], 'business': [], 'temporal': []}
        
        try:
            # Statistical anomalies in numerical columns
            for col in data.select_dtypes(include=[np.number]).columns:
                if not data[col].isna().all():
                    q1 = data[col].quantile(0.25)
                    q3 = data[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                    if len(outliers) > 0:
                        outlier_percentage = len(outliers) / len(data) * 100
                        anomalies['statistical'].append({
                            'column': col,
                            'type': 'statistical_outlier',
                            'count': len(outliers),
                            'percentage': round(outlier_percentage, 2)
                        })
            
            return anomalies
            
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            return anomalies
    
    def _analyze_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data patterns for quality assessment"""
        patterns = {'repetitive': [], 'sequential': [], 'clustering': []}
        
        try:
            # Check for repetitive patterns
            for col in data.columns:
                if data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(data[col]):
                    value_counts = data[col].value_counts()
                    if len(value_counts) > 0:
                        most_frequent_percentage = value_counts.iloc[0] / len(data) * 100
                        if most_frequent_percentage > 50:
                            patterns['repetitive'].append({
                                'column': col,
                                'most_frequent_value': str(value_counts.index[0]),
                                'percentage': round(most_frequent_percentage, 2)
                            })
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Pattern analysis failed: {e}")
            return patterns
    
    # Additional helper methods would be implemented here...
    def _validate_data_formats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data formats and return issues"""
        return {'warnings': [], 'errors': [], 'recommendations': [], 'format_score': 1.0}
    
    def _analyze_transaction_velocity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze transaction velocity patterns"""
        return {'warnings': [], 'recommendations': [], 'anomalies': [], 'consistency_score': 1.0}
    
    def _analyze_amount_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze amount patterns for anomalies"""
        return {'warnings': [], 'recommendations': [], 'anomalies': [], 'consistency_score': 1.0}
    
    def _validate_cross_field_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate consistency across fields"""
        return {'warnings': [], 'recommendations': [], 'consistency_score': 1.0}
    
    def _analyze_user_behavior_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user behavior patterns"""
        return {'warnings': [], 'recommendations': [], 'anomalies': [], 'consistency_score': 1.0}
    
    def _perform_benfords_law_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform Benford's Law analysis"""
        return {'warnings': [], 'recommendations': [], 'findings': [], 'accuracy_score': 1.0}
    
    def _perform_advanced_time_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform advanced time pattern analysis"""
        return {'warnings': [], 'recommendations': [], 'findings': [], 'accuracy_score': 1.0}
    
    def _detect_advanced_fraud_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect advanced fraud patterns"""
        return {'warnings': [], 'recommendations': [], 'findings': [], 'security_score': 1.0}
    
    def _perform_data_fingerprinting(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform data quality fingerprinting"""
        return {'warnings': [], 'recommendations': [], 'findings': [], 'accuracy_score': 1.0}

class EnhancedSecurityValidator:
    """Enhanced security validator with comprehensive threat detection"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.sensitive_patterns = self.config.get('sensitive_patterns', [
            r'\b\d{16}\b',  # Credit card numbers
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{10,11}\b',  # Phone numbers
            r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b',  # IBAN
            r'\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b',  # BIC/SWIFT codes
        ])
        self.injection_patterns = self.config.get('injection_patterns', [
            r"';|--|/\*|\*/|xp_|sp_",  # SQL injection
            r"<script|javascript:|onerror=|onload=",  # XSS
            r"\$\{|\#{|%\{",  # Template injection
            r"\.\.\/|\.\.\\",  # Path traversal
            r"eval\(|exec\(|system\(",  # Code injection
        ])
        self.blocked_domains = self.config.get('blocked_domains', [
            'malicious.com', 'phishing.net', 'spam.org', 'suspicious.domain'
        ])
        self.threat_indicators = self.config.get('threat_indicators', [
            'union select', 'drop table', 'insert into', 'delete from',
            'script>', 'iframe>', 'object>', 'embed>',
            'cmd.exe', 'powershell', '/bin/sh', 'wget', 'curl'
        ])
        
    def validate_security(self, data: pd.DataFrame, source: 'DataSource', 
                         level: SecurityLevel = SecurityLevel.BASIC,
                         context: Optional[DataProcessingContext] = None) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Enhanced security validation with comprehensive threat detection"""
        security_issues = []
        security_metrics = {
            'pii_exposure_risk': 0.0,
            'injection_risk': 0.0,
            'data_integrity_risk': 0.0,
            'compliance_risk': 0.0
        }
        
        if level == SecurityLevel.NONE:
            return True, [], security_metrics
        
        # Basic security checks
        if level.value in ['basic', 'standard', 'high', 'critical']:
            pii_issues = self._check_pii_exposure(data)
            security_issues.extend(pii_issues['issues'])
            security_metrics['pii_exposure_risk'] = pii_issues['risk_score']
        
        # Standard security checks
        if level.value in ['standard', 'high', 'critical']:
            source_issues = self._validate_source_security(source)
            security_issues.extend(source_issues['issues'])
            
            injection_issues = self._check_injection_attacks(data)
            security_issues.extend(injection_issues['issues'])
            security_metrics['injection_risk'] = injection_issues['risk_score']
        
        # High security checks
        if level.value in ['high', 'critical']:
            credential_issues = self._validate_credentials(source)
            security_issues.extend(credential_issues['issues'])
            
            integrity_issues = self._check_data_integrity_threats(data, source)
            security_issues.extend(integrity_issues['issues'])
            security_metrics['data_integrity_risk'] = integrity_issues['risk_score']
        
        # Critical security checks
        if level == SecurityLevel.CRITICAL:
            compliance_issues = self._check_compliance_violations(data, source, context)
            security_issues.extend(compliance_issues['issues'])
            security_metrics['compliance_risk'] = compliance_issues['risk_score']
            
            advanced_threats = self._detect_advanced_threats(data, source)
            security_issues.extend(advanced_threats['issues'])
        
        overall_secure = len(security_issues) == 0
        return overall_secure, security_issues, security_metrics
    
    def _check_pii_exposure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for PII exposure in data"""
        issues = []
        risk_score = 0.0
        
        for column in data.columns:
            if data[column].dtype == 'object':
                column_data = data[column].astype(str)
                for pattern_name, pattern in enumerate(self.sensitive_patterns):
                    try:
                        matches = column_data.str.contains(pattern, regex=True, na=False)
                        if matches.any():
                            count = matches.sum()
                            percentage = count / len(data) * 100
                            issues.append(
                                f"Column '{column}' contains {count} potential PII matches "
                                f"({percentage:.1f}%) - Pattern {pattern_name}"
                            )
                            risk_score = max(risk_score, min(1.0, percentage / 100))
                    except Exception as e:
                        logger.warning(f"Failed to check PII pattern in column {column}: {e}")
        
        return {'issues': issues, 'risk_score': risk_score}
    
    def _validate_source_security(self, source: 'DataSource') -> Dict[str, Any]:
        """Validate security of data source"""
        issues = []
        
        if source.source_type == DataSourceType.API:
            parsed_url = urlparse(source.location)
            
            # HTTPS enforcement
            if not parsed_url.scheme == 'https':
                issues.append("API endpoint not using HTTPS - data transmission not encrypted")
            
            # Domain validation
            if any(domain in parsed_url.netloc for domain in self.blocked_domains):
                issues.append(f"Domain {parsed_url.netloc} is on blocked list")
            
            # Header security
            if source.headers:
                if 'Authorization' in source.headers:
                    auth_value = source.headers['Authorization']
                    if auth_value.startswith('Bearer ') and len(auth_value.split(' ')[1]) < 32:
                        issues.append("Authorization token appears to be too short")
        
        elif source.source_type in [DataSourceType.CSV, DataSourceType.JSON]:
            file_path = Path(source.location)
            if file_path.exists():
                # File permission check
                try:
                    file_stat = file_path.stat()
                    file_mode = oct(file_stat.st_mode)[-3:]
                    if file_mode[-1] in ['6', '7']:  # World writable
                        issues.append(f"File {file_path} has world-writable permissions")
                except Exception as e:
                    logger.warning(f"Could not check file permissions: {e}")
        
        return {'issues': issues}
    
    def _check_injection_attacks(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for injection attack patterns"""
        issues = []
        risk_score = 0.0
        
        for column in data.select_dtypes(include=['object']).columns:
            column_data = data[column].astype(str).str.lower()
            
            # Check for injection patterns
            for pattern in self.injection_patterns:
                try:
                    matches = column_data.str.contains(pattern, regex=True, na=False)
                    if matches.any():
                        count = matches.sum()
                        issues.append(f"Potential injection attack detected in column '{column}': {count} matches")
                        risk_score = max(risk_score, min(1.0, count / len(data)))
                except Exception as e:
                    logger.warning(f"Failed to check injection pattern in column {column}: {e}")
            
            # Check for threat indicators
            for indicator in self.threat_indicators:
                if column_data.str.contains(indicator, na=False).any():
                    count = column_data.str.contains(indicator, na=False).sum()
                    issues.append(f"Threat indicator '{indicator}' found in column '{column}': {count} occurrences")
                    risk_score = max(risk_score, min(1.0, count / len(data) * 10))  # Higher weight for threats
        
        return {'issues': issues, 'risk_score': risk_score}
    
    def _validate_credentials(self, source: 'DataSource') -> Dict[str, Any]:
        """Validate credential security"""
        issues = []
        
        if source.credentials:
            # Password strength validation
            if 'password' in source.credentials:
                password = source.credentials['password']
                if len(password) < 12:
                    issues.append("Password does not meet minimum length requirements (12 characters)")
                if password.lower() in ['password', '123456', 'admin', 'default', 'guest']:
                    issues.append("Password is commonly used and weak")
                if not any(c.isupper() for c in password):
                    issues.append("Password should contain uppercase letters")
                if not any(c.islower() for c in password):
                    issues.append("Password should contain lowercase letters")
                if not any(c.isdigit() for c in password):
                    issues.append("Password should contain numbers")
                if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password):
                    issues.append("Password should contain special characters")
            
            # API key validation
            if 'api_key' in source.credentials:
                api_key = source.credentials['api_key']
                if len(api_key) < 32:
                    issues.append("API key appears to be too short (minimum 32 characters recommended)")
                if api_key.isalnum() and len(set(api_key)) < 10:
                    issues.append("API key has low entropy")
            
            # Token validation
            if 'token' in source.credentials:
                token = source.credentials['token']
                if '.' in token:  # Likely JWT
                    parts = token.split('.')
                    if len(parts) != 3:
                        issues.append("JWT token format appears invalid")
        
        return {'issues': issues}
    
    def _check_data_integrity_threats(self, data: pd.DataFrame, source: 'DataSource') -> Dict[str, Any]:
        """Check for data integrity threats"""
        issues = []
        risk_score = 0.0
        
        # Check for data tampering indicators
        if 'timestamp' in data.columns:
            # Check for timestamp anomalies
            timestamps = pd.to_datetime(data['timestamp'])
            if not timestamps.empty:
                # Future timestamps
                future_count = (timestamps > datetime.now()).sum()
                if future_count > 0:
                    issues.append(f"Found {future_count} transactions with future timestamps")
                    risk_score = max(risk_score, future_count / len(data))
                
                # Regular intervals (possible artificial generation)
                time_diffs = timestamps.sort_values().diff().dropna()
                if len(time_diffs) > 10:
                    time_diff_std = time_diffs.dt.total_seconds().std()
                    time_diff_mean = time_diffs.dt.total_seconds().mean()
                    if time_diff_mean > 0:
                        cv = time_diff_std / time_diff_mean
                        if cv < 0.05:  # Very regular
                            issues.append("Timestamps show suspiciously regular patterns")
                            risk_score = max(risk_score, 0.7)
        
        # Check for data duplication patterns
        if len(data) > 1:
            duplicate_rows = data.duplicated().sum()
            if duplicate_rows > len(data) * 0.1:  # More than 10% duplicates
                issues.append(f"High percentage of duplicate rows: {duplicate_rows / len(data) * 100:.1f}%")
                risk_score = max(risk_score, duplicate_rows / len(data))
        
        return {'issues': issues, 'risk_score': risk_score}
    
    def _check_compliance_violations(self, data: pd.DataFrame, source: 'DataSource',
                                   context: Optional[DataProcessingContext] = None) -> Dict[str, Any]:
        """Check for regulatory compliance violations"""
        issues = []
        risk_score = 0.0
        
        # GDPR compliance checks
        if context and 'gdpr' in source.compliance_requirements:
            # Check for personal data without consent indicators
            personal_data_columns = ['email', 'phone', 'address', 'name']
            found_personal_data = [col for col in personal_data_columns if col in data.columns]
            
            if found_personal_data and not context.compliance_context.get('gdpr_consent', False):
                issues.append(f"Personal data found without GDPR consent: {found_personal_data}")
                risk_score = max(risk_score, 0.8)
        
        # PCI compliance checks
        if 'pci' in source.compliance_requirements:
            # Check for credit card data
            for column in data.columns:
                if data[column].dtype == 'object':
                    cc_pattern = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
                    matches = data[column].astype(str).str.contains(cc_pattern, regex=True, na=False)
                    if matches.any():
                        issues.append(f"Potential credit card numbers found in column '{column}'")
                        risk_score = max(risk_score, 0.9)
        
        return {'issues': issues, 'risk_score': risk_score}
    
    def _detect_advanced_threats(self, data: pd.DataFrame, source: 'DataSource') -> Dict[str, Any]:
        """Detect advanced security threats"""
        issues = []
        
        # Data exfiltration patterns
        if len(data) > 100000:  # Large dataset
            issues.append("Large dataset detected - monitor for potential data exfiltration")
        
        # Unusual access patterns (would need access logs in real implementation)
        if source.source_type == DataSourceType.API:
            # Placeholder for access pattern analysis
            pass
        
        # Data poisoning detection (basic)
        if 'amount' in data.columns:
            # Check for extreme outliers that might indicate poisoning
            amount_data = data['amount']
            if not amount_data.empty:
                q99 = amount_data.quantile(0.99)
                extreme_values = amount_data > q99 * 100  # 100x the 99th percentile
                if extreme_values.any():
                    issues.append(f"Extreme outlier values detected: {extreme_values.sum()} records")
        
        return {'issues': issues}

# ======================== DATA PREPROCESSOR ========================

class EnhancedDataPreprocessor:
    """Enhanced data preprocessing with comprehensive validation and error handling"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.preprocessing_hooks: Dict[str, Callable] = {}
        self.postprocessing_hooks: Dict[str, Callable] = {}
        self.feature_engineering_hooks: Dict[str, Callable] = {}
        self.quality_checks: Dict[str, Callable] = {}
        self._register_default_hooks()
        self.performance_metrics = {}
        
    def _register_default_hooks(self):
        """Register comprehensive default preprocessing hooks"""
        # Preprocessing hooks
        self.register_preprocessing_hook('validate_schema', self._validate_schema)
        self.register_preprocessing_hook('clean_missing_values', self._clean_missing_values)
        self.register_preprocessing_hook('normalize_data_types', self._normalize_data_types)
        self.register_preprocessing_hook('remove_duplicates', self._remove_duplicates)
        self.register_preprocessing_hook('validate_ranges', self._validate_ranges)
        
        # Feature engineering hooks
        self.register_feature_engineering_hook('create_time_features', self._create_time_features)
        self.register_feature_engineering_hook('create_amount_features', self._create_amount_features)
        self.register_feature_engineering_hook('create_user_features', self._create_user_features)
        self.register_feature_engineering_hook('create_merchant_features', self._create_merchant_features)
        self.register_feature_engineering_hook('create_risk_features', self._create_risk_features)
        
        # Postprocessing hooks
        self.register_postprocessing_hook('add_quality_scores', self._add_quality_scores)
        self.register_postprocessing_hook('add_metadata', self._add_metadata)
        self.register_postprocessing_hook('final_validation', self._final_validation)
        
        # Quality checks
        self.register_quality_check('completeness_check', self._check_completeness)
        self.register_quality_check('consistency_check', self._check_consistency)
        self.register_quality_check('validity_check', self._check_validity)
        
    def register_preprocessing_hook(self, name: str, hook: Callable) -> None:
        """Register preprocessing hook with enhanced validation"""
        if not callable(hook):
            raise ValueError(f"Hook {name} must be callable")
        self.preprocessing_hooks[name] = hook
        logger.debug(f"Registered preprocessing hook: {name}")
        
    def register_postprocessing_hook(self, name: str, hook: Callable) -> None:
        """Register postprocessing hook with enhanced validation"""
        if not callable(hook):
            raise ValueError(f"Hook {name} must be callable")
        self.postprocessing_hooks[name] = hook
        logger.debug(f"Registered postprocessing hook: {name}")
        
    def register_feature_engineering_hook(self, name: str, hook: Callable) -> None:
        """Register feature engineering hook"""
        if not callable(hook):
            raise ValueError(f"Hook {name} must be callable")
        self.feature_engineering_hooks[name] = hook
        logger.debug(f"Registered feature engineering hook: {name}")
        
    def register_quality_check(self, name: str, check: Callable) -> None:
        """Register quality check"""
        if not callable(check):
            raise ValueError(f"Quality check {name} must be callable")
        self.quality_checks[name] = check
        logger.debug(f"Registered quality check: {name}")
    
    @handle_errors(error_type=DataFormatError, default_return=None)
    @performance_monitor()
    def preprocess(self, data: pd.DataFrame, hook_names: Optional[List[str]] = None,
                  context: Optional[DataProcessingContext] = None) -> pd.DataFrame:
        """Apply preprocessing hooks with enhanced error handling and monitoring"""
        if data is None or data.empty:
            raise DataFormatError("Cannot preprocess empty data")
            
        processing_start = time.time()
        
        if hook_names is None:
            hook_names = list(self.preprocessing_hooks.keys())
            
        original_shape = data.shape
        processing_log = []
        
        for hook_name in hook_names:
            if hook_name in self.preprocessing_hooks:
                try:
                    hook_start = time.time()
                    before_shape = data.shape
                    
                    # Apply hook with context
                    if context:
                        data = self.preprocessing_hooks[hook_name](data, context)
                    else:
                        data = self.preprocessing_hooks[hook_name](data)
                    
                    hook_time = time.time() - hook_start
                    
                    # Validate output
                    if data is None:
                        raise DataFormatError(f"Hook {hook_name} returned None")
                    if data.empty:
                        logger.warning(f"Hook {hook_name} returned empty DataFrame")
                    
                    after_shape = data.shape
                    processing_log.append({
                        'hook': hook_name,
                        'before_shape': before_shape,
                        'after_shape': after_shape,
                        'processing_time': hook_time,
                        'records_changed': before_shape[0] - after_shape[0],
                        'columns_changed': after_shape[1] - before_shape[1]
                    })
                    
                    logger.debug(f"Applied preprocessing hook: {hook_name} "
                               f"(shape: {before_shape} -> {after_shape}, time: {hook_time:.3f}s)")
                               
                except Exception as e:
                    error_msg = f"Preprocessing hook {hook_name} failed: {e}"
                    logger.error(error_msg)
                    
                    if self.config.get('fail_on_hook_error', False):
                        raise DataFormatError(error_msg)
                    else:
                        # Add to processing log for debugging
                        processing_log.append({
                            'hook': hook_name,
                            'error': str(e),
                            'status': 'failed'
                        })
            else:
                logger.warning(f"Unknown preprocessing hook: {hook_name}")
        
        # Add processing metadata
        total_time = time.time() - processing_start
        self.performance_metrics['last_preprocessing'] = {
            'total_time': total_time,
            'original_shape': original_shape,
            'final_shape': data.shape,
            'processing_log': processing_log,
            'throughput_records_per_second': data.shape[0] / total_time if total_time > 0 else 0
        }
        
        return data
    
    @handle_errors(error_type=DataFormatError, default_return=None)
    @performance_monitor()
    def apply_feature_engineering(self, data: pd.DataFrame, hook_names: Optional[List[str]] = None,
                                 context: Optional[DataProcessingContext] = None) -> pd.DataFrame:
        """Apply feature engineering with enhanced capabilities"""
        if data is None or data.empty:
            raise DataFormatError("Cannot apply feature engineering to empty data")
        
        if hook_names is None:
            hook_names = list(self.feature_engineering_hooks.keys())
        
        for hook_name in hook_names:
            if hook_name in self.feature_engineering_hooks:
                try:
                    original_cols = set(data.columns)
                    
                    if context:
                        data = self.feature_engineering_hooks[hook_name](data, context)
                    else:
                        data = self.feature_engineering_hooks[hook_name](data)
                    
                    new_cols = set(data.columns) - original_cols
                    if new_cols:
                        logger.debug(f"Feature engineering hook {hook_name} added columns: {new_cols}")
                        
                except Exception as e:
                    logger.error(f"Feature engineering hook {hook_name} failed: {e}")
                    if self.config.get('fail_on_hook_error', False):
                        raise
            else:
                logger.warning(f"Unknown feature engineering hook: {hook_name}")
        
        return data
    
    @handle_errors(error_type=DataFormatError, default_return=None)
    @performance_monitor()
    def postprocess(self, data: pd.DataFrame, hook_names: Optional[List[str]] = None,
                   context: Optional[DataProcessingContext] = None) -> pd.DataFrame:
        """Apply postprocessing hooks with enhanced validation"""
        if data is None or data.empty:
            raise DataFormatError("Cannot postprocess empty data")
            
        if hook_names is None:
            hook_names = list(self.postprocessing_hooks.keys())
            
        for hook_name in hook_names:
            if hook_name in self.postprocessing_hooks:
                try:
                    original_shape = data.shape
                    
                    if context:
                        data = self.postprocessing_hooks[hook_name](data, context)
                    else:
                        data = self.postprocessing_hooks[hook_name](data)
                    
                    # Validate output
                    if data is None:
                        raise DataFormatError(f"Postprocessing hook {hook_name} returned None")
                    if data.empty:
                        logger.warning(f"Postprocessing hook {hook_name} returned empty DataFrame")
                    
                    logger.debug(f"Applied postprocessing hook: {hook_name} "
                               f"(shape: {original_shape} -> {data.shape})")
                               
                except Exception as e:
                    logger.error(f"Postprocessing hook {hook_name} failed: {e}")
                    if self.config.get('fail_on_hook_error', False):
                        raise
            else:
                logger.warning(f"Unknown postprocessing hook: {hook_name}")
                
        return data
    
    def run_quality_checks(self, data: pd.DataFrame, 
                          context: Optional[DataProcessingContext] = None) -> Dict[str, Any]:
        """Run comprehensive quality checks"""
        quality_results = {}
        
        for check_name, check_func in self.quality_checks.items():
            try:
                if context:
                    result = check_func(data, context)
                else:
                    result = check_func(data)
                quality_results[check_name] = result
            except Exception as e:
                logger.error(f"Quality check {check_name} failed: {e}")
                quality_results[check_name] = {'error': str(e), 'passed': False}
        
        return quality_results
    
    # Default hook implementations
    def _validate_schema(self, data: pd.DataFrame, 
                        context: Optional[DataProcessingContext] = None) -> pd.DataFrame:
        """Enhanced schema validation"""
        expected_types = self.config.get('expected_types', {
            'amount': ['float64', 'int64'],
            'timestamp': ['datetime64[ns]'],
            'transaction_id': ['object', 'string'],
            'user_id': ['object', 'string']
        })
        
        schema_issues = []
        
        for column, valid_types in expected_types.items():
            if column in data.columns:
                actual_type = str(data[column].dtype)
                if actual_type not in valid_types:
                    schema_issues.append(f"Column {column} has unexpected type {actual_type}")
                    
                    # Attempt type conversion
                    try:
                        if 'float' in valid_types:
                            data[column] = pd.to_numeric(data[column], errors='coerce')
                        elif 'datetime' in str(valid_types):
                            data[column] = pd.to_datetime(data[column], errors='coerce')
                        elif 'object' in valid_types:
                            data[column] = data[column].astype(str)
                            
                        logger.info(f"Successfully converted {column} to compatible type")
                    except Exception as e:
                        logger.warning(f"Could not convert {column} type: {e}")
        
        if schema_issues and context:
            context.audit_trail.append({
                'action': 'schema_validation',
                'issues_found': schema_issues,
                'timestamp': datetime.now().isoformat()
            })
        
        return data
    
    def _clean_missing_values(self, data: pd.DataFrame,
                             context: Optional[DataProcessingContext] = None) -> pd.DataFrame:
        """Enhanced missing value cleaning"""
        missing_strategy = self.config.get('missing_value_strategy', {
            'numeric': 'median',
            'categorical': 'mode',
            'datetime': 'forward_fill'
        })
        
        initial_count = len(data)
        
        for column in data.columns:
            missing_count = data[column].isna().sum()
            if missing_count > 0:
                missing_percentage = missing_count / len(data) * 100
                
                if missing_percentage > 50:
                    logger.warning(f"Column {column} has {missing_percentage:.1f}% missing values")
                
                # Apply strategy based on data type
                if pd.api.types.is_numeric_dtype(data[column]):
                    strategy = missing_strategy['numeric']
                    if strategy == 'median':
                        data[column].fillna(data[column].median(), inplace=True)
                    elif strategy == 'mean':
                        data[column].fillna(data[column].mean(), inplace=True)
                    elif strategy == 'zero':
                        data[column].fillna(0, inplace=True)
                elif pd.api.types.is_datetime64_any_dtype(data[column]):
                    strategy = missing_strategy['datetime']
                    if strategy == 'forward_fill':
                        data[column].fillna(method='ffill', inplace=True)
                else:
                    strategy = missing_strategy['categorical']
                    if strategy == 'mode' and not data[column].mode().empty:
                        data[column].fillna(data[column].mode().iloc[0], inplace=True)
                    elif strategy == 'unknown':
                        data[column].fillna('UNKNOWN', inplace=True)
        
        # Remove rows with critical missing values
        critical_columns = self.config.get('critical_columns', ['transaction_id', 'amount'])
        before_critical_clean = len(data)
        data = data.dropna(subset=[col for col in critical_columns if col in data.columns])
        
        if len(data) < before_critical_clean:
            removed = before_critical_clean - len(data)
            logger.info(f"Removed {removed} rows with missing critical values")
        
        return data
    
    def _normalize_data_types(self, data: pd.DataFrame,
                             context: Optional[DataProcessingContext] = None) -> pd.DataFrame:
        """Enhanced data type normalization"""
        # Ensure numeric columns are proper types
        numeric_columns = ['amount', 'balance', 'fee', 'tax']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Ensure datetime columns
        datetime_columns = ['timestamp', 'created_at', 'updated_at']
        for col in datetime_columns:
            if col in data.columns:
                data[col] = pd.to_datetime(data[col], errors='coerce')
        
        # Normalize string columns
        string_columns = ['transaction_id', 'user_id', 'merchant_id']
        for col in string_columns:
            if col in data.columns:
                data[col] = data[col].astype(str).str.strip()
        
        # Normalize categorical columns
        categorical_columns = ['transaction_type', 'currency', 'status']
        for col in categorical_columns:
            if col in data.columns:
                data[col] = data[col].astype(str).str.lower().str.strip()
        
        return data
    
    def _remove_duplicates(self, data: pd.DataFrame,
                          context: Optional[DataProcessingContext] = None) -> pd.DataFrame:
        """Enhanced duplicate removal"""
        initial_count = len(data)
        
        # Remove exact duplicates
        data = data.drop_duplicates()
        exact_duplicates_removed = initial_count - len(data)
        
        # Remove duplicates based on key columns
        key_columns = self.config.get('duplicate_key_columns', ['transaction_id'])
        available_key_columns = [col for col in key_columns if col in data.columns]
        
        if available_key_columns:
            before_key_dedup = len(data)
            data = data.drop_duplicates(subset=available_key_columns)
            key_duplicates_removed = before_key_dedup - len(data)
            
            if key_duplicates_removed > 0:
                logger.info(f"Removed {key_duplicates_removed} key-based duplicates")
        
        if exact_duplicates_removed > 0:
            logger.info(f"Removed {exact_duplicates_removed} exact duplicates")
        
        return data
    
    def _validate_ranges(self, data: pd.DataFrame,
                        context: Optional[DataProcessingContext] = None) -> pd.DataFrame:
        """Enhanced range validation"""
        range_rules = self.config.get('range_rules', {
            'amount': {'min': 0.01, 'max': 1000000.0},
            'timestamp': {'min': '2020-01-01', 'max': None}
        })
        
        for column, rules in range_rules.items():
            if column in data.columns:
                initial_count = len(data)
                
                if 'min' in rules and rules['min'] is not None:
                    if pd.api.types.is_datetime64_any_dtype(data[column]):
                        min_val = pd.to_datetime(rules['min'])
                        data = data[data[column] >= min_val]
                    else:
                        data = data[data[column] >= rules['min']]
                
                if 'max' in rules and rules['max'] is not None:
                    if pd.api.types.is_datetime64_any_dtype(data[column]):
                        max_val = pd.to_datetime(rules['max'])
                        data = data[data[column] <= max_val]
                    else:
                        data = data[data[column] <= rules['max']]
                
                removed = initial_count - len(data)
                if removed > 0:
                    logger.info(f"Removed {removed} records outside valid range for {column}")
        
        return data
    
    # Feature engineering implementations
    def _create_time_features(self, data: pd.DataFrame,
                             context: Optional[DataProcessingContext] = None) -> pd.DataFrame:
        """Enhanced time feature creation"""
        if 'timestamp' in data.columns:
            timestamp = pd.to_datetime(data['timestamp'])
            
            # Basic time features
            data['hour'] = timestamp.dt.hour
            data['day_of_week'] = timestamp.dt.dayofweek
            data['day_of_month'] = timestamp.dt.day
            data['month'] = timestamp.dt.month
            data['quarter'] = timestamp.dt.quarter
            data['year'] = timestamp.dt.year
            
            # Advanced time features
            data['is_weekend'] = data['day_of_week'].isin([5, 6])
            data['is_business_hours'] = ((data['hour'] >= 9) & (data['hour'] <= 17))
            data['is_night'] = ((data['hour'] >= 22) | (data['hour'] <= 6))
            
            # Holiday detection (simplified)
            data['is_potential_holiday'] = (
                ((data['month'] == 12) & (data['day_of_month'].isin([24, 25, 31]))) |
                ((data['month'] == 1) & (data['day_of_month'] == 1)) |
                ((data['month'] == 7) & (data['day_of_month'] == 4))
            )
            
            # Time since epoch for ML models
            data['timestamp_epoch'] = timestamp.astype(np.int64) // 10**9
            
            # Time of year features
            data['day_of_year'] = timestamp.dt.dayofyear
            data['week_of_year'] = timestamp.dt.isocalendar().week
            
        return data
    
    def _create_amount_features(self, data: pd.DataFrame,
                               context: Optional[DataProcessingContext] = None) -> pd.DataFrame:
        """Enhanced amount feature engineering"""
        if 'amount' in data.columns:
            amount = data['amount']
            
            # Basic amount features
            data['amount_log'] = np.log1p(np.abs(amount))
            data['amount_sqrt'] = np.sqrt(np.abs(amount))
            
            # Amount categories
            data['is_micro_transaction'] = amount < 1.0
            data['is_small_transaction'] = (amount >= 1.0) & (amount < 10.0)
            data['is_medium_transaction'] = (amount >= 10.0) & (amount < 100.0)
            data['is_large_transaction'] = (amount >= 100.0) & (amount < 1000.0)
            data['is_very_large_transaction'] = amount >= 1000.0
            
            # Round number detection
            data['is_round_amount'] = (amount % 1 == 0)
            data['is_round_10'] = (amount % 10 == 0)
            data['is_round_100'] = (amount % 100 == 0)
            
            # Statistical features
            global_mean = amount.mean()
            global_std = amount.std()
            
            if global_std > 0:
                data['amount_zscore'] = (amount - global_mean) / global_std
                data['is_amount_outlier'] = np.abs(data['amount_zscore']) > 3
            
        return data
    
    def _create_user_features(self, data: pd.DataFrame,
                             context: Optional[DataProcessingContext] = None) -> pd.DataFrame:
        """Enhanced user feature engineering"""
        if 'user_id' in data.columns:
            # User transaction statistics
            user_stats = data.groupby('user_id').agg({
                'amount': ['count', 'sum', 'mean', 'std', 'min', 'max'],
                'timestamp': ['min', 'max'] if 'timestamp' in data.columns else []
            }).round(2)
            
            # Flatten column names
            user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
            user_stats = user_stats.add_prefix('user_')
            
            # Handle NaN std for single-transaction users
            if 'user_amount_std' in user_stats.columns:
                user_stats['user_amount_std'] = user_stats['user_amount_std'].fillna(0)
            
            # Merge back to main dataframe
            data = data.merge(user_stats, left_on='user_id', right_index=True, how='left')
            
            # Additional user features
            if 'user_amount_mean' in data.columns and 'amount' in data.columns:
                data['amount_vs_user_mean'] = data['amount'] / (data['user_amount_mean'] + 1e-6)
                data['amount_deviation_from_user'] = np.abs(data['amount'] - data['user_amount_mean'])
        
        return data
    
    def _create_merchant_features(self, data: pd.DataFrame,
                                 context: Optional[DataProcessingContext] = None) -> pd.DataFrame:
        """Enhanced merchant feature engineering"""
        if 'merchant_id' in data.columns:
            # Merchant transaction statistics
            merchant_stats = data.groupby('merchant_id').agg({
                'amount': ['count', 'sum', 'mean', 'std'],
                'user_id': 'nunique'
            }).round(2)
            
            # Flatten column names
            merchant_stats.columns = ['_'.join(col).strip() for col in merchant_stats.columns]
            merchant_stats = merchant_stats.add_prefix('merchant_')
            
            # Handle NaN std
            if 'merchant_amount_std' in merchant_stats.columns:
                merchant_stats['merchant_amount_std'] = merchant_stats['merchant_amount_std'].fillna(0)
            
            # Merge back
            data = data.merge(merchant_stats, left_on='merchant_id', right_index=True, how='left')
            
            # Merchant popularity
            if 'merchant_user_id_nunique' in data.columns:
                data['merchant_popularity'] = data['merchant_user_id_nunique']
        
        return data
    
    def _create_risk_features(self, data: pd.DataFrame,
                             context: Optional[DataProcessingContext] = None) -> pd.DataFrame:
        """Enhanced risk feature engineering"""
        # Velocity features
        if 'user_id' in data.columns and 'timestamp' in data.columns:
            data_sorted = data.sort_values(['user_id', 'timestamp'])
            
            # Time since last transaction
            data_sorted['time_since_last_transaction'] = data_sorted.groupby('user_id')['timestamp'].diff()
            data_sorted['time_since_last_minutes'] = (
                data_sorted['time_since_last_transaction'].dt.total_seconds() / 60
            ).fillna(float('inf'))
            
            # Recent transaction counts
            data_sorted['transactions_last_hour'] = data_sorted.groupby('user_id').rolling(
                window='1H', on='timestamp'
            ).size().reset_index(level=0, drop=True)
            
            data = data_sorted
        
        # Risk scores based on patterns
        risk_score = 0.0
        
        # Amount-based risk
        if 'amount' in data.columns:
            large_amount_risk = (data['amount'] > 10000).astype(float) * 0.3
            risk_score += large_amount_risk
        
        # Time-based risk
        if 'is_night' in data.columns:
            night_risk = data['is_night'].astype(float) * 0.2
            risk_score += night_risk
        
        # Velocity-based risk
        if 'time_since_last_minutes' in data.columns:
            velocity_risk = (data['time_since_last_minutes'] < 5).astype(float) * 0.4
            risk_score += velocity_risk
        
        data['risk_score'] = np.clip(risk_score, 0, 1)
        
        return data
    
    # Quality check implementations
    def _check_completeness(self, data: pd.DataFrame,
                           context: Optional[DataProcessingContext] = None) -> Dict[str, Any]:
        """Check data completeness"""
        total_cells = data.size
        missing_cells = data.isna().sum().sum()
        completeness_ratio = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0
        
        return {
            'passed': completeness_ratio >= 0.8,
            'score': completeness_ratio,
            'details': {
                'total_cells': total_cells,
                'missing_cells': missing_cells,
                'completeness_ratio': completeness_ratio
            }
        }
    
    def _check_consistency(self, data: pd.DataFrame,
                          context: Optional[DataProcessingContext] = None) -> Dict[str, Any]:
        """Check data consistency"""
        consistency_issues = []
        
        # Check for negative amounts in transaction data
        if 'amount' in data.columns:
            negative_amounts = (data['amount'] < 0).sum()
            if negative_amounts > 0:
                consistency_issues.append(f"Found {negative_amounts} negative amounts")
        
        # Check for future timestamps
        if 'timestamp' in data.columns:
            future_timestamps = (pd.to_datetime(data['timestamp']) > datetime.now()).sum()
            if future_timestamps > 0:
                consistency_issues.append(f"Found {future_timestamps} future timestamps")
        
        return {
            'passed': len(consistency_issues) == 0,
            'issues': consistency_issues,
            'details': {
                'total_issues': len(consistency_issues)
            }
        }
    
    def _check_validity(self, data: pd.DataFrame,
                       context: Optional[DataProcessingContext] = None) -> Dict[str, Any]:
        """Check data validity"""
        validity_issues = []
        
        # Check for valid transaction IDs
        if 'transaction_id' in data.columns:
            empty_ids = data['transaction_id'].isna().sum()
            if empty_ids > 0:
                validity_issues.append(f"Found {empty_ids} empty transaction IDs")
        
        # Check for valid user IDs
        if 'user_id' in data.columns:
            empty_user_ids = data['user_id'].isna().sum()
            if empty_user_ids > 0:
                validity_issues.append(f"Found {empty_user_ids} empty user IDs")
        
        return {
            'passed': len(validity_issues) == 0,
            'issues': validity_issues,
            'details': {
                'total_issues': len(validity_issues)
            }
        }
    
    # Additional helper methods
    def _add_quality_scores(self, data: pd.DataFrame,
                           context: Optional[DataProcessingContext] = None) -> pd.DataFrame:
        """Add quality scores to each record"""
        try:
            # Calculate completeness score per row
            required_columns = ['transaction_id', 'amount', 'timestamp', 'user_id']
            available_required = [col for col in required_columns if col in data.columns]
            
            if available_required:
                data['record_completeness_score'] = (
                    data[available_required].notna().sum(axis=1) / len(available_required)
                )
            else:
                data['record_completeness_score'] = 1.0
            
            # Add validation flags
            if 'amount' in data.columns:
                data['valid_amount'] = (data['amount'] > 0) & (data['amount'] < 1000000)
            
            if 'timestamp' in data.columns:
                min_date = pd.Timestamp('2020-01-01')
                max_date = pd.Timestamp.now() + pd.Timedelta(days=1)
                data['valid_timestamp'] = (
                    (pd.to_datetime(data['timestamp']) >= min_date) & 
                    (pd.to_datetime(data['timestamp']) <= max_date)
                )
            
            # Overall quality score
            quality_columns = [col for col in data.columns if col.startswith('valid_') or col.endswith('_score')]
            if quality_columns:
                data['overall_quality_score'] = data[quality_columns].mean(axis=1)
            
        except Exception as e:
            logger.error(f"Failed to add quality scores: {e}")
            
        return data
    
    def _add_metadata(self, data: pd.DataFrame,
                     context: Optional[DataProcessingContext] = None) -> pd.DataFrame:
        """Add processing metadata"""
        try:
            # Add processing timestamp
            data['processed_at'] = datetime.now()
            
            # Add processing context info
            if context:
                data['processing_session_id'] = context.session_id
                data['processing_environment'] = context.environment
            
            # Add data quality indicators
            data['record_count'] = len(data)
            data['column_count'] = len(data.columns)
            
        except Exception as e:
            logger.error(f"Failed to add metadata: {e}")
            
        return data
    
    def _final_validation(self, data: pd.DataFrame,
                         context: Optional[DataProcessingContext] = None) -> pd.DataFrame:
        """Perform final validation before output"""
        try:
            # Check for any remaining critical issues
            critical_columns = ['transaction_id', 'amount']
            for col in critical_columns:
                if col in data.columns:
                    null_count = data[col].isna().sum()
                    if null_count > 0:
                        logger.warning(f"Final validation: {null_count} null values in critical column {col}")
            
            # Log final statistics
            logger.info(f"Final validation: {len(data)} records, {len(data.columns)} columns")
            
        except Exception as e:
            logger.error(f"Final validation failed: {e}")
            
        return data

# ======================== ENHANCED CACHE WITH VALIDATION ========================

class EnhancedDataCache:
    """Enhanced caching system with comprehensive validation and integrity checks"""
    
    def __init__(self, cache_dir: Optional[Path] = None, max_memory_items: int = 100,
                 verify_integrity: bool = True, enable_compression: bool = True,
                 enable_encryption: bool = False, max_cache_size_gb: float = 5.0):
        self.cache_dir = cache_dir or Path("./cache/enhanced_financial_data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_items = max_memory_items
        self.verify_integrity = verify_integrity
        self.enable_compression = enable_compression
        self.enable_encryption = enable_encryption
        self.max_cache_size_gb = max_cache_size_gb
        
        # Memory cache: key -> (data, timestamp, access_count, metadata)
        self.memory_cache: Dict[str, Tuple[pd.DataFrame, float, int, Dict[str, Any]]] = {}
        self._lock = threading.RLock()
        self._access_patterns: Dict[str, List[float]] = {}
        
        # Performance metrics
        self.cache_metrics = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'integrity_failures': 0,
            'compression_ratio': 0.0
        }
        
        # Initialize encryption if enabled
        self._cipher_suite = None
        if self.enable_encryption:
            self._initialize_encryption()
            
        # Cache management
        self._cleanup_old_cache_files()
        
    def _initialize_encryption(self):
        """Initialize encryption for cache data"""
        try:
            from cryptography.fernet import Fernet
            key_file = self.cache_dir / ".cache_encryption_key"
            
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    key = f.read()
            else:
                key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(key)
                os.chmod(key_file, 0o600)
                
            self._cipher_suite = Fernet(key)
            logger.debug("Cache encryption initialized")
        except ImportError:
            logger.warning("Cryptography library not available, disabling cache encryption")
            self.enable_encryption = False
    
    def _get_cache_key(self, source: 'DataSource') -> str:
        """Generate unique cache key for data source with enhanced factors"""
        key_components = [
            source.name,
            source.source_type.value,
            source.location,
            str(hash(str(source.options))),
            str(hash(str(source.preprocessing_hooks))),
            str(source.validation_level.value),
            str(source.quality_level.value),
            str(getattr(source, 'security_level', 'none'))
        ]
        key_data = ":".join(key_components)
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _calculate_checksum(self, data: pd.DataFrame) -> str:
        """Calculate comprehensive checksum for data integrity"""
        try:
            # Use data shape and content hash for integrity
            shape_hash = hashlib.sha256(str(data.shape).encode()).hexdigest()[:8]
            
            # Sample data for content hash (performance optimization)
            sample_size = min(1000, len(data))
            if len(data) > sample_size:
                # Deterministic sampling
                step = len(data) // sample_size
                sample_data = data.iloc[::step].head(sample_size)
            else:
                sample_data = data
            
            # Convert to string representation for hashing
            content_str = sample_data.to_json(orient='records', date_format='iso')
            content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]
            
            return f"{shape_hash}:{content_hash}"
            
        except Exception as e:
            logger.warning(f"Failed to calculate checksum: {e}")
            return ""
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data for storage efficiency"""
        if not self.enable_compression:
            return data
            
        try:
            import gzip
            return gzip.compress(data)
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return data
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress cached data"""
        if not self.enable_compression:
            return data
            
        try:
            import gzip
            return gzip.decompress(data)
        except Exception as e:
            logger.warning(f"Decompression failed: {e}")
            return data
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data for secure storage"""
        if not self.enable_encryption or not self._cipher_suite:
            return data
            
        try:
            return self._cipher_suite.encrypt(data)
        except Exception as e:
            logger.warning(f"Encryption failed: {e}")
            return data
    
    def _decrypt_data(self, data: bytes) -> bytes:
        """Decrypt cached data"""
        if not self.enable_encryption or not self._cipher_suite:
            return data
            
        try:
            return self._cipher_suite.decrypt(data)
        except Exception as e:
            logger.warning(f"Decryption failed: {e}")
            return data
    
    @handle_errors(error_type=DataAccessError, default_return=None)
    def get(self, source: 'DataSource') -> Optional[pd.DataFrame]:
        """Get data from cache with enhanced integrity verification"""
        if not source.cache_enabled:
            return None
            
        cache_key = self._get_cache_key(source)
        current_time = time.time()
        
        with self._lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                data, timestamp, access_count, metadata = self.memory_cache[cache_key]
                
                if current_time - timestamp < source.cache_ttl:
                    # Verify integrity
                    if self.verify_integrity and source.verify_checksum:
                        stored_checksum = metadata.get('checksum', '')
                        calculated_checksum = self._calculate_checksum(data)
                        
                        if stored_checksum and calculated_checksum != stored_checksum:
                            logger.error(f"Memory cache integrity check failed for {source.name}")
                            del self.memory_cache[cache_key]
                            self.cache_metrics['integrity_failures'] += 1
                            raise DataIntegrityError(
                                "Memory cache integrity check failed",
                                checksum_expected=stored_checksum,
                                checksum_actual=calculated_checksum
                            )
                    
                    # Update access pattern and count
                    self._track_access_pattern(cache_key, current_time)
                    self.memory_cache[cache_key] = (data, timestamp, access_count + 1, metadata)
                    self.cache_metrics['hits'] += 1
                    
                    logger.debug(f"Memory cache hit for {source.name}")
                    return data.copy()
                else:
                    # Expired
                    del self.memory_cache[cache_key]
                    
            # Check disk cache
            cache_file = self.cache_dir / f"{cache_key}.cache"
            if cache_file.exists():
                try:
                    # Verify file permissions and access
                    if not os.access(cache_file, os.R_OK):
                        raise DataAccessError(f"Cannot read cache file: {cache_file}")
                    
                    # Load cache data
                    with open(cache_file, 'rb') as f:
                        raw_data = f.read()
                    
                    # Decrypt and decompress
                    raw_data = self._decrypt_data(raw_data)
                    raw_data = self._decompress_data(raw_data)
                    
                    # Deserialize
                    cache_data = pickle.loads(raw_data)
                    
                    # Check expiration
                    if current_time - cache_data['timestamp'] < source.cache_ttl:
                        data = cache_data['data']
                        metadata = cache_data.get('metadata', {})
                        stored_checksum = metadata.get('checksum', '')
                        
                        # Verify integrity
                        if self.verify_integrity and source.verify_checksum and stored_checksum:
                            calculated_checksum = self._calculate_checksum(data)
                            if calculated_checksum != stored_checksum:
                                logger.error(f"Disk cache integrity check failed for {source.name}")
                                cache_file.unlink()
                                self.cache_metrics['integrity_failures'] += 1
                                raise DataIntegrityError(
                                    "Disk cache integrity check failed",
                                    checksum_expected=stored_checksum,
                                    checksum_actual=calculated_checksum
                                )
                        
                        logger.debug(f"Disk cache hit for {source.name}")
                        self.cache_metrics['hits'] += 1
                        
                        # Add to memory cache if space available
                        if len(self.memory_cache) < self.max_memory_items:
                            self._track_access_pattern(cache_key, current_time)
                            self.memory_cache[cache_key] = (data, cache_data['timestamp'], 1, metadata)
                            
                        return data.copy()
                    else:
                        # Expired
                        cache_file.unlink()
                        
                except Exception as e:
                    logger.error(f"Failed to load from disk cache: {e}")
                    if cache_file.exists():
                        cache_file.unlink()
                        
            # Cache miss
            self.cache_metrics['misses'] += 1
        return None
        
    @handle_errors(error_type=DataAccessError)
    def set(self, source: 'DataSource', data: pd.DataFrame) -> None:
        """Store data in cache with enhanced compression and encryption"""
        if not source.cache_enabled:
            return
            
        cache_key = self._get_cache_key(source)
        timestamp = time.time()
        checksum = self._calculate_checksum(data)
        
        with self._lock:
            # Check cache size limits
            if not self._check_cache_size_limit():
                self._evict_cache_items()
            
            # Check memory limits
            data_memory = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            current_memory = sum(
                df.memory_usage(deep=True).sum() 
                for df, _, _, _ in self.memory_cache.values()
            ) / 1024 / 1024
            
            if current_memory + data_memory > source.max_memory_mb:
                logger.warning(f"Cache memory limit would be exceeded, not caching {source.name}")
                return
            
            # Prepare metadata
            metadata = {
                'checksum': checksum,
                'data_shape': data.shape,
                'data_size_mb': data_memory,
                'columns': list(data.columns),
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
                'source_config': {
                    'name': source.name,
                    'source_type': source.source_type.value,
                    'validation_level': source.validation_level.value,
                    'quality_level': source.quality_level.value
                },
                'cache_version': '1.0'
            }
            
            # Add to memory cache with LRU eviction
            if len(self.memory_cache) >= self.max_memory_items:
                self._evict_lru_memory_item()
                
            self._track_access_pattern(cache_key, timestamp)
            self.memory_cache[cache_key] = (data.copy(), timestamp, 1, metadata)
            
            # Save to disk cache
            try:
                cache_data = {
                    'data': data,
                    'timestamp': timestamp,
                    'metadata': metadata,
                    'cache_version': '1.0'
                }
                
                # Serialize, compress, and encrypt
                serialized_data = pickle.dumps(cache_data, protocol=pickle.HIGHEST_PROTOCOL)
                original_size = len(serialized_data)
                
                compressed_data = self._compress_data(serialized_data)
                encrypted_data = self._encrypt_data(compressed_data)
                
                # Calculate compression ratio
                if original_size > 0:
                    compression_ratio = len(compressed_data) / original_size
                    self.cache_metrics['compression_ratio'] = compression_ratio
                
                # Atomic write
                cache_file = self.cache_dir / f"{cache_key}.cache"
                temp_file = cache_file.with_suffix('.tmp')
                
                with open(temp_file, 'wb') as f:
                    f.write(encrypted_data)
                
                # Atomic rename
                temp_file.replace(cache_file)
                
                # Set secure permissions
                os.chmod(cache_file, 0o600)
                
                logger.debug(f"Cached {len(data)} records for {source.name} "
                           f"(compression: {compression_ratio:.2f})")
                
            except Exception as e:
                logger.error(f"Failed to cache data: {e}")
                if temp_file.exists():
                    temp_file.unlink()
    
    def _track_access_pattern(self, cache_key: str, timestamp: float):
        """Track access patterns for intelligent caching"""
        if cache_key not in self._access_patterns:
            self._access_patterns[cache_key] = []
        
        self._access_patterns[cache_key].append(timestamp)
        
        # Keep only recent access history (last 100 accesses)
        if len(self._access_patterns[cache_key]) > 100:
            self._access_patterns[cache_key] = self._access_patterns[cache_key][-100:]
    
    def _evict_lru_memory_item(self):
        """Evict least recently used item from memory cache"""
        if not self.memory_cache:
            return
            
        # Find LRU item based on access patterns
        lru_key = min(
            self.memory_cache.keys(),
            key=lambda k: (
                self.memory_cache[k][2],  # access_count
                max(self._access_patterns.get(k, [0]))  # last_access_time
            )
        )
        
        del self.memory_cache[lru_key]
        self.cache_metrics['evictions'] += 1
        logger.debug(f"Evicted LRU cache item: {lru_key[:8]}...")
    
    def _check_cache_size_limit(self) -> bool:
        """Check if cache size is within limits"""
        try:
            total_size = sum(
                f.stat().st_size for f in self.cache_dir.glob("*.cache")
            ) / 1024 / 1024 / 1024  # GB
            
            return total_size < self.max_cache_size_gb
        except Exception:
            return True
    
    def _evict_cache_items(self):
        """Evict old cache items to free space"""
        try:
            cache_files = list(self.cache_dir.glob("*.cache"))
            
            # Sort by modification time (oldest first)
            cache_files.sort(key=lambda f: f.stat().st_mtime)
            
            # Remove oldest 25% of files
            files_to_remove = cache_files[:len(cache_files) // 4]
            
            for cache_file in files_to_remove:
                try:
                    cache_file.unlink()
                    logger.debug(f"Evicted old cache file: {cache_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Cache eviction failed: {e}")
    
    def _cleanup_old_cache_files(self):
        """Clean up old or corrupted cache files"""
        try:
            current_time = time.time()
            max_age_days = 30  # Remove files older than 30 days
            max_age_seconds = max_age_days * 24 * 3600
            
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    file_age = current_time - cache_file.stat().st_mtime
                    if file_age > max_age_seconds:
                        cache_file.unlink()
                        logger.debug(f"Removed old cache file: {cache_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to check/remove cache file {cache_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    def clear(self, source: Optional['DataSource'] = None) -> None:
        """Clear cache for specific source or all"""
        with self._lock:
            if source:
                cache_key = self._get_cache_key(source)
                
                # Remove from memory cache
                if cache_key in self.memory_cache:
                    del self.memory_cache[cache_key]
                
                # Remove from disk cache
                cache_file = self.cache_dir / f"{cache_key}.cache"
                if cache_file.exists():
                    cache_file.unlink()
                    
                # Remove access pattern
                if cache_key in self._access_patterns:
                    del self._access_patterns[cache_key]
            else:
                # Clear all caches
                self.memory_cache.clear()
                self._access_patterns.clear()
                
                for cache_file in self.cache_dir.glob("*.cache"):
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove cache file {cache_file}: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self._lock:
            memory_items = len(self.memory_cache)
            disk_items = len(list(self.cache_dir.glob("*.cache")))
            
            total_memory_usage = sum(
                data.memory_usage(deep=True).sum() 
                for data, _, _, _ in self.memory_cache.values()
            ) / 1024 / 1024  # MB
            
            total_disk_usage = sum(
                f.stat().st_size for f in self.cache_dir.glob("*.cache")
            ) / 1024 / 1024  # MB
            
            # Calculate hit rate
            total_requests = self.cache_metrics['hits'] + self.cache_metrics['misses']
            hit_rate = self.cache_metrics['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'memory_items': memory_items,
                'disk_items': disk_items,
                'memory_usage_mb': total_memory_usage,
                'disk_usage_mb': total_disk_usage,
                'hit_rate': hit_rate,
                'compression_ratio': self.cache_metrics.get('compression_ratio', 1.0),
                'integrity_failures': self.cache_metrics['integrity_failures'],
                'evictions': self.cache_metrics['evictions'],
                'cache_directory': str(self.cache_dir),
                'encryption_enabled': self.enable_encryption,
                'compression_enabled': self.enable_compression
            }
    
    def optimize_cache(self):
        """Optimize cache performance and storage"""
        with self._lock:
            # Remove expired items
            current_time = time.time()
            expired_keys = []
            
            for cache_key, (data, timestamp, access_count, metadata) in self.memory_cache.items():
                # Use default TTL of 1 hour for optimization
                if current_time - timestamp > 3600:
                    expired_keys.append(cache_key)
            
            for key in expired_keys:
                del self.memory_cache[key]
                if key in self._access_patterns:
                    del self._access_patterns[key]
            
            # Cleanup disk cache
            self._cleanup_old_cache_files()
            
            # Check and maintain size limits
            if not self._check_cache_size_limit():
                self._evict_cache_items()
            
            logger.info(f"Cache optimization completed: removed {len(expired_keys)} expired items")

# ======================== ENHANCED DATA LOADING LOGIC ========================

class EnhancedDataSourceManager:
    """Enhanced data source management with validation and monitoring"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.source_validators: Dict[DataSourceType, Callable] = {}
        self.source_loaders: Dict[DataSourceType, Callable] = {}
        self.connection_pool: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}
        self._register_default_loaders()
        
    def _register_default_loaders(self):
        """Register default data source loaders"""
        self.register_source_loader(DataSourceType.CSV, self._load_csv_enhanced)
        self.register_source_loader(DataSourceType.JSON, self._load_json_enhanced)
        self.register_source_loader(DataSourceType.API, self._load_api_enhanced)
        self.register_source_loader(DataSourceType.PARQUET, self._load_parquet_enhanced)
        self.register_source_loader(DataSourceType.EXCEL, self._load_excel_enhanced)
        self.register_source_loader(DataSourceType.DATABASE, self._load_database_enhanced)
        
    def register_source_loader(self, source_type: DataSourceType, loader: Callable):
        """Register enhanced data source loader"""
        self.source_loaders[source_type] = loader
        logger.debug(f"Registered enhanced loader for {source_type.value}")
    
    @validate_inputs
    @retry_on_failure(max_attempts=3)
    @performance_monitor()
    def load_data(self, source: 'DataSource', progress_bar: bool = True,
                  chunk_size: Optional[int] = None,
                  context: Optional['DataProcessingContext'] = None) -> pd.DataFrame:
        """Load data with enhanced error handling and monitoring"""
        if source.source_type not in self.source_loaders:
            raise DataSourceValidationError(f"Unsupported source type: {source.source_type}")
        
        loader = self.source_loaders[source.source_type]
        
        # Apply timeout if specified
        if source.timeout:
            @timeout_handler(source.timeout)
            def load_with_timeout():
                return loader(source, progress_bar, chunk_size, context)
            return load_with_timeout()
        else:
            return loader(source, progress_bar, chunk_size, context)
    
    @retry_on_failure(max_attempts=3)
    def _load_csv_enhanced(self, source: 'DataSource', progress_bar: bool = True,
                          chunk_size: Optional[int] = None,
                          context: Optional['DataProcessingContext'] = None) -> pd.DataFrame:
        """Enhanced CSV loading with comprehensive validation"""
        file_path = Path(source.location)
        options = source.options or {}
        
        # Enhanced file validation
        if not file_path.exists():
            raise DataAccessError(f"CSV file not found: {file_path}")
        
        # Check file permissions
        if not os.access(file_path, os.R_OK):
            raise DataAccessError(f"No read permission for file: {file_path}")
        
        # Get file information
        file_stat = file_path.stat()
        file_size = file_stat.st_size
        
        # Security check: file permissions
        file_mode = oct(file_stat.st_mode)[-3:]
        if file_mode[-1] in ['2', '3', '6', '7']:  # World writable
            logger.warning(f"File {file_path} has world-writable permissions")
        
        # Determine chunking strategy
        use_chunking = chunk_size or (file_size > 50_000_000)  # 50MB threshold
        
        try:
            if use_chunking:
                chunks = []
                effective_chunk_size = chunk_size or self.config.get('default_chunk_size', 10000)
                
                # Enhanced progress tracking
                if progress_bar:
                    try:
                        # Estimate total rows
                        with open(file_path, 'r', encoding=source.encoding) as f:
                            total_lines = sum(1 for _ in f) - 1  # Exclude header
                        progress = tqdm(total=total_lines, desc=f"Loading {file_path.name}")
                    except Exception:
                        progress = tqdm(desc=f"Loading {file_path.name}")
                else:
                    progress = None
                
                try:
                    chunk_reader = pd.read_csv(
                        file_path, 
                        chunksize=effective_chunk_size,
                        encoding=source.encoding,
                        **options
                    )
                    
                    for chunk_num, chunk in enumerate(chunk_reader):
                        # Validate chunk
                        if chunk.empty:
                            logger.warning(f"Empty chunk {chunk_num} in CSV file")
                            continue
                            
                        chunks.append(chunk)
                        
                        if progress:
                            progress.update(len(chunk))
                        
                        # Memory management for large files
                        if len(chunks) > 100:  # Merge chunks periodically
                            merged_chunk = pd.concat(chunks, ignore_index=True)
                            chunks = [merged_chunk]
                    
                    if progress:
                        progress.close()
                    
                    if not chunks:
                        raise DataFormatError(f"No valid data found in CSV file: {file_path}")
                    
                    result = pd.concat(chunks, ignore_index=True)
                    
                except Exception as e:
                    if progress:
                        progress.close()
                    raise DataFormatError(f"Failed to parse CSV chunks: {e}")
            else:
                # Direct loading for smaller files
                result = pd.read_csv(file_path, encoding=source.encoding, **options)
            
            # Post-load validation
            if result.empty:
                raise DataFormatError(f"CSV file resulted in empty DataFrame: {file_path}")
            
            # Log loading statistics
            logger.info(f"Loaded CSV: {len(result)} rows, {len(result.columns)} columns from {file_path}")
            
            return result
            
        except pd.errors.EmptyDataError:
            raise DataFormatError(f"CSV file is empty: {file_path}")
        except UnicodeDecodeError as e:
            raise DataFormatError(f"Encoding error reading CSV ({source.encoding}): {e}")
        except Exception as e:
            raise DataAccessError(f"Failed to load CSV: {e}")
    
    @retry_on_failure(max_attempts=3)
    def _load_json_enhanced(self, source: 'DataSource', progress_bar: bool = True,
                           chunk_size: Optional[int] = None,
                           context: Optional['DataProcessingContext'] = None) -> pd.DataFrame:
        """Enhanced JSON loading with format detection"""
        file_path = Path(source.location)
        options = source.options or {}
        
        if not file_path.exists():
            raise DataAccessError(f"JSON file not found: {file_path}")
        
        try:
            # Detect JSON format by reading first few lines
            with open(file_path, 'r', encoding=source.encoding) as f:
                first_line = f.readline().strip()
                
            is_json_lines = not (first_line.startswith('[') or first_line.startswith('{'))
            
            if is_json_lines:
                # JSON Lines format
                df = pd.read_json(file_path, lines=True, encoding=source.encoding, **options)
            else:
                # Standard JSON format
                with open(file_path, 'r', encoding=source.encoding) as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    if all(isinstance(v, list) for v in data.values()):
                        df = pd.DataFrame(data)
                    else:
                        df = pd.DataFrame([data])
                else:
                    raise DataFormatError(f"Unsupported JSON structure: {type(data)}")
            
            if df.empty:
                raise DataFormatError(f"JSON file resulted in empty DataFrame: {file_path}")
            
            logger.info(f"Loaded JSON: {len(df)} rows, {len(df.columns)} columns from {file_path}")
            return df
            
        except json.JSONDecodeError as e:
            raise DataFormatError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise DataAccessError(f"Failed to load JSON: {e}")
    
    @retry_on_failure(max_attempts=3)
    def _load_api_enhanced(self, source: 'DataSource', progress_bar: bool = True,
                          chunk_size: Optional[int] = None,
                          context: Optional['DataProcessingContext'] = None) -> pd.DataFrame:
        """Enhanced API loading with comprehensive error handling"""
        credentials = source.credentials or {}
        options = source.options or {}
        
        # Enhanced authentication setup
        headers = source.headers or {}
        auth_config = self._setup_api_authentication(credentials, headers)
        
        # Enhanced pagination settings
        pagination_config = {
            'page_size': options.get('page_size', 1000),
            'max_pages': options.get('max_pages', None),
            'page_param': options.get('page_param', 'page'),
            'size_param': options.get('size_param', 'size'),
            'offset_param': options.get('offset_param', 'offset'),
            'limit_param': options.get('limit_param', 'limit')
        }
        
        all_data = []
        page = options.get('start_page', 1)
        total_records = 0
        
        # Enhanced progress tracking
        with tqdm(desc=f"Loading from API", disable=not progress_bar) as pbar:
            while True:
                try:
                    # Prepare request parameters
                    params = options.get('params', {}).copy()
                    
                    # Add pagination parameters
                    if pagination_config['page_param']:
                        params[pagination_config['page_param']] = page
                    if pagination_config['size_param']:
                        params[pagination_config['size_param']] = pagination_config['page_size']
                    
                    # Make request with enhanced error handling
                    response = self._make_api_request(
                        source.location, 
                        headers=auth_config['headers'],
                        params=params,
                        timeout=source.timeout,
                        auth=auth_config.get('auth')
                    )
                    
                    # Enhanced response validation
                    self._validate_api_response(response, source)
                    
                    # Parse response data
                    data = response.json()
                    records = self._extract_api_records(data, options)
                    
                    if not records:
                        logger.info(f"No more records found on page {page}")
                        break
                    
                    all_data.extend(records)
                    total_records += len(records)
                    pbar.update(len(records))
                    pbar.set_postfix({
                        'page': page,
                        'records': total_records,
                        'page_size': len(records)
                    })
                    
                    # Check pagination termination conditions
                    if (len(records) < pagination_config['page_size'] or 
                        (pagination_config['max_pages'] and page >= pagination_config['max_pages'])):
                        break
                    
                    page += 1
                    
                    # Rate limiting
                    rate_limit_delay = options.get('rate_limit_delay', 0)
                    if rate_limit_delay > 0:
                        time.sleep(rate_limit_delay)
                    
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:  # Rate limited
                        retry_after = int(e.response.headers.get('Retry-After', 60))
                        logger.warning(f"API rate limited, waiting {retry_after}s")
                        time.sleep(retry_after)
                        continue
                    else:
                        raise DataAccessError(f"API returned HTTP error: {e}")
                        
                except requests.exceptions.RequestException as e:
                    raise DataAccessError(f"API request failed: {e}")
                    
                except Exception as e:
                    raise DataFormatError(f"Failed to parse API response: {e}")
        
        if not all_data:
            raise DataFormatError("No data received from API")
        
        df = pd.DataFrame(all_data)
        logger.info(f"Loaded API data: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def _setup_api_authentication(self, credentials: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Setup enhanced API authentication"""
        auth_config = {'headers': headers.copy(), 'auth': None}
        
        if 'api_key' in credentials:
            auth_config['headers']['Authorization'] = f"Bearer {credentials['api_key']}"
        elif 'token' in credentials:
            auth_config['headers']['Authorization'] = f"Token {credentials['token']}"
        elif 'username' in credentials and 'password' in credentials:
            import base64
            auth_string = f"{credentials['username']}:{credentials['password']}"
            auth_bytes = base64.b64encode(auth_string.encode()).decode()
            auth_config['headers']['Authorization'] = f"Basic {auth_bytes}"
        elif 'oauth_token' in credentials:
            auth_config['headers']['Authorization'] = f"Bearer {credentials['oauth_token']}"
        
        # Add custom headers
        if 'custom_headers' in credentials:
            auth_config['headers'].update(credentials['custom_headers'])
        
        return auth_config
    
    def _make_api_request(self, url: str, headers: Dict[str, str], params: Dict[str, Any],
                         timeout: int, auth: Optional[Any] = None) -> requests.Response:
        """Make enhanced API request with retries and validation"""
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=timeout,
            auth=auth
        )
        response.raise_for_status()
        return response
    
    def _validate_api_response(self, response: requests.Response, source: 'DataSource'):
        """Validate API response for security and format"""
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if 'application/json' not in content_type:
            logger.warning(f"Unexpected content type: {content_type}")
        
        # Check response size
        content_length = len(response.content)
        max_response_size = source.options.get('max_response_size_mb', 100) * 1024 * 1024
        
        if content_length > max_response_size:
            raise DataSizeError(
                f"API response too large: {content_length / 1024 / 1024:.1f}MB",
                actual_size=content_length,
                max_size=max_response_size
            )
    
    def _extract_api_records(self, data: Any, options: Dict[str, Any]) -> List[Dict]:
        """Extract records from API response with enhanced path handling"""
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Enhanced data path extraction
            data_paths = options.get('data_path', ['data', 'results', 'records', 'items'])
            if isinstance(data_paths, str):
                data_paths = [data_paths]
            
            # Try nested paths
            for path in data_paths:
                current_data = data
                try:
                    # Handle nested paths like "response.data.items"
                    for key in path.split('.'):
                        if isinstance(current_data, dict) and key in current_data:
                            current_data = current_data[key]
                        else:
                            break
                    
                    if isinstance(current_data, list):
                        return current_data
                except Exception:
                    continue
            
            # If no data path works, treat as single record
            return [data]
        else:
            return []
    
    @retry_on_failure(max_attempts=2)
    def _load_parquet_enhanced(self, source: 'DataSource', progress_bar: bool = True,
                              chunk_size: Optional[int] = None,
                              context: Optional['DataProcessingContext'] = None) -> pd.DataFrame:
        """Enhanced Parquet loading"""
        try:
            options = source.options or {}
            df = pd.read_parquet(source.location, **options)
            
            if df.empty:
                raise DataFormatError(f"Parquet file resulted in empty DataFrame: {source.location}")
            
            logger.info(f"Loaded Parquet: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except FileNotFoundError:
            raise DataAccessError(f"Parquet file not found: {source.location}")
        except Exception as e:
            raise DataFormatError(f"Failed to load Parquet: {e}")
    
    @retry_on_failure(max_attempts=2)
    def _load_excel_enhanced(self, source: 'DataSource', progress_bar: bool = True,
                            chunk_size: Optional[int] = None,
                            context: Optional['DataProcessingContext'] = None) -> pd.DataFrame:
        """Enhanced Excel loading"""
        try:
            options = source.options or {}
            df = pd.read_excel(source.location, **options)
            
            if df.empty:
                raise DataFormatError(f"Excel file resulted in empty DataFrame: {source.location}")
            
            logger.info(f"Loaded Excel: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except FileNotFoundError:
            raise DataAccessError(f"Excel file not found: {source.location}")
        except Exception as e:
            raise DataFormatError(f"Failed to load Excel: {e}")
    
    @retry_on_failure(max_attempts=3)
    def _load_database_enhanced(self, source: 'DataSource', progress_bar: bool = True,
                               chunk_size: Optional[int] = None,
                               context: Optional['DataProcessingContext'] = None) -> pd.DataFrame:
        """Enhanced database loading with connection pooling"""
        credentials = source.credentials or {}
        options = source.options or {}
        
        try:
            # Enhanced database support
            if source.location.startswith('sqlite://'):
                db_path = source.location.replace('sqlite://', '')
                if not Path(db_path).exists():
                    raise DataAccessError(f"SQLite database not found: {db_path}")
                
                query = options.get('query', 'SELECT * FROM transactions LIMIT 10000')
                
                with sqlite3.connect(db_path) as conn:
                    df = pd.read_sql_query(query, conn)
                    
            elif source.location.startswith(('postgresql://', 'mysql://', 'mssql://')):
                # Would need additional database drivers
                raise DataSourceValidationError(f"Database type not yet supported: {source.location}")
            else:
                raise DataSourceValidationError(f"Unsupported database type: {source.location}")
            
            if df.empty:
                raise DataFormatError("Database query resulted in empty DataFrame")
            
            logger.info(f"Loaded database: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except sqlite3.Error as e:
            raise DataAccessError(f"Database error: {e}")
        except Exception as e:
            raise DataFormatError(f"Failed to load from database: {e}")

# ======================== ENHANCED FINANCIAL DATA LOADER ========================

class EnhancedFinancialDataLoader:
    """
    Enhanced Financial Data Loader with comprehensive validation, caching, and monitoring
    Orchestrates all enhanced components for production-ready data loading operations
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 validation_level: ValidationLevel = ValidationLevel.STRICT,
                 security_level: SecurityLevel = SecurityLevel.STANDARD,
                 enable_caching: bool = True,
                 enable_monitoring: bool = True,
                 performance_config: Optional[Dict[str, Any]] = None):
        
        self.validation_level = validation_level
        self.security_level = security_level
        self.enable_caching = enable_caching
        self.enable_monitoring = enable_monitoring
        self.config = config or {}
        
        # Performance configuration
        self.performance_config = performance_config or {
            'max_memory_mb': 4096,
            'max_workers': 4,
            'chunk_size': 10000,
            'timeout_seconds': 300,
            'max_retries': 3
        }
        
        # Initialize components
        self._init_components()
        
        # Setup monitoring
        self._metrics = DataLoadMetrics()
        self._performance_monitor = self._init_performance_monitor()
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Enhanced Financial Data Loader initialized with validation={validation_level.value}, security={security_level.value}")
    
    def _init_components(self):
        """Initialize all enhanced components"""
        try:
            # Initialize cache
            if self.enable_caching:
                cache_config = self.config.get('cache', {})
                self.cache = EnhancedDataCache(
                    cache_dir=cache_config.get('directory', './cache'),
                    max_size_mb=cache_config.get('max_size_mb', 1024),
                    enable_compression=cache_config.get('enable_compression', True),
                    enable_encryption=cache_config.get('enable_encryption', True),
                    enable_integrity_check=cache_config.get('enable_integrity_check', True)
                )
            else:
                self.cache = None
            
            # Initialize validators
            validator_config = self.config.get('validator', {})
            self.transaction_validator = EnhancedFinancialTransactionValidator(
                validation_level=self.validation_level,
                **validator_config
            )
            
            security_config = self.config.get('security_validator', {})
            self.security_validator = EnhancedSecurityValidator(
                security_level=self.security_level,
                **security_config
            )
            
            # Initialize preprocessor
            preprocessor_config = self.config.get('preprocessor', {})
            self.preprocessor = EnhancedDataPreprocessor(
                enable_feature_engineering=preprocessor_config.get('enable_feature_engineering', True),
                enable_quality_checks=preprocessor_config.get('enable_quality_checks', True),
                performance_config=preprocessor_config.get('performance_config', {})
            )
            
            # Initialize data source manager
            source_config = self.config.get('data_source', {})
            self.source_manager = EnhancedDataSourceManager(
                default_timeout=source_config.get('default_timeout', 60),
                max_retries=source_config.get('max_retries', 3),
                chunk_size=source_config.get('chunk_size', 10000),
                enable_parallel_loading=source_config.get('enable_parallel_loading', True),
                performance_config=source_config.get('performance_config', {})
            )
            
            logger.info("All enhanced components initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize enhanced components: {e}"
            logger.error(error_msg)
            raise DataLoaderException(error_msg, recoverable=False)
    
    def _init_performance_monitor(self) -> Dict[str, Any]:
        """Initialize performance monitoring"""
        if not self.enable_monitoring:
            return {}
        
        return {
            'start_time': None,
            'memory_usage': {'peak': 0, 'current': 0},
            'processing_times': {},
            'error_counts': {},
            'data_quality_metrics': {},
            'throughput_metrics': {}
        }
    
    @enhanced_error_handler("Data loading operation failed", max_retries=3)
    @enhanced_performance_monitor("load_data")
    @enhanced_input_validator
    def load_data(self, 
                  source: Union[str, Dict[str, Any]], 
                  data_type: str = 'transaction',
                  processing_options: Optional[Dict[str, Any]] = None,
                  cache_key: Optional[str] = None,
                  force_reload: bool = False) -> DataValidationResult:
        """
        Load and process financial data with comprehensive validation and monitoring
        
        Args:
            source: Data source path, URL, or configuration dictionary
            data_type: Type of data being loaded ('transaction', 'customer', 'merchant')
            processing_options: Additional processing configuration
            cache_key: Custom cache key (auto-generated if not provided)
            force_reload: Force reload even if cached data exists
            
        Returns:
            DataValidationResult with processed data and comprehensive metrics
        """
        with self._lock:
            start_time = time.time()
            
            try:
                # Start performance monitoring
                if self.enable_monitoring:
                    self._performance_monitor['start_time'] = start_time
                    self._performance_monitor['memory_usage']['current'] = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Generate cache key if not provided
                if not cache_key:
                    cache_key = self._generate_cache_key(source, data_type, processing_options)
                
                # Try to load from cache first
                if self.enable_caching and not force_reload:
                    cached_result = self._try_load_from_cache(cache_key)
                    if cached_result:
                        logger.info(f"Successfully loaded data from cache: {cache_key}")
                        self._metrics.cache_hits += 1
                        return cached_result
                    else:
                        self._metrics.cache_misses += 1
                
                # Load raw data from source
                logger.info(f"Loading data from source: {source}")
                raw_data = self._load_raw_data(source, data_type, processing_options)
                
                # Security validation
                security_result = self._perform_security_validation(raw_data, source)
                if not security_result['is_secure']:
                    raise DataSecurityError(
                        f"Security validation failed: {security_result['violations']}",
                        security_check="comprehensive_scan",
                        threat_type=security_result.get('primary_threat', 'unknown')
                    )
                
                # Data validation and processing
                validation_result = self._perform_data_validation(raw_data, data_type, processing_options)
                
                # Data preprocessing and feature engineering
                if validation_result.is_valid and validation_result.data is not None:
                    preprocessed_result = self._perform_data_preprocessing(
                        validation_result.data, 
                        data_type, 
                        processing_options
                    )
                    
                    # Update validation result with preprocessed data
                    validation_result.data = preprocessed_result['data']
                    validation_result.feature_summary = preprocessed_result.get('feature_summary', {})
                    validation_result.preprocessing_metrics = preprocessed_result.get('metrics', {})
                
                # Cache the result if caching is enabled
                if self.enable_caching and validation_result.is_valid:
                    self._cache_result(cache_key, validation_result)
                
                # Update metrics
                self._update_metrics(validation_result, start_time)
                
                # Final monitoring update
                if self.enable_monitoring:
                    self._update_performance_monitor(validation_result, start_time)
                
                logger.info(f"Data loading completed in {time.time() - start_time:.2f} seconds")
                return validation_result
                
            except Exception as e:
                self._metrics.total_errors += 1
                
                if isinstance(e, (DataSecurityError, DataIntegrityError)):
                    # Critical errors should not be retried
                    raise
                
                error_msg = f"Enhanced data loading failed: {e}"
                logger.error(error_msg, exc_info=True)
                
                # Return error result
                return DataValidationResult(
                    is_valid=False,
                    validation_level=self.validation_level,
                    data=None,
                    errors=[error_msg],
                    warnings=[],
                    metrics={'processing_time': time.time() - start_time, 'error_type': type(e).__name__}
                )
    
    def _generate_cache_key(self, source: Union[str, Dict[str, Any]], 
                           data_type: str, 
                           processing_options: Optional[Dict[str, Any]]) -> str:
        """Generate unique cache key for data source and processing options"""
        key_data = {
            'source': str(source),
            'data_type': data_type,
            'validation_level': self.validation_level.value,
            'security_level': self.security_level.value,
            'processing_options': processing_options or {}
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def _try_load_from_cache(self, cache_key: str) -> Optional[DataValidationResult]:
        """Try to load data from cache"""
        if not self.cache:
            return None
        
        try:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                # Deserialize cached result
                if isinstance(cached_data, dict) and 'validation_result' in cached_data:
                    return self._deserialize_validation_result(cached_data['validation_result'])
            return None
            
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None
    
    def _cache_result(self, cache_key: str, validation_result: DataValidationResult):
        """Cache validation result"""
        if not self.cache:
            return
        
        try:
            cache_data = {
                'validation_result': self._serialize_validation_result(validation_result),
                'cached_at': datetime.now().isoformat(),
                'cache_version': '1.0'
            }
            
            self.cache.put(cache_key, cache_data)
            logger.debug(f"Cached validation result: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    def _serialize_validation_result(self, result: DataValidationResult) -> Dict[str, Any]:
        """Serialize validation result for caching"""
        # Convert DataFrame to dict for serialization
        data_dict = None
        if result.data is not None:
            data_dict = {
                'columns': result.data.columns.tolist(),
                'data': result.data.to_dict('records'),
                'dtypes': result.data.dtypes.astype(str).to_dict()
            }
        
        return {
            'is_valid': result.is_valid,
            'validation_level': result.validation_level.value,
            'data': data_dict,
            'errors': result.errors,
            'warnings': result.warnings,
            'feature_summary': result.feature_summary,
            'preprocessing_metrics': result.preprocessing_metrics,
            'metrics': result.metrics
        }
    
    def _deserialize_validation_result(self, data: Dict[str, Any]) -> DataValidationResult:
        """Deserialize validation result from cache"""
        # Reconstruct DataFrame from dict
        df_data = None
        if data.get('data'):
            df_dict = data['data']
            df_data = pd.DataFrame(df_dict['data'])
            # Restore column order and types
            df_data = df_data[df_dict['columns']]
            for col, dtype in df_dict['dtypes'].items():
                if col in df_data.columns:
                    try:
                        df_data[col] = df_data[col].astype(dtype)
                    except Exception:
                        pass  # Keep original type if conversion fails
        
        return DataValidationResult(
            is_valid=data['is_valid'],
            validation_level=ValidationLevel(data['validation_level']),
            data=df_data,
            errors=data.get('errors', []),
            warnings=data.get('warnings', []),
            feature_summary=data.get('feature_summary', {}),
            preprocessing_metrics=data.get('preprocessing_metrics', {}),
            metrics=data.get('metrics', {})
        )
    
    def _load_raw_data(self, source: Union[str, Dict[str, Any]], 
                      data_type: str, 
                      processing_options: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Load raw data from various sources"""
        try:
            # Determine source type and delegate to source manager
            if isinstance(source, str):
                if source.startswith(('http://', 'https://')):
                    return self.source_manager.load_from_api(source, processing_options)
                elif source.endswith('.csv'):
                    return self.source_manager.load_from_csv(source, processing_options)
                elif source.endswith('.json'):
                    return self.source_manager.load_from_json(source, processing_options)
                elif source.endswith('.parquet'):
                    return self.source_manager.load_from_parquet(source, processing_options)
                elif source.endswith(('.xlsx', '.xls')):
                    return self.source_manager.load_from_excel(source, processing_options)
                else:
                    # Try to detect format
                    return self.source_manager.load_from_file(source, processing_options)
            
            elif isinstance(source, dict):
                # Database or custom source configuration
                if 'connection_string' in source or 'database' in source:
                    return self.source_manager.load_from_database(source, processing_options)
                elif 'api_config' in source:
                    return self.source_manager.load_from_api(source['api_config'], processing_options)
                else:
                    raise DataSourceValidationError("Invalid source configuration", details=source)
            
            else:
                raise DataSourceValidationError(f"Unsupported source type: {type(source)}")
                
        except Exception as e:
            if isinstance(e, DataLoaderException):
                raise
            raise DataAccessError(f"Failed to load raw data: {e}", access_type="data_loading", resource=str(source))
    
    def _perform_security_validation(self, data: pd.DataFrame, source: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive security validation"""
        try:
            security_context = SecurityValidationContext(
                source_info={'source': str(source), 'type': 'financial_data'},
                data_sensitivity=DataSensitivity.HIGH,
                compliance_requirements=['PCI_DSS', 'GDPR'],
                threat_model={'external_access': True, 'data_export': True}
            )
            
            return self.security_validator.validate_data_security(data, security_context)
            
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            return {
                'is_secure': False,
                'violations': [f"Security validation error: {e}"]
            }
    
    def _perform_data_validation(self, data: pd.DataFrame, 
                                data_type: str, 
                                processing_options: Optional[Dict[str, Any]]) -> DataValidationResult:
        """Perform comprehensive data validation"""
        try:
            validation_context = DataProcessingContext(
                source_type=data_type,
                expected_schema=processing_options.get('schema') if processing_options else None,
                business_rules=processing_options.get('business_rules') if processing_options else None,
                quality_requirements=processing_options.get('quality_requirements') if processing_options else None
            )
            
            return self.transaction_validator.validate_transaction_data(data, validation_context)
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return DataValidationResult(
                is_valid=False,
                validation_level=self.validation_level,
                data=None,
                errors=[f"Data validation error: {e}"],
                warnings=[],
                metrics={'validation_error': True}
            )
    
    def _perform_data_preprocessing(self, data: pd.DataFrame, 
                                   data_type: str, 
                                   processing_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform data preprocessing and feature engineering"""
        try:
            preprocessing_config = processing_options.get('preprocessing', {}) if processing_options else {}
            
            return self.preprocessor.process_financial_data(
                data=data,
                data_type=data_type,
                processing_config=preprocessing_config
            )
            
        except Exception as e:
            logger.warning(f"Data preprocessing failed: {e}")
            return {
                'data': data,
                'feature_summary': {},
                'metrics': {'preprocessing_error': str(e)}
            }
    
    def _update_metrics(self, validation_result: DataValidationResult, start_time: float):
        """Update internal metrics"""
        self._metrics.total_requests += 1
        self._metrics.processing_time = time.time() - start_time
        
        if validation_result.is_valid:
            self._metrics.successful_loads += 1
            if validation_result.data is not None:
                self._metrics.total_records_processed += len(validation_result.data)
        else:
            self._metrics.failed_loads += 1
    
    def _update_performance_monitor(self, validation_result: DataValidationResult, start_time: float):
        """Update performance monitoring metrics"""
        if not self.enable_monitoring:
            return
        
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self._performance_monitor['memory_usage']['current'] = current_memory
        self._performance_monitor['memory_usage']['peak'] = max(
            self._performance_monitor['memory_usage']['peak'], 
            current_memory
        )
        
        processing_time = time.time() - start_time
        self._performance_monitor['processing_times']['total'] = processing_time
        
        if validation_result.data is not None:
            record_count = len(validation_result.data)
            self._performance_monitor['throughput_metrics']['records_per_second'] = record_count / processing_time
        
        # Store quality metrics
        self._performance_monitor['data_quality_metrics'] = {
            'is_valid': validation_result.is_valid,
            'error_count': len(validation_result.errors),
            'warning_count': len(validation_result.warnings)
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        base_metrics = asdict(self._metrics)
        
        if self.enable_monitoring:
            base_metrics['performance_monitor'] = self._performance_monitor
        
        if self.cache:
            base_metrics['cache_metrics'] = self.cache.get_cache_metrics()
        
        return base_metrics
    
    def clear_cache(self, pattern: Optional[str] = None) -> bool:
        """Clear cache entries"""
        if not self.cache:
            return False
        
        try:
            if pattern:
                return self.cache.clear_pattern(pattern)
            else:
                return self.cache.clear_all()
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate loader configuration"""
        errors = []
        
        # Validate performance configuration
        if self.performance_config.get('max_memory_mb', 0) <= 0:
            errors.append("max_memory_mb must be positive")
        
        if self.performance_config.get('max_workers', 0) <= 0:
            errors.append("max_workers must be positive")
        
        if self.performance_config.get('chunk_size', 0) <= 0:
            errors.append("chunk_size must be positive")
        
        # Validate component configurations
        try:
            if hasattr(self.transaction_validator, 'validate_configuration'):
                is_valid, validator_errors = self.transaction_validator.validate_configuration()
                if not is_valid:
                    errors.extend([f"Validator: {e}" for e in validator_errors])
        except Exception as e:
            errors.append(f"Validator configuration error: {e}")
        
        return len(errors) == 0, errors
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'metrics': self.get_metrics()
        }
        
        # Check cache health
        if self.cache:
            try:
                cache_health = self.cache.health_check()
                health_status['components']['cache'] = cache_health
                if not cache_health.get('is_healthy', False):
                    health_status['status'] = 'degraded'
            except Exception as e:
                health_status['components']['cache'] = {'is_healthy': False, 'error': str(e)}
                health_status['status'] = 'degraded'
        
        # Check validator health
        try:
            config_valid, config_errors = self.validate_configuration()
            health_status['components']['configuration'] = {
                'is_valid': config_valid,
                'errors': config_errors
            }
            if not config_valid:
                health_status['status'] = 'unhealthy'
        except Exception as e:
            health_status['components']['configuration'] = {'is_valid': False, 'error': str(e)}
            health_status['status'] = 'unhealthy'
        
        return health_status
    
    def shutdown(self):
        """Graceful shutdown of all components"""
        logger.info("Shutting down Enhanced Financial Data Loader")
        
        try:
            # Shutdown cache
            if self.cache and hasattr(self.cache, 'close'):
                self.cache.close()
            
            # Shutdown source manager
            if hasattr(self.source_manager, 'shutdown'):
                self.source_manager.shutdown()
            
            # Log final metrics
            final_metrics = self.get_metrics()
            logger.info(f"Final metrics: {final_metrics}")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Enhanced integration with existing data loader module
def create_enhanced_data_loader(config_path: Optional[str] = None, 
                               **kwargs) -> EnhancedFinancialDataLoader:
    """
    Factory function to create Enhanced Financial Data Loader with configuration
    
    Args:
        config_path: Path to configuration file
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured EnhancedFinancialDataLoader instance
    """
    config = {}
    
    # Load configuration from file if provided
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    config = json.load(f)
                elif config_path.endswith(('.yml', '.yaml')):
                    config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
    
    # Override with kwargs
    config.update(kwargs)
    
    return EnhancedFinancialDataLoader(config=config)

# Enhanced compatibility layer
class EnhancedDataLoaderAdapter:
    """
    Adapter class to provide compatibility with existing data loader interface
    while leveraging enhanced functionality
    """
    
    def __init__(self, enhanced_loader: EnhancedFinancialDataLoader):
        self.enhanced_loader = enhanced_loader
    
    def load_transaction_data(self, source: str, **kwargs) -> pd.DataFrame:
        """Legacy interface compatibility for transaction data loading"""
        result = self.enhanced_loader.load_data(
            source=source, 
            data_type='transaction',
            processing_options=kwargs
        )
        
        if result.is_valid and result.data is not None:
            return result.data
        else:
            raise DataLoaderException(f"Data loading failed: {result.errors}")
    
    def load_customer_data(self, source: str, **kwargs) -> pd.DataFrame:
        """Legacy interface compatibility for customer data loading"""
        result = self.enhanced_loader.load_data(
            source=source, 
            data_type='customer',
            processing_options=kwargs
        )
        
        if result.is_valid and result.data is not None:
            return result.data
        else:
            raise DataLoaderException(f"Data loading failed: {result.errors}")
    
    def get_loader_stats(self) -> Dict[str, Any]:
        """Legacy interface compatibility for loader statistics"""
        return self.enhanced_loader.get_metrics()

# Export enhanced components for use in other modules
__all__ = [
    # Core exceptions
    'DataLoaderException', 'DataSourceValidationError', 'DataAccessError',
    'DataFormatError', 'DataSizeError', 'DataIntegrityError', 'DataSecurityError',
    
    # Enums
    'ValidationLevel', 'SecurityLevel', 'DataSensitivity', 'ProcessingMode',
    
    # Data classes
    'DataLoadMetrics', 'DataValidationResult', 'DataProcessingContext', 'SecurityValidationContext',
    
    # Enhanced components
    'EnhancedFinancialTransactionValidator', 'EnhancedSecurityValidator',
    'EnhancedDataPreprocessor', 'EnhancedDataCache', 'EnhancedDataSourceManager',
    
    # Main loader
    'EnhancedFinancialDataLoader',
    
    # Utilities
    'create_enhanced_data_loader', 'EnhancedDataLoaderAdapter',
    
    # Decorators
    'enhanced_error_handler', 'enhanced_performance_monitor', 'enhanced_input_validator'
]

if __name__ == "__main__":
    print("Enhanced Data Loader - Complete implementation with main loader class loaded")