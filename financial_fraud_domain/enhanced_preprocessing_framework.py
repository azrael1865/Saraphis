"""
Enhanced Data Preprocessor - Chunk 1: Core Exceptions and Validation Framework
Comprehensive preprocessing system with enhanced error handling, performance monitoring,
security validation, and integration testing for financial transaction data.
"""

import logging
import pandas as pd
import numpy as np
import re
import json
import time
import psutil
import warnings
import traceback
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set, Callable, Pattern, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from functools import wraps
import hashlib
import signal

# Import existing components for enhancement
try:
    from data_validator import (
        ValidationLevel, ValidationSeverity, ComplianceStandard,
        ValidationIssue, ValidationResult, ValidationRule
    )
    from enhanced_validator_framework import (
        ValidationException, enhanced_input_validator, enhanced_error_handler,
        enhanced_performance_monitor, ValidationMetricsCollector
    )
    EXISTING_COMPONENTS = True
except ImportError as e:
    EXISTING_COMPONENTS = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import existing components: {e}")

# Configure logging
logger = logging.getLogger(__name__)

# ======================== ENHANCED EXCEPTION HIERARCHY ========================

class PreprocessingException(ValidationException if EXISTING_COMPONENTS else Exception):
    """Base exception for preprocessing errors with enhanced context"""
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        if EXISTING_COMPONENTS:
            super().__init__(message, details=details)
        else:
            super().__init__(message)
            self.details = details or {}
            self.timestamp = datetime.now()
        
        self.error_code = error_code or "PREPROCESSING_ERROR"
        self.preprocessing_context = {
            'error_code': self.error_code,
            'timestamp': self.timestamp.isoformat() if hasattr(self, 'timestamp') else datetime.now().isoformat(),
            'recoverable': getattr(self, 'recoverable', True)
        }

class PreprocessingConfigError(PreprocessingException):
    """Raised when preprocessing configuration is invalid"""
    def __init__(self, message: str, config_field: str = None, **kwargs):
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)
        self.config_field = config_field

class PreprocessingInputError(PreprocessingException):
    """Raised when input data is invalid or incompatible"""
    def __init__(self, message: str, input_type: str = None, **kwargs):
        super().__init__(message, error_code="INPUT_ERROR", **kwargs)
        self.input_type = input_type

class PreprocessingTimeoutError(PreprocessingException):
    """Raised when preprocessing operation times out"""
    def __init__(self, message: str, operation: str = None, timeout_seconds: float = None, **kwargs):
        super().__init__(message, error_code="TIMEOUT_ERROR", **kwargs)
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        self.recoverable = False

class PreprocessingMemoryError(PreprocessingException):
    """Raised when preprocessing exceeds memory limits"""
    def __init__(self, message: str, memory_used_mb: float = None, memory_limit_mb: float = None, **kwargs):
        super().__init__(message, error_code="MEMORY_ERROR", **kwargs)
        self.memory_used_mb = memory_used_mb
        self.memory_limit_mb = memory_limit_mb
        self.recoverable = False

class PreprocessingSecurityError(PreprocessingException):
    """Raised when preprocessing violates security constraints"""
    def __init__(self, message: str, security_violation: str = None, **kwargs):
        super().__init__(message, error_code="SECURITY_ERROR", **kwargs)
        self.security_violation = security_violation
        self.recoverable = False

class PreprocessingIntegrationError(PreprocessingException):
    """Raised when preprocessing fails integration checks"""
    def __init__(self, message: str, integration_point: str = None, **kwargs):
        super().__init__(message, error_code="INTEGRATION_ERROR", **kwargs)
        self.integration_point = integration_point

class FeatureEngineeringError(PreprocessingException):
    """Raised when feature engineering fails"""
    def __init__(self, message: str, feature_type: str = None, **kwargs):
        super().__init__(message, error_code="FEATURE_ENGINEERING_ERROR", **kwargs)
        self.feature_type = feature_type

class EncodingError(PreprocessingException):
    """Raised when categorical encoding fails"""
    def __init__(self, message: str, encoding_method: str = None, column: str = None, **kwargs):
        super().__init__(message, error_code="ENCODING_ERROR", **kwargs)
        self.encoding_method = encoding_method
        self.column = column

class ScalingError(PreprocessingException):
    """Raised when feature scaling fails"""
    def __init__(self, message: str, scaling_method: str = None, column: str = None, **kwargs):
        super().__init__(message, error_code="SCALING_ERROR", **kwargs)
        self.scaling_method = scaling_method
        self.column = column

class ImputationError(PreprocessingException):
    """Raised when data imputation fails"""
    def __init__(self, message: str, imputation_method: str = None, column: str = None, **kwargs):
        super().__init__(message, error_code="IMPUTATION_ERROR", **kwargs)
        self.imputation_method = imputation_method
        self.column = column

class PartialPreprocessingError(PreprocessingException):
    """Raised when preprocessing partially fails but can continue"""
    def __init__(self, message: str, successful_steps: List[str], 
                 failed_steps: List[str], partial_data: pd.DataFrame = None, **kwargs):
        super().__init__(message, error_code="PARTIAL_PREPROCESSING_ERROR", **kwargs)
        self.successful_steps = successful_steps
        self.failed_steps = failed_steps
        self.partial_data = partial_data
        self.recoverable = True

# ======================== ENHANCED CONFIGURATION CLASSES ========================

@dataclass
class SecurityConfig:
    """Security configuration for preprocessing operations"""
    max_memory_mb: float = 2048.0
    max_execution_time_seconds: float = 3600.0
    allowed_data_types: Set[str] = field(default_factory=lambda: {
        'int64', 'float64', 'object', 'datetime64', 'bool', 'category'
    })
    sanitize_column_names: bool = True
    check_data_leakage: bool = True
    protect_sensitive_features: bool = True
    sensitive_patterns: Set[str] = field(default_factory=lambda: {
        'ssn', 'pin', 'password', 'cvv', 'secret', 'token', 'key'
    })
    max_column_count: int = 10000
    max_row_count: int = 10000000
    enable_column_validation: bool = True

@dataclass
class ValidationConfig:
    """Configuration for preprocessing validation"""
    validate_inputs: bool = True
    validate_outputs: bool = True
    validate_pipeline_steps: bool = True
    check_data_consistency: bool = True
    check_feature_ranges: bool = True
    check_missing_threshold: float = 0.95
    check_cardinality_threshold: int = 1000
    check_correlation_threshold: float = 0.95
    check_data_quality: bool = True
    enable_profiling: bool = True
    enable_statistical_validation: bool = True
    validate_feature_types: bool = True

@dataclass
class PerformanceConfig:
    """Performance configuration for preprocessing operations"""
    enable_monitoring: bool = True
    enable_profiling: bool = True
    profile_memory: bool = True
    profile_cpu: bool = True
    warning_memory_threshold_mb: float = 1024.0
    warning_time_threshold_seconds: float = 300.0
    enable_parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 10000

# ======================== ENHANCED VALIDATION CLASSES ========================

class InputValidator:
    """Comprehensive input validation for preprocessing operations"""
    
    def __init__(self, validation_config: ValidationConfig):
        self.config = validation_config
        self.validation_history = []
        self._lock = threading.RLock()
    
    def validate_dataframe(self, data: pd.DataFrame, context: str = "input") -> Tuple[bool, List[str]]:
        """
        Validate DataFrame structure and content with comprehensive checks
        
        Args:
            data: DataFrame to validate
            context: Context for validation (e.g., "fit_input", "transform_input")
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        with self._lock:
            validation_start = time.time()
            errors = []
            
            try:
                # Basic structure validation
                if data is None:
                    errors.append(f"{context}: DataFrame is None")
                    return False, errors
                
                if not isinstance(data, pd.DataFrame):
                    errors.append(f"{context}: Input is not a DataFrame (type: {type(data)})")
                    return False, errors
                
                # Check if DataFrame is empty
                if data.empty:
                    errors.append(f"{context}: DataFrame is empty")
                    return False, errors
                
                # Check DataFrame shape
                if data.shape[0] == 0:
                    errors.append(f"{context}: DataFrame has 0 rows")
                    return False, errors
                
                if data.shape[1] == 0:
                    errors.append(f"{context}: DataFrame has 0 columns")
                    return False, errors
                
                # Check for reasonable size limits
                if data.shape[0] > 10000000:  # 10M rows
                    errors.append(f"{context}: DataFrame too large ({data.shape[0]} rows)")
                
                if data.shape[1] > 10000:  # 10K columns
                    errors.append(f"{context}: Too many columns ({data.shape[1]} columns)")
                
                # Check for excessive missing data
                if self.config.check_missing_threshold:
                    try:
                        missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
                        if missing_ratio > self.config.check_missing_threshold:
                            errors.append(f"{context}: Excessive missing data ({missing_ratio:.2%})")
                    except Exception as e:
                        errors.append(f"{context}: Error calculating missing data ratio: {e}")
                
                # Check data types
                if self.config.validate_feature_types:
                    for col in data.columns:
                        try:
                            dtype = str(data[col].dtype)
                            if dtype == 'object':
                                # Check for mixed types in object columns
                                sample_data = data[col].dropna().head(1000)
                                if len(sample_data) > 0:
                                    unique_types = sample_data.apply(type).unique()
                                    if len(unique_types) > 1:
                                        type_names = [t.__name__ for t in unique_types]
                                        errors.append(f"{context}: Mixed types in column '{col}': {type_names}")
                        except Exception as e:
                            errors.append(f"{context}: Error checking column '{col}': {str(e)}")
                
                # Check for duplicate columns
                duplicate_cols = data.columns[data.columns.duplicated()].tolist()
                if duplicate_cols:
                    errors.append(f"{context}: Duplicate columns found: {duplicate_cols}")
                
                # Check for invalid column names
                invalid_cols = []
                for col in data.columns:
                    if not isinstance(col, str):
                        invalid_cols.append(f"Non-string column name: {col}")
                    elif len(col.strip()) == 0:
                        invalid_cols.append("Empty column name")
                    elif len(col) > 255:
                        invalid_cols.append(f"Column name too long: {col[:50]}...")
                
                if invalid_cols:
                    errors.append(f"{context}: Invalid column names: {invalid_cols}")
                
                # Check for data consistency
                if self.config.check_data_consistency:
                    try:
                        # Check for completely null columns
                        null_columns = data.columns[data.isnull().all()].tolist()
                        if null_columns:
                            errors.append(f"{context}: Completely null columns: {null_columns}")
                        
                        # Check for constant columns
                        constant_columns = []
                        for col in data.select_dtypes(include=[np.number]).columns:
                            if data[col].nunique() == 1:
                                constant_columns.append(col)
                        
                        if constant_columns and len(constant_columns) > len(data.columns) * 0.1:
                            errors.append(f"{context}: Many constant columns ({len(constant_columns)}): {constant_columns[:5]}")
                    
                    except Exception as e:
                        errors.append(f"{context}: Error in consistency check: {e}")
                
                # Record validation
                validation_time = time.time() - validation_start
                self.validation_history.append({
                    'context': context,
                    'timestamp': datetime.now(),
                    'is_valid': len(errors) == 0,
                    'error_count': len(errors),
                    'validation_time': validation_time,
                    'data_shape': data.shape
                })
                
                return len(errors) == 0, errors
                
            except Exception as e:
                errors.append(f"{context}: Unexpected error during validation: {str(e)}")
                return False, errors
    
    def validate_preprocessing_config(self, config: 'PreprocessingConfig') -> Tuple[bool, List[str]]:
        """
        Validate preprocessing configuration with comprehensive checks
        
        Args:
            config: Preprocessing configuration to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Validate missing strategy
            valid_missing_strategies = ['drop', 'fill', 'interpolate', 'forward_fill', 'backward_fill']
            if hasattr(config, 'missing_strategy') and config.missing_strategy not in valid_missing_strategies:
                errors.append(f"Invalid missing_strategy: {config.missing_strategy}. Must be one of: {valid_missing_strategies}")
            
            # Validate outlier method
            valid_outlier_methods = ['iqr', 'zscore', 'isolation_forest', 'percentile']
            if hasattr(config, 'outlier_method') and config.outlier_method not in valid_outlier_methods:
                errors.append(f"Invalid outlier_method: {config.outlier_method}. Must be one of: {valid_outlier_methods}")
            
            # Validate categorical encoding
            valid_encodings = ['onehot', 'label', 'target', 'frequency', 'binary']
            if hasattr(config, 'categorical_encoding') and config.categorical_encoding not in valid_encodings:
                errors.append(f"Invalid categorical_encoding: {config.categorical_encoding}. Must be one of: {valid_encodings}")
            
            # Validate feature selection method
            valid_selection_methods = ['mutual_info', 'f_classif', 'rfe', 'variance_threshold']
            if hasattr(config, 'selection_method') and config.selection_method not in valid_selection_methods:
                errors.append(f"Invalid selection_method: {config.selection_method}. Must be one of: {valid_selection_methods}")
            
            # Validate numeric parameters
            if hasattr(config, 'max_categories') and config.max_categories <= 0:
                errors.append(f"max_categories must be positive: {config.max_categories}")
            
            if hasattr(config, 'polynomial_degree') and config.polynomial_degree < 1:
                errors.append(f"polynomial_degree must be >= 1: {config.polynomial_degree}")
            
            if hasattr(config, 'lag_periods') and config.lag_periods:
                if any(p <= 0 for p in config.lag_periods):
                    errors.append("All lag_periods must be positive")
            
            # Validate scaling method
            if hasattr(config, 'scaling_method'):
                valid_scaling_methods = ['standard', 'minmax', 'robust', 'normalize']
                scaling_method_value = config.scaling_method.value if hasattr(config.scaling_method, 'value') else config.scaling_method
                if scaling_method_value not in valid_scaling_methods:
                    errors.append(f"Invalid scaling_method: {scaling_method_value}. Must be one of: {valid_scaling_methods}")
            
            # Validate thresholds
            if hasattr(config, 'validation_config') and config.validation_config:
                val_config = config.validation_config
                
                if hasattr(val_config, 'check_missing_threshold'):
                    if not (0 <= val_config.check_missing_threshold <= 1):
                        errors.append(f"check_missing_threshold must be between 0 and 1: {val_config.check_missing_threshold}")
                
                if hasattr(val_config, 'check_correlation_threshold'):
                    if not (0 <= val_config.check_correlation_threshold <= 1):
                        errors.append(f"check_correlation_threshold must be between 0 and 1: {val_config.check_correlation_threshold}")
                
                if hasattr(val_config, 'check_cardinality_threshold'):
                    if val_config.check_cardinality_threshold <= 0:
                        errors.append(f"check_cardinality_threshold must be positive: {val_config.check_cardinality_threshold}")
            
            # Validate security config
            if hasattr(config, 'security_config') and config.security_config:
                sec_config = config.security_config
                
                if hasattr(sec_config, 'max_memory_mb') and sec_config.max_memory_mb <= 0:
                    errors.append(f"max_memory_mb must be positive: {sec_config.max_memory_mb}")
                
                if hasattr(sec_config, 'max_execution_time_seconds') and sec_config.max_execution_time_seconds <= 0:
                    errors.append(f"max_execution_time_seconds must be positive: {sec_config.max_execution_time_seconds}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Unexpected error during config validation: {str(e)}")
            return False, errors
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation history"""
        with self._lock:
            if not self.validation_history:
                return {"total_validations": 0}
            
            total_validations = len(self.validation_history)
            successful_validations = sum(1 for v in self.validation_history if v['is_valid'])
            
            return {
                "total_validations": total_validations,
                "successful_validations": successful_validations,
                "failed_validations": total_validations - successful_validations,
                "success_rate": successful_validations / total_validations if total_validations > 0 else 0,
                "average_validation_time": sum(v['validation_time'] for v in self.validation_history) / total_validations,
                "contexts": list(set(v['context'] for v in self.validation_history))
            }

class PerformanceMonitor:
    """Comprehensive performance monitoring for preprocessing operations"""
    
    def __init__(self, max_memory_mb: float = 2048.0, max_time_seconds: float = 3600.0):
        self.max_memory_mb = max_memory_mb
        self.max_time_seconds = max_time_seconds
        self.start_time = None
        self.start_memory = None
        self.operation_history = []
        self._stop_flag = threading.Event()
        self._lock = threading.RLock()
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """
        Context manager for monitoring operations with comprehensive metrics
        
        Args:
            operation_name: Name of the operation being monitored
            
        Yields:
            Operation context
        """
        operation_start = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_usage()
        
        # Start monitoring thread
        self._stop_flag.clear()
        monitor_thread = threading.Thread(
            target=self._monitor_resources, 
            args=(operation_name, operation_start),
            daemon=True
        )
        monitor_thread.start()
        
        try:
            yield
            
        finally:
            # Stop monitoring
            self._stop_flag.set()
            
            # Calculate final metrics
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = self._get_cpu_usage()
            
            elapsed_time = end_time - operation_start
            memory_delta = end_memory - start_memory
            avg_cpu = (start_cpu + end_cpu) / 2
            
            # Record operation
            with self._lock:
                self.operation_history.append({
                    'operation': operation_name,
                    'start_time': operation_start,
                    'end_time': end_time,
                    'duration': elapsed_time,
                    'start_memory_mb': start_memory,
                    'end_memory_mb': end_memory,
                    'memory_delta_mb': memory_delta,
                    'avg_cpu_percent': avg_cpu,
                    'timestamp': datetime.now()
                })
            
            logger.info(
                f"Performance: {operation_name} completed in {elapsed_time:.2f}s, "
                f"memory delta: {memory_delta:+.2f}MB, avg CPU: {avg_cpu:.1f}%"
            )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0
    
    def _monitor_resources(self, operation_name: str, start_time: float):
        """Monitor resources in background thread with enhanced checks"""
        peak_memory = 0.0
        peak_cpu = 0.0
        
        while not self._stop_flag.is_set():
            try:
                elapsed_time = time.time() - start_time
                current_memory = self._get_memory_usage()
                current_cpu = self._get_cpu_usage()
                
                # Track peaks
                peak_memory = max(peak_memory, current_memory)
                peak_cpu = max(peak_cpu, current_cpu)
                
                # Check time limit
                if elapsed_time > self.max_time_seconds:
                    logger.error(f"{operation_name} exceeded time limit ({self.max_time_seconds}s)")
                    raise PreprocessingTimeoutError(
                        f"{operation_name} exceeded time limit",
                        operation=operation_name,
                        timeout_seconds=self.max_time_seconds,
                        details={
                            "elapsed_time": elapsed_time,
                            "limit": self.max_time_seconds,
                            "peak_memory_mb": peak_memory,
                            "peak_cpu_percent": peak_cpu
                        }
                    )
                
                # Check memory limit
                if current_memory > self.max_memory_mb:
                    logger.error(f"{operation_name} exceeded memory limit ({self.max_memory_mb}MB)")
                    raise PreprocessingMemoryError(
                        f"{operation_name} exceeded memory limit",
                        memory_used_mb=current_memory,
                        memory_limit_mb=self.max_memory_mb,
                        details={
                            "current_memory_mb": current_memory,
                            "limit_mb": self.max_memory_mb,
                            "elapsed_time": elapsed_time,
                            "peak_cpu_percent": peak_cpu
                        }
                    )
                
                # Check for resource warnings
                if current_memory > self.max_memory_mb * 0.8:
                    logger.warning(f"{operation_name} high memory usage: {current_memory:.1f}MB")
                
                if current_cpu > 90:
                    logger.warning(f"{operation_name} high CPU usage: {current_cpu:.1f}%")
                
                time.sleep(1)
                
            except (PreprocessingTimeoutError, PreprocessingMemoryError):
                raise
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(5)  # Wait longer on error
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self._lock:
            if not self.operation_history:
                return {"total_operations": 0}
            
            operations = self.operation_history
            total_operations = len(operations)
            
            # Calculate aggregated metrics
            total_time = sum(op['duration'] for op in operations)
            avg_duration = total_time / total_operations
            max_duration = max(op['duration'] for op in operations)
            min_duration = min(op['duration'] for op in operations)
            
            total_memory_delta = sum(op['memory_delta_mb'] for op in operations)
            avg_memory_delta = total_memory_delta / total_operations
            max_memory_usage = max(op['end_memory_mb'] for op in operations)
            
            avg_cpu = sum(op['avg_cpu_percent'] for op in operations) / total_operations
            
            # Operation breakdown
            operation_counts = {}
            for op in operations:
                op_name = op['operation']
                if op_name not in operation_counts:
                    operation_counts[op_name] = 0
                operation_counts[op_name] += 1
            
            return {
                "total_operations": total_operations,
                "total_time_seconds": total_time,
                "average_duration_seconds": avg_duration,
                "max_duration_seconds": max_duration,
                "min_duration_seconds": min_duration,
                "total_memory_delta_mb": total_memory_delta,
                "average_memory_delta_mb": avg_memory_delta,
                "peak_memory_usage_mb": max_memory_usage,
                "average_cpu_percent": avg_cpu,
                "operation_counts": operation_counts,
                "operations_per_second": total_operations / total_time if total_time > 0 else 0
            }

class SecurityValidator:
    """Comprehensive security validation for preprocessing operations"""
    
    def __init__(self, security_config: SecurityConfig):
        self.config = security_config
        self.security_violations = []
        self._lock = threading.RLock()
    
    def validate_column_names(self, columns: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate column names for security issues with comprehensive checks
        
        Args:
            columns: List of column names to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        with self._lock:
            errors = []
            
            # SQL injection patterns
            sql_patterns = [
                r'drop\s+table', r'delete\s+from', r'insert\s+into', r'update\s+set',
                r'alter\s+table', r'create\s+table', r'truncate\s+table',
                r'union\s+select', r'exec\s*\(', r'execute\s*\(',
                r'--', r';', r'\/\*', r'\*\/', r'xp_', r'sp_'
            ]
            
            for col in columns:
                if not isinstance(col, str):
                    errors.append(f"Non-string column name: {col}")
                    continue
                
                col_lower = col.lower()
                
                # Check for SQL injection patterns
                for pattern in sql_patterns:
                    if re.search(pattern, col_lower):
                        errors.append(f"Potentially malicious SQL pattern in column: '{col}'")
                        self._record_security_violation("sql_injection", col, pattern)
                        break
                
                # Check for sensitive data patterns
                if self.config.protect_sensitive_features:
                    for pattern in self.config.sensitive_patterns:
                        if pattern in col_lower:
                            errors.append(f"Sensitive data pattern '{pattern}' found in column: '{col}'")
                            self._record_security_violation("sensitive_data", col, pattern)
                
                # Check for script injection patterns
                script_patterns = [r'<script', r'javascript:', r'vbscript:', r'onload=', r'onerror=']
                for pattern in script_patterns:
                    if re.search(pattern, col_lower):
                        errors.append(f"Script injection pattern in column: '{col}'")
                        self._record_security_violation("script_injection", col, pattern)
                        break
                
                # Check for path traversal patterns
                if '..' in col or '/' in col or '\\' in col:
                    errors.append(f"Path traversal pattern in column: '{col}'")
                    self._record_security_violation("path_traversal", col, "path_chars")
                
                # Check for extremely long column names
                if len(col) > 255:
                    errors.append(f"Column name too long ({len(col)} chars): '{col[:50]}...'")
                    self._record_security_violation("long_column_name", col, f"length_{len(col)}")
            
            return len(errors) == 0, errors
    
    def sanitize_column_names(self, columns: List[str]) -> List[str]:
        """
        Sanitize column names for security with comprehensive cleaning
        
        Args:
            columns: List of column names to sanitize
            
        Returns:
            List of sanitized column names
        """
        if not self.config.sanitize_column_names:
            return columns
        
        sanitized = []
        used_names = set()
        
        for col in columns:
            if not isinstance(col, str):
                col = str(col)
            
            # Remove or replace dangerous characters
            sanitized_col = re.sub(r'[<>"\'/\\;(){}[\]=+*&^%$#@!?|`~]', '_', col)
            
            # Replace whitespace with underscores
            sanitized_col = re.sub(r'\s+', '_', sanitized_col)
            
            # Remove multiple consecutive underscores
            sanitized_col = re.sub(r'_+', '_', sanitized_col)
            
            # Ensure doesn't start with number or underscore
            if sanitized_col and (sanitized_col[0].isdigit() or sanitized_col[0] == '_'):
                sanitized_col = f'col_{sanitized_col}'
            
            # Ensure doesn't end with underscore
            sanitized_col = sanitized_col.rstrip('_')
            
            # Ensure not empty
            if not sanitized_col:
                sanitized_col = f'col_{len(sanitized)}'
            
            # Limit length
            if len(sanitized_col) > 100:
                sanitized_col = sanitized_col[:100]
            
            # Ensure uniqueness
            original_col = sanitized_col
            counter = 1
            while sanitized_col in used_names:
                sanitized_col = f"{original_col}_{counter}"
                counter += 1
            
            used_names.add(sanitized_col)
            sanitized.append(sanitized_col)
        
        return sanitized
    
    def validate_data_content(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data content for security issues
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        with self._lock:
            errors = []
            
            # Check for suspicious string patterns in data
            string_columns = data.select_dtypes(include=['object']).columns
            
            for col in string_columns:
                try:
                    sample_data = data[col].dropna().astype(str).head(1000)
                    
                    for idx, value in sample_data.items():
                        # Check for script injection
                        if '<script' in value.lower() or 'javascript:' in value.lower():
                            errors.append(f"Script injection detected in column '{col}' at row {idx}")
                            self._record_security_violation("data_script_injection", col, value[:100])
                        
                        # Check for SQL injection patterns
                        if any(pattern in value.lower() for pattern in ['drop table', 'delete from', 'insert into']):
                            errors.append(f"SQL injection pattern detected in column '{col}' at row {idx}")
                            self._record_security_violation("data_sql_injection", col, value[:100])
                        
                        # Check for extremely long values
                        if len(value) > 10000:
                            errors.append(f"Extremely long value in column '{col}' at row {idx}")
                            self._record_security_violation("long_value", col, f"length_{len(value)}")
                
                except Exception as e:
                    errors.append(f"Error validating data content in column '{col}': {e}")
            
            return len(errors) == 0, errors
    
    def _record_security_violation(self, violation_type: str, context: str, details: str):
        """Record security violation for monitoring"""
        violation = {
            'type': violation_type,
            'context': context,
            'details': details,
            'timestamp': datetime.now()
        }
        
        self.security_violations.append(violation)
        
        # Keep only last 1000 violations
        if len(self.security_violations) > 1000:
            self.security_violations = self.security_violations[-1000:]
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get comprehensive security summary"""
        with self._lock:
            if not self.security_violations:
                return {"total_violations": 0}
            
            violations = self.security_violations
            total_violations = len(violations)
            
            # Group by type
            violation_types = {}
            for violation in violations:
                v_type = violation['type']
                if v_type not in violation_types:
                    violation_types[v_type] = 0
                violation_types[v_type] += 1
            
            # Recent violations (last hour)
            recent_violations = [
                v for v in violations 
                if (datetime.now() - v['timestamp']).total_seconds() < 3600
            ]
            
            return {
                "total_violations": total_violations,
                "violation_types": violation_types,
                "recent_violations_count": len(recent_violations),
                "most_common_violations": sorted(
                    violation_types.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
            }

class QualityValidator:
    """Comprehensive quality validation for preprocessing operations"""
    
    def __init__(self, validation_config: ValidationConfig):
        self.config = validation_config
        self.quality_checks = []
        self._lock = threading.RLock()
    
    def validate_preprocessing_result(self, original_data: pd.DataFrame, 
                                    processed_data: pd.DataFrame,
                                    context: str = "result") -> Tuple[bool, List[str]]:
        """
        Validate preprocessing results with comprehensive quality checks
        
        Args:
            original_data: Original DataFrame before preprocessing
            processed_data: DataFrame after preprocessing
            context: Context for validation
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        with self._lock:
            errors = []
            
            try:
                # Check for data loss
                original_rows = len(original_data)
                processed_rows = len(processed_data)
                
                if processed_rows < original_rows * 0.1:
                    errors.append(f"{context}: Excessive data loss (>90% rows removed): {original_rows} -> {processed_rows}")
                elif processed_rows < original_rows * 0.5:
                    errors.append(f"{context}: Significant data loss (>50% rows removed): {original_rows} -> {processed_rows}")
                
                # Check for feature explosion
                original_cols = len(original_data.columns)
                processed_cols = len(processed_data.columns)
                
                if processed_cols > original_cols * 100:
                    errors.append(f"{context}: Excessive feature expansion (>100x columns): {original_cols} -> {processed_cols}")
                elif processed_cols > original_cols * 10:
                    errors.append(f"{context}: Significant feature expansion (>10x columns): {original_cols} -> {processed_cols}")
                
                # Check for completely null columns
                null_columns = processed_data.columns[processed_data.isnull().all()].tolist()
                if null_columns:
                    errors.append(f"{context}: Completely null columns: {null_columns}")
                
                # Check for constant features
                constant_features = []
                for col in processed_data.select_dtypes(include=[np.number]).columns:
                    try:
                        if processed_data[col].nunique() == 1:
                            constant_features.append(col)
                    except Exception:
                        pass
                
                if constant_features:
                    if len(constant_features) > processed_cols * 0.1:
                        errors.append(f"{context}: Many constant features ({len(constant_features)}): {constant_features[:10]}")
                    else:
                        errors.append(f"{context}: Constant features detected: {constant_features[:10]}")
                
                # Check for high correlation
                if self.config.check_correlation_threshold and processed_cols > 1:
                    try:
                        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 1:
                            # Sample for performance if too many columns
                            if len(numeric_cols) > 100:
                                sample_cols = np.random.choice(numeric_cols, 100, replace=False)
                                corr_data = processed_data[sample_cols]
                            else:
                                corr_data = processed_data[numeric_cols]
                            
                            corr_matrix = corr_data.corr().abs()
                            high_corr_pairs = []
                            
                            for i in range(len(corr_matrix.columns)):
                                for j in range(i + 1, len(corr_matrix.columns)):
                                    if corr_matrix.iloc[i, j] > self.config.check_correlation_threshold:
                                        high_corr_pairs.append((
                                            corr_matrix.columns[i], 
                                            corr_matrix.columns[j],
                                            corr_matrix.iloc[i, j]
                                        ))
                            
                            if high_corr_pairs:
                                errors.append(f"{context}: High correlation pairs found: {high_corr_pairs[:5]}")
                    
                    except Exception as e:
                        errors.append(f"{context}: Error in correlation check: {e}")
                
                # Check for missing data patterns
                if self.config.check_missing_threshold:
                    try:
                        missing_ratios = processed_data.isnull().sum() / len(processed_data)
                        high_missing_cols = missing_ratios[missing_ratios > self.config.check_missing_threshold].index.tolist()
                        
                        if high_missing_cols:
                            errors.append(f"{context}: Columns with excessive missing data: {high_missing_cols}")
                    
                    except Exception as e:
                        errors.append(f"{context}: Error in missing data check: {e}")
                
                # Check for extreme values
                try:
                    numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
                    extreme_value_cols = []
                    
                    for col in numeric_cols:
                        col_data = processed_data[col].dropna()
                        if len(col_data) > 0:
                            if (col_data == np.inf).any() or (col_data == -np.inf).any():
                                extreme_value_cols.append(f"{col} (inf)")
                            elif col_data.max() > 1e10 or col_data.min() < -1e10:
                                extreme_value_cols.append(f"{col} (extreme)")
                    
                    if extreme_value_cols:
                        errors.append(f"{context}: Extreme values detected: {extreme_value_cols[:5]}")
                
                except Exception as e:
                    errors.append(f"{context}: Error in extreme value check: {e}")
                
                # Check for data type consistency
                try:
                    string_cols = processed_data.select_dtypes(include=['object']).columns
                    inconsistent_cols = []
                    
                    for col in string_cols:
                        sample_data = processed_data[col].dropna().head(1000)
                        if len(sample_data) > 0:
                            unique_types = sample_data.apply(type).unique()
                            if len(unique_types) > 1:
                                inconsistent_cols.append(col)
                    
                    if inconsistent_cols:
                        errors.append(f"{context}: Data type inconsistencies: {inconsistent_cols}")
                
                except Exception as e:
                    errors.append(f"{context}: Error in data type check: {e}")
                
                # Record quality check
                self.quality_checks.append({
                    'context': context,
                    'timestamp': datetime.now(),
                    'is_valid': len(errors) == 0,
                    'error_count': len(errors),
                    'original_shape': original_data.shape,
                    'processed_shape': processed_data.shape,
                    'data_loss_ratio': 1 - (processed_rows / original_rows) if original_rows > 0 else 0,
                    'feature_expansion_ratio': processed_cols / original_cols if original_cols > 0 else 0
                })
                
                return len(errors) == 0, errors
                
            except Exception as e:
                errors.append(f"{context}: Unexpected error during quality validation: {str(e)}")
                return False, errors
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get comprehensive quality summary"""
        with self._lock:
            if not self.quality_checks:
                return {"total_checks": 0}
            
            checks = self.quality_checks
            total_checks = len(checks)
            successful_checks = sum(1 for check in checks if check['is_valid'])
            
            # Calculate averages
            avg_data_loss = sum(check['data_loss_ratio'] for check in checks) / total_checks
            avg_feature_expansion = sum(check['feature_expansion_ratio'] for check in checks) / total_checks
            
            return {
                "total_checks": total_checks,
                "successful_checks": successful_checks,
                "failed_checks": total_checks - successful_checks,
                "success_rate": successful_checks / total_checks if total_checks > 0 else 0,
                "average_data_loss_ratio": avg_data_loss,
                "average_feature_expansion_ratio": avg_feature_expansion,
                "contexts": list(set(check['context'] for check in checks))
            }

# Export enhanced framework components
__all__ = [
    # Exceptions
    'PreprocessingException',
    'PreprocessingConfigError',
    'PreprocessingInputError',
    'PreprocessingTimeoutError',
    'PreprocessingMemoryError',
    'PreprocessingSecurityError',
    'PreprocessingIntegrationError',
    'FeatureEngineeringError',
    'EncodingError',
    'ScalingError',
    'ImputationError',
    'PartialPreprocessingError',
    
    # Configuration
    'SecurityConfig',
    'ValidationConfig', 
    'PerformanceConfig',
    
    # Validators
    'InputValidator',
    'PerformanceMonitor',
    'SecurityValidator',
    'QualityValidator'
]

if __name__ == "__main__":
    print("Enhanced Data Preprocessor - Chunk 1: Core Exceptions and Validation Framework loaded")