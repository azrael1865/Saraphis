"""
Enhanced Validator Framework - Chunk 1: Enhanced Decorators and Exception Hierarchy
Comprehensive validation system with production-ready error handling, decorators,
and enhanced exception management for financial fraud detection.
"""

import logging
import pandas as pd
import numpy as np
import re
import json
import time
import asyncio
import warnings
import threading
import psutil
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set, Callable, Pattern, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from functools import wraps, lru_cache
from contextlib import contextmanager
import hashlib
import signal
import inspect

# Import existing validator components
try:
    from data_validator import (
        ValidationLevel, ValidationSeverity, ComplianceStandard,
        ValidationIssue, ValidationResult, ValidationRule, FinancialDataValidator
    )
    from enhanced_data_validator import (
        ValidationException, InputValidationError, ValidationTimeoutError,
        ValidationConfigError, ValidationSecurityError, ValidationIntegrationError,
        DataCorruptionError, PartialValidationError
    )
    EXISTING_COMPONENTS = True
except ImportError as e:
    # Fallback definitions if imports fail
    EXISTING_COMPONENTS = False
    logger.warning(f"Could not import existing validator components: {e}")

# Configure logging
logger = logging.getLogger(__name__)

# ======================== ENHANCED EXCEPTION HIERARCHY ========================

class TransactionValidationError(ValidationException if EXISTING_COMPONENTS else Exception):
    """Enhanced transaction-specific validation error with detailed context"""
    
    def __init__(self, message: str, 
                 transaction_id: Optional[str] = None,
                 field: Optional[str] = None,
                 expected_value: Optional[Any] = None,
                 actual_value: Optional[Any] = None,
                 validation_rule: Optional[str] = None,
                 recoverable: bool = True,
                 **kwargs):
        if EXISTING_COMPONENTS:
            super().__init__(message, **kwargs)
        else:
            super().__init__(message)
            self.details = kwargs.get('details', {})
            self.timestamp = datetime.now()
        
        self.transaction_id = transaction_id
        self.field = field
        self.expected_value = expected_value
        self.actual_value = actual_value
        self.validation_rule = validation_rule
        self.recoverable = recoverable
        self.error_code = f"TXN_VAL_{field.upper() if field else 'UNKNOWN'}"
        
        # Enhanced context information
        self.validation_context = {
            'transaction_id': transaction_id,
            'field': field,
            'rule': validation_rule,
            'timestamp': self.timestamp.isoformat(),
            'recoverable': recoverable
        }

class BusinessRuleValidationError(ValidationException if EXISTING_COMPONENTS else Exception):
    """Enhanced business rule validation error"""
    
    def __init__(self, message: str,
                 rule_name: Optional[str] = None,
                 rule_config: Optional[Dict[str, Any]] = None,
                 affected_records: Optional[List[str]] = None,
                 severity_level: str = "ERROR",
                 **kwargs):
        if EXISTING_COMPONENTS:
            super().__init__(message, **kwargs)
        else:
            super().__init__(message)
            self.details = kwargs.get('details', {})
            self.timestamp = datetime.now()
        
        self.rule_name = rule_name
        self.rule_config = rule_config or {}
        self.affected_records = affected_records or []
        self.severity_level = severity_level
        self.error_code = f"BIZ_RULE_{rule_name.upper() if rule_name else 'UNKNOWN'}"

class ThresholdValidationError(ValidationException if EXISTING_COMPONENTS else Exception):
    """Enhanced threshold validation error with range context"""
    
    def __init__(self, message: str,
                 threshold_name: Optional[str] = None,
                 actual_value: Optional[float] = None,
                 expected_range: Optional[Tuple[float, float]] = None,
                 threshold_type: str = "range",
                 **kwargs):
        if EXISTING_COMPONENTS:
            super().__init__(message, **kwargs)
        else:
            super().__init__(message)
            self.details = kwargs.get('details', {})
            self.timestamp = datetime.now()
        
        self.threshold_name = threshold_name
        self.actual_value = actual_value
        self.expected_range = expected_range
        self.threshold_type = threshold_type
        self.error_code = f"THRESHOLD_{threshold_name.upper() if threshold_name else 'UNKNOWN'}"
        
        # Calculate deviation if range is provided
        self.deviation = None
        if expected_range and actual_value is not None:
            min_val, max_val = expected_range
            if actual_value < min_val:
                self.deviation = actual_value - min_val
            elif actual_value > max_val:
                self.deviation = actual_value - max_val

class ContextValidationError(ValidationException if EXISTING_COMPONENTS else Exception):
    """Enhanced validation context error"""
    
    def __init__(self, message: str,
                 context_type: Optional[str] = None,
                 environment: Optional[str] = None,
                 user_role: Optional[str] = None,
                 required_permissions: Optional[List[str]] = None,
                 **kwargs):
        if EXISTING_COMPONENTS:
            super().__init__(message, **kwargs)
        else:
            super().__init__(message)
            self.details = kwargs.get('details', {})
            self.timestamp = datetime.now()
        
        self.context_type = context_type
        self.environment = environment
        self.user_role = user_role
        self.required_permissions = required_permissions or []
        self.error_code = f"CONTEXT_{context_type.upper() if context_type else 'UNKNOWN'}"

class PerformanceValidationError(ValidationException if EXISTING_COMPONENTS else Exception):
    """Enhanced performance validation error"""
    
    def __init__(self, message: str,
                 operation_name: Optional[str] = None,
                 actual_time: Optional[float] = None,
                 timeout_limit: Optional[float] = None,
                 memory_usage: Optional[float] = None,
                 memory_limit: Optional[float] = None,
                 **kwargs):
        if EXISTING_COMPONENTS:
            super().__init__(message, **kwargs)
        else:
            super().__init__(message)
            self.details = kwargs.get('details', {})
            self.timestamp = datetime.now()
        
        self.operation_name = operation_name
        self.actual_time = actual_time
        self.timeout_limit = timeout_limit
        self.memory_usage = memory_usage
        self.memory_limit = memory_limit
        self.error_code = f"PERF_{operation_name.upper() if operation_name else 'UNKNOWN'}"

class RecoveryValidationError(ValidationException if EXISTING_COMPONENTS else Exception):
    """Enhanced validation error with recovery information"""
    
    def __init__(self, message: str,
                 recovery_strategy: Optional[str] = None,
                 recovery_possible: bool = True,
                 partial_results: Optional[Any] = None,
                 recovery_metadata: Optional[Dict[str, Any]] = None,
                 **kwargs):
        if EXISTING_COMPONENTS:
            super().__init__(message, **kwargs)
        else:
            super().__init__(message)
            self.details = kwargs.get('details', {})
            self.timestamp = datetime.now()
        
        self.recovery_strategy = recovery_strategy
        self.recovery_possible = recovery_possible
        self.partial_results = partial_results
        self.recovery_metadata = recovery_metadata or {}
        self.error_code = f"RECOVERY_{recovery_strategy.upper() if recovery_strategy else 'UNKNOWN'}"

# ======================== ENHANCED VALIDATION DECORATORS ========================

def enhanced_input_validator(required_fields: Optional[List[str]] = None,
                           max_size_mb: Optional[float] = None,
                           min_records: Optional[int] = None,
                           max_records: Optional[int] = None,
                           allowed_dtypes: Optional[Dict[str, str]] = None,
                           custom_validator: Optional[Callable] = None):
    """
    Enhanced input validation decorator with comprehensive checks
    
    Args:
        required_fields: List of required column names
        max_size_mb: Maximum DataFrame size in MB
        min_records: Minimum number of records
        max_records: Maximum number of records
        allowed_dtypes: Expected data types for columns
        custom_validator: Custom validation function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Extract DataFrame from arguments
                data = None
                if args and isinstance(args[0], pd.DataFrame):
                    data = args[0]
                elif 'data' in kwargs and isinstance(kwargs['data'], pd.DataFrame):
                    data = kwargs['data']
                
                if data is not None:
                    # Check for empty DataFrame
                    if data.empty:
                        raise InputValidationError(
                            "Empty DataFrame provided",
                            details={'records': 0, 'columns': 0}
                        )
                    
                    # Check required fields
                    if required_fields:
                        missing_fields = set(required_fields) - set(data.columns)
                        if missing_fields:
                            raise InputValidationError(
                                f"Missing required fields: {missing_fields}",
                                details={
                                    'missing_fields': list(missing_fields),
                                    'available_fields': list(data.columns),
                                    'required_fields': required_fields
                                }
                            )
                    
                    # Check size limits
                    if max_size_mb:
                        size_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
                        if size_mb > max_size_mb:
                            raise InputValidationError(
                                f"DataFrame size ({size_mb:.2f}MB) exceeds limit ({max_size_mb}MB)",
                                details={
                                    'actual_size_mb': size_mb,
                                    'max_size_mb': max_size_mb,
                                    'records': len(data),
                                    'columns': len(data.columns)
                                }
                            )
                    
                    # Check record count limits
                    record_count = len(data)
                    if min_records and record_count < min_records:
                        raise InputValidationError(
                            f"Insufficient records ({record_count}) - minimum required: {min_records}",
                            details={
                                'actual_records': record_count,
                                'min_records': min_records
                            }
                        )
                    
                    if max_records and record_count > max_records:
                        raise InputValidationError(
                            f"Too many records ({record_count}) - maximum allowed: {max_records}",
                            details={
                                'actual_records': record_count,
                                'max_records': max_records
                            }
                        )
                    
                    # Check data types
                    if allowed_dtypes:
                        dtype_issues = []
                        for column, expected_dtype in allowed_dtypes.items():
                            if column in data.columns:
                                actual_dtype = str(data[column].dtype)
                                if not actual_dtype.startswith(expected_dtype):
                                    dtype_issues.append({
                                        'column': column,
                                        'expected': expected_dtype,
                                        'actual': actual_dtype
                                    })
                        
                        if dtype_issues:
                            raise InputValidationError(
                                f"Data type mismatches found",
                                details={'dtype_issues': dtype_issues}
                            )
                    
                    # Custom validation
                    if custom_validator:
                        try:
                            custom_validator(data)
                        except Exception as e:
                            raise InputValidationError(
                                f"Custom validation failed: {e}",
                                details={'custom_error': str(e)}
                            )
                
                return func(*args, **kwargs)
                
            except (InputValidationError, ValidationException):
                raise
            except Exception as e:
                raise InputValidationError(
                    f"Input validation failed: {e}",
                    details={
                        'function': func.__name__,
                        'original_error': str(e),
                        'traceback': traceback.format_exc()
                    }
                )
        
        return wrapper
    return decorator

def enhanced_error_handler(recovery_strategy: str = 'partial',
                         log_errors: bool = True,
                         max_retries: int = 3,
                         retry_delay: float = 1.0,
                         escalate_on_failure: bool = True):
    """
    Enhanced error handling decorator with recovery strategies
    
    Args:
        recovery_strategy: 'partial', 'retry', 'fallback', 'skip'
        log_errors: Whether to log errors
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries (seconds)
        escalate_on_failure: Whether to escalate unrecoverable errors
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(self, *args, **kwargs)
                    
                except ValidationException as e:
                    last_exception = e
                    
                    if log_errors:
                        logger.error(
                            f"Validation error in {func.__name__} (attempt {attempt + 1}): {e}",
                            extra={
                                'function': func.__name__,
                                'attempt': attempt + 1,
                                'max_retries': max_retries,
                                'error_type': type(e).__name__,
                                'recoverable': getattr(e, 'recoverable', True)
                            }
                        )
                    
                    # Check if error is recoverable
                    if not getattr(e, 'recoverable', True):
                        break
                    
                    # Try recovery based on strategy
                    if recovery_strategy == 'partial' and hasattr(e, 'partial_results'):
                        if e.partial_results is not None:
                            logger.info(f"Recovered partial results from {func.__name__}")
                            return e.partial_results
                    
                    elif recovery_strategy == 'retry' and attempt < max_retries:
                        logger.info(f"Retrying {func.__name__} in {retry_delay}s...")
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                    
                    elif recovery_strategy == 'fallback' and hasattr(self, 'get_fallback_result'):
                        try:
                            fallback_result = self.get_fallback_result(*args, **kwargs)
                            logger.info(f"Using fallback result for {func.__name__}")
                            return fallback_result
                        except Exception as fallback_error:
                            logger.error(f"Fallback failed: {fallback_error}")
                    
                    elif recovery_strategy == 'skip':
                        logger.warning(f"Skipping {func.__name__} due to error")
                        return None
                    
                    # If we're here and it's not the last attempt, continue to retry
                    if attempt < max_retries:
                        continue
                    else:
                        break
                
                except Exception as e:
                    last_exception = e
                    
                    if log_errors:
                        logger.error(
                            f"Unexpected error in {func.__name__}: {e}\n{traceback.format_exc()}",
                            extra={
                                'function': func.__name__,
                                'attempt': attempt + 1,
                                'error_type': type(e).__name__
                            }
                        )
                    
                    # Wrap in ValidationException
                    wrapped_error = ValidationException(
                        f"Validation failed in {func.__name__}: {e}",
                        details={
                            'original_error': str(e),
                            'error_type': type(e).__name__,
                            'traceback': traceback.format_exc(),
                            'function': func.__name__,
                            'attempt': attempt + 1
                        }
                    )
                    last_exception = wrapped_error
                    
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                        continue
                    else:
                        break
            
            # If we get here, all attempts failed
            if escalate_on_failure:
                if isinstance(last_exception, ValidationException):
                    raise last_exception
                else:
                    raise ValidationException(
                        f"All {max_retries + 1} attempts failed for {func.__name__}",
                        details={'last_error': str(last_exception)}
                    )
            else:
                logger.error(f"All attempts failed for {func.__name__}, returning None")
                return None
        
        return wrapper
    return decorator

def enhanced_performance_monitor(operation_name: Optional[str] = None,
                               timeout_seconds: Optional[float] = None,
                               memory_limit_mb: Optional[float] = None,
                               log_performance: bool = True,
                               track_detailed_metrics: bool = True):
    """
    Enhanced performance monitoring decorator with resource tracking
    
    Args:
        operation_name: Name for the operation being monitored
        timeout_seconds: Maximum allowed execution time
        memory_limit_mb: Maximum allowed memory usage
        log_performance: Whether to log performance metrics
        track_detailed_metrics: Whether to track detailed metrics
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            op_name = operation_name or func.__name__
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Set up timeout if specified
            if timeout_seconds:
                def timeout_handler(signum, frame):
                    raise PerformanceValidationError(
                        f"Operation '{op_name}' timed out after {timeout_seconds}s",
                        operation_name=op_name,
                        timeout_limit=timeout_seconds
                    )
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout_seconds))
            
            try:
                # Monitor memory during execution
                if memory_limit_mb:
                    def check_memory():
                        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                        if current_memory > memory_limit_mb:
                            raise PerformanceValidationError(
                                f"Memory usage ({current_memory:.1f}MB) exceeds limit ({memory_limit_mb}MB)",
                                operation_name=op_name,
                                memory_usage=current_memory,
                                memory_limit=memory_limit_mb
                            )
                    
                    # Check memory before execution
                    check_memory()
                
                # Execute the function
                result = func(self, *args, **kwargs)
                
                # Calculate performance metrics
                end_time = time.time()
                duration = end_time - start_time
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_delta = end_memory - start_memory
                peak_memory = max(start_memory, end_memory)
                
                # Store metrics if tracking is enabled
                if track_detailed_metrics and hasattr(self, '_performance_metrics'):
                    self._performance_metrics[op_name] = {
                        'duration': duration,
                        'start_time': start_time,
                        'end_time': end_time,
                        'memory_start': start_memory,
                        'memory_end': end_memory,
                        'memory_delta': memory_delta,
                        'memory_peak': peak_memory,
                        'timestamp': datetime.now().isoformat(),
                        'success': True
                    }
                
                # Record in performance monitor if available
                if hasattr(self, 'performance_monitor'):
                    self.performance_monitor.record_rule_time(op_name, duration)
                
                # Log performance if enabled
                if log_performance:
                    logger.info(
                        f"Performance: {op_name} completed in {duration:.3f}s "
                        f"(memory: {memory_delta:+.1f}MB, peak: {peak_memory:.1f}MB)",
                        extra={
                            'operation': op_name,
                            'duration': duration,
                            'memory_delta': memory_delta,
                            'memory_peak': peak_memory
                        }
                    )
                
                return result
                
            except PerformanceValidationError:
                # Re-raise performance errors as-is
                raise
                
            except Exception as e:
                # Calculate metrics for failed operations
                end_time = time.time()
                duration = end_time - start_time
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Store failure metrics
                if track_detailed_metrics and hasattr(self, '_performance_metrics'):
                    self._performance_metrics[f"{op_name}_failed"] = {
                        'duration': duration,
                        'start_time': start_time,
                        'end_time': end_time,
                        'memory_start': start_memory,
                        'memory_end': end_memory,
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'timestamp': datetime.now().isoformat(),
                        'success': False
                    }
                
                if log_performance:
                    logger.error(
                        f"Performance: {op_name} failed after {duration:.3f}s with error: {e}",
                        extra={
                            'operation': op_name,
                            'duration': duration,
                            'error': str(e),
                            'error_type': type(e).__name__
                        }
                    )
                
                raise
                
            finally:
                # Clear timeout alarm
                if timeout_seconds:
                    signal.alarm(0)
        
        return wrapper
    return decorator

def validation_context_manager(context_type: str = "validation",
                             environment: str = "development",
                             auto_cleanup: bool = True,
                             track_resources: bool = True):
    """
    Enhanced validation context manager decorator
    
    Args:
        context_type: Type of validation context
        environment: Execution environment
        auto_cleanup: Whether to auto-cleanup resources
        track_resources: Whether to track resource usage
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            context_id = f"{context_type}_{func.__name__}_{int(time.time())}"
            
            # Initialize context tracking
            if track_resources and hasattr(self, '_context_tracker'):
                self._context_tracker[context_id] = {
                    'start_time': time.time(),
                    'function': func.__name__,
                    'context_type': context_type,
                    'environment': environment,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys()),
                    'thread_id': threading.get_ident(),
                    'process_id': os.getpid() if hasattr(os, 'getpid') else None
                }
            
            try:
                # Set validation context
                if hasattr(self, 'set_validation_context'):
                    self.set_validation_context(context_type, environment)
                
                # Execute function
                result = wrapper(self, *args, **kwargs)
                
                # Update context with success
                if track_resources and hasattr(self, '_context_tracker'):
                    self._context_tracker[context_id].update({
                        'end_time': time.time(),
                        'success': True,
                        'result_type': type(result).__name__
                    })
                
                return result
                
            except Exception as e:
                # Update context with failure
                if track_resources and hasattr(self, '_context_tracker'):
                    self._context_tracker[context_id].update({
                        'end_time': time.time(),
                        'success': False,
                        'error': str(e),
                        'error_type': type(e).__name__
                    })
                
                # Enhance error with context information
                if isinstance(e, ValidationException):
                    e.details['validation_context'] = {
                        'context_id': context_id,
                        'context_type': context_type,
                        'environment': environment,
                        'function': func.__name__
                    }
                
                raise
                
            finally:
                # Cleanup if enabled
                if auto_cleanup:
                    if hasattr(self, 'cleanup_validation_context'):
                        try:
                            self.cleanup_validation_context(context_id)
                        except Exception as cleanup_error:
                            logger.warning(f"Context cleanup failed: {cleanup_error}")
                
                # Remove from tracker after a delay to allow for debugging
                if track_resources and hasattr(self, '_context_tracker'):
                    def delayed_cleanup():
                        time.sleep(300)  # Keep for 5 minutes
                        self._context_tracker.pop(context_id, None)
                    
                    threading.Thread(target=delayed_cleanup, daemon=True).start()
        
        return wrapper
    return decorator

def business_rule_validator(rule_name: str,
                          rule_config: Optional[Dict[str, Any]] = None,
                          required_fields: Optional[List[str]] = None,
                          severity_level: str = "ERROR"):
    """
    Enhanced business rule validation decorator
    
    Args:
        rule_name: Name of the business rule
        rule_config: Configuration for the rule
        required_fields: Fields required for the rule
        severity_level: Severity level for violations
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, data: pd.DataFrame, *args, **kwargs):
            try:
                # Validate required fields
                if required_fields:
                    missing_fields = set(required_fields) - set(data.columns)
                    if missing_fields:
                        raise BusinessRuleValidationError(
                            f"Business rule '{rule_name}' missing required fields: {missing_fields}",
                            rule_name=rule_name,
                            rule_config=rule_config,
                            severity_level=severity_level
                        )
                
                # Execute rule validation
                result = func(self, data, *args, **kwargs)
                
                # Log successful rule execution
                logger.debug(
                    f"Business rule '{rule_name}' executed successfully",
                    extra={
                        'rule_name': rule_name,
                        'records_processed': len(data),
                        'result_type': type(result).__name__
                    }
                )
                
                return result
                
            except BusinessRuleValidationError:
                raise
                
            except Exception as e:
                raise BusinessRuleValidationError(
                    f"Business rule '{rule_name}' execution failed: {e}",
                    rule_name=rule_name,
                    rule_config=rule_config,
                    severity_level=severity_level,
                    affected_records=data.index.tolist() if hasattr(data, 'index') else []
                )
        
        return wrapper
    return decorator

# ======================== ENHANCED VALIDATION UTILITIES ========================

class ValidationMetricsCollector:
    """Enhanced metrics collection for validation operations"""
    
    def __init__(self):
        self.metrics = {
            'validation_operations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'partial_validations': 0,
            'total_records_processed': 0,
            'total_issues_found': 0,
            'error_rates_by_type': {},
            'performance_metrics': {},
            'rule_execution_counts': {},
            'recovery_attempts': 0,
            'successful_recoveries': 0
        }
        self.start_time = time.time()
        self._lock = threading.RLock()
    
    def record_validation(self, success: bool, records_processed: int, 
                         issues_found: int, duration: float,
                         validation_type: str = "standard"):
        """Record validation metrics"""
        with self._lock:
            self.metrics['validation_operations'] += 1
            self.metrics['total_records_processed'] += records_processed
            self.metrics['total_issues_found'] += issues_found
            
            if success:
                self.metrics['successful_validations'] += 1
            else:
                self.metrics['failed_validations'] += 1
            
            # Update performance metrics
            if validation_type not in self.metrics['performance_metrics']:
                self.metrics['performance_metrics'][validation_type] = {
                    'count': 0,
                    'total_duration': 0.0,
                    'avg_duration': 0.0,
                    'min_duration': float('inf'),
                    'max_duration': 0.0
                }
            
            perf = self.metrics['performance_metrics'][validation_type]
            perf['count'] += 1
            perf['total_duration'] += duration
            perf['avg_duration'] = perf['total_duration'] / perf['count']
            perf['min_duration'] = min(perf['min_duration'], duration)
            perf['max_duration'] = max(perf['max_duration'], duration)
    
    def record_error(self, error_type: str, recoverable: bool = True):
        """Record error metrics"""
        with self._lock:
            if error_type not in self.metrics['error_rates_by_type']:
                self.metrics['error_rates_by_type'][error_type] = {
                    'count': 0,
                    'recoverable_count': 0,
                    'non_recoverable_count': 0
                }
            
            self.metrics['error_rates_by_type'][error_type]['count'] += 1
            if recoverable:
                self.metrics['error_rates_by_type'][error_type]['recoverable_count'] += 1
            else:
                self.metrics['error_rates_by_type'][error_type]['non_recoverable_count'] += 1
    
    def record_recovery_attempt(self, success: bool, strategy: str):
        """Record recovery attempt metrics"""
        with self._lock:
            self.metrics['recovery_attempts'] += 1
            if success:
                self.metrics['successful_recoveries'] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        with self._lock:
            runtime = time.time() - self.start_time
            total_validations = self.metrics['validation_operations']
            
            summary = {
                'runtime_seconds': runtime,
                'total_validations': total_validations,
                'success_rate': (self.metrics['successful_validations'] / total_validations * 100 
                               if total_validations > 0 else 0),
                'failure_rate': (self.metrics['failed_validations'] / total_validations * 100 
                               if total_validations > 0 else 0),
                'recovery_success_rate': (self.metrics['successful_recoveries'] / 
                                        self.metrics['recovery_attempts'] * 100 
                                        if self.metrics['recovery_attempts'] > 0 else 0),
                'avg_records_per_validation': (self.metrics['total_records_processed'] / 
                                             total_validations if total_validations > 0 else 0),
                'avg_issues_per_validation': (self.metrics['total_issues_found'] / 
                                            total_validations if total_validations > 0 else 0),
                'throughput_records_per_second': (self.metrics['total_records_processed'] / 
                                                runtime if runtime > 0 else 0),
                'error_breakdown': self.metrics['error_rates_by_type'].copy(),
                'performance_breakdown': self.metrics['performance_metrics'].copy()
            }
            
            return summary

# Export enhanced components
__all__ = [
    # Enhanced exceptions
    'TransactionValidationError',
    'BusinessRuleValidationError', 
    'ThresholdValidationError',
    'ContextValidationError',
    'PerformanceValidationError',
    'RecoveryValidationError',
    
    # Enhanced decorators
    'enhanced_input_validator',
    'enhanced_error_handler',
    'enhanced_performance_monitor',
    'validation_context_manager',
    'business_rule_validator',
    
    # Utilities
    'ValidationMetricsCollector'
]

if __name__ == "__main__":
    print("Enhanced Validator Framework - Chunk 1: Decorators and Exceptions loaded")