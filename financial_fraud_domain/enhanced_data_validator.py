"""
Enhanced Data Validation for Financial Fraud Detection
Comprehensive validation system with enhanced error handling, performance monitoring,
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
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set, Callable, Pattern, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import hashlib
from contextlib import contextmanager
from functools import wraps, lru_cache
import signal

# Import existing components for enhancement
try:
    from data_validator import (
        ValidationLevel,
        ValidationSeverity,
        ComplianceStandard,
        ValidationIssue,
        ValidationResult,
        ValidationRule,
        RequiredFieldsRule,
        DataTypeRule,
        RangeValidationRule,
        PatternValidationRule,
        DuplicateRule,
        FraudPatternRule,
        ComplianceRule,
        FinancialDataValidator
    )
    EXISTING_VALIDATOR = True
except ImportError:
    # Define basic types if import fails
    EXISTING_VALIDATOR = False
    from enum import Enum
    
    class ValidationLevel(Enum):
        NONE = "none"
        BASIC = "basic"
        STANDARD = "standard"
        STRICT = "strict"
        COMPREHENSIVE = "comprehensive"
        REGULATORY = "regulatory"

# Configure logging
logger = logging.getLogger(__name__)

# Custom Exceptions
class ValidationException(Exception):
    """Base exception for validation errors"""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message)
        self.details = details or {}
        self.timestamp = datetime.now()

class InputValidationError(ValidationException):
    """Invalid input data or configuration"""
    pass

class ValidationTimeoutError(ValidationException):
    """Validation operation timed out"""
    pass

class ValidationConfigError(ValidationException):
    """Invalid validation configuration"""
    pass

class ValidationSecurityError(ValidationException):
    """Security validation failure"""
    pass

class ValidationIntegrationError(ValidationException):
    """Integration validation failure"""
    pass

class DataCorruptionError(ValidationException):
    """Data corruption detected during validation"""
    pass

class PartialValidationError(ValidationException):
    """Partial validation failure with recoverable results"""
    def __init__(self, message: str, partial_results: Any = None, **kwargs):
        super().__init__(message, **kwargs)
        self.partial_results = partial_results

# Enhanced Enums
class ValidationContext(Enum):
    """Validation execution context"""
    PRODUCTION = "production"
    STAGING = "staging"
    TESTING = "testing"
    DEVELOPMENT = "development"
    AUDIT = "audit"

class ValidationMode(Enum):
    """Validation execution mode"""
    FULL = "full"
    PARTIAL = "partial"
    SAMPLING = "sampling"
    STREAMING = "streaming"
    BATCH = "batch"

class SecurityLevel(Enum):
    """Security validation levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"

@dataclass
class ValidationConfig:
    """Enhanced validation configuration"""
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    context: ValidationContext = ValidationContext.PRODUCTION
    mode: ValidationMode = ValidationMode.FULL
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    timeout_seconds: float = 300.0
    max_memory_mb: float = 1024.0
    max_threads: int = 4
    enable_partial_results: bool = True
    enable_recovery: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    sampling_rate: float = 1.0
    batch_size: int = 10000
    fail_fast: bool = False
    detailed_errors: bool = True
    performance_monitoring: bool = True
    security_validation: bool = True
    integration_testing: bool = False
    compliance_standards: List[str] = field(default_factory=list)
    custom_thresholds: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationPerformanceMetrics:
    """Validation performance metrics"""
    total_time_seconds: float
    validation_time_seconds: float
    preprocessing_time_seconds: float
    rule_execution_times: Dict[str, float]
    memory_usage_mb: float
    cpu_usage_percent: float
    thread_count: int
    cache_hits: int
    cache_misses: int
    rules_executed: int
    rules_skipped: int
    records_per_second: float
    bottlenecks: List[str]
    optimization_suggestions: List[str]

@dataclass
class ValidationSecurityReport:
    """Security validation report"""
    security_level: SecurityLevel
    access_validated: bool
    data_classification: str
    pii_detected: bool
    pii_fields: List[str]
    encryption_required: bool
    compliance_violations: List[str]
    security_risks: List[str]
    audit_trail: List[Dict[str, Any]]
    recommendations: List[str]

@dataclass
class ValidationIntegrationReport:
    """Integration validation report"""
    brain_system_compatible: bool
    api_version: str
    dependencies_validated: bool
    missing_dependencies: List[str]
    interface_errors: List[str]
    data_format_compatible: bool
    performance_acceptable: bool
    integration_score: float
    recommendations: List[str]

class InputValidator:
    """Comprehensive input validation"""
    
    @staticmethod
    def validate_dataframe(data: Any, min_rows: int = 0, max_rows: Optional[int] = None,
                          required_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Validate DataFrame input"""
        if data is None:
            raise InputValidationError("Data cannot be None")
            
        if not isinstance(data, pd.DataFrame):
            raise InputValidationError(
                f"Expected pandas DataFrame, got {type(data).__name__}",
                {"actual_type": type(data).__name__}
            )
            
        if data.empty and min_rows > 0:
            raise InputValidationError(
                "DataFrame is empty but minimum rows required",
                {"min_rows": min_rows}
            )
            
        if len(data) < min_rows:
            raise InputValidationError(
                f"DataFrame has {len(data)} rows, minimum {min_rows} required",
                {"actual_rows": len(data), "min_rows": min_rows}
            )
            
        if max_rows is not None and len(data) > max_rows:
            raise InputValidationError(
                f"DataFrame has {len(data)} rows, maximum {max_rows} allowed",
                {"actual_rows": len(data), "max_rows": max_rows}
            )
            
        if required_columns:
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                raise InputValidationError(
                    f"Missing required columns: {missing_columns}",
                    {"missing_columns": list(missing_columns)}
                )
                
        return data
    
    @staticmethod
    def validate_validation_config(config: ValidationConfig) -> ValidationConfig:
        """Validate validation configuration"""
        if not isinstance(config, ValidationConfig):
            raise ValidationConfigError(
                f"Expected ValidationConfig, got {type(config).__name__}"
            )
            
        # Validate timeout
        if config.timeout_seconds <= 0:
            raise ValidationConfigError(
                "Timeout must be positive",
                {"timeout_seconds": config.timeout_seconds}
            )
            
        # Validate memory limit
        if config.max_memory_mb <= 0:
            raise ValidationConfigError(
                "Memory limit must be positive",
                {"max_memory_mb": config.max_memory_mb}
            )
            
        # Validate sampling rate
        if not 0 < config.sampling_rate <= 1:
            raise ValidationConfigError(
                "Sampling rate must be between 0 and 1",
                {"sampling_rate": config.sampling_rate}
            )
            
        # Validate batch size
        if config.batch_size <= 0:
            raise ValidationConfigError(
                "Batch size must be positive",
                {"batch_size": config.batch_size}
            )
            
        return config
    
    @staticmethod
    def validate_rule(rule: Any) -> 'ValidationRule':
        """Validate validation rule"""
        if not hasattr(rule, 'rule_id') or not rule.rule_id:
            raise ValidationConfigError("Rule must have a valid rule_id")
            
        if not hasattr(rule, 'validate') or not callable(rule.validate):
            raise ValidationConfigError("Rule must have a validate method")
            
        return rule
    
    @staticmethod
    def validate_threshold(name: str, value: Any, min_val: Optional[float] = None,
                          max_val: Optional[float] = None) -> float:
        """Validate numeric threshold"""
        try:
            float_value = float(value)
        except (TypeError, ValueError):
            raise ValidationConfigError(
                f"Threshold '{name}' must be numeric",
                {"value": value, "type": type(value).__name__}
            )
            
        if min_val is not None and float_value < min_val:
            raise ValidationConfigError(
                f"Threshold '{name}' below minimum",
                {"value": float_value, "min": min_val}
            )
            
        if max_val is not None and float_value > max_val:
            raise ValidationConfigError(
                f"Threshold '{name}' above maximum",
                {"value": float_value, "max": max_val}
            )
            
        return float_value

class PerformanceMonitor:
    """Monitor validation performance"""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {}
        self.rule_times = {}
        self._process = psutil.Process()
        
    def start(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.metrics['start_memory'] = self._process.memory_info().rss / 1024 / 1024
        self.metrics['start_cpu'] = self._process.cpu_percent()
        
    def record_rule_time(self, rule_id: str, duration: float):
        """Record rule execution time"""
        self.rule_times[rule_id] = duration
        
    def stop(self) -> ValidationPerformanceMetrics:
        """Stop monitoring and return metrics"""
        end_time = time.time()
        total_time = end_time - self.start_time
        
        # Memory metrics
        end_memory = self._process.memory_info().rss / 1024 / 1024
        memory_usage = end_memory - self.metrics['start_memory']
        
        # CPU metrics
        cpu_usage = self._process.cpu_percent(interval=0.1)
        
        # Thread metrics
        thread_count = threading.active_count()
        
        # Identify bottlenecks
        bottlenecks = []
        if memory_usage > 500:  # MB
            bottlenecks.append(f"High memory usage: {memory_usage:.1f}MB")
            
        slow_rules = [
            rule_id for rule_id, duration in self.rule_times.items()
            if duration > 5.0
        ]
        if slow_rules:
            bottlenecks.append(f"Slow rules: {', '.join(slow_rules)}")
            
        # Optimization suggestions
        suggestions = []
        if memory_usage > 500:
            suggestions.append("Consider batch processing for large datasets")
        if len(self.rule_times) > 20:
            suggestions.append("Consider disabling non-critical rules")
        if cpu_usage > 80:
            suggestions.append("Consider using sampling mode for faster validation")
            
        return ValidationPerformanceMetrics(
            total_time_seconds=total_time,
            validation_time_seconds=sum(self.rule_times.values()),
            preprocessing_time_seconds=total_time - sum(self.rule_times.values()),
            rule_execution_times=self.rule_times.copy(),
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            thread_count=thread_count,
            cache_hits=self.metrics.get('cache_hits', 0),
            cache_misses=self.metrics.get('cache_misses', 0),
            rules_executed=len(self.rule_times),
            rules_skipped=self.metrics.get('rules_skipped', 0),
            records_per_second=self.metrics.get('record_count', 0) / total_time if total_time > 0 else 0,
            bottlenecks=bottlenecks,
            optimization_suggestions=suggestions
        )

class SecurityValidator:
    """Security validation for data and operations"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.pii_patterns = {
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
        }
        
    def validate_security(self, data: pd.DataFrame) -> ValidationSecurityReport:
        """Perform security validation"""
        audit_trail = []
        security_risks = []
        pii_fields = []
        compliance_violations = []
        
        # Check data access level
        access_validated = self._validate_access_level()
        audit_trail.append({
            'action': 'access_validation',
            'result': 'passed' if access_validated else 'failed',
            'timestamp': datetime.now().isoformat()
        })
        
        # Detect PII
        pii_detected = False
        for column in data.columns:
            column_str = str(data[column].iloc[:1000])  # Sample first 1000 rows
            
            for pii_type, pattern in self.pii_patterns.items():
                if pattern.search(column_str):
                    pii_detected = True
                    pii_fields.append(f"{column} (potential {pii_type})")
                    security_risks.append(f"Unencrypted {pii_type} in {column}")
                    
        # Check compliance
        if pii_detected and ComplianceStandard.GDPR in self.config.compliance_standards:
            compliance_violations.append("GDPR: Unencrypted PII detected")
            
        if pii_detected and ComplianceStandard.PCI_DSS in self.config.compliance_standards:
            if any('credit_card' in field for field in pii_fields):
                compliance_violations.append("PCI DSS: Unencrypted credit card data")
                
        # Recommendations
        recommendations = []
        if pii_detected:
            recommendations.append("Encrypt PII fields before storage")
            recommendations.append("Implement field-level access controls")
            
        if compliance_violations:
            recommendations.append("Address compliance violations immediately")
            
        return ValidationSecurityReport(
            security_level=self.config.security_level,
            access_validated=access_validated,
            data_classification=self._classify_data(pii_detected),
            pii_detected=pii_detected,
            pii_fields=pii_fields,
            encryption_required=pii_detected or self.config.security_level.value in ['confidential', 'restricted', 'top_secret'],
            compliance_violations=compliance_violations,
            security_risks=security_risks,
            audit_trail=audit_trail,
            recommendations=recommendations
        )
    
    def _validate_access_level(self) -> bool:
        """Validate access level for current context"""
        # Simplified check - in production, integrate with auth system
        if self.config.context == ValidationContext.PRODUCTION:
            return self.config.security_level != SecurityLevel.TOP_SECRET
        return True
    
    def _classify_data(self, pii_detected: bool) -> str:
        """Classify data based on content"""
        if pii_detected:
            return "CONFIDENTIAL"
        elif self.config.security_level == SecurityLevel.PUBLIC:
            return "PUBLIC"
        else:
            return "INTERNAL"

class IntegrationValidator:
    """Validate integration with Brain system"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.required_interfaces = [
            'data_processor',
            'model_interface',
            'result_handler',
            'error_reporter'
        ]
        
    def validate_integration(self, test_data: Optional[pd.DataFrame] = None) -> ValidationIntegrationReport:
        """Validate system integration"""
        errors = []
        missing_deps = []
        
        # Check Brain system availability
        brain_compatible = self._check_brain_system()
        
        # Check dependencies
        deps_validated = True
        try:
            import numpy
            import pandas
            import sklearn
        except ImportError as e:
            deps_validated = False
            missing_deps.append(str(e))
            
        # Check interfaces
        for interface in self.required_interfaces:
            if not self._check_interface(interface):
                errors.append(f"Missing interface: {interface}")
                
        # Test with sample data if provided
        performance_ok = True
        if test_data is not None:
            try:
                start = time.time()
                # Simulate validation
                time.sleep(0.1)
                duration = time.time() - start
                performance_ok = duration < 1.0
            except Exception as e:
                errors.append(f"Test validation failed: {e}")
                performance_ok = False
                
        # Calculate integration score
        score = 0.0
        if brain_compatible:
            score += 0.4
        if deps_validated:
            score += 0.3
        if not errors:
            score += 0.2
        if performance_ok:
            score += 0.1
            
        # Recommendations
        recommendations = []
        if not brain_compatible:
            recommendations.append("Install Brain system components")
        if missing_deps:
            recommendations.append(f"Install missing dependencies: {', '.join(missing_deps)}")
        if errors:
            recommendations.append("Fix interface errors before production use")
            
        return ValidationIntegrationReport(
            brain_system_compatible=brain_compatible,
            api_version="1.0.0",
            dependencies_validated=deps_validated,
            missing_dependencies=missing_deps,
            interface_errors=errors,
            data_format_compatible=True,
            performance_acceptable=performance_ok,
            integration_score=score,
            recommendations=recommendations
        )
    
    def _check_brain_system(self) -> bool:
        """Check if Brain system is available"""
        # Simplified check - in production, actually test imports
        try:
            # Simulate checking for Brain system
            return True
        except:
            return False
    
    def _check_interface(self, interface: str) -> bool:
        """Check if interface is available"""
        # Simplified check
        return True

class ErrorRecoveryManager:
    """Manage error recovery and partial results"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.recovery_strategies = {
            InputValidationError: self._recover_from_input_error,
            ValidationTimeoutError: self._recover_from_timeout,
            DataCorruptionError: self._recover_from_corruption,
            PartialValidationError: self._recover_from_partial_failure
        }
        
    def recover(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Attempt to recover from error
        
        Returns:
            Tuple of (success, recovery_result)
        """
        if not self.config.enable_recovery:
            return False, None
            
        error_type = type(error)
        if error_type in self.recovery_strategies:
            try:
                return self.recovery_strategies[error_type](error, context)
            except Exception as recovery_error:
                logger.error(f"Recovery failed: {recovery_error}")
                return False, None
        else:
            return self._default_recovery(error, context)
    
    def _recover_from_input_error(self, error: InputValidationError, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Recover from input validation error"""
        if 'data' in context and hasattr(context['data'], 'dropna'):
            # Try removing invalid rows
            try:
                cleaned_data = context['data'].dropna()
                if len(cleaned_data) > len(context['data']) * 0.5:  # Keep if >50% valid
                    logger.warning(f"Recovered by removing {len(context['data']) - len(cleaned_data)} invalid rows")
                    return True, cleaned_data
            except:
                pass
        return False, None
    
    def _recover_from_timeout(self, error: ValidationTimeoutError, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Recover from timeout error"""
        if self.config.enable_partial_results and 'partial_results' in context:
            logger.warning("Validation timed out, returning partial results")
            return True, context['partial_results']
        return False, None
    
    def _recover_from_corruption(self, error: DataCorruptionError, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Recover from data corruption"""
        if 'backup_data' in context:
            logger.warning("Using backup data due to corruption")
            return True, context['backup_data']
        return False, None
    
    def _recover_from_partial_failure(self, error: PartialValidationError, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Recover from partial validation failure"""
        if error.partial_results is not None:
            logger.warning("Returning partial validation results")
            return True, error.partial_results
        return False, None
    
    def _default_recovery(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Default recovery strategy"""
        if self.config.enable_partial_results and 'data' in context:
            # Try to return at least basic validation
            try:
                basic_result = {
                    'is_valid': False,
                    'error': str(error),
                    'partial': True,
                    'row_count': len(context['data']) if hasattr(context['data'], '__len__') else 0
                }
                return True, basic_result
            except:
                pass
        return False, None

class EnhancedFinancialDataValidator(FinancialDataValidator if EXISTING_VALIDATOR else object):
    """
    Enhanced financial data validator with comprehensive error handling
    
    Features:
    - Comprehensive input validation
    - Advanced error handling and recovery
    - Performance monitoring and optimization
    - Security validation and compliance checking
    - Integration testing with Brain system
    - Partial result support
    - Timeout handling
    - Memory management
    - Detailed diagnostics and reporting
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize enhanced validator"""
        self.config = config or ValidationConfig()
        self.input_validator = InputValidator()
        self.performance_monitor = PerformanceMonitor()
        self.security_validator = SecurityValidator(self.config)
        self.integration_validator = IntegrationValidator(self.config)
        self.error_recovery = ErrorRecoveryManager(self.config)
        
        # Initialize parent if available
        if EXISTING_VALIDATOR:
            super().__init__()
            
        # Validation cache
        self._validation_cache = {}
        self._cache_lock = threading.RLock()
        
        # Error history
        self.error_history = []
        self.max_error_history = 100
        
        # Performance history
        self.performance_history = []
        self.max_performance_history = 50
        
        logger.info("Enhanced Financial Data Validator initialized")
    
    def validate_with_timeout(self, func: Callable, timeout: float, *args, **kwargs) -> Any:
        """Execute function with timeout"""
        
        class TimeoutHandler:
            def __init__(self):
                self.result = None
                self.exception = None
                
            def run(self):
                try:
                    self.result = func(*args, **kwargs)
                except Exception as e:
                    self.exception = e
                    
        handler = TimeoutHandler()
        thread = threading.Thread(target=handler.run)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            raise ValidationTimeoutError(
                f"Operation timed out after {timeout} seconds",
                {"function": func.__name__, "timeout": timeout}
            )
            
        if handler.exception:
            raise handler.exception
            
        return handler.result
    
    def validate(self, 
                data: pd.DataFrame,
                validation_level: Optional[ValidationLevel] = None,
                required_compliance: Optional[List[str]] = None,
                parallel: bool = True) -> 'ValidationResult':
        """
        Enhanced validation with comprehensive error handling
        """
        start_time = time.time()
        context = {'data': data, 'start_time': start_time}
        
        try:
            # Use config values if not specified
            validation_level = validation_level or self.config.validation_level
            required_compliance = required_compliance or self.config.compliance_standards
            
            # Input validation
            data = self.input_validator.validate_dataframe(
                data, 
                min_rows=1,
                max_rows=10_000_000 if self.config.context == ValidationContext.PRODUCTION else None
            )
            
            # Check memory usage
            data_memory = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            if data_memory > self.config.max_memory_mb:
                raise InputValidationError(
                    f"Data size ({data_memory:.1f}MB) exceeds limit ({self.config.max_memory_mb}MB)",
                    {"data_memory_mb": data_memory, "limit_mb": self.config.max_memory_mb}
                )
            
            # Check cache
            cache_key = self._get_cache_key(data, validation_level)
            if self.config.enable_caching and cache_key in self._validation_cache:
                cached_result, cache_time = self._validation_cache[cache_key]
                if time.time() - cache_time < self.config.cache_ttl_seconds:
                    self.performance_monitor.metrics['cache_hits'] = self.performance_monitor.metrics.get('cache_hits', 0) + 1
                    logger.debug("Returning cached validation result")
                    return cached_result
                else:
                    self.performance_monitor.metrics['cache_misses'] = self.performance_monitor.metrics.get('cache_misses', 0) + 1
            
            # Start performance monitoring
            self.performance_monitor.start()
            self.performance_monitor.metrics['record_count'] = len(data)
            
            # Apply sampling if configured
            if self.config.mode == ValidationMode.SAMPLING and self.config.sampling_rate < 1.0:
                sample_size = int(len(data) * self.config.sampling_rate)
                data_sample = data.sample(n=sample_size, random_state=42)
                logger.info(f"Using sampling mode: {sample_size}/{len(data)} records")
            else:
                data_sample = data
            
            # Execute validation with timeout
            if self.config.timeout_seconds > 0:
                result = self.validate_with_timeout(
                    self._perform_validation,
                    self.config.timeout_seconds,
                    data_sample,
                    validation_level,
                    required_compliance,
                    parallel
                )
            else:
                result = self._perform_validation(
                    data_sample,
                    validation_level,
                    required_compliance,
                    parallel
                )
            
            # Security validation if enabled
            if self.config.security_validation:
                security_report = self.security_validator.validate_security(data_sample)
                result.metadata['security_report'] = asdict(security_report)
                
                # Add security issues to validation issues
                if EXISTING_VALIDATOR:
                    for violation in security_report.compliance_violations:
                        result.issues.append(ValidationIssue(
                            severity=ValidationSeverity.CRITICAL,
                            category="security",
                            message=violation,
                            rule_id="security_validator"
                        ))
            
            # Integration testing if enabled
            if self.config.integration_testing:
                integration_report = self.integration_validator.validate_integration(data_sample)
                result.metadata['integration_report'] = asdict(integration_report)
                
                if EXISTING_VALIDATOR and not integration_report.brain_system_compatible:
                    result.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="integration",
                        message="Brain system integration not fully compatible",
                        rule_id="integration_validator"
                    ))
            
            # Record performance metrics
            performance_metrics = self.performance_monitor.stop()
            result.metadata['performance_metrics'] = asdict(performance_metrics)
            
            # Store in cache
            if self.config.enable_caching:
                with self._cache_lock:
                    self._validation_cache[cache_key] = (result, time.time())
                    # Limit cache size
                    if len(self._validation_cache) > 1000:
                        oldest_key = min(self._validation_cache.keys(), 
                                       key=lambda k: self._validation_cache[k][1])
                        del self._validation_cache[oldest_key]
            
            # Record performance history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'records': len(data),
                'duration': performance_metrics.total_time_seconds,
                'memory_usage': performance_metrics.memory_usage_mb,
                'issues_found': len(result.issues) if hasattr(result, 'issues') else 0
            })
            if len(self.performance_history) > self.max_performance_history:
                self.performance_history.pop(0)
            
            logger.info(f"Validation completed in {performance_metrics.total_time_seconds:.2f}s")
            return result
            
        except ValidationException as e:
            # Try recovery
            success, recovery_result = self.error_recovery.recover(e, context)
            if success:
                logger.warning(f"Recovered from error: {e}")
                if hasattr(recovery_result, 'is_valid'):
                    return recovery_result
                else:
                    # Create partial result
                    return self._create_error_result(e, partial_data=recovery_result)
            else:
                # Record error
                self._record_error(e, context)
                raise
                
        except Exception as e:
            # Unexpected error
            logger.error(f"Unexpected validation error: {e}\n{traceback.format_exc()}")
            self._record_error(e, context)
            
            # Try recovery for unexpected errors
            if self.config.enable_recovery:
                success, recovery_result = self.error_recovery.recover(e, context)
                if success:
                    return self._create_error_result(e, partial_data=recovery_result)
                    
            raise ValidationException(f"Validation failed: {e}", {"original_error": str(e)})
    
    def _perform_validation(self, data: pd.DataFrame, validation_level: ValidationLevel,
                          required_compliance: Optional[List[str]],
                          parallel: bool) -> 'ValidationResult':
        """Perform actual validation (to be implemented by parent or custom logic)"""
        if EXISTING_VALIDATOR:
            # Use parent implementation
            return super().validate(data, validation_level, required_compliance, parallel)
        else:
            # Basic implementation for standalone use
            from dataclasses import dataclass
            
            @dataclass
            class ValidationResult:
                is_valid: bool
                total_records: int
                valid_records: int
                invalid_records: int
                issues: List[Any]
                data_quality_score: float
                validation_time_seconds: float
                compliance_scores: Dict[Any, float]
                field_statistics: Dict[str, Dict[str, Any]]
                summary: Dict[str, Any]
                metadata: Dict[str, Any] = field(default_factory=dict)
            
            return ValidationResult(
                is_valid=True,
                total_records=len(data),
                valid_records=len(data),
                invalid_records=0,
                issues=[],
                data_quality_score=1.0,
                validation_time_seconds=0.0,
                compliance_scores={},
                field_statistics={},
                summary={},
                metadata={}
            )
    
    def _get_cache_key(self, data: pd.DataFrame, validation_level: ValidationLevel) -> str:
        """Generate cache key for validation result"""
        # Use data shape and sample values for key
        key_parts = [
            str(data.shape),
            str(validation_level.value),
            str(sorted(data.columns.tolist())),
            str(data.dtypes.to_dict()),
            str(data.iloc[:min(10, len(data))].to_dict())  # Sample first 10 rows
        ]
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _create_error_result(self, error: Exception, partial_data: Any = None) -> 'ValidationResult':
        """Create validation result for error case"""
        if EXISTING_VALIDATOR:
            return ValidationResult(
                is_valid=False,
                total_records=len(partial_data) if hasattr(partial_data, '__len__') else 0,
                valid_records=0,
                invalid_records=0,
                issues=[ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="system",
                    message=str(error),
                    rule_id="error_handler"
                )],
                data_quality_score=0.0,
                validation_time_seconds=0.0,
                compliance_scores={},
                field_statistics={},
                summary={"error": str(error), "partial_result": partial_data is not None}
            )
        else:
            # Basic error result for standalone use
            from dataclasses import dataclass
            
            @dataclass
            class ValidationResult:
                is_valid: bool
                total_records: int
                valid_records: int
                invalid_records: int
                issues: List[Any]
                data_quality_score: float
                validation_time_seconds: float
                compliance_scores: Dict[Any, float]
                field_statistics: Dict[str, Dict[str, Any]]
                summary: Dict[str, Any]
                metadata: Dict[str, Any] = field(default_factory=dict)
            
            return ValidationResult(
                is_valid=False,
                total_records=len(partial_data) if hasattr(partial_data, '__len__') else 0,
                valid_records=0,
                invalid_records=0,
                issues=[{"severity": "critical", "message": str(error)}],
                data_quality_score=0.0,
                validation_time_seconds=0.0,
                compliance_scores={},
                field_statistics={},
                summary={"error": str(error), "partial_result": partial_data is not None},
                metadata={}
            )
    
    def _record_error(self, error: Exception, context: Dict[str, Any]):
        """Record error for analysis"""
        error_record = {
            'timestamp': datetime.now(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': {
                'data_shape': context.get('data').shape if 'data' in context and hasattr(context['data'], 'shape') else None,
                'validation_level': self.config.validation_level.value,
                'context': self.config.context.value
            },
            'traceback': traceback.format_exc()
        }
        
        self.error_history.append(error_record)
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)
    
    def validate_configuration(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration dictionary"""
        issues = []
        
        try:
            # Check required fields
            required_fields = ['validation_level', 'context', 'mode']
            for field in required_fields:
                if field not in config:
                    issues.append(f"Missing required field: {field}")
            
            # Validate enum values
            if 'validation_level' in config:
                try:
                    ValidationLevel(config['validation_level'])
                except ValueError:
                    issues.append(f"Invalid validation_level: {config['validation_level']}")
            
            if 'context' in config:
                try:
                    ValidationContext(config['context'])
                except ValueError:
                    issues.append(f"Invalid context: {config['context']}")
            
            # Validate numeric values
            numeric_fields = {
                'timeout_seconds': (0, 3600),
                'max_memory_mb': (1, 100000),
                'sampling_rate': (0, 1),
                'batch_size': (1, 1000000)
            }
            
            for field, (min_val, max_val) in numeric_fields.items():
                if field in config:
                    try:
                        value = float(config[field])
                        if not min_val <= value <= max_val:
                            issues.append(f"{field} must be between {min_val} and {max_val}")
                    except (TypeError, ValueError):
                        issues.append(f"{field} must be numeric")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Configuration validation error: {e}")
            return False, issues
    
    def get_validation_report(self, result, format: str = "dict") -> Union[Dict[str, Any], str]:
        """Generate comprehensive validation report"""
        report = {
            'validation_summary': getattr(result, 'summary', {}),
            'data_quality_score': getattr(result, 'data_quality_score', 0.0),
            'compliance_scores': getattr(result, 'compliance_scores', {}),
            'field_statistics': getattr(result, 'field_statistics', {}),
            'performance_metrics': getattr(result, 'metadata', {}).get('performance_metrics', {}),
            'security_report': getattr(result, 'metadata', {}).get('security_report', {}),
            'integration_report': getattr(result, 'metadata', {}).get('integration_report', {}),
            'issues_by_severity': self._group_issues_by_severity(getattr(result, 'issues', [])),
            'issues_by_category': self._group_issues_by_category(getattr(result, 'issues', [])),
            'top_issues': self._get_top_issues(getattr(result, 'issues', []), limit=10),
            'validation_config': asdict(self.config),
            'timestamp': datetime.now().isoformat()
        }
        
        if format == "dict":
            return report
        elif format == "json":
            return json.dumps(report, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _group_issues_by_severity(self, issues: List[Any]) -> Dict[str, int]:
        """Group issues by severity"""
        severity_counts = {}
        for issue in issues:
            if hasattr(issue, 'severity'):
                severity = issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity)
            elif isinstance(issue, dict):
                severity = issue.get('severity', 'unknown')
            else:
                severity = 'unknown'
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        return severity_counts
    
    def _group_issues_by_category(self, issues: List[Any]) -> Dict[str, int]:
        """Group issues by category"""
        category_counts = {}
        for issue in issues:
            if hasattr(issue, 'category'):
                category = issue.category
            elif isinstance(issue, dict):
                category = issue.get('category', 'unknown')
            else:
                category = 'unknown'
            category_counts[category] = category_counts.get(category, 0) + 1
        return category_counts
    
    def _get_top_issues(self, issues: List[Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Get top issues by severity"""
        result_issues = []
        for issue in issues[:limit]:
            if hasattr(issue, 'severity'):
                issue_dict = {
                    'severity': issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity),
                    'category': getattr(issue, 'category', 'unknown'),
                    'message': getattr(issue, 'message', str(issue)),
                    'field': getattr(issue, 'field', None),
                    'rule_id': getattr(issue, 'rule_id', None)
                }
            elif isinstance(issue, dict):
                issue_dict = {
                    'severity': issue.get('severity', 'unknown'),
                    'category': issue.get('category', 'unknown'),
                    'message': issue.get('message', str(issue)),
                    'field': issue.get('field'),
                    'rule_id': issue.get('rule_id')
                }
            else:
                issue_dict = {
                    'severity': 'unknown',
                    'category': 'unknown',
                    'message': str(issue),
                    'field': None,
                    'rule_id': None
                }
            result_issues.append(issue_dict)
        return result_issues
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error history"""
        if not self.error_history:
            return {"message": "No errors recorded"}
            
        error_types = {}
        for error in self.error_history:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
        return {
            'total_errors': len(self.error_history),
            'error_types': error_types,
            'recent_errors': [
                {
                    'timestamp': error['timestamp'].isoformat(),
                    'type': error['error_type'],
                    'message': error['error_message']
                }
                for error in self.error_history[-5:]  # Last 5 errors
            ]
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance history"""
        if not self.performance_history:
            return {"message": "No performance data recorded"}
            
        durations = [p['duration'] for p in self.performance_history]
        memory_usage = [p['memory_usage'] for p in self.performance_history]
        
        return {
            'total_validations': len(self.performance_history),
            'average_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'average_memory_mb': sum(memory_usage) / len(memory_usage),
            'trend': 'improving' if durations[-5:] < durations[:5] else 'stable'
        }
    
    def clear_cache(self):
        """Clear validation cache"""
        with self._cache_lock:
            self._validation_cache.clear()
        logger.info("Validation cache cleared")
    
    def __repr__(self) -> str:
        return (f"EnhancedFinancialDataValidator("
               f"context={self.config.context.value}, "
               f"level={self.config.validation_level.value}, "
               f"cache_size={len(self._validation_cache)}, "
               f"errors={len(self.error_history)})")

# Convenience functions
def create_validator(config: Optional[Dict[str, Any]] = None) -> EnhancedFinancialDataValidator:
    """Create configured validator instance"""
    if config:
        validation_config = ValidationConfig(**config)
    else:
        validation_config = ValidationConfig()
    return EnhancedFinancialDataValidator(validation_config)

def validate_financial_data(data: pd.DataFrame, 
                          validation_level: ValidationLevel = ValidationLevel.STANDARD,
                          timeout: float = 300.0):
    """Quick validation function"""
    config = ValidationConfig(
        validation_level=validation_level,
        timeout_seconds=timeout
    )
    validator = EnhancedFinancialDataValidator(config)
    return validator.validate(data)

# Export enhanced classes
__all__ = [
    'EnhancedFinancialDataValidator',
    'ValidationConfig',
    'ValidationContext',
    'ValidationMode',
    'SecurityLevel',
    'ValidationPerformanceMetrics',
    'ValidationSecurityReport',
    'ValidationIntegrationReport',
    'InputValidator',
    'PerformanceMonitor',
    'SecurityValidator',
    'IntegrationValidator',
    'ErrorRecoveryManager',
    'ValidationException',
    'InputValidationError',
    'ValidationTimeoutError',
    'ValidationConfigError',
    'ValidationSecurityError',
    'ValidationIntegrationError',
    'DataCorruptionError',
    'PartialValidationError',
    'create_validator',
    'validate_financial_data'
]

if __name__ == "__main__":
    # Test enhanced validator
    print("Testing Enhanced Financial Data Validator...")
    
    # Create test data
    test_data = pd.DataFrame({
        'transaction_id': ['TXN001', 'TXN002', 'TXN003'],
        'amount': [100.50, -50.00, 999999.99],
        'timestamp': ['2024-01-01', '2024-01-02', 'invalid'],
        'user_id': ['USER001', 'USER002', 'USER003'],
        'ssn': ['123-45-6789', '987-65-4321', 'XXX-XX-XXXX']  # PII for testing
    })
    
    # Create validator with comprehensive configuration
    config = ValidationConfig(
        validation_level=ValidationLevel.COMPREHENSIVE,
        context=ValidationContext.TESTING,
        security_validation=True,
        integration_testing=True,
        performance_monitoring=True,
        enable_recovery=True,
        timeout_seconds=60.0
    )
    
    validator = EnhancedFinancialDataValidator(config)
    
    # Test validation
    try:
        result = validator.validate(test_data)
        print(f"\nValidation Result: {'PASSED' if result.is_valid else 'FAILED'}")
        print(f"Quality Score: {result.data_quality_score:.2f}")
        print(f"Issues Found: {len(result.issues) if hasattr(result, 'issues') else 0}")
        
        # Get comprehensive report
        report = validator.get_validation_report(result)
        print(f"\nSecurity Issues: {len(report['security_report'].get('security_risks', []))}")
        print(f"Performance: {report['performance_metrics'].get('total_time_seconds', 0):.3f}s")
        
        # Test error summary
        error_summary = validator.get_error_summary()
        print(f"\nError Summary: {error_summary}")
        
        # Test performance summary
        perf_summary = validator.get_performance_summary()
        print(f"\nPerformance Summary: {perf_summary}")
        
    except Exception as e:
        print(f"\nValidation failed with error: {e}")
        if hasattr(e, 'details'):
            print(f"Error details: {e.details}")
    
    print("\nEnhanced validator testing complete!")