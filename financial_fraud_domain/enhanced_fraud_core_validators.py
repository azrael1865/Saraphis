"""
Enhanced Fraud Detection Core - Chunk 2: Validation Framework with Specialized Validators
Comprehensive validation framework for the enhanced fraud detection system
"""

import logging
import time
import threading
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from functools import wraps
import traceback

# Import core exceptions and enums
try:
    from enhanced_fraud_core_exceptions import (
        EnhancedFraudException, ValidationError, ConfigurationError, SecurityError,
        PerformanceError, TransactionValidationError, StrategyValidationError,
        ComponentValidationError, PerformanceValidationError, SecurityValidationError,
        DetectionStrategy, ValidationLevel, SecurityLevel, AlertSeverity,
        ErrorContext, create_error_context, log_exception
    )
except ImportError:
    try:
        from enhanced_fraud_core_exceptions import (
            EnhancedFraudException, ValidationError, ConfigurationError, SecurityError,
            PerformanceError, TransactionValidationError, StrategyValidationError,
            ComponentValidationError, PerformanceValidationError, SecurityValidationError,
            DetectionStrategy, ValidationLevel, SecurityLevel, AlertSeverity,
            ErrorContext, create_error_context, log_exception
        )
    except ImportError:
        # Fallback definitions
        class EnhancedFraudException(Exception): pass
        class ValidationError(EnhancedFraudException): pass
        class ConfigurationError(EnhancedFraudException): pass
        class SecurityError(EnhancedFraudException): pass
        class PerformanceError(EnhancedFraudException): pass
        class TransactionValidationError(ValidationError): pass
        class StrategyValidationError(ValidationError): pass
        class ComponentValidationError(ValidationError): pass
        class PerformanceValidationError(ValidationError): pass
        class SecurityValidationError(ValidationError): pass
        
        class DetectionStrategy:
            RULES_ONLY = "rules_only"
            ML_ONLY = "ml_only"
            HYBRID = "hybrid"
            ENSEMBLE = "ensemble"
        
        class ValidationLevel:
            MINIMAL = "minimal"
            STANDARD = "standard"
            STRICT = "strict"
            PARANOID = "paranoid"
        
        class SecurityLevel:
            LOW = "low"
            MEDIUM = "medium"
            HIGH = "high"
        
        class AlertSeverity:
            INFO = "info"
            WARNING = "warning"
            ERROR = "error"
            CRITICAL = "critical"
        
        def create_error_context(**kwargs):
            return kwargs
        
        def log_exception(e, context=None):
            logger.error(f"Exception: {e}")
        
        class ErrorContext:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

# Configure logging
logger = logging.getLogger(__name__)

# ======================== VALIDATION CONFIGURATION ========================

@dataclass
class ValidationConfig:
    """Configuration for validation framework"""
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    enable_async_validation: bool = True
    validation_timeout: float = 30.0
    max_validation_threads: int = 4
    enable_performance_validation: bool = True
    enable_security_validation: bool = True
    enable_data_quality_validation: bool = True
    enable_preprocessing_validation: bool = True  # New: Preprocessing validation
    validation_cache_size: int = 1000
    validation_cache_ttl: int = 300  # 5 minutes
    
    # Thresholds
    performance_threshold_ms: float = 1000.0
    data_quality_threshold: float = 0.9
    security_score_threshold: float = 0.8
    preprocessing_quality_threshold: float = 0.8  # New: Preprocessing quality threshold
    
    # Preprocessing validation settings
    preprocessing_config: dict = field(default_factory=lambda: {
        'feature_engineering': {
            'min_features': 10,
            'max_features': 1000,
            'required_feature_types': ['amount', 'time', 'merchant'],
            'validate_feature_ranges': True
        },
        'data_quality': {
            'max_missing_ratio': 0.3,
            'outlier_detection_enabled': True,
            'duplicate_detection_enabled': True
        },
        'feature_selection': {
            'validate_correlation_matrix': True,
            'max_correlation_threshold': 0.99,
            'min_feature_variance': 0.01
        },
        'scaling': {
            'validate_scale_ranges': True,
            'expected_scale_method': 'standard'
        }
    })
    
    # Retry settings
    max_validation_retries: int = 3
    validation_retry_delay: float = 1.0
    
    # Logging settings
    log_validation_results: bool = True
    log_performance_metrics: bool = True

# ======================== BASE VALIDATOR ========================

class BaseValidator(ABC):
    """Abstract base class for all validators"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.validation_cache = {}
        self.performance_metrics = {}
        self.lock = threading.Lock()
        
    @abstractmethod
    def validate(self, data: Any, context: Optional[ErrorContext] = None) -> Dict[str, Any]:
        """Validate data and return results"""
        pass
    
    @abstractmethod
    def get_validation_rules(self) -> List[str]:
        """Get list of validation rules"""
        pass
    
    def clear_cache(self) -> None:
        """Clear validation cache"""
        with self.lock:
            self.validation_cache.clear()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get validation performance metrics"""
        with self.lock:
            return self.performance_metrics.copy()
    
    def _cache_key(self, data: Any) -> str:
        """Generate cache key for data"""
        return str(hash(str(data)))
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid"""
        if cache_key not in self.validation_cache:
            return False
        
        entry = self.validation_cache[cache_key]
        if time.time() - entry['timestamp'] > self.config.validation_cache_ttl:
            return False
        
        return True
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache validation result"""
        with self.lock:
            if len(self.validation_cache) >= self.config.validation_cache_size:
                # Remove oldest entry
                oldest_key = min(self.validation_cache.keys(), 
                               key=lambda k: self.validation_cache[k]['timestamp'])
                del self.validation_cache[oldest_key]
            
            self.validation_cache[cache_key] = {
                'result': result,
                'timestamp': time.time()
            }
    
    def _record_performance(self, validation_type: str, duration: float) -> None:
        """Record validation performance"""
        if not self.config.log_performance_metrics:
            return
        
        with self.lock:
            if validation_type not in self.performance_metrics:
                self.performance_metrics[validation_type] = {
                    'count': 0,
                    'total_time': 0.0,
                    'avg_time': 0.0,
                    'min_time': float('inf'),
                    'max_time': 0.0
                }
            
            metrics = self.performance_metrics[validation_type]
            metrics['count'] += 1
            metrics['total_time'] += duration
            metrics['avg_time'] = metrics['total_time'] / metrics['count']
            metrics['min_time'] = min(metrics['min_time'], duration)
            metrics['max_time'] = max(metrics['max_time'], duration)

# ======================== TRANSACTION VALIDATOR ========================

class TransactionValidator(BaseValidator):
    """Validator for transaction data"""
    
    def __init__(self, config: ValidationConfig):
        super().__init__(config)
        self.required_fields = [
            'transaction_id', 'user_id', 'amount', 'timestamp', 'merchant_id'
        ]
        self.numeric_fields = ['amount', 'account_balance']
        self.string_fields = ['transaction_id', 'user_id', 'merchant_id']
        self.date_fields = ['timestamp']
    
    def validate(self, transaction: Dict[str, Any], context: Optional[ErrorContext] = None) -> Dict[str, Any]:
        """Validate transaction data"""
        start_time = time.time()
        cache_key = self._cache_key(transaction)
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            result = self.validation_cache[cache_key]['result']
            self._record_performance('transaction_validation', time.time() - start_time)
            return result
        
        validation_errors = []
        warnings = []
        
        try:
            # Required fields validation
            missing_fields = self._validate_required_fields(transaction)
            if missing_fields:
                validation_errors.extend([f"Missing required field: {field}" for field in missing_fields])
            
            # Field type validation
            type_errors = self._validate_field_types(transaction)
            validation_errors.extend(type_errors)
            
            # Business logic validation
            business_errors, business_warnings = self._validate_business_logic(transaction)
            validation_errors.extend(business_errors)
            warnings.extend(business_warnings)
            
            # Data quality validation
            if self.config.enable_data_quality_validation:
                quality_errors, quality_warnings = self._validate_data_quality(transaction)
                validation_errors.extend(quality_errors)
                warnings.extend(quality_warnings)
            
            # Security validation
            if self.config.enable_security_validation:
                security_errors = self._validate_security(transaction)
                validation_errors.extend(security_errors)
            
            result = {
                'is_valid': len(validation_errors) == 0,
                'errors': validation_errors,
                'warnings': warnings,
                'validation_level': self.config.validation_level.value,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache result
            self._cache_result(cache_key, result)
            
            # Log validation result
            if self.config.log_validation_results:
                logger.info(f"Transaction validation completed: {result['is_valid']}")
            
            self._record_performance('transaction_validation', time.time() - start_time)
            return result
            
        except Exception as e:
            error_context = context or create_error_context(
                component='TransactionValidator',
                operation='validate'
            )
            raise TransactionValidationError(
                f"Transaction validation failed: {str(e)}",
                transaction_id=transaction.get('transaction_id'),
                validation_failures=validation_errors,
                context=error_context
            )
    
    def _validate_required_fields(self, transaction: Dict[str, Any]) -> List[str]:
        """Validate required fields are present"""
        missing_fields = []
        for field in self.required_fields:
            if field not in transaction or transaction[field] is None:
                missing_fields.append(field)
        return missing_fields
    
    def _validate_field_types(self, transaction: Dict[str, Any]) -> List[str]:
        """Validate field data types"""
        type_errors = []
        
        # Validate numeric fields
        for field in self.numeric_fields:
            if field in transaction:
                try:
                    float(transaction[field])
                except (ValueError, TypeError):
                    type_errors.append(f"Field '{field}' must be numeric")
        
        # Validate string fields
        for field in self.string_fields:
            if field in transaction and not isinstance(transaction[field], str):
                type_errors.append(f"Field '{field}' must be string")
        
        # Validate date fields
        for field in self.date_fields:
            if field in transaction:
                try:
                    if isinstance(transaction[field], str):
                        datetime.fromisoformat(transaction[field].replace('Z', '+00:00'))
                    elif not isinstance(transaction[field], datetime):
                        type_errors.append(f"Field '{field}' must be datetime or ISO string")
                except ValueError:
                    type_errors.append(f"Field '{field}' has invalid date format")
        
        return type_errors
    
    def _validate_business_logic(self, transaction: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate business logic rules"""
        errors = []
        warnings = []
        
        # Amount validation
        if 'amount' in transaction:
            amount = float(transaction['amount'])
            if amount <= 0:
                errors.append("Transaction amount must be positive")
            elif amount > 1000000:  # 1 million threshold
                warnings.append("Large transaction amount detected")
        
        # Timestamp validation
        if 'timestamp' in transaction:
            try:
                if isinstance(transaction['timestamp'], str):
                    tx_time = datetime.fromisoformat(transaction['timestamp'].replace('Z', '+00:00'))
                else:
                    tx_time = transaction['timestamp']
                
                now = datetime.now()
                if tx_time > now:
                    errors.append("Transaction timestamp cannot be in the future")
                elif (now - tx_time).days > 365:
                    warnings.append("Transaction timestamp is very old")
                    
            except Exception:
                errors.append("Invalid timestamp format")
        
        return errors, warnings
    
    def _validate_data_quality(self, transaction: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate data quality"""
        errors = []
        warnings = []
        
        # Check for suspicious patterns
        if 'description' in transaction:
            description = transaction['description'].lower()
            suspicious_keywords = ['test', 'dummy', 'fake', 'sample']
            if any(keyword in description for keyword in suspicious_keywords):
                warnings.append("Suspicious transaction description detected")
        
        # Check for duplicate transaction IDs (would need database lookup in real implementation)
        # This is a placeholder for actual duplicate detection
        
        return errors, warnings
    
    def _validate_security(self, transaction: Dict[str, Any]) -> List[str]:
        """Validate security aspects"""
        errors = []
        
        # Check for potential injection attacks
        string_fields = ['description', 'merchant_name', 'category']
        for field in string_fields:
            if field in transaction:
                value = str(transaction[field])
                if any(pattern in value.lower() for pattern in ['<script', 'javascript:', 'drop table']):
                    errors.append(f"Potential security threat detected in field '{field}'")
        
        return errors
    
    def get_validation_rules(self) -> List[str]:
        """Get list of validation rules"""
        return [
            "Required fields validation",
            "Field type validation",
            "Business logic validation",
            "Data quality validation",
            "Security validation"
        ]

# ======================== STRATEGY VALIDATOR ========================

class StrategyValidator(BaseValidator):
    """Validator for detection strategies"""
    
    def __init__(self, config: ValidationConfig):
        super().__init__(config)
        self.valid_strategies = [strategy.value for strategy in DetectionStrategy]
    
    def validate(self, strategy_config: Dict[str, Any], context: Optional[ErrorContext] = None) -> Dict[str, Any]:
        """Validate detection strategy configuration"""
        start_time = time.time()
        cache_key = self._cache_key(strategy_config)
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            result = self.validation_cache[cache_key]['result']
            self._record_performance('strategy_validation', time.time() - start_time)
            return result
        
        validation_errors = []
        warnings = []
        
        try:
            # Strategy type validation
            if 'strategy' not in strategy_config:
                validation_errors.append("Strategy type is required")
            elif strategy_config['strategy'] not in self.valid_strategies:
                validation_errors.append(f"Invalid strategy: {strategy_config['strategy']}")
            
            # Strategy-specific validation
            strategy_type = strategy_config.get('strategy')
            if strategy_type == DetectionStrategy.RULES_ONLY.value:
                errors, warns = self._validate_rules_strategy(strategy_config)
                validation_errors.extend(errors)
                warnings.extend(warns)
            elif strategy_type == DetectionStrategy.ML_ONLY.value:
                errors, warns = self._validate_ml_strategy(strategy_config)
                validation_errors.extend(errors)
                warnings.extend(warns)
            elif strategy_type == DetectionStrategy.HYBRID.value:
                errors, warns = self._validate_hybrid_strategy(strategy_config)
                validation_errors.extend(errors)
                warnings.extend(warns)
            
            result = {
                'is_valid': len(validation_errors) == 0,
                'errors': validation_errors,
                'warnings': warnings,
                'strategy': strategy_type,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache result
            self._cache_result(cache_key, result)
            
            self._record_performance('strategy_validation', time.time() - start_time)
            return result
            
        except Exception as e:
            error_context = context or create_error_context(
                component='StrategyValidator',
                operation='validate'
            )
            raise StrategyValidationError(
                f"Strategy validation failed: {str(e)}",
                strategy=DetectionStrategy(strategy_config.get('strategy')) if strategy_config.get('strategy') in self.valid_strategies else None,
                validation_failures=validation_errors,
                context=error_context
            )
    
    def _validate_rules_strategy(self, config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate rules-based strategy"""
        errors = []
        warnings = []
        
        if 'rules' not in config:
            errors.append("Rules configuration is required for rules-based strategy")
        else:
            rules = config['rules']
            if not isinstance(rules, list) or len(rules) == 0:
                errors.append("At least one rule must be specified")
            
            for i, rule in enumerate(rules):
                if not isinstance(rule, dict):
                    errors.append(f"Rule {i} must be a dictionary")
                elif 'condition' not in rule:
                    errors.append(f"Rule {i} must have a condition")
        
        return errors, warnings
    
    def _validate_ml_strategy(self, config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate ML-based strategy"""
        errors = []
        warnings = []
        
        if 'model' not in config:
            errors.append("Model configuration is required for ML strategy")
        else:
            model_config = config['model']
            if 'model_type' not in model_config:
                errors.append("Model type is required")
            if 'model_path' not in model_config:
                errors.append("Model path is required")
        
        return errors, warnings
    
    def _validate_hybrid_strategy(self, config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate hybrid strategy"""
        errors = []
        warnings = []
        
        # Must have both rules and ML components
        if 'rules' not in config:
            errors.append("Rules configuration is required for hybrid strategy")
        if 'model' not in config:
            errors.append("Model configuration is required for hybrid strategy")
        
        # Validate combination logic
        if 'combination_method' not in config:
            warnings.append("No combination method specified, using default")
        
        return errors, warnings
    
    def get_validation_rules(self) -> List[str]:
        """Get list of validation rules"""
        return [
            "Strategy type validation",
            "Rules-based strategy validation",
            "ML-based strategy validation",
            "Hybrid strategy validation",
            "Configuration completeness validation"
        ]

# ======================== COMPONENT VALIDATOR ========================

class ComponentValidator(BaseValidator):
    """Validator for system components"""
    
    def __init__(self, config: ValidationConfig):
        super().__init__(config)
        self.required_methods = ['initialize', 'process', 'cleanup']
    
    def validate(self, component: Any, context: Optional[ErrorContext] = None) -> Dict[str, Any]:
        """Validate system component"""
        start_time = time.time()
        
        validation_errors = []
        warnings = []
        
        try:
            # Interface validation
            interface_errors = self._validate_interface(component)
            validation_errors.extend(interface_errors)
            
            # Configuration validation
            config_errors = self._validate_configuration(component)
            validation_errors.extend(config_errors)
            
            # Health check validation
            health_errors, health_warnings = self._validate_health(component)
            validation_errors.extend(health_errors)
            warnings.extend(health_warnings)
            
            result = {
                'is_valid': len(validation_errors) == 0,
                'errors': validation_errors,
                'warnings': warnings,
                'component_type': type(component).__name__,
                'timestamp': datetime.now().isoformat()
            }
            
            self._record_performance('component_validation', time.time() - start_time)
            return result
            
        except Exception as e:
            error_context = context or create_error_context(
                component='ComponentValidator',
                operation='validate'
            )
            raise ComponentValidationError(
                f"Component validation failed: {str(e)}",
                component_name=type(component).__name__,
                validation_failures=validation_errors,
                context=error_context
            )
    
    def _validate_interface(self, component: Any) -> List[str]:
        """Validate component interface"""
        errors = []
        
        for method_name in self.required_methods:
            if not hasattr(component, method_name):
                errors.append(f"Component missing required method: {method_name}")
            elif not callable(getattr(component, method_name)):
                errors.append(f"Component method '{method_name}' is not callable")
        
        return errors
    
    def _validate_configuration(self, component: Any) -> List[str]:
        """Validate component configuration"""
        errors = []
        
        if hasattr(component, 'config'):
            config = component.config
            if config is None:
                errors.append("Component configuration is None")
            elif not isinstance(config, dict):
                errors.append("Component configuration must be a dictionary")
        else:
            errors.append("Component missing configuration attribute")
        
        return errors
    
    def _validate_health(self, component: Any) -> Tuple[List[str], List[str]]:
        """Validate component health"""
        errors = []
        warnings = []
        
        if hasattr(component, 'health_check'):
            try:
                health_result = component.health_check()
                if not health_result.get('healthy', False):
                    errors.append(f"Component health check failed: {health_result.get('message', 'Unknown error')}")
            except Exception as e:
                errors.append(f"Component health check raised exception: {str(e)}")
        else:
            warnings.append("Component does not implement health check")
        
        return errors, warnings
    
    def get_validation_rules(self) -> List[str]:
        """Get list of validation rules"""
        return [
            "Interface validation",
            "Configuration validation",
            "Health check validation"
        ]

# ======================== PERFORMANCE VALIDATOR ========================

class PerformanceValidator(BaseValidator):
    """Validator for performance metrics"""
    
    def __init__(self, config: ValidationConfig):
        super().__init__(config)
        self.performance_thresholds = {
            'response_time': config.performance_threshold_ms,
            'throughput': 100.0,  # transactions per second
            'error_rate': 0.05,   # 5% error rate
            'cpu_usage': 0.8,     # 80% CPU usage
            'memory_usage': 0.8   # 80% memory usage
        }
    
    def validate(self, metrics: Dict[str, Any], context: Optional[ErrorContext] = None) -> Dict[str, Any]:
        """Validate performance metrics"""
        start_time = time.time()
        
        validation_errors = []
        warnings = []
        
        try:
            # Response time validation
            if 'response_time' in metrics:
                if metrics['response_time'] > self.performance_thresholds['response_time']:
                    validation_errors.append(f"Response time {metrics['response_time']}ms exceeds threshold {self.performance_thresholds['response_time']}ms")
            
            # Throughput validation
            if 'throughput' in metrics:
                if metrics['throughput'] < self.performance_thresholds['throughput']:
                    warnings.append(f"Throughput {metrics['throughput']} below optimal threshold {self.performance_thresholds['throughput']}")
            
            # Error rate validation
            if 'error_rate' in metrics:
                if metrics['error_rate'] > self.performance_thresholds['error_rate']:
                    validation_errors.append(f"Error rate {metrics['error_rate']} exceeds threshold {self.performance_thresholds['error_rate']}")
            
            # Resource usage validation
            if 'cpu_usage' in metrics:
                if metrics['cpu_usage'] > self.performance_thresholds['cpu_usage']:
                    warnings.append(f"CPU usage {metrics['cpu_usage']} is high")
            
            if 'memory_usage' in metrics:
                if metrics['memory_usage'] > self.performance_thresholds['memory_usage']:
                    warnings.append(f"Memory usage {metrics['memory_usage']} is high")
            
            result = {
                'is_valid': len(validation_errors) == 0,
                'errors': validation_errors,
                'warnings': warnings,
                'performance_score': self._calculate_performance_score(metrics),
                'timestamp': datetime.now().isoformat()
            }
            
            self._record_performance('performance_validation', time.time() - start_time)
            return result
            
        except Exception as e:
            error_context = context or create_error_context(
                component='PerformanceValidator',
                operation='validate'
            )
            raise PerformanceValidationError(
                f"Performance validation failed: {str(e)}",
                context=error_context
            )
    
    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        score = 1.0
        
        # Response time score
        if 'response_time' in metrics:
            response_time_score = max(0, 1 - (metrics['response_time'] / self.performance_thresholds['response_time']))
            score *= response_time_score
        
        # Throughput score
        if 'throughput' in metrics:
            throughput_score = min(1, metrics['throughput'] / self.performance_thresholds['throughput'])
            score *= throughput_score
        
        # Error rate score
        if 'error_rate' in metrics:
            error_rate_score = max(0, 1 - (metrics['error_rate'] / self.performance_thresholds['error_rate']))
            score *= error_rate_score
        
        return score
    
    def get_validation_rules(self) -> List[str]:
        """Get list of validation rules"""
        return [
            "Response time validation",
            "Throughput validation",
            "Error rate validation",
            "Resource usage validation"
        ]

# ======================== SECURITY VALIDATOR ========================

class SecurityValidator(BaseValidator):
    """Validator for security aspects"""
    
    def __init__(self, config: ValidationConfig):
        super().__init__(config)
        self.security_checks = [
            'authentication', 'authorization', 'encryption',
            'input_validation', 'rate_limiting', 'audit_logging'
        ]
    
    def validate(self, security_context: Dict[str, Any], context: Optional[ErrorContext] = None) -> Dict[str, Any]:
        """Validate security aspects"""
        start_time = time.time()
        
        validation_errors = []
        warnings = []
        
        try:
            # Authentication validation
            auth_errors = self._validate_authentication(security_context)
            validation_errors.extend(auth_errors)
            
            # Authorization validation
            authz_errors = self._validate_authorization(security_context)
            validation_errors.extend(authz_errors)
            
            # Encryption validation
            encryption_errors, encryption_warnings = self._validate_encryption(security_context)
            validation_errors.extend(encryption_errors)
            warnings.extend(encryption_warnings)
            
            # Input validation
            input_errors = self._validate_input_security(security_context)
            validation_errors.extend(input_errors)
            
            result = {
                'is_valid': len(validation_errors) == 0,
                'errors': validation_errors,
                'warnings': warnings,
                'security_score': self._calculate_security_score(security_context),
                'timestamp': datetime.now().isoformat()
            }
            
            self._record_performance('security_validation', time.time() - start_time)
            return result
            
        except Exception as e:
            error_context = context or create_error_context(
                component='SecurityValidator',
                operation='validate'
            )
            raise SecurityValidationError(
                f"Security validation failed: {str(e)}",
                context=error_context
            )
    
    def _validate_authentication(self, security_context: Dict[str, Any]) -> List[str]:
        """Validate authentication"""
        errors = []
        
        if 'user_id' not in security_context:
            errors.append("User ID is required for authentication")
        
        if 'session_token' not in security_context:
            errors.append("Session token is required")
        elif len(security_context['session_token']) < 32:
            errors.append("Session token too short")
        
        return errors
    
    def _validate_authorization(self, security_context: Dict[str, Any]) -> List[str]:
        """Validate authorization"""
        errors = []
        
        if 'permissions' not in security_context:
            errors.append("Permissions are required for authorization")
        
        required_permissions = ['read_transactions', 'detect_fraud']
        user_permissions = security_context.get('permissions', [])
        
        for permission in required_permissions:
            if permission not in user_permissions:
                errors.append(f"Missing required permission: {permission}")
        
        return errors
    
    def _validate_encryption(self, security_context: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate encryption"""
        errors = []
        warnings = []
        
        if 'encryption_level' not in security_context:
            warnings.append("Encryption level not specified")
        elif security_context['encryption_level'] < 256:
            warnings.append("Encryption level below recommended 256-bit")
        
        return errors, warnings
    
    def _validate_input_security(self, security_context: Dict[str, Any]) -> List[str]:
        """Validate input security"""
        errors = []
        
        # Check for potential injection attacks
        if 'input_data' in security_context:
            input_data = str(security_context['input_data'])
            dangerous_patterns = ['<script', 'javascript:', 'drop table', 'union select']
            
            for pattern in dangerous_patterns:
                if pattern.lower() in input_data.lower():
                    errors.append(f"Potential injection attack detected: {pattern}")
        
        return errors
    
    def _calculate_security_score(self, security_context: Dict[str, Any]) -> float:
        """Calculate overall security score"""
        score = 0.0
        max_score = len(self.security_checks)
        
        # Authentication score
        if 'user_id' in security_context and 'session_token' in security_context:
            score += 1.0
        
        # Authorization score
        if 'permissions' in security_context:
            score += 1.0
        
        # Encryption score
        if security_context.get('encryption_level', 0) >= 256:
            score += 1.0
        
        # Add scores for other security checks
        score += min(3.0, len(security_context.get('security_features', [])))
        
        return score / max_score
    
    def get_validation_rules(self) -> List[str]:
        """Get list of validation rules"""
        return [
            "Authentication validation",
            "Authorization validation",
            "Encryption validation",
            "Input security validation",
            "Rate limiting validation",
            "Audit logging validation"
        ]

class PreprocessingValidator(BaseValidator):
    """Validator for preprocessing aspects and data quality"""
    
    def __init__(self, config: ValidationConfig):
        super().__init__(config)
        self.preprocessing_config = config.preprocessing_config
        self.quality_threshold = config.preprocessing_quality_threshold
        
    def validate(self, preprocessing_result: Dict[str, Any], context: Optional[ErrorContext] = None) -> Dict[str, Any]:
        """Validate preprocessing results"""
        start_time = time.time()
        
        validation_errors = []
        warnings = []
        
        try:
            # Extract preprocessing data
            processed_data = preprocessing_result.get('processed_data', {})
            metadata = preprocessing_result.get('metadata', {})
            quality_assessment = preprocessing_result.get('quality_assessment', {})
            
            # Feature engineering validation
            feature_errors, feature_warnings = self._validate_feature_engineering(processed_data, metadata)
            validation_errors.extend(feature_errors)
            warnings.extend(feature_warnings)
            
            # Data quality validation
            quality_errors, quality_warnings = self._validate_data_quality(quality_assessment, metadata)
            validation_errors.extend(quality_errors)
            warnings.extend(quality_warnings)
            
            # Feature selection validation
            selection_errors = self._validate_feature_selection(processed_data, metadata)
            validation_errors.extend(selection_errors)
            
            # Scaling validation
            scaling_errors = self._validate_scaling(processed_data, metadata)
            validation_errors.extend(scaling_errors)
            
            # Calculate overall preprocessing score
            preprocessing_score = self._calculate_preprocessing_score(metadata, quality_assessment)
            
            result = {
                'is_valid': len(validation_errors) == 0 and preprocessing_score >= self.quality_threshold,
                'errors': validation_errors,
                'warnings': warnings,
                'preprocessing_score': preprocessing_score,
                'feature_count': metadata.get('feature_count', 0),
                'quality_score': quality_assessment.get('overall_score', 0.0),
                'timestamp': datetime.now().isoformat()
            }
            
            self._record_performance('preprocessing_validation', time.time() - start_time)
            return result
            
        except Exception as e:
            error_context = context or create_error_context(
                component='PreprocessingValidator',
                operation='validate'
            )
            raise ValidationError(
                f"Preprocessing validation failed: {str(e)}",
                context=error_context
            )
    
    def _validate_feature_engineering(self, processed_data: Dict[str, Any], metadata: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate feature engineering results"""
        errors = []
        warnings = []
        config = self.preprocessing_config['feature_engineering']
        
        feature_count = metadata.get('feature_count', 0)
        
        # Check feature count
        if feature_count < config['min_features']:
            errors.append(f"Too few features generated: {feature_count} < {config['min_features']}")
        elif feature_count > config['max_features']:
            warnings.append(f"Many features generated: {feature_count} > {config['max_features']}")
        
        # Check required feature types
        required_types = config['required_feature_types']
        found_types = []
        
        for feature_name in processed_data.keys():
            for req_type in required_types:
                if req_type in feature_name.lower():
                    found_types.append(req_type)
                    break
        
        missing_types = set(required_types) - set(found_types)
        if missing_types:
            errors.append(f"Missing required feature types: {missing_types}")
        
        # Validate feature ranges if enabled
        if config.get('validate_feature_ranges', True):
            range_issues = self._check_feature_ranges(processed_data)
            warnings.extend(range_issues)
        
        return errors, warnings
    
    def _validate_data_quality(self, quality_assessment: Dict[str, Any], metadata: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate data quality metrics"""
        errors = []
        warnings = []
        config = self.preprocessing_config['data_quality']
        
        overall_score = quality_assessment.get('overall_score', 0.0)
        
        if overall_score < self.quality_threshold:
            errors.append(f"Data quality score too low: {overall_score:.3f} < {self.quality_threshold}")
        
        # Check for quality issues
        quality_issues = quality_assessment.get('issues', [])
        if quality_issues:
            for issue in quality_issues:
                if 'missing' in issue.lower():
                    warnings.append(f"Data quality issue: {issue}")
                else:
                    warnings.append(f"Quality issue: {issue}")
        
        # Check completeness
        completeness = quality_assessment.get('completeness', 1.0)
        max_missing = config.get('max_missing_ratio', 0.3)
        
        if (1.0 - completeness) > max_missing:
            errors.append(f"Too much missing data: {(1.0 - completeness):.2%} > {max_missing:.2%}")
        
        return errors, warnings
    
    def _validate_feature_selection(self, processed_data: Dict[str, Any], metadata: Dict[str, Any]) -> List[str]:
        """Validate feature selection process"""
        errors = []
        config = self.preprocessing_config['feature_selection']
        
        if not config.get('validate_correlation_matrix', True):
            return errors
        
        # Check for highly correlated features (simplified check)
        feature_values = [v for v in processed_data.values() if isinstance(v, (int, float))]
        
        if len(feature_values) > 1:
            # Check for potential duplicate features (same values)
            unique_values = set(feature_values)
            if len(unique_values) < len(feature_values) * 0.8:  # 80% unique threshold
                errors.append("Potential duplicate or highly correlated features detected")
        
        return errors
    
    def _validate_scaling(self, processed_data: Dict[str, Any], metadata: Dict[str, Any]) -> List[str]:
        """Validate feature scaling"""
        errors = []
        config = self.preprocessing_config['scaling']
        
        if not config.get('validate_scale_ranges', True):
            return errors
        
        # Check for reasonable scaling ranges
        numeric_features = {k: v for k, v in processed_data.items() 
                           if isinstance(v, (int, float)) and not k.endswith('_bin')}
        
        if numeric_features:
            values = list(numeric_features.values())
            min_val, max_val = min(values), max(values)
            
            # Check for extreme values that might indicate scaling issues
            if max_val > 1000 or min_val < -1000:
                errors.append(f"Potential scaling issue - extreme values detected: [{min_val:.2f}, {max_val:.2f}]")
            
            # Check for features with very small variance
            if len(set(values)) == 1:
                errors.append("Features with zero variance detected - may indicate scaling problems")
        
        return errors
    
    def _check_feature_ranges(self, processed_data: Dict[str, Any]) -> List[str]:
        """Check feature value ranges for anomalies"""
        warnings = []
        
        for feature_name, value in processed_data.items():
            if isinstance(value, (int, float)):
                # Check for infinite or NaN values
                if np.isnan(value) or np.isinf(value):
                    warnings.append(f"Invalid value in feature '{feature_name}': {value}")
                
                # Check for extremely large values
                elif abs(value) > 1e6:
                    warnings.append(f"Extremely large value in feature '{feature_name}': {value}")
        
        return warnings
    
    def _calculate_preprocessing_score(self, metadata: Dict[str, Any], quality_assessment: Dict[str, Any]) -> float:
        """Calculate overall preprocessing score"""
        score_components = []
        
        # Quality score (40% weight)
        quality_score = quality_assessment.get('overall_score', 0.0)
        score_components.append(quality_score * 0.4)
        
        # Feature count score (20% weight)
        feature_count = metadata.get('feature_count', 0)
        min_features = self.preprocessing_config['feature_engineering']['min_features']
        max_features = self.preprocessing_config['feature_engineering']['max_features']
        
        if feature_count >= min_features:
            feature_score = min(1.0, feature_count / max_features)
        else:
            feature_score = feature_count / min_features
        
        score_components.append(feature_score * 0.2)
        
        # Final quality check score (20% weight)
        final_quality = metadata.get('final_quality', {})
        final_score = final_quality.get('quality_score', quality_score)
        score_components.append(final_score * 0.2)
        
        # Completeness score (20% weight)
        completeness = quality_assessment.get('completeness', 1.0)
        score_components.append(completeness * 0.2)
        
        return sum(score_components)
    
    def get_validation_rules(self) -> List[str]:
        """Get list of validation rules"""
        return [
            "Feature engineering validation",
            "Data quality validation", 
            "Feature selection validation",
            "Feature scaling validation",
            "Feature range validation",
            "Preprocessing completeness validation"
        ]

# ======================== VALIDATION FRAMEWORK ========================

class ValidationFramework:
    """Comprehensive validation framework"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.validators = {
            'transaction': TransactionValidator(config),
            'strategy': StrategyValidator(config),
            'component': ComponentValidator(config),
            'performance': PerformanceValidator(config),
            'security': SecurityValidator(config),
            'preprocessing': PreprocessingValidator(config)
        }
        self.executor = ThreadPoolExecutor(max_workers=config.max_validation_threads)
    
    def validate_all(self, data: Dict[str, Any], context: Optional[ErrorContext] = None) -> Dict[str, Any]:
        """Run all applicable validations"""
        results = {}
        
        if self.config.enable_async_validation:
            results = self._validate_async(data, context)
        else:
            results = self._validate_sync(data, context)
        
        # Aggregate results
        overall_valid = all(result.get('is_valid', False) for result in results.values())
        all_errors = []
        all_warnings = []
        
        for result in results.values():
            all_errors.extend(result.get('errors', []))
            all_warnings.extend(result.get('warnings', []))
        
        return {
            'is_valid': overall_valid,
            'errors': all_errors,
            'warnings': all_warnings,
            'validation_results': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def _validate_sync(self, data: Dict[str, Any], context: Optional[ErrorContext] = None) -> Dict[str, Any]:
        """Run validations synchronously"""
        results = {}
        
        for validator_name, validator in self.validators.items():
            try:
                if validator_name in data:
                    results[validator_name] = validator.validate(data[validator_name], context)
            except Exception as e:
                results[validator_name] = {
                    'is_valid': False,
                    'errors': [str(e)],
                    'warnings': [],
                    'timestamp': datetime.now().isoformat()
                }
        
        return results
    
    def _validate_async(self, data: Dict[str, Any], context: Optional[ErrorContext] = None) -> Dict[str, Any]:
        """Run validations asynchronously"""
        futures = {}
        
        for validator_name, validator in self.validators.items():
            if validator_name in data:
                future = self.executor.submit(validator.validate, data[validator_name], context)
                futures[validator_name] = future
        
        results = {}
        for validator_name, future in futures.items():
            try:
                results[validator_name] = future.result(timeout=self.config.validation_timeout)
            except Exception as e:
                results[validator_name] = {
                    'is_valid': False,
                    'errors': [str(e)],
                    'warnings': [],
                    'timestamp': datetime.now().isoformat()
                }
        
        return results
    
    def get_validator(self, validator_name: str) -> BaseValidator:
        """Get specific validator"""
        return self.validators.get(validator_name)
    
    def clear_all_caches(self) -> None:
        """Clear all validator caches"""
        for validator in self.validators.values():
            validator.clear_cache()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from all validators"""
        metrics = {}
        for validator_name, validator in self.validators.items():
            metrics[validator_name] = validator.get_performance_metrics()
        return metrics
    
    def shutdown(self) -> None:
        """Shutdown validation framework"""
        self.executor.shutdown(wait=True)