"""
Enhanced Core Fraud Detection Logic with Comprehensive Validation.
Main orchestrator for the fraud detection domain with production-ready
error handling, validation, and reliability features.
"""

import logging
import json
import time
import pickle
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
import pandas as pd
import numpy as np
import traceback
import signal
from contextlib import contextmanager
from functools import wraps
import re
import hashlib
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Custom Exception Classes for Fraud Detection
# ============================================================================

class FraudDetectionException(Exception):
    """Base exception for fraud detection errors"""
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()

class FraudValidationError(FraudDetectionException):
    """Raised when fraud detection input validation fails"""
    pass

class TransactionValidationError(FraudDetectionException):
    """Raised when transaction validation fails"""
    pass

class StrategyValidationError(FraudDetectionException):
    """Raised when detection strategy validation fails"""
    pass

class ComponentIntegrationError(FraudDetectionException):
    """Raised when component integration fails"""
    pass

class FraudDetectionTimeoutError(FraudDetectionException):
    """Raised when fraud detection operations timeout"""
    pass

class FraudPerformanceError(FraudDetectionException):
    """Raised when performance thresholds are exceeded"""
    pass

class FraudSecurityError(FraudDetectionException):
    """Raised when security validation fails"""
    pass

class FraudReliabilityError(FraudDetectionException):
    """Raised when reliability checks fail"""
    pass

class FraudAccuracyError(FraudDetectionException):
    """Raised when accuracy validation fails"""
    pass

class FraudCacheError(FraudDetectionException):
    """Raised when cache operations fail"""
    pass


# ============================================================================
# Validation Classes and Utilities
# ============================================================================

class TransactionValidator:
    """Validates transaction data for fraud detection"""
    
    REQUIRED_FIELDS = ['transaction_id']
    RECOMMENDED_FIELDS = ['amount', 'timestamp', 'merchant_name', 'user_id']
    
    @staticmethod
    def validate_transaction(transaction: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate single transaction data"""
        errors = []
        
        # Check transaction is dict
        if not isinstance(transaction, dict):
            errors.append("Transaction must be a dictionary")
            return False, errors
        
        # Check required fields
        for field in TransactionValidator.REQUIRED_FIELDS:
            if field not in transaction:
                errors.append(f"Missing required field: {field}")
        
        # Validate transaction_id format
        if 'transaction_id' in transaction:
            tx_id = transaction['transaction_id']
            if not isinstance(tx_id, str) or not tx_id:
                errors.append("transaction_id must be non-empty string")
            elif len(tx_id) > 100:
                errors.append("transaction_id too long (max 100 chars)")
        
        # Validate amount if present
        if 'amount' in transaction:
            amount = transaction['amount']
            if not isinstance(amount, (int, float)):
                errors.append("amount must be numeric")
            elif amount < 0:
                errors.append("amount cannot be negative")
            elif amount > 1e9:  # $1 billion limit
                errors.append("amount exceeds maximum allowed value")
        
        # Validate timestamp if present
        if 'timestamp' in transaction:
            timestamp = transaction['timestamp']
            if isinstance(timestamp, str):
                try:
                    pd.to_datetime(timestamp)
                except:
                    errors.append("timestamp string format invalid")
            elif not isinstance(timestamp, (datetime, pd.Timestamp)):
                errors.append("timestamp must be datetime or string")
        
        # Validate merchant_name if present
        if 'merchant_name' in transaction:
            merchant = transaction['merchant_name']
            if not isinstance(merchant, str):
                errors.append("merchant_name must be string")
            elif len(merchant) > 200:
                errors.append("merchant_name too long (max 200 chars)")
        
        # Check for potentially malicious content
        for key, value in transaction.items():
            if isinstance(value, str):
                if re.search(r'<script|javascript:|eval\(|exec\(', value, re.IGNORECASE):
                    errors.append(f"Potentially malicious content in {key}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_transaction_batch(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate batch of transactions"""
        errors = []
        
        # Check if DataFrame
        if not isinstance(df, pd.DataFrame):
            errors.append("Transaction batch must be pandas DataFrame")
            return False, errors
        
        # Check if empty
        if df.empty:
            errors.append("Transaction batch is empty")
            return False, errors
        
        # Check size limits
        if len(df) > 1000000:  # 1M row limit
            errors.append(f"Batch size {len(df)} exceeds maximum (1M rows)")
        
        # Check required columns
        for field in TransactionValidator.REQUIRED_FIELDS:
            if field not in df.columns:
                errors.append(f"Missing required column: {field}")
        
        # Check for duplicates
        if 'transaction_id' in df.columns:
            duplicates = df['transaction_id'].duplicated().sum()
            if duplicates > 0:
                errors.append(f"Found {duplicates} duplicate transaction IDs")
        
        # Sample validation
        sample_size = min(100, len(df))
        sample_indices = np.random.choice(df.index, sample_size, replace=False)
        
        invalid_count = 0
        for idx in sample_indices:
            row = df.loc[idx].to_dict()
            is_valid, row_errors = TransactionValidator.validate_transaction(row)
            if not is_valid:
                invalid_count += 1
        
        if invalid_count > sample_size * 0.1:  # More than 10% invalid
            errors.append(f"High proportion of invalid transactions in sample ({invalid_count}/{sample_size})")
        
        return len(errors) == 0, errors

class StrategyValidator:
    """Validates detection strategies"""
    
    @staticmethod
    def validate_strategy(strategy: Optional['DetectionStrategy']) -> Tuple[bool, str]:
        """Validate detection strategy"""
        if strategy is None:
            return True, "Using default strategy"
        
        if not isinstance(strategy, DetectionStrategy):
            return False, f"Strategy must be DetectionStrategy enum, got {type(strategy)}"
        
        # Check if strategy is supported
        supported_strategies = list(DetectionStrategy)
        if strategy not in supported_strategies:
            return False, f"Unsupported strategy: {strategy}"
        
        return True, f"Valid strategy: {strategy.value}"

class ComponentValidator:
    """Validates component integration"""
    
    @staticmethod
    def validate_component(component: Any, component_name: str) -> Tuple[bool, str]:
        """Validate component availability and interface"""
        if component is None:
            return False, f"{component_name} not initialized"
        
        # Check for required methods based on component type
        required_methods = {
            'data_loader': ['load_data'],
            'preprocessor': ['preprocess'],
            'ml_predictor': ['predict'],
            'symbolic_reasoner': ['add_fact', 'reason'],
            'proof_verifier': ['generate_proof', 'verify_proof'],
        }
        
        if component_name in required_methods:
            for method in required_methods[component_name]:
                if not hasattr(component, method):
                    return False, f"{component_name} missing required method: {method}"
        
        return True, f"{component_name} validated"

class PerformanceValidator:
    """Validates performance metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.max_processing_time_ms = config.get('max_processing_time_ms', 5000)
        self.max_memory_mb = config.get('max_memory_mb', 512)
        self.max_cpu_percent = config.get('max_cpu_percent', 80)
    
    def validate_timing(self, operation: str, duration_ms: float) -> Tuple[bool, str]:
        """Validate operation timing"""
        if duration_ms > self.max_processing_time_ms:
            return False, f"{operation} took {duration_ms:.2f}ms (max: {self.max_processing_time_ms}ms)"
        return True, f"{operation} completed in {duration_ms:.2f}ms"
    
    def validate_resources(self) -> Tuple[bool, str]:
        """Validate system resource usage"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent(interval=0.1)
            
            if memory_mb > self.max_memory_mb:
                return False, f"Memory usage {memory_mb:.1f}MB exceeds limit {self.max_memory_mb}MB"
            
            if cpu_percent > self.max_cpu_percent:
                return False, f"CPU usage {cpu_percent:.1f}% exceeds limit {self.max_cpu_percent}%"
            
            return True, f"Resources OK (Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%)"
            
        except Exception as e:
            return False, f"Resource check failed: {str(e)}"

class SecurityValidator:
    """Validates security aspects of fraud detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.allowed_sources = set(config.get('allowed_sources', []))
        self.max_request_size_mb = config.get('max_request_size_mb', 10)
        self.rate_limits = config.get('rate_limits', {})
        self.access_log = defaultdict(list)
    
    def validate_access(self, user_context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """Validate access permissions"""
        if not user_context:
            return True, "No access restrictions"
        
        user_id = user_context.get('user_id', 'anonymous')
        
        # Check rate limiting
        if user_id in self.rate_limits:
            now = datetime.now()
            recent_accesses = [
                ts for ts in self.access_log[user_id]
                if now - ts < timedelta(minutes=1)
            ]
            
            if len(recent_accesses) >= self.rate_limits[user_id]:
                return False, f"Rate limit exceeded for user {user_id}"
        
        self.access_log[user_id].append(datetime.now())
        
        # Cleanup old entries
        cutoff = datetime.now() - timedelta(hours=1)
        self.access_log[user_id] = [
            ts for ts in self.access_log[user_id]
            if ts > cutoff
        ]
        
        return True, f"Access granted for {user_id}"
    
    def validate_data_size(self, data: Any) -> Tuple[bool, str]:
        """Validate data size to prevent DoS"""
        try:
            # Estimate size
            if isinstance(data, pd.DataFrame):
                size_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
            else:
                size_mb = len(str(data)) / 1024 / 1024
            
            if size_mb > self.max_request_size_mb:
                return False, f"Data size {size_mb:.1f}MB exceeds limit {self.max_request_size_mb}MB"
            
            return True, f"Data size {size_mb:.1f}MB within limits"
            
        except Exception as e:
            return False, f"Size validation failed: {str(e)}"

class FraudRiskLevel(Enum):
    """Fraud risk levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


# ============================================================================
# Decorators for Validation and Error Handling
# ============================================================================

def validate_inputs(validation_func: Callable) -> Callable:
    """Decorator for input validation"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Run validation
            is_valid, message = validation_func(*args, **kwargs)
            if not is_valid:
                raise FraudValidationError(f"Input validation failed: {message}")
            
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def with_timeout(timeout_seconds: int) -> Callable:
    """Decorator to add timeout to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def timeout_handler(signum, frame):
                raise FraudDetectionTimeoutError(
                    f"Operation timed out after {timeout_seconds} seconds"
                )
            
            # Set alarm for timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator

def with_error_recovery(max_retries: int = 3, backoff: float = 1.0) -> Callable:
    """Decorator for automatic error recovery"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except FraudDetectionTimeoutError:
                    # Don't retry timeouts
                    raise
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = backoff * (2 ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {str(e)}. "
                            f"Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed")
            
            if last_error:
                raise last_error
                
        return wrapper
    return decorator

def monitor_performance(operation_name: str) -> Callable:
    """Decorator to monitor performance"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                result = func(self, *args, **kwargs)
                
                # Log performance metrics
                duration_ms = (time.time() - start_time) * 1000
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_used = end_memory - start_memory
                
                logger.debug(
                    f"{operation_name} - Duration: {duration_ms:.2f}ms, "
                    f"Memory: {memory_used:.2f}MB"
                )
                
                # Validate performance if validator exists
                if hasattr(self, 'performance_validator'):
                    is_valid, msg = self.performance_validator.validate_timing(
                        operation_name, duration_ms
                    )
                    if not is_valid:
                        logger.warning(f"Performance warning: {msg}")
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"{operation_name} failed after {duration_ms:.2f}ms: {str(e)}"
                )
                raise
                
        return wrapper
    return decorator

class DetectionStrategy(Enum):
    """Fraud detection strategies"""
    RULES_ONLY = "rules_only"
    ML_ONLY = "ml_only"
    SYMBOLIC_ONLY = "symbolic_only"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"


@dataclass
class FraudDetectionResult:
    """Enhanced result of fraud detection with validation"""
    transaction_id: str
    fraud_score: float
    is_fraud: bool
    risk_level: FraudRiskLevel
    detection_method: str
    confidence: float
    explanations: List[str]
    anomaly_indicators: Dict[str, float]
    ml_scores: Optional[Dict[str, float]] = None
    rule_violations: Optional[List[str]] = None
    symbolic_inferences: Optional[List[Dict[str, Any]]] = None
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Validation attributes
    validation_status: str = "pending"
    validation_errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate result after initialization"""
        self.validate()
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate detection result"""
        errors = []
        
        # Validate required fields
        if not self.transaction_id:
            errors.append("transaction_id is required")
        
        # Validate fraud score
        if not 0 <= self.fraud_score <= 1:
            errors.append(f"fraud_score {self.fraud_score} must be between 0 and 1")
        
        # Validate confidence
        if not 0 <= self.confidence <= 1:
            errors.append(f"confidence {self.confidence} must be between 0 and 1")
        
        # Validate risk level consistency
        expected_risk = self._calculate_risk_level(self.fraud_score)
        if self.risk_level != expected_risk:
            errors.append(
                f"Risk level {self.risk_level} inconsistent with score {self.fraud_score}"
            )
        
        # Validate detection method
        try:
            DetectionStrategy(self.detection_method)
        except ValueError:
            errors.append(f"Invalid detection method: {self.detection_method}")
        
        # Update validation status
        if errors:
            self.validation_status = "failed"
            self.validation_errors = errors
        else:
            self.validation_status = "passed"
            self.validation_errors = []
        
        return len(errors) == 0, errors
    
    def _calculate_risk_level(self, score: float) -> FraudRiskLevel:
        """Calculate expected risk level from score"""
        if score >= 0.9:
            return FraudRiskLevel.CRITICAL
        elif score >= 0.7:
            return FraudRiskLevel.HIGH
        elif score >= 0.5:
            return FraudRiskLevel.MEDIUM
        elif score >= 0.3:
            return FraudRiskLevel.LOW
        else:
            return FraudRiskLevel.VERY_LOW


@dataclass
class FraudDetectionMetrics:
    """Enhanced metrics for fraud detection performance"""
    total_transactions: int = 0
    fraud_detected: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    average_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0
    min_processing_time_ms: float = float('inf')
    
    detection_rate: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Enhanced metrics
    validation_errors: int = 0
    timeout_errors: int = 0
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Strategy-specific metrics
    metrics_by_strategy: Dict[DetectionStrategy, Dict[str, float]] = field(default_factory=dict)
    
    # Time-based metrics
    hourly_detection_rate: Dict[str, float] = field(default_factory=dict)
    
    def update_classification_metrics(self):
        """Update classification metrics"""
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        
        if total > 0:
            self.accuracy = (self.true_positives + self.true_negatives) / total
        
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)
        
        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        
        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        
        if self.total_transactions > 0:
            self.detection_rate = self.fraud_detected / self.total_transactions

# ============================================================================
# Error Recovery Manager
# ============================================================================

class ErrorRecoveryManager:
    """Manages error recovery strategies for fraud detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_retries = config.get('max_retries', 3)
        self.recovery_strategies = {
            'component_failure': self._recover_from_component_failure,
            'timeout': self._recover_from_timeout,
            'resource_exhaustion': self._recover_from_resource_exhaustion,
            'validation_failure': self._recover_from_validation_failure,
        }
        self.recovery_history = []
    
    def attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> Optional[Any]:
        """Attempt to recover from error"""
        error_type = self._classify_error(error)
        
        if error_type in self.recovery_strategies:
            recovery_strategy = self.recovery_strategies[error_type]
            
            try:
                logger.info(f"Attempting recovery for {error_type}: {str(error)}")
                result = recovery_strategy(error, context)
                
                self.recovery_history.append({
                    'timestamp': datetime.now(),
                    'error_type': error_type,
                    'success': result is not None,
                    'context': context
                })
                
                return result
                
            except Exception as recovery_error:
                logger.error(f"Recovery failed: {str(recovery_error)}")
                return None
        
        return None
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error type for recovery strategy"""
        if isinstance(error, FraudDetectionTimeoutError):
            return 'timeout'
        elif isinstance(error, (FraudPerformanceError, MemoryError)):
            return 'resource_exhaustion'
        elif isinstance(error, (FraudValidationError, TransactionValidationError)):
            return 'validation_failure'
        elif isinstance(error, ComponentIntegrationError):
            return 'component_failure'
        else:
            return 'unknown'
    
    def _recover_from_component_failure(self, error: Exception, context: Dict[str, Any]) -> Optional[Any]:
        """Recover from component failure"""
        # Try to use alternative components or fallback methods
        component_name = context.get('component_name')
        
        if component_name == 'ml_predictor':
            # Fall back to rules-only detection
            logger.info("ML predictor failed, falling back to rules-only detection")
            context['fallback_strategy'] = DetectionStrategy.RULES_ONLY
            return context
        
        elif component_name == 'symbolic_reasoner':
            # Skip symbolic reasoning
            logger.info("Symbolic reasoner failed, skipping symbolic analysis")
            context['skip_symbolic'] = True
            return context
        
        return None
    
    def _recover_from_timeout(self, error: Exception, context: Dict[str, Any]) -> Optional[Any]:
        """Recover from timeout error"""
        # Try with reduced complexity
        if 'batch_size' in context:
            # Reduce batch size
            new_batch_size = max(1, context['batch_size'] // 2)
            logger.info(f"Reducing batch size from {context['batch_size']} to {new_batch_size}")
            context['batch_size'] = new_batch_size
            return context
        
        elif 'strategy' in context:
            # Use simpler strategy
            if context['strategy'] == DetectionStrategy.ENSEMBLE:
                logger.info("Timeout with ensemble, falling back to hybrid")
                context['strategy'] = DetectionStrategy.HYBRID
                return context
            elif context['strategy'] == DetectionStrategy.HYBRID:
                logger.info("Timeout with hybrid, falling back to rules-only")
                context['strategy'] = DetectionStrategy.RULES_ONLY
                return context
        
        return None
    
    def _recover_from_resource_exhaustion(self, error: Exception, context: Dict[str, Any]) -> Optional[Any]:
        """Recover from resource exhaustion"""
        # Free up resources and retry with reduced load
        import gc
        gc.collect()
        
        # Reduce parallel processing
        if 'max_workers' in context:
            context['max_workers'] = max(1, context['max_workers'] // 2)
            logger.info(f"Reduced parallel workers to {context['max_workers']}")
        
        # Clear caches if possible
        if 'clear_cache' in context:
            context['clear_cache'] = True
        
        return context
    
    def _recover_from_validation_failure(self, error: Exception, context: Dict[str, Any]) -> Optional[Any]:
        """Recover from validation failure"""
        # Try to clean/fix data
        if 'transaction' in context:
            transaction = context['transaction']
            
            # Add missing required fields with defaults
            if 'transaction_id' not in transaction:
                transaction['transaction_id'] = f"RECOVERY_{datetime.now().timestamp()}"
                logger.info("Added default transaction_id")
            
            # Fix invalid values
            if 'amount' in transaction and not isinstance(transaction['amount'], (int, float)):
                try:
                    transaction['amount'] = float(transaction['amount'])
                except:
                    transaction['amount'] = 0.0
                logger.info("Fixed invalid amount")
            
            context['transaction'] = transaction
            return context
        
        return None


class FinancialFraudCore:
    """
    Enhanced Core fraud detection orchestrator with comprehensive validation.
    
    This class coordinates all fraud detection components and provides
    the main interface for fraud detection operations with production-ready
    error handling, validation, and reliability features.
    """
    
    def __init__(self, config_manager=None, use_enhanced=True, validation_config=None):
        """
        Initialize enhanced fraud detection core.
        
        Args:
            config_manager: Configuration manager instance
            use_enhanced: Whether to use enhanced validation features
            validation_config: Configuration for validation framework
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config_manager = config_manager
        self.config = self._load_config()
        self.use_enhanced = use_enhanced
        
        # Enhanced validation framework
        self.validators = self._initialize_validators(validation_config or {})
        self.error_recovery = ErrorRecoveryManager(self.config.get('error_recovery', {}))
        self.performance_monitor = self._initialize_performance_monitor()
        
        # Initialize components
        self.data_loader = None
        self.preprocessor = None
        self.ml_predictor = None
        self.symbolic_reasoner = None
        self.proof_verifier = None
        self.rule_engine = None
        
        # Detection cache with enhanced error handling
        self._detection_cache = {}
        self._cache_ttl = timedelta(minutes=self.config.get('cache_ttl_minutes', 5))
        self._cache_stats = {'hits': 0, 'misses': 0, 'errors': 0}
        
        # Enhanced metrics
        self.metrics = FraudDetectionMetrics()
        
        # Thread safety with enhanced monitoring
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 4),
            thread_name_prefix='fraud_detection'
        )
        
        # Storage with validation
        self.storage_path = Path(self.config.get('storage_path', 'fraud_detection'))
        try:
            self.storage_path.mkdir(exist_ok=True, parents=True)
        except PermissionError as e:
            raise FraudDetectionException(f"Cannot create storage directory: {e}")
        
        # Security context
        self.security_context = {
            'max_transaction_size': self.config.get('max_transaction_size', 1000000),
            'enable_security_validation': self.config.get('enable_security_validation', True),
            'allowed_currencies': self.config.get('allowed_currencies', ['USD', 'EUR', 'GBP'])
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            'max_detection_time': self.config.get('max_detection_time', 30.0),
            'max_memory_usage_mb': self.config.get('max_memory_usage_mb', 1024),
            'max_concurrent_detections': self.config.get('max_concurrent_detections', 100)
        }
        
        # Initialize components with enhanced error handling
        self._initialize_components_enhanced()
        
        self.logger.info("Enhanced FinancialFraudCore initialized successfully")
    
    def _initialize_validators(self, validation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize validation framework"""
        try:
            return {
                'transaction': TransactionValidator(validation_config.get('transaction', {})),
                'strategy': StrategyValidator(validation_config.get('strategy', {})),
                'component': ComponentValidator(validation_config.get('component', {})),
                'performance': PerformanceValidator(validation_config.get('performance', {})),
                'security': SecurityValidator(validation_config.get('security', {}))
            }
        except Exception as e:
            logger.warning(f"Failed to initialize enhanced validators: {e}")
            return {}
    
    def _initialize_performance_monitor(self) -> Dict[str, Any]:
        """Initialize performance monitoring"""
        return {
            'detection_times': [],
            'memory_usage': [],
            'error_counts': defaultdict(int),
            'start_time': datetime.now(),
            'total_detections': 0,
            'successful_detections': 0
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load enhanced configuration"""
        default_config = {
            'detection_strategy': DetectionStrategy.HYBRID.value,
            'fraud_threshold': 0.7,
            'cache_ttl_minutes': 5,
            'max_workers': 4,
            'enable_ml': True,
            'enable_rules': True,
            'enable_symbolic': True,
            'batch_size': 100,
            'auto_persist': True,
            'log_level': 'INFO',
            # Enhanced configuration
            'enable_validation': True,
            'enable_security_checks': True,
            'enable_performance_monitoring': True,
            'max_detection_time': 30.0,
            'max_memory_usage_mb': 1024,
            'max_concurrent_detections': 100,
            'enable_error_recovery': True,
            'validation_strict_mode': False,
            'cache_validation': True
        }
        
        if self.config_manager:
            return self.config_manager.get_config('fraud_detection', default_config)
        return default_config
    
    def _initialize_components_enhanced(self):
        """Initialize all fraud detection components with enhanced error handling."""
        component_errors = []
        
        try:
            self.logger.info("Initializing enhanced fraud detection components...")
            
            # Component initialization with validation
            components_to_init = [
                ('data_loader', self._init_data_loader),
                ('preprocessor', self._init_preprocessor),
                ('ml_predictor', self._init_ml_predictor),
                ('symbolic_reasoner', self._init_symbolic_reasoner),
                ('proof_verifier', self._init_proof_verifier),
                ('rule_engine', self._init_rule_engine)
            ]
            
            for component_name, init_func in components_to_init:
                try:
                    with monitor_performance('component_init', {'component': component_name}):
                        result = init_func()
                        if result:
                            self.logger.info(f\"Enhanced {component_name} initialized successfully\")
                        else:
                            self.logger.warning(f\"{component_name} initialization returned None\")
                            
                except Exception as e:
                    error_msg = f\"Failed to initialize {component_name}: {str(e)}\"\n                    self.logger.error(error_msg)\n                    component_errors.append(error_msg)\n                    \n                    # Try error recovery\n                    recovery_context = {\n                        'component_name': component_name,\n                        'error': e,\n                        'config': self.config\n                    }\n                    \n                    recovery_result = self.error_recovery.attempt_recovery(e, recovery_context)\n                    if recovery_result:\n                        self.logger.info(f\"Successfully recovered from {component_name} initialization error\")\n                    else:\n                        # Set component to None for graceful degradation\n                        setattr(self, component_name, None)\n            \n            # Validate component integration\n            if self.use_enhanced and self.validators.get('component'):\n                try:\n                    self._validate_component_integration()\n                except ComponentIntegrationError as e:\n                    self.logger.warning(f\"Component integration validation failed: {e}\")\n                    if self.config.get('validation_strict_mode', False):\n                        raise\n            \n            if component_errors and self.config.get('validation_strict_mode', False):\n                raise ComponentIntegrationError(f\"Component initialization failed: {component_errors}\")\n            \n            self.logger.info(f\"Enhanced fraud detection components initialized (errors: {len(component_errors)})\")\n            \n        except Exception as e:\n            self.logger.error(f\"Critical failure in component initialization: {e}\")\n            raise ComponentIntegrationError(f\"Cannot initialize fraud detection core: {e}\")
    
    def _init_data_loader(self):
        """Initialize data loader with enhanced error handling"""
        try:
            from ..data.data_loader import FinancialDataLoader
            self.data_loader = FinancialDataLoader(self.config_manager)
            return True
        except ImportError:
            self.logger.warning("DataLoader not available, using placeholder")
            self.data_loader = None
            return False
    
    def _init_preprocessor(self):
        """Initialize preprocessor with enhanced features"""
        try:
            # Try enhanced preprocessor first
            if self.use_enhanced:
                try:
                    from ..enhanced_data_preprocessor import EnhancedFinancialDataPreprocessor
                    self.preprocessor = EnhancedFinancialDataPreprocessor(
                        config_manager=self.config_manager,
                        validation_config=self.config.get('preprocessor_validation', {})
                    )
                    self.logger.info("Enhanced preprocessor initialized")
                    return True
                except ImportError:
                    self.logger.info("Enhanced preprocessor not available, falling back to standard")
            
            # Fall back to standard preprocessor
            from ..data_preprocessor import FinancialDataPreprocessor
            self.preprocessor = FinancialDataPreprocessor(self.config_manager)
            return True
            
        except ImportError:
            self.logger.warning("Preprocessor not available, using placeholder")
            self.preprocessor = None
            return False
    
    def _init_ml_predictor(self):
        """Initialize ML predictor with enhanced features"""
        try:
            # Try enhanced ML predictor first
            if self.use_enhanced:
                try:
                    from ..enhanced_ml_predictor import EnhancedFinancialMLPredictor
                    self.ml_predictor = EnhancedFinancialMLPredictor(
                        config_manager=self.config_manager,
                        validation_config=self.config.get('ml_validation', {})
                    )
                    self.logger.info("Enhanced ML predictor initialized")
                    return True
                except ImportError:
                    self.logger.info("Enhanced ML predictor not available, falling back to standard")
            
            # Fall back to standard ML predictor
            from ..ml_predictor import FinancialMLPredictor
            self.ml_predictor = FinancialMLPredictor(self.config_manager)
            return True
            
        except ImportError:
            self.logger.warning("ML predictor not available, using placeholder")
            self.ml_predictor = None
            return False
    
    def _init_symbolic_reasoner(self):
        """Initialize symbolic reasoner with enhanced features"""
        try:
            from ..symbolic_reasoner import FinancialSymbolicReasoner
            self.symbolic_reasoner = FinancialSymbolicReasoner(
                config=self.config.get('symbolic_config', {}),
                use_enhanced=self.use_enhanced
            )
            return True
        except ImportError:
            self.logger.warning("Symbolic reasoner not available, using placeholder")
            self.symbolic_reasoner = None
            return False
    
    def _init_proof_verifier(self):
        """Initialize proof verifier with enhanced features"""
        try:
            from ..proof_verifier import FinancialProofVerifier
            self.proof_verifier = FinancialProofVerifier(
                config_manager=self.config_manager,
                use_enhanced=self.use_enhanced
            )
            return True
        except ImportError:
            self.logger.warning("Proof verifier not available, using placeholder")
            self.proof_verifier = None
            return False
    
    def _init_rule_engine(self):
        """Initialize enhanced rule engine"""
        self.rule_engine = {
            'high_amount': lambda t: t.get('amount', 0) > self.config.get('high_amount_threshold', 10000),
            'unusual_time': lambda t: self._is_unusual_time(t),
            'rapid_velocity': lambda t: self._check_velocity(t),
            'geographic_anomaly': lambda t: self._check_geographic(t),
            'merchant_risk': lambda t: self._check_merchant_risk(t),
            # Enhanced rules
            'currency_validation': lambda t: self._validate_currency(t),
            'amount_validation': lambda t: self._validate_amount_range(t),
            'timestamp_validation': lambda t: self._validate_timestamp(t)
        }
        return True
    
    def _validate_component_integration(self):
        \"\"\"Validate that components are properly integrated\"\"\"
        if self.validators.get('component'):
            validator = self.validators['component']
            
            components = {
                'data_loader': self.data_loader,
                'preprocessor': self.preprocessor,
                'ml_predictor': self.ml_predictor,
                'symbolic_reasoner': self.symbolic_reasoner,
                'proof_verifier': self.proof_verifier,
                'rule_engine': self.rule_engine
            }
            
            is_valid, validation_errors = validator.validate_integration(components)
            if not is_valid:
                raise ComponentIntegrationError(f\"Component integration validation failed: {validation_errors}\")
    
    
    @validate_inputs
    @with_timeout(30.0)
    @with_error_recovery
    @monitor_performance
    def detect_fraud(self, 
                    transaction: Union[Dict[str, Any], pd.DataFrame],
                    strategy: Optional[DetectionStrategy] = None,
                    generate_proof: bool = True,
                    validation_config: Optional[Dict[str, Any]] = None) -> Union[FraudDetectionResult, List[FraudDetectionResult]]:
        """
        Enhanced fraud detection with comprehensive validation and error handling.
        
        Args:
            transaction: Single transaction dict or DataFrame of transactions
            strategy: Detection strategy to use (None = use configured default)
            generate_proof: Whether to generate verifiable proof
            validation_config: Additional validation configuration
            
        Returns:
            Fraud detection result(s)
            
        Raises:
            FraudValidationError: If input validation fails
            FraudDetectionTimeoutError: If detection times out
            FraudPerformanceError: If performance thresholds exceeded
        """
        detection_id = f"fraud_detect_{datetime.now().timestamp()}"
        context = {
            'detection_id': detection_id,
            'transaction_count': 1 if isinstance(transaction, dict) else len(transaction),
            'strategy': strategy,
            'generate_proof': generate_proof
        }
        
        start_time = time.time()
        
        try:
            # Update performance monitoring
            self.performance_monitor['total_detections'] += 1
            
            # Enhanced input validation
            if self.use_enhanced and self.validators.get('transaction'):
                self._validate_detection_inputs(transaction, strategy, validation_config)
            
            # Convert to DataFrame if needed
            if isinstance(transaction, dict):
                df = pd.DataFrame([transaction])
                single_transaction = True
            else:
                df = transaction.copy()
                single_transaction = False
            
            # Security validation
            if self.config.get('enable_security_checks', True):
                self._perform_security_validation(df)
            
            # Use configured strategy if not specified
            if strategy is None:
                strategy = DetectionStrategy(self.config.get('detection_strategy'))
            
            # Strategy validation
            if self.use_enhanced and self.validators.get('strategy'):
                self._validate_strategy(strategy, df)
            
            self.logger.info(f"Enhanced fraud detection for {len(df)} transactions using {strategy.value} strategy")
            
            # Check performance limits
            if self.config.get('enable_performance_monitoring', True):
                self._check_performance_limits(df)
            
            # Process transactions with enhanced error handling
            try:
                if len(df) > self.config.get('batch_size', 100):
                    # Process in batches for large datasets
                    results = self._batch_detect_enhanced(df, strategy, generate_proof, context)
                else:
                    # Process all at once
                    results = self._detect_fraud_batch_enhanced(df, strategy, generate_proof, context)
            except Exception as e:
                # Try error recovery
                recovery_context = {
                    **context,
                    'error': e,
                    'batch_size': len(df),
                    'strategy': strategy
                }
                
                recovery_result = self.error_recovery.attempt_recovery(e, recovery_context)
                if recovery_result:
                    # Retry with recovery parameters
                    if 'strategy' in recovery_result:
                        strategy = recovery_result['strategy']
                    if 'batch_size' in recovery_result:
                        self.config['batch_size'] = recovery_result['batch_size']
                    
                    # Retry detection
                    results = self._detect_fraud_batch_enhanced(df, strategy, generate_proof, context)
                else:
                    raise
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics_enhanced(results, processing_time, context)
            self.performance_monitor['successful_detections'] += 1
            
            # Performance validation
            if self.use_enhanced and self.validators.get('performance'):
                self._validate_detection_performance(processing_time, results)
            
            # Auto-persist if enabled
            if self.config.get('auto_persist', True) and len(results) > 0:
                self._executor.submit(self.persist_results_enhanced, results, context)
            
            return results[0] if single_transaction else results
            
        except Exception as e:
            self.performance_monitor['error_counts'][type(e).__name__] += 1
            self.logger.error(f"Enhanced fraud detection failed: {e}", exc_info=True)
            
            if isinstance(e, (FraudValidationError, FraudDetectionTimeoutError, FraudPerformanceError)):
                raise
            else:
                raise FraudDetectionException(f"Fraud detection error: {str(e)}", 
                                            error_code="DETECTION_FAILURE",
                                            details=context)
    
    # ============================================================================
    # Enhanced Validation Helper Methods
    # ============================================================================
    
    def _validate_detection_inputs(self, transaction: Union[Dict[str, Any], pd.DataFrame], 
                                 strategy: Optional[DetectionStrategy],
                                 validation_config: Optional[Dict[str, Any]]):
        """Validate fraud detection inputs"""
        validator = self.validators['transaction']
        
        if isinstance(transaction, dict):
            is_valid, errors = validator.validate_transaction(transaction)
            if not is_valid:
                raise FraudValidationError(f"Transaction validation failed: {errors}")
        else:
            # Validate DataFrame
            if transaction.empty:
                raise FraudValidationError("Empty transaction DataFrame provided")
            
            for idx, row in transaction.iterrows():
                is_valid, errors = validator.validate_transaction(row.to_dict())
                if not is_valid:
                    raise FraudValidationError(f"Transaction validation failed at row {idx}: {errors}")
    
    def _perform_security_validation(self, df: pd.DataFrame):
        """Perform security validation on transaction data"""
        if not self.validators.get('security'):
            return
            
        validator = self.validators['security']
        
        for idx, row in df.iterrows():
            transaction = row.to_dict()
            
            # Check for security threats
            is_valid, threats = validator.validate_security_context(transaction, self.security_context)
            if not is_valid:
                raise FraudSecurityError(f"Security validation failed for transaction {idx}: {threats}")
            
            # Validate transaction amounts against limits
            amount = transaction.get('amount', 0)
            if amount > self.security_context.get('max_transaction_size', 1000000):
                raise FraudSecurityError(f"Transaction amount {amount} exceeds security limit")
    
    def _validate_strategy(self, strategy: DetectionStrategy, df: pd.DataFrame):
        """Validate detection strategy"""
        validator = self.validators['strategy']
        
        is_valid, errors = validator.validate_strategy(strategy, {
            'transaction_count': len(df),
            'available_components': {
                'ml_predictor': self.ml_predictor is not None,
                'symbolic_reasoner': self.symbolic_reasoner is not None,
                'proof_verifier': self.proof_verifier is not None
            }
        })
        
        if not is_valid:
            raise StrategyValidationError(f"Strategy validation failed: {errors}")
    
    def _check_performance_limits(self, df: pd.DataFrame):
        """Check performance limits before processing"""
        # Check memory usage
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        if memory_usage > self.performance_thresholds['max_memory_usage_mb']:
            raise FraudPerformanceError(f"Memory usage {memory_usage}MB exceeds limit")
        
        # Check concurrent detections
        if self.performance_monitor['total_detections'] > self.performance_thresholds['max_concurrent_detections']:
            raise FraudPerformanceError("Maximum concurrent detections exceeded")
        
        # Check transaction count
        if len(df) > 10000:  # Large batch warning
            self.logger.warning(f"Processing large batch of {len(df)} transactions")
    
    def _validate_detection_performance(self, processing_time: float, results: List[FraudDetectionResult]):
        """Validate detection performance meets thresholds"""
        validator = self.validators['performance']
        
        performance_metrics = {
            'processing_time': processing_time,
            'result_count': len(results),
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,
            'success_rate': len([r for r in results if r.validation_status]) / len(results) if results else 0
        }
        
        is_valid, errors = validator.validate_performance(performance_metrics, self.performance_thresholds)
        if not is_valid:
            raise FraudPerformanceError(f"Performance validation failed: {errors}")
    
    def _validate_currency(self, transaction: Dict[str, Any]) -> bool:
        """Validate transaction currency"""
        currency = transaction.get('currency', 'USD').upper()
        return currency in self.security_context.get('allowed_currencies', ['USD', 'EUR', 'GBP'])
    
    def _validate_amount_range(self, transaction: Dict[str, Any]) -> bool:
        """Validate transaction amount is in acceptable range"""
        amount = transaction.get('amount', 0)
        return 0 <= amount <= self.security_context.get('max_transaction_size', 1000000)
    
    def _validate_timestamp(self, transaction: Dict[str, Any]) -> bool:
        """Validate transaction timestamp"""
        timestamp = transaction.get('timestamp')
        if not timestamp:
            return False
        
        try:
            if isinstance(timestamp, str):
                datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif isinstance(timestamp, datetime):
                pass  # Already a datetime
            else:
                return False
            return True
        except:
            return False
    
    # ============================================================================
    # Enhanced Batch Processing Methods
    # ============================================================================
    
    def _batch_detect_enhanced(self, df: pd.DataFrame, 
                             strategy: DetectionStrategy,
                             generate_proof: bool,
                             context: Dict[str, Any]) -> List[FraudDetectionResult]:
        """Enhanced batch processing with validation and error recovery"""
        batch_size = self.config.get('batch_size', 100)
        all_results = []
        failed_batches = []
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_context = {**context, 'batch_index': i//batch_size, 'batch_size': len(batch)}
            
            try:
                with monitor_performance('batch_processing', batch_context):
                    results = self._detect_fraud_batch_enhanced(batch, strategy, generate_proof, batch_context)
                    all_results.extend(results)
                    
                self.logger.debug(f"Processed enhanced batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
                
            except Exception as e:
                self.logger.error(f"Batch {i//batch_size} failed: {e}")
                failed_batches.append({'batch_index': i//batch_size, 'error': str(e)})
                
                # Try error recovery for this batch
                recovery_context = {**batch_context, 'error': e}
                recovery_result = self.error_recovery.attempt_recovery(e, recovery_context)
                
                if recovery_result:
                    try:
                        # Retry with recovery parameters
                        results = self._detect_fraud_batch_enhanced(batch, strategy, generate_proof, recovery_context)
                        all_results.extend(results)
                        self.logger.info(f"Successfully recovered batch {i//batch_size}")
                    except Exception as retry_error:
                        self.logger.error(f"Batch recovery failed: {retry_error}")
                        # Create error results for failed transactions
                        error_results = self._create_error_results(batch, str(retry_error))
                        all_results.extend(error_results)
                else:
                    # Create error results for failed transactions
                    error_results = self._create_error_results(batch, str(e))
                    all_results.extend(error_results)
        
        if failed_batches:
            self.logger.warning(f"Failed to process {len(failed_batches)} batches")
        
        return all_results
    
    def _detect_fraud_batch_enhanced(self, df: pd.DataFrame,
                                   strategy: DetectionStrategy,
                                   generate_proof: bool,
                                   context: Dict[str, Any]) -> List[FraudDetectionResult]:
        """Enhanced fraud detection for a batch with comprehensive validation"""
        results = []
        
        try:
            if strategy == DetectionStrategy.ENSEMBLE:
                # Enhanced ensemble processing
                results = self._process_ensemble_enhanced(df, generate_proof, context)
            else:
                # Enhanced single strategy processing
                results = self._process_single_strategy_enhanced(df, strategy, generate_proof, context)
            
            # Validate all results
            if self.use_enhanced:
                self._validate_batch_results(results, context)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Enhanced batch detection failed: {e}")
            raise
    
    def _create_error_results(self, df: pd.DataFrame, error_message: str) -> List[FraudDetectionResult]:
        """Create error results for failed transactions"""
        error_results = []
        
        for _, row in df.iterrows():
            transaction = row.to_dict()
            result = FraudDetectionResult(
                transaction_id=transaction.get('transaction_id', 'unknown'),
                is_fraud=False,
                confidence_score=0.0,
                risk_score=0.0,
                fraud_probability=0.0,
                explanations=[f"Detection failed: {error_message}"],
                validation_status=False,
                error_details={'error': error_message, 'timestamp': datetime.now().isoformat()}
            )
            error_results.append(result)
        
        return error_results
    
    def _validate_batch_results(self, results: List[FraudDetectionResult], context: Dict[str, Any]):
        """Validate batch processing results"""
        if not results:
            raise FraudValidationError("No results generated for batch")
        
        # Check for validation failures
        failed_validations = [r for r in results if not r.validation_status]
        if failed_validations:
            self.logger.warning(f"{len(failed_validations)} results failed validation in batch")
        
        # Check result consistency
        fraud_count = len([r for r in results if r.is_fraud])
        if fraud_count == len(results):
            self.logger.warning("All transactions flagged as fraud - possible detection issue")
    
    def _update_metrics_enhanced(self, results: List[FraudDetectionResult], 
                               processing_time: float, context: Dict[str, Any]):
        """Update enhanced metrics with additional context"""
        # Update performance monitoring
        self.performance_monitor['detection_times'].append(processing_time)
        self.performance_monitor['memory_usage'].append(psutil.Process().memory_info().rss / 1024 / 1024)
        
        # Keep only recent metrics (last 1000 entries)
        if len(self.performance_monitor['detection_times']) > 1000:
            self.performance_monitor['detection_times'] = self.performance_monitor['detection_times'][-1000:]
        if len(self.performance_monitor['memory_usage']) > 1000:
            self.performance_monitor['memory_usage'] = self.performance_monitor['memory_usage'][-1000:]
        
        # Update traditional metrics
        self._update_metrics(results, processing_time)
        
        # Enhanced metrics logging
        self.logger.info(f"Enhanced detection completed: {len(results)} results, "
                        f"{processing_time:.2f}ms, context: {context.get('detection_id', 'unknown')}")
    
    def persist_results_enhanced(self, results: List[FraudDetectionResult], context: Dict[str, Any]):
        """Enhanced result persistence with context"""
        try:
            # Add context to persistence
            enhanced_results = []
            for result in results:
                enhanced_result = result.__dict__.copy()
                enhanced_result['detection_context'] = context
                enhanced_result['persistence_timestamp'] = datetime.now().isoformat()
                enhanced_results.append(enhanced_result)
            
            # Persist with enhanced data
            persistence_path = self.storage_path / f"detection_{context.get('detection_id', 'unknown')}.json"
            with open(persistence_path, 'w') as f:
                json.dump(enhanced_results, f, indent=2, default=str)
            
            self.logger.debug(f"Enhanced results persisted to {persistence_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to persist enhanced results: {e}")
    
    def _batch_detect(self, df: pd.DataFrame, 
                     strategy: DetectionStrategy,
                     generate_proof: bool) -> List[FraudDetectionResult]:
        """Process large dataset in batches"""
        batch_size = self.config.get('batch_size', 100)
        all_results = []
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            results = self._detect_fraud_batch(batch, strategy, generate_proof)
            all_results.extend(results)
            
            self.logger.debug(f"Processed batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
        
        return all_results
    
    def _detect_fraud_batch(self, df: pd.DataFrame,
                          strategy: DetectionStrategy,
                          generate_proof: bool) -> List[FraudDetectionResult]:
        """Detect fraud in a batch of transactions"""
        results = []
        
        if strategy == DetectionStrategy.ENSEMBLE:
            # Run all strategies in parallel
            futures = {
                self._executor.submit(self._detect_fraud_batch, df, s, False): s
                for s in [DetectionStrategy.RULES_ONLY, DetectionStrategy.ML_ONLY, 
                         DetectionStrategy.SYMBOLIC_ONLY]
            }
            
            # Collect results from all strategies
            strategy_results = defaultdict(list)
            for future in as_completed(futures):
                strategy_type = futures[future]
                try:
                    strategy_results[strategy_type] = future.result()
                except Exception as e:
                    self.logger.error(f"Strategy {strategy_type} failed: {e}")
            
            # Aggregate ensemble results
            for i in range(len(df)):
                ensemble_result = self._aggregate_ensemble(
                    {s: strategy_results[s][i] if i < len(strategy_results[s]) else None
                     for s in strategy_results},
                    df.iloc[i].to_dict()
                )
                results.append(ensemble_result)
        else:
            # Process each transaction
            for _, row in df.iterrows():
                transaction_dict = row.to_dict()
                
                # Check cache
                cache_key = self._get_cache_key(transaction_dict)
                cached_result = self._get_cached_result(cache_key)
                
                if cached_result:
                    results.append(cached_result)
                    continue
                
                # Analyze transaction
                result = self._analyze_single_transaction(
                    transaction_dict, strategy, generate_proof
                )
                
                # Cache result
                self._cache_result(cache_key, result)
                results.append(result)
        
        return results
    
    def _analyze_single_transaction(self, transaction: Dict[str, Any],
                                  strategy: DetectionStrategy,
                                  generate_proof: bool) -> FraudDetectionResult:
        """Analyze a single transaction"""
        start_time = time.time()
        
        # Initialize result components
        ml_scores = {}
        rule_violations = []
        symbolic_inferences = []
        anomaly_indicators = {}
        explanations = []
        
        # Apply detection based on strategy
        if strategy in [DetectionStrategy.RULES_ONLY, DetectionStrategy.HYBRID]:
            rule_score, violations = self._apply_rules(transaction)
            rule_violations = violations
            anomaly_indicators['rule_score'] = rule_score
            
            if violations:
                explanations.extend([f"Rule violation: {v}" for v in violations])
        
        if strategy in [DetectionStrategy.ML_ONLY, DetectionStrategy.HYBRID] and self.ml_predictor:
            try:
                ml_result = self.ml_predictor.predict(transaction)
                ml_scores = ml_result.get('scores', {})
                anomaly_indicators['ml_score'] = ml_result.get('fraud_score', 0.0)
                
                if ml_result.get('fraud_score', 0) > 0.5:
                    explanations.append(f"ML detected anomaly (score: {ml_result['fraud_score']:.2f})")
            except Exception as e:
                self.logger.warning(f"ML prediction failed: {e}")
        
        if strategy in [DetectionStrategy.SYMBOLIC_ONLY, DetectionStrategy.HYBRID] and self.symbolic_reasoner:
            try:
                # Add transaction as fact
                fact_id = self.symbolic_reasoner.add_fact({
                    'fact_type': 'TRANSACTION',
                    'subject': transaction.get('transaction_id', 'unknown'),
                    'predicate': 'has_properties',
                    'object': transaction,
                    'source': 'fraud_detection'
                })
                
                # Perform reasoning
                inferences = self.symbolic_reasoner.reason()
                symbolic_inferences = [
                    {
                        'conclusion': inf.conclusion.to_triple(),
                        'confidence': inf.confidence
                    }
                    for inf in inferences
                ]
                
                if symbolic_inferences:
                    anomaly_indicators['symbolic_score'] = max(
                        inf['confidence'] for inf in symbolic_inferences
                    )
                    explanations.append("Symbolic reasoning detected patterns")
            except Exception as e:
                self.logger.warning(f"Symbolic reasoning failed: {e}")
        
        # Calculate final fraud score
        fraud_score = self._calculate_fraud_score(
            anomaly_indicators, strategy
        )
        
        # Determine risk level
        risk_level = self._get_risk_level(fraud_score)
        is_fraud = fraud_score >= self.config.get('fraud_threshold', 0.7)
        
        # Generate proof if requested
        if generate_proof and is_fraud and self.proof_verifier:
            try:
                proof = self.proof_verifier.generate_proof({
                    'transaction_id': transaction.get('transaction_id', 'unknown'),
                    'fraud_probability': fraud_score,
                    'risk_score': fraud_score,
                    'evidence': {
                        'anomaly_indicators': anomaly_indicators,
                        'rule_violations': rule_violations
                    }
                })
                if proof:
                    explanations.append("Verifiable proof generated")
            except Exception as e:
                self.logger.warning(f"Proof generation failed: {e}")
        
        # Create result
        processing_time = (time.time() - start_time) * 1000
        
        return FraudDetectionResult(
            transaction_id=transaction.get('transaction_id', 'unknown'),
            fraud_score=fraud_score,
            is_fraud=is_fraud,
            risk_level=risk_level,
            detection_method=strategy.value,
            confidence=self._calculate_confidence(anomaly_indicators, strategy),
            explanations=explanations if explanations else ["Normal transaction pattern"],
            anomaly_indicators=anomaly_indicators,
            ml_scores=ml_scores,
            rule_violations=rule_violations,
            symbolic_inferences=symbolic_inferences,
            processing_time_ms=processing_time
        )
    
    def _apply_rules(self, transaction: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Apply rule-based detection"""
        violations = []
        scores = []
        
        for rule_name, rule_func in self.rule_engine.items():
            try:
                if rule_func(transaction):
                    violations.append(rule_name)
                    scores.append(self._get_rule_weight(rule_name))
            except Exception as e:
                self.logger.debug(f"Rule {rule_name} failed: {e}")
        
        # Calculate rule score
        if scores:
            rule_score = min(sum(scores), 1.0)
        else:
            rule_score = 0.0
        
        return rule_score, violations
    
    def _calculate_fraud_score(self, indicators: Dict[str, float],
                             strategy: DetectionStrategy) -> float:
        """Calculate final fraud score based on indicators"""
        if not indicators:
            return 0.0
        
        if strategy == DetectionStrategy.RULES_ONLY:
            return indicators.get('rule_score', 0.0)
        elif strategy == DetectionStrategy.ML_ONLY:
            return indicators.get('ml_score', 0.0)
        elif strategy == DetectionStrategy.SYMBOLIC_ONLY:
            return indicators.get('symbolic_score', 0.0)
        elif strategy == DetectionStrategy.HYBRID:
            # Weighted average
            weights = {
                'rule_score': 0.3,
                'ml_score': 0.5,
                'symbolic_score': 0.2
            }
            total_weight = sum(weights.get(k, 0) for k in indicators)
            if total_weight > 0:
                return sum(
                    indicators.get(k, 0) * weights.get(k, 0)
                    for k in indicators
                ) / total_weight
        
        return max(indicators.values())
    
    def _calculate_confidence(self, indicators: Dict[str, float],
                            strategy: DetectionStrategy) -> float:
        """Calculate confidence in detection"""
        if not indicators:
            return 0.0
        
        # Higher confidence when multiple methods agree
        if len(indicators) > 1:
            scores = list(indicators.values())
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Low standard deviation means high agreement
            if std_score < 0.1:
                return min(mean_score + 0.2, 1.0)
            else:
                return mean_score
        else:
            return list(indicators.values())[0]
    
    def _aggregate_ensemble(self, strategy_results: Dict[DetectionStrategy, Optional[FraudDetectionResult]],
                          transaction: Dict[str, Any]) -> FraudDetectionResult:
        """Aggregate results from multiple strategies"""
        valid_results = [r for r in strategy_results.values() if r is not None]
        
        if not valid_results:
            # Fallback result
            return FraudDetectionResult(
                transaction_id=transaction.get('transaction_id', 'unknown'),
                fraud_score=0.0,
                is_fraud=False,
                risk_level=FraudRiskLevel.VERY_LOW,
                detection_method=DetectionStrategy.ENSEMBLE.value,
                confidence=0.0,
                explanations=["No detection methods available"],
                anomaly_indicators={}
            )
        
        # Aggregate scores
        fraud_scores = [r.fraud_score for r in valid_results]
        mean_score = np.mean(fraud_scores)
        
        # Voting for is_fraud
        fraud_votes = sum(1 for r in valid_results if r.is_fraud)
        is_fraud = fraud_votes > len(valid_results) / 2
        
        # Combine explanations
        all_explanations = []
        for r in valid_results:
            all_explanations.extend(r.explanations)
        
        # Combine anomaly indicators
        all_indicators = {}
        for r in valid_results:
            for k, v in r.anomaly_indicators.items():
                all_indicators[f"{r.detection_method}_{k}"] = v
        
        return FraudDetectionResult(
            transaction_id=transaction.get('transaction_id', 'unknown'),
            fraud_score=mean_score,
            is_fraud=is_fraud,
            risk_level=self._get_risk_level(mean_score),
            detection_method=DetectionStrategy.ENSEMBLE.value,
            confidence=self._calculate_ensemble_confidence(valid_results),
            explanations=list(set(all_explanations)),
            anomaly_indicators=all_indicators,
            processing_time_ms=sum(r.processing_time_ms for r in valid_results)
        )
    
    def _calculate_ensemble_confidence(self, results: List[FraudDetectionResult]) -> float:
        """Calculate confidence for ensemble result"""
        if not results:
            return 0.0
        
        # Agreement between methods increases confidence
        fraud_scores = [r.fraud_score for r in results]
        std_dev = np.std(fraud_scores)
        
        base_confidence = np.mean([r.confidence for r in results])
        
        # Bonus for agreement
        if std_dev < 0.1:
            return min(base_confidence + 0.1, 1.0)
        else:
            return base_confidence
    
    def analyze_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single transaction in detail.
        
        Args:
            transaction: Transaction data
            
        Returns:
            Detailed analysis results
        """
        try:
            self.logger.info(f"Analyzing transaction {transaction.get('transaction_id', 'unknown')}")
            
            # Run detection with all strategies
            results = {}
            for strategy in DetectionStrategy:
                try:
                    result = self.detect_fraud(transaction, strategy=strategy, generate_proof=False)
                    results[strategy.value] = {
                        'fraud_score': result.fraud_score,
                        'is_fraud': result.is_fraud,
                        'risk_level': result.risk_level.value,
                        'confidence': result.confidence,
                        'explanations': result.explanations,
                        'processing_time_ms': result.processing_time_ms
                    }
                except Exception as e:
                    self.logger.warning(f"Strategy {strategy.value} failed: {e}")
                    results[strategy.value] = {'error': str(e)}
            
            # Add transaction metadata
            analysis = {
                'transaction_id': transaction.get('transaction_id', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'transaction_data': transaction,
                'detection_results': results,
                'recommendations': self._generate_recommendations(results)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Transaction analysis failed: {e}")
            raise
    
    def aggregate_results(self, results: List[FraudDetectionResult]) -> Dict[str, Any]:
        """
        Aggregate multiple fraud detection results.
        
        Args:
            results: List of detection results
            
        Returns:
            Aggregated statistics and insights
        """
        if not results:
            return {'error': 'No results to aggregate'}
        
        try:
            # Basic statistics
            fraud_count = sum(1 for r in results if r.is_fraud)
            total_count = len(results)
            
            # Risk level distribution
            risk_distribution = defaultdict(int)
            for r in results:
                risk_distribution[r.risk_level.value] += 1
            
            # Method performance
            method_stats = defaultdict(lambda: {
                'count': 0, 'fraud_detected': 0, 'avg_score': 0.0,
                'avg_confidence': 0.0, 'avg_time_ms': 0.0
            })
            
            for r in results:
                method = r.detection_method
                method_stats[method]['count'] += 1
                if r.is_fraud:
                    method_stats[method]['fraud_detected'] += 1
                method_stats[method]['avg_score'] += r.fraud_score
                method_stats[method]['avg_confidence'] += r.confidence
                method_stats[method]['avg_time_ms'] += r.processing_time_ms
            
            # Calculate averages
            for method, stats in method_stats.items():
                count = stats['count']
                if count > 0:
                    stats['avg_score'] /= count
                    stats['avg_confidence'] /= count
                    stats['avg_time_ms'] /= count
                    stats['detection_rate'] = stats['fraud_detected'] / count
            
            # Time-based analysis
            hourly_stats = defaultdict(lambda: {'count': 0, 'fraud_count': 0})
            for r in results:
                hour_key = r.timestamp.strftime('%Y-%m-%d %H:00')
                hourly_stats[hour_key]['count'] += 1
                if r.is_fraud:
                    hourly_stats[hour_key]['fraud_count'] += 1
            
            # Common explanations
            explanation_counts = defaultdict(int)
            for r in results:
                for exp in r.explanations:
                    explanation_counts[exp] += 1
            
            # Top explanations
            top_explanations = sorted(
                explanation_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            aggregation = {
                'summary': {
                    'total_transactions': total_count,
                    'fraud_detected': fraud_count,
                    'fraud_rate': fraud_count / total_count if total_count > 0 else 0,
                    'average_fraud_score': np.mean([r.fraud_score for r in results]),
                    'average_confidence': np.mean([r.confidence for r in results]),
                    'average_processing_time_ms': np.mean([r.processing_time_ms for r in results])
                },
                'risk_distribution': dict(risk_distribution),
                'method_performance': dict(method_stats),
                'hourly_statistics': dict(hourly_stats),
                'top_explanations': top_explanations,
                'timestamp': datetime.now().isoformat()
            }
            
            return aggregation
            
        except Exception as e:
            self.logger.error(f"Result aggregation failed: {e}")
            raise
    
    def validate_detection(self, result: FraudDetectionResult,
                         ground_truth: Optional[bool] = None) -> Dict[str, Any]:
        """
        Validate fraud detection result.
        
        Args:
            result: Detection result to validate
            ground_truth: Actual fraud status if known
            
        Returns:
            Validation results
        """
        validation = {
            'result_id': result.transaction_id,
            'timestamp': datetime.now().isoformat(),
            'is_valid': True,
            'issues': []
        }
        
        # Validate score range
        if not 0 <= result.fraud_score <= 1:
            validation['is_valid'] = False
            validation['issues'].append(f"Invalid fraud score: {result.fraud_score}")
        
        # Validate risk level consistency
        expected_risk = self._get_risk_level(result.fraud_score)
        if result.risk_level != expected_risk:
            validation['issues'].append(
                f"Risk level mismatch: {result.risk_level.value} vs expected {expected_risk.value}"
            )
        
        # Validate confidence
        if not 0 <= result.confidence <= 1:
            validation['is_valid'] = False
            validation['issues'].append(f"Invalid confidence: {result.confidence}")
        
        # Validate detection consistency
        threshold = self.config.get('fraud_threshold', 0.7)
        expected_fraud = result.fraud_score >= threshold
        if result.is_fraud != expected_fraud:
            validation['issues'].append(
                f"Fraud flag inconsistent with score: {result.is_fraud} vs score {result.fraud_score}"
            )
        
        # Update metrics if ground truth provided
        if ground_truth is not None:
            validation['ground_truth'] = ground_truth
            validation['classification'] = self._classify_prediction(
                result.is_fraud, ground_truth
            )
            
            # Update global metrics
            with self._lock:
                self._update_classification_metrics(
                    result.is_fraud, ground_truth
                )
        
        validation['is_valid'] = validation['is_valid'] and len(validation['issues']) == 0
        
        return validation
    
    def get_metrics(self) -> FraudDetectionMetrics:
        """Get current fraud detection metrics"""
        with self._lock:
            self.metrics.update_classification_metrics()
            return self.metrics
    
    def reset_metrics(self):
        """Reset fraud detection metrics"""
        with self._lock:
            self.metrics = FraudDetectionMetrics()
            self.logger.info("Metrics reset")
    
    def persist_results(self, results: List[FraudDetectionResult]) -> bool:
        """
        Persist detection results to storage.
        
        Args:
            results: Results to persist
            
        Returns:
            Success status
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = self.storage_path / f'detection_results_{timestamp}.pkl'
            
            with open(filename, 'wb') as f:
                pickle.dump(results, f)
            
            self.logger.info(f"Persisted {len(results)} results to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to persist results: {e}")
            return False
    
    def load_results(self, filename: str) -> Optional[List[FraudDetectionResult]]:
        """
        Load detection results from storage.
        
        Args:
            filename: Filename to load from
            
        Returns:
            Loaded results or None
        """
        try:
            filepath = self.storage_path / filename
            
            with open(filepath, 'rb') as f:
                results = pickle.load(f)
            
            self.logger.info(f"Loaded {len(results)} results from {filepath}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to load results: {e}")
            return None
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export metrics in JSON-serializable format"""
        metrics = self.get_metrics()
        
        return {
            'summary': {
                'total_transactions': metrics.total_transactions,
                'fraud_detected': metrics.fraud_detected,
                'detection_rate': metrics.detection_rate,
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score
            },
            'performance': {
                'avg_processing_time_ms': metrics.average_processing_time_ms,
                'max_processing_time_ms': metrics.max_processing_time_ms,
                'min_processing_time_ms': metrics.min_processing_time_ms
            },
            'classification': {
                'true_positives': metrics.true_positives,
                'false_positives': metrics.false_positives,
                'true_negatives': metrics.true_negatives,
                'false_negatives': metrics.false_negatives
            },
            'strategy_metrics': metrics.metrics_by_strategy,
            'hourly_detection_rate': metrics.hourly_detection_rate,
            'export_timestamp': datetime.now().isoformat()
        }
    
    # Helper methods
    
    def _get_risk_level(self, score: float) -> FraudRiskLevel:
        """Convert fraud score to risk level."""
        if score >= 0.9:
            return FraudRiskLevel.CRITICAL
        elif score >= 0.7:
            return FraudRiskLevel.HIGH
        elif score >= 0.5:
            return FraudRiskLevel.MEDIUM
        elif score >= 0.3:
            return FraudRiskLevel.LOW
        else:
            return FraudRiskLevel.VERY_LOW
    
    def _is_unusual_time(self, transaction: Dict[str, Any]) -> bool:
        """Check if transaction occurred at unusual time"""
        try:
            timestamp = pd.to_datetime(transaction.get('timestamp', datetime.now()))
            hour = timestamp.hour
            return hour < 6 or hour > 22
        except:
            return False
    
    def _check_velocity(self, transaction: Dict[str, Any]) -> bool:
        """Check transaction velocity"""
        # Simplified velocity check - in production would check against history
        return False
    
    def _check_geographic(self, transaction: Dict[str, Any]) -> bool:
        """Check for geographic anomalies"""
        # Simplified geographic check
        return False
    
    def _check_merchant_risk(self, transaction: Dict[str, Any]) -> bool:
        """Check merchant risk level"""
        # Simplified merchant risk check
        merchant = transaction.get('merchant_name', '').lower()
        risky_keywords = ['casino', 'gambling', 'crypto', 'forex']
        return any(keyword in merchant for keyword in risky_keywords)
    
    def _get_rule_weight(self, rule_name: str) -> float:
        """Get weight for a specific rule"""
        weights = {
            'high_amount': 0.4,
            'unusual_time': 0.3,
            'rapid_velocity': 0.5,
            'geographic_anomaly': 0.6,
            'merchant_risk': 0.3
        }
        return weights.get(rule_name, 0.3)
    
    def _get_cache_key(self, transaction: Dict[str, Any]) -> str:
        """Generate cache key for transaction"""
        # Use transaction ID if available
        if 'transaction_id' in transaction:
            return f"tx_{transaction['transaction_id']}"
        
        # Otherwise use hash of transaction data
        import hashlib
        tx_str = json.dumps(transaction, sort_keys=True)
        return f"tx_{hashlib.md5(tx_str.encode()).hexdigest()}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[FraudDetectionResult]:
        """Get cached detection result"""
        with self._lock:
            if cache_key in self._detection_cache:
                result, timestamp = self._detection_cache[cache_key]
                
                # Check if cache is still valid
                if datetime.now() - timestamp < self._cache_ttl:
                    self.logger.debug(f"Cache hit for {cache_key}")
                    return result
                else:
                    # Remove expired entry
                    del self._detection_cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: FraudDetectionResult):
        """Cache detection result"""
        with self._lock:
            self._detection_cache[cache_key] = (result, datetime.now())
            
            # Limit cache size
            max_cache_size = self.config.get('max_cache_size', 1000)
            if len(self._detection_cache) > max_cache_size:
                # Remove oldest entries
                sorted_keys = sorted(
                    self._detection_cache.keys(),
                    key=lambda k: self._detection_cache[k][1]
                )
                for key in sorted_keys[:len(self._detection_cache) - max_cache_size]:
                    del self._detection_cache[key]
    
    def _classify_prediction(self, predicted: bool, actual: bool) -> str:
        """Classify prediction result"""
        if predicted and actual:
            return 'true_positive'
        elif predicted and not actual:
            return 'false_positive'
        elif not predicted and actual:
            return 'false_negative'
        else:
            return 'true_negative'
    
    def _update_classification_metrics(self, predicted: bool, actual: bool):
        """Update classification metrics"""
        classification = self._classify_prediction(predicted, actual)
        
        if classification == 'true_positive':
            self.metrics.true_positives += 1
        elif classification == 'false_positive':
            self.metrics.false_positives += 1
        elif classification == 'false_negative':
            self.metrics.false_negatives += 1
        else:
            self.metrics.true_negatives += 1
    
    def _update_metrics(self, results: List[FraudDetectionResult], 
                       batch_processing_time: float):
        """Update metrics with detection results"""
        with self._lock:
            # Update counts
            self.metrics.total_transactions += len(results)
            self.metrics.fraud_detected += sum(1 for r in results if r.is_fraud)
            
            # Update processing times
            processing_times = [r.processing_time_ms for r in results]
            if processing_times:
                avg_time = np.mean(processing_times)
                max_time = max(processing_times)
                min_time = min(processing_times)
                
                # Update running average
                if self.metrics.average_processing_time_ms == 0:
                    self.metrics.average_processing_time_ms = avg_time
                else:
                    # Exponential moving average
                    alpha = 0.1
                    self.metrics.average_processing_time_ms = (
                        alpha * avg_time + 
                        (1 - alpha) * self.metrics.average_processing_time_ms
                    )
                
                self.metrics.max_processing_time_ms = max(
                    self.metrics.max_processing_time_ms, max_time
                )
                self.metrics.min_processing_time_ms = min(
                    self.metrics.min_processing_time_ms, min_time
                )
            
            # Update hourly stats
            hour_key = datetime.now().strftime('%Y-%m-%d_%H')
            if hour_key not in self.metrics.hourly_detection_rate:
                self.metrics.hourly_detection_rate[hour_key] = 0.0
            
            hourly_fraud = sum(1 for r in results if r.is_fraud)
            if len(results) > 0:
                self.metrics.hourly_detection_rate[hour_key] = hourly_fraud / len(results)
            
            # Update strategy metrics
            for r in results:
                strategy = DetectionStrategy(r.detection_method)
                if strategy not in self.metrics.metrics_by_strategy:
                    self.metrics.metrics_by_strategy[strategy] = {
                        'count': 0,
                        'fraud_detected': 0,
                        'avg_score': 0.0,
                        'avg_confidence': 0.0,
                        'avg_time': 0.0
                    }
                
                strategy_metrics = self.metrics.metrics_by_strategy[strategy]
                strategy_metrics['count'] += 1
                if r.is_fraud:
                    strategy_metrics['fraud_detected'] += 1
                
                # Update averages
                n = strategy_metrics['count']
                strategy_metrics['avg_score'] = (
                    (strategy_metrics['avg_score'] * (n - 1) + r.fraud_score) / n
                )
                strategy_metrics['avg_confidence'] = (
                    (strategy_metrics['avg_confidence'] * (n - 1) + r.confidence) / n
                )
                strategy_metrics['avg_time'] = (
                    (strategy_metrics['avg_time'] * (n - 1) + r.processing_time_ms) / n
                )
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Check for consistent high risk
        high_risk_count = sum(
            1 for r in results.values()
            if isinstance(r, dict) and r.get('risk_level') in ['high', 'critical']
        )
        
        if high_risk_count >= 2:
            recommendations.append("Multiple strategies indicate high risk - recommend manual review")
        
        # Check for method disagreement
        fraud_flags = [
            r.get('is_fraud', False) 
            for r in results.values() 
            if isinstance(r, dict)
        ]
        
        if fraud_flags and not all(f == fraud_flags[0] for f in fraud_flags):
            recommendations.append("Detection methods disagree - consider additional analysis")
        
        # Performance recommendations
        slow_methods = [
            method for method, r in results.items()
            if isinstance(r, dict) and r.get('processing_time_ms', 0) > 100
        ]
        
        if slow_methods:
            recommendations.append(
                f"Consider optimizing slow methods: {', '.join(slow_methods)}"
            )
        
        return recommendations
    
    def __repr__(self) -> str:
        return (f"FinancialFraudCore(strategy={self.config.get('detection_strategy')}, "
               f"components={sum(1 for c in [self.data_loader, self.preprocessor, "
               f"self.ml_predictor, self.symbolic_reasoner, self.proof_verifier] if c)}/5)")


# Maintain backward compatibility
FraudDetectionCore = FinancialFraudCore


if __name__ == "__main__":
    # Example usage
    import logging
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize fraud detection core
    fraud_core = FinancialFraudCore()
    
    # Example transaction
    transaction = {
        'transaction_id': 'TX123456',
        'amount': 15000,
        'timestamp': datetime.now(),
        'merchant_name': 'Online Store',
        'card_number': '****1234',
        'location': 'New York, NY'
    }
    
    print("\n=== Single Transaction Detection ===")
    result = fraud_core.detect_fraud(transaction)
    print(f"Transaction ID: {result.transaction_id}")
    print(f"Fraud Score: {result.fraud_score:.2f}")
    print(f"Is Fraud: {result.is_fraud}")
    print(f"Risk Level: {result.risk_level.value}")
    print(f"Explanations: {', '.join(result.explanations)}")
    
    print("\n=== Transaction Analysis ===")
    analysis = fraud_core.analyze_transaction(transaction)
    print(f"Analyzed with {len(analysis['detection_results'])} strategies")
    for strategy, result in analysis['detection_results'].items():
        if 'error' not in result:
            print(f"  {strategy}: score={result['fraud_score']:.2f}, "
                  f"risk={result['risk_level']}")
    
    print("\n=== Batch Detection ===")
    # Create batch of transactions
    transactions = pd.DataFrame([
        {
            'transaction_id': f'TX{i}',
            'amount': np.random.uniform(100, 20000),
            'timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 24)),
            'merchant_name': np.random.choice(['Store A', 'Store B', 'Casino XYZ']),
        }
        for i in range(10)
    ])
    
    batch_results = fraud_core.detect_fraud(transactions)
    
    print(f"Processed {len(batch_results)} transactions")
    fraud_count = sum(1 for r in batch_results if r.is_fraud)
    print(f"Fraud detected: {fraud_count}/{len(batch_results)}")
    
    # Aggregate results
    print("\n=== Result Aggregation ===")
    aggregation = fraud_core.aggregate_results(batch_results)
    print(f"Average fraud score: {aggregation['summary']['average_fraud_score']:.2f}")
    print(f"Fraud rate: {aggregation['summary']['fraud_rate']:.2%}")
    print(f"Risk distribution: {aggregation['risk_distribution']}")
    
    # Get metrics
    print("\n=== Performance Metrics ===")
    metrics = fraud_core.get_metrics()
    print(f"Total transactions: {metrics.total_transactions}")
    print(f"Average processing time: {metrics.average_processing_time_ms:.2f}ms")
    
    print("\nFraud detection core operational!")