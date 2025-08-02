"""
Financial Fraud Detection Domain Routing - Enhanced with Validation and Error Handling
Specialized routing implementation for fraud detection that extends the Universal AI Core Brain routing system
Now includes comprehensive validation framework with enhanced security checks, performance monitoring,
error recovery, and production-ready reliability features.
"""

import logging
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import json
import hashlib
from collections import defaultdict

# Import base routing components from independent_core
try:
    from independent_core.domain_router import (
        DomainRouter, DomainPattern, RoutingResult, 
        RoutingStrategy, RoutingConfidence
    )
except ImportError:
    # Fallback for testing - define minimal interfaces
    from domain_router import (
        DomainRouter, DomainPattern, RoutingResult,
        RoutingStrategy, RoutingConfidence
    )

# Enhanced imports for production-ready features
try:
    from enhanced_domain_routing import (
        EnhancedFraudRoutingContext, EnhancedRoutingValidationConfig,
        EnhancedTransactionValidator, EnhancedPatternValidator, EnhancedSecurityValidator,
        EnhancedCircuitBreaker, EnhancedPerformanceMonitor,
        ValidationLevel, ValidationMode, SecurityLevel, PerformanceMode,
        EnhancedFraudCategory, EnhancedFraudIndicator, EnhancedRoutingHealthStatus,
        EnhancedValidationContext, EnhancedValidationResult
    )
    ENHANCED_AVAILABLE = True
except ImportError:
    logger.warning("Enhanced routing features not available, using standard validation")
    ENHANCED_AVAILABLE = False


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ============= CUSTOM EXCEPTIONS =============

class FraudRoutingException(Exception):
    """Base exception for fraud routing errors"""
    def __init__(self, message: str, error_code: str = "FRAUD_ROUTING_ERROR", details: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now()


class ValidationException(FraudRoutingException):
    """Exception for validation errors"""
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(
            message, 
            error_code="VALIDATION_ERROR",
            details={"field": field, "value": value}
        )


class TransactionValidationException(ValidationException):
    """Exception for transaction validation errors"""
    def __init__(self, message: str, transaction_data: Dict[str, Any] = None):
        super().__init__(message, field="transaction")
        self.details["transaction_data"] = transaction_data


class PatternValidationException(ValidationException):
    """Exception for pattern validation errors"""
    def __init__(self, message: str, pattern: str = None):
        super().__init__(message, field="pattern", value=pattern)


class ConfidenceValidationException(ValidationException):
    """Exception for confidence score validation errors"""
    def __init__(self, message: str, confidence_score: float = None):
        super().__init__(message, field="confidence_score", value=confidence_score)


class RoutingRuleException(ValidationException):
    """Exception for routing rule validation errors"""
    def __init__(self, message: str, rule: Dict[str, Any] = None):
        super().__init__(message, field="routing_rule")
        self.details["rule"] = rule


class RoutingTimeoutException(FraudRoutingException):
    """Exception for routing timeout errors"""
    def __init__(self, message: str, timeout_ms: float = None):
        super().__init__(
            message,
            error_code="ROUTING_TIMEOUT",
            details={"timeout_ms": timeout_ms}
        )


class RoutingPerformanceException(FraudRoutingException):
    """Exception for routing performance issues"""
    def __init__(self, message: str, performance_metrics: Dict[str, Any] = None):
        super().__init__(
            message,
            error_code="PERFORMANCE_ERROR",
            details={"metrics": performance_metrics}
        )


class RoutingSecurityException(FraudRoutingException):
    """Exception for routing security violations"""
    def __init__(self, message: str, security_context: Dict[str, Any] = None):
        super().__init__(
            message,
            error_code="SECURITY_ERROR",
            details={"security_context": security_context}
        )


class RoutingIntegrationException(FraudRoutingException):
    """Exception for routing integration failures"""
    def __init__(self, message: str, integration_point: str = None):
        super().__init__(
            message,
            error_code="INTEGRATION_ERROR",
            details={"integration_point": integration_point}
        )


# ============= VALIDATION CLASSES =============

@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validated_data: Optional[Dict[str, Any]] = None
    validation_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'validated_data': self.validated_data,
            'validation_time_ms': self.validation_time_ms
        }


@dataclass
class RoutingValidationConfig:
    """Configuration for routing validation"""
    # Transaction validation
    min_transaction_amount: float = 0.01
    max_transaction_amount: float = 10_000_000.0
    allowed_transaction_types: Set[str] = field(default_factory=lambda: {
        "purchase", "withdrawal", "transfer", "payment", "refund", "deposit"
    })
    transaction_id_pattern: str = r'^[A-Z0-9]{8,32}$'
    
    # Pattern validation
    max_pattern_length: int = 1000
    max_keyword_length: int = 100
    max_regex_complexity: int = 50
    
    # Confidence validation
    min_confidence_score: float = 0.0
    max_confidence_score: float = 1.0
    confidence_precision: int = 3
    
    # Performance validation
    max_routing_time_ms: float = 1000.0
    max_validation_time_ms: float = 500.0
    performance_check_interval: int = 100  # Check every N requests
    
    # Security validation
    max_request_size_bytes: int = 1_048_576  # 1MB
    allowed_input_types: Set[str] = field(default_factory=lambda: {
        "text", "transaction", "payment_data", "fraud_check", 
        "risk_assessment", "compliance", "investigation"
    })
    enable_sql_injection_check: bool = True
    enable_xss_check: bool = True
    
    # Reliability validation
    min_health_check_interval_seconds: int = 60
    max_consecutive_failures: int = 5
    circuit_breaker_threshold: float = 0.5
    circuit_breaker_timeout_seconds: int = 300


# ============= ENHANCED ENUMS =============

class FraudRoutingCategory(Enum):
    """Categories for fraud detection routing"""
    TRANSACTION_ANALYSIS = "transaction_analysis"
    PATTERN_DETECTION = "pattern_detection"
    RISK_ASSESSMENT = "risk_assessment"
    COMPLIANCE_CHECK = "compliance_check"
    ALERT_GENERATION = "alert_generation"
    ANOMALY_DETECTION = "anomaly_detection"
    FRAUD_INVESTIGATION = "fraud_investigation"
    REPORTING = "reporting"
    MONITORING = "monitoring"


class FraudIndicatorType(Enum):
    """Types of fraud indicators for routing"""
    HIGH_AMOUNT = "high_amount"
    VELOCITY = "velocity"
    GEOGRAPHIC_ANOMALY = "geographic_anomaly"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    BLACKLIST_MATCH = "blacklist_match"
    PATTERN_MATCH = "pattern_match"
    ML_PREDICTION = "ml_prediction"
    RULE_VIOLATION = "rule_violation"
    COMPLIANCE_FLAG = "compliance_flag"


class RoutingHealthStatus(Enum):
    """Health status for routing system"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CIRCUIT_OPEN = "circuit_open"


# ============= ENHANCED CONTEXT AND RESULT CLASSES =============

@dataclass
class FraudRoutingContext:
    """Context for fraud-specific routing decisions"""
    transaction_amount: Optional[float] = None
    transaction_type: Optional[str] = None
    transaction_id: Optional[str] = None
    merchant_category: Optional[str] = None
    user_risk_profile: Optional[str] = None
    geographic_data: Optional[Dict[str, Any]] = None
    time_data: Optional[Dict[str, Any]] = None
    historical_data: Optional[Dict[str, Any]] = None
    compliance_requirements: List[str] = field(default_factory=list)
    fraud_indicators: List[FraudIndicatorType] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    security_context: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate request ID if not provided"""
        if not self.request_id:
            self.request_id = self._generate_request_id()
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        timestamp = datetime.now().isoformat()
        data = f"{timestamp}_{self.transaction_id or 'unknown'}"
        return hashlib.md5(data.encode()).hexdigest()[:16]


@dataclass
class FraudRoutingResult(RoutingResult):
    """Extended routing result for fraud detection"""
    fraud_category: Optional[FraudRoutingCategory] = None
    risk_score: float = 0.0
    fraud_indicators: List[FraudIndicatorType] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    compliance_flags: List[str] = field(default_factory=list)
    investigation_priority: str = "normal"
    routing_context: Optional[FraudRoutingContext] = None
    validation_result: Optional[ValidationResult] = None
    error_recovery_applied: bool = False
    fallback_reason: Optional[str] = None


# ============= VALIDATOR CLASSES =============

class TransactionValidator:
    """Validator for transaction data"""
    
    def __init__(self, config: RoutingValidationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate(self, transaction_data: Dict[str, Any]) -> ValidationResult:
        """Validate transaction data"""
        start_time = time.time()
        errors = []
        warnings = []
        validated_data = {}
        
        try:
            # Validate transaction amount
            amount = transaction_data.get('amount')
            if amount is None:
                errors.append("Transaction amount is required")
            else:
                try:
                    amount = float(amount)
                    if amount < self.config.min_transaction_amount:
                        errors.append(f"Transaction amount {amount} is below minimum {self.config.min_transaction_amount}")
                    elif amount > self.config.max_transaction_amount:
                        errors.append(f"Transaction amount {amount} exceeds maximum {self.config.max_transaction_amount}")
                    else:
                        validated_data['amount'] = amount
                except (TypeError, ValueError):
                    errors.append(f"Invalid transaction amount: {amount}")
            
            # Validate transaction type
            tx_type = transaction_data.get('type')
            if tx_type:
                if tx_type not in self.config.allowed_transaction_types:
                    errors.append(f"Invalid transaction type: {tx_type}")
                else:
                    validated_data['type'] = tx_type
            else:
                warnings.append("Transaction type not specified")
            
            # Validate transaction ID
            tx_id = transaction_data.get('id')
            if tx_id:
                if not re.match(self.config.transaction_id_pattern, str(tx_id)):
                    errors.append(f"Invalid transaction ID format: {tx_id}")
                else:
                    validated_data['id'] = tx_id
            else:
                warnings.append("Transaction ID not provided")
            
            # Validate timestamp
            timestamp = transaction_data.get('timestamp')
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        datetime.fromisoformat(timestamp)
                    validated_data['timestamp'] = timestamp
                except ValueError:
                    errors.append(f"Invalid timestamp format: {timestamp}")
            
            # Validate merchant data
            merchant = transaction_data.get('merchant')
            if merchant and isinstance(merchant, dict):
                if 'name' in merchant:
                    validated_data['merchant'] = merchant
                else:
                    warnings.append("Merchant data missing required fields")
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                validated_data=validated_data if is_valid else None,
                validation_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            self.logger.error(f"Transaction validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                validation_time_ms=(time.time() - start_time) * 1000
            )


class PatternValidator:
    """Validator for routing patterns"""
    
    def __init__(self, config: RoutingValidationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate(self, pattern: DomainPattern) -> ValidationResult:
        """Validate domain pattern"""
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Validate pattern name
            if not pattern.domain_name:
                errors.append("Domain name is required")
            elif not re.match(r'^[a-z_]+[a-z0-9_]*$', pattern.domain_name):
                errors.append(f"Invalid domain name format: {pattern.domain_name}")
            
            # Validate keywords
            for keyword in pattern.keywords:
                if len(keyword) > self.config.max_keyword_length:
                    errors.append(f"Keyword too long: {keyword[:50]}...")
                if not keyword.strip():
                    errors.append("Empty keyword found")
            
            # Validate patterns
            for p in pattern.patterns:
                if len(p) > self.config.max_pattern_length:
                    errors.append(f"Pattern too long: {p[:50]}...")
            
            # Validate regex patterns
            for regex in pattern.regex_patterns:
                try:
                    compiled = re.compile(regex)
                    # Check complexity (simple heuristic)
                    if len(regex) > self.config.max_regex_complexity:
                        warnings.append(f"Complex regex pattern may impact performance: {regex[:30]}...")
                except re.error as e:
                    errors.append(f"Invalid regex pattern: {regex} - {str(e)}")
            
            # Validate priority
            if not 1 <= pattern.priority <= 10:
                errors.append(f"Priority must be between 1 and 10, got {pattern.priority}")
            
            # Validate confidence boost
            if not -1.0 <= pattern.confidence_boost <= 1.0:
                errors.append(f"Confidence boost must be between -1.0 and 1.0, got {pattern.confidence_boost}")
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                validation_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            self.logger.error(f"Pattern validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                validation_time_ms=(time.time() - start_time) * 1000
            )


class SecurityValidator:
    """Validator for security checks"""
    
    def __init__(self, config: RoutingValidationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Common SQL injection patterns
        self.sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b)",
            r"(--|\||;|\/\*|\*\/)",
            r"(\bOR\b\s*\d+\s*=\s*\d+)",
            r"(\bAND\b\s*\d+\s*=\s*\d+)"
        ]
        
        # Common XSS patterns
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>"
        ]
    
    def validate_input_security(self, input_data: Any) -> ValidationResult:
        """Validate input for security threats"""
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            input_str = str(input_data)
            
            # Check input size
            input_size = len(input_str.encode('utf-8'))
            if input_size > self.config.max_request_size_bytes:
                errors.append(f"Input size {input_size} bytes exceeds maximum {self.config.max_request_size_bytes}")
            
            # SQL injection check
            if self.config.enable_sql_injection_check:
                for pattern in self.sql_patterns:
                    if re.search(pattern, input_str, re.IGNORECASE):
                        errors.append(f"Potential SQL injection detected")
                        break
            
            # XSS check
            if self.config.enable_xss_check:
                for pattern in self.xss_patterns:
                    if re.search(pattern, input_str, re.IGNORECASE):
                        errors.append(f"Potential XSS attack detected")
                        break
            
            # Check for suspicious patterns
            if input_str.count('..') > 2:
                warnings.append("Multiple directory traversal patterns detected")
            
            if '\x00' in input_str:
                errors.append("Null byte detected in input")
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                validation_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            self.logger.error(f"Security validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Security validation error: {str(e)}"],
                validation_time_ms=(time.time() - start_time) * 1000
            )


# ============= CIRCUIT BREAKER =============

class CircuitBreaker:
    """Circuit breaker for handling failures"""
    
    def __init__(self, failure_threshold: float = 0.5, 
                 timeout_seconds: int = 300,
                 min_calls: int = 10):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.min_calls = min_calls
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = RoutingHealthStatus.HEALTHY
        self._lock = threading.RLock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == RoutingHealthStatus.CIRCUIT_OPEN:
                if self._should_attempt_reset():
                    self.state = RoutingHealthStatus.DEGRADED
                else:
                    raise RoutingIntegrationException(
                        "Circuit breaker is open",
                        integration_point="circuit_breaker"
                    )
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise
    
    def _record_success(self):
        """Record successful call"""
        with self._lock:
            self.success_count += 1
            if self.state == RoutingHealthStatus.DEGRADED:
                # Reset if we've had enough successful calls
                total_calls = self.success_count + self.failure_count
                if total_calls >= self.min_calls:
                    if self.failure_count / total_calls < self.failure_threshold:
                        self.state = RoutingHealthStatus.HEALTHY
                        self.failure_count = 0
                        self.success_count = 0
    
    def _record_failure(self):
        """Record failed call"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            total_calls = self.success_count + self.failure_count
            if total_calls >= self.min_calls:
                failure_rate = self.failure_count / total_calls
                if failure_rate >= self.failure_threshold:
                    self.state = RoutingHealthStatus.CIRCUIT_OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if not self.last_failure_time:
            return True
        
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.timeout_seconds
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        with self._lock:
            total_calls = self.success_count + self.failure_count
            failure_rate = self.failure_count / total_calls if total_calls > 0 else 0.0
            
            return {
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'failure_rate': failure_rate,
                'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None
            }


# ============= ENHANCED FINANCIAL FRAUD ROUTER =============

class FinancialFraudRouter:
    """
    Enhanced Financial Fraud Router with comprehensive validation and error handling
    """
    
    def __init__(self, base_router: Optional[DomainRouter] = None, 
                 config: Optional[Dict[str, Any]] = None,
                 use_enhanced: bool = True,
                 validation_level: Optional[ValidationLevel] = None):
        """Initialize the Enhanced Financial Fraud Router"""
        self.config = config or self._get_default_config()
        self.base_router = base_router or DomainRouter(config=self.config)
        self.use_enhanced = use_enhanced and ENHANCED_AVAILABLE
        
        # Enhanced validation configuration
        if self.use_enhanced:
            self.enhanced_config = EnhancedRoutingValidationConfig()
            if validation_level:
                self.enhanced_config.validation_level = validation_level
            self._update_enhanced_config(self.config.get('validation', {}))
            
            # Initialize enhanced validators
            self.enhanced_transaction_validator = EnhancedTransactionValidator(self.enhanced_config)
            self.enhanced_pattern_validator = EnhancedPatternValidator(self.enhanced_config)
            self.enhanced_security_validator = EnhancedSecurityValidator(self.enhanced_config)
            
            # Enhanced circuit breaker and monitoring
            self.enhanced_circuit_breaker = EnhancedCircuitBreaker(self.enhanced_config)
            self.enhanced_performance_monitor = EnhancedPerformanceMonitor(self.enhanced_config)
        else:
            # Fallback to standard validation
            self.validation_config = RoutingValidationConfig()
            self._update_validation_config(self.config.get('validation', {}))
        
        # Initialize standard validators (always available for backward compatibility)
        self.transaction_validator = TransactionValidator(
            self.enhanced_config if self.use_enhanced else self.validation_config
        )
        self.pattern_validator = PatternValidator(
            self.enhanced_config if self.use_enhanced else self.validation_config
        )
        self.security_validator = SecurityValidator(
            self.enhanced_config if self.use_enhanced else self.validation_config
        )
        
        # Circuit breaker for resilience
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.validation_config.circuit_breaker_threshold,
            timeout_seconds=self.validation_config.circuit_breaker_timeout_seconds
        )
        
        # Performance monitoring
        self.performance_monitor = {
            'request_count': 0,
            'total_routing_time': 0.0,
            'slow_requests': 0,
            'validation_failures': 0,
            'routing_failures': 0,
            'last_health_check': datetime.now()
        }
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Health status
        self.health_status = RoutingHealthStatus.HEALTHY
        self.consecutive_failures = 0
        
        # Initialize patterns with validation
        self._initialize_validated_patterns()
        
        logger.info("Enhanced FinancialFraudRouter initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration with validation settings"""
        return {
            'default_strategy': RoutingStrategy.HYBRID.value,
            'confidence_threshold': 0.8,
            'enable_fallback': True,
            'fallback_domain': 'fraud_investigation',
            'cache_max_size': 1000,
            'high_amount_threshold': 10000.0,
            'velocity_threshold': 5,
            'risk_score_threshold': 0.7,
            'enable_ml_routing': True,
            'enable_pattern_caching': True,
            'log_level': 'INFO',
            'validation': {
                'enable_transaction_validation': True,
                'enable_pattern_validation': True,
                'enable_security_validation': True,
                'enable_performance_validation': True,
                'max_routing_time_ms': 1000.0,
                'max_validation_time_ms': 500.0
            }
        }
    
    def _update_validation_config(self, validation_settings: Dict[str, Any]):
        """Update validation configuration from settings"""
        for key, value in validation_settings.items():
            if hasattr(self.validation_config, key):
                setattr(self.validation_config, key, value)
    
    def _update_enhanced_config(self, validation_settings: Dict[str, Any]):
        """Update enhanced validation configuration from settings"""
        if not self.use_enhanced:
            return
        
        for key, value in validation_settings.items():
            if hasattr(self.enhanced_config, key):
                setattr(self.enhanced_config, key, value)
    
    def _initialize_validated_patterns(self):
        """Initialize patterns with validation"""
        patterns = [
            DomainPattern(
                domain_name="transaction_analysis",
                keywords=["transaction", "payment", "transfer", "purchase", "withdrawal"],
                patterns=["analyze transaction", "check payment", "verify transfer"],
                regex_patterns=[
                    r'trans(action)?[\s_-]?id[\s:=]\w+',
                    r'amount[\s:=]\$?\d+\.?\d*'
                ],
                priority=9,
                confidence_boost=0.15
            ),
            DomainPattern(
                domain_name="fraud_detection",
                keywords=["fraud", "suspicious", "anomaly", "unusual", "alert"],
                patterns=["detect fraud", "suspicious activity", "fraud alert"],
                regex_patterns=[
                    r'fraud[\s_-]?score',
                    r'risk[\s_-]?level'
                ],
                priority=10,
                confidence_boost=0.2
            ),
            DomainPattern(
                domain_name="risk_assessment",
                keywords=["risk", "score", "assessment", "evaluate", "profile"],
                patterns=["assess risk", "calculate risk score"],
                priority=8,
                confidence_boost=0.1
            ),
            DomainPattern(
                domain_name="compliance_check",
                keywords=["compliance", "regulation", "aml", "kyc", "sanctions"],
                patterns=["compliance check", "regulatory review"],
                priority=9,
                confidence_boost=0.15
            )
        ]
        
        for pattern in patterns:
            try:
                # Validate pattern before adding
                validation_result = self.pattern_validator.validate(pattern)
                if validation_result.is_valid:
                    self.base_router.add_domain_pattern(pattern)
                    logger.info(f"Added validated pattern: {pattern.domain_name}")
                else:
                    logger.error(f"Pattern validation failed for {pattern.domain_name}: {validation_result.errors}")
            except Exception as e:
                logger.error(f"Failed to add pattern {pattern.domain_name}: {e}")
    
    def route_fraud_request(self, input_data: Any, 
                           context: Optional[FraudRoutingContext] = None,
                           strategy: Optional[RoutingStrategy] = None,
                           timeout_ms: Optional[float] = None) -> FraudRoutingResult:
        """
        Route fraud detection request with comprehensive validation and error handling
        """
        start_time = time.time()
        timeout_ms = timeout_ms or self.validation_config.max_routing_time_ms
        
        try:
            # Record request
            self.performance_monitor['request_count'] += 1
            
            # Step 1: Security validation
            if self.config.get('validation', {}).get('enable_security_validation', True):
                security_result = self._validate_security(input_data, context)
                if not security_result.is_valid:
                    raise RoutingSecurityException(
                        f"Security validation failed: {security_result.errors}",
                        security_context={'input_preview': str(input_data)[:100]}
                    )
            
            # Step 2: Input validation
            validation_results = self._validate_input(input_data, context)
            
            # Step 3: Route with circuit breaker
            future = self.executor.submit(
                self.circuit_breaker.call,
                self._perform_routing,
                input_data, context, strategy, validation_results
            )
            
            # Wait with timeout
            try:
                result = future.result(timeout=timeout_ms / 1000.0)
            except FutureTimeoutError:
                raise RoutingTimeoutException(
                    f"Routing timeout after {timeout_ms}ms",
                    timeout_ms=timeout_ms
                )
            
            # Step 4: Post-routing validation
            result = self._validate_routing_result(result)
            
            # Step 5: Performance monitoring
            routing_time = (time.time() - start_time) * 1000
            self._monitor_performance(routing_time)
            
            result.routing_time_ms = routing_time
            
            # Record success
            self.consecutive_failures = 0
            return result
            
        except FraudRoutingException:
            # Re-raise our custom exceptions
            raise
            
        except Exception as e:
            # Handle unexpected errors
            self.consecutive_failures += 1
            self.performance_monitor['routing_failures'] += 1
            
            logger.error(f"Routing failed: {e}")
            
            # Check if we should go into degraded mode
            if self.consecutive_failures >= self.validation_config.max_consecutive_failures:
                self.health_status = RoutingHealthStatus.DEGRADED
                logger.warning("Router entering degraded mode due to consecutive failures")
            
            # Attempt fallback routing
            return self._fallback_routing(input_data, context, str(e))
    
    def _validate_security(self, input_data: Any, 
                          context: Optional[FraudRoutingContext]) -> ValidationResult:
        """Perform security validation"""
        try:
            # Validate input data
            input_result = self.security_validator.validate_input_security(input_data)
            
            # Additional context-based security checks
            if context and context.security_context:
                # Check for suspicious patterns in security context
                if context.security_context.get('failed_attempts', 0) > 5:
                    input_result.errors.append("Too many failed attempts")
                
                if context.security_context.get('ip_blacklisted', False):
                    input_result.errors.append("Request from blacklisted IP")
            
            return input_result
            
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Security validation failed: {str(e)}"]
            )
    
    def _validate_input(self, input_data: Any, 
                       context: Optional[FraudRoutingContext]) -> Dict[str, ValidationResult]:
        """Validate all inputs"""
        validation_results = {}
        
        # Validate transaction data if present
        if context and context.transaction_amount is not None:
            transaction_data = {
                'amount': context.transaction_amount,
                'type': context.transaction_type,
                'id': context.transaction_id
            }
            
            if self.config.get('validation', {}).get('enable_transaction_validation', True):
                tx_result = self.transaction_validator.validate(transaction_data)
                validation_results['transaction'] = tx_result
                
                if not tx_result.is_valid:
                    self.performance_monitor['validation_failures'] += 1
                    raise TransactionValidationException(
                        f"Transaction validation failed: {tx_result.errors}",
                        transaction_data=transaction_data
                    )
        
        # Validate confidence thresholds
        if hasattr(self, 'confidence_threshold'):
            if not 0.0 <= self.config.get('confidence_threshold', 0.8) <= 1.0:
                raise ConfidenceValidationException(
                    "Invalid confidence threshold",
                    confidence_score=self.config.get('confidence_threshold')
                )
        
        return validation_results
    
    def _perform_routing(self, input_data: Any, 
                        context: Optional[FraudRoutingContext],
                        strategy: Optional[RoutingStrategy],
                        validation_results: Dict[str, ValidationResult]) -> FraudRoutingResult:
        """Perform actual routing with error recovery"""
        try:
            # Perform base routing
            base_result = self.base_router.route_request(
                input_data,
                strategy=strategy,
                input_type=self._determine_input_type(input_data, context)
            )
            
            # Enhance with fraud-specific routing
            fraud_result = self._enhance_fraud_routing(base_result, input_data, context)
            fraud_result.validation_result = validation_results.get('transaction')
            
            # Apply confidence scoring
            fraud_result = self._apply_confidence_scoring(fraud_result, context)
            
            # Apply risk assessment
            fraud_result = self._apply_risk_assessment(fraud_result, context)
            
            # Additional fraud processing
            fraud_result.fraud_category = self._determine_fraud_category(fraud_result)
            fraud_result.recommended_actions = self._generate_recommendations(fraud_result)
            
            if context and context.compliance_requirements:
                fraud_result.compliance_flags = self._check_compliance(fraud_result, context)
            
            fraud_result.investigation_priority = self._determine_priority(fraud_result)
            
            return fraud_result
            
        except Exception as e:
            logger.error(f"Routing execution failed: {e}")
            
            # Attempt error recovery
            if self.config.get('enable_fallback', True):
                logger.info("Attempting error recovery with fallback routing")
                return self._error_recovery_routing(input_data, context, str(e))
            else:
                raise
    
    def _validate_routing_result(self, result: FraudRoutingResult) -> FraudRoutingResult:
        """Validate routing result"""
        try:
            # Validate confidence score
            if not 0.0 <= result.confidence_score <= 1.0:
                logger.warning(f"Invalid confidence score: {result.confidence_score}, clamping to valid range")
                result.confidence_score = max(0.0, min(1.0, result.confidence_score))
            
            # Validate risk score
            if not 0.0 <= result.risk_score <= 1.0:
                logger.warning(f"Invalid risk score: {result.risk_score}, clamping to valid range")
                result.risk_score = max(0.0, min(1.0, result.risk_score))
            
            # Validate target domain
            available_domains = self.base_router.get_available_domains()
            if result.target_domain not in available_domains:
                logger.warning(f"Target domain {result.target_domain} not in available domains")
                result.target_domain = self.config.get('fallback_domain', 'fraud_investigation')
                result.fallback_reason = "Invalid target domain"
            
            # Validate priority
            valid_priorities = ["critical", "high", "medium", "normal", "low"]
            if result.investigation_priority not in valid_priorities:
                result.investigation_priority = "normal"
            
            return result
            
        except Exception as e:
            logger.error(f"Result validation failed: {e}")
            return result
    
    def _monitor_performance(self, routing_time_ms: float):
        """Monitor routing performance"""
        try:
            self.performance_monitor['total_routing_time'] += routing_time_ms
            
            if routing_time_ms > self.validation_config.max_routing_time_ms:
                self.performance_monitor['slow_requests'] += 1
                logger.warning(f"Slow routing detected: {routing_time_ms:.2f}ms")
            
            # Check if performance monitoring interval reached
            if self.performance_monitor['request_count'] % self.validation_config.performance_check_interval == 0:
                self._check_performance_health()
                
        except Exception as e:
            logger.error(f"Performance monitoring failed: {e}")
    
    def _check_performance_health(self):
        """Check overall performance health"""
        try:
            avg_routing_time = (
                self.performance_monitor['total_routing_time'] / 
                self.performance_monitor['request_count']
            )
            
            slow_request_rate = (
                self.performance_monitor['slow_requests'] / 
                self.performance_monitor['request_count']
            )
            
            if avg_routing_time > self.validation_config.max_routing_time_ms * 0.8:
                logger.warning(f"Average routing time {avg_routing_time:.2f}ms approaching limit")
            
            if slow_request_rate > 0.1:  # More than 10% slow requests
                logger.warning(f"High slow request rate: {slow_request_rate:.2%}")
                
        except Exception as e:
            logger.error(f"Performance health check failed: {e}")
    
    def _error_recovery_routing(self, input_data: Any,
                               context: Optional[FraudRoutingContext],
                               error_message: str) -> FraudRoutingResult:
        """Attempt error recovery with degraded routing"""
        try:
            logger.info("Attempting error recovery routing")
            
            # Use simplified pattern matching
            simple_result = self._simple_pattern_routing(input_data, context)
            simple_result.error_recovery_applied = True
            simple_result.fallback_reason = f"Error recovery: {error_message}"
            
            return simple_result
            
        except Exception as e:
            logger.error(f"Error recovery failed: {e}")
            # Last resort - return minimal valid result
            return self._minimal_fallback_result(input_data, context, error_message)
    
    def _simple_pattern_routing(self, input_data: Any,
                               context: Optional[FraudRoutingContext]) -> FraudRoutingResult:
        """Simple pattern-based routing for error recovery"""
        input_str = str(input_data).lower()
        
        # Simple keyword matching
        if any(word in input_str for word in ["fraud", "suspicious", "alert"]):
            target = "fraud_detection"
            confidence = 0.6
        elif any(word in input_str for word in ["transaction", "payment", "transfer"]):
            target = "transaction_analysis"
            confidence = 0.5
        elif any(word in input_str for word in ["risk", "score", "assess"]):
            target = "risk_assessment"
            confidence = 0.5
        else:
            target = "fraud_investigation"
            confidence = 0.3
        
        return FraudRoutingResult(
            target_domain=target,
            confidence_score=confidence,
            confidence_level=self._get_confidence_level(confidence),
            reasoning="Simple pattern matching (error recovery mode)",
            fraud_category=self._map_domain_to_category(target),
            risk_score=0.5,  # Default medium risk
            investigation_priority="high"  # High priority for error cases
        )
    
    def _minimal_fallback_result(self, input_data: Any,
                                context: Optional[FraudRoutingContext],
                                error_message: str) -> FraudRoutingResult:
        """Minimal fallback result when all else fails"""
        return FraudRoutingResult(
            target_domain=self.config.get('fallback_domain', 'fraud_investigation'),
            confidence_score=0.1,
            confidence_level=RoutingConfidence.VERY_LOW,
            reasoning=f"Minimal fallback due to: {error_message}",
            fraud_category=FraudRoutingCategory.FRAUD_INVESTIGATION,
            risk_score=0.7,  # Higher risk for fallback
            investigation_priority="critical",
            error_recovery_applied=True,
            fallback_reason=error_message,
            recommended_actions=["Manual review required", "Check system logs"]
        )
    
    def _determine_input_type(self, input_data: Any,
                             context: Optional[FraudRoutingContext]) -> str:
        """Determine input type with validation"""
        if context:
            if context.transaction_type:
                return "transaction"
            elif context.compliance_requirements:
                return "compliance"
            elif context.fraud_indicators:
                return "fraud_check"
        
        # Analyze input data
        input_str = str(input_data).lower()
        
        # Map to allowed input types
        if any(word in input_str for word in ["transaction", "payment", "transfer"]):
            input_type = "transaction"
        elif any(word in input_str for word in ["fraud", "suspicious", "alert"]):
            input_type = "fraud_check"
        elif any(word in input_str for word in ["risk", "score", "assess"]):
            input_type = "risk_assessment"
        elif any(word in input_str for word in ["compliance", "aml", "kyc"]):
            input_type = "compliance"
        else:
            input_type = "text"
        
        # Validate against allowed types
        if input_type not in self.validation_config.allowed_input_types:
            logger.warning(f"Input type {input_type} not in allowed types, using 'text'")
            input_type = "text"
        
        return input_type
    
    def _enhance_fraud_routing(self, base_result: RoutingResult,
                              input_data: Any,
                              context: Optional[FraudRoutingContext]) -> FraudRoutingResult:
        """Enhance routing with fraud-specific information"""
        fraud_result = FraudRoutingResult(
            target_domain=base_result.target_domain,
            confidence_score=base_result.confidence_score,
            confidence_level=base_result.confidence_level,
            reasoning=base_result.reasoning,
            alternative_domains=base_result.alternative_domains,
            routing_strategy=base_result.routing_strategy,
            routing_time_ms=base_result.routing_time_ms,
            metadata=base_result.metadata,
            routing_context=context
        )
        
        if context:
            fraud_result.fraud_indicators = self._detect_fraud_indicators(input_data, context)
        
        return fraud_result
    
    def _detect_fraud_indicators(self, input_data: Any,
                                context: FraudRoutingContext) -> List[FraudIndicatorType]:
        """Detect fraud indicators with error handling"""
        indicators = []
        
        try:
            # Transaction amount check
            if context.transaction_amount is not None:
                if context.transaction_amount > self.config.get('high_amount_threshold', 10000):
                    indicators.append(FraudIndicatorType.HIGH_AMOUNT)
            
            # Velocity check
            if context.historical_data:
                velocity = context.historical_data.get('transaction_count_24h', 0)
                if velocity > self.config.get('velocity_threshold', 5):
                    indicators.append(FraudIndicatorType.VELOCITY)
            
            # Geographic anomaly
            if context.geographic_data:
                if context.geographic_data.get('unusual_location', False):
                    indicators.append(FraudIndicatorType.GEOGRAPHIC_ANOMALY)
            
            # Behavioral anomaly
            if context.metadata.get('behavioral_score', 1.0) < 0.5:
                indicators.append(FraudIndicatorType.BEHAVIORAL_ANOMALY)
            
            # Pattern matching
            input_str = str(input_data).lower()
            if any(word in input_str for word in ['fraud', 'suspicious', 'unusual']):
                indicators.append(FraudIndicatorType.PATTERN_MATCH)
                
        except Exception as e:
            logger.error(f"Error detecting fraud indicators: {e}")
            
        return indicators
    
    def _apply_confidence_scoring(self, result: FraudRoutingResult,
                                 context: Optional[FraudRoutingContext]) -> FraudRoutingResult:
        """Apply confidence scoring with validation"""
        try:
            base_confidence = result.confidence_score
            
            # Validate base confidence
            if not 0.0 <= base_confidence <= 1.0:
                raise ConfidenceValidationException(
                    f"Invalid base confidence score: {base_confidence}"
                )
            
            # Apply adjustments
            if result.fraud_indicators:
                boost = len(result.fraud_indicators) * 0.05
                base_confidence = min(1.0, base_confidence + boost)
            
            if context and context.user_risk_profile == "high":
                if result.target_domain not in ["fraud_detection", "fraud_investigation"]:
                    base_confidence *= 0.7
            
            # Final validation
            result.confidence_score = round(
                max(0.0, min(1.0, base_confidence)), 
                self.validation_config.confidence_precision
            )
            result.confidence_level = self._get_confidence_level(result.confidence_score)
            
        except Exception as e:
            logger.error(f"Confidence scoring failed: {e}")
            result.confidence_score = 0.5  # Default medium confidence
            result.confidence_level = RoutingConfidence.MEDIUM
            
        return result
    
    def _apply_risk_assessment(self, result: FraudRoutingResult,
                              context: Optional[FraudRoutingContext]) -> FraudRoutingResult:
        """Apply risk assessment with validation"""
        try:
            risk_score = 0.0
            
            # Calculate risk from indicators
            indicator_weights = {
                FraudIndicatorType.HIGH_AMOUNT: 0.3,
                FraudIndicatorType.VELOCITY: 0.25,
                FraudIndicatorType.GEOGRAPHIC_ANOMALY: 0.2,
                FraudIndicatorType.BEHAVIORAL_ANOMALY: 0.25,
                FraudIndicatorType.BLACKLIST_MATCH: 0.5,
                FraudIndicatorType.PATTERN_MATCH: 0.15,
                FraudIndicatorType.ML_PREDICTION: 0.35,
                FraudIndicatorType.RULE_VIOLATION: 0.3,
                FraudIndicatorType.COMPLIANCE_FLAG: 0.25
            }
            
            for indicator in result.fraud_indicators:
                risk_score += indicator_weights.get(indicator, 0.1)
            
            # Apply context modifiers
            if context:
                if context.user_risk_profile == "high":
                    risk_score *= 1.3
                elif context.user_risk_profile == "low":
                    risk_score *= 0.7
            
            # Validate and set risk score
            result.risk_score = round(max(0.0, min(1.0, risk_score)), 3)
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            result.risk_score = 0.5  # Default medium risk
            
        return result
    
    def _determine_fraud_category(self, result: FraudRoutingResult) -> FraudRoutingCategory:
        """Determine fraud category with validation"""
        return self._map_domain_to_category(result.target_domain)
    
    def _map_domain_to_category(self, domain: str) -> FraudRoutingCategory:
        """Map domain to fraud category"""
        domain_to_category = {
            "transaction_analysis": FraudRoutingCategory.TRANSACTION_ANALYSIS,
            "fraud_detection": FraudRoutingCategory.PATTERN_DETECTION,
            "risk_assessment": FraudRoutingCategory.RISK_ASSESSMENT,
            "compliance_check": FraudRoutingCategory.COMPLIANCE_CHECK,
            "alert_generation": FraudRoutingCategory.ALERT_GENERATION,
            "fraud_investigation": FraudRoutingCategory.FRAUD_INVESTIGATION
        }
        
        return domain_to_category.get(domain, FraudRoutingCategory.MONITORING)
    
    def _generate_recommendations(self, result: FraudRoutingResult) -> List[str]:
        """Generate recommendations with error handling"""
        recommendations = []
        
        try:
            # Risk-based recommendations
            if result.risk_score > self.config.get('risk_score_threshold', 0.7):
                recommendations.extend([
                    "Block transaction pending review",
                    "Initiate fraud investigation",
                    "Send high-priority alert"
                ])
            elif result.risk_score > 0.5:
                recommendations.extend([
                    "Flag for manual review",
                    "Request additional verification",
                    "Monitor subsequent activity"
                ])
            
            # Indicator-specific recommendations
            if FraudIndicatorType.HIGH_AMOUNT in result.fraud_indicators:
                recommendations.append("Verify transaction amount with customer")
            
            if FraudIndicatorType.VELOCITY in result.fraud_indicators:
                recommendations.append("Check for account compromise")
            
            if FraudIndicatorType.GEOGRAPHIC_ANOMALY in result.fraud_indicators:
                recommendations.append("Verify customer location")
            
            # Validation failure recommendations
            if result.validation_result and not result.validation_result.is_valid:
                recommendations.append("Address validation errors before processing")
            
            # Error recovery recommendations
            if result.error_recovery_applied:
                recommendations.append("Review error logs for root cause")
                recommendations.append("Consider manual intervention")
                
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            recommendations.append("Review transaction manually")
            
        return recommendations
    
    def _check_compliance(self, result: FraudRoutingResult,
                         context: FraudRoutingContext) -> List[str]:
        """Check compliance with error handling"""
        compliance_flags = []
        
        try:
            for requirement in context.compliance_requirements:
                requirement_lower = requirement.lower()
                
                if "aml" in requirement_lower:
                    if result.risk_score > 0.6:
                        compliance_flags.append("AML_HIGH_RISK")
                    if FraudIndicatorType.BLACKLIST_MATCH in result.fraud_indicators:
                        compliance_flags.append("AML_BLACKLIST")
                
                if "kyc" in requirement_lower:
                    if context.metadata.get('kyc_complete', True) is False:
                        compliance_flags.append("KYC_INCOMPLETE")
                
                if "pci" in requirement_lower:
                    if context.transaction_type == "card":
                        compliance_flags.append("PCI_CHECK_REQUIRED")
                
                if "sanctions" in requirement_lower:
                    compliance_flags.append("SANCTIONS_SCREENING_REQUIRED")
                    
        except Exception as e:
            logger.error(f"Compliance check failed: {e}")
            compliance_flags.append("COMPLIANCE_CHECK_ERROR")
            
        return compliance_flags
    
    def _determine_priority(self, result: FraudRoutingResult) -> str:
        """Determine investigation priority"""
        try:
            if result.error_recovery_applied:
                return "critical"
            elif result.risk_score > 0.9:
                return "critical"
            elif result.risk_score > 0.7:
                return "high"
            elif result.risk_score > 0.5:
                return "medium"
            elif len(result.fraud_indicators) > 2:
                return "medium"
            else:
                return "normal"
        except Exception as e:
            logger.error(f"Priority determination failed: {e}")
            return "high"  # Default to high for safety
    
    def _get_confidence_level(self, score: float) -> RoutingConfidence:
        """Get confidence level from score"""
        if score >= 0.9:
            return RoutingConfidence.VERY_HIGH
        elif score >= 0.7:
            return RoutingConfidence.HIGH
        elif score >= 0.5:
            return RoutingConfidence.MEDIUM
        elif score >= 0.3:
            return RoutingConfidence.LOW
        else:
            return RoutingConfidence.VERY_LOW
    
    def _fallback_routing(self, input_data: Any,
                         context: Optional[FraudRoutingContext],
                         error_message: str) -> FraudRoutingResult:
        """Enhanced fallback routing"""
        logger.warning(f"Using fallback routing: {error_message}")
        
        try:
            # Try error recovery first
            return self._error_recovery_routing(input_data, context, error_message)
        except Exception as e:
            logger.error(f"Fallback routing failed: {e}")
            # Return minimal result
            return self._minimal_fallback_result(input_data, context, 
                                               f"Fallback failed: {error_message}")
    
    def perform_health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        health_status = {
            'status': self.health_status.value,
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        try:
            # Base router health
            health_status['checks']['base_router'] = {
                'status': 'healthy' if self.base_router else 'unhealthy',
                'available_domains': len(self.base_router.get_available_domains()) if self.base_router else 0
            }
            
            # Circuit breaker health
            cb_state = self.circuit_breaker.get_state()
            health_status['checks']['circuit_breaker'] = {
                'status': cb_state['state'],
                'failure_rate': cb_state['failure_rate']
            }
            
            # Performance health
            if self.performance_monitor['request_count'] > 0:
                avg_time = (self.performance_monitor['total_routing_time'] / 
                           self.performance_monitor['request_count'])
                slow_rate = (self.performance_monitor['slow_requests'] / 
                            self.performance_monitor['request_count'])
                
                health_status['checks']['performance'] = {
                    'status': 'healthy' if avg_time < self.validation_config.max_routing_time_ms else 'degraded',
                    'average_routing_time_ms': avg_time,
                    'slow_request_rate': slow_rate
                }
            
            # Validation health
            if self.performance_monitor['request_count'] > 0:
                validation_failure_rate = (self.performance_monitor['validation_failures'] / 
                                         self.performance_monitor['request_count'])
                
                health_status['checks']['validation'] = {
                    'status': 'healthy' if validation_failure_rate < 0.1 else 'degraded',
                    'failure_rate': validation_failure_rate
                }
            
            # Overall health determination
            all_healthy = all(
                check.get('status') in ['healthy', RoutingHealthStatus.HEALTHY.value]
                for check in health_status['checks'].values()
            )
            
            if all_healthy:
                health_status['status'] = RoutingHealthStatus.HEALTHY.value
            else:
                health_status['status'] = RoutingHealthStatus.DEGRADED.value
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status['status'] = RoutingHealthStatus.UNHEALTHY.value
            health_status['error'] = str(e)
            
        return health_status
    
    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get comprehensive routing metrics"""
        try:
            base_metrics = self.base_router.get_routing_metrics() if self.base_router else {}
            
            fraud_metrics = {
                'performance': {
                    'total_requests': self.performance_monitor['request_count'],
                    'average_routing_time_ms': (
                        self.performance_monitor['total_routing_time'] / 
                        max(self.performance_monitor['request_count'], 1)
                    ),
                    'slow_requests': self.performance_monitor['slow_requests'],
                    'validation_failures': self.performance_monitor['validation_failures'],
                    'routing_failures': self.performance_monitor['routing_failures']
                },
                'health': {
                    'status': self.health_status.value,
                    'consecutive_failures': self.consecutive_failures,
                    'circuit_breaker_state': self.circuit_breaker.get_state()
                }
            }
            
            return {**base_metrics, 'fraud_routing': fraud_metrics}
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {}
    
    def validate_routing_rule(self, rule: Dict[str, Any]) -> ValidationResult:
        """Validate a routing rule"""
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Check required fields
            required_fields = ['name', 'condition', 'action']
            for field in required_fields:
                if field not in rule:
                    errors.append(f"Missing required field: {field}")
            
            # Validate condition
            if 'condition' in rule:
                condition = rule['condition']
                if not isinstance(condition, (dict, str)):
                    errors.append("Condition must be dict or string")
                
                # Check for dangerous patterns in string conditions
                if isinstance(condition, str):
                    if any(danger in condition.lower() for danger in ['exec', 'eval', '__']):
                        errors.append("Potentially dangerous condition detected")
            
            # Validate action
            if 'action' in rule:
                action = rule['action']
                if not isinstance(action, dict):
                    errors.append("Action must be a dictionary")
                elif 'target_domain' not in action:
                    errors.append("Action missing target_domain")
            
            # Validate priority
            if 'priority' in rule:
                priority = rule['priority']
                if not isinstance(priority, (int, float)) or not 0 <= priority <= 100:
                    errors.append("Priority must be between 0 and 100")
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                validation_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            logger.error(f"Rule validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                validation_time_ms=(time.time() - start_time) * 1000
            )
    
    def shutdown(self):
        """Gracefully shutdown the router"""
        try:
            logger.info("Shutting down FinancialFraudRouter")
            
            # Shutdown executor
            self.executor.shutdown(wait=True, timeout=5)
            
            # Save final metrics
            final_metrics = self.get_routing_metrics()
            logger.info(f"Final metrics: {json.dumps(final_metrics, indent=2)}")
            
            logger.info("FinancialFraudRouter shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# ============= TESTING UTILITIES =============

def test_enhanced_fraud_routing():
    """Test enhanced fraud routing functionality"""
    import json
    
    # Configure logging for tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n=== Testing Enhanced Financial Fraud Router ===\n")
    
    try:
        # Create router instance
        router = FinancialFraudRouter()
        
        # Test 1: Valid transaction routing
        print("Test 1: Valid Transaction Routing")
        print("-" * 50)
        context1 = FraudRoutingContext(
            transaction_amount=5000.0,
            transaction_type="purchase",
            transaction_id="TXN12345678",
            user_risk_profile="medium"
        )
        result1 = router.route_fraud_request(
            "Analyze transaction TXN12345678 for $5000",
            context=context1
        )
        print(f"Target Domain: {result1.target_domain}")
        print(f"Confidence: {result1.confidence_score:.3f}")
        print(f"Risk Score: {result1.risk_score:.3f}")
        print(f"Validation: {'✓' if result1.validation_result and result1.validation_result.is_valid else '✗'}")
        print()
        
        # Test 2: Invalid transaction (exceeds limit)
        print("Test 2: Invalid Transaction (Exceeds Limit)")
        print("-" * 50)
        context2 = FraudRoutingContext(
            transaction_amount=15_000_000.0,  # Exceeds max
            transaction_type="transfer"
        )
        try:
            result2 = router.route_fraud_request(
                "Process large transfer",
                context=context2
            )
            print("ERROR: Should have raised validation exception")
        except TransactionValidationException as e:
            print(f"✓ Correctly caught validation error: {e}")
            print(f"  Error code: {e.error_code}")
            print(f"  Details: {e.details}")
        print()
        
        # Test 3: Security validation (SQL injection attempt)
        print("Test 3: Security Validation")
        print("-" * 50)
        try:
            malicious_input = "'; DROP TABLE transactions; --"
            result3 = router.route_fraud_request(malicious_input)
            print("ERROR: Should have caught SQL injection")
        except RoutingSecurityException as e:
            print(f"✓ Correctly caught security threat: {e}")
            print(f"  Error code: {e.error_code}")
        print()
        
        # Test 4: Performance testing with timeout
        print("Test 4: Performance Testing")
        print("-" * 50)
        import time
        
        # Simulate slow routing by adding delay
        def slow_route(*args, **kwargs):
            time.sleep(0.1)  # 100ms delay
            return router._perform_routing(*args, **kwargs)
        
        # Temporarily replace method
        original_method = router._perform_routing
        router._perform_routing = slow_route
        
        try:
            result4 = router.route_fraud_request(
                "Quick fraud check",
                timeout_ms=50  # 50ms timeout
            )
            print("ERROR: Should have timed out")
        except RoutingTimeoutException as e:
            print(f"✓ Correctly timed out: {e}")
            print(f"  Timeout: {e.details['timeout_ms']}ms")
        finally:
            router._perform_routing = original_method
        print()
        
        # Test 5: Circuit breaker test
        print("Test 5: Circuit Breaker")
        print("-" * 50)
        
        # Force multiple failures
        def failing_route(*args, **kwargs):
            raise Exception("Simulated failure")
        
        router._perform_routing = failing_route
        
        failures = 0
        for i in range(15):
            try:
                router.route_fraud_request(f"Request {i}")
            except:
                failures += 1
        
        print(f"Failures recorded: {failures}")
        cb_state = router.circuit_breaker.get_state()
        print(f"Circuit breaker state: {cb_state['state']}")
        print(f"Failure rate: {cb_state['failure_rate']:.2%}")
        
        # Restore original method
        router._perform_routing = original_method
        print()
        
        # Test 6: Error recovery
        print("Test 6: Error Recovery")
        print("-" * 50)
        
        # Force an error that triggers recovery
        def error_route(*args, **kwargs):
            if hasattr(error_route, 'call_count'):
                error_route.call_count += 1
            else:
                error_route.call_count = 1
            
            if error_route.call_count == 1:
                raise Exception("First attempt fails")
            return original_method(*args, **kwargs)
        
        router._perform_routing = error_route
        
        result6 = router.route_fraud_request("Fraud detection with recovery")
        print(f"Target Domain: {result6.target_domain}")
        print(f"Error Recovery Applied: {result6.error_recovery_applied}")
        print(f"Fallback Reason: {result6.fallback_reason}")
        
        router._perform_routing = original_method
        print()
        
        # Test 7: Rule validation
        print("Test 7: Routing Rule Validation")
        print("-" * 50)
        
        # Valid rule
        valid_rule = {
            'name': 'high_amount_rule',
            'condition': {'amount': {'$gt': 10000}},
            'action': {'target_domain': 'fraud_detection'},
            'priority': 90
        }
        
        valid_result = router.validate_routing_rule(valid_rule)
        print(f"Valid rule validation: {'✓' if valid_result.is_valid else '✗'}")
        
        # Invalid rule
        invalid_rule = {
            'name': 'bad_rule',
            'condition': 'exec("malicious_code")',
            'action': {}  # Missing target_domain
        }
        
        invalid_result = router.validate_routing_rule(invalid_rule)
        print(f"Invalid rule validation: {'✓' if not invalid_result.is_valid else '✗'}")
        print(f"Errors detected: {invalid_result.errors}")
        print()
        
        # Test 8: Health check
        print("Test 8: Health Check")
        print("-" * 50)
        health = router.perform_health_check()
        print(json.dumps(health, indent=2))
        print()
        
        # Test 9: Metrics
        print("Test 9: Routing Metrics")
        print("-" * 50)
        metrics = router.get_routing_metrics()
        print(json.dumps(metrics.get('fraud_routing', {}), indent=2))
        print()
        
        # Test 10: Pattern validation
        print("Test 10: Pattern Validation")
        print("-" * 50)
        
        # Valid pattern
        valid_pattern = DomainPattern(
            domain_name="test_domain",
            keywords=["test", "validation"],
            regex_patterns=[r'\d{4}-\d{2}-\d{2}'],  # Date pattern
            priority=5,
            confidence_boost=0.1
        )
        
        pattern_result = router.pattern_validator.validate(valid_pattern)
        print(f"Valid pattern: {'✓' if pattern_result.is_valid else '✗'}")
        
        # Invalid pattern
        invalid_pattern = DomainPattern(
            domain_name="invalid domain!",  # Invalid characters
            keywords=["x" * 200],  # Too long
            regex_patterns=[r'['],  # Invalid regex
            priority=15,  # Out of range
            confidence_boost=2.0  # Out of range
        )
        
        invalid_pattern_result = router.pattern_validator.validate(invalid_pattern)
        print(f"Invalid pattern: {'✓' if not invalid_pattern_result.is_valid else '✗'}")
        print(f"Pattern errors: {invalid_pattern_result.errors}")
        print()
        
        # Cleanup
        router.shutdown()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


# ============= ENHANCED HELPER FUNCTIONS =============

def create_enhanced_fraud_router(validation_level: str = "standard", 
                                config: Optional[Dict[str, Any]] = None,
                                use_enhanced: bool = True) -> FinancialFraudRouter:
    """
    Create enhanced fraud router with specified validation level
    
    Args:
        validation_level: Validation level ('basic', 'standard', 'strict', 'paranoid')
        config: Optional configuration dictionary
        use_enhanced: Whether to use enhanced features if available
    
    Returns:
        FinancialFraudRouter instance with enhanced capabilities
    """
    
    # Map string to enum
    level_mapping = {
        'basic': ValidationLevel.BASIC if ENHANCED_AVAILABLE else None,
        'standard': ValidationLevel.STANDARD if ENHANCED_AVAILABLE else None,
        'strict': ValidationLevel.STRICT if ENHANCED_AVAILABLE else None,
        'paranoid': ValidationLevel.PARANOID if ENHANCED_AVAILABLE else None
    }
    
    validation_enum = level_mapping.get(validation_level.lower())
    
    if use_enhanced and not ENHANCED_AVAILABLE:
        logger.warning("Enhanced features requested but not available, falling back to standard validation")
    
    return FinancialFraudRouter(
        config=config,
        use_enhanced=use_enhanced,
        validation_level=validation_enum
    )


def validate_and_route_fraud_request(input_data: Any,
                                   context: Optional[Dict[str, Any]] = None,
                                   validation_level: str = "standard",
                                   timeout_ms: Optional[float] = None) -> Dict[str, Any]:
    """
    Validate and route fraud detection request with enhanced capabilities
    
    Args:
        input_data: Input data to route
        context: Optional context information
        validation_level: Validation level to use
        timeout_ms: Optional timeout in milliseconds
    
    Returns:
        Dictionary containing routing result and validation information
    """
    
    # Create enhanced router
    router = create_enhanced_fraud_router(validation_level=validation_level)
    
    # Convert context to enhanced context if available
    routing_context = None
    if context and ENHANCED_AVAILABLE:
        routing_context = EnhancedFraudRoutingContext(**context)
    elif context:
        routing_context = FraudRoutingContext(**context)
    
    try:
        # Route the request
        result = router.route_fraud_request(
            input_data,
            context=routing_context,
            timeout_ms=timeout_ms
        )
        
        # Convert result to dictionary
        result_dict = {
            'success': True,
            'target_domain': result.target_domain,
            'confidence_score': result.confidence_score,
            'confidence_level': result.confidence_level.value if hasattr(result.confidence_level, 'value') else str(result.confidence_level),
            'reasoning': result.reasoning,
            'routing_time_ms': getattr(result, 'routing_time_ms', 0),
            'enhanced_features_used': router.use_enhanced
        }
        
        # Add enhanced fields if available
        if hasattr(result, 'fraud_category') and result.fraud_category:
            result_dict['fraud_category'] = result.fraud_category.value if hasattr(result.fraud_category, 'value') else str(result.fraud_category)
        
        if hasattr(result, 'risk_score'):
            result_dict['risk_score'] = result.risk_score
        
        if hasattr(result, 'fraud_indicators'):
            result_dict['fraud_indicators'] = [
                indicator.value if hasattr(indicator, 'value') else str(indicator)
                for indicator in result.fraud_indicators
            ]
        
        if hasattr(result, 'recommended_actions'):
            result_dict['recommended_actions'] = result.recommended_actions
        
        if hasattr(result, 'validation_result') and result.validation_result:
            result_dict['validation_result'] = result.validation_result.to_dict()
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Enhanced fraud routing failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'enhanced_features_used': router.use_enhanced
        }


if __name__ == "__main__":
    test_enhanced_fraud_routing()