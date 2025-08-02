"""
Enhanced Financial Fraud Detection Domain Routing
Comprehensive validation and error handling for production-ready domain routing
"""

import asyncio
import json
import logging
import os
import time
import threading
import pickle
import hashlib
import re
import socket
import ssl
import certifi
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Type
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError
import traceback
import psutil
import numpy as np
from urllib.parse import urlparse
import ipaddress

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== ENHANCED EXCEPTION HIERARCHY ====================

class EnhancedDomainRoutingError(Exception):
    """Base exception for enhanced domain routing failures"""
    def __init__(self, message: str, error_code: str = "ROUTING_ERROR", 
                 details: Dict[str, Any] = None, recoverable: bool = True):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now()
        self.recoverable = recoverable
        self.retry_count = 0
        self.stack_trace = traceback.format_exc()


class EnhancedRoutingValidationError(EnhancedDomainRoutingError):
    """Raised when enhanced routing validation fails"""
    def __init__(self, message: str, validation_errors: List[str] = None, 
                 field: str = None, **kwargs):
        super().__init__(message, error_code="ENHANCED_VALIDATION_ERROR", **kwargs)
        self.validation_errors = validation_errors or []
        self.field = field


class EnhancedRoutingConfigurationError(EnhancedDomainRoutingError):
    """Raised when enhanced routing configuration is invalid"""
    def __init__(self, message: str, config_field: str = None, 
                 expected_type: Type = None, actual_value: Any = None, **kwargs):
        super().__init__(message, error_code="ENHANCED_CONFIG_ERROR", **kwargs)
        self.config_field = config_field
        self.expected_type = expected_type
        self.actual_value = actual_value


class EnhancedRoutingSecurityError(EnhancedDomainRoutingError):
    """Raised when enhanced security validation fails"""
    def __init__(self, message: str, security_check: str = None, 
                 severity: str = "HIGH", threat_type: str = None, **kwargs):
        super().__init__(message, error_code="ENHANCED_SECURITY_ERROR", recoverable=False, **kwargs)
        self.security_check = security_check
        self.severity = severity
        self.threat_type = threat_type


class EnhancedRoutingPerformanceError(EnhancedDomainRoutingError):
    """Raised when enhanced performance thresholds are exceeded"""
    def __init__(self, message: str, performance_metrics: Dict[str, Any] = None, 
                 threshold_type: str = None, **kwargs):
        super().__init__(message, error_code="ENHANCED_PERFORMANCE_ERROR", **kwargs)
        self.performance_metrics = performance_metrics or {}
        self.threshold_type = threshold_type


class EnhancedRoutingResourceError(EnhancedDomainRoutingError):
    """Raised when enhanced resource requirements are not met"""
    def __init__(self, message: str, resource_type: str = None, 
                 required: Any = None, available: Any = None, **kwargs):
        super().__init__(message, error_code="ENHANCED_RESOURCE_ERROR", **kwargs)
        self.resource_type = resource_type
        self.required = required
        self.available = available


class EnhancedRoutingCircuitBreakerError(EnhancedDomainRoutingError):
    """Raised when enhanced circuit breaker is open"""
    def __init__(self, message: str, circuit_state: str = None, 
                 failure_rate: float = None, **kwargs):
        super().__init__(message, error_code="ENHANCED_CIRCUIT_BREAKER_ERROR", **kwargs)
        self.circuit_state = circuit_state
        self.failure_rate = failure_rate


class EnhancedRoutingTimeoutError(EnhancedDomainRoutingError):
    """Raised when enhanced routing operations timeout"""
    def __init__(self, message: str, timeout_ms: float = None, 
                 operation: str = None, **kwargs):
        super().__init__(message, error_code="ENHANCED_TIMEOUT_ERROR", **kwargs)
        self.timeout_ms = timeout_ms
        self.operation = operation


class EnhancedRoutingIntegrationError(EnhancedDomainRoutingError):
    """Raised when enhanced routing integration fails"""
    def __init__(self, message: str, integration_point: str = None, 
                 component: str = None, **kwargs):
        super().__init__(message, error_code="ENHANCED_INTEGRATION_ERROR", **kwargs)
        self.integration_point = integration_point
        self.component = component


# ==================== VALIDATION LEVELS AND MODES ====================

class ValidationLevel(Enum):
    """Validation levels for enhanced routing"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


class ValidationMode(Enum):
    """Validation modes for different environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class SecurityLevel(Enum):
    """Security validation levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PerformanceMode(Enum):
    """Performance optimization modes"""
    LATENCY_OPTIMIZED = "latency_optimized"
    THROUGHPUT_OPTIMIZED = "throughput_optimized"
    BALANCED = "balanced"
    RESOURCE_CONSERVATIVE = "resource_conservative"


# ==================== ENHANCED VALIDATION RESULTS ====================

@dataclass
class EnhancedValidationResult:
    """Enhanced result of validation operation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info_messages: List[str] = field(default_factory=list)
    validated_data: Optional[Dict[str, Any]] = None
    validation_time_ms: float = 0.0
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    security_score: float = 1.0
    performance_score: float = 1.0
    reliability_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'info_messages': self.info_messages,
            'validated_data': self.validated_data,
            'validation_time_ms': self.validation_time_ms,
            'validation_level': self.validation_level.value,
            'security_score': self.security_score,
            'performance_score': self.performance_score,
            'reliability_score': self.reliability_score,
            'metadata': self.metadata
        }
    
    def add_error(self, error: str, error_code: str = None):
        """Add an error with optional error code"""
        self.errors.append(error)
        if error_code:
            self.metadata.setdefault('error_codes', []).append(error_code)
        self.is_valid = False
    
    def add_warning(self, warning: str, warning_code: str = None):
        """Add a warning with optional warning code"""
        self.warnings.append(warning)
        if warning_code:
            self.metadata.setdefault('warning_codes', []).append(warning_code)
    
    def add_info(self, info: str):
        """Add an info message"""
        self.info_messages.append(info)
    
    def merge(self, other: 'EnhancedValidationResult') -> 'EnhancedValidationResult':
        """Merge with another validation result"""
        merged = EnhancedValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            info_messages=self.info_messages + other.info_messages,
            validation_time_ms=self.validation_time_ms + other.validation_time_ms,
            validation_level=max(self.validation_level, other.validation_level, key=lambda x: x.value),
            security_score=min(self.security_score, other.security_score),
            performance_score=min(self.performance_score, other.performance_score),
            reliability_score=min(self.reliability_score, other.reliability_score)
        )
        
        # Merge validated data
        if self.validated_data and other.validated_data:
            merged.validated_data = {**self.validated_data, **other.validated_data}
        elif self.validated_data:
            merged.validated_data = self.validated_data
        elif other.validated_data:
            merged.validated_data = other.validated_data
        
        # Merge metadata
        merged.metadata = {**self.metadata, **other.metadata}
        
        return merged


@dataclass
class EnhancedValidationContext:
    """Context for enhanced validation operations"""
    request_id: str
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    validation_mode: ValidationMode = ValidationMode.PRODUCTION
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    performance_mode: PerformanceMode = PerformanceMode.BALANCED
    timeout_ms: float = 5000.0
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    enable_monitoring: bool = True
    enable_recovery: bool = True
    custom_validators: List[str] = field(default_factory=list)
    validation_metadata: Dict[str, Any] = field(default_factory=dict)
    security_context: Dict[str, Any] = field(default_factory=dict)
    performance_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate request ID if not provided"""
        if not self.request_id:
            self.request_id = self._generate_request_id()
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        timestamp = datetime.now().isoformat()
        random_part = hashlib.md5(f"{timestamp}_{time.time()}".encode()).hexdigest()[:8]
        return f"enhanced_validation_{random_part}"


# ==================== ENHANCED VALIDATION CONFIGURATION ====================

@dataclass
class EnhancedRoutingValidationConfig:
    """Enhanced configuration for routing validation"""
    
    # Validation levels configuration
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    validation_mode: ValidationMode = ValidationMode.PRODUCTION
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    performance_mode: PerformanceMode = PerformanceMode.BALANCED
    
    # Transaction validation enhanced settings
    min_transaction_amount: float = 0.01
    max_transaction_amount: float = 10_000_000.0
    allowed_transaction_types: Set[str] = field(default_factory=lambda: {
        "purchase", "withdrawal", "transfer", "payment", "refund", 
        "deposit", "authorization", "capture", "void", "chargeback"
    })
    transaction_id_pattern: str = r'^[A-Z0-9]{8,32}$'
    transaction_amount_precision: int = 2
    transaction_currency_codes: Set[str] = field(default_factory=lambda: {
        "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY"
    })
    
    # Pattern validation enhanced settings
    max_pattern_length: int = 2000
    max_keyword_length: int = 200
    max_regex_complexity: int = 100
    pattern_cache_size: int = 1000
    enable_pattern_learning: bool = True
    pattern_confidence_threshold: float = 0.7
    
    # Confidence validation enhanced settings
    min_confidence_score: float = 0.0
    max_confidence_score: float = 1.0
    confidence_precision: int = 4
    confidence_boost_limit: float = 0.5
    confidence_penalty_limit: float = -0.3
    dynamic_confidence_adjustment: bool = True
    
    # Performance validation enhanced settings
    max_routing_time_ms: float = 2000.0
    max_validation_time_ms: float = 1000.0
    max_concurrent_requests: int = 1000
    performance_check_interval: int = 50
    performance_degradation_threshold: float = 0.8
    performance_alert_threshold: float = 0.9
    enable_performance_profiling: bool = True
    
    # Security validation enhanced settings
    max_request_size_bytes: int = 2_097_152  # 2MB
    allowed_input_types: Set[str] = field(default_factory=lambda: {
        "text", "transaction", "payment_data", "fraud_check", 
        "risk_assessment", "compliance", "investigation",
        "monitoring", "alert", "reporting"
    })
    enable_sql_injection_check: bool = True
    enable_xss_check: bool = True
    enable_path_traversal_check: bool = True
    enable_command_injection_check: bool = True
    enable_file_inclusion_check: bool = True
    enable_xxe_check: bool = True
    security_scan_depth: int = 3
    ip_whitelist: Set[str] = field(default_factory=set)
    ip_blacklist: Set[str] = field(default_factory=set)
    rate_limit_requests_per_minute: int = 1000
    
    # Reliability validation enhanced settings
    min_health_check_interval_seconds: int = 30
    max_consecutive_failures: int = 3
    circuit_breaker_threshold: float = 0.3
    circuit_breaker_timeout_seconds: int = 180
    circuit_breaker_recovery_threshold: float = 0.8
    enable_graceful_degradation: bool = True
    fallback_routing_enabled: bool = True
    retry_max_attempts: int = 3
    retry_backoff_multiplier: float = 2.0
    retry_max_delay_seconds: float = 60.0
    
    # Resource monitoring enhanced settings
    max_memory_usage_mb: int = 512
    max_cpu_usage_percent: float = 80.0
    max_disk_usage_percent: float = 90.0
    resource_check_interval_seconds: int = 60
    enable_resource_alerts: bool = True
    enable_auto_scaling: bool = False
    
    # Caching enhanced settings
    enable_validation_caching: bool = True
    cache_max_size: int = 10000
    cache_ttl_seconds: int = 300
    cache_cleanup_interval_seconds: int = 900
    enable_distributed_cache: bool = False
    cache_compression_enabled: bool = True
    
    # Monitoring and alerting enhanced settings
    enable_detailed_monitoring: bool = True
    enable_real_time_alerts: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["log", "metrics"])
    metrics_collection_interval_seconds: int = 60
    enable_performance_tracking: bool = True
    enable_error_tracking: bool = True
    enable_audit_logging: bool = True
    
    # Integration enhanced settings
    enable_webhook_notifications: bool = False
    webhook_endpoints: List[str] = field(default_factory=list)
    enable_external_validators: bool = False
    external_validator_timeout_ms: float = 5000.0
    enable_ml_validation: bool = False
    ml_model_confidence_threshold: float = 0.8
    
    # Development and testing settings
    enable_debug_mode: bool = False
    enable_verbose_logging: bool = False
    enable_test_mode: bool = False
    test_failure_rate: float = 0.0
    enable_chaos_engineering: bool = False
    
    def get_validation_settings(self, level: ValidationLevel) -> Dict[str, Any]:
        """Get validation settings for specific level"""
        base_settings = {
            'enable_basic_validation': True,
            'enable_type_checking': True,
            'enable_range_validation': True
        }
        
        if level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            base_settings.update({
                'enable_security_validation': True,
                'enable_performance_validation': True,
                'enable_format_validation': True
            })
        
        if level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            base_settings.update({
                'enable_deep_security_scan': True,
                'enable_comprehensive_testing': True,
                'enable_resource_validation': True,
                'enable_compliance_checking': True
            })
        
        if level == ValidationLevel.PARANOID:
            base_settings.update({
                'enable_ml_validation': True,
                'enable_anomaly_detection': True,
                'enable_advanced_threat_detection': True,
                'enable_behavioral_analysis': True,
                'enable_predictive_validation': True
            })
        
        return base_settings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {}
        for field_name, field_def in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            if isinstance(value, Enum):
                result[field_name] = value.value
            elif isinstance(value, set):
                result[field_name] = list(value)
            else:
                result[field_name] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedRoutingValidationConfig':
        """Create configuration from dictionary"""
        # Handle enum conversions
        if 'validation_level' in data:
            data['validation_level'] = ValidationLevel(data['validation_level'])
        if 'validation_mode' in data:
            data['validation_mode'] = ValidationMode(data['validation_mode'])
        if 'security_level' in data:
            data['security_level'] = SecurityLevel(data['security_level'])
        if 'performance_mode' in data:
            data['performance_mode'] = PerformanceMode(data['performance_mode'])
        
        # Handle set conversions
        set_fields = [
            'allowed_transaction_types', 'transaction_currency_codes',
            'allowed_input_types', 'ip_whitelist', 'ip_blacklist'
        ]
        for field in set_fields:
            if field in data and isinstance(data[field], list):
                data[field] = set(data[field])
        
        return cls(**data)


# ==================== ENHANCED FRAUD ROUTING ENUMS ====================

class EnhancedFraudCategory(Enum):
    """Enhanced categories for fraud detection routing"""
    TRANSACTION_ANALYSIS = "transaction_analysis"
    PATTERN_DETECTION = "pattern_detection"
    RISK_ASSESSMENT = "risk_assessment"
    COMPLIANCE_CHECK = "compliance_check"
    ALERT_GENERATION = "alert_generation"
    ANOMALY_DETECTION = "anomaly_detection"
    FRAUD_INVESTIGATION = "fraud_investigation"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    ML_PREDICTION = "ml_prediction"
    REAL_TIME_MONITORING = "real_time_monitoring"
    FORENSIC_ANALYSIS = "forensic_analysis"
    REPORTING = "reporting"
    AUDIT_TRAIL = "audit_trail"
    CUSTOMER_VERIFICATION = "customer_verification"
    MERCHANT_VALIDATION = "merchant_validation"


class EnhancedFraudIndicator(Enum):
    """Enhanced types of fraud indicators for routing"""
    HIGH_AMOUNT = "high_amount"
    VELOCITY = "velocity"
    GEOGRAPHIC_ANOMALY = "geographic_anomaly"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    TEMPORAL_ANOMALY = "temporal_anomaly"
    DEVICE_ANOMALY = "device_anomaly"
    BLACKLIST_MATCH = "blacklist_match"
    WHITELIST_VIOLATION = "whitelist_violation"
    PATTERN_MATCH = "pattern_match"
    ML_PREDICTION = "ml_prediction"
    RULE_VIOLATION = "rule_violation"
    COMPLIANCE_FLAG = "compliance_flag"
    REPUTATION_RISK = "reputation_risk"
    NETWORK_ANOMALY = "network_anomaly"
    PAYMENT_METHOD_RISK = "payment_method_risk"
    MERCHANT_RISK = "merchant_risk"
    CUSTOMER_RISK = "customer_risk"


class EnhancedRoutingHealthStatus(Enum):
    """Enhanced health status for routing system"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    CIRCUIT_OPEN = "circuit_open"
    MAINTENANCE = "maintenance"
    RECOVERING = "recovering"


class EnhancedRoutingStrategy(Enum):
    """Enhanced routing strategies"""
    KEYWORD_BASED = "keyword_based"
    PATTERN_BASED = "pattern_based"
    ML_BASED = "ml_based"
    RULE_BASED = "rule_based"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"
    RISK_WEIGHTED = "risk_weighted"
    PERFORMANCE_OPTIMIZED = "performance_optimized"


class EnhancedConfidenceLevel(Enum):
    """Enhanced confidence levels for routing decisions"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM_LOW = "medium_low"
    MEDIUM = "medium"
    MEDIUM_HIGH = "medium_high"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"


# ==================== ENHANCED CONTEXT AND RESULT CLASSES ====================

@dataclass
class EnhancedFraudRoutingContext:
    """Enhanced context for fraud-specific routing decisions"""
    
    # Core transaction data
    transaction_amount: Optional[float] = None
    transaction_type: Optional[str] = None
    transaction_id: Optional[str] = None
    transaction_currency: Optional[str] = None
    transaction_timestamp: Optional[datetime] = None
    
    # Enhanced merchant data
    merchant_id: Optional[str] = None
    merchant_category: Optional[str] = None
    merchant_risk_score: Optional[float] = None
    merchant_reputation: Optional[str] = None
    
    # Enhanced customer data
    customer_id: Optional[str] = None
    customer_risk_profile: Optional[str] = None
    customer_verification_level: Optional[str] = None
    customer_account_age_days: Optional[int] = None
    customer_transaction_history: Optional[Dict[str, Any]] = None
    
    # Enhanced geographic data
    geographic_data: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    country_code: Optional[str] = None
    region_code: Optional[str] = None
    city: Optional[str] = None
    coordinates: Optional[Tuple[float, float]] = None
    
    # Enhanced temporal data
    time_data: Optional[Dict[str, Any]] = None
    local_time: Optional[datetime] = None
    timezone: Optional[str] = None
    is_business_hours: Optional[bool] = None
    day_of_week: Optional[int] = None
    
    # Enhanced device and session data
    device_data: Optional[Dict[str, Any]] = None
    device_fingerprint: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    session_duration_minutes: Optional[int] = None
    
    # Enhanced historical and behavioral data
    historical_data: Optional[Dict[str, Any]] = None
    behavioral_patterns: Optional[Dict[str, Any]] = None
    velocity_metrics: Optional[Dict[str, Any]] = None
    spending_patterns: Optional[Dict[str, Any]] = None
    
    # Enhanced compliance and regulatory data
    compliance_requirements: List[str] = field(default_factory=list)
    regulatory_flags: List[str] = field(default_factory=list)
    sanctions_screening_result: Optional[str] = None
    aml_risk_score: Optional[float] = None
    kyc_status: Optional[str] = None
    
    # Enhanced fraud indicators and risk data
    fraud_indicators: List[EnhancedFraudIndicator] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    external_risk_scores: Dict[str, float] = field(default_factory=dict)
    ml_model_scores: Dict[str, float] = field(default_factory=dict)
    
    # Enhanced metadata and context
    metadata: Dict[str, Any] = field(default_factory=dict)
    security_context: Optional[Dict[str, Any]] = None
    business_context: Optional[Dict[str, Any]] = None
    technical_context: Optional[Dict[str, Any]] = None
    
    # Enhanced tracking and monitoring
    request_id: Optional[str] = None
    parent_request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    
    # Enhanced processing metadata
    processing_start_time: Optional[datetime] = None
    processing_timeout_ms: Optional[float] = None
    processing_priority: Optional[str] = None
    processing_mode: Optional[str] = None
    
    def __post_init__(self):
        """Initialize derived fields and validate context"""
        if not self.request_id:
            self.request_id = self._generate_request_id()
        
        if not self.processing_start_time:
            self.processing_start_time = datetime.now()
        
        if not self.correlation_id:
            self.correlation_id = self._generate_correlation_id()
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        timestamp = datetime.now().isoformat()
        data = f"{timestamp}_{self.transaction_id or 'unknown'}_{time.time()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def _generate_correlation_id(self) -> str:
        """Generate correlation ID for tracking"""
        timestamp = int(time.time() * 1000)
        random_part = hashlib.sha256(f"{timestamp}_{self.request_id}".encode()).hexdigest()[:12]
        return f"corr_{timestamp}_{random_part}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        result = {}
        for field_name, field_def in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            if isinstance(value, datetime):
                result[field_name] = value.isoformat() if value else None
            elif isinstance(value, Enum):
                result[field_name] = value.value if value else None
            elif isinstance(value, list) and value and isinstance(value[0], Enum):
                result[field_name] = [item.value for item in value]
            else:
                result[field_name] = value
        return result
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of risk indicators and scores"""
        return {
            'fraud_indicators': [indicator.value for indicator in self.fraud_indicators],
            'risk_factors': self.risk_factors,
            'external_risk_scores': self.external_risk_scores,
            'ml_model_scores': self.ml_model_scores,
            'merchant_risk_score': self.merchant_risk_score,
            'aml_risk_score': self.aml_risk_score,
            'customer_risk_profile': self.customer_risk_profile,
            'compliance_requirements': self.compliance_requirements,
            'regulatory_flags': self.regulatory_flags
        }
    
    def add_fraud_indicator(self, indicator: EnhancedFraudIndicator, 
                           confidence: float = 1.0, source: str = "system"):
        """Add fraud indicator with metadata"""
        if indicator not in self.fraud_indicators:
            self.fraud_indicators.append(indicator)
            
        # Store indicator metadata
        indicator_metadata = self.metadata.setdefault('fraud_indicators', {})
        indicator_metadata[indicator.value] = {
            'confidence': confidence,
            'source': source,
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_risk_score(self) -> float:
        """Calculate overall risk score from context"""
        risk_score = 0.0
        
        # Weight fraud indicators
        indicator_weights = {
            EnhancedFraudIndicator.HIGH_AMOUNT: 0.2,
            EnhancedFraudIndicator.VELOCITY: 0.25,
            EnhancedFraudIndicator.GEOGRAPHIC_ANOMALY: 0.15,
            EnhancedFraudIndicator.BEHAVIORAL_ANOMALY: 0.2,
            EnhancedFraudIndicator.BLACKLIST_MATCH: 0.4,
            EnhancedFraudIndicator.ML_PREDICTION: 0.3,
            EnhancedFraudIndicator.DEVICE_ANOMALY: 0.2,
            EnhancedFraudIndicator.TEMPORAL_ANOMALY: 0.15
        }
        
        for indicator in self.fraud_indicators:
            weight = indicator_weights.get(indicator, 0.1)
            confidence = self.metadata.get('fraud_indicators', {}).get(indicator.value, {}).get('confidence', 1.0)
            risk_score += weight * confidence
        
        # Add external risk scores
        for score in self.external_risk_scores.values():
            risk_score += score * 0.1
        
        # Add ML model scores
        for score in self.ml_model_scores.values():
            risk_score += score * 0.15
        
        return min(1.0, risk_score)


# ==================== ENHANCED VALIDATOR CLASSES ====================

class EnhancedTransactionValidator:
    """Enhanced validator for transaction data with comprehensive validation"""
    
    def __init__(self, config: EnhancedRoutingValidationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validation_cache = {}
        self._cache_lock = threading.RLock()
    
    def validate(self, transaction_data: Dict[str, Any], 
                context: Optional[EnhancedValidationContext] = None) -> EnhancedValidationResult:
        """Enhanced validation of transaction data"""
        start_time = time.time()
        validation_id = f"tx_val_{time.time()}"
        
        result = EnhancedValidationResult(
            is_valid=True,
            validation_level=context.validation_level if context else self.config.validation_level
        )
        
        try:
            # Check cache if enabled
            if self.config.enable_validation_caching and context and context.enable_caching:
                cache_key = self._generate_cache_key(transaction_data)
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    cached_result.metadata['from_cache'] = True
                    return cached_result
            
            # Enhanced transaction amount validation
            self._validate_transaction_amount(transaction_data, result)
            
            # Enhanced transaction type validation
            self._validate_transaction_type(transaction_data, result)
            
            # Enhanced transaction ID validation
            self._validate_transaction_id(transaction_data, result)
            
            # Enhanced currency validation
            self._validate_transaction_currency(transaction_data, result)
            
            # Enhanced timestamp validation
            self._validate_transaction_timestamp(transaction_data, result)
            
            # Enhanced merchant validation
            self._validate_merchant_data(transaction_data, result)
            
            # Enhanced customer validation
            self._validate_customer_data(transaction_data, result)
            
            # Enhanced compliance validation
            if result.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                self._validate_compliance_requirements(transaction_data, result)
            
            # Enhanced security validation
            if result.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                self._validate_transaction_security(transaction_data, result)
            
            # Enhanced business rules validation
            if result.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                self._validate_business_rules(transaction_data, result)
            
            # ML-based validation
            if result.validation_level == ValidationLevel.PARANOID and self.config.enable_ml_validation:
                self._validate_with_ml_models(transaction_data, result)
            
            # Calculate scores
            self._calculate_validation_scores(result)
            
            # Set validation metadata
            result.validation_time_ms = (time.time() - start_time) * 1000
            result.metadata.update({
                'validation_id': validation_id,
                'validation_timestamp': datetime.now().isoformat(),
                'validator_version': '2.0.0',
                'validation_fields_checked': len(transaction_data)
            })
            
            # Cache result if enabled
            if self.config.enable_validation_caching and context and context.enable_caching:
                self._cache_result(cache_key, result, context.cache_ttl_seconds)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Transaction validation failed: {e}")
            result.add_error(f"Validation error: {str(e)}", "VALIDATION_EXCEPTION")
            result.validation_time_ms = (time.time() - start_time) * 1000
            return result
    
    def _validate_transaction_amount(self, data: Dict[str, Any], result: EnhancedValidationResult):
        """Enhanced transaction amount validation"""
        amount = data.get('amount')
        
        if amount is None:
            result.add_error("Transaction amount is required", "MISSING_AMOUNT")
            return
        
        try:
            amount = float(amount)
            
            # Basic range validation
            if amount < self.config.min_transaction_amount:
                result.add_error(
                    f"Transaction amount {amount} is below minimum {self.config.min_transaction_amount}",
                    "AMOUNT_TOO_LOW"
                )
            elif amount > self.config.max_transaction_amount:
                result.add_error(
                    f"Transaction amount {amount} exceeds maximum {self.config.max_transaction_amount}",
                    "AMOUNT_TOO_HIGH"
                )
            
            # Precision validation
            if len(str(amount).split('.')[-1]) > self.config.transaction_amount_precision:
                result.add_warning(
                    f"Amount precision exceeds {self.config.transaction_amount_precision} decimal places",
                    "PRECISION_WARNING"
                )
            
            # Suspicious amount patterns
            if self._is_suspicious_amount(amount):
                result.add_warning("Suspicious amount pattern detected", "SUSPICIOUS_AMOUNT")
            
            if result.is_valid:
                result.validated_data = result.validated_data or {}
                result.validated_data['amount'] = amount
                
        except (TypeError, ValueError) as e:
            result.add_error(f"Invalid transaction amount format: {amount}", "INVALID_AMOUNT_FORMAT")
    
    def _validate_transaction_type(self, data: Dict[str, Any], result: EnhancedValidationResult):
        """Enhanced transaction type validation"""
        tx_type = data.get('type')
        
        if not tx_type:
            result.add_warning("Transaction type not specified", "MISSING_TYPE")
            return
        
        if not isinstance(tx_type, str):
            result.add_error("Transaction type must be a string", "INVALID_TYPE_FORMAT")
            return
        
        tx_type = tx_type.lower().strip()
        
        if tx_type not in {t.lower() for t in self.config.allowed_transaction_types}:
            result.add_error(f"Invalid transaction type: {tx_type}", "INVALID_TYPE")
            return
        
        # Enhanced type-specific validation
        if tx_type in ['withdrawal', 'transfer'] and data.get('amount', 0) > 50000:
            result.add_warning("High-value withdrawal/transfer requires additional verification", "HIGH_VALUE_WITHDRAWAL")
        
        if tx_type == 'chargeback':
            result.add_info("Chargeback transaction requires special handling")
        
        if result.is_valid:
            result.validated_data = result.validated_data or {}
            result.validated_data['type'] = tx_type
    
    def _validate_transaction_id(self, data: Dict[str, Any], result: EnhancedValidationResult):
        """Enhanced transaction ID validation"""
        tx_id = data.get('id') or data.get('transaction_id')
        
        if not tx_id:
            result.add_warning("Transaction ID not provided", "MISSING_ID")
            return
        
        tx_id_str = str(tx_id).strip()
        
        if not re.match(self.config.transaction_id_pattern, tx_id_str):
            result.add_error(f"Invalid transaction ID format: {tx_id_str}", "INVALID_ID_FORMAT")
            return
        
        # Check for duplicate or suspicious IDs
        if self._is_suspicious_transaction_id(tx_id_str):
            result.add_warning("Suspicious transaction ID pattern detected", "SUSPICIOUS_ID")
        
        if result.is_valid:
            result.validated_data = result.validated_data or {}
            result.validated_data['id'] = tx_id_str
    
    def _validate_transaction_currency(self, data: Dict[str, Any], result: EnhancedValidationResult):
        """Enhanced currency validation"""
        currency = data.get('currency')
        
        if not currency:
            result.add_warning("Currency not specified, assuming USD", "MISSING_CURRENCY")
            currency = "USD"
        
        currency = str(currency).upper().strip()
        
        if currency not in self.config.transaction_currency_codes:
            result.add_error(f"Unsupported currency: {currency}", "INVALID_CURRENCY")
            return
        
        # Currency-specific validations
        if currency in ['JPY', 'KRW'] and data.get('amount', 0) < 1:
            result.add_warning("Sub-unit amounts unusual for this currency", "CURRENCY_PRECISION_WARNING")
        
        if result.is_valid:
            result.validated_data = result.validated_data or {}
            result.validated_data['currency'] = currency
    
    def _validate_transaction_timestamp(self, data: Dict[str, Any], result: EnhancedValidationResult):
        """Enhanced timestamp validation"""
        timestamp = data.get('timestamp')
        
        if not timestamp:
            result.add_warning("Transaction timestamp not provided", "MISSING_TIMESTAMP")
            return
        
        try:
            if isinstance(timestamp, str):
                parsed_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif isinstance(timestamp, (int, float)):
                parsed_time = datetime.fromtimestamp(timestamp)
            elif isinstance(timestamp, datetime):
                parsed_time = timestamp
            else:
                raise ValueError("Unsupported timestamp format")
            
            # Validate timestamp is reasonable
            now = datetime.now()
            if parsed_time > now:
                result.add_error("Transaction timestamp is in the future", "FUTURE_TIMESTAMP")
            elif (now - parsed_time).days > 90:
                result.add_warning("Transaction timestamp is more than 90 days old", "OLD_TIMESTAMP")
            
            if result.is_valid:
                result.validated_data = result.validated_data or {}
                result.validated_data['timestamp'] = parsed_time.isoformat()
                
        except ValueError as e:
            result.add_error(f"Invalid timestamp format: {timestamp}", "INVALID_TIMESTAMP_FORMAT")
    
    def _validate_merchant_data(self, data: Dict[str, Any], result: EnhancedValidationResult):
        """Enhanced merchant data validation"""
        merchant = data.get('merchant')
        
        if not merchant:
            result.add_warning("Merchant data not provided", "MISSING_MERCHANT")
            return
        
        if not isinstance(merchant, dict):
            result.add_error("Merchant data must be an object", "INVALID_MERCHANT_FORMAT")
            return
        
        # Validate required merchant fields
        required_fields = ['id', 'name']
        for field in required_fields:
            if field not in merchant:
                result.add_error(f"Missing required merchant field: {field}", f"MISSING_MERCHANT_{field.upper()}")
        
        # Validate merchant ID format
        merchant_id = merchant.get('id')
        if merchant_id and not re.match(r'^[A-Z0-9]{6,20}$', str(merchant_id)):
            result.add_warning("Unusual merchant ID format", "UNUSUAL_MERCHANT_ID")
        
        # Validate merchant category
        merchant_category = merchant.get('category')
        if merchant_category and len(merchant_category) != 4:
            result.add_warning("Merchant category should be 4-digit MCC code", "INVALID_MCC")
        
        if result.is_valid and not result.errors:
            result.validated_data = result.validated_data or {}
            result.validated_data['merchant'] = merchant
    
    def _validate_customer_data(self, data: Dict[str, Any], result: EnhancedValidationResult):
        """Enhanced customer data validation"""
        customer = data.get('customer')
        
        if not customer:
            result.add_warning("Customer data not provided", "MISSING_CUSTOMER")
            return
        
        if not isinstance(customer, dict):
            result.add_error("Customer data must be an object", "INVALID_CUSTOMER_FORMAT")
            return
        
        # Validate customer ID
        customer_id = customer.get('id')
        if customer_id and not re.match(r'^[A-Z0-9]{8,32}$', str(customer_id)):
            result.add_warning("Unusual customer ID format", "UNUSUAL_CUSTOMER_ID")
        
        # Validate customer verification level
        verification_level = customer.get('verification_level')
        if verification_level and verification_level not in ['basic', 'enhanced', 'premium']:
            result.add_warning(f"Unknown verification level: {verification_level}", "UNKNOWN_VERIFICATION_LEVEL")
        
        if result.is_valid and not result.errors:
            result.validated_data = result.validated_data or {}
            result.validated_data['customer'] = customer
    
    def _validate_compliance_requirements(self, data: Dict[str, Any], result: EnhancedValidationResult):
        """Enhanced compliance requirements validation"""
        # AML compliance checks
        amount = data.get('amount', 0)
        if amount > 10000:  # Large transaction threshold
            result.add_info("Transaction exceeds AML reporting threshold")
            result.metadata.setdefault('compliance_flags', []).append('AML_LARGE_TRANSACTION')
        
        # PCI compliance for card transactions
        if data.get('payment_method') == 'card':
            result.add_info("Card transaction requires PCI compliance")
            result.metadata.setdefault('compliance_flags', []).append('PCI_REQUIRED')
        
        # International transaction compliance
        if data.get('cross_border', False):
            result.add_info("Cross-border transaction requires additional compliance checks")
            result.metadata.setdefault('compliance_flags', []).append('CROSS_BORDER')
    
    def _validate_transaction_security(self, data: Dict[str, Any], result: EnhancedValidationResult):
        """Enhanced security validation for transactions"""
        # Check for suspicious patterns in transaction data
        for key, value in data.items():
            if isinstance(value, str):
                if self._contains_suspicious_patterns(value):
                    result.add_warning(f"Suspicious pattern detected in {key}", "SUSPICIOUS_PATTERN")
        
        # Validate IP address if present
        ip_address = data.get('ip_address')
        if ip_address and not self._is_valid_ip(ip_address):
            result.add_error("Invalid IP address format", "INVALID_IP")
        
        # Check for known malicious indicators
        if self._contains_malicious_indicators(data):
            result.add_error("Malicious indicators detected", "MALICIOUS_INDICATORS")
    
    def _validate_business_rules(self, data: Dict[str, Any], result: EnhancedValidationResult):
        """Enhanced business rules validation"""
        amount = data.get('amount', 0)
        tx_type = data.get('type', '').lower()
        
        # Business hour validation for large transactions
        if amount > 50000 and not self._is_business_hours():
            result.add_warning("Large transaction outside business hours", "AFTER_HOURS_LARGE_TX")
        
        # Velocity checks
        customer_id = data.get('customer', {}).get('id')
        if customer_id and self._check_velocity_violation(customer_id, amount):
            result.add_error("Transaction velocity limit exceeded", "VELOCITY_EXCEEDED")
        
        # Amount limits per transaction type
        type_limits = {
            'withdrawal': 25000,
            'transfer': 100000,
            'purchase': 50000
        }
        
        if tx_type in type_limits and amount > type_limits[tx_type]:
            result.add_warning(f"{tx_type.title()} amount exceeds recommended limit", "TYPE_LIMIT_EXCEEDED")
    
    def _validate_with_ml_models(self, data: Dict[str, Any], result: EnhancedValidationResult):
        """ML-based validation (placeholder for actual ML integration)"""
        # This would integrate with actual ML models
        result.add_info("ML validation would be performed here")
        result.metadata['ml_validation'] = {
            'fraud_score': 0.15,  # Mock score
            'anomaly_score': 0.08,  # Mock score
            'model_version': '1.2.3'
        }
    
    def _calculate_validation_scores(self, result: EnhancedValidationResult):
        """Calculate validation scores based on findings"""
        # Security score (1.0 = secure, 0.0 = high risk)
        security_penalties = 0.0
        for error in result.errors:
            if any(keyword in error.lower() for keyword in ['security', 'malicious', 'suspicious']):
                security_penalties += 0.3
        
        result.security_score = max(0.0, 1.0 - security_penalties)
        
        # Performance score (based on validation time)
        if result.validation_time_ms > self.config.max_validation_time_ms:
            result.performance_score = 0.5
        elif result.validation_time_ms > self.config.max_validation_time_ms * 0.8:
            result.performance_score = 0.8
        else:
            result.performance_score = 1.0
        
        # Reliability score (based on error count)
        error_count = len(result.errors)
        if error_count == 0:
            result.reliability_score = 1.0
        elif error_count <= 2:
            result.reliability_score = 0.8
        elif error_count <= 5:
            result.reliability_score = 0.6
        else:
            result.reliability_score = 0.3
    
    def _is_suspicious_amount(self, amount: float) -> bool:
        """Check for suspicious amount patterns"""
        # Round numbers might be suspicious
        if amount == round(amount) and amount > 1000:
            return True
        
        # Amounts just under reporting thresholds
        suspicious_thresholds = [9999, 4999, 2999]
        for threshold in suspicious_thresholds:
            if abs(amount - threshold) < 1:
                return True
        
        return False
    
    def _is_suspicious_transaction_id(self, tx_id: str) -> bool:
        """Check for suspicious transaction ID patterns"""
        # Sequential patterns
        if re.search(r'(\d)\1{4,}', tx_id):  # Repeated digits
            return True
        
        # Common test patterns
        test_patterns = ['TEST', '1234', 'ABCD', '0000']
        return any(pattern in tx_id.upper() for pattern in test_patterns)
    
    def _contains_suspicious_patterns(self, value: str) -> bool:
        """Check for suspicious patterns in string values"""
        suspicious_patterns = [
            r'<script',
            r'javascript:',
            r'SELECT.*FROM',
            r'DROP.*TABLE',
            r'\.\./',
            r'\\x[0-9a-f]{2}'
        ]
        
        return any(re.search(pattern, value, re.IGNORECASE) for pattern in suspicious_patterns)
    
    def _is_valid_ip(self, ip_address: str) -> bool:
        """Validate IP address format"""
        try:
            ipaddress.ip_address(ip_address)
            return True
        except ValueError:
            return False
    
    def _contains_malicious_indicators(self, data: Dict[str, Any]) -> bool:
        """Check for known malicious indicators"""
        # This would integrate with threat intelligence feeds
        # For now, check for obvious malicious patterns
        malicious_patterns = ['<script>', 'javascript:', 'data:text/html']
        
        for value in data.values():
            if isinstance(value, str):
                if any(pattern in value.lower() for pattern in malicious_patterns):
                    return True
        
        return False
    
    def _is_business_hours(self) -> bool:
        """Check if current time is within business hours"""
        now = datetime.now()
        return 9 <= now.hour <= 17 and now.weekday() < 5  # 9 AM - 5 PM, Monday-Friday
    
    def _check_velocity_violation(self, customer_id: str, amount: float) -> bool:
        """Check velocity limits (placeholder for actual implementation)"""
        # This would check against actual transaction history
        return False  # Mock implementation
    
    def _generate_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key for transaction data"""
        # Create hash of essential transaction data
        essential_data = {
            'amount': data.get('amount'),
            'type': data.get('type'),
            'currency': data.get('currency'),
            'merchant_id': data.get('merchant', {}).get('id')
        }
        
        data_str = json.dumps(essential_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[EnhancedValidationResult]:
        """Get cached validation result"""
        with self._cache_lock:
            if cache_key in self._validation_cache:
                cached_item = self._validation_cache[cache_key]
                if time.time() - cached_item['timestamp'] < self.config.cache_ttl_seconds:
                    return cached_item['result']
                else:
                    del self._validation_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: EnhancedValidationResult, ttl_seconds: int):
        """Cache validation result"""
        with self._cache_lock:
            # Implement simple LRU eviction
            if len(self._validation_cache) >= self.config.cache_max_size:
                oldest_key = min(self._validation_cache.keys(), 
                               key=lambda k: self._validation_cache[k]['timestamp'])
                del self._validation_cache[oldest_key]
            
            self._validation_cache[cache_key] = {
                'result': result,
                'timestamp': time.time(),
                'ttl': ttl_seconds
            }


class EnhancedPatternValidator:
    """Enhanced validator for routing patterns with ML capabilities"""
    
    def __init__(self, config: EnhancedRoutingValidationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._pattern_cache = {}
        self._pattern_performance = defaultdict(list)
        self._cache_lock = threading.RLock()
    
    def validate(self, pattern: Any, 
                context: Optional[EnhancedValidationContext] = None) -> EnhancedValidationResult:
        """Enhanced validation of domain patterns"""
        start_time = time.time()
        validation_id = f"pattern_val_{time.time()}"
        
        result = EnhancedValidationResult(
            is_valid=True,
            validation_level=context.validation_level if context else self.config.validation_level
        )
        
        try:
            # Handle different pattern types
            if hasattr(pattern, 'domain_name'):
                self._validate_domain_pattern(pattern, result)
            elif isinstance(pattern, dict):
                self._validate_pattern_dict(pattern, result)
            elif isinstance(pattern, str):
                self._validate_pattern_string(pattern, result)
            else:
                result.add_error("Unsupported pattern type", "UNSUPPORTED_PATTERN_TYPE")
            
            # Enhanced pattern analysis
            if result.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                self._analyze_pattern_complexity(pattern, result)
                self._check_pattern_security(pattern, result)
            
            # ML-based pattern validation
            if result.validation_level == ValidationLevel.PARANOID and self.config.enable_ml_validation:
                self._validate_pattern_with_ml(pattern, result)
            
            # Calculate scores
            self._calculate_pattern_scores(pattern, result)
            
            # Set validation metadata
            result.validation_time_ms = (time.time() - start_time) * 1000
            result.metadata.update({
                'validation_id': validation_id,
                'validation_timestamp': datetime.now().isoformat(),
                'validator_version': '2.0.0',
                'pattern_type': type(pattern).__name__
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pattern validation failed: {e}")
            result.add_error(f"Validation error: {str(e)}", "VALIDATION_EXCEPTION")
            result.validation_time_ms = (time.time() - start_time) * 1000
            return result
    
    def _validate_domain_pattern(self, pattern: Any, result: EnhancedValidationResult):
        """Validate domain pattern object"""
        # Validate domain name
        if not hasattr(pattern, 'domain_name') or not pattern.domain_name:
            result.add_error("Domain name is required", "MISSING_DOMAIN_NAME")
        elif not re.match(r'^[a-z_]+[a-z0-9_]*$', pattern.domain_name):
            result.add_error(f"Invalid domain name format: {pattern.domain_name}", "INVALID_DOMAIN_NAME")
        
        # Validate keywords
        if hasattr(pattern, 'keywords'):
            self._validate_keywords(pattern.keywords, result)
        
        # Validate patterns
        if hasattr(pattern, 'patterns'):
            self._validate_pattern_list(pattern.patterns, result)
        
        # Validate regex patterns
        if hasattr(pattern, 'regex_patterns'):
            self._validate_regex_patterns(pattern.regex_patterns, result)
        
        # Validate priority
        if hasattr(pattern, 'priority'):
            self._validate_priority(pattern.priority, result)
        
        # Validate confidence boost
        if hasattr(pattern, 'confidence_boost'):
            self._validate_confidence_boost(pattern.confidence_boost, result)
    
    def _validate_pattern_dict(self, pattern: Dict[str, Any], result: EnhancedValidationResult):
        """Validate pattern dictionary"""
        required_fields = ['name', 'patterns']
        for field in required_fields:
            if field not in pattern:
                result.add_error(f"Missing required field: {field}", f"MISSING_{field.upper()}")
        
        # Validate individual fields
        if 'patterns' in pattern:
            self._validate_pattern_list(pattern['patterns'], result)
        
        if 'keywords' in pattern:
            self._validate_keywords(pattern['keywords'], result)
    
    def _validate_pattern_string(self, pattern: str, result: EnhancedValidationResult):
        """Validate pattern string"""
        if len(pattern) > self.config.max_pattern_length:
            result.add_error(f"Pattern too long: {len(pattern)} > {self.config.max_pattern_length}", "PATTERN_TOO_LONG")
        
        if not pattern.strip():
            result.add_error("Empty pattern string", "EMPTY_PATTERN")
        
        # Check for suspicious content
        if self._contains_suspicious_content(pattern):
            result.add_error("Pattern contains suspicious content", "SUSPICIOUS_PATTERN")
    
    def _validate_keywords(self, keywords: List[str], result: EnhancedValidationResult):
        """Validate keyword list"""
        if not isinstance(keywords, list):
            result.add_error("Keywords must be a list", "INVALID_KEYWORDS_TYPE")
            return
        
        for i, keyword in enumerate(keywords):
            if not isinstance(keyword, str):
                result.add_error(f"Keyword {i} must be a string", f"INVALID_KEYWORD_{i}_TYPE")
            elif len(keyword) > self.config.max_keyword_length:
                result.add_error(f"Keyword {i} too long: {keyword[:50]}...", f"KEYWORD_{i}_TOO_LONG")
            elif not keyword.strip():
                result.add_error(f"Keyword {i} is empty", f"EMPTY_KEYWORD_{i}")
            elif self._contains_suspicious_content(keyword):
                result.add_warning(f"Keyword {i} contains suspicious content", f"SUSPICIOUS_KEYWORD_{i}")
    
    def _validate_pattern_list(self, patterns: List[str], result: EnhancedValidationResult):
        """Validate pattern list"""
        if not isinstance(patterns, list):
            result.add_error("Patterns must be a list", "INVALID_PATTERNS_TYPE")
            return
        
        for i, pattern in enumerate(patterns):
            if not isinstance(pattern, str):
                result.add_error(f"Pattern {i} must be a string", f"INVALID_PATTERN_{i}_TYPE")
            elif len(pattern) > self.config.max_pattern_length:
                result.add_error(f"Pattern {i} too long: {pattern[:50]}...", f"PATTERN_{i}_TOO_LONG")
            elif not pattern.strip():
                result.add_error(f"Pattern {i} is empty", f"EMPTY_PATTERN_{i}")
    
    def _validate_regex_patterns(self, regex_patterns: List[str], result: EnhancedValidationResult):
        """Validate regex patterns"""
        if not isinstance(regex_patterns, list):
            result.add_error("Regex patterns must be a list", "INVALID_REGEX_TYPE")
            return
        
        for i, regex in enumerate(regex_patterns):
            try:
                compiled = re.compile(regex)
                
                # Check complexity
                if len(regex) > self.config.max_regex_complexity:
                    result.add_warning(f"Regex {i} may be too complex: {regex[:30]}...", f"COMPLEX_REGEX_{i}")
                
                # Check for dangerous patterns
                if self._is_dangerous_regex(regex):
                    result.add_error(f"Regex {i} contains dangerous patterns", f"DANGEROUS_REGEX_{i}")
                
            except re.error as e:
                result.add_error(f"Invalid regex {i}: {regex} - {str(e)}", f"INVALID_REGEX_{i}")
    
    def _validate_priority(self, priority: Union[int, float], result: EnhancedValidationResult):
        """Validate priority value"""
        if not isinstance(priority, (int, float)):
            result.add_error("Priority must be a number", "INVALID_PRIORITY_TYPE")
        elif not 1 <= priority <= 10:
            result.add_error(f"Priority must be between 1 and 10, got {priority}", "INVALID_PRIORITY_RANGE")
    
    def _validate_confidence_boost(self, boost: Union[int, float], result: EnhancedValidationResult):
        """Validate confidence boost value"""
        if not isinstance(boost, (int, float)):
            result.add_error("Confidence boost must be a number", "INVALID_BOOST_TYPE")
        elif not self.config.confidence_penalty_limit <= boost <= self.config.confidence_boost_limit:
            result.add_error(
                f"Confidence boost must be between {self.config.confidence_penalty_limit} and {self.config.confidence_boost_limit}, got {boost}",
                "INVALID_BOOST_RANGE"
            )
    
    def _analyze_pattern_complexity(self, pattern: Any, result: EnhancedValidationResult):
        """Analyze pattern complexity for performance implications"""
        complexity_score = 0
        
        # Analyze based on pattern type
        if hasattr(pattern, 'regex_patterns'):
            for regex in pattern.regex_patterns:
                complexity_score += self._calculate_regex_complexity(regex)
        
        if hasattr(pattern, 'keywords'):
            complexity_score += len(pattern.keywords) * 0.1
        
        if complexity_score > 10:
            result.add_warning("High complexity pattern may impact performance", "HIGH_COMPLEXITY")
        
        result.metadata['complexity_score'] = complexity_score
    
    def _check_pattern_security(self, pattern: Any, result: EnhancedValidationResult):
        """Check pattern for security issues"""
        security_issues = []
        
        # Check all string fields for suspicious content
        if hasattr(pattern, '__dict__'):
            for attr_name, attr_value in pattern.__dict__.items():
                if isinstance(attr_value, str):
                    if self._contains_suspicious_content(attr_value):
                        security_issues.append(f"Suspicious content in {attr_name}")
                elif isinstance(attr_value, list):
                    for item in attr_value:
                        if isinstance(item, str) and self._contains_suspicious_content(item):
                            security_issues.append(f"Suspicious content in {attr_name} list")
        
        for issue in security_issues:
            result.add_warning(issue, "SECURITY_ISSUE")
    
    def _validate_pattern_with_ml(self, pattern: Any, result: EnhancedValidationResult):
        """ML-based pattern validation"""
        # Placeholder for ML-based validation
        result.add_info("ML pattern validation would be performed here")
        result.metadata['ml_pattern_validation'] = {
            'malicious_probability': 0.05,  # Mock score
            'effectiveness_score': 0.85,  # Mock score
            'model_version': '1.1.0'
        }
    
    def _calculate_pattern_scores(self, pattern: Any, result: EnhancedValidationResult):
        """Calculate pattern validation scores"""
        # Security score
        security_penalties = len([e for e in result.errors if 'security' in e.lower() or 'suspicious' in e.lower()]) * 0.2
        result.security_score = max(0.0, 1.0 - security_penalties)
        
        # Performance score
        complexity_score = result.metadata.get('complexity_score', 0)
        if complexity_score > 15:
            result.performance_score = 0.3
        elif complexity_score > 10:
            result.performance_score = 0.6
        elif complexity_score > 5:
            result.performance_score = 0.8
        else:
            result.performance_score = 1.0
        
        # Reliability score
        error_count = len(result.errors)
        if error_count == 0:
            result.reliability_score = 1.0
        elif error_count <= 2:
            result.reliability_score = 0.8
        else:
            result.reliability_score = 0.5
    
    def _contains_suspicious_content(self, content: str) -> bool:
        """Check for suspicious content in patterns"""
        suspicious_patterns = [
            r'<script',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
            r'file://',
            r'\\x[0-9a-f]{2}',
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'\.\./',
            r'<iframe',
            r'<object'
        ]
        
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in suspicious_patterns)
    
    def _is_dangerous_regex(self, regex: str) -> bool:
        """Check if regex contains dangerous patterns"""
        dangerous_patterns = [
            r'\(\?\=',  # Positive lookahead
            r'\(\?\!',  # Negative lookahead
            r'\*\+',    # Catastrophic backtracking
            r'\+\*',    # Catastrophic backtracking
            r'\{[0-9]+,\}.*\*',  # Complex quantifiers
        ]
        
        return any(re.search(pattern, regex) for pattern in dangerous_patterns)
    
    def _calculate_regex_complexity(self, regex: str) -> float:
        """Calculate regex complexity score"""
        complexity = 0
        
        # Count quantifiers
        complexity += len(re.findall(r'[*+?{}]', regex)) * 0.5
        
        # Count character classes
        complexity += len(re.findall(r'\[.*?\]', regex)) * 0.3
        
        # Count groups
        complexity += len(re.findall(r'\(.*?\)', regex)) * 0.4
        
        # Count lookaheads/lookbehinds
        complexity += len(re.findall(r'\(\?[=!<]', regex)) * 1.0
        
        return complexity


class EnhancedSecurityValidator:
    """Enhanced security validator with comprehensive threat detection"""
    
    def __init__(self, config: EnhancedRoutingValidationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._threat_cache = {}
        self._ip_reputation = {}
        self._cache_lock = threading.RLock()
        
        # Initialize threat detection patterns
        self._init_threat_patterns()
    
    def _init_threat_patterns(self):
        """Initialize threat detection patterns"""
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|EXEC)\b)",
            r"(--|\||;|\/\*|\*\/|#)",
            r"(\bOR\b\s*\d+\s*=\s*\d+)",
            r"(\bAND\b\s*\d+\s*=\s*\d+)",
            r"(\bUNION\b.*\bSELECT\b)",
            r"(\bDROP\b.*\bTABLE\b)",
            r"(\bEXEC\b.*\bXP_)",
            r"(\bSP_EXECUTESQL\b)",
            r"(\bBULK\b.*\bINSERT\b)"
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"<img[^>]*onerror[^>]*>",
            r"<svg[^>]*onload[^>]*>",
            r"javascript:",
            r"vbscript:",
            r"data:text/html",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
            r"<link[^>]*>",
            r"<style[^>]*>.*?</style>",
            r"expression\s*\(",
            r"@import"
        ]
        
        self.path_traversal_patterns = [
            r"\.\.[\\/]",
            r"[\\/]\.\.[\\/]",
            r"\.\.%2f",
            r"%2e%2e%2f",
            r"\.\.\\",
            r"%2e%2e\\",
            r"..%255c",
            r"..%c0%af"
        ]
        
        self.command_injection_patterns = [
            r";\s*(cat|ls|pwd|whoami|id|uname)",
            r"\|\s*(cat|ls|pwd|whoami|id|uname)",
            r"&&\s*(cat|ls|pwd|whoami|id|uname)",
            r"`[^`]*`",
            r"\$\([^)]*\)",
            r"%(SYSTEMROOT|WINDIR|TEMP|TMP)%",
            r"\$\{[^}]*\}",
            r"nc\s+-",
            r"wget\s+",
            r"curl\s+"
        ]
        
        self.file_inclusion_patterns = [
            r"(file|http|https|ftp)://",
            r"php://filter",
            r"php://input",
            r"data://",
            r"expect://",
            r"zip://",
            r"compress.zlib://",
            r"\\\\[^\\]*\\",
            r"/proc/self/environ",
            r"/etc/passwd"
        ]
        
        self.xxe_patterns = [
            r"<!ENTITY[^>]*>",
            r"<!DOCTYPE[^>]*\[",
            r"SYSTEM\s+[\"'][^\"']*[\"']",
            r"PUBLIC\s+[\"'][^\"']*[\"']",
            r"&[a-zA-Z]+;",
            r"&#[0-9]+;",
            r"&#x[0-9a-fA-F]+;"
        ]
    
    def validate_input_security(self, input_data: Any, 
                               context: Optional[EnhancedValidationContext] = None) -> EnhancedValidationResult:
        """Enhanced security validation of input data"""
        start_time = time.time()
        validation_id = f"sec_val_{time.time()}"
        
        result = EnhancedValidationResult(
            is_valid=True,
            validation_level=context.validation_level if context else self.config.validation_level
        )
        
        try:
            input_str = str(input_data)
            
            # Size validation
            self._validate_input_size(input_str, result)
            
            # Basic security validations
            if self.config.enable_sql_injection_check:
                self._check_sql_injection(input_str, result)
            
            if self.config.enable_xss_check:
                self._check_xss(input_str, result)
            
            if self.config.enable_path_traversal_check:
                self._check_path_traversal(input_str, result)
            
            if self.config.enable_command_injection_check:
                self._check_command_injection(input_str, result)
            
            if self.config.enable_file_inclusion_check:
                self._check_file_inclusion(input_str, result)
            
            if self.config.enable_xxe_check:
                self._check_xxe(input_str, result)
            
            # Enhanced security validations for higher levels
            if result.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                self._check_advanced_threats(input_str, result)
                self._check_malware_indicators(input_str, result)
                self._check_cryptocurrency_addresses(input_str, result)
            
            # Deep security scan for paranoid level
            if result.validation_level == ValidationLevel.PARANOID:
                self._perform_deep_security_scan(input_str, result)
            
            # Context-based security validation
            if context and context.security_context:
                self._validate_security_context(context.security_context, result)
            
            # Calculate security scores
            self._calculate_security_scores(result)
            
            # Set validation metadata
            result.validation_time_ms = (time.time() - start_time) * 1000
            result.metadata.update({
                'validation_id': validation_id,
                'validation_timestamp': datetime.now().isoformat(),
                'validator_version': '2.0.0',
                'security_scan_depth': self.config.security_scan_depth,
                'input_size_bytes': len(input_str.encode('utf-8'))
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Security validation failed: {e}")
            result.add_error(f"Security validation error: {str(e)}", "SECURITY_VALIDATION_EXCEPTION")
            result.validation_time_ms = (time.time() - start_time) * 1000
            return result
    
    def _validate_input_size(self, input_str: str, result: EnhancedValidationResult):
        """Validate input size"""
        input_size = len(input_str.encode('utf-8'))
        if input_size > self.config.max_request_size_bytes:
            result.add_error(
                f"Input size {input_size} bytes exceeds maximum {self.config.max_request_size_bytes}",
                "INPUT_TOO_LARGE"
            )
    
    def _check_sql_injection(self, input_str: str, result: EnhancedValidationResult):
        """Check for SQL injection patterns"""
        threats_found = []
        
        for i, pattern in enumerate(self.sql_injection_patterns):
            matches = re.finditer(pattern, input_str, re.IGNORECASE)
            for match in matches:
                threats_found.append({
                    'type': 'sql_injection',
                    'pattern_id': i,
                    'match': match.group(),
                    'position': match.span()
                })
        
        if threats_found:
            result.add_error("SQL injection patterns detected", "SQL_INJECTION_DETECTED")
            result.metadata.setdefault('security_threats', []).extend(threats_found)
    
    def _check_xss(self, input_str: str, result: EnhancedValidationResult):
        """Check for XSS patterns"""
        threats_found = []
        
        for i, pattern in enumerate(self.xss_patterns):
            matches = re.finditer(pattern, input_str, re.IGNORECASE | re.DOTALL)
            for match in matches:
                threats_found.append({
                    'type': 'xss',
                    'pattern_id': i,
                    'match': match.group()[:100],  # Limit match length
                    'position': match.span()
                })
        
        if threats_found:
            result.add_error("XSS patterns detected", "XSS_DETECTED")
            result.metadata.setdefault('security_threats', []).extend(threats_found)
    
    def _check_path_traversal(self, input_str: str, result: EnhancedValidationResult):
        """Check for path traversal patterns"""
        threats_found = []
        
        for i, pattern in enumerate(self.path_traversal_patterns):
            matches = re.finditer(pattern, input_str, re.IGNORECASE)
            for match in matches:
                threats_found.append({
                    'type': 'path_traversal',
                    'pattern_id': i,
                    'match': match.group(),
                    'position': match.span()
                })
        
        if threats_found:
            result.add_error("Path traversal patterns detected", "PATH_TRAVERSAL_DETECTED")
            result.metadata.setdefault('security_threats', []).extend(threats_found)
    
    def _check_command_injection(self, input_str: str, result: EnhancedValidationResult):
        """Check for command injection patterns"""
        threats_found = []
        
        for i, pattern in enumerate(self.command_injection_patterns):
            matches = re.finditer(pattern, input_str, re.IGNORECASE)
            for match in matches:
                threats_found.append({
                    'type': 'command_injection',
                    'pattern_id': i,
                    'match': match.group(),
                    'position': match.span()
                })
        
        if threats_found:
            result.add_error("Command injection patterns detected", "COMMAND_INJECTION_DETECTED")
            result.metadata.setdefault('security_threats', []).extend(threats_found)
    
    def _check_file_inclusion(self, input_str: str, result: EnhancedValidationResult):
        """Check for file inclusion patterns"""
        threats_found = []
        
        for i, pattern in enumerate(self.file_inclusion_patterns):
            matches = re.finditer(pattern, input_str, re.IGNORECASE)
            for match in matches:
                threats_found.append({
                    'type': 'file_inclusion',
                    'pattern_id': i,
                    'match': match.group(),
                    'position': match.span()
                })
        
        if threats_found:
            result.add_warning("File inclusion patterns detected", "FILE_INCLUSION_DETECTED")
            result.metadata.setdefault('security_threats', []).extend(threats_found)
    
    def _check_xxe(self, input_str: str, result: EnhancedValidationResult):
        """Check for XXE patterns"""
        threats_found = []
        
        for i, pattern in enumerate(self.xxe_patterns):
            matches = re.finditer(pattern, input_str, re.IGNORECASE | re.DOTALL)
            for match in matches:
                threats_found.append({
                    'type': 'xxe',
                    'pattern_id': i,
                    'match': match.group()[:100],
                    'position': match.span()
                })
        
        if threats_found:
            result.add_error("XXE patterns detected", "XXE_DETECTED")
            result.metadata.setdefault('security_threats', []).extend(threats_found)
    
    def _check_advanced_threats(self, input_str: str, result: EnhancedValidationResult):
        """Check for advanced threat patterns"""
        advanced_patterns = [
            (r'eval\s*\(', 'code_injection'),
            (r'exec\s*\(', 'code_injection'),
            (r'system\s*\(', 'system_call'),
            (r'shell_exec\s*\(', 'shell_execution'),
            (r'passthru\s*\(', 'command_execution'),
            (r'base64_decode\s*\(', 'obfuscation'),
            (r'gzinflate\s*\(', 'compression_obfuscation'),
            (r'str_rot13\s*\(', 'encoding_obfuscation'),
            (r'\\x[0-9a-f]{2}', 'hex_encoding'),
            (r'%[0-9a-f]{2}', 'url_encoding'),
            (r'\\u[0-9a-f]{4}', 'unicode_encoding')
        ]
        
        threats_found = []
        for pattern, threat_type in advanced_patterns:
            matches = re.finditer(pattern, input_str, re.IGNORECASE)
            for match in matches:
                threats_found.append({
                    'type': threat_type,
                    'match': match.group(),
                    'position': match.span()
                })
        
        if threats_found:
            result.add_warning("Advanced threat patterns detected", "ADVANCED_THREATS_DETECTED")
            result.metadata.setdefault('advanced_threats', []).extend(threats_found)
    
    def _check_malware_indicators(self, input_str: str, result: EnhancedValidationResult):
        """Check for malware indicators"""
        malware_patterns = [
            r'metasploit',
            r'meterpreter',
            r'cobalt.?strike',
            r'empire.?framework',
            r'mimikatz',
            r'powersploit',
            r'bloodhound',
            r'sharphound',
            r'rubeus',
            r'certify\.exe'
        ]
        
        indicators_found = []
        for pattern in malware_patterns:
            matches = re.finditer(pattern, input_str, re.IGNORECASE)
            for match in matches:
                indicators_found.append({
                    'type': 'malware_indicator',
                    'indicator': match.group(),
                    'position': match.span()
                })
        
        if indicators_found:
            result.add_error("Malware indicators detected", "MALWARE_INDICATORS_DETECTED")
            result.metadata.setdefault('malware_indicators', []).extend(indicators_found)
    
    def _check_cryptocurrency_addresses(self, input_str: str, result: EnhancedValidationResult):
        """Check for cryptocurrency addresses (potential ransomware indicators)"""
        crypto_patterns = [
            r'[13][a-km-zA-HJ-NP-Z1-9]{25,34}',  # Bitcoin
            r'0x[a-fA-F0-9]{40}',  # Ethereum
            r'[LM3][a-km-zA-HJ-NP-Z1-9]{26,33}',  # Litecoin
            r'r[a-zA-Z0-9]{24,34}',  # Ripple
            r'bc1[a-z0-9]{39,59}'  # Bitcoin Bech32
        ]
        
        addresses_found = []
        for pattern in crypto_patterns:
            matches = re.finditer(pattern, input_str)
            for match in matches:
                addresses_found.append({
                    'type': 'cryptocurrency_address',
                    'address': match.group(),
                    'position': match.span()
                })
        
        if addresses_found:
            result.add_warning("Cryptocurrency addresses detected", "CRYPTO_ADDRESSES_DETECTED")
            result.metadata.setdefault('crypto_addresses', []).extend(addresses_found)
    
    def _perform_deep_security_scan(self, input_str: str, result: EnhancedValidationResult):
        """Perform deep security scan for paranoid validation level"""
        # Entropy analysis
        entropy = self._calculate_entropy(input_str)
        if entropy > 4.5:  # High entropy might indicate obfuscation
            result.add_warning(f"High entropy detected: {entropy:.2f}", "HIGH_ENTROPY")
        
        # Check for null bytes
        if '\x00' in input_str:
            result.add_error("Null bytes detected in input", "NULL_BYTES_DETECTED")
        
        # Check for control characters
        control_chars = [c for c in input_str if ord(c) < 32 and c not in '\t\n\r']
        if control_chars:
            result.add_warning(f"Control characters detected: {len(control_chars)}", "CONTROL_CHARS_DETECTED")
        
        # Check for long single-character sequences
        repeated_chars = re.findall(r'(.)\1{50,}', input_str)
        if repeated_chars:
            result.add_warning("Long repeated character sequences detected", "REPEATED_CHARS_DETECTED")
        
        result.metadata.update({
            'entropy': entropy,
            'control_chars_count': len(control_chars),
            'repeated_sequences': len(repeated_chars)
        })
    
    def _validate_security_context(self, security_context: Dict[str, Any], result: EnhancedValidationResult):
        """Validate security context information"""
        # Check failed attempts
        failed_attempts = security_context.get('failed_attempts', 0)
        if failed_attempts > 5:
            result.add_error("Too many failed attempts", "EXCESSIVE_FAILED_ATTEMPTS")
        elif failed_attempts > 3:
            result.add_warning("Multiple failed attempts detected", "MULTIPLE_FAILED_ATTEMPTS")
        
        # Check IP reputation
        ip_address = security_context.get('ip_address')
        if ip_address:
            if ip_address in self.config.ip_blacklist:
                result.add_error("Request from blacklisted IP", "BLACKLISTED_IP")
            elif self.config.ip_whitelist and ip_address not in self.config.ip_whitelist:
                result.add_warning("Request from non-whitelisted IP", "NON_WHITELISTED_IP")
        
        # Check rate limiting
        request_count = security_context.get('requests_per_minute', 0)
        if request_count > self.config.rate_limit_requests_per_minute:
            result.add_error("Rate limit exceeded", "RATE_LIMIT_EXCEEDED")
        
        # Check user agent
        user_agent = security_context.get('user_agent', '')
        if self._is_suspicious_user_agent(user_agent):
            result.add_warning("Suspicious user agent detected", "SUSPICIOUS_USER_AGENT")
    
    def _calculate_security_scores(self, result: EnhancedValidationResult):
        """Calculate security-related scores"""
        # Security score based on threats found
        threat_count = len(result.metadata.get('security_threats', []))
        advanced_threat_count = len(result.metadata.get('advanced_threats', []))
        malware_indicator_count = len(result.metadata.get('malware_indicators', []))
        
        total_threats = threat_count + advanced_threat_count + malware_indicator_count
        
        if total_threats == 0:
            result.security_score = 1.0
        elif total_threats <= 2:
            result.security_score = 0.7
        elif total_threats <= 5:
            result.security_score = 0.4
        else:
            result.security_score = 0.1
        
        # Performance score (security validation can be slow)
        if result.validation_time_ms > 1000:
            result.performance_score = 0.5
        elif result.validation_time_ms > 500:
            result.performance_score = 0.8
        else:
            result.performance_score = 1.0
        
        # Reliability score
        error_count = len([e for e in result.errors if 'security' in e.lower()])
        if error_count == 0:
            result.reliability_score = 1.0
        elif error_count <= 2:
            result.reliability_score = 0.8
        else:
            result.reliability_score = 0.5
    
    def _calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of string"""
        if not data:
            return 0
        
        # Count character frequencies
        freq = {}
        for char in data:
            freq[char] = freq.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0
        data_len = len(data)
        for count in freq.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check if user agent is suspicious"""
        suspicious_agents = [
            'sqlmap',
            'nikto',
            'nmap',
            'masscan',
            'gobuster',
            'dirb',
            'curl',
            'wget',
            'python-requests',
            'scanner'
        ]
        
        user_agent_lower = user_agent.lower()
        return any(agent in user_agent_lower for agent in suspicious_agents)


# ==================== ENHANCED CIRCUIT BREAKER AND MONITORING ====================

class EnhancedCircuitBreaker:
    """Enhanced circuit breaker with comprehensive monitoring and recovery"""
    
    def __init__(self, config: EnhancedRoutingValidationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Circuit breaker state
        self.failure_threshold = config.circuit_breaker_threshold
        self.timeout_seconds = config.circuit_breaker_timeout_seconds
        self.recovery_threshold = config.circuit_breaker_recovery_threshold
        self.min_calls = 10
        
        # State tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.state = EnhancedRoutingHealthStatus.HEALTHY
        self.state_change_time = datetime.now()
        
        # Enhanced monitoring
        self.failure_history = deque(maxlen=100)
        self.response_times = deque(maxlen=100)
        self.error_types = defaultdict(int)
        self.recovery_attempts = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Metrics
        self.metrics = {
            'total_calls': 0,
            'total_failures': 0,
            'total_successes': 0,
            'state_changes': 0,
            'last_state_change': datetime.now().isoformat(),
            'average_response_time_ms': 0.0,
            'failure_rate': 0.0
        }
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with enhanced circuit breaker protection"""
        start_time = time.time()
        
        with self._lock:
            self.metrics['total_calls'] += 1
            
            # Check circuit state
            if self.state == EnhancedRoutingHealthStatus.CIRCUIT_OPEN:
                if self._should_attempt_reset():
                    self.state = EnhancedRoutingHealthStatus.RECOVERING
                    self.state_change_time = datetime.now()
                    self.recovery_attempts += 1
                    self.logger.info("Circuit breaker attempting recovery")
                else:
                    raise EnhancedRoutingCircuitBreakerError(
                        "Circuit breaker is open",
                        circuit_state=self.state.value,
                        failure_rate=self._calculate_failure_rate()
                    )
        
        try:
            result = func(*args, **kwargs)
            response_time = (time.time() - start_time) * 1000
            
            self._record_success(response_time)
            return result
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self._record_failure(e, response_time)
            raise
    
    def _record_success(self, response_time: float):
        """Record successful call with enhanced metrics"""
        with self._lock:
            self.success_count += 1
            self.last_success_time = datetime.now()
            self.response_times.append(response_time)
            self.metrics['total_successes'] += 1
            
            # Update average response time
            if self.response_times:
                self.metrics['average_response_time_ms'] = sum(self.response_times) / len(self.response_times)
            
            # Check for state transitions
            if self.state == EnhancedRoutingHealthStatus.RECOVERING:
                # Check if we should fully recover
                total_calls = self.success_count + self.failure_count
                if total_calls >= self.min_calls:
                    success_rate = self.success_count / total_calls
                    if success_rate >= self.recovery_threshold:
                        self._change_state(EnhancedRoutingHealthStatus.HEALTHY)
                        self.logger.info("Circuit breaker recovered to healthy state")
            
            elif self.state == EnhancedRoutingHealthStatus.DEGRADED:
                # Check if we should return to healthy
                recent_success_rate = self._calculate_recent_success_rate()
                if recent_success_rate >= 0.9:  # 90% success rate
                    self._change_state(EnhancedRoutingHealthStatus.HEALTHY)
                    self.logger.info("Circuit breaker recovered from degraded state")
    
    def _record_failure(self, error: Exception, response_time: float):
        """Record failed call with enhanced error tracking"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            self.response_times.append(response_time)
            self.metrics['total_failures'] += 1
            
            # Track error types
            error_type = type(error).__name__
            self.error_types[error_type] += 1
            
            # Record failure in history
            self.failure_history.append({
                'timestamp': datetime.now().isoformat(),
                'error_type': error_type,
                'error_message': str(error)[:200],  # Truncate long messages
                'response_time_ms': response_time
            })
            
            # Update failure rate
            self.metrics['failure_rate'] = self._calculate_failure_rate()
            
            # Check for state transitions
            total_calls = self.success_count + self.failure_count
            if total_calls >= self.min_calls:
                failure_rate = self.failure_count / total_calls
                
                if self.state == EnhancedRoutingHealthStatus.HEALTHY:
                    if failure_rate >= self.failure_threshold:
                        self._change_state(EnhancedRoutingHealthStatus.CIRCUIT_OPEN)
                        self.logger.warning(f"Circuit breaker opened due to failure rate: {failure_rate:.2%}")
                    elif failure_rate >= self.failure_threshold * 0.7:  # 70% of threshold
                        self._change_state(EnhancedRoutingHealthStatus.DEGRADED)
                        self.logger.warning(f"Circuit breaker degraded due to failure rate: {failure_rate:.2%}")
                
                elif self.state == EnhancedRoutingHealthStatus.RECOVERING:
                    if failure_rate >= self.failure_threshold:
                        self._change_state(EnhancedRoutingHealthStatus.CIRCUIT_OPEN)
                        self.logger.warning("Circuit breaker recovery failed, reopening")
    
    def _should_attempt_reset(self) -> bool:
        """Enhanced logic for determining if circuit should attempt reset"""
        if not self.last_failure_time:
            return True
        
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        
        # Base timeout
        if time_since_failure < self.timeout_seconds:
            return False
        
        # Exponential backoff for repeated failures
        backoff_multiplier = min(2 ** self.recovery_attempts, 8)  # Cap at 8x
        adjusted_timeout = self.timeout_seconds * backoff_multiplier
        
        return time_since_failure >= adjusted_timeout
    
    def _calculate_failure_rate(self) -> float:
        """Calculate current failure rate"""
        total_calls = self.success_count + self.failure_count
        if total_calls == 0:
            return 0.0
        return self.failure_count / total_calls
    
    def _calculate_recent_success_rate(self, window_size: int = 20) -> float:
        """Calculate success rate for recent calls"""
        if len(self.failure_history) < window_size:
            return 1.0  # Assume success if not enough data
        
        recent_failures = sum(1 for f in list(self.failure_history)[-window_size:])
        return 1.0 - (recent_failures / window_size)
    
    def _change_state(self, new_state: EnhancedRoutingHealthStatus):
        """Change circuit breaker state with logging"""
        old_state = self.state
        self.state = new_state
        self.state_change_time = datetime.now()
        self.metrics['state_changes'] += 1
        self.metrics['last_state_change'] = self.state_change_time.isoformat()
        
        # Reset counters on state change
        if new_state == EnhancedRoutingHealthStatus.HEALTHY:
            self.failure_count = 0
            self.success_count = 0
            self.recovery_attempts = 0
        
        self.logger.info(f"Circuit breaker state changed: {old_state.value} -> {new_state.value}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker state"""
        with self._lock:
            recent_errors = list(self.failure_history)[-10:]  # Last 10 errors
            
            return {
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'failure_rate': self.metrics['failure_rate'],
                'recovery_attempts': self.recovery_attempts,
                'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
                'last_success_time': self.last_success_time.isoformat() if self.last_success_time else None,
                'state_change_time': self.state_change_time.isoformat(),
                'average_response_time_ms': self.metrics['average_response_time_ms'],
                'error_types': dict(self.error_types),
                'recent_errors': recent_errors,
                'metrics': self.metrics.copy()
            }
    
    def force_state(self, state: EnhancedRoutingHealthStatus, reason: str = "Manual override"):
        """Force circuit breaker to specific state"""
        with self._lock:
            self.logger.warning(f"Forcing circuit breaker state to {state.value}: {reason}")
            self._change_state(state)
    
    def reset(self):
        """Reset circuit breaker to initial state"""
        with self._lock:
            self.logger.info("Resetting circuit breaker")
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.last_success_time = None
            self.recovery_attempts = 0
            self.failure_history.clear()
            self.response_times.clear()
            self.error_types.clear()
            self._change_state(EnhancedRoutingHealthStatus.HEALTHY)


class EnhancedPerformanceMonitor:
    """Enhanced performance monitor with comprehensive metrics collection"""
    
    def __init__(self, config: EnhancedRoutingValidationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance metrics
        self.metrics = {
            'request_count': 0,
            'total_routing_time_ms': 0.0,
            'total_validation_time_ms': 0.0,
            'slow_requests': 0,
            'validation_failures': 0,
            'routing_failures': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'security_threats_detected': 0,
            'last_reset_time': datetime.now().isoformat()
        }
        
        # Performance history
        self.response_times = deque(maxlen=1000)
        self.validation_times = deque(maxlen=1000)
        self.hourly_stats = defaultdict(lambda: {
            'requests': 0,
            'avg_response_time': 0.0,
            'failures': 0
        })
        
        # Percentile tracking
        self.percentile_data = deque(maxlen=1000)
        
        # Resource monitoring
        self.resource_usage = {
            'cpu_percent': 0.0,
            'memory_mb': 0.0,
            'disk_usage_percent': 0.0,
            'last_check': datetime.now().isoformat()
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Start resource monitoring thread
        if config.enable_detailed_monitoring:
            self._start_resource_monitoring()
    
    def record_request(self, routing_time_ms: float, validation_time_ms: float = 0.0,
                      success: bool = True, request_type: str = "unknown"):
        """Record request performance metrics"""
        with self._lock:
            self.metrics['request_count'] += 1
            self.metrics['total_routing_time_ms'] += routing_time_ms
            self.metrics['total_validation_time_ms'] += validation_time_ms
            
            # Track response times
            self.response_times.append(routing_time_ms)
            self.validation_times.append(validation_time_ms)
            self.percentile_data.append(routing_time_ms)
            
            # Check for slow requests
            if routing_time_ms > self.config.max_routing_time_ms:
                self.metrics['slow_requests'] += 1
                self.logger.warning(f"Slow request detected: {routing_time_ms:.2f}ms")
            
            # Record failures
            if not success:
                if 'validation' in request_type.lower():
                    self.metrics['validation_failures'] += 1
                else:
                    self.metrics['routing_failures'] += 1
            
            # Update hourly stats
            current_hour = datetime.now().hour
            self.hourly_stats[current_hour]['requests'] += 1
            if not success:
                self.hourly_stats[current_hour]['failures'] += 1
            
            # Update rolling average for current hour
            hour_responses = [t for t in self.response_times if True]  # All recent responses
            if hour_responses:
                self.hourly_stats[current_hour]['avg_response_time'] = sum(hour_responses) / len(hour_responses)
    
    def record_cache_event(self, hit: bool):
        """Record cache hit/miss event"""
        with self._lock:
            if hit:
                self.metrics['cache_hits'] += 1
            else:
                self.metrics['cache_misses'] += 1
    
    def record_security_threat(self, threat_type: str):
        """Record security threat detection"""
        with self._lock:
            self.metrics['security_threats_detected'] += 1
            self.logger.warning(f"Security threat detected: {threat_type}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        with self._lock:
            # Calculate derived metrics
            avg_routing_time = (
                self.metrics['total_routing_time_ms'] / max(self.metrics['request_count'], 1)
            )
            
            avg_validation_time = (
                self.metrics['total_validation_time_ms'] / max(self.metrics['request_count'], 1)
            )
            
            cache_hit_rate = 0.0
            total_cache_requests = self.metrics['cache_hits'] + self.metrics['cache_misses']
            if total_cache_requests > 0:
                cache_hit_rate = self.metrics['cache_hits'] / total_cache_requests
            
            failure_rate = 0.0
            total_failures = self.metrics['validation_failures'] + self.metrics['routing_failures']
            if self.metrics['request_count'] > 0:
                failure_rate = total_failures / self.metrics['request_count']
            
            # Calculate percentiles
            percentiles = self._calculate_percentiles()
            
            return {
                'basic_metrics': self.metrics.copy(),
                'derived_metrics': {
                    'average_routing_time_ms': avg_routing_time,
                    'average_validation_time_ms': avg_validation_time,
                    'cache_hit_rate': cache_hit_rate,
                    'failure_rate': failure_rate,
                    'requests_per_second': self._calculate_rps(),
                    'slow_request_rate': self.metrics['slow_requests'] / max(self.metrics['request_count'], 1)
                },
                'percentiles': percentiles,
                'hourly_stats': dict(self.hourly_stats),
                'resource_usage': self.resource_usage.copy(),
                'health_indicators': self._get_health_indicators()
            }
    
    def _calculate_percentiles(self) -> Dict[str, float]:
        """Calculate response time percentiles"""
        if not self.percentile_data:
            return {}
        
        sorted_data = sorted(self.percentile_data)
        
        def percentile(data, p):
            index = int((len(data) - 1) * p)
            return data[index]
        
        return {
            'p50': percentile(sorted_data, 0.5),
            'p90': percentile(sorted_data, 0.9),
            'p95': percentile(sorted_data, 0.95),
            'p99': percentile(sorted_data, 0.99)
        }
    
    def _calculate_rps(self) -> float:
        """Calculate requests per second over recent window"""
        if not self.response_times:
            return 0.0
        
        # Use recent window of 60 seconds
        window_size = min(60, len(self.response_times))
        return window_size / 60.0 if window_size > 0 else 0.0
    
    def _get_health_indicators(self) -> Dict[str, str]:
        """Get performance health indicators"""
        indicators = {}
        
        # Response time health
        avg_time = self.metrics['total_routing_time_ms'] / max(self.metrics['request_count'], 1)
        if avg_time < self.config.max_routing_time_ms * 0.5:
            indicators['response_time'] = 'healthy'
        elif avg_time < self.config.max_routing_time_ms * 0.8:
            indicators['response_time'] = 'good'
        elif avg_time < self.config.max_routing_time_ms:
            indicators['response_time'] = 'degraded'
        else:
            indicators['response_time'] = 'poor'
        
        # Failure rate health
        total_failures = self.metrics['validation_failures'] + self.metrics['routing_failures']
        failure_rate = total_failures / max(self.metrics['request_count'], 1)
        if failure_rate < 0.01:  # Less than 1%
            indicators['reliability'] = 'healthy'
        elif failure_rate < 0.05:  # Less than 5%
            indicators['reliability'] = 'good'
        elif failure_rate < 0.1:  # Less than 10%
            indicators['reliability'] = 'degraded'
        else:
            indicators['reliability'] = 'poor'
        
        # Cache efficiency
        total_cache_requests = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total_cache_requests > 0:
            cache_hit_rate = self.metrics['cache_hits'] / total_cache_requests
            if cache_hit_rate > 0.9:
                indicators['cache_efficiency'] = 'excellent'
            elif cache_hit_rate > 0.7:
                indicators['cache_efficiency'] = 'good'
            elif cache_hit_rate > 0.5:
                indicators['cache_efficiency'] = 'average'
            else:
                indicators['cache_efficiency'] = 'poor'
        else:
            indicators['cache_efficiency'] = 'no_data'
        
        return indicators
    
    def _start_resource_monitoring(self):
        """Start background resource monitoring"""
        def monitor_resources():
            while True:
                try:
                    # Get system resource usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory_info = psutil.virtual_memory()
                    disk_info = psutil.disk_usage('/')
                    
                    with self._lock:
                        self.resource_usage.update({
                            'cpu_percent': cpu_percent,
                            'memory_mb': memory_info.used / (1024 * 1024),
                            'memory_percent': memory_info.percent,
                            'disk_usage_percent': disk_info.percent,
                            'last_check': datetime.now().isoformat()
                        })
                    
                    # Check thresholds and log warnings
                    if cpu_percent > self.config.max_cpu_usage_percent:
                        self.logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                    
                    if memory_info.percent > 80:  # 80% memory usage
                        self.logger.warning(f"High memory usage: {memory_info.percent:.1f}%")
                    
                    if disk_info.percent > self.config.max_disk_usage_percent:
                        self.logger.warning(f"High disk usage: {disk_info.percent:.1f}%")
                    
                    time.sleep(self.config.resource_check_interval_seconds)
                    
                except Exception as e:
                    self.logger.error(f"Resource monitoring error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        thread = threading.Thread(target=monitor_resources, daemon=True)
        thread.start()
    
    def reset_metrics(self):
        """Reset all performance metrics"""
        with self._lock:
            self.logger.info("Resetting performance metrics")
            
            self.metrics = {
                'request_count': 0,
                'total_routing_time_ms': 0.0,
                'total_validation_time_ms': 0.0,
                'slow_requests': 0,
                'validation_failures': 0,
                'routing_failures': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'security_threats_detected': 0,
                'last_reset_time': datetime.now().isoformat()
            }
            
            self.response_times.clear()
            self.validation_times.clear()
            self.percentile_data.clear()
            self.hourly_stats.clear()
    
    def check_performance_thresholds(self) -> List[str]:
        """Check if performance thresholds are exceeded"""
        warnings = []
        
        with self._lock:
            # Check average response time
            if self.metrics['request_count'] > 0:
                avg_time = self.metrics['total_routing_time_ms'] / self.metrics['request_count']
                if avg_time > self.config.max_routing_time_ms * self.config.performance_alert_threshold:
                    warnings.append(f"Average response time ({avg_time:.2f}ms) exceeds alert threshold")
            
            # Check slow request rate
            if self.metrics['request_count'] > 0:
                slow_rate = self.metrics['slow_requests'] / self.metrics['request_count']
                if slow_rate > 0.1:  # More than 10% slow requests
                    warnings.append(f"High slow request rate: {slow_rate:.2%}")
            
            # Check failure rate
            total_failures = self.metrics['validation_failures'] + self.metrics['routing_failures']
            if self.metrics['request_count'] > 0:
                failure_rate = total_failures / self.metrics['request_count']
                if failure_rate > 0.05:  # More than 5% failure rate
                    warnings.append(f"High failure rate: {failure_rate:.2%}")
            
            # Check resource usage
            if self.resource_usage['cpu_percent'] > self.config.max_cpu_usage_percent:
                warnings.append(f"High CPU usage: {self.resource_usage['cpu_percent']:.1f}%")
            
            memory_usage_mb = self.resource_usage.get('memory_mb', 0)
            if memory_usage_mb > self.config.max_memory_usage_mb:
                warnings.append(f"High memory usage: {memory_usage_mb:.1f}MB")
        
        return warnings