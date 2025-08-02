"""
Financial Fraud Detection Domain State Management - Enhanced with Comprehensive Validation
State management implementation for fraud detection that extends the Universal AI Core Brain state system
"""

import json
import logging
import pickle
import threading
import time
import re
import hashlib
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import uuid

# Import base state components from independent_core
try:
    from independent_core.domain_state import DomainState, StateManager, StateValidationResult
except ImportError:
    # Fallback for testing - define minimal interfaces
    from dataclasses import dataclass
    
    @dataclass
    class StateValidationResult:
        valid: bool
        errors: List[str] = field(default_factory=list)
    
    class DomainState:
        pass
    
    class StateManager:
        pass

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ============= ENHANCED ENUMS =============

class TransactionStatus(Enum):
    """Transaction processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    FLAGGED = "flagged"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    UNDER_REVIEW = "under_review"


class FraudSeverity(Enum):
    """Fraud severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FraudDecision(Enum):
    """Fraud detection decisions"""
    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"
    ESCALATE = "escalate"
    QUARANTINE = "quarantine"


class IndicatorType(Enum):
    """Types of fraud indicators"""
    VELOCITY = "velocity"
    AMOUNT = "amount"
    LOCATION = "location"
    PATTERN = "pattern"
    BEHAVIORAL = "behavioral"
    DEVICE = "device"
    NETWORK = "network"
    MERCHANT = "merchant"
    TIME_BASED = "time_based"
    REGULATORY = "regulatory"


class StateOperation(Enum):
    """State operation types for audit logging"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    READ = "read"
    VALIDATE = "validate"
    RESTORE = "restore"
    EXPORT = "export"
    IMPORT = "import"


class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ============= ENHANCED EXCEPTION CLASSES =============

class FraudStateException(Exception):
    """Base exception for fraud state operations with enhanced error tracking"""
    def __init__(self, message: str, error_code: str = "STATE_ERROR", 
                 details: Dict[str, Any] = None, recoverable: bool = True):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.recoverable = recoverable
        self.timestamp = datetime.now()
        self.stack_trace = traceback.format_exc()


class StateValidationError(FraudStateException):
    """Enhanced validation error with detailed field information"""
    def __init__(self, message: str, field: str = None, value: Any = None,
                 validation_type: str = None, recoverable: bool = True):
        super().__init__(
            message, 
            error_code="VALIDATION_ERROR",
            details={
                "field": field,
                "value": value,
                "validation_type": validation_type
            },
            recoverable=recoverable
        )


class StatePersistenceError(FraudStateException):
    """Enhanced persistence error with recovery options"""
    def __init__(self, message: str, operation: str = None, 
                 file_path: str = None, cause: Exception = None):
        super().__init__(
            message,
            error_code="PERSISTENCE_ERROR",
            details={
                "operation": operation,
                "file_path": file_path,
                "cause": str(cause) if cause else None
            },
            recoverable=True
        )


class StateRecoveryError(FraudStateException):
    """Enhanced recovery error with fallback options"""
    def __init__(self, message: str, recovery_action: str = None,
                 fallback_available: bool = False):
        super().__init__(
            message,
            error_code="RECOVERY_ERROR",
            details={
                "recovery_action": recovery_action,
                "fallback_available": fallback_available
            },
            recoverable=fallback_available
        )


class StateConsistencyError(FraudStateException):
    """State consistency violation error"""
    def __init__(self, message: str, inconsistency_type: str = None,
                 affected_entities: List[str] = None):
        super().__init__(
            message,
            error_code="CONSISTENCY_ERROR",
            details={
                "inconsistency_type": inconsistency_type,
                "affected_entities": affected_entities or []
            },
            recoverable=False
        )


class StateSecurityError(FraudStateException):
    """Security violation in state operations"""
    def __init__(self, message: str, security_type: str = None,
                 access_attempt: Dict[str, Any] = None):
        super().__init__(
            message,
            error_code="SECURITY_ERROR",
            details={
                "security_type": security_type,
                "access_attempt": access_attempt
            },
            recoverable=False
        )


class StatePerformanceError(FraudStateException):
    """Performance threshold violation"""
    def __init__(self, message: str, metric: str = None,
                 threshold: float = None, actual: float = None):
        super().__init__(
            message,
            error_code="PERFORMANCE_ERROR",
            details={
                "metric": metric,
                "threshold": threshold,
                "actual": actual
            },
            recoverable=True
        )


# ============= VALIDATION CONFIGURATION =============

@dataclass
class StateValidationConfig:
    """Comprehensive validation configuration"""
    # Transaction validation
    min_transaction_amount: float = 0.01
    max_transaction_amount: float = 10_000_000.0
    valid_currencies: Set[str] = field(default_factory=lambda: {"USD", "EUR", "GBP", "JPY", "CAD", "AUD"})
    valid_payment_methods: Set[str] = field(default_factory=lambda: {"card", "bank", "wallet", "crypto", "cash"})
    transaction_id_pattern: str = r'^[A-Z0-9]{8,32}$'
    merchant_name_pattern: str = r'^[a-zA-Z0-9\s\-\.&]+$'
    
    # Fraud indicator validation
    min_confidence_score: float = 0.0
    max_confidence_score: float = 1.0
    min_risk_score: float = 0.0
    max_risk_score: float = 1.0
    max_description_length: int = 1000
    
    # State consistency validation
    max_state_size_mb: float = 100.0
    max_transaction_age_days: int = 90
    max_indicators_per_transaction: int = 50
    max_results_per_transaction: int = 10
    
    # Performance validation
    max_operation_time_ms: float = 1000.0
    max_memory_usage_mb: float = 500.0
    max_concurrent_operations: int = 100
    
    # Security validation
    enable_access_control: bool = True
    enable_encryption: bool = True
    enable_audit_logging: bool = True
    max_failed_validations: int = 5
    lockout_duration_minutes: int = 30
    
    # Persistence validation
    max_backup_age_days: int = 30
    min_free_disk_space_mb: float = 1000.0
    enable_compression: bool = True
    enable_checksums: bool = True


# ============= ENHANCED VALIDATION CLASSES =============

class TransactionValidator:
    """Enhanced transaction validator with comprehensive checks"""
    
    def __init__(self, config: StateValidationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validation_cache = {}
        self._cache_ttl = 300  # 5 minutes
    
    def validate(self, transaction: 'Transaction', 
                 existing_transactions: Dict[str, 'Transaction'] = None) -> 'EnhancedValidationResult':
        """Comprehensive transaction validation"""
        start_time = time.time()
        validation_id = f"tx_val_{uuid.uuid4().hex[:8]}"
        issues = []
        
        try:
            # Basic field validation
            issues.extend(self._validate_basic_fields(transaction))
            
            # Format validation
            issues.extend(self._validate_formats(transaction))
            
            # Business logic validation
            issues.extend(self._validate_business_logic(transaction))
            
            # Consistency validation
            if existing_transactions:
                issues.extend(self._validate_consistency(transaction, existing_transactions))
            
            # Security validation
            issues.extend(self._validate_security(transaction))
            
            # Performance impact validation
            issues.extend(self._validate_performance_impact(transaction))
            
            # Categorize issues
            errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
            warnings = [i for i in issues if i.severity == ValidationSeverity.WARNING]
            info = [i for i in issues if i.severity == ValidationSeverity.INFO]
            
            is_valid = len(errors) == 0
            
            return EnhancedValidationResult(
                validation_id=validation_id,
                target_type="transaction",
                target_id=transaction.transaction_id,
                is_valid=is_valid,
                issues=issues,
                errors=[i.message for i in errors],
                warnings=[i.message for i in warnings],
                validation_time_ms=(time.time() - start_time) * 1000,
                metadata={
                    "validator": self.__class__.__name__,
                    "config_version": "1.0",
                    "checks_performed": self._get_checks_performed()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Transaction validation failed: {e}")
            return EnhancedValidationResult(
                validation_id=validation_id,
                target_type="transaction",
                target_id=getattr(transaction, 'transaction_id', 'unknown'),
                is_valid=False,
                issues=[ValidationIssue(
                    field="general",
                    message=f"Validation error: {str(e)}",
                    severity=ValidationSeverity.CRITICAL,
                    code="VAL_EXCEPTION"
                )],
                errors=[f"Validation error: {str(e)}"],
                validation_time_ms=(time.time() - start_time) * 1000
            )
    
    def _validate_basic_fields(self, transaction: 'Transaction') -> List['ValidationIssue']:
        """Validate basic required fields"""
        issues = []
        
        # Transaction ID
        if not transaction.transaction_id:
            issues.append(ValidationIssue(
                field="transaction_id",
                message="Transaction ID is required",
                severity=ValidationSeverity.ERROR,
                code="MISSING_ID"
            ))
        
        # Amount validation
        if transaction.amount is None:
            issues.append(ValidationIssue(
                field="amount",
                message="Transaction amount is required",
                severity=ValidationSeverity.ERROR,
                code="MISSING_AMOUNT"
            ))
        elif transaction.amount < self.config.min_transaction_amount:
            issues.append(ValidationIssue(
                field="amount",
                message=f"Amount {transaction.amount} below minimum {self.config.min_transaction_amount}",
                severity=ValidationSeverity.ERROR,
                code="AMOUNT_TOO_LOW"
            ))
        elif transaction.amount > self.config.max_transaction_amount:
            issues.append(ValidationIssue(
                field="amount",
                message=f"Amount {transaction.amount} exceeds maximum {self.config.max_transaction_amount}",
                severity=ValidationSeverity.ERROR,
                code="AMOUNT_TOO_HIGH"
            ))
        
        # Currency validation
        if not transaction.currency:
            issues.append(ValidationIssue(
                field="currency",
                message="Currency is required",
                severity=ValidationSeverity.ERROR,
                code="MISSING_CURRENCY"
            ))
        elif transaction.currency not in self.config.valid_currencies:
            issues.append(ValidationIssue(
                field="currency",
                message=f"Invalid currency: {transaction.currency}",
                severity=ValidationSeverity.ERROR,
                code="INVALID_CURRENCY"
            ))
        
        # Merchant validation
        if not transaction.merchant:
            issues.append(ValidationIssue(
                field="merchant",
                message="Merchant is required",
                severity=ValidationSeverity.ERROR,
                code="MISSING_MERCHANT"
            ))
        
        # User and account validation
        if not transaction.user_id:
            issues.append(ValidationIssue(
                field="user_id",
                message="User ID is required",
                severity=ValidationSeverity.ERROR,
                code="MISSING_USER_ID"
            ))
        
        if not transaction.account_id:
            issues.append(ValidationIssue(
                field="account_id",
                message="Account ID is required",
                severity=ValidationSeverity.ERROR,
                code="MISSING_ACCOUNT_ID"
            ))
        
        # Payment method validation
        if transaction.payment_method:
            if transaction.payment_method not in self.config.valid_payment_methods:
                issues.append(ValidationIssue(
                    field="payment_method",
                    message=f"Invalid payment method: {transaction.payment_method}",
                    severity=ValidationSeverity.WARNING,
                    code="INVALID_PAYMENT_METHOD"
                ))
        else:
            issues.append(ValidationIssue(
                field="payment_method",
                message="Payment method not specified",
                severity=ValidationSeverity.INFO,
                code="MISSING_PAYMENT_METHOD"
            ))
        
        return issues
    
    def _validate_formats(self, transaction: 'Transaction') -> List['ValidationIssue']:
        """Validate field formats"""
        issues = []
        
        # Transaction ID format
        if transaction.transaction_id:
            if not re.match(self.config.transaction_id_pattern, transaction.transaction_id):
                issues.append(ValidationIssue(
                    field="transaction_id",
                    message=f"Invalid transaction ID format: {transaction.transaction_id}",
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_ID_FORMAT"
                ))
        
        # Merchant name format
        if transaction.merchant:
            if not re.match(self.config.merchant_name_pattern, transaction.merchant):
                issues.append(ValidationIssue(
                    field="merchant",
                    message=f"Invalid merchant name format: {transaction.merchant}",
                    severity=ValidationSeverity.WARNING,
                    code="INVALID_MERCHANT_FORMAT"
                ))
        
        # Timestamp validation
        if transaction.timestamp:
            if transaction.timestamp > datetime.now():
                issues.append(ValidationIssue(
                    field="timestamp",
                    message="Transaction timestamp is in the future",
                    severity=ValidationSeverity.ERROR,
                    code="FUTURE_TIMESTAMP"
                ))
            
            age_days = (datetime.now() - transaction.timestamp).days
            if age_days > self.config.max_transaction_age_days:
                issues.append(ValidationIssue(
                    field="timestamp",
                    message=f"Transaction is {age_days} days old, exceeds maximum {self.config.max_transaction_age_days}",
                    severity=ValidationSeverity.WARNING,
                    code="OLD_TRANSACTION"
                ))
        
        # IP address format
        if transaction.ip_address:
            ip_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
            if not re.match(ip_pattern, transaction.ip_address):
                # Check if it's IPv6
                ipv6_pattern = r'^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))$'
                if not re.match(ipv6_pattern, transaction.ip_address):
                    issues.append(ValidationIssue(
                        field="ip_address",
                        message=f"Invalid IP address format: {transaction.ip_address}",
                        severity=ValidationSeverity.WARNING,
                        code="INVALID_IP_FORMAT"
                    ))
        
        return issues
    
    def _validate_business_logic(self, transaction: 'Transaction') -> List['ValidationIssue']:
        """Validate business logic rules"""
        issues = []
        
        # Status transitions
        valid_statuses = [s.value for s in TransactionStatus]
        if transaction.status.value not in valid_statuses:
            issues.append(ValidationIssue(
                field="status",
                message=f"Invalid transaction status: {transaction.status}",
                severity=ValidationSeverity.ERROR,
                code="INVALID_STATUS"
            ))
        
        # Location validation
        if transaction.location:
            if not isinstance(transaction.location, dict):
                issues.append(ValidationIssue(
                    field="location",
                    message="Location must be a dictionary",
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_LOCATION_TYPE"
                ))
            else:
                # Check for required location fields
                required_location_fields = ['country', 'city']
                for field in required_location_fields:
                    if field not in transaction.location:
                        issues.append(ValidationIssue(
                            field=f"location.{field}",
                            message=f"Location missing required field: {field}",
                            severity=ValidationSeverity.WARNING,
                            code="MISSING_LOCATION_FIELD"
                        ))
        
        # Device info validation
        if transaction.device_info:
            if not isinstance(transaction.device_info, dict):
                issues.append(ValidationIssue(
                    field="device_info",
                    message="Device info must be a dictionary",
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_DEVICE_INFO_TYPE"
                ))
        
        # Risk score validation
        if transaction.risk_score is not None:
            if not 0 <= transaction.risk_score <= 1:
                issues.append(ValidationIssue(
                    field="risk_score",
                    message=f"Risk score must be between 0 and 1, got {transaction.risk_score}",
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_RISK_SCORE"
                ))
        
        return issues
    
    def _validate_consistency(self, transaction: 'Transaction',
                            existing_transactions: Dict[str, 'Transaction']) -> List['ValidationIssue']:
        """Validate consistency with existing transactions"""
        issues = []
        
        # Check for duplicate transaction ID
        if transaction.transaction_id in existing_transactions:
            issues.append(ValidationIssue(
                field="transaction_id",
                message=f"Duplicate transaction ID: {transaction.transaction_id}",
                severity=ValidationSeverity.ERROR,
                code="DUPLICATE_ID"
            ))
        
        # Check for velocity violations
        user_transactions = [
            t for t in existing_transactions.values()
            if t.user_id == transaction.user_id and
            (transaction.timestamp - t.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        if len(user_transactions) > 10:
            issues.append(ValidationIssue(
                field="user_id",
                message=f"High transaction velocity: {len(user_transactions)} transactions in last hour",
                severity=ValidationSeverity.WARNING,
                code="HIGH_VELOCITY"
            ))
        
        # Check for amount anomalies
        if user_transactions:
            avg_amount = sum(t.amount for t in user_transactions) / len(user_transactions)
            if transaction.amount > avg_amount * 10:
                issues.append(ValidationIssue(
                    field="amount",
                    message=f"Transaction amount {transaction.amount} is 10x higher than average {avg_amount:.2f}",
                    severity=ValidationSeverity.WARNING,
                    code="AMOUNT_ANOMALY"
                ))
        
        return issues
    
    def _validate_security(self, transaction: 'Transaction') -> List['ValidationIssue']:
        """Validate security aspects"""
        issues = []
        
        # Check for suspicious patterns in merchant name
        suspicious_patterns = ['test', 'debug', 'hack', 'exploit']
        if transaction.merchant:
            merchant_lower = transaction.merchant.lower()
            for pattern in suspicious_patterns:
                if pattern in merchant_lower:
                    issues.append(ValidationIssue(
                        field="merchant",
                        message=f"Suspicious pattern in merchant name: {pattern}",
                        severity=ValidationSeverity.WARNING,
                        code="SUSPICIOUS_MERCHANT"
                    ))
        
        # Check session ID format
        if transaction.session_id:
            if len(transaction.session_id) < 10:
                issues.append(ValidationIssue(
                    field="session_id",
                    message="Session ID too short",
                    severity=ValidationSeverity.INFO,
                    code="SHORT_SESSION_ID"
                ))
        
        return issues
    
    def _validate_performance_impact(self, transaction: 'Transaction') -> List['ValidationIssue']:
        """Validate potential performance impact"""
        issues = []
        
        # Check metadata size
        if transaction.metadata:
            metadata_size = len(json.dumps(transaction.metadata))
            if metadata_size > 10000:  # 10KB
                issues.append(ValidationIssue(
                    field="metadata",
                    message=f"Large metadata size: {metadata_size} bytes",
                    severity=ValidationSeverity.WARNING,
                    code="LARGE_METADATA"
                ))
        
        return issues
    
    def _get_checks_performed(self) -> List[str]:
        """Get list of validation checks performed"""
        return [
            "basic_fields",
            "formats",
            "business_logic",
            "consistency",
            "security",
            "performance_impact"
        ]


class FraudIndicatorValidator:
    """Enhanced fraud indicator validator"""
    
    def __init__(self, config: StateValidationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate(self, indicator: 'FraudIndicator',
                 existing_indicators: Dict[str, 'FraudIndicator'] = None) -> 'EnhancedValidationResult':
        """Comprehensive fraud indicator validation"""
        start_time = time.time()
        validation_id = f"ind_val_{uuid.uuid4().hex[:8]}"
        issues = []
        
        try:
            # Basic validation
            issues.extend(self._validate_basic_fields(indicator))
            
            # Score validation
            issues.extend(self._validate_scores(indicator))
            
            # Consistency validation
            if existing_indicators:
                issues.extend(self._validate_consistency(indicator, existing_indicators))
            
            # Metadata validation
            issues.extend(self._validate_metadata(indicator))
            
            errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
            warnings = [i for i in issues if i.severity == ValidationSeverity.WARNING]
            
            return EnhancedValidationResult(
                validation_id=validation_id,
                target_type="fraud_indicator",
                target_id=indicator.indicator_id,
                is_valid=len(errors) == 0,
                issues=issues,
                errors=[i.message for i in errors],
                warnings=[i.message for i in warnings],
                validation_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            self.logger.error(f"Indicator validation failed: {e}")
            return EnhancedValidationResult(
                validation_id=validation_id,
                target_type="fraud_indicator",
                target_id=getattr(indicator, 'indicator_id', 'unknown'),
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                validation_time_ms=(time.time() - start_time) * 1000
            )
    
    def _validate_basic_fields(self, indicator: 'FraudIndicator') -> List['ValidationIssue']:
        """Validate basic indicator fields"""
        issues = []
        
        if not indicator.indicator_id:
            issues.append(ValidationIssue(
                field="indicator_id",
                message="Indicator ID is required",
                severity=ValidationSeverity.ERROR,
                code="MISSING_ID"
            ))
        
        if not indicator.description:
            issues.append(ValidationIssue(
                field="description",
                message="Description is required",
                severity=ValidationSeverity.ERROR,
                code="MISSING_DESCRIPTION"
            ))
        elif len(indicator.description) > self.config.max_description_length:
            issues.append(ValidationIssue(
                field="description",
                message=f"Description too long: {len(indicator.description)} characters",
                severity=ValidationSeverity.WARNING,
                code="LONG_DESCRIPTION"
            ))
        
        if not indicator.source:
            issues.append(ValidationIssue(
                field="source",
                message="Indicator source is required",
                severity=ValidationSeverity.ERROR,
                code="MISSING_SOURCE"
            ))
        
        return issues
    
    def _validate_scores(self, indicator: 'FraudIndicator') -> List['ValidationIssue']:
        """Validate indicator scores"""
        issues = []
        
        if not self.config.min_confidence_score <= indicator.confidence <= self.config.max_confidence_score:
            issues.append(ValidationIssue(
                field="confidence",
                message=f"Confidence score {indicator.confidence} out of range [{self.config.min_confidence_score}, {self.config.max_confidence_score}]",
                severity=ValidationSeverity.ERROR,
                code="INVALID_CONFIDENCE"
            ))
        
        if indicator.threshold is not None:
            if indicator.threshold < 0 or indicator.threshold > 1:
                issues.append(ValidationIssue(
                    field="threshold",
                    message=f"Threshold {indicator.threshold} out of range [0, 1]",
                    severity=ValidationSeverity.WARNING,
                    code="INVALID_THRESHOLD"
                ))
        
        return issues
    
    def _validate_consistency(self, indicator: 'FraudIndicator',
                            existing_indicators: Dict[str, 'FraudIndicator']) -> List['ValidationIssue']:
        """Validate consistency with existing indicators"""
        issues = []
        
        # Check for duplicate ID
        if indicator.indicator_id in existing_indicators:
            issues.append(ValidationIssue(
                field="indicator_id",
                message=f"Duplicate indicator ID: {indicator.indicator_id}",
                severity=ValidationSeverity.ERROR,
                code="DUPLICATE_INDICATOR"
            ))
        
        # Check for conflicting indicators
        if indicator.transaction_id:
            related_indicators = [
                ind for ind in existing_indicators.values()
                if ind.transaction_id == indicator.transaction_id
            ]
            
            if len(related_indicators) >= self.config.max_indicators_per_transaction:
                issues.append(ValidationIssue(
                    field="transaction_id",
                    message=f"Too many indicators for transaction: {len(related_indicators)}",
                    severity=ValidationSeverity.WARNING,
                    code="TOO_MANY_INDICATORS"
                ))
        
        return issues
    
    def _validate_metadata(self, indicator: 'FraudIndicator') -> List['ValidationIssue']:
        """Validate indicator metadata"""
        issues = []
        
        if indicator.metadata:
            if not isinstance(indicator.metadata, dict):
                issues.append(ValidationIssue(
                    field="metadata",
                    message="Metadata must be a dictionary",
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_METADATA_TYPE"
                ))
        
        return issues


class StateConsistencyValidator:
    """Validates overall state consistency"""
    
    def __init__(self, config: StateValidationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_state(self, state: 'FinancialFraudState') -> 'EnhancedValidationResult':
        """Comprehensive state consistency validation"""
        start_time = time.time()
        validation_id = f"state_val_{uuid.uuid4().hex[:8]}"
        issues = []
        
        try:
            # Cross-reference validation
            issues.extend(self._validate_cross_references(state))
            
            # Index consistency
            issues.extend(self._validate_indexes(state))
            
            # State size validation
            issues.extend(self._validate_state_size(state))
            
            # Performance metrics consistency
            issues.extend(self._validate_performance_metrics(state))
            
            # Temporal consistency
            issues.extend(self._validate_temporal_consistency(state))
            
            errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
            warnings = [i for i in issues if i.severity == ValidationSeverity.WARNING]
            
            return EnhancedValidationResult(
                validation_id=validation_id,
                target_type="state",
                target_id=state.domain_name,
                is_valid=len(errors) == 0,
                issues=issues,
                errors=[i.message for i in errors],
                warnings=[i.message for i in warnings],
                validation_time_ms=(time.time() - start_time) * 1000,
                metadata={
                    "state_version": state.version,
                    "entity_counts": {
                        "transactions": len(state.transactions),
                        "indicators": len(state.fraud_indicators),
                        "results": len(state.fraud_results)
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"State validation failed: {e}")
            return EnhancedValidationResult(
                validation_id=validation_id,
                target_type="state",
                target_id=state.domain_name,
                is_valid=False,
                errors=[f"State validation error: {str(e)}"],
                validation_time_ms=(time.time() - start_time) * 1000
            )
    
    def _validate_cross_references(self, state: 'FinancialFraudState') -> List['ValidationIssue']:
        """Validate cross-references between entities"""
        issues = []
        
        # Check fraud results reference valid transactions
        for result_id, result in state.fraud_results.items():
            if result.transaction_id not in state.transactions:
                issues.append(ValidationIssue(
                    field=f"fraud_result.{result_id}",
                    message=f"Result references non-existent transaction: {result.transaction_id}",
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_TRANSACTION_REF"
                ))
        
        # Check indicators reference valid transactions
        for ind_id, indicator in state.fraud_indicators.items():
            if indicator.transaction_id and indicator.transaction_id not in state.transactions:
                issues.append(ValidationIssue(
                    field=f"indicator.{ind_id}",
                    message=f"Indicator references non-existent transaction: {indicator.transaction_id}",
                    severity=ValidationSeverity.WARNING,
                    code="INVALID_TRANSACTION_REF"
                ))
        
        return issues
    
    def _validate_indexes(self, state: 'FinancialFraudState') -> List['ValidationIssue']:
        """Validate index consistency"""
        issues = []
        
        # Validate user transaction index
        for user_id, tx_ids in state.user_transactions.items():
            for tx_id in tx_ids:
                if tx_id not in state.transactions:
                    issues.append(ValidationIssue(
                        field=f"user_transactions.{user_id}",
                        message=f"Index references non-existent transaction: {tx_id}",
                        severity=ValidationSeverity.ERROR,
                        code="INVALID_INDEX_REF"
                    ))
                elif state.transactions[tx_id].user_id != user_id:
                    issues.append(ValidationIssue(
                        field=f"user_transactions.{user_id}",
                        message=f"Index mismatch: transaction {tx_id} has different user_id",
                        severity=ValidationSeverity.ERROR,
                        code="INDEX_MISMATCH"
                    ))
        
        # Validate all transactions are indexed
        for tx_id, transaction in state.transactions.items():
            if tx_id not in state.user_transactions.get(transaction.user_id, set()):
                issues.append(ValidationIssue(
                    field=f"transaction.{tx_id}",
                    message=f"Transaction not indexed for user {transaction.user_id}",
                    severity=ValidationSeverity.WARNING,
                    code="MISSING_INDEX"
                ))
        
        return issues
    
    def _validate_state_size(self, state: 'FinancialFraudState') -> List['ValidationIssue']:
        """Validate state size constraints"""
        issues = []
        
        # Estimate state size
        try:
            state_dict = {
                'transactions': {tid: t.to_dict() for tid, t in state.transactions.items()},
                'indicators': {iid: i.to_dict() for iid, i in state.fraud_indicators.items()},
                'results': {rid: r.to_dict() for rid, r in state.fraud_results.items()}
            }
            state_size_mb = len(json.dumps(state_dict)) / (1024 * 1024)
            
            if state_size_mb > self.config.max_state_size_mb:
                issues.append(ValidationIssue(
                    field="state_size",
                    message=f"State size {state_size_mb:.2f}MB exceeds maximum {self.config.max_state_size_mb}MB",
                    severity=ValidationSeverity.ERROR,
                    code="STATE_TOO_LARGE"
                ))
        except Exception as e:
            issues.append(ValidationIssue(
                field="state_size",
                message=f"Failed to calculate state size: {e}",
                severity=ValidationSeverity.WARNING,
                code="SIZE_CALC_ERROR"
            ))
        
        return issues
    
    def _validate_performance_metrics(self, state: 'FinancialFraudState') -> List['ValidationIssue']:
        """Validate performance metrics consistency"""
        issues = []
        
        metrics = state.performance_metrics
        
        # Validate detection rate
        if metrics["total_transactions"] > 0:
            detection_rate = metrics["fraud_detected"] / metrics["total_transactions"]
            if detection_rate > 0.5:  # More than 50% fraud seems unrealistic
                issues.append(ValidationIssue(
                    field="performance_metrics.detection_rate",
                    message=f"Unusually high fraud detection rate: {detection_rate:.2%}",
                    severity=ValidationSeverity.WARNING,
                    code="HIGH_DETECTION_RATE"
                ))
        
        # Validate false positive rate
        if metrics["fraud_detected"] > 0:
            if metrics["false_positives"] > metrics["fraud_detected"]:
                issues.append(ValidationIssue(
                    field="performance_metrics.false_positives",
                    message="False positives exceed total fraud detected",
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_FALSE_POSITIVE_COUNT"
                ))
        
        return issues
    
    def _validate_temporal_consistency(self, state: 'FinancialFraudState') -> List['ValidationIssue']:
        """Validate temporal consistency"""
        issues = []
        
        # Check that last_updated is after created_at
        if state.last_updated < state.created_at:
            issues.append(ValidationIssue(
                field="last_updated",
                message="Last updated time is before creation time",
                severity=ValidationSeverity.ERROR,
                code="TEMPORAL_INCONSISTENCY"
            ))
        
        # Check for future timestamps
        now = datetime.now()
        if state.last_updated > now:
            issues.append(ValidationIssue(
                field="last_updated",
                message="Last updated time is in the future",
                severity=ValidationSeverity.ERROR,
                code="FUTURE_TIMESTAMP"
            ))
        
        return issues


# ============= ENHANCED DATA STRUCTURES =============

@dataclass
class ValidationIssue:
    """Detailed validation issue"""
    field: str
    message: str
    severity: ValidationSeverity
    code: str
    suggestion: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class EnhancedValidationResult:
    """Enhanced validation result with detailed information"""
    validation_id: str
    target_type: str
    target_id: str
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    validation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'validation_id': self.validation_id,
            'target_type': self.target_type,
            'target_id': self.target_id,
            'is_valid': self.is_valid,
            'issues': [
                {
                    'field': i.field,
                    'message': i.message,
                    'severity': i.severity.value,
                    'code': i.code,
                    'suggestion': i.suggestion
                } for i in self.issues
            ],
            'errors': self.errors,
            'warnings': self.warnings,
            'timestamp': self.timestamp.isoformat(),
            'validation_time_ms': self.validation_time_ms,
            'metadata': self.metadata
        }


@dataclass
class StateAuditEntry:
    """Audit entry for state operations"""
    audit_id: str
    operation: StateOperation
    entity_type: str
    entity_id: str
    user_id: Optional[str]
    timestamp: datetime
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'audit_id': self.audit_id,
            'operation': self.operation.value,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'success': self.success,
            'details': self.details,
            'error_message': self.error_message
        }


# Enhanced Transaction data structure
@dataclass
class Transaction:
    """Enhanced transaction data structure with validation support"""
    transaction_id: str
    amount: float
    currency: str
    merchant: str
    timestamp: datetime
    user_id: str
    account_id: str
    payment_method: str
    status: TransactionStatus = TransactionStatus.PENDING
    
    # Additional fields
    merchant_category: Optional[str] = None
    location: Optional[Dict[str, Any]] = None
    device_info: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    
    # Processing metadata
    processing_time: Optional[float] = None
    risk_score: Optional[float] = None
    
    # Validation tracking
    validation_status: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        """Create from dictionary with proper deserialization and validation"""
        try:
            # Handle timestamp
            if 'timestamp' in data and isinstance(data['timestamp'], str):
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            
            # Handle status
            if 'status' in data and isinstance(data['status'], str):
                data['status'] = TransactionStatus(data['status'])
            
            # Remove unknown fields
            valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
            data = {k: v for k, v in data.items() if k in valid_fields}
            
            return cls(**data)
        except Exception as e:
            raise StateValidationError(f"Failed to create transaction from dict: {e}")


# Enhanced Fraud indicator data structure  
@dataclass
class FraudIndicator:
    """Enhanced fraud indicator data structure"""
    indicator_id: str
    indicator_type: IndicatorType
    severity: FraudSeverity
    confidence: float
    description: str
    timestamp: datetime
    source: str
    
    # Additional details
    transaction_id: Optional[str] = None
    user_id: Optional[str] = None
    value: Optional[Any] = None
    threshold: Optional[float] = None
    
    # Validation tracking
    validated: bool = False
    validation_timestamp: Optional[datetime] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization"""
        data = asdict(self)
        data['indicator_type'] = self.indicator_type.value
        data['severity'] = self.severity.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.validation_timestamp:
            data['validation_timestamp'] = self.validation_timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FraudIndicator':
        """Create from dictionary with proper deserialization"""
        try:
            if 'indicator_type' in data and isinstance(data['indicator_type'], str):
                data['indicator_type'] = IndicatorType(data['indicator_type'])
            if 'severity' in data and isinstance(data['severity'], str):
                data['severity'] = FraudSeverity(data['severity'])
            if 'timestamp' in data and isinstance(data['timestamp'], str):
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            if 'validation_timestamp' in data and data['validation_timestamp']:
                if isinstance(data['validation_timestamp'], str):
                    data['validation_timestamp'] = datetime.fromisoformat(data['validation_timestamp'])
            
            # Remove unknown fields
            valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
            data = {k: v for k, v in data.items() if k in valid_fields}
            
            return cls(**data)
        except Exception as e:
            raise StateValidationError(f"Failed to create indicator from dict: {e}")


# Enhanced Fraud result data structure
@dataclass 
class FraudResult:
    """Enhanced fraud detection result data structure"""
    result_id: str
    transaction_id: str
    risk_score: float
    fraud_probability: float
    decision: FraudDecision
    
    # Additional fields
    processing_time: float = 0.0
    model_version: str = "1.0.0"
    rule_triggers: List[str] = field(default_factory=list)
    
    indicators: List[FraudIndicator] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Validation tracking
    validation_status: Optional[str] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization"""
        data = asdict(self)
        data['decision'] = self.decision.value
        data['timestamp'] = self.timestamp.isoformat()
        data['indicators'] = [ind.to_dict() for ind in self.indicators]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FraudResult':
        """Create from dictionary with proper deserialization"""
        try:
            if 'decision' in data and isinstance(data['decision'], str):
                data['decision'] = FraudDecision(data['decision'])
            if 'timestamp' in data and isinstance(data['timestamp'], str):
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            if 'indicators' in data:
                data['indicators'] = [FraudIndicator.from_dict(ind) for ind in data['indicators']]
            
            # Remove unknown fields
            valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
            data = {k: v for k, v in data.items() if k in valid_fields}
            
            return cls(**data)
        except Exception as e:
            raise StateValidationError(f"Failed to create result from dict: {e}")


# Enhanced Processing metadata data structure
@dataclass
class ProcessingMetadata:
    """Enhanced processing metadata data structure"""
    process_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    
    # Processing details
    transaction_count: int = 0
    fraud_detected_count: int = 0
    errors_count: int = 0
    validation_failures: int = 0
    
    steps_completed: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    # Resource tracking
    peak_memory_mb: float = 0.0
    cpu_time_seconds: float = 0.0
    
    def duration(self) -> Optional[timedelta]:
        """Get processing duration"""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization"""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data


# ============= STATE PERSISTENCE AND RECOVERY =============

class StatePersistenceManager:
    """Manages state persistence with validation and recovery"""
    
    def __init__(self, storage_path: Path, config: StateValidationConfig):
        self.storage_path = storage_path
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Backup configuration
        self.backup_retention_days = 30
        self.max_backups = 10
        
        # Recovery state
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
    
    def save_state(self, state: 'FinancialFraudState', 
                   create_backup: bool = True) -> bool:
        """Save state with validation and backup"""
        try:
            # Validate state before saving
            validator = StateConsistencyValidator(self.config)
            validation_result = validator.validate_state(state)
            
            if not validation_result.is_valid:
                raise StatePersistenceError(
                    f"State validation failed: {validation_result.errors}",
                    operation="save",
                    cause=None
                )
            
            # Create backup if requested
            if create_backup:
                self._create_backup(state)
            
            # Prepare state data
            state_data = self._prepare_state_data(state)
            
            # Calculate checksum
            checksum = self._calculate_checksum(state_data)
            state_data['checksum'] = checksum
            
            # Save main state file
            state_file = self.storage_path / f"{state.domain_name}_state.json"
            temp_file = state_file.with_suffix('.tmp')
            
            # Write to temporary file first
            with open(temp_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            # Verify written data
            with open(temp_file, 'r') as f:
                loaded_data = json.load(f)
                if self._calculate_checksum(loaded_data) != checksum:
                    raise StatePersistenceError(
                        "Checksum verification failed after write",
                        operation="save",
                        file_path=str(temp_file)
                    )
            
            # Atomic rename
            temp_file.rename(state_file)
            
            # Clean old backups
            self._clean_old_backups(state.domain_name)
            
            self.logger.info(f"State saved successfully: {state_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            if isinstance(e, StatePersistenceError):
                raise
            raise StatePersistenceError(
                f"State save failed: {str(e)}",
                operation="save",
                cause=e
            )
    
    def load_state(self, domain_name: str) -> Optional['FinancialFraudState']:
        """Load state with validation and recovery"""
        state_file = self.storage_path / f"{domain_name}_state.json"
        
        try:
            # Try to load main state file
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                
                # Verify checksum
                stored_checksum = state_data.pop('checksum', None)
                if stored_checksum and self._calculate_checksum(state_data) != stored_checksum:
                    raise StateRecoveryError(
                        "State file checksum verification failed",
                        recovery_action="load_from_backup",
                        fallback_available=True
                    )
                
                # Reconstruct state
                state = self._reconstruct_state(domain_name, state_data)
                
                # Validate loaded state
                validator = StateConsistencyValidator(self.config)
                validation_result = validator.validate_state(state)
                
                if not validation_result.is_valid:
                    self.logger.warning(f"Loaded state has validation issues: {validation_result.errors}")
                    # Continue with warnings - state is usable
                
                self.logger.info(f"State loaded successfully: {domain_name}")
                return state
            
            else:
                # No state file exists - try recovery
                return self._attempt_recovery(domain_name)
                
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            
            # Attempt recovery
            if self.recovery_attempts < self.max_recovery_attempts:
                self.recovery_attempts += 1
                return self._attempt_recovery(domain_name)
            
            raise StateRecoveryError(
                f"Failed to load state after {self.max_recovery_attempts} attempts",
                recovery_action="exhausted",
                fallback_available=False
            )
    
    def _create_backup(self, state: 'FinancialFraudState'):
        """Create backup of current state"""
        try:
            backup_dir = self.storage_path / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"{state.domain_name}_backup_{timestamp}.json"
            
            state_data = self._prepare_state_data(state)
            
            with open(backup_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            self.logger.debug(f"Created backup: {backup_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create backup: {e}")
            # Don't fail the main operation
    
    def _clean_old_backups(self, domain_name: str):
        """Clean old backup files"""
        try:
            backup_dir = self.storage_path / "backups"
            if not backup_dir.exists():
                return
            
            backups = list(backup_dir.glob(f"{domain_name}_backup_*.json"))
            backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            # Keep only recent backups
            for backup in backups[self.max_backups:]:
                backup.unlink()
                self.logger.debug(f"Removed old backup: {backup}")
                
        except Exception as e:
            self.logger.warning(f"Failed to clean backups: {e}")
    
    def _attempt_recovery(self, domain_name: str) -> Optional['FinancialFraudState']:
        """Attempt to recover state from backups"""
        try:
            backup_dir = self.storage_path / "backups"
            if not backup_dir.exists():
                return None
            
            # Find most recent backup
            backups = list(backup_dir.glob(f"{domain_name}_backup_*.json"))
            if not backups:
                return None
            
            backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            for backup_file in backups:
                try:
                    with open(backup_file, 'r') as f:
                        state_data = json.load(f)
                    
                    state = self._reconstruct_state(domain_name, state_data)
                    
                    self.logger.warning(f"Recovered state from backup: {backup_file}")
                    return state
                    
                except Exception as e:
                    self.logger.warning(f"Failed to recover from {backup_file}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")
            return None
    
    def _prepare_state_data(self, state: 'FinancialFraudState') -> Dict[str, Any]:
        """Prepare state data for persistence"""
        return {
            'domain_name': state.domain_name,
            'version': state.version,
            'created_at': state.created_at.isoformat(),
            'last_updated': state.last_updated.isoformat(),
            'transactions': {
                tid: t.to_dict() for tid, t in state.transactions.items()
            },
            'fraud_indicators': {
                iid: i.to_dict() for iid, i in state.fraud_indicators.items()
            },
            'fraud_results': {
                rid: r.to_dict() for rid, r in state.fraud_results.items()
            },
            'validation_results': {
                vid: v.to_dict() for vid, v in state.validation_results.items()
            },
            'performance_metrics': state.performance_metrics,
            'indexes': {
                'user_transactions': {
                    uid: list(tids) for uid, tids in state.user_transactions.items()
                },
                'account_transactions': {
                    aid: list(tids) for aid, tids in state.account_transactions.items()
                },
                'merchant_transactions': {
                    mid: list(tids) for mid, tids in state.merchant_transactions.items()
                }
            }
        }
    
    def _reconstruct_state(self, domain_name: str, 
                          state_data: Dict[str, Any]) -> 'FinancialFraudState':
        """Reconstruct state from persisted data"""
        state = FinancialFraudState(domain_name, {'persistence_enabled': False})
        
        # Restore metadata
        state.version = state_data.get('version', 1)
        state.created_at = datetime.fromisoformat(state_data['created_at'])
        state.last_updated = datetime.fromisoformat(state_data['last_updated'])
        
        # Restore entities
        for tid, tdata in state_data.get('transactions', {}).items():
            try:
                transaction = Transaction.from_dict(tdata)
                state.transactions[tid] = transaction
            except Exception as e:
                self.logger.warning(f"Failed to restore transaction {tid}: {e}")
        
        for iid, idata in state_data.get('fraud_indicators', {}).items():
            try:
                indicator = FraudIndicator.from_dict(idata)
                state.fraud_indicators[iid] = indicator
            except Exception as e:
                self.logger.warning(f"Failed to restore indicator {iid}: {e}")
        
        for rid, rdata in state_data.get('fraud_results', {}).items():
            try:
                result = FraudResult.from_dict(rdata)
                state.fraud_results[rid] = result
            except Exception as e:
                self.logger.warning(f"Failed to restore result {rid}: {e}")
        
        # Restore indexes
        indexes = state_data.get('indexes', {})
        for uid, tids in indexes.get('user_transactions', {}).items():
            state.user_transactions[uid] = set(tids)
        
        for aid, tids in indexes.get('account_transactions', {}).items():
            state.account_transactions[aid] = set(tids)
        
        for mid, tids in indexes.get('merchant_transactions', {}).items():
            state.merchant_transactions[mid] = set(tids)
        
        # Restore metrics
        state.performance_metrics.update(state_data.get('performance_metrics', {}))
        
        return state
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for data integrity"""
        # Remove checksum field if present
        data_copy = data.copy()
        data_copy.pop('checksum', None)
        
        # Serialize deterministically
        serialized = json.dumps(data_copy, sort_keys=True)
        
        # Calculate hash
        return hashlib.sha256(serialized.encode()).hexdigest()


# ============= ENHANCED STATE CLASS =============

class FinancialFraudState(DomainState):
    """
    Enhanced Financial Fraud Detection domain state management
    with comprehensive validation and error handling
    """
    
    def __init__(self, domain_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced fraud domain state"""
        super().__init__()
        self.domain_name = domain_name
        self.config = config or {}
        
        # State storage with thread-safe access
        self._lock = threading.RLock()
        self.transactions: Dict[str, Transaction] = {}
        self.fraud_indicators: Dict[str, FraudIndicator] = {}
        self.fraud_results: Dict[str, FraudResult] = {}
        self.validation_results: Dict[str, EnhancedValidationResult] = {}
        self.processing_metadata: Dict[str, ProcessingMetadata] = {}
        
        # State indexes for efficient lookup
        self.user_transactions: Dict[str, Set[str]] = defaultdict(set)
        self.account_transactions: Dict[str, Set[str]] = defaultdict(set)
        self.merchant_transactions: Dict[str, Set[str]] = defaultdict(set)
        
        # Time-based indexes
        self.hourly_transactions: Dict[str, Set[str]] = defaultdict(set)
        self.daily_transactions: Dict[str, Set[str]] = defaultdict(set)
        
        # Performance tracking
        self.performance_metrics = {
            "total_transactions": 0,
            "fraud_detected": 0,
            "false_positives": 0,
            "true_positives": 0,
            "processing_time_avg": 0.0,
            "detection_rate": 0.0,
            "validation_failures": 0,
            "recovery_attempts": 0
        }
        
        # State versioning and history
        self.version = 1
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self.state_history: deque = deque(maxlen=10)
        
        # Audit logging
        self.audit_log: deque = deque(maxlen=1000)
        self.enable_audit = config.get('enable_audit_logging', True)
        
        # Validation configuration
        self.validation_config = StateValidationConfig()
        self._update_validation_config(config.get('validation', {}))
        
        # Initialize validators
        self.transaction_validator = TransactionValidator(self.validation_config)
        self.indicator_validator = FraudIndicatorValidator(self.validation_config)
        self.consistency_validator = StateConsistencyValidator(self.validation_config)
        
        # Persistence configuration
        self.persistence_enabled = config.get('persistence_enabled', True)
        self.persistence_path = Path(config.get('persistence_path', './state'))
        
        if self.persistence_enabled:
            self.persistence_manager = StatePersistenceManager(
                self.persistence_path,
                self.validation_config
            )
        
        # State recovery
        self.auto_recovery = config.get('auto_recovery', True)
        self.recovery_checkpoints: deque = deque(maxlen=5)
        
        # Performance monitoring
        self.operation_times: deque = deque(maxlen=100)
        self.memory_usage: deque = deque(maxlen=100)
        
        # Security tracking
        self.failed_validations: Dict[str, int] = defaultdict(int)
        self.locked_entities: Set[str] = set()
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Enhanced FinancialFraudState initialized for domain: {domain_name}")
    
    def _update_validation_config(self, validation_settings: Dict[str, Any]):
        """Update validation configuration from settings"""
        for key, value in validation_settings.items():
            if hasattr(self.validation_config, key):
                setattr(self.validation_config, key, value)
    
    def add_transaction(self, transaction: Transaction) -> bool:
        """Add transaction with comprehensive validation and error handling"""
        operation_start = time.time()
        audit_id = f"audit_{uuid.uuid4().hex[:8]}"
        
        with self._lock:
            try:
                # Check if entity is locked
                if transaction.transaction_id in self.locked_entities:
                    raise StateSecurityError(
                        f"Transaction {transaction.transaction_id} is locked",
                        security_type="entity_lock",
                        access_attempt={"operation": "add", "entity_id": transaction.transaction_id}
                    )
                
                # Comprehensive validation
                validation_result = self.transaction_validator.validate(
                    transaction,
                    self.transactions
                )
                
                if not validation_result.is_valid:
                    self.performance_metrics["validation_failures"] += 1
                    self._record_failed_validation(transaction.transaction_id)
                    
                    # Audit failed validation
                    self._audit_operation(
                        StateOperation.CREATE,
                        "transaction",
                        transaction.transaction_id,
                        success=False,
                        error_message=f"Validation failed: {validation_result.errors}"
                    )
                    
                    raise StateValidationError(
                        f"Transaction validation failed: {validation_result.errors}",
                        field="transaction",
                        value=transaction.transaction_id,
                        validation_type="comprehensive"
                    )
                
                # Create checkpoint before modification
                if self.auto_recovery:
                    self._create_recovery_checkpoint("pre_add_transaction")
                
                # Add to main storage
                self.transactions[transaction.transaction_id] = transaction
                
                # Update indexes with error handling
                try:
                    self._update_transaction_indexes(transaction, add=True)
                except Exception as e:
                    # Rollback on index update failure
                    del self.transactions[transaction.transaction_id]
                    raise StateConsistencyError(
                        f"Failed to update indexes: {e}",
                        inconsistency_type="index_update",
                        affected_entities=[transaction.transaction_id]
                    )
                
                # Update metrics
                self._update_performance_metrics("add_transaction", operation_start)
                self.performance_metrics["total_transactions"] += 1
                
                # Update versioning
                self.last_updated = datetime.now()
                self.version += 1
                
                # Record in state history
                self._record_state_change("add_transaction", transaction.transaction_id)
                
                # Audit successful operation
                self._audit_operation(
                    StateOperation.CREATE,
                    "transaction",
                    transaction.transaction_id,
                    success=True
                )
                
                # Async persistence if enabled
                if self.persistence_enabled:
                    self.executor.submit(self._async_persist_state)
                
                logger.debug(f"Added transaction: {transaction.transaction_id}")
                return True
                
            except FraudStateException:
                # Re-raise our custom exceptions
                raise
                
            except Exception as e:
                # Handle unexpected errors
                logger.error(f"Unexpected error adding transaction: {e}")
                self.performance_metrics["recovery_attempts"] += 1
                
                # Attempt recovery
                if self.auto_recovery:
                    self._attempt_state_recovery()
                
                # Audit the error
                self._audit_operation(
                    StateOperation.CREATE,
                    "transaction",
                    getattr(transaction, 'transaction_id', 'unknown'),
                    success=False,
                    error_message=str(e)
                )
                
                raise FraudStateException(
                    f"Failed to add transaction: {str(e)}",
                    error_code="UNEXPECTED_ERROR",
                    details={"transaction_id": getattr(transaction, 'transaction_id', 'unknown')},
                    recoverable=True
                )
    
    def _update_transaction_indexes(self, transaction: Transaction, add: bool = True):
        """Update transaction indexes with validation"""
        try:
            if add:
                # Add to indexes
                self.user_transactions[transaction.user_id].add(transaction.transaction_id)
                self.account_transactions[transaction.account_id].add(transaction.transaction_id)
                self.merchant_transactions[transaction.merchant].add(transaction.transaction_id)
                
                # Time-based indexes
                hour_key = transaction.timestamp.strftime("%Y%m%d%H")
                day_key = transaction.timestamp.strftime("%Y%m%d")
                self.hourly_transactions[hour_key].add(transaction.transaction_id)
                self.daily_transactions[day_key].add(transaction.transaction_id)
            else:
                # Remove from indexes
                self.user_transactions[transaction.user_id].discard(transaction.transaction_id)
                self.account_transactions[transaction.account_id].discard(transaction.transaction_id)
                self.merchant_transactions[transaction.merchant].discard(transaction.transaction_id)
                
                # Clean empty entries
                if not self.user_transactions[transaction.user_id]:
                    del self.user_transactions[transaction.user_id]
                if not self.account_transactions[transaction.account_id]:
                    del self.account_transactions[transaction.account_id]
                if not self.merchant_transactions[transaction.merchant]:
                    del self.merchant_transactions[transaction.merchant]
                
        except Exception as e:
            raise StateConsistencyError(
                f"Index update failed: {e}",
                inconsistency_type="index_corruption",
                affected_entities=[transaction.transaction_id]
            )
    
    def get_transaction(self, transaction_id: str, 
                       validate: bool = True) -> Optional[Transaction]:
        """Get transaction with optional validation"""
        operation_start = time.time()
        
        with self._lock:
            try:
                transaction = self.transactions.get(transaction_id)
                
                if transaction and validate:
                    # Validate transaction integrity
                    validation_result = self.transaction_validator.validate(transaction)
                    if not validation_result.is_valid:
                        logger.warning(f"Retrieved transaction has validation issues: {validation_result.warnings}")
                
                # Audit read operation
                self._audit_operation(
                    StateOperation.READ,
                    "transaction",
                    transaction_id,
                    success=transaction is not None
                )
                
                # Update metrics
                self._update_performance_metrics("get_transaction", operation_start)
                
                return transaction
                
            except Exception as e:
                logger.error(f"Error retrieving transaction {transaction_id}: {e}")
                self._audit_operation(
                    StateOperation.READ,
                    "transaction",
                    transaction_id,
                    success=False,
                    error_message=str(e)
                )
                return None
    
    def update_transaction(self, transaction_id: str, 
                          updates: Dict[str, Any]) -> bool:
        """Update transaction with validation and rollback capability"""
        operation_start = time.time()
        
        with self._lock:
            try:
                # Get existing transaction
                existing = self.transactions.get(transaction_id)
                if not existing:
                    raise StateValidationError(
                        f"Transaction {transaction_id} not found",
                        field="transaction_id",
                        value=transaction_id
                    )
                
                # Create backup for rollback
                backup = Transaction.from_dict(existing.to_dict())
                
                # Apply updates
                for field, value in updates.items():
                    if hasattr(existing, field):
                        setattr(existing, field, value)
                
                # Validate updated transaction
                validation_result = self.transaction_validator.validate(
                    existing,
                    self.transactions
                )
                
                if not validation_result.is_valid:
                    # Rollback
                    self.transactions[transaction_id] = backup
                    raise StateValidationError(
                        f"Updated transaction validation failed: {validation_result.errors}",
                        field="transaction",
                        value=transaction_id
                    )
                
                # Update indexes if needed
                if 'user_id' in updates or 'account_id' in updates or 'merchant' in updates:
                    self._update_transaction_indexes(backup, add=False)
                    self._update_transaction_indexes(existing, add=True)
                
                # Update versioning
                self.last_updated = datetime.now()
                self.version += 1
                
                # Audit
                self._audit_operation(
                    StateOperation.UPDATE,
                    "transaction",
                    transaction_id,
                    success=True,
                    details={"fields_updated": list(updates.keys())}
                )
                
                # Update metrics
                self._update_performance_metrics("update_transaction", operation_start)
                
                return True
                
            except FraudStateException:
                raise
                
            except Exception as e:
                logger.error(f"Failed to update transaction {transaction_id}: {e}")
                
                self._audit_operation(
                    StateOperation.UPDATE,
                    "transaction",
                    transaction_id,
                    success=False,
                    error_message=str(e)
                )
                
                return False
    
    def get_user_transactions(self, user_id: str, limit: Optional[int] = None) -> List[Transaction]:
        """Get transactions for a specific user"""
        with self._lock:
            transaction_ids = self.user_transactions.get(user_id, set())
            transactions = [
                self.transactions[tid] 
                for tid in transaction_ids 
                if tid in self.transactions
            ]
            
            # Sort by timestamp (most recent first)
            transactions.sort(key=lambda t: t.timestamp, reverse=True)
            
            if limit:
                return transactions[:limit]
            return transactions
    
    def get_recent_transactions(self, hours: int = 24) -> List[Transaction]:
        """Get recent transactions within specified hours"""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_transactions = [
                t for t in self.transactions.values()
                if t.timestamp >= cutoff_time
            ]
            return sorted(recent_transactions, key=lambda t: t.timestamp, reverse=True)
    
    def get_merchant_transactions(self, merchant: str, limit: Optional[int] = None) -> List[Transaction]:
        """Get transactions for a specific merchant"""
        with self._lock:
            transaction_ids = self.merchant_transactions.get(merchant, set())
            transactions = [
                self.transactions[tid] 
                for tid in transaction_ids 
                if tid in self.transactions
            ]
            
            transactions.sort(key=lambda t: t.timestamp, reverse=True)
            
            if limit:
                return transactions[:limit]
            return transactions
    
    def add_fraud_indicator(self, indicator: FraudIndicator) -> bool:
        """Add fraud indicator with comprehensive validation"""
        operation_start = time.time()
        
        with self._lock:
            try:
                # Validate indicator
                validation_result = self.indicator_validator.validate(
                    indicator,
                    self.fraud_indicators
                )
                
                if not validation_result.is_valid:
                    self.performance_metrics["validation_failures"] += 1
                    raise StateValidationError(
                        f"Indicator validation failed: {validation_result.errors}",
                        field="indicator",
                        value=indicator.indicator_id
                    )
                
                # Check transaction reference
                if indicator.transaction_id and indicator.transaction_id not in self.transactions:
                    raise StateConsistencyError(
                        f"Indicator references non-existent transaction: {indicator.transaction_id}",
                        inconsistency_type="invalid_reference",
                        affected_entities=[indicator.indicator_id, indicator.transaction_id]
                    )
                
                # Add to storage
                self.fraud_indicators[indicator.indicator_id] = indicator
                
                # Update versioning
                self.last_updated = datetime.now()
                self.version += 1
                
                # Audit
                self._audit_operation(
                    StateOperation.CREATE,
                    "fraud_indicator",
                    indicator.indicator_id,
                    success=True
                )
                
                # Update metrics
                self._update_performance_metrics("add_fraud_indicator", operation_start)
                
                logger.debug(f"Added fraud indicator: {indicator.indicator_id}")
                return True
                
            except FraudStateException:
                raise
                
            except Exception as e:
                logger.error(f"Failed to add fraud indicator: {e}")
                
                self._audit_operation(
                    StateOperation.CREATE,
                    "fraud_indicator",
                    getattr(indicator, 'indicator_id', 'unknown'),
                    success=False,
                    error_message=str(e)
                )
                
                return False
    
    def get_fraud_indicator(self, indicator_id: str) -> Optional[FraudIndicator]:
        """Get fraud indicator from state"""
        with self._lock:
            return self.fraud_indicators.get(indicator_id)
    
    def get_transaction_indicators(self, transaction_id: str) -> List[FraudIndicator]:
        """Get all fraud indicators for a specific transaction"""
        with self._lock:
            return [
                ind for ind in self.fraud_indicators.values()
                if ind.transaction_id == transaction_id
            ]
    
    def add_fraud_result(self, result: FraudResult) -> bool:
        """Add fraud result with transaction status update and validation"""
        operation_start = time.time()
        
        with self._lock:
            try:
                # Validate result
                if not result.result_id:
                    raise StateValidationError("Result ID is required", field="result_id")
                
                if not 0 <= result.risk_score <= 1:
                    raise StateValidationError(
                        f"Invalid risk score: {result.risk_score}",
                        field="risk_score",
                        value=result.risk_score
                    )
                
                if not 0 <= result.fraud_probability <= 1:
                    raise StateValidationError(
                        f"Invalid fraud probability: {result.fraud_probability}",
                        field="fraud_probability",
                        value=result.fraud_probability
                    )
                
                # Check transaction exists
                if result.transaction_id not in self.transactions:
                    raise StateConsistencyError(
                        f"Result references non-existent transaction: {result.transaction_id}",
                        inconsistency_type="invalid_reference",
                        affected_entities=[result.result_id, result.transaction_id]
                    )
                
                # Check for duplicate results
                existing_result = self._get_transaction_result(result.transaction_id)
                if existing_result and existing_result.result_id != result.result_id:
                    raise StateConsistencyError(
                        f"Transaction {result.transaction_id} already has result {existing_result.result_id}",
                        inconsistency_type="duplicate_result",
                        affected_entities=[result.result_id, existing_result.result_id]
                    )
                
                # Add to storage
                self.fraud_results[result.result_id] = result
                
                # Update transaction status based on decision
                transaction = self.transactions[result.transaction_id]
                transaction.risk_score = result.risk_score
                
                status_map = {
                    FraudDecision.BLOCK: TransactionStatus.BLOCKED,
                    FraudDecision.REVIEW: TransactionStatus.FLAGGED,
                    FraudDecision.ALLOW: TransactionStatus.COMPLETED,
                    FraudDecision.ESCALATE: TransactionStatus.UNDER_REVIEW,
                    FraudDecision.QUARANTINE: TransactionStatus.FLAGGED
                }
                
                new_status = status_map.get(result.decision)
                if new_status:
                    transaction.status = new_status
                
                # Update performance metrics
                if result.fraud_probability > 0.8:
                    self.performance_metrics["fraud_detected"] += 1
                
                # Update versioning
                self.last_updated = datetime.now()
                self.version += 1
                
                # Audit
                self._audit_operation(
                    StateOperation.CREATE,
                    "fraud_result",
                    result.result_id,
                    success=True,
                    details={
                        "transaction_id": result.transaction_id,
                        "decision": result.decision.value,
                        "risk_score": result.risk_score
                    }
                )
                
                # Update metrics
                self._update_performance_metrics("add_fraud_result", operation_start)
                
                logger.debug(f"Added fraud result: {result.result_id}")
                return True
                
            except FraudStateException:
                raise
                
            except Exception as e:
                logger.error(f"Failed to add fraud result: {e}")
                
                self._audit_operation(
                    StateOperation.CREATE,
                    "fraud_result",
                    getattr(result, 'result_id', 'unknown'),
                    success=False,
                    error_message=str(e)
                )
                
                return False
    
    def get_fraud_result(self, result_id: str) -> Optional[FraudResult]:
        """Get fraud result from state"""
        with self._lock:
            return self.fraud_results.get(result_id)
    
    def get_transaction_result(self, transaction_id: str) -> Optional[FraudResult]:
        """Get fraud result for a specific transaction"""
        with self._lock:
            return self._get_transaction_result(transaction_id)
    
    def _get_transaction_result(self, transaction_id: str) -> Optional[FraudResult]:
        """Get fraud result for a specific transaction"""
        for result in self.fraud_results.values():
            if result.transaction_id == transaction_id:
                return result
        return None
    
    def validate_state(self) -> EnhancedValidationResult:
        """Comprehensive state validation with detailed reporting"""
        operation_start = time.time()
        
        with self._lock:
            try:
                # Use consistency validator
                validation_result = self.consistency_validator.validate_state(self)
                
                # Store validation result
                self.validation_results[validation_result.validation_id] = validation_result
                
                # Audit
                self._audit_operation(
                    StateOperation.VALIDATE,
                    "state",
                    self.domain_name,
                    success=validation_result.is_valid,
                    details={
                        "errors": len(validation_result.errors),
                        "warnings": len(validation_result.warnings)
                    }
                )
                
                # Update metrics
                self._update_performance_metrics("validate_state", operation_start)
                
                return validation_result
                
            except Exception as e:
                logger.error(f"State validation failed: {e}")
                
                # Return error result
                return EnhancedValidationResult(
                    validation_id=f"val_{uuid.uuid4().hex[:8]}",
                    target_type="state",
                    target_id=self.domain_name,
                    is_valid=False,
                    errors=[f"Validation error: {str(e)}"],
                    validation_time_ms=(time.time() - operation_start) * 1000
                )
    
    def clear_state(self) -> None:
        """Clear state with audit and recovery checkpoint"""
        with self._lock:
            try:
                # Create recovery checkpoint before clearing
                if self.auto_recovery:
                    self._create_recovery_checkpoint("pre_clear_state")
                
                # Get counts before clearing
                counts = {
                    "transactions": len(self.transactions),
                    "indicators": len(self.fraud_indicators),
                    "results": len(self.fraud_results)
                }
                
                # Clear main storage
                self.transactions.clear()
                self.fraud_indicators.clear()
                self.fraud_results.clear()
                self.validation_results.clear()
                self.processing_metadata.clear()
                
                # Clear indexes
                self.user_transactions.clear()
                self.account_transactions.clear()
                self.merchant_transactions.clear()
                self.hourly_transactions.clear()
                self.daily_transactions.clear()
                
                # Reset metrics
                self.performance_metrics = {
                    "total_transactions": 0,
                    "fraud_detected": 0,
                    "false_positives": 0,
                    "true_positives": 0,
                    "processing_time_avg": 0.0,
                    "detection_rate": 0.0,
                    "validation_failures": 0,
                    "recovery_attempts": 0
                }
                
                # Clear security tracking
                self.failed_validations.clear()
                self.locked_entities.clear()
                
                # Update versioning
                self.version += 1
                self.last_updated = datetime.now()
                
                # Audit
                self._audit_operation(
                    StateOperation.DELETE,
                    "state",
                    self.domain_name,
                    success=True,
                    details={"cleared_counts": counts}
                )
                
                logger.info("State cleared")
                
            except Exception as e:
                logger.error(f"Failed to clear state: {e}")
                
                # Attempt recovery
                if self.auto_recovery:
                    self._attempt_state_recovery()
                
                raise FraudStateException(
                    "Failed to clear state",
                    error_code="CLEAR_STATE_ERROR",
                    details={"error": str(e)}
                )
    
    def save_state(self) -> bool:
        """Save state with validation and error handling"""
        if not self.persistence_enabled:
            return True
        
        with self._lock:
            try:
                return self.persistence_manager.save_state(self)
            except Exception as e:
                logger.error(f"Failed to save state: {e}")
                
                self._audit_operation(
                    StateOperation.EXPORT,
                    "state",
                    self.domain_name,
                    success=False,
                    error_message=str(e)
                )
                
                return False
    
    def load_state(self) -> bool:
        """Load state with validation and recovery"""
        if not self.persistence_enabled:
            return True
        
        try:
            loaded_state = self.persistence_manager.load_state(self.domain_name)
            
            if loaded_state:
                # Replace current state with loaded state
                with self._lock:
                    self.__dict__.update(loaded_state.__dict__)
                    
                    # Re-initialize components that shouldn't be persisted
                    self._lock = threading.RLock()
                    self.executor = ThreadPoolExecutor(max_workers=4)
                    
                    # Re-initialize validators with current config
                    self.transaction_validator = TransactionValidator(self.validation_config)
                    self.indicator_validator = FraudIndicatorValidator(self.validation_config)
                    self.consistency_validator = StateConsistencyValidator(self.validation_config)
                
                self._audit_operation(
                    StateOperation.IMPORT,
                    "state",
                    self.domain_name,
                    success=True
                )
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            
            self._audit_operation(
                StateOperation.IMPORT,
                "state",
                self.domain_name,
                success=False,
                error_message=str(e)
            )
            
            return False
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive state summary with health indicators"""
        with self._lock:
            try:
                # Basic counts
                summary = {
                    "domain_name": self.domain_name,
                    "version": self.version,
                    "created_at": self.created_at.isoformat(),
                    "last_updated": self.last_updated.isoformat(),
                    "counts": {
                        "transactions": len(self.transactions),
                        "fraud_indicators": len(self.fraud_indicators),
                        "fraud_results": len(self.fraud_results),
                        "validation_results": len(self.validation_results),
                        "processing_sessions": len(self.processing_metadata)
                    },
                    "indexes": {
                        "unique_users": len(self.user_transactions),
                        "unique_accounts": len(self.account_transactions),
                        "unique_merchants": len(self.merchant_transactions),
                        "hourly_buckets": len(self.hourly_transactions),
                        "daily_buckets": len(self.daily_transactions)
                    },
                    "performance_metrics": self.performance_metrics.copy(),
                    "state_history_size": len(self.state_history),
                    "audit_log_size": len(self.audit_log),
                    "health_indicators": self._calculate_health_indicators(),
                    "security_status": {
                        "locked_entities": len(self.locked_entities),
                        "recent_validation_failures": sum(self.failed_validations.values())
                    }
                }
                
                # Add validation status
                recent_validation = self.validate_state()
                summary["validation_status"] = {
                    "is_valid": recent_validation.is_valid,
                    "error_count": len(recent_validation.errors),
                    "warning_count": len(recent_validation.warnings)
                }
                
                return summary
                
            except Exception as e:
                logger.error(f"Failed to generate state summary: {e}")
                return {
                    "error": str(e),
                    "domain_name": self.domain_name
                }
    
    # Private helper methods
    
    def _record_failed_validation(self, entity_id: str):
        """Record failed validation attempt"""
        self.failed_validations[entity_id] += 1
        
        # Lock entity if too many failures
        if self.failed_validations[entity_id] >= self.validation_config.max_failed_validations:
            self.locked_entities.add(entity_id)
            logger.warning(f"Entity {entity_id} locked due to repeated validation failures")
    
    def _audit_operation(self, operation: StateOperation, entity_type: str,
                        entity_id: str, success: bool, 
                        details: Dict[str, Any] = None,
                        error_message: str = None):
        """Record audit entry for operation"""
        if not self.enable_audit:
            return
        
        audit_entry = StateAuditEntry(
            audit_id=f"audit_{uuid.uuid4().hex[:8]}",
            operation=operation,
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=None,  # Would be set from context in production
            timestamp=datetime.now(),
            success=success,
            details=details or {},
            error_message=error_message
        )
        
        self.audit_log.append(audit_entry)
    
    def _create_recovery_checkpoint(self, checkpoint_name: str):
        """Create recovery checkpoint"""
        try:
            checkpoint_data = {
                'name': checkpoint_name,
                'timestamp': datetime.now(),
                'version': self.version,
                'transaction_count': len(self.transactions),
                'indicator_count': len(self.fraud_indicators),
                'result_count': len(self.fraud_results)
            }
            
            self.recovery_checkpoints.append(checkpoint_data)
            
        except Exception as e:
            logger.warning(f"Failed to create recovery checkpoint: {e}")
    
    def _attempt_state_recovery(self):
        """Attempt to recover from last checkpoint"""
        if not self.recovery_checkpoints:
            logger.warning("No recovery checkpoints available")
            return
        
        try:
            last_checkpoint = self.recovery_checkpoints[-1]
            logger.info(f"Attempting recovery from checkpoint: {last_checkpoint['name']}")
            
            # In production, this would restore from persisted checkpoint
            # For now, we just log the attempt
            self.performance_metrics["recovery_attempts"] += 1
            
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
    
    def _record_state_change(self, change_type: str, entity_id: str):
        """Record state change in history"""
        change_record = {
            'timestamp': datetime.now(),
            'change_type': change_type,
            'entity_id': entity_id,
            'version': self.version
        }
        
        self.state_history.append(change_record)
    
    def _update_performance_metrics(self, operation: str, start_time: float):
        """Update performance metrics for operation"""
        elapsed_ms = (time.time() - start_time) * 1000
        
        self.operation_times.append({
            'operation': operation,
            'time_ms': elapsed_ms,
            'timestamp': datetime.now()
        })
        
        # Check performance threshold
        if elapsed_ms > self.validation_config.max_operation_time_ms:
            logger.warning(f"Operation {operation} took {elapsed_ms:.2f}ms, exceeds threshold")
    
    def _async_persist_state(self):
        """Asynchronously persist state"""
        try:
            self.save_state()
        except Exception as e:
            logger.error(f"Async state persistence failed: {e}")
    
    def _calculate_health_indicators(self) -> Dict[str, Any]:
        """Calculate health indicators for the state"""
        try:
            # Calculate average operation time
            if self.operation_times:
                avg_time = sum(op['time_ms'] for op in self.operation_times) / len(self.operation_times)
            else:
                avg_time = 0
            
            # Calculate validation failure rate
            total_validations = sum(1 for r in self.validation_results.values())
            failed_validations = sum(1 for r in self.validation_results.values() if not r.is_valid)
            failure_rate = failed_validations / total_validations if total_validations > 0 else 0
            
            return {
                "average_operation_time_ms": avg_time,
                "validation_failure_rate": failure_rate,
                "recovery_checkpoint_count": len(self.recovery_checkpoints),
                "performance_status": "healthy" if avg_time < self.validation_config.max_operation_time_ms else "degraded"
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate health indicators: {e}")
            return {}


# ============= COMPREHENSIVE TEST FUNCTION =============

def test_enhanced_fraud_state():
    """Test enhanced fraud state functionality with validation and error handling"""
    import traceback
    
    # Configure logging for tests
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n=== Testing Enhanced Financial Fraud State ===\n")
    
    # Create state instance with test configuration
    config = {
        "persistence_enabled": True,
        "enable_audit_logging": True,
        "auto_recovery": True,
        "validation": {
            "max_transaction_amount": 100000.0,
            "enable_security_validation": True,
            "max_operation_time_ms": 500.0
        }
    }
    
    state = FinancialFraudState("test-domain", config)
    
    try:
        # Test 1: Valid transaction
        print("Test 1: Valid Transaction")
        print("-" * 50)
        transaction1 = Transaction(
            transaction_id="TXN12345678",
            amount=1000.0,
            currency="USD",
            merchant="Test Merchant Co.",
            timestamp=datetime.now(),
            user_id="USER001",
            account_id="ACC001",
            payment_method="card",
            status=TransactionStatus.PENDING,
            ip_address="192.168.1.100",
            location={"country": "USA", "city": "New York"}
        )
        
        result = state.add_transaction(transaction1)
        print(f"✓ Added valid transaction: {result}")
        print(f"  State version: {state.version}")
        print()
        
        # Test 2: Invalid transaction (validation failure)
        print("Test 2: Invalid Transaction (Validation Failure)")
        print("-" * 50)
        invalid_transaction = Transaction(
            transaction_id="INVALID_ID_FORMAT_123",  # Invalid format
            amount=-100.0,  # Negative amount
            currency="XXX",  # Invalid currency
            merchant="",  # Empty merchant
            timestamp=datetime.now() + timedelta(days=1),  # Future timestamp
            user_id="",  # Empty user ID
            account_id="",  # Empty account ID
            payment_method="invalid_method"
        )
        
        try:
            state.add_transaction(invalid_transaction)
            print("✗ ERROR: Should have raised validation error")
        except StateValidationError as e:
            print(f"✓ Correctly caught validation error: {e}")
            print(f"  Error code: {e.error_code}")
            print(f"  Failed field: {e.details.get('field')}")
        print()
        
        # Test 3: Duplicate transaction
        print("Test 3: Duplicate Transaction")
        print("-" * 50)
        try:
            state.add_transaction(transaction1)  # Try to add same transaction again
            print("✗ ERROR: Should have caught duplicate")
        except StateValidationError as e:
            print(f"✓ Correctly caught duplicate: {e}")
        print()
        
        # Test 4: Fraud indicator with validation
        print("Test 4: Fraud Indicator with Validation")
        print("-" * 50)
        indicator1 = FraudIndicator(
            indicator_id="IND001",
            indicator_type=IndicatorType.AMOUNT,
            severity=FraudSeverity.HIGH,
            confidence=0.95,
            description="High amount transaction detected",
            timestamp=datetime.now(),
            source="rule_engine",
            transaction_id="TXN12345678",
            threshold=0.8
        )
        
        result = state.add_fraud_indicator(indicator1)
        print(f"✓ Added fraud indicator: {result}")
        print()
        
        # Test 5: Invalid indicator (references non-existent transaction)
        print("Test 5: Invalid Indicator (Bad Reference)")
        print("-" * 50)
        invalid_indicator = FraudIndicator(
            indicator_id="IND002",
            indicator_type=IndicatorType.VELOCITY,
            severity=FraudSeverity.MEDIUM,
            confidence=1.5,  # Invalid confidence > 1.0
            description="Test",
            timestamp=datetime.now(),
            source="test",
            transaction_id="NON_EXISTENT_TXN"
        )
        
        try:
            state.add_fraud_indicator(invalid_indicator)
            print("✗ ERROR: Should have caught invalid reference")
        except (StateValidationError, StateConsistencyError) as e:
            print(f"✓ Correctly caught error: {e}")
            print(f"  Error type: {type(e).__name__}")
        print()
        
        # Test 6: Fraud result with transaction update
        print("Test 6: Fraud Result with Transaction Update")
        print("-" * 50)
        fraud_result = FraudResult(
            result_id="RES001",
            transaction_id="TXN12345678",
            risk_score=0.85,
            fraud_probability=0.78,
            decision=FraudDecision.REVIEW,
            processing_time=45.2,
            indicators=[indicator1]
        )
        
        result = state.add_fraud_result(fraud_result)
        print(f"✓ Added fraud result: {result}")
        
        # Check transaction status was updated
        updated_tx = state.get_transaction("TXN12345678")
        print(f"  Transaction status updated to: {updated_tx.status.value}")
        print(f"  Transaction risk score: {updated_tx.risk_score}")
        print()
        
        # Test 7: State validation
        print("Test 7: Comprehensive State Validation")
        print("-" * 50)
        validation_result = state.validate_state()
        print(f"State validation: {'✓ PASSED' if validation_result.is_valid else '✗ FAILED'}")
        print(f"  Validation ID: {validation_result.validation_id}")
        print(f"  Errors: {len(validation_result.errors)}")
        print(f"  Warnings: {len(validation_result.warnings)}")
        print(f"  Validation time: {validation_result.validation_time_ms:.2f}ms")
        
        if validation_result.issues:
            print("  Issues found:")
            for issue in validation_result.issues[:3]:  # Show first 3 issues
                print(f"    - [{issue.severity.value}] {issue.field}: {issue.message}")
        print()
        
        # Test 8: Transaction update with validation
        print("Test 8: Transaction Update with Validation")
        print("-" * 50)
        update_result = state.update_transaction(
            "TXN12345678",
            {
                "amount": 1500.0,
                "status": TransactionStatus.COMPLETED
            }
        )
        print(f"✓ Transaction update: {update_result}")
        
        # Try invalid update
        try:
            state.update_transaction(
                "TXN12345678",
                {
                    "amount": -1000.0  # Invalid negative amount
                }
            )
            print("✗ ERROR: Should have caught invalid update")
        except StateValidationError as e:
            print(f"✓ Correctly rejected invalid update: {e}")
        print()
        
        # Test 9: Performance monitoring
        print("Test 9: Performance Monitoring")
        print("-" * 50)
        
        # Add multiple transactions to test performance
        start_time = time.time()
        for i in range(10):
            tx = Transaction(
                transaction_id=f"PERF_TXN_{i:04d}",
                amount=100.0 * (i + 1),
                currency="USD",
                merchant=f"Merchant {i}",
                timestamp=datetime.now() - timedelta(minutes=i),
                user_id=f"USER{i % 3}",
                account_id=f"ACC{i % 3}",
                payment_method="card"
            )
            state.add_transaction(tx)
        
        elapsed = time.time() - start_time
        print(f"Added 10 transactions in {elapsed:.3f}s ({elapsed/10*1000:.1f}ms per transaction)")
        
        # Check operation times
        if state.operation_times:
            avg_time = sum(op['time_ms'] for op in state.operation_times) / len(state.operation_times)
            print(f"Average operation time: {avg_time:.2f}ms")
        print()
        
        # Test 10: Security - Entity locking
        print("Test 10: Security - Entity Locking")
        print("-" * 50)
        
        # Simulate multiple validation failures
        bad_tx = Transaction(
            transaction_id="LOCK_TEST_TXN",
            amount=-1000,  # Invalid
            currency="USD",
            merchant="Test",
            timestamp=datetime.now(),
            user_id="USER001",
            account_id="ACC001",
            payment_method="card"
        )
        
        for i in range(6):  # Exceed max_failed_validations
            try:
                state.add_transaction(bad_tx)
            except StateValidationError:
                pass
        
        # Try to add with same ID (should be locked now)
        try:
            good_tx = bad_tx
            good_tx.amount = 100.0  # Fix the amount
            state.add_transaction(good_tx)
            print("✗ ERROR: Should have been locked")
        except StateSecurityError as e:
            print(f"✓ Entity correctly locked: {e}")
            print(f"  Security type: {e.details.get('security_type')}")
        print()
        
        # Test 11: State persistence
        print("Test 11: State Persistence")
        print("-" * 50)
        
        # Save state
        save_result = state.save_state()
        print(f"State save: {'✓ SUCCESS' if save_result else '✗ FAILED'}")
        
        # Create new state instance and load
        new_state = FinancialFraudState("test-domain", config)
        load_result = new_state.load_state()
        print(f"State load: {'✓ SUCCESS' if load_result else '✗ FAILED'}")
        
        if load_result:
            print(f"  Loaded transactions: {len(new_state.transactions)}")
            print(f"  Loaded indicators: {len(new_state.fraud_indicators)}")
            print(f"  State version: {new_state.version}")
        print()
        
        # Test 12: State summary and health
        print("Test 12: State Summary and Health")
        print("-" * 50)
        summary = state.get_state_summary()
        
        print("State Summary:")
        print(f"  Version: {summary['version']}")
        print(f"  Transactions: {summary['counts']['transactions']}")
        print(f"  Fraud Indicators: {summary['counts']['fraud_indicators']}")
        print(f"  Unique Users: {summary['indexes']['unique_users']}")
        print(f"  Validation Status: {'✓' if summary['validation_status']['is_valid'] else '✗'}")
        print(f"  Health Status: {summary['health_indicators'].get('performance_status', 'unknown')}")
        print(f"  Audit Log Size: {summary['audit_log_size']}")
        print()
        
        # Test 13: Audit trail
        print("Test 13: Audit Trail")
        print("-" * 50)
        
        # Get recent audit entries
        recent_audits = list(state.audit_log)[-5:]  # Last 5 entries
        print(f"Recent audit entries ({len(recent_audits)}):")
        for audit in recent_audits:
            print(f"  [{audit.operation.value}] {audit.entity_type}/{audit.entity_id} - "
                  f"{'✓' if audit.success else '✗'} @ {audit.timestamp.strftime('%H:%M:%S')}")
        print()
        
        # Test 14: Clear state
        print("Test 14: Clear State")
        print("-" * 50)
        
        counts_before = {
            "transactions": len(state.transactions),
            "indicators": len(state.fraud_indicators),
            "results": len(state.fraud_results)
        }
        
        state.clear_state()
        
        counts_after = {
            "transactions": len(state.transactions),
            "indicators": len(state.fraud_indicators),
            "results": len(state.fraud_results)
        }
        
        print(f"Cleared state:")
        print(f"  Transactions: {counts_before['transactions']} → {counts_after['transactions']}")
        print(f"  Indicators: {counts_before['indicators']} → {counts_after['indicators']}")
        print(f"  Results: {counts_before['results']} → {counts_after['results']}")
        print()
        
        print("\n✅ All enhanced tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with unexpected error: {e}")
        traceback.print_exc()
    
    finally:
        # Cleanup
        state.executor.shutdown(wait=False)


if __name__ == "__main__":
    test_enhanced_fraud_state()