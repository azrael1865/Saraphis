"""
Enhanced Proof System Verifier for Financial Fraud Detection
Advanced proof system with comprehensive validation, error handling,
security features, and production-ready reliability mechanisms.
"""

import logging
import json
import hashlib
import time
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set, Type, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import numpy as np
import pandas as pd
import traceback
import psutil
import signal
from contextlib import contextmanager
from functools import wraps
import re
import secrets
import hmac

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# Custom Exception Classes
# ============================================================================

class ProofVerificationException(Exception):
    """Base exception for proof verification errors"""
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()

class ProofConfigurationError(ProofVerificationException):
    """Raised when proof system configuration is invalid"""
    pass

class ProofGenerationError(ProofVerificationException):
    """Raised when proof generation fails"""
    pass

class ProofValidationError(ProofVerificationException):
    """Raised when proof validation fails"""
    pass

class ProofTimeoutError(ProofVerificationException):
    """Raised when proof operations timeout"""
    pass

class ProofSecurityError(ProofVerificationException):
    """Raised when proof operations violate security constraints"""
    pass

class ProofIntegrityError(ProofVerificationException):
    """Raised when proof integrity checks fail"""
    pass

class ClaimValidationError(ProofVerificationException):
    """Raised when claim validation fails"""
    pass

class EvidenceValidationError(ProofVerificationException):
    """Raised when evidence validation fails"""
    pass

class ProofSystemError(ProofVerificationException):
    """Raised when underlying proof system fails"""
    pass

class ProofStorageError(ProofVerificationException):
    """Raised when proof storage operations fail"""
    pass

class CryptographicError(ProofVerificationException):
    """Raised when cryptographic operations fail"""
    pass

class ProofExpiredError(ProofVerificationException):
    """Raised when accessing expired proofs"""
    pass

class ResourceLimitError(ProofVerificationException):
    """Raised when resource limits are exceeded"""
    pass

# ============================================================================
# Validation and Security Components
# ============================================================================

class SecurityValidator:
    """Validates security aspects of proof operations"""
    
    @staticmethod
    def validate_input_safety(data: Any) -> Tuple[bool, str]:
        """Validate input data for safety"""
        try:
            # Check for injection patterns
            if isinstance(data, str):
                dangerous_patterns = [
                    r'<script[^>]*>.*?</script>',
                    r'javascript:',
                    r'on\w+\s*=',
                    r'eval\s*\(',
                    r'exec\s*\(',
                    r'import\s+',
                    r'__import__\s*\(',
                    r'getattr\s*\(',
                    r'setattr\s*\(',
                    r'delattr\s*\(',
                    r'hasattr\s*\(',
                ]
                
                for pattern in dangerous_patterns:
                    if re.search(pattern, data, re.IGNORECASE):
                        return False, f"Potentially dangerous pattern detected: {pattern}"
            
            # Check for nested depth (prevent DoS)
            if isinstance(data, (dict, list)):
                depth = SecurityValidator._get_nested_depth(data)
                if depth > 20:
                    return False, f"Nested depth {depth} exceeds maximum allowed depth"
            
            # Check data size
            data_size = len(str(data)) if data else 0
            if data_size > 1024 * 1024:  # 1MB limit
                return False, f"Data size {data_size} exceeds maximum allowed size"
            
            return True, "Input validation passed"
            
        except Exception as e:
            return False, f"Security validation error: {str(e)}"
    
    @staticmethod
    def _get_nested_depth(obj: Any, depth: int = 0) -> int:
        """Calculate nested depth of data structure"""
        if depth > 50:  # Prevent infinite recursion
            return depth
            
        if isinstance(obj, dict):
            return max([SecurityValidator._get_nested_depth(v, depth + 1) for v in obj.values()] + [depth])
        elif isinstance(obj, list):
            return max([SecurityValidator._get_nested_depth(item, depth + 1) for item in obj] + [depth])
        else:
            return depth
    
    @staticmethod
    def validate_proof_permissions(user_context: Dict[str, Any], operation: str) -> Tuple[bool, str]:
        """Validate user permissions for proof operations"""
        try:
            user_role = user_context.get('role', 'guest')
            user_permissions = user_context.get('permissions', [])
            
            required_permissions = {
                'generate_proof': ['proof:generate', 'proof:write'],
                'verify_proof': ['proof:verify', 'proof:read'],
                'delete_proof': ['proof:delete', 'proof:admin'],
                'export_data': ['proof:export', 'proof:admin']
            }
            
            required = required_permissions.get(operation, [])
            if not required:
                return True, "No specific permissions required"
            
            # Check if user has any of the required permissions
            if any(perm in user_permissions for perm in required):
                return True, "Permissions validated"
            
            # Special handling for admin role
            if user_role == 'admin':
                return True, "Admin permissions"
            
            return False, f"Insufficient permissions for {operation}"
            
        except Exception as e:
            return False, f"Permission validation error: {str(e)}"

class ResourceMonitor:
    """Monitors and limits resource usage"""
    
    def __init__(self, max_memory_mb: int = 512, max_cpu_percent: float = 50.0):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.process = psutil.Process()
    
    @contextmanager
    def monitor_resources(self, operation_name: str):
        """Context manager for resource monitoring"""
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            memory_used = end_memory - start_memory
            duration = end_time - start_time
            
            logger.debug(f"{operation_name} - Memory: {memory_used:.2f}MB, Duration: {duration:.2f}s")
            
            if memory_used > self.max_memory_mb:
                logger.warning(f"{operation_name} exceeded memory limit: {memory_used:.2f}MB > {self.max_memory_mb}MB")
    
    def check_resource_limits(self) -> Tuple[bool, str]:
        """Check if current resource usage is within limits"""
        try:
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent()
            
            if memory_mb > self.max_memory_mb:
                return False, f"Memory usage {memory_mb:.2f}MB exceeds limit {self.max_memory_mb}MB"
            
            if cpu_percent > self.max_cpu_percent:
                return False, f"CPU usage {cpu_percent:.2f}% exceeds limit {self.max_cpu_percent}%"
            
            return True, "Resource usage within limits"
            
        except Exception as e:
            return False, f"Resource check error: {str(e)}"

def timeout_decorator(timeout_seconds: int):
    """Decorator to add timeout to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def timeout_handler(signum, frame):
                raise ProofTimeoutError(f"Operation timed out after {timeout_seconds} seconds")
            
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

def validation_decorator(validator_func: Callable[[Any], Tuple[bool, str]]):
    """Decorator to add validation to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate all arguments
            for arg in args:
                is_valid, message = validator_func(arg)
                if not is_valid:
                    raise ProofValidationError(f"Validation failed: {message}")
            
            # Validate all keyword arguments
            for key, value in kwargs.items():
                is_valid, message = validator_func(value)
                if not is_valid:
                    raise ProofValidationError(f"Validation failed for {key}: {message}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# ============================================================================
# Enhanced Data Classes
# ============================================================================

class ProofType(Enum):
    """Types of fraud detection proofs"""
    TRANSACTION_FRAUD = "transaction_fraud"
    PATTERN_FRAUD = "pattern_fraud"
    RULE_VIOLATION = "rule_violation"
    ML_PREDICTION = "ml_prediction"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    NETWORK_FRAUD = "network_fraud"
    COMPOSITE = "composite"
    RISK_ASSESSMENT = "risk_assessment"
    COMPLIANCE_CHECK = "compliance_check"

class ProofStatus(Enum):
    """Proof verification status"""
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    EXPIRED = "expired"
    INVALID = "invalid"
    PROCESSING = "processing"
    PARTIAL = "partial"
    SUSPENDED = "suspended"

class ProofLevel(Enum):
    """Proof confidence levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class SecurityLevel(Enum):
    """Security levels for proof operations"""
    BASIC = "basic"
    STANDARD = "standard"
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"

@dataclass
class EnhancedProofClaim:
    """Enhanced fraud detection claim with comprehensive validation"""
    claim_id: str
    claim_type: ProofType
    transaction_id: str
    timestamp: datetime
    
    # Core fraud indicators
    fraud_probability: float
    risk_score: float
    confidence_level: ProofLevel
    evidence: Dict[str, Any]
    
    # Rule violations
    violated_rules: List[str] = field(default_factory=list)
    rule_severity: Dict[str, str] = field(default_factory=dict)
    
    # ML model information
    model_version: Optional[str] = None
    model_confidence: float = 0.0
    model_features: List[str] = field(default_factory=list)
    
    # Security and validation
    security_level: SecurityLevel = SecurityLevel.STANDARD
    validation_status: str = "pending"
    validation_errors: List[str] = field(default_factory=list)
    
    # Context information
    user_context: Dict[str, Any] = field(default_factory=dict)
    system_context: Dict[str, Any] = field(default_factory=dict)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Comprehensive claim validation"""
        errors = []
        
        # Basic field validation
        if not self.claim_id:
            errors.append("claim_id is required")
        
        if not self.transaction_id:
            errors.append("transaction_id is required")
        
        # Probability validation
        if not 0 <= self.fraud_probability <= 1:
            errors.append("fraud_probability must be between 0 and 1")
        
        if not 0 <= self.risk_score <= 1:
            errors.append("risk_score must be between 0 and 1")
        
        if not 0 <= self.model_confidence <= 1:
            errors.append("model_confidence must be between 0 and 1")
        
        # Evidence validation
        if not isinstance(self.evidence, dict):
            errors.append("evidence must be a dictionary")
        
        # Security validation
        is_safe, safety_msg = SecurityValidator.validate_input_safety(self.evidence)
        if not is_safe:
            errors.append(f"Evidence security validation failed: {safety_msg}")
        
        # Context validation
        if self.user_context:
            is_safe, safety_msg = SecurityValidator.validate_input_safety(self.user_context)
            if not is_safe:
                errors.append(f"User context security validation failed: {safety_msg}")
        
        return len(errors) == 0, errors

@dataclass 
class EnhancedProofEvidence:
    """Enhanced evidence with comprehensive validation and security"""
    evidence_id: str
    evidence_type: str
    source: str
    timestamp: datetime
    
    # Evidence data and metadata
    data: Dict[str, Any]
    confidence: float
    reliability_score: float = 1.0
    
    # Validation and verification
    verified: bool = False
    verification_method: Optional[str] = None
    verification_timestamp: Optional[datetime] = None
    validation_errors: List[str] = field(default_factory=list)
    
    # Security attributes
    integrity_hash: Optional[str] = None
    signature: Optional[str] = None
    security_level: SecurityLevel = SecurityLevel.STANDARD
    
    # Chain of custody
    chain_of_custody: List[Dict[str, Any]] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate evidence"""
        errors = []
        
        # Basic validation
        if not self.evidence_id:
            errors.append("evidence_id is required")
        
        if not 0 <= self.confidence <= 1:
            errors.append("confidence must be between 0 and 1")
        
        if not 0 <= self.reliability_score <= 1:
            errors.append("reliability_score must be between 0 and 1")
        
        # Data validation
        if not isinstance(self.data, dict):
            errors.append("data must be a dictionary")
        
        # Security validation
        is_safe, safety_msg = SecurityValidator.validate_input_safety(self.data)
        if not is_safe:
            errors.append(f"Data security validation failed: {safety_msg}")
        
        return len(errors) == 0, errors
    
    def calculate_integrity_hash(self) -> str:
        """Calculate integrity hash for evidence"""
        evidence_str = json.dumps({
            'evidence_id': self.evidence_id,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence
        }, sort_keys=True)
        
        return hashlib.sha256(evidence_str.encode()).hexdigest()

@dataclass
class EnhancedProofResult:
    """Enhanced proof result with comprehensive tracking"""
    proof_id: str
    claim_id: str
    status: ProofStatus
    confidence: float
    timestamp: datetime
    
    # Verification details
    verification_method: str
    verification_time_ms: float
    verification_steps: List[str] = field(default_factory=list)
    
    # Evidence and reasoning
    evidence_used: List[str] = field(default_factory=list)
    reasoning: Dict[str, Any] = field(default_factory=dict)
    decision_tree: Dict[str, Any] = field(default_factory=dict)
    
    # Security and integrity
    proof_hash: Optional[str] = None
    signature: Optional[str] = None
    security_level: SecurityLevel = SecurityLevel.STANDARD
    integrity_verified: bool = False
    
    # Validity and lifecycle
    valid_until: Optional[datetime] = None
    revoked: bool = False
    revocation_reason: Optional[str] = None
    
    # Quality metrics
    quality_score: float = 0.0
    reliability_score: float = 0.0
    
    # Audit trail
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if proof is still valid"""
        if self.revoked:
            return False
        
        if self.status not in [ProofStatus.VERIFIED, ProofStatus.PARTIAL]:
            return False
        
        if self.valid_until and datetime.now() > self.valid_until:
            return False
        
        return True
    
    def add_audit_entry(self, action: str, details: Dict[str, Any]) -> None:
        """Add entry to audit trail"""
        self.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        })

# ============================================================================
# Enhanced Proof Systems
# ============================================================================

class BaseProofSystem(ABC):
    """Enhanced abstract base class for proof systems"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.security_validator = SecurityValidator()
        self.resource_monitor = ResourceMonitor()
    
    @abstractmethod
    def generate_proof(self, claim: EnhancedProofClaim, evidence: List[EnhancedProofEvidence]) -> Dict[str, Any]:
        """Generate proof for a claim"""
        pass
    
    @abstractmethod
    def verify_proof(self, proof: Dict[str, Any], claim: EnhancedProofClaim) -> Tuple[bool, float]:
        """Verify a proof"""
        pass
    
    def validate_inputs(self, claim: EnhancedProofClaim, evidence: List[EnhancedProofEvidence]) -> None:
        """Validate inputs before processing"""
        # Validate claim
        is_valid, errors = claim.validate()
        if not is_valid:
            raise ClaimValidationError(f"Claim validation failed: {', '.join(errors)}")
        
        # Validate evidence
        for ev in evidence:
            is_valid, errors = ev.validate()
            if not is_valid:
                raise EvidenceValidationError(f"Evidence {ev.evidence_id} validation failed: {', '.join(errors)}")

class EnhancedRuleBasedProofSystem(BaseProofSystem):
    """Enhanced rule-based proof system with comprehensive validation"""
    
    def __init__(self, rules: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.rules = rules or self._default_rules()
        self.rule_weights = self.config.get('rule_weights', {})
    
    def _default_rules(self) -> Dict[str, Any]:
        """Enhanced default fraud detection rules"""
        return {
            'transaction_limits': {
                'max_amount': 10000,
                'max_daily_amount': 50000,
                'max_daily_transactions': 100,
                'max_velocity_transactions_per_hour': 20,
                'max_velocity_amount_per_hour': 25000
            },
            'geographical_rules': {
                'max_distance_km_per_hour': 1000,
                'restricted_countries': ['XX', 'YY'],
                'suspicious_countries': ['AA', 'BB'],
                'allow_international': True
            },
            'behavioral_rules': {
                'unusual_time_threshold_std': 3,
                'unusual_amount_threshold_std': 3,
                'min_historical_transactions': 10,
                'account_age_days_threshold': 30
            },
            'pattern_rules': {
                'round_number_threshold': 0.8,
                'repeated_merchant_threshold': 5,
                'card_testing_threshold': 10,
                'velocity_burst_threshold': 1.5
            },
            'compliance_rules': {
                'kyc_required': True,
                'aml_screening_required': True,
                'sanctions_check_required': True,
                'pep_check_required': True
            }
        }
    
    @timeout_decorator(30)
    @validation_decorator(SecurityValidator.validate_input_safety)
    def generate_proof(self, claim: EnhancedProofClaim, evidence: List[EnhancedProofEvidence]) -> Dict[str, Any]:
        """Generate enhanced rule-based proof"""
        with self.resource_monitor.monitor_resources("rule_proof_generation"):
            self.validate_inputs(claim, evidence)
            
            violated_rules = []
            rule_scores = {}
            rule_details = {}
            
            # Aggregate evidence data
            evidence_data = {}
            for ev in evidence:
                evidence_data.update(ev.data)
                evidence_data[f'_evidence_{ev.evidence_id}_confidence'] = ev.confidence
            
            # Check each rule category with detailed analysis
            for category, rules in self.rules.items():
                try:
                    category_violations, category_score, category_details = self._check_rule_category_enhanced(
                        category, rules, claim, evidence_data
                    )
                    
                    if category_violations:
                        violated_rules.extend(category_violations)
                        rule_scores[category] = category_score
                        rule_details[category] = category_details
                        
                except Exception as e:
                    logger.error(f"Rule category {category} check failed: {str(e)}")
                    rule_details[category] = {'error': str(e)}
            
            # Calculate weighted overall score
            overall_score = self._calculate_weighted_score(rule_scores)
            
            # Determine severity
            severity = self._determine_severity(overall_score, violated_rules)
            
            return {
                'proof_type': 'enhanced_rule_based',
                'violated_rules': violated_rules,
                'rule_scores': rule_scores,
                'rule_details': rule_details,
                'overall_score': overall_score,
                'severity': severity,
                'confidence_factors': self._calculate_confidence_factors(evidence),
                'timestamp': datetime.now().isoformat(),
                'rule_engine_version': '2.0.0',
                'evidence_count': len(evidence),
                'evidence_reliability': np.mean([ev.reliability_score for ev in evidence]) if evidence else 0
            }
    
    def verify_proof(self, proof: Dict[str, Any], claim: EnhancedProofClaim) -> Tuple[bool, float]:
        """Verify enhanced rule-based proof"""
        try:
            if proof.get('proof_type') != 'enhanced_rule_based':
                return False, 0.0
            
            # Verify proof structure
            required_fields = ['violated_rules', 'overall_score', 'severity', 'rule_engine_version']
            for field in required_fields:
                if field not in proof:
                    logger.warning(f"Missing required field in proof: {field}")
                    return False, 0.0
            
            # Verify proof integrity
            violated_rules = proof.get('violated_rules', [])
            overall_score = proof.get('overall_score', 0)
            severity = proof.get('severity', 'low')
            
            # Proof is valid if rules were violated and score is reasonable
            is_valid = len(violated_rules) > 0 and overall_score > 0
            
            # Calculate confidence based on multiple factors
            base_confidence = min(overall_score * 100, 100) if is_valid else 0
            
            # Adjust confidence based on evidence reliability
            evidence_reliability = proof.get('evidence_reliability', 0.5)
            confidence = base_confidence * (0.5 + 0.5 * evidence_reliability)
            
            # Severity adjustment
            severity_multipliers = {'low': 0.8, 'medium': 1.0, 'high': 1.2, 'critical': 1.5}
            confidence *= severity_multipliers.get(severity, 1.0)
            
            return is_valid, min(confidence, 100.0)
            
        except Exception as e:
            logger.error(f"Rule-based proof verification failed: {str(e)}")
            return False, 0.0
    
    def _check_rule_category_enhanced(self, category: str, rules: Dict[str, Any], 
                                    claim: EnhancedProofClaim, evidence_data: Dict[str, Any]) -> Tuple[List[str], float, Dict[str, Any]]:
        """Enhanced rule category checking with detailed analysis"""
        violations = []
        details = {}
        
        if category == 'transaction_limits':
            violations, details = self._check_transaction_limits(rules, evidence_data)
        elif category == 'geographical_rules':
            violations, details = self._check_geographical_rules(rules, evidence_data)
        elif category == 'behavioral_rules':
            violations, details = self._check_behavioral_rules(rules, evidence_data)
        elif category == 'pattern_rules':
            violations, details = self._check_pattern_rules(rules, evidence_data)
        elif category == 'compliance_rules':
            violations, details = self._check_compliance_rules(rules, evidence_data)
        
        # Calculate category score
        if violations:
            severity_weights = {'low': 1, 'medium': 2, 'high': 3, 'critical': 5}
            total_weight = sum(severity_weights.get(details.get(v, {}).get('severity', 'low'), 1) for v in violations)
            max_possible_weight = len(rules) * severity_weights['critical']
            score = min(total_weight / max_possible_weight, 1.0) if max_possible_weight > 0 else 0
        else:
            score = 0.0
        
        return violations, score, details
    
    def _check_transaction_limits(self, rules: Dict[str, Any], evidence_data: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        """Check transaction limit rules"""
        violations = []
        details = {}
        
        amount = evidence_data.get('amount', evidence_data.get('transaction_amount', 0))
        daily_amount = evidence_data.get('daily_amount', 0)
        daily_count = evidence_data.get('daily_transaction_count', 0)
        hourly_count = evidence_data.get('hourly_transaction_count', 0)
        hourly_amount = evidence_data.get('hourly_amount', 0)
        
        # Check individual transaction limit
        if amount > rules.get('max_amount', float('inf')):
            violations.append('max_transaction_amount_exceeded')
            details['max_transaction_amount_exceeded'] = {
                'severity': 'high',
                'actual': amount,
                'limit': rules['max_amount'],
                'ratio': amount / rules['max_amount']
            }
        
        # Check daily amount limit
        if daily_amount > rules.get('max_daily_amount', float('inf')):
            violations.append('max_daily_amount_exceeded')
            details['max_daily_amount_exceeded'] = {
                'severity': 'medium',
                'actual': daily_amount,
                'limit': rules['max_daily_amount'],
                'ratio': daily_amount / rules['max_daily_amount']
            }
        
        # Check daily transaction count
        if daily_count > rules.get('max_daily_transactions', float('inf')):
            violations.append('max_daily_transactions_exceeded')
            details['max_daily_transactions_exceeded'] = {
                'severity': 'medium',
                'actual': daily_count,
                'limit': rules['max_daily_transactions'],
                'ratio': daily_count / rules['max_daily_transactions']
            }
        
        # Check velocity rules
        if hourly_count > rules.get('max_velocity_transactions_per_hour', float('inf')):
            violations.append('velocity_transaction_count_exceeded')
            details['velocity_transaction_count_exceeded'] = {
                'severity': 'high',
                'actual': hourly_count,
                'limit': rules['max_velocity_transactions_per_hour'],
                'ratio': hourly_count / rules['max_velocity_transactions_per_hour']
            }
        
        if hourly_amount > rules.get('max_velocity_amount_per_hour', float('inf')):
            violations.append('velocity_amount_exceeded')
            details['velocity_amount_exceeded'] = {
                'severity': 'high',
                'actual': hourly_amount,
                'limit': rules['max_velocity_amount_per_hour'],
                'ratio': hourly_amount / rules['max_velocity_amount_per_hour']
            }
        
        return violations, details
    
    def _check_geographical_rules(self, rules: Dict[str, Any], evidence_data: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        """Check geographical rules"""
        violations = []
        details = {}
        
        country = evidence_data.get('country', evidence_data.get('transaction_country'))
        distance_km = evidence_data.get('distance_from_last_transaction_km', 0)
        time_since_last_transaction_hours = evidence_data.get('time_since_last_transaction_hours', 24)
        
        # Check restricted countries
        if country in rules.get('restricted_countries', []):
            violations.append('restricted_country_transaction')
            details['restricted_country_transaction'] = {
                'severity': 'critical',
                'country': country,
                'restriction_reason': 'blacklisted'
            }
        
        # Check suspicious countries
        elif country in rules.get('suspicious_countries', []):
            violations.append('suspicious_country_transaction')
            details['suspicious_country_transaction'] = {
                'severity': 'medium',
                'country': country,
                'restriction_reason': 'high_risk'
            }
        
        # Check impossible travel
        if time_since_last_transaction_hours > 0:
            max_possible_distance = rules.get('max_distance_km_per_hour', 1000) * time_since_last_transaction_hours
            if distance_km > max_possible_distance:
                violations.append('impossible_travel_detected')
                details['impossible_travel_detected'] = {
                    'severity': 'critical',
                    'distance_km': distance_km,
                    'time_hours': time_since_last_transaction_hours,
                    'max_possible_distance': max_possible_distance,
                    'impossible_ratio': distance_km / max_possible_distance
                }
        
        return violations, details
    
    def _check_behavioral_rules(self, rules: Dict[str, Any], evidence_data: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        """Check behavioral rules"""
        violations = []
        details = {}
        
        transaction_hour = evidence_data.get('transaction_hour')
        amount = evidence_data.get('amount', 0)
        user_avg_amount = evidence_data.get('user_average_amount', amount)
        user_std_amount = evidence_data.get('user_amount_std', amount * 0.1)
        account_age_days = evidence_data.get('account_age_days', 999)
        historical_transaction_count = evidence_data.get('historical_transaction_count', 999)
        
        # Check unusual timing
        if transaction_hour is not None:
            user_avg_hour = evidence_data.get('user_average_transaction_hour', 12)
            user_std_hour = evidence_data.get('user_transaction_hour_std', 6)
            
            if user_std_hour > 0:
                hour_z_score = abs(transaction_hour - user_avg_hour) / user_std_hour
                if hour_z_score > rules.get('unusual_time_threshold_std', 3):
                    violations.append('unusual_transaction_time')
                    details['unusual_transaction_time'] = {
                        'severity': 'low',
                        'transaction_hour': transaction_hour,
                        'user_avg_hour': user_avg_hour,
                        'z_score': hour_z_score,
                        'threshold': rules['unusual_time_threshold_std']
                    }
        
        # Check unusual amount
        if user_std_amount > 0:
            amount_z_score = abs(amount - user_avg_amount) / user_std_amount
            if amount_z_score > rules.get('unusual_amount_threshold_std', 3):
                violations.append('unusual_transaction_amount')
                details['unusual_transaction_amount'] = {
                    'severity': 'medium',
                    'amount': amount,
                    'user_avg_amount': user_avg_amount,
                    'z_score': amount_z_score,
                    'threshold': rules['unusual_amount_threshold_std']
                }
        
        # Check new account behavior
        if account_age_days < rules.get('account_age_days_threshold', 30):
            violations.append('new_account_high_risk')
            details['new_account_high_risk'] = {
                'severity': 'medium',
                'account_age_days': account_age_days,
                'threshold': rules['account_age_days_threshold']
            }
        
        # Check insufficient history
        if historical_transaction_count < rules.get('min_historical_transactions', 10):
            violations.append('insufficient_transaction_history')
            details['insufficient_transaction_history'] = {
                'severity': 'low',
                'transaction_count': historical_transaction_count,
                'minimum_required': rules['min_historical_transactions']
            }
        
        return violations, details
    
    def _check_pattern_rules(self, rules: Dict[str, Any], evidence_data: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        """Check pattern-based rules"""
        violations = []
        details = {}
        
        amount = evidence_data.get('amount', 0)
        merchant_frequency = evidence_data.get('merchant_transaction_count_24h', 0)
        card_attempts = evidence_data.get('card_attempts_1h', 0)
        velocity_ratio = evidence_data.get('velocity_ratio', 1.0)
        
        # Check round number amounts (potential structuring)
        if amount > 0:
            if amount % 100 == 0 and amount >= 1000:  # Round hundreds or thousands
                round_suspicion = min(amount / 10000, 1.0)  # Higher amounts more suspicious
                if round_suspicion > rules.get('round_number_threshold', 0.8):
                    violations.append('suspicious_round_amount')
                    details['suspicious_round_amount'] = {
                        'severity': 'low',
                        'amount': amount,
                        'suspicion_score': round_suspicion,
                        'threshold': rules['round_number_threshold']
                    }
        
        # Check repeated merchant transactions
        if merchant_frequency > rules.get('repeated_merchant_threshold', 5):
            violations.append('excessive_merchant_frequency')
            details['excessive_merchant_frequency'] = {
                'severity': 'medium',
                'frequency': merchant_frequency,
                'threshold': rules['repeated_merchant_threshold']
            }
        
        # Check card testing patterns
        if card_attempts > rules.get('card_testing_threshold', 10):
            violations.append('potential_card_testing')
            details['potential_card_testing'] = {
                'severity': 'high',
                'attempts': card_attempts,
                'threshold': rules['card_testing_threshold']
            }
        
        # Check velocity burst patterns
        if velocity_ratio > rules.get('velocity_burst_threshold', 1.5):
            violations.append('velocity_burst_detected')
            details['velocity_burst_detected'] = {
                'severity': 'medium',
                'velocity_ratio': velocity_ratio,
                'threshold': rules['velocity_burst_threshold']
            }
        
        return violations, details
    
    def _check_compliance_rules(self, rules: Dict[str, Any], evidence_data: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        """Check compliance rules"""
        violations = []
        details = {}
        
        kyc_status = evidence_data.get('kyc_verified', False)
        aml_status = evidence_data.get('aml_cleared', False)
        sanctions_status = evidence_data.get('sanctions_cleared', False)
        pep_status = evidence_data.get('pep_cleared', False)
        
        # Check KYC compliance
        if rules.get('kyc_required', True) and not kyc_status:
            violations.append('kyc_verification_required')
            details['kyc_verification_required'] = {
                'severity': 'high',
                'status': kyc_status,
                'required': True
            }
        
        # Check AML compliance
        if rules.get('aml_screening_required', True) and not aml_status:
            violations.append('aml_screening_required')
            details['aml_screening_required'] = {
                'severity': 'critical',
                'status': aml_status,
                'required': True
            }
        
        # Check sanctions compliance
        if rules.get('sanctions_check_required', True) and not sanctions_status:
            violations.append('sanctions_check_required')
            details['sanctions_check_required'] = {
                'severity': 'critical',
                'status': sanctions_status,
                'required': True
            }
        
        # Check PEP compliance
        if rules.get('pep_check_required', True) and not pep_status:
            violations.append('pep_check_required')
            details['pep_check_required'] = {
                'severity': 'high',
                'status': pep_status,
                'required': True
            }
        
        return violations, details
    
    def _calculate_weighted_score(self, rule_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        if not rule_scores:
            return 0.0
        
        default_weights = {
            'transaction_limits': 1.5,
            'geographical_rules': 2.0,
            'behavioral_rules': 1.2,
            'pattern_rules': 1.3,
            'compliance_rules': 2.5
        }
        
        weights = {**default_weights, **self.rule_weights}
        
        weighted_sum = sum(score * weights.get(category, 1.0) for category, score in rule_scores.items())
        total_weight = sum(weights.get(category, 1.0) for category in rule_scores.keys())
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _determine_severity(self, overall_score: float, violated_rules: List[str]) -> str:
        """Determine severity based on score and rules"""
        critical_rules = [rule for rule in violated_rules if 'critical' in rule or 'sanctions' in rule or 'aml' in rule]
        high_rules = [rule for rule in violated_rules if 'impossible' in rule or 'restricted' in rule]
        
        if critical_rules or overall_score > 0.8:
            return 'critical'
        elif high_rules or overall_score > 0.6:
            return 'high'
        elif overall_score > 0.3:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_confidence_factors(self, evidence: List[EnhancedProofEvidence]) -> Dict[str, float]:
        """Calculate confidence factors from evidence"""
        if not evidence:
            return {}
        
        return {
            'evidence_count': len(evidence),
            'avg_confidence': np.mean([ev.confidence for ev in evidence]),
            'avg_reliability': np.mean([ev.reliability_score for ev in evidence]),
            'verified_evidence_ratio': sum(1 for ev in evidence if ev.verified) / len(evidence),
            'evidence_diversity': len(set(ev.evidence_type for ev in evidence)) / len(evidence) if evidence else 0
        }

class EnhancedMLProofSystem(BaseProofSystem):
    """Enhanced ML-based proof system with comprehensive validation"""
    
    def __init__(self, model_threshold: float = 0.7, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model_threshold = model_threshold
        self.ensemble_threshold = config.get('ensemble_threshold', 0.8) if config else 0.8
        self.drift_threshold = config.get('drift_threshold', 0.1) if config else 0.1
    
    @timeout_decorator(60)
    @validation_decorator(SecurityValidator.validate_input_safety)
    def generate_proof(self, claim: EnhancedProofClaim, evidence: List[EnhancedProofEvidence]) -> Dict[str, Any]:
        """Generate enhanced ML-based proof"""
        with self.resource_monitor.monitor_resources("ml_proof_generation"):
            self.validate_inputs(claim, evidence)
            
            # Extract ML predictions from evidence
            ml_predictions = []
            model_info = {}
            
            for ev in evidence:
                if ev.evidence_type == 'ml_prediction':
                    prediction_data = {
                        'model': ev.source,
                        'probability': ev.data.get('fraud_probability', 0),
                        'confidence': ev.confidence,
                        'reliability': ev.reliability_score,
                        'features_used': ev.data.get('features_used', []),
                        'model_version': ev.data.get('model_version', 'unknown'),
                        'timestamp': ev.timestamp.isoformat()
                    }
                    ml_predictions.append(prediction_data)
                    model_info[ev.source] = ev.data
            
            if not ml_predictions:
                return {
                    'proof_type': 'enhanced_ml_based',
                    'valid': False,
                    'reason': 'no_ml_predictions_available',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Analyze predictions
            analysis = self._analyze_predictions(ml_predictions)
            
            # Check for model drift
            drift_analysis = self._check_model_drift(ml_predictions, claim)
            
            # Calculate ensemble metrics
            ensemble_metrics = self._calculate_ensemble_metrics(ml_predictions)
            
            # Determine proof validity
            is_valid = (
                analysis['avg_probability'] > self.model_threshold and
                analysis['consensus_ratio'] > 0.5 and
                not drift_analysis['significant_drift']
            )
            
            return {
                'proof_type': 'enhanced_ml_based',
                'valid': is_valid,
                'predictions': ml_predictions,
                'analysis': analysis,
                'ensemble_metrics': ensemble_metrics,
                'drift_analysis': drift_analysis,
                'model_consensus': analysis['consensus_ratio'],
                'confidence_interval': analysis['confidence_interval'],
                'feature_importance': self._analyze_feature_importance(ml_predictions),
                'prediction_stability': self._calculate_prediction_stability(ml_predictions),
                'timestamp': datetime.now().isoformat(),
                'ml_engine_version': '2.0.0'
            }
    
    def verify_proof(self, proof: Dict[str, Any], claim: EnhancedProofClaim) -> Tuple[bool, float]:
        """Verify enhanced ML-based proof"""
        try:
            if proof.get('proof_type') != 'enhanced_ml_based':
                return False, 0.0
            
            if not proof.get('valid', False):
                return False, 0.0
            
            analysis = proof.get('analysis', {})
            ensemble_metrics = proof.get('ensemble_metrics', {})
            drift_analysis = proof.get('drift_analysis', {})
            
            # Check basic validity criteria
            avg_prob = analysis.get('avg_probability', 0)
            consensus = proof.get('model_consensus', 0)
            stability = proof.get('prediction_stability', {}).get('overall_stability', 0)
            
            if avg_prob < self.model_threshold:
                return False, 0.0
            
            if consensus < 0.5:
                return False, 0.0
            
            # Calculate confidence with multiple factors
            base_confidence = avg_prob * 100
            
            # Adjust for ensemble strength
            ensemble_factor = min(ensemble_metrics.get('ensemble_strength', 0.5) * 2, 1.0)
            
            # Adjust for prediction stability
            stability_factor = stability
            
            # Adjust for drift (penalty for high drift)
            drift_penalty = 1.0 - min(drift_analysis.get('max_drift_score', 0), 0.5)
            
            # Adjust for consensus
            consensus_factor = consensus
            
            # Calculate final confidence
            confidence = (
                base_confidence * 
                ensemble_factor * 
                stability_factor * 
                drift_penalty * 
                consensus_factor
            )
            
            return True, min(confidence, 100.0)
            
        except Exception as e:
            logger.error(f"ML proof verification failed: {str(e)}")
            return False, 0.0
    
    def _analyze_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze ML predictions comprehensively"""
        if not predictions:
            return {}
        
        probabilities = [p['probability'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        reliabilities = [p['reliability'] for p in predictions]
        
        # Basic statistics
        avg_probability = np.mean(probabilities)
        std_probability = np.std(probabilities)
        min_probability = np.min(probabilities)
        max_probability = np.max(probabilities)
        
        # Consensus analysis
        above_threshold = sum(1 for p in probabilities if p > self.model_threshold)
        consensus_ratio = above_threshold / len(predictions)
        
        # Confidence interval (assuming normal distribution)
        n = len(probabilities)
        se = std_probability / np.sqrt(n) if n > 1 else 0
        confidence_interval = {
            'lower': max(avg_probability - 1.96 * se, 0),
            'upper': min(avg_probability + 1.96 * se, 1),
            'margin_of_error': 1.96 * se
        }
        
        # Quality metrics
        avg_confidence = np.mean(confidences)
        avg_reliability = np.mean(reliabilities)
        
        return {
            'avg_probability': avg_probability,
            'std_probability': std_probability,
            'min_probability': min_probability,
            'max_probability': max_probability,
            'consensus_ratio': consensus_ratio,
            'confidence_interval': confidence_interval,
            'avg_model_confidence': avg_confidence,
            'avg_reliability': avg_reliability,
            'prediction_count': len(predictions),
            'probability_distribution': {
                'q25': np.percentile(probabilities, 25),
                'median': np.median(probabilities),
                'q75': np.percentile(probabilities, 75)
            }
        }
    
    def _check_model_drift(self, predictions: List[Dict[str, Any]], claim: EnhancedProofClaim) -> Dict[str, Any]:
        """Check for model drift indicators"""
        drift_scores = {}
        
        for prediction in predictions:
            model_name = prediction['model']
            model_probability = prediction['probability']
            baseline_probability = claim.fraud_probability  # Use claim as baseline
            
            # Calculate drift as difference from baseline
            if baseline_probability > 0:
                drift_score = abs(model_probability - baseline_probability) / baseline_probability
            else:
                drift_score = abs(model_probability)
            
            drift_scores[model_name] = drift_score
        
        max_drift = max(drift_scores.values()) if drift_scores else 0
        avg_drift = np.mean(list(drift_scores.values())) if drift_scores else 0
        
        return {
            'drift_scores': drift_scores,
            'max_drift_score': max_drift,
            'avg_drift_score': avg_drift,
            'significant_drift': max_drift > self.drift_threshold,
            'drift_threshold': self.drift_threshold,
            'models_with_drift': [model for model, score in drift_scores.items() if score > self.drift_threshold]
        }
    
    def _calculate_ensemble_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate ensemble-specific metrics"""
        if len(predictions) < 2:
            return {'ensemble_strength': 0, 'diversity': 0}
        
        probabilities = [p['probability'] for p in predictions]
        models = [p['model'] for p in predictions]
        
        # Calculate diversity (variance of predictions)
        diversity = np.var(probabilities)
        
        # Calculate ensemble strength (how much models agree on high-confidence predictions)
        high_conf_predictions = [p for p in probabilities if p > 0.7]
        if len(high_conf_predictions) > 1:
            ensemble_strength = 1 - np.std(high_conf_predictions)
        else:
            ensemble_strength = 0.5
        
        # Model diversity (unique models vs total predictions)
        model_diversity = len(set(models)) / len(models)
        
        return {
            'ensemble_strength': max(ensemble_strength, 0),
            'diversity': diversity,
            'model_diversity': model_diversity,
            'effective_ensemble_size': len(set(models))
        }
    
    def _analyze_feature_importance(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze feature importance across models"""
        all_features = []
        feature_counts = {}
        
        for prediction in predictions:
            features = prediction.get('features_used', [])
            all_features.extend(features)
            for feature in features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        if not all_features:
            return {}
        
        # Calculate feature importance by frequency across models
        total_predictions = len(predictions)
        feature_importance = {
            feature: count / total_predictions 
            for feature, count in feature_counts.items()
        }
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'feature_importance': dict(sorted_features),
            'top_features': [f[0] for f in sorted_features[:10]],
            'feature_coverage': len(feature_counts) / len(set(all_features)) if all_features else 0,
            'total_unique_features': len(set(all_features))
        }
    
    def _calculate_prediction_stability(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate prediction stability metrics"""
        if len(predictions) < 2:
            return {'overall_stability': 1.0}
        
        probabilities = [p['probability'] for p in predictions]
        
        # Calculate coefficient of variation
        mean_prob = np.mean(probabilities)
        std_prob = np.std(probabilities)
        cv = std_prob / mean_prob if mean_prob > 0 else float('inf')
        
        # Stability is inverse of coefficient of variation
        stability = 1 / (1 + cv) if cv != float('inf') else 0
        
        # Calculate pairwise stability
        pairwise_differences = []
        for i in range(len(probabilities)):
            for j in range(i + 1, len(probabilities)):
                diff = abs(probabilities[i] - probabilities[j])
                pairwise_differences.append(diff)
        
        avg_pairwise_diff = np.mean(pairwise_differences) if pairwise_differences else 0
        pairwise_stability = 1 - avg_pairwise_diff  # Higher differences = lower stability
        
        return {
            'overall_stability': stability,
            'coefficient_of_variation': cv,
            'pairwise_stability': max(pairwise_stability, 0),
            'avg_pairwise_difference': avg_pairwise_diff,
            'stability_rating': 'high' if stability > 0.8 else 'medium' if stability > 0.5 else 'low'
        }

class EnhancedCryptographicProofSystem(BaseProofSystem):
    """Enhanced cryptographic proof system with advanced security"""
    
    def __init__(self, secret_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.secret_key = secret_key or secrets.token_hex(32)
        self.hash_algorithm = config.get('hash_algorithm', 'sha256') if config else 'sha256'
        self.signature_algorithm = config.get('signature_algorithm', 'hmac') if config else 'hmac'
        
    @timeout_decorator(30)
    @validation_decorator(SecurityValidator.validate_input_safety)
    def generate_proof(self, claim: EnhancedProofClaim, evidence: List[EnhancedProofEvidence]) -> Dict[str, Any]:
        """Generate enhanced cryptographic proof"""
        with self.resource_monitor.monitor_resources("crypto_proof_generation"):
            self.validate_inputs(claim, evidence)
            
            try:
                # Create comprehensive proof data
                proof_data = {
                    'claim_id': claim.claim_id,
                    'transaction_id': claim.transaction_id,
                    'fraud_probability': claim.fraud_probability,
                    'risk_score': claim.risk_score,
                    'timestamp': claim.timestamp.isoformat(),
                    'evidence_hashes': [self._hash_evidence_enhanced(ev) for ev in evidence],
                    'evidence_count': len(evidence),
                    'claim_hash': self._hash_claim(claim),
                    'security_level': claim.security_level.value,
                    'nonce': secrets.token_hex(16)
                }
                
                # Generate proof hash using specified algorithm
                proof_string = json.dumps(proof_data, sort_keys=True)
                proof_hash = self._generate_hash(proof_string)
                
                # Generate enhanced signature
                signature_data = {
                    'proof_hash': proof_hash,
                    'timestamp': datetime.now().isoformat(),
                    'algorithm': self.signature_algorithm
                }
                signature = self._generate_signature(signature_data)
                
                # Generate merkle tree for evidence integrity
                merkle_root = self._generate_merkle_root([ev.calculate_integrity_hash() for ev in evidence])
                
                # Calculate proof strength
                proof_strength = self._calculate_proof_strength(claim, evidence)
                
                return {
                    'proof_type': 'enhanced_cryptographic',
                    'proof_data': proof_data,
                    'proof_hash': proof_hash,
                    'signature': signature,
                    'signature_algorithm': self.signature_algorithm,
                    'hash_algorithm': self.hash_algorithm,
                    'merkle_root': merkle_root,
                    'proof_strength': proof_strength,
                    'cryptographic_version': '2.0.0',
                    'security_level': claim.security_level.value,
                    'generation_timestamp': datetime.now().isoformat(),
                    'verification_required': True
                }
                
            except Exception as e:
                logger.error(f"Cryptographic proof generation failed: {str(e)}")
                raise CryptographicError(f"Failed to generate cryptographic proof: {str(e)}")
    
    def verify_proof(self, proof: Dict[str, Any], claim: EnhancedProofClaim) -> Tuple[bool, float]:
        """Verify enhanced cryptographic proof"""
        try:
            if proof.get('proof_type') != 'enhanced_cryptographic':
                return False, 0.0
            
            # Verify proof structure
            required_fields = ['proof_data', 'proof_hash', 'signature', 'merkle_root']
            for field in required_fields:
                if field not in proof:
                    logger.warning(f"Missing required field in cryptographic proof: {field}")
                    return False, 0.0
            
            # Verify proof hash
            proof_data = proof.get('proof_data', {})
            proof_string = json.dumps(proof_data, sort_keys=True)
            expected_hash = self._generate_hash(proof_string)
            
            if expected_hash != proof.get('proof_hash'):
                logger.warning("Proof hash verification failed")
                return False, 0.0
            
            # Verify signature
            signature_data = {
                'proof_hash': proof.get('proof_hash'),
                'timestamp': proof.get('generation_timestamp'),
                'algorithm': proof.get('signature_algorithm', self.signature_algorithm)
            }
            expected_signature = self._generate_signature(signature_data)
            
            if expected_signature != proof.get('signature'):
                logger.warning("Signature verification failed")
                return False, 0.0
            
            # Verify merkle root if evidence is available
            evidence_hashes = proof_data.get('evidence_hashes', [])
            if evidence_hashes:
                expected_merkle = self._generate_merkle_root(evidence_hashes)
                if expected_merkle != proof.get('merkle_root'):
                    logger.warning("Merkle root verification failed")
                    return False, 0.0
            
            # Calculate confidence based on proof strength and security level
            proof_strength = proof.get('proof_strength', 0.5)
            security_level = proof.get('security_level', 'standard')
            
            security_multipliers = {
                'basic': 0.7,
                'standard': 1.0,
                'enhanced': 1.2,
                'maximum': 1.5
            }
            
            base_confidence = 95.0  # High confidence for valid crypto proofs
            confidence = base_confidence * security_multipliers.get(security_level, 1.0) * proof_strength
            
            return True, min(confidence, 100.0)
            
        except Exception as e:
            logger.error(f"Cryptographic proof verification failed: {str(e)}")
            return False, 0.0
    
    def _hash_evidence_enhanced(self, evidence: EnhancedProofEvidence) -> str:
        """Generate enhanced hash of evidence"""
        evidence_data = {
            'evidence_id': evidence.evidence_id,
            'evidence_type': evidence.evidence_type,
            'timestamp': evidence.timestamp.isoformat(),
            'confidence': evidence.confidence,
            'reliability_score': evidence.reliability_score,
            'data_hash': self._generate_hash(json.dumps(evidence.data, sort_keys=True))
        }
        
        evidence_string = json.dumps(evidence_data, sort_keys=True)
        return self._generate_hash(evidence_string)
    
    def _hash_claim(self, claim: EnhancedProofClaim) -> str:
        """Generate hash of claim"""
        claim_data = {
            'claim_id': claim.claim_id,
            'transaction_id': claim.transaction_id,
            'fraud_probability': claim.fraud_probability,
            'risk_score': claim.risk_score,
            'claim_type': claim.claim_type.value,
            'confidence_level': claim.confidence_level.value
        }
        
        claim_string = json.dumps(claim_data, sort_keys=True)
        return self._generate_hash(claim_string)
    
    def _generate_hash(self, data: str) -> str:
        """Generate hash using specified algorithm"""
        if self.hash_algorithm == 'sha256':
            return hashlib.sha256(data.encode()).hexdigest()
        elif self.hash_algorithm == 'sha512':
            return hashlib.sha512(data.encode()).hexdigest()
        elif self.hash_algorithm == 'blake2b':
            return hashlib.blake2b(data.encode()).hexdigest()
        else:
            return hashlib.sha256(data.encode()).hexdigest()  # fallback
    
    def _generate_signature(self, data: Dict[str, Any]) -> str:
        """Generate signature using specified algorithm"""
        data_string = json.dumps(data, sort_keys=True)
        
        if self.signature_algorithm == 'hmac':
            return hmac.new(
                self.secret_key.encode(),
                data_string.encode(),
                hashlib.sha256
            ).hexdigest()
        else:
            # Fallback to simple hash-based signature
            return hashlib.sha256((data_string + self.secret_key).encode()).hexdigest()
    
    def _generate_merkle_root(self, hashes: List[str]) -> str:
        """Generate Merkle tree root for evidence integrity"""
        if not hashes:
            return ""
        
        if len(hashes) == 1:
            return hashes[0]
        
        # Build Merkle tree bottom-up
        current_level = hashes[:]
        
        while len(current_level) > 1:
            next_level = []
            
            # Process pairs
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    # Combine pair
                    combined = current_level[i] + current_level[i + 1]
                else:
                    # Odd number, duplicate last hash
                    combined = current_level[i] + current_level[i]
                
                next_level.append(self._generate_hash(combined))
            
            current_level = next_level
        
        return current_level[0]
    
    def _calculate_proof_strength(self, claim: EnhancedProofClaim, evidence: List[EnhancedProofEvidence]) -> float:
        """Calculate cryptographic proof strength"""
        factors = []
        
        # Evidence quantity factor
        evidence_factor = min(len(evidence) / 10, 1.0)  # Optimal around 10 pieces
        factors.append(evidence_factor)
        
        # Evidence quality factor
        if evidence:
            avg_confidence = np.mean([ev.confidence for ev in evidence])
            avg_reliability = np.mean([ev.reliability_score for ev in evidence])
            quality_factor = (avg_confidence + avg_reliability) / 2
            factors.append(quality_factor)
        
        # Claim confidence factor
        claim_factor = (claim.fraud_probability + claim.risk_score) / 2
        factors.append(claim_factor)
        
        # Security level factor
        security_factors = {
            SecurityLevel.BASIC: 0.7,
            SecurityLevel.STANDARD: 0.85,
            SecurityLevel.ENHANCED: 0.95,
            SecurityLevel.MAXIMUM: 1.0
        }
        security_factor = security_factors.get(claim.security_level, 0.85)
        factors.append(security_factor)
        
        # Calculate overall strength as weighted average
        return np.mean(factors) if factors else 0.5

# ============================================================================
# Enhanced Main Proof Verifier
# ============================================================================

class FinancialProofVerifier:
    """
    Enhanced Proof System Verifier for Financial Fraud Detection
    
    Advanced features:
    - Comprehensive validation and error handling
    - Multiple enhanced proof systems (rule-based, ML, cryptographic)
    - Security validation and access control
    - Resource monitoring and timeout protection
    - Detailed audit trails and metrics
    - Production-ready reliability mechanisms
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced proof verifier
        
        Args:
            config: Configuration dictionary with enhanced options
        """
        self.config = config or self._default_config()
        
        # Initialize enhanced proof systems
        self.proof_systems = {
            'enhanced_rule_based': EnhancedRuleBasedProofSystem(
                self.config.get('rules'),
                self.config.get('rule_system_config', {})
            ),
            'enhanced_ml_based': EnhancedMLProofSystem(
                self.config.get('ml_threshold', 0.7),
                self.config.get('ml_system_config', {})
            ),
            'enhanced_cryptographic': EnhancedCryptographicProofSystem(
                self.config.get('secret_key'),
                self.config.get('crypto_system_config', {})
            )
        }
        
        # Enhanced storage and monitoring
        self.proofs: Dict[str, EnhancedProofResult] = {}
        self.claims: Dict[str, EnhancedProofClaim] = {}
        self.evidence: Dict[str, List[EnhancedProofEvidence]] = {}
        
        # Security and validation components
        self.security_validator = SecurityValidator()
        self.resource_monitor = ResourceMonitor(
            self.config.get('max_memory_mb', 512),
            self.config.get('max_cpu_percent', 50.0)
        )
        
        # Enhanced metrics and monitoring
        self.metrics = self._initialize_enhanced_metrics()
        
        # Thread safety and concurrency
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 4),
            thread_name_prefix='ProofVerifier'
        )
        
        # Enhanced persistence
        self.storage_path = Path(self.config.get('storage_path', 'enhanced_proofs'))
        self.storage_path.mkdir(exist_ok=True)
        
        # Audit trail
        self.audit_trail: List[Dict[str, Any]] = []
        
        logger.info("Enhanced FinancialProofVerifier initialized with advanced security and monitoring")
    
    def _default_config(self) -> Dict[str, Any]:
        """Enhanced default configuration"""
        return {
            'ml_threshold': 0.7,
            'proof_validity_hours': 24,
            'max_evidence_per_claim': 100,
            'enable_cryptographic_proofs': True,
            'concurrent_verification': True,
            'auto_persist': True,
            'max_memory_mb': 512,
            'max_cpu_percent': 50.0,
            'max_workers': 4,
            'enable_audit_trail': True,
            'security_level': 'enhanced',
            'timeout_seconds': 60,
            'enable_resource_monitoring': True,
            'validation_strict_mode': True,
            'enable_drift_detection': True,
            'min_evidence_confidence': 0.3,
            'max_proof_age_hours': 168  # 1 week
        }
    
    def _initialize_enhanced_metrics(self) -> Dict[str, Any]:
        """Initialize enhanced metrics tracking"""
        return {
            'total_proofs_generated': 0,
            'total_proofs_verified': 0,
            'total_proofs_rejected': 0,
            'total_security_violations': 0,
            'total_timeouts': 0,
            'total_resource_limit_exceeded': 0,
            'average_generation_time_ms': 0.0,
            'average_verification_time_ms': 0.0,
            'verification_success_rate': 0.0,
            'system_performance': {
                'enhanced_rule_based': {'generated': 0, 'verified': 0, 'avg_time_ms': 0},
                'enhanced_ml_based': {'generated': 0, 'verified': 0, 'avg_time_ms': 0},
                'enhanced_cryptographic': {'generated': 0, 'verified': 0, 'avg_time_ms': 0}
            },
            'hourly_metrics': {},
            'error_metrics': {
                'validation_errors': 0,
                'security_errors': 0,
                'timeout_errors': 0,
                'resource_errors': 0
            }
        }
    
    @timeout_decorator(60)
    def generate_proof(self, claim: Dict[str, Any], 
                      evidence: Optional[List[Dict[str, Any]]] = None,
                      user_context: Optional[Dict[str, Any]] = None) -> Optional[EnhancedProofResult]:
        """
        Generate enhanced proof for fraud detection claim
        
        Args:
            claim: Fraud claim data
            evidence: Supporting evidence
            user_context: User context for security validation
            
        Returns:
            Enhanced proof result or None if generation fails
        """
        start_time = time.time()
        operation_id = self._generate_id('OP')
        
        try:
            # Add audit entry
            self._add_audit_entry('proof_generation_started', {
                'operation_id': operation_id,
                'claim_id': claim.get('claim_id'),
                'user_context': user_context
            })
            
            # Security validation
            if user_context:
                is_authorized, auth_msg = self.security_validator.validate_proof_permissions(
                    user_context, 'generate_proof'
                )
                if not is_authorized:
                    raise ProofSecurityError(f"Permission denied: {auth_msg}")
            
            # Resource monitoring
            with self.resource_monitor.monitor_resources("proof_generation"):
                # Input validation
                is_safe, safety_msg = self.security_validator.validate_input_safety(claim)
                if not is_safe:
                    raise ProofSecurityError(f"Claim security validation failed: {safety_msg}")
                
                # Create enhanced claim object
                claim_obj = self._create_enhanced_claim(claim, user_context)
                
                # Validate claim
                is_valid, errors = claim_obj.validate()
                if not is_valid:
                    raise ClaimValidationError(f"Claim validation failed: {', '.join(errors)}")
                
                # Collect and validate evidence
                evidence_objs = self._collect_enhanced_evidence(claim_obj, evidence)
                
                # Store claim and evidence
                with self._lock:
                    self.claims[claim_obj.claim_id] = claim_obj
                    self.evidence[claim_obj.claim_id] = evidence_objs
                
                # Generate proofs using enhanced systems
                proofs = {}
                proof_scores = {}
                generation_details = {}
                
                for system_name, system in self.proof_systems.items():
                    if self._should_use_enhanced_system(system_name, claim_obj):
                        try:
                            system_start = time.time()
                            proof = system.generate_proof(claim_obj, evidence_objs)
                            system_time = (time.time() - system_start) * 1000
                            
                            proofs[system_name] = proof
                            generation_details[system_name] = {'time_ms': system_time}
                            
                            # Verify the generated proof
                            is_valid, confidence = system.verify_proof(proof, claim_obj)
                            if is_valid:
                                proof_scores[system_name] = confidence
                                
                            # Update system metrics
                            self._update_system_metrics(system_name, 'generation', system_time)
                            
                        except Exception as e:
                            logger.error(f"Enhanced proof generation failed for {system_name}: {str(e)}")
                            generation_details[system_name] = {'error': str(e)}
                            self._update_error_metrics('generation_error')
                
                if not proofs:
                    logger.warning(f"No proofs generated for claim {claim_obj.claim_id}")
                    return None
                
                # Create enhanced proof result
                proof_result = self._create_enhanced_proof_result(
                    claim_obj, proofs, proof_scores, generation_details
                )
                
                # Update metrics
                generation_time = (time.time() - start_time) * 1000
                self._update_metrics('generation', generation_time, proof_result.status)
                
                # Store proof
                with self._lock:
                    self.proofs[proof_result.proof_id] = proof_result
                
                # Persist if configured
                if self.config.get('auto_persist'):
                    self._persist_enhanced_proof(proof_result)
                
                # Add completion audit entry
                self._add_audit_entry('proof_generation_completed', {
                    'operation_id': operation_id,
                    'proof_id': proof_result.proof_id,
                    'status': proof_result.status.value,
                    'confidence': proof_result.confidence,
                    'generation_time_ms': generation_time
                })
                
                logger.info(f"Enhanced proof {proof_result.proof_id} generated for claim {claim_obj.claim_id} "
                           f"with confidence {proof_result.confidence:.2f}")
                
                return proof_result
                
        except ProofVerificationException as e:
            logger.error(f"Proof generation failed with known error: {str(e)}")
            self._update_error_metrics(e.__class__.__name__)
            self._add_audit_entry('proof_generation_failed', {
                'operation_id': operation_id,
                'error': str(e),
                'error_type': e.__class__.__name__
            })
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error in proof generation: {str(e)}")
            self._update_error_metrics('unexpected_error')
            self._add_audit_entry('proof_generation_error', {
                'operation_id': operation_id,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            return None
    
    @timeout_decorator(30)
    def verify_proof(self, proof: Dict[str, Any], 
                    user_context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Verify enhanced fraud detection proof
        
        Args:
            proof: Proof data to verify
            user_context: User context for security validation
            
        Returns:
            Tuple of (is_valid, message, details)
        """
        start_time = time.time()
        operation_id = self._generate_id('OP')
        
        try:
            # Add audit entry
            self._add_audit_entry('proof_verification_started', {
                'operation_id': operation_id,
                'proof_id': proof.get('proof_id'),
                'user_context': user_context
            })
            
            # Security validation
            if user_context:
                is_authorized, auth_msg = self.security_validator.validate_proof_permissions(
                    user_context, 'verify_proof'
                )
                if not is_authorized:
                    raise ProofSecurityError(f"Permission denied: {auth_msg}")
            
            # Resource monitoring
            with self.resource_monitor.monitor_resources("proof_verification"):
                # Input safety validation
                is_safe, safety_msg = self.security_validator.validate_input_safety(proof)
                if not is_safe:
                    raise ProofSecurityError(f"Proof security validation failed: {safety_msg}")
                
                # Extract proof ID
                proof_id = proof.get('proof_id')
                if not proof_id:
                    return False, "Missing proof ID", {}
                
                # Check if proof exists
                with self._lock:
                    stored_proof = self.proofs.get(proof_id)
                
                if not stored_proof:
                    # Try loading from storage
                    stored_proof = self._load_enhanced_proof(proof_id)
                    if not stored_proof:
                        return False, "Proof not found", {}
                
                # Check validity and expiration
                if not stored_proof.is_valid():
                    return False, f"Proof expired or invalid (status: {stored_proof.status.value})", {
                        'status': stored_proof.status.value,
                        'valid_until': stored_proof.valid_until.isoformat() if stored_proof.valid_until else None,
                        'revoked': stored_proof.revoked
                    }
                
                # Enhanced integrity verification
                integrity_result = self._verify_enhanced_proof_integrity(proof, stored_proof)
                if not integrity_result['valid']:
                    return False, f"Proof integrity check failed: {integrity_result['reason']}", integrity_result
                
                # Re-verify with enhanced proof systems if configured
                reverification_results = {}
                if self.config.get('reverify_on_check'):
                    claim = self.claims.get(stored_proof.claim_id)
                    if claim:
                        for system_name in stored_proof.reasoning.get('systems_used', []):
                            system = self.proof_systems.get(system_name)
                            if system:
                                system_proof = stored_proof.reasoning.get('proofs', {}).get(system_name, {})
                                is_valid, confidence = system.verify_proof(system_proof, claim)
                                reverification_results[system_name] = {
                                    'valid': is_valid,
                                    'confidence': confidence
                                }
                                if not is_valid:
                                    return False, f"{system_name} re-verification failed", {
                                        'reverification_results': reverification_results
                                    }
                
                # Update metrics
                verification_time = (time.time() - start_time) * 1000
                self._update_metrics('verification', verification_time, ProofStatus.VERIFIED)
                
                # Add to audit trail
                stored_proof.add_audit_entry('verification_success', {
                    'operation_id': operation_id,
                    'verification_time_ms': verification_time,
                    'reverification_results': reverification_results
                })
                
                # Add completion audit entry
                self._add_audit_entry('proof_verification_completed', {
                    'operation_id': operation_id,
                    'proof_id': proof_id,
                    'result': 'valid',
                    'verification_time_ms': verification_time
                })
                
                verification_details = {
                    'proof_status': stored_proof.status.value,
                    'confidence': stored_proof.confidence,
                    'security_level': stored_proof.security_level.value,
                    'verification_time_ms': verification_time,
                    'integrity_check': integrity_result,
                    'reverification_results': reverification_results
                }
                
                return True, "Enhanced proof verified successfully", verification_details
                
        except ProofVerificationException as e:
            logger.error(f"Proof verification failed with known error: {str(e)}")
            self._update_error_metrics(e.__class__.__name__)
            self._add_audit_entry('proof_verification_failed', {
                'operation_id': operation_id,
                'error': str(e),
                'error_type': e.__class__.__name__
            })
            return False, f"Verification error: {str(e)}", {'error_type': e.__class__.__name__}
            
        except Exception as e:
            logger.error(f"Unexpected error in proof verification: {str(e)}")
            self._update_error_metrics('unexpected_error')
            self._add_audit_entry('proof_verification_error', {
                'operation_id': operation_id,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            return False, f"Verification error: {str(e)}", {'error_type': 'unexpected_error'}
    
    def validate_claim(self, claim: Dict[str, Any], 
                      user_context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]:
        """
        Enhanced fraud detection claim validation
        
        Args:
            claim: Claim data to validate
            user_context: User context for security validation
            
        Returns:
            Tuple of (is_valid, message, detailed_errors)
        """
        try:
            # Security validation
            if user_context:
                is_authorized, auth_msg = self.security_validator.validate_proof_permissions(
                    user_context, 'validate_claim'
                )
                if not is_authorized:
                    return False, f"Permission denied: {auth_msg}", []
            
            # Input safety validation
            is_safe, safety_msg = self.security_validator.validate_input_safety(claim)
            if not is_safe:
                return False, f"Claim security validation failed: {safety_msg}", []
            
            errors = []
            
            # Check required fields
            required_fields = ['transaction_id', 'fraud_probability', 'risk_score']
            for field in required_fields:
                if field not in claim:
                    errors.append(f"Missing required field: {field}")
            
            # Enhanced validation rules
            fraud_prob = claim.get('fraud_probability', 0)
            risk_score = claim.get('risk_score', 0)
            
            if not 0 <= fraud_prob <= 1:
                errors.append("Fraud probability must be between 0 and 1")
            
            if not 0 <= risk_score <= 1:
                errors.append("Risk score must be between 0 and 1")
            
            # Validate evidence if provided
            evidence = claim.get('evidence', {})
            if evidence and not isinstance(evidence, dict):
                errors.append("Evidence must be a dictionary")
            
            # Enhanced security checks
            if evidence:
                is_evidence_safe, evidence_safety_msg = self.security_validator.validate_input_safety(evidence)
                if not is_evidence_safe:
                    errors.append(f"Evidence security validation failed: {evidence_safety_msg}")
            
            # Business logic validation
            if fraud_prob > 0.9 and len(evidence) == 0:
                errors.append("High fraud probability requires supporting evidence")
            
            # Check for consistency
            if abs(fraud_prob - risk_score) > 0.5:
                errors.append("Fraud probability and risk score are inconsistent")
            
            # Validate claim type if provided
            claim_type = claim.get('claim_type')
            if claim_type and claim_type not in [pt.value for pt in ProofType]:
                errors.append(f"Invalid claim type: {claim_type}")
            
            # Validate security level if provided
            security_level = claim.get('security_level')
            if security_level and security_level not in [sl.value for sl in SecurityLevel]:
                errors.append(f"Invalid security level: {security_level}")
            
            is_valid = len(errors) == 0
            message = "Enhanced claim validation passed" if is_valid else "Enhanced claim validation failed"
            
            return is_valid, message, errors
            
        except Exception as e:
            logger.error(f"Enhanced claim validation failed: {str(e)}")
            return False, f"Validation error: {str(e)}", [str(e)]
    
    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get comprehensive enhanced metrics"""
        with self._lock:
            current_time = datetime.now()
            
            # Calculate additional metrics
            active_proofs = len([p for p in self.proofs.values() if p.is_valid()])
            expired_proofs = len([p for p in self.proofs.values() if not p.is_valid()])
            
            # Calculate system health metrics
            resource_status = self.resource_monitor.check_resource_limits()
            
            return {
                'summary': {
                    'total_proofs_generated': self.metrics['total_proofs_generated'],
                    'total_proofs_verified': self.metrics['total_proofs_verified'],
                    'total_proofs_rejected': self.metrics['total_proofs_rejected'],
                    'verification_success_rate': self.metrics['verification_success_rate'],
                    'active_proofs': active_proofs,
                    'expired_proofs': expired_proofs
                },
                'performance': {
                    'avg_generation_time_ms': self.metrics['average_generation_time_ms'],
                    'avg_verification_time_ms': self.metrics['average_verification_time_ms'],
                    'system_performance': self.metrics['system_performance']
                },
                'security': {
                    'total_security_violations': self.metrics['total_security_violations'],
                    'resource_limit_exceeded': self.metrics['total_resource_limit_exceeded'],
                    'timeout_errors': self.metrics['total_timeouts']
                },
                'errors': self.metrics['error_metrics'],
                'system_health': {
                    'resource_status': resource_status,
                    'storage_path_exists': self.storage_path.exists(),
                    'active_workers': self._executor._threads if hasattr(self._executor, '_threads') else 0
                },
                'timestamp': current_time.isoformat()
            }
    
    def cleanup_expired_proofs(self) -> Dict[str, int]:
        """Enhanced cleanup of expired proofs"""
        removed_counts = {'expired': 0, 'revoked': 0, 'old': 0}
        current_time = datetime.now()
        max_age = timedelta(hours=self.config.get('max_proof_age_hours', 168))
        
        with self._lock:
            expired_ids = []
            for proof_id, proof in self.proofs.items():
                age = current_time - proof.timestamp
                
                if proof.revoked:
                    expired_ids.append((proof_id, 'revoked'))
                elif not proof.is_valid() and proof.status == ProofStatus.EXPIRED:
                    expired_ids.append((proof_id, 'expired'))
                elif age > max_age:
                    expired_ids.append((proof_id, 'old'))
            
            for proof_id, reason in expired_ids:
                del self.proofs[proof_id]
                removed_counts[reason] += 1
                
                # Remove from storage if exists
                proof_file = self.storage_path / f"{proof_id}.pkl"
                if proof_file.exists():
                    proof_file.unlink()
        
        total_removed = sum(removed_counts.values())
        logger.info(f"Enhanced cleanup removed {total_removed} proofs: {removed_counts}")
        
        # Add audit entry
        self._add_audit_entry('proof_cleanup', {
            'removed_counts': removed_counts,
            'total_removed': total_removed
        })
        
        return removed_counts
    
    def export_enhanced_audit_trail(self) -> List[Dict[str, Any]]:
        """Export complete audit trail"""
        with self._lock:
            return self.audit_trail.copy()
    
    def _create_enhanced_claim(self, claim_data: Dict[str, Any], 
                             user_context: Optional[Dict[str, Any]]) -> EnhancedProofClaim:
        """Create enhanced claim object from data"""
        return EnhancedProofClaim(
            claim_id=claim_data.get('claim_id', self._generate_id('CLAIM')),
            claim_type=ProofType(claim_data.get('claim_type', ProofType.TRANSACTION_FRAUD.value)),
            transaction_id=claim_data['transaction_id'],
            timestamp=datetime.now(),
            fraud_probability=claim_data['fraud_probability'],
            risk_score=claim_data['risk_score'],
            confidence_level=ProofLevel(claim_data.get('confidence_level', ProofLevel.MEDIUM.value)),
            evidence=claim_data.get('evidence', {}),
            violated_rules=claim_data.get('violated_rules', []),
            rule_severity=claim_data.get('rule_severity', {}),
            model_version=claim_data.get('model_version'),
            model_confidence=claim_data.get('model_confidence', 0),
            model_features=claim_data.get('model_features', []),
            security_level=SecurityLevel(claim_data.get('security_level', SecurityLevel.STANDARD.value)),
            user_context=user_context or {},
            system_context=claim_data.get('system_context', {}),
            metadata=claim_data.get('metadata', {}),
            tags=claim_data.get('tags', [])
        )
    
    def _collect_enhanced_evidence(self, claim: EnhancedProofClaim, 
                                 evidence_data: Optional[List[Dict[str, Any]]]) -> List[EnhancedProofEvidence]:
        """Collect and validate enhanced evidence"""
        evidence_list = []
        
        # Add ML prediction evidence
        if claim.model_version:
            evidence_list.append(EnhancedProofEvidence(
                evidence_id=self._generate_id('EVD'),
                evidence_type='ml_prediction',
                source=claim.model_version,
                timestamp=datetime.now(),
                data={
                    'fraud_probability': claim.fraud_probability,
                    'risk_score': claim.risk_score,
                    'model_confidence': claim.model_confidence,
                    'features_used': claim.model_features
                },
                confidence=claim.model_confidence,
                reliability_score=0.9,  # ML predictions generally reliable
                security_level=claim.security_level
            ))
        
        # Add rule violation evidence
        for rule in claim.violated_rules:
            severity = claim.rule_severity.get(rule, 'medium')
            confidence_map = {'low': 0.6, 'medium': 0.8, 'high': 0.95, 'critical': 1.0}
            
            evidence_list.append(EnhancedProofEvidence(
                evidence_id=self._generate_id('EVD'),
                evidence_type='rule_violation',
                source='enhanced_rule_engine',
                timestamp=datetime.now(),
                data={'rule': rule, 'violated': True, 'severity': severity},
                confidence=confidence_map.get(severity, 0.8),
                reliability_score=1.0,  # Rule violations are definitive
                security_level=claim.security_level
            ))
        
        # Add custom evidence with enhanced validation
        if evidence_data:
            for ev in evidence_data:
                # Validate evidence data
                is_safe, safety_msg = self.security_validator.validate_input_safety(ev.get('data', {}))
                if not is_safe:
                    logger.warning(f"Skipping unsafe evidence: {safety_msg}")
                    continue
                
                enhanced_evidence = EnhancedProofEvidence(
                    evidence_id=ev.get('evidence_id', self._generate_id('EVD')),
                    evidence_type=ev.get('type', 'custom'),
                    source=ev.get('source', 'unknown'),
                    timestamp=datetime.now(),
                    data=ev.get('data', {}),
                    confidence=ev.get('confidence', 0.5),
                    reliability_score=ev.get('reliability_score', 0.7),
                    security_level=SecurityLevel(ev.get('security_level', claim.security_level.value)),
                    metadata=ev.get('metadata', {})
                )
                
                # Calculate integrity hash
                enhanced_evidence.integrity_hash = enhanced_evidence.calculate_integrity_hash()
                
                evidence_list.append(enhanced_evidence)
        
        # Filter by minimum confidence if configured
        min_confidence = self.config.get('min_evidence_confidence', 0.3)
        evidence_list = [ev for ev in evidence_list if ev.confidence >= min_confidence]
        
        # Limit evidence count
        max_evidence = self.config.get('max_evidence_per_claim', 100)
        if len(evidence_list) > max_evidence:
            # Sort by confidence and reliability, keep top evidence
            evidence_list.sort(key=lambda x: (x.confidence * x.reliability_score), reverse=True)
            evidence_list = evidence_list[:max_evidence]
        
        return evidence_list
    
    def _should_use_enhanced_system(self, system_name: str, claim: EnhancedProofClaim) -> bool:
        """Determine if enhanced proof system should be used for claim"""
        # Enhanced rule-based for rule violations
        if system_name == 'enhanced_rule_based' and claim.violated_rules:
            return True
        
        # Enhanced ML-based for ML predictions
        if system_name == 'enhanced_ml_based' and claim.model_version:
            return True
        
        # Enhanced cryptographic for high-security claims
        if system_name == 'enhanced_cryptographic':
            return (
                self.config.get('enable_cryptographic_proofs', True) and
                (claim.fraud_probability > 0.8 or 
                 claim.security_level in [SecurityLevel.ENHANCED, SecurityLevel.MAXIMUM])
            )
        
        return False
    
    def _create_enhanced_proof_result(self, claim: EnhancedProofClaim, 
                                    proofs: Dict[str, Dict[str, Any]], 
                                    scores: Dict[str, float],
                                    generation_details: Dict[str, Dict[str, Any]]) -> EnhancedProofResult:
        """Create enhanced composite proof result"""
        # Calculate overall confidence with advanced weighting
        if scores:
            # Weight scores by system reliability
            system_weights = {
                'enhanced_rule_based': 1.2,  # Rules are definitive
                'enhanced_ml_based': 1.0,    # ML is standard
                'enhanced_cryptographic': 1.5  # Crypto is most secure
            }
            
            weighted_scores = []
            for system, score in scores.items():
                weight = system_weights.get(system, 1.0)
                weighted_scores.append(score * weight)
            
            overall_confidence = np.mean(weighted_scores)
        else:
            overall_confidence = 0.0
        
        # Determine enhanced status
        if overall_confidence >= 90:
            status = ProofStatus.VERIFIED
        elif overall_confidence >= 70:
            status = ProofStatus.VERIFIED  # Still verified but lower confidence
        elif overall_confidence >= 40:
            status = ProofStatus.PARTIAL
        else:
            status = ProofStatus.REJECTED
        
        # Calculate quality and reliability scores
        quality_score = self._calculate_quality_score(claim, proofs, scores)
        reliability_score = self._calculate_reliability_score(claim, proofs)
        
        # Create enhanced proof hash
        proof_data = {
            'claim_id': claim.claim_id,
            'proofs': proofs,
            'scores': scores,
            'generation_details': generation_details,
            'timestamp': datetime.now().isoformat(),
            'security_level': claim.security_level.value
        }
        proof_hash = hashlib.sha256(
            json.dumps(proof_data, sort_keys=True).encode()
        ).hexdigest()
        
        # Calculate validity period based on security level
        validity_hours_map = {
            SecurityLevel.BASIC: 12,
            SecurityLevel.STANDARD: 24,
            SecurityLevel.ENHANCED: 48,
            SecurityLevel.MAXIMUM: 72
        }
        validity_hours = validity_hours_map.get(claim.security_level, 24)
        valid_until = datetime.now() + timedelta(hours=validity_hours)
        
        return EnhancedProofResult(
            proof_id=self._generate_id('PROOF'),
            claim_id=claim.claim_id,
            status=status,
            confidence=overall_confidence,
            timestamp=datetime.now(),
            verification_method='enhanced_composite',
            verification_time_ms=0,  # Will be updated
            verification_steps=[f"Enhanced {system}" for system in proofs.keys()],
            evidence_used=[e.evidence_id for e in self.evidence.get(claim.claim_id, [])],
            reasoning={
                'systems_used': list(proofs.keys()),
                'individual_scores': scores,
                'proofs': proofs,
                'generation_details': generation_details,
                'weighted_calculation': True
            },
            decision_tree={
                'confidence_threshold_met': overall_confidence >= 40,
                'multiple_systems_agree': len(scores) > 1,
                'high_security_validated': claim.security_level in [SecurityLevel.ENHANCED, SecurityLevel.MAXIMUM],
                'evidence_sufficient': len(self.evidence.get(claim.claim_id, [])) >= 1
            },
            proof_hash=proof_hash,
            security_level=claim.security_level,
            valid_until=valid_until,
            quality_score=quality_score,
            reliability_score=reliability_score
        )
    
    def _verify_enhanced_proof_integrity(self, provided_proof: Dict[str, Any], 
                                       stored_proof: EnhancedProofResult) -> Dict[str, Any]:
        """Enhanced proof integrity verification"""
        integrity_result = {
            'valid': True,
            'reason': '',
            'checks_performed': []
        }
        
        # Check proof hash if provided
        if 'proof_hash' in provided_proof:
            integrity_result['checks_performed'].append('proof_hash')
            if provided_proof['proof_hash'] != stored_proof.proof_hash:
                integrity_result['valid'] = False
                integrity_result['reason'] = 'Proof hash mismatch'
                return integrity_result
        
        # Check basic fields
        integrity_result['checks_performed'].append('basic_fields')
        if not (provided_proof.get('proof_id') == stored_proof.proof_id and
                provided_proof.get('claim_id') == stored_proof.claim_id):
            integrity_result['valid'] = False
            integrity_result['reason'] = 'Basic field mismatch'
            return integrity_result
        
        # Check integrity hash if available
        if stored_proof.proof_hash and not stored_proof.integrity_verified:
            integrity_result['checks_performed'].append('integrity_verification')
            # Re-calculate and verify proof hash
            expected_hash = self._recalculate_proof_hash(stored_proof)
            if expected_hash != stored_proof.proof_hash:
                integrity_result['valid'] = False
                integrity_result['reason'] = 'Proof integrity hash verification failed'
                return integrity_result
            else:
                stored_proof.integrity_verified = True
        
        # Check for tampering indicators
        integrity_result['checks_performed'].append('tampering_detection')
        if self._detect_tampering(stored_proof):
            integrity_result['valid'] = False
            integrity_result['reason'] = 'Tampering detected'
            return integrity_result
        
        return integrity_result
    
    def _calculate_quality_score(self, claim: EnhancedProofClaim, 
                               proofs: Dict[str, Dict[str, Any]], 
                               scores: Dict[str, float]) -> float:
        """Calculate proof quality score"""
        quality_factors = []
        
        # Evidence quality
        evidence_objs = self.evidence.get(claim.claim_id, [])
        if evidence_objs:
            avg_evidence_confidence = np.mean([ev.confidence for ev in evidence_objs])
            avg_evidence_reliability = np.mean([ev.reliability_score for ev in evidence_objs])
            evidence_quality = (avg_evidence_confidence + avg_evidence_reliability) / 2
            quality_factors.append(evidence_quality)
        
        # Proof system diversity
        diversity_factor = len(proofs) / len(self.proof_systems)
        quality_factors.append(diversity_factor)
        
        # Score consistency
        if len(scores) > 1:
            score_std = np.std(list(scores.values()))
            consistency_factor = 1 - min(score_std / 100, 1.0)  # Lower std = higher consistency
            quality_factors.append(consistency_factor)
        
        # Security level factor
        security_factors = {
            SecurityLevel.BASIC: 0.7,
            SecurityLevel.STANDARD: 0.85,
            SecurityLevel.ENHANCED: 0.95,
            SecurityLevel.MAXIMUM: 1.0
        }
        security_factor = security_factors.get(claim.security_level, 0.85)
        quality_factors.append(security_factor)
        
        return np.mean(quality_factors) if quality_factors else 0.5
    
    def _calculate_reliability_score(self, claim: EnhancedProofClaim, 
                                   proofs: Dict[str, Dict[str, Any]]) -> float:
        """Calculate proof reliability score"""
        reliability_factors = []
        
        # Claim consistency
        prob_risk_consistency = 1 - abs(claim.fraud_probability - claim.risk_score)
        reliability_factors.append(prob_risk_consistency)
        
        # Model confidence if available
        if claim.model_confidence > 0:
            reliability_factors.append(claim.model_confidence)
        
        # Rule violation confidence
        if claim.violated_rules:
            rule_reliability = min(len(claim.violated_rules) / 5, 1.0)  # More rules = higher reliability
            reliability_factors.append(rule_reliability)
        
        # Proof system agreement
        if len(proofs) > 1:
            # All systems generated proofs, indicating agreement
            reliability_factors.append(0.9)
        
        return np.mean(reliability_factors) if reliability_factors else 0.5
    
    def _recalculate_proof_hash(self, proof: EnhancedProofResult) -> str:
        """Recalculate proof hash for integrity verification"""
        proof_data = {
            'claim_id': proof.claim_id,
            'status': proof.status.value,
            'confidence': proof.confidence,
            'timestamp': proof.timestamp.isoformat(),
            'reasoning': proof.reasoning
        }
        
        proof_string = json.dumps(proof_data, sort_keys=True)
        return hashlib.sha256(proof_string.encode()).hexdigest()
    
    def _detect_tampering(self, proof: EnhancedProofResult) -> bool:
        """Detect potential tampering in proof"""
        # Check for suspicious patterns
        
        # Confidence too high for low evidence
        evidence_count = len(proof.evidence_used)
        if proof.confidence > 95 and evidence_count < 2:
            return True
        
        # Status inconsistent with confidence
        if proof.status == ProofStatus.VERIFIED and proof.confidence < 40:
            return True
        
        if proof.status == ProofStatus.REJECTED and proof.confidence > 80:
            return True
        
        # Audit trail manipulation
        if len(proof.audit_trail) == 0 and proof.timestamp < datetime.now() - timedelta(hours=1):
            return True
        
        return False
    
    def _persist_enhanced_proof(self, proof: EnhancedProofResult) -> None:
        """Persist enhanced proof to storage"""
        try:
            proof_file = self.storage_path / f"{proof.proof_id}.pkl"
            with open(proof_file, 'wb') as f:
                pickle.dump(proof, f)
            logger.debug(f"Persisted enhanced proof {proof.proof_id}")
        except Exception as e:
            logger.error(f"Failed to persist enhanced proof: {str(e)}")
            raise ProofStorageError(f"Failed to persist proof: {str(e)}")
    
    def _load_enhanced_proof(self, proof_id: str) -> Optional[EnhancedProofResult]:
        """Load enhanced proof from storage"""
        try:
            proof_file = self.storage_path / f"{proof_id}.pkl"
            if proof_file.exists():
                with open(proof_file, 'rb') as f:
                    proof = pickle.load(f)
                
                # Cache in memory
                with self._lock:
                    self.proofs[proof_id] = proof
                
                return proof
        except Exception as e:
            logger.error(f"Failed to load enhanced proof: {str(e)}")
            raise ProofStorageError(f"Failed to load proof: {str(e)}")
        
        return None
    
    def _update_metrics(self, operation: str, time_ms: float, status: ProofStatus) -> None:
        """Update enhanced metrics"""
        with self._lock:
            if operation == 'generation':
                self.metrics['total_proofs_generated'] += 1
                
                # Update average time
                total = self.metrics['total_proofs_generated']
                current_avg = self.metrics['average_generation_time_ms']
                self.metrics['average_generation_time_ms'] = (
                    (current_avg * (total - 1) + time_ms) / total
                )
                
            elif operation == 'verification':
                self.metrics['total_proofs_verified'] += 1
                
                if status == ProofStatus.REJECTED:
                    self.metrics['total_proofs_rejected'] += 1
                
                # Update average time
                total = self.metrics['total_proofs_verified']
                current_avg = self.metrics['average_verification_time_ms']
                self.metrics['average_verification_time_ms'] = (
                    (current_avg * (total - 1) + time_ms) / total
                )
                
                # Update success rate
                if self.metrics['total_proofs_verified'] > 0:
                    self.metrics['verification_success_rate'] = (
                        (self.metrics['total_proofs_verified'] - self.metrics['total_proofs_rejected']) /
                        self.metrics['total_proofs_verified']
                    )
            
            # Update hourly metrics
            hour_key = datetime.now().strftime('%Y-%m-%d_%H')
            if hour_key not in self.metrics['hourly_metrics']:
                self.metrics['hourly_metrics'][hour_key] = 0
            self.metrics['hourly_metrics'][hour_key] += 1
    
    def _update_system_metrics(self, system_name: str, operation: str, time_ms: float) -> None:
        """Update system-specific metrics"""
        with self._lock:
            if system_name not in self.metrics['system_performance']:
                self.metrics['system_performance'][system_name] = {
                    'generated': 0, 'verified': 0, 'avg_time_ms': 0
                }
            
            system_metrics = self.metrics['system_performance'][system_name]
            
            if operation == 'generation':
                system_metrics['generated'] += 1
            elif operation == 'verification':
                system_metrics['verified'] += 1
            
            # Update average time
            total_ops = system_metrics['generated'] + system_metrics['verified']
            if total_ops > 0:
                current_avg = system_metrics['avg_time_ms']
                system_metrics['avg_time_ms'] = (
                    (current_avg * (total_ops - 1) + time_ms) / total_ops
                )
    
    def _update_error_metrics(self, error_type: str) -> None:
        """Update error metrics"""
        with self._lock:
            if 'security' in error_type.lower():
                self.metrics['total_security_violations'] += 1
                self.metrics['error_metrics']['security_errors'] += 1
            elif 'timeout' in error_type.lower():
                self.metrics['total_timeouts'] += 1
                self.metrics['error_metrics']['timeout_errors'] += 1
            elif 'resource' in error_type.lower():
                self.metrics['total_resource_limit_exceeded'] += 1
                self.metrics['error_metrics']['resource_errors'] += 1
            elif 'validation' in error_type.lower():
                self.metrics['error_metrics']['validation_errors'] += 1
    
    def _add_audit_entry(self, action: str, details: Dict[str, Any]) -> None:
        """Add entry to audit trail"""
        if not self.config.get('enable_audit_trail', True):
            return
        
        with self._lock:
            audit_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': action,
                'details': details,
                'audit_id': self._generate_id('AUDIT')
            }
            
            self.audit_trail.append(audit_entry)
            
            # Limit audit trail size
            max_audit_entries = self.config.get('max_audit_entries', 10000)
            if len(self.audit_trail) > max_audit_entries:
                self.audit_trail = self.audit_trail[-max_audit_entries:]
    
    def _generate_id(self, prefix: str) -> str:
        """Generate cryptographically secure unique ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_part = secrets.token_hex(8)
        return f"{prefix}_{timestamp}_{random_part}"
    
    def __repr__(self) -> str:
        return (f"Enhanced FinancialProofVerifier(proofs={len(self.proofs)}, "
               f"claims={len(self.claims)}, "
               f"success_rate={self.metrics['verification_success_rate']:.2%}, "
               f"security_level={self.config.get('security_level', 'standard')})")

# Legacy compatibility
ProofVerifier = FinancialProofVerifier

# Export enhanced classes
__all__ = [
    'FinancialProofVerifier',
    'ProofVerifier',  # Legacy compatibility
    'EnhancedProofClaim',
    'EnhancedProofEvidence', 
    'EnhancedProofResult',
    'ProofType',
    'ProofStatus',
    'ProofLevel',
    'SecurityLevel',
    'BaseProofSystem',
    'EnhancedRuleBasedProofSystem',
    'EnhancedMLProofSystem',
    'EnhancedCryptographicProofSystem',
    'SecurityValidator',
    'ResourceMonitor',
    # Exception classes
    'ProofVerificationException',
    'ProofConfigurationError',
    'ProofGenerationError',
    'ProofValidationError',
    'ProofTimeoutError',
    'ProofSecurityError',
    'ProofIntegrityError',
    'ClaimValidationError',
    'EvidenceValidationError',
    'ProofSystemError',
    'ProofStorageError',
    'CryptographicError',
    'ProofExpiredError',
    'ResourceLimitError'
]

if __name__ == "__main__":
    # Enhanced example usage and testing
    
    # Initialize enhanced verifier with comprehensive config
    config = {
        'ml_threshold': 0.7,
        'enable_cryptographic_proofs': True,
        'proof_validity_hours': 24,
        'security_level': 'enhanced',
        'max_memory_mb': 256,
        'max_cpu_percent': 40.0,
        'enable_audit_trail': True,
        'validation_strict_mode': True,
        'enable_resource_monitoring': True
    }
    
    verifier = FinancialProofVerifier(config)
    print("Enhanced ProofVerifier initialized with advanced security")
    
    # Create comprehensive sample claim
    claim = {
        'transaction_id': 'TXN_ENHANCED_12345',
        'fraud_probability': 0.89,
        'risk_score': 0.92,
        'claim_type': 'transaction_fraud',
        'confidence_level': 'high',
        'security_level': 'enhanced',
        'evidence': {
            'unusual_amount': True,
            'velocity_violation': True,
            'geographical_anomaly': True,
            'behavioral_deviation': True
        },
        'violated_rules': ['max_transaction_amount', 'velocity_limit', 'geographical_restriction'],
        'rule_severity': {
            'max_transaction_amount': 'high',
            'velocity_limit': 'medium',
            'geographical_restriction': 'critical'
        },
        'model_version': 'enhanced_rf_model_v2.1',
        'model_confidence': 0.94,
        'model_features': ['amount', 'velocity', 'geography', 'time', 'merchant_type'],
        'metadata': {
            'transaction_category': 'high_risk',
            'customer_tier': 'premium',
            'review_required': True
        },
        'tags': ['high_value', 'cross_border', 'velocity_anomaly']
    }
    
    # Enhanced validation
    print("\nPerforming enhanced claim validation...")
    is_valid, message, errors = verifier.validate_claim(claim)
    print(f"Validation result: {is_valid} - {message}")
    if errors:
        print(f"Validation errors: {errors}")
    
    # Enhanced evidence
    evidence = [
        {
            'type': 'transaction_analysis',
            'source': 'enhanced_analytics_engine',
            'confidence': 0.91,
            'reliability_score': 0.95,
            'data': {
                'amount_zscore': 4.2,
                'velocity_ratio': 3.8,
                'risk_indicators': ['high_amount', 'rapid_succession', 'new_merchant']
            },
            'metadata': {'analysis_version': 'v3.0', 'timestamp': datetime.now().isoformat()}
        },
        {
            'type': 'behavioral_analysis',
            'source': 'behavioral_ml_model',
            'confidence': 0.87,
            'reliability_score': 0.89,
            'data': {
                'deviation_score': 0.93,
                'pattern_match': 'known_fraud_pattern_47',
                'historical_comparison': 'significant_deviation'
            },
            'metadata': {'model_version': 'behavioral_v2.3'}
        }
    ]
    
    # Generate enhanced proof
    print("\nGenerating enhanced proof with comprehensive validation...")
    proof_result = verifier.generate_proof(claim, evidence)
    
    if proof_result:
        print(f"Enhanced proof generated: {proof_result.proof_id}")
        print(f"Status: {proof_result.status.value}")
        print(f"Confidence: {proof_result.confidence:.2f}")
        print(f"Quality Score: {proof_result.quality_score:.2f}")
        print(f"Reliability Score: {proof_result.reliability_score:.2f}")
        print(f"Security Level: {proof_result.security_level.value}")
        print(f"Valid until: {proof_result.valid_until}")
        print(f"Systems used: {proof_result.reasoning['systems_used']}")
        
        # Enhanced verification
        print("\nPerforming enhanced proof verification...")
        proof_data = {
            'proof_id': proof_result.proof_id,
            'claim_id': proof_result.claim_id,
            'proof_hash': proof_result.proof_hash,
            'systems_used': proof_result.reasoning['systems_used']
        }
        
        is_valid, message, details = verifier.verify_proof(proof_data)
        print(f"Verification result: {is_valid} - {message}")
        print(f"Verification details: {json.dumps(details, indent=2)}")
    
    # Enhanced metrics
    metrics = verifier.get_enhanced_metrics()
    print(f"\nEnhanced Metrics:")
    print(json.dumps(metrics, indent=2))
    
    # Audit trail
    audit_trail = verifier.export_enhanced_audit_trail()
    print(f"\nAudit Trail Entries: {len(audit_trail)}")
    if audit_trail:
        print("Latest audit entry:", json.dumps(audit_trail[-1], indent=2))
    
    # Cleanup demonstration
    print("\nPerforming enhanced cleanup...")
    cleanup_results = verifier.cleanup_expired_proofs()
    print(f"Cleanup results: {cleanup_results}")
    
    print("\nEnhanced FinancialProofVerifier ready for production use!")
    print("Features: Advanced validation, comprehensive security, detailed monitoring")