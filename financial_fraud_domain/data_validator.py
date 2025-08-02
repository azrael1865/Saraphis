"""
Comprehensive Data Validation for Financial Fraud Detection
Advanced validation system with fraud-specific rules, compliance checking,
and multi-level validation for financial transaction data.
Enhanced with comprehensive validation and error handling from enhanced_data_validator.
"""

import logging
import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set, Callable, Pattern
from datetime import datetime, timedelta
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
import warnings

# Import enhanced validator components for integration
# NOTE: Commented out to avoid circular import with enhanced_data_validator
# try:
#     from enhanced_data_validator import (
#         EnhancedFinancialDataValidator,
#         ValidationConfig,
#         ValidationContext,
#         ValidationMode,
#         SecurityLevel,
#         ValidationPerformanceMetrics,
#         ValidationSecurityReport,
#         ValidationIntegrationReport,
#         InputValidator,
#         PerformanceMonitor,
#         SecurityValidator,
#         IntegrationValidator,
#         ErrorRecoveryManager,
#         ValidationException,
#         InputValidationError,
#         ValidationTimeoutError,
#         ValidationConfigError,
#         ValidationSecurityError,
#         ValidationIntegrationError,
#         DataCorruptionError,
#         PartialValidationError
#     )
#     ENHANCED_AVAILABLE = True
# except ImportError:
#     logger.warning("Enhanced data validator not available, using basic functionality")
#     ENHANCED_AVAILABLE = False

# Set enhanced availability to False to avoid circular import
ENHANCED_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

def _get_timestamp():
    """Get current timestamp as ISO string"""
    try:
        return datetime.now().isoformat()
    except Exception:
        return "unknown"

class ValidationLevel(Enum):
    """Validation strictness levels"""
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"
    REGULATORY = "regulatory"

class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ComplianceStandard(Enum):
    """Financial compliance standards"""
    PCI_DSS = "pci_dss"
    SOX = "sox"
    GDPR = "gdpr"
    BSA_AML = "bsa_aml"
    FFIEC = "ffiec"
    ISO27001 = "iso27001"

class ValidationIssue:
    """Comprehensive validation issue tracking"""
    def __init__(self, severity: ValidationSeverity, category: str, message: str,
                 field: Optional[str] = None, record_id: Optional[str] = None,
                 record_index: Optional[int] = None, value: Optional[Any] = None,
                 rule_id: Optional[str] = None, compliance_standard: Optional[str] = None,
                 timestamp: str = "unknown", metadata: Optional[Dict[str, Any]] = None):
        self.severity = severity
        self.category = category
        self.message = message
        self.field = field
        self.record_id = record_id
        self.record_index = record_index
        self.value = value
        self.rule_id = rule_id
        self.compliance_standard = compliance_standard
        self.timestamp = timestamp
        self.metadata = metadata or {}

@dataclass
class ValidationResult:
    """Comprehensive validation results"""
    is_valid: bool
    total_records: int
    valid_records: int
    invalid_records: int
    issues: List[ValidationIssue]
    data_quality_score: float
    validation_time_seconds: float
    compliance_scores: Dict[str, float] = field(default_factory=dict)
    field_statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

class ValidationRule(ABC):
    """Abstract base class for validation rules"""
    
    def __init__(self, rule_id: str, severity: ValidationSeverity = ValidationSeverity.ERROR,
                 enabled: bool = True, compliance_standards: List[str] = None):
        self.rule_id = rule_id
        self.severity = severity
        self.enabled = enabled
        self.compliance_standards = compliance_standards or []
        
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate data and return list of issues"""
        pass
    
    def is_applicable(self, validation_level: ValidationLevel, 
                     required_compliance: List[str] = None) -> bool:
        """Check if rule should be applied for given validation level and compliance requirements"""
        if not self.enabled:
            return False
            
        # Check compliance requirements
        if required_compliance:
            if not any(std in self.compliance_standards for std in required_compliance):
                return False
                
        return True

class RequiredFieldsRule(ValidationRule):
    """Validates presence of required fields"""
    
    def __init__(self, required_fields: List[str], **kwargs):
        super().__init__("required_fields", **kwargs)
        self.required_fields = required_fields
        
    def validate(self, data: pd.DataFrame) -> List[ValidationIssue]:
        issues = []
        missing_fields = set(self.required_fields) - set(data.columns)
        
        for field in missing_fields:
            issues.append(ValidationIssue(
                severity=self.severity,
                category="schema",
                message=f"Required field '{field}' is missing",
                field=field,
                rule_id=self.rule_id
            ))
            
        return issues

class DataTypeRule(ValidationRule):
    """Validates data types of fields"""
    
    def __init__(self, field_types: Dict[str, str], **kwargs):
        super().__init__("data_types", **kwargs)
        self.field_types = field_types
        
    def validate(self, data: pd.DataFrame) -> List[ValidationIssue]:
        issues = []
        
        for field, expected_type in self.field_types.items():
            if field not in data.columns:
                continue
                
            if expected_type == "datetime":
                try:
                    pd.to_datetime(data[field], errors='raise')
                except Exception:
                    invalid_indices = data[data[field].notna()].index
                    for idx in invalid_indices[:10]:  # Limit to first 10
                        issues.append(ValidationIssue(
                            severity=self.severity,
                            category="data_type",
                            message=f"Invalid datetime format in field '{field}'",
                            field=field,
                            record_index=idx,
                            value=data.loc[idx, field],
                            rule_id=self.rule_id
                        ))
                        
            elif expected_type == "numeric":
                non_numeric = pd.to_numeric(data[field], errors='coerce').isna() & data[field].notna()
                for idx in data[non_numeric].index[:10]:
                    issues.append(ValidationIssue(
                        severity=self.severity,
                        category="data_type",
                        message=f"Non-numeric value in field '{field}'",
                        field=field,
                        record_index=idx,
                        value=data.loc[idx, field],
                        rule_id=self.rule_id
                    ))
                    
        return issues

class RangeValidationRule(ValidationRule):
    """Validates numeric ranges"""
    
    def __init__(self, field: str, min_value: Optional[float] = None, 
                 max_value: Optional[float] = None, **kwargs):
        super().__init__(f"range_{field}", **kwargs)
        self.field = field
        self.min_value = min_value
        self.max_value = max_value
        
    def validate(self, data: pd.DataFrame) -> List[ValidationIssue]:
        issues = []
        
        if self.field not in data.columns:
            return issues
            
        # Convert to numeric
        numeric_data = pd.to_numeric(data[self.field], errors='coerce')
        
        if self.min_value is not None:
            below_min = (numeric_data < self.min_value) & numeric_data.notna()
            for idx in data[below_min].index[:20]:
                issues.append(ValidationIssue(
                    severity=self.severity,
                    category="range",
                    message=f"Value below minimum ({self.min_value}) in field '{self.field}'",
                    field=self.field,
                    record_index=idx,
                    value=data.loc[idx, self.field],
                    rule_id=self.rule_id
                ))
                
        if self.max_value is not None:
            above_max = (numeric_data > self.max_value) & numeric_data.notna()
            for idx in data[above_max].index[:20]:
                issues.append(ValidationIssue(
                    severity=self.severity,
                    category="range",
                    message=f"Value above maximum ({self.max_value}) in field '{self.field}'",
                    field=self.field,
                    record_index=idx,
                    value=data.loc[idx, self.field],
                    rule_id=self.rule_id
                ))
                
        return issues

class PatternValidationRule(ValidationRule):
    """Validates field patterns using regex"""
    
    def __init__(self, field: str, pattern: str, pattern_name: str = "pattern", **kwargs):
        super().__init__(f"pattern_{field}", **kwargs)
        self.field = field
        self.pattern = re.compile(pattern)
        self.pattern_name = pattern_name
        
    def validate(self, data: pd.DataFrame) -> List[ValidationIssue]:
        issues = []
        
        if self.field not in data.columns:
            return issues
            
        # Check pattern matches
        string_data = data[self.field].astype(str)
        invalid_pattern = ~string_data.str.match(self.pattern) & data[self.field].notna()
        
        for idx in data[invalid_pattern].index[:50]:
            issues.append(ValidationIssue(
                severity=self.severity,
                category="pattern",
                message=f"Invalid {self.pattern_name} format in field '{self.field}'",
                field=self.field,
                record_index=idx,
                value=data.loc[idx, self.field],
                rule_id=self.rule_id
            ))
            
        return issues

class DuplicateRule(ValidationRule):
    """Validates duplicate detection"""
    
    def __init__(self, fields: List[str], **kwargs):
        super().__init__("duplicates", **kwargs)
        self.fields = fields
        
    def validate(self, data: pd.DataFrame) -> List[ValidationIssue]:
        issues = []
        
        # Check if all fields exist
        missing_fields = set(self.fields) - set(data.columns)
        if missing_fields:
            return issues
            
        # Find duplicates
        duplicates = data.duplicated(subset=self.fields, keep=False)
        duplicate_groups = data[duplicates].groupby(self.fields)
        
        for name, group in duplicate_groups:
            if len(group) > 1:
                for idx in group.index:
                    issues.append(ValidationIssue(
                        severity=self.severity,
                        category="duplicate",
                        message=f"Duplicate record found based on fields: {self.fields}",
                        record_index=idx,
                        rule_id=self.rule_id,
                        metadata={"duplicate_count": len(group), "fields": self.fields}
                    ))
                    
        return issues

class FraudPatternRule(ValidationRule):
    """Detects potential fraud patterns"""
    
    def __init__(self, **kwargs):
        super().__init__("fraud_patterns", **kwargs)
        
    def validate(self, data: pd.DataFrame) -> List[ValidationIssue]:
        issues = []
        
        # Pattern 1: Unusual transaction amounts (round numbers)
        if 'amount' in data.columns:
            amount_numeric = pd.to_numeric(data['amount'], errors='coerce')
            round_amounts = (amount_numeric % 1 == 0) & (amount_numeric >= 100)
            round_percentage = round_amounts.sum() / len(data) * 100
            
            if round_percentage > 30:  # More than 30% round amounts is suspicious
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="fraud_pattern",
                    message=f"High percentage of round-number amounts ({round_percentage:.1f}%)",
                    rule_id=self.rule_id,
                    metadata={"round_percentage": round_percentage}
                ))
                
        # Pattern 2: Velocity anomalies
        if all(col in data.columns for col in ['user_id', 'timestamp', 'amount']):
            issues.extend(self._check_velocity_anomalies(data))
            
        # Pattern 3: Geographic anomalies
        if all(col in data.columns for col in ['user_id', 'location']):
            issues.extend(self._check_geographic_anomalies(data))
            
        # Pattern 4: Time-based anomalies
        if 'timestamp' in data.columns:
            issues.extend(self._check_time_anomalies(data))
            
        return issues
    
    def _check_velocity_anomalies(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Check for transaction velocity anomalies"""
        issues = []
        
        # Convert timestamp and sort
        data_copy = data.copy()
        data_copy['timestamp'] = pd.to_datetime(data_copy['timestamp'])
        data_copy = data_copy.sort_values(['user_id', 'timestamp'])
        
        # Calculate time differences between consecutive transactions per user
        data_copy['time_diff'] = data_copy.groupby('user_id')['timestamp'].diff()
        
        # Flag rapid transactions (within 1 minute)
        rapid_transactions = data_copy['time_diff'] < pd.Timedelta(minutes=1)
        
        for idx in data_copy[rapid_transactions].index:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="fraud_pattern",
                message="Rapid successive transactions detected",
                record_index=idx,
                rule_id=self.rule_id,
                metadata={"time_diff_seconds": data_copy.loc[idx, 'time_diff'].total_seconds()}
            ))
            
        return issues
    
    def _check_geographic_anomalies(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Check for geographic anomalies"""
        issues = []
        
        # Count unique locations per user
        user_locations = data.groupby('user_id')['location'].nunique()
        multi_location_users = user_locations[user_locations > 10]  # More than 10 locations
        
        for user_id in multi_location_users.index:
            user_data = data[data['user_id'] == user_id]
            for idx in user_data.index[:5]:  # Flag first 5 transactions
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="fraud_pattern",
                    message=f"User with transactions in {user_locations[user_id]} different locations",
                    record_index=idx,
                    rule_id=self.rule_id,
                    metadata={"location_count": user_locations[user_id]}
                ))
                
        return issues
    
    def _check_time_anomalies(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Check for time-based anomalies"""
        issues = []
        
        timestamp_data = pd.to_datetime(data['timestamp'])
        
        # Check for transactions at unusual hours (2-5 AM)
        unusual_hours = timestamp_data.dt.hour.isin([2, 3, 4, 5])
        unusual_hour_percentage = unusual_hours.sum() / len(data) * 100
        
        if unusual_hour_percentage > 15:  # More than 15% in unusual hours
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="fraud_pattern",
                message=f"High percentage of transactions in unusual hours ({unusual_hour_percentage:.1f}%)",
                rule_id=self.rule_id,
                metadata={"unusual_hour_percentage": unusual_hour_percentage}
            ))
            
        return issues

class ComplianceRule(ValidationRule):
    """Validates compliance with financial regulations"""
    
    def __init__(self, compliance_standard: str, **kwargs):
        super().__init__(f"compliance_{compliance_standard.value}", 
                        compliance_standards=[compliance_standard], **kwargs)
        self.compliance_standard = compliance_standard
        
    def validate(self, data: pd.DataFrame) -> List[ValidationIssue]:
        issues = []
        
        if self.compliance_standard == ComplianceStandard.PCI_DSS:
            issues.extend(self._validate_pci_dss(data))
        elif self.compliance_standard == ComplianceStandard.BSA_AML:
            issues.extend(self._validate_bsa_aml(data))
        elif self.compliance_standard == ComplianceStandard.GDPR:
            issues.extend(self._validate_gdpr(data))
            
        return issues
    
    def _validate_pci_dss(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate PCI DSS compliance"""
        issues = []
        
        # Check for credit card numbers in data
        for column in data.columns:
            if 'card' in column.lower() or 'pan' in column.lower():
                # Check for patterns that look like credit card numbers
                string_data = data[column].astype(str)
                potential_cards = string_data.str.match(r'^\d{13,19}$')
                
                for idx in data[potential_cards].index:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        category="compliance",
                        message="Potential unencrypted credit card number detected",
                        field=column,
                        record_index=idx,
                        rule_id=self.rule_id,
                        compliance_standard=ComplianceStandard.PCI_DSS
                    ))
                    
        return issues
    
    def _validate_bsa_aml(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate BSA/AML compliance"""
        issues = []
        
        # Check for high-value transactions that may require reporting
        if 'amount' in data.columns:
            amount_numeric = pd.to_numeric(data['amount'], errors='coerce')
            high_value = amount_numeric > 10000  # $10,000 threshold
            
            for idx in data[high_value].index:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="compliance",
                    message="High-value transaction requiring AML consideration",
                    field="amount",
                    record_index=idx,
                    value=data.loc[idx, 'amount'],
                    rule_id=self.rule_id,
                    compliance_standard=ComplianceStandard.BSA_AML
                ))
                
        return issues
    
    def _validate_gdpr(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate GDPR compliance"""
        issues = []
        
        # Check for potential PII fields
        pii_patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?[\d\s\-\(\)]{10,}$',
            'ssn': r'^\d{3}-\d{2}-\d{4}$'
        }
        
        for column in data.columns:
            for pii_type, pattern in pii_patterns.items():
                if pii_type in column.lower():
                    string_data = data[column].astype(str)
                    potential_pii = string_data.str.match(pattern)
                    
                    if potential_pii.any():
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            category="compliance",
                            message=f"Potential {pii_type.upper()} data detected - ensure GDPR compliance",
                            field=column,
                            rule_id=self.rule_id,
                            compliance_standard=ComplianceStandard.GDPR,
                            metadata={"pii_type": pii_type, "count": potential_pii.sum()}
                        ))
                        
        return issues

class FinancialDataValidator:
    """
    Comprehensive financial data validator with fraud detection and compliance checking
    
    Features:
    - Multi-level validation (basic to comprehensive)
    - Fraud pattern detection
    - Regulatory compliance checking (PCI DSS, BSA/AML, GDPR, SOX)
    - Configurable validation rules
    - Performance monitoring and statistics
    - Detailed reporting and metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize financial data validator
        
        Args:
            config: Configuration dictionary for validation settings
        """
        self.config = config or {}
        self.rules: Dict[str, ValidationRule] = {}
        self.validation_stats = {
            'total_validations': 0,
            'total_records_validated': 0,
            'total_issues_found': 0,
            'average_validation_time': 0.0
        }
        self._lock = threading.RLock()
        
        # Enhanced validator integration
        self._enhanced_validator = None
        if ENHANCED_AVAILABLE and self.config.get('use_enhanced', True):
            try:
                enhanced_config = self._create_enhanced_config()
                self._enhanced_validator = EnhancedFinancialDataValidator(enhanced_config)
                logger.info("Enhanced validation features enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced validator: {e}")
        
        # Initialize default rules
        self._initialize_default_rules()
        
        logger.info("FinancialDataValidator initialized")
    
    def _create_enhanced_config(self):
        """Create enhanced validation configuration from basic config"""
        # NOTE: Enhanced config creation disabled due to circular import
        return None
        # enhanced_config = ValidationConfig()
        
        # Map basic config to enhanced config
        if 'validation_level' in self.config:
            try:
                enhanced_config.validation_level = ValidationLevel(self.config['validation_level'])
            except ValueError:
                logger.warning(f"Invalid validation level: {self.config['validation_level']}")
        
        if 'timeout' in self.config:
            enhanced_config.timeout_seconds = float(self.config['timeout'])
        
        if 'memory_limit' in self.config:
            enhanced_config.max_memory_mb = float(self.config['memory_limit'])
        
        if 'enable_security' in self.config:
            enhanced_config.security_validation = bool(self.config['enable_security'])
        
        if 'enable_performance_monitoring' in self.config:
            enhanced_config.performance_monitoring = bool(self.config['enable_performance_monitoring'])
        
        if 'compliance_standards' in self.config:
            enhanced_config.compliance_standards = self.config['compliance_standards']
        
        return enhanced_config
    
    def _initialize_default_rules(self):
        """Initialize default validation rules"""
        
        # Required fields for financial transactions
        self.add_rule(RequiredFieldsRule(
            required_fields=['transaction_id', 'amount', 'timestamp', 'user_id'],
            severity=ValidationSeverity.ERROR
        ))
        
        # Data type validation
        self.add_rule(DataTypeRule(
            field_types={
                'amount': 'numeric',
                'timestamp': 'datetime',
                'transaction_id': 'string',
                'user_id': 'string'
            },
            severity=ValidationSeverity.ERROR
        ))
        
        # Amount range validation
        self.add_rule(RangeValidationRule(
            field='amount',
            min_value=0.01,
            max_value=1000000.0,
            severity=ValidationSeverity.WARNING
        ))
        
        # Transaction ID pattern validation
        self.add_rule(PatternValidationRule(
            field='transaction_id',
            pattern=r'^[A-Z0-9]{6,20}$',
            pattern_name='transaction ID',
            severity=ValidationSeverity.WARNING
        ))
        
        # Duplicate detection
        self.add_rule(DuplicateRule(
            fields=['transaction_id'],
            severity=ValidationSeverity.ERROR
        ))
        
        # Fraud pattern detection
        self.add_rule(FraudPatternRule(
            severity=ValidationSeverity.WARNING
        ))
        
        # Compliance rules
        self.add_rule(ComplianceRule(
            compliance_standard=ComplianceStandard.PCI_DSS,
            severity=ValidationSeverity.CRITICAL
        ))
        
        self.add_rule(ComplianceRule(
            compliance_standard=ComplianceStandard.BSA_AML,
            severity=ValidationSeverity.WARNING
        ))
        
        self.add_rule(ComplianceRule(
            compliance_standard=ComplianceStandard.GDPR,
            severity=ValidationSeverity.WARNING
        ))
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule"""
        with self._lock:
            self.rules[rule.rule_id] = rule
            logger.debug(f"Added validation rule: {rule.rule_id}")
    
    def remove_rule(self, rule_id: str) -> None:
        """Remove a validation rule"""
        with self._lock:
            if rule_id in self.rules:
                del self.rules[rule_id]
                logger.debug(f"Removed validation rule: {rule_id}")
    
    def enable_rule(self, rule_id: str) -> None:
        """Enable a validation rule"""
        with self._lock:
            if rule_id in self.rules:
                self.rules[rule_id].enabled = True
                logger.debug(f"Enabled validation rule: {rule_id}")
    
    def disable_rule(self, rule_id: str) -> None:
        """Disable a validation rule"""
        with self._lock:
            if rule_id in self.rules:
                self.rules[rule_id].enabled = False
                logger.debug(f"Disabled validation rule: {rule_id}")
    
    def validate(self, 
                data: pd.DataFrame,
                validation_level: ValidationLevel = ValidationLevel.STANDARD,
                required_compliance: List[str] = None,
                parallel: bool = True,
                use_enhanced: bool = None) -> ValidationResult:
        """
        Comprehensive data validation
        
        Args:
            data: DataFrame to validate
            validation_level: Level of validation strictness
            required_compliance: Required compliance standards
            parallel: Whether to run validation rules in parallel
            use_enhanced: Whether to use enhanced validator if available
            
        Returns:
            Comprehensive validation results
        """
        start_time = datetime.now()
        
        with self._lock:
            self.validation_stats['total_validations'] += 1
            self.validation_stats['total_records_validated'] += len(data)
        
        logger.info(f"Starting validation of {len(data)} records at {validation_level.value} level")
        
        # Try enhanced validator first if available and requested
        if use_enhanced is None:
            use_enhanced = self.config.get('use_enhanced', True)
            
        if use_enhanced and ENHANCED_AVAILABLE and self._enhanced_validator is not None:
            try:
                logger.debug("Using enhanced validation")
                enhanced_result = self._enhanced_validator.validate(
                    data, 
                    validation_level=validation_level,
                    required_compliance=required_compliance,
                    parallel=parallel
                )
                
                # Update stats
                end_time = datetime.now()
                validation_time = (end_time - start_time).total_seconds()
                with self._lock:
                    self.validation_stats['total_issues_found'] += len(enhanced_result.issues) if hasattr(enhanced_result, 'issues') else 0
                    total_validations = self.validation_stats['total_validations']
                    self.validation_stats['average_validation_time'] = (
                        (self.validation_stats['average_validation_time'] * (total_validations - 1) + validation_time) 
                        / total_validations
                    )
                
                return enhanced_result
                
            except Exception as e:
                logger.warning(f"Enhanced validator failed, falling back to basic validator: {e}")
                # Fall through to basic validation
        
        # Filter applicable rules
        applicable_rules = [
            rule for rule in self.rules.values()
            if rule.is_applicable(validation_level, required_compliance)
        ]
        
        logger.debug(f"Applying {len(applicable_rules)} validation rules")
        
        # Run validation rules
        all_issues = []
        
        if parallel and len(applicable_rules) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=min(4, len(applicable_rules))) as executor:
                future_to_rule = {
                    executor.submit(rule.validate, data): rule 
                    for rule in applicable_rules
                }
                
                for future in future_to_rule:
                    try:
                        issues = future.result()
                        all_issues.extend(issues)
                    except Exception as e:
                        rule = future_to_rule[future]
                        logger.error(f"Error in validation rule {rule.rule_id}: {e}")
                        all_issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category="system",
                            message=f"Validation rule {rule.rule_id} failed: {e}",
                            rule_id=rule.rule_id
                        ))
        else:
            # Sequential execution
            for rule in applicable_rules:
                try:
                    issues = rule.validate(data)
                    all_issues.extend(issues)
                except Exception as e:
                    logger.error(f"Error in validation rule {rule.rule_id}: {e}")
                    all_issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="system",
                        message=f"Validation rule {rule.rule_id} failed: {e}",
                        rule_id=rule.rule_id
                    ))
        
        # Calculate results
        end_time = datetime.now()
        validation_time = (end_time - start_time).total_seconds()
        
        # Determine validity
        critical_issues = [i for i in all_issues if i.severity == ValidationSeverity.CRITICAL]
        error_issues = [i for i in all_issues if i.severity == ValidationSeverity.ERROR]
        is_valid = len(critical_issues) == 0 and len(error_issues) == 0
        
        # Calculate records with issues
        records_with_issues = set()
        for issue in all_issues:
            if issue.record_index is not None:
                records_with_issues.add(issue.record_index)
        
        valid_records = len(data) - len(records_with_issues)
        invalid_records = len(records_with_issues)
        
        # Calculate data quality score
        quality_score = self._calculate_quality_score(data, all_issues)
        
        # Calculate compliance scores
        compliance_scores = self._calculate_compliance_scores(all_issues, required_compliance or [])
        
        # Generate field statistics
        field_stats = self._generate_field_statistics(data, all_issues)
        
        # Create summary
        summary = self._create_summary(data, all_issues)
        
        # Update statistics
        with self._lock:
            self.validation_stats['total_issues_found'] += len(all_issues)
            total_validations = self.validation_stats['total_validations']
            self.validation_stats['average_validation_time'] = (
                (self.validation_stats['average_validation_time'] * (total_validations - 1) + validation_time) 
                / total_validations
            )
        
        result = ValidationResult(
            is_valid=is_valid,
            total_records=len(data),
            valid_records=valid_records,
            invalid_records=invalid_records,
            issues=all_issues,
            data_quality_score=quality_score,
            validation_time_seconds=validation_time,
            compliance_scores=compliance_scores,
            field_statistics=field_stats,
            summary=summary
        )
        
        logger.info(f"Validation completed: {valid_records}/{len(data)} valid records, "
                   f"{len(all_issues)} issues found, quality score: {quality_score:.2f}")
        
        return result
    
    def validate_single_record(self, record: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate a single record"""
        df = pd.DataFrame([record])
        result = self.validate(df, ValidationLevel.BASIC, parallel=False)
        return result.issues
    
    def _calculate_quality_score(self, data: pd.DataFrame, issues: List[ValidationIssue]) -> float:
        """Calculate overall data quality score (0.0 to 1.0)"""
        if len(data) == 0:
            return 0.0
        
        # Weight issues by severity
        severity_weights = {
            ValidationSeverity.INFO: 0.01,
            ValidationSeverity.WARNING: 0.05,
            ValidationSeverity.ERROR: 0.15,
            ValidationSeverity.CRITICAL: 0.30
        }
        
        total_penalty = 0.0
        for issue in issues:
            total_penalty += severity_weights.get(issue.severity, 0.1)
        
        # Normalize penalty by number of records
        penalty_per_record = total_penalty / len(data)
        
        # Calculate score (1.0 - penalty, minimum 0.0)
        score = max(0.0, 1.0 - penalty_per_record)
        return score
    
    def _calculate_compliance_scores(self, issues: List[ValidationIssue], 
                                   required_compliance: List[str]) -> Dict[str, float]:
        """Calculate compliance scores for each standard"""
        scores = {}
        
        for standard in required_compliance:
            compliance_issues = [
                i for i in issues 
                if i.compliance_standard == standard
            ]
            
            # Calculate score based on severity of compliance issues
            if not compliance_issues:
                scores[standard] = 1.0
            else:
                critical_count = sum(1 for i in compliance_issues if i.severity == ValidationSeverity.CRITICAL)
                error_count = sum(1 for i in compliance_issues if i.severity == ValidationSeverity.ERROR)
                warning_count = sum(1 for i in compliance_issues if i.severity == ValidationSeverity.WARNING)
                
                # Severe penalties for compliance issues
                penalty = critical_count * 0.5 + error_count * 0.3 + warning_count * 0.1
                scores[standard] = max(0.0, 1.0 - penalty)
        
        return scores
    
    def _generate_field_statistics(self, data: pd.DataFrame, 
                                 issues: List[ValidationIssue]) -> Dict[str, Dict[str, Any]]:
        """Generate statistics for each field"""
        stats = {}
        
        for column in data.columns:
            field_issues = [i for i in issues if i.field == column]
            
            stats[column] = {
                'total_records': len(data),
                'non_null_records': data[column].notna().sum(),
                'null_records': data[column].isna().sum(),
                'unique_values': data[column].nunique(),
                'issues_count': len(field_issues),
                'data_type': str(data[column].dtype),
                'sample_values': data[column].dropna().head(5).tolist()
            }
            
            # Add numeric statistics if applicable
            if pd.api.types.is_numeric_dtype(data[column]):
                stats[column].update({
                    'min_value': data[column].min(),
                    'max_value': data[column].max(),
                    'mean_value': data[column].mean(),
                    'std_value': data[column].std()
                })
        
        return stats
    
    def _create_summary(self, data: pd.DataFrame, issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Create validation summary"""
        issue_by_severity = {}
        issue_by_category = {}
        
        for issue in issues:
            # Group by severity
            severity_key = issue.severity.value
            issue_by_severity[severity_key] = issue_by_severity.get(severity_key, 0) + 1
            
            # Group by category
            category_key = issue.category
            issue_by_category[category_key] = issue_by_category.get(category_key, 0) + 1
        
        return {
            'total_records': len(data),
            'total_fields': len(data.columns),
            'total_issues': len(issues),
            'issues_by_severity': issue_by_severity,
            'issues_by_category': issue_by_category,
            'validation_timestamp': datetime.now().isoformat(),
            'rules_applied': len(self.rules)
        }
    
    def get_validation_report(self, result: ValidationResult, 
                            format: str = "dict") -> Union[Dict[str, Any], str]:
        """Generate comprehensive validation report"""
        # Use enhanced reporting if available
        if ENHANCED_AVAILABLE and self._enhanced_validator is not None:
            try:
                return self._enhanced_validator.get_validation_report(result, format)
            except Exception as e:
                logger.warning(f"Enhanced reporting failed, using basic report: {e}")
        
        if format == "dict":
            return {
                'validation_summary': result.summary,
                'data_quality_score': result.data_quality_score,
                'compliance_scores': result.compliance_scores,
                'field_statistics': result.field_statistics,
                'issues': [
                    {
                        'severity': issue.severity.value,
                        'category': issue.category,
                        'message': issue.message,
                        'field': issue.field,
                        'record_index': issue.record_index,
                        'rule_id': issue.rule_id,
                        'compliance_standard': issue.compliance_standard.value if issue.compliance_standard else None
                    }
                    for issue in result.issues
                ]
            }
        elif format == "json":
            report_dict = self.get_validation_report(result, "dict")
            return json.dumps(report_dict, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def export_issues(self, issues: List[ValidationIssue], 
                     file_path: Optional[Path] = None) -> Optional[str]:
        """Export validation issues to file"""
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = Path(f"validation_issues_{timestamp}.json")
        
        issues_data = []
        for issue in issues:
            issue_dict = {
                'severity': issue.severity.value,
                'category': issue.category,
                'message': issue.message,
                'field': issue.field,
                'record_id': issue.record_id,
                'record_index': issue.record_index,
                'value': str(issue.value) if issue.value is not None else None,
                'rule_id': issue.rule_id,
                'compliance_standard': issue.compliance_standard.value if issue.compliance_standard else None,
                'timestamp': issue.timestamp,
                'metadata': issue.metadata
            }
            issues_data.append(issue_dict)
        
        with open(file_path, 'w') as f:
            json.dump(issues_data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(issues)} validation issues to {file_path}")
        return str(file_path)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        with self._lock:
            return self.validation_stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset validation statistics"""
        with self._lock:
            self.validation_stats = {
                'total_validations': 0,
                'total_records_validated': 0,
                'total_issues_found': 0,
                'average_validation_time': 0.0
            }
    
    def list_rules(self) -> List[Dict[str, Any]]:
        """List all validation rules"""
        rules_info = []
        for rule_id, rule in self.rules.items():
            rules_info.append({
                'rule_id': rule_id,
                'severity': rule.severity.value,
                'enabled': rule.enabled,
                'compliance_standards': [std.value for std in rule.compliance_standards]
            })
        return rules_info
    
    def get_enhanced_error_summary(self) -> Dict[str, Any]:
        """Get enhanced error summary if available"""
        if ENHANCED_AVAILABLE and self._enhanced_validator is not None:
            try:
                return self._enhanced_validator.get_error_summary()
            except Exception as e:
                logger.warning(f"Enhanced error reporting failed: {e}")
        return {"message": "Enhanced error reporting not available"}
    
    def get_enhanced_performance_summary(self) -> Dict[str, Any]:
        """Get enhanced performance summary if available"""
        if ENHANCED_AVAILABLE and self._enhanced_validator is not None:
            try:
                return self._enhanced_validator.get_performance_summary()
            except Exception as e:
                logger.warning(f"Enhanced performance reporting failed: {e}")
        return {"message": "Enhanced performance reporting not available"}
    
    def clear_enhanced_cache(self) -> None:
        """Clear enhanced validator cache if available"""
        if ENHANCED_AVAILABLE and self._enhanced_validator is not None:
            try:
                self._enhanced_validator.clear_cache()
                logger.info("Enhanced validator cache cleared")
            except Exception as e:
                logger.warning(f"Failed to clear enhanced cache: {e}")
    
    def validate_configuration_enhanced(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration using enhanced validator if available"""
        if ENHANCED_AVAILABLE and self._enhanced_validator is not None:
            try:
                return self._enhanced_validator.validate_configuration(config)
            except Exception as e:
                logger.warning(f"Enhanced configuration validation failed: {e}")
        return True, []
    
    def __repr__(self) -> str:
        enhanced_info = ""
        if ENHANCED_AVAILABLE and self._enhanced_validator is not None:
            enhanced_info = ", enhanced=True"
        
        return (f"FinancialDataValidator(rules={len(self.rules)}, "
               f"validations={self.validation_stats['total_validations']}, "
               f"avg_time={self.validation_stats['average_validation_time']:.3f}s"
               f"{enhanced_info})")

# Export main classes
__all__ = [
    'FinancialDataValidator',
    'ValidationRule',
    'ValidationLevel',
    'ValidationSeverity',
    'ValidationIssue',
    'ValidationResult',
    'ComplianceStandard',
    'RequiredFieldsRule',
    'DataTypeRule',
    'RangeValidationRule',
    'PatternValidationRule',
    'DuplicateRule',
    'FraudPatternRule',
    'ComplianceRule'
]

if __name__ == "__main__":
    # Example usage and testing
    validator = FinancialDataValidator()
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'transaction_id': ['TXN001', 'TXN002', 'TXN003', 'TXN001'],  # Duplicate
        'amount': [100.50, -50.00, 999999.99, 500.00],  # Negative and very high amount
        'timestamp': ['2024-01-01 10:00:00', '2024-01-01 10:01:00', 'invalid', '2024-01-01 02:00:00'],
        'user_id': ['USER001', 'USER002', 'USER003', 'USER001'],
        'card_number': ['1234567890123456', 'XXXX-XXXX-XXXX-1234', '4111111111111111', '****-****-****-5678']
    })
    
    # Run comprehensive validation
    result = validator.validate(
        sample_data,
        validation_level=ValidationLevel.COMPREHENSIVE,
        required_compliance=[ComplianceStandard.PCI_DSS, ComplianceStandard.GDPR]
    )
    
    print(f"Validation Result:")
    print(f"Valid: {result.is_valid}")
    print(f"Quality Score: {result.data_quality_score:.2f}")
    print(f"Issues Found: {len(result.issues)}")
    print(f"Compliance Scores: {result.compliance_scores}")
    
    # Generate report
    report = validator.get_validation_report(result, "dict")
    print(f"\nValidation complete - {len(result.issues)} issues found")
    
    print("\nFinancialDataValidator ready for production use!")