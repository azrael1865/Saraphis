"""
Enhanced Transaction Validator - Chunk 2: Transaction Validation Rules
Comprehensive transaction-specific validation rules with fraud detection,
business logic validation, and financial compliance checking.
"""

import logging
import pandas as pd
import numpy as np
import re
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set, Callable, Pattern
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
from collections import defaultdict, Counter
import hashlib
import warnings

# Configure logging
logger = logging.getLogger(__name__)

# Import enhanced framework components
try:
    from enhanced_validator_framework import (
        TransactionValidationError, BusinessRuleValidationError, ThresholdValidationError,
        enhanced_input_validator, enhanced_error_handler, enhanced_performance_monitor,
        business_rule_validator, ValidationMetricsCollector
    )
    from data_validator import (
        ValidationLevel, ValidationSeverity, ComplianceStandard,
        ValidationIssue, ValidationResult, ValidationRule
    )
    FRAMEWORK_AVAILABLE = True
except ImportError as e:
    FRAMEWORK_AVAILABLE = False
    logger.warning(f"Enhanced framework not available: {e}")
    
    # Provide fallback decorators when framework is not available
    def enhanced_input_validator(required_fields=None):
        def decorator(func):
            return func
        return decorator
    
    def enhanced_error_handler(recovery_strategy=None):
        def decorator(func):
            return func
        return decorator
    
    def enhanced_performance_monitor(metric_name=None):
        def decorator(func):
            return func
        return decorator
    
    def business_rule_validator(rules=None):
        def decorator(func):
            return func
        return decorator
    
    # Provide fallback classes when framework is not available
    class ValidationIssue:
        def __init__(self, field=None, message=None, severity=None, **kwargs):
            self.field = field
            self.message = message
            self.severity = severity
    
    class ValidationResult:
        def __init__(self, success=True, issues=None, **kwargs):
            self.success = success
            self.issues = issues or []
    
    class ValidationRule:
        def __init__(self, name=None, **kwargs):
            self.name = name
    
    class ValidationLevel:
        BASIC = "basic"
        COMPREHENSIVE = "comprehensive"
    
    class ValidationSeverity:
        LOW = "low"
        MEDIUM = "medium"  
        HIGH = "high"
    
    class ComplianceStandard:
        PCI_DSS = "pci_dss"
        SOX = "sox"

# ======================== TRANSACTION VALIDATION ENUMS ========================

class TransactionType(Enum):
    """Financial transaction types"""
    PURCHASE = "purchase"
    REFUND = "refund"
    TRANSFER = "transfer"
    WITHDRAWAL = "withdrawal"
    DEPOSIT = "deposit"
    PAYMENT = "payment"
    CHARGEBACK = "chargeback"
    ADJUSTMENT = "adjustment"
    FEE = "fee"
    INTEREST = "interest"

class CurrencyCode(Enum):
    """ISO 4217 currency codes (major currencies)"""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    CAD = "CAD"
    AUD = "AUD"
    CNY = "CNY"
    SEK = "SEK"
    NOK = "NOK"

class PaymentMethod(Enum):
    """Payment method types"""
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    BANK_TRANSFER = "bank_transfer"
    DIGITAL_WALLET = "digital_wallet"
    CASH = "cash"
    CHECK = "check"
    CRYPTOCURRENCY = "cryptocurrency"
    GIFT_CARD = "gift_card"

class RiskLevel(Enum):
    """Transaction risk levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"

# ======================== TRANSACTION DATA CLASSES ========================

@dataclass
class TransactionPattern:
    """Pattern definition for transaction analysis"""
    pattern_id: str
    pattern_type: str  # 'velocity', 'amount', 'time', 'location', 'behavior'
    description: str
    threshold_config: Dict[str, Any]
    risk_score: float
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FraudIndicator:
    """Fraud indicator with scoring"""
    indicator_id: str
    indicator_type: str
    description: str
    weight: float
    confidence: float
    evidence: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ValidationContext:
    """Enhanced validation context for transactions"""
    validation_timestamp: datetime
    environment: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    business_rules: Dict[str, Any] = field(default_factory=dict)
    risk_tolerance: RiskLevel = RiskLevel.MEDIUM
    compliance_requirements: List[str] = field(default_factory=list)

# ======================== ENHANCED TRANSACTION VALIDATION RULES ========================

class EnhancedTransactionFieldValidator:
    """Enhanced field-level validation for transactions"""
    
    def __init__(self):
        self.field_patterns = {
            'transaction_id': re.compile(r'^[A-Z0-9]{8,32}$'),
            'user_id': re.compile(r'^[A-Z0-9]{6,20}$'),
            'merchant_id': re.compile(r'^(MERCHANT|MER)_[A-Z0-9]{4,16}$'),
            'card_number': re.compile(r'^\*+\d{4}$'),  # Masked card numbers
            'authorization_code': re.compile(r'^[A-Z0-9]{6,12}$'),
            'reference_number': re.compile(r'^[A-Z0-9\-]{8,25}$')
        }
        
        self.currency_symbols = {
            'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥',
            'CHF': 'CHF', 'CAD': 'C$', 'AUD': 'A$'
        }
        
        self.country_codes = {
            'US', 'CA', 'GB', 'DE', 'FR', 'JP', 'AU', 'CH', 'SE', 'NO'
        }
    
    @enhanced_input_validator(required_fields=['transaction_id'])
    @enhanced_error_handler(recovery_strategy='partial')
    @enhanced_performance_monitor('field_validation')
    def validate_transaction_fields(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Comprehensive field validation for transactions"""
        issues = []
        
        try:
            # Validate transaction IDs
            if 'transaction_id' in data.columns:
                issues.extend(self._validate_transaction_ids(data))
            
            # Validate monetary amounts
            if 'amount' in data.columns:
                issues.extend(self._validate_amounts(data))
            
            # Validate timestamps
            if 'timestamp' in data.columns:
                issues.extend(self._validate_timestamps(data))
            
            # Validate currency codes
            if 'currency' in data.columns:
                issues.extend(self._validate_currency_codes(data))
            
            # Validate payment methods
            if 'payment_method' in data.columns:
                issues.extend(self._validate_payment_methods(data))
            
            # Validate geographic data
            if any(col in data.columns for col in ['country', 'country_code']):
                issues.extend(self._validate_geographic_data(data))
            
            # Cross-field validation
            issues.extend(self._validate_field_relationships(data))
            
        except Exception as e:
            raise TransactionValidationError(
                f"Field validation failed: {e}",
                field="multiple",
                validation_rule="field_validation"
            )
        
        return issues
    
    def _validate_transaction_ids(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate transaction ID format and uniqueness"""
        issues = []
        
        # Format validation
        invalid_format = ~data['transaction_id'].astype(str).str.match(
            self.field_patterns['transaction_id']
        )
        
        for idx in data[invalid_format].index[:100]:  # Limit to prevent memory issues
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="transaction_id_format",
                message="Invalid transaction ID format",
                field="transaction_id",
                record_index=idx,
                value=str(data.loc[idx, 'transaction_id']),
                rule_id="txn_id_format"
            ))
        
        # Uniqueness validation
        duplicate_ids = data.duplicated(subset=['transaction_id'], keep=False)
        if duplicate_ids.any():
            duplicate_count = duplicate_ids.sum()
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="transaction_id_duplicate",
                message=f"Found {duplicate_count} duplicate transaction IDs",
                field="transaction_id",
                rule_id="txn_id_unique",
                metadata={'duplicate_count': duplicate_count}
            ))
        
        # Pattern analysis for suspicious IDs
        id_patterns = data['transaction_id'].astype(str).str.extract(r'([A-Z]+)(\d+)')
        if len(id_patterns.columns) >= 2:
            # Check for sequential patterns
            numeric_parts = pd.to_numeric(id_patterns[1], errors='coerce')
            if not numeric_parts.isna().all():
                sequential_count = (numeric_parts.diff() == 1).sum()
                if sequential_count > len(data) * 0.8:  # More than 80% sequential
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="transaction_id_pattern",
                        message="Suspicious sequential transaction ID pattern detected",
                        field="transaction_id",
                        rule_id="txn_id_pattern",
                        metadata={'sequential_percentage': sequential_count / len(data) * 100}
                    ))
        
        return issues
    
    def _validate_amounts(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Enhanced amount validation with fraud detection"""
        issues = []
        
        try:
            amounts = pd.to_numeric(data['amount'], errors='coerce')
            
            # Basic format validation
            invalid_amounts = amounts.isna() & data['amount'].notna()
            for idx in data[invalid_amounts].index[:50]:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="amount_format",
                    message="Non-numeric amount value",
                    field="amount",
                    record_index=idx,
                    value=str(data.loc[idx, 'amount']),
                    rule_id="amount_format"
                ))
            
            # Zero and negative amount validation
            zero_amounts = (amounts == 0)
            negative_amounts = (amounts < 0)
            
            # Check transaction type context for negative amounts
            if 'transaction_type' in data.columns:
                refund_types = data['transaction_type'].isin(['refund', 'chargeback', 'adjustment'])
                invalid_negative = negative_amounts & ~refund_types
                
                for idx in data[invalid_negative].index[:20]:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="amount_negative",
                        message="Negative amount for non-refund transaction",
                        field="amount",
                        record_index=idx,
                        value=float(amounts.loc[idx]),
                        rule_id="amount_sign"
                    ))
            
            # Statistical anomaly detection
            if len(amounts.dropna()) > 100:
                issues.extend(self._detect_amount_anomalies(amounts, data))
            
            # Benford's Law analysis
            if len(amounts.dropna()) > 1000:
                issues.extend(self._benford_law_analysis(amounts))
            
        except Exception as e:
            raise TransactionValidationError(
                f"Amount validation failed: {e}",
                field="amount",
                validation_rule="amount_validation"
            )
        
        return issues
    
    def _detect_amount_anomalies(self, amounts: pd.Series, data: pd.DataFrame) -> List[ValidationIssue]:
        """Detect statistical anomalies in transaction amounts"""
        issues = []
        
        clean_amounts = amounts.dropna()
        if len(clean_amounts) == 0:
            return issues
        
        # Calculate statistical measures
        q75, q25 = np.percentile(clean_amounts, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        # Detect outliers
        outliers = (amounts < lower_bound) | (amounts > upper_bound)
        outlier_count = outliers.sum()
        
        if outlier_count > len(amounts) * 0.05:  # More than 5% outliers
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="amount_outliers",
                message=f"High number of amount outliers detected: {outlier_count} ({outlier_count/len(amounts)*100:.1f}%)",
                field="amount",
                rule_id="amount_outliers",
                metadata={
                    'outlier_count': outlier_count,
                    'outlier_percentage': outlier_count / len(amounts) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
            ))
        
        # Round number analysis
        round_amounts = clean_amounts[clean_amounts == clean_amounts.round()]
        round_percentage = len(round_amounts) / len(clean_amounts) * 100
        
        if round_percentage > 80:  # More than 80% round numbers
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="amount_round_numbers",
                message=f"Suspicious high percentage of round amounts: {round_percentage:.1f}%",
                field="amount",
                rule_id="amount_round",
                metadata={'round_percentage': round_percentage}
            ))
        
        return issues
    
    def _benford_law_analysis(self, amounts: pd.Series) -> List[ValidationIssue]:
        """Apply Benford's Law analysis to detect potential fraud"""
        issues = []
        
        try:
            # Get first digits of positive amounts
            positive_amounts = amounts[amounts > 0]
            first_digits = positive_amounts.astype(str).str[0].astype(int)
            digit_freq = first_digits.value_counts(normalize=True).sort_index()
            
            # Expected Benford's Law distribution
            benford_expected = {
                1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097, 5: 0.079,
                6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046
            }
            
            # Calculate chi-square statistic
            chi_square = 0
            deviations = {}
            
            for digit in range(1, 10):
                expected = benford_expected[digit]
                actual = digit_freq.get(digit, 0)
                deviation = abs(actual - expected)
                deviations[digit] = deviation
                
                if expected > 0:
                    chi_square += ((actual - expected) ** 2) / expected
                
                # Flag significant deviations
                if deviation > 0.05:  # 5% deviation threshold
                    severity = ValidationSeverity.WARNING if deviation < 0.1 else ValidationSeverity.ERROR
                    issues.append(ValidationIssue(
                        severity=severity,
                        category="benford_law_deviation",
                        message=f"Benford's Law deviation for digit {digit}: expected {expected:.1%}, got {actual:.1%}",
                        field="amount",
                        rule_id="benford_law",
                        metadata={
                            'digit': digit,
                            'expected_frequency': expected,
                            'actual_frequency': actual,
                            'deviation': deviation
                        }
                    ))
            
            # Overall Benford's Law compliance
            if chi_square > 15.507:  # Critical value for 8 degrees of freedom at p=0.05
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="benford_law_overall",
                    message=f"Data does not follow Benford's Law (chi-square: {chi_square:.3f})",
                    field="amount",
                    rule_id="benford_law_overall",
                    metadata={
                        'chi_square': chi_square,
                        'critical_value': 15.507,
                        'sample_size': len(positive_amounts)
                    }
                ))
            
        except Exception as e:
            logger.warning(f"Benford's Law analysis failed: {e}")
        
        return issues
    
    def _validate_timestamps(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Enhanced timestamp validation with pattern analysis"""
        issues = []
        
        try:
            timestamps = pd.to_datetime(data['timestamp'], errors='coerce')
            
            # Basic format validation
            invalid_times = timestamps.isna() & data['timestamp'].notna()
            for idx in data[invalid_times].index[:50]:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="timestamp_format",
                    message="Invalid timestamp format",
                    field="timestamp",
                    record_index=idx,
                    value=str(data.loc[idx, 'timestamp']),
                    rule_id="timestamp_format"
                ))
            
            # Future timestamp detection
            future_threshold = datetime.now() + timedelta(hours=1)  # Allow 1 hour tolerance
            future_times = timestamps > future_threshold
            
            for idx in data[future_times].index[:20]:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="timestamp_future",
                    message="Future timestamp detected",
                    field="timestamp",
                    record_index=idx,
                    value=str(timestamps.loc[idx]),
                    rule_id="timestamp_future"
                ))
            
            # Temporal pattern analysis
            if len(timestamps.dropna()) > 100:
                issues.extend(self._analyze_temporal_patterns(timestamps))
            
        except Exception as e:
            raise TransactionValidationError(
                f"Timestamp validation failed: {e}",
                field="timestamp",
                validation_rule="timestamp_validation"
            )
        
        return issues
    
    def _analyze_temporal_patterns(self, timestamps: pd.Series) -> List[ValidationIssue]:
        """Analyze temporal patterns for anomalies"""
        issues = []
        
        clean_timestamps = timestamps.dropna().sort_values()
        if len(clean_timestamps) < 100:
            return issues
        
        # Time interval analysis
        time_diffs = clean_timestamps.diff().dropna()
        median_interval = time_diffs.median()
        
        # Detect suspicious clustering
        if median_interval < pd.Timedelta(seconds=1):
            clustering_percentage = (time_diffs < pd.Timedelta(seconds=1)).mean() * 100
            if clustering_percentage > 50:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="timestamp_clustering",
                    message=f"Suspicious timestamp clustering: {clustering_percentage:.1f}% within 1 second",
                    field="timestamp",
                    rule_id="timestamp_clustering",
                    metadata={
                        'median_interval_seconds': median_interval.total_seconds(),
                        'clustering_percentage': clustering_percentage
                    }
                ))
        
        # Business hours analysis
        business_hours = (
            (clean_timestamps.dt.hour >= 9) & 
            (clean_timestamps.dt.hour <= 17) &
            (clean_timestamps.dt.weekday < 5)
        )
        
        off_hours_percentage = (1 - business_hours.mean()) * 100
        if off_hours_percentage > 70:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="timestamp_off_hours",
                message=f"High percentage of off-hours transactions: {off_hours_percentage:.1f}%",
                field="timestamp",
                rule_id="timestamp_business_hours",
                metadata={'off_hours_percentage': off_hours_percentage}
            ))
        
        return issues
    
    def _validate_currency_codes(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate currency codes against ISO 4217"""
        issues = []
        
        valid_currencies = {code.value for code in CurrencyCode}
        currency_col = data['currency'].astype(str).str.upper()
        
        invalid_currencies = ~currency_col.isin(valid_currencies)
        for idx in data[invalid_currencies].index[:50]:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="currency_invalid",
                message="Invalid currency code",
                field="currency",
                record_index=idx,
                value=str(data.loc[idx, 'currency']),
                rule_id="currency_code"
            ))
        
        return issues
    
    def _validate_payment_methods(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate payment method values"""
        issues = []
        
        valid_methods = {method.value for method in PaymentMethod}
        payment_col = data['payment_method'].astype(str).str.lower()
        
        invalid_methods = ~payment_col.isin(valid_methods)
        for idx in data[invalid_methods].index[:50]:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="payment_method_invalid",
                message="Unknown payment method",
                field="payment_method",
                record_index=idx,
                value=str(data.loc[idx, 'payment_method']),
                rule_id="payment_method"
            ))
        
        return issues
    
    def _validate_geographic_data(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate geographic/location data"""
        issues = []
        
        if 'country_code' in data.columns:
            country_codes = data['country_code'].astype(str).str.upper()
            invalid_countries = ~country_codes.isin(self.country_codes)
            
            for idx in data[invalid_countries].index[:50]:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="country_code_invalid",
                    message="Invalid or suspicious country code",
                    field="country_code",
                    record_index=idx,
                    value=str(data.loc[idx, 'country_code']),
                    rule_id="country_code"
                ))
        
        return issues
    
    def _validate_field_relationships(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate relationships between fields"""
        issues = []
        
        # Amount vs transaction type consistency
        if all(col in data.columns for col in ['amount', 'transaction_type']):
            amounts = pd.to_numeric(data['amount'], errors='coerce')
            
            # Refunds should typically be negative or have specific indicators
            refund_mask = data['transaction_type'].str.lower().isin(['refund', 'chargeback'])
            positive_refunds = refund_mask & (amounts > 0)
            
            if positive_refunds.any():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="amount_type_mismatch",
                    message=f"Found {positive_refunds.sum()} refund transactions with positive amounts",
                    field="amount,transaction_type",
                    rule_id="amount_type_consistency",
                    metadata={'mismatch_count': positive_refunds.sum()}
                ))
        
        # Currency vs amount formatting
        if all(col in data.columns for col in ['amount', 'currency']):
            # Check for currency symbols in amount field
            amount_strings = data['amount'].astype(str)
            has_currency_symbols = amount_strings.str.contains(r'[$€£¥]', regex=True, na=False)
            
            if has_currency_symbols.any():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="amount_currency_format",
                    message=f"Found {has_currency_symbols.sum()} amounts with currency symbols",
                    field="amount,currency",
                    rule_id="amount_currency_format",
                    metadata={'symbol_count': has_currency_symbols.sum()}
                ))
        
        return issues

class EnhancedBusinessRuleValidator:
    """Enhanced business rule validation for financial transactions"""
    
    def __init__(self):
        self.velocity_thresholds = {
            'hourly_transaction_count': 50,
            'daily_transaction_count': 500,
            'hourly_amount_limit': 10000.0,
            'daily_amount_limit': 50000.0
        }
        
        self.suspicious_patterns = [
            TransactionPattern(
                pattern_id="round_amount_velocity",
                pattern_type="amount",
                description="High frequency of round number transactions",
                threshold_config={'round_percentage': 80, 'min_transactions': 10},
                risk_score=0.6
            ),
            TransactionPattern(
                pattern_id="micro_transaction_burst",
                pattern_type="velocity",
                description="Burst of micro-transactions",
                threshold_config={'amount_threshold': 1.0, 'count_threshold': 20, 'time_window': 3600},
                risk_score=0.7
            ),
            TransactionPattern(
                pattern_id="sequential_amounts",
                pattern_type="amount",
                description="Sequential or arithmetic progression in amounts",
                threshold_config={'progression_threshold': 0.8},
                risk_score=0.8
            )
        ]
    
    @business_rule_validator("velocity_limits", required_fields=['user_id', 'timestamp', 'amount'])
    @enhanced_performance_monitor('velocity_validation')
    def validate_velocity_limits(self, data: pd.DataFrame, context: ValidationContext) -> List[ValidationIssue]:
        """Validate transaction velocity limits"""
        issues = []
        
        try:
            # Convert timestamp
            timestamps = pd.to_datetime(data['timestamp'])
            amounts = pd.to_numeric(data['amount'], errors='coerce')
            
            # Group by user for velocity analysis
            for user_id, user_data in data.groupby('user_id'):
                user_timestamps = pd.to_datetime(user_data['timestamp'])
                user_amounts = pd.to_numeric(user_data['amount'], errors='coerce')
                
                # Hourly velocity check
                now = datetime.now()
                hour_ago = now - timedelta(hours=1)
                recent_transactions = user_timestamps >= hour_ago
                
                hourly_count = recent_transactions.sum()
                hourly_amount = user_amounts[recent_transactions].sum()
                
                # Check hourly transaction count
                if hourly_count > self.velocity_thresholds['hourly_transaction_count']:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="velocity_exceeded",
                        message=f"User {user_id} exceeded hourly transaction limit: {hourly_count} transactions",
                        field="user_id,timestamp",
                        rule_id="hourly_velocity",
                        metadata={
                            'user_id': user_id,
                            'transaction_count': hourly_count,
                            'limit': self.velocity_thresholds['hourly_transaction_count']
                        }
                    ))
                
                # Check hourly amount limit
                if hourly_amount > self.velocity_thresholds['hourly_amount_limit']:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="amount_velocity_exceeded",
                        message=f"User {user_id} exceeded hourly amount limit: ${hourly_amount:,.2f}",
                        field="user_id,amount",
                        rule_id="hourly_amount_velocity",
                        metadata={
                            'user_id': user_id,
                            'total_amount': hourly_amount,
                            'limit': self.velocity_thresholds['hourly_amount_limit']
                        }
                    ))
        
        except Exception as e:
            raise BusinessRuleValidationError(
                f"Velocity validation failed: {e}",
                rule_name="velocity_limits"
            )
        
        return issues
    
    @business_rule_validator("fraud_patterns", required_fields=['user_id', 'amount', 'timestamp'])
    @enhanced_performance_monitor('fraud_pattern_detection')
    def detect_fraud_patterns(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Detect sophisticated fraud patterns"""
        issues = []
        
        try:
            for pattern in self.suspicious_patterns:
                if not pattern.enabled:
                    continue
                
                pattern_issues = self._check_pattern(data, pattern)
                issues.extend(pattern_issues)
        
        except Exception as e:
            raise BusinessRuleValidationError(
                f"Fraud pattern detection failed: {e}",
                rule_name="fraud_patterns"
            )
        
        return issues
    
    def _check_pattern(self, data: pd.DataFrame, pattern: TransactionPattern) -> List[ValidationIssue]:
        """Check specific fraud pattern"""
        issues = []
        
        if pattern.pattern_type == "amount":
            if pattern.pattern_id == "round_amount_velocity":
                issues.extend(self._check_round_amount_pattern(data, pattern))
            elif pattern.pattern_id == "sequential_amounts":
                issues.extend(self._check_sequential_amount_pattern(data, pattern))
        
        elif pattern.pattern_type == "velocity":
            if pattern.pattern_id == "micro_transaction_burst":
                issues.extend(self._check_micro_transaction_burst(data, pattern))
        
        return issues
    
    def _check_round_amount_pattern(self, data: pd.DataFrame, pattern: TransactionPattern) -> List[ValidationIssue]:
        """Check for suspicious round amount patterns"""
        issues = []
        
        amounts = pd.to_numeric(data['amount'], errors='coerce').dropna()
        if len(amounts) < pattern.threshold_config['min_transactions']:
            return issues
        
        # Check for round numbers
        round_amounts = amounts[amounts == amounts.round()]
        round_percentage = len(round_amounts) / len(amounts) * 100
        
        if round_percentage >= pattern.threshold_config['round_percentage']:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="fraud_pattern_round_amounts",
                message=f"Suspicious round amount pattern: {round_percentage:.1f}% round numbers",
                field="amount",
                rule_id=pattern.pattern_id,
                metadata={
                    'round_percentage': round_percentage,
                    'risk_score': pattern.risk_score,
                    'pattern_description': pattern.description
                }
            ))
        
        return issues
    
    def _check_sequential_amount_pattern(self, data: pd.DataFrame, pattern: TransactionPattern) -> List[ValidationIssue]:
        """Check for sequential amount patterns"""
        issues = []
        
        amounts = pd.to_numeric(data['amount'], errors='coerce').dropna().sort_values()
        if len(amounts) < 5:  # Need minimum transactions
            return issues
        
        # Check for arithmetic progression
        diffs = amounts.diff().dropna()
        if len(diffs) > 0:
            # Check if most differences are the same (indicating arithmetic progression)
            most_common_diff = diffs.mode().iloc[0] if not diffs.mode().empty else 0
            same_diff_count = (diffs == most_common_diff).sum()
            progression_ratio = same_diff_count / len(diffs)
            
            if progression_ratio >= pattern.threshold_config['progression_threshold']:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="fraud_pattern_sequential",
                    message=f"Sequential amount pattern detected: {progression_ratio:.1%} arithmetic progression",
                    field="amount",
                    rule_id=pattern.pattern_id,
                    metadata={
                        'progression_ratio': progression_ratio,
                        'common_difference': most_common_diff,
                        'risk_score': pattern.risk_score
                    }
                ))
        
        return issues
    
    def _check_micro_transaction_burst(self, data: pd.DataFrame, pattern: TransactionPattern) -> List[ValidationIssue]:
        """Check for micro-transaction bursts"""
        issues = []
        
        amounts = pd.to_numeric(data['amount'], errors='coerce')
        timestamps = pd.to_datetime(data['timestamp'], errors='coerce')
        
        # Filter micro-transactions
        micro_mask = amounts <= pattern.threshold_config['amount_threshold']
        micro_data = data[micro_mask]
        
        if len(micro_data) < pattern.threshold_config['count_threshold']:
            return issues
        
        # Check time clustering
        micro_timestamps = pd.to_datetime(micro_data['timestamp']).sort_values()
        time_window = timedelta(seconds=pattern.threshold_config['time_window'])
        
        # Find bursts
        for i in range(len(micro_timestamps) - pattern.threshold_config['count_threshold'] + 1):
            window_start = micro_timestamps.iloc[i]
            window_end = window_start + time_window
            
            transactions_in_window = ((micro_timestamps >= window_start) & 
                                    (micro_timestamps <= window_end)).sum()
            
            if transactions_in_window >= pattern.threshold_config['count_threshold']:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="fraud_pattern_micro_burst",
                    message=f"Micro-transaction burst: {transactions_in_window} transactions under ${pattern.threshold_config['amount_threshold']} in {pattern.threshold_config['time_window']}s",
                    field="amount,timestamp",
                    rule_id=pattern.pattern_id,
                    metadata={
                        'burst_count': transactions_in_window,
                        'time_window': pattern.threshold_config['time_window'],
                        'amount_threshold': pattern.threshold_config['amount_threshold'],
                        'risk_score': pattern.risk_score
                    }
                ))
                break  # Only report first burst to avoid spam
        
        return issues

# Export enhanced transaction validation components
__all__ = [
    # Enums
    'TransactionType', 'CurrencyCode', 'PaymentMethod', 'RiskLevel',
    
    # Data classes
    'TransactionPattern', 'FraudIndicator', 'ValidationContext',
    
    # Validators
    'EnhancedTransactionFieldValidator', 'EnhancedBusinessRuleValidator'
]

if __name__ == "__main__":
    print("Enhanced Transaction Validator - Chunk 2: Transaction Validation Rules loaded")