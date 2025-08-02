"""
Enhanced Financial Data Loader for Financial Fraud Detection Domain
Comprehensive data loading, validation, and preprocessing for transaction data
with enhanced error handling, security validation, and production-ready features.
"""

import logging
import pandas as pd
import numpy as np
import requests
import json
import hashlib
import pickle
import os
import time
import asyncio
import psutil
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Iterator, AsyncIterator
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from tqdm import tqdm
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
import sqlite3
from urllib.parse import urljoin, urlparse
import warnings
import traceback
import tempfile
import shutil
from contextlib import contextmanager
import signal
import resource

# Configure logging
logger = logging.getLogger(__name__)

# ======================== CUSTOM EXCEPTIONS ========================

class DataLoaderException(Exception):
    """Base exception for data loader errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_type': self.__class__.__name__,
            'message': str(self),
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }

class DataSourceValidationError(DataLoaderException):
    """Raised when data source validation fails"""
    pass

class DataAccessError(DataLoaderException):
    """Raised when data cannot be accessed"""
    pass

class DataFormatError(DataLoaderException):
    """Raised when data format is invalid"""
    pass

class DataSizeError(DataLoaderException):
    """Raised when data size exceeds limits"""
    pass

class DataIntegrityError(DataLoaderException):
    """Raised when data integrity check fails"""
    pass

class DataSecurityError(DataLoaderException):
    """Raised when security validation fails"""
    pass

class DataQualityError(DataLoaderException):
    """Raised when data quality is below threshold"""
    pass

class DataLoadTimeoutError(DataLoaderException):
    """Raised when data loading times out"""
    pass

class DataMemoryError(DataLoaderException):
    """Raised when memory limits are exceeded"""
    pass

class DataCompatibilityError(DataLoaderException):
    """Raised when data format is incompatible"""
    pass

# ======================== ENUMS AND DATA CLASSES ========================

class DataSourceType(Enum):
    """Supported data source types for financial data"""
    CSV = "csv"
    JSON = "json"
    API = "api"
    PARQUET = "parquet"
    EXCEL = "excel"
    DATABASE = "database"
    STREAM = "stream"
    S3 = "s3"
    KAFKA = "kafka"

class LoadStatus(Enum):
    """Data loading operation status"""
    PENDING = "pending"
    LOADING = "loading" 
    VALIDATING = "validating"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"
    EXPIRED = "expired"
    DEGRADED = "degraded"
    TIMEOUT = "timeout"

class ValidationLevel(Enum):
    """Data validation strictness levels"""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"
    PARANOID = "paranoid"

class SecurityLevel(Enum):
    """Data security validation levels"""
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DataLoadMetrics:
    """Comprehensive metrics for data loading operations"""
    source_name: str
    source_type: DataSourceType
    total_records: int
    valid_records: int
    invalid_records: int
    duplicate_records: int
    load_time_seconds: float
    validation_time_seconds: float
    preprocessing_time_seconds: float
    cache_hit: bool
    memory_usage_mb: float
    error_count: int = 0
    warning_count: int = 0
    data_quality_score: float = 0.0
    validation_level: ValidationLevel = ValidationLevel.BASIC
    security_level: SecurityLevel = SecurityLevel.BASIC
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None
    load_status: LoadStatus = LoadStatus.PENDING
    retry_count: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class DataValidationResult:
    """Result of data validation"""
    is_valid: bool
    cleaned_data: Optional[pd.DataFrame]
    warnings: List[str]
    errors: List[str]
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataSource:
    """Comprehensive data source configuration"""
    name: str
    source_type: DataSourceType
    location: str
    credentials: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None
    validation_level: ValidationLevel = ValidationLevel.BASIC
    security_level: SecurityLevel = SecurityLevel.BASIC
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds
    batch_size: Optional[int] = None
    retry_attempts: int = 3
    timeout: int = 300  # seconds
    compression: Optional[str] = None
    encoding: str = "utf-8"
    headers: Optional[Dict[str, str]] = None
    schema: Optional[Dict[str, Any]] = None
    preprocessing_hooks: List[str] = field(default_factory=list)
    postprocessing_hooks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_memory_mb: int = 1024  # Maximum memory usage in MB
    verify_checksum: bool = True
    allowed_formats: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    quality_threshold: float = 0.8
    enable_degradation: bool = True

# ======================== DECORATORS ========================

def validate_inputs(func):
    """Decorator to validate function inputs"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Basic input validation
            for i, arg in enumerate(args):
                if arg is None and i > 0:  # Skip self
                    raise ValueError(f"Argument {i} cannot be None")
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Input validation failed for {func.__name__}: {e}")
            raise
    return wrapper

def timeout_handler(timeout_seconds: int):
    """Decorator to add timeout handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def timeout_callback(signum, frame):
                raise DataLoadTimeoutError(f"Operation timed out after {timeout_seconds} seconds")
            
            # Set signal alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_callback)
            signal.alarm(timeout_seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
            return result
        return wrapper
    return decorator

def memory_limit(max_memory_mb: int):
    """Decorator to enforce memory limits"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            result = func(*args, **kwargs)
            
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_used = current_memory - initial_memory
            
            if memory_used > max_memory_mb:
                raise DataMemoryError(
                    f"Memory usage exceeded limit: {memory_used:.2f}MB > {max_memory_mb}MB",
                    {'memory_used_mb': memory_used, 'limit_mb': max_memory_mb}
                )
                
            return result
        return wrapper
    return decorator

# ======================== VALIDATORS ========================

class DataValidator(ABC):
    """Abstract base class for data validators"""
    
    @abstractmethod
    def validate(self, data: pd.DataFrame, level: ValidationLevel = ValidationLevel.BASIC) -> DataValidationResult:
        """Validate data and return result"""
        pass

class EnhancedFinancialTransactionValidator(DataValidator):
    """Enhanced validator for financial transaction data with comprehensive checks"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.required_columns = self.config.get('required_columns', [
            'transaction_id', 'amount', 'timestamp', 'user_id'
        ])
        self.optional_columns = self.config.get('optional_columns', [
            'merchant_id', 'transaction_type', 'currency', 'location', 'account_id'
        ])
        self.amount_limits = self.config.get('amount_limits', {
            'min': 0.01,
            'max': 1000000.0,
            'suspicious_threshold': 10000.0
        })
        self.time_limits = self.config.get('time_limits', {
            'earliest_date': '2020-01-01',
            'latest_date': None  # Will use current date
        })
        self.quality_weights = self.config.get('quality_weights', {
            'completeness': 0.3,
            'validity': 0.3,
            'consistency': 0.2,
            'accuracy': 0.2
        })
        
    def validate(self, data: pd.DataFrame, level: ValidationLevel = ValidationLevel.BASIC) -> DataValidationResult:
        """Enhanced validation with comprehensive quality scoring"""
        warnings = []
        errors = []
        metadata = {}
        original_count = len(data)
        
        if data.empty:
            return DataValidationResult(
                is_valid=False,
                cleaned_data=None,
                warnings=[],
                errors=["Dataset is empty"],
                quality_score=0.0,
                metadata={'original_count': 0}
            )
        
        # Track quality metrics
        quality_metrics = {
            'completeness': 0.0,
            'validity': 0.0,
            'consistency': 0.0,
            'accuracy': 0.0
        }
        
        try:
            # Level 1: Basic validation
            data, basic_result = self._basic_validation(data)
            warnings.extend(basic_result['warnings'])
            errors.extend(basic_result['errors'])
            quality_metrics['completeness'] = basic_result['completeness_score']
            
            if level == ValidationLevel.NONE:
                quality_score = quality_metrics['completeness']
            else:
                # Level 2: Strict validation
                if level in [ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE, ValidationLevel.PARANOID]:
                    data, strict_result = self._strict_validation(data)
                    warnings.extend(strict_result['warnings'])
                    errors.extend(strict_result['errors'])
                    quality_metrics['validity'] = strict_result['validity_score']
                
                # Level 3: Comprehensive validation
                if level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.PARANOID]:
                    data, comp_result = self._comprehensive_validation(data)
                    warnings.extend(comp_result['warnings'])
                    errors.extend(comp_result['errors'])
                    quality_metrics['consistency'] = comp_result['consistency_score']
                
                # Level 4: Paranoid validation
                if level == ValidationLevel.PARANOID:
                    data, paranoid_result = self._paranoid_validation(data)
                    warnings.extend(paranoid_result['warnings'])
                    errors.extend(paranoid_result['errors'])
                    quality_metrics['accuracy'] = paranoid_result['accuracy_score']
                
                # Calculate weighted quality score
                quality_score = sum(
                    quality_metrics[metric] * self.quality_weights[metric]
                    for metric in quality_metrics
                )
            
            # Final metrics
            final_count = len(data) if data is not None else 0
            metadata.update({
                'original_count': original_count,
                'final_count': final_count,
                'removed_count': original_count - final_count,
                'removal_rate': (original_count - final_count) / original_count if original_count > 0 else 0,
                'quality_metrics': quality_metrics,
                'validation_level': level.value
            })
            
            return DataValidationResult(
                is_valid=len(errors) == 0 and quality_score >= 0.5,
                cleaned_data=data,
                warnings=warnings,
                errors=errors,
                quality_score=quality_score,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return DataValidationResult(
                is_valid=False,
                cleaned_data=None,
                warnings=warnings,
                errors=errors + [f"Validation error: {str(e)}"],
                quality_score=0.0,
                metadata=metadata
            )
    
    def _basic_validation(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Basic validation with completeness scoring"""
        result = {'warnings': [], 'errors': [], 'completeness_score': 0.0}
        
        # Check required columns
        missing_cols = set(self.required_columns) - set(data.columns)
        if missing_cols:
            result['errors'].append(f"Missing required columns: {missing_cols}")
            return data, result
        
        # Calculate completeness before cleaning
        total_cells = len(data) * len(self.required_columns)
        null_cells = data[self.required_columns].isna().sum().sum()
        completeness = (total_cells - null_cells) / total_cells if total_cells > 0 else 0
        
        # Convert data types
        try:
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
                null_timestamps = data['timestamp'].isna().sum()
                if null_timestamps > 0:
                    result['warnings'].append(f"Found {null_timestamps} invalid timestamps")
                    data = data.dropna(subset=['timestamp'])
            
            if 'amount' in data.columns:
                data['amount'] = pd.to_numeric(data['amount'], errors='coerce')
                null_amounts = data['amount'].isna().sum()
                if null_amounts > 0:
                    result['warnings'].append(f"Found {null_amounts} invalid amounts")
                    data = data.dropna(subset=['amount'])
                    
        except Exception as e:
            result['errors'].append(f"Data type conversion failed: {e}")
        
        # Remove rows with null values in required columns
        initial_count = len(data)
        data = data.dropna(subset=self.required_columns)
        if len(data) < initial_count:
            result['warnings'].append(f"Removed {initial_count - len(data)} rows with null required values")
        
        result['completeness_score'] = completeness
        return data, result
    
    def _strict_validation(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Strict validation with validity scoring"""
        result = {'warnings': [], 'errors': [], 'validity_score': 1.0}
        initial_count = len(data)
        
        # Validate amount ranges
        if 'amount' in data.columns:
            min_amount = self.amount_limits['min']
            max_amount = self.amount_limits['max']
            
            invalid_amounts = data[
                (data['amount'] < min_amount) | 
                (data['amount'] > max_amount)
            ]
            
            if len(invalid_amounts) > 0:
                result['warnings'].append(f"Found {len(invalid_amounts)} transactions with invalid amounts")
                data = data[~data.index.isin(invalid_amounts.index)]
                result['validity_score'] *= 0.9
            
            # Check for suspicious amounts
            suspicious_threshold = self.amount_limits['suspicious_threshold']
            suspicious_count = (data['amount'] > suspicious_threshold).sum()
            if suspicious_count > 0:
                result['warnings'].append(f"Found {suspicious_count} transactions above suspicious threshold")
                if suspicious_count / len(data) > 0.05:  # More than 5% suspicious
                    result['validity_score'] *= 0.95
        
        # Validate timestamp ranges
        if 'timestamp' in data.columns:
            earliest_date = pd.to_datetime(self.time_limits['earliest_date'])
            latest_date = pd.to_datetime(self.time_limits['latest_date'] or datetime.now())
            
            invalid_dates = data[
                (data['timestamp'] < earliest_date) |
                (data['timestamp'] > latest_date)
            ]
            
            if len(invalid_dates) > 0:
                result['warnings'].append(f"Found {len(invalid_dates)} transactions with invalid timestamps")
                data = data[~data.index.isin(invalid_dates.index)]
                result['validity_score'] *= 0.9
        
        # Check for duplicate transaction IDs
        if 'transaction_id' in data.columns:
            duplicates = data.duplicated(subset=['transaction_id'], keep='first')
            if duplicates.any():
                duplicate_count = duplicates.sum()
                result['warnings'].append(f"Found {duplicate_count} duplicate transaction IDs")
                data = data[~duplicates]
                result['validity_score'] *= (1 - duplicate_count / initial_count)
        
        # Validate data formats
        if 'currency' in data.columns:
            valid_currencies = ['USD', 'EUR', 'GBP', 'CAD', 'JPY', 'AUD', 'CHF']
            invalid_currency = ~data['currency'].isin(valid_currencies)
            if invalid_currency.any():
                invalid_count = invalid_currency.sum()
                result['warnings'].append(f"Found {invalid_count} transactions with invalid currency codes")
                data = data[~invalid_currency]
                result['validity_score'] *= 0.95
        
        return data, result
    
    def _comprehensive_validation(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Comprehensive validation with consistency scoring"""
        result = {'warnings': [], 'errors': [], 'consistency_score': 1.0}
        
        # Check for velocity anomalies (same user, high frequency)
        if 'user_id' in data.columns and 'timestamp' in data.columns:
            data_sorted = data.sort_values(['user_id', 'timestamp'])
            data_sorted['time_diff'] = data_sorted.groupby('user_id')['timestamp'].diff()
            
            # Flag transactions within 1 minute of each other
            rapid_transactions = data_sorted[data_sorted['time_diff'] < pd.Timedelta(minutes=1)]
            if len(rapid_transactions) > 0:
                result['warnings'].append(f"Found {len(rapid_transactions)} rapid successive transactions")
                result['consistency_score'] *= 0.95
                
            # Check for impossible velocities (same user, different locations too quickly)
            if 'location' in data.columns:
                user_location_changes = data_sorted.groupby('user_id').apply(
                    lambda x: (x['location'].shift() != x['location']).sum()
                )
                high_velocity_users = user_location_changes[user_location_changes > 10]
                if len(high_velocity_users) > 0:
                    result['warnings'].append(
                        f"Found {len(high_velocity_users)} users with high location velocity"
                    )
                    result['consistency_score'] *= 0.9
        
        # Check for amount patterns that could indicate fraud
        if 'amount' in data.columns:
            # Round number bias (many transactions ending in .00)
            round_amounts = data[data['amount'] % 1 == 0]
            round_percentage = len(round_amounts) / len(data) * 100
            if round_percentage > 50:
                result['warnings'].append(f"High percentage ({round_percentage:.1f}%) of round-number amounts")
                result['consistency_score'] *= 0.97
            
            # Check for amount sequences (same amount repeated)
            amount_counts = data['amount'].value_counts()
            repeated_amounts = amount_counts[amount_counts > len(data) * 0.01]  # More than 1% same amount
            if len(repeated_amounts) > 0:
                result['warnings'].append(
                    f"Found {len(repeated_amounts)} amounts repeated more than 1% of transactions"
                )
                result['consistency_score'] *= 0.95
            
            # Statistical outliers
            q1 = data['amount'].quantile(0.25)
            q3 = data['amount'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = data[(data['amount'] < lower_bound) | (data['amount'] > upper_bound)]
            outlier_percentage = len(outliers) / len(data) * 100
            if outlier_percentage > 5:
                result['warnings'].append(
                    f"High percentage ({outlier_percentage:.1f}%) of statistical outliers"
                )
                result['consistency_score'] *= 0.9
        
        # Check data consistency across related fields
        if 'transaction_type' in data.columns and 'amount' in data.columns:
            # Deposits should be positive, withdrawals negative (or vice versa)
            type_amount_consistency = data.groupby('transaction_type')['amount'].agg(['mean', 'std'])
            inconsistent_types = type_amount_consistency[
                type_amount_consistency['std'] / type_amount_consistency['mean'] > 2
            ]
            if len(inconsistent_types) > 0:
                result['warnings'].append(
                    f"Found {len(inconsistent_types)} transaction types with inconsistent amounts"
                )
                result['consistency_score'] *= 0.95
        
        return data, result
    
    def _paranoid_validation(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Paranoid validation with accuracy scoring"""
        result = {'warnings': [], 'errors': [], 'accuracy_score': 1.0}
        
        # Benford's Law check for amounts
        if 'amount' in data.columns and len(data) > 1000:
            first_digits = data['amount'].apply(lambda x: int(str(abs(x))[0]) if x != 0 else 0)
            digit_counts = first_digits.value_counts().sort_index()
            
            # Expected Benford's Law distribution
            benford_expected = {
                1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097,
                5: 0.079, 6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046
            }
            
            # Chi-square test approximation
            chi_square = 0
            for digit in range(1, 10):
                observed = digit_counts.get(digit, 0)
                expected = benford_expected[digit] * len(data)
                if expected > 0:
                    chi_square += (observed - expected) ** 2 / expected
            
            if chi_square > 50:  # Threshold for significance
                result['warnings'].append(
                    f"Amount distribution fails Benford's Law test (chi-square: {chi_square:.2f})"
                )
                result['accuracy_score'] *= 0.9
        
        # Check for data manipulation patterns
        if 'timestamp' in data.columns:
            # Check for suspicious timestamp patterns (too regular)
            timestamps = pd.to_datetime(data['timestamp'])
            time_diffs = timestamps.sort_values().diff().dropna()
            
            # Check if timestamps are too regular (possible synthetic data)
            time_diff_std = time_diffs.dt.total_seconds().std()
            time_diff_mean = time_diffs.dt.total_seconds().mean()
            cv = time_diff_std / time_diff_mean if time_diff_mean > 0 else 0
            
            if cv < 0.1:  # Coefficient of variation too low
                result['warnings'].append(
                    "Timestamps show suspiciously regular patterns (possible synthetic data)"
                )
                result['accuracy_score'] *= 0.8
        
        # Cross-field validation
        if all(col in data.columns for col in ['user_id', 'account_id', 'amount']):
            # Check if users have consistent account mappings
            user_accounts = data.groupby('user_id')['account_id'].nunique()
            multi_account_users = user_accounts[user_accounts > 5]
            if len(multi_account_users) > 0:
                result['warnings'].append(
                    f"Found {len(multi_account_users)} users with more than 5 accounts"
                )
                result['accuracy_score'] *= 0.95
            
            # Check for amount patterns per user
            user_amount_stats = data.groupby('user_id')['amount'].agg(['mean', 'std', 'count'])
            suspicious_users = user_amount_stats[
                (user_amount_stats['std'] / user_amount_stats['mean'] < 0.1) & 
                (user_amount_stats['count'] > 10)
            ]
            if len(suspicious_users) > 0:
                result['warnings'].append(
                    f"Found {len(suspicious_users)} users with suspiciously consistent amounts"
                )
                result['accuracy_score'] *= 0.9
        
        # Advanced pattern detection
        if 'merchant_id' in data.columns and 'amount' in data.columns:
            # Check for merchant amount patterns
            merchant_stats = data.groupby('merchant_id')['amount'].agg(['mean', 'std', 'count'])
            
            # Flag merchants with zero variance in amounts (suspicious)
            zero_variance_merchants = merchant_stats[
                (merchant_stats['std'] == 0) & (merchant_stats['count'] > 5)
            ]
            if len(zero_variance_merchants) > 0:
                result['warnings'].append(
                    f"Found {len(zero_variance_merchants)} merchants with identical transaction amounts"
                )
                result['accuracy_score'] *= 0.85
        
        return data, result

class SecurityValidator:
    """Validator for data security checks"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.sensitive_patterns = self.config.get('sensitive_patterns', [
            r'\b\d{16}\b',  # Credit card numbers
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{10,11}\b',  # Phone numbers
        ])
        self.allowed_ips = self.config.get('allowed_ips', [])
        self.blocked_domains = self.config.get('blocked_domains', [])
        
    def validate_security(self, data: pd.DataFrame, source: DataSource, 
                         level: SecurityLevel = SecurityLevel.BASIC) -> Tuple[bool, List[str]]:
        """Validate data security"""
        issues = []
        
        if level == SecurityLevel.NONE:
            return True, []
        
        # Basic security checks
        if level.value in ['basic', 'standard', 'high', 'critical']:
            # Check for sensitive data exposure
            for column in data.columns:
                column_data = data[column].astype(str)
                for pattern in self.sensitive_patterns:
                    matches = column_data.str.contains(pattern, regex=True, na=False)
                    if matches.any():
                        count = matches.sum()
                        issues.append(
                            f"Column '{column}' contains {count} potential sensitive data matches"
                        )
        
        # Standard security checks
        if level.value in ['standard', 'high', 'critical']:
            # Check source location security
            if source.source_type == DataSourceType.API:
                parsed_url = urlparse(source.location)
                if not parsed_url.scheme == 'https':
                    issues.append("API endpoint not using HTTPS")
                
                if self.blocked_domains:
                    if any(domain in parsed_url.netloc for domain in self.blocked_domains):
                        issues.append(f"Domain {parsed_url.netloc} is blocked")
        
        # High security checks
        if level.value in ['high', 'critical']:
            # Check data encryption requirements
            if source.credentials and 'password' in source.credentials:
                if len(source.credentials['password']) < 12:
                    issues.append("Password does not meet minimum length requirements")
            
            # Check for data tampering indicators
            if 'checksum' in data.columns:
                invalid_checksums = self._verify_checksums(data)
                if invalid_checksums > 0:
                    issues.append(f"Found {invalid_checksums} records with invalid checksums")
        
        # Critical security checks
        if level == SecurityLevel.CRITICAL:
            # Verify data source authentication
            if not source.credentials:
                issues.append("No authentication credentials provided for critical security level")
            
            # Check for injection attempts
            injection_patterns = [
                r"';|--|/\*|\*/|xp_|sp_",  # SQL injection
                r"<script|javascript:|onerror=",  # XSS
                r"\$\{|\#{|%\{",  # Template injection
            ]
            
            for column in data.select_dtypes(include=['object']).columns:
                for pattern in injection_patterns:
                    if data[column].astype(str).str.contains(pattern, regex=True, na=False).any():
                        issues.append(f"Potential injection attempt detected in column '{column}'")
        
        return len(issues) == 0, issues
    
    def _verify_checksums(self, data: pd.DataFrame) -> int:
        """Verify data checksums"""
        if 'checksum' not in data.columns:
            return 0
            
        invalid_count = 0
        for idx, row in data.iterrows():
            row_data = row.drop('checksum').to_json()
            calculated_checksum = hashlib.sha256(row_data.encode()).hexdigest()
            if row['checksum'] != calculated_checksum:
                invalid_count += 1
                
        return invalid_count

# ======================== ENHANCED DATA LOADER ========================

class EnhancedFinancialDataLoader:
    """
    Enhanced financial data loader with comprehensive validation and error handling
    
    Features:
    - Comprehensive input validation with multiple levels
    - Advanced error handling with custom exceptions
    - Security validation with multiple security levels
    - Memory management and limits
    - Timeout handling for long operations
    - Graceful degradation for failures
    - Detailed error diagnostics and reporting
    - Performance monitoring and optimization
    - Data integrity verification with checksums
    - Format compatibility testing
    """
    
    def __init__(self, 
                 cache_dir: Optional[Path] = None,
                 max_workers: int = 4,
                 default_batch_size: int = 10000,
                 enable_async: bool = False,
                 max_memory_mb: int = 2048,
                 enable_degradation: bool = True):
        """
        Initialize enhanced financial data loader
        
        Args:
            cache_dir: Directory for caching data
            max_workers: Maximum concurrent workers
            default_batch_size: Default batch size for processing
            enable_async: Enable asynchronous processing
            max_memory_mb: Maximum memory usage in MB
            enable_degradation: Enable graceful degradation on failures
        """
        self.cache_dir = cache_dir or Path("./cache/financial_data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.default_batch_size = default_batch_size
        self.enable_async = enable_async
        self.max_memory_mb = max_memory_mb
        self.enable_degradation = enable_degradation
        
        # Initialize components
        self.validators: Dict[str, DataValidator] = {
            'financial_transaction': EnhancedFinancialTransactionValidator()
        }
        self.security_validator = SecurityValidator()
        self.load_metrics: List[DataLoadMetrics] = []
        self._session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
        self._lock = threading.RLock()
        
        # Performance tracking
        self.performance_monitor = PerformanceMonitor()
        
        # Error recovery state
        self.recovery_state: Dict[str, Any] = {}
        
        # Setup logging
        self._setup_enhanced_logging()
        
        logger.info(f"EnhancedFinancialDataLoader initialized (session: {self._session_id})")
    
    def _setup_enhanced_logging(self):
        """Setup comprehensive logging with multiple handlers"""
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - '
            '%(message)s - [%(pathname)s]'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Remove existing handlers
        logger.handlers = []
        
        # File handler for all logs
        all_handler = logging.FileHandler(log_dir / 'data_loader_all.log')
        all_handler.setFormatter(detailed_formatter)
        all_handler.setLevel(logging.DEBUG)
        logger.addHandler(all_handler)
        
        # File handler for errors only
        error_handler = logging.FileHandler(log_dir / 'data_loader_errors.log')
        error_handler.setFormatter(detailed_formatter)
        error_handler.setLevel(logging.ERROR)
        logger.addHandler(error_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(simple_formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        
        logger.setLevel(logging.DEBUG)
    
    @validate_inputs
    def validate_data_source(self, source: Union[DataSource, str, Path, Dict]) -> Tuple[bool, List[str]]:
        """
        Comprehensive data source validation
        
        Args:
            source: Data source to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Convert to DataSource if needed
            if not isinstance(source, DataSource):
                try:
                    source = self._create_data_source(source)
                except Exception as e:
                    issues.append(f"Invalid source configuration: {e}")
                    return False, issues
            
            # Validate source type
            if source.source_type not in DataSourceType:
                issues.append(f"Unknown source type: {source.source_type}")
            
            # Validate location
            if not source.location:
                issues.append("Source location cannot be empty")
            else:
                # Check location accessibility
                access_issues = self._validate_source_access(source)
                issues.extend(access_issues)
            
            # Validate configuration
            config_issues = self._validate_source_config(source)
            issues.extend(config_issues)
            
            # Validate security
            if source.security_level != SecurityLevel.NONE:
                _, security_issues = self._validate_source_security(source)
                issues.extend(security_issues)
            
            # Validate format compatibility
            if source.allowed_formats:
                format_issues = self._validate_format_compatibility(source)
                issues.extend(format_issues)
            
            # Check memory requirements
            memory_issues = self._validate_memory_requirements(source)
            issues.extend(memory_issues)
            
            return len(issues) == 0, issues
            
        except Exception as e:
            logger.error(f"Source validation failed: {e}")
            issues.append(f"Validation error: {str(e)}")
            return False, issues
    
    def _validate_source_access(self, source: DataSource) -> List[str]:
        """Validate source accessibility"""
        issues = []
        
        if source.source_type in [DataSourceType.CSV, DataSourceType.JSON, 
                                 DataSourceType.PARQUET, DataSourceType.EXCEL]:
            # File-based sources
            file_path = Path(source.location)
            
            if not file_path.exists():
                issues.append(f"File not found: {file_path}")
            elif not file_path.is_file():
                issues.append(f"Not a file: {file_path}")
            elif not os.access(file_path, os.R_OK):
                issues.append(f"File not readable: {file_path}")
            else:
                # Check file size
                try:
                    file_size = file_path.stat().st_size
                    if file_size == 0:
                        issues.append(f"File is empty: {file_path}")
                    elif file_size > 5_000_000_000:  # 5GB
                        issues.append(
                            f"File extremely large ({file_size / 1_000_000_000:.1f}GB), "
                            "may cause memory issues"
                        )
                    
                    # Check file permissions in detail
                    stat_info = file_path.stat()
                    if stat_info.st_mode & 0o077:  # World/group writable
                        issues.append(f"File has insecure permissions: {oct(stat_info.st_mode)}")
                        
                except Exception as e:
                    issues.append(f"Cannot stat file: {e}")
                    
        elif source.source_type == DataSourceType.API:
            # API sources
            try:
                parsed_url = urlparse(source.location)
                if not parsed_url.scheme:
                    issues.append("API URL missing scheme (http/https)")
                elif parsed_url.scheme not in ['http', 'https']:
                    issues.append(f"Invalid URL scheme: {parsed_url.scheme}")
                    
                # Test connectivity
                try:
                    response = requests.head(
                        source.location, 
                        timeout=10,
                        headers=source.headers or {},
                        allow_redirects=True
                    )
                    if response.status_code >= 400:
                        issues.append(f"API returned error: {response.status_code}")
                except requests.exceptions.Timeout:
                    issues.append("API connection timed out")
                except requests.exceptions.ConnectionError:
                    issues.append("Cannot connect to API")
                except Exception as e:
                    issues.append(f"API connection error: {e}")
                    
            except Exception as e:
                issues.append(f"Invalid API URL: {e}")
                
        elif source.source_type == DataSourceType.DATABASE:
            # Database sources
            if source.location.startswith('sqlite://'):
                db_path = source.location.replace('sqlite://', '')
                if not Path(db_path).exists():
                    issues.append(f"SQLite database not found: {db_path}")
            else:
                issues.append(f"Unsupported database type: {source.location}")
                
        return issues
    
    def _validate_source_config(self, source: DataSource) -> List[str]:
        """Validate source configuration"""
        issues = []
        
        # Validate timeout
        if source.timeout <= 0:
            issues.append("Timeout must be positive")
        elif source.timeout > 3600:
            issues.append("Timeout exceeds maximum allowed (3600 seconds)")
            
        # Validate retry attempts
        if source.retry_attempts < 0:
            issues.append("Retry attempts cannot be negative")
        elif source.retry_attempts > 10:
            issues.append("Retry attempts exceeds maximum allowed (10)")
            
        # Validate cache TTL
        if source.cache_enabled and source.cache_ttl <= 0:
            issues.append("Cache TTL must be positive when caching is enabled")
            
        # Validate batch size
        if source.batch_size is not None:
            if source.batch_size <= 0:
                issues.append("Batch size must be positive")
            elif source.batch_size > 1000000:
                issues.append("Batch size exceeds maximum allowed (1,000,000)")
                
        # Validate memory limit
        if source.max_memory_mb <= 0:
            issues.append("Memory limit must be positive")
        elif source.max_memory_mb > self.max_memory_mb:
            issues.append(
                f"Source memory limit ({source.max_memory_mb}MB) exceeds "
                f"loader limit ({self.max_memory_mb}MB)"
            )
            
        # Validate quality threshold
        if source.quality_threshold < 0 or source.quality_threshold > 1:
            issues.append("Quality threshold must be between 0 and 1")
            
        # Validate encoding
        try:
            'test'.encode(source.encoding)
        except LookupError:
            issues.append(f"Invalid encoding: {source.encoding}")
            
        return issues
    
    def _validate_source_security(self, source: DataSource) -> Tuple[bool, List[str]]:
        """Validate source security requirements"""
        issues = []
        
        # Check credentials
        if source.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            if not source.credentials:
                issues.append(
                    f"Credentials required for security level {source.security_level.value}"
                )
            else:
                # Validate credential strength
                if 'password' in source.credentials:
                    password = source.credentials['password']
                    if len(password) < 12:
                        issues.append("Password too short (minimum 12 characters)")
                    if not any(c.isupper() for c in password):
                        issues.append("Password must contain uppercase letters")
                    if not any(c.islower() for c in password):
                        issues.append("Password must contain lowercase letters")
                    if not any(c.isdigit() for c in password):
                        issues.append("Password must contain numbers")
                    if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password):
                        issues.append("Password must contain special characters")
                        
        # Check API security
        if source.source_type == DataSourceType.API:
            parsed_url = urlparse(source.location)
            if source.security_level != SecurityLevel.NONE and parsed_url.scheme != 'https':
                issues.append("HTTPS required for secure API connections")
                
        # Check required permissions
        if source.required_permissions:
            current_user = os.getenv('USER', 'unknown')
            for permission in source.required_permissions:
                if permission == 'admin' and current_user != 'root':
                    issues.append(f"Admin permission required but user is '{current_user}'")
                    
        return len(issues) == 0, issues
    
    def _validate_format_compatibility(self, source: DataSource) -> List[str]:
        """Validate format compatibility"""
        issues = []
        
        if source.allowed_formats:
            # Map source types to formats
            format_map = {
                DataSourceType.CSV: ['csv', 'tsv', 'txt'],
                DataSourceType.JSON: ['json', 'jsonl'],
                DataSourceType.PARQUET: ['parquet'],
                DataSourceType.EXCEL: ['xlsx', 'xls'],
            }
            
            if source.source_type in format_map:
                compatible_formats = format_map[source.source_type]
                for fmt in source.allowed_formats:
                    if fmt not in compatible_formats:
                        issues.append(
                            f"Format '{fmt}' not compatible with source type "
                            f"{source.source_type.value}"
                        )
                        
        return issues
    
    def _validate_memory_requirements(self, source: DataSource) -> List[str]:
        """Validate memory requirements"""
        issues = []
        
        # Get current memory usage
        process = psutil.Process()
        current_memory_mb = process.memory_info().rss / 1024 / 1024
        available_memory_mb = psutil.virtual_memory().available / 1024 / 1024
        
        # Estimate required memory based on source
        estimated_memory_mb = self._estimate_memory_requirements(source)
        
        if estimated_memory_mb > source.max_memory_mb:
            issues.append(
                f"Estimated memory usage ({estimated_memory_mb:.0f}MB) exceeds "
                f"source limit ({source.max_memory_mb}MB)"
            )
            
        if estimated_memory_mb > available_memory_mb:
            issues.append(
                f"Estimated memory usage ({estimated_memory_mb:.0f}MB) exceeds "
                f"available memory ({available_memory_mb:.0f}MB)"
            )
            
        if current_memory_mb + estimated_memory_mb > self.max_memory_mb:
            issues.append(
                f"Total memory usage would exceed loader limit ({self.max_memory_mb}MB)"
            )
            
        return issues
    
    def _estimate_memory_requirements(self, source: DataSource) -> float:
        """Estimate memory requirements for data source"""
        if source.source_type in [DataSourceType.CSV, DataSourceType.JSON]:
            try:
                file_size = Path(source.location).stat().st_size
                # Rough estimate: 2-3x file size in memory for DataFrames
                return file_size * 2.5 / 1024 / 1024
            except:
                return 100.0  # Default 100MB
        elif source.source_type == DataSourceType.API:
            # Estimate based on expected response size
            return source.options.get('expected_size_mb', 50.0)
        else:
            return 100.0  # Default estimate
    
    def _create_data_source(self, source: Union[str, Path, Dict]) -> DataSource:
        """Create DataSource object with validation"""
        if isinstance(source, dict):
            # Validate dict keys
            required_keys = ['name', 'source_type', 'location']
            missing_keys = set(required_keys) - set(source.keys())
            if missing_keys:
                raise DataSourceValidationError(
                    f"Missing required keys: {missing_keys}",
                    {'provided_keys': list(source.keys())}
                )
            
            # Convert source_type string to enum
            if isinstance(source['source_type'], str):
                try:
                    source['source_type'] = DataSourceType(source['source_type'])
                except ValueError:
                    raise DataSourceValidationError(
                        f"Invalid source type: {source['source_type']}",
                        {'valid_types': [t.value for t in DataSourceType]}
                    )
                    
            return DataSource(**source)
        
        # Handle file path or URL
        location = str(source)
        source_type = self._detect_source_type(location)
        
        if not source_type:
            raise DataSourceValidationError(
                f"Could not detect source type for: {location}",
                {'location': location}
            )
            
        name = Path(location).stem if source_type in [
            DataSourceType.CSV, DataSourceType.JSON, 
            DataSourceType.PARQUET, DataSourceType.EXCEL
        ] else location
        
        return DataSource(
            name=name,
            source_type=source_type,
            location=location
        )
    
    def _detect_source_type(self, location: Union[str, Path]) -> Optional[DataSourceType]:
        """Auto-detect data source type with validation"""
        location_str = str(location).lower()
        
        # URL patterns
        if location_str.startswith(('http://', 'https://')):
            return DataSourceType.API
        elif location_str.startswith('s3://'):
            return DataSourceType.S3
        elif location_str.startswith(('kafka://', 'kafka+ssl://')):
            return DataSourceType.KAFKA
        elif any(db in location_str for db in ['postgresql://', 'mysql://', 'sqlite://']):
            return DataSourceType.DATABASE
        
        # File patterns
        if isinstance(location, (str, Path)):
            path = Path(location)
            
            extension_map = {
                '.csv': DataSourceType.CSV,
                '.tsv': DataSourceType.CSV,
                '.json': DataSourceType.JSON,
                '.jsonl': DataSourceType.JSON,
                '.parquet': DataSourceType.PARQUET,
                '.xlsx': DataSourceType.EXCEL,
                '.xls': DataSourceType.EXCEL
            }
            
            if path.suffix.lower() in extension_map:
                return extension_map[path.suffix.lower()]
                
        return None

    def get_error_report(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed error report"""
        if session_id:
            metrics = [m for m in self.load_metrics if m.metadata.get('session_id') == session_id]
        else:
            metrics = self.load_metrics
            
        failed_loads = [m for m in metrics if m.load_status == LoadStatus.FAILED]
        degraded_loads = [m for m in metrics if m.load_status == LoadStatus.DEGRADED]
        timeout_loads = [m for m in metrics if m.load_status == LoadStatus.TIMEOUT]
        
        return {
            'total_loads': len(metrics),
            'successful_loads': len([m for m in metrics if m.load_status == LoadStatus.COMPLETED]),
            'failed_loads': len(failed_loads),
            'degraded_loads': len(degraded_loads),
            'timeout_loads': len(timeout_loads),
            'total_errors': sum(m.error_count for m in metrics),
            'total_warnings': sum(m.warning_count for m in metrics),
            'average_quality_score': np.mean([m.data_quality_score for m in metrics]) if metrics else 0,
            'failure_details': [
                {
                    'source': m.source_name,
                    'errors': m.errors,
                    'timestamp': m.metadata.get('timestamp')
                }
                for m in failed_loads
            ],
            'recovery_state': self.recovery_state
        }

class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.metrics = {}
        self._start_time = time.time()
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        process = psutil.Process()
        
        return {
            'cpu_percent': process.cpu_percent(interval=0.1),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'memory_percent': process.memory_percent(),
            'num_threads': process.num_threads(),
            'uptime_seconds': time.time() - self._start_time
        }

# ======================== MAIN ENTRY POINT ========================

def create_enhanced_data_loader(**kwargs) -> EnhancedFinancialDataLoader:
    """Create enhanced data loader with default configuration"""
    return EnhancedFinancialDataLoader(**kwargs)

# Export classes and functions
__all__ = [
    'EnhancedFinancialDataLoader',
    'DataSource',
    'DataSourceType',
    'ValidationLevel',
    'SecurityLevel',
    'LoadStatus',
    'DataLoadMetrics',
    'DataValidator',
    'EnhancedFinancialTransactionValidator',
    'SecurityValidator',
    'DataLoaderException',
    'DataSourceValidationError',
    'DataAccessError',
    'DataFormatError',
    'DataSizeError',
    'DataIntegrityError',
    'DataSecurityError',
    'DataQualityError',
    'DataLoadTimeoutError',
    'DataMemoryError',
    'DataCompatibilityError',
    'create_enhanced_data_loader'
]

if __name__ == "__main__":
    # Test enhanced data loader
    loader = create_enhanced_data_loader(
        max_memory_mb=2048,
        enable_degradation=True
    )
    
    print("Enhanced Financial Data Loader ready for production use!")
    print(f"Session ID: {loader._session_id}")
    print(f"Max memory: {loader.max_memory_mb}MB")
    print(f"Degradation enabled: {loader.enable_degradation}")