#!/usr/bin/env python3
"""
Validation Utilities for Universal AI Core
==========================================

This module provides comprehensive validation utilities adapted from Saraphis patterns.
Extracted from validation patterns across the codebase, these utilities offer
domain-agnostic validation, type checking, and data quality assurance capabilities.

Features:
- Multi-level validation system with pluggable validators
- Schema validation and type checking
- Statistical validation and outlier detection
- Security validation and input sanitization
- Custom validation rules engine
- Data quality scoring and reporting
- Validation result aggregation and analysis
- Performance-optimized validation operations
"""

import logging
import time
import re
import warnings
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
import threading
import hashlib
import json

import numpy as np
import pandas as pd

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Categories of validation issues"""
    SCHEMA = "schema"
    TYPE = "type"
    RANGE = "range"
    FORMAT = "format"
    BUSINESS_RULE = "business_rule"
    STATISTICAL = "statistical"
    SECURITY = "security"
    CONSISTENCY = "consistency"
    COMPLETENESS = "completeness"
    UNIQUENESS = "uniqueness"


@dataclass
class ValidationIssue:
    """Individual validation issue adapted from Saraphis patterns"""
    category: ValidationCategory
    severity: ValidationSeverity
    rule_name: str
    message: str
    affected_rows: Optional[List[int]] = None
    affected_columns: Optional[List[str]] = None
    affected_values: Optional[List[Any]] = None
    suggestion: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    errors: List[ValidationIssue] = field(default_factory=list)
    critical_errors: List[ValidationIssue] = field(default_factory=list)
    validation_time: float = 0.0
    data_quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Categorize issues by severity"""
        for issue in self.issues:
            if issue.severity == ValidationSeverity.WARNING:
                self.warnings.append(issue)
            elif issue.severity == ValidationSeverity.ERROR:
                self.errors.append(issue)
            elif issue.severity == ValidationSeverity.CRITICAL:
                self.critical_errors.append(issue)
        
        # Update validity based on errors
        self.is_valid = len(self.errors) == 0 and len(self.critical_errors) == 0


@dataclass
class ValidationConfig:
    """Configuration for validation operations"""
    check_schema: bool = True
    check_types: bool = True
    check_ranges: bool = True
    check_formats: bool = True
    check_business_rules: bool = True
    check_statistics: bool = True
    check_security: bool = True
    check_consistency: bool = True
    check_completeness: bool = True
    check_uniqueness: bool = True
    
    # Thresholds
    missing_value_threshold: float = 0.1  # 10% missing values allowed
    outlier_threshold: float = 0.05  # 5% outliers allowed
    duplicate_threshold: float = 0.01  # 1% duplicates allowed
    
    # Statistical parameters
    outlier_method: str = "iqr"  # iqr, zscore, modified_zscore
    outlier_factor: float = 1.5
    correlation_threshold: float = 0.95
    
    # Security settings
    enable_sql_injection_check: bool = True
    enable_xss_check: bool = True
    enable_path_traversal_check: bool = True
    max_string_length: int = 10000
    
    # Performance settings
    enable_caching: bool = True
    cache_size: int = 1000
    parallel_processing: bool = True
    max_workers: int = 4


class ValidatorInterface(ABC):
    """Abstract interface for validators"""
    
    @abstractmethod
    def validate(self, data: Any, config: ValidationConfig) -> List[ValidationIssue]:
        """Validate data and return list of issues"""
        pass
    
    @abstractmethod
    def get_validator_name(self) -> str:
        """Get validator name"""
        pass


class SchemaValidator(ValidatorInterface):
    """
    Schema and structure validator adapted from Saraphis patterns.
    Validates data structure, column presence, and data types.
    """
    
    def __init__(self):
        self.cache = {}
        self._lock = threading.Lock()
    
    def validate(self, data: Any, config: ValidationConfig) -> List[ValidationIssue]:
        """Validate data schema and structure"""
        issues = []
        
        if not config.check_schema:
            return issues
        
        try:
            if isinstance(data, pd.DataFrame):
                issues.extend(self._validate_dataframe_schema(data, config))
            elif isinstance(data, dict):
                issues.extend(self._validate_dict_schema(data, config))
            elif isinstance(data, list):
                issues.extend(self._validate_list_schema(data, config))
            elif isinstance(data, np.ndarray):
                issues.extend(self._validate_array_schema(data, config))
            else:
                issues.append(ValidationIssue(
                    category=ValidationCategory.SCHEMA,
                    severity=ValidationSeverity.WARNING,
                    rule_name="unsupported_type",
                    message=f"Unsupported data type for schema validation: {type(data)}",
                    suggestion="Convert data to supported type (DataFrame, dict, list, or ndarray)"
                ))
                
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            issues.append(ValidationIssue(
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.ERROR,
                rule_name="validation_error",
                message=f"Schema validation failed: {str(e)}"
            ))
        
        return issues
    
    def _validate_dataframe_schema(self, df: pd.DataFrame, config: ValidationConfig) -> List[ValidationIssue]:
        """Validate DataFrame schema"""
        issues = []
        
        # Check for empty DataFrame
        if df.empty:
            issues.append(ValidationIssue(
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.WARNING,
                rule_name="empty_dataframe",
                message="DataFrame is empty",
                suggestion="Provide non-empty data for analysis"
            ))
            return issues
        
        # Check column names
        if df.columns.duplicated().any():
            duplicate_cols = df.columns[df.columns.duplicated()].tolist()
            issues.append(ValidationIssue(
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.ERROR,
                rule_name="duplicate_columns",
                message=f"Duplicate column names found: {duplicate_cols}",
                affected_columns=duplicate_cols,
                suggestion="Rename duplicate columns to have unique names"
            ))
        
        # Check for unnamed columns
        unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
        if unnamed_cols:
            issues.append(ValidationIssue(
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.WARNING,
                rule_name="unnamed_columns",
                message=f"Unnamed columns found: {unnamed_cols}",
                affected_columns=unnamed_cols,
                suggestion="Provide meaningful column names"
            ))
        
        # Check data types
        if config.check_types:
            for col in df.columns:
                col_dtype = df[col].dtype
                
                # Check for object columns that might need type conversion
                if col_dtype == 'object':
                    # Try to identify if it should be numeric
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        numeric_convertible = 0
                        for val in non_null_values.head(100):  # Sample check
                            try:
                                float(val)
                                numeric_convertible += 1
                            except (ValueError, TypeError):
                                break
                        
                        if numeric_convertible > len(non_null_values) * 0.8:
                            issues.append(ValidationIssue(
                                category=ValidationCategory.TYPE,
                                severity=ValidationSeverity.WARNING,
                                rule_name="potential_type_conversion",
                                message=f"Column '{col}' appears to contain numeric data but is stored as object",
                                affected_columns=[col],
                                suggestion=f"Convert column '{col}' to numeric type"
                            ))
        
        return issues
    
    def _validate_dict_schema(self, data: dict, config: ValidationConfig) -> List[ValidationIssue]:
        """Validate dictionary schema"""
        issues = []
        
        if not data:
            issues.append(ValidationIssue(
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.WARNING,
                rule_name="empty_dict",
                message="Dictionary is empty"
            ))
        
        # Check for reserved keys or problematic keys
        problematic_keys = []
        for key in data.keys():
            if not isinstance(key, (str, int, float)):
                problematic_keys.append(key)
            elif isinstance(key, str) and len(key) > 100:
                problematic_keys.append(key)
        
        if problematic_keys:
            issues.append(ValidationIssue(
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.WARNING,
                rule_name="problematic_keys",
                message=f"Potentially problematic dictionary keys: {problematic_keys[:5]}",
                suggestion="Use simple string or numeric keys"
            ))
        
        return issues
    
    def _validate_list_schema(self, data: list, config: ValidationConfig) -> List[ValidationIssue]:
        """Validate list schema"""
        issues = []
        
        if not data:
            issues.append(ValidationIssue(
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.WARNING,
                rule_name="empty_list",
                message="List is empty"
            ))
            return issues
        
        # Check type consistency
        if config.check_types and len(data) > 1:
            first_type = type(data[0])
            inconsistent_types = []
            
            for i, item in enumerate(data[:100]):  # Sample check
                if type(item) != first_type:
                    inconsistent_types.append((i, type(item)))
            
            if inconsistent_types:
                issues.append(ValidationIssue(
                    category=ValidationCategory.TYPE,
                    severity=ValidationSeverity.WARNING,
                    rule_name="inconsistent_types",
                    message=f"List contains inconsistent types. Expected {first_type}, found {len(inconsistent_types)} different types",
                    suggestion="Ensure all list items have the same type"
                ))
        
        return issues
    
    def _validate_array_schema(self, data: np.ndarray, config: ValidationConfig) -> List[ValidationIssue]:
        """Validate numpy array schema"""
        issues = []
        
        if data.size == 0:
            issues.append(ValidationIssue(
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.WARNING,
                rule_name="empty_array",
                message="Array is empty"
            ))
        
        # Check for extreme dimensions
        if len(data.shape) > 4:
            issues.append(ValidationIssue(
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.WARNING,
                rule_name="high_dimensionality",
                message=f"Array has {len(data.shape)} dimensions, which may be excessive",
                suggestion="Consider reducing dimensionality for better performance"
            ))
        
        # Check for very large arrays
        if data.nbytes > 1e9:  # 1GB
            issues.append(ValidationIssue(
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.WARNING,
                rule_name="large_array",
                message=f"Array size is {data.nbytes / 1e9:.1f}GB, which may cause memory issues",
                suggestion="Consider processing in chunks or reducing data size"
            ))
        
        return issues
    
    def get_validator_name(self) -> str:
        return "schema_validator"


class StatisticalValidator(ValidatorInterface):
    """
    Statistical validator adapted from Saraphis patterns.
    Validates data distributions, outliers, and statistical properties.
    """
    
    def __init__(self):
        self.cache = {}
        self._lock = threading.Lock()
    
    def validate(self, data: Any, config: ValidationConfig) -> List[ValidationIssue]:
        """Validate statistical properties of data"""
        issues = []
        
        if not config.check_statistics:
            return issues
        
        try:
            if isinstance(data, pd.DataFrame):
                issues.extend(self._validate_dataframe_statistics(data, config))
            elif isinstance(data, np.ndarray):
                issues.extend(self._validate_array_statistics(data, config))
            else:
                # Convert to array if possible
                try:
                    array_data = np.array(data)
                    issues.extend(self._validate_array_statistics(array_data, config))
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Statistical validation failed: {e}")
            issues.append(ValidationIssue(
                category=ValidationCategory.STATISTICAL,
                severity=ValidationSeverity.ERROR,
                rule_name="validation_error",
                message=f"Statistical validation failed: {str(e)}"
            ))
        
        return issues
    
    def _validate_dataframe_statistics(self, df: pd.DataFrame, config: ValidationConfig) -> List[ValidationIssue]:
        """Validate DataFrame statistics"""
        issues = []
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        for col, missing_count in missing_counts.items():
            if missing_count > 0:
                missing_ratio = missing_count / len(df)
                if missing_ratio > config.missing_value_threshold:
                    issues.append(ValidationIssue(
                        category=ValidationCategory.COMPLETENESS,
                        severity=ValidationSeverity.WARNING if missing_ratio < 0.5 else ValidationSeverity.ERROR,
                        rule_name="excessive_missing_values",
                        message=f"Column '{col}' has {missing_ratio:.1%} missing values (threshold: {config.missing_value_threshold:.1%})",
                        affected_columns=[col],
                        suggestion=f"Consider imputation or removal of column '{col}'"
                    ))
        
        # Check for duplicates
        if config.check_uniqueness:
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                duplicate_ratio = duplicate_count / len(df)
                if duplicate_ratio > config.duplicate_threshold:
                    issues.append(ValidationIssue(
                        category=ValidationCategory.UNIQUENESS,
                        severity=ValidationSeverity.WARNING,
                        rule_name="excessive_duplicates",
                        message=f"Found {duplicate_count} duplicate rows ({duplicate_ratio:.1%} of data)",
                        suggestion="Consider removing duplicate rows"
                    ))
        
        # Check numeric columns for outliers
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            col_data = df[col].dropna()
            if len(col_data) > 10:  # Need sufficient data for outlier detection
                outliers = self._detect_outliers(col_data.values, config)
                if len(outliers) > 0:
                    outlier_ratio = len(outliers) / len(col_data)
                    if outlier_ratio > config.outlier_threshold:
                        issues.append(ValidationIssue(
                            category=ValidationCategory.STATISTICAL,
                            severity=ValidationSeverity.WARNING,
                            rule_name="excessive_outliers",
                            message=f"Column '{col}' has {len(outliers)} outliers ({outlier_ratio:.1%} of non-null values)",
                            affected_columns=[col],
                            affected_rows=outliers.tolist(),
                            suggestion=f"Review outliers in column '{col}' for data quality issues"
                        ))
        
        # Check for highly correlated columns
        if len(numeric_columns) > 1 and config.correlation_threshold < 1.0:
            try:
                corr_matrix = df[numeric_columns].corr().abs()
                high_corr_pairs = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        if corr_matrix.iloc[i, j] > config.correlation_threshold:
                            high_corr_pairs.append((
                                corr_matrix.columns[i],
                                corr_matrix.columns[j],
                                corr_matrix.iloc[i, j]
                            ))
                
                if high_corr_pairs:
                    for col1, col2, corr_value in high_corr_pairs[:5]:  # Limit to first 5
                        issues.append(ValidationIssue(
                            category=ValidationCategory.STATISTICAL,
                            severity=ValidationSeverity.WARNING,
                            rule_name="high_correlation",
                            message=f"Columns '{col1}' and '{col2}' are highly correlated (r={corr_value:.3f})",
                            affected_columns=[col1, col2],
                            suggestion="Consider removing one of the highly correlated columns"
                        ))
            except Exception as e:
                logger.warning(f"Correlation analysis failed: {e}")
        
        return issues
    
    def _validate_array_statistics(self, data: np.ndarray, config: ValidationConfig) -> List[ValidationIssue]:
        """Validate array statistics"""
        issues = []
        
        # Check for missing values (NaN)
        if data.dtype.kind in ['f', 'c']:  # Float or complex
            nan_count = np.sum(np.isnan(data))
            if nan_count > 0:
                nan_ratio = nan_count / data.size
                if nan_ratio > config.missing_value_threshold:
                    issues.append(ValidationIssue(
                        category=ValidationCategory.COMPLETENESS,
                        severity=ValidationSeverity.WARNING if nan_ratio < 0.5 else ValidationSeverity.ERROR,
                        rule_name="excessive_nan_values",
                        message=f"Array has {nan_ratio:.1%} NaN values (threshold: {config.missing_value_threshold:.1%})",
                        suggestion="Consider NaN handling strategy"
                    ))
        
        # Check for outliers in numeric data
        if data.dtype.kind in ['f', 'i'] and data.size > 10:
            # Flatten array for outlier detection
            flat_data = data.flatten()
            valid_data = flat_data[~np.isnan(flat_data)] if data.dtype.kind == 'f' else flat_data
            
            if len(valid_data) > 10:
                outlier_indices = self._detect_outliers(valid_data, config)
                if len(outlier_indices) > 0:
                    outlier_ratio = len(outlier_indices) / len(valid_data)
                    if outlier_ratio > config.outlier_threshold:
                        issues.append(ValidationIssue(
                            category=ValidationCategory.STATISTICAL,
                            severity=ValidationSeverity.WARNING,
                            rule_name="excessive_outliers",
                            message=f"Array has {len(outlier_indices)} outliers ({outlier_ratio:.1%} of values)",
                            suggestion="Review outliers for data quality issues"
                        ))
        
        # Check for infinite values
        if data.dtype.kind in ['f', 'c']:
            inf_count = np.sum(np.isinf(data))
            if inf_count > 0:
                issues.append(ValidationIssue(
                    category=ValidationCategory.STATISTICAL,
                    severity=ValidationSeverity.ERROR,
                    rule_name="infinite_values",
                    message=f"Array contains {inf_count} infinite values",
                    suggestion="Replace infinite values with appropriate finite values"
                ))
        
        return issues
    
    def _detect_outliers(self, data: np.ndarray, config: ValidationConfig) -> np.ndarray:
        """Detect outliers using specified method"""
        if config.outlier_method == "iqr":
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - config.outlier_factor * IQR
            upper_bound = Q3 + config.outlier_factor * IQR
            outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
            
        elif config.outlier_method == "zscore":
            z_scores = np.abs(stats.zscore(data)) if SCIPY_AVAILABLE else np.abs((data - np.mean(data)) / np.std(data))
            outliers = np.where(z_scores > config.outlier_factor)[0]
            
        elif config.outlier_method == "modified_zscore":
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad if mad != 0 else np.zeros_like(data)
            outliers = np.where(np.abs(modified_z_scores) > config.outlier_factor)[0]
            
        else:
            outliers = np.array([])
        
        return outliers
    
    def get_validator_name(self) -> str:
        return "statistical_validator"


class SecurityValidator(ValidatorInterface):
    """
    Security validator adapted from Saraphis patterns.
    Validates input for security threats and malicious content.
    """
    
    def __init__(self):
        # Common attack patterns
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"('|(\\')|(;)|(\\)|(\-\-)|(/\\*)|(\\*/)|(@)|(\bOR\b))",
            r"(\b(AND|OR)\b.*(\b(SELECT|INSERT|UPDATE|DELETE)\b))",
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>",
            r"<object[^>]*>.*?</object>",
        ]
        
        self.path_traversal_patterns = [
            r"\.\./",
            r"\.\.\\",
            r"%2e%2e%2f",
            r"%2e%2e%5c",
        ]
        
        self.compiled_patterns = {
            'sql': [re.compile(pattern, re.IGNORECASE) for pattern in self.sql_injection_patterns],
            'xss': [re.compile(pattern, re.IGNORECASE) for pattern in self.xss_patterns],
            'path': [re.compile(pattern, re.IGNORECASE) for pattern in self.path_traversal_patterns]
        }
    
    def validate(self, data: Any, config: ValidationConfig) -> List[ValidationIssue]:
        """Validate data for security issues"""
        issues = []
        
        if not config.check_security:
            return issues
        
        try:
            if isinstance(data, str):
                issues.extend(self._validate_string_security(data, config))
            elif isinstance(data, (list, tuple)):
                for i, item in enumerate(data):
                    if isinstance(item, str):
                        item_issues = self._validate_string_security(item, config)
                        # Add context about list position
                        for issue in item_issues:
                            issue.context = {'list_index': i}
                        issues.extend(item_issues)
            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str):
                        value_issues = self._validate_string_security(value, config)
                        # Add context about dict key
                        for issue in value_issues:
                            issue.context = {'dict_key': key}
                        issues.extend(value_issues)
                    if isinstance(key, str):
                        key_issues = self._validate_string_security(key, config)
                        # Add context about dict key
                        for issue in key_issues:
                            issue.context = {'dict_key_validation': True}
                        issues.extend(key_issues)
            elif isinstance(data, pd.DataFrame):
                issues.extend(self._validate_dataframe_security(data, config))
                
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            issues.append(ValidationIssue(
                category=ValidationCategory.SECURITY,
                severity=ValidationSeverity.ERROR,
                rule_name="validation_error",
                message=f"Security validation failed: {str(e)}"
            ))
        
        return issues
    
    def _validate_string_security(self, text: str, config: ValidationConfig) -> List[ValidationIssue]:
        """Validate string for security issues"""
        issues = []
        
        # Check string length
        if len(text) > config.max_string_length:
            issues.append(ValidationIssue(
                category=ValidationCategory.SECURITY,
                severity=ValidationSeverity.WARNING,
                rule_name="excessive_string_length",
                message=f"String length ({len(text)}) exceeds maximum ({config.max_string_length})",
                suggestion="Truncate or validate string length"
            ))
        
        # Check for SQL injection
        if config.enable_sql_injection_check:
            for pattern in self.compiled_patterns['sql']:
                if pattern.search(text):
                    issues.append(ValidationIssue(
                        category=ValidationCategory.SECURITY,
                        severity=ValidationSeverity.CRITICAL,
                        rule_name="sql_injection_detected",
                        message="Potential SQL injection pattern detected",
                        affected_values=[text[:100]],
                        suggestion="Sanitize input and use parameterized queries"
                    ))
                    break
        
        # Check for XSS
        if config.enable_xss_check:
            for pattern in self.compiled_patterns['xss']:
                if pattern.search(text):
                    issues.append(ValidationIssue(
                        category=ValidationCategory.SECURITY,
                        severity=ValidationSeverity.CRITICAL,
                        rule_name="xss_detected",
                        message="Potential XSS pattern detected",
                        affected_values=[text[:100]],
                        suggestion="Sanitize HTML content and validate user input"
                    ))
                    break
        
        # Check for path traversal
        if config.enable_path_traversal_check:
            for pattern in self.compiled_patterns['path']:
                if pattern.search(text):
                    issues.append(ValidationIssue(
                        category=ValidationCategory.SECURITY,
                        severity=ValidationSeverity.CRITICAL,
                        rule_name="path_traversal_detected",
                        message="Potential path traversal pattern detected",
                        affected_values=[text[:100]],
                        suggestion="Validate and sanitize file paths"
                    ))
                    break
        
        return issues
    
    def _validate_dataframe_security(self, df: pd.DataFrame, config: ValidationConfig) -> List[ValidationIssue]:
        """Validate DataFrame for security issues"""
        issues = []
        
        # Check string columns for security issues
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            col_data = df[col].dropna()
            for idx, value in col_data.head(100).items():  # Sample check
                if isinstance(value, str):
                    value_issues = self._validate_string_security(value, config)
                    # Add context about DataFrame location
                    for issue in value_issues:
                        issue.context = {'column': col, 'row': idx}
                        issue.affected_columns = [col]
                        issue.affected_rows = [idx]
                    issues.extend(value_issues)
        
        return issues
    
    def get_validator_name(self) -> str:
        return "security_validator"


class CompletenessValidator(ValidatorInterface):
    """
    Completeness validator for data quality assessment.
    Checks for missing values, empty strings, and data availability.
    """
    
    def validate(self, data: Any, config: ValidationConfig) -> List[ValidationIssue]:
        """Validate data completeness"""
        issues = []
        
        if not config.check_completeness:
            return issues
        
        try:
            if isinstance(data, pd.DataFrame):
                issues.extend(self._validate_dataframe_completeness(data, config))
            elif isinstance(data, np.ndarray):
                issues.extend(self._validate_array_completeness(data, config))
            elif isinstance(data, (list, tuple)):
                issues.extend(self._validate_list_completeness(data, config))
            elif isinstance(data, dict):
                issues.extend(self._validate_dict_completeness(data, config))
                
        except Exception as e:
            logger.error(f"Completeness validation failed: {e}")
            issues.append(ValidationIssue(
                category=ValidationCategory.COMPLETENESS,
                severity=ValidationSeverity.ERROR,
                rule_name="validation_error",
                message=f"Completeness validation failed: {str(e)}"
            ))
        
        return issues
    
    def _validate_dataframe_completeness(self, df: pd.DataFrame, config: ValidationConfig) -> List[ValidationIssue]:
        """Validate DataFrame completeness"""
        issues = []
        
        # Check for completely empty columns
        for col in df.columns:
            if df[col].isnull().all():
                issues.append(ValidationIssue(
                    category=ValidationCategory.COMPLETENESS,
                    severity=ValidationSeverity.ERROR,
                    rule_name="empty_column",
                    message=f"Column '{col}' contains no data",
                    affected_columns=[col],
                    suggestion=f"Remove empty column '{col}' or populate with data"
                ))
            elif df[col].dtype == 'object':
                # Check for empty strings
                empty_strings = df[col].apply(lambda x: x == '' if isinstance(x, str) else False).sum()
                if empty_strings > 0:
                    empty_ratio = empty_strings / len(df)
                    if empty_ratio > config.missing_value_threshold:
                        issues.append(ValidationIssue(
                            category=ValidationCategory.COMPLETENESS,
                            severity=ValidationSeverity.WARNING,
                            rule_name="empty_strings",
                            message=f"Column '{col}' has {empty_strings} empty strings ({empty_ratio:.1%})",
                            affected_columns=[col],
                            suggestion=f"Handle empty strings in column '{col}'"
                        ))
        
        return issues
    
    def _validate_array_completeness(self, data: np.ndarray, config: ValidationConfig) -> List[ValidationIssue]:
        """Validate array completeness"""
        issues = []
        
        if data.size == 0:
            issues.append(ValidationIssue(
                category=ValidationCategory.COMPLETENESS,
                severity=ValidationSeverity.ERROR,
                rule_name="empty_array",
                message="Array is completely empty",
                suggestion="Provide non-empty data"
            ))
        
        return issues
    
    def _validate_list_completeness(self, data: list, config: ValidationConfig) -> List[ValidationIssue]:
        """Validate list completeness"""
        issues = []
        
        if len(data) == 0:
            issues.append(ValidationIssue(
                category=ValidationCategory.COMPLETENESS,
                severity=ValidationSeverity.WARNING,
                rule_name="empty_list",
                message="List is empty"
            ))
        else:
            # Check for None values
            none_count = sum(1 for item in data if item is None)
            if none_count > 0:
                none_ratio = none_count / len(data)
                if none_ratio > config.missing_value_threshold:
                    issues.append(ValidationIssue(
                        category=ValidationCategory.COMPLETENESS,
                        severity=ValidationSeverity.WARNING,
                        rule_name="excessive_none_values",
                        message=f"List has {none_count} None values ({none_ratio:.1%})",
                        suggestion="Handle None values in list"
                    ))
        
        return issues
    
    def _validate_dict_completeness(self, data: dict, config: ValidationConfig) -> List[ValidationIssue]:
        """Validate dictionary completeness"""
        issues = []
        
        if len(data) == 0:
            issues.append(ValidationIssue(
                category=ValidationCategory.COMPLETENESS,
                severity=ValidationSeverity.WARNING,
                rule_name="empty_dict",
                message="Dictionary is empty"
            ))
        else:
            # Check for None values
            none_keys = [key for key, value in data.items() if value is None]
            if none_keys:
                issues.append(ValidationIssue(
                    category=ValidationCategory.COMPLETENESS,
                    severity=ValidationSeverity.WARNING,
                    rule_name="none_values_in_dict",
                    message=f"Dictionary has None values for keys: {none_keys[:5]}",
                    suggestion="Handle None values in dictionary"
                ))
        
        return issues
    
    def get_validator_name(self) -> str:
        return "completeness_validator"


class ValidationEngine:
    """
    Comprehensive validation engine adapted from Saraphis patterns.
    Orchestrates multiple validators and provides unified validation results.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.validators: Dict[str, ValidatorInterface] = {}
        self.validation_history: List[Dict[str, Any]] = []
        self.cache = {} if self.config.enable_caching else None
        self._lock = threading.Lock()
        
        # Register default validators
        self.register_validator(SchemaValidator())
        self.register_validator(StatisticalValidator())
        self.register_validator(SecurityValidator())
        self.register_validator(CompletenessValidator())
        
        logger.info(f"Validation engine initialized with {len(self.validators)} validators")
    
    def register_validator(self, validator: ValidatorInterface):
        """Register a validator"""
        name = validator.get_validator_name()
        self.validators[name] = validator
        logger.info(f"Registered validator: {name}")
    
    def validate(self, data: Any, validators: Optional[List[str]] = None) -> ValidationResult:
        """
        Perform comprehensive validation on data.
        Adapted from Saraphis validation orchestration patterns.
        """
        start_time = time.time()
        
        # Check cache
        if self.cache is not None:
            data_hash = self._hash_data(data)
            cache_key = f"{data_hash}_{str(sorted(validators or []))}"
            
            with self._lock:
                if cache_key in self.cache:
                    cached_result = self.cache[cache_key]
                    logger.info("Validation result retrieved from cache")
                    return cached_result
        
        # Determine which validators to use
        if validators is None:
            validators_to_use = list(self.validators.keys())
        else:
            validators_to_use = [v for v in validators if v in self.validators]
        
        all_issues = []
        
        # Run validators
        for validator_name in validators_to_use:
            try:
                validator = self.validators[validator_name]
                issues = validator.validate(data, self.config)
                all_issues.extend(issues)
                logger.debug(f"Validator {validator_name} found {len(issues)} issues")
            except Exception as e:
                logger.error(f"Validator {validator_name} failed: {e}")
                all_issues.append(ValidationIssue(
                    category=ValidationCategory.SCHEMA,
                    severity=ValidationSeverity.ERROR,
                    rule_name="validator_error",
                    message=f"Validator {validator_name} failed: {str(e)}"
                ))
        
        # Calculate data quality score
        quality_score = self._calculate_quality_score(all_issues)
        
        # Create result
        validation_time = time.time() - start_time
        result = ValidationResult(
            is_valid=True,  # Will be updated in __post_init__
            issues=all_issues,
            validation_time=validation_time,
            data_quality_score=quality_score,
            metadata={
                'validators_used': validators_to_use,
                'total_issues': len(all_issues),
                'data_type': type(data).__name__,
                'data_size': self._get_data_size(data)
            }
        )
        
        # Cache result
        if self.cache is not None and len(self.cache) < self.config.cache_size:
            with self._lock:
                self.cache[cache_key] = result
        
        # Record validation
        self.validation_history.append({
            'timestamp': datetime.utcnow(),
            'validators_used': validators_to_use,
            'issues_found': len(all_issues),
            'quality_score': quality_score,
            'validation_time': validation_time
        })
        
        logger.info(f"Validation completed: {len(all_issues)} issues found, quality score: {quality_score:.3f}")
        return result
    
    def _calculate_quality_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate overall data quality score"""
        if not issues:
            return 1.0
        
        # Weight issues by severity
        severity_weights = {
            ValidationSeverity.INFO: 0.01,
            ValidationSeverity.WARNING: 0.05,
            ValidationSeverity.ERROR: 0.15,
            ValidationSeverity.CRITICAL: 0.30
        }
        
        total_penalty = 0.0
        for issue in issues:
            total_penalty += severity_weights.get(issue.severity, 0.05)
        
        # Calculate score (max penalty of 1.0)
        quality_score = max(0.0, 1.0 - min(1.0, total_penalty))
        return quality_score
    
    def _hash_data(self, data: Any) -> str:
        """Create hash for data caching"""
        try:
            if isinstance(data, pd.DataFrame):
                return hashlib.md5(str(data.shape).encode() + str(data.dtypes).encode()).hexdigest()
            elif isinstance(data, np.ndarray):
                return hashlib.md5(str(data.shape).encode() + str(data.dtype).encode()).hexdigest()
            else:
                return hashlib.md5(str(type(data)).encode() + str(len(str(data))).encode()).hexdigest()
        except:
            return "no_hash"
    
    def _get_data_size(self, data: Any) -> str:
        """Get human-readable data size"""
        try:
            if isinstance(data, pd.DataFrame):
                return f"{data.shape[0]} rows x {data.shape[1]} columns"
            elif isinstance(data, np.ndarray):
                return f"Array shape: {data.shape}"
            elif isinstance(data, (list, tuple)):
                return f"Length: {len(data)}"
            elif isinstance(data, dict):
                return f"Keys: {len(data)}"
            else:
                return f"Type: {type(data).__name__}"
        except:
            return "Unknown"
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation history"""
        if not self.validation_history:
            return {'message': 'No validations performed yet'}
        
        return {
            'total_validations': len(self.validation_history),
            'average_quality_score': np.mean([v['quality_score'] for v in self.validation_history]),
            'average_validation_time': np.mean([v['validation_time'] for v in self.validation_history]),
            'registered_validators': list(self.validators.keys()),
            'cache_enabled': self.cache is not None,
            'cache_size': len(self.cache) if self.cache else 0
        }


# Utility functions for validation

def validate_data_types(data: pd.DataFrame, expected_types: Dict[str, str]) -> List[ValidationIssue]:
    """Validate DataFrame column types against expected types"""
    issues = []
    
    for column, expected_type in expected_types.items():
        if column not in data.columns:
            issues.append(ValidationIssue(
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.ERROR,
                rule_name="missing_column",
                message=f"Expected column '{column}' not found",
                affected_columns=[column]
            ))
        else:
            actual_type = str(data[column].dtype)
            if expected_type not in actual_type:
                issues.append(ValidationIssue(
                    category=ValidationCategory.TYPE,
                    severity=ValidationSeverity.WARNING,
                    rule_name="type_mismatch",
                    message=f"Column '{column}' has type '{actual_type}', expected '{expected_type}'",
                    affected_columns=[column]
                ))
    
    return issues


def validate_ranges(data: pd.DataFrame, range_constraints: Dict[str, Tuple[float, float]]) -> List[ValidationIssue]:
    """Validate numeric columns against range constraints"""
    issues = []
    
    for column, (min_val, max_val) in range_constraints.items():
        if column not in data.columns:
            continue
            
        col_data = data[column]
        if pd.api.types.is_numeric_dtype(col_data):
            out_of_range = col_data[(col_data < min_val) | (col_data > max_val)]
            if len(out_of_range) > 0:
                issues.append(ValidationIssue(
                    category=ValidationCategory.RANGE,
                    severity=ValidationSeverity.WARNING,
                    rule_name="value_out_of_range",
                    message=f"Column '{column}' has {len(out_of_range)} values outside range [{min_val}, {max_val}]",
                    affected_columns=[column],
                    affected_rows=out_of_range.index.tolist()
                ))
    
    return issues


# Export public API
__all__ = [
    'ValidationSeverity', 'ValidationCategory', 'ValidationIssue', 'ValidationResult',
    'ValidationConfig', 'ValidatorInterface', 'SchemaValidator', 'StatisticalValidator',
    'SecurityValidator', 'CompletenessValidator', 'ValidationEngine',
    'validate_data_types', 'validate_ranges'
]