"""
Comprehensive Data Preprocessing for Financial Fraud Detection
Advanced feature engineering, data transformation, and preprocessing pipeline
with comprehensive validation and error handling.
Enhanced with production-ready error recovery and validation mechanisms.
"""

import logging
import pandas as pd
import numpy as np
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Set, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hashlib
import time
import traceback
import psutil
import signal
from contextlib import contextmanager
from functools import wraps
import re

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# Custom Exception Classes
# ============================================================================

class PreprocessingException(Exception):
    """Base exception for preprocessing errors"""
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()

class PreprocessingConfigError(PreprocessingException):
    """Raised when preprocessing configuration is invalid"""
    pass

class PreprocessingInputError(PreprocessingException):
    """Raised when input data is invalid or incompatible"""
    pass

class PreprocessingTimeoutError(PreprocessingException):
    """Raised when preprocessing operation times out"""
    pass

class PreprocessingMemoryError(PreprocessingException):
    """Raised when preprocessing exceeds memory limits"""
    pass

class PreprocessingSecurityError(PreprocessingException):
    """Raised when preprocessing violates security constraints"""
    pass

class PreprocessingIntegrationError(PreprocessingException):
    """Raised when preprocessing fails integration checks"""
    pass

class FeatureEngineeringError(PreprocessingException):
    """Raised when feature engineering fails"""
    pass

class EncodingError(PreprocessingException):
    """Raised when categorical encoding fails"""
    pass

class ScalingError(PreprocessingException):
    """Raised when feature scaling fails"""
    pass

class ImputationError(PreprocessingException):
    """Raised when data imputation fails"""
    pass

class PartialPreprocessingError(PreprocessingException):
    """Raised when preprocessing partially fails but can continue"""
    def __init__(self, message: str, successful_steps: List[str], failed_steps: List[str], 
                 partial_data: pd.DataFrame = None, **kwargs):
        super().__init__(message, **kwargs)
        self.successful_steps = successful_steps
        self.failed_steps = failed_steps
        self.partial_data = partial_data

# ============================================================================
# Validation and Security Components
# ============================================================================

@dataclass
class SecurityConfig:
    """Security configuration for preprocessing"""
    max_memory_mb: float = 2048.0
    max_execution_time_seconds: float = 300.0
    allowed_data_types: List[str] = field(default_factory=lambda: ['int64', 'float64', 'object', 'datetime64', 'bool'])
    sanitize_column_names: bool = True
    check_data_leakage: bool = True
    protect_sensitive_features: bool = True
    sensitive_patterns: List[str] = field(default_factory=lambda: ['ssn', 'pin', 'password', 'cvv', 'secret'])

@dataclass
class ValidationConfig:
    """Configuration for preprocessing validation"""
    validate_inputs: bool = True
    validate_outputs: bool = True
    check_data_consistency: bool = True
    check_feature_ranges: bool = True
    check_missing_threshold: float = 0.5  # Max allowed missing data ratio
    check_cardinality_threshold: int = 1000  # Max unique values for categorical
    check_correlation_threshold: float = 0.95  # Max correlation between features
    enable_profiling: bool = True

class InputValidator:
    """Validates preprocessing inputs"""
    
    def __init__(self, validation_config: ValidationConfig):
        self.config = validation_config
    
    def validate_dataframe(self, data: pd.DataFrame, context: str = "input") -> Tuple[bool, List[str]]:
        """Validate DataFrame structure and content"""
        errors = []
        
        # Check if DataFrame is empty
        if data.empty:
            errors.append(f"{context}: DataFrame is empty")
            return False, errors
        
        # Check shape
        if data.shape[0] == 0 or data.shape[1] == 0:
            errors.append(f"{context}: Invalid DataFrame shape {data.shape}")
            return False, errors
        
        # Check for excessive missing data
        if self.config.check_missing_threshold:
            missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
            if missing_ratio > self.config.check_missing_threshold:
                errors.append(f"{context}: Excessive missing data ({missing_ratio:.2%})")
        
        # Check data types
        for col, dtype in data.dtypes.items():
            if dtype == 'object':
                # Check for mixed types in object columns
                try:
                    unique_types = data[col].dropna().apply(type).unique()
                    if len(unique_types) > 1:
                        errors.append(f"{context}: Mixed types in column '{col}'")
                except Exception as e:
                    errors.append(f"{context}: Error checking column '{col}': {str(e)}")
        
        # Check for duplicate columns
        duplicate_cols = data.columns[data.columns.duplicated()].tolist()
        if duplicate_cols:
            errors.append(f"{context}: Duplicate columns found: {duplicate_cols}")
        
        return len(errors) == 0, errors
    
    def validate_preprocessing_config(self, config: 'PreprocessingConfig') -> Tuple[bool, List[str]]:
        """Validate preprocessing configuration"""
        errors = []
        
        # Validate missing strategy
        valid_missing_strategies = ['drop', 'fill', 'interpolate']
        if config.missing_strategy not in valid_missing_strategies:
            errors.append(f"Invalid missing_strategy: {config.missing_strategy}")
        
        # Validate outlier method
        valid_outlier_methods = ['iqr', 'zscore', 'isolation_forest']
        if config.outlier_method not in valid_outlier_methods:
            errors.append(f"Invalid outlier_method: {config.outlier_method}")
        
        # Validate categorical encoding
        valid_encodings = ['onehot', 'label', 'target', 'frequency']
        if config.categorical_encoding not in valid_encodings:
            errors.append(f"Invalid categorical_encoding: {config.categorical_encoding}")
        
        # Validate selection method
        valid_selection_methods = ['mutual_info', 'f_classif', 'rfe']
        if config.selection_method not in valid_selection_methods:
            errors.append(f"Invalid selection_method: {config.selection_method}")
        
        # Validate numerical parameters
        if config.max_categories <= 0:
            errors.append(f"max_categories must be positive: {config.max_categories}")
        
        if config.polynomial_degree < 1:
            errors.append(f"polynomial_degree must be >= 1: {config.polynomial_degree}")
        
        # Validate lag periods
        if config.lag_periods and any(p <= 0 for p in config.lag_periods):
            errors.append("All lag_periods must be positive")
        
        return len(errors) == 0, errors

class PerformanceMonitor:
    """Monitors preprocessing performance"""
    
    def __init__(self, max_memory_mb: float = 2048.0, max_time_seconds: float = 300.0):
        self.max_memory_mb = max_memory_mb
        self.max_time_seconds = max_time_seconds
        self.start_time = None
        self.start_memory = None
        self._stop_flag = threading.Event()
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager for monitoring operations"""
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
        
        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(operation_name,)
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            yield
        finally:
            self._stop_flag.set()
            elapsed_time = time.time() - self.start_time
            memory_used = self._get_memory_usage() - self.start_memory
            
            logger.info(f"{operation_name} completed in {elapsed_time:.2f}s, "
                       f"used {memory_used:.2f}MB memory")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _monitor_resources(self, operation_name: str):
        """Monitor resources in background thread"""
        while not self._stop_flag.is_set():
            elapsed_time = time.time() - self.start_time
            current_memory = self._get_memory_usage()
            memory_used = current_memory - self.start_memory
            
            # Check time limit
            if elapsed_time > self.max_time_seconds:
                logger.error(f"{operation_name} exceeded time limit ({self.max_time_seconds}s)")
                raise PreprocessingTimeoutError(
                    f"{operation_name} exceeded time limit",
                    error_code="TIMEOUT",
                    details={'elapsed_time': elapsed_time, 'limit': self.max_time_seconds}
                )
            
            # Check memory limit
            if memory_used > self.max_memory_mb:
                logger.error(f"{operation_name} exceeded memory limit ({self.max_memory_mb}MB)")
                raise PreprocessingMemoryError(
                    f"{operation_name} exceeded memory limit",
                    error_code="MEMORY_EXCEEDED",
                    details={'memory_used': memory_used, 'limit': self.max_memory_mb}
                )
            
            time.sleep(1)  # Check every second

class SecurityValidator:
    """Validates security constraints"""
    
    def __init__(self, security_config: SecurityConfig):
        self.config = security_config
    
    def validate_column_names(self, columns: List[str]) -> Tuple[bool, List[str]]:
        """Validate column names for security issues"""
        errors = []
        
        for col in columns:
            # Check for SQL injection patterns
            if any(pattern in col.lower() for pattern in ['drop', 'delete', 'insert', 'update', ';', '--']):
                errors.append(f"Potentially malicious column name: '{col}'")
            
            # Check for sensitive data patterns
            if self.config.protect_sensitive_features:
                for pattern in self.config.sensitive_patterns:
                    if pattern in col.lower():
                        errors.append(f"Sensitive data pattern found in column: '{col}'")
        
        return len(errors) == 0, errors
    
    def sanitize_column_names(self, columns: List[str]) -> List[str]:
        """Sanitize column names for security"""
        if not self.config.sanitize_column_names:
            return columns
        
        sanitized = []
        for col in columns:
            # Remove special characters
            sanitized_col = re.sub(r'[^a-zA-Z0-9_]', '_', col)
            # Ensure doesn't start with number
            if sanitized_col[0].isdigit():
                sanitized_col = f"col_{sanitized_col}"
            sanitized.append(sanitized_col)
        
        return sanitized

class QualityValidator:
    """Validates preprocessing quality"""
    
    def __init__(self, validation_config: ValidationConfig):
        self.config = validation_config
    
    def validate_preprocessing_result(self, 
                                    original_data: pd.DataFrame,
                                    processed_data: pd.DataFrame,
                                    context: str = "result") -> Tuple[bool, List[str]]:
        """Validate preprocessing results"""
        errors = []
        
        # Check data loss
        if processed_data.shape[0] < original_data.shape[0] * 0.1:
            errors.append(f"{context}: Excessive data loss (>90% rows removed)")
        
        # Check feature explosion
        if processed_data.shape[1] > original_data.shape[1] * 100:
            errors.append(f"{context}: Excessive feature expansion (>100x columns)")
        
        # Check for constant features
        constant_features = []
        for col in processed_data.select_dtypes(include=[np.number]).columns:
            if processed_data[col].nunique() == 1:
                constant_features.append(col)
        
        if constant_features:
            errors.append(f"{context}: Constant features detected: {constant_features[:5]}")
        
        # Check for high correlation
        if self.config.check_correlation_threshold:
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = processed_data[numeric_cols].corr().abs()
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if corr_matrix.iloc[i, j] > self.config.check_correlation_threshold:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                
                if high_corr_pairs:
                    errors.append(f"{context}: High correlation pairs found: {high_corr_pairs[:3]}")
        
        return len(errors) == 0, errors

# ============================================================================
# Enhanced Preprocessing Components with Error Handling
# ============================================================================

class PreprocessingStep(Enum):
    """Preprocessing step types"""
    CLEANING = "cleaning"
    TRANSFORMATION = "transformation"
    FEATURE_ENGINEERING = "feature_engineering"
    SCALING = "scaling"
    ENCODING = "encoding"
    FEATURE_SELECTION = "feature_selection"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"

class FeatureType(Enum):
    """Feature type classifications"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    TEXTUAL = "textual"
    BINARY = "binary"
    ORDINAL = "ordinal"

class ScalingMethod(Enum):
    """Data scaling methods"""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    NORMALIZE = "normalize"

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing operations"""
    # Cleaning settings
    handle_missing: bool = True
    missing_strategy: str = "drop"  # drop, fill, interpolate
    remove_duplicates: bool = True
    outlier_detection: bool = True
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    
    # Feature engineering settings
    create_time_features: bool = True
    create_aggregation_features: bool = True
    create_ratio_features: bool = True
    create_lag_features: bool = True
    lag_periods: List[int] = field(default_factory=lambda: [1, 7, 30])
    
    # Scaling settings
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    scale_features: List[str] = field(default_factory=list)
    
    # Encoding settings
    categorical_encoding: str = "onehot"  # onehot, label, target, frequency
    max_categories: int = 20
    
    # Feature selection settings
    feature_selection: bool = True
    selection_method: str = "mutual_info"  # mutual_info, f_classif, rfe
    max_features: Optional[int] = None
    
    # Advanced settings
    dimensionality_reduction: bool = False
    pca_components: Optional[int] = None
    create_interaction_features: bool = False
    polynomial_features: bool = False
    polynomial_degree: int = 2
    
    # Validation and security settings
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    security_config: SecurityConfig = field(default_factory=SecurityConfig)
    enable_error_recovery: bool = True
    max_retries: int = 3
    fallback_strategy: str = "partial"  # partial, skip, fail

@dataclass
class PreprocessingResult:
    """Results from preprocessing operations"""
    processed_data: pd.DataFrame
    feature_names: List[str]
    feature_types: Dict[str, FeatureType]
    preprocessing_steps: List[str]
    preprocessing_time_seconds: float
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    scaling_params: Dict[str, Any] = field(default_factory=dict)
    encoding_params: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_report: Dict[str, Any] = field(default_factory=dict)
    errors_encountered: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class PreprocessingStepBase(ABC):
    """Abstract base class for preprocessing steps with error handling"""
    
    def __init__(self, step_name: str, step_type: PreprocessingStep):
        self.step_name = step_name
        self.step_type = step_type
        self.fitted = False
        self.fit_params: Dict[str, Any] = {}
        self.error_count = 0
        self.warnings: List[str] = []
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'PreprocessingStepBase':
        """Fit the preprocessing step to the data"""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using fitted parameters"""
        pass
    
    def fit_transform(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(data, target).transform(data)
    
    def _handle_error(self, error: Exception, context: str, 
                     fallback_data: pd.DataFrame = None) -> pd.DataFrame:
        """Handle errors with recovery strategies"""
        self.error_count += 1
        logger.error(f"Error in {self.step_name}.{context}: {str(error)}")
        
        if fallback_data is not None:
            self.warnings.append(f"Error in {context}, using fallback data")
            return fallback_data
        
        raise PreprocessingException(
            f"Failed in {self.step_name}.{context}: {str(error)}",
            error_code=f"{self.step_name.upper()}_ERROR",
            details={'context': context, 'original_error': str(error)}
        )

class DataCleaner(PreprocessingStepBase):
    """Comprehensive data cleaning step with validation"""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__("data_cleaner", PreprocessingStep.CLEANING)
        self.config = config
        self.input_validator = InputValidator(config.validation_config)
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'DataCleaner':
        """Fit cleaning parameters with validation"""
        try:
            # Validate input
            is_valid, errors = self.input_validator.validate_dataframe(data, "fit_input")
            if not is_valid:
                raise PreprocessingInputError(
                    f"Invalid input data: {'; '.join(errors)}",
                    error_code="INVALID_FIT_INPUT"
                )
            
            self.fit_params = {
                'missing_columns': data.isnull().sum().to_dict(),
                'duplicate_count': data.duplicated().sum(),
                'outlier_bounds': {},
                'original_dtypes': data.dtypes.to_dict()
            }
            
            # Calculate outlier bounds for numerical columns
            if self.config.outlier_detection:
                numerical_cols = data.select_dtypes(include=[np.number]).columns
                for col in numerical_cols:
                    try:
                        if self.config.outlier_method == "iqr":
                            Q1 = data[col].quantile(0.25)
                            Q3 = data[col].quantile(0.75)
                            IQR = Q3 - Q1
                            if IQR == 0:
                                self.warnings.append(f"Zero IQR for column '{col}', using std method")
                                mean = data[col].mean()
                                std = data[col].std()
                                lower_bound = mean - 3*std
                                upper_bound = mean + 3*std
                            else:
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                            self.fit_params['outlier_bounds'][col] = (lower_bound, upper_bound)
                        elif self.config.outlier_method == "zscore":
                            mean = data[col].mean()
                            std = data[col].std()
                            if std == 0:
                                self.warnings.append(f"Zero std for column '{col}', skipping outlier detection")
                                continue
                            self.fit_params['outlier_bounds'][col] = (mean - 3*std, mean + 3*std)
                    except Exception as e:
                        self.warnings.append(f"Failed to calculate outlier bounds for '{col}': {str(e)}")
                        continue
            
            self.fitted = True
            return self
            
        except PreprocessingException:
            raise
        except Exception as e:
            return self._handle_error(e, "fit")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning transformations with error recovery"""
        if not self.fitted:
            raise ValueError("DataCleaner must be fitted before transform")
        
        try:
            # Validate input
            is_valid, errors = self.input_validator.validate_dataframe(data, "transform_input")
            if not is_valid and not self.config.enable_error_recovery:
                raise PreprocessingInputError(
                    f"Invalid input data: {'; '.join(errors)}",
                    error_code="INVALID_TRANSFORM_INPUT"
                )
            
            cleaned_data = data.copy()
            steps_applied = []
            
            # Handle missing values with error recovery
            if self.config.handle_missing:
                try:
                    if self.config.missing_strategy == "drop":
                        original_len = len(cleaned_data)
                        cleaned_data = cleaned_data.dropna()
                        if len(cleaned_data) == 0:
                            raise PreprocessingException("All rows removed by dropna")
                        steps_applied.append(f"dropped {original_len - len(cleaned_data)} rows with missing values")
                        
                    elif self.config.missing_strategy == "fill":
                        # Fill numerical with median, categorical with mode
                        for col in cleaned_data.columns:
                            if cleaned_data[col].isnull().any():
                                try:
                                    if cleaned_data[col].dtype in ['int64', 'float64']:
                                        fill_value = cleaned_data[col].median()
                                        if pd.isna(fill_value):
                                            fill_value = 0
                                        cleaned_data[col].fillna(fill_value, inplace=True)
                                    else:
                                        mode_values = cleaned_data[col].mode()
                                        fill_value = mode_values.iloc[0] if not mode_values.empty else 'unknown'
                                        cleaned_data[col].fillna(fill_value, inplace=True)
                                except Exception as e:
                                    self.warnings.append(f"Failed to fill missing values in '{col}': {str(e)}")
                                    
                    elif self.config.missing_strategy == "interpolate":
                        try:
                            cleaned_data = cleaned_data.interpolate(limit_direction='both')
                        except Exception as e:
                            self.warnings.append(f"Interpolation failed, using forward fill: {str(e)}")
                            cleaned_data = cleaned_data.fillna(method='ffill').fillna(method='bfill')
                            
                except Exception as e:
                    if self.config.enable_error_recovery:
                        self.warnings.append(f"Missing value handling failed: {str(e)}")
                    else:
                        raise
            
            # Remove duplicates with validation
            if self.config.remove_duplicates:
                try:
                    original_len = len(cleaned_data)
                    cleaned_data = cleaned_data.drop_duplicates()
                    steps_applied.append(f"removed {original_len - len(cleaned_data)} duplicate rows")
                except Exception as e:
                    self.warnings.append(f"Duplicate removal failed: {str(e)}")
            
            # Remove outliers with validation
            if self.config.outlier_detection and self.fit_params.get('outlier_bounds'):
                try:
                    original_len = len(cleaned_data)
                    for col, (lower_bound, upper_bound) in self.fit_params['outlier_bounds'].items():
                        if col in cleaned_data.columns:
                            mask = (cleaned_data[col] >= lower_bound) & (cleaned_data[col] <= upper_bound)
                            outlier_count = (~mask).sum()
                            if outlier_count > 0:
                                cleaned_data = cleaned_data[mask]
                                steps_applied.append(f"removed {outlier_count} outliers from '{col}'")
                    
                    # Check if too many rows were removed
                    if len(cleaned_data) < original_len * 0.5:
                        self.warnings.append("More than 50% of data removed as outliers")
                        
                except Exception as e:
                    self.warnings.append(f"Outlier removal failed: {str(e)}")
            
            # Final validation
            if cleaned_data.empty:
                raise PreprocessingException("Data cleaning resulted in empty DataFrame")
            
            return cleaned_data
            
        except PreprocessingException:
            raise
        except Exception as e:
            return self._handle_error(e, "transform", data)

class FeatureEngineer(PreprocessingStepBase):
    """Advanced feature engineering with comprehensive error handling"""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__("feature_engineer", PreprocessingStep.FEATURE_ENGINEERING)
        self.config = config
        self.performance_monitor = PerformanceMonitor(
            config.security_config.max_memory_mb,
            config.security_config.max_execution_time_seconds
        )
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'FeatureEngineer':
        """Fit feature engineering parameters with monitoring"""
        try:
            with self.performance_monitor.monitor_operation("feature_engineering_fit"):
                self.fit_params = {
                    'user_stats': {},
                    'merchant_stats': {},
                    'time_patterns': {},
                    'amount_stats': {},
                    'feature_creation_errors': []
                }
                
                # Calculate user-based statistics with error handling
                if 'user_id' in data.columns and 'amount' in data.columns:
                    try:
                        user_stats = data.groupby('user_id')['amount'].agg([
                            'count', 'mean', 'std', 'min', 'max', 'sum'
                        ]).fillna(0).to_dict()
                        self.fit_params['user_stats'] = user_stats
                    except Exception as e:
                        self.fit_params['feature_creation_errors'].append(f"user_stats: {str(e)}")
                
                # Calculate merchant-based statistics
                if 'merchant_id' in data.columns and 'amount' in data.columns:
                    try:
                        merchant_stats = data.groupby('merchant_id')['amount'].agg([
                            'count', 'mean', 'std', 'min', 'max'
                        ]).fillna(0).to_dict()
                        self.fit_params['merchant_stats'] = merchant_stats
                    except Exception as e:
                        self.fit_params['feature_creation_errors'].append(f"merchant_stats: {str(e)}")
                
                # Analyze time patterns with validation
                if 'timestamp' in data.columns:
                    try:
                        timestamps = pd.to_datetime(data['timestamp'], errors='coerce')
                        valid_timestamps = timestamps.dropna()
                        if len(valid_timestamps) > 0:
                            self.fit_params['time_patterns'] = {
                                'business_hours_ratio': ((valid_timestamps.dt.hour >= 9) & 
                                                       (valid_timestamps.dt.hour <= 17)).mean(),
                                'weekend_ratio': valid_timestamps.dt.dayofweek.isin([5, 6]).mean(),
                                'night_ratio': ((valid_timestamps.dt.hour >= 22) | 
                                              (valid_timestamps.dt.hour <= 6)).mean()
                            }
                    except Exception as e:
                        self.fit_params['feature_creation_errors'].append(f"time_patterns: {str(e)}")
                
                # Amount distribution statistics with validation
                if 'amount' in data.columns:
                    try:
                        amount_data = pd.to_numeric(data['amount'], errors='coerce').dropna()
                        if len(amount_data) > 0:
                            self.fit_params['amount_stats'] = {
                                'mean': amount_data.mean(),
                                'std': amount_data.std(),
                                'q25': amount_data.quantile(0.25),
                                'q75': amount_data.quantile(0.75),
                                'round_number_ratio': (amount_data % 1 == 0).mean()
                            }
                    except Exception as e:
                        self.fit_params['feature_creation_errors'].append(f"amount_stats: {str(e)}")
                
                self.fitted = True
                return self
                
        except PreprocessingTimeoutError:
            raise
        except PreprocessingMemoryError:
            raise
        except Exception as e:
            return self._handle_error(e, "fit")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering with comprehensive error handling"""
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        try:
            with self.performance_monitor.monitor_operation("feature_engineering_transform"):
                engineered_data = data.copy()
                created_features = []
                failed_features = []
                
                # Time-based features with error handling
                if self.config.create_time_features and 'timestamp' in data.columns:
                    try:
                        engineered_data = self._create_time_features(engineered_data)
                        created_features.append("time_features")
                    except Exception as e:
                        failed_features.append(f"time_features: {str(e)}")
                        if not self.config.enable_error_recovery:
                            raise
                
                # User-based aggregation features
                if self.config.create_aggregation_features:
                    try:
                        engineered_data = self._create_user_features(engineered_data)
                        created_features.append("user_features")
                    except Exception as e:
                        failed_features.append(f"user_features: {str(e)}")
                    
                    try:
                        engineered_data = self._create_merchant_features(engineered_data)
                        created_features.append("merchant_features")
                    except Exception as e:
                        failed_features.append(f"merchant_features: {str(e)}")
                
                # Amount-based features
                if 'amount' in data.columns:
                    try:
                        engineered_data = self._create_amount_features(engineered_data)
                        created_features.append("amount_features")
                    except Exception as e:
                        failed_features.append(f"amount_features: {str(e)}")
                
                # Ratio and interaction features
                if self.config.create_ratio_features:
                    try:
                        engineered_data = self._create_ratio_features(engineered_data)
                        created_features.append("ratio_features")
                    except Exception as e:
                        failed_features.append(f"ratio_features: {str(e)}")
                
                # Lag features
                if self.config.create_lag_features:
                    try:
                        engineered_data = self._create_lag_features(engineered_data)
                        created_features.append("lag_features")
                    except Exception as e:
                        failed_features.append(f"lag_features: {str(e)}")
                
                # Transaction velocity features
                if all(col in data.columns for col in ['user_id', 'timestamp']):
                    try:
                        engineered_data = self._create_velocity_features(engineered_data)
                        created_features.append("velocity_features")
                    except Exception as e:
                        failed_features.append(f"velocity_features: {str(e)}")
                
                # Handle partial failures
                if failed_features:
                    if self.config.fallback_strategy == "partial":
                        self.warnings.extend(failed_features)
                        logger.warning(f"Feature engineering partially failed: {failed_features}")
                    elif self.config.fallback_strategy == "fail":
                        raise PartialPreprocessingError(
                            "Feature engineering partially failed",
                            successful_steps=created_features,
                            failed_steps=failed_features,
                            partial_data=engineered_data
                        )
                
                # Validate result
                if engineered_data.shape[1] == data.shape[1]:
                    self.warnings.append("No new features were created")
                
                return engineered_data
                
        except (PreprocessingTimeoutError, PreprocessingMemoryError):
            raise
        except PartialPreprocessingError:
            raise
        except Exception as e:
            return self._handle_error(e, "transform", data)
    
    def _create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features with error handling"""
        try:
            timestamp = pd.to_datetime(data['timestamp'], errors='coerce')
            
            # Check for valid timestamps
            valid_mask = timestamp.notna()
            if valid_mask.sum() == 0:
                raise ValueError("No valid timestamps found")
            
            # Basic time features
            data.loc[valid_mask, 'hour'] = timestamp[valid_mask].dt.hour
            data.loc[valid_mask, 'day_of_week'] = timestamp[valid_mask].dt.dayofweek
            data.loc[valid_mask, 'day_of_month'] = timestamp[valid_mask].dt.day
            data.loc[valid_mask, 'month'] = timestamp[valid_mask].dt.month
            data.loc[valid_mask, 'quarter'] = timestamp[valid_mask].dt.quarter
            data.loc[valid_mask, 'year'] = timestamp[valid_mask].dt.year
            
            # Advanced time features with safe operations
            data.loc[valid_mask, 'is_weekend'] = timestamp[valid_mask].dt.dayofweek.isin([5, 6]).astype(int)
            data.loc[valid_mask, 'is_business_hours'] = (
                (timestamp[valid_mask].dt.hour >= 9) & 
                (timestamp[valid_mask].dt.hour <= 17)
            ).astype(int)
            data.loc[valid_mask, 'is_night'] = (
                (timestamp[valid_mask].dt.hour >= 22) | 
                (timestamp[valid_mask].dt.hour <= 6)
            ).astype(int)
            
            # Cyclical encoding with validation
            data.loc[valid_mask, 'hour_sin'] = np.sin(2 * np.pi * timestamp[valid_mask].dt.hour / 24)
            data.loc[valid_mask, 'hour_cos'] = np.cos(2 * np.pi * timestamp[valid_mask].dt.hour / 24)
            data.loc[valid_mask, 'day_of_week_sin'] = np.sin(2 * np.pi * timestamp[valid_mask].dt.dayofweek / 7)
            data.loc[valid_mask, 'day_of_week_cos'] = np.cos(2 * np.pi * timestamp[valid_mask].dt.dayofweek / 7)
            
            # Fill invalid timestamps with defaults
            for col in ['hour', 'day_of_week', 'day_of_month', 'month', 'quarter', 'year']:
                if col in data.columns:
                    data[col].fillna(0, inplace=True)
            
            return data
            
        except Exception as e:
            raise FeatureEngineeringError(
                f"Time feature creation failed: {str(e)}",
                error_code="TIME_FEATURES_ERROR"
            )
    
    def _create_user_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create user-based features with error handling"""
        if 'user_id' not in data.columns:
            return data
        
        try:
            user_stats = self.fit_params.get('user_stats', {})
            
            # Map user statistics with safe operations
            for stat_name, stat_dict in user_stats.items():
                col_name = f'user_{stat_name}'
                data[col_name] = data['user_id'].map(stat_dict).fillna(0)
                
                # Validate created feature
                if data[col_name].isna().all():
                    self.warnings.append(f"All values are NaN for {col_name}")
            
            # User behavior patterns with safe division
            if 'amount' in data.columns and 'user_mean' in data.columns:
                epsilon = 1e-8
                data['amount_vs_user_mean'] = data['amount'] / (data['user_mean'] + epsilon)
                data['amount_vs_user_mean'] = data['amount_vs_user_mean'].replace([np.inf, -np.inf], 0)
                
                if 'user_std' in data.columns:
                    data['amount_vs_user_std'] = (
                        np.abs(data['amount'] - data['user_mean']) / (data['user_std'] + epsilon)
                    )
                    data['amount_vs_user_std'] = data['amount_vs_user_std'].replace([np.inf, -np.inf], 0)
            
            return data
            
        except Exception as e:
            raise FeatureEngineeringError(
                f"User feature creation failed: {str(e)}",
                error_code="USER_FEATURES_ERROR"
            )
    
    def _create_merchant_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create merchant-based features with error handling"""
        if 'merchant_id' not in data.columns:
            return data
        
        try:
            merchant_stats = self.fit_params.get('merchant_stats', {})
            
            # Map merchant statistics
            for stat_name, stat_dict in merchant_stats.items():
                col_name = f'merchant_{stat_name}'
                data[col_name] = data['merchant_id'].map(stat_dict).fillna(0)
            
            # Merchant behavior patterns
            if 'amount' in data.columns and 'merchant_mean' in data.columns:
                epsilon = 1e-8
                data['amount_vs_merchant_mean'] = data['amount'] / (data['merchant_mean'] + epsilon)
                data['amount_vs_merchant_mean'] = data['amount_vs_merchant_mean'].replace([np.inf, -np.inf], 0)
            
            return data
            
        except Exception as e:
            raise FeatureEngineeringError(
                f"Merchant feature creation failed: {str(e)}",
                error_code="MERCHANT_FEATURES_ERROR"
            )
    
    def _create_amount_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create amount-based features with validation"""
        try:
            amount_stats = self.fit_params.get('amount_stats', {})
            
            # Validate amount column
            amount_numeric = pd.to_numeric(data['amount'], errors='coerce')
            valid_mask = amount_numeric.notna() & (amount_numeric >= 0)
            
            if valid_mask.sum() == 0:
                raise ValueError("No valid amount values found")
            
            # Basic amount features with safe operations
            data['amount_log'] = np.log1p(amount_numeric.clip(lower=0))
            data['amount_sqrt'] = np.sqrt(amount_numeric.clip(lower=0))
            data['amount_squared'] = amount_numeric.clip(upper=1e6) ** 2  # Clip to prevent overflow
            
            # Amount distribution features
            if amount_stats:
                mean = amount_stats.get('mean', 0)
                std = amount_stats.get('std', 1)
                data['amount_zscore'] = (amount_numeric - mean) / (std + 1e-8)
                data['amount_zscore'] = data['amount_zscore'].clip(-10, 10)  # Clip extreme values
            
            # Amount patterns
            data['is_round_amount'] = (amount_numeric % 1 == 0).astype(int)
            data['is_round_10'] = (amount_numeric % 10 == 0).astype(int)
            data['is_round_100'] = (amount_numeric % 100 == 0).astype(int)
            
            return data
            
        except Exception as e:
            raise FeatureEngineeringError(
                f"Amount feature creation failed: {str(e)}",
                error_code="AMOUNT_FEATURES_ERROR"
            )
    
    def _create_ratio_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create ratio features with overflow protection"""
        try:
            numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            numerical_cols = [col for col in numerical_cols if not col.startswith('is_') and not col.endswith('_id')]
            
            # Limit number of ratios to prevent explosion
            max_ratios = 10
            created_ratios = 0
            
            for i, col1 in enumerate(numerical_cols[:5]):
                for col2 in numerical_cols[i+1:6]:
                    if col1 != col2 and created_ratios < max_ratios:
                        try:
                            # Safe division with epsilon
                            epsilon = 1e-8
                            ratio_col = f'{col1}_to_{col2}_ratio'
                            data[ratio_col] = data[col1] / (data[col2] + epsilon)
                            
                            # Replace infinite values
                            data[ratio_col] = data[ratio_col].replace([np.inf, -np.inf], 0)
                            
                            # Clip extreme values
                            percentile_99 = data[ratio_col].quantile(0.99)
                            percentile_1 = data[ratio_col].quantile(0.01)
                            data[ratio_col] = data[ratio_col].clip(percentile_1, percentile_99)
                            
                            created_ratios += 1
                        except Exception as e:
                            self.warnings.append(f"Failed to create ratio {col1}/{col2}: {str(e)}")
            
            return data
            
        except Exception as e:
            raise FeatureEngineeringError(
                f"Ratio feature creation failed: {str(e)}",
                error_code="RATIO_FEATURES_ERROR"
            )
    
    def _create_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features with memory efficiency"""
        if not all(col in data.columns for col in ['user_id', 'timestamp']):
            return data
        
        try:
            # Sort by user and timestamp
            data_sorted = data.sort_values(['user_id', 'timestamp'])
            
            # Limit lag periods for memory efficiency
            lag_periods = self.config.lag_periods[:3]  # Max 3 lags
            
            # Create lag features for amount
            if 'amount' in data_sorted.columns:
                for lag in lag_periods:
                    try:
                        lag_col = f'amount_lag_{lag}'
                        data_sorted[lag_col] = data_sorted.groupby('user_id')['amount'].shift(lag)
                        
                        # Calculate changes only if lag was successful
                        if not data_sorted[lag_col].isna().all():
                            data_sorted[f'amount_change_{lag}'] = data_sorted['amount'] - data_sorted[lag_col]
                            
                            # Safe percentage change
                            epsilon = 1e-8
                            data_sorted[f'amount_pct_change_{lag}'] = (
                                data_sorted['amount'] / (data_sorted[lag_col] + epsilon) - 1
                            ).clip(-10, 10)
                    except Exception as e:
                        self.warnings.append(f"Failed to create lag {lag}: {str(e)}")
            
            return data_sorted
            
        except Exception as e:
            raise FeatureEngineeringError(
                f"Lag feature creation failed: {str(e)}",
                error_code="LAG_FEATURES_ERROR"
            )
    
    def _create_velocity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create transaction velocity features with efficiency"""
        try:
            # Sort by user and timestamp
            data_sorted = data.sort_values(['user_id', 'timestamp'])
            data_sorted['timestamp'] = pd.to_datetime(data_sorted['timestamp'], errors='coerce')
            
            # Time since last transaction
            data_sorted['time_since_last'] = data_sorted.groupby('user_id')['timestamp'].diff()
            data_sorted['time_since_last_minutes'] = data_sorted['time_since_last'].dt.total_seconds() / 60
            data_sorted['time_since_last_hours'] = data_sorted['time_since_last'].dt.total_seconds() / 3600
            
            # Cap extreme values
            data_sorted['time_since_last_minutes'] = data_sorted['time_since_last_minutes'].clip(0, 10080)  # Max 1 week
            data_sorted['time_since_last_hours'] = data_sorted['time_since_last_hours'].clip(0, 168)  # Max 1 week
            
            # Rolling window features with memory limits
            if 'amount' in data_sorted.columns:
                for window in [3, 5]:  # Limit windows
                    try:
                        data_sorted[f'amount_mean_{window}'] = (
                            data_sorted.groupby('user_id')['amount']
                            .rolling(window=window, min_periods=1)
                            .mean()
                            .reset_index(0, drop=True)
                        )
                    except Exception as e:
                        self.warnings.append(f"Failed to create rolling mean {window}: {str(e)}")
            
            return data_sorted
            
        except Exception as e:
            raise FeatureEngineeringError(
                f"Velocity feature creation failed: {str(e)}",
                error_code="VELOCITY_FEATURES_ERROR"
            )

class FeatureScaler(PreprocessingStepBase):
    """Feature scaling with validation and error handling"""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__("feature_scaler", PreprocessingStep.SCALING)
        self.config = config
        self.scalers = {}
        self.security_validator = SecurityValidator(config.security_config)
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'FeatureScaler':
        """Fit scaling parameters with validation"""
        try:
            # Determine columns to scale
            if self.config.scale_features:
                columns_to_scale = self.config.scale_features
            else:
                # Auto-detect numerical columns
                columns_to_scale = data.select_dtypes(include=[np.number]).columns.tolist()
                # Exclude binary features and IDs
                columns_to_scale = [
                    col for col in columns_to_scale 
                    if not col.endswith('_id') and data[col].nunique() > 2
                ]
            
            # Validate column names for security
            is_valid, errors = self.security_validator.validate_column_names(columns_to_scale)
            if not is_valid and not self.config.enable_error_recovery:
                raise PreprocessingSecurityError(
                    f"Security validation failed: {'; '.join(errors)}",
                    error_code="INVALID_COLUMN_NAMES"
                )
            
            # Initialize scaler based on method
            scaler_class = {
                ScalingMethod.STANDARD: StandardScaler,
                ScalingMethod.MINMAX: MinMaxScaler,
                ScalingMethod.ROBUST: RobustScaler
            }.get(self.config.scaling_method, StandardScaler)
            
            # Fit scalers for each column with error handling
            for col in columns_to_scale:
                if col in data.columns:
                    try:
                        col_data = data[[col]].dropna()
                        if len(col_data) > 0:
                            self.scalers[col] = scaler_class()
                            self.scalers[col].fit(col_data)
                    except Exception as e:
                        self.warnings.append(f"Failed to fit scaler for '{col}': {str(e)}")
            
            self.fitted = True
            return self
            
        except PreprocessingException:
            raise
        except Exception as e:
            return self._handle_error(e, "fit")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling with validation and error recovery"""
        if not self.fitted:
            raise ValueError("FeatureScaler must be fitted before transform")
        
        try:
            scaled_data = data.copy()
            scaled_columns = []
            failed_columns = []
            
            # Apply scaling to each column
            for col, scaler in self.scalers.items():
                if col in scaled_data.columns:
                    try:
                        # Handle missing values
                        non_null_mask = scaled_data[col].notna()
                        if non_null_mask.sum() > 0:
                            scaled_values = scaler.transform(scaled_data.loc[non_null_mask, [col]]).flatten()
                            scaled_data.loc[non_null_mask, col] = scaled_values
                            
                            # Validate scaled values
                            if np.isnan(scaled_values).any() or np.isinf(scaled_values).any():
                                raise ValueError(f"Scaling produced invalid values for '{col}'")
                            
                            scaled_columns.append(col)
                    except Exception as e:
                        failed_columns.append(f"{col}: {str(e)}")
                        if self.config.enable_error_recovery:
                            self.warnings.append(f"Scaling failed for '{col}', keeping original values")
                        else:
                            raise
            
            # Report results
            if failed_columns:
                if self.config.fallback_strategy == "fail":
                    raise ScalingError(
                        f"Scaling failed for columns: {failed_columns}",
                        error_code="SCALING_FAILED"
                    )
            
            if not scaled_columns:
                self.warnings.append("No columns were successfully scaled")
            
            return scaled_data
            
        except ScalingError:
            raise
        except Exception as e:
            return self._handle_error(e, "transform", data)

class CategoricalEncoder(PreprocessingStepBase):
    """Categorical encoding with comprehensive error handling"""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__("categorical_encoder", PreprocessingStep.ENCODING)
        self.config = config
        self.encoders = {}
        self.encoded_columns = []
        self.encoding_mappings = {}
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'CategoricalEncoder':
        """Fit encoding parameters with validation"""
        try:
            # Identify categorical columns
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            for col in categorical_columns:
                try:
                    unique_values = data[col].nunique()
                    
                    # Validate cardinality
                    if unique_values > self.config.max_categories:
                        self.warnings.append(
                            f"Column '{col}' has {unique_values} categories (max: {self.config.max_categories})"
                        )
                        continue
                    
                    if unique_values <= 1:
                        self.warnings.append(f"Column '{col}' has only {unique_values} unique values")
                        continue
                    
                    # Choose encoding method based on configuration
                    if self.config.categorical_encoding == "onehot":
                        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                        encoder.fit(data[[col]].fillna('missing'))
                        self.encoders[col] = encoder
                        
                    elif self.config.categorical_encoding == "label":
                        encoder = LabelEncoder()
                        encoder.fit(data[col].fillna('missing').astype(str))
                        self.encoders[col] = encoder
                        self.encoding_mappings[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                        
                    elif self.config.categorical_encoding == "frequency":
                        frequency_map = data[col].value_counts(normalize=True).to_dict()
                        self.encoders[col] = frequency_map
                        self.encoding_mappings[col] = frequency_map
                    
                    self.encoded_columns.append(col)
                    
                except Exception as e:
                    self.warnings.append(f"Failed to fit encoder for '{col}': {str(e)}")
            
            if not self.encoded_columns:
                self.warnings.append("No categorical columns were encoded")
            
            self.fitted = True
            return self
            
        except Exception as e:
            return self._handle_error(e, "fit")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply encoding with error recovery"""
        if not self.fitted:
            raise ValueError("CategoricalEncoder must be fitted before transform")
        
        try:
            encoded_data = data.copy()
            successfully_encoded = []
            failed_encodings = []
            
            for col in self.encoded_columns:
                if col not in encoded_data.columns:
                    self.warnings.append(f"Column '{col}' not found in transform data")
                    continue
                
                try:
                    if self.config.categorical_encoding == "onehot":
                        encoder = self.encoders[col]
                        # Handle missing values
                        col_data = encoded_data[[col]].fillna('missing')
                        encoded_values = encoder.transform(col_data)
                        
                        # Create feature names
                        feature_names = []
                        for i, cat in enumerate(encoder.categories_[0]):
                            # Sanitize feature name
                            safe_cat = str(cat).replace(' ', '_').replace('-', '_')
                            feature_names.append(f"{col}_{safe_cat}")
                        
                        # Add encoded columns
                        for i, feature_name in enumerate(feature_names):
                            encoded_data[feature_name] = encoded_values[:, i]
                        
                        # Drop original column
                        encoded_data = encoded_data.drop(columns=[col])
                        successfully_encoded.append(col)
                        
                    elif self.config.categorical_encoding == "label":
                        encoder = self.encoders[col]
                        # Handle unknown categories
                        col_data = encoded_data[col].fillna('missing').astype(str)
                        known_categories = set(encoder.classes_)
                        
                        # Map unknown categories to a default value
                        col_data_mapped = col_data.apply(
                            lambda x: x if x in known_categories else 'missing'
                        )
                        
                        encoded_data[col] = encoder.transform(col_data_mapped)
                        successfully_encoded.append(col)
                        
                    elif self.config.categorical_encoding == "frequency":
                        frequency_map = self.encoders[col]
                        # Handle unknown categories with default frequency
                        default_freq = 1.0 / (len(frequency_map) + 1)
                        encoded_data[col] = encoded_data[col].map(frequency_map).fillna(default_freq)
                        successfully_encoded.append(col)
                        
                except Exception as e:
                    failed_encodings.append(f"{col}: {str(e)}")
                    if not self.config.enable_error_recovery:
                        raise
            
            # Handle failures
            if failed_encodings:
                if self.config.fallback_strategy == "fail":
                    raise EncodingError(
                        f"Encoding failed for columns: {failed_encodings}",
                        error_code="ENCODING_FAILED"
                    )
                else:
                    self.warnings.extend(failed_encodings)
            
            return encoded_data
            
        except EncodingError:
            raise
        except Exception as e:
            return self._handle_error(e, "transform", data)

class FeatureSelector(PreprocessingStepBase):
    """Feature selection with quality validation"""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__("feature_selector", PreprocessingStep.FEATURE_SELECTION)
        self.config = config
        self.selector = None
        self.selected_features = []
        self.feature_scores = {}
        self.quality_validator = QualityValidator(config.validation_config)
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'FeatureSelector':
        """Fit feature selection with validation"""
        try:
            if not self.config.feature_selection or target is None:
                self.selected_features = data.columns.tolist()
                self.fitted = True
                return self
            
            # Prepare features
            X = data.select_dtypes(include=[np.number])
            
            # Validate features
            if X.shape[1] == 0:
                raise ValueError("No numerical features available for selection")
            
            # Remove constant features
            constant_features = []
            for col in X.columns:
                if X[col].nunique() <= 1:
                    constant_features.append(col)
            
            if constant_features:
                self.warnings.append(f"Removing constant features: {constant_features}")
                X = X.drop(columns=constant_features)
            
            # Determine number of features to select
            k = min(self.config.max_features or X.shape[1], X.shape[1])
            
            # Initialize selector
            if self.config.selection_method == "mutual_info":
                self.selector = SelectKBest(score_func=mutual_info_classif, k=k)
            elif self.config.selection_method == "f_classif":
                self.selector = SelectKBest(score_func=f_classif, k=k)
            else:
                self.selector = SelectKBest(score_func=mutual_info_classif, k=k)
            
            # Fit selector with error handling
            try:
                self.selector.fit(X, target)
                
                # Get selected features and scores
                selected_indices = self.selector.get_support(indices=True)
                self.selected_features = X.columns[selected_indices].tolist()
                
                # Store feature scores
                if hasattr(self.selector, 'scores_'):
                    self.feature_scores = dict(zip(X.columns, self.selector.scores_))
                
            except Exception as e:
                self.warnings.append(f"Feature selection failed: {str(e)}")
                # Fallback to selecting all features
                self.selected_features = X.columns.tolist()
            
            # Add non-numerical columns back
            non_numerical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
            self.selected_features.extend(non_numerical_cols)
            
            self.fitted = True
            return self
            
        except Exception as e:
            return self._handle_error(e, "fit")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection with validation"""
        if not self.fitted:
            raise ValueError("FeatureSelector must be fitted before transform")
        
        try:
            # Select only the chosen features
            available_features = [col for col in self.selected_features if col in data.columns]
            missing_features = [col for col in self.selected_features if col not in data.columns]
            
            if missing_features:
                self.warnings.append(f"Missing expected features: {missing_features}")
            
            if not available_features:
                raise PreprocessingException("No selected features available in data")
            
            selected_data = data[available_features]
            
            # Validate result
            is_valid, errors = self.quality_validator.validate_preprocessing_result(
                data, selected_data, "feature_selection"
            )
            
            if not is_valid and not self.config.enable_error_recovery:
                raise PreprocessingException(
                    f"Feature selection quality check failed: {'; '.join(errors)}"
                )
            elif not is_valid:
                self.warnings.extend(errors)
            
            return selected_data
            
        except PreprocessingException:
            raise
        except Exception as e:
            return self._handle_error(e, "transform", data)

# ============================================================================
# Enhanced Financial Data Preprocessor
# ============================================================================

class EnhancedFinancialDataPreprocessor:
    """
    Comprehensive financial data preprocessor with advanced error handling
    
    Features:
    - Comprehensive input validation for all preprocessing operations
    - Custom exception hierarchy for specific error types
    - Error recovery mechanisms with fallback strategies
    - Performance monitoring with timeout and memory limits
    - Security validation for data protection
    - Quality validation for preprocessing results
    - Integration validation for pipeline compatibility
    - Detailed error reporting and diagnostics
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize financial data preprocessor with validation
        
        Args:
            config: Preprocessing configuration with validation settings
        """
        self.config = config or PreprocessingConfig()
        self.preprocessing_steps: List[PreprocessingStepBase] = []
        self.fitted = False
        self.feature_names = []
        self.feature_types = {}
        self.preprocessing_stats = {
            'total_preprocessings': 0,
            'successful_preprocessings': 0,
            'failed_preprocessings': 0,
            'partial_preprocessings': 0,
            'total_records_processed': 0,
            'average_processing_time': 0.0,
            'errors_encountered': []
        }
        self._lock = threading.RLock()
        
        # Initialize validators
        self.input_validator = InputValidator(self.config.validation_config)
        self.security_validator = SecurityValidator(self.config.security_config)
        self.quality_validator = QualityValidator(self.config.validation_config)
        self.performance_monitor = PerformanceMonitor(
            self.config.security_config.max_memory_mb,
            self.config.security_config.max_execution_time_seconds
        )
        
        # Validate configuration
        self._validate_configuration()
        
        # Initialize preprocessing pipeline
        self._initialize_pipeline()
        
        logger.info("Enhanced FinancialDataPreprocessor initialized with validation")
    
    def _validate_configuration(self):
        """Validate preprocessing configuration"""
        is_valid, errors = self.input_validator.validate_preprocessing_config(self.config)
        if not is_valid:
            raise PreprocessingConfigError(
                f"Invalid preprocessing configuration: {'; '.join(errors)}",
                error_code="INVALID_CONFIG"
            )
    
    def _initialize_pipeline(self):
        """Initialize the preprocessing pipeline with error handling"""
        try:
            self.preprocessing_steps = [
                DataCleaner(self.config),
                FeatureEngineer(self.config),
                CategoricalEncoder(self.config),
                FeatureScaler(self.config),
                FeatureSelector(self.config)
            ]
        except Exception as e:
            raise PreprocessingConfigError(
                f"Failed to initialize preprocessing pipeline: {str(e)}",
                error_code="PIPELINE_INIT_ERROR"
            )
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'EnhancedFinancialDataPreprocessor':
        """
        Fit the preprocessing pipeline with comprehensive validation
        
        Args:
            data: Input DataFrame
            target: Target variable for supervised feature selection
            
        Returns:
            Fitted preprocessor
        """
        start_time = time.time()
        errors_encountered = []
        
        try:
            # Validate input data
            is_valid, errors = self.input_validator.validate_dataframe(data, "fit_input")
            if not is_valid:
                raise PreprocessingInputError(
                    f"Invalid input data for fit: {'; '.join(errors)}",
                    error_code="INVALID_FIT_DATA"
                )
            
            # Security validation
            is_valid, errors = self.security_validator.validate_column_names(data.columns.tolist())
            if not is_valid and not self.config.enable_error_recovery:
                raise PreprocessingSecurityError(
                    f"Security validation failed: {'; '.join(errors)}",
                    error_code="SECURITY_VALIDATION_FAILED"
                )
            
            logger.info(f"Fitting preprocessor on {len(data)} records with {len(data.columns)} features")
            
            current_data = data.copy()
            successful_steps = []
            failed_steps = []
            
            # Fit each preprocessing step with error handling
            for step in self.preprocessing_steps:
                try:
                    logger.debug(f"Fitting {step.step_name}")
                    
                    # Monitor performance
                    with self.performance_monitor.monitor_operation(f"fit_{step.step_name}"):
                        step.fit(current_data, target)
                        current_data = step.transform(current_data)
                    
                    successful_steps.append(step.step_name)
                    
                except (PreprocessingTimeoutError, PreprocessingMemoryError) as e:
                    # Critical errors - cannot continue
                    failed_steps.append(f"{step.step_name}: {str(e)}")
                    raise
                    
                except Exception as e:
                    failed_steps.append(f"{step.step_name}: {str(e)}")
                    errors_encountered.append({
                        'step': step.step_name,
                        'error': str(e),
                        'error_type': type(e).__name__
                    })
                    
                    if self.config.enable_error_recovery:
                        logger.warning(f"Error in {step.step_name}, attempting recovery: {str(e)}")
                        # Try to continue with current data
                        if step.step_name in ['feature_selector', 'categorical_encoder']:
                            # Non-critical steps - can skip
                            continue
                    else:
                        raise
            
            # Store final feature information
            self.feature_names = current_data.columns.tolist()
            self.feature_types = self._determine_feature_types(current_data)
            
            # Validate final result
            is_valid, errors = self.quality_validator.validate_preprocessing_result(
                data, current_data, "fit_result"
            )
            if not is_valid:
                logger.warning(f"Quality validation warnings: {'; '.join(errors)}")
            
            self.fitted = True
            
            # Update statistics
            processing_time = time.time() - start_time
            with self._lock:
                self.preprocessing_stats['successful_preprocessings'] += 1
                if failed_steps:
                    self.preprocessing_stats['partial_preprocessings'] += 1
            
            logger.info(f"Preprocessor fitted successfully in {processing_time:.2f}s. "
                       f"Final feature count: {len(self.feature_names)}")
            
            if failed_steps:
                logger.warning(f"Some steps failed during fit: {failed_steps}")
            
            return self
            
        except PreprocessingException:
            with self._lock:
                self.preprocessing_stats['failed_preprocessings'] += 1
                self.preprocessing_stats['errors_encountered'].extend(errors_encountered)
            raise
            
        except Exception as e:
            with self._lock:
                self.preprocessing_stats['failed_preprocessings'] += 1
            
            raise PreprocessingException(
                f"Unexpected error during fit: {str(e)}",
                error_code="FIT_ERROR",
                details={'original_error': str(e), 'error_type': type(e).__name__}
            )
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data with comprehensive error handling and recovery
        
        Args:
            data: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        start_time = time.time()
        errors_encountered = []
        warnings = []
        
        try:
            # Validate input data
            is_valid, errors = self.input_validator.validate_dataframe(data, "transform_input")
            if not is_valid and not self.config.enable_error_recovery:
                raise PreprocessingInputError(
                    f"Invalid input data for transform: {'; '.join(errors)}",
                    error_code="INVALID_TRANSFORM_DATA"
                )
            elif not is_valid:
                warnings.extend(errors)
            
            with self._lock:
                self.preprocessing_stats['total_preprocessings'] += 1
                self.preprocessing_stats['total_records_processed'] += len(data)
            
            logger.info(f"Transforming {len(data)} records")
            
            current_data = data.copy()
            applied_steps = []
            failed_steps = []
            
            # Apply each preprocessing step with error recovery
            for step in self.preprocessing_steps:
                if not step.fitted:
                    logger.warning(f"Skipping unfitted step: {step.step_name}")
                    continue
                
                try:
                    logger.debug(f"Applying {step.step_name}")
                    
                    # Monitor performance
                    with self.performance_monitor.monitor_operation(f"transform_{step.step_name}"):
                        # Store data before transformation for recovery
                        data_before = current_data.copy() if self.config.enable_error_recovery else None
                        
                        # Apply transformation
                        current_data = step.transform(current_data)
                        
                        # Collect warnings from step
                        if hasattr(step, 'warnings'):
                            warnings.extend(step.warnings)
                    
                    applied_steps.append(step.step_name)
                    
                except (PreprocessingTimeoutError, PreprocessingMemoryError) as e:
                    # Critical errors
                    failed_steps.append(f"{step.step_name}: {str(e)}")
                    raise
                    
                except PartialPreprocessingError as e:
                    # Partial failure - use partial data if available
                    if e.partial_data is not None and self.config.fallback_strategy == "partial":
                        current_data = e.partial_data
                        warnings.append(f"Partial failure in {step.step_name}: {str(e)}")
                        applied_steps.append(f"{step.step_name} (partial)")
                    else:
                        failed_steps.append(f"{step.step_name}: {str(e)}")
                        if self.config.fallback_strategy == "fail":
                            raise
                    
                except Exception as e:
                    failed_steps.append(f"{step.step_name}: {str(e)}")
                    errors_encountered.append({
                        'step': step.step_name,
                        'error': str(e),
                        'error_type': type(e).__name__
                    })
                    
                    if self.config.enable_error_recovery:
                        logger.warning(f"Error in {step.step_name}, using recovery strategy: {str(e)}")
                        
                        if self.config.fallback_strategy == "skip":
                            # Skip this step
                            continue
                        elif self.config.fallback_strategy == "partial" and data_before is not None:
                            # Revert to data before this step
                            current_data = data_before
                            warnings.append(f"Reverted {step.step_name} due to error")
                        else:
                            raise
                    else:
                        raise
            
            # Validate final result
            if current_data.empty:
                raise PreprocessingException("Preprocessing resulted in empty DataFrame")
            
            # Quality validation
            is_valid, quality_errors = self.quality_validator.validate_preprocessing_result(
                data, current_data, "transform_result"
            )
            if not is_valid:
                warnings.extend(quality_errors)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            with self._lock:
                total_preprocessings = self.preprocessing_stats['total_preprocessings']
                self.preprocessing_stats['average_processing_time'] = (
                    (self.preprocessing_stats['average_processing_time'] * (total_preprocessings - 1) + processing_time) 
                    / total_preprocessings
                )
                
                if failed_steps:
                    self.preprocessing_stats['partial_preprocessings'] += 1
                else:
                    self.preprocessing_stats['successful_preprocessings'] += 1
                
                self.preprocessing_stats['errors_encountered'].extend(errors_encountered)
            
            logger.info(f"Preprocessing completed in {processing_time:.2f}s")
            
            if warnings:
                logger.warning(f"Preprocessing warnings: {warnings[:5]}")  # Show first 5 warnings
            
            if failed_steps:
                logger.warning(f"Some steps failed during transform: {failed_steps}")
            
            return current_data
            
        except PreprocessingException:
            with self._lock:
                self.preprocessing_stats['failed_preprocessings'] += 1
            raise
            
        except Exception as e:
            with self._lock:
                self.preprocessing_stats['failed_preprocessings'] += 1
            
            raise PreprocessingException(
                f"Unexpected error during transform: {str(e)}",
                error_code="TRANSFORM_ERROR",
                details={'original_error': str(e), 'error_type': type(e).__name__}
            )
    
    def fit_transform(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform with unified error handling
        
        Args:
            data: Input DataFrame
            target: Target variable for supervised feature selection
            
        Returns:
            Preprocessed DataFrame
        """
        return self.fit(data, target).transform(data)
    
    def _determine_feature_types(self, data: pd.DataFrame) -> Dict[str, FeatureType]:
        """Determine feature types with validation"""
        feature_types = {}
        
        for col in data.columns:
            try:
                if data[col].dtype in ['int64', 'float64']:
                    unique_count = data[col].nunique()
                    if unique_count == 2:
                        feature_types[col] = FeatureType.BINARY
                    elif unique_count < 10 and unique_count == len(data[col].dropna().unique()):
                        # Might be ordinal
                        feature_types[col] = FeatureType.ORDINAL
                    else:
                        feature_types[col] = FeatureType.NUMERICAL
                elif data[col].dtype == 'object':
                    # Check if it's a datetime string
                    try:
                        pd.to_datetime(data[col].dropna().iloc[:10])
                        feature_types[col] = FeatureType.TEMPORAL
                    except:
                        feature_types[col] = FeatureType.CATEGORICAL
                elif data[col].dtype == 'category':
                    feature_types[col] = FeatureType.CATEGORICAL
                elif 'datetime' in str(data[col].dtype):
                    feature_types[col] = FeatureType.TEMPORAL
                else:
                    feature_types[col] = FeatureType.TEXTUAL
            except Exception as e:
                logger.warning(f"Failed to determine type for column '{col}': {str(e)}")
                feature_types[col] = FeatureType.TEXTUAL
        
        return feature_types
    
    def get_preprocessing_result(self, data: pd.DataFrame, 
                               processed_data: pd.DataFrame,
                               processing_time: float) -> PreprocessingResult:
        """
        Create comprehensive preprocessing result with validation info
        
        Args:
            data: Original data
            processed_data: Preprocessed data
            processing_time: Time taken for preprocessing
            
        Returns:
            Enhanced preprocessing result with validation report
        """
        # Get applied steps
        applied_steps = [step.step_name for step in self.preprocessing_steps if step.fitted]
        
        # Collect all warnings
        all_warnings = []
        for step in self.preprocessing_steps:
            if hasattr(step, 'warnings'):
                all_warnings.extend(step.warnings)
        
        # Calculate feature importance if available
        feature_importance = {}
        for step in self.preprocessing_steps:
            if isinstance(step, FeatureSelector) and hasattr(step, 'feature_scores'):
                feature_importance.update(step.feature_scores)
        
        # Get scaling and encoding parameters
        scaling_params = {}
        encoding_params = {}
        
        for step in self.preprocessing_steps:
            if isinstance(step, FeatureScaler):
                scaling_params = {col: {
                    'method': self.config.scaling_method.value,
                    'fitted': True
                } for col in step.scalers.keys()}
            elif isinstance(step, CategoricalEncoder):
                encoding_params = {col: {
                    'method': self.config.categorical_encoding,
                    'mapping': step.encoding_mappings.get(col, {})
                } for col in step.encoded_columns}
        
        # Create validation report
        validation_report = {
            'input_validation': self.input_validator.validate_dataframe(data)[1],
            'output_validation': self.quality_validator.validate_preprocessing_result(
                data, processed_data
            )[1],
            'security_validation': self.security_validator.validate_column_names(
                processed_data.columns.tolist()
            )[1],
            'performance_metrics': {
                'processing_time': processing_time,
                'memory_usage_mb': self.performance_monitor._get_memory_usage()
            }
        }
        
        # Get error summary
        with self._lock:
            errors_encountered = self.preprocessing_stats['errors_encountered'].copy()
        
        return PreprocessingResult(
            processed_data=processed_data,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            preprocessing_steps=applied_steps,
            preprocessing_time_seconds=processing_time,
            original_shape=data.shape,
            final_shape=processed_data.shape,
            scaling_params=scaling_params,
            encoding_params=encoding_params,
            feature_importance=feature_importance,
            metadata={
                'config': self.config.__dict__,
                'timestamp': datetime.now().isoformat(),
                'validation_enabled': self.config.validation_config.validate_inputs,
                'error_recovery_enabled': self.config.enable_error_recovery
            },
            validation_report=validation_report,
            errors_encountered=errors_encountered,
            warnings=all_warnings
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive preprocessing statistics"""
        with self._lock:
            stats = self.preprocessing_stats.copy()
            stats.update({
                'fitted': self.fitted,
                'feature_count': len(self.feature_names),
                'feature_types': {
                    ft.value: sum(1 for t in self.feature_types.values() if t == ft) 
                    for ft in FeatureType
                },
                'pipeline_steps': [step.step_name for step in self.preprocessing_steps],
                'error_recovery_enabled': self.config.enable_error_recovery,
                'validation_enabled': self.config.validation_config.validate_inputs,
                'security_enabled': self.config.security_config.protect_sensitive_features
            })
            
            # Calculate success rate
            total = stats['total_preprocessings']
            if total > 0:
                stats['success_rate'] = (
                    (stats['successful_preprocessings'] + stats['partial_preprocessings']) / total
                )
                stats['failure_rate'] = stats['failed_preprocessings'] / total
            
            return stats
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered"""
        with self._lock:
            errors = self.preprocessing_stats['errors_encountered']
            
            # Group errors by type
            error_types = {}
            for error in errors:
                error_type = error.get('error_type', 'Unknown')
                if error_type not in error_types:
                    error_types[error_type] = []
                error_types[error_type].append(error)
            
            # Get most common errors
            error_counts = {
                error_type: len(errors) 
                for error_type, errors in error_types.items()
            }
            
            return {
                'total_errors': len(errors),
                'error_types': error_types,
                'error_counts': error_counts,
                'most_common_errors': sorted(
                    error_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
            }
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"EnhancedFinancialDataPreprocessor(fitted={self.fitted}, "
               f"features={len(self.feature_names)}, "
               f"steps={len(self.preprocessing_steps)}, "
               f"processings={stats['total_preprocessings']}, "
               f"success_rate={stats.get('success_rate', 0):.2%})")

# Convenience functions
def create_enhanced_preprocessor(config: Optional[Dict[str, Any]] = None) -> EnhancedFinancialDataPreprocessor:
    """Create configured enhanced preprocessor instance"""
    if config:
        preprocessing_config = PreprocessingConfig(**config)
    else:
        preprocessing_config = PreprocessingConfig()
    return EnhancedFinancialDataPreprocessor(preprocessing_config)

# Export enhanced classes
__all__ = [
    'EnhancedFinancialDataPreprocessor',
    'PreprocessingConfig',
    'PreprocessingResult',
    'FeatureType',
    'ScalingMethod',
    'PreprocessingStep',
    'DataCleaner',
    'FeatureEngineer',
    'FeatureScaler',
    'CategoricalEncoder',
    'FeatureSelector',
    # Validation components
    'ValidationConfig',
    'SecurityConfig',
    'InputValidator',
    'PerformanceMonitor',
    'SecurityValidator',
    'QualityValidator',
    # Exception classes
    'PreprocessingException',
    'PreprocessingConfigError',
    'PreprocessingInputError',
    'PreprocessingTimeoutError',
    'PreprocessingMemoryError',
    'PreprocessingSecurityError',
    'PreprocessingIntegrationError',
    'FeatureEngineeringError',
    'EncodingError',
    'ScalingError',
    'ImputationError',
    'PartialPreprocessingError',
    'create_enhanced_preprocessor'
]

if __name__ == "__main__":
    # Test enhanced preprocessor
    print("Testing Enhanced Financial Data Preprocessor...")
    
    # Create test data
    np.random.seed(42)
    n_samples = 1000
    
    test_data = pd.DataFrame({
        'transaction_id': [f'TXN{i:06d}' for i in range(n_samples)],
        'user_id': [f'USER{np.random.randint(1, 100):03d}' for _ in range(n_samples)],
        'merchant_id': [f'MERCH{np.random.randint(1, 50):03d}' for _ in range(n_samples)],
        'amount': np.random.lognormal(mean=4, sigma=1.2, size=n_samples),
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='1h'),
        'location': [f'LOC{np.random.randint(1, 20):03d}' for _ in range(n_samples)],
        'payment_method': np.random.choice(['credit_card', 'debit_card', 'digital_wallet'], n_samples),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
    })
    
    # Introduce some data quality issues for testing
    test_data.loc[100:110, 'amount'] = np.nan  # Missing values
    test_data.loc[200:210, 'amount'] = 1000000  # Outliers
    test_data.loc[300:310, 'timestamp'] = 'invalid'  # Invalid timestamps
    
    # Create target variable
    target = test_data['is_fraud']
    test_data = test_data.drop('is_fraud', axis=1)
    
    # Initialize preprocessor with comprehensive config
    config = PreprocessingConfig(
        create_time_features=True,
        create_aggregation_features=True,
        create_ratio_features=True,
        feature_selection=True,
        max_features=30,
        enable_error_recovery=True,
        fallback_strategy="partial",
        validation_config=ValidationConfig(
            validate_inputs=True,
            validate_outputs=True,
            check_data_consistency=True
        ),
        security_config=SecurityConfig(
            max_memory_mb=1024,
            max_execution_time_seconds=60,
            protect_sensitive_features=True
        )
    )
    
    # Create preprocessor
    preprocessor = EnhancedFinancialDataPreprocessor(config)
    
    # Test preprocessing with error handling
    try:
        print("Original data shape:", test_data.shape)
        
        # Fit and transform with error recovery
        processed_data = preprocessor.fit_transform(test_data, target)
        print("Processed data shape:", processed_data.shape)
        print("Feature count:", len(preprocessor.feature_names))
        
        # Get preprocessing result with validation report
        result = preprocessor.get_preprocessing_result(
            test_data, processed_data, 
            preprocessor.preprocessing_stats['average_processing_time']
        )
        
        print("\nValidation Report:")
        print(f"- Input validation issues: {len(result.validation_report['input_validation'])}")
        print(f"- Output validation issues: {len(result.validation_report['output_validation'])}")
        print(f"- Warnings encountered: {len(result.warnings)}")
        print(f"- Errors encountered: {len(result.errors_encountered)}")
        
        # Get statistics
        stats = preprocessor.get_statistics()
        print(f"\nPreprocessing Statistics:")
        print(f"- Success rate: {stats.get('success_rate', 0):.2%}")
        print(f"- Average processing time: {stats['average_processing_time']:.2f}s")
        
        # Get error summary
        error_summary = preprocessor.get_error_summary()
        print(f"\nError Summary:")
        print(f"- Total errors: {error_summary['total_errors']}")
        print(f"- Error types: {list(error_summary['error_types'].keys())}")
        
    except PreprocessingException as e:
        print(f"\nPreprocessing failed: {e}")
        print(f"Error code: {e.error_code}")
        print(f"Details: {e.details}")
    
    print("\nEnhanced FinancialDataPreprocessor ready for production use!")