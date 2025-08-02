"""
Enhanced Data Preprocessor - Chunk 2: Individual Preprocessing Steps with Error Handling
Comprehensive preprocessing steps with advanced error handling, validation, and recovery
mechanisms for financial transaction data.
"""

import logging
import pandas as pd
import numpy as np
import re
import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.cluster import KMeans
import warnings

# Import enhanced framework components
try:
    from enhanced_preprocessing_framework import (
        PreprocessingException, PreprocessingInputError, PreprocessingTimeoutError,
        PreprocessingMemoryError, FeatureEngineeringError, EncodingError,
        ScalingError, ImputationError, PartialPreprocessingError,
        SecurityConfig, ValidationConfig, PerformanceConfig,
        InputValidator, PerformanceMonitor, SecurityValidator, QualityValidator
    )
    FRAMEWORK_AVAILABLE = True
except ImportError as e:
    FRAMEWORK_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Enhanced framework not available: {e}")
    
    # Provide fallback classes when framework is not available
    class ValidationConfig:
        def __init__(self, **kwargs):
            pass
    
    class SecurityConfig:
        def __init__(self, **kwargs):
            pass
    
    class PerformanceConfig:
        def __init__(self, **kwargs):
            pass

# Configure logging
logger = logging.getLogger(__name__)

# ======================== PREPROCESSING ENUMS AND DATACLASSES ========================

class PreprocessingStep(Enum):
    """Preprocessing step types"""
    CLEANING = "cleaning"
    TRANSFORMATION = "transformation"
    FEATURE_ENGINEERING = "feature_engineering"
    SCALING = "scaling"
    ENCODING = "encoding"
    FEATURE_SELECTION = "feature_selection"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    IMPUTATION = "imputation"

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

class ImputationMethod(Enum):
    """Data imputation methods"""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    KNN = "knn"
    ITERATIVE = "iterative"

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing operations"""
    # Data cleaning
    handle_missing: bool = True
    missing_strategy: str = "drop"
    imputation_method: ImputationMethod = ImputationMethod.MEDIAN
    remove_duplicates: bool = True
    outlier_detection: bool = True
    outlier_method: str = "iqr"
    
    # Feature engineering
    create_time_features: bool = True
    create_aggregation_features: bool = True
    create_ratio_features: bool = True
    create_lag_features: bool = True
    lag_periods: List[int] = field(default_factory=lambda: [1, 3, 7])
    
    # Scaling
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    scale_features: bool = True
    
    # Encoding
    categorical_encoding: str = "onehot"
    max_categories: int = 50
    
    # Feature selection
    feature_selection: bool = True
    selection_method: str = "mutual_info"
    max_features: Optional[int] = None
    
    # Dimensionality reduction
    pca_components: Optional[float] = None
    
    # Advanced options
    create_interaction_features: bool = False
    polynomial_features: bool = False
    polynomial_degree: int = 2
    
    # Validation
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    security_config: SecurityConfig = field(default_factory=SecurityConfig)
    performance_config: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Error recovery
    enable_error_recovery: bool = True
    max_retries: int = 3
    fallback_strategy: str = "partial"

# ======================== BASE PREPROCESSING STEP CLASS ========================

class PreprocessingStepBase(ABC):
    """Abstract base class for preprocessing steps with comprehensive error handling"""
    
    def __init__(self, step_name: str, step_type: PreprocessingStep, config: PreprocessingConfig):
        self.step_name = step_name
        self.step_type = step_type
        self.config = config
        self.fitted = False
        self.fit_params = {}
        self.error_count = 0
        self.warnings = []
        self.processing_history = []
        self._lock = threading.RLock()
        
        # Initialize validators
        if FRAMEWORK_AVAILABLE:
            self.input_validator = InputValidator(config.validation_config)
            self.performance_monitor = PerformanceMonitor(
                config.security_config.max_memory_mb,
                config.security_config.max_execution_time_seconds
            )
            self.security_validator = SecurityValidator(config.security_config)
            self.quality_validator = QualityValidator(config.validation_config)
        else:
            self.input_validator = None
            self.performance_monitor = None
            self.security_validator = None
            self.quality_validator = None
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> None:
        """Fit the preprocessing step to the data"""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using fitted parameters"""
        pass
    
    def fit_transform(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step with error handling"""
        try:
            self.fit(data, target)
            return self.transform(data)
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in {self.step_name}.fit_transform: {str(e)}")
            
            if self.config.enable_error_recovery:
                self.warnings.append(f"fit_transform failed, returning original data: {str(e)}")
                return data
            else:
                raise
    
    def _handle_error(self, error: Exception, context: str, fallback_data: pd.DataFrame = None) -> pd.DataFrame:
        """Handle errors with recovery strategies"""
        with self._lock:
            self.error_count += 1
            error_record = {
                'timestamp': datetime.now(),
                'context': context,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'step_name': self.step_name
            }
            self.processing_history.append(error_record)
            
            logger.error(f"Error in {self.step_name}.{context}: {str(error)}")
            
            if self.config.enable_error_recovery:
                if fallback_data is not None:
                    self.warnings.append(f"Error in {context}, using fallback data: {str(error)}")
                    return fallback_data
                else:
                    self.warnings.append(f"Error in {context}, no fallback available: {str(error)}")
                    raise PreprocessingException(
                        f"Failed in {self.step_name}.{context} with no fallback",
                        error_code=f"{self.step_type.value.upper()}_ERROR",
                        details={"original_error": str(error), "context": context}
                    )
            else:
                raise PreprocessingException(
                    f"Failed in {self.step_name}.{context}: {str(error)}",
                    error_code=f"{self.step_type.value.upper()}_ERROR",
                    details={"original_error": str(error), "context": context}
                )
    
    def _validate_input(self, data: pd.DataFrame, context: str) -> bool:
        """Validate input data"""
        if self.input_validator:
            is_valid, errors = self.input_validator.validate_dataframe(data, f"{self.step_name}_{context}")
            if not is_valid:
                if self.config.enable_error_recovery:
                    self.warnings.extend(errors)
                    return False
                else:
                    raise PreprocessingInputError(
                        f"Input validation failed for {self.step_name}: {'; '.join(errors)}",
                        details={"errors": errors, "context": context}
                    )
        return True
    
    def get_step_summary(self) -> Dict[str, Any]:
        """Get summary of step processing"""
        with self._lock:
            return {
                "step_name": self.step_name,
                "step_type": self.step_type.value,
                "fitted": self.fitted,
                "error_count": self.error_count,
                "warning_count": len(self.warnings),
                "processing_history_count": len(self.processing_history),
                "recent_warnings": self.warnings[-5:] if self.warnings else []
            }

# ======================== DATA CLEANER ========================

class DataCleaner(PreprocessingStepBase):
    """Comprehensive data cleaning step with enhanced validation and recovery"""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__("data_cleaner", PreprocessingStep.CLEANING, config)
        self.imputer = None
        self.outlier_bounds = {}
        self.duplicate_info = {}
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> None:
        """Fit cleaning parameters with comprehensive validation"""
        try:
            if self.performance_monitor:
                with self.performance_monitor.monitor_operation("data_cleaner_fit"):
                    self._fit_internal(data, target)
            else:
                self._fit_internal(data, target)
            
            self.fitted = True
            logger.info(f"DataCleaner fitted on {len(data)} records")
            
        except Exception as e:
            self._handle_error(e, "fit", None)
    
    def _fit_internal(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> None:
        """Internal fit method with detailed parameter calculation"""
        # Validate input
        self._validate_input(data, "fit")
        
        # Initialize fit parameters
        self.fit_params = {
            "missing_columns": {},
            "duplicate_count": 0,
            "outlier_bounds": {},
            "original_dtypes": {},
            "imputation_strategy": {},
            "cleaning_stats": {}
        }
        
        # Analyze missing data
        try:
            missing_info = data.isnull().sum()
            self.fit_params["missing_columns"] = missing_info[missing_info > 0].to_dict()
            
            # Set up imputation strategy
            if self.config.handle_missing and self.config.missing_strategy == "fill":
                self._setup_imputation_strategy(data)
        
        except Exception as e:
            self.warnings.append(f"Error analyzing missing data: {e}")
        
        # Analyze duplicates
        try:
            duplicate_mask = data.duplicated(keep=False)
            self.fit_params["duplicate_count"] = duplicate_mask.sum()
            self.duplicate_info = {
                "duplicate_indices": data[duplicate_mask].index.tolist(),
                "duplicate_count": duplicate_mask.sum()
            }
        except Exception as e:
            self.warnings.append(f"Error analyzing duplicates: {e}")
        
        # Calculate outlier bounds
        if self.config.outlier_detection:
            try:
                self._calculate_outlier_bounds(data)
            except Exception as e:
                self.warnings.append(f"Error calculating outlier bounds: {e}")
        
        # Store original dtypes
        self.fit_params["original_dtypes"] = data.dtypes.to_dict()
        
        # Calculate cleaning statistics
        self.fit_params["cleaning_stats"] = {
            "total_records": len(data),
            "total_columns": len(data.columns),
            "missing_data_ratio": data.isnull().sum().sum() / (len(data) * len(data.columns)),
            "duplicate_ratio": self.fit_params["duplicate_count"] / len(data) if len(data) > 0 else 0
        }
    
    def _setup_imputation_strategy(self, data: pd.DataFrame) -> None:
        """Setup imputation strategy based on data types and missing patterns"""
        imputation_strategy = {}
        
        for col in data.columns:
            if data[col].isnull().any():
                try:
                    if data[col].dtype in ['int64', 'float64']:
                        # Numerical columns
                        if self.config.imputation_method == ImputationMethod.MEAN:
                            imputation_strategy[col] = {'method': 'mean', 'value': data[col].mean()}
                        elif self.config.imputation_method == ImputationMethod.MEDIAN:
                            imputation_strategy[col] = {'method': 'median', 'value': data[col].median()}
                        else:
                            imputation_strategy[col] = {'method': 'median', 'value': data[col].median()}
                    else:
                        # Categorical columns
                        mode_values = data[col].mode()
                        mode_value = mode_values[0] if len(mode_values) > 0 else "unknown"
                        imputation_strategy[col] = {'method': 'mode', 'value': mode_value}
                        
                except Exception as e:
                    self.warnings.append(f"Error setting imputation strategy for '{col}': {e}")
                    imputation_strategy[col] = {'method': 'constant', 'value': 0}
        
        self.fit_params["imputation_strategy"] = imputation_strategy
    
    def _calculate_outlier_bounds(self, data: pd.DataFrame) -> None:
        """Calculate outlier bounds for numerical columns"""
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            try:
                if self.config.outlier_method == "iqr":
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    if IQR == 0:
                        # Fallback to standard deviation method
                        mean = data[col].mean()
                        std = data[col].std()
                        if std == 0:
                            continue  # Skip constant columns
                        lower_bound = mean - 3 * std
                        upper_bound = mean + 3 * std
                    else:
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                elif self.config.outlier_method == "zscore":
                    mean = data[col].mean()
                    std = data[col].std()
                    
                    if std == 0:
                        continue  # Skip constant columns
                    
                    lower_bound = mean - 3 * std
                    upper_bound = mean + 3 * std
                    
                else:
                    # Percentile method
                    lower_bound = data[col].quantile(0.01)
                    upper_bound = data[col].quantile(0.99)
                
                self.fit_params["outlier_bounds"][col] = {
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'method': self.config.outlier_method
                }
                
            except Exception as e:
                self.warnings.append(f"Error calculating outlier bounds for '{col}': {e}")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning transformations with comprehensive error recovery"""
        if not self.fitted:
            raise ValueError("DataCleaner must be fitted before transform")
        
        try:
            if self.performance_monitor:
                with self.performance_monitor.monitor_operation("data_cleaner_transform"):
                    return self._transform_internal(data)
            else:
                return self._transform_internal(data)
                
        except Exception as e:
            return self._handle_error(e, "transform", data)
    
    def _transform_internal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Internal transform method with detailed processing"""
        # Validate input
        self._validate_input(data, "transform")
        
        cleaned_data = data.copy()
        applied_steps = []
        
        # Handle missing values
        if self.config.handle_missing:
            try:
                cleaned_data, missing_steps = self._handle_missing_values(cleaned_data)
                applied_steps.extend(missing_steps)
            except Exception as e:
                if self.config.enable_error_recovery:
                    self.warnings.append(f"Missing value handling failed: {e}")
                else:
                    raise
        
        # Remove duplicates
        if self.config.remove_duplicates:
            try:
                original_len = len(cleaned_data)
                cleaned_data = cleaned_data.drop_duplicates()
                removed_count = original_len - len(cleaned_data)
                if removed_count > 0:
                    applied_steps.append(f"removed {removed_count} duplicate rows")
            except Exception as e:
                if self.config.enable_error_recovery:
                    self.warnings.append(f"Duplicate removal failed: {e}")
                else:
                    raise
        
        # Handle outliers
        if self.config.outlier_detection:
            try:
                cleaned_data, outlier_steps = self._handle_outliers(cleaned_data)
                applied_steps.extend(outlier_steps)
            except Exception as e:
                if self.config.enable_error_recovery:
                    self.warnings.append(f"Outlier handling failed: {e}")
                else:
                    raise
        
        # Validate result
        if len(cleaned_data) == 0:
            raise PreprocessingInputError("Data cleaning resulted in empty DataFrame")
        
        # Log applied steps
        if applied_steps:
            logger.info(f"DataCleaner applied: {', '.join(applied_steps)}")
        
        return cleaned_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Handle missing values with multiple strategies"""
        steps = []
        
        if self.config.missing_strategy == "drop":
            original_len = len(data)
            data = data.dropna()
            if len(data) == 0:
                raise PreprocessingInputError("All rows removed by dropna")
            steps.append(f"dropped {original_len - len(data)} rows with missing values")
            
        elif self.config.missing_strategy == "fill":
            imputation_strategy = self.fit_params.get("imputation_strategy", {})
            
            for col, strategy in imputation_strategy.items():
                if col in data.columns and data[col].isnull().any():
                    try:
                        fill_value = strategy['value']
                        if pd.isna(fill_value):
                            # Fallback value
                            if data[col].dtype in ['int64', 'float64']:
                                fill_value = 0
                            else:
                                fill_value = "unknown"
                        
                        missing_count = data[col].isnull().sum()
                        data[col].fillna(fill_value, inplace=True)
                        steps.append(f"filled {missing_count} missing values in '{col}' with {strategy['method']}")
                        
                    except Exception as e:
                        self.warnings.append(f"Failed to fill missing values in '{col}': {e}")
                        # Simple fallback
                        if data[col].dtype in ['int64', 'float64']:
                            data[col].fillna(0, inplace=True)
                        else:
                            data[col].fillna("unknown", inplace=True)
                        
        elif self.config.missing_strategy == "interpolate":
            try:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if data[col].isnull().any():
                        original_nulls = data[col].isnull().sum()
                        data[col] = data[col].interpolate(method='linear', limit_direction='both')
                        remaining_nulls = data[col].isnull().sum()
                        if remaining_nulls > 0:
                            data[col].fillna(data[col].median(), inplace=True)
                        steps.append(f"interpolated {original_nulls - remaining_nulls} values in '{col}'")
            except Exception as e:
                self.warnings.append(f"Interpolation failed: {e}")
                # Fallback to forward fill
                data = data.fillna(method='ffill').fillna(method='bfill')
                steps.append("used forward/backward fill as interpolation fallback")
        
        return data, steps
    
    def _handle_outliers(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Handle outliers with bounds calculated during fit"""
        steps = []
        outlier_bounds = self.fit_params.get("outlier_bounds", {})
        
        for col, bounds in outlier_bounds.items():
            if col in data.columns:
                try:
                    lower_bound = bounds['lower_bound']
                    upper_bound = bounds['upper_bound']
                    
                    outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                    outlier_count = outlier_mask.sum()
                    
                    if outlier_count > 0:
                        # Remove outliers
                        data = data[~outlier_mask]
                        steps.append(f"removed {outlier_count} outliers from '{col}'")
                        
                except Exception as e:
                    self.warnings.append(f"Failed to handle outliers in '{col}': {e}")
        
        return data, steps

# ======================== FEATURE ENGINEER ========================

class FeatureEngineer(PreprocessingStepBase):
    """Advanced feature engineering with comprehensive error handling and recovery"""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__("feature_engineer", PreprocessingStep.FEATURE_ENGINEERING, config)
        self.feature_generators = {}
        self.feature_stats = {}
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> None:
        """Fit feature engineering parameters with comprehensive analysis"""
        try:
            if self.performance_monitor:
                with self.performance_monitor.monitor_operation("feature_engineer_fit"):
                    self._fit_internal(data, target)
            else:
                self._fit_internal(data, target)
                
            self.fitted = True
            logger.info(f"FeatureEngineer fitted on {len(data)} records")
            
        except Exception as e:
            self._handle_error(e, "fit", None)
    
    def _fit_internal(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> None:
        """Internal fit method with feature analysis"""
        # Validate input
        self._validate_input(data, "fit")
        
        self.fit_params = {
            "user_stats": {},
            "merchant_stats": {},
            "time_patterns": {},
            "amount_stats": {},
            "feature_creation_errors": [],
            "available_features": list(data.columns)
        }
        
        # Analyze user patterns
        if 'user_id' in data.columns and 'amount' in data.columns:
            try:
                user_stats = self._calculate_user_statistics(data)
                self.fit_params["user_stats"] = user_stats
            except Exception as e:
                self.fit_params["feature_creation_errors"].append(f"user_stats: {str(e)}")
        
        # Analyze merchant patterns
        if 'merchant_id' in data.columns and 'amount' in data.columns:
            try:
                merchant_stats = self._calculate_merchant_statistics(data)
                self.fit_params["merchant_stats"] = merchant_stats
            except Exception as e:
                self.fit_params["feature_creation_errors"].append(f"merchant_stats: {str(e)}")
        
        # Analyze time patterns
        if 'timestamp' in data.columns:
            try:
                time_patterns = self._analyze_time_patterns(data)
                self.fit_params["time_patterns"] = time_patterns
            except Exception as e:
                self.fit_params["feature_creation_errors"].append(f"time_patterns: {str(e)}")
        
        # Analyze amount patterns
        if 'amount' in data.columns:
            try:
                amount_stats = self._calculate_amount_statistics(data)
                self.fit_params["amount_stats"] = amount_stats
            except Exception as e:
                self.fit_params["feature_creation_errors"].append(f"amount_stats: {str(e)}")
    
    def _calculate_user_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate user-level statistics with error handling"""
        try:
            user_stats = data.groupby('user_id')['amount'].agg([
                'count', 'mean', 'std', 'min', 'max', 'median'
            ]).to_dict('index')
            
            # Add derived statistics
            for user_id, stats in user_stats.items():
                try:
                    stats['coeff_var'] = stats['std'] / stats['mean'] if stats['mean'] != 0 else 0
                    stats['range'] = stats['max'] - stats['min']
                    stats['is_high_volume'] = stats['count'] > data.groupby('user_id')['amount'].count().quantile(0.9)
                except Exception as e:
                    self.warnings.append(f"Error calculating derived stats for user {user_id}: {e}")
            
            return user_stats
            
        except Exception as e:
            self.warnings.append(f"Error calculating user statistics: {e}")
            return {}
    
    def _calculate_merchant_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate merchant-level statistics with error handling"""
        try:
            merchant_stats = data.groupby('merchant_id')['amount'].agg([
                'count', 'mean', 'std', 'min', 'max', 'median'
            ]).to_dict('index')
            
            # Add risk indicators
            for merchant_id, stats in merchant_stats.items():
                try:
                    stats['transaction_volume'] = stats['count']
                    stats['avg_transaction_size'] = stats['mean']
                    stats['volatility'] = stats['std'] / stats['mean'] if stats['mean'] != 0 else 0
                except Exception as e:
                    self.warnings.append(f"Error calculating merchant stats for {merchant_id}: {e}")
            
            return merchant_stats
            
        except Exception as e:
            self.warnings.append(f"Error calculating merchant statistics: {e}")
            return {}
    
    def _analyze_time_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns with comprehensive error handling"""
        try:
            timestamps = pd.to_datetime(data['timestamp'], errors='coerce')
            valid_timestamps = timestamps.dropna()
            
            if len(valid_timestamps) == 0:
                return {}
            
            time_patterns = {
                "business_hours_ratio": ((valid_timestamps.dt.hour >= 9) & 
                                       (valid_timestamps.dt.hour <= 17)).mean(),
                "weekend_ratio": (valid_timestamps.dt.dayofweek.isin([5, 6])).mean(),
                "night_ratio": ((valid_timestamps.dt.hour >= 22) | 
                               (valid_timestamps.dt.hour <= 6)).mean(),
                "peak_hours": valid_timestamps.dt.hour.mode().tolist(),
                "peak_days": valid_timestamps.dt.dayofweek.mode().tolist()
            }
            
            # Add seasonal patterns
            if len(valid_timestamps) > 30:  # Need sufficient data
                try:
                    time_patterns["monthly_pattern"] = valid_timestamps.dt.month.value_counts().to_dict()
                    time_patterns["daily_pattern"] = valid_timestamps.dt.day.value_counts().to_dict()
                except Exception as e:
                    self.warnings.append(f"Error calculating seasonal patterns: {e}")
            
            return time_patterns
            
        except Exception as e:
            self.warnings.append(f"Error analyzing time patterns: {e}")
            return {}
    
    def _calculate_amount_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate amount-based statistics with error handling"""
        try:
            amount_data = pd.to_numeric(data['amount'], errors='coerce').dropna()
            
            if len(amount_data) == 0:
                return {}
            
            amount_stats = {
                "mean": amount_data.mean(),
                "std": amount_data.std(),
                "median": amount_data.median(),
                "q25": amount_data.quantile(0.25),
                "q75": amount_data.quantile(0.75),
                "min": amount_data.min(),
                "max": amount_data.max(),
                "round_number_ratio": (amount_data % 1 == 0).mean(),
                "zero_count": (amount_data == 0).sum(),
                "negative_count": (amount_data < 0).sum()
            }
            
            # Add distribution analysis
            try:
                amount_stats["skewness"] = amount_data.skew()
                amount_stats["kurtosis"] = amount_data.kurtosis()
                amount_stats["outlier_ratio"] = self._calculate_outlier_ratio(amount_data)
            except Exception as e:
                self.warnings.append(f"Error calculating distribution stats: {e}")
            
            return amount_stats
            
        except Exception as e:
            self.warnings.append(f"Error calculating amount statistics: {e}")
            return {}
    
    def _calculate_outlier_ratio(self, data: pd.Series) -> float:
        """Calculate ratio of outliers using IQR method"""
        try:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                return 0.0
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (data < lower_bound) | (data > upper_bound)
            return outliers.mean()
            
        except Exception:
            return 0.0
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering with comprehensive error handling"""
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        try:
            if self.performance_monitor:
                with self.performance_monitor.monitor_operation("feature_engineer_transform"):
                    return self._transform_internal(data)
            else:
                return self._transform_internal(data)
                
        except Exception as e:
            return self._handle_error(e, "transform", data)
    
    def _transform_internal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Internal transform method with feature creation"""
        # Validate input
        self._validate_input(data, "transform")
        
        engineered_data = data.copy()
        created_features = []
        failed_features = []
        
        # Create time features
        if self.config.create_time_features and 'timestamp' in data.columns:
            try:
                engineered_data = self._create_time_features(engineered_data)
                created_features.append("time_features")
            except Exception as e:
                failed_features.append(f"time_features: {str(e)}")
                if not self.config.enable_error_recovery:
                    raise
        
        # Create user aggregation features
        if (self.config.create_aggregation_features and 
            'user_id' in data.columns and 
            self.fit_params.get("user_stats")):
            try:
                engineered_data = self._create_user_features(engineered_data)
                created_features.append("user_features")
            except Exception as e:
                failed_features.append(f"user_features: {str(e)}")
        
        # Create merchant aggregation features
        if (self.config.create_aggregation_features and 
            'merchant_id' in data.columns and 
            self.fit_params.get("merchant_stats")):
            try:
                engineered_data = self._create_merchant_features(engineered_data)
                created_features.append("merchant_features")
            except Exception as e:
                failed_features.append(f"merchant_features: {str(e)}")
        
        # Create amount features
        if 'amount' in data.columns:
            try:
                engineered_data = self._create_amount_features(engineered_data)
                created_features.append("amount_features")
            except Exception as e:
                failed_features.append(f"amount_features: {str(e)}")
        
        # Create ratio features
        if self.config.create_ratio_features:
            try:
                engineered_data = self._create_ratio_features(engineered_data)
                created_features.append("ratio_features")
            except Exception as e:
                failed_features.append(f"ratio_features: {str(e)}")
        
        # Create lag features
        if self.config.create_lag_features:
            try:
                engineered_data = self._create_lag_features(engineered_data)
                created_features.append("lag_features")
            except Exception as e:
                failed_features.append(f"lag_features: {str(e)}")
        
        # Handle failures
        if failed_features:
            if self.config.fallback_strategy == "partial":
                self.warnings.extend(failed_features)
            elif self.config.fallback_strategy == "fail":
                raise PartialPreprocessingError(
                    "Feature engineering partially failed",
                    successful_steps=created_features,
                    failed_steps=failed_features,
                    partial_data=engineered_data
                )
        
        # Log results
        if created_features:
            logger.info(f"FeatureEngineer created: {', '.join(created_features)}")
        
        return engineered_data
    
    def _create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features with comprehensive error handling"""
        try:
            timestamps = pd.to_datetime(data['timestamp'], errors='coerce')
            valid_mask = timestamps.notna()
            
            if valid_mask.sum() == 0:
                raise FeatureEngineeringError("No valid timestamps found")
            
            # Basic time components
            data.loc[valid_mask, 'hour'] = timestamps.dt.hour
            data.loc[valid_mask, 'day_of_week'] = timestamps.dt.dayofweek
            data.loc[valid_mask, 'day_of_month'] = timestamps.dt.day
            data.loc[valid_mask, 'month'] = timestamps.dt.month
            data.loc[valid_mask, 'quarter'] = timestamps.dt.quarter
            data.loc[valid_mask, 'year'] = timestamps.dt.year
            
            # Binary time features
            data.loc[valid_mask, 'is_weekend'] = timestamps.dt.dayofweek.isin([5, 6]).astype(int)
            data.loc[valid_mask, 'is_business_hours'] = (
                (timestamps.dt.hour >= 9) & (timestamps.dt.hour <= 17)
            ).astype(int)
            data.loc[valid_mask, 'is_night'] = (
                (timestamps.dt.hour >= 22) | (timestamps.dt.hour <= 6)
            ).astype(int)
            
            # Cyclical encoding for temporal features
            data.loc[valid_mask, 'hour_sin'] = np.sin(2 * np.pi * timestamps.dt.hour / 24)
            data.loc[valid_mask, 'hour_cos'] = np.cos(2 * np.pi * timestamps.dt.hour / 24)
            data.loc[valid_mask, 'day_of_week_sin'] = np.sin(2 * np.pi * timestamps.dt.dayofweek / 7)
            data.loc[valid_mask, 'day_of_week_cos'] = np.cos(2 * np.pi * timestamps.dt.dayofweek / 7)
            data.loc[valid_mask, 'month_sin'] = np.sin(2 * np.pi * timestamps.dt.month / 12)
            data.loc[valid_mask, 'month_cos'] = np.cos(2 * np.pi * timestamps.dt.month / 12)
            
            # Fill missing values for new features
            time_features = ['hour', 'day_of_week', 'day_of_month', 'month', 'quarter', 'year',
                           'is_weekend', 'is_business_hours', 'is_night',
                           'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                           'month_sin', 'month_cos']
            
            for feature in time_features:
                if feature in data.columns:
                    data[feature] = data[feature].fillna(0)
            
            return data
            
        except Exception as e:
            raise FeatureEngineeringError(
                f"Time feature creation failed: {str(e)}",
                feature_type="time"
            )
    
    def _create_user_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create user-based features with error handling"""
        try:
            user_stats = self.fit_params.get("user_stats", {})
            
            if not user_stats:
                return data
            
            # Map user statistics to data
            for stat_name in ['count', 'mean', 'std', 'min', 'max', 'median']:
                col_name = f"user_{stat_name}"
                stat_dict = {user_id: stats.get(stat_name, 0) for user_id, stats in user_stats.items()}
                data[col_name] = data['user_id'].map(stat_dict).fillna(0)
            
            # Create ratio features
            if 'amount' in data.columns and 'user_mean' in data.columns:
                epsilon = 1e-10
                data['amount_vs_user_mean'] = data['amount'] / (data['user_mean'] + epsilon)
                data['amount_vs_user_mean'] = data['amount_vs_user_mean'].replace([np.inf, -np.inf], 0)
                
                if 'user_std' in data.columns:
                    data['amount_vs_user_std'] = np.where(
                        data['user_std'] > epsilon,
                        (data['amount'] - data['user_mean']) / (data['user_std'] + epsilon),
                        0
                    )
                    data['amount_vs_user_std'] = data['amount_vs_user_std'].replace([np.inf, -np.inf], 0)
            
            return data
            
        except Exception as e:
            raise FeatureEngineeringError(
                f"User feature creation failed: {str(e)}",
                feature_type="user"
            )
    
    def _create_merchant_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create merchant-based features with error handling"""
        try:
            merchant_stats = self.fit_params.get("merchant_stats", {})
            
            if not merchant_stats:
                return data
            
            # Map merchant statistics to data
            for stat_name in ['count', 'mean', 'std', 'min', 'max', 'median']:
                col_name = f"merchant_{stat_name}"
                stat_dict = {merchant_id: stats.get(stat_name, 0) 
                           for merchant_id, stats in merchant_stats.items()}
                data[col_name] = data['merchant_id'].map(stat_dict).fillna(0)
            
            # Create merchant risk features
            if 'amount' in data.columns and 'merchant_mean' in data.columns:
                epsilon = 1e-10
                data['amount_vs_merchant_mean'] = data['amount'] / (data['merchant_mean'] + epsilon)
                data['amount_vs_merchant_mean'] = data['amount_vs_merchant_mean'].replace([np.inf, -np.inf], 0)
            
            return data
            
        except Exception as e:
            raise FeatureEngineeringError(
                f"Merchant feature creation failed: {str(e)}",
                feature_type="merchant"
            )
    
    def _create_amount_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create amount-based features with validation"""
        try:
            amount_stats = self.fit_params.get("amount_stats", {})
            amount_numeric = pd.to_numeric(data['amount'], errors='coerce')
            valid_mask = amount_numeric.notna()
            
            if valid_mask.sum() == 0:
                raise FeatureEngineeringError("No valid amount values found")
            
            # Mathematical transformations
            data['amount_log'] = np.log1p(amount_numeric.clip(lower=0))
            data['amount_sqrt'] = np.sqrt(amount_numeric.clip(lower=0))
            data['amount_squared'] = amount_numeric.clip(upper=1e6) ** 2
            
            # Statistical features
            if amount_stats:
                mean = amount_stats.get('mean', 0)
                std = amount_stats.get('std', 1)
                
                # Z-score normalization
                data['amount_zscore'] = (amount_numeric - mean) / (std + 1e-10)
                data['amount_zscore'] = data['amount_zscore'].fillna(0).replace([np.inf, -np.inf], 0)
                
                # Percentile features
                data['amount_percentile'] = amount_numeric.rank(pct=True)
            
            # Pattern-based features
            data['is_round_amount'] = (amount_numeric % 1 == 0).astype(int)
            data['is_round_10'] = (amount_numeric % 10 == 0).astype(int)
            data['is_round_100'] = (amount_numeric % 100 == 0).astype(int)
            data['is_zero_amount'] = (amount_numeric == 0).astype(int)
            data['is_negative_amount'] = (amount_numeric < 0).astype(int)
            
            # Digit analysis
            data['amount_digit_count'] = amount_numeric.astype(str).str.replace('.', '').str.len()
            data['amount_first_digit'] = amount_numeric.astype(str).str[0].astype(int, errors='ignore')
            
            # Fill missing values
            amount_features = ['amount_log', 'amount_sqrt', 'amount_squared', 'amount_zscore',
                             'amount_percentile', 'is_round_amount', 'is_round_10', 'is_round_100',
                             'is_zero_amount', 'is_negative_amount', 'amount_digit_count', 'amount_first_digit']
            
            for feature in amount_features:
                if feature in data.columns:
                    data[feature] = data[feature].fillna(0)
            
            return data
            
        except Exception as e:
            raise FeatureEngineeringError(
                f"Amount feature creation failed: {str(e)}",
                feature_type="amount"
            )
    
    def _create_ratio_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create ratio features with overflow protection"""
        try:
            numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            base_cols = [col for col in numerical_cols 
                        if not col.startswith('is_') and not col.endswith('_id')]
            
            max_ratios = 10
            created_ratios = 0
            
            for i, col1 in enumerate(base_cols[:10]):
                for col2 in base_cols[i+1:10]:
                    if col1 != col2 and created_ratios < max_ratios:
                        try:
                            epsilon = 1e-10
                            ratio_col = f"{col1}_to_{col2}_ratio"
                            
                            # Calculate ratio with protection
                            denominator = data[col2] + epsilon
                            data[ratio_col] = data[col1] / denominator
                            data[ratio_col] = data[ratio_col].replace([np.inf, -np.inf], 0)
                            
                            # Clip extreme values
                            if data[ratio_col].notna().any():
                                percentile_99 = data[ratio_col].quantile(0.99)
                                percentile_1 = data[ratio_col].quantile(0.01)
                                data[ratio_col] = data[ratio_col].clip(percentile_1, percentile_99)
                            
                            created_ratios += 1
                            
                        except Exception as e:
                            self.warnings.append(f"Failed to create ratio {col1}/{col2}: {str(e)}")
                    
                    if created_ratios >= max_ratios:
                        break
                
                if created_ratios >= max_ratios:
                    break
            
            return data
            
        except Exception as e:
            raise FeatureEngineeringError(
                f"Ratio feature creation failed: {str(e)}",
                feature_type="ratio"
            )
    
    def _create_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features with memory efficiency"""
        if not all(col in data.columns for col in ['user_id', 'timestamp']):
            return data
        
        try:
            # Sort by user and time
            data_sorted = data.sort_values(['user_id', 'timestamp']).copy()
            lag_periods = self.config.lag_periods or [1]
            
            # Create lag features for amount
            if 'amount' in data.columns:
                for lag in lag_periods[:3]:  # Limit to first 3 lags
                    try:
                        lag_col = f"amount_lag_{lag}"
                        data_sorted[lag_col] = data_sorted.groupby('user_id')['amount'].shift(lag)
                        
                        # Calculate change features
                        if data_sorted[lag_col].notna().any():
                            data_sorted[f"amount_change_{lag}"] = data_sorted['amount'] - data_sorted[lag_col]
                            
                            # Percentage change with protection
                            epsilon = 1e-10
                            data_sorted[f"amount_pct_change_{lag}"] = (
                                data_sorted[f"amount_change_{lag}"] / (data_sorted[lag_col] + epsilon)
                            ).fillna(0).replace([np.inf, -np.inf], 0)
                            
                    except Exception as e:
                        self.warnings.append(f"Failed to create lag {lag} features: {str(e)}")
            
            # Fill missing values
            lag_features = [col for col in data_sorted.columns 
                          if 'lag' in col or 'change' in col or 'pct_change' in col]
            
            for feature in lag_features:
                data_sorted[feature] = data_sorted[feature].fillna(0)
            
            return data_sorted
            
        except Exception as e:
            raise FeatureEngineeringError(
                f"Lag feature creation failed: {str(e)}",
                feature_type="lag"
            )

# Export preprocessing step components
__all__ = [
    # Enums
    'PreprocessingStep',
    'FeatureType',
    'ScalingMethod',
    'ImputationMethod',
    
    # Configuration
    'PreprocessingConfig',
    
    # Base class
    'PreprocessingStepBase',
    
    # Preprocessing steps
    'DataCleaner',
    'FeatureEngineer'
]

if __name__ == "__main__":
    print("Enhanced Data Preprocessor - Chunk 2: Individual Preprocessing Steps with Error Handling loaded")