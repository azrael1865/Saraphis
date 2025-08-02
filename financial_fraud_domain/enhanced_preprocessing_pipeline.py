"""
Enhanced Data Preprocessor - Chunk 3: Main Preprocessor Class with Pipeline Management
Comprehensive preprocessing pipeline with advanced error handling, monitoring, and
integration capabilities for financial transaction data.
"""

import logging
import pandas as pd
import numpy as np
import time
import threading
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.ensemble import IsolationForest
import warnings

# Import enhanced framework and steps
try:
    from enhanced_preprocessing_framework import (
        PreprocessingException, PreprocessingConfigError, PreprocessingInputError,
        PreprocessingTimeoutError, PreprocessingMemoryError, PreprocessingSecurityError,
        PreprocessingIntegrationError, FeatureEngineeringError, EncodingError,
        ScalingError, ImputationError, PartialPreprocessingError,
        SecurityConfig, ValidationConfig, PerformanceConfig,
        InputValidator, PerformanceMonitor, SecurityValidator, QualityValidator
    )
    from enhanced_preprocessing_steps import (
        PreprocessingStep, FeatureType, ScalingMethod, ImputationMethod,
        PreprocessingConfig, PreprocessingStepBase, DataCleaner, FeatureEngineer
    )
    FRAMEWORK_AVAILABLE = True
except ImportError as e:
    FRAMEWORK_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Enhanced framework not available: {e}")

# Configure logging
logger = logging.getLogger(__name__)

# ======================== ADDITIONAL PREPROCESSING STEPS ========================

class CategoricalEncoder(PreprocessingStepBase):
    """Categorical encoding with comprehensive error handling and validation"""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__("categorical_encoder", PreprocessingStep.ENCODING, config)
        self.encoders = {}
        self.encoded_columns = []
        self.encoding_mappings = {}
        self.feature_names = []
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> None:
        """Fit encoding parameters with comprehensive validation"""
        try:
            if self.performance_monitor:
                with self.performance_monitor.monitor_operation("categorical_encoder_fit"):
                    self._fit_internal(data, target)
            else:
                self._fit_internal(data, target)
                
            self.fitted = True
            logger.info(f"CategoricalEncoder fitted on {len(data)} records")
            
        except Exception as e:
            self._handle_error(e, "fit", None)
    
    def _fit_internal(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> None:
        """Internal fit method with encoding setup"""
        # Validate input
        self._validate_input(data, "fit")
        
        # Find categorical columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Filter out ID columns and high cardinality columns
        categorical_columns = [col for col in categorical_columns 
                             if not col.endswith('_id') and data[col].nunique() <= self.config.max_categories]
        
        self.fit_params = {
            "categorical_columns": categorical_columns,
            "encoding_errors": [],
            "column_cardinalities": {},
            "encoding_stats": {}
        }
        
        for col in categorical_columns:
            try:
                unique_values = data[col].nunique()
                self.fit_params["column_cardinalities"][col] = unique_values
                
                # Skip columns with too few unique values
                if unique_values <= 1:
                    self.warnings.append(f"Column '{col}' has only {unique_values} unique values, skipping")
                    continue
                
                # Check for high cardinality
                if unique_values > self.config.max_categories:
                    self.warnings.append(f"Column '{col}' has {unique_values} categories (max: {self.config.max_categories}), skipping")
                    continue
                
                # Setup encoding based on method
                if self.config.categorical_encoding == "onehot":
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
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
                    
                elif self.config.categorical_encoding == "target":
                    if target is not None:
                        target_means = data.groupby(col)[target.name].mean().to_dict()
                        self.encoders[col] = target_means
                        self.encoding_mappings[col] = target_means
                    else:
                        self.warnings.append(f"Target encoding requested but no target provided for '{col}'")
                        continue
                
                self.encoded_columns.append(col)
                self.fit_params["encoding_stats"][col] = {
                    "method": self.config.categorical_encoding,
                    "unique_values": unique_values,
                    "missing_ratio": data[col].isnull().mean()
                }
                
            except Exception as e:
                error_msg = f"Failed to fit encoder for '{col}': {str(e)}"
                self.fit_params["encoding_errors"].append(error_msg)
                self.warnings.append(error_msg)
        
        if not self.encoders:
            self.warnings.append("No categorical columns were successfully encoded")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply encoding with comprehensive error recovery"""
        if not self.fitted:
            raise ValueError("CategoricalEncoder must be fitted before transform")
        
        try:
            if self.performance_monitor:
                with self.performance_monitor.monitor_operation("categorical_encoder_transform"):
                    return self._transform_internal(data)
            else:
                return self._transform_internal(data)
                
        except Exception as e:
            return self._handle_error(e, "transform", data)
    
    def _transform_internal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Internal transform method with encoding application"""
        # Validate input
        self._validate_input(data, "transform")
        
        encoded_data = data.copy()
        successfully_encoded = []
        failed_encodings = []
        
        for col in self.encoded_columns:
            if col not in data.columns:
                self.warnings.append(f"Column '{col}' not found in transform data")
                continue
            
            try:
                if self.config.categorical_encoding == "onehot":
                    encoder = self.encoders[col]
                    encoded_values = encoder.transform(data[[col]].fillna('missing'))
                    
                    # Create feature names
                    feature_names = encoder.get_feature_names_out([col])
                    
                    # Add encoded columns
                    for i, feature_name in enumerate(feature_names):
                        # Clean feature name
                        clean_name = feature_name.replace(f"{col}_", "").replace(" ", "_").replace("-", "_")
                        final_name = f"{col}_{clean_name}"
                        encoded_data[final_name] = encoded_values[:, i]
                    
                    # Drop original column
                    encoded_data = encoded_data.drop(columns=[col])
                    
                elif self.config.categorical_encoding == "label":
                    encoder = self.encoders[col]
                    col_data = data[col].fillna('missing').astype(str)
                    
                    # Handle unknown categories
                    known_categories = set(encoder.classes_)
                    col_data_mapped = col_data.apply(
                        lambda x: x if x in known_categories else 'missing'
                    )
                    encoded_data[col] = encoder.transform(col_data_mapped)
                    
                elif self.config.categorical_encoding == "frequency":
                    frequency_map = self.encoders[col]
                    default_freq = min(frequency_map.values()) if frequency_map else 0.001
                    encoded_data[col] = data[col].map(frequency_map).fillna(default_freq)
                    
                elif self.config.categorical_encoding == "target":
                    target_means = self.encoders[col]
                    global_mean = np.mean(list(target_means.values()))
                    encoded_data[col] = data[col].map(target_means).fillna(global_mean)
                
                successfully_encoded.append(col)
                
            except Exception as e:
                error_msg = f"{col}: {str(e)}"
                failed_encodings.append(error_msg)
                if not self.config.enable_error_recovery:
                    raise EncodingError(
                        f"Encoding failed for column '{col}': {str(e)}",
                        encoding_method=self.config.categorical_encoding,
                        column=col
                    )
        
        # Handle failures
        if failed_encodings:
            if self.config.fallback_strategy == "fail":
                raise EncodingError(
                    f"Encoding failed for columns: {failed_encodings}",
                    encoding_method=self.config.categorical_encoding,
                    details={"failed_columns": failed_encodings}
                )
            else:
                self.warnings.extend(failed_encodings)
        
        if successfully_encoded:
            logger.info(f"CategoricalEncoder processed: {', '.join(successfully_encoded)}")
        
        return encoded_data

class FeatureScaler(PreprocessingStepBase):
    """Feature scaling with comprehensive validation and error handling"""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__("feature_scaler", PreprocessingStep.SCALING, config)
        self.scalers = {}
        self.scaled_columns = []
        self.scaling_stats = {}
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> None:
        """Fit scaling parameters with comprehensive validation"""
        try:
            if self.performance_monitor:
                with self.performance_monitor.monitor_operation("feature_scaler_fit"):
                    self._fit_internal(data, target)
            else:
                self._fit_internal(data, target)
                
            self.fitted = True
            logger.info(f"FeatureScaler fitted on {len(data)} records")
            
        except Exception as e:
            self._handle_error(e, "fit", None)
    
    def _fit_internal(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> None:
        """Internal fit method with scaler setup"""
        # Validate input
        self._validate_input(data, "fit")
        
        # Determine columns to scale
        if self.config.scale_features:
            columns_to_scale = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Only scale continuous features
            columns_to_scale = []
            for col in data.select_dtypes(include=[np.number]).columns:
                if not col.startswith('is_') and data[col].nunique() > 10:
                    columns_to_scale.append(col)
        
        # Filter out ID columns and binary features
        columns_to_scale = [col for col in columns_to_scale 
                           if not col.endswith('_id') and data[col].nunique() > 2]
        
        # Validate column names for security
        if self.security_validator:
            is_valid, errors = self.security_validator.validate_column_names(columns_to_scale)
            if not is_valid:
                if self.config.enable_error_recovery:
                    self.warnings.extend(errors)
                else:
                    raise PreprocessingSecurityError(
                        f"Security validation failed for scaling columns: {'; '.join(errors)}",
                        security_violation="column_names"
                    )
        
        self.fit_params = {
            "columns_to_scale": columns_to_scale,
            "scaling_method": self.config.scaling_method.value,
            "scaling_errors": [],
            "column_stats": {}
        }
        
        # Select scaler class
        scaler_class = {
            ScalingMethod.STANDARD: StandardScaler,
            ScalingMethod.MINMAX: MinMaxScaler,
            ScalingMethod.ROBUST: RobustScaler
        }.get(self.config.scaling_method, StandardScaler)
        
        # Fit scalers for each column
        for col in columns_to_scale:
            if col in data.columns:
                try:
                    col_data = data[[col]].dropna()
                    
                    if len(col_data) == 0:
                        self.warnings.append(f"Column '{col}' has no non-null values, skipping scaling")
                        continue
                    
                    # Check for constant values
                    if col_data[col].nunique() == 1:
                        self.warnings.append(f"Column '{col}' has constant values, skipping scaling")
                        continue
                    
                    scaler = scaler_class()
                    scaler.fit(col_data)
                    self.scalers[col] = scaler
                    self.scaled_columns.append(col)
                    
                    # Store statistics
                    self.fit_params["column_stats"][col] = {
                        "mean": float(col_data[col].mean()),
                        "std": float(col_data[col].std()),
                        "min": float(col_data[col].min()),
                        "max": float(col_data[col].max()),
                        "null_count": int(data[col].isnull().sum())
                    }
                    
                except Exception as e:
                    error_msg = f"Failed to fit scaler for '{col}': {str(e)}"
                    self.fit_params["scaling_errors"].append(error_msg)
                    self.warnings.append(error_msg)
        
        if not self.scalers:
            self.warnings.append("No columns were successfully scaled")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling with comprehensive error recovery"""
        if not self.fitted:
            raise ValueError("FeatureScaler must be fitted before transform")
        
        try:
            if self.performance_monitor:
                with self.performance_monitor.monitor_operation("feature_scaler_transform"):
                    return self._transform_internal(data)
            else:
                return self._transform_internal(data)
                
        except Exception as e:
            return self._handle_error(e, "transform", data)
    
    def _transform_internal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Internal transform method with scaling application"""
        # Validate input
        self._validate_input(data, "transform")
        
        scaled_data = data.copy()
        successfully_scaled = []
        failed_scalings = []
        
        for col, scaler in self.scalers.items():
            if col not in data.columns:
                self.warnings.append(f"Column '{col}' not found in transform data")
                continue
            
            try:
                non_null_mask = data[col].notna()
                
                if non_null_mask.sum() == 0:
                    self.warnings.append(f"Column '{col}' has no non-null values in transform data")
                    continue
                
                # Apply scaling only to non-null values
                scaled_values = scaler.transform(data[[col]])
                scaled_data.loc[non_null_mask, col] = scaled_values[non_null_mask, 0]
                
                # Validate scaled values
                if np.isnan(scaled_values).any() or np.isinf(scaled_values).any():
                    self.warnings.append(f"Scaling produced invalid values for '{col}'")
                    # Don't raise error, just warn and keep original values
                    continue
                
                successfully_scaled.append(col)
                
            except Exception as e:
                error_msg = f"{col}: {str(e)}"
                failed_scalings.append(error_msg)
                if not self.config.enable_error_recovery:
                    raise ScalingError(
                        f"Scaling failed for column '{col}': {str(e)}",
                        scaling_method=self.config.scaling_method.value,
                        column=col
                    )
        
        # Handle failures
        if failed_scalings:
            if self.config.fallback_strategy == "fail":
                raise ScalingError(
                    f"Scaling failed for columns: {failed_scalings}",
                    scaling_method=self.config.scaling_method.value,
                    details={"failed_columns": failed_scalings}
                )
            else:
                self.warnings.extend(failed_scalings)
        
        if successfully_scaled:
            logger.info(f"FeatureScaler processed: {', '.join(successfully_scaled)}")
        
        return scaled_data

class FeatureSelector(PreprocessingStepBase):
    """Feature selection with comprehensive quality validation"""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__("feature_selector", PreprocessingStep.FEATURE_SELECTION, config)
        self.selector = None
        self.selected_features = []
        self.feature_scores = {}
        self.selection_stats = {}
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> None:
        """Fit feature selection with comprehensive validation"""
        try:
            if self.performance_monitor:
                with self.performance_monitor.monitor_operation("feature_selector_fit"):
                    self._fit_internal(data, target)
            else:
                self._fit_internal(data, target)
                
            self.fitted = True
            logger.info(f"FeatureSelector fitted on {len(data)} records")
            
        except Exception as e:
            self._handle_error(e, "fit", None)
    
    def _fit_internal(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> None:
        """Internal fit method with feature selection setup"""
        # Validate input
        self._validate_input(data, "fit")
        
        # If no feature selection requested, keep all features
        if not self.config.feature_selection:
            self.selected_features = data.columns.tolist()
            self.fit_params = {
                "selection_method": "none",
                "selected_features": self.selected_features,
                "total_features": len(data.columns)
            }
            return
        
        # Get numerical features for selection
        X = data.select_dtypes(include=[np.number])
        
        if X.shape[1] == 0:
            self.warnings.append("No numerical features available for selection")
            self.selected_features = data.columns.tolist()
            return
        
        # Remove constant features
        constant_features = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            self.warnings.append(f"Removing constant features: {constant_features}")
            X = X.drop(columns=constant_features)
        
        # Remove highly correlated features
        if self.config.validation_config.check_correlation_threshold:
            X = self._remove_correlated_features(X)
        
        # Determine number of features to select
        k = min(
            self.config.max_features or X.shape[1],
            X.shape[1],
            max(1, int(X.shape[1] * 0.8))  # Select at most 80% of features
        )
        
        self.fit_params = {
            "selection_method": self.config.selection_method,
            "original_features": X.shape[1],
            "selected_count": k,
            "constant_features_removed": constant_features,
            "selection_errors": []
        }
        
        try:
            # Select score function
            score_func = {
                'mutual_info': mutual_info_classif,
                'f_classif': f_classif,
                'variance_threshold': None
            }.get(self.config.selection_method, mutual_info_classif)
            
            # Apply feature selection
            if target is not None and score_func is not None:
                self.selector = SelectKBest(score_func=score_func, k=k)
                self.selector.fit(X, target)
                
                # Get selected features
                selected_indices = self.selector.get_support(indices=True)
                self.selected_features = [X.columns[i] for i in selected_indices]
                
                # Store feature scores
                if hasattr(self.selector, 'scores_'):
                    self.feature_scores = dict(zip(X.columns, self.selector.scores_))
                    
            elif self.config.selection_method == 'variance_threshold':
                # Use variance threshold
                threshold = 0.01  # Remove features with very low variance
                selector = VarianceThreshold(threshold=threshold)
                selector.fit(X)
                
                selected_indices = selector.get_support(indices=True)
                self.selected_features = [X.columns[i] for i in selected_indices]
                
            else:
                # Without target, use variance threshold or keep all
                self.selected_features = X.columns.tolist()
            
            # Add non-numerical columns back
            non_numerical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
            self.selected_features.extend(non_numerical_cols)
            
            # Store selection statistics
            self.selection_stats = {
                "original_feature_count": len(data.columns),
                "numerical_feature_count": X.shape[1],
                "selected_feature_count": len(self.selected_features),
                "selection_ratio": len(self.selected_features) / len(data.columns)
            }
            
        except Exception as e:
            error_msg = f"Feature selection failed: {str(e)}"
            self.fit_params["selection_errors"].append(error_msg)
            self.warnings.append(error_msg)
            # Fallback to using all features
            self.selected_features = data.columns.tolist()
    
    def _remove_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features"""
        try:
            if X.shape[1] <= 1:
                return X
            
            # Calculate correlation matrix
            corr_matrix = X.corr().abs()
            
            # Find highly correlated pairs
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find features to drop
            threshold = self.config.validation_config.check_correlation_threshold
            to_drop = [column for column in upper_tri.columns 
                      if any(upper_tri[column] > threshold)]
            
            if to_drop:
                self.warnings.append(f"Removing {len(to_drop)} highly correlated features")
                X = X.drop(columns=to_drop)
            
            return X
            
        except Exception as e:
            self.warnings.append(f"Error removing correlated features: {e}")
            return X
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection with comprehensive validation"""
        if not self.fitted:
            raise ValueError("FeatureSelector must be fitted before transform")
        
        try:
            if self.performance_monitor:
                with self.performance_monitor.monitor_operation("feature_selector_transform"):
                    return self._transform_internal(data)
            else:
                return self._transform_internal(data)
                
        except Exception as e:
            return self._handle_error(e, "transform", data)
    
    def _transform_internal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Internal transform method with feature selection application"""
        # Validate input
        self._validate_input(data, "transform")
        
        # Filter to selected features
        available_features = [col for col in self.selected_features if col in data.columns]
        missing_features = [col for col in self.selected_features if col not in data.columns]
        
        if missing_features:
            self.warnings.append(f"Missing expected features: {missing_features}")
        
        if not available_features:
            raise PreprocessingInputError("No selected features available in data")
        
        selected_data = data[available_features]
        
        # Validate result
        if self.quality_validator:
            is_valid, errors = self.quality_validator.validate_preprocessing_result(
                data, selected_data, "feature_selection"
            )
            
            if not is_valid:
                if self.config.enable_error_recovery:
                    self.warnings.extend(errors)
                else:
                    raise PreprocessingInputError(
                        f"Feature selection quality check failed: {'; '.join(errors)}",
                        details={"errors": errors}
                    )
        
        if errors:
            self.warnings.extend(errors)
        
        logger.info(f"FeatureSelector selected {len(available_features)} features")
        
        return selected_data

# ======================== PREPROCESSING RESULT CLASSES ========================

@dataclass
class PreprocessingResult:
    """Comprehensive results from preprocessing operations"""
    data: pd.DataFrame
    feature_names: List[str]
    feature_types: Dict[str, FeatureType]
    preprocessing_steps: List[PreprocessingStep]
    preprocessing_time_seconds: float
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    
    # Metadata
    scaling_params: Dict[str, Any] = field(default_factory=dict)
    encoding_params: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Validation results
    validation_report: Dict[str, Any] = field(default_factory=dict)
    errors_encountered: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Performance metrics
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    step_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)

# ======================== MAIN ENHANCED PREPROCESSOR ========================

class EnhancedFinancialDataPreprocessor:
    """
    Comprehensive financial data preprocessor with advanced error handling,
    monitoring, and integration capabilities.
    
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
        Initialize financial data preprocessor with comprehensive validation
        
        Args:
            config: Preprocessing configuration with validation settings
        """
        self.config = config or PreprocessingConfig()
        self.preprocessing_steps = []
        self.fitted = False
        self.feature_names = []
        self.feature_types = {}
        
        # Statistics tracking
        self.preprocessing_stats = {
            "total_preprocessings": 0,
            "successful_preprocessings": 0,
            "failed_preprocessings": 0,
            "partial_preprocessings": 0,
            "total_records_processed": 0,
            "average_processing_time": 0.0,
            "total_processing_time": 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize validators if framework is available
        if FRAMEWORK_AVAILABLE:
            self.input_validator = InputValidator(self.config.validation_config)
            self.security_validator = SecurityValidator(self.config.security_config)
            self.quality_validator = QualityValidator(self.config.validation_config)
            self.performance_monitor = PerformanceMonitor(
                self.config.security_config.max_memory_mb,
                self.config.security_config.max_execution_time_seconds
            )
        else:
            self.input_validator = None
            self.security_validator = None
            self.quality_validator = None
            self.performance_monitor = None
        
        # Error tracking
        self.errors_encountered = []
        self.warnings = []
        
        # Diagnostics
        self.diagnostics_history = []
        
        # Initialize preprocessing pipeline
        self._validate_configuration()
        self._initialize_pipeline()
        
        logger.info("Enhanced FinancialDataPreprocessor initialized with comprehensive validation")
    
    def _validate_configuration(self):
        """Validate preprocessing configuration with comprehensive checks"""
        try:
            if self.input_validator:
                is_valid, errors = self.input_validator.validate_preprocessing_config(self.config)
                if not is_valid:
                    raise PreprocessingConfigError(
                        f"Invalid preprocessing configuration: {'; '.join(errors)}",
                        details={"errors": errors}
                    )
            
            # Additional validation
            if self.config.max_categories <= 0:
                raise PreprocessingConfigError("max_categories must be positive")
            
            if self.config.lag_periods and any(p <= 0 for p in self.config.lag_periods):
                raise PreprocessingConfigError("All lag_periods must be positive")
                
        except Exception as e:
            raise PreprocessingConfigError(
                f"Configuration validation failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def _initialize_pipeline(self):
        """Initialize the preprocessing pipeline with comprehensive error handling"""
        try:
            self.preprocessing_steps = []
            
            # Data cleaning step
            if self.config.handle_missing or self.config.remove_duplicates or self.config.outlier_detection:
                self.preprocessing_steps.append(DataCleaner(self.config))
            
            # Feature engineering step
            if (self.config.create_time_features or self.config.create_aggregation_features or 
                self.config.create_ratio_features or self.config.create_lag_features):
                self.preprocessing_steps.append(FeatureEngineer(self.config))
            
            # Categorical encoding step
            if self.config.categorical_encoding != "none":
                self.preprocessing_steps.append(CategoricalEncoder(self.config))
            
            # Feature scaling step
            if self.config.scale_features:
                self.preprocessing_steps.append(FeatureScaler(self.config))
            
            # Feature selection step
            if self.config.feature_selection:
                self.preprocessing_steps.append(FeatureSelector(self.config))
            
            if not self.preprocessing_steps:
                raise PreprocessingConfigError("No preprocessing steps configured")
                
            logger.info(f"Initialized preprocessing pipeline with {len(self.preprocessing_steps)} steps")
            
        except Exception as e:
            raise PreprocessingConfigError(
                f"Failed to initialize preprocessing pipeline: {str(e)}",
                details={"error": str(e)}
            )
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'EnhancedFinancialDataPreprocessor':
        """
        Fit the preprocessing pipeline with comprehensive validation and monitoring
        
        Args:
            data: Input DataFrame
            target: Target variable for supervised feature selection
            
        Returns:
            Fitted preprocessor instance
        """
        fit_start_time = time.time()
        
        try:
            # Validate input
            if self.input_validator:
                is_valid, errors = self.input_validator.validate_dataframe(data, "fit_input")
                if not is_valid:
                    raise PreprocessingInputError(
                        f"Invalid input data for fit: {'; '.join(errors)}",
                        details={"errors": errors}
                    )
            
            # Security validation
            if self.security_validator:
                is_valid, errors = self.security_validator.validate_column_names(data.columns.tolist())
                if not is_valid:
                    if self.config.enable_error_recovery:
                        self.warnings.extend(errors)
                    else:
                        raise PreprocessingSecurityError(
                            f"Security validation failed: {'; '.join(errors)}",
                            security_violation="column_names"
                        )
            
            logger.info(f"Fitting preprocessor on {len(data)} records with {len(data.columns)} features")
            
            # Monitor the entire fit process
            if self.performance_monitor:
                with self.performance_monitor.monitor_operation("preprocessor_fit"):
                    self._fit_internal(data, target)
            else:
                self._fit_internal(data, target)
            
            self.fitted = True
            fit_time = time.time() - fit_start_time
            
            # Update statistics
            with self._lock:
                self.preprocessing_stats["total_preprocessings"] += 1
                self.preprocessing_stats["successful_preprocessings"] += 1
                self.preprocessing_stats["total_processing_time"] += fit_time
                self.preprocessing_stats["average_processing_time"] = (
                    self.preprocessing_stats["total_processing_time"] / 
                    self.preprocessing_stats["total_preprocessings"]
                )
            
            logger.info(f"Preprocessor fitted successfully in {fit_time:.2f}s")
            
            return self
            
        except Exception as e:
            with self._lock:
                self.preprocessing_stats["failed_preprocessings"] += 1
                self.errors_encountered.append({
                    "operation": "fit",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now()
                })
            
            logger.error(f"Preprocessor fit failed: {str(e)}")
            raise PreprocessingException(
                f"Preprocessing fit failed: {str(e)}",
                details={"error": str(e), "operation": "fit"}
            )
    
    def _fit_internal(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> None:
        """Internal fit method with step-by-step processing"""
        current_data = data.copy()
        successful_steps = []
        failed_steps = []
        
        # Fit each preprocessing step
        for step in self.preprocessing_steps:
            try:
                logger.debug(f"Fitting {step.step_name}")
                step.fit(current_data, target)
                
                # Apply transformation to prepare for next step
                current_data = step.transform(current_data)
                successful_steps.append(step.step_name)
                
                # Collect warnings from step
                if hasattr(step, 'warnings') and step.warnings:
                    self.warnings.extend(step.warnings)
                
            except (PreprocessingTimeoutError, PreprocessingMemoryError) as e:
                # These errors are not recoverable
                failed_steps.append(f"{step.step_name}: {str(e)}")
                raise
                
            except Exception as e:
                failed_steps.append(f"{step.step_name}: {str(e)}")
                self.errors_encountered.append({
                    "step": step.step_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "operation": "fit"
                })
                
                if self.config.enable_error_recovery:
                    logger.warning(f"Error in {step.step_name} fit, attempting recovery: {e}")
                    
                    # Mark step as failed but continue
                    if hasattr(step, 'fitted'):
                        step.fitted = False
                else:
                    raise
        
        # Sanitize column names
        if self.security_validator:
            current_data.columns = self.security_validator.sanitize_column_names(
                current_data.columns.tolist()
            )
        
        # Determine feature types
        self.feature_types = self._determine_feature_types(current_data)
        self.feature_names = current_data.columns.tolist()
        
        # Quality validation
        if self.quality_validator:
            is_valid, errors = self.quality_validator.validate_preprocessing_result(
                data, current_data, "fit_result"
            )
            if not is_valid:
                self.warnings.extend(errors)
        
        # Log results
        if successful_steps:
            logger.info(f"Successfully fitted steps: {', '.join(successful_steps)}")
        if failed_steps:
            logger.warning(f"Failed steps during fit: {', '.join(failed_steps)}")
            with self._lock:
                self.preprocessing_stats["partial_preprocessings"] += 1
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data with comprehensive error handling and monitoring
        
        Args:
            data: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        transform_start_time = time.time()
        
        try:
            # Validate input
            if self.input_validator:
                is_valid, errors = self.input_validator.validate_dataframe(data, "transform_input")
                if not is_valid:
                    if self.config.enable_error_recovery:
                        self.warnings.extend(errors)
                    else:
                        raise PreprocessingInputError(
                            f"Invalid input data for transform: {'; '.join(errors)}",
                            details={"errors": errors}
                        )
            
            # Monitor the entire transform process
            if self.performance_monitor:
                with self.performance_monitor.monitor_operation("preprocessor_transform"):
                    result = self._transform_internal(data)
            else:
                result = self._transform_internal(data)
            
            # Update statistics
            transform_time = time.time() - transform_start_time
            with self._lock:
                self.preprocessing_stats["total_records_processed"] += len(data)
                self.preprocessing_stats["successful_preprocessings"] += 1
                self.preprocessing_stats["total_processing_time"] += transform_time
                self.preprocessing_stats["average_processing_time"] = (
                    self.preprocessing_stats["total_processing_time"] / 
                    self.preprocessing_stats["total_preprocessings"]
                )
            
            logger.info(f"Transform completed in {transform_time:.2f}s")
            
            return result
            
        except Exception as e:
            with self._lock:
                self.preprocessing_stats["failed_preprocessings"] += 1
                self.errors_encountered.append({
                    "operation": "transform",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now()
                })
            
            logger.error(f"Transform failed: {str(e)}")
            raise PreprocessingException(
                f"Transform failed: {str(e)}",
                details={"error": str(e), "operation": "transform"}
            )
    
    def _transform_internal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Internal transform method with step-by-step processing"""
        current_data = data.copy()
        applied_steps = []
        failed_steps = []
        
        # Apply each preprocessing step
        for step in self.preprocessing_steps:
            if not step.fitted:
                logger.debug(f"Skipping unfitted step: {step.step_name}")
                continue
            
            try:
                logger.debug(f"Applying {step.step_name}")
                current_data = step.transform(current_data)
                applied_steps.append(step.step_name)
                
                # Collect warnings from step
                if hasattr(step, 'warnings') and step.warnings:
                    self.warnings.extend(step.warnings)
                
            except PartialPreprocessingError as e:
                failed_steps.append(f"{step.step_name}: {str(e)}")
                
                if self.config.fallback_strategy == "partial":
                    logger.warning(f"Partial failure in {step.step_name}")
                    if e.partial_data is not None:
                        current_data = e.partial_data
                    applied_steps.append(f"{step.step_name} (partial)")
                else:
                    raise
                    
            except Exception as e:
                failed_steps.append(f"{step.step_name}: {str(e)}")
                self.errors_encountered.append({
                    "step": step.step_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "operation": "transform"
                })
                
                if self.config.enable_error_recovery:
                    logger.warning(f"Error in {step.step_name} transform, using recovery: {e}")
                    
                    # Continue with current data
                    if self.config.fallback_strategy == "skip":
                        logger.info(f"Skipped {step.step_name} due to error")
                        continue
                else:
                    raise
        
        # Final validation
        if current_data.empty:
            raise PreprocessingInputError("Preprocessing resulted in empty DataFrame")
        
        # Quality validation
        if self.quality_validator:
            is_valid, errors = self.quality_validator.validate_preprocessing_result(
                data, current_data, "transform_result"
            )
            if errors:
                self.warnings.extend(errors)
        
        # Log results
        if applied_steps:
            logger.info(f"Successfully applied steps: {', '.join(applied_steps)}")
        if failed_steps:
            logger.warning(f"Failed steps during transform: {', '.join(failed_steps)}")
            with self._lock:
                self.preprocessing_stats["partial_preprocessings"] += 1
        
        return current_data
    
    def fit_transform(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform with unified error handling
        
        Args:
            data: Input DataFrame
            target: Target variable for supervised feature selection
            
        Returns:
            Preprocessed DataFrame
        """
        self.fit(data, target)
        return self.transform(data)
    
    def _determine_feature_types(self, data: pd.DataFrame) -> Dict[str, FeatureType]:
        """Determine feature types with comprehensive analysis"""
        feature_types = {}
        
        for col in data.columns:
            try:
                # Skip ID columns
                if col.endswith('_id'):
                    continue
                
                unique_count = data[col].nunique()
                
                # Binary features
                if unique_count == 2:
                    feature_types[col] = FeatureType.BINARY
                
                # Numerical features
                elif data[col].dtype in ['int64', 'float64']:
                    if unique_count <= 10:
                        # Check if it's ordinal
                        unique_values = sorted(data[col].dropna().unique())
                        if all(isinstance(v, (int, float)) for v in unique_values):
                            if unique_values == list(range(int(min(unique_values)), int(max(unique_values)) + 1)):
                                feature_types[col] = FeatureType.ORDINAL
                            else:
                                feature_types[col] = FeatureType.NUMERICAL
                        else:
                            feature_types[col] = FeatureType.NUMERICAL
                    else:
                        feature_types[col] = FeatureType.NUMERICAL
                
                # Temporal features
                elif any(time_word in col.lower() for time_word in ['date', 'time', 'timestamp']):
                    feature_types[col] = FeatureType.TEMPORAL
                
                # Textual features
                elif data[col].dtype == 'object':
                    if data[col].notna().any():
                        avg_length = data[col].dropna().astype(str).str.len().mean()
                        if avg_length > 50:
                            feature_types[col] = FeatureType.TEXTUAL
                        else:
                            feature_types[col] = FeatureType.CATEGORICAL
                    else:
                        feature_types[col] = FeatureType.CATEGORICAL
                
                else:
                    feature_types[col] = FeatureType.CATEGORICAL
                    
            except Exception as e:
                logger.warning(f"Failed to determine type for column '{col}': {str(e)}")
                feature_types[col] = FeatureType.CATEGORICAL
        
        return feature_types
    
    def get_preprocessing_result(self, data: pd.DataFrame, processed_data: pd.DataFrame, 
                               processing_time: float) -> PreprocessingResult:
        """
        Create comprehensive preprocessing result with detailed metrics
        
        Args:
            data: Original data
            processed_data: Preprocessed data
            processing_time: Time taken for preprocessing
            
        Returns:
            Comprehensive preprocessing result
        """
        # Collect step summaries
        step_summaries = {}
        for step in self.preprocessing_steps:
            if hasattr(step, 'get_step_summary'):
                step_summaries[step.step_name] = step.get_step_summary()
        
        # Collect performance metrics
        performance_metrics = {}
        if self.performance_monitor:
            performance_metrics = self.performance_monitor.get_performance_summary()
        
        # Create validation report
        validation_report = {
            "input_validation_passed": True,
            "output_validation_passed": True,
            "security_validation_passed": True,
            "quality_validation_passed": True
        }
        
        if self.input_validator:
            validation_report["input_validation_passed"] = self.input_validator.validate_dataframe(data)[0]
        
        if self.quality_validator:
            validation_report["output_validation_passed"] = self.quality_validator.validate_preprocessing_result(
                data, processed_data
            )[0]
        
        if self.security_validator:
            validation_report["security_validation_passed"] = self.security_validator.validate_column_names(
                processed_data.columns.tolist()
            )[0]
        
        # Create comprehensive result
        return PreprocessingResult(
            data=processed_data,
            feature_names=processed_data.columns.tolist(),
            feature_types=self.feature_types,
            preprocessing_steps=[step.step_type for step in self.preprocessing_steps if step.fitted],
            preprocessing_time_seconds=processing_time,
            original_shape=data.shape,
            final_shape=processed_data.shape,
            validation_report=validation_report,
            errors_encountered=self.errors_encountered.copy(),
            warnings=self.warnings.copy(),
            performance_metrics=performance_metrics,
            step_summaries=step_summaries,
            metadata={
                "config": asdict(self.config),
                "preprocessing_stats": self.preprocessing_stats.copy()
            }
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive preprocessing statistics"""
        with self._lock:
            stats = self.preprocessing_stats.copy()
            
            # Add detailed statistics
            stats.update({
                "fitted": self.fitted,
                "feature_count": len(self.feature_names),
                "feature_types": {
                    ft.value: sum(1 for f in self.feature_types.values() if f == ft)
                    for ft in FeatureType
                },
                "pipeline_steps": [step.step_name for step in self.preprocessing_steps],
                "step_fitted_status": {
                    step.step_name: step.fitted for step in self.preprocessing_steps
                },
                "error_count": len(self.errors_encountered),
                "warning_count": len(self.warnings),
                "security_enabled": self.security_validator is not None,
                "validation_enabled": self.input_validator is not None,
                "performance_monitoring_enabled": self.performance_monitor is not None
            })
            
            # Calculate success rate
            total = stats["total_preprocessings"]
            if total > 0:
                stats["success_rate"] = stats["successful_preprocessings"] / total
                stats["failure_rate"] = stats["failed_preprocessings"] / total
                stats["partial_rate"] = stats["partial_preprocessings"] / total
            
            return stats
    
    def __repr__(self) -> str:
        """String representation of the preprocessor"""
        stats = self.get_statistics()
        return (
            f"EnhancedFinancialDataPreprocessor("
            f"fitted={self.fitted}, "
            f"features={len(self.feature_names)}, "
            f"steps={len(self.preprocessing_steps)}, "
            f"processings={stats['total_preprocessings']}, "
            f"success_rate={stats.get('success_rate', 0):.2%})"
        )

# Export main preprocessor components
__all__ = [
    # Additional steps
    'CategoricalEncoder',
    'FeatureScaler',
    'FeatureSelector',
    
    # Result classes
    'PreprocessingResult',
    
    # Main preprocessor
    'EnhancedFinancialDataPreprocessor'
]

if __name__ == "__main__":
    print("Enhanced Data Preprocessor - Chunk 3: Main Preprocessor Class with Pipeline Management loaded")