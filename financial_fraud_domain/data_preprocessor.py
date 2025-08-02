"""
Comprehensive Data Preprocessing for Financial Fraud Detection
Advanced feature engineering, data transformation, and preprocessing pipeline
for financial transaction data with fraud detection capabilities.
Enhanced with production-ready error recovery and validation mechanisms.
"""

import logging
import pandas as pd
import numpy as np
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hashlib
import time

# Enhanced import
try:
    from enhanced_data_preprocessor import (
        EnhancedFinancialDataPreprocessor,
        PreprocessingConfig as EnhancedPreprocessingConfig,
        PreprocessingResult as EnhancedPreprocessingResult,
        PreprocessingException,
        FeatureEngineeringError,
        ValidationConfig,
        SecurityConfig
    )
    ENHANCED_AVAILABLE = True
except ImportError as e:
    ENHANCED_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Enhanced preprocessor not available: {e}")

# Configure logging
logger = logging.getLogger(__name__)

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

class PreprocessingStep(ABC):
    """Abstract base class for preprocessing steps"""
    
    def __init__(self, step_name: str, step_type: PreprocessingStep):
        self.step_name = step_name
        self.step_type = step_type
        self.fitted = False
        self.fit_params: Dict[str, Any] = {}
        
    @abstractmethod
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'PreprocessingStep':
        """Fit the preprocessing step to the data"""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using fitted parameters"""
        pass
    
    def fit_transform(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(data, target).transform(data)

class DataCleaner(PreprocessingStep):
    """Comprehensive data cleaning step"""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__("data_cleaner", PreprocessingStep.CLEANING)
        self.config = config
        
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'DataCleaner':
        """Fit cleaning parameters"""
        self.fit_params = {
            'missing_columns': data.isnull().sum().to_dict(),
            'duplicate_count': data.duplicated().sum(),
            'outlier_bounds': {}
        }
        
        # Calculate outlier bounds for numerical columns
        if self.config.outlier_detection:
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if self.config.outlier_method == "iqr":
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    self.fit_params['outlier_bounds'][col] = (lower_bound, upper_bound)
                elif self.config.outlier_method == "zscore":
                    mean = data[col].mean()
                    std = data[col].std()
                    self.fit_params['outlier_bounds'][col] = (mean - 3*std, mean + 3*std)
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning transformations"""
        if not self.fitted:
            raise ValueError("DataCleaner must be fitted before transform")
        
        cleaned_data = data.copy()
        
        # Handle missing values
        if self.config.handle_missing:
            if self.config.missing_strategy == "drop":
                cleaned_data = cleaned_data.dropna()
            elif self.config.missing_strategy == "fill":
                # Fill numerical with median, categorical with mode
                for col in cleaned_data.columns:
                    if cleaned_data[col].dtype in ['int64', 'float64']:
                        cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)
                    else:
                        cleaned_data[col].fillna(cleaned_data[col].mode().iloc[0] if not cleaned_data[col].mode().empty else 'unknown', inplace=True)
            elif self.config.missing_strategy == "interpolate":
                cleaned_data = cleaned_data.interpolate()
        
        # Remove duplicates
        if self.config.remove_duplicates:
            cleaned_data = cleaned_data.drop_duplicates()
        
        # Remove outliers
        if self.config.outlier_detection:
            for col, (lower_bound, upper_bound) in self.fit_params['outlier_bounds'].items():
                if col in cleaned_data.columns:
                    mask = (cleaned_data[col] >= lower_bound) & (cleaned_data[col] <= upper_bound)
                    cleaned_data = cleaned_data[mask]
        
        return cleaned_data

class FeatureEngineer(PreprocessingStep):
    """Advanced feature engineering for fraud detection"""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__("feature_engineer", PreprocessingStep.FEATURE_ENGINEERING)
        self.config = config
        
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'FeatureEngineer':
        """Fit feature engineering parameters"""
        self.fit_params = {
            'user_stats': {},
            'merchant_stats': {},
            'time_patterns': {},
            'amount_stats': {}
        }
        
        # Calculate user-based statistics
        if 'user_id' in data.columns and 'amount' in data.columns:
            user_stats = data.groupby('user_id')['amount'].agg([
                'count', 'mean', 'std', 'min', 'max', 'sum'
            ]).to_dict()
            self.fit_params['user_stats'] = user_stats
        
        # Calculate merchant-based statistics
        if 'merchant_id' in data.columns and 'amount' in data.columns:
            merchant_stats = data.groupby('merchant_id')['amount'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).to_dict()
            self.fit_params['merchant_stats'] = merchant_stats
        
        # Analyze time patterns
        if 'timestamp' in data.columns:
            timestamps = pd.to_datetime(data['timestamp'])
            self.fit_params['time_patterns'] = {
                'business_hours_ratio': ((timestamps.dt.hour >= 9) & (timestamps.dt.hour <= 17)).mean(),
                'weekend_ratio': timestamps.dt.dayofweek.isin([5, 6]).mean(),
                'night_ratio': ((timestamps.dt.hour >= 22) | (timestamps.dt.hour <= 6)).mean()
            }
        
        # Amount distribution statistics
        if 'amount' in data.columns:
            self.fit_params['amount_stats'] = {
                'mean': data['amount'].mean(),
                'std': data['amount'].std(),
                'q25': data['amount'].quantile(0.25),
                'q75': data['amount'].quantile(0.75),
                'round_number_ratio': (data['amount'] % 1 == 0).mean()
            }
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations"""
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        engineered_data = data.copy()
        
        # Time-based features
        if self.config.create_time_features and 'timestamp' in data.columns:
            engineered_data = self._create_time_features(engineered_data)
        
        # User-based aggregation features
        if self.config.create_aggregation_features:
            engineered_data = self._create_user_features(engineered_data)
            engineered_data = self._create_merchant_features(engineered_data)
        
        # Amount-based features
        if 'amount' in data.columns:
            engineered_data = self._create_amount_features(engineered_data)
        
        # Ratio and interaction features
        if self.config.create_ratio_features:
            engineered_data = self._create_ratio_features(engineered_data)
        
        # Lag features
        if self.config.create_lag_features:
            engineered_data = self._create_lag_features(engineered_data)
        
        # Transaction velocity features
        if all(col in data.columns for col in ['user_id', 'timestamp']):
            engineered_data = self._create_velocity_features(engineered_data)
        
        return engineered_data
    
    def _create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        timestamp = pd.to_datetime(data['timestamp'])
        
        # Basic time features
        data['hour'] = timestamp.dt.hour
        data['day_of_week'] = timestamp.dt.dayofweek
        data['day_of_month'] = timestamp.dt.day
        data['month'] = timestamp.dt.month
        data['quarter'] = timestamp.dt.quarter
        data['year'] = timestamp.dt.year
        
        # Advanced time features
        data['is_weekend'] = timestamp.dt.dayofweek.isin([5, 6]).astype(int)
        data['is_business_hours'] = ((timestamp.dt.hour >= 9) & (timestamp.dt.hour <= 17)).astype(int)
        data['is_night'] = ((timestamp.dt.hour >= 22) | (timestamp.dt.hour <= 6)).astype(int)
        data['is_early_morning'] = ((timestamp.dt.hour >= 1) & (timestamp.dt.hour <= 5)).astype(int)
        
        # Holiday indicators (simplified)
        data['is_holiday_season'] = (
            ((timestamp.dt.month == 12) & (timestamp.dt.day >= 20)) |
            ((timestamp.dt.month == 1) & (timestamp.dt.day <= 5)) |
            ((timestamp.dt.month == 7) & (timestamp.dt.day == 4)) |  # Independence Day
            ((timestamp.dt.month == 11) & (timestamp.dt.day >= 22) & (timestamp.dt.day <= 28))  # Thanksgiving week
        ).astype(int)
        
        # Cyclical encoding for time features
        data['hour_sin'] = np.sin(2 * np.pi * timestamp.dt.hour / 24)
        data['hour_cos'] = np.cos(2 * np.pi * timestamp.dt.hour / 24)
        data['day_of_week_sin'] = np.sin(2 * np.pi * timestamp.dt.dayofweek / 7)
        data['day_of_week_cos'] = np.cos(2 * np.pi * timestamp.dt.dayofweek / 7)
        data['month_sin'] = np.sin(2 * np.pi * timestamp.dt.month / 12)
        data['month_cos'] = np.cos(2 * np.pi * timestamp.dt.month / 12)
        
        return data
    
    def _create_user_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create user-based aggregation features"""
        if 'user_id' not in data.columns:
            return data
        
        user_stats = self.fit_params['user_stats']
        
        # Map user statistics
        for stat_name, stat_dict in user_stats.items():
            data[f'user_{stat_name}'] = data['user_id'].map(stat_dict).fillna(0)
        
        # User behavior patterns
        if 'amount' in data.columns:
            data['amount_vs_user_mean'] = data['amount'] / (data['user_mean'] + 1e-8)
            data['amount_vs_user_std'] = np.abs(data['amount'] - data['user_mean']) / (data['user_std'] + 1e-8)
            data['is_user_max_amount'] = (data['amount'] == data['user_max']).astype(int)
            data['is_user_min_amount'] = (data['amount'] == data['user_min']).astype(int)
        
        return data
    
    def _create_merchant_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create merchant-based aggregation features"""
        if 'merchant_id' not in data.columns:
            return data
        
        merchant_stats = self.fit_params['merchant_stats']
        
        # Map merchant statistics
        for stat_name, stat_dict in merchant_stats.items():
            data[f'merchant_{stat_name}'] = data['merchant_id'].map(stat_dict).fillna(0)
        
        # Merchant behavior patterns
        if 'amount' in data.columns:
            data['amount_vs_merchant_mean'] = data['amount'] / (data['merchant_mean'] + 1e-8)
            data['amount_vs_merchant_std'] = np.abs(data['amount'] - data['merchant_mean']) / (data['merchant_std'] + 1e-8)
        
        return data
    
    def _create_amount_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create amount-based features"""
        amount_stats = self.fit_params['amount_stats']
        
        # Basic amount features
        data['amount_log'] = np.log1p(data['amount'])
        data['amount_sqrt'] = np.sqrt(data['amount'])
        data['amount_squared'] = data['amount'] ** 2
        
        # Amount distribution features
        data['amount_zscore'] = (data['amount'] - amount_stats['mean']) / (amount_stats['std'] + 1e-8)
        data['amount_vs_median'] = data['amount'] / (amount_stats['q25'] + amount_stats['q75']) * 2
        data['is_round_amount'] = (data['amount'] % 1 == 0).astype(int)
        data['is_round_10'] = (data['amount'] % 10 == 0).astype(int)
        data['is_round_100'] = (data['amount'] % 100 == 0).astype(int)
        
        # Amount patterns
        data['amount_cents'] = (data['amount'] * 100) % 100
        data['amount_last_digit'] = (data['amount'] * 100) % 10
        data['has_cents'] = (data['amount'] % 1 != 0).astype(int)
        
        # Outlier indicators
        iqr = amount_stats['q75'] - amount_stats['q25']
        lower_bound = amount_stats['q25'] - 1.5 * iqr
        upper_bound = amount_stats['q75'] + 1.5 * iqr
        data['is_amount_outlier'] = ((data['amount'] < lower_bound) | (data['amount'] > upper_bound)).astype(int)
        
        return data
    
    def _create_ratio_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create ratio and interaction features"""
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if not col.startswith('is_') and not col.endswith('_id')]
        
        # Create some key ratios
        if len(numerical_cols) >= 2:
            for i, col1 in enumerate(numerical_cols[:5]):  # Limit to avoid explosion
                for col2 in numerical_cols[i+1:6]:
                    if col1 != col2:
                        # Ratio features
                        data[f'{col1}_to_{col2}_ratio'] = data[col1] / (data[col2] + 1e-8)
                        
                        # Difference features
                        data[f'{col1}_minus_{col2}'] = data[col1] - data[col2]
        
        return data
    
    def _create_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features for time series patterns"""
        if not all(col in data.columns for col in ['user_id', 'timestamp']):
            return data
        
        # Sort by user and timestamp
        data_sorted = data.sort_values(['user_id', 'timestamp'])
        
        # Create lag features for amount
        if 'amount' in data.columns:
            for lag in self.config.lag_periods:
                data_sorted[f'amount_lag_{lag}'] = data_sorted.groupby('user_id')['amount'].shift(lag)
                data_sorted[f'amount_change_{lag}'] = data_sorted['amount'] - data_sorted[f'amount_lag_{lag}']
                data_sorted[f'amount_pct_change_{lag}'] = data_sorted['amount'] / (data_sorted[f'amount_lag_{lag}'] + 1e-8) - 1
        
        return data_sorted
    
    def _create_velocity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create transaction velocity features"""
        # Sort by user and timestamp
        data_sorted = data.sort_values(['user_id', 'timestamp'])
        data_sorted['timestamp'] = pd.to_datetime(data_sorted['timestamp'])
        
        # Time since last transaction
        data_sorted['time_since_last'] = data_sorted.groupby('user_id')['timestamp'].diff()
        data_sorted['time_since_last_minutes'] = data_sorted['time_since_last'].dt.total_seconds() / 60
        data_sorted['time_since_last_hours'] = data_sorted['time_since_last'].dt.total_seconds() / 3600
        
        # Transaction frequency features
        data_sorted['hour_of_day'] = data_sorted['timestamp'].dt.hour
        
        # Rolling window features (last N transactions)
        for window in [3, 5, 10]:
            data_sorted[f'amount_mean_{window}'] = data_sorted.groupby('user_id')['amount'].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
            data_sorted[f'amount_std_{window}'] = data_sorted.groupby('user_id')['amount'].rolling(window=window, min_periods=1).std().reset_index(0, drop=True)
            data_sorted[f'transaction_count_{window}'] = data_sorted.groupby('user_id').cumcount() + 1
        
        return data_sorted

class FeatureScaler(PreprocessingStep):
    """Feature scaling and normalization"""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__("feature_scaler", PreprocessingStep.SCALING)
        self.config = config
        self.scalers = {}
        
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'FeatureScaler':
        """Fit scaling parameters"""
        # Determine columns to scale
        if self.config.scale_features:
            columns_to_scale = self.config.scale_features
        else:
            # Auto-detect numerical columns
            columns_to_scale = data.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude binary features and IDs
            columns_to_scale = [col for col in columns_to_scale if not col.endswith('_id') and data[col].nunique() > 2]
        
        # Initialize scaler based on method
        if self.config.scaling_method == ScalingMethod.STANDARD:
            scaler_class = StandardScaler
        elif self.config.scaling_method == ScalingMethod.MINMAX:
            scaler_class = MinMaxScaler
        elif self.config.scaling_method == ScalingMethod.ROBUST:
            scaler_class = RobustScaler
        else:
            scaler_class = StandardScaler
        
        # Fit scalers for each column
        for col in columns_to_scale:
            if col in data.columns:
                self.scalers[col] = scaler_class()
                self.scalers[col].fit(data[[col]])
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling transformations"""
        if not self.fitted:
            raise ValueError("FeatureScaler must be fitted before transform")
        
        scaled_data = data.copy()
        
        # Apply scaling to each column
        for col, scaler in self.scalers.items():
            if col in scaled_data.columns:
                scaled_data[col] = scaler.transform(scaled_data[[col]]).flatten()
        
        return scaled_data

class CategoricalEncoder(PreprocessingStep):
    """Categorical feature encoding"""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__("categorical_encoder", PreprocessingStep.ENCODING)
        self.config = config
        self.encoders = {}
        self.encoded_columns = []
        
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'CategoricalEncoder':
        """Fit encoding parameters"""
        # Identify categorical columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_columns:
            unique_values = data[col].nunique()
            
            # Skip if too many categories
            if unique_values > self.config.max_categories:
                logger.warning(f"Column {col} has {unique_values} categories, skipping encoding")
                continue
            
            if self.config.categorical_encoding == "onehot":
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoder.fit(data[[col]])
                self.encoders[col] = encoder
                
            elif self.config.categorical_encoding == "label":
                encoder = LabelEncoder()
                encoder.fit(data[col].fillna('missing'))
                self.encoders[col] = encoder
                
            elif self.config.categorical_encoding == "frequency":
                frequency_map = data[col].value_counts().to_dict()
                self.encoders[col] = frequency_map
                
            self.encoded_columns.append(col)
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply encoding transformations"""
        if not self.fitted:
            raise ValueError("CategoricalEncoder must be fitted before transform")
        
        encoded_data = data.copy()
        
        for col in self.encoded_columns:
            if col not in encoded_data.columns:
                continue
                
            if self.config.categorical_encoding == "onehot":
                encoder = self.encoders[col]
                encoded_values = encoder.transform(encoded_data[[col]])
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                
                # Add encoded columns
                for i, feature_name in enumerate(feature_names):
                    encoded_data[feature_name] = encoded_values[:, i]
                
                # Drop original column
                encoded_data = encoded_data.drop(columns=[col])
                
            elif self.config.categorical_encoding == "label":
                encoder = self.encoders[col]
                encoded_data[col] = encoder.transform(encoded_data[col].fillna('missing'))
                
            elif self.config.categorical_encoding == "frequency":
                frequency_map = self.encoders[col]
                encoded_data[col] = encoded_data[col].map(frequency_map).fillna(0)
        
        return encoded_data

class FeatureSelector(PreprocessingStep):
    """Feature selection and dimensionality reduction"""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__("feature_selector", PreprocessingStep.FEATURE_SELECTION)
        self.config = config
        self.selector = None
        self.selected_features = []
        
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'FeatureSelector':
        """Fit feature selection parameters"""
        if not self.config.feature_selection or target is None:
            self.selected_features = data.columns.tolist()
            self.fitted = True
            return self
        
        # Prepare features (exclude target from features)
        X = data.select_dtypes(include=[np.number])
        
        # Determine number of features to select
        k = self.config.max_features or min(50, X.shape[1])
        
        # Initialize selector based on method
        if self.config.selection_method == "mutual_info":
            self.selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif self.config.selection_method == "f_classif":
            self.selector = SelectKBest(score_func=f_classif, k=k)
        else:
            self.selector = SelectKBest(score_func=mutual_info_classif, k=k)
        
        # Fit selector
        self.selector.fit(X, target)
        
        # Get selected feature names
        selected_indices = self.selector.get_support(indices=True)
        self.selected_features = X.columns[selected_indices].tolist()
        
        # Add non-numerical columns back
        non_numerical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
        self.selected_features.extend(non_numerical_cols)
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection"""
        if not self.fitted:
            raise ValueError("FeatureSelector must be fitted before transform")
        
        # Select only the chosen features
        available_features = [col for col in self.selected_features if col in data.columns]
        return data[available_features]

class FinancialDataPreprocessor:
    """
    Comprehensive financial data preprocessor for fraud detection
    
    Features:
    - Advanced data cleaning and outlier detection
    - Sophisticated feature engineering for fraud detection
    - Multiple scaling and encoding methods
    - Automated feature selection and dimensionality reduction
    - Enhanced validation and error handling
    - Integration with enhanced preprocessor when available
    - Transaction velocity and behavioral pattern features
    - Time-based and cyclical feature encoding
    - User and merchant aggregation features
    - Configurable preprocessing pipeline
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None, use_enhanced: bool = True):
        """
        Initialize financial data preprocessor
        
        Args:
            config: Preprocessing configuration
            use_enhanced: Whether to use enhanced preprocessor when available
        """
        self.config = config or PreprocessingConfig()
        self.preprocessing_steps: List[PreprocessingStep] = []
        self.fitted = False
        self.feature_names = []
        self.feature_types = {}
        self.preprocessing_stats = {
            'total_preprocessings': 0,
            'total_records_processed': 0,
            'average_processing_time': 0.0
        }
        self._lock = threading.RLock()
        
        # Enhanced preprocessor integration
        self.use_enhanced = use_enhanced and ENHANCED_AVAILABLE
        self._enhanced_preprocessor = None
        
        if self.use_enhanced:
            try:
                # Convert config to enhanced config if needed
                enhanced_config = self._convert_to_enhanced_config(self.config)
                self._enhanced_preprocessor = EnhancedFinancialDataPreprocessor(enhanced_config)
                logger.info("Using enhanced data preprocessor with comprehensive validation")
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced preprocessor: {e}, falling back to standard")
                self.use_enhanced = False
        
        # Initialize preprocessing pipeline (fallback)
        if not self.use_enhanced:
            self._initialize_pipeline()
        
        logger.info(f"FinancialDataPreprocessor initialized (enhanced: {self.use_enhanced})")
    
    def _initialize_pipeline(self):
        """Initialize the preprocessing pipeline"""
        # Add preprocessing steps based on configuration
        self.preprocessing_steps = [
            DataCleaner(self.config),
            FeatureEngineer(self.config),
            CategoricalEncoder(self.config),
            FeatureScaler(self.config),
            FeatureSelector(self.config)
        ]
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'FinancialDataPreprocessor':
        """
        Fit the preprocessing pipeline to the data
        
        Args:
            data: Input DataFrame
            target: Target variable for supervised feature selection
            
        Returns:
            Fitted preprocessor
        """
        # Use enhanced preprocessor if available
        if self.use_enhanced and self._enhanced_preprocessor:
            try:
                self._enhanced_preprocessor.fit(data, target)
                self.feature_names = self._enhanced_preprocessor.feature_names
                self.feature_types = self._enhanced_preprocessor.feature_types
                self.fitted = True
                logger.info(f"Enhanced preprocessor fitted successfully. Final feature count: {len(self.feature_names)}")
                return self
            except Exception as e:
                logger.error(f"Enhanced preprocessor fit failed: {e}, falling back to standard")
                self.use_enhanced = False
        
        # Standard preprocessing
        logger.info(f"Fitting preprocessor on {len(data)} records with {len(data.columns)} features")
        
        current_data = data.copy()
        
        # Fit each preprocessing step
        for step in self.preprocessing_steps:
            logger.debug(f"Fitting {step.step_name}")
            step.fit(current_data, target)
            current_data = step.transform(current_data)
        
        # Store final feature information
        self.feature_names = current_data.columns.tolist()
        self.feature_types = self._determine_feature_types(current_data)
        
        self.fitted = True
        logger.info(f"Preprocessor fitted successfully. Final feature count: {len(self.feature_names)}")
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessing pipeline
        
        Args:
            data: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Use enhanced preprocessor if available
        if self.use_enhanced and self._enhanced_preprocessor:
            try:
                result = self._enhanced_preprocessor.transform(data)
                self._update_stats(len(data), 0)  # Stats updated in enhanced
                logger.info(f"Enhanced preprocessor transformed {len(data)} records")
                return result
            except Exception as e:
                logger.error(f"Enhanced preprocessor transform failed: {e}, falling back to standard")
                self.use_enhanced = False
        
        # Standard preprocessing
        start_time = time.time()
        
        with self._lock:
            self.preprocessing_stats['total_preprocessings'] += 1
            self.preprocessing_stats['total_records_processed'] += len(data)
        
        logger.info(f"Transforming {len(data)} records")
        
        current_data = data.copy()
        applied_steps = []
        
        # Apply each preprocessing step
        for step in self.preprocessing_steps:
            logger.debug(f"Applying {step.step_name}")
            current_data = step.transform(current_data)
            applied_steps.append(step.step_name)
        
        processing_time = time.time() - start_time
        
        # Update statistics
        with self._lock:
            total_preprocessings = self.preprocessing_stats['total_preprocessings']
            self.preprocessing_stats['average_processing_time'] = (
                (self.preprocessing_stats['average_processing_time'] * (total_preprocessings - 1) + processing_time) 
                / total_preprocessings
            )
        
        logger.info(f"Preprocessing completed in {processing_time:.2f} seconds")
        return current_data
    
    def fit_transform(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform in one step
        
        Args:
            data: Input DataFrame
            target: Target variable for supervised feature selection
            
        Returns:
            Preprocessed DataFrame
        """
        return self.fit(data, target).transform(data)
    
    def get_preprocessing_result(self, data: pd.DataFrame, 
                               processed_data: pd.DataFrame,
                               processing_time: float) -> PreprocessingResult:
        """
        Create comprehensive preprocessing result
        
        Args:
            data: Original data
            processed_data: Preprocessed data
            processing_time: Time taken for preprocessing
            
        Returns:
            Preprocessing result with detailed information
        """
        # Get applied steps
        applied_steps = [step.step_name for step in self.preprocessing_steps if step.fitted]
        
        # Calculate feature importance if available
        feature_importance = {}
        for step in self.preprocessing_steps:
            if isinstance(step, FeatureSelector) and hasattr(step.selector, 'scores_'):
                scores = step.selector.scores_
                feature_names = step.selector.feature_names_in_
                feature_importance = dict(zip(feature_names, scores))
        
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
                    'categories': len(step.encoders.get(col, {}))
                } for col in step.encoded_columns}
        
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
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def _determine_feature_types(self, data: pd.DataFrame) -> Dict[str, FeatureType]:
        """Determine feature types for each column"""
        feature_types = {}
        
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                if data[col].nunique() == 2:
                    feature_types[col] = FeatureType.BINARY
                else:
                    feature_types[col] = FeatureType.NUMERICAL
            elif data[col].dtype == 'object':
                if 'time' in col.lower() or 'date' in col.lower():
                    feature_types[col] = FeatureType.TEMPORAL
                else:
                    feature_types[col] = FeatureType.CATEGORICAL
            elif data[col].dtype == 'category':
                feature_types[col] = FeatureType.CATEGORICAL
            else:
                feature_types[col] = FeatureType.TEXTUAL
        
        return feature_types
    
    def create_fraud_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create specialized fraud detection features
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with additional fraud-specific features
        """
        fraud_data = data.copy()
        
        # Risk score features
        if 'amount' in data.columns:
            # Amount-based risk indicators
            fraud_data['high_amount_risk'] = (fraud_data['amount'] > fraud_data['amount'].quantile(0.95)).astype(int)
            fraud_data['round_amount_risk'] = (fraud_data['amount'] % 100 == 0).astype(int)
            fraud_data['micro_amount_risk'] = (fraud_data['amount'] < 1.0).astype(int)
        
        # Time-based risk indicators
        if 'timestamp' in data.columns:
            timestamp = pd.to_datetime(fraud_data['timestamp'])
            fraud_data['night_transaction_risk'] = ((timestamp.dt.hour >= 23) | (timestamp.dt.hour <= 5)).astype(int)
            fraud_data['weekend_risk'] = timestamp.dt.dayofweek.isin([5, 6]).astype(int)
            fraud_data['holiday_risk'] = fraud_data.get('is_holiday_season', 0)
        
        # Velocity-based risk indicators
        if all(col in data.columns for col in ['user_id', 'timestamp']):
            # Sort by user and timestamp
            fraud_data = fraud_data.sort_values(['user_id', 'timestamp'])
            
            # Calculate transaction frequency
            fraud_data['transactions_per_day'] = fraud_data.groupby([
                'user_id', 
                pd.to_datetime(fraud_data['timestamp']).dt.date
            ]).cumcount() + 1
            
            fraud_data['high_frequency_risk'] = (fraud_data['transactions_per_day'] > 10).astype(int)
        
        # Geographic risk indicators
        if all(col in data.columns for col in ['user_id', 'location']):
            user_location_count = fraud_data.groupby('user_id')['location'].nunique()
            fraud_data['multi_location_risk'] = fraud_data['user_id'].map(
                user_location_count > 5
            ).astype(int)
        
        # Device/channel risk indicators
        if 'payment_method' in data.columns:
            risky_methods = ['digital_wallet', 'online']
            fraud_data['risky_payment_method'] = fraud_data['payment_method'].isin(risky_methods).astype(int)
        
        return fraud_data
    
    def _convert_to_enhanced_config(self, config: PreprocessingConfig) -> 'EnhancedPreprocessingConfig':
        """Convert standard config to enhanced config"""
        try:
            # Create enhanced config with compatible parameters
            enhanced_config = EnhancedPreprocessingConfig()
            
            # Copy compatible parameters
            for attr in ['handle_missing', 'missing_strategy', 'remove_duplicates', 
                        'outlier_detection', 'outlier_method', 'create_time_features',
                        'create_aggregation_features', 'create_ratio_features',
                        'create_lag_features', 'lag_periods', 'categorical_encoding',
                        'max_categories', 'feature_selection', 'selection_method',
                        'max_features', 'scaling_method', 'scale_features',
                        'dimensionality_reduction', 'pca_components', 
                        'create_interaction_features', 'polynomial_features',
                        'polynomial_degree']:
                if hasattr(config, attr) and hasattr(enhanced_config, attr):
                    setattr(enhanced_config, attr, getattr(config, attr))
            
            # Set enhanced-specific defaults with validation and security
            enhanced_config.validation_config = ValidationConfig(
                validate_inputs=True,
                validate_outputs=True,
                check_data_consistency=True,
                check_feature_ranges=True
            )
            enhanced_config.security_config = SecurityConfig(
                max_memory_mb=2048.0,
                max_execution_time_seconds=300.0,
                protect_sensitive_features=True
            )
            enhanced_config.enable_error_recovery = True
            enhanced_config.fallback_strategy = "partial"
            
            return enhanced_config
        except Exception as e:
            logger.error(f"Failed to convert config: {e}")
            return EnhancedPreprocessingConfig()
    
    def _update_stats(self, records_processed: int, processing_time: float):
        """Update processing statistics"""
        with self._lock:
            self.preprocessing_stats['total_preprocessings'] += 1
            self.preprocessing_stats['total_records_processed'] += records_processed
            if processing_time > 0:
                total = self.preprocessing_stats['total_preprocessings']
                avg_time = self.preprocessing_stats['average_processing_time']
                self.preprocessing_stats['average_processing_time'] = (
                    (avg_time * (total - 1) + processing_time) / total
                )
    
    def get_enhanced_features(self) -> Dict[str, Any]:
        """Get enhanced features if available"""
        if self.use_enhanced and self._enhanced_preprocessor:
            try:
                return self._enhanced_preprocessor.get_statistics()
            except Exception as e:
                logger.warning(f"Failed to get enhanced features: {e}")
        return {}
    
    def create_behavioral_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral pattern features
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with behavioral features
        """
        behavioral_data = data.copy()
        
        if not all(col in data.columns for col in ['user_id', 'timestamp', 'amount']):
            return behavioral_data
        
        # Sort by user and timestamp
        behavioral_data = behavioral_data.sort_values(['user_id', 'timestamp'])
        behavioral_data['timestamp'] = pd.to_datetime(behavioral_data['timestamp'])
        
        # User spending patterns
        user_groups = behavioral_data.groupby('user_id')
        
        # Spending consistency
        behavioral_data['user_amount_cv'] = user_groups['amount'].transform(
            lambda x: x.std() / (x.mean() + 1e-8)
        )
        
        # Transaction timing patterns
        behavioral_data['user_hour_consistency'] = user_groups['timestamp'].transform(
            lambda x: x.dt.hour.std()
        )
        
        # Spending velocity changes
        behavioral_data['amount_rolling_mean_7'] = user_groups['amount'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        behavioral_data['amount_deviation_from_trend'] = (
            behavioral_data['amount'] - behavioral_data['amount_rolling_mean_7']
        ) / (behavioral_data['amount_rolling_mean_7'] + 1e-8)
        
        # Day-of-week patterns
        behavioral_data['day_of_week'] = behavioral_data['timestamp'].dt.dayofweek
        user_dow_mode = user_groups['day_of_week'].transform(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0
        )
        behavioral_data['unusual_day_pattern'] = (
            behavioral_data['day_of_week'] != user_dow_mode
        ).astype(int)
        
        return behavioral_data
    
    def save_preprocessor(self, file_path: Path) -> None:
        """Save fitted preprocessor to file"""
        if not self.fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        save_data = {
            'config': self.config,
            'preprocessing_steps': self.preprocessing_steps,
            'feature_names': self.feature_names,
            'feature_types': self.feature_types,
            'fitted': self.fitted
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Preprocessor saved to {file_path}")
    
    def load_preprocessor(self, file_path: Path) -> 'FinancialDataPreprocessor':
        """Load fitted preprocessor from file"""
        with open(file_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.config = save_data['config']
        self.preprocessing_steps = save_data['preprocessing_steps']
        self.feature_names = save_data['feature_names']
        self.feature_types = save_data['feature_types']
        self.fitted = save_data['fitted']
        
        logger.info(f"Preprocessor loaded from {file_path}")
        return self
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        for step in self.preprocessing_steps:
            if isinstance(step, FeatureSelector) and hasattr(step.selector, 'scores_'):
                scores = step.selector.scores_
                feature_names = step.selector.feature_names_in_
                return dict(zip(feature_names, scores))
        return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        with self._lock:
            stats = self.preprocessing_stats.copy()
            stats.update({
                'fitted': self.fitted,
                'feature_count': len(self.feature_names),
                'feature_types': {ft.value: sum(1 for t in self.feature_types.values() if t == ft) 
                               for ft in FeatureType},
                'using_enhanced': self.use_enhanced,
                'enhanced_available': ENHANCED_AVAILABLE
            })
            
            # Add enhanced stats if available
            if self.use_enhanced and self._enhanced_preprocessor:
                try:
                    enhanced_stats = self._enhanced_preprocessor.get_statistics()
                    stats['enhanced_stats'] = enhanced_stats
                except Exception as e:
                    logger.warning(f"Failed to get enhanced stats: {e}")
            
            return stats
    
    def reset_statistics(self) -> None:
        """Reset preprocessing statistics"""
        with self._lock:
            self.preprocessing_stats = {
                'total_preprocessings': 0,
                'total_records_processed': 0,
                'average_processing_time': 0.0
            }
    
    def __repr__(self) -> str:
        return (f"FinancialDataPreprocessor(fitted={self.fitted}, "
               f"features={len(self.feature_names)}, "
               f"steps={len(self.preprocessing_steps)}, "
               f"processings={self.preprocessing_stats['total_preprocessings']})")

# Export main classes
__all__ = [
    'FinancialDataPreprocessor',
    'PreprocessingConfig',
    'PreprocessingResult',
    'FeatureType',
    'ScalingMethod',
    'PreprocessingStep',
    'DataCleaner',
    'FeatureEngineer',
    'FeatureScaler',
    'CategoricalEncoder',
    'FeatureSelector'
]

if __name__ == "__main__":
    # Example usage and testing
    
    # Create sample financial transaction data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'transaction_id': [f'TXN{i:06d}' for i in range(n_samples)],
        'user_id': [f'USER{np.random.randint(1, 100):03d}' for _ in range(n_samples)],
        'merchant_id': [f'MERCH{np.random.randint(1, 50):03d}' for _ in range(n_samples)],
        'amount': np.random.lognormal(mean=4, sigma=1.2, size=n_samples),
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='1h'),
        'location': [f'LOC{np.random.randint(1, 20):03d}' for _ in range(n_samples)],
        'payment_method': np.random.choice(['credit_card', 'debit_card', 'digital_wallet'], n_samples),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
    })
    
    # Create target variable
    target = sample_data['is_fraud']
    sample_data = sample_data.drop('is_fraud', axis=1)
    
    # Initialize preprocessor with custom config
    config = PreprocessingConfig(
        create_time_features=True,
        create_aggregation_features=True,
        create_ratio_features=True,
        feature_selection=True,
        max_features=30
    )
    
    preprocessor = FinancialDataPreprocessor(config)
    
    # Fit and transform
    print("Original data shape:", sample_data.shape)
    processed_data = preprocessor.fit_transform(sample_data, target)
    print("Processed data shape:", processed_data.shape)
    print("Feature count:", len(preprocessor.feature_names))
    
    # Add fraud-specific features
    fraud_features = preprocessor.create_fraud_features(processed_data)
    print("With fraud features shape:", fraud_features.shape)
    
    # Add behavioral features
    behavioral_features = preprocessor.create_behavioral_features(fraud_features)
    print("With behavioral features shape:", behavioral_features.shape)
    
    # Get feature importance
    importance = preprocessor.get_feature_importance()
    print(f"Feature importance calculated for {len(importance)} features")
    
    # Get statistics
    stats = preprocessor.get_statistics()
    print("Preprocessing statistics:", stats)
    
    print("\nFinancialDataPreprocessor ready for production use!")