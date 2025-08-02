#!/usr/bin/env python3
"""
Data Processing Utilities for Universal AI Core
===============================================

This module provides comprehensive data processing utilities adapted from Saraphis patterns.
Extracted from molecular_analyzer.py and property_predictor.py, these utilities offer
domain-agnostic data processing, transformation, and feature engineering capabilities.

Features:
- Generic feature extraction pipeline with pluggable extractors
- Data preprocessing and transformation utilities
- Batch processing with memory optimization
- Data validation and quality checks
- Type conversion and normalization utilities
- Missing data handling strategies
- Data splitting and sampling utilities
- Performance-optimized data operations
"""

import logging
import time
import warnings
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import threading
from collections import defaultdict, deque
import pickle
import gzip
import json

try:
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import scipy.sparse as sp
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DataProcessingConfig:
    """Configuration for data processing operations"""
    batch_size: int = 1000
    n_jobs: int = -1
    memory_limit_mb: int = 2048
    cache_enabled: bool = True
    cache_size: int = 10000
    chunk_size: int = 10000
    parallel_processing: bool = True
    handle_missing_data: str = "impute"  # drop, impute, flag
    scaling_method: str = "standard"  # standard, robust, minmax, none
    feature_selection: bool = True
    max_features: Optional[int] = None
    validation_enabled: bool = True
    random_state: int = 42


@dataclass
class ProcessingResult:
    """Result of data processing operation"""
    data: Union[np.ndarray, pd.DataFrame]
    metadata: Dict[str, Any]
    processing_time: float
    status: str = "success"
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)


class FeatureExtractorInterface(ABC):
    """Abstract interface for feature extractors adapted from Saraphis patterns"""
    
    @abstractmethod
    def extract_features(self, data: Any) -> np.ndarray:
        """Extract features from input data"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features"""
        pass
    
    @abstractmethod
    def get_feature_types(self) -> List[str]:
        """Get types of extracted features"""
        pass


class NumericFeatureExtractor(FeatureExtractorInterface):
    """
    Numeric feature extractor adapted from molecular descriptor patterns.
    Extracts statistical and mathematical features from numeric data.
    """
    
    def __init__(self, config: Optional[DataProcessingConfig] = None):
        self.config = config or DataProcessingConfig()
        self.feature_cache = {} if self.config.cache_enabled else None
        self._lock = threading.Lock()
    
    def extract_features(self, data: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Extract numeric statistical features"""
        try:
            # Convert to numpy array
            if isinstance(data, list):
                data = np.array(data)
            
            # Check cache
            if self.feature_cache is not None:
                cache_key = hash(data.tobytes())
                with self._lock:
                    if cache_key in self.feature_cache:
                        return self.feature_cache[cache_key]
            
            # Handle missing values
            data_clean = data[~np.isnan(data)] if len(data.shape) == 1 else data
            
            if len(data_clean) == 0:
                return np.zeros(self._get_feature_count())
            
            # Extract features
            features = {}
            
            # Basic statistics
            features.update({
                'mean': np.mean(data_clean),
                'std': np.std(data_clean),
                'var': np.var(data_clean),
                'min': np.min(data_clean),
                'max': np.max(data_clean),
                'median': np.median(data_clean),
                'range': np.ptp(data_clean),
                'sum': np.sum(data_clean),
                'count': len(data_clean)
            })
            
            # Percentiles
            percentiles = [10, 25, 75, 90, 95, 99]
            for p in percentiles:
                features[f'percentile_{p}'] = np.percentile(data_clean, p)
            
            # Distribution statistics
            if SCIPY_AVAILABLE and len(data_clean) > 3:
                features.update({
                    'skewness': stats.skew(data_clean),
                    'kurtosis': stats.kurtosis(data_clean),
                    'iqr': np.percentile(data_clean, 75) - np.percentile(data_clean, 25)
                })
            else:
                features.update({'skewness': 0.0, 'kurtosis': 0.0, 'iqr': 0.0})
            
            # Advanced features
            if len(data_clean) > 1:
                features.update({
                    'coeff_variation': np.std(data_clean) / np.mean(data_clean) if np.mean(data_clean) != 0 else 0,
                    'mean_abs_deviation': np.mean(np.abs(data_clean - np.mean(data_clean))),
                    'median_abs_deviation': np.median(np.abs(data_clean - np.median(data_clean)))
                })
            else:
                features.update({'coeff_variation': 0.0, 'mean_abs_deviation': 0.0, 'median_abs_deviation': 0.0})
            
            # Convert to array
            feature_array = np.array(list(features.values()))
            
            # Cache result
            if self.feature_cache is not None:
                with self._lock:
                    if len(self.feature_cache) < self.config.cache_size:
                        self.feature_cache[cache_key] = feature_array
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Numeric feature extraction failed: {e}")
            return np.zeros(self._get_feature_count())
    
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features"""
        base_features = ['mean', 'std', 'var', 'min', 'max', 'median', 'range', 'sum', 'count']
        percentile_features = [f'percentile_{p}' for p in [10, 25, 75, 90, 95, 99]]
        distribution_features = ['skewness', 'kurtosis', 'iqr']
        advanced_features = ['coeff_variation', 'mean_abs_deviation', 'median_abs_deviation']
        
        return base_features + percentile_features + distribution_features + advanced_features
    
    def get_feature_types(self) -> List[str]:
        """Get types of extracted features"""
        return ['numerical'] * self._get_feature_count()
    
    def _get_feature_count(self) -> int:
        """Get total number of features"""
        return len(self.get_feature_names())


class TextFeatureExtractor(FeatureExtractorInterface):
    """
    Text feature extractor adapted from Saraphis patterns.
    Extracts linguistic and statistical features from text data.
    """
    
    def __init__(self, config: Optional[DataProcessingConfig] = None):
        self.config = config or DataProcessingConfig()
        self.feature_cache = {} if self.config.cache_enabled else None
        self._lock = threading.Lock()
    
    def extract_features(self, data: str) -> np.ndarray:
        """Extract text features"""
        try:
            # Check cache
            if self.feature_cache is not None:
                cache_key = hash(data)
                with self._lock:
                    if cache_key in self.feature_cache:
                        return self.feature_cache[cache_key]
            
            # Handle empty text
            if not data or not isinstance(data, str):
                return np.zeros(self._get_feature_count())
            
            # Extract features
            features = {}
            
            # Basic text statistics
            words = data.split()
            sentences = [s.strip() for s in data.split('.') if s.strip()]
            
            features.update({
                'char_count': len(data),
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
                'avg_sentence_length': np.mean([len(sent.split()) for sent in sentences]) if sentences else 0,
                'unique_words': len(set(words)),
                'vocabulary_richness': len(set(words)) / len(words) if words else 0
            })
            
            # Character-level features
            features.update({
                'uppercase_ratio': sum(1 for c in data if c.isupper()) / len(data) if data else 0,
                'lowercase_ratio': sum(1 for c in data if c.islower()) / len(data) if data else 0,
                'digit_ratio': sum(1 for c in data if c.isdigit()) / len(data) if data else 0,
                'space_ratio': sum(1 for c in data if c.isspace()) / len(data) if data else 0,
                'punctuation_ratio': sum(1 for c in data if not c.isalnum() and not c.isspace()) / len(data) if data else 0
            })
            
            # Linguistic features
            vowels = 'aeiouAEIOU'
            features.update({
                'vowel_ratio': sum(1 for c in data if c in vowels) / len(data) if data else 0,
                'consonant_ratio': sum(1 for c in data if c.isalpha() and c not in vowels) / len(data) if data else 0
            })
            
            # Word frequency features
            if words:
                word_freq = {}
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                
                max_freq = max(word_freq.values())
                features.update({
                    'max_word_freq': max_freq,
                    'max_word_freq_ratio': max_freq / len(words),
                    'repeated_words_ratio': (len(words) - len(set(words))) / len(words)
                })
            else:
                features.update({
                    'max_word_freq': 0,
                    'max_word_freq_ratio': 0,
                    'repeated_words_ratio': 0
                })
            
            # Convert to array
            feature_array = np.array(list(features.values()))
            
            # Cache result
            if self.feature_cache is not None:
                with self._lock:
                    if len(self.feature_cache) < self.config.cache_size:
                        self.feature_cache[cache_key] = feature_array
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Text feature extraction failed: {e}")
            return np.zeros(self._get_feature_count())
    
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features"""
        return [
            'char_count', 'word_count', 'sentence_count', 'avg_word_length', 'avg_sentence_length',
            'unique_words', 'vocabulary_richness', 'uppercase_ratio', 'lowercase_ratio',
            'digit_ratio', 'space_ratio', 'punctuation_ratio', 'vowel_ratio', 'consonant_ratio',
            'max_word_freq', 'max_word_freq_ratio', 'repeated_words_ratio'
        ]
    
    def get_feature_types(self) -> List[str]:
        """Get types of extracted features"""
        return ['numerical'] * self._get_feature_count()
    
    def _get_feature_count(self) -> int:
        """Get total number of features"""
        return len(self.get_feature_names())


class DataProcessor:
    """
    Universal data processor adapted from Saraphis molecular analyzer patterns.
    Provides comprehensive data processing pipeline with pluggable components.
    """
    
    def __init__(self, config: Optional[DataProcessingConfig] = None):
        self.config = config or DataProcessingConfig()
        self.extractors: Dict[str, FeatureExtractorInterface] = {}
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.imputers: Dict[str, Any] = {}
        self.feature_selectors: Dict[str, Any] = {}
        self.processing_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
        # Initialize default extractors
        self.register_extractor('numeric', NumericFeatureExtractor(config))
        self.register_extractor('text', TextFeatureExtractor(config))
        
        logger.info("Data processor initialized with default extractors")
    
    def register_extractor(self, name: str, extractor: FeatureExtractorInterface):
        """Register a feature extractor"""
        self.extractors[name] = extractor
        logger.info(f"Registered feature extractor: {name}")
    
    def prepare_features(self, data: Union[pd.DataFrame, np.ndarray, List], 
                        extractor_names: Optional[List[str]] = None) -> ProcessingResult:
        """
        Prepare features from input data using specified extractors.
        Adapted from Saraphis _prepare_features patterns.
        """
        start_time = time.time()
        
        try:
            # Determine extractors to use
            if extractor_names is None:
                extractor_names = list(self.extractors.keys())
            
            # Process data based on type
            if isinstance(data, pd.DataFrame):
                return self._process_dataframe(data, extractor_names, start_time)
            elif isinstance(data, (np.ndarray, list)):
                return self._process_array(data, extractor_names, start_time)
            else:
                return ProcessingResult(
                    data=np.array([]),
                    metadata={},
                    processing_time=time.time() - start_time,
                    status="error",
                    error_message=f"Unsupported data type: {type(data)}"
                )
                
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return ProcessingResult(
                data=np.array([]),
                metadata={},
                processing_time=time.time() - start_time,
                status="error",
                error_message=str(e)
            )
    
    def _process_dataframe(self, df: pd.DataFrame, extractor_names: List[str], 
                          start_time: float) -> ProcessingResult:
        """Process pandas DataFrame"""
        features_list = []
        feature_names = []
        warnings_list = []
        statistics = {}
        
        # Process each column based on its type
        for column in df.columns:
            try:
                col_data = df[column]
                col_dtype = col_data.dtype
                
                # Determine appropriate extractor
                if pd.api.types.is_numeric_dtype(col_dtype) and 'numeric' in extractor_names:
                    extractor = self.extractors['numeric']
                    features = extractor.extract_features(col_data.values)
                    names = [f"{column}_{name}" for name in extractor.get_feature_names()]
                    
                elif pd.api.types.is_string_dtype(col_dtype) and 'text' in extractor_names:
                    # Process text column (concatenate all text)
                    text_data = ' '.join(col_data.astype(str).values)
                    extractor = self.extractors['text']
                    features = extractor.extract_features(text_data)
                    names = [f"{column}_{name}" for name in extractor.get_feature_names()]
                    
                else:
                    # Skip unsupported column types
                    warnings_list.append(f"Skipped column {column} with type {col_dtype}")
                    continue
                
                features_list.append(features)
                feature_names.extend(names)
                
                # Collect statistics
                statistics[column] = {
                    'type': str(col_dtype),
                    'feature_count': len(features),
                    'null_count': col_data.isnull().sum(),
                    'unique_count': col_data.nunique()
                }
                
            except Exception as e:
                logger.error(f"Error processing column {column}: {e}")
                warnings_list.append(f"Failed to process column {column}: {str(e)}")
        
        # Combine features
        if features_list:
            combined_features = np.column_stack(features_list)
        else:
            combined_features = np.array([]).reshape(len(df), 0)
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            data=combined_features,
            metadata={
                'feature_names': feature_names,
                'feature_count': len(feature_names),
                'sample_count': len(df),
                'extractors_used': extractor_names,
                'original_columns': list(df.columns)
            },
            processing_time=processing_time,
            warnings=warnings_list,
            statistics=statistics
        )
    
    def _process_array(self, data: Union[np.ndarray, List], extractor_names: List[str], 
                      start_time: float) -> ProcessingResult:
        """Process numpy array or list"""
        # Convert to numpy array
        if isinstance(data, list):
            data = np.array(data)
        
        features_list = []
        feature_names = []
        warnings_list = []
        
        # Determine data type and process accordingly
        if data.dtype.kind in ['i', 'f']:  # Numeric data
            if 'numeric' in extractor_names:
                extractor = self.extractors['numeric']
                if len(data.shape) == 1:
                    # 1D array - treat as single feature vector
                    features = extractor.extract_features(data)
                    features_list.append(features)
                    feature_names.extend(extractor.get_feature_names())
                else:
                    # 2D array - process each column
                    for i in range(data.shape[1]):
                        features = extractor.extract_features(data[:, i])
                        features_list.append(features)
                        names = [f"col_{i}_{name}" for name in extractor.get_feature_names()]
                        feature_names.extend(names)
        
        elif data.dtype.kind in ['U', 'S', 'O']:  # String/object data
            if 'text' in extractor_names:
                extractor = self.extractors['text']
                # Concatenate all text
                text_data = ' '.join(str(item) for item in data.flat)
                features = extractor.extract_features(text_data)
                features_list.append(features)
                feature_names.extend(extractor.get_feature_names())
        
        # Combine features
        if features_list:
            if len(features_list) == 1:
                combined_features = features_list[0].reshape(1, -1)
            else:
                combined_features = np.column_stack(features_list)
        else:
            combined_features = np.array([]).reshape(1, 0)
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            data=combined_features,
            metadata={
                'feature_names': feature_names,
                'feature_count': len(feature_names),
                'sample_count': len(combined_features),
                'extractors_used': extractor_names,
                'original_shape': data.shape,
                'original_dtype': str(data.dtype)
            },
            processing_time=processing_time,
            warnings=warnings_list,
            statistics={'data_type': str(data.dtype), 'shape': data.shape}
        )
    
    def scale_features(self, data: np.ndarray, method: str = "standard", 
                      fit_scaler: bool = True, scaler_key: str = "default") -> ProcessingResult:
        """
        Scale features using specified method.
        Adapted from Saraphis scaling patterns.
        """
        start_time = time.time()
        
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("Scikit-learn not available, returning original data")
                return ProcessingResult(
                    data=data,
                    metadata={'scaling_method': 'none', 'scaler_fitted': False},
                    processing_time=time.time() - start_time
                )
            
            # Get or create scaler
            if fit_scaler or scaler_key not in self.scalers:
                if method == "standard":
                    scaler = StandardScaler()
                elif method == "robust":
                    scaler = RobustScaler()
                elif method == "minmax":
                    scaler = MinMaxScaler()
                else:
                    return ProcessingResult(
                        data=data,
                        metadata={'scaling_method': method},
                        processing_time=time.time() - start_time,
                        status="error",
                        error_message=f"Unknown scaling method: {method}"
                    )
                
                # Fit scaler
                scaler.fit(data)
                self.scalers[scaler_key] = scaler
                scaler_fitted = True
            else:
                scaler = self.scalers[scaler_key]
                scaler_fitted = False
            
            # Transform data
            scaled_data = scaler.transform(data)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                data=scaled_data,
                metadata={
                    'scaling_method': method,
                    'scaler_key': scaler_key,
                    'scaler_fitted': scaler_fitted,
                    'original_shape': data.shape,
                    'scaled_shape': scaled_data.shape
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Feature scaling failed: {e}")
            return ProcessingResult(
                data=data,
                metadata={},
                processing_time=time.time() - start_time,
                status="error",
                error_message=str(e)
            )
    
    def handle_missing_data(self, data: Union[pd.DataFrame, np.ndarray], 
                           strategy: str = "impute", imputer_key: str = "default") -> ProcessingResult:
        """
        Handle missing data using specified strategy.
        Adapted from Saraphis missing data handling patterns.
        """
        start_time = time.time()
        
        try:
            if isinstance(data, pd.DataFrame):
                return self._handle_missing_dataframe(data, strategy, imputer_key, start_time)
            elif isinstance(data, np.ndarray):
                return self._handle_missing_array(data, strategy, imputer_key, start_time)
            else:
                return ProcessingResult(
                    data=data,
                    metadata={},
                    processing_time=time.time() - start_time,
                    status="error",
                    error_message=f"Unsupported data type: {type(data)}"
                )
                
        except Exception as e:
            logger.error(f"Missing data handling failed: {e}")
            return ProcessingResult(
                data=data,
                metadata={},
                processing_time=time.time() - start_time,
                status="error",
                error_message=str(e)
            )
    
    def _handle_missing_dataframe(self, df: pd.DataFrame, strategy: str, 
                                 imputer_key: str, start_time: float) -> ProcessingResult:
        """Handle missing data in DataFrame"""
        original_shape = df.shape
        missing_counts = df.isnull().sum()
        
        if strategy == "drop":
            # Drop rows with any missing values
            cleaned_df = df.dropna()
            
        elif strategy == "impute":
            if not SKLEARN_AVAILABLE:
                # Simple imputation without sklearn
                cleaned_df = df.fillna(df.mean(numeric_only=True))
                cleaned_df = cleaned_df.fillna(df.mode().iloc[0])  # For non-numeric
            else:
                # Use sklearn imputers
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
                
                cleaned_df = df.copy()
                
                # Impute numeric columns
                if len(numeric_columns) > 0:
                    if imputer_key not in self.imputers:
                        self.imputers[imputer_key] = SimpleImputer(strategy='mean')
                        self.imputers[imputer_key].fit(df[numeric_columns])
                    
                    imputed_numeric = self.imputers[imputer_key].transform(df[numeric_columns])
                    cleaned_df[numeric_columns] = imputed_numeric
                
                # Impute non-numeric columns
                if len(non_numeric_columns) > 0:
                    for col in non_numeric_columns:
                        mode_value = df[col].mode()
                        if len(mode_value) > 0:
                            cleaned_df[col] = df[col].fillna(mode_value.iloc[0])
        
        elif strategy == "flag":
            # Add missing value flags and impute
            cleaned_df = df.copy()
            for col in df.columns:
                if df[col].isnull().any():
                    cleaned_df[f"{col}_missing"] = df[col].isnull().astype(int)
            
            # Simple imputation after flagging
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
        
        else:
            cleaned_df = df
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            data=cleaned_df,
            metadata={
                'strategy': strategy,
                'original_shape': original_shape,
                'final_shape': cleaned_df.shape,
                'missing_counts': missing_counts.to_dict(),
                'rows_dropped': original_shape[0] - cleaned_df.shape[0] if strategy == "drop" else 0
            },
            processing_time=processing_time
        )
    
    def _handle_missing_array(self, data: np.ndarray, strategy: str, 
                             imputer_key: str, start_time: float) -> ProcessingResult:
        """Handle missing data in numpy array"""
        original_shape = data.shape
        
        # Check for missing values
        if data.dtype.kind in ['f', 'c']:  # Float or complex
            missing_mask = np.isnan(data)
        else:
            # For other types, assume no missing values
            missing_mask = np.zeros_like(data, dtype=bool)
        
        missing_count = np.sum(missing_mask)
        
        if missing_count == 0:
            # No missing values
            return ProcessingResult(
                data=data,
                metadata={
                    'strategy': strategy,
                    'missing_count': 0,
                    'original_shape': original_shape
                },
                processing_time=time.time() - start_time
            )
        
        if strategy == "drop":
            # Drop rows with any missing values
            if len(data.shape) == 1:
                cleaned_data = data[~missing_mask]
            else:
                row_has_missing = np.any(missing_mask, axis=1)
                cleaned_data = data[~row_has_missing]
                
        elif strategy == "impute":
            cleaned_data = data.copy()
            if SKLEARN_AVAILABLE and data.dtype.kind in ['f', 'i']:
                # Use sklearn imputer for numeric data
                if imputer_key not in self.imputers:
                    self.imputers[imputer_key] = SimpleImputer(strategy='mean')
                    if len(data.shape) == 1:
                        self.imputers[imputer_key].fit(data.reshape(-1, 1))
                    else:
                        self.imputers[imputer_key].fit(data)
                
                if len(data.shape) == 1:
                    imputed = self.imputers[imputer_key].transform(data.reshape(-1, 1))
                    cleaned_data = imputed.flatten()
                else:
                    cleaned_data = self.imputers[imputer_key].transform(data)
            else:
                # Simple mean imputation
                if len(data.shape) == 1:
                    mean_val = np.nanmean(data)
                    cleaned_data[missing_mask] = mean_val
                else:
                    for col in range(data.shape[1]):
                        col_mean = np.nanmean(data[:, col])
                        cleaned_data[missing_mask[:, col], col] = col_mean
        else:
            cleaned_data = data
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            data=cleaned_data,
            metadata={
                'strategy': strategy,
                'original_shape': original_shape,
                'final_shape': cleaned_data.shape,
                'missing_count': missing_count,
                'rows_dropped': original_shape[0] - cleaned_data.shape[0] if strategy == "drop" and len(original_shape) > 1 else 0
            },
            processing_time=processing_time
        )
    
    def split_data(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                   test_size: float = 0.2, validation_size: float = 0.1, 
                   stratify: bool = False) -> Dict[str, np.ndarray]:
        """
        Split data into train/validation/test sets.
        Adapted from Saraphis data splitting patterns.
        """
        try:
            if not SKLEARN_AVAILABLE:
                # Simple splitting without sklearn
                n_samples = len(X)
                n_test = int(n_samples * test_size)
                n_val = int(n_samples * validation_size)
                n_train = n_samples - n_test - n_val
                
                indices = np.random.RandomState(self.config.random_state).permutation(n_samples)
                
                train_idx = indices[:n_train]
                val_idx = indices[n_train:n_train + n_val]
                test_idx = indices[n_train + n_val:]
                
                result = {
                    'X_train': X[train_idx],
                    'X_val': X[val_idx],
                    'X_test': X[test_idx]
                }
                
                if y is not None:
                    result.update({
                        'y_train': y[train_idx],
                        'y_val': y[val_idx],
                        'y_test': y[test_idx]
                    })
                
                return result
            
            # Use sklearn for sophisticated splitting
            if y is not None and stratify:
                # Stratified split
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=self.config.random_state, stratify=y
                )
                
                if validation_size > 0:
                    val_size_adjusted = validation_size / (1 - test_size)
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_temp, y_temp, test_size=val_size_adjusted, 
                        random_state=self.config.random_state, stratify=y_temp
                    )
                else:
                    X_train, X_val, y_train, y_val = X_temp, np.array([]), y_temp, np.array([])
            else:
                # Regular split
                if y is not None:
                    X_temp, X_test, y_temp, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=self.config.random_state
                    )
                else:
                    X_temp, X_test = train_test_split(
                        X, test_size=test_size, random_state=self.config.random_state
                    )
                    y_temp, y_test = None, None
                
                if validation_size > 0:
                    val_size_adjusted = validation_size / (1 - test_size)
                    if y_temp is not None:
                        X_train, X_val, y_train, y_val = train_test_split(
                            X_temp, y_temp, test_size=val_size_adjusted, 
                            random_state=self.config.random_state
                        )
                    else:
                        X_train, X_val = train_test_split(
                            X_temp, test_size=val_size_adjusted, 
                            random_state=self.config.random_state
                        )
                        y_train, y_val = None, None
                else:
                    X_train, X_val = X_temp, np.array([])
                    y_train, y_val = y_temp, np.array([]) if y_temp is not None else None
            
            result = {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test
            }
            
            if y is not None:
                result.update({
                    'y_train': y_train,
                    'y_val': y_val,
                    'y_test': y_test
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Data splitting failed: {e}")
            return {'X_train': X}
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing operations"""
        return {
            'registered_extractors': list(self.extractors.keys()),
            'fitted_scalers': list(self.scalers.keys()),
            'fitted_imputers': list(self.imputers.keys()),
            'processing_operations': len(self.processing_history),
            'config': self.config.__dict__
        }


# Utility functions adapted from Saraphis patterns

def batch_process_data(data: Union[pd.DataFrame, np.ndarray], 
                      processor_func: Callable, 
                      batch_size: int = 1000,
                      **kwargs) -> List[Any]:
    """
    Process data in batches for memory efficiency.
    Adapted from Saraphis batch processing patterns.
    """
    results = []
    
    if isinstance(data, pd.DataFrame):
        n_samples = len(data)
        for i in range(0, n_samples, batch_size):
            batch = data.iloc[i:i + batch_size]
            result = processor_func(batch, **kwargs)
            results.append(result)
    elif isinstance(data, np.ndarray):
        n_samples = len(data)
        for i in range(0, n_samples, batch_size):
            batch = data[i:i + batch_size]
            result = processor_func(batch, **kwargs)
            results.append(result)
    else:
        # For other iterables
        batch = []
        for item in data:
            batch.append(item)
            if len(batch) >= batch_size:
                result = processor_func(batch, **kwargs)
                results.append(result)
                batch = []
        
        # Process remaining items
        if batch:
            result = processor_func(batch, **kwargs)
            results.append(result)
    
    return results


def normalize_data_types(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize data types for consistent processing.
    Adapted from Saraphis data type handling patterns.
    """
    normalized_df = data.copy()
    
    for column in normalized_df.columns:
        col_data = normalized_df[column]
        
        # Handle object columns that might be numeric
        if col_data.dtype == 'object':
            # Try to convert to numeric
            try:
                numeric_data = pd.to_numeric(col_data, errors='coerce')
                if not numeric_data.isnull().all():
                    normalized_df[column] = numeric_data
            except:
                pass
        
        # Handle boolean columns
        elif col_data.dtype == 'bool':
            normalized_df[column] = col_data.astype(int)
        
        # Handle datetime columns
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            # Convert to timestamp
            normalized_df[column] = col_data.astype('int64') // 10**9
    
    return normalized_df


def detect_data_quality_issues(data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
    """
    Detect data quality issues.
    Adapted from Saraphis data validation patterns.
    """
    issues = {
        'missing_data': {},
        'duplicates': {},
        'outliers': {},
        'inconsistencies': {},
        'overall_score': 1.0
    }
    
    try:
        if isinstance(data, pd.DataFrame):
            # Missing data analysis
            missing_counts = data.isnull().sum()
            issues['missing_data'] = {
                'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
                'total_missing_cells': missing_counts.sum(),
                'missing_percentage': (missing_counts.sum() / data.size) * 100
            }
            
            # Duplicate analysis
            duplicate_rows = data.duplicated().sum()
            issues['duplicates'] = {
                'duplicate_rows': duplicate_rows,
                'duplicate_percentage': (duplicate_rows / len(data)) * 100
            }
            
            # Outlier detection for numeric columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            outlier_info = {}
            
            for col in numeric_columns:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    outlier_info[col] = {
                        'outlier_count': len(outliers),
                        'outlier_percentage': (len(outliers) / len(col_data)) * 100
                    }
            
            issues['outliers'] = outlier_info
            
        elif isinstance(data, np.ndarray):
            # Basic analysis for numpy arrays
            if data.dtype.kind in ['f', 'c']:
                missing_count = np.sum(np.isnan(data))
                issues['missing_data'] = {
                    'missing_count': missing_count,
                    'missing_percentage': (missing_count / data.size) * 100
                }
            
            # Outlier detection for numeric arrays
            if data.dtype.kind in ['f', 'i']:
                flat_data = data.flatten()
                valid_data = flat_data[~np.isnan(flat_data)] if data.dtype.kind == 'f' else flat_data
                
                if len(valid_data) > 0:
                    Q1 = np.percentile(valid_data, 25)
                    Q3 = np.percentile(valid_data, 75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]
                    issues['outliers'] = {
                        'outlier_count': len(outliers),
                        'outlier_percentage': (len(outliers) / len(valid_data)) * 100
                    }
        
        # Calculate overall quality score
        quality_score = 1.0
        
        # Penalize for missing data
        missing_penalty = min(0.5, issues['missing_data'].get('missing_percentage', 0) / 100)
        quality_score -= missing_penalty
        
        # Penalize for duplicates
        if 'duplicate_percentage' in issues['duplicates']:
            duplicate_penalty = min(0.3, issues['duplicates']['duplicate_percentage'] / 100)
            quality_score -= duplicate_penalty
        
        # Penalize for excessive outliers
        if issues['outliers']:
            avg_outlier_pct = np.mean([info.get('outlier_percentage', 0) for info in issues['outliers'].values()])
            outlier_penalty = min(0.2, avg_outlier_pct / 100)
            quality_score -= outlier_penalty
        
        issues['overall_score'] = max(0.0, quality_score)
        
    except Exception as e:
        logger.error(f"Data quality analysis failed: {e}")
        issues['error'] = str(e)
    
    return issues


# Export public API
__all__ = [
    'DataProcessingConfig', 'ProcessingResult', 'FeatureExtractorInterface',
    'NumericFeatureExtractor', 'TextFeatureExtractor', 'DataProcessor',
    'batch_process_data', 'normalize_data_types', 'detect_data_quality_issues'
]