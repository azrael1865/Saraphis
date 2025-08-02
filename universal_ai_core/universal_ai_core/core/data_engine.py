#!/usr/bin/env python3
"""
Universal AI Core Data Processing Engine
=======================================

This module provides comprehensive data processing capabilities for the Universal AI Core system.
Extracted and adapted from the Saraphis data processing system, made domain-agnostic while preserving
all sophisticated data handling capabilities.

Features:
- Multi-format data ingestion and processing
- Feature engineering and transformation pipeline
- Data validation and sanitization
- Batch processing with memory optimization
- Data serialization and persistence
- Async processing with worker pools
- Comprehensive caching and optimization
- Error handling and recovery mechanisms
"""

import asyncio
import json
import logging
import hashlib
import time
import threading
import pickle
import gzip
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import weakref
import gc
import psutil
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class DataFormat(Enum):
    """Supported data formats"""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    PICKLE = "pickle"
    NUMPY = "numpy"
    FEATHER = "feather"
    EXCEL = "excel"


class ProcessingStatus(Enum):
    """Data processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"
    TIMEOUT = "timeout"


class ValidationLevel(Enum):
    """Data validation levels"""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    CUSTOM = "custom"


@dataclass
class DataSchema:
    """Data schema definition"""
    columns: Dict[str, str]  # column_name -> data_type
    required_columns: List[str] = field(default_factory=list)
    optional_columns: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data against schema"""
        errors = []
        
        # Check required columns
        missing_cols = set(self.required_columns) - set(data.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check data types
        for col, expected_type in self.columns.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                if not self._is_compatible_type(actual_type, expected_type):
                    errors.append(f"Column {col} has type {actual_type}, expected {expected_type}")
        
        # Check constraints
        for col, constraint in self.constraints.items():
            if col in data.columns:
                if not self._validate_constraint(data[col], constraint):
                    errors.append(f"Column {col} violates constraint: {constraint}")
        
        return len(errors) == 0, errors
    
    def _is_compatible_type(self, actual: str, expected: str) -> bool:
        """Check if data types are compatible"""
        type_mapping = {
            'object': ['string', 'text', 'categorical'],
            'int64': ['integer', 'int', 'numeric'],
            'float64': ['float', 'numeric', 'decimal'],
            'bool': ['boolean', 'bool'],
            'datetime64': ['datetime', 'timestamp']
        }
        
        actual_base = actual.split('[')[0]  # Remove dtype specifics
        compatible_types = type_mapping.get(actual_base, [actual_base])
        return expected.lower() in compatible_types
    
    def _validate_constraint(self, series: pd.Series, constraint: Dict[str, Any]) -> bool:
        """Validate series against constraint"""
        try:
            if 'min' in constraint:
                if series.min() < constraint['min']:
                    return False
            if 'max' in constraint:
                if series.max() > constraint['max']:
                    return False
            if 'values' in constraint:
                if not series.isin(constraint['values']).all():
                    return False
            return True
        except Exception:
            return False


@dataclass
class ProcessingResult:
    """Data processing result container"""
    data: Optional[pd.DataFrame]
    status: ProcessingStatus
    processing_time: float
    memory_used: float
    records_processed: int
    features_generated: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    cache_key: Optional[str] = None


class DataProcessor(ABC):
    """Abstract base class for data processors"""
    
    @abstractmethod
    async def process(self, data: pd.DataFrame, **kwargs) -> ProcessingResult:
        """Process data and return result"""
        pass
    
    @abstractmethod
    def validate_input(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate input data"""
        pass


class FeatureEngineer:
    """
    Feature engineering framework for data transformation.
    
    Extracted and adapted from molecular_analyzer.py data processing patterns,
    made domain-agnostic while preserving sophisticated capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.transformers = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_cache = {}
        
        # Initialize transformers
        self._initialize_transformers()
        
    def _initialize_transformers(self):
        """Initialize feature transformation functions"""
        self.transformers = {
            'normalize': self._normalize_features,
            'standardize': self._standardize_features,
            'min_max_scale': self._min_max_scale,
            'robust_scale': self._robust_scale,
            'log_transform': self._log_transform,
            'polynomial': self._polynomial_features,
            'interaction': self._interaction_features,
            'binning': self._bin_features,
            'one_hot': self._one_hot_encode,
            'label_encode': self._label_encode,
            'target_encode': self._target_encode
        }
    
    async def engineer_features(self, data: pd.DataFrame, 
                              transformations: List[str],
                              target_column: Optional[str] = None) -> ProcessingResult:
        """Apply feature engineering transformations"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            logger.info(f"🔧 Starting feature engineering with {len(transformations)} transformations")
            
            processed_data = data.copy()
            generated_features = 0
            warnings_list = []
            
            for transformation in transformations:
                if transformation in self.transformers:
                    try:
                        transformer_func = self.transformers[transformation]
                        
                        if transformation in ['target_encode'] and target_column:
                            processed_data = transformer_func(processed_data, target_column)
                        else:
                            processed_data = transformer_func(processed_data)
                        
                        generated_features += len(processed_data.columns) - len(data.columns)
                        logger.debug(f"✅ Applied transformation: {transformation}")
                        
                    except Exception as e:
                        warning_msg = f"Failed to apply transformation {transformation}: {e}"
                        warnings_list.append(warning_msg)
                        logger.warning(warning_msg)
                else:
                    warning_msg = f"Unknown transformation: {transformation}"
                    warnings_list.append(warning_msg)
            
            processing_time = time.time() - start_time
            memory_used = self._get_memory_usage() - start_memory
            
            result = ProcessingResult(
                data=processed_data,
                status=ProcessingStatus.COMPLETED,
                processing_time=processing_time,
                memory_used=memory_used,
                records_processed=len(processed_data),
                features_generated=generated_features,
                warnings=warnings_list
            )
            
            logger.info(f"✅ Feature engineering completed: {generated_features} features generated")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Feature engineering failed: {e}")
            
            return ProcessingResult(
                data=None,
                status=ProcessingStatus.FAILED,
                processing_time=processing_time,
                memory_used=0.0,
                records_processed=0,
                errors=[str(e)]
            )
    
    def _normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize numeric features to [0, 1]"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        result = data.copy()
        
        for col in numeric_cols:
            min_val = result[col].min()
            max_val = result[col].max()
            if max_val > min_val:
                result[col] = (result[col] - min_val) / (max_val - min_val)
        
        return result
    
    def _standardize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize features to mean=0, std=1"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        result = data.copy()
        
        for col in numeric_cols:
            mean_val = result[col].mean()
            std_val = result[col].std()
            if std_val > 0:
                result[col] = (result[col] - mean_val) / std_val
        
        return result
    
    def _min_max_scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Min-max scaling"""
        return self._normalize_features(data)
    
    def _robust_scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Robust scaling using median and IQR"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        result = data.copy()
        
        for col in numeric_cols:
            median_val = result[col].median()
            q75 = result[col].quantile(0.75)
            q25 = result[col].quantile(0.25)
            iqr = q75 - q25
            if iqr > 0:
                result[col] = (result[col] - median_val) / iqr
        
        return result
    
    def _log_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation to numeric features"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        result = data.copy()
        
        for col in numeric_cols:
            if (result[col] > 0).all():
                result[f"{col}_log"] = np.log(result[col])
        
        return result
    
    def _polynomial_features(self, data: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Generate polynomial features"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        result = data.copy()
        
        for col in numeric_cols:
            for d in range(2, degree + 1):
                result[f"{col}_pow_{d}"] = result[col] ** d
        
        return result
    
    def _interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features between numeric columns"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        result = data.copy()
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                result[f"{col1}_x_{col2}"] = result[col1] * result[col2]
        
        return result
    
    def _bin_features(self, data: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
        """Create binned versions of numeric features"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        result = data.copy()
        
        for col in numeric_cols:
            try:
                result[f"{col}_binned"] = pd.cut(result[col], bins=bins, labels=False)
            except Exception:
                continue
        
        return result
    
    def _one_hot_encode(self, data: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical features"""
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        result = data.copy()
        
        for col in categorical_cols:
            if result[col].nunique() <= 20:  # Limit to prevent explosion
                dummies = pd.get_dummies(result[col], prefix=col)
                result = pd.concat([result, dummies], axis=1)
                result = result.drop(columns=[col])
        
        return result
    
    def _label_encode(self, data: pd.DataFrame) -> pd.DataFrame:
        """Label encode categorical features"""
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        result = data.copy()
        
        for col in categorical_cols:
            unique_values = result[col].unique()
            label_map = {val: idx for idx, val in enumerate(unique_values)}
            result[f"{col}_encoded"] = result[col].map(label_map)
        
        return result
    
    def _target_encode(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Target encode categorical features"""
        if target_column not in data.columns:
            return data
        
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        result = data.copy()
        
        for col in categorical_cols:
            if col != target_column:
                target_mean = result.groupby(col)[target_column].mean()
                result[f"{col}_target_encoded"] = result[col].map(target_mean)
        
        return result
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024


class DataValidator:
    """
    Data validation and sanitization framework.
    
    Extracted and adapted from data validation patterns,
    providing comprehensive data quality assurance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.validation_rules = {}
        self.sanitization_rules = {}
        
        # Initialize validation rules
        self._initialize_validation_rules()
        self._initialize_sanitization_rules()
    
    def _initialize_validation_rules(self):
        """Initialize data validation rules"""
        self.validation_rules = {
            'null_check': self._check_nulls,
            'type_check': self._check_types,
            'range_check': self._check_ranges,
            'uniqueness_check': self._check_uniqueness,
            'format_check': self._check_formats,
            'consistency_check': self._check_consistency
        }
    
    def _initialize_sanitization_rules(self):
        """Initialize data sanitization rules"""
        self.sanitization_rules = {
            'remove_nulls': self._remove_nulls,
            'fill_nulls': self._fill_nulls,
            'remove_duplicates': self._remove_duplicates,
            'fix_types': self._fix_types,
            'clip_outliers': self._clip_outliers,
            'normalize_text': self._normalize_text
        }
    
    async def validate_data(self, data: pd.DataFrame, 
                          schema: Optional[DataSchema] = None,
                          validation_level: ValidationLevel = ValidationLevel.BASIC) -> ProcessingResult:
        """Validate data quality"""
        start_time = time.time()
        
        try:
            errors = []
            warnings = []
            
            logger.info(f"🔍 Validating data with {validation_level.value} validation")
            
            # Schema validation if provided
            if schema:
                is_valid, schema_errors = schema.validate_data(data)
                if not is_valid:
                    errors.extend(schema_errors)
            
            # Apply validation rules based on level
            if validation_level == ValidationLevel.BASIC:
                validation_rules = ['null_check', 'type_check']
            elif validation_level == ValidationLevel.STRICT:
                validation_rules = list(self.validation_rules.keys())
            else:
                validation_rules = []
            
            for rule_name in validation_rules:
                rule_func = self.validation_rules.get(rule_name)
                if rule_func:
                    rule_errors, rule_warnings = rule_func(data)
                    errors.extend(rule_errors)
                    warnings.extend(rule_warnings)
            
            processing_time = time.time() - start_time
            status = ProcessingStatus.COMPLETED if not errors else ProcessingStatus.FAILED
            
            result = ProcessingResult(
                data=data,
                status=status,
                processing_time=processing_time,
                memory_used=0.0,
                records_processed=len(data),
                errors=errors,
                warnings=warnings,
                metadata={'validation_level': validation_level.value}
            )
            
            logger.info(f"✅ Data validation completed: {len(errors)} errors, {len(warnings)} warnings")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Data validation failed: {e}")
            
            return ProcessingResult(
                data=None,
                status=ProcessingStatus.FAILED,
                processing_time=processing_time,
                memory_used=0.0,
                records_processed=0,
                errors=[str(e)]
            )
    
    async def sanitize_data(self, data: pd.DataFrame, 
                          sanitization_rules: List[str]) -> ProcessingResult:
        """Sanitize data based on rules"""
        start_time = time.time()
        
        try:
            sanitized_data = data.copy()
            warnings_list = []
            
            logger.info(f"🧹 Sanitizing data with {len(sanitization_rules)} rules")
            
            for rule_name in sanitization_rules:
                rule_func = self.sanitization_rules.get(rule_name)
                if rule_func:
                    try:
                        sanitized_data = rule_func(sanitized_data)
                        logger.debug(f"✅ Applied sanitization rule: {rule_name}")
                    except Exception as e:
                        warning_msg = f"Failed to apply sanitization rule {rule_name}: {e}"
                        warnings_list.append(warning_msg)
                        logger.warning(warning_msg)
            
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                data=sanitized_data,
                status=ProcessingStatus.COMPLETED,
                processing_time=processing_time,
                memory_used=0.0,
                records_processed=len(sanitized_data),
                warnings=warnings_list
            )
            
            logger.info(f"✅ Data sanitization completed")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Data sanitization failed: {e}")
            
            return ProcessingResult(
                data=None,
                status=ProcessingStatus.FAILED,
                processing_time=processing_time,
                memory_used=0.0,
                records_processed=0,
                errors=[str(e)]
            )
    
    def _check_nulls(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Check for null values"""
        errors = []
        warnings = []
        
        null_counts = data.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                percentage = (count / len(data)) * 100
                if percentage > 50:
                    errors.append(f"Column {col} has {percentage:.1f}% null values")
                elif percentage > 10:
                    warnings.append(f"Column {col} has {percentage:.1f}% null values")
        
        return errors, warnings
    
    def _check_types(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Check data types consistency"""
        errors = []
        warnings = []
        
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check for mixed types
                sample_types = set(type(x).__name__ for x in data[col].dropna().head(100))
                if len(sample_types) > 1:
                    warnings.append(f"Column {col} has mixed types: {sample_types}")
        
        return errors, warnings
    
    def _check_ranges(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Check for values outside expected ranges"""
        errors = []
        warnings = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            if outliers > 0:
                percentage = (outliers / len(data)) * 100
                if percentage > 10:
                    warnings.append(f"Column {col} has {percentage:.1f}% outliers")
        
        return errors, warnings
    
    def _check_uniqueness(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Check for unexpected duplicates"""
        errors = []
        warnings = []
        
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            percentage = (duplicates / len(data)) * 100
            if percentage > 10:
                errors.append(f"Data has {percentage:.1f}% duplicate rows")
            else:
                warnings.append(f"Data has {percentage:.1f}% duplicate rows")
        
        return errors, warnings
    
    def _check_formats(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Check data format consistency"""
        errors = []
        warnings = []
        
        # Check for common format issues
        for col in data.select_dtypes(include=['object']).columns:
            sample_values = data[col].dropna().head(100)
            
            # Check for email format if column suggests it
            if 'email' in col.lower():
                invalid_emails = sum(1 for val in sample_values if '@' not in str(val))
                if invalid_emails > 0:
                    warnings.append(f"Column {col} has {invalid_emails} invalid email formats in sample")
        
        return errors, warnings
    
    def _check_consistency(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Check for data consistency issues"""
        errors = []
        warnings = []
        
        # Basic consistency checks
        for col in data.columns:
            if data[col].dtype in ['datetime64[ns]']:
                # Check for future dates if inappropriate
                future_dates = (data[col] > pd.Timestamp.now()).sum()
                if future_dates > 0:
                    warnings.append(f"Column {col} has {future_dates} future dates")
        
        return errors, warnings
    
    def _remove_nulls(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with null values"""
        return data.dropna()
    
    def _fill_nulls(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill null values with appropriate defaults"""
        result = data.copy()
        
        for col in result.columns:
            if result[col].dtype in ['int64', 'float64']:
                result[col] = result[col].fillna(result[col].median())
            elif result[col].dtype == 'object':
                result[col] = result[col].fillna(result[col].mode().iloc[0] if not result[col].mode().empty else 'unknown')
        
        return result
    
    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        return data.drop_duplicates()
    
    def _fix_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix data type issues"""
        result = data.copy()
        
        for col in result.columns:
            if result[col].dtype == 'object':
                # Try to convert to numeric if possible
                try:
                    numeric_data = pd.to_numeric(result[col], errors='coerce')
                    if not numeric_data.isna().all():
                        result[col] = numeric_data
                except Exception:
                    continue
        
        return result
    
    def _clip_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers to reasonable ranges"""
        result = data.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            q1 = result[col].quantile(0.01)
            q99 = result[col].quantile(0.99)
            result[col] = result[col].clip(lower=q1, upper=q99)
        
        return result
    
    def _normalize_text(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize text data"""
        result = data.copy()
        text_cols = result.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            result[col] = result[col].astype(str).str.strip().str.lower()
        
        return result


class DataEngine:
    """
    Main data processing engine orchestrating all data operations.
    
    Extracted and adapted from molecular_analyzer.py data handling patterns,
    made domain-agnostic while preserving sophisticated capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.DataEngine")
        
        # Core components
        self.feature_engineer = FeatureEngineer(config)
        self.data_validator = DataValidator(config)
        
        # Processing configuration
        self.max_workers = self.config.get('max_workers', 4)
        self.batch_size = self.config.get('batch_size', 10000)
        self.memory_limit_mb = self.config.get('memory_limit_mb', 4096)
        self.cache_enabled = self.config.get('cache_enabled', True)
        
        # State management
        self.processing_cache = {}
        self.data_store = {}
        self.processing_queue = asyncio.Queue()
        self.worker_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.running = False
        
        # Statistics
        self.statistics = {
            'total_processed': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'cache_hits': 0,
            'total_processing_time': 0.0,
            'memory_peak': 0.0
        }
        
        self.logger.info("🏭 Data Processing Engine initialized")
    
    async def start(self):
        """Start the data engine"""
        if self.running:
            return
        
        self.running = True
        self.logger.info("🚀 Data Engine started")
    
    async def stop(self):
        """Stop the data engine"""
        if not self.running:
            return
        
        self.running = False
        self.worker_pool.shutdown(wait=True)
        self.logger.info("🛑 Data Engine stopped")
    
    async def load_data(self, source: Union[str, Path, pd.DataFrame], 
                       data_format: Optional[DataFormat] = None,
                       **kwargs) -> ProcessingResult:
        """Load data from various sources"""
        start_time = time.time()
        
        try:
            if isinstance(source, pd.DataFrame):
                data = source.copy()
                self.logger.info(f"📊 Loaded DataFrame with {len(data)} rows, {len(data.columns)} columns")
            
            elif isinstance(source, (str, Path)):
                source_path = Path(source)
                
                # Auto-detect format if not specified
                if data_format is None:
                    data_format = self._detect_format(source_path)
                
                data = await self._load_from_file(source_path, data_format, **kwargs)
                self.logger.info(f"📁 Loaded {data_format.value} file: {len(data)} rows, {len(data.columns)} columns")
            
            else:
                raise ValueError(f"Unsupported data source type: {type(source)}")
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                data=data,
                status=ProcessingStatus.COMPLETED,
                processing_time=processing_time,
                memory_used=self._get_dataframe_memory(data),
                records_processed=len(data),
                metadata={'source': str(source), 'format': data_format.value if data_format else 'dataframe'}
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Failed to load data from {source}: {e}")
            
            return ProcessingResult(
                data=None,
                status=ProcessingStatus.FAILED,
                processing_time=processing_time,
                memory_used=0.0,
                records_processed=0,
                errors=[str(e)]
            )
    
    async def save_data(self, data: pd.DataFrame, 
                       destination: Union[str, Path],
                       data_format: DataFormat,
                       **kwargs) -> ProcessingResult:
        """Save data to various formats"""
        start_time = time.time()
        
        try:
            destination_path = Path(destination)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            
            await self._save_to_file(data, destination_path, data_format, **kwargs)
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"💾 Saved data to {destination_path} ({data_format.value})")
            
            return ProcessingResult(
                data=data,
                status=ProcessingStatus.COMPLETED,
                processing_time=processing_time,
                memory_used=0.0,
                records_processed=len(data),
                metadata={'destination': str(destination), 'format': data_format.value}
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Failed to save data to {destination}: {e}")
            
            return ProcessingResult(
                data=None,
                status=ProcessingStatus.FAILED,
                processing_time=processing_time,
                memory_used=0.0,
                records_processed=0,
                errors=[str(e)]
            )
    
    async def process_batch(self, data: pd.DataFrame,
                          operations: List[Dict[str, Any]]) -> ProcessingResult:
        """Process data in batches with multiple operations"""
        start_time = time.time()
        
        try:
            processed_data = data.copy()
            total_features_generated = 0
            all_warnings = []
            all_errors = []
            
            self.logger.info(f"🔄 Processing batch: {len(data)} rows, {len(operations)} operations")
            
            # Process in chunks if data is large
            if len(data) > self.batch_size:
                return await self._process_large_batch(data, operations)
            
            # Apply operations sequentially
            for operation in operations:
                op_type = operation.get('type')
                op_params = operation.get('params', {})
                
                if op_type == 'validate':
                    schema = op_params.get('schema')
                    validation_level = ValidationLevel(op_params.get('level', 'basic'))
                    result = await self.data_validator.validate_data(processed_data, schema, validation_level)
                    
                elif op_type == 'sanitize':
                    rules = op_params.get('rules', [])
                    result = await self.data_validator.sanitize_data(processed_data, rules)
                    
                elif op_type == 'engineer_features':
                    transformations = op_params.get('transformations', [])
                    target_column = op_params.get('target_column')
                    result = await self.feature_engineer.engineer_features(
                        processed_data, transformations, target_column
                    )
                    
                else:
                    continue
                
                if result.status == ProcessingStatus.COMPLETED:
                    processed_data = result.data
                    total_features_generated += result.features_generated
                    all_warnings.extend(result.warnings)
                else:
                    all_errors.extend(result.errors)
            
            processing_time = time.time() - start_time
            status = ProcessingStatus.COMPLETED if not all_errors else ProcessingStatus.FAILED
            
            result = ProcessingResult(
                data=processed_data,
                status=status,
                processing_time=processing_time,
                memory_used=self._get_dataframe_memory(processed_data),
                records_processed=len(processed_data),
                features_generated=total_features_generated,
                errors=all_errors,
                warnings=all_warnings
            )
            
            # Update statistics
            self.statistics['total_processed'] += len(data)
            if status == ProcessingStatus.COMPLETED:
                self.statistics['successful_operations'] += 1
            else:
                self.statistics['failed_operations'] += 1
            self.statistics['total_processing_time'] += processing_time
            
            self.logger.info(f"✅ Batch processing completed: {len(processed_data)} rows processed")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Batch processing failed: {e}")
            
            self.statistics['failed_operations'] += 1
            
            return ProcessingResult(
                data=None,
                status=ProcessingStatus.FAILED,
                processing_time=processing_time,
                memory_used=0.0,
                records_processed=0,
                errors=[str(e)]
            )
    
    async def _process_large_batch(self, data: pd.DataFrame, 
                                 operations: List[Dict[str, Any]]) -> ProcessingResult:
        """Process large datasets in chunks"""
        start_time = time.time()
        
        try:
            chunks = [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            processed_chunks = []
            total_features_generated = 0
            all_warnings = []
            all_errors = []
            
            self.logger.info(f"📊 Processing large batch in {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                self.logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
                
                chunk_result = await self.process_batch(chunk, operations)
                
                if chunk_result.status == ProcessingStatus.COMPLETED:
                    processed_chunks.append(chunk_result.data)
                    total_features_generated += chunk_result.features_generated
                    all_warnings.extend(chunk_result.warnings)
                else:
                    all_errors.extend(chunk_result.errors)
                
                # Memory management
                if self._get_memory_usage() > self.memory_limit_mb:
                    gc.collect()
            
            # Combine processed chunks
            if processed_chunks:
                final_data = pd.concat(processed_chunks, ignore_index=True)
            else:
                final_data = None
            
            processing_time = time.time() - start_time
            status = ProcessingStatus.COMPLETED if final_data is not None else ProcessingStatus.FAILED
            
            return ProcessingResult(
                data=final_data,
                status=status,
                processing_time=processing_time,
                memory_used=self._get_dataframe_memory(final_data) if final_data is not None else 0.0,
                records_processed=len(final_data) if final_data is not None else 0,
                features_generated=total_features_generated,
                errors=all_errors,
                warnings=all_warnings,
                metadata={'chunks_processed': len(chunks)}
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Large batch processing failed: {e}")
            
            return ProcessingResult(
                data=None,
                status=ProcessingStatus.FAILED,
                processing_time=processing_time,
                memory_used=0.0,
                records_processed=0,
                errors=[str(e)]
            )
    
    def _detect_format(self, file_path: Path) -> DataFormat:
        """Auto-detect file format from extension"""
        extension = file_path.suffix.lower()
        
        format_map = {
            '.csv': DataFormat.CSV,
            '.json': DataFormat.JSON,
            '.parquet': DataFormat.PARQUET,
            '.h5': DataFormat.HDF5,
            '.hdf5': DataFormat.HDF5,
            '.pkl': DataFormat.PICKLE,
            '.pickle': DataFormat.PICKLE,
            '.npy': DataFormat.NUMPY,
            '.feather': DataFormat.FEATHER,
            '.xlsx': DataFormat.EXCEL,
            '.xls': DataFormat.EXCEL
        }
        
        return format_map.get(extension, DataFormat.CSV)
    
    async def _load_from_file(self, file_path: Path, 
                            data_format: DataFormat, **kwargs) -> pd.DataFrame:
        """Load data from file based on format"""
        if data_format == DataFormat.CSV:
            return pd.read_csv(file_path, **kwargs)
        elif data_format == DataFormat.JSON:
            return pd.read_json(file_path, **kwargs)
        elif data_format == DataFormat.PARQUET:
            return pd.read_parquet(file_path, **kwargs)
        elif data_format == DataFormat.HDF5:
            return pd.read_hdf(file_path, **kwargs)
        elif data_format == DataFormat.PICKLE:
            return pd.read_pickle(file_path, **kwargs)
        elif data_format == DataFormat.FEATHER:
            return pd.read_feather(file_path, **kwargs)
        elif data_format == DataFormat.EXCEL:
            return pd.read_excel(file_path, **kwargs)
        elif data_format == DataFormat.NUMPY:
            array = np.load(file_path)
            return pd.DataFrame(array)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
    
    async def _save_to_file(self, data: pd.DataFrame, file_path: Path,
                          data_format: DataFormat, **kwargs):
        """Save data to file based on format"""
        if data_format == DataFormat.CSV:
            data.to_csv(file_path, index=False, **kwargs)
        elif data_format == DataFormat.JSON:
            data.to_json(file_path, **kwargs)
        elif data_format == DataFormat.PARQUET:
            data.to_parquet(file_path, **kwargs)
        elif data_format == DataFormat.HDF5:
            data.to_hdf(file_path, key='data', mode='w', **kwargs)
        elif data_format == DataFormat.PICKLE:
            data.to_pickle(file_path, **kwargs)
        elif data_format == DataFormat.FEATHER:
            data.to_feather(file_path, **kwargs)
        elif data_format == DataFormat.EXCEL:
            data.to_excel(file_path, index=False, **kwargs)
        elif data_format == DataFormat.NUMPY:
            np.save(file_path, data.values)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
    
    def _get_dataframe_memory(self, data: pd.DataFrame) -> float:
        """Get DataFrame memory usage in MB"""
        if data is None:
            return 0.0
        return data.memory_usage(deep=True).sum() / 1024 / 1024
    
    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data engine statistics"""
        current_memory = self._get_memory_usage()
        self.statistics['memory_peak'] = max(self.statistics['memory_peak'], current_memory)
        
        return {
            **self.statistics,
            'current_memory_mb': current_memory,
            'cache_size': len(self.processing_cache),
            'data_store_size': len(self.data_store),
            'running': self.running
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.processing_cache.clear()
        self.feature_engineer.feature_cache.clear()
        gc.collect()
        self.logger.info("🧹 Cleared all caches")


# Main async function for testing
async def main():
    """Main function for testing the data engine"""
    print("🏭 UNIVERSAL AI CORE DATA PROCESSING ENGINE")
    print("=" * 60)
    
    # Initialize data engine
    config = {
        'max_workers': 4,
        'batch_size': 1000,
        'memory_limit_mb': 2048,
        'cache_enabled': True
    }
    
    engine = DataEngine(config)
    
    try:
        # Start engine
        await engine.start()
        
        # Create test data
        print("\n📊 Creating test dataset...")
        test_data = pd.DataFrame({
            'feature_1': np.random.randn(1000),
            'feature_2': np.random.randint(0, 100, 1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000),
            'target': np.random.randint(0, 2, 1000)
        })
        
        # Test data loading
        load_result = await engine.load_data(test_data)
        print(f"✅ Data loaded: {load_result.status.value}")
        print(f"📊 Records: {load_result.records_processed}")
        
        # Test batch processing with multiple operations
        print("\n🔄 Testing batch processing...")
        operations = [
            {
                'type': 'validate',
                'params': {'level': 'basic'}
            },
            {
                'type': 'sanitize',
                'params': {'rules': ['fill_nulls', 'remove_duplicates']}
            },
            {
                'type': 'engineer_features',
                'params': {
                    'transformations': ['normalize', 'polynomial', 'one_hot'],
                    'target_column': 'target'
                }
            }
        ]
        
        batch_result = await engine.process_batch(test_data, operations)
        print(f"✅ Batch processing: {batch_result.status.value}")
        print(f"🔧 Features generated: {batch_result.features_generated}")
        print(f"⏱️ Processing time: {batch_result.processing_time:.3f}s")
        
        # Test data saving
        print("\n💾 Testing data saving...")
        save_result = await engine.save_data(
            batch_result.data, 
            "test_output.csv", 
            DataFormat.CSV
        )
        print(f"✅ Data saved: {save_result.status.value}")
        
        # Show statistics
        stats = engine.get_statistics()
        print(f"\n📊 Data Engine Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Data engine test completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())