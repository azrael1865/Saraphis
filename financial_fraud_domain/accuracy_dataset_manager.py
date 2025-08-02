"""
Accuracy Dataset Manager for Financial Fraud Detection
Complete dataset management system for train/validation/test splits with comprehensive production features.
Part of the Saraphis recursive methodology for accuracy tracking.
"""

import logging
import json
import pickle
import hashlib
import warnings
import threading
import time
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    TimeSeriesSplit,
    StratifiedShuffleSplit,
    ShuffleSplit
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
import psutil
import traceback
from contextlib import contextmanager
import gzip
import yaml

# Configure logging
logger = logging.getLogger(__name__)

# ======================== CUSTOM EXCEPTIONS ========================

class DatasetManagerError(Exception):
    """Base exception for dataset manager errors"""
    pass

class DataValidationError(DatasetManagerError):
    """Raised when data validation fails"""
    pass

class SplitConfigurationError(DatasetManagerError):
    """Raised when split configuration is invalid"""
    pass

class DataLeakageError(DatasetManagerError):
    """Raised when data leakage is detected"""
    pass

class ResourceLimitError(DatasetManagerError):
    """Raised when resource limits are exceeded"""
    pass

class TemporalLeakageError(DatasetManagerError):
    """Raised when temporal leakage is detected"""
    pass

class ConfigurationError(DatasetManagerError):
    """Raised when configuration is invalid"""
    pass

# ======================== ENUMS AND CONSTANTS ========================

class SplitType(Enum):
    """Types of data splits"""
    STANDARD = "standard"
    HOLDOUT = "holdout"
    TIME_STRATIFIED = "time_stratified"
    CROSS_VALIDATION = "cross_validation"
    NESTED_CV = "nested_cv"

class ValidationLevel(Enum):
    """Validation levels"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"

class ResourceLimit(Enum):
    """Resource limit types"""
    MEMORY_GB = "memory_gb"
    CPU_CORES = "cpu_cores"
    TIME_SECONDS = "time_seconds"

# ======================== DATA STRUCTURES ========================

@dataclass
class SplitConfiguration:
    """Configuration for dataset splits"""
    split_type: SplitType
    test_size: float = 0.2
    val_size: float = 0.2
    stratify: bool = True
    random_state: Optional[int] = 42
    temporal_column: Optional[str] = None
    cv_folds: int = 5
    nested_cv_outer_folds: int = 5
    nested_cv_inner_folds: int = 3
    shuffle: bool = True
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    resource_limits: Dict[ResourceLimit, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DatasetMetadata:
    """Metadata for dataset tracking"""
    dataset_id: str
    creation_timestamp: datetime
    source_hash: str
    n_samples: int
    n_features: int
    split_config: SplitConfiguration
    label_distribution: Dict[Any, int]
    feature_names: List[str]
    transformations: List[Dict[str, Any]]
    validation_results: Dict[str, Any]
    resource_usage: Dict[str, float]
    additional_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SplitResult:
    """Result of dataset splitting operation"""
    train_indices: np.ndarray
    val_indices: Optional[np.ndarray]
    test_indices: np.ndarray
    metadata: DatasetMetadata
    validation_report: Dict[str, Any]
    performance_metrics: Dict[str, float]

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    is_valid: bool
    validation_timestamp: datetime
    checks_performed: List[str]
    issues_found: List[Dict[str, Any]]
    recommendations: List[str]
    metrics: Dict[str, Any]
    
# ======================== MAIN CLASS ========================

class TrainValidationTestManager:
    """
    Complete dataset management system for train/validation/test splits.
    Provides production-ready functionality for dataset splitting, validation,
    versioning, and performance optimization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TrainValidationTestManager with comprehensive configuration.
        
        Args:
            config: Configuration dictionary with validation
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Load and validate configuration
        self.config = self._load_and_validate_config(config)
        
        # Initialize components
        self._init_storage()
        self._init_caching()
        self._init_resource_monitoring()
        self._init_validation_engine()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self.performance_stats = defaultdict(list)
        self.operation_counter = 0
        
        # Dataset tracking
        self.dataset_registry = {}
        self.active_splits = {}
        
        # Initialize thread pool for parallel operations
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 4)
        )
        
        self.logger.info(
            f"TrainValidationTestManager initialized with config: "
            f"storage_path={self.storage_path}, "
            f"cache_enabled={self.cache_enabled}, "
            f"validation_level={self.config.get('validation_level', 'standard')}"
        )
    
    def _load_and_validate_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load and validate configuration with defaults"""
        default_config = {
            'storage_path': Path('dataset_splits'),
            'cache_enabled': True,
            'cache_size_mb': 1000,
            'cache_ttl_seconds': 3600,
            'validation_level': ValidationLevel.STANDARD.value,
            'enable_compression': True,
            'compression_level': 6,
            'max_workers': 4,
            'resource_monitoring_enabled': True,
            'memory_limit_gb': 8.0,
            'cpu_limit_percent': 80.0,
            'operation_timeout_seconds': 3600,
            'enable_progress_tracking': True,
            'enable_detailed_logging': False,
            'metadata_version': '1.0',
            'auto_cleanup_days': 30,
            'enable_lineage_tracking': True
        }
        
        if config:
            # Validate user config
            self._validate_user_config(config)
            default_config.update(config)
        
        # Convert string paths to Path objects
        if isinstance(default_config['storage_path'], str):
            default_config['storage_path'] = Path(default_config['storage_path'])
        
        return default_config
    
    def _validate_user_config(self, config: Dict[str, Any]) -> None:
        """Validate user-provided configuration"""
        # Validate numeric limits
        if 'memory_limit_gb' in config:
            if not 0 < config['memory_limit_gb'] <= 1024:
                raise ConfigurationError(f"Invalid memory_limit_gb: {config['memory_limit_gb']}")
        
        if 'cpu_limit_percent' in config:
            if not 0 < config['cpu_limit_percent'] <= 100:
                raise ConfigurationError(f"Invalid cpu_limit_percent: {config['cpu_limit_percent']}")
        
        if 'cache_size_mb' in config:
            if not 0 < config['cache_size_mb'] <= 10240:
                raise ConfigurationError(f"Invalid cache_size_mb: {config['cache_size_mb']}")
        
        # Validate validation level
        if 'validation_level' in config:
            valid_levels = [level.value for level in ValidationLevel]
            if config['validation_level'] not in valid_levels:
                raise ConfigurationError(
                    f"Invalid validation_level: {config['validation_level']}. "
                    f"Must be one of {valid_levels}"
                )
    
    def _init_storage(self) -> None:
        """Initialize storage system"""
        self.storage_path = self.config['storage_path']
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.splits_path = self.storage_path / 'splits'
        self.metadata_path = self.storage_path / 'metadata'
        self.configs_path = self.storage_path / 'configs'
        self.lineage_path = self.storage_path / 'lineage'
        
        for path in [self.splits_path, self.metadata_path, 
                     self.configs_path, self.lineage_path]:
            path.mkdir(exist_ok=True)
        
        self.logger.info(f"Storage initialized at {self.storage_path}")
    
    def _init_caching(self) -> None:
        """Initialize caching system"""
        self.cache_enabled = self.config['cache_enabled']
        self.cache = {}
        self.cache_access_times = {}
        self.cache_size_bytes = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        if self.cache_enabled:
            self.cache_max_bytes = self.config['cache_size_mb'] * 1024 * 1024
            self.cache_ttl = self.config['cache_ttl_seconds']
            self.logger.info(f"Cache initialized with {self.config['cache_size_mb']}MB limit")
    
    def _init_resource_monitoring(self) -> None:
        """Initialize resource monitoring"""
        self.resource_monitoring = self.config['resource_monitoring_enabled']
        self.resource_limits = {
            ResourceLimit.MEMORY_GB: self.config['memory_limit_gb'],
            ResourceLimit.CPU_CORES: psutil.cpu_count() * (self.config['cpu_limit_percent'] / 100),
            ResourceLimit.TIME_SECONDS: self.config['operation_timeout_seconds']
        }
        
        if self.resource_monitoring:
            self._start_resource_monitor()
    
    def _init_validation_engine(self) -> None:
        """Initialize validation engine"""
        self.validation_level = ValidationLevel(self.config['validation_level'])
        self.validation_rules = self._load_validation_rules()
        
    # ======================== DATASET SPLITTING METHODS ========================
    
    def create_standard_splits(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, pd.Series],
        test_size: float = 0.2,
        val_size: float = 0.2,
        stratify: bool = True,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create standard train/validation/test splits with full stratification.
        
        Args:
            data: Feature data
            labels: Target labels
            test_size: Test set proportion
            val_size: Validation set proportion (from training data)
            stratify: Whether to stratify splits
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        with self._monitor_operation("create_standard_splits"):
            # Validate inputs
            self._validate_split_inputs(data, labels, test_size, val_size)
            
            # Check resource limits
            self._check_resource_limits(data)
            
            # Create split configuration
            split_config = SplitConfiguration(
                split_type=SplitType.STANDARD,
                test_size=test_size,
                val_size=val_size,
                stratify=stratify,
                random_state=random_state
            )
            
            # Perform splitting with progress tracking
            with self._progress_tracker("Creating standard splits", total=3) as progress:
                # First split: train+val vs test
                if stratify:
                    stratify_labels = labels
                else:
                    stratify_labels = None
                
                X_temp, X_test, y_temp, y_test = train_test_split(
                    data, labels,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify_labels
                )
                progress.update("Created test split")
                
                # Second split: train vs val
                val_size_adjusted = val_size / (1 - test_size)
                
                if stratify:
                    stratify_labels = y_temp
                else:
                    stratify_labels = None
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp,
                    test_size=val_size_adjusted,
                    random_state=random_state,
                    stratify=stratify_labels
                )
                progress.update("Created validation split")
                
                # Validate splits
                validation_report = self.validate_splits(
                    X_train, X_val, X_test,
                    y_train, y_val, y_test
                )
                progress.update("Validated splits")
            
            # Create and store metadata
            metadata = self._create_split_metadata(
                data, labels, split_config,
                train_indices=self._get_indices(data, X_train),
                val_indices=self._get_indices(data, X_val),
                test_indices=self._get_indices(data, X_test),
                validation_report=validation_report
            )
            
            # Cache results if enabled
            if self.cache_enabled:
                cache_key = self._generate_cache_key(
                    "standard_splits", data, labels, split_config
                )
                self._cache_result(cache_key, (X_train, X_val, X_test, y_train, y_val, y_test))
            
            # Log performance metrics
            self._log_split_metrics(metadata)
            
            return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_holdout_test_set(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, pd.Series],
        holdout_ratio: float = 0.1,
        temporal_column: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create holdout test set with temporal awareness.
        
        Args:
            data: Feature data
            labels: Target labels
            holdout_ratio: Proportion of data to hold out
            temporal_column: Column name for temporal ordering
            
        Returns:
            Tuple of (X_train, X_holdout, y_train, y_holdout)
        """
        with self._monitor_operation("create_holdout_test_set"):
            # Validate inputs
            if not 0 < holdout_ratio < 1:
                raise ValueError(f"holdout_ratio must be between 0 and 1, got {holdout_ratio}")
            
            # Check for temporal column
            if temporal_column and isinstance(data, pd.DataFrame):
                if temporal_column not in data.columns:
                    raise ValueError(f"Temporal column '{temporal_column}' not found in data")
                
                # Sort by temporal column
                sorted_indices = data[temporal_column].argsort()
                
                # Take last portion as holdout
                holdout_size = int(len(data) * holdout_ratio)
                holdout_indices = sorted_indices[-holdout_size:]
                train_indices = sorted_indices[:-holdout_size]
                
                # Split data
                if isinstance(data, pd.DataFrame):
                    X_train = data.iloc[train_indices]
                    X_holdout = data.iloc[holdout_indices]
                    y_train = labels.iloc[train_indices]
                    y_holdout = labels.iloc[holdout_indices]
                else:
                    X_train = data[train_indices]
                    X_holdout = data[holdout_indices]
                    y_train = labels[train_indices]
                    y_holdout = labels[holdout_indices]
                
                # Check temporal leakage
                if self.validation_level != ValidationLevel.MINIMAL:
                    leakage_report = self.detect_temporal_leakage(
                        data, temporal_column,
                        {'train': train_indices, 'holdout': holdout_indices}
                    )
                    if leakage_report['leakage_detected']:
                        warnings.warn(
                            f"Temporal leakage detected: {leakage_report['details']}",
                            TemporalLeakageError
                        )
            else:
                # Standard random holdout
                X_train, X_holdout, y_train, y_holdout = train_test_split(
                    data, labels,
                    test_size=holdout_ratio,
                    random_state=42
                )
            
            # Create metadata
            split_config = SplitConfiguration(
                split_type=SplitType.HOLDOUT,
                test_size=holdout_ratio,
                temporal_column=temporal_column
            )
            
            self._create_split_metadata(
                data, labels, split_config,
                train_indices=self._get_indices(data, X_train),
                test_indices=self._get_indices(data, X_holdout)
            )
            
            return X_train, X_holdout, y_train, y_holdout
    
    def create_time_stratified_split(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, pd.Series],
        time_column: str,
        split_date: Optional[Union[str, datetime]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create time-based splitting with validation.
        
        Args:
            data: Feature data (must be DataFrame with time column)
            labels: Target labels
            time_column: Name of the time column
            split_date: Date to split on (if None, uses 60/20/20 split)
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        with self._monitor_operation("create_time_stratified_split"):
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a DataFrame for time-stratified split")
            
            if time_column not in data.columns:
                raise ValueError(f"Time column '{time_column}' not found in data")
            
            # Convert time column to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
                data = data.copy()
                data[time_column] = pd.to_datetime(data[time_column])
            
            # Sort by time
            sorted_indices = data[time_column].argsort()
            data_sorted = data.iloc[sorted_indices]
            labels_sorted = labels.iloc[sorted_indices] if isinstance(labels, pd.Series) else labels[sorted_indices]
            
            if split_date:
                # Split based on provided date
                if isinstance(split_date, str):
                    split_date = pd.to_datetime(split_date)
                
                train_mask = data_sorted[time_column] < split_date
                test_mask = ~train_mask
                
                # Further split test into val and test
                test_data = data_sorted[test_mask]
                test_labels = labels_sorted[test_mask]
                
                val_size = len(test_data) // 2
                
                X_train = data_sorted[train_mask]
                y_train = labels_sorted[train_mask]
                X_val = test_data.iloc[:val_size]
                y_val = test_labels.iloc[:val_size] if isinstance(test_labels, pd.Series) else test_labels[:val_size]
                X_test = test_data.iloc[val_size:]
                y_test = test_labels.iloc[val_size:] if isinstance(test_labels, pd.Series) else test_labels[val_size:]
            else:
                # Default 60/20/20 split based on time
                n_samples = len(data_sorted)
                train_end = int(n_samples * 0.6)
                val_end = int(n_samples * 0.8)
                
                X_train = data_sorted.iloc[:train_end]
                y_train = labels_sorted[:train_end] if isinstance(labels_sorted, np.ndarray) else labels_sorted.iloc[:train_end]
                X_val = data_sorted.iloc[train_end:val_end]
                y_val = labels_sorted[train_end:val_end] if isinstance(labels_sorted, np.ndarray) else labels_sorted.iloc[train_end:val_end]
                X_test = data_sorted.iloc[val_end:]
                y_test = labels_sorted[val_end:] if isinstance(labels_sorted, np.ndarray) else labels_sorted.iloc[val_end:]
            
            # Validate no temporal leakage
            leakage_report = self.detect_temporal_leakage(
                data_sorted, time_column,
                {
                    'train': self._get_indices(data_sorted, X_train),
                    'val': self._get_indices(data_sorted, X_val),
                    'test': self._get_indices(data_sorted, X_test)
                }
            )
            
            if leakage_report['leakage_detected']:
                raise TemporalLeakageError(
                    f"Temporal leakage detected in time-stratified split: "
                    f"{leakage_report['details']}"
                )
            
            # Create metadata
            split_config = SplitConfiguration(
                split_type=SplitType.TIME_STRATIFIED,
                temporal_column=time_column,
                metadata={'split_date': str(split_date) if split_date else 'auto'}
            )
            
            self._create_split_metadata(
                data, labels, split_config,
                train_indices=self._get_indices(data, X_train),
                val_indices=self._get_indices(data, X_val),
                test_indices=self._get_indices(data, X_test)
            )
            
            return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_cross_validation_splits(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, pd.Series],
        cv_folds: int = 5,
        stratified: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create cross-validation splits.
        
        Args:
            data: Feature data
            labels: Target labels
            cv_folds: Number of CV folds
            stratified: Whether to use stratified CV
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        with self._monitor_operation("create_cross_validation_splits"):
            # Validate inputs
            self._validate_split_inputs(data, labels)
            
            if cv_folds < 2:
                raise ValueError(f"cv_folds must be at least 2, got {cv_folds}")
            
            # Create CV splitter
            if stratified:
                cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            else:
                cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Generate splits with parallel processing if large dataset
            n_samples = len(data)
            splits = []
            
            if n_samples > 10000 and self.config['max_workers'] > 1:
                # Parallel processing for large datasets
                with self._progress_tracker(f"Creating {cv_folds} CV splits", total=cv_folds) as progress:
                    futures = []
                    
                    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(data, labels)):
                        future = self.executor.submit(
                            self._validate_cv_fold,
                            fold_idx, train_idx, test_idx, data, labels
                        )
                        futures.append((future, train_idx, test_idx))
                    
                    for future, train_idx, test_idx in futures:
                        validation_result = future.result()
                        if validation_result['is_valid']:
                            splits.append((train_idx, test_idx))
                        else:
                            self.logger.warning(
                                f"CV fold validation failed: {validation_result['issues']}"
                            )
                        progress.update(f"Processed fold {len(splits)}/{cv_folds}")
            else:
                # Sequential processing for smaller datasets
                for train_idx, test_idx in cv_splitter.split(data, labels):
                    splits.append((train_idx, test_idx))
            
            # Validate all splits
            self._validate_cv_splits(splits, data, labels)
            
            # Create metadata
            split_config = SplitConfiguration(
                split_type=SplitType.CROSS_VALIDATION,
                cv_folds=cv_folds,
                stratify=stratified
            )
            
            self._create_split_metadata(
                data, labels, split_config,
                metadata={'n_splits': len(splits)}
            )
            
            return splits
    
    def create_nested_cv_splits(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, pd.Series],
        outer_folds: int = 5,
        inner_folds: int = 3
    ) -> List[Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]]:
        """
        Create nested cross-validation splits for hyperparameter tuning.
        
        Args:
            data: Feature data
            labels: Target labels
            outer_folds: Number of outer CV folds
            inner_folds: Number of inner CV folds
            
        Returns:
            List of (train_indices, test_indices, inner_splits) tuples
        """
        with self._monitor_operation("create_nested_cv_splits"):
            # Validate inputs
            self._validate_split_inputs(data, labels)
            
            if outer_folds < 2 or inner_folds < 2:
                raise ValueError(
                    f"Both outer_folds ({outer_folds}) and inner_folds ({inner_folds}) "
                    f"must be at least 2"
                )
            
            # Create outer CV
            outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)
            
            nested_splits = []
            
            with self._progress_tracker(f"Creating {outer_folds}x{inner_folds} nested CV splits", 
                                      total=outer_folds) as progress:
                for fold_idx, (train_val_idx, test_idx) in enumerate(outer_cv.split(data, labels)):
                    # Get training+validation data for inner CV
                    if isinstance(data, pd.DataFrame):
                        X_train_val = data.iloc[train_val_idx]
                        y_train_val = labels.iloc[train_val_idx]
                    else:
                        X_train_val = data[train_val_idx]
                        y_train_val = labels[train_val_idx]
                    
                    # Create inner CV
                    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=42+fold_idx)
                    inner_splits = []
                    
                    for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val):
                        # Convert to original indices
                        inner_train_original = train_val_idx[inner_train_idx]
                        inner_val_original = train_val_idx[inner_val_idx]
                        inner_splits.append((inner_train_original, inner_val_original))
                    
                    nested_splits.append((train_val_idx, test_idx, inner_splits))
                    progress.update(f"Processed outer fold {fold_idx + 1}/{outer_folds}")
            
            # Validate nested structure
            self._validate_nested_cv_splits(nested_splits, data, labels)
            
            # Create metadata
            split_config = SplitConfiguration(
                split_type=SplitType.NESTED_CV,
                nested_cv_outer_folds=outer_folds,
                nested_cv_inner_folds=inner_folds
            )
            
            self._create_split_metadata(
                data, labels, split_config,
                metadata={
                    'outer_folds': outer_folds,
                    'inner_folds': inner_folds,
                    'total_splits': outer_folds * inner_folds
                }
            )
            
            return nested_splits
    
    # ======================== DATA VALIDATION AND QUALITY ========================
    
    def validate_splits(
        self,
        train: Union[np.ndarray, pd.DataFrame],
        val: Optional[Union[np.ndarray, pd.DataFrame]],
        test: Union[np.ndarray, pd.DataFrame],
        labels_train: Union[np.ndarray, pd.Series],
        labels_val: Optional[Union[np.ndarray, pd.Series]],
        labels_test: Union[np.ndarray, pd.Series]
    ) -> Dict[str, Any]:
        """
        Comprehensive split validation.
        
        Args:
            train, val, test: Data splits
            labels_train, labels_val, labels_test: Corresponding labels
            
        Returns:
            Validation report dictionary
        """
        validation_results = {
            'is_valid': True,
            'checks_performed': [],
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Check 1: Size validation
        self._validate_split_sizes(
            train, val, test, labels_train, labels_val, labels_test,
            validation_results
        )
        
        # Check 2: Data type consistency
        self._validate_data_types(
            train, val, test,
            validation_results
        )
        
        # Check 3: Feature consistency
        if self.validation_level != ValidationLevel.MINIMAL:
            self._validate_feature_consistency(
                train, val, test,
                validation_results
            )
        
        # Check 4: Label distribution
        self._validate_label_distribution(
            labels_train, labels_val, labels_test,
            validation_results
        )
        
        # Check 5: Data leakage
        if self.validation_level == ValidationLevel.COMPREHENSIVE:
            indices = {
                'train': self._get_indices_from_split(train),
                'test': self._get_indices_from_split(test)
            }
            if val is not None:
                indices['val'] = self._get_indices_from_split(val)
            
            leakage_report = self.check_data_leakage(
                indices['train'],
                indices.get('val'),
                indices['test']
            )
            
            if leakage_report['leakage_detected']:
                validation_results['is_valid'] = False
                validation_results['issues'].append({
                    'type': 'data_leakage',
                    'details': leakage_report['details']
                })
        
        # Check 6: Statistical tests
        if self.validation_level != ValidationLevel.MINIMAL:
            self._validate_statistical_properties(
                train, val, test,
                validation_results
            )
        
        # Generate recommendations
        validation_results['recommendations'] = self._generate_validation_recommendations(
            validation_results
        )
        
        return validation_results
    
    def check_data_leakage(
        self,
        train_indices: np.ndarray,
        val_indices: Optional[np.ndarray],
        test_indices: np.ndarray
    ) -> Dict[str, Any]:
        """
        Data leakage detection.
        
        Args:
            train_indices: Training set indices
            val_indices: Validation set indices (optional)
            test_indices: Test set indices
            
        Returns:
            Leakage detection report
        """
        leakage_report = {
            'leakage_detected': False,
            'details': {},
            'overlapping_indices': {},
            'recommendations': []
        }
        
        # Convert to sets for efficient comparison
        train_set = set(train_indices.flatten())
        test_set = set(test_indices.flatten())
        
        # Check train-test overlap
        train_test_overlap = train_set & test_set
        if train_test_overlap:
            leakage_report['leakage_detected'] = True
            leakage_report['details']['train_test_overlap'] = len(train_test_overlap)
            leakage_report['overlapping_indices']['train_test'] = list(train_test_overlap)[:10]
            leakage_report['recommendations'].append(
                "Remove overlapping samples between train and test sets"
            )
        
        # Check validation overlap if provided
        if val_indices is not None:
            val_set = set(val_indices.flatten())
            
            # Train-val overlap
            train_val_overlap = train_set & val_set
            if train_val_overlap:
                leakage_report['leakage_detected'] = True
                leakage_report['details']['train_val_overlap'] = len(train_val_overlap)
                leakage_report['overlapping_indices']['train_val'] = list(train_val_overlap)[:10]
                leakage_report['recommendations'].append(
                    "Remove overlapping samples between train and validation sets"
                )
            
            # Val-test overlap
            val_test_overlap = val_set & test_set
            if val_test_overlap:
                leakage_report['leakage_detected'] = True
                leakage_report['details']['val_test_overlap'] = len(val_test_overlap)
                leakage_report['overlapping_indices']['val_test'] = list(val_test_overlap)[:10]
                leakage_report['recommendations'].append(
                    "Remove overlapping samples between validation and test sets"
                )
        
        # Check for duplicate indices within sets
        for name, indices in [('train', train_indices), ('val', val_indices), ('test', test_indices)]:
            if indices is not None:
                unique_count = len(np.unique(indices))
                total_count = len(indices)
                if unique_count < total_count:
                    leakage_report['details'][f'{name}_duplicates'] = total_count - unique_count
                    leakage_report['recommendations'].append(
                        f"Remove {total_count - unique_count} duplicate indices from {name} set"
                    )
        
        return leakage_report
    
    def validate_stratification(
        self,
        labels_train: Union[np.ndarray, pd.Series],
        labels_val: Optional[Union[np.ndarray, pd.Series]],
        labels_test: Union[np.ndarray, pd.Series]
    ) -> Dict[str, Any]:
        """
        Stratification validation.
        
        Args:
            labels_train, labels_val, labels_test: Label arrays
            
        Returns:
            Stratification validation report
        """
        stratification_report = {
            'is_properly_stratified': True,
            'label_distributions': {},
            'chi_square_test': {},
            'warnings': []
        }
        
        # Calculate label distributions
        train_dist = pd.Series(labels_train).value_counts(normalize=True).to_dict()
        test_dist = pd.Series(labels_test).value_counts(normalize=True).to_dict()
        
        stratification_report['label_distributions']['train'] = train_dist
        stratification_report['label_distributions']['test'] = test_dist
        
        if labels_val is not None:
            val_dist = pd.Series(labels_val).value_counts(normalize=True).to_dict()
            stratification_report['label_distributions']['val'] = val_dist
        
        # Check if all classes are present in all splits
        all_classes = set(train_dist.keys()) | set(test_dist.keys())
        if labels_val is not None:
            all_classes |= set(val_dist.keys())
        
        for split_name, dist in stratification_report['label_distributions'].items():
            missing_classes = all_classes - set(dist.keys())
            if missing_classes:
                stratification_report['is_properly_stratified'] = False
                stratification_report['warnings'].append(
                    f"{split_name} split missing classes: {missing_classes}"
                )
        
        # Statistical test for distribution similarity
        try:
            from scipy.stats import chi2_contingency
            
            # Create contingency table
            contingency_data = []
            for label in sorted(all_classes):
                row = []
                row.append(train_dist.get(label, 0) * len(labels_train))
                if labels_val is not None:
                    row.append(val_dist.get(label, 0) * len(labels_val))
                row.append(test_dist.get(label, 0) * len(labels_test))
                contingency_data.append(row)
            
            # Perform chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency_data)
            
            stratification_report['chi_square_test'] = {
                'statistic': chi2,
                'p_value': p_value,
                'degrees_of_freedom': dof
            }
            
            # Check if distributions are significantly different
            if p_value < 0.05:
                stratification_report['is_properly_stratified'] = False
                stratification_report['warnings'].append(
                    f"Label distributions significantly different (p={p_value:.4f})"
                )
        except ImportError:
            stratification_report['warnings'].append(
                "scipy not available for chi-square test"
            )
        
        # Check for severe imbalance
        for split_name, dist in stratification_report['label_distributions'].items():
            min_class_ratio = min(dist.values()) if dist else 0
            if min_class_ratio < 0.01:  # Less than 1%
                stratification_report['warnings'].append(
                    f"Severe class imbalance in {split_name} split: "
                    f"minority class ratio = {min_class_ratio:.2%}"
                )
        
        return stratification_report
    
    def assess_split_quality(
        self,
        splits_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Split quality metrics.
        
        Args:
            splits_info: Information about the splits
            
        Returns:
            Quality assessment report
        """
        quality_report = {
            'overall_quality_score': 0.0,
            'metrics': {},
            'issues': [],
            'recommendations': []
        }
        
        # Metric 1: Size balance
        if 'sizes' in splits_info:
            sizes = splits_info['sizes']
            total_size = sum(sizes.values())
            size_ratios = {k: v/total_size for k, v in sizes.items()}
            
            # Check if sizes match expected ratios
            expected_ratios = splits_info.get('expected_ratios', {'train': 0.6, 'val': 0.2, 'test': 0.2})
            size_score = 0.0
            
            for split, expected in expected_ratios.items():
                if split in size_ratios:
                    deviation = abs(size_ratios[split] - expected)
                    split_score = max(0, 1 - deviation * 10)  # Penalize deviations
                    size_score += split_score
                    
                    if deviation > 0.05:
                        quality_report['issues'].append(
                            f"{split} size deviation: expected {expected:.1%}, "
                            f"got {size_ratios[split]:.1%}"
                        )
            
            quality_report['metrics']['size_balance_score'] = size_score / len(expected_ratios)
        
        # Metric 2: Feature distribution similarity
        if 'feature_stats' in splits_info:
            feature_similarity_scores = []
            
            for feature_name, stats in splits_info['feature_stats'].items():
                if 'train' in stats and 'test' in stats:
                    # Compare means and stds
                    mean_diff = abs(stats['train']['mean'] - stats['test']['mean'])
                    std_ratio = stats['train']['std'] / (stats['test']['std'] + 1e-8)
                    
                    # Score based on similarity
                    mean_score = max(0, 1 - mean_diff / (abs(stats['train']['mean']) + 1e-8))
                    std_score = max(0, 1 - abs(1 - std_ratio))
                    
                    feature_score = (mean_score + std_score) / 2
                    feature_similarity_scores.append(feature_score)
                    
                    if feature_score < 0.8:
                        quality_report['issues'].append(
                            f"Feature '{feature_name}' distribution differs significantly "
                            f"between splits"
                        )
            
            if feature_similarity_scores:
                quality_report['metrics']['feature_similarity_score'] = np.mean(feature_similarity_scores)
        
        # Metric 3: Label balance
        if 'label_distributions' in splits_info:
            label_balance_scores = []
            
            for split, dist in splits_info['label_distributions'].items():
                if dist:
                    # Calculate entropy as measure of balance
                    probs = list(dist.values())
                    entropy = -sum(p * np.log2(p + 1e-8) for p in probs)
                    max_entropy = np.log2(len(probs))
                    balance_score = entropy / max_entropy if max_entropy > 0 else 0
                    label_balance_scores.append(balance_score)
            
            if label_balance_scores:
                quality_report['metrics']['label_balance_score'] = np.mean(label_balance_scores)
        
        # Calculate overall quality score
        if quality_report['metrics']:
            quality_report['overall_quality_score'] = np.mean(list(quality_report['metrics'].values()))
        
        # Generate recommendations
        if quality_report['overall_quality_score'] < 0.7:
            quality_report['recommendations'].append(
                "Consider re-splitting the data with different parameters"
            )
        
        if any('size deviation' in issue for issue in quality_report['issues']):
            quality_report['recommendations'].append(
                "Adjust split ratios to better match expected proportions"
            )
        
        if any('distribution differs' in issue for issue in quality_report['issues']):
            quality_report['recommendations'].append(
                "Consider using stratified sampling to maintain feature distributions"
            )
        
        return quality_report
    
    def detect_temporal_leakage(
        self,
        data: pd.DataFrame,
        time_column: str,
        splits: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Temporal leakage detection.
        
        Args:
            data: DataFrame with temporal information
            time_column: Name of time column
            splits: Dictionary of split names to indices
            
        Returns:
            Temporal leakage report
        """
        if time_column not in data.columns:
            return {
                'leakage_detected': False,
                'details': f"Time column '{time_column}' not found"
            }
        
        leakage_report = {
            'leakage_detected': False,
            'details': {},
            'temporal_overlaps': {},
            'recommendations': []
        }
        
        # Get time ranges for each split
        time_ranges = {}
        for split_name, indices in splits.items():
            if indices is not None and len(indices) > 0:
                split_times = data.iloc[indices][time_column]
                time_ranges[split_name] = {
                    'min': split_times.min(),
                    'max': split_times.max(),
                    'mean': split_times.mean()
                }
        
        # Check for temporal overlaps
        split_names = list(time_ranges.keys())
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                split1, split2 = split_names[i], split_names[j]
                range1, range2 = time_ranges[split1], time_ranges[split2]
                
                # Check if ranges overlap
                overlap = not (range1['max'] < range2['min'] or range2['max'] < range1['min'])
                
                if overlap:
                    # For train/val/test, we expect train < val < test
                    expected_order = ['train', 'val', 'test']
                    if split1 in expected_order and split2 in expected_order:
                        idx1 = expected_order.index(split1)
                        idx2 = expected_order.index(split2)
                        
                        if idx1 < idx2 and range1['mean'] > range2['mean']:
                            leakage_report['leakage_detected'] = True
                            leakage_report['temporal_overlaps'][f'{split1}_{split2}'] = {
                                'type': 'future_data_in_past_split',
                                f'{split1}_range': f"{range1['min']} to {range1['max']}",
                                f'{split2}_range': f"{range2['min']} to {range2['max']}"
                            }
                            leakage_report['recommendations'].append(
                                f"Ensure {split1} data is strictly before {split2} data"
                            )
                        elif overlap and abs(idx1 - idx2) == 1:
                            # Adjacent splits shouldn't overlap
                            leakage_report['details'][f'{split1}_{split2}_overlap'] = {
                                'overlap_start': max(range1['min'], range2['min']),
                                'overlap_end': min(range1['max'], range2['max'])
                            }
                            leakage_report['recommendations'].append(
                                f"Remove temporal overlap between {split1} and {split2}"
                            )
        
        return leakage_report
    
    # ======================== DATASET VERSIONING AND PROVENANCE ========================
    
    def create_dataset_metadata(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, pd.Series],
        split_config: SplitConfiguration
    ) -> DatasetMetadata:
        """
        Complete metadata creation.
        
        Args:
            data: Feature data
            labels: Target labels
            split_config: Split configuration
            
        Returns:
            Dataset metadata object
        """
        # Generate unique dataset ID
        dataset_id = self._generate_dataset_id()
        
        # Calculate data hash
        source_hash = self.generate_dataset_hash(data, labels)
        
        # Get data dimensions
        n_samples = len(data)
        n_features = data.shape[1] if len(data.shape) > 1 else 1
        
        # Get feature names
        if isinstance(data, pd.DataFrame):
            feature_names = data.columns.tolist()
        else:
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Calculate label distribution
        label_counts = pd.Series(labels).value_counts().to_dict()
        
        # Create metadata
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            creation_timestamp=datetime.now(),
            source_hash=source_hash,
            n_samples=n_samples,
            n_features=n_features,
            split_config=split_config,
            label_distribution=label_counts,
            feature_names=feature_names,
            transformations=[],
            validation_results={},
            resource_usage=self._get_current_resource_usage()
        )
        
        # Store in registry
        with self._lock:
            self.dataset_registry[dataset_id] = metadata
        
        return metadata
    
    def generate_dataset_hash(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, pd.Series]
    ) -> str:
        """
        Dataset fingerprinting.
        
        Args:
            data: Feature data
            labels: Target labels
            
        Returns:
            Hash string
        """
        hasher = hashlib.sha256()
        
        # Hash data
        if isinstance(data, pd.DataFrame):
            # Include column names in hash
            hasher.update(str(data.columns.tolist()).encode())
            data_array = data.values
        else:
            data_array = data
        
        # Sample data for hashing (for large datasets)
        if len(data_array) > 10000:
            # Use stratified sampling
            sample_indices = np.random.RandomState(42).choice(
                len(data_array), 
                size=10000, 
                replace=False
            )
            data_sample = data_array[sample_indices]
            labels_sample = labels[sample_indices] if hasattr(labels, '__getitem__') else labels.iloc[sample_indices]
        else:
            data_sample = data_array
            labels_sample = labels
        
        # Hash data content
        hasher.update(data_sample.tobytes())
        hasher.update(np.array(labels_sample).tobytes())
        
        # Include shape information
        hasher.update(str(data_array.shape).encode())
        
        return hasher.hexdigest()
    
    def save_split_configuration(
        self,
        config: Union[SplitConfiguration, Dict[str, Any]],
        output_path: Union[str, Path]
    ) -> bool:
        """
        Configuration persistence.
        
        Args:
            config: Split configuration
            output_path: Output file path
            
        Returns:
            Success status
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dictionary if needed
            if isinstance(config, SplitConfiguration):
                config_dict = asdict(config)
                # Convert enums to strings
                config_dict['split_type'] = config.split_type.value
                config_dict['validation_level'] = config.validation_level.value
                config_dict['resource_limits'] = {
                    k.value: v for k, v in config.resource_limits.items()
                }
            else:
                config_dict = config
            
            # Add metadata
            config_dict['_metadata'] = {
                'version': self.config['metadata_version'],
                'created_at': datetime.now().isoformat(),
                'manager_version': '1.0.0'
            }
            
            # Save based on extension
            if output_path.suffix == '.yaml':
                with open(output_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            elif output_path.suffix == '.json':
                with open(output_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            else:
                # Default to pickle with compression
                with gzip.open(output_path, 'wb') as f:
                    pickle.dump(config_dict, f)
            
            self.logger.info(f"Saved split configuration to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def load_split_configuration(
        self,
        config_path: Union[str, Path]
    ) -> Optional[SplitConfiguration]:
        """
        Configuration loading.
        
        Args:
            config_path: Configuration file path
            
        Returns:
            Split configuration or None
        """
        try:
            config_path = Path(config_path)
            
            if not config_path.exists():
                self.logger.error(f"Configuration file not found: {config_path}")
                return None
            
            # Load based on extension
            if config_path.suffix == '.yaml':
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
            else:
                # Try pickle with compression
                try:
                    with gzip.open(config_path, 'rb') as f:
                        config_dict = pickle.load(f)
                except:
                    with open(config_path, 'rb') as f:
                        config_dict = pickle.load(f)
            
            # Remove metadata
            config_dict.pop('_metadata', None)
            
            # Convert strings back to enums
            if 'split_type' in config_dict:
                config_dict['split_type'] = SplitType(config_dict['split_type'])
            if 'validation_level' in config_dict:
                config_dict['validation_level'] = ValidationLevel(config_dict['validation_level'])
            if 'resource_limits' in config_dict:
                config_dict['resource_limits'] = {
                    ResourceLimit(k): v for k, v in config_dict['resource_limits'].items()
                }
            
            # Create configuration object
            config = SplitConfiguration(**config_dict)
            
            self.logger.info(f"Loaded split configuration from {config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return None
    
    def track_dataset_lineage(
        self,
        source_data: Union[str, Path, Dict[str, Any]],
        transformations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Data lineage tracking.
        
        Args:
            source_data: Source data reference
            transformations: List of transformations applied
            
        Returns:
            Lineage tracking information
        """
        lineage_id = f"lineage_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
        
        lineage_info = {
            'lineage_id': lineage_id,
            'created_at': datetime.now().isoformat(),
            'source': str(source_data),
            'transformations': transformations,
            'dataset_versions': [],
            'dependencies': []
        }
        
        # Track transformations
        for idx, transform in enumerate(transformations):
            transform_record = {
                'step': idx + 1,
                'operation': transform.get('operation', 'unknown'),
                'parameters': transform.get('parameters', {}),
                'timestamp': transform.get('timestamp', datetime.now().isoformat()),
                'input_hash': transform.get('input_hash', ''),
                'output_hash': transform.get('output_hash', '')
            }
            lineage_info['dataset_versions'].append(transform_record)
        
        # Save lineage information
        if self.config['enable_lineage_tracking']:
            lineage_file = self.lineage_path / f"{lineage_id}.json"
            with open(lineage_file, 'w') as f:
                json.dump(lineage_info, f, indent=2)
            
            self.logger.info(f"Tracked dataset lineage: {lineage_id}")
        
        return lineage_info
    
    # ======================== PERFORMANCE AND OPTIMIZATION ========================
    
    @contextmanager
    def _monitor_operation(self, operation_name: str):
        """Context manager for monitoring operations"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            self.operation_counter += 1
            self.logger.debug(f"Starting operation: {operation_name}")
            yield
        finally:
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_used = end_memory - start_memory
            
            # Record performance stats
            with self._lock:
                self.performance_stats[operation_name].append({
                    'duration': duration,
                    'memory_used_mb': memory_used,
                    'timestamp': datetime.now()
                })
            
            self.logger.info(
                f"Operation '{operation_name}' completed in {duration:.2f}s, "
                f"memory delta: {memory_used:.1f}MB"
            )
    
    @contextmanager
    def _progress_tracker(self, description: str, total: int):
        """Context manager for progress tracking"""
        if not self.config['enable_progress_tracking']:
            yield lambda x: None
            return
        
        class ProgressTracker:
            def __init__(self, desc, total, logger):
                self.description = desc
                self.total = total
                self.current = 0
                self.logger = logger
                self.start_time = time.time()
                
            def update(self, message: str = ""):
                self.current += 1
                elapsed = time.time() - self.start_time
                rate = self.current / elapsed if elapsed > 0 else 0
                eta = (self.total - self.current) / rate if rate > 0 else 0
                
                self.logger.info(
                    f"{self.description}: {self.current}/{self.total} "
                    f"({self.current/self.total*100:.1f}%) - "
                    f"Rate: {rate:.1f}/s - ETA: {eta:.1f}s"
                    f"{' - ' + message if message else ''}"
                )
        
        tracker = ProgressTracker(description, total, self.logger)
        yield tracker.update
    
    def _check_resource_limits(self, data: Union[np.ndarray, pd.DataFrame]) -> None:
        """Check if operation exceeds resource limits"""
        if not self.resource_monitoring:
            return
        
        # Check memory usage
        current_memory_gb = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
        data_memory_gb = data.nbytes / 1024 / 1024 / 1024 if hasattr(data, 'nbytes') else 0
        
        total_expected_memory = current_memory_gb + data_memory_gb * 3  # Estimate for splits
        
        if total_expected_memory > self.resource_limits[ResourceLimit.MEMORY_GB]:
            raise ResourceLimitError(
                f"Operation would exceed memory limit: "
                f"{total_expected_memory:.1f}GB > {self.resource_limits[ResourceLimit.MEMORY_GB]:.1f}GB"
            )
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > self.config['cpu_limit_percent']:
            self.logger.warning(
                f"High CPU usage detected: {cpu_percent:.1f}% > {self.config['cpu_limit_percent']:.1f}%"
            )
    
    def _cache_result(self, key: str, result: Any) -> None:
        """Cache operation result"""
        if not self.cache_enabled:
            return
        
        # Serialize result
        serialized = pickle.dumps(result)
        size_bytes = len(serialized)
        
        # Check cache size
        with self._lock:
            # Evict old entries if needed
            while self.cache_size_bytes + size_bytes > self.cache_max_bytes and self.cache:
                # Remove oldest entry
                oldest_key = min(self.cache_access_times, key=self.cache_access_times.get)
                old_size = len(pickle.dumps(self.cache[oldest_key]))
                del self.cache[oldest_key]
                del self.cache_access_times[oldest_key]
                self.cache_size_bytes -= old_size
            
            # Add new entry
            self.cache[key] = result
            self.cache_access_times[key] = time.time()
            self.cache_size_bytes += size_bytes
    
    def _get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result if available"""
        if not self.cache_enabled:
            return None
        
        with self._lock:
            if key in self.cache:
                # Check TTL
                if time.time() - self.cache_access_times[key] < self.cache_ttl:
                    self.cache_hits += 1
                    self.cache_access_times[key] = time.time()
                    return self.cache[key]
                else:
                    # Expired
                    del self.cache[key]
                    del self.cache_access_times[key]
            
            self.cache_misses += 1
            return None
    
    def _generate_cache_key(self, *args) -> str:
        """Generate cache key from arguments"""
        key_parts = []
        for arg in args:
            if isinstance(arg, (np.ndarray, pd.DataFrame, pd.Series)):
                # Use shape and sample of data
                key_parts.append(str(arg.shape))
                if hasattr(arg, 'iloc'):
                    sample = str(arg.iloc[:5].values)
                else:
                    sample = str(arg[:5])
                key_parts.append(sample)
            else:
                key_parts.append(str(arg))
        
        return hashlib.md5('_'.join(key_parts).encode()).hexdigest()
    
    # ======================== HELPER METHODS ========================
    
    def _validate_split_inputs(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, pd.Series],
        test_size: float = None,
        val_size: float = None
    ) -> None:
        """Validate inputs for splitting operations"""
        # Check data and labels
        if data is None or labels is None:
            raise ValueError("Data and labels cannot be None")
        
        # Check shapes
        n_samples_data = len(data)
        n_samples_labels = len(labels)
        
        if n_samples_data != n_samples_labels:
            raise ValueError(
                f"Data and labels must have same number of samples: "
                f"{n_samples_data} != {n_samples_labels}"
            )
        
        if n_samples_data < 10:
            raise ValueError(
                f"Insufficient samples for splitting: {n_samples_data} < 10"
            )
        
        # Check split sizes
        if test_size is not None:
            if not 0 < test_size < 1:
                raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        
        if val_size is not None:
            if not 0 < val_size < 1:
                raise ValueError(f"val_size must be between 0 and 1, got {val_size}")
        
        if test_size and val_size:
            if test_size + val_size >= 1:
                raise ValueError(
                    f"test_size + val_size must be less than 1, "
                    f"got {test_size} + {val_size} = {test_size + val_size}"
                )
    
    def _get_indices(
        self,
        original_data: Union[np.ndarray, pd.DataFrame],
        subset_data: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Get indices of subset in original data"""
        if isinstance(original_data, pd.DataFrame) and isinstance(subset_data, pd.DataFrame):
            # Use index for DataFrames
            return subset_data.index.values
        else:
            # For arrays, we need to track indices during splitting
            # This is a simplified version - in production, indices should be tracked during split
            return np.arange(len(subset_data))
    
    def _get_indices_from_split(self, split_data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Get indices from split data"""
        if isinstance(split_data, pd.DataFrame):
            return split_data.index.values
        else:
            return np.arange(len(split_data))
    
    def _create_split_metadata(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, pd.Series],
        split_config: SplitConfiguration,
        **kwargs
    ) -> DatasetMetadata:
        """Create and store split metadata"""
        metadata = self.create_dataset_metadata(data, labels, split_config)
        
        # Add split-specific information
        for key, value in kwargs.items():
            if hasattr(metadata, 'additional_info'):
                metadata.additional_info[key] = value
        
        # Save metadata
        metadata_file = self.metadata_path / f"{metadata.dataset_id}.json"
        with open(metadata_file, 'w') as f:
            # Convert to dict for JSON serialization
            metadata_dict = asdict(metadata)
            metadata_dict['creation_timestamp'] = metadata.creation_timestamp.isoformat()
            metadata_dict['split_config']['split_type'] = metadata.split_config.split_type.value
            metadata_dict['split_config']['validation_level'] = metadata.split_config.validation_level.value
            
            json.dump(metadata_dict, f, indent=2)
        
        return metadata
    
    def _validate_split_sizes(
        self,
        train, val, test,
        labels_train, labels_val, labels_test,
        results: Dict[str, Any]
    ) -> None:
        """Validate split sizes"""
        results['checks_performed'].append('split_sizes')
        
        # Check if splits are empty
        for name, split in [('train', train), ('val', val), ('test', test)]:
            if split is not None and len(split) == 0:
                results['is_valid'] = False
                results['issues'].append({
                    'type': 'empty_split',
                    'split': name
                })
        
        # Check size ratios
        total_size = len(train) + (len(val) if val is not None else 0) + len(test)
        
        train_ratio = len(train) / total_size
        test_ratio = len(test) / total_size
        val_ratio = len(val) / total_size if val is not None else 0
        
        results['metrics']['split_ratios'] = {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        }
        
        # Warn if unusual ratios
        if train_ratio < 0.5:
            results['warnings'].append(
                f"Training set is small: {train_ratio:.1%} of total data"
            )
        
        if test_ratio < 0.1:
            results['warnings'].append(
                f"Test set is small: {test_ratio:.1%} of total data"
            )
    
    def _validate_data_types(
        self,
        train, val, test,
        results: Dict[str, Any]
    ) -> None:
        """Validate data type consistency"""
        results['checks_performed'].append('data_types')
        
        # Get types
        train_type = type(train)
        test_type = type(test)
        
        if train_type != test_type:
            results['is_valid'] = False
            results['issues'].append({
                'type': 'type_mismatch',
                'details': f"Train type ({train_type}) != Test type ({test_type})"
            })
        
        if val is not None:
            val_type = type(val)
            if val_type != train_type:
                results['is_valid'] = False
                results['issues'].append({
                    'type': 'type_mismatch',
                    'details': f"Val type ({val_type}) != Train type ({train_type})"
                })
    
    def _validate_feature_consistency(
        self,
        train, val, test,
        results: Dict[str, Any]
    ) -> None:
        """Validate feature consistency across splits"""
        results['checks_performed'].append('feature_consistency')
        
        # Check feature dimensions
        train_features = train.shape[1] if len(train.shape) > 1 else 1
        test_features = test.shape[1] if len(test.shape) > 1 else 1
        
        if train_features != test_features:
            results['is_valid'] = False
            results['issues'].append({
                'type': 'feature_mismatch',
                'details': f"Train features ({train_features}) != Test features ({test_features})"
            })
        
        if val is not None:
            val_features = val.shape[1] if len(val.shape) > 1 else 1
            if val_features != train_features:
                results['is_valid'] = False
                results['issues'].append({
                    'type': 'feature_mismatch',
                    'details': f"Val features ({val_features}) != Train features ({train_features})"
                })
        
        # For DataFrames, check column names
        if isinstance(train, pd.DataFrame):
            train_cols = set(train.columns)
            test_cols = set(test.columns)
            
            if train_cols != test_cols:
                missing_in_test = train_cols - test_cols
                extra_in_test = test_cols - train_cols
                
                results['issues'].append({
                    'type': 'column_mismatch',
                    'missing_in_test': list(missing_in_test),
                    'extra_in_test': list(extra_in_test)
                })
    
    def _validate_label_distribution(
        self,
        labels_train, labels_val, labels_test,
        results: Dict[str, Any]
    ) -> None:
        """Validate label distributions"""
        results['checks_performed'].append('label_distribution')
        
        # Get unique labels
        train_unique = set(np.unique(labels_train))
        test_unique = set(np.unique(labels_test))
        
        # Check if all labels are present
        if train_unique != test_unique:
            missing_in_test = train_unique - test_unique
            missing_in_train = test_unique - train_unique
            
            if missing_in_test:
                results['warnings'].append(
                    f"Labels {missing_in_test} present in train but not in test"
                )
            if missing_in_train:
                results['warnings'].append(
                    f"Labels {missing_in_train} present in test but not in train"
                )
        
        if labels_val is not None:
            val_unique = set(np.unique(labels_val))
            missing_in_val = train_unique - val_unique
            
            if missing_in_val:
                results['warnings'].append(
                    f"Labels {missing_in_val} present in train but not in validation"
                )
    
    def _validate_statistical_properties(
        self,
        train, val, test,
        results: Dict[str, Any]
    ) -> None:
        """Validate statistical properties of splits"""
        results['checks_performed'].append('statistical_properties')
        
        # Convert to numpy for statistics
        if isinstance(train, pd.DataFrame):
            train_array = train.values
            test_array = test.values
            val_array = val.values if val is not None else None
        else:
            train_array = train
            test_array = test
            val_array = val
        
        # Compare distributions using Kolmogorov-Smirnov test
        try:
            from scipy import stats
            
            n_features = train_array.shape[1] if len(train_array.shape) > 1 else 1
            
            significant_differences = 0
            for i in range(min(n_features, 10)):  # Check first 10 features
                if len(train_array.shape) > 1:
                    train_feature = train_array[:, i]
                    test_feature = test_array[:, i]
                else:
                    train_feature = train_array
                    test_feature = test_array
                
                # KS test
                ks_stat, p_value = stats.ks_2samp(train_feature, test_feature)
                
                if p_value < 0.05:
                    significant_differences += 1
            
            if significant_differences > n_features * 0.3:  # More than 30% of features differ
                results['warnings'].append(
                    f"{significant_differences}/{min(n_features, 10)} features show "
                    f"significant distribution differences between train and test"
                )
        except ImportError:
            results['warnings'].append(
                "scipy not available for statistical tests"
            )
    
    def _generate_validation_recommendations(
        self,
        results: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if not results['is_valid']:
            recommendations.append("Fix critical issues before using these splits")
        
        # Based on specific issues
        for issue in results['issues']:
            if issue['type'] == 'data_leakage':
                recommendations.append(
                    "Re-split the data ensuring no overlap between sets"
                )
            elif issue['type'] == 'feature_mismatch':
                recommendations.append(
                    "Ensure consistent feature preprocessing across all splits"
                )
            elif issue['type'] == 'empty_split':
                recommendations.append(
                    f"Increase data size or adjust split ratios to avoid empty {issue['split']} set"
                )
        
        # Based on warnings
        for warning in results['warnings']:
            if 'small' in warning.lower():
                recommendations.append(
                    "Consider collecting more data or adjusting split ratios"
                )
            elif 'distribution differences' in warning:
                recommendations.append(
                    "Consider stratified sampling to maintain feature distributions"
                )
            elif 'missing' in warning.lower() and 'label' in warning.lower():
                recommendations.append(
                    "Ensure all classes have sufficient representation in all splits"
                )
        
        return list(set(recommendations))  # Remove duplicates
    
    def _validate_cv_fold(
        self,
        fold_idx: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        data: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, pd.Series]
    ) -> Dict[str, Any]:
        """Validate a single CV fold"""
        validation_result = {
            'fold_idx': fold_idx,
            'is_valid': True,
            'issues': []
        }
        
        # Check for overlap
        if len(set(train_idx) & set(test_idx)) > 0:
            validation_result['is_valid'] = False
            validation_result['issues'].append('Train/test overlap detected')
        
        # Check sizes
        if len(train_idx) == 0 or len(test_idx) == 0:
            validation_result['is_valid'] = False
            validation_result['issues'].append('Empty train or test set')
        
        # Check label distribution
        train_labels = labels[train_idx] if isinstance(labels, np.ndarray) else labels.iloc[train_idx]
        test_labels = labels[test_idx] if isinstance(labels, np.ndarray) else labels.iloc[test_idx]
        
        train_unique = set(np.unique(train_labels))
        test_unique = set(np.unique(test_labels))
        
        if len(train_unique) < 2:
            validation_result['issues'].append('Insufficient classes in training set')
        
        if train_unique != test_unique:
            validation_result['issues'].append(
                f'Class mismatch: train={train_unique}, test={test_unique}'
            )
        
        return validation_result
    
    def _validate_cv_splits(
        self,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        data: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, pd.Series]
    ) -> None:
        """Validate all CV splits"""
        # Check that each sample appears in test set exactly once
        all_test_indices = []
        for train_idx, test_idx in splits:
            all_test_indices.extend(test_idx)
        
        unique_test_indices = set(all_test_indices)
        if len(unique_test_indices) != len(data):
            raise DataValidationError(
                f"CV splits invalid: {len(unique_test_indices)} unique test samples "
                f"!= {len(data)} total samples"
            )
        
        # Check no data leakage across folds
        for i, (train_i, test_i) in enumerate(splits):
            for j, (train_j, test_j) in enumerate(splits[i+1:], i+1):
                # Test sets should not overlap
                if len(set(test_i) & set(test_j)) > 0:
                    raise DataLeakageError(
                        f"Test sets overlap between folds {i} and {j}"
                    )
    
    def _validate_nested_cv_splits(
        self,
        nested_splits: List[Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]],
        data: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, pd.Series]
    ) -> None:
        """Validate nested CV structure"""
        for outer_idx, (train_val_idx, test_idx, inner_splits) in enumerate(nested_splits):
            # Check outer split
            if len(set(train_val_idx) & set(test_idx)) > 0:
                raise DataLeakageError(
                    f"Outer fold {outer_idx}: train/val and test sets overlap"
                )
            
            # Check inner splits
            for inner_idx, (train_idx, val_idx) in enumerate(inner_splits):
                # Inner splits should be subsets of train_val
                if not (set(train_idx) <= set(train_val_idx) and 
                       set(val_idx) <= set(train_val_idx)):
                    raise DataValidationError(
                        f"Inner fold {inner_idx} of outer fold {outer_idx} "
                        f"contains indices not in train/val set"
                    )
                
                # No overlap between inner train and val
                if len(set(train_idx) & set(val_idx)) > 0:
                    raise DataLeakageError(
                        f"Inner fold {inner_idx} of outer fold {outer_idx}: "
                        f"train and val sets overlap"
                    )
    
    def _generate_dataset_id(self) -> str:
        """Generate unique dataset ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = os.urandom(4).hex()
        return f"dataset_{timestamp}_{random_suffix}"
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules based on validation level"""
        rules = {
            ValidationLevel.MINIMAL: {
                'check_sizes': True,
                'check_types': True,
                'check_label_distribution': True
            },
            ValidationLevel.STANDARD: {
                'check_sizes': True,
                'check_types': True,
                'check_feature_consistency': True,
                'check_label_distribution': True,
                'check_statistical_properties': True
            },
            ValidationLevel.COMPREHENSIVE: {
                'check_sizes': True,
                'check_types': True,
                'check_feature_consistency': True,
                'check_label_distribution': True,
                'check_statistical_properties': True,
                'check_data_leakage': True,
                'check_temporal_leakage': True
            }
        }
        
        return rules.get(self.validation_level, rules[ValidationLevel.STANDARD])
    
    def _start_resource_monitor(self) -> None:
        """Start resource monitoring thread"""
        def monitor_resources():
            while True:
                try:
                    # Check memory
                    memory_gb = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
                    if memory_gb > self.resource_limits[ResourceLimit.MEMORY_GB] * 0.9:
                        self.logger.warning(
                            f"High memory usage: {memory_gb:.1f}GB "
                            f"(90% of {self.resource_limits[ResourceLimit.MEMORY_GB]:.1f}GB limit)"
                        )
                    
                    # Check CPU
                    cpu_percent = psutil.cpu_percent(interval=1)
                    if cpu_percent > self.config['cpu_limit_percent'] * 0.9:
                        self.logger.warning(
                            f"High CPU usage: {cpu_percent:.1f}% "
                            f"(90% of {self.config['cpu_limit_percent']:.1f}% limit)"
                        )
                    
                    time.sleep(10)  # Check every 10 seconds
                    
                except Exception as e:
                    self.logger.error(f"Resource monitoring error: {e}")
                    time.sleep(30)
        
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
    
    def _get_current_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        process = psutil.Process()
        return {
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(interval=0.1),
            'num_threads': process.num_threads(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _log_split_metrics(self, metadata: DatasetMetadata) -> None:
        """Log split operation metrics"""
        self.logger.info(
            f"Split completed - Dataset ID: {metadata.dataset_id}, "
            f"Samples: {metadata.n_samples}, Features: {metadata.n_features}, "
            f"Label distribution: {metadata.label_distribution}"
        )
    
    # ======================== PUBLIC UTILITY METHODS ========================
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of all operations"""
        with self._lock:
            summary = {
                'total_operations': self.operation_counter,
                'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) 
                                  if (self.cache_hits + self.cache_misses) > 0 else 0,
                'operations': {}
            }
            
            for op_name, stats in self.performance_stats.items():
                if stats:
                    durations = [s['duration'] for s in stats]
                    memory_usage = [s['memory_used_mb'] for s in stats]
                    
                    summary['operations'][op_name] = {
                        'count': len(stats),
                        'avg_duration': np.mean(durations),
                        'max_duration': np.max(durations),
                        'avg_memory_mb': np.mean(memory_usage),
                        'max_memory_mb': np.max(memory_usage)
                    }
            
            return summary
    
    def cleanup_old_data(self, days: int = None) -> int:
        """Clean up old split data and metadata"""
        if days is None:
            days = self.config['auto_cleanup_days']
        
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_count = 0
        
        # Clean metadata files
        for metadata_file in self.metadata_path.glob('*.json'):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                creation_date = datetime.fromisoformat(metadata['creation_timestamp'])
                if creation_date < cutoff_date:
                    metadata_file.unlink()
                    cleaned_count += 1
                    
                    # Also remove associated split files
                    dataset_id = metadata_file.stem
                    for split_file in self.splits_path.glob(f"{dataset_id}*"):
                        split_file.unlink()
                        
            except Exception as e:
                self.logger.error(f"Error cleaning file {metadata_file}: {e}")
        
        self.logger.info(f"Cleaned up {cleaned_count} old datasets")
        return cleaned_count
    
    def export_split_report(
        self,
        dataset_id: str,
        output_path: Union[str, Path]
    ) -> bool:
        """Export comprehensive split report"""
        try:
            output_path = Path(output_path)
            
            # Load metadata
            metadata_file = self.metadata_path / f"{dataset_id}.json"
            if not metadata_file.exists():
                self.logger.error(f"Dataset {dataset_id} not found")
                return False
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Create report
            report = {
                'dataset_id': dataset_id,
                'creation_timestamp': metadata['creation_timestamp'],
                'split_configuration': metadata['split_config'],
                'data_summary': {
                    'n_samples': metadata['n_samples'],
                    'n_features': metadata['n_features'],
                    'label_distribution': metadata['label_distribution']
                },
                'validation_results': metadata.get('validation_results', {}),
                'resource_usage': metadata.get('resource_usage', {}),
                'performance_summary': self.get_performance_summary()
            }
            
            # Save report
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.suffix == '.json':
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
            elif output_path.suffix == '.yaml':
                with open(output_path, 'w') as f:
                    yaml.dump(report, f, default_flow_style=False)
            else:
                # HTML report
                html_content = self._generate_html_report(report)
                with open(output_path, 'w') as f:
                    f.write(html_content)
            
            self.logger.info(f"Exported split report to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export report: {e}")
            return False
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report"""
        html = f"""
        <html>
        <head>
            <title>Dataset Split Report - {report['dataset_id']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #e7f3ff; padding: 10px; margin: 10px 0; }}
                .warning {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; }}
                .error {{ background-color: #f8d7da; padding: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Dataset Split Report</h1>
            <h2>Dataset: {report['dataset_id']}</h2>
            <p>Created: {report['creation_timestamp']}</p>
            
            <h3>Data Summary</h3>
            <div class="metric">
                <p>Samples: {report['data_summary']['n_samples']}</p>
                <p>Features: {report['data_summary']['n_features']}</p>
            </div>
            
            <h3>Label Distribution</h3>
            <table>
                <tr><th>Label</th><th>Count</th><th>Percentage</th></tr>
        """
        
        total_labels = sum(report['data_summary']['label_distribution'].values())
        for label, count in report['data_summary']['label_distribution'].items():
            percentage = (count / total_labels) * 100
            html += f"<tr><td>{label}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        
        html += """
            </table>
            
            <h3>Split Configuration</h3>
            <table>
        """
        
        for key, value in report['split_configuration'].items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# ======================== USAGE EXAMPLES ========================

if __name__ == "__main__":
    # Example usage demonstrating all features
    import numpy as np
    import pandas as pd
    
    # Create sample data
    np.random.seed(42)
    n_samples = 10000
    n_features = 20
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels with some class imbalance
    y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    # Convert to DataFrame for some examples
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Add temporal column for time-based examples
    X_df['timestamp'] = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    
    print("=== TrainValidationTestManager Examples ===\n")
    
    # Initialize manager
    config = {
        'validation_level': 'comprehensive',
        'enable_progress_tracking': True,
        'cache_enabled': True
    }
    
    manager = TrainValidationTestManager(config)
    
    # Example 1: Standard splits
    print("1. Creating standard train/val/test splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = manager.create_standard_splits(
        X, y, test_size=0.2, val_size=0.2, stratify=True, random_state=42
    )
    
    print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"   Class distribution - Train: {np.bincount(y_train)}, "
          f"Val: {np.bincount(y_val)}, Test: {np.bincount(y_test)}\n")
    
    # Example 2: Holdout test set
    print("2. Creating holdout test set...")
    X_train_h, X_holdout, y_train_h, y_holdout = manager.create_holdout_test_set(
        X_df, y, holdout_ratio=0.1, temporal_column='timestamp'
    )
    
    print(f"   Train: {X_train_h.shape}, Holdout: {X_holdout.shape}")
    print(f"   Temporal range - Train: {X_train_h['timestamp'].min()} to {X_train_h['timestamp'].max()}")
    print(f"   Temporal range - Holdout: {X_holdout['timestamp'].min()} to {X_holdout['timestamp'].max()}\n")
    
    # Example 3: Time-stratified split
    print("3. Creating time-stratified splits...")
    X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t = manager.create_time_stratified_split(
        X_df, y, time_column='timestamp'
    )
    
    print(f"   Train: {X_train_t.shape}, Val: {X_val_t.shape}, Test: {X_test_t.shape}\n")
    
    # Example 4: Cross-validation splits
    print("4. Creating cross-validation splits...")
    cv_splits = manager.create_cross_validation_splits(
        X[:1000], y[:1000], cv_folds=5, stratified=True  # Using subset for speed
    )
    
    print(f"   Created {len(cv_splits)} CV folds")
    for i, (train_idx, test_idx) in enumerate(cv_splits[:2]):  # Show first 2
        print(f"   Fold {i}: Train={len(train_idx)}, Test={len(test_idx)}")
    print("   ...\n")
    
    # Example 5: Nested CV splits
    print("5. Creating nested cross-validation splits...")
    nested_splits = manager.create_nested_cv_splits(
        X[:1000], y[:1000], outer_folds=3, inner_folds=2  # Using subset for speed
    )
    
    print(f"   Created {len(nested_splits)} outer folds")
    for i, (train_val_idx, test_idx, inner_splits) in enumerate(nested_splits):
        print(f"   Outer fold {i}: Train+Val={len(train_val_idx)}, Test={len(test_idx)}, "
              f"Inner splits={len(inner_splits)}")
    print()
    
    # Example 6: Data validation
    print("6. Validating splits...")
    validation_report = manager.validate_splits(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    print(f"   Validation result: {'PASS' if validation_report['is_valid'] else 'FAIL'}")
    print(f"   Checks performed: {', '.join(validation_report['checks_performed'])}")
    if validation_report['warnings']:
        print(f"   Warnings: {validation_report['warnings'][0]}")
    print()
    
    # Example 7: Check data leakage
    print("7. Checking for data leakage...")
    train_indices = np.arange(8000)
    val_indices = np.arange(8000, 9000)
    test_indices = np.arange(9000, 10000)
    
    # Intentionally create overlap for demonstration
    test_indices[0] = train_indices[0]  # Create leakage
    
    leakage_report = manager.check_data_leakage(
        train_indices, val_indices, test_indices
    )
    
    print(f"   Leakage detected: {leakage_report['leakage_detected']}")
    if leakage_report['leakage_detected']:
        print(f"   Details: {leakage_report['details']}")
    print()
    
    # Example 8: Performance summary
    print("8. Performance summary:")
    perf_summary = manager.get_performance_summary()
    
    print(f"   Total operations: {perf_summary['total_operations']}")
    print(f"   Cache hit rate: {perf_summary['cache_hit_rate']:.1%}")
    for op, stats in perf_summary['operations'].items():
        print(f"   {op}: {stats['count']} calls, avg {stats['avg_duration']:.2f}s")
    print()
    
    # Example 9: Save and load configuration
    print("9. Saving split configuration...")
    split_config = SplitConfiguration(
        split_type=SplitType.STANDARD,
        test_size=0.2,
        val_size=0.2,
        stratify=True,
        random_state=42
    )
    
    config_path = Path("split_config.yaml")
    if manager.save_split_configuration(split_config, config_path):
        print(f"   Configuration saved to {config_path}")
    
    loaded_config = manager.load_split_configuration(config_path)
    if loaded_config:
        print(f"   Configuration loaded: split_type={loaded_config.split_type.value}")
    
    # Cleanup
    config_path.unlink(missing_ok=True)
    
    print("\n=== All examples completed successfully! ===")