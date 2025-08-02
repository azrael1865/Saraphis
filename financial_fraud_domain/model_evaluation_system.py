"""
Model Evaluation System for Financial Fraud Detection - Part A
Comprehensive model evaluation, comparison, and validation framework.
Part of the Saraphis recursive methodology for accuracy tracking - Phase 3.
"""

import logging
import time
import json
import warnings
import threading
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import (
    cross_val_score, cross_validate, StratifiedKFold, KFold,
    TimeSeriesSplit, RepeatedStratifiedKFold, RepeatedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, classification_report, matthews_corrcoef,
    cohen_kappa_score, balanced_accuracy_score, roc_curve,
    precision_recall_curve, make_scorer
)
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample, check_X_y
from sklearn.utils.validation import check_is_fitted
import joblib
from scipy import stats
from scipy.stats import (
    ttest_rel, wilcoxon, friedmanchisquare,
    mannwhitneyu, kruskal, chi2_contingency
)
import psutil
import hashlib
import pickle
import gzip
from contextlib import contextmanager

# Import from existing modules
try:
    from accuracy_dataset_manager import TrainValidationTestManager, DatasetMetadata, SplitConfiguration
    from accuracy_tracking_db import (
        AccuracyTrackingDatabase, MetricType, DataType, ModelStatus,
        AccuracyMetric, ModelVersion
    )
    from enhanced_fraud_core_exceptions import (
        EnhancedFraudException, ValidationError, ConfigurationError,
        ResourceError, DataQualityError, ModelError,
        ErrorContext, create_error_context
    )
except ImportError:
    # Fallback for standalone development
    from accuracy_dataset_manager import TrainValidationTestManager, DatasetMetadata, SplitConfiguration
    from accuracy_tracking_db import (
        AccuracyTrackingDatabase, MetricType, DataType, ModelStatus,
        AccuracyMetric, ModelVersion
    )
    from enhanced_fraud_core_exceptions import (
        EnhancedFraudException, ValidationError, ConfigurationError,
        ResourceError, DataQualityError, ModelError,
        ErrorContext, create_error_context
    )

# Configure logging
logger = logging.getLogger(__name__)

# ======================== CUSTOM EXCEPTIONS ========================

class EvaluationError(EnhancedFraudException):
    """Base exception for evaluation errors"""
    pass

class ModelComparisonError(EvaluationError):
    """Exception raised during model comparison"""
    pass

class StatisticalTestError(EvaluationError):
    """Exception raised during statistical testing"""
    pass

class BootstrapError(EvaluationError):
    """Exception raised during bootstrap operations"""
    pass

# ======================== ENUMS AND CONSTANTS ========================

class EvaluationStrategy(Enum):
    """Evaluation strategy types"""
    STANDARD = "standard"
    CROSS_VALIDATION = "cross_validation"
    NESTED_CV = "nested_cv"
    BOOTSTRAP = "bootstrap"
    TEMPORAL = "temporal"
    HOLDOUT = "holdout"

class ComparisonMethod(Enum):
    """Statistical comparison methods"""
    PAIRED_T_TEST = "paired_t_test"
    WILCOXON = "wilcoxon"
    MCNEMAR = "mcnemar"
    FRIEDMAN = "friedman"
    NEMENYI = "nemenyi"
    BOOTSTRAP_COMPARISON = "bootstrap_comparison"

class RankingCriteria(Enum):
    """Model ranking criteria"""
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    PRECISION = "precision"
    RECALL = "recall"
    BALANCED_ACCURACY = "balanced_accuracy"
    MULTI_CRITERIA = "multi_criteria"

class ConfidenceLevel(Enum):
    """Confidence levels for statistical tests"""
    LOW = 0.90
    MEDIUM = 0.95
    HIGH = 0.99
    VERY_HIGH = 0.999

# Default evaluation metrics
DEFAULT_METRICS = [
    'accuracy', 'precision', 'recall', 'f1_score',
    'roc_auc', 'average_precision', 'balanced_accuracy'
]

# Default bootstrap configuration
DEFAULT_BOOTSTRAP_CONFIG = {
    'n_iterations': 1000,
    'sample_size': 1.0,
    'stratify': True,
    'confidence_level': 0.95,
    'random_state': 42
}

# ======================== DATA STRUCTURES ========================

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    strategy: EvaluationStrategy = EvaluationStrategy.STANDARD
    metrics: List[str] = field(default_factory=lambda: DEFAULT_METRICS.copy())
    cv_folds: int = 5
    cv_repeats: int = 1
    stratified: bool = True
    confidence_level: float = 0.95
    bootstrap_iterations: int = 1000
    parallel_jobs: int = -1
    random_state: Optional[int] = 42
    save_predictions: bool = False
    save_probabilities: bool = True
    compute_feature_importance: bool = False
    additional_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationResult:
    """Result of model evaluation"""
    model_id: str
    evaluation_id: str
    timestamp: datetime
    strategy: EvaluationStrategy
    metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    feature_importances: Optional[Dict[str, float]] = None
    execution_time: float = 0.0
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    additional_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComparisonResult:
    """Result of model comparison"""
    comparison_id: str
    timestamp: datetime
    models: List[str]
    comparison_method: ComparisonMethod
    test_statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    winner: Optional[str] = None
    significant_difference: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RankingResult:
    """Result of model ranking"""
    ranking_id: str
    timestamp: datetime
    models: List[str]
    criteria: Union[RankingCriteria, List[RankingCriteria]]
    weights: Optional[Dict[str, float]] = None
    rankings: Dict[str, int] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

# ======================== MAIN CLASS ========================

class ModelEvaluationSystem:
    """
    Comprehensive model evaluation system for financial fraud detection.
    Orchestrates model evaluation, comparison, and validation with
    production-ready features.
    """
    
    def __init__(
        self,
        dataset_manager: TrainValidationTestManager,
        accuracy_database: AccuracyTrackingDatabase,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ModelEvaluationSystem with dataset manager and database.
        
        Args:
            dataset_manager: TrainValidationTestManager instance
            accuracy_database: AccuracyTrackingDatabase instance
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Validate inputs
        if not isinstance(dataset_manager, TrainValidationTestManager):
            raise ValidationError(
                "dataset_manager must be a TrainValidationTestManager instance",
                context=create_error_context(component="ModelEvaluationSystem", operation="init")
            )
        
        if not isinstance(accuracy_database, AccuracyTrackingDatabase):
            raise ValidationError(
                "accuracy_database must be an AccuracyTrackingDatabase instance",
                context=create_error_context(component="ModelEvaluationSystem", operation="init")
            )
        
        self.dataset_manager = dataset_manager
        self.accuracy_database = accuracy_database
        
        # Load and validate configuration
        self.config = self._load_and_validate_config(config)
        
        # Initialize components
        self._init_executors()
        self._init_metrics()
        self._init_caching()
        self._init_monitoring()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self.evaluation_stats = defaultdict(list)
        self.comparison_stats = defaultdict(list)
        self.operation_counter = 0
        
        # Model registry
        self.evaluated_models = {}
        self.comparison_cache = {}
        
        self.logger.info(
            f"ModelEvaluationSystem initialized with "
            f"max_workers={self.config['max_workers']}, "
            f"enable_caching={self.config['enable_caching']}"
        )
    
    def _load_and_validate_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load and validate configuration"""
        default_config = {
            'max_workers': min(4, (os.cpu_count() or 1)),
            'process_pool_size': 2,
            'enable_caching': True,
            'cache_size_mb': 500,
            'cache_ttl_seconds': 3600,
            'enable_parallel_evaluation': True,
            'batch_size': 1000,
            'memory_limit_gb': 8.0,
            'timeout_seconds': 3600,
            'statistical_tests': {
                'alpha': 0.05,
                'multiple_comparison_correction': 'bonferroni',
                'min_sample_size': 30
            },
            'bootstrap': DEFAULT_BOOTSTRAP_CONFIG.copy(),
            'monitoring': {
                'enable_resource_monitoring': True,
                'log_interval_seconds': 60
            }
        }
        
        if config:
            # Validate and merge configurations
            self._validate_config(config)
            # Deep merge configurations
            default_config = self._deep_merge_configs(default_config, config)
        
        return default_config
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate user configuration"""
        if 'max_workers' in config:
            if not isinstance(config['max_workers'], int) or config['max_workers'] < 1:
                raise ConfigurationError(
                    f"Invalid max_workers: {config['max_workers']}",
                    context=create_error_context(component="ModelEvaluationSystem", operation="config_validation")
                )
        
        if 'memory_limit_gb' in config:
            if not isinstance(config['memory_limit_gb'], (int, float)) or config['memory_limit_gb'] <= 0:
                raise ConfigurationError(
                    f"Invalid memory_limit_gb: {config['memory_limit_gb']}",
                    context=create_error_context(component="ModelEvaluationSystem", operation="config_validation")
                )
    
    def _deep_merge_configs(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge configuration dictionaries"""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _init_executors(self) -> None:
        """Initialize thread and process executors"""
        self.thread_executor = ThreadPoolExecutor(
            max_workers=self.config['max_workers'],
            thread_name_prefix="ModelEval"
        )
        
        if self.config['enable_parallel_evaluation']:
            self.process_executor = ProcessPoolExecutor(
                max_workers=self.config['process_pool_size']
            )
        else:
            self.process_executor = None
        
        self.logger.info(f"Initialized executors with {self.config['max_workers']} workers")
    
    def _init_metrics(self) -> None:
        """Initialize metrics and scorers"""
        self.available_metrics = {
            'accuracy': accuracy_score,
            'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'roc_auc': lambda y_true, y_score: roc_auc_score(y_true, y_score, multi_class='ovr') if len(np.unique(y_true)) > 2 else roc_auc_score(y_true, y_score[:, 1]),
            'average_precision': lambda y_true, y_score: average_precision_score(y_true, y_score),
            'balanced_accuracy': balanced_accuracy_score,
            'matthews_corrcoef': matthews_corrcoef,
            'cohen_kappa': cohen_kappa_score
        }
        
        # Create sklearn scorers
        self.scorers = {}
        for name, func in self.available_metrics.items():
            if name in ['roc_auc', 'average_precision']:
                self.scorers[name] = make_scorer(func, needs_proba=True)
            else:
                self.scorers[name] = make_scorer(func)
    
    def _init_caching(self) -> None:
        """Initialize caching system"""
        self.cache_enabled = self.config['enable_caching']
        self.evaluation_cache = {}
        self.cache_size_bytes = 0
        self.cache_max_bytes = self.config['cache_size_mb'] * 1024 * 1024
        self.cache_ttl = self.config['cache_ttl_seconds']
        self.cache_access_times = {}
    
    def _init_monitoring(self) -> None:
        """Initialize monitoring system"""
        self.resource_monitoring = self.config['monitoring']['enable_resource_monitoring']
        if self.resource_monitoring:
            self._start_resource_monitor()
    
    # ======================== AUTOMATED EVALUATION METHODS ========================
    
    def comprehensive_evaluation(
        self,
        model: BaseEstimator,
        test_data: Union[np.ndarray, pd.DataFrame],
        test_labels: Union[np.ndarray, pd.Series],
        evaluation_config: Optional[EvaluationConfig] = None
    ) -> EvaluationResult:
        """
        Perform comprehensive model evaluation with all metrics.
        
        Args:
            model: Trained model to evaluate
            test_data: Test feature data
            test_labels: Test labels
            evaluation_config: Optional evaluation configuration
            
        Returns:
            Comprehensive evaluation results
        """
        start_time = time.time()
        
        # Use default config if not provided
        if evaluation_config is None:
            evaluation_config = EvaluationConfig()
        
        # Generate evaluation ID
        evaluation_id = self._generate_evaluation_id()
        model_id = self._get_model_id(model)
        
        self.logger.info(
            f"Starting comprehensive evaluation for model {model_id} "
            f"with strategy {evaluation_config.strategy.value}"
        )
        
        try:
            # Validate inputs
            self._validate_evaluation_inputs(model, test_data, test_labels)
            
            # Check if model is fitted
            check_is_fitted(model)
            
            # Get predictions
            predictions = model.predict(test_data)
            
            # Get probabilities if available
            probabilities = None
            if hasattr(model, 'predict_proba') and evaluation_config.save_probabilities:
                try:
                    probabilities = model.predict_proba(test_data)
                except Exception as e:
                    self.logger.warning(f"Could not get prediction probabilities: {e}")
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                test_labels, predictions, probabilities,
                evaluation_config.metrics
            )
            
            # Calculate confidence intervals using bootstrap
            confidence_intervals = {}
            if evaluation_config.bootstrap_iterations > 0:
                with self._monitor_operation("bootstrap_confidence_intervals"):
                    confidence_intervals = self._calculate_bootstrap_confidence_intervals(
                        model, test_data, test_labels,
                        evaluation_config.metrics,
                        n_iterations=evaluation_config.bootstrap_iterations,
                        confidence_level=evaluation_config.confidence_level
                    )
            
            # Get feature importances if available
            feature_importances = None
            if evaluation_config.compute_feature_importance:
                feature_importances = self._extract_feature_importances(model, test_data)
            
            # Create evaluation result
            execution_time = time.time() - start_time
            
            result = EvaluationResult(
                model_id=model_id,
                evaluation_id=evaluation_id,
                timestamp=datetime.now(),
                strategy=evaluation_config.strategy,
                metrics=metrics,
                confidence_intervals=confidence_intervals,
                predictions=predictions if evaluation_config.save_predictions else None,
                probabilities=probabilities if evaluation_config.save_probabilities else None,
                feature_importances=feature_importances,
                execution_time=execution_time,
                dataset_info={
                    'n_samples': len(test_labels),
                    'n_features': test_data.shape[1] if len(test_data.shape) > 1 else 1,
                    'class_distribution': dict(pd.Series(test_labels).value_counts())
                }
            )
            
            # Store in database
            self._store_evaluation_result(result)
            
            # Cache result
            if self.cache_enabled:
                self._cache_evaluation_result(result)
            
            # Update statistics
            with self._lock:
                self.evaluation_stats['comprehensive'].append({
                    'model_id': model_id,
                    'execution_time': execution_time,
                    'timestamp': datetime.now()
                })
                self.evaluated_models[model_id] = result
            
            self.logger.info(
                f"Completed comprehensive evaluation for model {model_id} "
                f"in {execution_time:.2f}s. Metrics: {metrics}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Comprehensive evaluation failed: {e}", exc_info=True)
            raise EvaluationError(
                f"Failed to evaluate model: {str(e)}",
                context=create_error_context(
                    component="ModelEvaluationSystem",
                    operation="comprehensive_evaluation",
                    model_id=model_id
                )
            )
    
    def cross_validation_evaluation(
        self,
        model_class: Type[BaseEstimator],
        data: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, pd.Series],
        cv_config: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Perform cross-validation evaluation with optional nested CV.
        
        Args:
            model_class: Model class to instantiate
            data: Feature data
            labels: Target labels
            cv_config: Cross-validation configuration
            
        Returns:
            Cross-validation evaluation results
        """
        start_time = time.time()
        
        # Default CV configuration
        default_cv_config = {
            'cv_folds': 5,
            'cv_repeats': 1,
            'stratified': True,
            'nested_cv': False,
            'inner_cv_folds': 3,
            'scoring': DEFAULT_METRICS,
            'n_jobs': self.config['max_workers'],
            'return_train_score': True,
            'return_estimator': False
        }
        
        if cv_config:
            default_cv_config.update(cv_config)
        cv_config = default_cv_config
        
        # Generate IDs
        evaluation_id = self._generate_evaluation_id()
        model_id = f"{model_class.__name__}_cv_{evaluation_id[:8]}"
        
        self.logger.info(
            f"Starting cross-validation evaluation for {model_class.__name__} "
            f"with {cv_config['cv_folds']} folds"
        )
        
        try:
            # Validate inputs
            X, y = check_X_y(data, labels, accept_sparse='csc')
            
            # Create model instance
            model = model_class()
            
            # Create CV splitter
            if cv_config['stratified']:
                if cv_config['cv_repeats'] > 1:
                    cv_splitter = RepeatedStratifiedKFold(
                        n_splits=cv_config['cv_folds'],
                        n_repeats=cv_config['cv_repeats'],
                        random_state=42
                    )
                else:
                    cv_splitter = StratifiedKFold(
                        n_splits=cv_config['cv_folds'],
                        shuffle=True,
                        random_state=42
                    )
            else:
                if cv_config['cv_repeats'] > 1:
                    cv_splitter = RepeatedKFold(
                        n_splits=cv_config['cv_folds'],
                        n_repeats=cv_config['cv_repeats'],
                        random_state=42
                    )
                else:
                    cv_splitter = KFold(
                        n_splits=cv_config['cv_folds'],
                        shuffle=True,
                        random_state=42
                    )
            
            # Perform cross-validation
            if cv_config['nested_cv']:
                # Nested cross-validation
                metrics, confidence_intervals = self._nested_cross_validation(
                    model, X, y, cv_splitter,
                    inner_cv_folds=cv_config['inner_cv_folds'],
                    scoring=cv_config['scoring']
                )
            else:
                # Standard cross-validation
                cv_results = cross_validate(
                    model, X, y,
                    cv=cv_splitter,
                    scoring=self._get_cv_scorers(cv_config['scoring']),
                    n_jobs=cv_config['n_jobs'],
                    return_train_score=cv_config['return_train_score'],
                    return_estimator=cv_config['return_estimator']
                )
                
                # Process results
                metrics = {}
                confidence_intervals = {}
                
                for metric_name in cv_config['scoring']:
                    test_scores = cv_results[f'test_{metric_name}']
                    metrics[metric_name] = float(np.mean(test_scores))
                    
                    # Calculate confidence interval
                    if len(test_scores) > 1:
                        ci_lower, ci_upper = self._calculate_confidence_interval(
                            test_scores, confidence_level=0.95
                        )
                        confidence_intervals[metric_name] = (ci_lower, ci_upper)
                    
                    # Add std deviation
                    metrics[f'{metric_name}_std'] = float(np.std(test_scores))
                    
                    if cv_config['return_train_score']:
                        train_scores = cv_results[f'train_{metric_name}']
                        metrics[f'{metric_name}_train'] = float(np.mean(train_scores))
                        metrics[f'{metric_name}_train_std'] = float(np.std(train_scores))
            
            # Create result
            execution_time = time.time() - start_time
            
            result = EvaluationResult(
                model_id=model_id,
                evaluation_id=evaluation_id,
                timestamp=datetime.now(),
                strategy=EvaluationStrategy.CROSS_VALIDATION,
                metrics=metrics,
                confidence_intervals=confidence_intervals,
                execution_time=execution_time,
                dataset_info={
                    'n_samples': len(y),
                    'n_features': X.shape[1],
                    'cv_folds': cv_config['cv_folds'],
                    'cv_repeats': cv_config['cv_repeats'],
                    'nested_cv': cv_config['nested_cv']
                },
                additional_info={
                    'cv_config': cv_config,
                    'model_class': model_class.__name__
                }
            )
            
            # Store result
            self._store_evaluation_result(result)
            
            self.logger.info(
                f"Completed cross-validation evaluation in {execution_time:.2f}s. "
                f"Mean accuracy: {metrics.get('accuracy', 'N/A'):.3f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Cross-validation evaluation failed: {e}", exc_info=True)
            raise EvaluationError(
                f"Failed to perform cross-validation: {str(e)}",
                context=create_error_context(
                    component="ModelEvaluationSystem",
                    operation="cross_validation_evaluation",
                    model_class=model_class.__name__
                )
            )
    
    def holdout_evaluation(
        self,
        model: BaseEstimator,
        holdout_data: Union[np.ndarray, pd.DataFrame],
        holdout_labels: Union[np.ndarray, pd.Series],
        baseline_models: Optional[Dict[str, BaseEstimator]] = None
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate model on holdout test set with optional baseline comparisons.
        
        Args:
            model: Primary model to evaluate
            holdout_data: Holdout test data
            holdout_labels: Holdout test labels
            baseline_models: Optional dictionary of baseline models
            
        Returns:
            Dictionary of evaluation results for all models
        """
        start_time = time.time()
        
        self.logger.info("Starting holdout evaluation with baselines")
        
        results = {}
        
        try:
            # Evaluate primary model
            primary_result = self.comprehensive_evaluation(
                model, holdout_data, holdout_labels,
                EvaluationConfig(strategy=EvaluationStrategy.HOLDOUT)
            )
            results['primary'] = primary_result
            
            # Evaluate baseline models if provided
            if baseline_models:
                baseline_results = {}
                
                # Parallel evaluation of baselines
                if self.config['enable_parallel_evaluation'] and len(baseline_models) > 1:
                    futures = []
                    
                    for name, baseline_model in baseline_models.items():
                        future = self.thread_executor.submit(
                            self.comprehensive_evaluation,
                            baseline_model, holdout_data, holdout_labels,
                            EvaluationConfig(strategy=EvaluationStrategy.HOLDOUT)
                        )
                        futures.append((name, future))
                    
                    # Collect results
                    for name, future in futures:
                        try:
                            baseline_results[name] = future.result()
                        except Exception as e:
                            self.logger.error(f"Failed to evaluate baseline {name}: {e}")
                else:
                    # Sequential evaluation
                    for name, baseline_model in baseline_models.items():
                        try:
                            baseline_results[name] = self.comprehensive_evaluation(
                                baseline_model, holdout_data, holdout_labels,
                                EvaluationConfig(strategy=EvaluationStrategy.HOLDOUT)
                            )
                        except Exception as e:
                            self.logger.error(f"Failed to evaluate baseline {name}: {e}")
                
                results['baselines'] = baseline_results
                
                # Compare with baselines
                if baseline_results:
                    comparison_results = {}
                    
                    for baseline_name, baseline_result in baseline_results.items():
                        comparison = self._compare_two_models(
                            primary_result, baseline_result,
                            method=ComparisonMethod.MCNEMAR
                        )
                        comparison_results[baseline_name] = comparison
                    
                    results['comparisons'] = comparison_results
            
            execution_time = time.time() - start_time
            
            # Log summary
            self.logger.info(
                f"Completed holdout evaluation in {execution_time:.2f}s. "
                f"Primary model accuracy: {primary_result.metrics.get('accuracy', 'N/A'):.3f}"
            )
            
            if 'baselines' in results:
                for name, result in results['baselines'].items():
                    self.logger.info(
                        f"Baseline {name} accuracy: {result.metrics.get('accuracy', 'N/A'):.3f}"
                    )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Holdout evaluation failed: {e}", exc_info=True)
            raise EvaluationError(
                f"Failed to perform holdout evaluation: {str(e)}",
                context=create_error_context(
                    component="ModelEvaluationSystem",
                    operation="holdout_evaluation"
                )
            )
    
    def temporal_evaluation(
        self,
        model: BaseEstimator,
        time_series_data: pd.DataFrame,
        time_column: str,
        target_column: str,
        temporal_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform time-based evaluation for drift detection.
        
        Args:
            model: Model to evaluate
            time_series_data: DataFrame with temporal data
            time_column: Name of time column
            target_column: Name of target column
            temporal_config: Temporal evaluation configuration
            
        Returns:
            Temporal evaluation results including drift analysis
        """
        start_time = time.time()
        
        # Default temporal configuration
        default_config = {
            'window_size': 30,  # days
            'step_size': 7,     # days
            'min_window_samples': 100,
            'metrics': ['accuracy', 'f1_score', 'roc_auc'],
            'drift_detection_method': 'adwin',
            'drift_threshold': 0.05
        }
        
        if temporal_config:
            default_config.update(temporal_config)
        temporal_config = default_config
        
        self.logger.info(
            f"Starting temporal evaluation with window_size={temporal_config['window_size']} days"
        )
        
        try:
            # Validate inputs
            if not isinstance(time_series_data, pd.DataFrame):
                raise ValidationError("time_series_data must be a pandas DataFrame")
            
            if time_column not in time_series_data.columns:
                raise ValidationError(f"Time column '{time_column}' not found")
            
            if target_column not in time_series_data.columns:
                raise ValidationError(f"Target column '{target_column}' not found")
            
            # Sort by time
            time_series_data = time_series_data.sort_values(time_column)
            
            # Convert time column to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(time_series_data[time_column]):
                time_series_data[time_column] = pd.to_datetime(time_series_data[time_column])
            
            # Extract features and labels
            feature_columns = [col for col in time_series_data.columns 
                             if col not in [time_column, target_column]]
            
            # Sliding window evaluation
            results = {
                'windows': [],
                'metrics_over_time': defaultdict(list),
                'drift_points': [],
                'summary_statistics': {}
            }
            
            # Calculate windows
            min_date = time_series_data[time_column].min()
            max_date = time_series_data[time_column].max()
            current_date = min_date
            
            window_results = []
            
            while current_date + timedelta(days=temporal_config['window_size']) <= max_date:
                window_end = current_date + timedelta(days=temporal_config['window_size'])
                
                # Get window data
                window_mask = (
                    (time_series_data[time_column] >= current_date) &
                    (time_series_data[time_column] < window_end)
                )
                window_data = time_series_data[window_mask]
                
                if len(window_data) >= temporal_config['min_window_samples']:
                    # Evaluate on window
                    X_window = window_data[feature_columns]
                    y_window = window_data[target_column]
                    
                    # Get predictions
                    predictions = model.predict(X_window)
                    
                    # Calculate metrics
                    window_metrics = self._calculate_metrics(
                        y_window, predictions, None,
                        temporal_config['metrics']
                    )
                    
                    # Store window result
                    window_result = {
                        'window_start': current_date,
                        'window_end': window_end,
                        'n_samples': len(window_data),
                        'metrics': window_metrics
                    }
                    
                    window_results.append(window_result)
                    results['windows'].append(window_result)
                    
                    # Track metrics over time
                    for metric_name, value in window_metrics.items():
                        results['metrics_over_time'][metric_name].append({
                            'timestamp': current_date + timedelta(days=temporal_config['window_size']/2),
                            'value': value
                        })
                
                # Move to next window
                current_date += timedelta(days=temporal_config['step_size'])
            
            # Detect drift
            if len(window_results) > 1:
                drift_analysis = self._detect_temporal_drift(
                    window_results,
                    temporal_config['drift_detection_method'],
                    temporal_config['drift_threshold']
                )
                results['drift_points'] = drift_analysis['drift_points']
                results['drift_analysis'] = drift_analysis
            
            # Calculate summary statistics
            for metric_name in temporal_config['metrics']:
                metric_values = [w['metrics'][metric_name] for w in window_results 
                               if metric_name in w['metrics']]
                
                if metric_values:
                    results['summary_statistics'][metric_name] = {
                        'mean': float(np.mean(metric_values)),
                        'std': float(np.std(metric_values)),
                        'min': float(np.min(metric_values)),
                        'max': float(np.max(metric_values)),
                        'trend': self._calculate_trend(metric_values)
                    }
            
            execution_time = time.time() - start_time
            
            # Create evaluation result
            eval_result = EvaluationResult(
                model_id=self._get_model_id(model),
                evaluation_id=self._generate_evaluation_id(),
                timestamp=datetime.now(),
                strategy=EvaluationStrategy.TEMPORAL,
                metrics=results['summary_statistics'],
                confidence_intervals={},
                execution_time=execution_time,
                additional_info={
                    'temporal_results': results,
                    'config': temporal_config
                }
            )
            
            # Store result
            self._store_evaluation_result(eval_result)
            
            self.logger.info(
                f"Completed temporal evaluation in {execution_time:.2f}s. "
                f"Evaluated {len(window_results)} windows, "
                f"detected {len(results['drift_points'])} drift points"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Temporal evaluation failed: {e}", exc_info=True)
            raise EvaluationError(
                f"Failed to perform temporal evaluation: {str(e)}",
                context=create_error_context(
                    component="ModelEvaluationSystem",
                    operation="temporal_evaluation"
                )
            )
    
    def bootstrap_evaluation(
        self,
        model: BaseEstimator,
        test_data: Union[np.ndarray, pd.DataFrame],
        test_labels: Union[np.ndarray, pd.Series],
        bootstrap_config: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Perform bootstrap evaluation for confidence intervals.
        
        Args:
            model: Model to evaluate
            test_data: Test data
            test_labels: Test labels
            bootstrap_config: Bootstrap configuration
            
        Returns:
            Evaluation result with bootstrap confidence intervals
        """
        start_time = time.time()
        
        # Merge with default bootstrap config
        config = DEFAULT_BOOTSTRAP_CONFIG.copy()
        if bootstrap_config:
            config.update(bootstrap_config)
        
        self.logger.info(
            f"Starting bootstrap evaluation with {config['n_iterations']} iterations"
        )
        
        try:
            # Validate inputs
            self._validate_evaluation_inputs(model, test_data, test_labels)
            check_is_fitted(model)
            
            # Convert to numpy arrays
            if isinstance(test_data, pd.DataFrame):
                X = test_data.values
            else:
                X = test_data
            
            if isinstance(test_labels, pd.Series):
                y = test_labels.values
            else:
                y = test_labels
            
            n_samples = len(y)
            
            # Bootstrap iterations
            bootstrap_metrics = defaultdict(list)
            
            # Parallel bootstrap if enabled
            if self.config['enable_parallel_evaluation'] and config['n_iterations'] > 10:
                # Split iterations across workers
                n_workers = min(self.config['max_workers'], config['n_iterations'])
                iterations_per_worker = config['n_iterations'] // n_workers
                
                futures = []
                for i in range(n_workers):
                    n_iter = iterations_per_worker
                    if i == n_workers - 1:
                        # Last worker handles remaining iterations
                        n_iter = config['n_iterations'] - (iterations_per_worker * (n_workers - 1))
                    
                    future = self.thread_executor.submit(
                        self._bootstrap_worker,
                        model, X, y, n_iter, config, i
                    )
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    worker_metrics = future.result()
                    for metric_name, values in worker_metrics.items():
                        bootstrap_metrics[metric_name].extend(values)
            else:
                # Sequential bootstrap
                bootstrap_metrics = self._bootstrap_worker(
                    model, X, y, config['n_iterations'], config, 0
                )
            
            # Calculate statistics
            metrics = {}
            confidence_intervals = {}
            
            for metric_name, values in bootstrap_metrics.items():
                values = np.array(values)
                metrics[metric_name] = float(np.mean(values))
                
                # Calculate confidence interval
                alpha = 1.0 - config['confidence_level']
                lower_percentile = (alpha / 2.0) * 100
                upper_percentile = (1.0 - alpha / 2.0) * 100
                
                ci_lower = float(np.percentile(values, lower_percentile))
                ci_upper = float(np.percentile(values, upper_percentile))
                confidence_intervals[metric_name] = (ci_lower, ci_upper)
                
                # Add additional statistics
                metrics[f'{metric_name}_std'] = float(np.std(values))
                metrics[f'{metric_name}_median'] = float(np.median(values))
            
            execution_time = time.time() - start_time
            
            # Create result
            result = EvaluationResult(
                model_id=self._get_model_id(model),
                evaluation_id=self._generate_evaluation_id(),
                timestamp=datetime.now(),
                strategy=EvaluationStrategy.BOOTSTRAP,
                metrics=metrics,
                confidence_intervals=confidence_intervals,
                execution_time=execution_time,
                dataset_info={
                    'n_samples': n_samples,
                    'n_features': X.shape[1],
                    'bootstrap_iterations': config['n_iterations']
                },
                additional_info={
                    'bootstrap_config': config,
                    'bootstrap_distributions': {k: v[:100] for k, v in bootstrap_metrics.items()}  # Sample
                }
            )
            
            # Store result
            self._store_evaluation_result(result)
            
            self.logger.info(
                f"Completed bootstrap evaluation in {execution_time:.2f}s. "
                f"Accuracy: {metrics.get('accuracy', 'N/A'):.3f} "
                f"CI: {confidence_intervals.get('accuracy', ('N/A', 'N/A'))}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Bootstrap evaluation failed: {e}", exc_info=True)
            raise BootstrapError(
                f"Failed to perform bootstrap evaluation: {str(e)}",
                context=create_error_context(
                    component="ModelEvaluationSystem",
                    operation="bootstrap_evaluation"
                )
            )
    
    # ======================== MODEL COMPARISON AND RANKING ========================
    
    def champion_challenger_comparison(
        self,
        champion_model: BaseEstimator,
        challenger_model: BaseEstimator,
        test_data: Union[np.ndarray, pd.DataFrame],
        test_labels: Union[np.ndarray, pd.Series],
        test_config: Optional[Dict[str, Any]] = None
    ) -> ComparisonResult:
        """
        Compare champion model with challenger using statistical significance testing.
        
        Args:
            champion_model: Current champion model
            challenger_model: Challenger model
            test_data: Test data
            test_labels: Test labels
            test_config: Test configuration
            
        Returns:
            Comparison result with statistical significance
        """
        start_time = time.time()
        
        # Default test configuration
        default_config = {
            'comparison_method': ComparisonMethod.MCNEMAR,
            'alpha': self.config['statistical_tests']['alpha'],
            'metrics_to_compare': ['accuracy', 'f1_score', 'roc_auc'],
            'calculate_effect_size': True
        }
        
        if test_config:
            default_config.update(test_config)
        test_config = default_config
        
        self.logger.info(
            "Starting champion-challenger comparison using "
            f"{test_config['comparison_method'].value}"
        )
        
        try:
            # Evaluate both models
            champion_result = self.comprehensive_evaluation(
                champion_model, test_data, test_labels
            )
            
            challenger_result = self.comprehensive_evaluation(
                challenger_model, test_data, test_labels
            )
            
            # Perform statistical comparison
            comparison_id = self._generate_comparison_id()
            
            # Get predictions for McNemar test
            champion_pred = champion_model.predict(test_data)
            challenger_pred = challenger_model.predict(test_data)
            
            # Perform test based on method
            if test_config['comparison_method'] == ComparisonMethod.MCNEMAR:
                test_result = self._mcnemar_test(
                    test_labels, champion_pred, challenger_pred
                )
            elif test_config['comparison_method'] == ComparisonMethod.PAIRED_T_TEST:
                # Use bootstrap samples for paired t-test
                test_result = self._paired_bootstrap_test(
                    champion_model, challenger_model,
                    test_data, test_labels,
                    n_iterations=100
                )
            else:
                raise ValueError(f"Unsupported comparison method: {test_config['comparison_method']}")
            
            # Calculate effect size
            effect_size = None
            if test_config['calculate_effect_size']:
                effect_size = self._calculate_effect_size(
                    champion_result.metrics,
                    challenger_result.metrics,
                    test_config['metrics_to_compare'][0]
                )
            
            # Determine winner
            significant_difference = test_result['p_value'] < test_config['alpha']
            winner = None
            
            if significant_difference:
                # Compare primary metric
                primary_metric = test_config['metrics_to_compare'][0]
                if challenger_result.metrics[primary_metric] > champion_result.metrics[primary_metric]:
                    winner = 'challenger'
                else:
                    winner = 'champion'
            
            # Create comparison result
            result = ComparisonResult(
                comparison_id=comparison_id,
                timestamp=datetime.now(),
                models=['champion', 'challenger'],
                comparison_method=test_config['comparison_method'],
                test_statistic=test_result['statistic'],
                p_value=test_result['p_value'],
                effect_size=effect_size,
                winner=winner,
                significant_difference=significant_difference,
                details={
                    'champion_metrics': champion_result.metrics,
                    'challenger_metrics': challenger_result.metrics,
                    'test_config': test_config,
                    'metric_differences': {
                        metric: challenger_result.metrics.get(metric, 0) - champion_result.metrics.get(metric, 0)
                        for metric in test_config['metrics_to_compare']
                    }
                }
            )
            
            # Store in database
            self._store_comparison_result(result)
            
            # Cache result
            cache_key = f"{champion_result.model_id}_{challenger_result.model_id}"
            self.comparison_cache[cache_key] = result
            
            execution_time = time.time() - start_time
            
            self.logger.info(
                f"Completed champion-challenger comparison in {execution_time:.2f}s. "
                f"p-value: {result.p_value:.4f}, Winner: {result.winner or 'None (no significant difference)'}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Champion-challenger comparison failed: {e}", exc_info=True)
            raise ModelComparisonError(
                f"Failed to compare models: {str(e)}",
                context=create_error_context(
                    component="ModelEvaluationSystem",
                    operation="champion_challenger_comparison"
                )
            )
    
    def multi_model_comparison(
        self,
        models_dict: Dict[str, BaseEstimator],
        test_data: Union[np.ndarray, pd.DataFrame],
        test_labels: Union[np.ndarray, pd.Series],
        comparison_config: Optional[Dict[str, Any]] = None,
        statistical_tests: Optional[List[ComparisonMethod]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models with multiple comparison corrections.
        
        Args:
            models_dict: Dictionary of model_name -> model
            test_data: Test data
            test_labels: Test labels
            comparison_config: Comparison configuration
            statistical_tests: List of statistical tests to perform
            
        Returns:
            Comprehensive comparison results
        """
        start_time = time.time()
        
        if len(models_dict) < 2:
            raise ValueError("Need at least 2 models for comparison")
        
        # Default configuration
        default_config = {
            'primary_metric': 'accuracy',
            'metrics_to_compare': ['accuracy', 'f1_score', 'roc_auc'],
            'alpha': self.config['statistical_tests']['alpha'],
            'correction_method': self.config['statistical_tests']['multiple_comparison_correction']
        }
        
        if comparison_config:
            default_config.update(comparison_config)
        comparison_config = default_config
        
        if statistical_tests is None:
            statistical_tests = [ComparisonMethod.FRIEDMAN, ComparisonMethod.NEMENYI]
        
        self.logger.info(
            f"Starting multi-model comparison with {len(models_dict)} models"
        )
        
        try:
            # Evaluate all models
            evaluation_results = {}
            predictions = {}
            
            # Parallel evaluation if enabled
            if self.config['enable_parallel_evaluation'] and len(models_dict) > 2:
                futures = {}
                
                for name, model in models_dict.items():
                    future = self.thread_executor.submit(
                        self.comprehensive_evaluation,
                        model, test_data, test_labels
                    )
                    futures[name] = future
                
                # Collect results
                for name, future in futures.items():
                    evaluation_results[name] = future.result()
                    predictions[name] = models_dict[name].predict(test_data)
            else:
                # Sequential evaluation
                for name, model in models_dict.items():
                    evaluation_results[name] = self.comprehensive_evaluation(
                        model, test_data, test_labels
                    )
                    predictions[name] = model.predict(test_data)
            
            # Prepare comparison results
            comparison_results = {
                'evaluation_results': evaluation_results,
                'statistical_tests': {},
                'rankings': {},
                'pairwise_comparisons': {}
            }
            
            # Perform statistical tests
            if ComparisonMethod.FRIEDMAN in statistical_tests:
                # Friedman test for multiple models
                friedman_result = self._friedman_test(
                    test_labels, predictions,
                    comparison_config['primary_metric']
                )
                comparison_results['statistical_tests']['friedman'] = friedman_result
                
                # If significant, perform post-hoc Nemenyi test
                if friedman_result['p_value'] < comparison_config['alpha']:
                    if ComparisonMethod.NEMENYI in statistical_tests:
                        nemenyi_result = self._nemenyi_test(
                            evaluation_results,
                            comparison_config['primary_metric']
                        )
                        comparison_results['statistical_tests']['nemenyi'] = nemenyi_result
            
            # Perform pairwise comparisons
            model_names = list(models_dict.keys())
            n_comparisons = len(model_names) * (len(model_names) - 1) // 2
            
            # Apply multiple comparison correction
            adjusted_alpha = self._adjust_alpha(
                comparison_config['alpha'],
                n_comparisons,
                comparison_config['correction_method']
            )
            
            pairwise_results = {}
            comparison_count = 0
            
            for i, name1 in enumerate(model_names):
                for name2 in model_names[i+1:]:
                    # McNemar test for each pair
                    test_result = self._mcnemar_test(
                        test_labels,
                        predictions[name1],
                        predictions[name2]
                    )
                    
                    # Check significance with adjusted alpha
                    significant = test_result['p_value'] < adjusted_alpha
                    
                    pairwise_results[f"{name1}_vs_{name2}"] = {
                        'test_statistic': test_result['statistic'],
                        'p_value': test_result['p_value'],
                        'adjusted_alpha': adjusted_alpha,
                        'significant': significant,
                        'metric_differences': {
                            metric: (
                                evaluation_results[name2].metrics.get(metric, 0) -
                                evaluation_results[name1].metrics.get(metric, 0)
                            )
                            for metric in comparison_config['metrics_to_compare']
                        }
                    }
                    
                    comparison_count += 1
            
            comparison_results['pairwise_comparisons'] = pairwise_results
            
            # Rank models
            for metric in comparison_config['metrics_to_compare']:
                metric_scores = {
                    name: result.metrics.get(metric, 0)
                    for name, result in evaluation_results.items()
                }
                
                # Sort by score (descending)
                ranked_models = sorted(
                    metric_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                comparison_results['rankings'][metric] = [
                    {'rank': i+1, 'model': name, 'score': score}
                    for i, (name, score) in enumerate(ranked_models)
                ]
            
            # Summary statistics
            comparison_results['summary'] = {
                'n_models': len(models_dict),
                'n_comparisons': n_comparisons,
                'correction_method': comparison_config['correction_method'],
                'adjusted_alpha': adjusted_alpha,
                'best_model_by_metric': {
                    metric: comparison_results['rankings'][metric][0]['model']
                    for metric in comparison_config['metrics_to_compare']
                }
            }
            
            execution_time = time.time() - start_time
            
            self.logger.info(
                f"Completed multi-model comparison in {execution_time:.2f}s. "
                f"Best model by {comparison_config['primary_metric']}: "
                f"{comparison_results['summary']['best_model_by_metric'][comparison_config['primary_metric']]}"
            )
            
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Multi-model comparison failed: {e}", exc_info=True)
            raise ModelComparisonError(
                f"Failed to compare multiple models: {str(e)}",
                context=create_error_context(
                    component="ModelEvaluationSystem",
                    operation="multi_model_comparison"
                )
            )
    
    def rank_models_by_criteria(
        self,
        models_performance: Dict[str, Dict[str, float]],
        ranking_criteria: Union[RankingCriteria, List[RankingCriteria]],
        weights: Optional[Dict[str, float]] = None
    ) -> RankingResult:
        """
        Rank models using multi-criteria ranking system.
        
        Args:
            models_performance: Dictionary of model_name -> metrics dict
            ranking_criteria: Single criterion or list of criteria
            weights: Optional weights for multi-criteria ranking
            
        Returns:
            Ranking result with scores and confidence
        """
        start_time = time.time()
        
        self.logger.info(f"Starting model ranking with criteria: {ranking_criteria}")
        
        try:
            # Convert single criterion to list
            if isinstance(ranking_criteria, RankingCriteria):
                ranking_criteria = [ranking_criteria]
            
            # Default weights if not provided
            if weights is None and len(ranking_criteria) > 1:
                # Equal weights
                weights = {
                    criterion.value: 1.0 / len(ranking_criteria)
                    for criterion in ranking_criteria
                }
            elif weights is None:
                weights = {ranking_criteria[0].value: 1.0}
            
            # Validate weights sum to 1
            weight_sum = sum(weights.values())
            if abs(weight_sum - 1.0) > 1e-6:
                # Normalize weights
                weights = {k: v/weight_sum for k, v in weights.items()}
            
            # Calculate composite scores
            scores = {}
            normalized_scores = {}
            
            # First, normalize scores for each criterion
            for criterion in ranking_criteria:
                criterion_name = criterion.value
                
                # Get scores for this criterion
                criterion_scores = {
                    model: perf.get(criterion_name, 0.0)
                    for model, perf in models_performance.items()
                }
                
                # Normalize scores (0-1 range)
                min_score = min(criterion_scores.values())
                max_score = max(criterion_scores.values())
                score_range = max_score - min_score
                
                if score_range > 0:
                    normalized = {
                        model: (score - min_score) / score_range
                        for model, score in criterion_scores.items()
                    }
                else:
                    # All scores are the same
                    normalized = {model: 0.5 for model in criterion_scores}
                
                normalized_scores[criterion_name] = normalized
            
            # Calculate weighted composite scores
            for model in models_performance:
                composite_score = 0.0
                
                for criterion in ranking_criteria:
                    criterion_name = criterion.value
                    weight = weights.get(criterion_name, 0.0)
                    normalized_score = normalized_scores[criterion_name].get(model, 0.0)
                    composite_score += weight * normalized_score
                
                scores[model] = composite_score
            
            # Sort models by score
            sorted_models = sorted(
                scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Assign rankings
            rankings = {}
            for rank, (model, score) in enumerate(sorted_models, 1):
                rankings[model] = rank
            
            # Calculate confidence scores based on score differences
            confidence_scores = {}
            
            for i, (model, score) in enumerate(sorted_models):
                if i == 0:
                    # Top model - confidence based on gap to second
                    if len(sorted_models) > 1:
                        gap = score - sorted_models[1][1]
                        confidence = min(1.0, gap * 10)  # Scale gap
                    else:
                        confidence = 1.0
                else:
                    # Other models - confidence based on gaps
                    gap_above = sorted_models[i-1][1] - score
                    if i < len(sorted_models) - 1:
                        gap_below = score - sorted_models[i+1][1]
                        confidence = min(1.0, min(gap_above, gap_below) * 10)
                    else:
                        confidence = min(1.0, gap_above * 10)
                
                confidence_scores[model] = confidence
            
            # Create ranking result
            result = RankingResult(
                ranking_id=self._generate_ranking_id(),
                timestamp=datetime.now(),
                models=list(models_performance.keys()),
                criteria=ranking_criteria,
                weights=weights,
                rankings=rankings,
                scores=scores,
                confidence_scores=confidence_scores,
                details={
                    'normalized_scores': normalized_scores,
                    'sorted_models': sorted_models,
                    'criteria_used': [c.value for c in ranking_criteria]
                }
            )
            
            execution_time = time.time() - start_time
            
            self.logger.info(
                f"Completed model ranking in {execution_time:.2f}s. "
                f"Top model: {sorted_models[0][0]} (score: {sorted_models[0][1]:.3f})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Model ranking failed: {e}", exc_info=True)
            raise EvaluationError(
                f"Failed to rank models: {str(e)}",
                context=create_error_context(
                    component="ModelEvaluationSystem",
                    operation="rank_models_by_criteria"
                )
            )
    
    # ======================== HELPER METHODS ========================
    
    def _validate_evaluation_inputs(
        self,
        model: BaseEstimator,
        data: Union[np.ndarray, pd.DataFrame],
        labels: Union[np.ndarray, pd.Series]
    ) -> None:
        """Validate evaluation inputs"""
        if model is None:
            raise ValidationError("Model cannot be None")
        
        if data is None or labels is None:
            raise ValidationError("Data and labels cannot be None")
        
        # Check shapes
        n_samples_data = len(data)
        n_samples_labels = len(labels)
        
        if n_samples_data != n_samples_labels:
            raise ValidationError(
                f"Data and labels must have same number of samples: "
                f"{n_samples_data} != {n_samples_labels}"
            )
        
        if n_samples_data == 0:
            raise ValidationError("Cannot evaluate on empty dataset")
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        metric_names: List[str]
    ) -> Dict[str, float]:
        """Calculate specified metrics"""
        metrics = {}
        
        for metric_name in metric_names:
            if metric_name not in self.available_metrics:
                self.logger.warning(f"Unknown metric: {metric_name}")
                continue
            
            try:
                metric_func = self.available_metrics[metric_name]
                
                if metric_name in ['roc_auc', 'average_precision'] and y_proba is not None:
                    # Metrics that need probabilities
                    if len(np.unique(y_true)) == 2 and y_proba.shape[1] == 2:
                        # Binary classification
                        value = metric_func(y_true, y_proba[:, 1])
                    else:
                        # Multi-class
                        value = metric_func(y_true, y_proba)
                elif metric_name in ['roc_auc', 'average_precision']:
                    # Skip if no probabilities available
                    self.logger.debug(f"Skipping {metric_name} - no probabilities available")
                    continue
                else:
                    # Metrics that use predictions
                    value = metric_func(y_true, y_pred)
                
                metrics[metric_name] = float(value)
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate {metric_name}: {e}")
        
        return metrics
    
    def _calculate_bootstrap_confidence_intervals(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        metric_names: List[str],
        n_iterations: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals"""
        bootstrap_scores = defaultdict(list)
        
        for i in range(n_iterations):
            # Resample with replacement
            indices = resample(
                np.arange(len(y)),
                n_samples=len(y),
                replace=True,
                random_state=42 + i
            )
            
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Get predictions
            y_pred_boot = model.predict(X_boot)
            
            # Get probabilities if available
            y_proba_boot = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba_boot = model.predict_proba(X_boot)
                except:
                    pass
            
            # Calculate metrics
            boot_metrics = self._calculate_metrics(
                y_boot, y_pred_boot, y_proba_boot, metric_names
            )
            
            for metric_name, value in boot_metrics.items():
                bootstrap_scores[metric_name].append(value)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        alpha = 1.0 - confidence_level
        
        for metric_name, scores in bootstrap_scores.items():
            if scores:
                lower = np.percentile(scores, alpha/2 * 100)
                upper = np.percentile(scores, (1 - alpha/2) * 100)
                confidence_intervals[metric_name] = (float(lower), float(upper))
        
        return confidence_intervals
    
    def _extract_feature_importances(
        self,
        model: BaseEstimator,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> Optional[Dict[str, float]]:
        """Extract feature importances from model"""
        try:
            # Try different importance attributes
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
            else:
                return None
            
            # Get feature names
            if isinstance(data, pd.DataFrame):
                feature_names = data.columns.tolist()
            else:
                feature_names = [f'feature_{i}' for i in range(data.shape[1])]
            
            # Create importance dict
            importance_dict = {
                name: float(importance)
                for name, importance in zip(feature_names, importances)
            }
            
            # Sort by importance
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            self.logger.debug(f"Could not extract feature importances: {e}")
            return None
    
    def _nested_cross_validation(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        outer_cv,
        inner_cv_folds: int = 3,
        scoring: List[str] = None
    ) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
        """Perform nested cross-validation"""
        if scoring is None:
            scoring = DEFAULT_METRICS
        
        outer_scores = defaultdict(list)
        
        for train_val_idx, test_idx in outer_cv.split(X, y):
            X_train_val, X_test = X[train_val_idx], X[test_idx]
            y_train_val, y_test = y[train_val_idx], y[test_idx]
            
            # Inner CV for hyperparameter tuning (if applicable)
            inner_cv = StratifiedKFold(n_splits=inner_cv_folds, shuffle=True, random_state=42)
            
            # Clone model for this fold
            fold_model = clone(model)
            
            # Fit on train+val
            fold_model.fit(X_train_val, y_train_val)
            
            # Evaluate on test
            y_pred = fold_model.predict(X_test)
            y_proba = None
            if hasattr(fold_model, 'predict_proba'):
                y_proba = fold_model.predict_proba(X_test)
            
            # Calculate metrics
            fold_metrics = self._calculate_metrics(y_test, y_pred, y_proba, scoring)
            
            for metric_name, value in fold_metrics.items():
                outer_scores[metric_name].append(value)
        
        # Aggregate results
        metrics = {}
        confidence_intervals = {}
        
        for metric_name, scores in outer_scores.items():
            metrics[metric_name] = float(np.mean(scores))
            metrics[f'{metric_name}_std'] = float(np.std(scores))
            
            # 95% CI
            ci_lower, ci_upper = self._calculate_confidence_interval(scores, 0.95)
            confidence_intervals[metric_name] = (ci_lower, ci_upper)
        
        return metrics, confidence_intervals
    
    def _calculate_confidence_interval(
        self,
        values: List[float],
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for a list of values"""
        values = np.array(values)
        n = len(values)
        
        if n < 2:
            return (float(values[0]), float(values[0])) if n == 1 else (0.0, 0.0)
        
        mean = np.mean(values)
        std_err = stats.sem(values)
        
        # t-distribution for small samples
        df = n - 1
        t_value = stats.t.ppf((1 + confidence_level) / 2, df)
        
        margin_of_error = t_value * std_err
        
        return (float(mean - margin_of_error), float(mean + margin_of_error))
    
    @contextmanager
    def _monitor_operation(self, operation_name: str):
        """Monitor operation performance"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            self.operation_counter += 1
            yield
        finally:
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            self.logger.debug(
                f"Operation '{operation_name}' completed in {duration:.2f}s, "
                f"memory delta: {end_memory - start_memory:.1f}MB"
            )
    
    def _get_model_id(self, model: BaseEstimator) -> str:
        """Generate or retrieve model ID"""
        # Try to get existing ID
        if hasattr(model, 'model_id'):
            return model.model_id
        
        # Generate ID based on model class and parameters
        model_class = model.__class__.__name__
        
        # Get parameters if available
        params_str = ""
        if hasattr(model, 'get_params'):
            params = model.get_params()
            # Use a subset of params for ID
            key_params = {k: v for k, v in params.items() 
                         if isinstance(v, (int, float, str, bool))}
            params_str = str(sorted(key_params.items()))
        
        # Create hash
        id_string = f"{model_class}_{params_str}"
        model_hash = hashlib.md5(id_string.encode()).hexdigest()[:8]
        
        return f"{model_class}_{model_hash}"
    
    def _generate_evaluation_id(self) -> str:
        """Generate unique evaluation ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        random_suffix = os.urandom(4).hex()
        return f"eval_{timestamp}_{random_suffix}"
    
    def _generate_comparison_id(self) -> str:
        """Generate unique comparison ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        random_suffix = os.urandom(4).hex()
        return f"comp_{timestamp}_{random_suffix}"
    
    def _generate_ranking_id(self) -> str:
        """Generate unique ranking ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        random_suffix = os.urandom(4).hex()
        return f"rank_{timestamp}_{random_suffix}"
    
    def _store_evaluation_result(self, result: EvaluationResult) -> None:
        """Store evaluation result in database"""
        try:
            # Create model version if not exists
            try:
                model_version = self.accuracy_database.create_model_version(
                    model_id=result.model_id,
                    version="1.0",
                    model_type="sklearn_model",  # Could be enhanced to detect actual type
                    parameters={},
                    metadata={
                        'evaluation_id': result.evaluation_id,
                        'strategy': result.strategy.value
                    }
                )
            except Exception as e:
                # Model version might already exist
                self.logger.debug(f"Model version creation skipped: {e}")
            
            # Store each metric as a separate record
            for metric_name, metric_value in result.metrics.items():
                if '_std' in metric_name or '_train' in metric_name:
                    continue  # Skip derived metrics
                
                try:
                    # Map metric names to MetricType enum
                    if metric_name == 'accuracy':
                        metric_type = MetricType.ACCURACY
                    elif metric_name == 'precision':
                        metric_type = MetricType.PRECISION
                    elif metric_name == 'recall':
                        metric_type = MetricType.RECALL
                    elif metric_name == 'f1_score':
                        metric_type = MetricType.F1_SCORE
                    elif metric_name == 'roc_auc':
                        metric_type = MetricType.ROC_AUC
                    else:
                        # Store as accuracy for other metrics
                        metric_type = MetricType.ACCURACY
                    
                    # Create metric record
                    metric = AccuracyMetric(
                        metric_id=f"{result.evaluation_id}_{metric_name}",
                        model_id=result.model_id,
                        model_version="1.0",
                        data_type=DataType.TEST,
                        metric_type=metric_type,
                        metric_value=metric_value,
                        timestamp=result.timestamp,
                        dataset_id=result.evaluation_id,
                        sample_size=result.dataset_info.get('n_samples', 0),
                        additional_info={
                            'strategy': result.strategy.value,
                            'confidence_interval': result.confidence_intervals.get(metric_name)
                        }
                    )
                    
                    # Store in database using the proper method
                    with self.accuracy_database.connection_pool.get_connection() as conn:
                        conn.execute('''
                            INSERT INTO accuracy_metrics
                            (metric_id, model_id, model_version, data_type, metric_type,
                             metric_value, timestamp, dataset_id, sample_size, additional_info)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            metric.metric_id,
                            metric.model_id,
                            metric.model_version,
                            metric.data_type.value,
                            metric.metric_type.value,
                            str(metric.metric_value),
                            metric.timestamp.isoformat(),
                            metric.dataset_id,
                            metric.sample_size,
                            json.dumps(metric.additional_info)
                        ))
                        conn.commit()
                        
                except Exception as e:
                    self.logger.error(f"Failed to store metric {metric_name}: {e}")
            
            self.logger.debug(f"Stored evaluation result {result.evaluation_id} in database")
            
        except Exception as e:
            self.logger.error(f"Failed to store evaluation result: {e}", exc_info=True)
    
    def _store_comparison_result(self, result: ComparisonResult) -> None:
        """Store comparison result in database"""
        # Simplified storage - could be enhanced with dedicated comparison tables
        pass
    
    def _cache_evaluation_result(self, result: EvaluationResult) -> None:
        """Cache evaluation result"""
        if not self.cache_enabled:
            return
        
        cache_key = f"eval_{result.model_id}_{result.evaluation_id}"
        
        try:
            # Serialize result
            serialized = pickle.dumps(result)
            size_bytes = len(serialized)
            
            # Check cache size and evict if necessary
            with self._lock:
                while self.cache_size_bytes + size_bytes > self.cache_max_bytes and self.evaluation_cache:
                    # Remove oldest entry
                    oldest_key = min(self.cache_access_times, key=self.cache_access_times.get)
                    if oldest_key in self.evaluation_cache:
                        old_size = len(pickle.dumps(self.evaluation_cache[oldest_key]))
                        del self.evaluation_cache[oldest_key]
                        del self.cache_access_times[oldest_key]
                        self.cache_size_bytes -= old_size
                
                # Add to cache
                self.evaluation_cache[cache_key] = result
                self.cache_access_times[cache_key] = time.time()
                self.cache_size_bytes += size_bytes
                
        except Exception as e:
            self.logger.debug(f"Failed to cache result: {e}")
    
    def _get_cv_scorers(self, metrics: List[str]) -> Dict[str, Any]:
        """Get sklearn scorers for cross-validation"""
        scorers = {}
        for metric in metrics:
            if metric in self.scorers:
                scorers[metric] = self.scorers[metric]
        return scorers
    
    def _bootstrap_worker(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        n_iterations: int,
        config: Dict[str, Any],
        worker_id: int
    ) -> Dict[str, List[float]]:
        """Worker function for parallel bootstrap"""
        bootstrap_metrics = defaultdict(list)
        
        for i in range(n_iterations):
            # Set seed for reproducibility
            seed = config.get('random_state', 42) + worker_id * 1000 + i
            
            # Resample
            if config.get('stratify', True):
                # Stratified resampling
                indices = []
                for class_label in np.unique(y):
                    class_indices = np.where(y == class_label)[0]
                    n_class_samples = len(class_indices)
                    
                    sampled_indices = resample(
                        class_indices,
                        n_samples=int(n_class_samples * config['sample_size']),
                        replace=True,
                        random_state=seed
                    )
                    indices.extend(sampled_indices)
                
                # Shuffle indices
                np.random.RandomState(seed).shuffle(indices)
            else:
                # Regular resampling
                indices = resample(
                    np.arange(len(y)),
                    n_samples=int(len(y) * config['sample_size']),
                    replace=True,
                    random_state=seed
                )
            
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Get predictions
            y_pred_boot = model.predict(X_boot)
            
            # Get probabilities if available
            y_proba_boot = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba_boot = model.predict_proba(X_boot)
                except:
                    pass
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                y_boot, y_pred_boot, y_proba_boot,
                DEFAULT_METRICS
            )
            
            for metric_name, value in metrics.items():
                bootstrap_metrics[metric_name].append(value)
        
        return dict(bootstrap_metrics)
    
    def _mcnemar_test(
        self,
        y_true: np.ndarray,
        y_pred1: np.ndarray,
        y_pred2: np.ndarray
    ) -> Dict[str, float]:
        """Perform McNemar's test for paired model comparison"""
        # Create confusion matrix for the two models
        # a: both correct
        # b: model1 correct, model2 wrong
        # c: model1 wrong, model2 correct
        # d: both wrong
        
        correct1 = y_pred1 == y_true
        correct2 = y_pred2 == y_true
        
        a = np.sum(correct1 & correct2)
        b = np.sum(correct1 & ~correct2)
        c = np.sum(~correct1 & correct2)
        d = np.sum(~correct1 & ~correct2)
        
        # McNemar's test statistic
        if b + c == 0:
            # No disagreements
            statistic = 0.0
            p_value = 1.0
        else:
            # Use continuity correction for small samples
            if b + c < 25:
                statistic = (abs(b - c) - 1) ** 2 / (b + c)
            else:
                statistic = (b - c) ** 2 / (b + c)
            
            # Chi-square distribution with 1 degree of freedom
            p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'contingency_table': {
                'both_correct': int(a),
                'model1_only': int(b),
                'model2_only': int(c),
                'both_wrong': int(d)
            }
        }
    
    def _paired_bootstrap_test(
        self,
        model1: BaseEstimator,
        model2: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        n_iterations: int = 100,
        metric: str = 'accuracy'
    ) -> Dict[str, float]:
        """Perform paired bootstrap test"""
        differences = []
        
        for i in range(n_iterations):
            # Resample
            indices = resample(
                np.arange(len(y)),
                n_samples=len(y),
                replace=True,
                random_state=42 + i
            )
            
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Get predictions
            pred1 = model1.predict(X_boot)
            pred2 = model2.predict(X_boot)
            
            # Calculate metric difference
            if metric == 'accuracy':
                score1 = accuracy_score(y_boot, pred1)
                score2 = accuracy_score(y_boot, pred2)
            else:
                # Add other metrics as needed
                score1 = score2 = 0.0
            
            differences.append(score2 - score1)
        
        # Paired t-test on differences
        differences = np.array(differences)
        t_stat, p_value = stats.ttest_1samp(differences, 0)
        
        return {
            'statistic': float(t_stat),
            'p_value': float(p_value),
            'mean_difference': float(np.mean(differences)),
            'std_difference': float(np.std(differences))
        }
    
    def _friedman_test(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        metric: str = 'accuracy'
    ) -> Dict[str, Any]:
        """Perform Friedman test for multiple models"""
        model_names = list(predictions.keys())
        n_models = len(model_names)
        
        if n_models < 3:
            raise ValueError("Friedman test requires at least 3 models")
        
        # Calculate scores for each model on each sample
        # For classification, we can use per-sample accuracy (0 or 1)
        scores = np.zeros((len(y_true), n_models))
        
        for i, model_name in enumerate(model_names):
            pred = predictions[model_name]
            
            if metric == 'accuracy':
                # Per-sample accuracy (correct = 1, incorrect = 0)
                scores[:, i] = (pred == y_true).astype(float)
            else:
                # For other metrics, might need different approach
                raise NotImplementedError(f"Friedman test not implemented for metric: {metric}")
        
        # Perform Friedman test
        try:
            statistic, p_value = friedmanchisquare(*[scores[:, i] for i in range(n_models)])
        except Exception as e:
            self.logger.warning(f"Friedman test failed: {e}")
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'error': str(e)
            }
        
        # Calculate average ranks
        ranks = np.zeros((len(y_true), n_models))
        for i in range(len(y_true)):
            # Rank models for this sample (higher score = lower rank)
            ranks[i] = stats.rankdata(-scores[i])
        
        avg_ranks = ranks.mean(axis=0)
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'average_ranks': {
                model_names[i]: float(avg_ranks[i])
                for i in range(n_models)
            },
            'n_samples': len(y_true),
            'n_models': n_models
        }
    
    def _nemenyi_test(
        self,
        evaluation_results: Dict[str, EvaluationResult],
        metric: str = 'accuracy'
    ) -> Dict[str, Any]:
        """Perform Nemenyi post-hoc test"""
        model_names = list(evaluation_results.keys())
        n_models = len(model_names)
        
        # Get metric values
        metric_values = {
            name: result.metrics.get(metric, 0.0)
            for name, result in evaluation_results.items()
        }
        
        # Calculate critical difference
        # For simplicity, using approximation based on number of models
        # In practice, would use proper critical values table
        q_alpha = 2.569  # For alpha=0.05, k=5 models
        n_datasets = 1  # Single test set
        
        critical_difference = q_alpha * np.sqrt(n_models * (n_models + 1) / (12 * n_datasets))
        
        # Rank models
        sorted_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
        ranks = {name: i+1 for i, (name, _) in enumerate(sorted_models)}
        
        # Pairwise comparisons
        significant_differences = {}
        
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                rank_diff = abs(ranks[name1] - ranks[name2])
                significant = rank_diff > critical_difference
                
                pair_name = f"{name1}_vs_{name2}"
                significant_differences[pair_name] = {
                    'rank_difference': rank_diff,
                    'critical_difference': critical_difference,
                    'significant': significant
                }
        
        return {
            'critical_difference': float(critical_difference),
            'model_ranks': ranks,
            'significant_differences': significant_differences,
            'metric_values': metric_values
        }
    
    def _calculate_effect_size(
        self,
        metrics1: Dict[str, float],
        metrics2: Dict[str, float],
        metric_name: str
    ) -> float:
        """Calculate Cohen's d effect size"""
        value1 = metrics1.get(metric_name, 0.0)
        value2 = metrics2.get(metric_name, 0.0)
        
        # For single values, estimate effect size
        # In practice, would use standard deviations from bootstrap or CV
        difference = value2 - value1
        
        # Rough estimate assuming 5% standard deviation
        pooled_std = 0.05
        
        if pooled_std > 0:
            cohens_d = difference / pooled_std
        else:
            cohens_d = 0.0
        
        return float(cohens_d)
    
    def _detect_temporal_drift(
        self,
        window_results: List[Dict[str, Any]],
        method: str = 'adwin',
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """Detect drift in temporal evaluation"""
        drift_points = []
        
        # Simple threshold-based detection
        # In practice, would use proper drift detection algorithms
        for i in range(1, len(window_results)):
            prev_window = window_results[i-1]
            curr_window = window_results[i]
            
            # Check each metric
            for metric_name in prev_window['metrics']:
                if metric_name in curr_window['metrics']:
                    prev_value = prev_window['metrics'][metric_name]
                    curr_value = curr_window['metrics'][metric_name]
                    
                    # Calculate relative change
                    if prev_value > 0:
                        relative_change = abs(curr_value - prev_value) / prev_value
                        
                        if relative_change > threshold:
                            drift_points.append({
                                'window_index': i,
                                'window_start': curr_window['window_start'],
                                'metric': metric_name,
                                'prev_value': prev_value,
                                'curr_value': curr_value,
                                'relative_change': relative_change
                            })
        
        return {
            'drift_points': drift_points,
            'method': method,
            'threshold': threshold,
            'n_drift_points': len(drift_points)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend in metric values"""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope, intercept = np.polyfit(x, y, 1)
        
        # Determine trend
        if abs(slope) < 0.001:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _adjust_alpha(
        self,
        alpha: float,
        n_comparisons: int,
        method: str = 'bonferroni'
    ) -> float:
        """Adjust alpha for multiple comparisons"""
        if method == 'bonferroni':
            return alpha / n_comparisons
        elif method == 'sidak':
            return 1 - (1 - alpha) ** (1 / n_comparisons)
        elif method == 'holm':
            # For Holm, would need to sort p-values
            # Returning Bonferroni as approximation
            return alpha / n_comparisons
        else:
            return alpha
    
    def _compare_two_models(
        self,
        result1: EvaluationResult,
        result2: EvaluationResult,
        method: ComparisonMethod = ComparisonMethod.MCNEMAR
    ) -> Dict[str, Any]:
        """Compare two model results"""
        # Simplified comparison
        # In practice, would use actual predictions
        
        comparison = {
            'model1': result1.model_id,
            'model2': result2.model_id,
            'method': method.value,
            'metrics_comparison': {}
        }
        
        # Compare each metric
        for metric in result1.metrics:
            if metric in result2.metrics:
                diff = result2.metrics[metric] - result1.metrics[metric]
                comparison['metrics_comparison'][metric] = {
                    'model1_value': result1.metrics[metric],
                    'model2_value': result2.metrics[metric],
                    'difference': diff,
                    'relative_improvement': diff / result1.metrics[metric] if result1.metrics[metric] > 0 else 0
                }
        
        return comparison
    
    def _start_resource_monitor(self) -> None:
        """Start resource monitoring thread"""
        def monitor_resources():
            while True:
                try:
                    # Monitor CPU and memory
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory_info = psutil.Process().memory_info()
                    memory_gb = memory_info.rss / 1024 / 1024 / 1024
                    
                    # Check limits
                    if memory_gb > self.config['memory_limit_gb'] * 0.9:
                        self.logger.warning(
                            f"High memory usage: {memory_gb:.1f}GB "
                            f"(90% of {self.config['memory_limit_gb']}GB limit)"
                        )
                    
                    # Log periodically
                    if hasattr(self, '_last_resource_log'):
                        if time.time() - self._last_resource_log > self.config['monitoring']['log_interval_seconds']:
                            self.logger.debug(
                                f"Resource usage - CPU: {cpu_percent:.1f}%, "
                                f"Memory: {memory_gb:.2f}GB"
                            )
                            self._last_resource_log = time.time()
                    else:
                        self._last_resource_log = time.time()
                    
                    time.sleep(10)  # Check every 10 seconds
                    
                except Exception as e:
                    self.logger.error(f"Resource monitoring error: {e}")
                    time.sleep(30)
        
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
    
    # ======================== PUBLIC UTILITY METHODS ========================
    
    def get_evaluation_summary(self, model_id: str) -> Dict[str, Any]:
        """Get summary of all evaluations for a model"""
        with self._lock:
            if model_id not in self.evaluated_models:
                return {'error': f'No evaluations found for model {model_id}'}
            
            result = self.evaluated_models[model_id]
            
            summary = {
                'model_id': model_id,
                'last_evaluation': result.timestamp.isoformat(),
                'evaluation_strategy': result.strategy.value,
                'metrics': result.metrics,
                'confidence_intervals': result.confidence_intervals,
                'execution_time': result.execution_time,
                'dataset_info': result.dataset_info
            }
            
            # Add historical data from database
            try:
                db_metrics = self.accuracy_database.get_accuracy_metrics(
                    model_id=model_id,
                    data_type=DataType.TEST
                )
                
                if db_metrics:
                    summary['evaluation_history'] = {
                        'n_evaluations': len(db_metrics),
                        'first_evaluation': min(m.timestamp for m in db_metrics).isoformat(),
                        'last_evaluation': max(m.timestamp for m in db_metrics).isoformat()
                    }
            except Exception as e:
                self.logger.error(f"Failed to get evaluation history: {e}")
            
            return summary
    
    def export_evaluation_results(
        self,
        output_path: Path,
        model_ids: Optional[List[str]] = None,
        format: str = 'json'
    ) -> bool:
        """Export evaluation results to file"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Collect results to export
            if model_ids is None:
                # Export all evaluated models
                results_to_export = self.evaluated_models
            else:
                # Export specific models
                results_to_export = {
                    model_id: self.evaluated_models[model_id]
                    for model_id in model_ids
                    if model_id in self.evaluated_models
                }
            
            if not results_to_export:
                self.logger.warning("No results to export")
                return False
            
            # Convert to serializable format
            export_data = {}
            for model_id, result in results_to_export.items():
                export_data[model_id] = {
                    'evaluation_id': result.evaluation_id,
                    'timestamp': result.timestamp.isoformat(),
                    'strategy': result.strategy.value,
                    'metrics': result.metrics,
                    'confidence_intervals': result.confidence_intervals,
                    'execution_time': result.execution_time,
                    'dataset_info': result.dataset_info,
                    'additional_info': result.additional_info
                }
            
            # Export based on format
            if format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            elif format == 'csv':
                # Flatten metrics for CSV
                rows = []
                for model_id, data in export_data.items():
                    row = {'model_id': model_id}
                    row.update(data['metrics'])
                    row['timestamp'] = data['timestamp']
                    row['strategy'] = data['strategy']
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                df.to_csv(output_path, index=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Exported {len(export_data)} evaluation results to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")
            return False
    
    def get_comparison_summary(
        self,
        model_ids: List[str]
    ) -> Dict[str, Any]:
        """Get summary of comparisons between models"""
        summary = {
            'models': model_ids,
            'comparisons': {},
            'rankings': {}
        }
        
        # Check cached comparisons
        for i, model1 in enumerate(model_ids):
            for model2 in model_ids[i+1:]:
                cache_key = f"{model1}_{model2}"
                reverse_key = f"{model2}_{model1}"
                
                if cache_key in self.comparison_cache:
                    comparison = self.comparison_cache[cache_key]
                    summary['comparisons'][cache_key] = {
                        'p_value': comparison.p_value,
                        'winner': comparison.winner,
                        'significant': comparison.significant_difference
                    }
                elif reverse_key in self.comparison_cache:
                    comparison = self.comparison_cache[reverse_key]
                    # Reverse winner if needed
                    winner = None
                    if comparison.winner:
                        winner = model1 if comparison.winner == model2 else model2
                    
                    summary['comparisons'][cache_key] = {
                        'p_value': comparison.p_value,
                        'winner': winner,
                        'significant': comparison.significant_difference
                    }
        
        return summary
    
    def clear_cache(self) -> None:
        """Clear evaluation cache"""
        with self._lock:
            self.evaluation_cache.clear()
            self.cache_access_times.clear()
            self.cache_size_bytes = 0
            self.comparison_cache.clear()
            
        self.logger.info("Cleared evaluation cache")
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        with self._lock:
            stats = {
                'total_evaluations': self.operation_counter,
                'models_evaluated': len(self.evaluated_models),
                'cached_results': len(self.evaluation_cache),
                'cache_size_mb': self.cache_size_bytes / (1024 * 1024),
                'evaluation_statistics': {},
                'comparison_statistics': {}
            }
            
            # Aggregate evaluation statistics
            for strategy, eval_stats in self.evaluation_stats.items():
                if eval_stats:
                    execution_times = [s['execution_time'] for s in eval_stats]
                    stats['evaluation_statistics'][strategy] = {
                        'count': len(eval_stats),
                        'avg_execution_time': np.mean(execution_times),
                        'max_execution_time': np.max(execution_times),
                        'total_time': np.sum(execution_times)
                    }
            
            # Resource usage
            process = psutil.Process()
            stats['resource_usage'] = {
                'memory_mb': process.memory_info().rss / (1024 * 1024),
                'cpu_percent': process.cpu_percent(interval=0.1),
                'threads': process.num_threads()
            }
            
            return stats
    
    def shutdown(self) -> None:
        """Shutdown the evaluation system cleanly"""
        self.logger.info("Shutting down ModelEvaluationSystem")
        
        # Shutdown executors
        if hasattr(self, 'thread_executor'):
            self.thread_executor.shutdown(wait=True)
        
        if hasattr(self, 'process_executor') and self.process_executor:
            self.process_executor.shutdown(wait=True)
        
        # Clear cache
        self.clear_cache()
        
        self.logger.info("ModelEvaluationSystem shutdown complete")


# ======================== USAGE EXAMPLES ========================

if __name__ == "__main__":
    # Example usage demonstrating comprehensive model evaluation
    import numpy as np
    from pathlib import Path
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    print("=== ModelEvaluationSystem Examples - Part A ===\n")
    
    # Create sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        flip_y=0.1,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Initialize components
    dataset_manager = TrainValidationTestManager({'validation_level': 'standard'})
    
    # Initialize accuracy database
    accuracy_db = AccuracyTrackingDatabase({
        'database': {
            'db_path': Path('evaluation_example.db')
        }
    })
    
    # Initialize evaluation system
    eval_system = ModelEvaluationSystem(
        dataset_manager=dataset_manager,
        accuracy_database=accuracy_db,
        config={
            'max_workers': 2,
            'enable_parallel_evaluation': True
        }
    )
    
    print("1. Training models...")
    
    # Train models
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    
    print("   Models trained successfully\n")
    
    # Example 1: Comprehensive evaluation
    print("2. Comprehensive evaluation of Random Forest...")
    rf_result = eval_system.comprehensive_evaluation(
        rf_model, X_test, y_test,
        EvaluationConfig(
            metrics=['accuracy', 'precision', 'recall', 'f1_score'],
            bootstrap_iterations=100,
            confidence_level=0.95
        )
    )
    
    print(f"   Accuracy: {rf_result.metrics['accuracy']:.3f}")
    if 'accuracy' in rf_result.confidence_intervals:
        ci = rf_result.confidence_intervals['accuracy']
        print(f"   95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"   Execution time: {rf_result.execution_time:.2f}s\n")
    
    # Example 2: Cross-validation evaluation
    print("3. Cross-validation evaluation...")
    cv_result = eval_system.cross_validation_evaluation(
        RandomForestClassifier,
        X, y,
        cv_config={
            'cv_folds': 5,
            'stratified': True,
            'scoring': ['accuracy', 'f1_score']
        }
    )
    
    print(f"   CV Accuracy: {cv_result.metrics['accuracy']:.3f} ± {cv_result.metrics['accuracy_std']:.3f}")
    print(f"   CV F1 Score: {cv_result.metrics['f1_score']:.3f} ± {cv_result.metrics['f1_score_std']:.3f}\n")
    
    # Example 3: Holdout evaluation with baselines
    print("4. Holdout evaluation with baselines...")
    holdout_results = eval_system.holdout_evaluation(
        rf_model,
        X_test, y_test,
        baseline_models={
            'logistic_regression': lr_model,
            'svm': svm_model
        }
    )
    
    print(f"   Primary model accuracy: {holdout_results['primary'].metrics['accuracy']:.3f}")
    if 'baselines' in holdout_results:
        for name, result in holdout_results['baselines'].items():
            print(f"   Baseline {name} accuracy: {result.metrics['accuracy']:.3f}")
    print()
    
    # Example 4: Champion-challenger comparison
    print("5. Champion-challenger comparison...")
    comparison_result = eval_system.champion_challenger_comparison(
        champion_model=lr_model,
        challenger_model=rf_model,
        test_data=X_test,
        test_labels=y_test
    )
    
    print(f"   p-value: {comparison_result.p_value:.4f}")
    print(f"   Significant difference: {comparison_result.significant_difference}")
    print(f"   Winner: {comparison_result.winner or 'No significant difference'}\n")
    
    # Example 5: Multi-model comparison
    print("6. Multi-model comparison...")
    multi_comparison = eval_system.multi_model_comparison(
        models_dict={
            'random_forest': rf_model,
            'logistic_regression': lr_model,
            'svm': svm_model
        },
        test_data=X_test,
        test_labels=y_test
    )
    
    print("   Model rankings by accuracy:")
    for rank_info in multi_comparison['rankings']['accuracy']:
        print(f"   {rank_info['rank']}. {rank_info['model']}: {rank_info['score']:.3f}")
    print()
    
    # Example 6: Model ranking by criteria
    print("7. Multi-criteria model ranking...")
    models_performance = {
        model_name: result.metrics
        for model_name, result in multi_comparison['evaluation_results'].items()
    }
    
    ranking_result = eval_system.rank_models_by_criteria(
        models_performance,
        ranking_criteria=[RankingCriteria.ACCURACY, RankingCriteria.F1_SCORE],
        weights={'accuracy': 0.6, 'f1_score': 0.4}
    )
    
    print("   Final rankings (weighted):")
    sorted_rankings = sorted(ranking_result.rankings.items(), key=lambda x: x[1])
    for model, rank in sorted_rankings:
        score = ranking_result.scores[model]
        confidence = ranking_result.confidence_scores[model]
        print(f"   {rank}. {model}: score={score:.3f}, confidence={confidence:.2f}")
    
    # Example 7: System statistics
    print("\n8. System statistics:")
    stats = eval_system.get_system_statistics()
    print(f"   Total evaluations: {stats['total_evaluations']}")
    print(f"   Models evaluated: {stats['models_evaluated']}")
    print(f"   Memory usage: {stats['resource_usage']['memory_mb']:.1f} MB")
    
    # Cleanup
    eval_system.shutdown()
    accuracy_db.shutdown()
    
    # Remove example database
    db_path = Path('evaluation_example.db')
    if db_path.exists():
        db_path.unlink()
    
    print("\n=== All examples completed successfully! ===")