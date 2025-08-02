"""
Enhanced ML Predictor - Chunk 1: Core ML Framework with Base Classes and Validation
Comprehensive ML framework with advanced validation, error handling, and monitoring
for financial fraud detection systems.
"""

# DISABLE ALL FALLBACKS - Force real components to identify failure points
DISABLE_FALLBACKS = False  # Re-enabled after fixing import issues

import logging
import json
import hashlib
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Type
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
import warnings
import traceback

# Configure logging
logger = logging.getLogger(__name__)

# Import existing components for integration
try:
    from enhanced_preprocessing_integration import (
        PreprocessingIntegrationManager, IntegrationConfig as PreprocessingConfig
    )
    from enhanced_data_validator import EnhancedFinancialDataValidator
    from enhanced_transaction_validator import EnhancedTransactionFieldValidator
    ENHANCED_COMPONENTS = True
except ImportError as e:
    if DISABLE_FALLBACKS:
        raise ImportError(f"FALLBACK DISABLED: Enhanced components import failed: {e}")
    ENHANCED_COMPONENTS = False
    logger.warning(f"Enhanced components not available: {e}")

# Import IEEE data loader for integration
try:
    from ieee_fraud_data_loader import IEEEFraudDataLoader, load_ieee_fraud_data
    IEEE_LOADER_AVAILABLE = True
except ImportError as e:
    if DISABLE_FALLBACKS:
        raise ImportError(f"FALLBACK DISABLED: IEEE loader import failed: {e}")
    IEEE_LOADER_AVAILABLE = False
    logger.warning(f"IEEE fraud data loader not available: {e}")

# ======================== ENHANCED ML EXCEPTION HIERARCHY ========================

class MLPredictorError(Exception):
    """Base exception for ML predictor errors with enhanced context"""
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code or "ML_PREDICTOR_ERROR"
        self.details = details or {}
        self.timestamp = datetime.now()
        self.ml_context = {
            'error_code': self.error_code,
            'timestamp': self.timestamp.isoformat(),
            'recoverable': getattr(self, 'recoverable', True),
            'error_type': self.__class__.__name__
        }

class ModelNotFittedError(MLPredictorError):
    """Raised when trying to use an unfitted model"""
    def __init__(self, message: str = "Model must be fitted before use", **kwargs):
        super().__init__(message, error_code="MODEL_NOT_FITTED", **kwargs)
        self.recoverable = False

class InvalidInputError(MLPredictorError):
    """Raised when input data is invalid"""
    def __init__(self, message: str, input_type: str = None, **kwargs):
        super().__init__(message, error_code="INVALID_INPUT", **kwargs)
        self.input_type = input_type
        self.recoverable = True

class ModelValidationError(MLPredictorError):
    """Raised when model fails validation"""
    def __init__(self, message: str, validation_metric: str = None, **kwargs):
        super().__init__(message, error_code="MODEL_VALIDATION_ERROR", **kwargs)
        self.validation_metric = validation_metric
        self.recoverable = False

class FeatureMismatchError(MLPredictorError):
    """Raised when features don't match expected format"""
    def __init__(self, message: str, expected_features: List[str] = None, 
                 actual_features: List[str] = None, **kwargs):
        super().__init__(message, error_code="FEATURE_MISMATCH", **kwargs)
        self.expected_features = expected_features or []
        self.actual_features = actual_features or []
        self.recoverable = True

class ModelDriftError(MLPredictorError):
    """Raised when model drift is detected"""
    def __init__(self, message: str, drift_score: float = None, 
                 drift_threshold: float = None, **kwargs):
        super().__init__(message, error_code="MODEL_DRIFT", **kwargs)
        self.drift_score = drift_score
        self.drift_threshold = drift_threshold
        self.recoverable = True

class ModelLoadError(MLPredictorError):
    """Raised when model loading fails"""
    def __init__(self, message: str, model_path: str = None, **kwargs):
        super().__init__(message, error_code="MODEL_LOAD_ERROR", **kwargs)
        self.model_path = model_path
        self.recoverable = False

class HyperparameterError(MLPredictorError):
    """Raised when hyperparameters are invalid"""
    def __init__(self, message: str, parameter_name: str = None, 
                 parameter_value: Any = None, **kwargs):
        super().__init__(message, error_code="HYPERPARAMETER_ERROR", **kwargs)
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        self.recoverable = True

class ModelTrainingError(MLPredictorError):
    """Raised when model training fails"""
    def __init__(self, message: str, training_stage: str = None, **kwargs):
        super().__init__(message, error_code="MODEL_TRAINING_ERROR", **kwargs)
        self.training_stage = training_stage
        self.recoverable = False

class PredictionError(MLPredictorError):
    """Raised when prediction fails"""
    def __init__(self, message: str, prediction_stage: str = None, **kwargs):
        super().__init__(message, error_code="PREDICTION_ERROR", **kwargs)
        self.prediction_stage = prediction_stage
        self.recoverable = True

class ModelDeploymentError(MLPredictorError):
    """Raised when model deployment fails"""
    def __init__(self, message: str, deployment_stage: str = None, **kwargs):
        super().__init__(message, error_code="MODEL_DEPLOYMENT_ERROR", **kwargs)
        self.deployment_stage = deployment_stage
        self.recoverable = False

# ======================== ENHANCED VALIDATION DECORATORS ========================

def validate_input(validation_func: Callable) -> Callable:
    """Decorator for comprehensive input validation"""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            try:
                # For fit method: args = (X, y), pass both to validation
                if len(args) >= 2:
                    validation_func(args[0], args[1])
                elif len(args) == 1:
                    validation_func(args[0])
                return func(self, *args, **kwargs)
            except Exception as e:
                logger.error(f"Input validation failed for {func.__name__}: {str(e)}")
                raise InvalidInputError(
                    f"Invalid input for {func.__name__}: {str(e)}",
                    input_type=type(args[0]).__name__ if args else "unknown"
                )
        return wrapper
    return decorator

def handle_errors(fallback_value=None, error_action: str = "raise") -> Callable:
    """Decorator for comprehensive error handling with fallback"""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                logger.debug(f"Error details: {traceback.format_exc()}")
                
                # Update error statistics if available
                if hasattr(self, '_error_stats'):
                    self._error_stats[func.__name__] = self._error_stats.get(func.__name__, 0) + 1
                
                if error_action == "fallback" and fallback_value is not None:
                    logger.warning(f"Using fallback value for {func.__name__}")
                    return fallback_value
                elif error_action == "suppress":
                    logger.warning(f"Suppressing error in {func.__name__}")
                    return None
                else:
                    raise
        return wrapper
    return decorator

def monitor_performance(metric_name: str = None) -> Callable:
    """Decorator for performance monitoring"""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            function_name = metric_name or func.__name__
            
            try:
                result = func(self, *args, **kwargs)
                
                # Record success metrics
                duration = time.time() - start_time
                if hasattr(self, '_performance_metrics'):
                    if function_name not in self._performance_metrics:
                        self._performance_metrics[function_name] = {
                            'call_count': 0,
                            'total_time': 0,
                            'avg_time': 0,
                            'min_time': float('inf'),
                            'max_time': 0,
                            'error_count': 0
                        }
                    
                    metrics = self._performance_metrics[function_name]
                    metrics['call_count'] += 1
                    metrics['total_time'] += duration
                    metrics['avg_time'] = metrics['total_time'] / metrics['call_count']
                    import builtins
                    metrics['min_time'] = builtins.min(metrics['min_time'], duration)
                    metrics['max_time'] = max(metrics['max_time'], duration)
                
                return result
                
            except Exception as e:
                # Record error metrics
                duration = time.time() - start_time
                if hasattr(self, '_performance_metrics'):
                    if function_name not in self._performance_metrics:
                        self._performance_metrics[function_name] = {
                            'call_count': 0, 'total_time': 0, 'avg_time': 0,
                            'min_time': float('inf'), 'max_time': 0, 'error_count': 0
                        }
                    self._performance_metrics[function_name]['error_count'] += 1
                
                raise
        return wrapper
    return decorator

# ======================== ENHANCED INPUT VALIDATION ========================

class InputValidator:
    """Comprehensive input validation for ML operations"""
    
    @staticmethod
    def validate_data_shape(X: Union[pd.DataFrame, np.ndarray], 
                           y: Optional[Union[pd.Series, np.ndarray]] = None):
        """Validate data shape and consistency with enhanced checks"""
        if X is None:
            raise InvalidInputError("Input data X cannot be None")
            
        # Validate X format and shape
        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise InvalidInputError("Input DataFrame is empty")
            if X.isnull().all().any():
                raise InvalidInputError("Input DataFrame contains columns with all null values")
            n_samples = len(X)
            n_features = len(X.columns)
        elif isinstance(X, np.ndarray):
            if X.size == 0:
                raise InvalidInputError("Input array is empty")
            if len(X.shape) != 2:
                raise InvalidInputError(f"Input array must be 2D, got shape {X.shape}")
            if np.isnan(X).all():
                raise InvalidInputError("Input array contains only NaN values")
            n_samples = X.shape[0]
            n_features = X.shape[1]
        else:
            raise InvalidInputError(f"Invalid input type for X: {type(X)}")
            
        # Validate minimum data requirements
        if n_samples < 10:
            raise InvalidInputError(f"Insufficient data: need at least 10 samples, got {n_samples}")
        if n_features < 1:
            raise InvalidInputError(f"No features provided: {n_features}")
            
        # Validate y if provided
        if y is not None:
            if isinstance(y, pd.Series):
                if y.empty:
                    raise InvalidInputError("Target Series is empty")
                y_samples = len(y)
            elif isinstance(y, np.ndarray):
                if y.size == 0:
                    raise InvalidInputError("Target array is empty")
                if len(y.shape) != 1:
                    raise InvalidInputError(f"Target array must be 1D, got shape {y.shape}")
                y_samples = y.shape[0]
            else:
                raise InvalidInputError(f"Invalid input type for y: {type(y)}")
                
            if n_samples != y_samples:
                raise InvalidInputError(
                    f"X and y have different number of samples: {n_samples} vs {y_samples}"
                )
    
    @staticmethod
    def validate_features(features: Union[Dict[str, Any], pd.DataFrame, np.ndarray], 
                         expected_features: Optional[List[str]] = None):
        """Validate feature format and names with enhanced checks"""
        if features is None:
            raise InvalidInputError("Features cannot be None")
            
        if isinstance(features, dict):
            if not features:
                raise InvalidInputError("Feature dictionary is empty")
            
            # Check for invalid values
            for key, value in features.items():
                if value is None:
                    logger.warning(f"Feature {key} has None value, will be handled")
                elif isinstance(value, str) and value.strip() == "":
                    logger.warning(f"Feature {key} has empty string value")
                elif isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                    raise InvalidInputError(f"Feature {key} has invalid numeric value: {value}")
            
            if expected_features:
                missing_features = set(expected_features) - set(features.keys())
                extra_features = set(features.keys()) - set(expected_features)
                
                if missing_features:
                    logger.warning(f"Missing features (will use defaults): {missing_features}")
                if extra_features:
                    logger.warning(f"Extra features (will be ignored): {extra_features}")
        
        elif isinstance(features, pd.DataFrame):
            if features.empty:
                raise InvalidInputError("Feature DataFrame is empty")
            
            # Check for data quality issues
            if features.isnull().all().any():
                null_cols = features.columns[features.isnull().all()].tolist()
                raise InvalidInputError(f"Columns with all null values: {null_cols}")
            
            if expected_features:
                missing_features = set(expected_features) - set(features.columns)
                if missing_features:
                    raise FeatureMismatchError(
                        f"Missing required features: {missing_features}",
                        expected_features=expected_features,
                        actual_features=list(features.columns)
                    )
                    
        elif isinstance(features, np.ndarray):
            if features.size == 0:
                raise InvalidInputError("Feature array is empty")
            
            if len(features.shape) != 2:
                raise InvalidInputError(f"Feature array must be 2D, got shape {features.shape}")
            
            # Check for invalid values
            if np.isnan(features).all():
                raise InvalidInputError("Feature array contains only NaN values")
            if np.isinf(features).any():
                raise InvalidInputError("Feature array contains infinite values")
            
            if expected_features and features.shape[1] != len(expected_features):
                raise FeatureMismatchError(
                    f"Feature count mismatch: expected {len(expected_features)}, got {features.shape[1]}",
                    expected_features=expected_features,
                    actual_features=[f"feature_{i}" for i in range(features.shape[1])]
                )
        else:
            raise InvalidInputError(f"Invalid feature type: {type(features)}")
    
    @staticmethod
    def validate_model_params(model_params: Dict[str, Any], model_type: str):
        """Validate model hyperparameters with enhanced constraint checking"""
        if not isinstance(model_params, dict):
            raise HyperparameterError("Model parameters must be a dictionary")
            
        # Define comprehensive parameter constraints
        param_constraints = {
            "random_forest": {
                "n_estimators": {"type": int, "min": 1, "max": 1000, "default": 100},
                "max_depth": {"type": (int, type(None)), "min": 1, "max": 100, "default": None},
                "min_samples_split": {"type": int, "min": 2, "max": 50, "default": 2},
                "min_samples_leaf": {"type": int, "min": 1, "max": 50, "default": 1},
                "max_features": {"type": (str, int, float, type(None)), "options": ["auto", "sqrt", "log2"], "default": "sqrt"},
                "bootstrap": {"type": bool, "default": True},
                "oob_score": {"type": bool, "default": False},
                "random_state": {"type": (int, type(None)), "default": None}
            },
            "xgboost": {
                "n_estimators": {"type": int, "min": 1, "max": 1000, "default": 100},
                "max_depth": {"type": int, "min": 1, "max": 20, "default": 6},
                "learning_rate": {"type": float, "min": 0.001, "max": 1.0, "default": 0.3},
                "subsample": {"type": float, "min": 0.1, "max": 1.0, "default": 1.0},
                "colsample_bytree": {"type": float, "min": 0.1, "max": 1.0, "default": 1.0},
                "reg_alpha": {"type": float, "min": 0.0, "max": 1000.0, "default": 0.0},
                "reg_lambda": {"type": float, "min": 0.0, "max": 1000.0, "default": 1.0},
                "gamma": {"type": float, "min": 0.0, "max": 1000.0, "default": 0.0}
            },
            "lightgbm": {
                "n_estimators": {"type": int, "min": 1, "max": 1000, "default": 100},
                "num_leaves": {"type": int, "min": 2, "max": 1000, "default": 31},
                "learning_rate": {"type": float, "min": 0.001, "max": 1.0, "default": 0.1},
                "feature_fraction": {"type": float, "min": 0.1, "max": 1.0, "default": 1.0},
                "bagging_fraction": {"type": float, "min": 0.1, "max": 1.0, "default": 1.0},
                "bagging_freq": {"type": int, "min": 0, "max": 100, "default": 0},
                "min_child_samples": {"type": int, "min": 1, "max": 1000, "default": 20},
                "reg_alpha": {"type": float, "min": 0.0, "max": 1000.0, "default": 0.0},
                "reg_lambda": {"type": float, "min": 0.0, "max": 1000.0, "default": 0.0}
            },
            "logistic_regression": {
                "C": {"type": float, "min": 0.001, "max": 1000.0, "default": 1.0},
                "penalty": {"type": str, "options": ["l1", "l2", "elasticnet", "none"], "default": "l2"},
                "solver": {"type": str, "options": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"], "default": "lbfgs"},
                "max_iter": {"type": int, "min": 1, "max": 10000, "default": 1000},
                "tol": {"type": float, "min": 1e-6, "max": 1e-1, "default": 1e-4}
            }
        }
        
        if model_type not in param_constraints:
            logger.warning(f"Unknown model type for validation: {model_type}")
            return
            
        constraints = param_constraints[model_type]
        
        # Validate each parameter
        for param, value in model_params.items():
            if param not in constraints:
                logger.warning(f"Unknown parameter {param} for {model_type}")
                continue
                
            constraint = constraints[param]
            
            # Type validation
            if not isinstance(value, constraint["type"]):
                raise HyperparameterError(
                    f"Parameter {param} must be of type {constraint['type']}, got {type(value)}",
                    parameter_name=param,
                    parameter_value=value
                )
            
            # Range validation
            if "min" in constraint and "max" in constraint:
                if not constraint["min"] <= value <= constraint["max"]:
                    raise HyperparameterError(
                        f"Parameter {param} value {value} is outside valid range "
                        f"[{constraint['min']}, {constraint['max']}]",
                        parameter_name=param,
                        parameter_value=value
                    )
            
            # Options validation
            if "options" in constraint and value not in constraint["options"]:
                raise HyperparameterError(
                    f"Parameter {param} value {value} not in valid options: {constraint['options']}",
                    parameter_name=param,
                    parameter_value=value
                )
    
    @staticmethod
    def validate_binary_labels(y: Union[pd.Series, np.ndarray]):
        """Validate binary classification labels with enhanced checks"""
        if y is None:
            raise InvalidInputError("Labels cannot be None")
            
        # Convert to numpy for easier handling
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
            
        # Check for empty labels
        if len(y_values) == 0:
            raise InvalidInputError("Labels array is empty")
            
        # Check for null values
        if np.isnan(y_values).any():
            raise InvalidInputError("Labels contain NaN values")
            
        # Get unique values
        unique_values = np.unique(y_values[~np.isnan(y_values)])
        
        # Check for binary classification
        if len(unique_values) == 0:
            raise InvalidInputError("No valid labels found")
        elif len(unique_values) == 1:
            logger.warning(f"Only one class found in labels: {unique_values[0]}")
        elif len(unique_values) > 2:
            raise InvalidInputError(
                f"Expected binary labels (0/1), got {len(unique_values)} unique values: {unique_values}"
            )
        
        # Check label format
        if not all(val in [0, 1] for val in unique_values):
            raise InvalidInputError(
                f"Binary labels must be 0 and 1, got {unique_values}"
            )
        
        # Check class balance
        if len(unique_values) == 2:
            counts = np.bincount(y_values.astype(int))
            minority_ratio = min(counts) / max(counts)
            if minority_ratio < 0.01:
                logger.warning(f"Severe class imbalance detected: {minority_ratio:.3f}")
    
    @staticmethod
    def validate_prediction_input(features: Union[Dict[str, Any], pd.DataFrame, np.ndarray]):
        """Validate input for prediction with enhanced checks"""
        if features is None:
            raise InvalidInputError("Prediction features cannot be None")
            
        if isinstance(features, dict):
            if not features:
                raise InvalidInputError("Empty feature dictionary provided")
            
            # Check for required minimum features
            numeric_features = sum(1 for v in features.values() 
                                 if isinstance(v, (int, float)) and not (np.isnan(v) or np.isinf(v)))
            if numeric_features == 0:
                raise InvalidInputError("No valid numeric features found for prediction")
                
        elif isinstance(features, pd.DataFrame):
            if features.empty:
                raise InvalidInputError("Empty DataFrame provided for prediction")
            
            # Check for all-null rows
            if features.isnull().all(axis=1).any():
                raise InvalidInputError("DataFrame contains rows with all null values")
                
        elif isinstance(features, np.ndarray):
            if features.size == 0:
                raise InvalidInputError("Empty array provided for prediction")
            
            if len(features.shape) != 2:
                raise InvalidInputError(f"Prediction array must be 2D, got shape {features.shape}")
            
            # Check for invalid values
            if np.isnan(features).all():
                raise InvalidInputError("Prediction array contains only NaN values")
            if np.isinf(features).any():
                raise InvalidInputError("Prediction array contains infinite values")
        else:
            raise InvalidInputError(f"Invalid feature type for prediction: {type(features)}")

# ======================== ENHANCED ML ENUMS ========================

class ModelType(Enum):
    """Supported ML model types with enhanced options"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LOGISTIC_REGRESSION = "logistic_regression"
    MLP = "mlp"
    SVM = "svm"
    DECISION_TREE = "decision_tree"
    NAIVE_BAYES = "naive_bayes"
    ISOLATION_FOREST = "isolation_forest"
    ENSEMBLE = "ensemble"
    VOTING_CLASSIFIER = "voting_classifier"
    STACKING_CLASSIFIER = "stacking_classifier"
    NEURAL_NETWORK = "neural_network"
    LSTM = "lstm"
    TRANSFORMER = "transformer"

class ModelStatus(Enum):
    """Model lifecycle status with enhanced states"""
    INITIALIZING = "initializing"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    VALIDATED = "validated"
    TUNING = "tuning"
    CALIBRATING = "calibrating"
    TESTING = "testing"
    DEPLOYED = "deployed"
    MONITORING = "monitoring"
    ARCHIVED = "archived"
    FAILED = "failed"
    DEPRECATED = "deprecated"

class ValidationLevel(Enum):
    """Validation level for model validation"""
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"

class RiskLevel(Enum):
    """Risk level classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"

class PredictionConfidence(Enum):
    """Prediction confidence levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

# ======================== ENHANCED ML CONFIGURATION ========================

@dataclass
class ModelConfig:
    """Enhanced configuration for ML models with comprehensive validation"""
    
    # Model selection
    model_type: ModelType = ModelType.RANDOM_FOREST
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Data splitting
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    stratify: bool = True
    shuffle: bool = True
    
    # Cross-validation
    cv_folds: int = 5
    cv_scoring: str = "roc_auc"
    cv_stratify: bool = True
    
    # Hyperparameter tuning
    hyperparameter_tuning: bool = True
    tuning_method: str = "grid"  # 'grid', 'random', 'bayesian'
    tuning_iterations: int = 50
    tuning_cv_folds: int = 3
    tuning_scoring: str = "roc_auc"
    tuning_timeout: int = 3600  # 1 hour
    
    # Model calibration
    calibrate_model: bool = True
    calibration_method: str = "sigmoid"  # 'sigmoid', 'isotonic'
    calibration_cv_folds: int = 3
    
    # Fraud detection thresholds
    fraud_threshold: float = 0.5
    high_risk_threshold: float = 0.7
    critical_risk_threshold: float = 0.9
    
    # Feature engineering
    calculate_feature_importance: bool = True
    max_features_to_show: int = 20
    feature_selection: bool = False
    feature_selection_method: str = "mutual_info"
    max_features_selected: int = 50
    
    # Model persistence
    save_model: bool = True
    model_path: Optional[Path] = None
    model_compression: bool = True
    
    # Performance thresholds
    min_auc: float = 0.85
    min_precision: float = 0.8
    min_recall: float = 0.6
    min_f1: float = 0.7
    max_training_time: int = 7200  # 2 hours
    
    # Drift detection
    enable_drift_detection: bool = True
    drift_threshold: float = 0.1
    drift_detection_window: int = 1000
    
    # Preprocessing settings
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)
    
    # Validation settings
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    raise_on_error: bool = False
    enable_warnings: bool = True
    
    # Security settings
    enable_model_encryption: bool = True
    model_signature: Optional[str] = None
    audit_predictions: bool = True
    
    # Performance settings
    n_jobs: int = -1
    memory_limit_mb: int = 4096
    prediction_batch_size: int = 1000
    
    # Monitoring settings
    enable_monitoring: bool = True
    monitoring_interval: int = 300  # 5 minutes
    performance_tracking: bool = True
    
    def __post_init__(self):
        """Enhanced validation after initialization"""
        self._validate_config()
        self._set_defaults()
    
    def _validate_config(self):
        """Comprehensive configuration validation"""
        # Validate data splitting parameters
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")
        if not 0 <= self.validation_size < 1:
            raise ValueError(f"validation_size must be between 0 and 1, got {self.validation_size}")
        if self.test_size + self.validation_size >= 1:
            raise ValueError("test_size + validation_size must be less than 1")
        
        # Validate thresholds
        for threshold_name, threshold_value in [
            ("fraud_threshold", self.fraud_threshold),
            ("high_risk_threshold", self.high_risk_threshold),
            ("critical_risk_threshold", self.critical_risk_threshold),
            ("min_auc", self.min_auc),
            ("min_precision", self.min_precision),
            ("min_recall", self.min_recall),
            ("min_f1", self.min_f1),
            ("drift_threshold", self.drift_threshold)
        ]:
            if not 0 <= threshold_value <= 1:
                raise ValueError(f"{threshold_name} must be between 0 and 1, got {threshold_value}")
        
        # Validate threshold ordering
        if self.fraud_threshold > self.high_risk_threshold:
            raise ValueError("fraud_threshold cannot be greater than high_risk_threshold")
        if self.high_risk_threshold > self.critical_risk_threshold:
            raise ValueError("high_risk_threshold cannot be greater than critical_risk_threshold")
        
        # Validate CV parameters
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be at least 2, got {self.cv_folds}")
        if self.tuning_cv_folds < 2:
            raise ValueError(f"tuning_cv_folds must be at least 2, got {self.tuning_cv_folds}")
        if self.calibration_cv_folds < 2:
            raise ValueError(f"calibration_cv_folds must be at least 2, got {self.calibration_cv_folds}")
        
        # Validate tuning parameters
        if self.tuning_method not in ["grid", "random", "bayesian"]:
            raise ValueError(f"tuning_method must be 'grid', 'random', or 'bayesian', got {self.tuning_method}")
        if self.tuning_iterations < 1:
            raise ValueError(f"tuning_iterations must be at least 1, got {self.tuning_iterations}")
        
        # Validate calibration method
        if self.calibration_method not in ["sigmoid", "isotonic"]:
            raise ValueError(f"calibration_method must be 'sigmoid' or 'isotonic', got {self.calibration_method}")
        
        # Validate feature selection
        if self.feature_selection_method not in ["mutual_info", "f_classif", "chi2", "variance"]:
            raise ValueError(f"Invalid feature_selection_method: {self.feature_selection_method}")
        if self.max_features_selected < 1:
            raise ValueError(f"max_features_selected must be at least 1, got {self.max_features_selected}")
        
        # Validate resource limits
        if self.max_training_time < 60:
            raise ValueError(f"max_training_time must be at least 60 seconds, got {self.max_training_time}")
        if self.memory_limit_mb < 256:
            raise ValueError(f"memory_limit_mb must be at least 256, got {self.memory_limit_mb}")
        if self.prediction_batch_size < 1:
            raise ValueError(f"prediction_batch_size must be at least 1, got {self.prediction_batch_size}")
        
        # Validate model params if provided
        if self.model_params:
            InputValidator.validate_model_params(self.model_params, self.model_type.value)
    
    def _set_defaults(self):
        """Set intelligent defaults based on configuration"""
        # Set model path if not provided
        if self.save_model and not self.model_path:
            self.model_path = Path("models")
            
        # Adjust CV folds based on data size expectations
        if self.cv_folds > 10:
            logger.warning("High number of CV folds may slow training")
        
        # Set n_jobs based on system capabilities
        if self.n_jobs == -1:
            import multiprocessing
            self.n_jobs = min(multiprocessing.cpu_count(), 8)  # Cap at 8 cores
        
        # Generate model signature if security enabled
        if self.enable_model_encryption and not self.model_signature:
            config_str = f"{self.model_type.value}_{self.random_state}_{time.time()}"
            self.model_signature = hashlib.md5(config_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (Enum, Path)):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict
    
    def copy(self) -> 'ModelConfig':
        """Create a copy of the configuration"""
        return ModelConfig(**self.to_dict())

# ======================== ENHANCED ML BASE CLASSES ========================

class BaseModel(ABC):
    """Enhanced abstract base class for ML models with comprehensive validation"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.fitted = False
        self.feature_names = []
        self.feature_importance_ = {}
        self.training_time = 0
        self.prediction_time = 0
        self._performance_metrics = {}
        self._error_stats = {}
        self._lock = threading.RLock()
        
        # Validate configuration
        self._validate_config()
        
        # Initialize monitoring
        if self.config.enable_monitoring:
            self._initialize_monitoring()
    
    def _validate_config(self):
        """Validate model configuration"""
        if not isinstance(self.config, ModelConfig):
            raise ValueError("config must be a ModelConfig instance")
    
    def _initialize_monitoring(self):
        """Initialize performance monitoring"""
        self._performance_metrics = {
            'fit': {'call_count': 0, 'total_time': 0, 'avg_time': 0, 'min_time': float('inf'), 'max_time': 0, 'error_count': 0},
            'predict': {'call_count': 0, 'total_time': 0, 'avg_time': 0, 'min_time': float('inf'), 'max_time': 0, 'error_count': 0},
            'predict_proba': {'call_count': 0, 'total_time': 0, 'avg_time': 0, 'min_time': float('inf'), 'max_time': 0, 'error_count': 0}
        }
    
    @abstractmethod
    def create_model(self):
        """Create the underlying ML model - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores - must be implemented by subclasses"""
        pass
    
    @monitor_performance('fit')
    @validate_input(InputValidator.validate_data_shape)
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """Fit the model with enhanced validation and monitoring"""
        start_time = time.time()
        
        try:
            # Additional validation for training data
            self._validate_training_data(X, y)
            
            # Extract feature names
            if isinstance(X, pd.DataFrame):
                self.feature_names = X.columns.tolist()
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            # Create model if not exists
            if self.model is None:
                self.model = self.create_model()
            
            # Fit the model
            self.model.fit(X, y)
            self.fitted = True
            
            # Calculate training time
            self.training_time = time.time() - start_time
            
            # Calculate feature importance if enabled
            if self.config.calculate_feature_importance:
                self.feature_importance_ = self.get_feature_importance()
            
            logger.info(f"Model fitted successfully in {self.training_time:.2f} seconds")
            return self
            
        except Exception as e:
            logger.error(f"Model fitting failed: {str(e)}")
            raise ModelTrainingError(f"Failed to fit model: {str(e)}", training_stage="fit")
    
    @monitor_performance('predict')
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict with enhanced validation and monitoring"""
        if not self.fitted:
            raise ModelNotFittedError("Model must be fitted before prediction")
        
        start_time = time.time()
        
        try:
            # Validate prediction input
            InputValidator.validate_prediction_input(X)
            
            # Validate feature consistency
            if isinstance(X, pd.DataFrame):
                self._validate_feature_consistency(X.columns.tolist())
            
            # Make prediction
            predictions = self.model.predict(X)
            
            # Update prediction time
            self.prediction_time = time.time() - start_time
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError(f"Failed to predict: {str(e)}", prediction_stage="predict")
    
    @monitor_performance('predict_proba')
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict probabilities with enhanced validation and monitoring"""
        if not self.fitted:
            raise ModelNotFittedError("Model must be fitted before prediction")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ModelValidationError("Model does not support probability prediction")
        
        start_time = time.time()
        
        try:
            # Validate prediction input
            InputValidator.validate_prediction_input(X)
            
            # Validate feature consistency
            if isinstance(X, pd.DataFrame):
                self._validate_feature_consistency(X.columns.tolist())
            
            # Make prediction
            probabilities = self.model.predict_proba(X)
            
            # Validate probability output
            self._validate_probabilities(probabilities)
            
            # Update prediction time
            self.prediction_time = time.time() - start_time
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Probability prediction failed: {str(e)}")
            raise PredictionError(f"Failed to predict probabilities: {str(e)}", 
                                prediction_stage="predict_proba")
    
    def _validate_training_data(self, X: Union[pd.DataFrame, np.ndarray], 
                              y: Union[pd.Series, np.ndarray]):
        """Validate training data quality"""
        # Check for sufficient data
        n_samples = len(X) if isinstance(X, pd.DataFrame) else X.shape[0]
        if n_samples < 100:
            logger.warning(f"Small training set: {n_samples} samples")
        
        # Check for class balance
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
            
        unique_classes, counts = np.unique(y_values, return_counts=True)
        if len(unique_classes) == 2:
            minority_ratio = min(counts) / max(counts)
            if minority_ratio < 0.1:
                logger.warning(f"Class imbalance detected: {minority_ratio:.3f}")
    
    def _validate_feature_consistency(self, input_features: List[str]):
        """Validate feature consistency with training data"""
        if not self.feature_names:
            logger.warning("No feature names stored from training")
            return
        
        missing_features = set(self.feature_names) - set(input_features)
        extra_features = set(input_features) - set(self.feature_names)
        
        if missing_features:
            raise FeatureMismatchError(
                f"Missing features: {missing_features}",
                expected_features=self.feature_names,
                actual_features=input_features
            )
        
        if extra_features:
            logger.warning(f"Extra features (will be ignored): {extra_features}")
    
    def _validate_probabilities(self, probabilities: np.ndarray):
        """Validate probability predictions"""
        if probabilities.ndim != 2:
            raise PredictionError(f"Probabilities must be 2D, got shape {probabilities.shape}")
        
        if probabilities.shape[1] != 2:
            raise PredictionError(f"Expected 2 classes, got {probabilities.shape[1]}")
        
        # Check probability ranges
        if not np.all((probabilities >= 0) & (probabilities <= 1)):
            raise PredictionError("Probabilities must be between 0 and 1")
        
        # Check probability sums
        row_sums = probabilities.sum(axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1e-6):
            raise PredictionError("Probabilities must sum to 1 for each row")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        with self._lock:
            return {
                'training_time': self.training_time,
                'prediction_time': self.prediction_time,
                'fitted': self.fitted,
                'feature_count': len(self.feature_names),
                'performance_metrics': self._performance_metrics.copy(),
                'error_stats': self._error_stats.copy()
            }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        with self._lock:
            self._performance_metrics = {}
            self._error_stats = {}
            self.training_time = 0
            self.prediction_time = 0
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(fitted={self.fitted}, "
                f"features={len(self.feature_names)}, "
                f"training_time={self.training_time:.2f}s)")

# Export framework components
__all__ = [
    # Exceptions
    'MLPredictorError', 'ModelNotFittedError', 'InvalidInputError', 
    'ModelValidationError', 'FeatureMismatchError', 'ModelDriftError',
    'ModelLoadError', 'HyperparameterError', 'ModelTrainingError',
    'PredictionError', 'ModelDeploymentError',
    
    # Decorators
    'validate_input', 'handle_errors', 'monitor_performance',
    
    # Validation
    'InputValidator',
    
    # Enums
    'ModelType', 'ModelStatus', 'ValidationLevel', 'RiskLevel', 'PredictionConfidence',
    
    # Configuration
    'ModelConfig',
    
    # Base classes
    'BaseModel'
]

if __name__ == "__main__":
    print("Enhanced ML Framework - Chunk 1: Core ML framework with base classes and validation loaded")
    
    # Basic validation test
    try:
        config = ModelConfig(model_type=ModelType.RANDOM_FOREST)
        print("✓ ModelConfig created successfully")
        
        # Test input validation
        validator = InputValidator()
        test_data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        test_labels = pd.Series([0, 1, 0])
        validator.validate_data_shape(test_data, test_labels)
        print("✓ Input validation working")
        
        print("✓ Enhanced ML Framework Chunk 1 loaded successfully")
        
    except Exception as e:
        print(f"✗ Framework test failed: {e}")