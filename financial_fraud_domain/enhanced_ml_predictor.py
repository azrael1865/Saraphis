"""
Machine Learning Predictor for Financial Fraud Detection
Advanced ML models for fraud detection with comprehensive validation,
error handling, model versioning, evaluation, and production-ready features.
"""

import logging
import json
import pickle
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from enhanced_ml_framework import ModelType
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib
import time
import traceback
from contextlib import contextmanager

# ML Libraries
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    IsolationForest,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

# Configure logging
logger = logging.getLogger(__name__)

# Custom Exception Classes
class MLPredictorError(Exception):
    """Base exception for ML predictor errors"""
    pass

class ModelNotFittedError(MLPredictorError):
    """Raised when trying to use an unfitted model"""
    pass

class InvalidInputError(MLPredictorError):
    """Raised when input data is invalid"""
    pass

class ModelValidationError(MLPredictorError):
    """Raised when model fails validation"""
    pass

class FeatureMismatchError(MLPredictorError):
    """Raised when features don't match expected format"""
    pass

class ModelDriftError(MLPredictorError):
    """Raised when model drift is detected"""
    pass

class ModelLoadError(MLPredictorError):
    """Raised when model loading fails"""
    pass

class HyperparameterError(MLPredictorError):
    """Raised when hyperparameters are invalid"""
    pass

class ModelTrainingError(MLPredictorError):
    """Raised when model training fails"""
    pass

class PredictionError(MLPredictorError):
    """Raised when prediction fails"""
    pass

# Validation Decorators
def validate_input(validation_func: Callable):
    """Decorator for input validation"""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            try:
                validation_func(*args, **kwargs)
                return func(self, *args, **kwargs)
            except Exception as e:
                logger.error(f"Input validation failed for {func.__name__}: {str(e)}")
                raise InvalidInputError(f"Invalid input for {func.__name__}: {str(e)}")
        return wrapper
    return decorator

def handle_errors(fallback_value=None):
    """Decorator for error handling with fallback"""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                logger.debug(traceback.format_exc())
                if fallback_value is not None:
                    return fallback_value
                raise
        return wrapper
    return decorator

# Input Validators
class InputValidator:
    """Comprehensive input validation for ML operations"""
    
    @staticmethod
    def validate_data_shape(X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None):
        """Validate data shape and consistency"""
        if X is None:
            raise InvalidInputError("Input data X cannot be None")
        
        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise InvalidInputError("Input DataFrame is empty")
            n_samples = len(X)
        elif isinstance(X, np.ndarray):
            if X.size == 0:
                raise InvalidInputError("Input array is empty")
            n_samples = X.shape[0]
        else:
            raise InvalidInputError(f"Invalid input type for X: {type(X)}")
        
        if y is not None:
            if isinstance(y, pd.Series):
                y_samples = len(y)
            elif isinstance(y, np.ndarray):
                y_samples = y.shape[0]
            else:
                raise InvalidInputError(f"Invalid input type for y: {type(y)}")
            
            if n_samples != y_samples:
                raise InvalidInputError(f"X and y have different number of samples: {n_samples} vs {y_samples}")
    
    @staticmethod
    def validate_features(features: Union[Dict[str, Any], pd.DataFrame, np.ndarray], 
                         expected_features: Optional[List[str]] = None):
        """Validate feature format and names"""
        if features is None:
            raise InvalidInputError("Features cannot be None")
        
        if isinstance(features, dict):
            if not features:
                raise InvalidInputError("Feature dictionary is empty")
            if expected_features:
                missing_features = set(expected_features) - set(features.keys())
                if missing_features:
                    logger.warning(f"Missing features: {missing_features}")
        
        elif isinstance(features, pd.DataFrame):
            if features.empty:
                raise InvalidInputError("Feature DataFrame is empty")
            if expected_features:
                missing_features = set(expected_features) - set(features.columns)
                if missing_features:
                    raise FeatureMismatchError(f"Missing required features: {missing_features}")
        
        elif isinstance(features, np.ndarray):
            if features.size == 0:
                raise InvalidInputError("Feature array is empty")
            if expected_features and len(features.shape) > 1:
                if features.shape[1] != len(expected_features):
                    raise FeatureMismatchError(
                        f"Feature count mismatch: expected {len(expected_features)}, got {features.shape[1]}"
                    )
    
    @staticmethod
    def validate_model_params(model_params: Dict[str, Any], model_type: str):
        """Validate model hyperparameters"""
        if not isinstance(model_params, dict):
            raise HyperparameterError("Model parameters must be a dictionary")
        
        # Define parameter constraints for each model type
        param_constraints = {
            'random_forest': {
                'n_estimators': (1, 10000),
                'max_depth': (1, 100),
                'min_samples_split': (2, 100),
                'min_samples_leaf': (1, 100)
            },
            'xgboost': {
                'n_estimators': (1, 10000),
                'max_depth': (1, 20),
                'learning_rate': (0.001, 1.0),
                'subsample': (0.1, 1.0)
            },
            'lightgbm': {
                'n_estimators': (1, 10000),
                'num_leaves': (2, 300),
                'learning_rate': (0.001, 1.0),
                'feature_fraction': (0.1, 1.0)
            }
        }
        
        if model_type in param_constraints:
            constraints = param_constraints[model_type]
            for param, value in model_params.items():
                if param in constraints:
                    min_val, max_val = constraints[param]
                    if not (min_val <= value <= max_val):
                        raise HyperparameterError(
                            f"Parameter {param} value {value} is outside valid range [{min_val}, {max_val}]"
                        )
    
    @staticmethod
    def validate_binary_labels(y: Union[np.ndarray, pd.Series]):
        """Validate binary classification labels"""
        unique_values = np.unique(y)
        if len(unique_values) > 2:
            raise InvalidInputError(f"Expected binary labels, got {len(unique_values)} unique values")
        if not all(val in [0, 1] for val in unique_values):
            raise InvalidInputError(f"Binary labels must be 0 and 1, got {unique_values}")


class ModelStatus(Enum):
    """Model lifecycle status"""
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"

@dataclass
class ModelConfig:
    """Configuration for ML models with validation"""
    model_type: ModelType = ModelType.RANDOM_FOREST
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    stratify: bool = True
    
    # Cross-validation
    cv_folds: int = 5
    cv_scoring: str = "roc_auc"
    
    # Hyperparameter tuning
    hyperparameter_tuning: bool = True
    tuning_method: str = "grid"  # grid, random, bayesian
    tuning_iterations: int = 50
    tuning_cv_folds: int = 3
    
    # Model calibration
    calibrate_model: bool = True
    calibration_method: str = "sigmoid"  # sigmoid, isotonic
    
    # Thresholds
    fraud_threshold: float = 0.5
    high_risk_threshold: float = 0.8
    
    # Feature importance
    calculate_feature_importance: bool = True
    max_features_to_show: int = 20
    
    # Model persistence
    save_model: bool = True
    model_path: Optional[Path] = None
    
    # Performance thresholds
    min_auc: float = 0.85
    min_precision: float = 0.8
    min_recall: float = 0.7
    
    # Drift detection
    enable_drift_detection: bool = True
    drift_threshold: float = 0.1
    
    # Error handling
    raise_on_error: bool = True
    
    # Security
    enable_model_encryption: bool = False
    model_signature: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate splits
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")
        if not 0 <= self.validation_size < 1:
            raise ValueError(f"validation_size must be between 0 and 1, got {self.validation_size}")
        if self.test_size + self.validation_size >= 1:
            raise ValueError("test_size + validation_size must be less than 1")
        
        # Validate thresholds
        if not 0 <= self.fraud_threshold <= 1:
            raise ValueError(f"fraud_threshold must be between 0 and 1, got {self.fraud_threshold}")
        if not 0 <= self.high_risk_threshold <= 1:
            raise ValueError(f"high_risk_threshold must be between 0 and 1, got {self.high_risk_threshold}")
        
        # Validate performance thresholds
        for threshold_name, threshold_value in [
            ('min_auc', self.min_auc),
            ('min_precision', self.min_precision),
            ('min_recall', self.min_recall)
        ]:
            if not 0 <= threshold_value <= 1:
                raise ValueError(f"{threshold_name} must be between 0 and 1, got {threshold_value}")
        
        # Validate model parameters
        if self.model_params:
            InputValidator.validate_model_params(self.model_params, self.model_type.value)

@dataclass
class ModelMetrics:
    """ML model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]
    
    # Additional metrics
    true_positive_rate: float = 0.0
    false_positive_rate: float = 0.0
    precision_recall_auc: float = 0.0
    
    # Threshold-specific metrics
    thresholds: List[float] = field(default_factory=list)
    precision_by_threshold: List[float] = field(default_factory=list)
    recall_by_threshold: List[float] = field(default_factory=list)
    
    # Feature importance
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Model metadata
    training_time_seconds: float = 0.0
    prediction_time_seconds: float = 0.0
    model_size_bytes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc_roc': self.auc_roc,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'tpr': self.true_positive_rate,
            'fpr': self.false_positive_rate,
            'pr_auc': self.precision_recall_auc,
            'training_time': self.training_time_seconds,
            'model_size': self.model_size_bytes
        }

@dataclass
class PredictionResult:
    """Results from fraud prediction"""
    transaction_id: str
    fraud_probability: float
    risk_score: float
    is_fraud: bool
    risk_level: str  # low, medium, high, critical
    confidence: float
    
    # Additional insights
    risk_factors: List[str] = field(default_factory=list)
    anomaly_score: float = 0.0
    prediction_time: datetime = field(default_factory=datetime.now)
    model_version: str = ""
    
    # Feature contributions
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Validation
    prediction_valid: bool = True
    validation_warnings: List[str] = field(default_factory=list)

@dataclass
class ModelVersion:
    """Model version information"""
    version_id: str
    model_type: ModelType
    created_at: datetime
    metrics: ModelMetrics
    status: ModelStatus
    config: ModelConfig
    
    # Version metadata
    training_data_hash: str = ""
    feature_names: List[str] = field(default_factory=list)
    model_path: Optional[Path] = None
    notes: str = ""
    
    # Deployment info
    is_active: bool = False
    deployed_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None
    
    # Drift monitoring
    baseline_metrics: Optional[ModelMetrics] = None
    drift_scores: List[float] = field(default_factory=list)

class ModelDriftDetector:
    """Detect and monitor model drift"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.baseline_metrics: Optional[ModelMetrics] = None
        self.drift_history: List[Dict[str, Any]] = []
    
    def set_baseline(self, metrics: ModelMetrics):
        """Set baseline metrics for drift detection"""
        self.baseline_metrics = metrics
    
    def calculate_drift(self, current_metrics: ModelMetrics) -> float:
        """Calculate drift score between baseline and current metrics"""
        if not self.baseline_metrics:
            raise ValueError("Baseline metrics not set")
        
        # Calculate drift based on key metrics
        baseline_scores = [
            self.baseline_metrics.accuracy,
            self.baseline_metrics.precision,
            self.baseline_metrics.recall,
            self.baseline_metrics.auc_roc
        ]
        
        current_scores = [
            current_metrics.accuracy,
            current_metrics.precision,
            current_metrics.recall,
            current_metrics.auc_roc
        ]
        
        # Calculate average percentage change
        drift_scores = []
        for baseline, current in zip(baseline_scores, current_scores):
            if baseline > 0:
                drift_scores.append(abs(baseline - current) / baseline)
        
        drift_score = np.mean(drift_scores) if drift_scores else 0.0
        
        # Record drift
        self.drift_history.append({
            'timestamp': datetime.now(),
            'drift_score': drift_score,
            'metrics': current_metrics.to_dict()
        })
        
        return drift_score
    
    def check_drift(self, current_metrics: ModelMetrics) -> Tuple[bool, float]:
        """Check if model drift exceeds threshold"""
        drift_score = self.calculate_drift(current_metrics)
        is_drifted = drift_score > self.config.drift_threshold
        
        if is_drifted:
            logger.warning(f"Model drift detected: {drift_score:.3f} > {self.config.drift_threshold}")
        
        return is_drifted, drift_score

class BaseModel(ABC):
    """Abstract base class for ML models with validation"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.fitted = False
        self.feature_names = []
        self._validate_config()
    
    def _validate_config(self):
        """Validate model configuration"""
        if not isinstance(self.config, ModelConfig):
            raise ValueError("config must be a ModelConfig instance")
    
    @abstractmethod
    def create_model(self) -> Any:
        """Create the underlying ML model"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        pass
    
    @validate_input(InputValidator.validate_data_shape)
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """Fit the model with validation"""
        try:
            self.model = self.create_model()
            self.model.fit(X, y)
            self.fitted = True
            return self
        except Exception as e:
            logger.error(f"Model fitting failed: {str(e)}")
            raise ModelTrainingError(f"Failed to fit model: {str(e)}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict fraud labels with validation"""
        if not self.fitted:
            raise ModelNotFittedError("Model must be fitted before prediction")
        
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError(f"Failed to predict: {str(e)}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict fraud probabilities with validation"""
        if not self.fitted:
            raise ModelNotFittedError("Model must be fitted before prediction")
        
        try:
            return self.model.predict_proba(X)
        except Exception as e:
            logger.error(f"Probability prediction failed: {str(e)}")
            raise PredictionError(f"Failed to predict probabilities: {str(e)}")

class RandomForestModel(BaseModel):
    """Random Forest implementation with validation"""
    
    def create_model(self) -> RandomForestClassifier:
        """Create Random Forest model with validated parameters"""
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'random_state': self.config.random_state,
            'class_weight': 'balanced'
        }
        
        # Merge and validate parameters
        params = {**default_params, **self.config.model_params}
        
        # Additional validation
        if params['min_samples_split'] <= params['min_samples_leaf']:
            params['min_samples_split'] = params['min_samples_leaf'] + 1
        
        return RandomForestClassifier(**params)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Random Forest"""
        if not self.fitted or not self.feature_names:
            return {}
        
        try:
            return dict(zip(self.feature_names, self.model.feature_importances_))
        except Exception as e:
            logger.error(f"Failed to get feature importance: {str(e)}")
            return {}

class XGBoostModel(BaseModel):
    """XGBoost implementation with validation"""
    
    def create_model(self) -> xgb.XGBClassifier:
        """Create XGBoost model with validated parameters"""
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'use_label_encoder': False,
            'random_state': self.config.random_state,
            'n_jobs': -1
        }
        
        params = {**default_params, **self.config.model_params}
        return xgb.XGBClassifier(**params)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from XGBoost"""
        if not self.fitted or not self.feature_names:
            return {}
        
        try:
            importance = self.model.get_booster().get_score(importance_type='gain')
            # Map feature names
            return {self.feature_names[int(k[1:])]: v for k, v in importance.items() if k.startswith('f')}
        except Exception as e:
            logger.error(f"Failed to get feature importance: {str(e)}")
            return {}

class LightGBMModel(BaseModel):
    """LightGBM implementation with validation"""
    
    def create_model(self) -> lgb.LGBMClassifier:
        """Create LightGBM model with validated parameters"""
        default_params = {
            'n_estimators': 100,
            'max_depth': -1,
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0,
            'lambda_l2': 0,
            'min_child_samples': 20,
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'random_state': self.config.random_state,
            'n_jobs': -1,
            'verbose': -1
        }
        
        params = {**default_params, **self.config.model_params}
        return lgb.LGBMClassifier(**params)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from LightGBM"""
        if not self.fitted or not self.feature_names:
            return {}
        
        try:
            return dict(zip(self.feature_names, self.model.feature_importances_))
        except Exception as e:
            logger.error(f"Failed to get feature importance: {str(e)}")
            return {}

class EnsembleModel(BaseModel):
    """Ensemble of multiple models with validation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_models = []
    
    def create_model(self) -> VotingClassifier:
        """Create ensemble model with error handling"""
        try:
            # Create diverse base models
            models = [
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                ('xgb', xgb.XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss')),
                ('lgb', lgb.LGBMClassifier(n_estimators=50, verbose=-1)),
                ('lr', LogisticRegression(max_iter=1000))
            ]
            
            return VotingClassifier(estimators=models, voting='soft', n_jobs=-1)
        except Exception as e:
            logger.error(f"Failed to create ensemble model: {str(e)}")
            raise ModelTrainingError(f"Failed to create ensemble: {str(e)}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get averaged feature importance from ensemble"""
        if not self.fitted:
            return {}
        
        try:
            # Average importance from tree-based models
            importance_dict = {}
            tree_models = ['rf', 'xgb', 'lgb']
            
            for name, model in self.model.estimators_:
                if name in tree_models and hasattr(model, 'feature_importances_'):
                    for i, imp in enumerate(model.feature_importances_):
                        feature_name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'
                        if feature_name not in importance_dict:
                            importance_dict[feature_name] = []
                        importance_dict[feature_name].append(imp)
            
            # Average the importances
            return {k: np.mean(v) for k, v in importance_dict.items()}
        except Exception as e:
            logger.error(f"Failed to get ensemble feature importance: {str(e)}")
            return {}

class EnhancedFinancialMLPredictor:
    """
    Advanced Machine Learning Predictor for Financial Fraud Detection
    with comprehensive validation and error handling
    
    Features:
    - Multiple ML algorithms with validation
    - Model versioning and management
    - Hyperparameter tuning with validation
    - Model calibration
    - Comprehensive evaluation metrics
    - Production-ready prediction pipeline with error recovery
    - Model drift detection
    - Security and reliability features
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize ML predictor with validation
        
        Args:
            config: Model configuration
        """
        try:
            self.config = config or ModelConfig()
            self.current_model: Optional[BaseModel] = None
            self.model_versions: Dict[str, ModelVersion] = {}
            self.active_version: Optional[str] = None
            self.drift_detector = ModelDriftDetector(self.config) if self.config.enable_drift_detection else None
            self.prediction_stats = {
                'total_predictions': 0,
                'total_frauds_detected': 0,
                'average_prediction_time': 0.0,
                'last_prediction_time': None,
                'failed_predictions': 0,
                'drift_warnings': 0
            }
            self._lock = threading.RLock()
            self._fallback_model: Optional[BaseModel] = None
            
            # Initialize model directory
            if self.config.model_path is None:
                self.config.model_path = Path("models")
            self.config.model_path.mkdir(exist_ok=True)
            
            logger.info(f"Enhanced ML Predictor initialized with {self.config.model_type.value}")
        except Exception as e:
            logger.error(f"Failed to initialize ML predictor: {str(e)}")
            raise MLPredictorError(f"Initialization failed: {str(e)}")
    
    def _create_model(self, model_type: ModelType) -> BaseModel:
        """Create model instance based on type with validation"""
        model_map = {
            ModelType.RANDOM_FOREST: RandomForestModel,
            ModelType.XGBOOST: XGBoostModel,
            ModelType.LIGHTGBM: LightGBMModel,
            ModelType.ENSEMBLE: EnsembleModel,
            ModelType.NEURAL_NETWORK: lambda c: self._raise_not_implemented("NEURAL_NETWORK"),
            ModelType.LSTM: lambda c: self._raise_not_implemented("LSTM"),
            ModelType.TRANSFORMER: lambda c: self._raise_not_implemented("TRANSFORMER"),
        }
        
        model_class = model_map.get(model_type)
        if model_class is None:
            # NO FALLBACK - FAIL LOUD FOR ALL UNSUPPORTED TYPES
            raise NotImplementedError(f"Model type {model_type} is not implemented in enhanced_ml_predictor.py")
        
        try:
            return model_class(self.config)
        except Exception as e:
            logger.error(f"Failed to create model {model_type}: {str(e)}")
            raise ModelTrainingError(f"Failed to create model: {str(e)}")
    
    def _raise_not_implemented(self, model_name: str):
        """Raise NotImplementedError for unimplemented neural network models"""
        raise NotImplementedError(f"{model_name} model not yet implemented in enhanced_ml_predictor.py")
    
    @validate_input(InputValidator.validate_data_shape)
    def train_model(self, 
                   X: Union[np.ndarray, pd.DataFrame], 
                   y: Union[np.ndarray, pd.Series],
                   model_type: Optional[ModelType] = None,
                   version_notes: str = "") -> ModelVersion:
        """
        Train a new fraud detection model with comprehensive validation
        
        Args:
            X: Feature matrix
            y: Target labels
            model_type: Type of model to train
            version_notes: Notes about this model version
            
        Returns:
            Model version information
        """
        start_time = time.time()
        model_type = model_type or self.config.model_type
        
        try:
            logger.info(f"Starting model training with {model_type.value}")
            
            # Validate binary labels
            InputValidator.validate_binary_labels(y)
            
            # Convert to numpy if needed
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
                X = X.values
            else:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
            if isinstance(y, pd.Series):
                y = y.values
            
            # Check data quality
            self._validate_data_quality(X, y)
            
            # Split data with validation
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state,
                    stratify=y if self.config.stratify else None
                )
            except ValueError as e:
                logger.error(f"Data split failed: {str(e)}")
                raise ModelTrainingError(f"Failed to split data: {str(e)}")
            
            # Further split for validation if needed
            if self.config.validation_size > 0:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train,
                    test_size=self.config.validation_size,
                    random_state=self.config.random_state,
                    stratify=y_train if self.config.stratify else None
                )
            
            # Create and train model with error recovery
            model = self._create_model(model_type)
            model.feature_names = feature_names
            
            # Train with error handling
            training_attempts = 0
            max_attempts = 3
            
            while training_attempts < max_attempts:
                try:
                    # Hyperparameter tuning if enabled
                    if self.config.hyperparameter_tuning:
                        logger.info("Performing hyperparameter tuning...")
                        model = self._tune_hyperparameters(model, X_train, y_train)
                    else:
                        model.fit(X_train, y_train)
                    break
                except Exception as e:
                    training_attempts += 1
                    logger.warning(f"Training attempt {training_attempts} failed: {str(e)}")
                    if training_attempts >= max_attempts:
                        raise ModelTrainingError(f"Failed after {max_attempts} attempts: {str(e)}")
                    time.sleep(1)  # Brief pause before retry
            
            # Model calibration if enabled
            if self.config.calibrate_model:
                logger.info("Calibrating model probabilities...")
                model = self._calibrate_model(model, X_val if self.config.validation_size > 0 else X_test, 
                                            y_val if self.config.validation_size > 0 else y_test)
            
            # Evaluate model
            metrics = self._evaluate_model(model, X_test, y_test)
            
            # Calculate feature importance
            if self.config.calculate_feature_importance:
                metrics.feature_importance = model.get_feature_importance()
            
            training_time = time.time() - start_time
            metrics.training_time_seconds = training_time
            
            # Create version
            version_id = self._generate_version_id()
            version = ModelVersion(
                version_id=version_id,
                model_type=model_type,
                created_at=datetime.now(),
                metrics=metrics,
                status=ModelStatus.TRAINED,
                config=self.config,
                training_data_hash=self._hash_data(X, y),
                feature_names=feature_names,
                notes=version_notes
            )
            
            # Save model if configured
            if self.config.save_model:
                model_path = self.config.model_path / f"model_{version_id}.pkl"
                self._save_model(model, model_path)
                version.model_path = model_path
                metrics.model_size_bytes = model_path.stat().st_size
            
            # Store version
            with self._lock:
                self.model_versions[version_id] = version
                self.current_model = model
                
                # Set as fallback if it's the first model
                if self._fallback_model is None:
                    self._fallback_model = model
            
            # Validate against thresholds
            if self._validate_model(metrics):
                version.status = ModelStatus.VALIDATED
                logger.info(f"Model {version_id} validated successfully")
                
                # Set baseline for drift detection
                if self.drift_detector:
                    self.drift_detector.set_baseline(metrics)
            else:
                version.status = ModelStatus.FAILED
                logger.warning(f"Model {version_id} failed validation")
            
            logger.info(f"Model training completed in {training_time:.2f} seconds")
            logger.info(f"Model performance - AUC: {metrics.auc_roc:.3f}, "
                       f"Precision: {metrics.precision:.3f}, Recall: {metrics.recall:.3f}")
            
            return version
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            logger.debug(traceback.format_exc())
            raise ModelTrainingError(f"Training failed: {str(e)}")
    
    def predict(self, features: Union[Dict[str, Any], pd.DataFrame, np.ndarray],
               transaction_id: Optional[str] = None,
               use_fallback: bool = True) -> Union[PredictionResult, List[PredictionResult]]:
        """
        Predict fraud probability for transaction(s) with error recovery
        
        Args:
            features: Transaction features
            transaction_id: Optional transaction ID
            use_fallback: Whether to use fallback model on failure
            
        Returns:
            Prediction result(s)
        """
        if self.current_model is None or not self.current_model.fitted:
            if use_fallback and self._fallback_model is not None:
                logger.warning("Using fallback model for prediction")
                model = self._fallback_model
            else:
                raise ModelNotFittedError("No trained model available for prediction")
        else:
            model = self.current_model
        
        start_time = time.time()
        
        try:
            # Validate and prepare features
            if isinstance(features, dict):
                X = self._prepare_features_dict(features)
                single_prediction = True
            elif isinstance(features, pd.DataFrame):
                InputValidator.validate_features(features, model.feature_names)
                X = features.values
                single_prediction = False
            else:
                X = features
                single_prediction = len(X.shape) == 1
                if single_prediction:
                    X = X.reshape(1, -1)
            
            # Validate feature dimensions
            if X.shape[1] != len(model.feature_names):
                raise FeatureMismatchError(
                    f"Feature count mismatch: expected {len(model.feature_names)}, got {X.shape[1]}"
                )
            
            # Make predictions with error handling
            prediction_attempts = 0
            max_attempts = 2
            
            while prediction_attempts < max_attempts:
                try:
                    probabilities = model.predict_proba(X)
                    fraud_probabilities = probabilities[:, 1]
                    break
                except Exception as e:
                    prediction_attempts += 1
                    logger.warning(f"Prediction attempt {prediction_attempts} failed: {str(e)}")
                    
                    if prediction_attempts >= max_attempts:
                        if use_fallback and self._fallback_model is not None and model != self._fallback_model:
                            logger.warning("Switching to fallback model")
                            model = self._fallback_model
                            prediction_attempts = 0  # Reset for fallback
                        else:
                            with self._lock:
                                self.prediction_stats['failed_predictions'] += 1
                            raise PredictionError(f"Prediction failed after {max_attempts} attempts: {str(e)}")
            
            # Create prediction results
            results = []
            for i, prob in enumerate(fraud_probabilities):
                # Validate probability
                if not 0 <= prob <= 1:
                    logger.warning(f"Invalid probability {prob}, clamping to [0, 1]")
                    prob = np.clip(prob, 0, 1)
                
                # Calculate risk score and level
                risk_score = self._calculate_risk_score(prob)
                risk_level = self._determine_risk_level(risk_score)
                
                # Validate prediction
                validation_warnings = self._validate_prediction(prob, risk_score)
                
                # Create result
                result = PredictionResult(
                    transaction_id=transaction_id if single_prediction else f"TXN_{i}",
                    fraud_probability=float(prob),
                    risk_score=float(risk_score),
                    is_fraud=prob >= self.config.fraud_threshold,
                    risk_level=risk_level,
                    confidence=float(max(prob, 1 - prob)),
                    model_version=self.active_version or "unknown",
                    risk_factors=self._identify_risk_factors(X[i], prob),
                    prediction_valid=len(validation_warnings) == 0,
                    validation_warnings=validation_warnings
                )
                results.append(result)
                
                # Update statistics
                with self._lock:
                    self.prediction_stats['total_predictions'] += 1
                    if result.is_fraud:
                        self.prediction_stats['total_frauds_detected'] += 1
            
            prediction_time = time.time() - start_time
            
            # Update timing statistics
            with self._lock:
                total_predictions = self.prediction_stats['total_predictions']
                avg_time = self.prediction_stats['average_prediction_time']
                self.prediction_stats['average_prediction_time'] = (
                    (avg_time * (total_predictions - len(results)) + prediction_time) / total_predictions
                )
                self.prediction_stats['last_prediction_time'] = datetime.now()
            
            logger.info(f"Predicted {len(results)} transactions in {prediction_time:.3f} seconds")
            
            return results[0] if single_prediction else results
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            logger.debug(traceback.format_exc())
            
            with self._lock:
                self.prediction_stats['failed_predictions'] += 1
            
            # Return safe default if fallback is enabled
            if use_fallback:
                logger.warning("Returning safe default prediction due to error")
                return PredictionResult(
                    transaction_id=transaction_id or "ERROR",
                    fraud_probability=0.5,
                    risk_score=0.5,
                    is_fraud=True,  # Conservative approach
                    risk_level="high",
                    confidence=0.0,
                    prediction_valid=False,
                    validation_warnings=[f"Prediction error: {str(e)}"]
                )
            else:
                raise
    
    def _validate_data_quality(self, X: np.ndarray, y: np.ndarray):
        """Validate data quality before training"""
        # Check for NaN values
        if np.any(np.isnan(X)):
            raise InvalidInputError("Input data contains NaN values")
        if np.any(np.isnan(y)):
            raise InvalidInputError("Target data contains NaN values")
        
        # Check for infinite values
        if np.any(np.isinf(X)):
            raise InvalidInputError("Input data contains infinite values")
        
        # Check class balance
        unique, counts = np.unique(y, return_counts=True)
        class_ratio = counts.min() / counts.max()
        if class_ratio < 0.01:
            logger.warning(f"Severe class imbalance detected: {class_ratio:.3f}")
        
        # Check variance
        variances = np.var(X, axis=0)
        zero_variance_features = np.sum(variances == 0)
        if zero_variance_features > 0:
            logger.warning(f"Found {zero_variance_features} zero-variance features")
    
    def _validate_prediction(self, probability: float, risk_score: float) -> List[str]:
        """Validate individual prediction"""
        warnings = []
        
        # Check probability bounds
        if not 0 <= probability <= 1:
            warnings.append(f"Probability {probability} outside valid range")
        
        # Check risk score bounds
        if not 0 <= risk_score <= 1:
            warnings.append(f"Risk score {risk_score} outside valid range")
        
        # Check consistency
        if probability > 0.8 and risk_score < 0.5:
            warnings.append("Inconsistent probability and risk score")
        
        return warnings
    
    @handle_errors(fallback_value=False)
    def validate_model_performance(self, X: Union[np.ndarray, pd.DataFrame], 
                                 y: Union[np.ndarray, pd.Series]) -> bool:
        """
        Validate model performance on new data
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Whether model meets performance thresholds
        """
        if self.current_model is None:
            raise ModelNotFittedError("No model available for validation")
        
        # Evaluate model
        metrics = self._evaluate_model(self.current_model, X, y)
        
        # Check drift if enabled
        if self.drift_detector and self.config.enable_drift_detection:
            is_drifted, drift_score = self.drift_detector.check_drift(metrics)
            if is_drifted:
                with self._lock:
                    self.prediction_stats['drift_warnings'] += 1
                logger.warning(f"Model drift detected: {drift_score:.3f}")
        
        # Validate against thresholds
        return self._validate_model(metrics)
    
    def _evaluate_model(self, model: BaseModel, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Internal method to evaluate model with error handling"""
        try:
            # Make predictions
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]
            
            # Calculate metrics with error handling
            metrics = ModelMetrics(
                accuracy=accuracy_score(y, y_pred),
                precision=precision_score(y, y_pred, zero_division=0),
                recall=recall_score(y, y_pred, zero_division=0),
                f1_score=f1_score(y, y_pred, zero_division=0),
                auc_roc=roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0,
                confusion_matrix=confusion_matrix(y, y_pred),
                classification_report=classification_report(y, y_pred, output_dict=True, zero_division=0)
            )
            
            # Calculate additional metrics
            tn, fp, fn, tp = metrics.confusion_matrix.ravel()
            metrics.true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics.false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # Precision-Recall curve
            try:
                precision_curve, recall_curve, thresholds = precision_recall_curve(y, y_proba)
                metrics.precision_recall_auc = np.trapz(precision_curve, recall_curve)
                
                # Store threshold analysis (limited)
                metrics.thresholds = thresholds.tolist()[:100]
                metrics.precision_by_threshold = precision_curve.tolist()[:100]
                metrics.recall_by_threshold = recall_curve.tolist()[:100]
            except Exception as e:
                logger.warning(f"Failed to calculate PR curve: {str(e)}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            # Return default metrics on failure
            return ModelMetrics(
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                auc_roc=0.0,
                confusion_matrix=np.array([[0, 0], [0, 0]]),
                classification_report={}
            )
    
    @handle_errors()
    def _tune_hyperparameters(self, model: BaseModel, X: np.ndarray, y: np.ndarray) -> BaseModel:
        """Tune model hyperparameters with validation"""
        # Define parameter grids for different model types
        param_grids = {
            ModelType.RANDOM_FOREST: {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [10, 20, 50],
                'min_samples_leaf': [5, 10, 20]
            },
            ModelType.XGBOOST: {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0]
            },
            ModelType.LIGHTGBM: {
                'n_estimators': [50, 100, 200],
                'num_leaves': [15, 31, 63],
                'learning_rate': [0.01, 0.1, 0.3],
                'feature_fraction': [0.6, 0.8, 1.0]
            }
        }
        
        # Get parameter grid for model type
        param_grid = param_grids.get(self.config.model_type, {})
        
        if not param_grid:
            # No tuning for this model type
            model.fit(X, y)
            return model
        
        try:
            # Perform grid search with timeout
            if self.config.tuning_method == "grid":
                search = GridSearchCV(
                    model.model,
                    param_grid,
                    cv=self.config.tuning_cv_folds,
                    scoring=self.config.cv_scoring,
                    n_jobs=-1,
                    verbose=1,
                    error_score='raise'
                )
            else:
                search = RandomizedSearchCV(
                    model.model,
                    param_grid,
                    n_iter=self.config.tuning_iterations,
                    cv=self.config.tuning_cv_folds,
                    scoring=self.config.cv_scoring,
                    n_jobs=-1,
                    verbose=1,
                    error_score='raise'
                )
            
            search.fit(X, y)
            
            # Update model with best parameters
            model.model = search.best_estimator_
            model.fitted = True
            
            logger.info(f"Best parameters: {search.best_params_}")
            logger.info(f"Best CV score: {search.best_score_:.3f}")
            
            return model
            
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {str(e)}")
            logger.warning("Falling back to default parameters")
            model.fit(X, y)
            return model
    
    @handle_errors()
    def _calibrate_model(self, model: BaseModel, X: np.ndarray, y: np.ndarray) -> BaseModel:
        """Calibrate model probabilities with error handling"""
        try:
            calibrated = CalibratedClassifierCV(
                model.model,
                method=self.config.calibration_method,
                cv='prefit'
            )
            calibrated.fit(X, y)
            
            # Wrap calibrated model
            model.model = calibrated
            return model
            
        except Exception as e:
            logger.error(f"Model calibration failed: {str(e)}")
            logger.warning("Using uncalibrated model")
            return model
    
    def _calculate_risk_score(self, fraud_probability: float) -> float:
        """Calculate risk score from fraud probability with validation"""
        # Ensure probability is valid
        fraud_probability = np.clip(fraud_probability, 0, 1)
        
        # Apply non-linear transformation for better risk differentiation
        if fraud_probability < 0.5:
            risk_score = fraud_probability * 0.6
        elif fraud_probability < 0.8:
            risk_score = 0.3 + (fraud_probability - 0.5) * 1.2
        else:
            risk_score = 0.66 + (fraud_probability - 0.8) * 1.7
        
        return min(risk_score, 1.0)
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level from risk score"""
        if risk_score < 0.3:
            return "low"
        elif risk_score < 0.6:
            return "medium"
        elif risk_score < 0.8:
            return "high"
        else:
            return "critical"
    
    def _identify_risk_factors(self, features: np.ndarray, fraud_probability: float) -> List[str]:
        """Identify top risk factors for a prediction"""
        risk_factors = []
        
        try:
            # Get feature importance if available
            if self.current_model and hasattr(self.current_model, 'get_feature_importance'):
                importance = self.current_model.get_feature_importance()
                if importance:
                    # Sort by importance
                    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    
                    # Add top contributing features
                    for feature, imp in sorted_features[:5]:
                        if imp > 0.05:  # Only significant features
                            risk_factors.append(f"{feature} (importance: {imp:.2f})")
        except Exception as e:
            logger.warning(f"Failed to get risk factors: {str(e)}")
        
        # Add probability-based factors
        if fraud_probability > 0.9:
            risk_factors.append("Very high fraud probability")
        elif fraud_probability > 0.7:
            risk_factors.append("High fraud probability")
        
        return risk_factors
    
    def _validate_model(self, metrics: ModelMetrics) -> bool:
        """Validate model against performance thresholds"""
        return (
            metrics.auc_roc >= self.config.min_auc and
            metrics.precision >= self.config.min_precision and
            metrics.recall >= self.config.min_recall
        )
    
    def _prepare_features_dict(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare features from dictionary with validation"""
        if not self.current_model or not self.current_model.feature_names:
            raise ModelNotFittedError("Model must be trained before preparing features")
        
        # Extract features in correct order
        feature_values = []
        missing_features = []
        
        for feature_name in self.current_model.feature_names:
            if feature_name in features:
                value = features[feature_name]
                # Validate numeric value
                if not isinstance(value, (int, float)):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        logger.warning(f"Non-numeric value for {feature_name}, using 0")
                        value = 0
                feature_values.append(value)
            else:
                missing_features.append(feature_name)
                feature_values.append(0)  # Default to 0 if missing
        
        if missing_features:
            logger.warning(f"Missing features (using defaults): {missing_features[:10]}")  # Limit log size
        
        return np.array(feature_values).reshape(1, -1)
    
    def _generate_version_id(self) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"{self.config.model_type.value}_{timestamp}_{random_suffix}"
    
    def _hash_data(self, X: np.ndarray, y: np.ndarray) -> str:
        """Generate hash of training data"""
        data_str = f"{X.shape}_{y.shape}_{np.mean(X)}_{np.mean(y)}"
        return hashlib.md5(data_str.encode()).hexdigest()
    
    @handle_errors()
    def _save_model(self, model: BaseModel, path: Path) -> None:
        """Save model to disk with error handling"""
        try:
            # Create signature if enabled
            if self.config.model_signature:
                signature = {
                    'model_type': self.config.model_type.value,
                    'feature_names': model.feature_names,
                    'created_at': datetime.now().isoformat(),
                    'config_hash': hashlib.md5(str(self.config.__dict__).encode()).hexdigest()
                }
                
                # Save model and signature
                save_data = {
                    'model': model,
                    'signature': signature
                }
            else:
                save_data = model
            
            with open(path, 'wb') as f:
                joblib.dump(save_data, f)
            
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise ModelLoadError(f"Failed to save model: {str(e)}")
    
    @handle_errors(fallback_value=False)
    def load_model(self, model_name: str) -> bool:
        """
        Load a trained model with validation
        
        Args:
            model_name: Model version ID or path
            
        Returns:
            Success status
        """
        try:
            # Check if it's a version ID
            if model_name in self.model_versions:
                version = self.model_versions[model_name]
                if version.model_path and version.model_path.exists():
                    with open(version.model_path, 'rb') as f:
                        loaded_data = joblib.load(f)
                    
                    # Validate signature if enabled
                    if self.config.model_signature and isinstance(loaded_data, dict):
                        signature = loaded_data.get('signature', {})
                        if signature.get('model_type') != self.config.model_type.value:
                            raise ModelLoadError("Model type mismatch")
                        self.current_model = loaded_data['model']
                    else:
                        self.current_model = loaded_data
                    
                    self.active_version = model_name
                    logger.info(f"Loaded model version {model_name}")
                    return True
            
            # Try as file path
            model_path = Path(model_name)
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    loaded_data = joblib.load(f)
                
                if self.config.model_signature and isinstance(loaded_data, dict):
                    self.current_model = loaded_data['model']
                else:
                    self.current_model = loaded_data
                
                self.active_version = model_path.stem
                logger.info(f"Loaded model from {model_path}")
                return True
            
            logger.error(f"Model {model_name} not found")
            return False
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    @handle_errors(fallback_value=False)
    def deploy_model(self, version_id: str) -> bool:
        """
        Deploy a model version to production with validation
        
        Args:
            version_id: Version to deploy
            
        Returns:
            Success status
        """
        if version_id not in self.model_versions:
            logger.error(f"Version {version_id} not found")
            return False
        
        version = self.model_versions[version_id]
        
        # Validate model is ready
        if version.status != ModelStatus.VALIDATED:
            logger.error(f"Model {version_id} is not validated")
            return False
        
        # Load model
        if not self.load_model(version_id):
            return False
        
        # Update version status
        with self._lock:
            # Retire current active version
            for v_id, v in self.model_versions.items():
                if v.is_active and v_id != version_id:
                    v.is_active = False
                    v.retired_at = datetime.now()
            
            # Activate new version
            version.is_active = True
            version.deployed_at = datetime.now()
            version.status = ModelStatus.DEPLOYED
            self.active_version = version_id
        
        logger.info(f"Model {version_id} deployed successfully")
        return True
    
    def get_model_versions(self) -> List[ModelVersion]:
        """Get all model versions"""
        with self._lock:
            return list(self.model_versions.values())
    
    def get_active_model_info(self) -> Optional[ModelVersion]:
        """Get information about active model"""
        if self.active_version:
            return self.model_versions.get(self.active_version)
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get predictor statistics with enhanced metrics"""
        with self._lock:
            stats = self.prediction_stats.copy()
            stats.update({
                'total_models_trained': len(self.model_versions),
                'active_model': self.active_version,
                'fraud_detection_rate': (
                    self.prediction_stats['total_frauds_detected'] / 
                    self.prediction_stats['total_predictions']
                    if self.prediction_stats['total_predictions'] > 0 else 0
                ),
                'prediction_success_rate': (
                    1 - (self.prediction_stats['failed_predictions'] / 
                         max(self.prediction_stats['total_predictions'], 1))
                ),
                'has_fallback_model': self._fallback_model is not None
            })
            return stats
    
    def reset_statistics(self) -> None:
        """Reset prediction statistics"""
        with self._lock:
            self.prediction_stats = {
                'total_predictions': 0,
                'total_frauds_detected': 0,
                'average_prediction_time': 0.0,
                'last_prediction_time': None,
                'failed_predictions': 0,
                'drift_warnings': 0
            }
    
    @contextmanager
    def model_monitoring(self):
        """Context manager for model monitoring"""
        start_time = time.time()
        initial_stats = self.get_statistics()
        
        try:
            yield self
        finally:
            # Log monitoring results
            duration = time.time() - start_time
            final_stats = self.get_statistics()
            
            predictions_made = final_stats['total_predictions'] - initial_stats['total_predictions']
            failures = final_stats['failed_predictions'] - initial_stats['failed_predictions']
            
            logger.info(f"Model monitoring session completed:")
            logger.info(f"  Duration: {duration:.2f} seconds")
            logger.info(f"  Predictions: {predictions_made}")
            logger.info(f"  Failures: {failures}")
            logger.info(f"  Success rate: {1 - (failures / max(predictions_made, 1)):.2%}")
    
    def __repr__(self) -> str:
        return (f"EnhancedFinancialMLPredictor(model_type={self.config.model_type.value}, "
               f"versions={len(self.model_versions)}, "
               f"active={self.active_version}, "
               f"predictions={self.prediction_stats['total_predictions']}, "
               f"failures={self.prediction_stats['failed_predictions']})")

# Legacy compatibility - alias for backward compatibility
MLPredictor = EnhancedFinancialMLPredictor
FinancialMLPredictor = EnhancedFinancialMLPredictor

# Export main classes
__all__ = [
    'FinancialMLPredictor',
    'MLPredictor',  # Legacy compatibility
    'ModelConfig',
    'ModelMetrics',
    'PredictionResult',
    'ModelVersion',
    'ModelType',
    'ModelStatus',
    'MLPredictorError',
    'ModelNotFittedError',
    'InvalidInputError',
    'ModelValidationError',
    'FeatureMismatchError',
    'ModelDriftError',
    'ModelLoadError',
    'HyperparameterError',
    'ModelTrainingError',
    'PredictionError',
    'InputValidator',
    'ModelDriftDetector'
]