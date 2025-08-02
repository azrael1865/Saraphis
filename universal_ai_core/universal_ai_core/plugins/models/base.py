#!/usr/bin/env python3
"""
Model Plugin Base Classes
=========================

This module provides abstract base classes for machine learning model plugins in the Universal AI Core system.
Adapted from existing model patterns in the Saraphis codebase, made domain-agnostic.

Base Classes:
- ModelPlugin: Abstract base for all ML models
- TrainingResult: Standardized training result container
- PredictionResult: Standardized prediction result container
- ModelMetadata: Model metadata and versioning
"""

import logging
import time
import pickle
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of machine learning models"""
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    CLUSTERER = "clusterer"
    TRANSFORMER = "transformer"
    ENCODER = "encoder"
    DECODER = "decoder"
    GENERATIVE = "generative"
    REINFORCEMENT = "reinforcement"


class TaskType(Enum):
    """Types of machine learning tasks"""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    FEATURE_SELECTION = "feature_selection"
    ANOMALY_DETECTION = "anomaly_detection"


class ModelStatus(Enum):
    """Status of model operations"""
    SUCCESS = "success"
    FAILED = "failed"
    TRAINING = "training"
    PREDICTING = "predicting"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class ModelMetadata:
    """Metadata for model plugins"""
    name: str
    version: str
    author: str
    description: str
    model_type: ModelType
    supported_task_types: List[TaskType]
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    dependencies: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    model_id: str = ""
    
    def __post_init__(self):
        """Generate model ID if not provided"""
        if not self.model_id:
            content = f"{self.name}:{self.version}:{self.author}:{self.model_type.value}"
            self.model_id = hashlib.md5(content.encode()).hexdigest()


@dataclass
class TrainingResult:
    """Result container for model training operations"""
    success: bool
    training_time: float
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    model_size: Optional[int] = None
    epochs_completed: Optional[int] = None
    early_stopped: bool = False
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate training result"""
        if not self.success and not self.error_message:
            self.error_message = "Training failed"


@dataclass
class PredictionResult:
    """Result container for model prediction operations"""
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    confidence_scores: Optional[np.ndarray] = None
    prediction_time: float = 0.0
    uncertainty: Optional[np.ndarray] = None
    explanations: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class ModelPlugin(ABC):
    """
    Abstract base class for machine learning model plugins.
    
    This class defines the interface that all model plugins must implement.
    Adapted from existing model patterns in the Saraphis codebase.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model plugin.
        
        Args:
            config: Plugin-specific configuration dictionary
        """
        self.config = config or {}
        self._metadata = self._create_metadata()
        self._model = None
        self._is_trained = False
        self._is_initialized = False
        self._training_history = []
        self._prediction_cache = {}
        self._feature_names = []
        self._target_names = []
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"Initialized model: {self._metadata.name}")
    
    @abstractmethod
    def _create_metadata(self) -> ModelMetadata:
        """
        Create metadata for this model plugin.
        
        Returns:
            ModelMetadata instance with model information
        """
        pass
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> TrainingResult:
        """
        Train the model on provided data.
        
        Args:
            X: Training features
            y: Training targets
            validation_data: Optional validation data tuple (X_val, y_val)
            
        Returns:
            TrainingResult containing training metrics and status
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        Make predictions on provided data.
        
        Args:
            X: Input features for prediction
            
        Returns:
            PredictionResult containing predictions and metadata
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> bool:
        """
        Save the trained model to file.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model from file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def initialize(self) -> bool:
        """
        Initialize the model plugin. Called before first use.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self._perform_initialization()
            self._is_initialized = True
            logger.info(f"Model {self._metadata.name} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize model {self._metadata.name}: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown the model plugin and clean up resources"""
        try:
            self._perform_shutdown()
            self._is_initialized = False
            self._model = None
            logger.info(f"Model {self._metadata.name} shutdown successfully")
        except Exception as e:
            logger.error(f"Error shutting down model {self._metadata.name}: {e}")
    
    def get_metadata(self) -> ModelMetadata:
        """Get model metadata"""
        return self._metadata
    
    def is_trained(self) -> bool:
        """Check if model is trained"""
        return self._is_trained
    
    def is_initialized(self) -> bool:
        """Check if model is initialized"""
        return self._is_initialized
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters"""
        return self._metadata.hyperparameters.copy()
    
    def set_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Set hyperparameters for the model.
        
        Args:
            hyperparams: Dictionary of hyperparameter values
        """
        self._validate_hyperparameters(hyperparams)
        self._metadata.hyperparameters.update(hyperparams)
        self._metadata.updated_at = datetime.utcnow()
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores if available.
        
        Returns:
            Array of feature importance scores or None
        """
        return None
    
    def explain_prediction(self, X: np.ndarray, method: str = "default") -> Dict[str, Any]:
        """
        Explain model predictions.
        
        Args:
            X: Input features to explain
            method: Explanation method to use
            
        Returns:
            Dictionary containing explanation results
        """
        return {"explanation": "Not implemented for this model"}
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Features for cross-validation
            y: Targets for cross-validation
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation metrics
        """
        # Basic implementation - override in subclasses for specific models
        from sklearn.model_selection import cross_val_score
        
        if not hasattr(self._model, 'fit') or not hasattr(self._model, 'score'):
            return {"error": "Model not compatible with sklearn cross_val_score"}
        
        try:
            scores = cross_val_score(self._model, X, y, cv=cv_folds)
            return {
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "scores": scores.tolist()
            }
        except Exception as e:
            return {"error": str(e)}
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on provided data.
        
        Args:
            X: Features for evaluation
            y: True targets for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        prediction_result = self.predict(X)
        predictions = prediction_result.predictions
        
        return self._compute_metrics(y, predictions, prediction_result.probabilities)
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get the training history"""
        return self._training_history.copy()
    
    def clear_cache(self) -> None:
        """Clear the prediction cache"""
        self._prediction_cache.clear()
        logger.info(f"Cleared prediction cache for {self._metadata.name}")
    
    def get_model_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the model.
        
        Returns:
            Dictionary with model statistics
        """
        return {
            "model_name": self._metadata.name,
            "model_type": self._metadata.model_type.value,
            "is_trained": self._is_trained,
            "is_initialized": self._is_initialized,
            "training_runs": len(self._training_history),
            "cache_size": len(self._prediction_cache),
            "feature_count": len(self._feature_names),
            "target_count": len(self._target_names),
            "model_id": self._metadata.model_id
        }
    
    def _validate_config(self) -> None:
        """Validate model configuration. Override in subclasses."""
        pass
    
    def _validate_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """Validate hyperparameters. Override in subclasses."""
        pass
    
    def _perform_initialization(self) -> None:
        """Perform model-specific initialization. Override in subclasses."""
        pass
    
    def _perform_shutdown(self) -> None:
        """Perform model-specific shutdown. Override in subclasses."""
        pass
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute evaluation metrics based on task type.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            y_prob: Predicted probabilities (for classification)
            
        Returns:
            Dictionary with computed metrics
        """
        metrics = {}
        
        try:
            # Import sklearn metrics
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                mean_squared_error, mean_absolute_error, r2_score,
                roc_auc_score
            )
            
            # Determine task type from data
            if len(np.unique(y_true)) <= 10 and np.all(y_true == y_true.astype(int)):
                # Classification task
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                
                # Handle multiclass vs binary
                avg_method = 'binary' if len(np.unique(y_true)) == 2 else 'macro'
                
                metrics['precision'] = precision_score(y_true, y_pred, average=avg_method)
                metrics['recall'] = recall_score(y_true, y_pred, average=avg_method)
                metrics['f1_score'] = f1_score(y_true, y_pred, average=avg_method)
                
                # AUC for binary classification with probabilities
                if len(np.unique(y_true)) == 2 and y_prob is not None:
                    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                        # Use positive class probability
                        y_prob_binary = y_prob[:, 1]
                    else:
                        y_prob_binary = y_prob.flatten()
                    metrics['auc'] = roc_auc_score(y_true, y_prob_binary)
            
            else:
                # Regression task
                metrics['mse'] = mean_squared_error(y_true, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['mae'] = mean_absolute_error(y_true, y_pred)
                metrics['r2'] = r2_score(y_true, y_pred)
        
        except ImportError:
            logger.warning("Scikit-learn not available for metric computation")
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
        
        return metrics
    
    def _generate_cache_key(self, X: np.ndarray) -> str:
        """Generate cache key for predictions"""
        try:
            # Create hash of input data
            return hashlib.md5(X.tobytes()).hexdigest()
        except Exception:
            # Fallback for non-numeric data
            return hashlib.md5(str(X).encode()).hexdigest()


class BaseClassifier(ModelPlugin):
    """Base class for classification models"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._classes = None
        self._n_classes = 0
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Array of class probabilities
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Default implementation - override in subclasses
        pred_result = self.predict(X)
        return pred_result.probabilities if pred_result.probabilities is not None else np.array([])
    
    def get_classes(self) -> np.ndarray:
        """Get the classes learned during training"""
        if self._classes is None:
            return np.array([])
        return self._classes.copy()


class BaseRegressor(ModelPlugin):
    """Base class for regression models"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._output_range = None
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        pred_result = self.predict(X)
        uncertainties = pred_result.uncertainty if pred_result.uncertainty is not None else np.zeros_like(pred_result.predictions)
        return pred_result.predictions, uncertainties


# Example implementation for testing
class ExampleClassifier(BaseClassifier):
    """Example classifier for testing purposes"""
    
    def _create_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="ExampleClassifier",
            version="1.0.0",
            author="Universal AI Core",
            description="Example classifier for testing",
            model_type=ModelType.CLASSIFIER,
            supported_task_types=[TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION],
            hyperparameters={"n_estimators": 100, "random_state": 42}
        )
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> TrainingResult:
        """Train the example classifier"""
        start_time = time.time()
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            # Initialize model
            self._model = RandomForestClassifier(**self._metadata.hyperparameters)
            
            # Train model
            self._model.fit(X, y)
            
            # Store training info
            self._classes = np.unique(y)
            self._n_classes = len(self._classes)
            self._feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            self._is_trained = True
            
            training_time = time.time() - start_time
            
            # Compute training metrics
            train_pred = self._model.predict(X)
            train_metrics = self._compute_metrics(y, train_pred)
            
            # Validation metrics if provided
            val_metrics = {}
            if validation_data is not None:
                X_val, y_val = validation_data
                val_pred = self._model.predict(X_val)
                val_metrics = self._compute_metrics(y_val, val_pred)
            
            result = TrainingResult(
                success=True,
                training_time=training_time,
                metrics={**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
                model_size=len(pickle.dumps(self._model))
            )
            
            self._training_history.append({
                "timestamp": datetime.utcnow(),
                "result": result,
                "data_shape": X.shape
            })
            
            return result
            
        except Exception as e:
            return TrainingResult(
                success=False,
                training_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """Make predictions with the classifier"""
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        start_time = time.time()
        
        try:
            predictions = self._model.predict(X)
            probabilities = self._model.predict_proba(X)
            confidence_scores = np.max(probabilities, axis=1)
            
            return PredictionResult(
                predictions=predictions,
                probabilities=probabilities,
                confidence_scores=confidence_scores,
                prediction_time=time.time() - start_time
            )
            
        except Exception as e:
            return PredictionResult(
                predictions=np.array([]),
                prediction_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def save_model(self, filepath: str) -> bool:
        """Save the model"""
        try:
            model_data = {
                'model': self._model,
                'metadata': self._metadata,
                'classes': self._classes,
                'feature_names': self._feature_names,
                'is_trained': self._is_trained
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load the model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self._model = model_data['model']
            self._metadata = model_data['metadata']
            self._classes = model_data['classes']
            self._feature_names = model_data['feature_names']
            self._is_trained = model_data['is_trained']
            
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance from the random forest"""
        if self._model and hasattr(self._model, 'feature_importances_'):
            return self._model.feature_importances_
        return None


# Plugin factory function
def create_model(model_type: str, config: Optional[Dict[str, Any]] = None) -> ModelPlugin:
    """
    Factory function to create model plugins.
    
    Args:
        model_type: Type of model to create
        config: Configuration for the model
        
    Returns:
        ModelPlugin instance
    """
    models = {
        'example_classifier': ExampleClassifier,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](config)


if __name__ == "__main__":
    # Test the model plugin base classes
    print("🤖 Model Plugin Base Classes Test")
    print("=" * 50)
    
    # Create test data
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = np.random.randint(0, 3, 100)
    
    # Test example classifier
    model = create_model('example_classifier')
    
    # Initialize
    success = model.initialize()
    print(f"✅ Model initialized: {success}")
    
    # Test metadata
    metadata = model.get_metadata()
    print(f"🤖 Model: {metadata.name} v{metadata.version}")
    print(f"📋 Type: {metadata.model_type.value}")
    
    # Test training
    result = model.train(X, y)
    print(f"🏋️ Training success: {result.success}")
    print(f"⏱️ Training time: {result.training_time:.4f}s")
    print(f"📊 Training accuracy: {result.metrics.get('accuracy', 'N/A'):.4f}")
    
    # Test prediction
    pred_result = model.predict(X[:10])
    print(f"🔮 Predictions shape: {pred_result.predictions.shape}")
    print(f"📈 Probabilities shape: {pred_result.probabilities.shape}")
    
    # Test evaluation
    eval_metrics = model.evaluate(X, y)
    print(f"📊 Evaluation metrics: {list(eval_metrics.keys())}")
    
    # Test feature importance
    importance = model.get_feature_importance()
    if importance is not None:
        print(f"🎯 Feature importance available: {len(importance)} features")
    
    # Test model stats
    stats = model.get_model_stats()
    print(f"📈 Model stats: {stats['training_runs']} training runs")
    
    # Shutdown
    model.shutdown()
    print("\n✅ Model plugin test completed!")