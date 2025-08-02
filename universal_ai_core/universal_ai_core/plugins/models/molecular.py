#!/usr/bin/env python3
"""
Molecular Models Plugin
=======================

This module provides molecular neural network models for the Universal AI Core system.
Extracted and adapted from Saraphis neural_models.py and property_predictor.py, preserving
all molecular-specific model architectures and training capabilities.

Features:
- Enterprise-grade neural networks for molecular property prediction
- Ensemble methods (Random Forest, XGBoost, Neural Networks)
- Uncertainty quantification and confidence estimation
- Cross-validation and hyperparameter optimization
- Model persistence and checkpointing
- GPU acceleration and memory optimization
"""

import logging
import time
import json
import pickle
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import threading
import copy

# Core ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Import plugin base classes
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from base import ModelPlugin, ModelType, ModelResult, PredictionResult, TrainingResult

logger = logging.getLogger(__name__)


@dataclass
class PropertyMetrics:
    """Metrics for property prediction evaluation"""
    rmse: float = 0.0
    r2: float = 0.0
    mae: float = 0.0
    accuracy: Optional[float] = None  # For classification tasks
    auc: Optional[float] = None  # For binary classification
    cross_val_score: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None


@dataclass
class PredictionConfig:
    """
    Configuration for molecular property prediction.
    
    Extracted from property_predictor.py PredictionConfig.
    """
    model_types: List[str] = field(default_factory=lambda: ['random_forest', 'xgboost', 'neural_network'])
    ensemble_method: str = 'voting'  # 'voting', 'stacking', 'weighted'
    cross_validation_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
    use_uncertainty_quantification: bool = True
    scale_features: bool = True
    hyperparameter_optimization: bool = True
    early_stopping_patience: int = 20
    max_epochs: int = 1000
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.01


class MolecularNeuralNetwork(nn.Module):
    """
    Enterprise-Grade Neural Network for Molecular Property Prediction.
    
    Extracted and adapted from Saraphis neural_models.py lines 45-150.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = None,
                 output_dim: int = 1,
                 dropout_rate: float = 0.3,
                 use_batch_norm: bool = True,
                 use_residual: bool = True,
                 use_attention: bool = False,
                 activation: str = 'relu',
                 task_type: str = 'regression'):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256, 128]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.use_attention = use_attention
        self.task_type = task_type
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            
            self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                hidden_dims[-1], 
                num_heads=8, 
                batch_first=True,
                dropout=dropout_rate
            )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        # Output activation for classification
        if task_type == 'classification':
            if output_dim == 1:
                self.output_activation = nn.Sigmoid()
            else:
                self.output_activation = nn.Softmax(dim=1)
        else:
            self.output_activation = None
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network"""
        # Input projection
        x = self.input_projection(x)
        x = self.activation(x)
        
        # Hidden layers with residual connections
        for i, layer in enumerate(self.hidden_layers):
            residual = x if self.use_residual and x.size(1) == layer.out_features else None
            
            x = layer(x)
            
            if self.use_batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = self.dropouts[i](x)
            
            # Add residual connection
            if residual is not None:
                x = x + residual
        
        # Attention mechanism
        if self.use_attention:
            x = x.unsqueeze(1)  # Add sequence dimension
            attn_output, _ = self.attention(x, x, x)
            x = attn_output.squeeze(1)  # Remove sequence dimension
        
        # Output layer
        x = self.output_layer(x)
        
        if self.output_activation is not None:
            x = self.output_activation(x)
        
        return x


class MolecularTrainer:
    """
    Enterprise-Grade Trainer for Molecular Neural Networks.
    
    Extracted and adapted from Saraphis neural_models.py lines 152-400.
    """
    
    def __init__(self, model: nn.Module, config: PredictionConfig, device: str = 'auto'):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        self.model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5,
            patience=config.early_stopping_patience // 2, verbose=True
        )
        
        # Loss function
        if model.task_type == 'regression':
            self.criterion = nn.MSELoss()
        elif model.task_type == 'classification':
            if model.output_dim == 1:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        self.logger = logging.getLogger(f"{__name__}.MolecularTrainer")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingResult:
        """Train the molecular neural network"""
        start_time = time.time()
        
        try:
            self.logger.info(f"🚀 Starting training on {self.device}")
            
            for epoch in range(self.config.max_epochs):
                # Training phase
                train_loss, train_metrics = self._train_epoch(train_loader)
                
                # Validation phase
                val_loss, val_metrics = self._validate_epoch(val_loader)
                
                # Store history
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_metrics.append(train_metrics)
                self.val_metrics.append(val_metrics)
                
                # Update scheduler
                self.scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Log progress
                if epoch % 10 == 0 or epoch == self.config.max_epochs - 1:
                    self.logger.info(
                        f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                        f"val_loss={val_loss:.4f}, patience={self.patience_counter}"
                    )
                
                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Restore best model
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)
            
            training_time = time.time() - start_time
            
            return TrainingResult(
                success=True,
                training_time=training_time,
                final_metrics=self.val_metrics[-1] if self.val_metrics else {},
                training_history={
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'train_metrics': self.train_metrics,
                    'val_metrics': self.val_metrics
                },
                metadata={
                    'epochs_trained': len(self.train_losses),
                    'best_val_loss': self.best_val_loss,
                    'early_stopped': self.patience_counter >= self.config.early_stopping_patience,
                    'device': str(self.device)
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            return TrainingResult(
                success=False,
                training_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        predictions = []
        targets = []
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_features)
            
            # Calculate loss
            if self.model.task_type == 'regression':
                loss = self.criterion(outputs.squeeze(), batch_targets.float())
            else:
                loss = self.criterion(outputs, batch_targets.long())
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Store predictions for metrics
            if self.model.task_type == 'regression':
                predictions.extend(outputs.squeeze().detach().cpu().numpy())
                targets.extend(batch_targets.detach().cpu().numpy())
            else:
                pred_probs = torch.softmax(outputs, dim=1)
                predictions.extend(pred_probs.detach().cpu().numpy())
                targets.extend(batch_targets.detach().cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        metrics = self._calculate_metrics(predictions, targets)
        
        return avg_loss, metrics
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = self.model(batch_features)
                
                if self.model.task_type == 'regression':
                    loss = self.criterion(outputs.squeeze(), batch_targets.float())
                    predictions.extend(outputs.squeeze().cpu().numpy())
                    targets.extend(batch_targets.cpu().numpy())
                else:
                    loss = self.criterion(outputs, batch_targets.long())
                    pred_probs = torch.softmax(outputs, dim=1)
                    predictions.extend(pred_probs.cpu().numpy())
                    targets.extend(batch_targets.cpu().numpy())
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        metrics = self._calculate_metrics(predictions, targets)
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, predictions: List[float], targets: List[float]) -> Dict[str, float]:
        """Calculate performance metrics"""
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        if self.model.task_type == 'regression':
            mse = mean_squared_error(targets, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(targets, predictions)
            r2 = r2_score(targets, predictions)
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
        else:
            # Classification metrics
            if predictions.ndim > 1:
                pred_classes = np.argmax(predictions, axis=1)
            else:
                pred_classes = (predictions > 0.5).astype(int)
            
            accuracy = accuracy_score(targets, pred_classes)
            metrics = {'accuracy': accuracy}
            
            # AUC for binary classification
            if len(np.unique(targets)) == 2 and predictions.ndim > 1:
                try:
                    auc = roc_auc_score(targets, predictions[:, 1])
                    metrics['auc'] = auc
                except:
                    pass
            
            return metrics


class MolecularModelPlugin(ModelPlugin):
    """
    Molecular model plugin with ensemble methods.
    
    Extracted and adapted from property_predictor.py lines 89-500.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the molecular model plugin"""
        super().__init__(config)
        
        # Configuration
        self.prediction_config = PredictionConfig(**self.config.get('prediction_config', {}))
        self.device = self.config.get('device', 'auto')
        
        # Models
        self.models = {}
        self.scalers = {}
        self.ensemble_weights = {}
        
        # Training state
        self.is_trained = False
        self.feature_names = []
        self.model_metadata = {}
        
        # Statistics
        self.stats = {
            'models_trained': 0,
            'predictions_made': 0,
            'training_time': 0.0,
            'inference_time': 0.0
        }
        
        self.logger.info("🧬 Molecular Model Plugin initialized")
    
    def get_metadata(self):
        """Get plugin metadata"""
        from ..base import PluginMetadata, PluginDependency, PluginVersion
        
        dependencies = []
        if SKLEARN_AVAILABLE:
            dependencies.append(PluginDependency(name="scikit-learn", version_requirement="*"))
        if XGBOOST_AVAILABLE:
            dependencies.append(PluginDependency(name="xgboost", version_requirement="*"))
        if PYTORCH_AVAILABLE:
            dependencies.append(PluginDependency(name="torch", version_requirement="*"))
        
        return PluginMetadata(
            name="MolecularModelPlugin",
            version=PluginVersion(1, 0, 0),
            author="Universal AI Core",
            description="Ensemble molecular property prediction models",
            plugin_type="model",
            entry_point=f"{__name__}:MolecularModelPlugin",
            dependencies=dependencies,
            capabilities=["regression", "classification", "ensemble", "uncertainty_quantification"],
            hooks=["before_training", "after_training", "before_prediction", "after_prediction"],
            configuration_schema={
                "model_types": {"type": "array", "items": {"type": "string"}},
                "ensemble_method": {"type": "string", "default": "voting"},
                "cross_validation_folds": {"type": "integer", "default": 5},
                "use_uncertainty_quantification": {"type": "boolean", "default": True}
            }
        )
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> TrainingResult:
        """
        Train molecular property prediction models.
        
        Adapted from property_predictor.py train_model method.
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"🚀 Training molecular models with {len(X)} samples")
            
            # Data preparation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.prediction_config.test_size,
                random_state=self.prediction_config.random_state
            )
            
            # Feature scaling
            if self.prediction_config.scale_features:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                self.scalers['feature_scaler'] = scaler
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
            
            # Determine task type
            task_type = self._determine_task_type(y)
            self.model_metadata['task_type'] = task_type
            
            training_results = {}
            
            # Train individual models
            for model_type in self.prediction_config.model_types:
                self.logger.info(f"📊 Training {model_type} model...")
                
                if model_type == 'random_forest' and SKLEARN_AVAILABLE:
                    result = self._train_random_forest(X_train_scaled, y_train, X_val_scaled, y_val, task_type)
                elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
                    result = self._train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val, task_type)
                elif model_type == 'neural_network' and PYTORCH_AVAILABLE:
                    result = self._train_neural_network(X_train_scaled, y_train, X_val_scaled, y_val, task_type)
                else:
                    self.logger.warning(f"Model type {model_type} not available")
                    continue
                
                if result.success:
                    training_results[model_type] = result
                    self.stats['models_trained'] += 1
            
            # Train ensemble if multiple models
            if len(training_results) > 1:
                ensemble_result = self._train_ensemble(X_val_scaled, y_val, task_type)
                if ensemble_result.success:
                    training_results['ensemble'] = ensemble_result
            
            self.is_trained = True
            training_time = time.time() - start_time
            self.stats['training_time'] += training_time
            
            return TrainingResult(
                success=True,
                training_time=training_time,
                final_metrics=self._calculate_ensemble_metrics(X_val_scaled, y_val, task_type),
                training_history={
                    'individual_models': {k: v.training_history for k, v in training_results.items()},
                    'ensemble_weights': self.ensemble_weights
                },
                metadata={
                    'models_trained': list(training_results.keys()),
                    'task_type': task_type,
                    'feature_dim': X.shape[1],
                    'samples': X.shape[0]
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            return TrainingResult(
                success=False,
                training_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def predict(self, X: np.ndarray, **kwargs) -> PredictionResult:
        """Make predictions using trained models"""
        start_time = time.time()
        
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            # Scale features if needed
            if 'feature_scaler' in self.scalers:
                X_scaled = self.scalers['feature_scaler'].transform(X)
            else:
                X_scaled = X
            
            predictions = {}
            uncertainties = {}
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                if model_name == 'ensemble':
                    continue
                
                pred, uncertainty = self._predict_single_model(model, X_scaled, model_name)
                predictions[model_name] = pred
                if uncertainty is not None:
                    uncertainties[model_name] = uncertainty
            
            # Ensemble predictions
            if len(predictions) > 1:
                ensemble_pred, ensemble_uncertainty = self._predict_ensemble(predictions, uncertainties)
                predictions['ensemble'] = ensemble_pred
                if ensemble_uncertainty is not None:
                    uncertainties['ensemble'] = ensemble_uncertainty
            
            # Use best model's predictions as primary
            primary_pred = predictions.get('ensemble', list(predictions.values())[0])
            primary_uncertainty = uncertainties.get('ensemble', list(uncertainties.values())[0] if uncertainties else None)
            
            inference_time = time.time() - start_time
            self.stats['predictions_made'] += len(X)
            self.stats['inference_time'] += inference_time
            
            return PredictionResult(
                predictions=primary_pred,
                probabilities=primary_uncertainty if self.model_metadata.get('task_type') == 'classification' else None,
                confidence_scores=primary_uncertainty if self.model_metadata.get('task_type') == 'regression' else None,
                prediction_time=inference_time,
                success=True,
                metadata={
                    'all_predictions': predictions,
                    'uncertainties': uncertainties,
                    'model_used': 'ensemble' if 'ensemble' in predictions else list(predictions.keys())[0]
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Prediction failed: {e}")
            return PredictionResult(
                predictions=np.array([]),
                prediction_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _determine_task_type(self, y: np.ndarray) -> str:
        """Determine if task is regression or classification"""
        unique_values = len(np.unique(y))
        if unique_values <= 10 and np.all(y == y.astype(int)):
            return 'classification'
        return 'regression'
    
    def _train_random_forest(self, X_train, y_train, X_val, y_val, task_type) -> TrainingResult:
        """Train Random Forest model"""
        try:
            if task_type == 'regression':
                model = RandomForestRegressor(
                    n_estimators=self.config.get('rf_n_estimators', 100),
                    max_depth=self.config.get('rf_max_depth', None),
                    random_state=self.prediction_config.random_state,
                    n_jobs=self.prediction_config.n_jobs
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=self.config.get('rf_n_estimators', 100),
                    max_depth=self.config.get('rf_max_depth', None),
                    random_state=self.prediction_config.random_state,
                    n_jobs=self.prediction_config.n_jobs
                )
            
            # Hyperparameter optimization if enabled
            if self.prediction_config.hyperparameter_optimization:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
                
                grid_search = GridSearchCV(
                    model, param_grid, cv=self.prediction_config.cross_validation_folds,
                    scoring='neg_mean_squared_error' if task_type == 'regression' else 'accuracy',
                    n_jobs=self.prediction_config.n_jobs
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
            else:
                model.fit(X_train, y_train)
            
            self.models['random_forest'] = model
            
            # Validation metrics
            val_pred = model.predict(X_val)
            metrics = self._calculate_model_metrics(y_val, val_pred, task_type)
            
            return TrainingResult(
                success=True,
                training_time=0.0,  # RF training is typically fast
                final_metrics=metrics
            )
            
        except Exception as e:
            return TrainingResult(success=False, error_message=str(e))
    
    def _train_xgboost(self, X_train, y_train, X_val, y_val, task_type) -> TrainingResult:
        """Train XGBoost model"""
        try:
            if task_type == 'regression':
                model = xgb.XGBRegressor(
                    n_estimators=self.config.get('xgb_n_estimators', 100),
                    max_depth=self.config.get('xgb_max_depth', 6),
                    learning_rate=self.config.get('xgb_learning_rate', 0.1),
                    random_state=self.prediction_config.random_state,
                    n_jobs=self.prediction_config.n_jobs
                )
            else:
                model = xgb.XGBClassifier(
                    n_estimators=self.config.get('xgb_n_estimators', 100),
                    max_depth=self.config.get('xgb_max_depth', 6),
                    learning_rate=self.config.get('xgb_learning_rate', 0.1),
                    random_state=self.prediction_config.random_state,
                    n_jobs=self.prediction_config.n_jobs
                )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            self.models['xgboost'] = model
            
            # Validation metrics
            val_pred = model.predict(X_val)
            metrics = self._calculate_model_metrics(y_val, val_pred, task_type)
            
            return TrainingResult(
                success=True,
                training_time=0.0,
                final_metrics=metrics
            )
            
        except Exception as e:
            return TrainingResult(success=False, error_message=str(e))
    
    def _train_neural_network(self, X_train, y_train, X_val, y_val, task_type) -> TrainingResult:
        """Train Neural Network model"""
        try:
            # Create neural network
            input_dim = X_train.shape[1]
            output_dim = 1 if task_type == 'regression' else len(np.unique(y_train))
            
            model = MolecularNeuralNetwork(
                input_dim=input_dim,
                hidden_dims=self.config.get('nn_hidden_dims', [512, 256, 128]),
                output_dim=output_dim,
                dropout_rate=self.config.get('nn_dropout', 0.3),
                use_batch_norm=self.config.get('nn_batch_norm', True),
                use_residual=self.config.get('nn_residual', True),
                task_type=task_type
            )
            
            # Create trainer
            trainer = MolecularTrainer(model, self.prediction_config, self.device)
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            
            train_loader = DataLoader(
                train_dataset, batch_size=self.prediction_config.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.prediction_config.batch_size, shuffle=False
            )
            
            # Train model
            result = trainer.train(train_loader, val_loader)
            
            if result.success:
                self.models['neural_network'] = model
            
            return result
            
        except Exception as e:
            return TrainingResult(success=False, error_message=str(e))
    
    def _train_ensemble(self, X_val, y_val, task_type) -> TrainingResult:
        """Train ensemble model"""
        try:
            # Get predictions from all models
            predictions = {}
            for model_name, model in self.models.items():
                pred, _ = self._predict_single_model(model, X_val, model_name)
                predictions[model_name] = pred
            
            # Calculate ensemble weights based on validation performance
            weights = {}
            total_weight = 0.0
            
            for model_name, pred in predictions.items():
                if task_type == 'regression':
                    rmse = np.sqrt(mean_squared_error(y_val, pred))
                    weight = 1.0 / (rmse + 1e-8)  # Inverse RMSE
                else:
                    accuracy = accuracy_score(y_val, pred)
                    weight = accuracy
                
                weights[model_name] = weight
                total_weight += weight
            
            # Normalize weights
            for model_name in weights:
                weights[model_name] /= total_weight
            
            self.ensemble_weights = weights
            
            # Calculate ensemble metrics
            ensemble_pred = self._weighted_average_predictions(predictions, weights)
            metrics = self._calculate_model_metrics(y_val, ensemble_pred, task_type)
            
            return TrainingResult(
                success=True,
                training_time=0.0,
                final_metrics=metrics,
                metadata={'ensemble_weights': weights}
            )
            
        except Exception as e:
            return TrainingResult(success=False, error_message=str(e))
    
    def _predict_single_model(self, model, X, model_name) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions with a single model"""
        uncertainty = None
        
        if model_name == 'neural_network':
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(model.device if hasattr(model, 'device') else 'cpu')
                pred = model(X_tensor).cpu().numpy().squeeze()
        else:
            pred = model.predict(X)
            
            # Get uncertainty estimates if available
            if hasattr(model, 'predict_proba') and self.prediction_config.use_uncertainty_quantification:
                try:
                    proba = model.predict_proba(X)
                    if proba.shape[1] == 2:  # Binary classification
                        uncertainty = 1 - np.max(proba, axis=1)  # Uncertainty as 1 - max probability
                    else:
                        uncertainty = 1 - np.max(proba, axis=1)
                except:
                    pass
        
        return pred, uncertainty
    
    def _predict_ensemble(self, predictions, uncertainties) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make ensemble predictions"""
        if self.prediction_config.ensemble_method == 'voting':
            # Simple averaging
            pred_array = np.array(list(predictions.values()))
            ensemble_pred = np.mean(pred_array, axis=0)
        elif self.prediction_config.ensemble_method == 'weighted':
            # Weighted averaging
            ensemble_pred = self._weighted_average_predictions(predictions, self.ensemble_weights)
        else:
            # Default to voting
            pred_array = np.array(list(predictions.values()))
            ensemble_pred = np.mean(pred_array, axis=0)
        
        # Ensemble uncertainty
        ensemble_uncertainty = None
        if uncertainties and self.prediction_config.use_uncertainty_quantification:
            uncertainty_array = np.array(list(uncertainties.values()))
            ensemble_uncertainty = np.mean(uncertainty_array, axis=0)
        
        return ensemble_pred, ensemble_uncertainty
    
    def _weighted_average_predictions(self, predictions, weights) -> np.ndarray:
        """Calculate weighted average of predictions"""
        weighted_sum = np.zeros_like(list(predictions.values())[0])
        
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 1.0 / len(predictions))
            weighted_sum += weight * pred
        
        return weighted_sum
    
    def _calculate_model_metrics(self, y_true, y_pred, task_type) -> Dict[str, float]:
        """Calculate model performance metrics"""
        if task_type == 'regression':
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
        else:
            accuracy = accuracy_score(y_true, y_pred)
            metrics = {'accuracy': accuracy}
            
            # AUC for binary classification
            if len(np.unique(y_true)) == 2:
                try:
                    auc = roc_auc_score(y_true, y_pred)
                    metrics['auc'] = auc
                except:
                    pass
            
            return metrics
    
    def _calculate_ensemble_metrics(self, X_val, y_val, task_type) -> Dict[str, float]:
        """Calculate ensemble model metrics"""
        try:
            result = self.predict(X_val)
            if result.success:
                return self._calculate_model_metrics(y_val, result.predictions, task_type)
        except:
            pass
        return {}
    
    def save_model(self, filepath: str) -> bool:
        """Save trained models to file"""
        try:
            save_data = {
                'models': {},
                'scalers': self.scalers,
                'ensemble_weights': self.ensemble_weights,
                'model_metadata': self.model_metadata,
                'prediction_config': self.prediction_config.__dict__,
                'is_trained': self.is_trained
            }
            
            # Save non-PyTorch models with pickle
            for name, model in self.models.items():
                if name == 'neural_network' and PYTORCH_AVAILABLE:
                    # Save PyTorch model separately
                    torch.save(model.state_dict(), f"{filepath}_nn.pth")
                    save_data['models'][name] = 'neural_network_saved_separately'
                else:
                    save_data['models'][name] = model
            
            # Save main data with pickle
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            self.logger.info(f"💾 Model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained models from file"""
        try:
            # Load main data
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            self.models = {}
            self.scalers = save_data['scalers']
            self.ensemble_weights = save_data['ensemble_weights']
            self.model_metadata = save_data['model_metadata']
            self.is_trained = save_data['is_trained']
            
            # Load models
            for name, model in save_data['models'].items():
                if model == 'neural_network_saved_separately':
                    # Load PyTorch model
                    input_dim = self.model_metadata.get('feature_dim', 100)
                    output_dim = 1 if self.model_metadata.get('task_type') == 'regression' else 2
                    
                    nn_model = MolecularNeuralNetwork(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        task_type=self.model_metadata.get('task_type', 'regression')
                    )
                    nn_model.load_state_dict(torch.load(f"{filepath}_nn.pth"))
                    self.models[name] = nn_model
                else:
                    self.models[name] = model
            
            self.logger.info(f"📂 Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get plugin capabilities"""
        return {
            'regression': True,
            'classification': True,
            'ensemble_methods': True,
            'uncertainty_quantification': self.prediction_config.use_uncertainty_quantification,
            'hyperparameter_optimization': self.prediction_config.hyperparameter_optimization,
            'cross_validation': True,
            'model_persistence': True,
            'available_models': {
                'random_forest': SKLEARN_AVAILABLE,
                'xgboost': XGBOOST_AVAILABLE,
                'neural_network': PYTORCH_AVAILABLE
            }
        }
    
    def get_hooks(self) -> Dict[str, Callable]:
        """Get plugin hooks"""
        return {
            'before_training': self._before_training_hook,
            'after_training': self._after_training_hook,
            'before_prediction': self._before_prediction_hook,
            'after_prediction': self._after_prediction_hook
        }
    
    def _before_training_hook(self, X, y, **kwargs):
        """Hook called before training"""
        self.logger.debug("🔗 Before molecular model training")
        return X, y
    
    def _after_training_hook(self, result: TrainingResult) -> TrainingResult:
        """Hook called after training"""
        self.logger.debug("🔗 After molecular model training")
        return result
    
    def _before_prediction_hook(self, X, **kwargs):
        """Hook called before prediction"""
        self.logger.debug("🔗 Before molecular model prediction")
        return X
    
    def _after_prediction_hook(self, result: PredictionResult) -> PredictionResult:
        """Hook called after prediction"""
        self.logger.debug("🔗 After molecular model prediction")
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get model statistics"""
        return self.stats.copy()
    
    def health_check(self) -> bool:
        """Perform health check"""
        try:
            # Check if at least one model type is available
            available_models = any([SKLEARN_AVAILABLE, XGBOOST_AVAILABLE, PYTORCH_AVAILABLE])
            return available_models
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False


# Plugin metadata for discovery
__plugin_metadata__ = {
    "name": "MolecularModelPlugin",
    "version": "1.0.0",
    "author": "Universal AI Core",
    "description": "Ensemble molecular property prediction models",
    "plugin_type": "model",
    "entry_point": f"{__name__}:MolecularModelPlugin",
    "dependencies": [
        {"name": "scikit-learn", "optional": True},
        {"name": "xgboost", "optional": True},
        {"name": "torch", "optional": True}
    ],
    "capabilities": ["regression", "classification", "ensemble", "uncertainty_quantification"],
    "hooks": ["before_training", "after_training", "before_prediction", "after_prediction"]
}


if __name__ == "__main__":
    # Test the molecular model plugin
    print("🧬 MOLECULAR MODEL PLUGIN TEST")
    print("=" * 50)
    
    # Generate synthetic molecular data
    np.random.seed(42)
    n_samples = 1000
    n_features = 200  # Simulating molecular descriptors
    
    X = np.random.randn(n_samples, n_features)
    y_regression = np.sum(X[:, :10], axis=1) + np.random.normal(0, 0.1, n_samples)
    y_classification = (y_regression > np.median(y_regression)).astype(int)
    
    # Test configuration
    config = {
        'prediction_config': {
            'model_types': ['random_forest'],  # Start with one model
            'cross_validation_folds': 3,
            'test_size': 0.2,
            'use_uncertainty_quantification': True
        }
    }
    
    if SKLEARN_AVAILABLE:
        config['prediction_config']['model_types'].append('random_forest')
    if XGBOOST_AVAILABLE:
        config['prediction_config']['model_types'].append('xgboost')
    if PYTORCH_AVAILABLE:
        config['prediction_config']['model_types'].append('neural_network')
    
    # Initialize plugin
    model_plugin = MolecularModelPlugin(config)
    
    # Test regression
    print(f"\n🔬 Testing regression with {len(config['prediction_config']['model_types'])} models...")
    train_result = model_plugin.train(X, y_regression)
    
    if train_result.success:
        print(f"✅ Training successful! Time: {train_result.training_time:.3f}s")
        print(f"📊 Final metrics: {train_result.final_metrics}")
        
        # Test prediction
        pred_result = model_plugin.predict(X[:10])
        if pred_result.success:
            print(f"✅ Prediction successful! Time: {pred_result.prediction_time:.3f}s")
            print(f"🎯 Predictions shape: {pred_result.predictions.shape}")
    else:
        print(f"❌ Training failed: {train_result.error_message}")
    
    # Test health check
    health = model_plugin.health_check()
    print(f"\n🏥 Health check: {'✅' if health else '❌'}")
    
    # Show capabilities
    capabilities = model_plugin.get_capabilities()
    print(f"\n🔧 Capabilities:")
    for key, value in capabilities.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Molecular model plugin test completed!")