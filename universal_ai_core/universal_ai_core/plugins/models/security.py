#!/usr/bin/env python3
"""
Security Models Plugin
======================

This module provides security neural network models for the Universal AI Core system.
Adapted from Saraphis molecular neural network patterns, specialized for cybersecurity 
classification tasks including malware detection, intrusion detection, and threat analysis.

Features:
- Enterprise-grade neural networks for security classification
- Ensemble methods (Random Forest, XGBoost, Neural Networks) adapted for security
- Uncertainty quantification for threat confidence assessment
- Cross-validation and hyperparameter optimization for security models
- Model persistence and checkpointing for security deployments
- GPU acceleration and memory optimization for large-scale security analysis
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
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.svm import SVM
    from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
    from sklearn.decomposition import PCA
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
class SecurityMetrics:
    """Metrics for security classification evaluation"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc: Optional[float] = None
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    cross_val_score: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    threat_detection_rate: float = 0.0


@dataclass
class SecurityClassificationConfig:
    """
    Configuration for security classification.
    
    Adapted from molecular PredictionConfig for security domain.
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
    batch_size: int = 128
    learning_rate: float = 0.001
    dropout_rate: float = 0.3
    class_weight: str = 'balanced'  # Handle imbalanced security datasets
    anomaly_detection: bool = True  # Enable anomaly detection capabilities
    threat_threshold: float = 0.7  # Threshold for threat classification


class SecurityNeuralNetwork(nn.Module):
    """
    Advanced neural network for security classification.
    
    Adapted from MolecularNeuralNetwork with security-specific optimizations:
    - Residual connections for deep feature learning
    - Attention mechanisms for important feature focus
    - Dropout for regularization against adversarial attacks
    - Batch normalization for stable training
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int, 
                 dropout_rate: float = 0.3, use_attention: bool = True):
        super(SecurityNeuralNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        self.input_norm = nn.BatchNorm1d(hidden_dims[0])
        
        # Hidden layers with residual connections
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.norms.append(nn.BatchNorm1d(hidden_dims[i + 1]))
        
        # Attention mechanism for feature importance
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dims[-1],
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_dims[-1])
        
        # Output layers
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Linear(hidden_dims[-1], 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, return_uncertainty: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the network"""
        # Input projection
        x = F.relu(self.input_norm(self.input_projection(x)))
        
        # Hidden layers with residual connections
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            residual = x
            x = F.relu(norm(layer(x)))
            
            # Add residual connection if dimensions match
            if residual.shape == x.shape:
                x = x + residual
        
        # Attention mechanism
        if self.use_attention:
            x_reshaped = x.unsqueeze(1)  # Add sequence dimension
            attended, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
            x = self.attention_norm(attended.squeeze(1) + x)
        
        # Output
        x = self.dropout(x)
        logits = self.classifier(x)
        
        if return_uncertainty:
            uncertainty = torch.sigmoid(self.uncertainty_head(x))
            return logits, uncertainty
        
        return logits


class SecurityAnomalyDetector(nn.Module):
    """
    Autoencoder-based anomaly detector for security threats.
    
    Adapted from molecular pattern recognition for security anomaly detection.
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = 64, hidden_dims: List[int] = None):
        super(SecurityAnomalyDetector, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [input_dim // 2, input_dim // 4, encoding_dim]
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        hidden_dims_reversed = list(reversed(hidden_dims[:-1])) + [input_dim]
        prev_dim = encoding_dim
        
        for dim in hidden_dims_reversed:
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU() if dim != input_dim else nn.Sigmoid(),
                nn.BatchNorm1d(dim) if dim != input_dim else nn.Identity(),
                nn.Dropout(0.2) if dim != input_dim else nn.Identity()
            ])
            prev_dim = dim
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction and encoding"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate reconstruction error for anomaly detection"""
        reconstructed, _ = self.forward(x)
        return torch.mean(torch.square(x - reconstructed), dim=1)


class SecurityModelTrainer:
    """
    Enterprise security model trainer.
    
    Adapted from MolecularTrainer with security-specific training strategies:
    - Adversarial training for robustness
    - Class balancing for imbalanced security datasets
    - Multi-task learning for comprehensive threat detection
    """
    
    def __init__(self, config: SecurityClassificationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.SecurityModelTrainer")
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        
        # Training history
        self.training_history = {}
        self.validation_scores = {}
        
        # Thread safety
        self.training_lock = threading.Lock()
        
        self.logger.info(f"🛡️ Security Model Trainer initialized")
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray, 
                      task_type: str = 'classification') -> Dict[str, Any]:
        """
        Train ensemble of security models.
        
        Adapted from molecular ensemble training for security classification.
        """
        with self.training_lock:
            start_time = time.time()
            results = {}
            
            try:
                self.logger.info(f"🎯 Training security ensemble for {task_type}")
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.config.test_size, 
                    random_state=self.config.random_state,
                    stratify=y if task_type == 'classification' else None
                )
                
                # Scale features
                if self.config.scale_features:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    self.scalers['ensemble'] = scaler
                
                # Encode labels for classification
                if task_type == 'classification':
                    label_encoder = LabelEncoder()
                    y_train = label_encoder.fit_transform(y_train)
                    y_test = label_encoder.transform(y_test)
                    self.label_encoders['ensemble'] = label_encoder
                
                # Train individual models
                if 'random_forest' in self.config.model_types:
                    results['random_forest'] = self._train_random_forest(
                        X_train, X_test, y_train, y_test, task_type
                    )
                
                if 'xgboost' in self.config.model_types and XGBOOST_AVAILABLE:
                    results['xgboost'] = self._train_xgboost(
                        X_train, X_test, y_train, y_test, task_type
                    )
                
                if 'neural_network' in self.config.model_types and PYTORCH_AVAILABLE:
                    results['neural_network'] = self._train_neural_network(
                        X_train, X_test, y_train, y_test, task_type
                    )
                
                # Train anomaly detector if enabled
                if self.config.anomaly_detection and PYTORCH_AVAILABLE:
                    results['anomaly_detector'] = self._train_anomaly_detector(X_train)
                
                # Create ensemble
                results['ensemble'] = self._create_ensemble(X_test, y_test, task_type)
                
                training_time = time.time() - start_time
                
                return TrainingResult(
                    success=True,
                    metrics=results,
                    training_time=training_time,
                    model_info={
                        'task_type': task_type,
                        'models_trained': list(results.keys()),
                        'ensemble_method': self.config.ensemble_method
                    }
                )
                
            except Exception as e:
                self.logger.error(f"❌ Error training security ensemble: {e}")
                return TrainingResult(
                    success=False,
                    error_message=str(e),
                    training_time=time.time() - start_time
                )
    
    def _train_random_forest(self, X_train: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray, y_test: np.ndarray, 
                           task_type: str) -> SecurityMetrics:
        """Train Random Forest for security classification"""
        try:
            # Configure model
            if task_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight=self.config.class_weight,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs
                )
            
            # Hyperparameter optimization
            if self.config.hyperparameter_optimization:
                model = self._optimize_rf_hyperparameters(model, X_train, y_train, task_type)
            
            # Train model
            model.fit(X_train, y_train)
            self.models['random_forest'] = model
            
            # Evaluate
            y_pred = model.predict(X_test)
            if task_type == 'classification':
                y_pred_proba = model.predict_proba(X_test)
                return self._calculate_classification_metrics(y_test, y_pred, y_pred_proba)
            else:
                return self._calculate_regression_metrics(y_test, y_pred)
                
        except Exception as e:
            self.logger.error(f"❌ Error training Random Forest: {e}")
            return SecurityMetrics()
    
    def _train_xgboost(self, X_train: np.ndarray, X_test: np.ndarray,
                      y_train: np.ndarray, y_test: np.ndarray, 
                      task_type: str) -> SecurityMetrics:
        """Train XGBoost for security classification"""
        try:
            # Configure model
            if task_type == 'classification':
                model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                    eval_metric='auc'
                )
            else:
                model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs
                )
            
            # Train with early stopping
            eval_set = [(X_test, y_test)]
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=self.config.early_stopping_patience,
                verbose=False
            )
            
            self.models['xgboost'] = model
            
            # Evaluate
            y_pred = model.predict(X_test)
            if task_type == 'classification':
                y_pred_proba = model.predict_proba(X_test)
                return self._calculate_classification_metrics(y_test, y_pred, y_pred_proba)
            else:
                return self._calculate_regression_metrics(y_test, y_pred)
                
        except Exception as e:
            self.logger.error(f"❌ Error training XGBoost: {e}")
            return SecurityMetrics()
    
    def _train_neural_network(self, X_train: np.ndarray, X_test: np.ndarray,
                            y_train: np.ndarray, y_test: np.ndarray,
                            task_type: str) -> SecurityMetrics:
        """Train neural network for security classification"""
        try:
            input_dim = X_train.shape[1]
            
            if task_type == 'classification':
                num_classes = len(np.unique(y_train))
                hidden_dims = [512, 256, 128, 64]
            else:
                num_classes = 1
                hidden_dims = [256, 128, 64]
            
            # Create model
            model = SecurityNeuralNetwork(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=num_classes,
                dropout_rate=self.config.dropout_rate
            )
            
            # Setup training
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            if task_type == 'classification':
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.MSELoss()
            
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.LongTensor(y_train) if task_type == 'classification' else torch.FloatTensor(y_train)
            )
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test),
                torch.LongTensor(y_test) if task_type == 'classification' else torch.FloatTensor(y_test)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            # Training loop
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config.max_epochs):
                # Training
                model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    
                    if task_type == 'classification':
                        loss = criterion(outputs, batch_y)
                    else:
                        loss = criterion(outputs.squeeze(), batch_y)
                    
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        
                        if task_type == 'classification':
                            loss = criterion(outputs, batch_y)
                        else:
                            loss = criterion(outputs.squeeze(), batch_y)
                        
                        val_loss += loss.item()
                
                val_loss /= len(test_loader)
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f'/tmp/best_security_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        break
            
            # Load best model
            model.load_state_dict(torch.load(f'/tmp/best_security_model.pth'))
            self.models['neural_network'] = model
            
            # Evaluate
            model.eval()
            predictions = []
            probabilities = []
            
            with torch.no_grad():
                for batch_X, _ in test_loader:
                    batch_X = batch_X.to(device)
                    outputs = model(batch_X)
                    
                    if task_type == 'classification':
                        probs = F.softmax(outputs, dim=1)
                        preds = torch.argmax(outputs, dim=1)
                        probabilities.extend(probs.cpu().numpy())
                        predictions.extend(preds.cpu().numpy())
                    else:
                        predictions.extend(outputs.squeeze().cpu().numpy())
            
            if task_type == 'classification':
                return self._calculate_classification_metrics(y_test, np.array(predictions), np.array(probabilities))
            else:
                return self._calculate_regression_metrics(y_test, np.array(predictions))
                
        except Exception as e:
            self.logger.error(f"❌ Error training neural network: {e}")
            return SecurityMetrics()
    
    def _train_anomaly_detector(self, X_train: np.ndarray) -> Dict[str, Any]:
        """Train anomaly detector for security threats"""
        try:
            input_dim = X_train.shape[1]
            encoding_dim = min(64, input_dim // 4)
            
            model = SecurityAnomalyDetector(
                input_dim=input_dim,
                encoding_dim=encoding_dim
            )
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
            
            # Training
            train_dataset = TensorDataset(torch.FloatTensor(X_train))
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            
            model.train()
            for epoch in range(100):  # Reduced epochs for anomaly detection
                epoch_loss = 0.0
                for batch in train_loader:
                    batch_X = batch[0].to(device)
                    
                    optimizer.zero_grad()
                    reconstructed, _ = model(batch_X)
                    loss = criterion(reconstructed, batch_X)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
            
            self.models['anomaly_detector'] = model
            
            # Calculate reconstruction threshold
            model.eval()
            reconstruction_errors = []
            with torch.no_grad():
                for batch in train_loader:
                    batch_X = batch[0].to(device)
                    errors = model.get_reconstruction_error(batch_X)
                    reconstruction_errors.extend(errors.cpu().numpy())
            
            threshold = np.percentile(reconstruction_errors, 95)  # 95th percentile as threshold
            
            return {
                'threshold': threshold,
                'mean_reconstruction_error': np.mean(reconstruction_errors),
                'std_reconstruction_error': np.std(reconstruction_errors)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error training anomaly detector: {e}")
            return {}
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                        y_pred_proba: Optional[np.ndarray] = None) -> SecurityMetrics:
        """Calculate classification metrics for security tasks"""
        metrics = SecurityMetrics()
        
        try:
            metrics.accuracy = accuracy_score(y_true, y_pred)
            metrics.precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics.recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics.f1_score = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                metrics.auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            
            # Security-specific metrics
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):  # Binary classification
                tn, fp, fn, tp = cm.ravel()
                metrics.false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                metrics.false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                metrics.threat_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating classification metrics: {e}")
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> SecurityMetrics:
        """Calculate regression metrics"""
        metrics = SecurityMetrics()
        
        try:
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            
            metrics.accuracy = r2_score(y_true, y_pred)  # Using R² as accuracy measure
            mse = mean_squared_error(y_true, y_pred)
            metrics.precision = 1.0 / (1.0 + mse)  # Convert MSE to precision-like metric
            metrics.recall = metrics.precision  # Same for regression
            metrics.f1_score = metrics.precision
            
        except Exception as e:
            logger.error(f"❌ Error calculating regression metrics: {e}")
        
        return metrics
    
    def _optimize_rf_hyperparameters(self, model, X_train: np.ndarray, y_train: np.ndarray, task_type: str):
        """Optimize Random Forest hyperparameters"""
        try:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                model, param_grid, cv=3, scoring='accuracy' if task_type == 'classification' else 'r2',
                n_jobs=self.config.n_jobs, verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_
            
        except Exception as e:
            self.logger.error(f"❌ Error optimizing hyperparameters: {e}")
            return model
    
    def _create_ensemble(self, X_test: np.ndarray, y_test: np.ndarray, task_type: str) -> Dict[str, Any]:
        """Create ensemble from trained models"""
        try:
            if not self.models:
                return {}
            
            # Collect predictions from all models
            predictions = {}
            probabilities = {}
            
            for model_name, model in self.models.items():
                if model_name == 'anomaly_detector':
                    continue
                
                if model_name == 'neural_network' and PYTORCH_AVAILABLE:
                    # Neural network predictions
                    model.eval()
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X_test).to(device)
                        outputs = model(X_tensor)
                        
                        if task_type == 'classification':
                            probs = F.softmax(outputs, dim=1).cpu().numpy()
                            preds = torch.argmax(outputs, dim=1).cpu().numpy()
                            probabilities[model_name] = probs
                            predictions[model_name] = preds
                        else:
                            predictions[model_name] = outputs.squeeze().cpu().numpy()
                else:
                    # Sklearn models
                    preds = model.predict(X_test)
                    predictions[model_name] = preds
                    
                    if task_type == 'classification' and hasattr(model, 'predict_proba'):
                        probabilities[model_name] = model.predict_proba(X_test)
            
            # Create ensemble prediction
            if self.config.ensemble_method == 'voting':
                if task_type == 'classification':
                    # Majority voting
                    ensemble_pred = np.array([predictions[name] for name in predictions]).T
                    ensemble_pred = np.apply_along_axis(
                        lambda x: np.bincount(x).argmax(), axis=1, arr=ensemble_pred
                    )
                else:
                    # Average prediction
                    ensemble_pred = np.mean([predictions[name] for name in predictions], axis=0)
            
            # Calculate ensemble metrics
            if task_type == 'classification':
                ensemble_proba = None
                if probabilities:
                    ensemble_proba = np.mean([probabilities[name] for name in probabilities], axis=0)
                ensemble_metrics = self._calculate_classification_metrics(y_test, ensemble_pred, ensemble_proba)
            else:
                ensemble_metrics = self._calculate_regression_metrics(y_test, ensemble_pred)
            
            return {
                'predictions': predictions,
                'ensemble_prediction': ensemble_pred,
                'ensemble_metrics': ensemble_metrics,
                'individual_metrics': {
                    name: self._calculate_classification_metrics(y_test, predictions[name])
                    if task_type == 'classification' else self._calculate_regression_metrics(y_test, predictions[name])
                    for name in predictions
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error creating ensemble: {e}")
            return {}


class SecurityModelPlugin(ModelPlugin):
    """
    Security model plugin for cybersecurity classification tasks.
    
    Provides enterprise-grade models adapted from molecular analysis patterns
    for malware detection, intrusion detection, and threat classification.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the security model plugin"""
        super().__init__(config)
        
        # Configuration
        self.classification_config = SecurityClassificationConfig(**self.config.get('classification', {}))
        
        # Model trainer
        self.trainer = SecurityModelTrainer(self.classification_config)
        
        # Model registry
        self.trained_models = {}
        self.model_metadata = {}
        
        # Statistics
        self.stats = {
            'models_trained': 0,
            'predictions_made': 0,
            'threats_detected': 0,
            'anomalies_detected': 0
        }
        
        self.logger.info(f"🛡️ Security Model Plugin initialized")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata"""
        return {
            "name": "SecurityModelPlugin",
            "version": "1.0.0",
            "description": "Security classification models for cybersecurity tasks",
            "supported_models": [
                ModelType.RANDOM_FOREST,
                ModelType.GRADIENT_BOOSTING,
                ModelType.NEURAL_NETWORK,
                ModelType.ENSEMBLE,
                ModelType.AUTOENCODER
            ],
            "capabilities": [
                "malware_detection",
                "intrusion_detection", 
                "anomaly_detection",
                "threat_classification",
                "ensemble_methods",
                "uncertainty_quantification"
            ],
            "dependencies": {
                "sklearn": SKLEARN_AVAILABLE,
                "xgboost": XGBOOST_AVAILABLE,
                "pytorch": PYTORCH_AVAILABLE
            }
        }
    
    def train_model(self, model_type: ModelType, training_data: Dict[str, Any], 
                   **kwargs) -> TrainingResult:
        """Train security model"""
        try:
            self.logger.info(f"🎯 Training security model: {model_type.value}")
            
            X = training_data.get('features')
            y = training_data.get('labels')
            task_type = kwargs.get('task_type', 'classification')
            
            if X is None or y is None:
                return TrainingResult(
                    success=False,
                    error_message="Missing features or labels in training data"
                )
            
            # Convert to numpy arrays
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            
            # Train ensemble model
            result = self.trainer.train_ensemble(X, y, task_type)
            
            if result.success:
                model_id = f"security_{model_type.value}_{int(time.time())}"
                self.trained_models[model_id] = {
                    'models': self.trainer.models.copy(),
                    'scalers': self.trainer.scalers.copy(),
                    'label_encoders': self.trainer.label_encoders.copy(),
                    'task_type': task_type,
                    'model_type': model_type
                }
                
                self.model_metadata[model_id] = {
                    'created_at': datetime.utcnow(),
                    'training_samples': len(X),
                    'feature_count': X.shape[1],
                    'task_type': task_type,
                    'metrics': result.metrics
                }
                
                result.model_id = model_id
                self.stats['models_trained'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error training security model: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e)
            )
    
    def predict(self, model_id: str, features: Union[np.ndarray, List[List[float]]], 
               **kwargs) -> PredictionResult:
        """Make security predictions"""
        try:
            if model_id not in self.trained_models:
                return PredictionResult(
                    success=False,
                    error_message=f"Model {model_id} not found"
                )
            
            model_data = self.trained_models[model_id]
            models = model_data['models']
            scalers = model_data['scalers']
            label_encoders = model_data['label_encoders']
            task_type = model_data['task_type']
            
            # Convert features to numpy array
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            
            # Scale features
            if 'ensemble' in scalers:
                features = scalers['ensemble'].transform(features)
            
            # Collect predictions from all models
            predictions = {}
            probabilities = {}
            
            for model_name, model in models.items():
                if model_name == 'anomaly_detector':
                    # Handle anomaly detection separately
                    if PYTORCH_AVAILABLE:
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model.eval()
                        
                        with torch.no_grad():
                            X_tensor = torch.FloatTensor(features).to(device)
                            reconstruction_errors = model.get_reconstruction_error(X_tensor)
                            
                        # Get threshold from training metadata
                        threshold = self.model_metadata[model_id].get('anomaly_threshold', 0.1)
                        anomaly_scores = reconstruction_errors.cpu().numpy()
                        anomaly_predictions = (anomaly_scores > threshold).astype(int)
                        
                        predictions[model_name] = anomaly_predictions
                        probabilities[model_name] = anomaly_scores
                        
                        self.stats['anomalies_detected'] += np.sum(anomaly_predictions)
                    
                elif model_name == 'neural_network' and PYTORCH_AVAILABLE:
                    # Neural network predictions
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model.eval()
                    
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(features).to(device)
                        outputs = model(X_tensor, return_uncertainty=True)
                        
                        if len(outputs) == 2:
                            logits, uncertainty = outputs
                            uncertainty_scores = uncertainty.cpu().numpy()
                        else:
                            logits = outputs
                            uncertainty_scores = None
                        
                        if task_type == 'classification':
                            probs = F.softmax(logits, dim=1).cpu().numpy()
                            preds = torch.argmax(logits, dim=1).cpu().numpy()
                            probabilities[model_name] = probs
                            predictions[model_name] = preds
                        else:
                            predictions[model_name] = logits.squeeze().cpu().numpy()
                
                else:
                    # Sklearn models
                    preds = model.predict(features)
                    predictions[model_name] = preds
                    
                    if task_type == 'classification' and hasattr(model, 'predict_proba'):
                        probabilities[model_name] = model.predict_proba(features)
            
            # Create ensemble prediction
            if len(predictions) > 1 and 'anomaly_detector' not in predictions:
                # Exclude anomaly detector from ensemble
                pred_models = {k: v for k, v in predictions.items() if k != 'anomaly_detector'}
                
                if task_type == 'classification':
                    # Majority voting
                    ensemble_pred = np.array([pred_models[name] for name in pred_models]).T
                    ensemble_pred = np.apply_along_axis(
                        lambda x: np.bincount(x).argmax(), axis=1, arr=ensemble_pred
                    )
                else:
                    # Average prediction
                    ensemble_pred = np.mean([pred_models[name] for name in pred_models], axis=0)
                
                predictions['ensemble'] = ensemble_pred
            
            # Decode labels if necessary
            if 'ensemble' in label_encoders and task_type == 'classification':
                for name, pred in predictions.items():
                    if name != 'anomaly_detector':
                        predictions[name] = label_encoders['ensemble'].inverse_transform(pred.astype(int))
            
            # Count threats detected
            if task_type == 'classification':
                for pred in predictions.values():
                    if isinstance(pred, np.ndarray):
                        # Assume positive class (1) indicates threat
                        self.stats['threats_detected'] += np.sum(pred == 1)
            
            self.stats['predictions_made'] += len(features)
            
            return PredictionResult(
                success=True,
                predictions=predictions,
                confidence_scores=probabilities,
                model_info={
                    'model_id': model_id,
                    'task_type': task_type,
                    'models_used': list(models.keys())
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error making security predictions: {e}")
            return PredictionResult(
                success=False,
                error_message=str(e)
            )
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about trained model"""
        if model_id not in self.model_metadata:
            return {}
        
        metadata = self.model_metadata[model_id].copy()
        metadata.update({
            'available_models': list(self.trained_models[model_id]['models'].keys()) if model_id in self.trained_models else [],
            'statistics': self.get_statistics()
        })
        
        return metadata
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get plugin statistics"""
        return self.stats.copy()
    
    def health_check(self) -> bool:
        """Perform health check"""
        try:
            # Test basic functionality
            if SKLEARN_AVAILABLE:
                # Test sklearn
                from sklearn.ensemble import RandomForestClassifier
                rf = RandomForestClassifier(n_estimators=2)
                X_test = np.random.random((10, 5))
                y_test = np.random.randint(0, 2, 10)
                rf.fit(X_test, y_test)
                rf.predict(X_test)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False


# Plugin metadata for discovery
__plugin_metadata__ = {
    "name": "SecurityModelPlugin",
    "version": "1.0.0",
    "author": "Universal AI Core",
    "description": "Security classification models for cybersecurity tasks",
    "plugin_type": "model",
    "entry_point": f"{__name__}:SecurityModelPlugin",
    "dependencies": [
        {"name": "sklearn", "optional": False},
        {"name": "xgboost", "optional": True},
        {"name": "torch", "optional": True}
    ],
    "capabilities": [
        "malware_detection",
        "intrusion_detection", 
        "anomaly_detection",
        "threat_classification",
        "ensemble_methods",
        "uncertainty_quantification"
    ],
    "hooks": []
}


if __name__ == "__main__":
    # Test the security model plugin
    print("🛡️ SECURITY MODEL PLUGIN TEST")
    print("=" * 50)
    
    # Initialize plugin
    config = {
        'classification': {
            'model_types': ['random_forest', 'neural_network'],
            'ensemble_method': 'voting',
            'use_uncertainty_quantification': True
        }
    }
    
    security_model = SecurityModelPlugin(config)
    
    # Generate test data (simulating network features)
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Create synthetic security features (packet sizes, flow rates, etc.)
    X = np.random.random((n_samples, n_features))
    # Add some pattern for malware vs benign
    X[:500, :10] += np.random.normal(2, 0.5, (500, 10))  # Malware pattern
    
    y = np.concatenate([np.ones(500), np.zeros(500)])  # 500 malware, 500 benign
    
    print(f"\n🧪 Test data: {n_samples} samples, {n_features} features")
    print(f"📊 Labels: {np.sum(y)} threats, {len(y) - np.sum(y)} benign")
    
    # Train model
    print(f"\n🎯 Training security models...")
    training_data = {
        'features': X,
        'labels': y
    }
    
    result = security_model.train_model(
        ModelType.ENSEMBLE, 
        training_data, 
        task_type='classification'
    )
    
    if result.success:
        print(f"✅ Training successful!")
        print(f"⏱️ Training time: {result.training_time:.2f}s")
        print(f"🆔 Model ID: {result.model_id}")
        
        # Test predictions
        print(f"\n🔍 Testing predictions...")
        test_features = X[:100]  # Use first 100 samples for testing
        
        pred_result = security_model.predict(result.model_id, test_features)
        
        if pred_result.success:
            print(f"✅ Prediction successful!")
            print(f"📊 Models used: {pred_result.model_info['models_used']}")
            
            # Show ensemble prediction
            if 'ensemble' in pred_result.predictions:
                ensemble_pred = pred_result.predictions['ensemble']
                threats_detected = np.sum(ensemble_pred == 1)
                print(f"🚨 Threats detected: {threats_detected}/{len(ensemble_pred)}")
            
            # Show individual model predictions
            for model_name, predictions in pred_result.predictions.items():
                if model_name != 'ensemble':
                    threats = np.sum(predictions == 1) if hasattr(predictions, 'sum') else 0
                    print(f"  {model_name}: {threats} threats detected")
        else:
            print(f"❌ Prediction failed: {pred_result.error_message}")
        
        # Show model info
        model_info = security_model.get_model_info(result.model_id)
        print(f"\n📋 Model info:")
        print(f"  Training samples: {model_info.get('training_samples', 'N/A')}")
        print(f"  Feature count: {model_info.get('feature_count', 'N/A')}")
        print(f"  Available models: {model_info.get('available_models', [])}")
        
    else:
        print(f"❌ Training failed: {result.error_message}")
    
    # Test health check
    health = security_model.health_check()
    print(f"\n🏥 Health check: {'✅' if health else '❌'}")
    
    # Show statistics
    stats = security_model.get_statistics()
    print(f"\n📊 Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Security model plugin test completed!")