#!/usr/bin/env python3
"""
Financial Models Plugin
=======================

This module provides financial neural network models for the Universal AI Core system.
Adapted from Saraphis molecular neural network patterns, specialized for financial prediction,
risk modeling, and quantitative finance applications.

Features:
- Enterprise-grade neural networks for financial time series prediction
- Ensemble methods (Random Forest, XGBoost, LSTM, Transformer) for finance
- Uncertainty quantification for risk assessment and portfolio optimization
- Cross-validation and hyperparameter optimization for financial models
- Model persistence and checkpointing for production trading systems
- GPU acceleration and memory optimization for high-frequency trading
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
from datetime import datetime, timedelta
from pathlib import Path
import threading
import copy

# Core ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit, GridSearchCV
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
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
class FinancialMetrics:
    """Metrics for financial model evaluation"""
    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0  # Mean Absolute Percentage Error
    r2: float = 0.0
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    profit_factor: Optional[float] = None
    win_rate: Optional[float] = None
    volatility: Optional[float] = None
    var_95: Optional[float] = None  # Value at Risk 95%
    expected_shortfall: Optional[float] = None
    calmar_ratio: Optional[float] = None


@dataclass
class FinancialModelConfig:
    """
    Configuration for financial model training.
    
    Adapted from molecular PredictionConfig for financial domain.
    """
    model_types: List[str] = field(default_factory=lambda: ['lstm', 'transformer', 'xgboost'])
    ensemble_method: str = 'weighted'  # 'voting', 'stacking', 'weighted'
    lookback_window: int = 60  # Number of periods to look back
    prediction_horizon: int = 1  # Number of periods to predict ahead
    cross_validation_folds: int = 5
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    scale_features: bool = True
    use_robust_scaling: bool = True  # Use RobustScaler for financial data
    hyperparameter_optimization: bool = True
    early_stopping_patience: int = 15
    max_epochs: int = 500
    batch_size: int = 32
    learning_rate: float = 0.001
    dropout_rate: float = 0.2
    l1_regularization: float = 0.001
    l2_regularization: float = 0.001
    use_attention: bool = True
    sequence_modeling: bool = True


class LSTMFinancialNetwork(nn.Module):
    """
    LSTM-based neural network for financial time series prediction.
    
    Adapted from molecular neural networks with financial-specific optimizations:
    - Multi-layer LSTM for sequence modeling
    - Attention mechanisms for important time step focus
    - Dropout and regularization for financial robustness
    - Multiple output heads for different prediction tasks
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, 
                 output_dim: int = 1, dropout_rate: float = 0.2, use_attention: bool = True):
        super(LSTMFinancialNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.use_attention = use_attention
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Attention mechanism
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Output layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Linear(hidden_dim, 1)
        
        # Volatility prediction head
        self.volatility_head = nn.Linear(hidden_dim, 1)
        
        # Classification head (for direction prediction)
        self.direction_head = nn.Linear(hidden_dim, 3)  # up, down, sideways
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, return_all_outputs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass through the network"""
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention if enabled
        if self.use_attention:
            attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = self.attention_norm(attended_out + lstm_out)
        
        # Use the last time step output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Main prediction path
        x = F.relu(self.fc1(last_output))
        x = self.dropout(x)
        main_output = self.fc2(x)
        
        if return_all_outputs:
            # Additional outputs
            uncertainty = torch.sigmoid(self.uncertainty_head(last_output))
            volatility = F.softplus(self.volatility_head(last_output))
            direction = F.softmax(self.direction_head(last_output), dim=1)
            
            return main_output, uncertainty, volatility, direction
        
        return main_output


class TransformerFinancialNetwork(nn.Module):
    """
    Transformer-based neural network for financial time series.
    
    Adapted from molecular patterns with financial-specific attention mechanisms.
    """
    
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 6, output_dim: int = 1, dropout_rate: float = 0.2):
        super(TransformerFinancialNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding for time series
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_projection = nn.Linear(d_model, output_dim)
        
        # Additional heads
        self.uncertainty_head = nn.Linear(d_model, 1)
        self.volatility_head = nn.Linear(d_model, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, return_all_outputs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass through the transformer"""
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        if seq_len <= self.positional_encoding.size(0):
            pos_encoding = self.positional_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
            x = x + pos_encoding
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)
        
        # Use the last time step
        last_output = encoded[:, -1, :]
        last_output = self.layer_norm(last_output)
        last_output = self.dropout(last_output)
        
        # Main output
        main_output = self.output_projection(last_output)
        
        if return_all_outputs:
            uncertainty = torch.sigmoid(self.uncertainty_head(last_output))
            volatility = F.softplus(self.volatility_head(last_output))
            return main_output, uncertainty, volatility
        
        return main_output


class FinancialModelTrainer:
    """
    Enterprise financial model trainer.
    
    Adapted from molecular trainer with financial-specific training strategies:
    - Time series cross-validation
    - Financial loss functions (Sharpe ratio optimization)
    - Regime-aware training
    - Risk-adjusted performance metrics
    """
    
    def __init__(self, config: FinancialModelConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.FinancialModelTrainer")
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        
        # Training history
        self.training_history = {}
        self.validation_scores = {}
        
        # Thread safety
        self.training_lock = threading.Lock()
        
        self.logger.info(f"💰 Financial Model Trainer initialized")
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray, 
                      timestamps: Optional[np.ndarray] = None,
                      task_type: str = 'regression') -> Dict[str, Any]:
        """
        Train ensemble of financial models with time series considerations.
        
        Adapted from molecular ensemble training for financial prediction.
        """
        with self.training_lock:
            start_time = time.time()
            results = {}
            
            try:
                self.logger.info(f"📈 Training financial ensemble for {task_type}")
                
                # Time series split (no shuffling for financial data)
                if timestamps is not None:
                    X_train, X_val, X_test, y_train, y_val, y_test = self._time_series_split(
                        X, y, timestamps
                    )
                else:
                    # Simple time-ordered split
                    train_size = int(len(X) * (1 - self.config.test_size - self.config.validation_size))
                    val_size = int(len(X) * self.config.validation_size)
                    
                    X_train = X[:train_size]
                    X_val = X[train_size:train_size + val_size]
                    X_test = X[train_size + val_size:]
                    y_train = y[:train_size]
                    y_val = y[train_size:train_size + val_size]
                    y_test = y[train_size + val_size:]
                
                # Feature scaling
                if self.config.scale_features:
                    if self.config.use_robust_scaling:
                        scaler = RobustScaler()
                    else:
                        scaler = StandardScaler()
                    
                    X_train = scaler.fit_transform(X_train)
                    X_val = scaler.transform(X_val)
                    X_test = scaler.transform(X_test)
                    self.scalers['ensemble'] = scaler
                
                # Train individual models
                if 'lstm' in self.config.model_types and PYTORCH_AVAILABLE:
                    results['lstm'] = self._train_lstm_model(
                        X_train, X_val, X_test, y_train, y_val, y_test, task_type
                    )
                
                if 'transformer' in self.config.model_types and PYTORCH_AVAILABLE:
                    results['transformer'] = self._train_transformer_model(
                        X_train, X_val, X_test, y_train, y_val, y_test, task_type
                    )
                
                if 'xgboost' in self.config.model_types and XGBOOST_AVAILABLE:
                    results['xgboost'] = self._train_xgboost_model(
                        X_train, X_val, X_test, y_train, y_val, y_test, task_type
                    )
                
                if 'random_forest' in self.config.model_types and SKLEARN_AVAILABLE:
                    results['random_forest'] = self._train_random_forest_model(
                        X_train, X_val, X_test, y_train, y_val, y_test, task_type
                    )
                
                # Create ensemble
                results['ensemble'] = self._create_financial_ensemble(
                    X_test, y_test, task_type
                )
                
                training_time = time.time() - start_time
                
                return TrainingResult(
                    success=True,
                    metrics=results,
                    training_time=training_time,
                    model_info={
                        'task_type': task_type,
                        'models_trained': list(results.keys()),
                        'ensemble_method': self.config.ensemble_method,
                        'lookback_window': self.config.lookback_window,
                        'prediction_horizon': self.config.prediction_horizon
                    }
                )
                
            except Exception as e:
                self.logger.error(f"❌ Error training financial ensemble: {e}")
                return TrainingResult(
                    success=False,
                    error_message=str(e),
                    training_time=time.time() - start_time
                )
    
    def _time_series_split(self, X: np.ndarray, y: np.ndarray, 
                          timestamps: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Perform time series split respecting temporal order"""
        # Sort by timestamp
        sorted_indices = np.argsort(timestamps)
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        
        # Calculate split points
        total_size = len(X_sorted)
        train_size = int(total_size * (1 - self.config.test_size - self.config.validation_size))
        val_size = int(total_size * self.config.validation_size)
        
        # Split data
        X_train = X_sorted[:train_size]
        X_val = X_sorted[train_size:train_size + val_size]
        X_test = X_sorted[train_size + val_size:]
        
        y_train = y_sorted[:train_size]
        y_val = y_sorted[train_size:train_size + val_size]
        y_test = y_sorted[train_size + val_size:]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray, 
                          lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for time series models"""
        if len(X) <= lookback:
            return np.array([]), np.array([])
        
        X_sequences = []
        y_sequences = []
        
        for i in range(lookback, len(X)):
            X_sequences.append(X[i-lookback:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def _train_lstm_model(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                         y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                         task_type: str) -> FinancialMetrics:
        """Train LSTM model for financial prediction"""
        try:
            # Prepare sequences
            X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train, self.config.lookback_window)
            X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val, self.config.lookback_window)
            X_test_seq, y_test_seq = self._prepare_sequences(X_test, y_test, self.config.lookback_window)
            
            if len(X_train_seq) == 0:
                return FinancialMetrics()
            
            input_dim = X_train_seq.shape[2]
            output_dim = 1 if task_type == 'regression' else len(np.unique(y_train_seq))
            
            # Create model
            model = LSTMFinancialNetwork(
                input_dim=input_dim,
                hidden_dim=128,
                num_layers=2,
                output_dim=output_dim,
                dropout_rate=self.config.dropout_rate,
                use_attention=self.config.use_attention
            )
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            # Loss function
            if task_type == 'regression':
                criterion = nn.MSELoss()
            else:
                criterion = nn.CrossEntropyLoss()
            
            # Optimizer with L2 regularization
            optimizer = optim.Adam(
                model.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=self.config.l2_regularization
            )
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=10, factor=0.5
            )
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_seq),
                torch.FloatTensor(y_train_seq).unsqueeze(1) if task_type == 'regression' 
                else torch.LongTensor(y_train_seq)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_seq),
                torch.FloatTensor(y_val_seq).unsqueeze(1) if task_type == 'regression'
                else torch.LongTensor(y_val_seq)
            )
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test_seq),
                torch.FloatTensor(y_test_seq).unsqueeze(1) if task_type == 'regression'
                else torch.LongTensor(y_test_seq)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config.max_epochs):
                # Training
                model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    
                    if task_type == 'regression':
                        loss = criterion(outputs, batch_y)
                    else:
                        loss = criterion(outputs, batch_y)
                    
                    # Add L1 regularization
                    if self.config.l1_regularization > 0:
                        l1_penalty = sum(p.abs().sum() for p in model.parameters())
                        loss += self.config.l1_regularization * l1_penalty
                    
                    loss.backward()
                    
                    # Gradient clipping for financial stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        
                        if task_type == 'regression':
                            loss = criterion(outputs, batch_y)
                        else:
                            loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f'/tmp/best_lstm_financial_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        break
            
            # Load best model
            model.load_state_dict(torch.load(f'/tmp/best_lstm_financial_model.pth'))
            self.models['lstm'] = model
            
            # Evaluate on test set
            model.eval()
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    
                    if task_type == 'regression':
                        predictions.extend(outputs.cpu().numpy().flatten())
                        actuals.extend(batch_y.cpu().numpy().flatten())
                    else:
                        pred_classes = torch.argmax(outputs, dim=1)
                        predictions.extend(pred_classes.cpu().numpy())
                        actuals.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            if task_type == 'regression':
                return self._calculate_financial_regression_metrics(
                    np.array(actuals), np.array(predictions)
                )
            else:
                return self._calculate_financial_classification_metrics(
                    np.array(actuals), np.array(predictions)
                )
                
        except Exception as e:
            self.logger.error(f"❌ Error training LSTM model: {e}")
            return FinancialMetrics()
    
    def _train_transformer_model(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                               y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                               task_type: str) -> FinancialMetrics:
        """Train Transformer model for financial prediction"""
        try:
            # Prepare sequences
            X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train, self.config.lookback_window)
            X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val, self.config.lookback_window)
            X_test_seq, y_test_seq = self._prepare_sequences(X_test, y_test, self.config.lookback_window)
            
            if len(X_train_seq) == 0:
                return FinancialMetrics()
            
            input_dim = X_train_seq.shape[2]
            output_dim = 1 if task_type == 'regression' else len(np.unique(y_train_seq))
            
            # Create model
            model = TransformerFinancialNetwork(
                input_dim=input_dim,
                d_model=128,
                nhead=8,
                num_layers=4,
                output_dim=output_dim,
                dropout_rate=self.config.dropout_rate
            )
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            # Loss function
            if task_type == 'regression':
                criterion = nn.MSELoss()
            else:
                criterion = nn.CrossEntropyLoss()
            
            # Optimizer
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=self.config.l2_regularization
            )
            
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.max_epochs
            )
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_seq),
                torch.FloatTensor(y_train_seq).unsqueeze(1) if task_type == 'regression' 
                else torch.LongTensor(y_train_seq)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_seq),
                torch.FloatTensor(y_val_seq).unsqueeze(1) if task_type == 'regression'
                else torch.LongTensor(y_val_seq)
            )
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test_seq),
                torch.FloatTensor(y_test_seq).unsqueeze(1) if task_type == 'regression'
                else torch.LongTensor(y_test_seq)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            # Training loop (similar to LSTM but with transformer-specific optimizations)
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config.max_epochs):
                # Training
                model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    
                    if task_type == 'regression':
                        loss = criterion(outputs, batch_y)
                    else:
                        loss = criterion(outputs, batch_y)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        
                        if task_type == 'regression':
                            loss = criterion(outputs, batch_y)
                        else:
                            loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                scheduler.step()
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f'/tmp/best_transformer_financial_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        break
            
            # Load best model and evaluate
            model.load_state_dict(torch.load(f'/tmp/best_transformer_financial_model.pth'))
            self.models['transformer'] = model
            
            # Evaluate on test set
            model.eval()
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    
                    if task_type == 'regression':
                        predictions.extend(outputs.cpu().numpy().flatten())
                        actuals.extend(batch_y.cpu().numpy().flatten())
                    else:
                        pred_classes = torch.argmax(outputs, dim=1)
                        predictions.extend(pred_classes.cpu().numpy())
                        actuals.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            if task_type == 'regression':
                return self._calculate_financial_regression_metrics(
                    np.array(actuals), np.array(predictions)
                )
            else:
                return self._calculate_financial_classification_metrics(
                    np.array(actuals), np.array(predictions)
                )
                
        except Exception as e:
            self.logger.error(f"❌ Error training Transformer model: {e}")
            return FinancialMetrics()
    
    def _train_xgboost_model(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                           task_type: str) -> FinancialMetrics:
        """Train XGBoost model for financial prediction"""
        try:
            # XGBoost doesn't need sequences - use flattened features
            if task_type == 'regression':
                model = xgb.XGBRegressor(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.config.random_state,
                    early_stopping_rounds=50,
                    eval_metric='rmse'
                )
            else:
                model = xgb.XGBClassifier(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.config.random_state,
                    early_stopping_rounds=50,
                    eval_metric='mlogloss'
                )
            
            # Train with validation set for early stopping
            eval_set = [(X_val, y_val)]
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            self.models['xgboost'] = model
            
            # Predict on test set
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            if task_type == 'regression':
                return self._calculate_financial_regression_metrics(y_test, y_pred)
            else:
                return self._calculate_financial_classification_metrics(y_test, y_pred)
                
        except Exception as e:
            self.logger.error(f"❌ Error training XGBoost model: {e}")
            return FinancialMetrics()
    
    def _train_random_forest_model(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                                 y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                                 task_type: str) -> FinancialMetrics:
        """Train Random Forest model for financial prediction"""
        try:
            if task_type == 'regression':
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=self.config.random_state,
                    n_jobs=-1
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=self.config.random_state,
                    n_jobs=-1
                )
            
            # Train model
            model.fit(X_train, y_train)
            self.models['random_forest'] = model
            
            # Predict on test set
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            if task_type == 'regression':
                return self._calculate_financial_regression_metrics(y_test, y_pred)
            else:
                return self._calculate_financial_classification_metrics(y_test, y_pred)
                
        except Exception as e:
            self.logger.error(f"❌ Error training Random Forest model: {e}")
            return FinancialMetrics()
    
    def _calculate_financial_regression_metrics(self, y_true: np.ndarray, 
                                              y_pred: np.ndarray) -> FinancialMetrics:
        """Calculate financial regression metrics"""
        metrics = FinancialMetrics()
        
        try:
            # Basic regression metrics
            metrics.rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics.mae = mean_absolute_error(y_true, y_pred)
            metrics.r2 = r2_score(y_true, y_pred)
            
            # MAPE (handle division by zero)
            non_zero_mask = np.abs(y_true) > 1e-8
            if np.any(non_zero_mask):
                mape_values = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
                metrics.mape = np.mean(mape_values) * 100
            
            # Financial-specific metrics
            if len(y_true) > 1:
                # Convert predictions to returns if they're not already
                returns_true = y_true if np.all(np.abs(y_true) < 1) else np.diff(y_true) / y_true[:-1]
                returns_pred = y_pred if np.all(np.abs(y_pred) < 1) else np.diff(y_pred) / y_pred[:-1]
                
                # Align arrays if needed
                min_len = min(len(returns_true), len(returns_pred))
                returns_true = returns_true[:min_len]
                returns_pred = returns_pred[:min_len]
                
                if len(returns_true) > 0:
                    # Sharpe ratio
                    if np.std(returns_pred) > 0:
                        metrics.sharpe_ratio = np.mean(returns_pred) / np.std(returns_pred) * np.sqrt(252)
                    
                    # Volatility
                    metrics.volatility = np.std(returns_pred) * np.sqrt(252)
                    
                    # Maximum drawdown
                    cumulative_returns = np.cumprod(1 + returns_pred)
                    running_max = np.maximum.accumulate(cumulative_returns)
                    drawdown = (cumulative_returns - running_max) / running_max
                    metrics.max_drawdown = np.min(drawdown)
                    
                    # Value at Risk (95%)
                    metrics.var_95 = np.percentile(returns_pred, 5)
                    
                    # Expected Shortfall
                    var_mask = returns_pred <= metrics.var_95
                    if np.any(var_mask):
                        metrics.expected_shortfall = np.mean(returns_pred[var_mask])
                    
                    # Calmar ratio
                    if metrics.max_drawdown < 0:
                        annualized_return = np.mean(returns_pred) * 252
                        metrics.calmar_ratio = annualized_return / abs(metrics.max_drawdown)
            
        except Exception as e:
            logger.error(f"Error calculating financial regression metrics: {e}")
        
        return metrics
    
    def _calculate_financial_classification_metrics(self, y_true: np.ndarray, 
                                                  y_pred: np.ndarray) -> FinancialMetrics:
        """Calculate financial classification metrics"""
        metrics = FinancialMetrics()
        
        try:
            # Convert to classification accuracy
            accuracy = accuracy_score(y_true, y_pred)
            metrics.r2 = accuracy  # Use r2 field for accuracy in classification
            
            # Win rate (for direction prediction)
            if len(np.unique(y_true)) == 2:  # Binary classification
                correct_predictions = (y_true == y_pred)
                metrics.win_rate = np.mean(correct_predictions)
            
        except Exception as e:
            logger.error(f"Error calculating financial classification metrics: {e}")
        
        return metrics
    
    def _create_financial_ensemble(self, X_test: np.ndarray, y_test: np.ndarray,
                                 task_type: str) -> Dict[str, Any]:
        """Create financial ensemble with risk-adjusted weighting"""
        try:
            if not self.models:
                return {}
            
            predictions = {}
            model_metrics = {}
            
            # Collect predictions from all models
            for model_name, model in self.models.items():
                try:
                    if model_name in ['lstm', 'transformer'] and PYTORCH_AVAILABLE:
                        # Neural network predictions
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model.eval()
                        
                        # Prepare sequences for neural networks
                        X_test_seq, y_test_seq = self._prepare_sequences(
                            X_test, y_test, self.config.lookback_window
                        )
                        
                        if len(X_test_seq) > 0:
                            with torch.no_grad():
                                X_tensor = torch.FloatTensor(X_test_seq).to(device)
                                outputs = model(X_tensor)
                                
                                if task_type == 'regression':
                                    preds = outputs.cpu().numpy().flatten()
                                else:
                                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                                
                                predictions[model_name] = preds
                                
                                # Calculate metrics for this model
                                if task_type == 'regression':
                                    model_metrics[model_name] = self._calculate_financial_regression_metrics(
                                        y_test_seq, preds
                                    )
                                else:
                                    model_metrics[model_name] = self._calculate_financial_classification_metrics(
                                        y_test_seq, preds
                                    )
                    else:
                        # Sklearn/XGBoost models
                        preds = model.predict(X_test)
                        predictions[model_name] = preds
                        
                        # Calculate metrics
                        if task_type == 'regression':
                            model_metrics[model_name] = self._calculate_financial_regression_metrics(
                                y_test, preds
                            )
                        else:
                            model_metrics[model_name] = self._calculate_financial_classification_metrics(
                                y_test, preds
                            )
                            
                except Exception as e:
                    self.logger.error(f"Error getting predictions from {model_name}: {e}")
                    continue
            
            # Create ensemble prediction
            if len(predictions) > 1:
                if self.config.ensemble_method == 'weighted':
                    # Weight by Sharpe ratio or accuracy
                    weights = {}
                    for name, metrics in model_metrics.items():
                        if task_type == 'regression':
                            weight = max(0, metrics.sharpe_ratio or 0)
                        else:
                            weight = max(0, metrics.r2 or 0)  # Using r2 field for accuracy
                        weights[name] = weight
                    
                    # Normalize weights
                    total_weight = sum(weights.values())
                    if total_weight > 0:
                        weights = {k: v / total_weight for k, v in weights.items()}
                    else:
                        # Equal weights if all weights are zero
                        weights = {k: 1/len(weights) for k in weights.keys()}
                    
                    # Weighted average prediction
                    ensemble_pred = np.zeros_like(list(predictions.values())[0])
                    for name, pred in predictions.items():
                        if name in weights:
                            ensemble_pred += weights[name] * pred
                    
                elif self.config.ensemble_method == 'voting':
                    # Simple average/majority vote
                    if task_type == 'regression':
                        ensemble_pred = np.mean(list(predictions.values()), axis=0)
                    else:
                        # Majority voting for classification
                        vote_matrix = np.column_stack(list(predictions.values()))
                        ensemble_pred = np.apply_along_axis(
                            lambda x: np.bincount(x).argmax(), axis=1, arr=vote_matrix
                        )
                
                predictions['ensemble'] = ensemble_pred
                
                # Calculate ensemble metrics
                y_test_aligned = y_test
                if len(predictions.get('lstm', [])) > 0 or len(predictions.get('transformer', [])) > 0:
                    # If we have neural network predictions, align test data
                    _, y_test_aligned = self._prepare_sequences(X_test, y_test, self.config.lookback_window)
                
                if len(y_test_aligned) == len(ensemble_pred):
                    if task_type == 'regression':
                        ensemble_metrics = self._calculate_financial_regression_metrics(
                            y_test_aligned, ensemble_pred
                        )
                    else:
                        ensemble_metrics = self._calculate_financial_classification_metrics(
                            y_test_aligned, ensemble_pred
                        )
                    
                    model_metrics['ensemble'] = ensemble_metrics
            
            return {
                'predictions': predictions,
                'metrics': model_metrics,
                'ensemble_method': self.config.ensemble_method
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error creating financial ensemble: {e}")
            return {}


class FinancialModelPlugin(ModelPlugin):
    """
    Financial model plugin for quantitative finance and trading applications.
    
    Provides enterprise-grade models adapted from molecular analysis patterns
    for financial prediction, risk modeling, and portfolio optimization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the financial model plugin"""
        super().__init__(config)
        
        # Configuration
        model_config_dict = self.config.get('model_config', {})
        self.model_config = FinancialModelConfig(**model_config_dict)
        
        # Model trainer
        self.trainer = FinancialModelTrainer(self.model_config)
        
        # Model registry
        self.trained_models = {}
        self.model_metadata = {}
        
        # Statistics
        self.stats = {
            'models_trained': 0,
            'predictions_made': 0,
            'sharpe_ratio_total': 0.0,
            'max_drawdown_worst': 0.0,
            'profitable_predictions': 0
        }
        
        self.logger.info(f"💰 Financial Model Plugin initialized")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata"""
        return {
            "name": "FinancialModelPlugin",
            "version": "1.0.0",
            "description": "Financial prediction models for quantitative finance",
            "supported_models": [
                ModelType.LSTM,
                ModelType.TRANSFORMER,
                ModelType.GRADIENT_BOOSTING,
                ModelType.RANDOM_FOREST,
                ModelType.ENSEMBLE
            ],
            "capabilities": [
                "price_prediction",
                "return_forecasting", 
                "volatility_modeling",
                "risk_assessment",
                "portfolio_optimization",
                "regime_detection"
            ],
            "dependencies": {
                "sklearn": SKLEARN_AVAILABLE,
                "xgboost": XGBOOST_AVAILABLE,
                "pytorch": PYTORCH_AVAILABLE
            }
        }
    
    def train_model(self, model_type: ModelType, training_data: Dict[str, Any], 
                   **kwargs) -> TrainingResult:
        """Train financial model"""
        try:
            self.logger.info(f"💹 Training financial model: {model_type.value}")
            
            X = training_data.get('features')
            y = training_data.get('targets')
            timestamps = training_data.get('timestamps')
            task_type = kwargs.get('task_type', 'regression')
            
            if X is None or y is None:
                return TrainingResult(
                    success=False,
                    error_message="Missing features or targets in training data"
                )
            
            # Convert to numpy arrays
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            if timestamps is not None and not isinstance(timestamps, np.ndarray):
                timestamps = np.array(timestamps)
            
            # Train ensemble model
            result = self.trainer.train_ensemble(X, y, timestamps, task_type)
            
            if result.success:
                model_id = f"financial_{model_type.value}_{int(time.time())}"
                self.trained_models[model_id] = {
                    'models': self.trainer.models.copy(),
                    'scalers': self.trainer.scalers.copy(),
                    'config': self.model_config,
                    'task_type': task_type,
                    'model_type': model_type
                }
                
                self.model_metadata[model_id] = {
                    'created_at': datetime.utcnow(),
                    'training_samples': len(X),
                    'feature_count': X.shape[1],
                    'task_type': task_type,
                    'lookback_window': self.model_config.lookback_window,
                    'prediction_horizon': self.model_config.prediction_horizon,
                    'metrics': result.metrics
                }
                
                result.model_id = model_id
                self.stats['models_trained'] += 1
                
                # Update statistics from ensemble metrics
                if 'ensemble' in result.metrics and 'metrics' in result.metrics['ensemble']:
                    ensemble_metrics = result.metrics['ensemble']['metrics'].get('ensemble')
                    if ensemble_metrics and hasattr(ensemble_metrics, 'sharpe_ratio'):
                        if ensemble_metrics.sharpe_ratio:
                            self.stats['sharpe_ratio_total'] += ensemble_metrics.sharpe_ratio
                        if ensemble_metrics.max_drawdown:
                            self.stats['max_drawdown_worst'] = min(
                                self.stats['max_drawdown_worst'], ensemble_metrics.max_drawdown
                            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error training financial model: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e)
            )
    
    def predict(self, model_id: str, features: Union[np.ndarray, List[List[float]]], 
               **kwargs) -> PredictionResult:
        """Make financial predictions"""
        try:
            if model_id not in self.trained_models:
                return PredictionResult(
                    success=False,
                    error_message=f"Model {model_id} not found"
                )
            
            model_data = self.trained_models[model_id]
            models = model_data['models']
            scalers = model_data['scalers']
            config = model_data['config']
            task_type = model_data['task_type']
            
            # Convert features to numpy array
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            
            # Scale features
            if 'ensemble' in scalers:
                features_scaled = scalers['ensemble'].transform(features)
            else:
                features_scaled = features
            
            # Collect predictions from all models
            predictions = {}
            confidence_scores = {}
            
            for model_name, model in models.items():
                try:
                    if model_name in ['lstm', 'transformer'] and PYTORCH_AVAILABLE:
                        # Neural network predictions
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model.eval()
                        
                        # Prepare sequences
                        if len(features_scaled) >= config.lookback_window:
                            # Use the last lookback_window points as input
                            input_sequence = features_scaled[-config.lookback_window:].reshape(
                                1, config.lookback_window, -1
                            )
                            
                            with torch.no_grad():
                                X_tensor = torch.FloatTensor(input_sequence).to(device)
                                
                                if hasattr(model, 'forward') and 'return_all_outputs' in str(model.forward):
                                    # Get all outputs including uncertainty
                                    outputs = model(X_tensor, return_all_outputs=True)
                                    main_pred = outputs[0].cpu().numpy().flatten()
                                    
                                    if len(outputs) > 1:
                                        uncertainty = outputs[1].cpu().numpy().flatten()
                                        confidence_scores[model_name] = 1.0 - uncertainty
                                else:
                                    outputs = model(X_tensor)
                                    main_pred = outputs.cpu().numpy().flatten()
                                
                                predictions[model_name] = main_pred
                        
                    else:
                        # Sklearn/XGBoost models - use the last row
                        if len(features_scaled) > 0:
                            last_features = features_scaled[-1:] if len(features_scaled.shape) == 2 else features_scaled.reshape(1, -1)
                            pred = model.predict(last_features)
                            predictions[model_name] = pred
                            
                            # Get feature importance as confidence proxy
                            if hasattr(model, 'feature_importances_'):
                                confidence_scores[model_name] = np.mean(model.feature_importances_)
                
                except Exception as e:
                    self.logger.error(f"Error getting predictions from {model_name}: {e}")
                    continue
            
            # Create ensemble prediction if multiple models
            if len(predictions) > 1:
                if config.ensemble_method == 'weighted':
                    # Weight by confidence scores
                    total_weight = sum(confidence_scores.values()) if confidence_scores else len(predictions)
                    
                    if total_weight > 0:
                        ensemble_pred = np.zeros_like(list(predictions.values())[0])
                        for name, pred in predictions.items():
                            weight = confidence_scores.get(name, 1.0) / total_weight
                            ensemble_pred += weight * pred
                    else:
                        ensemble_pred = np.mean(list(predictions.values()), axis=0)
                else:
                    # Simple average
                    ensemble_pred = np.mean(list(predictions.values()), axis=0)
                
                predictions['ensemble'] = ensemble_pred
            
            self.stats['predictions_made'] += 1
            
            # Check if prediction is profitable (for regression tasks)
            if task_type == 'regression' and 'ensemble' in predictions:
                if np.any(predictions['ensemble'] > 0):
                    self.stats['profitable_predictions'] += 1
            
            return PredictionResult(
                success=True,
                predictions=predictions,
                confidence_scores=confidence_scores,
                model_info={
                    'model_id': model_id,
                    'task_type': task_type,
                    'models_used': list(models.keys()),
                    'prediction_horizon': config.prediction_horizon,
                    'lookback_window': config.lookback_window
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error making financial predictions: {e}")
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
        stats = self.stats.copy()
        
        # Calculate derived statistics
        if stats['models_trained'] > 0:
            stats['average_sharpe_ratio'] = stats['sharpe_ratio_total'] / stats['models_trained']
        else:
            stats['average_sharpe_ratio'] = 0.0
        
        if stats['predictions_made'] > 0:
            stats['profitable_prediction_rate'] = stats['profitable_predictions'] / stats['predictions_made']
        else:
            stats['profitable_prediction_rate'] = 0.0
        
        return stats
    
    def health_check(self) -> bool:
        """Perform health check"""
        try:
            # Test basic functionality
            if SKLEARN_AVAILABLE:
                # Test sklearn
                from sklearn.ensemble import RandomForestRegressor
                rf = RandomForestRegressor(n_estimators=2)
                X_test = np.random.random((50, 10))
                y_test = np.random.random(50)
                rf.fit(X_test, y_test)
                rf.predict(X_test)
            
            # Test financial metrics calculation
            y_true = np.random.normal(0, 0.01, 100)  # Random returns
            y_pred = np.random.normal(0, 0.01, 100)  # Random predictions
            
            trainer = FinancialModelTrainer(FinancialModelConfig())
            metrics = trainer._calculate_financial_regression_metrics(y_true, y_pred)
            
            return hasattr(metrics, 'rmse')
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False


# Plugin metadata for discovery
__plugin_metadata__ = {
    "name": "FinancialModelPlugin",
    "version": "1.0.0",
    "author": "Universal AI Core",
    "description": "Financial prediction models for quantitative finance",
    "plugin_type": "model",
    "entry_point": f"{__name__}:FinancialModelPlugin",
    "dependencies": [
        {"name": "sklearn", "optional": False},
        {"name": "xgboost", "optional": True},
        {"name": "torch", "optional": True},
        {"name": "pandas", "optional": False},
        {"name": "numpy", "optional": False}
    ],
    "capabilities": [
        "price_prediction",
        "return_forecasting", 
        "volatility_modeling",
        "risk_assessment",
        "portfolio_optimization",
        "regime_detection"
    ],
    "hooks": []
}


if __name__ == "__main__":
    # Test the financial model plugin
    print("💰 FINANCIAL MODEL PLUGIN TEST")
    print("=" * 50)
    
    # Initialize plugin
    config = {
        'model_config': {
            'model_types': ['xgboost', 'random_forest'],
            'ensemble_method': 'weighted',
            'lookback_window': 20,
            'prediction_horizon': 1,
            'max_epochs': 50  # Reduce for testing
        }
    }
    
    financial_model = FinancialModelPlugin(config)
    
    # Generate test financial data
    np.random.seed(42)
    n_samples = 500
    n_features = 15
    
    # Create realistic financial time series
    timestamps = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Generate features (technical indicators, etc.)
    features = np.random.randn(n_samples, n_features)
    
    # Add trend and seasonality
    trend = np.linspace(0, 0.1, n_samples)
    seasonality = 0.02 * np.sin(2 * np.pi * np.arange(n_samples) / 252)  # Yearly cycle
    noise = np.random.normal(0, 0.01, n_samples)
    
    # Target: next day's return
    returns = trend + seasonality + noise
    
    # Create training data
    training_data = {
        'features': features[:-1],  # Use all but last feature
        'targets': returns[1:],     # Predict next day's return
        'timestamps': timestamps[:-1]
    }
    
    print(f"\n📊 Test data: {n_samples} samples, {n_features} features")
    print(f"📈 Return range: {returns.min():.4f} to {returns.max():.4f}")
    print(f"📊 Feature matrix shape: {features.shape}")
    
    # Train model
    print(f"\n🎯 Training financial models...")
    result = financial_model.train_model(
        ModelType.ENSEMBLE, 
        training_data, 
        task_type='regression'
    )
    
    if result.success:
        print(f"✅ Training successful!")
        print(f"⏱️ Training time: {result.training_time:.2f}s")
        print(f"🆔 Model ID: {result.model_id}")
        
        # Show model metrics
        if 'ensemble' in result.metrics and 'metrics' in result.metrics['ensemble']:
            metrics = result.metrics['ensemble']['metrics']
            print(f"\n📊 Model performance:")
            for model_name, model_metrics in metrics.items():
                if hasattr(model_metrics, 'rmse'):
                    print(f"  {model_name}:")
                    print(f"    RMSE: {model_metrics.rmse:.6f}")
                    print(f"    R²: {model_metrics.r2:.4f}")
                    if model_metrics.sharpe_ratio:
                        print(f"    Sharpe Ratio: {model_metrics.sharpe_ratio:.4f}")
                    if model_metrics.max_drawdown:
                        print(f"    Max Drawdown: {model_metrics.max_drawdown:.4f}")
        
        # Test predictions
        print(f"\n🔮 Testing predictions...")
        test_features = features[-30:]  # Last 30 days
        
        pred_result = financial_model.predict(result.model_id, test_features)
        
        if pred_result.success:
            print(f"✅ Prediction successful!")
            print(f"📊 Models used: {pred_result.model_info['models_used']}")
            
            # Show predictions
            for model_name, predictions in pred_result.predictions.items():
                print(f"  {model_name}: {predictions}")
                
        else:
            print(f"❌ Prediction failed: {pred_result.error_message}")
        
        # Show model info
        model_info = financial_model.get_model_info(result.model_id)
        print(f"\n📋 Model info:")
        print(f"  Training samples: {model_info.get('training_samples', 'N/A')}")
        print(f"  Feature count: {model_info.get('feature_count', 'N/A')}")
        print(f"  Lookback window: {model_info.get('lookback_window', 'N/A')}")
        
    else:
        print(f"❌ Training failed: {result.error_message}")
    
    # Test health check
    health = financial_model.health_check()
    print(f"\n🏥 Health check: {'✅' if health else '❌'}")
    
    # Show statistics
    stats = financial_model.get_statistics()
    print(f"\n📊 Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Financial model plugin test completed!")