#!/usr/bin/env python3
"""
Golf Neural Network - Neural network architectures for golf prediction.
Supports standard, attention-based, and ensemble models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

from .domain_config import ModelConfig


@dataclass
class NetworkPrediction:
    """Result from neural network prediction."""
    predictions: torch.Tensor
    confidence: torch.Tensor
    feature_importance: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None
    uncertainty_estimates: Optional[torch.Tensor] = None


class GolfNeuralNetwork(nn.Module, ABC):
    """Base class for golf neural networks."""
    
    def __init__(self, input_dim: int, config: ModelConfig):
        super().__init__()
        self.input_dim = input_dim
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Common layers
        self.dropout = nn.Dropout(config.dropout_rate)
        self.batch_norm_enabled = config.batch_norm
        
        # Activation function
        self.activation = self._get_activation_function(config.activation)
        
        # Initialize layers
        self._build_network()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _get_activation_function(self, activation_name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'tanh': nn.Tanh()
        }
        return activations.get(activation_name.lower(), nn.ReLU())
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    @abstractmethod
    def _build_network(self):
        """Build the network architecture."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> NetworkPrediction:
        """Forward pass through the network."""
        pass
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 10) -> NetworkPrediction:
        """Predict with uncertainty estimation using Monte Carlo dropout."""
        self.train()  # Enable dropout for uncertainty estimation
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred.predictions)
        
        self.eval()  # Return to eval mode
        
        predictions = torch.stack(predictions)
        mean_prediction = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return NetworkPrediction(
            predictions=mean_prediction,
            confidence=1.0 - uncertainty / (uncertainty.max() + 1e-8),  # Normalize uncertainty
            uncertainty_estimates=uncertainty
        )


class StandardGolfNetwork(GolfNeuralNetwork):
    """Standard feedforward neural network for golf predictions."""
    
    def _build_network(self):
        """Build standard feedforward network."""
        layers = []
        
        # Input layer
        current_dim = self.input_dim
        
        # Hidden layers
        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            if self.batch_norm_enabled:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self.activation)
            layers.append(nn.Dropout(self.config.dropout_rate))
            
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, 1))  # Single output for fantasy points prediction
        
        self.network = nn.Sequential(*layers)
        
        # Feature importance layer (for interpretability)
        self.feature_importance = nn.Linear(self.input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> NetworkPrediction:
        """Forward pass through standard network."""
        # Main prediction
        predictions = self.network(x)
        
        # Feature importance (simplified)
        importance_weights = torch.sigmoid(self.feature_importance(x))
        feature_importance = importance_weights * x.abs().mean(dim=0, keepdim=True)
        
        # Confidence based on prediction variance
        confidence = torch.sigmoid(predictions)  # Simple confidence estimate
        
        return NetworkPrediction(
            predictions=predictions.squeeze(-1),
            confidence=confidence.squeeze(-1),
            feature_importance=feature_importance
        )


class AttentionGolfNetwork(GolfNeuralNetwork):
    """Attention-based neural network for golf predictions."""
    
    def _build_network(self):
        """Build attention-based network."""
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.config.attention_dim)
        
        # Multi-head attention
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.config.attention_dim,
            num_heads=self.config.attention_heads,
            dropout=self.config.dropout_rate,
            batch_first=True
        )
        
        # Post-attention normalization
        self.attention_norm = nn.LayerNorm(self.config.attention_dim)
        
        # Feedforward layers after attention
        layers = []
        current_dim = self.config.attention_dim
        
        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            if self.batch_norm_enabled:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self.activation)
            layers.append(nn.Dropout(self.config.dropout_rate))
            
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, 1))
        
        self.feedforward = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> NetworkPrediction:
        """Forward pass through attention network."""
        batch_size = x.size(0)
        
        # Project input to attention dimension
        x_proj = self.input_projection(x)  # [batch_size, attention_dim]
        
        # Reshape for attention (add sequence dimension)
        x_seq = x_proj.unsqueeze(1)  # [batch_size, 1, attention_dim]
        
        # Self-attention
        attended, attention_weights = self.multihead_attention(x_seq, x_seq, x_seq)
        attended = attended.squeeze(1)  # [batch_size, attention_dim]
        
        # Residual connection and normalization
        x_attended = self.attention_norm(attended + x_proj)
        
        # Feedforward layers
        predictions = self.feedforward(x_attended)
        
        # Confidence based on attention weights
        attention_confidence = attention_weights.mean(dim=1).squeeze(-1)  # Average attention
        confidence = torch.sigmoid(attention_confidence)
        
        return NetworkPrediction(
            predictions=predictions.squeeze(-1),
            confidence=confidence,
            attention_weights=attention_weights.squeeze(1)  # Remove sequence dimension
        )


class GolfEnsembleNetwork(nn.Module):
    """Ensemble of multiple golf neural networks."""
    
    def __init__(self, input_dim: int, config: ModelConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger('GolfEnsembleNetwork')
        
        # Create ensemble of networks
        self.networks = nn.ModuleList()
        
        for i in range(config.num_models):
            # Alternate between network types for diversity
            if i % 2 == 0:
                network = StandardGolfNetwork(input_dim, config)
            else:
                network = AttentionGolfNetwork(input_dim, config)
            
            self.networks.append(network)
        
        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(config.num_models) / config.num_models)
        
        # Meta-learner for combining predictions
        self.meta_learner = nn.Sequential(
            nn.Linear(config.num_models, config.num_models * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.num_models * 2, config.num_models),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> NetworkPrediction:
        """Forward pass through ensemble."""
        # Get predictions from all networks
        network_predictions = []
        network_confidences = []
        attention_weights_list = []
        
        for network in self.networks:
            pred = network(x)
            network_predictions.append(pred.predictions)
            network_confidences.append(pred.confidence)
            
            if pred.attention_weights is not None:
                attention_weights_list.append(pred.attention_weights)
        
        # Stack predictions
        all_predictions = torch.stack(network_predictions, dim=-1)  # [batch_size, num_models]
        all_confidences = torch.stack(network_confidences, dim=-1)  # [batch_size, num_models]
        
        # Learn ensemble weights using meta-learner
        learned_weights = self.meta_learner(all_predictions)
        
        # Combine with fixed ensemble weights
        final_weights = F.softmax(self.ensemble_weights, dim=0) * learned_weights
        final_weights = final_weights / final_weights.sum(dim=-1, keepdim=True)
        
        # Weighted ensemble prediction
        ensemble_prediction = (all_predictions * final_weights).sum(dim=-1)
        
        # Weighted confidence
        ensemble_confidence = (all_confidences * final_weights).sum(dim=-1)
        
        # Average attention weights if available
        avg_attention = None
        if attention_weights_list:
            avg_attention = torch.stack(attention_weights_list).mean(dim=0)
        
        # Calculate prediction uncertainty
        prediction_variance = ((all_predictions - ensemble_prediction.unsqueeze(-1)) ** 2).mean(dim=-1)
        uncertainty = torch.sqrt(prediction_variance)
        
        return NetworkPrediction(
            predictions=ensemble_prediction,
            confidence=ensemble_confidence,
            attention_weights=avg_attention,
            uncertainty_estimates=uncertainty
        )
    
    def get_individual_predictions(self, x: torch.Tensor) -> List[NetworkPrediction]:
        """Get predictions from individual networks in ensemble."""
        predictions = []
        for network in self.networks:
            pred = network(x)
            predictions.append(pred)
        return predictions


class GolfNeuralFactory:
    """Factory for creating golf neural networks."""
    
    @staticmethod
    def create_network(model_type: str, input_dim: int, config: ModelConfig) -> GolfNeuralNetwork:
        """Create a golf neural network of specified type."""
        model_type = model_type.lower()
        
        if model_type == 'standard':
            return StandardGolfNetwork(input_dim, config)
        elif model_type == 'attention':
            return AttentionGolfNetwork(input_dim, config)
        elif model_type == 'ensemble':
            return GolfEnsembleNetwork(input_dim, config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def create_optimizer(network: nn.Module, config: ModelConfig) -> optim.Optimizer:
        """Create optimizer for the network."""
        if config.optimizer.lower() == 'adam':
            return optim.Adam(
                network.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer.lower() == 'adamw':
            return optim.AdamW(
                network.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer.lower() == 'sgd':
            return optim.SGD(
                network.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=0.9
            )
        else:
            return optim.Adam(
                network.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
    
    @staticmethod
    def create_scheduler(optimizer: optim.Optimizer, config: ModelConfig) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if config.scheduler.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.epochs,
                eta_min=config.learning_rate * 0.01
            )
        elif config.scheduler.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.epochs // 3,
                gamma=0.1
            )
        elif config.scheduler.lower() == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.95
            )
        else:
            return None
    
    @staticmethod
    def create_loss_function(loss_type: str = 'mse') -> nn.Module:
        """Create loss function for training."""
        if loss_type.lower() == 'mse':
            return nn.MSELoss()
        elif loss_type.lower() == 'mae':
            return nn.L1Loss()
        elif loss_type.lower() == 'huber':
            return nn.HuberLoss()
        elif loss_type.lower() == 'smooth_l1':
            return nn.SmoothL1Loss()
        else:
            return nn.MSELoss()


class GolfNeuralTrainer:
    """Trainer for golf neural networks."""
    
    def __init__(self, network: nn.Module, config: ModelConfig):
        self.network = network
        self.config = config
        self.logger = logging.getLogger('GolfNeuralTrainer')
        
        # Create optimizer and scheduler
        self.optimizer = GolfNeuralFactory.create_optimizer(network, config)
        self.scheduler = GolfNeuralFactory.create_scheduler(self.optimizer, config)
        self.loss_function = GolfNeuralFactory.create_loss_function('mse')
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, float]:
        """Train for one epoch."""
        self.network.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.network(batch_features)
            
            # Calculate loss
            if isinstance(predictions, NetworkPrediction):
                loss = self.loss_function(predictions.predictions, batch_targets)
            else:
                loss = self.loss_function(predictions, batch_targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Update learning rate
        if self.scheduler:
            self.scheduler.step()
        
        return {'loss': avg_loss, 'lr': self.optimizer.param_groups[0]['lr']}
    
    def validate(self, val_loader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, float]:
        """Validate the network."""
        self.network.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                predictions = self.network(batch_features)
                
                if isinstance(predictions, NetworkPrediction):
                    loss = self.loss_function(predictions.predictions, batch_targets)
                else:
                    loss = self.loss_function(predictions, batch_targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self, filepath: str, epoch: int, val_loss: float):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'training_history': self.training_history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.training_history = checkpoint.get('training_history', [])
        
        self.logger.info(f"Loaded checkpoint from {filepath}")
        return checkpoint