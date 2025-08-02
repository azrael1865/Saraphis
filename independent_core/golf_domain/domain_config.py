#!/usr/bin/env python3
"""
Golf Domain Configuration - Configuration management for golf gambling domain.
Integrates with Saraphis Brain system configuration framework.
"""

import json
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
from datetime import datetime

# Import Saraphis domain configuration
from ..domain_registry import DomainConfig, DomainType, DomainStatus


@dataclass
class ModelConfig:
    """Configuration for neural network models."""
    # Architecture
    model_type: str = 'ensemble'  # 'standard', 'attention', 'ensemble'
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout_rate: float = 0.3
    batch_norm: bool = True
    activation: str = 'relu'
    
    # Attention (if using attention model)
    attention_heads: int = 8
    attention_dim: int = 64
    
    # Ensemble
    num_models: int = 3
    ensemble_weight_existing: float = 0.4
    ensemble_weight_saraphis: float = 0.4
    ensemble_weight_statistical: float = 0.2
    
    # Training
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    
    # Neural Network specific
    nn_learning_rate: float = 0.001
    nn_batch_size: int = 32
    nn_epochs: int = 100


@dataclass
class EnvironmentConfig:
    """Configuration for golf environment."""
    # Tournament settings
    default_salary_cap: float = 50000.0
    default_lineup_size: int = 6
    position_constraints: Dict[str, int] = field(default_factory=lambda: {
        'G': 1,  # Golfer positions if applicable
    })
    
    # Simulation
    monte_carlo_runs: int = 1000
    variance_factor: float = 0.15
    correlation_factor: float = 0.3
    
    # Risk management
    max_exposure_per_player: float = 0.3
    min_salary_usage: float = 0.95
    
    # Existing golf gambler integration
    preserve_existing_functionality: bool = True
    use_existing_rl_model: bool = True
    existing_model_weight: float = 0.5


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    # Data sources
    player_data_path: str = './data/golf/player_data.csv'
    historical_results_path: str = './data/golf/historical_results.csv'
    course_data_path: str = './data/golf/course_data.csv'
    weather_data_path: str = './data/golf/weather_data.csv'
    
    # Processing
    normalize_features: bool = True
    handle_missing_values: str = 'mean'  # 'mean', 'median', 'drop'
    outlier_removal: bool = True
    feature_engineering: bool = True
    
    # Caching
    use_cache: bool = True
    cache_directory: str = './cache/golf/'
    cache_expiry_hours: int = 24


@dataclass 
class TrainingConfig:
    """Configuration for training processes."""
    # General training
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    # Optimization
    optimizer: str = 'adam'
    scheduler: str = 'cosine'
    warmup_epochs: int = 5
    
    # Regularization
    l2_regularization: float = 1e-4
    dropout_rate: float = 0.3
    gradient_clipping: float = 1.0
    
    # Hardware
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    mixed_precision: bool = True
    num_workers: int = 4
    max_workers: int = 8
    
    # Checkpointing
    save_best_model: bool = True
    checkpoint_frequency: int = 10
    max_checkpoints: int = 5


@dataclass
class BrainIntegrationConfig:
    """Configuration for Brain system integration."""
    # Domain registration
    domain_name: str = 'golf_gambling'
    domain_priority: int = 3
    max_memory_mb: int = 2048
    enable_caching: bool = True
    
    # Orchestrator integration
    use_orchestrators: bool = True
    orchestrator_timeout: float = 30.0
    
    # Proof system integration
    use_proof_strategies: bool = True
    proof_confidence_threshold: float = 0.7
    
    # Brain features
    enable_uncertainty_quantification: bool = True
    enable_reasoning: bool = True
    enable_decision_making: bool = True


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    # Portfolio risk
    max_portfolio_variance: float = 0.25
    max_correlation_exposure: float = 0.6
    diversification_bonus: float = 0.05
    
    # Individual player risk
    max_player_ownership: float = 0.4
    injury_risk_multiplier: float = 0.8
    weather_risk_factor: float = 0.1
    
    # Kelly criterion
    use_kelly_criterion: bool = True
    kelly_fraction: float = 0.25
    max_kelly_bet: float = 0.1


class GolfDomainConfig:
    """
    Main configuration class for golf gambling domain.
    Integrates with Saraphis Brain system configuration.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize golf domain configuration."""
        self.logger = logging.getLogger('GolfDomainConfig')
        
        # Sub-configurations
        self.model = ModelConfig()
        self.environment = EnvironmentConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()
        self.brain_integration = BrainIntegrationConfig()
        self.risk = RiskConfig()
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Model validation
        if self.model.ensemble_weight_existing + self.model.ensemble_weight_saraphis + self.model.ensemble_weight_statistical != 1.0:
            total = self.model.ensemble_weight_existing + self.model.ensemble_weight_saraphis + self.model.ensemble_weight_statistical
            self.logger.warning(f"Ensemble weights sum to {total}, normalizing to 1.0")
            self.model.ensemble_weight_existing /= total
            self.model.ensemble_weight_saraphis /= total
            self.model.ensemble_weight_statistical /= total
        
        # Training validation
        if self.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.training.epochs <= 0:
            raise ValueError("Epochs must be positive")
        
        # Risk validation
        if self.risk.max_player_ownership > 1.0:
            raise ValueError("Max player ownership cannot exceed 100%")
        
        # Brain integration validation
        if self.brain_integration.domain_priority < 1 or self.brain_integration.domain_priority > 10:
            raise ValueError("Domain priority must be between 1 and 10")
    
    def load_from_file(self, config_path: str):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Update each sub-configuration
            if 'model' in config_data:
                self._update_dataclass(self.model, config_data['model'])
            
            if 'environment' in config_data:
                self._update_dataclass(self.environment, config_data['environment'])
            
            if 'data' in config_data:
                self._update_dataclass(self.data, config_data['data'])
            
            if 'training' in config_data:
                self._update_dataclass(self.training, config_data['training'])
            
            if 'brain_integration' in config_data:
                self._update_dataclass(self.brain_integration, config_data['brain_integration'])
            
            if 'risk' in config_data:
                self._update_dataclass(self.risk, config_data['risk'])
            
            self.logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise RuntimeError(f"Configuration loading failed: {e}")
    
    def _update_dataclass(self, dataclass_instance: Any, updates: Dict[str, Any]):
        """Update dataclass fields from dictionary."""
        for key, value in updates.items():
            if hasattr(dataclass_instance, key):
                setattr(dataclass_instance, key, value)
            else:
                self.logger.warning(f"Unknown configuration key: {key}")
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file."""
        try:
            config_data = {
                'model': asdict(self.model),
                'environment': asdict(self.environment),
                'data': asdict(self.data),
                'training': asdict(self.training),
                'brain_integration': asdict(self.brain_integration),
                'risk': asdict(self.risk),
                'timestamp': datetime.now().isoformat()
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self.logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise RuntimeError(f"Configuration saving failed: {e}")
    
    def get_saraphis_domain_config(self) -> DomainConfig:
        """Get Saraphis-compatible domain configuration."""
        return DomainConfig(
            domain_type=DomainType.SPECIALIZED,
            description="Golf gambling lineup optimization with neural networks and reinforcement learning",
            priority=self.brain_integration.domain_priority,
            max_memory_mb=self.brain_integration.max_memory_mb,
            enable_caching=self.brain_integration.enable_caching,
            cache_ttl=3600,  # 1 hour
            model_config={
                'model_type': self.model.model_type,
                'learning_rate': self.model.learning_rate,
                'batch_size': self.training.batch_size,
                'epochs': self.training.epochs
            },
            training_config={
                'validation_split': self.training.validation_split,
                'early_stopping': self.training.early_stopping_patience,
                'optimizer': self.training.optimizer,
                'device': self.training.device
            },
            domain_specific_config={
                'salary_cap': self.environment.default_salary_cap,
                'lineup_size': self.environment.default_lineup_size,
                'use_existing_rl': self.environment.use_existing_rl_model,
                'ensemble_weights': {
                    'existing': self.model.ensemble_weight_existing,
                    'saraphis': self.model.ensemble_weight_saraphis,
                    'statistical': self.model.ensemble_weight_statistical
                },
                'risk_parameters': {
                    'max_player_ownership': self.risk.max_player_ownership,
                    'max_portfolio_variance': self.risk.max_portfolio_variance
                }
            }
        )
    
    def get_runtime_config(self) -> Dict[str, Any]:
        """Get runtime configuration for enhanced golf core."""
        return {
            'model_config': asdict(self.model),
            'environment_config': asdict(self.environment),
            'data_config': asdict(self.data),
            'training_config': asdict(self.training),
            'brain_config': asdict(self.brain_integration),
            'risk_config': asdict(self.risk)
        }
    
    def update_runtime_config(self, updates: Dict[str, Any]):
        """Update configuration at runtime."""
        if not updates:
            return
        
        try:
            # Update model config
            if 'model' in updates:
                self._update_dataclass(self.model, updates['model'])
            
            # Update training config
            if 'training' in updates:
                self._update_dataclass(self.training, updates['training'])
            
            # Update environment config
            if 'environment' in updates:
                self._update_dataclass(self.environment, updates['environment'])
            
            # Update risk config
            if 'risk' in updates:
                self._update_dataclass(self.risk, updates['risk'])
            
            # Re-validate after updates
            self._validate_config()
            
            self.logger.info("Runtime configuration updated successfully")
        except Exception as e:
            self.logger.error(f"Failed to update runtime configuration: {e}")
            raise RuntimeError(f"Runtime configuration update failed: {e}")
    
    def get_data_paths(self) -> Dict[str, str]:
        """Get all data file paths."""
        return {
            'player_data': self.data.player_data_path,
            'historical_results': self.data.historical_results_path,
            'course_data': self.data.course_data_path,
            'weather_data': self.data.weather_data_path,
            'cache_directory': self.data.cache_directory
        }
    
    def validate_data_paths(self) -> Dict[str, bool]:
        """Validate that all data paths exist."""
        paths = self.get_data_paths()
        validation = {}
        
        for name, path in paths.items():
            if name == 'cache_directory':
                # Create cache directory if it doesn't exist
                try:
                    os.makedirs(path, exist_ok=True)
                    validation[name] = True
                except Exception:
                    validation[name] = False
            else:
                validation[name] = os.path.exists(path)
        
        return validation
    
    def get_model_architecture_config(self) -> Dict[str, Any]:
        """Get model architecture configuration."""
        return {
            'type': self.model.model_type,
            'hidden_dims': self.model.hidden_dims,
            'dropout_rate': self.model.dropout_rate,
            'batch_norm': self.model.batch_norm,
            'activation': self.model.activation,
            'attention_config': {
                'heads': self.model.attention_heads,
                'dim': self.model.attention_dim
            } if self.model.model_type in ['attention', 'ensemble'] else None,
            'ensemble_config': {
                'num_models': self.model.num_models,
                'weights': {
                    'existing': self.model.ensemble_weight_existing,
                    'saraphis': self.model.ensemble_weight_saraphis,
                    'statistical': self.model.ensemble_weight_statistical
                }
            } if self.model.model_type == 'ensemble' else None
        }
    
    def get_brain_connector_config(self) -> Dict[str, Any]:
        """Get configuration for brain connector."""
        return {
            'domain_name': self.brain_integration.domain_name,
            'use_orchestrators': self.brain_integration.use_orchestrators,
            'orchestrator_timeout': self.brain_integration.orchestrator_timeout,
            'use_proof_strategies': self.brain_integration.use_proof_strategies,
            'proof_confidence_threshold': self.brain_integration.proof_confidence_threshold,
            'enable_uncertainty': self.brain_integration.enable_uncertainty_quantification,
            'enable_reasoning': self.brain_integration.enable_reasoning,
            'enable_decision_making': self.brain_integration.enable_decision_making,
            'max_memory_mb': self.brain_integration.max_memory_mb
        }
    
    def create_default_config_file(self, output_path: str):
        """Create a default configuration file with comments."""
        config_template = {
            "_comment": "Golf Domain Configuration for Saraphis Brain System",
            "model": {
                "_comment": "Neural network model configuration",
                "model_type": "ensemble",
                "hidden_dims": [512, 256, 128],
                "dropout_rate": 0.3,
                "learning_rate": 0.001,
                "ensemble_weight_existing": 0.4,
                "ensemble_weight_saraphis": 0.4,
                "ensemble_weight_statistical": 0.2
            },
            "environment": {
                "_comment": "Golf tournament environment settings",
                "default_salary_cap": 50000.0,
                "default_lineup_size": 6,
                "preserve_existing_functionality": True,
                "use_existing_rl_model": True
            },
            "data": {
                "_comment": "Data loading and processing configuration",
                "player_data_path": "./data/golf/player_data.csv",
                "normalize_features": True,
                "use_cache": True,
                "cache_directory": "./cache/golf/"
            },
            "training": {
                "_comment": "Training process configuration",
                "batch_size": 32,
                "epochs": 100,
                "validation_split": 0.2,
                "device": "auto"
            },
            "brain_integration": {
                "_comment": "Saraphis Brain system integration",
                "domain_name": "golf_gambling",
                "domain_priority": 3,
                "use_orchestrators": True,
                "use_proof_strategies": True
            },
            "risk": {
                "_comment": "Risk management parameters",
                "max_player_ownership": 0.4,
                "max_portfolio_variance": 0.25,
                "use_kelly_criterion": True
            }
        }
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(config_template, f, indent=2)
            
            self.logger.info(f"Created default configuration file at {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to create default configuration: {e}")
            raise RuntimeError(f"Default configuration creation failed: {e}")
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"GolfDomainConfig(domain={self.brain_integration.domain_name}, model={self.model.model_type})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"GolfDomainConfig("
                f"model_type={self.model.model_type}, "
                f"domain_priority={self.brain_integration.domain_priority}, "
                f"use_orchestrators={self.brain_integration.use_orchestrators})")