#!/usr/bin/env python3
"""
Comprehensive test suite for GolfDomainConfig.
Tests all configuration management, validation, and integration functionality.
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
import unittest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict
from datetime import datetime

import sys
# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the config classes directly
from golf_domain.domain_config import (
    GolfDomainConfig,
    ModelConfig,
    EnvironmentConfig,
    DataConfig,
    TrainingConfig,
    BrainIntegrationConfig,
    RiskConfig
)
from domain_registry import DomainConfig, DomainType, DomainStatus


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig dataclass."""
    
    def test_default_initialization(self):
        """Test ModelConfig initializes with correct defaults."""
        config = ModelConfig()
        
        self.assertEqual(config.model_type, 'ensemble')
        self.assertEqual(config.hidden_dims, [512, 256, 128])
        self.assertEqual(config.dropout_rate, 0.3)
        self.assertTrue(config.batch_norm)
        self.assertEqual(config.activation, 'relu')
        self.assertEqual(config.attention_heads, 8)
        self.assertEqual(config.attention_dim, 64)
        self.assertEqual(config.num_models, 3)
        self.assertEqual(config.ensemble_weight_existing, 0.4)
        self.assertEqual(config.ensemble_weight_saraphis, 0.4)
        self.assertEqual(config.ensemble_weight_statistical, 0.2)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.weight_decay, 1e-4)
        self.assertEqual(config.gradient_clip, 1.0)
        self.assertEqual(config.nn_learning_rate, 0.001)
        self.assertEqual(config.nn_batch_size, 32)
        self.assertEqual(config.nn_epochs, 100)
    
    def test_custom_initialization(self):
        """Test ModelConfig with custom values."""
        config = ModelConfig(
            model_type='attention',
            hidden_dims=[1024, 512],
            dropout_rate=0.5,
            learning_rate=0.01
        )
        
        self.assertEqual(config.model_type, 'attention')
        self.assertEqual(config.hidden_dims, [1024, 512])
        self.assertEqual(config.dropout_rate, 0.5)
        self.assertEqual(config.learning_rate, 0.01)
    
    def test_dataclass_conversion(self):
        """Test conversion to/from dict."""
        config = ModelConfig()
        config_dict = asdict(config)
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['model_type'], 'ensemble')
        self.assertEqual(config_dict['hidden_dims'], [512, 256, 128])


class TestEnvironmentConfig(unittest.TestCase):
    """Test EnvironmentConfig dataclass."""
    
    def test_default_initialization(self):
        """Test EnvironmentConfig initializes with correct defaults."""
        config = EnvironmentConfig()
        
        self.assertEqual(config.default_salary_cap, 50000.0)
        self.assertEqual(config.default_lineup_size, 6)
        self.assertEqual(config.position_constraints, {'G': 1})
        self.assertEqual(config.monte_carlo_runs, 1000)
        self.assertEqual(config.variance_factor, 0.15)
        self.assertEqual(config.correlation_factor, 0.3)
        self.assertEqual(config.max_exposure_per_player, 0.3)
        self.assertEqual(config.min_salary_usage, 0.95)
        self.assertTrue(config.preserve_existing_functionality)
        self.assertTrue(config.use_existing_rl_model)
        self.assertEqual(config.existing_model_weight, 0.5)
    
    def test_custom_position_constraints(self):
        """Test custom position constraints."""
        config = EnvironmentConfig()
        config.position_constraints = {'G': 2, 'F': 3}
        
        self.assertEqual(config.position_constraints['G'], 2)
        self.assertEqual(config.position_constraints['F'], 3)


class TestDataConfig(unittest.TestCase):
    """Test DataConfig dataclass."""
    
    def test_default_initialization(self):
        """Test DataConfig initializes with correct defaults."""
        config = DataConfig()
        
        self.assertEqual(config.player_data_path, './data/golf/player_data.csv')
        self.assertEqual(config.historical_results_path, './data/golf/historical_results.csv')
        self.assertEqual(config.course_data_path, './data/golf/course_data.csv')
        self.assertEqual(config.weather_data_path, './data/golf/weather_data.csv')
        self.assertTrue(config.normalize_features)
        self.assertEqual(config.handle_missing_values, 'mean')
        self.assertTrue(config.outlier_removal)
        self.assertTrue(config.feature_engineering)
        self.assertTrue(config.use_cache)
        self.assertEqual(config.cache_directory, './cache/golf/')
        self.assertEqual(config.cache_expiry_hours, 24)
    
    def test_missing_value_strategies(self):
        """Test different missing value handling strategies."""
        for strategy in ['mean', 'median', 'drop']:
            config = DataConfig(handle_missing_values=strategy)
            self.assertEqual(config.handle_missing_values, strategy)


class TestTrainingConfig(unittest.TestCase):
    """Test TrainingConfig dataclass."""
    
    def test_default_initialization(self):
        """Test TrainingConfig initializes with correct defaults."""
        config = TrainingConfig()
        
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.epochs, 100)
        self.assertEqual(config.validation_split, 0.2)
        self.assertEqual(config.early_stopping_patience, 10)
        self.assertEqual(config.optimizer, 'adam')
        self.assertEqual(config.scheduler, 'cosine')
        self.assertEqual(config.warmup_epochs, 5)
        self.assertEqual(config.l2_regularization, 1e-4)
        self.assertEqual(config.dropout_rate, 0.3)
        self.assertEqual(config.gradient_clipping, 1.0)
        self.assertEqual(config.device, 'auto')
        self.assertTrue(config.mixed_precision)
        self.assertEqual(config.num_workers, 4)
        self.assertEqual(config.max_workers, 8)
        self.assertTrue(config.save_best_model)
        self.assertEqual(config.checkpoint_frequency, 10)
        self.assertEqual(config.max_checkpoints, 5)
    
    def test_device_options(self):
        """Test different device options."""
        for device in ['auto', 'cpu', 'cuda']:
            config = TrainingConfig(device=device)
            self.assertEqual(config.device, device)


class TestBrainIntegrationConfig(unittest.TestCase):
    """Test BrainIntegrationConfig dataclass."""
    
    def test_default_initialization(self):
        """Test BrainIntegrationConfig initializes with correct defaults."""
        config = BrainIntegrationConfig()
        
        self.assertEqual(config.domain_name, 'golf_gambling')
        self.assertEqual(config.domain_priority, 3)
        self.assertEqual(config.max_memory_mb, 2048)
        self.assertTrue(config.enable_caching)
        self.assertTrue(config.use_orchestrators)
        self.assertEqual(config.orchestrator_timeout, 30.0)
        self.assertTrue(config.use_proof_strategies)
        self.assertEqual(config.proof_confidence_threshold, 0.7)
        self.assertTrue(config.enable_uncertainty_quantification)
        self.assertTrue(config.enable_reasoning)
        self.assertTrue(config.enable_decision_making)


class TestRiskConfig(unittest.TestCase):
    """Test RiskConfig dataclass."""
    
    def test_default_initialization(self):
        """Test RiskConfig initializes with correct defaults."""
        config = RiskConfig()
        
        self.assertEqual(config.max_portfolio_variance, 0.25)
        self.assertEqual(config.max_correlation_exposure, 0.6)
        self.assertEqual(config.diversification_bonus, 0.05)
        self.assertEqual(config.max_player_ownership, 0.4)
        self.assertEqual(config.injury_risk_multiplier, 0.8)
        self.assertEqual(config.weather_risk_factor, 0.1)
        self.assertTrue(config.use_kelly_criterion)
        self.assertEqual(config.kelly_fraction, 0.25)
        self.assertEqual(config.max_kelly_bet, 0.1)


class TestGolfDomainConfig(unittest.TestCase):
    """Test main GolfDomainConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.json')
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_default_initialization(self):
        """Test GolfDomainConfig initializes with default sub-configs."""
        config = GolfDomainConfig()
        
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.environment, EnvironmentConfig)
        self.assertIsInstance(config.data, DataConfig)
        self.assertIsInstance(config.training, TrainingConfig)
        self.assertIsInstance(config.brain_integration, BrainIntegrationConfig)
        self.assertIsInstance(config.risk, RiskConfig)
    
    def test_validation_ensemble_weights_normalization(self):
        """Test ensemble weights are normalized if they don't sum to 1."""
        config = GolfDomainConfig()
        
        # Set weights that don't sum to 1
        config.model.ensemble_weight_existing = 0.5
        config.model.ensemble_weight_saraphis = 0.5
        config.model.ensemble_weight_statistical = 0.5
        
        config._validate_config()
        
        # Check weights are normalized
        total = (config.model.ensemble_weight_existing + 
                config.model.ensemble_weight_saraphis + 
                config.model.ensemble_weight_statistical)
        self.assertAlmostEqual(total, 1.0, places=7)
    
    def test_validation_batch_size_error(self):
        """Test validation raises error for invalid batch size."""
        config = GolfDomainConfig()
        config.training.batch_size = 0
        
        with self.assertRaises(ValueError) as context:
            config._validate_config()
        self.assertIn("Batch size must be positive", str(context.exception))
    
    def test_validation_epochs_error(self):
        """Test validation raises error for invalid epochs."""
        config = GolfDomainConfig()
        config.training.epochs = -1
        
        with self.assertRaises(ValueError) as context:
            config._validate_config()
        self.assertIn("Epochs must be positive", str(context.exception))
    
    def test_validation_player_ownership_error(self):
        """Test validation raises error for invalid player ownership."""
        config = GolfDomainConfig()
        config.risk.max_player_ownership = 1.5
        
        with self.assertRaises(ValueError) as context:
            config._validate_config()
        self.assertIn("Max player ownership cannot exceed 100%", str(context.exception))
    
    def test_validation_domain_priority_error(self):
        """Test validation raises error for invalid domain priority."""
        config = GolfDomainConfig()
        
        # Test below minimum
        config.brain_integration.domain_priority = 0
        with self.assertRaises(ValueError) as context:
            config._validate_config()
        self.assertIn("Domain priority must be between 1 and 10", str(context.exception))
        
        # Test above maximum
        config.brain_integration.domain_priority = 11
        with self.assertRaises(ValueError) as context:
            config._validate_config()
        self.assertIn("Domain priority must be between 1 and 10", str(context.exception))
    
    def test_save_to_file(self):
        """Test saving configuration to JSON file."""
        config = GolfDomainConfig()
        config.model.learning_rate = 0.002
        config.training.batch_size = 64
        
        config.save_to_file(self.config_file)
        
        self.assertTrue(os.path.exists(self.config_file))
        
        with open(self.config_file, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data['model']['learning_rate'], 0.002)
        self.assertEqual(saved_data['training']['batch_size'], 64)
        self.assertIn('timestamp', saved_data)
    
    def test_load_from_file(self):
        """Test loading configuration from JSON file."""
        # Create test config file
        test_config = {
            'model': {'learning_rate': 0.005, 'dropout_rate': 0.4},
            'training': {'batch_size': 128, 'epochs': 50},
            'risk': {'max_player_ownership': 0.35}
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(test_config, f)
        
        config = GolfDomainConfig(config_path=self.config_file)
        
        self.assertEqual(config.model.learning_rate, 0.005)
        self.assertEqual(config.model.dropout_rate, 0.4)
        self.assertEqual(config.training.batch_size, 128)
        self.assertEqual(config.training.epochs, 50)
        self.assertEqual(config.risk.max_player_ownership, 0.35)
    
    def test_load_from_invalid_file(self):
        """Test loading from non-existent file doesn't crash."""
        config = GolfDomainConfig(config_path='/non/existent/file.json')
        
        # Should initialize with defaults
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.training, TrainingConfig)
    
    def test_update_dataclass(self):
        """Test updating dataclass fields from dictionary."""
        config = GolfDomainConfig()
        
        updates = {
            'learning_rate': 0.01,
            'dropout_rate': 0.5,
            'unknown_field': 'ignored'
        }
        
        config._update_dataclass(config.model, updates)
        
        self.assertEqual(config.model.learning_rate, 0.01)
        self.assertEqual(config.model.dropout_rate, 0.5)
    
    def test_get_saraphis_domain_config(self):
        """Test getting Saraphis-compatible domain configuration."""
        config = GolfDomainConfig()
        config.brain_integration.domain_priority = 5
        config.brain_integration.max_memory_mb = 1024
        
        domain_config = config.get_saraphis_domain_config()
        
        self.assertIsInstance(domain_config, DomainConfig)
        self.assertEqual(domain_config.domain_type, DomainType.SPECIALIZED)
        self.assertEqual(domain_config.priority, 5)
        self.assertEqual(domain_config.max_memory_mb, 1024)
        self.assertIn('model_type', domain_config.model_config)
        self.assertIn('validation_split', domain_config.training_config)
        self.assertIn('salary_cap', domain_config.domain_specific_config)
    
    def test_get_runtime_config(self):
        """Test getting runtime configuration."""
        config = GolfDomainConfig()
        runtime_config = config.get_runtime_config()
        
        self.assertIsInstance(runtime_config, dict)
        self.assertIn('model_config', runtime_config)
        self.assertIn('environment_config', runtime_config)
        self.assertIn('data_config', runtime_config)
        self.assertIn('training_config', runtime_config)
        self.assertIn('brain_config', runtime_config)
        self.assertIn('risk_config', runtime_config)
        
        self.assertEqual(runtime_config['model_config']['model_type'], 'ensemble')
        self.assertEqual(runtime_config['training_config']['batch_size'], 32)
    
    def test_update_runtime_config(self):
        """Test updating configuration at runtime."""
        config = GolfDomainConfig()
        
        updates = {
            'model': {'learning_rate': 0.01},
            'training': {'batch_size': 64, 'epochs': 200},
            'risk': {'max_player_ownership': 0.3}
        }
        
        config.update_runtime_config(updates)
        
        self.assertEqual(config.model.learning_rate, 0.01)
        self.assertEqual(config.training.batch_size, 64)
        self.assertEqual(config.training.epochs, 200)
        self.assertEqual(config.risk.max_player_ownership, 0.3)
    
    def test_update_runtime_config_validation(self):
        """Test runtime update triggers validation."""
        config = GolfDomainConfig()
        
        updates = {
            'training': {'batch_size': -1}
        }
        
        with self.assertRaises(RuntimeError) as context:
            config.update_runtime_config(updates)
        self.assertIn("Runtime configuration update failed", str(context.exception))
    
    def test_update_runtime_config_empty(self):
        """Test updating with empty dict doesn't crash."""
        config = GolfDomainConfig()
        original_lr = config.model.learning_rate
        
        config.update_runtime_config({})
        
        self.assertEqual(config.model.learning_rate, original_lr)
    
    def test_get_data_paths(self):
        """Test getting all data file paths."""
        config = GolfDomainConfig()
        paths = config.get_data_paths()
        
        self.assertIsInstance(paths, dict)
        self.assertIn('player_data', paths)
        self.assertIn('historical_results', paths)
        self.assertIn('course_data', paths)
        self.assertIn('weather_data', paths)
        self.assertIn('cache_directory', paths)
        
        self.assertEqual(paths['player_data'], './data/golf/player_data.csv')
        self.assertEqual(paths['cache_directory'], './cache/golf/')
    
    def test_validate_data_paths(self):
        """Test validating data path existence."""
        config = GolfDomainConfig()
        config.data.cache_directory = self.temp_dir
        
        validation = config.validate_data_paths()
        
        self.assertIsInstance(validation, dict)
        self.assertIn('player_data', validation)
        self.assertIn('cache_directory', validation)
        
        # Cache directory should be created
        self.assertTrue(validation['cache_directory'])
        self.assertTrue(os.path.exists(self.temp_dir))
        
        # Data files likely don't exist
        self.assertFalse(validation['player_data'])
    
    def test_get_model_architecture_config(self):
        """Test getting model architecture configuration."""
        config = GolfDomainConfig()
        
        # Test standard model
        config.model.model_type = 'standard'
        arch_config = config.get_model_architecture_config()
        
        self.assertEqual(arch_config['type'], 'standard')
        self.assertEqual(arch_config['hidden_dims'], [512, 256, 128])
        self.assertIsNone(arch_config['attention_config'])
        self.assertIsNone(arch_config['ensemble_config'])
        
        # Test attention model
        config.model.model_type = 'attention'
        arch_config = config.get_model_architecture_config()
        
        self.assertIsNotNone(arch_config['attention_config'])
        self.assertEqual(arch_config['attention_config']['heads'], 8)
        self.assertEqual(arch_config['attention_config']['dim'], 64)
        
        # Test ensemble model
        config.model.model_type = 'ensemble'
        arch_config = config.get_model_architecture_config()
        
        self.assertIsNotNone(arch_config['attention_config'])
        self.assertIsNotNone(arch_config['ensemble_config'])
        self.assertEqual(arch_config['ensemble_config']['num_models'], 3)
        self.assertIn('weights', arch_config['ensemble_config'])
    
    def test_get_brain_connector_config(self):
        """Test getting brain connector configuration."""
        config = GolfDomainConfig()
        connector_config = config.get_brain_connector_config()
        
        self.assertIsInstance(connector_config, dict)
        self.assertEqual(connector_config['domain_name'], 'golf_gambling')
        self.assertTrue(connector_config['use_orchestrators'])
        self.assertEqual(connector_config['orchestrator_timeout'], 30.0)
        self.assertTrue(connector_config['use_proof_strategies'])
        self.assertEqual(connector_config['proof_confidence_threshold'], 0.7)
        self.assertTrue(connector_config['enable_uncertainty'])
        self.assertTrue(connector_config['enable_reasoning'])
        self.assertTrue(connector_config['enable_decision_making'])
        self.assertEqual(connector_config['max_memory_mb'], 2048)
    
    def test_create_default_config_file(self):
        """Test creating a default configuration file."""
        config = GolfDomainConfig()
        default_file = os.path.join(self.temp_dir, 'default_config.json')
        
        config.create_default_config_file(default_file)
        
        self.assertTrue(os.path.exists(default_file))
        
        with open(default_file, 'r') as f:
            default_data = json.load(f)
        
        self.assertIn('_comment', default_data)
        self.assertIn('model', default_data)
        self.assertIn('environment', default_data)
        self.assertIn('data', default_data)
        self.assertIn('training', default_data)
        self.assertIn('brain_integration', default_data)
        self.assertIn('risk', default_data)
        
        # Check for comments
        self.assertIn('_comment', default_data['model'])
        self.assertEqual(default_data['model']['model_type'], 'ensemble')
    
    def test_string_representations(self):
        """Test string representations of config."""
        config = GolfDomainConfig()
        
        str_repr = str(config)
        self.assertIn('GolfDomainConfig', str_repr)
        self.assertIn('golf_gambling', str_repr)
        self.assertIn('ensemble', str_repr)
        
        detailed_repr = repr(config)
        self.assertIn('GolfDomainConfig', detailed_repr)
        self.assertIn('model_type=ensemble', detailed_repr)
        self.assertIn('domain_priority=3', detailed_repr)
        self.assertIn('use_orchestrators=True', detailed_repr)
    
    @patch('logging.getLogger')
    def test_logging_initialization(self, mock_logger):
        """Test logger is properly initialized."""
        config = GolfDomainConfig()
        mock_logger.assert_called_with('GolfDomainConfig')
    
    def test_error_handling_in_load(self):
        """Test error handling when loading invalid JSON."""
        # Create invalid JSON file
        with open(self.config_file, 'w') as f:
            f.write("invalid json content")
        
        with self.assertRaises(RuntimeError) as context:
            config = GolfDomainConfig(config_path=self.config_file)
        self.assertIn("Configuration loading failed", str(context.exception))
    
    def test_error_handling_in_save(self):
        """Test error handling when saving to invalid path."""
        config = GolfDomainConfig()
        invalid_path = '/invalid/path/that/doesnt/exist/config.json'
        
        with self.assertRaises(RuntimeError) as context:
            config.save_to_file(invalid_path)
        self.assertIn("Configuration saving failed", str(context.exception))
    
    def test_complex_configuration_scenario(self):
        """Test complex configuration scenario with multiple operations."""
        # Initialize with defaults
        config = GolfDomainConfig()
        
        # Update some values
        config.model.learning_rate = 0.01
        config.training.batch_size = 64
        config.risk.max_player_ownership = 0.35
        
        # Save to file
        config.save_to_file(self.config_file)
        
        # Create new config and load
        new_config = GolfDomainConfig(config_path=self.config_file)
        
        # Verify loaded values
        self.assertEqual(new_config.model.learning_rate, 0.01)
        self.assertEqual(new_config.training.batch_size, 64)
        self.assertEqual(new_config.risk.max_player_ownership, 0.35)
        
        # Update runtime config
        new_config.update_runtime_config({
            'model': {'dropout_rate': 0.5},
            'training': {'epochs': 200}
        })
        
        # Get various configs
        saraphis_config = new_config.get_saraphis_domain_config()
        runtime_config = new_config.get_runtime_config()
        brain_config = new_config.get_brain_connector_config()
        
        # Verify configs are generated correctly
        self.assertIsInstance(saraphis_config, DomainConfig)
        self.assertIsInstance(runtime_config, dict)
        self.assertIsInstance(brain_config, dict)
        
        # Verify updated values persist
        self.assertEqual(new_config.model.dropout_rate, 0.5)
        self.assertEqual(new_config.training.epochs, 200)


class TestIntegrationWithDomainRegistry(unittest.TestCase):
    """Test integration with parent DomainConfig from domain_registry."""
    
    def test_saraphis_domain_config_compatibility(self):
        """Test that generated DomainConfig is compatible with domain_registry."""
        golf_config = GolfDomainConfig()
        domain_config = golf_config.get_saraphis_domain_config()
        
        # Test it's the right type
        self.assertIsInstance(domain_config, DomainConfig)
        
        # Test required fields are present
        self.assertEqual(domain_config.domain_type, DomainType.SPECIALIZED)
        self.assertIsInstance(domain_config.description, str)
        self.assertIsInstance(domain_config.priority, int)
        self.assertIsInstance(domain_config.max_memory_mb, int)
        self.assertIsInstance(domain_config.enable_caching, bool)
        
        # Test custom fields are present
        self.assertIn('model_config', dir(domain_config))
        self.assertIn('training_config', dir(domain_config))
        self.assertIn('domain_specific_config', dir(domain_config))
        
        # Test values are correctly transferred
        self.assertEqual(domain_config.priority, golf_config.brain_integration.domain_priority)
        self.assertEqual(domain_config.max_memory_mb, golf_config.brain_integration.max_memory_mb)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_extreme_values(self):
        """Test configuration with extreme values."""
        config = GolfDomainConfig()
        
        # Test very small learning rate
        config.model.learning_rate = 1e-10
        config._validate_config()  # Should not raise
        
        # Test very large batch size
        config.training.batch_size = 10000
        config._validate_config()  # Should not raise
        
        # Test boundary values for risk
        config.risk.max_player_ownership = 1.0
        config._validate_config()  # Should not raise
        
        config.risk.max_player_ownership = 0.0
        config._validate_config()  # Should not raise
    
    def test_empty_hidden_layers(self):
        """Test with empty hidden layers list."""
        config = GolfDomainConfig()
        config.model.hidden_dims = []
        
        # Should still work
        arch_config = config.get_model_architecture_config()
        self.assertEqual(arch_config['hidden_dims'], [])
    
    def test_special_characters_in_paths(self):
        """Test handling special characters in file paths."""
        config = GolfDomainConfig()
        config.data.player_data_path = "./data/golf/player's_data.csv"
        config.data.cache_directory = "./cache/golf & gambling/"
        
        paths = config.get_data_paths()
        self.assertIn("player's_data", paths['player_data'])
        self.assertIn("golf & gambling", paths['cache_directory'])
    
    def test_unicode_in_configuration(self):
        """Test handling unicode characters in configuration."""
        config = GolfDomainConfig()
        config.brain_integration.domain_name = 'golf_⛳_gambling'
        
        connector_config = config.get_brain_connector_config()
        self.assertEqual(connector_config['domain_name'], 'golf_⛳_gambling')
    
    def test_concurrent_modifications(self):
        """Test configuration handles concurrent modifications."""
        config = GolfDomainConfig()
        
        # Simulate concurrent modifications
        config.model.learning_rate = 0.001
        config.training.batch_size = 32
        config.model.learning_rate = 0.002  # Override
        
        self.assertEqual(config.model.learning_rate, 0.002)
        self.assertEqual(config.training.batch_size, 32)


if __name__ == '__main__':
    unittest.main(verbosity=2)