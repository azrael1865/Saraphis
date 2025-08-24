"""
Comprehensive unit tests for Golf Domain Configuration.
Tests configuration loading, validation, and integration with Brain system.
"""

import unittest
import json
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

# Import the module to test
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from golf_domain.domain_config import GolfDomainConfig


class TestModelConfig(unittest.TestCase):
    """Test suite for ModelConfig dataclass."""
    
    def test_default_initialization(self):
        """Test default ModelConfig initialization."""
        config = ModelConfig()
        
        self.assertEqual(config.model_type, 'ensemble')
        self.assertEqual(config.hidden_dims, [512, 256, 128])
        self.assertEqual(config.dropout_rate, 0.3)
        self.assertTrue(config.batch_norm)
        self.assertEqual(config.activation, 'relu')
        self.assertEqual(config.attention_heads, 8)
        self.assertEqual(config.num_models, 3)
    
    def test_custom_initialization(self):
        """Test ModelConfig with custom values."""
        config = ModelConfig(
            model_type='attention',
            hidden_dims=[256, 128],
            dropout_rate=0.5,
            learning_rate=0.002
        )
        
        self.assertEqual(config.model_type, 'attention')
        self.assertEqual(config.hidden_dims, [256, 128])
        self.assertEqual(config.dropout_rate, 0.5)
        self.assertEqual(config.learning_rate, 0.002)
    
    def test_ensemble_weights(self):
        """Test ensemble weight configuration."""
        config = ModelConfig()
        
        total_weight = (config.ensemble_weight_existing + 
                       config.ensemble_weight_saraphis + 
                       config.ensemble_weight_statistical)
        
        self.assertAlmostEqual(total_weight, 1.0, places=7)


class TestEnvironmentConfig(unittest.TestCase):
    """Test suite for EnvironmentConfig dataclass."""
    
    def test_default_initialization(self):
        """Test default EnvironmentConfig initialization."""
        config = EnvironmentConfig()
        
        self.assertEqual(config.default_salary_cap, 50000.0)
        self.assertEqual(config.default_lineup_size, 6)
        self.assertEqual(config.monte_carlo_runs, 1000)
        self.assertTrue(config.preserve_existing_functionality)
        self.assertTrue(config.use_existing_rl_model)
    
    def test_position_constraints(self):
        """Test position constraints configuration."""
        config = EnvironmentConfig()
        
        self.assertIsInstance(config.position_constraints, dict)
        self.assertEqual(config.position_constraints.get('G'), 1)
    
    def test_risk_parameters(self):
        """Test risk management parameters."""
        config = EnvironmentConfig()
        
        self.assertEqual(config.max_exposure_per_player, 0.3)
        self.assertEqual(config.min_salary_usage, 0.95)
        self.assertEqual(config.variance_factor, 0.15)


class TestDataConfig(unittest.TestCase):
    """Test suite for DataConfig dataclass."""
    
    def test_default_paths(self):
        """Test default data paths."""
        config = DataConfig()
        
        self.assertEqual(config.player_data_path, './data/golf/player_data.csv')
        self.assertEqual(config.cache_directory, './cache/golf/')
        self.assertTrue(config.normalize_features)
        self.assertTrue(config.use_cache)
    
    def test_processing_options(self):
        """Test data processing configuration."""
        config = DataConfig()
        
        self.assertEqual(config.handle_missing_values, 'mean')
        self.assertTrue(config.outlier_removal)
        self.assertTrue(config.feature_engineering)
        self.assertEqual(config.cache_expiry_hours, 24)


class TestTrainingConfig(unittest.TestCase):
    """Test suite for TrainingConfig dataclass."""
    
    def test_training_parameters(self):
        """Test training configuration parameters."""
        config = TrainingConfig()
        
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.epochs, 100)
        self.assertEqual(config.validation_split, 0.2)
        self.assertEqual(config.early_stopping_patience, 10)
    
    def test_optimization_settings(self):
        """Test optimization settings."""
        config = TrainingConfig()
        
        self.assertEqual(config.optimizer, 'adam')
        self.assertEqual(config.scheduler, 'cosine')
        self.assertEqual(config.warmup_epochs, 5)
        self.assertEqual(config.gradient_clipping, 1.0)
    
    def test_hardware_settings(self):
        """Test hardware configuration."""
        config = TrainingConfig()
        
        self.assertEqual(config.device, 'auto')
        self.assertTrue(config.mixed_precision)
        self.assertEqual(config.num_workers, 4)


class TestBrainIntegrationConfig(unittest.TestCase):
    """Test suite for BrainIntegrationConfig dataclass."""
    
    def test_domain_registration(self):
        """Test domain registration settings."""
        config = BrainIntegrationConfig()
        
        self.assertEqual(config.domain_name, 'golf_gambling')
        self.assertEqual(config.domain_priority, 3)
        self.assertEqual(config.max_memory_mb, 2048)
        self.assertTrue(config.enable_caching)
    
    def test_orchestrator_settings(self):
        """Test orchestrator integration settings."""
        config = BrainIntegrationConfig()
        
        self.assertTrue(config.use_orchestrators)
        self.assertEqual(config.orchestrator_timeout, 30.0)
        self.assertTrue(config.use_proof_strategies)
        self.assertEqual(config.proof_confidence_threshold, 0.7)
    
    def test_brain_features(self):
        """Test brain feature flags."""
        config = BrainIntegrationConfig()
        
        self.assertTrue(config.enable_uncertainty_quantification)
        self.assertTrue(config.enable_reasoning)
        self.assertTrue(config.enable_decision_making)


class TestRiskConfig(unittest.TestCase):
    """Test suite for RiskConfig dataclass."""
    
    def test_portfolio_risk(self):
        """Test portfolio risk parameters."""
        config = RiskConfig()
        
        self.assertEqual(config.max_portfolio_variance, 0.25)
        self.assertEqual(config.max_correlation_exposure, 0.6)
        self.assertEqual(config.diversification_bonus, 0.05)
    
    def test_player_risk(self):
        """Test individual player risk parameters."""
        config = RiskConfig()
        
        self.assertEqual(config.max_player_ownership, 0.4)
        self.assertEqual(config.injury_risk_multiplier, 0.8)
        self.assertEqual(config.weather_risk_factor, 0.1)
    
    def test_kelly_criterion(self):
        """Test Kelly criterion settings."""
        config = RiskConfig()
        
        self.assertTrue(config.use_kelly_criterion)
        self.assertEqual(config.kelly_fraction, 0.25)
        self.assertEqual(config.max_kelly_bet, 0.1)


class TestGolfDomainConfig(unittest.TestCase):
    """Test suite for main GolfDomainConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.json')
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test GolfDomainConfig initialization."""
        config = GolfDomainConfig()
        
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.environment, EnvironmentConfig)
        self.assertIsInstance(config.data, DataConfig)
        self.assertIsInstance(config.training, TrainingConfig)
        self.assertIsInstance(config.brain_integration, BrainIntegrationConfig)
        self.assertIsInstance(config.risk, RiskConfig)
    
    def test_validation_ensemble_weights(self):
        """Test ensemble weight validation and normalization."""
        config = GolfDomainConfig()
        
        # Modify weights to not sum to 1
        config.model.ensemble_weight_existing = 0.5
        config.model.ensemble_weight_saraphis = 0.5
        config.model.ensemble_weight_statistical = 0.5
        
        # Validate should normalize them
        config._validate_config()
        
        total = (config.model.ensemble_weight_existing + 
                config.model.ensemble_weight_saraphis + 
                config.model.ensemble_weight_statistical)
        
        self.assertAlmostEqual(total, 1.0, places=7)
    
    def test_validation_errors(self):
        """Test validation error handling."""
        config = GolfDomainConfig()
        
        # Test invalid batch size
        config.training.batch_size = -1
        with self.assertRaises(ValueError):
            config._validate_config()
        
        config.training.batch_size = 32
        
        # Test invalid epochs
        config.training.epochs = 0
        with self.assertRaises(ValueError):
            config._validate_config()
        
        config.training.epochs = 100
        
        # Test invalid player ownership
        config.risk.max_player_ownership = 1.5
        with self.assertRaises(ValueError):
            config._validate_config()
        
        config.risk.max_player_ownership = 0.4
        
        # Test invalid domain priority
        config.brain_integration.domain_priority = 11
        with self.assertRaises(ValueError):
            config._validate_config()
    
    def test_save_to_file(self):
        """Test saving configuration to file."""
        config = GolfDomainConfig()
        config.save_to_file(self.config_file)
        
        self.assertTrue(os.path.exists(self.config_file))
        
        # Verify file contents
        with open(self.config_file, 'r') as f:
            data = json.load(f)
        
        self.assertIn('model', data)
        self.assertIn('environment', data)
        self.assertIn('training', data)
        self.assertIn('timestamp', data)
    
    def test_load_from_file(self):
        """Test loading configuration from file."""
        # Create test config file
        test_config = {
            'model': {
                'model_type': 'standard',
                'learning_rate': 0.005,
                'dropout_rate': 0.4
            },
            'training': {
                'batch_size': 64,
                'epochs': 200
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(test_config, f)
        
        # Load config
        config = GolfDomainConfig(self.config_file)
        
        self.assertEqual(config.model.model_type, 'standard')
        self.assertEqual(config.model.learning_rate, 0.005)
        self.assertEqual(config.model.dropout_rate, 0.4)
        self.assertEqual(config.training.batch_size, 64)
        self.assertEqual(config.training.epochs, 200)
    
    def test_load_invalid_file(self):
        """Test loading from invalid file."""
        with self.assertRaises(RuntimeError):
            GolfDomainConfig('/nonexistent/file.json')
    
    def test_update_runtime_config(self):
        """Test runtime configuration updates."""
        config = GolfDomainConfig()
        
        updates = {
            'model': {
                'learning_rate': 0.002,
                'dropout_rate': 0.5
            },
            'training': {
                'batch_size': 64
            }
        }
        
        config.update_runtime_config(updates)
        
        self.assertEqual(config.model.learning_rate, 0.002)
        self.assertEqual(config.model.dropout_rate, 0.5)
        self.assertEqual(config.training.batch_size, 64)
    
    def test_get_data_paths(self):
        """Test getting data paths."""
        config = GolfDomainConfig()
        paths = config.get_data_paths()
        
        self.assertIn('player_data', paths)
        self.assertIn('historical_results', paths)
        self.assertIn('course_data', paths)
        self.assertIn('weather_data', paths)
        self.assertIn('cache_directory', paths)
    
    def test_validate_data_paths(self):
        """Test data path validation."""
        config = GolfDomainConfig()
        
        # Set cache directory to temp dir
        config.data.cache_directory = os.path.join(self.temp_dir, 'cache')
        
        validation = config.validate_data_paths()
        
        # Cache directory should be created
        self.assertTrue(validation['cache_directory'])
        self.assertTrue(os.path.exists(config.data.cache_directory))
        
        # Other paths probably don't exist
        self.assertFalse(validation['player_data'])
    
    def test_get_model_architecture_config(self):
        """Test getting model architecture configuration."""
        config = GolfDomainConfig()
        arch_config = config.get_model_architecture_config()
        
        self.assertEqual(arch_config['type'], 'ensemble')
        self.assertIn('hidden_dims', arch_config)
        self.assertIn('ensemble_config', arch_config)
        self.assertIn('weights', arch_config['ensemble_config'])
    
    def test_get_brain_connector_config(self):
        """Test getting brain connector configuration."""
        config = GolfDomainConfig()
        brain_config = config.get_brain_connector_config()
        
        self.assertEqual(brain_config['domain_name'], 'golf_gambling')
        self.assertTrue(brain_config['use_orchestrators'])
        self.assertTrue(brain_config['use_proof_strategies'])
        self.assertEqual(brain_config['max_memory_mb'], 2048)
    
    @patch('golf_domain.domain_config.DomainConfig')
    def test_get_saraphis_domain_config(self, mock_domain_config):
        """Test getting Saraphis-compatible domain configuration."""
        config = GolfDomainConfig()
        
        # Mock the DomainConfig class
        mock_instance = Mock()
        mock_domain_config.return_value = mock_instance
        
        result = config.get_saraphis_domain_config()
        
        # Verify DomainConfig was called with correct parameters
        mock_domain_config.assert_called_once()
        call_args = mock_domain_config.call_args[1]
        
        self.assertEqual(call_args['priority'], 3)
        self.assertEqual(call_args['max_memory_mb'], 2048)
        self.assertTrue(call_args['enable_caching'])
    
    def test_get_runtime_config(self):
        """Test getting runtime configuration."""
        config = GolfDomainConfig()
        runtime_config = config.get_runtime_config()
        
        self.assertIn('model_config', runtime_config)
        self.assertIn('environment_config', runtime_config)
        self.assertIn('training_config', runtime_config)
        self.assertIn('brain_config', runtime_config)
        self.assertIn('risk_config', runtime_config)
    
    def test_create_default_config_file(self):
        """Test creating default configuration file."""
        config = GolfDomainConfig()
        default_file = os.path.join(self.temp_dir, 'default.json')
        
        config.create_default_config_file(default_file)
        
        self.assertTrue(os.path.exists(default_file))
        
        # Verify contents
        with open(default_file, 'r') as f:
            data = json.load(f)
        
        self.assertIn('_comment', data)
        self.assertIn('model', data)
        self.assertIn('environment', data)
        self.assertIn('brain_integration', data)
    
    def test_string_representations(self):
        """Test string representations."""
        config = GolfDomainConfig()
        
        str_repr = str(config)
        self.assertIn('GolfDomainConfig', str_repr)
        self.assertIn('golf_gambling', str_repr)
        
        repr_str = repr(config)
        self.assertIn('model_type=ensemble', repr_str)
        self.assertIn('domain_priority=3', repr_str)
    
    def test_update_dataclass(self):
        """Test updating dataclass fields."""
        config = GolfDomainConfig()
        
        updates = {
            'learning_rate': 0.003,
            'dropout_rate': 0.6,
            'unknown_field': 'should_be_ignored'
        }
        
        with self.assertLogs('GolfDomainConfig', level='WARNING') as cm:
            config._update_dataclass(config.model, updates)
        
        self.assertEqual(config.model.learning_rate, 0.003)
        self.assertEqual(config.model.dropout_rate, 0.6)
        
        # Check warning was logged for unknown field
        self.assertTrue(any('unknown_field' in log for log in cm.output))
    
    def test_config_file_with_missing_sections(self):
        """Test loading config file with missing sections."""
        # Create partial config file
        partial_config = {
            'model': {
                'model_type': 'attention'
            }
            # Missing other sections
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(partial_config, f)
        
        # Should load without error
        config = GolfDomainConfig(self.config_file)
        
        self.assertEqual(config.model.model_type, 'attention')
        # Other configs should have defaults
        self.assertEqual(config.training.batch_size, 32)
    
    def test_empty_update_runtime_config(self):
        """Test update_runtime_config with empty updates."""
        config = GolfDomainConfig()
        original_lr = config.model.learning_rate
        
        # Should handle empty updates gracefully
        config.update_runtime_config({})
        config.update_runtime_config(None)
        
        self.assertEqual(config.model.learning_rate, original_lr)


class TestIntegration(unittest.TestCase):
    """Integration tests for domain configuration system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_config_workflow(self):
        """Test complete configuration workflow."""
        # Create config
        config = GolfDomainConfig()
        
        # Modify some values
        config.model.learning_rate = 0.002
        config.training.epochs = 150
        config.risk.max_player_ownership = 0.35
        
        # Save to file
        config_file = os.path.join(self.temp_dir, 'workflow.json')
        config.save_to_file(config_file)
        
        # Load in new instance
        new_config = GolfDomainConfig(config_file)
        
        # Verify values preserved
        self.assertEqual(new_config.model.learning_rate, 0.002)
        self.assertEqual(new_config.training.epochs, 150)
        self.assertEqual(new_config.risk.max_player_ownership, 0.35)
    
    def test_config_validation_chain(self):
        """Test configuration validation chain."""
        config = GolfDomainConfig()
        
        # Set invalid ensemble weights
        config.model.ensemble_weight_existing = 2.0
        config.model.ensemble_weight_saraphis = 1.0
        config.model.ensemble_weight_statistical = 0.5
        
        # Validation should normalize
        config._validate_config()
        
        # Verify normalization
        total = (config.model.ensemble_weight_existing + 
                config.model.ensemble_weight_saraphis + 
                config.model.ensemble_weight_statistical)
        self.assertAlmostEqual(total, 1.0, places=7)
        
        # Now update at runtime with valid values
        updates = {
            'model': {
                'ensemble_weight_existing': 0.33,
                'ensemble_weight_saraphis': 0.33,
                'ensemble_weight_statistical': 0.34
            }
        }
        
        config.update_runtime_config(updates)
        
        # Verify update succeeded
        self.assertAlmostEqual(config.model.ensemble_weight_existing, 0.33, places=7)


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)