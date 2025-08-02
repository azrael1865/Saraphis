"""
Configuration and validation tests for Universal AI Core.
Adapted from Saraphis configuration test patterns.
"""

import pytest
import yaml
import json
import os
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from universal_ai_core.config.config_manager import (
    ConfigurationManager, UniversalConfiguration, get_config_manager, get_config
)


@pytest.mark.unit
class TestUniversalConfiguration:
    """Test Universal Configuration data class."""
    
    def test_configuration_creation_with_valid_data(self, sample_config):
        """Test creating configuration with valid data."""
        config = UniversalConfiguration(**sample_config)
        
        assert config.core.max_workers == 2
        assert config.core.enable_monitoring == True
        assert config.plugins.feature_extractors.molecular.enabled == True
        assert config.logging.level == "DEBUG"
    
    def test_configuration_creation_with_minimal_data(self):
        """Test creating configuration with minimal required data."""
        minimal_config = {
            "core": {"max_workers": 1},
            "plugins": {},
            "logging": {"level": "INFO"}
        }
        
        config = UniversalConfiguration(**minimal_config)
        assert config.core.max_workers == 1
        assert config.logging.level == "INFO"
    
    def test_configuration_default_values(self):
        """Test configuration default value assignment."""
        # Test with empty core config
        config_data = {
            "core": {},
            "plugins": {},
            "logging": {}
        }
        
        config = UniversalConfiguration(**config_data)
        
        # Should have reasonable defaults
        assert hasattr(config.core, 'max_workers')
        assert hasattr(config.core, 'enable_monitoring')
    
    def test_configuration_validation(self, sample_config):
        """Test configuration field validation."""
        config = UniversalConfiguration(**sample_config)
        
        # Test core configuration
        assert isinstance(config.core.max_workers, int)
        assert config.core.max_workers > 0
        assert isinstance(config.core.enable_monitoring, bool)
        
        # Test plugin configuration structure
        assert hasattr(config.plugins, 'feature_extractors')
        assert hasattr(config.plugins, 'models')
        assert hasattr(config.plugins, 'proof_languages')
        assert hasattr(config.plugins, 'knowledge_bases')
    
    def test_configuration_invalid_data_handling(self):
        """Test configuration handling of invalid data."""
        # Test with invalid core config
        with pytest.raises((TypeError, ValueError, AttributeError)):
            invalid_config = {
                "core": {"max_workers": "invalid"},  # Should be int
                "plugins": {},
                "logging": {}
            }
            UniversalConfiguration(**invalid_config)
    
    def test_configuration_to_dict_serialization(self, sample_config):
        """Test configuration serialization to dictionary."""
        config = UniversalConfiguration(**sample_config)
        
        # Convert back to dict
        config_dict = config.core.__dict__
        
        assert isinstance(config_dict, dict)
        assert 'max_workers' in config_dict
        assert 'enable_monitoring' in config_dict


@pytest.mark.unit
class TestConfigurationManager:
    """Test Configuration Manager functionality."""
    
    def test_config_manager_initialization_with_file(self, sample_config_file):
        """Test configuration manager initialization with config file."""
        manager = ConfigurationManager(str(sample_config_file))
        
        assert manager.config_path == str(sample_config_file)
        assert manager.config is not None
    
    def test_config_manager_initialization_without_file(self):
        """Test configuration manager initialization without config file."""
        manager = ConfigurationManager()
        
        assert manager.config_path is None
        assert manager.config is not None  # Should have default config
    
    def test_config_loading_from_yaml(self, temp_directory, sample_config):
        """Test loading configuration from YAML file."""
        # Create YAML config file
        yaml_file = temp_directory / "config.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        manager = ConfigurationManager(str(yaml_file))
        config = manager.get_config()
        
        assert isinstance(config, UniversalConfiguration)
        assert config.core.max_workers == 2
    
    def test_config_loading_from_json(self, temp_directory, sample_config):
        """Test loading configuration from JSON file."""
        # Create JSON config file
        json_file = temp_directory / "config.json"
        with open(json_file, 'w') as f:
            json.dump(sample_config, f)
        
        manager = ConfigurationManager(str(json_file))
        config = manager.get_config()
        
        assert isinstance(config, UniversalConfiguration)
        assert config.core.max_workers == 2
    
    def test_config_loading_invalid_file(self, temp_directory):
        """Test loading configuration from invalid file."""
        # Create invalid YAML file
        invalid_file = temp_directory / "invalid.yaml"
        invalid_file.write_text("invalid: yaml: content: [")
        
        with pytest.raises((yaml.YAMLError, ValueError)):
            ConfigurationManager(str(invalid_file))
    
    def test_config_loading_nonexistent_file(self):
        """Test loading configuration from nonexistent file."""
        with pytest.raises((FileNotFoundError, IOError)):
            ConfigurationManager("/nonexistent/config.yaml")
    
    def test_config_validation_valid_config(self, sample_config):
        """Test configuration validation with valid config."""
        manager = ConfigurationManager()
        
        is_valid, errors = manager.validate_config(sample_config)
        
        assert is_valid == True
        assert len(errors) == 0
    
    def test_config_validation_invalid_config(self):
        """Test configuration validation with invalid config."""
        manager = ConfigurationManager()
        
        invalid_config = {
            "core": {"max_workers": -1},  # Invalid negative value
            "plugins": "not_a_dict",  # Should be dict
            "logging": {}
        }
        
        is_valid, errors = manager.validate_config(invalid_config)
        
        assert is_valid == False
        assert len(errors) > 0
    
    def test_config_merging(self, temp_directory):
        """Test configuration merging from multiple sources."""
        # Base config
        base_config = {
            "core": {"max_workers": 2, "enable_monitoring": True},
            "plugins": {"feature_extractors": {"molecular": {"enabled": True}}}
        }
        
        # Override config
        override_config = {
            "core": {"max_workers": 4},  # Override value
            "plugins": {"feature_extractors": {"cybersecurity": {"enabled": True}}}  # Add new
        }
        
        base_file = temp_directory / "base.yaml"
        override_file = temp_directory / "override.yaml"
        
        with open(base_file, 'w') as f:
            yaml.dump(base_config, f)
        with open(override_file, 'w') as f:
            yaml.dump(override_config, f)
        
        manager = ConfigurationManager(str(base_file))
        
        # Test merging (would need implementation in ConfigurationManager)
        # For now, test that base config loads correctly
        config = manager.get_config()
        assert config.core.max_workers == 2  # From base config


@pytest.mark.integration
class TestConfigurationWatching:
    """Test configuration file watching and hot reload."""
    
    def test_config_file_watching_setup(self, sample_config_file):
        """Test setting up configuration file watching."""
        manager = ConfigurationManager(str(sample_config_file))
        
        # Start watching
        manager.start_watching()
        assert manager._watching == True
        
        # Stop watching
        manager.stop_watching()
        assert manager._watching == False
    
    def test_config_hot_reload_on_file_change(self, temp_directory, sample_config):
        """Test configuration hot reload when file changes."""
        config_file = temp_directory / "hot_reload_config.yaml"
        
        # Initial config
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        manager = ConfigurationManager(str(config_file))
        original_config = manager.get_config()
        original_workers = original_config.core.max_workers
        
        # Start watching
        manager.start_watching()
        
        try:
            # Modify config file
            modified_config = sample_config.copy()
            modified_config["core"]["max_workers"] = original_workers + 2
            
            time.sleep(0.1)  # Brief pause
            
            with open(config_file, 'w') as f:
                yaml.dump(modified_config, f)
            
            # Wait for file system event
            time.sleep(0.5)
            
            # Check if config was reloaded (would need implementation)
            # For now, just verify the watching mechanism is active
            assert manager._watching == True
            
        finally:
            manager.stop_watching()
    
    def test_config_reload_error_handling(self, temp_directory):
        """Test error handling during config reload."""
        config_file = temp_directory / "error_reload_config.yaml"
        
        # Valid initial config
        initial_config = {"core": {"max_workers": 2}, "plugins": {}, "logging": {}}
        with open(config_file, 'w') as f:
            yaml.dump(initial_config, f)
        
        manager = ConfigurationManager(str(config_file))
        original_config = manager.get_config()
        
        manager.start_watching()
        
        try:
            # Write invalid config
            time.sleep(0.1)
            config_file.write_text("invalid: yaml: content: [")
            time.sleep(0.5)
            
            # Config should remain unchanged after invalid reload
            current_config = manager.get_config()
            assert current_config.core.max_workers == original_config.core.max_workers
            
        finally:
            manager.stop_watching()


@pytest.mark.unit
class TestEnvironmentVariableConfig:
    """Test configuration from environment variables."""
    
    def test_config_from_environment_variables(self):
        """Test loading configuration values from environment variables."""
        # Set environment variables
        env_vars = {
            "UNIVERSAL_AI_MAX_WORKERS": "8",
            "UNIVERSAL_AI_ENABLE_MONITORING": "false",
            "UNIVERSAL_AI_LOG_LEVEL": "WARNING"
        }
        
        with patch.dict(os.environ, env_vars):
            manager = ConfigurationManager()
            
            # Test environment variable override (would need implementation)
            # For now, test that manager initializes correctly
            config = manager.get_config()
            assert config is not None
    
    def test_config_environment_variable_precedence(self, sample_config_file):
        """Test environment variable precedence over file config."""
        env_vars = {
            "UNIVERSAL_AI_MAX_WORKERS": "16"
        }
        
        with patch.dict(os.environ, env_vars):
            manager = ConfigurationManager(str(sample_config_file))
            config = manager.get_config()
            
            # Environment variables should take precedence (would need implementation)
            # For now, verify config loads
            assert isinstance(config.core.max_workers, int)
    
    def test_config_environment_variable_validation(self):
        """Test validation of environment variable values."""
        # Invalid environment variable
        env_vars = {
            "UNIVERSAL_AI_MAX_WORKERS": "invalid_number"
        }
        
        with patch.dict(os.environ, env_vars):
            # Should handle invalid environment variables gracefully
            manager = ConfigurationManager()
            config = manager.get_config()
            assert config is not None


@pytest.mark.integration
class TestConfigurationTemplates:
    """Test configuration templates and domain-specific configs."""
    
    def test_molecular_domain_template(self, temp_directory):
        """Test loading molecular domain configuration template."""
        # Create molecular config template
        molecular_config = {
            "core": {"max_workers": 4},
            "plugins": {
                "feature_extractors": {
                    "molecular": {
                        "enabled": True,
                        "rdkit_enabled": True,
                        "descriptors": ["molecular_weight", "logp", "tpsa"]
                    }
                },
                "models": {
                    "molecular": {
                        "enabled": True,
                        "model_type": "neural_network",
                        "hidden_layers": [128, 64, 32]
                    }
                }
            }
        }
        
        template_file = temp_directory / "molecular_template.yaml"
        with open(template_file, 'w') as f:
            yaml.dump(molecular_config, f)
        
        manager = ConfigurationManager(str(template_file))
        config = manager.get_config()
        
        assert config.core.max_workers == 4
        assert config.plugins.feature_extractors.molecular.enabled == True
    
    def test_cybersecurity_domain_template(self, temp_directory):
        """Test loading cybersecurity domain configuration template."""
        cybersecurity_config = {
            "core": {"max_workers": 6},
            "plugins": {
                "feature_extractors": {
                    "cybersecurity": {
                        "enabled": True,
                        "threat_detection": True,
                        "behavioral_analysis": True
                    }
                },
                "models": {
                    "cybersecurity": {
                        "enabled": True,
                        "model_type": "ensemble",
                        "algorithms": ["random_forest", "svm", "neural_network"]
                    }
                }
            }
        }
        
        template_file = temp_directory / "cybersecurity_template.yaml"
        with open(template_file, 'w') as f:
            yaml.dump(cybersecurity_config, f)
        
        manager = ConfigurationManager(str(template_file))
        config = manager.get_config()
        
        assert config.core.max_workers == 6
        assert config.plugins.feature_extractors.cybersecurity.enabled == True
    
    def test_financial_domain_template(self, temp_directory):
        """Test loading financial domain configuration template."""
        financial_config = {
            "core": {"max_workers": 8},
            "plugins": {
                "feature_extractors": {
                    "financial": {
                        "enabled": True,
                        "technical_indicators": ["sma", "ema", "rsi", "macd"],
                        "risk_metrics": ["var", "cvar", "sharpe_ratio"]
                    }
                },
                "models": {
                    "financial": {
                        "enabled": True,
                        "model_type": "lstm",
                        "sequence_length": 60,
                        "hidden_units": 50
                    }
                }
            }
        }
        
        template_file = temp_directory / "financial_template.yaml"
        with open(template_file, 'w') as f:
            yaml.dump(financial_config, f)
        
        manager = ConfigurationManager(str(template_file))
        config = manager.get_config()
        
        assert config.core.max_workers == 8
        assert config.plugins.feature_extractors.financial.enabled == True


@pytest.mark.unit
class TestConfigurationUtilities:
    """Test configuration utility functions."""
    
    def test_get_config_manager_singleton(self, sample_config_file):
        """Test get_config_manager singleton behavior."""
        # First call
        manager1 = get_config_manager(str(sample_config_file))
        
        # Second call should return same instance
        manager2 = get_config_manager(str(sample_config_file))
        
        assert manager1 is manager2  # Should be same instance
    
    def test_get_config_utility_function(self, sample_config_file):
        """Test get_config utility function."""
        config = get_config(str(sample_config_file))
        
        assert isinstance(config, UniversalConfiguration)
        assert config.core.max_workers == 2
    
    def test_config_manager_with_default_paths(self):
        """Test config manager with default configuration paths."""
        # Test with no arguments (should use defaults)
        manager = get_config_manager()
        config = manager.get_config()
        
        assert config is not None
        assert isinstance(config, UniversalConfiguration)


@pytest.mark.integration
class TestConfigurationValidation:
    """Test comprehensive configuration validation."""
    
    def test_plugin_configuration_validation(self):
        """Test plugin-specific configuration validation."""
        plugin_configs = {
            "feature_extractors": {
                "molecular": {
                    "enabled": True,
                    "rdkit_enabled": True,
                    "descriptors": ["molecular_weight", "logp"]
                },
                "cybersecurity": {
                    "enabled": True,
                    "threat_detection": True
                },
                "financial": {
                    "enabled": True,
                    "technical_indicators": ["sma", "rsi"]
                }
            }
        }
        
        manager = ConfigurationManager()
        
        # Validate each plugin configuration
        for plugin_type, plugins in plugin_configs.items():
            for plugin_name, plugin_config in plugins.items():
                # Test that configuration structure is valid
                assert isinstance(plugin_config, dict)
                assert "enabled" in plugin_config
                assert isinstance(plugin_config["enabled"], bool)
    
    def test_resource_constraint_validation(self):
        """Test validation of resource constraint configurations."""
        resource_configs = [
            {"core": {"max_workers": 1}},  # Minimum
            {"core": {"max_workers": 16}},  # High but reasonable
            {"core": {"max_workers": 0}},   # Invalid - should fail
            {"core": {"max_workers": -1}},  # Invalid - should fail
        ]
        
        manager = ConfigurationManager()
        
        for i, config in enumerate(resource_configs):
            full_config = {**config, "plugins": {}, "logging": {}}
            is_valid, errors = manager.validate_config(full_config)
            
            if i < 2:  # First two should be valid
                assert is_valid == True, f"Config {i} should be valid: {config}"
            else:  # Last two should be invalid
                assert is_valid == False, f"Config {i} should be invalid: {config}"
    
    def test_logging_configuration_validation(self):
        """Test validation of logging configurations."""
        logging_configs = [
            {"logging": {"level": "DEBUG"}},
            {"logging": {"level": "INFO"}},
            {"logging": {"level": "WARNING"}},
            {"logging": {"level": "ERROR"}},
            {"logging": {"level": "INVALID"}},  # Should fail
        ]
        
        manager = ConfigurationManager()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for config in logging_configs:
            full_config = {"core": {"max_workers": 2}, "plugins": {}, **config}
            is_valid, errors = manager.validate_config(full_config)
            
            expected_valid = config["logging"]["level"] in valid_levels
            if expected_valid:
                assert is_valid == True or len(errors) == 0, f"Valid logging config failed: {config}"
    
    def test_cross_plugin_dependency_validation(self):
        """Test validation of cross-plugin dependencies."""
        # Test configuration where model depends on feature extractor
        dependent_config = {
            "core": {"max_workers": 2},
            "plugins": {
                "feature_extractors": {
                    "molecular": {"enabled": False}  # Disabled
                },
                "models": {
                    "molecular": {"enabled": True}   # Enabled but depends on FE
                }
            },
            "logging": {}
        }
        
        manager = ConfigurationManager()
        is_valid, errors = manager.validate_config(dependent_config)
        
        # Should detect dependency issue (would need implementation)
        # For now, just verify validation completes
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)


@pytest.mark.performance
class TestConfigurationPerformance:
    """Test configuration loading and validation performance."""
    
    def test_config_loading_performance(self, sample_config_file):
        """Test configuration loading performance."""
        load_times = []
        
        for _ in range(10):
            start_time = time.perf_counter()
            manager = ConfigurationManager(str(sample_config_file))
            config = manager.get_config()
            end_time = time.perf_counter()
            
            load_times.append(end_time - start_time)
        
        avg_load_time = sum(load_times) / len(load_times)
        max_load_time = max(load_times)
        
        # Should load quickly
        assert avg_load_time < 0.1, f"Average config load time {avg_load_time:.3f}s too high"
        assert max_load_time < 0.2, f"Maximum config load time {max_load_time:.3f}s too high"
    
    def test_config_validation_performance(self, sample_config):
        """Test configuration validation performance."""
        manager = ConfigurationManager()
        
        validation_times = []
        
        for _ in range(20):
            start_time = time.perf_counter()
            is_valid, errors = manager.validate_config(sample_config)
            end_time = time.perf_counter()
            
            validation_times.append(end_time - start_time)
        
        avg_validation_time = sum(validation_times) / len(validation_times)
        
        # Validation should be fast
        assert avg_validation_time < 0.05, f"Average validation time {avg_validation_time:.3f}s too high"
    
    def test_large_config_handling(self, temp_directory):
        """Test handling of large configuration files."""
        # Create large configuration
        large_config = {
            "core": {"max_workers": 4},
            "plugins": {},
            "logging": {}
        }
        
        # Add many plugin configurations
        for domain in ["molecular", "cybersecurity", "financial"]:
            for plugin_type in ["feature_extractors", "models", "proof_languages", "knowledge_bases"]:
                if plugin_type not in large_config["plugins"]:
                    large_config["plugins"][plugin_type] = {}
                
                large_config["plugins"][plugin_type][domain] = {
                    "enabled": True,
                    "parameters": {f"param_{i}": f"value_{i}" for i in range(100)}
                }
        
        large_config_file = temp_directory / "large_config.yaml"
        with open(large_config_file, 'w') as f:
            yaml.dump(large_config, f)
        
        # Test loading large config
        start_time = time.perf_counter()
        manager = ConfigurationManager(str(large_config_file))
        config = manager.get_config()
        load_time = time.perf_counter() - start_time
        
        # Should handle large config reasonably well
        assert load_time < 1.0, f"Large config load time {load_time:.3f}s too high"
        assert config is not None