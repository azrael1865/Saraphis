"""
Comprehensive Test Suite for GACConfig
Tests all configuration components, validation, persistence, and management
"""

import unittest
import json
import tempfile
import os
from pathlib import Path
from dataclasses import asdict, fields
from typing import Dict, Any
import shutil

# Import the modules to test
from independent_core.gac_system.gac_config import (
    GACMode,
    ComponentPriority,
    PIDConfig,
    MetaLearningConfig,
    ReinforcementLearningConfig,
    ThresholdConfig,
    MonitoringConfig,
    ComponentConfig,
    SystemConfig,
    IntegrationConfig,
    GACConfig,
    GACConfigManager,
    create_default_config,
    load_config_from_file
)


class TestEnums(unittest.TestCase):
    """Test enum definitions"""
    
    def test_gac_mode_enum(self):
        """Test GACMode enum values"""
        self.assertEqual(GACMode.CONSERVATIVE.value, "conservative")
        self.assertEqual(GACMode.BALANCED.value, "balanced")
        self.assertEqual(GACMode.AGGRESSIVE.value, "aggressive")
        self.assertEqual(GACMode.ADAPTIVE.value, "adaptive")
        
        # Test enum creation from string
        mode = GACMode("balanced")
        self.assertEqual(mode, GACMode.BALANCED)
    
    def test_component_priority_enum(self):
        """Test ComponentPriority enum values"""
        self.assertEqual(ComponentPriority.LOW.value, 1)
        self.assertEqual(ComponentPriority.MEDIUM.value, 2)
        self.assertEqual(ComponentPriority.HIGH.value, 3)
        self.assertEqual(ComponentPriority.CRITICAL.value, 4)
        
        # Test priority comparison
        self.assertLess(ComponentPriority.LOW.value, ComponentPriority.HIGH.value)


class TestPIDConfig(unittest.TestCase):
    """Test PIDConfig dataclass"""
    
    def test_default_values(self):
        """Test PIDConfig default values"""
        config = PIDConfig()
        self.assertEqual(config.kp, 1.0)
        self.assertEqual(config.ki, 0.1)
        self.assertEqual(config.kd, 0.05)
        self.assertEqual(config.setpoint, 0.0)
        self.assertEqual(config.integral_windup_limit, 100.0)
        self.assertEqual(config.derivative_smoothing, 0.1)
        self.assertEqual(config.output_limits, (-10.0, 10.0))
        self.assertTrue(config.auto_tune)
        self.assertEqual(config.sample_time, 0.1)
    
    def test_custom_values(self):
        """Test PIDConfig with custom values"""
        config = PIDConfig(
            kp=2.5,
            ki=0.5,
            kd=0.1,
            setpoint=5.0,
            output_limits=(-20.0, 20.0)
        )
        self.assertEqual(config.kp, 2.5)
        self.assertEqual(config.ki, 0.5)
        self.assertEqual(config.setpoint, 5.0)
        self.assertEqual(config.output_limits, (-20.0, 20.0))
    
    def test_serialization(self):
        """Test PIDConfig serialization"""
        config = PIDConfig(kp=1.5, ki=0.2)
        config_dict = asdict(config)
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['kp'], 1.5)
        self.assertEqual(config_dict['ki'], 0.2)
        self.assertIn('auto_tune', config_dict)


class TestMetaLearningConfig(unittest.TestCase):
    """Test MetaLearningConfig dataclass"""
    
    def test_default_values(self):
        """Test MetaLearningConfig default values"""
        config = MetaLearningConfig()
        self.assertEqual(config.learning_rate, 0.01)
        self.assertEqual(config.adaptation_threshold, 0.1)
        self.assertEqual(config.history_limit, 1000)
        self.assertEqual(config.minimum_samples, 10)
        self.assertEqual(config.hidden_layers, [128, 64, 32])
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.epochs, 100)
    
    def test_custom_hidden_layers(self):
        """Test custom hidden layers configuration"""
        config = MetaLearningConfig(
            hidden_layers=[256, 128, 64, 32]
        )
        self.assertEqual(len(config.hidden_layers), 4)
        self.assertEqual(config.hidden_layers[0], 256)
    
    def test_validation_bounds(self):
        """Test configuration value bounds"""
        config = MetaLearningConfig(
            learning_rate=0.001,
            dropout_rate=0.5,
            early_stopping_patience=20
        )
        self.assertGreater(config.learning_rate, 0)
        self.assertLessEqual(config.dropout_rate, 1.0)
        self.assertGreater(config.early_stopping_patience, 0)


class TestReinforcementLearningConfig(unittest.TestCase):
    """Test ReinforcementLearningConfig dataclass"""
    
    def test_default_values(self):
        """Test RL config default values"""
        config = ReinforcementLearningConfig()
        self.assertEqual(config.learning_rate, 0.1)
        self.assertEqual(config.discount_factor, 0.95)
        self.assertEqual(config.epsilon, 0.1)
        self.assertEqual(config.exploration_strategy, "epsilon_greedy")
        
        # Check default dictionaries
        self.assertIn("performance_buckets", config.state_discretization)
        self.assertIn("performance_improvement", config.reward_weights)
        self.assertIn("increase_threshold", config.action_space)
    
    def test_action_space(self):
        """Test action space configuration"""
        config = ReinforcementLearningConfig()
        expected_actions = [
            "increase_threshold", "decrease_threshold", "maintain",
            "boost_component", "reduce_sensitivity", "emergency_stop"
        ]
        self.assertEqual(config.action_space, expected_actions)
    
    def test_reward_weights(self):
        """Test reward weight configuration"""
        config = ReinforcementLearningConfig()
        self.assertEqual(config.reward_weights["performance_improvement"], 1.0)
        self.assertEqual(config.reward_weights["error_penalty"], -2.0)
        
        # Test custom weights
        custom_weights = {
            "performance_improvement": 2.0,
            "stability_bonus": 1.0,
            "efficiency_bonus": 0.5,
            "error_penalty": -3.0
        }
        config = ReinforcementLearningConfig(reward_weights=custom_weights)
        self.assertEqual(config.reward_weights["performance_improvement"], 2.0)


class TestThresholdConfig(unittest.TestCase):
    """Test ThresholdConfig dataclass"""
    
    def test_default_thresholds(self):
        """Test default threshold values"""
        config = ThresholdConfig()
        self.assertEqual(config.gradient_magnitude, 1.0)
        self.assertEqual(config.processing_time, 5.0)
        self.assertEqual(config.error_rate, 0.05)
        self.assertEqual(config.memory_usage, 0.8)
        self.assertEqual(config.cpu_usage, 0.7)
        self.assertTrue(config.auto_adjust)
    
    def test_emergency_threshold(self):
        """Test emergency threshold multiplier"""
        config = ThresholdConfig()
        normal_threshold = config.gradient_magnitude
        emergency_threshold = normal_threshold * config.emergency_threshold_multiplier
        self.assertEqual(emergency_threshold, 2.0)
    
    def test_threshold_modification(self):
        """Test threshold modification"""
        config = ThresholdConfig(
            gradient_magnitude=2.0,
            error_rate=0.01,
            auto_adjust=False
        )
        self.assertEqual(config.gradient_magnitude, 2.0)
        self.assertEqual(config.error_rate, 0.01)
        self.assertFalse(config.auto_adjust)


class TestMonitoringConfig(unittest.TestCase):
    """Test MonitoringConfig dataclass"""
    
    def test_default_values(self):
        """Test monitoring config defaults"""
        config = MonitoringConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.sampling_interval, 1.0)
        self.assertEqual(config.metrics_retention, 3600)
        self.assertIn("log", config.notification_channels)
        self.assertTrue(config.automatic_recovery)
    
    def test_alert_thresholds(self):
        """Test alert threshold configuration"""
        config = MonitoringConfig()
        self.assertEqual(config.alert_thresholds["high_error_rate"], 0.1)
        self.assertEqual(config.alert_thresholds["high_latency"], 10.0)
        self.assertEqual(config.alert_thresholds["memory_pressure"], 0.9)
    
    def test_escalation_levels(self):
        """Test escalation level configuration"""
        config = MonitoringConfig()
        self.assertEqual(config.escalation_levels["warning"], 1)
        self.assertEqual(config.escalation_levels["critical"], 3)
        self.assertEqual(config.escalation_levels["emergency"], 4)


class TestComponentConfig(unittest.TestCase):
    """Test ComponentConfig dataclass"""
    
    def test_required_component_id(self):
        """Test that component_id is required"""
        config = ComponentConfig(component_id="test_component")
        self.assertEqual(config.component_id, "test_component")
    
    def test_default_values(self):
        """Test component config defaults"""
        config = ComponentConfig(component_id="test")
        self.assertEqual(config.priority, ComponentPriority.MEDIUM)
        self.assertTrue(config.enabled)
        self.assertTrue(config.auto_start)
        self.assertTrue(config.restart_on_failure)
        self.assertEqual(config.max_restart_attempts, 3)
        self.assertEqual(config.timeout, 30.0)
        self.assertIsNone(config.memory_limit)
        self.assertEqual(len(config.dependencies), 0)
    
    def test_custom_configuration(self):
        """Test custom component configuration"""
        config = ComponentConfig(
            component_id="critical_component",
            priority=ComponentPriority.CRITICAL,
            timeout=60.0,
            memory_limit=1024,
            cpu_limit=0.5,
            dependencies=["dep1", "dep2"],
            custom_config={"key": "value"}
        )
        self.assertEqual(config.priority, ComponentPriority.CRITICAL)
        self.assertEqual(config.timeout, 60.0)
        self.assertEqual(config.memory_limit, 1024)
        self.assertEqual(len(config.dependencies), 2)
        self.assertEqual(config.custom_config["key"], "value")


class TestSystemConfig(unittest.TestCase):
    """Test SystemConfig dataclass"""
    
    def test_default_values(self):
        """Test system config defaults"""
        config = SystemConfig()
        self.assertEqual(config.mode, GACMode.BALANCED)
        self.assertEqual(config.max_workers, 8)
        self.assertEqual(config.worker_timeout, 30.0)
        self.assertEqual(config.checkpoint_interval, 300)
        self.assertTrue(config.state_persistence)
        self.assertFalse(config.debug_mode)
        self.assertEqual(config.log_level, "INFO")
    
    def test_mode_configuration(self):
        """Test system mode configuration"""
        config = SystemConfig(mode=GACMode.AGGRESSIVE)
        self.assertEqual(config.mode, GACMode.AGGRESSIVE)
        
        config = SystemConfig(mode=GACMode.CONSERVATIVE)
        self.assertEqual(config.mode, GACMode.CONSERVATIVE)


class TestIntegrationConfig(unittest.TestCase):
    """Test IntegrationConfig dataclass"""
    
    def test_default_values(self):
        """Test integration config defaults"""
        config = IntegrationConfig()
        self.assertTrue(config.brain_integration)
        self.assertEqual(config.hook_timeout, 5.0)
        self.assertTrue(config.async_processing)
        self.assertTrue(config.batch_processing)
        self.assertEqual(config.batch_size, 100)
        self.assertTrue(config.event_driven)
    
    def test_pipeline_stages(self):
        """Test pipeline stages configuration"""
        config = IntegrationConfig()
        expected_stages = ["preprocessing", "analysis", "processing", "postprocessing"]
        self.assertEqual(config.pipeline_stages, expected_stages)
    
    def test_custom_external_apis(self):
        """Test external API configuration"""
        apis = {
            "api1": {"url": "http://api1.com", "key": "key1"},
            "api2": {"url": "http://api2.com", "key": "key2"}
        }
        config = IntegrationConfig(external_apis=apis)
        self.assertEqual(len(config.external_apis), 2)
        self.assertEqual(config.external_apis["api1"]["url"], "http://api1.com")


class TestGACConfig(unittest.TestCase):
    """Test main GACConfig dataclass"""
    
    def test_default_initialization(self):
        """Test GACConfig with all defaults"""
        config = GACConfig()
        self.assertIsInstance(config.system, SystemConfig)
        self.assertIsInstance(config.pid, PIDConfig)
        self.assertIsInstance(config.meta_learning, MetaLearningConfig)
        self.assertIsInstance(config.reinforcement_learning, ReinforcementLearningConfig)
        self.assertIsInstance(config.thresholds, ThresholdConfig)
        self.assertIsInstance(config.monitoring, MonitoringConfig)
        self.assertIsInstance(config.integration, IntegrationConfig)
        self.assertEqual(len(config.components), 0)
        self.assertEqual(len(config.custom_settings), 0)
    
    def test_partial_initialization(self):
        """Test GACConfig with partial custom values"""
        custom_pid = PIDConfig(kp=2.0, ki=0.5)
        custom_system = SystemConfig(mode=GACMode.AGGRESSIVE)
        
        config = GACConfig(
            system=custom_system,
            pid=custom_pid
        )
        
        self.assertEqual(config.pid.kp, 2.0)
        self.assertEqual(config.system.mode, GACMode.AGGRESSIVE)
        # Other configs should still have defaults
        self.assertEqual(config.thresholds.gradient_magnitude, 1.0)
    
    def test_complete_serialization(self):
        """Test complete GACConfig serialization"""
        config = GACConfig()
        config_dict = asdict(config)
        
        self.assertIn("system", config_dict)
        self.assertIn("pid", config_dict)
        self.assertIn("meta_learning", config_dict)
        self.assertIn("reinforcement_learning", config_dict)
        self.assertIn("thresholds", config_dict)
        self.assertIn("monitoring", config_dict)
        self.assertIn("integration", config_dict)
        self.assertIn("components", config_dict)
        
        # Test nested serialization
        self.assertIsInstance(config_dict["system"], dict)
        self.assertEqual(config_dict["pid"]["kp"], 1.0)


class TestGACConfigManager(unittest.TestCase):
    """Test GACConfigManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization_without_file(self):
        """Test manager initialization without config file"""
        manager = GACConfigManager(self.config_path, auto_load=False)
        self.assertIsInstance(manager.config, GACConfig)
        self.assertEqual(manager.config_path, Path(self.config_path))
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration"""
        # Create and save config
        manager = GACConfigManager(self.config_path, auto_load=False)
        manager.config.system.mode = GACMode.AGGRESSIVE
        manager.config.pid.kp = 2.5
        
        success = manager.save_config(create_backup=False)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(self.config_path))
        
        # Load config in new manager
        new_manager = GACConfigManager(self.config_path, auto_load=True)
        self.assertEqual(new_manager.config.system.mode, GACMode.AGGRESSIVE)
        self.assertEqual(new_manager.config.pid.kp, 2.5)
    
    def test_backup_creation(self):
        """Test backup file creation when saving"""
        manager = GACConfigManager(self.config_path, auto_load=False)
        
        # First save
        manager.save_config(create_backup=False)
        
        # Modify and save with backup
        manager.config.system.debug_mode = True
        manager.save_config(create_backup=True)
        
        backup_path = Path(self.config_path).with_suffix('.backup')
        self.assertTrue(backup_path.exists())
    
    def test_component_config_management(self):
        """Test component configuration management"""
        manager = GACConfigManager(self.config_path, auto_load=False)
        
        # Add component
        component = ComponentConfig(
            component_id="test_comp",
            priority=ComponentPriority.HIGH
        )
        manager.add_component_config(component)
        
        # Get component
        retrieved = manager.get_component_config("test_comp")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.priority, ComponentPriority.HIGH)
        
        # Remove component
        manager.remove_component_config("test_comp")
        self.assertIsNone(manager.get_component_config("test_comp"))
    
    def test_config_validation(self):
        """Test configuration validation"""
        manager = GACConfigManager(self.config_path, auto_load=False)
        
        # Valid configuration
        is_valid, errors = manager.validate_config()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Invalid PID Kp
        manager.config.pid.kp = -1.0
        is_valid, errors = manager.validate_config()
        self.assertFalse(is_valid)
        self.assertIn("PID Kp must be non-negative", errors)
        
        # Invalid learning rate
        manager.config.pid.kp = 1.0
        manager.config.meta_learning.learning_rate = 1.5
        is_valid, errors = manager.validate_config()
        self.assertFalse(is_valid)
        self.assertIn("Meta-learning rate must be between 0 and 1", errors)
        
        # Invalid gradient threshold
        manager.config.meta_learning.learning_rate = 0.01
        manager.config.thresholds.gradient_magnitude = -1.0
        is_valid, errors = manager.validate_config()
        self.assertFalse(is_valid)
        self.assertIn("Gradient magnitude threshold must be positive", errors)
    
    def test_preset_configurations(self):
        """Test preset configuration creation"""
        manager = GACConfigManager(self.config_path, auto_load=False)
        
        # Development preset
        success = manager.create_preset_config("development")
        self.assertTrue(success)
        self.assertTrue(manager.config.system.debug_mode)
        self.assertEqual(manager.config.system.log_level, "DEBUG")
        self.assertEqual(manager.config.monitoring.sampling_interval, 0.5)
        
        # Production preset
        success = manager.create_preset_config("production")
        self.assertTrue(success)
        self.assertFalse(manager.config.system.debug_mode)
        self.assertEqual(manager.config.system.log_level, "WARNING")
        self.assertEqual(manager.config.monitoring.sampling_interval, 5.0)
        
        # Conservative preset
        success = manager.create_preset_config("conservative")
        self.assertTrue(success)
        self.assertEqual(manager.config.system.mode, GACMode.CONSERVATIVE)
        self.assertEqual(manager.config.thresholds.gradient_magnitude, 0.5)
        self.assertEqual(manager.config.pid.kp, 0.5)
        
        # Aggressive preset
        success = manager.create_preset_config("aggressive")
        self.assertTrue(success)
        self.assertEqual(manager.config.system.mode, GACMode.AGGRESSIVE)
        self.assertEqual(manager.config.thresholds.gradient_magnitude, 2.0)
        self.assertEqual(manager.config.pid.kp, 2.0)
        
        # Invalid preset
        success = manager.create_preset_config("invalid_preset")
        self.assertFalse(success)
    
    def test_config_update(self):
        """Test configuration update functionality"""
        manager = GACConfigManager(self.config_path, auto_load=False)
        
        updates = {
            "system": {
                "max_workers": 16,
                "debug_mode": True
            },
            "pid": {
                "kp": 1.5
            },
            "thresholds": {
                "error_rate": 0.01
            }
        }
        
        success = manager.update_config(updates)
        self.assertTrue(success)
        self.assertEqual(manager.config.system.max_workers, 16)
        self.assertTrue(manager.config.system.debug_mode)
        self.assertEqual(manager.config.pid.kp, 1.5)
        self.assertEqual(manager.config.thresholds.error_rate, 0.01)
    
    def test_config_summary(self):
        """Test configuration summary generation"""
        manager = GACConfigManager(self.config_path, auto_load=False)
        
        # Add a component
        component = ComponentConfig(component_id="test")
        manager.add_component_config(component)
        
        summary = manager.get_config_summary()
        
        self.assertIn("system_mode", summary)
        self.assertIn("debug_mode", summary)
        self.assertIn("max_workers", summary)
        self.assertIn("monitoring_enabled", summary)
        self.assertIn("component_count", summary)
        self.assertIn("brain_integration", summary)
        self.assertIn("thresholds", summary)
        
        self.assertEqual(summary["system_mode"], "balanced")
        self.assertEqual(summary["component_count"], 1)
        self.assertTrue(summary["brain_integration"])
    
    def test_load_from_dict(self):
        """Test loading configuration from dictionary"""
        manager = GACConfigManager(self.config_path, auto_load=False)
        
        config_dict = {
            "system": {
                "mode": "aggressive",
                "max_workers": 12,
                "debug_mode": True
            },
            "pid": {
                "kp": 1.8,
                "ki": 0.3
            },
            "components": {
                "comp1": {
                    "priority": "HIGH",
                    "enabled": True,
                    "timeout": 45.0
                }
            },
            "custom_settings": {
                "custom_key": "custom_value"
            }
        }
        
        manager._load_from_dict(config_dict)
        
        self.assertEqual(manager.config.system.mode, GACMode.AGGRESSIVE)
        self.assertEqual(manager.config.system.max_workers, 12)
        self.assertEqual(manager.config.pid.kp, 1.8)
        
        comp1 = manager.get_component_config("comp1")
        self.assertIsNotNone(comp1)
        self.assertEqual(comp1.priority, ComponentPriority.HIGH)
        self.assertEqual(comp1.timeout, 45.0)
        
        self.assertEqual(manager.config.custom_settings["custom_key"], "custom_value")
    
    def test_deep_update(self):
        """Test deep update functionality"""
        manager = GACConfigManager(self.config_path, auto_load=False)
        
        # Set initial values
        manager.config.monitoring.alert_thresholds["custom_alert"] = 0.5
        
        updates = {
            "monitoring": {
                "alert_thresholds": {
                    "high_error_rate": 0.2,
                    "new_threshold": 0.3
                }
            }
        }
        
        manager.update_config(updates)
        
        # Check updates were applied
        self.assertEqual(manager.config.monitoring.alert_thresholds["high_error_rate"], 0.2)
        self.assertEqual(manager.config.monitoring.alert_thresholds["new_threshold"], 0.3)
        # Check existing values were preserved
        self.assertEqual(manager.config.monitoring.alert_thresholds["custom_alert"], 0.5)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_create_default_config(self):
        """Test create_default_config function"""
        config = create_default_config()
        self.assertIsInstance(config, GACConfig)
        self.assertEqual(config.system.mode, GACMode.BALANCED)
        self.assertEqual(config.pid.kp, 1.0)
    
    def test_load_config_from_file(self):
        """Test load_config_from_file function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test.json")
            
            # Create a config file
            config_data = {
                "system": {"mode": "conservative"},
                "pid": {"kp": 0.8}
            }
            with open(config_path, 'w') as f:
                json.dump(config_data, f)
            
            # Load config
            config = load_config_from_file(config_path)
            self.assertIsNotNone(config)
            self.assertEqual(config.system.mode, GACMode.CONSERVATIVE)
            self.assertEqual(config.pid.kp, 0.8)
            
            # Test with non-existent file
            config = load_config_from_file("/non/existent/path.json")
            self.assertIsNone(config)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_config_file(self):
        """Test loading empty config file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "empty.json")
            with open(config_path, 'w') as f:
                json.dump({}, f)
            
            manager = GACConfigManager(config_path, auto_load=True)
            # Should use defaults
            self.assertEqual(manager.config.system.mode, GACMode.BALANCED)
    
    def test_malformed_json(self):
        """Test handling of malformed JSON"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "malformed.json")
            with open(config_path, 'w') as f:
                f.write("{invalid json}")
            
            manager = GACConfigManager(config_path, auto_load=True)
            # Should fall back to defaults
            self.assertIsInstance(manager.config, GACConfig)
    
    def test_component_with_invalid_priority(self):
        """Test component with invalid priority value in JSON"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test.json")
            config_data = {
                "components": {
                    "comp1": {
                        "priority": "INVALID",
                        "enabled": True
                    }
                }
            }
            with open(config_path, 'w') as f:
                json.dump(config_data, f)
            
            manager = GACConfigManager(config_path, auto_load=True)
            # Component should not be loaded due to invalid priority
            # This tests error handling in the loading process
    
    def test_circular_dependencies(self):
        """Test handling of circular component dependencies"""
        manager = GACConfigManager(auto_load=False)
        
        comp1 = ComponentConfig(
            component_id="comp1",
            dependencies=["comp2"]
        )
        comp2 = ComponentConfig(
            component_id="comp2",
            dependencies=["comp1"]
        )
        
        manager.add_component_config(comp1)
        manager.add_component_config(comp2)
        
        # Validation should still pass (dependency resolution is not part of validation)
        is_valid, errors = manager.validate_config()
        self.assertTrue(is_valid)
    
    def test_extreme_values(self):
        """Test configuration with extreme values"""
        config = GACConfig()
        
        # Test very large values
        config.system.max_workers = 1000000
        config.pid.kp = 1e10
        config.thresholds.gradient_magnitude = 1e-10
        
        # Should be able to handle extreme values
        config_dict = asdict(config)
        self.assertEqual(config_dict["system"]["max_workers"], 1000000)
        self.assertEqual(config_dict["pid"]["kp"], 1e10)
    
    def test_unicode_handling(self):
        """Test handling of unicode in configuration"""
        manager = GACConfigManager(auto_load=False)
        
        # Add component with unicode name
        component = ComponentConfig(
            component_id="测试组件",
            custom_config={"message": "Hello 世界"}
        )
        manager.add_component_config(component)
        
        retrieved = manager.get_component_config("测试组件")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.custom_config["message"], "Hello 世界")


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios"""
    
    def test_full_workflow(self):
        """Test complete configuration workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "workflow.json")
            
            # 1. Create manager and apply preset
            manager = GACConfigManager(config_path, auto_load=False)
            manager.create_preset_config("development")
            
            # 2. Add components
            for i in range(3):
                component = ComponentConfig(
                    component_id=f"component_{i}",
                    priority=ComponentPriority.MEDIUM
                )
                manager.add_component_config(component)
            
            # 3. Update configuration
            updates = {
                "system": {"max_workers": 4},
                "monitoring": {"sampling_interval": 2.0}
            }
            manager.update_config(updates)
            
            # 4. Validate
            is_valid, errors = manager.validate_config()
            self.assertTrue(is_valid)
            
            # 5. Save
            success = manager.save_config()
            self.assertTrue(success)
            
            # 6. Load in new manager
            new_manager = GACConfigManager(config_path, auto_load=True)
            
            # 7. Verify
            self.assertEqual(new_manager.config.system.max_workers, 4)
            self.assertEqual(len(new_manager.config.components), 3)
            self.assertTrue(new_manager.config.system.debug_mode)  # From development preset
    
    def test_migration_scenario(self):
        """Test configuration migration scenario"""
        with tempfile.TemporaryDirectory() as temp_dir:
            old_config_path = os.path.join(temp_dir, "old.json")
            new_config_path = os.path.join(temp_dir, "new.json")
            
            # Create old configuration
            old_config = {
                "system": {"mode": "balanced"},
                "pid": {"kp": 1.0}
            }
            with open(old_config_path, 'w') as f:
                json.dump(old_config, f)
            
            # Load old configuration
            old_manager = GACConfigManager(old_config_path, auto_load=True)
            
            # Migrate to new configuration with additions
            new_manager = GACConfigManager(new_config_path, auto_load=False)
            new_manager.config = old_manager.config
            
            # Add new components and settings
            new_manager.add_component_config(
                ComponentConfig(component_id="new_component")
            )
            new_manager.config.custom_settings["version"] = "2.0"
            
            # Save new configuration
            new_manager.save_config()
            
            # Verify migration
            verification_manager = GACConfigManager(new_config_path, auto_load=True)
            self.assertEqual(verification_manager.config.system.mode, GACMode.BALANCED)
            self.assertIsNotNone(verification_manager.get_component_config("new_component"))
            self.assertEqual(verification_manager.config.custom_settings["version"], "2.0")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)