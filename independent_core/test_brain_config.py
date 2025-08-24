"""
Comprehensive test suite for BrainConfig component.
Tests configuration validation, initialization, and all edge cases.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import json
import os

# Import the component to test
from brain_core import BrainConfig


class TestBrainConfigDefaults(unittest.TestCase):
    """Test default initialization and values of BrainConfig."""
    
    def test_default_initialization(self):
        """Test that BrainConfig initializes with correct default values."""
        config = BrainConfig()
        
        # Test shared knowledge settings
        self.assertEqual(config.shared_memory_size, 10000)
        self.assertTrue(config.knowledge_persistence)
        self.assertIsNone(config.knowledge_path)
        
        # Test reasoning settings
        self.assertEqual(config.reasoning_depth, 5)
        self.assertEqual(config.context_window_size, 2048)
        
        # Test performance settings
        self.assertTrue(config.enable_caching)
        self.assertEqual(config.cache_size, 1000)
        self.assertTrue(config.thread_safe)
        
        # Test uncertainty settings
        self.assertTrue(config.enable_uncertainty)
        self.assertEqual(config.uncertainty_history_size, 1000)
        self.assertEqual(config.confidence_threshold, 0.7)
        self.assertEqual(config.reliability_window, 100)
        
        # Test logging settings
        self.assertEqual(config.log_level, "INFO")
        self.assertEqual(config.log_format, "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    def test_dataclass_fields_exist(self):
        """Test that all expected fields exist in the dataclass."""
        config = BrainConfig()
        expected_fields = [
            'shared_memory_size', 'knowledge_persistence', 'knowledge_path',
            'reasoning_depth', 'context_window_size',
            'enable_caching', 'cache_size', 'thread_safe',
            'enable_uncertainty', 'uncertainty_history_size', 
            'confidence_threshold', 'reliability_window',
            'log_level', 'log_format'
        ]
        
        for field in expected_fields:
            self.assertTrue(hasattr(config, field), f"Missing field: {field}")


class TestBrainConfigCustomInitialization(unittest.TestCase):
    """Test custom initialization of BrainConfig."""
    
    def test_custom_shared_knowledge_settings(self):
        """Test custom shared knowledge settings."""
        config = BrainConfig(
            shared_memory_size=20000,
            knowledge_persistence=False,
            knowledge_path=Path("/custom/path")
        )
        
        self.assertEqual(config.shared_memory_size, 20000)
        self.assertFalse(config.knowledge_persistence)
        self.assertEqual(config.knowledge_path, Path("/custom/path"))
    
    def test_custom_reasoning_settings(self):
        """Test custom reasoning settings."""
        config = BrainConfig(
            reasoning_depth=10,
            context_window_size=4096
        )
        
        self.assertEqual(config.reasoning_depth, 10)
        self.assertEqual(config.context_window_size, 4096)
    
    def test_custom_performance_settings(self):
        """Test custom performance settings."""
        config = BrainConfig(
            enable_caching=False,
            cache_size=5000,
            thread_safe=False
        )
        
        self.assertFalse(config.enable_caching)
        self.assertEqual(config.cache_size, 5000)
        self.assertFalse(config.thread_safe)
    
    def test_custom_uncertainty_settings(self):
        """Test custom uncertainty settings."""
        config = BrainConfig(
            enable_uncertainty=False,
            uncertainty_history_size=2000,
            confidence_threshold=0.9,
            reliability_window=200
        )
        
        self.assertFalse(config.enable_uncertainty)
        self.assertEqual(config.uncertainty_history_size, 2000)
        self.assertEqual(config.confidence_threshold, 0.9)
        self.assertEqual(config.reliability_window, 200)
    
    def test_custom_logging_settings(self):
        """Test custom logging settings."""
        config = BrainConfig(
            log_level="DEBUG",
            log_format="%(message)s"
        )
        
        self.assertEqual(config.log_level, "DEBUG")
        self.assertEqual(config.log_format, "%(message)s")
    
    def test_partial_custom_initialization(self):
        """Test that partial custom initialization preserves defaults."""
        config = BrainConfig(
            shared_memory_size=15000,
            confidence_threshold=0.8
        )
        
        # Custom values
        self.assertEqual(config.shared_memory_size, 15000)
        self.assertEqual(config.confidence_threshold, 0.8)
        
        # Default values should be preserved
        self.assertTrue(config.knowledge_persistence)
        self.assertEqual(config.reasoning_depth, 5)
        self.assertTrue(config.enable_caching)


class TestBrainConfigValidation(unittest.TestCase):
    """Test validation logic in BrainConfig.__post_init__."""
    
    def test_negative_shared_memory_size_raises_error(self):
        """Test that negative shared_memory_size raises ValueError."""
        with self.assertRaises(ValueError) as context:
            BrainConfig(shared_memory_size=-1)
        self.assertIn("shared_memory_size must be positive", str(context.exception))
    
    def test_zero_shared_memory_size_raises_error(self):
        """Test that zero shared_memory_size raises ValueError."""
        with self.assertRaises(ValueError) as context:
            BrainConfig(shared_memory_size=0)
        self.assertIn("shared_memory_size must be positive", str(context.exception))
    
    def test_negative_reasoning_depth_raises_error(self):
        """Test that negative reasoning_depth raises ValueError."""
        with self.assertRaises(ValueError) as context:
            BrainConfig(reasoning_depth=-1)
        self.assertIn("reasoning_depth must be positive", str(context.exception))
    
    def test_zero_reasoning_depth_raises_error(self):
        """Test that zero reasoning_depth raises ValueError."""
        with self.assertRaises(ValueError) as context:
            BrainConfig(reasoning_depth=0)
        self.assertIn("reasoning_depth must be positive", str(context.exception))
    
    def test_negative_context_window_size_raises_error(self):
        """Test that negative context_window_size raises ValueError."""
        with self.assertRaises(ValueError) as context:
            BrainConfig(context_window_size=-1)
        self.assertIn("context_window_size must be positive", str(context.exception))
    
    def test_zero_context_window_size_raises_error(self):
        """Test that zero context_window_size raises ValueError."""
        with self.assertRaises(ValueError) as context:
            BrainConfig(context_window_size=0)
        self.assertIn("context_window_size must be positive", str(context.exception))
    
    def test_confidence_threshold_below_zero_raises_error(self):
        """Test that confidence_threshold < 0 raises ValueError."""
        with self.assertRaises(ValueError) as context:
            BrainConfig(confidence_threshold=-0.1)
        self.assertIn("confidence_threshold must be between 0 and 1", str(context.exception))
    
    def test_confidence_threshold_above_one_raises_error(self):
        """Test that confidence_threshold > 1 raises ValueError."""
        with self.assertRaises(ValueError) as context:
            BrainConfig(confidence_threshold=1.1)
        self.assertIn("confidence_threshold must be between 0 and 1", str(context.exception))
    
    def test_confidence_threshold_boundary_values(self):
        """Test that confidence_threshold accepts boundary values 0.0 and 1.0."""
        # Test 0.0
        config_zero = BrainConfig(confidence_threshold=0.0)
        self.assertEqual(config_zero.confidence_threshold, 0.0)
        
        # Test 1.0
        config_one = BrainConfig(confidence_threshold=1.0)
        self.assertEqual(config_one.confidence_threshold, 1.0)
    
    def test_valid_positive_values(self):
        """Test that all positive integer fields accept valid positive values."""
        config = BrainConfig(
            shared_memory_size=1,
            reasoning_depth=1,
            context_window_size=1,
            cache_size=1,
            uncertainty_history_size=1,
            reliability_window=1
        )
        
        self.assertEqual(config.shared_memory_size, 1)
        self.assertEqual(config.reasoning_depth, 1)
        self.assertEqual(config.context_window_size, 1)
        self.assertEqual(config.cache_size, 1)
        self.assertEqual(config.uncertainty_history_size, 1)
        self.assertEqual(config.reliability_window, 1)


class TestBrainConfigKnowledgePath(unittest.TestCase):
    """Test knowledge path creation and handling."""
    
    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_knowledge_path_directory_creation(self):
        """Test that knowledge_path parent directory is created when persistence is enabled."""
        knowledge_file = self.temp_path / "subdir" / "knowledge.json"
        
        # Verify subdirectory doesn't exist yet
        self.assertFalse((self.temp_path / "subdir").exists())
        
        # Create config with knowledge persistence
        config = BrainConfig(
            knowledge_persistence=True,
            knowledge_path=knowledge_file
        )
        
        # Verify directory was created
        self.assertTrue((self.temp_path / "subdir").exists())
        self.assertTrue((self.temp_path / "subdir").is_dir())
        self.assertEqual(config.knowledge_path, knowledge_file)
    
    def test_knowledge_path_string_conversion(self):
        """Test that string paths are converted to Path objects."""
        knowledge_file = str(self.temp_path / "knowledge.json")
        
        config = BrainConfig(
            knowledge_persistence=True,
            knowledge_path=knowledge_file
        )
        
        # Should be converted to Path object
        self.assertIsInstance(config.knowledge_path, Path)
        self.assertEqual(config.knowledge_path, Path(knowledge_file))
    
    def test_knowledge_path_not_created_when_persistence_disabled(self):
        """Test that directory is not created when persistence is disabled."""
        knowledge_file = self.temp_path / "no_create" / "knowledge.json"
        
        # Create config with persistence disabled
        config = BrainConfig(
            knowledge_persistence=False,
            knowledge_path=knowledge_file
        )
        
        # Directory should not be created
        self.assertFalse((self.temp_path / "no_create").exists())
        # But path should still be set
        self.assertEqual(config.knowledge_path, knowledge_file)
    
    def test_knowledge_path_none_with_persistence_enabled(self):
        """Test that persistence enabled with None path doesn't cause errors."""
        config = BrainConfig(
            knowledge_persistence=True,
            knowledge_path=None
        )
        
        self.assertTrue(config.knowledge_persistence)
        self.assertIsNone(config.knowledge_path)
    
    def test_existing_directory_not_recreated(self):
        """Test that existing directories are not recreated."""
        existing_dir = self.temp_path / "existing"
        existing_dir.mkdir()
        
        # Create a file in the directory to verify it's not deleted
        test_file = existing_dir / "test.txt"
        test_file.write_text("test content")
        
        knowledge_file = existing_dir / "knowledge.json"
        
        config = BrainConfig(
            knowledge_persistence=True,
            knowledge_path=knowledge_file
        )
        
        # Directory and file should still exist
        self.assertTrue(existing_dir.exists())
        self.assertTrue(test_file.exists())
        self.assertEqual(test_file.read_text(), "test content")


class TestBrainConfigEdgeCases(unittest.TestCase):
    """Test edge cases and unusual inputs for BrainConfig."""
    
    def test_very_large_values(self):
        """Test that very large values are accepted for size parameters."""
        config = BrainConfig(
            shared_memory_size=10**9,  # 1 billion
            cache_size=10**8,  # 100 million
            uncertainty_history_size=10**7,  # 10 million
            reasoning_depth=1000,
            context_window_size=10**6  # 1 million
        )
        
        self.assertEqual(config.shared_memory_size, 10**9)
        self.assertEqual(config.cache_size, 10**8)
        self.assertEqual(config.uncertainty_history_size, 10**7)
        self.assertEqual(config.reasoning_depth, 1000)
        self.assertEqual(config.context_window_size, 10**6)
    
    def test_float_values_for_integer_fields(self):
        """Test that float values work for integer fields (Python's duck typing)."""
        config = BrainConfig(
            shared_memory_size=100.5,
            reasoning_depth=5.7,
            context_window_size=2048.9
        )
        
        # Python allows floats for these fields
        self.assertEqual(config.shared_memory_size, 100.5)
        self.assertEqual(config.reasoning_depth, 5.7)
        self.assertEqual(config.context_window_size, 2048.9)
    
    def test_confidence_threshold_precision(self):
        """Test that confidence_threshold maintains precision."""
        config = BrainConfig(confidence_threshold=0.123456789)
        self.assertEqual(config.confidence_threshold, 0.123456789)
    
    def test_empty_log_format(self):
        """Test that empty log_format is accepted."""
        config = BrainConfig(log_format="")
        self.assertEqual(config.log_format, "")
    
    def test_special_characters_in_log_format(self):
        """Test that special characters in log_format are preserved."""
        special_format = "%(asctime)s | [%(levelname)s] >>> %(message)s <<<\n\t"
        config = BrainConfig(log_format=special_format)
        self.assertEqual(config.log_format, special_format)
    
    def test_unusual_log_levels(self):
        """Test that any string is accepted for log_level."""
        unusual_levels = ["CRITICAL", "WARNING", "ERROR", "debug", "TRACE", "CUSTOM", "123", ""]
        
        for level in unusual_levels:
            config = BrainConfig(log_level=level)
            self.assertEqual(config.log_level, level)
    
    def test_path_with_special_characters(self):
        """Test that paths with special characters are handled correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            special_path = Path(temp_dir) / "path with spaces" / "特殊文字" / "knowledge.json"
            
            config = BrainConfig(
                knowledge_persistence=True,
                knowledge_path=special_path
            )
            
            self.assertEqual(config.knowledge_path, special_path)
            self.assertTrue(special_path.parent.exists())


class TestBrainConfigTypeHandling(unittest.TestCase):
    """Test type handling and conversions in BrainConfig."""
    
    def test_boolean_fields_with_truthy_values(self):
        """Test that boolean fields handle truthy/falsy values."""
        # Truthy values
        config_true = BrainConfig(
            knowledge_persistence=1,
            enable_caching="yes",
            thread_safe=[1],
            enable_uncertainty={"key": "value"}
        )
        
        # Python's truthiness applies
        self.assertEqual(config_true.knowledge_persistence, 1)
        self.assertEqual(config_true.enable_caching, "yes")
        self.assertEqual(config_true.thread_safe, [1])
        self.assertEqual(config_true.enable_uncertainty, {"key": "value"})
    
    def test_none_values_for_optional_fields(self):
        """Test that None is properly handled for optional fields."""
        config = BrainConfig(knowledge_path=None)
        self.assertIsNone(config.knowledge_path)
    
    def test_pathlib_path_object(self):
        """Test that Path objects are handled correctly."""
        path_obj = Path("/tmp/knowledge/brain.json")
        config = BrainConfig(knowledge_path=path_obj)
        
        self.assertIsInstance(config.knowledge_path, Path)
        self.assertEqual(config.knowledge_path, path_obj)


class TestBrainConfigIntegration(unittest.TestCase):
    """Test BrainConfig integration scenarios."""
    
    def test_config_creation_from_dict(self):
        """Test creating BrainConfig from dictionary (common use case)."""
        config_dict = {
            'shared_memory_size': 5000,
            'knowledge_persistence': True,
            'reasoning_depth': 3,
            'confidence_threshold': 0.85,
            'log_level': 'DEBUG'
        }
        
        config = BrainConfig(**config_dict)
        
        self.assertEqual(config.shared_memory_size, 5000)
        self.assertTrue(config.knowledge_persistence)
        self.assertEqual(config.reasoning_depth, 3)
        self.assertEqual(config.confidence_threshold, 0.85)
        self.assertEqual(config.log_level, 'DEBUG')
        
        # Defaults should still be set
        self.assertEqual(config.context_window_size, 2048)
        self.assertTrue(config.enable_caching)
    
    def test_config_serialization_compatibility(self):
        """Test that config values can be serialized (useful for saving/loading)."""
        config = BrainConfig(
            shared_memory_size=7500,
            knowledge_path=Path("/tmp/brain.json"),
            confidence_threshold=0.75
        )
        
        # Create a dict representation (simulating serialization)
        config_data = {
            'shared_memory_size': config.shared_memory_size,
            'knowledge_persistence': config.knowledge_persistence,
            'knowledge_path': str(config.knowledge_path) if config.knowledge_path else None,
            'reasoning_depth': config.reasoning_depth,
            'context_window_size': config.context_window_size,
            'enable_caching': config.enable_caching,
            'cache_size': config.cache_size,
            'thread_safe': config.thread_safe,
            'enable_uncertainty': config.enable_uncertainty,
            'uncertainty_history_size': config.uncertainty_history_size,
            'confidence_threshold': config.confidence_threshold,
            'reliability_window': config.reliability_window,
            'log_level': config.log_level,
            'log_format': config.log_format
        }
        
        # Verify it can be JSON serialized
        json_str = json.dumps(config_data)
        loaded_data = json.loads(json_str)
        
        # Create new config from loaded data
        new_config = BrainConfig(**loaded_data)
        
        self.assertEqual(new_config.shared_memory_size, config.shared_memory_size)
        self.assertEqual(new_config.confidence_threshold, config.confidence_threshold)
        self.assertEqual(str(new_config.knowledge_path), str(config.knowledge_path))
    
    def test_multiple_configs_independence(self):
        """Test that multiple BrainConfig instances are independent."""
        config1 = BrainConfig(shared_memory_size=1000)
        config2 = BrainConfig(shared_memory_size=2000)
        
        self.assertEqual(config1.shared_memory_size, 1000)
        self.assertEqual(config2.shared_memory_size, 2000)
        
        # Modifying one shouldn't affect the other
        config1.shared_memory_size = 3000
        self.assertEqual(config1.shared_memory_size, 3000)
        self.assertEqual(config2.shared_memory_size, 2000)


class TestBrainConfigStressTests(unittest.TestCase):
    """Stress tests for BrainConfig to ensure robustness."""
    
    def test_rapid_creation_destruction(self):
        """Test rapid creation and destruction of configs."""
        configs = []
        for i in range(1000):
            config = BrainConfig(
                shared_memory_size=i + 1,
                confidence_threshold=min(i / 1000, 1.0)
            )
            configs.append(config)
        
        # Verify a sample of configs
        self.assertEqual(configs[0].shared_memory_size, 1)
        self.assertEqual(configs[500].shared_memory_size, 501)
        self.assertEqual(configs[999].shared_memory_size, 1000)
        self.assertAlmostEqual(configs[700].confidence_threshold, 0.7, places=5)
    
    def test_all_parameters_simultaneously(self):
        """Test setting all parameters at once."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BrainConfig(
                shared_memory_size=12345,
                knowledge_persistence=False,
                knowledge_path=Path(temp_dir) / "test.json",
                reasoning_depth=8,
                context_window_size=4096,
                enable_caching=False,
                cache_size=2500,
                thread_safe=False,
                enable_uncertainty=False,
                uncertainty_history_size=1500,
                confidence_threshold=0.65,
                reliability_window=150,
                log_level="WARNING",
                log_format="%(levelname)s: %(message)s"
            )
            
            # Verify all values
            self.assertEqual(config.shared_memory_size, 12345)
            self.assertFalse(config.knowledge_persistence)
            self.assertEqual(config.knowledge_path, Path(temp_dir) / "test.json")
            self.assertEqual(config.reasoning_depth, 8)
            self.assertEqual(config.context_window_size, 4096)
            self.assertFalse(config.enable_caching)
            self.assertEqual(config.cache_size, 2500)
            self.assertFalse(config.thread_safe)
            self.assertFalse(config.enable_uncertainty)
            self.assertEqual(config.uncertainty_history_size, 1500)
            self.assertEqual(config.confidence_threshold, 0.65)
            self.assertEqual(config.reliability_window, 150)
            self.assertEqual(config.log_level, "WARNING")
            self.assertEqual(config.log_format, "%(levelname)s: %(message)s")


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)