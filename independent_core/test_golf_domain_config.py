"""
Test suite for Golf Domain Configuration.
Tests configuration validation, serialization, and hard failure behaviors.
"""

import unittest
import json
import tempfile
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from golf_domain.domain_config import GolfDomainConfig


class TestGolfDomainConfig(unittest.TestCase):
    """Test suite for GolfDomainConfig."""
    
    def test_default_initialization(self):
        """Test default GolfDomainConfig initialization."""
        config = GolfDomainConfig()
        
        self.assertEqual(config.domain_name, "golf")
        self.assertEqual(config.version, "1.0.0")
        self.assertEqual(config.max_strokes, 10)
        self.assertEqual(config.scoring_method, "stroke_play")
        self.assertTrue(config.handicap_enabled)
        self.assertEqual(config.max_handicap, 36)
        self.assertEqual(config.num_holes, 18)
        self.assertEqual(len(config.tee_positions), 3)
        
    def test_custom_initialization(self):
        """Test GolfDomainConfig with custom values."""
        config = GolfDomainConfig(
            max_strokes=15,
            scoring_method="match_play",
            num_holes=9,
            max_handicap=24
        )
        
        self.assertEqual(config.max_strokes, 15)
        self.assertEqual(config.scoring_method, "match_play")
        self.assertEqual(config.num_holes, 9)
        self.assertEqual(config.max_handicap, 24)
        
    def test_par_values(self):
        """Test par values configuration."""
        config = GolfDomainConfig()
        
        self.assertEqual(config.par_values[3], 3)
        self.assertEqual(config.par_values[4], 4)
        self.assertEqual(config.par_values[5], 5)
        
    def test_invalid_num_holes_hard_failure(self):
        """Test that invalid num_holes causes hard failure."""
        with self.assertRaises(ValueError) as context:
            GolfDomainConfig(num_holes=12)
        
        self.assertIn("num_holes must be 9 or 18", str(context.exception))
        
    def test_invalid_max_handicap_hard_failure(self):
        """Test that invalid max_handicap causes hard failure."""
        # Test negative handicap
        with self.assertRaises(ValueError) as context:
            GolfDomainConfig(max_handicap=-1)
        
        self.assertIn("max_handicap must be between 0 and 54", str(context.exception))
        
        # Test too high handicap
        with self.assertRaises(ValueError) as context:
            GolfDomainConfig(max_handicap=55)
        
        self.assertIn("max_handicap must be between 0 and 54", str(context.exception))
        
    def test_invalid_scoring_method_hard_failure(self):
        """Test that invalid scoring_method causes hard failure."""
        with self.assertRaises(ValueError) as context:
            GolfDomainConfig(scoring_method="invalid_method")
        
        self.assertIn("Invalid scoring_method", str(context.exception))
        
    def test_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = GolfDomainConfig(
            max_strokes=12,
            scoring_method="stableford"
        )
        
        config_dict = config.to_dict()
        
        self.assertEqual(config_dict["domain_name"], "golf")
        self.assertEqual(config_dict["max_strokes"], 12)
        self.assertEqual(config_dict["scoring_method"], "stableford")
        self.assertIn("par_values", config_dict)
        self.assertIn("tee_positions", config_dict)
        
    def test_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "domain_name": "golf_test",
            "version": "2.0.0",
            "max_strokes": 8,
            "par_values": {3: 3, 4: 4, 5: 5},
            "scoring_method": "match_play",
            "handicap_enabled": False,
            "max_handicap": 30,
            "num_holes": 9,
            "tee_positions": ["back", "front"],
            "track_statistics": True,
            "statistics_window": 10,
            "good_score_threshold": 0.85,
            "excellent_score_threshold": 0.75,
            "enable_ml_predictions": True,
            "enable_weather_integration": True
        }
        
        config = GolfDomainConfig.from_dict(config_dict)
        
        self.assertEqual(config.domain_name, "golf_test")
        self.assertEqual(config.version, "2.0.0")
        self.assertEqual(config.max_strokes, 8)
        self.assertEqual(config.scoring_method, "match_play")
        self.assertFalse(config.handicap_enabled)
        self.assertEqual(config.num_holes, 9)
        
    def test_save_and_load(self):
        """Test configuration save and load functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            
            # Create and save config
            original_config = GolfDomainConfig(
                max_strokes=15,
                scoring_method="stableford",
                num_holes=9
            )
            original_config.save(config_path)
            
            # Verify file exists
            self.assertTrue(config_path.exists())
            
            # Load config
            loaded_config = GolfDomainConfig.load(config_path)
            
            # Verify loaded config matches original
            self.assertEqual(loaded_config.max_strokes, original_config.max_strokes)
            self.assertEqual(loaded_config.scoring_method, original_config.scoring_method)
            self.assertEqual(loaded_config.num_holes, original_config.num_holes)
            
    def test_load_nonexistent_file_hard_failure(self):
        """Test that loading nonexistent file causes hard failure."""
        nonexistent_path = Path("/nonexistent/config.json")
        
        with self.assertRaises(FileNotFoundError) as context:
            GolfDomainConfig.load(nonexistent_path)
        
        self.assertIn("Configuration file not found", str(context.exception))
        
    def test_score_thresholds(self):
        """Test score threshold configurations."""
        config = GolfDomainConfig()
        
        self.assertEqual(config.good_score_threshold, 0.9)
        self.assertEqual(config.excellent_score_threshold, 0.8)
        
        # Test custom thresholds
        custom_config = GolfDomainConfig(
            good_score_threshold=0.95,
            excellent_score_threshold=0.85
        )
        
        self.assertEqual(custom_config.good_score_threshold, 0.95)
        self.assertEqual(custom_config.excellent_score_threshold, 0.85)
        
    def test_integration_settings(self):
        """Test integration settings configuration."""
        config = GolfDomainConfig()
        
        self.assertFalse(config.enable_ml_predictions)
        self.assertFalse(config.enable_weather_integration)
        
        # Test with integrations enabled
        enabled_config = GolfDomainConfig(
            enable_ml_predictions=True,
            enable_weather_integration=True
        )
        
        self.assertTrue(enabled_config.enable_ml_predictions)
        self.assertTrue(enabled_config.enable_weather_integration)


if __name__ == "__main__":
    unittest.main(verbosity=2)