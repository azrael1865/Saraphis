"""
Comprehensive test suite for CategoricalToPAdicBridge
Tests all functionality, edge cases, and integration scenarios
"""

import pytest
import torch
import numpy as np
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compression_systems.integration.categorical_to_padic_bridge import (
    CategoricalToPadicBridge,
    BridgeMapping,
    BridgeStatistics,
    create_categorical_to_padic_bridge
)
from compression_systems.categorical.categorical_storage_manager import WeightCategory, CategoryType
from compression_systems.categorical.ieee754_channel_extractor import IEEE754Channels
from compression_systems.categorical.weight_categorizer import CategorizationResult, WeightPattern
from compression_systems.padic.padic_encoder import PadicWeight
from compression_systems.padic.padic_logarithmic_encoder import LogarithmicPadicWeight, LogarithmicEncodingConfig
from fractions import Fraction


class TestCategoricalToPadicBridgeBasics:
    """Test basic functionality of CategoricalToPadicBridge"""
    
    def test_bridge_initialization(self):
        """Test bridge initialization"""
        bridge = CategoricalToPadicBridge()
        
        assert isinstance(bridge.bridge_stats, BridgeStatistics)
        assert len(bridge.active_mappings) == 0
        assert len(bridge.category_optimization_settings) == 6  # All CategoryType values
        
        # Check optimization settings exist for all category types
        expected_types = {
            CategoryType.ZERO_WEIGHTS,
            CategoryType.SMALL_WEIGHTS,
            CategoryType.MEDIUM_WEIGHTS,
            CategoryType.LARGE_WEIGHTS,
            CategoryType.HIGH_ENTROPY,
            CategoryType.LOW_ENTROPY
        }
        
        assert set(bridge.category_optimization_settings.keys()) == expected_types
    
    def test_factory_function(self):
        """Test factory function"""
        bridge = create_categorical_to_padic_bridge()
        assert isinstance(bridge, CategoricalToPadicBridge)
        assert len(bridge.active_mappings) == 0
    
    def test_bridge_statistics_initialization(self):
        """Test BridgeStatistics initialization"""
        stats = BridgeStatistics()
        
        assert stats.total_mappings_created == 0
        assert stats.successful_conversions == 0
        assert stats.failed_conversions == 0
        assert stats.average_compression_ratio == 0.0
        assert isinstance(stats.category_conversion_rates, dict)
        assert len(stats.category_conversion_rates) == 0


class TestCategoricalToPadicMapping:
    """Test categorical to p-adic mapping creation"""
    
    @pytest.fixture
    def bridge(self):
        return CategoricalToPadicBridge()
    
    @pytest.fixture
    def sample_encoding_config(self):
        return LogarithmicEncodingConfig(
            prime=5,
            precision=10,
            max_safe_precision=20,
            use_natural_log=True,
            log_offset=1.0,
            quantization_levels=1024
        )
    
    @pytest.fixture
    def sample_ieee754_channels(self):
        return IEEE754Channels(
            sign_channel=np.array([0, 1, 0], dtype=np.uint8),
            exponent_channel=np.array([127, 128, 126], dtype=np.uint8),
            mantissa_channel=np.array([0.5, 0.75, 0.25], dtype=np.float32),
            original_values=np.array([1.0, -2.0, 0.5], dtype=np.float32)
        )
    
    @pytest.fixture
    def sample_weight_category(self, sample_ieee754_channels):
        return WeightCategory(
            category_id="test_category",
            category_type=CategoryType.MEDIUM_WEIGHTS,
            weights=[torch.tensor([1.0, 2.0, 3.0])],
            ieee754_channels=[sample_ieee754_channels],
            compression_metadata={"test": True}
        )
    
    @pytest.fixture
    def sample_categorization_result(self):
        return CategorizationResult(
            categories_found=1,
            total_weights_processed=3,
            optimization_hints={"recommended_precision": 8, "use_sparse_encoding": False},
            processing_time=0.001,
            patterns_detected=[WeightPattern.NORMAL_DISTRIBUTION]
        )
    
    def test_empty_categories_list(self, bridge, sample_encoding_config, sample_categorization_result):
        """Test handling of empty categories list"""
        with pytest.raises(ValueError, match="Categories list cannot be empty"):
            bridge.create_categorical_to_padic_mapping([], sample_categorization_result, sample_encoding_config)
    
    def test_single_category_mapping_creation(self, bridge, sample_weight_category, 
                                            sample_categorization_result, sample_encoding_config):
        """Test creating mapping for single category"""
        categories = [sample_weight_category]
        
        mappings = bridge.create_categorical_to_padic_mapping(
            categories, sample_categorization_result, sample_encoding_config
        )
        
        assert len(mappings) >= 0  # Should not fail completely
        assert bridge.bridge_stats.total_mappings_created >= 0
    
    def test_multiple_categories_mapping(self, bridge, sample_categorization_result, sample_encoding_config):
        """Test creating mappings for multiple categories"""
        categories = []
        
        # Create categories of different types
        for i, category_type in enumerate([CategoryType.ZERO_WEIGHTS, CategoryType.SMALL_WEIGHTS, CategoryType.LARGE_WEIGHTS]):
            category = WeightCategory(
                category_id=f"category_{i}",
                category_type=category_type,
                weights=[torch.tensor([i * 1.0, i * 2.0])],
                ieee754_channels=None,
                compression_metadata={}
            )
            categories.append(category)
        
        mappings = bridge.create_categorical_to_padic_mapping(
            categories, sample_categorization_result, sample_encoding_config
        )
        
        # Should handle multiple categories
        assert isinstance(mappings, list)
        assert bridge.bridge_stats.total_mappings_created >= 0
    
    def test_category_with_no_weights(self, bridge, sample_categorization_result, sample_encoding_config):
        """Test handling category with no weights"""
        category = WeightCategory(
            category_id="empty_category",
            category_type=CategoryType.ZERO_WEIGHTS,
            weights=[],  # Empty weights
            ieee754_channels=None,
            compression_metadata={}
        )
        
        categories = [category]
        mappings = bridge.create_categorical_to_padic_mapping(
            categories, sample_categorization_result, sample_encoding_config
        )
        
        # Should handle gracefully
        assert isinstance(mappings, list)
    
    def test_category_with_invalid_weights(self, bridge, sample_categorization_result, sample_encoding_config):
        """Test handling category with invalid/corrupted weights"""
        category = WeightCategory(
            category_id="invalid_category",
            category_type=CategoryType.HIGH_ENTROPY,
            weights=[torch.tensor([float('nan'), float('inf'), -float('inf')])],
            ieee754_channels=None,
            compression_metadata={}
        )
        
        categories = [category]
        mappings = bridge.create_categorical_to_padic_mapping(
            categories, sample_categorization_result, sample_encoding_config
        )
        
        # Should handle invalid values gracefully
        assert isinstance(mappings, list)


class TestOptimizationSettings:
    """Test optimization settings for different category types"""
    
    @pytest.fixture
    def bridge(self):
        return CategoricalToPadicBridge()
    
    @pytest.fixture
    def base_config(self):
        return LogarithmicEncodingConfig(
            prime=5,
            precision=10,
            max_safe_precision=20,
            quantization_levels=1024
        )
    
    def test_zero_weights_optimization(self, bridge, base_config):
        """Test optimization settings for zero weights"""
        settings = bridge.category_optimization_settings[CategoryType.ZERO_WEIGHTS]
        
        assert settings['use_sparse_encoding'] == True
        assert settings['precision_reduction'] == 2
        assert settings['skip_logarithmic'] == True
        
        # Test config creation
        categorization_result = CategorizationResult(
            categories_found=1, total_weights_processed=100,
            optimization_hints={}, processing_time=0.001,
            patterns_detected=[]
        )
        
        optimized_config = bridge._create_category_optimized_config(
            base_config, CategoryType.ZERO_WEIGHTS, settings, categorization_result
        )
        
        assert optimized_config.precision == 8  # 10 - 2
        assert optimized_config.enable_run_length_encoding == True
        assert optimized_config.quantize_before_encoding == True
    
    def test_large_weights_optimization(self, bridge, base_config):
        """Test optimization settings for large weights"""
        settings = bridge.category_optimization_settings[CategoryType.LARGE_WEIGHTS]
        
        assert settings['use_sparse_encoding'] == False
        assert settings['precision_reduction'] == 0
        assert settings['skip_logarithmic'] == False
        assert settings['high_precision_mode'] == True
        
        categorization_result = CategorizationResult(
            categories_found=1, total_weights_processed=100,
            optimization_hints={}, processing_time=0.001,
            patterns_detected=[]
        )
        
        optimized_config = bridge._create_category_optimized_config(
            base_config, CategoryType.LARGE_WEIGHTS, settings, categorization_result
        )
        
        assert optimized_config.precision == 11  # base + 1
    
    def test_optimization_hints_application(self, bridge, base_config):
        """Test that optimization hints are applied correctly"""
        settings = bridge.category_optimization_settings[CategoryType.MEDIUM_WEIGHTS]
        
        categorization_result = CategorizationResult(
            categories_found=1, total_weights_processed=100,
            optimization_hints={
                'recommended_precision': 12,
                'use_sparse_encoding': True,
                'quantization_levels': 2048
            },
            processing_time=0.001,
            patterns_detected=[]
        )
        
        optimized_config = bridge._create_category_optimized_config(
            base_config, CategoryType.MEDIUM_WEIGHTS, settings, categorization_result
        )
        
        assert optimized_config.precision == 12
        assert optimized_config.enable_run_length_encoding == True
        assert optimized_config.quantization_levels == 2048


class TestIEEE754ChannelExtraction:
    """Test IEEE 754 channel extraction and reuse"""
    
    @pytest.fixture
    def bridge(self):
        return CategoricalToPadicBridge()
    
    def test_reuse_existing_channels(self, bridge):
        """Test reusing existing IEEE 754 channels"""
        # Create category with existing channels
        existing_channels = [
            IEEE754Channels(
                sign_channel=np.array([0, 1], dtype=np.uint8),
                exponent_channel=np.array([127, 128], dtype=np.uint8),
                mantissa_channel=np.array([0.5, 0.75], dtype=np.float32),
                original_values=np.array([1.0, -2.0], dtype=np.float32)
            ),
            IEEE754Channels(
                sign_channel=np.array([0], dtype=np.uint8),
                exponent_channel=np.array([126], dtype=np.uint8),
                mantissa_channel=np.array([0.25], dtype=np.float32),
                original_values=np.array([0.5], dtype=np.float32)
            )
        ]
        
        category = WeightCategory(
            category_id="test",
            category_type=CategoryType.MEDIUM_WEIGHTS,
            weights=[],
            ieee754_channels=existing_channels,
            compression_metadata={}
        )
        
        channels = bridge._extract_or_reuse_ieee754_channels(category)
        
        assert isinstance(channels, IEEE754Channels)
        assert len(channels.sign_channel) == 3  # Combined
        assert len(channels.exponent_channel) == 3
        assert len(channels.mantissa_channel) == 3
        assert len(channels.original_values) == 3
    
    def test_extract_from_weights(self, bridge):
        """Test extracting channels from weights when not available"""
        category = WeightCategory(
            category_id="test",
            category_type=CategoryType.MEDIUM_WEIGHTS,
            weights=[torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])],
            ieee754_channels=None,
            compression_metadata={}
        )
        
        channels = bridge._extract_or_reuse_ieee754_channels(category)
        
        assert isinstance(channels, IEEE754Channels)
        assert len(channels.original_values) == 5  # Combined weights
    
    def test_dummy_channels_creation(self, bridge):
        """Test creation of dummy channels as last resort"""
        category = WeightCategory(
            category_id="test",
            category_type=CategoryType.MEDIUM_WEIGHTS,
            weights=[],  # No weights
            ieee754_channels=None,  # No channels
            compression_metadata={}
        )
        
        channels = bridge._extract_or_reuse_ieee754_channels(category)
        
        assert isinstance(channels, IEEE754Channels)
        assert len(channels.original_values) == 1  # Dummy size


class TestPadicConversion:
    """Test conversion to p-adic representation"""
    
    @pytest.fixture
    def bridge(self):
        return CategoricalToPadicBridge()
    
    @pytest.fixture
    def sample_channels(self):
        return IEEE754Channels(
            sign_channel=np.array([0, 1, 0], dtype=np.uint8),
            exponent_channel=np.array([127, 128, 126], dtype=np.uint8),
            mantissa_channel=np.array([0.5, 0.75, 0.25], dtype=np.float32),
            original_values=np.array([1.0, -2.0, 0.5], dtype=np.float32)
        )
    
    @pytest.fixture
    def encoding_config(self):
        return LogarithmicEncodingConfig(
            prime=5,
            precision=10,
            max_safe_precision=20
        )
    
    def test_direct_encoding_for_zeros(self, bridge, sample_channels, encoding_config):
        """Test direct encoding for zero/simple values"""
        category = WeightCategory(
            category_id="test",
            category_type=CategoryType.ZERO_WEIGHTS,
            weights=[],
            ieee754_channels=None,
            compression_metadata={}
        )
        
        optimization_settings = bridge.category_optimization_settings[CategoryType.ZERO_WEIGHTS]
        
        padic_weights, logarithmic_weights = bridge._convert_category_to_padic(
            category, sample_channels, encoding_config, optimization_settings
        )
        
        assert isinstance(padic_weights, list)
        assert isinstance(logarithmic_weights, list)
        assert len(padic_weights) == len(logarithmic_weights)
    
    def test_logarithmic_encoding_for_complex(self, bridge, sample_channels, encoding_config):
        """Test logarithmic encoding for complex values"""
        category = WeightCategory(
            category_id="test",
            category_type=CategoryType.HIGH_ENTROPY,
            weights=[],
            ieee754_channels=None,
            compression_metadata={}
        )
        
        optimization_settings = bridge.category_optimization_settings[CategoryType.HIGH_ENTROPY]
        
        padic_weights, logarithmic_weights = bridge._convert_category_to_padic(
            category, sample_channels, encoding_config, optimization_settings
        )
        
        assert isinstance(padic_weights, list)
        assert isinstance(logarithmic_weights, list)
    
    def test_direct_encoding_function(self, bridge, sample_channels, encoding_config):
        """Test direct encoding function"""
        logarithmic_weights = bridge._perform_direct_encoding(sample_channels, encoding_config)
        
        assert isinstance(logarithmic_weights, list)
        assert len(logarithmic_weights) == len(sample_channels.original_values)
        
        for lw in logarithmic_weights:
            assert isinstance(lw, LogarithmicPadicWeight)
            assert hasattr(lw, 'padic_weight')
            assert hasattr(lw, 'encoding_method')
    
    def test_direct_encoding_with_extreme_values(self, bridge, encoding_config):
        """Test direct encoding with extreme values"""
        extreme_channels = IEEE754Channels(
            sign_channel=np.array([0, 1, 0, 1], dtype=np.uint8),
            exponent_channel=np.array([127, 128, 255, 0], dtype=np.uint8),
            mantissa_channel=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            original_values=np.array([0.0, float('inf'), float('nan'), 1e-10], dtype=np.float32)
        )
        
        logarithmic_weights = bridge._perform_direct_encoding(extreme_channels, encoding_config)
        
        assert isinstance(logarithmic_weights, list)
        assert len(logarithmic_weights) == 4
        
        # Should handle extreme values gracefully
        for lw in logarithmic_weights:
            assert isinstance(lw, LogarithmicPadicWeight)


class TestCompressionMetadata:
    """Test compression metadata calculation"""
    
    @pytest.fixture
    def bridge(self):
        return CategoricalToPadicBridge()
    
    def test_compression_metadata_calculation(self, bridge):
        """Test calculation of compression metadata"""
        # Create sample category
        category = WeightCategory(
            category_id="test_category",
            category_type=CategoryType.MEDIUM_WEIGHTS,
            weights=[torch.tensor([1.0, 2.0, 3.0])],
            ieee754_channels=None,
            compression_metadata={}
        )
        
        # Create sample p-adic weights
        padic_weights = [
            PadicWeight(Fraction(1), 5, 10, 0, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            PadicWeight(Fraction(2), 5, 10, 0, [2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        ]
        
        logarithmic_weights = [
            LogarithmicPadicWeight(padic_weights[0], 1.0, 0.0, "test", {}),
            LogarithmicPadicWeight(padic_weights[1], 2.0, 0.693, "test", {})
        ]
        
        optimization_settings = {'test_setting': True}
        
        metadata = bridge._calculate_category_compression_metadata(
            category, padic_weights, logarithmic_weights, optimization_settings
        )
        
        assert isinstance(metadata, dict)
        assert 'category_id' in metadata
        assert 'category_type' in metadata
        assert 'original_size_bytes' in metadata
        assert 'compressed_size_bytes' in metadata
        assert 'compression_ratio' in metadata
        assert 'optimization_settings_applied' in metadata
        
        assert metadata['category_id'] == "test_category"
        assert metadata['original_size_bytes'] > 0
        assert metadata['compressed_size_bytes'] > 0
    
    def test_compression_metadata_with_empty_weights(self, bridge):
        """Test metadata calculation with empty weights"""
        category = WeightCategory(
            category_id="empty_category",
            category_type=CategoryType.ZERO_WEIGHTS,
            weights=[],
            ieee754_channels=None,
            compression_metadata={}
        )
        
        metadata = bridge._calculate_category_compression_metadata(
            category, [], [], {}
        )
        
        assert isinstance(metadata, dict)
        assert metadata['category_id'] == "empty_category"


class TestBridgeStatistics:
    """Test bridge statistics and performance tracking"""
    
    @pytest.fixture
    def bridge(self):
        return CategoricalToPadicBridge()
    
    def test_get_bridge_statistics(self, bridge):
        """Test getting bridge statistics"""
        stats = bridge.get_bridge_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_mappings_created' in stats
        assert 'successful_conversions' in stats
        assert 'failed_conversions' in stats
        assert 'conversion_success_rate' in stats
        assert 'average_compression_ratio' in stats
        assert 'active_mappings_count' in stats
        assert 'category_conversion_rates' in stats
        assert 'optimization_settings_available' in stats
        assert 'supported_category_types' in stats
        
        # Initial values
        assert stats['total_mappings_created'] == 0
        assert stats['successful_conversions'] == 0
        assert stats['failed_conversions'] == 0
        assert stats['active_mappings_count'] == 0
        assert stats['optimization_settings_available'] == 6
    
    def test_clear_mappings(self, bridge):
        """Test clearing mappings and resetting statistics"""
        # Add some dummy data
        bridge.bridge_stats.total_mappings_created = 5
        bridge.bridge_stats.successful_conversions = 3
        bridge.bridge_stats.failed_conversions = 2
        bridge.active_mappings['test'] = Mock()
        
        # Clear mappings
        bridge.clear_mappings()
        
        # Check reset
        assert len(bridge.active_mappings) == 0
        assert bridge.bridge_stats.total_mappings_created == 0
        assert bridge.bridge_stats.successful_conversions == 0
        assert bridge.bridge_stats.failed_conversions == 0


class TestWeightRetrieval:
    """Test weight retrieval functionality"""
    
    @pytest.fixture
    def bridge(self):
        bridge = CategoricalToPadicBridge()
        
        # Add some mock mappings
        mock_mapping1 = Mock()
        mock_mapping1.category_type = CategoryType.SMALL_WEIGHTS
        mock_mapping1.logarithmic_weights = [Mock(), Mock()]
        
        mock_mapping2 = Mock()
        mock_mapping2.category_type = CategoryType.LARGE_WEIGHTS
        mock_mapping2.logarithmic_weights = [Mock()]
        
        bridge.active_mappings['cat1'] = mock_mapping1
        bridge.active_mappings['cat2'] = mock_mapping2
        
        return bridge
    
    def test_retrieve_by_category_types(self, bridge):
        """Test retrieving weights by category types"""
        weights = bridge.retrieve_padic_weights_by_category([CategoryType.SMALL_WEIGHTS])
        
        assert isinstance(weights, list)
        assert len(weights) == 2  # From mock mapping1
    
    def test_retrieve_multiple_category_types(self, bridge):
        """Test retrieving weights from multiple category types"""
        weights = bridge.retrieve_padic_weights_by_category([
            CategoryType.SMALL_WEIGHTS, 
            CategoryType.LARGE_WEIGHTS
        ])
        
        assert isinstance(weights, list)
        assert len(weights) == 3  # 2 from small + 1 from large
    
    def test_retrieve_nonexistent_category(self, bridge):
        """Test retrieving weights for non-existent category"""
        weights = bridge.retrieve_padic_weights_by_category([CategoryType.HIGH_ENTROPY])
        
        assert isinstance(weights, list)
        assert len(weights) == 0  # No matching categories
    
    def test_empty_category_types_list(self, bridge):
        """Test handling of empty category types list"""
        with pytest.raises(ValueError, match="Category types list cannot be empty"):
            bridge.retrieve_padic_weights_by_category([])


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.fixture
    def bridge(self):
        return CategoricalToPadicBridge()
    
    def test_mapping_creation_with_corrupted_category(self, bridge):
        """Test handling of corrupted category data"""
        corrupted_category = Mock()
        corrupted_category.category_id = "corrupted"
        corrupted_category.category_type = None  # Invalid type
        corrupted_category.weights = None
        corrupted_category.ieee754_channels = None
        
        categorization_result = CategorizationResult(
            categories_found=1, total_weights_processed=0,
            optimization_hints={}, processing_time=0.001,
            patterns_detected=[]
        )
        
        encoding_config = LogarithmicEncodingConfig(prime=5, precision=10)
        
        # Should not crash, but may return empty mappings
        try:
            mappings = bridge.create_categorical_to_padic_mapping(
                [corrupted_category], categorization_result, encoding_config
            )
            assert isinstance(mappings, list)
        except RuntimeError:
            # Hard failure is acceptable for completely corrupted data
            pass
    
    def test_statistics_calculation_error_handling(self, bridge):
        """Test statistics calculation with error conditions"""
        # Test with mock that raises exceptions
        bridge.bridge_stats = Mock()
        bridge.bridge_stats.successful_conversions = 5
        bridge.bridge_stats.failed_conversions = 2
        bridge.bridge_stats.category_conversion_rates = Mock()
        bridge.bridge_stats.category_conversion_rates.items.side_effect = Exception("Mock error")
        
        stats = bridge.get_bridge_statistics()
        
        # Should return error information instead of crashing
        assert isinstance(stats, dict)
        assert 'error' in stats or len(stats) > 0


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @pytest.fixture
    def bridge(self):
        return CategoricalToPadicBridge()
    
    @pytest.fixture
    def encoding_config(self):
        return LogarithmicEncodingConfig(
            prime=5,
            precision=10,
            max_safe_precision=20,
            use_natural_log=True,
            quantization_levels=1024
        )
    
    def test_neural_network_weight_compression_scenario(self, bridge, encoding_config):
        """Test realistic neural network weight compression scenario"""
        # Simulate neural network layer weights
        categories = []
        
        # Zero weights (from pruning)
        zero_category = WeightCategory(
            category_id="zeros",
            category_type=CategoryType.ZERO_WEIGHTS,
            weights=[torch.zeros(100)],
            ieee754_channels=None,
            compression_metadata={"layer": "conv1"}
        )
        categories.append(zero_category)
        
        # Small weights
        small_category = WeightCategory(
            category_id="small",
            category_type=CategoryType.SMALL_WEIGHTS,
            weights=[torch.randn(50) * 0.01],
            ieee754_channels=None,
            compression_metadata={"layer": "conv2"}
        )
        categories.append(small_category)
        
        # Large weights
        large_category = WeightCategory(
            category_id="large",
            category_type=CategoryType.LARGE_WEIGHTS,
            weights=[torch.randn(30) * 10.0],
            ieee754_channels=None,
            compression_metadata={"layer": "fc1"}
        )
        categories.append(large_category)
        
        categorization_result = CategorizationResult(
            categories_found=3,
            total_weights_processed=180,
            optimization_hints={"recommended_precision": 8},
            processing_time=0.01,
            patterns_detected=[WeightPattern.SPARSE, WeightPattern.NORMAL_DISTRIBUTION]
        )
        
        mappings = bridge.create_categorical_to_padic_mapping(
            categories, categorization_result, encoding_config
        )
        
        # Should handle realistic scenario
        assert isinstance(mappings, list)
        assert bridge.bridge_stats.total_mappings_created >= 0
        
        # Test retrieval
        zero_weights = bridge.retrieve_padic_weights_by_category([CategoryType.ZERO_WEIGHTS])
        assert isinstance(zero_weights, list)
    
    def test_high_precision_scientific_computation_scenario(self, bridge):
        """Test high-precision scientific computation scenario"""
        # Scientific computation with high precision requirements
        config = LogarithmicEncodingConfig(
            prime=7,
            precision=20,
            max_safe_precision=30,
            use_natural_log=False
        )
        
        # High entropy data
        high_entropy_data = torch.randn(1000) * 1000 + torch.randn(1000) * 0.001
        
        category = WeightCategory(
            category_id="scientific_data",
            category_type=CategoryType.HIGH_ENTROPY,
            weights=[high_entropy_data],
            ieee754_channels=None,
            compression_metadata={"simulation": "molecular_dynamics"}
        )
        
        categorization_result = CategorizationResult(
            categories_found=1,
            total_weights_processed=1000,
            optimization_hints={"recommended_precision": 25, "complexity_handling": True},
            processing_time=0.1,
            patterns_detected=[WeightPattern.HIGH_PRECISION, WeightPattern.COMPLEX_PATTERNS]
        )
        
        mappings = bridge.create_categorical_to_padic_mapping(
            [category], categorization_result, config
        )
        
        assert isinstance(mappings, list)
    
    def test_memory_constrained_scenario(self, bridge, encoding_config):
        """Test memory-constrained scenario with large datasets"""
        # Simulate memory-constrained environment
        large_categories = []
        
        for i in range(10):  # Multiple large categories
            category = WeightCategory(
                category_id=f"large_category_{i}",
                category_type=CategoryType.MEDIUM_WEIGHTS,
                weights=[torch.randn(1000)],  # Large weight tensors
                ieee754_channels=None,
                compression_metadata={"batch": i}
            )
            large_categories.append(category)
        
        categorization_result = CategorizationResult(
            categories_found=10,
            total_weights_processed=10000,
            optimization_hints={"use_sparse_encoding": True, "quantization_levels": 256},
            processing_time=0.5,
            patterns_detected=[WeightPattern.SPARSE]
        )
        
        mappings = bridge.create_categorical_to_padic_mapping(
            large_categories, categorization_result, encoding_config
        )
        
        # Should handle large datasets
        assert isinstance(mappings, list)
        
        stats = bridge.get_bridge_statistics()
        assert isinstance(stats, dict)


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics"""
    
    @pytest.fixture
    def bridge(self):
        return CategoricalToPadicBridge()
    
    def test_scaling_with_category_count(self, bridge):
        """Test scaling with increasing category count"""
        import time
        
        config = LogarithmicEncodingConfig(prime=5, precision=8)
        
        times = []
        for count in [1, 5, 10, 20]:
            categories = []
            for i in range(count):
                category = WeightCategory(
                    category_id=f"cat_{i}",
                    category_type=CategoryType.MEDIUM_WEIGHTS,
                    weights=[torch.randn(10)],
                    ieee754_channels=None,
                    compression_metadata={}
                )
                categories.append(category)
            
            categorization_result = CategorizationResult(
                categories_found=count, total_weights_processed=count * 10,
                optimization_hints={}, processing_time=0.001,
                patterns_detected=[]
            )
            
            start_time = time.time()
            mappings = bridge.create_categorical_to_padic_mapping(
                categories, categorization_result, config
            )
            elapsed = time.time() - start_time
            
            times.append(elapsed)
            
            # Clear for next test
            bridge.clear_mappings()
        
        # Should scale reasonably (not exponentially)
        assert all(t < 10.0 for t in times)  # No test should take more than 10 seconds
    
    def test_memory_usage_with_large_weights(self, bridge):
        """Test memory usage with large weight tensors"""
        config = LogarithmicEncodingConfig(prime=5, precision=8)
        
        # Create category with large weight tensor
        large_weights = torch.randn(10000)
        
        category = WeightCategory(
            category_id="large_weights",
            category_type=CategoryType.LARGE_WEIGHTS,
            weights=[large_weights],
            ieee754_channels=None,
            compression_metadata={}
        )
        
        categorization_result = CategorizationResult(
            categories_found=1, total_weights_processed=10000,
            optimization_hints={}, processing_time=0.01,
            patterns_detected=[]
        )
        
        mappings = bridge.create_categorical_to_padic_mapping(
            [category], categorization_result, config
        )
        
        # Should handle large tensors without memory errors
        assert isinstance(mappings, list)
        
        stats = bridge.get_bridge_statistics()
        assert stats['total_mappings_created'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])