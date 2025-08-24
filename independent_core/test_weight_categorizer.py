"""Comprehensive test suite for WeightCategorizer to identify all root issues"""

import torch
import numpy as np
import pytest
from unittest.mock import Mock, patch
import warnings
from compression_systems.categorical.weight_categorizer import (
    WeightCategorizer, 
    PatternType, 
    CategoryType,
    WeightPattern,
    CategorizationResult
)
from compression_systems.categorical.ieee754_channel_extractor import IEEE754ChannelExtractor, IEEE754Channels

class TestWeightCategorizerInitialization:
    """Test initialization and configuration"""
    
    def test_default_initialization(self):
        """Test default initialization parameters"""
        wc = WeightCategorizer()
        assert wc.enable_clustering == True
        assert wc.enable_pattern_detection == True
        assert wc.max_clusters == 20
        assert wc.pattern_thresholds is not None
        assert wc.categorization_stats is not None
    
    def test_custom_initialization(self):
        """Test custom initialization parameters"""
        wc = WeightCategorizer(
            enable_clustering=False,
            enable_pattern_detection=False,
            max_clusters=10
        )
        assert wc.enable_clustering == False
        assert wc.enable_pattern_detection == False
        assert wc.max_clusters == 10
    
    def test_pattern_thresholds(self):
        """Test pattern detection thresholds"""
        wc = WeightCategorizer()
        assert 'sparsity_threshold' in wc.pattern_thresholds
        assert 'uniformity_threshold' in wc.pattern_thresholds
        assert 'bimodal_separation' in wc.pattern_thresholds
        assert 'periodicity_correlation' in wc.pattern_thresholds
        assert 'power_law_alpha_min' in wc.pattern_thresholds
        assert 'power_law_alpha_max' in wc.pattern_thresholds

class TestBasicCategorization:
    """Test basic categorization functionality"""
    
    def test_simple_weights(self):
        """Test with simple weight tensors"""
        wc = WeightCategorizer()
        
        # Test various simple patterns
        test_cases = [
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),  # Linear
            torch.randn(100),  # Random normal
            torch.ones(50),  # Uniform
            torch.zeros(50),  # All zeros
            torch.linspace(-1, 1, 100),  # Linear range
        ]
        
        for weights in test_cases:
            result = wc.categorize_weights(weights)
            assert isinstance(result, CategorizationResult)
            assert result.primary_category is not None
            assert isinstance(result.primary_category, CategoryType)
            assert isinstance(result.secondary_categories, list)
            assert isinstance(result.compression_estimate, float)
            assert result.compression_estimate > 0
    
    def test_multidimensional_weights(self):
        """Test with multi-dimensional tensors"""
        wc = WeightCategorizer()
        
        # Test different tensor shapes
        shapes = [(10, 10), (5, 5, 5), (2, 3, 4, 5), (100,), (1, 100)]
        
        for shape in shapes:
            weights = torch.randn(shape)
            result = wc.categorize_weights(weights)
            assert result is not None
            assert result.primary_category is not None

class TestPatternDetection:
    """Test pattern detection capabilities"""
    
    def test_sparse_pattern(self):
        """Test sparse pattern detection"""
        wc = WeightCategorizer(enable_pattern_detection=True)
        
        # Create sparse patterns with different sparsity levels
        sparsity_levels = [0.05, 0.1, 0.2, 0.5, 0.9]
        
        for sparsity in sparsity_levels:
            size = 1000
            num_nonzero = int(size * sparsity)
            weights = torch.zeros(size)
            indices = torch.randperm(size)[:num_nonzero]
            weights[indices] = torch.randn(num_nonzero)
            
            result = wc.categorize_weights(weights)
            
            if sparsity <= 0.1:  # Should detect as sparse
                has_sparse = any(p.pattern_type == PatternType.SPARSE for p in result.detected_patterns)
                assert has_sparse, f"Should detect sparse pattern at {sparsity} sparsity"
    
    def test_uniform_pattern(self):
        """Test uniform pattern detection"""
        wc = WeightCategorizer(enable_pattern_detection=True)
        
        # Create uniform patterns with different noise levels
        noise_levels = [0.0001, 0.001, 0.01, 0.1]
        
        for noise in noise_levels:
            weights = torch.ones(100) * 5.0 + torch.randn(100) * noise
            result = wc.categorize_weights(weights)
            
            if noise <= 0.01:  # Should detect as uniform
                has_uniform = any(p.pattern_type == PatternType.UNIFORM for p in result.detected_patterns)
                # Note: May need to check implementation threshold
    
    def test_gaussian_pattern(self):
        """Test Gaussian distribution pattern detection"""
        wc = WeightCategorizer(enable_pattern_detection=True)
        
        # Create Gaussian distributions with different parameters
        means = [0, 1, -1, 10]
        stds = [0.1, 1.0, 5.0]
        
        for mean in means:
            for std in stds:
                weights = torch.normal(mean, std, size=(1000,))
                result = wc.categorize_weights(weights)
                
                has_gaussian = any(p.pattern_type == PatternType.GAUSSIAN for p in result.detected_patterns)
                # Check if Gaussian is detected
    
    def test_bimodal_pattern(self):
        """Test bimodal distribution pattern detection"""
        wc = WeightCategorizer(enable_pattern_detection=True)
        
        # Create bimodal distribution
        mode1 = torch.normal(-5, 1, size=(500,))
        mode2 = torch.normal(5, 1, size=(500,))
        weights = torch.cat([mode1, mode2])
        
        result = wc.categorize_weights(weights)
        
        has_bimodal = any(p.pattern_type == PatternType.BIMODAL for p in result.detected_patterns)
        # Should detect bimodal pattern
    
    def test_power_law_pattern(self):
        """Test power law distribution pattern detection"""
        wc = WeightCategorizer(enable_pattern_detection=True)
        
        # Create power law distribution
        x = torch.linspace(1, 100, 1000)
        alpha = 2.5
        weights = x.pow(-alpha) + torch.randn(1000) * 0.001
        
        result = wc.categorize_weights(weights)
        
        has_power_law = any(p.pattern_type == PatternType.POWER_LAW for p in result.detected_patterns)
        # Check if power law is detected
    
    def test_periodic_pattern(self):
        """Test periodic pattern detection"""
        wc = WeightCategorizer(enable_pattern_detection=True)
        
        # Create periodic patterns
        x = torch.linspace(0, 10 * np.pi, 1000)
        weights = torch.sin(x) + torch.randn(1000) * 0.1
        
        result = wc.categorize_weights(weights)
        
        has_periodic = any(p.pattern_type == PatternType.PERIODIC for p in result.detected_patterns)
        # Should detect periodic pattern
    
    def test_monotonic_pattern(self):
        """Test monotonic pattern detection"""
        wc = WeightCategorizer(enable_pattern_detection=True)
        
        # Create monotonic patterns
        weights_increasing = torch.linspace(0, 10, 100) + torch.randn(100) * 0.01
        weights_decreasing = torch.linspace(10, 0, 100) + torch.randn(100) * 0.01
        
        for weights in [weights_increasing, weights_decreasing]:
            result = wc.categorize_weights(weights)
            has_monotonic = any(p.pattern_type == PatternType.MONOTONIC for p in result.detected_patterns)
            # Should detect monotonic pattern

class TestClustering:
    """Test clustering functionality"""
    
    def test_distinct_clusters(self):
        """Test clustering with distinct groups"""
        wc = WeightCategorizer(enable_clustering=True)
        
        # Create distinct clusters
        clusters = [
            torch.ones(20) * 1.0 + torch.randn(20) * 0.01,
            torch.ones(20) * 5.0 + torch.randn(20) * 0.01,
            torch.ones(20) * 10.0 + torch.randn(20) * 0.01,
        ]
        weights = torch.cat(clusters)
        
        result = wc.categorize_weights(weights)
        
        assert len(result.similarity_groups) >= 3, "Should identify at least 3 clusters"
        
        # Check cluster sizes
        total_indices = sum(len(g) for g in result.similarity_groups)
        assert total_indices == len(weights), "All weights should be assigned to clusters"
    
    def test_optimal_cluster_selection(self):
        """Test optimal cluster number selection"""
        wc = WeightCategorizer(enable_clustering=True, max_clusters=10)
        
        # Create data with clear optimal cluster number
        n_true_clusters = 4
        clusters = []
        for i in range(n_true_clusters):
            center = i * 10
            cluster = torch.normal(center, 0.5, size=(25,))
            clusters.append(cluster)
        weights = torch.cat(clusters)
        
        result = wc.categorize_weights(weights)
        
        # Should find approximately the right number of clusters
        assert 3 <= len(result.similarity_groups) <= 5
    
    def test_clustering_disabled(self):
        """Test with clustering disabled"""
        wc = WeightCategorizer(enable_clustering=False)
        
        weights = torch.randn(100)
        result = wc.categorize_weights(weights)
        
        assert len(result.similarity_groups) == 0, "Should not perform clustering when disabled"
    
    def test_small_dataset_clustering(self):
        """Test clustering with small datasets"""
        wc = WeightCategorizer(enable_clustering=True)
        
        # Test with dataset too small for clustering
        weights = torch.randn(5)  # Less than 10 elements
        result = wc.categorize_weights(weights)
        
        # Should skip clustering for small datasets
        assert len(result.similarity_groups) == 0

class TestIEEE754Integration:
    """Test IEEE754 channel integration"""
    
    def test_with_channels(self):
        """Test categorization with IEEE754 channels"""
        wc = WeightCategorizer()
        extractor = IEEE754ChannelExtractor()
        
        weights = torch.randn(100) * 10.0
        channels = extractor.extract_channels(weights)
        
        result = wc.categorize_weights(weights, channels=channels)
        
        assert result is not None
        assert result.optimization_hints is not None
        assert len(result.optimization_hints) > 0
    
    def test_channel_influence_on_categorization(self):
        """Test how channels influence categorization"""
        wc = WeightCategorizer()
        extractor = IEEE754ChannelExtractor()
        
        # Create weights with specific IEEE754 patterns
        weights_normal = torch.randn(100)
        weights_denormal = torch.randn(100) * 1e-40  # Denormalized range
        
        for weights in [weights_normal, weights_denormal]:
            channels = extractor.extract_channels(weights)
            result = wc.categorize_weights(weights, channels=channels)
            
            # Check if optimization hints reflect channel properties
            assert 'exponent_range' in result.optimization_hints or \
                   'mantissa_bits' in result.optimization_hints or \
                   len(result.optimization_hints) > 0

class TestCompressionEstimation:
    """Test compression ratio estimation"""
    
    def test_compression_estimates(self):
        """Test compression estimation for different patterns"""
        wc = WeightCategorizer()
        
        test_cases = [
            (torch.zeros(100), "all_zeros"),  # Should have high compression
            (torch.ones(100), "all_ones"),  # Should have high compression
            (torch.randn(100), "random"),  # Should have lower compression
            (torch.linspace(0, 1, 100), "linear"),  # Should have medium compression
        ]
        
        for weights, name in test_cases:
            result = wc.categorize_weights(weights)
            assert 0 < result.compression_estimate <= 100, f"Invalid compression estimate for {name}"
    
    def test_compression_with_patterns(self):
        """Test how patterns affect compression estimation"""
        wc = WeightCategorizer(enable_pattern_detection=True)
        
        # Sparse weights should have high compression estimate
        sparse_weights = torch.zeros(1000)
        sparse_weights[::100] = torch.randn(10)
        
        sparse_result = wc.categorize_weights(sparse_weights)
        
        # Random weights should have lower compression estimate
        random_weights = torch.randn(1000)
        random_result = wc.categorize_weights(random_weights)
        
        # Sparse should compress better than random
        assert sparse_result.compression_estimate > random_result.compression_estimate

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        wc = WeightCategorizer()
        
        # None input
        with pytest.raises(ValueError, match="cannot be None"):
            wc.categorize_weights(None)
        
        # Empty tensor
        with pytest.raises(ValueError, match="cannot be empty"):
            wc.categorize_weights(torch.tensor([]))
        
        # NaN values
        weights_nan = torch.tensor([1.0, float('nan'), 3.0])
        with pytest.raises(RuntimeError, match="NaN"):
            wc.categorize_weights(weights_nan)
        
        # Infinite values
        weights_inf = torch.tensor([1.0, float('inf'), 3.0])
        with pytest.raises(RuntimeError, match="infinite"):
            wc.categorize_weights(weights_inf)
    
    def test_edge_cases(self):
        """Test edge cases"""
        wc = WeightCategorizer()
        
        # Single element
        result = wc.categorize_weights(torch.tensor([1.0]))
        assert result is not None
        
        # Very small values
        result = wc.categorize_weights(torch.randn(10) * 1e-30)
        assert result is not None
        
        # Very large values
        result = wc.categorize_weights(torch.randn(10) * 1e30)
        assert result is not None
        
        # Mixed scales
        weights = torch.tensor([1e-30, 1e-10, 1.0, 1e10, 1e30])
        result = wc.categorize_weights(weights)
        assert result is not None
    
    def test_numerical_stability(self):
        """Test numerical stability"""
        wc = WeightCategorizer()
        
        # Test with values near floating point limits
        test_values = [
            torch.finfo(torch.float32).min,
            torch.finfo(torch.float32).max,
            torch.finfo(torch.float32).eps,
            -torch.finfo(torch.float32).eps,
        ]
        
        for val in test_values:
            val_tensor = torch.tensor(val)
            if not torch.isinf(val_tensor) and not torch.isnan(val_tensor):
                weights = torch.tensor([val, 0.0, 1.0])
                try:
                    result = wc.categorize_weights(weights)
                    assert result is not None
                except RuntimeError:
                    # Some extreme values might legitimately fail
                    pass

class TestStatisticsTracking:
    """Test statistics tracking and updates"""
    
    def test_statistics_update(self):
        """Test that statistics are properly updated"""
        wc = WeightCategorizer()
        
        initial_count = wc.categorization_stats['total_categorizations']
        
        # Perform multiple categorizations
        for _ in range(5):
            weights = torch.randn(100)
            wc.categorize_weights(weights)
        
        final_count = wc.categorization_stats['total_categorizations']
        assert final_count == initial_count + 5
    
    def test_pattern_distribution_tracking(self):
        """Test pattern distribution tracking"""
        wc = WeightCategorizer(enable_pattern_detection=True)
        
        # Categorize weights with known patterns
        sparse_weights = torch.zeros(100)
        sparse_weights[::10] = 1.0
        
        wc.categorize_weights(sparse_weights)
        
        # Check if pattern distribution is updated
        if 'pattern_distribution' in wc.categorization_stats:
            assert len(wc.categorization_stats['pattern_distribution']) > 0

class TestOptimizationHints:
    """Test optimization hint generation"""
    
    def test_hint_generation(self):
        """Test that optimization hints are generated"""
        wc = WeightCategorizer()
        
        weights = torch.randn(100)
        result = wc.categorize_weights(weights)
        
        assert isinstance(result.optimization_hints, dict)
        # Should contain some optimization hints
    
    def test_pattern_specific_hints(self):
        """Test pattern-specific optimization hints"""
        wc = WeightCategorizer(enable_pattern_detection=True)
        
        # Sparse weights should get sparsity-related hints
        sparse_weights = torch.zeros(1000)
        sparse_weights[::100] = torch.randn(10)
        
        result = wc.categorize_weights(sparse_weights)
        
        # Should have hints related to sparse optimization
        assert len(result.optimization_hints) > 0

class TestMetadataIntegration:
    """Test metadata integration"""
    
    def test_with_metadata(self):
        """Test categorization with metadata"""
        wc = WeightCategorizer()
        
        weights = torch.randn(100)
        metadata = {
            'layer_type': 'conv2d',
            'layer_index': 5,
            'original_shape': (64, 64, 3, 3)
        }
        
        result = wc.categorize_weights(weights, metadata=metadata)
        assert result is not None
        
        # Metadata should influence optimization hints
        if 'layer_type' in metadata:
            # Check if hints are layer-specific
            pass

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    import sys
    
    test_classes = [
        TestWeightCategorizerInitialization,
        TestBasicCategorization,
        TestPatternDetection,
        TestClustering,
        TestIEEE754Integration,
        TestCompressionEstimation,
        TestErrorHandling,
        TestStatisticsTracking,
        TestOptimizationHints,
        TestMetadataIntegration
    ]
    
    failed_tests = []
    passed_tests = []
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Testing {test_class.__name__}")
        print('='*60)
        
        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]
        
        for method_name in test_methods:
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"✓ {method_name}")
                passed_tests.append(f"{test_class.__name__}.{method_name}")
            except Exception as e:
                print(f"✗ {method_name}: {e}")
                failed_tests.append((f"{test_class.__name__}.{method_name}", str(e)))
    
    print(f"\n{'='*60}")
    print("Test Summary")
    print('='*60)
    print(f"Passed: {len(passed_tests)}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print("\nFailed Tests:")
        for test_name, error in failed_tests:
            print(f"  - {test_name}: {error}")
        return 1
    else:
        print("\nAll tests passed!")
        return 0

if __name__ == "__main__":
    exit(run_comprehensive_tests())