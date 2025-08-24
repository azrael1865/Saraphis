"""
Comprehensive test suite for AdaptivePrecisionWrapper
Tests all methods, edge cases, performance, and p-adic operations
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import concurrent.futures
from typing import Dict, Any, List
import logging

from compression_systems.padic.adaptive_precision_wrapper import (
    AdaptivePrecisionWrapper,
    AdaptivePrecisionConfig, 
    PrecisionAllocation,
    create_adaptive_wrapper
)


class TestAdaptivePrecisionConfig:
    """Test AdaptivePrecisionConfig dataclass"""
    
    def test_default_config_creation(self):
        """Test default configuration values"""
        config = AdaptivePrecisionConfig()
        
        assert config.prime == 257
        assert config.base_precision == 4
        assert config.min_precision == 2
        assert config.max_precision == 4
        assert config.target_error == 1e-6
        assert config.importance_threshold == 0.1
        assert config.batch_size == 1024
        assert config.enable_gpu_acceleration == True
        assert config.enable_memory_tracking == True
        assert config.enable_dynamic_switching == True
        assert config.compression_priority == 0.5
        assert config.device == 'cpu'
    
    def test_custom_config_creation(self):
        """Test custom configuration"""
        config = AdaptivePrecisionConfig(
            prime=251,
            base_precision=8,
            min_precision=4,
            max_precision=16,
            target_error=1e-8,
            batch_size=2048,
            device='cuda'
        )
        
        assert config.prime == 251
        assert config.base_precision == 8
        assert config.min_precision == 4
        assert config.max_precision == 16
        assert config.target_error == 1e-8
        assert config.batch_size == 2048
        assert config.device == 'cuda'


class TestPrecisionAllocation:
    """Test PrecisionAllocation dataclass"""
    
    def test_precision_allocation_creation(self):
        """Test creation of PrecisionAllocation"""
        weights = [Mock(), Mock()]
        precision_map = torch.tensor([4, 6])
        error_map = torch.tensor([1e-6, 2e-6])
        original_shape = torch.Size([2, 1])
        
        allocation = PrecisionAllocation(
            padic_weights=weights,
            precision_map=precision_map,
            error_map=error_map,
            compression_ratio=0.5,
            bits_used=128,
            original_shape=original_shape
        )
        
        assert len(allocation.padic_weights) == 2
        assert allocation.compression_ratio == 0.5
        assert allocation.bits_used == 128
        assert allocation.original_shape == original_shape
    
    def test_get_average_precision(self):
        """Test get_average_precision method"""
        # Test with weights having precision attribute
        weight1 = Mock()
        weight1.precision = 4
        weight2 = Mock()
        weight2.precision = 6
        
        allocation = PrecisionAllocation(
            padic_weights=[weight1, weight2],
            precision_map=torch.tensor([4, 6]),
            error_map=torch.tensor([1e-6, 2e-6]),
            compression_ratio=0.5,
            bits_used=128,
            original_shape=torch.Size([2])
        )
        
        assert allocation.get_average_precision() == 5.0
    
    def test_get_average_precision_no_weights(self):
        """Test get_average_precision with no weights"""
        allocation = PrecisionAllocation(
            padic_weights=[],
            precision_map=torch.tensor([]),
            error_map=torch.tensor([]),
            compression_ratio=0.0,
            bits_used=0,
            original_shape=torch.Size([0])
        )
        
        assert allocation.get_average_precision() == 0.0
    
    def test_get_average_precision_weights_without_precision(self):
        """Test get_average_precision with weights without precision attribute"""
        weight1 = Mock()
        # No precision attribute
        weight2 = Mock()
        # No precision attribute
        
        allocation = PrecisionAllocation(
            padic_weights=[weight1, weight2],
            precision_map=torch.tensor([4, 6]),
            error_map=torch.tensor([1e-6, 2e-6]),
            compression_ratio=0.5,
            bits_used=128,
            original_shape=torch.Size([2])
        )
        
        assert allocation.get_average_precision() == 0.0
    
    def test_get_error_statistics(self):
        """Test get_error_statistics method"""
        error_map = torch.tensor([[1e-6, 2e-6], [3e-6, 4e-6]])
        
        allocation = PrecisionAllocation(
            padic_weights=[Mock(), Mock(), Mock(), Mock()],
            precision_map=torch.tensor([[4, 5], [6, 7]]),
            error_map=error_map,
            compression_ratio=0.5,
            bits_used=128,
            original_shape=torch.Size([2, 2])
        )
        
        stats = allocation.get_error_statistics()
        assert 'mean' in stats
        assert 'std' in stats
        assert 'max' in stats
        assert 'min' in stats
        assert stats['mean'] == 2.5e-6
        assert stats['max'] == 4e-6
        assert stats['min'] == 1e-6
    
    def test_get_error_statistics_empty(self):
        """Test get_error_statistics with empty error map"""
        allocation = PrecisionAllocation(
            padic_weights=[],
            precision_map=torch.tensor([]),
            error_map=torch.tensor([]),
            compression_ratio=0.0,
            bits_used=0,
            original_shape=torch.Size([0])
        )
        
        stats = allocation.get_error_statistics()
        assert stats['mean'] == 0.0
        assert stats['std'] == 0.0
        assert stats['max'] == 0.0
        assert stats['min'] == 0.0


class TestAdaptivePrecisionWrapperInitialization:
    """Test AdaptivePrecisionWrapper initialization"""
    
    def test_default_initialization(self):
        """Test initialization with default config"""
        wrapper = AdaptivePrecisionWrapper()
        
        assert wrapper.config is not None
        assert wrapper.config.prime == 257
        assert wrapper.math_ops is None
        assert wrapper.device == 'cpu'
        assert len(wrapper._precision_ops_cache) == 0
        assert len(wrapper._precision_math_ops_cache) == 0
    
    def test_custom_config_initialization(self):
        """Test initialization with custom config"""
        config = AdaptivePrecisionConfig(prime=251, base_precision=8)
        mock_math_ops = Mock()
        
        wrapper = AdaptivePrecisionWrapper(config=config, math_ops=mock_math_ops)
        
        assert wrapper.config.prime == 251
        assert wrapper.config.base_precision == 8
        assert wrapper.math_ops is mock_math_ops
        assert wrapper.prime == 251
    
    def test_performance_stats_initialization(self):
        """Test performance stats are properly initialized"""
        wrapper = AdaptivePrecisionWrapper()
        
        stats = wrapper.performance_stats
        assert stats['tensors_processed'] == 0
        assert stats['total_elements'] == 0
        assert stats['vectorized_operations'] == 0
        assert stats['batched_operations'] == 0
        assert stats['serial_operations'] == 0


class TestConvertTensor:
    """Test convert_tensor method (main entry point)"""
    
    def setup_method(self):
        """Set up mock math_ops for testing"""
        self.mock_math_ops = Mock()
        self.mock_math_ops.prime = 257
        self.mock_math_ops.to_padic = Mock(return_value="mock_padic_weight")
        self.mock_math_ops.from_padic = Mock(return_value=1.0)
        
        # Mock create_with_precision method
        precision_ops = Mock()
        precision_ops.to_padic = Mock(return_value="precision_padic_weight")
        precision_ops.from_padic = Mock(return_value=1.0)
        self.mock_math_ops.create_with_precision = Mock(return_value=precision_ops)
    
    def test_convert_small_tensor(self):
        """Test tensor conversion with small tensor (serial processing)"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        # Small tensor that should trigger serial processing
        tensor = torch.tensor([1.0, 2.0, 3.0])
        
        result = wrapper.convert_tensor(tensor)
        
        assert isinstance(result, PrecisionAllocation)
        assert len(result.padic_weights) == 3
        assert result.original_shape == torch.Size([3])
        assert wrapper.performance_stats['tensors_processed'] == 1
        assert wrapper.performance_stats['serial_operations'] == 1
    
    def test_convert_medium_tensor(self):
        """Test tensor conversion with medium tensor (batched processing)"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        # Medium tensor that should trigger batched processing
        tensor = torch.randn(20000)  # Between MAX_SERIAL_ELEMENTS and 100000
        
        result = wrapper.convert_tensor(tensor)
        
        assert isinstance(result, PrecisionAllocation)
        assert len(result.padic_weights) == 20000
        assert result.original_shape == torch.Size([20000])
        assert wrapper.performance_stats['batched_operations'] == 1
    
    def test_convert_large_tensor(self):
        """Test tensor conversion with large tensor (parallel processing)"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        # Large tensor that should trigger parallel processing
        tensor = torch.randn(150000)  # > 100000
        
        result = wrapper.convert_tensor(tensor)
        
        assert isinstance(result, PrecisionAllocation)
        assert len(result.padic_weights) == 150000
        assert result.original_shape == torch.Size([150000])
        assert wrapper.performance_stats['vectorized_operations'] == 1
    
    def test_convert_tensor_with_importance_scores(self):
        """Test tensor conversion with provided importance scores"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        tensor = torch.tensor([1.0, 2.0, 3.0])
        importance_scores = torch.tensor([0.1, 0.5, 0.9])
        
        result = wrapper.convert_tensor(tensor, importance_scores)
        
        assert isinstance(result, PrecisionAllocation)
        assert len(result.padic_weights) == 3
    
    def test_convert_multidimensional_tensor(self):
        """Test conversion of multidimensional tensor"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        result = wrapper.convert_tensor(tensor)
        
        assert result.original_shape == torch.Size([2, 2])
        assert result.error_map.shape == torch.Size([2, 2])
        assert len(result.padic_weights) == 4  # Flattened


class TestProcessingMethods:
    """Test internal processing methods"""
    
    def setup_method(self):
        """Set up mock math_ops for testing"""
        self.mock_math_ops = Mock()
        self.mock_math_ops.prime = 257
        self.mock_math_ops.to_padic = Mock(return_value="mock_padic_weight")
        self.mock_math_ops.from_padic = Mock(return_value=1.0)
        
        # Mock create_with_precision method
        precision_ops = Mock()
        precision_ops.to_padic = Mock(return_value="precision_padic_weight")
        precision_ops.from_padic = Mock(return_value=1.0)
        self.mock_math_ops.create_with_precision = Mock(return_value=precision_ops)
    
    def test_process_serial_optimized(self):
        """Test _process_serial_optimized method"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        flat_tensor = torch.tensor([1.0, 2.0, 3.0])
        precision_allocation = torch.tensor([4, 5, 6])
        
        weights, error_map = wrapper._process_serial_optimized(flat_tensor, precision_allocation)
        
        assert len(weights) == 3
        assert error_map.shape == torch.Size([3])
    
    def test_process_serial_optimized_with_small_values(self):
        """Test serial processing with very small values"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        flat_tensor = torch.tensor([1e-12, 2.0, 0.0])  # Include very small and zero values
        precision_allocation = torch.tensor([4, 5, 6])
        
        weights, error_map = wrapper._process_serial_optimized(flat_tensor, precision_allocation)
        
        assert len(weights) == 3
        assert error_map.shape == torch.Size([3])
    
    def test_process_batched(self):
        """Test _process_batched method"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        # Create tensor larger than batch size to test batching
        size = wrapper.BATCH_SIZE * 2 + 100  # Spans multiple batches
        flat_tensor = torch.randn(size)
        precision_allocation = torch.randint(2, 8, (size,))
        
        weights, error_map = wrapper._process_batched(flat_tensor, precision_allocation)
        
        assert len(weights) == size
        assert error_map.shape == torch.Size([size])
    
    def test_process_batch(self):
        """Test _process_batch method"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        batch_values = torch.tensor([1.0, 2.0, 3.0])
        batch_precisions = torch.tensor([4, 5, 6])
        offset = 0
        
        weights, errors = wrapper._process_batch(batch_values, batch_precisions, offset)
        
        assert len(weights) == 3
        assert errors.shape == torch.Size([3])
    
    def test_process_batch_with_errors(self):
        """Test batch processing with error handling"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        # Mock precision ops to raise an error
        def mock_get_precision_ops(precision):
            ops = Mock()
            ops.to_padic = Mock(side_effect=ValueError("Test error"))
            return ops
        
        wrapper._get_precision_ops = mock_get_precision_ops
        
        batch_values = torch.tensor([1.0, 2.0, 3.0])
        batch_precisions = torch.tensor([4, 5, 6])
        
        weights, errors = wrapper._process_batch(batch_values, batch_precisions, 0)
        
        assert len(weights) == 3  # Should fallback to math_ops
        assert torch.all(errors == wrapper.config.target_error)  # Should use target error
    
    def test_process_parallel_batched(self):
        """Test _process_parallel_batched method"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        size = wrapper.BATCH_SIZE * 3  # Multiple batches
        flat_tensor = torch.randn(size)
        precision_allocation = torch.randint(2, 8, (size,))
        
        weights, error_map = wrapper._process_parallel_batched(flat_tensor, precision_allocation)
        
        assert len(weights) == size
        assert error_map.shape == torch.Size([size])


class TestPrecisionAllocation:
    """Test precision allocation methods"""
    
    def setup_method(self):
        """Set up wrapper for testing"""
        config = AdaptivePrecisionConfig(prime=257, min_precision=2, max_precision=8)
        self.wrapper = AdaptivePrecisionWrapper(config=config)
    
    def test_allocate_precision_by_importance(self):
        """Test precision allocation based on importance"""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        importance_scores = torch.tensor([[0.1, 0.3], [0.5, 0.9]])
        total_bits = 1000
        
        precision_map = self.wrapper.allocate_precision_by_importance(
            tensor, importance_scores, total_bits
        )
        
        assert precision_map.shape == tensor.shape
        assert torch.all(precision_map >= self.wrapper.config.min_precision)
        assert torch.all(precision_map <= self.wrapper.config.max_precision)
    
    def test_allocate_precision_uniform_importance(self):
        """Test precision allocation with uniform importance"""
        tensor = torch.ones(4)
        importance_scores = torch.ones(4)  # Uniform importance
        total_bits = 1000
        
        precision_map = self.wrapper.allocate_precision_by_importance(
            tensor, importance_scores, total_bits
        )
        
        # Should allocate similar precision to all elements
        assert len(torch.unique(precision_map)) <= 2  # Allow some variation due to rounding
    
    def test_allocate_precision_zero_importance(self):
        """Test precision allocation with zero total importance"""
        tensor = torch.ones(4)
        importance_scores = torch.zeros(4)  # No importance
        total_bits = 1000
        
        precision_map = self.wrapper.allocate_precision_by_importance(
            tensor, importance_scores, total_bits
        )
        
        # Should allocate equal precision when importance is zero
        assert len(torch.unique(precision_map)) == 1
    
    @patch('compression_systems.padic.adaptive_precision_wrapper.get_safe_precision')
    def test_allocate_precision_respects_safe_limits(self, mock_get_safe_precision):
        """Test that precision allocation respects safe precision limits"""
        mock_get_safe_precision.return_value = 6  # Safe limit is 6
        
        config = AdaptivePrecisionConfig(prime=257, min_precision=2, max_precision=20)
        wrapper = AdaptivePrecisionWrapper(config=config)
        
        tensor = torch.ones(4)
        importance_scores = torch.ones(4) * 1000  # Very high importance
        total_bits = 10000  # Lots of bits
        
        precision_map = wrapper.allocate_precision_by_importance(
            tensor, importance_scores, total_bits
        )
        
        # Should not exceed safe precision limit of 6
        assert torch.all(precision_map <= 6)


class TestLegacyInterfaces:
    """Test legacy interface methods"""
    
    def setup_method(self):
        """Set up mock math_ops for testing"""
        self.mock_math_ops = Mock()
        self.mock_math_ops.prime = 257
        self.mock_math_ops.to_padic = Mock(return_value="mock_padic_weight")
        self.mock_math_ops.from_padic = Mock(return_value=1.0)
        
        precision_ops = Mock()
        precision_ops.to_padic = Mock(return_value="precision_padic_weight")
        precision_ops.from_padic = Mock(return_value=1.0)
        self.mock_math_ops.create_with_precision = Mock(return_value=precision_ops)
    
    def test_compute_adaptive_precision(self):
        """Test compute_adaptive_precision legacy method"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        tensor = torch.tensor([1.0, 2.0, 3.0])
        target_error = 1e-5
        
        weights, precision_map = wrapper.compute_adaptive_precision(tensor, target_error)
        
        assert len(weights) == 3
        assert precision_map.shape == tensor.shape
    
    def test_batch_compress_with_adaptive_precision(self):
        """Test batch_compress_with_adaptive_precision legacy method"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        tensor = torch.tensor([1.0, 2.0, 3.0])
        importance_scores = torch.tensor([0.1, 0.5, 0.9])
        
        result = wrapper.batch_compress_with_adaptive_precision(tensor, importance_scores)
        
        assert isinstance(result, PrecisionAllocation)
        assert len(result.padic_weights) == 3


class TestHelperMethods:
    """Test helper and utility methods"""
    
    def setup_method(self):
        """Set up mock math_ops for testing"""
        self.mock_math_ops = Mock()
        self.mock_math_ops.prime = 257
        self.mock_math_ops.to_padic = Mock(return_value="mock_zero_weight")
    
    def test_create_zero_weight(self):
        """Test _create_zero_weight method"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        zero_weight = wrapper._create_zero_weight()
        
        assert zero_weight == "mock_zero_weight"
        self.mock_math_ops.to_padic.assert_called_once_with(0.0)
    
    def test_calculate_bits_for_weight_none(self):
        """Test _calculate_bits_for_weight with None weight"""
        wrapper = AdaptivePrecisionWrapper()
        
        bits = wrapper._calculate_bits_for_weight(None)
        
        assert bits == 1
    
    def test_calculate_bits_for_weight_zero(self):
        """Test _calculate_bits_for_weight with zero weight"""
        wrapper = AdaptivePrecisionWrapper()
        
        bits = wrapper._calculate_bits_for_weight(0)
        
        assert bits == 1
    
    def test_calculate_bits_for_weight_with_length(self):
        """Test _calculate_bits_for_weight with weight having length"""
        wrapper = AdaptivePrecisionWrapper()
        
        mock_weight = [1, 2, 3, 4, 5]  # Has __len__
        bits = wrapper._calculate_bits_for_weight(mock_weight)
        
        expected_bits = int(len(mock_weight) * wrapper.bits_per_digit)
        assert bits == expected_bits
    
    def test_calculate_bits_for_weight_default(self):
        """Test _calculate_bits_for_weight with default case"""
        wrapper = AdaptivePrecisionWrapper()
        
        mock_weight = object()  # No __len__
        bits = wrapper._calculate_bits_for_weight(mock_weight)
        
        expected_bits = int(wrapper.bits_per_digit * 4)
        assert bits == expected_bits


class TestMathOpsIntegration:
    """Test math operations integration"""
    
    def test_get_math_ops_for_precision_with_factory(self):
        """Test _get_math_ops_for_precision with factory method"""
        mock_math_ops = Mock()
        precision_specific_ops = Mock()
        mock_math_ops.create_with_precision = Mock(return_value=precision_specific_ops)
        
        wrapper = AdaptivePrecisionWrapper(math_ops=mock_math_ops)
        
        result = wrapper._get_math_ops_for_precision(4)
        
        assert result is precision_specific_ops
        mock_math_ops.create_with_precision.assert_called_once_with(4)
    
    @patch('compression_systems.padic.adaptive_precision_wrapper.PadicMathematicalOperations')
    def test_get_math_ops_for_precision_fallback(self, mock_padic_ops):
        """Test _get_math_ops_for_precision fallback creation"""
        mock_math_ops = Mock()
        mock_math_ops.prime = 257
        # No create_with_precision method
        
        fallback_ops = Mock()
        mock_padic_ops.return_value = fallback_ops
        
        wrapper = AdaptivePrecisionWrapper(math_ops=mock_math_ops)
        
        result = wrapper._get_math_ops_for_precision(4)
        
        assert result is fallback_ops
        mock_padic_ops.assert_called_once_with(257, 4)
    
    def test_get_precision_ops_caching(self):
        """Test that _get_precision_ops caches results"""
        mock_math_ops = Mock()
        precision_ops = Mock()
        precision_ops.to_padic = Mock(return_value="test")
        precision_ops.from_padic = Mock(return_value=1.0)
        mock_math_ops.create_with_precision = Mock(return_value=precision_ops)
        
        wrapper = AdaptivePrecisionWrapper(math_ops=mock_math_ops)
        
        # Call twice
        ops1 = wrapper._get_precision_ops(4)
        ops2 = wrapper._get_precision_ops(4)
        
        # Should be same object (cached)
        assert ops1 is ops2
        # Should only create once
        mock_math_ops.create_with_precision.assert_called_once()
    
    def test_get_precision_ops_type_conversion(self):
        """Test _get_precision_ops handles numpy types"""
        mock_math_ops = Mock()
        precision_ops = Mock()
        precision_ops.to_padic = Mock(return_value="test")
        precision_ops.from_padic = Mock(return_value=1.0)
        mock_math_ops.create_with_precision = Mock(return_value=precision_ops)
        
        wrapper = AdaptivePrecisionWrapper(math_ops=mock_math_ops)
        
        # Test with numpy int64
        numpy_precision = np.int64(4)
        ops = wrapper._get_precision_ops(numpy_precision)
        
        # Should work without error
        assert ops is not None
        mock_math_ops.create_with_precision.assert_called_once_with(4)


class TestPerformanceStats:
    """Test performance statistics tracking"""
    
    def test_get_performance_stats_empty(self):
        """Test get_performance_stats with no operations"""
        wrapper = AdaptivePrecisionWrapper()
        
        stats = wrapper.get_performance_stats()
        
        assert stats['tensors_processed'] == 0
        assert stats['total_elements'] == 0
        assert stats['average_elements_per_tensor'] == 0
        assert stats['processing_strategies']['serial'] == 0
        assert stats['processing_strategies']['batched'] == 0
        assert stats['processing_strategies']['parallel'] == 0
        assert stats['strategy_percentages']['serial'] == 0
        assert stats['strategy_percentages']['batched'] == 0
        assert stats['strategy_percentages']['parallel'] == 0
    
    def test_get_performance_stats_with_operations(self):
        """Test get_performance_stats with operations"""
        wrapper = AdaptivePrecisionWrapper()
        
        # Simulate some operations
        wrapper.performance_stats.update({
            'tensors_processed': 3,
            'total_elements': 1500,
            'serial_operations': 1,
            'batched_operations': 1,
            'vectorized_operations': 1
        })
        
        stats = wrapper.get_performance_stats()
        
        assert stats['tensors_processed'] == 3
        assert stats['total_elements'] == 1500
        assert stats['average_elements_per_tensor'] == 500
        assert stats['processing_strategies']['serial'] == 1
        assert stats['processing_strategies']['batched'] == 1
        assert stats['processing_strategies']['parallel'] == 1
        
        # Each strategy should be 33.33% (100/3)
        expected_percentage = 100 / 3
        assert abs(stats['strategy_percentages']['serial'] - expected_percentage) < 0.01
        assert abs(stats['strategy_percentages']['batched'] - expected_percentage) < 0.01
        assert abs(stats['strategy_percentages']['parallel'] - expected_percentage) < 0.01


class TestFactoryFunction:
    """Test create_adaptive_wrapper factory function"""
    
    @patch('compression_systems.padic.adaptive_precision_wrapper.get_safe_precision')
    def test_create_adaptive_wrapper_default(self, mock_get_safe_precision):
        """Test factory function with default parameters"""
        mock_get_safe_precision.return_value = 16
        
        wrapper = create_adaptive_wrapper()
        
        assert isinstance(wrapper, AdaptivePrecisionWrapper)
        assert wrapper.config.prime == 257
        assert wrapper.config.base_precision == 4
        assert wrapper.config.max_precision == 16
        mock_get_safe_precision.assert_called_once_with(257)
    
    @patch('compression_systems.padic.adaptive_precision_wrapper.get_safe_precision')
    def test_create_adaptive_wrapper_custom_prime(self, mock_get_safe_precision):
        """Test factory function with custom prime"""
        mock_get_safe_precision.return_value = 8
        
        wrapper = create_adaptive_wrapper(prime=251)
        
        assert wrapper.config.prime == 251
        assert wrapper.config.max_precision == 8
        mock_get_safe_precision.assert_called_once_with(251)
    
    @patch('compression_systems.padic.adaptive_precision_wrapper.get_safe_precision')
    def test_create_adaptive_wrapper_with_kwargs(self, mock_get_safe_precision):
        """Test factory function with additional kwargs"""
        mock_get_safe_precision.return_value = 12
        
        wrapper = create_adaptive_wrapper(
            prime=101,
            target_error=1e-8,
            batch_size=512,
            device='cuda'
        )
        
        assert wrapper.config.prime == 101
        assert wrapper.config.target_error == 1e-8
        assert wrapper.config.batch_size == 512
        assert wrapper.config.device == 'cuda'
        assert wrapper.config.max_precision == 12
    
    @patch('compression_systems.padic.adaptive_precision_wrapper.get_safe_precision')
    def test_create_adaptive_wrapper_safe_base_precision(self, mock_get_safe_precision):
        """Test that factory ensures safe base precision"""
        mock_get_safe_precision.return_value = 2  # Very low safe limit
        
        wrapper = create_adaptive_wrapper(prime=13)
        
        # Base precision should not exceed safe limit
        assert wrapper.config.base_precision <= 2
        assert wrapper.config.max_precision == 2


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def setup_method(self):
        """Set up mock math_ops for testing"""
        self.mock_math_ops = Mock()
        self.mock_math_ops.prime = 257
        self.mock_math_ops.to_padic = Mock(return_value="mock_weight")
        self.mock_math_ops.from_padic = Mock(return_value=1.0)
        
        precision_ops = Mock()
        precision_ops.to_padic = Mock(return_value="precision_weight")
        precision_ops.from_padic = Mock(return_value=1.0)
        self.mock_math_ops.create_with_precision = Mock(return_value=precision_ops)
    
    def test_convert_empty_tensor(self):
        """Test conversion of empty tensor"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        tensor = torch.tensor([])
        
        result = wrapper.convert_tensor(tensor)
        
        assert len(result.padic_weights) == 0
        assert result.error_map.numel() == 0
    
    def test_convert_single_element_tensor(self):
        """Test conversion of single element tensor"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        tensor = torch.tensor([5.0])
        
        result = wrapper.convert_tensor(tensor)
        
        assert len(result.padic_weights) == 1
        assert result.error_map.shape == torch.Size([1])
    
    def test_convert_tensor_with_nan(self):
        """Test conversion of tensor containing NaN"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        tensor = torch.tensor([1.0, float('nan'), 3.0])
        
        # Should handle NaN gracefully
        result = wrapper.convert_tensor(tensor)
        
        assert len(result.padic_weights) == 3
        assert result.error_map.shape == torch.Size([3])
    
    def test_convert_tensor_with_inf(self):
        """Test conversion of tensor containing infinity"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        tensor = torch.tensor([1.0, float('inf'), -float('inf')])
        
        # Should handle infinity gracefully
        result = wrapper.convert_tensor(tensor)
        
        assert len(result.padic_weights) == 3
        assert result.error_map.shape == torch.Size([3])
    
    def test_parallel_processing_with_exception(self):
        """Test parallel processing handles exceptions gracefully"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        # Mock _process_batch to raise exception for some batches
        original_process_batch = wrapper._process_batch
        call_count = [0]
        
        def mock_process_batch(*args):
            call_count[0] += 1
            if call_count[0] == 2:  # Second call raises exception
                raise RuntimeError("Test exception")
            return original_process_batch(*args)
        
        wrapper._process_batch = mock_process_batch
        
        # Large tensor to trigger parallel processing
        tensor = torch.randn(150000)
        
        # Should handle exception and fallback
        result = wrapper.convert_tensor(tensor)
        
        assert len(result.padic_weights) == 150000
    
    def test_zero_total_bits_allocation(self):
        """Test precision allocation with zero total bits"""
        wrapper = AdaptivePrecisionWrapper()
        
        tensor = torch.ones(4)
        importance_scores = torch.ones(4)
        total_bits = 0  # Zero bits
        
        precision_map = wrapper.allocate_precision_by_importance(
            tensor, importance_scores, total_bits
        )
        
        # Should still respect minimum precision
        assert torch.all(precision_map >= wrapper.config.min_precision)


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    def setup_method(self):
        """Set up realistic math ops mock"""
        self.mock_math_ops = Mock()
        self.mock_math_ops.prime = 257
        
        # More realistic mock that varies return values
        def mock_to_padic(value):
            if abs(value) < 1e-10:
                return [0]  # Zero representation
            else:
                # Simple mock representation
                return list(range(int(abs(value) % 5) + 1))
        
        def mock_from_padic(weight):
            if not weight or weight == [0]:
                return 0.0
            else:
                # Simple reconstruction with some error
                return len(weight) * 1.1
        
        self.mock_math_ops.to_padic = mock_to_padic
        self.mock_math_ops.from_padic = mock_from_padic
        
        precision_ops = Mock()
        precision_ops.to_padic = mock_to_padic
        precision_ops.from_padic = mock_from_padic
        self.mock_math_ops.create_with_precision = Mock(return_value=precision_ops)
    
    def test_neural_network_layer_compression(self):
        """Test compressing a realistic neural network layer"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        # Simulate a small neural network layer (weights)
        layer_weights = torch.randn(64, 32)  # 64 input, 32 output neurons
        
        # Some weights are more important (larger magnitude)
        layer_weights[0:10, 0:10] *= 10  # High importance region
        
        result = wrapper.convert_tensor(layer_weights)
        
        assert result.original_shape == torch.Size([64, 32])
        assert len(result.padic_weights) == 64 * 32
        assert result.compression_ratio > 0
        assert result.bits_used > 0
        
        # Check error statistics are reasonable
        error_stats = result.get_error_statistics()
        assert error_stats['mean'] >= 0
        assert error_stats['max'] >= error_stats['mean']
    
    def test_batch_processing_multiple_tensors(self):
        """Test processing multiple tensors in sequence"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        tensors = [
            torch.randn(100),      # Small tensor
            torch.randn(15000),    # Medium tensor  
            torch.randn(120000)    # Large tensor
        ]
        
        results = []
        for tensor in tensors:
            result = wrapper.convert_tensor(tensor)
            results.append(result)
        
        # All should succeed
        assert len(results) == 3
        assert all(isinstance(r, PrecisionAllocation) for r in results)
        
        # Performance stats should show different strategies used
        stats = wrapper.get_performance_stats()
        assert stats['tensors_processed'] == 3
        assert stats['serial_operations'] >= 1
        assert stats['batched_operations'] >= 1
        assert stats['vectorized_operations'] >= 1
    
    def test_precision_adaptation_based_on_importance(self):
        """Test that precision adapts based on importance scores"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        # Create tensor with varying importance
        tensor = torch.ones(100)
        importance_scores = torch.zeros(100)
        importance_scores[0:20] = 1.0  # High importance
        importance_scores[20:50] = 0.5  # Medium importance
        importance_scores[50:100] = 0.1  # Low importance
        
        result = wrapper.convert_tensor(tensor, importance_scores)
        
        # High importance elements should get higher precision
        high_importance_precision = result.precision_map[0:20]
        low_importance_precision = result.precision_map[50:100]
        
        # On average, high importance should have higher precision
        assert high_importance_precision.float().mean() > low_importance_precision.float().mean()


class TestRobustness:
    """Test robustness against various inputs and conditions"""
    
    def setup_method(self):
        """Set up mock math ops"""
        self.mock_math_ops = Mock()
        self.mock_math_ops.prime = 257
        self.mock_math_ops.to_padic = Mock(return_value=[1, 2, 3])
        self.mock_math_ops.from_padic = Mock(return_value=1.0)
        
        precision_ops = Mock()
        precision_ops.to_padic = Mock(return_value=[1, 2, 3])
        precision_ops.from_padic = Mock(return_value=1.0)
        self.mock_math_ops.create_with_precision = Mock(return_value=precision_ops)
    
    def test_extreme_precision_values(self):
        """Test with extreme precision values"""
        config = AdaptivePrecisionConfig(min_precision=1, max_precision=1000)
        wrapper = AdaptivePrecisionWrapper(config=config, math_ops=self.mock_math_ops)
        
        tensor = torch.ones(10)
        
        # Should handle extreme values gracefully
        result = wrapper.convert_tensor(tensor)
        
        assert len(result.padic_weights) == 10
    
    def test_memory_stress_large_tensor(self):
        """Test memory handling with large tensors"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        # Very large tensor that tests memory efficiency
        large_tensor = torch.randn(200000)
        
        result = wrapper.convert_tensor(large_tensor)
        
        assert len(result.padic_weights) == 200000
        assert result.original_shape == torch.Size([200000])
    
    def test_concurrent_processing(self):
        """Test thread safety during concurrent processing"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        def process_tensor():
            tensor = torch.randn(1000)
            return wrapper.convert_tensor(tensor)
        
        # Process multiple tensors concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_tensor) for _ in range(10)]
            results = [f.result() for f in futures]
        
        # All should succeed
        assert len(results) == 10
        assert all(isinstance(r, PrecisionAllocation) for r in results)
    
    def test_different_tensor_dtypes(self):
        """Test with different tensor data types"""
        wrapper = AdaptivePrecisionWrapper(math_ops=self.mock_math_ops)
        
        dtypes = [torch.float32, torch.float64, torch.float16]
        
        for dtype in dtypes:
            if dtype == torch.float16:
                # Skip float16 if not supported
                continue
                
            tensor = torch.randn(100, dtype=dtype)
            result = wrapper.convert_tensor(tensor)
            
            assert len(result.padic_weights) == 100
            assert result.original_shape == torch.Size([100])


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])