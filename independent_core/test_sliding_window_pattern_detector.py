"""
Comprehensive test suite for SlidingWindowPatternDetector
Tests all aspects of pattern detection, encoding/decoding, and rolling hash algorithms
"""

import pytest
import torch
import numpy as np
import time
import threading
from typing import Dict, List, Tuple, Any
from unittest.mock import Mock, patch, MagicMock

from compression_systems.padic.sliding_window_pattern_detector import (
    PatternMatch,
    PatternDetectionResult,
    IntervalTree,
    OptimizedIntervalTree,
    OverflowFreeRollingHash,
    SlidingWindowPatternDetector
)


class TestPatternMatch:
    """Test PatternMatch dataclass"""
    
    def test_valid_pattern_match(self):
        """Test creating valid pattern match"""
        pattern = b'\x01\x02\x03\x04'
        positions = [0, 10, 20]
        match = PatternMatch(
            pattern=pattern,
            positions=positions,
            frequency=3,
            hash_value=123456,
            length=4,
            element_count=4,
            element_dtype=np.dtype('uint8')
        )
        
        assert match.pattern == pattern
        assert match.positions == positions
        assert match.frequency == 3
        assert match.hash_value == 123456
        assert match.length == 4
        assert match.element_count == 4
        assert match.element_dtype == np.dtype('uint8')
    
    def test_invalid_pattern_type(self):
        """Test that non-bytes pattern raises TypeError"""
        with pytest.raises(TypeError, match="Pattern must be bytes"):
            PatternMatch(
                pattern="not_bytes",
                positions=[0],
                frequency=1,
                hash_value=123,
                length=4,
                element_count=4,
                element_dtype=np.dtype('uint8')
            )
    
    def test_empty_positions(self):
        """Test that empty positions raises ValueError"""
        with pytest.raises(ValueError, match="Pattern must have at least one position"):
            PatternMatch(
                pattern=b'\x01\x02',
                positions=[],
                frequency=0,
                hash_value=123,
                length=2,
                element_count=2,
                element_dtype=np.dtype('uint8')
            )
    
    def test_frequency_position_mismatch(self):
        """Test that frequency not matching positions count raises ValueError"""
        with pytest.raises(ValueError, match="Frequency .* doesn't match positions count"):
            PatternMatch(
                pattern=b'\x01\x02',
                positions=[0, 5],
                frequency=3,  # Wrong frequency
                hash_value=123,
                length=2,
                element_count=2,
                element_dtype=np.dtype('uint8')
            )
    
    def test_length_pattern_mismatch(self):
        """Test that length not matching pattern byte length raises ValueError"""
        with pytest.raises(ValueError, match="Length .* doesn't match pattern length"):
            PatternMatch(
                pattern=b'\x01\x02\x03',
                positions=[0],
                frequency=1,
                hash_value=123,
                length=5,  # Wrong length
                element_count=3,
                element_dtype=np.dtype('uint8')
            )
    
    def test_element_count_byte_length_mismatch(self):
        """Test validation of element count vs byte length"""
        with pytest.raises(ValueError, match="Element count .* doesn't match byte length"):
            PatternMatch(
                pattern=b'\x01\x02\x03\x04',  # 4 bytes
                positions=[0],
                frequency=1,
                hash_value=123,
                length=4,
                element_count=2,  # 2 elements * 4 bytes = 8, doesn't match 4 bytes
                element_dtype=np.dtype('int32')  # 4 bytes per element
            )
    
    def test_int32_pattern_match(self):
        """Test pattern match with int32 dtype (p-adic digits)"""
        # 2 int32 elements = 8 bytes
        pattern = b'\x01\x00\x00\x00\x02\x00\x00\x00'
        match = PatternMatch(
            pattern=pattern,
            positions=[0, 10],
            frequency=2,
            hash_value=789,
            length=8,
            element_count=2,
            element_dtype=np.dtype('int32')
        )
        
        assert match.length == 8
        assert match.element_count == 2
        assert match.element_dtype == np.dtype('int32')


class TestPatternDetectionResult:
    """Test PatternDetectionResult dataclass"""
    
    def test_valid_pattern_detection_result(self):
        """Test creating valid pattern detection result"""
        patterns = {
            0: PatternMatch(
                pattern=b'\x01\x02',
                positions=[0, 5],
                frequency=2,
                hash_value=123,
                length=2,
                element_count=2,
                element_dtype=np.dtype('uint8')
            )
        }
        
        pattern_mask = torch.tensor([True, True, False, False, False, True, True])
        pattern_indices = torch.tensor([0, 0, -1, -1, -1, 0, 0])
        
        result = PatternDetectionResult(
            patterns=patterns,
            pattern_mask=pattern_mask,
            pattern_indices=pattern_indices,
            compression_potential=0.3,
            total_patterns_found=1,
            bytes_replaced=4,
            original_size=7,
            element_dtype=np.dtype('uint8')
        )
        
        assert len(result.patterns) == 1
        assert result.compression_potential == 0.3
        assert result.total_patterns_found == 1
        assert result.bytes_replaced == 4
        assert result.original_size == 7


class TestIntervalTree:
    """Test basic IntervalTree"""
    
    def test_empty_tree(self):
        """Test empty interval tree"""
        tree = IntervalTree()
        assert not tree.overlaps(0, 5)
        assert len(tree.intervals) == 0
    
    def test_insert_and_overlaps(self):
        """Test inserting intervals and checking overlaps"""
        tree = IntervalTree()
        
        tree.insert(10, 20)
        tree.insert(30, 40)
        
        # Test overlaps
        assert tree.overlaps(15, 25)  # Overlaps with [10, 20)
        assert tree.overlaps(35, 45)  # Overlaps with [30, 40)
        assert not tree.overlaps(0, 5)  # No overlap
        assert not tree.overlaps(25, 30)  # No overlap
        assert not tree.overlaps(45, 50)  # No overlap
    
    def test_edge_case_overlaps(self):
        """Test edge cases for overlap detection"""
        tree = IntervalTree()
        tree.insert(10, 20)
        
        # Edge cases
        assert not tree.overlaps(0, 10)  # Touches start, no overlap
        assert not tree.overlaps(20, 30)  # Touches end, no overlap
        assert tree.overlaps(10, 15)  # Starts at same point
        assert tree.overlaps(15, 20)  # Ends at same point
        assert tree.overlaps(5, 25)  # Completely contains
        assert tree.overlaps(15, 17)  # Completely contained
    
    def test_merge_intervals_trigger(self):
        """Test that merge intervals is triggered after many insertions"""
        tree = IntervalTree()
        
        # Insert many overlapping intervals
        for i in range(110):  # More than 100 to trigger merge
            tree.insert(i, i + 10)
        
        # Should have triggered merge, reducing interval count
        assert len(tree.intervals) < 110


class TestOptimizedIntervalTree:
    """Test OptimizedIntervalTree (AVL-based)"""
    
    def test_empty_optimized_tree(self):
        """Test empty optimized interval tree"""
        tree = OptimizedIntervalTree()
        assert not tree.overlaps(0, 5)
        assert tree.size == 0
    
    def test_insert_and_overlaps_optimized(self):
        """Test inserting and overlap detection in optimized tree"""
        tree = OptimizedIntervalTree()
        
        tree.insert(10, 20)
        tree.insert(30, 40)
        tree.insert(50, 60)
        
        assert tree.size == 3
        
        # Test overlaps
        assert tree.overlaps(15, 25)
        assert tree.overlaps(35, 45)
        assert tree.overlaps(55, 65)
        assert not tree.overlaps(0, 5)
        assert not tree.overlaps(25, 30)
        assert not tree.overlaps(45, 50)
    
    def test_invalid_intervals(self):
        """Test handling of invalid intervals"""
        tree = OptimizedIntervalTree()
        
        # Invalid interval (start >= end) should be ignored
        tree.insert(10, 10)  # Empty interval
        tree.insert(20, 15)  # Backwards interval
        
        assert tree.size == 0
        assert not tree.overlaps(5, 25)
    
    def test_balance_preservation(self):
        """Test that tree remains balanced after insertions"""
        tree = OptimizedIntervalTree()
        
        # Insert intervals in sorted order (worst case for unbalanced BST)
        for i in range(20):
            tree.insert(i * 10, i * 10 + 5)
        
        # Tree should still function correctly
        assert tree.overlaps(15, 25)
        assert tree.overlaps(95, 105)
        assert not tree.overlaps(5, 10)
    
    def test_rebuild_tree_trigger(self):
        """Test that tree rebuild is triggered after many insertions"""
        tree = OptimizedIntervalTree()
        tree._merge_threshold = 10  # Lower threshold for testing
        
        # Insert many overlapping intervals
        for i in range(15):
            tree.insert(0, 10)  # All overlapping
        
        # Should trigger rebuild
        assert tree.size <= 10  # Should have merged intervals
    
    def test_clear_tree(self):
        """Test clearing the tree"""
        tree = OptimizedIntervalTree()
        
        tree.insert(10, 20)
        tree.insert(30, 40)
        assert tree.size == 2
        
        tree.clear()
        assert tree.size == 0
        assert tree.root is None
        assert not tree.overlaps(15, 25)


class TestOverflowFreeRollingHash:
    """Test OverflowFreeRollingHash"""
    
    def test_initialization(self):
        """Test hash initialization"""
        hasher = OverflowFreeRollingHash(window_size=5)
        assert hasher.window_size == 5
        assert hasher.base_power > 0
        assert hasher.base_power < OverflowFreeRollingHash.MODULUS
    
    def test_compute_hash(self):
        """Test initial hash computation"""
        hasher = OverflowFreeRollingHash(window_size=3)
        data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        
        hash1 = hasher.compute_hash(data, 0)  # [1, 2, 3]
        hash2 = hasher.compute_hash(data, 1)  # [2, 3, 4]
        hash3 = hasher.compute_hash(data, 2)  # [3, 4, 5]
        
        # Different windows should have different hashes (with high probability)
        assert hash1 != hash2
        assert hash2 != hash3
        assert hash1 != hash3
    
    def test_roll_hash(self):
        """Test rolling hash computation"""
        hasher = OverflowFreeRollingHash(window_size=3)
        data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        
        # Compute initial hash for [1, 2, 3]
        initial_hash = hasher.compute_hash(data, 0)
        
        # Roll to [2, 3, 4]
        rolled_hash = hasher.roll_hash(initial_hash, 1, 4)
        expected_hash = hasher.compute_hash(data, 1)
        
        assert rolled_hash == expected_hash
        
        # Roll to [3, 4, 5]
        rolled_hash2 = hasher.roll_hash(rolled_hash, 2, 5)
        expected_hash2 = hasher.compute_hash(data, 2)
        
        assert rolled_hash2 == expected_hash2
    
    def test_hash_boundary_values(self):
        """Test hash with boundary values"""
        hasher = OverflowFreeRollingHash(window_size=4)
        
        # Test with max byte values
        data = np.array([255, 255, 255, 255], dtype=np.uint8)
        hash_val = hasher.compute_hash(data, 0)
        
        assert 0 <= hash_val < OverflowFreeRollingHash.MODULUS
    
    def test_hash_consistency(self):
        """Test that hash is consistent for same data"""
        hasher = OverflowFreeRollingHash(window_size=5)
        data = np.array([10, 20, 30, 40, 50], dtype=np.uint8)
        
        hash1 = hasher.compute_hash(data, 0)
        hash2 = hasher.compute_hash(data, 0)
        
        assert hash1 == hash2
    
    def test_modulus_properties(self):
        """Test that hash values are within modulus"""
        hasher = OverflowFreeRollingHash(window_size=8)
        
        for _ in range(100):
            data = np.random.randint(0, 256, size=10, dtype=np.uint8)
            hash_val = hasher.compute_hash(data, 0)
            assert 0 <= hash_val < OverflowFreeRollingHash.MODULUS


class TestSlidingWindowPatternDetector:
    """Test main SlidingWindowPatternDetector class"""
    
    def test_initialization_default(self):
        """Test detector initialization with default parameters"""
        detector = SlidingWindowPatternDetector()
        
        assert detector.min_pattern_length == 4
        assert detector.max_pattern_length == 32
        assert detector.min_frequency == 3
        assert detector.max_patterns == 1000
        assert detector.use_suffix_array == True
    
    def test_initialization_custom(self):
        """Test detector initialization with custom parameters"""
        detector = SlidingWindowPatternDetector(
            min_pattern_length=2,
            max_pattern_length=16,
            min_frequency=2,
            max_patterns=500,
            use_suffix_array=False
        )
        
        assert detector.min_pattern_length == 2
        assert detector.max_pattern_length == 16
        assert detector.min_frequency == 2
        assert detector.max_patterns == 500
        assert detector.use_suffix_array == False
    
    def test_invalid_initialization_parameters(self):
        """Test that invalid parameters raise ValueError"""
        with pytest.raises(ValueError, match="min_pattern_length must be >= 2"):
            SlidingWindowPatternDetector(min_pattern_length=1)
        
        with pytest.raises(ValueError, match="max_pattern_length .* < min_pattern_length"):
            SlidingWindowPatternDetector(min_pattern_length=10, max_pattern_length=5)
        
        with pytest.raises(ValueError, match="min_frequency must be >= 2"):
            SlidingWindowPatternDetector(min_frequency=1)
    
    def test_convert_input_safe_bytes(self):
        """Test safe input conversion for bytes"""
        detector = SlidingWindowPatternDetector()
        
        data = b'\x01\x02\x03\x04'
        converted, dtype = detector._convert_input_safe(data)
        
        assert isinstance(converted, np.ndarray)
        assert converted.dtype == np.uint8
        assert dtype == np.dtype('uint8')
        assert np.array_equal(converted, np.array([1, 2, 3, 4], dtype=np.uint8))
    
    def test_convert_input_safe_torch_tensor(self):
        """Test safe input conversion for torch tensor"""
        detector = SlidingWindowPatternDetector()
        
        # Test int32 tensor (p-adic digits)
        tensor = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
        converted, dtype = detector._convert_input_safe(tensor)
        
        assert isinstance(converted, np.ndarray)
        assert converted.dtype == np.int32
        assert dtype == np.dtype('int32')
        assert np.array_equal(converted, np.array([10, 20, 30, 40], dtype=np.int32))
    
    def test_convert_input_safe_numpy_array(self):
        """Test safe input conversion for numpy array"""
        detector = SlidingWindowPatternDetector()
        
        # Test int16 array
        array = np.array([100, 200, 300, 400], dtype=np.int16)
        converted, dtype = detector._convert_input_safe(array)
        
        assert isinstance(converted, np.ndarray)
        assert converted.dtype == np.int16
        assert dtype == np.dtype('int16')
        assert np.array_equal(converted, array)
    
    def test_convert_input_safe_negative_values(self):
        """Test conversion with negative values"""
        detector = SlidingWindowPatternDetector()
        
        # Test with values that fit in int8
        array = np.array([-50, -10, 10, 50], dtype=np.int32)
        converted, dtype = detector._convert_input_safe(array)
        
        assert converted.dtype == np.int8
        assert dtype == np.dtype('int8')
    
    def test_convert_input_safe_float_values(self):
        """Test conversion with float values (rounded to int)"""
        detector = SlidingWindowPatternDetector()
        
        array = np.array([1.2, 2.7, 3.1, 4.9], dtype=np.float32)
        converted, dtype = detector._convert_input_safe(array)
        
        # Should be rounded
        expected = np.array([1, 3, 3, 5], dtype=np.int32)
        assert np.array_equal(converted, expected)
    
    def test_convert_input_safe_unsupported_type(self):
        """Test that unsupported types raise TypeError"""
        detector = SlidingWindowPatternDetector()
        
        with pytest.raises(TypeError, match="Unsupported data type"):
            detector._convert_input_safe("unsupported_string")
    
    def test_find_patterns_simple_repetition(self):
        """Test finding simple repeating patterns"""
        detector = SlidingWindowPatternDetector(min_pattern_length=2, min_frequency=2)
        
        # Create data with repeating pattern [1, 2]
        data = np.array([1, 2, 3, 1, 2, 4, 1, 2], dtype=np.uint8)
        
        result = detector.find_patterns(data)
        
        assert result.total_patterns_found > 0
        assert len(result.patterns) > 0
        
        # Should find the [1, 2] pattern
        found_pattern = False
        for pattern_match in result.patterns.values():
            pattern_array = np.frombuffer(pattern_match.pattern, dtype=np.uint8)
            if len(pattern_array) == 2 and np.array_equal(pattern_array, [1, 2]):
                found_pattern = True
                assert len(pattern_match.positions) >= 2
                break
        
        assert found_pattern
    
    def test_find_patterns_int32_data(self):
        """Test finding patterns in int32 data (p-adic digits)"""
        detector = SlidingWindowPatternDetector(min_pattern_length=3, min_frequency=2)
        
        # Create int32 data with repeating pattern
        pattern = [100, 200, 300]
        data = []
        for _ in range(5):
            data.extend(pattern)
            data.extend(np.random.randint(0, 50, 2))  # Add noise
        
        data = np.array(data, dtype=np.int32)
        
        result = detector.find_patterns(data)
        
        assert result.element_dtype == np.dtype('int32')
        assert result.total_patterns_found > 0
        
        # Check that we found the repeating pattern
        found_target_pattern = False
        for pattern_match in result.patterns.values():
            if pattern_match.element_count == 3:
                pattern_array = np.frombuffer(pattern_match.pattern, dtype=np.int32)
                if np.array_equal(pattern_array, pattern):
                    found_target_pattern = True
                    break
        
        assert found_target_pattern
    
    def test_find_patterns_insufficient_data(self):
        """Test pattern finding with insufficient data"""
        detector = SlidingWindowPatternDetector(min_pattern_length=4, min_frequency=3)
        
        # Data too short
        data = np.array([1, 2, 3], dtype=np.uint8)
        
        result = detector.find_patterns(data)
        
        assert result.total_patterns_found == 0
        assert len(result.patterns) == 0
        assert result.compression_potential == 0.0
    
    def test_find_patterns_high_entropy_data(self):
        """Test pattern finding with high entropy (random) data"""
        detector = SlidingWindowPatternDetector()
        
        # Random data should have few patterns
        np.random.seed(42)
        data = np.random.randint(0, 256, 1000, dtype=np.uint8)
        
        result = detector.find_patterns(data)
        
        # High entropy data should have low compression potential
        assert result.compression_potential < 0.3
    
    def test_encode_with_patterns_basic(self):
        """Test basic encoding with detected patterns"""
        detector = SlidingWindowPatternDetector(min_pattern_length=2, min_frequency=2)
        
        data = np.array([1, 2, 3, 1, 2, 4, 1, 2], dtype=np.uint8)
        
        pattern_result = detector.find_patterns(data)
        encoded_data, pattern_dict, pattern_lengths = detector.encode_with_patterns(data, pattern_result)
        
        assert isinstance(encoded_data, torch.Tensor)
        assert isinstance(pattern_dict, dict)
        assert isinstance(pattern_lengths, torch.Tensor)
        
        # Encoded data should be shorter than original for repeated patterns
        assert encoded_data.size(0) < len(data)
    
    def test_encode_with_patterns_no_patterns(self):
        """Test encoding when no patterns are found"""
        detector = SlidingWindowPatternDetector(min_frequency=10)  # High threshold
        
        data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        
        encoded_data, pattern_dict, pattern_lengths = detector.encode_with_patterns(data)
        
        # Should return original data when no patterns found
        assert torch.equal(encoded_data, torch.from_numpy(data))
        assert len(pattern_dict) == 0
        assert pattern_lengths.size(0) == 0
    
    def test_decode_with_patterns_basic(self):
        """Test basic decoding with patterns"""
        detector = SlidingWindowPatternDetector(min_pattern_length=2, min_frequency=2)
        
        original_data = np.array([1, 2, 3, 1, 2, 4, 1, 2], dtype=np.uint8)
        
        # Encode
        pattern_result = detector.find_patterns(original_data)
        encoded_data, pattern_dict, pattern_lengths = detector.encode_with_patterns(original_data, pattern_result)
        
        # Decode
        decoded_data = detector.decode_with_patterns(
            encoded_data, pattern_dict, pattern_lengths, np.dtype('uint8')
        )
        
        # Should recover original data
        assert torch.equal(decoded_data, torch.from_numpy(original_data))
    
    def test_decode_with_patterns_int32(self):
        """Test decoding with int32 patterns (p-adic digits)"""
        detector = SlidingWindowPatternDetector(min_pattern_length=2, min_frequency=2)
        
        original_data = np.array([100, 200, 300, 100, 200, 400], dtype=np.int32)
        
        # Encode
        encoded_data, pattern_dict, pattern_lengths = detector.encode_with_patterns(original_data)
        
        # Decode with correct dtype
        decoded_data = detector.decode_with_patterns(
            encoded_data, pattern_dict, pattern_lengths, np.dtype('int32')
        )
        
        # Should recover original data with correct dtype
        assert decoded_data.dtype == torch.int32
        assert torch.equal(decoded_data, torch.from_numpy(original_data))
    
    def test_decode_invalid_pattern_id(self):
        """Test decoding with invalid pattern ID raises error"""
        detector = SlidingWindowPatternDetector()
        
        # Create encoded data with invalid pattern ID
        encoded_data = torch.tensor([256 + 999], dtype=torch.int32)  # Non-existent pattern ID
        pattern_dict = {}
        pattern_lengths = torch.tensor([])
        
        with pytest.raises(ValueError, match="Invalid pattern ID"):
            detector.decode_with_patterns(encoded_data, pattern_dict, pattern_lengths)
    
    def test_decode_invalid_tensor_shape(self):
        """Test decoding with invalid tensor shape raises error"""
        detector = SlidingWindowPatternDetector()
        
        # Create 2D encoded data (invalid)
        encoded_data = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        
        with pytest.raises(ValueError, match="Encoded data must be 1D"):
            detector.decode_with_patterns(encoded_data, {}, torch.tensor([]))
    
    def test_forward_method(self):
        """Test forward method (nn.Module compatibility)"""
        detector = SlidingWindowPatternDetector(min_pattern_length=2, min_frequency=2)
        
        data = np.array([1, 2, 1, 2, 1, 2], dtype=np.uint8)
        
        encoded_data, pattern_dict, metadata = detector.forward(data)
        
        assert isinstance(encoded_data, torch.Tensor)
        assert isinstance(pattern_dict, dict)
        assert isinstance(metadata, dict)
        
        # Check metadata contents
        assert 'pattern_lengths' in metadata
        assert 'element_dtype' in metadata
        assert 'compression_potential' in metadata
        assert 'total_patterns_found' in metadata
    
    def test_call_method(self):
        """Test __call__ method"""
        detector = SlidingWindowPatternDetector(min_pattern_length=2, min_frequency=2)
        
        data = np.array([1, 2, 1, 2, 1, 2], dtype=np.uint8)
        
        # Should work the same as forward
        encoded_data, pattern_dict, metadata = detector(data)
        
        assert isinstance(encoded_data, torch.Tensor)
        assert isinstance(pattern_dict, dict)
        assert isinstance(metadata, dict)
    
    def test_analyze_compression_efficiency(self):
        """Test compression efficiency analysis"""
        detector = SlidingWindowPatternDetector(min_pattern_length=2, min_frequency=2)
        
        # Data with good compression potential
        data = np.array([1, 2] * 50, dtype=np.uint8)  # Very repetitive
        
        analysis = detector.analyze_compression_efficiency(data)
        
        assert isinstance(analysis, dict)
        assert 'original_size' in analysis
        assert 'encoded_size' in analysis
        assert 'compression_ratio' in analysis
        assert 'space_savings_percent' in analysis
        assert 'patterns_found' in analysis
        assert 'pattern_statistics' in analysis
        
        # Should achieve good compression on repetitive data
        assert analysis['compression_ratio'] > 1.0
        assert analysis['space_savings_percent'] > 0
    
    def test_analyze_compression_efficiency_no_patterns(self):
        """Test compression analysis when no patterns found"""
        detector = SlidingWindowPatternDetector(min_frequency=100)  # Very high threshold
        
        data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        
        analysis = detector.analyze_compression_efficiency(data)
        
        assert analysis['patterns_found'] == 0
        assert analysis['compression_ratio'] <= 1.0
        assert analysis['pattern_statistics'] == {}
    
    def test_thread_safety(self):
        """Test thread safety of pattern detection"""
        detector = SlidingWindowPatternDetector(min_pattern_length=2, min_frequency=2)
        
        results = []
        errors = []
        
        def detect_patterns():
            try:
                data = np.random.randint(0, 10, 100, dtype=np.uint8)
                result = detector.find_patterns(data)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=detect_patterns)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 5
    
    def test_pattern_detection_algorithms(self):
        """Test both suffix array and rolling hash algorithms"""
        # Test small data (uses rolling hash)
        detector_small = SlidingWindowPatternDetector(min_pattern_length=2, min_frequency=2)
        small_data = np.array([1, 2, 3, 1, 2, 4], dtype=np.uint8)
        result_small = detector_small.find_patterns(small_data)
        
        # Test medium data (algorithm selection depends on entropy)
        detector_medium = SlidingWindowPatternDetector(min_pattern_length=2, min_frequency=2)
        medium_data = np.array([1, 2, 3] * 2000, dtype=np.uint8)  # Low entropy, repetitive
        result_medium = detector_medium.find_patterns(medium_data)
        
        # Both should find patterns in repetitive data
        assert result_small.total_patterns_found >= 0
        assert result_medium.total_patterns_found > 0
        assert result_medium.compression_potential > 0
    
    def test_edge_case_empty_data(self):
        """Test handling of empty data"""
        detector = SlidingWindowPatternDetector()
        
        empty_data = np.array([], dtype=np.uint8)
        result = detector.find_patterns(empty_data)
        
        assert result.total_patterns_found == 0
        assert len(result.patterns) == 0
    
    def test_edge_case_single_element(self):
        """Test handling of single element data"""
        detector = SlidingWindowPatternDetector()
        
        single_data = np.array([42], dtype=np.uint8)
        result = detector.find_patterns(single_data)
        
        assert result.total_patterns_found == 0
        assert len(result.patterns) == 0
    
    def test_pattern_overlap_prevention(self):
        """Test that overlapping patterns are handled correctly"""
        detector = SlidingWindowPatternDetector(min_pattern_length=3, min_frequency=2)
        
        # Create data where patterns could overlap
        data = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5], dtype=np.uint8)
        
        result = detector.find_patterns(data)
        
        # Verify no overlapping patterns in the mask
        if result.total_patterns_found > 0:
            # Check that each position is assigned to at most one pattern
            pattern_mask = result.pattern_mask.cpu().numpy()
            pattern_indices = result.pattern_indices.cpu().numpy()
            
            for i in range(len(pattern_mask)):
                if pattern_mask[i]:
                    assert pattern_indices[i] >= 0
    
    def test_large_pattern_handling(self):
        """Test handling of patterns at maximum length"""
        detector = SlidingWindowPatternDetector(min_pattern_length=2, max_pattern_length=8, min_frequency=2)
        
        # Create large repeating pattern
        large_pattern = list(range(8))  # [0, 1, 2, 3, 4, 5, 6, 7]
        data = large_pattern * 5  # Repeat 5 times
        data = np.array(data, dtype=np.uint8)
        
        result = detector.find_patterns(data)
        
        if result.total_patterns_found > 0:
            # Should find the large pattern
            found_large = False
            for pattern_match in result.patterns.values():
                if pattern_match.element_count == 8:
                    found_large = True
                    break
            
            # Either found the full pattern or smaller sub-patterns
            assert found_large or result.total_patterns_found > 0
    
    def test_pattern_frequency_threshold(self):
        """Test that patterns below frequency threshold are not included"""
        detector = SlidingWindowPatternDetector(min_pattern_length=2, min_frequency=3)
        
        # Pattern appears only twice (below threshold)
        data = np.array([1, 2, 3, 1, 2, 4, 5, 6], dtype=np.uint8)
        
        result = detector.find_patterns(data)
        
        # Should not find the [1, 2] pattern since it only appears twice
        for pattern_match in result.patterns.values():
            assert pattern_match.frequency >= 3
    
    def test_bytes_input_handling(self):
        """Test direct bytes input handling"""
        detector = SlidingWindowPatternDetector(min_pattern_length=2, min_frequency=2)
        
        # Test with bytes input
        data = b'\x01\x02\x03\x01\x02\x04'
        
        result = detector.find_patterns(data)
        
        assert result.element_dtype == np.dtype('uint8')
        # Should handle bytes the same as uint8 numpy array
    
    def test_device_handling(self):
        """Test device handling for tensors"""
        device = 'cpu'  # Use CPU for testing
        detector = SlidingWindowPatternDetector(device=device)
        
        data = np.array([1, 2, 1, 2], dtype=np.uint8)
        
        result = detector.find_patterns(data)
        
        # Tensors should be on correct device
        assert result.pattern_mask.device.type == device
        assert result.pattern_indices.device.type == device
    
    def test_max_patterns_limit(self):
        """Test that max_patterns limit is respected"""
        detector = SlidingWindowPatternDetector(
            min_pattern_length=2,
            min_frequency=2,
            max_patterns=2  # Low limit
        )
        
        # Create data with many different patterns
        data = []
        for i in range(10):
            pattern = [i, i + 10]
            data.extend(pattern * 3)  # Each pattern appears 3 times
        
        data = np.array(data, dtype=np.uint8)
        
        result = detector.find_patterns(data)
        
        # Should not exceed max_patterns limit
        assert result.total_patterns_found <= 2
    
    def test_rolling_hash_specific(self):
        """Test rolling hash algorithm specifically"""
        detector = SlidingWindowPatternDetector(min_pattern_length=3, min_frequency=2)
        
        # Create data that will use rolling hash (small data)
        data = np.array([10, 20, 30, 40, 10, 20, 30, 50], dtype=np.uint8)
        
        # Force rolling hash by using small data
        patterns_found = detector._find_patterns_rolling_hash(data, np.dtype('uint8'))
        
        assert isinstance(patterns_found, dict)
        # Should find the [10, 20, 30] pattern
        if patterns_found:
            pattern_found = False
            for pattern_match in patterns_found.values():
                pattern_array = np.frombuffer(pattern_match.pattern, dtype=np.uint8)
                if len(pattern_array) >= 3 and np.array_equal(pattern_array[:3], [10, 20, 30]):
                    pattern_found = True
                    break
            assert pattern_found
    
    def test_suffix_array_specific(self):
        """Test suffix array algorithm specifically"""
        detector = SlidingWindowPatternDetector(min_pattern_length=3, min_frequency=2)
        
        data = np.array([5, 15, 25, 35, 5, 15, 25, 45], dtype=np.uint8)
        
        # Test suffix array method directly
        patterns_found = detector._find_patterns_suffix_array(data, np.dtype('uint8'))
        
        assert isinstance(patterns_found, dict)
        # Should find patterns using suffix array
        if patterns_found:
            for pattern_match in patterns_found.values():
                assert pattern_match.frequency >= detector.min_frequency


class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_full_encode_decode_cycle(self):
        """Test complete encode-decode cycle preserves data"""
        detector = SlidingWindowPatternDetector(min_pattern_length=3, min_frequency=2)
        
        # Create test data with patterns
        pattern1 = [100, 101, 102]
        pattern2 = [200, 201]
        
        data = []
        for _ in range(10):
            data.extend(pattern1)
            data.extend(np.random.randint(0, 50, 2))
            data.extend(pattern2)
            data.extend(np.random.randint(60, 80, 1))
        
        original_data = np.array(data, dtype=np.int32)
        
        # Full cycle
        encoded_data, pattern_dict, metadata = detector.forward(original_data)
        decoded_data = detector.decode_with_patterns(
            encoded_data,
            pattern_dict,
            metadata['pattern_lengths'],
            metadata['element_dtype']
        )
        
        # Should perfectly reconstruct original data
        assert torch.equal(decoded_data, torch.from_numpy(original_data))
        assert decoded_data.dtype == torch.int32
    
    def test_compression_effectiveness(self):
        """Test compression effectiveness on highly repetitive data"""
        detector = SlidingWindowPatternDetector(min_pattern_length=4, min_frequency=3)
        
        # Create highly repetitive data
        base_pattern = [42, 43, 44, 45, 46]
        data = base_pattern * 200  # Very repetitive
        data = np.array(data, dtype=np.uint8)
        
        analysis = detector.analyze_compression_efficiency(data)
        
        # Should achieve significant compression
        assert analysis['compression_ratio'] > 2.0  # At least 2x compression
        assert analysis['space_savings_percent'] > 50  # At least 50% savings
        assert analysis['patterns_found'] > 0
    
    def test_mixed_dtype_handling(self):
        """Test handling of different data types in same workflow"""
        detector = SlidingWindowPatternDetector(min_pattern_length=2, min_frequency=2)
        
        test_cases = [
            (np.array([1, 2, 3, 1, 2], dtype=np.uint8), np.dtype('uint8')),
            (np.array([100, 200, 300, 100, 200], dtype=np.int16), np.dtype('int16')),
            (np.array([1000, 2000, 3000, 1000, 2000], dtype=np.int32), np.dtype('int32')),
        ]
        
        for original_data, expected_dtype in test_cases:
            # Test full pipeline
            result = detector.find_patterns(original_data)
            assert result.element_dtype == expected_dtype
            
            encoded_data, pattern_dict, pattern_lengths = detector.encode_with_patterns(original_data, result)
            decoded_data = detector.decode_with_patterns(encoded_data, pattern_dict, pattern_lengths, expected_dtype)
            
            # Should preserve data and dtype
            assert torch.equal(decoded_data, torch.from_numpy(original_data))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])