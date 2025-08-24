"""
Comprehensive Test Suite for EntropyPAdicBridge
Tests all entropy analysis, encoding/decoding, and bridge functionality
"""

import unittest
import torch
import numpy as np
import math
import time
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
from unittest.mock import MagicMock, patch

# Import the modules to test
from .entropy_bridge import (
    EntropyPAdicBridge, EntropyBridgeConfig, EntropyAnalysis,
    validate_entropy_bridge
)


class TestEntropyBridgeConfig(unittest.TestCase):
    """Test EntropyBridgeConfig dataclass"""
    
    def test_config_creation(self):
        """Test creating config with default values"""
        config = EntropyBridgeConfig()
        
        self.assertEqual(config.huffman_threshold, 2.0)
        self.assertEqual(config.arithmetic_threshold, 6.0)
        self.assertEqual(config.hybrid_low_threshold, 3.0)
        self.assertEqual(config.hybrid_high_threshold, 5.0)
        self.assertEqual(config.frequency_split_ratio, 0.2)
        self.assertTrue(config.enable_pattern_detection)
        self.assertEqual(config.compression_level, 6)
    
    def test_config_validation(self):
        """Test configuration parameter validation"""
        # Valid config should pass
        config = EntropyBridgeConfig(
            huffman_threshold=1.0,
            arithmetic_threshold=5.0,
            frequency_split_ratio=0.3,
            max_tensor_size=100000,
            compression_level=5
        )
        self.assertIsInstance(config, EntropyBridgeConfig)
        
        # Invalid threshold order should fail
        with self.assertRaises(ValueError):
            EntropyBridgeConfig(huffman_threshold=5.0, arithmetic_threshold=2.0)
        
        # Invalid frequency split ratio should fail
        with self.assertRaises(ValueError):
            EntropyBridgeConfig(frequency_split_ratio=0.0)
        with self.assertRaises(ValueError):
            EntropyBridgeConfig(frequency_split_ratio=1.5)
        
        # Invalid max tensor size should fail
        with self.assertRaises(ValueError):
            EntropyBridgeConfig(max_tensor_size=0)
        
        # Invalid compression level should fail
        with self.assertRaises(ValueError):
            EntropyBridgeConfig(compression_level=0)
        with self.assertRaises(ValueError):
            EntropyBridgeConfig(compression_level=10)


class TestEntropyAnalysis(unittest.TestCase):
    """Test EntropyAnalysis dataclass"""
    
    def test_analysis_creation(self):
        """Test creating entropy analysis"""
        analysis = EntropyAnalysis(
            shannon_entropy=2.5,
            normalized_entropy=0.8,
            unique_symbols=10,
            total_symbols=1000,
            max_symbol=15,
            min_symbol=0
        )
        
        self.assertEqual(analysis.shannon_entropy, 2.5)
        self.assertEqual(analysis.normalized_entropy, 0.8)
        self.assertEqual(analysis.unique_symbols, 10)
        self.assertEqual(analysis.total_symbols, 1000)
        self.assertEqual(analysis.recommended_method, "huffman")
        self.assertEqual(analysis.confidence_score, 0.0)
    
    def test_expected_compression_calculation(self):
        """Test compression ratio calculation"""
        analysis = EntropyAnalysis(
            shannon_entropy=2.0,
            normalized_entropy=0.5,
            unique_symbols=5,
            total_symbols=1000,
            max_symbol=4,
            min_symbol=0
        )
        
        ratio = analysis.calculate_expected_compression()
        self.assertGreater(ratio, 1.0)  # Should compress
        self.assertLess(ratio, 8.0)     # Reasonable bounds
        
        # Test edge cases
        analysis_zero = EntropyAnalysis(
            shannon_entropy=0.0,
            normalized_entropy=0.0,
            unique_symbols=1,
            total_symbols=0,
            max_symbol=0,
            min_symbol=0
        )
        self.assertEqual(analysis_zero.calculate_expected_compression(), 1.0)


class TestEntropyPAdicBridge(unittest.TestCase):
    """Test EntropyPAdicBridge main functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.prime = 257
        self.config = EntropyBridgeConfig(max_tensor_size=1000)  # Small for testing
        self.bridge = EntropyPAdicBridge(self.prime, self.config)
        
        # Test data
        self.small_tensor = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
        self.uniform_tensor = torch.tensor(list(range(10)) * 20)
        self.skewed_tensor = torch.tensor([0] * 100 + [1] * 10 + [2] * 5)
        self.random_tensor = torch.randint(0, self.prime, (100,))
        self.empty_tensor = torch.tensor([])
    
    def test_initialization(self):
        """Test bridge initialization"""
        bridge = EntropyPAdicBridge(prime=31)
        self.assertEqual(bridge.prime, 31)
        self.assertIsInstance(bridge.config, EntropyBridgeConfig)
        self.assertEqual(bridge.total_compressions, 0)
        self.assertEqual(bridge.failure_count, 0)
        self.assertEqual(bridge.cache_hits, 0)
        self.assertEqual(bridge.cache_misses, 0)
        
        # Test invalid prime
        with self.assertRaises(ValueError):
            EntropyPAdicBridge(prime=1)
        with self.assertRaises(ValueError):
            EntropyPAdicBridge(prime=-5)
    
    def test_entropy_analysis_basic(self):
        """Test basic entropy analysis"""
        # Test with list input
        digits = [0, 0, 0, 1, 1, 2]
        analysis = self.bridge.analyze_entropy(digits)
        
        self.assertIsInstance(analysis, EntropyAnalysis)
        self.assertGreater(analysis.shannon_entropy, 0.0)
        self.assertEqual(analysis.total_symbols, 6)
        self.assertEqual(analysis.unique_symbols, 3)
        self.assertEqual(analysis.max_symbol, 2)
        self.assertEqual(analysis.min_symbol, 0)
        
        # Check frequency distribution
        expected_freq = {0: 3, 1: 2, 2: 1}
        self.assertEqual(analysis.frequency_distribution, expected_freq)
        
        # Check probability distribution
        expected_prob = {0: 0.5, 1: 1/3, 2: 1/6}
        for symbol, prob in analysis.probability_distribution.items():
            self.assertAlmostEqual(prob, expected_prob[symbol], places=6)
    
    def test_entropy_analysis_tensor_inputs(self):
        """Test entropy analysis with different input types"""
        digits = [0, 1, 0, 1, 2, 2]
        
        # Test with numpy array
        np_array = np.array(digits)
        analysis_np = self.bridge.analyze_entropy(np_array)
        
        # Test with torch tensor
        tensor = torch.tensor(digits)
        analysis_torch = self.bridge.analyze_entropy(tensor)
        
        # Test with list
        analysis_list = self.bridge.analyze_entropy(digits)
        
        # All should give same results
        self.assertEqual(analysis_np.total_symbols, analysis_torch.total_symbols)
        self.assertEqual(analysis_torch.total_symbols, analysis_list.total_symbols)
        self.assertAlmostEqual(analysis_np.shannon_entropy, analysis_torch.shannon_entropy, places=6)
        self.assertAlmostEqual(analysis_torch.shannon_entropy, analysis_list.shannon_entropy, places=6)
    
    def test_entropy_analysis_edge_cases(self):
        """Test entropy analysis edge cases"""
        # Empty input should fail
        with self.assertRaises(ValueError):
            self.bridge.analyze_entropy([])
        
        with self.assertRaises(ValueError):
            self.bridge.analyze_entropy(torch.tensor([]))
        
        # Out of range digits should fail
        with self.assertRaises(ValueError):
            self.bridge.analyze_entropy([0, 1, self.prime])
        
        with self.assertRaises(ValueError):
            self.bridge.analyze_entropy([-1, 0, 1])
        
        # Single value (zero entropy)
        analysis = self.bridge.analyze_entropy([5, 5, 5, 5])
        self.assertEqual(analysis.shannon_entropy, 0.0)
        self.assertEqual(analysis.unique_symbols, 1)
    
    def test_pattern_detection(self):
        """Test pattern detection functionality"""
        # Create pattern with repeating sequence
        pattern = [0, 1, 2, 3] * 50  # Clear pattern
        analysis = self.bridge.analyze_entropy(pattern)
        
        self.assertTrue(analysis.has_patterns)
        self.assertGreater(analysis.pattern_ratio, 0.0)
        
        # Test with random data (should have few patterns)
        random_data = np.random.randint(0, self.prime, 200).tolist()
        analysis_random = self.bridge.analyze_entropy(random_data)
        
        # Random data typically has lower pattern ratio
        self.assertLessEqual(analysis_random.pattern_ratio, analysis.pattern_ratio)
    
    def test_run_length_analysis(self):
        """Test run-length analysis"""
        # Create data with runs
        runs_data = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4, 4, 4]
        analysis = self.bridge.analyze_entropy(runs_data)
        
        self.assertGreater(analysis.run_length_ratio, 0.5)  # Most symbols are in runs
        
        # Test with no runs
        no_runs = [0, 1, 2, 3, 4, 5, 6, 7]
        analysis_no_runs = self.bridge.analyze_entropy(no_runs)
        
        self.assertEqual(analysis_no_runs.run_length_ratio, 0.0)
    
    def test_encoding_method_selection(self):
        """Test automatic encoding method selection"""
        # Low entropy data should select Huffman
        low_entropy = [0] * 100 + [1] * 10  # Heavily skewed
        analysis_low = self.bridge.analyze_entropy(low_entropy)
        self.assertEqual(analysis_low.recommended_method, "huffman")
        
        # High entropy data should select Arithmetic (if available)
        high_entropy = list(range(min(self.prime, 100))) * 10  # More uniform
        analysis_high = self.bridge.analyze_entropy(high_entropy)
        
        # Medium entropy with patterns might select hybrid
        medium_entropy = [0, 1, 2] * 100 + [3, 4] * 50
        analysis_medium = self.bridge.analyze_entropy(medium_entropy)
        # Method depends on exact entropy and pattern detection
    
    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        # Very low entropy should have high confidence for Huffman
        very_low = [0] * 1000 + [1] * 10
        analysis = self.bridge.analyze_entropy(very_low)
        if analysis.recommended_method == "huffman":
            self.assertGreater(analysis.confidence_score, 0.8)
        
        # Check confidence bounds
        self.assertGreaterEqual(analysis.confidence_score, 0.0)
        self.assertLessEqual(analysis.confidence_score, 1.0)
    
    def test_cache_functionality(self):
        """Test frequency table caching"""
        # Enable caching
        bridge = EntropyPAdicBridge(self.prime, EntropyBridgeConfig(cache_frequency_tables=True))
        
        digits = [0, 1, 2, 0, 1, 2] * 10
        
        # First analysis (cache miss)
        analysis1 = bridge.analyze_entropy(digits)
        self.assertEqual(bridge.cache_hits, 0)
        self.assertEqual(bridge.cache_misses, 1)
        
        # Second analysis (cache hit)
        analysis2 = bridge.analyze_entropy(digits)
        self.assertEqual(bridge.cache_hits, 1)
        self.assertEqual(bridge.cache_misses, 1)
        
        # Results should be identical
        self.assertEqual(analysis1.shannon_entropy, analysis2.shannon_entropy)
    
    def test_encode_decode_tensor_small(self):
        """Test encoding and decoding small tensors"""
        test_tensors = [
            self.small_tensor,
            self.uniform_tensor[:20],  # Smaller subset
            self.skewed_tensor[:30],   # Smaller subset
            torch.tensor([0, 0, 0]),  # All same value
            torch.tensor([0, 1]),     # Minimal case
            torch.tensor([42])        # Single element
        ]
        
        for tensor in test_tensors:
            with self.subTest(tensor_shape=tensor.shape):
                # Encode
                compressed, metadata = self.bridge.encode_padic_tensor(tensor)
                
                # Check metadata
                self.assertIn("encoding_method", metadata)
                self.assertIn("entropy_analysis", metadata)
                self.assertIn("compression_metrics", metadata)
                self.assertEqual(metadata["original_shape"], list(tensor.shape))
                self.assertEqual(metadata["prime"], self.prime)
                
                # Check compression metrics
                metrics = metadata["compression_metrics"]
                self.assertGreater(metrics["original_bytes"], 0)
                self.assertGreater(len(compressed), 0)
                self.assertGreater(metrics["compression_ratio"], 0)
                
                # Decode
                decoded = self.bridge.decode_padic_tensor(compressed, metadata)
                
                # Verify reconstruction
                self.assertEqual(decoded.shape, tensor.shape)
                self.assertTrue(torch.equal(tensor.long(), decoded.long()))
    
    def test_encode_decode_empty_tensor(self):
        """Test encoding and decoding empty tensors"""
        empty_tensor = torch.tensor([])
        
        # Encode
        compressed, metadata = self.bridge.encode_padic_tensor(empty_tensor)
        
        # Check empty tensor handling
        self.assertEqual(compressed, b'')
        self.assertTrue(metadata["empty"])
        self.assertEqual(metadata["encoding_method"], "none")
        self.assertEqual(metadata["original_shape"], list(empty_tensor.shape))
        
        # Decode
        decoded = self.bridge.decode_padic_tensor(compressed, metadata)
        
        # Verify reconstruction
        self.assertEqual(decoded.shape, empty_tensor.shape)
        self.assertEqual(decoded.numel(), 0)
    
    def test_encode_decode_different_shapes(self):
        """Test encoding and decoding tensors with different shapes"""
        test_shapes = [
            (10,),           # 1D
            (5, 4),          # 2D
            (2, 3, 4),       # 3D
            (2, 2, 2, 3),    # 4D
        ]
        
        for shape in test_shapes:
            with self.subTest(shape=shape):
                tensor = torch.randint(0, 10, shape)
                
                # Encode
                compressed, metadata = self.bridge.encode_padic_tensor(tensor)
                
                # Decode
                decoded = self.bridge.decode_padic_tensor(compressed, metadata)
                
                # Verify shape preservation
                self.assertEqual(decoded.shape, tensor.shape)
                self.assertTrue(torch.equal(tensor.long(), decoded.long()))
    
    def test_force_encoding_methods(self):
        """Test forcing specific encoding methods"""
        tensor = torch.tensor([0, 1, 2, 0, 1, 2] * 10)
        
        methods = ["huffman", "arithmetic", "hybrid"]
        
        for method in methods:
            with self.subTest(method=method):
                # Encode with forced method
                compressed, metadata = self.bridge.encode_padic_tensor(tensor, force_method=method)
                
                # Check method was used
                self.assertEqual(metadata["encoding_method"], method)
                
                # Decode
                decoded = self.bridge.decode_padic_tensor(compressed, metadata)
                
                # Verify reconstruction
                self.assertTrue(torch.equal(tensor.long(), decoded.long()))
    
    def test_large_tensor_chunking(self):
        """Test chunking for large tensors"""
        # Create tensor larger than max_tensor_size
        large_tensor = torch.randint(0, 10, (1500,))  # Larger than config.max_tensor_size=1000
        
        # Encode (should trigger chunking)
        compressed, metadata = self.bridge.encode_padic_tensor(large_tensor)
        
        # Check if chunking was used
        if metadata.get("chunked", False):
            self.assertTrue(metadata["chunked"])
            self.assertGreater(metadata["num_chunks"], 1)
            self.assertIn("chunks_metadata", metadata)
        
        # Decode
        decoded = self.bridge.decode_padic_tensor(compressed, metadata)
        
        # Verify reconstruction
        self.assertEqual(decoded.shape, large_tensor.shape)
        self.assertTrue(torch.equal(large_tensor.long(), decoded.long()))
    
    def test_dtype_handling(self):
        """Test different tensor dtypes"""
        test_data = [0, 1, 2, 3, 4]
        
        dtypes = [torch.int32, torch.int64, torch.float32, torch.float64]
        
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                tensor = torch.tensor(test_data, dtype=dtype)
                
                # Encode
                compressed, metadata = self.bridge.encode_padic_tensor(tensor)
                
                # Check dtype is preserved in metadata
                self.assertIn("original_dtype", metadata)
                
                # Decode
                decoded = self.bridge.decode_padic_tensor(compressed, metadata)
                
                # Verify data (convert to long for comparison)
                self.assertTrue(torch.equal(tensor.long(), decoded.long()))
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker pattern for error handling"""
        # Test failure recording
        self.bridge.failure_count = 0
        
        test_error = ValueError("Test error")
        self.bridge._record_failure(test_error)
        self.assertEqual(self.bridge.failure_count, 1)
        
        # Test circuit breaker activation
        self.bridge.failure_count = self.bridge.max_failures
        self.bridge._check_circuit_breaker()  # Should reset counter
        self.assertEqual(self.bridge.failure_count, 0)
    
    def test_statistics_tracking(self):
        """Test performance statistics tracking"""
        # Initial statistics
        stats = self.bridge.get_statistics()
        self.assertEqual(stats["total_compressions"], 0)
        self.assertEqual(stats["method_usage"]["huffman"], 0)
        self.assertEqual(stats["average_compression_ratio"], 0.0)
        
        # Perform some compressions
        test_tensors = [
            torch.tensor([0, 1, 2]),
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([1, 2, 3, 4, 5])
        ]
        
        for tensor in test_tensors:
            self.bridge.encode_padic_tensor(tensor)
        
        # Check updated statistics
        stats = self.bridge.get_statistics()
        self.assertEqual(stats["total_compressions"], 3)
        self.assertGreater(stats["average_compression_ratio"], 0.0)
        
        # Test reset
        self.bridge.reset_statistics()
        stats_reset = self.bridge.get_statistics()
        self.assertEqual(stats_reset["total_compressions"], 0)
        self.assertEqual(stats_reset["average_compression_ratio"], 0.0)
    
    def test_error_handling_decode(self):
        """Test error handling in decode operations"""
        # Test decode with invalid metadata
        with self.assertRaises(ValueError):
            self.bridge.decode_padic_tensor(b"test", {})
        
        # Test decode with missing method
        invalid_metadata = {"original_shape": [5], "prime": self.prime}
        with self.assertRaises(ValueError):
            self.bridge.decode_padic_tensor(b"test", invalid_metadata)
        
        # Test decode with unknown method
        invalid_method_metadata = {
            "original_shape": [5],
            "prime": self.prime,
            "encoding_method": "unknown_method"
        }
        with self.assertRaises(ValueError):
            self.bridge.decode_padic_tensor(b"test", invalid_method_metadata)
    
    def test_adaptive_threshold_adjustment(self):
        """Test adaptive threshold adjustment"""
        config = EntropyBridgeConfig(adaptive_threshold=True)
        bridge = EntropyPAdicBridge(self.prime, config)
        
        # Test with few unique symbols
        sparse_data = [0] * 100 + [1] * 5
        analysis_sparse = bridge.analyze_entropy(sparse_data)
        
        # Test with many unique symbols
        dense_data = list(range(50)) * 4
        analysis_dense = bridge.analyze_entropy(dense_data)
        
        # Both should have recommendations (specific depends on entropy values)
        self.assertIn(analysis_sparse.recommended_method, ["huffman", "arithmetic", "hybrid"])
        self.assertIn(analysis_dense.recommended_method, ["huffman", "arithmetic", "hybrid"])


class TestEntropyBridgeIntegration(unittest.TestCase):
    """Integration tests for EntropyPAdicBridge with different scenarios"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.bridge = EntropyPAdicBridge(prime=31)  # Smaller prime for testing
    
    def test_different_entropy_distributions(self):
        """Test bridge with different entropy distributions"""
        test_cases = [
            # Very low entropy (single symbol repeated)
            ("uniform_single", torch.tensor([5] * 100)),
            
            # Low entropy (heavily skewed)
            ("low_entropy", torch.tensor([0] * 80 + [1] * 15 + [2] * 5)),
            
            # Medium entropy (moderate distribution)
            ("medium_entropy", torch.tensor([0, 1, 2, 3] * 25)),
            
            # High entropy (uniform distribution)
            ("high_entropy", torch.tensor(list(range(20)) * 5)),
            
            # Pattern-based (repeating sequences)
            ("patterned", torch.tensor([0, 1, 2, 0, 1, 2] * 30)),
            
            # Run-length (consecutive repeats)
            ("run_length", torch.tensor([0] * 20 + [1] * 30 + [2] * 25 + [3] * 25))
        ]
        
        compression_results = {}
        
        for name, tensor in test_cases:
            with self.subTest(distribution=name):
                # Analyze entropy first
                analysis = self.bridge.analyze_entropy(tensor)
                
                # Encode and decode
                compressed, metadata = self.bridge.encode_padic_tensor(tensor)
                decoded = self.bridge.decode_padic_tensor(compressed, metadata)
                
                # Verify correctness
                self.assertTrue(torch.equal(tensor.long(), decoded.long()))
                
                # Record compression performance
                compression_results[name] = {
                    "entropy": analysis.shannon_entropy,
                    "method": metadata["encoding_method"],
                    "ratio": metadata["compression_metrics"]["compression_ratio"],
                    "confidence": analysis.confidence_score
                }
        
        # Low entropy should generally compress better
        low_ratio = compression_results["low_entropy"]["ratio"]
        high_ratio = compression_results["high_entropy"]["ratio"]
        self.assertGreaterEqual(low_ratio, high_ratio * 0.8)  # Allow some variance
    
    def test_prime_boundaries(self):
        """Test with different p-adic primes"""
        test_primes = [2, 3, 7, 31, 127, 257]
        
        for prime in test_primes:
            with self.subTest(prime=prime):
                bridge = EntropyPAdicBridge(prime)
                
                # Create test data within prime range
                max_val = min(prime - 1, 10)  # Keep reasonable size
                tensor = torch.randint(0, max_val + 1, (50,))
                
                # Test encode/decode cycle
                compressed, metadata = bridge.encode_padic_tensor(tensor)
                decoded = bridge.decode_padic_tensor(compressed, metadata)
                
                self.assertTrue(torch.equal(tensor.long(), decoded.long()))
    
    def test_stress_conditions(self):
        """Test bridge under stress conditions"""
        bridge = EntropyPAdicBridge(prime=127)
        
        stress_tests = [
            # Very large values (within prime range)
            ("large_values", torch.tensor([120, 126, 125] * 50)),
            
            # All zeros
            ("all_zeros", torch.zeros(100, dtype=torch.long)),
            
            # All maximum value
            ("all_max", torch.full((100,), 126, dtype=torch.long)),
            
            # Alternating pattern
            ("alternating", torch.tensor([0, 126] * 50)),
            
            # Single element
            ("single", torch.tensor([42])),
        ]
        
        for name, tensor in stress_tests:
            with self.subTest(condition=name):
                # Should handle all stress conditions without error
                compressed, metadata = bridge.encode_padic_tensor(tensor)
                decoded = bridge.decode_padic_tensor(compressed, metadata)
                
                self.assertTrue(torch.equal(tensor.long(), decoded.long()))
    
    def test_metadata_completeness(self):
        """Test that all required metadata is preserved"""
        tensor = torch.randint(0, 50, (10, 5))  # 2D tensor
        
        compressed, metadata = self.bridge.encode_padic_tensor(tensor)
        
        # Check required metadata fields
        required_fields = [
            "original_shape", "original_dtype", "prime", "encoding_method",
            "entropy_analysis", "encoding_metadata", "compression_metrics"
        ]
        
        for field in required_fields:
            self.assertIn(field, metadata, f"Missing required field: {field}")
        
        # Check entropy analysis completeness
        entropy_fields = [
            "shannon_entropy", "normalized_entropy", "unique_symbols",
            "total_symbols", "confidence_score"
        ]
        
        for field in entropy_fields:
            self.assertIn(field, metadata["entropy_analysis"], f"Missing entropy field: {field}")
        
        # Check compression metrics completeness
        metric_fields = [
            "original_bytes", "compressed_bytes", "compression_ratio",
            "analysis_time_ms", "encoding_time_ms"
        ]
        
        for field in metric_fields:
            self.assertIn(field, metadata["compression_metrics"], f"Missing metric field: {field}")
    
    def test_backwards_compatibility(self):
        """Test backwards compatibility with different metadata formats"""
        tensor = torch.tensor([1, 2, 3, 4, 5])
        
        # Encode normally
        compressed, metadata = self.bridge.encode_padic_tensor(tensor)
        
        # Test different metadata formats for decoding
        
        # Format 1: method in encoding_metadata
        metadata_format1 = metadata.copy()
        if "encoding_metadata" in metadata_format1:
            metadata_format1["encoding_metadata"]["method"] = metadata["encoding_method"]
        
        decoded1 = self.bridge.decode_padic_tensor(compressed, metadata_format1)
        self.assertTrue(torch.equal(tensor.long(), decoded1.long()))
        
        # Format 2: method directly in metadata
        metadata_format2 = metadata.copy()
        metadata_format2["method"] = metadata["encoding_method"]
        
        decoded2 = self.bridge.decode_padic_tensor(compressed, metadata_format2)
        self.assertTrue(torch.equal(tensor.long(), decoded2.long()))


class TestErrorRecoveryAndEdgeCases(unittest.TestCase):
    """Test error recovery and edge case handling"""
    
    def setUp(self):
        """Set up error recovery test fixtures"""
        self.bridge = EntropyPAdicBridge(prime=127, config=EntropyBridgeConfig(max_failures=3))
    
    def test_invalid_tensor_inputs(self):
        """Test handling of invalid tensor inputs"""
        # Non-tensor input should fail
        with self.assertRaises(TypeError):
            self.bridge.encode_padic_tensor("not a tensor")
        
        with self.assertRaises(TypeError):
            self.bridge.encode_padic_tensor([1, 2, 3])
        
        with self.assertRaises(TypeError):
            self.bridge.encode_padic_tensor(np.array([1, 2, 3]))
    
    def test_reconstruction_errors(self):
        """Test reconstruction error handling"""
        tensor = torch.tensor([1, 2, 3, 4])
        
        compressed, metadata = self.bridge.encode_padic_tensor(tensor)
        
        # Corrupt shape metadata
        corrupted_metadata = metadata.copy()
        corrupted_metadata["original_shape"] = [10]  # Wrong shape
        
        with self.assertRaises(ValueError):
            self.bridge.decode_padic_tensor(compressed, corrupted_metadata)
    
    def test_prime_validation_edge_cases(self):
        """Test edge cases in prime validation"""
        # Boundary values
        valid_primes = [2, 3, 5, 7, 11]
        for prime in valid_primes:
            bridge = EntropyPAdicBridge(prime)
            self.assertEqual(bridge.prime, prime)
        
        # Invalid primes
        invalid_primes = [0, 1, -1, -5]
        for prime in invalid_primes:
            with self.assertRaises(ValueError):
                EntropyPAdicBridge(prime)
    
    def test_memory_efficient_processing(self):
        """Test memory-efficient processing of different sizes"""
        sizes = [1, 10, 100, 1000, 5000]  # Various sizes
        
        for size in sizes:
            with self.subTest(size=size):
                tensor = torch.randint(0, 10, (size,))
                
                compressed, metadata = self.bridge.encode_padic_tensor(tensor)
                decoded = self.bridge.decode_padic_tensor(compressed, metadata)
                
                self.assertTrue(torch.equal(tensor.long(), decoded.long()))


if __name__ == "__main__":
    # Set random seed for reproducible tests
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run all tests
    unittest.main(verbosity=2)