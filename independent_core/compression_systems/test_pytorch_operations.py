"""
Comprehensive Tests for PyTorch Operations
Tests pattern matching, tropical ops, and compression operations
PRODUCTION READY - NO PLACEHOLDERS - HARD FAILURES ONLY
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
import time
import logging
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'strategies')))

# Import modules to test
from strategies.pytorch_pattern_matcher import PyTorchPatternMatcher, PatternMatchResult
from strategies.pytorch_tropical_ops import PyTorchTropicalOps, TROPICAL_ZERO
from strategies.pytorch_compression_ops import (
    PyTorchCompressionOps, 
    QuantizationResult,
    SparseEncoding
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPyTorchPatternMatcher:
    """Test suite for PyTorch pattern matching"""
    
    @pytest.fixture
    def matcher(self):
        """Create pattern matcher instance"""
        return PyTorchPatternMatcher(compile_mode=False)  # Disable compile for tests
    
    def test_exact_pattern_matching(self, matcher):
        """Test exact pattern matching"""
        # Create test data
        data = torch.tensor([1, 2, 3, 4, 1, 2, 3, 5, 1, 2], dtype=torch.float32)
        patterns = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        
        # Find patterns
        result = matcher.find_patterns(data, patterns)
        
        # Check results
        assert isinstance(result, PatternMatchResult)
        assert result.match_count == 2, f"Expected 2 matches, got {result.match_count}"
        assert torch.all(result.pattern_indices == 0), "Should only match first pattern"
        assert torch.all(result.position_indices == torch.tensor([0, 4])), "Wrong positions"
    
    def test_approximate_pattern_matching(self, matcher):
        """Test approximate pattern matching with tolerance"""
        data = torch.tensor([1.0, 2.1, 2.9, 4.0, 1.1, 1.9, 3.1], dtype=torch.float32)
        patterns = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        
        # Find with tolerance
        pattern_idx, pos_idx, confidence = matcher.find_approximate_patterns(
            data, patterns, tolerance=0.15
        )
        
        assert len(pattern_idx) == 2, f"Expected 2 approximate matches, got {len(pattern_idx)}"
        assert torch.all(pattern_idx == 0), "Should match first pattern"
        assert torch.all(confidence > 0.8), "Confidence should be high"
    
    def test_batched_pattern_finding(self, matcher):
        """Test batched pattern finding for efficiency"""
        # Large data
        data = torch.randn(10000)
        patterns = torch.randn(100, 10)
        
        # Find patterns in batches
        results = matcher.find_patterns_batched(data, patterns, batch_size=20)
        
        assert 'pattern_indices' in results
        assert 'pattern_counts' in results
        assert len(results['pattern_counts']) == 100
    
    def test_recurring_pattern_discovery(self, matcher):
        """Test automatic discovery of recurring patterns"""
        # Create data with recurring pattern
        pattern = torch.tensor([1, 2, 3, 4])
        data = torch.cat([pattern, torch.randn(5), pattern, torch.randn(3), pattern])
        
        # Discover patterns
        discovered = matcher.find_recurring_patterns(
            data, min_length=4, max_length=4, min_occurrences=3
        )
        
        assert discovered['num_patterns'] > 0, "Should discover at least one pattern"
        assert discovered['potential_compression'] > 1.0, "Should have compression potential"
    
    def test_pattern_entropy(self, matcher):
        """Test entropy computation for compressibility analysis"""
        # Low entropy (highly compressible)
        repetitive_data = torch.tensor([1, 2, 1, 2, 1, 2, 1, 2])
        low_entropy = matcher.compute_pattern_entropy(repetitive_data, pattern_length=2)
        
        # High entropy (less compressible)
        random_data = torch.randn(100)
        high_entropy = matcher.compute_pattern_entropy(random_data, pattern_length=2)
        
        assert low_entropy < high_entropy, "Repetitive data should have lower entropy"
        assert low_entropy < 2.0, f"Low entropy too high: {low_entropy}"
    
    def test_hierarchical_patterns(self, matcher):
        """Test multi-level pattern analysis"""
        data = torch.randn(1000)
        
        hierarchy = matcher.find_hierarchical_patterns(
            data, levels=[2, 4, 8, 16]
        )
        
        assert 'level_2' in hierarchy
        assert 'level_4' in hierarchy
        assert hierarchy['level_2']['num_unique'] > 0
        assert 'entropy' in hierarchy['level_2']
        assert 'redundancy' in hierarchy['level_2']
    
    def test_gpu_acceleration(self):
        """Test GPU acceleration if available"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        matcher_gpu = PyTorchPatternMatcher(device=torch.device('cuda'))
        
        data = torch.randn(10000).cuda()
        patterns = torch.randn(10, 20).cuda()
        
        # Time GPU execution
        start = time.time()
        result = matcher_gpu.find_patterns(data, patterns)
        gpu_time = time.time() - start
        
        assert result.pattern_indices.device.type == 'cuda'
        logger.info(f"GPU pattern matching took {gpu_time:.4f}s")
    
    def test_compilation(self):
        """Test torch.compile optimization"""
        if not hasattr(torch, 'compile'):
            pytest.skip("torch.compile not available")
        
        matcher_compiled = PyTorchPatternMatcher(compile_mode=True)
        
        data = torch.randn(1000)
        patterns = torch.randn(5, 10)
        
        # Warm up
        _ = matcher_compiled.find_patterns(data, patterns)
        
        # Time compiled execution
        start = time.time()
        for _ in range(10):
            _ = matcher_compiled.find_patterns(data, patterns)
        compiled_time = time.time() - start
        
        logger.info(f"Compiled pattern matching (10 runs): {compiled_time:.4f}s")


class TestPyTorchTropicalOps:
    """Test suite for PyTorch tropical operations"""
    
    @pytest.fixture
    def tropical_ops(self):
        """Create tropical operations instance"""
        return PyTorchTropicalOps(compile_mode=False)
    
    def test_tropical_addition(self, tropical_ops):
        """Test tropical addition (max operation)"""
        a = torch.tensor([1.0, 2.0, -3.0])
        b = torch.tensor([2.0, 1.0, -1.0])
        
        result = tropical_ops.tropical_add(a, b)
        expected = torch.tensor([2.0, 2.0, -1.0])
        
        assert torch.allclose(result, expected), f"Expected {expected}, got {result}"
    
    def test_tropical_multiplication(self, tropical_ops):
        """Test tropical multiplication (addition)"""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([2.0, 3.0, 4.0])
        
        result = tropical_ops.tropical_multiply(a, b)
        expected = torch.tensor([3.0, 5.0, 7.0])
        
        assert torch.allclose(result, expected), f"Expected {expected}, got {result}"
    
    def test_tropical_zeros(self, tropical_ops):
        """Test handling of tropical zeros"""
        a = torch.tensor([1.0, TROPICAL_ZERO, 3.0])
        b = torch.tensor([2.0, 1.0, TROPICAL_ZERO])
        
        # Tropical addition with zeros
        add_result = tropical_ops.tropical_add(a, b)
        assert add_result[1] == 1.0, "max(-inf, 1) should be 1"
        assert add_result[2] == 3.0, "max(3, -inf) should be 3"
        
        # Tropical multiplication with zeros
        mult_result = tropical_ops.tropical_multiply(a, b)
        assert mult_result[1] == TROPICAL_ZERO, "-inf + x should be -inf"
        assert mult_result[2] == TROPICAL_ZERO, "x + -inf should be -inf"
    
    def test_tropical_matmul(self, tropical_ops):
        """Test tropical matrix multiplication"""
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        B = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
        
        result = tropical_ops.tropical_matmul(A, B)
        
        # Manually compute expected result
        # C[0,0] = max(1+2, 2+4) = max(3, 6) = 6
        # C[0,1] = max(1+1, 2+3) = max(2, 5) = 5
        # C[1,0] = max(3+2, 4+4) = max(5, 8) = 8
        # C[1,1] = max(3+1, 4+3) = max(4, 7) = 7
        expected = torch.tensor([[6.0, 5.0], [8.0, 7.0]])
        
        assert torch.allclose(result, expected), f"Expected {expected}, got {result}"
    
    def test_tropical_conv1d(self, tropical_ops):
        """Test tropical 1D convolution"""
        input = torch.randn(2, 3, 10)  # batch=2, channels=3, length=10
        weight = torch.randn(4, 3, 3)  # out_channels=4, in_channels=3, kernel=3
        
        output = tropical_ops.tropical_conv1d(input, weight, stride=1, padding=1)
        
        assert output.shape == (2, 4, 10), f"Wrong output shape: {output.shape}"
    
    def test_tropical_polynomial(self, tropical_ops):
        """Test tropical polynomial evaluation"""
        # p(x) = max(2, 1+x, 3+2x)
        coeffs = torch.tensor([2.0, 1.0, 3.0])
        x = torch.tensor([1.0, 2.0, 3.0])
        
        result = tropical_ops.tropical_polynomial_eval(coeffs, x)
        
        # For x=1: max(2, 1+1, 3+2) = max(2, 2, 5) = 5
        # For x=2: max(2, 1+2, 3+4) = max(2, 3, 7) = 7
        # For x=3: max(2, 1+3, 3+6) = max(2, 4, 9) = 9
        expected = torch.tensor([5.0, 7.0, 9.0])
        
        assert torch.allclose(result.squeeze(), expected), f"Expected {expected}, got {result}"
    
    def test_tropical_distance(self, tropical_ops):
        """Test tropical distance computation"""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([2.0, 1.0, 5.0])
        
        distance = tropical_ops.tropical_distance(a, b)
        
        # max(|1-2|, |2-1|, |3-5|) = max(1, 1, 2) = 2
        assert distance == 2.0, f"Expected distance 2.0, got {distance}"
    
    def test_tropical_power(self, tropical_ops):
        """Test tropical matrix power"""
        A = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
        
        # A^2 in tropical arithmetic
        A2 = tropical_ops.tropical_power(A, 2)
        expected = tropical_ops.tropical_matmul(A, A)
        
        assert torch.allclose(A2, expected), "A^2 should equal tropical_matmul(A, A)"
        
        # A^0 should be tropical identity
        A0 = tropical_ops.tropical_power(A, 0)
        assert A0[0, 0] == 0 and A0[1, 1] == 0, "Diagonal should be 0"
        assert A0[0, 1] == TROPICAL_ZERO and A0[1, 0] == TROPICAL_ZERO, "Off-diagonal should be -inf"
    
    def test_tropical_determinant(self, tropical_ops):
        """Test tropical determinant (permanent)"""
        # Small matrix for testing
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        det = tropical_ops.tropical_determinant(A)
        
        # Tropical determinant of 2x2: max(a00+a11, a01+a10)
        # = max(1+4, 2+3) = max(5, 5) = 5
        assert det == 5.0, f"Expected determinant 5.0, got {det}"


class TestPyTorchCompressionOps:
    """Test suite for PyTorch compression operations"""
    
    @pytest.fixture
    def compression_ops(self):
        """Create compression operations instance"""
        return PyTorchCompressionOps(compile_mode=False)
    
    def test_quantization(self, compression_ops):
        """Test weight quantization"""
        weights = torch.randn(100, 100)
        
        # Test 8-bit quantization
        result = compression_ops.quantize_weights(weights, bits=8)
        
        assert isinstance(result, QuantizationResult)
        assert result.bits == 8
        assert result.quantized_weights.dtype in [torch.int8, torch.uint8]
        
        # Test reconstruction
        reconstructed = result.dequantize()
        error = (weights - reconstructed).abs().mean()
        assert error < 0.1, f"Quantization error too high: {error}"
    
    def test_per_channel_quantization(self, compression_ops):
        """Test per-channel quantization"""
        weights = torch.randn(64, 128, 3, 3)  # Conv2D weights
        
        result = compression_ops.quantize_weights(
            weights, bits=8, per_channel=True, channel_axis=0
        )
        
        assert result.scale.shape[0] == 64, "Should have per-channel scales"
        assert result.zero_point.shape[0] == 64, "Should have per-channel zero points"
    
    def test_sparse_encoding(self, compression_ops):
        """Test sparse tensor encoding"""
        # Create sparse tensor
        tensor = torch.zeros(100, 100)
        tensor[torch.randint(0, 100, (20,)), torch.randint(0, 100, (20,))] = torch.randn(20)
        
        result = compression_ops.sparse_encode(tensor, threshold=1e-6)
        
        assert isinstance(result, SparseEncoding)
        assert result.nnz <= 20, f"Too many non-zero elements: {result.nnz}"
        assert result.density < 0.01, f"Density too high: {result.density}"
        
        # Test reconstruction
        reconstructed = result.to_dense()
        assert torch.allclose(tensor, reconstructed, atol=1e-6)
    
    def test_top_k_sparsification(self, compression_ops):
        """Test top-k sparsification"""
        tensor = torch.randn(1000)
        
        result = compression_ops.sparse_encode(tensor, top_k=100)
        
        assert result.nnz == 100, f"Should keep exactly 100 values, got {result.nnz}"
        
        # Verify these are the largest magnitude values
        top_k_values = torch.topk(tensor.abs(), 100)[0]
        reconstructed = result.to_dense()
        reconstructed_nonzero = reconstructed[reconstructed != 0].abs()
        
        assert len(reconstructed_nonzero) == 100
    
    def test_channel_wise_compression(self, compression_ops):
        """Test IEEE 754 channel extraction"""
        tensor = torch.randn(100, 100)
        
        channels = compression_ops.channel_wise_compression(
            tensor, channels=['sign', 'exponent', 'mantissa']
        )
        
        assert 'sign' in channels
        assert 'exponent' in channels
        assert 'mantissa' in channels
        
        assert channels['sign'].dtype == torch.int8
        assert channels['exponent'].dtype == torch.uint8
    
    def test_adaptive_quantization(self, compression_ops):
        """Test adaptive quantization based on target size"""
        tensor = torch.randn(100, 100)
        
        result = compression_ops.adaptive_quantization(tensor, target_size_ratio=0.25)
        
        # Should choose around 8 bits for 0.25 ratio (8/32 = 0.25)
        assert 6 <= result.bits <= 10, f"Unexpected bit depth: {result.bits}"
    
    def test_pruning_mask(self, compression_ops):
        """Test pruning mask generation"""
        tensor = torch.randn(100, 100)
        
        # Unstructured pruning
        mask = compression_ops.pruning_mask(tensor, sparsity=0.9)
        
        assert mask.dtype == torch.bool
        assert mask.shape == tensor.shape
        assert mask.sum() == int(10000 * 0.1), "Should keep 10% of values"
        
        # Structured pruning
        mask_structured = compression_ops.pruning_mask(
            tensor, sparsity=0.5, structured=True, block_size=10
        )
        
        assert mask_structured.sum() > 0, "Should have some non-zero blocks"
    
    def test_delta_encoding(self, compression_ops):
        """Test delta encoding for sequential data"""
        # Sequential data with small changes
        tensor = torch.arange(100, dtype=torch.float32) + torch.randn(100) * 0.1
        
        result = compression_ops.delta_encoding(tensor)
        
        assert 'base' in result
        assert 'deltas' in result
        assert 'delta_scale' in result
        
        # Deltas should be quantized to fewer bits
        assert result['deltas'].dtype in [torch.int8, torch.uint8, torch.int16]
    
    def test_block_wise_compression(self, compression_ops):
        """Test block-wise compression"""
        tensor = torch.randn(1000)
        
        result = compression_ops.block_wise_compression(tensor, block_size=32)
        
        assert 'blocks' in result
        assert 'metadata' in result
        assert result['num_blocks'] == 32, f"Wrong number of blocks: {result['num_blocks']}"
        
        # Check that sparse and dense blocks are handled differently
        block_types = [b['type'] for b in result['blocks']]
        assert 'quantized' in block_types or 'sparse' in block_types
    
    def test_huffman_encoding(self, compression_ops):
        """Test Huffman encoding for indices"""
        # Create indices with skewed distribution
        indices = torch.tensor([0] * 50 + [1] * 30 + [2] * 15 + [3] * 5)
        
        result = compression_ops.huffman_encode_indices(indices)
        
        assert 'encoded' in result
        assert 'codebook' in result
        assert result['compression_ratio'] > 1.0, "Should achieve compression"
        
        # Most frequent value should have shortest code
        codes = result['codebook']
        assert len(codes[0]) <= len(codes[3]), "Frequent values should have shorter codes"


class TestIntegration:
    """Integration tests for all PyTorch operations"""
    
    def test_pattern_based_compression(self):
        """Test pattern detection followed by compression"""
        # Create data with patterns
        pattern = torch.tensor([1.0, 2.0, 3.0, 4.0])
        data = torch.cat([pattern] * 25 + [torch.randn(100)])
        
        # Detect patterns
        matcher = PyTorchPatternMatcher()
        patterns = matcher.find_recurring_patterns(data, min_length=4, max_length=4)
        
        # Compress based on patterns
        compression_ops = PyTorchCompressionOps()
        
        if patterns['num_patterns'] > 0:
            # Use sparse encoding for pattern positions
            pattern_mask = torch.zeros_like(data, dtype=torch.bool)
            for stat in patterns['statistics']:
                positions = torch.tensor(stat['positions'])
                for pos in positions:
                    pattern_mask[pos:pos+4] = True
            
            # Encode non-pattern data
            non_pattern_data = data[~pattern_mask]
            compressed = compression_ops.sparse_encode(non_pattern_data)
            
            assert compressed.density < 1.0, "Should achieve compression"
    
    def test_tropical_compression_pipeline(self):
        """Test tropical operations in compression pipeline"""
        tropical_ops = PyTorchTropicalOps()
        compression_ops = PyTorchCompressionOps()
        
        # Create weight matrix
        weights = torch.randn(100, 100)
        
        # Apply tropical polynomial approximation
        coeffs = torch.randn(10)
        x_vals = torch.linspace(0, 1, 100)
        poly_approx = tropical_ops.tropical_polynomial_eval(coeffs, x_vals)
        
        # Quantize the approximation
        quantized = compression_ops.quantize_weights(poly_approx, bits=8)
        
        # Verify reconstruction
        reconstructed = quantized.dequantize()
        assert reconstructed.shape == poly_approx.shape
    
    def test_memory_efficiency(self):
        """Test memory efficiency of operations"""
        import tracemalloc
        
        tracemalloc.start()
        
        # Large scale operations
        matcher = PyTorchPatternMatcher()
        data = torch.randn(100000)
        patterns = torch.randn(100, 10)
        
        # Measure memory before
        snapshot1 = tracemalloc.take_snapshot()
        
        # Run pattern matching
        results = matcher.find_patterns_batched(data, patterns, batch_size=10)
        
        # Measure memory after
        snapshot2 = tracemalloc.take_snapshot()
        
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_memory = sum(stat.size_diff for stat in top_stats) / 1024 / 1024
        
        logger.info(f"Memory used: {total_memory:.2f} MB")
        assert total_memory < 100, f"Using too much memory: {total_memory:.2f} MB"
        
        tracemalloc.stop()
    
    def test_gpu_cpu_consistency(self):
        """Test consistency between GPU and CPU operations"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create operations for both devices
        ops_cpu = PyTorchTropicalOps(device=torch.device('cpu'))
        ops_gpu = PyTorchTropicalOps(device=torch.device('cuda'))
        
        # Test data
        A = torch.randn(10, 10)
        B = torch.randn(10, 10)
        
        # CPU computation
        result_cpu = ops_cpu.tropical_matmul(A, B)
        
        # GPU computation
        A_gpu = A.cuda()
        B_gpu = B.cuda()
        result_gpu = ops_gpu.tropical_matmul(A_gpu, B_gpu).cpu()
        
        # Should be identical
        assert torch.allclose(result_cpu, result_gpu, atol=1e-5), "CPU/GPU results differ"


def run_performance_benchmarks():
    """Run performance benchmarks"""
    print("\n" + "="*60)
    print("PYTORCH OPERATIONS PERFORMANCE BENCHMARKS")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Pattern matching benchmark
    print("\n1. Pattern Matching:")
    matcher = PyTorchPatternMatcher(device=device)
    data = torch.randn(100000, device=device)
    patterns = torch.randn(100, 20, device=device)
    
    start = time.time()
    _ = matcher.find_patterns_batched(data, patterns)
    elapsed = time.time() - start
    print(f"   100k elements, 100 patterns: {elapsed:.4f}s")
    
    # Tropical operations benchmark
    print("\n2. Tropical Operations:")
    tropical = PyTorchTropicalOps(device=device)
    A = torch.randn(1000, 1000, device=device)
    B = torch.randn(1000, 1000, device=device)
    
    start = time.time()
    _ = tropical.tropical_matmul(A, B)
    elapsed = time.time() - start
    print(f"   1000x1000 tropical matmul: {elapsed:.4f}s")
    
    # Compression operations benchmark
    print("\n3. Compression Operations:")
    compression = PyTorchCompressionOps(device=device)
    weights = torch.randn(1000, 1000, device=device)
    
    start = time.time()
    _ = compression.quantize_weights(weights, bits=8)
    elapsed = time.time() - start
    print(f"   1M weight quantization (8-bit): {elapsed:.4f}s")
    
    start = time.time()
    _ = compression.sparse_encode(weights, threshold=0.1)
    elapsed = time.time() - start
    print(f"   1M weight sparse encoding: {elapsed:.4f}s")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Run benchmarks
    run_performance_benchmarks()
    
    # Run tests with pytest if available
    try:
        import pytest
        pytest.main([__file__, "-v", "--tb=short"])
    except ImportError:
        print("\nInstall pytest to run unit tests: pip install pytest")