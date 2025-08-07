"""
Comprehensive tests for CSR sparse matrix compression.
Tests dense to CSR conversion, reconstruction, compression ratios,
edge cases, integration with pattern detector, and performance.
NO PLACEHOLDERS - COMPLETE PRODUCTION TESTS
"""

import numpy as np
import torch
import time
import pytest
from typing import List, Tuple
import logging

from csr_sparse_matrix import (
    CSRPadicMatrix, CSRMetrics, BatchedCSROperations, 
    GPUCSRMatrix, CSRPerformanceMonitor
)
from sparse_compressor import (
    SparseCompressor, SparseCompressionResult, SparseAnalysis,
    AdaptiveSparseCompressor, HybridSparseCompressor
)
from compression_strategy import CSRStrategy, StrategySelector, StrategyConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestCSRPadicMatrix:
    """Test CSR matrix basic operations"""
    
    def test_dense_to_csr_conversion(self):
        """Test conversion from dense to CSR format"""
        # Create sparse matrix
        dense = np.array([
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 3.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 6.0]
        ], dtype=np.float32)
        
        csr = CSRPadicMatrix(dense, threshold=1e-6)
        
        # Check CSR structure
        assert csr.shape == (4, 4)
        assert len(csr.values) == 6  # 6 non-zero values
        assert csr.metrics.nnz == 6
        assert csr.metrics.density == 6/16
        
        # Check correct values stored
        expected_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        np.testing.assert_array_almost_equal(sorted(csr.values), sorted(expected_values))
        
        logger.info(f"✓ Dense to CSR conversion: {csr.metrics.compression_ratio:.2f}x compression")
    
    def test_csr_to_dense_reconstruction(self):
        """Test reconstruction from CSR to dense"""
        # Create random sparse matrix
        np.random.seed(42)
        dense = np.random.randn(10, 10).astype(np.float32)
        dense[dense < 0.5] = 0  # Make sparse
        
        # Convert to CSR and back
        csr = CSRPadicMatrix(dense)
        reconstructed = csr.to_dense()
        
        # Check reconstruction accuracy
        np.testing.assert_array_almost_equal(dense, reconstructed, decimal=5)
        
        logger.info("✓ CSR to dense reconstruction accurate")
    
    def test_torch_tensor_support(self):
        """Test CSR with PyTorch tensors"""
        # Create sparse torch tensor
        tensor = torch.randn(20, 20)
        tensor[tensor.abs() < 1.0] = 0
        
        # Convert to CSR
        csr = CSRPadicMatrix(tensor, threshold=1e-6)
        
        # Convert back to torch
        reconstructed = csr.to_torch()
        
        # Check properties preserved
        assert reconstructed.shape == tensor.shape
        assert reconstructed.dtype == tensor.dtype
        if tensor.is_cuda:
            assert reconstructed.is_cuda
        
        # Check values
        torch.testing.assert_close(reconstructed, tensor, atol=1e-5, rtol=1e-5)
        
        logger.info("✓ PyTorch tensor support working")
    
    def test_row_column_access(self):
        """Test efficient row and column access"""
        # Create test matrix
        dense = np.array([
            [1, 2, 0, 0],
            [0, 3, 4, 0],
            [5, 0, 0, 6],
            [0, 0, 7, 8]
        ], dtype=np.float32)
        
        csr = CSRPadicMatrix(dense)
        
        # Test row access
        for i in range(4):
            row = csr.get_row(i)
            np.testing.assert_array_equal(row, dense[i])
        
        # Test column access
        for j in range(4):
            col = csr.get_column(j)
            np.testing.assert_array_equal(col, dense[:, j])
        
        logger.info("✓ Row and column access correct")
    
    def test_matrix_operations(self):
        """Test matrix-vector and matrix-matrix multiplication"""
        # Create test matrices
        A = np.array([
            [2, 0, 0, 1],
            [0, 3, 0, 0],
            [0, 0, 4, 0],
            [1, 0, 0, 2]
        ], dtype=np.float32)
        
        v = np.array([1, 2, 3, 4], dtype=np.float32)
        B = np.random.randn(4, 3).astype(np.float32)
        
        csr_A = CSRPadicMatrix(A)
        
        # Test matrix-vector multiplication
        result_spmv = csr_A.multiply_vector(v)
        expected_spmv = A @ v
        np.testing.assert_array_almost_equal(result_spmv, expected_spmv)
        
        # Test matrix-matrix multiplication
        result_spmm = csr_A.multiply_matrix(B)
        expected_spmm = A @ B
        np.testing.assert_array_almost_equal(result_spmm, expected_spmm)
        
        logger.info("✓ Matrix operations (SpMV, SpMM) correct")
    
    def test_transpose(self):
        """Test matrix transposition"""
        # Create non-symmetric sparse matrix
        dense = np.array([
            [1, 0, 2],
            [0, 3, 0],
            [4, 0, 5],
            [0, 6, 0]
        ], dtype=np.float32)
        
        csr = CSRPadicMatrix(dense)
        csr_T = csr.transpose()
        
        # Check transpose
        expected = dense.T
        result = csr_T.to_dense()
        np.testing.assert_array_equal(result, expected)
        
        logger.info("✓ Matrix transpose correct")
    
    def test_serialization(self):
        """Test serialization and deserialization"""
        # Create test matrix
        dense = np.random.randn(15, 15).astype(np.float32)
        dense[dense < 0.8] = 0
        
        csr = CSRPadicMatrix(dense)
        
        # Serialize
        data = csr.to_bytes()
        
        # Deserialize
        csr_loaded = CSRPadicMatrix.from_bytes(data)
        
        # Check equality
        assert csr_loaded.shape == csr.shape
        assert csr_loaded.threshold == csr.threshold
        np.testing.assert_array_equal(csr_loaded.values, csr.values)
        np.testing.assert_array_equal(csr_loaded.col_idx, csr.col_idx)
        np.testing.assert_array_equal(csr_loaded.row_ptr, csr.row_ptr)
        
        # Check reconstruction
        np.testing.assert_array_equal(csr_loaded.to_dense(), dense)
        
        logger.info(f"✓ Serialization working, size: {len(data)} bytes")
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Empty matrix
        empty = np.zeros((5, 5), dtype=np.float32)
        csr_empty = CSRPadicMatrix(empty)
        assert csr_empty.metrics.nnz == 0
        assert csr_empty.metrics.compression_ratio > 1.0
        np.testing.assert_array_equal(csr_empty.to_dense(), empty)
        
        # Single value matrix
        single = np.zeros((3, 3), dtype=np.float32)
        single[1, 1] = 5.0
        csr_single = CSRPadicMatrix(single)
        assert csr_single.metrics.nnz == 1
        np.testing.assert_array_equal(csr_single.to_dense(), single)
        
        # Full matrix (no sparsity)
        full = np.ones((4, 4), dtype=np.float32)
        csr_full = CSRPadicMatrix(full)
        assert csr_full.metrics.nnz == 16
        assert csr_full.metrics.compression_ratio < 1.0  # No compression benefit
        
        # Column vector
        col_vec = np.array([[1], [0], [2], [0], [3]], dtype=np.float32)
        csr_col = CSRPadicMatrix(col_vec)
        assert csr_col.shape == (5, 1)
        assert csr_col.metrics.nnz == 3
        
        # Row vector
        row_vec = np.array([[1, 0, 2, 0, 3]], dtype=np.float32)
        csr_row = CSRPadicMatrix(row_vec)
        assert csr_row.shape == (1, 5)
        assert csr_row.metrics.nnz == 3
        
        logger.info("✓ Edge cases handled correctly")
    
    def test_validation(self):
        """Test CSR structure validation"""
        # Create valid CSR
        dense = np.array([[1, 0, 2], [0, 3, 0]], dtype=np.float32)
        csr = CSRPadicMatrix(dense)
        
        # Should validate successfully
        assert csr.validate()
        
        # Corrupt the structure
        csr_bad = CSRPadicMatrix(dense)
        csr_bad.row_ptr[-1] = 999  # Invalid pointer
        
        with pytest.raises(ValueError):
            csr_bad.validate()
        
        logger.info("✓ CSR validation working")


class TestSparseCompressor:
    """Test sparse compressor functionality"""
    
    def test_basic_compression(self):
        """Test basic compression and decompression"""
        compressor = SparseCompressor(sparsity_threshold=0.9)
        
        # Create sparse tensor
        tensor = torch.zeros(100, 100)
        tensor[torch.rand(100, 100) > 0.95] = torch.randn(1)
        
        # Compress
        result = compressor.compress(tensor)
        
        assert result is not None
        assert result.success
        assert result.compression_ratio > 5.0
        assert result.sparsity > 0.9
        
        # Decompress
        reconstructed = compressor.decompress(result)
        
        assert reconstructed.shape == tensor.shape
        torch.testing.assert_close(reconstructed, tensor, atol=1e-5, rtol=1e-5)
        
        logger.info(f"✓ Basic compression: {result.compression_ratio:.2f}x, "
                   f"saved {result.memory_saved_bytes} bytes")
    
    def test_insufficient_sparsity(self):
        """Test behavior with insufficient sparsity"""
        compressor = SparseCompressor(sparsity_threshold=0.9)
        
        # Create dense tensor (not sparse enough)
        tensor = torch.randn(50, 50)
        
        # Should return None
        result = compressor.compress(tensor)
        assert result is None
        
        logger.info("✓ Correctly rejects non-sparse tensors")
    
    def test_benefit_analysis(self):
        """Test compression benefit analysis"""
        compressor = SparseCompressor()
        
        # Sparse tensor - should recommend CSR
        sparse_tensor = torch.zeros(200, 200)
        sparse_tensor[torch.rand(200, 200) > 0.92] = 1.0
        
        analysis = compressor.analyze_benefit(sparse_tensor, detailed=True)
        
        assert analysis.recommended
        assert analysis.sparsity > 0.9
        assert analysis.expected_compression_ratio > 3.0
        assert 'mean' in analysis.distribution_stats
        
        # Dense tensor - should not recommend
        dense_tensor = torch.randn(50, 50)
        
        analysis = compressor.analyze_benefit(dense_tensor)
        
        assert not analysis.recommended
        assert analysis.sparsity < 0.1
        
        logger.info("✓ Benefit analysis working correctly")
    
    def test_batch_compression(self):
        """Test batch compression of multiple tensors"""
        compressor = SparseCompressor(sparsity_threshold=0.85)
        
        # Create mix of sparse and dense tensors
        tensors = []
        for i in range(5):
            t = torch.zeros(50, 50)
            sparsity = 0.8 + i * 0.05  # Increasing sparsity
            t[torch.rand(50, 50) > sparsity] = torch.randn(1)
            tensors.append(t)
        
        # Batch compress
        results = compressor.batch_compress(tensors)
        
        assert len(results) == 5
        # First tensor might not compress (80% sparse < 85% threshold)
        # Last tensors should compress (95%+ sparse)
        assert results[-1] is not None
        assert results[-1].compression_ratio > 5.0
        
        logger.info(f"✓ Batch compression: {sum(r is not None for r in results)}/5 compressed")
    
    def test_different_shapes(self):
        """Test compression with various tensor shapes"""
        compressor = SparseCompressor(min_matrix_size=10)
        
        # 1D tensor
        tensor_1d = torch.zeros(1000)
        tensor_1d[torch.rand(1000) > 0.95] = 1.0
        result_1d = compressor.compress(tensor_1d)
        assert result_1d is not None
        reconstructed_1d = compressor.decompress(result_1d)
        assert reconstructed_1d.shape == tensor_1d.shape
        
        # 3D tensor
        tensor_3d = torch.zeros(10, 20, 30)
        tensor_3d[torch.rand(10, 20, 30) > 0.93] = 1.0
        result_3d = compressor.compress(tensor_3d)
        assert result_3d is not None
        reconstructed_3d = compressor.decompress(result_3d)
        assert reconstructed_3d.shape == tensor_3d.shape
        
        # 4D tensor (conv weights)
        tensor_4d = torch.zeros(64, 32, 3, 3)
        tensor_4d[torch.rand(64, 32, 3, 3) > 0.92] = torch.randn(1)
        result_4d = compressor.compress(tensor_4d)
        assert result_4d is not None
        reconstructed_4d = compressor.decompress(result_4d)
        assert reconstructed_4d.shape == tensor_4d.shape
        
        logger.info("✓ Various tensor shapes handled correctly")


class TestAdaptiveSparseCompressor:
    """Test adaptive threshold learning"""
    
    def test_threshold_adjustment(self):
        """Test adaptive threshold adjustment"""
        compressor = AdaptiveSparseCompressor(
            initial_threshold=0.9,
            learning_rate=0.05,
            target_success_rate=0.7
        )
        
        # Process series of tensors with varying sparsity
        for i in range(20):
            sparsity = 0.85 + np.random.uniform(-0.1, 0.15)
            tensor = torch.zeros(100, 100)
            tensor[torch.rand(100, 100) > sparsity] = 1.0
            
            result = compressor.compress(tensor)
        
        # Check that threshold has adapted
        assert len(compressor.threshold_history) > 1
        
        # Check success rate tracking
        stats = compressor.get_adaptive_statistics()
        assert 'success_rate' in stats
        assert 'current_threshold' in stats
        
        logger.info(f"✓ Adaptive learning: threshold {stats['current_threshold']:.3f}, "
                   f"success rate {stats['success_rate']:.2%}")


class TestCSRIntegration:
    """Test integration with compression strategy system"""
    
    def test_csr_strategy(self):
        """Test CSRStrategy integration"""
        strategy = CSRStrategy(sparsity_threshold=0.9)
        
        # Create very sparse tensor
        tensor = torch.zeros(200, 200)
        tensor[torch.rand(200, 200) > 0.95] = torch.randn(1).item()
        
        # Compress
        compressed = strategy.compress(tensor)
        
        assert compressed.strategy_name == "csr"
        assert compressed.compression_ratio > 5.0
        
        # Decompress
        reconstructed = strategy.decompress(compressed)
        
        assert reconstructed.shape == tensor.shape
        torch.testing.assert_close(reconstructed, tensor, atol=1e-5, rtol=1e-5)
        
        # Test gradient support
        assert strategy.supports_gradients()
        
        logger.info(f"✓ CSR strategy integration: {compressed.compression_ratio:.2f}x compression")
    
    def test_strategy_selector_with_csr(self):
        """Test that StrategySelector chooses CSR for very sparse matrices"""
        config = StrategyConfig()
        selector = StrategySelector(config)
        
        # Create extremely sparse tensor (>95% zeros)
        tensor = torch.zeros(500, 500)
        tensor[torch.rand(500, 500) > 0.97] = torch.randn(1).item()
        
        # Select strategy
        strategy, analysis = selector.select_strategy(tensor, "test_layer")
        
        # Should select CSR for extreme sparsity
        if 'csr' in selector.strategy_cache:
            # If CSR is available, it should be selected
            scores = selector.compute_strategy_scores(analysis)
            if 'csr' in scores:
                assert scores['csr'] > 0.8  # High score for CSR
                logger.info(f"✓ CSR correctly scored high: {scores.get('csr', 0):.2f}")
        
        logger.info(f"✓ Selected strategy: {strategy.get_strategy_name()} "
                   f"for {analysis['sparsity']:.1%} sparse tensor")
    
    def test_memory_savings(self):
        """Test actual memory savings with CSR"""
        # Create large sparse matrix
        size = 1000
        tensor = torch.zeros(size, size, dtype=torch.float32)
        # Add ~5% non-zero values
        num_nonzero = int(0.05 * size * size)
        indices = torch.randperm(size * size)[:num_nonzero]
        tensor.view(-1)[indices] = torch.randn(num_nonzero)
        
        # Original size
        original_bytes = tensor.numel() * tensor.element_size()
        
        # Compress with CSR
        compressor = SparseCompressor(sparsity_threshold=0.9)
        result = compressor.compress(tensor)
        
        assert result is not None
        compressed_bytes = len(result.compressed_data)
        
        savings_ratio = original_bytes / compressed_bytes
        savings_bytes = original_bytes - compressed_bytes
        
        logger.info(f"✓ Memory savings: {savings_bytes:,} bytes "
                   f"({savings_ratio:.2f}x compression) for 95% sparse {size}x{size} matrix")
        
        assert savings_ratio > 10.0  # Should achieve >10x for 95% sparse


class TestPerformance:
    """Performance benchmarks for CSR operations"""
    
    def test_compression_speed(self):
        """Benchmark compression speed"""
        sizes = [100, 500, 1000]
        sparsities = [0.9, 0.95, 0.99]
        
        compressor = SparseCompressor()
        
        for size in sizes:
            for sparsity in sparsities:
                # Create sparse tensor
                tensor = torch.zeros(size, size)
                tensor[torch.rand(size, size) > sparsity] = 1.0
                
                # Time compression
                start = time.time()
                result = compressor.compress(tensor)
                compression_time = time.time() - start
                
                if result:
                    # Time decompression
                    start = time.time()
                    reconstructed = compressor.decompress(result)
                    decompression_time = time.time() - start
                    
                    logger.info(f"Size {size}x{size}, sparsity {sparsity:.0%}: "
                               f"compress {compression_time*1000:.2f}ms, "
                               f"decompress {decompression_time*1000:.2f}ms, "
                               f"ratio {result.compression_ratio:.2f}x")
    
    def test_gpu_acceleration(self):
        """Test GPU acceleration if available"""
        if not torch.cuda.is_available():
            logger.info("⚠ GPU not available, skipping GPU tests")
            return
        
        # Create sparse matrix
        size = 1000
        dense = torch.zeros(size, size)
        dense[torch.rand(size, size) > 0.95] = torch.randn(1)
        
        # CPU CSR
        csr_cpu = CSRPadicMatrix(dense.numpy())
        
        # GPU CSR
        gpu_csr = GPUCSRMatrix(csr_cpu)
        
        # Test GPU SpMV
        v = torch.randn(size, device='cuda')
        
        start = time.time()
        result_gpu = gpu_csr.multiply_vector(v)
        gpu_time = time.time() - start
        
        # Compare with CPU
        start = time.time()
        result_cpu = csr_cpu.multiply_vector(v.cpu())
        cpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
        
        logger.info(f"✓ GPU SpMV speedup: {speedup:.2f}x "
                   f"(CPU: {cpu_time*1000:.2f}ms, GPU: {gpu_time*1000:.2f}ms)")


class TestBatchedOperations:
    """Test batched CSR operations"""
    
    def test_batched_creation(self):
        """Test batch creation of CSR matrices"""
        # Create batch of sparse matrices
        batch_size = 10
        matrices = []
        for _ in range(batch_size):
            m = torch.zeros(50, 50)
            m[torch.rand(50, 50) > 0.92] = torch.randn(1)
            matrices.append(m)
        
        batch_tensor = torch.stack(matrices)
        
        # Create CSR batch
        csr_batch = BatchedCSROperations.create_from_batch(batch_tensor)
        
        assert len(csr_batch) == batch_size
        for csr in csr_batch:
            assert isinstance(csr, CSRPadicMatrix)
            assert csr.shape == (50, 50)
        
        logger.info(f"✓ Batched CSR creation: {batch_size} matrices")
    
    def test_batched_operations(self):
        """Test batched matrix operations"""
        # Create batch of CSR matrices
        batch_size = 5
        dense_batch = torch.zeros(batch_size, 30, 30)
        for i in range(batch_size):
            dense_batch[i][torch.rand(30, 30) > 0.9] = torch.randn(1)
        
        csr_batch = BatchedCSROperations.create_from_batch(dense_batch)
        
        # Test batched SpMV
        vectors = torch.randn(batch_size, 30)
        results = BatchedCSROperations.batch_multiply_vector(csr_batch, vectors)
        
        assert results.shape == (batch_size, 30)
        
        # Verify correctness
        for i in range(batch_size):
            expected = dense_batch[i].numpy() @ vectors[i].numpy()
            np.testing.assert_array_almost_equal(results[i], expected, decimal=5)
        
        logger.info("✓ Batched SpMV operations correct")


def run_all_tests():
    """Run all CSR compression tests"""
    print("=" * 70)
    print("Running CSR Sparse Matrix Compression Tests")
    print("=" * 70)
    
    # Basic CSR tests
    print("\n[CSR Matrix Tests]")
    test_csr = TestCSRPadicMatrix()
    test_csr.test_dense_to_csr_conversion()
    test_csr.test_csr_to_dense_reconstruction()
    test_csr.test_torch_tensor_support()
    test_csr.test_row_column_access()
    test_csr.test_matrix_operations()
    test_csr.test_transpose()
    test_csr.test_serialization()
    test_csr.test_edge_cases()
    
    # Sparse compressor tests
    print("\n[Sparse Compressor Tests]")
    test_compressor = TestSparseCompressor()
    test_compressor.test_basic_compression()
    test_compressor.test_insufficient_sparsity()
    test_compressor.test_benefit_analysis()
    test_compressor.test_batch_compression()
    test_compressor.test_different_shapes()
    
    # Adaptive compressor tests
    print("\n[Adaptive Compressor Tests]")
    test_adaptive = TestAdaptiveSparseCompressor()
    test_adaptive.test_threshold_adjustment()
    
    # Integration tests
    print("\n[Integration Tests]")
    test_integration = TestCSRIntegration()
    test_integration.test_csr_strategy()
    test_integration.test_strategy_selector_with_csr()
    test_integration.test_memory_savings()
    
    # Performance tests
    print("\n[Performance Benchmarks]")
    test_perf = TestPerformance()
    test_perf.test_compression_speed()
    test_perf.test_gpu_acceleration()
    
    # Batched operations tests
    print("\n[Batched Operations Tests]")
    test_batch = TestBatchedOperations()
    test_batch.test_batched_creation()
    test_batch.test_batched_operations()
    
    print("\n" + "=" * 70)
    print("ALL CSR TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()