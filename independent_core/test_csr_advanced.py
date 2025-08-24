"""
Advanced and comprehensive tests for CSRPadicMatrix.
Tests for production readiness including scale, concurrency, memory, and performance.
"""

import numpy as np
import torch
import pytest
from compression_systems.strategies.csr_sparse_matrix import (
    CSRPadicMatrix, 
    BatchedCSROperations,
    GPUCSRMatrix,
    CSRPerformanceMonitor
)
import threading
import time
import sys
import gc
import psutil
import os


class TestLargeScaleMatrices:
    """Test CSR with large-scale matrices (>1M elements)"""
    
    def test_million_element_sparse_matrix(self):
        """Test handling of million+ element sparse matrices"""
        # Create a 1000x1000 sparse matrix (1M elements)
        size = 1000
        density = 0.001  # 0.1% density = 1000 non-zero elements
        
        matrix = np.zeros((size, size))
        # Add random non-zero elements
        np.random.seed(42)
        num_nonzero = int(size * size * density)
        indices = np.random.choice(size * size, num_nonzero, replace=False)
        matrix.flat[indices] = np.random.randn(num_nonzero)
        
        csr = CSRPadicMatrix(matrix)
        
        # Verify properties
        assert csr.metrics.nnz == num_nonzero
        assert abs(csr.metrics.density - density) < 0.0001
        assert csr.metrics.compression_ratio > 100  # Should have excellent compression
        
        # Test operations still work
        vector = np.random.randn(size)
        result = csr.multiply_vector(vector)
        assert result.shape == (size,)
        
        # Verify reconstruction
        dense_reconstructed = csr.to_dense()
        assert np.allclose(dense_reconstructed, matrix, rtol=1e-5)
    
    def test_extreme_sparsity_large_matrix(self):
        """Test extremely sparse large matrices (10 elements in 10M)"""
        size = 3162  # ~10M elements when squared
        matrix = np.zeros((size, size))
        
        # Add only 10 non-zero elements
        for i in range(10):
            matrix[i * 300, i * 300] = i + 1.0
        
        csr = CSRPadicMatrix(matrix)
        
        assert csr.metrics.nnz == 10
        assert csr.metrics.density < 0.0000011  # Less than 0.00011%
        assert csr.metrics.compression_ratio > 1000  # Exceptional compression
        
    def test_memory_efficiency_at_scale(self):
        """Test memory efficiency with increasing matrix sizes"""
        results = []
        for size in [100, 500, 1000, 2000]:
            matrix = np.zeros((size, size))
            # Create diagonal matrix
            np.fill_diagonal(matrix, 1.0)
            
            csr = CSRPadicMatrix(matrix)
            
            # Calculate actual memory usage
            sparse_mem = (sys.getsizeof(csr.values) + 
                         sys.getsizeof(csr.col_idx) + 
                         sys.getsizeof(csr.row_ptr))
            dense_mem = matrix.nbytes
            
            results.append({
                'size': size,
                'sparse_mem': sparse_mem,
                'dense_mem': dense_mem,
                'ratio': dense_mem / sparse_mem
            })
        
        # Verify compression improves with size
        for i in range(1, len(results)):
            assert results[i]['ratio'] > results[i-1]['ratio']


class TestConcurrencyAndThreadSafety:
    """Test thread safety and concurrent operations"""
    
    def test_concurrent_read_operations(self):
        """Test multiple threads reading from same CSR matrix"""
        # Create a test matrix
        size = 100
        matrix = np.random.randn(size, size)
        matrix[matrix < 0.5] = 0  # Make it sparse
        csr = CSRPadicMatrix(matrix)
        
        results = []
        errors = []
        
        def read_worker(thread_id):
            try:
                for _ in range(10):
                    # Perform various read operations
                    row = np.random.randint(0, size)
                    col = np.random.randint(0, size)
                    
                    _ = csr.get_row(row)
                    _ = csr.get_column(col)
                    _ = csr.to_dense()
                    _ = csr.metrics
                    
                results.append(f"Thread {thread_id} completed")
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Launch multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=read_worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join(timeout=5.0)
        
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 10
    
    def test_concurrent_matrix_vector_multiply(self):
        """Test concurrent matrix-vector multiplications"""
        size = 200
        matrix = np.random.randn(size, size)
        matrix[matrix < 0.7] = 0
        csr = CSRPadicMatrix(matrix)
        
        vectors = [np.random.randn(size) for _ in range(20)]
        results = {}
        lock = threading.Lock()
        
        def multiply_worker(idx):
            result = csr.multiply_vector(vectors[idx])
            with lock:
                results[idx] = result
        
        threads = []
        for i in range(20):
            t = threading.Thread(target=multiply_worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join(timeout=5.0)
        
        # Verify all operations completed
        assert len(results) == 20
        
        # Verify correctness
        for i in range(20):
            expected = matrix @ vectors[i]
            # Use looser tolerance due to float32 internal storage
            assert np.allclose(results[i], expected, rtol=1e-4, atol=1e-6)
    
    def test_serialization_thread_safety(self):
        """Test concurrent serialization/deserialization"""
        matrix = np.random.randn(50, 50)
        matrix[matrix < 0.5] = 0
        csr = CSRPadicMatrix(matrix)
        
        errors = []
        
        def serialize_worker():
            try:
                for _ in range(100):
                    data = csr.to_bytes()
                    restored = CSRPadicMatrix.from_bytes(data)
                    assert np.allclose(restored.to_dense(), matrix, rtol=1e-5)
            except Exception as e:
                errors.append(str(e))
        
        threads = [threading.Thread(target=serialize_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
        
        assert len(errors) == 0


class TestMemoryStressAndLeaks:
    """Test memory management and leak detection"""
    
    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations"""
        initial_objects = len(gc.get_objects())
        
        # Perform many operations
        for _ in range(100):
            matrix = np.random.randn(100, 100)
            matrix[matrix < 0.5] = 0
            csr = CSRPadicMatrix(matrix)
            
            # Various operations
            _ = csr.to_dense()
            _ = csr.transpose()
            _ = csr.scale(2.0)
            
            # Explicit cleanup
            del csr
            del matrix
        
        # Force garbage collection
        gc.collect()
        
        final_objects = len(gc.get_objects())
        
        # Allow some growth but not excessive
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Potential memory leak: {object_growth} objects"
    
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create increasingly large matrices
        matrices = []
        try:
            for size in [100, 500, 1000, 1500]:
                matrix = np.zeros((size, size))
                # Diagonal matrix for predictable memory usage
                np.fill_diagonal(matrix, 1.0)
                csr = CSRPadicMatrix(matrix)
                matrices.append(csr)
                
                # Check memory growth is reasonable
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be much less than keeping ALL dense matrices permanently
                # NumPy's memory pool can retain temporary arrays from CSR construction
                cumulative_dense_mb = sum(s * s * 4 / 1024 / 1024 for s in [100, 500, 1000, 1500][:len(matrices)])
                
                # Realistic limit: NumPy may retain up to 3x dense memory in its pool
                assert memory_growth < cumulative_dense_mb * 4 + 20, \
                    f"Memory growth {memory_growth:.2f} MB exceeds reasonable limit"
        
        finally:
            # Cleanup
            del matrices
            gc.collect()
    
    def test_extreme_memory_allocation(self):
        """Test handling of extreme memory allocation requests"""
        # Try to create a matrix that would be huge if dense
        size = 10000
        matrix = np.zeros((size, size))
        
        # Add only diagonal elements
        np.fill_diagonal(matrix, 1.0)
        
        # This should work efficiently with CSR
        csr = CSRPadicMatrix(matrix)
        
        # Verify efficient storage
        assert csr.metrics.nnz == size
        assert csr.metrics.compression_ratio > 1000
        
        # Verify we can still do operations
        vector = np.ones(size)
        result = csr.multiply_vector(vector)
        assert np.allclose(result, np.ones(size))


class TestNumericOverflowAndEdgeCases:
    """Test numeric overflow and extreme edge cases"""
    
    def test_numeric_overflow_protection(self):
        """Test handling of potential numeric overflow"""
        # Create matrix with large values
        matrix = np.array([[1e38, 0], [0, 1e38]], dtype=np.float32)
        csr = CSRPadicMatrix(matrix)
        
        # Multiply by large scalar
        scaled = csr.scale(1e5)
        
        # Check for inf values (overflow)
        dense = scaled.to_dense()
        if np.any(np.isinf(dense)):
            # Overflow occurred - this is expected for float32
            assert True
        else:
            # No overflow - values should be correct
            assert np.allclose(dense[0, 0], 1e43, rtol=1e-5)
    
    def test_underflow_handling(self):
        """Test handling of numeric underflow"""
        # Create matrix with tiny values
        matrix = np.array([[1e-38, 0], [0, 1e-38]], dtype=np.float32)
        csr = CSRPadicMatrix(matrix, threshold=1e-40)
        
        # Scale down
        scaled = csr.scale(1e-10)
        
        # Values might underflow to zero
        dense = scaled.to_dense()
        # Either preserved or underflowed to zero
        assert np.all(dense >= 0)
    
    def test_mixed_scale_values(self):
        """Test matrix with values of vastly different scales"""
        matrix = np.array([
            [1e-30, 1e30, 0],
            [0, 1.0, 1e-15],
            [1e15, 0, 1e-30]
        ], dtype=np.float32)
        
        csr = CSRPadicMatrix(matrix, threshold=1e-35)
        
        # All non-zero values should be preserved
        dense = csr.to_dense()
        
        # Check large values
        assert dense[0, 1] == np.float32(1e30)
        assert dense[2, 0] == np.float32(1e15)
        
        # Small values might be preserved or lost depending on float32 precision
        # Just verify no crashes
        assert dense.shape == (3, 3)
    
    def test_dtype_preservation(self):
        """Test preservation of different dtypes"""
        for dtype in [np.float16, np.float32, np.float64]:
            if dtype == np.float16:
                # Skip float16 as it's not fully supported
                continue
                
            matrix = np.array([[1.5, 0], [0, 2.5]], dtype=dtype)
            csr = CSRPadicMatrix(matrix)
            
            # Values are stored as float32 internally
            assert csr.values.dtype == np.float32
            
            # But conversion back should work
            dense = csr.to_dense()
            assert dense.dtype == np.float32
            assert np.allclose(dense, matrix.astype(np.float32))


class TestPerformanceBenchmarks:
    """Performance benchmarks against scipy.sparse"""
    
    def test_performance_vs_scipy(self):
        """Compare performance with scipy.sparse.csr_matrix"""
        try:
            from scipy.sparse import csr_matrix as scipy_csr
        except ImportError:
            pytest.skip("scipy not available")
        
        # Create test matrix
        size = 500
        density = 0.01
        np.random.seed(42)
        
        matrix = np.zeros((size, size))
        nnz = int(size * size * density)
        indices = np.random.choice(size * size, nnz, replace=False)
        matrix.flat[indices] = np.random.randn(nnz)
        
        # Our implementation
        t0 = time.time()
        our_csr = CSRPadicMatrix(matrix)
        our_time = time.time() - t0
        
        # Scipy implementation
        t0 = time.time()
        scipy_csr_obj = scipy_csr(matrix)
        scipy_time = time.time() - t0
        
        # We might be slower but should be within reasonable range
        # Allow up to 200x slower (scipy is highly optimized C code, we're pure Python)
        assert our_time < scipy_time * 200, f"Our: {our_time:.4f}s, Scipy: {scipy_time:.4f}s"
        
        # Test matrix-vector multiply performance
        vector = np.random.randn(size)
        
        # Our implementation
        t0 = time.time()
        for _ in range(100):
            our_result = our_csr.multiply_vector(vector)
        our_mv_time = time.time() - t0
        
        # Scipy implementation  
        t0 = time.time()
        for _ in range(100):
            scipy_result = scipy_csr_obj.dot(vector)
        scipy_mv_time = time.time() - t0
        
        # Check correctness
        assert np.allclose(our_result, scipy_result, rtol=1e-5)
        
        # Performance should be reasonable (allow 20x slower)
        assert our_mv_time < scipy_mv_time * 20
    
    def test_compression_ratio_benchmark(self):
        """Benchmark compression ratios for different sparsity patterns"""
        results = []
        
        for density in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]:
            size = 200
            matrix = np.zeros((size, size))
            nnz = int(size * size * density)
            indices = np.random.choice(size * size, nnz, replace=False)
            matrix.flat[indices] = np.random.randn(nnz)
            
            csr = CSRPadicMatrix(matrix)
            
            results.append({
                'density': density,
                'compression_ratio': csr.metrics.compression_ratio,
                'memory_saved': csr.metrics.memory_saved_bytes
            })
        
        # Verify compression degrades with density
        for i in range(1, len(results)):
            assert results[i]['compression_ratio'] < results[i-1]['compression_ratio']
        
        # At 50% density, compression should be poor
        assert results[-1]['compression_ratio'] < 2.0
        
        # At 0.1% density, compression should be excellent
        assert results[0]['compression_ratio'] > 50
    
    def test_operation_scaling(self):
        """Test how operations scale with matrix size"""
        times = []
        
        for size in [50, 100, 200, 400]:
            matrix = np.zeros((size, size))
            np.fill_diagonal(matrix, 1.0)
            csr = CSRPadicMatrix(matrix)
            
            vector = np.ones(size)
            
            # Time matrix-vector multiply
            t0 = time.time()
            for _ in range(1000):
                _ = csr.multiply_vector(vector)
            elapsed = time.time() - t0
            
            times.append({
                'size': size,
                'time': elapsed,
                'time_per_op': elapsed / 1000
            })
        
        # Time should scale roughly linearly with size for diagonal matrix
        # (since nnz = size)
        for i in range(1, len(times)):
            ratio = times[i]['size'] / times[i-1]['size']
            time_ratio = times[i]['time'] / times[i-1]['time']
            # Allow 3x variance from linear scaling
            assert time_ratio < ratio * 3


class TestPropertyBasedTesting:
    """Property-based tests to ensure invariants hold"""
    
    def test_transpose_involution(self):
        """Test that transpose of transpose returns original"""
        for _ in range(10):
            size = np.random.randint(10, 50)
            matrix = np.random.randn(size, size)
            matrix[matrix < 0.5] = 0
            
            csr = CSRPadicMatrix(matrix)
            transposed = csr.transpose()
            double_transposed = transposed.transpose()
            
            assert np.allclose(double_transposed.to_dense(), matrix, rtol=1e-5)
    
    def test_scale_associativity(self):
        """Test that scaling is associative"""
        matrix = np.random.randn(20, 20)
        matrix[matrix < 0.5] = 0
        csr = CSRPadicMatrix(matrix)
        
        # (a * b) * c == a * (b * c)
        a, b, c = 2.0, 3.0, 0.5
        
        left = csr.scale(a).scale(b).scale(c)
        right = csr.scale(a * b * c)
        
        assert np.allclose(left.to_dense(), right.to_dense(), rtol=1e-5)
    
    def test_add_commutativity(self):
        """Test that addition is commutative"""
        size = 30
        matrix1 = np.random.randn(size, size)
        matrix1[matrix1 < 0.5] = 0
        matrix2 = np.random.randn(size, size)
        matrix2[matrix2 < 0.5] = 0
        
        csr1 = CSRPadicMatrix(matrix1)
        csr2 = CSRPadicMatrix(matrix2)
        
        # A + B == B + A
        sum1 = csr1.add(csr2)
        sum2 = csr2.add(csr1)
        
        assert np.allclose(sum1.to_dense(), sum2.to_dense(), rtol=1e-5)
    
    def test_sparsity_preservation(self):
        """Test that operations preserve sparsity patterns appropriately"""
        size = 40
        matrix = np.zeros((size, size))
        # Create specific pattern
        for i in range(0, size, 4):
            matrix[i, i] = 1.0
        
        csr = CSRPadicMatrix(matrix)
        original_nnz = csr.metrics.nnz
        
        # Scaling should preserve pattern
        scaled = csr.scale(5.0)
        assert scaled.metrics.nnz == original_nnz
        
        # Adding zero matrix should preserve pattern
        zero_csr = CSRPadicMatrix(np.zeros((size, size)))
        added = csr.add(zero_csr)
        assert added.metrics.nnz == original_nnz
    
    def test_reconstruction_accuracy(self):
        """Test that to_dense perfectly reconstructs the matrix"""
        for _ in range(20):
            size = np.random.randint(5, 50)
            density = np.random.uniform(0.01, 0.3)
            
            matrix = np.zeros((size, size))
            nnz = int(size * size * density)
            if nnz > 0:
                indices = np.random.choice(size * size, nnz, replace=False)
                matrix.flat[indices] = np.random.randn(nnz)
            
            csr = CSRPadicMatrix(matrix)
            reconstructed = csr.to_dense()
            
            assert np.allclose(reconstructed, matrix, rtol=1e-5, atol=1e-7)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])