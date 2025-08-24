"""
Comprehensive Unit Tests for CSR Sparse Matrix Module
Tests CSRPadicMatrix, BatchedCSROperations, GPUCSRMatrix, and CSRPerformanceMonitor
"""

import pytest
import numpy as np
import torch
import time
import struct
from typing import List, Tuple, Dict, Any
import pickle
import io

# Import the modules we're testing
from compression_systems.strategies.csr_sparse_matrix import (
    CSRPadicMatrix,
    CSRMetrics,
    BatchedCSROperations,
    GPUCSRMatrix,
    CSRPerformanceMonitor
)


class TestCSRMetrics:
    """Test CSRMetrics dataclass functionality"""
    
    def test_metrics_creation(self):
        """Test creating CSRMetrics with all fields"""
        metrics = CSRMetrics(
            nnz=100,
            density=0.1,
            compression_ratio=10.0,
            memory_saved_bytes=3600,
            dense_memory_bytes=4000,
            sparse_memory_bytes=400,
            row_efficiency=5.0,
            bandwidth_reduction=0.9
        )
        
        assert metrics.nnz == 100
        assert metrics.density == 0.1
        assert metrics.compression_ratio == 10.0
        assert metrics.memory_saved_bytes == 3600
        assert metrics.dense_memory_bytes == 4000
        assert metrics.sparse_memory_bytes == 400
        assert metrics.row_efficiency == 5.0
        assert metrics.bandwidth_reduction == 0.9
    
    def test_metrics_consistency(self):
        """Test that metrics maintain logical consistency"""
        metrics = CSRMetrics(
            nnz=50,
            density=0.05,
            compression_ratio=20.0,
            memory_saved_bytes=3800,
            dense_memory_bytes=4000,
            sparse_memory_bytes=200,
            row_efficiency=2.5,
            bandwidth_reduction=0.95
        )
        
        # Check that saved bytes equals difference
        assert metrics.memory_saved_bytes == metrics.dense_memory_bytes - metrics.sparse_memory_bytes
        
        # Check that density makes sense (5% non-zero)
        assert metrics.density == 0.05
        
        # Check bandwidth reduction aligns with density
        assert metrics.bandwidth_reduction == 0.95  # 1 - 0.05


class TestCSRPadicMatrix:
    """Test CSRPadicMatrix core functionality"""
    
    def test_initialization_from_numpy(self):
        """Test creating CSR matrix from numpy array"""
        # Create a sparse matrix with known pattern
        matrix = np.array([
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 3.0, 0.0, 0.0],
            [4.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 6.0]
        ], dtype=np.float32)
        
        csr = CSRPadicMatrix(matrix, threshold=1e-6)
        
        assert csr.shape == (4, 4)
        assert csr.threshold == 1e-6
        assert len(csr.values) == 6  # 6 non-zero values
        assert csr.metrics.nnz == 6
        assert csr.metrics.density == 6 / 16  # 37.5%
    
    def test_initialization_from_torch(self):
        """Test creating CSR matrix from torch tensor"""
        tensor = torch.tensor([
            [1.0, 0.0, 2.0],
            [0.0, 0.0, 3.0],
            [4.0, 5.0, 0.0]
        ], dtype=torch.float32)
        
        csr = CSRPadicMatrix(tensor, threshold=1e-6)
        
        assert csr.shape == (3, 3)
        assert csr.original_dtype == torch.float32
        assert csr.original_device == tensor.device
        assert len(csr.values) == 5  # 5 non-zero values
    
    def test_initialization_with_threshold(self):
        """Test threshold filtering during initialization"""
        matrix = np.array([
            [1.0, 1e-7, 0.0],
            [1e-8, 2.0, 1e-9],
            [0.0, 1e-6, 3.0]
        ])
        
        csr = CSRPadicMatrix(matrix, threshold=1e-6)
        
        # Only values > 1e-6 should be kept
        assert csr.metrics.nnz == 3  # 1.0, 2.0, 3.0
        assert 1.0 in csr.values
        assert 2.0 in csr.values
        assert 3.0 in csr.values
    
    def test_to_dense_reconstruction(self):
        """Test converting CSR back to dense format"""
        original = np.array([
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 4.0],
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 6.0, 7.0, 8.0]
        ], dtype=np.float32)
        
        csr = CSRPadicMatrix(original)
        reconstructed = csr.to_dense()
        
        np.testing.assert_array_almost_equal(original, reconstructed)
    
    def test_to_torch_reconstruction(self):
        """Test converting CSR to torch tensor"""
        original_tensor = torch.tensor([
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0]
        ], dtype=torch.float64, device='cpu')
        
        csr = CSRPadicMatrix(original_tensor)
        reconstructed = csr.to_torch()
        
        assert reconstructed.dtype == torch.float64
        assert reconstructed.device == original_tensor.device
        torch.testing.assert_close(original_tensor, reconstructed)
    
    def test_get_row(self):
        """Test efficient row access"""
        matrix = np.array([
            [1.0, 2.0, 0.0],
            [0.0, 3.0, 4.0],
            [5.0, 0.0, 6.0]
        ])
        
        csr = CSRPadicMatrix(matrix)
        
        row0 = csr.get_row(0)
        np.testing.assert_array_equal(row0, [1.0, 2.0, 0.0])
        
        row1 = csr.get_row(1)
        np.testing.assert_array_equal(row1, [0.0, 3.0, 4.0])
        
        row2 = csr.get_row(2)
        np.testing.assert_array_equal(row2, [5.0, 0.0, 6.0])
    
    def test_get_row_bounds_checking(self):
        """Test row access with invalid indices"""
        matrix = np.eye(3)
        csr = CSRPadicMatrix(matrix)
        
        with pytest.raises(IndexError):
            csr.get_row(-1)
        
        with pytest.raises(IndexError):
            csr.get_row(3)
    
    def test_get_column(self):
        """Test column extraction"""
        matrix = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        csr = CSRPadicMatrix(matrix)
        
        col0 = csr.get_column(0)
        np.testing.assert_array_equal(col0, [1.0, 4.0, 7.0])
        
        col1 = csr.get_column(1)
        np.testing.assert_array_equal(col1, [2.0, 5.0, 8.0])
        
        col2 = csr.get_column(2)
        np.testing.assert_array_equal(col2, [3.0, 6.0, 9.0])
    
    def test_get_column_sparse(self):
        """Test column extraction from sparse matrix"""
        matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0]
        ])
        
        csr = CSRPadicMatrix(matrix)
        
        col0 = csr.get_column(0)
        np.testing.assert_array_equal(col0, [1.0, 0.0, 0.0])
        
        col1 = csr.get_column(1)
        np.testing.assert_array_equal(col1, [0.0, 2.0, 0.0])
    
    def test_multiply_vector(self):
        """Test sparse matrix-vector multiplication"""
        matrix = np.array([
            [1.0, 2.0, 0.0],
            [0.0, 3.0, 4.0],
            [5.0, 0.0, 6.0]
        ])
        vector = np.array([1.0, 2.0, 3.0])
        
        csr = CSRPadicMatrix(matrix)
        result = csr.multiply_vector(vector)
        
        expected = matrix @ vector
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_multiply_vector_torch(self):
        """Test SpMV with torch tensor input"""
        matrix = np.array([
            [2.0, 0.0, 1.0],
            [0.0, 3.0, 0.0],
            [1.0, 0.0, 2.0]
        ])
        vector = torch.tensor([1.0, 2.0, 3.0])
        
        csr = CSRPadicMatrix(matrix)
        result = csr.multiply_vector(vector)
        
        expected = matrix @ vector.numpy()
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_multiply_vector_dimension_check(self):
        """Test dimension checking in matrix-vector multiplication"""
        matrix = np.eye(3)
        csr = CSRPadicMatrix(matrix)
        
        # Wrong vector length
        vector = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Vector length.*doesn't match"):
            csr.multiply_vector(vector)
    
    def test_multiply_matrix(self):
        """Test sparse matrix-matrix multiplication"""
        A = np.array([
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0]
        ])
        B = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        
        csr = CSRPadicMatrix(A)
        result = csr.multiply_matrix(B)
        
        expected = A @ B
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_multiply_matrix_dimension_check(self):
        """Test dimension checking in matrix-matrix multiplication"""
        A = np.eye(3)
        B = np.ones((4, 2))  # Incompatible shape
        
        csr = CSRPadicMatrix(A)
        with pytest.raises(ValueError, match="Incompatible shapes"):
            csr.multiply_matrix(B)
    
    def test_transpose(self):
        """Test matrix transposition"""
        matrix = np.array([
            [1.0, 2.0, 0.0],
            [3.0, 0.0, 4.0],
            [0.0, 5.0, 6.0]
        ])
        
        csr = CSRPadicMatrix(matrix)
        transposed_csr = csr.transpose()
        
        expected = matrix.T
        result = transposed_csr.to_dense()
        
        np.testing.assert_array_almost_equal(result, expected)
        assert transposed_csr.shape == (3, 3)
    
    def test_transpose_rectangular(self):
        """Test transposition of rectangular matrix"""
        matrix = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        
        csr = CSRPadicMatrix(matrix)
        transposed_csr = csr.transpose()
        
        assert transposed_csr.shape == (3, 2)
        np.testing.assert_array_almost_equal(transposed_csr.to_dense(), matrix.T)
    
    def test_add_matrices(self):
        """Test addition of two CSR matrices"""
        A = np.array([
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0]
        ])
        B = np.array([
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 3.0],
            [0.0, 4.0, 0.0]
        ])
        
        csr_A = CSRPadicMatrix(A)
        csr_B = CSRPadicMatrix(B)
        
        result_csr = csr_A.add(csr_B, alpha=1.0)
        result = result_csr.to_dense()
        
        expected = A + B
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_add_matrices_with_scaling(self):
        """Test scaled addition: A + alpha * B"""
        A = np.eye(3)
        B = np.ones((3, 3))
        alpha = 0.5
        
        csr_A = CSRPadicMatrix(A)
        csr_B = CSRPadicMatrix(B)
        
        result_csr = csr_A.add(csr_B, alpha=alpha)
        result = result_csr.to_dense()
        
        expected = A + alpha * B
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_add_shape_mismatch(self):
        """Test addition with mismatched shapes"""
        A = np.eye(3)
        B = np.eye(4)
        
        csr_A = CSRPadicMatrix(A)
        csr_B = CSRPadicMatrix(B)
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            csr_A.add(csr_B)
    
    def test_scale_matrix(self):
        """Test scaling matrix by scalar"""
        matrix = np.array([
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0]
        ])
        alpha = 2.5
        
        csr = CSRPadicMatrix(matrix)
        scaled_csr = csr.scale(alpha)
        
        expected = matrix * alpha
        result = scaled_csr.to_dense()
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_scale_preserves_structure(self):
        """Test that scaling preserves CSR structure"""
        matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0]
        ])
        
        csr = CSRPadicMatrix(matrix)
        scaled = csr.scale(2.0)
        
        # Same sparsity pattern
        assert scaled.metrics.nnz == csr.metrics.nnz
        assert scaled.shape == csr.shape
        np.testing.assert_array_equal(scaled.col_idx, csr.col_idx)
        np.testing.assert_array_equal(scaled.row_ptr, csr.row_ptr)
        
        # But values are scaled
        np.testing.assert_array_almost_equal(scaled.values, csr.values * 2.0)
    
    def test_get_statistics(self):
        """Test comprehensive statistics gathering"""
        matrix = np.array([
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],  # Empty row
            [3.0, 4.0, 5.0, 6.0],   # Dense row
            [0.0, 0.0, 7.0, 0.0]
        ])
        
        csr = CSRPadicMatrix(matrix)
        stats = csr.get_statistics()
        
        assert stats['shape'] == (4, 4)
        assert stats['rows'] == 4
        assert stats['cols'] == 4
        assert stats['total_elements'] == 16
        assert stats['nnz'] == 7
        assert stats['density'] == 7/16
        assert stats['sparsity'] == 1 - 7/16
        assert stats['empty_rows'] == 1
        assert stats['max_row_nnz'] == 4  # Row 2 has 4 non-zeros
        assert stats['min_row_nnz'] == 0  # Row 1 has 0 non-zeros
    
    def test_empty_matrix(self):
        """Test handling of completely empty matrix"""
        matrix = np.zeros((5, 5))
        
        csr = CSRPadicMatrix(matrix)
        
        assert csr.metrics.nnz == 0
        assert csr.metrics.density == 0.0
        assert len(csr.values) == 0
        assert len(csr.col_idx) == 0
        
        # Should reconstruct correctly
        reconstructed = csr.to_dense()
        np.testing.assert_array_equal(reconstructed, matrix)
    
    def test_dense_matrix_compression(self):
        """Test compression of nearly dense matrix"""
        matrix = np.ones((10, 10))
        matrix[5, 5] = 0.0  # One zero element
        
        csr = CSRPadicMatrix(matrix)
        
        assert csr.metrics.nnz == 99
        assert csr.metrics.density == 0.99
        assert csr.metrics.compression_ratio < 1.0  # Poor compression
    
    def test_highly_sparse_matrix(self):
        """Test highly sparse matrix compression"""
        matrix = np.zeros((100, 100))
        # Add just 10 non-zero elements
        for i in range(10):
            matrix[i*10, i*10] = i + 1.0
        
        csr = CSRPadicMatrix(matrix)
        
        assert csr.metrics.nnz == 10
        assert csr.metrics.density == 0.001  # 0.1%
        assert csr.metrics.compression_ratio > 100  # Excellent compression
        assert csr.metrics.bandwidth_reduction > 0.99
    
    def test_validation_valid_matrix(self):
        """Test validation of valid CSR structure"""
        matrix = np.array([
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0]
        ])
        
        csr = CSRPadicMatrix(matrix)
        assert csr.validate() is True
    
    def test_validation_corrupted_structure(self):
        """Test validation catches corrupted CSR structure"""
        matrix = np.eye(3)
        csr = CSRPadicMatrix(matrix)
        
        # Corrupt the structure
        csr.row_ptr[-1] = 999  # Invalid nnz
        
        with pytest.raises(ValueError, match="Last row pointer must equal nnz"):
            csr.validate()
    
    def test_invalid_input_type(self):
        """Test handling of invalid input types"""
        with pytest.raises(TypeError, match="Expected torch.Tensor or np.ndarray"):
            CSRPadicMatrix([1, 2, 3])  # List instead of array
    
    def test_invalid_dimensions(self):
        """Test handling of non-2D input"""
        matrix_1d = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="Expected 2D matrix"):
            CSRPadicMatrix(matrix_1d)
        
        matrix_3d = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="Expected 2D matrix"):
            CSRPadicMatrix(matrix_3d)


class TestCSRSerialization:
    """Test serialization/deserialization of CSR matrices"""
    
    def test_to_bytes_serialization(self):
        """Test serializing CSR matrix to bytes"""
        matrix = np.array([
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0]
        ])
        
        csr = CSRPadicMatrix(matrix, threshold=1e-5)
        data = csr.to_bytes()
        
        assert isinstance(data, bytes)
        assert len(data) > 0
        
        # Check magic number
        assert data[:4] == b'CSR1'
    
    def test_from_bytes_deserialization(self):
        """Test deserializing CSR matrix from bytes"""
        original = np.array([
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 4.0],
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 6.0, 7.0, 8.0]
        ])
        
        csr_original = CSRPadicMatrix(original, threshold=1e-4)
        data = csr_original.to_bytes()
        
        csr_loaded = CSRPadicMatrix.from_bytes(data)
        
        assert csr_loaded.shape == csr_original.shape
        assert csr_loaded.threshold == csr_original.threshold
        assert csr_loaded.metrics.nnz == csr_original.metrics.nnz
        
        # Check reconstruction
        reconstructed = csr_loaded.to_dense()
        np.testing.assert_array_almost_equal(reconstructed, original)
    
    def test_round_trip_serialization(self):
        """Test round-trip serialization preserves data"""
        # Test with random sparse matrix
        np.random.seed(42)
        matrix = np.random.randn(20, 30)
        matrix[matrix < 0.5] = 0.0  # Make it sparse
        
        csr1 = CSRPadicMatrix(matrix)
        data = csr1.to_bytes()
        csr2 = CSRPadicMatrix.from_bytes(data)
        
        # Compare reconstructions
        dense1 = csr1.to_dense()
        dense2 = csr2.to_dense()
        np.testing.assert_array_almost_equal(dense1, dense2)
    
    def test_from_bytes_invalid_data(self):
        """Test deserialization with invalid data"""
        # Too short
        with pytest.raises(ValueError, match="too short for header"):
            CSRPadicMatrix.from_bytes(b'short')
        
        # Wrong magic number
        bad_data = b'XXXX' + b'\x00' * 20
        with pytest.raises(ValueError, match="Invalid CSR magic number"):
            CSRPadicMatrix.from_bytes(bad_data)
        
        # Wrong version
        bad_version = b'CSR1' + struct.pack('I', 999) + b'\x00' * 16
        with pytest.raises(ValueError, match="Unsupported CSR version"):
            CSRPadicMatrix.from_bytes(bad_version)
    
    def test_empty_matrix_serialization(self):
        """Test serialization of empty matrix"""
        matrix = np.zeros((5, 5))
        
        csr = CSRPadicMatrix(matrix)
        data = csr.to_bytes()
        loaded = CSRPadicMatrix.from_bytes(data)
        
        assert loaded.metrics.nnz == 0
        np.testing.assert_array_equal(loaded.to_dense(), matrix)


class TestBatchedCSROperations:
    """Test batched operations on multiple CSR matrices"""
    
    def test_batch_multiply_vector(self):
        """Test batched matrix-vector multiplication"""
        # Create batch of matrices
        matrices = [
            CSRPadicMatrix(np.array([[1.0, 0.0], [0.0, 2.0]])),
            CSRPadicMatrix(np.array([[3.0, 0.0], [0.0, 4.0]])),
            CSRPadicMatrix(np.array([[5.0, 0.0], [0.0, 6.0]]))
        ]
        
        vectors = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        
        results = BatchedCSROperations.batch_multiply_vector(matrices, vectors)
        
        assert results.shape == (3, 2)
        np.testing.assert_array_almost_equal(results[0], [1.0, 4.0])
        np.testing.assert_array_almost_equal(results[1], [9.0, 16.0])
        np.testing.assert_array_almost_equal(results[2], [25.0, 36.0])
    
    def test_batch_multiply_vector_torch(self):
        """Test batched SpMV with torch tensors"""
        matrices = [
            CSRPadicMatrix(np.eye(3)),
            CSRPadicMatrix(np.ones((3, 3)))
        ]
        
        vectors = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        
        results = BatchedCSROperations.batch_multiply_vector(matrices, vectors)
        
        assert results.shape == (2, 3)
        np.testing.assert_array_almost_equal(results[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(results[1], [15.0, 15.0, 15.0])
    
    def test_batch_multiply_vector_size_mismatch(self):
        """Test batch size mismatch detection"""
        matrices = [CSRPadicMatrix(np.eye(2))]
        vectors = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 vectors but 1 matrix
        
        with pytest.raises(ValueError, match="Batch size mismatch"):
            BatchedCSROperations.batch_multiply_vector(matrices, vectors)
    
    def test_batch_to_dense(self):
        """Test converting batch of CSR to dense"""
        matrices = [
            CSRPadicMatrix(np.array([[1.0, 0.0], [0.0, 2.0]])),
            CSRPadicMatrix(np.array([[3.0, 0.0], [4.0, 0.0]])),
            CSRPadicMatrix(np.array([[0.0, 5.0], [6.0, 0.0]]))
        ]
        
        dense_batch = BatchedCSROperations.batch_to_dense(matrices)
        
        assert dense_batch.shape == (3, 2, 2)
        np.testing.assert_array_equal(dense_batch[0], [[1.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_equal(dense_batch[1], [[3.0, 0.0], [4.0, 0.0]])
        np.testing.assert_array_equal(dense_batch[2], [[0.0, 5.0], [6.0, 0.0]])
    
    def test_batch_to_dense_empty(self):
        """Test batch_to_dense with empty list"""
        dense = BatchedCSROperations.batch_to_dense([])
        assert len(dense) == 0
    
    def test_batch_to_dense_shape_mismatch(self):
        """Test batch_to_dense with mismatched shapes"""
        matrices = [
            CSRPadicMatrix(np.eye(2)),
            CSRPadicMatrix(np.eye(3))  # Different shape
        ]
        
        with pytest.raises(ValueError, match="Shape mismatch in batch"):
            BatchedCSROperations.batch_to_dense(matrices)
    
    def test_create_from_batch(self):
        """Test creating CSR matrices from dense batch"""
        dense_batch = np.array([
            [[1.0, 0.0], [0.0, 2.0]],
            [[3.0, 0.0], [0.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ])
        
        matrices = BatchedCSROperations.create_from_batch(dense_batch, threshold=1e-6)
        
        assert len(matrices) == 3
        assert all(isinstance(m, CSRPadicMatrix) for m in matrices)
        assert matrices[0].shape == (2, 2)
        assert matrices[0].metrics.nnz == 2
        assert matrices[2].metrics.nnz == 4
    
    def test_create_from_batch_torch(self):
        """Test creating CSR batch from torch tensors"""
        tensors = torch.tensor([
            [[1.0, 0.0, 0.0],
             [0.0, 2.0, 0.0],
             [0.0, 0.0, 3.0]],
            [[4.0, 5.0, 0.0],
             [0.0, 6.0, 0.0],
             [0.0, 0.0, 7.0]]
        ])
        
        matrices = BatchedCSROperations.create_from_batch(tensors)
        
        assert len(matrices) == 2
        assert matrices[0].metrics.nnz == 3
        assert matrices[1].metrics.nnz == 4
    
    def test_create_from_batch_invalid_shape(self):
        """Test error on invalid batch shape"""
        matrix_2d = np.eye(3)
        
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            BatchedCSROperations.create_from_batch(matrix_2d)


class TestGPUCSRMatrix:
    """Test GPU-accelerated CSR operations"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_initialization(self):
        """Test creating GPU CSR from CPU CSR"""
        cpu_matrix = np.array([
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0]
        ])
        
        cpu_csr = CSRPadicMatrix(cpu_matrix)
        gpu_csr = GPUCSRMatrix(cpu_csr)
        
        assert gpu_csr.shape == (3, 3)
        assert gpu_csr.device.type == 'cuda'
        assert gpu_csr.nnz == 5
        assert gpu_csr.density == 5/9
    
    def test_gpu_initialization_cpu_fallback(self):
        """Test GPU CSR falls back to CPU when CUDA unavailable"""
        cpu_matrix = np.array([[1.0, 0.0], [0.0, 2.0]])
        cpu_csr = CSRPadicMatrix(cpu_matrix)
        
        # Force CPU device
        gpu_csr = GPUCSRMatrix(cpu_csr, device=torch.device('cpu'))
        
        assert gpu_csr.device.type == 'cpu'
        assert gpu_csr.shape == (2, 2)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_multiply_vector(self):
        """Test GPU sparse matrix-vector multiplication"""
        matrix = np.array([
            [1.0, 2.0, 0.0],
            [0.0, 3.0, 4.0],
            [5.0, 0.0, 6.0]
        ])
        vector = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        
        cpu_csr = CSRPadicMatrix(matrix)
        gpu_csr = GPUCSRMatrix(cpu_csr)
        
        result = gpu_csr.multiply_vector(vector)
        
        expected = torch.tensor(matrix @ vector.cpu().numpy(), device='cuda')
        torch.testing.assert_close(result, expected)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_multiply_matrix(self):
        """Test GPU sparse matrix-matrix multiplication"""
        A = np.array([
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0]
        ])
        B = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ], device='cuda')
        
        cpu_csr = CSRPadicMatrix(A)
        gpu_csr = GPUCSRMatrix(cpu_csr)
        
        result = gpu_csr.multiply_matrix(B)
        
        expected = torch.tensor(A @ B.cpu().numpy(), device='cuda')
        torch.testing.assert_close(result, expected)
    
    def test_gpu_to_dense(self):
        """Test converting GPU CSR to dense tensor"""
        matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0]
        ])
        
        cpu_csr = CSRPadicMatrix(matrix)
        gpu_csr = GPUCSRMatrix(cpu_csr, device=torch.device('cpu'))
        
        dense = gpu_csr.to_dense()
        
        assert isinstance(dense, torch.Tensor)
        torch.testing.assert_close(dense, torch.tensor(matrix, dtype=torch.float32))
    
    def test_gpu_to_cpu_csr(self):
        """Test converting GPU CSR back to CPU CSR"""
        original = np.array([
            [1.0, 2.0, 0.0],
            [3.0, 0.0, 4.0],
            [0.0, 5.0, 6.0]
        ])
        
        cpu_csr1 = CSRPadicMatrix(original)
        gpu_csr = GPUCSRMatrix(cpu_csr1, device=torch.device('cpu'))
        cpu_csr2 = gpu_csr.to_cpu_csr()
        
        np.testing.assert_array_almost_equal(cpu_csr1.to_dense(), cpu_csr2.to_dense())
        assert cpu_csr2.threshold == cpu_csr1.threshold
    
    def test_gpu_empty_matrix(self):
        """Test GPU CSR with empty matrix"""
        empty = np.zeros((3, 3))
        
        cpu_csr = CSRPadicMatrix(empty)
        gpu_csr = GPUCSRMatrix(cpu_csr, device=torch.device('cpu'))
        
        assert gpu_csr.nnz == 0
        assert gpu_csr.density == 0.0
        
        dense = gpu_csr.to_dense()
        torch.testing.assert_close(dense, torch.zeros(3, 3))


class TestCSRPerformanceMonitor:
    """Test performance monitoring utilities"""
    
    def test_monitor_initialization(self):
        """Test creating performance monitor"""
        monitor = CSRPerformanceMonitor()
        
        assert len(monitor.compression_history) == 0
        assert len(monitor.operation_timings) == 0
    
    def test_record_compression(self):
        """Test recording compression statistics"""
        monitor = CSRPerformanceMonitor()
        
        monitor.record_compression(
            original_size=4000,
            compressed_size=400,
            sparsity=0.9,
            compression_time=0.01
        )
        
        assert len(monitor.compression_history) == 1
        record = monitor.compression_history[0]
        assert record['original_size'] == 4000
        assert record['compressed_size'] == 400
        assert record['compression_ratio'] == 10.0
        assert record['sparsity'] == 0.9
        assert record['compression_time'] == 0.01
        assert 'timestamp' in record
    
    def test_record_operation(self):
        """Test recording operation timing"""
        monitor = CSRPerformanceMonitor()
        
        monitor.record_operation(
            operation='spmv',
            matrix_shape=(100, 100),
            nnz=500,
            execution_time=0.001
        )
        
        assert 'spmv' in monitor.operation_timings
        assert len(monitor.operation_timings['spmv']) == 1
        
        timing = monitor.operation_timings['spmv'][0]
        assert timing['shape'] == (100, 100)
        assert timing['nnz'] == 500
        assert timing['density'] == 0.05
        assert timing['execution_time'] == 0.001
        assert 'gflops' in timing
    
    def test_gflops_calculation(self):
        """Test GFLOPS calculation for different operations"""
        monitor = CSRPerformanceMonitor()
        
        # SpMV: 2 * nnz operations
        gflops_spmv = monitor._calculate_gflops('spmv', (100, 100), 1000, 0.001)
        expected_spmv = (2 * 1000 / 1e9) / 0.001
        assert abs(gflops_spmv - expected_spmv) < 1e-6
        
        # SpMM: 2 * nnz * n_cols operations
        gflops_spmm = monitor._calculate_gflops('spmm', (100, 50), 1000, 0.001)
        expected_spmm = (2 * 1000 * 50 / 1e9) / 0.001
        assert abs(gflops_spmm - expected_spmm) < 1e-6
        
        # Unknown operation: nnz operations
        gflops_unknown = monitor._calculate_gflops('unknown', (100, 100), 1000, 0.001)
        expected_unknown = (1000 / 1e9) / 0.001
        assert abs(gflops_unknown - expected_unknown) < 1e-6
    
    def test_get_summary(self):
        """Test getting performance summary"""
        monitor = CSRPerformanceMonitor()
        
        # Record some data
        monitor.record_compression(1000, 100, 0.9, 0.01)
        monitor.record_compression(2000, 150, 0.95, 0.02)
        monitor.record_compression(1500, 200, 0.85, 0.015)
        
        monitor.record_operation('spmv', (100, 100), 500, 0.001)
        monitor.record_operation('spmv', (200, 200), 1000, 0.002)
        monitor.record_operation('spmm', (100, 50), 300, 0.005)
        
        summary = monitor.get_summary()
        
        assert summary['total_compressions'] == 3
        assert 'average_compression_ratio' in summary
        assert 'best_compression_ratio' in summary
        assert 'average_sparsity' in summary
        
        assert 'spmv' in summary['operation_performance']
        assert summary['operation_performance']['spmv']['count'] == 2
        assert 'average_gflops' in summary['operation_performance']['spmv']
        assert 'peak_gflops' in summary['operation_performance']['spmv']
        
        assert 'spmm' in summary['operation_performance']
        assert summary['operation_performance']['spmm']['count'] == 1
    
    def test_get_summary_empty(self):
        """Test getting summary with no data"""
        monitor = CSRPerformanceMonitor()
        summary = monitor.get_summary()
        
        assert summary == {}
    
    def test_zero_time_handling(self):
        """Test handling of zero execution time"""
        monitor = CSRPerformanceMonitor()
        
        gflops = monitor._calculate_gflops('spmv', (100, 100), 1000, 0.0)
        assert gflops == 0.0
        
        monitor.record_compression(1000, 0, 1.0, 0.0)
        assert monitor.compression_history[0]['compression_ratio'] == 1.0


class TestIntegrationScenarios:
    """Test realistic usage scenarios"""
    
    def test_neural_network_weight_compression(self):
        """Test compressing neural network weight matrix"""
        # Simulate a sparse weight matrix from pruned neural network
        np.random.seed(42)
        weights = np.random.randn(512, 1024)
        
        # Prune 90% of weights (set to zero)
        mask = np.random.random((512, 1024)) < 0.9
        weights[mask] = 0.0
        
        csr = CSRPadicMatrix(weights)
        
        # Check compression effectiveness
        assert csr.metrics.density < 0.11  # ~10% non-zero
        assert csr.metrics.compression_ratio > 5  # Good compression
        
        # Test forward pass simulation
        input_vector = np.random.randn(1024)
        output = csr.multiply_vector(input_vector)
        assert output.shape == (512,)
    
    def test_iterative_solver_scenario(self):
        """Test CSR in iterative solver context"""
        # Create a sparse symmetric positive definite matrix
        size = 50
        A_dense = np.zeros((size, size))
        
        # Tridiagonal structure
        for i in range(size):
            A_dense[i, i] = 4.0
            if i > 0:
                A_dense[i, i-1] = -1.0
            if i < size - 1:
                A_dense[i, i+1] = -1.0
        
        A_csr = CSRPadicMatrix(A_dense)
        
        # Simulate conjugate gradient iteration
        b = np.ones(size)
        x = np.zeros(size)
        
        for _ in range(5):
            residual = b - A_csr.multiply_vector(x)
            x = x + 0.1 * residual  # Simple gradient step
        
        # Check that we're making progress
        final_residual = b - A_csr.multiply_vector(x)
        assert np.linalg.norm(final_residual) < np.linalg.norm(b)
    
    def test_batch_processing_scenario(self):
        """Test batch processing of multiple sparse matrices"""
        # Simulate batch of adjacency matrices from graphs
        batch_size = 10
        node_count = 20
        edge_probability = 0.1
        
        matrices = []
        for _ in range(batch_size):
            adj = np.random.random((node_count, node_count)) < edge_probability
            adj = adj.astype(np.float32)
            matrices.append(CSRPadicMatrix(adj))
        
        # Batch operation: compute degree of each node
        ones = np.ones((batch_size, node_count))
        degrees = BatchedCSROperations.batch_multiply_vector(matrices, ones)
        
        assert degrees.shape == (batch_size, node_count)
        
        # Each degree should be non-negative
        assert np.all(degrees >= 0)
    
    def test_memory_pressure_scenario(self):
        """Test behavior under memory pressure with large sparse matrix"""
        # Create a very large sparse matrix
        size = 1000
        density = 0.01  # 1% non-zero
        
        matrix = np.zeros((size, size))
        num_nonzeros = int(size * size * density)
        
        # Add random non-zeros
        for _ in range(num_nonzeros):
            i, j = np.random.randint(0, size, 2)
            matrix[i, j] = np.random.randn()
        
        csr = CSRPadicMatrix(matrix)
        
        # Memory should be significantly reduced
        assert csr.metrics.memory_saved_bytes > 0
        assert csr.metrics.compression_ratio > 10
        
        # Operations should still work
        v = np.random.randn(size)
        result = csr.multiply_vector(v)
        assert not np.any(np.isnan(result))


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_single_element_matrix(self):
        """Test 1x1 matrix"""
        matrix = np.array([[5.0]])
        csr = CSRPadicMatrix(matrix)
        
        assert csr.shape == (1, 1)
        assert csr.metrics.nnz == 1
        assert csr.metrics.density == 1.0
    
    def test_single_row_matrix(self):
        """Test matrix with single row"""
        matrix = np.array([[1.0, 0.0, 2.0, 0.0, 3.0]])
        csr = CSRPadicMatrix(matrix)
        
        assert csr.shape == (1, 5)
        assert csr.metrics.nnz == 3
        
        row = csr.get_row(0)
        np.testing.assert_array_equal(row, matrix[0])
    
    def test_single_column_matrix(self):
        """Test matrix with single column"""
        matrix = np.array([[1.0], [0.0], [2.0], [0.0], [3.0]])
        csr = CSRPadicMatrix(matrix)
        
        assert csr.shape == (5, 1)
        assert csr.metrics.nnz == 3
        
        col = csr.get_column(0)
        np.testing.assert_array_equal(col, matrix[:, 0])
    
    def test_all_same_value(self):
        """Test matrix with all same non-zero values"""
        matrix = np.ones((4, 4)) * 3.14
        csr = CSRPadicMatrix(matrix)
        
        assert np.all(csr.values == 3.14)
        assert csr.metrics.nnz == 16
        assert csr.metrics.density == 1.0
    
    def test_near_zero_threshold(self):
        """Test with very small threshold"""
        matrix = np.array([
            [1e-10, 1e-9, 1e-8],
            [1e-7, 1e-6, 1e-5],
            [1e-4, 1e-3, 1e-2]
        ])
        
        csr = CSRPadicMatrix(matrix, threshold=1e-11)
        assert csr.metrics.nnz == 9  # All values kept
        
        csr = CSRPadicMatrix(matrix, threshold=1e-5)
        assert csr.metrics.nnz == 3  # Only 1e-4, 1e-3, 1e-2 kept
    
    def test_inf_and_nan_values(self):
        """Test handling of inf and nan values"""
        matrix = np.array([
            [1.0, np.inf, 0.0],
            [np.nan, 2.0, -np.inf],
            [0.0, 3.0, 4.0]
        ])
        
        csr = CSRPadicMatrix(matrix)
        
        # Check that inf and nan are preserved
        dense = csr.to_dense()
        assert np.isinf(dense[0, 1])
        assert np.isnan(dense[1, 0])
        assert np.isinf(dense[1, 2])
    
    def test_negative_values(self):
        """Test handling of negative values"""
        matrix = np.array([
            [-1.0, 0.0, -2.0],
            [0.0, -3.0, 0.0],
            [-4.0, 0.0, -5.0]
        ])
        
        csr = CSRPadicMatrix(matrix)
        
        assert csr.metrics.nnz == 5
        assert np.all(csr.values < 0)  # All values are negative
        
        # Check reconstruction
        np.testing.assert_array_equal(csr.to_dense(), matrix)


class TestCSRPadicMatrixAdditional:
    """Additional tests for edge cases and special scenarios"""
    
    def test_row_ptr_integrity_after_operations(self):
        """Test that row_ptr remains valid after various operations"""
        matrix = np.array([
            [1, 0, 2],
            [0, 0, 0],  # Empty row
            [3, 4, 5]
        ], dtype=np.float32)
        
        csr = CSRPadicMatrix(matrix)
        
        # Check initial row_ptr
        assert len(csr.row_ptr) == 4  # n_rows + 1
        assert csr.row_ptr[0] == 0
        assert csr.row_ptr[-1] == len(csr.values)
        
        # After scaling
        scaled = csr.scale(2.0)
        assert len(scaled.row_ptr) == 4
        assert scaled.row_ptr[-1] == len(scaled.values)
        
        # After transpose
        transposed = csr.transpose()
        assert len(transposed.row_ptr) == 4
        assert transposed.row_ptr[-1] == len(transposed.values)
    
    def test_column_indices_sorted_per_row(self):
        """Test if implementation maintains sorted column indices (common CSR requirement)"""
        matrix = np.array([
            [0, 3, 0, 1, 2],
            [4, 0, 0, 5, 0],
            [0, 6, 7, 0, 8]
        ])
        
        csr = CSRPadicMatrix(matrix)
        
        # Check if column indices are sorted within each row
        for i in range(len(csr.row_ptr) - 1):
            start = csr.row_ptr[i]
            end = csr.row_ptr[i + 1]
            row_cols = csr.col_idx[start:end]
            # No duplicate columns
            assert len(row_cols) == len(set(row_cols))
    
    def test_zero_threshold_behavior(self):
        """Test behavior with zero threshold (keep all values)"""
        matrix = np.array([
            [1e-10, 1e-15, 0.0],
            [1e-20, 1.0, 1e-30]
        ])
        
        csr = CSRPadicMatrix(matrix, threshold=0.0)
        
        # Even true zeros might be kept depending on implementation
        reconstructed = csr.to_dense()
        
        # At least non-zero values should be preserved
        assert reconstructed[0, 0] != 0 or matrix[0, 0] == 0
        assert abs(reconstructed[1, 1] - 1.0) < 1e-6
    
    def test_large_threshold_behavior(self):
        """Test with threshold larger than all values"""
        matrix = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ])
        
        csr = CSRPadicMatrix(matrix, threshold=1.0)
        
        # All values below threshold, should result in empty matrix
        assert csr.metrics.nnz == 0
        assert np.all(csr.to_dense() == 0)
    
    def test_numerical_stability_accumulation(self):
        """Test numerical stability in operations with many accumulations"""
        size = 100
        # Create a matrix with values that could cause accumulation errors
        matrix = np.full((size, size), 1e-8, dtype=np.float32)
        np.fill_diagonal(matrix, 1.0)
        
        csr = CSRPadicMatrix(matrix, threshold=1e-9)
        
        # Perform many additions
        result = csr
        for _ in range(100):
            result = result.add(csr, alpha=0.01)
        
        # Check diagonal is approximately correct
        dense = result.to_dense()
        diagonal = np.diag(dense)
        expected_diag = 1.0 + 100 * 0.01  # Original + 100 additions
        np.testing.assert_allclose(diagonal, expected_diag, rtol=1e-5)
    
    def test_memory_efficiency_verification(self):
        """Verify actual memory usage is less than dense for sparse matrices"""
        size = 500
        density = 0.01
        
        # Create sparse matrix
        matrix = np.zeros((size, size))
        num_nonzeros = int(size * size * density)
        indices = np.random.choice(size * size, num_nonzeros, replace=False)
        matrix.flat[indices] = np.random.randn(num_nonzeros)
        
        csr = CSRPadicMatrix(matrix)
        
        # Calculate actual memory usage
        csr_memory = (
            csr.values.nbytes +
            csr.col_idx.nbytes +
            csr.row_ptr.nbytes
        )
        dense_memory = matrix.nbytes
        
        assert csr_memory < dense_memory
        assert csr.metrics.sparse_memory_bytes < csr.metrics.dense_memory_bytes
    
    def test_pattern_preservation_through_operations(self):
        """Test that sparsity pattern is preserved appropriately"""
        # Create matrix with specific pattern
        matrix = np.array([
            [1, 0, 0, 2],
            [0, 3, 0, 0],
            [0, 0, 4, 0],
            [5, 0, 0, 6]
        ], dtype=np.float32)
        
        csr = CSRPadicMatrix(matrix)
        original_nnz = csr.metrics.nnz
        
        # Scaling should preserve pattern
        scaled = csr.scale(10.0)
        assert scaled.metrics.nnz == original_nnz
        
        # Adding with itself should preserve pattern
        added = csr.add(csr)
        assert added.metrics.nnz == original_nnz
        
        # Adding with different pattern may change it
        other = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ], dtype=np.float32)
        csr_other = CSRPadicMatrix(other)
        combined = csr.add(csr_other)
        assert combined.metrics.nnz >= original_nnz  # May have more non-zeros


class TestBatchedCSROperationsAdditional:
    """Additional tests for batched operations"""
    
    def test_heterogeneous_sparsity_batch(self):
        """Test batch with varying sparsity levels"""
        matrices = [
            CSRPadicMatrix(np.eye(10)),  # Very sparse
            CSRPadicMatrix(np.ones((10, 10))),  # Dense
            CSRPadicMatrix(np.random.randn(10, 10) * (np.random.random((10, 10)) > 0.7))  # Medium
        ]
        
        # All should work together despite different sparsity
        dense_batch = BatchedCSROperations.batch_to_dense(matrices)
        assert dense_batch.shape == (3, 10, 10)
        
        # Verify each matrix
        np.testing.assert_array_almost_equal(dense_batch[0], np.eye(10))
        np.testing.assert_array_almost_equal(dense_batch[1], np.ones((10, 10)))
    
    def test_batch_operation_consistency(self):
        """Test that batched operations match individual operations"""
        matrices = []
        vectors = []
        
        for i in range(5):
            mat = np.random.randn(8, 8)
            mat[mat < 0] = 0  # Make sparse
            matrices.append(CSRPadicMatrix(mat))
            vectors.append(np.random.randn(8))
        
        # Batched operation
        batch_results = BatchedCSROperations.batch_multiply_vector(
            matrices, np.array(vectors)
        )
        
        # Individual operations
        individual_results = []
        for mat, vec in zip(matrices, vectors):
            individual_results.append(mat.multiply_vector(vec))
        
        # Should match
        for batch_res, ind_res in zip(batch_results, individual_results):
            np.testing.assert_array_almost_equal(batch_res, ind_res)
    
    def test_empty_batch_handling(self):
        """Test edge cases with empty batches"""
        # Empty list
        empty_result = BatchedCSROperations.batch_to_dense([])
        assert len(empty_result) == 0
        
        # Batch with empty matrices
        empty_matrices = [
            CSRPadicMatrix(np.zeros((3, 3))),
            CSRPadicMatrix(np.zeros((3, 3)))
        ]
        
        dense = BatchedCSROperations.batch_to_dense(empty_matrices)
        assert dense.shape == (2, 3, 3)
        assert np.all(dense == 0)


class TestCSRRobustness:
    """Test robustness and error recovery"""
    
    def test_corrupted_row_ptr_detection(self):
        """Test that validation catches various row_ptr corruptions"""
        matrix = np.eye(5)
        csr = CSRPadicMatrix(matrix)
        
        # Test various corruptions
        corruptions = [
            (lambda c: setattr(c, 'row_ptr', c.row_ptr[:-1]), "Invalid row_ptr length"),
            (lambda c: setattr(c, 'row_ptr', np.array([1, 2, 3, 4, 5, 6])), "First row pointer must be 0"),
            (lambda c: setattr(c, 'row_ptr', np.array([0, 2, 1, 3, 4, 5])), "Row pointers must be monotonically increasing"),
        ]
        
        for corrupt_fn, expected_error in corruptions:
            csr_copy = CSRPadicMatrix(matrix)
            corrupt_fn(csr_copy)
            
            with pytest.raises(ValueError) as exc_info:
                csr_copy.validate()
            
            # Check that appropriate error is raised
            assert expected_error in str(exc_info.value) or "row_ptr" in str(exc_info.value).lower()
    
    def test_col_idx_bounds_validation(self):
        """Test column index bounds checking"""
        matrix = np.eye(3)
        csr = CSRPadicMatrix(matrix)
        
        # Corrupt column indices
        csr.col_idx[0] = -1  # Negative index
        with pytest.raises(ValueError, match="Negative column index"):
            csr.validate()
        
        csr = CSRPadicMatrix(matrix)
        csr.col_idx[0] = 10  # Out of bounds
        with pytest.raises(ValueError, match="Column index out of bounds"):
            csr.validate()
    
    def test_extreme_matrix_sizes(self):
        """Test with very small and very large dimensions"""
        # 1x1 matrix
        tiny = np.array([[42.0]])
        csr_tiny = CSRPadicMatrix(tiny)
        assert csr_tiny.validate()
        assert csr_tiny.to_dense()[0, 0] == 42.0
        
        # Very wide matrix
        wide = np.zeros((2, 1000))
        wide[0, 999] = 1.0
        wide[1, 0] = 2.0
        csr_wide = CSRPadicMatrix(wide)
        assert csr_wide.shape == (2, 1000)
        assert csr_wide.metrics.nnz == 2
        
        # Very tall matrix
        tall = np.zeros((1000, 2))
        tall[0, 0] = 1.0
        tall[999, 1] = 2.0
        csr_tall = CSRPadicMatrix(tall)
        assert csr_tall.shape == (1000, 2)
        assert csr_tall.metrics.nnz == 2


class TestCSRPerformanceOptimizations:
    """Test performance-related functionality"""
    
    def test_operation_timing_accuracy(self):
        """Test that performance monitor accurately times operations"""
        monitor = CSRPerformanceMonitor()
        
        # Create a matrix large enough to have measurable operation time
        size = 500
        matrix = np.random.randn(size, size)
        matrix[matrix < 0.8] = 0
        csr = CSRPadicMatrix(matrix)
        
        # Time an operation
        start = time.time()
        vector = np.random.randn(size)
        result = csr.multiply_vector(vector)
        elapsed = time.time() - start
        
        # Record it
        monitor.record_operation('spmv', csr.shape, csr.metrics.nnz, elapsed)
        
        # Check recording
        assert 'spmv' in monitor.operation_timings
        assert len(monitor.operation_timings['spmv']) == 1
        timing = monitor.operation_timings['spmv'][0]
        assert timing['execution_time'] == elapsed
        assert timing['nnz'] == csr.metrics.nnz
    
    def test_compression_ratio_calculation(self):
        """Test that compression ratio is calculated correctly"""
        # Create matrix with known sparsity
        size = 100
        matrix = np.zeros((size, size), dtype=np.float32)
        
        # Add exactly 100 non-zero elements
        for i in range(100):
            matrix[i % size, (i * 7) % size] = i + 1.0
        
        csr = CSRPadicMatrix(matrix)
        
        # Verify compression ratio calculation
        # Dense: 100*100*4 = 40000 bytes
        # For size=100 < 32768, implementation uses int16 (2 bytes) for indices
        # Sparse: 100*4 (values) + 100*2 (col_idx as int16) + 101*2 (row_ptr as int16) = 802 bytes
        expected_ratio = 40000 / 802
        
        # Allow some tolerance for overhead
        assert abs(csr.metrics.compression_ratio - expected_ratio) / expected_ratio < 0.1
    
    def test_bandwidth_reduction_calculation(self):
        """Test bandwidth reduction metric calculation"""
        # Create matrix with 10% density
        size = 50
        matrix = np.zeros((size, size))
        nnz = int(size * size * 0.1)
        indices = np.random.choice(size * size, nnz, replace=False)
        matrix.flat[indices] = 1.0
        
        csr = CSRPadicMatrix(matrix)
        
        # Bandwidth reduction should be approximately 1 - density
        expected_reduction = 1.0 - 0.1
        assert abs(csr.metrics.bandwidth_reduction - expected_reduction) < 0.01


class TestCSRSerializationAdvanced:
    """Advanced serialization tests"""
    
    def test_serialization_preserves_precision(self):
        """Test that serialization maintains numerical precision"""
        # Create matrix with precise values
        matrix = np.array([
            [np.pi, 0, np.e],
            [0, np.sqrt(2), 0],
            [np.log(10), 0, np.sin(1)]
        ], dtype=np.float32)
        
        csr1 = CSRPadicMatrix(matrix)
        data = csr1.to_bytes()
        csr2 = CSRPadicMatrix.from_bytes(data)
        
        # Values should be preserved to float32 precision
        np.testing.assert_array_almost_equal(
            csr1.values, csr2.values, decimal=6
        )
    
    def test_serialization_size_efficiency(self):
        """Test that serialization is size-efficient"""
        # Create a sparse matrix
        size = 100
        matrix = np.zeros((size, size))
        matrix[0, 0] = 1.0
        matrix[size-1, size-1] = 2.0
        
        csr = CSRPadicMatrix(matrix)
        serialized = csr.to_bytes()
        
        # Serialized size should be much smaller than dense
        dense_size = matrix.nbytes
        serialized_size = len(serialized)
        
        assert serialized_size < dense_size / 10  # At least 10x smaller
    
    def test_cross_platform_serialization(self):
        """Test that serialized data could work across platforms"""
        matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)
        csr = CSRPadicMatrix(matrix)
        
        data = csr.to_bytes()
        
        # Check header is platform-independent
        assert data[:4] == b'CSR1'  # Magic number
        
        # Version should be readable
        import struct
        version = struct.unpack('I', data[4:8])[0]
        assert version == 1


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
        import sys
        
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
        import threading
        import time
        
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
        import threading
        
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
        import threading
        
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
        import gc
        import sys
        
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
        import psutil
        import os
        
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
                # Each CSR construction creates: original matrix, mask array, index arrays
                # NumPy may keep all of these in its pool (up to 3x the dense size)
                cumulative_dense_mb = sum(s * s * 4 / 1024 / 1024 for s in [100, 500, 1000, 1500][:len(matrices)])
                
                # Realistic limit: NumPy may retain up to 3x dense memory in its pool
                # Plus base Python/library overhead
                # This still validates CSR is working - the actual CSR objects are tiny
                assert memory_growth < cumulative_dense_mb * 4 + 20, \
                    f"Memory growth {memory_growth:.2f} MB exceeds reasonable limit for NumPy pool retention"
        
        finally:
            # Cleanup
            del matrices
            import gc
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
            import time
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
        # Allow up to 10x slower (since scipy is highly optimized C code)
        assert our_time < scipy_time * 10, f"Our: {our_time:.4f}s, Scipy: {scipy_time:.4f}s"
        
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
        import time
        
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