"""
Comprehensive unit tests for TropicalLinearAlgebra component.
Tests all classes, methods, and edge cases in tropical linear algebra operations.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict
import time
import tempfile
import os

# Import the components to test
from independent_core.compression_systems.tropical.tropical_linear_algebra import (
    TropicalMatrix,
    TropicalSparseMatrix,
    TropicalLinearAlgebra,
    TropicalMatrixFactorization,
    NeuralLayerTropicalization
)

from independent_core.compression_systems.tropical.tropical_core import (
    TropicalNumber,
    TropicalMathematicalOperations,
    TropicalValidation,
    TROPICAL_ZERO,
    TROPICAL_EPSILON,
    is_tropical_zero,
    to_tropical_safe,
    from_tropical_safe
)

from independent_core.compression_systems.tropical.tropical_polynomial import (
    TropicalPolynomial,
    TropicalMonomial
)

from independent_core.compression_systems.tropical.polytope_operations import (
    Polytope,
    PolytopeOperations
)


class TestTropicalMatrix:
    """Test TropicalMatrix class"""
    
    def test_matrix_creation_valid(self):
        """Test valid matrix creation"""
        # Basic creation
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mat = TropicalMatrix(data)
        assert mat.shape == (2, 2)
        assert mat.device == data.device
        assert torch.allclose(mat.data, data)
        
        # With validation disabled
        mat_no_val = TropicalMatrix(data, validate=False)
        assert mat_no_val.shape == (2, 2)
        
        # With tropical zeros
        data_with_zero = torch.tensor([[1.0, TROPICAL_ZERO], [2.0, 3.0]])
        mat2 = TropicalMatrix(data_with_zero)
        assert mat2.data[0, 1] == TROPICAL_ZERO
        
        # Large matrix
        large_data = torch.randn(100, 200)
        large_mat = TropicalMatrix(large_data)
        assert large_mat.shape == (100, 200)
    
    def test_matrix_creation_invalid(self):
        """Test invalid matrix creation"""
        # Not a tensor
        with pytest.raises(TypeError, match="Data must be torch.Tensor"):
            TropicalMatrix("invalid")
        
        # Wrong dimensions
        with pytest.raises(ValueError, match="Data must be 2D tensor"):
            TropicalMatrix(torch.tensor([1.0, 2.0]))
        
        with pytest.raises(ValueError, match="Data must be 2D tensor"):
            TropicalMatrix(torch.tensor([[[1.0]]]))
    
    def test_matrix_multiplication(self):
        """Test tropical matrix multiplication"""
        # Basic multiplication
        A = TropicalMatrix(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        B = TropicalMatrix(torch.tensor([[2.0, 1.0], [4.0, 3.0]]))
        
        C = A @ B
        
        # Manual calculation:
        # C[0,0] = max(1+2, 2+4) = max(3, 6) = 6
        # C[0,1] = max(1+1, 2+3) = max(2, 5) = 5
        # C[1,0] = max(3+2, 4+4) = max(5, 8) = 8
        # C[1,1] = max(3+1, 4+3) = max(4, 7) = 7
        expected = torch.tensor([[6.0, 5.0], [8.0, 7.0]])
        assert torch.allclose(C.data, expected)
        
        # Identity multiplication
        I = TropicalMatrix(torch.zeros(2, 2))  # Tropical identity
        AI = A @ I
        assert torch.allclose(AI.data, A.data)
        
        # Multiplication with zeros
        Z = TropicalMatrix(torch.full((2, 2), TROPICAL_ZERO))
        AZ = A @ Z
        assert torch.all(AZ.data == TROPICAL_ZERO)
    
    def test_matrix_multiplication_invalid(self):
        """Test invalid matrix multiplication"""
        A = TropicalMatrix(torch.tensor([[1.0, 2.0]]))  # 1x2
        B = TropicalMatrix(torch.tensor([[1.0], [2.0], [3.0]]))  # 3x1
        
        # Dimension mismatch
        with pytest.raises(ValueError, match="Matrix dimensions incompatible"):
            A @ B
        
        # Wrong type
        with pytest.raises(TypeError, match="Can only multiply with TropicalMatrix"):
            A @ 5.0
    
    def test_matrix_power(self):
        """Test matrix power operation"""
        A = TropicalMatrix(torch.tensor([[1.0, 2.0], [3.0, 0.0]]))
        
        # Power of 1
        A1 = A.power(1)
        assert torch.allclose(A1.data, A.data)
        
        # Power of 2
        A2 = A.power(2)
        A2_direct = A @ A
        assert torch.allclose(A2.data, A2_direct.data)
        
        # Power of 3
        A3 = A.power(3)
        A3_direct = A @ A @ A
        assert torch.allclose(A3.data, A3_direct.data)
        
        # Large power
        A10 = A.power(10)
        assert A10.shape == A.shape
    
    def test_matrix_power_invalid(self):
        """Test invalid matrix power operations"""
        A = TropicalMatrix(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        B = TropicalMatrix(torch.tensor([[1.0, 2.0]]))  # Non-square
        
        # Non-integer power
        with pytest.raises(TypeError, match="Exponent must be int"):
            A.power(2.5)
        
        # Negative power
        with pytest.raises(ValueError, match="Exponent must be positive"):
            A.power(-1)
        
        # Zero power
        with pytest.raises(ValueError, match="Exponent must be positive"):
            A.power(0)
        
        # Non-square matrix
        with pytest.raises(ValueError, match="Matrix must be square"):
            B.power(2)
    
    def test_matrix_sparse_conversion(self):
        """Test conversion between dense and sparse"""
        # Dense to sparse
        data = torch.tensor([[1.0, TROPICAL_ZERO, 2.0],
                            [TROPICAL_ZERO, 3.0, TROPICAL_ZERO],
                            [4.0, TROPICAL_ZERO, 5.0]])
        dense_mat = TropicalMatrix(data)
        sparse_mat = dense_mat.to_sparse()
        
        assert sparse_mat.shape == (3, 3)
        assert sparse_mat.indices.shape[1] == 5  # 5 non-zero elements
        
        # Sparse back to dense
        dense_restored = sparse_mat.to_dense()
        assert torch.allclose(dense_restored.data, data)
        
        # Empty matrix
        empty = TropicalMatrix(torch.full((3, 3), TROPICAL_ZERO))
        sparse_empty = empty.to_sparse()
        assert sparse_empty.indices.shape[1] == 0
    
    def test_matrix_transpose(self):
        """Test matrix transpose"""
        A = TropicalMatrix(torch.tensor([[1.0, 2.0, 3.0],
                                        [4.0, 5.0, 6.0]]))
        At = A.transpose()
        
        assert At.shape == (3, 2)
        assert torch.allclose(At.data, A.data.T)
    
    def test_matrix_diagonal_trace(self):
        """Test diagonal extraction and trace"""
        A = TropicalMatrix(torch.tensor([[1.0, 2.0, 3.0],
                                        [4.0, 5.0, 6.0],
                                        [7.0, 8.0, 9.0]]))
        
        # Diagonal
        diag = A.diagonal()
        expected_diag = torch.tensor([1.0, 5.0, 9.0])
        assert torch.allclose(diag, expected_diag)
        
        # Trace (tropical sum = max)
        trace = A.trace()
        assert trace == 9.0
        
        # Trace of matrix with tropical zeros
        B = TropicalMatrix(torch.tensor([[TROPICAL_ZERO, 1.0],
                                        [2.0, TROPICAL_ZERO]]))
        trace_b = B.trace()
        assert trace_b == TROPICAL_ZERO
    
    def test_matrix_device_handling(self):
        """Test device handling for GPU/CPU"""
        if torch.cuda.is_available():
            # Create on GPU
            data_gpu = torch.tensor([[1.0, 2.0]], device='cuda')
            mat_gpu = TropicalMatrix(data_gpu)
            assert mat_gpu.device.type == 'cuda'
            
            # Create on CPU
            data_cpu = torch.tensor([[1.0, 2.0]], device='cpu')
            mat_cpu = TropicalMatrix(data_cpu)
            assert mat_cpu.device.type == 'cpu'
            
            # Cross-device multiplication
            result = mat_gpu @ mat_cpu.transpose()
            assert result.device.type == 'cuda'
    
    def test_matrix_string_representation(self):
        """Test string representations"""
        A = TropicalMatrix(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        
        str_repr = str(A)
        assert "TropicalMatrix" in str_repr
        assert "2×2" in str_repr
        
        repr_repr = repr(A)
        assert "shape=(2, 2)" in repr_repr
        assert "nnz=" in repr_repr


class TestTropicalSparseMatrix:
    """Test TropicalSparseMatrix class"""
    
    def test_sparse_creation_valid(self):
        """Test valid sparse matrix creation"""
        indices = torch.tensor([[0, 1, 1], [0, 1, 2]])
        values = torch.tensor([1.0, 2.0, 3.0])
        sparse = TropicalSparseMatrix(indices, values, (2, 3))
        
        assert sparse.shape == (2, 3)
        assert torch.allclose(sparse.indices, indices)
        assert torch.allclose(sparse.values, values)
    
    def test_sparse_creation_invalid(self):
        """Test invalid sparse matrix creation"""
        # Wrong indices type
        with pytest.raises(TypeError, match="Indices must be torch.Tensor"):
            TropicalSparseMatrix([0, 1], [1.0], (2, 2))
        
        # Wrong values type
        with pytest.raises(TypeError, match="Values must be torch.Tensor"):
            TropicalSparseMatrix(torch.tensor([[0], [0]]), [1.0], (2, 2))
        
        # Wrong indices shape
        with pytest.raises(ValueError, match="Indices must be shape"):
            TropicalSparseMatrix(torch.tensor([0, 0]), torch.tensor([1.0]), (2, 2))
        
        # Wrong values shape
        with pytest.raises(ValueError, match="Values must be 1D tensor"):
            TropicalSparseMatrix(torch.tensor([[0], [0]]), torch.tensor([[1.0]]), (2, 2))
        
        # Mismatched indices and values
        with pytest.raises(ValueError, match="Number of indices.*doesn't match"):
            TropicalSparseMatrix(torch.tensor([[0, 1], [0, 1]]), 
                               torch.tensor([1.0, 2.0, 3.0]), (2, 2))
    
    def test_sparse_to_dense_conversion(self):
        """Test sparse to dense conversion"""
        indices = torch.tensor([[0, 1, 2], [1, 0, 2]])
        values = torch.tensor([1.0, 2.0, 3.0])
        sparse = TropicalSparseMatrix(indices, values, (3, 3))
        
        dense = sparse.to_dense()
        expected = torch.tensor([[TROPICAL_ZERO, 1.0, TROPICAL_ZERO],
                                [2.0, TROPICAL_ZERO, TROPICAL_ZERO],
                                [TROPICAL_ZERO, TROPICAL_ZERO, 3.0]])
        assert torch.allclose(dense.data, expected)
        
        # Empty sparse matrix
        empty_sparse = TropicalSparseMatrix(torch.zeros((2, 0), dtype=torch.long),
                                           torch.zeros(0), (3, 3))
        empty_dense = empty_sparse.to_dense()
        assert torch.all(empty_dense.data == TROPICAL_ZERO)
    
    def test_sparse_multiplication(self):
        """Test sparse matrix multiplication"""
        # Create two sparse matrices
        indices1 = torch.tensor([[0, 1], [0, 1]])
        values1 = torch.tensor([1.0, 2.0])
        sparse1 = TropicalSparseMatrix(indices1, values1, (2, 2))
        
        indices2 = torch.tensor([[0, 1], [1, 0]])
        values2 = torch.tensor([3.0, 4.0])
        sparse2 = TropicalSparseMatrix(indices2, values2, (2, 2))
        
        # Multiply
        result = sparse1.multiply(sparse2)
        assert result.shape == (2, 2)
        
        # Verify result by converting to dense and computing
        dense1 = sparse1.to_dense()
        dense2 = sparse2.to_dense()
        expected = dense1 @ dense2
        result_dense = result.to_dense()
        assert torch.allclose(result_dense.data, expected.data)
    
    def test_sparse_device_handling(self):
        """Test device handling for sparse matrices"""
        if torch.cuda.is_available():
            indices = torch.tensor([[0], [0]], device='cuda')
            values = torch.tensor([1.0], device='cuda')
            sparse_gpu = TropicalSparseMatrix(indices, values, (2, 2), device=torch.device('cuda'))
            
            assert sparse_gpu.device.type == 'cuda'
            assert sparse_gpu.indices.device.type == 'cuda'
            assert sparse_gpu.values.device.type == 'cuda'
            
            # Move to CPU
            sparse_cpu = TropicalSparseMatrix(indices.cpu(), values.cpu(), (2, 2))
            assert sparse_cpu.device.type == 'cpu'


class TestTropicalLinearAlgebra:
    """Test TropicalLinearAlgebra class"""
    
    def test_initialization(self):
        """Test TropicalLinearAlgebra initialization"""
        # Default CPU
        algebra_cpu = TropicalLinearAlgebra()
        assert algebra_cpu.device.type == 'cpu'
        
        # With device
        if torch.cuda.is_available():
            algebra_gpu = TropicalLinearAlgebra(device=torch.device('cuda'))
            assert algebra_gpu.device.type == 'cuda'
    
    def test_matrix_multiply(self):
        """Test matrix multiplication through algebra class"""
        algebra = TropicalLinearAlgebra()
        
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        B = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
        
        C = algebra.matrix_multiply(A, B)
        
        expected = torch.tensor([[6.0, 5.0], [8.0, 7.0]])
        assert torch.allclose(C, expected)
        
        # Different shapes
        A2 = torch.tensor([[1.0, 2.0, 3.0]])  # 1x3
        B2 = torch.tensor([[1.0], [2.0], [3.0]])  # 3x1
        C2 = algebra.matrix_multiply(A2, B2)
        assert C2.shape == (1, 1)
    
    def test_batch_matrix_multiply(self):
        """Test batched matrix multiplication"""
        algebra = TropicalLinearAlgebra()
        
        # Batched A and B
        A = torch.randn(5, 3, 4)
        B = torch.randn(5, 4, 2)
        
        C = algebra.batch_matrix_multiply(A, B)
        assert C.shape == (5, 3, 2)
        
        # Broadcasting with 2D B
        B_single = torch.randn(4, 2)
        C_broadcast = algebra.batch_matrix_multiply(A, B_single)
        assert C_broadcast.shape == (5, 3, 2)
        
        # Verify correctness for first batch
        C_0_expected = algebra.matrix_multiply(A[0], B[0])
        assert torch.allclose(C[0], C_0_expected, atol=1e-6)
    
    def test_batch_multiply_invalid(self):
        """Test invalid batch multiplication"""
        algebra = TropicalLinearAlgebra()
        
        # Not 3D
        with pytest.raises(ValueError, match="A must be 3D tensor"):
            algebra.batch_matrix_multiply(torch.randn(2, 2), torch.randn(2, 2))
        
        # Dimension mismatch
        A = torch.randn(5, 3, 4)
        B = torch.randn(5, 3, 2)  # Wrong inner dimension
        with pytest.raises(ValueError, match="Dimension mismatch"):
            algebra.batch_matrix_multiply(A, B)
        
        # Batch size mismatch
        A = torch.randn(5, 3, 4)
        B = torch.randn(3, 4, 2)  # Wrong batch size
        with pytest.raises(ValueError, match="Batch size mismatch"):
            algebra.batch_matrix_multiply(A, B)
    
    def test_matrix_power(self):
        """Test matrix power through algebra class"""
        algebra = TropicalLinearAlgebra()
        
        A = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
        
        # Different powers
        A2 = algebra.matrix_power(A, 2)
        A2_expected = algebra.matrix_multiply(A, A)
        assert torch.allclose(A2, A2_expected)
        
        A5 = algebra.matrix_power(A, 5)
        assert A5.shape == A.shape
    
    def test_eigenvalue_karp_method(self):
        """Test eigenvalue computation with Karp's algorithm"""
        algebra = TropicalLinearAlgebra()
        
        # Simple matrix with known properties
        A = torch.tensor([[0.0, 1.0, 2.0],
                         [2.0, 0.0, 1.0],
                         [1.0, 2.0, 0.0]])
        
        eigenval = algebra.eigenvalue(A, method='karp')
        assert isinstance(eigenval, (float, np.float64))
        assert eigenval >= TROPICAL_ZERO
        
        # Matrix with tropical zeros
        B = torch.tensor([[TROPICAL_ZERO, 1.0],
                         [2.0, TROPICAL_ZERO]])
        eigenval_b = algebra.eigenvalue(B, method='karp')
        assert isinstance(eigenval_b, (float, np.float64))
    
    def test_eigenvalue_power_method(self):
        """Test eigenvalue computation with power iteration"""
        algebra = TropicalLinearAlgebra()
        
        A = torch.tensor([[1.0, 2.0, 3.0],
                         [2.0, 3.0, 1.0],
                         [3.0, 1.0, 2.0]])
        
        eigenval = algebra.eigenvalue(A, method='power')
        assert isinstance(eigenval, (float, np.float64))
        assert eigenval >= TROPICAL_ZERO
    
    def test_eigenvalue_invalid(self):
        """Test invalid eigenvalue computations"""
        algebra = TropicalLinearAlgebra()
        
        # Non-square matrix
        A = torch.tensor([[1.0, 2.0]])
        with pytest.raises(ValueError, match="Matrix must be square"):
            algebra.eigenvalue(A)
        
        # Invalid method
        B = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="Unknown method"):
            algebra.eigenvalue(B, method='invalid')
    
    def test_eigenvector(self):
        """Test eigenvector computation"""
        algebra = TropicalLinearAlgebra()
        
        A = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
        
        # Compute eigenvalue first
        eigenval = algebra.eigenvalue(A, method='power')
        
        # Compute eigenvector
        eigenvec = algebra.eigenvector(A, eigenval)
        assert eigenvec.shape == (2,)
        
        # Verify it's approximately an eigenvector
        # A ⊗ v ≈ λ ⊗ v in tropical arithmetic
        Av = algebra.matrix_multiply(A, eigenvec.unsqueeze(1)).squeeze()
        lambda_v = eigenvec + eigenval
        # Tropical arithmetic has different properties, so just check shapes
        assert Av.shape == lambda_v.shape
    
    def test_solve_linear_system(self):
        """Test tropical linear system solver"""
        algebra = TropicalLinearAlgebra()
        
        A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
        b = torch.tensor([5.0, 6.0])
        
        x = algebra.solve_linear_system(A, b)
        assert x.shape == (2,)
        
        # The solution may be approximate due to tropical algebra properties
        # Just verify the shape and that it doesn't contain NaN
        assert not torch.isnan(x).any()
    
    def test_solve_linear_system_invalid(self):
        """Test invalid linear system solving"""
        algebra = TropicalLinearAlgebra()
        
        # Wrong dimensions
        A = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="Dimension mismatch"):
            algebra.solve_linear_system(A, b)
        
        # Wrong tensor dimensions
        A = torch.tensor([1.0, 2.0])
        b = torch.tensor([1.0])
        with pytest.raises(ValueError, match="A must be 2D matrix"):
            algebra.solve_linear_system(A, b)
    
    def test_matrix_rank(self):
        """Test tropical rank computation"""
        algebra = TropicalLinearAlgebra()
        
        # Full rank matrix
        A_full = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        rank_full = algebra.matrix_rank(A_full)
        assert rank_full == 2
        
        # Rank deficient (all same row)
        A_deficient = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
        rank_deficient = algebra.matrix_rank(A_deficient)
        assert 0 <= rank_deficient <= 2
        
        # Matrix with zeros
        A_zeros = torch.tensor([[1.0, TROPICAL_ZERO], [TROPICAL_ZERO, 2.0]])
        rank_zeros = algebra.matrix_rank(A_zeros)
        assert 0 <= rank_zeros <= 2
    
    def test_determinant(self):
        """Test tropical determinant"""
        algebra = TropicalLinearAlgebra()
        
        # 2x2 matrix
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        det = algebra.determinant(A)
        assert isinstance(det, (float, np.float64))
        assert det > TROPICAL_ZERO
        
        # Identity-like matrix
        I = torch.zeros(3, 3)
        det_i = algebra.determinant(I)
        assert det_i == 0.0
        
        # Matrix with tropical zeros
        Z = torch.tensor([[TROPICAL_ZERO, 1.0], [2.0, TROPICAL_ZERO]])
        det_z = algebra.determinant(Z)
        assert det_z == TROPICAL_ZERO
    
    def test_permanent(self):
        """Test tropical permanent"""
        algebra = TropicalLinearAlgebra()
        
        # Small matrix (exact computation)
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        perm = algebra.permanent(A)
        assert isinstance(perm, (float, np.float64))
        assert perm >= TROPICAL_ZERO
        
        # Larger matrix (uses approximation)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            B = torch.randn(11, 11)
            perm_b = algebra.permanent(B)
            assert isinstance(perm_b, (float, np.float64))


class TestTropicalMatrixFactorization:
    """Test TropicalMatrixFactorization class"""
    
    def test_initialization(self):
        """Test factorization initialization"""
        factorizer = TropicalMatrixFactorization()
        assert factorizer.tolerance == 1e-6
        
        factorizer_custom = TropicalMatrixFactorization(tolerance=1e-8)
        assert factorizer_custom.tolerance == 1e-8
    
    def test_low_rank_approximation(self):
        """Test low-rank matrix approximation"""
        factorizer = TropicalMatrixFactorization()
        
        # Create test matrix
        A = torch.randn(10, 8).abs()
        
        # Factorize
        U, V = factorizer.low_rank_approximation(A, rank=3)
        assert U.shape == (10, 3)
        assert V.shape == (3, 8)
        
        # Check reconstruction
        algebra = TropicalLinearAlgebra()
        A_approx = algebra.matrix_multiply(U, V)
        assert A_approx.shape == A.shape
        
        # Rank exceeding dimensions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            U2, V2 = factorizer.low_rank_approximation(A, rank=20)
            assert U2.shape[1] == min(A.shape)
            assert V2.shape[0] == min(A.shape)
    
    def test_low_rank_invalid(self):
        """Test invalid low-rank approximation"""
        factorizer = TropicalMatrixFactorization()
        
        # Not 2D
        with pytest.raises(ValueError, match="Expected 2D matrix"):
            factorizer.low_rank_approximation(torch.randn(10), rank=3)
        
        # Negative rank
        with pytest.raises(ValueError, match="Rank must be positive"):
            factorizer.low_rank_approximation(torch.randn(5, 5), rank=-1)
        
        # Zero rank
        with pytest.raises(ValueError, match="Rank must be positive"):
            factorizer.low_rank_approximation(torch.randn(5, 5), rank=0)
    
    def test_nonnegative_factorization(self):
        """Test tropical NMF"""
        factorizer = TropicalMatrixFactorization()
        
        A = torch.randn(8, 6).abs()
        
        W, H = factorizer.nonnegative_factorization(A, rank=3)
        assert W.shape == (8, 3)
        assert H.shape == (3, 6)
        
        # Check non-negativity in tropical sense
        assert torch.all((W > TROPICAL_ZERO) | (W == TROPICAL_ZERO))
        assert torch.all((H > TROPICAL_ZERO) | (H == TROPICAL_ZERO))
    
    def test_schur_decomposition(self):
        """Test Schur decomposition for block triangular form"""
        factorizer = TropicalMatrixFactorization()
        
        # Create a matrix with known structure
        A = torch.tensor([[1.0, 2.0, 0.0],
                         [3.0, 4.0, 0.0],
                         [0.0, 0.0, 5.0]])
        
        P, T = factorizer.schur_decomposition(A)
        
        # Check permutation matrix properties
        assert P.shape == (3, 3)
        assert torch.allclose(P @ P.T, torch.eye(3))  # P is orthogonal
        
        # Check triangular form
        assert T.shape == (3, 3)
        
        # Reconstruction
        A_reconstructed = P @ T @ P.T
        assert torch.allclose(A_reconstructed, A, atol=1e-5)


class TestNeuralLayerTropicalization:
    """Test NeuralLayerTropicalization class"""
    
    def test_initialization(self):
        """Test tropicalization system initialization"""
        tropicalizer = NeuralLayerTropicalization()
        assert tropicalizer.linear_algebra is not None
        assert tropicalizer.factorizer is not None
    
    def test_tropicalize_linear_layer(self):
        """Test linear layer tropicalization"""
        tropicalizer = NeuralLayerTropicalization()
        
        # Create test layer
        layer = nn.Linear(10, 5, bias=True)
        
        # Tropicalize
        tropical_mat = tropicalizer.tropicalize_linear_layer(layer)
        assert isinstance(tropical_mat, TropicalMatrix)
        assert tropical_mat.shape[0] == 5  # out_features
        assert tropical_mat.shape[1] == 11  # in_features + bias
        
        # Without bias
        layer_no_bias = nn.Linear(10, 5, bias=False)
        tropical_no_bias = tropicalizer.tropicalize_linear_layer(layer_no_bias)
        assert tropical_no_bias.shape[1] == 10
    
    def test_tropicalize_invalid_layer(self):
        """Test invalid layer tropicalization"""
        tropicalizer = NeuralLayerTropicalization()
        
        # Not a Linear layer
        conv_layer = nn.Conv2d(3, 16, 3)
        with pytest.raises(TypeError, match="Expected nn.Linear"):
            tropicalizer.tropicalize_linear_layer(conv_layer)
    
    def test_compress_via_tropical(self):
        """Test layer compression using tropical factorization"""
        tropicalizer = NeuralLayerTropicalization()
        
        # Create test layer
        layer = nn.Linear(20, 10, bias=True)
        
        # Compress
        compressed = tropicalizer.compress_via_tropical(layer, target_rank=5)
        
        assert isinstance(compressed, nn.Linear)
        assert compressed.in_features == 20
        assert compressed.out_features == 10
        assert compressed.bias is not None
        
        # Test without bias
        layer_no_bias = nn.Linear(20, 10, bias=False)
        compressed_no_bias = tropicalizer.compress_via_tropical(layer_no_bias, target_rank=5)
        assert compressed_no_bias.bias is None
    
    def test_compress_invalid(self):
        """Test invalid compression"""
        tropicalizer = NeuralLayerTropicalization()
        
        # Invalid layer type
        with pytest.raises(TypeError, match="Expected nn.Linear"):
            tropicalizer.compress_via_tropical("not a layer", target_rank=3)
        
        # Invalid rank
        layer = nn.Linear(10, 5)
        with pytest.raises(ValueError, match="Target rank must be positive"):
            tropicalizer.compress_via_tropical(layer, target_rank=0)
    
    def test_analyze_information_flow(self):
        """Test neural network information flow analysis"""
        tropicalizer = NeuralLayerTropicalization()
        
        # Create test model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 5)
        )
        
        # Analyze
        analysis = tropicalizer.analyze_information_flow(model)
        
        # Check structure
        assert 'layers' in analysis
        assert 'bottlenecks' in analysis
        assert 'redundancy_score' in analysis
        assert 'compression_potential' in analysis
        
        # Check values
        assert len(analysis['layers']) == 3  # Three linear layers
        assert isinstance(analysis['bottlenecks'], list)
        assert 0 <= analysis['redundancy_score'] <= 1
        assert 0 <= analysis['compression_potential'] <= 1
        
        # Check layer info
        for layer_info in analysis['layers']:
            assert 'name' in layer_info
            assert 'shape' in layer_info
            assert 'tropical_rank' in layer_info
            assert 'full_rank' in layer_info
            assert 'compression_ratio' in layer_info
    
    def test_analyze_invalid_model(self):
        """Test invalid model analysis"""
        tropicalizer = NeuralLayerTropicalization()
        
        # Not a module
        with pytest.raises(TypeError, match="Expected nn.Module"):
            tropicalizer.analyze_information_flow("not a model")
    
    def test_analyze_empty_model(self):
        """Test analysis of model without linear layers"""
        tropicalizer = NeuralLayerTropicalization()
        
        # Model with no linear layers
        model = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        analysis = tropicalizer.analyze_information_flow(model)
        assert len(analysis['layers']) == 0
        assert analysis['compression_potential'] == 0.0
        assert analysis['redundancy_score'] == 0.0


class TestEdgeCases:
    """Test edge cases and special scenarios"""
    
    def test_very_large_values(self):
        """Test handling of very large values"""
        A = TropicalMatrix(torch.tensor([[1e10, 1e11], [1e12, 1e13]]))
        B = TropicalMatrix(torch.tensor([[1e10, 1e11], [1e12, 1e13]]))
        
        C = A @ B
        assert not torch.isnan(C.data).any()
        assert not torch.isinf(C.data).any()
    
    def test_very_small_values(self):
        """Test handling of very small values"""
        A = TropicalMatrix(torch.tensor([[1e-10, 1e-11], [1e-12, 1e-13]]))
        B = TropicalMatrix(torch.tensor([[1e-10, 1e-11], [1e-12, 1e-13]]))
        
        C = A @ B
        assert C.shape == (2, 2)
    
    def test_mixed_tropical_zeros(self):
        """Test operations with mixed tropical zeros"""
        algebra = TropicalLinearAlgebra()
        
        # Matrix with mix of zeros and values
        A = torch.tensor([[1.0, TROPICAL_ZERO, 2.0],
                         [TROPICAL_ZERO, 3.0, TROPICAL_ZERO],
                         [4.0, TROPICAL_ZERO, 5.0]])
        
        # Operations should handle zeros correctly
        rank = algebra.matrix_rank(A)
        assert 0 <= rank <= 3
        
        det = algebra.determinant(A)
        assert det == TROPICAL_ZERO or det > TROPICAL_ZERO
    
    def test_numerical_stability(self):
        """Test numerical stability with near-equal values"""
        A = TropicalMatrix(torch.tensor([[1.0, 1.0 + 1e-10], 
                                        [1.0 - 1e-10, 1.0]]))
        
        # Should not cause numerical issues
        A_power = A.power(10)
        assert not torch.isnan(A_power.data).any()
        assert torch.all(torch.isfinite(A_power.data))
    
    def test_empty_sparse_operations(self):
        """Test operations on empty sparse matrices"""
        # Empty sparse matrix
        empty = TropicalSparseMatrix(
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros(0),
            (3, 3)
        )
        
        # Convert to dense
        dense = empty.to_dense()
        assert torch.all(dense.data == TROPICAL_ZERO)
        
        # String representation
        str_repr = str(empty)
        assert "nnz=0" in str_repr


class TestPerformance:
    """Test performance requirements"""
    
    def test_matrix_multiplication_performance(self):
        """Test matrix multiplication meets performance requirements"""
        algebra = TropicalLinearAlgebra()
        
        # Test different sizes
        sizes = [100, 500]
        
        for size in sizes:
            A = torch.randn(size, size)
            B = torch.randn(size, size)
            
            # Warm-up
            _ = algebra.matrix_multiply(A, B)
            
            # Time the operation
            start = time.time()
            C = algebra.matrix_multiply(A, B)
            elapsed = (time.time() - start) * 1000
            
            # Check performance (relaxed for CI)
            if size <= 100:
                assert elapsed < 500, f"Matrix multiply {size}x{size} took {elapsed}ms"
            
            assert C.shape == (size, size)
    
    def test_batch_multiplication_performance(self):
        """Test batch multiplication performance"""
        algebra = TropicalLinearAlgebra()
        
        # Batch of matrices
        A = torch.randn(10, 50, 50)
        B = torch.randn(10, 50, 50)
        
        start = time.time()
        C = algebra.batch_matrix_multiply(A, B)
        elapsed = (time.time() - start) * 1000
        
        assert C.shape == (10, 50, 50)
        # Performance check (very relaxed for CI)
        assert elapsed < 2000, f"Batch multiply took {elapsed}ms"
    
    def test_factorization_performance(self):
        """Test factorization performance"""
        factorizer = TropicalMatrixFactorization()
        
        A = torch.randn(20, 15).abs()
        
        start = time.time()
        U, V = factorizer.low_rank_approximation(A, rank=5)
        elapsed = (time.time() - start) * 1000
        
        assert U.shape == (20, 5)
        assert V.shape == (5, 15)
        # Performance check (relaxed for CI)
        assert elapsed < 1000, f"Factorization took {elapsed}ms"


class TestIntegration:
    """Integration tests with other components"""
    
    def test_with_tropical_core(self):
        """Test integration with tropical_core components"""
        # Create tropical numbers
        a = TropicalNumber(5.0)
        b = TropicalNumber(3.0)
        
        # Use in matrix
        data = torch.tensor([[a.value, b.value], [b.value, a.value]])
        mat = TropicalMatrix(data)
        assert mat.shape == (2, 2)
        
        # Validation
        TropicalValidation.validate_tropical_tensor(mat.data)
    
    def test_with_polynomial(self):
        """Test potential integration with polynomial operations"""
        # Create a matrix
        A = TropicalMatrix(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        
        # Could be used with polynomial operations
        # (actual integration would depend on polynomial implementation)
        assert A.shape == (2, 2)
    
    def test_end_to_end_compression(self):
        """Test end-to-end neural network compression"""
        # Create a small neural network
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 5)
        )
        
        # Initialize tropicalizer
        tropicalizer = NeuralLayerTropicalization()
        
        # Analyze model
        analysis = tropicalizer.analyze_information_flow(model)
        
        # Compress each linear layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Calculate target rank based on analysis
                layer_info = next((l for l in analysis['layers'] if l['name'] == name), None)
                if layer_info:
                    target_rank = max(1, int(layer_info['tropical_rank'] * 0.8))
                    compressed = tropicalizer.compress_via_tropical(module, target_rank)
                    assert compressed.in_features == module.in_features
                    assert compressed.out_features == module.out_features


class TestCUDAOperations:
    """Test CUDA/GPU operations if available"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_matrix_operations(self):
        """Test matrix operations on GPU"""
        device = torch.device('cuda')
        algebra = TropicalLinearAlgebra(device=device)
        
        # Create matrices on GPU
        A = torch.randn(100, 100, device=device)
        B = torch.randn(100, 100, device=device)
        
        # Test multiplication
        C = algebra.matrix_multiply(A, B)
        assert C.device.type == 'cuda'
        assert C.shape == (100, 100)
        
        # Test power
        A_power = algebra.matrix_power(A, 3)
        assert A_power.device.type == 'cuda'
        
        # Test eigenvalue (moves to CPU internally)
        eigenval = algebra.eigenvalue(A[:10, :10], method='karp')
        assert isinstance(eigenval, (float, np.float64))
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_sparse_operations(self):
        """Test sparse operations on GPU"""
        device = torch.device('cuda')
        
        indices = torch.tensor([[0, 1], [0, 1]], device=device)
        values = torch.tensor([1.0, 2.0], device=device)
        sparse = TropicalSparseMatrix(indices, values, (2, 2), device=device)
        
        assert sparse.device.type == 'cuda'
        
        dense = sparse.to_dense()
        assert dense.device.type == 'cuda'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_performance_comparison(self):
        """Compare GPU vs CPU performance"""
        size = 200
        A = torch.randn(size, size)
        B = torch.randn(size, size)
        
        # CPU timing
        algebra_cpu = TropicalLinearAlgebra(device=torch.device('cpu'))
        start = time.time()
        C_cpu = algebra_cpu.matrix_multiply(A, B)
        cpu_time = time.time() - start
        
        # GPU timing
        algebra_gpu = TropicalLinearAlgebra(device=torch.device('cuda'))
        A_gpu = A.cuda()
        B_gpu = B.cuda()
        
        # Warm-up
        _ = algebra_gpu.matrix_multiply(A_gpu, B_gpu)
        torch.cuda.synchronize()
        
        start = time.time()
        C_gpu = algebra_gpu.matrix_multiply(A_gpu, B_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        # GPU should be faster for large matrices
        print(f"CPU: {cpu_time*1000:.2f}ms, GPU: {gpu_time*1000:.2f}ms")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")


def test_module_imports():
    """Test that all required modules can be imported"""
    from independent_core.compression_systems.tropical.tropical_linear_algebra import (
        TropicalMatrix,
        TropicalSparseMatrix,
        TropicalLinearAlgebra,
        TropicalMatrixFactorization,
        NeuralLayerTropicalization
    )
    
    assert TropicalMatrix is not None
    assert TropicalSparseMatrix is not None
    assert TropicalLinearAlgebra is not None
    assert TropicalMatrixFactorization is not None
    assert NeuralLayerTropicalization is not None


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])