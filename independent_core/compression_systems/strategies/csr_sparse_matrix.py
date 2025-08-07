"""
CSR (Compressed Sparse Row) sparse matrix compression for p-adic weights.
Provides efficient storage and operations for sparse weight matrices with >70% zeros.
NO PLACEHOLDERS - COMPLETE PRODUCTION IMPLEMENTATION
"""

import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass
import logging
import struct

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CSRMetrics:
    """Metrics for CSR compression performance"""
    nnz: int  # Number of non-zero elements
    density: float  # Percentage of non-zero elements
    compression_ratio: float  # Ratio of dense to sparse size
    memory_saved_bytes: int  # Bytes saved by compression
    dense_memory_bytes: int  # Original dense memory requirement
    sparse_memory_bytes: int  # CSR memory requirement
    row_efficiency: float  # Average non-zeros per row
    bandwidth_reduction: float  # Reduction in memory bandwidth requirements


class CSRPadicMatrix:
    """
    Compressed Sparse Row format for p-adic weight matrices.
    
    Efficiently stores sparse matrices using three arrays:
    - values: Non-zero values
    - col_idx: Column indices of non-zero values
    - row_ptr: Pointers to start of each row in values/col_idx
    
    Memory complexity: O(nnz + m) where nnz = non-zeros, m = rows
    """
    
    def __init__(self, matrix: Union[torch.Tensor, np.ndarray], threshold: float = 1e-6):
        """
        Initialize CSR representation from dense matrix.
        
        Args:
            matrix: Dense torch tensor or numpy array
            threshold: Values with absolute value below this are considered zero
        """
        if not isinstance(matrix, (torch.Tensor, np.ndarray)):
            raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(matrix)}")
        
        # Convert to numpy for processing
        if isinstance(matrix, torch.Tensor):
            self.original_device = matrix.device
            self.original_dtype = matrix.dtype
            matrix_np = matrix.detach().cpu().numpy()
        else:
            self.original_device = None
            self.original_dtype = matrix.dtype
            matrix_np = matrix
        
        if matrix_np.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got shape {matrix_np.shape}")
        
        self.shape = matrix_np.shape
        self.threshold = threshold
        
        # Build CSR structure
        self.values = []
        self.col_idx = []
        self.row_ptr = [0]
        
        # Process each row
        for i in range(self.shape[0]):
            row = matrix_np[i]
            for j in range(self.shape[1]):
                if abs(row[j]) > threshold:
                    self.values.append(row[j])
                    self.col_idx.append(j)
            self.row_ptr.append(len(self.values))
        
        # Convert to numpy arrays for efficient operations
        self.values = np.array(self.values, dtype=np.float32)
        self.col_idx = np.array(self.col_idx, dtype=np.int32)
        self.row_ptr = np.array(self.row_ptr, dtype=np.int32)
        
        # Calculate compression metrics
        self.metrics = self._calculate_compression_metrics()
        
        # Log compression results
        logger.info(f"CSR compression: shape={self.shape}, nnz={self.metrics.nnz}, "
                   f"density={self.metrics.density:.2%}, ratio={self.metrics.compression_ratio:.2f}x, "
                   f"saved={self.metrics.memory_saved_bytes} bytes")
    
    def _calculate_compression_metrics(self) -> CSRMetrics:
        """Calculate detailed compression metrics"""
        nnz = len(self.values)
        total_elements = self.shape[0] * self.shape[1]
        density = nnz / total_elements if total_elements > 0 else 0.0
        
        # Calculate memory requirements
        # Dense: m * n * sizeof(float32)
        dense_size = total_elements * 4  # 4 bytes per float32
        
        # Sparse: nnz * (sizeof(float32) + sizeof(int32)) + (m+1) * sizeof(int32)
        # values: nnz * 4 bytes
        # col_idx: nnz * 4 bytes  
        # row_ptr: (m+1) * 4 bytes
        sparse_size = nnz * 8 + (self.shape[0] + 1) * 4
        
        compression_ratio = dense_size / sparse_size if sparse_size > 0 else 1.0
        memory_saved = max(0, dense_size - sparse_size)
        
        # Calculate row efficiency (average non-zeros per row)
        row_efficiency = nnz / self.shape[0] if self.shape[0] > 0 else 0.0
        
        # Calculate bandwidth reduction
        # For matrix-vector multiply: dense needs m*n reads, sparse needs nnz reads
        bandwidth_reduction = 1.0 - (nnz / total_elements) if total_elements > 0 else 0.0
        
        return CSRMetrics(
            nnz=nnz,
            density=density,
            compression_ratio=compression_ratio,
            memory_saved_bytes=memory_saved,
            dense_memory_bytes=dense_size,
            sparse_memory_bytes=sparse_size,
            row_efficiency=row_efficiency,
            bandwidth_reduction=bandwidth_reduction
        )
    
    def to_dense(self) -> np.ndarray:
        """
        Convert CSR back to dense representation.
        
        Returns:
            Dense numpy array reconstructed from CSR format
        """
        dense = np.zeros(self.shape, dtype=np.float32)
        
        for i in range(self.shape[0]):
            start = self.row_ptr[i]
            end = self.row_ptr[i + 1]
            
            for idx in range(start, end):
                j = self.col_idx[idx]
                dense[i, j] = self.values[idx]
        
        return dense
    
    def to_torch(self) -> torch.Tensor:
        """
        Convert CSR to dense torch tensor.
        
        Returns:
            Dense torch tensor on original device with original dtype
        """
        dense = self.to_dense()
        tensor = torch.from_numpy(dense)
        
        if self.original_device is not None:
            tensor = tensor.to(self.original_device)
        if self.original_dtype is not None:
            tensor = tensor.to(self.original_dtype)
        
        return tensor
    
    def get_row(self, i: int) -> np.ndarray:
        """
        Efficient row access without reconstructing entire matrix.
        
        Args:
            i: Row index
            
        Returns:
            Dense representation of row i
        """
        if i < 0 or i >= self.shape[0]:
            raise IndexError(f"Row index {i} out of bounds for matrix with {self.shape[0]} rows")
        
        row = np.zeros(self.shape[1], dtype=np.float32)
        start = self.row_ptr[i]
        end = self.row_ptr[i + 1]
        
        for idx in range(start, end):
            j = self.col_idx[idx]
            row[j] = self.values[idx]
        
        return row
    
    def get_column(self, j: int) -> np.ndarray:
        """
        Extract a column from CSR format (less efficient than row access).
        
        Args:
            j: Column index
            
        Returns:
            Dense representation of column j
        """
        if j < 0 or j >= self.shape[1]:
            raise IndexError(f"Column index {j} out of bounds for matrix with {self.shape[1]} columns")
        
        col = np.zeros(self.shape[0], dtype=np.float32)
        
        for i in range(self.shape[0]):
            start = self.row_ptr[i]
            end = self.row_ptr[i + 1]
            
            for idx in range(start, end):
                if self.col_idx[idx] == j:
                    col[i] = self.values[idx]
                    break
        
        return col
    
    def multiply_vector(self, v: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Sparse matrix-vector multiplication (SpMV).
        Complexity: O(nnz) instead of O(m*n) for dense.
        
        Args:
            v: Vector to multiply (length must equal number of columns)
            
        Returns:
            Result vector of length m (number of rows)
        """
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        
        v = v.flatten()
        if len(v) != self.shape[1]:
            raise ValueError(f"Vector length {len(v)} doesn't match matrix columns {self.shape[1]}")
        
        result = np.zeros(self.shape[0], dtype=np.float32)
        
        for i in range(self.shape[0]):
            start = self.row_ptr[i]
            end = self.row_ptr[i + 1]
            
            for idx in range(start, end):
                j = self.col_idx[idx]
                result[i] += self.values[idx] * v[j]
        
        return result
    
    def multiply_matrix(self, B: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Sparse matrix-matrix multiplication (SpMM).
        
        Args:
            B: Matrix to multiply (shape must be compatible)
            
        Returns:
            Result matrix C = self @ B
        """
        if isinstance(B, torch.Tensor):
            B = B.detach().cpu().numpy()
        
        if B.shape[0] != self.shape[1]:
            raise ValueError(f"Incompatible shapes for matrix multiplication: "
                           f"{self.shape} @ {B.shape}")
        
        result = np.zeros((self.shape[0], B.shape[1]), dtype=np.float32)
        
        for i in range(self.shape[0]):
            start = self.row_ptr[i]
            end = self.row_ptr[i + 1]
            
            for idx in range(start, end):
                j = self.col_idx[idx]
                result[i] += self.values[idx] * B[j]
        
        return result
    
    def transpose(self) -> 'CSRPadicMatrix':
        """
        Compute transpose of CSR matrix.
        
        Returns:
            New CSRPadicMatrix representing the transpose
        """
        # Convert to COO format for easier transposition
        rows = []
        cols = []
        data = []
        
        for i in range(self.shape[0]):
            start = self.row_ptr[i]
            end = self.row_ptr[i + 1]
            
            for idx in range(start, end):
                rows.append(i)
                cols.append(self.col_idx[idx])
                data.append(self.values[idx])
        
        # Create transposed matrix in dense format
        transposed = np.zeros((self.shape[1], self.shape[0]), dtype=np.float32)
        for r, c, v in zip(rows, cols, data):
            transposed[c, r] = v
        
        # Convert back to CSR
        return CSRPadicMatrix(transposed, threshold=self.threshold)
    
    def add(self, other: 'CSRPadicMatrix', alpha: float = 1.0) -> 'CSRPadicMatrix':
        """
        Add two CSR matrices: result = self + alpha * other.
        
        Args:
            other: Another CSR matrix with same shape
            alpha: Scalar multiplier for other matrix
            
        Returns:
            New CSR matrix containing the sum
        """
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        
        # Convert both to dense for addition (can be optimized later)
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        result = dense_self + alpha * dense_other
        
        return CSRPadicMatrix(result, threshold=self.threshold)
    
    def scale(self, alpha: float) -> 'CSRPadicMatrix':
        """
        Scale matrix by scalar: result = alpha * self.
        
        Args:
            alpha: Scalar multiplier
            
        Returns:
            New scaled CSR matrix
        """
        scaled = CSRPadicMatrix.__new__(CSRPadicMatrix)
        scaled.shape = self.shape
        scaled.threshold = self.threshold
        scaled.original_device = self.original_device
        scaled.original_dtype = self.original_dtype
        
        # Scale values in-place
        scaled.values = self.values * alpha
        scaled.col_idx = self.col_idx.copy()
        scaled.row_ptr = self.row_ptr.copy()
        
        # Recalculate metrics
        scaled.metrics = scaled._calculate_compression_metrics()
        
        return scaled
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Return comprehensive compression and matrix statistics.
        
        Returns:
            Dictionary containing all statistics and metrics
        """
        stats = {
            # Shape information
            'shape': self.shape,
            'rows': self.shape[0],
            'cols': self.shape[1],
            'total_elements': self.shape[0] * self.shape[1],
            
            # Sparsity metrics
            'nnz': self.metrics.nnz,
            'density': self.metrics.density,
            'sparsity': 1.0 - self.metrics.density,
            'threshold': self.threshold,
            
            # Compression metrics
            'compression_ratio': self.metrics.compression_ratio,
            'memory_saved_bytes': self.metrics.memory_saved_bytes,
            'dense_memory_bytes': self.metrics.dense_memory_bytes,
            'sparse_memory_bytes': self.metrics.sparse_memory_bytes,
            'bandwidth_reduction': self.metrics.bandwidth_reduction,
            
            # Distribution metrics
            'row_efficiency': self.metrics.row_efficiency,
            'max_row_nnz': int(np.max(np.diff(self.row_ptr))) if len(self.row_ptr) > 1 else 0,
            'min_row_nnz': int(np.min(np.diff(self.row_ptr))) if len(self.row_ptr) > 1 else 0,
            'empty_rows': int(np.sum(np.diff(self.row_ptr) == 0)) if len(self.row_ptr) > 1 else 0,
            
            # Value statistics
            'value_mean': float(np.mean(self.values)) if len(self.values) > 0 else 0.0,
            'value_std': float(np.std(self.values)) if len(self.values) > 0 else 0.0,
            'value_min': float(np.min(self.values)) if len(self.values) > 0 else 0.0,
            'value_max': float(np.max(self.values)) if len(self.values) > 0 else 0.0,
            'value_range': float(np.max(self.values) - np.min(self.values)) if len(self.values) > 0 else 0.0,
        }
        
        return stats
    
    def to_bytes(self) -> bytes:
        """
        Serialize CSR matrix to bytes for storage/transmission.
        
        Format:
        - Header (24 bytes): magic, version, shape, threshold
        - Row pointers: (m+1) * 4 bytes
        - Column indices: nnz * 4 bytes
        - Values: nnz * 4 bytes
        
        Returns:
            Serialized bytes representation
        """
        # Create header
        magic = b'CSR1'  # 4 bytes magic number
        version = struct.pack('I', 1)  # 4 bytes version
        shape = struct.pack('II', self.shape[0], self.shape[1])  # 8 bytes shape
        threshold = struct.pack('d', self.threshold)  # 8 bytes threshold
        
        header = magic + version + shape + threshold
        
        # Serialize arrays
        row_ptr_bytes = self.row_ptr.astype(np.int32).tobytes()
        col_idx_bytes = self.col_idx.astype(np.int32).tobytes()
        values_bytes = self.values.astype(np.float32).tobytes()
        
        return header + row_ptr_bytes + col_idx_bytes + values_bytes
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'CSRPadicMatrix':
        """
        Deserialize CSR matrix from bytes.
        
        Args:
            data: Serialized bytes from to_bytes()
            
        Returns:
            Reconstructed CSRPadicMatrix
        """
        # Parse header
        if len(data) < 24:
            raise ValueError("Invalid CSR data: too short for header")
        
        magic = data[:4]
        if magic != b'CSR1':
            raise ValueError(f"Invalid CSR magic number: {magic}")
        
        version = struct.unpack('I', data[4:8])[0]
        if version != 1:
            raise ValueError(f"Unsupported CSR version: {version}")
        
        rows, cols = struct.unpack('II', data[8:16])
        threshold = struct.unpack('d', data[16:24])[0]
        
        # Calculate array sizes
        row_ptr_size = (rows + 1) * 4
        offset = 24
        
        # Parse row pointers
        row_ptr = np.frombuffer(data[offset:offset + row_ptr_size], dtype=np.int32)
        offset += row_ptr_size
        
        # Calculate nnz from row pointers
        nnz = row_ptr[-1]
        
        # Parse column indices and values
        col_idx_size = nnz * 4
        values_size = nnz * 4
        
        col_idx = np.frombuffer(data[offset:offset + col_idx_size], dtype=np.int32)
        offset += col_idx_size
        
        values = np.frombuffer(data[offset:offset + values_size], dtype=np.float32)
        
        # Create CSR matrix directly
        matrix = cls.__new__(cls)
        matrix.shape = (rows, cols)
        matrix.threshold = threshold
        matrix.values = values
        matrix.col_idx = col_idx
        matrix.row_ptr = row_ptr
        matrix.original_device = None
        matrix.original_dtype = None
        matrix.metrics = matrix._calculate_compression_metrics()
        
        return matrix
    
    def validate(self) -> bool:
        """
        Validate CSR structure integrity.
        
        Returns:
            True if valid, raises ValueError if invalid
        """
        # Check array lengths
        if len(self.row_ptr) != self.shape[0] + 1:
            raise ValueError(f"Invalid row_ptr length: {len(self.row_ptr)} != {self.shape[0] + 1}")
        
        if len(self.values) != len(self.col_idx):
            raise ValueError(f"values and col_idx length mismatch: {len(self.values)} != {len(self.col_idx)}")
        
        # Check row pointers are monotonic
        if not np.all(np.diff(self.row_ptr) >= 0):
            raise ValueError("Row pointers must be monotonically increasing")
        
        # Check row pointers bounds
        if self.row_ptr[0] != 0:
            raise ValueError(f"First row pointer must be 0, got {self.row_ptr[0]}")
        
        if self.row_ptr[-1] != len(self.values):
            raise ValueError(f"Last row pointer must equal nnz: {self.row_ptr[-1]} != {len(self.values)}")
        
        # Check column indices are in bounds
        if len(self.col_idx) > 0:
            if np.min(self.col_idx) < 0:
                raise ValueError(f"Negative column index found: {np.min(self.col_idx)}")
            
            if np.max(self.col_idx) >= self.shape[1]:
                raise ValueError(f"Column index out of bounds: {np.max(self.col_idx)} >= {self.shape[1]}")
        
        return True


class BatchedCSROperations:
    """Efficient batched operations for multiple CSR matrices"""
    
    @staticmethod
    def batch_multiply_vector(matrices: List[CSRPadicMatrix], 
                             vectors: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Multiply multiple CSR matrices with corresponding vectors.
        
        Args:
            matrices: List of CSR matrices
            vectors: Array of vectors (shape: [batch, n])
            
        Returns:
            Results array (shape: [batch, m])
        """
        if isinstance(vectors, torch.Tensor):
            vectors = vectors.detach().cpu().numpy()
        
        if len(matrices) != vectors.shape[0]:
            raise ValueError(f"Batch size mismatch: {len(matrices)} matrices vs {vectors.shape[0]} vectors")
        
        results = []
        for matrix, vector in zip(matrices, vectors):
            results.append(matrix.multiply_vector(vector))
        
        return np.array(results, dtype=np.float32)
    
    @staticmethod
    def batch_to_dense(matrices: List[CSRPadicMatrix]) -> np.ndarray:
        """
        Convert multiple CSR matrices to dense format.
        
        Args:
            matrices: List of CSR matrices
            
        Returns:
            Dense array (shape: [batch, m, n])
        """
        if not matrices:
            return np.array([])
        
        # Check all matrices have same shape
        shape = matrices[0].shape
        for matrix in matrices[1:]:
            if matrix.shape != shape:
                raise ValueError(f"Shape mismatch in batch: {matrix.shape} vs {shape}")
        
        batch_size = len(matrices)
        dense = np.zeros((batch_size, shape[0], shape[1]), dtype=np.float32)
        
        for i, matrix in enumerate(matrices):
            dense[i] = matrix.to_dense()
        
        return dense
    
    @staticmethod
    def create_from_batch(tensors: Union[np.ndarray, torch.Tensor], 
                         threshold: float = 1e-6) -> List[CSRPadicMatrix]:
        """
        Create multiple CSR matrices from a batch of dense matrices.
        
        Args:
            tensors: Batch of dense matrices (shape: [batch, m, n])
            threshold: Sparsity threshold
            
        Returns:
            List of CSR matrices
        """
        if isinstance(tensors, torch.Tensor):
            tensors = tensors.detach().cpu().numpy()
        
        if tensors.ndim != 3:
            raise ValueError(f"Expected 3D tensor for batch, got shape {tensors.shape}")
        
        matrices = []
        for i in range(tensors.shape[0]):
            matrices.append(CSRPadicMatrix(tensors[i], threshold=threshold))
        
        return matrices


class GPUCSRMatrix:
    """GPU-accelerated CSR operations using PyTorch sparse tensors"""
    
    def __init__(self, csr_matrix: CSRPadicMatrix, device: Optional[torch.device] = None):
        """
        Create GPU-accelerated CSR representation.
        
        Args:
            csr_matrix: CPU CSR matrix to accelerate
            device: Target device (defaults to cuda if available)
        """
        self.shape = csr_matrix.shape
        self.threshold = csr_matrix.threshold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert CSR to COO format for PyTorch
        rows = []
        cols = []
        values = []
        
        for i in range(self.shape[0]):
            start = csr_matrix.row_ptr[i]
            end = csr_matrix.row_ptr[i + 1]
            
            for idx in range(start, end):
                rows.append(i)
                cols.append(csr_matrix.col_idx[idx])
                values.append(csr_matrix.values[idx])
        
        if len(values) > 0:
            indices = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
            values_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
            
            self.sparse_tensor = torch.sparse_coo_tensor(
                indices, values_tensor, self.shape, 
                dtype=torch.float32, device=self.device
            )
        else:
            # Empty matrix
            self.sparse_tensor = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=self.device),
                torch.zeros(0, dtype=torch.float32, device=self.device),
                self.shape, dtype=torch.float32, device=self.device
            )
        
        # Cache metrics
        self.nnz = len(values)
        self.density = self.nnz / (self.shape[0] * self.shape[1]) if self.shape[0] * self.shape[1] > 0 else 0.0
    
    def multiply_vector(self, v: torch.Tensor) -> torch.Tensor:
        """
        GPU-accelerated sparse matrix-vector multiplication.
        
        Args:
            v: Vector on GPU
            
        Returns:
            Result vector on GPU
        """
        v = v.to(self.device)
        if v.dim() == 1:
            v = v.unsqueeze(1)
        
        result = torch.sparse.mm(self.sparse_tensor, v)
        return result.squeeze(1)
    
    def multiply_matrix(self, B: torch.Tensor) -> torch.Tensor:
        """
        GPU-accelerated sparse matrix-matrix multiplication.
        
        Args:
            B: Dense matrix on GPU
            
        Returns:
            Result matrix on GPU
        """
        B = B.to(self.device)
        return torch.sparse.mm(self.sparse_tensor, B)
    
    def to_dense(self) -> torch.Tensor:
        """Convert to dense tensor on GPU"""
        return self.sparse_tensor.to_dense()
    
    def to_cpu_csr(self) -> CSRPadicMatrix:
        """Convert back to CPU CSR format"""
        dense = self.sparse_tensor.to_dense().cpu().numpy()
        return CSRPadicMatrix(dense, threshold=self.threshold)


# Performance monitoring utilities
class CSRPerformanceMonitor:
    """Monitor and analyze CSR compression performance"""
    
    def __init__(self):
        self.compression_history = []
        self.operation_timings = {}
    
    def record_compression(self, original_size: int, compressed_size: int, 
                          sparsity: float, compression_time: float):
        """Record compression statistics"""
        self.compression_history.append({
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': original_size / compressed_size if compressed_size > 0 else 1.0,
            'sparsity': sparsity,
            'compression_time': compression_time,
            'timestamp': np.datetime64('now')
        })
    
    def record_operation(self, operation: str, matrix_shape: Tuple[int, int], 
                        nnz: int, execution_time: float):
        """Record operation timing"""
        if operation not in self.operation_timings:
            self.operation_timings[operation] = []
        
        self.operation_timings[operation].append({
            'shape': matrix_shape,
            'nnz': nnz,
            'density': nnz / (matrix_shape[0] * matrix_shape[1]),
            'execution_time': execution_time,
            'gflops': self._calculate_gflops(operation, matrix_shape, nnz, execution_time)
        })
    
    def _calculate_gflops(self, operation: str, shape: Tuple[int, int], 
                         nnz: int, time: float) -> float:
        """Calculate GFLOPS for operation"""
        if time <= 0:
            return 0.0
        
        if operation == 'spmv':  # Sparse matrix-vector
            flops = 2 * nnz  # One multiply, one add per non-zero
        elif operation == 'spmm':  # Sparse matrix-matrix
            flops = 2 * nnz * shape[1]  # For each output column
        else:
            flops = nnz  # Default
        
        return (flops / 1e9) / time
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.compression_history:
            return {}
        
        compression_ratios = [h['compression_ratio'] for h in self.compression_history]
        sparsities = [h['sparsity'] for h in self.compression_history]
        
        summary = {
            'total_compressions': len(self.compression_history),
            'average_compression_ratio': np.mean(compression_ratios),
            'best_compression_ratio': np.max(compression_ratios),
            'average_sparsity': np.mean(sparsities),
            'operation_performance': {}
        }
        
        for op, timings in self.operation_timings.items():
            if timings:
                gflops = [t['gflops'] for t in timings]
                summary['operation_performance'][op] = {
                    'count': len(timings),
                    'average_gflops': np.mean(gflops),
                    'peak_gflops': np.max(gflops)
                }
        
        return summary