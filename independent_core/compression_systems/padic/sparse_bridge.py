"""
sparse_bridge.py - Modified version without any Triton dependencies
Uses pure PyTorch for all sparse operations
"""

import torch
import torch.sparse
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import sys
import os
import logging
import time

# Configure logging
logger = logging.getLogger(__name__)

# NO TRITON IMPORTS - Using pure PyTorch instead


class SparsePAdicBridge:
    """
    Bridge between p-adic weights and CSR sparse format using pure PyTorch.
    No Triton dependencies - all operations use standard PyTorch.
    """

    def __init__(self, sparsity_threshold: float = 1e-6, use_gpu: bool = True, device: Optional[torch.device] = None,
                 use_optimized_sparse: bool = True):
        """
        Initialize sparse bridge with configurable threshold.
        
        Args:
            sparsity_threshold: Values below this are considered zero
            use_gpu: Whether to use GPU acceleration when available
            device: Specific device to use (auto-detects if None)
            use_optimized_sparse: Use optimized sparse operations
        """
        self.threshold = sparsity_threshold
        self.use_optimized_sparse = use_optimized_sparse
        
        # Device handling - use provided device or auto-detect
        if device is not None:
            # Handle both string and torch.device inputs
            if isinstance(device, str):
                self.device = torch.device(device)
            else:
                self.device = device
            self.use_gpu = (self.device.type == 'cuda')
        else:
            self.use_gpu = use_gpu and torch.cuda.is_available()
            self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # Cache for frequently used sparse patterns
        self.pattern_cache: Dict[int, torch.Tensor] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # NO TRITON FLAGS
        self.use_triton = False  # Always False - no Triton support
        self.use_memory_pool = False  # No memory pool without Triton
        self.pooled_bridge = None  # No pooled bridge
        
        logger.info(f"SparsePAdicBridge initialized (PyTorch-only): threshold={sparsity_threshold}, "
                   f"device={self.device}, optimized={self.use_optimized_sparse}")
    
    def set_device(self, device: torch.device):
        """Set the device for tensor operations"""
        self.device = device
        self.use_gpu = (device.type == 'cuda')
        self.clear_cache()

    def padic_to_sparse(self, padic_digits: torch.Tensor, 
                        valuations: torch.Tensor,
                        original_shape: list = None) -> torch.sparse.Tensor:
        """
        Convert p-adic representation to sparse CSR format using pure PyTorch.
        
        Args:
            padic_digits: P-adic digit tensor [batch_size, precision]
            valuations: Valuation tensor [batch_size]
            original_shape: Original shape to preserve in metadata
            
        Returns:
            Sparse CSR tensor with valuations stored as attribute
        """
        start_time = time.perf_counter() if logger.isEnabledFor(logging.DEBUG) else None
        
        # Use standard PyTorch sparse encoding
        result = self._padic_to_sparse_pytorch(padic_digits, valuations)
        
        # Store shape metadata
        result.digit_tensor_shape = list(padic_digits.shape)
        if original_shape is not None:
            result.original_shape = original_shape
        
        if start_time:
            logger.debug(f"PyTorch sparse encoding: {(time.perf_counter() - start_time)*1000:.2f} ms")
        
        return result
    
    def _padic_to_sparse_pytorch(self, padic_digits: torch.Tensor,
                                  valuations: torch.Tensor) -> torch.sparse.Tensor:
        """Pure PyTorch sparse encoding implementation"""
        batch_size, precision = padic_digits.shape
        
        # Validate inputs
        if padic_digits.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {padic_digits.dim()}D")
        
        if valuations.shape[0] != batch_size:
            raise ValueError(f"Valuation size mismatch: {valuations.shape[0]} != {batch_size}")
        
        # Move to appropriate device
        padic_digits = padic_digits.to(self.device, non_blocking=True)
        valuations = valuations.to(self.device, non_blocking=True)
        
        # Find non-zero elements (above threshold)
        with torch.no_grad():
            mask = torch.abs(padic_digits) > self.threshold
        
        # Get indices of non-zero elements using standard PyTorch
        indices = torch.nonzero(mask, as_tuple=False)
        
        if len(indices) == 0:
            # Completely sparse matrix - create empty CSR
            crow_indices = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            col_indices = torch.zeros(0, dtype=torch.long, device=self.device)
            values = torch.zeros(0, dtype=padic_digits.dtype, device=self.device)
            
            sparse_tensor = torch.sparse_csr_tensor(
                crow_indices=crow_indices,
                col_indices=col_indices,
                values=values,
                size=padic_digits.shape,
                dtype=padic_digits.dtype,
                device=self.device
            )
        else:
            # Extract row and column indices
            row_indices = indices[:, 0]
            col_indices = indices[:, 1]
            values = padic_digits[mask]
            
            # Create CSR indices using PyTorch operations
            crow_indices = self._create_csr_indices_pytorch(row_indices, batch_size)
            sorted_col_indices, sorted_values = self._sort_csr_data_pytorch(row_indices, col_indices, values)
            
            # Create CSR tensor
            sparse_tensor = torch.sparse_csr_tensor(
                crow_indices=crow_indices,
                col_indices=sorted_col_indices,
                values=sorted_values,
                size=padic_digits.shape,
                dtype=padic_digits.dtype,
                device=self.device
            )
        
        # Store valuations separately
        sparse_tensor.valuations = valuations
        sparse_tensor.digit_tensor_shape = list(padic_digits.shape)
        
        # Record compression metrics
        nnz = len(indices)
        total_elements = batch_size * precision
        sparsity = 1.0 - (nnz / total_elements if total_elements > 0 else 0.0)
        
        logger.debug(f"P-adic to sparse CSR: shape={padic_digits.shape}, nnz={nnz}, "
                    f"sparsity={sparsity:.2%}")
        
        return sparse_tensor
    
    def _create_csr_indices_pytorch(self, row_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Create CSR crow_indices using pure PyTorch"""
        # Count non-zeros per row
        row_counts = torch.bincount(row_indices, minlength=batch_size)
        
        # Create crow_indices (cumulative sum)
        crow_indices = torch.zeros(batch_size + 1, dtype=torch.long, device=row_indices.device)
        crow_indices[1:] = torch.cumsum(row_counts, dim=0)
        
        return crow_indices
    
    def _sort_csr_data_pytorch(self, row_indices: torch.Tensor, col_indices: torch.Tensor, 
                               values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sort indices and values for CSR format using pure PyTorch"""
        # Sort by row index for CSR format
        sort_idx = torch.argsort(row_indices, stable=True)
        sorted_col_indices = col_indices[sort_idx]
        sorted_values = values[sort_idx]
        
        return sorted_col_indices, sorted_values

    def sparse_to_padic(self, sparse_tensor: torch.sparse.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert sparse CSR back to dense p-adic format.
        
        Args:
            sparse_tensor: Sparse CSR tensor with valuations attribute
            
        Returns:
            Tuple of (padic_digits, valuations) tensors
        """
        # Convert to dense - works for both CSR and COO
        padic_digits = sparse_tensor.to_dense()
        
        # Use stored digit_tensor_shape if available for accurate reconstruction
        if hasattr(sparse_tensor, 'digit_tensor_shape'):
            expected_shape = sparse_tensor.digit_tensor_shape
            if list(padic_digits.shape) != expected_shape:
                logger.debug(f"Reshaping sparse tensor from {list(padic_digits.shape)} to stored shape {expected_shape}")
                # Ensure we have the right number of elements
                if padic_digits.numel() == expected_shape[0] * expected_shape[1]:
                    padic_digits = padic_digits.reshape(expected_shape)
                else:
                    logger.warning(f"Shape mismatch: sparse has {padic_digits.numel()} elements, expected {expected_shape[0] * expected_shape[1]}")
        
        # Retrieve stored valuations
        if hasattr(sparse_tensor, 'valuations'):
            valuations = sparse_tensor.valuations
        else:
            # Fallback: create zero valuations
            logger.warning("Sparse tensor missing valuations attribute, using zeros")
            valuations = torch.zeros(padic_digits.shape[0], 
                                    dtype=torch.int32, 
                                    device=padic_digits.device)
        
        return padic_digits, valuations

    def get_compression_ratio(self, sparse_tensor: torch.sparse.Tensor) -> float:
        """Calculate actual compression ratio achieved."""
        shape = sparse_tensor.shape
        dense_size = shape[0] * shape[1] * 4  # float32 = 4 bytes
        
        # CSR storage: values + col_indices + crow_indices
        nnz = sparse_tensor._nnz() if hasattr(sparse_tensor, '_nnz') else sparse_tensor.values().numel()
        sparse_size = (
            nnz * 4 +               # values (float32)
            nnz * 8 +               # col_indices (int64)
            (shape[0] + 1) * 8      # crow_indices (int64)
        )
        
        # Add valuation storage
        sparse_size += shape[0] * 4  # valuations (int32)
        
        ratio = dense_size / sparse_size if sparse_size > 0 else 1.0
        
        logger.debug(f"Compression ratio: {ratio:.2f}x "
                    f"(dense={dense_size} bytes, sparse={sparse_size} bytes)")
        
        return ratio

    def batch_padic_to_sparse(self, padic_batch: torch.Tensor,
                              valuations_batch: torch.Tensor) -> List[torch.sparse.Tensor]:
        """Convert batch of p-adic weights to list of sparse tensors."""
        batch_size, num_weights, precision = padic_batch.shape
        
        # Convert each batch item separately
        sparse_results = []
        for i in range(batch_size):
            padic_item = padic_batch[i]  # Shape: [num_weights, precision]
            valuations_item = valuations_batch[i]  # Shape: [num_weights]
            
            sparse_tensor = self.padic_to_sparse(padic_item, valuations_item)
            sparse_results.append(sparse_tensor)
        
        return sparse_results

    def batch_sparse_to_padic(self, sparse_tensor: torch.sparse.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert sparse tensor back to batch of p-adic weights."""
        # Convert to dense
        padic_flat, valuations_flat = self.sparse_to_padic(sparse_tensor)
        
        # Reshape to original batch dimensions
        if hasattr(sparse_tensor, 'original_shape'):
            batch_shape = sparse_tensor.original_shape
            val_shape = sparse_tensor.valuations_shape
            
            padic_batch = padic_flat.reshape(batch_shape)
            valuations_batch = valuations_flat.reshape(val_shape)
        else:
            # Fallback: return flat tensors
            logger.warning("Sparse tensor missing original_shape, returning flat tensors")
            padic_batch = padic_flat
            valuations_batch = valuations_flat
        
        return padic_batch, valuations_batch

    def optimize_for_pattern(self, sparse_tensor: torch.sparse.Tensor,
                            pattern_type: str = 'block', pattern: str = None) -> torch.sparse.Tensor:
        """Optimize sparse tensor for specific sparsity patterns."""
        # Support both parameter names for compatibility
        if pattern is not None:
            pattern_type = pattern
        
        if pattern_type == 'block':
            return self._optimize_block_sparse(sparse_tensor)
        elif pattern_type == 'diagonal':
            return self._optimize_diagonal_sparse(sparse_tensor)
        elif pattern_type == 'banded':
            return self._optimize_banded_sparse(sparse_tensor)
        else:
            return sparse_tensor

    def _optimize_block_sparse(self, sparse_tensor: torch.sparse.Tensor,
                               block_size: int = 4) -> torch.sparse.Tensor:
        """Optimize for block sparse patterns using PyTorch."""
        # Convert to dense temporarily for block analysis
        dense = sparse_tensor.to_dense()
        shape = dense.shape
        
        # Pad to multiple of block_size
        pad_rows = (block_size - shape[0] % block_size) % block_size
        pad_cols = (block_size - shape[1] % block_size) % block_size
        
        if pad_rows > 0 or pad_cols > 0:
            dense = torch.nn.functional.pad(dense, (0, pad_cols, 0, pad_rows))
        
        # Analyze blocks
        blocked_shape = (dense.shape[0] // block_size, 
                        dense.shape[1] // block_size)
        
        # Find non-zero blocks
        block_mask = torch.zeros(blocked_shape, dtype=torch.bool, device=dense.device)
        
        for i in range(blocked_shape[0]):
            for j in range(blocked_shape[1]):
                block = dense[i*block_size:(i+1)*block_size,
                            j*block_size:(j+1)*block_size]
                if torch.any(torch.abs(block) > self.threshold):
                    block_mask[i, j] = True
        
        # Create optimized sparse tensor with block structure
        indices = []
        values = []
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                block_i = i // block_size
                block_j = j // block_size
                if block_i < blocked_shape[0] and block_j < blocked_shape[1]:
                    if block_mask[block_i, block_j] and abs(dense[i, j]) > self.threshold:
                        indices.append([i, j])
                        values.append(dense[i, j])
        
        if len(values) > 0:
            indices_tensor = torch.tensor(indices, device=dense.device).T
            values_tensor = torch.tensor(values, dtype=dense.dtype, device=dense.device)
            
            # Create new sparse tensor with block structure
            optimized = torch.sparse_coo_tensor(
                indices=indices_tensor,
                values=values_tensor,
                size=shape,
                dtype=dense.dtype,
                device=dense.device
            ).coalesce()
        else:
            # Empty tensor
            optimized = torch.sparse_coo_tensor(
                indices=torch.zeros((2, 0), dtype=torch.long, device=dense.device),
                values=torch.zeros(0, dtype=dense.dtype, device=dense.device),
                size=shape,
                dtype=dense.dtype,
                device=dense.device
            ).coalesce()
        
        # Preserve valuations
        if hasattr(sparse_tensor, 'valuations'):
            optimized.valuations = sparse_tensor.valuations
        
        return optimized

    def _optimize_diagonal_sparse(self, sparse_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
        """Optimize for diagonal/near-diagonal patterns using PyTorch."""
        # Get COO representation
        indices = sparse_tensor.indices()
        values = sparse_tensor.values()
        
        if len(indices) > 0:
            # Sort by diagonal distance (row - col)
            diagonal_dist = indices[0] - indices[1]
            sorted_idx = torch.argsort(diagonal_dist)
            
            # Reorder for better cache performance
            sorted_indices = indices[:, sorted_idx]
            sorted_values = values[sorted_idx]
            
            # Recreate sparse tensor with optimized ordering
            optimized = torch.sparse_coo_tensor(
                indices=sorted_indices,
                values=sorted_values,
                size=sparse_tensor.shape,
                dtype=sparse_tensor.dtype,
                device=sparse_tensor.device
            ).coalesce()
        else:
            optimized = sparse_tensor
        
        # Preserve valuations
        if hasattr(sparse_tensor, 'valuations'):
            optimized.valuations = sparse_tensor.valuations
        
        return optimized

    def _optimize_banded_sparse(self, sparse_tensor: torch.sparse.Tensor,
                                bandwidth: Optional[int] = None) -> torch.sparse.Tensor:
        """Optimize for banded matrix patterns using PyTorch."""
        # Convert to dense for band analysis
        dense = sparse_tensor.to_dense()
        shape = dense.shape
        
        # Auto-detect bandwidth if not provided
        if bandwidth is None:
            bandwidth = self._detect_bandwidth(dense)
        
        # Create banded sparse tensor
        indices = []
        values = []
        
        for i in range(shape[0]):
            for j in range(max(0, i - bandwidth), min(shape[1], i + bandwidth + 1)):
                if abs(dense[i, j]) > self.threshold:
                    indices.append([i, j])
                    values.append(dense[i, j])
        
        if len(values) > 0:
            indices_tensor = torch.tensor(indices, device=dense.device).T
            values_tensor = torch.tensor(values, dtype=dense.dtype, device=dense.device)
            
            optimized = torch.sparse_coo_tensor(
                indices=indices_tensor,
                values=values_tensor,
                size=shape,
                dtype=dense.dtype,
                device=dense.device
            ).coalesce()
        else:
            optimized = torch.sparse_coo_tensor(
                indices=torch.zeros((2, 0), dtype=torch.long, device=dense.device),
                values=torch.zeros(0, dtype=dense.dtype, device=dense.device),
                size=shape,
                dtype=dense.dtype,
                device=dense.device
            ).coalesce()
        
        # Preserve valuations
        if hasattr(sparse_tensor, 'valuations'):
            optimized.valuations = sparse_tensor.valuations
        
        return optimized

    def _detect_bandwidth(self, matrix: torch.Tensor) -> int:
        """Auto-detect bandwidth of a matrix using PyTorch."""
        max_dist = 0
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if abs(matrix[i, j]) > self.threshold:
                    dist = abs(i - j)
                    max_dist = max(max_dist, dist)
        
        return max_dist

    def get_memory_stats(self, sparse_tensor: torch.sparse.Tensor) -> Dict[str, Any]:
        """Get detailed memory statistics for sparse tensor."""
        shape = sparse_tensor.shape
        nnz = sparse_tensor._nnz() if hasattr(sparse_tensor, '_nnz') else sparse_tensor.values().numel()
        
        # Calculate memory usage
        dense_memory = shape[0] * shape[1] * 4  # float32
        sparse_memory = nnz * 4 + nnz * 4 + (shape[0] + 1) * 4
        
        if hasattr(sparse_tensor, 'valuations'):
            valuations_memory = sparse_tensor.valuations.numel() * 4
        else:
            valuations_memory = 0
        
        total_sparse_memory = sparse_memory + valuations_memory
        
        stats = {
            'shape': shape,
            'nnz': nnz,
            'density': nnz / (shape[0] * shape[1]) if shape[0] * shape[1] > 0 else 0.0,
            'sparsity': 1.0 - (nnz / (shape[0] * shape[1]) if shape[0] * shape[1] > 0 else 0.0),
            'sparsity_ratio': 1.0 - (nnz / (shape[0] * shape[1]) if shape[0] * shape[1] > 0 else 0.0),
            'dense_memory_bytes': dense_memory,
            'sparse_memory_bytes': sparse_memory,
            'valuations_memory_bytes': valuations_memory,
            'total_sparse_memory_bytes': total_sparse_memory,
            'compression_ratio': dense_memory / total_sparse_memory if total_sparse_memory > 0 else 1.0,
            'memory_saved_bytes': dense_memory - total_sparse_memory,
            'memory_saved_percent': ((dense_memory - total_sparse_memory) / dense_memory * 100 
                                     if dense_memory > 0 else 0.0),
            'device': str(sparse_tensor.device),
            'dtype': str(sparse_tensor.dtype),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': (self.cache_hits / (self.cache_hits + self.cache_misses) 
                              if (self.cache_hits + self.cache_misses) > 0 else 0.0)
        }
        
        return stats

    def validate_compression(self, original: torch.Tensor,
                           sparse: torch.sparse.Tensor,
                           tolerance: float = 1e-5) -> bool:
        """Validate that sparse representation preserves information."""
        # Reconstruct from sparse
        reconstructed = sparse.to_dense()
        
        # Check reconstruction error
        diff = torch.abs(original - reconstructed)
        max_diff = torch.max(diff).item()
        
        if max_diff > tolerance:
            logger.warning(f"Compression validation failed: max_diff={max_diff} > {tolerance}")
            return False
        
        return True

    def clear_cache(self):
        """Clear pattern cache to free memory."""
        self.pattern_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Pattern cache cleared")


# Additional CSR support functions that don't use Triton
class AdaptiveSparsityManager:
    """Manages adaptive sparsity thresholds based on compression targets."""
    
    def __init__(self, target_compression_ratio: float = 20.0,
                 min_threshold: float = 1e-8,
                 max_threshold: float = 1e-3):
        """Initialize adaptive sparsity manager."""
        self.target_ratio = target_compression_ratio
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.current_threshold = 1e-6
        self.history = []
        
    def update_threshold(self, achieved_ratio: float) -> float:
        """Update sparsity threshold based on achieved compression ratio."""
        # Record history
        self.history.append({
            'threshold': self.current_threshold,
            'ratio': achieved_ratio
        })
        
        # Adjust threshold
        if achieved_ratio < self.target_ratio:
            # Need more sparsity - increase threshold
            self.current_threshold = min(
                self.current_threshold * 1.5,
                self.max_threshold
            )
        elif achieved_ratio > self.target_ratio * 1.2:  # More sensitive threshold
            # Too sparse - can reduce threshold
            self.current_threshold = max(
                self.current_threshold * 0.75,
                self.min_threshold
            )
        
        logger.debug(f"Adaptive threshold updated: {self.current_threshold:.2e} "
                    f"(ratio: {achieved_ratio:.2f}x, target: {self.target_ratio:.2f}x)")
        
        return self.current_threshold
    
    def get_optimal_threshold(self) -> float:
        """Get optimal threshold based on history."""
        if not self.history:
            return self.current_threshold
        
        # Find threshold that achieved closest to target ratio
        best_entry = min(self.history, 
                        key=lambda x: abs(x['ratio'] - self.target_ratio))
        
        return best_entry['threshold']
