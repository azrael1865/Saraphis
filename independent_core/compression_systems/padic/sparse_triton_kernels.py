"""
Triton kernels for GPU-accelerated sparse operations
Optimized for p-adic sparse tensor encoding without dynamic shapes
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def count_nonzeros_kernel(
    data_ptr,
    counts_ptr,
    threshold,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Count non-zero elements per row in parallel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    data = tl.load(data_ptr + offsets, mask=mask, other=0.0)
    
    # Count non-zeros
    nonzero_mask = tl.abs(data) > threshold
    count = tl.sum(tl.where(nonzero_mask & mask, 1, 0))
    
    # Atomic add to global counter
    tl.atomic_add(counts_ptr, count)


@triton.jit
def extract_nonzeros_kernel(
    data_ptr,
    row_ptr,
    col_ptr,
    indices_out_ptr,
    values_out_ptr,
    write_idx_ptr,
    threshold,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Extract non-zero indices and values in parallel"""
    pid = tl.program_id(0)
    
    # Calculate which elements this thread processes
    elements_per_thread = (n_rows * n_cols + tl.num_programs(0) - 1) // tl.num_programs(0)
    start_idx = pid * elements_per_thread
    end_idx = tl.minimum(start_idx + elements_per_thread, n_rows * n_cols)
    
    for idx in range(start_idx, end_idx):
        # Convert flat index to 2D
        row = idx // n_cols
        col = idx % n_cols
        
        # Load value
        value = tl.load(data_ptr + idx)
        
        # Check if non-zero
        if tl.abs(value) > threshold:
            # Atomically get write position
            write_pos = tl.atomic_add(write_idx_ptr, 1)
            
            # Write indices and value
            tl.store(indices_out_ptr + write_pos * 2, row)
            tl.store(indices_out_ptr + write_pos * 2 + 1, col)
            tl.store(values_out_ptr + write_pos, value)


@triton.jit
def create_csr_indices_kernel(
    row_indices_ptr,
    crow_indices_ptr,
    n_rows: tl.constexpr,
    n_nonzeros: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Create CSR crow_indices from row indices"""
    pid = tl.program_id(0)
    
    if pid < n_rows:
        # Count occurrences of this row index
        count = 0
        for i in range(n_nonzeros):
            row_idx = tl.load(row_indices_ptr + i)
            if row_idx == pid:
                count += 1
        
        # Store count for this row
        tl.store(crow_indices_ptr + pid + 1, count)


class TritonSparseOps:
    """GPU-accelerated sparse operations using Triton"""
    
    @staticmethod
    def count_nonzeros(data: torch.Tensor, threshold: float) -> int:
        """Count non-zero elements above threshold"""
        n_elements = data.numel()
        counts = torch.zeros(1, dtype=torch.int32, device=data.device)
        
        # Launch kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        count_nonzeros_kernel[grid](
            data, counts, threshold, n_elements, BLOCK_SIZE=1024
        )
        
        return counts.item()
    
    @staticmethod
    def extract_nonzeros(data: torch.Tensor, threshold: float, 
                        max_nnz: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract non-zero indices and values"""
        n_rows, n_cols = data.shape
        n_elements = data.numel()
        
        # Pre-allocate output arrays
        indices = torch.zeros((max_nnz, 2), dtype=torch.long, device=data.device)
        values = torch.zeros(max_nnz, dtype=data.dtype, device=data.device)
        write_idx = torch.zeros(1, dtype=torch.int32, device=data.device)
        
        # Create row and column index tensors
        row_indices = torch.arange(n_rows, device=data.device).unsqueeze(1).expand(-1, n_cols).flatten()
        col_indices = torch.arange(n_cols, device=data.device).unsqueeze(0).expand(n_rows, -1).flatten()
        
        # Launch kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        extract_nonzeros_kernel[grid](
            data.flatten(), row_indices, col_indices,
            indices, values, write_idx,
            threshold, n_rows, n_cols, BLOCK_SIZE=256
        )
        
        # Get actual number of non-zeros
        actual_nnz = write_idx.item()
        
        # Return only the filled portion
        return indices[:actual_nnz], values[:actual_nnz]
    
    @staticmethod
    def create_csr_format(row_indices: torch.Tensor, col_indices: torch.Tensor,
                         values: torch.Tensor, shape: Tuple[int, int]) -> torch.sparse.Tensor:
        """Create CSR sparse tensor from indices and values"""
        n_rows = shape[0]
        
        # Create crow_indices using histogram
        crow_indices = torch.zeros(n_rows + 1, dtype=torch.long, device=row_indices.device)
        row_counts = torch.bincount(row_indices, minlength=n_rows)
        crow_indices[1:] = torch.cumsum(row_counts, dim=0)
        
        # Sort by row index for CSR format
        sort_idx = torch.argsort(row_indices, stable=True)
        sorted_col_indices = col_indices[sort_idx]
        sorted_values = values[sort_idx]
        
        # Create CSR tensor
        sparse_tensor = torch.sparse_csr_tensor(
            crow_indices=crow_indices,
            col_indices=sorted_col_indices,
            values=sorted_values,
            size=shape,
            dtype=values.dtype,
            device=values.device
        )
        
        return sparse_tensor


def gpu_sparse_encode(padic_digits: torch.Tensor, valuations: torch.Tensor,
                     threshold: float = 1e-6) -> torch.sparse.Tensor:
    """
    GPU-optimized sparse encoding using Triton kernels
    
    Args:
        padic_digits: P-adic digit tensor [batch_size, precision]
        valuations: Valuation tensor [batch_size]
        threshold: Sparsity threshold
        
    Returns:
        Sparse CSR tensor with valuations attached
    """
    if not padic_digits.is_cuda:
        raise ValueError("GPU sparse encoding requires CUDA tensors")
    
    ops = TritonSparseOps()
    
    # Count non-zeros first
    nnz = ops.count_nonzeros(padic_digits, threshold)
    
    if nnz == 0:
        # Create empty sparse tensor
        batch_size = padic_digits.shape[0]
        crow_indices = torch.zeros(batch_size + 1, dtype=torch.long, device=padic_digits.device)
        col_indices = torch.zeros(0, dtype=torch.long, device=padic_digits.device)
        values = torch.zeros(0, dtype=padic_digits.dtype, device=padic_digits.device)
        
        sparse_tensor = torch.sparse_csr_tensor(
            crow_indices=crow_indices,
            col_indices=col_indices,
            values=values,
            size=padic_digits.shape,
            dtype=padic_digits.dtype,
            device=padic_digits.device
        )
    else:
        # Extract non-zeros
        indices, values = ops.extract_nonzeros(padic_digits, threshold, nnz)
        row_indices = indices[:, 0]
        col_indices = indices[:, 1]
        
        # Create CSR format
        sparse_tensor = ops.create_csr_format(row_indices, col_indices, values, padic_digits.shape)
    
    # Attach valuations
    sparse_tensor.valuations = valuations
    
    return sparse_tensor