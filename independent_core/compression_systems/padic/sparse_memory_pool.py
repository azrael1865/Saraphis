"""
Memory pool for efficient sparse tensor operations
Reduces allocation overhead for dynamic sparse operations
"""

import torch
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SparseMemoryPool:
    """
    Memory pool for sparse tensor operations to reduce allocation overhead
    Pre-allocates buffers and reuses them across operations
    """
    
    def __init__(self, device: torch.device, initial_size: int = 100000):
        """
        Initialize memory pool
        
        Args:
            device: Device to allocate memory on
            initial_size: Initial pool size for each buffer type
        """
        self.device = device
        self.initial_size = initial_size
        
        # Pre-allocated buffers
        self.buffers = {
            'indices': torch.zeros((initial_size, 2), dtype=torch.long, device=device),
            'values': torch.zeros(initial_size, dtype=torch.float32, device=device),
            'row_indices': torch.zeros(initial_size, dtype=torch.long, device=device),
            'col_indices': torch.zeros(initial_size, dtype=torch.long, device=device),
            'crow_indices': torch.zeros(initial_size // 100, dtype=torch.long, device=device),
            'mask': torch.zeros(initial_size, dtype=torch.bool, device=device)
        }
        
        # Track usage
        self.allocated_sizes = {k: v.shape[0] for k, v in self.buffers.items()}
        self.peak_usage = {k: 0 for k in self.buffers.keys()}
        self.allocation_count = 0
        self.reuse_count = 0
        
        logger.info(f"SparseMemoryPool initialized on {device} with size {initial_size}")
    
    def get_buffer(self, buffer_type: str, size: int, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Get a buffer from the pool, resizing if necessary
        
        Args:
            buffer_type: Type of buffer ('indices', 'values', etc.)
            size: Required size
            dtype: Optional dtype override
            
        Returns:
            Buffer tensor view of requested size
        """
        if buffer_type not in self.buffers:
            raise ValueError(f"Unknown buffer type: {buffer_type}")
        
        current_buffer = self.buffers[buffer_type]
        current_size = self.allocated_sizes[buffer_type]
        
        # Check if resize needed
        if size > current_size:
            # Grow by 50% to reduce future allocations
            new_size = int(size * 1.5)
            logger.debug(f"Resizing {buffer_type} buffer from {current_size} to {new_size}")
            
            # Create new buffer
            if buffer_type == 'indices':
                new_buffer = torch.zeros((new_size, 2), dtype=torch.long, device=self.device)
            elif buffer_type in ['mask']:
                new_buffer = torch.zeros(new_size, dtype=torch.bool, device=self.device)
            else:
                new_buffer = torch.zeros(new_size, dtype=dtype or current_buffer.dtype, device=self.device)
            
            # Copy existing data
            if buffer_type == 'indices':
                new_buffer[:current_size] = current_buffer
            else:
                new_buffer[:current_size] = current_buffer
            
            self.buffers[buffer_type] = new_buffer
            self.allocated_sizes[buffer_type] = new_size
            self.allocation_count += 1
        else:
            self.reuse_count += 1
        
        # Update peak usage
        self.peak_usage[buffer_type] = max(self.peak_usage[buffer_type], size)
        
        # Return view of requested size
        if buffer_type == 'indices':
            return self.buffers[buffer_type][:size]
        else:
            return self.buffers[buffer_type][:size]
    
    def get_sparse_buffers(self, max_nnz: int, n_rows: int) -> Dict[str, torch.Tensor]:
        """
        Get all buffers needed for sparse tensor creation
        
        Args:
            max_nnz: Maximum number of non-zeros
            n_rows: Number of rows
            
        Returns:
            Dictionary of buffer views
        """
        return {
            'row_indices': self.get_buffer('row_indices', max_nnz),
            'col_indices': self.get_buffer('col_indices', max_nnz),
            'values': self.get_buffer('values', max_nnz),
            'crow_indices': self.get_buffer('crow_indices', n_rows + 1)
        }
    
    def clear(self):
        """Clear all buffers (fill with zeros)"""
        for buffer in self.buffers.values():
            buffer.zero_()
    
    def get_stats(self) -> Dict[str, any]:
        """Get memory pool statistics"""
        total_allocated = sum(
            buf.numel() * buf.element_size() for buf in self.buffers.values()
        )
        
        total_peak_usage = sum(
            self.peak_usage[k] * self.buffers[k].element_size() 
            for k in self.buffers.keys()
        )
        
        return {
            'total_allocated_mb': total_allocated / (1024 * 1024),
            'total_peak_usage_mb': total_peak_usage / (1024 * 1024),
            'allocation_count': self.allocation_count,
            'reuse_count': self.reuse_count,
            'reuse_ratio': self.reuse_count / (self.allocation_count + self.reuse_count) if (self.allocation_count + self.reuse_count) > 0 else 0,
            'buffer_sizes': self.allocated_sizes,
            'peak_usage': self.peak_usage
        }
    
    def optimize_sizes(self):
        """Optimize buffer sizes based on peak usage"""
        for buffer_type, peak in self.peak_usage.items():
            current_size = self.allocated_sizes[buffer_type]
            
            # If peak usage is much smaller than allocated, shrink
            if peak < current_size * 0.5 and peak > 0:
                new_size = int(peak * 1.2)  # 20% headroom
                logger.info(f"Shrinking {buffer_type} buffer from {current_size} to {new_size}")
                
                # Create smaller buffer
                if buffer_type == 'indices':
                    self.buffers[buffer_type] = torch.zeros((new_size, 2), dtype=torch.long, device=self.device)
                elif buffer_type in ['mask']:
                    self.buffers[buffer_type] = torch.zeros(new_size, dtype=torch.bool, device=self.device)
                else:
                    self.buffers[buffer_type] = torch.zeros(new_size, dtype=self.buffers[buffer_type].dtype, device=self.device)
                
                self.allocated_sizes[buffer_type] = new_size


class PooledSparseBridge:
    """Sparse bridge implementation using memory pooling"""
    
    def __init__(self, device: torch.device, threshold: float = 1e-6):
        self.device = device
        self.threshold = threshold
        self.pool = SparseMemoryPool(device)
    
    def encode_with_pool(self, padic_digits: torch.Tensor, valuations: torch.Tensor) -> torch.sparse.Tensor:
        """
        Encode sparse tensor using pooled memory
        
        Args:
            padic_digits: P-adic digits [batch_size, precision]
            valuations: Valuations [batch_size]
            
        Returns:
            Sparse CSR tensor
        """
        batch_size, precision = padic_digits.shape
        max_nnz = batch_size * precision
        
        # Get mask buffer
        mask_buffer = self.pool.get_buffer('mask', max_nnz)
        flat_digits = padic_digits.view(-1)
        
        # Create mask in buffer
        mask_view = mask_buffer[:max_nnz]
        torch.greater(torch.abs(flat_digits), self.threshold, out=mask_view)
        
        # Count non-zeros
        nnz = mask_view.sum().item()
        
        if nnz == 0:
            # Empty sparse tensor
            crow_indices = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            col_indices = torch.zeros(0, dtype=torch.long, device=self.device)
            values = torch.zeros(0, dtype=padic_digits.dtype, device=self.device)
        else:
            # Get sparse buffers
            buffers = self.pool.get_sparse_buffers(nnz, batch_size)
            
            # Extract non-zero indices and values
            flat_indices = torch.arange(max_nnz, device=self.device)[mask_view]
            row_indices = flat_indices // precision
            col_indices = flat_indices % precision
            
            # Copy to buffers
            buffers['row_indices'][:nnz] = row_indices
            buffers['col_indices'][:nnz] = col_indices
            buffers['values'][:nnz] = flat_digits[mask_view]
            
            # Create CSR indices
            row_counts = torch.bincount(row_indices, minlength=batch_size)
            buffers['crow_indices'][0] = 0
            torch.cumsum(row_counts, dim=0, out=buffers['crow_indices'][1:batch_size+1])
            
            # Sort for CSR
            sort_idx = torch.argsort(row_indices)
            sorted_col_indices = col_indices[sort_idx]
            sorted_values = buffers['values'][:nnz][sort_idx]
            
            # Use views for CSR tensor
            crow_indices = buffers['crow_indices'][:batch_size+1]
            col_indices = sorted_col_indices
            values = sorted_values
        
        # Create sparse tensor
        sparse_tensor = torch.sparse_csr_tensor(
            crow_indices=crow_indices,
            col_indices=col_indices,
            values=values,
            size=padic_digits.shape,
            dtype=padic_digits.dtype,
            device=self.device
        )
        
        # Attach valuations
        sparse_tensor.valuations = valuations
        
        return sparse_tensor