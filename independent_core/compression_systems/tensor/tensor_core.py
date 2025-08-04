"""
Tensor Decomposition Core Classes for Independent Core
NO FALLBACKS - HARD FAILURES ONLY

This module implements the core tensor decomposition compression system
with support for CP, Tucker, and HOSVD decompositions.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
from abc import ABC, abstractmethod
import math
import logging
from datetime import datetime
import warnings

# Import base compression class
from ..base.compression_base import CompressionAlgorithm as CompressionBase


class DecompositionType(Enum):
    """Supported tensor decomposition types"""
    CP = auto()  # CANDECOMP/PARAFAC
    TUCKER = auto()  # Tucker decomposition
    HOSVD = auto()  # Higher-Order SVD
    TT = auto()  # Tensor-Train (future)
    TR = auto()  # Tensor-Ring (future)


@dataclass
class TensorDecomposition:
    """Container for tensor decomposition results"""
    decomposition_type: DecompositionType
    factors: List[torch.Tensor]
    core: Optional[torch.Tensor] = None
    ranks: List[int] = field(default_factory=list)
    compression_ratio: float = 0.0
    reconstruction_error: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_memory_usage(self) -> int:
        """Calculate total memory usage of decomposition"""
        total_memory = 0
        for factor in self.factors:
            total_memory += factor.element_size() * factor.nelement()
        if self.core is not None:
            total_memory += self.core.element_size() * self.core.nelement()
        return total_memory
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert decomposition to dictionary for serialization"""
        return {
            'type': self.decomposition_type.name,
            'factors': [f.cpu().numpy().tolist() for f in self.factors],
            'core': self.core.cpu().numpy().tolist() if self.core is not None else None,
            'ranks': self.ranks,
            'compression_ratio': self.compression_ratio,
            'reconstruction_error': self.reconstruction_error,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


class TensorValidator:
    """Validation utilities for tensor operations"""
    
    @staticmethod
    def validate_tensor(tensor: torch.Tensor, name: str = "tensor") -> None:
        """Validate tensor input - NO FALLBACKS"""
        if tensor is None:
            raise ValueError(f"{name} cannot be None")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be torch.Tensor, got {type(tensor)}")
        if tensor.numel() == 0:
            raise ValueError(f"{name} cannot be empty")
        if torch.isnan(tensor).any():
            raise ValueError(f"{name} contains NaN values")
        if torch.isinf(tensor).any():
            raise ValueError(f"{name} contains infinite values")
    
    @staticmethod
    def validate_ranks(ranks: List[int], tensor_shape: torch.Size) -> None:
        """Validate decomposition ranks"""
        if not ranks:
            raise ValueError("Ranks list cannot be empty")
        if len(ranks) != len(tensor_shape):
            raise ValueError(f"Ranks length {len(ranks)} must match tensor dimensions {len(tensor_shape)}")
        for i, (rank, dim) in enumerate(zip(ranks, tensor_shape)):
            if rank <= 0:
                raise ValueError(f"Rank at dimension {i} must be positive, got {rank}")
            if rank > dim:
                raise ValueError(f"Rank {rank} at dimension {i} exceeds tensor dimension {dim}")
    
    @staticmethod
    def validate_compression_ratio(ratio: float) -> None:
        """Validate compression ratio"""
        if not isinstance(ratio, (int, float)):
            raise TypeError(f"Compression ratio must be numeric, got {type(ratio)}")
        if not 0 < ratio <= 1:
            raise ValueError(f"Compression ratio must be in (0, 1], got {ratio}")


class DecompositionMethod(ABC):
    """Abstract base class for tensor decomposition methods"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = self.config.get('dtype', torch.float32)
        self.max_iterations = self.config.get('max_iterations', 100)
        self.tolerance = self.config.get('tolerance', 1e-6)
    
    @abstractmethod
    def decompose(self, tensor: torch.Tensor, ranks: List[int]) -> TensorDecomposition:
        """Perform tensor decomposition - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def reconstruct(self, decomposition: TensorDecomposition) -> torch.Tensor:
        """Reconstruct tensor from decomposition - must be implemented by subclasses"""
        pass
    
    def compute_reconstruction_error(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Compute relative reconstruction error"""
        TensorValidator.validate_tensor(original, "original")
        TensorValidator.validate_tensor(reconstructed, "reconstructed")
        
        if original.shape != reconstructed.shape:
            raise ValueError(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")
        
        error = torch.norm(original - reconstructed) / torch.norm(original)
        return error.item()


class CPDecomposer(DecompositionMethod):
    """CANDECOMP/PARAFAC decomposition implementation"""
    
    def decompose(self, tensor: torch.Tensor, ranks: List[int]) -> TensorDecomposition:
        """Perform CP decomposition using ALS algorithm"""
        TensorValidator.validate_tensor(tensor, "tensor")
        
        # For CP, all ranks must be the same
        rank = ranks[0] if isinstance(ranks, list) else ranks
        if isinstance(ranks, list) and not all(r == rank for r in ranks):
            raise ValueError("CP decomposition requires uniform rank across all modes")
        
        ndim = tensor.ndim
        shape = tensor.shape
        device = tensor.device
        dtype = tensor.dtype
        
        # Initialize factor matrices randomly
        factors = []
        for dim_size in shape:
            factor = torch.randn(dim_size, rank, device=device, dtype=dtype)
            factor = torch.qr(factor)[0]  # Orthogonalize
            factors.append(factor)
        
        # ALS iterations
        for iteration in range(self.max_iterations):
            old_factors = [f.clone() for f in factors]
            
            # Update each factor matrix
            for n in range(ndim):
                # Compute Khatri-Rao product of all factors except n
                V = self._khatri_rao_product([factors[i] for i in range(ndim) if i != n])
                
                # Unfold tensor along mode n
                unfolded = self._unfold(tensor, n)
                
                # Update factor
                factors[n] = unfolded @ V @ torch.linalg.pinv(V.T @ V)
            
            # Check convergence
            if self._check_convergence(old_factors, factors):
                break
        
        # Compute compression ratio
        original_size = tensor.numel() * tensor.element_size()
        compressed_size = sum(f.numel() * f.element_size() for f in factors)
        compression_ratio = compressed_size / original_size
        
        # Compute reconstruction error
        reconstructed = self.reconstruct(TensorDecomposition(
            decomposition_type=DecompositionType.CP,
            factors=factors,
            ranks=[rank] * ndim
        ))
        reconstruction_error = self.compute_reconstruction_error(tensor, reconstructed)
        
        return TensorDecomposition(
            decomposition_type=DecompositionType.CP,
            factors=factors,
            ranks=[rank] * ndim,
            compression_ratio=compression_ratio,
            reconstruction_error=reconstruction_error,
            metadata={'iterations': iteration + 1}
        )
    
    def reconstruct(self, decomposition: TensorDecomposition) -> torch.Tensor:
        """Reconstruct tensor from CP decomposition"""
        if decomposition.decomposition_type != DecompositionType.CP:
            raise ValueError(f"Expected CP decomposition, got {decomposition.decomposition_type}")
        
        factors = decomposition.factors
        rank = decomposition.ranks[0]
        
        # Start with ones vector
        result = torch.ones(rank, device=factors[0].device, dtype=factors[0].dtype)
        
        # Reconstruct by outer products
        for factor in reversed(factors):
            result = torch.einsum('r,ir->ir', result, factor.T)
            result = result.reshape(-1, *[f.shape[0] for f in factors[len(factors)-factors.index(factor):]])
        
        return result.squeeze()
    
    def _unfold(self, tensor: torch.Tensor, mode: int) -> torch.Tensor:
        """Unfold tensor along specified mode"""
        shape = list(tensor.shape)
        n = shape.pop(mode)
        return tensor.permute(mode, *[i for i in range(len(shape)+1) if i != mode]).reshape(n, -1)
    
    def _khatri_rao_product(self, matrices: List[torch.Tensor]) -> torch.Tensor:
        """Compute Khatri-Rao product of matrices"""
        if not matrices:
            raise ValueError("Cannot compute Khatri-Rao product of empty list")
        
        result = matrices[0]
        for matrix in matrices[1:]:
            n_rows = result.shape[0] * matrix.shape[0]
            n_cols = result.shape[1]
            temp = torch.zeros(n_rows, n_cols, device=result.device, dtype=result.dtype)
            
            for j in range(n_cols):
                temp[:, j] = torch.kron(result[:, j], matrix[:, j])
            
            result = temp
        
        return result
    
    def _check_convergence(self, old_factors: List[torch.Tensor], new_factors: List[torch.Tensor]) -> bool:
        """Check if ALS has converged"""
        for old, new in zip(old_factors, new_factors):
            if torch.norm(old - new) / torch.norm(old) > self.tolerance:
                return False
        return True


class TuckerDecomposer(DecompositionMethod):
    """Tucker decomposition implementation"""
    
    def decompose(self, tensor: torch.Tensor, ranks: List[int]) -> TensorDecomposition:
        """Perform Tucker decomposition using HOSVD"""
        TensorValidator.validate_tensor(tensor, "tensor")
        TensorValidator.validate_ranks(ranks, tensor.shape)
        
        ndim = tensor.ndim
        factors = []
        
        # Compute factor matrices via SVD on each mode
        core = tensor.clone()
        for n in range(ndim):
            # Unfold tensor along mode n
            unfolded = self._unfold(core, n)
            
            # Compute SVD
            U, S, V = torch.svd(unfolded)
            
            # Truncate to rank
            U_truncated = U[:, :ranks[n]]
            factors.append(U_truncated)
            
            # Update core tensor
            core = self._mode_n_product(core, U_truncated.T, n)
        
        # Compute compression ratio
        original_size = tensor.numel() * tensor.element_size()
        compressed_size = core.numel() * core.element_size()
        compressed_size += sum(f.numel() * f.element_size() for f in factors)
        compression_ratio = compressed_size / original_size
        
        # Compute reconstruction error
        reconstructed = self.reconstruct(TensorDecomposition(
            decomposition_type=DecompositionType.TUCKER,
            factors=factors,
            core=core,
            ranks=ranks
        ))
        reconstruction_error = self.compute_reconstruction_error(tensor, reconstructed)
        
        return TensorDecomposition(
            decomposition_type=DecompositionType.TUCKER,
            factors=factors,
            core=core,
            ranks=ranks,
            compression_ratio=compression_ratio,
            reconstruction_error=reconstruction_error
        )
    
    def reconstruct(self, decomposition: TensorDecomposition) -> torch.Tensor:
        """Reconstruct tensor from Tucker decomposition"""
        if decomposition.decomposition_type != DecompositionType.TUCKER:
            raise ValueError(f"Expected Tucker decomposition, got {decomposition.decomposition_type}")
        
        if decomposition.core is None:
            raise ValueError("Tucker decomposition requires core tensor")
        
        result = decomposition.core
        for n, factor in enumerate(decomposition.factors):
            result = self._mode_n_product(result, factor, n)
        
        return result
    
    def _unfold(self, tensor: torch.Tensor, mode: int) -> torch.Tensor:
        """Unfold tensor along specified mode"""
        shape = list(tensor.shape)
        n = shape[mode]
        
        # Move mode to front
        perm = [mode] + [i for i in range(len(shape)) if i != mode]
        tensor_permuted = tensor.permute(perm)
        
        # Reshape
        return tensor_permuted.reshape(n, -1)
    
    def _mode_n_product(self, tensor: torch.Tensor, matrix: torch.Tensor, mode: int) -> torch.Tensor:
        """Compute mode-n product of tensor and matrix"""
        # Unfold tensor
        unfolded = self._unfold(tensor, mode)
        
        # Matrix multiply
        result_unfolded = matrix @ unfolded
        
        # Get result shape
        shape = list(tensor.shape)
        shape[mode] = matrix.shape[0]
        
        # Fold back
        perm = [mode] + [i for i in range(len(shape)) if i != mode]
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        
        return result_unfolded.reshape([shape[p] for p in perm]).permute(inv_perm)


class TensorTrainDecomposer(DecompositionMethod):
    """Tensor-Train decomposition implementation"""
    
    def decompose(self, tensor: torch.Tensor, ranks: List[int]) -> TensorDecomposition:
        """Perform Tensor-Train decomposition using SVD-based algorithm"""
        TensorValidator.validate_tensor(tensor, "tensor")
        
        ndim = tensor.ndim
        shape = tensor.shape
        
        # Validate TT ranks (boundary ranks must be 1)
        if len(ranks) != ndim - 1:
            raise ValueError(f"TT ranks length {len(ranks)} must be {ndim - 1}")
        
        # Initialize TT cores
        cores = []
        C = tensor.clone()
        
        # Left-to-right SVD decomposition
        left_rank = 1
        for k in range(ndim - 1):
            # Reshape for SVD
            C_reshaped = C.reshape(left_rank * shape[k], -1)
            
            # SVD
            U, S, V = torch.svd(C_reshaped)
            
            # Truncate to rank
            rank = min(ranks[k], U.shape[1], V.shape[0])
            U = U[:, :rank]
            S = S[:rank]
            V = V[:, :rank]
            
            # Form TT core
            core = U.reshape(left_rank, shape[k], rank)
            cores.append(core)
            
            # Update for next iteration
            C = torch.diag(S) @ V.T
            C = C.reshape(rank, *shape[k+1:])
            left_rank = rank
        
        # Last core
        cores.append(C)
        
        # Compute compression ratio
        original_size = tensor.numel() * tensor.element_size()
        compressed_size = sum(core.numel() * core.element_size() for core in cores)
        compression_ratio = compressed_size / original_size
        
        # Compute reconstruction error
        reconstructed = self.reconstruct(TensorDecomposition(
            decomposition_type=DecompositionType.TT,
            factors=cores,
            ranks=ranks
        ))
        reconstruction_error = self.compute_reconstruction_error(tensor, reconstructed)
        
        return TensorDecomposition(
            decomposition_type=DecompositionType.TT,
            factors=cores,
            ranks=ranks,
            compression_ratio=compression_ratio,
            reconstruction_error=reconstruction_error
        )
    
    def reconstruct(self, decomposition: TensorDecomposition) -> torch.Tensor:
        """Reconstruct tensor from TT decomposition"""
        if decomposition.decomposition_type != DecompositionType.TT:
            raise ValueError(f"Expected TT decomposition, got {decomposition.decomposition_type}")
        
        cores = decomposition.factors
        if not cores:
            raise ValueError("TT cores cannot be empty")
        
        # Contract TT cores
        result = cores[0]
        for core in cores[1:]:
            # Contract adjacent cores
            result = torch.tensordot(result, core, dims=([result.ndim-1], [0]))
        
        return result.squeeze()


class TensorRankOptimizer:
    """Optimize tensor ranks for target compression ratio"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_rank_search_iterations = self.config.get('max_rank_search_iterations', 20)
        self.rank_search_tolerance = self.config.get('rank_search_tolerance', 0.05)
    
    def optimize_ranks(self, tensor: torch.Tensor, target_compression_ratio: float,
                      decomposition_type: DecompositionType) -> List[int]:
        """Find optimal ranks for target compression ratio"""
        TensorValidator.validate_tensor(tensor, "tensor")
        TensorValidator.validate_compression_ratio(target_compression_ratio)
        
        shape = tensor.shape
        ndim = tensor.ndim
        
        # Start with conservative ranks
        if decomposition_type == DecompositionType.CP:
            # CP requires uniform rank
            max_rank = min(shape)
            min_rank = 1
            
            # Binary search for optimal rank
            for _ in range(self.max_rank_search_iterations):
                mid_rank = (min_rank + max_rank) // 2
                compression_ratio = self._estimate_cp_compression_ratio(shape, mid_rank)
                
                if abs(compression_ratio - target_compression_ratio) < self.rank_search_tolerance:
                    return [mid_rank] * ndim
                
                if compression_ratio > target_compression_ratio:
                    max_rank = mid_rank - 1
                else:
                    min_rank = mid_rank + 1
            
            return [mid_rank] * ndim
        
        elif decomposition_type == DecompositionType.TT:
            # TT ranks optimization
            max_rank = min(shape[:-1])
            tt_ranks = [max(1, int(max_rank * target_compression_ratio))] * (ndim - 1)
            
            # Iterative adjustment
            for _ in range(self.max_rank_search_iterations):
                compression_ratio = self._estimate_tt_compression_ratio(shape, tt_ranks)
                
                if abs(compression_ratio - target_compression_ratio) < self.rank_search_tolerance:
                    break
                
                # Adjust ranks
                adjustment = target_compression_ratio / compression_ratio
                for i in range(len(tt_ranks)):
                    tt_ranks[i] = max(1, int(tt_ranks[i] * adjustment))
            
            return tt_ranks
        
        else:  # Tucker/HOSVD
            # Start with proportional ranks
            total_params = tensor.numel()
            target_params = int(total_params * target_compression_ratio)
            
            # Initialize ranks proportionally
            ranks = []
            scale_factor = target_compression_ratio ** (1.0 / ndim)
            for dim in shape:
                rank = max(1, int(dim * scale_factor))
                ranks.append(rank)
            
            # Iteratively adjust ranks
            for _ in range(self.max_rank_search_iterations):
                compression_ratio = self._estimate_tucker_compression_ratio(shape, ranks)
                
                if abs(compression_ratio - target_compression_ratio) < self.rank_search_tolerance:
                    break
                
                # Adjust ranks
                adjustment = target_compression_ratio / compression_ratio
                for i in range(len(ranks)):
                    ranks[i] = max(1, min(shape[i], int(ranks[i] * adjustment)))
            
            return ranks
    
    def _estimate_cp_compression_ratio(self, shape: torch.Size, rank: int) -> float:
        """Estimate compression ratio for CP decomposition"""
        original_size = np.prod(shape)
        compressed_size = sum(dim * rank for dim in shape)
        return compressed_size / original_size
    
    def _estimate_tucker_compression_ratio(self, shape: torch.Size, ranks: List[int]) -> float:
        """Estimate compression ratio for Tucker decomposition"""
        original_size = np.prod(shape)
        core_size = np.prod(ranks)
        factors_size = sum(shape[i] * ranks[i] for i in range(len(shape)))
        compressed_size = core_size + factors_size
        return compressed_size / original_size
    
    def _estimate_tt_compression_ratio(self, shape: torch.Size, tt_ranks: List[int]) -> float:
        """Estimate compression ratio for TT decomposition"""
        original_size = np.prod(shape)
        compressed_size = 0
        
        # First core
        compressed_size += shape[0] * tt_ranks[0]
        
        # Middle cores
        for i in range(1, len(shape) - 1):
            compressed_size += tt_ranks[i-1] * shape[i] * tt_ranks[i]
        
        # Last core
        compressed_size += tt_ranks[-1] * shape[-1]
        
        return compressed_size / original_size


class TensorCompressionSystem(CompressionBase):
    """Main tensor decomposition compression system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Initialize components
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = self.config.get('dtype', torch.float32)
        self.default_method = self.config.get('default_method', DecompositionType.TUCKER)
        self.target_compression_ratio = self.config.get('target_compression_ratio', 0.1)
        
        # Initialize decomposition methods
        self.methods = {
            DecompositionType.CP: CPDecomposer(config),
            DecompositionType.TUCKER: TuckerDecomposer(config),
            DecompositionType.TT: TensorTrainDecomposer(config)
        }
        
        # Initialize rank optimizer
        self.rank_optimizer = TensorRankOptimizer(config)
        
        # GPU memory management
        self.gpu_memory_threshold = self.config.get('gpu_memory_threshold', 0.9)
        self.enable_memory_management = self.config.get('enable_memory_management', True)
        
        # Cache for decompositions
        self.decomposition_cache = {}
        self.cache_size_limit = self.config.get('cache_size_limit', 100)
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def compress(self, data: torch.Tensor) -> TensorDecomposition:
        """Compress tensor using specified decomposition method"""
        # NO FALLBACKS - HARD FAILURES ONLY
        TensorValidator.validate_tensor(data, "input tensor")
        
        # Check GPU memory if enabled
        if self.enable_memory_management and data.is_cuda:
            self._check_gpu_memory()
        
        # Get decomposition method
        method_type = self.config.get('method', self.default_method)
        if method_type not in self.methods:
            raise ValueError(f"Unsupported decomposition method: {method_type}")
        
        method = self.methods[method_type]
        
        # Get or optimize ranks
        ranks = self.config.get('ranks')
        if ranks is None:
            ranks = self.rank_optimizer.optimize_ranks(
                data, self.target_compression_ratio, method_type
            )
        else:
            if method_type != DecompositionType.TT:
                TensorValidator.validate_ranks(ranks, data.shape)
        
        # Perform decomposition
        start_time = torch.cuda.Event(enable_timing=True) if data.is_cuda else None
        end_time = torch.cuda.Event(enable_timing=True) if data.is_cuda else None
        
        if start_time:
            start_time.record()
        
        decomposition = method.decompose(data, ranks)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            decomposition.metadata['compression_time_ms'] = start_time.elapsed_time(end_time)
        
        # Update metrics
        self._update_compression_metrics(data, decomposition)
        
        # Cache if enabled
        if self.config.get('enable_caching', True):
            self._cache_decomposition(data, decomposition)
        
        return decomposition
    
    def decompress(self, compressed_data: TensorDecomposition) -> torch.Tensor:
        """Reconstruct tensor from decomposition"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if compressed_data is None:
            raise ValueError("Compressed data cannot be None")
        if not isinstance(compressed_data, TensorDecomposition):
            raise TypeError(f"Expected TensorDecomposition, got {type(compressed_data)}")
        
        # Get appropriate method
        method = self.methods.get(compressed_data.decomposition_type)
        if method is None:
            raise ValueError(f"No method found for {compressed_data.decomposition_type}")
        
        # Reconstruct tensor
        return method.reconstruct(compressed_data)
    
    def compress_batch(self, tensors: List[torch.Tensor]) -> List[TensorDecomposition]:
        """Compress multiple tensors efficiently"""
        if not tensors:
            raise ValueError("Tensor list cannot be empty")
        
        results = []
        for i, tensor in enumerate(tensors):
            try:
                result = self.compress(tensor)
                results.append(result)
            except Exception as e:
                raise RuntimeError(f"Failed to compress tensor {i}: {str(e)}")
        
        return results
    
    def decompress_batch(self, decompositions: List[TensorDecomposition]) -> List[torch.Tensor]:
        """Decompress multiple tensors efficiently"""
        if not decompositions:
            raise ValueError("Decomposition list cannot be empty")
        
        results = []
        for i, decomp in enumerate(decompositions):
            try:
                result = self.decompress(decomp)
                results.append(result)
            except Exception as e:
                raise RuntimeError(f"Failed to decompress tensor {i}: {str(e)}")
        
        return results
    
    def set_decomposition_method(self, method: DecompositionType) -> None:
        """Set the decomposition method to use"""
        if method not in self.methods:
            raise ValueError(f"Unsupported method: {method}")
        self.default_method = method
        self.config['method'] = method
    
    def set_target_compression_ratio(self, ratio: float) -> None:
        """Set target compression ratio"""
        TensorValidator.validate_compression_ratio(ratio)
        self.target_compression_ratio = ratio
        self.config['target_compression_ratio'] = ratio
    
    def get_supported_methods(self) -> List[str]:
        """Get list of supported decomposition methods"""
        return [method.name for method in self.methods.keys()]
    
    def _check_gpu_memory(self) -> None:
        """Check GPU memory usage and raise if threshold exceeded"""
        if not torch.cuda.is_available():
            return
        
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        usage_ratio = allocated / total
        
        if usage_ratio > self.gpu_memory_threshold:
            raise RuntimeError(f"GPU memory usage {usage_ratio:.2%} exceeds threshold {self.gpu_memory_threshold:.2%}")
    
    def _update_compression_metrics(self, original: torch.Tensor, decomposition: TensorDecomposition) -> None:
        """Update compression metrics"""
        metrics = {
            'original_size': original.numel() * original.element_size(),
            'compressed_size': decomposition.get_memory_usage(),
            'compression_ratio': decomposition.compression_ratio,
            'reconstruction_error': decomposition.reconstruction_error,
            'method': decomposition.decomposition_type.name,
            'ranks': decomposition.ranks,
            'timestamp': decomposition.timestamp
        }
        
        # Store in history
        if 'history' not in self.compression_metrics:
            self.compression_metrics['history'] = []
        self.compression_metrics['history'].append(metrics)
        
        # Update summary statistics
        self.compression_metrics['total_compressions'] = len(self.compression_metrics['history'])
        self.compression_metrics['average_compression_ratio'] = np.mean([m['compression_ratio'] for m in self.compression_metrics['history']])
        self.compression_metrics['average_reconstruction_error'] = np.mean([m['reconstruction_error'] for m in self.compression_metrics['history']])
    
    def _cache_decomposition(self, tensor: torch.Tensor, decomposition: TensorDecomposition) -> None:
        """Cache decomposition for reuse"""
        # Generate cache key from tensor properties
        cache_key = (tensor.shape, tensor.dtype, tensor.device, id(tensor))
        
        # Limit cache size
        if len(self.decomposition_cache) >= self.cache_size_limit:
            # Remove oldest entry
            oldest_key = next(iter(self.decomposition_cache))
            del self.decomposition_cache[oldest_key]
        
        self.decomposition_cache[cache_key] = decomposition
    
    def clear_cache(self) -> None:
        """Clear decomposition cache"""
        self.decomposition_cache.clear()
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get comprehensive compression statistics"""
        return {
            'metrics': self.compression_metrics,
            'cache_size': len(self.decomposition_cache),
            'supported_methods': self.get_supported_methods(),
            'current_method': self.default_method.name,
            'target_compression_ratio': self.target_compression_ratio
        }


# Factory function for creating decomposition methods
def create_decomposition_method(method_type: DecompositionType, config: Dict[str, Any] = None) -> DecompositionMethod:
    """Factory function to create decomposition methods"""
    methods = {
        DecompositionType.CP: CPDecomposer,
        DecompositionType.TUCKER: TuckerDecomposer,
        DecompositionType.TT: TensorTrainDecomposer
    }
    
    if method_type not in methods:
        raise ValueError(f"Unsupported decomposition method: {method_type}")
    
    return methods[method_type](config)


# Integration hooks for existing systems
class TensorDecompositionIntegration:
    """Integration utilities for existing systems"""
    
    @staticmethod
    def register_with_brain_core(brain_core: Any, system: TensorCompressionSystem) -> None:
        """Register tensor decomposition system with BrainCore"""
        brain_core.register_compression_system('tensor_decomposition', system)
        brain_core.tensor_decompositions = system
    
    @staticmethod
    def integrate_with_training_manager(training_manager: Any, system: TensorCompressionSystem) -> None:
        """Integrate with TrainingManager tensor operations"""
        # Add tensor operations
        training_manager.tensor_operations['cp_decomposition'] = lambda t: system.compress(t)
        training_manager.tensor_operations['tucker_decomposition'] = lambda t: system.compress(t)
        training_manager.tensor_operations['tt_decomposition'] = lambda t: system.compress(t)
        training_manager.tensor_operations['tensor_reconstruct'] = lambda d: system.decompress(d)
    
    @staticmethod
    def integrate_with_gradient_compression(gradient_component: Any, system: TensorCompressionSystem) -> None:
        """Integrate with GradientCompressionComponent"""
        # Add decomposition methods
        gradient_component.tensor_decomposition_methods['cp'] = system.methods[DecompositionType.CP]
        gradient_component.tensor_decomposition_methods['tucker'] = system.methods[DecompositionType.TUCKER]
        gradient_component.tensor_decomposition_methods['tt'] = system.methods[DecompositionType.TT]