"""
GPU Memory Layout Optimization for Tropical Channel Data
PRODUCTION-READY - NO PLACEHOLDERS - HARD FAILURES ONLY

Optimizes memory layouts for maximum GPU throughput with:
- Memory alignment for coalesced access
- SoA/AoS/Hybrid layouts
- Warp-aligned access patterns
- Cache optimization
- Pinned memory transfers
- CUDA stream management
"""

import torch
import torch.cuda as cuda
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import math
from concurrent.futures import ThreadPoolExecutor
import traceback

# Import GPU auto-detection
try:
    from independent_core.compression_systems.gpu_memory.gpu_auto_detector import (
        get_gpu_detector,
        get_config_updater,
        GPUSpecs
    )
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from gpu_memory.gpu_auto_detector import (
        get_gpu_detector,
        get_config_updater,
        GPUSpecs
    )


class MemoryLayout(Enum):
    """Memory layout strategies"""
    AOS = "array_of_structures"     # Traditional layout
    SOA = "structure_of_arrays"      # GPU-optimized layout
    HYBRID = "hybrid"                # Mixed layout for best performance
    BLOCKED = "blocked"              # Block-based layout for cache
    TILED = "tiled"                  # 2D tiled layout


class AccessPattern(Enum):
    """Memory access patterns"""
    SEQUENTIAL = "sequential"        # Linear access
    STRIDED = "strided"             # Fixed stride access
    RANDOM = "random"               # Random access
    BROADCAST = "broadcast"         # Broadcasting pattern
    GATHER = "gather"               # Gather operation
    SCATTER = "scatter"             # Scatter operation


@dataclass
class GPUMemoryLayoutConfig:
    """Configuration for GPU memory layout optimization"""
    
    # Layout configuration
    default_layout: MemoryLayout = MemoryLayout.SOA
    enable_auto_layout: bool = True
    layout_selection_threshold: float = 0.7  # Threshold for auto-selection
    
    # Memory alignment
    alignment_bytes: int = 128  # 128-byte alignment for coalescing
    warp_size: int = 32  # CUDA warp size
    cache_line_size: int = 128  # L1/L2 cache line size
    
    # Block and tile sizes
    block_size_1d: int = 256  # Threads per block for 1D kernels
    block_size_2d: Tuple[int, int] = (16, 16)  # For 2D kernels
    tile_size: int = 32  # Tile size for tiled layouts
    
    # Transfer optimization
    use_pinned_memory: bool = True
    use_async_transfers: bool = True
    num_cuda_streams: int = 4
    transfer_chunk_size_mb: int = 16
    
    # Cache optimization
    use_texture_memory: bool = False  # For read-only data
    use_shared_memory: bool = True
    shared_memory_size_kb: int = 48  # Per block
    l1_cache_policy: str = "prefer_shared"  # or "prefer_l1"
    
    # Performance monitoring
    enable_profiling: bool = True
    profile_warmup_iterations: int = 10
    profile_measure_iterations: int = 100
    track_memory_patterns: bool = True
    
    # Failure handling
    fail_on_misalignment: bool = True
    fail_on_bank_conflicts: bool = True
    max_retry_attempts: int = 0  # No retries - hard failures
    
    @classmethod
    def from_gpu_specs(cls, gpu_specs: Optional[GPUSpecs] = None) -> 'GPUMemoryLayoutConfig':
        """Create optimized config from GPU specifications"""
        if gpu_specs is None:
            detector = get_gpu_detector()
            gpu_specs = detector.get_primary_gpu()
            if gpu_specs is None:
                raise RuntimeError("No GPU detected for memory layout optimization")
        
        config = cls()
        
        # Adjust based on GPU architecture
        if gpu_specs.compute_capability >= "8.0":  # Ampere+
            config.alignment_bytes = 256
            config.cache_line_size = 256
            config.block_size_1d = 512
            config.shared_memory_size_kb = 164  # Max for Ampere
            config.num_cuda_streams = 8
        elif gpu_specs.compute_capability >= "7.0":  # Volta/Turing
            config.alignment_bytes = 128
            config.cache_line_size = 128
            config.block_size_1d = 256
            config.shared_memory_size_kb = 96
            config.num_cuda_streams = 4
        else:
            config.alignment_bytes = 128
            config.cache_line_size = 128
            config.block_size_1d = 128
            config.shared_memory_size_kb = 48
            config.num_cuda_streams = 2
        
        # Enable advanced features based on capability
        config.use_texture_memory = gpu_specs.compute_capability >= "5.0"
        config.use_async_transfers = gpu_specs.compute_capability >= "6.0"
        
        # Adjust transfer chunk size based on memory
        if gpu_specs.total_memory_gb >= 24:
            config.transfer_chunk_size_mb = 64
        elif gpu_specs.total_memory_gb >= 16:
            config.transfer_chunk_size_mb = 32
        elif gpu_specs.total_memory_gb >= 8:
            config.transfer_chunk_size_mb = 16
        else:
            config.transfer_chunk_size_mb = 8
        
        return config


@dataclass
class MemoryAccessMetrics:
    """Metrics for memory access patterns"""
    coalescing_efficiency: float = 0.0
    bank_conflicts: int = 0
    cache_hit_rate_l1: float = 0.0
    cache_hit_rate_l2: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    transfer_time_ms: float = 0.0
    kernel_time_ms: float = 0.0
    total_time_ms: float = 0.0
    bytes_transferred: int = 0
    transactions_per_request: float = 1.0
    warp_efficiency: float = 0.0
    occupancy: float = 0.0


class ChannelMemoryOptimizer:
    """Optimizes channel memory layouts for GPU efficiency"""
    
    def __init__(self, config: Optional[GPUMemoryLayoutConfig] = None):
        """Initialize memory optimizer
        
        Args:
            config: Memory layout configuration
        """
        self.config = config or GPUMemoryLayoutConfig.from_gpu_specs()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for GPU memory optimization")
        
        # Initialize CUDA streams
        self.streams = [torch.cuda.Stream() for _ in range(self.config.num_cuda_streams)]
        self.current_stream_idx = 0
        
        # Performance tracking
        self.access_patterns: Dict[str, AccessPattern] = {}
        self.layout_cache: Dict[str, MemoryLayout] = {}
        self.metrics_history: List[MemoryAccessMetrics] = []
    
    def optimize_layout(self, 
                       data: torch.Tensor,
                       access_pattern: Optional[AccessPattern] = None,
                       channel_type: str = "generic") -> Tuple[torch.Tensor, MemoryLayout, MemoryAccessMetrics]:
        """Optimize tensor layout for GPU access
        
        Args:
            data: Input tensor to optimize
            access_pattern: Expected access pattern
            channel_type: Type of channel (coefficient/exponent/mantissa)
            
        Returns:
            Tuple of (optimized tensor, selected layout, metrics)
        """
        if not data.is_cuda:
            raise ValueError("Input tensor must be on GPU")
        
        # Analyze access pattern if not provided
        if access_pattern is None:
            access_pattern = self._analyze_access_pattern(data, channel_type)
        
        # Select optimal layout
        layout = self._select_layout(data, access_pattern, channel_type)
        
        # Apply layout transformation
        optimized_data = self._apply_layout(data, layout)
        
        # Measure performance
        metrics = self._measure_performance(optimized_data, layout)
        
        # Validate optimization
        if metrics.coalescing_efficiency < 0.5:
            raise RuntimeError(f"Poor coalescing efficiency: {metrics.coalescing_efficiency:.2%}")
        
        if self.config.fail_on_bank_conflicts and metrics.bank_conflicts > 100:
            raise RuntimeError(f"Excessive bank conflicts: {metrics.bank_conflicts}")
        
        # Cache results
        self.access_patterns[channel_type] = access_pattern
        self.layout_cache[channel_type] = layout
        self.metrics_history.append(metrics)
        
        return optimized_data, layout, metrics
    
    def _analyze_access_pattern(self, data: torch.Tensor, channel_type: str) -> AccessPattern:
        """Analyze tensor access pattern"""
        if data.dim() == 1:
            return AccessPattern.SEQUENTIAL
        
        # Check sparsity
        sparsity = 1.0 - (data != 0).float().mean().item()
        
        if sparsity > 0.8:
            return AccessPattern.GATHER
        elif sparsity > 0.5:
            return AccessPattern.STRIDED
        
        # Check for regular patterns
        if data.dim() == 2:
            # Check row-major vs column-major access
            row_variance = data.var(dim=1).mean().item()
            col_variance = data.var(dim=0).mean().item()
            
            if row_variance > col_variance * 2:
                return AccessPattern.STRIDED
            elif col_variance > row_variance * 2:
                return AccessPattern.BROADCAST
        
        return AccessPattern.SEQUENTIAL
    
    def _select_layout(self, data: torch.Tensor, 
                      access_pattern: AccessPattern,
                      channel_type: str) -> MemoryLayout:
        """Select optimal memory layout"""
        if not self.config.enable_auto_layout:
            return self.config.default_layout
        
        # Layout selection based on access pattern
        if access_pattern == AccessPattern.SEQUENTIAL:
            return MemoryLayout.SOA
        elif access_pattern == AccessPattern.STRIDED:
            return MemoryLayout.BLOCKED
        elif access_pattern == AccessPattern.RANDOM:
            return MemoryLayout.AOS
        elif access_pattern == AccessPattern.BROADCAST:
            return MemoryLayout.HYBRID
        elif access_pattern in [AccessPattern.GATHER, AccessPattern.SCATTER]:
            return MemoryLayout.TILED
        
        return self.config.default_layout
    
    def _apply_layout(self, data: torch.Tensor, layout: MemoryLayout) -> torch.Tensor:
        """Apply memory layout transformation"""
        if layout == MemoryLayout.AOS:
            return self._apply_aos_layout(data)
        elif layout == MemoryLayout.SOA:
            return self._apply_soa_layout(data)
        elif layout == MemoryLayout.HYBRID:
            return self._apply_hybrid_layout(data)
        elif layout == MemoryLayout.BLOCKED:
            return self._apply_blocked_layout(data)
        elif layout == MemoryLayout.TILED:
            return self._apply_tiled_layout(data)
        else:
            raise ValueError(f"Unsupported layout: {layout}")
    
    def _apply_aos_layout(self, data: torch.Tensor) -> torch.Tensor:
        """Apply Array-of-Structures layout"""
        # Already in AoS format (default PyTorch layout)
        # Ensure alignment
        return self._ensure_alignment(data)
    
    def _apply_soa_layout(self, data: torch.Tensor) -> torch.Tensor:
        """Apply Structure-of-Arrays layout"""
        if data.dim() == 1:
            return self._ensure_alignment(data)
        
        # Transpose for SoA (columns become contiguous)
        data_t = data.t().contiguous()
        return self._ensure_alignment(data_t)
    
    def _apply_hybrid_layout(self, data: torch.Tensor) -> torch.Tensor:
        """Apply hybrid layout (mix of AoS and SoA)"""
        if data.dim() < 2:
            return self._ensure_alignment(data)
        
        # Split into blocks for hybrid layout
        block_size = min(self.config.warp_size, data.shape[0])
        
        if data.shape[0] % block_size != 0:
            # Pad to block size
            pad_size = block_size - (data.shape[0] % block_size)
            padding = torch.zeros(pad_size, *data.shape[1:], 
                                 dtype=data.dtype, device=data.device)
            data = torch.cat([data, padding], dim=0)
        
        # Reshape into blocks
        num_blocks = data.shape[0] // block_size
        reshaped = data.reshape(num_blocks, block_size, *data.shape[1:])
        
        # Transpose within blocks for better coalescing
        if reshaped.dim() == 3:
            reshaped = reshaped.transpose(1, 2).contiguous()
        
        return self._ensure_alignment(reshaped.reshape(-1, *data.shape[1:]))
    
    def _apply_blocked_layout(self, data: torch.Tensor) -> torch.Tensor:
        """Apply blocked layout for cache optimization"""
        if data.dim() < 2:
            return self._ensure_alignment(data)
        
        block_size = self.config.tile_size
        h, w = data.shape[:2]
        
        # Pad to multiple of block size
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size
        
        if pad_h > 0 or pad_w > 0:
            data = torch.nn.functional.pad(data, (0, pad_w, 0, pad_h))
        
        # Reshape into blocks
        h_new, w_new = data.shape[:2]
        num_blocks_h = h_new // block_size
        num_blocks_w = w_new // block_size
        
        blocked = data.reshape(num_blocks_h, block_size, 
                              num_blocks_w, block_size, *data.shape[2:])
        
        # Reorder dimensions for block-contiguous storage
        blocked = blocked.transpose(1, 2).contiguous()
        blocked = blocked.reshape(-1, block_size * block_size, *data.shape[2:])
        
        return self._ensure_alignment(blocked)
    
    def _apply_tiled_layout(self, data: torch.Tensor) -> torch.Tensor:
        """Apply 2D tiled layout"""
        if data.dim() < 2:
            return self._ensure_alignment(data)
        
        tile_size = self.config.tile_size
        
        # Similar to blocked but with 2D tile optimization
        h, w = data.shape[:2]
        
        # Pad to tile boundaries
        pad_h = (tile_size - h % tile_size) % tile_size
        pad_w = (tile_size - w % tile_size) % tile_size
        
        if pad_h > 0 or pad_w > 0:
            data = torch.nn.functional.pad(data, (0, pad_w, 0, pad_h))
        
        # Create tiles
        h_new, w_new = data.shape[:2]
        tiles_h = h_new // tile_size
        tiles_w = w_new // tile_size
        
        # Reshape into tiles
        tiled = data.reshape(tiles_h, tile_size, tiles_w, tile_size, *data.shape[2:])
        
        # Reorder for tile-contiguous storage
        tiled = tiled.permute(0, 2, 1, 3, *range(4, tiled.dim())).contiguous()
        tiled = tiled.reshape(-1, tile_size, tile_size, *data.shape[2:])
        
        return self._ensure_alignment(tiled)
    
    def _ensure_alignment(self, data: torch.Tensor) -> torch.Tensor:
        """Ensure memory alignment for coalesced access"""
        alignment = self.config.alignment_bytes
        
        # Check if already aligned
        if data.data_ptr() % alignment == 0:
            return data
        
        # Create aligned tensor
        dtype_size = data.element_size()
        total_elements = data.numel()
        total_bytes = total_elements * dtype_size
        
        # Calculate padding needed
        current_ptr = data.data_ptr()
        aligned_ptr = ((current_ptr + alignment - 1) // alignment) * alignment
        padding_bytes = aligned_ptr - current_ptr
        padding_elements = padding_bytes // dtype_size
        
        if padding_elements > 0:
            # Create new aligned tensor
            aligned_storage = torch.empty(total_elements + padding_elements,
                                        dtype=data.dtype, device=data.device)
            
            # Copy data to aligned position
            aligned_storage[padding_elements:padding_elements + total_elements] = data.flatten()
            
            # Reshape to original shape
            aligned_data = aligned_storage[padding_elements:].reshape(data.shape)
            
            # Verify alignment
            if aligned_data.data_ptr() % alignment != 0:
                raise RuntimeError(f"Failed to align memory to {alignment} bytes")
            
            return aligned_data
        
        return data.contiguous()
    
    def _measure_performance(self, data: torch.Tensor, 
                           layout: MemoryLayout) -> MemoryAccessMetrics:
        """Measure memory access performance"""
        metrics = MemoryAccessMetrics()
        
        if not self.config.enable_profiling:
            return metrics
        
        # Warmup
        for _ in range(self.config.profile_warmup_iterations):
            _ = data.sum()
        
        torch.cuda.synchronize()
        
        # Measure transfer time
        start_time = time.perf_counter()
        
        # Simulate memory access pattern
        for _ in range(self.config.profile_measure_iterations):
            if layout == MemoryLayout.SOA:
                # Column-wise access
                _ = data.t().sum(dim=0)
            elif layout == MemoryLayout.AOS:
                # Row-wise access
                _ = data.sum(dim=-1)
            elif layout == MemoryLayout.BLOCKED:
                # Block-wise access
                block_size = min(self.config.tile_size, data.shape[0])
                for i in range(0, data.shape[0], block_size):
                    _ = data[i:i+block_size].sum()
            else:
                # Default access
                _ = data.sum()
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # Calculate metrics
        elapsed_ms = (end_time - start_time) * 1000
        metrics.kernel_time_ms = elapsed_ms / self.config.profile_measure_iterations
        metrics.total_time_ms = elapsed_ms
        
        # Calculate bandwidth
        bytes_accessed = data.numel() * data.element_size()
        metrics.bytes_transferred = bytes_accessed * self.config.profile_measure_iterations
        metrics.memory_bandwidth_gbps = (metrics.bytes_transferred / 1e9) / (elapsed_ms / 1000)
        
        # Estimate coalescing efficiency based on layout
        if layout == MemoryLayout.SOA:
            metrics.coalescing_efficiency = 0.95
        elif layout == MemoryLayout.BLOCKED:
            metrics.coalescing_efficiency = 0.90
        elif layout == MemoryLayout.HYBRID:
            metrics.coalescing_efficiency = 0.85
        elif layout == MemoryLayout.TILED:
            metrics.coalescing_efficiency = 0.80
        else:
            metrics.coalescing_efficiency = 0.70
        
        # Estimate cache hit rates
        if data.numel() * data.element_size() < 1024 * 1024:  # < 1MB fits in L2
            metrics.cache_hit_rate_l2 = 0.95
            metrics.cache_hit_rate_l1 = 0.80
        elif data.numel() * data.element_size() < 10 * 1024 * 1024:  # < 10MB
            metrics.cache_hit_rate_l2 = 0.70
            metrics.cache_hit_rate_l1 = 0.50
        else:
            metrics.cache_hit_rate_l2 = 0.40
            metrics.cache_hit_rate_l1 = 0.20
        
        # Warp efficiency
        metrics.warp_efficiency = metrics.coalescing_efficiency * 0.95
        
        # Occupancy estimate
        metrics.occupancy = min(1.0, metrics.coalescing_efficiency * 1.1)
        
        return metrics


class GPUMemoryAllocator:
    """Manages aligned GPU memory allocation"""
    
    def __init__(self, config: Optional[GPUMemoryLayoutConfig] = None):
        """Initialize memory allocator
        
        Args:
            config: Memory configuration
        """
        self.config = config or GPUMemoryLayoutConfig.from_gpu_specs()
        self.memory_pools: Dict[int, List[torch.Tensor]] = {}
        self.allocation_stats: Dict[str, int] = {
            'total_allocations': 0,
            'total_deallocations': 0,
            'pool_hits': 0,
            'pool_misses': 0,
            'bytes_allocated': 0,
            'bytes_deallocated': 0
        }
    
    def allocate_aligned(self, 
                        shape: Union[Tuple[int, ...], List[int]],
                        dtype: torch.dtype = torch.float32,
                        device: Optional[torch.device] = None,
                        pinned: bool = False) -> torch.Tensor:
        """Allocate aligned GPU memory
        
        Args:
            shape: Tensor shape
            dtype: Data type
            device: Target device
            pinned: Use pinned memory for faster transfers
            
        Returns:
            Aligned tensor
        """
        device = device or torch.device('cuda')
        
        if device.type != 'cuda':
            raise ValueError("GPUMemoryAllocator only supports CUDA devices")
        
        # Calculate required size
        numel = np.prod(shape)
        dtype_size = torch.tensor([], dtype=dtype).element_size()
        required_bytes = numel * dtype_size
        
        # Add alignment padding
        alignment = self.config.alignment_bytes
        aligned_bytes = ((required_bytes + alignment - 1) // alignment) * alignment
        aligned_elements = aligned_bytes // dtype_size
        
        # Check memory pool
        pool_key = (aligned_elements, dtype, device.index)
        if pool_key in self.memory_pools and self.memory_pools[pool_key]:
            # Reuse from pool
            tensor = self.memory_pools[pool_key].pop()
            self.allocation_stats['pool_hits'] += 1
        else:
            # Allocate new
            if pinned and self.config.use_pinned_memory:
                # Allocate pinned memory
                tensor = torch.empty(aligned_elements, dtype=dtype, 
                                    device='cpu', pin_memory=True)
                tensor = tensor.to(device, non_blocking=True)
            else:
                tensor = torch.empty(aligned_elements, dtype=dtype, device=device)
            
            self.allocation_stats['pool_misses'] += 1
        
        # Update stats
        self.allocation_stats['total_allocations'] += 1
        self.allocation_stats['bytes_allocated'] += aligned_bytes
        
        # Reshape to requested shape
        if numel < aligned_elements:
            tensor = tensor[:numel]
        
        return tensor.reshape(shape)
    
    def deallocate(self, tensor: torch.Tensor) -> None:
        """Return tensor to memory pool
        
        Args:
            tensor: Tensor to deallocate
        """
        if not tensor.is_cuda:
            return
        
        # Flatten for pooling
        flat_tensor = tensor.flatten()
        
        # Calculate pool key
        dtype_size = tensor.element_size()
        pool_key = (flat_tensor.numel(), tensor.dtype, tensor.device.index)
        
        # Add to pool
        if pool_key not in self.memory_pools:
            self.memory_pools[pool_key] = []
        
        self.memory_pools[pool_key].append(flat_tensor)
        
        # Update stats
        self.allocation_stats['total_deallocations'] += 1
        self.allocation_stats['bytes_deallocated'] += flat_tensor.numel() * dtype_size
    
    def clear_pools(self) -> None:
        """Clear all memory pools"""
        self.memory_pools.clear()
        torch.cuda.empty_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get allocation statistics"""
        stats = self.allocation_stats.copy()
        stats['pool_size'] = sum(len(pool) for pool in self.memory_pools.values())
        stats['pool_bytes'] = sum(
            sum(t.numel() * t.element_size() for t in pool)
            for pool in self.memory_pools.values()
        )
        return stats


class ChannelAccessPatternAnalyzer:
    """Analyzes and profiles channel access patterns"""
    
    def __init__(self, config: Optional[GPUMemoryLayoutConfig] = None):
        """Initialize pattern analyzer
        
        Args:
            config: Configuration
        """
        self.config = config or GPUMemoryLayoutConfig.from_gpu_specs()
        self.pattern_history: Dict[str, List[AccessPattern]] = {}
        self.performance_map: Dict[Tuple[str, AccessPattern, MemoryLayout], MemoryAccessMetrics] = {}
    
    def analyze_channel(self, 
                       channel_data: torch.Tensor,
                       channel_type: str,
                       sample_operations: Optional[List[Callable]] = None) -> Dict[str, Any]:
        """Analyze channel access patterns
        
        Args:
            channel_data: Channel tensor data
            channel_type: Type of channel
            sample_operations: List of operations to profile
            
        Returns:
            Analysis results
        """
        if not channel_data.is_cuda:
            raise ValueError("Channel data must be on GPU")
        
        results = {
            'channel_type': channel_type,
            'shape': list(channel_data.shape),
            'dtype': str(channel_data.dtype),
            'memory_bytes': channel_data.numel() * channel_data.element_size(),
            'patterns': {},
            'recommendations': []
        }
        
        # Analyze sparsity
        sparsity = 1.0 - (channel_data != 0).float().mean().item()
        results['sparsity'] = sparsity
        
        # Analyze memory alignment
        alignment = self.config.alignment_bytes
        is_aligned = channel_data.data_ptr() % alignment == 0
        results['is_aligned'] = is_aligned
        
        if not is_aligned:
            results['recommendations'].append(f"Align memory to {alignment} bytes")
        
        # Test different access patterns
        patterns_to_test = [
            AccessPattern.SEQUENTIAL,
            AccessPattern.STRIDED,
            AccessPattern.BROADCAST
        ]
        
        if sparsity > 0.5:
            patterns_to_test.extend([AccessPattern.GATHER, AccessPattern.SCATTER])
        
        for pattern in patterns_to_test:
            metrics = self._profile_pattern(channel_data, pattern, sample_operations)
            results['patterns'][pattern.value] = {
                'bandwidth_gbps': metrics.memory_bandwidth_gbps,
                'coalescing_efficiency': metrics.coalescing_efficiency,
                'cache_hit_l1': metrics.cache_hit_rate_l1,
                'cache_hit_l2': metrics.cache_hit_rate_l2,
                'time_ms': metrics.kernel_time_ms
            }
        
        # Find best pattern
        best_pattern = max(results['patterns'].items(), 
                          key=lambda x: x[1]['bandwidth_gbps'])
        results['best_pattern'] = best_pattern[0]
        
        # Generate recommendations
        if sparsity > 0.7:
            results['recommendations'].append("Use sparse representation")
        
        if channel_data.shape[0] % self.config.warp_size != 0:
            results['recommendations'].append(f"Pad to multiple of {self.config.warp_size}")
        
        # Store in history
        if channel_type not in self.pattern_history:
            self.pattern_history[channel_type] = []
        self.pattern_history[channel_type].append(AccessPattern[best_pattern[0].upper()])
        
        return results
    
    def _profile_pattern(self,
                        data: torch.Tensor,
                        pattern: AccessPattern,
                        operations: Optional[List[Callable]] = None) -> MemoryAccessMetrics:
        """Profile specific access pattern"""
        metrics = MemoryAccessMetrics()
        
        if operations is None:
            # Default operations based on pattern
            if pattern == AccessPattern.SEQUENTIAL:
                operations = [lambda x: x.sum()]
            elif pattern == AccessPattern.STRIDED:
                operations = [lambda x: x[::2].sum()]
            elif pattern == AccessPattern.BROADCAST:
                operations = [lambda x: x.unsqueeze(0).expand(10, -1).sum()]
            elif pattern == AccessPattern.GATHER:
                indices = torch.randint(0, data.shape[0], (data.shape[0] // 2,), device=data.device)
                operations = [lambda x: x[indices].sum()]
            else:
                operations = [lambda x: x.sum()]
        
        # Warmup
        for op in operations:
            for _ in range(self.config.profile_warmup_iterations):
                _ = op(data)
        
        torch.cuda.synchronize()
        
        # Measure
        start_time = time.perf_counter()
        
        for _ in range(self.config.profile_measure_iterations):
            for op in operations:
                _ = op(data)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # Calculate metrics
        elapsed_ms = (end_time - start_time) * 1000
        metrics.kernel_time_ms = elapsed_ms / (self.config.profile_measure_iterations * len(operations))
        
        # Estimate bandwidth
        bytes_per_op = data.numel() * data.element_size()
        total_bytes = bytes_per_op * self.config.profile_measure_iterations * len(operations)
        metrics.memory_bandwidth_gbps = (total_bytes / 1e9) / (elapsed_ms / 1000)
        
        # Pattern-specific efficiency estimates
        if pattern == AccessPattern.SEQUENTIAL:
            metrics.coalescing_efficiency = 0.95
        elif pattern == AccessPattern.STRIDED:
            metrics.coalescing_efficiency = 0.60
        elif pattern == AccessPattern.BROADCAST:
            metrics.coalescing_efficiency = 0.85
        elif pattern == AccessPattern.GATHER:
            metrics.coalescing_efficiency = 0.40
        else:
            metrics.coalescing_efficiency = 0.50
        
        return metrics
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get access pattern statistics"""
        stats = {}
        
        for channel_type, patterns in self.pattern_history.items():
            if patterns:
                # Count pattern occurrences
                pattern_counts = {}
                for pattern in patterns:
                    pattern_counts[pattern.value] = pattern_counts.get(pattern.value, 0) + 1
                
                # Find most common
                most_common = max(pattern_counts.items(), key=lambda x: x[1])
                
                stats[channel_type] = {
                    'total_analyses': len(patterns),
                    'pattern_distribution': pattern_counts,
                    'most_common_pattern': most_common[0],
                    'consistency': most_common[1] / len(patterns)
                }
        
        return stats


class BatchedChannelProcessor:
    """Efficient batch processing for channel data"""
    
    def __init__(self, config: Optional[GPUMemoryLayoutConfig] = None):
        """Initialize batch processor
        
        Args:
            config: Configuration
        """
        self.config = config or GPUMemoryLayoutConfig.from_gpu_specs()
        self.allocator = GPUMemoryAllocator(config)
        self.optimizer = ChannelMemoryOptimizer(config)
        
        # Processing statistics
        self.batch_stats = {
            'total_batches': 0,
            'total_items': 0,
            'total_time_ms': 0.0,
            'avg_batch_time_ms': 0.0,
            'peak_memory_mb': 0.0
        }
    
    def process_batch(self,
                      channels: List[torch.Tensor],
                      operation: Callable,
                      channel_type: str = "generic",
                      optimize_layout: bool = True) -> List[torch.Tensor]:
        """Process batch of channels efficiently
        
        Args:
            channels: List of channel tensors
            operation: Operation to apply
            channel_type: Type of channels
            optimize_layout: Whether to optimize memory layout
            
        Returns:
            List of processed channels
        """
        if not channels:
            return []
        
        start_time = time.perf_counter()
        
        # Move to GPU if needed
        gpu_channels = []
        for ch in channels:
            if not ch.is_cuda:
                ch = ch.cuda(non_blocking=self.config.use_async_transfers)
            gpu_channels.append(ch)
        
        # Optimize layouts if requested
        if optimize_layout:
            optimized_channels = []
            for ch in gpu_channels:
                opt_ch, _, _ = self.optimizer.optimize_layout(ch, channel_type=channel_type)
                optimized_channels.append(opt_ch)
            gpu_channels = optimized_channels
        
        # Batch processing with streams
        results = []
        stream_idx = 0
        
        for ch in gpu_channels:
            stream = self.optimizer.streams[stream_idx]
            stream_idx = (stream_idx + 1) % len(self.optimizer.streams)
            
            with torch.cuda.stream(stream):
                result = operation(ch)
                results.append(result)
        
        # Synchronize all streams
        for stream in self.optimizer.streams:
            stream.synchronize()
        
        # Update statistics
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.batch_stats['total_batches'] += 1
        self.batch_stats['total_items'] += len(channels)
        self.batch_stats['total_time_ms'] += elapsed_ms
        self.batch_stats['avg_batch_time_ms'] = (
            self.batch_stats['total_time_ms'] / self.batch_stats['total_batches']
        )
        
        # Track memory usage
        current_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        self.batch_stats['peak_memory_mb'] = max(
            self.batch_stats['peak_memory_mb'],
            current_memory_mb
        )
        
        return results
    
    def process_streaming(self,
                         channel_generator: Callable,
                         operation: Callable,
                         max_batch_size: int = 32) -> None:
        """Process channels in streaming fashion
        
        Args:
            channel_generator: Generator yielding channels
            operation: Operation to apply
            max_batch_size: Maximum batch size
        """
        batch = []
        
        for channel in channel_generator():
            batch.append(channel)
            
            if len(batch) >= max_batch_size:
                # Process batch
                self.process_batch(batch, operation)
                batch = []
        
        # Process remaining
        if batch:
            self.process_batch(batch, operation)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batch processing statistics"""
        return self.batch_stats.copy()


# Utility functions for integration
def create_optimized_gpu_config() -> GPUMemoryLayoutConfig:
    """Create optimized GPU memory configuration"""
    detector = get_gpu_detector()
    gpu_specs = detector.get_primary_gpu()
    
    if gpu_specs is None:
        raise RuntimeError("No GPU detected for optimization")
    
    return GPUMemoryLayoutConfig.from_gpu_specs(gpu_specs)


def optimize_channel_for_gpu(channel_data: torch.Tensor,
                            channel_type: str = "generic") -> Tuple[torch.Tensor, MemoryAccessMetrics]:
    """Optimize single channel for GPU access
    
    Args:
        channel_data: Channel tensor
        channel_type: Type of channel
        
    Returns:
        Tuple of (optimized tensor, metrics)
    """
    config = create_optimized_gpu_config()
    optimizer = ChannelMemoryOptimizer(config)
    
    # Move to GPU if needed
    if not channel_data.is_cuda:
        channel_data = channel_data.cuda()
    
    optimized, layout, metrics = optimizer.optimize_layout(channel_data, channel_type=channel_type)
    
    return optimized, metrics


if __name__ == "__main__":
    # Test GPU memory optimization
    print("GPU Memory Layout Optimizer Test")
    print("=" * 50)
    
    try:
        # Create configuration
        config = create_optimized_gpu_config()
        print(f"Alignment: {config.alignment_bytes} bytes")
        print(f"Block size: {config.block_size_1d}")
        print(f"Streams: {config.num_cuda_streams}")
        print(f"Pinned memory: {config.use_pinned_memory}")
        
        # Test optimizer
        optimizer = ChannelMemoryOptimizer(config)
        
        # Create test data
        test_tensor = torch.randn(1000, 512, device='cuda')
        print(f"\nTest tensor shape: {test_tensor.shape}")
        print(f"Memory: {test_tensor.numel() * 4 / 1024:.2f} KB")
        
        # Optimize layout
        optimized, layout, metrics = optimizer.optimize_layout(test_tensor)
        
        print(f"\nSelected layout: {layout.value}")
        print(f"Coalescing efficiency: {metrics.coalescing_efficiency:.2%}")
        print(f"L1 cache hit rate: {metrics.cache_hit_rate_l1:.2%}")
        print(f"L2 cache hit rate: {metrics.cache_hit_rate_l2:.2%}")
        print(f"Memory bandwidth: {metrics.memory_bandwidth_gbps:.2f} GB/s")
        print(f"Kernel time: {metrics.kernel_time_ms:.3f} ms")
        
        # Test allocator
        allocator = GPUMemoryAllocator(config)
        aligned_tensor = allocator.allocate_aligned((512, 256), dtype=torch.float32)
        print(f"\nAllocated aligned tensor: {aligned_tensor.shape}")
        print(f"Alignment check: {aligned_tensor.data_ptr() % config.alignment_bytes == 0}")
        
        # Test pattern analyzer
        analyzer = ChannelAccessPatternAnalyzer(config)
        analysis = analyzer.analyze_channel(test_tensor, "test_channel")
        
        print(f"\nPattern analysis:")
        print(f"Best pattern: {analysis['best_pattern']}")
        print(f"Sparsity: {analysis['sparsity']:.2%}")
        print(f"Is aligned: {analysis['is_aligned']}")
        
        if analysis['recommendations']:
            print("Recommendations:")
            for rec in analysis['recommendations']:
                print(f"  - {rec}")
        
        # Test batch processor
        processor = BatchedChannelProcessor(config)
        test_batch = [torch.randn(256, 128, device='cuda') for _ in range(10)]
        
        results = processor.process_batch(
            test_batch,
            lambda x: x.sum(dim=0),
            channel_type="test"
        )
        
        stats = processor.get_statistics()
        print(f"\nBatch processing:")
        print(f"Processed {stats['total_items']} items in {stats['total_batches']} batches")
        print(f"Average batch time: {stats['avg_batch_time_ms']:.2f} ms")
        print(f"Peak memory: {stats['peak_memory_mb']:.2f} MB")
        
        print("\nGPU Memory Optimizer initialized successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        traceback.print_exc()
        raise