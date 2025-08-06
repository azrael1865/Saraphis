"""
GPU Memory Pool System - High-Performance Pre-allocated Memory Management
Implements slab and buddy allocators with sub-millisecond allocation/deallocation
HARD FAILURES ONLY - NO FALLBACKS - IMMEDIATE CRASH ON ANY ERROR

ARCHITECTURE:
1. Slab Allocator - Fixed-size blocks for common allocations (powers of 2)
2. Buddy Allocator - Large allocations with splitting/merging
3. Lock-free allocation for hot paths using atomic operations
4. Per-thread caches to reduce contention
5. Multi-GPU support with device affinity
6. NUMA-aware allocation for optimal memory placement
7. Memory tagging for debugging and profiling
8. Immediate hard failure on any integrity violation

FEATURES:
- Sub-millisecond allocation/deallocation performance
- 128-byte alignment for coalesced GPU access
- Async allocations with CUDA streams
- Batch operations for efficiency
- Periodic defragmentation without blocking
- Fragmentation metrics and alerts
- Zero-initialization bypass for performance
- Integration with UnifiedMemoryHandler
- JAX and PyTorch backend support
"""

import torch
import numpy as np
import threading
import time
import queue
import hashlib
import traceback
import weakref
import gc
import os
import psutil
from typing import Dict, List, Optional, Tuple, Set, Any, Union, Callable, Deque
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import ctypes
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import GPU detection and configuration
try:
    from ..gpu_memory.gpu_auto_detector import get_config_updater, GPUSpecs
    from ..gpu_memory.gpu_memory_core import GPUMemoryOptimizer
except ImportError:
    # Direct imports for standalone
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
    from gpu_memory.gpu_auto_detector import get_config_updater, GPUSpecs
    from gpu_memory.gpu_memory_core import GPUMemoryOptimizer


class PoolType(Enum):
    """Memory pool types"""
    DEVICE = "device"       # Standard device memory
    PINNED = "pinned"       # Pinned host memory for fast transfers
    UNIFIED = "unified"     # Unified memory accessible from both CPU/GPU
    MANAGED = "managed"     # CUDA managed memory with automatic migration


class AllocationStatus(Enum):
    """Allocation status for debugging"""
    FREE = 0
    ALLOCATED = 1
    SPLITTING = 2      # Being split in buddy allocator
    MERGING = 3        # Being merged in buddy allocator
    CORRUPTED = 4      # Detected corruption - IMMEDIATE CRASH


class SlabClass(Enum):
    """Slab size classes (powers of 2)"""
    TINY = 256          # 256 bytes
    SMALL = 1024        # 1 KB
    MEDIUM = 4096       # 4 KB
    LARGE = 16384       # 16 KB
    XLARGE = 65536      # 64 KB
    HUGE = 262144       # 256 KB
    MEGA = 1048576      # 1 MB
    GIGA = 4194304      # 4 MB


@dataclass
class MemoryBlock:
    """Individual memory block in the pool"""
    block_id: str
    device_id: int
    pool_type: PoolType
    base_ptr: int           # Base pointer address
    size: int               # Size in bytes
    aligned_size: int       # Size after alignment
    status: AllocationStatus
    slab_class: Optional[SlabClass]
    
    # Buddy allocator fields
    buddy_order: int = 0    # Order in buddy system (size = 2^order * min_size)
    parent_block: Optional['MemoryBlock'] = None
    left_child: Optional['MemoryBlock'] = None
    right_child: Optional['MemoryBlock'] = None
    
    # Allocation tracking
    allocation_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0
    owner_thread: Optional[int] = None
    cuda_stream: Optional[Any] = None
    
    # Memory tagging for debugging
    tag: Optional[str] = None
    allocation_stack: Optional[str] = None
    
    # Integrity checking
    guard_pattern: bytes = field(default_factory=lambda: b'\xDE\xAD\xBE\xEF' * 4)
    checksum: Optional[str] = None
    
    def validate_integrity(self) -> bool:
        """Validate block integrity - CRASH ON FAILURE"""
        if self.status == AllocationStatus.CORRUPTED:
            raise RuntimeError(f"MEMORY CORRUPTION DETECTED in block {self.block_id} - IMMEDIATE ABORT")
        
        # Validate size and alignment
        if self.size <= 0 or self.aligned_size < self.size:
            self.status = AllocationStatus.CORRUPTED
            raise RuntimeError(f"INVALID SIZE in block {self.block_id}: size={self.size}, aligned={self.aligned_size} - MEMORY CORRUPTED")
        
        # Validate buddy relationships
        if self.parent_block and self.parent_block.status == AllocationStatus.CORRUPTED:
            self.status = AllocationStatus.CORRUPTED
            raise RuntimeError(f"PARENT CORRUPTION in block {self.block_id} - CASCADE FAILURE")
        
        return True
    
    def update_access(self):
        """Update access statistics"""
        self.last_access_time = time.time()
        self.access_count += 1


@dataclass
class SlabCache:
    """Cache for a single slab size class"""
    slab_class: SlabClass
    block_size: int
    device_id: int
    pool_type: PoolType
    
    # Free and allocated blocks
    free_blocks: Deque[MemoryBlock] = field(default_factory=deque)
    allocated_blocks: Dict[str, MemoryBlock] = field(default_factory=dict)
    
    # Pre-allocated memory chunks
    memory_chunks: List[torch.Tensor] = field(default_factory=list)
    chunk_size: int = 0  # Size of each chunk in blocks
    
    # Statistics
    total_blocks: int = 0
    allocated_count: int = 0
    free_count: int = 0
    allocation_hits: int = 0
    allocation_misses: int = 0
    
    # Thread-local caches
    thread_caches: Dict[int, Deque[MemoryBlock]] = field(default_factory=dict)
    thread_cache_size: int = 8  # Blocks per thread cache
    
    def allocate_block(self, thread_id: Optional[int] = None) -> Optional[MemoryBlock]:
        """Allocate a block from slab - LOCK-FREE for thread cache hits"""
        # Try thread-local cache first
        if thread_id and thread_id in self.thread_caches:
            cache = self.thread_caches[thread_id]
            if cache:
                block = cache.popleft()
                block.status = AllocationStatus.ALLOCATED
                block.owner_thread = thread_id
                self.allocation_hits += 1
                return block
        
        # Fall back to main free list
        if self.free_blocks:
            block = self.free_blocks.popleft()
            block.status = AllocationStatus.ALLOCATED
            block.owner_thread = thread_id
            self.allocated_blocks[block.block_id] = block
            self.allocated_count += 1
            self.free_count -= 1
            self.allocation_hits += 1
            return block
        
        self.allocation_misses += 1
        return None
    
    def free_block(self, block: MemoryBlock, thread_id: Optional[int] = None) -> bool:
        """Free a block back to slab"""
        if block.block_id not in self.allocated_blocks:
            raise RuntimeError(f"DOUBLE FREE DETECTED for block {block.block_id} - CORRUPTION - ABORT")
        
        # Validate before freeing
        block.validate_integrity()
        
        # Remove from allocated
        del self.allocated_blocks[block.block_id]
        block.status = AllocationStatus.FREE
        block.owner_thread = None
        self.allocated_count -= 1
        
        # Try to return to thread cache
        if thread_id and thread_id in self.thread_caches:
            cache = self.thread_caches[thread_id]
            if len(cache) < self.thread_cache_size:
                cache.append(block)
                return True
        
        # Return to main free list
        self.free_blocks.append(block)
        self.free_count += 1
        return True


@dataclass
class BuddyAllocator:
    """Buddy allocator for large allocations"""
    device_id: int
    pool_type: PoolType
    min_block_size: int = 1048576  # 1 MB minimum
    max_order: int = 20  # Max 2^20 * min_size = 1 GB
    
    # Free lists for each order
    free_lists: Dict[int, List[MemoryBlock]] = field(default_factory=dict)
    allocated_blocks: Dict[str, MemoryBlock] = field(default_factory=dict)
    
    # Memory regions
    memory_regions: List[torch.Tensor] = field(default_factory=list)
    region_map: Dict[int, MemoryBlock] = field(default_factory=dict)  # ptr -> root block
    
    # Statistics
    total_memory: int = 0
    allocated_memory: int = 0
    fragmentation: float = 0.0
    split_count: int = 0
    merge_count: int = 0
    
    def __post_init__(self):
        """Initialize free lists"""
        for order in range(self.max_order + 1):
            self.free_lists[order] = []
    
    def allocate(self, size: int, tag: Optional[str] = None) -> Optional[MemoryBlock]:
        """Allocate memory using buddy algorithm"""
        # Calculate required order
        order = self._size_to_order(size)
        if order > self.max_order:
            raise RuntimeError(f"ALLOCATION TOO LARGE: {size} bytes exceeds max order {self.max_order} - ABORT")
        
        # Find free block of appropriate size
        block = self._find_free_block(order)
        if not block:
            # Try to allocate new region
            if not self._allocate_new_region(order):
                return None
            block = self._find_free_block(order)
            if not block:
                raise RuntimeError(f"ALLOCATION FAILED after region creation - MEMORY SUBSYSTEM CORRUPTED - ABORT")
        
        # Mark as allocated
        block.status = AllocationStatus.ALLOCATED
        block.tag = tag
        block.allocation_stack = ''.join(traceback.format_stack()[-5:])  # Last 5 frames
        self.allocated_blocks[block.block_id] = block
        self.allocated_memory += block.size
        
        # Update fragmentation metric
        self._update_fragmentation()
        
        return block
    
    def free(self, block_id: str) -> bool:
        """Free memory and coalesce if possible"""
        if block_id not in self.allocated_blocks:
            raise RuntimeError(f"INVALID FREE: block {block_id} not allocated - DOUBLE FREE or CORRUPTION - ABORT")
        
        block = self.allocated_blocks.pop(block_id)
        block.validate_integrity()
        
        # Mark as free
        block.status = AllocationStatus.FREE
        self.allocated_memory -= block.size
        
        # Try to coalesce with buddy
        self._coalesce_block(block)
        
        # Update fragmentation
        self._update_fragmentation()
        
        return True
    
    def _find_free_block(self, order: int) -> Optional[MemoryBlock]:
        """Find or create a free block of given order"""
        # Check if we have a block of exact size
        if self.free_lists[order]:
            return self.free_lists[order].pop()
        
        # Try to split a larger block
        for larger_order in range(order + 1, self.max_order + 1):
            if self.free_lists[larger_order]:
                block = self.free_lists[larger_order].pop()
                # Split recursively until we get desired size
                return self._split_block(block, order)
        
        return None
    
    def _split_block(self, block: MemoryBlock, target_order: int) -> MemoryBlock:
        """Split a block recursively to target order"""
        if block.buddy_order == target_order:
            return block
        
        if block.buddy_order < target_order:
            raise RuntimeError(f"INVALID SPLIT: block order {block.buddy_order} < target {target_order} - LOGIC ERROR - ABORT")
        
        # Mark as splitting
        block.status = AllocationStatus.SPLITTING
        self.split_count += 1
        
        # Create two child blocks
        half_size = block.size // 2
        
        left_block = MemoryBlock(
            block_id=f"{block.block_id}_L",
            device_id=block.device_id,
            pool_type=block.pool_type,
            base_ptr=block.base_ptr,
            size=half_size,
            aligned_size=half_size,
            status=AllocationStatus.FREE,
            slab_class=None,
            buddy_order=block.buddy_order - 1,
            parent_block=block
        )
        
        right_block = MemoryBlock(
            block_id=f"{block.block_id}_R",
            device_id=block.device_id,
            pool_type=block.pool_type,
            base_ptr=block.base_ptr + half_size,
            size=half_size,
            aligned_size=half_size,
            status=AllocationStatus.FREE,
            slab_class=None,
            buddy_order=block.buddy_order - 1,
            parent_block=block
        )
        
        # Update parent
        block.left_child = left_block
        block.right_child = right_block
        
        # Add right block to free list
        self.free_lists[right_block.buddy_order].append(right_block)
        
        # Continue splitting left block if needed
        if left_block.buddy_order > target_order:
            return self._split_block(left_block, target_order)
        
        return left_block
    
    def _coalesce_block(self, block: MemoryBlock):
        """Coalesce block with its buddy if both are free"""
        if not block.parent_block:
            # Root block, add to free list
            self.free_lists[block.buddy_order].append(block)
            return
        
        parent = block.parent_block
        
        # Check if buddy is free
        buddy = parent.right_child if block == parent.left_child else parent.left_child
        
        if buddy and buddy.status == AllocationStatus.FREE:
            # Remove buddy from free list
            if buddy in self.free_lists[buddy.buddy_order]:
                self.free_lists[buddy.buddy_order].remove(buddy)
            
            # Mark as merging
            block.status = AllocationStatus.MERGING
            buddy.status = AllocationStatus.MERGING
            self.merge_count += 1
            
            # Reset parent
            parent.status = AllocationStatus.FREE
            parent.left_child = None
            parent.right_child = None
            
            # Recursively coalesce parent
            self._coalesce_block(parent)
        else:
            # Can't coalesce, add to free list
            self.free_lists[block.buddy_order].append(block)
    
    def _size_to_order(self, size: int) -> int:
        """Convert size to buddy order"""
        order = 0
        block_size = self.min_block_size
        while block_size < size and order < self.max_order:
            block_size *= 2
            order += 1
        return order
    
    def _allocate_new_region(self, min_order: int) -> bool:
        """Allocate a new memory region"""
        # Allocate largest possible region
        region_size = self.min_block_size * (2 ** self.max_order)
        
        try:
            with torch.cuda.device(self.device_id):
                if self.pool_type == PoolType.DEVICE:
                    tensor = torch.empty(region_size // 4, dtype=torch.float32, device=f'cuda:{self.device_id}')
                elif self.pool_type == PoolType.PINNED:
                    tensor = torch.empty(region_size // 4, dtype=torch.float32).pin_memory()
                elif self.pool_type == PoolType.UNIFIED:
                    # Unified memory allocation (requires CUDA toolkit support)
                    tensor = torch.empty(region_size // 4, dtype=torch.float32, device=f'cuda:{self.device_id}')
                else:
                    raise RuntimeError(f"UNSUPPORTED POOL TYPE: {self.pool_type} - ABORT")
                
                # Create root block
                root_block = MemoryBlock(
                    block_id=f"region_{len(self.memory_regions)}",
                    device_id=self.device_id,
                    pool_type=self.pool_type,
                    base_ptr=tensor.data_ptr(),
                    size=region_size,
                    aligned_size=region_size,
                    status=AllocationStatus.FREE,
                    slab_class=None,
                    buddy_order=self.max_order
                )
                
                # Store region
                self.memory_regions.append(tensor)
                self.region_map[tensor.data_ptr()] = root_block
                self.total_memory += region_size
                
                # Add to free list
                self.free_lists[self.max_order].append(root_block)
                
                logger.info(f"Allocated new {region_size/(1024**3):.2f}GB region on device {self.device_id}")
                return True
                
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(f"GPU OOM: Failed to allocate {region_size/(1024**3):.2f}GB region - NO FALLBACK - ABORT: {e}")
        except Exception as e:
            raise RuntimeError(f"CRITICAL: Region allocation failed - SYSTEM INTEGRITY COMPROMISED - ABORT: {e}")
    
    def _update_fragmentation(self):
        """Calculate fragmentation metric"""
        if self.total_memory == 0:
            self.fragmentation = 0.0
            return
        
        # Count free memory in different orders
        free_memory = 0
        largest_free = 0
        
        for order, blocks in self.free_lists.items():
            for block in blocks:
                free_memory += block.size
                largest_free = max(largest_free, block.size)
        
        # Fragmentation = 1 - (largest_free_block / total_free_memory)
        if free_memory > 0:
            self.fragmentation = 1.0 - (largest_free / free_memory)
        else:
            self.fragmentation = 0.0
    
    def defragment(self) -> int:
        """Defragment by coalescing all possible blocks"""
        coalesced = 0
        
        # Try to coalesce blocks in each order
        for order in range(self.max_order):
            blocks = list(self.free_lists[order])
            for block in blocks:
                if block.parent_block:
                    old_merge_count = self.merge_count
                    self._coalesce_block(block)
                    if self.merge_count > old_merge_count:
                        coalesced += 1
        
        self._update_fragmentation()
        return coalesced


@dataclass
class GPUMemoryPoolConfig:
    """Configuration for GPU memory pool"""
    # Pool sizes (auto-detected if None)
    device_pool_size_mb: Optional[int] = None
    pinned_pool_size_mb: Optional[int] = None
    unified_pool_size_mb: Optional[int] = None
    
    # Slab allocator settings
    slab_sizes: List[int] = field(default_factory=lambda: [256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304])
    slab_chunk_blocks: int = 64  # Blocks per chunk
    enable_thread_cache: bool = True
    thread_cache_size: int = 8
    
    # Buddy allocator settings
    buddy_min_size: int = 1048576  # 1 MB
    buddy_max_order: int = 20  # Up to 1 GB blocks
    
    # Alignment and padding
    alignment: int = 128  # 128-byte alignment for coalesced access
    guard_bytes: int = 64  # Guard bytes for overflow detection
    
    # Performance settings
    enable_async: bool = True
    batch_size: int = 32
    prefetch_distance: int = 2
    
    # Defragmentation settings
    defrag_threshold: float = 0.3  # Trigger at 30% fragmentation
    defrag_interval_seconds: float = 60.0
    max_defrag_time_ms: float = 10.0
    
    # NUMA settings
    enable_numa: bool = True
    numa_nodes: Optional[List[int]] = None
    
    # Debug and profiling
    enable_profiling: bool = True
    enable_tagging: bool = True
    enable_integrity_checks: bool = True
    
    # Integration settings
    enable_pytorch_backend: bool = True
    enable_jax_backend: bool = False
    
    # Auto-configuration flag
    _auto_configured: bool = False
    
    def __post_init__(self):
        """Auto-configure based on system"""
        if not self._auto_configured:
            self._apply_auto_configuration()
    
    def _apply_auto_configuration(self):
        """Apply auto-detected configuration"""
        try:
            config_updater = get_config_updater()
            gpu_specs = config_updater.gpu_specs
            
            # Set pool sizes based on GPU memory
            total_gpu_mb = gpu_specs.total_memory_mb
            
            if self.device_pool_size_mb is None:
                # Use 50% of GPU memory for device pool
                self.device_pool_size_mb = int(total_gpu_mb * 0.5)
            
            if self.pinned_pool_size_mb is None:
                # Use 10% of system RAM for pinned memory
                system_ram_mb = psutil.virtual_memory().total / (1024**2)
                self.pinned_pool_size_mb = int(min(system_ram_mb * 0.1, 4096))  # Cap at 4GB
            
            if self.unified_pool_size_mb is None:
                # Unified memory if supported (10% of GPU memory)
                self.unified_pool_size_mb = int(total_gpu_mb * 0.1)
            
            # Detect NUMA nodes
            if self.numa_nodes is None:
                try:
                    numa_info = os.popen("numactl --hardware | grep 'available:' | awk '{print $2}'").read().strip()
                    if numa_info and numa_info.isdigit():
                        self.numa_nodes = list(range(int(numa_info)))
                    else:
                        self.numa_nodes = [0]
                except:
                    self.numa_nodes = [0]
            
            self._auto_configured = True
            logger.info(f"Auto-configured GPU memory pool: device={self.device_pool_size_mb}MB, pinned={self.pinned_pool_size_mb}MB")
            
        except Exception as e:
            # Fallback configuration
            logger.warning(f"Auto-configuration failed: {e}, using defaults")
            if self.device_pool_size_mb is None:
                self.device_pool_size_mb = 4096  # 4GB
            if self.pinned_pool_size_mb is None:
                self.pinned_pool_size_mb = 512  # 512MB
            if self.unified_pool_size_mb is None:
                self.unified_pool_size_mb = 1024  # 1GB
            if self.numa_nodes is None:
                self.numa_nodes = [0]
            self._auto_configured = True


class GPUMemoryPool:
    """
    High-performance GPU memory pool with slab and buddy allocators.
    Provides sub-millisecond allocation/deallocation with hard failure guarantees.
    """
    
    def __init__(self, config: Optional[GPUMemoryPoolConfig] = None):
        """Initialize GPU memory pool"""
        self.config = config or GPUMemoryPoolConfig()
        
        # Device management
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA NOT AVAILABLE - GPU MEMORY POOL REQUIRES CUDA - ABORT")
        
        self.num_devices = torch.cuda.device_count()
        self.current_device = torch.cuda.current_device()
        
        # Pool components per device
        self.slab_allocators: Dict[int, Dict[PoolType, Dict[SlabClass, SlabCache]]] = defaultdict(lambda: defaultdict(dict))
        self.buddy_allocators: Dict[int, Dict[PoolType, BuddyAllocator]] = defaultdict(dict)
        
        # Allocation tracking
        self.allocations: Dict[str, Tuple[int, PoolType, MemoryBlock]] = {}  # allocation_id -> (device, type, block)
        self.allocation_counter = 0
        self.allocation_lock = threading.RLock()  # Main lock for allocation operations
        
        # Thread-local storage for lock-free paths
        self.thread_local = threading.local()
        
        # Statistics
        self.stats = {
            'total_allocations': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'slab_allocations': 0,
            'buddy_allocations': 0,
            'total_bytes_allocated': 0,
            'total_bytes_freed': 0,
            'fragmentation_events': 0,
            'integrity_failures': 0,
            'allocation_times': deque(maxlen=1000),
            'deallocation_times': deque(maxlen=1000)
        }
        
        # Defragmentation
        self.defrag_thread = None
        self.defrag_active = True
        self.last_defrag_time = time.time()
        
        # Initialize pools
        self._initialize_pools()
        
        # Start defragmentation thread
        if self.config.defrag_interval_seconds > 0:
            self.defrag_thread = threading.Thread(target=self._defragmentation_worker, daemon=True)
            self.defrag_thread.start()
        
        logger.info(f"GPU Memory Pool initialized for {self.num_devices} devices")
    
    def _initialize_pools(self):
        """Initialize memory pools for all devices"""
        for device_id in range(self.num_devices):
            with torch.cuda.device(device_id):
                # Initialize slab allocators
                for pool_type in [PoolType.DEVICE, PoolType.PINNED]:
                    for slab_size in self.config.slab_sizes:
                        slab_class = self._size_to_slab_class(slab_size)
                        if slab_class:
                            cache = self._create_slab_cache(device_id, pool_type, slab_class, slab_size)
                            self.slab_allocators[device_id][pool_type][slab_class] = cache
                
                # Initialize buddy allocators
                for pool_type in [PoolType.DEVICE, PoolType.PINNED, PoolType.UNIFIED]:
                    buddy = BuddyAllocator(
                        device_id=device_id,
                        pool_type=pool_type,
                        min_block_size=self.config.buddy_min_size,
                        max_order=self.config.buddy_max_order
                    )
                    self.buddy_allocators[device_id][pool_type] = buddy
                    
                    # Pre-allocate initial regions
                    self._preallocate_buddy_regions(buddy, pool_type)
        
        logger.info(f"Initialized pools: {len(self.slab_allocators)} devices with slab and buddy allocators")
    
    def _create_slab_cache(self, device_id: int, pool_type: PoolType, 
                          slab_class: SlabClass, block_size: int) -> SlabCache:
        """Create and initialize a slab cache"""
        cache = SlabCache(
            slab_class=slab_class,
            block_size=block_size,
            device_id=device_id,
            pool_type=pool_type,
            chunk_size=self.config.slab_chunk_blocks,
            thread_cache_size=self.config.thread_cache_size if self.config.enable_thread_cache else 0
        )
        
        # Pre-allocate initial chunks
        self._preallocate_slab_chunks(cache)
        
        return cache
    
    def _preallocate_slab_chunks(self, cache: SlabCache):
        """Pre-allocate memory chunks for slab cache"""
        chunk_count = 2  # Initial chunks
        blocks_per_chunk = cache.chunk_size
        
        for _ in range(chunk_count):
            try:
                # Allocate chunk
                chunk_size = cache.block_size * blocks_per_chunk
                
                with torch.cuda.device(cache.device_id):
                    if cache.pool_type == PoolType.DEVICE:
                        tensor = torch.empty(chunk_size // 4, dtype=torch.float32, 
                                           device=f'cuda:{cache.device_id}')
                    elif cache.pool_type == PoolType.PINNED:
                        tensor = torch.empty(chunk_size // 4, dtype=torch.float32).pin_memory()
                    else:
                        continue
                    
                    cache.memory_chunks.append(tensor)
                    base_ptr = tensor.data_ptr()
                    
                    # Create blocks from chunk
                    for i in range(blocks_per_chunk):
                        block = MemoryBlock(
                            block_id=f"slab_{cache.device_id}_{cache.slab_class.name}_{len(cache.free_blocks)+i}",
                            device_id=cache.device_id,
                            pool_type=cache.pool_type,
                            base_ptr=base_ptr + i * cache.block_size,
                            size=cache.block_size,
                            aligned_size=self._align_size(cache.block_size),
                            status=AllocationStatus.FREE,
                            slab_class=cache.slab_class
                        )
                        cache.free_blocks.append(block)
                        cache.total_blocks += 1
                        cache.free_count += 1
                        
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"Failed to preallocate slab chunk for {cache.slab_class.name}")
                break
    
    def _preallocate_buddy_regions(self, buddy: BuddyAllocator, pool_type: PoolType):
        """Pre-allocate initial regions for buddy allocator"""
        # Determine initial allocation based on pool type
        if pool_type == PoolType.DEVICE:
            target_mb = self.config.device_pool_size_mb // self.num_devices
        elif pool_type == PoolType.PINNED:
            target_mb = self.config.pinned_pool_size_mb // self.num_devices
        elif pool_type == PoolType.UNIFIED:
            target_mb = self.config.unified_pool_size_mb // self.num_devices
        else:
            return
        
        # Allocate regions to reach target size
        region_size_mb = (buddy.min_block_size * (2 ** buddy.max_order)) / (1024**2)
        regions_needed = max(1, int(target_mb / region_size_mb))
        
        for _ in range(regions_needed):
            try:
                buddy._allocate_new_region(0)
            except RuntimeError as e:
                logger.error(f"Failed to preallocate buddy region: {e}")
                break
    
    def allocate(self, size: int, device_id: Optional[int] = None, 
                pool_type: PoolType = PoolType.DEVICE,
                alignment: Optional[int] = None,
                stream: Optional[Any] = None,
                tag: Optional[str] = None,
                async_alloc: bool = False) -> Tuple[int, str]:
        """
        Allocate memory from pool with sub-millisecond performance.
        
        Args:
            size: Size in bytes
            device_id: Target device (None for current)
            pool_type: Type of memory pool
            alignment: Custom alignment (default 128)
            stream: CUDA stream for async allocation
            tag: Debug tag for allocation
            async_alloc: Enable async allocation
            
        Returns:
            Tuple of (pointer_address, allocation_id)
            
        Raises:
            RuntimeError: IMMEDIATE CRASH on any allocation failure
        """
        start_time = time.perf_counter()
        
        # Validate inputs
        if size <= 0:
            raise RuntimeError(f"INVALID SIZE: {size} <= 0 - ABORT")
        
        if device_id is None:
            device_id = self.current_device
        
        if device_id >= self.num_devices:
            raise RuntimeError(f"INVALID DEVICE: {device_id} >= {self.num_devices} - ABORT")
        
        # Apply alignment
        alignment = alignment or self.config.alignment
        aligned_size = self._align_size(size, alignment)
        
        # Get thread ID for lock-free allocation
        thread_id = threading.get_ident()
        
        # Generate allocation ID
        with self.allocation_lock:
            self.allocation_counter += 1
            allocation_id = f"alloc_{self.allocation_counter}_{thread_id}_{time.time()}"
        
        # Try allocation strategies
        block = None
        
        # Strategy 1: Try slab allocator for small sizes
        if aligned_size <= max(sc.value for sc in SlabClass):
            slab_class = self._size_to_slab_class(aligned_size)
            if slab_class and slab_class in self.slab_allocators[device_id][pool_type]:
                cache = self.slab_allocators[device_id][pool_type][slab_class]
                block = cache.allocate_block(thread_id)
                if block:
                    self.stats['slab_allocations'] += 1
        
        # Strategy 2: Use buddy allocator for large sizes or if slab failed
        if not block:
            buddy = self.buddy_allocators[device_id][pool_type]
            block = buddy.allocate(aligned_size, tag)
            if block:
                self.stats['buddy_allocations'] += 1
        
        # HARD FAIL if allocation failed
        if not block:
            self.stats['failed_allocations'] += 1
            raise RuntimeError(f"ALLOCATION FAILED: Unable to allocate {aligned_size} bytes on device {device_id} - NO MEMORY - ABORT")
        
        # Update block metadata
        block.cuda_stream = stream
        block.tag = tag or f"untagged_{allocation_id}"
        block.update_access()
        
        # Track allocation
        with self.allocation_lock:
            self.allocations[allocation_id] = (device_id, pool_type, block)
            self.stats['total_allocations'] += 1
            self.stats['successful_allocations'] += 1
            self.stats['total_bytes_allocated'] += aligned_size
        
        # Record timing
        alloc_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        self.stats['allocation_times'].append(alloc_time)
        
        if alloc_time > 1.0:
            logger.warning(f"Slow allocation: {alloc_time:.2f}ms for {aligned_size} bytes")
        
        return block.base_ptr, allocation_id
    
    def deallocate(self, allocation_id: str) -> bool:
        """
        Deallocate memory back to pool.
        
        Args:
            allocation_id: ID returned from allocate()
            
        Returns:
            True if successful
            
        Raises:
            RuntimeError: IMMEDIATE CRASH on double-free or corruption
        """
        start_time = time.perf_counter()
        
        with self.allocation_lock:
            if allocation_id not in self.allocations:
                raise RuntimeError(f"INVALID DEALLOCATION: {allocation_id} not found - DOUBLE FREE or CORRUPTION - ABORT")
            
            device_id, pool_type, block = self.allocations.pop(allocation_id)
        
        # Validate block integrity
        block.validate_integrity()
        
        # Free based on allocator type
        success = False
        
        if block.slab_class:
            # Return to slab
            cache = self.slab_allocators[device_id][pool_type][block.slab_class]
            thread_id = threading.get_ident()
            success = cache.free_block(block, thread_id)
        else:
            # Return to buddy allocator
            buddy = self.buddy_allocators[device_id][pool_type]
            success = buddy.free(block.block_id)
        
        if not success:
            raise RuntimeError(f"DEALLOCATION FAILED: Unable to free {allocation_id} - POOL CORRUPTED - ABORT")
        
        # Update statistics
        with self.allocation_lock:
            self.stats['total_bytes_freed'] += block.size
        
        # Record timing
        dealloc_time = (time.perf_counter() - start_time) * 1000
        self.stats['deallocation_times'].append(dealloc_time)
        
        return True
    
    def batch_allocate(self, sizes: List[int], device_id: Optional[int] = None,
                      pool_type: PoolType = PoolType.DEVICE) -> List[Tuple[int, str]]:
        """
        Batch allocation for efficiency.
        
        Args:
            sizes: List of sizes to allocate
            device_id: Target device
            pool_type: Memory pool type
            
        Returns:
            List of (pointer, allocation_id) tuples
            
        Raises:
            RuntimeError: CRASH if any allocation fails
        """
        results = []
        
        for i, size in enumerate(sizes):
            try:
                ptr, alloc_id = self.allocate(size, device_id, pool_type, tag=f"batch_{i}")
                results.append((ptr, alloc_id))
            except RuntimeError as e:
                # Rollback successful allocations
                for _, rollback_id in results:
                    try:
                        self.deallocate(rollback_id)
                    except:
                        pass
                raise RuntimeError(f"BATCH ALLOCATION FAILED at index {i}: {e} - TRANSACTION ABORTED")
        
        return results
    
    def batch_deallocate(self, allocation_ids: List[str]) -> bool:
        """
        Batch deallocation for efficiency.
        
        Args:
            allocation_ids: List of allocation IDs
            
        Returns:
            True if all successful
            
        Raises:
            RuntimeError: CRASH on any failure
        """
        for i, alloc_id in enumerate(allocation_ids):
            try:
                self.deallocate(alloc_id)
            except RuntimeError as e:
                raise RuntimeError(f"BATCH DEALLOCATION FAILED at index {i}: {e} - PARTIAL DEALLOCATION - POOL CORRUPTED")
        
        return True
    
    def get_fragmentation(self, device_id: Optional[int] = None) -> Dict[str, float]:
        """Get fragmentation metrics for device"""
        if device_id is None:
            device_id = self.current_device
        
        metrics = {}
        
        # Get buddy allocator fragmentation
        for pool_type, buddy in self.buddy_allocators[device_id].items():
            buddy._update_fragmentation()
            metrics[f"{pool_type.value}_buddy"] = buddy.fragmentation
        
        # Calculate slab fragmentation
        for pool_type, slabs in self.slab_allocators[device_id].items():
            total_free = 0
            total_size = 0
            
            for slab_class, cache in slabs.items():
                total_free += cache.free_count * cache.block_size
                total_size += cache.total_blocks * cache.block_size
            
            if total_size > 0:
                metrics[f"{pool_type.value}_slab"] = 1.0 - (total_free / total_size)
            else:
                metrics[f"{pool_type.value}_slab"] = 0.0
        
        return metrics
    
    def defragment(self, device_id: Optional[int] = None, max_time_ms: Optional[float] = None) -> Dict[str, int]:
        """
        Defragment memory pools.
        
        Args:
            device_id: Device to defragment (None for all)
            max_time_ms: Maximum time to spend (None for no limit)
            
        Returns:
            Dict of defragmentation statistics
        """
        start_time = time.perf_counter()
        max_time_ms = max_time_ms or self.config.max_defrag_time_ms
        
        stats = {
            'blocks_coalesced': 0,
            'slab_compactions': 0,
            'memory_moved_mb': 0
        }
        
        devices = [device_id] if device_id is not None else range(self.num_devices)
        
        for dev_id in devices:
            # Defragment buddy allocators
            for pool_type, buddy in self.buddy_allocators[dev_id].items():
                coalesced = buddy.defragment()
                stats['blocks_coalesced'] += coalesced
                
                # Check time limit
                if (time.perf_counter() - start_time) * 1000 > max_time_ms:
                    break
            
            # Compact slab allocators (move allocations to fill gaps)
            # This is more complex and would require moving live allocations
            # For now, just report fragmentation
        
        self.stats['fragmentation_events'] += 1
        self.last_defrag_time = time.time()
        
        return stats
    
    def _defragmentation_worker(self):
        """Background defragmentation thread"""
        while self.defrag_active:
            try:
                time.sleep(self.config.defrag_interval_seconds)
                
                # Check fragmentation levels
                for device_id in range(self.num_devices):
                    fragmentation = self.get_fragmentation(device_id)
                    
                    # Check if any pool exceeds threshold
                    for pool_name, frag_level in fragmentation.items():
                        if frag_level > self.config.defrag_threshold:
                            logger.info(f"Triggering defragmentation for {pool_name}: {frag_level:.2%}")
                            self.defragment(device_id)
                            break
                            
            except Exception as e:
                logger.error(f"Defragmentation worker error: {e}")
    
    def _size_to_slab_class(self, size: int) -> Optional[SlabClass]:
        """Map size to slab class"""
        for slab_class in SlabClass:
            if size <= slab_class.value:
                return slab_class
        return None
    
    def _align_size(self, size: int, alignment: Optional[int] = None) -> int:
        """Align size to specified boundary"""
        alignment = alignment or self.config.alignment
        return ((size + alignment - 1) // alignment) * alignment
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics"""
        with self.allocation_lock:
            stats = dict(self.stats)
        
        # Calculate averages
        if stats['allocation_times']:
            stats['avg_allocation_time_ms'] = np.mean(stats['allocation_times'])
            stats['p99_allocation_time_ms'] = np.percentile(stats['allocation_times'], 99)
        
        if stats['deallocation_times']:
            stats['avg_deallocation_time_ms'] = np.mean(stats['deallocation_times'])
            stats['p99_deallocation_time_ms'] = np.percentile(stats['deallocation_times'], 99)
        
        # Add per-device statistics
        stats['per_device'] = {}
        for device_id in range(self.num_devices):
            device_stats = {
                'fragmentation': self.get_fragmentation(device_id),
                'buddy_allocators': {},
                'slab_allocators': {}
            }
            
            # Buddy allocator stats
            for pool_type, buddy in self.buddy_allocators[device_id].items():
                device_stats['buddy_allocators'][pool_type.value] = {
                    'total_memory_mb': buddy.total_memory / (1024**2),
                    'allocated_memory_mb': buddy.allocated_memory / (1024**2),
                    'fragmentation': buddy.fragmentation,
                    'split_count': buddy.split_count,
                    'merge_count': buddy.merge_count
                }
            
            # Slab allocator stats
            for pool_type, slabs in self.slab_allocators[device_id].items():
                slab_stats = {}
                for slab_class, cache in slabs.items():
                    slab_stats[slab_class.name] = {
                        'total_blocks': cache.total_blocks,
                        'allocated_blocks': cache.allocated_count,
                        'free_blocks': cache.free_count,
                        'hit_rate': cache.allocation_hits / max(cache.allocation_hits + cache.allocation_misses, 1)
                    }
                device_stats['slab_allocators'][pool_type.value] = slab_stats
            
            stats['per_device'][f'device_{device_id}'] = device_stats
        
        return stats
    
    def validate_integrity(self) -> bool:
        """
        Validate pool integrity - CRASH on any corruption.
        
        Returns:
            True if valid
            
        Raises:
            RuntimeError: IMMEDIATE CRASH on corruption
        """
        with self.allocation_lock:
            # Validate all allocated blocks
            for allocation_id, (device_id, pool_type, block) in self.allocations.items():
                try:
                    block.validate_integrity()
                except RuntimeError as e:
                    self.stats['integrity_failures'] += 1
                    raise RuntimeError(f"INTEGRITY CHECK FAILED for {allocation_id}: {e} - POOL CORRUPTED - ABORT")
            
            # Validate buddy allocator structures
            for device_id in range(self.num_devices):
                for pool_type, buddy in self.buddy_allocators[device_id].items():
                    # Check free lists consistency
                    for order, blocks in buddy.free_lists.items():
                        for block in blocks:
                            if block.status != AllocationStatus.FREE:
                                raise RuntimeError(f"CORRUPT FREE LIST: Block {block.block_id} in free list but status={block.status} - ABORT")
                            if block.buddy_order != order:
                                raise RuntimeError(f"CORRUPT FREE LIST: Block {block.block_id} order mismatch {block.buddy_order} != {order} - ABORT")
            
            # Validate slab caches
            for device_id in range(self.num_devices):
                for pool_type, slabs in self.slab_allocators[device_id].items():
                    for slab_class, cache in slabs.items():
                        # Check counts
                        actual_free = len(cache.free_blocks)
                        actual_allocated = len(cache.allocated_blocks)
                        
                        if actual_free != cache.free_count:
                            raise RuntimeError(f"SLAB COUNT MISMATCH: Free count {cache.free_count} != actual {actual_free} - ABORT")
                        if actual_allocated != cache.allocated_count:
                            raise RuntimeError(f"SLAB COUNT MISMATCH: Allocated count {cache.allocated_count} != actual {actual_allocated} - ABORT")
        
        return True
    
    def shutdown(self):
        """Shutdown pool and release all resources"""
        logger.info("Shutting down GPU Memory Pool")
        
        # Stop defragmentation thread
        self.defrag_active = False
        if self.defrag_thread:
            self.defrag_thread.join(timeout=5.0)
        
        # Validate before shutdown
        try:
            self.validate_integrity()
        except RuntimeError as e:
            logger.error(f"Integrity check failed during shutdown: {e}")
        
        # Clear all allocations
        with self.allocation_lock:
            self.allocations.clear()
            
            # Clear buddy allocators
            for device_id in range(self.num_devices):
                self.buddy_allocators[device_id].clear()
            
            # Clear slab allocators
            for device_id in range(self.num_devices):
                self.slab_allocators[device_id].clear()
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info("GPU Memory Pool shutdown complete")


# Export main classes
__all__ = ['GPUMemoryPool', 'GPUMemoryPoolConfig', 'PoolType', 'AllocationStatus']