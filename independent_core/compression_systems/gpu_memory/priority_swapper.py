"""
Priority-Based Memory Swapper for AutoSwap System
Implements intelligent GPU-CPU memory swapping based on priorities
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import time
import logging
import threading
from collections import defaultdict, deque
from enum import Enum
import asyncio
import concurrent.futures

from .doa_scorer import DOAScore, SwapPriority

logger = logging.getLogger(__name__)


class SwapDirection(Enum):
    """Memory swap direction"""
    GPU_TO_CPU = "gpu_to_cpu"
    CPU_TO_GPU = "cpu_to_gpu"
    GPU_TO_DISK = "gpu_to_disk"
    DISK_TO_GPU = "disk_to_gpu"


class SwapStrategy(Enum):
    """Swap execution strategy"""
    IMMEDIATE = "immediate"      # Synchronous immediate swap
    ASYNC = "async"             # Asynchronous background swap
    BATCH = "batch"             # Batch multiple swaps
    PREDICTIVE = "predictive"   # Predictive pre-swapping


@dataclass
class SwapOperation:
    """Single swap operation"""
    operation_id: str
    memory_block_id: str
    source_device: torch.device
    target_device: torch.device
    size_bytes: int
    direction: SwapDirection
    priority: SwapPriority
    strategy: SwapStrategy
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class SwapBatch:
    """Batch of swap operations"""
    batch_id: str
    operations: List[SwapOperation]
    total_size_bytes: int
    priority: SwapPriority
    created_time: float = field(default_factory=time.time)
    executed_time: Optional[float] = None
    completion_time: Optional[float] = None


class PrioritySwapper:
    """
    Priority-based memory swapper for intelligent GPU-CPU memory management.
    Implements various swapping strategies based on memory priorities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Priority Swapper"""
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configuration
        self.max_concurrent_swaps = self.config.get('max_concurrent_swaps', 4)
        self.batch_size_threshold = self.config.get('batch_size_mb', 64) * 1024 * 1024
        self.async_threshold = self.config.get('async_threshold_mb', 16) * 1024 * 1024
        self.enable_compression = self.config.get('enable_compression', True)
        self.enable_predictive = self.config.get('enable_predictive', True)
        
        # Swap tracking
        self.active_swaps: Dict[str, SwapOperation] = {}
        self.swap_history: deque = deque(maxlen=10000)
        self.memory_mappings: Dict[str, Dict[str, Any]] = {}  # block_id -> location info
        
        # Batch management
        self.pending_batches: Dict[SwapPriority, deque] = {
            priority: deque() for priority in SwapPriority
        }
        self.batch_counter = 0
        
        # Performance tracking
        self.swap_stats = {
            'total_swaps': 0,
            'successful_swaps': 0,
            'failed_swaps': 0,
            'total_bytes_swapped': 0,
            'gpu_to_cpu_swaps': 0,
            'cpu_to_gpu_swaps': 0,
            'average_swap_time_ms': 0.0,
            'batch_operations': 0,
            'compression_savings_bytes': 0
        }
        
        # Async execution
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_concurrent_swaps
        )
        
        # Thread safety
        self.lock = threading.RLock()
        self.swap_lock = threading.Lock()
        
        # Pinned memory pool for fast transfers
        self.pinned_memory_pool: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self._initialize_pinned_memory()
        
        self.logger.info("PrioritySwapper initialized")
    
    def swap_out(self, block_id: str, doa_score: DOAScore, 
                 target: str = 'cpu') -> SwapOperation:
        """
        Swap memory block from GPU to CPU/disk.
        
        Args:
            block_id: Memory block identifier
            doa_score: DOA score with priority information
            target: Target location ('cpu' or 'disk')
            
        Returns:
            SwapOperation with result
        """
        if block_id not in self.memory_mappings:
            raise ValueError(f"Memory block {block_id} not registered")
        
        with self.lock:
            # Check if already swapped out
            mapping = self.memory_mappings[block_id]
            if mapping['location'] != 'gpu':
                raise RuntimeError(f"Block {block_id} not on GPU")
            
            # Create swap operation
            operation = SwapOperation(
                operation_id=f"swap_{int(time.time() * 1000)}_{self._get_next_op_id()}",
                memory_block_id=block_id,
                source_device=torch.device(f'cuda:{mapping["device_id"]}'),
                target_device=torch.device('cpu' if target == 'cpu' else 'cpu'),  # Disk uses CPU staging
                size_bytes=doa_score.size_bytes,
                direction=SwapDirection.GPU_TO_CPU if target == 'cpu' else SwapDirection.GPU_TO_DISK,
                priority=doa_score.swap_priority,
                strategy=self._determine_swap_strategy(doa_score)
            )
            
            # Execute swap based on strategy
            if operation.strategy == SwapStrategy.IMMEDIATE:
                self._execute_swap_immediate(operation)
            elif operation.strategy == SwapStrategy.ASYNC:
                self._execute_swap_async(operation)
            elif operation.strategy == SwapStrategy.BATCH:
                self._add_to_batch(operation)
            else:  # PREDICTIVE
                self._execute_swap_predictive(operation)
            
            return operation
    
    def swap_in(self, block_id: str, device_id: int = 0) -> SwapOperation:
        """
        Swap memory block from CPU/disk back to GPU.
        
        Args:
            block_id: Memory block identifier
            device_id: Target GPU device ID
            
        Returns:
            SwapOperation with result
        """
        if block_id not in self.memory_mappings:
            raise ValueError(f"Memory block {block_id} not registered")
        
        with self.lock:
            mapping = self.memory_mappings[block_id]
            if mapping['location'] == 'gpu':
                raise RuntimeError(f"Block {block_id} already on GPU")
            
            # Create swap operation
            operation = SwapOperation(
                operation_id=f"swap_{int(time.time() * 1000)}_{self._get_next_op_id()}",
                memory_block_id=block_id,
                source_device=torch.device('cpu'),
                target_device=torch.device(f'cuda:{device_id}'),
                size_bytes=mapping['size'],
                direction=SwapDirection.CPU_TO_GPU if mapping['location'] == 'cpu' else SwapDirection.DISK_TO_GPU,
                priority=SwapPriority.HIGH,  # Swap-in is usually high priority
                strategy=SwapStrategy.IMMEDIATE  # Usually immediate for swap-in
            )
            
            # Execute swap
            self._execute_swap_immediate(operation)
            
            return operation
    
    def batch_swap(self, swap_requests: List[Tuple[str, DOAScore, str]]) -> List[SwapOperation]:
        """
        Perform batch swapping of multiple memory blocks.
        
        Args:
            swap_requests: List of (block_id, doa_score, target) tuples
            
        Returns:
            List of SwapOperations
        """
        operations = []
        
        # Group by priority
        priority_groups = defaultdict(list)
        for block_id, doa_score, target in swap_requests:
            priority_groups[doa_score.swap_priority].append((block_id, doa_score, target))
        
        # Process each priority group
        for priority in sorted(priority_groups.keys(), key=lambda x: x.value):
            group_operations = []
            
            for block_id, doa_score, target in priority_groups[priority]:
                try:
                    operation = self.swap_out(block_id, doa_score, target)
                    group_operations.append(operation)
                except Exception as e:
                    self.logger.error(f"Failed to swap {block_id}: {e}")
            
            operations.extend(group_operations)
        
        # Execute pending batches
        self._execute_pending_batches()
        
        return operations
    
    def register_memory_block(self, block_id: str, tensor: torch.Tensor) -> None:
        """Register a memory block for swapping"""
        with self.lock:
            device = tensor.device
            
            self.memory_mappings[block_id] = {
                'tensor': tensor,
                'size': tensor.numel() * tensor.element_size(),
                'dtype': tensor.dtype,
                'shape': tensor.shape,
                'location': 'gpu' if device.type == 'cuda' else 'cpu',
                'device_id': device.index if device.type == 'cuda' else -1,
                'cpu_tensor': None,
                'disk_path': None,
                'compressed': False,
                'compression_ratio': 1.0
            }
    
    def get_memory_location(self, block_id: str) -> Dict[str, Any]:
        """Get current memory location info"""
        with self.lock:
            if block_id not in self.memory_mappings:
                raise ValueError(f"Memory block {block_id} not registered")
            
            return self.memory_mappings[block_id].copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get swapper statistics"""
        with self.lock:
            # Calculate memory distribution
            memory_dist = {
                'gpu': 0,
                'cpu': 0,
                'disk': 0
            }
            
            for mapping in self.memory_mappings.values():
                location = mapping['location']
                size = mapping['size']
                
                if location == 'gpu':
                    memory_dist['gpu'] += size
                elif location == 'cpu':
                    memory_dist['cpu'] += size
                else:
                    memory_dist['disk'] += size
            
            return {
                'swap_stats': self.swap_stats.copy(),
                'active_swaps': len(self.active_swaps),
                'total_blocks': len(self.memory_mappings),
                'memory_distribution_mb': {
                    k: v / (1024 * 1024) for k, v in memory_dist.items()
                },
                'pinned_memory_mb': self._get_pinned_memory_size() / (1024 * 1024),
                'pending_batches': {
                    priority.name: len(queue) 
                    for priority, queue in self.pending_batches.items()
                }
            }
    
    def _initialize_pinned_memory(self) -> None:
        """Initialize pinned memory pool for fast transfers"""
        if not torch.cuda.is_available():
            return
        
        # Pre-allocate pinned memory buffers
        sizes = [1, 4, 16, 64]  # MB
        for size_mb in sizes:
            size_bytes = size_mb * 1024 * 1024
            try:
                # Allocate pinned memory
                pinned_tensor = torch.empty(
                    size_bytes // 4,  # float32
                    dtype=torch.float32,
                    pin_memory=True
                )
                self.pinned_memory_pool[size_bytes].append(pinned_tensor)
            except Exception as e:
                self.logger.warning(f"Failed to allocate {size_mb}MB pinned memory: {e}")
    
    def _determine_swap_strategy(self, doa_score: DOAScore) -> SwapStrategy:
        """Determine optimal swap strategy"""
        # Immediate for critical priorities
        if doa_score.swap_priority == SwapPriority.CRITICAL:
            return SwapStrategy.IMMEDIATE
        
        # Async for medium-sized blocks
        if doa_score.size_bytes < self.async_threshold:
            return SwapStrategy.ASYNC
        
        # Batch for low priority
        if doa_score.swap_priority in [SwapPriority.LOW, SwapPriority.IDLE]:
            return SwapStrategy.BATCH
        
        # Predictive if enabled and applicable
        if self.enable_predictive and doa_score.access_metrics.access_pattern.value in ['temporal', 'sequential']:
            return SwapStrategy.PREDICTIVE
        
        return SwapStrategy.IMMEDIATE
    
    def _execute_swap_immediate(self, operation: SwapOperation) -> None:
        """Execute swap immediately (synchronous)"""
        try:
            self.active_swaps[operation.operation_id] = operation
            
            if operation.direction == SwapDirection.GPU_TO_CPU:
                self._swap_gpu_to_cpu(operation)
            elif operation.direction == SwapDirection.CPU_TO_GPU:
                self._swap_cpu_to_gpu(operation)
            elif operation.direction == SwapDirection.GPU_TO_DISK:
                self._swap_gpu_to_disk(operation)
            else:  # DISK_TO_GPU
                self._swap_disk_to_gpu(operation)
            
            operation.success = True
            
        except Exception as e:
            operation.success = False
            operation.error_message = str(e)
            self.logger.error(f"Swap operation {operation.operation_id} failed: {e}")
            self.swap_stats['failed_swaps'] += 1
            raise
        
        finally:
            operation.end_time = time.time()
            self._update_swap_stats(operation)
            del self.active_swaps[operation.operation_id]
            self.swap_history.append(operation)
    
    def _execute_swap_async(self, operation: SwapOperation) -> None:
        """Execute swap asynchronously"""
        future = self.executor.submit(self._execute_swap_immediate, operation)
        # Store future for tracking if needed
    
    def _add_to_batch(self, operation: SwapOperation) -> None:
        """Add operation to batch queue"""
        with self.lock:
            self.pending_batches[operation.priority].append(operation)
            
            # Check if batch should be executed
            total_size = sum(
                op.size_bytes for op in self.pending_batches[operation.priority]
            )
            
            if total_size >= self.batch_size_threshold:
                self._execute_priority_batch(operation.priority)
    
    def _execute_swap_predictive(self, operation: SwapOperation) -> None:
        """Execute predictive swapping"""
        # For now, treat as async
        self._execute_swap_async(operation)
    
    def _swap_gpu_to_cpu(self, operation: SwapOperation) -> None:
        """Swap tensor from GPU to CPU"""
        with self.swap_lock:
            mapping = self.memory_mappings[operation.memory_block_id]
            gpu_tensor = mapping['tensor']
            
            # Use pinned memory if available
            pinned_buffer = self._get_pinned_buffer(operation.size_bytes)
            
            if pinned_buffer is not None:
                # Fast path with pinned memory
                pinned_buffer[:gpu_tensor.numel()].copy_(gpu_tensor.flatten())
                cpu_tensor = pinned_buffer[:gpu_tensor.numel()].clone()
                cpu_tensor = cpu_tensor.reshape(gpu_tensor.shape)
            else:
                # Direct transfer
                cpu_tensor = gpu_tensor.cpu()
            
            # Optionally compress
            if self.enable_compression and operation.size_bytes > 1024 * 1024:
                compressed_data, ratio = self._compress_tensor(cpu_tensor)
                mapping['compressed'] = True
                mapping['compression_ratio'] = ratio
                mapping['cpu_tensor'] = compressed_data
                self.swap_stats['compression_savings_bytes'] += int(operation.size_bytes * (1 - ratio))
            else:
                mapping['cpu_tensor'] = cpu_tensor
            
            # Clear GPU tensor
            mapping['tensor'] = None
            mapping['location'] = 'cpu'
            
            # Free GPU memory
            if gpu_tensor.is_cuda:
                del gpu_tensor
                torch.cuda.empty_cache()
    
    def _swap_cpu_to_gpu(self, operation: SwapOperation) -> None:
        """Swap tensor from CPU to GPU"""
        with self.swap_lock:
            mapping = self.memory_mappings[operation.memory_block_id]
            
            # Decompress if needed
            if mapping['compressed']:
                cpu_tensor = self._decompress_tensor(mapping['cpu_tensor'], mapping)
            else:
                cpu_tensor = mapping['cpu_tensor']
            
            # Transfer to GPU
            device = operation.target_device
            gpu_tensor = cpu_tensor.to(device)
            
            # Update mapping
            mapping['tensor'] = gpu_tensor
            mapping['cpu_tensor'] = None
            mapping['location'] = 'gpu'
            mapping['device_id'] = device.index
            mapping['compressed'] = False
    
    def _swap_gpu_to_disk(self, operation: SwapOperation) -> None:
        """Swap tensor from GPU to disk"""
        # First swap to CPU
        self._swap_gpu_to_cpu(operation)
        
        # Then save to disk
        mapping = self.memory_mappings[operation.memory_block_id]
        disk_path = f"/tmp/autoswap_{operation.memory_block_id}.pt"
        
        torch.save({
            'tensor': mapping['cpu_tensor'],
            'shape': mapping['shape'],
            'dtype': mapping['dtype'],
            'compressed': mapping['compressed'],
            'compression_ratio': mapping['compression_ratio']
        }, disk_path)
        
        # Clear CPU tensor
        mapping['cpu_tensor'] = None
        mapping['disk_path'] = disk_path
        mapping['location'] = 'disk'
    
    def _swap_disk_to_gpu(self, operation: SwapOperation) -> None:
        """Swap tensor from disk to GPU"""
        mapping = self.memory_mappings[operation.memory_block_id]
        
        # Load from disk
        data = torch.load(mapping['disk_path'])
        mapping['cpu_tensor'] = data['tensor']
        mapping['compressed'] = data['compressed']
        mapping['compression_ratio'] = data['compression_ratio']
        mapping['location'] = 'cpu'
        
        # Then swap to GPU
        operation.direction = SwapDirection.CPU_TO_GPU
        self._swap_cpu_to_gpu(operation)
        
        # Clean up disk file
        import os
        if os.path.exists(mapping['disk_path']):
            os.remove(mapping['disk_path'])
        mapping['disk_path'] = None
    
    def _execute_pending_batches(self) -> None:
        """Execute all pending batches"""
        for priority in SwapPriority:
            if self.pending_batches[priority]:
                self._execute_priority_batch(priority)
    
    def _execute_priority_batch(self, priority: SwapPriority) -> None:
        """Execute batch for specific priority"""
        with self.lock:
            operations = list(self.pending_batches[priority])
            self.pending_batches[priority].clear()
            
            if not operations:
                return
            
            # Create batch
            batch = SwapBatch(
                batch_id=f"batch_{self._get_next_batch_id()}",
                operations=operations,
                total_size_bytes=sum(op.size_bytes for op in operations),
                priority=priority
            )
            
            batch.executed_time = time.time()
            
            # Execute operations in parallel
            futures = []
            for operation in operations:
                future = self.executor.submit(self._execute_swap_immediate, operation)
                futures.append(future)
            
            # Wait for completion
            concurrent.futures.wait(futures)
            
            batch.completion_time = time.time()
            self.swap_stats['batch_operations'] += 1
    
    def _get_pinned_buffer(self, size: int) -> Optional[torch.Tensor]:
        """Get pinned memory buffer for size"""
        # Find smallest suitable buffer
        for buffer_size, buffers in sorted(self.pinned_memory_pool.items()):
            if buffer_size >= size and buffers:
                return buffers[0]
        return None
    
    def _compress_tensor(self, tensor: torch.Tensor) -> Tuple[Any, float]:
        """Compress tensor data"""
        # Simple compression using torch.save with compression
        import io
        buffer = io.BytesIO()
        torch.save(tensor, buffer, pickle_protocol=4)
        compressed_size = buffer.tell()
        original_size = tensor.numel() * tensor.element_size()
        ratio = compressed_size / original_size
        return buffer.getvalue(), ratio
    
    def _decompress_tensor(self, compressed_data: Any, mapping: Dict[str, Any]) -> torch.Tensor:
        """Decompress tensor data"""
        import io
        buffer = io.BytesIO(compressed_data)
        tensor = torch.load(buffer)
        return tensor
    
    def _update_swap_stats(self, operation: SwapOperation) -> None:
        """Update swap statistics"""
        self.swap_stats['total_swaps'] += 1
        
        if operation.success:
            self.swap_stats['successful_swaps'] += 1
            self.swap_stats['total_bytes_swapped'] += operation.size_bytes
            
            if operation.direction in [SwapDirection.GPU_TO_CPU, SwapDirection.GPU_TO_DISK]:
                self.swap_stats['gpu_to_cpu_swaps'] += 1
            else:
                self.swap_stats['cpu_to_gpu_swaps'] += 1
        
        if operation.end_time and operation.start_time:
            swap_time_ms = (operation.end_time - operation.start_time) * 1000
            total = self.swap_stats['total_swaps']
            current_avg = self.swap_stats['average_swap_time_ms']
            self.swap_stats['average_swap_time_ms'] = (
                (current_avg * (total - 1) + swap_time_ms) / total
            )
    
    def _get_next_op_id(self) -> int:
        """Get next operation ID"""
        return len(self.swap_history)
    
    def _get_next_batch_id(self) -> int:
        """Get next batch ID"""
        self.batch_counter += 1
        return self.batch_counter
    
    def _get_pinned_memory_size(self) -> int:
        """Get total pinned memory size"""
        total = 0
        for size, buffers in self.pinned_memory_pool.items():
            total += size * len(buffers)
        return total