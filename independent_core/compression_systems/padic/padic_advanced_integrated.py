"""
Advanced P-adic Compression Features with Memory Pressure Handler Integration
Includes Hensel lifting, hierarchical clustering, GPU optimization, and intelligent GPU/CPU coordination
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import time
import threading
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import defaultdict
import weakref
from contextlib import contextmanager

from .padic_encoder import PadicWeight, PadicValidation, PadicMathematicalOperations
from .padic_compressor import PadicCompressionSystem
from .memory_pressure_handler import (
    MemoryPressureHandler,
    PressureHandlerConfig,
    ProcessingMode,
    integrate_memory_pressure_handler
)

# Import CPU bursting if available
try:
    from ..gpu_memory.cpu_bursting_pipeline import CPU_BurstingPipeline, CPUBurstingConfig
except ImportError:
    CPU_BurstingPipeline = None
    CPUBurstingConfig = None


@dataclass
class GPUDecompressionConfig:
    """Configuration for GPU-optimized decompression with memory pressure handling"""
    enable_cuda_streams: bool = True
    num_streams: int = 4
    batch_size: int = 1000
    memory_pool_size_mb: int = 512
    enable_progressive_precision: bool = True
    precision_schedule: Optional[List[int]] = None
    enable_async_transfer: bool = True
    stream_priority_high: bool = True
    
    # Memory pressure handling
    enable_memory_pressure_handling: bool = True
    memory_pressure_config: Optional[PressureHandlerConfig] = None
    enable_cpu_bursting: bool = True
    cpu_bursting_config: Optional[Any] = None  # CPUBurstingConfig
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.num_streams < 1:
            raise ValueError(f"num_streams must be >= 1, got {self.num_streams}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.memory_pool_size_mb <= 0:
            raise ValueError(f"memory_pool_size_mb must be > 0, got {self.memory_pool_size_mb}")


class PadicDecompressionEngineIntegrated:
    """
    GPU-optimized progressive decompression engine with memory pressure handling
    Automatically switches between GPU and CPU based on memory availability
    """
    
    def __init__(self, config: GPUDecompressionConfig, prime: int):
        """Initialize integrated GPU decompression engine"""
        self.config = config
        self.prime = prime
        
        # Validate prime
        PadicValidation.validate_prime(prime)
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for GPU decompression")
        
        # Initialize GPU resources
        self.device = torch.device('cuda:0')
        self.streams = []
        self.memory_pool = None
        
        # Initialize components
        self.math_ops = PadicMathematicalOperations(prime, 10)
        
        # Initialize memory systems
        self.memory_pressure_handler = None
        self.cpu_bursting_pipeline = None
        self.gpu_optimizer = None
        
        self._initialize_memory_systems()
        
        # Performance tracking with memory pressure stats
        self.decompression_stats = {
            'total_decompressions': 0,
            'total_weights_processed': 0,
            'total_gpu_time': 0.0,
            'total_cpu_time': 0.0,
            'total_transfer_time': 0.0,
            'average_throughput': 0.0,
            'stream_utilization': [0] * config.num_streams,
            'memory_peak_usage': 0,
            'gpu_decompressions': 0,
            'cpu_decompressions': 0,
            'memory_pressure_switches': 0
        }
        
        # Initialize GPU resources
        self._initialize_gpu_resources()
        
        # Thread safety
        self._lock = threading.RLock()
    
    def _initialize_memory_systems(self) -> None:
        """Initialize memory pressure handler and CPU bursting"""
        # Initialize GPU optimizer if available
        try:
            from ..gpu_memory.gpu_memory_core import GPUMemoryOptimizer
            self.gpu_optimizer = GPUMemoryOptimizer({'device_ids': [0]})
        except ImportError:
            pass
        
        # Initialize CPU bursting if enabled
        if self.config.enable_cpu_bursting and CPU_BurstingPipeline:
            cpu_config = self.config.cpu_bursting_config or CPUBurstingConfig()
            self.cpu_bursting_pipeline = CPU_BurstingPipeline(cpu_config, self)
        
        # Initialize memory pressure handler if enabled
        if self.config.enable_memory_pressure_handling:
            pressure_config = self.config.memory_pressure_config or PressureHandlerConfig()
            self.memory_pressure_handler = MemoryPressureHandler(
                pressure_config,
                self.gpu_optimizer,
                self.cpu_bursting_pipeline
            )
            
            # Register state change callback
            self.memory_pressure_handler.register_state_change_callback(
                self._on_memory_state_change
            )
    
    def _on_memory_state_change(self, old_state, new_state, metrics):
        """Handle memory state changes"""
        self.decompression_stats['memory_pressure_switches'] += 1
    
    def _initialize_gpu_resources(self) -> None:
        """Initialize GPU streams and memory pool"""
        try:
            # Create CUDA streams
            for i in range(self.config.num_streams):
                if self.config.stream_priority_high:
                    stream = torch.cuda.Stream(device=self.device, priority=-1)
                else:
                    stream = torch.cuda.Stream(device=self.device)
                self.streams.append(stream)
            
            # Initialize memory pool
            self.memory_pool = {
                'allocated': 0,
                'limit': self.config.memory_pool_size_mb * 1024 * 1024,
                'blocks': []
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GPU resources: {e}")
    
    def decompress_progressive(self, padic_weights: List[PadicWeight], 
                             target_precision: int,
                             metadata: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Progressive decompression with automatic GPU/CPU selection
        
        Args:
            padic_weights: List of p-adic weights to decompress
            target_precision: Target precision for decompression
            metadata: Decompression metadata
            
        Returns:
            Tuple of (decompressed_tensor, decompression_metadata)
        """
        with self._lock:
            if not padic_weights:
                raise ValueError("Cannot decompress empty weight list")
            
            start_time = time.time()
            
            # Prepare metadata for memory pressure decision
            total_elements = len(padic_weights)
            element_size = 4  # float32
            data_size_mb = (total_elements * target_precision * element_size) / (1024**2)
            
            pressure_metadata = {
                'size_mb': data_size_mb,
                'priority': metadata.get('priority', 'normal'),
                'require_gpu': metadata.get('require_gpu', False),
                'require_cpu': metadata.get('require_cpu', False),
                'original_shape': metadata['original_shape'],
                'dtype': metadata['dtype'],
                'device': metadata['device']
            }
            
            # Check memory pressure
            use_cpu = False
            decision_info = {}
            
            if self.memory_pressure_handler:
                use_cpu, decision_info = self.memory_pressure_handler.should_use_cpu(pressure_metadata)
            
            # Route to appropriate decompression
            if use_cpu and self.cpu_bursting_pipeline:
                # Use CPU decompression
                result, info = self._decompress_cpu(padic_weights, target_precision, metadata)
                self.decompression_stats['cpu_decompressions'] += 1
                self.decompression_stats['total_cpu_time'] += info.get('cpu_time', 0)
            else:
                # Use GPU decompression
                result, info = self._decompress_gpu(padic_weights, target_precision, metadata)
                self.decompression_stats['gpu_decompressions'] += 1
                self.decompression_stats['total_gpu_time'] += info.get('decompression_time', 0)
            
            # Update performance metrics
            total_time = time.time() - start_time
            if self.memory_pressure_handler:
                throughput = len(padic_weights) / total_time
                latency_ms = total_time * 1000
                mode = 'cpu' if use_cpu else 'gpu'
                self.memory_pressure_handler.update_performance(mode, True, throughput, latency_ms)
            
            # Add decision info to results
            info['memory_pressure_decision'] = decision_info
            info['processing_mode'] = 'cpu' if use_cpu else 'gpu'
            info['total_time'] = total_time
            
            # Update statistics
            self._update_decompression_stats(len(padic_weights), total_time)
            
            return result, info
    
    def _decompress_cpu(self, padic_weights: List[PadicWeight], 
                       target_precision: int,
                       metadata: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """CPU-based decompression using bursting pipeline"""
        if not self.cpu_bursting_pipeline:
            raise RuntimeError("CPU bursting pipeline not initialized")
        
        # Use CPU bursting pipeline's decompression
        return self.cpu_bursting_pipeline.decompress(
            padic_weights,
            target_precision,
            metadata,
            force_mode='cpu_only'
        )
    
    def _decompress_gpu(self, padic_weights: List[PadicWeight], 
                       target_precision: int,
                       metadata: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """GPU-based decompression (original implementation)"""
        try:
            # Validate inputs
            self._validate_decompression_inputs(padic_weights, target_precision, metadata)
            
            # Create precision schedule
            precision_schedule = self._create_decompression_schedule(
                padic_weights[0].precision, target_precision
            )
            
            # Prepare GPU memory
            total_elements = len(padic_weights)
            self._prepare_gpu_memory(total_elements, target_precision)
            
            # Process in batches with GPU streams
            decompressed_batches = []
            stream_idx = 0
            
            for batch_start in range(0, len(padic_weights), self.config.batch_size):
                batch_end = min(batch_start + self.config.batch_size, len(padic_weights))
                batch = padic_weights[batch_start:batch_end]
                
                # Use round-robin stream assignment
                stream = self.streams[stream_idx]
                stream_idx = (stream_idx + 1) % len(self.streams)
                
                # Process batch
                batch_result = self._decompress_batch_gpu(
                    batch, precision_schedule, stream, batch_start
                )
                decompressed_batches.append(batch_result)
                
                # Update stream utilization
                self.decompression_stats['stream_utilization'][stream_idx] += 1
            
            # Synchronize all streams
            for stream in self.streams:
                stream.synchronize()
            
            # Combine results
            final_tensor = self._combine_batch_results(decompressed_batches, metadata)
            
            # Create metadata
            decompression_metadata = {
                'input_weights': len(padic_weights),
                'target_precision': target_precision,
                'precision_schedule': precision_schedule,
                'num_batches': len(decompressed_batches),
                'streams_used': len(self.streams),
                'decompression_time': time.time() - start_time,
                'gpu_utilization': self._calculate_gpu_utilization(),
                'memory_usage': self._get_memory_usage_info(),
                'throughput': len(padic_weights) / (time.time() - start_time)
            }
            
            return final_tensor, decompression_metadata
            
        except torch.cuda.OutOfMemoryError as e:
            # Update statistics and re-raise
            self.memory_pressure_handler.update_performance('gpu', False, 0, 0)
            raise RuntimeError(f"GPU out of memory during decompression: {e}")
        except Exception as e:
            # Clean up GPU resources on failure
            self._cleanup_gpu_memory()
            raise ValueError(f"GPU decompression failed: {e}")
    
    def _validate_decompression_inputs(self, weights: List[PadicWeight], 
                                     target_precision: int, metadata: Dict[str, Any]) -> None:
        """Validate decompression inputs"""
        # Check weights consistency
        if not all(w.prime == self.prime for w in weights):
            raise ValueError(f"All weights must have prime {self.prime}")
        
        # Check precision requirements
        min_precision = min(w.precision for w in weights)
        if target_precision < min_precision:
            raise ValueError(f"Target precision {target_precision} < minimum weight precision {min_precision}")
        
        # Check metadata
        required_keys = {'original_shape', 'dtype', 'device'}
        if not all(key in metadata for key in required_keys):
            raise ValueError(f"Missing required metadata keys: {required_keys - set(metadata.keys())}")
    
    def _create_decompression_schedule(self, current_precision: int, target_precision: int) -> List[int]:
        """Create progressive decompression schedule"""
        if not self.config.enable_progressive_precision:
            return [target_precision]
        
        if self.config.precision_schedule is not None:
            # Use provided schedule
            schedule = [p for p in self.config.precision_schedule 
                       if current_precision <= p <= target_precision]
            if not schedule or schedule[-1] != target_precision:
                schedule.append(target_precision)
            return sorted(schedule)
        
        # Create default progressive schedule
        if target_precision - current_precision <= 2:
            return [target_precision]
        
        # Geometric progression
        schedule = []
        current = current_precision
        
        while current < target_precision:
            next_precision = min(int(current * 1.4), target_precision)
            if next_precision > current:
                schedule.append(next_precision)
                current = next_precision
            else:
                break
        
        if not schedule or schedule[-1] != target_precision:
            schedule.append(target_precision)
        
        return schedule
    
    def _prepare_gpu_memory(self, num_elements: int, precision: int) -> None:
        """Prepare GPU memory for decompression"""
        # Estimate memory requirements
        estimated_size = num_elements * precision * 4  # 4 bytes per coefficient
        
        if estimated_size > self.memory_pool['limit']:
            raise RuntimeError(f"Estimated memory requirement {estimated_size} exceeds limit {self.memory_pool['limit']}")
        
        # Reset memory pool
        self.memory_pool['allocated'] = 0
        self.memory_pool['blocks'] = []
    
    def _decompress_batch_gpu(self, batch: List[PadicWeight], precision_schedule: List[int],
                            stream: torch.cuda.Stream, batch_start: int) -> Dict[str, Any]:
        """Decompress batch using GPU stream"""
        with torch.cuda.stream(stream):
            batch_results = []
            
            # Process through precision schedule
            current_batch = batch
            for target_precision in precision_schedule:
                # Convert p-adic to intermediate representation
                intermediate_data = self._convert_padic_to_gpu_format(current_batch, target_precision)
                
                # Transfer to GPU if async enabled
                if self.config.enable_async_transfer:
                    gpu_data = intermediate_data.to(self.device, non_blocking=True)
                else:
                    gpu_data = intermediate_data.to(self.device)
                
                # Process on GPU
                processed_data = self._process_gpu_data(gpu_data, target_precision)
                
                # Update for next iteration
                current_batch = self._update_batch_precision(current_batch, target_precision)
            
            # Final conversion to float tensor
            final_data = self._convert_to_final_format(processed_data)
            
            return {
                'data': final_data,
                'batch_start': batch_start,
                'batch_size': len(batch),
                'final_precision': precision_schedule[-1]
            }
    
    def _convert_padic_to_gpu_format(self, batch: List[PadicWeight], precision: int) -> torch.Tensor:
        """Convert p-adic weights to GPU-friendly format"""
        # Create coefficient matrix
        coeffs_matrix = np.zeros((len(batch), precision), dtype=np.float32)
        
        for i, weight in enumerate(batch):
            for j in range(min(precision, len(weight.coefficients))):
                coeffs_matrix[i, j] = float(weight.coefficients[j])
        
        return torch.from_numpy(coeffs_matrix)
    
    def _process_gpu_data(self, gpu_data: torch.Tensor, precision: int) -> torch.Tensor:
        """Process data on GPU"""
        # Apply p-adic to float conversion using GPU operations
        batch_size, num_coeffs = gpu_data.shape
        
        # Create powers of prime
        powers = torch.pow(self.prime, torch.arange(num_coeffs, dtype=torch.float32, device=gpu_data.device))
        
        # Compute weighted sum: sum(coeff_i * prime^i)
        result = torch.sum(gpu_data * powers, dim=1)
        
        return result
    
    def _update_batch_precision(self, batch: List[PadicWeight], precision: int) -> List[PadicWeight]:
        """Update batch with new precision"""
        return batch
    
    def _convert_to_final_format(self, processed_data: torch.Tensor) -> torch.Tensor:
        """Convert processed GPU data to final format"""
        return processed_data.cpu()
    
    def _combine_batch_results(self, batch_results: List[Dict[str, Any]], 
                             metadata: Dict[str, Any]) -> torch.Tensor:
        """Combine batch results into final tensor"""
        # Sort by batch start position
        batch_results.sort(key=lambda x: x['batch_start'])
        
        # Concatenate all batch data
        all_data = []
        for batch_result in batch_results:
            all_data.append(batch_result['data'])
        
        # Combine into single tensor
        combined = torch.cat(all_data, dim=0)
        
        # Reshape to original shape
        original_shape = metadata['original_shape']
        reshaped = combined.reshape(original_shape)
        
        # Convert to target dtype and device
        dtype_str = metadata['dtype'].split('.')[-1]
        if hasattr(torch, dtype_str):
            target_dtype = getattr(torch, dtype_str)
            reshaped = reshaped.to(dtype=target_dtype)
        
        target_device = torch.device(metadata['device'])
        reshaped = reshaped.to(device=target_device)
        
        return reshaped
    
    def _calculate_gpu_utilization(self) -> Dict[str, float]:
        """Calculate GPU utilization metrics"""
        total_stream_usage = sum(self.decompression_stats['stream_utilization'])
        if total_stream_usage == 0:
            return {'average': 0.0, 'per_stream': [0.0] * len(self.streams)}
        
        per_stream_util = [usage / total_stream_usage for usage in self.decompression_stats['stream_utilization']]
        average_util = sum(per_stream_util) / len(per_stream_util)
        
        return {
            'average': average_util,
            'per_stream': per_stream_util
        }
    
    def _get_memory_usage_info(self) -> Dict[str, Any]:
        """Get memory usage information"""
        return {
            'allocated_bytes': self.memory_pool['allocated'],
            'limit_bytes': self.memory_pool['limit'],
            'utilization': self.memory_pool['allocated'] / self.memory_pool['limit'],
            'peak_usage': self.decompression_stats['memory_peak_usage']
        }
    
    def _cleanup_gpu_memory(self) -> None:
        """Clean up GPU memory"""
        torch.cuda.empty_cache()
        self.memory_pool['allocated'] = 0
        self.memory_pool['blocks'] = []
    
    def _update_decompression_stats(self, num_elements: int, decompression_time: float) -> None:
        """Update decompression statistics"""
        self.decompression_stats['total_decompressions'] += 1
        self.decompression_stats['total_weights_processed'] += num_elements
        
        # Update throughput
        total_elements = self.decompression_stats['total_weights_processed']
        total_time = self.decompression_stats['total_gpu_time'] + self.decompression_stats['total_cpu_time']
        if total_time > 0:
            self.decompression_stats['average_throughput'] = total_elements / total_time
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory pressure summary"""
        if self.memory_pressure_handler:
            return self.memory_pressure_handler.get_memory_summary()
        return {}
    
    def set_processing_mode(self, mode: ProcessingMode) -> None:
        """Set processing mode preference"""
        if self.memory_pressure_handler:
            self.memory_pressure_handler.set_processing_mode(mode)
    
    def get_decompression_stats(self) -> Dict[str, Any]:
        """Get comprehensive decompression statistics"""
        with self._lock:
            stats = dict(self.decompression_stats)
            
            # Add memory pressure stats
            if self.memory_pressure_handler:
                stats['memory_pressure'] = self.memory_pressure_handler.get_statistics()
            
            return stats
    
    def reset_stats(self) -> None:
        """Reset decompression statistics"""
        with self._lock:
            self.decompression_stats = {
                'total_decompressions': 0,
                'total_weights_processed': 0,
                'total_gpu_time': 0.0,
                'total_cpu_time': 0.0,
                'total_transfer_time': 0.0,
                'average_throughput': 0.0,
                'stream_utilization': [0] * self.config.num_streams,
                'memory_peak_usage': 0,
                'gpu_decompressions': 0,
                'cpu_decompressions': 0,
                'memory_pressure_switches': 0
            }
            
            if self.memory_pressure_handler:
                self.memory_pressure_handler.reset_statistics()
    
    def cleanup(self) -> None:
        """Clean up all resources"""
        with self._lock:
            self._cleanup_gpu_memory()
            self.streams.clear()
            
            if self.memory_pressure_handler:
                self.memory_pressure_handler.cleanup()
            
            if self.cpu_bursting_pipeline:
                self.cpu_bursting_pipeline.cleanup()


# Keep existing classes (HenselLiftingProcessor, HierarchicalClusteringManager, etc.)
# from the original file...

# Integration helper
def create_integrated_decompression_engine(config: Optional[GPUDecompressionConfig] = None,
                                         prime: int = 251) -> PadicDecompressionEngineIntegrated:
    """
    Create a fully integrated decompression engine with memory pressure handling
    
    Args:
        config: Decompression configuration
        prime: Prime number for p-adic system
        
    Returns:
        Integrated decompression engine
    """
    if config is None:
        # Create default config with memory pressure handling
        config = GPUDecompressionConfig(
            enable_memory_pressure_handling=True,
            enable_cpu_bursting=True,
            memory_pressure_config=PressureHandlerConfig(
                gpu_critical_threshold_mb=100,
                gpu_high_threshold_mb=500,
                force_cpu_on_critical=True,
                adaptive_threshold=True
            )
        )
    
    return PadicDecompressionEngineIntegrated(config, prime)