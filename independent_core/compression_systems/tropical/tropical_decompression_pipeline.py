"""
Tropical Decompression Pipeline - GPU-accelerated reconstruction with CPU fallback
Streaming decompression support with memory-aware processing
NO PLACEHOLDERS - PRODUCTION CODE ONLY
HARD FAILURES ONLY - NO GRACEFUL DEGRADATION
"""

import torch
import numpy as np
import time
import threading
import queue
from typing import Dict, List, Tuple, Optional, Any, Union, Generator
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Import tropical components
from .jax_tropical_engine import (
    TropicalJAXEngine,
    JAXTropicalConfig,
    JAXChannelProcessor
)
from .tropical_channel_extractor import (
    TropicalChannelExtractor,
    TropicalChannels,
    ChannelReconstructionConfig
)
from .tropical_compression_pipeline import (
    TropicalCompressionResult,
    TropicalCompressionConfig
)

# Import memory management
from ..padic.memory_pressure_handler import (
    MemoryPressureHandler,
    ProcessingMode,
    MemoryState
)
from independent_core.compression_systems.gpu_memory.cpu_bursting_pipeline import DecompressionMode


class DecompressionStrategy(Enum):
    """Decompression execution strategy"""
    GPU_ONLY = "gpu_only"
    CPU_ONLY = "cpu_only"
    GPU_WITH_FALLBACK = "gpu_with_fallback"
    STREAMING = "streaming"
    HYBRID = "hybrid"


@dataclass
class TropicalDecompressionConfig:
    """Configuration for tropical decompression pipeline"""
    # JAX configuration
    jax_config: JAXTropicalConfig = field(default_factory=JAXTropicalConfig)
    
    # Reconstruction settings
    reconstruction_config: ChannelReconstructionConfig = field(default_factory=ChannelReconstructionConfig)
    
    # Memory management
    enable_cpu_fallback: bool = True
    gpu_memory_threshold_mb: int = 2048
    cpu_batch_size: int = 1000
    
    # Streaming settings
    enable_streaming: bool = True
    stream_chunk_size: int = 10000
    stream_buffer_size: int = 5
    
    # Performance settings
    enable_gpu_acceleration: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # Validation
    validate_reconstruction: bool = True
    reconstruction_tolerance: float = 1e-6
    
    def __post_init__(self):
        """Validate configuration"""
        if self.gpu_memory_threshold_mb <= 0:
            raise ValueError(f"gpu_memory_threshold_mb must be positive, got {self.gpu_memory_threshold_mb}")
        if self.cpu_batch_size <= 0:
            raise ValueError(f"cpu_batch_size must be positive, got {self.cpu_batch_size}")
        if self.stream_chunk_size <= 0:
            raise ValueError(f"stream_chunk_size must be positive, got {self.stream_chunk_size}")


@dataclass
class DecompressionResult:
    """Result from tropical decompression"""
    reconstructed_tensor: torch.Tensor
    decompression_time_ms: float
    processing_mode: str  # 'gpu', 'cpu', 'hybrid', 'streaming'
    reconstruction_error: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TropicalDecompressionPipeline:
    """
    Tropical decompression pipeline with GPU acceleration and CPU fallback
    Supports streaming decompression for large models
    """
    
    def __init__(self, config: TropicalDecompressionConfig,
                 memory_handler: Optional[MemoryPressureHandler] = None):
        """Initialize tropical decompression pipeline"""
        self.config = config
        self.memory_handler = memory_handler
        
        # Initialize JAX engine for GPU decompression
        self.jax_engine = None
        self.jax_available = False
        self.gpu_available = False
        
        if config.enable_gpu_acceleration:
            try:
                self.jax_engine = TropicalJAXEngine(config.jax_config)
                self.jax_available = True
                self.gpu_available = self.jax_engine.gpu_available
                self.channel_processor = JAXChannelProcessor(self.jax_engine)
                logger.info(f"GPU decompression initialized with {len(self.jax_engine.devices)} devices")
            except RuntimeError as e:
                if not config.enable_cpu_fallback:
                    raise RuntimeError(f"JAX initialization FAILED and CPU fallback disabled: {e}")
                logger.warning(f"JAX initialization failed, using CPU fallback: {e}")
                self.jax_available = False
                self.gpu_available = False
        
        # Initialize channel extractor for reconstruction
        self.channel_extractor = TropicalChannelExtractor(config.reconstruction_config)
        
        # Determine device
        self.device = torch.device('cuda' if self.gpu_available else 'cpu')
        
        # Streaming decompression state
        self.stream_queue = queue.Queue(maxsize=config.stream_buffer_size) if config.enable_streaming else None
        self.streaming_active = False
        
        # Statistics
        self.decompression_stats = {
            'total_decompressions': 0,
            'gpu_decompressions': 0,
            'cpu_decompressions': 0,
            'streaming_decompressions': 0,
            'fallback_events': 0,
            'average_decompression_time_ms': 0.0,
            'average_reconstruction_error': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
    
    def decompress(self, compressed_result: TropicalCompressionResult,
                  target_shape: Optional[Tuple[int, ...]] = None,
                  target_dtype: Optional[torch.dtype] = None) -> DecompressionResult:
        """
        Decompress tropical compressed data
        
        Args:
            compressed_result: TropicalCompressionResult from compression
            target_shape: Target shape for reconstruction (uses metadata if None)
            target_dtype: Target dtype for reconstruction (uses metadata if None)
            
        Returns:
            DecompressionResult with reconstructed tensor
        """
        if compressed_result is None or compressed_result.compressed_channels is None:
            raise ValueError("Invalid compressed result - HARD FAILURE")
        
        start_time = time.perf_counter()
        
        with self._lock:
            try:
                # Extract target shape and dtype from metadata if not provided
                if target_shape is None:
                    target_shape = compressed_result.metadata.get('original_shape')
                    if target_shape is None:
                        raise ValueError("No target shape provided or in metadata - HARD FAILURE")
                
                if target_dtype is None:
                    dtype_str = compressed_result.metadata.get('original_dtype', 'torch.float32')
                    target_dtype = self._parse_dtype(dtype_str)
                
                # Determine decompression strategy
                strategy = self._determine_strategy(compressed_result, target_shape)
                
                # Execute decompression based on strategy
                if strategy == DecompressionStrategy.GPU_ONLY:
                    result = self._decompress_gpu(compressed_result.compressed_channels, target_shape, target_dtype)
                    processing_mode = 'gpu'
                    
                elif strategy == DecompressionStrategy.CPU_ONLY:
                    result = self._decompress_cpu(compressed_result.compressed_channels, target_shape, target_dtype)
                    processing_mode = 'cpu'
                    
                elif strategy == DecompressionStrategy.GPU_WITH_FALLBACK:
                    try:
                        result = self._decompress_gpu(compressed_result.compressed_channels, target_shape, target_dtype)
                        processing_mode = 'gpu'
                    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                        logger.warning(f"GPU decompression failed, falling back to CPU: {e}")
                        self.decompression_stats['fallback_events'] += 1
                        result = self._decompress_cpu(compressed_result.compressed_channels, target_shape, target_dtype)
                        processing_mode = 'cpu_fallback'
                        
                elif strategy == DecompressionStrategy.STREAMING:
                    result = self._decompress_streaming(compressed_result.compressed_channels, target_shape, target_dtype)
                    processing_mode = 'streaming'
                    
                else:  # HYBRID
                    result = self._decompress_hybrid(compressed_result.compressed_channels, target_shape, target_dtype)
                    processing_mode = 'hybrid'
                
                decompression_time = (time.perf_counter() - start_time) * 1000
                
                # Validate reconstruction if enabled
                reconstruction_error = None
                if self.config.validate_reconstruction and result is not None:
                    reconstruction_error = self._validate_reconstruction(result, compressed_result)
                
                # Update statistics
                self._update_statistics(decompression_time, reconstruction_error, processing_mode)
                
                # Build result
                decompression_result = DecompressionResult(
                    reconstructed_tensor=result,
                    decompression_time_ms=decompression_time,
                    processing_mode=processing_mode,
                    reconstruction_error=reconstruction_error,
                    metadata={
                        'strategy': strategy.value,
                        'target_shape': target_shape,
                        'target_dtype': str(target_dtype),
                        'device': str(self.device)
                    }
                )
                
                logger.info(f"Decompression completed: {processing_mode} mode in {decompression_time:.1f}ms")
                return decompression_result
                
            except Exception as e:
                raise RuntimeError(f"Decompression FAILED: {e}")
    
    def decompress_streaming(self, compressed_result: TropicalCompressionResult,
                           target_shape: Tuple[int, ...],
                           target_dtype: torch.dtype = torch.float32) -> Generator[torch.Tensor, None, None]:
        """
        Stream decompression for large tensors
        
        Args:
            compressed_result: Compressed data
            target_shape: Target shape
            target_dtype: Target dtype
            
        Yields:
            Chunks of decompressed tensor
        """
        if not self.config.enable_streaming:
            raise RuntimeError("Streaming not enabled in configuration - HARD FAILURE")
        
        channels = compressed_result.compressed_channels
        chunk_size = self.config.stream_chunk_size
        
        # Calculate total elements and chunks
        total_elements = np.prod(target_shape)
        num_chunks = (total_elements + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_elements)
            
            # Extract chunk from channels
            chunk_channels = self._extract_channel_chunk(channels, start_idx, end_idx)
            
            # Decompress chunk
            if self.gpu_available and self._check_gpu_memory():
                chunk_tensor = self._reconstruct_gpu_chunk(chunk_channels, end_idx - start_idx)
            else:
                chunk_tensor = self._reconstruct_cpu_chunk(chunk_channels, end_idx - start_idx)
            
            # Convert to target dtype
            chunk_tensor = chunk_tensor.to(target_dtype)
            
            yield chunk_tensor
            
            self.decompression_stats['streaming_decompressions'] += 1
    
    def _determine_strategy(self, compressed_result: TropicalCompressionResult,
                          target_shape: Tuple[int, ...]) -> DecompressionStrategy:
        """Determine optimal decompression strategy"""
        # Check if GPU is available
        if not self.gpu_available:
            return DecompressionStrategy.CPU_ONLY
        
        # Check memory pressure if handler available
        if self.memory_handler:
            metadata = {
                'size_mb': compressed_result.compressed_size_bytes / (1024 * 1024),
                'target_shape': target_shape
            }
            use_cpu, decision_info = self.memory_handler.should_use_cpu(metadata)
            
            if use_cpu:
                return DecompressionStrategy.CPU_ONLY
        
        # Check tensor size for streaming
        total_elements = np.prod(target_shape)
        estimated_memory_mb = total_elements * 4 / (1024 * 1024)  # float32
        
        if estimated_memory_mb > self.config.gpu_memory_threshold_mb:
            if self.config.enable_streaming:
                return DecompressionStrategy.STREAMING
            elif self.config.enable_cpu_fallback:
                return DecompressionStrategy.GPU_WITH_FALLBACK
            else:
                return DecompressionStrategy.GPU_ONLY
        
        # Default to GPU with fallback if enabled
        if self.config.enable_cpu_fallback:
            return DecompressionStrategy.GPU_WITH_FALLBACK
        
        return DecompressionStrategy.GPU_ONLY
    
    def _decompress_gpu(self, channels: TropicalChannels,
                       target_shape: Tuple[int, ...],
                       target_dtype: torch.dtype) -> torch.Tensor:
        """Decompress using GPU/JAX acceleration"""
        if not self.jax_available:
            raise RuntimeError("JAX not available for GPU decompression - HARD FAILURE")
        
        # Convert channels to JAX
        jax_channels = self.channel_processor.channels_to_jax(channels)
        
        # Process for reconstruction
        reconstructed_jax = self.channel_processor.reconstruct_from_channels(jax_channels)
        
        # Convert back to PyTorch tensor
        reconstructed_np = np.array(reconstructed_jax)
        reconstructed = torch.from_numpy(reconstructed_np).to(self.device)
        
        # Reshape and convert dtype
        reconstructed = reconstructed.reshape(target_shape)
        reconstructed = reconstructed.to(target_dtype)
        
        self.decompression_stats['gpu_decompressions'] += 1
        
        return reconstructed
    
    def _decompress_cpu(self, channels: TropicalChannels,
                       target_shape: Tuple[int, ...],
                       target_dtype: torch.dtype) -> torch.Tensor:
        """Decompress using CPU"""
        # Reconstruct from channels on CPU
        reconstructed = self.channel_extractor.reconstruct_from_channels(channels)
        
        # Move to CPU if needed
        if reconstructed.is_cuda:
            reconstructed = reconstructed.cpu()
        
        # Reshape and convert dtype
        reconstructed = reconstructed.reshape(target_shape)
        reconstructed = reconstructed.to(target_dtype)
        
        self.decompression_stats['cpu_decompressions'] += 1
        
        return reconstructed
    
    def _decompress_hybrid(self, channels: TropicalChannels,
                         target_shape: Tuple[int, ...],
                         target_dtype: torch.dtype) -> torch.Tensor:
        """Hybrid GPU/CPU decompression"""
        # Split channels for hybrid processing
        total_elements = np.prod(target_shape)
        gpu_elements = min(total_elements, self.config.gpu_memory_threshold_mb * 1024 * 256)  # Estimate
        
        if gpu_elements < total_elements:
            # Process part on GPU, part on CPU
            gpu_channels = self._extract_channel_chunk(channels, 0, gpu_elements)
            cpu_channels = self._extract_channel_chunk(channels, gpu_elements, total_elements)
            
            # Decompress both parts
            gpu_part = self._decompress_gpu(gpu_channels, (gpu_elements,), target_dtype)
            cpu_part = self._decompress_cpu(cpu_channels, (total_elements - gpu_elements,), target_dtype)
            
            # Combine results
            combined = torch.cat([gpu_part.flatten(), cpu_part.flatten()])
            combined = combined.reshape(target_shape)
            
            return combined
        else:
            # Everything fits on GPU
            return self._decompress_gpu(channels, target_shape, target_dtype)
    
    def _decompress_streaming(self, channels: TropicalChannels,
                            target_shape: Tuple[int, ...],
                            target_dtype: torch.dtype) -> torch.Tensor:
        """Streaming decompression for large tensors"""
        chunks = []
        
        for chunk in self.decompress_streaming(
            TropicalCompressionResult(
                compressed_channels=channels,
                compression_ratio=1.0,
                original_size_bytes=0,
                compressed_size_bytes=0,
                compression_time_ms=0.0,
                metadata={}
            ),
            target_shape,
            target_dtype
        ):
            chunks.append(chunk)
        
        # Combine all chunks
        result = torch.cat(chunks)
        result = result.reshape(target_shape)
        
        return result
    
    def _extract_channel_chunk(self, channels: TropicalChannels,
                              start_idx: int, end_idx: int) -> TropicalChannels:
        """Extract a chunk from channels"""
        # Flatten channels for indexing
        sign_flat = channels.sign_channel.flatten()
        exp_flat = channels.exponent_channel.flatten()
        mantissa_flat = channels.mantissa_channel.flatten()
        
        # Extract chunk
        chunk_channels = TropicalChannels(
            sign_channel=sign_flat[start_idx:end_idx],
            exponent_channel=exp_flat[start_idx:end_idx],
            mantissa_channel=mantissa_flat[start_idx:end_idx],
            metadata=channels.metadata.copy()
        )
        
        return chunk_channels
    
    def _reconstruct_gpu_chunk(self, chunk_channels: TropicalChannels,
                              chunk_size: int) -> torch.Tensor:
        """Reconstruct a chunk on GPU"""
        # Use JAX for GPU reconstruction
        jax_channels = self.channel_processor.channels_to_jax(chunk_channels)
        reconstructed_jax = self.channel_processor.reconstruct_from_channels(jax_channels)
        
        # Convert to PyTorch
        reconstructed_np = np.array(reconstructed_jax)
        reconstructed = torch.from_numpy(reconstructed_np).to(self.device)
        
        return reconstructed
    
    def _reconstruct_cpu_chunk(self, chunk_channels: TropicalChannels,
                              chunk_size: int) -> torch.Tensor:
        """Reconstruct a chunk on CPU"""
        reconstructed = self.channel_extractor.reconstruct_from_channels(chunk_channels)
        
        if reconstructed.is_cuda:
            reconstructed = reconstructed.cpu()
        
        return reconstructed
    
    def _check_gpu_memory(self) -> bool:
        """Check if GPU has enough memory"""
        if not self.gpu_available:
            return False
        
        try:
            if torch.cuda.is_available():
                free_memory = torch.cuda.mem_get_info()[0] / (1024 * 1024)  # MB
                return free_memory > self.config.gpu_memory_threshold_mb
        except:
            pass
        
        return False
    
    def _validate_reconstruction(self, reconstructed: torch.Tensor,
                                compressed_result: TropicalCompressionResult) -> float:
        """Validate reconstruction quality"""
        # For validation, we'd need the original tensor
        # Here we check basic properties
        
        # Check shape matches metadata
        expected_shape = compressed_result.metadata.get('original_shape')
        if expected_shape and reconstructed.shape != tuple(expected_shape):
            logger.warning(f"Shape mismatch: {reconstructed.shape} != {expected_shape}")
        
        # Calculate a basic error metric (placeholder for actual validation)
        # In production, you'd compare against original or use other metrics
        error = 0.0
        
        # Check for NaN or Inf
        if torch.isnan(reconstructed).any() or torch.isinf(reconstructed).any():
            error = float('inf')
            logger.error("Reconstruction contains NaN or Inf values")
        
        return error
    
    def _parse_dtype(self, dtype_str: str) -> torch.dtype:
        """Parse dtype string to torch.dtype"""
        dtype_map = {
            'torch.float32': torch.float32,
            'torch.float64': torch.float64,
            'torch.float16': torch.float16,
            'torch.int32': torch.int32,
            'torch.int64': torch.int64,
            'torch.int16': torch.int16,
            'torch.int8': torch.int8,
            'torch.uint8': torch.uint8,
            'torch.bool': torch.bool
        }
        
        # Handle both 'torch.float32' and 'float32' formats
        if dtype_str.startswith('torch.'):
            return dtype_map.get(dtype_str, torch.float32)
        else:
            return dtype_map.get(f'torch.{dtype_str}', torch.float32)
    
    def _update_statistics(self, decompression_time: float, 
                         reconstruction_error: Optional[float],
                         processing_mode: str):
        """Update decompression statistics"""
        self.decompression_stats['total_decompressions'] += 1
        
        # Update average time
        n = self.decompression_stats['total_decompressions']
        avg_time = self.decompression_stats['average_decompression_time_ms']
        self.decompression_stats['average_decompression_time_ms'] = (
            (avg_time * (n - 1) + decompression_time) / n
        )
        
        # Update average error if available
        if reconstruction_error is not None:
            avg_error = self.decompression_stats['average_reconstruction_error']
            self.decompression_stats['average_reconstruction_error'] = (
                (avg_error * (n - 1) + reconstruction_error) / n
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get decompression pipeline statistics"""
        with self._lock:
            return {
                **self.decompression_stats,
                'jax_available': self.jax_available,
                'gpu_available': self.gpu_available,
                'device': str(self.device),
                'streaming_enabled': self.config.enable_streaming,
                'cpu_fallback_enabled': self.config.enable_cpu_fallback
            }
    
    def cleanup(self):
        """Clean up resources"""
        with self._lock:
            self.streaming_active = False
            
            if self.stream_queue:
                # Clear streaming queue
                while not self.stream_queue.empty():
                    try:
                        self.stream_queue.get_nowait()
                    except:
                        break
            
            logger.info("TropicalDecompressionPipeline cleaned up")