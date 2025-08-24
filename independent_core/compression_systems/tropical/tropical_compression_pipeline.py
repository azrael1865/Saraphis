"""
Tropical Compression Pipeline - JAX-accelerated compression with channel extraction
Integrates with existing FullCompressionPipeline architecture
NO PLACEHOLDERS - PRODUCTION CODE ONLY
HARD FAILURES ONLY - NO GRACEFUL DEGRADATION
"""

import torch
import numpy as np
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
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
    ChannelExtractionConfig
)
from .tropical_core import (
    TropicalNumber,
    TropicalMathematicalOperations,
    TROPICAL_ZERO,
    TROPICAL_EPSILON
)

# Import integration components
from ..integration.padic_tropical_bridge import (
    PadicTropicalConverter,
    ConversionConfig
)
from ..padic.padic_encoder import PadicWeight
from ..padic.padic_logarithmic_encoder import LogarithmicPadicWeight


@dataclass
class TropicalCompressionConfig:
    """Configuration for tropical compression pipeline"""
    # JAX configuration
    jax_config: JAXTropicalConfig = field(default_factory=JAXTropicalConfig)
    
    # Channel extraction
    enable_channel_extraction: bool = True
    channel_config: ChannelExtractionConfig = field(default_factory=ChannelExtractionConfig)
    
    # Batch processing
    batch_size: int = 10000
    enable_vmap: bool = True
    enable_pmap: bool = False  # Multi-GPU
    
    # Compression settings
    target_compression_ratio: float = 4.0
    quantization_bits: int = 8
    enable_sparsification: bool = True
    sparsity_threshold: float = 1e-8
    
    # Performance settings
    enable_gpu_acceleration: bool = True
    prefetch_batches: int = 2
    num_workers: int = 4
    
    # Bridge settings for p-adic compatibility
    enable_padic_bridge: bool = True
    conversion_config: ConversionConfig = field(default_factory=ConversionConfig)
    
    def __post_init__(self):
        """Validate configuration"""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.target_compression_ratio < 1.0:
            raise ValueError(f"target_compression_ratio must be >= 1.0, got {self.target_compression_ratio}")
        if not 1 <= self.quantization_bits <= 32:
            raise ValueError(f"quantization_bits must be in [1,32], got {self.quantization_bits}")


@dataclass
class TropicalCompressionResult:
    """Result from tropical compression"""
    compressed_channels: TropicalChannels
    compression_ratio: float
    original_size_bytes: int
    compressed_size_bytes: int
    compression_time_ms: float
    metadata: Dict[str, Any]
    jax_tensors: Optional[Dict[str, Any]] = None  # JAX arrays if available


class TropicalCompressionPipeline:
    """
    Tropical compression pipeline with JAX acceleration
    Integrates with existing FullCompressionPipeline architecture
    """
    
    def __init__(self, config: TropicalCompressionConfig):
        """Initialize tropical compression pipeline"""
        self.config = config
        
        # Initialize JAX engine - HARD FAILURE if JAX not available
        try:
            self.jax_engine = TropicalJAXEngine(config.jax_config)
            self.jax_available = True
            logger.info("TropicalCompressionPipeline initialized with JAX acceleration")
        except RuntimeError as e:
            raise RuntimeError(f"JAX initialization FAILED - HARD FAILURE: {e}")
        
        # Initialize channel processor
        self.channel_processor = JAXChannelProcessor(self.jax_engine)
        
        # Initialize channel extractor
        self.channel_extractor = TropicalChannelExtractor(config.channel_config)
        
        # Initialize p-adic bridge if enabled
        if config.enable_padic_bridge:
            self.padic_converter = PadicTropicalConverter(config.conversion_config)
        else:
            self.padic_converter = None
        
        # Initialize tropical operations
        device = torch.device('cuda' if config.enable_gpu_acceleration and torch.cuda.is_available() else 'cpu')
        self.tropical_ops = TropicalMathematicalOperations(device)
        self.device = device
        
        # Statistics tracking
        self.compression_stats = {
            'total_compressions': 0,
            'total_bytes_compressed': 0,
            'total_bytes_saved': 0,
            'average_compression_ratio': 0.0,
            'average_compression_time_ms': 0.0,
            'jax_compilations': 0,
            'batch_compressions': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Warm up JAX compilation
        self._warmup_jax()
    
    def _warmup_jax(self):
        """Warm up JAX JIT compilation"""
        try:
            # Create small test tensor
            test_tensor = torch.randn(10, 10, device=self.device)
            test_channels = self.channel_extractor.extract_channels(test_tensor)
            
            # Run through JAX pipeline to trigger compilation
            jax_channels = self.channel_processor.channels_to_jax(test_channels)
            _ = self.channel_processor.process_channels(
                jax_channels, 
                normalize=True,
                sparsify=False
            )
            
            self.compression_stats['jax_compilations'] += 1
            logger.info("JAX JIT compilation warmup completed")
            
        except Exception as e:
            raise RuntimeError(f"JAX warmup FAILED: {e}")
    
    def compress(self, tensor: torch.Tensor, 
                metadata: Optional[Dict[str, Any]] = None) -> TropicalCompressionResult:
        """
        Compress tensor using tropical mathematics with JAX acceleration
        
        Args:
            tensor: Input tensor to compress
            metadata: Optional metadata for compression hints
            
        Returns:
            TropicalCompressionResult with compressed data
        """
        if tensor is None:
            raise ValueError("Input tensor cannot be None - HARD FAILURE")
        if tensor.numel() == 0:
            raise ValueError("Input tensor cannot be empty - HARD FAILURE")
        
        start_time = time.perf_counter()
        
        with self._lock:
            try:
                # Record original size
                original_size = tensor.numel() * tensor.element_size()
                original_shape = tensor.shape
                original_dtype = tensor.dtype
                
                # Extract tropical channels
                extraction_start = time.perf_counter()
                channels = self._extract_channels(tensor)
                extraction_time = (time.perf_counter() - extraction_start) * 1000
                
                # Convert to JAX and process
                jax_start = time.perf_counter()
                compressed_channels = self._process_with_jax(channels)
                jax_time = (time.perf_counter() - jax_start) * 1000
                
                # Apply quantization and sparsification
                quantization_start = time.perf_counter()
                compressed_channels = self._apply_compression(compressed_channels)
                quantization_time = (time.perf_counter() - quantization_start) * 1000
                
                # Calculate compressed size
                compressed_size = self._calculate_compressed_size(compressed_channels)
                compression_ratio = original_size / compressed_size
                
                # Validate compression ratio
                if compression_ratio < 1.0:
                    raise RuntimeError(f"Compression failed: ratio {compression_ratio:.2f} < 1.0 - HARD FAILURE")
                
                total_time = (time.perf_counter() - start_time) * 1000
                
                # Update statistics
                self._update_statistics(original_size, compressed_size, total_time)
                
                # Build result
                result = TropicalCompressionResult(
                    compressed_channels=compressed_channels,
                    compression_ratio=compression_ratio,
                    original_size_bytes=original_size,
                    compressed_size_bytes=compressed_size,
                    compression_time_ms=total_time,
                    metadata={
                        'original_shape': original_shape,
                        'original_dtype': str(original_dtype),
                        'extraction_time_ms': extraction_time,
                        'jax_processing_time_ms': jax_time,
                        'quantization_time_ms': quantization_time,
                        'device': str(self.device),
                        'batch_size': self.config.batch_size,
                        **(metadata or {})
                    }
                )
                
                logger.info(f"Tropical compression completed: {compression_ratio:.2f}x ratio in {total_time:.1f}ms")
                return result
                
            except Exception as e:
                raise RuntimeError(f"Tropical compression FAILED: {e}")
    
    def compress_batch(self, tensors: List[torch.Tensor],
                      metadata: Optional[List[Dict[str, Any]]] = None) -> List[TropicalCompressionResult]:
        """
        Compress batch of tensors with vmap/pmap acceleration
        
        Args:
            tensors: List of tensors to compress
            metadata: Optional list of metadata dicts
            
        Returns:
            List of TropicalCompressionResult objects
        """
        if not tensors:
            raise ValueError("Empty tensor list - HARD FAILURE")
        
        batch_start = time.perf_counter()
        results = []
        
        with self._lock:
            try:
                # Process in configured batch sizes
                batch_size = self.config.batch_size
                num_batches = (len(tensors) + batch_size - 1) // batch_size
                
                for batch_idx in range(num_batches):
                    batch_tensors = tensors[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                    batch_metadata = None
                    if metadata:
                        batch_metadata = metadata[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                    
                    # Stack tensors for batch processing if same shape
                    if all(t.shape == batch_tensors[0].shape for t in batch_tensors):
                        # Can use vmap for same-shaped tensors
                        batch_results = self._compress_batch_vmap(batch_tensors, batch_metadata)
                        results.extend(batch_results)
                    else:
                        # Process individually for different shapes
                        for i, tensor in enumerate(batch_tensors):
                            meta = batch_metadata[i] if batch_metadata else None
                            result = self.compress(tensor, meta)
                            results.append(result)
                
                batch_time = (time.perf_counter() - batch_start) * 1000
                self.compression_stats['batch_compressions'] += 1
                
                logger.info(f"Batch compression completed: {len(tensors)} tensors in {batch_time:.1f}ms")
                return results
                
            except Exception as e:
                raise RuntimeError(f"Batch compression FAILED: {e}")
    
    def _compress_batch_vmap(self, tensors: List[torch.Tensor],
                           metadata: Optional[List[Dict[str, Any]]]) -> List[TropicalCompressionResult]:
        """Compress batch using vmap acceleration"""
        # Stack tensors
        stacked = torch.stack(tensors)
        
        # Extract channels for entire batch
        batch_channels = self.channel_extractor.extract_channels_batch(stacked)
        
        # Convert to JAX
        jax_channels = self.channel_processor.channels_to_jax(batch_channels)
        
        # Process with vmap
        if self.config.enable_vmap:
            processed = self.channel_processor.vmap_process_channels(
                jax_channels,
                normalize=True,
                sparsify=self.config.enable_sparsification
            )
        else:
            processed = self.channel_processor.process_channels(
                jax_channels,
                normalize=True,
                sparsify=self.config.enable_sparsification
            )
        
        # Split results back into individual compressions
        results = []
        for i in range(len(tensors)):
            # Extract individual result
            individual_channels = TropicalChannels(
                sign_channel=batch_channels.sign_channel[i],
                exponent_channel=batch_channels.exponent_channel[i],
                mantissa_channel=batch_channels.mantissa_channel[i],
                metadata=batch_channels.metadata.copy()
            )
            
            original_size = tensors[i].numel() * tensors[i].element_size()
            compressed_size = self._calculate_compressed_size(individual_channels)
            
            result = TropicalCompressionResult(
                compressed_channels=individual_channels,
                compression_ratio=original_size / compressed_size,
                original_size_bytes=original_size,
                compressed_size_bytes=compressed_size,
                compression_time_ms=0.0,  # Batch time divided later
                metadata={'batch_processed': True, **(metadata[i] if metadata else {})}
            )
            results.append(result)
        
        return results
    
    def _extract_channels(self, tensor: torch.Tensor) -> TropicalChannels:
        """Extract tropical channels from tensor"""
        # Move to appropriate device
        if tensor.device != self.device:
            tensor = tensor.to(self.device)
        
        # Extract channels
        channels = self.channel_extractor.extract_channels(tensor)
        
        # Validate channels
        if channels.sign_channel is None or channels.exponent_channel is None:
            raise RuntimeError("Channel extraction failed - HARD FAILURE")
        
        return channels
    
    def _process_with_jax(self, channels: TropicalChannels) -> TropicalChannels:
        """Process channels using JAX acceleration"""
        # Convert to JAX arrays
        jax_channels = self.channel_processor.channels_to_jax(channels)
        
        # Process channels with JAX operations
        processed = self.channel_processor.process_channels(
            jax_channels,
            normalize=True,
            sparsify=self.config.enable_sparsification,
            sparsity_threshold=self.config.sparsity_threshold
        )
        
        # Convert back to TropicalChannels
        return self.channel_processor.jax_to_channels(processed)
    
    def _apply_compression(self, channels: TropicalChannels) -> TropicalChannels:
        """Apply quantization and compression to channels"""
        # Quantize mantissa channel
        if self.config.quantization_bits < 32:
            channels.mantissa_channel = self._quantize_channel(
                channels.mantissa_channel,
                self.config.quantization_bits
            )
        
        # Apply sparsification
        if self.config.enable_sparsification:
            channels = self._sparsify_channels(channels)
        
        return channels
    
    def _quantize_channel(self, channel: torch.Tensor, bits: int) -> torch.Tensor:
        """Quantize channel to specified bits"""
        # Find min/max for quantization range
        min_val = channel.min()
        max_val = channel.max()
        
        if max_val == min_val:
            return torch.zeros_like(channel)
        
        # Quantize to n bits
        levels = 2 ** bits - 1
        scale = (max_val - min_val) / levels
        
        # Quantize and dequantize
        quantized = torch.round((channel - min_val) / scale)
        quantized = torch.clamp(quantized, 0, levels)
        dequantized = quantized * scale + min_val
        
        return dequantized
    
    def _sparsify_channels(self, channels: TropicalChannels) -> TropicalChannels:
        """Apply sparsification to channels"""
        threshold = self.config.sparsity_threshold
        
        # Sparsify mantissa channel
        mask = torch.abs(channels.mantissa_channel) > threshold
        channels.mantissa_channel = channels.mantissa_channel * mask
        
        # Record sparsity in metadata
        sparsity = 1.0 - mask.float().mean().item()
        channels.metadata['sparsity'] = sparsity
        
        return channels
    
    def _calculate_compressed_size(self, channels: TropicalChannels) -> int:
        """Calculate compressed size in bytes"""
        size = 0
        
        # Sign channel: 1 bit per element
        size += channels.sign_channel.numel() // 8
        
        # Exponent channel: 8 bits per element
        size += channels.exponent_channel.numel()
        
        # Mantissa channel: depends on quantization
        if self.config.quantization_bits < 32:
            size += channels.mantissa_channel.numel() * (self.config.quantization_bits // 8)
        else:
            size += channels.mantissa_channel.numel() * 4  # float32
        
        # Account for sparsity (only store non-zero values)
        if 'sparsity' in channels.metadata:
            sparsity = channels.metadata['sparsity']
            size = int(size * (1 - sparsity))
        
        return max(1, size)  # At least 1 byte
    
    def _update_statistics(self, original_size: int, compressed_size: int, time_ms: float):
        """Update compression statistics"""
        self.compression_stats['total_compressions'] += 1
        self.compression_stats['total_bytes_compressed'] += original_size
        self.compression_stats['total_bytes_saved'] += (original_size - compressed_size)
        
        # Update averages
        n = self.compression_stats['total_compressions']
        avg_ratio = self.compression_stats['average_compression_ratio']
        avg_time = self.compression_stats['average_compression_time_ms']
        
        new_ratio = original_size / compressed_size
        self.compression_stats['average_compression_ratio'] = (avg_ratio * (n-1) + new_ratio) / n
        self.compression_stats['average_compression_time_ms'] = (avg_time * (n-1) + time_ms) / n
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get compression pipeline statistics"""
        with self._lock:
            return {
                **self.compression_stats,
                'jax_device_count': len(self.jax_engine.devices) if self.jax_available else 0,
                'gpu_available': self.jax_engine.gpu_available if self.jax_available else False,
                'device': str(self.device)
            }
    
    def convert_from_padic(self, padic_weights: List[Union[PadicWeight, LogarithmicPadicWeight]],
                          shape: Tuple[int, ...]) -> TropicalCompressionResult:
        """
        Convert p-adic weights to tropical compression
        
        Args:
            padic_weights: List of p-adic weights
            shape: Original tensor shape
            
        Returns:
            TropicalCompressionResult
        """
        if not self.padic_converter:
            raise RuntimeError("P-adic bridge not enabled - HARD FAILURE")
        
        start_time = time.perf_counter()
        
        # Extract base p-adic weights if logarithmic
        base_weights = []
        for weight in padic_weights:
            if isinstance(weight, LogarithmicPadicWeight):
                base_weights.append(weight.padic_weight)
            else:
                base_weights.append(weight)
        
        # Convert to tropical tensor
        tropical_tensor = self.padic_converter.tensor_padic_to_tropical(base_weights, shape)
        
        # Compress tropical tensor
        result = self.compress(tropical_tensor)
        
        # Add conversion metadata
        result.metadata['converted_from_padic'] = True
        result.metadata['conversion_time_ms'] = (time.perf_counter() - start_time) * 1000
        
        return result
    
    def cleanup(self):
        """Clean up resources"""
        with self._lock:
            if self.padic_converter:
                self.padic_converter.clear_cache()
            
            # Clear JAX compilation cache if needed
            if hasattr(self.jax_engine, 'compilation_cache'):
                self.jax_engine.compilation_cache.clear()
            
            logger.info("TropicalCompressionPipeline cleaned up")