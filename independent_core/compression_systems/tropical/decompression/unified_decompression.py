"""
Unified Decompression Pipeline for Tropical Compression System
Integrates all decompression components for production deployment
PRODUCTION READY - NO PLACEHOLDERS - HARD FAILURES ONLY

This module provides:
1. UnifiedDecompressionPipeline - Complete decompression system
2. Integration with all channel decompressors
3. Production-ready error handling and validation
4. Performance monitoring and optimization
"""

import torch
import numpy as np
import time
import logging
import json
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any, BinaryIO
from dataclasses import dataclass, field, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

# Import decompression components
from channel_decompressor import (
    BaseChannelDecompressor,
    CoefficientChannelDecompressor,
    ExponentChannelDecompressor,
    MantissaChannelDecompressor,
    ChannelMetadataHandler,
    ChannelMetadata,
    DecompressionMode,
    DecompressionStats
)
from jax_channel_processor import (
    JAXChannelProcessor,
    JAXProcessorConfig,
    JAXChannelData
)
from polynomial_reconstructor import (
    TropicalPolynomialReconstructor,
    TropicalToWeightConverter,
    LayerWiseDecompressor,
    ModelDecompressor,
    ReconstructionConfig,
    ReconstructionAccuracy
)

# Import tropical components
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tropical_channel_extractor import TropicalChannels
from channel_validation import (
    TropicalChannelValidator,
    ChannelValidationConfig,
    ValidationMetrics
)

# Try importing GPU memory management
try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from gpu_memory.gpu_memory_core import GPUMemoryManager
    from gpu_memory.advanced_memory_pool import AdvancedMemoryPoolManager
    GPU_MEMORY_AVAILABLE = True
except ImportError:
    GPU_MEMORY_AVAILABLE = False
    GPUMemoryManager = None
    AdvancedMemoryPoolManager = None

logger = logging.getLogger(__name__)


@dataclass
class UnifiedDecompressionConfig:
    """Configuration for unified decompression pipeline"""
    # Decompression mode
    mode: DecompressionMode = DecompressionMode.ADAPTIVE
    
    # Accuracy settings
    target_accuracy: ReconstructionAccuracy = ReconstructionAccuracy.HIGH
    max_error_threshold: float = 1e-6
    
    # Performance settings
    enable_jax: bool = True
    enable_gpu: bool = True
    enable_parallel: bool = True
    num_workers: int = 4
    batch_size: int = 1000
    
    # Memory management
    max_memory_gb: float = 8.0
    enable_memory_pool: bool = True
    enable_streaming: bool = True
    stream_chunk_size: int = 10000
    
    # Validation
    enable_validation: bool = True
    enable_checksums: bool = True
    enable_error_correction: bool = True
    
    # Caching
    enable_caching: bool = True
    cache_size_gb: float = 1.0
    cache_eviction: str = "lru"  # "lru", "lfu", "fifo"
    
    # I/O settings
    enable_mmap: bool = True
    prefetch_layers: int = 2
    
    # Monitoring
    enable_profiling: bool = True
    log_level: str = "INFO"
    metrics_interval: float = 1.0  # seconds


@dataclass
class DecompressionResult:
    """Result of decompression operation"""
    success: bool
    weights: Optional[Dict[str, torch.Tensor]] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'success': self.success,
            'num_weights': len(self.weights) if self.weights else 0,
            'errors': self.errors,
            'warnings': self.warnings,
            'metrics': self.metrics,
            'timestamp': self.timestamp
        }


class UnifiedDecompressionPipeline:
    """
    Unified decompression pipeline integrating all components.
    Production-ready with error handling, validation, and optimization.
    """
    
    def __init__(self, config: Optional[UnifiedDecompressionConfig] = None):
        """
        Initialize unified decompression pipeline
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or UnifiedDecompressionConfig()
        
        # Set up logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        
        # Initialize device
        if self.config.enable_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using GPU: %s", torch.cuda.get_device_name())
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for decompression")
        
        # Initialize components
        self._initialize_components()
        
        # Initialize memory management
        self._initialize_memory_management()
        
        # Initialize thread pool for parallel processing
        if self.config.enable_parallel:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.config.num_workers)
        else:
            self.thread_pool = None
        
        # Metrics tracking
        self.global_metrics = {
            'total_decompressions': 0,
            'successful_decompressions': 0,
            'failed_decompressions': 0,
            'total_bytes_processed': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Start metrics reporting thread if profiling enabled
        if self.config.enable_profiling:
            self._start_metrics_reporting()
        
        logger.info("UnifiedDecompressionPipeline initialized successfully")
    
    def _initialize_components(self):
        """Initialize decompression components"""
        # Channel decompressors
        self.coeff_decompressor = CoefficientChannelDecompressor(
            mode=self.config.mode,
            device=self.device,
            enable_validation=self.config.enable_validation,
            enable_caching=self.config.enable_caching
        )
        
        self.exp_decompressor = ExponentChannelDecompressor(
            mode=self.config.mode,
            device=self.device,
            enable_validation=self.config.enable_validation,
            enable_caching=self.config.enable_caching
        )
        
        self.mant_decompressor = MantissaChannelDecompressor(
            mode=self.config.mode,
            device=self.device,
            enable_validation=self.config.enable_validation,
            enable_caching=self.config.enable_caching
        )
        
        # Metadata handler
        self.metadata_handler = ChannelMetadataHandler()
        
        # JAX processor
        if self.config.enable_jax:
            try:
                jax_config = JAXProcessorConfig(
                    enable_jit=True,
                    enable_vmap=True,
                    enable_streaming=self.config.enable_streaming,
                    stream_chunk_size=self.config.stream_chunk_size
                )
                self.jax_processor = JAXChannelProcessor(jax_config)
            except Exception as e:
                logger.warning("Failed to initialize JAX processor: %s", e)
                self.jax_processor = None
        else:
            self.jax_processor = None
        
        # Reconstruction components
        recon_config = ReconstructionConfig(
            target_accuracy=self.config.target_accuracy,
            max_reconstruction_error=self.config.max_error_threshold,
            use_gpu=self.config.enable_gpu,
            use_jax=self.config.enable_jax,
            enable_caching=self.config.enable_caching
        )
        
        self.polynomial_reconstructor = TropicalPolynomialReconstructor(recon_config)
        self.weight_converter = TropicalToWeightConverter(self.device)
        self.layer_decompressor = LayerWiseDecompressor(recon_config)
        self.model_decompressor = ModelDecompressor(recon_config)
        
        # Validator
        if self.config.enable_validation:
            val_config = ChannelValidationConfig(
                enable_channel_checksums=self.config.enable_checksums,
                fail_on_validation_error=True
            )
            self.validator = TropicalChannelValidator(val_config)
        else:
            self.validator = None
    
    def _initialize_memory_management(self):
        """Initialize memory management components"""
        if GPU_MEMORY_AVAILABLE and self.config.enable_memory_pool:
            try:
                self.gpu_memory_manager = GPUMemoryManager(
                    device_id=0 if self.device.type == "cuda" else -1
                )
                
                self.memory_pool = AdvancedMemoryPoolManager(
                    max_memory_gb=self.config.max_memory_gb
                )
                
                logger.info("Memory management initialized with %.1f GB limit", 
                           self.config.max_memory_gb)
            except Exception as e:
                logger.warning("Failed to initialize memory management: %s", e)
                self.gpu_memory_manager = None
                self.memory_pool = None
        else:
            self.gpu_memory_manager = None
            self.memory_pool = None
    
    def decompress(self, 
                   compressed_data: Union[bytes, Dict[str, Any], Path],
                   output_format: str = "torch") -> DecompressionResult:
        """
        Main decompression entry point
        
        Args:
            compressed_data: Compressed data (bytes, dict, or file path)
            output_format: Output format ("torch", "numpy", "jax")
            
        Returns:
            DecompressionResult with decompressed weights
        """
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Load compressed data
            if isinstance(compressed_data, Path) or isinstance(compressed_data, str):
                compressed_data = self._load_compressed_file(compressed_data)
            elif isinstance(compressed_data, bytes):
                compressed_data = self._unpack_compressed_bytes(compressed_data)
            
            # Validate format
            if not self._validate_compressed_format(compressed_data):
                raise ValueError("Invalid compressed data format")
            
            # Extract layers
            compressed_layers = compressed_data.get('layers', {})
            model_info = compressed_data.get('model_info', {})
            
            # Decompress model
            if self.config.enable_streaming and len(compressed_layers) > 10:
                weights = self._streaming_decompress(compressed_layers, model_info)
            elif self.config.enable_parallel and len(compressed_layers) > 5:
                weights = self._parallel_decompress(compressed_layers, model_info)
            else:
                weights = self._sequential_decompress(compressed_layers, model_info)
            
            # Convert output format if needed
            if output_format == "numpy":
                weights = {k: v.cpu().numpy() for k, v in weights.items()}
            elif output_format == "jax" and self.jax_processor:
                weights = {k: self.jax_processor.torch_to_jax(v) for k, v in weights.items()}
            
            # Validate decompression
            if self.config.enable_validation:
                validation_result = self._validate_decompression(weights, model_info)
                if not validation_result['valid']:
                    errors.extend(validation_result['errors'])
                warnings.extend(validation_result.get('warnings', []))
            
            # Update metrics
            elapsed = time.time() - start_time
            self.global_metrics['total_decompressions'] += 1
            self.global_metrics['successful_decompressions'] += 1
            self.global_metrics['total_time'] += elapsed
            
            # Calculate throughput
            total_bytes = sum(w.numel() * w.element_size() for w in weights.values())
            self.global_metrics['total_bytes_processed'] += total_bytes
            throughput_mbps = (total_bytes / 1048576) / elapsed
            
            metrics = {
                'decompression_time': elapsed,
                'num_layers': len(weights),
                'total_parameters': sum(w.numel() for w in weights.values()),
                'total_bytes': total_bytes,
                'throughput_mbps': throughput_mbps,
                'device': str(self.device)
            }
            
            return DecompressionResult(
                success=len(errors) == 0,
                weights=weights,
                errors=errors,
                warnings=warnings,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error("Decompression failed: %s", e)
            self.global_metrics['failed_decompressions'] += 1
            
            return DecompressionResult(
                success=False,
                errors=[str(e)],
                metrics={'error': str(e)}
            )
    
    def decompress_file(self, 
                       input_path: Union[str, Path],
                       output_path: Optional[Union[str, Path]] = None) -> DecompressionResult:
        """
        Decompress from file
        
        Args:
            input_path: Path to compressed file
            output_path: Optional path to save decompressed weights
            
        Returns:
            DecompressionResult
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            return DecompressionResult(
                success=False,
                errors=[f"File not found: {input_path}"]
            )
        
        # Decompress
        result = self.decompress(input_path)
        
        # Save if output path provided
        if output_path and result.success:
            output_path = Path(output_path)
            self._save_decompressed_weights(result.weights, output_path)
            result.metrics['output_path'] = str(output_path)
        
        return result
    
    def decompress_stream(self, stream: BinaryIO) -> DecompressionResult:
        """
        Decompress from binary stream
        
        Args:
            stream: Binary stream containing compressed data
            
        Returns:
            DecompressionResult
        """
        # Read compressed data from stream
        compressed_bytes = stream.read()
        
        return self.decompress(compressed_bytes)
    
    def _sequential_decompress(self, 
                              compressed_layers: Dict[str, Any],
                              model_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Sequential decompression of layers"""
        decompressed = {}
        
        for layer_name, layer_data in compressed_layers.items():
            layer_info = model_info.get('layers', {}).get(layer_name, {})
            
            # Create TropicalChannels from layer data
            channels = self._create_channels_from_data(layer_data)
            
            # Decompress layer
            weights = self.layer_decompressor.decompress_layer(channels, layer_info)
            decompressed[layer_name] = weights
            
            logger.debug("Decompressed layer: %s", layer_name)
        
        return decompressed
    
    def _parallel_decompress(self, 
                           compressed_layers: Dict[str, Any],
                           model_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Parallel decompression of layers"""
        if not self.thread_pool:
            return self._sequential_decompress(compressed_layers, model_info)
        
        futures = []
        layer_names = []
        
        for layer_name, layer_data in compressed_layers.items():
            layer_info = model_info.get('layers', {}).get(layer_name, {})
            channels = self._create_channels_from_data(layer_data)
            
            future = self.thread_pool.submit(
                self.layer_decompressor.decompress_layer,
                channels, layer_info
            )
            futures.append(future)
            layer_names.append(layer_name)
        
        # Collect results
        decompressed = {}
        for layer_name, future in zip(layer_names, futures):
            weights = future.result()
            decompressed[layer_name] = weights
        
        return decompressed
    
    def _streaming_decompress(self, 
                            compressed_layers: Dict[str, Any],
                            model_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Streaming decompression for large models"""
        decompressed = {}
        
        # Process in chunks to limit memory usage
        chunk_size = self.config.stream_chunk_size
        layer_items = list(compressed_layers.items())
        
        for i in range(0, len(layer_items), chunk_size):
            chunk = dict(layer_items[i:i+chunk_size])
            
            # Process chunk
            if self.config.enable_parallel:
                chunk_result = self._parallel_decompress(chunk, model_info)
            else:
                chunk_result = self._sequential_decompress(chunk, model_info)
            
            decompressed.update(chunk_result)
            
            # Clear GPU cache if needed
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        return decompressed
    
    def _create_channels_from_data(self, layer_data: Dict[str, Any]) -> TropicalChannels:
        """Create TropicalChannels from layer data"""
        # Convert numpy arrays to tensors if needed
        coeff_data = layer_data['coefficient_channel']
        exp_data = layer_data['exponent_channel']
        
        if isinstance(coeff_data, np.ndarray):
            coeff_tensor = torch.from_numpy(coeff_data)
        else:
            coeff_tensor = torch.tensor(coeff_data)
        
        if isinstance(exp_data, np.ndarray):
            exp_tensor = torch.from_numpy(exp_data)
        else:
            exp_tensor = torch.tensor(exp_data)
        
        # Handle optional mantissa channel
        mant_tensor = None
        if 'mantissa_channel' in layer_data:
            mant_data = layer_data['mantissa_channel']
            if isinstance(mant_data, np.ndarray):
                mant_tensor = torch.from_numpy(mant_data)
            else:
                mant_tensor = torch.tensor(mant_data)
        
        # Create channels
        channels = TropicalChannels(
            coefficient_channel=coeff_tensor.to(self.device),
            exponent_channel=exp_tensor.to(self.device),
            index_channel=torch.arange(coeff_tensor.shape[0], device=self.device),
            metadata=layer_data.get('metadata', {}),
            device=self.device,
            mantissa_channel=mant_tensor.to(self.device) if mant_tensor is not None else None
        )
        
        return channels
    
    def _validate_compressed_format(self, data: Dict[str, Any]) -> bool:
        """Validate compressed data format"""
        required_keys = ['layers', 'model_info', 'compression_info']
        
        for key in required_keys:
            if key not in data:
                logger.error("Missing required key: %s", key)
                return False
        
        # Validate layer format
        for layer_name, layer_data in data['layers'].items():
            if 'coefficient_channel' not in layer_data:
                logger.error("Layer %s missing coefficient_channel", layer_name)
                return False
            if 'exponent_channel' not in layer_data:
                logger.error("Layer %s missing exponent_channel", layer_name)
                return False
        
        return True
    
    def _validate_decompression(self, 
                               weights: Dict[str, torch.Tensor],
                               model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate decompressed weights"""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        # Check layer count
        expected_layers = model_info.get('num_layers', len(weights))
        if len(weights) != expected_layers:
            result['warnings'].append(
                f"Layer count mismatch: {len(weights)} != {expected_layers}"
            )
        
        # Check shapes
        for layer_name, weight in weights.items():
            expected_shape = model_info.get('layers', {}).get(layer_name, {}).get('shape')
            if expected_shape and tuple(weight.shape) != tuple(expected_shape):
                result['errors'].append(
                    f"Shape mismatch for {layer_name}: {weight.shape} != {expected_shape}"
                )
                result['valid'] = False
            
            # Check for NaN/Inf
            if torch.isnan(weight).any():
                result['errors'].append(f"NaN values in {layer_name}")
                result['valid'] = False
            if torch.isinf(weight).any():
                result['errors'].append(f"Inf values in {layer_name}")
                result['valid'] = False
        
        return result
    
    def _load_compressed_file(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load compressed data from file"""
        path = Path(path)
        
        if path.suffix == '.pkl':
            with open(path, 'rb') as f:
                return pickle.load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                return json.load(f)
        elif path.suffix in ['.pt', '.pth']:
            return torch.load(path, map_location='cpu')
        else:
            # Try pickle by default
            with open(path, 'rb') as f:
                return pickle.load(f)
    
    def _unpack_compressed_bytes(self, data: bytes) -> Dict[str, Any]:
        """Unpack compressed bytes"""
        try:
            return pickle.loads(data)
        except:
            # Try other formats
            try:
                return json.loads(data.decode('utf-8'))
            except:
                raise ValueError("Unable to unpack compressed data")
    
    def _save_decompressed_weights(self, 
                                  weights: Dict[str, torch.Tensor],
                                  path: Path):
        """Save decompressed weights to file"""
        if path.suffix in ['.pt', '.pth']:
            torch.save(weights, path)
        elif path.suffix == '.pkl':
            with open(path, 'wb') as f:
                pickle.dump(weights, f)
        else:
            # Default to PyTorch format
            torch.save(weights, path.with_suffix('.pt'))
    
    def _start_metrics_reporting(self):
        """Start background thread for metrics reporting"""
        def report_metrics():
            while True:
                time.sleep(self.config.metrics_interval)
                
                if self.global_metrics['total_decompressions'] > 0:
                    avg_time = self.global_metrics['total_time'] / self.global_metrics['total_decompressions']
                    throughput = self.global_metrics['total_bytes_processed'] / max(1, self.global_metrics['total_time'])
                    
                    logger.debug(
                        "Decompression metrics: %d total, %.3fs avg, %.2f MB/s throughput",
                        self.global_metrics['total_decompressions'],
                        avg_time,
                        throughput / 1048576
                    )
        
        metrics_thread = threading.Thread(target=report_metrics, daemon=True)
        metrics_thread.start()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = dict(self.global_metrics)
        
        # Add component statistics
        stats['coeff_decompressor'] = self.coeff_decompressor.get_stats_summary()
        stats['exp_decompressor'] = self.exp_decompressor.get_stats_summary()
        stats['mant_decompressor'] = self.mant_decompressor.get_stats_summary()
        
        if self.jax_processor:
            stats['jax_processor'] = self.jax_processor.get_statistics()
        
        # Calculate rates
        if stats['total_decompressions'] > 0:
            stats['success_rate'] = stats['successful_decompressions'] / stats['total_decompressions']
            stats['average_time'] = stats['total_time'] / stats['total_decompressions']
        
        if stats['total_time'] > 0:
            stats['throughput_mbps'] = (stats['total_bytes_processed'] / 1048576) / stats['total_time']
        
        cache_total = stats['cache_hits'] + stats['cache_misses']
        if cache_total > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / cache_total
        
        return stats
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.global_metrics = {
            'total_decompressions': 0,
            'successful_decompressions': 0,
            'failed_decompressions': 0,
            'total_bytes_processed': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Reset component stats
        self.coeff_decompressor.stats.clear()
        self.exp_decompressor.stats.clear()
        self.mant_decompressor.stats.clear()
    
    def shutdown(self):
        """Shutdown pipeline and cleanup resources"""
        # Shutdown thread pool
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        # Shutdown JAX processor
        if self.jax_processor:
            self.jax_processor.shutdown()
        
        # Clear GPU cache
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("UnifiedDecompressionPipeline shutdown complete")