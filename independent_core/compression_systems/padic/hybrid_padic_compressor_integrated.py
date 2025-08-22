"""
Hybrid P-adic Compression System with Memory Pressure Handler Integration
GPU-accelerated two-channel compression with intelligent GPU/CPU coordination
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import gc

from base.compression_base import CompressionAlgorithm, CompressionValidator, CompressionMetrics
from padic_encoder import PadicWeight, PadicValidation, PadicMathematicalOperations
from hybrid_padic_structures import (
    HybridPadicWeight,
    HybridPadicValidator,
    HybridPadicConverter,
    HybridPadicManager
)
from memory_pressure_handler import (
    MemoryPressureHandler,
    PressureHandlerConfig,
    ProcessingMode,
    integrate_memory_pressure_handler
)


class HybridPadicCompressionSystemIntegrated(CompressionAlgorithm):
    """GPU-accelerated hybrid p-adic compression with memory pressure handling"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize hybrid p-adic compression system with memory pressure handler"""
        super().__init__(config)
        
        # Extract and validate configuration
        self._extract_config()
        self._validate_full_config()
        
        # Initialize components
        self.math_ops = PadicMathematicalOperations(self.prime, self.precision)
        self.validator = PadicValidation()
        self.compression_validator = CompressionValidator()
        self.metrics_calculator = CompressionMetrics()
        
        # Initialize hybrid components
        self.hybrid_manager = HybridPadicManager(config)
        self.hybrid_validator = HybridPadicValidator()
        self.hybrid_converter = HybridPadicConverter()
        
        # Initialize GPU memory management
        self.gpu_memory_optimizer = None
        self.cpu_bursting_pipeline = None
        self.memory_pressure_handler = None
        
        self._initialize_memory_systems()
        
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for hybrid p-adic compression")
        
        # Performance tracking with memory pressure stats
        self.performance_stats = {
            'total_compressions': 0,
            'total_decompressions': 0,
            'hybrid_compressions': 0,
            'pure_padic_compressions': 0,
            'gpu_decompressions': 0,
            'cpu_decompressions': 0,
            'memory_pressure_events': 0,
            'average_compression_time': 0.0,
            'average_decompression_time': 0.0,
            'average_hybrid_time': 0.0,
            'average_pure_time': 0.0,
            'peak_memory_usage': 0,
            'gpu_memory_usage': 0
        }
        
        # Dynamic switching management
        self.dynamic_switching_manager = None
        if self.enable_dynamic_switching:
            self._initialize_dynamic_switching()
    
    def _initialize_memory_systems(self) -> None:
        """Initialize memory management systems"""
        try:
            # Initialize GPU memory optimizer
            from ..gpu_memory.gpu_memory_core import GPUMemoryOptimizer
            self.gpu_memory_optimizer = GPUMemoryOptimizer(self.config.get('gpu_config', {}))
            
            # Initialize CPU bursting pipeline
            from ..gpu_memory.cpu_bursting_pipeline import CPU_BurstingPipeline, CPUBurstingConfig
            cpu_config = CPUBurstingConfig(
                gpu_memory_threshold_mb=self.config.get('gpu_memory_threshold_mb', 100),
                memory_pressure_threshold=self.config.get('memory_pressure_threshold', 0.9),
                num_cpu_workers=self.config.get('cpu_workers', -1)
            )
            
            # Get decompression engine (will be created later if needed)
            if hasattr(self, 'decompression_engine'):
                self.cpu_bursting_pipeline = CPU_BurstingPipeline(cpu_config, self.decompression_engine)
            
            # Initialize memory pressure handler
            pressure_config = PressureHandlerConfig(
                gpu_critical_threshold_mb=self.config.get('gpu_critical_mb', 100),
                gpu_high_threshold_mb=self.config.get('gpu_high_mb', 500),
                gpu_moderate_threshold_mb=self.config.get('gpu_moderate_mb', 1000),
                gpu_critical_utilization=self.config.get('gpu_critical_util', 0.95),
                gpu_high_utilization=self.config.get('gpu_high_util', 0.85),
                gpu_moderate_utilization=self.config.get('gpu_moderate_util', 0.70),
                force_cpu_on_critical=self.config.get('force_cpu_on_critical', True),
                adaptive_threshold=self.config.get('adaptive_threshold', True)
            )
            
            self.memory_pressure_handler = MemoryPressureHandler(
                pressure_config,
                self.gpu_memory_optimizer,
                self.cpu_bursting_pipeline
            )
            
        except ImportError as e:
            # Hard failure if memory systems not available
            raise RuntimeError(f"Failed to initialize memory systems: {e}")
    
    def _extract_config(self) -> None:
        """Extract configuration parameters"""
        # Required parameters
        required_params = ['prime', 'precision', 'chunk_size', 'gpu_memory_limit_mb']
        for param in required_params:
            if param not in self.config:
                raise KeyError(f"Missing required configuration parameter: {param}")
        
        self.prime = self.config['prime']
        self.precision = self.config['precision']
        self.chunk_size = self.config['chunk_size']
        self.gpu_memory_limit = self.config['gpu_memory_limit_mb'] * 1024 * 1024
        
        # Hybrid-specific parameters
        self.enable_hybrid = self.config.get('enable_hybrid', True)
        self.hybrid_threshold = self.config.get('hybrid_threshold', 1000)
        self.force_hybrid = self.config.get('force_hybrid', False)
        self.enable_dynamic_switching = self.config.get('enable_dynamic_switching', True)
        self.require_gpu_for_hybrid = self.config.get('require_gpu_for_hybrid', True)
        
        # Memory pressure parameters
        self.enable_memory_pressure_handling = self.config.get('enable_memory_pressure', True)
        self.memory_pressure_mode = self.config.get('memory_pressure_mode', 'adaptive')
        
        # Optional parameters
        self.preserve_ultrametric = self.config.get('preserve_ultrametric', True)
        self.validate_reconstruction = self.config.get('validate_reconstruction', True)
        self.max_reconstruction_error = self.config.get('max_reconstruction_error', 1e-6)
        self.enable_gc = self.config.get('enable_gc', True)
    
    def decompress(self, compressed: Dict[str, Any]) -> torch.Tensor:
        """Decompress hybrid p-adic compressed data with memory pressure handling"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if compressed is None:
            raise ValueError("Compressed data cannot be None")
        if not isinstance(compressed, dict):
            raise TypeError(f"Compressed data must be dict, got {type(compressed)}")
        
        required_keys = ['encoded_data', 'metadata', 'original_shape', 'original_dtype']
        for key in required_keys:
            if key not in compressed:
                raise KeyError(f"Missing required key in compressed data: {key}")
        
        start_time = time.time()
        
        # Extract metadata
        metadata = compressed['metadata']
        
        # Add size information for memory pressure decision
        total_elements = metadata.get('total_elements', len(compressed['encoded_data']))
        element_size = 4  # float32
        data_size_mb = (total_elements * element_size) / (1024 * 1024)
        
        # Prepare metadata for memory pressure handler
        pressure_metadata = {
            'size_mb': data_size_mb,
            'priority': metadata.get('priority', 'normal'),
            'require_gpu': metadata.get('require_gpu', False),
            'require_cpu': metadata.get('require_cpu', False),
            'original_shape': compressed['original_shape'],
            'dtype': compressed['original_dtype'],
            'device': metadata.get('device', 'cuda:0')
        }
        
        # Check memory pressure if enabled
        use_cpu = False
        decision_info = {}
        
        if self.enable_memory_pressure_handling and self.memory_pressure_handler:
            use_cpu, decision_info = self.memory_pressure_handler.should_use_cpu(pressure_metadata)
            
            if use_cpu:
                self.performance_stats['cpu_decompressions'] += 1
                self.performance_stats['memory_pressure_events'] += 1
            else:
                self.performance_stats['gpu_decompressions'] += 1
        
        # Decode data based on decision
        if use_cpu and self.cpu_bursting_pipeline:
            # Use CPU decompression
            result = self._decode_with_cpu(compressed['encoded_data'], metadata)
        else:
            # Use GPU decompression (existing path)
            result = self.decode(compressed['encoded_data'], metadata)
        
        # Update performance metrics
        decompression_time = time.time() - start_time
        if self.memory_pressure_handler:
            throughput = total_elements / decompression_time
            latency_ms = decompression_time * 1000
            mode = 'cpu' if use_cpu else 'gpu'
            self.memory_pressure_handler.update_performance(mode, True, throughput, latency_ms)
        
        # Restore original shape and dtype
        result = result.reshape(compressed['original_shape'])
        dtype_str = compressed['original_dtype'].split('.')[-1]
        if hasattr(torch, dtype_str):
            dtype = getattr(torch, dtype_str)
            result = result.to(dtype)
        
        # Update statistics
        self.performance_stats['total_decompressions'] += 1
        avg_time = self.performance_stats['average_decompression_time']
        n = self.performance_stats['total_decompressions']
        self.performance_stats['average_decompression_time'] = (avg_time * (n-1) + decompression_time) / n
        
        return result
    
    def _decode_with_cpu(self, encoded_data: List[Union[PadicWeight, HybridPadicWeight]], 
                        metadata: Dict[str, Any]) -> torch.Tensor:
        """Decode using CPU when memory pressure is high"""
        # Ensure we're working with CPU tensors
        original_device = metadata.get('device', 'cuda:0')
        metadata['device'] = 'cpu'
        
        # Check compression type
        compression_type = metadata.get('compression_type', 'hybrid')
        
        if compression_type == 'hybrid':
            # Convert hybrid weights to CPU format
            cpu_encoded_data = []
            for weight in encoded_data:
                if isinstance(weight, HybridPadicWeight):
                    # Move channels to CPU
                    cpu_weight = HybridPadicWeight(
                        exponent_channel=weight.exponent_channel.cpu(),
                        mantissa_channel=weight.mantissa_channel.cpu(),
                        validation_checksum=weight.validation_checksum,
                        ultrametric_norm=weight.ultrametric_norm
                    )
                    cpu_encoded_data.append(cpu_weight)
                else:
                    cpu_encoded_data.append(weight)
            
            # Decode on CPU
            result = self._decode_hybrid(cpu_encoded_data, metadata)
        else:
            # Pure p-adic - already CPU compatible
            result = self._decode_pure_padic(encoded_data, metadata)
        
        # Move result back to original device if needed
        if original_device != 'cpu':
            result = result.to(torch.device(original_device))
        
        return result
    
    def decode(self, encoded_data: List[Union[PadicWeight, HybridPadicWeight]], 
              metadata: Dict[str, Any]) -> torch.Tensor:
        """Decode compressed data (existing method)"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if encoded_data is None:
            raise ValueError("Encoded data cannot be None")
        if metadata is None:
            raise ValueError("Metadata cannot be None")
        
        compression_type = metadata.get('compression_type', 'hybrid')
        
        if compression_type == 'hybrid':
            return self._decode_hybrid(encoded_data, metadata)
        elif compression_type == 'pure':
            return self._decode_pure_padic(encoded_data, metadata)
        else:
            raise ValueError(f"Unknown compression type: {compression_type}")
    
    def _decode_hybrid(self, encoded_data: List[HybridPadicWeight], 
                      metadata: Dict[str, Any]) -> torch.Tensor:
        """Decode hybrid p-adic representation"""
        # Implementation from original file
        if encoded_data is None:
            raise ValueError("Encoded data cannot be None")
        if not isinstance(encoded_data, list):
            raise TypeError(f"Encoded data must be list, got {type(encoded_data)}")
        if not encoded_data:
            raise ValueError("Encoded data cannot be empty")
        
        # Validate all weights are hybrid
        for i, weight in enumerate(encoded_data):
            if not isinstance(weight, HybridPadicWeight):
                raise TypeError(f"Weight {i} must be HybridPadicWeight, got {type(weight)}")
            self.hybrid_validator.validate_hybrid_weight(weight)
        
        # Convert hybrid weights back to pure p-adic
        pure_padic_weights = []
        for hybrid_weight in encoded_data:
            padic_weight = self.hybrid_manager.restore_padic_weight(hybrid_weight)
            pure_padic_weights.append(padic_weight)
        
        # Use existing pure p-adic decoding logic
        return self._decode_pure_padic(pure_padic_weights, metadata)
    
    def _decode_pure_padic(self, encoded_data: List[PadicWeight], 
                          metadata: Dict[str, Any]) -> torch.Tensor:
        """Decode pure p-adic representation"""
        # Implementation from original file
        if encoded_data is None:
            raise ValueError("Encoded data cannot be None")
        if not isinstance(encoded_data, list):
            raise TypeError(f"Encoded data must be list, got {type(encoded_data)}")
        if not encoded_data:
            raise ValueError("Encoded data cannot be empty")
        
        # Validate metadata
        required_keys = {'original_shape', 'prime', 'precision', 'dtype', 'device', 'total_elements'}
        missing_keys = required_keys - set(metadata.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys in metadata: {missing_keys}")
        
        # Validate configuration matches
        if metadata['prime'] != self.prime:
            raise ValueError(f"Prime mismatch: {metadata['prime']} != {self.prime}")
        if metadata['precision'] != self.precision:
            raise ValueError(f"Precision mismatch: {metadata['precision']} != {self.precision}")
        if metadata['total_elements'] != len(encoded_data):
            raise ValueError(f"Element count mismatch: {metadata['total_elements']} != {len(encoded_data)}")
        
        # Convert p-adic weights back to floats
        float_values = []
        processed_count = 0
        
        for i in range(0, len(encoded_data), self.chunk_size):
            chunk = encoded_data[i:i+self.chunk_size]
            chunk_floats = []
            
            for j, pw in enumerate(chunk):
                try:
                    float_val = self.math_ops.from_padic(pw)
                    chunk_floats.append(float_val)
                except Exception as e:
                    raise ValueError(f"Failed to convert p-adic weight at position {i + j}: {e}")
            
            float_values.extend(chunk_floats)
            processed_count += len(chunk)
            
            # Force garbage collection if enabled
            if self.enable_gc and processed_count % (self.chunk_size * 10) == 0:
                gc.collect()
        
        # Validate all weights were converted
        if len(float_values) != len(encoded_data):
            raise ValueError(f"Conversion count mismatch: {len(float_values)} != {len(encoded_data)}")
        
        # Reconstruct tensor
        try:
            # Parse dtype and device
            dtype_str = metadata['dtype'].split('.')[-1]
            if not hasattr(torch, dtype_str):
                raise ValueError(f"Invalid dtype: {metadata['dtype']}")
            dtype = getattr(torch, dtype_str)
            
            device = torch.device(metadata['device'])
            
            # Create tensor
            tensor = torch.tensor(float_values, dtype=dtype, device='cpu')
            tensor = tensor.reshape(metadata['original_shape'])
            tensor = tensor.to(device)
            
        except Exception as e:
            raise ValueError(f"Failed to reconstruct tensor: {e}")
        
        return tensor
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory pressure summary"""
        if self.memory_pressure_handler:
            return self.memory_pressure_handler.get_memory_summary()
        return {}
    
    def set_processing_mode(self, mode: str) -> None:
        """Set processing mode for memory pressure handler"""
        if self.memory_pressure_handler:
            mode_map = {
                'gpu_preferred': ProcessingMode.GPU_PREFERRED,
                'cpu_preferred': ProcessingMode.CPU_PREFERRED,
                'adaptive': ProcessingMode.ADAPTIVE
            }
            if mode in mode_map:
                self.memory_pressure_handler.set_processing_mode(mode_map[mode])
            else:
                raise ValueError(f"Invalid processing mode: {mode}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics including memory pressure"""
        stats = dict(self.performance_stats)
        
        if self.memory_pressure_handler:
            pressure_stats = self.memory_pressure_handler.get_statistics()
            stats['memory_pressure'] = pressure_stats
        
        return stats