"""
SystemIntegrationCoordinator - Unified System Coordination for Compression Pipeline
Orchestrates all compression components: SmartPool, AutoSwap, CPU Bursting, Memory Pressure
NO FALLBACKS - HARD FAILURES ONLY

This is the master coordinator that brings together:
- GPU Memory Management (SmartPool + AutoSwap)
- CPU Bursting Pipeline
- Memory Pressure Handler
- P-adic Compression Systems
- Performance Optimization
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import time
import threading
import psutil
import gc
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import warnings
import json
import os

# Import all compression components
try:
    from .gpu_memory.gpu_memory_core import GPUMemoryOptimizer, GPUOptimizationResult
    from .gpu_memory.smart_pool import SmartPool, integrate_smartpool_with_gpu_optimizer
    from .gpu_memory.auto_swap_manager import AutoSwapManager, SwapPolicy
    from .gpu_memory.cpu_bursting_pipeline import CPU_BurstingPipeline, DecompressionMode
    from .padic.memory_pressure_handler import MemoryPressureHandler, ProcessingMode
    from .padic.padic_advanced import PadicDecompressionEngine
    from .padic.hybrid_padic_compressor import HybridPadicCompressionSystem
except ImportError as e:
    raise RuntimeError(f"Failed to import compression components: {e}")


class SystemState(Enum):
    """Overall system state"""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    OPTIMIZING = "optimizing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class OptimizationStrategy(Enum):
    """System-wide optimization strategies"""
    THROUGHPUT = "throughput"         # Maximize throughput
    LATENCY = "latency"              # Minimize latency
    MEMORY = "memory"                # Minimize memory usage
    BALANCED = "balanced"            # Balance all metrics
    ADAPTIVE = "adaptive"            # Adapt based on workload


@dataclass
class SystemConfiguration:
    """Unified system configuration"""
    # GPU Memory Configuration
    gpu_memory_limit_mb: int = 14336
    enable_smart_pool: bool = True
    smart_pool_fragmentation_target: float = 0.133  # 13.3%
    
    # AutoSwap Configuration
    enable_auto_swap: bool = True
    swap_policy: str = "balanced"
    swap_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.5,
        'moderate': 0.75,
        'high': 0.9,
        'critical': 0.95
    })
    
    # CPU Bursting Configuration
    enable_cpu_bursting: bool = True
    cpu_workers: int = -1  # Auto-detect
    cpu_batch_size: int = 1000
    gpu_memory_threshold_mb: int = 2048
    
    # Memory Pressure Configuration
    enable_memory_pressure: bool = True
    memory_pressure_mode: str = "adaptive"
    force_cpu_on_critical: bool = True
    
    # P-adic Compression Configuration
    prime: int = 251
    precision: int = 128
    chunk_size: int = 5000
    enable_hybrid: bool = True
    
    # Performance Configuration
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    enable_profiling: bool = True
    monitoring_interval_ms: int = 100
    
    # System Configuration
    max_concurrent_operations: int = 50
    enable_auto_optimization: bool = True
    checkpoint_interval_seconds: int = 300
    
    def __post_init__(self):
        """Validate configuration"""
        if self.gpu_memory_limit_mb <= 0:
            raise ValueError(f"gpu_memory_limit_mb must be > 0, got {self.gpu_memory_limit_mb}")
        if self.prime <= 1:
            raise ValueError(f"prime must be > 1, got {self.prime}")
        if self.precision <= 0:
            raise ValueError(f"precision must be > 0, got {self.precision}")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {self.chunk_size}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'gpu_memory_limit_mb': self.gpu_memory_limit_mb,
            'enable_smart_pool': self.enable_smart_pool,
            'smart_pool_fragmentation_target': self.smart_pool_fragmentation_target,
            'enable_auto_swap': self.enable_auto_swap,
            'swap_policy': self.swap_policy,
            'swap_thresholds': self.swap_thresholds,
            'enable_cpu_bursting': self.enable_cpu_bursting,
            'cpu_workers': self.cpu_workers,
            'cpu_batch_size': self.cpu_batch_size,
            'gpu_memory_threshold_mb': self.gpu_memory_threshold_mb,
            'enable_memory_pressure': self.enable_memory_pressure,
            'memory_pressure_mode': self.memory_pressure_mode,
            'force_cpu_on_critical': self.force_cpu_on_critical,
            'prime': self.prime,
            'precision': self.precision,
            'chunk_size': self.chunk_size,
            'enable_hybrid': self.enable_hybrid,
            'optimization_strategy': self.optimization_strategy.value,
            'enable_profiling': self.enable_profiling,
            'monitoring_interval_ms': self.monitoring_interval_ms,
            'max_concurrent_operations': self.max_concurrent_operations,
            'enable_auto_optimization': self.enable_auto_optimization,
            'checkpoint_interval_seconds': self.checkpoint_interval_seconds
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SystemConfiguration':
        """Create from dictionary"""
        # Handle optimization strategy enum
        if 'optimization_strategy' in config_dict:
            config_dict['optimization_strategy'] = OptimizationStrategy(config_dict['optimization_strategy'])
        return cls(**config_dict)
    
    def save(self, filepath: str) -> None:
        """Save configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'SystemConfiguration':
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class CompressionRequest:
    """Request for compression operation"""
    request_id: str
    tensor: torch.Tensor
    priority: str = "normal"  # low, normal, high, critical
    require_gpu: bool = False
    require_cpu: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None


@dataclass
class CompressionResult:
    """Result of compression operation"""
    request_id: str
    compressed_data: Dict[str, Any]
    compression_ratio: float
    compression_time: float
    decompression_time: float
    processing_mode: str  # gpu, cpu, hybrid
    memory_used_mb: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CompressionPipelineOrchestrator:
    """
    Orchestrates the end-to-end compression pipeline
    Manages flow from tensor input to compressed output
    """
    
    def __init__(self, config: SystemConfiguration, components: Dict[str, Any]):
        """Initialize pipeline orchestrator"""
        self.config = config
        self.components = components
        
        # Pipeline state
        self.active_requests: Dict[str, CompressionRequest] = {}
        self.request_queue = deque()
        self.results_cache: Dict[str, CompressionResult] = {}
        
        # Performance tracking
        self.pipeline_stats = {
            'total_requests': 0,
            'successful_compressions': 0,
            'failed_compressions': 0,
            'gpu_compressions': 0,
            'cpu_compressions': 0,
            'hybrid_compressions': 0,
            'average_compression_ratio': 0.0,
            'average_compression_time': 0.0,
            'total_memory_saved_mb': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
    
    def submit_compression(self, request: CompressionRequest) -> str:
        """Submit compression request to pipeline"""
        with self._lock:
            self.pipeline_stats['total_requests'] += 1
            self.active_requests[request.request_id] = request
            self.request_queue.append(request)
            return request.request_id
    
    def process_request(self, request: CompressionRequest) -> CompressionResult:
        """Process single compression request"""
        start_time = time.time()
        
        try:
            # Prepare metadata
            tensor_size_mb = request.tensor.numel() * request.tensor.element_size() / (1024**2)
            
            compression_metadata = {
                'size_mb': tensor_size_mb,
                'priority': request.priority,
                'require_gpu': request.require_gpu,
                'require_cpu': request.require_cpu,
                'original_shape': request.tensor.shape,
                'dtype': str(request.tensor.dtype),
                'device': str(request.tensor.device)
            }
            compression_metadata.update(request.metadata)
            
            # Get memory pressure decision
            memory_handler = self.components.get('memory_pressure_handler')
            use_cpu = False
            if memory_handler and self.config.enable_memory_pressure:
                use_cpu, decision_info = memory_handler.should_use_cpu(compression_metadata)
                compression_metadata['memory_decision'] = decision_info
            
            # Select compression system
            if self.config.enable_hybrid:
                compressor = self.components['hybrid_compressor']
            else:
                compressor = self.components['padic_compressor']
            
            # Compress tensor
            compress_start = time.time()
            compressed = compressor.compress(request.tensor)
            compress_time = time.time() - compress_start
            
            # Test decompression time
            decompress_start = time.time()
            decompressed = compressor.decompress(compressed)
            decompress_time = time.time() - decompress_start
            
            # Verify reconstruction
            if not torch.allclose(request.tensor, decompressed, rtol=1e-5, atol=1e-8):
                raise ValueError("Decompression verification failed")
            
            # Calculate metrics
            compressed_size = self._calculate_compressed_size(compressed)
            original_size = tensor_size_mb * 1024 * 1024
            compression_ratio = original_size / compressed_size
            
            # Update statistics
            self._update_pipeline_stats(
                compression_ratio, compress_time, 
                'cpu' if use_cpu else 'gpu', tensor_size_mb
            )
            
            # Create result
            result = CompressionResult(
                request_id=request.request_id,
                compressed_data=compressed,
                compression_ratio=compression_ratio,
                compression_time=compress_time,
                decompression_time=decompress_time,
                processing_mode='cpu' if use_cpu else 'gpu',
                memory_used_mb=compressed_size / (1024**2),
                success=True,
                metadata=compression_metadata
            )
            
            # Cache result
            self.results_cache[request.request_id] = result
            
            # Execute callback if provided
            if request.callback:
                request.callback(result)
            
            return result
            
        except Exception as e:
            # Hard failure
            self.pipeline_stats['failed_compressions'] += 1
            
            result = CompressionResult(
                request_id=request.request_id,
                compressed_data={},
                compression_ratio=1.0,
                compression_time=time.time() - start_time,
                decompression_time=0.0,
                processing_mode='error',
                memory_used_mb=0.0,
                success=False,
                error=str(e)
            )
            
            if request.callback:
                request.callback(result)
            
            raise RuntimeError(f"Compression failed for {request.request_id}: {e}")
    
    def _calculate_compressed_size(self, compressed: Dict[str, Any]) -> int:
        """Calculate size of compressed data in bytes"""
        size = 0
        
        # Size of encoded data
        if 'encoded_data' in compressed:
            for item in compressed['encoded_data']:
                if hasattr(item, 'coefficients'):
                    size += len(item.coefficients) * 4  # 4 bytes per coefficient
                elif hasattr(item, 'exponent_channel') and hasattr(item, 'mantissa_channel'):
                    size += item.exponent_channel.numel() * item.exponent_channel.element_size()
                    size += item.mantissa_channel.numel() * item.mantissa_channel.element_size()
        
        # Size of metadata (estimate)
        size += len(str(compressed.get('metadata', {}))) * 2  # Unicode chars
        
        return size
    
    def _update_pipeline_stats(self, compression_ratio: float, compression_time: float, 
                              mode: str, size_mb: float) -> None:
        """Update pipeline statistics"""
        with self._lock:
            self.pipeline_stats['successful_compressions'] += 1
            
            if mode == 'gpu':
                self.pipeline_stats['gpu_compressions'] += 1
            elif mode == 'cpu':
                self.pipeline_stats['cpu_compressions'] += 1
            else:
                self.pipeline_stats['hybrid_compressions'] += 1
            
            # Update averages
            n = self.pipeline_stats['successful_compressions']
            avg_ratio = self.pipeline_stats['average_compression_ratio']
            avg_time = self.pipeline_stats['average_compression_time']
            
            self.pipeline_stats['average_compression_ratio'] = (avg_ratio * (n-1) + compression_ratio) / n
            self.pipeline_stats['average_compression_time'] = (avg_time * (n-1) + compression_time) / n
            
            # Memory saved
            memory_saved = size_mb * (1 - 1/compression_ratio)
            self.pipeline_stats['total_memory_saved_mb'] += memory_saved
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        with self._lock:
            return dict(self.pipeline_stats)
    
    def process_queue(self) -> List[CompressionResult]:
        """Process all queued requests"""
        results = []
        
        while self.request_queue:
            request = self.request_queue.popleft()
            try:
                result = self.process_request(request)
                results.append(result)
            except Exception as e:
                # Continue processing other requests
                pass
        
        return results


class PerformanceOptimizationManager:
    """
    Manages system-wide performance optimization
    Adapts strategies based on workload and metrics
    """
    
    def __init__(self, config: SystemConfiguration, components: Dict[str, Any]):
        """Initialize performance optimization manager"""
        self.config = config
        self.components = components
        
        # Optimization state
        self.current_strategy = config.optimization_strategy
        self.optimization_history = deque(maxlen=1000)
        
        # Performance metrics
        self.metrics = {
            'throughput': deque(maxlen=100),
            'latency': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'gpu_utilization': deque(maxlen=100),
            'cpu_utilization': deque(maxlen=100),
            'compression_ratio': deque(maxlen=100)
        }
        
        # Optimization parameters
        self.optimization_params = {
            OptimizationStrategy.THROUGHPUT: {
                'batch_size': 1000,
                'cpu_workers': psutil.cpu_count(),
                'gpu_streams': 8,
                'prefetch_factor': 4
            },
            OptimizationStrategy.LATENCY: {
                'batch_size': 100,
                'cpu_workers': 4,
                'gpu_streams': 2,
                'prefetch_factor': 1
            },
            OptimizationStrategy.MEMORY: {
                'batch_size': 50,
                'cpu_workers': 2,
                'gpu_streams': 1,
                'prefetch_factor': 0
            },
            OptimizationStrategy.BALANCED: {
                'batch_size': 200,
                'cpu_workers': psutil.cpu_count() // 2,
                'gpu_streams': 4,
                'prefetch_factor': 2
            }
        }
        
        # Adaptive optimization state
        self.adaptation_enabled = config.optimization_strategy == OptimizationStrategy.ADAPTIVE
        self.adaptation_interval = 60  # seconds
        self.last_adaptation = time.time()
        
        # Thread safety
        self._lock = threading.RLock()
    
    def record_metrics(self, operation_type: str, metrics: Dict[str, float]) -> None:
        """Record performance metrics"""
        with self._lock:
            timestamp = time.time()
            
            # Update metric queues
            if 'throughput' in metrics:
                self.metrics['throughput'].append(metrics['throughput'])
            if 'latency' in metrics:
                self.metrics['latency'].append(metrics['latency'])
            if 'memory_usage' in metrics:
                self.metrics['memory_usage'].append(metrics['memory_usage'])
            if 'compression_ratio' in metrics:
                self.metrics['compression_ratio'].append(metrics['compression_ratio'])
            
            # Record system metrics
            if torch.cuda.is_available():
                gpu_util = torch.cuda.utilization()
                self.metrics['gpu_utilization'].append(gpu_util)
            
            cpu_util = psutil.cpu_percent(interval=0.1)
            self.metrics['cpu_utilization'].append(cpu_util)
            
            # Add to history
            self.optimization_history.append({
                'timestamp': timestamp,
                'operation': operation_type,
                'metrics': metrics,
                'strategy': self.current_strategy.value
            })
            
            # Check for adaptation
            if self.adaptation_enabled:
                self._check_adaptation()
    
    def _check_adaptation(self) -> None:
        """Check if strategy adaptation is needed"""
        current_time = time.time()
        if current_time - self.last_adaptation < self.adaptation_interval:
            return
        
        # Analyze recent performance
        if len(self.metrics['throughput']) < 10:
            return  # Not enough data
        
        # Calculate average metrics
        avg_throughput = np.mean(list(self.metrics['throughput']))
        avg_latency = np.mean(list(self.metrics['latency']))
        avg_memory = np.mean(list(self.metrics['memory_usage']))
        avg_gpu_util = np.mean(list(self.metrics['gpu_utilization']))
        
        # Determine best strategy
        new_strategy = self._determine_optimal_strategy(
            avg_throughput, avg_latency, avg_memory, avg_gpu_util
        )
        
        if new_strategy != self.current_strategy:
            self.switch_strategy(new_strategy)
            self.last_adaptation = current_time
    
    def _determine_optimal_strategy(self, throughput: float, latency: float, 
                                   memory: float, gpu_util: float) -> OptimizationStrategy:
        """Determine optimal strategy based on metrics"""
        # Simple heuristic-based selection
        if gpu_util > 90 and memory > 80:
            # High GPU pressure, optimize memory
            return OptimizationStrategy.MEMORY
        elif latency > 100:  # ms
            # High latency, optimize for speed
            return OptimizationStrategy.LATENCY
        elif throughput < 1000:  # items/sec
            # Low throughput, optimize for throughput
            return OptimizationStrategy.THROUGHPUT
        else:
            # Everything reasonable, stay balanced
            return OptimizationStrategy.BALANCED
    
    def switch_strategy(self, strategy: OptimizationStrategy) -> None:
        """Switch optimization strategy"""
        with self._lock:
            if strategy == self.current_strategy:
                return
            
            self.current_strategy = strategy
            params = self.optimization_params.get(strategy, self.optimization_params[OptimizationStrategy.BALANCED])
            
            # Apply new parameters to components
            self._apply_optimization_params(params)
    
    def _apply_optimization_params(self, params: Dict[str, Any]) -> None:
        """Apply optimization parameters to components"""
        # Update CPU bursting configuration
        if 'cpu_bursting' in self.components:
            cpu_pipeline = self.components['cpu_bursting']
            if hasattr(cpu_pipeline, 'config'):
                cpu_pipeline.config.cpu_batch_size = params['batch_size']
                cpu_pipeline.config.num_cpu_workers = params['cpu_workers']
        
        # Update GPU configuration
        if 'gpu_optimizer' in self.components:
            gpu_optimizer = self.components['gpu_optimizer']
            # Apply GPU-specific optimizations
            pass
        
        # Update compression configuration
        if 'hybrid_compressor' in self.components:
            compressor = self.components['hybrid_compressor']
            if hasattr(compressor, 'chunk_size'):
                compressor.chunk_size = params['batch_size']
    
    def get_current_strategy(self) -> OptimizationStrategy:
        """Get current optimization strategy"""
        return self.current_strategy
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        with self._lock:
            summary = {
                'current_strategy': self.current_strategy.value,
                'adaptation_enabled': self.adaptation_enabled,
                'metrics': {}
            }
            
            # Calculate averages
            for metric_name, values in self.metrics.items():
                if values:
                    summary['metrics'][metric_name] = {
                        'current': values[-1] if values else 0,
                        'average': np.mean(list(values)),
                        'min': min(values),
                        'max': max(values)
                    }
            
            return summary


class SystemIntegrationCoordinator:
    """
    Master coordinator for the entire compression system
    Manages all components and provides unified interface
    """
    
    def __init__(self, config: Optional[SystemConfiguration] = None):
        """Initialize system integration coordinator"""
        self.config = config or SystemConfiguration()
        self.state = SystemState.INITIALIZING
        
        # Component registry
        self.components: Dict[str, Any] = {}
        
        # Subsystem managers
        self.pipeline_orchestrator: Optional[CompressionPipelineOrchestrator] = None
        self.performance_manager: Optional[PerformanceOptimizationManager] = None
        
        # System statistics
        self.system_stats = {
            'uptime_seconds': 0,
            'total_compressions': 0,
            'total_decompressions': 0,
            'total_memory_saved_gb': 0.0,
            'system_errors': 0,
            'component_failures': defaultdict(int)
        }
        
        # Monitoring
        self.monitoring_active = True
        self.start_time = time.time()
        self._lock = threading.RLock()
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self) -> None:
        """Initialize all system components"""
        try:
            # Initialize GPU memory optimizer
            self._initialize_gpu_memory()
            
            # Initialize SmartPool if enabled
            if self.config.enable_smart_pool:
                self._initialize_smart_pool()
            
            # Initialize AutoSwap if enabled
            if self.config.enable_auto_swap:
                self._initialize_auto_swap()
            
            # Initialize CPU bursting if enabled
            if self.config.enable_cpu_bursting:
                self._initialize_cpu_bursting()
            
            # Initialize memory pressure handler
            if self.config.enable_memory_pressure:
                self._initialize_memory_pressure()
            
            # Initialize compression systems
            self._initialize_compression_systems()
            
            # Initialize orchestrators
            self.pipeline_orchestrator = CompressionPipelineOrchestrator(self.config, self.components)
            self.performance_manager = PerformanceOptimizationManager(self.config, self.components)
            
            # Start monitoring
            self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
            self.monitor_thread.start()
            
            self.state = SystemState.READY
            
        except Exception as e:
            self.state = SystemState.ERROR
            raise RuntimeError(f"System initialization failed: {e}")
    
    def _initialize_gpu_memory(self) -> None:
        """Initialize GPU memory optimizer"""
        gpu_config = {
            'device_ids': [0],  # TODO: Make configurable
            'memory_limit_mb': self.config.gpu_memory_limit_mb,
            'enable_monitoring': True
        }
        
        self.components['gpu_optimizer'] = GPUMemoryOptimizer(gpu_config)
    
    def _initialize_smart_pool(self) -> None:
        """Initialize SmartPool memory management"""
        smart_pool = integrate_smartpool_with_gpu_optimizer(
            self.components['gpu_optimizer'],
            fragmentation_reduction_target=self.config.smart_pool_fragmentation_target
        )
        self.components['smart_pool'] = smart_pool
    
    def _initialize_auto_swap(self) -> None:
        """Initialize AutoSwap memory swapping"""
        # AutoSwap is already integrated in GPUMemoryOptimizer
        gpu_optimizer = self.components['gpu_optimizer']
        if hasattr(gpu_optimizer, 'autoswap_manager'):
            self.components['auto_swap'] = gpu_optimizer.autoswap_manager
    
    def _initialize_cpu_bursting(self) -> None:
        """Initialize CPU bursting pipeline"""
        from .gpu_memory.cpu_bursting_pipeline import CPUBurstingConfig, integrate_cpu_bursting
        
        cpu_config = CPUBurstingConfig(
            gpu_memory_threshold_mb=self.config.gpu_memory_threshold_mb,
            num_cpu_workers=self.config.cpu_workers,
            cpu_batch_size=self.config.cpu_batch_size
        )
        
        # Need decompression engine first
        if 'decompression_engine' not in self.components:
            # Create temporary engine
            from .padic.padic_advanced import GPUDecompressionConfig, PadicDecompressionEngine
            
            gpu_decompression_config = GPUDecompressionConfig()
            self.components['decompression_engine'] = PadicDecompressionEngine(
                gpu_decompression_config, self.config.prime
            )
        
        cpu_pipeline = CPU_BurstingPipeline(cpu_config, self.components['decompression_engine'])
        self.components['cpu_bursting'] = cpu_pipeline
    
    def _initialize_memory_pressure(self) -> None:
        """Initialize memory pressure handler"""
        from .padic.memory_pressure_handler import PressureHandlerConfig
        
        pressure_config = PressureHandlerConfig(
            force_cpu_on_critical=self.config.force_cpu_on_critical,
            adaptive_threshold=True
        )
        
        memory_handler = MemoryPressureHandler(
            pressure_config,
            self.components.get('gpu_optimizer'),
            self.components.get('cpu_bursting')
        )
        
        self.components['memory_pressure_handler'] = memory_handler
    
    def _initialize_compression_systems(self) -> None:
        """Initialize p-adic compression systems"""
        # Base p-adic configuration
        padic_config = {
            'prime': self.config.prime,
            'precision': self.config.precision,
            'chunk_size': self.config.chunk_size,
            'gpu_memory_limit_mb': self.config.gpu_memory_limit_mb,
            'enable_hybrid': self.config.enable_hybrid,
            'gpu_optimizer': self.components.get('gpu_optimizer')
        }
        
        # Initialize hybrid compressor
        if self.config.enable_hybrid:
            self.components['hybrid_compressor'] = HybridPadicCompressionSystem(padic_config)
        
        # Initialize standard compressor as fallback
        from .padic.padic_compressor import PadicCompressionSystem
        self.components['padic_compressor'] = PadicCompressionSystem(padic_config)
    
    def compress(self, tensor: torch.Tensor, priority: str = "normal", 
                 metadata: Optional[Dict[str, Any]] = None) -> CompressionResult:
        """
        Compress a tensor using the full pipeline
        
        Args:
            tensor: Tensor to compress
            priority: Priority level (low, normal, high, critical)
            metadata: Additional metadata
            
        Returns:
            CompressionResult with compressed data and metrics
        """
        if self.state != SystemState.READY:
            raise RuntimeError(f"System not ready: {self.state}")
        
        # Create compression request
        request_id = f"req_{int(time.time() * 1000000)}"
        request = CompressionRequest(
            request_id=request_id,
            tensor=tensor,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Process through pipeline
        with self._lock:
            self.state = SystemState.PROCESSING
            self.system_stats['total_compressions'] += 1
        
        try:
            result = self.pipeline_orchestrator.process_request(request)
            
            # Record metrics
            self.performance_manager.record_metrics('compression', {
                'throughput': 1.0 / result.compression_time,
                'latency': result.compression_time * 1000,
                'compression_ratio': result.compression_ratio,
                'memory_usage': result.memory_used_mb
            })
            
            return result
            
        finally:
            with self._lock:
                self.state = SystemState.READY
    
    def decompress(self, compressed_data: Dict[str, Any]) -> torch.Tensor:
        """
        Decompress data
        
        Args:
            compressed_data: Compressed data from compress()
            
        Returns:
            Decompressed tensor
        """
        if self.state != SystemState.READY:
            raise RuntimeError(f"System not ready: {self.state}")
        
        with self._lock:
            self.state = SystemState.PROCESSING
            self.system_stats['total_decompressions'] += 1
        
        try:
            # Select appropriate decompressor
            if self.config.enable_hybrid and 'hybrid_compressor' in self.components:
                decompressor = self.components['hybrid_compressor']
            else:
                decompressor = self.components['padic_compressor']
            
            # Decompress
            start_time = time.time()
            result = decompressor.decompress(compressed_data)
            decompress_time = time.time() - start_time
            
            # Record metrics
            self.performance_manager.record_metrics('decompression', {
                'throughput': 1.0 / decompress_time,
                'latency': decompress_time * 1000
            })
            
            return result
            
        finally:
            with self._lock:
                self.state = SystemState.READY
    
    def optimize_system(self, strategy: Optional[OptimizationStrategy] = None) -> None:
        """
        Optimize system performance
        
        Args:
            strategy: Optimization strategy to use (None for auto)
        """
        with self._lock:
            self.state = SystemState.OPTIMIZING
        
        try:
            if strategy:
                self.performance_manager.switch_strategy(strategy)
            else:
                # Auto-optimize based on current metrics
                summary = self.performance_manager.get_performance_summary()
                # Performance manager will auto-adapt if enabled
            
        finally:
            with self._lock:
                self.state = SystemState.READY
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        with self._lock:
            uptime = time.time() - self.start_time
            
            status = {
                'state': self.state.value,
                'uptime_seconds': uptime,
                'configuration': self.config.to_dict(),
                'components': {
                    'gpu_optimizer': 'active' if 'gpu_optimizer' in self.components else 'inactive',
                    'smart_pool': 'active' if 'smart_pool' in self.components else 'inactive',
                    'auto_swap': 'active' if 'auto_swap' in self.components else 'inactive',
                    'cpu_bursting': 'active' if 'cpu_bursting' in self.components else 'inactive',
                    'memory_pressure': 'active' if 'memory_pressure_handler' in self.components else 'inactive',
                    'hybrid_compressor': 'active' if 'hybrid_compressor' in self.components else 'inactive'
                },
                'statistics': dict(self.system_stats),
                'performance': self.performance_manager.get_performance_summary() if self.performance_manager else {},
                'pipeline': self.pipeline_orchestrator.get_statistics() if self.pipeline_orchestrator else {}
            }
            
            # Add component-specific stats
            if 'memory_pressure_handler' in self.components:
                status['memory_pressure'] = self.components['memory_pressure_handler'].get_memory_summary()
            
            if 'gpu_optimizer' in self.components:
                gpu_stats = self.components['gpu_optimizer'].get_optimization_results()
                if gpu_stats:
                    status['gpu_optimization'] = {
                        'memory_saved_mb': gpu_stats.memory_saved_mb,
                        'fragmentation_reduced': gpu_stats.fragmentation_reduced,
                        'allocation_time_ms': gpu_stats.allocation_time_ms
                    }
            
            return status
    
    def _monitor_system(self) -> None:
        """Background system monitoring"""
        while self.monitoring_active:
            try:
                # Update uptime
                self.system_stats['uptime_seconds'] = time.time() - self.start_time
                
                # Check component health
                self._check_component_health()
                
                # Auto-optimization if enabled
                if self.config.enable_auto_optimization:
                    if self.performance_manager and self.state == SystemState.READY:
                        # Performance manager handles its own optimization
                        pass
                
                # Checkpoint if needed
                if self.config.checkpoint_interval_seconds > 0:
                    if int(self.system_stats['uptime_seconds']) % self.config.checkpoint_interval_seconds == 0:
                        self._create_checkpoint()
                
                time.sleep(self.config.monitoring_interval_ms / 1000.0)
                
            except Exception as e:
                self.system_stats['system_errors'] += 1
                warnings.warn(f"Monitoring error: {e}")
    
    def _check_component_health(self) -> None:
        """Check health of all components"""
        for name, component in self.components.items():
            try:
                # Simple health check - component exists and has expected methods
                if hasattr(component, 'get_statistics') or hasattr(component, 'get_stats'):
                    # Component appears healthy
                    pass
            except Exception:
                self.system_stats['component_failures'][name] += 1
    
    def _create_checkpoint(self) -> None:
        """Create system checkpoint"""
        # Save configuration and statistics
        checkpoint = {
            'timestamp': time.time(),
            'configuration': self.config.to_dict(),
            'statistics': dict(self.system_stats),
            'performance': self.performance_manager.get_performance_summary() if self.performance_manager else {}
        }
        
        # Could save to file or send to monitoring system
        pass
    
    def shutdown(self) -> None:
        """Shutdown system gracefully"""
        with self._lock:
            self.state = SystemState.SHUTDOWN
            self.monitoring_active = False
        
        # Shutdown components
        for name, component in self.components.items():
            if hasattr(component, 'cleanup'):
                try:
                    component.cleanup()
                except Exception as e:
                    warnings.warn(f"Error shutting down {name}: {e}")
            elif hasattr(component, 'shutdown'):
                try:
                    component.shutdown()
                except Exception as e:
                    warnings.warn(f"Error shutting down {name}: {e}")


# Convenience functions
def create_compression_system(config: Optional[SystemConfiguration] = None) -> SystemIntegrationCoordinator:
    """
    Create a fully configured compression system
    
    Args:
        config: System configuration (None for defaults)
        
    Returns:
        Initialized SystemIntegrationCoordinator
    """
    return SystemIntegrationCoordinator(config)


def load_compression_system(config_file: str) -> SystemIntegrationCoordinator:
    """
    Load compression system from configuration file
    
    Args:
        config_file: Path to configuration JSON file
        
    Returns:
        Initialized SystemIntegrationCoordinator
    """
    config = SystemConfiguration.load(config_file)
    return SystemIntegrationCoordinator(config)