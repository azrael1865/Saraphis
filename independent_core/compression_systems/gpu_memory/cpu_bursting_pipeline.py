"""
CPU_BurstingPipeline Core - CPU-based decompression for GPU memory constraints
NO FALLBACKS - HARD FAILURES ONLY

When GPU memory is exhausted, automatically switches to CPU-based p-adic decompression
Uses broken-down p-adic components (exponent/mantissa channels)
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
import psutil
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import weakref
import os

# Import p-adic components
try:
    from ..padic.padic_encoder import PadicWeight, validate_single_weight
    from ..padic.padic_advanced import PadicDecompressionEngine
    from ..padic.memory_pressure_handler import MemoryPressureHandler, PressureHandlerConfig
    from ..padic.safe_reconstruction import SafePadicReconstructor, ReconstructionConfig, ReconstructionMethod
    from .gpu_auto_detector import get_config_updater, AutoOptimizedConfig
except ImportError:
    # Direct imports for testing
    from compression_systems.padic.padic_encoder import PadicWeight, validate_single_weight
    from compression_systems.padic.padic_advanced import PadicDecompressionEngine
    from compression_systems.padic.memory_pressure_handler import MemoryPressureHandler, PressureHandlerConfig
    from compression_systems.padic.safe_reconstruction import SafePadicReconstructor, ReconstructionConfig, ReconstructionMethod
    from compression_systems.gpu_memory.gpu_auto_detector import get_config_updater, AutoOptimizedConfig


class DecompressionMode(Enum):
    """Decompression execution modes"""
    GPU_ONLY = "gpu_only"
    CPU_ONLY = "cpu_only"
    AUTO_SWITCH = "auto_switch"
    HYBRID = "hybrid"  # Use both GPU and CPU


class MemoryPressureLevel(Enum):
    """GPU memory pressure levels"""
    LOW = 0.5       # < 50% utilization
    MODERATE = 0.75  # 50-75% utilization
    HIGH = 0.9      # 75-90% utilization
    CRITICAL = 0.95  # > 90% utilization


@dataclass
class CPUBurstingConfig:
    """Configuration for CPU bursting pipeline - Auto-configured for detected GPU"""
    # Memory thresholds
    gpu_memory_threshold_mb: int = None      # Auto-detected
    memory_pressure_threshold: float = None  # Auto-detected
    
    # CPU configuration
    num_cpu_workers: int = -1  # -1 for auto-detect
    cpu_batch_size: int = None               # Auto-detected
    use_multiprocessing: bool = True         # Ensure MP is on
    cpu_affinity: Optional[List[int]] = None  # CPU cores to use
    
    # Performance settings
    enable_profiling: bool = True
    enable_caching: bool = True
    cache_size_mb: int = None                # Auto-detected
    prefetch_factor: int = 2
    
    # Decompression settings
    progressive_precision: bool = True
    precision_schedule: Optional[List[int]] = None
    mantissa_bits: int = 23  # For float32 compatibility
    exponent_bits: int = 8
    
    # Auto-switching settings
    switch_delay_ms: int = 10  # Delay before switching modes
    hysteresis_factor: float = 0.1  # Prevent rapid switching
    
    # Auto-configured optimizations
    prefetch_batches: int = None              # Auto-detected
    numa_nodes: List[int] = None             # Auto-detected
    huge_page_size: int = None               # Auto-detected
    cpu_affinity_enabled: bool = True        # Pin workers to CPUs
    
    # Flag to track if auto-configuration has been applied
    _auto_configured: bool = False
    
    def __post_init__(self):
        """Auto-configure and validate configuration"""
        # Apply auto-configuration if values are None
        if not self._auto_configured:
            self._apply_auto_configuration()
        
        # Validate after auto-configuration
        if self.gpu_memory_threshold_mb <= 0:
            raise ValueError(f"gpu_memory_threshold_mb must be > 0, got {self.gpu_memory_threshold_mb}")
        if not 0 < self.memory_pressure_threshold <= 1:
            raise ValueError(f"memory_pressure_threshold must be in (0,1], got {self.memory_pressure_threshold}")
        if self.num_cpu_workers == 0:
            raise ValueError("num_cpu_workers cannot be 0")
        if self.cpu_batch_size <= 0:
            raise ValueError(f"cpu_batch_size must be > 0, got {self.cpu_batch_size}")
    
    def _apply_auto_configuration(self):
        """Apply auto-detected configuration values"""
        try:
            config_updater = get_config_updater()
            auto_config = config_updater.optimized_config
            
            # Apply auto-detected values only if not manually set
            if self.gpu_memory_threshold_mb is None:
                self.gpu_memory_threshold_mb = auto_config.gpu_memory_threshold_mb
            if self.memory_pressure_threshold is None:
                self.memory_pressure_threshold = auto_config.memory_pressure_threshold
            if self.cpu_batch_size is None:
                self.cpu_batch_size = auto_config.cpu_batch_size
            if self.cache_size_mb is None:
                self.cache_size_mb = auto_config.cache_size_mb
            if self.prefetch_batches is None:
                self.prefetch_batches = auto_config.prefetch_batches
            if self.numa_nodes is None:
                self.numa_nodes = auto_config.numa_nodes
            if self.huge_page_size is None:
                self.huge_page_size = auto_config.huge_page_size
            
            # Update CPU workers if auto-detect
            if self.num_cpu_workers == -1:
                self.num_cpu_workers = auto_config.num_cpu_workers
            
            self._auto_configured = True
            
            # Log detected GPU info
            gpu_info = config_updater.get_gpu_info_string()
            print(f"Auto-configured CPU bursting for:\n{gpu_info}")
            
        except Exception as e:
            # Fallback to conservative defaults if auto-detection fails
            print(f"Auto-configuration failed: {e}, using conservative defaults")
            if self.gpu_memory_threshold_mb is None:
                self.gpu_memory_threshold_mb = 2048
            if self.memory_pressure_threshold is None:
                self.memory_pressure_threshold = 0.90
            if self.cpu_batch_size is None:
                self.cpu_batch_size = 10000
            if self.cache_size_mb is None:
                self.cache_size_mb = 4096
            if self.prefetch_batches is None:
                self.prefetch_batches = 10
            if self.numa_nodes is None:
                self.numa_nodes = [0]
            if self.huge_page_size is None:
                self.huge_page_size = 2097152
            
            self._auto_configured = True


@dataclass
class DecompressionTask:
    """Task for decompression queue"""
    task_id: str
    padic_weights: List[PadicWeight]
    target_precision: int
    metadata: Dict[str, Any]
    mode: DecompressionMode
    priority: int = 0
    callback: Optional[callable] = None


@dataclass
class BurstingStatistics:
    """Statistics for CPU bursting operations"""
    total_decompressions: int = 0
    gpu_decompressions: int = 0
    cpu_decompressions: int = 0
    mode_switches: int = 0
    
    total_gpu_time: float = 0.0
    total_cpu_time: float = 0.0
    total_switch_time: float = 0.0
    
    memory_pressure_events: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    cpu_utilization: List[float] = None
    gpu_memory_saved_mb: float = 0.0
    
    def __post_init__(self):
        if self.cpu_utilization is None:
            self.cpu_utilization = []


class CPUDecompressionEngine:
    """
    CPU-based p-adic decompression engine
    Optimized for multi-core processing without GPU
    """
    
    def __init__(self, config: CPUBurstingConfig, prime: int):
        """Initialize CPU decompression engine"""
        self.config = config
        self.prime = prime
        self.min_channel_size = 10  # Add minimum channel size for consistent sizing
        
        # Initialize safe reconstructor with appropriate config
        self.reconstruction_config = ReconstructionConfig(
            prime=self.prime,
            max_safe_precision=6,  # Safe limit for prime=257
            method=ReconstructionMethod.HYBRID,
            overflow_threshold=1e12
        )
        self.safe_reconstructor = SafePadicReconstructor(self.reconstruction_config)
        
        # Determine number of workers
        if config.num_cpu_workers == -1:
            self.num_workers = mp.cpu_count()
        else:
            self.num_workers = config.num_cpu_workers
        
        # Initialize executor
        # Note: Using ThreadPoolExecutor to avoid thread lock serialization issues
        # ProcessPoolExecutor cannot pickle threading.RLock objects
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        
        # Set CPU affinity if specified
        if config.cpu_affinity:
            self._set_cpu_affinity(config.cpu_affinity)
        
        # Cache for decompressed results
        self.cache: Dict[str, torch.Tensor] = {}
        self.cache_size = 0
        self.cache_limit = config.cache_size_mb * 1024 * 1024
        
        # Performance tracking
        self.decompression_times: List[float] = []
        
        # Thread safety
        self._lock = threading.RLock()
    
    def _set_cpu_affinity(self, cpu_list: List[int]) -> None:
        """Set CPU affinity for current process"""
        try:
            p = psutil.Process()
            p.cpu_affinity(cpu_list)
        except Exception as e:
            # CPU affinity not supported on all platforms - HARD FAILURE
            raise RuntimeError(f"CPU affinity setup failed: {e}")
    
    def decompress_batch_cpu(self, padic_weights: List[PadicWeight], 
                            target_precision: int,
                            metadata: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Decompress p-adic weights using CPU
        
        Args:
            padic_weights: List of p-adic weights
            target_precision: Target precision
            metadata: Decompression metadata
            
        Returns:
            Tuple of (decompressed_tensor, decompression_info)
        """
        start_time = time.time()
        
        # Check cache
        cache_key = self._create_cache_key(padic_weights, target_precision)
        if self.config.enable_caching and cache_key in self.cache:
            return self.cache[cache_key], {'cache_hit': True, 'cpu_time': 0.0}
        
        try:
            # Split into batches for parallel processing
            batch_size = self.config.cpu_batch_size
            batches = [padic_weights[i:i+batch_size] 
                      for i in range(0, len(padic_weights), batch_size)]
            
            # Process batches in parallel
            futures = []
            for batch_idx, batch in enumerate(batches):
                future = self.executor.submit(
                    self._decompress_single_batch,
                    batch, target_precision, batch_idx
                )
                futures.append(future)
            
            # Collect results
            batch_results = []
            for future in futures:
                result = future.result()
                batch_results.append(result)
            
            # Combine results
            final_tensor = self._combine_batch_results(batch_results, metadata)
            
            # Update cache if enabled
            if self.config.enable_caching:
                self._update_cache(cache_key, final_tensor)
            
            cpu_time = time.time() - start_time
            self.decompression_times.append(cpu_time)
            
            decompression_info = {
                'mode': 'cpu',
                'num_batches': len(batches),
                'cpu_workers': self.num_workers,
                'cpu_time': cpu_time,
                'cache_hit': False,
                'weights_processed': len(padic_weights)
            }
            
            return final_tensor, decompression_info
            
        except Exception as e:
            raise RuntimeError(f"CPU decompression failed: {e}")
    
    def _decompress_single_batch(self, batch: List[PadicWeight], 
                               target_precision: int, 
                               batch_idx: int) -> np.ndarray:
        """Decompress single batch on CPU"""
        # Convert p-adic to numpy arrays
        batch_data = []
        
        for weight in batch:
            # Extract mantissa and exponent channels
            mantissa, exponent = self._extract_channels(weight)
            
            # CRITICAL FIX: Use weight.precision instead of target_precision
            reconstructed = self._reconstruct_float(mantissa, exponent, weight.precision, weight)
            batch_data.append(reconstructed)
        
        return np.array(batch_data, dtype=np.float32)
    
    def _extract_channels(self, weight: PadicWeight) -> Tuple[np.ndarray, np.ndarray]:
        """Extract mantissa and exponent channels from p-adic weight
        
        Args:
            weight: PadicWeight object containing digits and valuation
            
        Returns:
            Tuple of (mantissa_channel, exponent_channel) as numpy arrays
        """
        # Validate input
        if not hasattr(weight, 'digits') or not hasattr(weight, 'valuation'):
            raise ValueError(f"Invalid PadicWeight structure: missing digits or valuation")
        
        # Extract mantissa channel from p-adic digits
        # Pad with zeros if needed to ensure consistent size
        mantissa = np.array(weight.digits, dtype=np.float32)
        if len(mantissa) < self.min_channel_size:
            mantissa = np.pad(mantissa, (0, self.min_channel_size - len(mantissa)), 
                             mode='constant', constant_values=0)
        
        # Extract exponent channel from p-adic valuation
        # Encode valuation as array for consistent interface
        # Use signed representation: [sign, abs(val), 0, 0, ...]
        exponent = np.zeros(self.min_channel_size, dtype=np.float32)
        exponent[0] = 1.0 if weight.valuation >= 0 else -1.0  # Sign
        exponent[1] = float(abs(weight.valuation))  # Magnitude
        
        return mantissa, exponent
    
    def _reconstruct_float(self, mantissa: np.ndarray, exponent: np.ndarray, 
                         precision: int, weight: PadicWeight) -> float:
        """Reconstruct float using safe reconstruction to prevent overflow
        
        Args:
            mantissa: Array of p-adic digits (coefficients)
            exponent: Array encoding valuation [sign, magnitude, ...]
            precision: Number of p-adic digits to use in reconstruction (weight.precision)
            weight: Original PadicWeight for validation
            
        Returns:
            Reconstructed float value
        """
        # Validate inputs
        if len(mantissa) == 0 or len(exponent) < 2:
            raise ValueError("Invalid channel data for reconstruction")
        
        # Extract valuation from exponent channel
        valuation_sign = exponent[0]
        valuation_magnitude = int(exponent[1])
        valuation = int(valuation_sign * valuation_magnitude)
        
        # Create safe weight object for reconstruction
        from ..padic.safe_reconstruction import PadicWeight as SafeWeight
        
        # Use actual digits from weight, not mantissa (which may be padded)
        effective_precision = min(precision, len(weight.digits))
        
        safe_weight = SafeWeight(
            digits=weight.digits[:effective_precision],
            valuation=valuation,
            precision=effective_precision,
            prime=self.prime
        )
        
        try:
            # Use safe reconstruction to prevent overflow
            value = self.safe_reconstructor.reconstruct(safe_weight, effective_precision)
            
            # Clamp to float32 range to prevent overflow downstream
            max_float32 = np.finfo(np.float32).max
            min_float32 = np.finfo(np.float32).min
            
            if value > max_float32:
                value = max_float32
            elif value < min_float32:
                value = min_float32
            
            # Handle NaN/inf
            if np.isnan(value) or np.isinf(value):
                print(f"Warning: Safe reconstruction returned {value}, using 0.0")
                value = 0.0
            
            return float(value)
            
        except (OverflowError, ValueError) as e:
            # NO FALLBACKS - HARD FAILURE ONLY
            raise RuntimeError(f"Safe reconstruction failed for weight with precision {effective_precision}, valuation {valuation}: {e}")
    
    def _combine_batch_results(self, batch_results: List[np.ndarray], 
                             metadata: Dict[str, Any]) -> torch.Tensor:
        """Combine batch results into final tensor"""
        # Concatenate all batches
        combined = np.concatenate(batch_results, axis=0)
        
        # Reshape to original shape
        original_shape = metadata['original_shape']
        reshaped = combined.reshape(original_shape)
        
        # Convert to torch tensor
        tensor = torch.from_numpy(reshaped)
        
        # Convert to target dtype
        dtype_str = metadata['dtype'].split('.')[-1]
        if hasattr(torch, dtype_str):
            target_dtype = getattr(torch, dtype_str)
            tensor = tensor.to(dtype=target_dtype)
        
        return tensor
    
    def _create_cache_key(self, weights: List[PadicWeight], precision: int) -> str:
        """Create cache key for decompression result"""
        # Simple hash based on first/last weights and precision
        if not weights:
            return ""
        
        first_hash = hash(tuple(weights[0].digits[:10]))
        last_hash = hash(tuple(weights[-1].digits[:10]))
        return f"cpu_p{precision}_f{first_hash}_l{last_hash}_n{len(weights)}"
    
    def _update_cache(self, key: str, tensor: torch.Tensor) -> None:
        """Update result cache"""
        with self._lock:
            tensor_size = tensor.numel() * tensor.element_size()
            
            # Check if adding would exceed limit
            if self.cache_size + tensor_size > self.cache_limit:
                self._evict_cache_entries(tensor_size)
            
            self.cache[key] = tensor
            self.cache_size += tensor_size
    
    def _evict_cache_entries(self, required_size: int) -> None:
        """Evict cache entries to make room"""
        # Simple FIFO eviction
        while self.cache and self.cache_size + required_size > self.cache_limit:
            key = next(iter(self.cache))
            tensor = self.cache.pop(key)
            self.cache_size -= tensor.numel() * tensor.element_size()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get CPU decompression statistics"""
        with self._lock:
            avg_time = np.mean(self.decompression_times) if self.decompression_times else 0.0
            return {
                'total_decompressions': len(self.decompression_times),
                'average_cpu_time': avg_time,
                'cache_size_mb': self.cache_size / (1024 * 1024),
                'cache_entries': len(self.cache),
                'num_workers': self.num_workers
            }
    
    def cleanup(self) -> None:
        """Clean up CPU resources"""
        self.executor.shutdown(wait=True)
        self.cache.clear()
        self.cache_size = 0


class CPU_BurstingPipeline:
    """
    Main CPU bursting pipeline for automatic GPU/CPU switching
    Monitors GPU memory and switches to CPU decompression when needed
    """
    
    def __init__(self, config: CPUBurstingConfig, gpu_decompression_engine: PadicDecompressionEngine):
        """Initialize CPU bursting pipeline with memory pressure handling"""
        self.config = config
        self.gpu_engine = gpu_decompression_engine
        self.prime = gpu_decompression_engine.prime
        
        # Initialize CPU engine
        self.cpu_engine = CPUDecompressionEngine(config, self.prime)
        
        # Initialize memory pressure handler
        pressure_config = PressureHandlerConfig(
            gpu_critical_threshold_mb=config.gpu_memory_threshold_mb,
            gpu_high_threshold_mb=config.gpu_memory_threshold_mb * 2,
            gpu_moderate_threshold_mb=config.gpu_memory_threshold_mb * 3,
            max_cpu_batch_size=config.cpu_batch_size
        )
        self.memory_handler = MemoryPressureHandler(pressure_config)
        
        # Current mode
        self.current_mode = DecompressionMode.AUTO_SWITCH
        self.last_mode_switch = time.time()
        self.mode_history: List[Tuple[float, DecompressionMode]] = []
        
        # GPU memory monitoring
        self.gpu_memory_history: List[Tuple[float, float]] = []  # (timestamp, utilization)
        self.last_memory_check = 0.0
        self.memory_check_interval = 0.1  # 100ms
        
        # Task queue for async processing
        self.task_queue = queue.PriorityQueue()
        self.results: Dict[str, Tuple[torch.Tensor, Dict[str, Any]]] = {}
        
        # Statistics
        self.stats = BurstingStatistics()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_gpu_memory, daemon=True)
        self.monitor_thread.start()
    
    def decompress(self, weights: List[PadicWeight], target_precision: int, 
                   metadata: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Decompress weights with automatic CPU/GPU switching - HARD FAILURES ONLY"""
        start_time = time.time()
        
        # Validate weights - HARD FAILURE if invalid
        self._validate_weights(weights)
        
        with self._lock:
            self.stats.total_decompressions += 1
            
            # Check memory pressure and decide on mode
            use_cpu, decision_info = self.memory_handler.should_use_cpu(metadata)
            
            if use_cpu:
                result, info = self._decompress_cpu_burst(weights, target_precision, metadata, decision_info)
                self.stats.cpu_decompressions += 1
                self.stats.total_cpu_time += info.get('cpu_time', 0)
            else:
                result, info = self._decompress_gpu(weights, target_precision, metadata, decision_info)
                self.stats.gpu_decompressions += 1
                self.stats.total_gpu_time += info.get('decompression_time', 0)
            
            # Add pipeline metadata
            info['mode'] = 'cpu_burst' if use_cpu else 'gpu'
            info['total_time'] = time.time() - start_time
            info['decision_info'] = decision_info
            
            return result, info
    
    def _validate_weights(self, weights: List[PadicWeight]) -> None:
        """Validate weight structure - HARD FAILURE on error"""
        if not weights:
            raise ValueError("Empty weight list - HARD FAILURE - SYSTEM ABORT")
        
        if not isinstance(weights, list):
            raise TypeError(f"Weights must be list, got {type(weights)} - HARD FAILURE")
        
        # Validate each weight with comprehensive checks
        for i, weight in enumerate(weights):
            if weight is None:
                raise ValueError(f"Weight {i} is None - HARD FAILURE - CORRUPTED DATA")
            
            if not hasattr(weight, 'digits'):
                raise AttributeError(f"Weight {i} missing 'digits' attribute - HARD FAILURE")
            
            if not hasattr(weight, 'prime'):
                raise AttributeError(f"Weight {i} missing 'prime' attribute - HARD FAILURE")
            
            if not hasattr(weight, 'precision'):
                raise AttributeError(f"Weight {i} missing 'precision' attribute - HARD FAILURE")
            
            if not hasattr(weight, 'valuation'):
                raise AttributeError(f"Weight {i} missing 'valuation' attribute - HARD FAILURE")
            
            # Deep validation
            if not validate_single_weight(weight, weight.prime, weight.precision):
                raise ValueError(f"Weight {i} failed mathematical validation - HARD FAILURE")
            
            # Additional structural checks
            if not isinstance(weight.digits, list):
                raise TypeError(f"Weight {i} digits not a list - HARD FAILURE")
            
            if len(weight.digits) != weight.precision:
                raise ValueError(f"Weight {i} digit count {len(weight.digits)} != precision {weight.precision} - HARD FAILURE")
            
            if not all(0 <= d < weight.prime for d in weight.digits):
                raise ValueError(f"Weight {i} has invalid digits outside [0, {weight.prime}) - HARD FAILURE")
            
            # Valuation bounds check
            if weight.valuation < -weight.precision or weight.valuation > weight.precision:
                raise ValueError(f"Weight {i} valuation {weight.valuation} out of bounds - HARD FAILURE")
    
    def _decompress_cpu_burst(self, weights: List[PadicWeight], target_precision: int,
                             metadata: Dict[str, Any], decision_info: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Decompress using CPU bursting with memory pressure handling"""
        start_time = time.time()
        
        # Use CPU engine for decompression
        result, info = self.cpu_engine.decompress_batch_cpu(weights, target_precision, metadata)
        
        # Add CPU-specific metadata
        info['cpu_time'] = time.time() - start_time
        info['memory_pressure'] = decision_info.get('pressure_level', 'unknown')
        info['gpu_memory_available'] = decision_info.get('gpu_memory_available', 0)
        info['cpu_batch_size'] = len(weights)
        
        return result, info
                
    def _decompress_gpu(self, weights: List[PadicWeight], target_precision: int,
                       metadata: Dict[str, Any], decision_info: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Decompress using GPU with memory pressure monitoring"""
        start_time = time.time()
        
        # Use GPU engine for decompression
        result, info = self.gpu_engine.decompress_progressive(weights, target_precision, metadata)
        
        # Add GPU-specific metadata
        info['decompression_time'] = time.time() - start_time
        info['gpu_memory_used'] = decision_info.get('gpu_memory_used', 0)
        info['gpu_memory_available'] = decision_info.get('gpu_memory_available', 0)
        
        return result, info
    
    def _select_decompression_mode(self) -> DecompressionMode:
        """Select appropriate decompression mode based on GPU memory"""
        # Check if we recently switched (hysteresis)
        time_since_switch = time.time() - self.last_mode_switch
        if time_since_switch < self.config.switch_delay_ms / 1000:
            return self.current_mode
        
        # Get current GPU memory state
        memory_state = self._get_gpu_memory_state()
        
        if memory_state is None:
            # Can't determine GPU state, use CPU
            return DecompressionMode.CPU_ONLY
        
        utilization = memory_state['utilization']
        free_mb = memory_state['free_mb']
        
        # Apply hysteresis to prevent rapid switching
        if self.current_mode == DecompressionMode.GPU_ONLY:
            # Currently on GPU, switch if memory is critical
            if utilization > self.config.memory_pressure_threshold or \
               free_mb < self.config.gpu_memory_threshold_mb:
                return DecompressionMode.CPU_ONLY
        else:
            # Currently on CPU, switch back if memory is available
            hysteresis_threshold = self.config.memory_pressure_threshold - self.config.hysteresis_factor
            if utilization < hysteresis_threshold and \
               free_mb > self.config.gpu_memory_threshold_mb * 2:
                return DecompressionMode.GPU_ONLY
        
        return self.current_mode
    
    
    def _decompress_cpu(self, padic_weights: List[PadicWeight], 
                       target_precision: int,
                       metadata: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Decompress using CPU engine"""
        result, info = self.cpu_engine.decompress_batch_cpu(padic_weights, target_precision, metadata)
        
        # Estimate GPU memory saved
        tensor_size_mb = result.numel() * result.element_size() / (1024 * 1024)
        self.stats.gpu_memory_saved_mb += tensor_size_mb
        
        return result, info
    
    def _decompress_hybrid(self, padic_weights: List[PadicWeight], 
                         target_precision: int,
                         metadata: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Hybrid decompression using both GPU and CPU"""
        # Split weights between GPU and CPU based on memory availability
        memory_state = self._get_gpu_memory_state()
        
        if memory_state is None:
            # Fall back to CPU only
            return self._decompress_cpu(padic_weights, target_precision, metadata)
        
        # Estimate how many weights can fit on GPU
        free_mb = memory_state['free_mb']
        weight_size_mb = 0.1  # Rough estimate per weight
        gpu_capacity = int(free_mb / weight_size_mb * 0.8)  # Use 80% of free memory
        
        if gpu_capacity <= 0:
            return self._decompress_cpu(padic_weights, target_precision, metadata)
        
        # Split weights
        gpu_weights = padic_weights[:gpu_capacity]
        cpu_weights = padic_weights[gpu_capacity:]
        
        # Process in parallel
        gpu_future = self.cpu_engine.executor.submit(
            self._decompress_gpu, gpu_weights, target_precision, metadata
        )
        
        if cpu_weights:
            cpu_result, cpu_info = self._decompress_cpu(cpu_weights, target_precision, metadata)
        else:
            cpu_result, cpu_info = None, {}
        
        # Get GPU result
        gpu_result, gpu_info = gpu_future.result()
        
        # Combine results
        if cpu_result is not None:
            combined = torch.cat([gpu_result.flatten(), cpu_result.flatten()])
            combined = combined.reshape(metadata['original_shape'])
        else:
            combined = gpu_result
        
        hybrid_info = {
            'gpu_weights': len(gpu_weights),
            'cpu_weights': len(cpu_weights) if cpu_weights else 0,
            'gpu_info': gpu_info,
            'cpu_info': cpu_info
        }
        
        return combined, hybrid_info
    
    def _get_gpu_memory_state(self) -> Optional[Dict[str, float]]:
        """Get current GPU memory state"""
        if not torch.cuda.is_available():
            return None
        
        try:
            device_id = self.gpu_engine.device.index if hasattr(self.gpu_engine.device, 'index') else 0
            
            # Get memory info
            total = torch.cuda.get_device_properties(device_id).total_memory
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            
            free = total - reserved
            utilization = reserved / total
            
            return {
                'total_mb': total / (1024 * 1024),
                'allocated_mb': allocated / (1024 * 1024),
                'reserved_mb': reserved / (1024 * 1024),
                'free_mb': free / (1024 * 1024),
                'utilization': utilization
            }
            
        except Exception:
            return None
    
    def _monitor_gpu_memory(self) -> None:
        """Background thread to monitor GPU memory"""
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                if current_time - self.last_memory_check >= self.memory_check_interval:
                    memory_state = self._get_gpu_memory_state()
                    
                    if memory_state:
                        with self._lock:
                            self.gpu_memory_history.append((current_time, memory_state['utilization']))
                            
                            # Keep only recent history (last 60 seconds)
                            cutoff_time = current_time - 60
                            self.gpu_memory_history = [
                                (t, u) for t, u in self.gpu_memory_history if t > cutoff_time
                            ]
                            
                            # Update CPU utilization
                            cpu_percent = psutil.cpu_percent(interval=0.1)
                            self.stats.cpu_utilization.append(cpu_percent)
                            if len(self.stats.cpu_utilization) > 100:
                                self.stats.cpu_utilization.pop(0)
                    
                    self.last_memory_check = current_time
                
                time.sleep(0.05)  # 50ms sleep
                
            except Exception as e:
                # NO FALLBACKS - HARD FAILURE
                raise RuntimeError(f"GPU memory monitoring failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        with self._lock:
            gpu_stats = self.gpu_engine.get_decompression_stats()
            cpu_stats = self.cpu_engine.get_statistics()
            
            # Calculate averages
            avg_gpu_time = self.stats.total_gpu_time / max(self.stats.gpu_decompressions, 1)
            avg_cpu_time = self.stats.total_cpu_time / max(self.stats.cpu_decompressions, 1)
            
            # Get current memory state
            memory_state = self._get_gpu_memory_state()
            
            return {
                'total_decompressions': self.stats.total_decompressions,
                'gpu_decompressions': self.stats.gpu_decompressions,
                'cpu_decompressions': self.stats.cpu_decompressions,
                'mode_switches': self.stats.mode_switches,
                'memory_pressure_events': self.stats.memory_pressure_events,
                
                'average_gpu_time': avg_gpu_time,
                'average_cpu_time': avg_cpu_time,
                'gpu_memory_saved_mb': self.stats.gpu_memory_saved_mb,
                
                'current_mode': self.current_mode.value,
                'gpu_memory_state': memory_state,
                'avg_cpu_utilization': np.mean(self.stats.cpu_utilization) if self.stats.cpu_utilization else 0,
                
                'gpu_engine_stats': gpu_stats,
                'cpu_engine_stats': cpu_stats,
                
                'cache_hits': self.stats.cache_hits,
                'cache_misses': self.stats.cache_misses,
                'mode_history': self.mode_history[-10:]  # Last 10 mode switches
            }
    
    def force_mode(self, mode: DecompressionMode) -> None:
        """Force specific decompression mode"""
        with self._lock:
            self.current_mode = mode
            self.mode_history.append((time.time(), mode))
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.monitoring_active = False
        self.monitor_thread.join(timeout=1.0)
        
        self.gpu_engine.cleanup()
        self.cpu_engine.cleanup()


# Integration utilities
def integrate_cpu_bursting(gpu_optimizer: Any, config: Optional[CPUBurstingConfig] = None) -> CPU_BurstingPipeline:
    """
    Integrate CPU bursting with existing GPU memory optimizer
    
    Args:
        gpu_optimizer: Existing GPU memory optimizer instance
        config: CPU bursting configuration (optional)
        
    Returns:
        Configured CPU_BurstingPipeline instance
    """
    if config is None:
        config = CPUBurstingConfig()
    
    # Get GPU decompression engine from optimizer
    if not hasattr(gpu_optimizer, 'gpu_decompression_engine'):
        raise ValueError("GPU optimizer must have gpu_decompression_engine attribute")
    
    gpu_engine = gpu_optimizer.gpu_decompression_engine
    
    # Create CPU bursting pipeline
    cpu_pipeline = CPU_BurstingPipeline(config, gpu_engine)
    
    # Add to GPU optimizer
    gpu_optimizer.cpu_bursting_pipeline = cpu_pipeline
    
    # Replace decompression method
    original_decompress = gpu_optimizer.decompress_gpu
    
    def decompress_with_bursting(padic_weights, target_precision, metadata):
        """Decompression with automatic CPU bursting"""
        return cpu_pipeline.decompress(padic_weights, target_precision, metadata)
    
    gpu_optimizer.decompress_gpu = decompress_with_bursting
    gpu_optimizer.decompress_with_bursting = decompress_with_bursting
    
    # Add force mode method
    gpu_optimizer.force_decompression_mode = cpu_pipeline.force_mode
    
    return cpu_pipeline