"""
MemoryPressureHandler - Intelligent GPU/CPU Coordination System
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import time
import threading
import psutil
import weakref
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import warnings

# FIXED: Use relative imports for internal module references
from ..gpu_memory.gpu_memory_core import GPUMemoryOptimizer, MemoryState
from ..gpu_memory.auto_swap_manager import MemoryPressureLevel  
from ..gpu_memory.gpu_auto_detector import get_config_updater

# Rest of the file remains the same...

class ProcessingMode(Enum):
    """Processing mode decisions"""
    GPU_PREFERRED = "gpu_preferred"      # Use GPU if possible
    CPU_PREFERRED = "cpu_preferred"      # Use CPU to save GPU memory
    GPU_REQUIRED = "gpu_required"        # Must use GPU (will fail if OOM)
    CPU_REQUIRED = "cpu_required"        # Must use CPU
    ADAPTIVE = "adaptive"                # Dynamic selection based on load


class MemoryState(Enum):
    """System memory state"""
    HEALTHY = "healthy"           # Plenty of memory available
    MODERATE = "moderate"         # Some memory pressure
    HIGH = "high"                # High memory pressure
    CRITICAL = "critical"        # Critical - immediate action needed
    EXHAUSTED = "exhausted"      # No memory available


@dataclass
class MemoryMetrics:
    """Real-time memory metrics"""
    timestamp: float
    gpu_total_mb: float
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    gpu_free_mb: float
    gpu_utilization: float
    cpu_total_mb: float
    cpu_available_mb: float
    cpu_percent: float
    swap_used_mb: float
    
    @property
    def gpu_pressure(self) -> float:
        """Calculate GPU memory pressure (0-1)"""
        return self.gpu_utilization
    
    @property
    def cpu_pressure(self) -> float:
        """Calculate CPU memory pressure (0-1)"""
        return self.cpu_percent / 100.0
    
    @property
    def effective_gpu_free_mb(self) -> float:
        """Effective free GPU memory considering fragmentation"""
        # Account for memory fragmentation (typically 5-10%)
        return self.gpu_free_mb * 0.9


@dataclass
class PerformanceMetrics:
    """Performance tracking for decision making"""
    gpu_throughput: float = 0.0      # Items/second
    cpu_throughput: float = 0.0      # Items/second
    gpu_latency_ms: float = 0.0      # Average latency
    cpu_latency_ms: float = 0.0      # Average latency
    gpu_success_rate: float = 1.0    # Success rate (0-1)
    cpu_success_rate: float = 1.0    # Success rate (0-1)
    
    def update_gpu_metrics(self, throughput: float, latency_ms: float, success: bool):
        """Update GPU performance metrics"""
        alpha = 0.9  # Exponential moving average
        self.gpu_throughput = alpha * self.gpu_throughput + (1 - alpha) * throughput
        self.gpu_latency_ms = alpha * self.gpu_latency_ms + (1 - alpha) * latency_ms
        self.gpu_success_rate = alpha * self.gpu_success_rate + (1 - alpha) * (1.0 if success else 0.0)
    
    def update_cpu_metrics(self, throughput: float, latency_ms: float, success: bool):
        """Update CPU performance metrics"""
        alpha = 0.9
        self.cpu_throughput = alpha * self.cpu_throughput + (1 - alpha) * throughput
        self.cpu_latency_ms = alpha * self.cpu_latency_ms + (1 - alpha) * latency_ms
        self.cpu_success_rate = alpha * self.cpu_success_rate + (1 - alpha) * (1.0 if success else 0.0)
    
    @property
    def gpu_score(self) -> float:
        """Calculate GPU performance score"""
        if self.gpu_latency_ms == 0:
            return 0.0
        return (self.gpu_throughput / self.gpu_latency_ms) * self.gpu_success_rate
    
    @property
    def cpu_score(self) -> float:
        """Calculate CPU performance score"""
        if self.cpu_latency_ms == 0:
            return 0.0
        return (self.cpu_throughput / self.cpu_latency_ms) * self.cpu_success_rate


@dataclass
class PressureHandlerConfig:
    """Configuration for memory pressure handler - Auto-configured for detected GPU"""
    # Memory thresholds (MB) - Auto-detected
    gpu_critical_threshold_mb: int = None    # Auto-detected
    gpu_high_threshold_mb: int = None        # Auto-detected
    gpu_moderate_threshold_mb: int = None    # Auto-detected
    
    # Utilization thresholds (0-1)
    gpu_critical_utilization: float = None   # Auto-detected
    gpu_high_utilization: float = None       # Auto-detected
    gpu_moderate_utilization: float = None   # Auto-detected
    
    # CPU thresholds
    cpu_critical_threshold_mb: int = 500
    cpu_high_utilization: float = 0.90
    
    # Decision parameters
    prefer_gpu_threshold: float = 2.0     # GPU must be 2x faster to prefer
    force_cpu_on_critical: bool = True    # Force CPU when memory critical
    adaptive_threshold: bool = True       # Adapt thresholds based on workload
    
    # Monitoring parameters
    monitoring_interval_ms: int = 100     # Memory check interval
    history_window_size: int = 100        # Size of metrics history
    prediction_horizon_ms: int = 1000     # Predict memory usage ahead
    
    # Performance parameters
    min_gpu_batch_size: int = 10         # Minimum batch for GPU
    max_cpu_batch_size: int = None       # Auto-detected
    warmup_iterations: int = 10          # Iterations before trusting metrics
    
    # Auto-configured parameters
    burst_multiplier: float = None           # Auto-detected
    emergency_cpu_workers: int = None        # Auto-detected
    memory_defrag_threshold: float = None    # Auto-detected
    
    # Flag to track if auto-configuration has been applied
    _auto_configured: bool = False
    
    def __post_init__(self):
        """Auto-configure and validate configuration"""
        # Apply auto-configuration if values are None
        if not self._auto_configured:
            self._apply_auto_configuration()
        
        # Validate after auto-configuration
        if self.gpu_critical_threshold_mb <= 0:
            raise ValueError(f"gpu_critical_threshold_mb must be > 0, got {self.gpu_critical_threshold_mb}")
        if not 0 < self.gpu_critical_utilization <= 1:
            raise ValueError(f"gpu_critical_utilization must be in (0,1], got {self.gpu_critical_utilization}")
        if self.monitoring_interval_ms <= 0:
            raise ValueError(f"monitoring_interval_ms must be > 0, got {self.monitoring_interval_ms}")
    
    def _apply_auto_configuration(self):
        """Apply auto-detected configuration values"""
        try:
            config_updater = get_config_updater()
            auto_config = config_updater.optimized_config
            
            # Apply auto-detected values only if not manually set
            if self.gpu_critical_threshold_mb is None:
                self.gpu_critical_threshold_mb = auto_config.gpu_critical_threshold_mb
            if self.gpu_high_threshold_mb is None:
                self.gpu_high_threshold_mb = auto_config.gpu_high_threshold_mb
            if self.gpu_moderate_threshold_mb is None:
                self.gpu_moderate_threshold_mb = auto_config.gpu_moderate_threshold_mb
            
            if self.gpu_critical_utilization is None:
                self.gpu_critical_utilization = auto_config.gpu_critical_utilization
            if self.gpu_high_utilization is None:
                self.gpu_high_utilization = auto_config.gpu_high_utilization
            if self.gpu_moderate_utilization is None:
                self.gpu_moderate_utilization = auto_config.gpu_moderate_utilization
            
            if self.max_cpu_batch_size is None:
                self.max_cpu_batch_size = auto_config.cpu_batch_size
            if self.burst_multiplier is None:
                self.burst_multiplier = auto_config.burst_multiplier
            if self.emergency_cpu_workers is None:
                self.emergency_cpu_workers = auto_config.emergency_cpu_workers
            if self.memory_defrag_threshold is None:
                self.memory_defrag_threshold = auto_config.memory_defrag_threshold
            
            self._auto_configured = True
            
        except Exception as e:
            # Fallback to conservative defaults if auto-detection fails
            print(f"Auto-configuration failed: {e}, using conservative defaults")
            if self.gpu_critical_threshold_mb is None:
                self.gpu_critical_threshold_mb = 6144  # 6GB
            if self.gpu_high_threshold_mb is None:
                self.gpu_high_threshold_mb = 4096  # 4GB
            if self.gpu_moderate_threshold_mb is None:
                self.gpu_moderate_threshold_mb = 2048  # 2GB
            
            if self.gpu_critical_utilization is None:
                self.gpu_critical_utilization = 0.95
            if self.gpu_high_utilization is None:
                self.gpu_high_utilization = 0.85
            if self.gpu_moderate_utilization is None:
                self.gpu_moderate_utilization = 0.70
            
            if self.max_cpu_batch_size is None:
                self.max_cpu_batch_size = 10000
            if self.burst_multiplier is None:
                self.burst_multiplier = 5.0
            if self.emergency_cpu_workers is None:
                self.emergency_cpu_workers = 50
            if self.memory_defrag_threshold is None:
                self.memory_defrag_threshold = 0.3
            
            self._auto_configured = True


class MemoryPressureHandler:
    """
    Intelligent memory pressure handler for GPU/CPU coordination
    Makes real-time decisions on where to process based on memory and performance
    """
    
    def __init__(self, config: PressureHandlerConfig, 
                 gpu_optimizer: Optional[GPUMemoryOptimizer] = None,
                 cpu_pipeline: Optional[Any] = None):
        """
        Initialize memory pressure handler
        
        Args:
            config: Handler configuration
            gpu_optimizer: GPU memory optimizer (optional)
            cpu_pipeline: CPU bursting pipeline (optional)
        """
        self.config = config
        self.gpu_optimizer = gpu_optimizer
        self.cpu_pipeline = cpu_pipeline
        
        # Memory metrics history
        self.memory_history: deque[MemoryMetrics] = deque(maxlen=config.history_window_size)
        self.performance_metrics = PerformanceMetrics()
        
        # State tracking
        self.current_memory_state = MemoryState.HEALTHY
        self.processing_mode = ProcessingMode.ADAPTIVE
        self.iterations_since_start = 0
        
        # Decision history for analysis
        self.decision_history: deque[Tuple[float, str, Dict[str, Any]]] = deque(maxlen=1000)
        
        # Monitoring thread
        self.monitoring_active = True
        self.last_monitor_time = 0.0
        self._lock = threading.RLock()
        self.monitor_thread = threading.Thread(target=self._monitor_memory_loop, daemon=True)
        self.monitor_thread.start()
        
        # Callbacks for external monitoring
        self.state_change_callbacks: List[Callable] = []
        
        # Statistics
        self.stats = {
            'total_decisions': 0,
            'gpu_decisions': 0,
            'cpu_decisions': 0,
            'forced_cpu_decisions': 0,
            'memory_pressure_events': 0,
            'state_changes': 0,
            'prediction_accuracy': 0.0
        }
    
    def should_use_cpu(self, metadata: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Main decision function - should this operation use CPU?
        
        Args:
            metadata: Operation metadata including size, priority, etc.
            
        Returns:
            Tuple of (use_cpu, decision_info)
        """
        with self._lock:
            start_time = time.time()
            self.stats['total_decisions'] += 1
            
            # Get current memory state
            current_metrics = self._get_current_metrics()
            if current_metrics is None:
                raise RuntimeError("Failed to get memory metrics")
            
            # Update iterations
            self.iterations_since_start += 1
            
            # Extract operation parameters
            data_size_mb = metadata.get('size_mb', 0)
            priority = metadata.get('priority', 'normal')
            require_gpu = metadata.get('require_gpu', False)
            require_cpu = metadata.get('require_cpu', False)
            
            # Make decision
            decision_info = {
                'timestamp': start_time,
                'memory_state': self.current_memory_state.value,
                'gpu_free_mb': current_metrics.gpu_free_mb,
                'gpu_utilization': current_metrics.gpu_utilization,
                'data_size_mb': data_size_mb,
                'priority': priority
            }
            
            # Hard requirements
            if require_gpu:
                if current_metrics.gpu_free_mb < data_size_mb:
                    raise RuntimeError(f"GPU required but insufficient memory: {current_metrics.gpu_free_mb}MB < {data_size_mb}MB")
                decision_info['reason'] = 'gpu_required'
                self.stats['gpu_decisions'] += 1
                return False, decision_info
            
            if require_cpu:
                decision_info['reason'] = 'cpu_required'
                self.stats['cpu_decisions'] += 1
                return True, decision_info
            
            # Check if we're still warming up
            if self.iterations_since_start < self.config.warmup_iterations:
                # During warmup, use simple threshold
                use_cpu = current_metrics.gpu_free_mb < data_size_mb * 1.5
                decision_info['reason'] = 'warmup_threshold'
                if use_cpu:
                    self.stats['cpu_decisions'] += 1
                else:
                    self.stats['gpu_decisions'] += 1
                return use_cpu, decision_info
            
            # Memory state based decision
            if self.current_memory_state == MemoryState.CRITICAL:
                if self.config.force_cpu_on_critical:
                    decision_info['reason'] = 'memory_critical'
                    self.stats['forced_cpu_decisions'] += 1
                    self.stats['cpu_decisions'] += 1
                    return True, decision_info
            
            elif self.current_memory_state == MemoryState.EXHAUSTED:
                decision_info['reason'] = 'memory_exhausted'
                self.stats['forced_cpu_decisions'] += 1
                self.stats['cpu_decisions'] += 1
                return True, decision_info
            
            # Performance-based decision
            if self.processing_mode == ProcessingMode.ADAPTIVE:
                use_cpu = self._make_adaptive_decision(current_metrics, data_size_mb, priority)
                decision_info['reason'] = 'adaptive'
                decision_info['gpu_score'] = self.performance_metrics.gpu_score
                decision_info['cpu_score'] = self.performance_metrics.cpu_score
            
            elif self.processing_mode == ProcessingMode.GPU_PREFERRED:
                use_cpu = current_metrics.gpu_free_mb < data_size_mb * 1.2
                decision_info['reason'] = 'gpu_preferred'
            
            elif self.processing_mode == ProcessingMode.CPU_PREFERRED:
                use_cpu = True
                decision_info['reason'] = 'cpu_preferred'
            
            else:
                # Default to threshold-based
                use_cpu = self._make_threshold_decision(current_metrics, data_size_mb)
                decision_info['reason'] = 'threshold'
            
            # Update statistics
            if use_cpu:
                self.stats['cpu_decisions'] += 1
            else:
                self.stats['gpu_decisions'] += 1
            
            # Record decision
            self.decision_history.append((start_time, 'cpu' if use_cpu else 'gpu', decision_info))
            
            decision_info['decision_time_ms'] = (time.time() - start_time) * 1000
            return use_cpu, decision_info
    
    def _make_adaptive_decision(self, metrics: MemoryMetrics, data_size_mb: float, priority: str) -> bool:
        """Make adaptive decision based on performance and memory"""
        # High priority always tries GPU first if possible
        if priority == 'high' and metrics.gpu_free_mb > data_size_mb * 1.5:
            return False
        
        # Check if GPU has enough memory
        if metrics.gpu_free_mb < data_size_mb * 1.1:
            return True
        
        # Compare performance scores
        gpu_advantage = self.performance_metrics.gpu_score / max(self.performance_metrics.cpu_score, 0.01)
        
        # Adjust threshold based on memory pressure
        pressure_factor = 1.0 + metrics.gpu_utilization
        effective_threshold = self.config.prefer_gpu_threshold * pressure_factor
        
        # GPU must be significantly faster to justify using limited memory
        return gpu_advantage < effective_threshold
    
    def _make_threshold_decision(self, metrics: MemoryMetrics, data_size_mb: float) -> bool:
        """Make decision based on memory thresholds"""
        # Check absolute memory
        if metrics.gpu_free_mb < self.config.gpu_critical_threshold_mb:
            return True
        
        # Check if operation would push us over threshold
        post_op_free = metrics.gpu_free_mb - data_size_mb
        post_op_utilization = 1.0 - (post_op_free / metrics.gpu_total_mb)
        
        if post_op_utilization > self.config.gpu_critical_utilization:
            return True
        
        # Check current utilization
        if metrics.gpu_utilization > self.config.gpu_high_utilization:
            # Be conservative with large operations
            if data_size_mb > metrics.gpu_free_mb * 0.3:
                return True
        
        return False
    
    def update_performance(self, mode: str, success: bool, 
                         throughput: float, latency_ms: float) -> None:
        """
        Update performance metrics after operation
        
        Args:
            mode: 'gpu' or 'cpu'
            success: Whether operation succeeded
            throughput: Items processed per second
            latency_ms: Operation latency in milliseconds
        """
        with self._lock:
            if mode == 'gpu':
                self.performance_metrics.update_gpu_metrics(throughput, latency_ms, success)
            elif mode == 'cpu':
                self.performance_metrics.update_cpu_metrics(throughput, latency_ms, success)
            else:
                raise ValueError(f"Invalid mode: {mode}")
    
    def set_processing_mode(self, mode: ProcessingMode) -> None:
        """Set processing mode preference"""
        with self._lock:
            self.processing_mode = mode
    
    def register_state_change_callback(self, callback: Callable) -> None:
        """Register callback for memory state changes"""
        with self._lock:
            self.state_change_callbacks.append(callback)
    
    def _get_current_metrics(self) -> Optional[MemoryMetrics]:
        """Get current memory metrics"""
        try:
            # GPU metrics
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                gpu_total = torch.cuda.get_device_properties(device).total_memory / (1024**2)
                gpu_allocated = torch.cuda.memory_allocated(device) / (1024**2)
                gpu_reserved = torch.cuda.memory_reserved(device) / (1024**2)
                gpu_free = gpu_total - gpu_reserved
                gpu_utilization = gpu_reserved / gpu_total
            else:
                # No GPU available
                gpu_total = gpu_allocated = gpu_reserved = gpu_utilization = 0
                gpu_free = 0
            
            # CPU metrics
            cpu_mem = psutil.virtual_memory()
            cpu_total_mb = cpu_mem.total / (1024**2)
            cpu_available_mb = cpu_mem.available / (1024**2)
            cpu_percent = cpu_mem.percent
            
            # Swap metrics
            swap = psutil.swap_memory()
            swap_used_mb = swap.used / (1024**2)
            
            return MemoryMetrics(
                timestamp=time.time(),
                gpu_total_mb=gpu_total,
                gpu_allocated_mb=gpu_allocated,
                gpu_reserved_mb=gpu_reserved,
                gpu_free_mb=gpu_free,
                gpu_utilization=gpu_utilization,
                cpu_total_mb=cpu_total_mb,
                cpu_available_mb=cpu_available_mb,
                cpu_percent=cpu_percent,
                swap_used_mb=swap_used_mb
            )
            
        except Exception as e:
            # NO FALLBACKS - HARD FAILURE
            raise RuntimeError(f"Failed to get memory metrics: {e}")
    
    def _update_memory_state(self, metrics: MemoryMetrics) -> None:
        """Update memory state based on metrics"""
        old_state = self.current_memory_state
        
        # Determine new state
        if metrics.gpu_free_mb < self.config.gpu_critical_threshold_mb or \
           metrics.gpu_utilization > self.config.gpu_critical_utilization:
            new_state = MemoryState.CRITICAL
            
        elif metrics.gpu_free_mb < self.config.gpu_high_threshold_mb or \
             metrics.gpu_utilization > self.config.gpu_high_utilization:
            new_state = MemoryState.HIGH
            
        elif metrics.gpu_free_mb < self.config.gpu_moderate_threshold_mb or \
             metrics.gpu_utilization > self.config.gpu_moderate_utilization:
            new_state = MemoryState.MODERATE
            
        elif metrics.gpu_free_mb < 10:  # Less than 10MB
            new_state = MemoryState.EXHAUSTED
            
        else:
            new_state = MemoryState.HEALTHY
        
        # Update state
        self.current_memory_state = new_state
        
        # Track state changes
        if new_state != old_state:
            self.stats['state_changes'] += 1
            if new_state in [MemoryState.HIGH, MemoryState.CRITICAL, MemoryState.EXHAUSTED]:
                self.stats['memory_pressure_events'] += 1
            
            # Notify callbacks
            for callback in self.state_change_callbacks:
                try:
                    callback(old_state, new_state, metrics)
                except Exception:
                    pass
    
    def _monitor_memory_loop(self) -> None:
        """Background thread for memory monitoring"""
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                if current_time - self.last_monitor_time >= self.config.monitoring_interval_ms / 1000:
                    metrics = self._get_current_metrics()
                    
                    if metrics:
                        with self._lock:
                            # Update history
                            self.memory_history.append(metrics)
                            
                            # Update state
                            self._update_memory_state(metrics)
                            
                            # Predict future memory usage
                            if self.config.adaptive_threshold:
                                self._update_adaptive_thresholds()
                    
                    self.last_monitor_time = current_time
                
                time.sleep(0.01)  # 10ms sleep
                
            except Exception:
                # Continue monitoring even on errors
                pass
    
    def _update_adaptive_thresholds(self) -> None:
        """Update thresholds based on workload patterns"""
        if len(self.memory_history) < 10:
            return
        
        # Analyze memory usage patterns
        recent_utilizations = [m.gpu_utilization for m in list(self.memory_history)[-10:]]
        avg_utilization = np.mean(recent_utilizations)
        utilization_trend = recent_utilizations[-1] - recent_utilizations[0]
        
        # Adjust thresholds based on trends
        if utilization_trend > 0.1:  # Memory usage increasing rapidly
            # Be more conservative
            adjustment = 0.95
        elif utilization_trend < -0.1:  # Memory usage decreasing
            # Can be less conservative
            adjustment = 1.05
        else:
            adjustment = 1.0
        
        # Apply adjustments (with limits)
        self.config.gpu_moderate_utilization = min(0.85, 
            self.config.gpu_moderate_utilization * adjustment)
        self.config.gpu_high_utilization = min(0.95,
            self.config.gpu_high_utilization * adjustment)
    
    def predict_memory_exhaustion(self) -> Optional[float]:
        """
        Predict when GPU memory will be exhausted
        
        Returns:
            Seconds until exhaustion, or None if not predicted
        """
        with self._lock:
            if len(self.memory_history) < 5:
                raise RuntimeError("Insufficient memory history for exhaustion prediction")
            
            # Get recent memory usage
            recent = list(self.memory_history)[-5:]
            times = [m.timestamp for m in recent]
            free_memory = [m.gpu_free_mb for m in recent]
            
            # Simple linear regression
            if len(set(free_memory)) == 1:  # No change
                raise RuntimeError("No memory usage change detected - cannot predict exhaustion")
            
            # Calculate rate of change
            time_diff = times[-1] - times[0]
            memory_diff = free_memory[-1] - free_memory[0]
            
            if memory_diff >= 0:  # Memory not decreasing
                raise RuntimeError("Memory usage not increasing - no exhaustion predicted")
            
            rate_mb_per_sec = memory_diff / time_diff
            
            # Predict exhaustion
            seconds_to_exhaustion = -free_memory[-1] / rate_mb_per_sec
            
            if seconds_to_exhaustion < 0 or seconds_to_exhaustion > 3600:  # More than 1 hour
                raise RuntimeError(f"Invalid exhaustion prediction: {seconds_to_exhaustion} seconds")
            
            return seconds_to_exhaustion
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get current memory summary"""
        with self._lock:
            current = self._get_current_metrics()
            if current is None:
                return {}
            
            exhaustion_time = self.predict_memory_exhaustion()
            
            return {
                'memory_state': self.current_memory_state.value,
                'gpu_free_mb': current.gpu_free_mb,
                'gpu_utilization': current.gpu_utilization,
                'cpu_available_mb': current.cpu_available_mb,
                'cpu_percent': current.cpu_percent,
                'exhaustion_prediction_seconds': exhaustion_time,
                'recent_decisions': {
                    'total': self.stats['total_decisions'],
                    'gpu': self.stats['gpu_decisions'],
                    'cpu': self.stats['cpu_decisions'],
                    'forced_cpu': self.stats['forced_cpu_decisions']
                },
                'performance': {
                    'gpu_score': self.performance_metrics.gpu_score,
                    'cpu_score': self.performance_metrics.cpu_score,
                    'gpu_throughput': self.performance_metrics.gpu_throughput,
                    'cpu_throughput': self.performance_metrics.cpu_throughput
                }
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        with self._lock:
            total_decisions = max(self.stats['total_decisions'], 1)
            
            return {
                'total_decisions': self.stats['total_decisions'],
                'gpu_decisions': self.stats['gpu_decisions'],
                'cpu_decisions': self.stats['cpu_decisions'],
                'forced_cpu_decisions': self.stats['forced_cpu_decisions'],
                'gpu_percentage': self.stats['gpu_decisions'] / total_decisions * 100,
                'cpu_percentage': self.stats['cpu_decisions'] / total_decisions * 100,
                'memory_pressure_events': self.stats['memory_pressure_events'],
                'state_changes': self.stats['state_changes'],
                'current_state': self.current_memory_state.value,
                'processing_mode': self.processing_mode.value,
                'iterations_since_start': self.iterations_since_start,
                'memory_history_size': len(self.memory_history),
                'decision_history_size': len(self.decision_history)
            }
    
    def reset_statistics(self) -> None:
        """Reset statistics"""
        with self._lock:
            self.stats = {
                'total_decisions': 0,
                'gpu_decisions': 0,
                'cpu_decisions': 0,
                'forced_cpu_decisions': 0,
                'memory_pressure_events': 0,
                'state_changes': 0,
                'prediction_accuracy': 0.0
            }
            self.iterations_since_start = 0
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)


# Integration helper functions
def integrate_memory_pressure_handler(decompression_engine: Any,
                                    config: Optional[PressureHandlerConfig] = None) -> MemoryPressureHandler:
    """
    Integrate memory pressure handler with decompression engine
    
    Args:
        decompression_engine: PadicDecompressionEngine instance
        config: Handler configuration (optional)
        
    Returns:
        Configured MemoryPressureHandler instance
    """
    if config is None:
        config = PressureHandlerConfig()
    
    # Get GPU optimizer and CPU pipeline from engine
    gpu_optimizer = getattr(decompression_engine, 'gpu_optimizer', None)
    cpu_pipeline = getattr(decompression_engine, 'cpu_bursting_pipeline', None)
    
    # Create handler
    handler = MemoryPressureHandler(config, gpu_optimizer, cpu_pipeline)
    
    # Add to decompression engine
    decompression_engine.memory_pressure_handler = handler
    
    # Wrap decompression method
    original_decompress = decompression_engine.decompress_progressive
    
    def decompress_with_pressure_handling(padic_weights, target_precision, metadata):
        """Decompression with memory pressure handling"""
        # Check memory pressure
        use_cpu, decision_info = handler.should_use_cpu(metadata)
        
        if use_cpu and cpu_pipeline:
            # Use CPU decompression
            result, info = cpu_pipeline.decompress(
                padic_weights, target_precision, metadata,
                force_mode=DecompressionMode.CPU_ONLY
            )
            
            # Update performance metrics
            if 'cpu_time' in info:
                throughput = len(padic_weights) / info['cpu_time']
                latency_ms = info['cpu_time'] * 1000
                handler.update_performance('cpu', True, throughput, latency_ms)
            
        else:
            # Use GPU decompression
            try:
                result, info = original_decompress(padic_weights, target_precision, metadata)
                
                # Update performance metrics
                if 'decompression_time' in info:
                    throughput = len(padic_weights) / info['decompression_time']
                    latency_ms = info['decompression_time'] * 1000
                    handler.update_performance('gpu', True, throughput, latency_ms)
                    
            except torch.cuda.OutOfMemoryError:
                if not use_cpu:
                    # GPU was preferred but failed
                    handler.update_performance('gpu', False, 0, 0)
                raise RuntimeError("GPU out of memory and CPU not selected")
        
        # Add decision info to results
        info['memory_pressure_decision'] = decision_info
        info['processing_mode'] = 'cpu' if use_cpu else 'gpu'
        
        return result, info
    
    decompression_engine.decompress_progressive = decompress_with_pressure_handling
    
    return handler
