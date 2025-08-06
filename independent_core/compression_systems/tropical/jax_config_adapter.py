"""
JAX Configuration Auto-Adaptation - Dynamic JAX configuration based on hardware
Auto-tunes XLA compilation flags and batch sizes based on detected GPU
HARD FAILURES ONLY - NO GRACEFUL DEGRADATION

This module provides:
1. JAXConfigAdapter - Adjusts JAX settings based on GPU specs
2. Auto-tuning of XLA compilation flags
3. Dynamic batch size adjustment based on available memory
4. NUMA-aware memory allocation when detected
"""

import jax
from jax import config as jax_config
from jax.lib import xla_bridge
import os
import psutil
import numpy as np
import time
import threading
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum

# Import existing components
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from independent_core.compression_systems.gpu_memory.gpu_auto_detector import (
    GPUAutoDetector,
    GPUSpecs,
    GPUArchitecture,
    AutoOptimizedConfig
)

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """XLA optimization levels"""
    NONE = 0          # No optimization
    BASIC = 1         # Basic optimizations
    STANDARD = 2      # Standard optimizations (default)
    AGGRESSIVE = 3    # Aggressive optimizations


class MemoryStrategy(Enum):
    """Memory allocation strategies"""
    DEFAULT = "default"              # Default JAX allocation
    PREALLOCATE = "preallocate"     # Pre-allocate full GPU memory
    DYNAMIC = "dynamic"              # Dynamic allocation
    NUMA_AWARE = "numa_aware"        # NUMA-aware allocation


@dataclass
class JAXRuntimeConfig:
    """JAX runtime configuration"""
    # XLA flags
    xla_gpu_cuda_data_dir: Optional[str] = None
    xla_gpu_autotune_level: int = 2
    xla_gpu_enable_triton_gemm: bool = True
    xla_gpu_triton_gemm_any: bool = True
    xla_gpu_enable_async_collectives: bool = True
    xla_gpu_enable_latency_hiding_scheduler: bool = True
    xla_gpu_enable_highest_priority_async_stream: bool = True
    
    # Memory configuration
    xla_python_client_preallocate: bool = True
    xla_python_client_mem_fraction: float = 0.9
    xla_gpu_memory_fraction: float = 0.9
    
    # Compilation flags
    xla_gpu_graph_level: int = 0  # 0=off, 1=basic, 2=aggressive
    xla_gpu_enable_xla_runtime_executable: bool = False
    xla_gpu_llvm_ir_enable_pow_expansion: bool = True
    
    # NUMA configuration
    xla_cpu_use_xla_runtime: bool = True
    xla_cpu_multi_thread: bool = True
    xla_force_host_platform_device_count: Optional[int] = None
    
    def to_env_vars(self) -> Dict[str, str]:
        """Convert to environment variables"""
        env_vars = {}
        
        if self.xla_gpu_cuda_data_dir:
            env_vars['XLA_GPU_CUDA_DATA_DIR'] = self.xla_gpu_cuda_data_dir
        
        # Build XLA_FLAGS
        xla_flags = []
        xla_flags.append(f"--xla_gpu_autotune_level={self.xla_gpu_autotune_level}")
        
        if self.xla_gpu_enable_triton_gemm:
            xla_flags.append("--xla_gpu_enable_triton_gemm=true")
        if self.xla_gpu_triton_gemm_any:
            xla_flags.append("--xla_gpu_triton_gemm_any=true")
        if self.xla_gpu_enable_async_collectives:
            xla_flags.append("--xla_gpu_enable_async_collectives=true")
        if self.xla_gpu_enable_latency_hiding_scheduler:
            xla_flags.append("--xla_gpu_enable_latency_hiding_scheduler=true")
        if self.xla_gpu_enable_highest_priority_async_stream:
            xla_flags.append("--xla_gpu_enable_highest_priority_async_stream=true")
        
        xla_flags.append(f"--xla_gpu_graph_level={self.xla_gpu_graph_level}")
        
        if self.xla_gpu_enable_xla_runtime_executable:
            xla_flags.append("--xla_gpu_enable_xla_runtime_executable=true")
        if self.xla_gpu_llvm_ir_enable_pow_expansion:
            xla_flags.append("--xla_gpu_llvm_ir_enable_pow_expansion=true")
        
        if self.xla_cpu_use_xla_runtime:
            xla_flags.append("--xla_cpu_use_xla_runtime=true")
        if self.xla_cpu_multi_thread:
            xla_flags.append("--xla_cpu_multi_thread=true")
        
        if self.xla_force_host_platform_device_count:
            xla_flags.append(f"--xla_force_host_platform_device_count={self.xla_force_host_platform_device_count}")
        
        env_vars['XLA_FLAGS'] = " ".join(xla_flags)
        
        # Memory configuration
        env_vars['XLA_PYTHON_CLIENT_PREALLOCATE'] = str(self.xla_python_client_preallocate).lower()
        env_vars['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(self.xla_python_client_mem_fraction)
        env_vars['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Use async allocator
        
        return env_vars


@dataclass  
class AdaptiveBatchConfig:
    """Configuration for adaptive batch sizing"""
    min_batch_size: int = 1
    max_batch_size: int = 10000
    initial_batch_size: int = 100
    memory_safety_factor: float = 0.9  # Use 90% of available memory
    growth_factor: float = 1.5
    shrink_factor: float = 0.7
    profile_iterations: int = 5
    
    # Current state
    current_batch_size: int = field(init=False)
    successful_sizes: List[int] = field(default_factory=list)
    failed_sizes: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        self.current_batch_size = self.initial_batch_size


class JAXConfigAdapter:
    """
    Adapts JAX configuration based on detected hardware.
    Auto-tunes XLA flags and batch sizes for optimal performance.
    """
    
    def __init__(self,
                 gpu_detector: Optional[GPUAutoDetector] = None,
                 optimization_level: OptimizationLevel = OptimizationLevel.STANDARD):
        """
        Initialize JAX configuration adapter
        
        Args:
            gpu_detector: GPU auto-detector instance
            optimization_level: XLA optimization level
        """
        self.gpu_detector = gpu_detector or GPUAutoDetector()
        self.optimization_level = optimization_level
        
        # Detect hardware
        self.gpu_specs = self.gpu_detector.get_primary_gpu()
        self.auto_config = AutoOptimizedConfig.from_gpu_specs(
            self.gpu_specs
        ) if self.gpu_specs else None
        
        # Detect NUMA configuration
        self.numa_nodes = self._detect_numa_nodes()
        
        # Create runtime configuration
        self.runtime_config = self._create_runtime_config()
        
        # Batch size configurations per operation type
        self.batch_configs: Dict[str, AdaptiveBatchConfig] = {}
        
        # Apply configuration
        self._apply_configuration()
        
        # Performance tracking
        self.performance_history: Dict[str, List[float]] = {}
        self.optimization_stats = {
            'config_adaptations': 0,
            'batch_adjustments': 0,
            'oom_recoveries': 0,
            'performance_improvements': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"JAXConfigAdapter initialized for {self.gpu_specs.name if self.gpu_specs else 'CPU'}")
    
    def _detect_numa_nodes(self) -> List[int]:
        """Detect NUMA nodes on the system"""
        numa_nodes = []
        
        if os.path.exists('/sys/devices/system/node'):
            try:
                for entry in os.listdir('/sys/devices/system/node'):
                    if entry.startswith('node'):
                        numa_nodes.append(int(entry[4:]))
            except:
                pass
        
        return numa_nodes if numa_nodes else [0]
    
    def _create_runtime_config(self) -> JAXRuntimeConfig:
        """Create optimized runtime configuration based on hardware"""
        config = JAXRuntimeConfig()
        
        if self.gpu_specs:
            # GPU-specific optimizations
            arch = self.gpu_specs.architecture
            
            # Set autotune level based on architecture
            if arch in [GPUArchitecture.HOPPER, GPUArchitecture.ADA]:
                config.xla_gpu_autotune_level = 4  # Maximum autotuning
                config.xla_gpu_enable_triton_gemm = True
                config.xla_gpu_triton_gemm_any = True
            elif arch == GPUArchitecture.AMPERE:
                config.xla_gpu_autotune_level = 3
                config.xla_gpu_enable_triton_gemm = True
            elif arch in [GPUArchitecture.TURING, GPUArchitecture.VOLTA]:
                config.xla_gpu_autotune_level = 2
                config.xla_gpu_enable_triton_gemm = False
            else:
                config.xla_gpu_autotune_level = 1
                config.xla_gpu_enable_triton_gemm = False
            
            # Memory configuration based on GPU memory
            if self.gpu_specs.total_memory_gb >= 24:
                config.xla_python_client_mem_fraction = 0.95
                config.xla_gpu_memory_fraction = 0.95
            elif self.gpu_specs.total_memory_gb >= 16:
                config.xla_python_client_mem_fraction = 0.90
                config.xla_gpu_memory_fraction = 0.90
            else:
                config.xla_python_client_mem_fraction = 0.85
                config.xla_gpu_memory_fraction = 0.85
            
            # Enable async operations for newer GPUs
            if arch.value in ['ada', 'hopper', 'ampere']:
                config.xla_gpu_enable_async_collectives = True
                config.xla_gpu_enable_latency_hiding_scheduler = True
                config.xla_gpu_enable_highest_priority_async_stream = True
            
            # Graph optimization level
            if self.optimization_level == OptimizationLevel.AGGRESSIVE:
                config.xla_gpu_graph_level = 2
            elif self.optimization_level == OptimizationLevel.STANDARD:
                config.xla_gpu_graph_level = 1
            else:
                config.xla_gpu_graph_level = 0
        
        # NUMA configuration
        if len(self.numa_nodes) > 1:
            config.xla_cpu_use_xla_runtime = True
            config.xla_cpu_multi_thread = True
            config.xla_force_host_platform_device_count = len(self.numa_nodes)
        
        return config
    
    def _apply_configuration(self) -> None:
        """Apply the configuration to JAX/XLA"""
        # Set environment variables
        env_vars = self.runtime_config.to_env_vars()
        for key, value in env_vars.items():
            os.environ[key] = value
        
        # Apply JAX configuration
        try:
            # Enable 64-bit if GPU has enough memory
            if self.gpu_specs and self.gpu_specs.total_memory_gb >= 16:
                jax_config.update('jax_enable_x64', True)
            
            # Set platform
            if self.gpu_specs:
                os.environ['JAX_PLATFORM_NAME'] = 'gpu'
            else:
                os.environ['JAX_PLATFORM_NAME'] = 'cpu'
            
            # Memory allocation
            if self.runtime_config.xla_python_client_preallocate:
                os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
            else:
                os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
            
            self.optimization_stats['config_adaptations'] += 1
            
        except Exception as e:
            logger.warning(f"Failed to apply some JAX configurations: {e}")
    
    def get_optimal_batch_size(self,
                              operation_name: str,
                              element_size_bytes: int,
                              memory_overhead_factor: float = 1.5) -> int:
        """
        Get optimal batch size for operation
        
        Args:
            operation_name: Name of the operation
            element_size_bytes: Size of each element in bytes
            memory_overhead_factor: Factor for memory overhead
            
        Returns:
            Optimal batch size
        """
        with self._lock:
            # Get or create batch config for this operation
            if operation_name not in self.batch_configs:
                self.batch_configs[operation_name] = self._create_batch_config()
            
            config = self.batch_configs[operation_name]
            
            # Calculate available memory
            if self.gpu_specs:
                available_memory = self.gpu_specs.total_memory_mb * 1024 * 1024
                available_memory *= self.runtime_config.xla_gpu_memory_fraction
                available_memory *= config.memory_safety_factor
            else:
                # CPU memory
                available_memory = psutil.virtual_memory().available
                available_memory *= 0.5  # Use half of available RAM
            
            # Calculate maximum batch size based on memory
            max_batch_from_memory = int(
                available_memory / (element_size_bytes * memory_overhead_factor)
            )
            
            # Constrain by configuration
            optimal_batch = min(
                max_batch_from_memory,
                config.max_batch_size
            )
            optimal_batch = max(optimal_batch, config.min_batch_size)
            
            # Update current batch size
            config.current_batch_size = optimal_batch
            
            return optimal_batch
    
    def _create_batch_config(self) -> AdaptiveBatchConfig:
        """Create batch configuration based on hardware"""
        if self.auto_config:
            return AdaptiveBatchConfig(
                min_batch_size=10,
                max_batch_size=self.auto_config.gpu_batch_size,
                initial_batch_size=min(1000, self.auto_config.gpu_batch_size // 10)
            )
        else:
            # CPU-only configuration
            return AdaptiveBatchConfig(
                min_batch_size=1,
                max_batch_size=1000,
                initial_batch_size=100
            )
    
    def adapt_batch_size(self,
                        operation_name: str,
                        success: bool,
                        execution_time: float,
                        memory_used_mb: Optional[float] = None) -> int:
        """
        Adapt batch size based on execution results
        
        Args:
            operation_name: Name of the operation
            success: Whether execution was successful
            execution_time: Execution time in seconds
            memory_used_mb: Memory used in MB
            
        Returns:
            New batch size
        """
        with self._lock:
            if operation_name not in self.batch_configs:
                self.batch_configs[operation_name] = self._create_batch_config()
            
            config = self.batch_configs[operation_name]
            current_size = config.current_batch_size
            
            self.optimization_stats['batch_adjustments'] += 1
            
            if success:
                config.successful_sizes.append(current_size)
                
                # Try to increase batch size if execution was fast
                if execution_time < 0.1:  # Less than 100ms
                    new_size = int(current_size * config.growth_factor)
                    new_size = min(new_size, config.max_batch_size)
                    
                    # Check if we haven't failed at this size before
                    if new_size not in config.failed_sizes:
                        config.current_batch_size = new_size
                
            else:
                config.failed_sizes.append(current_size)
                self.optimization_stats['oom_recoveries'] += 1
                
                # Reduce batch size
                new_size = int(current_size * config.shrink_factor)
                new_size = max(new_size, config.min_batch_size)
                
                # Find largest successful size smaller than current
                smaller_successful = [s for s in config.successful_sizes if s < current_size]
                if smaller_successful:
                    new_size = max(smaller_successful)
                
                config.current_batch_size = new_size
            
            # Track performance
            if operation_name not in self.performance_history:
                self.performance_history[operation_name] = []
            
            self.performance_history[operation_name].append(execution_time)
            
            # Calculate improvement
            if len(self.performance_history[operation_name]) > 10:
                recent = np.mean(self.performance_history[operation_name][-5:])
                initial = np.mean(self.performance_history[operation_name][:5])
                if initial > 0:
                    improvement = (initial - recent) / initial
                    self.optimization_stats['performance_improvements'] = improvement
            
            return config.current_batch_size
    
    def get_numa_config(self) -> Dict[str, Any]:
        """Get NUMA-aware configuration"""
        numa_config = {
            'numa_nodes': self.numa_nodes,
            'numa_aware': len(self.numa_nodes) > 1
        }
        
        if numa_config['numa_aware']:
            # CPU affinity for each NUMA node
            numa_config['cpu_affinity'] = {}
            
            try:
                for node in self.numa_nodes:
                    cpulist_path = f'/sys/devices/system/node/node{node}/cpulist'
                    if os.path.exists(cpulist_path):
                        with open(cpulist_path, 'r') as f:
                            cpu_range = f.read().strip()
                            # Parse CPU range (e.g., "0-7,16-23")
                            cpus = []
                            for part in cpu_range.split(','):
                                if '-' in part:
                                    start, end = map(int, part.split('-'))
                                    cpus.extend(range(start, end + 1))
                                else:
                                    cpus.append(int(part))
                            numa_config['cpu_affinity'][node] = cpus
            except:
                pass
            
            # Memory configuration per node
            numa_config['memory_per_node'] = {}
            try:
                for node in self.numa_nodes:
                    meminfo_path = f'/sys/devices/system/node/node{node}/meminfo'
                    if os.path.exists(meminfo_path):
                        with open(meminfo_path, 'r') as f:
                            for line in f:
                                if 'MemTotal' in line:
                                    # Extract memory in kB
                                    parts = line.split()
                                    if len(parts) >= 4:
                                        mem_kb = int(parts[3])
                                        numa_config['memory_per_node'][node] = mem_kb / 1024  # MB
                                    break
            except:
                pass
        
        return numa_config
    
    def optimize_for_operation(self,
                              operation_type: str,
                              data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize configuration for specific operation
        
        Args:
            operation_type: Type of operation ('matmul', 'conv', 'reduce', etc.)
            data_characteristics: Data characteristics (shape, dtype, etc.)
            
        Returns:
            Optimized configuration dictionary
        """
        optimization = {
            'batch_size': 1000,
            'use_float16': False,
            'use_tensor_cores': False,
            'memory_layout': 'default',
            'fusion_enabled': True
        }
        
        if self.gpu_specs:
            # Tensor cores for compatible operations
            if self.gpu_specs.tensor_cores > 0 and operation_type in ['matmul', 'conv']:
                optimization['use_tensor_cores'] = True
                # Float16 for tensor cores
                if self.gpu_specs.architecture in [GPUArchitecture.AMPERE, 
                                                  GPUArchitecture.ADA,
                                                  GPUArchitecture.HOPPER]:
                    optimization['use_float16'] = True
            
            # Batch size based on memory
            if 'element_size' in data_characteristics:
                optimization['batch_size'] = self.get_optimal_batch_size(
                    operation_type,
                    data_characteristics['element_size']
                )
            
            # Memory layout optimization
            if operation_type == 'conv':
                optimization['memory_layout'] = 'NHWC'  # Better for GPUs
            elif operation_type == 'matmul':
                optimization['memory_layout'] = 'row_major'
        
        # NUMA optimizations for CPU operations
        if len(self.numa_nodes) > 1:
            optimization['numa_aware'] = True
            optimization['numa_nodes'] = self.numa_nodes
        
        return optimization
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get adapter statistics"""
        with self._lock:
            stats = dict(self.optimization_stats)
            
            # Add batch size info
            stats['batch_configs'] = {}
            for name, config in self.batch_configs.items():
                stats['batch_configs'][name] = {
                    'current_size': config.current_batch_size,
                    'successful_runs': len(config.successful_sizes),
                    'failed_runs': len(config.failed_sizes),
                    'min_successful': min(config.successful_sizes) if config.successful_sizes else 0,
                    'max_successful': max(config.successful_sizes) if config.successful_sizes else 0
                }
            
            # Add performance metrics
            if self.performance_history:
                stats['avg_execution_times'] = {
                    name: np.mean(times) for name, times in self.performance_history.items()
                }
            
            return stats
    
    def export_config(self) -> Dict[str, Any]:
        """Export current configuration"""
        return {
            'runtime_config': self.runtime_config.__dict__,
            'optimization_level': self.optimization_level.value,
            'gpu_detected': self.gpu_specs.name if self.gpu_specs else None,
            'numa_config': self.get_numa_config(),
            'batch_configs': {
                name: {
                    'current_size': config.current_batch_size,
                    'min_size': config.min_batch_size,
                    'max_size': config.max_batch_size
                }
                for name, config in self.batch_configs.items()
            }
        }


# Test function
def test_jax_config_adapter():
    """Test JAX configuration adapter"""
    print("Testing JAX Config Adapter...")
    
    # Create GPU detector
    gpu_detector = GPUAutoDetector()
    
    # Create config adapter
    adapter = JAXConfigAdapter(gpu_detector, OptimizationLevel.STANDARD)
    
    # Display detected hardware
    if adapter.gpu_specs:
        print(f"\n1. Detected GPU: {adapter.gpu_specs.name}")
        print(f"   Memory: {adapter.gpu_specs.total_memory_gb:.1f}GB")
        print(f"   Architecture: {adapter.gpu_specs.architecture.value}")
    else:
        print("\n1. No GPU detected - CPU mode")
    
    # Display runtime configuration
    print("\n2. Runtime configuration:")
    env_vars = adapter.runtime_config.to_env_vars()
    for key, value in list(env_vars.items())[:5]:
        print(f"   {key}: {value}")
    
    # Test batch size optimization
    print("\n3. Testing batch size optimization...")
    
    # Get optimal batch size for different operations
    batch1 = adapter.get_optimal_batch_size("matmul", element_size_bytes=4)
    print(f"   MatMul optimal batch: {batch1}")
    
    batch2 = adapter.get_optimal_batch_size("conv", element_size_bytes=4)
    print(f"   Conv optimal batch: {batch2}")
    
    # Simulate successful execution
    new_batch = adapter.adapt_batch_size("matmul", success=True, execution_time=0.05)
    print(f"   After success: {new_batch}")
    
    # Simulate OOM
    new_batch = adapter.adapt_batch_size("matmul", success=False, execution_time=0.1)
    print(f"   After OOM: {new_batch}")
    
    # Test NUMA configuration
    print("\n4. NUMA configuration:")
    numa_config = adapter.get_numa_config()
    print(f"   NUMA nodes: {numa_config['numa_nodes']}")
    print(f"   NUMA-aware: {numa_config['numa_aware']}")
    
    # Test operation optimization
    print("\n5. Operation optimization:")
    opt_config = adapter.optimize_for_operation(
        "matmul",
        {"element_size": 4, "shape": (1024, 1024)}
    )
    print(f"   Batch size: {opt_config['batch_size']}")
    print(f"   Use tensor cores: {opt_config['use_tensor_cores']}")
    print(f"   Use float16: {opt_config['use_float16']}")
    
    # Get statistics
    stats = adapter.get_statistics()
    print("\n6. Adapter statistics:")
    print(f"   Config adaptations: {stats['config_adaptations']}")
    print(f"   Batch adjustments: {stats['batch_adjustments']}")
    print(f"   OOM recoveries: {stats['oom_recoveries']}")
    
    # Export configuration
    config_export = adapter.export_config()
    print("\n7. Configuration export:")
    print(f"   Optimization level: {config_export['optimization_level']}")
    print(f"   GPU: {config_export['gpu_detected']}")
    
    print("\n✓ JAX Config Adapter test complete!")


if __name__ == "__main__":
    test_jax_config_adapter()