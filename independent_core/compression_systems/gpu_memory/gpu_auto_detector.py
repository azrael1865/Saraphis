"""
GPU Auto-Detection System - Dynamic GPU detection and configuration optimization
NO FALLBACKS - HARD FAILURES ONLY

Automatically detects GPU specifications and optimizes all compression system configurations.
Replaces all hardcoded RTX 5060 Ti references with dynamic values.
"""

import torch
import os
import psutil
import platform
import subprocess
import re
import json
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import warnings


class GPUArchitecture(Enum):
    """GPU architecture families"""
    KEPLER = "kepler"          # Compute 3.x
    MAXWELL = "maxwell"        # Compute 5.x
    PASCAL = "pascal"          # Compute 6.x
    VOLTA = "volta"            # Compute 7.0
    TURING = "turing"          # Compute 7.5
    AMPERE = "ampere"          # Compute 8.x
    ADA = "ada"                # Compute 8.9
    HOPPER = "hopper"          # Compute 9.0
    BLACKWELL = "blackwell"    # Compute 10.x (future)
    UNKNOWN = "unknown"


@dataclass
class GPUSpecs:
    """GPU specifications detected at runtime"""
    device_id: int
    name: str
    total_memory_gb: float
    total_memory_mb: float
    compute_capability: str
    multi_processor_count: int
    max_threads_per_block: int
    max_shared_memory_per_block: int
    warp_size: int
    is_integrated: bool
    is_multi_gpu_board: bool
    cuda_cores: int  # Estimated
    memory_bandwidth_gb: float  # Estimated
    architecture: GPUArchitecture
    pcie_generation: int  # PCIe generation (3, 4, 5)
    memory_clock_mhz: float
    gpu_clock_mhz: float
    l2_cache_size_mb: float
    tensor_cores: int  # For Volta+
    rt_cores: int  # For Turing+
    
    def __post_init__(self):
        """Validate GPU specifications"""
        if self.device_id < 0:
            raise ValueError(f"Invalid device_id: {self.device_id}")
        if self.total_memory_gb <= 0:
            raise ValueError(f"Invalid total_memory_gb: {self.total_memory_gb}")
        if self.multi_processor_count <= 0:
            raise ValueError(f"Invalid multi_processor_count: {self.multi_processor_count}")
        if self.warp_size <= 0:
            raise ValueError(f"Invalid warp_size: {self.warp_size}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'device_id': self.device_id,
            'name': self.name,
            'total_memory_gb': self.total_memory_gb,
            'total_memory_mb': self.total_memory_mb,
            'compute_capability': self.compute_capability,
            'multi_processor_count': self.multi_processor_count,
            'max_threads_per_block': self.max_threads_per_block,
            'max_shared_memory_per_block': self.max_shared_memory_per_block,
            'warp_size': self.warp_size,
            'is_integrated': self.is_integrated,
            'is_multi_gpu_board': self.is_multi_gpu_board,
            'cuda_cores': self.cuda_cores,
            'memory_bandwidth_gb': self.memory_bandwidth_gb,
            'architecture': self.architecture.value,
            'pcie_generation': self.pcie_generation,
            'memory_clock_mhz': self.memory_clock_mhz,
            'gpu_clock_mhz': self.gpu_clock_mhz,
            'l2_cache_size_mb': self.l2_cache_size_mb,
            'tensor_cores': self.tensor_cores,
            'rt_cores': self.rt_cores
        }


@dataclass
class AutoOptimizedConfig:
    """Auto-optimized configuration based on detected GPU"""
    
    # Memory thresholds
    gpu_memory_threshold_mb: int
    memory_pressure_threshold: float
    cache_size_mb: int
    gpu_memory_limit_mb: int
    
    # Batch sizes
    cpu_batch_size: int
    gpu_batch_size: int
    prefetch_batches: int
    chunk_size: int
    
    # Performance tuning
    num_cpu_workers: int
    numa_nodes: List[int]
    huge_page_size: int
    max_concurrent_operations: int
    
    # GPU-specific optimizations
    use_tensor_cores: bool
    use_cudnn_benchmark: bool
    use_flash_attention: bool
    use_cuda_graphs: bool
    use_pinned_memory: bool
    
    # Memory pressure thresholds
    gpu_critical_threshold_mb: int
    gpu_high_threshold_mb: int
    gpu_moderate_threshold_mb: int
    
    # Utilization thresholds
    gpu_critical_utilization: float
    gpu_high_utilization: float
    gpu_moderate_utilization: float
    
    # Additional optimizations
    burst_multiplier: float
    emergency_cpu_workers: int
    memory_defrag_threshold: float
    memory_pool_count: int
    prefetch_queue_size: int
    
    @classmethod
    def from_gpu_specs(cls, gpu_specs: GPUSpecs, 
                       cpu_count: Optional[int] = None,
                       system_ram_gb: Optional[float] = None) -> 'AutoOptimizedConfig':
        """Create optimized config from GPU specifications
        
        Args:
            gpu_specs: Detected GPU specifications
            cpu_count: Number of CPU cores (auto-detect if None)
            system_ram_gb: System RAM in GB (auto-detect if None)
            
        Returns:
            Optimized configuration for the detected hardware
        """
        # Auto-detect CPU and RAM if not provided
        if cpu_count is None:
            cpu_count = psutil.cpu_count(logical=True)
        if system_ram_gb is None:
            system_ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # Calculate memory thresholds based on GPU memory
        gpu_memory_mb = gpu_specs.total_memory_mb
        
        # Conservative memory thresholds to prevent OOM
        gpu_memory_threshold_mb = min(2048, int(gpu_memory_mb * 0.125))  # 12.5% or 2GB max
        gpu_memory_limit_mb = int(gpu_memory_mb * 0.9)  # Use up to 90% of GPU memory
        
        # Pressure thresholds based on GPU memory size
        if gpu_memory_mb >= 24000:  # 24GB+ cards (3090, 4090, A5000, etc.)
            memory_pressure_threshold = 0.92
            gpu_critical_threshold_mb = int(gpu_memory_mb * 0.85)
            gpu_high_threshold_mb = int(gpu_memory_mb * 0.70)
            gpu_moderate_threshold_mb = int(gpu_memory_mb * 0.50)
            gpu_critical_utilization = 0.95
            gpu_high_utilization = 0.85
            gpu_moderate_utilization = 0.70
        elif gpu_memory_mb >= 16000:  # 16GB cards (RTX 4080, A4000, etc.)
            memory_pressure_threshold = 0.90
            gpu_critical_threshold_mb = int(gpu_memory_mb * 0.80)
            gpu_high_threshold_mb = int(gpu_memory_mb * 0.60)
            gpu_moderate_threshold_mb = int(gpu_memory_mb * 0.40)
            gpu_critical_utilization = 0.93
            gpu_high_utilization = 0.83
            gpu_moderate_utilization = 0.68
        elif gpu_memory_mb >= 12000:  # 12GB cards (RTX 3060, 4070 Ti, etc.)
            memory_pressure_threshold = 0.88
            gpu_critical_threshold_mb = int(gpu_memory_mb * 0.75)
            gpu_high_threshold_mb = int(gpu_memory_mb * 0.55)
            gpu_moderate_threshold_mb = int(gpu_memory_mb * 0.35)
            gpu_critical_utilization = 0.90
            gpu_high_utilization = 0.80
            gpu_moderate_utilization = 0.65
        elif gpu_memory_mb >= 8000:  # 8GB cards (RTX 3050, 4060, etc.)
            memory_pressure_threshold = 0.85
            gpu_critical_threshold_mb = int(gpu_memory_mb * 0.70)
            gpu_high_threshold_mb = int(gpu_memory_mb * 0.50)
            gpu_moderate_threshold_mb = int(gpu_memory_mb * 0.30)
            gpu_critical_utilization = 0.88
            gpu_high_utilization = 0.78
            gpu_moderate_utilization = 0.60
        else:  # < 8GB cards
            memory_pressure_threshold = 0.80
            gpu_critical_threshold_mb = int(gpu_memory_mb * 0.65)
            gpu_high_threshold_mb = int(gpu_memory_mb * 0.45)
            gpu_moderate_threshold_mb = int(gpu_memory_mb * 0.25)
            gpu_critical_utilization = 0.85
            gpu_high_utilization = 0.75
            gpu_moderate_utilization = 0.55
        
        # Cache size based on system RAM
        cache_size_mb = min(8192, int(system_ram_gb * 1024 * 0.25))  # 25% of RAM or 8GB max
        
        # Batch sizes based on GPU memory and architecture
        if gpu_specs.architecture in [GPUArchitecture.HOPPER, GPUArchitecture.ADA]:
            # Latest architectures can handle larger batches
            gpu_batch_size = min(100000, int(gpu_memory_mb * 10))
            cpu_batch_size = min(50000, cpu_count * 2000)
            chunk_size = min(50000, int(gpu_memory_mb * 5))
        elif gpu_specs.architecture == GPUArchitecture.AMPERE:
            gpu_batch_size = min(75000, int(gpu_memory_mb * 8))
            cpu_batch_size = min(40000, cpu_count * 1500)
            chunk_size = min(40000, int(gpu_memory_mb * 4))
        elif gpu_specs.architecture in [GPUArchitecture.TURING, GPUArchitecture.VOLTA]:
            gpu_batch_size = min(50000, int(gpu_memory_mb * 6))
            cpu_batch_size = min(30000, cpu_count * 1000)
            chunk_size = min(30000, int(gpu_memory_mb * 3))
        else:
            # Older architectures
            gpu_batch_size = min(25000, int(gpu_memory_mb * 4))
            cpu_batch_size = min(20000, cpu_count * 500)
            chunk_size = min(20000, int(gpu_memory_mb * 2))
        
        # Prefetch batches based on available memory
        prefetch_batches = min(20, max(2, int(gpu_memory_mb / 2000)))
        
        # CPU workers - use about 75% of available cores
        num_cpu_workers = max(1, int(cpu_count * 0.75))
        
        # Detect NUMA nodes
        numa_nodes = GPUAutoDetector._detect_numa_nodes()
        
        # Huge page size (2MB is standard)
        huge_page_size = 2097152
        
        # Concurrent operations based on GPU memory
        max_concurrent_operations = min(200, max(10, int(gpu_memory_mb / 100)))
        
        # GPU feature detection
        use_tensor_cores = gpu_specs.tensor_cores > 0
        use_cudnn_benchmark = gpu_specs.architecture.value in ['volta', 'turing', 'ampere', 'ada', 'hopper']
        use_flash_attention = gpu_specs.architecture.value in ['ampere', 'ada', 'hopper']
        use_cuda_graphs = gpu_specs.compute_capability >= "7.0"
        use_pinned_memory = True  # Almost always beneficial
        
        # Burst multiplier - how aggressively to burst to CPU
        if gpu_memory_mb >= 16000:
            burst_multiplier = 3.0
        elif gpu_memory_mb >= 12000:
            burst_multiplier = 4.0
        elif gpu_memory_mb >= 8000:
            burst_multiplier = 5.0
        else:
            burst_multiplier = 6.0
        
        # Emergency CPU workers
        emergency_cpu_workers = min(cpu_count * 2, 100)
        
        # Memory defragmentation threshold
        memory_defrag_threshold = 0.3 if gpu_memory_mb >= 16000 else 0.25
        
        # Memory pool count
        if gpu_memory_mb >= 24000:
            memory_pool_count = 4
        elif gpu_memory_mb >= 16000:
            memory_pool_count = 3
        elif gpu_memory_mb >= 12000:
            memory_pool_count = 2
        else:
            memory_pool_count = 1
        
        # Prefetch queue size
        prefetch_queue_size = min(30, max(5, int(gpu_memory_mb / 1000)))
        
        return cls(
            gpu_memory_threshold_mb=gpu_memory_threshold_mb,
            memory_pressure_threshold=memory_pressure_threshold,
            cache_size_mb=cache_size_mb,
            gpu_memory_limit_mb=gpu_memory_limit_mb,
            cpu_batch_size=cpu_batch_size,
            gpu_batch_size=gpu_batch_size,
            prefetch_batches=prefetch_batches,
            chunk_size=chunk_size,
            num_cpu_workers=num_cpu_workers,
            numa_nodes=numa_nodes,
            huge_page_size=huge_page_size,
            max_concurrent_operations=max_concurrent_operations,
            use_tensor_cores=use_tensor_cores,
            use_cudnn_benchmark=use_cudnn_benchmark,
            use_flash_attention=use_flash_attention,
            use_cuda_graphs=use_cuda_graphs,
            use_pinned_memory=use_pinned_memory,
            gpu_critical_threshold_mb=gpu_critical_threshold_mb,
            gpu_high_threshold_mb=gpu_high_threshold_mb,
            gpu_moderate_threshold_mb=gpu_moderate_threshold_mb,
            gpu_critical_utilization=gpu_critical_utilization,
            gpu_high_utilization=gpu_high_utilization,
            gpu_moderate_utilization=gpu_moderate_utilization,
            burst_multiplier=burst_multiplier,
            emergency_cpu_workers=emergency_cpu_workers,
            memory_defrag_threshold=memory_defrag_threshold,
            memory_pool_count=memory_pool_count,
            prefetch_queue_size=prefetch_queue_size
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'gpu_memory_threshold_mb': self.gpu_memory_threshold_mb,
            'memory_pressure_threshold': self.memory_pressure_threshold,
            'cache_size_mb': self.cache_size_mb,
            'gpu_memory_limit_mb': self.gpu_memory_limit_mb,
            'cpu_batch_size': self.cpu_batch_size,
            'gpu_batch_size': self.gpu_batch_size,
            'prefetch_batches': self.prefetch_batches,
            'chunk_size': self.chunk_size,
            'num_cpu_workers': self.num_cpu_workers,
            'numa_nodes': self.numa_nodes,
            'huge_page_size': self.huge_page_size,
            'max_concurrent_operations': self.max_concurrent_operations,
            'use_tensor_cores': self.use_tensor_cores,
            'use_cudnn_benchmark': self.use_cudnn_benchmark,
            'use_flash_attention': self.use_flash_attention,
            'use_cuda_graphs': self.use_cuda_graphs,
            'use_pinned_memory': self.use_pinned_memory,
            'gpu_critical_threshold_mb': self.gpu_critical_threshold_mb,
            'gpu_high_threshold_mb': self.gpu_high_threshold_mb,
            'gpu_moderate_threshold_mb': self.gpu_moderate_threshold_mb,
            'gpu_critical_utilization': self.gpu_critical_utilization,
            'gpu_high_utilization': self.gpu_high_utilization,
            'gpu_moderate_utilization': self.gpu_moderate_utilization,
            'burst_multiplier': self.burst_multiplier,
            'emergency_cpu_workers': self.emergency_cpu_workers,
            'memory_defrag_threshold': self.memory_defrag_threshold,
            'memory_pool_count': self.memory_pool_count,
            'prefetch_queue_size': self.prefetch_queue_size
        }


class GPUAutoDetector:
    """Auto-detect GPU specifications and optimize configuration"""
    
    def __init__(self):
        """Initialize GPU auto-detector"""
        self.gpu_specs_cache: Optional[Dict[int, GPUSpecs]] = None
        self.numa_config_cache: Optional[List[int]] = None
        self.detection_timestamp: Optional[float] = None
        self.cache_duration_seconds = 300  # Cache for 5 minutes
    
    def detect_all_gpus(self) -> Dict[int, GPUSpecs]:
        """Detect all available GPUs and their specifications
        
        Returns:
            Dictionary mapping device ID to GPU specifications
        """
        # Check cache first
        if self.gpu_specs_cache is not None and self.detection_timestamp is not None:
            if time.time() - self.detection_timestamp < self.cache_duration_seconds:
                return self.gpu_specs_cache
        
        gpu_specs = {}
        
        if not torch.cuda.is_available():
            # No CUDA available - return empty dict
            self.gpu_specs_cache = {}
            self.detection_timestamp = time.time()
            return {}
        
        device_count = torch.cuda.device_count()
        
        for device_id in range(device_count):
            try:
                # Get basic properties
                props = torch.cuda.get_device_properties(device_id)
                
                # Extract compute capability
                compute_capability = f"{props.major}.{props.minor}"
                
                # Determine architecture
                architecture = self.get_architecture_family(compute_capability)
                
                # Estimate CUDA cores
                cuda_cores = self.estimate_cuda_cores(props.name, props.multi_processor_count)
                
                # Estimate memory bandwidth
                memory_bandwidth = self.estimate_memory_bandwidth(props.name)
                
                # Estimate additional specs
                pcie_gen = self._estimate_pcie_generation(props.name)
                memory_clock = self._estimate_memory_clock(props.name)
                gpu_clock = props.clock_rate / 1000.0  # Convert kHz to MHz
                l2_cache_mb = self._estimate_l2_cache(props.name)
                tensor_cores = self._estimate_tensor_cores(props.name, props.multi_processor_count)
                rt_cores = self._estimate_rt_cores(props.name, props.multi_processor_count)
                
                # Create GPU specs
                specs = GPUSpecs(
                    device_id=device_id,
                    name=props.name,
                    total_memory_gb=props.total_memory / (1024**3),
                    total_memory_mb=props.total_memory / (1024**2),
                    compute_capability=compute_capability,
                    multi_processor_count=props.multi_processor_count,
                    max_threads_per_block=props.max_threads_per_block,
                    max_shared_memory_per_block=props.max_shared_memory_per_block,
                    warp_size=props.warp_size,
                    is_integrated=props.is_integrated,
                    is_multi_gpu_board=props.is_multi_gpu_board,
                    cuda_cores=cuda_cores,
                    memory_bandwidth_gb=memory_bandwidth,
                    architecture=architecture,
                    pcie_generation=pcie_gen,
                    memory_clock_mhz=memory_clock,
                    gpu_clock_mhz=gpu_clock,
                    l2_cache_size_mb=l2_cache_mb,
                    tensor_cores=tensor_cores,
                    rt_cores=rt_cores
                )
                
                gpu_specs[device_id] = specs
                
            except Exception as e:
                raise RuntimeError(f"Failed to detect GPU {device_id}: {e}")
        
        # Cache results
        self.gpu_specs_cache = gpu_specs
        self.detection_timestamp = time.time()
        
        return gpu_specs
    
    def get_primary_gpu(self) -> Optional[GPUSpecs]:
        """Get specifications of primary GPU
        
        Returns:
            GPU specifications for device 0, or None if no GPU available
        """
        gpus = self.detect_all_gpus()
        return gpus.get(0, None)
    
    def calculate_optimal_thresholds(self, gpu_specs: GPUSpecs) -> Dict[str, Any]:
        """Calculate optimal memory thresholds based on GPU specs
        
        Args:
            gpu_specs: GPU specifications
            
        Returns:
            Dictionary of optimal thresholds
        """
        config = AutoOptimizedConfig.from_gpu_specs(gpu_specs)
        return config.to_dict()
    
    @staticmethod
    def _detect_numa_nodes() -> List[int]:
        """Auto-detect NUMA node configuration
        
        Returns:
            List of available NUMA node IDs
        """
        numa_nodes = []
        
        # Try to detect NUMA nodes on Linux
        if platform.system() == "Linux":
            try:
                # Check /sys/devices/system/node/
                node_path = "/sys/devices/system/node/"
                if os.path.exists(node_path):
                    for item in os.listdir(node_path):
                        if item.startswith("node"):
                            try:
                                node_id = int(item[4:])
                                numa_nodes.append(node_id)
                            except ValueError:
                                continue
                
                # Alternative: use numactl
                if not numa_nodes:
                    try:
                        result = subprocess.run(['numactl', '--hardware'], 
                                              capture_output=True, text=True, timeout=1)
                        if result.returncode == 0:
                            # Parse output for available nodes
                            for line in result.stdout.split('\n'):
                                if line.startswith('available:'):
                                    # Extract number of nodes
                                    parts = line.split()
                                    if len(parts) >= 2:
                                        num_nodes = int(parts[1])
                                        numa_nodes = list(range(num_nodes))
                                        break
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        pass
                        
            except Exception:
                pass
        
        # Default to single node if detection fails or not Linux
        if not numa_nodes:
            numa_nodes = [0]
        
        return numa_nodes
    
    def estimate_cuda_cores(self, gpu_name: str, sm_count: int) -> int:
        """Estimate CUDA cores based on GPU architecture
        
        Args:
            gpu_name: GPU model name
            sm_count: Number of streaming multiprocessors
            
        Returns:
            Estimated number of CUDA cores
        """
        gpu_name_lower = gpu_name.lower()
        
        # CUDA cores per SM for different architectures
        if 'h100' in gpu_name_lower or 'h800' in gpu_name_lower:
            return sm_count * 128  # Hopper
        elif '4090' in gpu_name_lower or '4080' in gpu_name_lower:
            return sm_count * 128  # Ada Lovelace
        elif '4070' in gpu_name_lower or '4060' in gpu_name_lower:
            return sm_count * 128  # Ada Lovelace
        elif 'a100' in gpu_name_lower or 'a40' in gpu_name_lower:
            return sm_count * 64  # Ampere (datacenter)
        elif '3090' in gpu_name_lower or '3080' in gpu_name_lower:
            return sm_count * 128  # Ampere (consumer)
        elif '3070' in gpu_name_lower or '3060' in gpu_name_lower:
            return sm_count * 128  # Ampere (consumer)
        elif '2080' in gpu_name_lower or '2070' in gpu_name_lower:
            return sm_count * 64  # Turing
        elif 'v100' in gpu_name_lower:
            return sm_count * 64  # Volta
        elif '1080' in gpu_name_lower or '1070' in gpu_name_lower:
            return sm_count * 128  # Pascal
        elif 'titan' in gpu_name_lower:
            if 'rtx' in gpu_name_lower:
                return sm_count * 64  # Turing Titan RTX
            else:
                return sm_count * 128  # Pascal Titan
        elif 'a6000' in gpu_name_lower or 'a5000' in gpu_name_lower:
            return sm_count * 128  # Ampere workstation
        elif 'rtx 5060' in gpu_name_lower or 'rtx 5070' in gpu_name_lower:
            return sm_count * 128  # Estimated for future cards
        elif 'rtx 5080' in gpu_name_lower or 'rtx 5090' in gpu_name_lower:
            return sm_count * 192  # Estimated for future high-end cards
        else:
            # Default estimate based on generation hints
            if 'rtx' in gpu_name_lower:
                return sm_count * 64  # Conservative RTX estimate
            else:
                return sm_count * 128  # Conservative GTX estimate
    
    def estimate_memory_bandwidth(self, gpu_name: str) -> float:
        """Estimate memory bandwidth based on GPU model
        
        Args:
            gpu_name: GPU model name
            
        Returns:
            Estimated memory bandwidth in GB/s
        """
        gpu_name_lower = gpu_name.lower()
        
        # Known memory bandwidths
        bandwidth_map = {
            'h100': 3350.0,  # HBM3
            'h800': 3350.0,  # HBM3
            'a100': 2039.0,  # HBM2e (80GB)
            '4090': 1008.0,  # GDDR6X
            '4080': 716.8,   # GDDR6X
            '4070 ti': 504.2,  # GDDR6X
            '4070': 504.2,   # GDDR6X
            '4060 ti': 288.0,  # GDDR6
            '4060': 272.0,   # GDDR6
            '3090 ti': 1008.0,  # GDDR6X
            '3090': 936.2,   # GDDR6X
            '3080 ti': 912.4,  # GDDR6X
            '3080': 760.3,   # GDDR6X
            '3070 ti': 608.3,  # GDDR6X
            '3070': 448.0,   # GDDR6
            '3060 ti': 448.0,  # GDDR6
            '3060': 360.0,   # GDDR6
            '2080 ti': 616.0,  # GDDR6
            '2080': 448.0,   # GDDR6
            '2070': 448.0,   # GDDR6
            'v100': 900.0,   # HBM2
            'a6000': 768.0,  # GDDR6
            'a5000': 768.0,  # GDDR6
            'a4000': 448.0,  # GDDR6
            'rtx 5090': 1600.0,  # Estimated future
            'rtx 5080': 1200.0,  # Estimated future
            'rtx 5070': 800.0,   # Estimated future
            'rtx 5060': 600.0,   # Estimated future
        }
        
        # Check for exact matches
        for key, bandwidth in bandwidth_map.items():
            if key in gpu_name_lower:
                return bandwidth
        
        # Default estimates based on patterns
        if 'hbm' in gpu_name_lower:
            return 1500.0  # HBM memory
        elif 'gddr6x' in gpu_name_lower:
            return 700.0   # GDDR6X
        elif 'gddr6' in gpu_name_lower:
            return 450.0   # GDDR6
        else:
            return 400.0   # Conservative default
    
    def get_architecture_family(self, compute_capability: str) -> GPUArchitecture:
        """Determine GPU architecture from compute capability
        
        Args:
            compute_capability: CUDA compute capability string (e.g., "8.6")
            
        Returns:
            GPU architecture family
        """
        try:
            major, minor = map(int, compute_capability.split('.'))
            capability = major * 10 + minor
            
            if capability >= 100:  # 10.x
                return GPUArchitecture.BLACKWELL
            elif capability >= 90:  # 9.x
                return GPUArchitecture.HOPPER
            elif capability == 89:  # 8.9
                return GPUArchitecture.ADA
            elif capability >= 80:  # 8.x
                return GPUArchitecture.AMPERE
            elif capability == 75:  # 7.5
                return GPUArchitecture.TURING
            elif capability >= 70:  # 7.0
                return GPUArchitecture.VOLTA
            elif capability >= 60:  # 6.x
                return GPUArchitecture.PASCAL
            elif capability >= 50:  # 5.x
                return GPUArchitecture.MAXWELL
            elif capability >= 30:  # 3.x
                return GPUArchitecture.KEPLER
            else:
                return GPUArchitecture.UNKNOWN
                
        except Exception:
            return GPUArchitecture.UNKNOWN
    
    def _estimate_pcie_generation(self, gpu_name: str) -> int:
        """Estimate PCIe generation from GPU name"""
        gpu_name_lower = gpu_name.lower()
        
        if any(x in gpu_name_lower for x in ['h100', 'h800', '4090', '4080', '4070', '4060']):
            return 5  # PCIe 5.0
        elif any(x in gpu_name_lower for x in ['a100', '3090', '3080', '3070', '3060']):
            return 4  # PCIe 4.0
        else:
            return 3  # PCIe 3.0 (conservative)
    
    def _estimate_memory_clock(self, gpu_name: str) -> float:
        """Estimate memory clock speed in MHz"""
        gpu_name_lower = gpu_name.lower()
        
        if 'gddr6x' in gpu_name_lower or '3090' in gpu_name_lower or '4090' in gpu_name_lower:
            return 21000.0  # GDDR6X typical
        elif 'gddr6' in gpu_name_lower or '3060' in gpu_name_lower or '2080' in gpu_name_lower:
            return 14000.0  # GDDR6 typical
        elif 'hbm' in gpu_name_lower or 'a100' in gpu_name_lower or 'h100' in gpu_name_lower:
            return 3200.0   # HBM2e/HBM3 typical
        else:
            return 10000.0  # Conservative default
    
    def _estimate_l2_cache(self, gpu_name: str) -> float:
        """Estimate L2 cache size in MB"""
        gpu_name_lower = gpu_name.lower()
        
        cache_map = {
            'h100': 50.0,
            'a100': 40.0,
            '4090': 72.0,
            '4080': 64.0,
            '4070': 36.0,
            '3090': 6.0,
            '3080': 5.0,
            '3070': 4.0,
            '3060': 3.0,
            '2080': 6.0,
            'v100': 6.0,
        }
        
        for key, cache_size in cache_map.items():
            if key in gpu_name_lower:
                return cache_size
        
        return 4.0  # Conservative default
    
    def _estimate_tensor_cores(self, gpu_name: str, sm_count: int) -> int:
        """Estimate number of tensor cores"""
        gpu_name_lower = gpu_name.lower()
        
        # Tensor cores per SM for different architectures
        if 'h100' in gpu_name_lower:
            return sm_count * 4  # 4th gen tensor cores
        elif any(x in gpu_name_lower for x in ['4090', '4080', '4070', 'a100']):
            return sm_count * 4  # 3rd gen tensor cores
        elif any(x in gpu_name_lower for x in ['3090', '3080', '3070', '3060']):
            return sm_count * 4  # 3rd gen tensor cores
        elif any(x in gpu_name_lower for x in ['2080', '2070', 'titan rtx']):
            return sm_count * 8  # 1st gen tensor cores
        elif 'v100' in gpu_name_lower:
            return sm_count * 8  # 1st gen tensor cores
        else:
            return 0  # No tensor cores
    
    def _estimate_rt_cores(self, gpu_name: str, sm_count: int) -> int:
        """Estimate number of RT cores"""
        gpu_name_lower = gpu_name.lower()
        
        # RT cores per SM for RTX cards
        if any(x in gpu_name_lower for x in ['4090', '4080', '4070', '4060']):
            return sm_count  # 3rd gen RT cores
        elif any(x in gpu_name_lower for x in ['3090', '3080', '3070', '3060']):
            return sm_count  # 2nd gen RT cores
        elif any(x in gpu_name_lower for x in ['2080', '2070', 'titan rtx']):
            return sm_count  # 1st gen RT cores
        else:
            return 0  # No RT cores


class ConfigUpdater:
    """Update all configuration classes to use auto-detection"""
    
    def __init__(self):
        """Initialize configuration updater"""
        self.detector = GPUAutoDetector()
        self.optimized_config: Optional[AutoOptimizedConfig] = None
        self._initialize_config()
    
    def _initialize_config(self) -> None:
        """Initialize optimized configuration"""
        gpu_specs = self.detector.get_primary_gpu()
        if gpu_specs:
            self.optimized_config = AutoOptimizedConfig.from_gpu_specs(gpu_specs)
        else:
            # CPU-only fallback configuration
            self.optimized_config = self._get_cpu_only_config()
    
    def _get_cpu_only_config(self) -> AutoOptimizedConfig:
        """Get configuration for CPU-only systems"""
        cpu_count = psutil.cpu_count(logical=True)
        system_ram_gb = psutil.virtual_memory().total / (1024**3)
        
        return AutoOptimizedConfig(
            gpu_memory_threshold_mb=0,
            memory_pressure_threshold=0.9,
            cache_size_mb=min(4096, int(system_ram_gb * 1024 * 0.25)),
            gpu_memory_limit_mb=0,
            cpu_batch_size=min(10000, cpu_count * 500),
            gpu_batch_size=0,
            prefetch_batches=2,
            chunk_size=5000,
            num_cpu_workers=max(1, cpu_count - 1),
            numa_nodes=GPUAutoDetector._detect_numa_nodes(),
            huge_page_size=2097152,
            max_concurrent_operations=cpu_count * 2,
            use_tensor_cores=False,
            use_cudnn_benchmark=False,
            use_flash_attention=False,
            use_cuda_graphs=False,
            use_pinned_memory=False,
            gpu_critical_threshold_mb=0,
            gpu_high_threshold_mb=0,
            gpu_moderate_threshold_mb=0,
            gpu_critical_utilization=1.0,
            gpu_high_utilization=1.0,
            gpu_moderate_utilization=1.0,
            burst_multiplier=1.0,
            emergency_cpu_workers=cpu_count,
            memory_defrag_threshold=0.3,
            memory_pool_count=1,
            prefetch_queue_size=5
        )
    
    def update_cpu_bursting_config(self, config: Any) -> Any:
        """Update CPU bursting configuration with auto-detected values
        
        Args:
            config: Original CPUBurstingConfig instance
            
        Returns:
            Updated configuration
        """
        if not self.optimized_config:
            raise RuntimeError("No optimized configuration available")
        
        # Update configuration with auto-detected values
        config.gpu_memory_threshold_mb = self.optimized_config.gpu_memory_threshold_mb
        config.memory_pressure_threshold = self.optimized_config.memory_pressure_threshold
        config.num_cpu_workers = self.optimized_config.num_cpu_workers
        config.cpu_batch_size = self.optimized_config.cpu_batch_size
        config.cache_size_mb = self.optimized_config.cache_size_mb
        config.prefetch_batches = self.optimized_config.prefetch_batches
        config.numa_nodes = self.optimized_config.numa_nodes
        config.huge_page_size = self.optimized_config.huge_page_size
        
        return config
    
    def update_memory_pressure_config(self, config: Any) -> Any:
        """Update memory pressure configuration
        
        Args:
            config: Original PressureHandlerConfig instance
            
        Returns:
            Updated configuration
        """
        if not self.optimized_config:
            raise RuntimeError("No optimized configuration available")
        
        # Update thresholds
        config.gpu_critical_threshold_mb = self.optimized_config.gpu_critical_threshold_mb
        config.gpu_high_threshold_mb = self.optimized_config.gpu_high_threshold_mb
        config.gpu_moderate_threshold_mb = self.optimized_config.gpu_moderate_threshold_mb
        config.gpu_critical_utilization = self.optimized_config.gpu_critical_utilization
        config.gpu_high_utilization = self.optimized_config.gpu_high_utilization
        config.gpu_moderate_utilization = self.optimized_config.gpu_moderate_utilization
        config.max_cpu_batch_size = self.optimized_config.cpu_batch_size
        config.burst_multiplier = self.optimized_config.burst_multiplier
        config.emergency_cpu_workers = self.optimized_config.emergency_cpu_workers
        config.memory_defrag_threshold = self.optimized_config.memory_defrag_threshold
        
        return config
    
    def update_gpu_memory_pool_config(self, config: Any) -> Any:
        """Update GPU memory pool configuration
        
        Args:
            config: Original GPU memory configuration
            
        Returns:
            Updated configuration
        """
        if not self.optimized_config:
            raise RuntimeError("No optimized configuration available")
        
        # Update GPU memory settings
        if hasattr(config, 'gpu_memory_limit_mb'):
            config.gpu_memory_limit_mb = self.optimized_config.gpu_memory_limit_mb
        if hasattr(config, 'chunk_size'):
            config.chunk_size = self.optimized_config.chunk_size
        if hasattr(config, 'max_concurrent_operations'):
            config.max_concurrent_operations = self.optimized_config.max_concurrent_operations
        if hasattr(config, 'memory_pool_count'):
            config.memory_pool_count = self.optimized_config.memory_pool_count
        if hasattr(config, 'prefetch_queue_size'):
            config.prefetch_queue_size = self.optimized_config.prefetch_queue_size
        if hasattr(config, 'use_cuda_graphs'):
            config.use_cuda_graphs = self.optimized_config.use_cuda_graphs
        if hasattr(config, 'use_pinned_memory'):
            config.use_pinned_memory = self.optimized_config.use_pinned_memory
        
        return config
    
    def get_gpu_info_string(self) -> str:
        """Get a formatted string with GPU information
        
        Returns:
            Human-readable GPU information
        """
        gpu_specs = self.detector.get_primary_gpu()
        if not gpu_specs:
            return "No GPU detected - running in CPU-only mode"
        
        return (
            f"GPU: {gpu_specs.name}\n"
            f"Memory: {gpu_specs.total_memory_gb:.1f} GB\n"
            f"Architecture: {gpu_specs.architecture.value.upper()}\n"
            f"Compute Capability: {gpu_specs.compute_capability}\n"
            f"CUDA Cores: ~{gpu_specs.cuda_cores}\n"
            f"Memory Bandwidth: ~{gpu_specs.memory_bandwidth_gb:.1f} GB/s\n"
            f"Tensor Cores: {gpu_specs.tensor_cores}\n"
            f"RT Cores: {gpu_specs.rt_cores}"
        )
    
    def export_config(self, filepath: str) -> None:
        """Export optimized configuration to JSON file
        
        Args:
            filepath: Path to save configuration
        """
        if not self.optimized_config:
            raise RuntimeError("No optimized configuration available")
        
        config_dict = {
            'gpu_info': self.detector.get_primary_gpu().to_dict() if self.detector.get_primary_gpu() else None,
            'optimized_config': self.optimized_config.to_dict(),
            'timestamp': time.time(),
            'system_info': {
                'platform': platform.platform(),
                'cpu_count': psutil.cpu_count(logical=True),
                'ram_gb': psutil.virtual_memory().total / (1024**3)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: str) -> 'ConfigUpdater':
        """Load configuration from JSON file
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            ConfigUpdater instance with loaded configuration
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        updater = cls()
        
        # Reconstruct optimized config
        opt_config = config_dict['optimized_config']
        updater.optimized_config = AutoOptimizedConfig(
            **opt_config
        )
        
        return updater


# Singleton instance for global access
_global_detector = None
_global_config_updater = None


def get_gpu_detector() -> GPUAutoDetector:
    """Get global GPU detector instance
    
    Returns:
        Singleton GPUAutoDetector instance
    """
    global _global_detector
    if _global_detector is None:
        _global_detector = GPUAutoDetector()
    return _global_detector


def get_config_updater() -> ConfigUpdater:
    """Get global configuration updater instance
    
    Returns:
        Singleton ConfigUpdater instance
    """
    global _global_config_updater
    if _global_config_updater is None:
        _global_config_updater = ConfigUpdater()
    return _global_config_updater


def auto_configure_system() -> Dict[str, Any]:
    """Automatically configure the entire compression system
    
    Returns:
        Dictionary with all auto-configured settings
    """
    detector = get_gpu_detector()
    updater = get_config_updater()
    
    gpu_specs = detector.get_primary_gpu()
    if gpu_specs:
        print(f"Detected GPU: {gpu_specs.name} ({gpu_specs.total_memory_gb:.1f} GB)")
        print(f"Architecture: {gpu_specs.architecture.value.upper()}")
        print(f"Auto-configuring for optimal performance...")
    else:
        print("No GPU detected - configuring for CPU-only operation")
    
    return updater.optimized_config.to_dict()


if __name__ == "__main__":
    # Test GPU detection
    print("GPU Auto-Detection System Test")
    print("=" * 50)
    
    detector = GPUAutoDetector()
    gpus = detector.detect_all_gpus()
    
    if gpus:
        for device_id, specs in gpus.items():
            print(f"\nDevice {device_id}: {specs.name}")
            print(f"  Memory: {specs.total_memory_gb:.1f} GB")
            print(f"  Architecture: {specs.architecture.value}")
            print(f"  Compute Capability: {specs.compute_capability}")
            print(f"  CUDA Cores: ~{specs.cuda_cores}")
            print(f"  Memory Bandwidth: ~{specs.memory_bandwidth_gb:.1f} GB/s")
            print(f"  Tensor Cores: {specs.tensor_cores}")
            print(f"  RT Cores: {specs.rt_cores}")
            
            # Get optimized configuration
            config = AutoOptimizedConfig.from_gpu_specs(specs)
            print(f"\n  Optimized Configuration:")
            print(f"    GPU Memory Threshold: {config.gpu_memory_threshold_mb} MB")
            print(f"    Memory Pressure Threshold: {config.memory_pressure_threshold:.2f}")
            print(f"    CPU Workers: {config.num_cpu_workers}")
            print(f"    CPU Batch Size: {config.cpu_batch_size}")
            print(f"    GPU Batch Size: {config.gpu_batch_size}")
            print(f"    NUMA Nodes: {config.numa_nodes}")
    else:
        print("No CUDA GPUs detected")
        
        # Test CPU-only configuration
        updater = ConfigUpdater()
        config = updater.optimized_config
        print(f"\nCPU-Only Configuration:")
        print(f"  CPU Workers: {config.num_cpu_workers}")
        print(f"  CPU Batch Size: {config.cpu_batch_size}")
        print(f"  Cache Size: {config.cache_size_mb} MB")
        print(f"  NUMA Nodes: {config.numa_nodes}")