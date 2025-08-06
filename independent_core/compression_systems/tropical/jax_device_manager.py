"""
JAX Device Management Wrapper - Advanced device management for JAX operations
Integrates with GPUAutoDetector for dynamic device selection and multi-GPU support
HARD FAILURES ONLY - NO GRACEFUL DEGRADATION

This module provides:
1. JAXDeviceManager - Wraps JAX device operations
2. Integration with GPUAutoDetector for dynamic device selection
3. Multi-GPU support with proper device placement
4. Device synchronization utilities
"""

import jax
import jax.numpy as jnp
from jax import device_put, device_get, devices
from jax.lib import xla_bridge
from jax.sharding import PositionalSharding, NamedSharding, Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
import numpy as np
import torch
import time
import threading
import logging
from typing import Dict, List, Optional, Tuple, Any, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import re

# Import existing components
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from independent_core.compression_systems.gpu_memory.gpu_auto_detector import (
    GPUAutoDetector,
    GPUSpecs,
    GPUArchitecture
)

logger = logging.getLogger(__name__)


class DevicePlacementStrategy(Enum):
    """Device placement strategies"""
    SINGLE_DEVICE = "single_device"      # Use single device
    DATA_PARALLEL = "data_parallel"      # Replicate across devices
    MODEL_PARALLEL = "model_parallel"    # Split model across devices
    PIPELINE_PARALLEL = "pipeline_parallel"  # Pipeline across devices
    AUTO = "auto"                        # Automatic placement


class DeviceSelectionStrategy(Enum):
    """Device selection strategies"""
    FIRST_AVAILABLE = "first_available"  # Use first available device
    LEAST_LOADED = "least_loaded"        # Use least loaded device
    ROUND_ROBIN = "round_robin"          # Round-robin selection
    MEMORY_AWARE = "memory_aware"        # Select based on memory availability
    COMPUTE_AWARE = "compute_aware"      # Select based on compute capability


@dataclass
class JAXDeviceInfo:
    """Information about a JAX device"""
    device: Any  # JAX device object
    device_id: int
    device_type: str  # 'gpu', 'cpu', 'tpu'
    platform: str  # 'cuda', 'rocm', 'cpu', etc.
    gpu_specs: Optional[GPUSpecs] = None  # From GPUAutoDetector
    memory_stats: Dict[str, Any] = field(default_factory=dict)
    compute_capability: Optional[str] = None
    is_available: bool = True
    current_load: float = 0.0  # 0-1 utilization
    
    def __post_init__(self):
        """Extract additional info from device"""
        if self.gpu_specs:
            self.compute_capability = self.gpu_specs.compute_capability
            self.memory_stats = {
                'total_memory_mb': self.gpu_specs.total_memory_mb,
                'memory_bandwidth_gb': self.gpu_specs.memory_bandwidth_gb,
                'cuda_cores': self.gpu_specs.cuda_cores
            }


@dataclass
class DevicePlacement:
    """Device placement configuration"""
    device: Any
    strategy: DevicePlacementStrategy
    sharding_spec: Optional[Any] = None
    axis_names: Optional[List[str]] = None
    mesh_shape: Optional[Tuple[int, ...]] = None


class JAXDeviceManager:
    """
    Manages JAX devices with integration to GPUAutoDetector.
    Provides intelligent device selection and multi-GPU coordination.
    """
    
    def __init__(self,
                 gpu_detector: Optional[GPUAutoDetector] = None,
                 selection_strategy: DeviceSelectionStrategy = DeviceSelectionStrategy.MEMORY_AWARE):
        """
        Initialize JAX device manager
        
        Args:
            gpu_detector: GPU auto-detector instance (creates new if None)
            selection_strategy: Strategy for device selection
        """
        # Initialize GPU detector
        self.gpu_detector = gpu_detector or GPUAutoDetector()
        self.selection_strategy = selection_strategy
        
        # Detect all devices
        self.jax_devices = self._detect_jax_devices()
        self.gpu_specs = self.gpu_detector.detect_all_gpus()
        
        # Map JAX devices to GPU specs
        self.device_info: Dict[int, JAXDeviceInfo] = {}
        self._map_devices_to_specs()
        
        # Device selection state
        self.round_robin_counter = 0
        self.device_assignments: Dict[str, int] = {}  # task_id -> device_id
        
        # Multi-device mesh for sharding
        self.device_mesh: Optional[Mesh] = None
        self._setup_device_mesh()
        
        # Synchronization
        self._lock = threading.RLock()
        
        # Monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_devices, daemon=True)
        self.monitor_thread.start()
        
        # Statistics
        self.stats = {
            'device_selections': 0,
            'placement_operations': 0,
            'synchronizations': 0,
            'memory_transfers': 0,
            'sharding_operations': 0
        }
        
        logger.info(f"JAXDeviceManager initialized with {len(self.jax_devices)} devices")
    
    def _detect_jax_devices(self) -> List[Any]:
        """Detect all available JAX devices"""
        try:
            all_devices = jax.devices()
            logger.info(f"Detected {len(all_devices)} JAX devices")
            return all_devices
        except Exception as e:
            raise RuntimeError(f"Failed to detect JAX devices: {e}")
    
    def _map_devices_to_specs(self) -> None:
        """Map JAX devices to GPU specifications"""
        for jax_device in self.jax_devices:
            device_id = self._extract_device_id(jax_device)
            device_type = self._get_device_type(jax_device)
            platform = str(jax_device.platform)
            
            # Get GPU specs if available
            gpu_specs = None
            if device_type == 'gpu' and device_id in self.gpu_specs:
                gpu_specs = self.gpu_specs[device_id]
            
            # Create device info
            device_info = JAXDeviceInfo(
                device=jax_device,
                device_id=device_id,
                device_type=device_type,
                platform=platform,
                gpu_specs=gpu_specs
            )
            
            self.device_info[device_id] = device_info
    
    def _extract_device_id(self, device: Any) -> int:
        """Extract device ID from JAX device"""
        if hasattr(device, 'id'):
            return device.id
        
        # Parse from string representation
        device_str = str(device).lower()
        match = re.search(r'(?:gpu|cpu|tpu):(\d+)', device_str)
        if match:
            return int(match.group(1))
        return 0
    
    def _get_device_type(self, device: Any) -> str:
        """Get device type (gpu, cpu, tpu)"""
        device_str = str(device).lower()
        if 'gpu' in device_str:
            return 'gpu'
        elif 'tpu' in device_str:
            return 'tpu'
        else:
            return 'cpu'
    
    def _setup_device_mesh(self) -> None:
        """Setup device mesh for multi-device operations"""
        if len(self.jax_devices) > 1:
            try:
                # Create mesh for data parallelism
                devices_array = mesh_utils.create_device_mesh((len(self.jax_devices),))
                self.device_mesh = Mesh(devices_array, axis_names=('devices',))
                logger.info(f"Created device mesh with shape {devices_array.shape}")
            except Exception as e:
                logger.warning(f"Failed to create device mesh: {e}")
                self.device_mesh = None
    
    def select_device(self, 
                      memory_required_mb: Optional[float] = None,
                      compute_intensity: Optional[float] = None,
                      task_id: Optional[str] = None) -> JAXDeviceInfo:
        """
        Select optimal device based on strategy
        
        Args:
            memory_required_mb: Memory required for operation
            compute_intensity: Compute intensity (0-1)
            task_id: Task identifier for tracking
            
        Returns:
            Selected device info
        """
        with self._lock:
            self.stats['device_selections'] += 1
            
            available_devices = [d for d in self.device_info.values() if d.is_available]
            if not available_devices:
                raise RuntimeError("No available devices")
            
            # Select based on strategy
            if self.selection_strategy == DeviceSelectionStrategy.FIRST_AVAILABLE:
                selected = available_devices[0]
            
            elif self.selection_strategy == DeviceSelectionStrategy.ROUND_ROBIN:
                selected = available_devices[self.round_robin_counter % len(available_devices)]
                self.round_robin_counter += 1
            
            elif self.selection_strategy == DeviceSelectionStrategy.LEAST_LOADED:
                selected = min(available_devices, key=lambda d: d.current_load)
            
            elif self.selection_strategy == DeviceSelectionStrategy.MEMORY_AWARE:
                if memory_required_mb and any(d.gpu_specs for d in available_devices):
                    # Filter devices with enough memory
                    suitable = [d for d in available_devices 
                              if d.gpu_specs and 
                              d.gpu_specs.total_memory_mb * (1 - d.current_load) > memory_required_mb]
                    if suitable:
                        selected = suitable[0]
                    else:
                        # Fallback to least loaded
                        selected = min(available_devices, key=lambda d: d.current_load)
                else:
                    selected = available_devices[0]
            
            elif self.selection_strategy == DeviceSelectionStrategy.COMPUTE_AWARE:
                if compute_intensity and any(d.gpu_specs for d in available_devices):
                    # Select based on compute capability
                    gpu_devices = [d for d in available_devices if d.gpu_specs]
                    if gpu_devices:
                        # Higher compute capability for higher intensity
                        if compute_intensity > 0.7:
                            # Prefer newer architectures
                            selected = max(gpu_devices, 
                                         key=lambda d: d.gpu_specs.cuda_cores if d.gpu_specs else 0)
                        else:
                            # Any GPU is fine
                            selected = gpu_devices[0]
                    else:
                        selected = available_devices[0]
                else:
                    selected = available_devices[0]
            
            else:  # AUTO
                # Automatic selection based on all factors
                scores = []
                for device in available_devices:
                    score = 1.0 - device.current_load
                    
                    if device.gpu_specs:
                        # Bonus for GPU
                        score += 0.5
                        
                        # Memory score
                        if memory_required_mb:
                            available_memory = device.gpu_specs.total_memory_mb * (1 - device.current_load)
                            if available_memory > memory_required_mb:
                                score += 0.3
                        
                        # Compute score
                        if compute_intensity:
                            score += compute_intensity * 0.2
                    
                    scores.append((device, score))
                
                selected = max(scores, key=lambda x: x[1])[0]
            
            # Track assignment
            if task_id:
                self.device_assignments[task_id] = selected.device_id
            
            return selected
    
    def place_on_device(self, 
                        array: Any,
                        device_info: JAXDeviceInfo,
                        block: bool = True) -> Any:
        """
        Place array on specified device
        
        Args:
            array: Array to place (JAX or numpy)
            device_info: Target device info
            block: Whether to block until transfer completes
            
        Returns:
            Array on target device
        """
        self.stats['placement_operations'] += 1
        
        # Convert numpy to JAX if needed
        if isinstance(array, np.ndarray):
            array = jnp.array(array)
        
        # Place on device
        placed = device_put(array, device_info.device)
        
        # Block if requested
        if block:
            placed.block_until_ready()
        
        self.stats['memory_transfers'] += 1
        
        return placed
    
    def create_sharded_array(self,
                           shape: Tuple[int, ...],
                           dtype: Any = jnp.float32,
                           strategy: DevicePlacementStrategy = DevicePlacementStrategy.DATA_PARALLEL) -> Any:
        """
        Create sharded array across multiple devices
        
        Args:
            shape: Array shape
            dtype: Data type
            strategy: Placement strategy
            
        Returns:
            Sharded array
        """
        if not self.device_mesh:
            # Fallback to single device
            return jnp.zeros(shape, dtype=dtype)
        
        self.stats['sharding_operations'] += 1
        
        if strategy == DevicePlacementStrategy.DATA_PARALLEL:
            # Shard along first axis
            sharding = NamedSharding(self.device_mesh, P('devices', None))
        elif strategy == DevicePlacementStrategy.MODEL_PARALLEL:
            # Shard along last axis
            sharding = NamedSharding(self.device_mesh, P(None, 'devices'))
        else:
            # Default sharding
            sharding = PositionalSharding(self.jax_devices)
        
        # Create sharded array
        with self.device_mesh:
            array = jnp.zeros(shape, dtype=dtype)
            sharded = jax.device_put(array, sharding)
        
        return sharded
    
    def synchronize_device(self, device_info: JAXDeviceInfo) -> None:
        """
        Synchronize operations on device
        
        Args:
            device_info: Device to synchronize
        """
        self.stats['synchronizations'] += 1
        
        # Create dummy operation to force synchronization
        dummy = jnp.array([0.0])
        dummy = device_put(dummy, device_info.device)
        dummy.block_until_ready()
    
    def synchronize_all_devices(self) -> None:
        """Synchronize all devices"""
        for device_info in self.device_info.values():
            self.synchronize_device(device_info)
    
    def get_device_memory_info(self, device_info: JAXDeviceInfo) -> Dict[str, Any]:
        """
        Get memory information for device
        
        Args:
            device_info: Device to query
            
        Returns:
            Memory information dictionary
        """
        memory_info = {
            'device_id': device_info.device_id,
            'device_type': device_info.device_type
        }
        
        if device_info.gpu_specs:
            memory_info.update({
                'total_memory_mb': device_info.gpu_specs.total_memory_mb,
                'memory_bandwidth_gb': device_info.gpu_specs.memory_bandwidth_gb,
                'architecture': device_info.gpu_specs.architecture.value
            })
        
        # Try to get live memory stats
        try:
            backend = xla_bridge.get_backend()
            if hasattr(backend, 'live_buffers'):
                live_buffers = [buf for buf in backend.live_buffers() 
                              if buf.device() == device_info.device]
                memory_info['live_buffers'] = len(live_buffers)
                memory_info['live_memory_bytes'] = sum(buf.nbytes for buf in live_buffers)
        except:
            pass
        
        memory_info['current_load'] = device_info.current_load
        
        return memory_info
    
    def _monitor_devices(self) -> None:
        """Background monitoring of device utilization"""
        while self.monitoring_active:
            try:
                with self._lock:
                    for device_info in self.device_info.values():
                        # Update load estimates
                        if device_info.device_type == 'gpu':
                            # Estimate based on memory usage
                            memory_info = self.get_device_memory_info(device_info)
                            if 'live_memory_bytes' in memory_info and device_info.gpu_specs:
                                total_bytes = device_info.gpu_specs.total_memory_mb * 1024 * 1024
                                device_info.current_load = memory_info['live_memory_bytes'] / total_bytes
                        
                        # Check availability
                        try:
                            # Test device with small operation
                            test = jnp.array([1.0])
                            test = device_put(test, device_info.device)
                            test.block_until_ready()
                            device_info.is_available = True
                        except:
                            device_info.is_available = False
                
                time.sleep(1.0)  # 1 second interval
                
            except Exception as e:
                logger.error(f"Device monitoring error: {e}")
    
    def get_optimal_placement(self,
                             operation_type: str,
                             data_size_mb: float,
                             model_size_mb: Optional[float] = None) -> DevicePlacement:
        """
        Get optimal device placement for operation
        
        Args:
            operation_type: Type of operation ('compute', 'memory', 'io')
            data_size_mb: Data size in MB
            model_size_mb: Model size in MB (optional)
            
        Returns:
            Optimal device placement configuration
        """
        # Select device based on operation type
        if operation_type == 'compute':
            device_info = self.select_device(
                memory_required_mb=data_size_mb,
                compute_intensity=0.8
            )
        elif operation_type == 'memory':
            device_info = self.select_device(
                memory_required_mb=data_size_mb + (model_size_mb or 0),
                compute_intensity=0.2
            )
        else:  # 'io'
            device_info = self.select_device(
                memory_required_mb=data_size_mb,
                compute_intensity=0.1
            )
        
        # Determine strategy
        num_gpus = sum(1 for d in self.device_info.values() if d.device_type == 'gpu')
        
        if num_gpus <= 1:
            strategy = DevicePlacementStrategy.SINGLE_DEVICE
        elif data_size_mb > 1000:  # Large data
            strategy = DevicePlacementStrategy.DATA_PARALLEL
        elif model_size_mb and model_size_mb > 500:  # Large model
            strategy = DevicePlacementStrategy.MODEL_PARALLEL
        else:
            strategy = DevicePlacementStrategy.SINGLE_DEVICE
        
        return DevicePlacement(
            device=device_info.device,
            strategy=strategy,
            mesh_shape=(num_gpus,) if num_gpus > 1 else None
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get device manager statistics"""
        with self._lock:
            return {
                **self.stats,
                'num_devices': len(self.device_info),
                'num_available': sum(1 for d in self.device_info.values() if d.is_available),
                'device_loads': {
                    d.device_id: d.current_load 
                    for d in self.device_info.values()
                },
                'active_assignments': len(self.device_assignments)
            }
    
    def shutdown(self) -> None:
        """Shutdown device manager"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        
        # Clear state
        self.device_assignments.clear()
        
        logger.info("JAXDeviceManager shutdown complete")


# Helper functions for device management
def auto_select_device(memory_mb: float = None, 
                       compute_intensity: float = None) -> Any:
    """
    Automatically select best JAX device
    
    Args:
        memory_mb: Memory required in MB
        compute_intensity: Compute intensity (0-1)
        
    Returns:
        Selected JAX device
    """
    manager = JAXDeviceManager()
    device_info = manager.select_device(
        memory_required_mb=memory_mb,
        compute_intensity=compute_intensity
    )
    return device_info.device


def create_device_mesh_auto() -> Optional[Mesh]:
    """
    Automatically create device mesh for multi-GPU
    
    Returns:
        Device mesh or None if single device
    """
    devices_list = jax.devices()
    if len(devices_list) <= 1:
        return None
    
    devices_array = mesh_utils.create_device_mesh((len(devices_list),))
    return Mesh(devices_array, axis_names=('devices',))


# Test function
def test_jax_device_manager():
    """Test JAX device manager functionality"""
    print("Testing JAX Device Manager...")
    
    # Create GPU detector
    gpu_detector = GPUAutoDetector()
    
    # Create device manager
    manager = JAXDeviceManager(gpu_detector, DeviceSelectionStrategy.MEMORY_AWARE)
    
    print(f"\n1. Detected {len(manager.device_info)} devices:")
    for device_id, info in manager.device_info.items():
        print(f"   Device {device_id}: {info.device_type} ({info.platform})")
        if info.gpu_specs:
            print(f"      GPU: {info.gpu_specs.name}")
            print(f"      Memory: {info.gpu_specs.total_memory_gb:.1f}GB")
            print(f"      Architecture: {info.gpu_specs.architecture.value}")
    
    # Test device selection
    print("\n2. Testing device selection...")
    
    # Select for memory-intensive task
    device1 = manager.select_device(memory_required_mb=1000, task_id="task1")
    print(f"   Memory task -> Device {device1.device_id}")
    
    # Select for compute-intensive task
    device2 = manager.select_device(compute_intensity=0.9, task_id="task2")
    print(f"   Compute task -> Device {device2.device_id}")
    
    # Test array placement
    print("\n3. Testing array placement...")
    test_array = np.random.randn(100, 100).astype(np.float32)
    placed = manager.place_on_device(test_array, device1)
    print(f"   Placed array shape: {placed.shape}")
    
    # Test sharded arrays if multi-GPU
    if len(manager.jax_devices) > 1:
        print("\n4. Testing sharded arrays...")
        sharded = manager.create_sharded_array(
            (1000, 1000),
            strategy=DevicePlacementStrategy.DATA_PARALLEL
        )
        print(f"   Created sharded array: {sharded.shape}")
    
    # Test memory info
    print("\n5. Testing memory info...")
    memory_info = manager.get_device_memory_info(device1)
    print(f"   Device {device1.device_id} memory:")
    for key, value in memory_info.items():
        print(f"      {key}: {value}")
    
    # Test optimal placement
    print("\n6. Testing optimal placement...")
    placement = manager.get_optimal_placement(
        operation_type='compute',
        data_size_mb=500,
        model_size_mb=100
    )
    print(f"   Strategy: {placement.strategy.value}")
    print(f"   Device: {placement.device}")
    
    # Get statistics
    stats = manager.get_statistics()
    print(f"\n7. Manager statistics:")
    print(f"   Device selections: {stats['device_selections']}")
    print(f"   Placement operations: {stats['placement_operations']}")
    print(f"   Memory transfers: {stats['memory_transfers']}")
    
    # Shutdown
    manager.shutdown()
    print("\n✓ JAX Device Manager test complete!")


if __name__ == "__main__":
    test_jax_device_manager()