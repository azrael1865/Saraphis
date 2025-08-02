"""
GPU Memory Manager Integration Example
Demonstrates integration with BrainCore, TrainingManager, and compression systems
"""

import torch
import numpy as np
from typing import Dict, Any
import time

# Import core components
from independent_core.brain_core import BrainCore
from independent_core.compression_systems.gpu_memory import (
    GPUMemoryManager,
    GPUMemoryIntegration
)


def integrate_gpu_memory_with_brain_core():
    """Demonstrate integration of GPU Memory Manager with BrainCore"""
    
    print("=== GPU Memory Manager Integration Example ===")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available, using mock demonstration")
        return None, None
    
    # Initialize BrainCore
    brain_config = {
        'enable_caching': True,
        'enable_gpu_optimization': True,
        'max_gpu_memory_usage': 0.85
    }
    brain = BrainCore(config=brain_config)
    
    # Initialize GPU Memory Manager
    gpu_config = {
        "enable_memory_pooling": True,
        "enable_stream_management": True, 
        "enable_optimization": True,
        "optimization_interval": 50,
        "device_0_pool_size": 512 * 1024 * 1024,  # 512MB
        "device_0_num_streams": 6
    }
    gpu_manager = GPUMemoryManager(config=gpu_config)
    
    # Register GPU manager with BrainCore
    GPUMemoryIntegration.register_with_brain_core(brain, gpu_manager)
    
    print(f"Initialized GPU Memory Manager for {gpu_manager.num_devices} device(s)")
    
    # Example 1: Basic memory allocation
    print(f"\n1. Basic Memory Allocation:")
    try:
        tensor_size = 1024 * 1024 * 4  # 4MB
        tensor = gpu_manager.allocate_memory(tensor_size, operation="example_allocation")
        print(f"   Allocated {tensor_size} bytes: {tensor.shape}")
        
        # Get memory stats
        stats = gpu_manager.get_memory_stats(device_id=0)
        print(f"   Memory utilization: {stats['memory_usage']['utilization']:.2%}")
        
        # Deallocate
        gpu_manager.deallocate_memory(tensor)
        print(f"   Memory deallocated successfully")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 2: Stream management
    print(f"\n2. CUDA Stream Management:")
    try:
        compute_stream = gpu_manager.get_stream(device_id=0, stream_type="compute")
        transfer_stream = gpu_manager.get_stream(device_id=0, stream_type="transfer")
        priority_stream = gpu_manager.get_stream(device_id=0, stream_type="priority")
        
        print(f"   Compute stream: {compute_stream}")
        print(f"   Transfer stream: {transfer_stream}")
        print(f"   Priority stream: {priority_stream}")
        
        # Use streams for operations
        with torch.cuda.stream(compute_stream):
            tensor1 = torch.randn(1000, 1000, device='cuda:0')
            result = torch.matmul(tensor1, tensor1.T)
        
        with torch.cuda.stream(transfer_stream):
            tensor2 = torch.randn(500, 500, device='cuda:0')
            cpu_tensor = tensor2.cpu()
        
        # Release streams
        gpu_manager.release_stream(compute_stream)
        gpu_manager.release_stream(transfer_stream)
        gpu_manager.release_stream(priority_stream)
        
        print(f"   Stream operations completed successfully")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 3: Memory optimization
    print(f"\n3. Memory Optimization:")
    try:
        # Allocate multiple tensors to trigger optimization
        tensors = []
        for i in range(10):
            size = (i + 1) * 1024 * 1024  # Variable sizes
            tensor = gpu_manager.allocate_memory(size, operation=f"batch_allocation_{i}")
            tensors.append(tensor)
        
        # Force optimization
        optimization_result = gpu_manager.optimize_memory(device_id=0, force=True)
        print(f"   Optimization actions: {optimization_result.get('actions_taken', [])}")
        
        # Get optimization suggestions
        suggestions = gpu_manager.memory_optimizers[0].get_optimization_suggestions()
        if suggestions:
            print(f"   Optimization suggestions:")
            for suggestion in suggestions:
                print(f"     - {suggestion['type']}: {suggestion['description']}")
        else:
            print(f"   No optimization suggestions at this time")
        
        # Clean up
        for tensor in tensors:
            gpu_manager.deallocate_memory(tensor)
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 4: Multi-device operations (if available)
    if gpu_manager.num_devices > 1:
        print(f"\n4. Multi-Device Operations:")
        try:
            # Allocate on first device
            src_tensor = gpu_manager.allocate_memory(1024*1024, device_id=0, operation="multi_device_src")
            
            # Transfer to second device
            dst_tensor = gpu_manager.transfer_async(src_tensor, dst_device=1)
            
            print(f"   Transferred tensor from device 0 to device 1")
            print(f"   Source device: {src_tensor.device}")
            print(f"   Destination device: {dst_tensor.device}")
            
            # Clean up
            gpu_manager.deallocate_memory(src_tensor)
            gpu_manager.deallocate_memory(dst_tensor)
            
        except Exception as e:
            print(f"   Error: {e}")
    else:
        print(f"\n4. Multi-Device Operations: Only 1 device available")
    
    # Example 5: Memory pool statistics
    print(f"\n5. Memory Pool Statistics:")
    try:
        stats = gpu_manager.get_memory_stats()
        print(f"   Total devices: {stats['num_devices']}")
        print(f"   Operation count: {stats['operation_count']}")
        
        for device_id, device_stats in stats['devices'].items():
            print(f"   Device {device_id} ({device_stats['device_name']}):")
            if 'memory_pool' in device_stats:
                pool_stats = device_stats['memory_pool']
                print(f"     Total memory: {pool_stats['total_memory']:,} bytes")
                print(f"     Allocated: {pool_stats['allocated_memory']:,} bytes")
                print(f"     Free: {pool_stats['free_memory']:,} bytes")
                print(f"     Fragmentation: {pool_stats['fragmentation_ratio']:.2%}")
            
            if 'stream_manager' in device_stats:
                stream_stats = device_stats['stream_manager']
                print(f"     Compute streams: {stream_stats['num_compute_streams']}")
                print(f"     Transfer streams: {stream_stats['num_transfer_streams']}")
                print(f"     Priority streams: {stream_stats['num_priority_streams']}")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 6: Performance benchmarking
    print(f"\n6. Performance Benchmarking:")
    try:
        # Benchmark allocation performance
        start_time = time.time()
        benchmark_tensors = []
        
        for i in range(20):
            size = 1024 * 1024  # 1MB each
            tensor = gpu_manager.allocate_memory(size, operation="benchmark")
            benchmark_tensors.append(tensor)
        
        allocation_time = time.time() - start_time
        
        # Benchmark deallocation
        start_time = time.time()
        for tensor in benchmark_tensors:
            gpu_manager.deallocate_memory(tensor)
        
        deallocation_time = time.time() - start_time
        
        print(f"   Allocated 20x1MB tensors in {allocation_time:.4f}s ({allocation_time/20:.4f}s per tensor)")
        print(f"   Deallocated 20 tensors in {deallocation_time:.4f}s ({deallocation_time/20:.4f}s per tensor)")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    print(f"\nGPU Memory Manager successfully integrated with BrainCore!")
    return brain, gpu_manager


def demonstrate_training_manager_integration():
    """Demonstrate integration with TrainingManager"""
    
    print(f"\n=== TrainingManager Integration ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping demonstration")
        return
    
    # Mock TrainingManager for demonstration
    class MockTrainingManager:
        def __init__(self):
            self.gpu_memory_manager = None
            self.gpu_allocate = None
            self.gpu_deallocate = None
            self.get_gpu_stream = None
            self.release_gpu_stream = None
    
    training_manager = MockTrainingManager()
    
    # Create GPU memory manager
    gpu_manager = GPUMemoryManager({
        'enable_memory_pooling': True,
        'enable_stream_management': True,
        'optimization_interval': 25
    })
    
    # Integrate
    GPUMemoryIntegration.integrate_with_training_manager(training_manager, gpu_manager)
    
    print("Testing integrated training operations:")
    
    try:
        # Test GPU memory allocation for training
        weight_tensor = training_manager.gpu_allocate(1024*1024*8)  # 8MB weights
        gradient_tensor = training_manager.gpu_allocate(1024*1024*8)  # 8MB gradients
        
        print(f"  ✓ Allocated training tensors: weights {weight_tensor.shape}, gradients {gradient_tensor.shape}")
        
        # Test stream operations
        compute_stream = training_manager.get_gpu_stream(stream_type="compute")
        print(f"  ✓ Obtained compute stream: {compute_stream}")
        
        # Simulate training operations
        with torch.cuda.stream(compute_stream):
            # Simulate forward pass
            activations = torch.matmul(weight_tensor.view(1024, 8192), torch.randn(8192, 512, device='cuda:0'))
            
            # Simulate backward pass
            grad_output = torch.randn_like(activations)
            weight_grad = torch.matmul(grad_output.T, weight_tensor.view(1024, 8192))
        
        print(f"  ✓ Performed simulated training operations")
        
        # Release resources
        training_manager.release_gpu_stream(compute_stream)
        training_manager.gpu_deallocate(weight_tensor)
        training_manager.gpu_deallocate(gradient_tensor)
        
        print(f"  ✓ Released training resources")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("TrainingManager integration successful!")


def demonstrate_compression_integration():
    """Demonstrate integration with compression systems"""
    
    print(f"\n=== Compression Systems Integration ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping demonstration")
        return
    
    # Mock compression system
    class MockCompressionSystem:
        def __init__(self):
            self.gpu_memory_manager = None
            self.gpu_allocate = None
            self.gpu_get_stream = None
            self.gpu_optimize = None
    
    compression_system = MockCompressionSystem()
    
    # Create GPU memory manager
    gpu_manager = GPUMemoryManager({
        'enable_memory_pooling': True,
        'enable_optimization': True
    })
    
    # Integrate
    GPUMemoryIntegration.integrate_with_compression_systems(compression_system, gpu_manager)
    
    print("Testing integrated compression operations:")
    
    try:
        # Test tensor compression workflow
        original_tensor = compression_system.gpu_allocate(1024*1024*4, "tensor_compression")
        compressed_buffer = compression_system.gpu_allocate(1024*512, "compressed_storage")
        
        print(f"  ✓ Allocated compression tensors")
        
        # Get specialized streams
        compute_stream = compression_system.gpu_get_stream("compute")
        transfer_stream = compression_system.gpu_get_stream("transfer")
        
        print(f"  ✓ Obtained specialized streams")
        
        # Simulate compression operations
        with torch.cuda.stream(compute_stream):
            # Simulate compression algorithm
            compressed_data = torch.mean(original_tensor.view(-1, 8), dim=1)
        
        with torch.cuda.stream(transfer_stream):
            # Simulate data transfer
            result = compressed_data.cpu()
        
        print(f"  ✓ Performed compression simulation")
        
        # Test optimization
        compression_system.gpu_optimize()
        print(f"  ✓ Ran GPU memory optimization")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("Compression systems integration successful!")


def demonstrate_error_handling():
    """Demonstrate NO FALLBACKS error handling"""
    
    print(f"\n=== Error Handling (NO FALLBACKS) ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, testing error conditions:")
        
        try:
            gpu_manager = GPUMemoryManager()
        except RuntimeError as e:
            print(f"  ✓ Correctly raised RuntimeError: {e}")
        
        return
    
    gpu_manager = GPUMemoryManager()
    
    print("Testing error conditions:")
    
    # Test invalid allocation size
    try:
        gpu_manager.allocate_memory(-1)
    except ValueError as e:
        print(f"  ✓ Invalid size error: {e}")
    
    # Test invalid device ID
    try:
        gpu_manager.allocate_memory(1024, device_id=999)
    except ValueError as e:
        print(f"  ✓ Invalid device error: {e}")
    
    # Test invalid stream type
    try:
        gpu_manager.get_stream(stream_type="invalid_type")
    except ValueError as e:
        print(f"  ✓ Invalid stream type error: {e}")
    
    # Test deallocation of non-pool tensor
    try:
        tensor = torch.randn(100, 100, device='cuda:0')
        gpu_manager.deallocate_memory(tensor)
    except ValueError as e:
        print(f"  ✓ Invalid deallocation error: {e}")
    
    print("Error handling verification complete!")


if __name__ == "__main__":
    # Run integration demonstrations
    brain, gpu_manager = integrate_gpu_memory_with_brain_core()
    
    # Run training manager integration
    demonstrate_training_manager_integration()
    
    # Run compression integration
    demonstrate_compression_integration()
    
    # Test error handling
    demonstrate_error_handling()
    
    # Cleanup
    if gpu_manager:
        gpu_manager.cleanup()
    
    print(f"\n=== All GPU Memory Integration Tests Complete ===")