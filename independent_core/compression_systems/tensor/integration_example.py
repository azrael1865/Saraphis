"""
Tensor Decomposition Compression System Integration Example
Demonstrates integration with BrainCore, TrainingManager, and GradientCompressionComponent
"""

import torch
import numpy as np
from typing import Dict, Any

# Import core components
from independent_core.brain_core import BrainCore
from independent_core.compression_systems.tensor import (
    TensorCompressionSystem,
    DecompositionType,
    TensorDecompositionIntegration
)


def integrate_tensor_with_brain_core():
    """Demonstrate integration of Tensor compression with BrainCore"""
    
    # Initialize BrainCore
    brain_config = {
        'enable_caching': True,
        'max_compression_systems': 10,
        'default_compression': 'tensor'
    }
    brain = BrainCore(config=brain_config)
    
    # Initialize Tensor compression system with specific configuration
    tensor_config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dtype': torch.float32,
        'default_method': DecompositionType.TUCKER,
        'target_compression_ratio': 0.1,
        'gpu_memory_threshold': 0.9,
        'enable_caching': True,
        'cache_size_limit': 50,
        'max_iterations': 100,
        'tolerance': 1e-6
    }
    tensor_system = TensorCompressionSystem(config=tensor_config)
    
    # Register Tensor system with BrainCore
    TensorDecompositionIntegration.register_with_brain_core(brain, tensor_system)
    brain.active_compression = 'tensor_decomposition'
    
    print("=== Tensor Decomposition Integration Example ===")
    
    # Example 1: Compress 3D tensor (typical neural network layer)
    layer_weights = torch.randn(128, 256, 64, dtype=torch.float32)
    
    print(f"\nCompressing 3D tensor with shape {layer_weights.shape}...")
    compressed_3d = tensor_system.compress(layer_weights)
    print(f"Method: {compressed_3d.decomposition_type.name}")
    print(f"Compression ratio: {compressed_3d.compression_ratio:.4f}")
    print(f"Reconstruction error: {compressed_3d.reconstruction_error:.6f}")
    print(f"Number of factors: {len(compressed_3d.factors)}")
    print(f"Ranks: {compressed_3d.ranks}")
    
    # Decompress and verify
    decompressed_3d = tensor_system.decompress(compressed_3d)
    print(f"Decompressed shape: {decompressed_3d.shape}")
    
    # Example 2: CP decomposition of 4D tensor
    tensor_system.set_decomposition_method(DecompositionType.CP)
    conv_weights = torch.randn(64, 128, 3, 3, dtype=torch.float32)  # Conv layer
    
    print(f"\nCompressing 4D conv tensor with CP decomposition...")
    compressed_cp = tensor_system.compress(conv_weights)
    print(f"Method: {compressed_cp.decomposition_type.name}")
    print(f"Compression ratio: {compressed_cp.compression_ratio:.4f}")
    print(f"Reconstruction error: {compressed_cp.reconstruction_error:.6f}")
    print(f"CP rank: {compressed_cp.ranks[0]}")
    
    # Example 3: Tensor-Train decomposition
    tensor_system.set_decomposition_method(DecompositionType.TT)
    large_tensor = torch.randn(16, 32, 16, 32, dtype=torch.float32)
    
    print(f"\nCompressing 4D tensor with Tensor-Train decomposition...")
    compressed_tt = tensor_system.compress(large_tensor)
    print(f"Method: {compressed_tt.decomposition_type.name}")
    print(f"Compression ratio: {compressed_tt.compression_ratio:.4f}")
    print(f"Reconstruction error: {compressed_tt.reconstruction_error:.6f}")
    print(f"TT ranks: {compressed_tt.ranks}")
    print(f"Number of TT cores: {len(compressed_tt.factors)}")
    
    # Example 4: Batch compression
    tensor_batch = [
        torch.randn(64, 128, dtype=torch.float32),
        torch.randn(128, 256, dtype=torch.float32),
        torch.randn(256, 512, dtype=torch.float32)
    ]
    
    print(f"\nBatch compressing {len(tensor_batch)} tensors...")
    tensor_system.set_decomposition_method(DecompositionType.TUCKER)
    compressed_batch = tensor_system.compress_batch(tensor_batch)
    
    for i, comp in enumerate(compressed_batch):
        print(f"  Tensor {i}: ratio={comp.compression_ratio:.4f}, error={comp.reconstruction_error:.6f}")
    
    # Decompress batch
    decompressed_batch = tensor_system.decompress_batch(compressed_batch)
    print(f"Decompressed batch: {[t.shape for t in decompressed_batch]}")
    
    # Example 5: GPU memory management demonstration
    if torch.cuda.is_available():
        print(f"\nGPU Memory Management:")
        gpu_tensor = torch.randn(512, 1024, 128, device='cuda', dtype=torch.float32)
        print(f"GPU tensor size: {gpu_tensor.shape}")
        
        try:
            compressed_gpu = tensor_system.compress(gpu_tensor)
            print(f"GPU compression successful: ratio={compressed_gpu.compression_ratio:.4f}")
            if 'compression_time_ms' in compressed_gpu.metadata:
                print(f"GPU compression time: {compressed_gpu.metadata['compression_time_ms']:.2f}ms")
        except RuntimeError as e:
            print(f"GPU memory management triggered: {e}")
    
    # Example 6: Rank optimization demonstration
    print(f"\nRank Optimization:")
    test_tensor = torch.randn(32, 64, 32, dtype=torch.float32)
    
    # Test different target compression ratios
    for target_ratio in [0.05, 0.1, 0.2, 0.3]:
        tensor_system.set_target_compression_ratio(target_ratio)
        tensor_system.set_decomposition_method(DecompositionType.TUCKER)
        
        compressed = tensor_system.compress(test_tensor)
        print(f"  Target: {target_ratio:.2f}, Actual: {compressed.compression_ratio:.4f}, "
              f"Ranks: {compressed.ranks}, Error: {compressed.reconstruction_error:.6f}")
    
    # Example 7: Error handling demonstration (NO FALLBACKS)
    print(f"\nError Handling (NO FALLBACKS):")
    
    try:
        # Invalid tensor
        tensor_system.compress(None)
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    try:
        # NaN tensor
        nan_tensor = torch.tensor([[float('nan'), 1.0], [2.0, 3.0]])
        tensor_system.compress(nan_tensor)
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for NaN: {e}")
    
    try:
        # Invalid decomposition data
        tensor_system.decompress(None)
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    # Example 8: Compression statistics
    print(f"\nCompression Statistics:")
    stats = tensor_system.get_compression_stats()
    print(f"  Total compressions: {stats['metrics'].get('total_compressions', 0)}")
    if 'average_compression_ratio' in stats['metrics']:
        print(f"  Average compression ratio: {stats['metrics']['average_compression_ratio']:.4f}")
        print(f"  Average reconstruction error: {stats['metrics']['average_reconstruction_error']:.6f}")
    print(f"  Cache size: {stats['cache_size']}")
    print(f"  Supported methods: {', '.join(stats['supported_methods'])}")
    print(f"  Current method: {stats['current_method']}")
    
    # Example 9: Advanced decomposition features
    print(f"\nAdvanced Features:")
    
    # Custom ranks
    custom_tensor = torch.randn(50, 60, 70, dtype=torch.float32)
    custom_config = tensor_config.copy()
    custom_config['ranks'] = [20, 25, 30]  # Custom Tucker ranks
    custom_config['method'] = DecompositionType.TUCKER
    
    custom_system = TensorCompressionSystem(config=custom_config)
    custom_compressed = custom_system.compress(custom_tensor)
    print(f"Custom ranks compression: {custom_compressed.ranks}")
    print(f"Custom compression ratio: {custom_compressed.compression_ratio:.4f}")
    
    # Method comparison
    print(f"\nMethod Comparison for same tensor:")
    comparison_tensor = torch.randn(32, 48, 32, dtype=torch.float32)
    
    for method in [DecompositionType.CP, DecompositionType.TUCKER, DecompositionType.TT]:
        tensor_system.set_decomposition_method(method)
        tensor_system.set_target_compression_ratio(0.15)
        
        try:
            comp = tensor_system.compress(comparison_tensor)
            print(f"  {method.name}: ratio={comp.compression_ratio:.4f}, "
                  f"error={comp.reconstruction_error:.6f}, ranks={comp.ranks}")
        except Exception as e:
            print(f"  {method.name}: Failed - {e}")
    
    # Example 10: Memory usage analysis
    print(f"\nMemory Usage Analysis:")
    analysis_tensor = torch.randn(64, 96, 64, dtype=torch.float32)
    original_memory = analysis_tensor.numel() * analysis_tensor.element_size()
    
    for method in [DecompositionType.TUCKER, DecompositionType.CP]:
        tensor_system.set_decomposition_method(method)
        comp = tensor_system.compress(analysis_tensor)
        
        compressed_memory = comp.get_memory_usage()
        savings = original_memory - compressed_memory
        
        print(f"  {method.name}:")
        print(f"    Original: {original_memory:,} bytes")
        print(f"    Compressed: {compressed_memory:,} bytes")
        print(f"    Savings: {savings:,} bytes ({100 * savings / original_memory:.1f}%)")
        print(f"    Reconstruction error: {comp.reconstruction_error:.6f}")
    
    print(f"\nTensor compression system successfully integrated with BrainCore!")
    return brain, tensor_system


def demonstrate_training_manager_integration():
    """Demonstrate integration with TrainingManager"""
    print(f"\n=== TrainingManager Integration ===")
    
    # Mock TrainingManager for demonstration
    class MockTrainingManager:
        def __init__(self):
            self.tensor_operations = {}
    
    training_manager = MockTrainingManager()
    
    # Create tensor system
    tensor_system = TensorCompressionSystem({
        'default_method': DecompositionType.TUCKER,
        'target_compression_ratio': 0.2
    })
    
    # Integrate
    TensorDecompositionIntegration.integrate_with_training_manager(training_manager, tensor_system)
    
    # Test tensor operations
    test_weight = torch.randn(128, 256, dtype=torch.float32)
    
    print("Testing integrated tensor operations:")
    
    # CP decomposition
    cp_result = training_manager.tensor_operations['cp_decomposition'](test_weight)
    print(f"CP operation: ratio={cp_result.compression_ratio:.4f}")
    
    # Tucker decomposition  
    tucker_result = training_manager.tensor_operations['tucker_decomposition'](test_weight)
    print(f"Tucker operation: ratio={tucker_result.compression_ratio:.4f}")
    
    # Reconstruction
    reconstructed = training_manager.tensor_operations['tensor_reconstruct'](tucker_result)
    print(f"Reconstruction shape: {reconstructed.shape}")
    
    print("TrainingManager integration successful!")


def demonstrate_gradient_compression_integration():
    """Demonstrate integration with GradientCompressionComponent"""
    print(f"\n=== GradientCompressionComponent Integration ===")
    
    # Mock GradientCompressionComponent
    class MockGradientComponent:
        def __init__(self):
            self.tensor_decomposition_methods = {}
    
    gradient_component = MockGradientComponent()
    
    # Create tensor system
    tensor_system = TensorCompressionSystem({
        'default_method': DecompositionType.TUCKER
    })
    
    # Integrate
    TensorDecompositionIntegration.integrate_with_gradient_compression(gradient_component, tensor_system)
    
    # Test gradient compression
    gradient_tensor = torch.randn(256, 512, dtype=torch.float32)
    
    print("Testing integrated gradient compression methods:")
    
    # Test each method
    for method_name in ['cp', 'tucker', 'tt']:
        if method_name in gradient_component.tensor_decomposition_methods:
            method = gradient_component.tensor_decomposition_methods[method_name]
            
            try:
                if method_name == 'cp':
                    ranks = [32] * gradient_tensor.ndim
                elif method_name == 'tt':
                    ranks = [32] * (gradient_tensor.ndim - 1)
                else:  # tucker
                    ranks = [128, 256]
                
                result = method.decompose(gradient_tensor, ranks)
                print(f"  {method_name.upper()}: ratio={result.compression_ratio:.4f}, "
                      f"error={result.reconstruction_error:.6f}")
            except Exception as e:
                print(f"  {method_name.upper()}: Error - {e}")
    
    print("GradientCompressionComponent integration successful!")


if __name__ == "__main__":
    # Run integration demonstrations
    brain, tensor_system = integrate_tensor_with_brain_core()
    
    # Run training manager integration
    demonstrate_training_manager_integration()
    
    # Run gradient compression integration
    demonstrate_gradient_compression_integration()
    
    print(f"\n=== All Integration Tests Complete ===")