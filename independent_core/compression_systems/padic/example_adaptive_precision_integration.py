"""
Example: Integration of Adaptive Precision Wrapper with P-adic Compression System
Shows how to use the adaptive precision wrapper in a real compression pipeline
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from independent_core.compression_systems.padic.adaptive_precision_wrapper import (
    AdaptivePrecisionWrapper,
    AdaptivePrecisionConfig
)


class ModelCompressionExample:
    """Example of compressing a neural network model with adaptive precision"""
    
    def __init__(self):
        # Create a simple model
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        # Initialize adaptive precision wrapper
        self.config = AdaptivePrecisionConfig(
            prime=257,
            base_precision=4,
            min_precision=2,
            max_precision=4,
            target_error=1e-6,
            compression_priority=0.7  # Favor compression
        )
        self.wrapper = AdaptivePrecisionWrapper(self.config)
    
    def compute_weight_importance(self, layer: nn.Module) -> torch.Tensor:
        """Compute importance scores for layer weights"""
        if isinstance(layer, nn.Linear):
            weight = layer.weight.data
            
            # Simple importance: magnitude + gradient proxy
            importance = torch.abs(weight)
            
            # Add variance-based importance
            importance += torch.var(weight, dim=1, keepdim=True).expand_as(weight) * 0.1
            
            return importance
        return None
    
    def compress_model(self):
        """Compress the entire model with adaptive precision"""
        print("Compressing neural network model...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        compressed_layers = []
        total_original_size = 0
        total_compressed_size = 0
        
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Linear):
                print(f"\nLayer {i} ({layer.__class__.__name__}):")
                print(f"  Shape: {layer.weight.shape}")
                
                # Get weight tensor
                weight_tensor = layer.weight.data
                
                # Compute importance scores
                importance = self.compute_weight_importance(layer)
                
                # Calculate bit budget (aim for 60% compression)
                original_bits = weight_tensor.numel() * 32
                target_bits = int(original_bits * 0.4)
                
                # Compress with adaptive precision
                allocation = self.wrapper.batch_compress_with_adaptive_precision(
                    weight_tensor,
                    importance,
                    target_bits
                )
                
                # Store results
                compressed_layers.append({
                    'layer_idx': i,
                    'allocation': allocation,
                    'original_shape': weight_tensor.shape
                })
                
                # Track sizes
                total_original_size += original_bits / 8
                total_compressed_size += allocation.total_bits / 8
                
                # Print statistics
                print(f"  Compression ratio: {allocation.compression_ratio:.2f}x")
                print(f"  Average precision: {allocation.get_average_precision():.2f}")
                error_stats = allocation.get_error_statistics()
                print(f"  Max error: {error_stats['max']:.2e}")
                print(f"  Mean error: {error_stats['mean']:.2e}")
        
        # Overall statistics
        print("\n" + "="*50)
        print("Overall Compression Results:")
        print(f"  Original size: {total_original_size/1024:.2f} KB")
        print(f"  Compressed size: {total_compressed_size/1024:.2f} KB")
        print(f"  Total compression ratio: {total_original_size/total_compressed_size:.2f}x")
        print(f"  Space saved: {(1 - total_compressed_size/total_original_size)*100:.1f}%")
        
        return compressed_layers
    
    def decompress_and_validate(self, compressed_layers):
        """Decompress weights and validate reconstruction"""
        print("\n" + "="*50)
        print("Validating Decompression:")
        
        max_error = 0
        
        for comp_data in compressed_layers:
            layer_idx = comp_data['layer_idx']
            allocation = comp_data['allocation']
            original_shape = comp_data['original_shape']
            
            # Get original layer
            layer = self.model[layer_idx]
            original_weights = layer.weight.data.clone()
            
            # Reconstruct from p-adic weights
            reconstructed = torch.zeros_like(original_weights.flatten())
            
            for i, padic_weight in enumerate(allocation.weights):
                # Get appropriate precision operations
                precision = padic_weight.precision
                if precision == self.config.base_precision:
                    reconstructed[i] = self.wrapper.math_ops.from_padic(padic_weight)
                else:
                    precision_ops = self.wrapper._get_precision_ops(precision)
                    reconstructed[i] = precision_ops.from_padic(padic_weight)
            
            reconstructed = reconstructed.reshape(original_shape)
            
            # Calculate reconstruction error
            error = torch.abs(original_weights - reconstructed)
            rel_error = error / (torch.abs(original_weights) + 1e-10)
            
            max_layer_error = rel_error.max().item()
            mean_layer_error = rel_error.mean().item()
            
            print(f"  Layer {layer_idx}:")
            print(f"    Max relative error: {max_layer_error:.2e}")
            print(f"    Mean relative error: {mean_layer_error:.2e}")
            
            max_error = max(max_error, max_layer_error)
            
            # Update model with reconstructed weights (for testing)
            layer.weight.data = reconstructed
        
        print(f"\nOverall max error: {max_error:.2e}")
        return max_error < self.config.target_error * 10  # Allow 10x target for validation


def gradient_based_compression_example():
    """Example: Compress gradients during training with adaptive precision"""
    print("\n" + "="*70)
    print("Gradient Compression Example")
    print("="*70)
    
    # Setup
    model = nn.Linear(100, 50)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    config = AdaptivePrecisionConfig(
        prime=257,
        base_precision=3,
        min_precision=2,
        max_precision=4,
        target_error=1e-5,
        compression_priority=0.8  # High compression for gradients
    )
    wrapper = AdaptivePrecisionWrapper(config)
    
    # Simulate training step
    x = torch.randn(32, 100)
    y = torch.randn(32, 50)
    
    # Forward pass
    output = model(x)
    loss = nn.MSELoss()(output, y)
    
    # Backward pass
    loss.backward()
    
    print("Original gradient statistics:")
    grad = model.weight.grad
    print(f"  Shape: {grad.shape}")
    print(f"  Mean: {grad.mean():.4f}")
    print(f"  Std: {grad.std():.4f}")
    print(f"  Size: {grad.numel() * 4} bytes")
    
    # Compress gradients with importance based on magnitude
    importance = torch.abs(grad) + 1e-8
    
    # Allocate precision based on gradient importance
    total_bits = grad.numel() * 8  # Target 75% compression
    allocation = wrapper.batch_compress_with_adaptive_precision(
        grad, importance, total_bits
    )
    
    print("\nCompressed gradient statistics:")
    print(f"  Compression ratio: {allocation.compression_ratio:.2f}x")
    print(f"  Average precision: {allocation.get_average_precision():.2f}")
    print(f"  Compressed size: {allocation.total_bits // 8} bytes")
    print(f"  Max error: {allocation.error_map.max():.2e}")
    
    # Monitor efficiency
    metrics = wrapper.monitor_precision_efficiency(allocation)
    print(f"  Efficiency score: {metrics['efficiency_score']:.3f}")
    
    # Show precision distribution
    print("\nPrecision distribution:")
    histogram = metrics['precision_histogram']
    for i, precision in enumerate(histogram['bins']):
        count = histogram['counts'][i]
        freq = histogram['frequencies'][i]
        print(f"  Precision {precision}: {count} weights ({freq*100:.1f}%)")


def streaming_compression_example():
    """Example: Stream processing large tensors with adaptive precision"""
    print("\n" + "="*70)
    print("Streaming Compression Example")
    print("="*70)
    
    config = AdaptivePrecisionConfig(
        prime=257,
        base_precision=3,
        batch_size=1024,
        enable_memory_tracking=True
    )
    wrapper = AdaptivePrecisionWrapper(config)
    
    # Simulate streaming large data
    total_elements = 10000
    chunk_size = 1000
    
    print(f"Processing {total_elements} elements in chunks of {chunk_size}...")
    
    all_compressions = []
    
    for chunk_idx in range(0, total_elements, chunk_size):
        # Generate chunk
        chunk = torch.randn(chunk_size) * (10 ** (chunk_idx / total_elements))
        
        # Adaptive compression based on value distribution
        importance = torch.abs(chunk) / chunk.abs().max()
        
        allocation = wrapper.batch_compress_with_adaptive_precision(
            chunk, importance
        )
        
        all_compressions.append(allocation)
        
        if chunk_idx % 2000 == 0:
            print(f"  Chunk {chunk_idx//chunk_size}: ratio={allocation.compression_ratio:.2f}x, "
                  f"avg_precision={allocation.get_average_precision():.2f}")
    
    # Aggregate statistics
    total_original = sum(chunk_size * 32 for _ in all_compressions)
    total_compressed = sum(a.total_bits for a in all_compressions)
    overall_ratio = total_original / total_compressed
    
    print(f"\nOverall streaming compression:")
    print(f"  Total compression ratio: {overall_ratio:.2f}x")
    print(f"  Space saved: {(1 - total_compressed/total_original)*100:.1f}%")
    
    # Get final statistics
    stats = wrapper.get_statistics()
    print(f"  Total weights processed: {stats['total_weights']}")
    print(f"  Average processing time: {stats.get('mean_processing_time', 0)*1000:.2f}ms")


def main():
    """Run all examples"""
    print("="*70)
    print("ADAPTIVE PRECISION WRAPPER - INTEGRATION EXAMPLES")
    print("="*70)
    
    # Example 1: Compress a neural network model
    print("\n1. Neural Network Model Compression")
    print("-"*40)
    model_example = ModelCompressionExample()
    compressed = model_example.compress_model()
    model_example.decompress_and_validate(compressed)
    
    # Example 2: Gradient compression during training
    gradient_based_compression_example()
    
    # Example 3: Streaming compression for large data
    streaming_compression_example()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()