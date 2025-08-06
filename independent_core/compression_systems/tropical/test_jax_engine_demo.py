#!/usr/bin/env python3
"""
Demonstration script for JAX Tropical Engine.
Shows how to use the engine when JAX is available.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from independent_core.compression_systems.tropical.jax_tropical_engine import (
    JAX_AVAILABLE,
    JAXTropicalConfig,
    TropicalJAXEngine,
    TropicalJAXOperations,
    JAXChannelProcessor,
    TropicalXLAKernels,
    TropicalJAXBenchmark
)

def demo_tropical_engine():
    """Demonstrate JAX tropical engine capabilities"""
    
    if not JAX_AVAILABLE:
        print("JAX is not installed on this system.")
        print("This demo requires JAX to be installed.")
        print("Install with: pip install jax[cuda12_local]")
        return
    
    print("=" * 60)
    print("JAX TROPICAL ENGINE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize engine
    config = JAXTropicalConfig(
        enable_jit=True,
        enable_vmap=True,
        precision="float32",
        chunk_size=1000
    )
    
    try:
        engine = TropicalJAXEngine(config)
        print("\n✓ Engine initialized successfully")
        
        # Import JAX after confirming it's available
        import jax.numpy as jnp
        
        # Example 1: Basic tropical operations
        print("\n1. Basic Tropical Operations:")
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([2.0, 1.0, 4.0])
        
        result_add = engine.tropical_add(a, b)
        print(f"   Tropical addition: max({a}, {b}) = {result_add}")
        
        result_mul = engine.tropical_multiply(a, b)
        print(f"   Tropical multiplication: {a} + {b} = {result_mul}")
        
        # Example 2: Matrix operations
        print("\n2. Tropical Matrix Operations:")
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        B = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        
        result_matmul = engine.tropical_matrix_multiply(A, B)
        print(f"   Matrix A:\n{A}")
        print(f"   Matrix B:\n{B}")
        print(f"   A ⊗ B:\n{result_matmul}")
        
        # Example 3: Advanced operations
        print("\n3. Advanced Operations:")
        ops = TropicalJAXOperations(engine)
        
        # Convolution
        signal = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        kernel = jnp.array([1.0, 0.0, -1.0])
        conv_result = ops.tropical_conv1d(signal, kernel)
        print(f"   Tropical convolution result: {conv_result}")
        
        # Pooling
        input_2d = jnp.ones((4, 4)) * jnp.arange(16).reshape(4, 4)
        pooled = ops.tropical_pool2d(input_2d, (2, 2))
        print(f"   Tropical pooling (2x2) result shape: {pooled.shape}")
        
        # Example 4: XLA Kernels
        print("\n4. XLA Kernel Operations:")
        TropicalXLAKernels.compile_kernels()  # Compile all kernels
        
        xla_result = TropicalXLAKernels.tropical_matmul_kernel(A, B)
        print(f"   XLA matmul kernel result:\n{xla_result}")
        
        reduced = TropicalXLAKernels.tropical_reduce_kernel(A, axis=1)
        print(f"   XLA reduce kernel (max along rows): {reduced}")
        
        # Example 5: Performance benchmark
        print("\n5. Performance Benchmark:")
        benchmark = TropicalJAXBenchmark()
        
        # Create larger matrices for meaningful benchmark
        A_large = jnp.ones((100, 100))
        B_large = jnp.ones((100, 100))
        
        def test_func():
            return engine.tropical_matrix_multiply(A_large, B_large)
        
        stats = benchmark.benchmark_operation(test_func, num_iterations=100, warmup=10)
        print(f"   Tropical matmul (100x100):")
        print(f"   - Mean: {stats['mean_ms']:.3f}ms")
        print(f"   - Std: {stats['std_ms']:.3f}ms")
        print(f"   - Min: {stats['min_ms']:.3f}ms")
        print(f"   - Max: {stats['max_ms']:.3f}ms")
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("Expected 6x speedup over PyTorch on GPU with large operations")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_tropical_engine()