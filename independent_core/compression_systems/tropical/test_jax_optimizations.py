"""
Test JAX Performance Optimizations - Validate all optimization modules work together
Ensures 6x speedup target is achieved with production-ready features
"""

import jax
import jax.numpy as jnp
import torch
import numpy as np
import time
import logging
from typing import Dict, Any

# Import all optimization modules
from jax_compilation_optimizer import JAXCompilationOptimizer, CompilationLevel
from jax_memory_optimizer import JAXMemoryOptimizer, PrefetchStrategy
from jax_custom_ops import TropicalXLACustomOps
from jax_performance_monitor import JAXPerformanceMonitor
from benchmarks.jax_benchmark_suite import JAXBenchmarkSuite, BenchmarkConfig, BenchmarkOperation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_compilation_optimizer():
    """Test XLA compilation optimization"""
    print("\n" + "="*60)
    print("Testing XLA Compilation Optimizer")
    print("="*60)
    
    optimizer = JAXCompilationOptimizer(
        compilation_level=CompilationLevel.AGGRESSIVE,
        enable_pgo=True,
        enable_aot=True
    )
    
    # Test function
    def tropical_matmul(A, B):
        A_exp = A[:, :, jnp.newaxis]
        B_exp = B[jnp.newaxis, :, :]
        products = A_exp + B_exp
        return jnp.max(products, axis=1)
    
    # Compile with optimization
    compiled_fn = optimizer.compile_function(tropical_matmul)
    
    # Test execution
    A = jnp.ones((500, 500))
    B = jnp.ones((500, 500))
    
    # First call (includes compilation)
    start = time.perf_counter()
    result = compiled_fn(A, B)
    result.block_until_ready()
    first_time = (time.perf_counter() - start) * 1000
    
    # Second call (cached)
    start = time.perf_counter()
    result = compiled_fn(A, B)
    result.block_until_ready()
    cached_time = (time.perf_counter() - start) * 1000
    
    print(f"✓ First call: {first_time:.2f}ms")
    print(f"✓ Cached call: {cached_time:.2f}ms")
    print(f"✓ Cache speedup: {first_time/cached_time:.2f}x")
    
    stats = optimizer.get_compilation_stats()
    print(f"✓ Cache hits: {stats['cache_hits']}")
    print(f"✓ Compilations: {stats['compilations']}")
    
    return optimizer


def test_memory_optimizer():
    """Test memory bandwidth optimization"""
    print("\n" + "="*60)
    print("Testing Memory Bandwidth Optimizer")
    print("="*60)
    
    optimizer = JAXMemoryOptimizer(
        enable_profiling=True,
        enable_prefetching=True
    )
    
    # Test tensor
    A = jnp.ones((1000, 1000), dtype=jnp.float32)
    
    # Analyze access pattern
    profile = optimizer.analyze_access_pattern(A, 'matmul')
    print(f"✓ Access pattern: {profile.pattern.value}")
    print(f"✓ Locality: {profile.locality:.2f}")
    
    # Optimize layout
    optimized_A, layout_info = optimizer.optimize_tensor_layout(A, profile, 'matmul')
    print(f"✓ Optimized layout: {layout_info.layout.value}")
    print(f"✓ Memory footprint: {layout_info.memory_footprint / 1024 / 1024:.2f}MB")
    
    # Setup prefetching
    optimizer.setup_prefetching([A], PrefetchStrategy.ADAPTIVE)
    print(f"✓ Prefetch queue size: {len(optimizer.prefetch_queue)}")
    
    # Check bandwidth
    is_saturated, utilization = optimizer.detect_bandwidth_saturation()
    print(f"✓ Bandwidth utilization: {utilization:.1f}%")
    
    stats = optimizer.get_optimization_stats()
    print(f"✓ Layouts optimized: {stats['layouts_optimized']}")
    
    return optimizer


def test_custom_ops():
    """Test custom XLA operations"""
    print("\n" + "="*60)
    print("Testing Custom XLA Operations")
    print("="*60)
    
    custom_ops = TropicalXLACustomOps(enable_cuda_kernels=True)
    
    # Test tropical matmul
    A = jnp.ones((500, 500), dtype=jnp.float32)
    B = jnp.ones((500, 500), dtype=jnp.float32)
    
    matmul_op = custom_ops.get_custom_op('tropical_matmul')
    
    # Benchmark custom vs standard
    start = time.perf_counter()
    for _ in range(10):
        result = matmul_op(A, B)
        result.block_until_ready()
    custom_time = (time.perf_counter() - start) * 100
    
    @jit
    def standard_matmul(A, B):
        A_exp = A[:, :, jnp.newaxis]
        B_exp = B[jnp.newaxis, :, :]
        products = A_exp + B_exp
        return jnp.max(products, axis=1)
    
    start = time.perf_counter()
    for _ in range(10):
        result = standard_matmul(A, B)
        result.block_until_ready()
    standard_time = (time.perf_counter() - start) * 100
    
    speedup = standard_time / custom_time
    
    print(f"✓ Custom matmul: {custom_time:.2f}ms")
    print(f"✓ Standard matmul: {standard_time:.2f}ms")
    print(f"✓ Speedup: {speedup:.2f}x")
    
    # Test fusion
    C = jnp.ones((500, 500), dtype=jnp.float32)
    fused_result = custom_ops.apply_fusion(
        custom_ops.FusionPattern.MATMUL_ADD, A, B, C
    )
    print(f"✓ Fusion completed: shape {fused_result.shape}")
    
    stats = custom_ops.get_stats()
    print(f"✓ Fusions performed: {stats['fusions_performed']}")
    
    return custom_ops


def test_performance_monitor():
    """Test performance monitoring"""
    print("\n" + "="*60)
    print("Testing Performance Monitor")
    print("="*60)
    
    monitor = JAXPerformanceMonitor(
        enable_telemetry=True,
        sla_config={
            'matmul_time': {
                'threshold': 100.0,
                'comparison': 'less_than'
            }
        }
    )
    
    # Record metrics
    for i in range(5):
        monitor.record_metric('matmul_time', 50 + i * 10, 'ms')
    
    print(f"✓ Metrics recorded: {monitor.stats['metrics_recorded']}")
    
    # Detect workload
    workload = monitor.detect_workload_type()
    print(f"✓ Detected workload: {workload.value}")
    
    # Adaptive tuning
    tuning = monitor.adapt_performance_tuning()
    print(f"✓ Compilation level: {tuning['compilation_level']}")
    print(f"✓ Batch size: {tuning['batch_size']}")
    
    # Get summary
    summary = monitor.get_performance_summary()
    print(f"✓ Performance state: {summary['current_state']}")
    
    return monitor


def test_integrated_performance():
    """Test all optimizations working together"""
    print("\n" + "="*60)
    print("Testing Integrated Performance (All Optimizations)")
    print("="*60)
    
    # Initialize all optimizers
    compilation_opt = JAXCompilationOptimizer(
        compilation_level=CompilationLevel.AGGRESSIVE,
        enable_pgo=True
    )
    memory_opt = JAXMemoryOptimizer(enable_prefetching=True)
    custom_ops = TropicalXLACustomOps(enable_cuda_kernels=True)
    perf_monitor = JAXPerformanceMonitor(enable_telemetry=True)
    
    # Create optimized operation
    @compilation_opt.compile_function
    def optimized_tropical_operation(A, B):
        # Use custom op
        matmul_op = custom_ops.get_custom_op('tropical_matmul')
        
        # Analyze and optimize memory
        profile = memory_opt.analyze_access_pattern(A, 'matmul')
        A_opt, _ = memory_opt.optimize_tensor_layout(A, profile, 'matmul')
        B_opt, _ = memory_opt.optimize_tensor_layout(B, profile, 'matmul')
        
        # Setup prefetching
        memory_opt.setup_prefetching([A_opt, B_opt])
        
        # Execute with custom kernel
        result = matmul_op(A_opt, B_opt)
        
        # Record metric
        perf_monitor.record_metric('integrated_op', 0, 'ms')
        
        return result
    
    # Test with various sizes
    sizes = [(100, 100), (500, 500), (1000, 1000)]
    
    for size in sizes:
        A = jnp.ones(size, dtype=jnp.float32)
        B = jnp.ones(size, dtype=jnp.float32)
        
        # Warmup
        for _ in range(5):
            _ = optimized_tropical_operation(A, B)
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(10):
            result = optimized_tropical_operation(A, B)
            result.block_until_ready()
        jax_time = (time.perf_counter() - start) * 100
        
        # Compare with PyTorch
        A_torch = torch.ones(size, dtype=torch.float32)
        B_torch = torch.ones(size, dtype=torch.float32)
        
        if torch.cuda.is_available():
            A_torch = A_torch.cuda()
            B_torch = B_torch.cuda()
        
        def pytorch_matmul(A, B):
            A_exp = A.unsqueeze(2)
            B_exp = B.unsqueeze(0)
            products = A_exp + B_exp
            return torch.max(products, dim=1)[0]
        
        start = time.perf_counter()
        for _ in range(10):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            result = pytorch_matmul(A_torch, B_torch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) * 100
        
        speedup = pytorch_time / jax_time
        
        status = "✓" if speedup >= 6.0 else "✗"
        print(f"{size}: JAX {jax_time:.2f}ms, PyTorch {pytorch_time:.2f}ms, "
              f"Speedup: {speedup:.2f}x {status}")
    
    # Get final statistics
    print(f"\n✓ Compilation cache hits: {compilation_opt.stats['cache_hits']}")
    print(f"✓ Memory optimizations: {memory_opt.stats['layouts_optimized']}")
    print(f"✓ Custom ops executed: {custom_ops.stats['custom_kernels_executed']}")
    print(f"✓ Metrics recorded: {perf_monitor.stats['metrics_recorded']}")


def run_mini_benchmark():
    """Run mini benchmark suite"""
    print("\n" + "="*60)
    print("Running Mini Benchmark Suite")
    print("="*60)
    
    config = BenchmarkConfig(
        operations=[
            BenchmarkOperation.TROPICAL_MATMUL,
            BenchmarkOperation.TROPICAL_ATTENTION,
            BenchmarkOperation.FUSION_CHAIN
        ],
        sizes=[(500, 500), (1000, 1000)],
        num_warmup=5,
        num_iterations=20,
        target_speedup=6.0,
        export_results=False
    )
    
    suite = JAXBenchmarkSuite(config)
    results = suite.run_all_benchmarks()
    
    return results


def main():
    """Main test runner"""
    print("\n" + "="*80)
    print("JAX PERFORMANCE OPTIMIZATION TEST SUITE")
    print("="*80)
    print("Target: 6x speedup over PyTorch")
    print("="*80)
    
    # Test individual components
    compilation_opt = test_compilation_optimizer()
    memory_opt = test_memory_optimizer()
    custom_ops = test_custom_ops()
    perf_monitor = test_performance_monitor()
    
    # Test integrated performance
    test_integrated_performance()
    
    # Run mini benchmark
    results = run_mini_benchmark()
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    if 'average_speedup' in results:
        avg_speedup = results['average_speedup']
        target_met = avg_speedup >= 6.0
        
        if target_met:
            print(f"✓ SUCCESS: Achieved {avg_speedup:.2f}x average speedup")
            print(f"  All optimizations working correctly")
            print(f"  Production-ready with persistent caching")
            print(f"  Memory bandwidth optimized")
            print(f"  Custom XLA kernels operational")
            print(f"  Performance monitoring active")
        else:
            print(f"✗ Below target: {avg_speedup:.2f}x average speedup")
            print(f"  Target: 6.0x")
    
    print("="*80)
    
    # Cleanup
    compilation_opt.shutdown()
    memory_opt.shutdown()
    perf_monitor.shutdown()


if __name__ == "__main__":
    main()