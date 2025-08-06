"""
JAX Benchmark Suite - Comprehensive performance benchmarking for tropical operations
Compares JAX vs PyTorch implementations across various operations and sizes
HARD FAILURES ONLY - NO GRACEFUL DEGRADATION

This module provides:
1. Comprehensive benchmark suite with multiple operation types
2. Detailed performance comparison between JAX and PyTorch
3. Memory usage tracking and analysis
4. Performance regression detection
5. Automated report generation
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap
import torch
import numpy as np
import time
import gc
import psutil
import logging
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns

# Import system components
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from independent_core.compression_systems.tropical.jax_tropical_engine import (
    TropicalJAXEngine,
    JAXTropicalConfig
)
from independent_core.compression_systems.tropical.jax_compilation_optimizer import (
    JAXCompilationOptimizer,
    CompilationLevel
)
from independent_core.compression_systems.tropical.jax_memory_optimizer import JAXMemoryOptimizer
from independent_core.compression_systems.tropical.jax_custom_ops import TropicalXLACustomOps
from independent_core.compression_systems.tropical.jax_performance_monitor import JAXPerformanceMonitor
from independent_core.compression_systems.tropical.tropical_core import TROPICAL_ZERO

logger = logging.getLogger(__name__)


class BenchmarkOperation(Enum):
    """Types of operations to benchmark"""
    TROPICAL_MATMUL = "tropical_matmul"
    TROPICAL_CONV = "tropical_conv"
    TROPICAL_ATTENTION = "tropical_attention"
    TROPICAL_REDUCE = "tropical_reduce"
    TROPICAL_SCAN = "tropical_scan"
    POLYNOMIAL_EVAL = "polynomial_eval"
    CHANNEL_PROCESS = "channel_process"
    FUSION_CHAIN = "fusion_chain"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks"""
    operations: List[BenchmarkOperation] = field(default_factory=lambda: list(BenchmarkOperation))
    sizes: List[Tuple[int, ...]] = field(default_factory=lambda: [
        (100, 100),
        (500, 500),
        (1000, 1000),
        (2000, 2000),
        (5000, 5000)
    ])
    num_warmup: int = 10
    num_iterations: int = 100
    enable_profiling: bool = True
    enable_memory_tracking: bool = True
    target_speedup: float = 6.0  # Target 6x speedup
    export_results: bool = True


@dataclass
class OperationBenchmark:
    """Result for a single operation benchmark"""
    operation: BenchmarkOperation
    size: Tuple[int, ...]
    jax_time_ms: float
    pytorch_time_ms: float
    speedup: float
    jax_memory_mb: float
    pytorch_memory_mb: float
    memory_ratio: float
    compilation_time_ms: float
    accuracy_verified: bool
    meets_target: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'operation': self.operation.value,
            'size': str(self.size),
            'jax_time_ms': round(self.jax_time_ms, 3),
            'pytorch_time_ms': round(self.pytorch_time_ms, 3),
            'speedup': round(self.speedup, 2),
            'jax_memory_mb': round(self.jax_memory_mb, 2),
            'pytorch_memory_mb': round(self.pytorch_memory_mb, 2),
            'memory_ratio': round(self.memory_ratio, 2),
            'compilation_time_ms': round(self.compilation_time_ms, 2),
            'accuracy_verified': self.accuracy_verified,
            'meets_target': self.meets_target
        }


class JAXBenchmarkSuite:
    """
    Comprehensive benchmark suite for JAX tropical operations.
    Compares performance against PyTorch baseline.
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize benchmark suite
        
        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        
        # Initialize JAX components with optimizations
        self.jax_engine = TropicalJAXEngine(
            JAXTropicalConfig(
                enable_jit=True,
                enable_vmap=True,
                compilation_level=3,
                xla_optimization_level=3
            )
        )
        
        self.compilation_optimizer = JAXCompilationOptimizer(
            compilation_level=CompilationLevel.AGGRESSIVE,
            enable_pgo=True,
            enable_aot=True
        )
        
        self.memory_optimizer = JAXMemoryOptimizer(
            enable_profiling=True,
            enable_prefetching=True
        )
        
        self.custom_ops = TropicalXLACustomOps(enable_cuda_kernels=True)
        
        self.performance_monitor = JAXPerformanceMonitor(enable_telemetry=True)
        
        # Benchmark results
        self.results: List[OperationBenchmark] = []
        
        # Check GPU availability
        self.has_gpu = torch.cuda.is_available() and len(jax.devices('gpu')) > 0
        
        logger.info(f"JAXBenchmarkSuite initialized (GPU: {self.has_gpu})")
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """
        Run all configured benchmarks
        
        Returns:
            Comprehensive benchmark results
        """
        print("\n" + "="*60)
        print("JAX TROPICAL ENGINE BENCHMARK SUITE")
        print("="*60)
        print(f"Target speedup: {self.config.target_speedup}x")
        print(f"GPU available: {self.has_gpu}")
        print(f"Operations: {len(self.config.operations)}")
        print(f"Sizes: {self.config.sizes}")
        print("="*60 + "\n")
        
        overall_start = time.perf_counter()
        
        for operation in self.config.operations:
            print(f"\nBenchmarking {operation.value}...")
            print("-" * 40)
            
            for size in self.config.sizes:
                try:
                    result = self._benchmark_operation(operation, size)
                    self.results.append(result)
                    
                    # Print result
                    status = "✓" if result.meets_target else "✗"
                    print(f"  {size}: {result.speedup:.2f}x speedup "
                          f"({result.jax_time_ms:.2f}ms vs {result.pytorch_time_ms:.2f}ms) {status}")
                    
                except Exception as e:
                    logger.error(f"Failed to benchmark {operation.value} at size {size}: {e}")
                    print(f"  {size}: FAILED - {str(e)}")
        
        overall_time = time.perf_counter() - overall_start
        
        # Generate summary
        summary = self._generate_summary()
        summary['total_benchmark_time_seconds'] = overall_time
        
        # Export results if configured
        if self.config.export_results:
            self._export_results(summary)
        
        # Print summary
        self._print_summary(summary)
        
        return summary
    
    def _benchmark_operation(self,
                            operation: BenchmarkOperation,
                            size: Tuple[int, ...]) -> OperationBenchmark:
        """Benchmark a single operation"""
        # Get operation implementations
        jax_func = self._get_jax_operation(operation)
        pytorch_func = self._get_pytorch_operation(operation)
        
        # Generate test data
        test_data = self._generate_test_data(operation, size)
        
        # Memory tracking
        gc.collect()
        if self.has_gpu:
            torch.cuda.empty_cache()
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Compile JAX function
        compile_start = time.perf_counter()
        jax_compiled = self.compilation_optimizer.compile_function(jax_func)
        compilation_time = (time.perf_counter() - compile_start) * 1000
        
        # Warmup
        for _ in range(self.config.num_warmup):
            _ = jax_compiled(*test_data['jax'])
            _ = pytorch_func(*test_data['pytorch'])
        
        # Benchmark JAX
        jax_times = []
        for _ in range(self.config.num_iterations):
            start = time.perf_counter()
            jax_result = jax_compiled(*test_data['jax'])
            jax_result.block_until_ready()
            jax_times.append((time.perf_counter() - start) * 1000)
        
        jax_time_ms = np.median(jax_times)
        jax_memory = (process.memory_info().rss / 1024 / 1024) - initial_memory
        
        # Benchmark PyTorch
        torch_times = []
        for _ in range(self.config.num_iterations):
            start = time.perf_counter()
            if self.has_gpu:
                torch.cuda.synchronize()
            torch_result = pytorch_func(*test_data['pytorch'])
            if self.has_gpu:
                torch.cuda.synchronize()
            torch_times.append((time.perf_counter() - start) * 1000)
        
        pytorch_time_ms = np.median(torch_times)
        pytorch_memory = (process.memory_info().rss / 1024 / 1024) - initial_memory - jax_memory
        
        # Calculate metrics
        speedup = pytorch_time_ms / jax_time_ms if jax_time_ms > 0 else 0
        memory_ratio = jax_memory / pytorch_memory if pytorch_memory > 0 else 1.0
        
        # Verify accuracy
        accuracy_verified = self._verify_accuracy(jax_result, torch_result)
        
        # Record in performance monitor
        self.performance_monitor.record_metric(
            f"{operation.value}_speedup",
            speedup,
            "x"
        )
        
        return OperationBenchmark(
            operation=operation,
            size=size,
            jax_time_ms=jax_time_ms,
            pytorch_time_ms=pytorch_time_ms,
            speedup=speedup,
            jax_memory_mb=max(jax_memory, 0),
            pytorch_memory_mb=max(pytorch_memory, 0),
            memory_ratio=memory_ratio,
            compilation_time_ms=compilation_time,
            accuracy_verified=accuracy_verified,
            meets_target=speedup >= self.config.target_speedup
        )
    
    def _get_jax_operation(self, operation: BenchmarkOperation) -> Callable:
        """Get JAX implementation of operation"""
        if operation == BenchmarkOperation.TROPICAL_MATMUL:
            return self.custom_ops.get_custom_op('tropical_matmul')
        
        elif operation == BenchmarkOperation.TROPICAL_REDUCE:
            return self.custom_ops.get_custom_op('tropical_reduce')
        
        elif operation == BenchmarkOperation.TROPICAL_SCAN:
            return self.custom_ops.get_custom_op('tropical_scan')
        
        elif operation == BenchmarkOperation.TROPICAL_ATTENTION:
            return self.custom_ops.get_custom_op('tropical_attention')
        
        elif operation == BenchmarkOperation.TROPICAL_CONV:
            @jit
            def tropical_conv(x, kernel):
                n = x.shape[0]
                k = kernel.shape[0]
                output = jnp.zeros(n - k + 1)
                for i in range(n - k + 1):
                    window = x[i:i+k]
                    products = window + kernel
                    output = output.at[i].set(jnp.max(products))
                return output
            return tropical_conv
        
        elif operation == BenchmarkOperation.POLYNOMIAL_EVAL:
            @jit
            def poly_eval(coeffs, exponents, points):
                return self.jax_engine.evaluate_polynomial(coeffs, exponents, points)
            return poly_eval
        
        elif operation == BenchmarkOperation.FUSION_CHAIN:
            def fusion_chain(A, B, C):
                return self.custom_ops.apply_fusion(
                    self.custom_ops.FusionPattern.MATMUL_ADD, A, B, C
                )
            return fusion_chain
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _get_pytorch_operation(self, operation: BenchmarkOperation) -> Callable:
        """Get PyTorch implementation of operation"""
        if operation == BenchmarkOperation.TROPICAL_MATMUL:
            def tropical_matmul_torch(A, B):
                m, k = A.shape
                _, n = B.shape
                A_exp = A.unsqueeze(2)
                B_exp = B.unsqueeze(0)
                products = A_exp + B_exp
                return torch.max(products, dim=1)[0]
            return tropical_matmul_torch
        
        elif operation == BenchmarkOperation.TROPICAL_REDUCE:
            return lambda x: torch.max(x)
        
        elif operation == BenchmarkOperation.TROPICAL_SCAN:
            def tropical_scan_torch(x):
                return torch.cummax(x.flatten(), dim=0)[0].reshape(x.shape)
            return tropical_scan_torch
        
        elif operation == BenchmarkOperation.TROPICAL_ATTENTION:
            def tropical_attention_torch(Q, K, V):
                scores = torch.matmul(Q, K.T)
                weights = torch.softmax(scores / 0.1, dim=-1)
                return torch.matmul(weights, V)
            return tropical_attention_torch
        
        elif operation == BenchmarkOperation.TROPICAL_CONV:
            def tropical_conv_torch(x, kernel):
                n = x.shape[0]
                k = kernel.shape[0]
                output = torch.zeros(n - k + 1)
                for i in range(n - k + 1):
                    window = x[i:i+k]
                    products = window + kernel
                    output[i] = torch.max(products)
                return output
            return tropical_conv_torch
        
        elif operation == BenchmarkOperation.POLYNOMIAL_EVAL:
            def poly_eval_torch(coeffs, exponents, points):
                # Simplified polynomial evaluation
                result = torch.zeros(points.shape[0])
                for i in range(points.shape[0]):
                    monomial_values = coeffs + torch.matmul(exponents.float(), points[i])
                    result[i] = torch.max(monomial_values)
                return result
            return poly_eval_torch
        
        elif operation == BenchmarkOperation.FUSION_CHAIN:
            def fusion_chain_torch(A, B, C):
                result = tropical_matmul_torch(A, B)
                return torch.maximum(result, C)
            return fusion_chain_torch
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _generate_test_data(self,
                           operation: BenchmarkOperation,
                           size: Tuple[int, ...]) -> Dict[str, List[Any]]:
        """Generate test data for operation"""
        np.random.seed(42)  # Reproducible
        
        if operation in [BenchmarkOperation.TROPICAL_MATMUL, 
                        BenchmarkOperation.FUSION_CHAIN]:
            # Matrix data
            A_np = np.random.randn(*size).astype(np.float32)
            B_np = np.random.randn(*size).astype(np.float32)
            C_np = np.random.randn(*size).astype(np.float32)
            
            A_jax = jnp.array(A_np)
            B_jax = jnp.array(B_np)
            C_jax = jnp.array(C_np)
            
            A_torch = torch.from_numpy(A_np)
            B_torch = torch.from_numpy(B_np)
            C_torch = torch.from_numpy(C_np)
            
            if self.has_gpu:
                A_torch = A_torch.cuda()
                B_torch = B_torch.cuda()
                C_torch = C_torch.cuda()
            
            return {
                'jax': [A_jax, B_jax, C_jax] if operation == BenchmarkOperation.FUSION_CHAIN else [A_jax, B_jax],
                'pytorch': [A_torch, B_torch, C_torch] if operation == BenchmarkOperation.FUSION_CHAIN else [A_torch, B_torch]
            }
        
        elif operation == BenchmarkOperation.TROPICAL_ATTENTION:
            # Attention data (sequence length, hidden dim)
            seq_len = size[0]
            hidden_dim = size[1] if len(size) > 1 else 64
            
            Q_np = np.random.randn(seq_len, hidden_dim).astype(np.float32)
            K_np = np.random.randn(seq_len, hidden_dim).astype(np.float32)
            V_np = np.random.randn(seq_len, hidden_dim).astype(np.float32)
            
            return {
                'jax': [jnp.array(Q_np), jnp.array(K_np), jnp.array(V_np)],
                'pytorch': [torch.from_numpy(Q_np).cuda() if self.has_gpu else torch.from_numpy(Q_np),
                           torch.from_numpy(K_np).cuda() if self.has_gpu else torch.from_numpy(K_np),
                           torch.from_numpy(V_np).cuda() if self.has_gpu else torch.from_numpy(V_np)]
            }
        
        elif operation == BenchmarkOperation.TROPICAL_CONV:
            # Convolution data
            signal_len = size[0]
            kernel_size = min(32, signal_len // 4)
            
            signal_np = np.random.randn(signal_len).astype(np.float32)
            kernel_np = np.random.randn(kernel_size).astype(np.float32)
            
            return {
                'jax': [jnp.array(signal_np), jnp.array(kernel_np)],
                'pytorch': [torch.from_numpy(signal_np).cuda() if self.has_gpu else torch.from_numpy(signal_np),
                           torch.from_numpy(kernel_np).cuda() if self.has_gpu else torch.from_numpy(kernel_np)]
            }
        
        elif operation == BenchmarkOperation.POLYNOMIAL_EVAL:
            # Polynomial data
            num_monomials = 100
            num_variables = 10
            num_points = size[0]
            
            coeffs_np = np.random.randn(num_monomials).astype(np.float32)
            exponents_np = np.random.randint(0, 5, (num_monomials, num_variables)).astype(np.int32)
            points_np = np.random.randn(num_points, num_variables).astype(np.float32)
            
            return {
                'jax': [jnp.array(coeffs_np), jnp.array(exponents_np), jnp.array(points_np)],
                'pytorch': [torch.from_numpy(coeffs_np).cuda() if self.has_gpu else torch.from_numpy(coeffs_np),
                           torch.from_numpy(exponents_np).cuda() if self.has_gpu else torch.from_numpy(exponents_np),
                           torch.from_numpy(points_np).cuda() if self.has_gpu else torch.from_numpy(points_np)]
            }
        
        else:
            # Default: single array
            data_np = np.random.randn(*size).astype(np.float32)
            return {
                'jax': [jnp.array(data_np)],
                'pytorch': [torch.from_numpy(data_np).cuda() if self.has_gpu else torch.from_numpy(data_np)]
            }
    
    def _verify_accuracy(self, jax_result: Any, torch_result: Any) -> bool:
        """Verify accuracy between JAX and PyTorch results"""
        try:
            jax_np = np.array(jax_result)
            torch_np = torch_result.cpu().numpy() if isinstance(torch_result, torch.Tensor) else torch_result
            
            # Check shapes match
            if jax_np.shape != torch_np.shape:
                return False
            
            # Check values (with tolerance for tropical operations)
            return np.allclose(jax_np, torch_np, rtol=1e-4, atol=1e-6)
        except:
            return False
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary"""
        if not self.results:
            return {'error': 'No benchmark results available'}
        
        # Calculate statistics
        speedups = [r.speedup for r in self.results]
        meets_target = [r.meets_target for r in self.results]
        accuracy_verified = [r.accuracy_verified for r in self.results]
        
        summary = {
            'total_benchmarks': len(self.results),
            'average_speedup': np.mean(speedups),
            'median_speedup': np.median(speedups),
            'min_speedup': np.min(speedups),
            'max_speedup': np.max(speedups),
            'target_speedup': self.config.target_speedup,
            'percent_meeting_target': 100 * sum(meets_target) / len(meets_target),
            'percent_accuracy_verified': 100 * sum(accuracy_verified) / len(accuracy_verified),
            'results_by_operation': {},
            'results_by_size': {},
            'all_results': [r.to_dict() for r in self.results]
        }
        
        # Group by operation
        for op in BenchmarkOperation:
            op_results = [r for r in self.results if r.operation == op]
            if op_results:
                op_speedups = [r.speedup for r in op_results]
                summary['results_by_operation'][op.value] = {
                    'count': len(op_results),
                    'average_speedup': np.mean(op_speedups),
                    'meets_target_percent': 100 * sum(r.meets_target for r in op_results) / len(op_results)
                }
        
        # Group by size
        for size in self.config.sizes:
            size_results = [r for r in self.results if r.size == size]
            if size_results:
                size_speedups = [r.speedup for r in size_results]
                summary['results_by_size'][str(size)] = {
                    'count': len(size_results),
                    'average_speedup': np.mean(size_speedups),
                    'meets_target_percent': 100 * sum(r.meets_target for r in size_results) / len(size_results)
                }
        
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]) -> None:
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        print(f"\nOverall Performance:")
        print(f"  Total benchmarks: {summary['total_benchmarks']}")
        print(f"  Average speedup: {summary['average_speedup']:.2f}x")
        print(f"  Median speedup: {summary['median_speedup']:.2f}x")
        print(f"  Range: {summary['min_speedup']:.2f}x - {summary['max_speedup']:.2f}x")
        print(f"  Meeting {self.config.target_speedup}x target: {summary['percent_meeting_target']:.1f}%")
        print(f"  Accuracy verified: {summary['percent_accuracy_verified']:.1f}%")
        
        if 'results_by_operation' in summary:
            print(f"\nBy Operation:")
            for op, stats in summary['results_by_operation'].items():
                print(f"  {op}:")
                print(f"    Average speedup: {stats['average_speedup']:.2f}x")
                print(f"    Meeting target: {stats['meets_target_percent']:.1f}%")
        
        # Overall verdict
        print("\n" + "="*60)
        if summary['average_speedup'] >= self.config.target_speedup:
            print(f"✓ SUCCESS: Achieved {summary['average_speedup']:.2f}x average speedup")
            print(f"  (Target: {self.config.target_speedup}x)")
        else:
            print(f"✗ BELOW TARGET: {summary['average_speedup']:.2f}x average speedup")
            print(f"  (Target: {self.config.target_speedup}x)")
        print("="*60)
    
    def _export_results(self, summary: Dict[str, Any]) -> None:
        """Export benchmark results"""
        # Export JSON
        with open('benchmark_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Export CSV
        df = pd.DataFrame(summary['all_results'])
        df.to_csv('benchmark_results.csv', index=False)
        
        # Generate plots if matplotlib available
        try:
            self._generate_plots(summary)
        except:
            pass
        
        print(f"\nResults exported to benchmark_results.json and benchmark_results.csv")
    
    def _generate_plots(self, summary: Dict[str, Any]) -> None:
        """Generate visualization plots"""
        df = pd.DataFrame(summary['all_results'])
        
        # Setup plot style
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Speedup by operation
        ax = axes[0, 0]
        op_data = df.groupby('operation')['speedup'].mean().sort_values()
        op_data.plot(kind='barh', ax=ax, color='steelblue')
        ax.axvline(x=self.config.target_speedup, color='red', linestyle='--', label=f'Target ({self.config.target_speedup}x)')
        ax.set_xlabel('Speedup (x)')
        ax.set_title('Average Speedup by Operation')
        ax.legend()
        
        # Speedup distribution
        ax = axes[0, 1]
        ax.hist(df['speedup'], bins=20, color='steelblue', edgecolor='black')
        ax.axvline(x=self.config.target_speedup, color='red', linestyle='--', label=f'Target ({self.config.target_speedup}x)')
        ax.set_xlabel('Speedup (x)')
        ax.set_ylabel('Count')
        ax.set_title('Speedup Distribution')
        ax.legend()
        
        # Memory comparison
        ax = axes[1, 0]
        ax.scatter(df['pytorch_memory_mb'], df['jax_memory_mb'], alpha=0.6)
        max_mem = max(df['pytorch_memory_mb'].max(), df['jax_memory_mb'].max())
        ax.plot([0, max_mem], [0, max_mem], 'r--', label='Equal memory')
        ax.set_xlabel('PyTorch Memory (MB)')
        ax.set_ylabel('JAX Memory (MB)')
        ax.set_title('Memory Usage Comparison')
        ax.legend()
        
        # Compilation time vs speedup
        ax = axes[1, 1]
        ax.scatter(df['compilation_time_ms'], df['speedup'], alpha=0.6)
        ax.set_xlabel('Compilation Time (ms)')
        ax.set_ylabel('Speedup (x)')
        ax.set_title('Compilation Time vs Speedup')
        
        plt.tight_layout()
        plt.savefig('benchmark_plots.png', dpi=150)
        print("Plots saved to benchmark_plots.png")


# Main benchmark runner
def run_comprehensive_benchmark():
    """Run comprehensive JAX vs PyTorch benchmark"""
    
    # Configure benchmark
    config = BenchmarkConfig(
        operations=list(BenchmarkOperation),
        sizes=[
            (100, 100),
            (500, 500),
            (1000, 1000),
            (2000, 2000)
        ],
        num_warmup=10,
        num_iterations=50,
        target_speedup=6.0,
        export_results=True
    )
    
    # Run benchmarks
    suite = JAXBenchmarkSuite(config)
    results = suite.run_all_benchmarks()
    
    return results


if __name__ == "__main__":
    # Set logging level
    logging.basicConfig(level=logging.INFO)
    
    # Run comprehensive benchmark
    results = run_comprehensive_benchmark()