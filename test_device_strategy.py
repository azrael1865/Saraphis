#!/usr/bin/env python3
"""
Comprehensive test script for P-adic Compression Pipeline with Device Strategy
Tests CPU compression → GPU decompression with hard failures and detailed metrics
NO FALLBACKS - FAIL FAST AND LOUD
"""

import torch
import torch.nn as nn
import numpy as np
import time
import gc
import psutil
import traceback
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from tabulate import tabulate
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from independent_core.compression_systems.padic.padic_compressor import (
    PadicCompressionSystem,
    CompressionConfig,
    CompressionResult,
    DecompressionResult
)


@dataclass
class TestCase:
    """Test case configuration"""
    name: str
    shape: Tuple[int, ...]
    data_generator: str  # 'zeros', 'uniform', 'normal', 'sparse', 'dense', 'structured'
    sparsity: float = 0.0  # For sparse patterns
    seed: int = 42


@dataclass
class TestMetrics:
    """Metrics collected during testing"""
    # Performance metrics
    compression_time: float
    decompression_time: float
    stage_times: Dict[str, float]
    
    # Efficiency metrics
    compression_ratio: float
    memory_used_cpu: int
    memory_used_gpu: int
    throughput_mbps: float
    
    # Accuracy metrics
    reconstruction_error_mse: float
    reconstruction_error_max: float
    shape_preserved: bool
    
    # System metrics
    tensor_size_bytes: int
    compressed_size_bytes: int
    element_count: int


class DeviceStrategyTester:
    """Comprehensive tester for p-adic compression device strategy"""
    
    def __init__(self, config: CompressionConfig):
        """Initialize tester with compression configuration"""
        self.config = config
        self.system = PadicCompressionSystem(config)
        self.results: List[Tuple[TestCase, TestMetrics, Optional[Exception]]] = []
        
        # System info
        self.cuda_available = torch.cuda.is_available()
        self.cpu_count = psutil.cpu_count()
        self.total_memory = psutil.virtual_memory().total
        
        if self.cuda_available:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory
        else:
            self.gpu_name = "N/A"
            self.gpu_memory = 0
            
        print(f"\n{'='*80}")
        print("P-ADIC COMPRESSION DEVICE STRATEGY TEST")
        print(f"{'='*80}")
        print(f"System Information:")
        print(f"  CPU Cores: {self.cpu_count}")
        print(f"  System Memory: {self.total_memory / (1024**3):.2f} GB")
        print(f"  CUDA Available: {self.cuda_available}")
        if self.cuda_available:
            print(f"  GPU: {self.gpu_name}")
            print(f"  GPU Memory: {self.gpu_memory / (1024**3):.2f} GB")
        print(f"\nDevice Strategy:")
        print(f"  GPU Enabled: {self.config.enable_gpu}")
        print(f"  System Device: {self.system.device}")
        print(f"  Hard Failure Mode: {self.config.raise_on_error}")
        print(f"{'='*80}\n")
    
    def generate_test_data(self, test_case: TestCase) -> torch.Tensor:
        """Generate test data based on pattern type"""
        torch.manual_seed(test_case.seed)
        np.random.seed(test_case.seed)
        
        if test_case.data_generator == 'zeros':
            return torch.zeros(test_case.shape, dtype=torch.float32)
            
        elif test_case.data_generator == 'uniform':
            return torch.rand(test_case.shape, dtype=torch.float32)
            
        elif test_case.data_generator == 'normal':
            return torch.randn(test_case.shape, dtype=torch.float32)
            
        elif test_case.data_generator == 'sparse':
            # Create sparse tensor with specified sparsity
            tensor = torch.randn(test_case.shape, dtype=torch.float32)
            mask = torch.rand(test_case.shape) > test_case.sparsity
            return tensor * mask
            
        elif test_case.data_generator == 'dense':
            # High-entropy dense data
            return torch.randn(test_case.shape, dtype=torch.float32) * 10 + 5
            
        elif test_case.data_generator == 'structured':
            # Structured patterns (sine waves, etc.)
            x = torch.linspace(0, 10 * np.pi, test_case.shape[0])
            y = torch.linspace(0, 10 * np.pi, test_case.shape[1] if len(test_case.shape) > 1 else 1)
            if len(test_case.shape) == 2:
                X, Y = torch.meshgrid(x, y, indexing='ij')
                return torch.sin(X) * torch.cos(Y)
            else:
                return torch.sin(x).unsqueeze(1).expand(test_case.shape)
                
        else:
            raise ValueError(f"Unknown data generator: {test_case.data_generator}")
    
    def measure_memory(self) -> Tuple[int, int]:
        """Measure current CPU and GPU memory usage"""
        cpu_memory = psutil.Process().memory_info().rss
        
        if self.cuda_available:
            gpu_memory = torch.cuda.memory_allocated()
        else:
            gpu_memory = 0
            
        return cpu_memory, gpu_memory
    
    def run_test_case(self, test_case: TestCase) -> Tuple[TestCase, TestMetrics, Optional[Exception]]:
        """Run a single test case and collect metrics"""
        print(f"\n{'='*60}")
        print(f"Test Case: {test_case.name}")
        print(f"  Shape: {test_case.shape}")
        print(f"  Pattern: {test_case.data_generator}")
        if test_case.sparsity > 0:
            print(f"  Sparsity: {test_case.sparsity:.1%}")
        
        try:
            # Clear GPU cache before test
            if self.cuda_available:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            # Generate test data
            print("\n[1/5] Generating test data...")
            tensor = self.generate_test_data(test_case)
            element_count = tensor.numel()
            tensor_size_bytes = element_count * tensor.element_size()
            print(f"  ✓ Generated tensor: {element_count:,} elements, {tensor_size_bytes / (1024**2):.2f} MB")
            
            # Generate importance scores (simulate gradients/importance)
            importance = torch.abs(tensor) + 0.1
            
            # Measure initial memory
            cpu_mem_start, gpu_mem_start = self.measure_memory()
            
            # COMPRESSION PHASE
            print("\n[2/5] Running compression (CPU)...")
            compression_start = time.perf_counter()
            
            result: CompressionResult = self.system.compress(tensor, importance)
            
            compression_time = time.perf_counter() - compression_start
            compressed_size_bytes = len(result.compressed_data)
            
            # Extract stage times
            stage_times = {stage: metrics['time'] for stage, metrics in result.stage_metrics.items()}
            
            print(f"  ✓ Compression complete in {compression_time:.3f}s")
            print(f"  ✓ Compressed size: {compressed_size_bytes / 1024:.2f} KB")
            print(f"  ✓ Compression ratio: {result.compression_ratio:.2f}x")
            
            # Print stage breakdown
            print("\n  Stage Breakdown:")
            for stage, time_taken in stage_times.items():
                percentage = (time_taken / compression_time) * 100
                print(f"    - {stage}: {time_taken:.3f}s ({percentage:.1f}%)")
            
            # DECOMPRESSION PHASE
            print("\n[3/5] Running decompression (GPU)...")
            decompression_start = time.perf_counter()
            
            decompressed: DecompressionResult = self.system.decompress(result.compressed_data)
            
            decompression_time = time.perf_counter() - decompression_start
            
            print(f"  ✓ Decompression complete in {decompression_time:.3f}s")
            
            # Measure peak memory
            cpu_mem_peak, gpu_mem_peak = self.measure_memory()
            memory_used_cpu = cpu_mem_peak - cpu_mem_start
            memory_used_gpu = gpu_mem_peak - gpu_mem_start
            
            # VALIDATION PHASE
            print("\n[4/5] Validating reconstruction...")
            
            # Check shape preservation
            shape_preserved = (decompressed.reconstructed_tensor.shape == tensor.shape)
            if shape_preserved:
                print(f"  ✓ Shape preserved: {tensor.shape}")
            else:
                print(f"  ✗ SHAPE MISMATCH: Expected {tensor.shape}, got {decompressed.reconstructed_tensor.shape}")
                raise ValueError(f"Shape mismatch: {tensor.shape} != {decompressed.reconstructed_tensor.shape}")
            
            # Calculate reconstruction errors
            mse = torch.nn.functional.mse_loss(tensor, decompressed.reconstructed_tensor).item()
            max_error = torch.max(torch.abs(tensor - decompressed.reconstructed_tensor)).item()
            
            print(f"  ✓ Reconstruction MSE: {mse:.2e}")
            print(f"  ✓ Max absolute error: {max_error:.2e}")
            
            # Check for NaN/Inf
            if torch.isnan(decompressed.reconstructed_tensor).any():
                raise ValueError("Reconstruction contains NaN values")
            if torch.isinf(decompressed.reconstructed_tensor).any():
                raise ValueError("Reconstruction contains Inf values")
            
            # Calculate throughput
            total_time = compression_time + decompression_time
            throughput_mbps = (tensor_size_bytes / (1024**2)) / total_time
            
            # METRICS COLLECTION
            print("\n[5/5] Collecting metrics...")
            metrics = TestMetrics(
                compression_time=compression_time,
                decompression_time=decompression_time,
                stage_times=stage_times,
                compression_ratio=result.compression_ratio,
                memory_used_cpu=memory_used_cpu,
                memory_used_gpu=memory_used_gpu,
                throughput_mbps=throughput_mbps,
                reconstruction_error_mse=mse,
                reconstruction_error_max=max_error,
                shape_preserved=shape_preserved,
                tensor_size_bytes=tensor_size_bytes,
                compressed_size_bytes=compressed_size_bytes,
                element_count=element_count
            )
            
            print(f"  ✓ Test completed successfully")
            return test_case, metrics, None
            
        except Exception as e:
            print(f"\n  ✗ TEST FAILED: {str(e)}")
            print(f"\nFull traceback:")
            traceback.print_exc()
            
            # Create empty metrics for failed test
            metrics = TestMetrics(
                compression_time=0,
                decompression_time=0,
                stage_times={},
                compression_ratio=0,
                memory_used_cpu=0,
                memory_used_gpu=0,
                throughput_mbps=0,
                reconstruction_error_mse=float('inf'),
                reconstruction_error_max=float('inf'),
                shape_preserved=False,
                tensor_size_bytes=0,
                compressed_size_bytes=0,
                element_count=0
            )
            
            return test_case, metrics, e
    
    def run_all_tests(self, test_cases: List[TestCase]):
        """Run all test cases and collect results"""
        print(f"\nRunning {len(test_cases)} test cases...")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[Test {i}/{len(test_cases)}]")
            result = self.run_test_case(test_case)
            self.results.append(result)
            
            # Clear caches between tests
            if self.cuda_available:
                torch.cuda.empty_cache()
            gc.collect()
    
    def print_summary(self):
        """Print comprehensive summary of test results"""
        print(f"\n{'='*80}")
        print("TEST SUMMARY")
        print(f"{'='*80}\n")
        
        # Overall statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for _, _, error in self.results if error is None)
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        
        # Performance table
        print(f"\n{'='*80}")
        print("PERFORMANCE METRICS")
        print(f"{'='*80}\n")
        
        perf_data = []
        for test_case, metrics, error in self.results:
            if error is None:
                perf_data.append([
                    test_case.name,
                    f"{test_case.shape}",
                    f"{metrics.compression_time:.3f}s",
                    f"{metrics.decompression_time:.3f}s",
                    f"{metrics.compression_time + metrics.decompression_time:.3f}s",
                    f"{metrics.throughput_mbps:.1f} MB/s"
                ])
            else:
                perf_data.append([
                    test_case.name,
                    f"{test_case.shape}",
                    "FAILED",
                    "FAILED",
                    "FAILED",
                    "FAILED"
                ])
        
        print(tabulate(perf_data, 
                      headers=["Test Case", "Shape", "Compression", "Decompression", "Total Time", "Throughput"],
                      tablefmt="grid"))
        
        # Compression efficiency table
        print(f"\n{'='*80}")
        print("COMPRESSION EFFICIENCY")
        print(f"{'='*80}\n")
        
        comp_data = []
        for test_case, metrics, error in self.results:
            if error is None:
                comp_data.append([
                    test_case.name,
                    f"{metrics.tensor_size_bytes / (1024**2):.2f} MB",
                    f"{metrics.compressed_size_bytes / 1024:.2f} KB",
                    f"{metrics.compression_ratio:.2f}x",
                    f"{metrics.memory_used_cpu / (1024**2):.1f} MB",
                    f"{metrics.memory_used_gpu / (1024**2):.1f} MB"
                ])
            else:
                comp_data.append([
                    test_case.name,
                    "N/A",
                    "N/A",
                    "FAILED",
                    "N/A",
                    "N/A"
                ])
        
        print(tabulate(comp_data,
                      headers=["Test Case", "Original Size", "Compressed Size", "Ratio", "CPU Mem", "GPU Mem"],
                      tablefmt="grid"))
        
        # Accuracy table
        print(f"\n{'='*80}")
        print("RECONSTRUCTION ACCURACY")
        print(f"{'='*80}\n")
        
        acc_data = []
        for test_case, metrics, error in self.results:
            if error is None:
                acc_data.append([
                    test_case.name,
                    f"{metrics.reconstruction_error_mse:.2e}",
                    f"{metrics.reconstruction_error_max:.2e}",
                    "✓" if metrics.shape_preserved else "✗",
                    "PASS" if metrics.reconstruction_error_mse < 1e-5 else "WARN"
                ])
            else:
                acc_data.append([
                    test_case.name,
                    "N/A",
                    "N/A",
                    "✗",
                    "FAILED"
                ])
        
        print(tabulate(acc_data,
                      headers=["Test Case", "MSE", "Max Error", "Shape OK", "Status"],
                      tablefmt="grid"))
        
        # Stage timing analysis (for successful tests)
        successful_results = [(tc, m) for tc, m, e in self.results if e is None]
        if successful_results:
            print(f"\n{'='*80}")
            print("STAGE TIMING ANALYSIS (Average)")
            print(f"{'='*80}\n")
            
            # Aggregate stage times
            all_stages = set()
            for _, metrics in successful_results:
                all_stages.update(metrics.stage_times.keys())
            
            stage_averages = {}
            for stage in all_stages:
                times = [m.stage_times.get(stage, 0) for _, m in successful_results]
                stage_averages[stage] = sum(times) / len(times)
            
            # Sort by average time
            sorted_stages = sorted(stage_averages.items(), key=lambda x: x[1], reverse=True)
            
            stage_data = []
            total_stage_time = sum(stage_averages.values())
            for stage, avg_time in sorted_stages:
                percentage = (avg_time / total_stage_time) * 100
                stage_data.append([stage, f"{avg_time:.3f}s", f"{percentage:.1f}%"])
            
            print(tabulate(stage_data,
                          headers=["Stage", "Avg Time", "% of Total"],
                          tablefmt="grid"))
        
        # Error summary
        if failed_tests > 0:
            print(f"\n{'='*80}")
            print("ERROR SUMMARY")
            print(f"{'='*80}\n")
            
            for test_case, _, error in self.results:
                if error is not None:
                    print(f"Test: {test_case.name}")
                    print(f"  Shape: {test_case.shape}")
                    print(f"  Error: {str(error)}")
                    print()
        
        # Recommendations
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS")
        print(f"{'='*80}\n")
        
        if passed_tests == total_tests:
            print("✓ All tests passed successfully!")
            print("✓ Device strategy (CPU compression → GPU decompression) is working correctly")
            print("✓ Shape preservation is maintained throughout the pipeline")
            
            # Performance recommendations
            avg_compression_ratio = sum(m.compression_ratio for _, m, e in self.results if e is None) / passed_tests
            print(f"\nPerformance Analysis:")
            print(f"  - Average compression ratio: {avg_compression_ratio:.2f}x")
            
            if avg_compression_ratio < 5:
                print("  - Consider adjusting sparsity threshold for better compression")
            
            # Find bottleneck stage
            if successful_results:
                avg_stage_times = {}
                for stage in all_stages:
                    times = [m.stage_times.get(stage, 0) for _, m in successful_results]
                    avg_stage_times[stage] = sum(times) / len(times)
                
                bottleneck = max(avg_stage_times.items(), key=lambda x: x[1])
                print(f"  - Bottleneck stage: {bottleneck[0]} ({bottleneck[1]:.3f}s average)")
                
                if bottleneck[0] == 'pattern_detection':
                    print("    → Consider optimizing pattern detection algorithm")
                elif bottleneck[0] == 'entropy_coding':
                    print("    → Consider parallel entropy coding for large tensors")
                elif bottleneck[0] == 'sparse_encoding':
                    print("    → Consider GPU-accelerated sparse operations")
        
        else:
            print(f"⚠ {failed_tests} tests failed!")
            print("Debug recommendations:")
            print("  1. Check CUDA availability and GPU memory")
            print("  2. Verify shape metadata preservation in compression pipeline")
            print("  3. Review error logs for specific failure points")
            print("  4. Consider reducing tensor sizes for memory-constrained systems")


def create_test_cases() -> List[TestCase]:
    """Create comprehensive test cases"""
    test_cases = []
    
    # Small tensors
    test_cases.extend([
        TestCase("Small Zeros", (100, 100), "zeros"),
        TestCase("Small Uniform", (100, 100), "uniform"),
        TestCase("Small Normal", (100, 100), "normal"),
        TestCase("Small Sparse", (100, 100), "sparse", sparsity=0.9),
        TestCase("Small Dense", (100, 100), "dense"),
        TestCase("Small Structured", (100, 100), "structured"),
    ])
    
    # Medium tensors
    test_cases.extend([
        TestCase("Medium Zeros", (500, 500), "zeros"),
        TestCase("Medium Uniform", (500, 500), "uniform"),
        TestCase("Medium Normal", (500, 500), "normal"),
        TestCase("Medium Sparse", (500, 500), "sparse", sparsity=0.9),
        TestCase("Medium Dense", (500, 500), "dense"),
        TestCase("Medium Structured", (500, 500), "structured"),
    ])
    
    # Large tensors
    test_cases.extend([
        TestCase("Large Zeros", (1000, 1000), "zeros"),
        TestCase("Large Uniform", (1000, 1000), "uniform"),
        TestCase("Large Normal", (1000, 1000), "normal"),
        TestCase("Large Sparse", (1000, 1000), "sparse", sparsity=0.95),
        TestCase("Large Dense", (1000, 1000), "dense"),
        TestCase("Large Structured", (1000, 1000), "structured"),
    ])
    
    # Edge cases
    test_cases.extend([
        TestCase("Tiny", (1, 1), "normal"),
        TestCase("Single Row", (1, 1000), "normal"),
        TestCase("Single Column", (1000, 1), "normal"),
        TestCase("Very Sparse", (500, 500), "sparse", sparsity=0.99),
        TestCase("High Variance", (300, 300), "normal", seed=123),
    ])
    
    # Extra large (optional - may cause memory issues)
    if "--include-xlarge" in sys.argv:
        test_cases.extend([
            TestCase("XLarge Normal", (2048, 2048), "normal"),
            TestCase("XLarge Sparse", (2048, 2048), "sparse", sparsity=0.98),
        ])
    
    return test_cases


def main():
    """Main test execution"""
    # Create configuration with hard failure mode
    config = CompressionConfig(
        prime=257,
        base_precision=4,
        min_precision=2,
        max_precision=4,
        target_error=1e-6,
        enable_gpu=torch.cuda.is_available(),
        validate_reconstruction=True,
        raise_on_error=True,  # HARD FAILURES
        enable_logging=True
    )
    
    # Create tester
    tester = DeviceStrategyTester(config)
    
    # Create test cases
    test_cases = create_test_cases()
    
    # Run tests
    start_time = time.perf_counter()
    tester.run_all_tests(test_cases)
    total_time = time.perf_counter() - start_time
    
    # Print summary
    tester.print_summary()
    
    print(f"\n{'='*80}")
    print(f"Total test time: {total_time:.2f}s")
    print(f"{'='*80}\n")
    
    # Return exit code based on results
    failed_tests = sum(1 for _, _, error in tester.results if error is not None)
    return 1 if failed_tests > 0 else 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)