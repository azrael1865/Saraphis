#!/usr/bin/env python3
"""
Comprehensive Compression System Test with Full Terminal Output
Tracks and displays ALL compression/decompression metrics
"""

import torch
import time
import numpy as np
from typing import Dict, Any, List
import sys
import os

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from independent_core.compression_systems.padic.padic_compressor import (
    PadicCompressionSystem,
    CompressionConfig
)


def print_separator(title: str = "", width: int = 100):
    """Print a formatted separator"""
    if title:
        padding = (width - len(title) - 2) // 2
        print("\n" + "=" * padding + f" {title} " + "=" * padding)
    else:
        print("=" * width)


def format_bytes(bytes_val: int) -> str:
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def format_time(seconds: float) -> str:
    """Format time to human readable string"""
    if seconds < 1e-3:
        return f"{seconds*1e6:.2f} μs"
    elif seconds < 1:
        return f"{seconds*1e3:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def run_comprehensive_test(tensor_name: str, tensor: torch.Tensor, 
                          system: PadicCompressionSystem,
                          importance_scores: torch.Tensor = None) -> Dict[str, Any]:
    """Run a single comprehensive test with full output"""
    
    print_separator(f"TESTING: {tensor_name}")
    
    # Input tensor stats
    original_bytes = tensor.numel() * tensor.element_size()
    print(f"\n📊 INPUT TENSOR STATISTICS:")
    print(f"  Shape: {list(tensor.shape)}")
    print(f"  Data type: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Total elements: {tensor.numel():,}")
    print(f"  Original size: {format_bytes(original_bytes)}")
    print(f"  Min value: {tensor.min().item():.6f}")
    print(f"  Max value: {tensor.max().item():.6f}")
    print(f"  Mean value: {tensor.mean().item():.6f}")
    print(f"  Std deviation: {tensor.std().item():.6f}")
    print(f"  Sparsity: {(tensor == 0).sum().item() / tensor.numel():.2%}")
    
    # ==================== COMPRESSION ====================
    print(f"\n🗜️  COMPRESSION PHASE:")
    print(f"  Starting compression...")
    
    compression_start = time.perf_counter()
    try:
        result = system.compress(tensor, importance_scores)
        compression_end = time.perf_counter()
        compression_time = compression_end - compression_start
        
        print(f"  ✅ Compression completed in {format_time(compression_time)}")
        
        # Compression metrics
        compressed_bytes = len(result.compressed_data)
        compression_ratio = original_bytes / compressed_bytes
        space_saved = original_bytes - compressed_bytes
        space_saved_pct = (space_saved / original_bytes) * 100
        
        print(f"\n📈 COMPRESSION METRICS:")
        print(f"  Compressed size: {format_bytes(compressed_bytes)}")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Space saved: {format_bytes(space_saved)} ({space_saved_pct:.1f}%)")
        print(f"  Compression speed: {format_bytes(int(original_bytes / compression_time))}/s")
        print(f"  Throughput: {tensor.numel() / compression_time / 1e6:.2f} M elements/s")
        print(f"  Memory usage: {format_bytes(result.memory_usage)}")
        print(f"  Validation: {'✅ PASSED' if result.validation_passed else '❌ FAILED'}")
        
        # Stage-by-stage breakdown
        print(f"\n⏱️  COMPRESSION STAGE BREAKDOWN:")
        total_stage_time = 0
        for stage, metrics in result.stage_metrics.items():
            stage_time = metrics.get('time', 0)
            total_stage_time += stage_time
            stage_pct = (stage_time / compression_time) * 100
            print(f"  {stage}:")
            print(f"    Time: {format_time(stage_time)} ({stage_pct:.1f}% of total)")
            
            # Additional stage-specific metrics
            if 'compression_ratio' in metrics:
                print(f"    Compression ratio: {metrics['compression_ratio']:.2f}x")
            if 'patterns_found' in metrics:
                print(f"    Patterns found: {metrics['patterns_found']}")
            if 'sparsity' in metrics:
                print(f"    Sparsity achieved: {metrics['sparsity']:.2%}")
            if 'average_precision' in metrics:
                print(f"    Average precision: {metrics['average_precision']:.2f}")
            if 'method' in metrics:
                print(f"    Encoding method: {metrics['method']}")
        
        # Error metrics if available
        if result.error_metrics:
            print(f"\n🎯 ERROR METRICS:")
            for metric, value in result.error_metrics.items():
                print(f"  {metric}: {value:.2e}")
        
    except Exception as e:
        print(f"  ❌ Compression FAILED: {e}")
        return {'compression_failed': True, 'error': str(e)}
    
    # ==================== DECOMPRESSION ====================
    print(f"\n🔓 DECOMPRESSION PHASE:")
    print(f"  Starting decompression...")
    
    decompression_start = time.perf_counter()
    try:
        decomp_result = system.decompress(result.compressed_data)
        decompression_end = time.perf_counter()
        decompression_time = decompression_end - decompression_start
        
        print(f"  ✅ Decompression completed in {format_time(decompression_time)}")
        
        # Decompression metrics
        print(f"\n📊 DECOMPRESSION METRICS:")
        print(f"  Decompression speed: {format_bytes(int(compressed_bytes / decompression_time))}/s")
        print(f"  Throughput: {tensor.numel() / decompression_time / 1e6:.2f} M elements/s")
        print(f"  Memory usage: {format_bytes(decomp_result.memory_usage)}")
        print(f"  Output shape: {list(decomp_result.reconstructed_tensor.shape)}")
        print(f"  Validation: {'✅ PASSED' if decomp_result.validation_passed else '❌ FAILED'}")
        
        # Stage-by-stage breakdown
        print(f"\n⏱️  DECOMPRESSION STAGE BREAKDOWN:")
        total_stage_time = 0
        for stage, metrics in decomp_result.stage_metrics.items():
            stage_time = metrics.get('time', 0)
            total_stage_time += stage_time
            stage_pct = (stage_time / decompression_time) * 100
            print(f"  {stage}:")
            print(f"    Time: {format_time(stage_time)} ({stage_pct:.1f}% of total)")
            
            # Additional stage-specific metrics
            if 'metadata_size' in metrics:
                print(f"    Metadata size: {format_bytes(metrics['metadata_size'])}")
            if 'values_decoded' in metrics:
                print(f"    Values decoded: {metrics['values_decoded']:,}")
            if 'tensor_shape' in metrics:
                print(f"    Tensor shape: {metrics['tensor_shape']}")
            if 'patterns_used' in metrics:
                print(f"    Patterns used: {metrics['patterns_used']}")
            if 'weights_reconstructed' in metrics:
                print(f"    Weights reconstructed: {metrics['weights_reconstructed']:,}")
        
        # Reconstruction quality
        print(f"\n🎯 RECONSTRUCTION QUALITY:")
        mse = torch.nn.functional.mse_loss(tensor, decomp_result.reconstructed_tensor)
        mae = torch.nn.functional.l1_loss(tensor, decomp_result.reconstructed_tensor)
        max_error = torch.max(torch.abs(tensor - decomp_result.reconstructed_tensor))
        rel_error = torch.norm(tensor - decomp_result.reconstructed_tensor) / torch.norm(tensor)
        
        print(f"  MSE: {mse.item():.2e}")
        print(f"  MAE: {mae.item():.2e}")
        print(f"  Max absolute error: {max_error.item():.2e}")
        print(f"  Relative error: {rel_error.item():.2e}")
        print(f"  Reconstruction error: {decomp_result.reconstruction_error:.2e}")
        
        # Check for NaN/Inf
        has_nan = torch.isnan(decomp_result.reconstructed_tensor).any()
        has_inf = torch.isinf(decomp_result.reconstructed_tensor).any()
        if has_nan or has_inf:
            print(f"  ⚠️  WARNING: Reconstruction contains {'NaN' if has_nan else ''} {'Inf' if has_inf else ''}")
        
    except Exception as e:
        print(f"  ❌ Decompression FAILED: {e}")
        return {'decompression_failed': True, 'error': str(e)}
    
    # ==================== PERFORMANCE COMPARISON ====================
    print(f"\n⚡ PERFORMANCE COMPARISON:")
    print(f"  Compression time: {format_time(compression_time)}")
    print(f"  Decompression time: {format_time(decompression_time)}")
    print(f"  Compression/Decompression ratio: {compression_time/decompression_time:.2f}x")
    print(f"  Total round-trip time: {format_time(compression_time + decompression_time)}")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Effective bandwidth: {format_bytes(int(original_bytes / (compression_time + decompression_time)))}/s")
    
    return {
        'success': True,
        'compression_ratio': compression_ratio,
        'compression_time': compression_time,
        'decompression_time': decompression_time,
        'mse': mse.item(),
        'max_error': max_error.item(),
        'memory_usage': max(result.memory_usage, decomp_result.memory_usage)
    }


def main():
    """Run comprehensive compression system tests with full terminal output"""
    
    print_separator("COMPREHENSIVE P-ADIC COMPRESSION SYSTEM TEST", 100)
    print("\nObjective: Test and display ALL metrics for compression/decompression pipeline")
    print("Target: 4x compression ratio with detailed performance analysis")
    
    # Configure system
    config = CompressionConfig(
        prime=257,
        base_precision=4,
        target_error=1e-6,
        enable_gpu=torch.cuda.is_available(),
        validate_reconstruction=True,
        enable_logging=False  # Reduce noise in output
    )
    
    # Initialize system
    print(f"\n🔧 SYSTEM CONFIGURATION:")
    print(f"  Prime: {config.prime}")
    print(f"  Base precision: {config.base_precision}")
    print(f"  Target error: {config.target_error:.0e}")
    print(f"  GPU enabled: {config.enable_gpu}")
    print(f"  Device: {torch.cuda.get_device_name() if config.enable_gpu else 'CPU'}")
    
    system = PadicCompressionSystem(config)
    print(f"  ✅ System initialized successfully")
    
    # Define test cases
    test_cases = [
        ("Small Dense Tensor", torch.randn(100, 100) * 0.1),
        ("Medium Sparse Tensor", torch.randn(500, 500) * (torch.rand(500, 500) > 0.9)),
        ("Large Dense Tensor", torch.randn(1000, 1000) * 0.01),
        ("High Variance Tensor", torch.randn(300, 300) * 10),
        ("Uniform Values", torch.ones(400, 400) * 0.5),
        ("Mixed Sparsity", torch.randn(600, 600) * (torch.rand(600, 600) > 0.7)),
        ("Neural Network Weights", torch.randn(2048, 1024) * np.sqrt(2.0 / 2048)),  # He initialization
    ]
    
    # Run tests and collect results
    all_results = []
    
    for name, tensor in test_cases:
        # Generate importance scores (simulating gradient magnitudes)
        importance = torch.abs(tensor) + 0.01
        
        result = run_comprehensive_test(name, tensor, system, importance)
        all_results.append((name, result))
    
    # ==================== SUMMARY ====================
    print_separator("FINAL SUMMARY", 100)
    
    successful_tests = [r for _, r in all_results if r.get('success', False)]
    failed_tests = [r for _, r in all_results if not r.get('success', False)]
    
    print(f"\n📊 TEST RESULTS:")
    print(f"  Total tests: {len(all_results)}")
    print(f"  Successful: {len(successful_tests)}")
    print(f"  Failed: {len(failed_tests)}")
    
    if successful_tests:
        print(f"\n📈 AGGREGATE METRICS (Successful Tests):")
        
        # Compression ratios
        ratios = [r['compression_ratio'] for r in successful_tests]
        print(f"  Compression Ratios:")
        print(f"    Average: {np.mean(ratios):.2f}x")
        print(f"    Min: {np.min(ratios):.2f}x")
        print(f"    Max: {np.max(ratios):.2f}x")
        print(f"    Std Dev: {np.std(ratios):.2f}")
        
        # Compression times
        comp_times = [r['compression_time'] for r in successful_tests]
        print(f"  Compression Times:")
        print(f"    Average: {format_time(np.mean(comp_times))}")
        print(f"    Min: {format_time(np.min(comp_times))}")
        print(f"    Max: {format_time(np.max(comp_times))}")
        
        # Decompression times
        decomp_times = [r['decompression_time'] for r in successful_tests]
        print(f"  Decompression Times:")
        print(f"    Average: {format_time(np.mean(decomp_times))}")
        print(f"    Min: {format_time(np.min(decomp_times))}")
        print(f"    Max: {format_time(np.max(decomp_times))}")
        
        # Reconstruction quality
        mses = [r['mse'] for r in successful_tests]
        print(f"  Reconstruction Quality (MSE):")
        print(f"    Average: {np.mean(mses):.2e}")
        print(f"    Min: {np.min(mses):.2e}")
        print(f"    Max: {np.max(mses):.2e}")
        
        # Memory usage
        memory = [r['memory_usage'] for r in successful_tests]
        print(f"  Peak Memory Usage:")
        print(f"    Average: {format_bytes(int(np.mean(memory)))}")
        print(f"    Max: {format_bytes(int(np.max(memory)))}")
    
    # System statistics
    print(f"\n💾 SYSTEM STATISTICS:")
    stats = system.get_statistics()
    print(f"  Total compressions: {stats['total_compressions']}")
    print(f"  Total decompressions: {stats['total_decompressions']}")
    print(f"  Total bytes compressed: {format_bytes(stats['total_bytes_compressed'])}")
    print(f"  Total bytes decompressed: {format_bytes(stats['total_bytes_decompressed'])}")
    print(f"  Average compression ratio: {stats['average_compression_ratio']:.2f}x")
    print(f"  Peak memory usage: {format_bytes(stats['peak_memory_usage'])}")
    
    # Stage time averages
    print(f"\n⏱️  AVERAGE STAGE TIMES:")
    for stage in ['adaptive_precision', 'pattern_detection', 'sparse_encoding', 
                  'entropy_coding', 'metadata_compression']:
        key = f'avg_{stage}_time'
        if key in stats:
            print(f"  {stage}: {format_time(stats[key])}")
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT:")
    avg_ratio = np.mean([r['compression_ratio'] for r in successful_tests]) if successful_tests else 0
    if avg_ratio >= 4.0:
        print(f"  ✅ TARGET ACHIEVED: Average {avg_ratio:.2f}x compression (target: 4x)")
    elif avg_ratio >= 3.0:
        print(f"  ⚠️  CLOSE TO TARGET: Average {avg_ratio:.2f}x compression (target: 4x)")
    else:
        print(f"  ❌ BELOW TARGET: Average {avg_ratio:.2f}x compression (target: 4x)")
    
    # Cleanup
    system.cleanup()
    print(f"\n✅ All tests completed. System cleaned up.")
    
    return all_results


if __name__ == "__main__":
    results = main()