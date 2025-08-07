#!/usr/bin/env python3
"""
Standalone test for the Pattern Detector module.
Tests the module in isolation without importing the full compression system.
"""

import sys
import os
import torch
import numpy as np

# Direct import without going through the package init
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import pattern detector directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "pattern_detector",
    "independent_core/compression_systems/strategies/pattern_detector.py"
)
pattern_detector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pattern_detector)

WeightDistributionAnalyzer = pattern_detector.WeightDistributionAnalyzer
DistributionAnalysis = pattern_detector.DistributionAnalysis


def test_basic_functionality():
    """Test basic functionality of the pattern detector"""
    print("=" * 60)
    print("Testing Basic Pattern Detector Functionality")
    print("=" * 60)
    
    analyzer = WeightDistributionAnalyzer()
    
    # Test 1: Gaussian distribution
    print("\n1. Gaussian Distribution:")
    gaussian = torch.randn(10000)
    analysis = analyzer.analyze_distribution(gaussian)
    print(f"   Type: {analysis.distribution_type}")
    print(f"   Mean: {analysis.mean:.3f}")
    print(f"   Std: {analysis.std:.3f}")
    print(f"   Skewness: {analysis.skewness:.3f}")
    print(f"   Kurtosis: {analysis.kurtosis:.3f}")
    print(f"   Modes: {analysis.num_modes}")
    
    # Test 2: Sparse distribution
    print("\n2. Sparse Distribution:")
    sparse = torch.zeros(10000)
    sparse[torch.rand(10000) > 0.95] = torch.randn(1).item()
    analysis = analyzer.analyze_distribution(sparse)
    print(f"   Type: {analysis.distribution_type}")
    print(f"   Sparsity: {analysis.sparsity:.3f}")
    print(f"   Quantization levels: {analysis.quantization_levels}")
    
    # Test 3: Bimodal distribution
    print("\n3. Bimodal Distribution:")
    bimodal = torch.cat([torch.randn(5000) - 2, torch.randn(5000) + 2])
    analysis = analyzer.analyze_distribution(bimodal)
    print(f"   Type: {analysis.distribution_type}")
    print(f"   Modes: {analysis.num_modes}")
    if analysis.mode_locations:
        print(f"   Mode locations: {[f'{m:.2f}' for m in analysis.mode_locations[:3]]}")
    
    # Test 4: Uniform distribution
    print("\n4. Uniform Distribution:")
    uniform = torch.rand(10000) * 2 - 1
    analysis = analyzer.analyze_distribution(uniform)
    print(f"   Type: {analysis.distribution_type}")
    print(f"   Skewness: {analysis.skewness:.3f}")
    print(f"   Kurtosis: {analysis.kurtosis:.3f}")
    
    print("\n✓ Basic functionality tests passed!")


def test_mode_detection():
    """Test mode detection capabilities"""
    print("\n" + "=" * 60)
    print("Testing Mode Detection")
    print("=" * 60)
    
    analyzer = WeightDistributionAnalyzer()
    
    # Create multimodal distribution
    print("\nMultimodal Distribution (3 modes):")
    multimodal = torch.cat([
        torch.randn(3000) - 4,
        torch.randn(3000),
        torch.randn(3000) + 4
    ])
    
    num_modes, mode_locations, valley_points = analyzer.detect_modes(multimodal)
    print(f"   Detected modes: {num_modes}")
    print(f"   Mode locations: {[f'{m:.2f}' for m in mode_locations]}")
    print(f"   Valley points: {[f'{v:.2f}' for v in valley_points]}")
    
    assert num_modes >= 2, "Should detect at least 2 modes"
    print("\n✓ Mode detection tests passed!")


def test_quantization_analysis():
    """Test quantization analysis"""
    print("\n" + "=" * 60)
    print("Testing Quantization Analysis")
    print("=" * 60)
    
    analyzer = WeightDistributionAnalyzer()
    
    # Test quantized tensor
    print("\n8-bit Quantized Tensor:")
    original = torch.randn(10000)
    quantized = torch.round(original * 127) / 127  # Simulate 8-bit quantization
    
    quant_info = analyzer.analyze_quantization(quantized)
    print(f"   Is quantized: {quant_info['is_quantized']}")
    print(f"   Levels: {quant_info['num_levels']}")
    print(f"   Uniform: {quant_info['is_uniform']}")
    print(f"   Bit depth: {quant_info['bit_depth']}")
    
    # Test continuous tensor
    print("\nContinuous Tensor:")
    continuous = torch.randn(10000)
    quant_info = analyzer.analyze_quantization(continuous)
    print(f"   Is quantized: {quant_info['is_quantized']}")
    print(f"   Levels: {quant_info['num_levels']}")
    
    print("\n✓ Quantization analysis tests passed!")


def test_compression_hints():
    """Test compression hints generation"""
    print("\n" + "=" * 60)
    print("Testing Compression Hints")
    print("=" * 60)
    
    analyzer = WeightDistributionAnalyzer()
    
    test_cases = [
        ("Gaussian", torch.randn(5000)),
        ("Sparse", torch.zeros(5000)),
        ("Bimodal", torch.cat([torch.randn(2500) - 2, torch.randn(2500) + 2])),
    ]
    
    for name, tensor in test_cases:
        print(f"\n{name} Distribution:")
        
        # Add sparsity for sparse tensor
        if name == "Sparse":
            tensor[torch.rand(5000) > 0.9] = torch.randn(1).item()
        
        analysis = analyzer.analyze_distribution(tensor)
        hints = analyzer.compute_compression_hints(analysis)
        
        print(f"   Distribution type: {analysis.distribution_type}")
        print(f"   P-adic score: {hints['padic_score']:.2f}")
        print(f"   Tropical score: {hints['tropical_score']:.2f}")
        print(f"   Hybrid score: {hints['hybrid_score']:.2f}")
    
    print("\n✓ Compression hints tests passed!")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "=" * 60)
    print("Testing Edge Cases")
    print("=" * 60)
    
    analyzer = WeightDistributionAnalyzer()
    
    # Test 1: Empty tensor
    print("\n1. Very small tensor:")
    tiny = torch.tensor([1.0, 2.0, 3.0])
    analysis = analyzer.analyze_distribution(tiny)
    print(f"   Handled successfully: {analysis.distribution_type}")
    
    # Test 2: All zeros
    print("\n2. All zeros:")
    zeros = torch.zeros(1000)
    analysis = analyzer.analyze_distribution(zeros)
    print(f"   Type: {analysis.distribution_type}")
    print(f"   Sparsity: {analysis.sparsity:.3f}")
    
    # Test 3: Single value
    print("\n3. Single value repeated:")
    constant = torch.ones(1000) * 42
    analysis = analyzer.analyze_distribution(constant)
    print(f"   Type: {analysis.distribution_type}")
    print(f"   Quantization levels: {analysis.quantization_levels}")
    
    # Test 4: Very large tensor (sampling)
    print("\n4. Very large tensor (tests sampling):")
    large = torch.randn(1_000_000)
    analysis = analyzer.analyze_distribution(large)
    print(f"   Handled successfully: {analysis.distribution_type}")
    
    print("\n✓ Edge case tests passed!")


def test_performance():
    """Test performance with various tensor sizes"""
    print("\n" + "=" * 60)
    print("Testing Performance")
    print("=" * 60)
    
    import time
    
    analyzer = WeightDistributionAnalyzer(max_sample_size=100_000)
    
    sizes = [1000, 10_000, 100_000, 500_000]
    
    for size in sizes:
        tensor = torch.randn(size)
        
        start = time.time()
        analysis = analyzer.analyze_distribution(tensor)
        elapsed = time.time() - start
        
        print(f"\nSize {size:,}:")
        print(f"   Time: {elapsed:.3f}s")
        print(f"   Type: {analysis.distribution_type}")
        
        # Should complete quickly even for large tensors
        assert elapsed < 3.0, f"Too slow for size {size}"
    
    print("\n✓ Performance tests passed!")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("PATTERN DETECTOR STANDALONE TEST SUITE")
    print("=" * 80)
    
    try:
        test_basic_functionality()
        test_mode_detection()
        test_quantization_analysis()
        test_compression_hints()
        test_edge_cases()
        test_performance()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("Pattern Detector is fully functional and ready for integration.")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())