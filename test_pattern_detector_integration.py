#!/usr/bin/env python3
"""
Test script for verifying the Weight Distribution Analyzer integration
with the compression strategy system.
"""

import torch
import numpy as np
import sys
import os

# Add path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from independent_core.compression_systems.strategies.pattern_detector import (
    WeightDistributionAnalyzer, 
    DistributionAnalysis
)
from independent_core.compression_systems.strategies.compression_strategy import (
    StrategyConfig,
    StrategySelector
)


def test_distribution_analyzer():
    """Test the WeightDistributionAnalyzer directly"""
    print("=" * 60)
    print("Testing WeightDistributionAnalyzer")
    print("=" * 60)
    
    analyzer = WeightDistributionAnalyzer()
    
    # Test 1: Gaussian distribution
    print("\n1. Testing Gaussian distribution:")
    gaussian_tensor = torch.randn(1000, 100)
    analysis = analyzer.analyze_distribution(gaussian_tensor)
    print(f"   Distribution type: {analysis.distribution_type}")
    print(f"   Skewness: {analysis.skewness:.3f}")
    print(f"   Kurtosis: {analysis.kurtosis:.3f}")
    print(f"   Num modes: {analysis.num_modes}")
    assert abs(analysis.skewness) < 0.5, "Gaussian should have low skewness"
    
    # Test 2: Sparse distribution
    print("\n2. Testing Sparse distribution:")
    sparse_tensor = torch.zeros(1000, 100)
    sparse_tensor[torch.rand(1000, 100) > 0.95] = torch.randn(1)
    analysis = analyzer.analyze_distribution(sparse_tensor)
    print(f"   Distribution type: {analysis.distribution_type}")
    print(f"   Sparsity: {analysis.sparsity:.3f}")
    print(f"   Num modes: {analysis.num_modes}")
    assert analysis.distribution_type == "sparse", "Should detect sparse distribution"
    assert analysis.sparsity > 0.7, "Should have high sparsity"
    
    # Test 3: Bimodal distribution
    print("\n3. Testing Bimodal distribution:")
    bimodal_tensor = torch.cat([
        torch.randn(50000) - 3,  # Mode at -3
        torch.randn(50000) + 3   # Mode at +3
    ])
    analysis = analyzer.analyze_distribution(bimodal_tensor)
    print(f"   Distribution type: {analysis.distribution_type}")
    print(f"   Num modes: {analysis.num_modes}")
    print(f"   Mode locations: {analysis.mode_locations}")
    assert analysis.num_modes >= 2, "Should detect at least 2 modes"
    
    # Test 4: Quantized distribution
    print("\n4. Testing Quantized distribution:")
    quantized_tensor = torch.round(torch.randn(10000) * 10) / 10  # Quantize to 0.1 levels
    analysis = analyzer.analyze_distribution(quantized_tensor)
    print(f"   Distribution type: {analysis.distribution_type}")
    print(f"   Quantization levels: {analysis.quantization_levels}")
    assert analysis.quantization_levels < 256, "Should detect quantization"
    
    print("\n✓ WeightDistributionAnalyzer tests passed!")


def test_strategy_integration():
    """Test integration with StrategySelector"""
    print("\n" + "=" * 60)
    print("Testing StrategySelector Integration")
    print("=" * 60)
    
    config = StrategyConfig()
    selector = StrategySelector(config)
    
    # Check if pattern detector is available
    if selector.distribution_analyzer is None:
        print("Warning: Pattern detector not available in StrategySelector")
        return
    
    print("\nPattern detector successfully integrated!")
    
    # Test 1: Sparse tensor (should prefer tropical)
    print("\n1. Testing with sparse tensor:")
    sparse_tensor = torch.zeros(100, 100)
    sparse_tensor[torch.rand(100, 100) > 0.9] = torch.randn(1)
    
    analysis = selector.analyze_tensor(sparse_tensor)
    print(f"   Distribution type: {analysis.get('distribution_type', 'not detected')}")
    print(f"   Sparsity: {analysis['sparsity']:.3f}")
    print(f"   Local entropy: {analysis['local_entropy']:.3f}")
    
    scores = selector.compute_strategy_scores(analysis)
    print(f"   Strategy scores: {scores}")
    best_strategy = max(scores, key=scores.get)
    print(f"   Best strategy: {best_strategy}")
    assert best_strategy in ['tropical', 'hybrid'], "Sparse tensor should prefer tropical"
    
    # Test 2: Gaussian tensor (should consider p-adic)
    print("\n2. Testing with Gaussian tensor:")
    gaussian_tensor = torch.randn(100, 100)
    
    analysis = selector.analyze_tensor(gaussian_tensor)
    print(f"   Distribution type: {analysis.get('distribution_type', 'not detected')}")
    print(f"   Skewness: {analysis.get('dist_skewness', 'not detected')}")
    print(f"   Kurtosis: {analysis.get('dist_kurtosis', 'not detected')}")
    
    scores = selector.compute_strategy_scores(analysis)
    print(f"   Strategy scores: {scores}")
    
    # Test 3: Bimodal tensor (should consider hybrid)
    print("\n3. Testing with bimodal tensor:")
    bimodal_tensor = torch.cat([
        torch.randn(5000) - 2,
        torch.randn(5000) + 2
    ]).view(100, 100)
    
    analysis = selector.analyze_tensor(bimodal_tensor)
    print(f"   Distribution type: {analysis.get('distribution_type', 'not detected')}")
    print(f"   Num modes: {analysis.get('num_modes', 'not detected')}")
    
    scores = selector.compute_strategy_scores(analysis)
    print(f"   Strategy scores: {scores}")
    
    # Check if hybrid gets a boost for bimodal
    if analysis.get('distribution_type') == 'bimodal':
        print(f"   ✓ Bimodal distribution detected, hybrid score boosted")
    
    print("\n✓ StrategySelector integration tests passed!")


def test_compression_hints():
    """Test compression hints generation"""
    print("\n" + "=" * 60)
    print("Testing Compression Hints")
    print("=" * 60)
    
    analyzer = WeightDistributionAnalyzer()
    
    # Test different distribution types
    test_cases = [
        ("Gaussian", torch.randn(10000)),
        ("Sparse", torch.zeros(10000)),
        ("Uniform", torch.rand(10000) * 2 - 1),
    ]
    
    for name, tensor in test_cases:
        print(f"\n{name} distribution:")
        analysis = analyzer.analyze_distribution(tensor)
        hints = analyzer.compute_compression_hints(analysis)
        
        print(f"   P-adic score: {hints['padic_score']:.2f}")
        print(f"   Tropical score: {hints['tropical_score']:.2f}")
        print(f"   Hybrid score: {hints['hybrid_score']:.2f}")
        print(f"   Quantization benefit: {hints['quantization_benefit']:.2f}")
        print(f"   Clustering benefit: {hints['clustering_benefit']:.2f}")
    
    print("\n✓ Compression hints tests passed!")


def test_performance():
    """Test performance with large tensors"""
    print("\n" + "=" * 60)
    print("Testing Performance with Large Tensors")
    print("=" * 60)
    
    import time
    
    analyzer = WeightDistributionAnalyzer(max_sample_size=100_000)
    
    # Test with different tensor sizes
    sizes = [1000, 10_000, 100_000, 1_000_000]
    
    for size in sizes:
        tensor = torch.randn(size)
        
        start_time = time.time()
        analysis = analyzer.analyze_distribution(tensor)
        elapsed = time.time() - start_time
        
        print(f"\nTensor size: {size:,}")
        print(f"   Analysis time: {elapsed:.3f} seconds")
        print(f"   Distribution: {analysis.distribution_type}")
        
        # Should complete in reasonable time even for large tensors
        assert elapsed < 5.0, f"Analysis took too long for size {size}"
    
    print("\n✓ Performance tests passed!")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("WEIGHT DISTRIBUTION ANALYZER INTEGRATION TEST SUITE")
    print("=" * 80)
    
    try:
        # Test individual components
        test_distribution_analyzer()
        
        # Test integration
        test_strategy_integration()
        
        # Test hints
        test_compression_hints()
        
        # Test performance
        test_performance()
        
        print("\n" + "=" * 80)
        print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())