"""
Isolated test script for compression strategy module.
Tests core functionality without full system imports.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

def test_basic_functionality():
    """Test basic strategy functionality in isolation"""
    print("=" * 60)
    print("Testing Compression Strategy Module (Isolated)")
    print("=" * 60)
    
    # Import only what we need
    from compression_strategy import (
        StrategyConfig,
        TropicalStrategy,
        PadicStrategy,
        HybridStrategy,
        CompressedData
    )
    
    # Test 1: StrategyConfig
    print("\n1. Testing StrategyConfig...")
    config = StrategyConfig()
    assert config.sparsity_threshold == 0.7
    assert config.prime == 251
    print("   ✓ StrategyConfig initialized correctly")
    
    # Test 2: TropicalStrategy basic compression
    print("\n2. Testing TropicalStrategy...")
    tropical = TropicalStrategy()
    
    # Create test tensor
    tensor = torch.randn(10, 10)
    tensor[tensor.abs() < 0.5] = 0  # Make sparse
    
    # Compress
    compressed = tropical.compress(tensor)
    assert isinstance(compressed, CompressedData)
    assert compressed.strategy_name == "tropical"
    assert compressed.original_shape == (10, 10)
    print(f"   ✓ Compressed sparse tensor, ratio: {compressed.compression_ratio:.2f}x")
    
    # Decompress
    reconstructed = tropical.decompress(compressed)
    assert reconstructed.shape == tensor.shape
    error = torch.nn.functional.mse_loss(reconstructed, tensor).item()
    print(f"   ✓ Decompressed successfully, MSE: {error:.6f}")
    
    # Test 3: PadicStrategy basic compression
    print("\n3. Testing PadicStrategy...")
    padic = PadicStrategy(prime=251, precision=16)
    
    # Create periodic tensor
    tensor2 = torch.sin(torch.linspace(0, 4 * np.pi, 100)).view(10, 10)
    
    # Compress
    compressed2 = padic.compress(tensor2)
    assert compressed2.strategy_name == "padic"
    print(f"   ✓ Compressed periodic tensor, ratio: {compressed2.compression_ratio:.2f}x")
    
    # Decompress
    reconstructed2 = padic.decompress(compressed2)
    assert reconstructed2.shape == tensor2.shape
    error2 = torch.nn.functional.mse_loss(reconstructed2, tensor2).item()
    print(f"   ✓ Decompressed successfully, MSE: {error2:.6f}")
    
    # Test 4: HybridStrategy basic compression
    print("\n4. Testing HybridStrategy...")
    from independent_core.compression_systems.integration.padic_tropical_bridge import ConversionConfig
    
    hybrid_config = ConversionConfig(prime=251, precision=16)
    hybrid = HybridStrategy(hybrid_config)
    
    # Create complex tensor
    tensor3 = torch.randn(20, 20)
    tensor3[::2, ::2] = 0  # Add structure
    
    # Compress
    compressed3 = hybrid.compress(tensor3)
    assert compressed3.strategy_name == "hybrid"
    print(f"   ✓ Compressed complex tensor, ratio: {compressed3.compression_ratio:.2f}x")
    
    # Decompress
    reconstructed3 = hybrid.decompress(compressed3)
    assert reconstructed3.shape == tensor3.shape
    error3 = torch.nn.functional.mse_loss(reconstructed3, tensor3).item()
    print(f"   ✓ Decompressed successfully, MSE: {error3:.6f}")
    
    # Test 5: Strategy selection (simplified)
    print("\n5. Testing Strategy Selection Logic...")
    from compression_strategy import StrategySelector
    
    selector = StrategySelector(config)
    
    # Analyze different tensor types
    sparse_tensor = torch.zeros(50, 50)
    sparse_tensor[torch.rand(50, 50) > 0.95] = torch.randn(1)
    
    analysis = selector.analyze_tensor(sparse_tensor)
    print(f"   Sparse tensor analysis:")
    print(f"     - Sparsity: {analysis['sparsity']:.3f}")
    print(f"     - Rank ratio: {analysis['rank_ratio']:.3f}")
    print(f"     - Dynamic range: {analysis['dynamic_range']:.2e}")
    
    scores = selector.compute_strategy_scores(analysis)
    print(f"   Strategy scores:")
    for strategy, score in scores.items():
        print(f"     - {strategy}: {score:.3f}")
    
    best_strategy = max(scores, key=scores.get)
    print(f"   ✓ Selected strategy: {best_strategy}")
    
    # Test 6: Model compression (simplified)
    print("\n6. Testing Model Compression...")
    from compression_strategy import AdaptiveStrategyManager
    
    manager = AdaptiveStrategyManager(config)
    
    # Create small test model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    
    # Get model size
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"   Model has {total_params} parameters ({total_bytes} bytes)")
    
    # Compress model
    compressed_layers = manager.compress_model(model)
    print(f"   ✓ Compressed {len(compressed_layers)} layers")
    
    # Calculate total compression
    compressed_bytes = sum(c.get_size_bytes() for c in compressed_layers.values())
    overall_ratio = total_bytes / compressed_bytes if compressed_bytes > 0 else 1.0
    print(f"   ✓ Overall compression ratio: {overall_ratio:.2f}x")
    
    # Get summary
    summary = manager.get_compression_summary()
    print(f"   ✓ Average compression ratio: {summary['average_compression_ratio']:.2f}x")
    print(f"   ✓ Average reconstruction error: {summary['average_reconstruction_error']:.6f}")
    
    # Test strategy distribution
    print(f"   Strategy distribution:")
    for strategy, count in summary['strategy_distribution'].items():
        print(f"     - {strategy}: {count} layers")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_basic_functionality()