"""
Test and example usage of DenseLayerAnalyzer
"""

import torch
import torch.nn as nn
from layer_analyzer import DenseLayerAnalyzer, CompressionMethod


def create_test_layers():
    """Create various test layers with different properties"""
    
    # Low-rank layer (good for tropical)
    low_rank_layer = nn.Linear(100, 50)
    with torch.no_grad():
        # Create rank-5 approximation
        U = torch.randn(50, 5)
        V = torch.randn(5, 100)
        low_rank_layer.weight.data = U @ V
        low_rank_layer.weight.data += torch.randn(50, 100) * 0.01  # Small noise
    
    # Sparse layer (good for tropical)
    sparse_layer = nn.Linear(64, 64)
    with torch.no_grad():
        sparse_layer.weight.data = torch.randn(64, 64)
        # Make 80% sparse
        mask = torch.rand(64, 64) > 0.2
        sparse_layer.weight.data[mask] = 0
    
    # Dense full-rank layer (good for p-adic)
    dense_layer = nn.Linear(32, 32)
    with torch.no_grad():
        # Well-conditioned dense matrix
        dense_layer.weight.data = torch.randn(32, 32) * 0.1 + torch.eye(32) * 0.5
    
    # Block diagonal layer (structured, good for tropical)
    block_diagonal_layer = nn.Linear(48, 48)
    with torch.no_grad():
        block_diagonal_layer.weight.data.zero_()
        # Create 3 blocks of 16x16
        for i in range(3):
            start = i * 16
            end = (i + 1) * 16
            block_diagonal_layer.weight.data[start:end, start:end] = torch.randn(16, 16)
    
    return {
        'low_rank': low_rank_layer,
        'sparse': sparse_layer,
        'dense': dense_layer,
        'block_diagonal': block_diagonal_layer
    }


def main():
    """Test layer analyzer on various layer types"""
    
    # Initialize analyzer
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'energy_threshold': 0.99,
        'numerical_tolerance': 1e-6,
        'near_zero_threshold': 1e-8,
        'use_randomized_svd': True,
        'max_svd_size': 1000
    }
    
    analyzer = DenseLayerAnalyzer(config)
    
    # Create test layers
    test_layers = create_test_layers()
    
    # Analyze each layer
    print("=" * 80)
    print("LAYER ANALYSIS RESULTS")
    print("=" * 80)
    
    for name, layer in test_layers.items():
        print(f"\n{'='*60}")
        print(f"Analyzing: {name}")
        print(f"{'='*60}")
        
        try:
            result = analyzer.analyze_layer(layer, name)
            
            # Print results
            print(f"\nLayer: {result.layer_name}")
            print(f"Type: {result.layer_type}")
            print(f"Shape: {result.shape}")
            print(f"Parameters: {result.parameter_count:,}")
            print(f"Device: {result.device}")
            print(f"Analysis time: {result.analysis_time_ms:.2f} ms")
            
            print(f"\nRank Analysis:")
            print(f"  Effective rank: {result.rank_analysis.effective_rank}")
            print(f"  Numerical rank: {result.rank_analysis.numerical_rank}")
            print(f"  Stable rank: {result.rank_analysis.stable_rank:.2f}")
            print(f"  Rank ratio: {result.rank_analysis.rank_ratio:.3f}")
            
            print(f"\nSparsity Analysis:")
            print(f"  Zero ratio: {result.sparsity_analysis.zero_ratio:.3f}")
            print(f"  Near-zero ratio: {result.sparsity_analysis.near_zero_ratio:.3f}")
            print(f"  Block sparsity: {result.sparsity_analysis.block_sparsity_detected}")
            print(f"  Structured pattern: {result.sparsity_analysis.structured_pattern}")
            
            print(f"\nNumerical Analysis:")
            print(f"  Condition number: {result.numerical_analysis.condition_number:.2f}")
            print(f"  Dynamic range: {result.numerical_analysis.dynamic_range:.2f}")
            print(f"  Frobenius norm: {result.numerical_analysis.frobenius_norm:.2f}")
            print(f"  Spectral norm: {result.numerical_analysis.spectral_norm:.2f}")
            
            print(f"\nCompression Recommendation:")
            print(f"  Method: {result.compression_recommendation.method.value}")
            print(f"  Tropical score: {result.compression_recommendation.tropical_score:.3f}")
            print(f"  P-adic score: {result.compression_recommendation.padic_score:.3f}")
            print(f"  Hybrid score: {result.compression_recommendation.hybrid_score:.3f}")
            print(f"  Confidence: {result.compression_recommendation.confidence:.3f}")
            print(f"  Estimated compression ratio: {result.compression_recommendation.estimated_compression_ratio:.2f}x")
            print(f"  Reasoning: {result.compression_recommendation.reasoning}")
            
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()