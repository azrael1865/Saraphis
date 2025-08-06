"""
Demo script showing convolutional analyzer capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from independent_core.compression_systems.neural_analysis.convolutional_analyzer import (
    ConvolutionalAnalyzer,
    ConvolutionalCompressionStrategy,
    AdaptiveConvCompressor
)


def create_sample_cnn():
    """Create a sample CNN for demonstration"""
    class SampleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            # Feature extraction layers
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            
            # Residual blocks
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            
            # Deeper layers
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
            self.bn4 = nn.BatchNorm2d(256)
            self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.bn5 = nn.BatchNorm2d(256)
            
            # Final layers
            self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
            self.bn6 = nn.BatchNorm2d(512)
            
            # Classifier
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, 1000)
        
        def forward(self, x):
            # Initial conv
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            
            # Residual blocks
            identity = x
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
            
            # Deeper layers
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))
            
            # Final conv
            x = F.relu(self.bn6(self.conv6(x)))
            
            # Classifier
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            
            return x
    
    return SampleCNN()


def analyze_model_layers(model, analyzer):
    """Analyze all convolutional layers in the model"""
    print("\n" + "="*60)
    print("LAYER-BY-LAYER ANALYSIS")
    print("="*60)
    
    total_conv_params = 0
    compressible_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"\n📊 Analyzing: {name}")
            print("-" * 40)
            
            # Analyze layer
            analysis = analyzer.analyze_conv2d(module)
            
            # Extract key metrics
            filter_analysis = analysis['filter_analysis']
            channel_analysis = analysis['channel_analysis']
            spatial_analysis = analysis['spatial_analysis']
            suggestions = analysis['decomposition_suggestions']
            
            # Print dimensions
            shape = analysis['shape']
            print(f"  Shape: {shape['out_channels']} × {shape['in_channels']} × "
                  f"{shape['kernel_size'][0]} × {shape['kernel_size'][1]}")
            print(f"  Parameters: {analysis['parameter_count']:,}")
            
            # Filter analysis
            print(f"\n  Filter Analysis:")
            print(f"    • Redundant filters: {len(filter_analysis.redundant_filters)}/{filter_analysis.num_filters}")
            print(f"    • Separable filters: {len(filter_analysis.separable_filters)}/{filter_analysis.num_filters}")
            print(f"    • Prunable ratio: {filter_analysis.prunable_ratio:.1%}")
            
            # Channel analysis
            print(f"\n  Channel Analysis:")
            print(f"    • Dead channels: {len(channel_analysis.dead_channels)}")
            print(f"    • Suggested groups: {channel_analysis.suggested_channel_groups}")
            
            # Spatial analysis
            print(f"\n  Spatial Analysis:")
            print(f"    • Spatial sparsity: {spatial_analysis.spatial_sparsity:.1%}")
            print(f"    • Depthwise separable: {spatial_analysis.is_depthwise_separable}")
            print(f"    • Edge detector score: {spatial_analysis.edge_detector_score:.2f}")
            print(f"    • Texture score: {spatial_analysis.texture_score:.2f}")
            
            # Compression suggestions
            print(f"\n  Compression Strategies:")
            for strategy_name, strategy_info in suggestions.items():
                if strategy_info.get('feasible', False):
                    ratio = strategy_info.get('compression_ratio', 1.0)
                    print(f"    ✅ {strategy_name}: {ratio:.2f}x compression")
                else:
                    print(f"    ❌ {strategy_name}: not feasible")
            
            # Track compressible parameters
            layer_params = analysis['parameter_count']
            total_conv_params += layer_params
            
            # Find best compression
            best_compression = 1.0
            for strategy_info in suggestions.values():
                if strategy_info.get('feasible', False):
                    ratio = strategy_info.get('compression_ratio', 1.0)
                    best_compression = max(best_compression, ratio)
            
            layer_compressible = layer_params * (1 - 1/best_compression)
            compressible_params += layer_compressible
            
            print(f"\n  Compression Score: {analysis['compression_score']:.2f}")
    
    return total_conv_params, compressible_params


def demonstrate_compression_strategies(model, strategy):
    """Demonstrate various compression strategies"""
    print("\n" + "="*60)
    print("COMPRESSION STRATEGY DEMONSTRATIONS")
    print("="*60)
    
    # Find a suitable conv layer for demonstration
    test_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels >= 64:
            test_layer = module
            break
    
    if test_layer is None:
        print("No suitable layer found for demonstration")
        return
    
    print(f"\nOriginal Layer: Conv2d({test_layer.in_channels}, {test_layer.out_channels}, "
          f"kernel_size={test_layer.kernel_size})")
    original_params = sum(p.numel() for p in test_layer.parameters())
    print(f"Original parameters: {original_params:,}")
    
    # 1. Structured Pruning
    print("\n1️⃣ Structured Pruning (50% filters)")
    pruned = strategy.structured_pruning(test_layer, pruning_ratio=0.5)
    pruned_params = sum(p.numel() for p in pruned.parameters())
    print(f"   After pruning: Conv2d({pruned.in_channels}, {pruned.out_channels})")
    print(f"   Parameters: {pruned_params:,} ({original_params/pruned_params:.2f}x compression)")
    
    # 2. Low-rank Decomposition
    print("\n2️⃣ Low-rank Decomposition (rank ratio=0.5)")
    decomposed = strategy.low_rank_decomposition(test_layer, rank_ratio=0.5)
    decomposed_params = sum(p.numel() for m in decomposed for p in m.parameters())
    print(f"   Decomposed into {len(decomposed)} layers")
    print(f"   Parameters: {decomposed_params:,} ({original_params/decomposed_params:.2f}x compression)")
    
    # 3. Depthwise Separable
    if test_layer.groups == 1:
        print("\n3️⃣ Depthwise Separable Convolution")
        depthwise = strategy.depthwise_separation(test_layer)
        depthwise_params = sum(p.numel() for m in depthwise for p in m.parameters())
        print(f"   Split into depthwise + pointwise")
        print(f"   Parameters: {depthwise_params:,} ({original_params/depthwise_params:.2f}x compression)")
    
    # 4. Spatial Decomposition (if square kernel)
    if test_layer.kernel_size[0] == test_layer.kernel_size[1] and test_layer.kernel_size[0] > 1:
        print("\n4️⃣ Spatial Decomposition (k×k → k×1 + 1×k)")
        spatial = strategy.spatial_decomposition(test_layer)
        spatial_params = sum(p.numel() for m in spatial for p in m.parameters())
        print(f"   Decomposed into vertical + horizontal")
        print(f"   Parameters: {spatial_params:,} ({original_params/spatial_params:.2f}x compression)")


def main():
    """Main demo function"""
    print("\n" + "="*60)
    print("🚀 CONVOLUTIONAL NEURAL NETWORK COMPRESSION DEMO")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    # Create model
    print("\n📦 Creating sample CNN model...")
    model = create_sample_cnn().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Create analyzer and strategy
    analyzer = ConvolutionalAnalyzer(device=device)
    strategy = ConvolutionalCompressionStrategy(device=device)
    
    # Analyze layers
    conv_params, compressible = analyze_model_layers(model, analyzer)
    
    # Summary statistics
    print("\n" + "="*60)
    print("COMPRESSION POTENTIAL SUMMARY")
    print("="*60)
    print(f"\n📈 Total convolutional parameters: {conv_params:,}")
    print(f"📉 Compressible parameters: {compressible:,}")
    print(f"🎯 Potential compression ratio: {conv_params/(conv_params-compressible):.2f}x")
    
    # Demonstrate strategies
    demonstrate_compression_strategies(model, strategy)
    
    # Adaptive compression
    print("\n" + "="*60)
    print("ADAPTIVE MODEL COMPRESSION")
    print("="*60)
    
    compressor = AdaptiveConvCompressor(target_compression=4.0, device=device)
    
    # Analyze compressibility
    print("\n🔍 Analyzing model compressibility...")
    analysis = compressor.analyze_cnn_compressibility(model)
    print(f"   Estimated compression ratio: {analysis['estimated_compression_ratio']:.2f}x")
    print(f"   Conv layers: {analysis['num_conv_layers']}")
    print(f"   Conv parameters: {analysis['conv_percentage']:.1%} of total")
    
    # Compress model
    print("\n🗜️ Compressing model...")
    compressed_model = compressor.compress_cnn(model)
    
    # Compare sizes
    original_size = sum(p.numel() for p in model.parameters())
    compressed_size = sum(p.numel() for p in compressed_model.parameters())
    actual_compression = original_size / compressed_size
    
    print(f"\n✅ Compression Complete!")
    print(f"   Original size: {original_size:,} parameters")
    print(f"   Compressed size: {compressed_size:,} parameters")
    print(f"   Actual compression: {actual_compression:.2f}x")
    
    # Show which layers were compressed
    print("\n📋 Compressed Layers:")
    for name, module in compressed_model.named_modules():
        if isinstance(module, nn.Sequential):
            # This is a compressed layer
            params = sum(p.numel() for p in module.parameters())
            print(f"   • {name}: Sequential module with {params:,} parameters")
    
    print("\n" + "="*60)
    print("🎉 Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    main()