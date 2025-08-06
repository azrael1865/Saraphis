#!/usr/bin/env python3
"""
Test script for LogarithmicPadicWeight compression functionality
Demonstrates the actual compression pipeline with real data
"""

import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def test_logarithmic_padic_compression():
    """Test the LogarithmicPadicWeight compression system"""
    
    print("=" * 60)
    print("LogarithmicPadicWeight Compression Test")
    print("=" * 60)
    
    try:
        # Import the logarithmic p-adic components
        from independent_core.compression_systems.padic.padic_logarithmic_encoder import (
            LogarithmicPadicWeight,
            PadicLogarithmicEncoder,
            LogarithmicEncodingConfig,
            IEEE754Channels
        )
        
        print("✓ Successfully imported LogarithmicPadicWeight components")
        
        # Create configuration with safe parameters
        config = LogarithmicEncodingConfig(
            prime=17,  # Small safe prime
            precision=2,  # Low precision for safety
            max_safe_precision=3,
            use_natural_log=True,
            normalize_before_log=True,
            scale_factor=100.0,
            separate_channel_encoding=True,
            enable_delta_encoding=True,
            quantization_levels=256
        )
        
        print(f"\n✓ Created configuration:")
        print(f"  Prime: {config.prime}")
        print(f"  Precision: {config.precision}")
        print(f"  Scale factor: {config.scale_factor}")
        print(f"  Quantization levels: {config.quantization_levels}")
        
        # Create encoder
        encoder = PadicLogarithmicEncoder(config)
        print("\n✓ Created PadicLogarithmicEncoder")
        
        # Test 1: Single tensor compression
        print("\n" + "-" * 40)
        print("Test 1: Single Tensor Compression")
        print("-" * 40)
        
        # Create test tensor
        test_tensor = torch.randn(10, 10) * 2.0  # Random values
        original_size = test_tensor.numel() * 4  # 4 bytes per float32
        
        print(f"Original tensor shape: {test_tensor.shape}")
        print(f"Original size: {original_size} bytes")
        print(f"Sample values: {test_tensor.flatten()[:5].tolist()}")
        
        # Compress tensor
        start_time = time.time()
        compressed_weights = encoder.encode_weights_logarithmically(test_tensor)
        compression_time = (time.time() - start_time) * 1000
        
        print(f"\n✓ Compressed {len(compressed_weights)} weights")
        print(f"  Compression time: {compression_time:.2f}ms")
        
        # Calculate compression statistics
        total_compressed_size = 0
        compression_ratios = []
        
        for i, weight in enumerate(compressed_weights[:3]):  # Show first 3
            ratio = weight.get_compression_ratio()
            compression_ratios.append(ratio)
            print(f"  Weight {i}: compression ratio = {ratio:.2f}x")
        
        avg_ratio = np.mean(compression_ratios) if compression_ratios else 1.0
        print(f"\nAverage compression ratio: {avg_ratio:.2f}x")
        
        # Test 2: Model weight compression
        print("\n" + "-" * 40)
        print("Test 2: Neural Network Model Compression")
        print("-" * 40)
        
        # Create a simple model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(20, 10)
                self.fc2 = nn.Linear(10, 5)
                self.fc3 = nn.Linear(5, 2)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.fc3(x)
        
        model = TestModel()
        
        # Count original parameters
        original_params = sum(p.numel() for p in model.parameters())
        original_bytes = original_params * 4  # float32
        
        print(f"Model architecture:")
        print(f"  fc1: Linear(20, 10)")
        print(f"  fc2: Linear(10, 5)")
        print(f"  fc3: Linear(5, 2)")
        print(f"  Total parameters: {original_params}")
        print(f"  Original size: {original_bytes} bytes ({original_bytes/1024:.2f} KB)")
        
        # Compress each layer
        compressed_model_weights = {}
        total_compression_time = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                start_time = time.time()
                compressed = encoder.encode_weights_logarithmically(param.data)
                layer_time = (time.time() - start_time) * 1000
                total_compression_time += layer_time
                
                compressed_model_weights[name] = compressed
                
                # Calculate layer compression ratio
                layer_ratios = [w.get_compression_ratio() for w in compressed]
                avg_layer_ratio = np.mean(layer_ratios) if layer_ratios else 1.0
                
                print(f"\n  Layer '{name}':")
                print(f"    Shape: {param.shape}")
                print(f"    Elements: {param.numel()}")
                print(f"    Compressed weights: {len(compressed)}")
                print(f"    Compression ratio: {avg_layer_ratio:.2f}x")
                print(f"    Time: {layer_time:.2f}ms")
        
        print(f"\nTotal compression time: {total_compression_time:.2f}ms")
        
        # Test 3: IEEE 754 channel extraction
        print("\n" + "-" * 40)
        print("Test 3: IEEE 754 Channel Processing")
        print("-" * 40)
        
        # Test with specific float values
        test_values = torch.tensor([
            3.14159,    # Pi
            2.71828,    # e
            1.41421,    # sqrt(2)
            0.57722,    # Euler-Mascheroni constant
            -1.23456,   # Negative value
            0.001,      # Small value
            1000.0,     # Large value
            0.0         # Zero
        ])
        
        print("Test values:")
        for i, val in enumerate(test_values):
            print(f"  {i}: {val:.6f}")
        
        # Extract IEEE 754 channels
        from independent_core.compression_systems.categorical.ieee754_channel_extractor import (
            IEEE754ChannelExtractor, extract_ieee754_from_tensor
        )
        channels = extract_ieee754_from_tensor(test_values)
        
        print(f"\n✓ Extracted IEEE 754 channels:")
        print(f"  Sign channel shape: {channels.sign_channel.shape}")
        print(f"  Exponent channel shape: {channels.exponent_channel.shape}")
        print(f"  Mantissa channel shape: {channels.mantissa_channel.shape}")
        
        # Compress each channel separately
        print("\nCompressing channels separately:")
        
        # Compress sign channel
        sign_tensor = torch.from_numpy(channels.sign_channel).float()
        sign_compressed = encoder.encode_weights_logarithmically(sign_tensor)
        sign_ratio = np.mean([w.get_compression_ratio() for w in sign_compressed])
        print(f"  Sign channel: {len(sign_compressed)} weights, ratio={sign_ratio:.2f}x")
        
        # Compress exponent channel
        exp_tensor = torch.from_numpy(channels.exponent_channel).float()
        exp_compressed = encoder.encode_weights_logarithmically(exp_tensor)
        exp_ratio = np.mean([w.get_compression_ratio() for w in exp_compressed])
        print(f"  Exponent channel: {len(exp_compressed)} weights, ratio={exp_ratio:.2f}x")
        
        # Compress mantissa channel
        mantissa_tensor = torch.from_numpy(channels.mantissa_channel).float()
        mantissa_compressed = encoder.encode_weights_logarithmically(mantissa_tensor)
        mantissa_ratio = np.mean([w.get_compression_ratio() for w in mantissa_compressed])
        print(f"  Mantissa channel: {len(mantissa_compressed)} weights, ratio={mantissa_ratio:.2f}x")
        
        # Test 4: Decompression (if available)
        print("\n" + "-" * 40)
        print("Test 4: Decompression Check")
        print("-" * 40)
        
        if hasattr(encoder, 'decode_logarithmic_padic_weights'):
            print("✓ Decompression method available")
            
            # Try to decompress weights
            try:
                decompressed = encoder.decode_logarithmic_padic_weights(compressed_weights[:10])
                print(f"  Successfully decompressed {decompressed.shape} tensor")
                print(f"  Original values: {test_tensor.flatten()[:5].tolist()}")
                print(f"  Decompressed values: {decompressed.flatten()[:5].tolist()}")
                
                # Calculate reconstruction error
                original_subset = test_tensor.flatten()[:decompressed.numel()]
                error = torch.mean(torch.abs(original_subset - decompressed.flatten()))
                print(f"  Mean absolute error: {error:.6e}")
            except Exception as e:
                print(f"  Decompression error: {e}")
        else:
            print("  Decompression method not available")
        
        # Summary
        print("\n" + "=" * 60)
        print("COMPRESSION SUMMARY")
        print("=" * 60)
        
        print("\n✅ Successfully tested:")
        print("  - LogarithmicPadicWeight creation")
        print("  - Tensor compression")
        print("  - Model weight compression")
        print("  - IEEE 754 channel extraction")
        print("  - Compression ratio calculation")
        
        print("\n📊 Performance metrics:")
        print(f"  Average compression ratio: {avg_ratio:.2f}x")
        print(f"  Model compression time: {total_compression_time:.2f}ms")
        print(f"  Throughput: {original_params / (total_compression_time/1000):.0f} params/sec")
        
        print("\n🎯 Next steps:")
        print("  1. Implement full decompression pipeline")
        print("  2. Add GPU acceleration for larger models")
        print("  3. Optimize compression ratios with better quantization")
        print("  4. Add model fine-tuning after compression")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install torch numpy")
        return False
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_logarithmic_padic_compression()
    sys.exit(0 if success else 1)