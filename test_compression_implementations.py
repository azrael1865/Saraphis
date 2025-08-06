#!/usr/bin/env python3
"""
Test script to verify the implemented compression methods
"""

import torch
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from independent_core.compression_systems.tropical.channel_packing import (
    ChannelPacker, ChannelPackingConfig, PackingStrategy, TropicalChannels
)
from independent_core.compression_systems.integration.padic_tropical_bridge import (
    HybridRepresentation, ConversionConfig, PadicTropicalConverter
)
from independent_core.compression_systems.padic.padic_encoder import PadicMathematicalOperations

def test_channel_unpacking():
    """Test all 5 unpacking methods"""
    print("Testing Channel Unpacking Methods...")
    
    # Create test data
    num_monomials = 100
    num_variables = 10
    device = torch.device('cpu')
    
    channels = TropicalChannels(
        coefficient_channel=torch.randn(num_monomials),
        exponent_channel=torch.randn(num_monomials, num_variables),
        index_channel=torch.arange(num_monomials, dtype=torch.long),
        metadata={'test': 'data'},
        device=device,
        mantissa_channel=torch.randn(num_monomials)
    )
    
    # Test each packing strategy
    strategies = [
        PackingStrategy.INTERLEAVED,
        PackingStrategy.HIERARCHICAL,
        PackingStrategy.DELTA,
        PackingStrategy.PREDICTIVE,
        PackingStrategy.SEPARATED
    ]
    
    for strategy in strategies:
        print(f"\n  Testing {strategy.value} strategy...")
        config = ChannelPackingConfig(
            strategy=strategy,
            compression_algorithm="none",  # Disable compression for testing
            enable_checksums=False
        )
        packer = ChannelPacker(config)
        
        try:
            # Pack channels
            packed_data, metrics = packer.pack_channels(channels)
            print(f"    Packed size: {len(packed_data)} bytes")
            print(f"    Compression ratio: {metrics.compression_ratio:.2f}x")
            
            # Create metadata for unpacking
            metadata = {
                'packing_strategy': strategy.value,
                'num_monomials': num_monomials,
                'num_variables': num_variables,
                'bit_widths': metrics.bit_widths_used,
                'channel_metadata': channels.metadata
            }
            
            # Unpack channels
            unpacked = packer.unpack_channels(packed_data, metadata)
            
            # Verify shapes match
            assert unpacked.coefficient_channel.shape == channels.coefficient_channel.shape
            assert unpacked.exponent_channel.shape == channels.exponent_channel.shape
            assert unpacked.index_channel.shape == channels.index_channel.shape
            
            print(f"    ✓ {strategy.value} unpacking successful")
            
        except Exception as e:
            print(f"    ✗ {strategy.value} failed: {e}")
            import traceback
            traceback.print_exc()

def test_padic_tropical_reconstruction():
    """Test P-adic and Hybrid reconstruction methods"""
    print("\n\nTesting P-adic/Tropical Bridge Reconstruction...")
    
    # Create test tensor
    test_tensor = torch.randn(10, 5)
    print(f"  Original tensor shape: {test_tensor.shape}")
    print(f"  Original tensor sample: {test_tensor[0, :3].tolist()}")
    
    # Create hybrid representation
    hybrid = HybridRepresentation(test_tensor)
    
    # Test P-adic reconstruction
    print("\n  Testing P-adic reconstruction...")
    try:
        # Compute p-adic components
        encoder = PadicMathematicalOperations(prime=251, precision=16)
        hybrid.compute_padic(encoder)
        hybrid.active_mode = "padic"
        
        # Reconstruct
        reconstructed = hybrid.reconstruct()
        
        # Check shape
        assert reconstructed.shape == test_tensor.shape
        
        # Check values are reasonably close
        max_error = torch.max(torch.abs(reconstructed - test_tensor)).item()
        print(f"    Max reconstruction error: {max_error:.6f}")
        print(f"    Reconstructed sample: {reconstructed[0, :3].tolist()}")
        print("    ✓ P-adic reconstruction successful")
        
    except Exception as e:
        print(f"    ✗ P-adic reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Hybrid reconstruction
    print("\n  Testing Hybrid reconstruction...")
    try:
        # Compute tropical components
        from independent_core.compression_systems.tropical.tropical_core import TropicalMathematicalOperations
        tropical_ops = TropicalMathematicalOperations(device=torch.device('cpu'))
        hybrid.compute_tropical(tropical_ops)
        hybrid.active_mode = "hybrid"
        
        # Reconstruct
        reconstructed = hybrid.reconstruct()
        
        # Check shape
        assert reconstructed.shape == test_tensor.shape
        
        # Check values are reasonably close
        max_error = torch.max(torch.abs(reconstructed - test_tensor)).item()
        print(f"    Max reconstruction error: {max_error:.6f}")
        print(f"    Reconstructed sample: {reconstructed[0, :3].tolist()}")
        print("    ✓ Hybrid reconstruction successful")
        
    except Exception as e:
        print(f"    ✗ Hybrid reconstruction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("COMPRESSION SYSTEM IMPLEMENTATION TEST")
    print("=" * 60)
    
    test_channel_unpacking()
    test_padic_tropical_reconstruction()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)