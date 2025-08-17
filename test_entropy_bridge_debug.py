#!/usr/bin/env python3
"""Debug script for entropy bridge issue"""

import torch
import sys
import os
import json

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from independent_core.compression_systems.padic.entropy_bridge import (
    EntropyPAdicBridge,
    EntropyBridgeConfig
)

def test_entropy_bridge():
    """Test entropy bridge directly"""
    print("Testing entropy bridge...")
    
    # Create config
    config = EntropyBridgeConfig()
    
    # Initialize bridge with prime
    bridge = EntropyPAdicBridge(7, config)
    
    # Create test tensor
    test_tensor = torch.randint(0, 7, (100,), dtype=torch.long)
    print(f"\nTest tensor shape: {test_tensor.shape}")
    print(f"Test tensor values (first 10): {test_tensor[:10].tolist()}")
    
    # Test encoding
    print("\n1. Testing encoding...")
    compressed, metadata = bridge.encode_padic_tensor(test_tensor)
    print(f"✓ Encoding successful")
    print(f"  Compressed size: {len(compressed)} bytes")
    print(f"  Metadata keys: {list(metadata.keys())}")
    
    # Check encoding_metadata
    if 'encoding_metadata' in metadata:
        print(f"  Encoding metadata keys: {list(metadata['encoding_metadata'].keys())}")
        if 'frequency_table' in metadata['encoding_metadata']:
            print(f"  ✓ Frequency table found in encoding_metadata")
        else:
            print(f"  ❌ Frequency table NOT found in encoding_metadata")
    else:
        print(f"  ❌ encoding_metadata not found")
    
    # Pretty print metadata structure
    print("\n2. Full metadata structure:")
    print(json.dumps({k: (v if not isinstance(v, (bytes, torch.Tensor)) else f"<{type(v).__name__}>") 
                      for k, v in metadata.items()}, indent=2))
    
    # Test decoding
    print("\n3. Testing decoding...")
    try:
        decoded_tensor = bridge.decode_padic_tensor(compressed, metadata)
        print(f"✓ Decoding successful")
        print(f"  Decoded shape: {decoded_tensor.shape}")
        print(f"  Decoded values (first 10): {decoded_tensor[:10].tolist()}")
        
        # Verify reconstruction
        if torch.equal(test_tensor, decoded_tensor):
            print(f"✓ Perfect reconstruction!")
        else:
            print(f"❌ Reconstruction mismatch")
            
    except Exception as e:
        print(f"❌ Decoding failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_entropy_bridge()