#!/usr/bin/env python3
"""Debug script to identify which encoding method is failing"""

import torch
from .entropy_bridge import EntropyPAdicBridge

def debug_methods():
    bridge = EntropyPAdicBridge(prime=7)
    tensor = torch.tensor([0, 1, 2, 0, 1, 2] * 10)
    
    methods = ["huffman", "arithmetic", "hybrid"]
    
    for method in methods:
        print(f"\n=== Testing {method} method ===")
        try:
            # Encode
            compressed, metadata = bridge.encode_padic_tensor(tensor, force_method=method)
            print(f"Encoded successfully with method: {metadata['encoding_method']}")
            print(f"Original tensor shape: {tensor.shape}")
            print(f"Compressed size: {len(compressed)} bytes")
            
            # Decode
            decoded = bridge.decode_padic_tensor(compressed, metadata)
            print(f"Decoded tensor shape: {decoded.shape}")
            print(f"Original tensor dtype: {tensor.dtype}")
            print(f"Decoded tensor dtype: {decoded.dtype}")
            
            # Compare
            equal = torch.equal(tensor.long(), decoded.long())
            print(f"Tensors equal: {equal}")
            
            if not equal:
                print(f"Original tensor (first 20): {tensor[:20]}")
                print(f"Decoded tensor (first 20): {decoded[:20]}")
                diff_indices = (tensor != decoded).nonzero()
                print(f"Differences at indices: {diff_indices[:10]}")
                
        except Exception as e:
            print(f"Error with {method}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_methods()