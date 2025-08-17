#!/usr/bin/env python3

import numpy as np
import torch
import traceback
from independent_core.compression_systems.padic.padic_compressor import PadicCompressionSystem

def debug_padic_compression():
    """Debug the exact error in P-adic compression"""
    try:
        # Create a simple tensor
        test_tensor = torch.randn(100, 100).cuda()
        
        # Initialize P-adic compressor with defaults
        compressor = PadicCompressionSystem()
        
        print("Starting P-adic compression...")
        result = compressor.compress(test_tensor)
        print("Success!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    debug_padic_compression()