#!/usr/bin/env python3
"""
Reproduce the exact "processed_data is None" error
"""

import sys
import os
sys.path.append('/home/will-casterlin/Desktop/Saraphis')

import torch
import numpy as np
from independent_core.compression_systems.padic.padic_advanced import PadicDecompressionEngine, GPUDecompressionConfig
from independent_core.compression_systems.padic.padic_encoder import PadicWeight
from fractions import Fraction

def reproduce_none_error():
    """Reproduce the exact error scenario"""
    print("=== Reproducing 'processed_data is None' Error ===")
    
    # Create the engine configuration
    config = GPUDecompressionConfig(
        batch_size=8,
        memory_pool_size_mb=1024,  # 1GB
        enable_async_transfer=True,
        num_streams=4
    )
    
    try:
        engine = PadicDecompressionEngine(config, prime=257)
        print("✓ Decompression engine created successfully")
    except Exception as e:
        print(f"✗ Failed to create engine: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create problematic weights that might cause overflow
    problematic_weights = []
    for i in range(10):
        # These weights have high values that cause overflow
        weight = PadicWeight(
            value=Fraction(123, 1),  # Some reasonable value
            prime=257,
            precision=6,
            valuation=0,
            digits=[1, 2, 3, 4, 5, 6]  # All 6 digits non-zero = OVERFLOW!
        )
        problematic_weights.append(weight)
    
    print(f"Created {len(problematic_weights)} problematic weights")
    
    # Try to decompress - this should trigger the error
    print("\n--- Attempting decompression that should fail ---")
    try:
        # Call the internal batch decompress method directly
        stream = torch.cuda.Stream()
        results = engine._decompress_batch_gpu(problematic_weights, [6], stream, 0)
        print(f"✗ Decompression succeeded unexpectedly: {results}")
    except Exception as e:
        print(f"✓ Expected error occurred: {e}")
        
        # Check if this is the specific error we're looking for
        if "processed_data is None" in str(e):
            print("✓ FOUND THE ROOT CAUSE: processed_data is None error!")
        elif "Overflow" in str(e) or "overflow" in str(e):
            print("✓ FOUND RELATED: Overflow error causing the issue")
        else:
            print(f"? Different error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    reproduce_none_error()