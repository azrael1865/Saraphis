#!/usr/bin/env python3
"""
Debug script to identify the root cause of GPU reconstruction returning None
"""

import sys
import os
sys.path.append('/home/will-casterlin/Desktop/Saraphis')

import torch
import numpy as np
from independent_core.compression_systems.padic.safe_reconstruction import (
    SafePadicReconstructor, ReconstructionConfig, ReconstructionMethod, PadicWeight
)

def debug_gpu_reconstruction():
    """Debug GPU reconstruction to find why it returns None"""
    print("=== GPU Reconstruction Debug ===")
    
    # Check GPU availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    
    # Create safe configuration for debugging
    config = ReconstructionConfig(
        prime=257,
        max_safe_precision=6,
        method=ReconstructionMethod.HYBRID,
        use_gpu=True,
        overflow_threshold=1e12
    )
    
    print(f"Config: prime={config.prime}, max_safe_precision={config.max_safe_precision}")
    
    # Create reconstructor
    try:
        reconstructor = SafePadicReconstructor(config)
        print("✓ SafePadicReconstructor created successfully")
    except Exception as e:
        print(f"✗ Failed to create SafePadicReconstructor: {e}")
        return
    
    # Create test weights with SMALL values to avoid overflow
    test_weights = []
    for i in range(3):
        weight = PadicWeight(
            digits=[1, 2, 0, 0, 0, 0],  # SMALL values - only first 2 digits non-zero
            valuation=0,
            precision=6,
            prime=257
        )
        test_weights.append(weight)
    
    print(f"Created {len(test_weights)} test weights")
    
    # Test CPU reconstruction first
    print("\n--- Testing CPU Reconstruction ---")
    try:
        cpu_results = reconstructor.reconstruct_batch_cpu(test_weights, target_precision=6)
        print(f"✓ CPU reconstruction successful: {cpu_results}")
        print(f"  Shape: {cpu_results.shape}, dtype: {cpu_results.dtype}")
        print(f"  Values: {cpu_results}")
    except Exception as e:
        print(f"✗ CPU reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test GPU reconstruction
    print("\n--- Testing GPU Reconstruction ---")
    try:
        gpu_results = reconstructor.reconstruct_batch_gpu(test_weights, target_precision=6)
        print(f"✓ GPU reconstruction result type: {type(gpu_results)}")
        if gpu_results is None:
            print("✗ GPU reconstruction returned None - This is the root cause!")
        else:
            print(f"✓ GPU reconstruction successful: {gpu_results}")
            print(f"  Shape: {gpu_results.shape}, dtype: {gpu_results.dtype}")
            print(f"  Device: {gpu_results.device}")
            print(f"  Values: {gpu_results}")
    except Exception as e:
        print(f"✗ GPU reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test individual components
    print("\n--- Testing GPU Kernel Components ---")
    if torch.cuda.is_available():
        try:
            # Test GPU kernel directly
            batch_size = len(test_weights)
            max_precision = 6
            
            # Create GPU tensors manually
            digits_tensor = torch.zeros((batch_size, max_precision), 
                                      dtype=torch.int32, device='cuda')
            valuations = torch.zeros(batch_size, dtype=torch.int32, device='cuda')
            
            # Fill tensors
            for i, weight in enumerate(test_weights):
                eff_prec = min(max_precision, len(weight.digits))
                digits_tensor[i, :eff_prec] = torch.tensor(
                    weight.digits[:eff_prec], dtype=torch.int32
                )
                valuations[i] = weight.valuation
            
            print(f"✓ Created GPU tensors:")
            print(f"  digits_tensor: {digits_tensor.shape}, device: {digits_tensor.device}")
            print(f"  valuations: {valuations.shape}, device: {valuations.device}")
            
            # Test GPU kernel
            kernel_results = reconstructor._gpu_reconstruct_kernel(
                digits_tensor, valuations, max_precision
            )
            
            if kernel_results is None:
                print("✗ GPU kernel returned None - This is the root cause!")
            else:
                print(f"✓ GPU kernel successful: {kernel_results}")
                print(f"  Shape: {kernel_results.shape}, dtype: {kernel_results.dtype}")
                print(f"  Values: {kernel_results}")
            
        except Exception as e:
            print(f"✗ GPU kernel test failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_gpu_reconstruction()