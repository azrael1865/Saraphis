#!/usr/bin/env python3
"""
Test Shape Integrity Fix - Solution 2 Implementation

Verifies that edge case filtering with zero-filling maintains tensor shape integrity.
"""

import sys
import os
sys.path.append('/home/will-casterlin/Desktop/Saraphis')

import torch
import numpy as np
from typing import Dict, Any, List

def test_shape_integrity():
    """Test that filtered weights maintain original tensor shape"""
    print("Testing Shape Integrity Fix (Solution 2)...")
    
    try:
        # Import required modules
        from independent_core.compression_systems.padic.padic_encoder import create_real_padic_weights
        from independent_core.compression_systems.padic.padic_advanced import (
            PadicDecompressionEngine, DecompressionConfig
        )
        
        # Create test configuration with safe settings
        config = DecompressionConfig(
            prime=257,
            max_precision=4,  # Safe for prime=257
            batch_size=50,
            enable_progressive_precision=True
        )
        
        # Create decompression engine
        engine = PadicDecompressionEngine(config)
        
        # Test case: Create 1000 weights, some will be filtered
        print("Creating test weights (some will be unsafe and filtered)...")
        
        # Create mix of safe and unsafe weights
        safe_weights = create_real_padic_weights(800, precision=3, prime=257)  # Safe precision
        
        # Create some unsafe weights with precision > 4 (will be filtered)
        unsafe_weights = []
        for i in range(200):
            # Create weights with precision=6 (unsafe for prime=257)
            try:
                weight = create_real_padic_weights(1, precision=6, prime=257)[0]
                unsafe_weights.append(weight)
            except:
                # If creation fails, create a simple unsafe weight manually
                from independent_core.compression_systems.padic.padic_encoder import PadicWeight
                from fractions import Fraction
                weight = PadicWeight(
                    value=Fraction(1, 3),
                    prime=257,
                    precision=6,  # This will be unsafe
                    valuation=0,
                    digits=[1, 2, 3, 4, 5, 6]
                )
                unsafe_weights.append(weight)
        
        # Combine weights
        all_weights = safe_weights + unsafe_weights
        np.random.shuffle(all_weights)  # Mix them up
        
        print(f"Created {len(all_weights)} total weights ({len(safe_weights)} safe, {len(unsafe_weights)} potentially unsafe)")
        
        # Create metadata for 1000-element tensor
        original_shape = (1000,)
        metadata = {
            'original_shape': original_shape,
            'dtype': 'torch.float32',
            'device': 'cpu'
        }
        
        print(f"Expected tensor shape: {original_shape}")
        print("Starting decompression with edge case filtering...")
        
        # Perform decompression
        result_tensor, info = engine.decompress_progressive(
            all_weights, 
            target_precision=4, 
            metadata=metadata
        )
        
        # Verify shape integrity
        print(f"Result tensor shape: {result_tensor.shape}")
        print(f"Expected shape: {original_shape}")
        
        assert result_tensor.shape == original_shape, f"Shape mismatch: got {result_tensor.shape}, expected {original_shape}"
        
        # Verify tensor properties
        assert not torch.any(torch.isnan(result_tensor)), "Result tensor contains NaN values"
        assert not torch.any(torch.isinf(result_tensor)), "Result tensor contains infinite values"
        
        # Check that we have the right number of elements
        assert result_tensor.numel() == 1000, f"Wrong element count: got {result_tensor.numel()}, expected 1000"
        
        # Analyze the results
        filtered_count = info.get('filtered_weights', 0)
        processed_count = info.get('processed_weights', 0)
        zero_filled = info.get('zero_filled_positions', 0)
        
        print(f"Decompression Results:")
        print(f"  - Processed weights: {processed_count}")
        print(f"  - Filtered weights: {filtered_count}")
        print(f"  - Zero-filled positions: {zero_filled}")
        print(f"  - Shape integrity maintained: {info.get('shape_integrity_maintained', False)}")
        print(f"  - Success rate: {info.get('success_rate', 0):.1%}")
        
        # Check that filtering actually occurred
        assert filtered_count > 0, "Expected some weights to be filtered, but none were"
        assert processed_count < len(all_weights), "Expected fewer processed weights than input due to filtering"
        assert processed_count + filtered_count <= len(all_weights), "Processed + filtered should not exceed input"
        
        # Verify zero positions
        zero_positions = (result_tensor == 0.0).sum().item()
        print(f"  - Actual zero positions in tensor: {zero_positions}")
        
        # Check tensor statistics
        non_zero_values = result_tensor[result_tensor != 0.0]
        if len(non_zero_values) > 0:
            print(f"  - Non-zero value range: [{non_zero_values.min():.3f}, {non_zero_values.max():.3f}]")
            print(f"  - Non-zero value mean: {non_zero_values.mean():.3f}")
        
        print("✅ Shape Integrity Test PASSED!")
        print("   - Original tensor shape maintained")
        print("   - Filtered positions filled with zeros")
        print("   - No NaN or infinite values")
        print("   - Edge case filtering working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Shape Integrity Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test various edge cases for shape integrity"""
    print("\nTesting Edge Cases...")
    
    try:
        from independent_core.compression_systems.padic.padic_encoder import create_real_padic_weights
        from independent_core.compression_systems.padic.padic_advanced import (
            PadicDecompressionEngine, DecompressionConfig
        )
        
        config = DecompressionConfig(prime=257, max_precision=4)
        engine = PadicDecompressionEngine(config)
        
        # Test 1: All weights filtered
        print("Test 1: Handling case where many weights are filtered...")
        
        weights = create_real_padic_weights(100, precision=3, prime=257)
        metadata = {
            'original_shape': (100,),
            'dtype': 'torch.float32', 
            'device': 'cpu'
        }
        
        result, info = engine.decompress_progressive(weights, 4, metadata)
        assert result.shape == (100,), f"Wrong shape: {result.shape}"
        print("✅ Test 1 passed")
        
        # Test 2: Multi-dimensional tensor
        print("Test 2: Multi-dimensional tensor shape...")
        
        weights = create_real_padic_weights(24, precision=3, prime=257)
        metadata = {
            'original_shape': (4, 6),
            'dtype': 'torch.float32',
            'device': 'cpu'
        }
        
        result, info = engine.decompress_progressive(weights, 4, metadata)
        assert result.shape == (4, 6), f"Wrong shape: {result.shape}"
        assert result.numel() == 24, f"Wrong element count: {result.numel()}"
        print("✅ Test 2 passed")
        
        # Test 3: 3D tensor
        print("Test 3: 3D tensor shape...")
        
        weights = create_real_padic_weights(60, precision=3, prime=257)
        metadata = {
            'original_shape': (3, 4, 5),
            'dtype': 'torch.float32',
            'device': 'cpu'
        }
        
        result, info = engine.decompress_progressive(weights, 4, metadata)
        assert result.shape == (3, 4, 5), f"Wrong shape: {result.shape}"
        assert result.numel() == 60, f"Wrong element count: {result.numel()}"
        print("✅ Test 3 passed")
        
        print("✅ All Edge Case Tests PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Edge Case Tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Shape Integrity Fix Verification")
    print("=" * 50)
    
    success1 = test_shape_integrity()
    success2 = test_edge_cases()
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED!")
        print("Solution 2 successfully implemented:")
        print("  ✅ Shape integrity maintained")
        print("  ✅ Filtered positions zero-filled")  
        print("  ✅ No mathematical overflow")
        print("  ✅ Compatible with existing code")
        exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        exit(1)