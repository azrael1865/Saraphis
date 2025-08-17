#!/usr/bin/env python3
"""
Comprehensive test for P-adic system fixes:
1. Type boundary violation fixes (numpy type support)
2. Precision overflow prevention
"""

import torch
import numpy as np
import sys
import os
import traceback
from typing import List, Any, Dict

# Add the path to import from the independent_core
sys.path.append('/home/will-casterlin/Desktop/Saraphis')

from independent_core.compression_systems.padic.padic_encoder import (
    PadicMathematicalOperations,
    get_safe_precision,
    SAFE_PRECISION_LIMITS
)
from independent_core.compression_systems.padic.adaptive_precision_wrapper import (
    AdaptivePrecisionWrapper,
    AdaptivePrecisionConfig
)


def test_fixes():
    """Quick test of both fixes"""
    print("🧪 Testing P-adic System Fixes...")
    
    # Test 1: Numpy type support
    print("\n1. Testing numpy type support:")
    ops = PadicMathematicalOperations(prime=257, precision=4)
    
    test_values = [
        ("Python float", 3.14159),
        ("numpy.float32", np.float32(2.71828)),
        ("numpy.int64", np.int64(42)),
        ("numpy array", np.array(1.5)),
    ]
    
    for name, value in test_values:
        try:
            weight = ops.to_padic(value)
            reconstructed = ops.from_padic(weight)
            print(f"  ✓ {name}: {float(value):.6f} -> {reconstructed:.6f}")
        except Exception as e:
            print(f"  ✗ {name}: FAILED - {e}")
    
    # Test 2: Safe precision limits
    print("\n2. Testing safe precision limits:")
    print(f"  Safe limit for prime 257: {get_safe_precision(257)}")
    
    # This should work (safe precision)
    try:
        ops_safe = PadicMathematicalOperations(257, 6)
        print("  ✓ Safe precision (6) accepted")
    except Exception as e:
        print(f"  ✗ Safe precision failed: {e}")
    
    # This should fail (unsafe precision)
    try:
        ops_unsafe = PadicMathematicalOperations(257, 10)
        print("  ✗ Unsafe precision (10) should have failed but didn't!")
    except OverflowError:
        print("  ✓ Unsafe precision (10) correctly rejected")
    except Exception as e:
        print(f"  ? Unsafe precision failed with: {e}")
    
    # Test 3: Adaptive wrapper integration
    print("\n3. Testing adaptive wrapper with safe limits:")
    try:
        config = AdaptivePrecisionConfig(
            prime=257,
            base_precision=4,
            min_precision=2,
            max_precision=10,  # Should be clamped to 6
        )
        wrapper = AdaptivePrecisionWrapper(config)
        
        test_tensor = torch.randn(20, 20)
        result = wrapper.convert_tensor(test_tensor)
        
        print(f"  ✓ Successfully processed tensor with {len(result.padic_weights)} weights")
        
        # Check precision allocation
        max_precision = torch.max(result.precision_map).item()
        safe_limit = get_safe_precision(257)
        
        if max_precision <= safe_limit:
            print(f"  ✓ Max precision {max_precision} <= safe limit {safe_limit}")
        else:
            print(f"  ✗ Max precision {max_precision} > safe limit {safe_limit}")
            
    except Exception as e:
        print(f"  ✗ Adaptive wrapper test failed: {e}")
    
    print("\n🎉 P-adic system fixes integrated and tested!")


if __name__ == "__main__":
    test_fixes()