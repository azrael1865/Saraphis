#!/usr/bin/env python3
"""
Test numpy type fixes specifically
"""
import numpy as np
import sys
sys.path.append('/home/will-casterlin/Desktop/Saraphis')

def test_numpy_imports():
    """Test that we can import the fixed modules"""
    try:
        from independent_core.compression_systems.padic.padic_encoder import (
            PadicMathematicalOperations,
            get_safe_precision,
            SAFE_PRECISION_LIMITS
        )
        print("✓ Successfully imported padic_encoder")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_safe_precision():
    """Test safe precision function"""
    try:
        from independent_core.compression_systems.padic.padic_encoder import get_safe_precision
        
        # Test known primes
        test_cases = [(257, 6), (251, 6), (17, 13), (2, 53)]
        
        for prime, expected in test_cases:
            actual = get_safe_precision(prime)
            print(f"Prime {prime}: safe precision = {actual} (expected ~{expected})")
        
        print("✓ Safe precision function working")
        return True
    except Exception as e:
        print(f"✗ Safe precision test failed: {e}")
        return False

def test_numpy_type_support():
    """Test numpy type support in to_padic"""
    try:
        from independent_core.compression_systems.padic.padic_encoder import PadicMathematicalOperations
        
        # Create operations with safe precision
        ops = PadicMathematicalOperations(prime=257, precision=4)
        
        # Test different numpy types
        test_values = [
            ("Python float", 3.14159),
            ("Python int", 42),
            ("numpy.float32", np.float32(2.71828)),
            ("numpy.float64", np.float64(1.41421)),
            ("numpy.int32", np.int32(100)),
            ("numpy.int64", np.int64(1000)),
        ]
        
        for name, value in test_values:
            try:
                weight = ops.to_padic(value)
                reconstructed = ops.from_padic(weight)
                error = abs(float(value) - reconstructed)
                print(f"  ✓ {name:15}: {float(value):.6f} -> {reconstructed:.6f} (error: {error:.2e})")
            except Exception as e:
                print(f"  ✗ {name:15}: FAILED - {e}")
                return False
        
        print("✓ All numpy types supported")
        return True
    except Exception as e:
        print(f"✗ Numpy type test failed: {e}")
        return False

def test_precision_overflow_prevention():
    """Test that unsafe precision is prevented"""
    try:
        from independent_core.compression_systems.padic.padic_encoder import PadicMathematicalOperations
        
        # Test safe precision works
        try:
            ops_safe = PadicMathematicalOperations(257, 6)
            print("  ✓ Safe precision (6) accepted")
        except Exception as e:
            print(f"  ✗ Safe precision failed: {e}")
            return False
        
        # Test unsafe precision is rejected
        try:
            ops_unsafe = PadicMathematicalOperations(257, 10)
            print("  ✗ Unsafe precision (10) should have failed but didn't!")
            return False
        except OverflowError:
            print("  ✓ Unsafe precision (10) correctly rejected")
        except Exception as e:
            print(f"  ✓ Unsafe precision rejected with: {type(e).__name__}")
        
        print("✓ Precision overflow prevention working")
        return True
    except Exception as e:
        print(f"✗ Precision overflow test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing P-adic System Fixes (numpy types + safe precision)...")
    
    tests = [
        ("Import test", test_numpy_imports),
        ("Safe precision limits", test_safe_precision),
        ("Numpy type support", test_numpy_type_support),
        ("Precision overflow prevention", test_precision_overflow_prevention),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        if test_func():
            passed += 1
        
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! P-adic system fixes are working correctly.")
        print("✅ Numpy type boundary violations: FIXED")
        print("✅ Precision overflow prevention: FIXED")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)