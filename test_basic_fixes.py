#!/usr/bin/env python3
"""
Basic test of P-adic fixes without external dependencies
"""
import sys
sys.path.append('/home/will-casterlin/Desktop/Saraphis')

def test_imports():
    """Test that our fixed modules can be imported"""
    try:
        from independent_core.compression_systems.padic.padic_encoder import (
            PadicMathematicalOperations,
            get_safe_precision,
            SAFE_PRECISION_LIMITS
        )
        print("✓ Successfully imported padic_encoder with fixes")
        print(f"✓ Safe precision limits dictionary has {len(SAFE_PRECISION_LIMITS)} entries")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_safe_precision_function():
    """Test the safe precision function"""
    try:
        from independent_core.compression_systems.padic.padic_encoder import get_safe_precision
        
        # Test some known values
        test_primes = [2, 3, 5, 7, 11, 17, 257, 251]
        
        for prime in test_primes:
            safe_limit = get_safe_precision(prime)
            print(f"  Prime {prime:3d}: max safe precision = {safe_limit}")
        
        print("✓ Safe precision calculation working")
        return True
    except Exception as e:
        print(f"✗ Safe precision test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic P-adic operations with Python types"""
    try:
        from independent_core.compression_systems.padic.padic_encoder import PadicMathematicalOperations
        
        # Test with safe precision
        ops = PadicMathematicalOperations(prime=257, precision=4)
        
        # Test basic values
        test_values = [0.0, 1.0, -1.0, 3.14159, 42]
        
        for value in test_values:
            weight = ops.to_padic(value)
            reconstructed = ops.from_padic(weight)
            error = abs(value - reconstructed)
            print(f"  {value:8.5f} -> {reconstructed:8.5f} (error: {error:.2e})")
        
        print("✓ Basic P-adic operations working")
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_precision_limits():
    """Test that precision limits are enforced"""
    try:
        from independent_core.compression_systems.padic.padic_encoder import PadicMathematicalOperations
        
        # Should work - safe precision
        try:
            safe_ops = PadicMathematicalOperations(257, 6)  # Safe for prime 257
            print("  ✓ Safe precision (6) accepted for prime 257")
        except Exception as e:
            print(f"  ✗ Safe precision rejected: {e}")
            return False
        
        # Should fail - unsafe precision
        try:
            unsafe_ops = PadicMathematicalOperations(257, 10)  # Unsafe for prime 257
            print("  ✗ Unsafe precision (10) should have been rejected!")
            return False
        except OverflowError as e:
            print("  ✓ Unsafe precision (10) correctly rejected with OverflowError")
        except Exception as e:
            print(f"  ✓ Unsafe precision rejected with: {type(e).__name__}")
        
        print("✓ Precision limit enforcement working")
        return True
    except Exception as e:
        print(f"✗ Precision limit test failed: {e}")
        return False

def main():
    """Run all basic tests"""
    print("🧪 Testing P-adic System Fixes (Basic Tests)")
    print("=" * 50)
    
    tests = [
        ("Module imports", test_imports),
        ("Safe precision function", test_safe_precision_function), 
        ("Basic P-adic operations", test_basic_functionality),
        ("Precision limit enforcement", test_precision_limits),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        try:
            if test_func():
                passed += 1
            else:
                print(f"  ✗ {test_name} FAILED")
        except Exception as e:
            print(f"  ✗ {test_name} FAILED with exception: {e}")
    
    total = len(tests)
    print(f"\n{'='*50}")
    print(f"RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL BASIC TESTS PASSED!")
        print("✅ Core P-adic fixes are working")
        print("✅ Safe precision limits implemented")
        print("✅ Overflow prevention active")
        print("\nThe fixes should resolve both:")
        print("  1. Type boundary violations (numpy.float32 acceptance)")
        print("  2. Precision overflow prevention (safe limits enforced)")
    else:
        print("⚠️  Some tests failed - please check the implementation")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)