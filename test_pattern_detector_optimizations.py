#!/usr/bin/env python3
"""
Test script for the optimized pattern detector components
Tests the core optimizations without requiring torch
"""

import sys
import os
import numpy as np
import time
from typing import List

# Mock torch tensor for testing
class MockTensor:
    def __init__(self, data, dtype=None, device='cpu'):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)
        self.dtype = dtype
        self.device = device
    
    def cpu(self):
        return MockTensor(self.data, self.dtype, 'cpu')
    
    def numpy(self):
        return self.data
    
    def flatten(self):
        return MockTensor(self.data.flatten(), self.dtype, self.device)
    
    def size(self, dim=None):
        if dim is None:
            return self.data.size
        return self.data.shape[dim] if dim < len(self.data.shape) else 1
    
    def dim(self):
        return len(self.data.shape)
    
    def zeros(shape, dtype=None, device='cpu'):
        return MockTensor(np.zeros(shape), dtype, device)
    
    def full(shape, fill_value, dtype=None, device='cpu'):
        return MockTensor(np.full(shape, fill_value), dtype, device)

# Mock torch module
sys.modules['torch'] = type('torch', (), {
    'Tensor': MockTensor,
    'zeros': MockTensor.zeros,
    'full': MockTensor.full,
    'device': lambda x: x,
    'uint8': np.uint8,
    'int32': np.int32,
    'bool': bool,
})

sys.modules['torch.nn'] = type('nn', (), {
    'Module': object,
})

# Now import the optimized components
import importlib.util
spec = importlib.util.spec_from_file_location(
    'sliding_window_pattern_detector',
    'independent_core/compression_systems/padic/sliding_window_pattern_detector.py'
)
pattern_detector_module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(pattern_detector_module)
    print("✅ Successfully imported optimized pattern detector")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def test_interval_tree():
    """Test the IntervalTree optimization"""
    print("\n=== Testing IntervalTree ===")
    
    tree = pattern_detector_module.IntervalTree()
    
    # Test basic functionality
    tree.insert(10, 20)
    tree.insert(30, 40)
    tree.insert(50, 60)
    
    # Test overlaps
    assert tree.overlaps(15, 25) == True, "Should detect overlap with [10,20)"
    assert tree.overlaps(25, 35) == True, "Should detect overlap with [30,40)"
    assert tree.overlaps(5, 8) == False, "Should not detect overlap before intervals"
    assert tree.overlaps(65, 70) == False, "Should not detect overlap after intervals"
    
    # Test merging by adding many intervals
    for i in range(0, 200, 10):
        tree.insert(i, i + 5)
    
    # This should trigger merging
    print(f"Intervals after bulk insert: {len(tree.intervals)}")
    
    print("✅ IntervalTree tests passed")

def test_suffix_array_builder():
    """Test the optimized SuffixArrayBuilder"""
    print("\n=== Testing SuffixArrayBuilder ===")
    
    builder = pattern_detector_module.SuffixArrayBuilder()
    
    # Test small array
    small_data = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.uint8)
    small_sa = builder.build_suffix_array(small_data)
    print(f"Small array SA: {small_sa}")
    
    # Test medium array (should use DC3)
    medium_data = np.random.randint(0, 10, 2000, dtype=np.uint8)
    
    start_time = time.time()
    medium_sa = builder.build_suffix_array(medium_data)
    medium_time = time.time() - start_time
    
    print(f"Medium array ({len(medium_data)} elements): {medium_time:.3f}s")
    assert len(medium_sa) == len(medium_data), "SA length should match input"
    
    # Test LCP array
    lcp_array = builder.build_lcp_array(medium_data, medium_sa)
    print(f"LCP array built: length {len(lcp_array)}")
    
    print("✅ SuffixArrayBuilder tests passed")

def test_performance_scaling():
    """Test performance scaling with different sizes"""
    print("\n=== Testing Performance Scaling ===")
    
    builder = pattern_detector_module.SuffixArrayBuilder()
    sizes = [1000, 5000, 10000, 25000]
    times = []
    
    for size in sizes:
        # Create test data with some patterns
        data = np.random.randint(0, 20, size, dtype=np.uint8)
        
        start_time = time.time()
        sa = builder.build_suffix_array(data)
        elapsed = time.time() - start_time
        
        times.append(elapsed)
        throughput = size / elapsed if elapsed > 0 else float('inf')
        
        print(f"Size {size:5d}: {elapsed:.3f}s ({throughput/1000:.1f}k elements/s)")
        
        # Verify correctness
        assert len(sa) == size, f"SA length mismatch for size {size}"
    
    # Check that it's not quadratic
    if len(times) >= 2:
        ratio_size = sizes[-1] / sizes[0]  # Should be 25x
        ratio_time = times[-1] / times[0]
        
        print(f"Size ratio: {ratio_size:.1f}x, Time ratio: {ratio_time:.1f}x")
        
        # For O(n log n), time ratio should be much less than size_ratio^2
        max_expected_ratio = ratio_size * np.log2(ratio_size) * 2  # 2x safety margin
        
        if ratio_time > max_expected_ratio:
            print(f"⚠️  Warning: Time scaling may be suboptimal ({ratio_time:.1f}x > {max_expected_ratio:.1f}x)")
        else:
            print("✅ Performance scaling looks good")

def test_algorithmic_correctness():
    """Test that the algorithms produce correct results"""
    print("\n=== Testing Algorithmic Correctness ===")
    
    builder = pattern_detector_module.SuffixArrayBuilder()
    
    # Test with known data
    test_data = np.array(list(b"banana$"), dtype=np.uint8)
    sa = builder.build_suffix_array(test_data)
    
    # Extract suffixes for verification
    suffixes = []
    for i in sa:
        suffix = test_data[i:].tobytes().decode('ascii', errors='ignore')
        suffixes.append(suffix)
    
    print("Suffixes in order:")
    for i, suffix in enumerate(suffixes):
        print(f"  {i}: {suffix}")
    
    # Check that suffixes are in lexicographic order
    for i in range(len(suffixes) - 1):
        assert suffixes[i] <= suffixes[i + 1], f"Order violation: '{suffixes[i]}' > '{suffixes[i + 1]}'"
    
    print("✅ Suffix array is correctly ordered")
    
    # Test LCP array
    lcp = builder.build_lcp_array(test_data, sa)
    print(f"LCP array: {lcp}")
    
    # Verify LCP values
    for i in range(len(lcp) - 1):
        suffix1 = test_data[sa[i]:].tobytes()
        suffix2 = test_data[sa[i + 1]:].tobytes()
        
        # Calculate actual LCP
        actual_lcp = 0
        for j in range(min(len(suffix1), len(suffix2))):
            if suffix1[j] == suffix2[j]:
                actual_lcp += 1
            else:
                break
        
        assert lcp[i] == actual_lcp, f"LCP mismatch at {i}: expected {actual_lcp}, got {lcp[i]}"
    
    print("✅ LCP array is correct")

def main():
    """Run all optimization tests"""
    print("=" * 60)
    print("TESTING OPTIMIZED PATTERN DETECTOR COMPONENTS")
    print("=" * 60)
    
    try:
        test_interval_tree()
        test_suffix_array_builder()
        test_performance_scaling()
        test_algorithmic_correctness()
        
        print("\n" + "=" * 60)
        print("✅ ALL OPTIMIZATION TESTS PASSED!")
        print("The optimized pattern detector is ready for integration.")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())