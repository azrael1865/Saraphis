#!/usr/bin/env python3
"""
Test function to verify the data type compatibility fix
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add Saraphis to path
SARAPHIS_PATH = Path("/home/will-casterlin/Desktop/Saraphis")
sys.path.insert(0, str(SARAPHIS_PATH))

def test_dtype_compatibility():
    """Test that data types flow correctly through the pipeline"""
    
    print("Testing data type compatibility fix...")
    
    # Create test data similar to p-adic digits
    test_prime = 257
    test_data = torch.randint(0, test_prime, (1000,), dtype=torch.int32)
    
    print(f"Original data dtype: {test_data.dtype}")
    print(f"Data range: [{test_data.min()}, {test_data.max()}]")
    
    # Test pattern detector
    from independent_core.compression_systems.padic.sliding_window_pattern_detector import SlidingWindowPatternDetector
    
    detector = SlidingWindowPatternDetector()
    converted_data = detector._convert_input_safe(test_data)
    print(f"Converted dtype: {converted_data.dtype}")
    assert converted_data.dtype == np.int32, f"Should preserve int32, got {converted_data.dtype}"
    
    # Test suffix array builder accepts the data
    from independent_core.compression_systems.padic.safe_suffix_array import SafeSuffixArrayBuilder
    builder = SafeSuffixArrayBuilder()
    
    # Should work with int32
    result = builder.build(converted_data)
    print(f"✓ Suffix array built successfully with {converted_data.dtype}")
    print(f"  Array length: {len(result.suffix_array)}")
    print(f"  Build time: {result.build_time:.4f}s")
    
    print("\n✅ Data type compatibility test passed!")

if __name__ == "__main__":
    test_dtype_compatibility()