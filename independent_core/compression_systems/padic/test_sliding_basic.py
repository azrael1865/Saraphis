"""
Quick basic test for Sliding Window Pattern Detector
"""

import torch
import numpy as np
from sliding_window_pattern_detector import SlidingWindowPatternDetector


def test_basic():
    """Quick test of basic functionality"""
    print("Testing Sliding Window Pattern Detector")
    print("-" * 40)
    
    # Create detector
    detector = SlidingWindowPatternDetector(
        min_pattern_length=3,
        max_pattern_length=8,
        min_frequency=2,
        hash_prime=31
    )
    
    # Create simple test data with obvious patterns
    pattern = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
    random_data = torch.randint(100, 200, (10,), dtype=torch.uint8)
    
    # Build test data: pattern, random, pattern, random, pattern
    test_data = torch.cat([
        pattern,
        random_data[:5],
        pattern,
        random_data[5:],
        pattern
    ])
    
    print(f"Test data size: {test_data.size(0)} bytes")
    print(f"Test data: {test_data.tolist()}")
    
    # Find patterns
    result = detector.find_patterns(test_data)
    
    print(f"\nResults:")
    print(f"  Patterns found: {result.total_patterns_found}")
    print(f"  Bytes replaced: {result.bytes_replaced}")
    print(f"  Compression potential: {result.compression_potential:.2%}")
    
    # Show detected patterns
    for pattern_id, pattern_match in result.patterns.items():
        print(f"\nPattern {pattern_id}:")
        print(f"  Pattern bytes: {list(pattern_match.pattern)}")
        print(f"  Length: {pattern_match.length}")
        print(f"  Positions: {pattern_match.positions}")
        print(f"  Frequency: {pattern_match.frequency}")
    
    # Test encoding/decoding
    print("\nTesting encoding/decoding...")
    encoded_data, pattern_dict, pattern_lengths = detector.encode_with_patterns(test_data)
    print(f"Encoded size: {encoded_data.size(0)} elements")
    
    decoded_data = detector.decode_with_patterns(encoded_data, pattern_dict, pattern_lengths)
    print(f"Decoded size: {decoded_data.size(0)} bytes")
    
    # Verify correctness
    is_correct = torch.equal(decoded_data, test_data)
    print(f"Decoding correct: {is_correct}")
    
    if is_correct:
        print("\n✓ Test PASSED!")
    else:
        print("\n✗ Test FAILED!")
        print(f"Original: {test_data.tolist()}")
        print(f"Decoded:  {decoded_data.tolist()}")
    
    return is_correct


if __name__ == "__main__":
    success = test_basic()
    exit(0 if success else 1)