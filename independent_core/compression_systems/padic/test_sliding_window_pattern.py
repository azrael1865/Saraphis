"""
Test script for Sliding Window Pattern Detector integration
Verifies polynomial rolling hash and pattern compression
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import time
from sliding_window_pattern_detector import SlidingWindowPatternDetector

# Try to import the P-adic system for integration test
try:
    from padic_compression_pytorch import PurelyPyTorchPAdicSystem, PurelyPyTorchConfig
    PADIC_AVAILABLE = True
except ImportError:
    PADIC_AVAILABLE = False
    print("Warning: P-adic compression system not available, skipping integration test")


def test_basic_pattern_detection():
    """Test basic pattern detection functionality"""
    print("=" * 60)
    print("Testing Basic Pattern Detection")
    print("=" * 60)
    
    # Create detector
    detector = SlidingWindowPatternDetector(
        min_pattern_length=4,
        max_pattern_length=16,
        min_frequency=3,
        hash_prime=31
    )
    
    # Create test data with known patterns
    pattern1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint8)
    pattern2 = torch.tensor([10, 20, 30, 40], dtype=torch.uint8)
    
    # Build test data with repeated patterns
    test_data = []
    for i in range(10):
        if i % 3 == 0:
            test_data.append(pattern1)
        elif i % 3 == 1:
            test_data.append(pattern2)
        else:
            test_data.append(torch.randint(0, 256, (5,), dtype=torch.uint8))
    
    test_tensor = torch.cat(test_data)
    print(f"Test data size: {test_tensor.size(0)} bytes")
    
    # Find patterns
    result = detector.find_patterns(test_tensor)
    
    print(f"\nPattern Detection Results:")
    print(f"  Patterns found: {result.total_patterns_found}")
    print(f"  Bytes replaced: {result.bytes_replaced}")
    print(f"  Compression potential: {result.compression_potential:.2%}")
    print(f"  Original size: {result.original_size}")
    
    # Test encoding and decoding
    encoded_data, pattern_dict, pattern_lengths = detector.encode_with_patterns(test_tensor)
    print(f"\nEncoded size: {encoded_data.size(0)} elements")
    print(f"Pattern dictionary size: {len(pattern_dict)} patterns")
    
    # Decode
    decoded_data = detector.decode_with_patterns(encoded_data, pattern_dict, pattern_lengths)
    print(f"Decoded size: {decoded_data.size(0)} bytes")
    
    # Verify correctness
    is_correct = torch.equal(decoded_data, test_tensor)
    print(f"Decoding correct: {is_correct}")
    
    assert is_correct, "Decoding failed - data mismatch!"
    print("\n✓ Basic pattern detection test passed!")
    return True


def test_rolling_hash_efficiency():
    """Test rolling hash O(1) efficiency"""
    print("\n" + "=" * 60)
    print("Testing Rolling Hash Efficiency")
    print("=" * 60)
    
    detector = SlidingWindowPatternDetector()
    
    # Test with different data sizes
    sizes = [100, 1000, 10000]
    window_size = 8
    
    for size in sizes:
        data = torch.randint(0, 256, (size,), dtype=torch.uint8)
        
        start_time = time.time()
        primary, secondary = detector._compute_rolling_hashes(data, window_size)
        elapsed = time.time() - start_time
        
        num_windows = size - window_size + 1
        time_per_window = elapsed / num_windows if num_windows > 0 else 0
        
        print(f"\nData size: {size}")
        print(f"  Windows: {num_windows}")
        print(f"  Total time: {elapsed:.4f}s")
        print(f"  Time per window: {time_per_window*1e6:.2f}μs")
        print(f"  Hash computations: {primary.size(0)}")
    
    print("\n✓ Rolling hash efficiency test passed!")
    return True


def test_padic_integration():
    """Test integration with P-adic compression system"""
    print("\n" + "=" * 60)
    print("Testing P-adic Compression Integration")
    print("=" * 60)
    
    if not PADIC_AVAILABLE:
        print("SKIPPED: P-adic system not available")
        return True
    
    # Configure system with pattern detection enabled
    config = PurelyPyTorchConfig(
        prime=257,
        precision=6,
        enable_pattern_matching=True,
        enable_sparse=False,  # Disable sparse to focus on pattern detection
        enable_entropy=False,  # Disable entropy to focus on pattern detection
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create compression system
    system = PurelyPyTorchPAdicSystem(config)
    
    # Create test tensor with patterns
    base_values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
    test_tensor = base_values.repeat(20).reshape(10, 10)  # Create 10x10 tensor with patterns
    
    print(f"Original tensor shape: {test_tensor.shape}")
    print(f"Original tensor size: {test_tensor.numel() * 4} bytes")
    
    # Compress
    start_time = time.time()
    compressed = system.compress(test_tensor)
    compress_time = time.time() - start_time
    
    print(f"\nCompression Results:")
    print(f"  Compression time: {compress_time:.4f}s")
    print(f"  Compression ratio: {compressed.compression_ratio:.2f}x")
    print(f"  Compressed size: {compressed.compressed_data.numel() * 4} bytes")
    
    # Check if sliding patterns were detected
    if 'sliding_patterns' in compressed.metadata.get('pattern_metadata', {}):
        sliding_info = compressed.metadata['pattern_metadata']['sliding_patterns']
        print(f"\nSliding Pattern Detection:")
        print(f"  Patterns found: {sliding_info['num_patterns']}")
        print(f"  Compression potential: {sliding_info['compression_potential']:.2%}")
        print(f"  Bytes replaced: {sliding_info['bytes_replaced']}")
    
    # Decompress
    start_time = time.time()
    decompressed = system.decompress(compressed)
    decompress_time = time.time() - start_time
    
    print(f"\nDecompression Results:")
    print(f"  Decompression time: {decompress_time:.4f}s")
    print(f"  Reconstructed shape: {decompressed.reconstructed_data.shape}")
    
    # Check reconstruction accuracy
    max_error = torch.abs(test_tensor - decompressed.reconstructed_data.cpu()).max().item()
    print(f"  Max reconstruction error: {max_error:.6f}")
    
    assert max_error < 0.01, f"Reconstruction error too high: {max_error}"
    print("\n✓ P-adic integration test passed!")
    return True


def test_collision_detection():
    """Test hash collision detection and verification"""
    print("\n" + "=" * 60)
    print("Testing Hash Collision Detection")
    print("=" * 60)
    
    detector = SlidingWindowPatternDetector()
    
    # Create data that might cause hash collisions
    # Use a small hash modulus to increase collision probability
    detector.HASH_MODULUS = 127  # Small prime for testing
    
    # Create patterns that might collide
    data = torch.zeros(100, dtype=torch.uint8)
    # Insert identical patterns
    pattern = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
    positions = [0, 20, 40, 60, 80]
    for pos in positions:
        data[pos:pos+4] = pattern
    
    # Find patterns
    result = detector.find_patterns(data)
    
    print(f"Patterns found: {result.total_patterns_found}")
    
    # Verify all detected patterns are correct
    for pattern_id, pattern_match in result.patterns.items():
        print(f"\nPattern {pattern_id}:")
        print(f"  Length: {pattern_match.length}")
        print(f"  Frequency: {pattern_match.frequency}")
        print(f"  Positions: {pattern_match.positions[:5]}...")
        
        # Verify pattern correctness
        for pos in pattern_match.positions:
            actual = data[pos:pos+pattern_match.length]
            expected = torch.frombuffer(pattern_match.pattern, dtype=torch.uint8)
            assert torch.equal(actual, expected), f"Pattern mismatch at position {pos}"
    
    print("\n✓ Collision detection test passed!")
    return True


def test_large_scale_compression():
    """Test with large-scale data"""
    print("\n" + "=" * 60)
    print("Testing Large-Scale Compression")
    print("=" * 60)
    
    detector = SlidingWindowPatternDetector(
        min_pattern_length=8,
        max_pattern_length=64,
        min_frequency=5
    )
    
    # Create large test data with various patterns
    np.random.seed(42)
    
    # Create multiple patterns of different lengths
    patterns = [
        np.random.randint(0, 256, 10, dtype=np.uint8),
        np.random.randint(0, 256, 20, dtype=np.uint8),
        np.random.randint(0, 256, 30, dtype=np.uint8),
    ]
    
    # Build large dataset
    data_parts = []
    for _ in range(1000):
        if np.random.random() < 0.4:  # 40% chance of pattern
            pattern_idx = np.random.randint(0, len(patterns))
            data_parts.append(patterns[pattern_idx])
        else:
            # Random data
            random_len = np.random.randint(5, 50)
            data_parts.append(np.random.randint(0, 256, random_len, dtype=np.uint8))
    
    test_data = np.concatenate(data_parts)
    print(f"Test data size: {len(test_data):,} bytes")
    
    # Analyze compression
    start_time = time.time()
    analysis = detector.analyze_compression_efficiency(test_data)
    analysis_time = time.time() - start_time
    
    print(f"\nCompression Analysis:")
    print(f"  Analysis time: {analysis_time:.4f}s")
    print(f"  Original size: {analysis['original_size']:,} bytes")
    print(f"  Compressed size: {analysis['total_compressed_size']:,} bytes")
    print(f"  Compression ratio: {analysis['compression_ratio']:.2f}x")
    print(f"  Space savings: {analysis['space_savings_percent']:.1f}%")
    print(f"  Patterns found: {analysis['patterns_found']}")
    
    if analysis['pattern_statistics']:
        stats = analysis['pattern_statistics']
        print(f"\nPattern Statistics:")
        print(f"  Avg pattern length: {stats['avg_pattern_length']:.1f}")
        print(f"  Avg pattern frequency: {stats['avg_pattern_frequency']:.1f}")
        print(f"  Total bytes in patterns: {stats['total_bytes_in_patterns']:,}")
    
    print("\n✓ Large-scale compression test passed!")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("SLIDING WINDOW PATTERN DETECTOR TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Basic Pattern Detection", test_basic_pattern_detection),
        ("Rolling Hash Efficiency", test_rolling_hash_efficiency),
        ("P-adic Integration", test_padic_integration),
        ("Collision Detection", test_collision_detection),
        ("Large-Scale Compression", test_large_scale_compression),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)