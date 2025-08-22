def test_padic_digit_expansion():
    """Test if the 6x expansion comes from p-adic digit representation"""
    
    print("\n" + "="*70)
    print("P-ADIC DIGIT EXPANSION ANALYSIS")
    print("="*70)
    print("Testing if 6x comes from p-adic representation of floats")
    
    config = CompressionConfig(
        prime=257,
        base_precision=4,
        min_precision=2,
        max_precision=8,
        target_error=1e-6,
        importance_threshold=0.1,
        compression_priority=0.5,
        enable_gpu=torch.cuda.is_available(),
        validate_reconstruction=False,
        chunk_size=1000,
        max_tensor_size=1_000_000,
        enable_memory_monitoring=False,
        sparsity_threshold=1e-6,
        huffman_threshold=2.0,
        arithmetic_threshold=6.0,
        enable_hybrid_entropy=True,
        raise_on_error=False
    )
    
    system = PadicCompressionSystem(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with different#!/usr/bin/env python3
"""
Diagnostic test to identify the root cause of pattern detection failures.
This test isolates each component to find where the corruption occurs.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
import traceback

# Configure paths
base_path = Path(__file__).parent
sys.path.insert(0, str(base_path / "independent_core"))

from compression_systems.padic.padic_compressor import (
    PadicCompressionSystem,
    CompressionConfig,
)

def test_pattern_detection_isolation():
    """Test pattern detection in isolation to identify the bug"""
    
    print("\n" + "="*70)
    print("PATTERN DETECTION ISOLATION TEST")
    print("="*70)
    
    # Create system with pattern detection DISABLED first
    print("\n[TEST 1] System with large chunk size (may skip pattern detection)")
    print("-"*50)
    
    # Since we can't disable pattern detection directly, try with minimal settings
    config_no_pattern = CompressionConfig(
        prime=257,
        base_precision=4,
        min_precision=2,
        max_precision=8,
        target_error=1e-6,
        importance_threshold=0.1,
        compression_priority=0.5,
        enable_gpu=torch.cuda.is_available(),
        validate_reconstruction=False,  # Disable validation to see raw results
        chunk_size=10000,  # Large chunk to potentially skip pattern detection
        max_tensor_size=1_000_000,
        enable_memory_monitoring=False,
        sparsity_threshold=1e-6,
        huffman_threshold=2.0,
        arithmetic_threshold=6.0,
        enable_hybrid_entropy=True,
        raise_on_error=False  # Don't fail immediately
    )
    
    system_no_pattern = PadicCompressionSystem(config_no_pattern)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test on 100x100 tensor
    test_tensor = torch.randn(100, 100, device=device)
    importance = torch.abs(test_tensor) + 0.1
    
    print(f"Testing 100x100 tensor (10,000 elements)")
    
    try:
        result = system_no_pattern.compress(test_tensor, importance)
        decompressed = system_no_pattern.decompress(result.compressed_data)
        error = torch.nn.functional.mse_loss(test_tensor, decompressed.reconstructed_tensor)
        
        print(f"✓ With large chunk size:")
        print(f"  Compression ratio: {result.compression_ratio:.4f}x")
        print(f"  Reconstruction MSE: {error:.2e}")
        
        if error < 1.0:
            print(f"  ✓ GOOD: Low error with large chunks!")
        else:
            print(f"  ✗ BAD: High error even with large chunks")
            
    except Exception as e:
        print(f"✗ Failed with large chunk size: {e}")
    
    # Now test with standard settings
    print("\n[TEST 2] System with standard chunk size (default)")
    print("-"*50)
    
    config_with_pattern = CompressionConfig(
        prime=257,
        base_precision=4,
        min_precision=2,
        max_precision=8,
        target_error=1e-6,
        importance_threshold=0.1,
        compression_priority=0.5,
        enable_gpu=torch.cuda.is_available(),
        validate_reconstruction=False,
        chunk_size=1000,
        max_tensor_size=1_000_000,
        enable_memory_monitoring=False,
        sparsity_threshold=1e-6,
        huffman_threshold=2.0,
        arithmetic_threshold=6.0,
        enable_hybrid_entropy=True,
        raise_on_error=False
    )
    
    system_with_pattern = PadicCompressionSystem(config_with_pattern)
    
    try:
        result = system_with_pattern.compress(test_tensor, importance)
        decompressed = system_with_pattern.decompress(result.compressed_data)
        error = torch.nn.functional.mse_loss(test_tensor, decompressed.reconstructed_tensor)
        
        print(f"✓ With standard settings:")
        print(f"  Compression ratio: {result.compression_ratio:.4f}x")
        print(f"  Reconstruction MSE: {error:.2e}")
        
        if error < 1.0:
            print(f"  ✓ GOOD: Standard settings working!")
        else:
            print(f"  ✗ BAD: Standard settings causing corruption!")
            
    except Exception as e:
        print(f"✗ Failed with standard settings: {e}")

def test_component_sizes():
    """Test to understand why pattern detection creates 6x data"""
    
    print("\n" + "="*70)
    print("COMPONENT SIZE ANALYSIS")
    print("="*70)
    
    config = CompressionConfig(
        prime=257,
        base_precision=4,
        min_precision=2,
        max_precision=8,
        target_error=1e-6,
        importance_threshold=0.1,
        compression_priority=0.5,
        enable_gpu=torch.cuda.is_available(),
        validate_reconstruction=False,
        chunk_size=1000,
        max_tensor_size=1_000_000,
        enable_memory_monitoring=True,  # Enable to see memory usage
        sparsity_threshold=1e-6,
        huffman_threshold=2.0,
        arithmetic_threshold=6.0,
        enable_hybrid_entropy=True,
        raise_on_error=False
    )
    
    system = PadicCompressionSystem(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different sizes to find the pattern
    test_sizes = [10, 25, 50, 100]
    
    print("\nSize analysis:")
    print("Input Size | Expected Pattern | Actual Pattern | Ratio")
    print("-"*60)
    
    for size in test_sizes:
        test_tensor = torch.randn(size, size, device=device)
        importance = torch.abs(test_tensor) + 0.1
        
        input_elements = size * size
        
        try:
            # Hook into compression to see intermediate sizes
            result = system.compress(test_tensor, importance)
            
            # Try to extract pattern size from stage metrics
            if 'pattern_detection' in result.stage_metrics:
                pattern_info = result.stage_metrics.get('pattern_detection', {})
                print(f"{input_elements:10} | {input_elements:16} | ??? | ???")
            else:
                print(f"{input_elements:10} | {input_elements:16} | No pattern info | N/A")
                
            # Check the actual compressed data structure
            if hasattr(result, 'compressed_data'):
                import struct
                data = result.compressed_data
                if len(data) > 13:
                    # Skip version byte
                    pos = 1
                    entropy_size = struct.unpack('<I', data[pos:pos+4])[0]
                    pos += 4
                    metadata_size = struct.unpack('<I', data[pos:pos+4])[0]
                    pos += 4
                    pattern_size = struct.unpack('<I', data[pos:pos+4])[0]
                    
                    print(f"  Compressed structure: entropy={entropy_size}B, metadata={metadata_size}B, pattern={pattern_size}B")
                    
        except Exception as e:
            print(f"{input_elements:10} | Error: {str(e)[:40]}")

def test_direct_pattern_detector():
    """Test the pattern detector component directly"""
    
    print("\n" + "="*70)
    print("DIRECT PATTERN DETECTOR TEST")
    print("="*70)
    
    try:
        # Try to import and test the pattern detector directly
        from compression_systems.padic.sliding_window_pattern_detector import SlidingWindowPatternDetector
        
        print("✓ Pattern detector module imported")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create detector
        detector = SlidingWindowPatternDetector(
            window_size=64,
            stride=32,
            min_pattern_length=4,
            max_patterns=1000,
            device=device
        )
        
        # Test with simple data
        test_sizes = [(2, 2), (10, 10), (100, 100)]
        
        for shape in test_sizes:
            test_tensor = torch.randn(shape, device=device)
            flat_tensor = test_tensor.flatten()
            
            print(f"\nTesting {shape[0]}x{shape[1]} tensor ({flat_tensor.numel()} elements)")
            
            try:
                # Detect patterns
                patterns = detector.detect_patterns(flat_tensor)
                
                if patterns:
                    print(f"  Found {len(patterns)} patterns")
                    
                    # Check if patterns are creating the 6x expansion
                    total_pattern_elements = sum(len(p['data']) for p in patterns if 'data' in p)
                    print(f"  Total pattern elements: {total_pattern_elements}")
                    print(f"  Expansion factor: {total_pattern_elements / flat_tensor.numel():.2f}x")
                    
                    # Apply patterns
                    compressed = detector.apply_patterns(flat_tensor, patterns)
                    print(f"  Compressed size: {compressed.numel() if hasattr(compressed, 'numel') else len(compressed)}")
                    
                    # Try reconstruction
                    reconstructed = detector.reconstruct_from_patterns(compressed, patterns, flat_tensor.numel())
                    
                    if reconstructed.numel() != flat_tensor.numel():
                        print(f"  ✗ SIZE MISMATCH: Expected {flat_tensor.numel()}, got {reconstructed.numel()}")
                        print(f"    This is the root cause!")
                    else:
                        error = torch.nn.functional.mse_loss(flat_tensor, reconstructed)
                        print(f"  Reconstruction MSE: {error:.2e}")
                else:
                    print(f"  No patterns found")
                    
            except Exception as e:
                print(f"  ✗ Pattern detection failed: {e}")
                traceback.print_exc()
                
    except ImportError as e:
        print(f"✗ Could not import pattern detector: {e}")
        
    except Exception as e:
        print(f"✗ Direct pattern detector test failed: {e}")
        traceback.print_exc()

def test_fix_suggestions():
    """Test potential fixes for the pattern detection issue"""
    
    print("\n" + "="*70)
    print("TESTING POTENTIAL FIXES")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # FIX 1: Try smaller chunk size
    print("\n[FIX 1] Smaller chunk size")
    print("-"*50)
    
    config_small_chunk = CompressionConfig(
        prime=257,
        base_precision=4,
        min_precision=2,
        max_precision=8,
        target_error=1e-6,
        importance_threshold=0.1,
        compression_priority=0.5,
        enable_gpu=torch.cuda.is_available(),
        validate_reconstruction=False,
        chunk_size=100,  # Much smaller chunk size
        max_tensor_size=1_000_000,
        enable_memory_monitoring=False,
        sparsity_threshold=1e-6,
        huffman_threshold=2.0,
        arithmetic_threshold=6.0,
        enable_hybrid_entropy=True,
        raise_on_error=False
    )
    
    system = PadicCompressionSystem(config_small_chunk)
    test_tensor = torch.randn(100, 100, device=device)
    importance = torch.abs(test_tensor) + 0.1
    
    try:
        result = system.compress(test_tensor, importance)
        decompressed = system.decompress(result.compressed_data)
        error = torch.nn.functional.mse_loss(test_tensor, decompressed.reconstructed_tensor)
        
        if error < 1.0:
            print(f"✓ FIX WORKS! Smaller chunk size helps (MSE: {error:.2e})")
        else:
            print(f"✗ Still broken with smaller chunks (MSE: {error:.2e})")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # FIX 2: Try different precision settings
    print("\n[FIX 2] Lower precision settings")
    print("-"*50)
    
    config_low_precision = CompressionConfig(
        prime=257,
        base_precision=2,  # Lower precision
        min_precision=1,
        max_precision=3,
        target_error=1e-3,  # More lenient
        importance_threshold=0.1,
        compression_priority=0.5,
        enable_gpu=torch.cuda.is_available(),
        validate_reconstruction=False,
        chunk_size=1000,
        max_tensor_size=1_000_000,
        enable_memory_monitoring=False,
        sparsity_threshold=1e-6,
        huffman_threshold=2.0,
        arithmetic_threshold=6.0,
        enable_hybrid_entropy=True,
        raise_on_error=False
    )
    
    system = PadicCompressionSystem(config_low_precision)
    
    try:
        result = system.compress(test_tensor, importance)
        decompressed = system.decompress(result.compressed_data)
        error = torch.nn.functional.mse_loss(test_tensor, decompressed.reconstructed_tensor)
        
        if error < 1.0:
            print(f"✓ FIX WORKS! Lower precision helps (MSE: {error:.2e})")
        else:
            print(f"✗ Still broken with lower precision (MSE: {error:.2e})")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # FIX 3: Try disabling entropy coding
    print("\n[FIX 3] Disable hybrid entropy")
    print("-"*50)
    
    config_no_entropy = CompressionConfig(
        prime=257,
        base_precision=4,
        min_precision=2,
        max_precision=8,
        target_error=1e-6,
        importance_threshold=0.1,
        compression_priority=0.5,
        enable_gpu=torch.cuda.is_available(),
        validate_reconstruction=False,
        chunk_size=1000,
        max_tensor_size=1_000_000,
        enable_memory_monitoring=False,
        sparsity_threshold=1e-6,
        huffman_threshold=2.0,
        arithmetic_threshold=6.0,
        enable_hybrid_entropy=False,  # Disable entropy coding
        raise_on_error=False
    )
    
    system = PadicCompressionSystem(config_no_entropy)
    
    try:
        result = system.compress(test_tensor, importance)
        decompressed = system.decompress(result.compressed_data)
        error = torch.nn.functional.mse_loss(test_tensor, decompressed.reconstructed_tensor)
        
        if error < 1.0:
            print(f"✓ FIX WORKS! Disabling entropy helps (MSE: {error:.2e})")
        else:
            print(f"✗ Still broken without entropy (MSE: {error:.2e})")
    except Exception as e:
        print(f"✗ Failed: {e}")

def test_pattern_reconstruction_mismatch():
    """Test to reproduce the specific pattern reconstruction size mismatch"""
    
    print("\n" + "="*70)
    print("PATTERN RECONSTRUCTION MISMATCH TEST")
    print("="*70)
    print("Reproducing: 'expected 60000, got 60333' error")
    
    import logging
    
    # Set up logging to capture warnings
    logging.basicConfig(level=logging.WARNING)
    
    config = CompressionConfig(
        prime=257,
        base_precision=4,
        min_precision=2,
        max_precision=8,
        target_error=1e-6,
        importance_threshold=0.1,
        compression_priority=0.5,
        enable_gpu=torch.cuda.is_available(),
        validate_reconstruction=True,  # Enable to see validation errors
        chunk_size=1000,
        max_tensor_size=1_000_000,
        enable_memory_monitoring=True,
        sparsity_threshold=1e-6,
        huffman_threshold=2.0,
        arithmetic_threshold=6.0,
        enable_hybrid_entropy=True,
        raise_on_error=False
    )
    
    system = PadicCompressionSystem(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with exact size that causes the issue
    test_tensor = torch.randn(100, 100, device=device)
    importance = torch.abs(test_tensor) + 0.1
    
    print(f"\nTesting with 100x100 tensor (10,000 elements)")
    print("Expected pattern array: ~60,000 elements (6x expansion)")
    
    try:
        # Compress
        result = system.compress(test_tensor, importance)
        
        # Check stage metrics for pattern info
        if 'pattern_detection' in result.stage_metrics:
            pattern_metrics = result.stage_metrics['pattern_detection']
            print(f"\nPattern detection metrics:")
            for key, value in pattern_metrics.items():
                print(f"  {key}: {value}")
        
        # Try to decompress and catch the size mismatch
        print("\nAttempting decompression...")
        decompressed = system.decompress(result.compressed_data)
        
        error = torch.nn.functional.mse_loss(test_tensor, decompressed.reconstructed_tensor)
        print(f"\nReconstruction MSE: {error:.2e}")
        
        if error > 1000:
            print("✓ Successfully reproduced the catastrophic failure!")
            print("  The pattern detection is definitely the culprit")
        else:
            print("✗ Could not reproduce the failure")
            
    except Exception as e:
        print(f"✗ Exception during test: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all diagnostic tests"""
    
    print("\n" + "="*70)
    print("P-ADIC PATTERN DETECTION DIAGNOSTIC SUITE")
    print("="*70)
    print("\nThis suite isolates the pattern detection bug that causes")
    print("catastrophic failure on tensors larger than 5x5")
    
    # Run diagnostic tests
    test_pattern_detection_isolation()
    test_pattern_reconstruction_mismatch()
    test_component_sizes()
    test_direct_pattern_detector()
    test_fix_suggestions()
    
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    print("\nBased on the tests above, look for:")
    print("• If MSE is ~3.8e6, the issue is in pattern reconstruction")
    print("• If you see 'expected 60000, got 60333', it's a size mismatch")
    print("• If you see 6x data expansion, it's in the suffix array creation")
    print("\nThe issue is likely:")
    print("1. Pattern detector creating 6x more data than input")
    print("2. Size mismatch between compression and decompression")
    print("3. Pattern reconstruction not properly bounded")
    print("\nRecommended fix:")
    print("• Check sliding_window_pattern_detector.py")
    print("• Look for why suffix array is 6x input size")
    print("• Ensure reconstruct_from_patterns respects original size")
    print("• Consider disabling pattern detection until fixed")

if __name__ == "__main__":
    try:
        main()
        print("\n✓ Diagnostic tests completed")
    except Exception as e:
        print(f"\n✗ Diagnostic suite failed: {e}")
        traceback.print_exc()
        sys.exit(1)
