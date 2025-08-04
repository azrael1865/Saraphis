#!/usr/bin/env python3
"""
Debug script to analyze why compression ratio is poor (0.40x-1.08x instead of 2-4x)
"""

import torch
import sys
import os

# Add the project root to path
sys.path.append('/home/will-casterlin/Desktop/Saraphis')

from independent_core.compression_systems.padic.padic_logarithmic_encoder import (
    PadicLogarithmicEncoder, LogarithmicEncodingConfig, LogarithmicPadicWeight
)
from independent_core.compression_systems.padic.padic_encoder import PadicWeight

def analyze_compression_calculation():
    """Analyze the actual compression calculation step by step"""
    print("=== COMPRESSION RATIO DEBUG ANALYSIS ===\n")
    
    # Create test configuration
    config = LogarithmicEncodingConfig(
        prime=257,
        precision=2,  # Should be low for good compression
        max_safe_precision=3
    )
    
    print(f"Configuration:")
    print(f"  - Prime: {config.prime}")
    print(f"  - Precision: {config.precision}")
    print(f"  - Max Safe Precision: {config.max_safe_precision}")
    print()
    
    # Create encoder
    encoder = PadicLogarithmicEncoder(config)
    
    # Create test weights (1000 weights as mentioned in the issue)
    test_weights = torch.randn(1000, dtype=torch.float32)
    print(f"Test weights:")
    print(f"  - Count: {len(test_weights)}")
    print(f"  - Original size: {test_weights.numel() * test_weights.element_size()} bytes")
    print(f"  - Element size: {test_weights.element_size()} bytes per weight")
    print()
    
    # Encode weights
    compressed_weights = []
    total_compressed_bytes = 0
    zero_weights = 0
    non_zero_weights = 0
    
    print("Analyzing first 10 weights in detail:")
    for i, weight in enumerate(test_weights[:10]):
        try:
            # Encode the weight
            log_weight = encoder.encode_logarithmic(float(weight))
            compressed_weights.append(log_weight)
            
            # Analyze the p-adic representation
            padic_weight = log_weight.padic_weight
            digits = padic_weight.digits
            
            # Count non-zero digits (sparse storage)
            non_zero_digits = [d for d in digits if d != 0]
            
            if not non_zero_digits:
                # Zero weight: minimal storage (1 byte marker)
                compressed_size = 1
                zero_weights += 1
            else:
                # Non-zero digits: 1 byte per digit + valuation + length
                compressed_size = len(non_zero_digits) * 1  # 1 byte per digit
                compressed_size += 2  # valuation (1 byte) + length (1 byte)
                non_zero_weights += 1
            
            total_compressed_bytes += compressed_size
            
            print(f"  Weight {i}: {float(weight):.6f}")
            print(f"    - Log value: {log_weight.log_value:.6f}")
            print(f"    - P-adic digits: {digits}")
            print(f"    - Non-zero digits: {non_zero_digits}")
            print(f"    - Compressed size: {compressed_size} bytes")
            print(f"    - Ratio for this weight: {4 / compressed_size:.2f}x")
            print()
            
        except Exception as e:
            print(f"  Weight {i}: FAILED - {e}")
            # Assume worst case: no compression
            total_compressed_bytes += 4
            continue
    
    # Calculate overall compression for first 10 weights
    original_size_10 = 10 * 4  # 10 weights * 4 bytes each
    ratio_10 = original_size_10 / total_compressed_bytes if total_compressed_bytes > 0 else 1.0
    
    print(f"First 10 weights summary:")
    print(f"  - Original size: {original_size_10} bytes")
    print(f"  - Compressed size: {total_compressed_bytes} bytes")
    print(f"  - Ratio: {ratio_10:.2f}x")
    print(f"  - Zero weights: {zero_weights}")
    print(f"  - Non-zero weights: {non_zero_weights}")
    print()
    
    # Now analyze the precision=2 impact
    print("Analyzing precision impact:")
    print(f"  - With precision=2, each p-adic weight has exactly {config.precision} digits")
    print(f"  - Each digit is in range [0, {config.prime-1}]")
    print(f"  - If precision=2 is correctly applied, we should see only 2 digits per weight")
    print()
    
    # Test a specific case to understand the digit distribution
    print("Testing digit distribution with more weights...")
    digit_stats = {"zero_digits": 0, "non_zero_digits": 0, "all_zero_weights": 0}
    
    for i, weight in enumerate(test_weights[:100]):  # Test 100 weights
        try:
            log_weight = encoder.encode_logarithmic(float(weight))
            digits = log_weight.padic_weight.digits
            
            all_zero = all(d == 0 for d in digits)
            if all_zero:
                digit_stats["all_zero_weights"] += 1
            
            for d in digits:
                if d == 0:
                    digit_stats["zero_digits"] += 1
                else:
                    digit_stats["non_zero_digits"] += 1
                    
        except Exception:
            continue
    
    total_digits = digit_stats["zero_digits"] + digit_stats["non_zero_digits"]
    if total_digits > 0:
        zero_percentage = (digit_stats["zero_digits"] / total_digits) * 100
        print(f"Digit distribution in 100 weights:")
        print(f"  - Total digits: {total_digits}")
        print(f"  - Zero digits: {digit_stats['zero_digits']} ({zero_percentage:.1f}%)")
        print(f"  - Non-zero digits: {digit_stats['non_zero_digits']} ({100-zero_percentage:.1f}%)")
        print(f"  - All-zero weights: {digit_stats['all_zero_weights']}")
        print()
    
    # Expected vs actual compression analysis
    print("Expected vs Actual Compression Analysis:")
    print("Expected scenario for 2-4x compression:")
    print("  - Original: 1000 weights × 4 bytes = 4000 bytes")
    print("  - Target compressed: 1000-2000 bytes (for 2-4x ratio)")
    print("  - This requires average of 1-2 bytes per weight")
    print()
    
    print("Current scenario analysis:")
    if zero_percentage > 50:
        expected_sparse_size = (digit_stats["all_zero_weights"] * 1) + ((100 - digit_stats["all_zero_weights"]) * 4)
        print(f"  - With {zero_percentage:.1f}% zero digits, sparse storage should help")
        print(f"  - Expected size for 100 weights: ~{expected_sparse_size} bytes")
        print(f"  - Expected ratio: {400 / expected_sparse_size:.2f}x")
    else:
        print(f"  - Only {zero_percentage:.1f}% zero digits - sparse storage won't help much")
        print("  - Problem: Not enough sparsity in the p-adic representation")
    
    print()
    print("DIAGNOSIS:")
    print("1. Check if precision=2 is actually being applied")
    print("2. Check if p-adic digits are mostly non-zero (poor sparsity)")
    print("3. Check if the calculation assumes 1 byte per digit correctly")
    print("4. Verify the logarithmic encoding is creating sparse representations")

if __name__ == "__main__":
    analyze_compression_calculation()