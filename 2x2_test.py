#!/usr/bin/env python3
"""
Verbose diagnostic test for P-adic compression system.
Tests with a tiny 2x2 tensor and prints detailed information at every stage.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
import traceback
import json

# Configure paths
base_path = Path(__file__).parent
sys.path.insert(0, str(base_path / "independent_core"))

from compression_systems.padic.padic_compressor import (
    PadicCompressionSystem,
    CompressionConfig,
    CompressionResult,
    DecompressionResult
)

def print_separator(title=""):
    """Print a visual separator"""
    if title:
        print(f"\n{'='*30} {title} {'='*30}")
    else:
        print(f"{'='*70}")

def print_tensor_details(tensor, name="Tensor", show_values=True):
    """Print detailed tensor information"""
    print(f"\n[{name}]")
    print(f"  Shape: {tensor.shape}")
    print(f"  Device: {tensor.device}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Min: {tensor.min().item():.8f}")
    print(f"  Max: {tensor.max().item():.8f}")
    print(f"  Mean: {tensor.mean().item():.8f}")
    print(f"  Std: {tensor.std().item():.8f}")
    
    if show_values and tensor.numel() <= 16:
        print(f"  Values (flat):")
        flat_tensor = tensor.flatten()
        for i, val in enumerate(flat_tensor):
            print(f"    [{i}]: {val.item():.8f}")

def inspect_padic_weight(weight, index=0):
    """Inspect a single p-adic weight object"""
    print(f"\n  P-adic Weight [{index}]:")
    if hasattr(weight, 'value'):
        print(f"    value: {weight.value} (type: {type(weight.value).__name__})")
    if hasattr(weight, 'digits'):
        print(f"    digits: {weight.digits[:10]}{'...' if len(weight.digits) > 10 else ''}")
        print(f"    digit count: {len(weight.digits)}")
    if hasattr(weight, 'valuation'):
        print(f"    valuation: {weight.valuation}")
    if hasattr(weight, 'prime'):
        print(f"    prime: {weight.prime}")
    if hasattr(weight, 'precision'):
        print(f"    precision: {weight.precision}")

def trace_reconstruction_step(value, stage, details=""):
    """Trace a reconstruction step"""
    print(f"\n  [{stage}]")
    if isinstance(value, (int, float)):
        print(f"    Value: {value:.12f}")
    elif isinstance(value, torch.Tensor):
        if value.numel() == 1:
            print(f"    Value: {value.item():.12f}")
        else:
            print(f"    Shape: {value.shape}, Min: {value.min():.6f}, Max: {value.max():.6f}")
    else:
        print(f"    Type: {type(value).__name__}")
        print(f"    Value: {value}")
    if details:
        print(f"    Details: {details}")

def run_verbose_diagnostic():
    """Run verbose diagnostic test with 2x2 tensor"""
    
    print_separator("P-ADIC COMPRESSION VERBOSE DIAGNOSTIC TEST")
    
    # Setup with minimal configuration
    print("\n[SETUP] Creating minimal test configuration...")
    
    config = CompressionConfig(
        prime=257,
        base_precision=4,
        min_precision=2,
        max_precision=4,
        target_error=1e-6,
        importance_threshold=0.1,
        compression_priority=0.5,
        enable_gpu=torch.cuda.is_available(),
        validate_reconstruction=False,  # Disable to see raw results
        chunk_size=1000,
        max_tensor_size=1_000_000,
        enable_memory_monitoring=False,  # Simplify output
        sparsity_threshold=1e-6,
        huffman_threshold=2.0,
        arithmetic_threshold=6.0,
        enable_hybrid_entropy=True,
        raise_on_error=True,
        max_reconstruction_error=1.0  # Very lenient for debugging
    )
    
    system = PadicCompressionSystem(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"  Prime: {config.prime}")
    print(f"  Base precision: {config.base_precision}")
    print(f"  Device: {device}")
    
    # Create small test tensor with known values
    print_separator("INPUT DATA")
    
    # Use simple values for easier debugging
    test_values = [
        [1.5, -0.25],
        [0.75, -2.0]
    ]
    
    test_tensor = torch.tensor(test_values, dtype=torch.float32, device=device)
    importance = torch.abs(test_tensor) + 0.1
    
    print_tensor_details(test_tensor, "ORIGINAL TENSOR")
    print_tensor_details(importance, "IMPORTANCE WEIGHTS")
    
    # COMPRESSION PHASE
    print_separator("COMPRESSION PHASE")
    
    try:
        print("\n[COMPRESS] Starting compression...")
        result = system.compress(test_tensor, importance)
        
        print(f"\n[COMPRESS] ✓ Compression completed")
        print(f"  Compression ratio: {result.compression_ratio:.4f}x")
        print(f"  Processing time: {result.processing_time:.4f}s")
        print(f"  Compressed size: {len(result.compressed_data)} bytes")
        
        # Inspect stage metrics
        print("\n[COMPRESS] Stage metrics:")
        for stage, metrics in result.stage_metrics.items():
            print(f"  {stage}: {metrics.get('time', 0):.4f}s")
            if 'compression_ratio' in metrics:
                print(f"    - compression_ratio: {metrics['compression_ratio']:.4f}")
            if 'average_precision' in metrics:
                print(f"    - average_precision: {metrics['average_precision']:.2f}")
        
    except Exception as e:
        print(f"\n[COMPRESS] ✗ Compression failed: {e}")
        traceback.print_exc()
        return False
    
    # MANUAL DECOMPRESSION WITH TRACING
    print_separator("DECOMPRESSION PHASE (WITH TRACING)")
    
    try:
        print("\n[DECOMPRESS] Starting manual decompression with tracing...")
        
        # Split compressed data
        compressed_data = result.compressed_data
        pos = 0
        
        # Read header
        version = compressed_data[0]
        print(f"\n  Version: {version}")
        pos = 1
        
        import struct
        entropy_size = struct.unpack('<I', compressed_data[pos:pos+4])[0]
        pos += 4
        metadata_size = struct.unpack('<I', compressed_data[pos:pos+4])[0]
        pos += 4
        pattern_size = struct.unpack('<I', compressed_data[pos:pos+4])[0]
        pos += 4
        
        print(f"  Entropy size: {entropy_size} bytes")
        print(f"  Metadata size: {metadata_size} bytes")
        print(f"  Pattern size: {pattern_size} bytes")
        
        if version == 2:
            pos += 4  # Skip checksum
        
        # Extract components
        entropy_data = compressed_data[pos:pos+entropy_size]
        pos += entropy_size
        metadata_data = compressed_data[pos:pos+metadata_size]
        pos += metadata_size
        pattern_data = compressed_data[pos:pos+pattern_size] if pattern_size > 0 else b''
        
        # Decompress metadata
        print("\n[METADATA] Decompressing metadata...")
        metadata = system.metadata_compressor.decompress_metadata(metadata_data)
        
        print(f"  Original shape: {metadata.get('original_shape')}")
        print(f"  Prime: {metadata.get('prime')}")
        print(f"  Precision: {metadata.get('precision')}")
        print(f"  Has patterns: {metadata.get('has_patterns')}")
        
        # Show precision map
        if 'precision_map' in metadata:
            precision_map = metadata['precision_map']
            print(f"  Precision map: {precision_map}")
        
        # Show valuations
        if 'valuations' in metadata:
            valuations = metadata['valuations']
            print(f"  Valuations: {valuations}")
        
        # Entropy decoding
        print("\n[ENTROPY] Decoding entropy data...")
        if entropy_data and not metadata.get('entropy_metadata', {}).get('empty'):
            sparse_values = system.entropy_bridge.decode_padic_tensor(
                entropy_data,
                metadata['entropy_metadata']
            )
            print(f"  Decoded {sparse_values.numel()} sparse values")
            if sparse_values.numel() <= 20:
                print(f"  Values: {sparse_values.cpu().numpy()}")
        else:
            print("  No entropy data to decode")
        
        # Now do the full decompression
        print("\n[DECOMPRESS] Running full decompression...")
        decompressed = system.decompress(result.compressed_data)
        
        print(f"\n[DECOMPRESS] ✓ Decompression completed")
        print_tensor_details(decompressed.reconstructed_tensor, "RECONSTRUCTED TENSOR")
        
    except Exception as e:
        print(f"\n[DECOMPRESS] ✗ Decompression failed: {e}")
        traceback.print_exc()
        return False
    
    # COMPARISON AND ANALYSIS
    print_separator("COMPARISON AND ANALYSIS")
    
    # Element-wise comparison
    print("\n[COMPARE] Element-wise comparison:")
    original_flat = test_tensor.flatten()
    reconstructed_flat = decompressed.reconstructed_tensor.flatten()
    
    print("\n  Index | Original      | Reconstructed  | Difference    | Rel Error")
    print("  " + "-"*70)
    
    for i in range(original_flat.numel()):
        orig = original_flat[i].item()
        recon = reconstructed_flat[i].item()
        diff = abs(orig - recon)
        rel_error = abs(diff / (orig + 1e-10))
        
        status = "✓" if rel_error < 0.01 else "✗"
        print(f"  [{i:2d}]  | {orig:13.8f} | {recon:14.8f} | {diff:13.8e} | {rel_error:9.2e} {status}")
    
    # Overall metrics
    mse = torch.nn.functional.mse_loss(test_tensor, decompressed.reconstructed_tensor)
    max_error = torch.max(torch.abs(test_tensor - decompressed.reconstructed_tensor))
    
    print(f"\n[METRICS]")
    print(f"  MSE: {mse.item():.8e}")
    print(f"  Max absolute error: {max_error.item():.8e}")
    print(f"  Compression ratio: {result.compression_ratio:.4f}x")
    
    # DEEP DIVE INTO P-ADIC WEIGHTS (if accessible)
    print_separator("P-ADIC WEIGHT ANALYSIS")
    
    try:
        # Try to access internal state if possible
        if hasattr(system, 'adaptive_precision'):
            print("\n[ADAPTIVE PRECISION] Checking adaptive precision wrapper...")
            
            # Try to get the p-adic weights from the last compression
            # This is implementation-specific and may not work
            test_result = system.adaptive_precision.convert_tensor(test_tensor, importance)
            
            print(f"  Number of p-adic weights: {len(test_result.padic_weights)}")
            print(f"  Average precision: {test_result.get_average_precision():.2f}")
            
            # Inspect first few weights
            for i in range(min(4, len(test_result.padic_weights))):
                inspect_padic_weight(test_result.padic_weights[i], i)
                
                # Try to reconstruct this single weight
                if hasattr(system, 'safe_reconstruction'):
                    try:
                        reconstructed_val = system.safe_reconstruction.reconstruct(
                            test_result.padic_weights[i]
                        )
                        original_val = test_tensor.flatten()[i].item()
                        
                        print(f"    Reconstructed value: {reconstructed_val:.8f}")
                        print(f"    Original value: {original_val:.8f}")
                        print(f"    Error: {abs(reconstructed_val - original_val):.8e}")
                    except Exception as e:
                        print(f"    Reconstruction failed: {e}")
    except Exception as e:
        print(f"  Could not analyze p-adic weights: {e}")
    
    # Final verdict
    print_separator("DIAGNOSTIC SUMMARY")
    
    success = mse.item() < 1e-3  # Very lenient threshold for debugging
    
    if success:
        print("\n✓ RECONSTRUCTION SUCCESSFUL")
        print(f"  MSE within acceptable range: {mse.item():.8e} < 1e-3")
    else:
        print("\n✗ RECONSTRUCTION FAILED")
        print(f"  MSE too high: {mse.item():.8e} >= 1e-3")
        
        # Provide diagnostic hints
        print("\n[DIAGNOSTIC HINTS]")
        if mse.item() > 1e10:
            print("  • Astronomical MSE suggests complete reconstruction failure")
            print("  • Check p-adic to float conversion in _reconstruct_fraction_from_digits")
            print("  • Verify SafePadicReconstructor is using correct precision")
            print("  • Check for overflow in p-adic digit to fraction conversion")
        elif mse.item() > 1.0:
            print("  • Large MSE suggests systematic error in reconstruction")
            print("  • Check precision map alignment with p-adic weights")
            print("  • Verify valuation application is correct")
        else:
            print("  • Moderate MSE suggests precision loss")
            print("  • Consider increasing base_precision")
            print("  • Check adaptive precision allocation")
    
    return success

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" STARTING VERBOSE DIAGNOSTIC TEST")
    print("="*70)
    
    try:
        success = run_verbose_diagnostic()
        
        if success:
            print("\n" + "="*70)
            print(" ✓✓✓ DIAGNOSTIC TEST PASSED ✓✓✓")
            print("="*70)
            sys.exit(0)
        else:
            print("\n" + "="*70)
            print(" ✗✗✗ DIAGNOSTIC TEST FAILED ✗✗✗")
            print("="*70)
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[FATAL] Test crashed: {e}")
        traceback.print_exc()
        print("\n" + "="*70)
        print(" ✗✗✗ DIAGNOSTIC TEST CRASHED ✗✗✗")
        print("="*70)
        sys.exit(2)
