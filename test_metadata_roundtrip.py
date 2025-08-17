#!/usr/bin/env python3
"""Test metadata roundtrip through compression"""

import sys
import os
import json

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from independent_core.compression_systems.padic.metadata_compressor import MetadataCompressor

def test_metadata_roundtrip():
    """Test that metadata with nested dicts survives compression/decompression"""
    print("Testing metadata roundtrip...")
    
    # Create test metadata similar to what the system uses
    test_metadata = {
        'version': 1,
        'prime': 7,
        'precision': 4,
        'original_shape': [1000],
        'entropy_metadata': {
            'original_shape': [100],
            'original_dtype': 'int64',
            'prime': 7,
            'encoding_method': 'huffman',
            'method': 'huffman',
            'encoding_metadata': {
                'method': 'huffman',
                'frequency_table': {
                    '0': 25,
                    '1': 17,
                    '2': 12,
                    '3': 12,
                    '4': 10,
                    '5': 13,
                    '6': 11
                },
                'encoding_table_size': 7
            },
            'compression_metrics': {
                'original_bytes': 800,
                'compressed_bytes': 40,
                'compression_ratio': 20.0
            }
        }
    }
    
    # Initialize compressor
    compressor = MetadataCompressor()
    
    # Test compression
    print("\n1. Original metadata structure:")
    print(json.dumps(test_metadata, indent=2))
    
    compressed = compressor.compress_metadata(test_metadata)
    print(f"\n2. Compressed size: {len(compressed)} bytes")
    
    # Test decompression
    decompressed = compressor.decompress_metadata(compressed)
    print("\n3. Decompressed metadata structure:")
    print(json.dumps(decompressed, indent=2))
    
    # Check if entropy_metadata survived
    if 'entropy_metadata' in decompressed:
        print("\n✓ entropy_metadata preserved")
        em = decompressed['entropy_metadata']
        
        if 'encoding_metadata' in em:
            print("✓ encoding_metadata preserved")
            enc_meta = em['encoding_metadata']
            
            if 'frequency_table' in enc_meta:
                print("✓ frequency_table preserved")
                print(f"  Frequency table: {enc_meta['frequency_table']}")
            else:
                print("❌ frequency_table LOST")
        else:
            print("❌ encoding_metadata LOST")
    else:
        print("❌ entropy_metadata LOST")
    
    # Verify exact match
    if test_metadata == decompressed:
        print("\n✅ Perfect metadata roundtrip!")
    else:
        print("\n❌ Metadata mismatch after roundtrip")
        # Show differences
        for key in test_metadata:
            if key not in decompressed:
                print(f"  Missing key: {key}")
            elif test_metadata[key] != decompressed[key]:
                print(f"  Different value for {key}")

if __name__ == "__main__":
    test_metadata_roundtrip()