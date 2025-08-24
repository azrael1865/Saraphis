#!/usr/bin/env python3
"""
Comprehensive test suite for MetadataCompressor
Tests all compression/decompression functionality and edge cases
"""

import unittest
import json
import struct
import zlib
import base64
from typing import Dict, Any, List
import numpy as np

from compression_systems.padic.metadata_compressor import MetadataCompressor, MetadataHeader


class TestMetadataHeader(unittest.TestCase):
    """Test MetadataHeader dataclass"""
    
    def test_default_initialization(self):
        """Test default MetadataHeader initialization"""
        header = MetadataHeader()
        
        self.assertEqual(header.version, 1)
        self.assertEqual(header.prime, 257)
        self.assertEqual(header.precision, 6)
        self.assertEqual(header.compression_flags, 0)
        self.assertEqual(header.original_shape, ())
    
    def test_custom_initialization(self):
        """Test custom MetadataHeader initialization"""
        header = MetadataHeader(
            version=2,
            prime=509,
            precision=8,
            compression_flags=3,
            original_shape=(10, 20, 30)
        )
        
        self.assertEqual(header.version, 2)
        self.assertEqual(header.prime, 509)
        self.assertEqual(header.precision, 8)
        self.assertEqual(header.compression_flags, 3)
        self.assertEqual(header.original_shape, (10, 20, 30))


class TestMetadataCompressorInit(unittest.TestCase):
    """Test MetadataCompressor initialization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.compressor = MetadataCompressor()
    
    def test_initialization(self):
        """Test proper initialization"""
        self.assertIsInstance(self.compressor.compression_stats, dict)
        self.assertEqual(self.compressor.compression_stats['fallback_decompressions'], 0)
        self.assertEqual(self.compressor.compression_stats['successful_decompressions'], 0)
        self.assertEqual(self.compressor.compression_stats['compression_attempts'], 0)
    
    def test_statistics_methods(self):
        """Test statistics methods"""
        stats = self.compressor.get_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn('fallback_decompressions', stats)
        self.assertIn('successful_decompressions', stats)
        self.assertIn('compression_attempts', stats)
        
        self.compressor.reset_statistics()
        stats = self.compressor.get_statistics()
        self.assertEqual(stats['fallback_decompressions'], 0)


class TestMetadataCompression(unittest.TestCase):
    """Test metadata compression functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.compressor = MetadataCompressor()
    
    def test_compress_empty_metadata(self):
        """Test compression of empty metadata"""
        result = self.compressor.compress_metadata({})
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)
    
    def test_compress_simple_metadata(self):
        """Test compression of simple metadata"""
        metadata = {
            'entropy_metadata': {
                'encoding_method': 'huffman',
                'method': 'huffman',
                'min_val': 0.0,
                'max_val': 1.0,
                'entropy': 2.5,
                'n_bins': 256
            }
        }
        
        result = self.compressor.compress_metadata(metadata)
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)
        
        # Check format flag
        self.assertEqual(result[0], 0x03)  # Full metadata format
    
    def test_compress_complex_metadata(self):
        """Test compression of complex metadata with various data types"""
        metadata = {
            'version': 1,
            'prime': 257,
            'precision': 6,
            'original_shape': [100, 200],
            'entropy_metadata': {
                'encoding_method': 'arithmetic',
                'min_val': -10.5,
                'max_val': 10.5,
                'entropy': 3.2,
                'n_bins': 512,
                'frequency_table': {1: 10, 2: 20, 3: 15}
            },
            'pattern_dict': {
                42: b'pattern_data_1',
                99: b'pattern_data_2'
            },
            'sparse_indices': [1, 5, 10, 15],
            'valuations': [0.1, 0.2, 0.3, 0.4]
        }
        
        result = self.compressor.compress_metadata(metadata)
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0], 0x03)
    
    def test_compress_metadata_with_bytes(self):
        """Test compression of metadata containing byte arrays"""
        metadata = {
            'entropy_metadata': {
                'encoding_method': 'custom',
                'binary_data': b'some binary data here'
            },
            'pattern_dict': {
                1: b'pattern1',
                2: b'pattern2'
            }
        }
        
        result = self.compressor.compress_metadata(metadata)
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)
    
    def test_compress_metadata_with_numpy_arrays(self):
        """Test compression of metadata with numpy arrays"""
        metadata = {
            'entropy_metadata': {
                'encoding_method': 'huffman'
            },
            'numpy_array': np.array([1, 2, 3, 4, 5]),
            'numpy_scalar': np.float32(3.14)
        }
        
        result = self.compressor.compress_metadata(metadata)
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)
    
    def test_compress_large_metadata_triggers_compression(self):
        """Test that large metadata triggers zlib compression"""
        # Create metadata larger than 100 bytes to trigger compression
        large_data = 'x' * 200
        metadata = {
            'entropy_metadata': {
                'encoding_method': 'huffman',
                'large_field': large_data
            }
        }
        
        result = self.compressor.compress_metadata(metadata)
        self.assertIsInstance(result, bytes)
        # Should have compression flag
        self.assertEqual(result[0], 0x03)  # Format flag
        self.assertEqual(result[1], 0x01)  # Compression flag
    
    def test_compress_small_metadata_no_compression(self):
        """Test that small metadata doesn't use zlib compression"""
        metadata = {
            'entropy_metadata': {
                'encoding_method': 'huffman'
            }
        }
        
        result = self.compressor.compress_metadata(metadata)
        self.assertIsInstance(result, bytes)
        # Should not have compression flag
        self.assertEqual(result[0], 0x03)  # Format flag
        self.assertEqual(result[1], 0x00)  # No compression flag


class TestMetadataDecompression(unittest.TestCase):
    """Test metadata decompression functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.compressor = MetadataCompressor()
    
    def test_decompress_empty_data(self):
        """Test decompression of empty data"""
        result = self.compressor.decompress_metadata(b'')
        self.assertEqual(result, {})
    
    def test_roundtrip_simple_metadata(self):
        """Test compression/decompression roundtrip for simple metadata"""
        original = {
            'entropy_metadata': {
                'encoding_method': 'huffman',
                'min_val': 0.0,
                'max_val': 1.0,
                'entropy': 2.5,
                'n_bins': 256
            }
        }
        
        compressed = self.compressor.compress_metadata(original)
        decompressed = self.compressor.decompress_metadata(compressed)
        
        self.assertEqual(decompressed, original)
    
    def test_roundtrip_complex_metadata(self):
        """Test roundtrip for complex metadata"""
        original = {
            'version': 1,
            'prime': 257,
            'precision': 6,
            'original_shape': [100, 200],
            'entropy_metadata': {
                'encoding_method': 'arithmetic',
                'min_val': -10.5,
                'max_val': 10.5,
                'entropy': 3.2,
                'n_bins': 512,
                'frequency_table': {1: 10, 2: 20, 3: 15}
            },
            'sparse_indices': [1, 5, 10, 15],
            'valuations': [0.1, 0.2, 0.3, 0.4]
        }
        
        compressed = self.compressor.compress_metadata(original)
        decompressed = self.compressor.decompress_metadata(compressed)
        
        self.assertEqual(decompressed, original)
    
    def test_roundtrip_pattern_dict(self):
        """Test roundtrip for metadata with pattern dictionary"""
        original = {
            'entropy_metadata': {
                'encoding_method': 'huffman'
            },
            'pattern_dict': {
                42: b'pattern_data_1',
                99: b'pattern_data_2',
                123: b'another_pattern'
            }
        }
        
        compressed = self.compressor.compress_metadata(original)
        decompressed = self.compressor.decompress_metadata(compressed)
        
        self.assertEqual(decompressed, original)
        self.assertIsInstance(decompressed['pattern_dict'], dict)
        # Check that keys are integers and values are bytes
        for key, value in decompressed['pattern_dict'].items():
            self.assertIsInstance(key, int)
            self.assertIsInstance(value, bytes)
    
    def test_roundtrip_with_bytes(self):
        """Test roundtrip for metadata with byte arrays"""
        original = {
            'entropy_metadata': {
                'encoding_method': 'custom',
                'binary_data': b'some binary data here\x00\x01\x02\xff'
            }
        }
        
        compressed = self.compressor.compress_metadata(original)
        decompressed = self.compressor.decompress_metadata(compressed)
        
        self.assertEqual(decompressed, original)
    
    def test_decompress_legacy_format(self):
        """Test decompression of legacy format data"""
        # Create a legacy format manually
        legacy_data = bytearray()
        legacy_data.append(0x02)  # Legacy entropy format
        legacy_data.append(0x00)  # No compression
        
        # Create JSON data for entropy metadata
        entropy_meta = {
            'encoding_method': 'huffman',
            'min_val': 0.0,
            'max_val': 1.0
        }
        json_data = json.dumps(entropy_meta).encode('utf-8')
        legacy_data.extend(struct.pack('>I', len(json_data)))
        legacy_data.extend(json_data)
        
        result = self.compressor.decompress_metadata(bytes(legacy_data))
        self.assertIn('entropy_metadata', result)
        self.assertEqual(result['entropy_metadata']['encoding_method'], 'huffman')


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with old formats"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.compressor = MetadataCompressor()
    
    def test_decompress_very_old_legacy_format(self):
        """Test decompression of very old legacy format"""
        # Create a very old format (no format flag)
        legacy_data = bytearray()
        # Pack basic entropy fields
        legacy_data.extend(struct.pack('>fffi', 0.0, 1.0, 2.5, 256))
        # Add method string
        method = b'huffman\x00\x00\x00\x00\x00\x00\x00\x00\x00'[:16]
        legacy_data.extend(method)
        
        result = self.compressor.decompress_metadata(bytes(legacy_data))
        self.assertIn('entropy_metadata', result)
        
        entropy_meta = result['entropy_metadata']
        self.assertEqual(entropy_meta['min_val'], 0.0)
        self.assertEqual(entropy_meta['max_val'], 1.0)
        self.assertEqual(entropy_meta['entropy'], 2.5)
        self.assertEqual(entropy_meta['n_bins'], 256)
        self.assertEqual(entropy_meta['encoding_method'], 'huffman')
    
    def test_fallback_on_corrupted_data(self):
        """Test fallback behavior on corrupted data"""
        corrupted_data = b'\x99\x88\x77\x66\x55\x44\x33\x22\x11\x00'
        
        result = self.compressor.decompress_metadata(corrupted_data)
        self.assertIsInstance(result, dict)
        # Should have minimal fallback data
        self.assertIn('entropy_metadata', result)
        entropy_meta = result['entropy_metadata']
        self.assertEqual(entropy_meta['encoding_method'], 'huffman')
        self.assertEqual(entropy_meta['method'], 'huffman')


class TestPatternDictHandling(unittest.TestCase):
    """Test pattern dictionary key/value handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.compressor = MetadataCompressor()
    
    def test_pattern_dict_integer_keys(self):
        """Test that pattern dictionary integer keys are preserved"""
        metadata = {
            'pattern_dict': {
                1: b'pattern1',
                42: b'pattern42',
                999: b'pattern999'
            }
        }
        
        compressed = self.compressor.compress_metadata(metadata)
        decompressed = self.compressor.decompress_metadata(compressed)
        
        pattern_dict = decompressed['pattern_dict']
        self.assertIn(1, pattern_dict)
        self.assertIn(42, pattern_dict)
        self.assertIn(999, pattern_dict)
        self.assertEqual(pattern_dict[1], b'pattern1')
        self.assertEqual(pattern_dict[42], b'pattern42')
        self.assertEqual(pattern_dict[999], b'pattern999')
    
    def test_pattern_dict_byte_values(self):
        """Test that pattern dictionary byte values are preserved"""
        metadata = {
            'pattern_dict': {
                1: b'\x00\x01\x02\x03',
                2: b'\xff\xfe\xfd\xfc',
                3: b'normal text'
            }
        }
        
        compressed = self.compressor.compress_metadata(metadata)
        decompressed = self.compressor.decompress_metadata(compressed)
        
        pattern_dict = decompressed['pattern_dict']
        self.assertEqual(pattern_dict[1], b'\x00\x01\x02\x03')
        self.assertEqual(pattern_dict[2], b'\xff\xfe\xfd\xfc')
        self.assertEqual(pattern_dict[3], b'normal text')


class TestEntropyMetadataCompatibility(unittest.TestCase):
    """Test entropy metadata compatibility features"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.compressor = MetadataCompressor()
    
    def test_ensure_compatibility_fields(self):
        """Test that compatibility fields are ensured"""
        # Test with method but no encoding_method
        entropy_meta = {'method': 'arithmetic'}
        self.compressor._ensure_compatibility_fields(entropy_meta)
        self.assertEqual(entropy_meta['encoding_method'], 'arithmetic')
        
        # Test with encoding_method but no method
        entropy_meta = {'encoding_method': 'huffman'}
        self.compressor._ensure_compatibility_fields(entropy_meta)
        self.assertEqual(entropy_meta['method'], 'huffman')
        
        # Test with neither field
        entropy_meta = {}
        self.compressor._ensure_compatibility_fields(entropy_meta)
        self.assertEqual(entropy_meta['encoding_method'], 'huffman')
        self.assertEqual(entropy_meta['method'], 'huffman')
    
    def test_frequency_table_key_conversion(self):
        """Test frequency table key conversion from strings to integers"""
        metadata = {
            'entropy_metadata': {
                'encoding_method': 'huffman',
                'frequency_table': {'1': 10, '2': 20, '-5': 5}
            }
        }
        
        compressed = self.compressor.compress_metadata(metadata)
        decompressed = self.compressor.decompress_metadata(compressed)
        
        freq_table = decompressed['entropy_metadata']['frequency_table']
        self.assertIn(1, freq_table)
        self.assertIn(2, freq_table)
        self.assertIn(-5, freq_table)
        self.assertEqual(freq_table[1], 10)
        self.assertEqual(freq_table[2], 20)
        self.assertEqual(freq_table[-5], 5)


class TestSerializationHelpers(unittest.TestCase):
    """Test serialization helper methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.compressor = MetadataCompressor()
    
    def test_prepare_for_serialization_basic_types(self):
        """Test serialization of basic Python types"""
        # Test None
        self.assertIsNone(self.compressor._prepare_for_serialization(None))
        
        # Test basic types
        self.assertEqual(self.compressor._prepare_for_serialization("string"), "string")
        self.assertEqual(self.compressor._prepare_for_serialization(42), 42)
        self.assertEqual(self.compressor._prepare_for_serialization(3.14), 3.14)
        self.assertEqual(self.compressor._prepare_for_serialization(True), True)
    
    def test_prepare_for_serialization_bytes(self):
        """Test serialization of bytes to base64"""
        data = b'\x00\x01\x02\x03\xff'
        result = self.compressor._prepare_for_serialization(data)
        
        # Should be base64 encoded
        self.assertIsInstance(result, str)
        # Verify it can be decoded back
        decoded = base64.b64decode(result)
        self.assertEqual(decoded, data)
    
    def test_prepare_for_serialization_dict(self):
        """Test serialization of dictionaries with integer keys"""
        data = {
            1: "value1",
            2: b"bytes_value",
            "string_key": "value"
        }
        
        result = self.compressor._prepare_for_serialization(data)
        
        # Integer keys should be converted to strings
        self.assertIn("1", result)
        self.assertIn("2", result)
        self.assertIn("string_key", result)
    
    def test_prepare_for_serialization_lists(self):
        """Test serialization of lists and tuples"""
        data = [1, 2.0, "string", b"bytes", [3, 4]]
        result = self.compressor._prepare_for_serialization(data)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 5)
    
    def test_prepare_for_serialization_numpy_arrays(self):
        """Test serialization of numpy arrays"""
        # Test numpy array
        arr = np.array([1, 2, 3])
        result = self.compressor._prepare_for_serialization(arr)
        self.assertEqual(result, [1, 2, 3])
        
        # Test numpy scalar
        scalar = np.float32(3.14)
        result = self.compressor._prepare_for_serialization(scalar)
        self.assertAlmostEqual(result, 3.14, places=5)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.compressor = MetadataCompressor()
    
    def test_compress_none_metadata(self):
        """Test compression with None metadata"""
        # Should not crash and should return some bytes
        result = self.compressor.compress_metadata(None)
        self.assertIsInstance(result, bytes)
        
        # Should be able to decompress
        decompressed = self.compressor.decompress_metadata(result)
        self.assertIsInstance(decompressed, dict)
    
    def test_compress_invalid_json_data(self):
        """Test compression with data that can't be JSON serialized"""
        # Create object that can't be serialized
        class NotSerializable:
            def __str__(self):
                return "not_serializable"
        
        metadata = {
            'entropy_metadata': {
                'encoding_method': 'huffman',
                'bad_object': NotSerializable()
            }
        }
        
        # Should not crash (fallback behavior)
        result = self.compressor.compress_metadata(metadata)
        self.assertIsInstance(result, bytes)
    
    def test_decompress_invalid_format_flag(self):
        """Test decompression with invalid format flag"""
        invalid_data = b'\xFF\x00\x00\x00\x00'
        
        result = self.compressor.decompress_metadata(invalid_data)
        self.assertIsInstance(result, dict)
        # Should have fallback entropy metadata
        self.assertIn('entropy_metadata', result)
    
    def test_decompress_truncated_data(self):
        """Test decompression with truncated data"""
        # Create valid header but truncated data
        truncated = b'\x03\x01\x00\x00\x00\x10'  # Claims 16 bytes but no data
        
        result = self.compressor.decompress_metadata(truncated)
        self.assertIsInstance(result, dict)
    
    def test_statistics_tracking(self):
        """Test that statistics are properly tracked"""
        initial_stats = self.compressor.get_statistics()
        
        # Perform some operations
        metadata = {'entropy_metadata': {'encoding_method': 'huffman'}}
        compressed = self.compressor.compress_metadata(metadata)
        decompressed = self.compressor.decompress_metadata(compressed)
        
        # Check stats updated
        final_stats = self.compressor.get_statistics()
        self.assertGreater(final_stats['compression_attempts'], initial_stats['compression_attempts'])
        self.assertGreater(final_stats['successful_decompressions'], initial_stats['successful_decompressions'])
    
    def test_fallback_decompression_tracking(self):
        """Test that fallback decompressions are tracked"""
        initial_stats = self.compressor.get_statistics()
        
        # Force fallback with corrupted data
        corrupted = b'\x99\x88\x77'
        self.compressor.decompress_metadata(corrupted)
        
        final_stats = self.compressor.get_statistics()
        self.assertGreater(final_stats['fallback_decompressions'], initial_stats['fallback_decompressions'])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.compressor = MetadataCompressor()
    
    def test_empty_pattern_dict(self):
        """Test handling of empty pattern dictionary"""
        metadata = {'pattern_dict': {}}
        
        compressed = self.compressor.compress_metadata(metadata)
        decompressed = self.compressor.decompress_metadata(compressed)
        
        self.assertEqual(decompressed['pattern_dict'], {})
    
    def test_very_large_pattern_dict(self):
        """Test handling of very large pattern dictionary"""
        large_pattern_dict = {i: f"pattern_{i}".encode() for i in range(1000)}
        metadata = {'pattern_dict': large_pattern_dict}
        
        compressed = self.compressor.compress_metadata(metadata)
        decompressed = self.compressor.decompress_metadata(compressed)
        
        self.assertEqual(len(decompressed['pattern_dict']), 1000)
        self.assertEqual(decompressed['pattern_dict'][500], b"pattern_500")
    
    def test_unicode_in_metadata(self):
        """Test handling of unicode strings in metadata"""
        metadata = {
            'entropy_metadata': {
                'encoding_method': 'huffman',
                'unicode_field': 'héllo wörld 🌍'
            }
        }
        
        compressed = self.compressor.compress_metadata(metadata)
        decompressed = self.compressor.decompress_metadata(compressed)
        
        self.assertEqual(decompressed['entropy_metadata']['unicode_field'], 'héllo wörld 🌍')
    
    def test_nested_data_structures(self):
        """Test deeply nested data structures"""
        metadata = {
            'level1': {
                'level2': {
                    'level3': {
                        'level4': {
                            'encoding_method': 'huffman',
                            'data': [1, 2, 3]
                        }
                    }
                }
            }
        }
        
        compressed = self.compressor.compress_metadata(metadata)
        decompressed = self.compressor.decompress_metadata(compressed)
        
        self.assertEqual(decompressed['level1']['level2']['level3']['level4']['data'], [1, 2, 3])
    
    def test_mixed_data_types_in_lists(self):
        """Test lists with mixed data types"""
        metadata = {
            'mixed_list': [
                1,
                'string',
                3.14,
                True,
                None,
                [1, 2, 3],
                {'nested': 'dict'}
            ]
        }
        
        compressed = self.compressor.compress_metadata(metadata)
        decompressed = self.compressor.decompress_metadata(compressed)
        
        self.assertEqual(decompressed['mixed_list'], metadata['mixed_list'])


if __name__ == '__main__':
    unittest.main()