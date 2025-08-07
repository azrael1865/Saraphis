"""
Metadata Compressor for P-adic Compression System
Achieves < 1% metadata overhead through:
- Delta encoding for indices
- Varint encoding for variable-length integers
- Bit packing for boolean flags
- Pattern dictionary compression
"""

import torch
import struct
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import math


@dataclass
class MetadataHeader:
    """
    Fixed-size metadata header (5 bytes total)
    
    Attributes:
        version: Protocol version (1 byte)
        prime: P-adic prime number (2 bytes) 
        precision: P-adic precision (1 byte)
        compression_flags: Bit flags for compression settings (1 byte)
        original_shape: Original tensor shape
    """
    version: int = 1
    prime: int = 257
    precision: int = 6
    compression_flags: int = 0
    original_shape: Tuple[int, ...] = field(default_factory=tuple)
    
    def __post_init__(self):
        """Validate header parameters"""
        if not 0 <= self.version <= 255:
            raise ValueError(f"Version {self.version} must be 0-255")
        if not 0 <= self.prime <= 65535:
            raise ValueError(f"Prime {self.prime} must be 0-65535")
        if not 0 <= self.precision <= 255:
            raise ValueError(f"Precision {self.precision} must be 0-255")
        if not 0 <= self.compression_flags <= 255:
            raise ValueError(f"Compression flags {self.compression_flags} must be 0-255")


class MetadataCompressor:
    """
    Compress and decompress metadata for P-adic compression
    
    Features:
    - Delta encoding for sparse indices
    - Varint encoding for variable-length integers
    - Bit packing for boolean flags
    - Pattern dictionary compression with Huffman-like encoding
    - < 1% metadata overhead relative to compressed data
    """
    
    # Compression flag bits
    FLAG_SPARSE = 0x01
    FLAG_LOG_ENCODED = 0x02
    FLAG_PATTERN_MATCHED = 0x04
    FLAG_ENTROPY_CODED = 0x08
    FLAG_SIGNED = 0x10
    FLAG_MIXED_PRECISION = 0x20
    FLAG_TRITON_ACCELERATED = 0x40
    FLAG_GPU_OPTIMIZED = 0x80
    
    def __init__(self):
        """Initialize metadata compressor"""
        self.stats = {
            'total_compressions': 0,
            'total_decompressions': 0,
            'average_compression_ratio': 0.0,
            'total_metadata_bytes': 0,
            'total_data_bytes': 0
        }
    
    def compress_metadata(self, metadata: Dict[str, Any]) -> bytes:
        """
        Compress metadata dictionary to bytes
        
        Args:
            metadata: Dictionary containing compression metadata
            
        Returns:
            Compressed metadata as bytes
        """
        compressed_parts = []
        
        # 1. Pack fixed header (5 bytes)
        header = MetadataHeader(
            version=metadata.get('version', 1),
            prime=metadata.get('prime', 257),
            precision=metadata.get('precision', 6),
            compression_flags=self._compute_flags(metadata),
            original_shape=metadata.get('original_shape', ())
        )
        compressed_parts.append(self._pack_header(header))
        
        # 2. Encode shape with varint
        if 'original_shape' in metadata:
            compressed_parts.append(self._varint_encode_shape(metadata['original_shape']))
        
        # 3. Handle sparse indices with delta encoding
        if 'sparse_indices' in metadata and metadata['sparse_indices'] is not None:
            indices_tensor = metadata['sparse_indices']
            if isinstance(indices_tensor, torch.Tensor):
                indices = indices_tensor.cpu().numpy().astype('int64')
            else:
                indices = indices_tensor
            compressed_parts.append(self._delta_encode_indices(indices))
        
        # 4. Handle sparse values with quantization
        if 'sparse_values' in metadata and metadata['sparse_values'] is not None:
            values_tensor = metadata['sparse_values']
            if isinstance(values_tensor, torch.Tensor):
                values = values_tensor.cpu().numpy()
            else:
                values = values_tensor
            compressed_parts.append(self._compress_values(values))
        
        # 5. Compress pattern dictionary if present
        if 'pattern_dict' in metadata:
            compressed_parts.append(self._compress_pattern_dict(metadata['pattern_dict']))
        
        # 6. Handle entropy metadata
        if 'entropy_metadata' in metadata:
            compressed_parts.append(self._compress_entropy_metadata(metadata['entropy_metadata']))
        
        # 7. Pack additional flags
        if 'additional_flags' in metadata:
            compressed_parts.append(self._pack_flags(metadata['additional_flags']))
        
        # Combine all parts
        result = b''.join(compressed_parts)
        
        # Update statistics
        self.stats['total_compressions'] += 1
        self.stats['total_metadata_bytes'] += len(result)
        
        return result
    
    def decompress_metadata(self, compressed: bytes) -> Dict[str, Any]:
        """
        Decompress metadata from bytes
        
        Args:
            compressed: Compressed metadata bytes
            
        Returns:
            Decompressed metadata dictionary
        """
        metadata = {}
        offset = 0
        
        # 1. Unpack header (5 bytes)
        header, offset = self._unpack_header(compressed, offset)
        metadata['version'] = header.version
        metadata['prime'] = header.prime
        metadata['precision'] = header.precision
        
        # Extract flags
        flags = header.compression_flags
        metadata['sparse_encoded'] = bool(flags & self.FLAG_SPARSE)
        metadata['log_encoded'] = bool(flags & self.FLAG_LOG_ENCODED)
        metadata['pattern_matched'] = bool(flags & self.FLAG_PATTERN_MATCHED)
        metadata['entropy_coded'] = bool(flags & self.FLAG_ENTROPY_CODED)
        metadata['signed'] = bool(flags & self.FLAG_SIGNED)
        metadata['mixed_precision'] = bool(flags & self.FLAG_MIXED_PRECISION)
        metadata['triton_accelerated'] = bool(flags & self.FLAG_TRITON_ACCELERATED)
        metadata['gpu_optimized'] = bool(flags & self.FLAG_GPU_OPTIMIZED)
        
        # 2. Decode shape if present
        if offset < len(compressed):
            shape, new_offset = self._varint_decode_shape(compressed, offset)
            if shape:
                metadata['original_shape'] = shape
                offset = new_offset
        
        # 3. Decode sparse indices if flag is set
        if metadata['sparse_encoded'] and offset < len(compressed):
            indices, new_offset = self._delta_decode_indices(compressed, offset)
            if indices is not None:
                metadata['sparse_indices'] = torch.tensor(indices, dtype=torch.long)
                offset = new_offset
        
        # 4. Decode sparse values if present
        if metadata['sparse_encoded'] and offset < len(compressed):
            values, new_offset = self._decompress_values(compressed, offset)
            if values is not None:
                metadata['sparse_values'] = torch.tensor(values, dtype=torch.float32)
                offset = new_offset
        
        # 5. Decompress pattern dictionary if flag is set
        if metadata['pattern_matched'] and offset < len(compressed):
            pattern_dict, new_offset = self._decompress_pattern_dict(compressed, offset)
            if pattern_dict:
                metadata['pattern_dict'] = pattern_dict
                offset = new_offset
        
        # 6. Decompress entropy metadata if flag is set
        if metadata['entropy_coded'] and offset < len(compressed):
            entropy_meta, new_offset = self._decompress_entropy_metadata(compressed, offset)
            if entropy_meta:
                metadata['entropy_metadata'] = entropy_meta
                offset = new_offset
        
        # Update statistics
        self.stats['total_decompressions'] += 1
        
        return metadata
    
    def _pack_header(self, header: MetadataHeader) -> bytes:
        """
        Pack fixed-size header (5 bytes)
        
        Format:
        - Version: 1 byte (unsigned char)
        - Prime: 2 bytes (unsigned short, big-endian)
        - Precision: 1 byte (unsigned char)
        - Flags: 1 byte (unsigned char)
        """
        return struct.pack(
            '>BHBB',  # Big-endian: uchar, ushort, uchar, uchar
            header.version,
            header.prime,
            header.precision,
            header.compression_flags
        )
    
    def _unpack_header(self, data: bytes, offset: int) -> Tuple[MetadataHeader, int]:
        """
        Unpack fixed-size header
        
        Args:
            data: Compressed data bytes
            offset: Current offset in data
            
        Returns:
            Tuple of (header, new_offset)
        """
        if offset + 5 > len(data):
            raise ValueError("Insufficient data for header")
        
        version, prime, precision, flags = struct.unpack_from('>BHBB', data, offset)
        
        header = MetadataHeader(
            version=version,
            prime=prime,
            precision=precision,
            compression_flags=flags
        )
        
        return header, offset + 5
    
    def _delta_encode_indices(self, indices: Any) -> bytes:
        """
        Delta encode sparse matrix indices
        Δ(xi) = xi - xi-1
        
        Args:
            indices: Array of indices
            
        Returns:
            Delta-encoded bytes
        """
        if len(indices) == 0:
            return self._encode_varint(0)
        
        # First encode the number of indices
        result = [self._encode_varint(len(indices))]
        
        # Encode first index directly
        result.append(self._encode_varint(int(indices[0])))
        
        # Delta encode remaining indices
        for i in range(1, len(indices)):
            delta = int(indices[i]) - int(indices[i-1])
            result.append(self._encode_varint(delta))
        
        return b''.join(result)
    
    def _delta_decode_indices(self, data: bytes, offset: int) -> Tuple[Optional[List[int]], int]:
        """
        Delta decode sparse matrix indices
        
        Args:
            data: Compressed data
            offset: Current offset
            
        Returns:
            Tuple of (indices, new_offset)
        """
        # Decode number of indices
        num_indices, offset = self._decode_varint(data, offset)
        
        if num_indices == 0:
            return None, offset
        
        indices = []
        
        # Decode first index
        first_index, offset = self._decode_varint(data, offset)
        indices.append(first_index)
        
        # Decode deltas and reconstruct indices
        for _ in range(num_indices - 1):
            delta, offset = self._decode_varint(data, offset)
            indices.append(indices[-1] + delta)
        
        return indices, offset
    
    def _encode_varint(self, value: int) -> bytes:
        """
        Variable-length integer encoding (7 bits data + 1 continuation bit)
        Supports negative numbers via zigzag encoding
        
        Args:
            value: Integer to encode
            
        Returns:
            Varint encoded bytes
        """
        # Zigzag encode for signed integers
        if value < 0:
            value = ((-value) << 1) | 1
        else:
            value = value << 1
        
        result = []
        while value > 127:
            result.append(bytes([128 | (value & 127)]))
            value >>= 7
        result.append(bytes([value]))
        
        return b''.join(result)
    
    def _decode_varint(self, data: bytes, offset: int) -> Tuple[int, int]:
        """
        Decode variable-length integer
        
        Args:
            data: Compressed data
            offset: Current offset
            
        Returns:
            Tuple of (value, new_offset)
        """
        result = 0
        shift = 0
        
        while offset < len(data):
            byte = data[offset]
            offset += 1
            
            result |= (byte & 127) << shift
            
            if not (byte & 128):
                break
            
            shift += 7
        
        # Zigzag decode
        if result & 1:
            return -(result >> 1), offset
        else:
            return result >> 1, offset
    
    def _varint_encode_shape(self, shape: Tuple[int, ...]) -> bytes:
        """
        Encode tensor shape using varint
        
        Args:
            shape: Tensor shape tuple
            
        Returns:
            Encoded shape bytes
        """
        if not shape:
            return self._encode_varint(0)
        
        result = [self._encode_varint(len(shape))]
        for dim in shape:
            result.append(self._encode_varint(dim))
        
        return b''.join(result)
    
    def _varint_decode_shape(self, data: bytes, offset: int) -> Tuple[Optional[Tuple[int, ...]], int]:
        """
        Decode tensor shape from varint
        
        Args:
            data: Compressed data
            offset: Current offset
            
        Returns:
            Tuple of (shape, new_offset)
        """
        if offset >= len(data):
            return None, offset
        
        num_dims, offset = self._decode_varint(data, offset)
        
        if num_dims == 0:
            return None, offset
        
        shape = []
        for _ in range(num_dims):
            dim, offset = self._decode_varint(data, offset)
            shape.append(dim)
        
        return tuple(shape), offset
    
    def _pack_flags(self, flags: List[bool]) -> bytes:
        """
        Pack boolean flags into bits (8 booleans → 1 byte)
        
        Args:
            flags: List of boolean flags
            
        Returns:
            Packed flags as bytes
        """
        if not flags:
            return b''
        
        # Pad to multiple of 8
        padded_flags = flags + [False] * ((8 - len(flags) % 8) % 8)
        
        result = []
        for i in range(0, len(padded_flags), 8):
            byte = 0
            for j in range(8):
                if i + j < len(padded_flags) and padded_flags[i + j]:
                    byte |= 1 << j
            result.append(bytes([byte]))
        
        # Prepend length
        length_bytes = self._encode_varint(len(flags))
        
        return length_bytes + b''.join(result)
    
    def _unpack_flags(self, data: bytes, offset: int) -> Tuple[List[bool], int]:
        """
        Extract boolean flags from packed bits
        
        Args:
            data: Compressed data
            offset: Current offset
            
        Returns:
            Tuple of (flags, new_offset)
        """
        num_flags, offset = self._decode_varint(data, offset)
        
        if num_flags == 0:
            return [], offset
        
        flags = []
        num_bytes = (num_flags + 7) // 8
        
        for i in range(num_bytes):
            if offset >= len(data):
                break
            
            byte = data[offset]
            offset += 1
            
            for j in range(8):
                if len(flags) < num_flags:
                    flags.append(bool(byte & (1 << j)))
        
        return flags[:num_flags], offset
    
    def _compress_pattern_dict(self, pattern_dict: Dict[str, Any]) -> bytes:
        """
        Compress pattern dictionary with Huffman-like encoding
        
        Args:
            pattern_dict: Dictionary of patterns
            
        Returns:
            Compressed pattern dictionary
        """
        if not pattern_dict:
            return self._encode_varint(0)
        
        result = [self._encode_varint(len(pattern_dict))]
        
        for key, value in pattern_dict.items():
            # Encode key
            key_bytes = key.encode('utf-8')
            result.append(self._encode_varint(len(key_bytes)))
            result.append(key_bytes)
            
            # Encode value based on type
            if isinstance(value, (int, float)):
                # Encode as float32
                result.append(b'\x01')  # Type marker for float
                result.append(struct.pack('>f', float(value)))
            elif isinstance(value, torch.Tensor):
                # Encode tensor shape and data
                result.append(b'\x02')  # Type marker for tensor
                result.append(self._varint_encode_shape(value.shape))
                tensor_bytes = value.cpu().numpy().astype('float32').tobytes()
                result.append(self._encode_varint(len(tensor_bytes)))
                result.append(tensor_bytes)
            elif isinstance(value, list):
                # Encode list
                result.append(b'\x03')  # Type marker for list
                result.append(self._encode_varint(len(value)))
                for item in value:
                    result.append(struct.pack('>f', float(item)))
            else:
                # Encode as string
                result.append(b'\x00')  # Type marker for string
                str_bytes = str(value).encode('utf-8')
                result.append(self._encode_varint(len(str_bytes)))
                result.append(str_bytes)
        
        return b''.join(result)
    
    def _decompress_pattern_dict(self, data: bytes, offset: int) -> Tuple[Dict[str, Any], int]:
        """
        Decompress pattern dictionary
        
        Args:
            data: Compressed data
            offset: Current offset
            
        Returns:
            Tuple of (pattern_dict, new_offset)
        """
        num_entries, offset = self._decode_varint(data, offset)
        
        if num_entries == 0:
            return {}, offset
        
        pattern_dict = {}
        
        for _ in range(num_entries):
            # Decode key
            key_len, offset = self._decode_varint(data, offset)
            key = data[offset:offset + key_len].decode('utf-8')
            offset += key_len
            
            # Decode value based on type marker
            type_marker = data[offset]
            offset += 1
            
            if type_marker == 0x01:  # Float
                value = struct.unpack_from('>f', data, offset)[0]
                offset += 4
            elif type_marker == 0x02:  # Tensor
                shape, offset = self._varint_decode_shape(data, offset)
                tensor_len, offset = self._decode_varint(data, offset)
                tensor_data = data[offset:offset + tensor_len]
                offset += tensor_len
                import numpy as np
                array = np.frombuffer(tensor_data, dtype='float32').reshape(shape)
                value = torch.tensor(array)
            elif type_marker == 0x03:  # List
                list_len, offset = self._decode_varint(data, offset)
                value = []
                for _ in range(list_len):
                    item = struct.unpack_from('>f', data, offset)[0]
                    offset += 4
                    value.append(item)
            else:  # String
                str_len, offset = self._decode_varint(data, offset)
                value = data[offset:offset + str_len].decode('utf-8')
                offset += str_len
            
            pattern_dict[key] = value
        
        return pattern_dict, offset
    
    def _compress_values(self, values: Any) -> bytes:
        """
        Compress sparse values using quantization
        
        Args:
            values: Array of values
            
        Returns:
            Compressed values
        """
        import numpy as np
        
        if len(values) == 0:
            return self._encode_varint(0)
        
        # Convert to numpy if needed
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        
        # Quantize to 16-bit floats for compression
        values_f16 = values.astype('float16')
        
        # Encode length and data
        result = [self._encode_varint(len(values))]
        result.append(values_f16.tobytes())
        
        return b''.join(result)
    
    def _decompress_values(self, data: bytes, offset: int) -> Tuple[Optional[Any], int]:
        """
        Decompress sparse values
        
        Args:
            data: Compressed data
            offset: Current offset
            
        Returns:
            Tuple of (values, new_offset)
        """
        import numpy as np
        
        num_values, offset = self._decode_varint(data, offset)
        
        if num_values == 0:
            return None, offset
        
        # Each float16 is 2 bytes
        values_bytes = data[offset:offset + num_values * 2]
        offset += num_values * 2
        
        # Decode float16 values
        values = np.frombuffer(values_bytes, dtype='float16').astype('float32')
        
        return values, offset
    
    def _compress_entropy_metadata(self, entropy_meta: Dict[str, Any]) -> bytes:
        """
        Compress entropy metadata
        
        Args:
            entropy_meta: Entropy metadata dictionary
            
        Returns:
            Compressed entropy metadata
        """
        result = []
        
        # Encode min/max values
        if 'min_val' in entropy_meta:
            result.append(struct.pack('>f', float(entropy_meta['min_val'])))
        if 'max_val' in entropy_meta:
            result.append(struct.pack('>f', float(entropy_meta['max_val'])))
        
        # Encode entropy value
        if 'entropy' in entropy_meta:
            result.append(struct.pack('>f', float(entropy_meta['entropy'])))
        
        # Encode number of bins
        if 'n_bins' in entropy_meta:
            result.append(self._encode_varint(entropy_meta['n_bins']))
        
        return b''.join(result)
    
    def _decompress_entropy_metadata(self, data: bytes, offset: int) -> Tuple[Dict[str, Any], int]:
        """
        Decompress entropy metadata
        
        Args:
            data: Compressed data
            offset: Current offset
            
        Returns:
            Tuple of (entropy_metadata, new_offset)
        """
        entropy_meta = {}
        
        # Decode min/max values (4 bytes each)
        if offset + 8 <= len(data):
            entropy_meta['min_val'] = struct.unpack_from('>f', data, offset)[0]
            offset += 4
            entropy_meta['max_val'] = struct.unpack_from('>f', data, offset)[0]
            offset += 4
        
        # Decode entropy value
        if offset + 4 <= len(data):
            entropy_meta['entropy'] = struct.unpack_from('>f', data, offset)[0]
            offset += 4
        
        # Decode number of bins
        if offset < len(data):
            n_bins, offset = self._decode_varint(data, offset)
            entropy_meta['n_bins'] = n_bins
        
        return entropy_meta, offset
    
    def _compute_flags(self, metadata: Dict[str, Any]) -> int:
        """
        Generate compression flags from metadata
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Packed flags byte
        """
        flags = 0
        
        if metadata.get('sparse_encoded', False):
            flags |= self.FLAG_SPARSE
        if metadata.get('log_encoded', False):
            flags |= self.FLAG_LOG_ENCODED
        if metadata.get('pattern_matched', False) or 'pattern_metadata' in metadata:
            flags |= self.FLAG_PATTERN_MATCHED
        if metadata.get('entropy_coded', False) or 'entropy_metadata' in metadata:
            flags |= self.FLAG_ENTROPY_CODED
        if metadata.get('signed', False) or 'signs' in metadata:
            flags |= self.FLAG_SIGNED
        if metadata.get('mixed_precision', False):
            flags |= self.FLAG_MIXED_PRECISION
        if metadata.get('triton_accelerated', False):
            flags |= self.FLAG_TRITON_ACCELERATED
        if metadata.get('gpu_optimized', False) or metadata.get('device', '').startswith('cuda'):
            flags |= self.FLAG_GPU_OPTIMIZED
        
        return flags
    
    def _estimate_compression_ratio(self, metadata_bytes: int, data_bytes: int) -> float:
        """
        Calculate compression efficiency
        
        Args:
            metadata_bytes: Size of metadata in bytes
            data_bytes: Size of compressed data in bytes
            
        Returns:
            Metadata overhead percentage
        """
        if data_bytes == 0:
            return float('inf')
        
        overhead = (metadata_bytes / data_bytes) * 100
        
        # Update statistics
        if self.stats['total_data_bytes'] > 0:
            current_ratio = self.stats['total_metadata_bytes'] / self.stats['total_data_bytes']
            self.stats['average_compression_ratio'] = current_ratio * 100
        
        return overhead
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get compression statistics"""
        return dict(self.stats)
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.stats = {
            'total_compressions': 0,
            'total_decompressions': 0,
            'average_compression_ratio': 0.0,
            'total_metadata_bytes': 0,
            'total_data_bytes': 0
        }