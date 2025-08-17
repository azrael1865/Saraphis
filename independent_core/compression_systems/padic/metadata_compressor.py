"""
Fixed metadata_compressor.py - Preserves complete entropy metadata including frequency_table
"""

import struct
import json
import zlib
import logging
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MetadataHeader:
    """
    Fixed-size metadata header for compatibility
    """
    version: int = 1
    prime: int = 257
    precision: int = 6
    compression_flags: int = 0
    original_shape: Tuple[int, ...] = field(default_factory=tuple)


class MetadataCompressor:
    """Enhanced metadata compressor with complete field preservation"""
    
    def __init__(self):
        self.compression_stats = {
            'fallback_decompressions': 0,
            'successful_decompressions': 0,
            'compression_attempts': 0
        }
    
    def compress_metadata(self, metadata: Dict[str, Any]) -> bytes:
        """
        Public API for compressing metadata dictionary
        """
        if 'entropy_metadata' in metadata:
            return self._compress_entropy_metadata(metadata['entropy_metadata'])
        else:
            # Compress the whole metadata dict as entropy metadata
            return self._compress_entropy_metadata(metadata)
    
    def decompress_metadata(self, compressed: bytes) -> Dict[str, Any]:
        """
        Public API for decompressing metadata from bytes
        """
        result, _ = self._decompress_entropy_metadata(compressed, 0)
        return result
        
    def _compress_entropy_metadata(self, entropy_meta: Dict[str, Any]) -> bytes:
        """
        Compress entropy metadata preserving ALL fields with enhanced functionality
        """
        if not entropy_meta:
            return b''
        
        self.compression_stats['compression_attempts'] += 1
        
        try:
            # Prepare metadata for JSON serialization
            serializable_meta = self._prepare_for_serialization(entropy_meta)
            
            # Serialize to JSON
            json_str = json.dumps(serializable_meta, separators=(',', ':'))
            json_bytes = json_str.encode('utf-8')
            
            # Build compressed format with proper header
            result = bytearray()
            
            # Format flag (0x02 for enhanced JSON with full metadata)
            result.append(0x02)
            
            # Apply compression if beneficial
            if len(json_bytes) > 100:
                compressed = zlib.compress(json_bytes, level=6)
                
                # Flag for compressed data
                result.append(0x01)
                
                # Original length (4 bytes)
                result.extend(struct.pack('>I', len(json_bytes)))
                
                # Compressed data length (4 bytes)
                result.extend(struct.pack('>I', len(compressed)))
                
                # Compressed data
                result.extend(compressed)
            else:
                # Flag for uncompressed data
                result.append(0x00)
                
                # Data length (4 bytes)
                result.extend(struct.pack('>I', len(json_bytes)))
                
                # Uncompressed JSON data
                result.extend(json_bytes)
            
            return bytes(result)
            
        except Exception as e:
            logger.error(f"Error compressing entropy metadata: {e}")
            # Fallback to legacy format
            return self._compress_entropy_metadata_legacy(entropy_meta)
    
    def _prepare_for_serialization(self, data: Any) -> Any:
        """Recursively prepare data for JSON serialization"""
        if data is None:
            return None
        elif isinstance(data, (str, int, float, bool)):
            return data
        elif isinstance(data, dict):
            return {k: self._prepare_for_serialization(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._prepare_for_serialization(item) for item in data]
        elif hasattr(data, 'tolist'):  # numpy arrays/tensors
            return data.tolist()
        elif hasattr(data, 'item'):  # single-element tensors
            return data.item()
        else:
            return str(data)
    
    def _decompress_entropy_metadata(self, data: bytes, offset: int) -> Tuple[Dict[str, Any], int]:
        """
        Decompress entropy metadata with full field preservation and backwards compatibility
        """
        if offset >= len(data):
            return {}, offset
        
        try:
            # Check format flag
            format_flag = data[offset]
            offset += 1
            
            logger.debug(f"Decompressing entropy metadata - format_flag: {format_flag:#x}, remaining bytes: {len(data) - offset}")
            
            # Enhanced format with full metadata preservation
            if format_flag == 0x02:
                return self._decompress_enhanced_format(data, offset)
            
            # Old compressed JSON format
            elif format_flag == 0x01:
                return self._decompress_old_json_format(data, offset)
            
            # Legacy format - try to recover what we can
            else:
                offset -= 1  # Reset offset since this wasn't a format flag
                return self._decompress_with_recovery(data, offset)
                
        except Exception as e:
            logger.warning(f"Error in entropy metadata decompression: {e}, falling back to recovery mode")
            return self._decompress_with_recovery(data, offset)
    
    def _decompress_enhanced_format(self, data: bytes, offset: int) -> Tuple[Dict[str, Any], int]:
        """Decompress enhanced format with full metadata"""
        try:
            # Check compression flag
            compression_flag = data[offset]
            offset += 1
            
            if compression_flag == 0x01:
                # Compressed data
                if offset + 8 > len(data):
                    raise ValueError("Insufficient data for compressed header")
                
                original_length = struct.unpack_from('>I', data, offset)[0]
                offset += 4
                compressed_length = struct.unpack_from('>I', data, offset)[0]
                offset += 4
                
                if offset + compressed_length > len(data):
                    raise ValueError("Insufficient data for compressed content")
                
                compressed_data = data[offset:offset + compressed_length]
                json_bytes = zlib.decompress(compressed_data)
                
                if len(json_bytes) != original_length:
                    raise ValueError(f"Decompression size mismatch: expected {original_length}, got {len(json_bytes)}")
                
                offset += compressed_length
                
            else:
                # Uncompressed data
                if offset + 4 > len(data):
                    raise ValueError("Insufficient data for uncompressed header")
                
                data_length = struct.unpack_from('>I', data, offset)[0]
                offset += 4
                
                if offset + data_length > len(data):
                    raise ValueError("Insufficient data for uncompressed content")
                
                json_bytes = data[offset:offset + data_length]
                offset += data_length
            
            # Parse JSON
            json_str = json_bytes.decode('utf-8')
            entropy_meta = json.loads(json_str)
            
            # Ensure backward compatibility fields
            self._ensure_compatibility_fields(entropy_meta)
            
            self.compression_stats['successful_decompressions'] += 1
            return entropy_meta, offset
            
        except Exception as e:
            logger.error(f"Failed to decompress enhanced format: {e}")
            raise
    
    def _decompress_old_json_format(self, data: bytes, offset: int) -> Tuple[Dict[str, Any], int]:
        """Decompress old JSON format (backward compatibility)"""
        try:
            if offset + 4 > len(data):
                raise ValueError("Insufficient data for old JSON format")
            
            original_length = struct.unpack_from('>I', data, offset)[0]
            offset += 4
            
            remaining_data = data[offset:]
            if not remaining_data:
                raise ValueError("No compressed data found")
            
            decompressed = zlib.decompress(remaining_data)
            
            if len(decompressed) == original_length:
                json_str = decompressed.decode('utf-8')
                entropy_meta = json.loads(json_str)
                
                self._ensure_compatibility_fields(entropy_meta)
                return entropy_meta, len(data)
            else:
                raise ValueError(f"Decompression size mismatch")
                
        except Exception as e:
            logger.warning(f"Failed to decompress old JSON format: {e}")
            raise
    
    def _decompress_with_recovery(self, data: bytes, offset: int) -> Tuple[Dict[str, Any], int]:
        """
        Attempt to recover metadata using multiple strategies
        """
        # Strategy 1: Try legacy format
        try:
            result, new_offset = self._decompress_entropy_metadata_legacy(data, offset)
            if result:
                self._ensure_compatibility_fields(result)
                return result, new_offset
        except:
            pass
        
        # Strategy 2: Try to find JSON data directly
        try:
            # Look for JSON markers
            remaining = data[offset:]
            if remaining.startswith(b'{') or remaining.startswith(b'{"'):
                # Find end of JSON
                end_pos = remaining.find(b'}')
                if end_pos > 0:
                    json_data = remaining[:end_pos + 1]
                    entropy_meta = json.loads(json_data.decode('utf-8'))
                    self._ensure_compatibility_fields(entropy_meta)
                    return entropy_meta, offset + end_pos + 1
        except:
            pass
        
        # Strategy 3: Try compressed JSON without header
        try:
            remaining = data[offset:]
            if len(remaining) > 0:
                decompressed = zlib.decompress(remaining)
                entropy_meta = json.loads(decompressed.decode('utf-8'))
                self._ensure_compatibility_fields(entropy_meta)
                return entropy_meta, len(data)
        except:
            pass
        
        # Final fallback: Return minimal metadata
        self.compression_stats['fallback_decompressions'] += 1
        return {
            'encoding_method': 'huffman',
            'method': 'huffman'
        }, offset
    
    def _decompress_entropy_metadata_legacy(self, data: bytes, offset: int) -> Tuple[Dict[str, Any], int]:
        """
        Legacy entropy metadata decompression for backwards compatibility
        """
        entropy_meta = {}
        
        if offset >= len(data):
            return entropy_meta, offset
        
        try:
            # Basic entropy fields (legacy format)
            if offset + 16 <= len(data):
                min_val, max_val, entropy, n_bins = struct.unpack_from('>fffi', data, offset)
                offset += 16
                
                entropy_meta.update({
                    'min_val': float(min_val),
                    'max_val': float(max_val),
                    'entropy': float(entropy),
                    'n_bins': int(n_bins)
                })
                
                # Read encoding method string
                if offset < len(data):
                    method_length = min(16, len(data) - offset)
                    method_bytes = data[offset:offset + method_length]
                    
                    # Find null terminator if present
                    null_pos = method_bytes.find(0)
                    if null_pos >= 0:
                        method_bytes = method_bytes[:null_pos]
                    
                    try:
                        encoding_method = method_bytes.decode('utf-8')
                        entropy_meta['encoding_method'] = encoding_method
                        entropy_meta['method'] = encoding_method
                    except UnicodeDecodeError:
                        entropy_meta['encoding_method'] = 'huffman'
                        entropy_meta['method'] = 'huffman'
                    
                    offset += method_length
        
        except (struct.error, UnicodeDecodeError) as e:
            logger.warning(f"Legacy decompression error: {e}")
        
        return entropy_meta, offset
    
    def _ensure_compatibility_fields(self, entropy_meta: Dict[str, Any]) -> None:
        """Ensure all required fields are present for compatibility"""
        # Ensure encoding_method is present
        if 'encoding_method' not in entropy_meta and 'method' in entropy_meta:
            entropy_meta['encoding_method'] = entropy_meta['method']
        elif 'encoding_method' not in entropy_meta:
            entropy_meta['encoding_method'] = 'huffman'
        
        # Ensure method is present
        if 'method' not in entropy_meta and 'encoding_method' in entropy_meta:
            entropy_meta['method'] = entropy_meta['encoding_method']
        elif 'method' not in entropy_meta:
            entropy_meta['method'] = 'huffman'
        
        # Convert string keys in frequency_table to integers if needed
        if 'frequency_table' in entropy_meta and entropy_meta['frequency_table']:
            freq_table = entropy_meta['frequency_table']
            if isinstance(freq_table, dict):
                first_key = next(iter(freq_table.keys()), None)
                if first_key is not None and isinstance(first_key, str):
                    try:
                        entropy_meta['frequency_table'] = {
                            int(k) if k.isdigit() or (k.startswith('-') and k[1:].isdigit()) else k: v 
                            for k, v in freq_table.items()
                        }
                    except:
                        pass
    
    def _compress_entropy_metadata_legacy(self, entropy_meta: Dict[str, Any]) -> bytes:
        """Legacy compression format for fallback"""
        result = bytearray()
        
        # Basic fields
        min_val = entropy_meta.get('min_val', 0.0)
        max_val = entropy_meta.get('max_val', 1.0)
        entropy = entropy_meta.get('entropy', 0.0)
        n_bins = entropy_meta.get('n_bins', 256)
        
        result.extend(struct.pack('>fffi', min_val, max_val, entropy, n_bins))
        
        # Encoding method
        method = entropy_meta.get('encoding_method', entropy_meta.get('method', 'huffman'))
        method_bytes = method.encode('utf-8')[:16]
        method_bytes = method_bytes.ljust(16, b'\x00')
        result.extend(method_bytes)
        
        return bytes(result)