"""
Sliding Window Pattern Detector for P-adic Digit Compression
Polynomial Rolling Hash with O(1) Window Comparison
NO PLACEHOLDERS - PRODUCTION READY
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Union, Any
from dataclasses import dataclass, field
import math
from collections import defaultdict


@dataclass
class PatternMatch:
    """Represents a detected pattern match"""
    pattern: bytes
    positions: List[int]
    frequency: int
    hash_value: int
    length: int
    
    def __post_init__(self):
        """Validate pattern match"""
        if not isinstance(self.pattern, bytes):
            raise TypeError(f"Pattern must be bytes, got {type(self.pattern)}")
        if not self.positions:
            raise ValueError("Pattern must have at least one position")
        if self.frequency != len(self.positions):
            raise ValueError(f"Frequency {self.frequency} doesn't match positions count {len(self.positions)}")
        if self.length != len(self.pattern):
            raise ValueError(f"Length {self.length} doesn't match pattern length {len(self.pattern)}")


@dataclass
class PatternDetectionResult:
    """Result of pattern detection"""
    patterns: Dict[int, PatternMatch]  # Pattern ID -> PatternMatch
    pattern_mask: torch.Tensor  # Boolean mask indicating pattern positions
    pattern_indices: torch.Tensor  # Pattern IDs at each position (-1 for no pattern)
    compression_potential: float  # Estimated compression ratio
    total_patterns_found: int
    bytes_replaced: int
    original_size: int


class SlidingWindowPatternDetector(nn.Module):
    """
    Sliding window pattern detector using polynomial rolling hash for O(1) comparison.
    Detects repeated byte sequences in p-adic digits for compression.
    """
    
    # Large primes for hashing to minimize collisions
    HASH_MODULUS = 2**31 - 1  # Mersenne prime for fast modulo
    SECONDARY_MODULUS = 2**61 - 1  # Another large prime for double hashing
    
    def __init__(
        self,
        min_pattern_length: int = 4,
        max_pattern_length: int = 32,
        min_frequency: int = 3,
        hash_prime: int = 31,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        enable_compile: bool = True
    ):
        """
        Initialize sliding window pattern detector.
        
        Args:
            min_pattern_length: Minimum pattern length to detect
            max_pattern_length: Maximum pattern length to detect
            min_frequency: Minimum occurrence frequency for a pattern
            hash_prime: Prime number for polynomial rolling hash
            device: Device for tensor operations
            enable_compile: Whether to use torch.compile optimization
        """
        super().__init__()
        
        # Validate parameters
        if min_pattern_length < 2:
            raise ValueError(f"min_pattern_length must be >= 2, got {min_pattern_length}")
        if max_pattern_length < min_pattern_length:
            raise ValueError(f"max_pattern_length {max_pattern_length} < min_pattern_length {min_pattern_length}")
        if min_frequency < 2:
            raise ValueError(f"min_frequency must be >= 2, got {min_frequency}")
        if hash_prime < 2:
            raise ValueError(f"hash_prime must be >= 2, got {hash_prime}")
        
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.min_frequency = min_frequency
        self.hash_prime = hash_prime
        self.device = torch.device(device)
        
        # Precompute prime powers for rolling hash
        self.prime_powers = self._precompute_prime_powers()
        
        # Compile critical methods if enabled
        if enable_compile and device == 'cuda':
            self._compute_rolling_hashes = torch.compile(
                self._compute_rolling_hashes,
                mode='reduce-overhead'
            )
    
    def _precompute_prime_powers(self) -> torch.Tensor:
        """Precompute prime powers for efficient rolling hash computation"""
        max_power = self.max_pattern_length
        powers = torch.zeros(max_power + 1, dtype=torch.int64, device=self.device)
        powers[0] = 1
        
        for i in range(1, max_power + 1):
            powers[i] = (powers[i-1] * self.hash_prime) % self.HASH_MODULUS
        
        return powers
    
    def _compute_rolling_hashes(
        self,
        data: torch.Tensor,
        window_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rolling polynomial hashes for all windows of given size.
        Uses double hashing to minimize collisions.
        
        Args:
            data: Input byte tensor
            window_size: Size of sliding window
            
        Returns:
            Tuple of (primary_hashes, secondary_hashes) for collision detection
        """
        if data.dim() != 1:
            raise ValueError(f"Data must be 1D tensor, got shape {data.shape}")
        
        n = data.size(0)
        if n < window_size:
            return torch.tensor([], dtype=torch.int64, device=self.device), \
                   torch.tensor([], dtype=torch.int64, device=self.device)
        
        num_windows = n - window_size + 1
        
        # Convert to int64 for hash computation
        data_int = data.to(torch.int64)
        
        # Primary hashes using HASH_MODULUS
        primary_hashes = torch.zeros(num_windows, dtype=torch.int64, device=self.device)
        
        # Compute first window hash
        first_hash = torch.zeros(1, dtype=torch.int64, device=self.device)
        for i in range(window_size):
            first_hash = (first_hash + data_int[i] * self.prime_powers[i]) % self.HASH_MODULUS
        primary_hashes[0] = first_hash
        
        # Rolling hash for subsequent windows
        # Formula: new_hash = (old_hash - old_first * prime^0) / prime + new_last * prime^(size-1)
        max_power = self.prime_powers[window_size - 1]
        prime_inv = pow(self.hash_prime, self.HASH_MODULUS - 2, self.HASH_MODULUS)  # Modular inverse
        
        for i in range(1, num_windows):
            old_byte = data_int[i - 1]
            new_byte = data_int[i + window_size - 1]
            
            # Remove old first byte and shift
            temp = (primary_hashes[i-1] - old_byte + self.HASH_MODULUS) % self.HASH_MODULUS
            temp = (temp * prime_inv) % self.HASH_MODULUS
            
            # Add new last byte
            primary_hashes[i] = (temp + new_byte * max_power) % self.HASH_MODULUS
        
        # Secondary hashes for collision detection (different prime and modulus)
        secondary_prime = 37
        secondary_hashes = torch.zeros(num_windows, dtype=torch.int64, device=self.device)
        
        # Compute secondary hashes
        for i in range(num_windows):
            window = data_int[i:i+window_size]
            hash_val = torch.zeros(1, dtype=torch.int64, device=self.device)
            for j, byte_val in enumerate(window):
                hash_val = (hash_val + byte_val * pow(secondary_prime, j, self.SECONDARY_MODULUS)) % self.SECONDARY_MODULUS
            secondary_hashes[i] = hash_val
        
        return primary_hashes, secondary_hashes
    
    def _verify_pattern_match(
        self,
        data: torch.Tensor,
        positions: List[int],
        window_size: int
    ) -> bool:
        """
        Verify that positions actually contain the same pattern (avoid hash collisions).
        
        Args:
            data: Original data tensor
            positions: List of starting positions
            window_size: Pattern length
            
        Returns:
            True if all positions contain identical patterns
        """
        if len(positions) < 2:
            return True
        
        # Get first pattern as reference
        reference = data[positions[0]:positions[0] + window_size]
        
        # Check all other positions
        for pos in positions[1:]:
            if pos + window_size > data.size(0):
                return False
            candidate = data[pos:pos + window_size]
            if not torch.equal(reference, candidate):
                return False
        
        return True
    
    def find_patterns(
        self,
        data: Union[torch.Tensor, np.ndarray, bytes],
        batch_process: bool = True
    ) -> PatternDetectionResult:
        """
        Find repeated patterns in input data using sliding window with rolling hash.
        
        Args:
            data: Input data (p-adic digits as bytes/tensor)
            batch_process: Whether to process multiple pattern lengths in parallel
            
        Returns:
            PatternDetectionResult with detected patterns and metadata
        """
        # Convert input to tensor
        if isinstance(data, bytes):
            data = torch.frombuffer(data, dtype=torch.uint8).to(self.device)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(torch.uint8).to(self.device)
        elif isinstance(data, torch.Tensor):
            data = data.to(torch.uint8).to(self.device)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        if data.dim() != 1:
            data = data.flatten()
        
        n = data.size(0)
        patterns_found = {}
        pattern_id_counter = 0
        
        # Track which positions are already covered by patterns
        covered_positions = set()
        
        # Process each window size
        for window_size in range(self.min_pattern_length, min(self.max_pattern_length + 1, n + 1)):
            if n < window_size:
                continue
            
            # Compute rolling hashes for this window size
            primary_hashes, secondary_hashes = self._compute_rolling_hashes(data, window_size)
            
            if primary_hashes.numel() == 0:
                continue
            
            # Group positions by hash value
            hash_to_positions = defaultdict(list)
            for i, (p_hash, s_hash) in enumerate(zip(primary_hashes, secondary_hashes)):
                # Use combined hash to reduce collisions
                combined_hash = (p_hash.item(), s_hash.item())
                hash_to_positions[combined_hash].append(i)
            
            # Find patterns that meet frequency threshold
            for combined_hash, positions in hash_to_positions.items():
                if len(positions) < self.min_frequency:
                    continue
                
                # Filter out positions already covered by longer patterns
                valid_positions = [
                    pos for pos in positions 
                    if not any(pos >= cp and pos < cp + clen 
                              for cp, clen in covered_positions)
                ]
                
                if len(valid_positions) < self.min_frequency:
                    continue
                
                # Verify actual pattern match (avoid hash collisions)
                if not self._verify_pattern_match(data, valid_positions, window_size):
                    continue
                
                # Extract pattern bytes
                pattern_bytes = data[valid_positions[0]:valid_positions[0] + window_size].cpu().numpy().tobytes()
                
                # Create pattern match
                pattern_match = PatternMatch(
                    pattern=pattern_bytes,
                    positions=valid_positions,
                    frequency=len(valid_positions),
                    hash_value=combined_hash[0],  # Primary hash
                    length=window_size
                )
                
                patterns_found[pattern_id_counter] = pattern_match
                pattern_id_counter += 1
                
                # Mark positions as covered
                for pos in valid_positions:
                    covered_positions.add((pos, window_size))
        
        # Create pattern mask and indices
        pattern_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        pattern_indices = torch.full((n,), -1, dtype=torch.int32, device=self.device)
        
        bytes_replaced = 0
        for pattern_id, pattern_match in patterns_found.items():
            for pos in pattern_match.positions:
                pattern_mask[pos:pos + pattern_match.length] = True
                pattern_indices[pos:pos + pattern_match.length] = pattern_id
                bytes_replaced += pattern_match.length
        
        # Calculate compression potential
        # Assuming each pattern reference takes 4 bytes (pattern ID + position)
        pattern_overhead = len(patterns_found) * 4  # Pattern dictionary overhead
        reference_size = sum(len(p.positions) * 4 for p in patterns_found.values())
        compressed_size = n - bytes_replaced + pattern_overhead + reference_size
        compression_potential = 1.0 - (compressed_size / n) if n > 0 else 0.0
        
        return PatternDetectionResult(
            patterns=patterns_found,
            pattern_mask=pattern_mask,
            pattern_indices=pattern_indices,
            compression_potential=compression_potential,
            total_patterns_found=len(patterns_found),
            bytes_replaced=bytes_replaced,
            original_size=n
        )
    
    def encode_with_patterns(
        self,
        data: Union[torch.Tensor, np.ndarray, bytes],
        pattern_result: Optional[PatternDetectionResult] = None
    ) -> Tuple[torch.Tensor, Dict[int, bytes], torch.Tensor]:
        """
        Encode data by replacing detected patterns with indices.
        
        Args:
            data: Original data to encode
            pattern_result: Pre-computed pattern detection result (optional)
            
        Returns:
            Tuple of (encoded_data, pattern_dictionary, pattern_lengths)
        """
        # Convert input to tensor
        if isinstance(data, bytes):
            data = torch.frombuffer(data, dtype=torch.uint8).to(self.device)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(torch.uint8).to(self.device)
        elif isinstance(data, torch.Tensor):
            data = data.to(torch.uint8).to(self.device)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        if data.dim() != 1:
            data = data.flatten()
        
        # Find patterns if not provided
        if pattern_result is None:
            pattern_result = self.find_patterns(data)
        
        if not pattern_result.patterns:
            # No patterns found, return original data
            return data, {}, torch.tensor([], dtype=torch.int32, device=self.device)
        
        # Create encoded output
        encoded_parts = []
        pattern_dictionary = {}
        pattern_lengths = []
        current_pos = 0
        
        # Sort patterns by position for sequential encoding
        pattern_positions = []
        for pattern_id, pattern_match in pattern_result.patterns.items():
            pattern_dictionary[pattern_id] = pattern_match.pattern
            pattern_lengths.append(pattern_match.length)
            for pos in pattern_match.positions:
                pattern_positions.append((pos, pattern_id, pattern_match.length))
        
        pattern_positions.sort(key=lambda x: x[0])
        
        # Encode data with pattern replacements
        encoded_data = []
        skip_until = 0
        
        for i in range(data.size(0)):
            if i < skip_until:
                continue
            
            # Check if current position starts a pattern
            pattern_found = False
            for pos, pattern_id, pattern_len in pattern_positions:
                if pos == i:
                    # Encode pattern reference (using special marker + pattern ID)
                    # Use values > 255 to distinguish from regular bytes
                    marker = torch.tensor([256 + pattern_id], dtype=torch.int32, device=self.device)
                    encoded_data.append(marker)
                    skip_until = i + pattern_len
                    pattern_found = True
                    break
            
            if not pattern_found:
                # Copy original byte
                encoded_data.append(data[i:i+1].to(torch.int32))
        
        # Concatenate encoded data
        if encoded_data:
            encoded_tensor = torch.cat(encoded_data)
        else:
            encoded_tensor = torch.tensor([], dtype=torch.int32, device=self.device)
        
        pattern_lengths_tensor = torch.tensor(pattern_lengths, dtype=torch.int32, device=self.device)
        
        return encoded_tensor, pattern_dictionary, pattern_lengths_tensor
    
    def decode_with_patterns(
        self,
        encoded_data: torch.Tensor,
        pattern_dictionary: Dict[int, bytes],
        pattern_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode data by replacing pattern indices with original patterns.
        
        Args:
            encoded_data: Encoded data with pattern references
            pattern_dictionary: Mapping of pattern IDs to original bytes
            pattern_lengths: Length of each pattern
            
        Returns:
            Decoded original data
        """
        if encoded_data.dim() != 1:
            raise ValueError(f"Encoded data must be 1D, got shape {encoded_data.shape}")
        
        decoded_parts = []
        
        for i in range(encoded_data.size(0)):
            value = encoded_data[i].item()
            
            if value >= 256:
                # Pattern reference
                pattern_id = value - 256
                if pattern_id in pattern_dictionary:
                    pattern_bytes = pattern_dictionary[pattern_id]
                    # Create a copy to avoid non-writable buffer warning
                    pattern_array = np.frombuffer(pattern_bytes, dtype=np.uint8).copy()
                    pattern_tensor = torch.from_numpy(pattern_array).to(self.device)
                    decoded_parts.append(pattern_tensor)
                else:
                    raise ValueError(f"Invalid pattern ID: {pattern_id}")
            else:
                # Regular byte
                decoded_parts.append(torch.tensor([value], dtype=torch.uint8, device=self.device))
        
        if decoded_parts:
            return torch.cat(decoded_parts)
        else:
            return torch.tensor([], dtype=torch.uint8, device=self.device)
    
    def analyze_compression_efficiency(
        self,
        data: Union[torch.Tensor, np.ndarray, bytes]
    ) -> Dict[str, Any]:
        """
        Analyze potential compression efficiency for given data.
        
        Args:
            data: Input data to analyze
            
        Returns:
            Dictionary with compression analysis metrics
        """
        # Find patterns
        pattern_result = self.find_patterns(data)
        
        # Encode with patterns
        encoded_data, pattern_dict, pattern_lengths = self.encode_with_patterns(
            data, 
            pattern_result
        )
        
        # Calculate metrics
        original_size = pattern_result.original_size
        encoded_size = encoded_data.size(0) * 4  # int32 elements
        pattern_dict_size = sum(len(p) for p in pattern_dict.values())
        total_compressed_size = encoded_size + pattern_dict_size
        
        compression_ratio = original_size / total_compressed_size if total_compressed_size > 0 else 1.0
        space_savings = 1.0 - (total_compressed_size / original_size) if original_size > 0 else 0.0
        
        # Pattern statistics
        pattern_stats = {}
        if pattern_result.patterns:
            pattern_lengths_list = [p.length for p in pattern_result.patterns.values()]
            pattern_freqs = [p.frequency for p in pattern_result.patterns.values()]
            
            pattern_stats = {
                'num_patterns': len(pattern_result.patterns),
                'avg_pattern_length': np.mean(pattern_lengths_list),
                'max_pattern_length': max(pattern_lengths_list),
                'min_pattern_length': min(pattern_lengths_list),
                'avg_pattern_frequency': np.mean(pattern_freqs),
                'max_pattern_frequency': max(pattern_freqs),
                'total_bytes_in_patterns': pattern_result.bytes_replaced
            }
        
        return {
            'original_size': original_size,
            'encoded_size': encoded_size,
            'pattern_dictionary_size': pattern_dict_size,
            'total_compressed_size': total_compressed_size,
            'compression_ratio': compression_ratio,
            'space_savings_percent': space_savings * 100,
            'patterns_found': pattern_result.total_patterns_found,
            'compression_potential': pattern_result.compression_potential,
            'pattern_statistics': pattern_stats
        }


def benchmark_pattern_detector():
    """Benchmark the sliding window pattern detector"""
    import time
    
    # Create detector
    detector = SlidingWindowPatternDetector(
        min_pattern_length=4,
        max_pattern_length=32,
        min_frequency=3,
        hash_prime=31
    )
    
    # Generate test data with repeating patterns
    np.random.seed(42)
    base_pattern = np.random.randint(0, 256, 20, dtype=np.uint8)
    
    # Create data with multiple occurrences of patterns
    test_data = []
    for _ in range(100):
        if np.random.random() < 0.3:  # 30% chance to insert pattern
            test_data.extend(base_pattern)
        else:
            test_data.extend(np.random.randint(0, 256, 10, dtype=np.uint8))
    
    # Add some shorter repeating patterns
    short_pattern = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    for i in range(0, len(test_data), 50):
        test_data[i:i+5] = short_pattern
    
    test_data = np.array(test_data, dtype=np.uint8)
    
    print(f"Test data size: {len(test_data)} bytes")
    
    # Benchmark pattern detection
    start_time = time.time()
    pattern_result = detector.find_patterns(test_data)
    detection_time = time.time() - start_time
    
    print(f"\nPattern Detection Results:")
    print(f"  Time: {detection_time:.4f} seconds")
    print(f"  Patterns found: {pattern_result.total_patterns_found}")
    print(f"  Bytes replaced: {pattern_result.bytes_replaced}")
    print(f"  Compression potential: {pattern_result.compression_potential:.2%}")
    
    # Benchmark encoding
    start_time = time.time()
    encoded_data, pattern_dict, pattern_lengths = detector.encode_with_patterns(
        test_data, 
        pattern_result
    )
    encoding_time = time.time() - start_time
    
    print(f"\nEncoding Results:")
    print(f"  Time: {encoding_time:.4f} seconds")
    print(f"  Original size: {len(test_data)} bytes")
    print(f"  Encoded size: {encoded_data.size(0) * 4} bytes")
    
    # Benchmark decoding
    start_time = time.time()
    decoded_data = detector.decode_with_patterns(
        encoded_data, 
        pattern_dict, 
        pattern_lengths
    )
    decoding_time = time.time() - start_time
    
    print(f"\nDecoding Results:")
    print(f"  Time: {decoding_time:.4f} seconds")
    print(f"  Decoded size: {decoded_data.size(0)} bytes")
    print(f"  Reconstruction accurate: {torch.equal(decoded_data.cpu(), torch.from_numpy(test_data))}")
    
    # Analyze compression efficiency
    analysis = detector.analyze_compression_efficiency(test_data)
    
    print(f"\nCompression Analysis:")
    print(f"  Compression ratio: {analysis['compression_ratio']:.2f}x")
    print(f"  Space savings: {analysis['space_savings_percent']:.1f}%")
    if analysis['pattern_statistics']:
        stats = analysis['pattern_statistics']
        print(f"  Average pattern length: {stats['avg_pattern_length']:.1f}")
        print(f"  Average pattern frequency: {stats['avg_pattern_frequency']:.1f}")


if __name__ == "__main__":
    # Run benchmark
    benchmark_pattern_detector()