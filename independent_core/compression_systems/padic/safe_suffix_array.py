"""
Safe Suffix Array Builder for Pattern Detection
HARD FAILURES - NO FALLBACKS
"""

import numpy as np
import torch
from typing import Optional, Tuple
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class SuffixArrayResult:
    """Result of suffix array construction"""
    suffix_array: np.ndarray
    lcp_array: np.ndarray
    build_time: float
    memory_used_mb: float


class SafeSuffixArrayBuilder:
    """
    Production suffix array builder with hard validation
    NO FALLBACKS - FAILS HARD ON ANY ISSUE
    """
    
    def __init__(self, max_length: int = 10_000_000, device: str = "cpu"):
        """
        Initialize suffix array builder
        
        Args:
            max_length: Maximum supported input length
            device: Device for computations
            
        Raises:
            ValueError: If parameters are invalid
        """
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        
        self.max_length = max_length
        self.device = device
        self.arrays_built = 0
        self.total_build_time = 0.0
        
        logger.info(f"SafeSuffixArrayBuilder initialized (max_length={max_length}, device={device})")
    
    def build(self, data: np.ndarray) -> SuffixArrayResult:
        """
        Build suffix array for input data
        FAILS HARD if construction fails
        
        Args:
            data: Input data array (int32, uint16, or uint8)
            
        Returns:
            SuffixArrayResult with suffix array, LCP array, and metrics
            
        Raises:
            ValueError: If input is invalid
            RuntimeError: If construction fails
        """
        start_time = time.perf_counter()
        
        # Validate input
        if not isinstance(data, np.ndarray):
            raise ValueError(f"Input must be numpy array, got {type(data)}")
        
        # Accept multiple integer types that pattern detector might produce
        acceptable_dtypes = [np.int32, np.int16, np.int8, np.uint32, np.uint16, np.uint8]
        
        if data.dtype not in acceptable_dtypes:
            raise ValueError(f"Input must be integer type, got {data.dtype}. Acceptable types: {acceptable_dtypes}")
        
        # Convert to int32 for internal processing
        # This ensures consistent behavior regardless of input type
        if data.dtype != np.int32:
            logger.debug(f"Converting {data.dtype} to int32 for suffix array construction")
            data = data.astype(np.int32)
        
        if len(data) == 0:
            raise ValueError("Input data cannot be empty")
        
        if len(data) > self.max_length:
            raise ValueError(f"Input length {len(data)} exceeds maximum {self.max_length}")
        
        # Build suffix array using DC3 algorithm
        try:
            suffix_array = self._build_suffix_array_dc3(data)
        except Exception as e:
            raise RuntimeError(f"Suffix array construction failed: {e}") from e
        
        # Build LCP array
        try:
            lcp_array = self._build_lcp_array(data, suffix_array)
        except Exception as e:
            raise RuntimeError(f"LCP array construction failed: {e}") from e
        
        build_time = time.perf_counter() - start_time
        memory_used = (suffix_array.nbytes + lcp_array.nbytes) / (1024**2)
        
        # Update statistics
        self.arrays_built += 1
        self.total_build_time += build_time
        
        result = SuffixArrayResult(
            suffix_array=suffix_array,
            lcp_array=lcp_array,
            build_time=build_time,
            memory_used_mb=memory_used
        )
        
        logger.info(f"Built suffix array: length={len(data)}, time={build_time:.3f}s, memory={memory_used:.1f}MB")
        
        return result
    
    def validate(self, data: np.ndarray, suffix_array: np.ndarray) -> None:
        """
        Validate suffix array correctness
        FAILS HARD if validation fails
        
        Args:
            data: Original input data
            suffix_array: Suffix array to validate
            
        Raises:
            ValueError: If validation fails
        """
        n = len(data)
        
        # Check basic properties
        if len(suffix_array) != n:
            raise ValueError(f"Suffix array length {len(suffix_array)} != data length {n}")
        
        # Check all indices are present
        expected_indices = set(range(n))
        actual_indices = set(suffix_array)
        
        if actual_indices != expected_indices:
            missing = expected_indices - actual_indices
            extra = actual_indices - expected_indices
            raise ValueError(f"Invalid suffix array indices. Missing: {missing}, Extra: {extra}")
        
        # Check lexicographic ordering
        for i in range(n - 1):
            suffix1_start = suffix_array[i]
            suffix2_start = suffix_array[i + 1]
            
            # Compare suffixes lexicographically
            if not self._suffix_less_than(data, suffix1_start, suffix2_start):
                raise ValueError(f"Suffix array not sorted: position {i} violates order")
        
        logger.info(f"Suffix array validation passed (length={n})")
    
    def _build_suffix_array_dc3(self, data: np.ndarray) -> np.ndarray:
        """
        Build suffix array using DC3/Skew algorithm
        O(n) average time complexity
        """
        n = len(data)
        
        if n <= 2:
            if n == 1:
                return np.array([0], dtype=np.int32)
            elif n == 2:
                return np.array([0, 1] if data[0] <= data[1] else [1, 0], dtype=np.int32)
        
        # For small arrays, use simple sorting
        if n <= 1000:
            return self._build_suffix_array_simple(data)
        
        # DC3 algorithm for larger arrays
        return self._dc3_algorithm(data)
    
    def _build_suffix_array_simple(self, data: np.ndarray) -> np.ndarray:
        """Simple O(n^2 log n) suffix array for small inputs"""
        n = len(data)
        suffixes = list(range(n))
        
        def compare_suffixes(i: int, j: int) -> int:
            while i < n and j < n:
                if data[i] < data[j]:
                    return -1
                elif data[i] > data[j]:
                    return 1
                i += 1
                j += 1
            
            if i >= n:
                return -1 if j < n else 0
            return 1
        
        from functools import cmp_to_key
        suffixes.sort(key=cmp_to_key(compare_suffixes))
        
        return np.array(suffixes, dtype=np.int32)
    
    def _dc3_algorithm(self, data: np.ndarray) -> np.ndarray:
        """
        DC3/Skew algorithm implementation
        Linear time suffix array construction
        """
        n = len(data)
        
        # Add padding to handle array bounds
        padded_data = np.concatenate([data, [0, 0, 0]])
        
        # Step 1: Build suffix array for positions ≡ 1,2 (mod 3)
        sample_positions = []
        for i in range(n):
            if i % 3 != 0:
                sample_positions.append(i)
        
        if not sample_positions:
            # All positions are ≡ 0 (mod 3)
            return self._build_mod0_only(data)
        
        # Create triples for sample positions
        triples = []
        for pos in sample_positions:
            triple = (padded_data[pos], padded_data[pos + 1], padded_data[pos + 2])
            triples.append((triple, pos))
        
        # Sort triples
        triples.sort()
        
        # Assign ranks to triples
        ranks = [0] * len(sample_positions)
        current_rank = 0
        prev_triple = None
        
        for i, (triple, pos) in enumerate(triples):
            if triple != prev_triple:
                current_rank += 1
                prev_triple = triple
            
            pos_index = sample_positions.index(pos)
            ranks[pos_index] = current_rank
        
        # Check if ranks are unique
        if len(set(ranks)) == len(ranks):
            # Ranks are unique, extract suffix array
            sample_sa = [sample_positions[i] for _, i in sorted(zip(ranks, range(len(ranks))))]
        else:
            # Recursively solve
            recursive_sa = self._build_suffix_array_dc3(np.array(ranks, dtype=np.int32))
            sample_sa = [sample_positions[i] for i in recursive_sa]
        
        # Step 2: Build suffix array for positions ≡ 0 (mod 3)
        mod0_positions = [i for i in range(n) if i % 3 == 0]
        
        # Create rank mapping for sample positions
        rank_map = {}
        for rank, pos in enumerate(sample_sa):
            rank_map[pos] = rank
        
        # Sort mod 0 positions
        mod0_with_rank = []
        for pos in mod0_positions:
            first_char = padded_data[pos]
            next_rank = rank_map.get(pos + 1, -1)
            mod0_with_rank.append((first_char, next_rank, pos))
        
        mod0_with_rank.sort()
        mod0_sa = [pos for _, _, pos in mod0_with_rank]
        
        # Step 3: Merge sample and mod0 suffix arrays
        return self._merge_suffix_arrays(padded_data, sample_sa, mod0_sa, n, rank_map)
    
    def _build_mod0_only(self, data: np.ndarray) -> np.ndarray:
        """Handle case where all positions are mod 0"""
        n = len(data)
        positions = list(range(0, n, 3))
        
        def compare_suffixes(i: int, j: int) -> int:
            while i < n and j < n:
                if data[i] < data[j]:
                    return -1
                elif data[i] > data[j]:
                    return 1
                i += 1
                j += 1
            
            if i >= n:
                return -1 if j < n else 0
            return 1
        
        from functools import cmp_to_key
        positions.sort(key=cmp_to_key(compare_suffixes))
        
        return np.array(positions, dtype=np.int32)
    
    def _merge_suffix_arrays(self, data: np.ndarray, sample_sa: list, mod0_sa: list, 
                           n: int, rank_map: dict) -> np.ndarray:
        """Merge sample and mod0 suffix arrays"""
        merged = []
        i = j = 0
        
        while i < len(sample_sa) and j < len(mod0_sa):
            pos1 = sample_sa[i]
            pos2 = mod0_sa[j]
            
            if pos1 >= n:
                merged.append(mod0_sa[j])
                j += 1
                continue
            
            if pos2 >= n:
                merged.append(sample_sa[i])
                i += 1
                continue
            
            # Compare suffixes
            if self._suffix_less_than(data, pos1, pos2):
                merged.append(pos1)
                i += 1
            else:
                merged.append(pos2)
                j += 1
        
        # Add remaining elements
        while i < len(sample_sa):
            if sample_sa[i] < n:
                merged.append(sample_sa[i])
            i += 1
        
        while j < len(mod0_sa):
            if mod0_sa[j] < n:
                merged.append(mod0_sa[j])
            j += 1
        
        return np.array(merged, dtype=np.int32)
    
    def _suffix_less_than(self, data: np.ndarray, i: int, j: int) -> bool:
        """Compare two suffixes lexicographically"""
        n = len(data)
        
        while i < n and j < n:
            if data[i] < data[j]:
                return True
            elif data[i] > data[j]:
                return False
            i += 1
            j += 1
        
        return i >= n and j < n
    
    def _build_lcp_array(self, data: np.ndarray, suffix_array: np.ndarray) -> np.ndarray:
        """
        Build LCP (Longest Common Prefix) array using Kasai's algorithm
        O(n) time complexity
        """
        n = len(data)
        lcp = np.zeros(n, dtype=np.int32)
        
        if n <= 1:
            return lcp
        
        # Build rank array (inverse of suffix array)
        rank = np.zeros(n, dtype=np.int32)
        rank[suffix_array] = np.arange(n)
        
        # Kasai's algorithm
        k = 0
        for i in range(n):
            if rank[i] == n - 1:
                k = 0
                continue
            
            j = suffix_array[rank[i] + 1]
            
            # Compute LCP
            while (i + k < n and j + k < n and 
                   data[i + k] == data[j + k]):
                k += 1
            
            lcp[rank[i]] = k
            
            if k > 0:
                k -= 1
        
        return lcp