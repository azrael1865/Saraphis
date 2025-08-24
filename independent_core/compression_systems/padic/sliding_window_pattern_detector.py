"""
Fast Suffix Array Pattern Detector for P-adic Digit Compression
O(n log n) Pattern Detection with Overflow-Free Rolling Hash
PRODUCTION READY - FIXED INT32 PATTERN HANDLING
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Union, Any
from dataclasses import dataclass, field
import math
from collections import defaultdict
import heapq
from functools import cmp_to_key
import logging
import time
from .safe_suffix_array import SafeSuffixArrayBuilder, SuffixArrayResult

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class PatternMatch:
    """Represents a detected pattern match"""
    pattern: bytes
    positions: List[int]
    frequency: int
    hash_value: int
    length: int  # Length in BYTES (not elements)
    element_count: int  # Number of elements (for proper reconstruction)
    element_dtype: np.dtype  # Data type of elements
    
    def __post_init__(self):
        """Validate pattern match"""
        if not isinstance(self.pattern, bytes):
            raise TypeError(f"Pattern must be bytes, got {type(self.pattern)}")
        if not self.positions:
            raise ValueError("Pattern must have at least one position")
        if self.frequency != len(self.positions):
            raise ValueError(f"Frequency {self.frequency} doesn't match positions count {len(self.positions)}")
        # FIXED: length is now in bytes, matching len(self.pattern)
        if self.length != len(self.pattern):
            raise ValueError(f"Length {self.length} doesn't match pattern length {len(self.pattern)}")
        # Validate element count matches byte length
        expected_bytes = self.element_count * self.element_dtype.itemsize
        if expected_bytes != self.length:
            raise ValueError(f"Element count {self.element_count} * dtype size {self.element_dtype.itemsize} = {expected_bytes} doesn't match byte length {self.length}")


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
    element_dtype: np.dtype  # Data type of original elements


class IntervalTree:
    """Simple interval tree for efficient overlap detection"""
    
    def __init__(self):
        self.intervals = []
    
    def insert(self, start: int, end: int):
        """Insert an interval [start, end)"""
        self.intervals.append((start, end))
        if len(self.intervals) > 100:
            # Merge overlapping intervals periodically
            self._merge_intervals()
    
    def overlaps(self, start: int, end: int) -> bool:
        """Check if [start, end) overlaps with any existing interval"""
        for s, e in self.intervals:
            if not (end <= s or start >= e):
                return True
        return False
    
    def _merge_intervals(self):
        """Merge overlapping intervals to prevent quadratic growth"""
        if not self.intervals:
            return
        
        self.intervals.sort()
        merged = []
        
        for start, end in self.intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        
        self.intervals = merged


class OptimizedIntervalTree:
    """
    Optimized interval tree for O(log n) overlap detection.
    Uses augmented balanced BST for efficient interval operations.
    """
    
    class Node:
        """Node in the interval tree"""
        def __init__(self, start: int, end: int):
            self.start = start
            self.end = end
            self.max_end = end
            self.left = None
            self.right = None
            self.height = 1
    
    def __init__(self):
        """Initialize empty interval tree"""
        self.root = None
        self.size = 0
        self._merge_threshold = 1000
    
    def insert(self, start: int, end: int):
        """Insert an interval [start, end) in O(log n) time"""
        if start >= end:
            return
        
        self.root = self._insert_node(self.root, start, end)
        self.size += 1
        
        if self.size > self._merge_threshold:
            self._rebuild_tree()
    
    def overlaps(self, start: int, end: int) -> bool:
        """Check if [start, end) overlaps with any existing interval in O(log n) time"""
        if start >= end:
            return False
        
        return self._overlaps_node(self.root, start, end)
    
    def _insert_node(self, node, start: int, end: int):
        """Insert interval into subtree rooted at node"""
        if node is None:
            return self.Node(start, end)
        
        if start < node.start:
            node.left = self._insert_node(node.left, start, end)
        else:
            node.right = self._insert_node(node.right, start, end)
        
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))
        node.max_end = max(node.end,
                          self._get_max_end(node.left),
                          self._get_max_end(node.right))
        
        return self._balance(node)
    
    def _overlaps_node(self, node, start: int, end: int) -> bool:
        """Check if query interval overlaps with any interval in subtree"""
        if node is None:
            return False
        
        if not (end <= node.start or start >= node.end):
            return True
        
        if node.left is not None and node.left.max_end > start:
            if self._overlaps_node(node.left, start, end):
                return True
        
        if node.right is not None and node.start < end:
            if self._overlaps_node(node.right, start, end):
                return True
        
        return False
    
    def _get_height(self, node):
        """Get height of node"""
        return node.height if node else 0
    
    def _get_max_end(self, node):
        """Get max_end of node"""
        return node.max_end if node else float('-inf')
    
    def _get_balance(self, node):
        """Get balance factor of node"""
        if node is None:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)
    
    def _balance(self, node):
        """Balance the tree using AVL rotations"""
        if node is None:
            return None
        
        balance = self._get_balance(node)
        
        if balance > 1:
            if self._get_balance(node.left) < 0:
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        
        if balance < -1:
            if self._get_balance(node.right) > 0:
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        
        return node
    
    def _rotate_left(self, z):
        """Perform left rotation"""
        y = z.right
        T2 = y.left
        
        y.left = z
        z.right = T2
        
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        
        z.max_end = max(z.end, self._get_max_end(z.left), self._get_max_end(z.right))
        y.max_end = max(y.end, self._get_max_end(y.left), self._get_max_end(y.right))
        
        return y
    
    def _rotate_right(self, z):
        """Perform right rotation"""
        y = z.left
        T3 = y.right
        
        y.right = z
        z.left = T3
        
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        
        z.max_end = max(z.end, self._get_max_end(z.left), self._get_max_end(z.right))
        y.max_end = max(y.end, self._get_max_end(y.left), self._get_max_end(y.right))
        
        return y
    
    def _collect_intervals(self, node, intervals):
        """Collect all intervals in the tree"""
        if node is None:
            return
        
        self._collect_intervals(node.left, intervals)
        intervals.append((node.start, node.end))
        self._collect_intervals(node.right, intervals)
    
    def _merge_intervals(self, intervals):
        """Merge overlapping intervals"""
        if not intervals:
            return []
        
        intervals.sort()
        merged = [intervals[0]]
        
        for start, end in intervals[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        
        return merged
    
    def _rebuild_tree(self):
        """Rebuild tree with merged intervals to prevent degradation"""
        intervals = []
        self._collect_intervals(self.root, intervals)
        
        merged = self._merge_intervals(intervals)
        
        if len(merged) < len(intervals) * 0.8:
            self.root = None
            self.size = 0
            
            for start, end in merged:
                self.root = self._insert_node(self.root, start, end)
                self.size += 1
    
    @property
    def intervals(self):
        """Get all intervals (for compatibility with basic IntervalTree)"""
        result = []
        self._collect_intervals(self.root, result)
        return result
    
    def clear(self):
        """Clear all intervals"""
        self.root = None
        self.size = 0


class OverflowFreeRollingHash:
    """Rolling hash with proper modular arithmetic to prevent overflow"""
    
    MODULUS = 2147483647  # 2^31 - 1
    BASE = 257  # Prime larger than alphabet size (256)
    
    def __init__(self, window_size: int):
        """Initialize rolling hash with precomputed powers"""
        self.window_size = window_size
        self.base_power = 1
        
        for _ in range(window_size - 1):
            self.base_power = (self.base_power * self.BASE) % self.MODULUS
    
    def compute_hash(self, data: np.ndarray, start: int) -> int:
        """Compute hash for window starting at position start"""
        h = 0
        for i in range(self.window_size):
            if start + i >= len(data):
                break
            h = (h * self.BASE + int(data[start + i])) % self.MODULUS
        return h
    
    def roll_hash(self, old_hash: int, old_byte: int, new_byte: int) -> int:
        """Roll hash by removing old byte and adding new byte"""
        old_contribution = (old_byte * self.base_power) % self.MODULUS
        h = (old_hash - old_contribution + self.MODULUS) % self.MODULUS
        
        h = (h * self.BASE + new_byte) % self.MODULUS
        return h


class SlidingWindowPatternDetector(nn.Module):
    """
    Fast pattern detector using suffix arrays and rolling hash.
    O(n log n) time complexity with overflow-free operations.
    FIXED: Proper handling of int32 p-adic digit patterns.
    """
    
    def __init__(
        self,
        min_pattern_length: int = 4,
        max_pattern_length: int = 32,
        min_frequency: int = 3,
        hash_prime: int = 257,
        device: str = 'cpu',
        enable_compile: bool = True,
        max_patterns: int = 1000,
        use_suffix_array: bool = True
    ):
        """Initialize fast pattern detector"""
        super().__init__()
        
        if min_pattern_length < 2:
            raise ValueError(f"min_pattern_length must be >= 2, got {min_pattern_length}")
        if max_pattern_length < min_pattern_length:
            raise ValueError(f"max_pattern_length {max_pattern_length} < min_pattern_length {min_pattern_length}")
        if min_frequency < 2:
            raise ValueError(f"min_frequency must be >= 2, got {min_frequency}")
        
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.min_frequency = min_frequency
        self.device = torch.device(device)
        self.max_patterns = max_patterns
        self.use_suffix_array = use_suffix_array
        
        self.suffix_builder = SafeSuffixArrayBuilder(
            max_length=10_000_000,
            device=str(self.device)
        )
        
        self.rolling_hashes = {}
        for size in range(min_pattern_length, min(max_pattern_length + 1, 128)):
            self.rolling_hashes[size] = OverflowFreeRollingHash(size)
    
    def _convert_input_safe(self, data: Union[torch.Tensor, np.ndarray, bytes]) -> Tuple[np.ndarray, np.dtype]:
        """
        Convert input data safely without corruption for p-adic digits.
        Returns both the data and its dtype for proper pattern handling.
        """
        if isinstance(data, bytes):
            return np.frombuffer(data, dtype=np.uint8), np.dtype('uint8')
        
        elif isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
            
            if data_np.dtype in [np.int32, np.int64]:
                min_val, max_val = data_np.min(), data_np.max()
                
                if min_val >= 0:
                    # P-adic digits - keep as int32
                    return data_np.astype(np.int32), np.dtype('int32')
                else:
                    # Other integer data
                    if min_val >= -128 and max_val <= 127:
                        return data_np.astype(np.int8), np.dtype('int8')
                    elif min_val >= -32768 and max_val <= 32767:
                        return data_np.astype(np.int16), np.dtype('int16')
                    else:
                        return data_np.astype(np.int32), np.dtype('int32')
            
            elif data_np.dtype in [np.float32, np.float64]:
                rounded = np.round(data_np).astype(np.int32)
                return rounded, np.dtype('int32')
            
            else:
                return data_np, data_np.dtype
        
        elif isinstance(data, np.ndarray):
            if data.dtype in [np.int32, np.int64]:
                min_val, max_val = data.min(), data.max()
                
                if min_val >= 0:
                    return data.astype(np.int32), np.dtype('int32')
                else:
                    if min_val >= -128 and max_val <= 127:
                        return data.astype(np.int8), np.dtype('int8')
                    elif min_val >= -32768 and max_val <= 32767:
                        return data.astype(np.int16), np.dtype('int16')
                    else:
                        return data.astype(np.int32), np.dtype('int32')
            
            elif data.dtype in [np.float32, np.float64]:
                rounded = np.round(data).astype(np.int32)
                return rounded, np.dtype('int32')
            
            else:
                return data, data.dtype
        
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    def _find_patterns_suffix_array(self, data: np.ndarray, element_dtype: np.dtype) -> Dict[int, PatternMatch]:
        """Find patterns using suffix array algorithm - O(n log n) with proper dtype handling"""
        n = len(data)
        if n < self.min_pattern_length:
            return {}
        
        sa_result = self.suffix_builder.build(data)
        suffix_array = sa_result.suffix_array
        lcp_array = sa_result.lcp_array
        
        patterns_found = {}
        pattern_id = 0
        
        interval_tree = OptimizedIntervalTree()
        
        pattern_candidates = []
        
        i = 0
        while i < n - 1:
            lcp_val = lcp_array[i]
            if lcp_val >= self.min_pattern_length:
                pattern_length = min(lcp_val, self.max_pattern_length)
                
                positions = [suffix_array[i]]
                j = i + 1
                
                while j < n:
                    if j - 1 < len(lcp_array) and lcp_array[j - 1] >= pattern_length:
                        positions.append(suffix_array[j])
                        j += 1
                    else:
                        break
                
                if len(positions) >= self.min_frequency:
                    pattern_start = positions[0]
                    pattern_data = data[pattern_start:pattern_start + pattern_length]
                    
                    value = pattern_length * len(positions)
                    
                    # Store pattern with element information
                    pattern_candidates.append((
                        value,  # Positive value now
                        pattern_length,
                        pattern_data.tobytes(),  # Convert to bytes immediately
                        positions
                    ))
                
                i = j
            else:
                i += 1

        # Sort by value (descending), then by pattern length (descending) to prefer longer patterns
        pattern_candidates.sort(key=lambda x: (-x[0], -x[1]))

        # Process candidates in sorted order
        for value, pattern_element_count, pattern_bytes, positions in pattern_candidates:
            if len(patterns_found) >= self.max_patterns:
                break
            
            valid_positions = []
            
            for pos in positions:
                if not interval_tree.overlaps(pos, pos + pattern_element_count):
                    valid_positions.append(pos)
            
            if len(valid_positions) >= self.min_frequency:
                # pattern_bytes is already converted to bytes
                
                # Create pattern match with proper length tracking
                pattern_match = PatternMatch(
                    pattern=pattern_bytes,  # Already in bytes format
                    positions=sorted(valid_positions),
                    frequency=len(valid_positions),
                    hash_value=hash(pattern_bytes),
                    length=len(pattern_bytes),  # Length in BYTES
                    element_count=pattern_element_count,  # Number of elements
                    element_dtype=element_dtype  # Data type for reconstruction
                )
                
                patterns_found[pattern_id] = pattern_match
                pattern_id += 1
                
                for pos in valid_positions:
                    interval_tree.insert(pos, pos + pattern_element_count)
        
        return patterns_found
    
    def _find_patterns_rolling_hash(self, data: np.ndarray, element_dtype: np.dtype) -> Dict[int, PatternMatch]:
        """Find patterns using rolling hash with proper dtype handling"""
        n = len(data)
        patterns_found = {}
        pattern_id = 0
        covered_positions = set()
        
        for window_size in range(min(self.max_pattern_length, n), 
                                self.min_pattern_length - 1, -1):
            if window_size not in self.rolling_hashes:
                continue
                
            hasher = self.rolling_hashes[window_size]
            hash_to_positions = defaultdict(list)
            
            current_hash = hasher.compute_hash(data, 0)
            hash_to_positions[current_hash].append(0)
            
            for i in range(1, n - window_size + 1):
                old_byte = int(data[i - 1])
                new_byte = int(data[i + window_size - 1])
                current_hash = hasher.roll_hash(current_hash, old_byte, new_byte)
                
                hash_to_positions[current_hash].append(i)
            
            for hash_val, positions in hash_to_positions.items():
                if len(positions) < self.min_frequency:
                    continue
                
                ref_pattern = data[positions[0]:positions[0] + window_size]
                valid_positions = []
                
                for pos in positions:
                    if np.array_equal(data[pos:pos + window_size], ref_pattern):
                        # Check if this position overlaps with already covered positions
                        overlaps = any(pos < cp + clen and pos + window_size > cp 
                                     for cp, clen in covered_positions)
                        if not overlaps:
                            valid_positions.append(pos)
                
                if len(valid_positions) >= self.min_frequency:
                    pattern_bytes = ref_pattern.tobytes()
                    
                    # Calculate compression benefit for this pattern
                    element_byte_size = element_dtype.itemsize
                    original_bytes = window_size * element_byte_size * len(valid_positions)
                    
                    # More realistic reference size calculation based on data size
                    data_size = len(data)
                    if data_size <= 255:
                        reference_size = 1  # uint8 for small datasets
                    elif data_size <= 65535:
                        reference_size = 2  # uint16 for medium datasets  
                    else:
                        reference_size = 4  # uint32 for large datasets
                    
                    compressed_bytes = len(pattern_bytes) + len(valid_positions) * reference_size
                    compression_benefit = original_bytes - compressed_bytes
                    
                    # Be more lenient for small datasets and short patterns
                    # Accept patterns that save at least 1 byte per 2 occurrences
                    min_benefit = -len(valid_positions) // 2  # Allow some overhead for small patterns
                    
                    if compression_benefit < min_benefit:
                        continue  # Skip patterns that don't compress well enough
                    
                    # FIXED: Proper length tracking
                    pattern_match = PatternMatch(
                        pattern=pattern_bytes,
                        positions=sorted(valid_positions),
                        frequency=len(valid_positions),
                        hash_value=hash_val,
                        length=len(pattern_bytes),  # Byte length
                        element_count=window_size,  # Element count
                        element_dtype=element_dtype  # Data type
                    )
                    
                    patterns_found[pattern_id] = pattern_match
                    pattern_id += 1
                    
                    for pos in valid_positions:
                        covered_positions.add((pos, window_size))
                    
                    if len(patterns_found) >= self.max_patterns:
                        return patterns_found
        
        return patterns_found
    
    def find_patterns(
        self,
        data: Union[torch.Tensor, np.ndarray, bytes],
        batch_process: bool = True
    ) -> PatternDetectionResult:
        """Find repeated patterns in input data with proper dtype handling"""
        # FIXED: Get both data and dtype
        data, element_dtype = self._convert_input_safe(data)
        
        if data.ndim != 1:
            data = data.flatten()
        
        n = len(data)
        
        if n < self.min_pattern_length * self.min_frequency:
            return PatternDetectionResult(
                patterns={},
                pattern_mask=torch.zeros(n, dtype=torch.bool, device=self.device),
                pattern_indices=torch.full((n,), -1, dtype=torch.int32, device=self.device),
                compression_potential=0.0,
                total_patterns_found=0,
                bytes_replaced=0,
                original_size=n * element_dtype.itemsize,  # Size in bytes
                element_dtype=element_dtype
            )
        
        if n > 100:
            sample_size = min(1000, n)
            sample_data = data[:sample_size] if n > sample_size else data
            unique_bytes = len(np.unique(sample_data))
            entropy_ratio = unique_bytes / sample_size
            
            if entropy_ratio > 0.8:
                return PatternDetectionResult(
                    patterns={},
                    pattern_mask=torch.zeros(n, dtype=torch.bool, device=self.device),
                    pattern_indices=torch.full((n,), -1, dtype=torch.int32, device=self.device),
                    compression_potential=0.0,
                    total_patterns_found=0,
                    bytes_replaced=0,
                    original_size=n * element_dtype.itemsize,
                    element_dtype=element_dtype
                )
        
        algorithm_start_time = time.perf_counter()
        
        # Choose algorithm and pass element_dtype
        if n <= 10000:
            patterns_found = self._find_patterns_rolling_hash(data, element_dtype)
            algorithm_used = "rolling_hash_small"
        elif n <= 50000:
            if entropy_ratio > 0.6:
                patterns_found = self._find_patterns_rolling_hash(data, element_dtype)
                algorithm_used = "rolling_hash_medium"
            else:
                patterns_found = self._find_patterns_suffix_array(data, element_dtype)
                algorithm_used = "suffix_array_medium"
        else:
            patterns_found = self._find_patterns_suffix_array(data, element_dtype)
            algorithm_used = "suffix_array_large"
        
        algorithm_time = time.perf_counter() - algorithm_start_time
        
        if algorithm_time > 1.0:
            logger.info(f"Pattern detection: {algorithm_used}, {n} elements, {algorithm_time:.3f}s")
        
        pattern_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        pattern_indices = torch.full((n,), -1, dtype=torch.int32, device=self.device)
        
        bytes_replaced = 0
        for pattern_id, pattern_match in patterns_found.items():
            for pos in pattern_match.positions:
                # Use element_count for mask indexing
                pattern_mask[pos:pos + pattern_match.element_count] = True
                pattern_indices[pos:pos + pattern_match.element_count] = pattern_id
                bytes_replaced += pattern_match.length  # Count bytes
        
        pattern_overhead = len(patterns_found) * 4
        reference_size = sum(len(p.positions) * 4 for p in patterns_found.values())
        original_byte_size = n * element_dtype.itemsize
        compressed_size = original_byte_size - bytes_replaced + pattern_overhead + reference_size
        compression_potential = 1.0 - (compressed_size / original_byte_size) if original_byte_size > 0 else 0.0
        
        return PatternDetectionResult(
            patterns=patterns_found,
            pattern_mask=pattern_mask,
            pattern_indices=pattern_indices,
            compression_potential=compression_potential,
            total_patterns_found=len(patterns_found),
            bytes_replaced=bytes_replaced,
            original_size=original_byte_size,
            element_dtype=element_dtype
        )
    
    def encode_with_patterns(
        self,
        data: Union[torch.Tensor, np.ndarray, bytes],
        pattern_result: Optional[PatternDetectionResult] = None
    ) -> Tuple[torch.Tensor, Dict[int, bytes], torch.Tensor]:
        """Encode data by replacing detected patterns with indices"""
        # Convert input, preserving dtype information
        if isinstance(data, bytes):
            data = torch.frombuffer(data, dtype=torch.uint8).to(self.device)
            element_dtype = np.dtype('uint8')
        elif isinstance(data, np.ndarray):
            element_dtype = data.dtype
            # Preserve original dtype in torch tensor
            if element_dtype == np.int32:
                data = torch.from_numpy(data).to(torch.int32).to(self.device)
            elif element_dtype == np.int16:
                data = torch.from_numpy(data).to(torch.int16).to(self.device)
            else:
                data = torch.from_numpy(data).to(torch.uint8).to(self.device)
        elif isinstance(data, torch.Tensor):
            data = data.to(self.device)
            # Infer dtype from tensor
            if data.dtype == torch.int32:
                element_dtype = np.dtype('int32')
            elif data.dtype == torch.int16:
                element_dtype = np.dtype('int16')
            else:
                element_dtype = np.dtype('uint8')
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        if data.dim() != 1:
            data = data.flatten()
        
        if pattern_result is None:
            pattern_result = self.find_patterns(data)
        
        if not pattern_result.patterns:
            return data, {}, torch.tensor([], dtype=torch.int32, device=self.device)
        
        pattern_dictionary = {}
        pattern_lengths = []
        
        pattern_positions = []
        for pattern_id, pattern_match in pattern_result.patterns.items():
            pattern_dictionary[pattern_id] = pattern_match.pattern
            # Store element count, not byte length
            pattern_lengths.append(pattern_match.element_count)
            for pos in pattern_match.positions:
                pattern_positions.append((pos, pattern_id, pattern_match.element_count))
        
        pattern_positions.sort(key=lambda x: x[0])
        
        encoded_data = []
        skip_until = 0
        
        for i in range(data.size(0)):
            if i < skip_until:
                continue
            
            pattern_found = False
            for pos, pattern_id, element_count in pattern_positions:
                if pos == i:
                    marker = torch.tensor([256 + pattern_id], dtype=torch.int32, device=self.device)
                    encoded_data.append(marker)
                    skip_until = i + element_count
                    pattern_found = True
                    break
            
            if not pattern_found:
                data_value = data[i:i+1].to(torch.int32)
                # If data value conflicts with pattern markers (>= 256), handle specially
                if data_value.item() >= 256:
                    # Encode as special escape sequence: [255, actual_value]
                    escape_marker = torch.tensor([255], dtype=torch.int32, device=self.device)
                    encoded_data.append(escape_marker)
                    encoded_data.append(data_value)
                else:
                    encoded_data.append(data_value)
        
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
        pattern_lengths: torch.Tensor,
        element_dtype: Optional[np.dtype] = None
    ) -> torch.Tensor:
        """Decode data by replacing pattern indices with original patterns"""
        if encoded_data.dim() != 1:
            raise ValueError(f"Encoded data must be 1D, got shape {encoded_data.shape}")
        
        # FIXED: Proper dtype handling for p-adic digits
        if element_dtype is not None:
            target_dtype = element_dtype
            if element_dtype == np.dtype('int32'):
                torch_dtype = torch.int32
            elif element_dtype == np.dtype('int16'):
                torch_dtype = torch.int16
            elif element_dtype == np.dtype('int8'):
                torch_dtype = torch.int8
            else:
                torch_dtype = torch.uint8
        else:
            target_dtype = np.dtype('uint8')
            torch_dtype = torch.uint8
        
        decoded_parts = []
        
        i = 0
        while i < encoded_data.size(0):
            value = encoded_data[i].item()
            
            if value == 255:
                # Escape sequence: next value is the actual data value
                if i + 1 < encoded_data.size(0):
                    actual_value = encoded_data[i + 1].item()
                    decoded_parts.append(torch.tensor([actual_value], dtype=torch_dtype, device=self.device))
                    i += 2  # Skip both escape marker and value
                else:
                    raise ValueError("Incomplete escape sequence at end of data")
            elif value >= 256:
                pattern_id = value - 256
                if pattern_id in pattern_dictionary:
                    pattern_bytes = pattern_dictionary[pattern_id]
                    # Decode using appropriate dtype
                    pattern_array = np.frombuffer(pattern_bytes, dtype=target_dtype).copy()
                    pattern_tensor = torch.from_numpy(pattern_array).to(dtype=torch_dtype, device=self.device)
                    decoded_parts.append(pattern_tensor)
                    i += 1
                else:
                    raise ValueError(f"Invalid pattern ID: {pattern_id}")
            else:
                decoded_parts.append(torch.tensor([value], dtype=torch_dtype, device=self.device))
                i += 1
        
        if decoded_parts:
            return torch.cat(decoded_parts)
        else:
            return torch.tensor([], dtype=torch_dtype, device=self.device)
    
    def forward(
        self, 
        data: Union[torch.Tensor, np.ndarray, bytes]
    ) -> Tuple[torch.Tensor, Dict[int, bytes], Dict[str, Any]]:
        """
        Forward pass for nn.Module compatibility.
        Returns encoded data, pattern dictionary, and metadata.
        """
        # Find patterns
        pattern_result = self.find_patterns(data)
        
        # Encode with patterns
        encoded_data, pattern_dict, pattern_lengths = self.encode_with_patterns(
            data, 
            pattern_result
        )
        
        # Create metadata dictionary
        metadata = {
            'pattern_lengths': pattern_lengths,
            'element_dtype': pattern_result.element_dtype,
            'compression_potential': pattern_result.compression_potential,
            'total_patterns_found': pattern_result.total_patterns_found,
            'bytes_replaced': pattern_result.bytes_replaced,
            'original_size': pattern_result.original_size
        }
        
        return encoded_data, pattern_dict, metadata
    
    def __call__(
        self, 
        data: Union[torch.Tensor, np.ndarray, bytes]
    ) -> Tuple[torch.Tensor, Dict[int, bytes], Dict[str, Any]]:
        """
        Make the module callable. Delegates to forward method.
        """
        return self.forward(data)
    
    def analyze_compression_efficiency(
        self,
        data: Union[torch.Tensor, np.ndarray, bytes]
    ) -> Dict[str, Any]:
        """Analyze potential compression efficiency for given data"""
        pattern_result = self.find_patterns(data)
        
        encoded_data, pattern_dict, pattern_lengths = self.encode_with_patterns(
            data, 
            pattern_result
        )
        
        # Calculate metrics using proper byte sizes
        original_size = pattern_result.original_size
        encoded_size = encoded_data.size(0) * 4
        pattern_dict_size = sum(len(p) for p in pattern_dict.values())
        total_compressed_size = encoded_size + pattern_dict_size
        
        compression_ratio = original_size / total_compressed_size if total_compressed_size > 0 else 1.0
        space_savings = 1.0 - (total_compressed_size / original_size) if original_size > 0 else 0.0
        
        pattern_stats = {}
        if pattern_result.patterns:
            # Use element_count for statistics
            pattern_element_counts = [p.element_count for p in pattern_result.patterns.values()]
            pattern_freqs = [p.frequency for p in pattern_result.patterns.values()]
            
            pattern_stats = {
                'num_patterns': len(pattern_result.patterns),
                'avg_pattern_length': np.mean(pattern_element_counts),
                'max_pattern_length': max(pattern_element_counts),
                'min_pattern_length': min(pattern_element_counts),
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
    """Benchmark the fixed pattern detector with int32 data"""
    import time
    
    detector = SlidingWindowPatternDetector(
        min_pattern_length=4,
        max_pattern_length=32,
        min_frequency=3,
        use_suffix_array=True
    )
    
    # Test with int32 data (simulating p-adic digits)
    np.random.seed(42)
    
    # Create int32 pattern (p-adic digits are typically 0-256)
    base_pattern = np.random.randint(0, 257, 20, dtype=np.int32)
    
    test_data = []
    for _ in range(100):
        if np.random.random() < 0.3:
            test_data.extend(base_pattern)
        else:
            test_data.extend(np.random.randint(0, 257, 10, dtype=np.int32))
    
    short_pattern = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    for i in range(0, len(test_data), 50):
        test_data[i:i+5] = short_pattern
    
    test_data = np.array(test_data, dtype=np.int32)
    
    print(f"Test data: {len(test_data)} int32 elements")
    print(f"Byte size: {len(test_data) * 4} bytes")
    
    # Test pattern detection
    start_time = time.time()
    pattern_result = detector.find_patterns(test_data)
    detection_time = time.time() - start_time
    
    print(f"\nPattern Detection Results:")
    print(f"  Time: {detection_time:.4f} seconds")
    print(f"  Patterns found: {pattern_result.total_patterns_found}")
    print(f"  Bytes replaced: {pattern_result.bytes_replaced}")
    print(f"  Compression potential: {pattern_result.compression_potential:.2%}")
    print(f"  Element dtype: {pattern_result.element_dtype}")
    
    # Test encoding
    start_time = time.time()
    encoded_data, pattern_dict, pattern_lengths = detector.encode_with_patterns(
        test_data, 
        pattern_result
    )
    encoding_time = time.time() - start_time
    
    print(f"\nEncoding Results:")
    print(f"  Time: {encoding_time:.4f} seconds")
    print(f"  Original size: {len(test_data) * 4} bytes")
    print(f"  Encoded size: {encoded_data.size(0) * 4} bytes")
    
    # Test decoding with proper dtype
    start_time = time.time()
    decoded_data = detector.decode_with_patterns(
        encoded_data, 
        pattern_dict, 
        pattern_lengths,
        element_dtype=np.dtype('int32')  # Pass the correct dtype
    )
    decoding_time = time.time() - start_time
    
    print(f"\nDecoding Results:")
    print(f"  Time: {decoding_time:.4f} seconds")
    print(f"  Decoded size: {decoded_data.size(0)} elements")
    
    # Verify reconstruction
    original_tensor = torch.from_numpy(test_data)
    match = torch.equal(decoded_data.cpu(), original_tensor)
    print(f"  Reconstruction accurate: {match}")
    
    if not match:
        print(f"  ERROR: Mismatch at indices where values differ")
        diff_indices = torch.where(decoded_data.cpu() != original_tensor)[0]
        if len(diff_indices) > 0:
            print(f"  First mismatch at index {diff_indices[0]}: {decoded_data[diff_indices[0]]} != {original_tensor[diff_indices[0]]}")
    
    # Analyze compression efficiency
    analysis = detector.analyze_compression_efficiency(test_data)
    
    print(f"\nCompression Analysis:")
    print(f"  Compression ratio: {analysis['compression_ratio']:.2f}x")
    print(f"  Space savings: {analysis['space_savings_percent']:.1f}%")
    if analysis['pattern_statistics']:
        stats = analysis['pattern_statistics']
        print(f"  Average pattern length: {stats['avg_pattern_length']:.1f} elements")
        print(f"  Average pattern frequency: {stats['avg_pattern_frequency']:.1f}")


if __name__ == "__main__":
    benchmark_pattern_detector()