# Saraphis P-adic Compression System: Pattern Detection Performance Fix

## PROBLEM CONTEXT

The Saraphis P-adic compression system is experiencing a critical performance bottleneck in **Stage 2 (Pattern Detection)** of its 5-stage compression pipeline. The system hangs for 10+ minutes when processing 500x500 tensors (250,000 elements), while 100x100 tensors complete in ~13 seconds. This creates an unacceptable performance cliff that blocks the entire compression pipeline.

## SYSTEM ARCHITECTURE OVERVIEW

The Saraphis system is a PyTorch-based advanced compression framework using mathematical p-adic number theory. The compression pipeline consists of:

1. **Stage 1: Adaptive Precision** - P-adic digit conversion with variable precision
2. **Stage 2: Pattern Detection** - Suffix array-based repeated pattern detection ⚠️ **BOTTLENECK HERE**
3. **Stage 3: Sparse Encoding** - Convert to sparse tensor format
4. **Stage 4: Entropy Coding** - Huffman/Arithmetic encoding
5. **Stage 5: Metadata Compression** - Final metadata optimization

## PERFORMANCE ANALYSIS

**Current Performance:**
- 100x100 tensors (10K elements): ~13 seconds ✅
- 500x500 tensors (250K elements): 10+ minutes (timeout) ❌
- Expected: Should scale roughly linearly, targeting <5 seconds for 500x500

**Root Cause:** The suffix array construction in `SlidingWindowPatternDetector` has algorithmic complexity issues that create quadratic behavior for larger inputs.

## COMPLETE CODE CONTEXT

### Main Pattern Detector Implementation

```python
# FILE: /independent_core/compression_systems/padic/sliding_window_pattern_detector.py

"""
Fast Suffix Array Pattern Detector for P-adic Digit Compression
O(n log n) Pattern Detection with Overflow-Free Rolling Hash
PRODUCTION READY - NO PLACEHOLDERS
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Union, Any
from dataclasses import dataclass, field
import math
from collections import defaultdict
import heapq


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


class SuffixArrayBuilder:
    """Fast suffix array construction using SA-IS algorithm"""
    
    @staticmethod
    def build_suffix_array(data: np.ndarray) -> np.ndarray:
        """
        Build suffix array in O(n log n) time using efficient radix-based sorting.
        Optimized for performance without O(n²) memory allocations.
        """
        n = len(data)
        if n == 0:
            return np.array([], dtype=np.int32)
        if n == 1:
            return np.array([0], dtype=np.int32)
        
        # Use radix-based suffix array construction for better performance
        # This avoids the O(n²) memory allocation issue
        return SuffixArrayBuilder._build_suffix_array_dc3(data)
    
    @staticmethod
    def _build_suffix_array_dc3(data: np.ndarray) -> np.ndarray:
        """
        DC3/Skew algorithm for suffix array construction in O(n) average time.
        Falls back to counting sort approach for better performance.
        """
        n = len(data)
        if n <= 2:
            if n == 1:
                return np.array([0], dtype=np.int32)
            elif n == 2:
                if data[0] <= data[1]:
                    return np.array([0, 1], dtype=np.int32)
                else:
                    return np.array([1, 0], dtype=np.int32)
        
        # For medium sizes, use efficient counting sort approach
        return SuffixArrayBuilder._build_suffix_array_counting(data)
    
    @staticmethod
    def _build_suffix_array_counting(data: np.ndarray) -> np.ndarray:
        """
        Build suffix array using optimized radix sort approach.
        O(n log n) time with O(n) space, highly optimized for performance.
        """
        n = len(data)
        if n == 0:
            return np.array([], dtype=np.int32)
        
        # Use numpy-optimized approach for better performance
        suffixes = np.arange(n, dtype=np.int32)
        
        # For small arrays, use direct comparison
        if n <= 1000:
            return SuffixArrayBuilder._build_suffix_array_small(data, suffixes)
        
        # For larger arrays, use optimized radix approach
        return SuffixArrayBuilder._build_suffix_array_radix(data, suffixes)
    
    @staticmethod
    def _build_suffix_array_small(data: np.ndarray, suffixes: np.ndarray) -> np.ndarray:
        """Optimized suffix array for small data using vectorized operations"""
        n = len(data)
        
        # Create comparison matrix for first few characters
        cmp_len = min(32, n)
        
        # Vectorized comparison using numpy broadcasting
        suffix_data = np.zeros((n, cmp_len), dtype=np.uint32)
        for i in range(n):
            end_idx = min(i + cmp_len, n)
            suffix_data[i, :end_idx-i] = data[i:end_idx]
            # Pad with zeros (which will sort first)
        
        # Use lexicographical sorting on the matrix
        # Convert to tuple for numpy lexsort
        sort_keys = [suffix_data[:, i] for i in range(cmp_len-1, -1, -1)]
        sorted_indices = np.lexsort(sort_keys)
        
        return sorted_indices.astype(np.int32)
    
    @staticmethod 
    def _build_suffix_array_radix(data: np.ndarray, suffixes: np.ndarray) -> np.ndarray:
        """Optimized radix-based suffix array for larger data"""
        n = len(data)
        
        # Use doubling technique with radix sort
        k = 1
        ranks = data.astype(np.int32)
        
        while k < n:
            # Create key pairs (rank[i], rank[i+k])
            first_keys = ranks[suffixes]
            second_keys = np.zeros(n, dtype=np.int32)
            
            # Handle second keys safely
            mask = suffixes + k < n
            second_keys[mask] = ranks[suffixes[mask] + k]
            
            # Sort by (first_key, second_key) pairs using numpy
            sort_indices = np.lexsort((second_keys, first_keys))
            suffixes = suffixes[sort_indices]
            
            # Update ranks
            new_ranks = np.zeros(n, dtype=np.int32)
            new_ranks[suffixes[0]] = 0
            
            for i in range(1, n):
                prev_suffix = suffixes[i-1]
                curr_suffix = suffixes[i]
                
                if (first_keys[sort_indices[i]] == first_keys[sort_indices[i-1]] and 
                    second_keys[sort_indices[i]] == second_keys[sort_indices[i-1]]):
                    new_ranks[curr_suffix] = new_ranks[prev_suffix]
                else:
                    new_ranks[curr_suffix] = new_ranks[prev_suffix] + 1
            
            ranks = new_ranks
            k *= 2
            
            # Early termination if all suffixes are distinct
            if ranks[suffixes[-1]] == n - 1:
                break
        
        return suffixes
    
    @staticmethod
    def build_lcp_array(data: np.ndarray, suffix_array: np.ndarray) -> np.ndarray:
        """Build Longest Common Prefix array in O(n) time using optimized Kasai's algorithm"""
        n = len(data)
        if n == 0:
            return np.array([], dtype=np.int32)
        if n == 1:
            return np.array([0], dtype=np.int32)
            
        lcp = np.zeros(n, dtype=np.int32)
        rank = np.zeros(n, dtype=np.int32)
        
        # Build rank array (inverse of suffix array) - vectorized
        rank[suffix_array] = np.arange(n)
        
        # Optimized Kasai's algorithm with bounds checking
        k = 0
        max_lcp = min(64, n)  # Limit LCP computation for efficiency
        
        for i in range(n):
            if rank[i] == n - 1:
                k = 0
                continue
                
            j = suffix_array[rank[i] + 1]
            
            # Optimized common prefix computation with early termination
            remaining_i = n - i
            remaining_j = n - j
            max_possible_lcp = min(remaining_i, remaining_j, max_lcp)
            
            # Use vectorized comparison for efficiency
            if max_possible_lcp > 8:
                # For longer potential LCPs, use numpy vectorized comparison
                end_pos = min(i + max_possible_lcp, n)
                end_pos_j = min(j + max_possible_lcp, n)
                
                # Get the actual comparison length
                cmp_len = min(end_pos - i, end_pos_j - j)
                
                if cmp_len > 0:
                    # Vectorized equality check
                    matches = data[i:i+cmp_len] == data[j:j+cmp_len]
                    if matches.size > 0:
                        # Find first mismatch position  
                        if matches.all():
                            mismatch_pos = len(matches)
                        else:
                            mismatch_pos = np.argmax(~matches)
                        k = max(k, mismatch_pos)
                    else:
                        k = max(k, 0)
            else:
                # For short LCPs, use direct comparison
                while (k < max_possible_lcp and 
                       i + k < n and j + k < n and 
                       data[i + k] == data[j + k]):
                    k += 1
                    
            lcp[rank[i]] = k
            
            if k > 0:
                k -= 1
                
        return lcp


class SlidingWindowPatternDetector(nn.Module):
    """
    Fast pattern detector using suffix arrays and rolling hash.
    O(n log n) time complexity with overflow-free operations.
    """
    
    def __init__(
        self,
        min_pattern_length: int = 4,
        max_pattern_length: int = 32,
        min_frequency: int = 3,
        hash_prime: int = 257,
        device: str = 'cpu',
        enable_compile: bool = True,
        max_patterns: int = 1000,  # Limit patterns for memory efficiency
        use_suffix_array: bool = True  # Use fast suffix array algorithm
    ):
        """Initialize fast pattern detector."""
        super().__init__()
        
        # Validate parameters
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
        
        # Initialize suffix array builder
        self.suffix_builder = SuffixArrayBuilder()
    
    def _find_patterns_suffix_array(self, data: np.ndarray) -> Dict[int, PatternMatch]:
        """Find patterns using suffix array algorithm - O(n log n) with optimizations"""
        n = len(data)
        if n < self.min_pattern_length:
            return {}
        
        # Build suffix array and LCP array
        suffix_array = self.suffix_builder.build_suffix_array(data)  # ⚠️ BOTTLENECK
        lcp_array = self.suffix_builder.build_lcp_array(data, suffix_array)
        
        patterns_found = {}
        pattern_id = 0
        
        # Use interval trees for efficient overlap checking
        covered_intervals = []  # List of (start, end) intervals
        
        # Use a priority queue to process patterns by value (length * frequency)
        pattern_queue = []
        
        # Optimized pattern extraction with vectorized operations
        i = 0
        while i < n - 1:
            lcp_val = lcp_array[i]
            if lcp_val >= self.min_pattern_length:
                # Found potential pattern
                pattern_length = min(lcp_val, self.max_pattern_length)
                
                # Efficiently collect all positions with this common prefix
                positions = [suffix_array[i]]
                j = i + 1
                
                # Vectorized comparison for collecting positions
                while j < n:
                    if j - 1 < len(lcp_array) and lcp_array[j - 1] >= pattern_length:
                        positions.append(suffix_array[j])
                        j += 1
                    else:
                        break
                
                if len(positions) >= self.min_frequency:
                    # Extract pattern using numpy for efficiency
                    pattern_start = positions[0]
                    pattern_data = data[pattern_start:pattern_start + pattern_length]
                    
                    # Calculate pattern value for prioritization
                    value = pattern_length * len(positions)
                    
                    # Add to priority queue (negative value for max heap)
                    heapq.heappush(pattern_queue, 
                                   (-value, pattern_length, pattern_data.tobytes(), positions))
                
                i = j
            else:
                i += 1
        
        # Process patterns in order of value with optimized overlap checking
        while pattern_queue and len(patterns_found) < self.max_patterns:
            _, pattern_length, pattern_data, positions = heapq.heappop(pattern_queue)
            
            # Optimized overlap checking using numpy operations
            valid_positions = []
            
            for pos in positions:
                # Check if position overlaps with any covered interval
                overlaps = False
                for start, end in covered_intervals:
                    if not (pos + pattern_length <= start or pos >= end):
                        overlaps = True
                        break
                
                if not overlaps:
                    valid_positions.append(pos)
            
            if len(valid_positions) >= self.min_frequency:
                # Create pattern match
                pattern_match = PatternMatch(
                    pattern=pattern_data,
                    positions=sorted(valid_positions),
                    frequency=len(valid_positions),
                    hash_value=hash(pattern_data),
                    length=pattern_length
                )
                
                patterns_found[pattern_id] = pattern_match
                pattern_id += 1
                
                # Mark positions as covered
                for pos in valid_positions:
                    covered_intervals.append((pos, pos + pattern_length))
                
                # Optimize covered_intervals to prevent quadratic growth
                if len(covered_intervals) > 100:
                    # Merge overlapping intervals
                    covered_intervals.sort()
                    merged = []
                    for start, end in covered_intervals:
                        if merged and start <= merged[-1][1]:
                            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                        else:
                            merged.append((start, end))
                    covered_intervals = merged
        
        return patterns_found

    def find_patterns(
        self,
        data: Union[torch.Tensor, np.ndarray, bytes],
        batch_process: bool = True
    ) -> PatternDetectionResult:
        """
        Find repeated patterns in input data.
        Uses suffix array for O(n log n) complexity.
        """
        # Convert input to numpy array for efficient processing
        if isinstance(data, bytes):
            data = np.frombuffer(data, dtype=np.uint8)
        elif isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
            # Handle potential out-of-range values for uint8
            if data_np.dtype in [np.int32, np.int64, np.float32, np.float64]:
                # Clamp values to uint8 range and convert
                data = np.clip(data_np, 0, 255).astype(np.uint8)
            else:
                data = data_np.astype(np.uint8)
        elif isinstance(data, np.ndarray):
            # Handle potential out-of-range values for uint8
            if data.dtype in [np.int32, np.int64, np.float32, np.float64]:
                # Clamp values to uint8 range and convert
                data = np.clip(data, 0, 255).astype(np.uint8)
            else:
                data = data.astype(np.uint8)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        if data.ndim != 1:
            data = data.flatten()
        
        n = len(data)
        
        # Early termination for data too small to have meaningful patterns
        if n < self.min_pattern_length * self.min_frequency:
            return PatternDetectionResult(
                patterns={},
                pattern_mask=torch.zeros(n, dtype=torch.bool, device=self.device),
                pattern_indices=torch.full((n,), -1, dtype=torch.int32, device=self.device),
                compression_potential=0.0,
                total_patterns_found=0,
                bytes_replaced=0,
                original_size=n
            )
        
        # Choose algorithm based on configuration and data size for optimal performance
        if self.use_suffix_array and n > 1000:
            # For large data, suffix array is more efficient
            patterns_found = self._find_patterns_suffix_array(data)  # ⚠️ BOTTLENECK CALL
        else:
            # For small to medium data, rolling hash can be faster
            patterns_found = self._find_patterns_rolling_hash(data)
        
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
```

### Integration Point in Main Compressor

```python
# FILE: /independent_core/compression_systems/padic/padic_compressor.py

class CompressionStage(Enum):
    """Compression pipeline stages"""
    ADAPTIVE_PRECISION = "adaptive_precision"
    PATTERN_DETECTION = "pattern_detection"      # ⚠️ BOTTLENECK STAGE
    SPARSE_ENCODING = "sparse_encoding"
    ENTROPY_CODING = "entropy_coding"
    METADATA_COMPRESSION = "metadata_compression"

def compress(self, tensor: torch.Tensor, 
            importance_scores: Optional[torch.Tensor] = None) -> CompressionResult:
    """Compress tensor through sequential pipeline"""
    start_time = time.perf_counter()
    stage_metrics = {}
    
    # ... Stage 1: Adaptive Precision (working fine) ...
    
    # Stage 2: Pattern Detection ⚠️ BOTTLENECK HERE
    stage_start = time.perf_counter()
    logger.info("Stage 2: Pattern Detection")
    
    flat_digits = digit_tensor.flatten()                                    # 250K elements for 500x500
    pattern_result = self.pattern_detector.find_patterns(flat_digits)      # ⚠️ HANGS HERE
    
    # Encode with patterns
    encoded_data, pattern_dict, pattern_lengths = self.pattern_detector.encode_with_patterns(
        flat_digits,
        pattern_result
    )
    
    stage_metrics['pattern_detection'] = {
        'time': time.perf_counter() - stage_start,
        'patterns_found': pattern_result.total_patterns_found,
        'bytes_replaced': pattern_result.bytes_replaced,
        'compression_potential': pattern_result.compression_potential
    }
    
    # ... Remaining stages ...
```

## ALGORITHMIC ANALYSIS

### Current Algorithm Performance Issues

**Problem 1: Suffix Array Construction Complexity**
- `_build_suffix_array_radix()` uses doubling technique with nested loops
- The inner rank update loop (lines 156-164) processes all n elements for each doubling step
- For 250K elements: k doubles ~18 times (log₂(250K)), each iteration processes 250K elements
- Total operations: ~18 × 250K × 250K = 1.125 billion operations

**Problem 2: LCP Array Bottleneck**
- `build_lcp_array()` has nested vectorized comparisons (lines 207-227)
- For each of n positions, compares up to 64 characters
- Vectorized operations still create temporary arrays of size up to 64 × 250K = 16M elements
- Memory allocation/deallocation overhead compounds the issue

**Problem 3: Pattern Extraction Inefficiencies**
- Pattern queue processing (lines 389-438) has quadratic overlap checking
- For each pattern, checks overlap against all covered intervals
- Interval merging happens every 100 intervals but still processes quadratically

### Expected vs Actual Complexity
- **Expected:** O(n log n) for suffix array + O(n) for LCP = O(n log n) overall
- **Actual:** O(n² log n) due to implementation bottlenecks in rank updates and memory allocations

## PERFORMANCE TARGETS

| Tensor Size | Current | Target | Improvement |
|-------------|---------|---------|-------------|
| 100x100 (10K) | ~13s | ~13s | ✅ Acceptable |
| 200x200 (40K) | ~60s est | ~20s | 3x faster |
| 500x500 (250K) | 600s+ | <5s | >120x faster |

## REQUIRED SOLUTION APPROACH

### 1. Optimize Suffix Array Construction
- Replace the doubling technique with a more efficient radix sort implementation
- Use incremental sorting instead of full re-sorting at each step
- Implement early termination when ranks stabilize
- Add memory-efficient batching for large arrays

### 2. Streamline LCP Array Construction  
- Reduce vectorized comparison sizes and use iterative approach
- Implement sliding window comparisons instead of full array allocations
- Add intelligent bounds checking to avoid unnecessary comparisons
- Use numba/torch JIT compilation for performance-critical loops

### 3. Optimize Pattern Processing
- Replace linear overlap checking with interval tree or segment tree
- Batch pattern validation operations
- Implement parallel processing for pattern extraction
- Add early termination heuristics based on compression potential

### 4. Add Intelligent Size-Based Algorithm Selection
- Use rolling hash for small inputs (<50K elements)
- Use optimized suffix array for medium inputs (50K-500K elements)  
- Implement chunking/streaming for very large inputs (>500K elements)
- Add entropy-based early termination for random data

## TESTING REQUIREMENTS

The solution must pass these performance benchmarks:

```python
# From test_comprehensive_saraphis_fixes.py
test_sizes = [
    (50, 50),      # 2,500 elements - serial processing
    (100, 100),    # 10,000 elements - serial processing  
    (200, 200),    # 40,000 elements - batched processing
    (500, 500),    # 250,000 elements - parallel batched processing ⚠️ CRITICAL
]

# Performance targets:
# 500x500 tensor should process in under 5 seconds (was 10+ minutes before)
large_tensor_result = benchmark_results[-1]
self.assertLess(large_tensor_result['avg_time'], 5.0,
               f"Large tensor processing too slow: {large_tensor_result['avg_time']:.2f}s")
```

## DELIVERABLE REQUIREMENTS

Please provide a complete optimized implementation that:

1. **Maintains API Compatibility** - All existing method signatures and return types
2. **Fixes Performance Bottlenecks** - Achieves <5s for 500x500 tensors
3. **Preserves Correctness** - Pattern detection results remain accurate
4. **Includes Comprehensive Error Handling** - Robust edge case handling
5. **Has Detailed Performance Logging** - Track timing of each optimization
6. **Provides Algorithmic Documentation** - Explain the optimization approaches used

The solution should focus on the `SuffixArrayBuilder` class methods and the `_find_patterns_suffix_array` method in `SlidingWindowPatternDetector`, as these contain the primary performance bottlenecks identified in the analysis.

Please provide the complete optimized code with detailed comments explaining the algorithmic improvements and expected performance gains.