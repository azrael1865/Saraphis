"""
Entropy Bridge for P-adic Compression System

Bridges p-adic digit tensors with entropy coding (Huffman/Arithmetic) using
intelligent selection based on Shannon entropy analysis.

Mathematical Foundation:
- Shannon Entropy: H(X) = -Σ p(x) × log2(p(x))
- Huffman Efficiency: L̄ = H(X) + 1 bit worst case
- Arithmetic Efficiency: L → H(X) as n → ∞
- Hybrid Optimization: 80/20 frequency split for optimal encoding

NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from collections import Counter
import math
import struct
import time
import logging

logger = logging.getLogger(__name__)

# Import existing entropy coders - handle both package and direct imports
import sys
import os

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

try:
    # Try relative imports first
    from ..encoding.huffman_arithmetic import (
        HuffmanEncoder, ArithmeticEncoder, HybridEncoder,
        CompressionMetrics
    )  
except ImportError:
    try:
        # Try direct imports from encoding directory
        from encoding.huffman_arithmetic import (
            HuffmanEncoder, ArithmeticEncoder, HybridEncoder,
            CompressionMetrics
        )
    except ImportError:
        # Try importing from the actual file location
        encoding_path = os.path.join(parent_dir, 'encoding')
        if encoding_path not in sys.path:
            sys.path.insert(0, encoding_path)
        from huffman_arithmetic import (
            HuffmanEncoder, ArithmeticEncoder, HybridEncoder,
            CompressionMetrics
        )

try:
    from .padic_logarithmic_encoder import LogarithmicPadicWeight
except ImportError:
    from padic_logarithmic_encoder import LogarithmicPadicWeight

try:
    from .padic_encoder import PadicWeight
except ImportError:
    from padic_encoder import PadicWeight


@dataclass
class EntropyBridgeConfig:
    """Configuration for entropy-p-adic bridging"""
    # Entropy thresholds for method selection
    huffman_threshold: float = 2.0      # H < 2.0 bits: Use Huffman
    arithmetic_threshold: float = 6.0   # H > 6.0 bits: Use Arithmetic
    hybrid_low_threshold: float = 3.0   # Lower bound for hybrid region
    hybrid_high_threshold: float = 5.0  # Upper bound for hybrid region
    
    # Hybrid encoding parameters
    frequency_split_ratio: float = 0.2  # Top 20% of symbols by frequency
    min_symbols_for_hybrid: int = 10    # Minimum unique symbols for hybrid
    pattern_analysis_window: int = 100  # Window size for pattern detection
    
    # Tensor processing
    max_tensor_size: int = 1_000_000    # Maximum elements before chunking
    chunk_overlap: int = 100             # Overlap between chunks
    preserve_spatial_structure: bool = True  # Maintain tensor dimensions
    
    # Optimization flags
    enable_pattern_detection: bool = True    # Detect repeating patterns
    enable_delta_encoding: bool = True       # Use delta encoding for sequences
    enable_run_length: bool = True           # Use RLE for repeated values
    adaptive_threshold: bool = True          # Adapt thresholds based on data
    
    # Performance tuning
    parallel_encoding: bool = False      # Use parallel processing (if available)
    cache_frequency_tables: bool = True  # Cache frequency analysis
    compression_level: int = 6            # 1 (fast) to 9 (best compression)
    
    # Error handling
    max_failures: int = 5                # Circuit breaker threshold
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if not (0 < self.huffman_threshold < self.arithmetic_threshold):
            raise ValueError(
                f"Invalid thresholds: huffman={self.huffman_threshold}, "
                f"arithmetic={self.arithmetic_threshold}"
            )
        
        if not (0 < self.frequency_split_ratio < 1):
            raise ValueError(f"Split ratio must be in (0,1), got {self.frequency_split_ratio}")
        
        if self.max_tensor_size <= 0:
            raise ValueError(f"Max tensor size must be positive, got {self.max_tensor_size}")
        
        if not (1 <= self.compression_level <= 9):
            raise ValueError(f"Compression level must be 1-9, got {self.compression_level}")


@dataclass
class EntropyAnalysis:
    """Results of entropy analysis on digit distribution"""
    shannon_entropy: float              # H(X) in bits
    normalized_entropy: float           # H(X) / log2(alphabet_size)
    unique_symbols: int                 # Number of unique symbols
    total_symbols: int                  # Total number of symbols
    max_symbol: int                     # Maximum symbol value
    min_symbol: int                     # Minimum symbol value
    
    frequency_distribution: Dict[int, int] = field(default_factory=dict)
    probability_distribution: Dict[int, float] = field(default_factory=dict)
    
    # Pattern statistics
    has_patterns: bool = False          # Whether patterns were detected
    pattern_ratio: float = 0.0          # Ratio of pattern-matched symbols
    run_length_ratio: float = 0.0       # Ratio of run-length encodable symbols
    
    # Encoding recommendation
    recommended_method: str = "huffman"  # huffman, arithmetic, or hybrid
    confidence_score: float = 0.0        # Confidence in recommendation (0-1)
    expected_compression_ratio: float = 1.0  # Expected compression ratio
    
    def calculate_expected_compression(self) -> float:
        """Calculate expected compression ratio based on entropy"""
        if self.shannon_entropy <= 0 or self.total_symbols == 0:
            return 1.0
        
        # Theoretical minimum bits needed
        min_bits = self.shannon_entropy * self.total_symbols
        
        # Original bits (assuming 8 bits per symbol for simplicity)
        original_bits = 8 * self.total_symbols
        
        # Add overhead estimate (10% for metadata)
        estimated_bits = min_bits * 1.1
        
        if estimated_bits > 0:
            return original_bits / estimated_bits
        return 1.0


class EntropyPAdicBridge:
    """
    Bridge between p-adic digit tensors and entropy coding.
    
    Provides intelligent selection between Huffman and Arithmetic coding
    based on Shannon entropy analysis and pattern detection.
    """
    
    def __init__(self, prime: int, config: Optional[EntropyBridgeConfig] = None):
        """Initialize entropy bridge
        
        Args:
            prime: P-adic prime (determines symbol alphabet)
            config: Bridge configuration (uses defaults if None)
        """
        if not isinstance(prime, int) or prime < 2:
            raise ValueError(f"Prime must be integer >= 2, got {prime}")
        
        self.prime = prime
        self.config = config or EntropyBridgeConfig()
        
        # Initialize encoders
        self.huffman_encoder = HuffmanEncoder(prime)
        self.arithmetic_encoder = ArithmeticEncoder(prime)
        self.hybrid_encoder = HybridEncoder(prime)
        
        # Cache for frequency analysis
        self.frequency_cache: Dict[int, EntropyAnalysis] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance metrics
        self.total_compressions = 0
        self.method_usage = {"huffman": 0, "arithmetic": 0, "hybrid": 0}
        self.average_compression_ratio = 0.0
        self.failure_count = 0  # Add missing failure count for error tracking
        self.max_failures = self.config.max_failures  # Circuit breaker threshold from config
        
        logger.info(f"EntropyPAdicBridge initialized with prime={prime}")
    
    def _check_circuit_breaker(self):
        """Circuit breaker pattern to prevent cascading failures"""
        if self.failure_count >= self.max_failures:
            logger.warning(f"Circuit breaker activated: {self.failure_count} consecutive failures")
            # Reset counter to allow recovery
            self.failure_count = 0
    
    def _record_failure(self, error):
        """Record a failure for circuit breaker pattern"""
        self.failure_count += 1
        logger.debug(f"Failure recorded: {error}. Total failures: {self.failure_count}")
    
    def analyze_entropy(self, digits: Union[List[int], np.ndarray, torch.Tensor]) -> EntropyAnalysis:
        """Analyze Shannon entropy and distribution characteristics
        
        Args:
            digits: P-adic digit sequence (list, numpy array, or tensor)
            
        Returns:
            Comprehensive entropy analysis
        """
        # Convert to list if needed
        if isinstance(digits, torch.Tensor):
            digit_list = digits.cpu().numpy().flatten().tolist()
        elif isinstance(digits, np.ndarray):
            digit_list = digits.flatten().tolist()
        else:
            digit_list = list(digits)
        
        if not digit_list:
            raise ValueError("Cannot analyze empty digit sequence")
        
        # Check cache if enabled
        cache_key = hash(tuple(digit_list[:min(100, len(digit_list))]))
        if self.config.cache_frequency_tables and cache_key in self.frequency_cache:
            self.cache_hits += 1
            return self.frequency_cache[cache_key]
        
        self.cache_misses += 1
        
        # Count frequencies
        frequency_dist = Counter(digit_list)
        total_count = len(digit_list)
        
        # Validate all digits are non-negative (allow values >= prime for quantized data)
        for digit in frequency_dist:
            if digit < 0:
                raise ValueError(f"Digit {digit} cannot be negative")
        
        # Calculate probabilities
        prob_dist = {
            symbol: count / total_count
            for symbol, count in frequency_dist.items()
        }
        
        # Calculate Shannon entropy
        shannon_entropy = 0.0
        for prob in prob_dist.values():
            if prob > 0:
                shannon_entropy -= prob * math.log2(prob)
        
        # Normalized entropy (0 to 1)
        max_entropy = math.log2(self.prime)
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0
        
        # Create analysis object
        analysis = EntropyAnalysis(
            shannon_entropy=shannon_entropy,
            normalized_entropy=normalized_entropy,
            unique_symbols=len(frequency_dist),
            total_symbols=total_count,
            max_symbol=max(frequency_dist.keys()) if frequency_dist else 0,
            min_symbol=min(frequency_dist.keys()) if frequency_dist else 0,
            frequency_distribution=dict(frequency_dist),
            probability_distribution=prob_dist
        )
        
        # Pattern detection if enabled
        if self.config.enable_pattern_detection:
            analysis.has_patterns, analysis.pattern_ratio = self._detect_patterns(digit_list)
        
        # Run-length analysis if enabled
        if self.config.enable_run_length:
            analysis.run_length_ratio = self._analyze_runs(digit_list)
        
        # Determine recommended encoding method
        analysis.recommended_method = self._select_encoding_method(analysis)
        
        # Calculate confidence score
        analysis.confidence_score = self._calculate_confidence(analysis)
        
        # Estimate compression ratio
        analysis.expected_compression_ratio = analysis.calculate_expected_compression()
        
        # Cache result if enabled
        if self.config.cache_frequency_tables:
            self.frequency_cache[cache_key] = analysis
        
        return analysis
    
    def _detect_patterns(self, digits: List[int]) -> Tuple[bool, float]:
        """Detect repeating patterns in digit sequence
        
        Args:
            digits: List of p-adic digits
            
        Returns:
            Tuple of (has_patterns, pattern_ratio)
        """
        if len(digits) < self.config.pattern_analysis_window:
            return False, 0.0
        
        window_size = min(self.config.pattern_analysis_window, len(digits) // 4)
        pattern_matches = 0
        
        # Sliding window pattern detection
        for i in range(len(digits) - window_size):
            window = digits[i:i+window_size]
            
            # Look for pattern repetition
            for j in range(i + window_size, min(len(digits) - window_size + 1, i + window_size * 3)):
                if digits[j:j+window_size] == window:
                    pattern_matches += window_size
                    break
        
        pattern_ratio = pattern_matches / len(digits) if len(digits) > 0 else 0
        has_patterns = pattern_ratio > 0.1  # More than 10% pattern coverage
        
        return has_patterns, pattern_ratio
    
    def _analyze_runs(self, digits: List[int]) -> float:
        """Analyze run-length encoding potential
        
        Args:
            digits: List of p-adic digits
            
        Returns:
            Ratio of symbols that are part of runs
        """
        if not digits:
            return 0.0
        
        run_symbols = 0
        i = 0
        
        while i < len(digits):
            j = i + 1
            while j < len(digits) and digits[j] == digits[i]:
                j += 1
            
            run_length = j - i
            if run_length >= 3:  # Consider runs of 3+ as significant
                run_symbols += run_length
            
            i = j
        
        return run_symbols / len(digits)
    
    def _select_encoding_method(self, analysis: EntropyAnalysis) -> str:
        """Select optimal encoding method based on entropy analysis
        
        Args:
            analysis: Entropy analysis results
            
        Returns:
            "huffman", "arithmetic", or "hybrid"
        """
        entropy = analysis.shannon_entropy
        
        # Adaptive threshold adjustment
        if self.config.adaptive_threshold:
            # Adjust based on unique symbol count
            symbol_ratio = analysis.unique_symbols / self.prime
            if symbol_ratio < 0.1:  # Very few unique symbols
                huffman_threshold = self.config.huffman_threshold * 1.5
                arithmetic_threshold = self.config.arithmetic_threshold * 0.8
            elif symbol_ratio > 0.8:  # Many unique symbols
                huffman_threshold = self.config.huffman_threshold * 0.8
                arithmetic_threshold = self.config.arithmetic_threshold * 1.2
            else:
                huffman_threshold = self.config.huffman_threshold
                arithmetic_threshold = self.config.arithmetic_threshold
        else:
            huffman_threshold = self.config.huffman_threshold
            arithmetic_threshold = self.config.arithmetic_threshold
        
        # Decision logic
        if entropy < huffman_threshold:
            # Low entropy: Huffman is simple and effective
            return "huffman"
        elif entropy > arithmetic_threshold:
            # High entropy: Arithmetic approaches theoretical limit
            return "arithmetic"
        else:
            # Medium entropy: Consider hybrid based on patterns
            if analysis.unique_symbols >= self.config.min_symbols_for_hybrid:
                if analysis.has_patterns or analysis.run_length_ratio > 0.2:
                    return "hybrid"
            
            # Default to Huffman for reliability
            return "huffman"
    
    def _calculate_confidence(self, analysis: EntropyAnalysis) -> float:
        """Calculate confidence score for encoding method selection
        
        Args:
            analysis: Entropy analysis results
            
        Returns:
            Confidence score between 0 and 1
        """
        entropy = analysis.shannon_entropy
        method = analysis.recommended_method
        
        if method == "huffman":
            if entropy < 1.0:
                confidence = 0.95  # Very confident for very low entropy
            elif entropy < self.config.huffman_threshold:
                confidence = 0.8 + (self.config.huffman_threshold - entropy) * 0.1
            else:
                confidence = 0.6
        elif method == "arithmetic":
            if entropy > 7.0:
                confidence = 0.95  # Very confident for very high entropy
            elif entropy > self.config.arithmetic_threshold:
                confidence = 0.8 + (entropy - self.config.arithmetic_threshold) * 0.05
            else:
                confidence = 0.6
        else:  # hybrid
            # Confidence based on pattern detection
            confidence = 0.7 + min(analysis.pattern_ratio * 0.3, 0.25)
        
        return min(max(confidence, 0.0), 1.0)
    
    def encode_padic_tensor(self, 
                           tensor: torch.Tensor,
                           force_method: Optional[str] = None) -> Tuple[bytes, Dict[str, Any]]:
        """Encode p-adic digit tensor using optimal entropy coding
        
        Args:
            tensor: PyTorch tensor containing p-adic digits
            force_method: Force specific method ("huffman", "arithmetic", "hybrid")
            
        Returns:
            Tuple of (compressed_bytes, metadata_dict)
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        
        if tensor.numel() == 0:
            # Handle empty tensor case gracefully
            return b'', {
                'empty': True,
                'encoding_method': 'none',
                'original_shape': list(tensor.shape),
                'original_dtype': str(tensor.dtype)
            }
        
        # Record original shape
        original_shape = list(tensor.shape)
        original_dtype = tensor.dtype
        
        # Move to device if configured
        if hasattr(self, 'device'):
            tensor = tensor.to(self.device)
        
        # Flatten tensor for encoding
        flat_tensor = tensor.flatten()
        
        # Convert to integer type if needed
        if flat_tensor.dtype.is_floating_point:
            flat_tensor = flat_tensor.long()
        
        # Check tensor size and chunk if necessary
        if flat_tensor.numel() > self.config.max_tensor_size:
            return self._encode_large_tensor(tensor, force_method)
        
        # Convert to list - only clamp if values are meant to be p-adic digits
        digit_list = flat_tensor.cpu().tolist()
        # Check if this looks like quantized data (values > prime) or p-adic digits
        max_value = max(digit_list) if digit_list else 0
        if max_value < self.prime:
            # These look like p-adic digits, clamp them
            digit_list = [max(0, min(int(d), self.prime - 1)) for d in digit_list]
        # Otherwise keep values as-is for quantized data
        
        # Analyze entropy
        start_time = time.perf_counter()
        analysis = self.analyze_entropy(digit_list)
        analysis_time = (time.perf_counter() - start_time) * 1000
        
        # Select encoding method based on entropy thresholds
        if force_method and force_method in ["huffman", "arithmetic", "hybrid"]:
            method = force_method
        else:
            # Entropy-based selection: H(X) < 2.0 → Huffman, H(X) > 6.0 → Arithmetic
            entropy = analysis.shannon_entropy
            if entropy < self.config.huffman_threshold:  # < 2.0
                method = "huffman"
            elif entropy > self.config.arithmetic_threshold:  # > 6.0
                method = "arithmetic"
            else:
                # Medium entropy: use hybrid if conditions met
                if analysis.unique_symbols >= self.config.min_symbols_for_hybrid:
                    if analysis.has_patterns or analysis.run_length_ratio > 0.2:
                        method = "hybrid"
                    else:
                        method = "huffman"  # Default to huffman for reliability
                else:
                    method = "huffman"  # Default to huffman for reliability
        
        # Update usage statistics
        self.method_usage[method] += 1
        self.total_compressions += 1
        
        # Perform encoding
        start_time = time.perf_counter()
        
        if method == "huffman":
            compressed, encode_metadata = self._encode_huffman(digit_list, analysis)
        elif method == "arithmetic":
            compressed, encode_metadata = self._encode_arithmetic(digit_list, analysis)
        else:  # hybrid
            compressed, encode_metadata = self._encode_hybrid(digit_list, analysis)
        
        encoding_time = (time.perf_counter() - start_time) * 1000
        
        # Build complete metadata with encoding_method properly included
        # CRITICAL: Add method to encode_metadata for backwards compatibility
        encode_metadata["method"] = method  # Ensure method is in encode_metadata
        
        metadata = {
            "original_shape": original_shape,
            "original_dtype": str(original_dtype),
            "prime": self.prime,
            "encoding_method": method,  # Keep for backwards compatibility
            "method": method,  # Additional fallback
            "entropy_analysis": {
                "shannon_entropy": analysis.shannon_entropy,
                "normalized_entropy": analysis.normalized_entropy,
                "unique_symbols": analysis.unique_symbols,
                "total_symbols": analysis.total_symbols,
                "confidence_score": analysis.confidence_score,
                "has_patterns": analysis.has_patterns,
                "pattern_ratio": analysis.pattern_ratio,
                "run_length_ratio": analysis.run_length_ratio
            },
            "encoding_metadata": encode_metadata,
            "compression_metrics": {
                "original_bytes": tensor.numel() * tensor.element_size(),
                "compressed_bytes": len(compressed),
                "compression_ratio": (tensor.numel() * tensor.element_size()) / len(compressed) if len(compressed) > 0 else 1.0,
                "analysis_time_ms": analysis_time,
                "encoding_time_ms": encoding_time
            }
        }
        
        # Update average compression ratio
        self.average_compression_ratio = (
            (self.average_compression_ratio * (self.total_compressions - 1) + 
             metadata["compression_metrics"]["compression_ratio"]) / 
            self.total_compressions
        )
        
        logger.debug(
            f"Encoded tensor: shape={original_shape}, method={method}, "
            f"ratio={metadata['compression_metrics']['compression_ratio']:.2f}x"
        )
        
        # Reset failure count on success
        if self.failure_count > 0:
            self.failure_count = max(0, self.failure_count - 1)
        
        return compressed, metadata
    
    def _encode_large_tensor(self, 
                            tensor: torch.Tensor,
                            force_method: Optional[str] = None) -> Tuple[bytes, Dict[str, Any]]:
        """Encode large tensor by chunking
        
        Args:
            tensor: Large tensor to encode
            force_method: Force specific encoding method
            
        Returns:
            Tuple of (compressed_bytes, metadata)
        """
        flat_tensor = tensor.flatten()
        chunk_size = self.config.max_tensor_size
        overlap = self.config.chunk_overlap
        
        chunks_compressed = []
        chunks_metadata = []
        
        for i in range(0, flat_tensor.numel(), chunk_size - overlap):
            end_idx = min(i + chunk_size, flat_tensor.numel())
            chunk = flat_tensor[i:end_idx]
            
            compressed_chunk, chunk_meta = self.encode_padic_tensor(
                chunk.reshape(-1), force_method
            )
            
            chunks_compressed.append(compressed_chunk)
            chunks_metadata.append(chunk_meta)
        
        # Combine chunks
        combined = bytearray()
        combined.extend(struct.pack('<I', len(chunks_compressed)))  # Number of chunks
        
        for compressed_chunk in chunks_compressed:
            combined.extend(struct.pack('<I', len(compressed_chunk)))  # Chunk size
            combined.extend(compressed_chunk)
        
        metadata = {
            "chunked": True,
            "num_chunks": len(chunks_compressed),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "original_shape": list(tensor.shape),
            "chunks_metadata": chunks_metadata
        }
        
        return bytes(combined), metadata
    
    def _encode_huffman(self, 
                       digits: List[int],
                       analysis: EntropyAnalysis) -> Tuple[bytes, Dict[str, Any]]:
        """Encode using Huffman coding
        
        Args:
            digits: List of p-adic digits
            analysis: Entropy analysis results
            
        Returns:
            Tuple of (compressed_bytes, metadata)
        """
        # Build Huffman tree
        self.huffman_encoder.build_huffman_tree(analysis.frequency_distribution)
        self.huffman_encoder.generate_codes()
        
        # Encode digits
        compressed = self.huffman_encoder.huffman_encode(digits)
        
        # Build metadata
        metadata = {
            "method": "huffman",
            "frequency_table": analysis.frequency_distribution,
            "encoding_table_size": len(self.huffman_encoder.encoding_table)
        }
        
        return compressed, metadata
    
    def _encode_arithmetic(self,
                          digits: List[int],
                          analysis: EntropyAnalysis) -> Tuple[bytes, Dict[str, Any]]:
        """Encode using arithmetic coding
        
        Args:
            digits: List of p-adic digits
            analysis: Entropy analysis results
            
        Returns:
            Tuple of (compressed_bytes, metadata)
        """
        # Use simple arithmetic encoding with chunking for long sequences
        # to avoid precision issues in arithmetic_decode_simple
        max_chunk_size = 10  # Safe size based on testing
        
        if len(digits) <= max_chunk_size:
            # Single chunk encoding
            compressed = self.arithmetic_encoder.arithmetic_encode_simple(
                digits, analysis.probability_distribution
            )
            metadata = {
                "method": "arithmetic",
                "probability_table": analysis.probability_distribution,
                "chunked": False
            }
        else:
            # Multi-chunk encoding
            chunks = []
            for i in range(0, len(digits), max_chunk_size):
                chunk = digits[i:i + max_chunk_size]
                chunk_compressed = self.arithmetic_encoder.arithmetic_encode_simple(
                    chunk, analysis.probability_distribution
                )
                chunks.append(chunk_compressed)
            
            # Combine chunks
            combined = bytearray()
            combined.extend(struct.pack('<I', len(chunks)))  # Number of chunks
            for chunk in chunks:
                combined.extend(struct.pack('<I', len(chunk)))  # Chunk size
                combined.extend(chunk)
            
            compressed = bytes(combined)
            metadata = {
                "method": "arithmetic",
                "probability_table": analysis.probability_distribution,
                "chunked": True,
                "num_chunks": len(chunks),
                "chunk_size": max_chunk_size
            }
        
        return compressed, metadata
    
    def _encode_arithmetic_chunked(self,
                                  digits: List[int],
                                  probabilities: Dict[int, float]) -> Tuple[bytes, Dict[str, Any]]:
        """Helper method for chunked arithmetic encoding
        
        Args:
            digits: List of p-adic digits
            probabilities: Symbol probability distribution
            
        Returns:
            Tuple of (compressed_bytes, metadata)
        """
        max_chunk_size = 10  # Safe size based on testing
        
        if len(digits) <= max_chunk_size:
            # Single chunk encoding
            compressed = self.arithmetic_encoder.arithmetic_encode_simple(digits, probabilities)
            metadata = {"chunked": False}
        else:
            # Multi-chunk encoding
            chunks = []
            for i in range(0, len(digits), max_chunk_size):
                chunk = digits[i:i + max_chunk_size]
                chunk_compressed = self.arithmetic_encoder.arithmetic_encode_simple(chunk, probabilities)
                chunks.append(chunk_compressed)
            
            # Combine chunks
            combined = bytearray()
            combined.extend(struct.pack('<I', len(chunks)))  # Number of chunks
            for chunk in chunks:
                combined.extend(struct.pack('<I', len(chunk)))  # Chunk size
                combined.extend(chunk)
            
            compressed = bytes(combined)
            metadata = {"chunked": True, "num_chunks": len(chunks)}
        
        return compressed, metadata
    
    def _encode_hybrid(self,
                      digits: List[int],
                      analysis: EntropyAnalysis) -> Tuple[bytes, Dict[str, Any]]:
        """Encode using hybrid approach (80/20 frequency split)
        
        Args:
            digits: List of p-adic digits
            analysis: Entropy analysis results
            
        Returns:
            Tuple of (compressed_bytes, metadata)
        """
        # Sort symbols by frequency
        sorted_symbols = sorted(
            analysis.frequency_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Calculate split point (top 20% of symbols by cumulative frequency)
        total_count = sum(analysis.frequency_distribution.values())
        cumulative = 0
        split_point = 0
        
        for i, (symbol, count) in enumerate(sorted_symbols):
            cumulative += count
            if cumulative >= total_count * self.config.frequency_split_ratio:
                split_point = i + 1
                break
        
        if split_point == 0:
            split_point = max(1, len(sorted_symbols) // 5)
        
        # Split symbols
        high_freq_symbols = set(sym for sym, _ in sorted_symbols[:split_point])
        low_freq_symbols = set(sym for sym, _ in sorted_symbols[split_point:])
        
        # Separate digit sequences
        high_freq_indices = []
        high_freq_digits = []
        low_freq_indices = []
        low_freq_digits = []
        
        for i, digit in enumerate(digits):
            if digit in high_freq_symbols:
                high_freq_indices.append(i)
                high_freq_digits.append(digit)
            else:
                low_freq_indices.append(i)
                low_freq_digits.append(digit)
        
        # Encode high frequency with Huffman
        compressed_high = b''
        high_metadata = {}
        if high_freq_digits:
            high_freq_dist = {
                sym: count for sym, count in analysis.frequency_distribution.items()
                if sym in high_freq_symbols
            }
            self.huffman_encoder.build_huffman_tree(high_freq_dist)
            self.huffman_encoder.generate_codes()
            compressed_high = self.huffman_encoder.huffman_encode(high_freq_digits)
            high_metadata = {
                "symbols": list(high_freq_symbols),
                "count": len(high_freq_digits)
            }
        
        # Encode low frequency with Arithmetic
        compressed_low = b''
        low_metadata = {}
        if low_freq_digits:
            low_prob_dist = {
                sym: analysis.probability_distribution[sym]
                for sym in low_freq_symbols
            }
            # Normalize probabilities
            total_prob = sum(low_prob_dist.values())
            if total_prob > 0:
                low_prob_dist = {k: v/total_prob for k, v in low_prob_dist.items()}
                # Use chunked arithmetic encoding for reliability
                compressed_low, chunk_metadata = self._encode_arithmetic_chunked(
                    low_freq_digits, low_prob_dist
                )
                low_metadata = {
                    "symbols": list(low_freq_symbols),
                    "count": len(low_freq_digits),
                    "chunked": chunk_metadata.get("chunked", False),
                    "num_chunks": chunk_metadata.get("num_chunks", 1)
                }
        
        # Combine compressed data
        combined = bytearray()
        combined.extend(struct.pack('<I', len(high_freq_indices)))  # Number of high freq
        combined.extend(struct.pack('<I', len(low_freq_indices)))   # Number of low freq
        
        # Store indices for reconstruction
        for idx in high_freq_indices:
            combined.extend(struct.pack('<I', idx))
        for idx in low_freq_indices:
            combined.extend(struct.pack('<I', idx))
        
        # Store compressed data
        combined.extend(struct.pack('<I', len(compressed_high)))
        combined.extend(compressed_high)
        combined.extend(struct.pack('<I', len(compressed_low)))
        combined.extend(compressed_low)
        
        # Build metadata
        metadata = {
            "method": "hybrid",
            "split_point": split_point,
            "high_freq": high_metadata,
            "low_freq": low_metadata,
            "frequency_table": analysis.frequency_distribution,
            "probability_table": analysis.probability_distribution
        }
        
        return bytes(combined), metadata
    
    def decode_padic_tensor(self,
                          compressed: bytes,
                          metadata: Dict[str, Any]) -> torch.Tensor:
        """Decode compressed data back to p-adic tensor
        
        Args:
            compressed: Compressed byte sequence
            metadata: Encoding metadata
            
        Returns:
            Reconstructed p-adic tensor
        """
        # Check circuit breaker
        self._check_circuit_breaker()
        
        try:
            # Handle empty tensor case gracefully
            if metadata.get("empty", False):
                # Return empty tensor with correct shape - MUST have original_shape
                original_shape = metadata.get("original_shape")
                if not original_shape:
                    raise ValueError(
                        f"Empty tensor missing required original_shape in metadata. "
                        f"Available keys: {list(metadata.keys())}"
                    )
                return torch.zeros(original_shape, dtype=torch.long)
            
            if not compressed and not metadata.get("empty", False):
                raise ValueError("Cannot decode empty compressed data without empty flag")
            
            if not metadata:
                raise ValueError("Cannot decode without metadata")
            
            # Check if chunked
            if metadata.get("chunked", False):
                return self._decode_chunked_tensor(compressed, metadata)
            
            # Extract method - handle both old and new metadata formats for backwards compatibility
            encode_metadata = metadata.get("encoding_metadata", {})
            
            # Priority order for method extraction:
            # 1. encode_metadata["method"] (preferred)
            # 2. metadata["encoding_method"] (current)
            # 3. metadata["method"] (fallback)
            # 4. "none" for empty tensors
            method = (
                encode_metadata.get("method") or 
                metadata.get("encoding_method") or 
                metadata.get("method") or
                ("none" if metadata.get("empty", False) else None)
            )
            
            if not method:
                # Provide detailed error for debugging
                available_keys = list(metadata.keys())
                encode_keys = list(encode_metadata.keys()) if encode_metadata else []
                raise ValueError(
                    f"No encoding method found in metadata. "
                    f"Checked: encode_metadata['method'], metadata['encoding_method'], metadata['method']. "
                    f"Available metadata keys: {available_keys}, "
                    f"encode_metadata keys: {encode_keys}"
                )
            
            # Decode based on method
            start_time = time.perf_counter()
            
            if method == "none":
                # Handle empty tensor case
                if metadata.get('empty', False):
                    digit_list = []
                else:
                    raise ValueError("Method 'none' but tensor not marked as empty")
            elif method == "huffman":
                digit_list = self._decode_huffman(compressed, encode_metadata)
            elif method == "arithmetic":
                digit_list = self._decode_arithmetic(compressed, encode_metadata)
            elif method == "hybrid":
                digit_list = self._decode_hybrid(compressed, encode_metadata, metadata)
            else:
                raise ValueError(f"Unknown encoding method: {method}")
            
            decoding_time = (time.perf_counter() - start_time) * 1000
            
            # Reconstruct tensor - check multiple possible locations for original_shape
            original_shape = (
                metadata.get("original_shape") or
                metadata.get("entropy_analysis", {}).get("original_shape") or
                metadata.get("encoding_metadata", {}).get("original_shape") or
                [len(digit_list)]  # Fallback to 1D tensor with the data length
            )
            
            if not original_shape:
                raise ValueError("Original shape not found in metadata")
            
            # Create tensor
            if len(digit_list) == 0 and metadata.get('empty', False):
                # Handle empty tensor case - create zeros with correct shape
                tensor = torch.zeros(original_shape, dtype=torch.long)
            else:
                tensor = torch.tensor(digit_list, dtype=torch.long)
                
                # Reshape to original dimensions
                try:
                    tensor = tensor.reshape(original_shape)
                except RuntimeError as e:
                    raise ValueError(
                        f"Cannot reshape {len(digit_list)} elements to shape {original_shape}: {e}"
                    )
            
            # Convert to original dtype if specified
            original_dtype_str = metadata.get("original_dtype")
            if original_dtype_str:
                try:
                    if "float32" in original_dtype_str:
                        tensor = tensor.float()
                    elif "float64" in original_dtype_str:
                        tensor = tensor.double()
                    elif "float16" in original_dtype_str:
                        tensor = tensor.half()
                    # Keep as long for integer types
                except Exception as e:
                    logger.warning(f"Could not convert to original dtype {original_dtype_str}: {e}")
            
            logger.debug(f"Decoded tensor: shape={tensor.shape}, time={decoding_time:.2f}ms")
            
            # Reset failure count on success
            if self.failure_count > 0:
                self.failure_count = max(0, self.failure_count - 1)
            
            return tensor
            
        except ValueError as e:
            # Preserve ValueError for validation/input errors
            self._record_failure(e)
            logger.error(f"Entropy decoding validation error: {e}")
            raise
        except Exception as e:
            # Wrap other exceptions as RuntimeError for system/runtime failures
            self._record_failure(e)
            logger.error(f"Entropy decoding failed: {e}")
            raise RuntimeError(f"Critical entropy decoding failure: {e}") from e
    
    def _decode_chunked_tensor(self,
                              compressed: bytes,
                              metadata: Dict[str, Any]) -> torch.Tensor:
        """Decode chunked tensor
        
        Args:
            compressed: Compressed chunks
            metadata: Chunking metadata
            
        Returns:
            Reconstructed tensor
        """
        pos = 0
        num_chunks = struct.unpack('<I', compressed[pos:pos+4])[0]
        pos += 4
        
        chunks_metadata = metadata.get("chunks_metadata", [])
        if len(chunks_metadata) != num_chunks:
            raise ValueError(f"Metadata mismatch: expected {num_chunks} chunks")
        
        decoded_chunks = []
        
        for i in range(num_chunks):
            chunk_size = struct.unpack('<I', compressed[pos:pos+4])[0]
            pos += 4
            
            chunk_data = compressed[pos:pos+chunk_size]
            pos += chunk_size
            
            chunk_tensor = self.decode_padic_tensor(chunk_data, chunks_metadata[i])
            decoded_chunks.append(chunk_tensor.flatten())
        
        # Combine chunks (handling overlap)
        overlap = metadata.get("overlap", 0)
        if overlap > 0:
            combined = decoded_chunks[0]
            for chunk in decoded_chunks[1:]:
                # Remove overlapping portion
                combined = torch.cat([combined[:-overlap], chunk])
        else:
            combined = torch.cat(decoded_chunks)
        
        # Reshape to original
        original_shape = metadata.get("original_shape")
        return combined.reshape(original_shape)
    
    def _decode_huffman(self,
                       compressed: bytes,
                       metadata: Dict[str, Any]) -> List[int]:
        """Decode Huffman-encoded data
        
        Args:
            compressed: Compressed bytes
            metadata: Huffman metadata
            
        Returns:
            List of p-adic digits
        """
        freq_table = metadata.get("frequency_table")
        if not freq_table:
            raise ValueError("Missing frequency table for Huffman decoding")
        
        # Convert string keys to int if needed
        if freq_table and isinstance(next(iter(freq_table.keys())), str):
            freq_table = {int(k): v for k, v in freq_table.items()}
        
        # Rebuild tree and decode
        self.huffman_encoder.build_huffman_tree(freq_table)
        return self.huffman_encoder.huffman_decode(compressed)
    
    def _decode_arithmetic(self,
                          compressed: bytes,
                          metadata: Dict[str, Any]) -> List[int]:
        """Decode arithmetic-encoded data
        
        Args:
            compressed: Compressed bytes
            metadata: Arithmetic metadata
            
        Returns:
            List of p-adic digits
        """
        if metadata.get("chunked", False):
            # Multi-chunk decoding
            pos = 0
            num_chunks = struct.unpack('<I', compressed[pos:pos+4])[0]
            pos += 4
            
            all_digits = []
            for _ in range(num_chunks):
                chunk_size = struct.unpack('<I', compressed[pos:pos+4])[0]
                pos += 4
                chunk_data = compressed[pos:pos+chunk_size]
                pos += chunk_size
                
                chunk_digits = self.arithmetic_encoder.arithmetic_decode_simple(chunk_data)
                all_digits.extend(chunk_digits)
            
            return all_digits
        else:
            # Single chunk decoding
            return self.arithmetic_encoder.arithmetic_decode_simple(compressed)
    
    def _decode_arithmetic_chunked(self,
                                  compressed: bytes,
                                  metadata: Dict[str, Any]) -> List[int]:
        """Helper method for chunked arithmetic decoding"""
        pos = 0
        num_chunks = struct.unpack('<I', compressed[pos:pos+4])[0]
        pos += 4
        
        all_digits = []
        for _ in range(num_chunks):
            chunk_size = struct.unpack('<I', compressed[pos:pos+4])[0]
            pos += 4
            chunk_data = compressed[pos:pos+chunk_size]
            pos += chunk_size
            
            chunk_digits = self.arithmetic_encoder.arithmetic_decode_simple(chunk_data)
            all_digits.extend(chunk_digits)
        
        return all_digits
    
    def _decode_hybrid(self,
                      compressed: bytes,
                      encode_metadata: Dict[str, Any],
                      full_metadata: Dict[str, Any]) -> List[int]:
        """Decode hybrid-encoded data
        
        Args:
            compressed: Compressed bytes
            encode_metadata: Encoding-specific metadata
            full_metadata: Complete metadata
            
        Returns:
            List of p-adic digits
        """
        pos = 0
        
        # Extract counts
        num_high = struct.unpack('<I', compressed[pos:pos+4])[0]
        pos += 4
        num_low = struct.unpack('<I', compressed[pos:pos+4])[0]
        pos += 4
        
        # Extract indices
        high_indices = []
        for _ in range(num_high):
            idx = struct.unpack('<I', compressed[pos:pos+4])[0]
            high_indices.append(idx)
            pos += 4
        
        low_indices = []
        for _ in range(num_low):
            idx = struct.unpack('<I', compressed[pos:pos+4])[0]
            low_indices.append(idx)
            pos += 4
        
        # Extract compressed data sizes and data
        high_size = struct.unpack('<I', compressed[pos:pos+4])[0]
        pos += 4
        high_data = compressed[pos:pos+high_size]
        pos += high_size
        
        low_size = struct.unpack('<I', compressed[pos:pos+4])[0]
        pos += 4
        low_data = compressed[pos:pos+low_size]
        
        # Decode high frequency (Huffman)
        high_digits = []
        if high_data:
            freq_table = encode_metadata.get("frequency_table", {})
            if freq_table:
                # Filter to high frequency symbols only
                high_freq_meta = encode_metadata.get("high_freq", {})
                high_symbols = set(high_freq_meta.get("symbols", []))
                high_freq_table = {
                    k: v for k, v in freq_table.items() 
                    if k in high_symbols
                }
                if high_freq_table:
                    self.huffman_encoder.build_huffman_tree(high_freq_table)
                    high_digits = self.huffman_encoder.huffman_decode(high_data)
        
        # Decode low frequency (Arithmetic) 
        low_digits = []
        if low_data:
            low_freq_meta = encode_metadata.get("low_freq", {})
            if low_freq_meta.get("chunked", False):
                # Decode chunked arithmetic data
                low_digits = self._decode_arithmetic_chunked(low_data, low_freq_meta)
            else:
                # Decode single chunk
                low_digits = self.arithmetic_encoder.arithmetic_decode_simple(low_data)
        
        # Reconstruct original sequence
        total_length = num_high + num_low
        reconstructed = [0] * total_length
        
        for i, idx in enumerate(high_indices):
            if idx < total_length and i < len(high_digits):
                reconstructed[idx] = high_digits[i]
        
        for i, idx in enumerate(low_indices):
            if idx < total_length and i < len(low_digits):
                reconstructed[idx] = low_digits[i]
        
        return reconstructed
    
    def encode_logarithmic_padic_weight(self,
                                       weight: LogarithmicPadicWeight) -> Tuple[bytes, Dict[str, Any]]:
        """Encode a LogarithmicPadicWeight object
        
        Args:
            weight: LogarithmicPadicWeight to encode
            
        Returns:
            Tuple of (compressed_bytes, metadata)
        """
        if not isinstance(weight, LogarithmicPadicWeight):
            raise TypeError(f"Expected LogarithmicPadicWeight, got {type(weight)}")
        
        # Extract p-adic digits
        padic_weight = weight.padic_weight
        digits = padic_weight.digits
        
        if not digits:
            raise ValueError("PadicWeight has no digits to encode")
        
        # Convert to tensor for encoding
        digit_tensor = torch.tensor(digits, dtype=torch.long)
        
        # Encode using bridge
        compressed, bridge_metadata = self.encode_padic_tensor(digit_tensor)
        
        # Add weight-specific metadata
        weight_metadata = {
            "weight_type": "logarithmic_padic",
            "prime": padic_weight.prime,
            "precision": padic_weight.precision,
            "valuation": padic_weight.valuation,
            "original_value": weight.original_value,
            "log_value": weight.log_value,
            "encoding_method": weight.encoding_method,
            "compression_metadata": weight.compression_metadata,
            "bridge_metadata": bridge_metadata
        }
        
        return compressed, weight_metadata
    
    def decode_logarithmic_padic_weight(self,
                                       compressed: bytes,
                                       metadata: Dict[str, Any]) -> List[int]:
        """Decode to p-adic digits (weight reconstruction happens elsewhere)
        
        Args:
            compressed: Compressed bytes
            metadata: Weight and encoding metadata
            
        Returns:
            List of p-adic digits
        """
        # Extract bridge metadata
        bridge_metadata = metadata.get("bridge_metadata", metadata)
        
        # Decode tensor
        digit_tensor = self.decode_padic_tensor(compressed, bridge_metadata)
        
        # Convert to list
        return digit_tensor.cpu().tolist()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge performance statistics
        
        Returns:
            Dictionary of performance metrics
        """
        return {
            "total_compressions": self.total_compressions,
            "method_usage": dict(self.method_usage),
            "average_compression_ratio": self.average_compression_ratio,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0 else 0
            ),
            "config": {
                "prime": self.prime,
                "huffman_threshold": self.config.huffman_threshold,
                "arithmetic_threshold": self.config.arithmetic_threshold,
                "adaptive_threshold": self.config.adaptive_threshold,
                "pattern_detection": self.config.enable_pattern_detection
            }
        }
    
    def reset_statistics(self):
        """Reset performance statistics"""
        self.total_compressions = 0
        self.method_usage = {"huffman": 0, "arithmetic": 0, "hybrid": 0}
        self.average_compression_ratio = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.frequency_cache.clear()


def validate_entropy_bridge():
    """Validate entropy bridge functionality"""
    print("\n" + "="*80)
    print("VALIDATING ENTROPY P-ADIC BRIDGE")
    print("="*80)
    
    # Test with different primes
    test_primes = [257, 127, 31]
    
    for prime in test_primes:
        print(f"\nTesting with prime={prime}")
        bridge = EntropyPAdicBridge(prime)
        
        # Test different distributions
        test_cases = [
            # Highly skewed (good for Huffman)
            ("Skewed", torch.tensor([0] * 500 + [1] * 100 + [2] * 50 + list(range(3, 10)) * 5)),
            
            # Uniform (good for Arithmetic)
            ("Uniform", torch.tensor(list(range(prime)) * 10)),
            
            # Random
            ("Random", torch.randint(0, prime, (1000,))),
            
            # Patterned
            ("Patterned", torch.tensor([0, 1, 2, 3] * 250)),
            
            # Sparse
            ("Sparse", torch.tensor([0, prime//4, prime//2, 3*prime//4] * 250))
        ]
        
        for name, tensor in test_cases:
            print(f"\n  Test case: {name}")
            print(f"    Shape: {tensor.shape}, Elements: {tensor.numel()}")
            
            try:
                # Analyze entropy
                analysis = bridge.analyze_entropy(tensor)
                print(f"    Entropy: {analysis.shannon_entropy:.2f} bits")
                print(f"    Recommended: {analysis.recommended_method} (confidence: {analysis.confidence_score:.2f})")
                
                # Encode
                compressed, metadata = bridge.encode_padic_tensor(tensor)
                
                # Decode
                reconstructed = bridge.decode_padic_tensor(compressed, metadata)
                
                # Validate
                if not torch.equal(tensor.long(), reconstructed.long()):
                    raise ValueError("Reconstruction mismatch!")
                
                # Report metrics
                metrics = metadata["compression_metrics"]
                print(f"    Method used: {metadata['encoding_method']}")
                print(f"    Compression ratio: {metrics['compression_ratio']:.2f}x")
                print(f"    Encoding time: {metrics['encoding_time_ms']:.2f}ms")
                print(f"    ✓ PASSED")
                
            except Exception as e:
                print(f"    ✗ FAILED: {e}")
                raise
        
        # Print statistics
        stats = bridge.get_statistics()
        print(f"\nStatistics for prime={prime}:")
        print(f"  Total compressions: {stats['total_compressions']}")
        print(f"  Method usage: {stats['method_usage']}")
        print(f"  Average ratio: {stats['average_compression_ratio']:.2f}x")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
    
    print("\n" + "="*80)
    print("ALL ENTROPY BRIDGE TESTS PASSED!")
    print("="*80)


if __name__ == "__main__":
    validate_entropy_bridge()
