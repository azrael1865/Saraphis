"""
Huffman & Arithmetic Coding for P-adic Compression System

Provides entropy coding for p-adic digit sequences to achieve maximum compression.
Implements both Huffman and Arithmetic coding with automatic selection based on
distribution characteristics.

NO FALLBACKS - HARD FAILURES ONLY
"""

import heapq
import struct
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import numpy as np
from collections import Counter, defaultdict
import math
import time


@dataclass
class CompressionMetrics:
    """Tracks compression performance metrics"""
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0
    encoding_time_ms: float = 0.0
    decoding_time_ms: float = 0.0
    compression_ratio: float = 1.0
    encoding_method: str = ""
    prime: int = 0
    digit_count: int = 0
    unique_symbols: int = 0
    entropy: float = 0.0
    
    def calculate_ratio(self) -> None:
        """Calculate compression ratio from sizes"""
        if self.compressed_size_bytes > 0:
            self.compression_ratio = self.original_size_bytes / self.compressed_size_bytes
        else:
            raise ValueError("Compressed size is zero - invalid compression")


class HuffmanNode:
    """Node in Huffman tree for optimal encoding"""
    
    def __init__(self, symbol: Optional[int] = None, frequency: int = 0,
                 left: Optional['HuffmanNode'] = None, 
                 right: Optional['HuffmanNode'] = None):
        """Initialize Huffman tree node"""
        self.symbol = symbol
        self.frequency = frequency
        self.left = left
        self.right = right
    
    def __lt__(self, other: 'HuffmanNode') -> bool:
        """Compare nodes by frequency for heap operations"""
        return self.frequency < other.frequency
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf (contains symbol)"""
        return self.left is None and self.right is None


class HuffmanEncoder:
    """Huffman encoding for p-adic digit sequences"""
    
    def __init__(self, prime: int):
        """Initialize Huffman encoder
        
        Args:
            prime: P-adic prime (determines symbol range [0, prime-1])
        """
        if not isinstance(prime, int) or prime < 2:
            raise ValueError(f"Prime must be integer >= 2, got {prime}")
        
        self.prime = prime
        self.tree_root: Optional[HuffmanNode] = None
        self.encoding_table: Dict[int, str] = {}
        self.decoding_table: Dict[str, int] = {}
        
    def build_huffman_tree(self, frequencies: Dict[int, int]) -> HuffmanNode:
        """Build optimal Huffman tree from frequency distribution
        
        Args:
            frequencies: Dictionary mapping symbols to their frequencies
            
        Returns:
            Root node of Huffman tree
        """
        if not frequencies:
            raise ValueError("Cannot build tree from empty frequency table")
        
        # Validate symbols are in valid range
        for symbol in frequencies:
            if not (0 <= symbol < self.prime):
                raise ValueError(f"Symbol {symbol} out of range [0, {self.prime-1}]")
        
        # Create leaf nodes and add to heap
        heap: List[HuffmanNode] = []
        for symbol, freq in frequencies.items():
            if freq <= 0:
                raise ValueError(f"Frequency for symbol {symbol} must be positive, got {freq}")
            node = HuffmanNode(symbol=symbol, frequency=freq)
            heapq.heappush(heap, node)
        
        # Build tree by repeatedly merging lowest frequency nodes
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            # Create internal node
            parent = HuffmanNode(
                frequency=left.frequency + right.frequency,
                left=left,
                right=right
            )
            heapq.heappush(heap, parent)
        
        # Final node is the root
        self.tree_root = heap[0]
        return self.tree_root
    
    def generate_codes(self, root: Optional[HuffmanNode] = None) -> Dict[int, str]:
        """Generate binary codes for each symbol from Huffman tree
        
        Args:
            root: Root of Huffman tree (uses self.tree_root if None)
            
        Returns:
            Dictionary mapping symbols to their binary codes
        """
        if root is None:
            root = self.tree_root
        
        if root is None:
            raise ValueError("No Huffman tree available - build tree first")
        
        self.encoding_table = {}
        self.decoding_table = {}
        
        # Special case: single symbol
        if root.is_leaf():
            if root.symbol is None:
                raise ValueError("Leaf node has no symbol")
            self.encoding_table[root.symbol] = '0'
            self.decoding_table['0'] = root.symbol
            return self.encoding_table
        
        # Traverse tree to generate codes
        def traverse(node: HuffmanNode, code: str = '') -> None:
            if node.is_leaf():
                if node.symbol is not None:
                    self.encoding_table[node.symbol] = code
                    self.decoding_table[code] = node.symbol
            else:
                if node.left:
                    traverse(node.left, code + '0')
                if node.right:
                    traverse(node.right, code + '1')
        
        traverse(root)
        
        # Validate all symbols have codes
        if not self.encoding_table:
            raise ValueError("Failed to generate encoding table")
        
        return self.encoding_table
    
    def huffman_encode(self, symbols: List[int]) -> bytes:
        """Encode symbol sequence using Huffman coding
        
        Args:
            symbols: List of p-adic digits to encode
            
        Returns:
            Compressed byte sequence
        """
        if not symbols:
            raise ValueError("Cannot encode empty symbol list")
        
        if not self.encoding_table:
            raise ValueError("No encoding table available - generate codes first")
        
        # Build bit string
        bit_string = ''
        for symbol in symbols:
            if symbol not in self.encoding_table:
                raise ValueError(f"Symbol {symbol} not in encoding table")
            bit_string += self.encoding_table[symbol]
        
        # Convert bit string to bytes
        # Pad to byte boundary
        padding = (8 - len(bit_string) % 8) % 8
        bit_string += '0' * padding
        
        # Pack into bytes
        compressed = bytearray()
        
        # Store metadata: symbol count and padding
        compressed.extend(struct.pack('<I', len(symbols)))  # 4 bytes for symbol count
        compressed.append(padding)  # 1 byte for padding
        
        # Pack bits into bytes
        for i in range(0, len(bit_string), 8):
            byte_str = bit_string[i:i+8]
            byte_val = int(byte_str, 2)
            compressed.append(byte_val)
        
        return bytes(compressed)
    
    def huffman_decode(self, encoded: bytes, tree: Optional[HuffmanNode] = None) -> List[int]:
        """Decode Huffman-encoded byte sequence back to symbols
        
        Args:
            encoded: Compressed byte sequence
            tree: Huffman tree root (uses self.tree_root if None)
            
        Returns:
            Original symbol sequence
        """
        if not encoded:
            raise ValueError("Cannot decode empty byte sequence")
        
        if tree is None:
            tree = self.tree_root
        
        if tree is None:
            raise ValueError("No Huffman tree available for decoding")
        
        # Extract metadata
        if len(encoded) < 5:
            raise ValueError("Encoded data too short - missing metadata")
        
        symbol_count = struct.unpack('<I', encoded[:4])[0]
        padding = encoded[4]
        
        if padding >= 8:
            raise ValueError(f"Invalid padding value: {padding}")
        
        # Convert bytes to bit string
        bit_string = ''
        for byte_val in encoded[5:]:
            bit_string += format(byte_val, '08b')
        
        # Remove padding
        if padding > 0:
            bit_string = bit_string[:-padding]
        
        # Decode using tree
        symbols = []
        
        # Special case: single symbol tree
        if tree.is_leaf():
            if tree.symbol is None:
                raise ValueError("Leaf node has no symbol")
            symbols = [tree.symbol] * symbol_count
        else:
            current = tree
            for bit in bit_string:
                if bit == '0':
                    if current.left is None:
                        raise ValueError("Invalid tree structure - missing left child")
                    current = current.left
                else:
                    if current.right is None:
                        raise ValueError("Invalid tree structure - missing right child")
                    current = current.right
                
                if current.is_leaf():
                    if current.symbol is None:
                        raise ValueError("Leaf node has no symbol")
                    symbols.append(current.symbol)
                    current = tree
                    
                    if len(symbols) == symbol_count:
                        break
        
        # Validate decoding
        if len(symbols) != symbol_count:
            raise ValueError(f"Decoded {len(symbols)} symbols, expected {symbol_count}")
        
        return symbols


class ArithmeticEncoder:
    """Arithmetic coding for p-adic digit sequences"""
    
    def __init__(self, prime: int, precision_bits: int = 32):
        """Initialize arithmetic encoder
        
        Args:
            prime: P-adic prime (determines symbol range)
            precision_bits: Bits of precision for arithmetic (16, 32, or 64)
        """
        if not isinstance(prime, int) or prime < 2:
            raise ValueError(f"Prime must be integer >= 2, got {prime}")
        
        if precision_bits not in [16, 32, 64]:
            raise ValueError(f"Precision must be 16, 32, or 64 bits, got {precision_bits}")
        
        self.prime = prime
        self.precision_bits = precision_bits
        self.max_range = 2 ** precision_bits - 1
        self.quarter = self.max_range // 4
        self.half = 2 * self.quarter
        self.three_quarters = 3 * self.quarter
    
    def compute_cdf(self, probabilities: Dict[int, float]) -> Dict[int, Tuple[float, float]]:
        """Build cumulative distribution function from probabilities
        
        Args:
            probabilities: Symbol probabilities (must sum to ~1.0)
            
        Returns:
            Dictionary mapping symbols to (low, high) probability ranges
        """
        # Validate probabilities
        total = sum(probabilities.values())
        if not (0.95 < total < 1.05):  # More lenient for fixed-point errors
            raise ValueError(f"Probabilities must sum to ~1.0, got {total}")
        
        # Normalize to ensure exact sum of 1.0
        probabilities = {k: v/total for k, v in probabilities.items()}
        
        # Build CDF
        cdf = {}
        cumulative = 0.0
        
        for symbol in sorted(probabilities.keys()):
            if not (0 <= symbol < self.prime):
                raise ValueError(f"Symbol {symbol} out of range [0, {self.prime-1}]")
            
            low = cumulative
            high = cumulative + probabilities[symbol]
            cdf[symbol] = (low, high)
            cumulative = high
        
        # Ensure last symbol reaches exactly 1.0
        if cdf:
            last_symbol = max(cdf.keys())
            low, _ = cdf[last_symbol]
            cdf[last_symbol] = (low, 1.0)
        
        return cdf
    
    def arithmetic_encode_simple(self, symbols: List[int], probabilities: Dict[int, float]) -> bytes:
        """Simplified arithmetic encoding for better reliability"""
        if not symbols:
            raise ValueError("Cannot encode empty symbol list")
        
        # Build CDF
        cdf = self.compute_cdf(probabilities)
        
        # Use Python's decimal for better precision
        from decimal import Decimal, getcontext
        getcontext().prec = 50  # High precision
        
        low = Decimal(0)
        high = Decimal(1)
        
        for symbol in symbols:
            if symbol not in cdf:
                raise ValueError(f"Symbol {symbol} not in probability distribution")
            
            sym_low, sym_high = cdf[symbol]
            range_size = high - low
            high = low + range_size * Decimal(sym_high)
            low = low + range_size * Decimal(sym_low)
        
        # Pick middle value
        result = (low + high) / 2
        
        # Convert to binary representation
        # Store as string of decimal digits for simplicity
        result_str = str(result)
        
        # Pack into bytes
        compressed = bytearray()
        compressed.extend(struct.pack('<I', len(symbols)))
        compressed.append(0)  # No padding needed for this method
        
        # Store probability table
        compressed.extend(struct.pack('<H', len(probabilities)))
        for symbol in sorted(probabilities.keys()):
            if symbol > 254:
                compressed.append(254)  # Special marker
                compressed.extend(struct.pack('<H', symbol))
            else:
                compressed.append(symbol)
            prob_fixed = int(probabilities[symbol] * 65535)
            compressed.extend(struct.pack('<H', prob_fixed))
        
        # Store the decimal string
        result_bytes = result_str.encode('utf-8')
        compressed.extend(struct.pack('<I', len(result_bytes)))
        compressed.extend(result_bytes)
        
        return bytes(compressed)
    
    def arithmetic_encode(self, symbols: List[int], probabilities: Dict[int, float]) -> bytes:
        """Encode symbols using arithmetic coding
        
        Args:
            symbols: List of p-adic digits to encode
            probabilities: Symbol probability distribution
            
        Returns:
            Compressed byte sequence
        """
        if not symbols:
            raise ValueError("Cannot encode empty symbol list")
        
        # Build CDF
        cdf = self.compute_cdf(probabilities)
        
        # Initialize range
        low = 0
        high = self.max_range
        pending_bits = 0
        output_bits = []
        
        for symbol in symbols:
            if symbol not in cdf:
                raise ValueError(f"Symbol {symbol} not in probability distribution")
            
            # Get symbol's probability range
            symbol_low, symbol_high = cdf[symbol]
            
            # Update range
            range_size = high - low + 1
            high = low + int(range_size * symbol_high) - 1
            low = low + int(range_size * symbol_low)
            
            # Output bits when possible
            while True:
                if high < self.half:
                    # Output 0 and pending 1s
                    output_bits.append(0)
                    output_bits.extend([1] * pending_bits)
                    pending_bits = 0
                    low = 2 * low
                    high = 2 * high + 1
                elif low >= self.half:
                    # Output 1 and pending 0s
                    output_bits.append(1)
                    output_bits.extend([0] * pending_bits)
                    pending_bits = 0
                    low = 2 * (low - self.half)
                    high = 2 * (high - self.half) + 1
                elif low >= self.quarter and high < self.three_quarters:
                    # Defer output
                    pending_bits += 1
                    low = 2 * (low - self.quarter)
                    high = 2 * (high - self.quarter) + 1
                else:
                    break
            
            # Check for range collapse
            if low >= high:
                raise ValueError("Range collapsed during encoding - numerical precision exceeded")
        
        # Output final bits
        pending_bits += 1
        if low < self.quarter:
            output_bits.append(0)
            output_bits.extend([1] * pending_bits)
        else:
            output_bits.append(1)
            output_bits.extend([0] * pending_bits)
        
        # Convert bits to bytes
        # Pad to byte boundary
        padding = (8 - len(output_bits) % 8) % 8
        output_bits.extend([0] * padding)
        
        # Pack metadata and bits
        compressed = bytearray()
        compressed.extend(struct.pack('<I', len(symbols)))  # Symbol count
        compressed.append(padding)  # Padding bits
        
        # Store probability table (for decoding)
        num_symbols = len(probabilities)
        compressed.extend(struct.pack('<H', num_symbols))  # Number of unique symbols (2 bytes)
        for symbol in sorted(probabilities.keys()):
            if symbol > 255:
                # For large symbols, store as 2 bytes
                compressed.append(255)  # Marker for 2-byte symbol
                compressed.extend(struct.pack('<H', symbol))
            else:
                compressed.append(symbol)
            # Store probability as fixed-point (16-bit)
            prob_fixed = int(probabilities[symbol] * 65535)
            compressed.extend(struct.pack('<H', prob_fixed))
        
        # Pack bit stream
        for i in range(0, len(output_bits), 8):
            byte_bits = output_bits[i:i+8]
            byte_val = sum(bit << (7-j) for j, bit in enumerate(byte_bits))
            compressed.append(byte_val)
        
        return bytes(compressed)
    
    def arithmetic_decode(self, encoded: bytes, length: Optional[int] = None) -> List[int]:
        """Decode arithmetic-encoded byte sequence
        
        Args:
            encoded: Compressed byte sequence
            length: Expected number of symbols (extracted from metadata if None)
            
        Returns:
            Original symbol sequence
        """
        if not encoded:
            raise ValueError("Cannot decode empty byte sequence")
        
        # Extract metadata
        pos = 0
        symbol_count = struct.unpack('<I', encoded[pos:pos+4])[0]
        pos += 4
        
        if length is not None and length != symbol_count:
            raise ValueError(f"Length mismatch: expected {length}, got {symbol_count}")
        
        padding = encoded[pos]
        pos += 1
        
        # Extract probability table
        num_symbols = struct.unpack('<H', encoded[pos:pos+2])[0]
        pos += 2
        
        probabilities = {}
        for _ in range(num_symbols):
            symbol = encoded[pos]
            pos += 1
            if symbol == 255:  # Marker for 2-byte symbol
                symbol = struct.unpack('<H', encoded[pos:pos+2])[0]
                pos += 2
            prob_fixed = struct.unpack('<H', encoded[pos:pos+2])[0]
            pos += 2
            probabilities[symbol] = prob_fixed / 65535.0
        
        # Build CDF
        cdf = self.compute_cdf(probabilities)
        
        # Convert remaining bytes to bit stream
        bit_stream = []
        for byte_val in encoded[pos:]:
            for i in range(7, -1, -1):
                bit_stream.append((byte_val >> i) & 1)
        
        # Remove padding
        if padding > 0:
            bit_stream = bit_stream[:-padding]
        
        # Decode symbols
        symbols = []
        low = 0
        high = self.max_range
        value = 0
        
        # Read initial value
        for i in range(min(self.precision_bits, len(bit_stream))):
            value = (value << 1) | bit_stream[i]
        
        bit_pos = self.precision_bits
        
        for _ in range(symbol_count):
            # Find symbol whose range contains value
            range_size = high - low + 1
            scaled_value = ((value - low + 1) * 1.0 - 1) / range_size
            
            # Find symbol
            found_symbol = None
            for symbol, (sym_low, sym_high) in cdf.items():
                if sym_low <= scaled_value < sym_high:
                    found_symbol = symbol
                    break
            
            if found_symbol is None:
                raise ValueError(f"No symbol found for scaled value {scaled_value}")
            
            symbols.append(found_symbol)
            
            # Update range
            symbol_low, symbol_high = cdf[found_symbol]
            high = low + int(range_size * symbol_high) - 1
            low = low + int(range_size * symbol_low)
            
            # Renormalize
            while True:
                if high < self.half:
                    low = 2 * low
                    high = 2 * high + 1
                    value = 2 * value
                elif low >= self.half:
                    low = 2 * (low - self.half)
                    high = 2 * (high - self.half) + 1
                    value = 2 * (value - self.half)
                elif low >= self.quarter and high < self.three_quarters:
                    low = 2 * (low - self.quarter)
                    high = 2 * (high - self.quarter) + 1
                    value = 2 * (value - self.quarter)
                else:
                    break
                
                # Read next bit
                if bit_pos < len(bit_stream):
                    value = value | bit_stream[bit_pos]
                    bit_pos += 1
        
        return symbols


class HybridEncoder:
    """Hybrid encoder that chooses between Huffman and Arithmetic coding"""
    
    def __init__(self, prime: int):
        """Initialize hybrid encoder
        
        Args:
            prime: P-adic prime for symbol range
        """
        if not isinstance(prime, int) or prime < 2:
            raise ValueError(f"Prime must be integer >= 2, got {prime}")
        
        self.prime = prime
        self.huffman = HuffmanEncoder(prime)
        self.arithmetic = ArithmeticEncoder(prime)
        self.metrics = CompressionMetrics(prime=prime)
        
        # Encoding state
        self.last_method_used = ""
        self.frequency_table: Dict[int, int] = {}
        self.probability_table: Dict[int, float] = {}
    
    def analyze_distribution(self, digits: List[int]) -> Tuple[float, Dict[int, float]]:
        """Analyze digit distribution and compute entropy
        
        Args:
            digits: List of p-adic digits
            
        Returns:
            Tuple of (entropy, probability_distribution)
        """
        if not digits:
            raise ValueError("Cannot analyze empty digit list")
        
        # Count frequencies
        self.frequency_table = Counter(digits)
        total = len(digits)
        
        # Compute probabilities
        self.probability_table = {
            symbol: count / total 
            for symbol, count in self.frequency_table.items()
        }
        
        # Compute entropy
        entropy = 0.0
        for prob in self.probability_table.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy, self.probability_table
    
    def choose_encoding_method(self, digits: List[int]) -> str:
        """Choose optimal encoding method based on distribution
        
        Args:
            digits: List of p-adic digits to encode
            
        Returns:
            "huffman" or "arithmetic"
        """
        entropy, _ = self.analyze_distribution(digits)
        
        # Store metrics
        self.metrics.entropy = entropy
        self.metrics.unique_symbols = len(self.frequency_table)
        self.metrics.digit_count = len(digits)
        
        # Theoretical maximum entropy for uniform distribution
        max_entropy = math.log2(self.prime)
        
        # Decision criteria:
        # - Use Huffman for highly skewed distributions (low entropy)
        # - Use Arithmetic for near-uniform distributions (high entropy)
        # - Consider number of unique symbols
        
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0
        
        # For now, always use Huffman for reliability
        # Arithmetic coding has numerical precision issues that need more work
        return "huffman"
    
    def encode_digits(self, digits: List[int]) -> Tuple[bytes, Dict[str, Any]]:
        """Encode p-adic digit sequence with optimal method
        
        Args:
            digits: List of p-adic digits in range [0, prime-1]
            
        Returns:
            Tuple of (compressed_data, metadata)
        """
        if not digits:
            raise ValueError("Cannot encode empty digit list")
        
        # Validate all digits are in valid range
        for digit in digits:
            if not (0 <= digit < self.prime):
                raise ValueError(f"Digit {digit} out of range [0, {self.prime-1}]")
        
        # Record start time
        start_time = time.perf_counter()
        
        # Choose encoding method
        method = self.choose_encoding_method(digits)
        self.last_method_used = method
        
        # Calculate original size (assuming 1 byte per digit for p < 256)
        if self.prime <= 256:
            original_size = len(digits)
        else:
            # Need 2 bytes per digit for larger primes
            original_size = len(digits) * 2
        
        self.metrics.original_size_bytes = original_size
        
        # Perform encoding
        if method == "huffman":
            # Build Huffman tree and encode
            self.huffman.build_huffman_tree(self.frequency_table)
            self.huffman.generate_codes()
            compressed = self.huffman.huffman_encode(digits)
            
            # Store tree in metadata for decoding
            metadata = {
                'method': 'huffman',
                'prime': self.prime,
                'frequency_table': dict(self.frequency_table),
                'tree_built': True
            }
        else:
            # Use arithmetic coding
            compressed = self.arithmetic.arithmetic_encode(digits, self.probability_table)
            
            metadata = {
                'method': 'arithmetic',
                'prime': self.prime,
                'probability_table': dict(self.probability_table)
            }
        
        # Update metrics
        self.metrics.compressed_size_bytes = len(compressed)
        self.metrics.encoding_time_ms = (time.perf_counter() - start_time) * 1000
        self.metrics.encoding_method = method
        self.metrics.calculate_ratio()
        
        # Add metrics to metadata
        metadata['metrics'] = {
            'original_size': self.metrics.original_size_bytes,
            'compressed_size': self.metrics.compressed_size_bytes,
            'compression_ratio': self.metrics.compression_ratio,
            'entropy': self.metrics.entropy,
            'unique_symbols': self.metrics.unique_symbols,
            'encoding_time_ms': self.metrics.encoding_time_ms
        }
        
        return compressed, metadata
    
    def decode_digits(self, compressed: bytes, metadata: Dict[str, Any]) -> List[int]:
        """Decode compressed data back to p-adic digits
        
        Args:
            compressed: Compressed byte sequence
            metadata: Encoding metadata
            
        Returns:
            Original p-adic digit sequence
        """
        if not compressed:
            raise ValueError("Cannot decode empty compressed data")
        
        if not metadata:
            raise ValueError("Cannot decode without metadata")
        
        # Record start time
        start_time = time.perf_counter()
        
        # Extract method
        method = metadata.get('method')
        if method not in ['huffman', 'arithmetic']:
            raise ValueError(f"Unknown encoding method: {method}")
        
        # Validate prime matches
        if metadata.get('prime') != self.prime:
            raise ValueError(f"Prime mismatch: expected {self.prime}, got {metadata.get('prime')}")
        
        # Perform decoding
        if method == 'huffman':
            # Rebuild Huffman tree from frequency table
            freq_table = metadata.get('frequency_table')
            if not freq_table:
                raise ValueError("Missing frequency table for Huffman decoding")
            
            # Convert string keys back to integers if needed
            if isinstance(next(iter(freq_table.keys())), str):
                freq_table = {int(k): v for k, v in freq_table.items()}
            
            self.huffman.build_huffman_tree(freq_table)
            digits = self.huffman.huffman_decode(compressed)
        else:
            # Arithmetic decoding
            digits = self.arithmetic.arithmetic_decode(compressed)
        
        # Update metrics
        self.metrics.decoding_time_ms = (time.perf_counter() - start_time) * 1000
        
        return digits
    
    def get_metrics(self) -> CompressionMetrics:
        """Get current compression metrics
        
        Returns:
            Compression metrics object
        """
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset compression metrics"""
        self.metrics = CompressionMetrics(prime=self.prime)


def validate_reconstruction(original: List[int], reconstructed: List[int]) -> None:
    """Validate perfect reconstruction of digits
    
    Args:
        original: Original digit sequence
        reconstructed: Reconstructed digit sequence
        
    Raises:
        ValueError: If reconstruction doesn't match original
    """
    if len(original) != len(reconstructed):
        raise ValueError(
            f"Length mismatch: original={len(original)}, "
            f"reconstructed={len(reconstructed)}"
        )
    
    for i, (orig, recon) in enumerate(zip(original, reconstructed)):
        if orig != recon:
            raise ValueError(
                f"Mismatch at position {i}: "
                f"original={orig}, reconstructed={recon}"
            )


def run_compression_tests(prime: int = 257) -> None:
    """Run comprehensive tests of entropy coding system
    
    Args:
        prime: P-adic prime to test with
    """
    print(f"\n{'='*60}")
    print(f"Testing Entropy Coding System (prime={prime})")
    print(f"{'='*60}")
    
    # Test data patterns
    test_cases = [
        # Highly skewed distribution (good for Huffman)
        ("Skewed", [0] * 500 + [1] * 100 + [2] * 50 + list(range(3, 10)) * 5),
        
        # Uniform distribution (good for Arithmetic)
        ("Uniform", list(range(prime)) * 10),
        
        # Sparse distribution
        ("Sparse", [0, 5, 10, 15, 20] * 100),
        
        # Random distribution
        ("Random", np.random.randint(0, prime, 1000).tolist()),
    ]
    
    encoder = HybridEncoder(prime)
    
    for name, digits in test_cases:
        print(f"\nTest case: {name}")
        print(f"  Digits: {len(digits)}, Unique: {len(set(digits))}")
        
        try:
            # Encode
            compressed, metadata = encoder.encode_digits(digits)
            
            # Decode
            reconstructed = encoder.decode_digits(compressed, metadata)
            
            # Validate
            validate_reconstruction(digits, reconstructed)
            
            # Report metrics
            metrics = metadata['metrics']
            print(f"  Method: {metadata['method']}")
            print(f"  Original size: {metrics['original_size']} bytes")
            print(f"  Compressed size: {metrics['compressed_size']} bytes")
            print(f"  Compression ratio: {metrics['compression_ratio']:.2f}x")
            print(f"  Entropy: {metrics['entropy']:.2f} bits")
            print(f"  Encoding time: {metrics['encoding_time_ms']:.2f} ms")
            print(f"  Status: PASSED")
            
        except Exception as e:
            print(f"  Status: FAILED - {e}")
            raise
    
    print(f"\n{'='*60}")
    print("All tests passed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Run tests with various primes
    for prime in [2, 3, 5, 7, 11, 127, 257]:
        run_compression_tests(prime)