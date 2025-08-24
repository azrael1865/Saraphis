"""
Complete PyTorch-Only P-adic Compression Pipeline
NO NUMPY - PURE PYTORCH WITH TRITON ACCELERATION
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
import math
import time
import gc

# Import PyTorch P-adic components
from pytorch_padic_engine import PyTorchPAdicEngine, PyTorchPAdicConfig

# Try to import Triton operations (optional)
try:
    from .triton_kernels import TritonPAdicOps
    TRITON_AVAILABLE = True
except ImportError:
    TritonPAdicOps = None
    TRITON_AVAILABLE = False

from sliding_window_pattern_detector import SlidingWindowPatternDetector, PatternDetectionResult
from metadata_compressor import MetadataCompressor


@dataclass
class CompressionResult:
    """Result of compression operation"""
    compressed_data: torch.Tensor
    metadata: Dict[str, Any]
    compression_ratio: float
    encoding_time: float
    sparse_indices: Optional[torch.Tensor] = None
    sparse_values: Optional[torch.Tensor] = None
    original_shape: Optional[Tuple[int, ...]] = None
    signs: Optional[torch.Tensor] = None
    compressed_metadata: Optional[bytes] = None  # Compressed metadata bytes


@dataclass
class DecompressionResult:
    """Result of decompression operation"""
    reconstructed_data: torch.Tensor
    reconstruction_error: float
    decoding_time: float
    metadata: Dict[str, Any]


@dataclass 
class PurelyPyTorchConfig:
    """Configuration for purely PyTorch P-adic compression"""
    # P-adic parameters
    prime: int = 257
    precision: int = 6  # Conservative for stability
    
    # PyTorch optimization
    compile_mode: str = "reduce-overhead"
    enable_triton: bool = False  # Triton permanently disabled
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: torch.dtype = torch.float32
    
    # Compression parameters
    sparse_threshold: float = 0.01
    enable_sparse: bool = True
    enable_log_encoding: bool = True
    enable_pattern_matching: bool = True
    
    # Mixed precision
    enable_mixed_precision: bool = True
    autocast_dtype: torch.dtype = torch.float16
    
    # Memory management
    max_batch_size: int = 1024
    enable_gradient_checkpointing: bool = True
    
    # Entropy coding
    enable_entropy: bool = True
    entropy_bins: int = 256
    
    def __post_init__(self):
        """Validate configuration"""
        if not torch.cuda.is_available() and self.device == 'cuda':
            self.device = 'cpu'
            self.enable_triton = False  # Triton requires CUDA
        
        if self.precision > 10:
            raise ValueError(f"Precision {self.precision} too high for stable operation")


class SparseCSREncoder(nn.Module):
    """
    Native PyTorch sparse CSR encoder
    
    Handles conversion between dense and sparse CSR representations
    """
    
    def __init__(self, threshold: float = 0.01):
        super().__init__()
        self.threshold = threshold
        self.stats = {
            'total_encodings': 0,
            'average_sparsity': 0.0,
            'max_sparsity': 0.0
        }
    
    def encode(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
        """
        Encode dense tensor to sparse CSR format
        
        Args:
            tensor: Dense tensor
            
        Returns:
            Tuple of (indices, values, shape)
        """
        original_shape = tensor.shape
        flat = tensor.flatten()
        
        # Create mask for significant values
        mask = torch.abs(flat) > self.threshold
        
        # Extract indices and values
        indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        values = flat[mask]
        
        # Calculate sparsity
        sparsity = 1.0 - (values.numel() / flat.numel())
        
        # Update statistics
        self.stats['total_encodings'] += 1
        self.stats['average_sparsity'] = (
            self.stats['average_sparsity'] * (self.stats['total_encodings'] - 1) + sparsity
        ) / self.stats['total_encodings']
        self.stats['max_sparsity'] = max(self.stats['max_sparsity'], sparsity)
        
        return indices, values, original_shape
    
    def decode(self, indices: torch.Tensor, values: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Decode sparse CSR to dense tensor
        
        Args:
            indices: Sparse indices
            values: Sparse values
            shape: Original tensor shape
            
        Returns:
            Dense tensor
        """
        # Calculate total size
        total_size = 1
        for dim in shape:
            total_size *= dim
        
        # Create dense tensor
        dense = torch.zeros(total_size, device=values.device, dtype=values.dtype)
        
        # Fill in sparse values
        if indices.numel() > 0:
            dense[indices] = values
        
        # Reshape to original shape
        return dense.reshape(shape)


class PyTorchPatternMatcher(nn.Module):
    """
    Pattern detection and matching for PyTorch tensors
    
    Identifies repeated patterns for better compression
    """
    
    def __init__(self, min_pattern_length: int = 4, max_pattern_length: int = 32):
        super().__init__()
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.pattern_cache = {}
    
    @torch.compile(mode="reduce-overhead")
    def find_patterns(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Find repeated patterns in tensor
        
        Args:
            tensor: Input tensor
            
        Returns:
            Dictionary with pattern information
        """
        flat = tensor.flatten()
        n = flat.numel()
        
        patterns = {}
        best_pattern = None
        best_score = 0
        
        # Try different pattern lengths
        for pattern_len in range(self.min_pattern_length, min(self.max_pattern_length + 1, n // 2)):
            # Check if tensor can be divided into this pattern length
            if n % pattern_len != 0:
                continue
            
            # Reshape to check for repeated patterns
            reshaped = flat[:n - n % pattern_len].reshape(-1, pattern_len)
            
            # Check if all rows are similar
            first_pattern = reshaped[0]
            diffs = torch.abs(reshaped - first_pattern.unsqueeze(0))
            max_diff = diffs.max().item()
            
            if max_diff < 0.01:  # Threshold for pattern similarity
                # Calculate compression score
                score = (n // pattern_len) / pattern_len
                if score > best_score:
                    best_score = score
                    best_pattern = {
                        'pattern': first_pattern,
                        'length': pattern_len,
                        'repetitions': n // pattern_len,
                        'max_diff': max_diff
                    }
        
        return best_pattern if best_pattern else {}
    
    def encode_with_patterns(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode tensor using pattern detection
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tuple of (encoded_tensor, pattern_metadata)
        """
        pattern_info = self.find_patterns(tensor)
        
        if pattern_info:
            # Store only the pattern and metadata
            encoded = pattern_info['pattern']
            metadata = {
                'has_pattern': True,
                'pattern_length': pattern_info['length'],
                'repetitions': pattern_info['repetitions'],
                'original_shape': tensor.shape
            }
        else:
            # No pattern found, store as-is
            encoded = tensor
            metadata = {
                'has_pattern': False,
                'original_shape': tensor.shape
            }
        
        return encoded, metadata
    
    def decode_with_patterns(self, encoded: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Decode tensor using pattern information
        
        Args:
            encoded: Encoded tensor
            metadata: Pattern metadata
            
        Returns:
            Reconstructed tensor
        """
        if metadata.get('has_pattern', False):
            # Reconstruct from pattern
            pattern = encoded
            repetitions = metadata['repetitions']
            reconstructed = pattern.repeat(repetitions)
            return reconstructed.reshape(metadata['original_shape'])
        else:
            # No pattern, return as-is
            return encoded.reshape(metadata['original_shape'])


class PyTorchEntropyEncoder(nn.Module):
    """
    Entropy encoding for PyTorch tensors
    
    Simple histogram-based entropy coding
    """
    
    def __init__(self, n_bins: int = 256):
        super().__init__()
        self.n_bins = n_bins
    
    def encode(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Entropy encode tensor
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tuple of (encoded_tensor, encoding_metadata)
        """
        # Compute histogram for probability estimation
        min_val = tensor.min()
        max_val = tensor.max()
        
        # Quantize to bins
        if max_val > min_val:
            normalized = (tensor - min_val) / (max_val - min_val)
            quantized = (normalized * (self.n_bins - 1)).round().long()
        else:
            quantized = torch.zeros_like(tensor, dtype=torch.long)
        
        # Compute histogram
        hist = torch.histc(quantized.float(), bins=self.n_bins, min=0, max=self.n_bins-1)
        probabilities = hist / hist.sum()
        
        # Calculate entropy
        entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-10))
        
        metadata = {
            'min_val': min_val.item(),
            'max_val': max_val.item(),
            'entropy': entropy.item(),
            'n_bins': self.n_bins
        }
        
        return quantized, metadata
    
    def decode(self, quantized: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Decode entropy encoded tensor
        
        Args:
            quantized: Quantized tensor
            metadata: Encoding metadata
            
        Returns:
            Reconstructed tensor
        """
        min_val = metadata['min_val']
        max_val = metadata['max_val']
        n_bins = metadata['n_bins']
        
        # Dequantize
        normalized = quantized.float() / (n_bins - 1)
        reconstructed = normalized * (max_val - min_val) + min_val
        
        return reconstructed


class PurelyPyTorchPAdicSystem(nn.Module):
    """
    Complete PyTorch-only P-adic compression system
    
    Features:
    - No NumPy dependencies
    - torch.compile optimization
    - Triton kernel acceleration
    - Native sparse CSR support
    - Pattern matching
    - Entropy coding
    """
    
    def __init__(self, config: Optional[PurelyPyTorchConfig] = None):
        """Initialize PyTorch-only P-adic system"""
        super().__init__()
        
        self.config = config or PurelyPyTorchConfig()
        
        # Initialize P-adic engine
        padic_config = PyTorchPAdicConfig(
            prime=self.config.prime,
            precision=self.config.precision,
            device=self.config.device,
            dtype=self.config.dtype,
            enable_mixed_precision=self.config.enable_mixed_precision,
            compile_mode=self.config.compile_mode,
            enable_triton=self.config.enable_triton,
            sparse_threshold=self.config.sparse_threshold
        )
        self.padic_engine = PyTorchPAdicEngine(
            prime=self.config.prime,
            precision=self.config.precision,
            config=padic_config
        )
        
        # Initialize Triton operations if available
        if self.config.enable_triton and torch.cuda.is_available() and TRITON_AVAILABLE:
            self.triton_ops = TritonPAdicOps(
                prime=self.config.prime,
                precision=self.config.precision,
                device=self.config.device
            )
        else:
            self.triton_ops = None
            if self.config.enable_triton and not TRITON_AVAILABLE:
                print("Warning: Triton requested but not available. Using CPU fallback.")
        
        # Initialize components
        self.sparse_encoder = SparseCSREncoder(self.config.sparse_threshold)
        self.pattern_matcher = PyTorchPatternMatcher()
        # Initialize sliding window pattern detector for advanced pattern detection
        self.sliding_pattern_detector = SlidingWindowPatternDetector(
            min_pattern_length=4,
            max_pattern_length=32,
            min_frequency=3,
            hash_prime=31,
            device=self.config.device,
            enable_compile=self.config.compile_mode != 'default'
        )
        self.entropy_encoder = PyTorchEntropyEncoder(self.config.entropy_bins)
        
        # Initialize metadata compressor for < 1% overhead
        self.metadata_compressor = MetadataCompressor()
        
        # Mixed precision support
        if self.config.enable_mixed_precision and torch.cuda.is_available():
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Statistics
        self.stats = {
            'total_compressions': 0,
            'total_decompressions': 0,
            'average_compression_ratio': 0.0,
            'average_reconstruction_error': 0.0,
            'total_compression_time': 0.0,
            'total_decompression_time': 0.0
        }
    
    def compress(self, tensor: torch.Tensor) -> CompressionResult:
        """
        Compress tensor using P-adic encoding
        
        Args:
            tensor: Input tensor to compress
            
        Returns:
            CompressionResult with compressed data and metadata
        """
        start_time = time.time()
        original_shape = tensor.shape
        
        # Move to correct device
        if tensor.device != self.padic_engine.device:
            tensor = tensor.to(self.padic_engine.device)
        
        # Pattern matching - use sliding window detector for p-adic digits
        pattern_metadata = {}
        sliding_pattern_result = None
        if self.config.enable_pattern_matching:
            # First try basic pattern matching
            tensor, pattern_metadata = self.pattern_matcher.encode_with_patterns(tensor)
        
        # Mixed precision context
        if self.config.enable_mixed_precision and torch.cuda.is_available():
            with autocast(dtype=self.config.autocast_dtype):
                # Log-space encoding for better dynamic range
                if self.config.enable_log_encoding:
                    if self.triton_ops:
                        encoded = self.triton_ops.log_encode(tensor)
                        # Convert to p-adic
                        padic_digits = self.triton_ops.batch_convert(encoded)
                    else:
                        # Manual log-space encoding since method not available
                        epsilon = 1e-10
                        x_safe = torch.where(torch.abs(tensor) < epsilon, epsilon, torch.abs(tensor))
                        log_x = torch.log(x_safe)
                        padic_digits, signs = self.padic_engine.to_padic(log_x), torch.sign(tensor)
                else:
                    # Direct p-adic conversion
                    if self.triton_ops:
                        padic_digits = self.triton_ops.batch_convert(tensor)
                        signs = torch.sign(tensor)
                    else:
                        padic_digits = self.padic_engine.to_padic(tensor)
                        signs = torch.sign(tensor)
        else:
            # Standard precision
            if self.config.enable_log_encoding:
                # Manual log-space encoding
                epsilon = 1e-10
                x_safe = torch.where(torch.abs(tensor) < epsilon, epsilon, torch.abs(tensor))
                log_x = torch.log(x_safe)
                padic_digits = self.padic_engine.to_padic(log_x)
                signs = torch.sign(tensor)
            else:
                padic_digits = self.padic_engine.to_padic(tensor)
                signs = torch.sign(tensor)
        
        # Apply sliding window pattern detection on p-adic digits
        sliding_pattern_data = None
        sliding_pattern_dict = None
        if self.config.enable_pattern_matching and padic_digits is not None:
            # Convert p-adic digits to bytes for pattern detection
            padic_bytes = padic_digits.flatten().to(torch.uint8)
            sliding_pattern_result = self.sliding_pattern_detector.find_patterns(padic_bytes)
            
            if sliding_pattern_result.total_patterns_found > 0:
                # Encode with detected patterns
                encoded_data, pattern_dict, pattern_lengths = self.sliding_pattern_detector.encode_with_patterns(
                    padic_bytes, sliding_pattern_result
                )
                sliding_pattern_data = encoded_data
                sliding_pattern_dict = pattern_dict
                pattern_metadata['sliding_patterns'] = {
                    'num_patterns': sliding_pattern_result.total_patterns_found,
                    'compression_potential': sliding_pattern_result.compression_potential,
                    'bytes_replaced': sliding_pattern_result.bytes_replaced,
                    'pattern_lengths': pattern_lengths.cpu().tolist() if pattern_lengths.numel() > 0 else []
                }
        
        # Sparse encoding
        sparse_indices = None
        sparse_values = None
        if self.config.enable_sparse:
            if self.triton_ops:
                sparse_indices, sparse_values = self.triton_ops.sparse_encode(
                    padic_digits, self.config.sparse_threshold
                )
            else:
                # Use our sparse encoder directly
                sparse_indices, sparse_values, _ = self.sparse_encoder.encode(padic_digits)
        
        # Entropy encoding
        entropy_metadata = {}
        if self.config.enable_entropy:
            if sparse_values is not None:
                sparse_values, entropy_metadata = self.entropy_encoder.encode(sparse_values)
            else:
                padic_digits, entropy_metadata = self.entropy_encoder.encode(padic_digits)
        
        # Calculate compression ratio
        original_size = tensor.numel() * 4  # Assuming float32
        if sparse_values is not None:
            compressed_size = sparse_values.numel() * 4 + sparse_indices.numel() * 8
        else:
            compressed_size = padic_digits.numel() * 4
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        # Update statistics
        self.stats['total_compressions'] += 1
        compression_time = time.time() - start_time
        self.stats['total_compression_time'] += compression_time
        self.stats['average_compression_ratio'] = (
            self.stats['average_compression_ratio'] * (self.stats['total_compressions'] - 1) + compression_ratio
        ) / self.stats['total_compressions']
        
        # Prepare metadata
        metadata = {
            'prime': self.config.prime,
            'precision': self.config.precision,
            'original_shape': original_shape,
            'pattern_metadata': pattern_metadata,
            'entropy_metadata': entropy_metadata,
            'sparse_encoded': sparse_values is not None,
            'log_encoded': self.config.enable_log_encoding,
            'compression_method': 'pytorch_padic',
            'triton_accelerated': self.triton_ops is not None,
            'sliding_pattern_encoded': sliding_pattern_data is not None,
            'sliding_pattern_dict': sliding_pattern_dict,
            'sparse_indices': sparse_indices,
            'sparse_values': sparse_values,
            'signs': signs if 'signs' in locals() else None,
            'mixed_precision': self.config.enable_mixed_precision,
            'device': str(self.config.device)
        }
        
        # Compress metadata to bytes for < 1% overhead
        compressed_metadata = self.metadata_compressor.compress_metadata(metadata)
        
        # Update metadata overhead statistics
        self.metadata_compressor.stats['total_data_bytes'] += compressed_size
        
        # Return result - use sliding pattern data if available
        if sliding_pattern_data is not None:
            compressed_data = sliding_pattern_data
        elif sparse_values is not None:
            compressed_data = sparse_values
        else:
            compressed_data = padic_digits
            
        return CompressionResult(
            compressed_data=compressed_data,
            metadata=metadata,
            compression_ratio=compression_ratio,
            encoding_time=compression_time,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            original_shape=original_shape,
            signs=signs if 'signs' in locals() else None,
            compressed_metadata=compressed_metadata
        )
    
    def decompress(self, result: CompressionResult) -> DecompressionResult:
        """
        Decompress P-adic encoded data
        
        Args:
            result: CompressionResult from compress()
            
        Returns:
            DecompressionResult with reconstructed tensor
        """
        start_time = time.time()
        
        # Decompress metadata if compressed bytes are available
        if hasattr(result, 'compressed_metadata') and result.compressed_metadata is not None:
            metadata = self.metadata_compressor.decompress_metadata(result.compressed_metadata)
            # Restore tensor types from decompressed metadata
            if 'sparse_indices' in metadata and metadata['sparse_indices'] is not None:
                result.sparse_indices = metadata['sparse_indices']
            if 'sparse_values' in metadata and metadata['sparse_values'] is not None:
                result.sparse_values = metadata['sparse_values']
            if 'signs' in metadata and metadata['signs'] is not None:
                result.signs = metadata['signs']
        else:
            metadata = result.metadata
        
        # Handle sliding pattern decoding first if present
        if metadata.get('sliding_pattern_encoded', False) and metadata.get('sliding_pattern_dict'):
            # Decode sliding window patterns
            pattern_dict = metadata['sliding_pattern_dict']
            pattern_lengths = metadata.get('pattern_metadata', {}).get('sliding_patterns', {}).get('pattern_lengths', [])
            pattern_lengths_tensor = torch.tensor(pattern_lengths, dtype=torch.int32, device=self.padic_engine.device)
            
            # Decode patterns to get p-adic bytes
            padic_bytes = self.sliding_pattern_detector.decode_with_patterns(
                result.compressed_data,
                pattern_dict,
                pattern_lengths_tensor
            )
            
            # Convert bytes back to p-adic digits with proper shape
            padic_digits = padic_bytes.to(torch.float32).reshape(*metadata['original_shape'], metadata['precision'])
        # Reconstruct p-adic digits
        elif result.sparse_values is not None:
            # Decode from sparse
            if metadata.get('entropy_metadata'):
                sparse_values = self.entropy_encoder.decode(
                    result.sparse_values, metadata['entropy_metadata']
                )
            else:
                sparse_values = result.sparse_values
            
            # Reconstruct dense p-adic digits
            total_size = 1
            for dim in metadata['original_shape']:
                total_size *= dim
            total_size *= metadata['precision']
            
            padic_digits = self.sparse_encoder.decode(
                result.sparse_indices, sparse_values, (total_size,)
            )
            padic_digits = padic_digits.reshape(*metadata['original_shape'], metadata['precision'])
        else:
            # Direct p-adic digits
            padic_digits = result.compressed_data
            if metadata.get('entropy_metadata'):
                padic_digits = self.entropy_encoder.decode(
                    padic_digits, metadata['entropy_metadata']
                )
        
        # Decode from p-adic
        if metadata.get('log_encoded', False):
            # Manual log-space decoding
            log_values = self.padic_engine.from_padic(padic_digits)
            reconstructed = torch.exp(log_values)
            # Apply signs if available
            if hasattr(result, 'signs') and result.signs is not None:
                reconstructed = reconstructed * result.signs
        else:
            reconstructed = self.padic_engine.from_padic(padic_digits)
            # Apply signs if available
            if hasattr(result, 'signs') and result.signs is not None:
                reconstructed = reconstructed * result.signs
        
        # Decode patterns
        if metadata.get('pattern_metadata', {}).get('has_pattern', False):
            reconstructed = self.pattern_matcher.decode_with_patterns(
                reconstructed, metadata['pattern_metadata']
            )
        
        # Reshape to original shape
        if reconstructed.shape != metadata['original_shape']:
            reconstructed = reconstructed.reshape(metadata['original_shape'])
        
        # Calculate reconstruction error (placeholder - requires original)
        reconstruction_error = 0.0  # Would need original tensor for actual error
        
        # Update statistics
        self.stats['total_decompressions'] += 1
        decompression_time = time.time() - start_time
        self.stats['total_decompression_time'] += decompression_time
        
        return DecompressionResult(
            reconstructed_data=reconstructed,
            reconstruction_error=reconstruction_error,
            decoding_time=decompression_time,
            metadata=metadata
        )
    
    def validate_compression(self, original: torch.Tensor, tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Validate compression by comparing original and reconstructed
        
        Args:
            original: Original tensor
            tolerance: Error tolerance
            
        Returns:
            Validation results
        """
        # Compress
        compressed = self.compress(original)
        
        # Decompress
        decompressed = self.decompress(compressed)
        reconstructed = decompressed.reconstructed_data
        
        # Move to same device for comparison
        if original.device != reconstructed.device:
            reconstructed = reconstructed.to(original.device)
        
        # Calculate errors
        abs_error = torch.abs(original - reconstructed)
        rel_error = abs_error / (torch.abs(original) + 1e-10)
        
        max_abs_error = abs_error.max().item()
        mean_abs_error = abs_error.mean().item()
        max_rel_error = rel_error.max().item()
        mean_rel_error = rel_error.mean().item()
        
        # Check if within tolerance
        is_valid = max_abs_error < tolerance
        
        return {
            'is_valid': is_valid,
            'max_abs_error': max_abs_error,
            'mean_abs_error': mean_abs_error,
            'max_rel_error': max_rel_error,
            'mean_rel_error': mean_rel_error,
            'compression_ratio': compressed.compression_ratio,
            'compression_time': compressed.encoding_time,
            'decompression_time': decompressed.decoding_time,
            'triton_accelerated': self.triton_ops is not None
        }
    
    def benchmark(self, tensor_sizes: List[Tuple[int, ...]], n_iterations: int = 10) -> Dict[str, Any]:
        """
        Benchmark compression performance
        
        Args:
            tensor_sizes: List of tensor sizes to test
            n_iterations: Number of iterations per size
            
        Returns:
            Benchmark results
        """
        results = []
        
        for size in tensor_sizes:
            size_results = {
                'size': size,
                'num_elements': math.prod(size),
                'compression_times': [],
                'decompression_times': [],
                'compression_ratios': []
            }
            
            for _ in range(n_iterations):
                # Generate random tensor
                tensor = torch.randn(size, device=self.padic_engine.device)
                
                # Compress
                compressed = self.compress(tensor)
                size_results['compression_times'].append(compressed.encoding_time)
                size_results['compression_ratios'].append(compressed.compression_ratio)
                
                # Decompress
                decompressed = self.decompress(compressed)
                size_results['decompression_times'].append(decompressed.decoding_time)
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate averages
            size_results['avg_compression_time'] = sum(size_results['compression_times']) / n_iterations
            size_results['avg_decompression_time'] = sum(size_results['decompression_times']) / n_iterations
            size_results['avg_compression_ratio'] = sum(size_results['compression_ratios']) / n_iterations
            
            results.append(size_results)
        
        return {
            'results': results,
            'config': {
                'prime': self.config.prime,
                'precision': self.config.precision,
                'device': self.config.device,
                'triton_enabled': self.triton_ops is not None,
                'compile_mode': self.config.compile_mode
            },
            'statistics': self.get_statistics()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get compression statistics"""
        stats = dict(self.stats)
        stats['padic_engine_stats'] = self.padic_engine.get_statistics()
        stats['sparse_encoder_stats'] = self.sparse_encoder.stats
        stats['metadata_compressor_stats'] = self.metadata_compressor.get_statistics()
        
        # Calculate metadata overhead percentage
        if self.metadata_compressor.stats['total_data_bytes'] > 0:
            metadata_overhead = (
                self.metadata_compressor.stats['total_metadata_bytes'] / 
                self.metadata_compressor.stats['total_data_bytes']
            ) * 100
            stats['metadata_overhead_percentage'] = metadata_overhead
        
        return stats
    
    def reset_statistics(self):
        """Reset all statistics"""
        for key in self.stats:
            if isinstance(self.stats[key], (int, float)):
                self.stats[key] = 0.0 if isinstance(self.stats[key], float) else 0
        self.padic_engine.reset_statistics()
        self.sparse_encoder.stats = {
            'total_encodings': 0,
            'average_sparsity': 0.0,
            'max_sparsity': 0.0
        }
        self.metadata_compressor.reset_statistics()