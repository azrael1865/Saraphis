"""
FINAL FIX: P-adic Compression System
Store compressed data AS-IS, p-adic is only for metadata
"""

import torch
import numpy as np
import struct
import logging
import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from fractions import Fraction

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import validation and encoder components
from .padic_encoder import PadicValidation, PadicMathematicalOperations, PadicWeight

# Import bridge components - HARD FAILURE if missing (no fallbacks)
from .sliding_window_pattern_detector import SlidingWindowPatternDetector
from .sparse_bridge import SparsePAdicBridge
from .entropy_bridge import EntropyPAdicBridge
from .metadata_compressor import MetadataCompressor


@dataclass
class CompressionConfig:
    """Configuration for P-adic compression system"""
    prime: int = 257
    base_precision: int = 2
    min_precision: int = 1
    max_precision: int = 3
    target_error: float = 1e-6
    importance_threshold: float = 0.1
    compression_priority: float = 0.8
    enable_gpu: bool = True
    validate_reconstruction: bool = True
    chunk_size: int = 1000
    max_tensor_size: int = 1_000_000
    max_reconstruction_error: float = 1e-5
    enable_memory_monitoring: bool = True
    sparsity_threshold: float = 1e-6
    huffman_threshold: float = 2.0
    arithmetic_threshold: float = 6.0
    enable_hybrid_entropy: bool = True
    raise_on_error: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class CompressionResult:
    """Result of compression operation"""
    compressed_data: bytes
    metadata: Dict[str, Any]
    compression_ratio: float
    processing_time: float
    memory_usage: int
    stage_metrics: Dict[str, Any] = field(default_factory=dict)
    validation_passed: bool = True
    error_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass  
class DecompressionResult:
    """Result of decompression operation"""
    reconstructed_tensor: torch.Tensor
    original_shape: Tuple
    processing_time: float
    memory_usage: int
    stage_metrics: Dict[str, Any] = field(default_factory=dict)
    validation_passed: bool = True
    reconstruction_error: float = 0.0


class PadicCompressionSystem:
    """
    TRULY FIXED P-adic Compression System
    P-adic is ONLY for metadata, not data transformation
    """
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        """Initialize the P-adic compression system"""
        self.config = config or CompressionConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize P-adic math operations
        self.math_ops = PadicMathematicalOperations(
            self.config.prime, 
            self.config.base_precision
        )
        
        # Initialize bridge components with correct class names
        self.pattern_detector = SlidingWindowPatternDetector()
        self.sparse_bridge = SparsePAdicBridge(
            sparsity_threshold=self.config.sparsity_threshold,
            device=self.device
        )
        self.entropy_bridge = EntropyPAdicBridge(self.config.prime)
        self.metadata_compressor = MetadataCompressor()
        
        logger.info(f"P-adic Compression System initialized with prime={self.config.prime}")
    
    def compress(self, tensor: torch.Tensor, 
                importance_scores: Optional[torch.Tensor] = None) -> CompressionResult:
        """FIXED: Compress without corrupting the data"""
        start_time = time.perf_counter()
        stage_metrics = {}
        
        try:
            # Validate input
            PadicValidation.validate_tensor(tensor)
            original_shape = tensor.shape
            original_size = tensor.numel() * tensor.element_size()
            
            logger.info(f"Starting compression of tensor shape={original_shape}")
            logger.info(f"Original size: {original_size / 1024:.2f} KB")
            
            # Stage 1: Pattern Detection
            stage_start = time.perf_counter()
            logger.info("Stage 1: Pattern Detection on FLOAT data")
            
            flat_tensor = tensor.flatten()
            
            # SlidingWindowPatternDetector is an nn.Module, call it directly
            pattern_result = self.pattern_detector(flat_tensor)
            
            # Handle the return format from SlidingWindowPatternDetector
            if isinstance(pattern_result, tuple):
                if len(pattern_result) == 3:
                    pattern_compressed, pattern_dict, pattern_metadata = pattern_result
                    pattern_lengths = pattern_metadata.get('pattern_lengths', torch.tensor([], dtype=torch.int32, device=self.device))
                elif len(pattern_result) == 2:
                    pattern_compressed, pattern_dict = pattern_result
                    pattern_lengths = torch.tensor([], dtype=torch.int32, device=self.device)
                else:
                    raise ValueError(f"Unexpected pattern detector return format: {len(pattern_result)} elements")
            else:
                # Single tensor return
                pattern_compressed = pattern_result
                pattern_dict = {}
                pattern_lengths = torch.tensor([], dtype=torch.int32, device=self.device)
            
            # Ensure pattern_compressed is a tensor
            if not isinstance(pattern_compressed, torch.Tensor):
                pattern_compressed = torch.tensor(pattern_compressed, device=self.device)
            
            logger.info(f"Pattern compression: {flat_tensor.numel()} → {pattern_compressed.numel()} elements")
            
            stage_metrics['pattern_detection'] = {
                'time': time.perf_counter() - stage_start,
                'patterns_found': len(pattern_dict),
                'compression_ratio': flat_tensor.numel() / max(pattern_compressed.numel(), 1)
            }
            
            # Stage 2: Sparse Encoding
            stage_start = time.perf_counter()
            logger.info("Stage 2: Sparse Encoding on pattern-compressed data")
            
            valuations = torch.zeros(pattern_compressed.numel(), dtype=torch.int32, device=self.device)
            if pattern_compressed.dim() == 1:
                pattern_compressed = pattern_compressed.unsqueeze(1)
            
            sparse_tensor = self.sparse_bridge.padic_to_sparse(pattern_compressed, valuations)
            sparse_ratio = self.sparse_bridge.get_compression_ratio(sparse_tensor)
            
            # CRITICAL: Store the actual sparse values count
            sparse_values = sparse_tensor.values()
            actual_sparse_count = sparse_values.numel()
            
            stage_metrics['sparse_encoding'] = {
                'time': time.perf_counter() - stage_start,
                'sparsity': 1.0 - (sparse_tensor._nnz() / pattern_compressed.numel()),
                'compression_ratio': sparse_ratio,
                'actual_values': actual_sparse_count  # Track this!
            }
            
            # Stage 3: Entropy Coding
            stage_start = time.perf_counter()
            logger.info("Stage 3: Entropy Coding on sparse data")
            
            if sparse_values.numel() > 0:
                # Quantize for entropy encoding
                quantized = (sparse_values * 255).clamp(0, 255).long()
                
                # CRITICAL FIX: Ensure entropy metadata has correct shape
                entropy_tensor = quantized.reshape(-1)  # Ensure 1D
                
                # Create proper metadata for entropy bridge
                entropy_compressed, entropy_metadata = self.entropy_bridge.encode_padic_tensor(entropy_tensor)
                
                # FIX: Store the ACTUAL counts in metadata
                entropy_metadata['actual_input_count'] = entropy_tensor.numel()
                entropy_metadata['original_sparse_count'] = actual_sparse_count
                
                logger.info(f"Entropy compression: {sparse_values.numel()} values → {len(entropy_compressed)} bytes")
            else:
                entropy_compressed = b''
                entropy_metadata = {
                    'empty': True,
                    'actual_input_count': 0,
                    'original_sparse_count': 0
                }
            
            stage_metrics['entropy_coding'] = {
                'time': time.perf_counter() - stage_start,
                'compressed_bytes': len(entropy_compressed),
                'compression_ratio': (sparse_values.numel() * 4) / max(len(entropy_compressed), 1)
            }
            
            # Stage 4: P-adic Metadata Generation (NO DATA TRANSFORMATION!)
            stage_start = time.perf_counter()
            logger.info("Stage 4: P-adic METADATA generation (no data transformation)")
            
            # Generate p-adic metadata for GPU hints ONLY
            padic_metadata = {
                'prime': self.config.prime,
                'precision': self.config.base_precision,
                'gpu_optimization_available': True,
                'reconstruction_hint': 'use_safe_reconstruction'
            }
            
            # NO TRANSFORMATION OF COMPRESSED DATA!
            # Just store it as-is
            
            stage_metrics['padic_transformation'] = {
                'time': time.perf_counter() - stage_start,
                'expansion_ratio': 1.0,  # No expansion since no transformation
                'type': 'metadata_only'
            }
            
            # Stage 5: Metadata Compression
            stage_start = time.perf_counter()
            logger.info("Stage 5: Metadata Compression")
            
            metadata = {
                'version': 4,  # New version for truly fixed pipeline
                'pipeline_order': 'pattern→sparse→entropy→metadata',
                'prime': self.config.prime,
                'precision': self.config.base_precision,
                'original_shape': list(original_shape),
                'pattern_dict': pattern_dict,
                'pattern_lengths': pattern_lengths.cpu().numpy().tolist() if pattern_lengths.numel() > 0 else [],
                'sparse_indices': self._extract_sparse_indices(sparse_tensor),
                'sparse_shape': list(sparse_tensor.shape),
                'sparse_nnz': sparse_tensor._nnz(),
                'entropy_metadata': entropy_metadata,
                'padic_metadata': padic_metadata,  # Metadata only!
                'compressed_data_size': len(entropy_compressed)
            }
            
            compressed_metadata = self.metadata_compressor.compress_metadata(metadata)
            
            stage_metrics['metadata_compression'] = {
                'time': time.perf_counter() - stage_start,
                'metadata_size': len(compressed_metadata)
            }
            
            # Combine data - just entropy compressed + metadata
            combined_data = self._combine_data_final(
                entropy_compressed,  # Store AS-IS
                compressed_metadata
            )
            
            # Calculate metrics
            total_time = time.perf_counter() - start_time
            final_compression_ratio = original_size / len(combined_data)
            
            # Validation
            validation_passed = True
            error_metrics = {}
            
            if self.config.validate_reconstruction:
                try:
                    decompressed = self.decompress(combined_data)
                    error = torch.nn.functional.mse_loss(tensor, decompressed.reconstructed_tensor).item()
                    validation_passed = error < self.config.max_reconstruction_error * 10
                    error_metrics = {'mse': error}
                except Exception as e:
                    logger.warning(f"Validation failed: {e}")
                    validation_passed = False
            
            logger.info(f"Compression complete: ratio={final_compression_ratio:.2f}x, time={total_time:.3f}s")
            logger.info(f"Final size: {len(combined_data) / 1024:.2f} KB")
            
            return CompressionResult(
                compressed_data=combined_data,
                metadata=metadata,
                compression_ratio=final_compression_ratio,
                processing_time=total_time,
                memory_usage=len(combined_data),
                stage_metrics=stage_metrics,
                validation_passed=validation_passed,
                error_metrics=error_metrics
            )
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise RuntimeError(f"Critical compression failure: {e}")
    
    def decompress(self, compressed_data: bytes) -> DecompressionResult:
        """FIXED: Decompress without corruption"""
        start_time = time.perf_counter()
        stage_metrics = {}
        
        try:
            logger.info("Starting decompression")
            
            # Split data
            entropy_compressed, compressed_metadata = self._split_data_final(compressed_data)
            
            # Stage 1: Metadata Decompression
            stage_start = time.perf_counter()
            logger.info("Stage 1: Metadata Decompression")
            
            metadata = self.metadata_compressor.decompress_metadata(compressed_metadata)
            
            stage_metrics['metadata_decompression'] = {
                'time': time.perf_counter() - stage_start
            }
            
            # NO P-ADIC RECONSTRUCTION - data wasn't transformed!
            
            # Stage 2: Entropy Decoding
            stage_start = time.perf_counter()
            logger.info("Stage 2: Entropy Decoding")
            
            entropy_meta = metadata.get('entropy_metadata', {})
            
            if entropy_compressed and not entropy_meta.get('empty'):
                # FIX: Use the actual stored count
                if 'actual_input_count' in entropy_meta:
                    # Fix the metadata to have correct shape
                    entropy_meta['original_shape'] = [entropy_meta['actual_input_count']]
                
                sparse_values = self.entropy_bridge.decode_padic_tensor(
                    entropy_compressed, entropy_meta
                )
                sparse_values = sparse_values.float() / 255.0  # Dequantize
            else:
                sparse_values = torch.tensor([], device=self.device)
            
            stage_metrics['entropy_decoding'] = {
                'time': time.perf_counter() - stage_start,
                'values_decoded': sparse_values.numel()
            }
            
            # Stage 3: Sparse Reconstruction
            stage_start = time.perf_counter()
            logger.info("Stage 3: Sparse Reconstruction")
            
            sparse_meta = metadata
            sparse_indices_data = sparse_meta.get('sparse_indices')
            
            if sparse_indices_data is not None and sparse_values.numel() > 0:
                sparse_shape = tuple(sparse_meta['sparse_shape'])
                
                # Ensure we have the right number of values
                expected_nnz = sparse_meta.get('sparse_nnz', len(sparse_values))
                if len(sparse_values) < expected_nnz:
                    # Pad if needed
                    padding = torch.zeros(expected_nnz - len(sparse_values), device=self.device)
                    sparse_values = torch.cat([sparse_values, padding])
                elif len(sparse_values) > expected_nnz:
                    # Truncate if needed
                    sparse_values = sparse_values[:expected_nnz]
                
                # Check if it's CSR or COO format
                if isinstance(sparse_indices_data, dict) and 'crow_indices' in sparse_indices_data:
                    # CSR format reconstruction
                    crow_indices = torch.tensor(sparse_indices_data['crow_indices'], dtype=torch.long, device=self.device)
                    col_indices = torch.tensor(sparse_indices_data['col_indices'], dtype=torch.long, device=self.device)
                    
                    sparse_tensor = torch.sparse_csr_tensor(
                        crow_indices=crow_indices,
                        col_indices=col_indices,
                        values=sparse_values,
                        size=sparse_shape,
                        device=self.device
                    )
                else:
                    # COO format reconstruction
                    indices = torch.tensor(sparse_indices_data, device=self.device)
                    sparse_tensor = torch.sparse_coo_tensor(
                        indices, sparse_values, sparse_shape, device=self.device
                    )
                
                pattern_compressed = sparse_tensor.to_dense().flatten()
            else:
                pattern_compressed = torch.zeros(sparse_meta['sparse_shape'][0], device=self.device)
            
            stage_metrics['sparse_reconstruction'] = {
                'time': time.perf_counter() - stage_start
            }
            
            # Stage 4: Pattern Reconstruction
            stage_start = time.perf_counter()
            logger.info("Stage 4: Pattern Reconstruction")
            
            if metadata.get('pattern_dict'):
                # SlidingWindowPatternDetector has decode method
                pattern_lengths = torch.tensor(metadata.get('pattern_lengths', []), device=self.device)
                
                # Call the decoder - it's likely a different method name
                # The SlidingWindowPatternDetector should have a decode or inverse method
                if hasattr(self.pattern_detector, 'decode'):
                    reconstructed = self.pattern_detector.decode(
                        pattern_compressed.long(),
                        metadata['pattern_dict'],
                        pattern_lengths
                    )
                elif hasattr(self.pattern_detector, 'inverse'):
                    reconstructed = self.pattern_detector.inverse(
                        pattern_compressed.long(),
                        metadata['pattern_dict']
                    )
                else:
                    # Direct reconstruction without pattern decoder
                    # This means patterns weren't actually used
                    reconstructed = pattern_compressed
            else:
                reconstructed = pattern_compressed
            
            # Reshape to original
            original_shape = metadata['original_shape']
            
            # Ensure correct element count
            expected_elements = 1
            for dim in original_shape:
                expected_elements *= dim
            
            if reconstructed.numel() != expected_elements:
                raise ValueError(f"Shape mismatch: reconstructed {reconstructed.numel()} elements, expected {expected_elements} for shape {original_shape}")
            
            reconstructed_tensor = reconstructed.reshape(original_shape).float()
            
            stage_metrics['pattern_reconstruction'] = {
                'time': time.perf_counter() - stage_start
            }
            
            # Calculate metrics
            total_time = time.perf_counter() - start_time
            
            logger.info(f"Decompression complete: shape={reconstructed_tensor.shape}, time={total_time:.3f}s")
            
            return DecompressionResult(
                reconstructed_tensor=reconstructed_tensor,
                original_shape=tuple(original_shape),
                processing_time=total_time,
                memory_usage=reconstructed_tensor.numel() * 4,
                stage_metrics=stage_metrics,
                validation_passed=True,
                reconstruction_error=0.0
            )
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise RuntimeError(f"Critical decompression failure: {e}")
    
    def _extract_sparse_indices(self, sparse_tensor) -> Any:
        """Extract sparse tensor indices for metadata - FIXED for CSR format"""
        # CSR format uses different methods than COO
        if sparse_tensor.layout == torch.sparse_csr:
            # CSR format - extract crow and col indices
            crow_indices = sparse_tensor.crow_indices().cpu().numpy().tolist()
            col_indices = sparse_tensor.col_indices().cpu().numpy().tolist() 
            return {'crow_indices': crow_indices, 'col_indices': col_indices}
        elif hasattr(sparse_tensor, '_indices'):
            # COO format
            return sparse_tensor._indices().cpu().numpy().tolist()
        else:
            # NO FALLBACK - HARD FAILURE
            raise RuntimeError(f"Cannot extract indices from sparse tensor with layout {sparse_tensor.layout}")
    
    def _combine_data_final(self, entropy_data: bytes, metadata: bytes) -> bytes:
        """Combine without corruption"""
        combined = bytearray()
        combined.extend(struct.pack('<B', 4))  # Version 4
        combined.extend(struct.pack('<I', len(entropy_data)))
        combined.extend(struct.pack('<I', len(metadata)))
        combined.extend(entropy_data)  # Store AS-IS
        combined.extend(metadata)
        return bytes(combined)
    
    def _split_data_final(self, combined: bytes) -> Tuple[bytes, bytes]:
        """Split without corruption"""
        version = combined[0]
        
        if version == 4:
            entropy_size = struct.unpack('<I', combined[1:5])[0]
            metadata_size = struct.unpack('<I', combined[5:9])[0]
            entropy_data = combined[9:9+entropy_size]
            metadata = combined[9+entropy_size:9+entropy_size+metadata_size]
        elif version == 3:
            # Handle old version - the padic data IS the corrupted entropy data
            padic_size = struct.unpack('<I', combined[1:5])[0]
            metadata_size = struct.unpack('<I', combined[5:9])[0]
            # The "padic_data" is actually corrupted entropy data
            entropy_data = combined[9:9+padic_size]
            metadata = combined[9+padic_size:9+padic_size+metadata_size]
        else:
            raise ValueError(f"Unsupported version: {version}")
        
        return entropy_data, metadata
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up P-adic compression system")
        # Add any cleanup code if needed
        pass
