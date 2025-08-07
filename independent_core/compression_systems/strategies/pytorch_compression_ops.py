"""
PyTorch Operations Optimized for Compression
Replaces JAX-based compression operations with pure PyTorch
PRODUCTION READY - NO PLACEHOLDERS - HARD FAILURES ONLY
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QuantizationResult:
    """Result from weight quantization"""
    quantized_weights: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    bits: int
    min_val: float
    max_val: float
    
    def dequantize(self) -> torch.Tensor:
        """Reconstruct original weights from quantized representation"""
        return (self.quantized_weights.float() - self.zero_point) * self.scale


@dataclass
class SparseEncoding:
    """Sparse tensor encoding result"""
    values: torch.Tensor
    indices: torch.Tensor
    shape: torch.Size
    density: float
    nnz: int  # Number of non-zero elements
    
    def to_dense(self) -> torch.Tensor:
        """Reconstruct dense tensor from sparse encoding"""
        dense = torch.zeros(self.shape, dtype=self.values.dtype, device=self.values.device)
        if self.indices.numel() > 0:
            # Unflatten indices for multi-dimensional tensors
            if len(self.shape) > 1:
                unraveled = []
                remaining = self.indices
                for dim_size in reversed(self.shape[1:]):
                    unraveled.append(remaining % dim_size)
                    remaining = remaining // dim_size
                unraveled.append(remaining)
                indices_tuple = tuple(reversed(unraveled))
            else:
                indices_tuple = (self.indices,)
            
            dense[indices_tuple] = self.values
        return dense


class PyTorchCompressionOps(nn.Module):
    """
    PyTorch operations optimized for compression.
    Provides quantization, sparsification, and channel-wise compression.
    """
    
    def __init__(self,
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32,
                 compile_mode: bool = True):
        """
        Initialize compression operations module.
        
        Args:
            device: Target device (cuda/cpu)
            dtype: Default data type
            compile_mode: Enable torch.compile optimization
        """
        super().__init__()
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.dtype = dtype
        self.compile_mode = compile_mode
        
        # Compile critical methods if available
        if compile_mode and hasattr(torch, 'compile'):
            try:
                self._quantize_impl = torch.compile(self._quantize_impl, fullgraph=True)
                self._sparse_encode_impl = torch.compile(self._sparse_encode_impl, fullgraph=True)
                logger.info("Compression ops compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Failed to compile compression ops: {e}")
    
    def _quantize_impl(self, 
                      weights: torch.Tensor, 
                      bits: int,
                      symmetric: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Core quantization implementation.
        
        Args:
            weights: Weights to quantize
            bits: Number of bits for quantization
            symmetric: Use symmetric quantization
            
        Returns:
            Tuple of (quantized_weights, scale, zero_point)
        """
        if symmetric:
            # Symmetric quantization
            max_val = weights.abs().max()
            qmax = 2**(bits - 1) - 1
            scale = max_val / qmax if max_val > 0 else 1.0
            zero_point = torch.tensor(0, dtype=torch.int32, device=weights.device)
            
            # Quantize
            quantized = torch.round(weights / scale)
            quantized = torch.clamp(quantized, -qmax, qmax).to(torch.int8 if bits <= 8 else torch.int16)
        else:
            # Asymmetric quantization
            min_val = weights.min()
            max_val = weights.max()
            qmin = 0
            qmax = 2**bits - 1
            
            scale = (max_val - min_val) / (qmax - qmin) if max_val > min_val else 1.0
            zero_point = qmin - torch.round(min_val / scale)
            zero_point = torch.clamp(zero_point, qmin, qmax).to(torch.int32)
            
            # Quantize
            quantized = torch.round(weights / scale + zero_point)
            quantized = torch.clamp(quantized, qmin, qmax).to(torch.uint8 if bits <= 8 else torch.int16)
        
        return quantized, scale, zero_point
    
    def quantize_weights(self, 
                        weights: torch.Tensor, 
                        bits: int = 8,
                        symmetric: bool = True,
                        per_channel: bool = False,
                        channel_axis: int = 0) -> QuantizationResult:
        """
        Quantize weights to specified bit depth.
        
        Args:
            weights: Weights to quantize
            bits: Number of bits (1-16)
            symmetric: Use symmetric quantization
            per_channel: Quantize per channel for better accuracy
            channel_axis: Axis for per-channel quantization
            
        Returns:
            QuantizationResult with quantized weights and metadata
        """
        if bits < 1 or bits > 16:
            raise ValueError(f"Bits must be between 1 and 16, got {bits}")
        
        weights = weights.to(device=self.device)
        original_shape = weights.shape
        
        if per_channel and len(original_shape) > 1:
            # Per-channel quantization
            num_channels = original_shape[channel_axis]
            
            # Move channel axis to first dimension
            perm = list(range(len(original_shape)))
            perm[0], perm[channel_axis] = perm[channel_axis], perm[0]
            weights_transposed = weights.permute(perm)
            weights_2d = weights_transposed.reshape(num_channels, -1)
            
            # Quantize each channel
            quantized_list = []
            scales = []
            zero_points = []
            
            for c in range(num_channels):
                q, s, z = self._quantize_impl(weights_2d[c], bits, symmetric)
                quantized_list.append(q)
                scales.append(s)
                zero_points.append(z)
            
            quantized = torch.stack(quantized_list)
            scale = torch.stack(scales)
            zero_point = torch.stack(zero_points)
            
            # Reshape back
            quantized = quantized.reshape(weights_transposed.shape)
            inv_perm = [0] * len(perm)
            for i, p in enumerate(perm):
                inv_perm[p] = i
            quantized = quantized.permute(inv_perm)
            
        else:
            # Global quantization
            quantized, scale, zero_point = self._quantize_impl(weights, bits, symmetric)
        
        return QuantizationResult(
            quantized_weights=quantized,
            scale=scale,
            zero_point=zero_point,
            bits=bits,
            min_val=weights.min().item(),
            max_val=weights.max().item()
        )
    
    def _sparse_encode_impl(self, 
                           tensor: torch.Tensor, 
                           threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core sparse encoding implementation.
        
        Args:
            tensor: Input tensor
            threshold: Threshold for sparsification
            
        Returns:
            Tuple of (values, indices)
        """
        # Find non-zero elements above threshold
        mask = torch.abs(tensor) > threshold
        values = tensor[mask]
        
        # Get flat indices
        indices = torch.nonzero(mask.flatten(), as_tuple=False).squeeze(-1)
        
        return values, indices
    
    def sparse_encode(self, 
                     tensor: torch.Tensor, 
                     threshold: float = 1e-6,
                     top_k: Optional[int] = None) -> SparseEncoding:
        """
        Encode sparse tensor efficiently.
        
        Args:
            tensor: Input tensor
            threshold: Magnitude threshold for sparsification
            top_k: Keep only top-k largest magnitude values
            
        Returns:
            SparseEncoding with values and indices
        """
        tensor = tensor.to(device=self.device)
        original_shape = tensor.shape
        flat_tensor = tensor.flatten()
        
        if top_k is not None:
            # Keep only top-k values
            if top_k > flat_tensor.numel():
                top_k = flat_tensor.numel()
            
            topk_values, topk_indices = torch.topk(torch.abs(flat_tensor), top_k)
            mask = torch.zeros_like(flat_tensor, dtype=torch.bool)
            mask[topk_indices] = True
            values = flat_tensor[mask]
            indices = topk_indices
        else:
            # Threshold-based sparsification
            values, indices = self._sparse_encode_impl(tensor, threshold)
        
        nnz = len(values)
        density = nnz / tensor.numel()
        
        return SparseEncoding(
            values=values,
            indices=indices,
            shape=original_shape,
            density=density,
            nnz=nnz
        )
    
    def channel_wise_compression(self, 
                                tensor: torch.Tensor,
                                channels: List[str]) -> Dict[str, torch.Tensor]:
        """
        Compress tensor channel-wise based on IEEE 754 decomposition.
        
        Args:
            tensor: Input tensor
            channels: List of channels to extract ('sign', 'exponent', 'mantissa')
            
        Returns:
            Dictionary with compressed channels
        """
        tensor = tensor.to(device=self.device, dtype=torch.float32)
        result = {}
        
        # Convert to int32 view for bit manipulation
        int_view = tensor.view(torch.int32)
        
        for channel in channels:
            if channel == 'sign':
                # Extract sign bit (bit 31)
                sign = (int_view >> 31) & 1
                result['sign'] = sign.to(torch.int8)
                
            elif channel == 'exponent':
                # Extract exponent bits (bits 23-30)
                exponent = (int_view >> 23) & 0xFF
                result['exponent'] = exponent.to(torch.uint8)
                
            elif channel == 'mantissa':
                # Extract mantissa bits (bits 0-22)
                mantissa = int_view & 0x7FFFFF
                # Compress mantissa using quantization
                mantissa_float = mantissa.float() / 0x7FFFFF
                result['mantissa'] = self.quantize_mantissa(mantissa_float)
                
            else:
                raise ValueError(f"Unknown channel: {channel}")
        
        return result
    
    def quantize_mantissa(self, 
                         mantissa: torch.Tensor,
                         bits: int = 16) -> torch.Tensor:
        """
        Quantize mantissa values for compression.
        
        Args:
            mantissa: Normalized mantissa values [0, 1]
            bits: Number of bits for quantization
            
        Returns:
            Quantized mantissa
        """
        scale = (2**bits - 1)
        quantized = torch.round(mantissa * scale)
        
        if bits <= 8:
            return quantized.to(torch.uint8)
        elif bits <= 16:
            return quantized.to(torch.uint16)
        else:
            return quantized.to(torch.int32)
    
    def adaptive_quantization(self, 
                            tensor: torch.Tensor,
                            target_size_ratio: float = 0.1) -> QuantizationResult:
        """
        Adaptively choose quantization bits based on target compression ratio.
        
        Args:
            tensor: Input tensor
            target_size_ratio: Target size as ratio of original
            
        Returns:
            QuantizationResult with optimal bit depth
        """
        original_bits = 32  # Assuming float32
        target_bits = int(original_bits * target_size_ratio)
        target_bits = max(1, min(16, target_bits))
        
        # Try different bit depths and choose best
        best_result = None
        best_error = float('inf')
        
        for bits in range(max(1, target_bits - 2), min(16, target_bits + 3)):
            result = self.quantize_weights(tensor, bits=bits)
            
            # Compute reconstruction error
            reconstructed = result.dequantize()
            error = (tensor - reconstructed).abs().mean().item()
            
            # Balance between compression and accuracy
            size_ratio = bits / original_bits
            score = error + 0.1 * abs(size_ratio - target_size_ratio)
            
            if score < best_error:
                best_error = score
                best_result = result
        
        return best_result
    
    def pruning_mask(self, 
                    tensor: torch.Tensor,
                    sparsity: float = 0.9,
                    structured: bool = False,
                    block_size: int = 4) -> torch.Tensor:
        """
        Create pruning mask for structured or unstructured sparsity.
        
        Args:
            tensor: Weight tensor
            sparsity: Target sparsity level (0-1)
            structured: Use structured pruning
            block_size: Block size for structured pruning
            
        Returns:
            Binary mask tensor
        """
        tensor = tensor.to(device=self.device)
        
        if structured and len(tensor.shape) >= 2:
            # Structured pruning (prune entire blocks)
            reshaped = tensor.view(-1, block_size)
            block_norms = reshaped.norm(dim=1)
            
            k = int(len(block_norms) * (1 - sparsity))
            _, indices = torch.topk(block_norms, k)
            
            mask = torch.zeros_like(reshaped, dtype=torch.bool)
            mask[indices] = True
            mask = mask.view(tensor.shape)
        else:
            # Unstructured pruning
            k = int(tensor.numel() * (1 - sparsity))
            _, indices = torch.topk(tensor.abs().flatten(), k)
            
            mask = torch.zeros_like(tensor.flatten(), dtype=torch.bool)
            mask[indices] = True
            mask = mask.view(tensor.shape)
        
        return mask
    
    def delta_encoding(self, 
                      tensor: torch.Tensor,
                      base: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Delta encoding for sequential data or weight updates.
        
        Args:
            tensor: Current tensor
            base: Base tensor for delta computation
            
        Returns:
            Dictionary with base and deltas
        """
        tensor = tensor.to(device=self.device)
        
        if base is None:
            # Use first element as base for sequential data
            if tensor.dim() == 1:
                base = tensor[0:1]
                deltas = tensor[1:] - tensor[:-1]
            else:
                base = tensor[0]
                deltas = tensor[1:] - tensor[:-1]
        else:
            base = base.to(device=self.device)
            deltas = tensor - base
        
        # Quantize deltas more aggressively since they're usually smaller
        delta_quant = self.quantize_weights(deltas, bits=4)
        
        return {
            'base': base,
            'deltas': delta_quant.quantized_weights,
            'delta_scale': delta_quant.scale,
            'delta_zero_point': delta_quant.zero_point
        }
    
    def huffman_encode_indices(self, indices: torch.Tensor) -> Dict[str, Any]:
        """
        Huffman encoding for sparse indices.
        
        Args:
            indices: Sparse tensor indices
            
        Returns:
            Dictionary with Huffman codes and codebook
        """
        indices = indices.cpu().numpy()
        
        # Count frequencies
        unique, counts = torch.unique(torch.tensor(indices), return_counts=True)
        
        # Build frequency table
        freq_table = {}
        for val, count in zip(unique.tolist(), counts.tolist()):
            freq_table[val] = count
        
        # Simple Huffman tree construction
        import heapq
        
        class Node:
            def __init__(self, freq, symbol=None, left=None, right=None):
                self.freq = freq
                self.symbol = symbol
                self.left = left
                self.right = right
            
            def __lt__(self, other):
                return self.freq < other.freq
        
        # Build tree
        heap = [Node(freq, symbol=sym) for sym, freq in freq_table.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            parent = Node(left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, parent)
        
        # Generate codes
        codes = {}
        
        def generate_codes(node, code=''):
            if node:
                if node.symbol is not None:
                    codes[node.symbol] = code if code else '0'
                else:
                    generate_codes(node.left, code + '0')
                    generate_codes(node.right, code + '1')
        
        if heap:
            generate_codes(heap[0])
        
        # Encode indices
        encoded = ''.join(codes.get(int(idx), '0') for idx in indices)
        
        return {
            'encoded': encoded,
            'codebook': codes,
            'original_length': len(indices),
            'compressed_bits': len(encoded),
            'compression_ratio': (len(indices) * 32) / max(1, len(encoded))
        }
    
    def block_wise_compression(self,
                             tensor: torch.Tensor,
                             block_size: int = 32) -> Dict[str, Any]:
        """
        Compress tensor in blocks for better cache efficiency.
        
        Args:
            tensor: Input tensor
            block_size: Size of blocks
            
        Returns:
            Dictionary with block-wise compressed data
        """
        tensor = tensor.to(device=self.device)
        flat = tensor.flatten()
        
        # Pad if necessary
        pad_size = (block_size - len(flat) % block_size) % block_size
        if pad_size > 0:
            flat = F.pad(flat, (0, pad_size))
        
        # Reshape into blocks
        blocks = flat.view(-1, block_size)
        
        compressed_blocks = []
        block_metadata = []
        
        for block in blocks:
            # Analyze block characteristics
            sparsity = (block.abs() < 1e-6).float().mean().item()
            
            if sparsity > 0.8:
                # High sparsity - use sparse encoding
                sparse = self.sparse_encode(block)
                compressed_blocks.append({
                    'type': 'sparse',
                    'values': sparse.values,
                    'indices': sparse.indices
                })
            else:
                # Dense block - use quantization
                quant = self.quantize_weights(block, bits=8)
                compressed_blocks.append({
                    'type': 'quantized',
                    'data': quant.quantized_weights,
                    'scale': quant.scale,
                    'zero_point': quant.zero_point
                })
            
            block_metadata.append({
                'sparsity': sparsity,
                'min': block.min().item(),
                'max': block.max().item(),
                'mean': block.mean().item()
            })
        
        return {
            'blocks': compressed_blocks,
            'metadata': block_metadata,
            'original_shape': tensor.shape,
            'block_size': block_size,
            'num_blocks': len(blocks)
        }
    
    def forward(self, 
               tensor: torch.Tensor,
               method: str = 'quantize',
               **kwargs) -> Union[QuantizationResult, SparseEncoding, Dict]:
        """
        Forward pass for nn.Module compatibility.
        
        Args:
            tensor: Input tensor
            method: Compression method to use
            **kwargs: Method-specific arguments
            
        Returns:
            Compression result based on method
        """
        if method == 'quantize':
            return self.quantize_weights(tensor, **kwargs)
        elif method == 'sparse':
            return self.sparse_encode(tensor, **kwargs)
        elif method == 'channel':
            return self.channel_wise_compression(tensor, **kwargs)
        elif method == 'block':
            return self.block_wise_compression(tensor, **kwargs)
        else:
            raise ValueError(f"Unknown compression method: {method}")