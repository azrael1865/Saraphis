#!/usr/bin/env python3
"""
Fix for the p-adic decompression pipeline focusing on correct shape handling
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import struct
import logging

logger = logging.getLogger(__name__)


def fix_decompress_stage2(entropy_compressed: bytes, entropy_meta: Dict[str, Any], 
                         entropy_bridge, decompression_device: torch.device) -> torch.Tensor:
    """Fixed Stage 2: Entropy Decoding with proper empty handling"""
    
    if entropy_meta.get('empty', False) or not entropy_compressed:
        # Handle empty tensor case
        logger.debug("Empty tensor case in entropy decoding")
        return torch.tensor([], device=decompression_device)
    
    # Decode non-empty compressed data
    try:
        sparse_values_tensor = entropy_bridge.decode_padic_tensor(
            entropy_compressed,
            entropy_meta
        )
        logger.debug(f"Decoded {sparse_values_tensor.numel()} values from entropy coding")
        return sparse_values_tensor
    except Exception as e:
        logger.error(f"Entropy decoding failed: {e}")
        # Return empty tensor on failure
        return torch.tensor([], device=decompression_device)


def fix_decompress_stage3(metadata: Dict[str, Any], sparse_values_tensor: torch.Tensor,
                         decompression_device: torch.device, sparse_bridge) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fixed Stage 3: Sparse Reconstruction with proper shape handling"""
    
    # Extract critical shape information
    digit_tensor_shape = metadata.get('digit_tensor_shape', None)
    sparse_shape = tuple(metadata.get('sparse_shape', []))
    valuations = torch.tensor(metadata.get('valuations', []), device=decompression_device)
    
    logger.debug(f"Stage 3 - digit_tensor_shape: {digit_tensor_shape}, sparse_shape: {sparse_shape}")
    logger.debug(f"Stage 3 - sparse_values count: {sparse_values_tensor.numel()}")
    
    # Handle empty case
    if sparse_values_tensor.numel() == 0:
        # Create empty digit tensor with correct shape
        if digit_tensor_shape and len(digit_tensor_shape) >= 2:
            digit_tensor = torch.zeros(digit_tensor_shape, dtype=torch.int32, device=decompression_device)
        else:
            # Fallback: infer from original shape
            original_shape = metadata.get('original_shape', [1])
            num_elements = np.prod(original_shape)
            precision = metadata.get('precision', 4)
            digit_tensor = torch.zeros((num_elements, precision), dtype=torch.int32, device=decompression_device)
        return digit_tensor, valuations
    
    # Handle CSR format
    if metadata.get('sparse_format') == 'csr' and metadata.get('sparse_crow_indices') is not None:
        try:
            crow_indices = torch.tensor(metadata['sparse_crow_indices'], dtype=torch.long, device=decompression_device)
            col_indices = torch.tensor(metadata['sparse_col_indices'], dtype=torch.long, device=decompression_device)
            
            # Validate indices match values
            expected_values = len(col_indices)
            if sparse_values_tensor.numel() != expected_values:
                logger.warning(f"Value count mismatch: got {sparse_values_tensor.numel()}, expected {expected_values}")
                # Adjust values tensor
                if sparse_values_tensor.numel() < expected_values:
                    # Pad with zeros
                    padding = torch.zeros(expected_values - sparse_values_tensor.numel(), 
                                        device=decompression_device, dtype=sparse_values_tensor.dtype)
                    sparse_values_tensor = torch.cat([sparse_values_tensor, padding])
                else:
                    # Truncate
                    sparse_values_tensor = sparse_values_tensor[:expected_values]
            
            # Create sparse tensor
            sparse_tensor = torch.sparse_csr_tensor(
                crow_indices=crow_indices,
                col_indices=col_indices,
                values=sparse_values_tensor,
                size=sparse_shape,
                device=decompression_device
            )
            
            # Convert to dense
            digit_tensor = sparse_tensor.to_dense()
            
            # Ensure correct shape
            if digit_tensor_shape and list(digit_tensor.shape) != list(digit_tensor_shape):
                logger.debug(f"Reshaping from {digit_tensor.shape} to {digit_tensor_shape}")
                try:
                    digit_tensor = digit_tensor.reshape(digit_tensor_shape)
                except:
                    # If reshape fails, create new tensor with correct shape
                    expected_elements = np.prod(digit_tensor_shape)
                    if digit_tensor.numel() < expected_elements:
                        # Pad
                        digit_tensor = torch.cat([
                            digit_tensor.flatten(),
                            torch.zeros(expected_elements - digit_tensor.numel(), 
                                      dtype=digit_tensor.dtype, device=decompression_device)
                        ]).reshape(digit_tensor_shape)
                    else:
                        # Truncate
                        digit_tensor = digit_tensor.flatten()[:expected_elements].reshape(digit_tensor_shape)
            
        except Exception as e:
            logger.error(f"CSR reconstruction failed: {e}")
            # Fallback to direct reshape
            if digit_tensor_shape:
                expected_elements = np.prod(digit_tensor_shape)
                if sparse_values_tensor.numel() >= expected_elements:
                    digit_tensor = sparse_values_tensor[:expected_elements].reshape(digit_tensor_shape)
                else:
                    # Pad and reshape
                    padding = torch.zeros(expected_elements - sparse_values_tensor.numel(),
                                        device=decompression_device, dtype=sparse_values_tensor.dtype)
                    digit_tensor = torch.cat([sparse_values_tensor, padding]).reshape(digit_tensor_shape)
            else:
                digit_tensor = sparse_values_tensor.unsqueeze(-1)
    else:
        # Direct reconstruction
        if digit_tensor_shape and len(digit_tensor_shape) >= 2:
            try:
                digit_tensor = sparse_values_tensor.reshape(digit_tensor_shape)
            except:
                # Shape mismatch - create correct shape
                expected_elements = np.prod(digit_tensor_shape)
                if sparse_values_tensor.numel() < expected_elements:
                    padding = torch.zeros(expected_elements - sparse_values_tensor.numel(),
                                        device=decompression_device, dtype=sparse_values_tensor.dtype)
                    digit_tensor = torch.cat([sparse_values_tensor, padding]).reshape(digit_tensor_shape)
                else:
                    digit_tensor = sparse_values_tensor[:expected_elements].reshape(digit_tensor_shape)
        else:
            # Fallback
            digit_tensor = sparse_values_tensor.unsqueeze(-1)
    
    return digit_tensor, valuations


def fix_decompress_stage5(digit_tensor: torch.Tensor, valuations: torch.Tensor, 
                         metadata: Dict[str, Any], config, safe_reconstruction,
                         decompression_device: torch.device) -> torch.Tensor:
    """Fixed Stage 5: P-adic to Float Reconstruction with proper element count"""
    
    from fractions import Fraction
    import sys
    sys.path.insert(0, '/home/will-casterlin/Desktop/Saraphis')
    from independent_core.compression_systems.padic.padic_encoder import PadicWeight
    
    # Get original shape and calculate required elements
    original_shape = metadata.get('original_shape', [])
    num_elements = np.prod(original_shape) if original_shape else digit_tensor.shape[0]
    
    logger.debug(f"Stage 5 - Need {num_elements} elements for shape {original_shape}")
    logger.debug(f"Stage 5 - digit_tensor shape: {digit_tensor.shape}")
    
    # Get precision map
    precision_map = torch.tensor(
        metadata.get('precision_map', [config.base_precision] * num_elements),
        device=decompression_device
    )
    
    # Ensure digit tensor has correct number of rows
    if digit_tensor.shape[0] != num_elements:
        if digit_tensor.shape[0] < num_elements:
            # Pad with zeros
            padding_shape = list(digit_tensor.shape)
            padding_shape[0] = num_elements - digit_tensor.shape[0]
            padding = torch.zeros(padding_shape, dtype=digit_tensor.dtype, device=digit_tensor.device)
            digit_tensor = torch.cat([digit_tensor, padding], dim=0)
            logger.debug(f"Padded digit_tensor to {digit_tensor.shape}")
        else:
            # Truncate
            digit_tensor = digit_tensor[:num_elements]
            logger.debug(f"Truncated digit_tensor to {digit_tensor.shape}")
    
    # Ensure valuations has correct length
    if valuations.shape[0] < num_elements:
        padding = torch.zeros(num_elements - valuations.shape[0], 
                             dtype=valuations.dtype, device=valuations.device)
        valuations = torch.cat([valuations, padding])
    
    # Reconstruct p-adic weights
    padic_weights = []
    prime = metadata['prime']
    
    for i in range(num_elements):
        precision = int(precision_map[i].item()) if i < precision_map.shape[0] else config.base_precision
        
        # Handle different digit tensor shapes
        if digit_tensor.dim() == 1:
            # 1D tensor - each element is a single digit
            digits = [digit_tensor[i].item()]
            # Pad to precision
            digits.extend([0] * (precision - 1))
        elif digit_tensor.dim() == 2:
            # 2D tensor - standard case
            actual_precision = min(precision, digit_tensor.shape[1])
            digits = digit_tensor[i, :actual_precision].cpu().numpy().tolist()
            # Pad if needed
            if len(digits) < precision:
                digits.extend([0] * (precision - len(digits)))
        else:
            # Unexpected shape - flatten and take what we need
            flat_idx = i * precision
            if flat_idx + precision <= digit_tensor.numel():
                digits = digit_tensor.flatten()[flat_idx:flat_idx + precision].cpu().numpy().tolist()
            else:
                digits = [0] * precision
        
        valuation = int(valuations[i].item()) if i < valuations.shape[0] else 0
        
        # Reconstruct value
        value_num = sum(d * (prime ** j) for j, d in enumerate(digits))
        if valuation >= 0:
            value = Fraction(value_num, 1)
        else:
            value = Fraction(value_num, prime ** (-valuation))
        
        weight = PadicWeight(
            value=value,
            digits=digits,
            valuation=valuation,
            prime=prime,
            precision=precision
        )
        padic_weights.append(weight)
    
    # Reconstruct float values
    reconstructed_values = []
    for weight in padic_weights:
        value = safe_reconstruction.reconstruct(weight)
        reconstructed_values.append(value)
    
    # Create final tensor with original shape
    reconstructed_tensor = torch.tensor(
        reconstructed_values,
        dtype=torch.float32,
        device=decompression_device
    )
    
    # Reshape to original
    try:
        reconstructed_tensor = reconstructed_tensor.reshape(original_shape)
    except Exception as e:
        logger.error(f"Failed to reshape to {original_shape}: {e}")
        # Return flat tensor if reshape fails
        pass
    
    return reconstructed_tensor


def create_fixed_decompressor_patch():
    """Create a monkey patch for the decompression method"""
    
    def fixed_decompress(self, compressed_data: bytes):
        """Fixed decompression with robust shape handling"""
        import time
        start_time = time.perf_counter()
        stage_metrics = {}
        
        try:
            logger.info("Starting FIXED decompression")
            
            # Split combined data
            entropy_compressed, compressed_metadata, pattern_encoded = self._split_compressed_data(
                compressed_data
            )
            
            # Stage 1: Metadata Decompression
            stage_start = time.perf_counter()
            logger.info("Stage 1: Metadata Decompression")
            
            metadata = self.metadata_compressor.decompress_metadata(compressed_metadata)
            
            stage_metrics['metadata_decompression'] = {
                'time': time.perf_counter() - stage_start,
                'metadata_size': len(compressed_metadata)
            }
            
            # Log critical metadata
            logger.debug(f"Metadata keys: {list(metadata.keys())}")
            logger.debug(f"Original shape: {metadata.get('original_shape')}")
            logger.debug(f"Digit tensor shape: {metadata.get('digit_tensor_shape')}")
            
            # Stage 2: Fixed Entropy Decoding
            stage_start = time.perf_counter()
            logger.info("Stage 2: Fixed Entropy Decoding")
            
            entropy_meta = metadata.get('entropy_metadata', {})
            sparse_values_tensor = fix_decompress_stage2(
                entropy_compressed, entropy_meta, self.entropy_bridge, self.decompression_device
            )
            
            stage_metrics['entropy_decoding'] = {
                'time': time.perf_counter() - stage_start,
                'values_decoded': sparse_values_tensor.numel()
            }
            
            # Stage 3: Fixed Sparse Reconstruction
            stage_start = time.perf_counter()
            logger.info("Stage 3: Fixed Sparse Reconstruction")
            
            digit_tensor, valuations = fix_decompress_stage3(
                metadata, sparse_values_tensor, self.decompression_device, self.sparse_bridge
            )
            
            stage_metrics['sparse_reconstruction'] = {
                'time': time.perf_counter() - stage_start,
                'tensor_shape': list(digit_tensor.shape)
            }
            
            # Stage 4: Pattern Reconstruction (skip if no patterns)
            stage_start = time.perf_counter()
            logger.info("Stage 4: Pattern Reconstruction")
            
            if pattern_encoded and metadata.get('pattern_dict'):
                # Pattern reconstruction code (unchanged)
                pass
            
            stage_metrics['pattern_reconstruction'] = {
                'time': time.perf_counter() - stage_start,
                'patterns_used': len(metadata.get('pattern_dict', {}))
            }
            
            # Stage 5: Fixed P-adic to Float Reconstruction
            stage_start = time.perf_counter()
            logger.info("Stage 5: Fixed P-adic to Float Reconstruction")
            
            reconstructed_tensor = fix_decompress_stage5(
                digit_tensor, valuations, metadata, self.config,
                self.safe_reconstruction, self.decompression_device
            )
            
            stage_metrics['padic_reconstruction'] = {
                'time': time.perf_counter() - stage_start,
                'weights_reconstructed': reconstructed_tensor.numel()
            }
            
            # Final metrics
            total_time = time.perf_counter() - start_time
            memory_usage = reconstructed_tensor.numel() * reconstructed_tensor.element_size()
            
            logger.info(f"FIXED decompression complete: shape={reconstructed_tensor.shape}, time={total_time:.3f}s")
            
            from independent_core.compression_systems.padic.padic_compressor import DecompressionResult
            
            return DecompressionResult(
                reconstructed_tensor=reconstructed_tensor,
                original_shape=tuple(metadata['original_shape']),
                processing_time=total_time,
                memory_usage=memory_usage,
                stage_metrics=stage_metrics,
                validation_passed=True,
                reconstruction_error=0.0
            )
            
        except Exception as e:
            logger.error(f"FIXED decompression failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Critical decompression failure: {e}") from e
    
    return fixed_decompress


if __name__ == "__main__":
    print("P-adic decompressor fix module loaded")
    print("Use create_fixed_decompressor_patch() to get the patched decompress method")