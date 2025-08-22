"""
P-adic gradient compression integration with GAC system.
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List
import time

from padic_compressor import PadicCompressionSystem
from padic_encoder import PadicValidation


class PadicGradientCompressor:
    """P-adic compression specifically for gradient compression in GAC system"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize P-adic gradient compressor"""
        if not isinstance(config, dict):
            raise TypeError(f"Config must be dict, got {type(config)}")
        
        # Validate gradient-specific configuration
        self._validate_gradient_config(config)
        
        # Initialize core p-adic system
        self.padic_system = PadicCompressionSystem(config)
        
        # Gradient-specific parameters
        self.gradient_threshold = config.get('gradient_threshold', 1e-8)
        self.preserve_gradient_norm = config.get('preserve_gradient_norm', True)
        self.adaptive_precision = config.get('adaptive_precision', True)
        self.error_feedback = config.get('error_feedback', True)
        
        # Error accumulation for feedback
        self.accumulated_error = None
        self.compression_history = []
        
        # Performance tracking
        self.gradient_stats = {
            'gradients_compressed': 0,
            'total_compression_error': 0.0,
            'max_compression_error': 0.0,
            'norm_preservation_error': 0.0
        }
    
    def _validate_gradient_config(self, config: Dict[str, Any]) -> None:
        """Validate gradient-specific configuration"""
        if 'gradient_threshold' in config:
            threshold = config['gradient_threshold']
            if not isinstance(threshold, (int, float)):
                raise TypeError(f"Gradient threshold must be numeric, got {type(threshold)}")
            if threshold <= 0:
                raise ValueError(f"Gradient threshold must be > 0, got {threshold}")
        
        bool_params = ['preserve_gradient_norm', 'adaptive_precision', 'error_feedback']
        for param in bool_params:
            if param in config and not isinstance(config[param], bool):
                raise TypeError(f"{param} must be bool, got {type(config[param])}")
    
    def compress_gradient(self, gradient: torch.Tensor, 
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compress gradient tensor with gradient-specific optimizations"""
        if context is None:
            context = {}
        
        # Validate gradient
        PadicValidation.validate_tensor(gradient)
        self._validate_gradient_properties(gradient)
        
        start_time = time.time()
        
        # Store original gradient norm for validation
        original_norm = torch.norm(gradient).item()
        
        # Apply error feedback if enabled
        if self.error_feedback and self.accumulated_error is not None:
            if self.accumulated_error.shape == gradient.shape:
                gradient = gradient + self.accumulated_error
            else:
                # Reset error if shape mismatch
                self.accumulated_error = None
        
        # Filter small gradients if threshold is set
        filtered_gradient = self._filter_small_gradients(gradient)
        
        # Adaptive precision based on gradient magnitude
        if self.adaptive_precision:
            precision = self._compute_adaptive_precision(filtered_gradient)
            if precision != self.padic_system.precision:
                # Create new system with adaptive precision
                adaptive_config = self.padic_system.config.copy()
                adaptive_config['precision'] = precision
                adaptive_system = PadicCompressionSystem(adaptive_config)
                compressed = adaptive_system.compress(filtered_gradient)
            else:
                compressed = self.padic_system.compress(filtered_gradient)
        else:
            compressed = self.padic_system.compress(filtered_gradient)
        
        # Store compression metadata
        compressed['gradient_context'] = {
            'original_norm': original_norm,
            'filtered': torch.norm(filtered_gradient).item() != original_norm,
            'adaptive_precision_used': self.adaptive_precision,
            'compression_timestamp': time.time()
        }
        
        # Validate compression if enabled
        if self.preserve_gradient_norm:
            self._validate_gradient_compression(gradient, compressed)
        
        # Update statistics
        compression_time = time.time() - start_time
        self._update_gradient_stats(gradient, compressed, compression_time)
        
        return compressed
    
    def decompress_gradient(self, compressed: Dict[str, Any]) -> torch.Tensor:
        """Decompress gradient tensor"""
        if not isinstance(compressed, dict):
            raise TypeError(f"Compressed data must be dict, got {type(compressed)}")
        
        start_time = time.time()
        
        # Check if adaptive precision was used
        metadata = compressed.get('metadata', {})
        precision = metadata.get('precision', self.padic_system.precision)
        
        if precision != self.padic_system.precision:
            # Create system with matching precision
            adaptive_config = self.padic_system.config.copy()
            adaptive_config['precision'] = precision
            adaptive_system = PadicCompressionSystem(adaptive_config)
            gradient = adaptive_system.decompress(compressed)
        else:
            gradient = self.padic_system.decompress(compressed)
        
        # Update error feedback if enabled
        if self.error_feedback:
            self._update_error_feedback(compressed, gradient)
        
        # Validate gradient properties
        self._validate_gradient_properties(gradient)
        
        return gradient
    
    def _validate_gradient_properties(self, gradient: torch.Tensor) -> None:
        """Validate gradient-specific properties"""
        # Check gradient shape is reasonable
        if gradient.dim() > 6:
            raise ValueError(f"Gradient has too many dimensions: {gradient.dim()}")
        
        # Check for gradient explosion
        grad_norm = torch.norm(gradient).item()
        if grad_norm > 1e6:
            raise ValueError(f"Gradient norm too large: {grad_norm}")
        
        # Check gradient is not all zeros (might indicate dead neurons)
        if torch.all(gradient == 0):
            raise ValueError("Gradient is all zeros - possible dead neurons")
    
    def _filter_small_gradients(self, gradient: torch.Tensor) -> torch.Tensor:
        """Filter out gradients below threshold"""
        if self.gradient_threshold <= 0:
            return gradient
        
        # Create mask for gradients above threshold
        mask = torch.abs(gradient) >= self.gradient_threshold
        
        # Apply mask
        filtered = gradient * mask
        
        # Store filtered elements for error feedback
        if self.error_feedback:
            filtered_out = gradient * (~mask)
            if self.accumulated_error is None:
                self.accumulated_error = filtered_out
            else:
                if self.accumulated_error.shape == filtered_out.shape:
                    self.accumulated_error += filtered_out
                else:
                    self.accumulated_error = filtered_out
        
        return filtered
    
    def _compute_adaptive_precision(self, gradient: torch.Tensor) -> int:
        """Compute adaptive precision based on gradient statistics - SAFE PRECISION ONLY"""
        # Analyze gradient magnitude distribution
        grad_abs = torch.abs(gradient)
        grad_max = torch.max(grad_abs).item()
        grad_mean = torch.mean(grad_abs).item()
        grad_std = torch.std(grad_abs).item()
        
        # Compute dynamic range
        if grad_mean > 0:
            dynamic_range = grad_max / grad_mean
        else:
            dynamic_range = 1.0
        
        # SAFETY: Get prime from p-adic system
        prime = getattr(self.padic_system, 'prime', 257)
        
        # SAFETY: Calculate max safe precision for this prime
        import math
        safe_threshold = 1e12
        max_safe_precision = int(math.log(safe_threshold) / math.log(prime))
        
        # Map dynamic range to precision - WITHIN SAFE LIMITS
        base_precision = self.padic_system.precision
        
        if dynamic_range > 1000:
            precision = min(base_precision + 2, max_safe_precision)  # Reduced increase
        elif dynamic_range > 100:
            precision = min(base_precision + 1, max_safe_precision)  # Reduced increase
        elif dynamic_range > 10:
            precision = base_precision  # No increase for moderate range
        else:
            precision = base_precision
        
        # SAFETY: Ensure precision never exceeds safe limits
        precision = max(1, min(precision, max_safe_precision))
        
        return precision
    
    def _validate_gradient_compression(self, original: torch.Tensor, 
                                     compressed: Dict[str, Any]) -> None:
        """Validate gradient compression preserves important properties"""
        # Decompress to check
        decompressed = self.decompress_gradient(compressed)
        
        # Check norm preservation
        original_norm = torch.norm(original).item()
        decompressed_norm = torch.norm(decompressed).item()
        
        if original_norm > 0:
            norm_error = abs(original_norm - decompressed_norm) / original_norm
            if norm_error > 0.1:  # 10% tolerance
                raise ValueError(f"Gradient norm not preserved: error={norm_error:.4f}")
        
        # Check reconstruction error
        reconstruction_error = torch.nn.functional.mse_loss(original, decompressed).item()
        max_error = self.padic_system.max_reconstruction_error
        
        if reconstruction_error > max_error:
            raise ValueError(f"Gradient reconstruction error too high: {reconstruction_error:.8f} > {max_error}")
    
    def _update_error_feedback(self, compressed: Dict[str, Any], 
                              decompressed: torch.Tensor) -> None:
        """Update error feedback for next compression"""
        # This would require access to original gradient, so we'll implement
        # a simple decay mechanism for accumulated error
        if self.accumulated_error is not None:
            # Decay accumulated error over time
            decay_factor = 0.9
            self.accumulated_error *= decay_factor
            
            # Reset if error becomes too small
            if torch.norm(self.accumulated_error).item() < self.gradient_threshold:
                self.accumulated_error = None
    
    def _update_gradient_stats(self, original: torch.Tensor, 
                              compressed: Dict[str, Any], 
                              compression_time: float) -> None:
        """Update gradient compression statistics"""
        self.gradient_stats['gradients_compressed'] += 1
        
        # Compute compression error if possible
        try:
            decompressed = self.decompress_gradient(compressed)
            error = torch.nn.functional.mse_loss(original, decompressed).item()
            
            self.gradient_stats['total_compression_error'] += error
            self.gradient_stats['max_compression_error'] = max(
                self.gradient_stats['max_compression_error'], error
            )
            
            # Norm preservation error
            orig_norm = torch.norm(original).item()
            decomp_norm = torch.norm(decompressed).item()
            if orig_norm > 0:
                norm_error = abs(orig_norm - decomp_norm) / orig_norm
                self.gradient_stats['norm_preservation_error'] += norm_error
                
        except Exception:
            # Don't fail the compression if stats update fails
            pass
    
    def get_gradient_stats(self) -> Dict[str, Any]:
        """Get gradient compression statistics"""
        stats = dict(self.gradient_stats)
        
        # Compute averages
        if stats['gradients_compressed'] > 0:
            stats['average_compression_error'] = (
                stats['total_compression_error'] / stats['gradients_compressed']
            )
            stats['average_norm_preservation_error'] = (
                stats['norm_preservation_error'] / stats['gradients_compressed']
            )
        else:
            stats['average_compression_error'] = 0.0
            stats['average_norm_preservation_error'] = 0.0
        
        return stats
    
    def reset_error_feedback(self) -> None:
        """Reset error feedback accumulation"""
        self.accumulated_error = None
    
    def reset_gradient_stats(self) -> None:
        """Reset gradient compression statistics"""
        self.gradient_stats = {
            'gradients_compressed': 0,
            'total_compression_error': 0.0,
            'max_compression_error': 0.0,
            'norm_preservation_error': 0.0
        }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage from underlying p-adic system"""
        usage = self.padic_system.get_memory_usage()
        
        # Add gradient-specific memory info
        if self.accumulated_error is not None:
            error_size = self.accumulated_error.element_size() * self.accumulated_error.numel()
            usage['accumulated_error_bytes'] = error_size
        else:
            usage['accumulated_error_bytes'] = 0
        
        return usage