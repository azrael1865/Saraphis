"""
GAC Basic Gradient Bounder
Basic gradient bounding functionality for the GAC Direction Components system
"""

import torch
import numpy as np
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from .gac_types import DirectionType, DirectionState
from .direction_state import DirectionStateManager
from .direction_validator import DirectionValidator

logger = logging.getLogger(__name__)


@dataclass
class BoundingResult:
    """Result of gradient bounding operation"""
    bounded_gradients: torch.Tensor
    applied_factor: float
    bounding_type: str
    metadata: Dict[str, Any]


class BasicGradientBoundingError(Exception):
    """Raised when basic gradient bounding fails"""
    pass


class BasicGradientBounder:
    """Basic gradient bounding with fundamental clipping and scaling operations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
            
        # Configuration
        self.clip_value = config.get('clip_value', 10.0)
        # If max_norm not specified, default based on clip_value for consistency
        default_max_norm = 1.0 if 'clip_value' not in config else self.clip_value * 2.0
        self.max_norm = config.get('max_norm', default_max_norm)
        self.min_norm = config.get('min_norm', 1e-8)
        self.enable_adaptive_scaling = config.get('enable_adaptive_scaling', True)
        self.scaling_factor = config.get('scaling_factor', 1.0)
        
        # Enable/disable specific bounding methods
        self.enable_norm_clipping = config.get('enable_norm_clipping', True)
        self.enable_value_clipping = config.get('enable_value_clipping', True)
        
        # Norm calculation method
        self.norm_type = config.get('norm_type', 2)  # L2 norm by default
        self.norm_dim = config.get('norm_dim', None)  # None = all dimensions
        
        # Safety thresholds
        self.explosion_threshold = config.get('explosion_threshold', 100.0)
        self.vanishing_threshold = config.get('vanishing_threshold', 1e-10)
        
        # Statistics
        self.total_bounds = 0
        self.norm_clips = 0
        self.value_clips = 0
        self.scaling_applications = 0
        self.explosion_detections = 0
        self.vanishing_detections = 0
        
        # Performance tracking
        self.last_bound_time = 0.0
        self.total_bound_time = 0.0
        
    def bound_gradients(self, gradients: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> BoundingResult:
        """Apply basic gradient bounding"""
        if gradients is None:
            raise BasicGradientBoundingError("Gradients cannot be None")
        if not isinstance(gradients, torch.Tensor):
            raise TypeError("Gradients must be torch.Tensor")
        if gradients.numel() == 0:
            raise BasicGradientBoundingError("Gradients tensor cannot be empty")
            
        if context is None:
            context = {}
            
        start_time = time.time()
        self.total_bounds += 1
        
        # Make a copy to avoid modifying original
        bounded_gradients = gradients.clone()
        applied_operations = []
        bounding_metadata = {}
        
        # Step 1: Check for gradient explosion
        explosion_check = self._check_gradient_explosion(bounded_gradients)
        if explosion_check['detected']:
            logger.warning(f"Gradient explosion detected: {explosion_check['max_value']}")
            bounded_gradients = self._handle_gradient_explosion(bounded_gradients)
            applied_operations.append('explosion_handling')
            bounding_metadata['explosion'] = explosion_check
            self.explosion_detections += 1
        
        # Step 2: Check for vanishing gradients
        vanishing_check = self._check_vanishing_gradients(bounded_gradients)
        if vanishing_check['detected']:
            logger.debug(f"Vanishing gradients detected: {vanishing_check['norm']}")
            bounded_gradients = self._handle_vanishing_gradients(bounded_gradients)
            applied_operations.append('vanishing_handling')
            bounding_metadata['vanishing'] = vanishing_check
            self.vanishing_detections += 1
        
        # Step 3: Apply value clipping first (element-wise bounds)
        if self.enable_value_clipping:
            value_result = self._apply_value_clipping(bounded_gradients)
            if value_result['clipped']:
                bounded_gradients = value_result['gradients']
                applied_operations.append('value_clipping')
                bounding_metadata['value_clipping'] = value_result
                self.value_clips += 1
        
        # Step 4: Apply norm clipping (global constraint)
        if self.enable_norm_clipping:
            norm_result = self._apply_norm_clipping(bounded_gradients)
            if norm_result['clipped']:
                bounded_gradients = norm_result['gradients']
                applied_operations.append('norm_clipping')
                bounding_metadata['norm_clipping'] = norm_result
                self.norm_clips += 1
        
        # Step 5: Apply adaptive scaling if enabled
        scaling_result = None
        if self.enable_adaptive_scaling:
            scaling_result = self._apply_adaptive_scaling(bounded_gradients, context)
            if scaling_result['scaled']:
                bounded_gradients = scaling_result['gradients']
                applied_operations.append('adaptive_scaling')
                bounding_metadata['adaptive_scaling'] = scaling_result
                self.scaling_applications += 1
        
        # Calculate final metrics
        original_norm = torch.norm(gradients, p=self.norm_type).item()
        final_norm = torch.norm(bounded_gradients, p=self.norm_type).item()
        applied_factor = final_norm / (original_norm + 1e-12)
        
        # Record timing
        end_time = time.time()
        bound_time = end_time - start_time
        self.last_bound_time = bound_time
        self.total_bound_time += bound_time
        
        # Create result
        result = BoundingResult(
            bounded_gradients=bounded_gradients,
            applied_factor=applied_factor,
            bounding_type="basic",
            metadata={
                'original_norm': original_norm,
                'final_norm': final_norm,
                'applied_operations': applied_operations,
                'bounding_details': bounding_metadata,
                'processing_time': bound_time,
                'context': context
            }
        )
        
        return result
    
    def _check_gradient_explosion(self, gradients: torch.Tensor) -> Dict[str, Any]:
        """Check for gradient explosion"""
        max_value = torch.max(torch.abs(gradients)).item()
        norm_value = torch.norm(gradients, p=self.norm_type).item()
        
        explosion_detected = (
            max_value > self.explosion_threshold or 
            norm_value > self.explosion_threshold * 10
        )
        
        return {
            'detected': explosion_detected,
            'max_value': max_value,
            'norm_value': norm_value,
            'threshold': self.explosion_threshold
        }
    
    def _handle_gradient_explosion(self, gradients: torch.Tensor) -> torch.Tensor:
        """Handle gradient explosion through aggressive clipping"""
        # Emergency clipping to prevent NaN propagation
        emergency_clip = self.explosion_threshold * 0.1
        clipped_gradients = torch.clamp(gradients, -emergency_clip, emergency_clip)
        
        # Additional norm-based scaling
        current_norm = torch.norm(clipped_gradients, p=self.norm_type)
        if current_norm > emergency_clip:
            scale_factor = emergency_clip / current_norm
            clipped_gradients = clipped_gradients * scale_factor
        
        return clipped_gradients
    
    def _check_vanishing_gradients(self, gradients: torch.Tensor) -> Dict[str, Any]:
        """Check for vanishing gradients"""
        norm_value = torch.norm(gradients, p=self.norm_type).item()
        max_abs_value = torch.max(torch.abs(gradients)).item()
        
        vanishing_detected = (
            norm_value < self.vanishing_threshold or
            max_abs_value < self.vanishing_threshold * 10
        )
        
        return {
            'detected': vanishing_detected,
            'norm': norm_value,
            'max_abs_value': max_abs_value,
            'threshold': self.vanishing_threshold
        }
    
    def _handle_vanishing_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        """Handle vanishing gradients through scaling"""
        current_norm = torch.norm(gradients, p=self.norm_type)
        
        if current_norm < self.vanishing_threshold:
            # Scale up to minimum viable norm
            target_norm = self.min_norm
            scale_factor = target_norm / (current_norm + 1e-12)
            # Cap the scaling to prevent overcorrection - increase limit for very small gradients
            scale_factor = min(scale_factor, 1e8)  # Allow much larger scaling for vanishing gradients
            return gradients * scale_factor
        
        return gradients
    
    def _apply_norm_clipping(self, gradients: torch.Tensor) -> Dict[str, Any]:
        """Apply gradient norm clipping"""
        current_norm = torch.norm(gradients, p=self.norm_type, dim=self.norm_dim, keepdim=True)
        
        # Check if clipping is needed
        if self.norm_dim is None:
            # Global norm clipping
            max_norm_exceeded = current_norm.item() > self.max_norm
            clipped_gradients = gradients
            
            if max_norm_exceeded:
                scale_factor = self.max_norm / current_norm
                clipped_gradients = gradients * scale_factor
        else:
            # Per-dimension norm clipping
            max_norm_tensor = torch.full_like(current_norm, self.max_norm)
            clip_coef = torch.minimum(max_norm_tensor / (current_norm + 1e-6), 
                                     torch.ones_like(current_norm))
            clipped_gradients = gradients * clip_coef
            max_norm_exceeded = torch.any(current_norm > self.max_norm).item()
        
        return {
            'clipped': max_norm_exceeded,
            'gradients': clipped_gradients,
            'original_norm': current_norm.item() if self.norm_dim is None else current_norm.mean().item(),
            'final_norm': torch.norm(clipped_gradients, p=self.norm_type).item(),
            'clip_threshold': self.max_norm
        }
    
    def _apply_value_clipping(self, gradients: torch.Tensor) -> Dict[str, Any]:
        """Apply gradient value clipping"""
        original_max = torch.max(torch.abs(gradients)).item()
        clipped_gradients = torch.clamp(gradients, -self.clip_value, self.clip_value)
        final_max = torch.max(torch.abs(clipped_gradients)).item()
        
        value_clipped = original_max > self.clip_value
        
        return {
            'clipped': value_clipped,
            'gradients': clipped_gradients,
            'original_max': original_max,
            'final_max': final_max,
            'clip_value': self.clip_value
        }
    
    def _apply_adaptive_scaling(self, gradients: torch.Tensor, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply adaptive scaling based on context"""
        scale_factor = self.scaling_factor
        scaling_applied = False
        scaling_reason = "none"
        
        # Context-based scaling adjustments
        if 'learning_rate' in context:
            lr = context['learning_rate']
            # Scale down gradients for high learning rates
            if lr > 0.01:
                lr_scale = min(1.0, 0.01 / lr)
                scale_factor *= lr_scale
                scaling_applied = True
                scaling_reason = "high_learning_rate"
        
        if 'iteration' in context:
            iteration = context['iteration']
            # Gradual scaling adjustment for early iterations
            if iteration < 100:
                warmup_scale = min(1.0, iteration / 100.0)
                scale_factor *= (0.5 + 0.5 * warmup_scale)
                scaling_applied = True
                scaling_reason = "iteration_warmup"
        
        if 'loss' in context:
            loss = context['loss']
            # Scale based on loss magnitude
            if loss > 10.0:
                loss_scale = min(1.0, 10.0 / loss)
                scale_factor *= loss_scale
                scaling_applied = True
                scaling_reason = "high_loss"
        
        # Apply scaling only if different from 1.0
        if abs(scale_factor - 1.0) > 1e-6:
            scaled_gradients = gradients * scale_factor
            scaling_applied = True
        else:
            scaled_gradients = gradients
        
        return {
            'scaled': scaling_applied,
            'gradients': scaled_gradients,
            'scale_factor': scale_factor,
            'scaling_reason': scaling_reason,
            'original_norm': torch.norm(gradients, p=self.norm_type).item(),
            'final_norm': torch.norm(scaled_gradients, p=self.norm_type).item()
        }
    
    def get_bounding_statistics(self) -> Dict[str, Any]:
        """Get comprehensive bounding statistics"""
        if self.total_bounds == 0:
            return {"status": "no_operations"}
        
        return {
            "operation_counts": {
                "total_bounds": self.total_bounds,
                "norm_clips": self.norm_clips,
                "value_clips": self.value_clips,
                "scaling_applications": self.scaling_applications,
                "explosion_detections": self.explosion_detections,
                "vanishing_detections": self.vanishing_detections
            },
            "operation_rates": {
                "norm_clip_rate": self.norm_clips / self.total_bounds,
                "value_clip_rate": self.value_clips / self.total_bounds,
                "scaling_rate": self.scaling_applications / self.total_bounds,
                "explosion_rate": self.explosion_detections / self.total_bounds,
                "vanishing_rate": self.vanishing_detections / self.total_bounds
            },
            "performance": {
                "total_time": self.total_bound_time,
                "average_time": self.total_bound_time / self.total_bounds,
                "last_operation_time": self.last_bound_time
            },
            "configuration": {
                "max_norm": self.max_norm,
                "min_norm": self.min_norm,
                "clip_value": self.clip_value,
                "norm_type": self.norm_type,
                "adaptive_scaling_enabled": self.enable_adaptive_scaling
            }
        }
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.total_bounds = 0
        self.norm_clips = 0
        self.value_clips = 0
        self.scaling_applications = 0
        self.explosion_detections = 0
        self.vanishing_detections = 0
        self.last_bound_time = 0.0
        self.total_bound_time = 0.0
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration parameters"""
        for key, value in new_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated {key} to {value}")
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
    
    def is_gradient_healthy(self, gradients: torch.Tensor) -> bool:
        """Quick check if gradients are in healthy range"""
        if gradients is None or gradients.numel() == 0:
            return False
        
        norm = torch.norm(gradients, p=self.norm_type).item()
        max_val = torch.max(torch.abs(gradients)).item()
        
        # Check for NaN or infinite values
        if torch.isnan(gradients).any() or torch.isinf(gradients).any():
            return False
        
        # Check if within reasonable bounds
        if norm > self.explosion_threshold or norm < self.vanishing_threshold:
            return False
        
        if max_val > self.explosion_threshold:
            return False
        
        return True
    
    def get_recommended_bounds(self, gradients: torch.Tensor) -> Dict[str, float]:
        """Get recommended bounding parameters based on gradient characteristics"""
        if gradients is None or gradients.numel() == 0:
            return {}
        
        norm = torch.norm(gradients, p=self.norm_type).item()
        max_val = torch.max(torch.abs(gradients)).item()
        std_val = torch.std(gradients).item()
        
        # Calculate recommended max_norm (slightly above current norm)
        recommended_max_norm = min(norm * 1.2, self.max_norm * 2)
        
        # Calculate recommended clip_value (based on distribution)
        recommended_clip_value = max_val * 1.1
        
        # Calculate recommended scaling factor
        if norm > 0:
            recommended_scaling = min(1.0, self.max_norm / norm)
        else:
            recommended_scaling = 1.0
        
        return {
            "max_norm": recommended_max_norm,
            "clip_value": recommended_clip_value,
            "scaling_factor": recommended_scaling,
            "current_norm": norm,
            "current_max": max_val,
            "current_std": std_val
        }