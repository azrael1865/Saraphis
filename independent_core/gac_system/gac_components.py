"""
GAC System Components
Example component implementations for the Gradient Ascent Clipping system
"""

import asyncio
import logging
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

from .gradient_ascent_clipping import GACComponent, ComponentState, EventType, ComponentMetrics

logger = logging.getLogger(__name__)

class GradientClippingComponent(GACComponent):
    """Basic gradient clipping component with adaptive thresholds"""
    
    def __init__(self, component_id: str = "gradient_clipper", config: Dict[str, Any] = None):
        super().__init__(component_id, config)
        self.clip_value = self.config.get('clip_value', 1.0)
        self.adaptive_clipping = self.config.get('adaptive_clipping', True)
        self.clip_history = []
        self.adaptation_rate = self.config.get('adaptation_rate', 0.01)
        
        self.pid_state.setpoint = self.clip_value
        self.pid_state.kp = self.config.get('pid_kp', 1.0)
        self.pid_state.ki = self.config.get('pid_ki', 0.1)
        self.pid_state.kd = self.config.get('pid_kd', 0.05)
    
    async def process_gradient(self, gradient: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        start_time = time.time()
        
        try:
            gradient_norm = torch.norm(gradient).item()
            self.clip_history.append(gradient_norm)
            
            if len(self.clip_history) > 1000:
                self.clip_history = self.clip_history[-1000:]
            
            if self.adaptive_clipping and len(self.clip_history) > 10:
                avg_norm = np.mean(self.clip_history[-10:])
                pid_output = self.calculate_pid_output(avg_norm)
                adaptive_clip_value = max(0.1, self.clip_value + pid_output * self.adaptation_rate)
            else:
                adaptive_clip_value = self.clip_value
            
            if gradient_norm > adaptive_clip_value:
                clipped_gradient = gradient * (adaptive_clip_value / gradient_norm)
                
                self.emit_event(EventType.GRADIENT_UPDATE, {
                    "original_norm": gradient_norm,
                    "clipped_norm": adaptive_clip_value,
                    "clip_ratio": adaptive_clip_value / gradient_norm,
                    "adaptive_clip_value": adaptive_clip_value
                })
            else:
                clipped_gradient = gradient
            
            self.last_gradient = clipped_gradient
            self.metrics.total_processing_time += time.time() - start_time
            
            if gradient_norm > self.clip_value * 2:
                self.metrics.success_rate = max(0.0, self.metrics.success_rate - 0.01)
            else:
                self.metrics.success_rate = min(1.0, self.metrics.success_rate + 0.001)
            
            return clipped_gradient
            
        except Exception as e:
            logger.error(f"Gradient clipping error: {e}")
            self.metrics.error_count += 1
            return gradient
    
    def get_component_info(self) -> Dict[str, Any]:
        avg_gradient_norm = np.mean(self.clip_history) if self.clip_history else 0.0
        return {
            "component_type": "gradient_clipper",
            "clip_value": self.clip_value,
            "adaptive_clipping": self.adaptive_clipping,
            "average_gradient_norm": avg_gradient_norm,
            "total_gradients_processed": len(self.clip_history),
            "current_pid_output": self.calculate_pid_output(avg_gradient_norm) if self.clip_history else 0.0
        }

class GradientNoiseComponent(GACComponent):
    """Component that adds controlled noise to gradients for regularization"""
    
    def __init__(self, component_id: str = "gradient_noise", config: Dict[str, Any] = None):
        super().__init__(component_id, config)
        self.noise_scale = self.config.get('noise_scale', 0.01)
        self.adaptive_noise = self.config.get('adaptive_noise', True)
        self.noise_decay = self.config.get('noise_decay', 0.999)
        self.min_noise_scale = self.config.get('min_noise_scale', 0.001)
        
    async def process_gradient(self, gradient: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        start_time = time.time()
        
        try:
            if self.adaptive_noise:
                gradient_norm = torch.norm(gradient).item()
                if gradient_norm > 1.0:
                    current_noise_scale = self.noise_scale * (1.0 / gradient_norm)
                else:
                    current_noise_scale = self.noise_scale
            else:
                current_noise_scale = self.noise_scale
            
            noise = torch.randn_like(gradient) * current_noise_scale
            noisy_gradient = gradient + noise
            
            if self.adaptive_noise:
                self.noise_scale = max(self.min_noise_scale, self.noise_scale * self.noise_decay)
            
            self.emit_event(EventType.GRADIENT_UPDATE, {
                "noise_scale": current_noise_scale,
                "noise_norm": torch.norm(noise).item(),
                "gradient_norm": torch.norm(gradient).item()
            })
            
            self.metrics.total_processing_time += time.time() - start_time
            self.metrics.success_rate = min(1.0, self.metrics.success_rate + 0.0001)
            
            return noisy_gradient
            
        except Exception as e:
            logger.error(f"Gradient noise error: {e}")
            self.metrics.error_count += 1
            return gradient
    
    def get_component_info(self) -> Dict[str, Any]:
        return {
            "component_type": "gradient_noise",
            "noise_scale": self.noise_scale,
            "adaptive_noise": self.adaptive_noise,
            "noise_decay": self.noise_decay,
            "min_noise_scale": self.min_noise_scale
        }

class GradientNormalizationComponent(GACComponent):
    """Component that normalizes gradients based on running statistics"""
    
    def __init__(self, component_id: str = "gradient_normalizer", config: Dict[str, Any] = None):
        super().__init__(component_id, config)
        self.momentum = self.config.get('momentum', 0.9)
        self.eps = self.config.get('eps', 1e-8)
        self.running_mean = None
        self.running_var = None
        self.step_count = 0
        
    async def process_gradient(self, gradient: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        start_time = time.time()
        
        try:
            if self.running_mean is None:
                self.running_mean = torch.zeros_like(gradient)
                self.running_var = torch.ones_like(gradient)
            
            self.step_count += 1
            
            # Update running statistics
            with torch.no_grad():
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * gradient
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * (gradient - self.running_mean) ** 2
            
            # Bias correction
            bias_correction1 = 1 - self.momentum ** self.step_count
            bias_correction2 = 1 - self.momentum ** self.step_count
            
            corrected_mean = self.running_mean / bias_correction1
            corrected_var = self.running_var / bias_correction2
            
            # Normalize gradient
            normalized_gradient = (gradient - corrected_mean) / (torch.sqrt(corrected_var) + self.eps)
            
            self.emit_event(EventType.GRADIENT_UPDATE, {
                "gradient_norm": torch.norm(gradient).item(),
                "normalized_norm": torch.norm(normalized_gradient).item(),
                "running_mean_norm": torch.norm(corrected_mean).item(),
                "running_var_mean": torch.mean(corrected_var).item()
            })
            
            self.metrics.total_processing_time += time.time() - start_time
            self.metrics.success_rate = min(1.0, self.metrics.success_rate + 0.0001)
            
            return normalized_gradient
            
        except Exception as e:
            logger.error(f"Gradient normalization error: {e}")
            self.metrics.error_count += 1
            return gradient
    
    def get_component_info(self) -> Dict[str, Any]:
        return {
            "component_type": "gradient_normalizer",
            "momentum": self.momentum,
            "eps": self.eps,
            "step_count": self.step_count,
            "has_running_stats": self.running_mean is not None
        }

class GradientAccumulationComponent(GACComponent):
    """Component that accumulates gradients over multiple steps"""
    
    def __init__(self, component_id: str = "gradient_accumulator", config: Dict[str, Any] = None):
        super().__init__(component_id, config)
        self.accumulation_steps = self.config.get('accumulation_steps', 4)
        self.accumulated_gradient = None
        self.current_step = 0
        self.normalize_accumulated = self.config.get('normalize_accumulated', True)
        
    async def process_gradient(self, gradient: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        start_time = time.time()
        
        try:
            if self.accumulated_gradient is None:
                self.accumulated_gradient = torch.zeros_like(gradient)
            
            self.accumulated_gradient += gradient
            self.current_step += 1
            
            if self.current_step >= self.accumulation_steps:
                if self.normalize_accumulated:
                    output_gradient = self.accumulated_gradient / self.accumulation_steps
                else:
                    output_gradient = self.accumulated_gradient.clone()
                
                self.emit_event(EventType.GRADIENT_UPDATE, {
                    "accumulated_steps": self.current_step,
                    "original_norm": torch.norm(gradient).item(),
                    "accumulated_norm": torch.norm(output_gradient).item(),
                    "accumulation_ratio": torch.norm(output_gradient).item() / torch.norm(gradient).item()
                })
                
                # Reset accumulation
                self.accumulated_gradient.zero_()
                self.current_step = 0
                
                self.metrics.total_processing_time += time.time() - start_time
                self.metrics.success_rate = min(1.0, self.metrics.success_rate + 0.001)
                
                return output_gradient
            else:
                # Return zero gradient until accumulation is complete
                return torch.zeros_like(gradient)
                
        except Exception as e:
            logger.error(f"Gradient accumulation error: {e}")
            self.metrics.error_count += 1
            return gradient
    
    def get_component_info(self) -> Dict[str, Any]:
        return {
            "component_type": "gradient_accumulator",
            "accumulation_steps": self.accumulation_steps,
            "current_step": self.current_step,
            "normalize_accumulated": self.normalize_accumulated,
            "has_accumulated_gradient": self.accumulated_gradient is not None
        }

class GradientCompressionComponent(GACComponent):
    """Component that compresses gradients using top-k sparsification"""
    
    def __init__(self, component_id: str = "gradient_compressor", config: Dict[str, Any] = None):
        super().__init__(component_id, config)
        self.compression_ratio = self.config.get('compression_ratio', 0.1)
        self.adaptive_compression = self.config.get('adaptive_compression', True)
        self.error_feedback = self.config.get('error_feedback', True)
        self.compression_error = None
        
    async def process_gradient(self, gradient: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        start_time = time.time()
        
        try:
            # Add previous compression error if using error feedback
            if self.error_feedback and self.compression_error is not None:
                gradient = gradient + self.compression_error
            
            # Determine number of elements to keep
            total_elements = gradient.numel()
            k = max(1, int(total_elements * self.compression_ratio))
            
            # Flatten gradient for top-k selection
            flat_gradient = gradient.view(-1)
            
            # Get top-k elements by magnitude
            _, top_k_indices = torch.topk(torch.abs(flat_gradient), k)
            
            # Create sparse gradient
            compressed_gradient = torch.zeros_like(flat_gradient)
            compressed_gradient[top_k_indices] = flat_gradient[top_k_indices]
            
            # Reshape back to original shape
            compressed_gradient = compressed_gradient.view(gradient.shape)
            
            # Calculate compression error for feedback
            if self.error_feedback:
                self.compression_error = gradient - compressed_gradient
            
            compression_ratio_actual = k / total_elements
            
            self.emit_event(EventType.GRADIENT_UPDATE, {
                "original_norm": torch.norm(gradient).item(),
                "compressed_norm": torch.norm(compressed_gradient).item(),
                "compression_ratio": compression_ratio_actual,
                "elements_kept": k,
                "total_elements": total_elements
            })
            
            self.metrics.total_processing_time += time.time() - start_time
            self.metrics.success_rate = min(1.0, self.metrics.success_rate + 0.0001)
            
            return compressed_gradient
            
        except Exception as e:
            logger.error(f"Gradient compression error: {e}")
            self.metrics.error_count += 1
            return gradient
    
    def get_component_info(self) -> Dict[str, Any]:
        return {
            "component_type": "gradient_compressor",
            "compression_ratio": self.compression_ratio,
            "adaptive_compression": self.adaptive_compression,
            "error_feedback": self.error_feedback,
            "has_compression_error": self.compression_error is not None
        }

class GradientMonitoringComponent(GACComponent):
    """Component that monitors gradient statistics and health"""
    
    def __init__(self, component_id: str = "gradient_monitor", config: Dict[str, Any] = None):
        super().__init__(component_id, config)
        self.statistics_window = self.config.get('statistics_window', 100)
        self.gradient_history = []
        self.norm_history = []
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'explosion_threshold': 10.0,
            'vanishing_threshold': 1e-6,
            'instability_threshold': 5.0
        })
        
    async def process_gradient(self, gradient: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        start_time = time.time()
        
        try:
            gradient_norm = torch.norm(gradient).item()
            gradient_mean = torch.mean(gradient).item()
            gradient_std = torch.std(gradient).item()
            
            self.norm_history.append(gradient_norm)
            if len(self.norm_history) > self.statistics_window:
                self.norm_history = self.norm_history[-self.statistics_window:]
            
            # Calculate statistics
            if len(self.norm_history) > 1:
                recent_norms = self.norm_history[-10:] if len(self.norm_history) >= 10 else self.norm_history
                avg_norm = np.mean(recent_norms)
                norm_std = np.std(recent_norms)
                norm_trend = np.mean(np.diff(recent_norms)) if len(recent_norms) > 1 else 0.0
            else:
                avg_norm = gradient_norm
                norm_std = 0.0
                norm_trend = 0.0
            
            # Check for alerts
            alerts = []
            if gradient_norm > self.alert_thresholds['explosion_threshold']:
                alerts.append("gradient_explosion")
            if gradient_norm < self.alert_thresholds['vanishing_threshold']:
                alerts.append("gradient_vanishing")
            if norm_std > self.alert_thresholds['instability_threshold']:
                alerts.append("gradient_instability")
            
            # Emit monitoring event
            self.emit_event(EventType.PERFORMANCE_METRIC, {
                "gradient_norm": gradient_norm,
                "gradient_mean": gradient_mean,
                "gradient_std": gradient_std,
                "average_norm": avg_norm,
                "norm_std": norm_std,
                "norm_trend": norm_trend,
                "alerts": alerts,
                "window_size": len(self.norm_history)
            })
            
            # Emit alerts if any
            for alert in alerts:
                self.emit_event(EventType.SYSTEM_ALERT, {
                    "alert_type": alert,
                    "gradient_norm": gradient_norm,
                    "threshold": self.alert_thresholds.get(alert.replace('gradient_', '') + '_threshold', 0),
                    "severity": "high" if alert == "gradient_explosion" else "medium"
                })
            
            self.metrics.total_processing_time += time.time() - start_time
            self.metrics.success_rate = 1.0 if not alerts else max(0.5, self.metrics.success_rate - 0.01)
            
            # Monitoring component passes gradient through unchanged
            return gradient
            
        except Exception as e:
            logger.error(f"Gradient monitoring error: {e}")
            self.metrics.error_count += 1
            return gradient
    
    def get_component_info(self) -> Dict[str, Any]:
        avg_norm = np.mean(self.norm_history) if self.norm_history else 0.0
        return {
            "component_type": "gradient_monitor",
            "statistics_window": self.statistics_window,
            "gradients_monitored": len(self.norm_history),
            "average_gradient_norm": avg_norm,
            "alert_thresholds": self.alert_thresholds,
            "current_health": "healthy" if self.metrics.success_rate > 0.8 else "degraded"
        }

def create_default_components() -> List[GACComponent]:
    """Create a default set of GAC components"""
    return [
        GradientMonitoringComponent("monitor", {"statistics_window": 100}),
        GradientClippingComponent("clipper", {"clip_value": 1.0, "adaptive_clipping": True}),
        GradientNoiseComponent("noise", {"noise_scale": 0.01, "adaptive_noise": True}),
        GradientNormalizationComponent("normalizer", {"momentum": 0.9})
    ]

def create_production_components() -> List[GACComponent]:
    """Create a production-ready set of GAC components"""
    return [
        GradientMonitoringComponent("production_monitor", {
            "statistics_window": 1000,
            "alert_thresholds": {
                "explosion_threshold": 5.0,
                "vanishing_threshold": 1e-7,
                "instability_threshold": 3.0
            }
        }),
        GradientClippingComponent("production_clipper", {
            "clip_value": 0.5,
            "adaptive_clipping": True,
            "adaptation_rate": 0.005
        }),
        GradientCompressionComponent("production_compressor", {
            "compression_ratio": 0.1,
            "error_feedback": True
        })
    ]

if __name__ == "__main__":
    # Example usage
    async def test_components():
        components = create_default_components()
        test_gradient = torch.randn(100, 100) * 2.0  # Large gradient for testing
        
        print(f"Original gradient norm: {torch.norm(test_gradient).item():.4f}")
        
        processed_gradient = test_gradient
        for component in components:
            component.update_state(ComponentState.ACTIVE)
            processed_gradient = await component.process_gradient(processed_gradient, {})
            print(f"{component.component_id}: {torch.norm(processed_gradient).item():.4f}")
            print(f"  Info: {component.get_component_info()}")
        
        print(f"Final gradient norm: {torch.norm(processed_gradient).item():.4f}")
    
    asyncio.run(test_components())