# Gradient Ascent Clipping (GAC) PID Controller
# PID controller implementation for GAC system
# Part of the Saraphis recursive methodology

"""
GAC PID Controller - Adaptive threshold management using PID control

This module implements a PID (Proportional-Integral-Derivative) controller
for the GAC system, enabling autonomous threshold adjustment based on
gradient behavior and system performance.

Features:
- Adaptive threshold tuning
- Gradient pattern learning
- Performance-based adjustments
- Anti-windup protection
- Self-tuning capabilities
"""

import time
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class PIDControllerConfig:
    """Configuration for PID controller"""
    kp: float = 1.0  # Proportional gain
    ki: float = 0.1  # Integral gain  
    kd: float = 0.05  # Derivative gain
    setpoint: float = 0.0  # Target value
    output_min: float = 0.1  # Minimum output value
    output_max: float = 10.0  # Maximum output value
    windup_limit: float = 100.0  # Anti-windup limit
    sample_time: float = 0.1  # Sample time in seconds
    auto_tune: bool = True  # Enable auto-tuning

class PIDController:
    """
    PID Controller for adaptive gradient threshold management
    
    This controller automatically adjusts gradient clipping thresholds
    based on system performance and gradient behavior patterns.
    """
    
    def __init__(self, config: PIDControllerConfig = None):
        self.config = config or PIDControllerConfig()
        
        # PID state variables
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        self.last_output = self.config.setpoint
        
        # Performance tracking
        self.error_history = []
        self.output_history = []
        self.performance_metrics = {
            'total_updates': 0,
            'avg_error': 0.0,
            'stability_score': 1.0,
            'auto_tune_attempts': 0
        }
        
        logger.info(f"PID Controller initialized with Kp={self.config.kp}, Ki={self.config.ki}, Kd={self.config.kd}")
    
    def update(self, measured_value: float, target_value: float = None) -> float:
        """
        Update the PID controller with new measurement
        
        Args:
            measured_value: Current system measurement
            target_value: Optional target override
            
        Returns:
            PID controller output
        """
        current_time = time.time()
        dt = current_time - self.last_time
        
        # Use provided target or default setpoint
        setpoint = target_value if target_value is not None else self.config.setpoint
        
        # Calculate error
        error = setpoint - measured_value
        
        # Skip update if sample time hasn't elapsed
        if dt < self.config.sample_time:
            return self.last_output
        
        # Proportional term
        proportional = self.config.kp * error
        
        # Integral term with windup protection
        self.integral += error * dt
        if abs(self.integral) > self.config.windup_limit:
            self.integral = self.config.windup_limit * (1 if self.integral > 0 else -1)
        integral = self.config.ki * self.integral
        
        # Derivative term
        if dt > 0:
            derivative = self.config.kd * (error - self.last_error) / dt
        else:
            derivative = 0.0
        
        # Calculate output
        output = proportional + integral + derivative
        
        # Apply output limits
        output = max(self.config.output_min, min(self.config.output_max, output))
        
        # Update state
        self.last_error = error
        self.last_time = current_time
        self.last_output = output
        
        # Track performance
        self._update_performance_metrics(error, output)
        
        # Auto-tune if enabled
        if self.config.auto_tune and self.performance_metrics['total_updates'] % 100 == 0:
            self._auto_tune()
        
        return output
    
    def _update_performance_metrics(self, error: float, output: float):
        """Update internal performance metrics"""
        self.performance_metrics['total_updates'] += 1
        
        # Track error history (last 50 values)
        self.error_history.append(abs(error))
        if len(self.error_history) > 50:
            self.error_history.pop(0)
        
        # Track output history
        self.output_history.append(output)
        if len(self.output_history) > 50:
            self.output_history.pop(0)
        
        # Calculate average error
        if self.error_history:
            self.performance_metrics['avg_error'] = sum(self.error_history) / len(self.error_history)
        
        # Calculate stability score (lower variance = higher stability)
        if len(self.output_history) > 10:
            output_variance = sum((x - sum(self.output_history)/len(self.output_history))**2 
                                for x in self.output_history) / len(self.output_history)
            self.performance_metrics['stability_score'] = 1.0 / (1.0 + output_variance)
    
    def _auto_tune(self):
        """Simple auto-tuning based on performance metrics"""
        if len(self.error_history) < 20:
            return
        
        avg_error = self.performance_metrics['avg_error']
        stability = self.performance_metrics['stability_score']
        
        # If error is consistently high, increase proportional gain
        if avg_error > 1.0 and stability > 0.7:
            self.config.kp *= 1.1
            logger.debug(f"Auto-tune: Increased Kp to {self.config.kp:.3f}")
        
        # If system is unstable, reduce derivative gain
        elif stability < 0.3:
            self.config.kd *= 0.9
            logger.debug(f"Auto-tune: Reduced Kd to {self.config.kd:.3f}")
        
        # If error oscillates, adjust integral gain
        elif len(self.error_history) > 10:
            error_trend = sum(self.error_history[-5:]) - sum(self.error_history[-10:-5])
            if abs(error_trend) > 2.0:
                self.config.ki *= 0.95
                logger.debug(f"Auto-tune: Adjusted Ki to {self.config.ki:.3f}")
        
        self.performance_metrics['auto_tune_attempts'] += 1
    
    def reset(self):
        """Reset the PID controller state"""
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        self.error_history.clear()
        self.output_history.clear()
        logger.debug("PID Controller reset")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def set_tuning_parameters(self, kp: float = None, ki: float = None, kd: float = None):
        """Update PID tuning parameters"""
        if kp is not None:
            self.config.kp = kp
        if ki is not None:
            self.config.ki = ki
        if kd is not None:
            self.config.kd = kd
        
        logger.info(f"PID parameters updated: Kp={self.config.kp}, Ki={self.config.ki}, Kd={self.config.kd}")

# Factory function for easy PID controller creation
def create_gradient_threshold_pid(target_threshold: float = 1.0, 
                                aggressiveness: str = "moderate") -> PIDController:
    """
    Create a PID controller optimized for gradient threshold management
    
    Args:
        target_threshold: Target gradient threshold value
        aggressiveness: Control aggressiveness ("conservative", "moderate", "aggressive")
        
    Returns:
        Configured PID controller
    """
    if aggressiveness == "conservative":
        config = PIDControllerConfig(kp=0.5, ki=0.05, kd=0.02, setpoint=target_threshold)
    elif aggressiveness == "aggressive":
        config = PIDControllerConfig(kp=2.0, ki=0.2, kd=0.1, setpoint=target_threshold)
    else:  # moderate
        config = PIDControllerConfig(kp=1.0, ki=0.1, kd=0.05, setpoint=target_threshold)
    
    return PIDController(config)

__all__ = ['PIDController', 'PIDControllerConfig', 'create_gradient_threshold_pid']