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
    output_min: float = -100.0  # Minimum output value
    output_max: float = 100.0  # Maximum output value
    windup_limit: float = 100.0  # Anti-windup limit
    sample_time: float = 0.0  # Sample time in seconds (0 = no limit)
    auto_tune: bool = True  # Enable auto-tuning

class PIDController:
    """
    PID Controller for adaptive gradient threshold management
    
    This controller automatically adjusts gradient clipping thresholds
    based on system performance and gradient behavior patterns.
    """
    
    def __init__(self, config: PIDControllerConfig = None, **kwargs):
        # Allow both config object and direct parameters
        if config is not None:
            self.config = config
        else:
            # Support direct parameter passing for backward compatibility
            self.config = PIDControllerConfig()
            
            # Map parameters
            if 'kp' in kwargs:
                self.config.kp = kwargs['kp']
            if 'ki' in kwargs:
                self.config.ki = kwargs['ki']
            if 'kd' in kwargs:
                self.config.kd = kwargs['kd']
            if 'setpoint' in kwargs:
                self.config.setpoint = kwargs['setpoint']
            
            # Handle output_limits as tuple
            if 'output_limits' in kwargs:
                limits = kwargs['output_limits']
                if isinstance(limits, (tuple, list)) and len(limits) == 2:
                    self.config.output_min = limits[0]
                    self.config.output_max = limits[1]
            
            # Handle individual output limits
            if 'output_min' in kwargs:
                self.config.output_min = kwargs['output_min']
            if 'output_max' in kwargs:
                self.config.output_max = kwargs['output_max']
            if 'windup_limit' in kwargs:
                self.config.windup_limit = kwargs['windup_limit']
            if 'sample_time' in kwargs:
                self.config.sample_time = kwargs['sample_time']
            if 'auto_tune' in kwargs:
                self.config.auto_tune = kwargs['auto_tune']
        
        # PID state variables
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = None  # None indicates first update
        self.last_output = self.config.setpoint  # Initialize to setpoint for smooth startup
        
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
        
        # Handle first update
        first_update = False
        if self.last_time is None:
            self.last_time = current_time
            dt = self.config.sample_time if self.config.sample_time > 0 else 0.001  # Use sample_time or small value for first update
            first_update = True
        else:
            dt = current_time - self.last_time
            
            # Skip update if sample time hasn't elapsed (unless sample_time is 0 which disables this check)
            if self.config.sample_time > 0 and dt < self.config.sample_time:
                return self.last_output
            
            # Ensure minimum dt for derivative calculation
            if dt < 0.0001:  # Less than 0.1ms is effectively instantaneous
                dt = 0.0001
        
        # Use provided target or default setpoint
        setpoint = target_value if target_value is not None else self.config.setpoint
        
        # Calculate error
        error = setpoint - measured_value
        
        # Proportional term
        proportional = self.config.kp * error
        
        # Integral term with anti-windup
        # Only integrate if we're not saturated or if error is reducing saturation
        temp_integral = self.integral + error * dt
        
        # Check if integral would cause saturation
        temp_output = proportional + self.config.ki * temp_integral
        if self.config.output_min <= temp_output <= self.config.output_max:
            # Not saturated, update integral
            self.integral = temp_integral
        elif (temp_output > self.config.output_max and error < 0) or \
             (temp_output < self.config.output_min and error > 0):
            # Error is reducing saturation, allow integral update
            self.integral = temp_integral
        
        # Apply windup limit
        if abs(self.integral) > self.config.windup_limit:
            self.integral = self.config.windup_limit * (1 if self.integral > 0 else -1)
        
        integral = self.config.ki * self.integral
        
        # Derivative term with filtering
        # On first update, use 0 derivative to avoid spike
        if first_update:
            derivative = 0.0
        else:
            derivative = self.config.kd * (error - self.last_error) / dt
        
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
        self.last_time = None  # Reset to None for proper first update handling
        self.last_output = self.config.setpoint  # Reset to setpoint
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

# Create alias for backward compatibility
GACPIDController = PIDController

__all__ = ['PIDController', 'PIDControllerConfig', 'create_gradient_threshold_pid', 'GACPIDController']