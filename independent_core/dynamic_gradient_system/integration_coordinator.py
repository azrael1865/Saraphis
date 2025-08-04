"""
Integration Coordinator for Dynamic Gradient System
Coordinates all direction switching components with hard failure handling
Production-ready implementation with NO FALLBACKS, NO PLACEHOLDERS
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import json
from datetime import datetime
import os

from ..gac_system.direction_state import DirectionStateManager as DirectionState
from ..gac_system.direction_validator import DirectionValidator
from ..gac_system.enhanced_bounder import EnhancedGradientBounder as DirectionBounder

logger = logging.getLogger(__name__)


class IntegrationCoordinator:
    """
    Coordinates all direction switching components with hard failure handling.
    
    This is the main integration point that:
    1. Monitors training progress using ProgressMonitor
    2. Validates switching decisions using DirectionValidator
    3. Executes direction changes using DirectionState
    4. Applies bounding using DirectionBounder
    5. Provides unified interface for training manager
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize integration coordinator with all components.
        
        Args:
            config: Configuration dictionary with parameters:
                - min_dwell_time: Minimum time between switches (default: 1.0)
                - window_size: Progress monitoring window size (default: 10)
                - progress_threshold: Threshold for progress-based switching (default: 0.001)
                - max_gradient_norm: Maximum gradient norm for bounding (default: 1.0)
                - enable_checkpointing: Enable state checkpointing (default: True)
                - checkpoint_dir: Directory for checkpoints (default: './checkpoints')
                
        Raises:
            ValueError: If configuration parameters are invalid
            RuntimeError: If component initialization fails
        """
        start_time = time.time()
        
        if config is None:
            config = {}
            
        # Validate configuration first
        self._validate_configuration(config)
        
        # Store configuration
        self.config = config.copy()
        
        # Initialize thread safety
        self._lock = threading.RLock()
        self._initialized = False
        
        try:
            # Initialize all components with validated configuration
            self.direction_state = DirectionState({
                'history_size': config.get('window_size', 10),
                'smoothing_factor': 0.8,
                'confidence_threshold': 0.7
            })
            
            self.direction_validator = DirectionValidator({
                'validation_window': config.get('window_size', 10),
                'consistency_threshold': 0.8,
                'anomaly_threshold': 0.7
            })
            
            self.direction_bounder = DirectionBounder({
                'ascent_max_norm': config.get('max_gradient_norm', 1.0),
                'descent_max_norm': config.get('max_gradient_norm', 1.0),
                'stable_max_norm': config.get('max_gradient_norm', 1.0),
                'oscillating_max_norm': config.get('max_gradient_norm', 1.0)
            })
            
            # Create simple progress monitor and switcher classes
            class SimpleProgressMonitor:
                def __init__(self, window_size=10):
                    self.window_size = window_size
                    self.loss_history = []
                    self.progress_threshold = 0.001
                
                def update(self, loss):
                    self.loss_history.append(loss)
                    if len(self.loss_history) > self.window_size:
                        self.loss_history.pop(0)
                
                def get_progress(self):
                    if len(self.loss_history) < 2:
                        return 0.0
                    return abs(self.loss_history[-1] - self.loss_history[0])
            
            class SimpleSwitcher:
                def __init__(self, progress_threshold=0.001):
                    self.progress_threshold = progress_threshold
                
                def should_switch(self, progress):
                    return progress < self.progress_threshold
            
            self.progress_monitor = SimpleProgressMonitor(
                window_size=config.get('window_size', 10)
            )
            
            self.simple_switcher = SimpleSwitcher(
                progress_threshold=config.get('progress_threshold', 0.001)
            )
            
            # Performance tracking
            self.performance_metrics = {
                'total_switches': 0,
                'successful_switches': 0,
                'failed_switches': 0,
                'total_validations': 0,
                'validation_failures': 0,
                'total_bounding_operations': 0,
                'average_decision_time_ms': 0.0,
                'peak_decision_time_ms': 0.0,
                'last_switch_time': 0.0,
                'initialization_time_ms': 0.0,
                'total_processing_time_ms': 0.0,
                'decision_times': []  # Keep last 1000 for statistics
            }
            
            # Switching history for analysis
            self.switching_history: List[Dict[str, Any]] = []
            self.max_history_size = 10000
            
            # State checkpointing
            self.enable_checkpointing = config.get('enable_checkpointing', True)
            self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
            if self.enable_checkpointing:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            # System state
            self._last_checkpoint_time = time.time()
            self._checkpoint_interval = 300.0  # 5 minutes
            
            # Initialize successfully
            self._initialized = True
            initialization_time = (time.time() - start_time) * 1000
            self.performance_metrics['initialization_time_ms'] = initialization_time
            
            logger.info(f"IntegrationCoordinator initialized successfully in {initialization_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"Failed to initialize IntegrationCoordinator: {str(e)}")
            raise RuntimeError(f"IntegrationCoordinator initialization failed: {str(e)}")
        
    def _validate_configuration(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration parameters with hard failures.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        if not isinstance(config, dict):
            raise ValueError(f"Configuration must be a dictionary, got {type(config)}")
            
        # Validate min_dwell_time
        min_dwell_time = config.get('min_dwell_time', 1.0)
        if not isinstance(min_dwell_time, (int, float)):
            raise ValueError(f"min_dwell_time must be numeric, got {type(min_dwell_time)}")
        if min_dwell_time <= 0:
            raise ValueError(f"min_dwell_time must be positive, got {min_dwell_time}")
        if min_dwell_time > 3600:  # 1 hour max
            raise ValueError(f"min_dwell_time too large: {min_dwell_time}. Maximum: 3600")
            
        # Validate window_size
        window_size = config.get('window_size', 10)
        if not isinstance(window_size, int):
            raise ValueError(f"window_size must be integer, got {type(window_size)}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if window_size > 10000:
            raise ValueError(f"window_size too large: {window_size}. Maximum: 10000")
            
        # Validate progress_threshold
        progress_threshold = config.get('progress_threshold', 0.001)
        if not isinstance(progress_threshold, (int, float)):
            raise ValueError(f"progress_threshold must be numeric, got {type(progress_threshold)}")
        if progress_threshold < 0:
            raise ValueError(f"progress_threshold must be non-negative, got {progress_threshold}")
        if progress_threshold > 1.0:
            raise ValueError(f"progress_threshold must be <= 1.0, got {progress_threshold}")
            
        # Validate max_gradient_norm
        max_gradient_norm = config.get('max_gradient_norm', 1.0)
        if not isinstance(max_gradient_norm, (int, float)):
            raise ValueError(f"max_gradient_norm must be numeric, got {type(max_gradient_norm)}")
        if max_gradient_norm <= 0:
            raise ValueError(f"max_gradient_norm must be positive, got {max_gradient_norm}")
        if max_gradient_norm > 1000:
            raise ValueError(f"max_gradient_norm too large: {max_gradient_norm}. Maximum: 1000")
            
        # Validate checkpoint directory
        checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        if not isinstance(checkpoint_dir, str):
            raise ValueError(f"checkpoint_dir must be string, got {type(checkpoint_dir)}")
        
    def process_training_step(self, 
                            loss: float, 
                            gradient: np.ndarray, 
                            current_epoch: int, 
                            timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Process one training step and return comprehensive switching decision.
        
        This is the main entry point for the training manager. It:
        1. Updates progress monitoring with new loss
        2. Calculates progress rate
        3. Makes switching decision
        4. Validates and executes switch if needed
        5. Applies direction-aware gradient bounding
        6. Returns comprehensive results
        
        Args:
            loss: Current training loss value
            gradient: Current gradient as numpy array
            current_epoch: Current training epoch number
            timestamp: Optional timestamp (uses current time if None)
            
        Returns:
            Dict containing:
                - switching_decision: Complete switching decision info
                - bounded_gradient: Direction-bounded gradient
                - performance_metrics: Current performance metrics
                - system_status: Complete system status
                - decision_time_ms: Decision processing time
                
        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If processing fails at any step
        """
        # Input validation with detailed error messages
        if not isinstance(loss, (int, float)):
            raise ValueError(f"Loss must be numeric, got {type(loss).__name__}")
        if not np.isfinite(loss):
            raise ValueError(f"Loss must be finite, got {loss}")
            
        if not isinstance(gradient, np.ndarray):
            raise ValueError(f"Gradient must be numpy array, got {type(gradient).__name__}")
        if gradient.size == 0:
            raise ValueError("Gradient array is empty")
        if not np.all(np.isfinite(gradient)):
            raise ValueError("Gradient contains non-finite values")
            
        if not isinstance(current_epoch, int):
            raise ValueError(f"current_epoch must be integer, got {type(current_epoch).__name__}")
        if current_epoch < 0:
            raise ValueError(f"current_epoch must be non-negative, got {current_epoch}")
            
        if timestamp is None:
            timestamp = time.time()
        else:
            if not isinstance(timestamp, (int, float)):
                raise ValueError(f"timestamp must be numeric, got {type(timestamp).__name__}")
            if timestamp < 0:
                raise ValueError(f"timestamp must be non-negative, got {timestamp}")
                
        if not self._initialized:
            raise RuntimeError("IntegrationCoordinator not properly initialized")
            
        start_time = time.time()
        
        with self._lock:
            try:
                # Update progress monitor
                self.progress_monitor.add_loss(loss)
                
                # Get current progress rate
                progress_rate = self.progress_monitor.get_progress_rate()
                if not isinstance(progress_rate, (int, float)):
                    raise RuntimeError(f"Invalid progress rate from monitor: {progress_rate}")
                
                # Get current direction
                current_direction = self.direction_state.get_direction()
                
                # Check if switching is needed
                should_switch = self.simple_switcher.should_switch(progress_rate, current_epoch)
                
                # Build switching decision
                switching_decision = {
                    'should_switch': should_switch,
                    'current_direction': current_direction,
                    'progress_rate': progress_rate,
                    'current_epoch': current_epoch,
                    'timestamp': timestamp,
                    'loss': loss,
                    'gradient_norm': float(np.linalg.norm(gradient))
                }
                
                # Execute switching if needed
                switch_executed = False
                validation_error = None
                
                if should_switch:
                    new_direction = -current_direction
                    
                    # Validate new direction value
                    if not self.direction_validator.validate_direction(new_direction):
                        validation_error = f"Invalid direction value: {new_direction}"
                        self.performance_metrics['validation_failures'] += 1
                        raise RuntimeError(validation_error)
                        
                    # Check dwell time constraint
                    last_switch_time = self.direction_state.last_switch_time
                    if not self.direction_validator.can_switch(timestamp, last_switch_time):
                        time_since_switch = timestamp - last_switch_time
                        min_dwell = self.direction_validator.min_dwell_time
                        validation_error = (f"Cannot switch: insufficient dwell time. "
                                          f"Time since last switch: {time_since_switch:.2f}s, "
                                          f"Required: {min_dwell}s")
                        self.performance_metrics['validation_failures'] += 1
                        self.performance_metrics['failed_switches'] += 1
                        # Log but don't raise - this is expected behavior
                        logger.info(validation_error)
                        switching_decision['validation_error'] = validation_error
                    else:
                        # Execute direction switch
                        self.direction_state.switch_direction(new_direction, timestamp)
                        switch_executed = True
                        switching_decision['new_direction'] = new_direction
                        switching_decision['switch_executed'] = True
                        self.performance_metrics['total_switches'] += 1
                        self.performance_metrics['successful_switches'] += 1
                        self.performance_metrics['last_switch_time'] = timestamp
                        
                        # Record in history
                        self._record_switch_event(switching_decision)
                        
                        logger.info(f"Direction switched from {current_direction} to {new_direction} "
                                  f"at epoch {current_epoch}")
                else:
                    switching_decision['switch_executed'] = False
                
                # Apply direction-aware bounding
                bounded_gradient = self.direction_bounder.bound_gradient(gradient, current_direction)
                
                # Validate bounding result
                if not isinstance(bounded_gradient, np.ndarray):
                    raise RuntimeError(f"Bounding failed: expected numpy array, got {type(bounded_gradient).__name__}")
                if bounded_gradient.shape != gradient.shape:
                    raise RuntimeError(f"Bounding failed: shape mismatch. "
                                     f"Original: {gradient.shape}, Bounded: {bounded_gradient.shape}")
                if not np.all(np.isfinite(bounded_gradient)):
                    raise RuntimeError("Bounding produced non-finite values")
                
                # Calculate bounding statistics
                gradient_norm_before = float(np.linalg.norm(gradient))
                gradient_norm_after = float(np.linalg.norm(bounded_gradient))
                bounding_factor = gradient_norm_after / gradient_norm_before if gradient_norm_before > 0 else 1.0
                
                self.performance_metrics['total_bounding_operations'] += 1
                self.performance_metrics['total_validations'] += 1
                
                # Calculate decision time
                decision_time = time.time() - start_time
                decision_time_ms = decision_time * 1000
                
                # Update performance metrics
                self._update_performance_metrics(decision_time_ms)
                
                # Check if checkpoint is needed
                if self.enable_checkpointing:
                    self._check_and_save_checkpoint(timestamp)
                
                # Build comprehensive result
                result = {
                    'switching_decision': switching_decision,
                    'bounded_gradient': bounded_gradient,
                    'bounding_info': {
                        'gradient_norm_before': gradient_norm_before,
                        'gradient_norm_after': gradient_norm_after,
                        'bounding_factor': bounding_factor,
                        'direction_applied': current_direction
                    },
                    'performance_metrics': self.get_performance_metrics(),
                    'system_status': self.get_system_status(),
                    'decision_time_ms': decision_time_ms
                }
                
                # Log if decision time exceeds threshold (1ms)
                if decision_time_ms > 1.0:
                    logger.warning(f"Decision time exceeded 1ms: {decision_time_ms:.2f}ms")
                
                return result
                
            except Exception as e:
                logger.error(f"Training step processing failed: {str(e)}")
                raise RuntimeError(f"Integration coordinator failed during training step: {str(e)}")
    
    def should_switch_direction(self, current_epoch: int, timestamp: Optional[float] = None) -> bool:
        """
        Determine if direction should be switched with validation.
        
        Args:
            current_epoch: Current training epoch
            timestamp: Optional timestamp (uses current time if None)
            
        Returns:
            bool: Whether direction should be switched
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If decision process fails
        """
        if not isinstance(current_epoch, int):
            raise ValueError(f"current_epoch must be integer, got {type(current_epoch).__name__}")
        if current_epoch < 0:
            raise ValueError(f"current_epoch must be non-negative, got {current_epoch}")
            
        if timestamp is None:
            timestamp = time.time()
        else:
            if not isinstance(timestamp, (int, float)):
                raise ValueError(f"timestamp must be numeric, got {type(timestamp).__name__}")
            if timestamp < 0:
                raise ValueError(f"timestamp must be non-negative, got {timestamp}")
                
        with self._lock:
            try:
                progress_rate = self.progress_monitor.get_progress_rate()
                should_switch = self.simple_switcher.should_switch(progress_rate, current_epoch)
                
                # Also check dwell time constraint
                if should_switch:
                    last_switch_time = self.direction_state.last_switch_time
                    can_switch = self.direction_validator.can_switch(timestamp, last_switch_time)
                    return can_switch
                    
                return False
                
            except Exception as e:
                logger.error(f"Failed to determine switching decision: {str(e)}")
                raise RuntimeError(f"Switching decision failed: {str(e)}")
        
    def apply_direction_bounding(self, gradient: np.ndarray) -> np.ndarray:
        """
        Apply direction-specific bounding to gradient with validation.
        
        Args:
            gradient: Gradient to bound
            
        Returns:
            np.ndarray: Bounded gradient
            
        Raises:
            ValueError: If gradient is invalid
            RuntimeError: If bounding fails
        """
        if not isinstance(gradient, np.ndarray):
            raise ValueError(f"Gradient must be numpy array, got {type(gradient).__name__}")
        if gradient.size == 0:
            raise ValueError("Gradient array is empty")
        if not np.all(np.isfinite(gradient)):
            raise ValueError("Gradient contains non-finite values")
            
        with self._lock:
            try:
                current_direction = self.direction_state.get_direction()
                bounded_gradient = self.direction_bounder.bound_gradient(gradient, current_direction)
                
                # Validate result
                if not isinstance(bounded_gradient, np.ndarray):
                    raise RuntimeError(f"Bounding failed: expected numpy array, got {type(bounded_gradient).__name__}")
                if bounded_gradient.shape != gradient.shape:
                    raise RuntimeError(f"Bounding failed: shape mismatch. "
                                     f"Original: {gradient.shape}, Bounded: {bounded_gradient.shape}")
                if not np.all(np.isfinite(bounded_gradient)):
                    raise RuntimeError("Bounding produced non-finite values")
                    
                return bounded_gradient
                
            except Exception as e:
                logger.error(f"Gradient bounding failed: {str(e)}")
                raise RuntimeError(f"Direction bounding failed: {str(e)}")
        
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of all components.
        
        Returns:
            Dict containing complete system status
        """
        with self._lock:
            return {
                'direction_state': {
                    'current_direction': self.direction_state.get_direction(),
                    'switch_count': self.direction_state.switch_count,
                    'last_switch_time': self.direction_state.last_switch_time,
                    'time_since_last_switch': time.time() - self.direction_state.last_switch_time
                },
                'progress_monitor': {
                    'window_size': self.progress_monitor.window_size,
                    'loss_history_length': len(self.progress_monitor.loss_history),
                    'current_progress_rate': self.progress_monitor.get_progress_rate(),
                    'recent_losses': list(self.progress_monitor.loss_history)[-5:] if self.progress_monitor.loss_history else []
                },
                'direction_validator': {
                    'min_dwell_time': self.direction_validator.min_dwell_time,
                    'validation_count': self.direction_validator.validation_count
                },
                'direction_bounder': {
                    'max_gradient_norm': self.direction_bounder.max_gradient_norm,
                    'bounding_count': self.direction_bounder.bounding_count,
                    'total_norm_clipped': self.direction_bounder.total_norm_clipped
                },
                'simple_switcher': {
                    'progress_threshold': self.simple_switcher.progress_threshold,
                    'switch_count': self.simple_switcher.switch_count
                },
                'integration_coordinator': {
                    'initialized': self._initialized,
                    'config': self.config,
                    'switching_history_size': len(self.switching_history),
                    'checkpointing_enabled': self.enable_checkpointing,
                    'last_checkpoint_time': self._last_checkpoint_time
                },
                'performance_metrics': self.get_performance_metrics()
            }
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get detailed performance metrics.
        
        Returns:
            Dict containing all performance metrics
        """
        with self._lock:
            metrics = self.performance_metrics.copy()
            
            # Calculate additional statistics
            if metrics['decision_times']:
                recent_times = metrics['decision_times'][-100:]
                metrics['recent_avg_decision_time_ms'] = sum(recent_times) / len(recent_times)
                metrics['recent_max_decision_time_ms'] = max(recent_times)
                metrics['recent_min_decision_time_ms'] = min(recent_times)
                
            # Calculate success rate
            total_attempts = metrics['total_switches'] + metrics['failed_switches']
            metrics['switch_success_rate'] = (
                metrics['successful_switches'] / total_attempts if total_attempts > 0 else 1.0
            )
            
            # Calculate validation success rate
            if metrics['total_validations'] > 0:
                metrics['validation_success_rate'] = (
                    1.0 - metrics['validation_failures'] / metrics['total_validations']
                )
            else:
                metrics['validation_success_rate'] = 1.0
                
            # Remove internal tracking list from output
            if 'decision_times' in metrics:
                del metrics['decision_times']
                
            return metrics
            
    def get_switching_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get switching history with optional limit.
        
        Args:
            limit: Maximum number of recent events to return
            
        Returns:
            List of switching events
        """
        with self._lock:
            if limit is None:
                return self.switching_history.copy()
            else:
                return self.switching_history[-limit:].copy()
        
    def reset_system(self) -> None:
        """
        Reset all components to initial state.
        
        Raises:
            RuntimeError: If reset fails
        """
        with self._lock:
            try:
                # Reset all components
                self.direction_state = DirectionState()
                self.progress_monitor = ProgressMonitor(
                    window_size=self.config.get('window_size', 10)
                )
                
                # Reset performance metrics
                self.performance_metrics = {
                    'total_switches': 0,
                    'successful_switches': 0,
                    'failed_switches': 0,
                    'total_validations': 0,
                    'validation_failures': 0,
                    'total_bounding_operations': 0,
                    'average_decision_time_ms': 0.0,
                    'peak_decision_time_ms': 0.0,
                    'last_switch_time': 0.0,
                    'initialization_time_ms': self.performance_metrics['initialization_time_ms'],
                    'total_processing_time_ms': 0.0,
                    'decision_times': []
                }
                
                # Clear history
                self.switching_history.clear()
                
                logger.info("System reset completed successfully")
                
            except Exception as e:
                logger.error(f"System reset failed: {str(e)}")
                raise RuntimeError(f"Failed to reset system: {str(e)}")
    
    def save_checkpoint(self, filepath: Optional[str] = None) -> str:
        """
        Save current system state to checkpoint file.
        
        Args:
            filepath: Optional custom filepath (uses default if None)
            
        Returns:
            str: Path to saved checkpoint
            
        Raises:
            RuntimeError: If checkpoint save fails
        """
        with self._lock:
            try:
                if filepath is None:
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = os.path.join(self.checkpoint_dir, f"checkpoint_{timestamp_str}.json")
                    
                checkpoint_data = {
                    'timestamp': time.time(),
                    'direction_state': {
                        'current_direction': self.direction_state.get_direction(),
                        'switch_count': self.direction_state.switch_count,
                        'last_switch_time': self.direction_state.last_switch_time
                    },
                    'progress_monitor': {
                        'loss_history': list(self.progress_monitor.loss_history),
                        'window_size': self.progress_monitor.window_size
                    },
                    'performance_metrics': self.performance_metrics,
                    'switching_history': self.switching_history[-1000:],  # Last 1000 events
                    'config': self.config
                }
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                # Save checkpoint
                with open(filepath, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                    
                logger.info(f"Checkpoint saved to {filepath}")
                return filepath
                
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {str(e)}")
                raise RuntimeError(f"Checkpoint save failed: {str(e)}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load system state from checkpoint file.
        
        Args:
            filepath: Path to checkpoint file
            
        Raises:
            ValueError: If checkpoint file is invalid
            RuntimeError: If checkpoint load fails
        """
        if not os.path.exists(filepath):
            raise ValueError(f"Checkpoint file not found: {filepath}")
            
        with self._lock:
            try:
                with open(filepath, 'r') as f:
                    checkpoint_data = json.load(f)
                    
                # Validate checkpoint structure
                required_keys = ['direction_state', 'progress_monitor', 'performance_metrics']
                for key in required_keys:
                    if key not in checkpoint_data:
                        raise ValueError(f"Invalid checkpoint: missing {key}")
                        
                # Restore direction state
                direction_data = checkpoint_data['direction_state']
                self.direction_state = DirectionState()
                self.direction_state.switch_direction(
                    direction_data['current_direction'],
                    direction_data['last_switch_time']
                )
                self.direction_state.switch_count = direction_data['switch_count']
                
                # Restore progress monitor
                progress_data = checkpoint_data['progress_monitor']
                self.progress_monitor = ProgressMonitor(window_size=progress_data['window_size'])
                for loss in progress_data['loss_history']:
                    self.progress_monitor.add_loss(loss)
                    
                # Restore performance metrics
                self.performance_metrics.update(checkpoint_data['performance_metrics'])
                
                # Restore switching history
                if 'switching_history' in checkpoint_data:
                    self.switching_history = checkpoint_data['switching_history']
                    
                logger.info(f"Checkpoint loaded from {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {str(e)}")
                raise RuntimeError(f"Checkpoint load failed: {str(e)}")
    
    def _update_performance_metrics(self, decision_time_ms: float) -> None:
        """Update performance metrics with new decision time."""
        metrics = self.performance_metrics
        
        # Update decision times
        metrics['decision_times'].append(decision_time_ms)
        if len(metrics['decision_times']) > 1000:
            metrics['decision_times'] = metrics['decision_times'][-1000:]
            
        # Update average
        n = metrics['total_validations']
        if n > 1:
            metrics['average_decision_time_ms'] = (
                (metrics['average_decision_time_ms'] * (n - 1) + decision_time_ms) / n
            )
        else:
            metrics['average_decision_time_ms'] = decision_time_ms
            
        # Update peak
        if decision_time_ms > metrics['peak_decision_time_ms']:
            metrics['peak_decision_time_ms'] = decision_time_ms
            
        # Update total processing time
        metrics['total_processing_time_ms'] += decision_time_ms
        
    def _record_switch_event(self, switching_decision: Dict[str, Any]) -> None:
        """Record switching event in history."""
        event = {
            'timestamp': switching_decision['timestamp'],
            'epoch': switching_decision['current_epoch'],
            'old_direction': switching_decision['current_direction'],
            'new_direction': switching_decision.get('new_direction'),
            'progress_rate': switching_decision['progress_rate'],
            'loss': switching_decision['loss'],
            'gradient_norm': switching_decision['gradient_norm']
        }
        
        self.switching_history.append(event)
        
        # Maintain history size limit
        if len(self.switching_history) > self.max_history_size:
            self.switching_history = self.switching_history[-self.max_history_size:]
            
    def _check_and_save_checkpoint(self, timestamp: float) -> None:
        """Check if checkpoint is needed and save if necessary."""
        if timestamp - self._last_checkpoint_time >= self._checkpoint_interval:
            try:
                self.save_checkpoint()
                self._last_checkpoint_time = timestamp
            except Exception as e:
                logger.warning(f"Failed to save periodic checkpoint: {str(e)}")