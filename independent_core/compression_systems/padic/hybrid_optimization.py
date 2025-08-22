"""
Hybrid Optimization - Hybrid-compatible optimization algorithms
NO FALLBACKS - HARD FAILURES ONLY
"""

import logging
import math
import time
import torch
import uuid
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

# Import existing optimization
from padic_advanced import PadicOptimizationManager

# Import hybrid structures
from hybrid_padic_structures import HybridPadicWeight, HybridPadicValidator
from padic_encoder import PadicWeight


@dataclass
class HybridOptimizerConfig:
    """Configuration for hybrid optimizers"""
    learning_rate: float = 0.01
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False
    
    # Adam-specific parameters
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    amsgrad: bool = False
    
    # RMSprop-specific parameters
    alpha: float = 0.99
    centered: bool = False
    
    # Hybrid-specific parameters
    enable_two_channel_optimization: bool = True
    exponent_channel_lr_multiplier: float = 1.0
    mantissa_channel_lr_multiplier: float = 1.0
    enable_gpu_acceleration: bool = True
    
    def __post_init__(self):
        """Validate optimizer configuration"""
        if not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.learning_rate}")
        if not isinstance(self.momentum, (int, float)) or not 0 <= self.momentum <= 1:
            raise ValueError(f"Momentum must be in [0, 1], got {self.momentum}")
        if not isinstance(self.beta1, (int, float)) or not 0 <= self.beta1 < 1:
            raise ValueError(f"Beta1 must be in [0, 1), got {self.beta1}")
        if not isinstance(self.beta2, (int, float)) or not 0 <= self.beta2 < 1:
            raise ValueError(f"Beta2 must be in [0, 1), got {self.beta2}")
        if not isinstance(self.alpha, (int, float)) or not 0 <= self.alpha <= 1:
            raise ValueError(f"Alpha must be in [0, 1], got {self.alpha}")
        if self.exponent_channel_lr_multiplier <= 0:
            raise ValueError(f"Exponent channel LR multiplier must be positive, got {self.exponent_channel_lr_multiplier}")
        if self.mantissa_channel_lr_multiplier <= 0:
            raise ValueError(f"Mantissa channel LR multiplier must be positive, got {self.mantissa_channel_lr_multiplier}")


@dataclass
class HybridOptimizerState:
    """State for hybrid optimizers"""
    optimizer_id: str
    optimizer_type: str
    config: HybridOptimizerConfig
    parameters: List[HybridPadicWeight]
    step_count: int = 0
    
    # SGD state
    momentum_buffer_exp: Optional[List[torch.Tensor]] = None
    momentum_buffer_man: Optional[List[torch.Tensor]] = None
    
    # Adam state
    exp_avg_exp: Optional[List[torch.Tensor]] = None
    exp_avg_man: Optional[List[torch.Tensor]] = None
    exp_avg_sq_exp: Optional[List[torch.Tensor]] = None
    exp_avg_sq_man: Optional[List[torch.Tensor]] = None
    max_exp_avg_sq_exp: Optional[List[torch.Tensor]] = None
    max_exp_avg_sq_man: Optional[List[torch.Tensor]] = None
    
    # RMSprop state
    square_avg_exp: Optional[List[torch.Tensor]] = None
    square_avg_man: Optional[List[torch.Tensor]] = None
    grad_avg_exp: Optional[List[torch.Tensor]] = None
    grad_avg_man: Optional[List[torch.Tensor]] = None
    
    # Performance tracking
    creation_time: datetime = field(default_factory=datetime.utcnow)
    last_step_time: Optional[datetime] = None
    total_step_time_ms: float = 0.0
    
    def __post_init__(self):
        """Initialize optimizer state"""
        if not isinstance(self.optimizer_id, str) or not self.optimizer_id.strip():
            raise ValueError("Optimizer ID must be non-empty string")
        if not isinstance(self.optimizer_type, str):
            raise TypeError("Optimizer type must be string")
        if not isinstance(self.config, HybridOptimizerConfig):
            raise TypeError("Config must be HybridOptimizerConfig")
        if not isinstance(self.parameters, list):
            raise TypeError("Parameters must be list")
        if not all(isinstance(p, HybridPadicWeight) for p in self.parameters):
            raise TypeError("All parameters must be HybridPadicWeight")


@dataclass
class HybridOptimizationStats:
    """Statistics for hybrid optimization operations"""
    total_optimizers_created: int = 0
    active_optimizers: int = 0
    total_optimization_steps: int = 0
    sgd_steps: int = 0
    adam_steps: int = 0
    rmsprop_steps: int = 0
    
    # Performance metrics
    average_step_time_ms: float = 0.0
    average_convergence_rate: float = 0.0
    gpu_memory_usage_mb: float = 0.0
    
    # Optimization quality
    successful_steps: int = 0
    failed_steps: int = 0
    numerical_instabilities: int = 0
    
    # Parameter statistics
    parameter_count_distribution: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    learning_rate_distribution: Dict[float, int] = field(default_factory=lambda: defaultdict(int))
    
    last_update: Optional[datetime] = None
    
    def update_step(self, optimizer_type: str, step_time_ms: float, success: bool, param_count: int):
        """Update statistics with optimization step"""
        self.total_optimization_steps += 1
        self.last_update = datetime.utcnow()
        
        if success:
            self.successful_steps += 1
        else:
            self.failed_steps += 1
        
        # Update type-specific counters
        if optimizer_type == "sgd":
            self.sgd_steps += 1
        elif optimizer_type == "adam":
            self.adam_steps += 1
        elif optimizer_type == "rmsprop":
            self.rmsprop_steps += 1
        
        # Update averages
        if self.total_optimization_steps > 1:
            old_avg = self.average_step_time_ms
            self.average_step_time_ms = (
                (old_avg * (self.total_optimization_steps - 1) + step_time_ms) / 
                self.total_optimization_steps
            )
        else:
            self.average_step_time_ms = step_time_ms
        
        # Update parameter count distribution
        self.parameter_count_distribution[param_count] += 1
        
        # Update success rate
        self.average_convergence_rate = self.successful_steps / self.total_optimization_steps


class HybridOptimizationManager:
    """
    Hybrid-compatible optimization algorithms.
    Provides GPU-accelerated optimizers for hybrid p-adic weights with two-channel processing.
    """
    
    def __init__(self, prime: int):
        """Initialize hybrid optimization manager"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not isinstance(prime, int) or prime <= 1:
            raise ValueError(f"Prime must be int > 1, got {prime}")
        
        self.prime = prime
        self.logger = logging.getLogger('HybridOptimizationManager')
        
        # Optimizer management
        self.optimizers: Dict[str, HybridOptimizerState] = {}
        self.validator = HybridPadicValidator()
        
        # Performance tracking
        self.optimization_stats = HybridOptimizationStats()
        self.operation_history: deque = deque(maxlen=1000)
        
        # GPU optimization
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for hybrid optimization")
        
        # Optimization caches
        self.gradient_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.max_cache_size = 1000
        
        self.logger.info(f"HybridOptimizationManager initialized with prime={prime}")
    
    def create_hybrid_sgd_optimizer(self, params: List[HybridPadicWeight], lr: float = 0.01,
                                  momentum: float = 0.0, dampening: float = 0.0,
                                  weight_decay: float = 0.0, nesterov: bool = False,
                                  name: Optional[str] = None) -> str:
        """
        Create SGD optimizer for hybrid p-adic weights.
        
        Args:
            params: List of hybrid parameters to optimize
            lr: Learning rate
            momentum: Momentum factor
            dampening: Dampening for momentum
            weight_decay: Weight decay (L2 penalty)
            nesterov: Enable Nesterov momentum
            name: Optional optimizer name
            
        Returns:
            Optimizer ID string
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(params, list):
            raise TypeError(f"Parameters must be list, got {type(params)}")
        if not params:
            raise ValueError("Parameters list cannot be empty")
        if not all(isinstance(p, HybridPadicWeight) for p in params):
            raise TypeError("All parameters must be HybridPadicWeight")
        
        # Validate all parameters
        for i, param in enumerate(params):
            try:
                self.validator.validate_hybrid_weight(param)
            except Exception as e:
                raise ValueError(f"Parameter {i} validation failed: {e}")
        
        # Create configuration
        config = HybridOptimizerConfig(
            learning_rate=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        
        # Generate optimizer ID
        optimizer_id = name or f"sgd_{uuid.uuid4().hex[:8]}"
        if optimizer_id in self.optimizers:
            raise ValueError(f"Optimizer ID {optimizer_id} already exists")
        
        # Create optimizer state
        optimizer_state = HybridOptimizerState(
            optimizer_id=optimizer_id,
            optimizer_type="sgd",
            config=config,
            parameters=params.copy()
        )
        
        # Initialize momentum buffers if needed
        if momentum > 0:
            optimizer_state.momentum_buffer_exp = [
                torch.zeros_like(p.exponent_channel) for p in params
            ]
            optimizer_state.momentum_buffer_man = [
                torch.zeros_like(p.mantissa_channel) for p in params
            ]
        
        # Store optimizer
        self.optimizers[optimizer_id] = optimizer_state
        self.optimization_stats.total_optimizers_created += 1
        self.optimization_stats.active_optimizers += 1
        
        self.logger.info(f"Created SGD optimizer {optimizer_id} with {len(params)} parameters")
        
        return optimizer_id
    
    def create_hybrid_adam_optimizer(self, params: List[HybridPadicWeight], lr: float = 0.001,
                                   betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                                   weight_decay: float = 0.0, amsgrad: bool = False,
                                   name: Optional[str] = None) -> str:
        """
        Create Adam optimizer for hybrid p-adic weights.
        
        Args:
            params: List of hybrid parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages
            eps: Term added for numerical stability
            weight_decay: Weight decay (L2 penalty)
            amsgrad: Use AMSGrad variant
            name: Optional optimizer name
            
        Returns:
            Optimizer ID string
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(params, list):
            raise TypeError(f"Parameters must be list, got {type(params)}")
        if not params:
            raise ValueError("Parameters list cannot be empty")
        if not all(isinstance(p, HybridPadicWeight) for p in params):
            raise TypeError("All parameters must be HybridPadicWeight")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError("Betas must be tuple/list of length 2")
        
        # Validate all parameters
        for i, param in enumerate(params):
            try:
                self.validator.validate_hybrid_weight(param)
            except Exception as e:
                raise ValueError(f"Parameter {i} validation failed: {e}")
        
        # Create configuration
        config = HybridOptimizerConfig(
            learning_rate=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        
        # Generate optimizer ID
        optimizer_id = name or f"adam_{uuid.uuid4().hex[:8]}"
        if optimizer_id in self.optimizers:
            raise ValueError(f"Optimizer ID {optimizer_id} already exists")
        
        # Create optimizer state
        optimizer_state = HybridOptimizerState(
            optimizer_id=optimizer_id,
            optimizer_type="adam",
            config=config,
            parameters=params.copy()
        )
        
        # Initialize Adam state
        optimizer_state.exp_avg_exp = [torch.zeros_like(p.exponent_channel) for p in params]
        optimizer_state.exp_avg_man = [torch.zeros_like(p.mantissa_channel) for p in params]
        optimizer_state.exp_avg_sq_exp = [torch.zeros_like(p.exponent_channel) for p in params]
        optimizer_state.exp_avg_sq_man = [torch.zeros_like(p.mantissa_channel) for p in params]
        
        if amsgrad:
            optimizer_state.max_exp_avg_sq_exp = [torch.zeros_like(p.exponent_channel) for p in params]
            optimizer_state.max_exp_avg_sq_man = [torch.zeros_like(p.mantissa_channel) for p in params]
        
        # Store optimizer
        self.optimizers[optimizer_id] = optimizer_state
        self.optimization_stats.total_optimizers_created += 1
        self.optimization_stats.active_optimizers += 1
        
        self.logger.info(f"Created Adam optimizer {optimizer_id} with {len(params)} parameters")
        
        return optimizer_id
    
    def create_hybrid_rmsprop_optimizer(self, params: List[HybridPadicWeight], lr: float = 0.01,
                                      alpha: float = 0.99, eps: float = 1e-8,
                                      weight_decay: float = 0.0, momentum: float = 0.0,
                                      centered: bool = False, name: Optional[str] = None) -> str:
        """
        Create RMSprop optimizer for hybrid p-adic weights.
        
        Args:
            params: List of hybrid parameters to optimize
            lr: Learning rate
            alpha: Smoothing constant
            eps: Term added for numerical stability
            weight_decay: Weight decay (L2 penalty)
            momentum: Momentum factor
            centered: Compute centered RMSprop
            name: Optional optimizer name
            
        Returns:
            Optimizer ID string
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(params, list):
            raise TypeError(f"Parameters must be list, got {type(params)}")
        if not params:
            raise ValueError("Parameters list cannot be empty")
        if not all(isinstance(p, HybridPadicWeight) for p in params):
            raise TypeError("All parameters must be HybridPadicWeight")
        
        # Validate all parameters
        for i, param in enumerate(params):
            try:
                self.validator.validate_hybrid_weight(param)
            except Exception as e:
                raise ValueError(f"Parameter {i} validation failed: {e}")
        
        # Create configuration
        config = HybridOptimizerConfig(
            learning_rate=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered
        )
        
        # Generate optimizer ID
        optimizer_id = name or f"rmsprop_{uuid.uuid4().hex[:8]}"
        if optimizer_id in self.optimizers:
            raise ValueError(f"Optimizer ID {optimizer_id} already exists")
        
        # Create optimizer state
        optimizer_state = HybridOptimizerState(
            optimizer_id=optimizer_id,
            optimizer_type="rmsprop",
            config=config,
            parameters=params.copy()
        )
        
        # Initialize RMSprop state
        optimizer_state.square_avg_exp = [torch.zeros_like(p.exponent_channel) for p in params]
        optimizer_state.square_avg_man = [torch.zeros_like(p.mantissa_channel) for p in params]
        
        if momentum > 0:
            optimizer_state.momentum_buffer_exp = [torch.zeros_like(p.exponent_channel) for p in params]
            optimizer_state.momentum_buffer_man = [torch.zeros_like(p.mantissa_channel) for p in params]
        
        if centered:
            optimizer_state.grad_avg_exp = [torch.zeros_like(p.exponent_channel) for p in params]
            optimizer_state.grad_avg_man = [torch.zeros_like(p.mantissa_channel) for p in params]
        
        # Store optimizer
        self.optimizers[optimizer_id] = optimizer_state
        self.optimization_stats.total_optimizers_created += 1
        self.optimization_stats.active_optimizers += 1
        
        self.logger.info(f"Created RMSprop optimizer {optimizer_id} with {len(params)} parameters")
        
        return optimizer_id
    
    def step_hybrid_optimizer(self, optimizer_name: str, 
                            gradients: List[Tuple[torch.Tensor, torch.Tensor]]) -> bool:
        """
        Perform optimization step with hybrid gradients.
        
        Args:
            optimizer_name: Name of optimizer
            gradients: List of (exponent_grad, mantissa_grad) tuples
            
        Returns:
            True if step successful
            
        Raises:
            ValueError: If optimizer or gradients are invalid
        """
        if not isinstance(optimizer_name, str):
            raise TypeError(f"Optimizer name must be string, got {type(optimizer_name)}")
        if optimizer_name not in self.optimizers:
            raise ValueError(f"Optimizer {optimizer_name} not found")
        if not isinstance(gradients, list):
            raise TypeError(f"Gradients must be list, got {type(gradients)}")
        
        optimizer_state = self.optimizers[optimizer_name]
        
        if len(gradients) != len(optimizer_state.parameters):
            raise ValueError(f"Gradient count {len(gradients)} != parameter count {len(optimizer_state.parameters)}")
        
        # Validate gradients
        for i, (exp_grad, man_grad) in enumerate(gradients):
            if not isinstance(exp_grad, torch.Tensor) or not isinstance(man_grad, torch.Tensor):
                raise TypeError(f"Gradient {i} must be tuple of torch.Tensor")
            param = optimizer_state.parameters[i]
            if exp_grad.shape != param.exponent_channel.shape:
                raise ValueError(f"Exponent gradient {i} shape mismatch: {exp_grad.shape} != {param.exponent_channel.shape}")
            if man_grad.shape != param.mantissa_channel.shape:
                raise ValueError(f"Mantissa gradient {i} shape mismatch: {man_grad.shape} != {param.mantissa_channel.shape}")
        
        start_time = time.time()
        
        try:
            # Dispatch to appropriate optimizer
            if optimizer_state.optimizer_type == "sgd":
                success = self._step_sgd(optimizer_state, gradients)
            elif optimizer_state.optimizer_type == "adam":
                success = self._step_adam(optimizer_state, gradients)
            elif optimizer_state.optimizer_type == "rmsprop":
                success = self._step_rmsprop(optimizer_state, gradients)
            else:
                raise RuntimeError(f"Unknown optimizer type: {optimizer_state.optimizer_type}")
            
            # Update timing
            step_time_ms = (time.time() - start_time) * 1000
            optimizer_state.total_step_time_ms += step_time_ms
            optimizer_state.last_step_time = datetime.utcnow()
            optimizer_state.step_count += 1
            
            # Update statistics
            self.optimization_stats.update_step(
                optimizer_state.optimizer_type, step_time_ms, success, len(optimizer_state.parameters)
            )
            
            # Record operation
            self.operation_history.append({
                'timestamp': datetime.utcnow(),
                'optimizer_id': optimizer_name,
                'optimizer_type': optimizer_state.optimizer_type,
                'step_time_ms': step_time_ms,
                'success': success,
                'step_count': optimizer_state.step_count
            })
            
            if success:
                self.logger.debug(f"Optimization step completed for {optimizer_name} in {step_time_ms:.2f}ms")
            else:
                self.logger.warning(f"Optimization step failed for {optimizer_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Optimization step failed: {e}")
            self.optimization_stats.failed_steps += 1
            return False
    
    def get_hybrid_optimization_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization statistics.
        
        Returns:
            Dictionary containing optimization statistics
        """
        return {
            'overall_stats': {
                'total_optimizers_created': self.optimization_stats.total_optimizers_created,
                'active_optimizers': self.optimization_stats.active_optimizers,
                'total_optimization_steps': self.optimization_stats.total_optimization_steps,
                'successful_steps': self.optimization_stats.successful_steps,
                'failed_steps': self.optimization_stats.failed_steps,
                'success_rate': (
                    self.optimization_stats.successful_steps / 
                    max(1, self.optimization_stats.total_optimization_steps)
                ),
                'average_step_time_ms': self.optimization_stats.average_step_time_ms,
                'average_convergence_rate': self.optimization_stats.average_convergence_rate
            },
            'optimizer_type_stats': {
                'sgd_steps': self.optimization_stats.sgd_steps,
                'adam_steps': self.optimization_stats.adam_steps,
                'rmsprop_steps': self.optimization_stats.rmsprop_steps,
                'sgd_usage_ratio': (
                    self.optimization_stats.sgd_steps / 
                    max(1, self.optimization_stats.total_optimization_steps)
                ),
                'adam_usage_ratio': (
                    self.optimization_stats.adam_steps / 
                    max(1, self.optimization_stats.total_optimization_steps)
                ),
                'rmsprop_usage_ratio': (
                    self.optimization_stats.rmsprop_steps / 
                    max(1, self.optimization_stats.total_optimization_steps)
                )
            },
            'performance_stats': {
                'gpu_memory_usage_mb': self.optimization_stats.gpu_memory_usage_mb,
                'numerical_instabilities': self.optimization_stats.numerical_instabilities,
                'gradient_cache_size': len(self.gradient_cache),
                'operations_history_length': len(self.operation_history)
            },
            'parameter_stats': {
                'parameter_count_distribution': dict(self.optimization_stats.parameter_count_distribution),
                'learning_rate_distribution': dict(self.optimization_stats.learning_rate_distribution),
                'most_common_parameter_count': self._get_most_common_parameter_count()
            },
            'optimizer_details': {
                optimizer_id: {
                    'type': state.optimizer_type,
                    'step_count': state.step_count,
                    'parameter_count': len(state.parameters),
                    'learning_rate': state.config.learning_rate,
                    'total_time_ms': state.total_step_time_ms,
                    'average_step_time_ms': (
                        state.total_step_time_ms / max(1, state.step_count)
                    ),
                    'creation_time': state.creation_time.isoformat(),
                    'last_step_time': state.last_step_time.isoformat() if state.last_step_time else None
                }
                for optimizer_id, state in self.optimizers.items()
            },
            'configuration': {
                'prime': self.prime,
                'device': str(self.device),
                'max_cache_size': self.max_cache_size
            },
            'last_update': self.optimization_stats.last_update.isoformat() if self.optimization_stats.last_update else None
        }
    
    def _step_sgd(self, optimizer_state: HybridOptimizerState, 
                gradients: List[Tuple[torch.Tensor, torch.Tensor]]) -> bool:
        """Perform SGD optimization step"""
        
        try:
            config = optimizer_state.config
            
            for i, (param, (exp_grad, man_grad)) in enumerate(zip(optimizer_state.parameters, gradients)):
                # Apply weight decay if specified
                if config.weight_decay != 0:
                    exp_grad = exp_grad.add(param.exponent_channel, alpha=config.weight_decay)
                    man_grad = man_grad.add(param.mantissa_channel, alpha=config.weight_decay)
                
                # Apply momentum if specified
                if config.momentum != 0:
                    if optimizer_state.momentum_buffer_exp is None or optimizer_state.momentum_buffer_man is None:
                        optimizer_state.momentum_buffer_exp = [torch.zeros_like(p.exponent_channel) for p in optimizer_state.parameters]
                        optimizer_state.momentum_buffer_man = [torch.zeros_like(p.mantissa_channel) for p in optimizer_state.parameters]
                    
                    buf_exp = optimizer_state.momentum_buffer_exp[i]
                    buf_man = optimizer_state.momentum_buffer_man[i]
                    
                    buf_exp.mul_(config.momentum).add_(exp_grad, alpha=1 - config.dampening)
                    buf_man.mul_(config.momentum).add_(man_grad, alpha=1 - config.dampening)
                    
                    if config.nesterov:
                        exp_grad = exp_grad.add(buf_exp, alpha=config.momentum)
                        man_grad = man_grad.add(buf_man, alpha=config.momentum)
                    else:
                        exp_grad = buf_exp
                        man_grad = buf_man
                
                # Apply updates with channel-specific learning rates
                exp_lr = config.learning_rate * config.exponent_channel_lr_multiplier
                man_lr = config.learning_rate * config.mantissa_channel_lr_multiplier
                
                param.exponent_channel.add_(exp_grad, alpha=-exp_lr)
                param.mantissa_channel.add_(man_grad, alpha=-man_lr)
                
                # Apply p-adic constraints to maintain validity
                self._apply_padic_constraints(param)
            
            return True
            
        except Exception as e:
            self.logger.error(f"SGD step failed: {e}")
            self.optimization_stats.numerical_instabilities += 1
            return False
    
    def _step_adam(self, optimizer_state: HybridOptimizerState, 
                 gradients: List[Tuple[torch.Tensor, torch.Tensor]]) -> bool:
        """Perform Adam optimization step"""
        
        try:
            config = optimizer_state.config
            optimizer_state.step_count += 1
            
            # Bias correction
            bias_correction1 = 1 - config.beta1 ** optimizer_state.step_count
            bias_correction2 = 1 - config.beta2 ** optimizer_state.step_count
            
            for i, (param, (exp_grad, man_grad)) in enumerate(zip(optimizer_state.parameters, gradients)):
                # Apply weight decay if specified
                if config.weight_decay != 0:
                    exp_grad = exp_grad.add(param.exponent_channel, alpha=config.weight_decay)
                    man_grad = man_grad.add(param.mantissa_channel, alpha=config.weight_decay)
                
                # Update biased first moment estimate
                optimizer_state.exp_avg_exp[i].mul_(config.beta1).add_(exp_grad, alpha=1 - config.beta1)
                optimizer_state.exp_avg_man[i].mul_(config.beta1).add_(man_grad, alpha=1 - config.beta1)
                
                # Update biased second raw moment estimate
                optimizer_state.exp_avg_sq_exp[i].mul_(config.beta2).addcmul_(exp_grad, exp_grad, value=1 - config.beta2)
                optimizer_state.exp_avg_sq_man[i].mul_(config.beta2).addcmul_(man_grad, man_grad, value=1 - config.beta2)
                
                # Compute bias-corrected first and second moment estimates
                exp_avg_exp_corrected = optimizer_state.exp_avg_exp[i] / bias_correction1
                exp_avg_man_corrected = optimizer_state.exp_avg_man[i] / bias_correction1
                
                if config.amsgrad:
                    # Maintain maximum of squared gradients
                    torch.maximum(optimizer_state.max_exp_avg_sq_exp[i], optimizer_state.exp_avg_sq_exp[i], 
                                 out=optimizer_state.max_exp_avg_sq_exp[i])
                    torch.maximum(optimizer_state.max_exp_avg_sq_man[i], optimizer_state.exp_avg_sq_man[i], 
                                 out=optimizer_state.max_exp_avg_sq_man[i])
                    
                    exp_avg_sq_exp_corrected = optimizer_state.max_exp_avg_sq_exp[i] / bias_correction2
                    exp_avg_sq_man_corrected = optimizer_state.max_exp_avg_sq_man[i] / bias_correction2
                else:
                    exp_avg_sq_exp_corrected = optimizer_state.exp_avg_sq_exp[i] / bias_correction2
                    exp_avg_sq_man_corrected = optimizer_state.exp_avg_sq_man[i] / bias_correction2
                
                # Compute step size
                exp_step_size = config.learning_rate * config.exponent_channel_lr_multiplier
                man_step_size = config.learning_rate * config.mantissa_channel_lr_multiplier
                
                # Apply updates
                param.exponent_channel.addcdiv_(exp_avg_exp_corrected, 
                                              exp_avg_sq_exp_corrected.sqrt().add_(config.eps), 
                                              value=-exp_step_size)
                param.mantissa_channel.addcdiv_(exp_avg_man_corrected, 
                                              exp_avg_sq_man_corrected.sqrt().add_(config.eps), 
                                              value=-man_step_size)
                
                # Apply p-adic constraints
                self._apply_padic_constraints(param)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Adam step failed: {e}")
            self.optimization_stats.numerical_instabilities += 1
            return False
    
    def _step_rmsprop(self, optimizer_state: HybridOptimizerState, 
                    gradients: List[Tuple[torch.Tensor, torch.Tensor]]) -> bool:
        """Perform RMSprop optimization step"""
        
        try:
            config = optimizer_state.config
            
            for i, (param, (exp_grad, man_grad)) in enumerate(zip(optimizer_state.parameters, gradients)):
                # Apply weight decay if specified
                if config.weight_decay != 0:
                    exp_grad = exp_grad.add(param.exponent_channel, alpha=config.weight_decay)
                    man_grad = man_grad.add(param.mantissa_channel, alpha=config.weight_decay)
                
                # Update squared gradient averages
                optimizer_state.square_avg_exp[i].mul_(config.alpha).addcmul_(exp_grad, exp_grad, value=1 - config.alpha)
                optimizer_state.square_avg_man[i].mul_(config.alpha).addcmul_(man_grad, man_grad, value=1 - config.alpha)
                
                if config.centered:
                    # Update gradient averages
                    optimizer_state.grad_avg_exp[i].mul_(config.alpha).add_(exp_grad, alpha=1 - config.alpha)
                    optimizer_state.grad_avg_man[i].mul_(config.alpha).add_(man_grad, alpha=1 - config.alpha)
                    
                    # Centered variance
                    exp_avg = optimizer_state.square_avg_exp[i].addcmul(optimizer_state.grad_avg_exp[i], 
                                                                       optimizer_state.grad_avg_exp[i], value=-1)
                    man_avg = optimizer_state.square_avg_man[i].addcmul(optimizer_state.grad_avg_man[i], 
                                                                       optimizer_state.grad_avg_man[i], value=-1)
                else:
                    exp_avg = optimizer_state.square_avg_exp[i]
                    man_avg = optimizer_state.square_avg_man[i]
                
                # Apply momentum if specified
                if config.momentum > 0:
                    if optimizer_state.momentum_buffer_exp is None or optimizer_state.momentum_buffer_man is None:
                        optimizer_state.momentum_buffer_exp = [torch.zeros_like(p.exponent_channel) for p in optimizer_state.parameters]
                        optimizer_state.momentum_buffer_man = [torch.zeros_like(p.mantissa_channel) for p in optimizer_state.parameters]
                    
                    buf_exp = optimizer_state.momentum_buffer_exp[i]
                    buf_man = optimizer_state.momentum_buffer_man[i]
                    
                    buf_exp.mul_(config.momentum).addcdiv_(exp_grad, exp_avg.sqrt().add_(config.eps))
                    buf_man.mul_(config.momentum).addcdiv_(man_grad, man_avg.sqrt().add_(config.eps))
                    
                    exp_step = buf_exp
                    man_step = buf_man
                else:
                    exp_step = exp_grad.div(exp_avg.sqrt().add_(config.eps))
                    man_step = man_grad.div(man_avg.sqrt().add_(config.eps))
                
                # Apply updates with channel-specific learning rates
                exp_lr = config.learning_rate * config.exponent_channel_lr_multiplier
                man_lr = config.learning_rate * config.mantissa_channel_lr_multiplier
                
                param.exponent_channel.add_(exp_step, alpha=-exp_lr)
                param.mantissa_channel.add_(man_step, alpha=-man_lr)
                
                # Apply p-adic constraints
                self._apply_padic_constraints(param)
            
            return True
            
        except Exception as e:
            self.logger.error(f"RMSprop step failed: {e}")
            self.optimization_stats.numerical_instabilities += 1
            return False
    
    def _apply_padic_constraints(self, param: HybridPadicWeight) -> None:
        """Apply p-adic constraints to maintain weight validity"""
        
        # Clamp values to reasonable ranges to maintain numerical stability
        with torch.no_grad():
            # Clamp exponent channel (hierarchical importance)
            param.exponent_channel.clamp_(-10.0, 10.0)
            
            # Clamp mantissa channel (fine-grained values)
            param.mantissa_channel.clamp_(-1.0, 1.0)
            
            # Ensure no NaN or inf values
            if torch.isnan(param.exponent_channel).any() or torch.isinf(param.exponent_channel).any():
                param.exponent_channel.fill_(0.0)
                self.optimization_stats.numerical_instabilities += 1
            
            if torch.isnan(param.mantissa_channel).any() or torch.isinf(param.mantissa_channel).any():
                param.mantissa_channel.fill_(0.0)
                self.optimization_stats.numerical_instabilities += 1
    
    def _get_most_common_parameter_count(self) -> int:
        """Get most common parameter count from distribution"""
        if not self.optimization_stats.parameter_count_distribution:
            return 0
        return max(self.optimization_stats.parameter_count_distribution.items(), key=lambda x: x[1])[0]
    
    def get_optimizer_info(self, optimizer_name: str) -> Dict[str, Any]:
        """
        Get detailed information about specific optimizer.
        
        Args:
            optimizer_name: Name of optimizer
            
        Returns:
            Dictionary containing optimizer information
        """
        if optimizer_name not in self.optimizers:
            raise ValueError(f"Optimizer {optimizer_name} not found")
        
        state = self.optimizers[optimizer_name]
        
        return {
            'optimizer_id': state.optimizer_id,
            'optimizer_type': state.optimizer_type,
            'step_count': state.step_count,
            'parameter_count': len(state.parameters),
            'configuration': {
                'learning_rate': state.config.learning_rate,
                'momentum': state.config.momentum,
                'beta1': state.config.beta1,
                'beta2': state.config.beta2,
                'alpha': state.config.alpha,
                'weight_decay': state.config.weight_decay,
                'enable_two_channel_optimization': state.config.enable_two_channel_optimization
            },
            'performance': {
                'total_step_time_ms': state.total_step_time_ms,
                'average_step_time_ms': (
                    state.total_step_time_ms / max(1, state.step_count)
                ),
                'creation_time': state.creation_time.isoformat(),
                'last_step_time': state.last_step_time.isoformat() if state.last_step_time else None
            },
            'state_info': {
                'has_momentum_buffer': state.momentum_buffer_exp is not None,
                'has_adam_state': state.exp_avg_exp is not None,
                'has_rmsprop_state': state.square_avg_exp is not None
            }
        }
    
    def remove_optimizer(self, optimizer_name: str) -> bool:
        """
        Remove optimizer and free its resources.
        
        Args:
            optimizer_name: Name of optimizer to remove
            
        Returns:
            True if removed successfully
        """
        if optimizer_name not in self.optimizers:
            return False
        
        # Clear GPU memory for state tensors
        state = self.optimizers[optimizer_name]
        
        # Clear all tensor buffers
        for attr_name in ['momentum_buffer_exp', 'momentum_buffer_man', 'exp_avg_exp', 'exp_avg_man',
                         'exp_avg_sq_exp', 'exp_avg_sq_man', 'max_exp_avg_sq_exp', 'max_exp_avg_sq_man',
                         'square_avg_exp', 'square_avg_man', 'grad_avg_exp', 'grad_avg_man']:
            buffer = getattr(state, attr_name)
            if buffer is not None:
                for tensor in buffer:
                    del tensor
                setattr(state, attr_name, None)
        
        # Remove optimizer
        del self.optimizers[optimizer_name]
        self.optimization_stats.active_optimizers -= 1
        
        self.logger.info(f"Removed optimizer {optimizer_name}")
        
        return True
    
    def clear_gradient_cache(self) -> None:
        """Clear gradient computation cache"""
        self.gradient_cache.clear()
        self.logger.info("Gradient cache cleared")
    
    def shutdown(self) -> None:
        """Shutdown hybrid optimization manager"""
        self.logger.info("Shutting down hybrid optimization manager")
        
        # Remove all optimizers
        optimizer_names = list(self.optimizers.keys())
        for name in optimizer_names:
            self.remove_optimizer(name)
        
        # Clear caches and data
        self.gradient_cache.clear()
        self.operation_history.clear()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Hybrid optimization manager shutdown complete")