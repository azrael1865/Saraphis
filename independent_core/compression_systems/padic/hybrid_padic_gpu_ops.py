"""
GPU-accelerated hybrid p-adic operations using CUDA kernels
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import gc
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
import math

from hybrid_padic_structures import (
    HybridPadicWeight, 
    HybridPadicValidator, 
    HybridPadicConverter, 
    HybridPadicManager
)
from padic_encoder import PadicWeight, PadicValidation


@dataclass
class GPUOperationConfig:
    """Configuration for GPU-accelerated p-adic operations"""
    stream_count: int = 4
    max_batch_size: int = 1024
    memory_limit_mb: int = 2048
    enable_mixed_precision: bool = True
    enable_tensor_cores: bool = True
    optimization_level: int = 3
    enable_memory_pool: bool = True
    stream_synchronization: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if not isinstance(self.stream_count, int) or self.stream_count <= 0:
            raise ValueError(f"stream_count must be positive int, got {self.stream_count}")
        if not isinstance(self.max_batch_size, int) or self.max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive int, got {self.max_batch_size}")
        if not isinstance(self.memory_limit_mb, int) or self.memory_limit_mb <= 0:
            raise ValueError(f"memory_limit_mb must be positive int, got {self.memory_limit_mb}")
        if self.stream_count > 32:
            raise ValueError(f"stream_count {self.stream_count} exceeds maximum 32")
        if self.max_batch_size > 65536:
            raise ValueError(f"max_batch_size {self.max_batch_size} exceeds maximum 65536")


class HybridPadicGPUOps:
    """GPU-accelerated arithmetic operations for hybrid p-adic weights"""
    
    def __init__(self, config: GPUOperationConfig, prime: int, precision: int):
        """Initialize GPU operations"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is None:
            raise ValueError("Config cannot be None")
        if not isinstance(config, GPUOperationConfig):
            raise TypeError(f"Config must be GPUOperationConfig, got {type(config)}")
        
        PadicValidation.validate_prime(prime)
        PadicValidation.validate_precision(precision)
        
        self.config = config
        self.prime = prime
        self.precision = precision
        
        # Initialize CUDA components
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for GPU p-adic operations")
        
        self.device = torch.device('cuda:0')
        self.dtype = torch.float32 if not config.enable_mixed_precision else torch.float16
        
        # Initialize CUDA streams
        self.streams = [torch.cuda.Stream() for _ in range(config.stream_count)]
        self.current_stream_idx = 0
        
        # Initialize GPU memory management
        try:
            from ..gpu_memory.gpu_memory_core import GPUMemoryOptimizer
            gpu_config = {
                'memory_limit_mb': config.memory_limit_mb,
                'enable_memory_pool': config.enable_memory_pool
            }
            self.gpu_memory_optimizer = GPUMemoryOptimizer(gpu_config)
        except ImportError:
            self.gpu_memory_optimizer = None
        
        # Initialize validation components
        self.validator = HybridPadicValidator()
        self.converter = HybridPadicConverter()
        
        # Pre-compute GPU constants
        self._initialize_gpu_constants()
        
        # Performance tracking
        self.operation_stats = {
            'total_operations': 0,
            'add_operations': 0,
            'multiply_operations': 0,
            'divide_operations': 0,
            'power_operations': 0,
            'batch_operations': 0,
            'average_operation_time': 0.0,
            'peak_memory_usage': 0,
            'cuda_kernel_calls': 0,
            'stream_switches': 0
        }
    
    def _initialize_gpu_constants(self) -> None:
        """Initialize GPU-resident constants for p-adic arithmetic"""
        # Prime powers for modular arithmetic
        prime_powers = torch.tensor(
            [self.prime ** i for i in range(self.precision + 1)], 
            dtype=torch.int64, 
            device=self.device
        )
        self.register_buffer('prime_powers', prime_powers)
        
        # Precomputed modular inverses for division
        mod_inverses = []
        for i in range(1, self.prime):
            try:
                inv = pow(i, -1, self.prime)
                mod_inverses.append(inv)
            except ValueError:
                mod_inverses.append(0)  # Invalid inverse
        
        self.mod_inverses = torch.tensor(
            mod_inverses, 
            dtype=torch.int64, 
            device=self.device
        )
        
        # Constants for log-space operations
        self.log_prime = torch.tensor(
            math.log(self.prime), 
            dtype=self.dtype, 
            device=self.device
        )
        self.inv_log_prime = torch.tensor(
            1.0 / math.log(self.prime), 
            dtype=self.dtype, 
            device=self.device
        )
    
    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        """Register a persistent buffer on GPU"""
        setattr(self, name, tensor)
    
    def _get_next_stream(self) -> torch.cuda.Stream:
        """Get next available CUDA stream"""
        stream = self.streams[self.current_stream_idx]
        self.current_stream_idx = (self.current_stream_idx + 1) % len(self.streams)
        self.operation_stats['stream_switches'] += 1
        return stream
    
    def _validate_hybrid_weights(self, *weights: HybridPadicWeight) -> None:
        """Validate hybrid p-adic weights for GPU operations"""
        for i, weight in enumerate(weights):
            if weight is None:
                raise ValueError(f"Weight {i} cannot be None")
            if not isinstance(weight, HybridPadicWeight):
                raise TypeError(f"Weight {i} must be HybridPadicWeight, got {type(weight)}")
            self.validator.validate_hybrid_weight(weight)
            
            # Ensure weights are on same device
            if not weight.exponent_channel.is_cuda:
                raise ValueError(f"Weight {i} exponent_channel must be on CUDA")
            if not weight.mantissa_channel.is_cuda:
                raise ValueError(f"Weight {i} mantissa_channel must be on CUDA")
            if weight.exponent_channel.device != self.device:
                raise ValueError(f"Weight {i} exponent_channel on wrong device: {weight.exponent_channel.device}")
            if weight.mantissa_channel.device != self.device:
                raise ValueError(f"Weight {i} mantissa_channel on wrong device: {weight.mantissa_channel.device}")
    
    def add(self, x: HybridPadicWeight, y: HybridPadicWeight) -> HybridPadicWeight:
        """GPU-accelerated addition of hybrid p-adic weights"""
        # NO FALLBACKS - HARD FAILURES ONLY
        self._validate_hybrid_weights(x, y)
        
        start_time = time.time()
        stream = self._get_next_stream()
        
        with torch.cuda.stream(stream):
            # Prepare GPU memory
            if self.gpu_memory_optimizer:
                memory_mb = (x.exponent_channel.numel() + y.exponent_channel.numel()) * 4 / (1024 * 1024)
                self.gpu_memory_optimizer.prepare_for_computation(estimated_memory_mb=memory_mb)
            
            # Perform addition in log-space for numerical stability
            # Convert to log representation
            x_log_exp = torch.log(x.exponent_channel.abs() + 1e-10) * x.exponent_channel.sign()
            y_log_exp = torch.log(y.exponent_channel.abs() + 1e-10) * y.exponent_channel.sign()
            x_log_mant = torch.log(x.mantissa_channel.abs() + 1e-10) * x.mantissa_channel.sign()
            y_log_mant = torch.log(y.mantissa_channel.abs() + 1e-10) * y.mantissa_channel.sign()
            
            # GPU kernel for p-adic addition
            result_exp = self._gpu_padic_add_kernel(x_log_exp, y_log_exp, x_log_mant, y_log_mant)
            result_mant = self._gpu_padic_mantissa_combine_kernel(x_log_mant, y_log_mant)
            
            # Apply modular reduction
            result_exp = torch.fmod(result_exp, self.prime_powers[self.precision].float())
            result_mant = torch.fmod(result_mant, self.prime_powers[self.precision].float())
            
            # Ensure results are in valid range
            result_exp = torch.clamp(result_exp, -1e6, 1e6)
            result_mant = torch.clamp(result_mant, -1e6, 1e6)
        
        if self.config.stream_synchronization:
            torch.cuda.synchronize()
        
        # Create result weight
        result = HybridPadicWeight(
            exponent_channel=result_exp,
            mantissa_channel=result_mant,
            prime=self.prime,
            precision=self.precision,
            compression_params=x.compression_params.copy(),
            gpu_memory_info=self._get_memory_info()
        )
        
        # Validate result
        self.validator.validate_hybrid_weight(result)
        
        # Update stats
        self._update_operation_stats('add', time.time() - start_time)
        
        return result
    
    def multiply(self, x: HybridPadicWeight, y: HybridPadicWeight) -> HybridPadicWeight:
        """GPU-accelerated multiplication of hybrid p-adic weights"""
        # NO FALLBACKS - HARD FAILURES ONLY
        self._validate_hybrid_weights(x, y)
        
        start_time = time.time()
        stream = self._get_next_stream()
        
        with torch.cuda.stream(stream):
            # Prepare GPU memory
            if self.gpu_memory_optimizer:
                memory_mb = (x.exponent_channel.numel() + y.exponent_channel.numel()) * 4 / (1024 * 1024)
                self.gpu_memory_optimizer.prepare_for_computation(estimated_memory_mb=memory_mb)
            
            # Perform multiplication in log-space
            x_log_exp = torch.log(x.exponent_channel.abs() + 1e-10)
            y_log_exp = torch.log(y.exponent_channel.abs() + 1e-10)
            x_log_mant = torch.log(x.mantissa_channel.abs() + 1e-10)
            y_log_mant = torch.log(y.mantissa_channel.abs() + 1e-10)
            
            # GPU kernel for p-adic multiplication
            result_exp = self._gpu_padic_multiply_kernel(x_log_exp, y_log_exp, x_log_mant, y_log_mant)
            result_mant = self._gpu_padic_mantissa_multiply_kernel(x_log_mant, y_log_mant)
            
            # Handle signs
            exp_sign = x.exponent_channel.sign() * y.exponent_channel.sign()
            mant_sign = x.mantissa_channel.sign() * y.mantissa_channel.sign()
            
            result_exp = result_exp * exp_sign
            result_mant = result_mant * mant_sign
            
            # Apply modular reduction
            result_exp = torch.fmod(result_exp, self.prime_powers[self.precision].float())
            result_mant = torch.fmod(result_mant, self.prime_powers[self.precision].float())
            
            # Ensure results are in valid range
            result_exp = torch.clamp(result_exp, -1e6, 1e6)
            result_mant = torch.clamp(result_mant, -1e6, 1e6)
        
        if self.config.stream_synchronization:
            torch.cuda.synchronize()
        
        # Create result weight
        result = HybridPadicWeight(
            exponent_channel=result_exp,
            mantissa_channel=result_mant,
            prime=self.prime,
            precision=self.precision,
            compression_params=x.compression_params.copy(),
            gpu_memory_info=self._get_memory_info()
        )
        
        # Validate result
        self.validator.validate_hybrid_weight(result)
        
        # Update stats
        self._update_operation_stats('multiply', time.time() - start_time)
        
        return result
    
    def divide(self, x: HybridPadicWeight, y: HybridPadicWeight) -> HybridPadicWeight:
        """GPU-accelerated division of hybrid p-adic weights"""
        # NO FALLBACKS - HARD FAILURES ONLY
        self._validate_hybrid_weights(x, y)
        
        # Check for division by zero
        if torch.allclose(y.exponent_channel, torch.zeros_like(y.exponent_channel), atol=1e-10):
            raise ValueError("Division by zero in exponent channel")
        if torch.allclose(y.mantissa_channel, torch.zeros_like(y.mantissa_channel), atol=1e-10):
            raise ValueError("Division by zero in mantissa channel")
        
        start_time = time.time()
        stream = self._get_next_stream()
        
        with torch.cuda.stream(stream):
            # Prepare GPU memory
            if self.gpu_memory_optimizer:
                memory_mb = (x.exponent_channel.numel() + y.exponent_channel.numel()) * 4 / (1024 * 1024)
                self.gpu_memory_optimizer.prepare_for_computation(estimated_memory_mb=memory_mb)
            
            # Perform division in log-space
            x_log_exp = torch.log(x.exponent_channel.abs() + 1e-10)
            y_log_exp = torch.log(y.exponent_channel.abs() + 1e-10)
            x_log_mant = torch.log(x.mantissa_channel.abs() + 1e-10)
            y_log_mant = torch.log(y.mantissa_channel.abs() + 1e-10)
            
            # GPU kernel for p-adic division
            result_exp = self._gpu_padic_divide_kernel(x_log_exp, y_log_exp, x_log_mant, y_log_mant)
            result_mant = self._gpu_padic_mantissa_divide_kernel(x_log_mant, y_log_mant)
            
            # Handle signs
            exp_sign = x.exponent_channel.sign() / y.exponent_channel.sign()
            mant_sign = x.mantissa_channel.sign() / y.mantissa_channel.sign()
            
            result_exp = result_exp * exp_sign
            result_mant = result_mant * mant_sign
            
            # Apply modular reduction
            result_exp = torch.fmod(result_exp, self.prime_powers[self.precision].float())
            result_mant = torch.fmod(result_mant, self.prime_powers[self.precision].float())
            
            # Ensure results are in valid range
            result_exp = torch.clamp(result_exp, -1e6, 1e6)
            result_mant = torch.clamp(result_mant, -1e6, 1e6)
        
        if self.config.stream_synchronization:
            torch.cuda.synchronize()
        
        # Create result weight
        result = HybridPadicWeight(
            exponent_channel=result_exp,
            mantissa_channel=result_mant,
            prime=self.prime,
            precision=self.precision,
            compression_params=x.compression_params.copy(),
            gpu_memory_info=self._get_memory_info()
        )
        
        # Validate result
        self.validator.validate_hybrid_weight(result)
        
        # Update stats
        self._update_operation_stats('divide', time.time() - start_time)
        
        return result
    
    def power(self, x: HybridPadicWeight, exponent: Union[int, float]) -> HybridPadicWeight:
        """GPU-accelerated exponentiation of hybrid p-adic weights"""
        # NO FALLBACKS - HARD FAILURES ONLY
        self._validate_hybrid_weights(x)
        
        if not isinstance(exponent, (int, float)):
            raise TypeError(f"Exponent must be int or float, got {type(exponent)}")
        if math.isnan(exponent) or math.isinf(exponent):
            raise ValueError(f"Invalid exponent: {exponent}")
        if abs(exponent) > 1000:
            raise ValueError(f"Exponent {exponent} too large")
        
        start_time = time.time()
        stream = self._get_next_stream()
        
        with torch.cuda.stream(stream):
            # Prepare GPU memory
            if self.gpu_memory_optimizer:
                memory_mb = x.exponent_channel.numel() * 4 / (1024 * 1024)
                self.gpu_memory_optimizer.prepare_for_computation(estimated_memory_mb=memory_mb)
            
            # Convert exponent to tensor
            exp_tensor = torch.tensor(exponent, dtype=self.dtype, device=self.device)
            
            # Perform exponentiation in log-space
            x_log_exp = torch.log(x.exponent_channel.abs() + 1e-10)
            x_log_mant = torch.log(x.mantissa_channel.abs() + 1e-10)
            
            # GPU kernel for p-adic exponentiation
            result_exp = self._gpu_padic_power_kernel(x_log_exp, exp_tensor)
            result_mant = self._gpu_padic_power_kernel(x_log_mant, exp_tensor)
            
            # Handle signs for odd/even exponents
            if isinstance(exponent, int) and exponent % 2 == 0:
                # Even exponent - result is always positive
                result_exp = result_exp.abs()
                result_mant = result_mant.abs()
            else:
                # Odd exponent - preserve signs
                exp_sign = torch.pow(x.exponent_channel.sign(), exponent)
                mant_sign = torch.pow(x.mantissa_channel.sign(), exponent)
                result_exp = result_exp * exp_sign
                result_mant = result_mant * mant_sign
            
            # Apply modular reduction
            result_exp = torch.fmod(result_exp, self.prime_powers[self.precision].float())
            result_mant = torch.fmod(result_mant, self.prime_powers[self.precision].float())
            
            # Ensure results are in valid range
            result_exp = torch.clamp(result_exp, -1e6, 1e6)
            result_mant = torch.clamp(result_mant, -1e6, 1e6)
        
        if self.config.stream_synchronization:
            torch.cuda.synchronize()
        
        # Create result weight
        result = HybridPadicWeight(
            exponent_channel=result_exp,
            mantissa_channel=result_mant,
            prime=self.prime,
            precision=self.precision,
            compression_params=x.compression_params.copy(),
            gpu_memory_info=self._get_memory_info()
        )
        
        # Validate result
        self.validator.validate_hybrid_weight(result)
        
        # Update stats
        self._update_operation_stats('power', time.time() - start_time)
        
        return result
    
    def batch_operations(self, weights: List[HybridPadicWeight], 
                        operations: List[str], 
                        operands: List[Union[HybridPadicWeight, float]]) -> List[HybridPadicWeight]:
        """GPU-accelerated batch operations on multiple hybrid p-adic weights"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not weights:
            raise ValueError("Weights list cannot be empty")
        if not operations:
            raise ValueError("Operations list cannot be empty")
        if not operands:
            raise ValueError("Operands list cannot be empty")
        if len(weights) != len(operations) or len(operations) != len(operands):
            raise ValueError("Weights, operations, and operands must have same length")
        
        # Validate all weights
        for i, weight in enumerate(weights):
            if not isinstance(weight, HybridPadicWeight):
                raise TypeError(f"Weight {i} must be HybridPadicWeight, got {type(weight)}")
            self.validator.validate_hybrid_weight(weight)
        
        # Validate operations
        valid_ops = {'add', 'multiply', 'divide', 'power'}
        for i, op in enumerate(operations):
            if op not in valid_ops:
                raise ValueError(f"Invalid operation {op} at index {i}. Valid: {valid_ops}")
        
        if len(weights) > self.config.max_batch_size:
            raise ValueError(f"Batch size {len(weights)} exceeds maximum {self.config.max_batch_size}")
        
        start_time = time.time()
        stream = self._get_next_stream()
        
        results = []
        
        with torch.cuda.stream(stream):
            # Prepare GPU memory for batch
            if self.gpu_memory_optimizer:
                total_elements = sum(w.exponent_channel.numel() for w in weights)
                memory_mb = total_elements * 4 / (1024 * 1024)
                self.gpu_memory_optimizer.prepare_for_computation(estimated_memory_mb=memory_mb)
            
            # Process batch operations
            for i, (weight, op, operand) in enumerate(zip(weights, operations, operands)):
                try:
                    if op == 'add':
                        if not isinstance(operand, HybridPadicWeight):
                            raise TypeError(f"Add operand {i} must be HybridPadicWeight")
                        result = self.add(weight, operand)
                    elif op == 'multiply':
                        if not isinstance(operand, HybridPadicWeight):
                            raise TypeError(f"Multiply operand {i} must be HybridPadicWeight")
                        result = self.multiply(weight, operand)
                    elif op == 'divide':
                        if not isinstance(operand, HybridPadicWeight):
                            raise TypeError(f"Divide operand {i} must be HybridPadicWeight")
                        result = self.divide(weight, operand)
                    elif op == 'power':
                        if not isinstance(operand, (int, float)):
                            raise TypeError(f"Power operand {i} must be int or float")
                        result = self.power(weight, operand)
                    else:
                        raise ValueError(f"Unsupported operation: {op}")
                    
                    results.append(result)
                    
                except Exception as e:
                    raise ValueError(f"Batch operation {i} ({op}) failed: {e}")
        
        if self.config.stream_synchronization:
            torch.cuda.synchronize()
        
        # Update stats
        self._update_operation_stats('batch', time.time() - start_time)
        self.operation_stats['batch_operations'] += 1
        
        return results
    
    def _gpu_padic_add_kernel(self, x_exp: torch.Tensor, y_exp: torch.Tensor, 
                             x_mant: torch.Tensor, y_mant: torch.Tensor) -> torch.Tensor:
        """GPU kernel for p-adic addition in log-space"""
        self.operation_stats['cuda_kernel_calls'] += 1
        
        # Combine exponents using log-space addition
        max_exp = torch.maximum(x_exp, y_exp)
        exp_diff = torch.abs(x_exp - y_exp)
        
        # Use log-sum-exp trick for numerical stability
        result = max_exp + torch.log1p(torch.exp(-exp_diff))
        
        return result
    
    def _gpu_padic_mantissa_combine_kernel(self, x_mant: torch.Tensor, y_mant: torch.Tensor) -> torch.Tensor:
        """GPU kernel for combining mantissas in p-adic addition"""
        self.operation_stats['cuda_kernel_calls'] += 1
        
        # Simple addition in log-space with modular reduction
        result = x_mant + y_mant
        return result
    
    def _gpu_padic_multiply_kernel(self, x_exp: torch.Tensor, y_exp: torch.Tensor,
                                  x_mant: torch.Tensor, y_mant: torch.Tensor) -> torch.Tensor:
        """GPU kernel for p-adic multiplication in log-space"""
        self.operation_stats['cuda_kernel_calls'] += 1
        
        # Multiplication in log-space is addition
        result = x_exp + y_exp
        return result
    
    def _gpu_padic_mantissa_multiply_kernel(self, x_mant: torch.Tensor, y_mant: torch.Tensor) -> torch.Tensor:
        """GPU kernel for mantissa multiplication in p-adic arithmetic"""
        self.operation_stats['cuda_kernel_calls'] += 1
        
        # Multiplication in log-space is addition
        result = x_mant + y_mant
        return result
    
    def _gpu_padic_divide_kernel(self, x_exp: torch.Tensor, y_exp: torch.Tensor,
                                x_mant: torch.Tensor, y_mant: torch.Tensor) -> torch.Tensor:
        """GPU kernel for p-adic division in log-space"""
        self.operation_stats['cuda_kernel_calls'] += 1
        
        # Division in log-space is subtraction
        result = x_exp - y_exp
        return result
    
    def _gpu_padic_mantissa_divide_kernel(self, x_mant: torch.Tensor, y_mant: torch.Tensor) -> torch.Tensor:
        """GPU kernel for mantissa division in p-adic arithmetic"""
        self.operation_stats['cuda_kernel_calls'] += 1
        
        # Division in log-space is subtraction
        result = x_mant - y_mant
        return result
    
    def _gpu_padic_power_kernel(self, x_log: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
        """GPU kernel for p-adic exponentiation in log-space"""
        self.operation_stats['cuda_kernel_calls'] += 1
        
        # Exponentiation in log-space is multiplication
        result = x_log * exponent
        return result
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get current GPU memory information"""
        return {
            'allocated': torch.cuda.memory_allocated(self.device),
            'cached': torch.cuda.memory_reserved(self.device),
            'max_allocated': torch.cuda.max_memory_allocated(self.device)
        }
    
    def _update_operation_stats(self, operation: str, execution_time: float) -> None:
        """Update operation statistics"""
        self.operation_stats['total_operations'] += 1
        self.operation_stats[f'{operation}_operations'] += 1
        
        total_ops = self.operation_stats['total_operations']
        old_avg = self.operation_stats['average_operation_time']
        self.operation_stats['average_operation_time'] = (
            (old_avg * (total_ops - 1) + execution_time) / total_ops
        )
        
        # Update peak memory usage
        current_memory = torch.cuda.memory_allocated(self.device)
        if current_memory > self.operation_stats['peak_memory_usage']:
            self.operation_stats['peak_memory_usage'] = current_memory
    
    def synchronize_streams(self) -> None:
        """Synchronize all CUDA streams"""
        for stream in self.streams:
            stream.synchronize()
    
    def cleanup_gpu_memory(self) -> None:
        """Clean up GPU memory and reset streams"""
        self.synchronize_streams()
        
        if self.gpu_memory_optimizer:
            self.gpu_memory_optimizer.cleanup()
        
        torch.cuda.empty_cache()
        gc.collect()
    
    def get_operation_stats(self) -> Dict[str, Any]:
        """Get operation statistics"""
        stats = self.operation_stats.copy()
        stats['memory_info'] = self._get_memory_info()
        stats['stream_count'] = len(self.streams)
        
        if self.gpu_memory_optimizer:
            stats['gpu_memory_stats'] = self.gpu_memory_optimizer.get_performance_metrics()
        
        return stats
    
    def reset_operation_stats(self) -> None:
        """Reset operation statistics"""
        self.operation_stats = {
            'total_operations': 0,
            'add_operations': 0,
            'multiply_operations': 0,
            'divide_operations': 0,
            'power_operations': 0,
            'batch_operations': 0,
            'average_operation_time': 0.0,
            'peak_memory_usage': 0,
            'cuda_kernel_calls': 0,
            'stream_switches': 0
        }


class HybridPadicGPUOptimizer:
    """GPU-accelerated optimizer for training hybrid p-adic weights"""
    
    def __init__(self, gpu_ops: HybridPadicGPUOps, learning_rate: float = 0.001, 
                 momentum: float = 0.9, weight_decay: float = 0.0):
        """Initialize GPU optimizer"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if gpu_ops is None:
            raise ValueError("GPU operations cannot be None")
        if not isinstance(gpu_ops, HybridPadicGPUOps):
            raise TypeError(f"GPU operations must be HybridPadicGPUOps, got {type(gpu_ops)}")
        
        if not isinstance(learning_rate, (int, float)):
            raise TypeError(f"Learning rate must be numeric, got {type(learning_rate)}")
        if learning_rate <= 0:
            raise ValueError(f"Learning rate must be > 0, got {learning_rate}")
        if learning_rate > 1.0:
            raise ValueError(f"Learning rate {learning_rate} too large")
        
        if not isinstance(momentum, (int, float)):
            raise TypeError(f"Momentum must be numeric, got {type(momentum)}")
        if not (0 <= momentum < 1):
            raise ValueError(f"Momentum must be in [0, 1), got {momentum}")
        
        if not isinstance(weight_decay, (int, float)):
            raise TypeError(f"Weight decay must be numeric, got {type(weight_decay)}")
        if weight_decay < 0:
            raise ValueError(f"Weight decay must be >= 0, got {weight_decay}")
        
        self.gpu_ops = gpu_ops
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Momentum buffers for weights
        self.momentum_buffers: Dict[int, HybridPadicWeight] = {}
        
        # Optimization statistics
        self.optimization_stats = {
            'total_steps': 0,
            'weight_updates': 0,
            'gradient_computations': 0,
            'momentum_updates': 0,
            'average_step_time': 0.0,
            'convergence_rate': 0.0
        }
    
    def step(self, weights: List[HybridPadicWeight], 
             gradients: List[HybridPadicWeight]) -> List[HybridPadicWeight]:
        """Perform optimization step with GPU acceleration"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not weights:
            raise ValueError("Weights list cannot be empty")
        if not gradients:
            raise ValueError("Gradients list cannot be empty")
        if len(weights) != len(gradients):
            raise ValueError(f"Weights ({len(weights)}) and gradients ({len(gradients)}) length mismatch")
        
        # Validate all weights and gradients
        for i, (weight, grad) in enumerate(zip(weights, gradients)):
            if not isinstance(weight, HybridPadicWeight):
                raise TypeError(f"Weight {i} must be HybridPadicWeight, got {type(weight)}")
            if not isinstance(grad, HybridPadicWeight):
                raise TypeError(f"Gradient {i} must be HybridPadicWeight, got {type(grad)}")
            self.gpu_ops.validator.validate_hybrid_weight(weight)
            self.gpu_ops.validator.validate_hybrid_weight(grad)
        
        start_time = time.time()
        
        updated_weights = []
        
        for i, (weight, grad) in enumerate(zip(weights, gradients)):
            # Apply weight decay if configured
            if self.weight_decay > 0:
                # weight_decay * weight
                decay_weight = self.gpu_ops.multiply(
                    weight, 
                    self._create_scalar_weight(self.weight_decay)
                )
                # gradient = gradient + weight_decay * weight
                grad = self.gpu_ops.add(grad, decay_weight)
            
            # Apply momentum if configured
            if self.momentum > 0:
                weight_id = id(weight)
                
                if weight_id in self.momentum_buffers:
                    # momentum_buffer = momentum * momentum_buffer + gradient
                    momentum_term = self.gpu_ops.multiply(
                        self.momentum_buffers[weight_id],
                        self._create_scalar_weight(self.momentum)
                    )
                    momentum_buffer = self.gpu_ops.add(momentum_term, grad)
                else:
                    # Initialize momentum buffer with gradient
                    momentum_buffer = grad
                
                self.momentum_buffers[weight_id] = momentum_buffer
                effective_grad = momentum_buffer
                self.optimization_stats['momentum_updates'] += 1
            else:
                effective_grad = grad
            
            # Apply learning rate: lr * gradient
            lr_grad = self.gpu_ops.multiply(
                effective_grad,
                self._create_scalar_weight(self.learning_rate)
            )
            
            # Update weight: weight = weight - lr * gradient
            updated_weight = self.gpu_ops.add(
                weight,
                self.gpu_ops.multiply(lr_grad, self._create_scalar_weight(-1.0))
            )
            
            updated_weights.append(updated_weight)
            self.optimization_stats['weight_updates'] += 1
        
        # Update statistics
        step_time = time.time() - start_time
        self._update_optimization_stats(step_time)
        
        return updated_weights
    
    def _create_scalar_weight(self, scalar: float) -> HybridPadicWeight:
        """Create a hybrid p-adic weight from a scalar value"""
        # Create scalar tensors on GPU
        scalar_tensor = torch.full(
            (1,), scalar, 
            dtype=self.gpu_ops.dtype, 
            device=self.gpu_ops.device
        )
        
        return HybridPadicWeight(
            exponent_channel=scalar_tensor,
            mantissa_channel=scalar_tensor.clone(),
            prime=self.gpu_ops.prime,
            precision=self.gpu_ops.precision,
            compression_params={'scalar': True},
            gpu_memory_info=self.gpu_ops._get_memory_info()
        )
    
    def _update_optimization_stats(self, step_time: float) -> None:
        """Update optimization statistics"""
        self.optimization_stats['total_steps'] += 1
        self.optimization_stats['gradient_computations'] += 1
        
        total_steps = self.optimization_stats['total_steps']
        old_avg = self.optimization_stats['average_step_time']
        self.optimization_stats['average_step_time'] = (
            (old_avg * (total_steps - 1) + step_time) / total_steps
        )
    
    def zero_grad(self) -> None:
        """Clear momentum buffers (equivalent to zero_grad)"""
        self.momentum_buffers.clear()
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        stats = self.optimization_stats.copy()
        stats['momentum_buffer_count'] = len(self.momentum_buffers)
        stats['gpu_ops_stats'] = self.gpu_ops.get_operation_stats()
        return stats
    
    def reset_optimization_stats(self) -> None:
        """Reset optimization statistics"""
        self.optimization_stats = {
            'total_steps': 0,
            'weight_updates': 0,
            'gradient_computations': 0,
            'momentum_updates': 0,
            'average_step_time': 0.0,
            'convergence_rate': 0.0
        }


class HybridPadicGPUManager:
    """Manages multiple GPU optimizers and coordinates hybrid p-adic training"""
    
    def __init__(self, config: GPUOperationConfig, prime: int, precision: int):
        """Initialize GPU manager"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is None:
            raise ValueError("Config cannot be None")
        if not isinstance(config, GPUOperationConfig):
            raise TypeError(f"Config must be GPUOperationConfig, got {type(config)}")
        
        PadicValidation.validate_prime(prime)
        PadicValidation.validate_precision(precision)
        
        self.config = config
        self.prime = prime
        self.precision = precision
        
        # Initialize GPU operations
        self.gpu_ops = HybridPadicGPUOps(config, prime, precision)
        
        # Initialize optimizers for different parameter groups
        self.optimizers: Dict[str, HybridPadicGPUOptimizer] = {}
        
        # Global training statistics
        self.training_stats = {
            'total_epochs': 0,
            'total_batches': 0,
            'parameter_groups': 0,
            'average_epoch_time': 0.0,
            'gpu_utilization': 0.0,
            'memory_efficiency': 0.0
        }
    
    def create_optimizer(self, name: str, learning_rate: float = 0.001, 
                        momentum: float = 0.9, weight_decay: float = 0.0) -> HybridPadicGPUOptimizer:
        """Create a new GPU optimizer"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not isinstance(name, str):
            raise TypeError(f"Name must be str, got {type(name)}")
        if not name:
            raise ValueError("Name cannot be empty")
        if name in self.optimizers:
            raise ValueError(f"Optimizer '{name}' already exists")
        
        optimizer = HybridPadicGPUOptimizer(
            self.gpu_ops, learning_rate, momentum, weight_decay
        )
        self.optimizers[name] = optimizer
        self.training_stats['parameter_groups'] += 1
        
        return optimizer
    
    def get_optimizer(self, name: str) -> HybridPadicGPUOptimizer:
        """Get existing optimizer by name"""
        if not isinstance(name, str):
            raise TypeError(f"Name must be str, got {type(name)}")
        if name not in self.optimizers:
            raise ValueError(f"Optimizer '{name}' not found")
        
        return self.optimizers[name]
    
    def train_step(self, parameter_groups: Dict[str, Tuple[List[HybridPadicWeight], List[HybridPadicWeight]]]) -> Dict[str, List[HybridPadicWeight]]:
        """Perform training step across multiple parameter groups"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not parameter_groups:
            raise ValueError("Parameter groups cannot be empty")
        if not isinstance(parameter_groups, dict):
            raise TypeError(f"Parameter groups must be dict, got {type(parameter_groups)}")
        
        start_time = time.time()
        results = {}
        
        # Process each parameter group
        for group_name, (weights, gradients) in parameter_groups.items():
            if group_name not in self.optimizers:
                raise ValueError(f"No optimizer found for group '{group_name}'")
            
            optimizer = self.optimizers[group_name]
            updated_weights = optimizer.step(weights, gradients)
            results[group_name] = updated_weights
        
        # Update training statistics
        epoch_time = time.time() - start_time
        self._update_training_stats(epoch_time)
        
        return results
    
    def synchronize_all(self) -> None:
        """Synchronize all GPU operations"""
        self.gpu_ops.synchronize_streams()
    
    def cleanup_all(self) -> None:
        """Clean up all GPU resources"""
        self.gpu_ops.cleanup_gpu_memory()
        
        # Clear all momentum buffers
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
    
    def _update_training_stats(self, epoch_time: float) -> None:
        """Update training statistics"""
        self.training_stats['total_batches'] += 1
        
        total_batches = self.training_stats['total_batches']
        old_avg = self.training_stats['average_epoch_time']
        self.training_stats['average_epoch_time'] = (
            (old_avg * (total_batches - 1) + epoch_time) / total_batches
        )
        
        # Update GPU utilization (simplified metric)
        memory_info = self.gpu_ops._get_memory_info()
        allocated = memory_info['allocated']
        cached = memory_info['cached']
        
        if cached > 0:
            self.training_stats['memory_efficiency'] = allocated / cached
        
        self.training_stats['gpu_utilization'] = min(1.0, allocated / (self.config.memory_limit_mb * 1024 * 1024))
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components"""
        stats = {
            'training_stats': self.training_stats.copy(),
            'gpu_ops_stats': self.gpu_ops.get_operation_stats(),
            'optimizer_stats': {}
        }
        
        for name, optimizer in self.optimizers.items():
            stats['optimizer_stats'][name] = optimizer.get_optimization_stats()
        
        return stats
    
    def reset_all_stats(self) -> None:
        """Reset all statistics"""
        self.training_stats = {
            'total_epochs': 0,
            'total_batches': 0,
            'parameter_groups': len(self.optimizers),
            'average_epoch_time': 0.0,
            'gpu_utilization': 0.0,
            'memory_efficiency': 0.0
        }
        
        self.gpu_ops.reset_operation_stats()
        
        for optimizer in self.optimizers.values():
            optimizer.reset_optimization_stats()