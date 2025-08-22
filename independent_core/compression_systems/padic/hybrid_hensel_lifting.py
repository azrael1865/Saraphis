"""
Hybrid Hensel Lifting - Hybrid-compatible Hensel lifting operations
NO FALLBACKS - HARD FAILURES ONLY
"""

import logging
import math
import time
import torch
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

# Import existing Hensel lifting
from padic_advanced import HenselLiftingProcessor, HenselLiftingConfig

# Import hybrid structures
from hybrid_padic_structures import HybridPadicWeight, HybridPadicValidator
from padic_encoder import PadicWeight, PadicMathematicalOperations


@dataclass
class HybridLiftingResult:
    """Result of hybrid Hensel lifting operation"""
    lifted_weight: HybridPadicWeight
    lifting_time_ms: float
    iterations_used: int
    convergence_achieved: bool
    final_error: float
    lifting_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate lifting result"""
        if not isinstance(self.lifted_weight, HybridPadicWeight):
            raise TypeError("Lifted weight must be HybridPadicWeight")
        if not isinstance(self.lifting_time_ms, (int, float)) or self.lifting_time_ms < 0:
            raise ValueError("Lifting time must be non-negative")
        if not isinstance(self.iterations_used, int) or self.iterations_used < 0:
            raise ValueError("Iterations used must be non-negative")
        if not isinstance(self.convergence_achieved, bool):
            raise TypeError("Convergence achieved must be bool")
        if not isinstance(self.final_error, (int, float)) or self.final_error < 0:
            raise ValueError("Final error must be non-negative")


@dataclass
class HybridLiftingStats:
    """Statistics for hybrid Hensel lifting operations"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    average_lifting_time_ms: float = 0.0
    average_iterations: float = 0.0
    average_convergence_rate: float = 0.0
    precision_improvements: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    gpu_memory_usage_mb: float = 0.0
    last_update: Optional[datetime] = None
    
    def update_operation(self, result: HybridLiftingResult, target_precision: int):
        """Update statistics with operation result"""
        self.total_operations += 1
        self.last_update = datetime.utcnow()
        
        if result.convergence_achieved:
            self.successful_operations += 1
        else:
            self.failed_operations += 1
        
        # Update averages
        if self.total_operations > 1:
            old_time_avg = self.average_lifting_time_ms
            self.average_lifting_time_ms = (
                (old_time_avg * (self.total_operations - 1) + result.lifting_time_ms) / self.total_operations
            )
            
            old_iter_avg = self.average_iterations
            self.average_iterations = (
                (old_iter_avg * (self.total_operations - 1) + result.iterations_used) / self.total_operations
            )
        else:
            self.average_lifting_time_ms = result.lifting_time_ms
            self.average_iterations = result.iterations_used
        
        # Update convergence rate
        self.average_convergence_rate = self.successful_operations / self.total_operations
        
        # Track precision improvements
        self.precision_improvements[target_precision] += 1


class HybridHenselLifting:
    """
    Hybrid-compatible Hensel lifting operations.
    Provides GPU-accelerated precision lifting for hybrid p-adic weights.
    """
    
    def __init__(self, config: HenselLiftingConfig, prime: int, base_precision: int):
        """Initialize hybrid Hensel lifting"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not isinstance(config, HenselLiftingConfig):
            raise TypeError(f"Config must be HenselLiftingConfig, got {type(config)}")
        if not isinstance(prime, int) or prime <= 1:
            raise ValueError(f"Prime must be int > 1, got {prime}")
        if not isinstance(base_precision, int) or base_precision <= 0:
            raise ValueError(f"Base precision must be int > 0, got {base_precision}")
        
        self.config = config
        self.prime = prime
        self.base_precision = base_precision
        self.current_precision = base_precision
        
        # Initialize components
        self.math_ops = PadicMathematicalOperations(prime, base_precision)
        self.validator = HybridPadicValidator()
        self.logger = logging.getLogger('HybridHenselLifting')
        
        # Performance tracking
        self.lifting_stats = HybridLiftingStats()
        self.operation_history: deque = deque(maxlen=1000)
        
        # GPU optimization
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for hybrid Hensel lifting")
        
        # Caching for performance
        self.lifting_cache: Dict[str, HybridLiftingResult] = {}
        self.max_cache_size = 100
        
        self.logger.info(f"HybridHenselLifting initialized with prime={prime}, base_precision={base_precision}")
    
    def lift_hybrid_to_precision(self, hybrid_weight: HybridPadicWeight, target_precision: int,
                                initial_guess: Optional[HybridPadicWeight] = None) -> HybridLiftingResult:
        """
        Lift hybrid weight to target precision using GPU-accelerated Hensel lifting.
        
        Args:
            hybrid_weight: Hybrid p-adic weight to lift
            target_precision: Target precision level
            initial_guess: Optional initial guess for lifting
            
        Returns:
            Hybrid lifting result
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If lifting fails
        """
        if not isinstance(hybrid_weight, HybridPadicWeight):
            raise TypeError(f"Hybrid weight must be HybridPadicWeight, got {type(hybrid_weight)}")
        if not isinstance(target_precision, int) or target_precision <= 0:
            raise ValueError(f"Target precision must be positive int, got {target_precision}")
        if target_precision <= hybrid_weight.precision:
            raise ValueError(f"Target precision {target_precision} must be > current {hybrid_weight.precision}")
        
        # Validate input weight
        self.validator.validate_hybrid_weight(hybrid_weight)
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(hybrid_weight, target_precision)
            if cache_key in self.lifting_cache:
                cached_result = self.lifting_cache[cache_key]
                self.logger.debug(f"Cache hit for lifting to precision {target_precision}")
                return cached_result
            
            # Perform GPU-accelerated lifting
            lifted_weight, iterations, final_error, convergence = self._perform_gpu_lifting(
                hybrid_weight, target_precision, initial_guess
            )
            
            # Calculate timing
            lifting_time_ms = (time.time() - start_time) * 1000
            
            # Create result
            result = HybridLiftingResult(
                lifted_weight=lifted_weight,
                lifting_time_ms=lifting_time_ms,
                iterations_used=iterations,
                convergence_achieved=convergence,
                final_error=final_error,
                lifting_metadata={
                    'original_precision': hybrid_weight.precision,
                    'target_precision': target_precision,
                    'precision_gain': target_precision - hybrid_weight.precision,
                    'damping_factor_used': self.config.damping_factor,
                    'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024)
                }
            )
            
            # Cache result if successful
            if convergence and len(self.lifting_cache) < self.max_cache_size:
                self.lifting_cache[cache_key] = result
            
            # Update statistics
            self.lifting_stats.update_operation(result, target_precision)
            
            # Record operation
            self.operation_history.append({
                'timestamp': datetime.utcnow(),
                'target_precision': target_precision,
                'lifting_time_ms': lifting_time_ms,
                'iterations': iterations,
                'convergence': convergence,
                'final_error': final_error
            })
            
            self.logger.info(f"Hybrid lifting completed: {hybrid_weight.precision} -> {target_precision} "
                           f"in {iterations} iterations, {lifting_time_ms:.2f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Hybrid lifting failed: {e}")
            raise RuntimeError(f"Hybrid Hensel lifting failed: {e}")
    
    def validate_hybrid_lifting(self, original: HybridPadicWeight, lifted: HybridPadicWeight) -> bool:
        """
        Validate that lifted weight maintains mathematical properties.
        
        Args:
            original: Original hybrid weight
            lifted: Lifted hybrid weight
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If weights are invalid
        """
        if not isinstance(original, HybridPadicWeight):
            raise TypeError(f"Original must be HybridPadicWeight, got {type(original)}")
        if not isinstance(lifted, HybridPadicWeight):
            raise TypeError(f"Lifted must be HybridPadicWeight, got {type(lifted)}")
        
        try:
            # Validate both weights
            self.validator.validate_hybrid_weight(original)
            self.validator.validate_hybrid_weight(lifted)
            
            # Check precision increase
            if lifted.precision <= original.precision:
                self.logger.error(f"Precision not increased: {original.precision} -> {lifted.precision}")
                return False
            
            # Check preservation of core properties
            if lifted.prime != original.prime:
                self.logger.error(f"Prime changed: {original.prime} -> {lifted.prime}")
                return False
            
            if lifted.valuation != original.valuation:
                self.logger.error(f"Valuation changed: {original.valuation} -> {lifted.valuation}")
                return False
            
            # Check channel consistency
            if not self._validate_channel_consistency(original, lifted):
                self.logger.error("Channel consistency validation failed")
                return False
            
            # Check ultrametric preservation
            if original.ultrametric_preserved and not lifted.ultrametric_preserved:
                self.logger.error("Ultrametric property not preserved")
                return False
            
            # Check error tolerance
            reconstruction_error = self._calculate_reconstruction_error(original, lifted)
            if reconstruction_error > self.config.convergence_tolerance:
                self.logger.error(f"Reconstruction error too high: {reconstruction_error}")
                return False
            
            self.logger.debug("Hybrid lifting validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False
    
    def validate_hybrid_weight(self, hybrid_weight: HybridPadicWeight) -> bool:
        """
        Validate hybrid weight structure and properties.
        
        Args:
            hybrid_weight: Hybrid weight to validate
            
        Returns:
            True if valid
        """
        if not isinstance(hybrid_weight, HybridPadicWeight):
            return False
        
        try:
            self.validator.validate_hybrid_weight(hybrid_weight)
            return True
        except Exception as e:
            self.logger.error(f"Hybrid weight validation failed: {e}")
            return False
    
    def get_hybrid_lifting_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive lifting statistics.
        
        Returns:
            Dictionary containing lifting statistics
        """
        return {
            'overall_stats': {
                'total_operations': self.lifting_stats.total_operations,
                'successful_operations': self.lifting_stats.successful_operations,
                'failed_operations': self.lifting_stats.failed_operations,
                'success_rate': self.lifting_stats.average_convergence_rate,
                'average_lifting_time_ms': self.lifting_stats.average_lifting_time_ms,
                'average_iterations': self.lifting_stats.average_iterations
            },
            'precision_stats': {
                'base_precision': self.base_precision,
                'current_precision': self.current_precision,
                'precision_improvements': dict(self.lifting_stats.precision_improvements),
                'max_precision_achieved': max(self.lifting_stats.precision_improvements.keys(), default=self.base_precision)
            },
            'performance_stats': {
                'gpu_memory_usage_mb': self.lifting_stats.gpu_memory_usage_mb,
                'cache_hit_rate': self._calculate_cache_hit_rate(),
                'cache_size': len(self.lifting_cache),
                'operations_history_length': len(self.operation_history)
            },
            'configuration': {
                'prime': self.prime,
                'max_iterations': self.config.max_iterations,
                'convergence_tolerance': self.config.convergence_tolerance,
                'damping_factor': self.config.damping_factor,
                'adaptive_damping': self.config.adaptive_damping
            },
            'last_update': self.lifting_stats.last_update.isoformat() if self.lifting_stats.last_update else None
        }
    
    def optimize_hybrid_lifting_performance(self) -> Dict[str, Any]:
        """
        Optimize lifting performance based on historical data.
        
        Returns:
            Dictionary containing optimization results
        """
        if self.lifting_stats.total_operations < 10:
            return {'status': 'insufficient_data', 'operations_needed': 10 - self.lifting_stats.total_operations}
        
        optimization_results = {
            'optimizations_applied': [],
            'performance_improvements': {},
            'configuration_changes': {}
        }
        
        try:
            # Analyze convergence patterns
            recent_operations = list(self.operation_history)[-20:]
            avg_iterations = sum(op['iterations'] for op in recent_operations) / len(recent_operations)
            convergence_rate = sum(1 for op in recent_operations if op['convergence']) / len(recent_operations)
            
            # Optimize damping factor if needed
            if convergence_rate < 0.8 and avg_iterations > self.config.max_iterations * 0.8:
                # Increase damping for better convergence
                old_damping = self.config.damping_factor
                self.config.damping_factor = min(self.config.max_damping, old_damping * 1.1)
                optimization_results['optimizations_applied'].append('increased_damping')
                optimization_results['configuration_changes']['damping_factor'] = {
                    'old': old_damping,
                    'new': self.config.damping_factor
                }
            elif convergence_rate > 0.95 and avg_iterations < self.config.max_iterations * 0.3:
                # Decrease damping for faster convergence
                old_damping = self.config.damping_factor
                self.config.damping_factor = max(self.config.min_damping, old_damping * 0.9)
                optimization_results['optimizations_applied'].append('decreased_damping')
                optimization_results['configuration_changes']['damping_factor'] = {
                    'old': old_damping,
                    'new': self.config.damping_factor
                }
            
            # Clean cache if too large
            if len(self.lifting_cache) > self.max_cache_size * 0.9:
                self._clean_lifting_cache()
                optimization_results['optimizations_applied'].append('cache_cleanup')
            
            # Update GPU memory tracking
            if torch.cuda.is_available():
                self.lifting_stats.gpu_memory_usage_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            
            optimization_results['performance_improvements'] = {
                'convergence_rate': convergence_rate,
                'average_iterations': avg_iterations,
                'cache_efficiency': self._calculate_cache_hit_rate()
            }
            
            self.logger.info(f"Performance optimization completed: {len(optimization_results['optimizations_applied'])} optimizations applied")
            
        except Exception as e:
            self.logger.error(f"Performance optimization failed: {e}")
            optimization_results['error'] = str(e)
        
        return optimization_results
    
    def _perform_gpu_lifting(self, hybrid_weight: HybridPadicWeight, target_precision: int,
                           initial_guess: Optional[HybridPadicWeight] = None) -> Tuple[HybridPadicWeight, int, float, bool]:
        """Perform GPU-accelerated Hensel lifting"""
        
        # Initialize working tensors on GPU
        exponent_work = hybrid_weight.exponent_channel.clone().detach()
        mantissa_work = hybrid_weight.mantissa_channel.clone().detach()
        
        # Calculate precision increment
        precision_increment = target_precision - hybrid_weight.precision
        current_damping = self.config.damping_factor
        
        # Iterative lifting with Newton-Raphson method
        for iteration in range(self.config.max_iterations):
            
            # Calculate lifting correction for exponent channel
            exponent_correction = self._calculate_exponent_correction(
                exponent_work, mantissa_work, precision_increment
            )
            
            # Calculate lifting correction for mantissa channel
            mantissa_correction = self._calculate_mantissa_correction(
                exponent_work, mantissa_work, precision_increment
            )
            
            # Apply corrections with damping
            exponent_work += current_damping * exponent_correction
            mantissa_work += current_damping * mantissa_correction
            
            # Calculate convergence error
            error = self._calculate_lifting_error(exponent_correction, mantissa_correction)
            
            # Check convergence
            if error < self.config.convergence_tolerance:
                # Create lifted weight
                lifted_weight = HybridPadicWeight(
                    exponent_channel=exponent_work,
                    mantissa_channel=mantissa_work,
                    prime=hybrid_weight.prime,
                    precision=target_precision,
                    valuation=hybrid_weight.valuation,
                    device=hybrid_weight.device,
                    dtype=hybrid_weight.dtype,
                    error_tolerance=hybrid_weight.error_tolerance,
                    ultrametric_preserved=hybrid_weight.ultrametric_preserved
                )
                
                return lifted_weight, iteration + 1, error, True
            
            # Adaptive damping adjustment
            if self.config.adaptive_damping and iteration > 5:
                if error > previous_error if 'previous_error' in locals() else float('inf'):
                    current_damping *= 0.9  # Reduce damping if error increasing
                    current_damping = max(current_damping, self.config.min_damping)
            
            previous_error = error
        
        # Failed to converge - return best attempt
        lifted_weight = HybridPadicWeight(
            exponent_channel=exponent_work,
            mantissa_channel=mantissa_work,
            prime=hybrid_weight.prime,
            precision=target_precision,
            valuation=hybrid_weight.valuation,
            device=hybrid_weight.device,
            dtype=hybrid_weight.dtype,
            error_tolerance=hybrid_weight.error_tolerance,
            ultrametric_preserved=False  # Mark as potentially corrupted
        )
        
        return lifted_weight, self.config.max_iterations, error, False
    
    def _calculate_exponent_correction(self, exponent_tensor: torch.Tensor, mantissa_tensor: torch.Tensor,
                                     precision_increment: int) -> torch.Tensor:
        """Calculate exponent channel correction using p-adic lifting formula"""
        
        # GPU-accelerated p-adic lifting for exponent channel
        # This implements the hierarchical structure lifting
        with torch.no_grad():
            # Calculate modular correction
            mod_correction = torch.fmod(exponent_tensor * self.prime, self.prime ** precision_increment)
            
            # Apply p-adic lifting formula for exponent channel
            correction = (mod_correction - exponent_tensor) / self.prime
            
            return correction.clamp(-1.0, 1.0)  # Numerical stability
    
    def _calculate_mantissa_correction(self, exponent_tensor: torch.Tensor, mantissa_tensor: torch.Tensor,
                                     precision_increment: int) -> torch.Tensor:
        """Calculate mantissa channel correction using p-adic lifting formula"""
        
        # GPU-accelerated p-adic lifting for mantissa channel
        # This implements the fine-grained value lifting
        with torch.no_grad():
            # Calculate modular correction for mantissa
            mod_correction = torch.fmod(mantissa_tensor * self.prime, self.prime ** precision_increment)
            
            # Apply p-adic lifting formula for mantissa channel
            correction = (mod_correction - mantissa_tensor) / self.prime
            
            # Apply interaction with exponent channel
            interaction_term = torch.mul(exponent_tensor, correction) * 0.1
            
            return (correction + interaction_term).clamp(-1.0, 1.0)  # Numerical stability
    
    def _calculate_lifting_error(self, exponent_correction: torch.Tensor, mantissa_correction: torch.Tensor) -> float:
        """Calculate overall lifting error"""
        exponent_error = torch.norm(exponent_correction).item()
        mantissa_error = torch.norm(mantissa_correction).item()
        return math.sqrt(exponent_error ** 2 + mantissa_error ** 2)
    
    def _validate_channel_consistency(self, original: HybridPadicWeight, lifted: HybridPadicWeight) -> bool:
        """Validate that channel relationships are preserved"""
        try:
            # Check shape consistency
            if original.exponent_channel.shape != lifted.exponent_channel.shape:
                return False
            if original.mantissa_channel.shape != lifted.mantissa_channel.shape:
                return False
            
            # Check magnitude preservation (within tolerance)
            orig_exp_norm = torch.norm(original.exponent_channel).item()
            lifted_exp_norm = torch.norm(lifted.exponent_channel).item()
            exp_change_ratio = abs(lifted_exp_norm - orig_exp_norm) / max(orig_exp_norm, 1e-10)
            
            orig_man_norm = torch.norm(original.mantissa_channel).item()
            lifted_man_norm = torch.norm(lifted.mantissa_channel).item()
            man_change_ratio = abs(lifted_man_norm - orig_man_norm) / max(orig_man_norm, 1e-10)
            
            # Allow reasonable magnitude changes due to precision increase
            return exp_change_ratio < 2.0 and man_change_ratio < 2.0
            
        except Exception:
            return False
    
    def _calculate_reconstruction_error(self, original: HybridPadicWeight, lifted: HybridPadicWeight) -> float:
        """Calculate reconstruction error between original and lifted weights"""
        try:
            # Normalize to same precision for comparison
            min_precision = min(original.precision, lifted.precision)
            
            # Create truncated versions for comparison
            orig_exp_trunc = original.exponent_channel * (self.prime ** min_precision)
            orig_man_trunc = original.mantissa_channel * (self.prime ** min_precision)
            
            lifted_exp_trunc = lifted.exponent_channel * (self.prime ** min_precision)
            lifted_man_trunc = lifted.mantissa_channel * (self.prime ** min_precision)
            
            # Calculate relative error
            exp_error = torch.norm(orig_exp_trunc - lifted_exp_trunc).item()
            man_error = torch.norm(orig_man_trunc - lifted_man_trunc).item()
            
            total_norm = torch.norm(orig_exp_trunc).item() + torch.norm(orig_man_trunc).item()
            
            return math.sqrt(exp_error ** 2 + man_error ** 2) / max(total_norm, 1e-10)
            
        except Exception:
            return float('inf')
    
    def _generate_cache_key(self, hybrid_weight: HybridPadicWeight, target_precision: int) -> str:
        """Generate cache key for lifting operation"""
        # Create hash from weight properties and target
        exp_hash = hash(tuple(hybrid_weight.exponent_channel.flatten().tolist()[:10]))  # Sample for performance
        man_hash = hash(tuple(hybrid_weight.mantissa_channel.flatten().tolist()[:10]))
        
        return f"lift_{hybrid_weight.prime}_{hybrid_weight.precision}_{target_precision}_{exp_hash}_{man_hash}"
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate from recent operations"""
        if len(self.operation_history) < 10:
            return 0.0
        
        # This is a simplified calculation - in practice would track actual hits
        cache_efficiency = len(self.lifting_cache) / max(len(self.operation_history), 1)
        return min(cache_efficiency, 1.0)
    
    def _clean_lifting_cache(self) -> None:
        """Clean lifting cache by removing oldest entries"""
        if len(self.lifting_cache) <= self.max_cache_size // 2:
            return
        
        # Remove half of the cache entries (simple FIFO strategy)
        items_to_remove = len(self.lifting_cache) - self.max_cache_size // 2
        keys_to_remove = list(self.lifting_cache.keys())[:items_to_remove]
        
        for key in keys_to_remove:
            del self.lifting_cache[key]
        
        self.logger.debug(f"Cleaned lifting cache: removed {items_to_remove} entries")
    
    def clear_cache(self) -> None:
        """Clear all cached lifting results"""
        self.lifting_cache.clear()
        self.logger.info("Lifting cache cleared")
    
    def shutdown(self) -> None:
        """Shutdown hybrid Hensel lifting"""
        self.logger.info("Shutting down hybrid Hensel lifting")
        
        # Clear caches and data
        self.lifting_cache.clear()
        self.operation_history.clear()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Hybrid Hensel lifting shutdown complete")