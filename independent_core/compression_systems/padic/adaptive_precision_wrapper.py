"""
Fixed adaptive_precision_wrapper.py - Vectorized P-adic processing for performance
"""

import torch
import numpy as np
import math
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import concurrent.futures
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class AdaptivePrecisionConfig:
    """Configuration for adaptive precision wrapper"""
    prime: int = 257
    base_precision: int = 4
    min_precision: int = 2
    max_precision: int = 4
    target_error: float = 1e-6
    importance_threshold: float = 0.1
    batch_size: int = 1024
    enable_gpu_acceleration: bool = True
    enable_memory_tracking: bool = True
    enable_dynamic_switching: bool = True
    compression_priority: float = 0.5
    device: str = 'cpu'


@dataclass
class PrecisionAllocation:
    """Result of precision allocation"""
    padic_weights: List[Any]
    precision_map: torch.Tensor
    error_map: torch.Tensor
    compression_ratio: float
    bits_used: int
    original_shape: torch.Size
    
    def get_average_precision(self) -> float:
        """Calculate average precision across all weights"""
        if not self.padic_weights:
            return 0.0
        
        total_precision = 0.0
        valid_weights = 0
        
        for weight in self.padic_weights:
            if hasattr(weight, 'precision'):
                total_precision += weight.precision
                valid_weights += 1
        
        return total_precision / valid_weights if valid_weights > 0 else 0.0
    
    def get_error_statistics(self) -> Dict[str, float]:
        """Calculate error statistics from error map"""
        if self.error_map is None or self.error_map.numel() == 0:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0}
        
        error_flat = self.error_map.flatten().float()
        
        return {
            'mean': float(error_flat.mean()),
            'std': float(error_flat.std()),
            'max': float(error_flat.max()),
            'min': float(error_flat.min())
        }


class AdaptivePrecisionWrapper:
    """
    Optimized adaptive precision wrapper with vectorized processing
    """
    
    # Performance thresholds
    MAX_SERIAL_ELEMENTS = 10000  # Process serially up to this size
    BATCH_SIZE = 5000  # Process in batches for large tensors
    MAX_PARALLEL_WORKERS = 4  # Max parallel workers for batch processing
    
    def __init__(self, config: Optional[AdaptivePrecisionConfig] = None, math_ops: Any = None, device: Optional[torch.device] = None):
        """Initialize adaptive precision wrapper
        
        Args:
            config: Configuration for adaptive precision (uses defaults if None)
            math_ops: Mathematical operations object (optional for performance mode)
            device: Device to use for operations (auto-detects if None)
        """
        self.config = config or AdaptivePrecisionConfig()
        self.math_ops = math_ops
        self.device = device or getattr(config, 'device', 'cpu')
        
        # Cache for precision operations
        self._precision_ops_cache = {}
        # FIXED: Cache for precision-specific math operations
        self._precision_math_ops_cache = {}
        
        # Performance stats
        self.performance_stats = {
            'tensors_processed': 0,
            'total_elements': 0,
            'vectorized_operations': 0,
            'batched_operations': 0,
            'serial_operations': 0
        }
        
        # Precompute common values
        self.prime = getattr(self.config, 'prime', 251)
        self.bits_per_digit = math.log2(self.prime)
        
    def convert_tensor(self,
                      tensor: torch.Tensor,
                      importance_scores: Optional[torch.Tensor] = None) -> PrecisionAllocation:
        """
        Convert tensor to p-adic representation with adaptive precision.
        Uses vectorized operations for performance.
        """
        # Update stats
        self.performance_stats['tensors_processed'] += 1
        
        # Store original shape
        original_shape = tensor.shape
        
        # Flatten tensor for processing
        flat_tensor = tensor.flatten()
        batch_size = flat_tensor.size(0)
        self.performance_stats['total_elements'] += batch_size
        
        logger.info(f"Processing tensor with {batch_size} elements")
        
        # Calculate total bit budget
        bits_per_float = 32
        compression_ratio = getattr(self.config, 'compression_ratio', 0.5)
        total_bits = int(batch_size * bits_per_float * compression_ratio)
        
        # Generate or use provided importance scores
        if importance_scores is None:
            importance_scores = torch.abs(tensor)
        else:
            importance_scores = importance_scores.to(self.device)
        
        # Allocate precision based on importance
        precision_allocation = self.allocate_precision_by_importance(
            tensor, importance_scores, total_bits
        )
        
        # Choose processing strategy based on tensor size
        if batch_size <= self.MAX_SERIAL_ELEMENTS:
            # Small tensor: use optimized serial processing
            padic_weights, error_map = self._process_serial_optimized(
                flat_tensor, precision_allocation.flatten()
            )
            self.performance_stats['serial_operations'] += 1
            
        elif batch_size <= 100000:
            # Medium tensor: use batched processing
            padic_weights, error_map = self._process_batched(
                flat_tensor, precision_allocation.flatten()
            )
            self.performance_stats['batched_operations'] += 1
            
        else:
            # Large tensor: use parallel batched processing
            padic_weights, error_map = self._process_parallel_batched(
                flat_tensor, precision_allocation.flatten()
            )
            self.performance_stats['vectorized_operations'] += 1
        
        # Calculate actual compression ratio
        actual_bits = sum(self._calculate_bits_for_weight(w) for w in padic_weights)
        actual_ratio = actual_bits / (batch_size * bits_per_float)
        
        return PrecisionAllocation(
            padic_weights=padic_weights,
            precision_map=precision_allocation,
            error_map=error_map.reshape(original_shape),
            compression_ratio=actual_ratio,
            bits_used=actual_bits,
            original_shape=original_shape
        )
    
    def _process_serial_optimized(self, 
                                 flat_tensor: torch.Tensor,
                                 precision_allocation: torch.Tensor) -> Tuple[List[Any], torch.Tensor]:
        """
        Optimized serial processing for small tensors.
        Uses caching and minimal tensor operations.
        FIXED: Handles type system boundary violation with numpy.int64 → Python int conversion
        """
        batch_size = flat_tensor.size(0)
        padic_weights = []
        error_map = torch.zeros_like(flat_tensor)
        
        # Convert to numpy for faster indexing
        values = flat_tensor.cpu().numpy()
        precisions_numpy = precision_allocation.cpu().numpy().astype(int)  # numpy.int64
        
        # Group by precision level for batch processing
        precision_groups = {}
        for i in range(batch_size):
            # CRITICAL FIX: Convert numpy.int64 to Python int to avoid validation failures
            prec = int(precisions_numpy[i])  # Explicit conversion to Python int
            if prec not in precision_groups:
                precision_groups[prec] = []
            precision_groups[prec].append((i, values[i]))
        
        # Process each precision group
        for precision, indices_values in precision_groups.items():
            # precision is now guaranteed to be Python int
            precision_ops = self._get_precision_ops(precision)
            
            for idx, value in indices_values:
                try:
                    # Skip very small values
                    if abs(value) < 1e-10:
                        weight = self._create_zero_weight()
                        error = 0.0
                    else:
                        # Convert to p-adic
                        weight = precision_ops.to_padic(value)
                        
                        # Calculate error
                        reconstructed = precision_ops.from_padic(weight)
                        error = abs(value - reconstructed) / (abs(value) + 1e-10)
                    
                    padic_weights.append((idx, weight))
                    error_map[idx] = torch.tensor(error, dtype=error_map.dtype, device=error_map.device)
                    
                except (ValueError, OverflowError):
                    # Fallback for problematic values
                    weight = self.math_ops.to_padic(value)
                    padic_weights.append((idx, weight))
                    error_map[idx] = torch.tensor(self.config.target_error, dtype=error_map.dtype, device=error_map.device)
        
        # Sort by original index
        padic_weights.sort(key=lambda x: x[0])
        padic_weights = [w for _, w in padic_weights]
        
        return padic_weights, error_map
    
    def _process_batched(self,
                        flat_tensor: torch.Tensor,
                        precision_allocation: torch.Tensor) -> Tuple[List[Any], torch.Tensor]:
        """
        Process tensor in batches for better memory efficiency.
        """
        batch_size = flat_tensor.size(0)
        padic_weights = []
        error_map = torch.zeros_like(flat_tensor)
        
        # Process in batches
        for start_idx in range(0, batch_size, self.BATCH_SIZE):
            end_idx = min(start_idx + self.BATCH_SIZE, batch_size)
            
            # Extract batch
            batch_values = flat_tensor[start_idx:end_idx]
            batch_precisions = precision_allocation[start_idx:end_idx]
            
            # Process batch
            batch_weights, batch_errors = self._process_batch(
                batch_values, batch_precisions, start_idx
            )
            
            # Store results
            padic_weights.extend(batch_weights)
            error_map[start_idx:end_idx] = batch_errors.to(error_map.device)
        
        return padic_weights, error_map
    
    def _process_batch(self,
                      batch_values: torch.Tensor,
                      batch_precisions: torch.Tensor,
                      offset: int) -> Tuple[List[Any], torch.Tensor]:
        """
        Process a single batch of values.
        FIXED: Handles type system boundary violation with numpy.int64 → Python int conversion
        """
        batch_size = batch_values.size(0)
        batch_weights = []
        batch_errors = torch.zeros_like(batch_values.cpu())
        
        # Convert to numpy for faster processing
        values = batch_values.cpu().numpy()
        precisions_numpy = batch_precisions.cpu().numpy().astype(int)  # numpy.int64
        
        # Process each element with type conversion
        for i in range(batch_size):
            value = float(values[i])  # Convert to Python float
            # CRITICAL FIX: Convert numpy.int64 to Python int to avoid validation failures
            precision = int(precisions_numpy[i])  # Explicit conversion to Python int
            
            # Get precision operations (now with Python int)
            precision_ops = self._get_precision_ops(precision)
            
            try:
                if abs(value) < 1e-10:
                    weight = self._create_zero_weight()
                    error = 0.0
                else:
                    weight = precision_ops.to_padic(value)
                    reconstructed = precision_ops.from_padic(weight)
                    error = abs(value - reconstructed) / (abs(value) + 1e-10)
                
                batch_weights.append(weight)
                batch_errors[i] = error
                
            except (ValueError, OverflowError):
                weight = self.math_ops.to_padic(value)
                batch_weights.append(weight)
                batch_errors[i] = self.config.target_error
        
        return batch_weights, batch_errors
    
    def _process_parallel_batched(self,
                                 flat_tensor: torch.Tensor,
                                 precision_allocation: torch.Tensor) -> Tuple[List[Any], torch.Tensor]:
        """
        Process large tensors using parallel batch processing.
        """
        batch_size = flat_tensor.size(0)
        num_batches = (batch_size + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        
        logger.info(f"Processing {batch_size} elements in {num_batches} parallel batches")
        
        # Prepare batches
        batches = []
        for i in range(num_batches):
            start_idx = i * self.BATCH_SIZE
            end_idx = min(start_idx + self.BATCH_SIZE, batch_size)
            batches.append((
                flat_tensor[start_idx:end_idx],
                precision_allocation[start_idx:end_idx],
                start_idx
            ))
        
        # Process batches in parallel
        all_weights = [None] * num_batches
        all_errors = [None] * num_batches
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_PARALLEL_WORKERS) as executor:
            futures = {
                executor.submit(self._process_batch, batch[0], batch[1], batch[2]): i
                for i, batch in enumerate(batches)
            }
            
            for future in concurrent.futures.as_completed(futures):
                batch_idx = futures[future]
                try:
                    weights, errors = future.result()
                    all_weights[batch_idx] = weights
                    all_errors[batch_idx] = errors
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    # Fallback to serial processing for this batch
                    batch = batches[batch_idx]
                    weights, errors = self._process_batch(batch[0], batch[1], batch[2])
                    all_weights[batch_idx] = weights
                    all_errors[batch_idx] = errors
        
        # Combine results
        padic_weights = []
        error_map = torch.zeros_like(flat_tensor)
        
        for i, (weights, errors) in enumerate(zip(all_weights, all_errors)):
            if weights is not None:
                padic_weights.extend(weights)
                start_idx = i * self.BATCH_SIZE
                end_idx = min(start_idx + self.BATCH_SIZE, batch_size)
                error_map[start_idx:end_idx] = errors
        
        return padic_weights, error_map
    
    def allocate_precision_by_importance(self,
                                        tensor: torch.Tensor,
                                        importance_scores: torch.Tensor,
                                        total_bits: int) -> torch.Tensor:
        """
        Allocate precision based on importance scores using vectorized operations.
        FIXED: Respects safe precision limits for the current prime
        """
        flat_importance = importance_scores.flatten()
        batch_size = flat_importance.size(0)
        
        # Normalize importance scores
        importance_sum = flat_importance.sum()
        if importance_sum > 0:
            normalized_importance = flat_importance / importance_sum
        else:
            normalized_importance = torch.ones_like(flat_importance) / batch_size
        
        # Calculate precision for each element
        bits_per_element = normalized_importance * total_bits
        
        # Convert to precision levels (digits in p-adic representation)
        precision_levels = (bits_per_element / self.bits_per_digit).ceil().int()
        
        # CRITICAL FIX: Clamp to safe precision limits for current prime
        min_precision = getattr(self.config, 'min_precision', 4)
        max_precision = getattr(self.config, 'max_precision', 32)
        
        # Import safe precision function
        from .padic_encoder import get_safe_precision
        safe_max_precision = get_safe_precision(self.prime)
        
        # Use the minimum of configured max and safe max
        effective_max_precision = min(max_precision, safe_max_precision)
        
        # Clamp to safe range
        precision_levels = torch.clamp(precision_levels, min_precision, effective_max_precision)
        
        logger.debug(f"Precision allocation: range=[{min_precision}, {effective_max_precision}], "
                    f"safe_limit={safe_max_precision}, configured_max={max_precision}")
        
        return precision_levels.reshape(tensor.shape)
    
    def compute_adaptive_precision(self, tensor: torch.Tensor, 
                                  target_error: Optional[float] = None) -> Tuple[List[Any], torch.Tensor]:
        """
        Compute adaptive precision for each weight (legacy interface).
        
        Args:
            tensor: Input tensor of weights
            target_error: Target error threshold (uses config default if None)
            
        Returns:
            - padic_weights: Variable precision p-adic representation
            - precision_map: Actual precision used per weight
        """
        result = self.convert_tensor(tensor)
        return result.padic_weights, result.precision_map
    
    def batch_compress_with_adaptive_precision(self, tensor: torch.Tensor,
                                              importance_scores: Optional[torch.Tensor] = None,
                                              total_bits: Optional[int] = None) -> PrecisionAllocation:
        """
        Batch processing of tensors with adaptive precision allocation (legacy interface).
        
        Args:
            tensor: Input tensor to compress
            importance_scores: Optional importance scores for each weight
            total_bits: Optional total bit budget (calculated if None)
            
        Returns:
            PrecisionAllocation with compressed weights and statistics
        """
        return self.convert_tensor(tensor, importance_scores)
    
    def _get_math_ops_for_precision(self, precision: int):
        """
        Get or create PadicMathematicalOperations instance for specific precision
        This is the KEY FIX: Create precision-specific instances instead of passing precision as parameter
        """
        if precision not in self._precision_math_ops_cache:
            # Create new instance with specific precision using factory method
            if hasattr(self.math_ops, 'create_with_precision'):
                self._precision_math_ops_cache[precision] = self.math_ops.create_with_precision(precision)
            else:
                # Fallback: create new instance manually
                from .padic_encoder import PadicMathematicalOperations
                self._precision_math_ops_cache[precision] = PadicMathematicalOperations(
                    self.math_ops.prime, precision
                )
        return self._precision_math_ops_cache[precision]
    
    @lru_cache(maxsize=128)
    def _get_precision_ops(self, precision: int):
        """
        FIXED: Create precision operations with proper method binding signatures
        
        CRITICAL FIX: Lambda signatures now include self parameter to match Python's method binding contract.
        When precision_ops.to_padic(value) is called, Python transforms it to
        PrecisionOps.to_padic(precision_ops, value), so the lambda must accept (self, x).
        
        Also includes defensive type conversion for numpy integers.
        """
        # Defensive type conversion: handle numpy.int64 and similar types
        if hasattr(precision, 'item'):  # numpy scalar
            precision = int(precision.item())
        elif not isinstance(precision, int):
            precision = int(precision)  # Convert any integer-like type
            
        if precision not in self._precision_ops_cache:
            # Get precision-specific math operations
            precision_math_ops = self._get_math_ops_for_precision(precision)
            
            # Create mock object with proper method signatures
            # CRITICAL FIX: Include self parameter in lambdas to match method binding contract
            self._precision_ops_cache[precision] = type('PrecisionOps', (), {
                'to_padic': lambda self, x: precision_math_ops.to_padic(x),
                #                  ^^^^  ^   ^^^^^^^^^^^^^^^^^^^ 
                #                  self  arg precision-specific ops
                #                  (required for method binding)
                
                'from_padic': lambda self, w: precision_math_ops.from_padic(w),
                #                    ^^^^  ^   ^^^^^^^^^^^^^^^^^^^
                #                    self  arg precision-specific ops  
                #                    (required for method binding)
                
                'math_ops': precision_math_ops  # Expose the underlying ops
            })()  # Empty constructor since we don't need parent reference
        
        return self._precision_ops_cache[precision]
    
    def _create_zero_weight(self):
        """Create a zero weight in p-adic representation"""
        return self.math_ops.to_padic(0.0)
    
    def _calculate_bits_for_weight(self, weight) -> int:
        """Calculate the number of bits used by a p-adic weight"""
        if weight is None or weight == 0:
            return 1
        
        # Estimate based on p-adic representation
        # This would depend on your actual p-adic implementation
        if hasattr(weight, '__len__'):
            return len(weight) * int(self.bits_per_digit)
        else:
            return int(self.bits_per_digit * 4)  # Default estimate
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_ops = (
            self.performance_stats['serial_operations'] +
            self.performance_stats['batched_operations'] +
            self.performance_stats['vectorized_operations']
        )
        
        return {
            'tensors_processed': self.performance_stats['tensors_processed'],
            'total_elements': self.performance_stats['total_elements'],
            'average_elements_per_tensor': (
                self.performance_stats['total_elements'] / 
                self.performance_stats['tensors_processed']
                if self.performance_stats['tensors_processed'] > 0 else 0
            ),
            'processing_strategies': {
                'serial': self.performance_stats['serial_operations'],
                'batched': self.performance_stats['batched_operations'],
                'parallel': self.performance_stats['vectorized_operations']
            },
            'strategy_percentages': {
                'serial': (
                    100 * self.performance_stats['serial_operations'] / total_ops
                    if total_ops > 0 else 0
                ),
                'batched': (
                    100 * self.performance_stats['batched_operations'] / total_ops
                    if total_ops > 0 else 0
                ),
                'parallel': (
                    100 * self.performance_stats['vectorized_operations'] / total_ops
                    if total_ops > 0 else 0
                )
            }
        }


def create_adaptive_wrapper(prime: int = 257, **kwargs) -> AdaptivePrecisionWrapper:
    """
    Factory function to create adaptive wrapper with safe defaults.
    
    Args:
        prime: Prime number for p-adic representation
        **kwargs: Additional configuration options
        
    Returns:
        Configured AdaptivePrecisionWrapper instance
    """
    from .padic_encoder import get_safe_precision
    
    # Get safe precision limit
    safe_max_precision = get_safe_precision(prime)
    
    # Create configuration with safe defaults
    config = AdaptivePrecisionConfig(
        prime=prime,
        base_precision=min(4, safe_max_precision),
        min_precision=2,
        max_precision=safe_max_precision,
        target_error=kwargs.get('target_error', 1e-6),
        importance_threshold=kwargs.get('importance_threshold', 0.1),
        batch_size=kwargs.get('batch_size', 1024),
        enable_gpu_acceleration=kwargs.get('enable_gpu_acceleration', True),
        enable_memory_tracking=kwargs.get('enable_memory_tracking', True),
        enable_dynamic_switching=kwargs.get('enable_dynamic_switching', True),
        compression_priority=kwargs.get('compression_priority', 0.5),
        device=kwargs.get('device', 'cpu')
    )
    
    return AdaptivePrecisionWrapper(config)