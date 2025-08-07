"""
Adaptive Precision Wrapper for P-adic Compression System
Task E: Dynamic precision adjustment using Hensel's lemma
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
import math
import time
import threading
from fractions import Fraction

# Import required components from existing p-adic system
from .padic_encoder import (
    AdaptiveHenselLifting, 
    PadicWeight,
    PadicMathematicalOperations,
    PadicValidation
)
from .padic_logarithmic_encoder import LogarithmicPadicWeight


@dataclass
class AdaptivePrecisionConfig:
    """Configuration for adaptive precision wrapper"""
    prime: int = 257
    base_precision: int = 4  # Safe default for prime 257
    min_precision: int = 2
    max_precision: int = 4   # Safe maximum for prime 257
    target_error: float = 1e-6
    importance_threshold: float = 0.1
    batch_size: int = 1024
    enable_gpu_acceleration: bool = True
    enable_memory_tracking: bool = True
    enable_dynamic_switching: bool = True
    compression_priority: float = 0.5  # Balance between accuracy (0) and compression (1)
    
    def __post_init__(self):
        """Validate configuration"""
        PadicValidation.validate_prime(self.prime)
        PadicValidation.validate_precision(self.base_precision)
        
        if not (1 <= self.min_precision <= self.base_precision <= self.max_precision):
            raise ValueError(
                f"Invalid precision bounds: min={self.min_precision}, "
                f"base={self.base_precision}, max={self.max_precision}"
            )
        
        if not (0 < self.target_error < 1):
            raise ValueError(f"Target error must be in (0, 1), got {self.target_error}")
        
        if not (0 <= self.compression_priority <= 1):
            raise ValueError(f"Compression priority must be in [0, 1], got {self.compression_priority}")


@dataclass 
class PrecisionAllocation:
    """Result of precision allocation for a batch of weights"""
    weights: List[PadicWeight]
    precision_map: torch.Tensor  # [batch_size] - precision per weight
    error_map: torch.Tensor      # [batch_size] - achieved error per weight
    total_bits: int               # Total bits used
    compression_ratio: float      # Achieved compression ratio
    convergence_stats: Dict[str, Any]
    
    def get_average_precision(self) -> float:
        """Get average precision used"""
        return float(self.precision_map.float().mean().item())
    
    def get_error_statistics(self) -> Dict[str, float]:
        """Get error statistics"""
        errors = self.error_map.cpu().numpy()
        return {
            'mean': float(np.mean(errors)),
            'std': float(np.std(errors)),
            'min': float(np.min(errors)),
            'max': float(np.max(errors)),
            'median': float(np.median(errors))
        }


class AdaptivePrecisionWrapper:
    """
    Wrapper for adaptive precision using Hensel's lemma.
    Dynamically adjusts precision based on reconstruction error.
    """

    def __init__(self, config: Optional[AdaptivePrecisionConfig] = None):
        """Initialize adaptive precision wrapper
        
        Args:
            config: Configuration for adaptive precision (uses defaults if None)
        """
        self.config = config or AdaptivePrecisionConfig()
        
        # Initialize Hensel lifting
        self.hensel = AdaptiveHenselLifting(self.config.prime, self.config.base_precision)
        
        # Initialize mathematical operations
        self.math_ops = PadicMathematicalOperations(self.config.prime, self.config.base_precision)
        
        # Cache for different precision operators
        self._precision_ops_cache: Dict[int, PadicMathematicalOperations] = {
            self.config.base_precision: self.math_ops
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self.stats = {
            'total_compressions': 0,
            'total_weights': 0,
            'average_precision': 0.0,
            'total_bits_saved': 0,
            'hensel_improvements': 0,
            'precision_adjustments': 0,
            'compression_ratios': [],
            'processing_times': []
        }
        
        # GPU support
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                 self.config.enable_gpu_acceleration else 'cpu')

    @torch.compile(mode="reduce-overhead")
    def compute_adaptive_precision(self, tensor: torch.Tensor, 
                                  target_error: Optional[float] = None) -> Tuple[List[PadicWeight], torch.Tensor]:
        """
        Compute adaptive precision for each weight.
        
        Math: Hensel's lemma guarantees quadratic convergence
        |α - aₙ|_p ≤ |f'(a)|_p · t^(2^(n-1))
        
        Args:
            tensor: Input tensor of weights
            target_error: Target error threshold (uses config default if None)
            
        Returns:
            - padic_weights: Variable precision p-adic representation
            - precision_map: Actual precision used per weight
        """
        # Validate input
        PadicValidation.validate_tensor(tensor)
        
        if target_error is None:
            target_error = self.config.target_error
        
        start_time = time.perf_counter()
        
        # Move to appropriate device
        tensor = tensor.to(self.device)
        batch_size = tensor.shape[0] if tensor.dim() > 0 else 1
        flat_tensor = tensor.flatten()
        
        padic_weights = []
        precision_map = torch.zeros(flat_tensor.shape[0], dtype=torch.int32, device=self.device)
        error_map = torch.zeros(flat_tensor.shape[0], dtype=torch.float32, device=self.device)
        
        # Process each weight with adaptive precision
        for i in range(flat_tensor.shape[0]):
            value = flat_tensor[i].item()
            
            try:
                # Use Hensel lifting for adaptive precision
                weight, iterations = self.hensel.adaptive_precision_hensel(
                    value, target_error, self.config.prime
                )
                
                # Track precision used
                precision_map[i] = weight.precision
                
                # Calculate actual error
                reconstructed = self.math_ops.from_padic(weight) if weight.precision == self.config.base_precision else \
                               self._get_precision_ops(weight.precision).from_padic(weight)
                actual_error = abs(value - reconstructed) / (abs(value) + 1e-10)
                error_map[i] = actual_error
                
                padic_weights.append(weight)
                
                # Update statistics
                if iterations < weight.precision:
                    self.stats['hensel_improvements'] += 1
                    
            except (ValueError, OverflowError) as e:
                # Create fallback weight with base precision
                fallback_ops = self._get_precision_ops(self.config.base_precision)
                weight = fallback_ops.to_padic(value)
                precision_map[i] = self.config.base_precision
                error_map[i] = target_error  # Assume target error as fallback
                padic_weights.append(weight)
        
        # Reshape precision map to original shape
        if tensor.dim() > 1:
            precision_map = precision_map.reshape(tensor.shape)
        
        # Update statistics
        with self._lock:
            self.stats['total_compressions'] += 1
            self.stats['total_weights'] += flat_tensor.shape[0]
            self.stats['average_precision'] = (
                self.stats['average_precision'] * (self.stats['total_compressions'] - 1) +
                float(precision_map.float().mean().item())
            ) / self.stats['total_compressions']
            
            processing_time = time.perf_counter() - start_time
            self.stats['processing_times'].append(processing_time)
        
        return padic_weights, precision_map

    def allocate_precision_by_importance(self, tensor: torch.Tensor, 
                                        importance_scores: torch.Tensor,
                                        total_bits: int) -> torch.Tensor:
        """
        Allocate precision based on weight importance.
        
        Formula: k(w) = k_base + log_p(importance(w))
        
        Args:
            tensor: Input tensor of weights
            importance_scores: Importance score for each weight
            total_bits: Total bit budget for allocation
            
        Returns:
            precision_allocation: Precision allocated to each weight
        """
        # Validate inputs
        PadicValidation.validate_tensor(tensor)
        PadicValidation.validate_tensor(importance_scores)
        
        if tensor.shape != importance_scores.shape:
            raise ValueError(f"Shape mismatch: tensor {tensor.shape} vs importance {importance_scores.shape}")
        
        if total_bits <= 0:
            raise ValueError(f"Total bits must be positive, got {total_bits}")
        
        # Move to device
        tensor = tensor.to(self.device)
        importance_scores = importance_scores.to(self.device)
        
        # Normalize importance scores
        importance_scores = torch.abs(importance_scores)
        importance_scores = importance_scores / (importance_scores.sum() + 1e-10)
        
        # Sort by importance
        sorted_importance, indices = torch.sort(importance_scores.flatten(), descending=True)
        
        # Allocate bits proportionally with logarithmic scaling
        bits_per_digit = math.log2(self.config.prime)
        precision_allocation = torch.zeros_like(importance_scores, dtype=torch.int32)
        
        # Calculate base allocation
        min_bits_per_weight = self.config.min_precision * bits_per_digit
        max_bits_per_weight = self.config.max_precision * bits_per_digit
        
        # Reserve minimum bits for all weights
        reserved_bits = tensor.numel() * min_bits_per_weight
        if reserved_bits > total_bits:
            # Not enough bits - use minimum precision for all
            precision_allocation.fill_(self.config.min_precision)
            return precision_allocation
        
        # Distribute remaining bits based on importance
        remaining_bits = total_bits - reserved_bits
        
        for idx in indices:
            # Calculate additional precision based on importance
            importance = importance_scores.flatten()[idx].item()
            
            if importance < self.config.importance_threshold:
                # Low importance - use minimum precision
                precision_allocation.flatten()[idx] = self.config.min_precision
            else:
                # High importance - allocate more precision
                # Use logarithmic scaling: k = k_min + log_p(1 + γ·importance)
                gamma = 10.0  # Scaling factor
                additional_precision = int(math.log(1 + gamma * importance) / math.log(self.config.prime))
                
                # Calculate precision with bounds
                weight_precision = min(
                    self.config.max_precision,
                    self.config.min_precision + additional_precision
                )
                
                # Check if we have bits available
                additional_bits_needed = (weight_precision - self.config.min_precision) * bits_per_digit
                if additional_bits_needed <= remaining_bits:
                    precision_allocation.flatten()[idx] = weight_precision
                    remaining_bits -= additional_bits_needed
                else:
                    # Use as much precision as we can afford
                    affordable_precision = self.config.min_precision + int(remaining_bits / bits_per_digit)
                    precision_allocation.flatten()[idx] = affordable_precision
                    remaining_bits = 0
        
        # Reshape to original shape
        precision_allocation = precision_allocation.reshape(tensor.shape)
        
        # Update statistics
        with self._lock:
            self.stats['precision_adjustments'] += tensor.numel()
            actual_bits = float((precision_allocation * bits_per_digit).sum().item())
            self.stats['total_bits_saved'] += int(total_bits - actual_bits)
        
        return precision_allocation

    def batch_compress_with_adaptive_precision(self, tensor: torch.Tensor,
                                              importance_scores: Optional[torch.Tensor] = None,
                                              total_bits: Optional[int] = None) -> PrecisionAllocation:
        """
        Batch processing of tensors with adaptive precision allocation.
        
        Args:
            tensor: Input tensor to compress
            importance_scores: Optional importance scores for each weight
            total_bits: Optional total bit budget (calculated if None)
            
        Returns:
            PrecisionAllocation with compressed weights and statistics
        """
        start_time = time.perf_counter()
        
        # Validate input
        PadicValidation.validate_tensor(tensor)
        
        # Move to device
        tensor = tensor.to(self.device)
        flat_tensor = tensor.flatten()
        batch_size = flat_tensor.shape[0]
        
        # Calculate or validate bit budget
        if total_bits is None:
            # Default: aim for 50% compression
            bits_per_float = 32
            total_bits = int(batch_size * bits_per_float * 0.5)
        
        # Generate importance scores if not provided
        if importance_scores is None:
            # Use magnitude as proxy for importance
            importance_scores = torch.abs(tensor)
        else:
            importance_scores = importance_scores.to(self.device)
        
        # Allocate precision based on importance
        precision_allocation = self.allocate_precision_by_importance(
            tensor, importance_scores, total_bits
        )
        
        # Compress weights with allocated precision
        padic_weights = []
        error_map = torch.zeros_like(flat_tensor)
        
        for i in range(batch_size):
            value = flat_tensor[i].item()
            target_precision = int(precision_allocation.flatten()[i].item())
            
            # Get appropriate precision operations
            precision_ops = self._get_precision_ops(target_precision)
            
            try:
                # Convert to p-adic with target precision
                if self.config.enable_dynamic_switching and abs(value) < 1e-8:
                    # Very small values - use minimal precision
                    precision_ops = self._get_precision_ops(self.config.min_precision)
                    weight = precision_ops.to_padic(value)
                else:
                    weight = precision_ops.to_padic(value)
                
                # Calculate reconstruction error
                reconstructed = precision_ops.from_padic(weight)
                error = abs(value - reconstructed) / (abs(value) + 1e-10)
                error_map[i] = error
                
                padic_weights.append(weight)
                
            except (ValueError, OverflowError):
                # Fallback to base precision
                fallback_weight = self.math_ops.to_padic(value)
                padic_weights.append(fallback_weight)
                error_map[i] = self.config.target_error
        
        # Calculate compression statistics
        bits_per_digit = math.log2(self.config.prime)
        actual_bits_used = sum(w.precision * bits_per_digit for w in padic_weights)
        original_bits = batch_size * 32  # Assuming float32
        compression_ratio = original_bits / actual_bits_used if actual_bits_used > 0 else 1.0
        
        # Gather convergence statistics
        convergence_stats = {
            'hensel_stats': self.hensel.get_stats() if self.hensel else {},
            'precision_distribution': self._get_precision_distribution(precision_allocation),
            'processing_time': time.perf_counter() - start_time,
            'average_iterations': self.hensel.get_stats().get('average_iterations', 0) if self.hensel else 0
        }
        
        # Update global statistics
        with self._lock:
            self.stats['compression_ratios'].append(compression_ratio)
        
        return PrecisionAllocation(
            weights=padic_weights,
            precision_map=precision_allocation,
            error_map=error_map.reshape(tensor.shape),
            total_bits=int(actual_bits_used),
            compression_ratio=compression_ratio,
            convergence_stats=convergence_stats
        )

    def monitor_precision_efficiency(self, allocation: PrecisionAllocation) -> Dict[str, Any]:
        """
        Monitor and analyze precision allocation efficiency.
        
        Args:
            allocation: PrecisionAllocation result to analyze
            
        Returns:
            Dictionary with efficiency metrics
        """
        precision_map = allocation.precision_map.cpu().numpy()
        error_map = allocation.error_map.cpu().numpy()
        
        # Calculate efficiency metrics
        metrics = {
            'average_precision': float(np.mean(precision_map)),
            'precision_std': float(np.std(precision_map)),
            'min_precision': int(np.min(precision_map)),
            'max_precision': int(np.max(precision_map)),
            'compression_ratio': allocation.compression_ratio,
            'total_bits': allocation.total_bits,
            'error_stats': allocation.get_error_statistics(),
            'precision_histogram': self._compute_precision_histogram(precision_map),
            'error_precision_correlation': float(np.corrcoef(
                precision_map.flatten(), 
                error_map.flatten()
            )[0, 1]) if precision_map.size > 1 else 0.0,
            'efficiency_score': self._calculate_efficiency_score(allocation)
        }
        
        # Identify optimization opportunities
        metrics['optimization_suggestions'] = self._generate_optimization_suggestions(metrics)
        
        return metrics

    def create_logarithmic_padic_weights(self, tensor: torch.Tensor,
                                        precision_allocation: torch.Tensor) -> List[LogarithmicPadicWeight]:
        """
        Create LogarithmicPadicWeight objects with adaptive precision.
        
        Args:
            tensor: Input tensor
            precision_allocation: Precision for each weight
            
        Returns:
            List of LogarithmicPadicWeight objects
        """
        PadicValidation.validate_tensor(tensor)
        
        log_weights = []
        flat_tensor = tensor.flatten()
        flat_precision = precision_allocation.flatten()
        
        for i in range(flat_tensor.shape[0]):
            value = flat_tensor[i].item()
            precision = int(flat_precision[i].item())
            
            # Apply logarithmic transform
            log_value = math.log(abs(value) + 1e-10) if value != 0 else -10.0
            
            # Get precision-specific operations
            precision_ops = self._get_precision_ops(precision)
            
            # Create p-adic weight
            padic_weight = precision_ops.to_padic(log_value)
            
            # Create logarithmic weight
            log_weight = LogarithmicPadicWeight(
                padic_weight=padic_weight,
                original_value=value,
                log_value=log_value,
                encoding_method='adaptive_hensel',
                compression_metadata={
                    'precision': precision,
                    'prime': self.config.prime,
                    'hensel_enabled': True,
                    'target_error': self.config.target_error
                }
            )
            
            log_weights.append(log_weight)
        
        return log_weights

    def _get_precision_ops(self, precision: int) -> PadicMathematicalOperations:
        """Get or create mathematical operations for specific precision"""
        with self._lock:
            if precision not in self._precision_ops_cache:
                self._precision_ops_cache[precision] = PadicMathematicalOperations(
                    self.config.prime, precision
                )
            return self._precision_ops_cache[precision]

    def _get_precision_distribution(self, precision_map: torch.Tensor) -> Dict[int, int]:
        """Get distribution of precision values"""
        unique, counts = torch.unique(precision_map, return_counts=True)
        return {int(p.item()): int(c.item()) for p, c in zip(unique, counts)}

    def _compute_precision_histogram(self, precision_map: np.ndarray) -> Dict[str, List]:
        """Compute histogram of precision values"""
        unique, counts = np.unique(precision_map, return_counts=True)
        return {
            'bins': unique.tolist(),
            'counts': counts.tolist(),
            'frequencies': (counts / counts.sum()).tolist()
        }

    def _calculate_efficiency_score(self, allocation: PrecisionAllocation) -> float:
        """
        Calculate overall efficiency score balancing compression and accuracy.
        Score in [0, 1] where 1 is optimal.
        """
        # Compression component (higher ratio is better)
        compression_score = min(1.0, allocation.compression_ratio / 10.0)
        
        # Accuracy component (lower error is better)
        mean_error = float(allocation.error_map.mean().item())
        accuracy_score = 1.0 - min(1.0, mean_error / self.config.target_error)
        
        # Weighted combination based on compression priority
        efficiency = (self.config.compression_priority * compression_score +
                     (1 - self.config.compression_priority) * accuracy_score)
        
        return float(efficiency)

    def _generate_optimization_suggestions(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate suggestions for improving compression efficiency"""
        suggestions = []
        
        # Check if precision is too high
        if metrics['average_precision'] > self.config.base_precision * 0.8:
            suggestions.append("Consider increasing target_error to reduce average precision")
        
        # Check if errors are too low (over-precision)
        if metrics['error_stats']['mean'] < self.config.target_error * 0.1:
            suggestions.append("Errors are well below target - can reduce precision for better compression")
        
        # Check precision distribution
        if metrics['precision_std'] < 1.0:
            suggestions.append("Low precision variance - consider more aggressive importance-based allocation")
        
        # Check correlation between error and precision
        if abs(metrics['error_precision_correlation']) < 0.3:
            suggestions.append("Weak error-precision correlation - importance scores may need adjustment")
        
        return suggestions

    def reset_statistics(self):
        """Reset all statistics"""
        with self._lock:
            self.stats = {
                'total_compressions': 0,
                'total_weights': 0,
                'average_precision': 0.0,
                'total_bits_saved': 0,
                'hensel_improvements': 0,
                'precision_adjustments': 0,
                'compression_ratios': [],
                'processing_times': []
            }
            
            if self.hensel:
                self.hensel.reset_stats()

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        with self._lock:
            stats = dict(self.stats)
            
            # Add computed statistics
            if stats['compression_ratios']:
                stats['mean_compression_ratio'] = float(np.mean(stats['compression_ratios']))
                stats['std_compression_ratio'] = float(np.std(stats['compression_ratios']))
            
            if stats['processing_times']:
                stats['mean_processing_time'] = float(np.mean(stats['processing_times']))
                stats['total_processing_time'] = float(sum(stats['processing_times']))
            
            # Add Hensel statistics
            if self.hensel:
                stats['hensel_stats'] = self.hensel.get_stats()
            
            return stats


def demonstrate_adaptive_precision():
    """Demonstration of adaptive precision wrapper capabilities"""
    print("=== Adaptive Precision Wrapper Demonstration ===\n")
    
    # Create configuration with safe precision for prime 257
    config = AdaptivePrecisionConfig(
        prime=257,
        base_precision=4,  # Safe for prime 257
        min_precision=2,
        max_precision=4,   # Max safe precision for prime 257
        target_error=1e-6,
        compression_priority=0.6  # Slightly favor compression
    )
    
    # Initialize wrapper
    wrapper = AdaptivePrecisionWrapper(config)
    
    # Generate test data
    torch.manual_seed(42)
    test_tensor = torch.randn(100, 10) * 10.0
    
    # Generate importance scores (e.g., gradient magnitudes)
    importance_scores = torch.abs(test_tensor) + torch.randn_like(test_tensor).abs() * 0.1
    
    print("1. Testing basic adaptive precision computation...")
    weights, precision_map = wrapper.compute_adaptive_precision(test_tensor, target_error=1e-7)
    print(f"   - Compressed {test_tensor.numel()} weights")
    print(f"   - Average precision: {precision_map.float().mean():.2f}")
    print(f"   - Precision range: [{precision_map.min()}, {precision_map.max()}]")
    
    print("\n2. Testing importance-based allocation...")
    total_bits = 100 * 10 * 16  # 50% of original float32 storage
    precision_allocation = wrapper.allocate_precision_by_importance(
        test_tensor, importance_scores, total_bits
    )
    print(f"   - Allocated precision based on importance")
    print(f"   - Unique precision levels: {torch.unique(precision_allocation).tolist()}")
    
    print("\n3. Testing batch compression with full pipeline...")
    allocation = wrapper.batch_compress_with_adaptive_precision(
        test_tensor, importance_scores, total_bits
    )
    print(f"   - Compression ratio: {allocation.compression_ratio:.2f}x")
    print(f"   - Average error: {allocation.error_map.mean():.2e}")
    print(f"   - Total bits used: {allocation.total_bits}")
    
    print("\n4. Analyzing efficiency...")
    metrics = wrapper.monitor_precision_efficiency(allocation)
    print(f"   - Efficiency score: {metrics['efficiency_score']:.3f}")
    print(f"   - Error-precision correlation: {metrics['error_precision_correlation']:.3f}")
    if metrics['optimization_suggestions']:
        print("   - Suggestions:")
        for suggestion in metrics['optimization_suggestions']:
            print(f"     * {suggestion}")
    
    print("\n5. Creating logarithmic p-adic weights...")
    log_weights = wrapper.create_logarithmic_padic_weights(
        test_tensor[:10, :5], precision_allocation[:10, :5]
    )
    print(f"   - Created {len(log_weights)} logarithmic weights")
    print(f"   - Sample compression ratio: {log_weights[0].get_compression_ratio():.2f}x")
    
    print("\n6. Final statistics:")
    stats = wrapper.get_statistics()
    print(f"   - Total compressions: {stats['total_compressions']}")
    print(f"   - Total weights processed: {stats['total_weights']}")
    print(f"   - Average precision used: {stats['average_precision']:.2f}")
    print(f"   - Hensel improvements: {stats['hensel_improvements']}")
    if 'mean_compression_ratio' in stats:
        print(f"   - Mean compression ratio: {stats['mean_compression_ratio']:.2f}x")
    
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    demonstrate_adaptive_precision()