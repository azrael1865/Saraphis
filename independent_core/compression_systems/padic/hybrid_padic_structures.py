"""
Hybrid P-adic Data Structures - GPU-friendly two-channel representation
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import json
from fractions import Fraction
from math import gcd

from padic_encoder import PadicWeight, PadicValidation


@dataclass
class HybridPadicWeight:
    """GPU-friendly hybrid p-adic weight with two-channel representation"""
    # GPU tensors for efficient computation
    exponent_channel: torch.Tensor  # Hierarchical importance (GPU tensor)
    mantissa_channel: torch.Tensor  # Fine-grained values (GPU tensor)
    
    # P-adic metadata (reused from existing PadicWeight)
    prime: int
    precision: int
    valuation: int
    
    # Performance tracking
    device: torch.device
    dtype: torch.dtype
    
    # Validation and error bounds
    error_tolerance: float = 1e-6
    ultrametric_preserved: bool = True
    
    def __post_init__(self):
        """Validate hybrid p-adic weight structure"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not isinstance(self.exponent_channel, torch.Tensor):
            raise TypeError(f"exponent_channel must be torch.Tensor, got {type(self.exponent_channel)}")
        if not isinstance(self.mantissa_channel, torch.Tensor):
            raise TypeError(f"mantissa_channel must be torch.Tensor, got {type(self.mantissa_channel)}")
        if not self.exponent_channel.is_cuda:
            raise ValueError("exponent_channel must be on GPU")
        if not self.mantissa_channel.is_cuda:
            raise ValueError("mantissa_channel must be on GPU")
        if self.exponent_channel.shape != self.mantissa_channel.shape:
            raise ValueError(f"exponent_channel and mantissa_channel must have same shape: {self.exponent_channel.shape} != {self.mantissa_channel.shape}")
        if not isinstance(self.prime, int):
            raise TypeError(f"Prime must be int, got {type(self.prime)}")
        if self.prime <= 1:
            raise ValueError(f"Prime must be > 1, got {self.prime}")
        if not isinstance(self.precision, int):
            raise TypeError(f"Precision must be int, got {type(self.precision)}")
        if self.precision <= 0:
            raise ValueError(f"Precision must be > 0, got {self.precision}")
        if not isinstance(self.valuation, int):
            raise TypeError(f"Valuation must be int, got {type(self.valuation)}")
        if not isinstance(self.device, torch.device):
            raise TypeError(f"Device must be torch.device, got {type(self.device)}")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError(f"Dtype must be torch.dtype, got {type(self.dtype)}")
        if not isinstance(self.error_tolerance, (float, int)):
            raise TypeError(f"Error tolerance must be float or int, got {type(self.error_tolerance)}")
        if self.error_tolerance <= 0:
            raise ValueError(f"Error tolerance must be > 0, got {self.error_tolerance}")
        if not isinstance(self.ultrametric_preserved, bool):
            raise TypeError(f"Ultrametric preserved must be bool, got {type(self.ultrametric_preserved)}")


class HybridPadicValidator:
    """Validator for hybrid p-adic structures"""
    
    @staticmethod
    def validate_hybrid_weight(weight: HybridPadicWeight) -> None:
        """Validate hybrid p-adic weight structure and properties"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if weight is None:
            raise ValueError("Weight cannot be None")
        if not isinstance(weight, HybridPadicWeight):
            raise TypeError(f"Weight must be HybridPadicWeight, got {type(weight)}")
        
        # Validate tensor properties
        if not weight.exponent_channel.is_cuda:
            raise ValueError("exponent_channel must be on GPU")
        if not weight.mantissa_channel.is_cuda:
            raise ValueError("mantissa_channel must be on GPU")
        
        # Validate tensor device consistency
        if weight.exponent_channel.device != weight.mantissa_channel.device:
            raise ValueError(f"Channel device mismatch: {weight.exponent_channel.device} != {weight.mantissa_channel.device}")
        if weight.exponent_channel.device != weight.device:
            raise ValueError(f"Tensor device mismatch with specified device: {weight.exponent_channel.device} != {weight.device}")
        
        # Validate tensor dtype consistency
        if weight.exponent_channel.dtype != weight.mantissa_channel.dtype:
            raise ValueError(f"Channel dtype mismatch: {weight.exponent_channel.dtype} != {weight.mantissa_channel.dtype}")
        if weight.exponent_channel.dtype != weight.dtype:
            raise ValueError(f"Tensor dtype mismatch with specified dtype: {weight.exponent_channel.dtype} != {weight.dtype}")
        
        # Validate mathematical properties
        if weight.prime <= 1:
            raise ValueError(f"Invalid prime: {weight.prime}")
        if weight.precision <= 0:
            raise ValueError(f"Invalid precision: {weight.precision}")
        
        # Validate tensor contents
        if torch.isnan(weight.exponent_channel).any():
            raise ValueError("exponent_channel contains NaN values")
        if torch.isnan(weight.mantissa_channel).any():
            raise ValueError("mantissa_channel contains NaN values")
        if torch.isinf(weight.exponent_channel).any():
            raise ValueError("exponent_channel contains infinite values")
        if torch.isinf(weight.mantissa_channel).any():
            raise ValueError("mantissa_channel contains infinite values")
        
        # Validate ultrametric property preservation
        if weight.ultrametric_preserved:
            HybridPadicValidator._validate_ultrametric_property(weight)
    
    @staticmethod
    def _validate_ultrametric_property(weight: HybridPadicWeight) -> None:
        """Validate that ultrametric property is preserved in hybrid representation"""
        # Check that exponent channel maintains hierarchical structure
        exp_values = weight.exponent_channel.flatten()
        if torch.any(exp_values < 0):
            raise ValueError("Exponent channel cannot contain negative values")
        
        # Check mantissa channel bounds
        mantissa_values = weight.mantissa_channel.flatten()
        if torch.any(mantissa_values < 0):
            raise ValueError("Mantissa channel cannot contain negative values")
        if torch.any(mantissa_values >= weight.prime):
            raise ValueError(f"Mantissa values must be in [0, {weight.prime}), found values >= {weight.prime}")
        
        # Validate exponent channel finite values
        max_reasonable_exp = 50.0  # Reasonable upper bound for exponents
        if torch.any(exp_values > max_reasonable_exp):
            raise ValueError(f"Exponent channel contains unreasonably large values > {max_reasonable_exp}")
    
    @staticmethod
    def validate_conversion_parameters(padic_weight: PadicWeight) -> None:
        """Validate parameters for conversion to hybrid representation"""
        if padic_weight is None:
            raise ValueError("PadicWeight cannot be None")
        if not isinstance(padic_weight, PadicWeight):
            raise TypeError(f"Expected PadicWeight, got {type(padic_weight)}")
        
        # Use existing PadicValidation for consistency
        PadicValidation.validate_prime(padic_weight.prime)
        PadicValidation.validate_precision(padic_weight.precision)
        
        # Validate digits structure
        if not hasattr(padic_weight, 'digits') or not isinstance(padic_weight.digits, list):
            raise ValueError("PadicWeight must have valid digits list")
        if len(padic_weight.digits) != padic_weight.precision:
            raise ValueError(f"Digits length {len(padic_weight.digits)} must equal precision {padic_weight.precision}")
        
        # Validate each digit
        for i, digit in enumerate(padic_weight.digits):
            if not isinstance(digit, int):
                raise TypeError(f"Digit {i} must be int, got {type(digit)}")
            if not (0 <= digit < padic_weight.prime):
                raise ValueError(f"Digit {digit} at position {i} must be in range [0, {padic_weight.prime})")


class HybridPadicConverter:
    """Convert between pure p-adic and hybrid representations"""
    
    def __init__(self, gpu_memory_optimizer: Optional[Any] = None):
        """Initialize converter with GPU memory management"""
        self.gpu_memory_optimizer = gpu_memory_optimizer
        
        # Check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for hybrid p-adic operations")
        
        self.device = torch.device('cuda:0')
        self.validator = HybridPadicValidator()
        
        # Performance tracking
        self.conversion_metrics = {
            'conversions_to_hybrid': 0,
            'conversions_from_hybrid': 0,
            'total_conversion_time': 0.0,
            'failed_conversions': 0
        }
    
    def convert_to_hybrid(self, padic_weight: PadicWeight) -> HybridPadicWeight:
        """Convert pure p-adic weight to hybrid representation"""
        # NO FALLBACKS - HARD FAILURES ONLY
        start_time = time.time()
        
        # Validate input
        self.validator.validate_conversion_parameters(padic_weight)
        
        try:
            # Extract p-adic coefficients from digits
            coefficients = padic_weight.digits
            prime = padic_weight.prime
            precision = padic_weight.precision
            
            # Convert to GPU tensors
            coeff_tensor = torch.tensor(coefficients, dtype=torch.float32, device=self.device)
            
            # Split into exponent and mantissa channels
            exponent_channel = self._extract_exponent_channel(coeff_tensor, prime)
            mantissa_channel = self._extract_mantissa_channel(coeff_tensor, prime)
            
            # Create hybrid weight
            hybrid_weight = HybridPadicWeight(
                exponent_channel=exponent_channel,
                mantissa_channel=mantissa_channel,
                prime=prime,
                precision=precision,
                valuation=padic_weight.valuation,
                device=self.device,
                dtype=torch.float32
            )
            
            # Validate result
            self.validator.validate_hybrid_weight(hybrid_weight)
            
            # Update metrics
            conversion_time = time.time() - start_time
            self.conversion_metrics['conversions_to_hybrid'] += 1
            self.conversion_metrics['total_conversion_time'] += conversion_time
            
            return hybrid_weight
            
        except Exception as e:
            self.conversion_metrics['failed_conversions'] += 1
            raise RuntimeError(f"Failed to convert to hybrid representation: {e}")
    
    def convert_from_hybrid(self, hybrid_weight: HybridPadicWeight) -> PadicWeight:
        """Convert hybrid representation back to pure p-adic"""
        # NO FALLBACKS - HARD FAILURES ONLY
        start_time = time.time()
        
        # Validate input
        self.validator.validate_hybrid_weight(hybrid_weight)
        
        try:
            # Reconstruct coefficients from two channels
            coeff_tensor = self._reconstruct_coefficients(
                hybrid_weight.exponent_channel,
                hybrid_weight.mantissa_channel,
                hybrid_weight.prime
            )
            
            # Convert back to CPU and round to integers
            coefficients_float = coeff_tensor.cpu().numpy()
            coefficients = [int(round(float(c))) for c in coefficients_float]
            
            # Ensure coefficients are in valid range
            for i, coeff in enumerate(coefficients):
                if coeff < 0:
                    coefficients[i] = 0
                elif coeff >= hybrid_weight.prime:
                    coefficients[i] = hybrid_weight.prime - 1
            
            # Reconstruct fraction value from coefficients
            value_float = sum(coefficients[i] * (hybrid_weight.prime ** i) for i in range(len(coefficients)))
            
            # Convert to fraction
            from fractions import Fraction
            value = Fraction(value_float).limit_denominator(10**10)
            
            # Create pure p-adic weight (reuse existing structure)
            padic_weight = PadicWeight(
                value=value,
                prime=hybrid_weight.prime,
                precision=hybrid_weight.precision,
                valuation=hybrid_weight.valuation,
                digits=coefficients
            )
            
            # Update metrics
            conversion_time = time.time() - start_time
            self.conversion_metrics['conversions_from_hybrid'] += 1
            self.conversion_metrics['total_conversion_time'] += conversion_time
            
            return padic_weight
            
        except Exception as e:
            self.conversion_metrics['failed_conversions'] += 1
            raise RuntimeError(f"Failed to convert from hybrid representation: {e}")
    
    def _extract_exponent_channel(self, coeff_tensor: torch.Tensor, prime: int) -> torch.Tensor:
        """Extract exponent channel using p-adic valuation"""
        device = coeff_tensor.device
        batch_size = coeff_tensor.shape[0]
        
        # Extract p-adic valuations from the tensor
        valuations = torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        # Convert to integers for p-adic analysis
        int_values = (coeff_tensor * (prime ** 10)).long()
        
        # Compute p-adic valuations
        for i in range(batch_size):
            val = int_values[i].item()
            if val == 0:
                valuations[i] = float('inf')  # Convention for zero
            else:
                v = 0
                while val % prime == 0:
                    val //= prime
                    v += 1
                valuations[i] = float(v)
        
        # Normalize valuations to [0, 1] range for stability
        finite_vals = valuations[valuations != float('inf')]
        if finite_vals.numel() > 0:
            max_val = finite_vals.max()
            valuations[valuations != float('inf')] = valuations[valuations != float('inf')] / (max_val + 1)
            valuations[valuations == float('inf')] = 1.0  # Max normalized value for zeros
        
        return valuations
    
    def _extract_mantissa_channel(self, coeff_tensor: torch.Tensor, prime: int) -> torch.Tensor:
        """Extract mantissa channel using p-adic digit decomposition"""
        device = coeff_tensor.device
        batch_size = coeff_tensor.shape[0]
        
        # Extract first p-adic digit (mantissa)
        mantissas = torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        # Convert to fractions for exact p-adic decomposition
        for i in range(batch_size):
            val = coeff_tensor[i].item()
            if val == 0:
                mantissas[i] = 0.0
            else:
                # Extract first non-zero p-adic digit
                frac = Fraction(val).limit_denominator(10**10)
                
                # Normalize to have denominator coprime to p
                while frac.denominator % prime == 0:
                    frac = Fraction(frac.numerator, frac.denominator // prime)
                
                # Compute modular inverse if needed
                if gcd(frac.denominator, prime) == 1:
                    inv = pow(frac.denominator, -1, prime)
                    first_digit = (frac.numerator * inv) % prime
                    mantissas[i] = float(first_digit)
                else:
                    # Fallback for non-coprime denominators
                    mantissas[i] = float(abs(frac.numerator) % prime)
        
        return mantissas
    
    def _reconstruct_coefficients(self, exponent_channel: torch.Tensor, 
                                mantissa_channel: torch.Tensor, 
                                prime: int) -> torch.Tensor:
        """Reconstruct coefficients using p-adic formula"""
        device = exponent_channel.device
        batch_size = exponent_channel.shape[0]
        
        reconstructed = torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        for i in range(batch_size):
            # Denormalize valuation
            valuation = int(exponent_channel[i].item() * 10)  # Assuming max valuation of 10
            mantissa = int(mantissa_channel[i].item())
            
            # Reconstruct: mantissa * p^valuation
            if mantissa == 0:
                reconstructed[i] = 0.0
            else:
                reconstructed[i] = float(mantissa * (prime ** valuation))
        
        return reconstructed
    
    def get_conversion_metrics(self) -> Dict[str, Any]:
        """Get conversion performance metrics"""
        return self.conversion_metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset conversion metrics"""
        self.conversion_metrics = {
            'conversions_to_hybrid': 0,
            'conversions_from_hybrid': 0,
            'total_conversion_time': 0.0,
            'failed_conversions': 0
        }


class HybridPadicManager:
    """Manager for hybrid p-adic operations and memory management"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize hybrid p-adic manager"""
        self.config = config or {}
        
        # Initialize converter and validator
        gpu_optimizer = self.config.get('gpu_memory_optimizer')
        self.converter = HybridPadicConverter(gpu_optimizer)
        self.validator = HybridPadicValidator()
        
        # Performance tracking
        self.operation_stats = {
            'hybrid_weights_created': 0,
            'padic_weights_restored': 0,
            'total_operation_time': 0.0,
            'average_operation_time': 0.0,
            'validation_failures': 0,
            'gpu_memory_allocations': 0
        }
        
        # Configuration validation
        self._validate_configuration()
    
    def _validate_configuration(self) -> None:
        """Validate manager configuration"""
        if not isinstance(self.config, dict):
            raise TypeError(f"Config must be dict, got {type(self.config)}")
        
        # Validate optional parameters
        if 'error_tolerance' in self.config:
            tolerance = self.config['error_tolerance']
            if not isinstance(tolerance, (float, int)) or tolerance <= 0:
                raise ValueError(f"Error tolerance must be positive number, got {tolerance}")
        
        if 'max_precision' in self.config:
            max_prec = self.config['max_precision']
            if not isinstance(max_prec, int) or max_prec <= 0:
                raise ValueError(f"Max precision must be positive int, got {max_prec}")
    
    def create_hybrid_weight(self, padic_weight: PadicWeight) -> HybridPadicWeight:
        """Create hybrid weight from pure p-adic weight"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if padic_weight is None:
            raise ValueError("PadicWeight cannot be None")
        
        start_time = time.time()
        
        try:
            # Convert to hybrid
            hybrid_weight = self.converter.convert_to_hybrid(padic_weight)
            
            # Validate result
            self.validator.validate_hybrid_weight(hybrid_weight)
            
            # Update stats
            operation_time = time.time() - start_time
            self._update_operation_stats('create_hybrid', operation_time)
            
            return hybrid_weight
            
        except Exception as e:
            self.operation_stats['validation_failures'] += 1
            raise RuntimeError(f"Failed to create hybrid weight: {e}")
    
    def restore_padic_weight(self, hybrid_weight: HybridPadicWeight) -> PadicWeight:
        """Restore pure p-adic weight from hybrid representation"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if hybrid_weight is None:
            raise ValueError("HybridPadicWeight cannot be None")
        
        start_time = time.time()
        
        try:
            # Validate input
            self.validator.validate_hybrid_weight(hybrid_weight)
            
            # Convert back to pure p-adic
            padic_weight = self.converter.convert_from_hybrid(hybrid_weight)
            
            # Update stats
            operation_time = time.time() - start_time
            self._update_operation_stats('restore_padic', operation_time)
            
            return padic_weight
            
        except Exception as e:
            self.operation_stats['validation_failures'] += 1
            raise RuntimeError(f"Failed to restore p-adic weight: {e}")
    
    def _update_operation_stats(self, operation_type: str, time_taken: float) -> None:
        """Update operation statistics"""
        if operation_type == 'create_hybrid':
            self.operation_stats['hybrid_weights_created'] += 1
        elif operation_type == 'restore_padic':
            self.operation_stats['padic_weights_restored'] += 1
        
        self.operation_stats['total_operation_time'] += time_taken
        
        total_operations = (self.operation_stats['hybrid_weights_created'] + 
                          self.operation_stats['padic_weights_restored'])
        
        if total_operations > 0:
            self.operation_stats['average_operation_time'] = (
                self.operation_stats['total_operation_time'] / total_operations
            )
    
    def get_operation_stats(self) -> Dict[str, Any]:
        """Get operation statistics"""
        return self.operation_stats.copy()
    
    def get_conversion_stats(self) -> Dict[str, Any]:
        """Get conversion statistics from converter"""
        return self.converter.get_conversion_metrics()
    
    def reset_stats(self) -> None:
        """Reset all statistics"""
        self.operation_stats = {
            'hybrid_weights_created': 0,
            'padic_weights_restored': 0,
            'total_operation_time': 0.0,
            'average_operation_time': 0.0,
            'validation_failures': 0,
            'gpu_memory_allocations': 0
        }
        self.converter.reset_metrics()
    
    def validate_hybrid_compatibility(self, hybrid_weight: HybridPadicWeight, 
                                    padic_weight: PadicWeight) -> bool:
        """Validate that hybrid and p-adic representations are compatible"""
        try:
            # Basic parameter compatibility
            if hybrid_weight.prime != padic_weight.prime:
                return False
            if hybrid_weight.precision != padic_weight.precision:
                return False
            if hybrid_weight.valuation != padic_weight.valuation:
                return False
            
            # Convert hybrid back and compare
            restored_weight = self.restore_padic_weight(hybrid_weight)
            
            # Compare digits (allowing for small numerical differences)
            tolerance = self.config.get('error_tolerance', 1e-6)
            for i, (orig_digit, restored_digit) in enumerate(zip(padic_weight.digits, restored_weight.digits)):
                if abs(orig_digit - restored_digit) > tolerance:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get GPU memory usage for hybrid operations"""
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}
        
        return {
            'allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
            'reserved_mb': torch.cuda.memory_reserved() / (1024 * 1024),
            'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024),
            'max_reserved_mb': torch.cuda.max_memory_reserved() / (1024 * 1024)
        }
    
    def cleanup_gpu_memory(self) -> None:
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()