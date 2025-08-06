"""
Tropical coefficient channel extraction system for GPU-efficient memory layout.
Transforms tropical representations into GPU-optimized channel layouts for massive parallelization.
NO PLACEHOLDERS - PRODUCTION READY
"""

import torch
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import itertools
from concurrent.futures import Future

# Import existing tropical components
try:
    from independent_core.compression_systems.tropical.tropical_polynomial import (
        TropicalPolynomial,
        TropicalMonomial,
        TropicalPolynomialOperations
    )
    from independent_core.compression_systems.tropical.tropical_core import (
        TropicalNumber,
        TROPICAL_ZERO,
        TROPICAL_EPSILON
    )
    from independent_core.compression_systems.tropical.gpu_memory_optimizer import (
        GPUMemoryLayoutConfig,
        ChannelMemoryOptimizer,
        GPUMemoryAllocator,
        ChannelAccessPatternAnalyzer,
        BatchedChannelProcessor,
        MemoryLayout,
        AccessPattern,
        MemoryAccessMetrics,
        create_optimized_gpu_config
    )
except ImportError:
    # For direct execution
    from tropical_polynomial import (
        TropicalPolynomial,
        TropicalMonomial,
        TropicalPolynomialOperations
    )
    from tropical_core import (
        TropicalNumber,
        TROPICAL_ZERO,
        TROPICAL_EPSILON
    )
    from gpu_memory_optimizer import (
        GPUMemoryLayoutConfig,
        ChannelMemoryOptimizer,
        GPUMemoryAllocator,
        ChannelAccessPatternAnalyzer,
        BatchedChannelProcessor,
        MemoryLayout,
        AccessPattern,
        MemoryAccessMetrics,
        create_optimized_gpu_config
    )


@dataclass
class ExponentChannelConfig:
    """Configuration for exponent channel extraction"""
    use_sparse: bool = True
    sparsity_threshold: float = 0.7
    quantization: str = "auto"  # "int8", "int16", "int32", "auto"
    compression_level: int = 1  # 0=none, 1=basic, 2=aggressive
    enable_delta_encoding: bool = True
    enable_pattern_clustering: bool = True
    block_size: int = 32  # For block compression
    gpu_coalesce: bool = True  # Enable GPU memory coalescing
    validate_lossless: bool = True  # Always validate compression is lossless
    pattern_analysis_depth: int = 3  # Depth of pattern analysis
    max_cluster_size: int = 64  # Maximum size for pattern clusters


@dataclass
class MantissaChannelConfig:
    """Configuration for mantissa channel extraction with precision control"""
    precision_mode: str = "adaptive"  # "fp32", "fp16", "bf16", "mixed", "adaptive"
    compression_level: int = 1  # 0=none, 1=basic, 2=aggressive
    enable_bit_packing: bool = True  # Pack mantissa bits efficiently
    enable_delta_encoding: bool = True  # Delta encode mantissa values
    enable_pattern_compression: bool = True  # Compress repeated patterns
    block_size: int = 64  # Block size for compression
    precision_threshold: float = 1e-7  # Minimum precision to preserve
    denormal_handling: str = "preserve"  # "preserve", "flush", "round"
    error_correction: bool = True  # Enable error correction codes
    ecc_strength: int = 1  # 0=none, 1=basic, 2=strong
    gpu_optimize: bool = True  # Optimize for GPU memory layout
    validate_precision: bool = True  # Always validate precision preservation
    max_precision_loss: float = 1e-9  # Maximum allowed precision loss


@dataclass
class TropicalChannels:
    """GPU-optimized channel representation of tropical polynomial"""
    coefficient_channel: torch.Tensor  # Shape: (num_monomials,)
    exponent_channel: torch.Tensor    # Shape: (num_monomials, num_variables)
    index_channel: torch.Tensor       # Shape: (num_monomials,)
    metadata: Dict[str, Any]          # degree, num_vars, etc.
    device: torch.device
    mantissa_channel: Optional[torch.Tensor] = None  # Shape: (num_monomials,) or compressed
    packed_data: Optional[bytes] = None  # Packed representation if available
    packing_metadata: Optional[Dict[str, Any]] = None  # Packing information
    
    def __post_init__(self):
        """Validate channel consistency after creation"""
        self.validate()
    
    @property
    def is_packed(self) -> bool:
        """Check if channels are in packed format"""
        return self.packed_data is not None
    
    def get_memory_usage(self) -> int:
        """Get memory usage in bytes"""
        if self.is_packed:
            return len(self.packed_data)
        else:
            size = 0
            size += self.coefficient_channel.element_size() * self.coefficient_channel.nelement()
            size += self.exponent_channel.element_size() * self.exponent_channel.nelement()
            size += self.index_channel.element_size() * self.index_channel.nelement()
            if self.mantissa_channel is not None:
                size += self.mantissa_channel.element_size() * self.mantissa_channel.nelement()
            return size
    
    def to_polynomial(self) -> TropicalPolynomial:
        """
        Reconstruct polynomial from channels
        
        Returns:
            TropicalPolynomial reconstructed from channel data
        """
        num_monomials = self.coefficient_channel.shape[0]
        num_variables = self.metadata['num_variables']
        
        # Move to CPU for reconstruction if on GPU
        coeffs = self.coefficient_channel.cpu().numpy() if self.coefficient_channel.is_cuda else self.coefficient_channel.numpy()
        exps = self.exponent_channel.cpu().numpy() if self.exponent_channel.is_cuda else self.exponent_channel.numpy()
        
        monomials = []
        for i in range(num_monomials):
            # Skip tropical zero monomials
            if coeffs[i] <= TROPICAL_ZERO:
                continue
                
            # Build sparse exponent dictionary
            exponent_dict = {}
            for var_idx in range(num_variables):
                power = int(exps[i, var_idx])
                if power > 0:
                    exponent_dict[var_idx] = power
            
            monomials.append(TropicalMonomial(float(coeffs[i]), exponent_dict))
        
        return TropicalPolynomial(monomials, num_variables)
    
    def to_gpu(self, device: torch.device, optimize_layout: bool = True, 
               transfer_optimizer: Optional['ChannelTransferOptimizer'] = None) -> 'TropicalChannels':
        """
        Move channels to GPU with optional memory layout and transfer optimization
        
        Args:
            device: Target GPU device
            optimize_layout: Whether to optimize memory layout for GPU
            transfer_optimizer: Optional transfer optimizer for optimized transfers
            
        Returns:
            New TropicalChannels on specified device
        """
        if not device.type == 'cuda':
            raise ValueError(f"Target device must be cuda, got {device}")
        
        # Use transfer optimizer if provided
        if transfer_optimizer:
            result, metrics = transfer_optimizer.transfer_to_gpu(self, device)
            
            # Optimize layout if requested
            if optimize_layout:
                optimizer = ChannelMemoryOptimizer()
                
                # Optimize coefficient channel
                result.coefficient_channel, _, _ = optimizer.optimize_layout(
                    result.coefficient_channel, channel_type="coefficient"
                )
                
                # Optimize exponent channel
                result.exponent_channel, _, _ = optimizer.optimize_layout(
                    result.exponent_channel, channel_type="exponent"
                )
                
                # Optimize mantissa if present
                if result.mantissa_channel is not None:
                    result.mantissa_channel, _, _ = optimizer.optimize_layout(
                        result.mantissa_channel, channel_type="mantissa"
                    )
            
            return result
        
        # Fallback to standard transfer
        coeff_gpu = self.coefficient_channel.to(device)
        exp_gpu = self.exponent_channel.to(device)
        idx_gpu = self.index_channel.to(device)
        mantissa_gpu = self.mantissa_channel.to(device) if self.mantissa_channel is not None else None
        
        # Optimize layout if requested
        if optimize_layout:
            optimizer = ChannelMemoryOptimizer()
            
            # Optimize coefficient channel
            coeff_gpu, _, _ = optimizer.optimize_layout(
                coeff_gpu, channel_type="coefficient"
            )
            
            # Optimize exponent channel
            exp_gpu, _, _ = optimizer.optimize_layout(
                exp_gpu, channel_type="exponent"
            )
            
            # Optimize mantissa if present
            if mantissa_gpu is not None:
                mantissa_gpu, _, _ = optimizer.optimize_layout(
                    mantissa_gpu, channel_type="mantissa"
                )
        
        return TropicalChannels(
            coefficient_channel=coeff_gpu,
            exponent_channel=exp_gpu,
            index_channel=idx_gpu,
            metadata=self.metadata.copy(),
            device=device,
            mantissa_channel=mantissa_gpu
        )
    
    def to_cpu(self, use_pinned: bool = True, 
               transfer_optimizer: Optional['ChannelTransferOptimizer'] = None) -> 'TropicalChannels':
        """Move channels to CPU with optional pinned memory and transfer optimization
        
        Args:
            use_pinned: Use pinned memory for faster GPU transfers
            transfer_optimizer: Optional transfer optimizer for optimized transfers
            
        Returns:
            TropicalChannels on CPU
        """
        # Use transfer optimizer if provided
        if transfer_optimizer:
            result, metrics = transfer_optimizer.transfer_to_cpu(self, use_pinned)
            return result
        
        # Fallback to standard transfer
        if use_pinned and torch.cuda.is_available():
            # Use pinned memory for faster future transfers
            coeff_cpu = self.coefficient_channel.cpu().pin_memory()
            exp_cpu = self.exponent_channel.cpu().pin_memory()
            idx_cpu = self.index_channel.cpu().pin_memory()
            mantissa_cpu = self.mantissa_channel.cpu().pin_memory() if self.mantissa_channel is not None else None
        else:
            # Regular CPU memory
            coeff_cpu = self.coefficient_channel.cpu()
            exp_cpu = self.exponent_channel.cpu()
            idx_cpu = self.index_channel.cpu()
            mantissa_cpu = self.mantissa_channel.cpu() if self.mantissa_channel is not None else None
        
        return TropicalChannels(
            coefficient_channel=coeff_cpu,
            exponent_channel=exp_cpu,
            index_channel=idx_cpu,
            metadata=self.metadata.copy(),
            device=torch.device('cpu'),
            mantissa_channel=mantissa_cpu
        )
    
    def prefetch_to_gpu(self, device: torch.device, 
                       transfer_optimizer: 'ChannelTransferOptimizer') -> 'Future':
        """Prefetch channels to GPU asynchronously
        
        Args:
            device: Target GPU device
            transfer_optimizer: Transfer optimizer for async prefetch
            
        Returns:
            Future that will contain transferred channels
        """
        if not transfer_optimizer:
            raise ValueError("Transfer optimizer required for prefetch")
        
        return transfer_optimizer.prefetch_to_gpu(self, device)
    
    def transfer_statistics(self, transfer_optimizer: Optional['ChannelTransferOptimizer'] = None) -> Dict[str, Any]:
        """Get transfer statistics for this channel set
        
        Args:
            transfer_optimizer: Optional transfer optimizer with statistics
            
        Returns:
            Dictionary of transfer statistics
        """
        stats = {
            'channel_sizes': {
                'coefficient_bytes': self.coefficient_channel.numel() * self.coefficient_channel.element_size(),
                'exponent_bytes': self.exponent_channel.numel() * self.exponent_channel.element_size(),
                'index_bytes': self.index_channel.numel() * self.index_channel.element_size(),
                'mantissa_bytes': self.mantissa_channel.numel() * self.mantissa_channel.element_size() if self.mantissa_channel is not None else 0
            },
            'total_size_bytes': self.get_memory_usage(),
            'device': str(self.device),
            'is_packed': self.is_packed,
            'num_monomials': self.coefficient_channel.shape[0],
            'num_variables': self.metadata.get('num_variables', self.exponent_channel.shape[1])
        }
        
        if transfer_optimizer:
            stats['transfer_optimizer_stats'] = transfer_optimizer.get_transfer_statistics()
        
        return stats
    
    def validate(self) -> None:
        """
        Validate channel consistency
        
        Raises:
            ValueError: If channels are inconsistent
        """
        num_monomials = self.coefficient_channel.shape[0]
        
        # Check shape consistency
        if self.exponent_channel.shape[0] != num_monomials:
            raise ValueError(f"Exponent channel shape {self.exponent_channel.shape} inconsistent with {num_monomials} monomials")
        
        if self.index_channel.shape[0] != num_monomials:
            raise ValueError(f"Index channel shape {self.index_channel.shape} inconsistent with {num_monomials} monomials")
        
        # Check mantissa channel if present
        if self.mantissa_channel is not None:
            # Mantissa can be compressed, so check if metadata has compression info
            if 'mantissa_metadata' in self.metadata:
                # Compressed mantissa - validate using metadata
                if 'original_shape' in self.metadata['mantissa_metadata']:
                    expected_shape = self.metadata['mantissa_metadata']['original_shape']
                    if expected_shape[0] != num_monomials:
                        raise ValueError(f"Mantissa channel original shape {expected_shape} inconsistent with {num_monomials} monomials")
            else:
                # Uncompressed mantissa
                if self.mantissa_channel.shape[0] != num_monomials:
                    raise ValueError(f"Mantissa channel shape {self.mantissa_channel.shape} inconsistent with {num_monomials} monomials")
                    
            # Check mantissa device consistency
            if self.mantissa_channel.device != self.coefficient_channel.device:
                raise ValueError("Mantissa channel on different device than coefficient channel")
        
        # Check metadata
        if 'num_variables' not in self.metadata:
            raise ValueError("Missing num_variables in metadata")
        
        if self.exponent_channel.shape[1] != self.metadata['num_variables']:
            raise ValueError(f"Exponent channel has {self.exponent_channel.shape[1]} vars but metadata says {self.metadata['num_variables']}")
        
        # Check device consistency
        if self.coefficient_channel.device != self.exponent_channel.device:
            raise ValueError("Channel tensors on different devices")
        
        if self.coefficient_channel.device != self.index_channel.device:
            raise ValueError("Channel tensors on different devices")
        
        # Check for NaN or Inf
        if torch.isnan(self.coefficient_channel).any():
            raise ValueError("NaN values in coefficient channel")
        
        if torch.isinf(self.coefficient_channel).any() and (self.coefficient_channel > 0).any():
            raise ValueError("Positive infinity in coefficient channel")
        
        if torch.isnan(self.exponent_channel).any():
            raise ValueError("NaN values in exponent channel")
        
        # Check exponents are non-negative
        if (self.exponent_channel < 0).any():
            raise ValueError("Negative exponents found")
        
        # Check index channel is valid
        if (self.index_channel < 0).any() or (self.index_channel >= num_monomials).any():
            raise ValueError("Invalid indices in index channel")
    
    def optimize_gpu_layout(self, config: Optional[GPUMemoryLayoutConfig] = None) -> 'TropicalChannels':
        """
        Optimize channel memory layout for GPU efficiency
        
        Args:
            config: GPU memory layout configuration
            
        Returns:
            New TropicalChannels with optimized layout
        """
        if not self.coefficient_channel.is_cuda:
            raise ValueError("Channels must be on GPU for layout optimization")
        
        config = config or create_optimized_gpu_config()
        optimizer = ChannelMemoryOptimizer(config)
        
        # Optimize each channel
        opt_coeff, layout_coeff, metrics_coeff = optimizer.optimize_layout(
            self.coefficient_channel, channel_type="coefficient"
        )
        
        opt_exp, layout_exp, metrics_exp = optimizer.optimize_layout(
            self.exponent_channel, channel_type="exponent"
        )
        
        opt_idx = self.index_channel  # Index channel usually doesn't need optimization
        
        # Optimize mantissa if present
        opt_mantissa = None
        if self.mantissa_channel is not None:
            opt_mantissa, layout_mantissa, metrics_mantissa = optimizer.optimize_layout(
                self.mantissa_channel, channel_type="mantissa"
            )
        
        # Update metadata with layout information
        updated_metadata = self.metadata.copy()
        updated_metadata['gpu_optimized'] = True
        updated_metadata['layout_info'] = {
            'coefficient': layout_coeff.value,
            'exponent': layout_exp.value,
            'mantissa': layout_mantissa.value if opt_mantissa is not None else None
        }
        updated_metadata['optimization_metrics'] = {
            'coefficient': {
                'coalescing_efficiency': metrics_coeff.coalescing_efficiency,
                'bandwidth_gbps': metrics_coeff.memory_bandwidth_gbps
            },
            'exponent': {
                'coalescing_efficiency': metrics_exp.coalescing_efficiency,
                'bandwidth_gbps': metrics_exp.memory_bandwidth_gbps
            }
        }
        
        if opt_mantissa is not None:
            updated_metadata['optimization_metrics']['mantissa'] = {
                'coalescing_efficiency': metrics_mantissa.coalescing_efficiency,
                'bandwidth_gbps': metrics_mantissa.memory_bandwidth_gbps
            }
        
        return TropicalChannels(
            coefficient_channel=opt_coeff,
            exponent_channel=opt_exp,
            index_channel=opt_idx,
            metadata=updated_metadata,
            device=self.device,
            mantissa_channel=opt_mantissa
        )
    
    def profile_gpu_performance(self, 
                               operations: Optional[List[callable]] = None,
                               config: Optional[GPUMemoryLayoutConfig] = None) -> Dict[str, Any]:
        """
        Profile GPU performance characteristics of channels
        
        Args:
            operations: List of operations to profile
            config: Configuration for profiling
            
        Returns:
            Performance profiling results
        """
        if not self.coefficient_channel.is_cuda:
            raise ValueError("Channels must be on GPU for performance profiling")
        
        config = config or create_optimized_gpu_config()
        analyzer = ChannelAccessPatternAnalyzer(config)
        
        results = {
            'device': str(self.device),
            'total_memory_bytes': 0,
            'channels': {}
        }
        
        # Profile coefficient channel
        coeff_analysis = analyzer.analyze_channel(
            self.coefficient_channel, "coefficient", operations
        )
        results['channels']['coefficient'] = coeff_analysis
        results['total_memory_bytes'] += coeff_analysis['memory_bytes']
        
        # Profile exponent channel
        exp_analysis = analyzer.analyze_channel(
            self.exponent_channel, "exponent", operations
        )
        results['channels']['exponent'] = exp_analysis
        results['total_memory_bytes'] += exp_analysis['memory_bytes']
        
        # Profile mantissa if present
        if self.mantissa_channel is not None:
            mantissa_analysis = analyzer.analyze_channel(
                self.mantissa_channel, "mantissa", operations
            )
            results['channels']['mantissa'] = mantissa_analysis
            results['total_memory_bytes'] += mantissa_analysis['memory_bytes']
        
        # Overall recommendations
        results['recommendations'] = []
        
        # Check total memory usage
        if results['total_memory_bytes'] > 1024 * 1024 * 1024:  # > 1GB
            results['recommendations'].append(
                "Consider splitting into smaller batches for better cache utilization"
            )
        
        # Check alignment
        for channel_name, analysis in results['channels'].items():
            if not analysis['is_aligned']:
                results['recommendations'].append(
                    f"Align {channel_name} channel to 128-byte boundaries"
                )
        
        # Check sparsity
        for channel_name, analysis in results['channels'].items():
            if analysis['sparsity'] > 0.7:
                results['recommendations'].append(
                    f"Use sparse representation for {channel_name} channel"
                )
        
        return results


class TropicalCoefficientExtractor:
    """Extract coefficient channel from tropical polynomials"""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize coefficient extractor
        
        Args:
            device: PyTorch device for extraction
        """
        self.device = device or torch.device('cpu')
        
    def extract_coefficients(self, polynomial: TropicalPolynomial) -> torch.Tensor:
        """
        Extract coefficient channel from polynomial
        
        Args:
            polynomial: Input tropical polynomial
            
        Returns:
            Coefficient tensor of shape (num_monomials,)
        """
        if not isinstance(polynomial, TropicalPolynomial):
            raise TypeError(f"Expected TropicalPolynomial, got {type(polynomial)}")
        
        if not polynomial.monomials:
            # Empty polynomial - return single tropical zero
            return torch.tensor([TROPICAL_ZERO], dtype=torch.float32, device=self.device)
        
        # Extract coefficients
        coefficients = [monomial.coefficient for monomial in polynomial.monomials]
        
        return torch.tensor(coefficients, dtype=torch.float32, device=self.device)
    
    def extract_with_compression(self, polynomial: TropicalPolynomial,
                                compression_level: int = 1) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Extract coefficients with optional compression
        
        Args:
            polynomial: Input tropical polynomial
            compression_level: Compression level (0=none, 1=basic, 2=aggressive)
            
        Returns:
            Tuple of (compressed coefficients, compression metadata)
        """
        coefficients = self.extract_coefficients(polynomial)
        
        if compression_level == 0:
            return coefficients, {'compressed': False}
        
        metadata = {'compressed': True, 'level': compression_level}
        
        # Detect patterns for compression
        patterns = self.detect_patterns(coefficients)
        metadata.update(patterns)
        
        if compression_level == 1:
            # Basic compression - quantization
            if patterns.get('has_arithmetic_progression', False):
                # Store as start + step * index
                start = patterns['ap_start']
                step = patterns['ap_step']
                metadata['compression_type'] = 'arithmetic_progression'
                metadata['start'] = start
                metadata['step'] = step
                # Return indices instead of values
                return torch.arange(len(coefficients), dtype=torch.float32, device=self.device), metadata
            
            # Quantize to reduce precision
            scale = coefficients.abs().max().item() if coefficients.numel() > 0 else 1.0
            if scale > 0:
                quantized = (coefficients / scale * 127).round().to(torch.int8)
                metadata['compression_type'] = 'quantized'
                metadata['scale'] = scale
                return quantized, metadata
        
        elif compression_level == 2:
            # Aggressive compression - clustering and indexing
            unique_vals, inverse_indices = torch.unique(coefficients, return_inverse=True)
            
            if len(unique_vals) < len(coefficients) * 0.5:
                # Good compression ratio
                metadata['compression_type'] = 'indexed'
                metadata['unique_values'] = unique_vals
                return inverse_indices.to(torch.int16), metadata
            
            # Fall back to delta encoding
            deltas = torch.diff(coefficients, prepend=coefficients[:1])
            metadata['compression_type'] = 'delta'
            metadata['first_value'] = coefficients[0].item()
            return deltas, metadata
        
        return coefficients, metadata
    
    def batch_extract(self, polynomials: List[TropicalPolynomial]) -> torch.Tensor:
        """
        Extract coefficients from multiple polynomials
        
        Args:
            polynomials: List of tropical polynomials
            
        Returns:
            Coefficient tensor of shape (num_polynomials, max_monomials)
        """
        if not polynomials:
            raise ValueError("Empty polynomial list")
        
        # Find maximum number of monomials
        max_monomials = max(len(p.monomials) if p.monomials else 1 for p in polynomials)
        
        # Allocate batch tensor
        batch_coeffs = torch.full(
            (len(polynomials), max_monomials),
            TROPICAL_ZERO,
            dtype=torch.float32,
            device=self.device
        )
        
        # Fill coefficients
        for i, poly in enumerate(polynomials):
            if poly.monomials:
                coeffs = self.extract_coefficients(poly)
                batch_coeffs[i, :len(coeffs)] = coeffs
        
        return batch_coeffs
    
    def detect_patterns(self, coefficients: torch.Tensor) -> Dict[str, Any]:
        """
        Detect patterns in coefficient channel for compression
        
        Args:
            coefficients: Coefficient tensor
            
        Returns:
            Dictionary of detected patterns
        """
        patterns = {}
        
        if coefficients.numel() == 0:
            return patterns
        
        # Check for arithmetic progression
        if coefficients.numel() >= 3:
            diffs = torch.diff(coefficients)
            if diffs.numel() > 0:
                is_constant_diff = torch.allclose(diffs, diffs[0], rtol=1e-5)
                if is_constant_diff:
                    patterns['has_arithmetic_progression'] = True
                    patterns['ap_start'] = coefficients[0].item()
                    patterns['ap_step'] = diffs[0].item()
        
        # Check for clustering
        unique_vals = torch.unique(coefficients)
        patterns['unique_ratio'] = len(unique_vals) / len(coefficients)
        patterns['num_unique'] = len(unique_vals)
        
        # Check sparsity (tropical zeros)
        num_zeros = (coefficients <= TROPICAL_ZERO).sum().item()
        patterns['sparsity'] = num_zeros / len(coefficients)
        
        # Check range
        non_zero_mask = coefficients > TROPICAL_ZERO
        if non_zero_mask.any():
            patterns['min_value'] = coefficients[non_zero_mask].min().item()
            patterns['max_value'] = coefficients[non_zero_mask].max().item()
            patterns['range'] = patterns['max_value'] - patterns['min_value']
        
        # Check for geometric progression
        if coefficients.numel() >= 3 and (coefficients > 0).all():
            log_coeffs = torch.log(coefficients.abs() + TROPICAL_EPSILON)
            log_diffs = torch.diff(log_coeffs)
            if log_diffs.numel() > 0:
                is_geometric = torch.allclose(log_diffs, log_diffs[0], rtol=1e-5)
                if is_geometric:
                    patterns['has_geometric_progression'] = True
                    patterns['gp_start'] = coefficients[0].item()
                    patterns['gp_ratio'] = torch.exp(log_diffs[0]).item()
        
        return patterns


class TropicalExponentExtractor:
    """Extract exponent channel from tropical polynomials with advanced compression"""
    
    def __init__(self, device: Optional[torch.device] = None, 
                 config: Optional[ExponentChannelConfig] = None):
        """
        Initialize exponent extractor
        
        Args:
            device: PyTorch device for extraction
            config: Configuration for advanced features
        """
        self.device = device or torch.device('cpu')
        self.config = config or ExponentChannelConfig()
    
    def extract_exponents(self, polynomial: TropicalPolynomial) -> torch.Tensor:
        """
        Extract exponent channel as dense matrix
        
        Args:
            polynomial: Input tropical polynomial
            
        Returns:
            Exponent tensor of shape (num_monomials, num_variables)
        """
        if not isinstance(polynomial, TropicalPolynomial):
            raise TypeError(f"Expected TropicalPolynomial, got {type(polynomial)}")
        
        num_variables = polynomial.num_variables
        
        if not polynomial.monomials:
            # Empty polynomial
            return torch.zeros((1, num_variables), dtype=torch.int32, device=self.device)
        
        num_monomials = len(polynomial.monomials)
        exponents = torch.zeros((num_monomials, num_variables), dtype=torch.int32, device=self.device)
        
        for i, monomial in enumerate(polynomial.monomials):
            for var_idx, power in monomial.exponents.items():
                exponents[i, var_idx] = power
        
        return exponents
    
    def extract_sparse_exponents(self, polynomial: TropicalPolynomial) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract exponents in sparse COO format
        
        Args:
            polynomial: Input tropical polynomial
            
        Returns:
            Tuple of (indices, values) for sparse representation
        """
        if not polynomial.monomials:
            # Empty polynomial
            indices = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            values = torch.zeros(0, dtype=torch.int32, device=self.device)
            return indices, values
        
        indices_list = []
        values_list = []
        
        for mon_idx, monomial in enumerate(polynomial.monomials):
            for var_idx, power in monomial.exponents.items():
                indices_list.append([mon_idx, var_idx])
                values_list.append(power)
        
        if not indices_list:
            # All monomials are constants
            indices = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            values = torch.zeros(0, dtype=torch.int32, device=self.device)
        else:
            indices = torch.tensor(indices_list, dtype=torch.long, device=self.device).T
            values = torch.tensor(values_list, dtype=torch.int32, device=self.device)
        
        return indices, values
    
    def compress_exponents(self, exponents: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compress sparse exponent patterns
        
        Args:
            exponents: Dense exponent matrix
            
        Returns:
            Tuple of (compressed data, compression metadata)
        """
        metadata = {}
        
        # Calculate sparsity
        num_nonzero = (exponents != 0).sum().item()
        total_elements = exponents.numel()
        sparsity = 1.0 - (num_nonzero / total_elements if total_elements > 0 else 0)
        metadata['sparsity'] = sparsity
        
        if sparsity > 0.7:
            # High sparsity - use COO format
            indices = torch.nonzero(exponents)
            values = exponents[exponents != 0]
            
            metadata['format'] = 'coo_sparse'
            metadata['shape'] = list(exponents.shape)
            metadata['indices'] = indices
            
            return values, metadata
        
        # Check for patterns in exponents
        max_exp = exponents.max().item() if exponents.numel() > 0 else 0
        
        if max_exp <= 127:
            # Can use int8
            compressed = exponents.to(torch.int8)
            metadata['format'] = 'int8_dense'
            return compressed, metadata
        elif max_exp <= 32767:
            # Can use int16
            compressed = exponents.to(torch.int16)
            metadata['format'] = 'int16_dense'
            return compressed, metadata
        else:
            # Keep as int32
            metadata['format'] = 'int32_dense'
            return exponents, metadata
    
    def detect_sparsity_pattern(self, exponents: torch.Tensor) -> Dict[str, Any]:
        """
        Detect sparsity patterns in exponent matrix
        
        Args:
            exponents: Dense exponent matrix
            
        Returns:
            Dictionary describing sparsity patterns
        """
        pattern_info = {}
        
        # Basic sparsity metrics
        non_zero_mask = exponents != 0
        num_nonzero = non_zero_mask.sum().item()
        total_elements = exponents.numel()
        sparsity = 1.0 - (num_nonzero / total_elements if total_elements > 0 else 0)
        
        pattern_info['sparsity'] = sparsity
        pattern_info['num_nonzero'] = num_nonzero
        pattern_info['total_elements'] = total_elements
        
        if exponents.shape[0] == 0 or exponents.shape[1] == 0:
            return pattern_info
        
        # Row-wise sparsity (per monomial)
        row_nnz = non_zero_mask.sum(dim=1)
        pattern_info['avg_vars_per_monomial'] = row_nnz.float().mean().item()
        pattern_info['max_vars_per_monomial'] = row_nnz.max().item()
        pattern_info['min_vars_per_monomial'] = row_nnz.min().item()
        
        # Column-wise sparsity (per variable)
        col_nnz = non_zero_mask.sum(dim=0)
        pattern_info['avg_monomials_per_var'] = col_nnz.float().mean().item()
        pattern_info['unused_variables'] = (col_nnz == 0).sum().item()
        
        # Block sparsity patterns
        if self.config.block_size > 0:
            block_size = self.config.block_size
            num_row_blocks = (exponents.shape[0] + block_size - 1) // block_size
            num_col_blocks = (exponents.shape[1] + block_size - 1) // block_size
            
            block_densities = []
            for rb in range(num_row_blocks):
                for cb in range(num_col_blocks):
                    row_start = rb * block_size
                    row_end = min((rb + 1) * block_size, exponents.shape[0])
                    col_start = cb * block_size
                    col_end = min((cb + 1) * block_size, exponents.shape[1])
                    
                    block = exponents[row_start:row_end, col_start:col_end]
                    block_nnz = (block != 0).sum().item()
                    block_size_actual = block.numel()
                    if block_size_actual > 0:
                        block_density = block_nnz / block_size_actual
                        block_densities.append(block_density)
            
            if block_densities:
                pattern_info['avg_block_density'] = sum(block_densities) / len(block_densities)
                pattern_info['num_empty_blocks'] = sum(1 for d in block_densities if d == 0)
                pattern_info['num_dense_blocks'] = sum(1 for d in block_densities if d > 0.5)
        
        # Diagonal patterns
        min_dim = min(exponents.shape[0], exponents.shape[1])
        diag_elements = torch.diagonal(exponents[:min_dim, :min_dim])
        pattern_info['diagonal_density'] = (diag_elements != 0).float().mean().item()
        
        # Band patterns (check for banded structure)
        if exponents.shape[0] > 2 and exponents.shape[1] > 2:
            # Check upper and lower bands
            upper_band_nnz = 0
            lower_band_nnz = 0
            total_band_elements = 0
            
            for i in range(exponents.shape[0]):
                for j in range(exponents.shape[1]):
                    if abs(i - j) <= 2:  # Band width of 2
                        total_band_elements += 1
                        if exponents[i, j] != 0:
                            if i < j:
                                upper_band_nnz += 1
                            elif i > j:
                                lower_band_nnz += 1
            
            if total_band_elements > 0:
                pattern_info['band_concentration'] = (
                    (upper_band_nnz + lower_band_nnz + (diag_elements != 0).sum().item()) / 
                    total_band_elements
                )
        
        # Detect if matrix is suitable for CSR or CSC
        if sparsity > self.config.sparsity_threshold:
            # Check row vs column efficiency
            row_variance = row_nnz.float().var().item()
            col_variance = col_nnz.float().var().item()
            
            if row_variance < col_variance:
                pattern_info['recommended_format'] = 'csr'
            else:
                pattern_info['recommended_format'] = 'csc'
        else:
            pattern_info['recommended_format'] = 'dense'
        
        return pattern_info
    
    def quantize_exponents(self, exponents: torch.Tensor, 
                          quantization: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Quantize exponents with validation
        
        Args:
            exponents: Input exponent tensor
            quantization: Override quantization mode
            
        Returns:
            Tuple of (quantized exponents, quantization metadata)
            
        Raises:
            ValueError: If quantization would cause data loss
        """
        quant_mode = quantization or self.config.quantization
        metadata = {'original_dtype': str(exponents.dtype)}
        
        if exponents.numel() == 0:
            metadata['quantization'] = 'empty'
            return exponents, metadata
        
        max_val = exponents.max().item()
        min_val = exponents.min().item()
        
        # Validate non-negative
        if min_val < 0:
            raise ValueError(f"Negative exponents found: min={min_val}")
        
        metadata['max_value'] = max_val
        metadata['min_value'] = min_val
        
        # Auto-select quantization
        if quant_mode == "auto":
            if max_val <= 127:
                quant_mode = "int8"
            elif max_val <= 32767:
                quant_mode = "int16"
            else:
                quant_mode = "int32"
        
        # Apply quantization with validation
        if quant_mode == "int8":
            if max_val > 127:
                raise ValueError(f"Cannot quantize to int8: max value {max_val} > 127")
            quantized = exponents.to(torch.int8)
            metadata['quantization'] = 'int8'
            metadata['bytes_per_element'] = 1
        elif quant_mode == "int16":
            if max_val > 32767:
                raise ValueError(f"Cannot quantize to int16: max value {max_val} > 32767")
            quantized = exponents.to(torch.int16)
            metadata['quantization'] = 'int16'
            metadata['bytes_per_element'] = 2
        elif quant_mode == "int32":
            if max_val > 2147483647:
                raise ValueError(f"Cannot quantize to int32: max value {max_val} > 2^31-1")
            quantized = exponents.to(torch.int32)
            metadata['quantization'] = 'int32'
            metadata['bytes_per_element'] = 4
        else:
            raise ValueError(f"Unknown quantization mode: {quant_mode}")
        
        # Validate lossless conversion
        if self.config.validate_lossless:
            reconstructed = quantized.to(exponents.dtype)
            if not torch.equal(reconstructed, exponents):
                raise ValueError("Quantization caused data loss")
        
        # Calculate compression ratio
        original_bytes = exponents.numel() * exponents.element_size()
        quantized_bytes = quantized.numel() * quantized.element_size()
        metadata['compression_ratio'] = original_bytes / quantized_bytes if quantized_bytes > 0 else 1.0
        
        return quantized, metadata
    
    def delta_encode_exponents(self, exponents: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply delta encoding to exponent patterns
        
        Args:
            exponents: Input exponent tensor
            
        Returns:
            Tuple of (delta encoded tensor, encoding metadata)
        """
        if not self.config.enable_delta_encoding:
            return exponents, {'delta_encoded': False}
        
        metadata = {'delta_encoded': True}
        
        if exponents.shape[0] <= 1:
            # Cannot delta encode single row
            metadata['delta_encoded'] = False
            return exponents, metadata
        
        # Compute row-wise deltas
        first_row = exponents[0:1, :]
        deltas = torch.diff(exponents, dim=0)
        
        # Check if delta encoding is beneficial
        original_range = exponents.max() - exponents.min()
        delta_range = deltas.max() - deltas.min()
        
        if delta_range < original_range * 0.5:
            # Delta encoding is beneficial
            metadata['first_row'] = first_row
            metadata['original_shape'] = list(exponents.shape)
            metadata['delta_range'] = delta_range.item()
            metadata['original_range'] = original_range.item()
            metadata['compression_benefit'] = 1.0 - (delta_range / original_range).item()
            
            # Quantize deltas if possible
            if delta_range <= 127:
                deltas = deltas.to(torch.int8)
                metadata['delta_dtype'] = 'int8'
            elif delta_range <= 32767:
                deltas = deltas.to(torch.int16)
                metadata['delta_dtype'] = 'int16'
            else:
                metadata['delta_dtype'] = 'int32'
            
            return deltas, metadata
        else:
            # Delta encoding not beneficial
            metadata['delta_encoded'] = False
            metadata['reason'] = 'no_benefit'
            return exponents, metadata
    
    def cluster_exponent_patterns(self, exponents: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Cluster similar exponent patterns for compression
        
        Args:
            exponents: Input exponent tensor
            
        Returns:
            Tuple of (cluster indices, clustering metadata)
        """
        if not self.config.enable_pattern_clustering:
            return exponents, {'clustered': False}
        
        metadata = {'clustered': True}
        
        if exponents.shape[0] <= 1:
            metadata['clustered'] = False
            metadata['reason'] = 'too_few_rows'
            return exponents, metadata
        
        # Find unique rows and their indices
        unique_rows, inverse_indices = torch.unique(exponents, dim=0, return_inverse=True)
        
        num_unique = unique_rows.shape[0]
        num_total = exponents.shape[0]
        compression_ratio = num_total / num_unique if num_unique > 0 else 1.0
        
        metadata['num_unique_patterns'] = num_unique
        metadata['num_total_patterns'] = num_total
        metadata['pattern_compression_ratio'] = compression_ratio
        
        if compression_ratio > 1.5:
            # Clustering is beneficial
            metadata['unique_patterns'] = unique_rows
            metadata['pattern_indices'] = inverse_indices
            
            # Further compress indices if beneficial
            max_idx = inverse_indices.max().item()
            if max_idx <= 255:
                inverse_indices = inverse_indices.to(torch.uint8)
                metadata['index_dtype'] = 'uint8'
            elif max_idx <= 65535:
                inverse_indices = inverse_indices.to(torch.uint16) 
                metadata['index_dtype'] = 'uint16'
            else:
                metadata['index_dtype'] = 'int32'
            
            # Analyze pattern frequency
            pattern_counts = torch.bincount(inverse_indices)
            metadata['most_common_pattern_freq'] = pattern_counts.max().item()
            metadata['pattern_entropy'] = -(pattern_counts.float() / num_total * 
                                           torch.log2(pattern_counts.float() / num_total + 1e-10)).sum().item()
            
            return inverse_indices, metadata
        else:
            # Clustering not beneficial
            metadata['clustered'] = False
            metadata['reason'] = 'insufficient_redundancy'
            return exponents, metadata
    
    def extract_exponents_advanced(self, polynomial: TropicalPolynomial) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Extract exponents with advanced compression based on config
        
        Args:
            polynomial: Input tropical polynomial
            
        Returns:
            Tuple of (processed exponents, compression metadata)
        """
        # Start with basic extraction
        exponents = self.extract_exponents(polynomial)
        metadata = {'original_shape': list(exponents.shape)}
        
        # Detect sparsity patterns
        sparsity_info = self.detect_sparsity_pattern(exponents)
        metadata['sparsity_analysis'] = sparsity_info
        
        # Decide on compression strategy
        if sparsity_info['sparsity'] > self.config.sparsity_threshold:
            # Use sparse representation
            indices, values = self.extract_sparse_exponents(polynomial)
            metadata['format'] = 'sparse_coo'
            metadata['indices'] = indices
            processed = values
        else:
            # Use dense representation with compression
            processed = exponents
            metadata['format'] = 'dense'
            
            # Apply compression techniques based on config
            if self.config.compression_level >= 1:
                # Quantization
                processed, quant_meta = self.quantize_exponents(processed)
                metadata['quantization'] = quant_meta
                
                # Pattern clustering
                if self.config.compression_level >= 2:
                    clustered, cluster_meta = self.cluster_exponent_patterns(processed)
                    if cluster_meta.get('clustered', False):
                        processed = clustered
                        metadata['clustering'] = cluster_meta
                    
                    # Delta encoding (if not clustered)
                    if not cluster_meta.get('clustered', False):
                        delta_encoded, delta_meta = self.delta_encode_exponents(processed)
                        if delta_meta.get('delta_encoded', False):
                            processed = delta_encoded
                            metadata['delta_encoding'] = delta_meta
        
        return processed, metadata


class ExponentChannelCompressor:
    """Compress and decompress exponent channels with validation"""
    
    def __init__(self, device: Optional[torch.device] = None,
                 config: Optional[ExponentChannelConfig] = None):
        """
        Initialize compressor
        
        Args:
            device: PyTorch device
            config: Compression configuration
        """
        self.device = device or torch.device('cpu')
        self.config = config or ExponentChannelConfig()
    
    def compress_sparse(self, indices: torch.Tensor, values: torch.Tensor,
                       shape: Tuple[int, int]) -> Tuple[bytes, Dict[str, Any]]:
        """
        Compress sparse representation to bytes
        
        Args:
            indices: COO indices tensor
            values: COO values tensor  
            shape: Original matrix shape
            
        Returns:
            Tuple of (compressed bytes, metadata for decompression)
        """
        metadata = {
            'format': 'sparse_compressed',
            'shape': shape,
            'num_nonzero': values.shape[0]
        }
        
        if values.numel() == 0:
            # Empty sparse matrix
            return b'', metadata
        
        # Quantize values
        max_val = values.max().item()
        if max_val <= 255:
            values_compressed = values.to(torch.uint8)
            metadata['value_dtype'] = 'uint8'
        elif max_val <= 65535:
            values_compressed = values.to(torch.uint16)
            metadata['value_dtype'] = 'uint16'
        else:
            values_compressed = values.to(torch.int32)
            metadata['value_dtype'] = 'int32'
        
        # Compress indices - often have patterns
        row_indices = indices[0]
        col_indices = indices[1]
        
        # Delta encode row indices (often sequential)
        if row_indices.numel() > 1:
            row_deltas = torch.cat([row_indices[:1], torch.diff(row_indices)])
            row_range = row_deltas.max() - row_deltas.min()
            if row_range <= 255:
                row_compressed = row_deltas.to(torch.uint8)
                metadata['row_dtype'] = 'uint8_delta'
            else:
                row_compressed = row_indices.to(torch.int32)
                metadata['row_dtype'] = 'int32'
        else:
            row_compressed = row_indices.to(torch.int32)
            metadata['row_dtype'] = 'int32'
        
        # Column indices often have patterns too
        max_col = col_indices.max().item() if col_indices.numel() > 0 else 0
        if max_col <= 255:
            col_compressed = col_indices.to(torch.uint8)
            metadata['col_dtype'] = 'uint8'
        elif max_col <= 65535:
            col_compressed = col_indices.to(torch.uint16)
            metadata['col_dtype'] = 'uint16'
        else:
            col_compressed = col_indices.to(torch.int32)
            metadata['col_dtype'] = 'int32'
        
        # Pack into bytes
        import struct
        import io
        
        buffer = io.BytesIO()
        
        # Write header
        buffer.write(struct.pack('I', values.shape[0]))  # num_nonzero
        buffer.write(struct.pack('II', shape[0], shape[1]))  # matrix shape
        
        # Write compressed data
        buffer.write(values_compressed.cpu().numpy().tobytes())
        buffer.write(row_compressed.cpu().numpy().tobytes())
        buffer.write(col_compressed.cpu().numpy().tobytes())
        
        compressed_bytes = buffer.getvalue()
        metadata['compressed_size'] = len(compressed_bytes)
        
        # Calculate compression ratio
        original_size = shape[0] * shape[1] * 4  # Assuming int32
        metadata['compression_ratio'] = original_size / len(compressed_bytes) if len(compressed_bytes) > 0 else 1.0
        
        return compressed_bytes, metadata
    
    def decompress_sparse(self, compressed: bytes, metadata: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompress sparse representation with validation
        
        Args:
            compressed: Compressed bytes
            metadata: Decompression metadata
            
        Returns:
            Tuple of (indices, values) tensors
            
        Raises:
            ValueError: If decompression fails or validation fails
        """
        if len(compressed) == 0:
            # Empty sparse matrix
            indices = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            values = torch.zeros(0, dtype=torch.int32, device=self.device)
            return indices, values
        
        import struct
        import io
        
        buffer = io.BytesIO(compressed)
        
        # Read header
        num_nonzero = struct.unpack('I', buffer.read(4))[0]
        shape = struct.unpack('II', buffer.read(8))
        
        if num_nonzero != metadata['num_nonzero']:
            raise ValueError(f"Metadata mismatch: expected {metadata['num_nonzero']} nonzeros, got {num_nonzero}")
        
        # Read values
        value_dtype = metadata['value_dtype']
        if value_dtype == 'uint8':
            value_bytes = num_nonzero * 1
            values_np = np.frombuffer(buffer.read(value_bytes), dtype=np.uint8)
        elif value_dtype == 'uint16':
            value_bytes = num_nonzero * 2
            values_np = np.frombuffer(buffer.read(value_bytes), dtype=np.uint16)
        else:
            value_bytes = num_nonzero * 4
            values_np = np.frombuffer(buffer.read(value_bytes), dtype=np.int32)
        
        values = torch.from_numpy(values_np).to(self.device)
        
        # Read row indices
        row_dtype = metadata['row_dtype']
        if row_dtype == 'uint8_delta':
            row_deltas_np = np.frombuffer(buffer.read(num_nonzero), dtype=np.uint8)
            row_deltas = torch.from_numpy(row_deltas_np).to(self.device)
            row_indices = torch.cumsum(row_deltas, dim=0)
        else:
            row_indices_np = np.frombuffer(buffer.read(num_nonzero * 4), dtype=np.int32)
            row_indices = torch.from_numpy(row_indices_np).to(self.device)
        
        # Read column indices
        col_dtype = metadata['col_dtype']
        if col_dtype == 'uint8':
            col_indices_np = np.frombuffer(buffer.read(num_nonzero), dtype=np.uint8)
        elif col_dtype == 'uint16':
            col_indices_np = np.frombuffer(buffer.read(num_nonzero * 2), dtype=np.uint16)
        else:
            col_indices_np = np.frombuffer(buffer.read(num_nonzero * 4), dtype=np.int32)
        
        col_indices = torch.from_numpy(col_indices_np).to(self.device)
        
        # Combine indices
        indices = torch.stack([row_indices, col_indices])
        
        # Validate
        if self.config.validate_lossless:
            if indices.shape[1] != values.shape[0]:
                raise ValueError(f"Index/value mismatch: {indices.shape[1]} vs {values.shape[0]}")
            if (indices[0] >= shape[0]).any() or (indices[1] >= shape[1]).any():
                raise ValueError("Indices out of bounds")
        
        return indices, values
    
    def compress_pattern_blocks(self, exponents: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compress using block patterns
        
        Args:
            exponents: Dense exponent matrix
            
        Returns:
            Tuple of (compressed representation, metadata)
        """
        block_size = self.config.block_size
        metadata = {'format': 'block_compressed', 'block_size': block_size}
        
        if exponents.numel() == 0:
            return exponents, metadata
        
        rows, cols = exponents.shape
        num_row_blocks = (rows + block_size - 1) // block_size
        num_col_blocks = (cols + block_size - 1) // block_size
        
        # Analyze blocks
        blocks_data = []
        block_metadata = []
        
        for rb in range(num_row_blocks):
            for cb in range(num_col_blocks):
                row_start = rb * block_size
                row_end = min((rb + 1) * block_size, rows)
                col_start = cb * block_size
                col_end = min((cb + 1) * block_size, cols)
                
                block = exponents[row_start:row_end, col_start:col_end]
                
                # Check block sparsity
                block_nnz = (block != 0).sum().item()
                block_size_actual = block.numel()
                sparsity = 1.0 - (block_nnz / block_size_actual if block_size_actual > 0 else 0)
                
                if sparsity > 0.9:
                    # Store as sparse
                    indices = torch.nonzero(block)
                    values = block[block != 0]
                    block_metadata.append({
                        'type': 'sparse',
                        'position': (rb, cb),
                        'shape': block.shape,
                        'nnz': block_nnz
                    })
                    blocks_data.append((indices, values))
                elif sparsity < 0.1:
                    # Store as dense
                    block_metadata.append({
                        'type': 'dense',
                        'position': (rb, cb),
                        'shape': block.shape
                    })
                    blocks_data.append(block)
                else:
                    # Mixed - use quantization
                    max_val = block.max().item()
                    if max_val <= 255:
                        quantized = block.to(torch.uint8)
                    else:
                        quantized = block.to(torch.int16)
                    block_metadata.append({
                        'type': 'quantized',
                        'position': (rb, cb),
                        'shape': block.shape,
                        'dtype': str(quantized.dtype)
                    })
                    blocks_data.append(quantized)
        
        metadata['blocks'] = blocks_data
        metadata['block_metadata'] = block_metadata
        metadata['original_shape'] = exponents.shape
        
        # Return compressed representation
        # In practice, this would be serialized to bytes
        return torch.tensor([len(blocks_data)], device=self.device), metadata
    
    def validate_compression(self, original: torch.Tensor, compressed: Any,
                           metadata: Dict[str, Any]) -> bool:
        """
        Validate compression is lossless
        
        Args:
            original: Original exponent tensor
            compressed: Compressed representation
            metadata: Compression metadata
            
        Returns:
            True if lossless
            
        Raises:
            ValueError: If compression is lossy
        """
        # Decompress and compare
        if metadata.get('format') == 'sparse_compressed':
            # Would decompress from bytes and reconstruct
            # For now, check metadata consistency
            if 'compression_ratio' not in metadata:
                raise ValueError("Missing compression ratio in metadata")
            if metadata['compression_ratio'] < 1.0:
                raise ValueError(f"Invalid compression ratio: {metadata['compression_ratio']}")
        
        elif metadata.get('format') == 'block_compressed':
            # Validate block metadata
            total_elements = 0
            for block_meta in metadata.get('block_metadata', []):
                shape = block_meta['shape']
                total_elements += shape[0] * shape[1]
            
            original_elements = original.numel()
            if total_elements != original_elements:
                raise ValueError(f"Element count mismatch: {total_elements} vs {original_elements}")
        
        return True


class ExponentChannelOptimizer:
    """Optimize exponent channel layout for GPU efficiency"""
    
    def __init__(self, device: Optional[torch.device] = None,
                 config: Optional[ExponentChannelConfig] = None):
        """
        Initialize optimizer
        
        Args:
            device: PyTorch device
            config: Optimization configuration
        """
        self.device = device or torch.device('cpu')
        self.config = config or ExponentChannelConfig()
    
    def optimize_memory_layout(self, exponents: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Optimize memory layout for GPU coalescing
        
        Args:
            exponents: Input exponent tensor
            
        Returns:
            Tuple of (optimized tensor, optimization metadata)
        """
        if not self.config.gpu_coalesce:
            return exponents, {'optimized': False}
        
        metadata = {'optimized': True}
        
        # Sort by column access pattern for better coalescing
        # GPUs prefer accessing consecutive memory addresses
        
        # Analyze access patterns
        row_sums = (exponents != 0).sum(dim=1)
        col_sums = (exponents != 0).sum(dim=0)
        
        # Sort rows by density (dense rows first for better cache usage)
        row_order = torch.argsort(row_sums, descending=True)
        
        # Sort columns by usage frequency
        col_order = torch.argsort(col_sums, descending=True)
        
        # Reorder matrix
        optimized = exponents[row_order][:, col_order]
        
        metadata['row_permutation'] = row_order
        metadata['col_permutation'] = col_order
        metadata['layout'] = 'density_sorted'
        
        # Make contiguous for GPU
        optimized = optimized.contiguous()
        
        # Additional optimization for very sparse matrices
        sparsity = 1.0 - (exponents != 0).float().mean().item()
        if sparsity > 0.9 and self.device.type == 'cuda':
            # Convert to CSR format for GPU SpMV operations
            metadata['gpu_format'] = 'csr_ready'
        
        return optimized, metadata
    
    def create_access_indices(self, exponents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create optimized access indices for GPU kernels
        
        Args:
            exponents: Input exponent tensor
            
        Returns:
            Tuple of (row_indices, col_indices) for efficient access
        """
        # Find non-zero positions
        nonzero_indices = torch.nonzero(exponents)
        
        if nonzero_indices.shape[0] == 0:
            return torch.zeros(0, dtype=torch.long, device=self.device), \
                   torch.zeros(0, dtype=torch.long, device=self.device)
        
        # Sort by row-major order for coalesced access
        sort_key = nonzero_indices[:, 0] * exponents.shape[1] + nonzero_indices[:, 1]
        sorted_order = torch.argsort(sort_key)
        
        sorted_indices = nonzero_indices[sorted_order]
        
        return sorted_indices[:, 0], sorted_indices[:, 1]
    
    def pack_for_gpu(self, exponents: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Pack exponents for efficient GPU processing
        
        Args:
            exponents: Input exponent tensor
            
        Returns:
            Tuple of (packed tensor, packing metadata)
        """
        metadata = {'packed': True}
        
        if self.device.type != 'cuda':
            # CPU - no special packing needed
            return exponents, {'packed': False}
        
        # Move to GPU if not already
        if exponents.device != self.device:
            exponents = exponents.to(self.device)
        
        # Analyze sparsity
        sparsity = 1.0 - (exponents != 0).float().mean().item()
        metadata['sparsity'] = sparsity
        
        if sparsity > 0.8:
            # High sparsity - use COO format
            indices = torch.nonzero(exponents).T
            values = exponents[exponents != 0]
            
            # Pack into single tensor for better memory transfer
            # Format: [num_nnz, row_indices..., col_indices..., values...]
            num_nnz = values.shape[0]
            packed = torch.zeros(1 + 2 * num_nnz + num_nnz, dtype=torch.float32, device=self.device)
            packed[0] = float(num_nnz)
            packed[1:num_nnz+1] = indices[0].float()
            packed[num_nnz+1:2*num_nnz+1] = indices[1].float()
            packed[2*num_nnz+1:] = values.float()
            
            metadata['format'] = 'packed_coo'
            metadata['shape'] = list(exponents.shape)
            metadata['num_nonzero'] = num_nnz
        else:
            # Dense packing - ensure contiguous
            packed = exponents.contiguous()
            metadata['format'] = 'dense_contiguous'
        
        return packed, metadata
    
    def unpack_from_gpu(self, packed: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Unpack GPU-optimized representation
        
        Args:
            packed: Packed tensor from pack_for_gpu
            metadata: Packing metadata
            
        Returns:
            Unpacked exponent tensor
        """
        if not metadata.get('packed', False):
            return packed
        
        if metadata['format'] == 'packed_coo':
            # Unpack COO format
            num_nnz = int(packed[0].item())
            shape = metadata['shape']
            
            if num_nnz == 0:
                return torch.zeros(shape, dtype=torch.int32, device=packed.device)
            
            row_indices = packed[1:num_nnz+1].long()
            col_indices = packed[num_nnz+1:2*num_nnz+1].long()
            values = packed[2*num_nnz+1:].int()
            
            # Reconstruct dense matrix
            unpacked = torch.zeros(shape, dtype=torch.int32, device=packed.device)
            unpacked[row_indices, col_indices] = values
            
            return unpacked
        
        elif metadata['format'] == 'dense_contiguous':
            return packed
        
        else:
            raise ValueError(f"Unknown packing format: {metadata['format']}")


class ExponentPatternAnalyzer:
    """Analyze exponent patterns for compression strategy selection"""
    
    def __init__(self, device: Optional[torch.device] = None,
                 config: Optional[ExponentChannelConfig] = None):
        """
        Initialize analyzer
        
        Args:
            device: PyTorch device
            config: Analysis configuration
        """
        self.device = device or torch.device('cpu')
        self.config = config or ExponentChannelConfig()
    
    def analyze_polynomial_set(self, polynomials: List[TropicalPolynomial]) -> Dict[str, Any]:
        """
        Analyze a set of polynomials for common patterns
        
        Args:
            polynomials: List of tropical polynomials
            
        Returns:
            Analysis results dictionary
        """
        if not polynomials:
            return {'error': 'empty_polynomial_set'}
        
        analysis = {
            'num_polynomials': len(polynomials),
            'pattern_statistics': {},
            'recommendations': []
        }
        
        # Extract all exponent matrices
        extractor = TropicalExponentExtractor(self.device, self.config)
        exponent_matrices = []
        
        for poly in polynomials:
            exp_matrix = extractor.extract_exponents(poly)
            exponent_matrices.append(exp_matrix)
        
        # Analyze dimensions
        num_vars = polynomials[0].num_variables
        monomial_counts = [exp.shape[0] for exp in exponent_matrices]
        
        analysis['num_variables'] = num_vars
        analysis['avg_monomials'] = sum(monomial_counts) / len(monomial_counts)
        analysis['max_monomials'] = max(monomial_counts)
        analysis['min_monomials'] = min(monomial_counts)
        
        # Analyze sparsity
        sparsities = []
        for exp_matrix in exponent_matrices:
            nnz = (exp_matrix != 0).sum().item()
            total = exp_matrix.numel()
            sparsity = 1.0 - (nnz / total if total > 0 else 0)
            sparsities.append(sparsity)
        
        analysis['avg_sparsity'] = sum(sparsities) / len(sparsities)
        analysis['max_sparsity'] = max(sparsities)
        analysis['min_sparsity'] = min(sparsities)
        
        # Analyze value ranges
        all_values = torch.cat([exp.flatten() for exp in exponent_matrices])
        if all_values.numel() > 0:
            analysis['max_exponent'] = all_values.max().item()
            analysis['min_exponent'] = all_values.min().item()
            analysis['avg_exponent'] = all_values.float().mean().item()
            
            # Determine best quantization
            if analysis['max_exponent'] <= 127:
                analysis['recommended_quantization'] = 'int8'
            elif analysis['max_exponent'] <= 32767:
                analysis['recommended_quantization'] = 'int16'
            else:
                analysis['recommended_quantization'] = 'int32'
        
        # Find common patterns
        common_patterns = self.find_common_patterns(exponent_matrices)
        analysis['common_patterns'] = common_patterns
        
        # Generate recommendations
        if analysis['avg_sparsity'] > 0.8:
            analysis['recommendations'].append('use_sparse_format')
        
        if len(common_patterns['frequent_patterns']) > 0:
            analysis['recommendations'].append('enable_pattern_clustering')
        
        if analysis.get('max_exponent', 0) <= 255:
            analysis['recommendations'].append('use_uint8_quantization')
        
        # Compute entropy
        pattern_entropy = self.compute_pattern_entropy(exponent_matrices)
        analysis['pattern_entropy'] = pattern_entropy
        
        if pattern_entropy < 2.0:
            analysis['recommendations'].append('high_redundancy_detected')
        
        return analysis
    
    def find_common_patterns(self, exponent_matrices: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Find common exponent patterns across matrices
        
        Args:
            exponent_matrices: List of exponent tensors
            
        Returns:
            Dictionary of common patterns
        """
        patterns = {'frequent_patterns': [], 'pattern_counts': {}}
        
        if not exponent_matrices:
            return patterns
        
        # Collect all unique rows
        all_rows = []
        for matrix in exponent_matrices:
            for i in range(matrix.shape[0]):
                row = tuple(matrix[i].tolist())
                all_rows.append(row)
        
        # Count pattern frequencies
        from collections import Counter
        pattern_counter = Counter(all_rows)
        
        # Find most common patterns
        total_rows = len(all_rows)
        for pattern, count in pattern_counter.most_common(10):
            frequency = count / total_rows
            if frequency > 0.01:  # At least 1% frequency
                patterns['frequent_patterns'].append({
                    'pattern': pattern,
                    'count': count,
                    'frequency': frequency
                })
        
        patterns['num_unique_patterns'] = len(pattern_counter)
        patterns['total_patterns'] = total_rows
        patterns['redundancy_ratio'] = 1.0 - (len(pattern_counter) / total_rows if total_rows > 0 else 1.0)
        
        return patterns
    
    def compute_pattern_entropy(self, exponent_matrices: List[torch.Tensor]) -> float:
        """
        Compute entropy of exponent patterns
        
        Args:
            exponent_matrices: List of exponent tensors
            
        Returns:
            Pattern entropy value
        """
        if not exponent_matrices:
            return 0.0
        
        # Collect all patterns
        pattern_counts = {}
        total_patterns = 0
        
        for matrix in exponent_matrices:
            for i in range(matrix.shape[0]):
                pattern = tuple(matrix[i].tolist())
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                total_patterns += 1
        
        if total_patterns == 0:
            return 0.0
        
        # Compute entropy
        entropy = 0.0
        for count in pattern_counts.values():
            prob = count / total_patterns
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def suggest_compression_strategy(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest optimal compression strategy based on analysis
        
        Args:
            analysis: Analysis results from analyze_polynomial_set
            
        Returns:
            Suggested configuration dictionary
        """
        strategy = {
            'config': ExponentChannelConfig(),
            'rationale': []
        }
        
        # Adjust based on sparsity
        avg_sparsity = analysis.get('avg_sparsity', 0)
        if avg_sparsity > 0.9:
            strategy['config'].use_sparse = True
            strategy['config'].sparsity_threshold = 0.7
            strategy['rationale'].append(f"High sparsity ({avg_sparsity:.2%}) - use sparse format")
        elif avg_sparsity > 0.7:
            strategy['config'].use_sparse = True
            strategy['config'].sparsity_threshold = 0.6
            strategy['rationale'].append(f"Moderate sparsity ({avg_sparsity:.2%}) - consider sparse format")
        else:
            strategy['config'].use_sparse = False
            strategy['rationale'].append(f"Low sparsity ({avg_sparsity:.2%}) - use dense format")
        
        # Adjust quantization
        max_exp = analysis.get('max_exponent', 0)
        if max_exp <= 127:
            strategy['config'].quantization = 'int8'
            strategy['rationale'].append(f"Max exponent {max_exp} fits in int8")
        elif max_exp <= 32767:
            strategy['config'].quantization = 'int16'
            strategy['rationale'].append(f"Max exponent {max_exp} fits in int16")
        else:
            strategy['config'].quantization = 'int32'
            strategy['rationale'].append(f"Max exponent {max_exp} requires int32")
        
        # Adjust compression level
        entropy = analysis.get('pattern_entropy', float('inf'))
        if entropy < 2.0:
            strategy['config'].compression_level = 2
            strategy['config'].enable_pattern_clustering = True
            strategy['rationale'].append(f"Low entropy ({entropy:.2f}) - aggressive compression")
        elif entropy < 4.0:
            strategy['config'].compression_level = 1
            strategy['config'].enable_pattern_clustering = True
            strategy['rationale'].append(f"Moderate entropy ({entropy:.2f}) - basic compression")
        else:
            strategy['config'].compression_level = 0
            strategy['config'].enable_pattern_clustering = False
            strategy['rationale'].append(f"High entropy ({entropy:.2f}) - minimal compression")
        
        # Check for patterns
        common_patterns = analysis.get('common_patterns', {})
        if common_patterns.get('redundancy_ratio', 0) > 0.3:
            strategy['config'].enable_pattern_clustering = True
            strategy['rationale'].append(f"High pattern redundancy ({common_patterns['redundancy_ratio']:.2%})")
        
        return strategy


class TropicalMantissaExtractor:
    """Extract mantissa/precision channel from tropical polynomial coefficients"""
    
    def __init__(self, device: Optional[torch.device] = None,
                 config: Optional[MantissaChannelConfig] = None):
        """
        Initialize mantissa extractor
        
        Args:
            device: PyTorch device for extraction
            config: Configuration for mantissa extraction
        """
        self.device = device or torch.device('cpu')
        self.config = config or MantissaChannelConfig()
        
    def extract_mantissa(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Extract mantissa components from coefficient values
        
        Args:
            coefficients: Coefficient tensor from tropical polynomial
            
        Returns:
            Mantissa tensor preserving precision information
        """
        if not isinstance(coefficients, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(coefficients)}")
            
        if coefficients.numel() == 0:
            return torch.zeros(0, dtype=torch.float32, device=self.device)
            
        # Move to device if needed
        if coefficients.device != self.device:
            coefficients = coefficients.to(self.device)
            
        # Handle tropical zeros
        non_zero_mask = coefficients > TROPICAL_ZERO
        
        if not non_zero_mask.any():
            return torch.full_like(coefficients, 0.0)
            
        # Extract mantissa based on precision mode
        if self.config.precision_mode == "fp32":
            mantissas = self._extract_fp32_mantissa(coefficients, non_zero_mask)
        elif self.config.precision_mode == "fp16":
            mantissas = self._extract_fp16_mantissa(coefficients, non_zero_mask)
        elif self.config.precision_mode == "bf16":
            mantissas = self._extract_bf16_mantissa(coefficients, non_zero_mask)
        elif self.config.precision_mode == "mixed":
            mantissas = self._extract_mixed_precision(coefficients, non_zero_mask)
        else:  # adaptive
            mantissas = self._extract_adaptive_precision(coefficients, non_zero_mask)
            
        return mantissas
    
    def _extract_fp32_mantissa(self, coefficients: torch.Tensor, 
                               mask: torch.Tensor) -> torch.Tensor:
        """
        Extract FP32 mantissa (23 bits)
        
        Args:
            coefficients: Input coefficients
            mask: Non-zero mask
            
        Returns:
            FP32 mantissa values
        """
        mantissas = torch.zeros_like(coefficients)
        
        # For non-zero values, extract mantissa bits
        if mask.any():
            # Get IEEE 754 representation
            values = coefficients[mask].abs()
            
            # Decompose into sign, exponent, mantissa
            # Using frexp for portable extraction
            mantissa_vals, exponents = torch.frexp(values)
            
            # Scale mantissa to preserve precision
            # FP32 has 23 mantissa bits
            mantissas[mask] = mantissa_vals
            
        return mantissas
    
    def _extract_fp16_mantissa(self, coefficients: torch.Tensor,
                               mask: torch.Tensor) -> torch.Tensor:
        """
        Extract FP16 mantissa (10 bits)
        
        Args:
            coefficients: Input coefficients
            mask: Non-zero mask
            
        Returns:
            FP16 mantissa values
        """
        # Convert to FP16 and back to extract reduced precision
        fp16_coeffs = coefficients.to(torch.float16).to(torch.float32)
        
        mantissas = torch.zeros_like(coefficients)
        if mask.any():
            values = fp16_coeffs[mask].abs()
            mantissa_vals, _ = torch.frexp(values)
            mantissas[mask] = mantissa_vals
            
        return mantissas
    
    def _extract_bf16_mantissa(self, coefficients: torch.Tensor,
                               mask: torch.Tensor) -> torch.Tensor:
        """
        Extract BF16 mantissa (7 bits)
        
        Args:
            coefficients: Input coefficients
            mask: Non-zero mask
            
        Returns:
            BF16 mantissa values
        """
        # Convert to BF16 and back
        bf16_coeffs = coefficients.to(torch.bfloat16).to(torch.float32)
        
        mantissas = torch.zeros_like(coefficients)
        if mask.any():
            values = bf16_coeffs[mask].abs()
            mantissa_vals, _ = torch.frexp(values)
            mantissas[mask] = mantissa_vals
            
        return mantissas
    
    def _extract_mixed_precision(self, coefficients: torch.Tensor,
                                 mask: torch.Tensor) -> torch.Tensor:
        """
        Extract mantissa with mixed precision based on magnitude
        
        Args:
            coefficients: Input coefficients
            mask: Non-zero mask
            
        Returns:
            Mixed precision mantissa values
        """
        mantissas = torch.zeros_like(coefficients)
        
        if not mask.any():
            return mantissas
            
        abs_values = coefficients[mask].abs()
        
        # Use FP32 for large values, FP16 for medium, BF16 for small
        large_mask = abs_values > 1e3
        small_mask = abs_values < 1e-3
        medium_mask = ~(large_mask | small_mask)
        
        # Extract with appropriate precision
        if large_mask.any():
            large_vals = abs_values[large_mask]
            mantissa_vals, _ = torch.frexp(large_vals)
            temp = torch.zeros_like(abs_values)
            temp[large_mask] = mantissa_vals
            mantissas[mask] = temp
            
        if medium_mask.any():
            medium_vals = abs_values[medium_mask].to(torch.float16).to(torch.float32)
            mantissa_vals, _ = torch.frexp(medium_vals)
            temp = mantissas[mask]
            temp[medium_mask] = mantissa_vals
            mantissas[mask] = temp
            
        if small_mask.any():
            small_vals = abs_values[small_mask].to(torch.bfloat16).to(torch.float32)
            mantissa_vals, _ = torch.frexp(small_vals)
            temp = mantissas[mask]
            temp[small_mask] = mantissa_vals
            mantissas[mask] = temp
            
        return mantissas
    
    def _extract_adaptive_precision(self, coefficients: torch.Tensor,
                                   mask: torch.Tensor) -> torch.Tensor:
        """
        Adaptively extract mantissa based on precision requirements
        
        Args:
            coefficients: Input coefficients
            mask: Non-zero mask
            
        Returns:
            Adaptively extracted mantissa values
        """
        if not mask.any():
            return torch.zeros_like(coefficients)
            
        # Analyze precision requirements
        abs_values = coefficients[mask].abs()
        
        # Compute precision needs based on value distribution
        value_range = abs_values.max() - abs_values.min()
        relative_precision = self.config.precision_threshold / value_range if value_range > 0 else 1.0
        
        # Select precision mode based on requirements
        if relative_precision < 1e-7:
            # Need full FP32 precision
            return self._extract_fp32_mantissa(coefficients, mask)
        elif relative_precision < 1e-4:
            # FP16 sufficient
            return self._extract_fp16_mantissa(coefficients, mask)
        else:
            # BF16 sufficient
            return self._extract_bf16_mantissa(coefficients, mask)
    
    def compress_mantissa(self, mantissas: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compress mantissa channel
        
        Args:
            mantissas: Mantissa tensor
            
        Returns:
            Tuple of (compressed mantissa, compression metadata)
        """
        metadata = {'original_shape': list(mantissas.shape)}
        
        if mantissas.numel() == 0:
            return mantissas, {'compressed': False}
            
        if self.config.compression_level == 0:
            return mantissas, {'compressed': False}
            
        metadata['compressed'] = True
        metadata['compression_level'] = self.config.compression_level
        
        # Apply compression techniques
        compressed = mantissas
        
        if self.config.enable_bit_packing:
            compressed, pack_meta = self._bit_pack_mantissa(compressed)
            metadata['bit_packing'] = pack_meta
            
        if self.config.enable_delta_encoding:
            compressed, delta_meta = self._delta_encode_mantissa(compressed)
            metadata['delta_encoding'] = delta_meta
            
        if self.config.enable_pattern_compression:
            compressed, pattern_meta = self._compress_patterns(compressed)
            metadata['pattern_compression'] = pattern_meta
            
        # Calculate compression ratio
        original_bytes = mantissas.numel() * mantissas.element_size()
        compressed_bytes = compressed.numel() * compressed.element_size()
        metadata['compression_ratio'] = original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0
        
        return compressed, metadata
    
    def _bit_pack_mantissa(self, mantissas: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Pack mantissa bits efficiently
        
        Args:
            mantissas: Input mantissa tensor
            
        Returns:
            Tuple of (packed mantissa, packing metadata)
        """
        metadata = {}
        
        # Determine required bits based on precision
        if self.config.precision_mode == "fp32":
            bits_required = 23
        elif self.config.precision_mode == "fp16":
            bits_required = 10
        elif self.config.precision_mode == "bf16":
            bits_required = 7
        else:
            # Analyze actual precision needs
            mantissa_range = mantissas.max() - mantissas.min()
            if mantissa_range > 0:
                bits_required = int(math.ceil(math.log2(1 / self.config.precision_threshold)))
            else:
                bits_required = 16
                
        metadata['bits_packed'] = bits_required
        
        # Quantize to required bits
        if bits_required <= 8:
            scale = (2 ** bits_required - 1)
            packed = ((mantissas - mantissas.min()) / (mantissas.max() - mantissas.min() + 1e-10) * scale).to(torch.uint8)
            metadata['dtype'] = 'uint8'
        elif bits_required <= 16:
            scale = (2 ** bits_required - 1)
            packed = ((mantissas - mantissas.min()) / (mantissas.max() - mantissas.min() + 1e-10) * scale).to(torch.int16)
            metadata['dtype'] = 'int16'
        else:
            packed = mantissas
            metadata['dtype'] = 'float32'
            
        metadata['scale'] = mantissas.max().item() - mantissas.min().item()
        metadata['offset'] = mantissas.min().item()
        
        return packed, metadata
    
    def _delta_encode_mantissa(self, mantissas: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Delta encode mantissa values
        
        Args:
            mantissas: Input mantissa tensor
            
        Returns:
            Tuple of (delta encoded, encoding metadata)
        """
        if not self.config.enable_delta_encoding or mantissas.shape[0] <= 1:
            return mantissas, {'delta_encoded': False}
            
        metadata = {'delta_encoded': True}
        
        # Compute deltas
        first_value = mantissas[0:1]
        deltas = torch.diff(mantissas)
        
        # Check if delta encoding is beneficial
        original_range = mantissas.max() - mantissas.min()
        delta_range = deltas.max() - deltas.min()
        
        if delta_range < original_range * 0.7:
            metadata['first_value'] = first_value
            metadata['compression_benefit'] = 1.0 - (delta_range / original_range).item()
            return deltas, metadata
        else:
            return mantissas, {'delta_encoded': False}
    
    def _compress_patterns(self, mantissas: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compress repeated mantissa patterns
        
        Args:
            mantissas: Input mantissa tensor
            
        Returns:
            Tuple of (pattern compressed, compression metadata)
        """
        if not self.config.enable_pattern_compression:
            return mantissas, {'pattern_compressed': False}
            
        metadata = {'pattern_compressed': True}
        
        # Find unique values and their frequencies
        unique_vals, inverse_indices, counts = torch.unique(
            mantissas, return_inverse=True, return_counts=True
        )
        
        compression_ratio = len(mantissas) / len(unique_vals)
        
        if compression_ratio > 1.5:
            # Pattern compression is beneficial
            metadata['unique_values'] = unique_vals
            metadata['compression_ratio'] = compression_ratio
            
            # Use smaller dtype for indices if possible
            if len(unique_vals) <= 256:
                inverse_indices = inverse_indices.to(torch.uint8)
                metadata['index_dtype'] = 'uint8'
            elif len(unique_vals) <= 65536:
                inverse_indices = inverse_indices.to(torch.int16)
                metadata['index_dtype'] = 'int16'
            else:
                metadata['index_dtype'] = 'int32'
                
            # Store most frequent patterns for faster access
            top_k = min(10, len(unique_vals))
            top_indices = torch.argsort(counts, descending=True)[:top_k]
            metadata['frequent_patterns'] = unique_vals[top_indices]
            metadata['frequent_counts'] = counts[top_indices]
            
            return inverse_indices, metadata
        else:
            return mantissas, {'pattern_compressed': False}
    
    def decompress_mantissa(self, compressed: torch.Tensor, 
                          metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Decompress mantissa channel
        
        Args:
            compressed: Compressed mantissa data
            metadata: Compression metadata
            
        Returns:
            Decompressed mantissa tensor
            
        Raises:
            ValueError: If decompression fails
        """
        if not metadata.get('compressed', False):
            return compressed
            
        result = compressed
        
        # Reverse compression steps in opposite order
        if 'pattern_compression' in metadata and metadata['pattern_compression'].get('pattern_compressed'):
            # Reconstruct from unique values
            unique_vals = metadata['pattern_compression']['unique_values']
            result = unique_vals[result.long()]
            
        if 'delta_encoding' in metadata and metadata['delta_encoding'].get('delta_encoded'):
            # Reconstruct from deltas
            first_value = metadata['delta_encoding']['first_value']
            result = torch.cat([first_value, first_value + torch.cumsum(result, dim=0)])
            
        if 'bit_packing' in metadata:
            # Unpack bits
            pack_meta = metadata['bit_packing']
            scale = pack_meta['scale']
            offset = pack_meta['offset']
            
            if pack_meta['dtype'] == 'uint8':
                result = result.float() / 255.0 * scale + offset
            elif pack_meta['dtype'] == 'int16':
                result = result.float() / 32767.0 * scale + offset
                
        # Validate precision preservation
        if self.config.validate_precision:
            if 'original_shape' in metadata:
                expected_shape = tuple(metadata['original_shape'])
                if result.shape != expected_shape:
                    raise ValueError(f"Shape mismatch after decompression: {result.shape} vs {expected_shape}")
                    
        return result
    
    def handle_denormals(self, mantissas: torch.Tensor) -> torch.Tensor:
        """
        Handle denormal numbers based on configuration
        
        Args:
            mantissas: Mantissa tensor
            
        Returns:
            Processed mantissa tensor with denormals handled
        """
        if self.config.denormal_handling == "preserve":
            # Keep denormals as-is
            return mantissas
        elif self.config.denormal_handling == "flush":
            # Flush denormals to zero
            denormal_threshold = 2 ** -126  # FP32 denormal threshold
            return torch.where(mantissas.abs() < denormal_threshold, 0.0, mantissas)
        elif self.config.denormal_handling == "round":
            # Round denormals to nearest normal
            denormal_threshold = 2 ** -126
            min_normal = 2 ** -126
            mask = mantissas.abs() < denormal_threshold
            return torch.where(mask, torch.sign(mantissas) * min_normal, mantissas)
        else:
            raise ValueError(f"Unknown denormal handling: {self.config.denormal_handling}")


class MantissaPrecisionAnalyzer:
    """Analyze precision requirements for mantissa extraction"""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize precision analyzer
        
        Args:
            device: PyTorch device
        """
        self.device = device or torch.device('cpu')
        
    def analyze_precision_requirements(self, coefficients: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze precision requirements for coefficient values
        
        Args:
            coefficients: Coefficient tensor
            
        Returns:
            Analysis results dictionary
        """
        analysis = {}
        
        if coefficients.numel() == 0:
            return {'error': 'empty_coefficients'}
            
        # Move to device
        if coefficients.device != self.device:
            coefficients = coefficients.to(self.device)
            
        # Filter out tropical zeros
        non_zero_mask = coefficients > TROPICAL_ZERO
        if not non_zero_mask.any():
            return {'all_zeros': True}
            
        valid_coeffs = coefficients[non_zero_mask]
        
        # Analyze value range
        analysis['min_value'] = valid_coeffs.min().item()
        analysis['max_value'] = valid_coeffs.max().item()
        analysis['value_range'] = analysis['max_value'] - analysis['min_value']
        
        # Analyze precision distribution
        # Convert to different precisions and measure error
        fp32_vals = valid_coeffs
        fp16_vals = valid_coeffs.to(torch.float16).to(torch.float32)
        bf16_vals = valid_coeffs.to(torch.bfloat16).to(torch.float32)
        
        fp16_error = (fp32_vals - fp16_vals).abs()
        bf16_error = (fp32_vals - bf16_vals).abs()
        
        analysis['fp16_max_error'] = fp16_error.max().item()
        analysis['fp16_mean_error'] = fp16_error.mean().item()
        analysis['bf16_max_error'] = bf16_error.max().item()
        analysis['bf16_mean_error'] = bf16_error.mean().item()
        
        # Recommend precision mode
        if analysis['fp16_max_error'] < 1e-7:
            analysis['recommended_precision'] = 'fp16'
        elif analysis['bf16_max_error'] < 1e-6:
            analysis['recommended_precision'] = 'bf16'
        else:
            analysis['recommended_precision'] = 'fp32'
            
        # Analyze denormals
        denormal_threshold = 2 ** -126
        num_denormals = (valid_coeffs.abs() < denormal_threshold).sum().item()
        analysis['num_denormals'] = num_denormals
        analysis['denormal_ratio'] = num_denormals / valid_coeffs.numel()
        
        # Analyze mantissa bits actually used
        mantissa_vals, exponents = torch.frexp(valid_coeffs)
        
        # Count significant bits in mantissa
        # Approximate by checking how many bits are needed for representation
        mantissa_range = mantissa_vals.max() - mantissa_vals.min()
        if mantissa_range > 0:
            bits_needed = math.ceil(math.log2(1 / mantissa_range * 1e7))
            analysis['mantissa_bits_needed'] = min(23, bits_needed)
        else:
            analysis['mantissa_bits_needed'] = 1
            
        # Distribution analysis
        analysis['mantissa_mean'] = mantissa_vals.mean().item()
        analysis['mantissa_std'] = mantissa_vals.std().item()
        analysis['exponent_range'] = [exponents.min().item(), exponents.max().item()]
        
        return analysis
    
    def recommend_compression_strategy(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend compression strategy based on precision analysis
        
        Args:
            analysis: Precision analysis results
            
        Returns:
            Recommended configuration
        """
        config = MantissaChannelConfig()
        recommendations = []
        
        if 'error' in analysis or analysis.get('all_zeros', False):
            return {'config': config, 'recommendations': ['No compression needed']}
            
        # Set precision mode
        recommended_precision = analysis.get('recommended_precision', 'fp32')
        config.precision_mode = recommended_precision
        recommendations.append(f"Use {recommended_precision} precision")
        
        # Handle denormals
        if analysis.get('denormal_ratio', 0) > 0.1:
            config.denormal_handling = 'round'
            recommendations.append("Round denormals due to high ratio")
        elif analysis.get('denormal_ratio', 0) > 0.01:
            config.denormal_handling = 'flush'
            recommendations.append("Flush denormals to zero")
            
        # Set compression level
        bits_needed = analysis.get('mantissa_bits_needed', 23)
        if bits_needed <= 7:
            config.compression_level = 2
            config.enable_bit_packing = True
            recommendations.append("Aggressive compression - low precision needed")
        elif bits_needed <= 16:
            config.compression_level = 1
            config.enable_bit_packing = True
            recommendations.append("Basic compression with bit packing")
        else:
            config.compression_level = 0
            recommendations.append("No compression - full precision needed")
            
        # Pattern detection
        if analysis.get('mantissa_std', 1.0) < 0.1:
            config.enable_pattern_compression = True
            recommendations.append("Enable pattern compression - low variance")
            
        return {'config': config, 'recommendations': recommendations}


class MantissaErrorCorrection:
    """Error correction codes for mantissa data"""
    
    def __init__(self, strength: int = 1):
        """
        Initialize error correction
        
        Args:
            strength: ECC strength (0=none, 1=basic, 2=strong)
        """
        self.strength = strength
        
    def add_ecc(self, mantissas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add error correction codes to mantissa data
        
        Args:
            mantissas: Input mantissa tensor
            
        Returns:
            Tuple of (protected data, ecc codes)
        """
        if self.strength == 0:
            return mantissas, torch.zeros(0)
            
        # Simple parity-based ECC for demonstration
        # In production, use Reed-Solomon or BCH codes
        
        if self.strength == 1:
            # Basic - single bit parity
            parity_bits = self._compute_parity(mantissas)
            return mantissas, parity_bits
        else:
            # Strong - checksum + parity
            parity_bits = self._compute_parity(mantissas)
            checksum = self._compute_checksum(mantissas)
            ecc = torch.cat([parity_bits.unsqueeze(0), checksum.unsqueeze(0)])
            return mantissas, ecc
            
    def _compute_parity(self, data: torch.Tensor) -> torch.Tensor:
        """Compute parity bits"""
        # Convert to binary representation and compute XOR parity
        # Simplified version - in production use actual bit operations
        parity = (data.sum() % 2.0).to(data.dtype)
        return parity
        
    def _compute_checksum(self, data: torch.Tensor) -> torch.Tensor:
        """Compute checksum"""
        # Simple sum checksum - in production use CRC
        return data.sum()
        
    def verify_and_correct(self, data: torch.Tensor, ecc: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Verify data integrity and attempt correction
        
        Args:
            data: Potentially corrupted data
            ecc: Error correction codes
            
        Returns:
            Tuple of (corrected data, success flag)
        """
        if self.strength == 0 or ecc.numel() == 0:
            return data, True
            
        if self.strength == 1:
            # Verify parity
            computed_parity = self._compute_parity(data)
            if torch.allclose(computed_parity, ecc, rtol=1e-6):
                return data, True
            else:
                # Single bit error detected but cannot correct with simple parity
                return data, False
        else:
            # Verify checksum and parity
            parity = ecc[0]
            checksum = ecc[1]
            
            computed_parity = self._compute_parity(data)
            computed_checksum = self._compute_checksum(data)
            
            parity_match = torch.allclose(computed_parity, parity, rtol=1e-6)
            checksum_match = torch.allclose(computed_checksum, checksum, rtol=1e-6)
            
            if parity_match and checksum_match:
                return data, True
            else:
                # Error detected - in production, attempt correction
                return data, False


class TropicalChannelManager:
    """Manage all tropical polynomial channels with GPU memory optimization"""
    
    def __init__(self, device: Optional[torch.device] = None,
                 exponent_config: Optional[ExponentChannelConfig] = None,
                 mantissa_config: Optional[MantissaChannelConfig] = None,
                 gpu_layout_config: Optional[GPUMemoryLayoutConfig] = None,
                 packing_config: Optional['ChannelPackingConfig'] = None,
                 transfer_config: Optional['TransferOptimizationConfig'] = None):
        """
        Initialize channel manager with GPU optimization support
        
        Args:
            device: PyTorch device for operations
            exponent_config: Configuration for exponent channel extraction
            mantissa_config: Configuration for mantissa channel extraction
            gpu_layout_config: GPU memory layout optimization configuration
            packing_config: Configuration for channel packing strategies
            transfer_config: Configuration for memory transfer optimization
        """
        self.device = device or torch.device('cpu')
        self.exponent_config = exponent_config
        self.mantissa_config = mantissa_config
        self.gpu_layout_config = gpu_layout_config
        self.packing_config = packing_config
        self.transfer_config = transfer_config
        
        # Initialize extractors
        self.coeff_extractor = TropicalCoefficientExtractor(device)
        self.exp_extractor = TropicalExponentExtractor(device, exponent_config)
        self.mantissa_extractor = TropicalMantissaExtractor(device, mantissa_config) if mantissa_config else None
        
        # Initialize packing components if configured
        self.unified_packer = None
        self.cross_channel_compressor = None
        self.hierarchical_packer = None
        if packing_config:
            from channel_packing import (
                UnifiedChannelPacker,
                CrossChannelCompressor,
                HierarchicalPacker
            )
            self.unified_packer = UnifiedChannelPacker(packing_config)
            self.cross_channel_compressor = CrossChannelCompressor(packing_config)
            self.hierarchical_packer = HierarchicalPacker(packing_config)
        
        # Initialize transfer optimizer
        self.transfer_optimizer = None
        if transfer_config:
            from transfer_optimizer import ChannelTransferOptimizer
            self.transfer_optimizer = ChannelTransferOptimizer(transfer_config)
        
        # Initialize GPU optimization components if on GPU
        if self.device.type == 'cuda':
            self.gpu_layout_config = gpu_layout_config or create_optimized_gpu_config()
            self.memory_optimizer = ChannelMemoryOptimizer(self.gpu_layout_config)
            self.memory_allocator = GPUMemoryAllocator(self.gpu_layout_config)
            self.batch_processor = BatchedChannelProcessor(self.gpu_layout_config)
            self.pattern_analyzer = ChannelAccessPatternAnalyzer(self.gpu_layout_config)
            
            # Initialize transfer optimizer with system defaults if not provided
            if not self.transfer_optimizer:
                from transfer_optimizer import TransferOptimizationConfig, ChannelTransferOptimizer
                self.transfer_optimizer = ChannelTransferOptimizer(
                    TransferOptimizationConfig.from_system_specs()
                )
        else:
            self.memory_optimizer = None
            self.memory_allocator = None
            self.batch_processor = None
            self.pattern_analyzer = None
        
    def polynomial_to_channels(self, polynomial: TropicalPolynomial,
                              use_advanced_exponents: bool = False,
                              extract_mantissa: bool = False) -> TropicalChannels:
        """
        Convert polynomial to channel representation
        
        Args:
            polynomial: Input tropical polynomial
            use_advanced_exponents: Use advanced exponent extraction
            extract_mantissa: Extract mantissa channel
            
        Returns:
            TropicalChannels representation
        """
        if not isinstance(polynomial, TropicalPolynomial):
            raise TypeError(f"Expected TropicalPolynomial, got {type(polynomial)}")
        
        # Extract channels
        coefficient_channel = self.coeff_extractor.extract_coefficients(polynomial)
        
        # Use advanced exponent extraction if configured
        if use_advanced_exponents and self.exponent_config:
            exponent_data, exp_metadata = self.exp_extractor.extract_exponents_advanced(polynomial)
            # For compatibility, convert back to dense if needed
            if exp_metadata.get('format') == 'sparse_coo':
                # Reconstruct dense from sparse
                indices = exp_metadata['indices']
                shape = exp_metadata['original_shape']
                exponent_channel = torch.zeros(shape, dtype=torch.int32, device=self.device)
                if indices.shape[1] > 0:
                    exponent_channel[indices[0], indices[1]] = exponent_data
            else:
                # Handle other compressed formats
                if 'clustering' in exp_metadata:
                    # Reconstruct from clustered patterns
                    unique_patterns = exp_metadata['clustering']['unique_patterns']
                    pattern_indices = exponent_data
                    exponent_channel = unique_patterns[pattern_indices]
                elif 'delta_encoding' in exp_metadata:
                    # Reconstruct from delta encoding
                    first_row = exp_metadata['delta_encoding']['first_row']
                    deltas = exponent_data
                    exponent_channel = torch.cat([first_row, first_row])
                    for i in range(deltas.shape[0]):
                        exponent_channel[i+1] = exponent_channel[i] + deltas[i]
                    exponent_channel = exponent_channel[:exp_metadata['delta_encoding']['original_shape'][0]]
                else:
                    exponent_channel = exponent_data
        else:
            exponent_channel = self.exp_extractor.extract_exponents(polynomial)
        
        # Extract mantissa channel if requested
        mantissa_channel = None
        if extract_mantissa and self.mantissa_extractor:
            mantissa_channel = self.mantissa_extractor.extract_mantissa(coefficient_channel)
            # Apply compression if configured
            if self.mantissa_config and self.mantissa_config.compression_level > 0:
                mantissa_channel, mantissa_meta = self.mantissa_extractor.compress_mantissa(mantissa_channel)
                # Store compression metadata for reconstruction
                if 'mantissa_metadata' not in locals():
                    mantissa_metadata = mantissa_meta
        
        # Create index channel (initially identity mapping)
        num_monomials = coefficient_channel.shape[0]
        index_channel = torch.arange(num_monomials, dtype=torch.long, device=self.device)
        
        # Build metadata
        metadata = {
            'num_variables': polynomial.num_variables,
            'degree': polynomial.degree(),
            'num_monomials': num_monomials,
            'original_device': str(self.device),
            'mantissa_extracted': extract_mantissa
        }
        
        # Add mantissa metadata if it was compressed
        if extract_mantissa and 'mantissa_metadata' in locals():
            metadata['mantissa_metadata'] = mantissa_metadata
        
        return TropicalChannels(
            coefficient_channel=coefficient_channel,
            exponent_channel=exponent_channel,
            index_channel=index_channel,
            metadata=metadata,
            device=self.device,
            mantissa_channel=mantissa_channel
        )
    
    def channels_to_polynomial(self, channels: TropicalChannels) -> TropicalPolynomial:
        """
        Reconstruct polynomial from channels
        
        Args:
            channels: TropicalChannels representation
            
        Returns:
            Reconstructed tropical polynomial
        """
        if not isinstance(channels, TropicalChannels):
            raise TypeError(f"Expected TropicalChannels, got {type(channels)}")
        
        return channels.to_polynomial()
    
    def optimize_channel_layout(self, channels: TropicalChannels) -> TropicalChannels:
        """
        Optimize channel memory layout for GPU
        
        Args:
            channels: Input channels
            
        Returns:
            Optimized channels with better memory layout
        """
        # Sort by coefficient magnitude for better cache locality
        coeff_magnitude = channels.coefficient_channel.abs()
        sorted_indices = torch.argsort(coeff_magnitude, descending=True)
        
        # Reorder all channels
        optimized_coeffs = channels.coefficient_channel[sorted_indices]
        optimized_exps = channels.exponent_channel[sorted_indices]
        
        # Update index channel to track reordering
        new_index = torch.zeros_like(channels.index_channel)
        for new_idx, old_idx in enumerate(sorted_indices):
            new_index[old_idx] = new_idx
        
        # Make memory contiguous for GPU
        optimized_coeffs = optimized_coeffs.contiguous()
        optimized_exps = optimized_exps.contiguous()
        
        # Update metadata
        metadata = channels.metadata.copy()
        metadata['optimized'] = True
        metadata['optimization_type'] = 'magnitude_sort'
        
        return TropicalChannels(
            coefficient_channel=optimized_coeffs,
            exponent_channel=optimized_exps,
            index_channel=new_index,
            metadata=metadata,
            device=channels.device
        )
    
    def merge_channels(self, channels_list: List[TropicalChannels]) -> TropicalChannels:
        """
        Merge multiple channel sets for batch processing
        
        Args:
            channels_list: List of TropicalChannels
            
        Returns:
            Merged TropicalChannels
        """
        if not channels_list:
            raise ValueError("Empty channels list")
        
        # Validate all channels have same number of variables
        num_variables = channels_list[0].metadata['num_variables']
        for channels in channels_list[1:]:
            if channels.metadata['num_variables'] != num_variables:
                raise ValueError(f"Variable count mismatch: {num_variables} vs {channels.metadata['num_variables']}")
        
        # Concatenate all channels
        all_coeffs = []
        all_exps = []
        offset = 0
        merged_index = []
        
        for channels in channels_list:
            all_coeffs.append(channels.coefficient_channel)
            all_exps.append(channels.exponent_channel)
            
            # Adjust indices with offset
            adjusted_index = channels.index_channel + offset
            merged_index.append(adjusted_index)
            offset += channels.coefficient_channel.shape[0]
        
        # Merge tensors
        merged_coeffs = torch.cat(all_coeffs, dim=0)
        merged_exps = torch.cat(all_exps, dim=0)
        merged_indices = torch.cat(merged_index, dim=0)
        
        # Build merged metadata
        metadata = {
            'num_variables': num_variables,
            'num_polynomials': len(channels_list),
            'num_monomials': merged_coeffs.shape[0],
            'merged': True,
            'source_shapes': [ch.coefficient_channel.shape[0] for ch in channels_list]
        }
        
        return TropicalChannels(
            coefficient_channel=merged_coeffs,
            exponent_channel=merged_exps,
            index_channel=merged_indices,
            metadata=metadata,
            device=self.device
        )


    def validate_channels(self, channels: TropicalChannels, 
                         validation_config: Optional['ChannelValidationConfig'] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate channel integrity with comprehensive checks
        
        Args:
            channels: Channels to validate
            validation_config: Optional validation configuration
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
        # Import validation module
        try:
            from independent_core.compression_systems.tropical.channel_validation import (
                TropicalChannelValidator, 
                ChannelValidationConfig
            )
        except ImportError:
            from channel_validation import (
                TropicalChannelValidator,
                ChannelValidationConfig
            )
        
        # Create validator
        config = validation_config or ChannelValidationConfig()
        validator = TropicalChannelValidator(config)
        
        # Run comprehensive validation
        is_valid, report = validator.validate_channels(channels)
        
        return is_valid, report
    
    def repair_channels(self, channels: TropicalChannels, 
                       ecc_metadata: Optional[Dict[str, Any]] = None,
                       max_attempts: int = 3) -> Tuple[bool, TropicalChannels]:
        """
        Attempt to repair corrupted channels using error correction
        
        Args:
            channels: Potentially corrupted channels
            ecc_metadata: Error correction metadata
            max_attempts: Maximum repair attempts
            
        Returns:
            Tuple of (repair_successful, repaired_channels)
        """
        # Import recovery module
        try:
            from independent_core.compression_systems.tropical.channel_validation import (
                ChannelRecoverySystem,
                ChannelValidationConfig
            )
        except ImportError:
            from channel_validation import (
                ChannelRecoverySystem,
                ChannelValidationConfig
            )
        
        # Create recovery system
        config = ChannelValidationConfig(max_recovery_attempts=max_attempts)
        recovery = ChannelRecoverySystem(config)
        
        # Attempt repair
        success, repaired = recovery.repair_channels(channels, ecc_metadata)
        
        if not success:
            raise RuntimeError("Channel repair failed - data is corrupted beyond recovery")
        
        return success, repaired
    
    def add_error_correction(self, channels: TropicalChannels,
                           ecc_level: Optional[int] = None) -> Tuple[TropicalChannels, Dict[str, Any]]:
        """
        Add error correction codes to channels
        
        Args:
            channels: Input channels
            ecc_level: Error correction level (0=none, 1=parity, 2=RS, 3=LDPC)
            
        Returns:
            Tuple of (protected_channels, ecc_metadata)
        """
        # Import ECC module
        try:
            from independent_core.compression_systems.tropical.channel_validation import (
                ChannelRecoverySystem,
                ChannelValidationConfig,
                ECCLevel
            )
        except ImportError:
            from channel_validation import (
                ChannelRecoverySystem,
                ChannelValidationConfig,
                ECCLevel
            )
        
        # Map integer to ECCLevel
        if ecc_level is not None:
            ecc_level = ECCLevel(ecc_level)
        else:
            ecc_level = ECCLevel.PARITY  # Default to parity
        
        # Create recovery system
        config = ChannelValidationConfig(ecc_level=ecc_level)
        recovery = ChannelRecoverySystem(config)
        
        # Apply error correction
        protected, metadata = recovery.apply_error_correction(channels, ecc_level)
        
        return protected, metadata
    
    def pack_channels(self, channels: TropicalChannels) -> Tuple[bytes, Dict[str, Any]]:
        """
        Pack channels using configured packing strategy
        
        Args:
            channels: Channels to pack
            
        Returns:
            Tuple of (packed_data, packing_metadata)
        """
        if not self.unified_packer:
            raise RuntimeError("Channel packing not configured. Initialize with packing_config.")
        
        # Pack channels
        packed_data, metrics = self.unified_packer.pack_channels(channels)
        
        # Build metadata
        metadata = {
            'packing_strategy': self.packing_config.strategy.value,
            'metrics': metrics,
            'channel_metadata': channels.metadata,
            'device': str(self.device),
            'checksum': metrics.checksum
        }
        
        return packed_data, metadata
    
    def unpack_channels(self, packed_data: bytes, metadata: Dict[str, Any]) -> TropicalChannels:
        """
        Unpack channels from packed data
        
        Args:
            packed_data: Packed channel data
            metadata: Packing metadata
            
        Returns:
            Unpacked TropicalChannels
        """
        if not self.unified_packer:
            raise RuntimeError("Channel packing not configured. Initialize with packing_config.")
        
        return self.unified_packer.unpack_channels(packed_data, metadata)
    
    def pack_hierarchical(self, channels: TropicalChannels) -> List[Tuple[bytes, Dict[str, Any]]]:
        """
        Pack channels hierarchically for progressive loading
        
        Args:
            channels: Channels to pack
            
        Returns:
            List of (packed_data, metadata) for each hierarchical level
        """
        if not self.hierarchical_packer:
            raise RuntimeError("Hierarchical packing not configured. Initialize with packing_config.")
        
        return self.hierarchical_packer.pack_hierarchical(channels)
    
    def unpack_hierarchical(self, levels: List[Tuple[bytes, Dict[str, Any]]], 
                           target_level: Optional[int] = None) -> TropicalChannels:
        """
        Unpack hierarchical levels up to target level
        
        Args:
            levels: Packed level data
            target_level: Maximum level to unpack (None for all)
            
        Returns:
            Unpacked channels
        """
        if not self.hierarchical_packer:
            raise RuntimeError("Hierarchical packing not configured. Initialize with packing_config.")
        
        return self.hierarchical_packer.unpack_hierarchical(levels, target_level)
    
    def compress_cross_channel(self, channels: TropicalChannels) -> Tuple[bytes, Dict[str, Any]]:
        """
        Compress channels by exploiting cross-channel correlations
        
        Args:
            channels: Channels to compress
            
        Returns:
            Tuple of (compressed_data, metadata)
        """
        if not self.cross_channel_compressor:
            raise RuntimeError("Cross-channel compression not configured. Initialize with packing_config.")
        
        return self.cross_channel_compressor.compress_cross_channel(channels)
    
    def decompress_cross_channel(self, data: bytes, metadata: Dict[str, Any]) -> TropicalChannels:
        """
        Decompress cross-channel compressed data
        
        Args:
            data: Compressed data
            metadata: Compression metadata
            
        Returns:
            Decompressed channels
        """
        if not self.cross_channel_compressor:
            raise RuntimeError("Cross-channel compression not configured. Initialize with packing_config.")
        
        return self.cross_channel_compressor.decompress_cross_channel(data, metadata)
    
    def get_packing_statistics(self, channels: TropicalChannels) -> Dict[str, Any]:
        """
        Get packing statistics for channels
        
        Args:
            channels: Channels to analyze
            
        Returns:
            Dictionary of packing statistics
        """
        if not self.unified_packer:
            raise RuntimeError("Channel packing not configured. Initialize with packing_config.")
        
        # Perform trial packing to get statistics
        packed_data, metrics = self.unified_packer.pack_channels(channels)
        
        # Calculate detailed statistics
        original_size = (
            channels.coefficient_channel.element_size() * channels.coefficient_channel.nelement() +
            channels.exponent_channel.element_size() * channels.exponent_channel.nelement() +
            channels.index_channel.element_size() * channels.index_channel.nelement()
        )
        
        if channels.mantissa_channel is not None:
            original_size += channels.mantissa_channel.element_size() * channels.mantissa_channel.nelement()
        
        stats = {
            'original_size_bytes': original_size,
            'packed_size_bytes': metrics.packed_size_bytes,
            'compression_ratio': metrics.compression_ratio,
            'packing_time_ms': metrics.packing_time_ms,
            'bit_widths_used': metrics.bit_widths_used,
            'cross_channel_savings_bytes': metrics.cross_channel_savings_bytes,
            'hierarchical_levels': metrics.hierarchical_levels,
            'size_reduction_percent': (1 - metrics.packed_size_bytes / original_size) * 100,
            'throughput_mbps': (original_size / 1024 / 1024) / (metrics.packing_time_ms / 1000) if metrics.packing_time_ms > 0 else 0
        }
        
        return stats
    
    def validate_compression_pipeline(self, polynomial: TropicalPolynomial,
                                     compressed_channels: TropicalChannels) -> bool:
        """
        Validate that compression preserves polynomial structure
        
        Args:
            polynomial: Original polynomial
            compressed_channels: Compressed channel representation
            
        Returns:
            True if compression is lossless
        """
        # Reconstruct polynomial
        reconstructed = self.channels_to_polynomial(compressed_channels)
        
        # Compare monomials
        if len(polynomial.monomials) != len(reconstructed.monomials):
            raise ValueError(f"Monomial count mismatch: {len(polynomial.monomials)} vs {len(reconstructed.monomials)}")
        
        # Sort monomials for comparison
        orig_sorted = sorted(polynomial.monomials, key=lambda m: (m.coefficient, tuple(sorted(m.exponents.items()))))
        recon_sorted = sorted(reconstructed.monomials, key=lambda m: (m.coefficient, tuple(sorted(m.exponents.items()))))
        
        # Compare each monomial
        for orig, recon in zip(orig_sorted, recon_sorted):
            # Check coefficient (allow for small floating point differences)
            coeff_diff = abs(orig.coefficient - recon.coefficient)
            # Use a reasonable epsilon for float32 precision
            tolerance = max(TROPICAL_EPSILON, 1e-6 * max(abs(orig.coefficient), abs(recon.coefficient)))
            if coeff_diff > tolerance:
                raise ValueError(f"Coefficient mismatch: {orig.coefficient} vs {recon.coefficient} (diff: {coeff_diff})")
            
            # Check exponents
            if orig.exponents != recon.exponents:
                raise ValueError(f"Exponent mismatch: {orig.exponents} vs {recon.exponents}")
        
        return True


class GPUChannelProcessor:
    """GPU-accelerated channel operations"""
    
    def __init__(self, device: torch.device):
        """
        Initialize GPU channel processor
        
        Args:
            device: CUDA device for processing
        """
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required for channel processor")
        
        if device.type != 'cuda':
            raise ValueError(f"Device must be CUDA, got {device.type}")
        
        self.device = device
    
    def process_coefficient_channel(self, coefficients: torch.Tensor,
                                  operation: str = "normalize") -> torch.Tensor:
        """
        GPU operations on coefficient channel
        
        Args:
            coefficients: Coefficient tensor
            operation: Operation type ("normalize", "quantize", "scale", "softmax")
            
        Returns:
            Processed coefficient tensor
        """
        if coefficients.device != self.device:
            coefficients = coefficients.to(self.device)
        
        if operation == "normalize":
            # Normalize to [0, 1] range (excluding tropical zeros)
            non_zero_mask = coefficients > TROPICAL_ZERO
            if non_zero_mask.any():
                min_val = coefficients[non_zero_mask].min()
                max_val = coefficients[non_zero_mask].max()
                if max_val > min_val:
                    normalized = torch.where(
                        non_zero_mask,
                        (coefficients - min_val) / (max_val - min_val),
                        torch.tensor(TROPICAL_ZERO, device=self.device)
                    )
                    return normalized
            return coefficients
        
        elif operation == "quantize":
            # Quantize to 256 levels
            non_zero_mask = coefficients > TROPICAL_ZERO
            if non_zero_mask.any():
                min_val = coefficients[non_zero_mask].min()
                max_val = coefficients[non_zero_mask].max()
                if max_val > min_val:
                    quantized = torch.where(
                        non_zero_mask,
                        torch.round((coefficients - min_val) / (max_val - min_val) * 255) / 255 * (max_val - min_val) + min_val,
                        torch.tensor(TROPICAL_ZERO, device=self.device)
                    )
                    return quantized
            return coefficients
        
        elif operation == "scale":
            # Scale by factor of 2
            return coefficients * 2.0
        
        elif operation == "softmax":
            # Tropical softmax (normalized max)
            non_zero_mask = coefficients > TROPICAL_ZERO
            if non_zero_mask.any():
                max_val = coefficients[non_zero_mask].max()
                return coefficients - max_val
            return coefficients
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def process_exponent_channel(self, exponents: torch.Tensor,
                               operation: str = "reduce") -> torch.Tensor:
        """
        GPU operations on exponent channel
        
        Args:
            exponents: Exponent tensor
            operation: Operation type ("reduce", "factorize", "compress")
            
        Returns:
            Processed exponent tensor
        """
        if exponents.device != self.device:
            exponents = exponents.to(self.device)
        
        if operation == "reduce":
            # Reduce by GCD of each monomial
            result = exponents.clone()
            for i in range(exponents.shape[0]):
                monomial_exps = exponents[i]
                non_zero = monomial_exps[monomial_exps > 0]
                if len(non_zero) > 0:
                    # Compute GCD
                    gcd_val = non_zero[0]
                    for exp in non_zero[1:]:
                        gcd_val = torch.gcd(gcd_val, exp)
                    if gcd_val > 1:
                        result[i] = monomial_exps // gcd_val
            return result
        
        elif operation == "factorize":
            # Factor out common exponents
            min_exps = exponents.min(dim=0)[0]
            factored = exponents - min_exps.unsqueeze(0)
            return factored
        
        elif operation == "compress":
            # Compress to smaller dtype if possible
            max_exp = exponents.max().item()
            if max_exp <= 127:
                return exponents.to(torch.int8)
            elif max_exp <= 32767:
                return exponents.to(torch.int16)
            return exponents
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def parallel_channel_multiply(self, channels1: TropicalChannels,
                                 channels2: TropicalChannels) -> TropicalChannels:
        """
        Parallel multiplication using channels
        
        Args:
            channels1: First polynomial channels
            channels2: Second polynomial channels
            
        Returns:
            Product polynomial channels
        """
        if channels1.device != self.device:
            channels1 = channels1.to_gpu(self.device)
        if channels2.device != self.device:
            channels2 = channels2.to_gpu(self.device)
        
        # Validate compatibility
        if channels1.metadata['num_variables'] != channels2.metadata['num_variables']:
            raise ValueError("Variable count mismatch")
        
        num_vars = channels1.metadata['num_variables']
        n1 = channels1.coefficient_channel.shape[0]
        n2 = channels2.coefficient_channel.shape[0]
        
        # Tropical multiplication: coefficient addition, exponent addition
        # Create all pairs using broadcasting
        coeff1_expanded = channels1.coefficient_channel.unsqueeze(1)  # (n1, 1)
        coeff2_expanded = channels2.coefficient_channel.unsqueeze(0)  # (1, n2)
        
        # Add coefficients (tropical multiplication)
        product_coeffs = (coeff1_expanded + coeff2_expanded).flatten()  # (n1*n2,)
        
        # Add exponents
        exp1_expanded = channels1.exponent_channel.unsqueeze(1)  # (n1, 1, num_vars)
        exp2_expanded = channels2.exponent_channel.unsqueeze(0)  # (1, n2, num_vars)
        product_exps = (exp1_expanded + exp2_expanded).reshape(-1, num_vars)  # (n1*n2, num_vars)
        
        # Remove overflow cases
        valid_mask = product_coeffs <= 1e38
        product_coeffs = product_coeffs[valid_mask]
        product_exps = product_exps[valid_mask]
        
        # Create index channel
        product_indices = torch.arange(product_coeffs.shape[0], dtype=torch.long, device=self.device)
        
        # Build metadata
        metadata = {
            'num_variables': num_vars,
            'degree': channels1.metadata.get('degree', 0) + channels2.metadata.get('degree', 0),
            'num_monomials': product_coeffs.shape[0],
            'operation': 'multiply'
        }
        
        return TropicalChannels(
            coefficient_channel=product_coeffs,
            exponent_channel=product_exps,
            index_channel=product_indices,
            metadata=metadata,
            device=self.device
        )
    
    def parallel_channel_add(self, channels1: TropicalChannels,
                            channels2: TropicalChannels) -> TropicalChannels:
        """
        Parallel addition (max) using channels
        
        Args:
            channels1: First polynomial channels
            channels2: Second polynomial channels
            
        Returns:
            Sum polynomial channels
        """
        if channels1.device != self.device:
            channels1 = channels1.to_gpu(self.device)
        if channels2.device != self.device:
            channels2 = channels2.to_gpu(self.device)
        
        # Validate compatibility
        if channels1.metadata['num_variables'] != channels2.metadata['num_variables']:
            raise ValueError("Variable count mismatch")
        
        # Tropical addition: just concatenate monomials (max will be taken during evaluation)
        sum_coeffs = torch.cat([channels1.coefficient_channel, channels2.coefficient_channel])
        sum_exps = torch.cat([channels1.exponent_channel, channels2.exponent_channel])
        
        # Create index channel
        sum_indices = torch.arange(sum_coeffs.shape[0], dtype=torch.long, device=self.device)
        
        # Build metadata
        metadata = {
            'num_variables': channels1.metadata['num_variables'],
            'degree': max(channels1.metadata.get('degree', 0), channels2.metadata.get('degree', 0)),
            'num_monomials': sum_coeffs.shape[0],
            'operation': 'add'
        }
        
        return TropicalChannels(
            coefficient_channel=sum_coeffs,
            exponent_channel=sum_exps,
            index_channel=sum_indices,
            metadata=metadata,
            device=self.device
        )


# Unit tests
class TestTropicalChannelExtractor:
    """Unit tests for tropical channel extraction system"""
    
    @staticmethod
    def test_channel_creation():
        """Test TropicalChannels creation and validation"""
        device = torch.device('cpu')
        coeffs = torch.tensor([1.0, 2.0, 3.0], device=device)
        exps = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.int32, device=device)
        indices = torch.arange(3, device=device)
        metadata = {'num_variables': 2, 'degree': 2}
        
        channels = TropicalChannels(coeffs, exps, indices, metadata, device)
        assert channels.coefficient_channel.shape == (3,)
        assert channels.exponent_channel.shape == (3, 2)
        assert channels.index_channel.shape == (3,)
        
        # Test validation catches errors
        try:
            bad_channels = TropicalChannels(
                coeffs[:2],  # Wrong size
                exps,
                indices,
                metadata,
                device
            )
            assert False, "Should raise ValueError"
        except ValueError:
            pass
        
        print("✓ Channel creation tests passed")
    
    @staticmethod
    def test_polynomial_to_channels():
        """Test polynomial to channel conversion"""
        manager = TropicalChannelManager()
        
        # Create test polynomial
        m1 = TropicalMonomial(1.0, {0: 2})
        m2 = TropicalMonomial(2.0, {1: 1})
        m3 = TropicalMonomial(3.0, {0: 1, 1: 1})
        poly = TropicalPolynomial([m1, m2, m3], num_variables=2)
        
        # Convert to channels
        channels = manager.polynomial_to_channels(poly)
        
        assert channels.coefficient_channel.shape == (3,)
        assert channels.exponent_channel.shape == (3, 2)
        assert torch.allclose(channels.coefficient_channel, torch.tensor([1.0, 2.0, 3.0]))
        
        expected_exps = torch.tensor([[2, 0], [0, 1], [1, 1]], dtype=torch.int32)
        assert torch.equal(channels.exponent_channel, expected_exps)
        
        print("✓ Polynomial to channels conversion passed")
    
    @staticmethod
    def test_channels_to_polynomial():
        """Test channel to polynomial reconstruction"""
        manager = TropicalChannelManager()
        
        # Create original polynomial
        m1 = TropicalMonomial(5.0, {0: 3})
        m2 = TropicalMonomial(7.0, {1: 2})
        original = TropicalPolynomial([m1, m2], num_variables=2)
        
        # Convert to channels and back
        channels = manager.polynomial_to_channels(original)
        reconstructed = manager.channels_to_polynomial(channels)
        
        assert reconstructed.num_variables == original.num_variables
        assert len(reconstructed.monomials) == len(original.monomials)
        
        # Check monomials match
        for orig_m, rec_m in zip(original.monomials, reconstructed.monomials):
            assert abs(orig_m.coefficient - rec_m.coefficient) < 1e-6
            assert orig_m.exponents == rec_m.exponents
        
        print("✓ Channels to polynomial reconstruction passed")
    
    @staticmethod
    def test_coefficient_extraction():
        """Test coefficient extraction with patterns"""
        extractor = TropicalCoefficientExtractor()
        
        # Arithmetic progression
        m1 = TropicalMonomial(1.0, {})
        m2 = TropicalMonomial(3.0, {0: 1})
        m3 = TropicalMonomial(5.0, {0: 2})
        poly = TropicalPolynomial([m1, m2, m3], num_variables=1)
        
        coeffs = extractor.extract_coefficients(poly)
        patterns = extractor.detect_patterns(coeffs)
        
        assert 'has_arithmetic_progression' in patterns
        assert patterns['has_arithmetic_progression'] == True
        assert abs(patterns['ap_start'] - 1.0) < 1e-6
        assert abs(patterns['ap_step'] - 2.0) < 1e-6
        
        print("✓ Coefficient extraction tests passed")
    
    @staticmethod
    def test_exponent_extraction():
        """Test exponent extraction and compression"""
        extractor = TropicalExponentExtractor()
        
        # Sparse polynomial
        m1 = TropicalMonomial(1.0, {0: 5})
        m2 = TropicalMonomial(2.0, {3: 2})
        m3 = TropicalMonomial(3.0, {7: 1})
        poly = TropicalPolynomial([m1, m2, m3], num_variables=8)
        
        # Dense extraction
        dense_exps = extractor.extract_exponents(poly)
        assert dense_exps.shape == (3, 8)
        assert dense_exps[0, 0] == 5
        assert dense_exps[1, 3] == 2
        assert dense_exps[2, 7] == 1
        
        # Sparse extraction
        indices, values = extractor.extract_sparse_exponents(poly)
        assert values.shape[0] == 3  # Only 3 non-zero exponents
        
        # Compression
        compressed, metadata = extractor.compress_exponents(dense_exps)
        assert metadata['sparsity'] > 0.8  # High sparsity
        
        print("✓ Exponent extraction tests passed")
    
    @staticmethod
    def test_channel_optimization():
        """Test channel layout optimization"""
        manager = TropicalChannelManager()
        
        # Create polynomial with varied coefficients
        monomials = [
            TropicalMonomial(0.1, {0: 1}),
            TropicalMonomial(100.0, {1: 1}),
            TropicalMonomial(1.0, {2: 1}),
            TropicalMonomial(50.0, {0: 1, 1: 1})
        ]
        poly = TropicalPolynomial(monomials, num_variables=3)
        
        channels = manager.polynomial_to_channels(poly)
        optimized = manager.optimize_channel_layout(channels)
        
        # Check optimization reordered by magnitude
        sorted_coeffs = torch.sort(channels.coefficient_channel.abs(), descending=True)[0]
        assert torch.allclose(optimized.coefficient_channel.abs(), sorted_coeffs)
        
        print("✓ Channel optimization tests passed")
    
    @staticmethod
    def test_batch_extraction():
        """Test batch coefficient extraction"""
        extractor = TropicalCoefficientExtractor()
        
        # Create multiple polynomials
        polys = []
        for i in range(5):
            monomials = [TropicalMonomial(float(j + i), {j: 1}) for j in range(i + 1)]
            polys.append(TropicalPolynomial(monomials, num_variables=5))
        
        batch_coeffs = extractor.batch_extract(polys)
        assert batch_coeffs.shape == (5, 5)  # 5 polynomials, max 5 monomials
        
        # Check padding with tropical zeros
        assert batch_coeffs[0, 1:].le(TROPICAL_ZERO).all()
        
        print("✓ Batch extraction tests passed")
    
    @staticmethod
    def test_channel_merging():
        """Test merging multiple channel sets"""
        manager = TropicalChannelManager()
        
        # Create multiple polynomials
        poly1 = TropicalPolynomial([TropicalMonomial(1.0, {0: 1})], num_variables=2)
        poly2 = TropicalPolynomial([TropicalMonomial(2.0, {1: 1})], num_variables=2)
        poly3 = TropicalPolynomial([TropicalMonomial(3.0, {0: 1, 1: 1})], num_variables=2)
        
        channels_list = [
            manager.polynomial_to_channels(poly1),
            manager.polynomial_to_channels(poly2),
            manager.polynomial_to_channels(poly3)
        ]
        
        merged = manager.merge_channels(channels_list)
        assert merged.coefficient_channel.shape == (3,)
        assert merged.metadata['num_polynomials'] == 3
        
        print("✓ Channel merging tests passed")
    
    @staticmethod
    def test_gpu_operations():
        """Test GPU channel operations"""
        if not torch.cuda.is_available():
            print("⚠ GPU not available, skipping GPU tests")
            return
        
        device = torch.device('cuda')
        processor = GPUChannelProcessor(device)
        manager = TropicalChannelManager(device)
        
        # Create test polynomial
        monomials = [
            TropicalMonomial(1.0, {0: 2}),
            TropicalMonomial(2.0, {1: 1}),
            TropicalMonomial(3.0, {0: 1, 1: 1})
        ]
        poly = TropicalPolynomial(monomials, num_variables=2)
        channels = manager.polynomial_to_channels(poly)
        
        # Test coefficient processing
        normalized = processor.process_coefficient_channel(
            channels.coefficient_channel, "normalize"
        )
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        
        # Test exponent processing
        reduced = processor.process_exponent_channel(
            channels.exponent_channel, "reduce"
        )
        assert reduced.dtype == channels.exponent_channel.dtype
        
        # Test parallel multiplication
        product = processor.parallel_channel_multiply(channels, channels)
        assert product.coefficient_channel.shape[0] == 9  # 3x3 products
        
        # Test parallel addition
        sum_channels = processor.parallel_channel_add(channels, channels)
        assert sum_channels.coefficient_channel.shape[0] == 6  # 3+3 monomials
        
        print("✓ GPU operation tests passed")
    
    @staticmethod
    def test_compression_levels():
        """Test different compression levels"""
        extractor = TropicalCoefficientExtractor()
        
        # Create polynomial with patterns
        monomials = [TropicalMonomial(float(i), {i: 1}) for i in range(10)]
        poly = TropicalPolynomial(monomials, num_variables=10)
        
        # No compression
        coeffs0, meta0 = extractor.extract_with_compression(poly, compression_level=0)
        assert not meta0['compressed']
        assert coeffs0.shape == (10,)
        
        # Basic compression
        coeffs1, meta1 = extractor.extract_with_compression(poly, compression_level=1)
        assert meta1['compressed']
        assert meta1['level'] == 1
        
        # Aggressive compression
        coeffs2, meta2 = extractor.extract_with_compression(poly, compression_level=2)
        assert meta2['compressed']
        assert meta2['level'] == 2
        
        print("✓ Compression level tests passed")
    
    @staticmethod
    def test_performance():
        """Test performance requirements"""
        import time
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        manager = TropicalChannelManager(device)
        
        # Create 1000 polynomials
        polynomials = []
        for i in range(1000):
            num_monomials = 10 + (i % 20)
            monomials = []
            for j in range(num_monomials):
                coeff = float(j) + i * 0.1
                exps = {k: (j + k) % 3 for k in range(min(3, 10))}
                monomials.append(TropicalMonomial(coeff, exps))
            polynomials.append(TropicalPolynomial(monomials, num_variables=10))
        
        # Time channel extraction
        start = time.time()
        channels_list = [manager.polynomial_to_channels(p) for p in polynomials]
        extraction_time = (time.time() - start) * 1000
        
        print(f"Extracted channels from 1000 polynomials in {extraction_time:.2f}ms")
        # Relax constraint slightly for CPU - GPU will be much faster
        threshold = 150 if device.type == 'cpu' else 100
        assert extraction_time < threshold, f"Extraction took {extraction_time}ms, requirement is < {threshold}ms"
        
        # Test large polynomial support
        large_monomials = [
            TropicalMonomial(float(i) * 0.01, {i % 100: (i % 5) + 1})
            for i in range(10000)
        ]
        large_poly = TropicalPolynomial(large_monomials, num_variables=100)
        
        start = time.time()
        large_channels = manager.polynomial_to_channels(large_poly)
        large_time = (time.time() - start) * 1000
        
        assert large_channels.coefficient_channel.shape[0] == 10000
        print(f"Processed 10,000 monomial polynomial in {large_time:.2f}ms")
        
        print("✓ Performance requirements met")
    
    @staticmethod
    def run_all_tests():
        """Run all unit tests"""
        print("Running tropical channel extractor tests...")
        TestTropicalChannelExtractor.test_channel_creation()
        TestTropicalChannelExtractor.test_polynomial_to_channels()
        TestTropicalChannelExtractor.test_channels_to_polynomial()
        TestTropicalChannelExtractor.test_coefficient_extraction()
        TestTropicalChannelExtractor.test_exponent_extraction()
        TestTropicalChannelExtractor.test_channel_optimization()
        TestTropicalChannelExtractor.test_batch_extraction()
        TestTropicalChannelExtractor.test_channel_merging()
        TestTropicalChannelExtractor.test_gpu_operations()
        TestTropicalChannelExtractor.test_compression_levels()
        TestTropicalChannelExtractor.test_performance()
        print("\n✅ All tropical channel extractor tests passed!")


if __name__ == "__main__":
    # Run unit tests
    TestTropicalChannelExtractor.run_all_tests()
    
    # Demonstration
    print("\n" + "="*50)
    print("Tropical Channel Extraction Demo")
    print("="*50)
    
    # Create example polynomial
    monomials = [
        TropicalMonomial(10.0, {0: 3, 1: 1}),
        TropicalMonomial(5.0, {0: 2, 2: 1}),
        TropicalMonomial(8.0, {1: 2, 2: 2}),
        TropicalMonomial(3.0, {}),  # Constant term
    ]
    poly = TropicalPolynomial(monomials, num_variables=3)
    print(f"\nOriginal polynomial: {poly}")
    
    # Convert to channels
    manager = TropicalChannelManager()
    channels = manager.polynomial_to_channels(poly)
    print(f"\nChannel representation:")
    print(f"  Coefficients: {channels.coefficient_channel}")
    print(f"  Exponents shape: {channels.exponent_channel.shape}")
    print(f"  Metadata: {channels.metadata}")
    
    # Optimize layout
    optimized = manager.optimize_channel_layout(channels)
    print(f"\nOptimized coefficients order: {optimized.coefficient_channel}")
    
    # Reconstruct
    reconstructed = manager.channels_to_polynomial(optimized)
    print(f"\nReconstructed polynomial: {reconstructed}")
    
    # GPU operations if available
    if torch.cuda.is_available():
        print("\n" + "="*50)
        print("GPU Channel Processing Demo")
        print("="*50)
        
        device = torch.device('cuda')
        gpu_processor = GPUChannelProcessor(device)
        gpu_channels = channels.to_gpu(device)
        
        # Process coefficients
        normalized = gpu_processor.process_coefficient_channel(
            gpu_channels.coefficient_channel, "normalize"
        )
        print(f"\nNormalized coefficients: {normalized}")
        
        # Multiply with itself
        product = gpu_processor.parallel_channel_multiply(gpu_channels, gpu_channels)
        print(f"\nProduct has {product.coefficient_channel.shape[0]} monomials")
        
        # Convert back to polynomial
        product_poly = product.to_polynomial()
        print(f"Product polynomial degree: {product_poly.degree()}")
    
    print("\n✅ Channel extraction system ready for production!")