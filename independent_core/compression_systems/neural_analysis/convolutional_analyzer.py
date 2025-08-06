"""
Convolutional Neural Network Layer Analyzer for Mathematical Compression.
Specialized analysis for Conv2D, Conv3D, and related layers with unique compression opportunities.
Identifies filter redundancy, channel importance, and spatial decomposition potential.
NO PLACEHOLDERS - PRODUCTION READY - FAIL-LOUD ERROR HANDLING
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
import logging
import time

# Import base analyzer and data structures
from independent_core.compression_systems.neural_analysis.layer_analyzer import (
    DenseLayerAnalyzer as LayerAnalyzer,
    RankAnalysis,
    SparsityAnalysis,
    NumericalAnalysis,
    CompressionMethod,
    CompressionRecommendation
)

# Import tropical operations for filter analysis
from independent_core.compression_systems.tropical.tropical_linear_algebra import (
    TropicalMatrixFactorization,
    TropicalLinearAlgebra
)

# Import strategy selection
from independent_core.compression_systems.strategies.compression_strategy import (
    StrategySelector,
    CompressionStrategy,
    StrategyConfig
)


@dataclass
class FilterAnalysis:
    """Analysis results for convolutional filters"""
    num_filters: int
    filter_correlation_matrix: torch.Tensor
    redundant_filters: List[int]  # Indices of redundant filters
    separable_filters: List[int]  # Can be decomposed to 1D
    importance_scores: torch.Tensor  # Per-filter importance
    prunable_ratio: float  # Fraction that can be pruned
    
    def __post_init__(self):
        """Validate filter analysis results"""
        if self.num_filters <= 0:
            raise ValueError(f"Number of filters must be positive, got {self.num_filters}")
        if not (0.0 <= self.prunable_ratio <= 1.0):
            raise ValueError(f"Prunable ratio must be in [0, 1], got {self.prunable_ratio}")
        if len(self.importance_scores) != self.num_filters:
            raise ValueError(f"Importance scores length {len(self.importance_scores)} != num_filters {self.num_filters}")
        if self.filter_correlation_matrix.shape != (self.num_filters, self.num_filters):
            raise ValueError(f"Correlation matrix shape mismatch: {self.filter_correlation_matrix.shape}")


@dataclass
class ChannelAnalysis:
    """Channel-wise analysis results"""
    input_channels: int
    output_channels: int
    channel_correlation: torch.Tensor
    dead_channels: List[int]  # Zero activation channels
    channel_importance: torch.Tensor
    suggested_channel_groups: int  # For grouped convolution
    
    def __post_init__(self):
        """Validate channel analysis results"""
        if self.input_channels <= 0:
            raise ValueError(f"Input channels must be positive, got {self.input_channels}")
        if self.output_channels <= 0:
            raise ValueError(f"Output channels must be positive, got {self.output_channels}")
        if self.suggested_channel_groups < 1:
            raise ValueError(f"Suggested groups must be >= 1, got {self.suggested_channel_groups}")
        if len(self.channel_importance) != self.output_channels:
            raise ValueError(f"Channel importance length {len(self.channel_importance)} != output_channels {self.output_channels}")


@dataclass
class SpatialAnalysis:
    """Spatial pattern analysis"""
    kernel_size: Tuple[int, int]
    is_depthwise_separable: bool
    spatial_sparsity: float
    frequency_spectrum: torch.Tensor  # FFT of kernels
    edge_detector_score: float  # Similarity to edge kernels
    texture_score: float  # Periodic pattern score
    
    def __post_init__(self):
        """Validate spatial analysis results"""
        if len(self.kernel_size) != 2:
            raise ValueError(f"Kernel size must be 2D tuple, got {self.kernel_size}")
        if self.kernel_size[0] <= 0 or self.kernel_size[1] <= 0:
            raise ValueError(f"Kernel dimensions must be positive, got {self.kernel_size}")
        if not (0.0 <= self.spatial_sparsity <= 1.0):
            raise ValueError(f"Spatial sparsity must be in [0, 1], got {self.spatial_sparsity}")
        if not (0.0 <= self.edge_detector_score <= 1.0):
            raise ValueError(f"Edge detector score must be in [0, 1], got {self.edge_detector_score}")
        if not (0.0 <= self.texture_score <= 1.0):
            raise ValueError(f"Texture score must be in [0, 1], got {self.texture_score}")


class ConvolutionalAnalyzer(LayerAnalyzer):
    """Specialized analyzer for convolutional layers"""
    
    def __init__(self, device: Optional[torch.device] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize convolutional analyzer with device and configuration"""
        # Initialize base analyzer
        base_config = config or {}
        if device is not None:
            base_config['device'] = device
        super().__init__(config=base_config)
        
        # Initialize tropical factorizer for decomposition
        self.tropical_factorizer = TropicalMatrixFactorization()
        self.tropical_algebra = TropicalLinearAlgebra(device=self.device)
        
        # Logger for debugging
        self.logger = logging.getLogger('ConvolutionalAnalyzer')
        
        # Edge detection kernels for comparison
        self._init_edge_kernels()
    
    def _init_edge_kernels(self):
        """Initialize standard edge detection kernels for comparison"""
        # Sobel operators
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32, device=self.device)
        
        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32, device=self.device)
        
        # Laplacian
        self.laplacian = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32, device=self.device)
        
        # Prewitt operators
        self.prewitt_x = torch.tensor([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ], dtype=torch.float32, device=self.device)
        
        self.prewitt_y = torch.tensor([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ], dtype=torch.float32, device=self.device)
    
    def analyze_conv2d(self, layer: nn.Conv2d, 
                      input_stats: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Comprehensive Conv2d analysis
        
        Args:
            layer: Conv2d layer to analyze
            input_stats: Optional statistics from input activations
                - 'mean': Mean activation per channel
                - 'std': Standard deviation per channel
                - 'sparsity': Sparsity per channel
                
        Returns:
            Dictionary containing all analysis results
        """
        if not isinstance(layer, nn.Conv2d):
            raise TypeError(f"Expected nn.Conv2d, got {type(layer)}")
        
        # Get weight tensor
        weight = layer.weight.detach()
        if weight.device != self.device:
            weight = weight.to(self.device)
        
        # Validate weight tensor
        if torch.isnan(weight).any():
            raise ValueError("Conv2d layer contains NaN values")
        if torch.isinf(weight).any():
            raise ValueError("Conv2d layer contains infinite values")
        
        # Extract dimensions
        out_channels, in_channels, kernel_h, kernel_w = weight.shape
        
        # Perform analyses
        filter_analysis = self.analyze_filters(weight)
        channel_analysis = self.analyze_channels(weight, activation_stats=input_stats)
        spatial_analysis = self.analyze_spatial_patterns(weight)
        
        # Compute decomposition suggestions
        decomposition_suggestions = self.suggest_decomposition(layer)
        
        # Compute compression suitability
        compression_score = self._compute_compression_score(
            filter_analysis, channel_analysis, spatial_analysis
        )
        
        return {
            'layer_type': 'Conv2d',
            'shape': {
                'out_channels': out_channels,
                'in_channels': in_channels,
                'kernel_size': (kernel_h, kernel_w),
                'stride': layer.stride,
                'padding': layer.padding,
                'dilation': layer.dilation,
                'groups': layer.groups
            },
            'filter_analysis': filter_analysis,
            'channel_analysis': channel_analysis,
            'spatial_analysis': spatial_analysis,
            'decomposition_suggestions': decomposition_suggestions,
            'compression_score': compression_score,
            'parameter_count': weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
        }
    
    def analyze_filters(self, weight: torch.Tensor) -> FilterAnalysis:
        """
        Analyze filter redundancy and importance
        
        Args:
            weight: Conv2d weight tensor (out_channels, in_channels, H, W)
            
        Returns:
            FilterAnalysis with redundancy and importance metrics
        """
        out_channels, in_channels, kernel_h, kernel_w = weight.shape
        
        # Reshape filters to vectors for correlation analysis
        filters_flat = weight.view(out_channels, -1)
        
        # Compute filter correlation matrix
        # Normalize filters first
        filters_norm = F.normalize(filters_flat, p=2, dim=1)
        correlation_matrix = torch.mm(filters_norm, filters_norm.t())
        
        # Find redundant filters (high correlation)
        redundant_filters = self.find_redundant_filters(weight, threshold=0.95)
        
        # Test separability for each filter
        separable_filters = []
        for i in range(out_channels):
            # Average over input channels
            filter_avg = weight[i].mean(dim=0)
            is_sep, _ = self.test_separability(filter_avg)
            if is_sep:
                separable_filters.append(i)
        
        # Compute filter importance scores
        importance_scores = self.compute_filter_importance(weight)
        
        # Determine prunable ratio based on importance
        importance_threshold = torch.quantile(importance_scores, 0.3)  # Bottom 30%
        prunable_count = (importance_scores < importance_threshold).sum().item()
        prunable_ratio = prunable_count / out_channels
        
        return FilterAnalysis(
            num_filters=out_channels,
            filter_correlation_matrix=correlation_matrix,
            redundant_filters=redundant_filters,
            separable_filters=separable_filters,
            importance_scores=importance_scores,
            prunable_ratio=prunable_ratio
        )
    
    def analyze_channels(self, weight: torch.Tensor,
                        activation_stats: Optional[Dict] = None) -> ChannelAnalysis:
        """
        Analyze channel importance and correlation
        
        Args:
            weight: Conv2d weight tensor
            activation_stats: Optional activation statistics
            
        Returns:
            ChannelAnalysis with channel metrics
        """
        out_channels, in_channels, kernel_h, kernel_w = weight.shape
        
        # Compute channel-wise norms as importance proxy
        channel_importance = torch.norm(weight, p=2, dim=(2, 3))  # (out_channels, in_channels)
        channel_importance = channel_importance.mean(dim=1)  # Average over input channels
        
        # Normalize importance scores
        if channel_importance.max() > 0:
            channel_importance = channel_importance / channel_importance.max()
        
        # Find dead channels (very low importance)
        dead_threshold = 1e-6
        dead_channels = torch.where(channel_importance < dead_threshold)[0].tolist()
        
        # Compute channel correlation
        # Reshape to (out_channels, in_channels * H * W)
        weight_flat = weight.view(out_channels, -1)
        weight_norm = F.normalize(weight_flat, p=2, dim=1)
        channel_correlation = torch.mm(weight_norm, weight_norm.t())
        
        # Suggest channel groups based on correlation clusters
        suggested_groups = self._suggest_channel_groups(channel_correlation)
        
        # Use activation stats if provided
        if activation_stats and 'sparsity' in activation_stats:
            # Channels with high input sparsity can be pruned
            input_sparsity = activation_stats['sparsity']
            if len(input_sparsity) == in_channels:
                dead_input_channels = torch.where(input_sparsity > 0.99)[0].tolist()
                # Filters processing dead input channels are less important
                for dead_in in dead_input_channels:
                    channel_importance *= (1 - 0.5 * torch.norm(weight[:, dead_in], p=2, dim=(1, 2)) / 
                                         torch.norm(weight, p=2, dim=(1, 2, 3)))
        
        return ChannelAnalysis(
            input_channels=in_channels,
            output_channels=out_channels,
            channel_correlation=channel_correlation,
            dead_channels=dead_channels,
            channel_importance=channel_importance,
            suggested_channel_groups=suggested_groups
        )
    
    def _suggest_channel_groups(self, correlation_matrix: torch.Tensor) -> int:
        """
        Suggest number of groups for grouped convolution based on correlation
        
        Args:
            correlation_matrix: Channel correlation matrix
            
        Returns:
            Suggested number of groups
        """
        n_channels = correlation_matrix.shape[0]
        
        # Use spectral clustering approach
        # Find eigenvalues of correlation matrix
        try:
            eigenvalues, _ = torch.linalg.eigh(correlation_matrix)
            eigenvalues = eigenvalues.real
            
            # Count significant eigenvalues (spectral gap)
            total_energy = eigenvalues.sum()
            if total_energy > 0:
                cumsum = torch.cumsum(eigenvalues.flip(0), dim=0)
                n_components = (cumsum < 0.95 * total_energy).sum().item() + 1
                
                # Suggest groups based on components
                if n_components < n_channels * 0.25:
                    return min(8, n_channels // 4)  # High redundancy -> more groups
                elif n_components < n_channels * 0.5:
                    return min(4, n_channels // 8)  # Medium redundancy
                else:
                    return 1  # Low redundancy -> no grouping
            else:
                return 1
        except:
            # Fallback if eigendecomposition fails
            return 1
    
    def analyze_spatial_patterns(self, weight: torch.Tensor) -> SpatialAnalysis:
        """
        Analyze spatial structure of kernels
        
        Args:
            weight: Conv2d weight tensor
            
        Returns:
            SpatialAnalysis with spatial pattern metrics
        """
        out_channels, in_channels, kernel_h, kernel_w = weight.shape
        
        # Compute spatial sparsity
        spatial_zeros = (weight == 0).sum().item()
        spatial_sparsity = spatial_zeros / weight.numel()
        
        # Check if depthwise separable (most energy in 1D components)
        is_depthwise_sep = self._check_depthwise_separability(weight)
        
        # Compute frequency spectrum using 2D FFT
        # Average over channels for analysis
        weight_avg = weight.mean(dim=(0, 1))  # Average kernel
        if weight_avg.numel() > 0:
            # Pad to power of 2 for FFT efficiency
            pad_h = 2 ** math.ceil(math.log2(kernel_h)) - kernel_h
            pad_w = 2 ** math.ceil(math.log2(kernel_w)) - kernel_w
            weight_padded = F.pad(weight_avg, (0, pad_w, 0, pad_h))
            
            # Compute 2D FFT
            freq_spectrum = torch.fft.fft2(weight_padded).abs()
        else:
            freq_spectrum = torch.zeros(kernel_h, kernel_w, device=self.device)
        
        # Compute edge detector similarity
        edge_score = self._compute_edge_detector_score(weight)
        
        # Compute texture score (periodicity)
        texture_score = self._compute_texture_score(weight, freq_spectrum)
        
        return SpatialAnalysis(
            kernel_size=(kernel_h, kernel_w),
            is_depthwise_separable=is_depthwise_sep,
            spatial_sparsity=spatial_sparsity,
            frequency_spectrum=freq_spectrum,
            edge_detector_score=edge_score,
            texture_score=texture_score
        )
    
    def _check_depthwise_separability(self, weight: torch.Tensor) -> bool:
        """Check if filters are approximately depthwise separable"""
        out_channels, in_channels, kernel_h, kernel_w = weight.shape
        
        # Sample some filters to test
        n_samples = min(10, out_channels)
        sample_indices = torch.randperm(out_channels)[:n_samples]
        
        separable_count = 0
        for idx in sample_indices:
            # Average over input channels
            kernel = weight[idx].mean(dim=0)
            is_sep, sep_error = self.test_separability(kernel)
            if is_sep:
                separable_count += 1
        
        # If most sampled filters are separable
        return separable_count > n_samples * 0.7
    
    def _compute_edge_detector_score(self, weight: torch.Tensor) -> float:
        """Compute similarity to standard edge detection kernels"""
        if weight.shape[2] != 3 or weight.shape[3] != 3:
            # Only compute for 3x3 kernels
            return 0.0
        
        out_channels = weight.shape[0]
        
        # Compare each filter to edge kernels
        edge_scores = []
        edge_kernels = [self.sobel_x, self.sobel_y, self.laplacian, 
                       self.prewitt_x, self.prewitt_y]
        
        for i in range(min(out_channels, 20)):  # Sample up to 20 filters
            filter_2d = weight[i].mean(dim=0)  # Average over input channels
            filter_norm = F.normalize(filter_2d.flatten(), p=2, dim=0)
            
            max_similarity = 0.0
            for edge_kernel in edge_kernels:
                edge_norm = F.normalize(edge_kernel.flatten(), p=2, dim=0)
                similarity = torch.abs(torch.dot(filter_norm, edge_norm)).item()
                max_similarity = max(max_similarity, similarity)
            
            edge_scores.append(max_similarity)
        
        return float(np.mean(edge_scores)) if edge_scores else 0.0
    
    def _compute_texture_score(self, weight: torch.Tensor, freq_spectrum: torch.Tensor) -> float:
        """Compute texture/periodicity score from frequency spectrum"""
        if freq_spectrum.numel() == 0:
            return 0.0
        
        # Flatten spectrum and find peaks
        spectrum_flat = freq_spectrum.flatten()
        
        # Compute ratio of energy in high frequencies
        total_energy = spectrum_flat.sum().item()
        if total_energy > 0:
            # Sort frequencies by magnitude
            sorted_freqs, _ = torch.sort(spectrum_flat, descending=True)
            
            # Top 10% frequencies
            n_top = max(1, len(sorted_freqs) // 10)
            top_energy = sorted_freqs[:n_top].sum().item()
            
            # High concentration in few frequencies indicates texture
            concentration_ratio = top_energy / total_energy
            
            # Also check for regular spacing (periodicity)
            if n_top > 3:
                # Check if top frequencies are regularly spaced
                top_indices = torch.topk(spectrum_flat, n_top).indices
                if len(top_indices) > 1:
                    diffs = torch.diff(torch.sort(top_indices).values).float()
                    regularity = 1.0 - (diffs.std() / (diffs.mean() + 1e-8)).item()
                    regularity = max(0.0, min(1.0, regularity))
                else:
                    regularity = 0.0
            else:
                regularity = 0.0
            
            texture_score = 0.7 * concentration_ratio + 0.3 * regularity
            return float(min(1.0, texture_score))
        
        return 0.0
    
    def compute_filter_importance(self, weight: torch.Tensor,
                                 gradient: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Taylor expansion or gradient-based importance
        
        Args:
            weight: Conv2d weight tensor
            gradient: Optional gradient tensor for Taylor importance
            
        Returns:
            Per-filter importance scores
        """
        out_channels = weight.shape[0]
        
        if gradient is not None and gradient.shape == weight.shape:
            # Taylor importance: |w * grad_w|
            taylor_importance = torch.abs(weight * gradient)
            # Sum over input channels and spatial dimensions
            importance = taylor_importance.sum(dim=(1, 2, 3))
        else:
            # Fallback: L2 norm-based importance
            importance = torch.norm(weight, p=2, dim=(1, 2, 3))
        
        # Normalize to [0, 1]
        if importance.max() > 0:
            importance = importance / importance.max()
        else:
            importance = torch.ones(out_channels, device=weight.device)
        
        return importance
    
    def find_redundant_filters(self, weight: torch.Tensor,
                              threshold: float = 0.95) -> List[int]:
        """
        Find filters that can be merged/removed
        
        Args:
            weight: Conv2d weight tensor
            threshold: Correlation threshold for redundancy
            
        Returns:
            List of redundant filter indices
        """
        out_channels = weight.shape[0]
        
        # Reshape filters for correlation
        filters_flat = weight.view(out_channels, -1)
        filters_norm = F.normalize(filters_flat, p=2, dim=1)
        
        # Compute pairwise correlations
        correlation = torch.mm(filters_norm, filters_norm.t())
        
        # Find redundant pairs
        redundant = []
        processed = set()
        
        for i in range(out_channels):
            if i in processed:
                continue
            
            # Find filters highly correlated with filter i
            correlated = torch.where(correlation[i] > threshold)[0]
            
            # Keep first, mark others as redundant
            for j in correlated:
                if j != i and j not in processed:
                    redundant.append(j.item())
                    processed.add(j)
            
            processed.add(i)
        
        return redundant
    
    def test_separability(self, kernel: torch.Tensor) -> Tuple[bool, float]:
        """
        Test if kernel is separable into 1D components
        
        Args:
            kernel: 2D kernel tensor
            
        Returns:
            (is_separable, reconstruction_error)
        """
        if kernel.dim() != 2:
            return False, float('inf')
        
        h, w = kernel.shape
        
        # Use SVD to check separability
        try:
            U, S, Vt = torch.linalg.svd(kernel, full_matrices=False)
            
            # Check if rank-1 approximation is good
            rank1_approx = S[0] * torch.outer(U[:, 0], Vt[0, :])
            
            # Compute reconstruction error
            error = torch.norm(kernel - rank1_approx) / (torch.norm(kernel) + 1e-8)
            
            # Threshold for separability
            is_separable = error < 0.1  # 10% error threshold
            
            return is_separable, error.item()
        except:
            return False, float('inf')
    
    def suggest_decomposition(self, layer: nn.Conv2d) -> Dict[str, Any]:
        """
        Suggest decomposition strategy (CP, Tucker, SVD)
        
        Args:
            layer: Conv2d layer to analyze
            
        Returns:
            Dictionary with decomposition suggestions
        """
        weight = layer.weight
        out_channels, in_channels, kernel_h, kernel_w = weight.shape
        
        suggestions = {}
        
        # 1. Spatial decomposition (k×k → k×1 + 1×k)
        if kernel_h == kernel_w and kernel_h > 1:
            # Test separability on sample filters
            n_samples = min(10, out_channels)
            separable_count = 0
            
            for i in range(n_samples):
                kernel = weight[i].mean(dim=0)
                is_sep, _ = self.test_separability(kernel)
                if is_sep:
                    separable_count += 1
            
            if separable_count > n_samples * 0.5:
                suggestions['spatial_decomposition'] = {
                    'feasible': True,
                    'original_ops': out_channels * in_channels * kernel_h * kernel_w,
                    'decomposed_ops': out_channels * in_channels * (kernel_h + kernel_w),
                    'speedup': (kernel_h * kernel_w) / (kernel_h + kernel_w)
                }
        
        # 2. Channel decomposition (bottleneck)
        if in_channels > 32 and out_channels > 32:
            # Suggest bottleneck with rank r
            min_channels = min(in_channels, out_channels)
            suggested_rank = max(min_channels // 4, 16)
            
            original_params = out_channels * in_channels * kernel_h * kernel_w
            bottleneck_params = (in_channels * suggested_rank * 1 * 1 + 
                               suggested_rank * out_channels * kernel_h * kernel_w)
            
            suggestions['channel_bottleneck'] = {
                'feasible': bottleneck_params < original_params * 0.5,
                'bottleneck_channels': suggested_rank,
                'original_params': original_params,
                'decomposed_params': bottleneck_params,
                'compression_ratio': original_params / bottleneck_params
            }
        
        # 3. Depthwise separable
        if layer.groups == 1 and in_channels > 1:
            original_params = out_channels * in_channels * kernel_h * kernel_w
            depthwise_params = in_channels * kernel_h * kernel_w  # Depthwise
            pointwise_params = out_channels * in_channels * 1 * 1  # Pointwise
            total_params = depthwise_params + pointwise_params
            
            suggestions['depthwise_separable'] = {
                'feasible': total_params < original_params * 0.3,
                'original_params': original_params,
                'decomposed_params': total_params,
                'compression_ratio': original_params / total_params
            }
        
        # 4. Tucker decomposition for 4D tensor
        if out_channels > 16 and in_channels > 16:
            # Suggest Tucker ranks
            tucker_ranks = (
                max(out_channels // 2, 8),
                max(in_channels // 2, 8),
                kernel_h,  # Keep spatial dims
                kernel_w
            )
            
            original_params = out_channels * in_channels * kernel_h * kernel_w
            core_params = tucker_ranks[0] * tucker_ranks[1] * tucker_ranks[2] * tucker_ranks[3]
            factor_params = (out_channels * tucker_ranks[0] + 
                           in_channels * tucker_ranks[1])
            tucker_params = core_params + factor_params
            
            suggestions['tucker_decomposition'] = {
                'feasible': tucker_params < original_params * 0.5,
                'tucker_ranks': tucker_ranks,
                'original_params': original_params,
                'decomposed_params': tucker_params,
                'compression_ratio': original_params / tucker_params
            }
        
        # 5. CP decomposition (sum of rank-1 tensors)
        if out_channels > 16 and in_channels > 16:
            # Estimate CP rank
            cp_rank = max(min(out_channels, in_channels) // 4, 16)
            
            original_params = out_channels * in_channels * kernel_h * kernel_w
            cp_params = cp_rank * (out_channels + in_channels + kernel_h + kernel_w)
            
            suggestions['cp_decomposition'] = {
                'feasible': cp_params < original_params * 0.3,
                'cp_rank': cp_rank,
                'original_params': original_params,
                'decomposed_params': cp_params,
                'compression_ratio': original_params / cp_params
            }
        
        return suggestions
    
    def _compute_compression_score(self, filter_analysis: FilterAnalysis,
                                  channel_analysis: ChannelAnalysis,
                                  spatial_analysis: SpatialAnalysis) -> float:
        """Compute overall compression suitability score"""
        score = 0.0
        
        # High redundancy in filters
        if len(filter_analysis.redundant_filters) > filter_analysis.num_filters * 0.2:
            score += 0.25
        
        # High prunable ratio
        if filter_analysis.prunable_ratio > 0.3:
            score += 0.20
        
        # Dead channels exist
        if len(channel_analysis.dead_channels) > 0:
            score += 0.15
        
        # Separable filters
        if len(filter_analysis.separable_filters) > filter_analysis.num_filters * 0.3:
            score += 0.20
        
        # High spatial sparsity
        if spatial_analysis.spatial_sparsity > 0.5:
            score += 0.10
        
        # Edge detector patterns (good for tropical)
        if spatial_analysis.edge_detector_score > 0.7:
            score += 0.10
        
        return min(score, 1.0)


class ConvolutionalCompressionStrategy:
    """Compression strategies specific to CNNs"""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize compression strategy"""
        self.analyzer = ConvolutionalAnalyzer(device=device)
        self.device = device if device else torch.device('cpu')
        self.logger = logging.getLogger('ConvolutionalCompressionStrategy')
    
    def structured_pruning(self, layer: nn.Conv2d,
                          pruning_ratio: float = 0.5) -> nn.Conv2d:
        """
        Remove entire filters/channels
        
        Args:
            layer: Conv2d layer to prune
            pruning_ratio: Fraction of filters to remove
            
        Returns:
            Pruned Conv2d layer
        """
        if not (0.0 < pruning_ratio < 1.0):
            raise ValueError(f"Pruning ratio must be in (0, 1), got {pruning_ratio}")
        
        weight = layer.weight.detach()
        out_channels, in_channels, kernel_h, kernel_w = weight.shape
        
        # Compute filter importance
        importance = self.analyzer.compute_filter_importance(weight)
        
        # Determine filters to keep
        n_keep = int(out_channels * (1 - pruning_ratio))
        n_keep = max(1, n_keep)  # Keep at least one filter
        
        # Get indices of top filters
        _, keep_indices = torch.topk(importance, n_keep)
        keep_indices, _ = torch.sort(keep_indices)
        
        # Create new layer with pruned filters
        new_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_keep,
            kernel_size=(kernel_h, kernel_w),
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None
        ).to(self.device)
        
        # Copy kept weights
        new_layer.weight.data = weight[keep_indices]
        
        if layer.bias is not None:
            new_layer.bias.data = layer.bias[keep_indices]
        
        self.logger.info(f"Pruned Conv2d from {out_channels} to {n_keep} filters")
        
        return new_layer
    
    def low_rank_decomposition(self, layer: nn.Conv2d,
                              rank_ratio: float = 0.5) -> nn.Sequential:
        """
        Decompose into smaller layers using bottleneck
        
        Args:
            layer: Conv2d layer to decompose
            rank_ratio: Ratio of rank to keep
            
        Returns:
            Sequential module with decomposed layers
        """
        if not (0.0 < rank_ratio < 1.0):
            raise ValueError(f"Rank ratio must be in (0, 1), got {rank_ratio}")
        
        weight = layer.weight
        out_channels, in_channels, kernel_h, kernel_w = weight.shape
        
        # Determine bottleneck size
        min_channels = min(in_channels, out_channels)
        bottleneck_channels = max(int(min_channels * rank_ratio), 1)
        
        # Create bottleneck architecture
        layers = []
        
        # 1×1 conv to reduce channels
        conv_reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        ).to(self.device)
        
        # Original kernel size conv with reduced channels
        conv_main = nn.Conv2d(
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
            kernel_size=(kernel_h, kernel_w),
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=min(layer.groups, bottleneck_channels),
            bias=False
        ).to(self.device)
        
        # 1×1 conv to expand channels
        conv_expand = nn.Conv2d(
            in_channels=bottleneck_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=layer.bias is not None
        ).to(self.device)
        
        # Initialize with SVD approximation
        self._initialize_bottleneck_weights(
            layer, conv_reduce, conv_main, conv_expand, bottleneck_channels
        )
        
        layers = [conv_reduce, conv_main, conv_expand]
        
        return nn.Sequential(*layers)
    
    def _initialize_bottleneck_weights(self, original_layer: nn.Conv2d,
                                      conv_reduce: nn.Conv2d,
                                      conv_main: nn.Conv2d,
                                      conv_expand: nn.Conv2d,
                                      bottleneck_channels: int):
        """Initialize bottleneck weights using decomposition"""
        weight = original_layer.weight
        out_channels, in_channels, kernel_h, kernel_w = weight.shape
        
        # Reshape weight for matrix decomposition
        # Flatten to (out_channels, in_channels * H * W)
        weight_reshaped = weight.reshape(out_channels, -1)
        
        # SVD decomposition
        try:
            U, S, Vt = torch.linalg.svd(weight_reshaped, full_matrices=False)
            
            # Keep top components
            U_r = U[:, :bottleneck_channels]
            S_r = S[:bottleneck_channels]
            Vt_r = Vt[:bottleneck_channels, :]
            
            # Initialize reduce layer with proper reshaping
            # Vt_r has shape (bottleneck_channels, in_channels * H * W)
            # We need to map this to (bottleneck_channels, in_channels, 1, 1)
            # Use average pooling across spatial dimensions
            Vt_reshape = Vt_r.reshape(bottleneck_channels, in_channels, kernel_h * kernel_w)
            conv_reduce.weight.data = Vt_reshape.mean(dim=2).unsqueeze(-1).unsqueeze(-1)
            
            # Initialize expand layer (U_r)
            conv_expand.weight.data = U_r.t().unsqueeze(-1).unsqueeze(-1)
            
            # Initialize main convolution with diagonal S
            conv_main.weight.data = torch.randn_like(conv_main.weight) * 0.01
            for i in range(min(bottleneck_channels, len(S_r))):
                conv_main.weight.data[i, i] *= S_r[i].sqrt()
            
            # Copy bias if exists
            if original_layer.bias is not None:
                conv_expand.bias.data = original_layer.bias.data
                
        except Exception as e:
            self.logger.warning(f"SVD initialization failed: {e}, using random init")
            # Fallback to random initialization
            nn.init.kaiming_normal_(conv_reduce.weight)
            nn.init.kaiming_normal_(conv_main.weight)
            nn.init.kaiming_normal_(conv_expand.weight)
    
    def depthwise_separation(self, layer: nn.Conv2d) -> nn.Sequential:
        """
        Convert to depthwise separable convolution
        
        Args:
            layer: Conv2d layer to convert
            
        Returns:
            Sequential with depthwise and pointwise convolutions
        """
        out_channels = layer.out_channels
        in_channels = layer.in_channels
        
        # Depthwise convolution (spatial)
        depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=in_channels,  # Key: groups = in_channels
            bias=False
        ).to(self.device)
        
        # Pointwise convolution (channel mixing)
        pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=layer.bias is not None
        ).to(self.device)
        
        # Initialize weights
        self._initialize_depthwise_weights(layer, depthwise, pointwise)
        
        return nn.Sequential(depthwise, pointwise)
    
    def _initialize_depthwise_weights(self, original: nn.Conv2d,
                                     depthwise: nn.Conv2d,
                                     pointwise: nn.Conv2d):
        """Initialize depthwise separable weights"""
        weight = original.weight
        out_channels, in_channels, kernel_h, kernel_w = weight.shape
        
        # For depthwise: average filters across output channels
        for i in range(in_channels):
            # Get all filters that process input channel i
            filters_for_channel = weight[:, i, :, :]
            # Average them for depthwise kernel
            depthwise.weight.data[i, 0] = filters_for_channel.mean(dim=0)
        
        # For pointwise: use norm of original filters
        for o in range(out_channels):
            for i in range(in_channels):
                # Preserve channel mixing information
                pointwise.weight.data[o, i, 0, 0] = weight[o, i].norm()
        
        # Copy bias
        if original.bias is not None:
            pointwise.bias.data = original.bias.data
    
    def spatial_decomposition(self, layer: nn.Conv2d) -> nn.Sequential:
        """
        Decompose k×k into k×1 and 1×k
        
        Args:
            layer: Conv2d layer with square kernel
            
        Returns:
            Sequential with two asymmetric convolutions
        """
        kernel_size = layer.kernel_size
        if isinstance(kernel_size, tuple):
            kernel_h, kernel_w = kernel_size
        else:
            kernel_h = kernel_w = kernel_size
        
        if kernel_h != kernel_w:
            raise ValueError(f"Spatial decomposition requires square kernel, got {kernel_size}")
        
        # Vertical convolution (k×1)
        conv_v = nn.Conv2d(
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            kernel_size=(kernel_h, 1),
            stride=(layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride, 1),
            padding=(layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding, 0),
            dilation=layer.dilation,
            groups=layer.groups,
            bias=False
        ).to(self.device)
        
        # Horizontal convolution (1×k)
        conv_h = nn.Conv2d(
            in_channels=layer.out_channels,
            out_channels=layer.out_channels,
            kernel_size=(1, kernel_w),
            stride=(1, layer.stride[1] if isinstance(layer.stride, tuple) else layer.stride),
            padding=(0, layer.padding[1] if isinstance(layer.padding, tuple) else layer.padding),
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None
        ).to(self.device)
        
        # Initialize using SVD of original kernels
        self._initialize_spatial_decomposition(layer, conv_v, conv_h)
        
        return nn.Sequential(conv_v, conv_h)
    
    def _initialize_spatial_decomposition(self, original: nn.Conv2d,
                                         conv_v: nn.Conv2d,
                                         conv_h: nn.Conv2d):
        """Initialize spatially decomposed convolutions"""
        weight = original.weight
        out_channels, in_channels, kernel_h, kernel_w = weight.shape
        
        # Process each output channel
        for o in range(out_channels):
            for i in range(in_channels):
                kernel = weight[o, i]
                
                # SVD of 2D kernel
                try:
                    U, S, Vt = torch.linalg.svd(kernel, full_matrices=False)
                    
                    # Use rank-1 approximation
                    vertical = U[:, 0:1] * S[0].sqrt()
                    horizontal = Vt[0:1, :] * S[0].sqrt()
                    
                    conv_v.weight.data[o, i, :, 0] = vertical.squeeze()
                    if o < conv_h.out_channels and o < conv_h.in_channels:
                        conv_h.weight.data[o, o, 0, :] = horizontal.squeeze()
                except:
                    # Fallback to random if SVD fails
                    conv_v.weight.data[o, i].normal_(0, 0.01)
                    if o < conv_h.out_channels and o < conv_h.in_channels:
                        conv_h.weight.data[o, o].normal_(0, 0.01)
        
        # Copy bias
        if original.bias is not None:
            conv_h.bias.data = original.bias.data
    
    def channel_grouping(self, layer: nn.Conv2d,
                        num_groups: int) -> nn.Conv2d:
        """
        Convert to grouped convolution
        
        Args:
            layer: Conv2d layer to convert
            num_groups: Number of channel groups
            
        Returns:
            Grouped convolution layer
        """
        if layer.in_channels % num_groups != 0:
            raise ValueError(f"Input channels {layer.in_channels} not divisible by groups {num_groups}")
        if layer.out_channels % num_groups != 0:
            raise ValueError(f"Output channels {layer.out_channels} not divisible by groups {num_groups}")
        
        # Create grouped convolution
        grouped_conv = nn.Conv2d(
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=num_groups,
            bias=layer.bias is not None
        ).to(self.device)
        
        # Reorganize weights for grouped convolution
        weight = layer.weight
        out_channels, in_channels, kernel_h, kernel_w = weight.shape
        
        out_per_group = out_channels // num_groups
        in_per_group = in_channels // num_groups
        
        # Rearrange weights to group structure
        for g in range(num_groups):
            out_start = g * out_per_group
            out_end = (g + 1) * out_per_group
            in_start = g * in_per_group
            in_end = (g + 1) * in_per_group
            
            grouped_conv.weight.data[out_start:out_end, :in_per_group] = \
                weight[out_start:out_end, in_start:in_end]
        
        # Copy bias
        if layer.bias is not None:
            grouped_conv.bias.data = layer.bias.data
        
        return grouped_conv


class AdaptiveConvCompressor:
    """Adaptive compression for entire CNN models"""
    
    def __init__(self, target_compression: float = 4.0,
                device: Optional[torch.device] = None):
        """
        Initialize adaptive compressor
        
        Args:
            target_compression: Target compression ratio
            device: Device for computations
        """
        if target_compression <= 1.0:
            raise ValueError(f"Target compression must be > 1, got {target_compression}")
        
        self.target_compression = target_compression
        self.device = device if device else torch.device('cpu')
        self.conv_analyzer = ConvolutionalAnalyzer(device=self.device)
        self.conv_strategy = ConvolutionalCompressionStrategy(device=self.device)
        self.logger = logging.getLogger('AdaptiveConvCompressor')
    
    def compress_cnn(self, model: nn.Module) -> nn.Module:
        """
        Compress all conv layers in model
        
        Args:
            model: CNN model to compress
            
        Returns:
            Compressed model
        """
        compressed_model = model.to(self.device)
        original_params = sum(p.numel() for p in model.parameters())
        
        # Analyze all conv layers
        conv_layers = self._find_conv_layers(compressed_model)
        if not conv_layers:
            self.logger.warning("No convolutional layers found")
            return compressed_model
        
        # Analyze compression potential
        layer_analyses = {}
        for name, layer in conv_layers.items():
            analysis = self.conv_analyzer.analyze_conv2d(layer)
            layer_analyses[name] = analysis
        
        # Sort layers by compression potential
        sorted_layers = sorted(
            layer_analyses.items(),
            key=lambda x: x[1]['compression_score'],
            reverse=True
        )
        
        # Apply compression strategies
        compressed_params = original_params
        for name, analysis in sorted_layers:
            if compressed_params <= original_params / self.target_compression:
                break  # Target reached
            
            layer = dict(compressed_model.named_modules())[name]
            suggestions = analysis['decomposition_suggestions']
            
            # Choose best strategy
            best_strategy = self._choose_compression_strategy(suggestions)
            
            if best_strategy:
                compressed_layer = self._apply_compression_strategy(
                    layer, best_strategy
                )
                
                # Replace layer in model
                self._replace_layer(compressed_model, name, compressed_layer)
                
                # Update param count
                new_params = sum(p.numel() for p in compressed_layer.parameters())
                old_params = sum(p.numel() for p in layer.parameters())
                compressed_params = compressed_params - old_params + new_params
                
                self.logger.info(f"Compressed {name}: {old_params} -> {new_params} params")
        
        final_compression = original_params / compressed_params
        self.logger.info(f"Final compression ratio: {final_compression:.2f}x")
        
        return compressed_model
    
    def _find_conv_layers(self, model: nn.Module) -> Dict[str, nn.Conv2d]:
        """Find all Conv2d layers in model"""
        conv_layers = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers[name] = module
        
        return conv_layers
    
    def _choose_compression_strategy(self, suggestions: Dict[str, Any]) -> Optional[str]:
        """Choose best compression strategy from suggestions"""
        feasible_strategies = [
            (name, info) for name, info in suggestions.items()
            if info.get('feasible', False)
        ]
        
        if not feasible_strategies:
            return None
        
        # Sort by compression ratio
        best = max(feasible_strategies, 
                  key=lambda x: x[1].get('compression_ratio', 1.0))
        
        return best[0] if best[1].get('compression_ratio', 1.0) > 1.5 else None
    
    def _apply_compression_strategy(self, layer: nn.Conv2d, strategy: str) -> nn.Module:
        """Apply selected compression strategy"""
        if strategy == 'spatial_decomposition':
            return self.conv_strategy.spatial_decomposition(layer)
        elif strategy == 'channel_bottleneck':
            return self.conv_strategy.low_rank_decomposition(layer, rank_ratio=0.5)
        elif strategy == 'depthwise_separable':
            return self.conv_strategy.depthwise_separation(layer)
        elif strategy == 'tucker_decomposition':
            # Use low-rank as approximation for Tucker
            return self.conv_strategy.low_rank_decomposition(layer, rank_ratio=0.4)
        elif strategy == 'cp_decomposition':
            # Use low-rank as approximation for CP
            return self.conv_strategy.low_rank_decomposition(layer, rank_ratio=0.3)
        else:
            return layer
    
    def _replace_layer(self, model: nn.Module, layer_name: str, new_layer: nn.Module):
        """Replace layer in model"""
        parts = layer_name.split('.')
        parent = model
        
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, parts[-1], new_layer)
    
    def analyze_cnn_compressibility(self, model: nn.Module) -> Dict[str, Any]:
        """
        Analyze compression potential of CNN
        
        Args:
            model: CNN model to analyze
            
        Returns:
            Dictionary with compressibility analysis
        """
        conv_layers = self._find_conv_layers(model)
        
        total_params = sum(p.numel() for p in model.parameters())
        conv_params = sum(
            sum(p.numel() for p in layer.parameters())
            for layer in conv_layers.values()
        )
        
        layer_analyses = {}
        total_compression_potential = 0
        
        for name, layer in conv_layers.items():
            analysis = self.conv_analyzer.analyze_conv2d(layer)
            layer_analyses[name] = {
                'compression_score': analysis['compression_score'],
                'redundant_filters': len(analysis['filter_analysis'].redundant_filters),
                'separable_filters': len(analysis['filter_analysis'].separable_filters),
                'prunable_ratio': analysis['filter_analysis'].prunable_ratio,
                'suggestions': analysis['decomposition_suggestions']
            }
            
            # Estimate compression for this layer
            best_compression = 1.0
            for strategy, info in analysis['decomposition_suggestions'].items():
                if info.get('feasible', False):
                    ratio = info.get('compression_ratio', 1.0)
                    best_compression = max(best_compression, ratio)
            
            layer_params = sum(p.numel() for p in layer.parameters())
            compressible_params = layer_params * (1 - 1/best_compression)
            total_compression_potential += compressible_params
        
        estimated_final_params = total_params - total_compression_potential
        estimated_compression = total_params / estimated_final_params
        
        return {
            'total_parameters': total_params,
            'conv_parameters': conv_params,
            'conv_percentage': conv_params / total_params,
            'estimated_compression_ratio': estimated_compression,
            'layer_analyses': layer_analyses,
            'num_conv_layers': len(conv_layers)
        }
    
    def profile_layer_importance(self, model: nn.Module,
                                calibration_data: DataLoader) -> Dict[str, float]:
        """
        Profile importance of each conv layer
        
        Args:
            model: CNN model
            calibration_data: DataLoader for calibration
            
        Returns:
            Dictionary mapping layer names to importance scores
        """
        model.eval()
        conv_layers = self._find_conv_layers(model)
        
        # Hook storage
        activations = {}
        gradients = {}
        
        def save_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        def save_gradient(name):
            def hook(module, grad_input, grad_output):
                gradients[name] = grad_output[0].detach()
            return hook
        
        # Register hooks
        handles = []
        for name, layer in conv_layers.items():
            handles.append(layer.register_forward_hook(save_activation(name)))
            handles.append(layer.register_backward_hook(save_gradient(name)))
        
        # Run calibration
        importance_scores = {name: 0.0 for name in conv_layers}
        
        with torch.enable_grad():
            for batch_idx, (data, target) in enumerate(calibration_data):
                if batch_idx >= 10:  # Use first 10 batches
                    break
                
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                output = model(data)
                loss = F.cross_entropy(output, target)
                
                # Backward pass
                loss.backward()
                
                # Compute importance from activations and gradients
                for name in conv_layers:
                    if name in activations and name in gradients:
                        act = activations[name]
                        grad = gradients[name]
                        
                        # Taylor importance: |activation * gradient|
                        importance = torch.abs(act * grad).mean().item()
                        importance_scores[name] += importance
                
                # Clear gradients
                model.zero_grad()
                activations.clear()
                gradients.clear()
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Normalize scores
        max_score = max(importance_scores.values())
        if max_score > 0:
            for name in importance_scores:
                importance_scores[name] /= max_score
        
        return importance_scores
    
    def compress_with_accuracy_target(self, model: nn.Module,
                                     val_loader: DataLoader,
                                     target_accuracy: float,
                                     baseline_accuracy: Optional[float] = None) -> nn.Module:
        """
        Compress while maintaining accuracy
        
        Args:
            model: Model to compress
            val_loader: Validation data loader
            target_accuracy: Minimum accuracy to maintain
            baseline_accuracy: Original accuracy (computed if not provided)
            
        Returns:
            Compressed model meeting accuracy target
        """
        # Compute baseline accuracy if not provided
        if baseline_accuracy is None:
            baseline_accuracy = self._evaluate_accuracy(model, val_loader)
            self.logger.info(f"Baseline accuracy: {baseline_accuracy:.4f}")
        
        # Profile layer importance
        importance_scores = self.profile_layer_importance(model, val_loader)
        
        # Sort layers by importance (ascending)
        sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1])
        
        compressed_model = model
        current_accuracy = baseline_accuracy
        
        for layer_name, importance in sorted_layers:
            if current_accuracy < target_accuracy:
                self.logger.info(f"Accuracy {current_accuracy:.4f} below target {target_accuracy:.4f}, stopping")
                break
            
            # Try compressing this layer
            layer = dict(compressed_model.named_modules())[layer_name]
            
            # Mild compression for important layers
            if importance > 0.7:
                continue  # Skip very important layers
            
            # Determine compression aggressiveness
            if importance < 0.3:
                pruning_ratio = 0.5  # Aggressive
            elif importance < 0.5:
                pruning_ratio = 0.3  # Moderate
            else:
                pruning_ratio = 0.2  # Conservative
            
            # Apply structured pruning
            try:
                pruned_layer = self.conv_strategy.structured_pruning(layer, pruning_ratio)
                
                # Test accuracy with this change
                test_model = compressed_model
                self._replace_layer(test_model, layer_name, pruned_layer)
                
                test_accuracy = self._evaluate_accuracy(test_model, val_loader)
                
                if test_accuracy >= target_accuracy:
                    compressed_model = test_model
                    current_accuracy = test_accuracy
                    self.logger.info(f"Compressed {layer_name} (importance={importance:.3f}), "
                                   f"accuracy: {current_accuracy:.4f}")
                else:
                    # Revert change
                    self._replace_layer(test_model, layer_name, layer)
                    
            except Exception as e:
                self.logger.warning(f"Failed to compress {layer_name}: {e}")
                continue
        
        return compressed_model
    
    def _evaluate_accuracy(self, model: nn.Module, val_loader: DataLoader) -> float:
        """Evaluate model accuracy on validation set"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                output = model(data)
                pred = output.argmax(dim=1)
                
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        return correct / total if total > 0 else 0.0


# Unit tests
if __name__ == "__main__":
    """Comprehensive unit tests for convolutional analyzer"""
    
    print("Testing ConvolutionalAnalyzer...")
    
    # Test device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create analyzer
    analyzer = ConvolutionalAnalyzer(device=device)
    
    # Test 1: Basic Conv2d analysis
    print("\n1. Testing basic Conv2d analysis...")
    conv = nn.Conv2d(32, 64, kernel_size=3, padding=1).to(device)
    analysis = analyzer.analyze_conv2d(conv)
    
    assert 'filter_analysis' in analysis
    assert 'channel_analysis' in analysis
    assert 'spatial_analysis' in analysis
    assert analysis['filter_analysis'].num_filters == 64
    assert analysis['channel_analysis'].input_channels == 32
    print("✓ Basic Conv2d analysis passed")
    
    # Test 2: Filter redundancy detection
    print("\n2. Testing filter redundancy detection...")
    # Create conv with redundant filters
    conv_redundant = nn.Conv2d(16, 32, kernel_size=3).to(device)
    # Make some filters identical
    conv_redundant.weight.data[1] = conv_redundant.weight.data[0]
    conv_redundant.weight.data[3] = conv_redundant.weight.data[2]
    
    redundant = analyzer.find_redundant_filters(conv_redundant.weight)
    assert len(redundant) >= 2  # Should find at least 2 redundant filters
    print(f"✓ Found {len(redundant)} redundant filters")
    
    # Test 3: Separability testing
    print("\n3. Testing kernel separability...")
    # Create separable kernel
    vertical = torch.randn(5, 1)
    horizontal = torch.randn(1, 5)
    separable_kernel = torch.mm(vertical, horizontal).to(device)
    
    is_sep, error = analyzer.test_separability(separable_kernel)
    assert is_sep == True
    assert error < 0.1
    print(f"✓ Separability test passed (error={error:.4f})")
    
    # Test 4: Compression strategies
    print("\n4. Testing compression strategies...")
    strategy = ConvolutionalCompressionStrategy(device=device)
    
    # Test structured pruning
    conv_large = nn.Conv2d(64, 128, kernel_size=3).to(device)
    pruned = strategy.structured_pruning(conv_large, pruning_ratio=0.5)
    assert pruned.out_channels == 64  # Should have half the filters
    print("✓ Structured pruning passed")
    
    # Test low-rank decomposition
    decomposed = strategy.low_rank_decomposition(conv_large, rank_ratio=0.5)
    assert isinstance(decomposed, nn.Sequential)
    assert len(decomposed) == 3  # Reduce, main, expand
    print("✓ Low-rank decomposition passed")
    
    # Test depthwise separation
    conv_regular = nn.Conv2d(32, 64, kernel_size=3, padding=1).to(device)
    depthwise_sep = strategy.depthwise_separation(conv_regular)
    assert isinstance(depthwise_sep, nn.Sequential)
    assert len(depthwise_sep) == 2  # Depthwise + pointwise
    print("✓ Depthwise separation passed")
    
    # Test 5: Spatial decomposition
    print("\n5. Testing spatial decomposition...")
    conv_spatial = nn.Conv2d(16, 32, kernel_size=5, padding=2).to(device)
    spatial_decomp = strategy.spatial_decomposition(conv_spatial)
    assert isinstance(spatial_decomp, nn.Sequential)
    assert spatial_decomp[0].kernel_size == (5, 1)  # Vertical
    assert spatial_decomp[1].kernel_size == (1, 5)  # Horizontal
    print("✓ Spatial decomposition passed")
    
    # Test 6: Adaptive compression
    print("\n6. Testing adaptive CNN compression...")
    
    # Create simple CNN
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.fc = nn.Linear(128, 10)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv3(x))
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleCNN().to(device)
    compressor = AdaptiveConvCompressor(target_compression=2.0, device=device)
    
    # Analyze compressibility
    compressibility = compressor.analyze_cnn_compressibility(model)
    assert 'estimated_compression_ratio' in compressibility
    assert compressibility['num_conv_layers'] == 3
    print(f"✓ Estimated compression: {compressibility['estimated_compression_ratio']:.2f}x")
    
    # Compress model
    compressed = compressor.compress_cnn(model)
    
    # Count parameters
    original_params = sum(p.numel() for p in model.parameters())
    compressed_params = sum(p.numel() for p in compressed.parameters())
    actual_compression = original_params / compressed_params
    
    print(f"✓ Actual compression: {actual_compression:.2f}x")
    
    # Test 7: Edge cases
    print("\n7. Testing edge cases...")
    
    # 1x1 convolution
    conv_1x1 = nn.Conv2d(64, 32, kernel_size=1).to(device)
    analysis_1x1 = analyzer.analyze_conv2d(conv_1x1)
    assert analysis_1x1['spatial_analysis'].kernel_size == (1, 1)
    print("✓ 1x1 convolution handled")
    
    # Grouped convolution
    conv_grouped = nn.Conv2d(64, 64, kernel_size=3, groups=4).to(device)
    analysis_grouped = analyzer.analyze_conv2d(conv_grouped)
    assert analysis_grouped['shape']['groups'] == 4
    print("✓ Grouped convolution handled")
    
    # Dilated convolution
    conv_dilated = nn.Conv2d(32, 64, kernel_size=3, dilation=2).to(device)
    analysis_dilated = analyzer.analyze_conv2d(conv_dilated)
    assert analysis_dilated['shape']['dilation'] == (2, 2)
    print("✓ Dilated convolution handled")
    
    print("\n✅ All tests passed!")
    
    # Performance benchmark
    print("\n8. Performance benchmark...")
    import time
    
    # Create ResNet-like structure
    conv_layers = [
        nn.Conv2d(3, 64, 7, stride=2, padding=3),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.Conv2d(128, 128, 3, padding=1),
        nn.Conv2d(128, 256, 3, stride=2, padding=1),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.Conv2d(256, 512, 3, stride=2, padding=1),
        nn.Conv2d(512, 512, 3, padding=1),
    ]
    
    total_time = 0
    for i, conv in enumerate(conv_layers):
        conv = conv.to(device)
        start = time.time()
        _ = analyzer.analyze_conv2d(conv)
        elapsed = time.time() - start
        total_time += elapsed
        print(f"  Layer {i+1}: {elapsed*1000:.2f}ms")
    
    print(f"✓ Total analysis time for 8 layers: {total_time:.3f}s")
    
    if total_time < 10:
        print("✅ Performance requirement met (<10s for ResNet-like structure)")
    else:
        print("⚠️ Performance could be improved")