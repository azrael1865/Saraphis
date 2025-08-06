"""
Dense/Linear layer analyzer for universal mathematical compression framework.
Analyzes PyTorch Linear layers to determine mathematical properties and compression suitability.
NO PLACEHOLDERS - PRODUCTION READY
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
from enum import Enum
import logging


class CompressionMethod(Enum):
    """Available compression methods"""
    TROPICAL = "tropical"
    PADIC = "padic"
    HYBRID = "hybrid"
    NONE = "none"


@dataclass
class RankAnalysis:
    """Rank analysis results for weight matrix"""
    effective_rank: int
    numerical_rank: int
    stable_rank: float
    rank_ratio: float  # effective_rank / min(m, n)
    singular_values: Optional[torch.Tensor] = None
    energy_threshold: float = 0.99
    numerical_tolerance: float = 1e-6
    
    def __post_init__(self):
        """Validate rank analysis results"""
        if self.effective_rank < 0:
            raise ValueError(f"Effective rank must be non-negative, got {self.effective_rank}")
        if self.numerical_rank < 0:
            raise ValueError(f"Numerical rank must be non-negative, got {self.numerical_rank}")
        if self.stable_rank < 0:
            raise ValueError(f"Stable rank must be non-negative, got {self.stable_rank}")
        if not (0.0 <= self.rank_ratio <= 1.0):
            raise ValueError(f"Rank ratio must be in [0, 1], got {self.rank_ratio}")
        if not (0.0 < self.energy_threshold <= 1.0):
            raise ValueError(f"Energy threshold must be in (0, 1], got {self.energy_threshold}")
        if self.numerical_tolerance <= 0:
            raise ValueError(f"Numerical tolerance must be positive, got {self.numerical_tolerance}")


@dataclass
class SparsityAnalysis:
    """Sparsity analysis results"""
    zero_ratio: float  # Exact zeros
    near_zero_ratio: float  # Near zeros with threshold
    near_zero_threshold: float
    block_sparsity_detected: bool
    block_sizes: Optional[List[Tuple[int, int]]] = None
    structured_pattern: Optional[str] = None  # 'diagonal', 'banded', 'block_diagonal', etc.
    sparsity_distribution: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Validate sparsity analysis results"""
        if not (0.0 <= self.zero_ratio <= 1.0):
            raise ValueError(f"Zero ratio must be in [0, 1], got {self.zero_ratio}")
        if not (0.0 <= self.near_zero_ratio <= 1.0):
            raise ValueError(f"Near zero ratio must be in [0, 1], got {self.near_zero_ratio}")
        if self.near_zero_threshold < 0:
            raise ValueError(f"Near zero threshold must be non-negative, got {self.near_zero_threshold}")


@dataclass
class NumericalAnalysis:
    """Numerical properties analysis"""
    condition_number: float
    dynamic_range: float  # max_abs / min_abs (excluding zeros)
    eigenvalue_spread: float  # max_eigenvalue / min_eigenvalue
    frobenius_norm: float
    spectral_norm: float  # largest singular value
    numerical_rank: int
    max_value: float
    min_value: float
    mean_value: float
    std_value: float
    
    def __post_init__(self):
        """Validate numerical analysis results"""
        if self.condition_number < 1.0:
            raise ValueError(f"Condition number must be >= 1, got {self.condition_number}")
        if self.dynamic_range < 1.0:
            raise ValueError(f"Dynamic range must be >= 1, got {self.dynamic_range}")
        if self.eigenvalue_spread < 1.0:
            raise ValueError(f"Eigenvalue spread must be >= 1, got {self.eigenvalue_spread}")
        if self.frobenius_norm < 0:
            raise ValueError(f"Frobenius norm must be non-negative, got {self.frobenius_norm}")
        if self.spectral_norm < 0:
            raise ValueError(f"Spectral norm must be non-negative, got {self.spectral_norm}")


@dataclass
class CompressionRecommendation:
    """Compression method recommendation"""
    method: CompressionMethod
    tropical_score: float  # 0-1
    padic_score: float  # 0-1
    hybrid_score: float  # 0-1
    confidence: float  # 0-1
    reasoning: str
    estimated_compression_ratio: float
    
    def __post_init__(self):
        """Validate compression recommendation"""
        if not isinstance(self.method, CompressionMethod):
            raise TypeError(f"Method must be CompressionMethod, got {type(self.method)}")
        scores = [self.tropical_score, self.padic_score, self.hybrid_score, self.confidence]
        for score in scores:
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"Scores must be in [0, 1], got {score}")
        if self.estimated_compression_ratio < 1.0:
            raise ValueError(f"Compression ratio must be >= 1, got {self.estimated_compression_ratio}")


@dataclass
class LayerAnalysisResult:
    """Complete analysis result for a layer"""
    layer_name: str
    layer_type: str
    parameter_count: int
    shape: Tuple[int, ...]
    rank_analysis: RankAnalysis
    sparsity_analysis: SparsityAnalysis
    numerical_analysis: NumericalAnalysis
    compression_recommendation: CompressionRecommendation
    analysis_time_ms: float
    device: str
    
    def __post_init__(self):
        """Validate layer analysis result"""
        if not self.layer_name:
            raise ValueError("Layer name cannot be empty")
        if not self.layer_type:
            raise ValueError("Layer type cannot be empty")
        if self.parameter_count < 0:
            raise ValueError(f"Parameter count must be non-negative, got {self.parameter_count}")
        if self.analysis_time_ms < 0:
            raise ValueError(f"Analysis time must be non-negative, got {self.analysis_time_ms}")


class DenseLayerAnalyzer:
    """Analyzer for dense/linear layers in neural networks"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize dense layer analyzer
        
        Args:
            config: Configuration dictionary with optional parameters:
                - device: torch.device or string (default: 'cpu')
                - energy_threshold: float in (0, 1] for rank computation (default: 0.99)
                - numerical_tolerance: float > 0 for numerical rank (default: 1e-6)
                - near_zero_threshold: float >= 0 for sparsity (default: 1e-8)
                - use_randomized_svd: bool for large matrices (default: True)
                - max_svd_size: int, use randomized SVD if larger (default: 1000)
                - block_size_threshold: int for block detection (default: 4)
        """
        self.config = config or {}
        self.logger = logging.getLogger('DenseLayerAnalyzer')
        
        # Device configuration
        device_str = self.config.get('device', 'cpu')
        if isinstance(device_str, str):
            self.device = torch.device(device_str)
        elif isinstance(device_str, torch.device):
            self.device = device_str
        else:
            raise TypeError(f"Device must be string or torch.device, got {type(device_str)}")
        
        # Analysis parameters
        self.energy_threshold = self.config.get('energy_threshold', 0.99)
        if not (0.0 < self.energy_threshold <= 1.0):
            raise ValueError(f"Energy threshold must be in (0, 1], got {self.energy_threshold}")
        
        self.numerical_tolerance = self.config.get('numerical_tolerance', 1e-6)
        if self.numerical_tolerance <= 0:
            raise ValueError(f"Numerical tolerance must be positive, got {self.numerical_tolerance}")
        
        self.near_zero_threshold = self.config.get('near_zero_threshold', 1e-8)
        if self.near_zero_threshold < 0:
            raise ValueError(f"Near zero threshold must be non-negative, got {self.near_zero_threshold}")
        
        self.use_randomized_svd = self.config.get('use_randomized_svd', True)
        self.max_svd_size = self.config.get('max_svd_size', 1000)
        if self.max_svd_size <= 0:
            raise ValueError(f"Max SVD size must be positive, got {self.max_svd_size}")
        
        self.block_size_threshold = self.config.get('block_size_threshold', 4)
        if self.block_size_threshold < 1:
            raise ValueError(f"Block size threshold must be >= 1, got {self.block_size_threshold}")
    
    def analyze_layer(self, module: nn.Linear, layer_name: str) -> LayerAnalysisResult:
        """
        Perform complete analysis of a linear layer
        
        Args:
            module: PyTorch Linear layer to analyze
            layer_name: Name/identifier for the layer
            
        Returns:
            LayerAnalysisResult with complete analysis
        """
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(module)}")
        if not layer_name:
            raise ValueError("Layer name cannot be empty")
        
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        else:
            import time
            cpu_start = time.perf_counter()
        
        # Get weight tensor
        weight = module.weight.detach()
        if weight.device != self.device:
            weight = weight.to(self.device)
        
        # Validate weight tensor
        if torch.isnan(weight).any():
            raise ValueError(f"Layer {layer_name} contains NaN values")
        if torch.isinf(weight).any():
            raise ValueError(f"Layer {layer_name} contains infinite values")
        
        # Get shape info
        out_features, in_features = weight.shape
        parameter_count = weight.numel()
        if module.bias is not None:
            parameter_count += module.bias.numel()
        
        # Perform analyses
        rank_analysis = self.compute_effective_rank(weight)
        sparsity_analysis = self.analyze_sparsity(weight)
        numerical_analysis = self.compute_numerical_properties(weight)
        
        # Get compression recommendation
        compression_recommendation = self.recommend_compression(
            rank_analysis, sparsity_analysis, numerical_analysis, weight.shape
        )
        
        # Compute timing
        if start_time:
            end_time.record()
            torch.cuda.synchronize()
            analysis_time_ms = start_time.elapsed_time(end_time)
        else:
            analysis_time_ms = (time.perf_counter() - cpu_start) * 1000
        
        return LayerAnalysisResult(
            layer_name=layer_name,
            layer_type='Linear',
            parameter_count=parameter_count,
            shape=(out_features, in_features),
            rank_analysis=rank_analysis,
            sparsity_analysis=sparsity_analysis,
            numerical_analysis=numerical_analysis,
            compression_recommendation=compression_recommendation,
            analysis_time_ms=analysis_time_ms,
            device=str(self.device)
        )
    
    def compute_effective_rank(self, weight: torch.Tensor) -> RankAnalysis:
        """
        Compute effective rank using SVD with energy threshold
        
        Args:
            weight: Weight matrix to analyze
            
        Returns:
            RankAnalysis with rank metrics
        """
        m, n = weight.shape
        min_dim = min(m, n)
        
        # Decide whether to use randomized SVD
        use_randomized = (
            self.use_randomized_svd and 
            max(m, n) > self.max_svd_size and
            min_dim > 100
        )
        
        if use_randomized:
            # Use randomized SVD for large matrices
            k = min(min_dim, max(100, min_dim // 2))  # Number of components
            U, S, V = self._randomized_svd(weight, k=k)
        else:
            # Full SVD for smaller matrices
            U, S, V = torch.linalg.svd(weight, full_matrices=False)
        
        # Ensure singular values are sorted (should be by default)
        S_sorted, _ = torch.sort(S, descending=True)
        
        # Compute energy (squared singular values)
        S_squared = S_sorted ** 2
        total_energy = S_squared.sum()
        
        if total_energy == 0:
            # Degenerate case: all zeros
            return RankAnalysis(
                effective_rank=0,
                numerical_rank=0,
                stable_rank=0.0,
                rank_ratio=0.0,
                singular_values=S_sorted,
                energy_threshold=self.energy_threshold,
                numerical_tolerance=self.numerical_tolerance
            )
        
        # Compute cumulative energy
        cumulative_energy = torch.cumsum(S_squared, dim=0) / total_energy
        
        # Find effective rank (energy-based)
        effective_rank = (cumulative_energy < self.energy_threshold).sum().item() + 1
        effective_rank = min(effective_rank, len(S_sorted))
        
        # Compute numerical rank (tolerance-based)
        numerical_rank = (S_sorted > self.numerical_tolerance * S_sorted[0]).sum().item()
        
        # Compute stable rank: ||A||_F^2 / ||A||_2^2
        frobenius_norm_sq = (weight ** 2).sum().item()
        spectral_norm_sq = S_sorted[0].item() ** 2 if len(S_sorted) > 0 else 0.0
        stable_rank = frobenius_norm_sq / spectral_norm_sq if spectral_norm_sq > 0 else 0.0
        
        # Compute rank ratio
        rank_ratio = effective_rank / min_dim
        
        return RankAnalysis(
            effective_rank=effective_rank,
            numerical_rank=numerical_rank,
            stable_rank=stable_rank,
            rank_ratio=rank_ratio,
            singular_values=S_sorted if len(S_sorted) <= 1000 else None,  # Limit storage
            energy_threshold=self.energy_threshold,
            numerical_tolerance=self.numerical_tolerance
        )
    
    def _randomized_svd(self, A: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomized SVD for large matrices
        
        Args:
            A: Matrix to decompose
            k: Number of singular values/vectors to compute
            
        Returns:
            U, S, V tensors (truncated)
        """
        m, n = A.shape
        k = min(k, min(m, n))
        
        # Generate random projection matrix
        if n > m:
            # Wide matrix: compute SVD of A.T instead
            return self._randomized_svd(A.t(), k)
        
        # Tall or square matrix
        omega = torch.randn(n, k + 10, device=A.device)  # Oversampling
        Y = A @ omega
        
        # QR decomposition for numerical stability
        Q, _ = torch.linalg.qr(Y)
        
        # Project A onto Q
        B = Q.t() @ A
        
        # SVD of smaller matrix
        U_tilde, S, Vt = torch.linalg.svd(B, full_matrices=False)
        
        # Recover full U
        U = Q @ U_tilde
        
        # Truncate to k components
        return U[:, :k], S[:k], Vt[:k, :]
    
    def analyze_sparsity(self, weight: torch.Tensor) -> SparsityAnalysis:
        """
        Analyze sparsity patterns in weight matrix
        
        Args:
            weight: Weight matrix to analyze
            
        Returns:
            SparsityAnalysis with sparsity metrics
        """
        total_elements = weight.numel()
        
        # Exact zero ratio
        zero_mask = weight == 0
        zero_count = zero_mask.sum().item()
        zero_ratio = zero_count / total_elements
        
        # Near-zero ratio
        near_zero_mask = torch.abs(weight) < self.near_zero_threshold
        near_zero_count = near_zero_mask.sum().item()
        near_zero_ratio = near_zero_count / total_elements
        
        # Detect structured patterns
        structured_pattern = self._detect_structured_pattern(weight, zero_mask)
        
        # Detect block sparsity
        block_sparsity_detected, block_sizes = self._detect_block_sparsity(weight, near_zero_mask)
        
        # Compute sparsity distribution
        sparsity_distribution = self._compute_sparsity_distribution(weight, near_zero_mask)
        
        return SparsityAnalysis(
            zero_ratio=zero_ratio,
            near_zero_ratio=near_zero_ratio,
            near_zero_threshold=self.near_zero_threshold,
            block_sparsity_detected=block_sparsity_detected,
            block_sizes=block_sizes,
            structured_pattern=structured_pattern,
            sparsity_distribution=sparsity_distribution
        )
    
    def _detect_structured_pattern(self, weight: torch.Tensor, zero_mask: torch.Tensor) -> Optional[str]:
        """Detect common structured sparsity patterns"""
        m, n = weight.shape
        
        # Check for diagonal pattern
        if m == n:
            diag_mask = torch.eye(m, device=weight.device, dtype=torch.bool)
            off_diag_zeros = zero_mask & ~diag_mask
            if off_diag_zeros.sum() > 0.9 * (m * n - m):
                return "diagonal"
        
        # Check for banded pattern
        bandwidth = self._estimate_bandwidth(weight, zero_mask)
        if bandwidth is not None and bandwidth < min(m, n) * 0.1:
            return "banded"
        
        # Check for block diagonal
        if self._is_block_diagonal(weight, zero_mask):
            return "block_diagonal"
        
        # Check for low-rank plus diagonal
        if self._is_low_rank_plus_diagonal(weight):
            return "low_rank_plus_diagonal"
        
        return None
    
    def _estimate_bandwidth(self, weight: torch.Tensor, zero_mask: torch.Tensor) -> Optional[int]:
        """Estimate bandwidth of potentially banded matrix"""
        m, n = weight.shape
        if m != n:
            return None
        
        # Check distance from diagonal for non-zero elements
        max_dist = 0
        for i in range(m):
            for j in range(n):
                if not zero_mask[i, j]:
                    max_dist = max(max_dist, abs(i - j))
        
        # If most elements are near diagonal
        total_nonzero = (~zero_mask).sum().item()
        near_diag_count = 0
        for k in range(-max_dist, max_dist + 1):
            diag = torch.diagonal(weight, k)
            near_diag_count += (~torch.diagonal(zero_mask, k)).sum().item()
        
        if near_diag_count > 0.95 * total_nonzero:
            return max_dist
        
        return None
    
    def _is_block_diagonal(self, weight: torch.Tensor, zero_mask: torch.Tensor) -> bool:
        """Check if matrix has block diagonal structure"""
        m, n = weight.shape
        if m != n:
            return False
        
        # Simple heuristic: divide into blocks and check
        block_size = max(4, min(m // 4, 32))
        num_blocks = m // block_size
        
        if num_blocks < 2:
            return False
        
        # Check off-diagonal blocks
        off_block_zeros = 0
        total_off_block = 0
        
        for i in range(num_blocks):
            for j in range(num_blocks):
                if i != j:
                    row_start = i * block_size
                    row_end = min((i + 1) * block_size, m)
                    col_start = j * block_size
                    col_end = min((j + 1) * block_size, n)
                    
                    block = zero_mask[row_start:row_end, col_start:col_end]
                    off_block_zeros += block.sum().item()
                    total_off_block += block.numel()
        
        # If most off-diagonal blocks are zero
        return (off_block_zeros / total_off_block) > 0.95 if total_off_block > 0 else False
    
    def _is_low_rank_plus_diagonal(self, weight: torch.Tensor) -> bool:
        """Check if matrix is low-rank plus diagonal"""
        m, n = weight.shape
        if m != n:
            return False
        
        # Remove diagonal
        weight_no_diag = weight - torch.diag(torch.diagonal(weight))
        
        # Check rank of off-diagonal part
        try:
            _, S, _ = torch.linalg.svd(weight_no_diag, full_matrices=False)
            rank = (S > self.numerical_tolerance * S[0]).sum().item()
            return rank < min(m, n) * 0.1
        except:
            return False
    
    def _detect_block_sparsity(self, weight: torch.Tensor, sparse_mask: torch.Tensor) -> Tuple[bool, Optional[List[Tuple[int, int]]]]:
        """Detect block sparsity patterns"""
        m, n = weight.shape
        
        # Try different block sizes
        block_sizes_to_try = [4, 8, 16, 32]
        best_block_size = None
        best_sparsity = 0.0
        
        for block_size in block_sizes_to_try:
            if block_size > min(m, n) // 2:
                continue
            
            # Count sparse blocks
            sparse_blocks = 0
            total_blocks = 0
            
            for i in range(0, m, block_size):
                for j in range(0, n, block_size):
                    row_end = min(i + block_size, m)
                    col_end = min(j + block_size, n)
                    
                    block = sparse_mask[i:row_end, j:col_end]
                    if block.all():  # All elements in block are sparse
                        sparse_blocks += 1
                    total_blocks += 1
            
            block_sparsity = sparse_blocks / total_blocks if total_blocks > 0 else 0
            
            if block_sparsity > best_sparsity and block_sparsity > 0.1:
                best_sparsity = block_sparsity
                best_block_size = block_size
        
        if best_block_size is not None:
            return True, [(best_block_size, best_block_size)]
        
        return False, None
    
    def _compute_sparsity_distribution(self, weight: torch.Tensor, sparse_mask: torch.Tensor) -> Dict[str, float]:
        """Compute distribution of sparsity across matrix"""
        m, n = weight.shape
        
        # Row-wise sparsity
        row_sparsity = sparse_mask.float().mean(dim=1)
        
        # Column-wise sparsity
        col_sparsity = sparse_mask.float().mean(dim=0)
        
        # Quadrant sparsity
        mid_m, mid_n = m // 2, n // 2
        quadrants = {
            'top_left': sparse_mask[:mid_m, :mid_n].float().mean().item(),
            'top_right': sparse_mask[:mid_m, mid_n:].float().mean().item(),
            'bottom_left': sparse_mask[mid_m:, :mid_n].float().mean().item(),
            'bottom_right': sparse_mask[mid_m:, mid_n:].float().mean().item()
        }
        
        return {
            'row_mean': row_sparsity.mean().item(),
            'row_std': row_sparsity.std().item(),
            'col_mean': col_sparsity.mean().item(),
            'col_std': col_sparsity.std().item(),
            **quadrants
        }
    
    def compute_numerical_properties(self, weight: torch.Tensor) -> NumericalAnalysis:
        """
        Compute numerical properties of weight matrix
        
        Args:
            weight: Weight matrix to analyze
            
        Returns:
            NumericalAnalysis with numerical metrics
        """
        # Basic statistics
        max_value = weight.max().item()
        min_value = weight.min().item()
        mean_value = weight.mean().item()
        std_value = weight.std().item()
        
        # Compute norms
        frobenius_norm = torch.norm(weight, p='fro').item()
        
        # Compute condition number and spectral norm via SVD
        try:
            # For large matrices, use randomized SVD
            if max(weight.shape) > self.max_svd_size and self.use_randomized_svd:
                k = min(100, min(weight.shape))
                _, S, _ = self._randomized_svd(weight, k)
            else:
                _, S, _ = torch.linalg.svd(weight, full_matrices=False)
            
            spectral_norm = S[0].item() if len(S) > 0 else 0.0
            
            # Condition number
            S_positive = S[S > self.numerical_tolerance]
            if len(S_positive) > 0:
                condition_number = S_positive[0].item() / S_positive[-1].item()
            else:
                condition_number = float('inf')
            
            # Numerical rank
            numerical_rank = len(S_positive)
        except Exception as e:
            self.logger.warning(f"SVD computation failed: {e}")
            spectral_norm = torch.norm(weight, p=2).item()
            condition_number = float('inf')
            numerical_rank = min(weight.shape)
        
        # Dynamic range (excluding zeros)
        non_zero_mask = weight != 0
        if non_zero_mask.any():
            non_zero_values = torch.abs(weight[non_zero_mask])
            dynamic_range = non_zero_values.max().item() / non_zero_values.min().item()
        else:
            dynamic_range = 1.0
        
        # Eigenvalue spread for square matrices
        eigenvalue_spread = 1.0
        if weight.shape[0] == weight.shape[1]:
            try:
                # Compute eigenvalues of W @ W.T (always positive)
                WWT = weight @ weight.t()
                eigenvalues = torch.linalg.eigvalsh(WWT)
                positive_eigs = eigenvalues[eigenvalues > self.numerical_tolerance]
                if len(positive_eigs) > 0:
                    eigenvalue_spread = positive_eigs[-1].item() / positive_eigs[0].item()
            except:
                eigenvalue_spread = condition_number  # Fallback
        else:
            eigenvalue_spread = condition_number  # Use condition number for non-square
        
        return NumericalAnalysis(
            condition_number=condition_number,
            dynamic_range=dynamic_range,
            eigenvalue_spread=eigenvalue_spread,
            frobenius_norm=frobenius_norm,
            spectral_norm=spectral_norm,
            numerical_rank=numerical_rank,
            max_value=max_value,
            min_value=min_value,
            mean_value=mean_value,
            std_value=std_value
        )
    
    def score_tropical_suitability(self, rank_analysis: RankAnalysis, 
                                  sparsity_analysis: SparsityAnalysis,
                                  numerical_analysis: NumericalAnalysis,
                                  shape: Tuple[int, int]) -> float:
        """
        Score suitability for tropical (max-plus) compression
        
        Tropical compression works well for:
        - Low-rank matrices (can be approximated with few max-plus rank-1 terms)
        - Matrices with clear dominant paths/structures
        - Sparse matrices with specific patterns
        
        Args:
            rank_analysis: Rank analysis results
            sparsity_analysis: Sparsity analysis results
            numerical_analysis: Numerical analysis results
            shape: Matrix shape
            
        Returns:
            Score between 0 and 1
        """
        score = 0.0
        
        # Low rank is excellent for tropical
        if rank_analysis.rank_ratio < 0.3:
            score += 0.4
        elif rank_analysis.rank_ratio < 0.5:
            score += 0.25
        elif rank_analysis.rank_ratio < 0.7:
            score += 0.1
        
        # Stable rank much less than min dimension
        min_dim = min(shape)
        if rank_analysis.stable_rank < 0.1 * min_dim:
            score += 0.2
        elif rank_analysis.stable_rank < 0.3 * min_dim:
            score += 0.1
        
        # High sparsity helps tropical
        if sparsity_analysis.near_zero_ratio > 0.8:
            score += 0.2
        elif sparsity_analysis.near_zero_ratio > 0.6:
            score += 0.1
        
        # Structured patterns are good
        if sparsity_analysis.structured_pattern in ['diagonal', 'banded', 'block_diagonal']:
            score += 0.15
        
        # High dynamic range is handled well by tropical
        if numerical_analysis.dynamic_range > 100:
            score += 0.1
        elif numerical_analysis.dynamic_range > 10:
            score += 0.05
        
        # Poor conditioning suggests low-rank structure
        if numerical_analysis.condition_number > 100:
            score += 0.05
        
        return min(score, 1.0)
    
    def score_padic_suitability(self, rank_analysis: RankAnalysis,
                               sparsity_analysis: SparsityAnalysis,
                               numerical_analysis: NumericalAnalysis,
                               shape: Tuple[int, int]) -> float:
        """
        Score suitability for p-adic compression
        
        P-adic compression works well for:
        - Matrices requiring high numerical precision
        - Dense matrices with specific numerical patterns
        - Matrices with repeating fractional values
        
        Args:
            rank_analysis: Rank analysis results
            sparsity_analysis: Sparsity analysis results
            numerical_analysis: Numerical analysis results
            shape: Matrix shape
            
        Returns:
            Score between 0 and 1
        """
        score = 0.0
        
        # Full rank matrices need precision
        if rank_analysis.rank_ratio > 0.8:
            score += 0.3
        elif rank_analysis.rank_ratio > 0.6:
            score += 0.2
        
        # Dense matrices suit p-adic
        if sparsity_analysis.zero_ratio < 0.2:
            score += 0.25
        elif sparsity_analysis.zero_ratio < 0.4:
            score += 0.15
        
        # Moderate dynamic range is good for p-adic
        if 2 < numerical_analysis.dynamic_range < 50:
            score += 0.2
        elif numerical_analysis.dynamic_range <= 2:
            score += 0.1
        
        # Well-conditioned matrices
        if numerical_analysis.condition_number < 10:
            score += 0.15
        elif numerical_analysis.condition_number < 50:
            score += 0.1
        
        # Small standard deviation suggests repeated values
        if abs(numerical_analysis.std_value) < 0.1 * abs(numerical_analysis.mean_value):
            score += 0.1
        
        # Reasonable matrix size (p-adic has overhead)
        if shape[0] * shape[1] > 10000:
            score += 0.1
        
        return min(score, 1.0)
    
    def recommend_compression(self, rank_analysis: RankAnalysis,
                            sparsity_analysis: SparsityAnalysis,
                            numerical_analysis: NumericalAnalysis,
                            shape: Tuple[int, int]) -> CompressionRecommendation:
        """
        Recommend compression method based on analyses
        
        Args:
            rank_analysis: Rank analysis results
            sparsity_analysis: Sparsity analysis results  
            numerical_analysis: Numerical analysis results
            shape: Matrix shape
            
        Returns:
            CompressionRecommendation with method and reasoning
        """
        # Score each method
        tropical_score = self.score_tropical_suitability(
            rank_analysis, sparsity_analysis, numerical_analysis, shape
        )
        padic_score = self.score_padic_suitability(
            rank_analysis, sparsity_analysis, numerical_analysis, shape
        )
        
        # Hybrid score when both have merit
        if tropical_score > 0.3 and padic_score > 0.3:
            hybrid_score = 0.7 * max(tropical_score, padic_score) + 0.3 * min(tropical_score, padic_score)
        else:
            hybrid_score = 0.0
        
        # Determine best method
        scores = {
            CompressionMethod.TROPICAL: tropical_score,
            CompressionMethod.PADIC: padic_score,
            CompressionMethod.HYBRID: hybrid_score
        }
        
        # Need minimum score to recommend compression
        max_score = max(scores.values())
        if max_score < 0.2:
            method = CompressionMethod.NONE
            reasoning = "Matrix properties do not strongly favor compression"
            estimated_ratio = 1.0
        else:
            method = max(scores.items(), key=lambda x: x[1])[0]
            
            # Generate reasoning
            reasons = []
            if method == CompressionMethod.TROPICAL:
                if rank_analysis.rank_ratio < 0.5:
                    reasons.append(f"low rank ({rank_analysis.effective_rank}/{min(shape)})")
                if sparsity_analysis.near_zero_ratio > 0.6:
                    reasons.append(f"high sparsity ({sparsity_analysis.near_zero_ratio:.1%})")
                if numerical_analysis.dynamic_range > 10:
                    reasons.append(f"high dynamic range ({numerical_analysis.dynamic_range:.1f})")
                reasoning = f"Tropical compression recommended: {', '.join(reasons)}"
                estimated_ratio = 1.5 + (1 - rank_analysis.rank_ratio) * 2
                
            elif method == CompressionMethod.PADIC:
                if rank_analysis.rank_ratio > 0.6:
                    reasons.append(f"high rank ({rank_analysis.effective_rank}/{min(shape)})")
                if sparsity_analysis.zero_ratio < 0.3:
                    reasons.append(f"dense matrix ({1-sparsity_analysis.zero_ratio:.1%} non-zero)")
                if numerical_analysis.condition_number < 50:
                    reasons.append(f"well-conditioned ({numerical_analysis.condition_number:.1f})")
                reasoning = f"P-adic compression recommended: {', '.join(reasons)}"
                estimated_ratio = 1.3 + (1 - sparsity_analysis.zero_ratio) * 0.5
                
            else:  # HYBRID
                reasons.append(f"mixed properties favor hybrid approach")
                reasons.append(f"tropical score {tropical_score:.2f}, p-adic score {padic_score:.2f}")
                reasoning = f"Hybrid compression recommended: {', '.join(reasons)}"
                estimated_ratio = 1.8 + min(tropical_score, padic_score)
        
        # Confidence based on score strength and consistency
        confidence = max_score
        if method == CompressionMethod.HYBRID:
            confidence *= 0.9  # Slightly less confident about hybrid
        
        return CompressionRecommendation(
            method=method,
            tropical_score=tropical_score,
            padic_score=padic_score,
            hybrid_score=hybrid_score,
            confidence=confidence,
            reasoning=reasoning,
            estimated_compression_ratio=estimated_ratio
        )