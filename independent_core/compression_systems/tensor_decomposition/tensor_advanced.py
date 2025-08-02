"""
Advanced Tensor Decomposition Module for Saraphis Compression System

Implements HOSVD, Tensor-Ring decomposition, advanced rank optimization,
and GPU acceleration for tensor operations.
"""

import torch
import torch.nn as nn
import torch.cuda as cuda
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
from abc import ABC, abstractmethod
import math
import logging
from datetime import datetime
import warnings
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import scipy.optimize
import scipy.linalg
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
import optuna
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel
from gpytorch.distributions import MultivariateNormal

# Import base classes from tensor_core.py
from .tensor_core import (
    DecompositionMethod, TensorDecomposition, DecompositionType,
    TensorValidator, TensorCompressionSystem, TensorRankOptimizer
)

# Import GPU memory management
from ..gpu_memory.gpu_memory_core import GPUMemoryManager, StreamManager, MemoryOptimizer

# Import training manager components
from ....training_manager import TrainingManager, TrainingConfig, TrainingSession

# Import GAC system components
from ....gac_system.gac_components import GradientCompressionComponent

# Configure logging
logger = logging.getLogger(__name__)


class HOSVDDecomposer(DecompositionMethod):
    """
    Higher-Order Singular Value Decomposition (HOSVD) implementation.
    
    HOSVD decomposes a tensor into a core tensor multiplied by a matrix
    along each mode, providing optimal approximation in the least-squares sense.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize HOSVD decomposer with configuration."""
        super().__init__(config)
        
        # HOSVD-specific parameters
        self.incremental = self.config.get('incremental', False)
        self.truncation_method = self.config.get('truncation_method', 'energy')
        self.energy_threshold = self.config.get('energy_threshold', 0.99)
        self.use_randomized_svd = self.config.get('use_randomized_svd', True)
        self.oversampling_factor = self.config.get('oversampling_factor', 1.1)
        
        # Performance tracking
        self.performance_metrics = {
            'decomposition_time': [],
            'reconstruction_time': [],
            'compression_ratio': [],
            'reconstruction_error': [],
            'memory_usage': []
        }
        
        # Validation
        self.validator = TensorValidator()
        
        logger.info(f"Initialized HOSVDDecomposer with config: {self.config}")
    
    def decompose(self, tensor: torch.Tensor, ranks: List[int]) -> TensorDecomposition:
        """
        Perform HOSVD decomposition on the input tensor.
        
        Args:
            tensor: Input tensor to decompose
            ranks: List of ranks for each mode
            
        Returns:
            TensorDecomposition object containing the decomposed components
            
        Raises:
            ValueError: If ranks are invalid or tensor is invalid
            RuntimeError: If decomposition fails
        """
        start_time = time.time()
        initial_memory = torch.cuda.memory_allocated() if tensor.is_cuda else 0
        
        try:
            # Validate inputs
            if not self.validator.validate_tensor(tensor):
                raise ValueError("Invalid input tensor")
            
            if len(ranks) != tensor.ndim:
                raise ValueError(f"Number of ranks ({len(ranks)}) must match tensor dimensions ({tensor.ndim})")
            
            # Validate ranks
            for i, (rank, dim_size) in enumerate(zip(ranks, tensor.shape)):
                if rank <= 0 or rank > dim_size:
                    raise ValueError(f"Invalid rank {rank} for dimension {i} with size {dim_size}")
            
            # Store original shape and device
            original_shape = tensor.shape
            device = tensor.device
            
            # Initialize factor matrices
            factor_matrices = []
            
            # Compute SVD for each mode
            working_tensor = tensor.clone()
            
            for mode in range(tensor.ndim):
                # Unfold tensor along current mode
                unfolded = self._unfold_tensor(working_tensor, mode)
                
                # Compute SVD
                if self.use_randomized_svd and min(unfolded.shape) > 100:
                    U, S, Vt = self._randomized_svd(unfolded, ranks[mode])
                else:
                    U, S, Vt = torch.svd(unfolded)
                
                # Truncate to specified rank
                U_truncated = U[:, :ranks[mode]]
                S_truncated = S[:ranks[mode]]
                
                # Apply truncation based on energy threshold if specified
                if self.truncation_method == 'energy':
                    energy_ratio = torch.cumsum(S_truncated**2, dim=0) / torch.sum(S**2)
                    cutoff_idx = torch.where(energy_ratio >= self.energy_threshold)[0]
                    if len(cutoff_idx) > 0:
                        actual_rank = min(cutoff_idx[0].item() + 1, ranks[mode])
                        U_truncated = U_truncated[:, :actual_rank]
                        S_truncated = S_truncated[:actual_rank]
                
                factor_matrices.append(U_truncated)
                
                # Update working tensor for next mode
                if mode < tensor.ndim - 1:
                    working_tensor = self._fold_tensor(
                        U_truncated.t() @ unfolded,
                        mode,
                        original_shape
                    )
            
            # Compute core tensor
            core_tensor = working_tensor
            
            # Create decomposition object
            decomposition = TensorDecomposition(
                type=DecompositionType.TUCKER,  # HOSVD is a special case of Tucker
                components={'core': core_tensor, 'factors': factor_matrices},
                original_shape=original_shape,
                compression_ratio=self._compute_compression_ratio(tensor, core_tensor, factor_matrices),
                metadata={
                    'method': 'HOSVD',
                    'ranks': ranks,
                    'actual_ranks': [f.shape[1] for f in factor_matrices],
                    'truncation_method': self.truncation_method,
                    'energy_threshold': self.energy_threshold
                }
            )
            
            # Update performance metrics
            decomposition_time = time.time() - start_time
            memory_usage = (torch.cuda.memory_allocated() if tensor.is_cuda else 0) - initial_memory
            
            self.performance_metrics['decomposition_time'].append(decomposition_time)
            self.performance_metrics['memory_usage'].append(memory_usage)
            self.performance_metrics['compression_ratio'].append(decomposition.compression_ratio)
            
            logger.info(f"HOSVD decomposition completed in {decomposition_time:.3f}s, "
                       f"compression ratio: {decomposition.compression_ratio:.3f}")
            
            return decomposition
            
        except Exception as e:
            logger.error(f"HOSVD decomposition failed: {str(e)}")
            raise RuntimeError(f"HOSVD decomposition failed: {str(e)}")
    
    def reconstruct(self, decomposition: TensorDecomposition) -> torch.Tensor:
        """
        Reconstruct tensor from HOSVD decomposition.
        
        Args:
            decomposition: TensorDecomposition object
            
        Returns:
            Reconstructed tensor
            
        Raises:
            ValueError: If decomposition is invalid
            RuntimeError: If reconstruction fails
        """
        start_time = time.time()
        
        try:
            # Validate decomposition
            if decomposition.type not in [DecompositionType.TUCKER, DecompositionType.CP]:
                raise ValueError(f"Invalid decomposition type for HOSVD: {decomposition.type}")
            
            if 'core' not in decomposition.components or 'factors' not in decomposition.components:
                raise ValueError("Invalid decomposition components for HOSVD")
            
            core_tensor = decomposition.components['core']
            factor_matrices = decomposition.components['factors']
            
            # Reconstruct by multiplying core tensor with factor matrices
            reconstructed = core_tensor
            
            for mode, factor_matrix in enumerate(factor_matrices):
                # Apply factor matrix to the appropriate mode
                reconstructed = self._mode_product(reconstructed, factor_matrix, mode)
            
            # Ensure correct shape
            if reconstructed.shape != decomposition.original_shape:
                raise RuntimeError(f"Reconstruction shape mismatch: {reconstructed.shape} vs {decomposition.original_shape}")
            
            # Update performance metrics
            reconstruction_time = time.time() - start_time
            self.performance_metrics['reconstruction_time'].append(reconstruction_time)
            
            # Calculate reconstruction error if original tensor is available
            if hasattr(decomposition, 'original_tensor'):
                error = torch.norm(decomposition.original_tensor - reconstructed) / torch.norm(decomposition.original_tensor)
                self.performance_metrics['reconstruction_error'].append(error.item())
                logger.info(f"HOSVD reconstruction error: {error.item():.6f}")
            
            logger.info(f"HOSVD reconstruction completed in {reconstruction_time:.3f}s")
            
            return reconstructed
            
        except Exception as e:
            logger.error(f"HOSVD reconstruction failed: {str(e)}")
            raise RuntimeError(f"HOSVD reconstruction failed: {str(e)}")
    
    def _unfold_tensor(self, tensor: torch.Tensor, mode: int) -> torch.Tensor:
        """Unfold tensor along specified mode."""
        shape = list(tensor.shape)
        n = shape[mode]
        
        # Move mode to front
        perm = [mode] + [i for i in range(len(shape)) if i != mode]
        tensor_permuted = tensor.permute(perm)
        
        # Reshape to matrix
        return tensor_permuted.reshape(n, -1)
    
    def _fold_tensor(self, matrix: torch.Tensor, mode: int, shape: Tuple[int, ...]) -> torch.Tensor:
        """Fold matrix back into tensor along specified mode."""
        # Compute target shape after folding
        target_shape = list(shape)
        target_shape[mode] = matrix.shape[0]
        prod = 1
        for i, s in enumerate(shape):
            if i != mode:
                prod *= s
        target_shape = [target_shape[mode]] + [s for i, s in enumerate(shape) if i != mode]
        
        # Reshape matrix
        tensor_folded = matrix.reshape(target_shape)
        
        # Permute back to original order
        perm = list(range(1, mode + 1)) + [0] + list(range(mode + 1, len(shape)))
        return tensor_folded.permute(perm)
    
    def _mode_product(self, tensor: torch.Tensor, matrix: torch.Tensor, mode: int) -> torch.Tensor:
        """Compute mode product of tensor and matrix."""
        # Unfold tensor
        unfolded = self._unfold_tensor(tensor, mode)
        
        # Matrix multiply
        result = matrix @ unfolded
        
        # Fold back
        new_shape = list(tensor.shape)
        new_shape[mode] = matrix.shape[0]
        
        return self._fold_tensor(result, mode, tuple(new_shape))
    
    def _randomized_svd(self, matrix: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute randomized SVD for large matrices."""
        m, n = matrix.shape
        k = min(rank, min(m, n))
        
        # Oversampling parameter
        p = int(k * self.oversampling_factor) - k
        l = k + p
        
        # Random sampling matrix
        omega = torch.randn(n, l, device=matrix.device, dtype=matrix.dtype)
        
        # Form sample matrix
        Y = matrix @ omega
        
        # Orthogonalize
        Q, _ = torch.qr(Y)
        
        # Project to lower dimension
        B = Q.t() @ matrix
        
        # SVD of smaller matrix
        U_tilde, S, Vt = torch.svd(B)
        
        # Recover full decomposition
        U = Q @ U_tilde
        
        return U[:, :k], S[:k], Vt[:k, :]
    
    def _compute_compression_ratio(self, original: torch.Tensor, core: torch.Tensor, 
                                 factors: List[torch.Tensor]) -> float:
        """Compute compression ratio."""
        original_size = original.numel()
        compressed_size = core.numel() + sum(f.numel() for f in factors)
        return compressed_size / original_size


class TensorRingDecomposer(DecompositionMethod):
    """
    Tensor-Ring (TR) decomposition implementation.
    
    TR decomposition represents a tensor as a sequence of 3D tensors
    connected in a ring structure, providing efficient representation
    for high-dimensional tensors.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Tensor-Ring decomposer with configuration."""
        super().__init__(config)
        
        # TR-specific parameters
        self.initialization_method = self.config.get('initialization_method', 'random')
        self.optimization_method = self.config.get('optimization_method', 'als')
        self.regularization = self.config.get('regularization', 1e-4)
        self.convergence_criterion = self.config.get('convergence_criterion', 'relative')
        
        # Performance tracking
        self.performance_metrics = {
            'decomposition_time': [],
            'reconstruction_time': [],
            'compression_ratio': [],
            'reconstruction_error': [],
            'iterations': []
        }
        
        # Validation
        self.validator = TensorValidator()
        
        logger.info(f"Initialized TensorRingDecomposer with config: {self.config}")
    
    def decompose(self, tensor: torch.Tensor, ranks: List[int]) -> TensorDecomposition:
        """
        Perform Tensor-Ring decomposition on the input tensor.
        
        Args:
            tensor: Input tensor to decompose
            ranks: List of TR-ranks (boundary dimensions)
            
        Returns:
            TensorDecomposition object containing TR cores
            
        Raises:
            ValueError: If ranks are invalid or tensor is invalid
            RuntimeError: If decomposition fails
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if not self.validator.validate_tensor(tensor):
                raise ValueError("Invalid input tensor")
            
            # Validate TR ranks
            if len(ranks) != tensor.ndim + 1:
                raise ValueError(f"Number of TR-ranks ({len(ranks)}) must be tensor.ndim + 1 ({tensor.ndim + 1})")
            
            if ranks[0] != ranks[-1]:
                raise ValueError(f"First and last TR-ranks must be equal for ring structure: {ranks[0]} != {ranks[-1]}")
            
            # Initialize TR cores
            cores = self._initialize_cores(tensor, ranks)
            
            # Optimize cores using specified method
            if self.optimization_method == 'als':
                cores, iterations = self._optimize_als(tensor, cores, ranks)
            elif self.optimization_method == 'gradient':
                cores, iterations = self._optimize_gradient(tensor, cores, ranks)
            else:
                raise ValueError(f"Unknown optimization method: {self.optimization_method}")
            
            # Create decomposition object
            decomposition = TensorDecomposition(
                type=DecompositionType.CP,  # Using CP as placeholder for TR
                components={'cores': cores},
                original_shape=tensor.shape,
                compression_ratio=self._compute_compression_ratio(tensor, cores),
                metadata={
                    'method': 'TensorRing',
                    'ranks': ranks,
                    'iterations': iterations,
                    'optimization_method': self.optimization_method
                }
            )
            
            # Update performance metrics
            decomposition_time = time.time() - start_time
            self.performance_metrics['decomposition_time'].append(decomposition_time)
            self.performance_metrics['compression_ratio'].append(decomposition.compression_ratio)
            self.performance_metrics['iterations'].append(iterations)
            
            # Calculate reconstruction error
            reconstructed = self.reconstruct(decomposition)
            error = torch.norm(tensor - reconstructed) / torch.norm(tensor)
            self.performance_metrics['reconstruction_error'].append(error.item())
            
            logger.info(f"TR decomposition completed in {decomposition_time:.3f}s, "
                       f"iterations: {iterations}, error: {error.item():.6f}")
            
            return decomposition
            
        except Exception as e:
            logger.error(f"Tensor-Ring decomposition failed: {str(e)}")
            raise RuntimeError(f"Tensor-Ring decomposition failed: {str(e)}")
    
    def reconstruct(self, decomposition: TensorDecomposition) -> torch.Tensor:
        """
        Reconstruct tensor from Tensor-Ring decomposition.
        
        Args:
            decomposition: TensorDecomposition object
            
        Returns:
            Reconstructed tensor
            
        Raises:
            ValueError: If decomposition is invalid
            RuntimeError: If reconstruction fails
        """
        start_time = time.time()
        
        try:
            # Validate decomposition
            if 'cores' not in decomposition.components:
                raise ValueError("Invalid decomposition components for Tensor-Ring")
            
            cores = decomposition.components['cores']
            
            # Contract TR cores to reconstruct tensor
            result = cores[0]
            
            for i in range(1, len(cores)):
                # Contract along the connecting dimension
                result = torch.tensordot(result, cores[i], dims=([-1], [0]))
            
            # Trace over first and last dimensions (ring closure)
            n_dims = len(result.shape)
            if n_dims > len(decomposition.original_shape):
                # Take diagonal to close the ring
                result = torch.diagonal(result, dim1=0, dim2=-1)
                # Move diagonal dimension to correct position
                perm = list(range(1, result.ndim)) + [0]
                result = result.permute(perm)
            
            # Reshape to original shape
            result = result.reshape(decomposition.original_shape)
            
            # Update performance metrics
            reconstruction_time = time.time() - start_time
            self.performance_metrics['reconstruction_time'].append(reconstruction_time)
            
            logger.info(f"TR reconstruction completed in {reconstruction_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Tensor-Ring reconstruction failed: {str(e)}")
            raise RuntimeError(f"Tensor-Ring reconstruction failed: {str(e)}")
    
    def _initialize_cores(self, tensor: torch.Tensor, ranks: List[int]) -> List[torch.Tensor]:
        """Initialize TR cores using specified method."""
        cores = []
        shape = tensor.shape
        device = tensor.device
        dtype = tensor.dtype
        
        for i in range(len(shape)):
            core_shape = (ranks[i], shape[i], ranks[i + 1])
            
            if self.initialization_method == 'random':
                # Random initialization with proper scaling
                scale = 1.0 / math.sqrt(np.prod(ranks))
                core = torch.randn(core_shape, device=device, dtype=dtype) * scale
            elif self.initialization_method == 'svd':
                # SVD-based initialization
                core = self._svd_initialization(tensor, i, ranks)
            else:
                raise ValueError(f"Unknown initialization method: {self.initialization_method}")
            
            cores.append(core)
        
        return cores
    
    def _svd_initialization(self, tensor: torch.Tensor, mode: int, ranks: List[int]) -> torch.Tensor:
        """Initialize TR core using SVD."""
        # Reshape tensor for SVD
        shape = tensor.shape
        left_dims = shape[:mode]
        right_dims = shape[mode + 1:]
        
        left_size = np.prod(left_dims) if left_dims else 1
        right_size = np.prod(right_dims) if right_dims else 1
        
        matrix = tensor.reshape(left_size * shape[mode], right_size)
        
        # Compute truncated SVD
        U, S, Vt = torch.svd(matrix)
        rank = min(ranks[mode], ranks[mode + 1], U.shape[1])
        
        # Form core
        core = U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]
        core = core.reshape(ranks[mode], shape[mode], ranks[mode + 1])
        
        return core
    
    def _optimize_als(self, tensor: torch.Tensor, cores: List[torch.Tensor], 
                     ranks: List[int]) -> Tuple[List[torch.Tensor], int]:
        """Optimize TR cores using Alternating Least Squares."""
        prev_error = float('inf')
        
        for iteration in range(self.max_iterations):
            # Update each core
            for i in range(len(cores)):
                # Compute environment tensor
                env = self._compute_environment(cores, i)
                
                # Solve least squares problem
                cores[i] = self._solve_core_ls(tensor, env, i, ranks)
            
            # Check convergence
            reconstructed = self.reconstruct(TensorDecomposition(
                type=DecompositionType.CP,
                components={'cores': cores},
                original_shape=tensor.shape,
                compression_ratio=0
            ))
            
            error = torch.norm(tensor - reconstructed) / torch.norm(tensor)
            
            if self.convergence_criterion == 'relative':
                if abs(prev_error - error) / prev_error < self.tolerance:
                    break
            else:
                if error < self.tolerance:
                    break
            
            prev_error = error
        
        return cores, iteration + 1
    
    def _optimize_gradient(self, tensor: torch.Tensor, cores: List[torch.Tensor], 
                          ranks: List[int]) -> Tuple[List[torch.Tensor], int]:
        """Optimize TR cores using gradient descent."""
        # Set requires_grad for cores
        for core in cores:
            core.requires_grad = True
        
        optimizer = torch.optim.Adam(cores, lr=0.01)
        
        for iteration in range(self.max_iterations):
            optimizer.zero_grad()
            
            # Reconstruct tensor
            reconstructed = self.reconstruct(TensorDecomposition(
                type=DecompositionType.CP,
                components={'cores': cores},
                original_shape=tensor.shape,
                compression_ratio=0
            ))
            
            # Compute loss
            loss = torch.norm(tensor - reconstructed) ** 2
            
            # Add regularization
            if self.regularization > 0:
                reg_loss = sum(self.regularization * torch.norm(core) ** 2 for core in cores)
                loss = loss + reg_loss
            
            # Backward pass
            loss.backward()
            
            # Update cores
            optimizer.step()
            
            # Check convergence
            error = math.sqrt(loss.item()) / torch.norm(tensor).item()
            if error < self.tolerance:
                break
        
        return cores, iteration + 1
    
    def _compute_environment(self, cores: List[torch.Tensor], core_idx: int) -> torch.Tensor:
        """Compute environment tensor for ALS update."""
        # Left contraction
        left = torch.eye(cores[0].shape[0], device=cores[0].device)
        for i in range(core_idx):
            left = torch.tensordot(left, cores[i], dims=([1], [0]))
        
        # Right contraction
        right = torch.eye(cores[-1].shape[-1], device=cores[-1].device)
        for i in range(len(cores) - 1, core_idx, -1):
            right = torch.tensordot(cores[i], right, dims=([-1], [0]))
        
        return left, right
    
    def _solve_core_ls(self, tensor: torch.Tensor, env: Tuple[torch.Tensor, torch.Tensor], 
                      core_idx: int, ranks: List[int]) -> torch.Tensor:
        """Solve least squares problem for core update."""
        left, right = env
        
        # Reshape tensor for current mode
        shape = tensor.shape
        mode_size = shape[core_idx]
        left_size = np.prod(shape[:core_idx]) if core_idx > 0 else 1
        right_size = np.prod(shape[core_idx + 1:]) if core_idx < len(shape) - 1 else 1
        
        tensor_reshaped = tensor.reshape(left_size, mode_size, right_size)
        
        # Contract with environment
        result = torch.tensordot(left, tensor_reshaped, dims=([0], [0]))
        result = torch.tensordot(result, right, dims=([-1], [0]))
        
        return result
    
    def _compute_compression_ratio(self, original: torch.Tensor, cores: List[torch.Tensor]) -> float:
        """Compute compression ratio for TR decomposition."""
        original_size = original.numel()
        compressed_size = sum(core.numel() for core in cores)
        return compressed_size / original_size


class AdvancedTensorRankOptimizer(TensorRankOptimizer):
    """
    Advanced rank optimization using multiple optimization strategies.
    
    Implements genetic algorithms, reinforcement learning, and Bayesian
    optimization for finding optimal tensor ranks.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize advanced rank optimizer."""
        super().__init__(config)
        
        # Optimization methods
        self.optimization_methods = {
            'genetic': self._optimize_genetic,
            'reinforcement': self._optimize_reinforcement,
            'bayesian': self._optimize_bayesian,
            'multi_objective': self._optimize_multi_objective
        }
        
        # Advanced parameters
        self.population_size = self.config.get('population_size', 50)
        self.generations = self.config.get('generations', 100)
        self.mutation_rate = self.config.get('mutation_rate', 0.1)
        self.crossover_rate = self.config.get('crossover_rate', 0.8)
        self.n_trials = self.config.get('n_trials', 100)
        self.exploration_rate = self.config.get('exploration_rate', 0.1)
        
        # Performance tracking
        self.optimization_history = {
            'methods_used': [],
            'best_ranks': [],
            'compression_ratios': [],
            'reconstruction_errors': [],
            'optimization_time': []
        }
        
        logger.info(f"Initialized AdvancedTensorRankOptimizer with config: {self.config}")
    
    def optimize_ranks(self, tensor: torch.Tensor, target_compression: float,
                      method: str = 'auto') -> List[int]:
        """
        Optimize tensor ranks using advanced methods.
        
        Args:
            tensor: Input tensor
            target_compression: Target compression ratio
            method: Optimization method ('genetic', 'reinforcement', 'bayesian', 'multi_objective', 'auto')
            
        Returns:
            Optimized ranks for each dimension
            
        Raises:
            ValueError: If method is invalid or optimization fails
        """
        start_time = time.time()
        
        try:
            # Auto-select method based on tensor properties
            if method == 'auto':
                method = self._select_optimization_method(tensor, target_compression)
            
            if method not in self.optimization_methods:
                raise ValueError(f"Unknown optimization method: {method}")
            
            logger.info(f"Using {method} optimization for rank selection")
            
            # Run optimization
            optimal_ranks = self.optimization_methods[method](tensor, target_compression)
            
            # Validate ranks
            if not self._validate_ranks(optimal_ranks, tensor.shape):
                raise ValueError("Invalid ranks produced by optimization")
            
            # Update history
            optimization_time = time.time() - start_time
            self.optimization_history['methods_used'].append(method)
            self.optimization_history['best_ranks'].append(optimal_ranks)
            self.optimization_history['optimization_time'].append(optimization_time)
            
            logger.info(f"Rank optimization completed in {optimization_time:.3f}s: {optimal_ranks}")
            
            return optimal_ranks
            
        except Exception as e:
            logger.error(f"Rank optimization failed: {str(e)}")
            raise RuntimeError(f"Rank optimization failed: {str(e)}")
    
    def _select_optimization_method(self, tensor: torch.Tensor, target_compression: float) -> str:
        """Auto-select optimization method based on tensor properties."""
        # Consider tensor size and target compression
        tensor_size = tensor.numel()
        n_dims = tensor.ndim
        
        if tensor_size < 1e6 and n_dims <= 4:
            # Small tensors: use Bayesian optimization
            return 'bayesian'
        elif target_compression < 0.1:
            # Aggressive compression: use multi-objective
            return 'multi_objective'
        elif n_dims > 6:
            # High-dimensional: use genetic algorithm
            return 'genetic'
        else:
            # Default to reinforcement learning
            return 'reinforcement'
    
    def _optimize_genetic(self, tensor: torch.Tensor, target_compression: float) -> List[int]:
        """Optimize ranks using genetic algorithm."""
        shape = tensor.shape
        n_dims = len(shape)
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            individual = []
            for dim_size in shape:
                # Random rank between 1 and dimension size
                rank = np.random.randint(1, min(dim_size, int(dim_size * target_compression) + 1))
                individual.append(rank)
            population.append(individual)
        
        # Evolution loop
        best_individual = None
        best_fitness = float('-inf')
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_fitness(tensor, individual, target_compression)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            # Selection
            parents = self._tournament_selection(population, fitness_scores)
            
            # Crossover and mutation
            new_population = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    # Crossover
                    if np.random.random() < self.crossover_rate:
                        child1, child2 = self._crossover(parents[i], parents[i + 1])
                    else:
                        child1, child2 = parents[i].copy(), parents[i + 1].copy()
                    
                    # Mutation
                    child1 = self._mutate(child1, shape)
                    child2 = self._mutate(child2, shape)
                    
                    new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        return best_individual
    
    def _optimize_reinforcement(self, tensor: torch.Tensor, target_compression: float) -> List[int]:
        """Optimize ranks using reinforcement learning approach."""
        shape = tensor.shape
        n_dims = len(shape)
        
        # Initialize Q-table (simplified approach)
        q_table = {}
        
        # State: current ranks, Action: increase/decrease rank for each dimension
        current_ranks = [min(10, s) for s in shape]  # Start with small ranks
        
        # RL parameters
        learning_rate = 0.1
        discount_factor = 0.9
        episodes = 100
        
        best_ranks = current_ranks.copy()
        best_reward = float('-inf')
        
        for episode in range(episodes):
            # Epsilon-greedy exploration
            epsilon = self.exploration_rate * (1 - episode / episodes)
            
            # Take action
            action = []
            for i in range(n_dims):
                if np.random.random() < epsilon:
                    # Explore: random action
                    action.append(np.random.choice([-2, -1, 0, 1, 2]))
                else:
                    # Exploit: best known action
                    state_key = tuple(current_ranks)
                    if state_key in q_table:
                        action.append(np.argmax(q_table[state_key][i]) - 2)
                    else:
                        action.append(0)
            
            # Apply action
            new_ranks = []
            for i, (rank, delta) in enumerate(zip(current_ranks, action)):
                new_rank = max(1, min(shape[i], rank + delta))
                new_ranks.append(new_rank)
            
            # Calculate reward
            reward = self._calculate_rl_reward(tensor, new_ranks, target_compression)
            
            # Update Q-table
            state_key = tuple(current_ranks)
            next_state_key = tuple(new_ranks)
            
            if state_key not in q_table:
                q_table[state_key] = [[0] * 5 for _ in range(n_dims)]
            
            for i, a in enumerate(action):
                action_idx = a + 2  # Convert to index
                old_q = q_table[state_key][i][action_idx]
                
                # Calculate max future Q-value
                if next_state_key in q_table:
                    max_future_q = max(q_table[next_state_key][i])
                else:
                    max_future_q = 0
                
                # Q-learning update
                new_q = old_q + learning_rate * (reward + discount_factor * max_future_q - old_q)
                q_table[state_key][i][action_idx] = new_q
            
            # Update best ranks
            if reward > best_reward:
                best_reward = reward
                best_ranks = new_ranks.copy()
            
            current_ranks = new_ranks
        
        return best_ranks
    
    def _optimize_bayesian(self, tensor: torch.Tensor, target_compression: float) -> List[int]:
        """Optimize ranks using Bayesian optimization with Optuna."""
        shape = tensor.shape
        n_dims = len(shape)
        
        def objective(trial):
            # Suggest ranks for each dimension
            ranks = []
            for i, dim_size in enumerate(shape):
                max_rank = min(dim_size, int(dim_size * target_compression * 2))
                rank = trial.suggest_int(f'rank_{i}', 1, max_rank)
                ranks.append(rank)
            
            # Evaluate objective
            fitness = self._evaluate_fitness(tensor, ranks, target_compression)
            return -fitness  # Optuna minimizes
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        # Extract best ranks
        best_ranks = []
        for i in range(n_dims):
            best_ranks.append(study.best_params[f'rank_{i}'])
        
        return best_ranks
    
    def _optimize_multi_objective(self, tensor: torch.Tensor, target_compression: float) -> List[int]:
        """Multi-objective optimization balancing compression and accuracy."""
        shape = tensor.shape
        n_dims = len(shape)
        
        def multi_objective(trial):
            # Suggest ranks
            ranks = []
            for i, dim_size in enumerate(shape):
                max_rank = min(dim_size, int(dim_size * target_compression * 2))
                rank = trial.suggest_int(f'rank_{i}', 1, max_rank)
                ranks.append(rank)
            
            # Calculate multiple objectives
            compression_ratio = self._calculate_compression_ratio(shape, ranks)
            
            # Estimate reconstruction error (simplified)
            error_estimate = sum(1.0 / r for r in ranks) / n_dims
            
            # Return both objectives
            return compression_ratio, error_estimate
        
        # Create multi-objective study
        study = optuna.create_study(
            directions=['minimize', 'minimize'],
            sampler=optuna.samplers.NSGAIISampler(seed=42)
        )
        
        # Optimize
        study.optimize(multi_objective, n_trials=self.n_trials * 2, show_progress_bar=False)
        
        # Select best trade-off solution
        best_trial = None
        best_score = float('inf')
        
        for trial in study.best_trials:
            # Weighted sum of objectives
            compression_weight = 0.7
            error_weight = 0.3
            score = compression_weight * trial.values[0] + error_weight * trial.values[1]
            
            if score < best_score:
                best_score = score
                best_trial = trial
        
        # Extract ranks
        best_ranks = []
        for i in range(n_dims):
            best_ranks.append(best_trial.params[f'rank_{i}'])
        
        return best_ranks
    
    def _evaluate_fitness(self, tensor: torch.Tensor, ranks: List[int], 
                         target_compression: float) -> float:
        """Evaluate fitness of rank configuration."""
        # Calculate compression ratio
        compression_ratio = self._calculate_compression_ratio(tensor.shape, ranks)
        
        # Penalize deviation from target
        compression_penalty = abs(compression_ratio - target_compression)
        
        # Estimate reconstruction quality (higher ranks = better quality)
        quality_score = sum(r / s for r, s in zip(ranks, tensor.shape)) / len(ranks)
        
        # Combined fitness
        fitness = quality_score - 2.0 * compression_penalty
        
        return fitness
    
    def _calculate_compression_ratio(self, shape: Tuple[int, ...], ranks: List[int]) -> float:
        """Calculate compression ratio for given ranks."""
        original_size = np.prod(shape)
        
        # Estimate compressed size (Tucker decomposition)
        core_size = np.prod(ranks)
        factor_sizes = sum(shape[i] * ranks[i] for i in range(len(shape)))
        compressed_size = core_size + factor_sizes
        
        return compressed_size / original_size
    
    def _calculate_rl_reward(self, tensor: torch.Tensor, ranks: List[int], 
                           target_compression: float) -> float:
        """Calculate reward for reinforcement learning."""
        compression_ratio = self._calculate_compression_ratio(tensor.shape, ranks)
        
        # Reward for achieving target compression
        compression_reward = -abs(compression_ratio - target_compression)
        
        # Penalty for very low ranks
        rank_penalty = -sum(1.0 / r for r in ranks if r < 5)
        
        return compression_reward + 0.1 * rank_penalty
    
    def _tournament_selection(self, population: List[List[int]], 
                            fitness_scores: List[float]) -> List[List[int]]:
        """Tournament selection for genetic algorithm."""
        tournament_size = 3
        selected = []
        
        for _ in range(len(population)):
            # Random tournament
            indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in indices]
            winner_idx = indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        return selected
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Crossover operation for genetic algorithm."""
        point = np.random.randint(1, len(parent1))
        
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        
        return child1, child2
    
    def _mutate(self, individual: List[int], shape: Tuple[int, ...]) -> List[int]:
        """Mutation operation for genetic algorithm."""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                # Random mutation within valid range
                max_rank = shape[i]
                delta = np.random.randint(-2, 3)
                mutated[i] = max(1, min(max_rank, mutated[i] + delta))
        
        return mutated
    
    def _validate_ranks(self, ranks: List[int], shape: Tuple[int, ...]) -> bool:
        """Validate rank configuration."""
        if len(ranks) != len(shape):
            return False
        
        for rank, dim_size in zip(ranks, shape):
            if rank <= 0 or rank > dim_size:
                return False
        
        return True


class TensorGPUAccelerator:
    """
    GPU acceleration for tensor operations with CUDA kernel integration.
    
    Provides optimized GPU operations for tensor decompositions with
    memory management and stream optimization.
    """
    
    def __init__(self, gpu_manager: GPUMemoryManager, config: Dict[str, Any] = None):
        """Initialize GPU accelerator."""
        self.gpu_manager = gpu_manager
        self.config = config or {}
        
        # Stream configuration
        self.n_streams = self.config.get('n_streams', 4)
        self.async_operations = self.config.get('async_operations', True)
        self.kernel_optimization = self.config.get('kernel_optimization', True)
        
        # Initialize CUDA streams
        self.streams = []
        self.stream_pool = []
        self._initialize_streams()
        
        # Performance monitoring
        self.performance_stats = {
            'kernel_launches': 0,
            'memory_transfers': 0,
            'stream_synchronizations': 0,
            'optimization_time': 0.0,
            'speedup_ratio': []
        }
        
        # Error handling
        self.error_recovery = self.config.get('error_recovery', True)
        self.max_retries = self.config.get('max_retries', 3)
        
        # CUDA kernel cache
        self.kernel_cache = {}
        
        logger.info(f"Initialized TensorGPUAccelerator with {self.n_streams} streams")
    
    def accelerate_decomposition(self, decomposer: DecompositionMethod, 
                               tensor: torch.Tensor, ranks: List[int]) -> TensorDecomposition:
        """
        Accelerate tensor decomposition using GPU optimization.
        
        Args:
            decomposer: Decomposition method to accelerate
            tensor: Input tensor
            ranks: Decomposition ranks
            
        Returns:
            Accelerated decomposition result
            
        Raises:
            RuntimeError: If GPU acceleration fails
        """
        start_time = time.time()
        
        try:
            # Ensure tensor is on GPU
            if not tensor.is_cuda:
                device_id = self.gpu_manager._select_device()
                tensor = tensor.to(f'cuda:{device_id}')
            
            # Allocate memory for decomposition
            memory_estimate = self._estimate_memory_requirement(tensor, ranks)
            allocated_tensor = self.gpu_manager.allocate_memory(
                memory_estimate,
                device_id=tensor.device.index,
                operation='tensor_decomposition'
            )
            
            # Select optimization strategy
            if isinstance(decomposer, HOSVDDecomposer):
                result = self._accelerate_hosvd(decomposer, tensor, ranks)
            elif isinstance(decomposer, TensorRingDecomposer):
                result = self._accelerate_tensor_ring(decomposer, tensor, ranks)
            else:
                # Fallback to standard acceleration
                result = self._generic_acceleration(decomposer, tensor, ranks)
            
            # Calculate speedup
            optimization_time = time.time() - start_time
            self.performance_stats['optimization_time'] += optimization_time
            
            # Update stats
            self.performance_stats['kernel_launches'] += 1
            
            logger.info(f"GPU acceleration completed in {optimization_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"GPU acceleration failed: {str(e)}")
            
            if self.error_recovery:
                return self._recover_from_error(decomposer, tensor, ranks, e)
            else:
                raise RuntimeError(f"GPU acceleration failed: {str(e)}")
    
    def optimize_memory_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Optimize tensor memory layout for GPU operations.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Optimized tensor with contiguous memory layout
        """
        try:
            # Ensure contiguous memory
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            
            # Optimize stride for coalesced memory access
            if self.kernel_optimization:
                # Analyze access patterns
                optimal_stride = self._calculate_optimal_stride(tensor)
                
                # Reorder dimensions if beneficial
                if optimal_stride != list(range(tensor.ndim)):
                    tensor = tensor.permute(optimal_stride)
            
            # Align memory for better performance
            if tensor.element_size() * tensor.numel() % 128 != 0:
                # Pad tensor for alignment
                padding_size = 128 // tensor.element_size()
                padding_needed = padding_size - (tensor.numel() % padding_size)
                
                if padding_needed < padding_size:
                    flat_tensor = tensor.flatten()
                    padded = torch.nn.functional.pad(flat_tensor, (0, padding_needed))
                    # Store original size for later
                    padded._original_size = tensor.shape
                    tensor = padded
            
            self.performance_stats['memory_transfers'] += 1
            
            return tensor
            
        except Exception as e:
            logger.error(f"Memory layout optimization failed: {str(e)}")
            return tensor
    
    def parallel_tensor_operations(self, operations: List[Callable], 
                                 tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Execute multiple tensor operations in parallel using CUDA streams.
        
        Args:
            operations: List of operations to perform
            tensors: List of input tensors
            
        Returns:
            List of operation results
            
        Raises:
            ValueError: If operations and tensors length mismatch
        """
        if len(operations) != len(tensors):
            raise ValueError("Number of operations must match number of tensors")
        
        results = [None] * len(operations)
        streams = self._get_stream_pool(len(operations))
        
        try:
            # Launch operations on different streams
            for i, (op, tensor, stream) in enumerate(zip(operations, tensors, streams)):
                with torch.cuda.stream(stream):
                    results[i] = op(tensor)
            
            # Synchronize all streams
            for stream in streams:
                stream.synchronize()
            
            self.performance_stats['stream_synchronizations'] += len(streams)
            
            return results
            
        except Exception as e:
            logger.error(f"Parallel tensor operations failed: {str(e)}")
            raise RuntimeError(f"Parallel operations failed: {str(e)}")
    
    def _initialize_streams(self):
        """Initialize CUDA streams for parallel operations."""
        try:
            for device_id in range(self.gpu_manager.num_devices):
                device_streams = []
                with torch.cuda.device(device_id):
                    for _ in range(self.n_streams):
                        stream = torch.cuda.Stream()
                        device_streams.append(stream)
                self.streams.append(device_streams)
            
            # Initialize stream pool
            self.stream_pool = [s for device_streams in self.streams for s in device_streams]
            
        except Exception as e:
            logger.error(f"Stream initialization failed: {str(e)}")
            self.streams = []
            self.stream_pool = []
    
    def _get_stream_pool(self, n_operations: int) -> List[torch.cuda.Stream]:
        """Get streams from pool for parallel operations."""
        if not self.stream_pool:
            return [torch.cuda.default_stream()] * n_operations
        
        # Cycle through available streams
        selected_streams = []
        for i in range(n_operations):
            stream_idx = i % len(self.stream_pool)
            selected_streams.append(self.stream_pool[stream_idx])
        
        return selected_streams
    
    def _accelerate_hosvd(self, decomposer: HOSVDDecomposer, 
                         tensor: torch.Tensor, ranks: List[int]) -> TensorDecomposition:
        """GPU-accelerated HOSVD decomposition."""
        # Use batched SVD operations
        factor_matrices = []
        streams = self._get_stream_pool(tensor.ndim)
        
        # Parallel SVD for each mode
        for mode, stream in zip(range(tensor.ndim), streams):
            with torch.cuda.stream(stream):
                # Unfold tensor
                unfolded = decomposer._unfold_tensor(tensor, mode)
                
                # GPU-optimized SVD
                if min(unfolded.shape) > 100:
                    # Use randomized SVD for large matrices
                    U, S, Vt = self._gpu_randomized_svd(unfolded, ranks[mode])
                else:
                    U, S, Vt = torch.svd(unfolded)
                
                # Truncate
                factor_matrices.append(U[:, :ranks[mode]])
        
        # Synchronize streams
        for stream in streams[:tensor.ndim]:
            stream.synchronize()
        
        # Compute core tensor using GPU operations
        core_tensor = tensor.clone()
        for mode, factor in enumerate(factor_matrices):
            core_tensor = decomposer._mode_product(core_tensor, factor.t(), mode)
        
        # Create decomposition
        return TensorDecomposition(
            type=DecompositionType.TUCKER,
            components={'core': core_tensor, 'factors': factor_matrices},
            original_shape=tensor.shape,
            compression_ratio=decomposer._compute_compression_ratio(tensor, core_tensor, factor_matrices)
        )
    
    def _accelerate_tensor_ring(self, decomposer: TensorRingDecomposer,
                              tensor: torch.Tensor, ranks: List[int]) -> TensorDecomposition:
        """GPU-accelerated Tensor-Ring decomposition."""
        # Initialize cores on GPU
        cores = []
        device = tensor.device
        
        # Use parallel initialization
        init_ops = []
        init_tensors = []
        
        for i in range(len(tensor.shape)):
            core_shape = (ranks[i], tensor.shape[i], ranks[i + 1])
            
            def init_core(shape=core_shape, device=device):
                scale = 1.0 / math.sqrt(np.prod(ranks))
                return torch.randn(shape, device=device) * scale
            
            init_ops.append(init_core)
            init_tensors.append(torch.empty(0))  # Dummy tensor
        
        # Parallel core initialization
        cores = self.parallel_tensor_operations(init_ops, init_tensors)
        
        # GPU-optimized ALS iterations
        prev_error = float('inf')
        
        for iteration in range(decomposer.max_iterations):
            # Parallel core updates
            update_ops = []
            for i in range(len(cores)):
                def update_core(core_idx=i):
                    # Compute environment in parallel
                    left, right = decomposer._compute_environment(cores, core_idx)
                    # Update core
                    return decomposer._solve_core_ls(tensor, (left, right), core_idx, ranks)
                
                update_ops.append(update_core)
            
            # Execute updates in parallel
            cores = self.parallel_tensor_operations(update_ops, cores)
            
            # Check convergence (on GPU)
            with torch.cuda.stream(self.streams[0][0]):
                reconstructed = decomposer.reconstruct(TensorDecomposition(
                    type=DecompositionType.CP,
                    components={'cores': cores},
                    original_shape=tensor.shape,
                    compression_ratio=0
                ))
                
                error = torch.norm(tensor - reconstructed) / torch.norm(tensor)
                error_val = error.item()
            
            if abs(prev_error - error_val) / prev_error < decomposer.tolerance:
                break
            
            prev_error = error_val
        
        # Create final decomposition
        return TensorDecomposition(
            type=DecompositionType.CP,
            components={'cores': cores},
            original_shape=tensor.shape,
            compression_ratio=decomposer._compute_compression_ratio(tensor, cores)
        )
    
    def _generic_acceleration(self, decomposer: DecompositionMethod,
                            tensor: torch.Tensor, ranks: List[int]) -> TensorDecomposition:
        """Generic GPU acceleration for any decomposition method."""
        # Optimize memory layout
        optimized_tensor = self.optimize_memory_layout(tensor)
        
        # Use async operations if possible
        if self.async_operations:
            # Create dedicated stream
            stream = self.streams[tensor.device.index][0]
            
            with torch.cuda.stream(stream):
                result = decomposer.decompose(optimized_tensor, ranks)
            
            stream.synchronize()
        else:
            result = decomposer.decompose(optimized_tensor, ranks)
        
        return result
    
    def _gpu_randomized_svd(self, matrix: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GPU-optimized randomized SVD."""
        m, n = matrix.shape
        k = min(rank, min(m, n))
        
        # Oversampling
        p = 10
        l = k + p
        
        # Random sampling on GPU
        with torch.cuda.device(matrix.device):
            omega = torch.randn(n, l, device=matrix.device, dtype=matrix.dtype)
            
            # Power iterations for better accuracy
            for _ in range(2):
                omega = matrix.t() @ (matrix @ omega)
                omega, _ = torch.qr(omega)
            
            # Form sample matrix
            Y = matrix @ omega
            Q, _ = torch.qr(Y)
            
            # Project and compute SVD
            B = Q.t() @ matrix
            U_tilde, S, Vt = torch.svd(B)
            
            # Recover full decomposition
            U = Q @ U_tilde
        
        return U[:, :k], S[:k], Vt[:k, :]
    
    def _estimate_memory_requirement(self, tensor: torch.Tensor, ranks: List[int]) -> int:
        """Estimate GPU memory requirement for decomposition."""
        # Base tensor size
        tensor_size = tensor.element_size() * tensor.numel()
        
        # Estimate decomposition overhead
        # Core tensor size
        core_size = tensor.element_size() * np.prod(ranks)
        
        # Factor matrices size
        factor_size = tensor.element_size() * sum(
            tensor.shape[i] * ranks[i] for i in range(tensor.ndim)
        )
        
        # Working memory (2x for safety)
        working_memory = 2 * max(tensor_size, core_size + factor_size)
        
        return int(tensor_size + core_size + factor_size + working_memory)
    
    def _calculate_optimal_stride(self, tensor: torch.Tensor) -> List[int]:
        """Calculate optimal dimension ordering for memory access."""
        # Simple heuristic: order dimensions by size (largest first)
        dim_sizes = list(enumerate(tensor.shape))
        dim_sizes.sort(key=lambda x: x[1], reverse=True)
        
        return [idx for idx, _ in dim_sizes]
    
    def _recover_from_error(self, decomposer: DecompositionMethod,
                          tensor: torch.Tensor, ranks: List[int],
                          error: Exception) -> TensorDecomposition:
        """Attempt to recover from GPU acceleration error."""
        logger.warning(f"Attempting recovery from GPU error: {str(error)}")
        
        retry_count = 0
        last_error = error
        
        while retry_count < self.max_retries:
            try:
                # Clear GPU cache
                torch.cuda.empty_cache()
                
                # Try with reduced memory usage
                if retry_count == 0:
                    # Move to CPU
                    cpu_tensor = tensor.cpu()
                    result = decomposer.decompose(cpu_tensor, ranks)
                    
                    # Move result back to GPU if needed
                    if tensor.is_cuda:
                        for key, value in result.components.items():
                            if isinstance(value, torch.Tensor):
                                result.components[key] = value.to(tensor.device)
                            elif isinstance(value, list):
                                result.components[key] = [
                                    v.to(tensor.device) if isinstance(v, torch.Tensor) else v
                                    for v in value
                                ]
                    
                    return result
                    
                elif retry_count == 1:
                    # Try with smaller batch size
                    return self._generic_acceleration(decomposer, tensor, ranks)
                    
                else:
                    # Final attempt: basic decomposition
                    return decomposer.decompose(tensor, ranks)
                    
            except Exception as e:
                last_error = e
                retry_count += 1
                logger.warning(f"Recovery attempt {retry_count} failed: {str(e)}")
        
        raise RuntimeError(f"GPU acceleration recovery failed after {self.max_retries} attempts: {str(last_error)}")
    
    def cleanup(self):
        """Clean up GPU resources."""
        try:
            # Synchronize all streams
            for device_streams in self.streams:
                for stream in device_streams:
                    stream.synchronize()
            
            # Clear kernel cache
            self.kernel_cache.clear()
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            logger.info("GPU accelerator cleanup completed")
            
        except Exception as e:
            logger.error(f"GPU cleanup failed: {str(e)}")


# Module initialization
__all__ = [
    'HOSVDDecomposer',
    'TensorRingDecomposer',
    'AdvancedTensorRankOptimizer',
    'TensorGPUAccelerator'
]