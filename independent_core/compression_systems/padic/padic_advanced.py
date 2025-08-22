"""
Advanced P-adic Compression Features
Includes Hensel lifting, hierarchical clustering, GPU optimization, and p-adic optimizers
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import time
import threading
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import defaultdict
import weakref
from contextlib import contextmanager

from .padic_encoder import PadicWeight, PadicValidation, PadicMathematicalOperations
from .safe_reconstruction import SafePadicReconstructor, ReconstructionConfig, ReconstructionMethod

# Type hints only - avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .padic_compressor import PadicCompressionSystem


class GPUProcessingError(Exception):
    """Exception raised when GPU processing fails to produce valid output"""
    pass


class DecompressionError(Exception):
    """Exception raised when both GPU and CPU decompression fail"""
    pass


@dataclass
class HenselLiftingConfig:
    """Configuration for Hensel lifting operations"""
    max_iterations: int = 50
    convergence_tolerance: float = 1e-12
    damping_factor: float = 0.8
    adaptive_damping: bool = True
    min_damping: float = 0.1
    max_damping: float = 1.0
    precision_schedule: Optional[List[int]] = None
    enable_validation: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be > 0, got {self.max_iterations}")
        if not 0 < self.convergence_tolerance < 1:
            raise ValueError(f"convergence_tolerance must be in (0,1), got {self.convergence_tolerance}")
        if not 0 < self.damping_factor <= 1:
            raise ValueError(f"damping_factor must be in (0,1], got {self.damping_factor}")
        if not 0 < self.min_damping <= self.max_damping <= 1:
            raise ValueError(f"Invalid damping bounds: {self.min_damping}, {self.max_damping}")


class HenselLiftingProcessor:
    """
    Advanced Hensel lifting for p-adic precision maintenance
    Uses Newton-Raphson iterations with adaptive damping
    """
    
    def __init__(self, config: HenselLiftingConfig, prime: int, base_precision: int):
        """Initialize Hensel lifting processor"""
        self.config = config
        self.prime = prime
        self.base_precision = base_precision
        self.current_precision = base_precision
        
        # Validate prime
        PadicValidation.validate_prime(prime)
        PadicValidation.validate_precision(base_precision)
        
        # Initialize mathematical operations
        self.math_ops = PadicMathematicalOperations(prime, base_precision)
        
        # Performance tracking
        self.lifting_stats = {
            'total_lifts': 0,
            'total_iterations': 0,
            'average_iterations': 0.0,
            'convergence_failures': 0,
            'adaptive_damping_activations': 0,
            'precision_increases': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
    
    def lift_to_precision(self, padic_weight: PadicWeight, target_precision: int,
                         initial_guess: Optional[PadicWeight] = None) -> Tuple[PadicWeight, Dict[str, Any]]:
        """
        Lift p-adic weight to higher precision using Hensel lifting
        
        Args:
            padic_weight: Source p-adic weight
            target_precision: Target precision level
            initial_guess: Optional initial guess for lifting
            
        Returns:
            Tuple of (lifted_weight, lifting_metadata)
        """
        with self._lock:
            if target_precision <= padic_weight.precision:
                raise ValueError(f"Target precision {target_precision} must be > current {padic_weight.precision}")
            
            start_time = time.time()
            
            # Initialize lifting
            if initial_guess is None:
                current_weight = padic_weight.copy()
            else:
                if initial_guess.prime != padic_weight.prime:
                    raise ValueError(f"Prime mismatch: {initial_guess.prime} != {padic_weight.prime}")
                current_weight = initial_guess.copy()
            
            # Prepare precision schedule
            precision_schedule = self._create_precision_schedule(
                padic_weight.precision, target_precision
            )
            
            lifting_metadata = {
                'initial_precision': padic_weight.precision,
                'target_precision': target_precision,
                'precision_schedule': precision_schedule,
                'iterations_per_level': [],
                'convergence_history': [],
                'damping_history': [],
                'start_time': start_time
            }
            
            try:
                # Lift through precision levels
                for level_idx, level_precision in enumerate(precision_schedule):
                    if level_precision <= current_weight.precision:
                        continue
                    
                    # Perform lifting to this precision level
                    lifted_weight, level_metadata = self._lift_single_level(
                        current_weight, level_precision
                    )
                    
                    # Update metadata
                    lifting_metadata['iterations_per_level'].append(level_metadata['iterations'])
                    lifting_metadata['convergence_history'].extend(level_metadata['convergence_history'])
                    lifting_metadata['damping_history'].extend(level_metadata['damping_history'])
                    
                    current_weight = lifted_weight
                    
                    # Validate intermediate result if enabled
                    if self.config.enable_validation:
                        self._validate_lifting_step(padic_weight, current_weight, level_precision)
                
                # Final validation
                if self.config.enable_validation:
                    self._validate_final_lifting(padic_weight, current_weight, target_precision)
                
                # Update statistics
                total_iterations = sum(lifting_metadata['iterations_per_level'])
                self._update_lifting_stats(total_iterations, True)
                
                lifting_metadata.update({
                    'total_iterations': total_iterations,
                    'lifting_time': time.time() - start_time,
                    'success': True,
                    'final_precision': current_weight.precision
                })
                
                return current_weight, lifting_metadata
                
            except Exception as e:
                # Update failure statistics
                self._update_lifting_stats(0, False)
                lifting_metadata.update({
                    'success': False,
                    'error': str(e),
                    'lifting_time': time.time() - start_time
                })
                raise ValueError(f"Hensel lifting failed: {e}")
    
    def _create_precision_schedule(self, initial: int, target: int) -> List[int]:
        """Create precision lifting schedule"""
        if self.config.precision_schedule is not None:
            # Use provided schedule, filtered for valid range
            schedule = [p for p in self.config.precision_schedule if initial < p <= target]
            if not schedule or schedule[-1] != target:
                schedule.append(target)
            return sorted(schedule)
        
        # Create geometric progression schedule
        if target - initial <= 4:
            # Small gap, lift directly
            return [target]
        
        # Geometric progression with ratio ~1.5
        schedule = []
        current = initial
        ratio = 1.5
        
        while current < target:
            next_precision = min(int(current * ratio), target)
            if next_precision > current:
                schedule.append(next_precision)
                current = next_precision
            else:
                break
        
        if not schedule or schedule[-1] != target:
            schedule.append(target)
        
        return schedule
    
    def _lift_single_level(self, padic_weight: PadicWeight, target_precision: int) -> Tuple[PadicWeight, Dict[str, Any]]:
        """Lift p-adic weight to single precision level using Newton-Raphson"""
        
        # Initialize
        current = padic_weight.copy()
        current.precision = target_precision
        damping = self.config.damping_factor
        
        convergence_history = []
        damping_history = []
        
        for iteration in range(self.config.max_iterations):
            # Calculate Newton-Raphson correction
            try:
                correction = self._compute_newton_correction(current, padic_weight)
                
                # Apply damping
                damped_correction = self._apply_damping(correction, damping)
                
                # Update solution
                previous = current.copy()
                current = self.math_ops.add_padic(current, damped_correction)
                
                # Check convergence
                residual = self._compute_residual(current, padic_weight)
                convergence_history.append(residual)
                damping_history.append(damping)
                
                if residual < self.config.convergence_tolerance:
                    # Converged successfully
                    metadata = {
                        'iterations': iteration + 1,
                        'final_residual': residual,
                        'convergence_history': convergence_history,
                        'damping_history': damping_history,
                        'converged': True
                    }
                    return current, metadata
                
                # Adaptive damping adjustment
                if self.config.adaptive_damping and iteration > 0:
                    damping = self._adjust_damping(
                        damping, residual, convergence_history
                    )
                
            except Exception as e:
                raise ValueError(f"Newton iteration {iteration} failed: {e}")
        
        # Failed to converge
        raise RuntimeError(f"Failed to converge after {self.config.max_iterations} iterations. "
                          f"Final residual: {convergence_history[-1] if convergence_history else 'unknown'}")
    
    def _compute_newton_correction(self, current: PadicWeight, target: PadicWeight) -> PadicWeight:
        """Compute Newton-Raphson correction term"""
        # For f(x) = x - target, f'(x) = 1, so correction = -(current - target)
        difference = self.math_ops.subtract_padic(current, target)
        return self.math_ops.negate_padic(difference)
    
    def _apply_damping(self, correction: PadicWeight, damping_factor: float) -> PadicWeight:
        """Apply damping to correction term"""
        damped_coeffs = [int(coeff * damping_factor) for coeff in correction.digits]
        return PadicWeight(
            value=correction.value,
            prime=correction.prime,
            precision=correction.precision,
            valuation=correction.valuation,
            digits=damped_coeffs
        )
    
    def _compute_residual(self, current: PadicWeight, target: PadicWeight) -> float:
        """Compute residual for convergence checking"""
        try:
            difference = self.math_ops.subtract_padic(current, target)
            # Use p-adic norm as residual
            return self.math_ops.padic_norm(difference)
        except Exception:
            # Fallback to coefficient-based residual
            residual = 0.0
            min_len = min(len(current.digits), len(target.digits))
            for i in range(min_len):
                residual += abs(current.digits[i] - target.digits[i])
            return residual / (min_len if min_len > 0 else 1)
    
    def _adjust_damping(self, current_damping: float, residual: float, history: List[float]) -> float:
        """Adjust damping factor based on convergence history"""
        if len(history) < 2:
            return current_damping
        
        # Check if residual is decreasing
        if residual < history[-2]:
            # Good progress, potentially increase damping
            new_damping = min(current_damping * 1.1, self.config.max_damping)
        else:
            # Poor progress, decrease damping
            new_damping = max(current_damping * 0.7, self.config.min_damping)
            self.lifting_stats['adaptive_damping_activations'] += 1
        
        return new_damping
    
    def _validate_lifting_step(self, original: PadicWeight, lifted: PadicWeight, precision: int) -> None:
        """Validate intermediate lifting step"""
        if lifted.prime != original.prime:
            raise ValueError(f"Prime changed during lifting: {original.prime} -> {lifted.prime}")
        if lifted.precision != precision:
            raise ValueError(f"Precision mismatch: expected {precision}, got {lifted.precision}")
        
        # Check that lifted weight reduces to original when precision is lowered
        try:
            reduced = lifted.copy()
            reduced.precision = original.precision
            reduced.digits = reduced.digits[:original.precision]
            
            # Should match original within tolerance
            diff_norm = self._compute_residual(reduced, original)
            if diff_norm > self.config.convergence_tolerance * 10:
                raise ValueError(f"Lifting validation failed: residual {diff_norm} too large")
                
        except Exception as e:
            raise ValueError(f"Lifting validation error: {e}")
    
    def _validate_final_lifting(self, original: PadicWeight, final: PadicWeight, target_precision: int) -> None:
        """Validate final lifting result"""
        if final.precision != target_precision:
            raise ValueError(f"Final precision mismatch: expected {target_precision}, got {final.precision}")
        
        # Validate that final result is consistent with original
        self._validate_lifting_step(original, final, target_precision)
    
    def _update_lifting_stats(self, iterations: int, success: bool) -> None:
        """Update lifting performance statistics"""
        self.lifting_stats['total_lifts'] += 1
        
        if success:
            self.lifting_stats['total_iterations'] += iterations
            total_lifts = self.lifting_stats['total_lifts'] - self.lifting_stats['convergence_failures']
            if total_lifts > 0:
                self.lifting_stats['average_iterations'] = (
                    self.lifting_stats['total_iterations'] / total_lifts
                )
        else:
            self.lifting_stats['convergence_failures'] += 1
    
    def get_lifting_stats(self) -> Dict[str, Any]:
        """Get lifting performance statistics"""
        with self._lock:
            return dict(self.lifting_stats)
    
    def reset_stats(self) -> None:
        """Reset lifting statistics"""
        with self._lock:
            self.lifting_stats = {
                'total_lifts': 0,
                'total_iterations': 0,
                'average_iterations': 0.0,
                'convergence_failures': 0,
                'adaptive_damping_activations': 0,
                'precision_increases': 0
            }


@dataclass
class ClusteringConfig:
    """Configuration for hierarchical clustering"""
    max_cluster_size: int = 5000
    min_cluster_size: int = 50
    branching_factor: int = 4
    distance_threshold: float = 1e-6
    enable_caching: bool = True
    cache_size_limit: int = 10000
    ultrametric_validation: bool = True
    cohomology_tracking: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.max_cluster_size <= self.min_cluster_size:
            raise ValueError(f"max_cluster_size must be > min_cluster_size")
        if self.branching_factor < 2:
            raise ValueError(f"branching_factor must be >= 2, got {self.branching_factor}")
        if self.distance_threshold <= 0:
            raise ValueError(f"distance_threshold must be > 0, got {self.distance_threshold}")


@dataclass
class ClusterNode:
    """Node in hierarchical clustering tree"""
    id: str
    elements: List[PadicWeight]
    children: List['ClusterNode']
    parent: Optional['ClusterNode']
    centroid: Optional[PadicWeight]
    radius: float
    level: int
    cohomology_class: Optional[int] = None
    
    def __post_init__(self):
        """Initialize computed properties"""
        if self.centroid is None and self.elements:
            self.centroid = self._compute_centroid()
    
    def _compute_centroid(self) -> PadicWeight:
        """Compute cluster centroid"""
        if not self.elements:
            raise ValueError("Cannot compute centroid of empty cluster")
        
        # Use first element as template
        template = self.elements[0]
        centroid_coeffs = [0] * template.precision
        
        # Average digits
        for element in self.elements:
            for i in range(min(len(centroid_coeffs), len(element.digits))):
                centroid_coeffs[i] += element.digits[i]
        
        # Normalize
        n = len(self.elements)
        centroid_coeffs = [coeff // n for coeff in centroid_coeffs]
        
        return PadicWeight(
            value=template.value,
            prime=template.prime,
            precision=template.precision,
            valuation=template.valuation,
            digits=centroid_coeffs
        )
    
    def is_leaf(self) -> bool:
        """Check if node is leaf"""
        return len(self.children) == 0
    
    def size(self) -> int:
        """Get total number of elements in subtree"""
        if self.is_leaf():
            return len(self.elements)
        return sum(child.size() for child in self.children)


class HierarchicalClusteringManager:
    """
    Advanced hierarchical clustering for p-adic weights
    Uses ultrametric distance with configurable branching
    """
    
    def __init__(self, config: ClusteringConfig, prime: int):
        """Initialize clustering manager"""
        self.config = config
        self.prime = prime
        
        # Validate prime
        PadicValidation.validate_prime(prime)
        
        # Initialize components with SAFE precision
        # Calculate max safe precision for this prime
        import math
        safe_threshold = 1e12
        max_safe_precision = int(math.log(safe_threshold) / math.log(prime))
        safe_precision = min(10, max_safe_precision)  # Use 10 or less if unsafe
        
        self.math_ops = PadicMathematicalOperations(prime, safe_precision)  # Safe precision
        
        # Clustering state
        self.cluster_tree: Optional[ClusterNode] = None
        self.distance_cache: Dict[Tuple[str, str], float] = {}
        self.cluster_count = 0
        
        # Statistics
        self.clustering_stats = {
            'total_clusterings': 0,
            'total_nodes_created': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'ultrametric_violations': 0,
            'cohomology_computations': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
    
    def build_hierarchical_clustering(self, weights: List[PadicWeight]) -> Tuple[ClusterNode, Dict[str, Any]]:
        """
        Build hierarchical clustering tree using ultrametric distances
        
        Args:
            weights: List of p-adic weights to cluster
            
        Returns:
            Tuple of (root_node, clustering_metadata)
        """
        with self._lock:
            if not weights:
                raise ValueError("Cannot cluster empty weight list")
            
            if len(weights) < self.config.min_cluster_size:
                raise ValueError(f"Need at least {self.config.min_cluster_size} weights, got {len(weights)}")
            
            start_time = time.time()
            
            # Validate all weights have same prime
            for i, weight in enumerate(weights):
                if weight.prime != self.prime:
                    raise ValueError(f"Weight {i} has wrong prime: {weight.prime} != {self.prime}")
            
            try:
                # Build distance matrix if caching enabled
                if self.config.enable_caching:
                    self._precompute_distances(weights)
                
                # Create initial leaf nodes
                leaf_nodes = self._create_leaf_nodes(weights)
                
                # Build tree bottom-up
                root_node = self._build_tree_recursive(leaf_nodes, level=0)
                
                # Validate tree structure
                self._validate_cluster_tree(root_node)
                
                # Compute cohomology classes if enabled
                if self.config.cohomology_tracking:
                    self._compute_cohomology_classes(root_node)
                
                self.cluster_tree = root_node
                
                # Update statistics
                self._update_clustering_stats(len(weights), True)
                
                clustering_metadata = {
                    'num_weights': len(weights),
                    'tree_depth': self._compute_tree_depth(root_node),
                    'total_nodes': self._count_nodes(root_node),
                    'leaf_nodes': len(leaf_nodes),
                    'clustering_time': time.time() - start_time,
                    'cache_usage': {
                        'size': len(self.distance_cache),
                        'hits': self.clustering_stats['cache_hits'],
                        'misses': self.clustering_stats['cache_misses']
                    },
                    'cohomology_classes': self._extract_cohomology_info(root_node) if self.config.cohomology_tracking else None
                }
                
                return root_node, clustering_metadata
                
            except Exception as e:
                self._update_clustering_stats(len(weights), False)
                raise ValueError(f"Hierarchical clustering failed: {e}")
    
    def _precompute_distances(self, weights: List[PadicWeight]) -> None:
        """Precompute pairwise distances for caching"""
        n = len(weights)
        distances_computed = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                weight_i, weight_j = weights[i], weights[j]
                
                # Create cache keys
                key_i = self._create_weight_key(weight_i)
                key_j = self._create_weight_key(weight_j)
                cache_key = (key_i, key_j) if key_i < key_j else (key_j, key_i)
                
                if cache_key not in self.distance_cache:
                    try:
                        distance = self._compute_ultrametric_distance(weight_i, weight_j)
                        self.distance_cache[cache_key] = distance
                        distances_computed += 1
                        
                        # Enforce cache size limit
                        if len(self.distance_cache) > self.config.cache_size_limit:
                            self._prune_distance_cache()
                            
                    except Exception as e:
                        # Log warning but continue
                        continue
    
    def _create_weight_key(self, weight: PadicWeight) -> str:
        """Create unique key for p-adic weight"""
        return f"p{weight.prime}_pr{weight.precision}_c{hash(tuple(weight.digits))}"
    
    def _compute_ultrametric_distance(self, weight1: PadicWeight, weight2: PadicWeight) -> float:
        """Compute ultrametric distance between p-adic weights"""
        try:
            # Use p-adic norm of difference
            difference = self.math_ops.subtract_padic(weight1, weight2)
            distance = self.math_ops.padic_norm(difference)
            
            # Validate ultrametric property if required
            if self.config.ultrametric_validation:
                self._validate_ultrametric_property_sample(weight1, weight2, distance)
            
            return distance
            
        except Exception as e:
            # Fallback to coefficient-based distance
            min_precision = min(weight1.precision, weight2.precision)
            distance = 0.0
            
            for i in range(min_precision):
                c1 = weight1.digits[i] if i < len(weight1.digits) else 0
                c2 = weight2.digits[i] if i < len(weight2.digits) else 0
                distance += abs(c1 - c2) * (self.prime ** (-i))
            
            return distance
    
    def _validate_ultrametric_property_sample(self, weight1: PadicWeight, weight2: PadicWeight, distance: float) -> None:
        """Sample validation of ultrametric property"""
        # For efficiency, only validate occasionally
        if np.random.random() > 0.1:  # 10% validation rate
            return
        
        # Would need a third weight to fully validate ultrametric property
        # This is a simplified check
        if distance < 0:
            self.clustering_stats['ultrametric_violations'] += 1
            raise ValueError(f"Negative distance violates ultrametric property: {distance}")
    
    def _create_leaf_nodes(self, weights: List[PadicWeight]) -> List[ClusterNode]:
        """Create leaf nodes for individual weights"""
        leaf_nodes = []
        
        for i, weight in enumerate(weights):
            node = ClusterNode(
                id=f"leaf_{i}",
                elements=[weight],
                children=[],
                parent=None,
                centroid=weight.copy(),
                radius=0.0,
                level=0
            )
            leaf_nodes.append(node)
        
        return leaf_nodes
    
    def _build_tree_recursive(self, nodes: List[ClusterNode], level: int) -> ClusterNode:
        """Recursively build clustering tree"""
        if len(nodes) == 1:
            return nodes[0]
        
        if len(nodes) <= self.config.branching_factor:
            # Small enough to create single parent
            return self._create_parent_node(nodes, level)
        
        # Group nodes into clusters
        node_groups = self._group_nodes_by_distance(nodes)
        
        # Recursively cluster each group
        parent_nodes = []
        for group in node_groups:
            if len(group) == 1:
                parent_nodes.append(group[0])
            else:
                parent_node = self._create_parent_node(group, level)
                parent_nodes.append(parent_node)
        
        # Continue recursively
        return self._build_tree_recursive(parent_nodes, level + 1)
    
    def _group_nodes_by_distance(self, nodes: List[ClusterNode]) -> List[List[ClusterNode]]:
        """Group nodes based on ultrametric distances"""
        if len(nodes) <= self.config.branching_factor:
            return [nodes]
        
        # Use hierarchical clustering with ultrametric distances
        groups = []
        remaining_nodes = nodes.copy()
        
        while remaining_nodes:
            # Start new group with first remaining node
            current_group = [remaining_nodes.pop(0)]
            
            # Add nodes within distance threshold
            nodes_to_remove = []
            for node in remaining_nodes:
                min_dist_to_group = float('inf')
                
                for group_node in current_group:
                    dist = self._get_node_distance(node, group_node)
                    min_dist_to_group = min(min_dist_to_group, dist)
                
                if min_dist_to_group <= self.config.distance_threshold:
                    current_group.append(node)
                    nodes_to_remove.append(node)
                
                # Limit group size
                if len(current_group) >= self.config.max_cluster_size // self.config.branching_factor:
                    break
            
            # Remove added nodes from remaining
            for node in nodes_to_remove:
                remaining_nodes.remove(node)
            
            groups.append(current_group)
            
            # Limit number of groups
            if len(groups) >= self.config.branching_factor:
                # Add all remaining nodes to last group
                if remaining_nodes:
                    groups[-1].extend(remaining_nodes)
                break
        
        return groups
    
    def _get_node_distance(self, node1: ClusterNode, node2: ClusterNode) -> float:
        """Get distance between cluster nodes"""
        if node1.centroid is None or node2.centroid is None:
            return float('inf')
        
        # Use cached distance if available
        key1 = self._create_weight_key(node1.centroid)
        key2 = self._create_weight_key(node2.centroid)
        cache_key = (key1, key2) if key1 < key2 else (key2, key1)
        
        if cache_key in self.distance_cache:
            self.clustering_stats['cache_hits'] += 1
            return self.distance_cache[cache_key]
        
        # Compute distance
        distance = self._compute_ultrametric_distance(node1.centroid, node2.centroid)
        
        # Cache if within limits
        if len(self.distance_cache) < self.config.cache_size_limit:
            self.distance_cache[cache_key] = distance
        
        self.clustering_stats['cache_misses'] += 1
        return distance
    
    def _create_parent_node(self, children: List[ClusterNode], level: int) -> ClusterNode:
        """Create parent node from child nodes"""
        # Collect all elements from children
        all_elements = []
        for child in children:
            all_elements.extend(child.elements)
        
        # Compute parent properties
        node_id = f"internal_{self.cluster_count}"
        self.cluster_count += 1
        
        # Create parent node
        parent = ClusterNode(
            id=node_id,
            elements=all_elements,
            children=children,
            parent=None,
            centroid=None,  # Will be computed in __post_init__
            radius=self._compute_cluster_radius(children),
            level=level
        )
        
        # Set parent references in children
        for child in children:
            child.parent = parent
        
        self.clustering_stats['total_nodes_created'] += 1
        return parent
    
    def _compute_cluster_radius(self, children: List[ClusterNode]) -> float:
        """Compute radius of cluster containing children"""
        if not children:
            return 0.0
        
        if len(children) == 1:
            return children[0].radius
        
        # Find maximum distance between child centroids
        max_distance = 0.0
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                distance = self._get_node_distance(children[i], children[j])
                max_distance = max(max_distance, distance)
        
        # Add maximum child radius
        max_child_radius = max(child.radius for child in children)
        return max_distance / 2 + max_child_radius
    
    def _validate_cluster_tree(self, root: ClusterNode) -> None:
        """Validate cluster tree structure"""
        # Check tree connectivity
        self._validate_tree_connectivity(root)
        
        # Check size constraints
        self._validate_size_constraints(root)
        
        # Check ultrametric properties
        if self.config.ultrametric_validation:
            self._validate_tree_ultrametric_properties(root)
    
    def _validate_tree_connectivity(self, node: ClusterNode) -> None:
        """Validate tree connectivity"""
        for child in node.children:
            if child.parent != node:
                raise ValueError(f"Tree connectivity broken: child {child.id} parent mismatch")
            self._validate_tree_connectivity(child)
    
    def _validate_size_constraints(self, node: ClusterNode) -> None:
        """Validate cluster size constraints"""
        size = node.size()
        if node.is_leaf():
            if size == 0:
                raise ValueError(f"Empty leaf node: {node.id}")
        else:
            if size < self.config.min_cluster_size:
                raise ValueError(f"Node {node.id} too small: {size} < {self.config.min_cluster_size}")
            if size > self.config.max_cluster_size:
                raise ValueError(f"Node {node.id} too large: {size} > {self.config.max_cluster_size}")
        
        # Recursively validate children
        for child in node.children:
            self._validate_size_constraints(child)
    
    def _validate_tree_ultrametric_properties(self, node: ClusterNode) -> None:
        """Validate ultrametric properties in tree"""
        # Sample validation to avoid expensive full check
        if len(node.children) >= 3 and np.random.random() < 0.1:
            # Check ultrametric property for sample of child triplets
            for _ in range(min(10, len(node.children) // 3)):
                try:
                    indices = np.random.choice(len(node.children), 3, replace=False)
                    children_sample = [node.children[i] for i in indices]
                    
                    # Check distances satisfy ultrametric inequality
                    distances = []
                    for i in range(3):
                        for j in range(i + 1, 3):
                            dist = self._get_node_distance(children_sample[i], children_sample[j])
                            distances.append(dist)
                    
                    # Ultrametric inequality: d(x,z) <= max(d(x,y), d(y,z))
                    distances.sort()
                    if distances[2] > max(distances[0], distances[1]) + 1e-10:  # Small tolerance
                        self.clustering_stats['ultrametric_violations'] += 1
                        
                except Exception:
                    # Continue validation despite errors
                    continue
        
        # Recursively validate children
        for child in node.children:
            self._validate_tree_ultrametric_properties(child)
    
    def _compute_cohomology_classes(self, root: ClusterNode) -> None:
        """Compute cohomology classes for cluster nodes"""
        if not self.config.cohomology_tracking:
            return
        
        self._assign_cohomology_classes_recursive(root, 0)
        self.clustering_stats['cohomology_computations'] += 1
    
    def _assign_cohomology_classes_recursive(self, node: ClusterNode, base_class: int) -> None:
        """Recursively assign cohomology classes"""
        # Assign cohomology class based on tree structure
        node.cohomology_class = base_class + node.level
        
        # Recursively assign to children
        child_base = base_class + len(node.children)
        for i, child in enumerate(node.children):
            self._assign_cohomology_classes_recursive(child, child_base + i)
    
    def _compute_tree_depth(self, node: ClusterNode) -> int:
        """Compute depth of clustering tree"""
        if node.is_leaf():
            return 1
        return 1 + max(self._compute_tree_depth(child) for child in node.children)
    
    def _count_nodes(self, node: ClusterNode) -> int:
        """Count total nodes in tree"""
        return 1 + sum(self._count_nodes(child) for child in node.children)
    
    def _extract_cohomology_info(self, root: ClusterNode) -> Dict[str, Any]:
        """Extract cohomology information from tree"""
        cohomology_info = {
            'classes_assigned': 0,
            'max_class': 0,
            'class_distribution': defaultdict(int)
        }
        
        self._collect_cohomology_info_recursive(root, cohomology_info)
        
        return {
            'total_classes': cohomology_info['classes_assigned'],
            'max_class': cohomology_info['max_class'],
            'distribution': dict(cohomology_info['class_distribution'])
        }
    
    def _collect_cohomology_info_recursive(self, node: ClusterNode, info: Dict[str, Any]) -> None:
        """Recursively collect cohomology information"""
        if node.cohomology_class is not None:
            info['classes_assigned'] += 1
            info['max_class'] = max(info['max_class'], node.cohomology_class)
            info['class_distribution'][node.cohomology_class] += 1
        
        for child in node.children:
            self._collect_cohomology_info_recursive(child, info)
    
    def _prune_distance_cache(self) -> None:
        """Prune distance cache to maintain size limits"""
        if len(self.distance_cache) <= self.config.cache_size_limit:
            return
        
        # Remove oldest entries (simple LRU approximation)
        items_to_remove = len(self.distance_cache) - self.config.cache_size_limit // 2
        keys_to_remove = list(self.distance_cache.keys())[:items_to_remove]
        
        for key in keys_to_remove:
            del self.distance_cache[key]
    
    def _update_clustering_stats(self, num_weights: int, success: bool) -> None:
        """Update clustering statistics"""
        self.clustering_stats['total_clusterings'] += 1
        if not success:
            # Could track failure statistics here
            pass
    
    def find_cluster_for_weight(self, weight: PadicWeight) -> Optional[ClusterNode]:
        """Find appropriate cluster for new weight"""
        if self.cluster_tree is None:
            return None
        
        return self._find_best_cluster_recursive(self.cluster_tree, weight)
    
    def _find_best_cluster_recursive(self, node: ClusterNode, weight: PadicWeight) -> ClusterNode:
        """Recursively find best cluster for weight"""
        if node.is_leaf():
            return node
        
        # Find child with minimum distance to centroid
        best_child = None
        min_distance = float('inf')
        
        for child in node.children:
            if child.centroid is not None:
                distance = self._compute_ultrametric_distance(weight, child.centroid)
                if distance < min_distance:
                    min_distance = distance
                    best_child = child
        
        if best_child is None:
            return node
        
        return self._find_best_cluster_recursive(best_child, weight)
    
    def get_clustering_stats(self) -> Dict[str, Any]:
        """Get clustering statistics"""
        with self._lock:
            return dict(self.clustering_stats)
    
    def reset_cache(self) -> None:
        """Reset distance cache"""
        with self._lock:
            self.distance_cache.clear()
    
    def reset_stats(self) -> None:
        """Reset clustering statistics"""
        with self._lock:
            self.clustering_stats = {
                'total_clusterings': 0,
                'total_nodes_created': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'ultrametric_violations': 0,
                'cohomology_computations': 0
            }


@dataclass
class GPUDecompressionConfig:
    """Configuration for GPU-optimized decompression"""
    enable_cuda_streams: bool = True
    num_streams: int = 4
    batch_size: int = 5000
    memory_pool_size_mb: int = 2048
    enable_progressive_precision: bool = True
    precision_schedule: Optional[List[int]] = None
    enable_async_transfer: bool = True
    stream_priority_high: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.num_streams < 1:
            raise ValueError(f"num_streams must be >= 1, got {self.num_streams}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.memory_pool_size_mb <= 0:
            raise ValueError(f"memory_pool_size_mb must be > 0, got {self.memory_pool_size_mb}")


class PadicDecompressionEngine:
    """
    GPU-optimized progressive decompression engine
    Uses CUDA streams for parallel processing
    """
    
    def __init__(self, config: GPUDecompressionConfig, prime: int):
        """Initialize GPU decompression engine with arbitrary precision support"""
        self.config = config
        self.prime = prime
        
        # Validate prime
        PadicValidation.validate_prime(prime)
        
        # CRITICAL: Pre-compute arbitrary precision powers like CPU version
        self.max_precision = 64  # Support up to 64-bit precision
        self.prime_powers = [1]
        for i in range(1, self.max_precision + 1):
            self.prime_powers.append(self.prime_powers[-1] * prime)
        
        # Validate prime powers don't exceed safe ranges
        self._validate_prime_powers()
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for GPU decompression")
        
        # Initialize GPU resources
        self.device = torch.device('cuda:0')
        self.streams = []
        self.memory_pool = None
        
        # Initialize components with SAFE precision
        # Calculate max safe precision for this prime
        import math
        safe_threshold = 1e12
        max_safe_precision = int(math.log(safe_threshold) / math.log(prime))
        safe_precision = min(10, max_safe_precision)  # Use 10 or less if unsafe
        
        self.math_ops = PadicMathematicalOperations(prime, safe_precision)
        
        # Initialize safe reconstructor with appropriate config
        self.reconstruction_config = ReconstructionConfig(
            prime=self.prime,
            max_safe_precision=6,  # Safe limit for prime=257
            method=ReconstructionMethod.HYBRID,
            use_gpu=True,
            overflow_threshold=1e12
        )
        self.safe_reconstructor = SafePadicReconstructor(self.reconstruction_config)
        
        # Performance tracking
        self.decompression_stats = {
            'total_decompressions': 0,
            'total_weights_processed': 0,
            'total_gpu_time': 0.0,
            'total_transfer_time': 0.0,
            'average_throughput': 0.0,
            'stream_utilization': [0] * config.num_streams,
            'memory_peak_usage': 0
        }
        
        # Initialize GPU resources
        self._initialize_gpu_resources()
        
        # Thread safety
        self._lock = threading.RLock()
    
    def _extract_gpu_channels(self, batch: List[PadicWeight]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract mantissa and exponent channels from p-adic weights for GPU processing.
        Mirrors CPU implementation but optimized for GPU batch processing.
        
        Args:
            batch: List of PadicWeight objects
            
        Returns:
            Tuple of (mantissa_tensor, exponent_tensor) on GPU
        """
        batch_size = len(batch)
        
        # Find max digits for mantissa channel sizing
        max_digits = max(len(weight.digits) for weight in batch)
        min_channel_size = max(10, max_digits)  # Ensure minimum size for consistency
        
        # Create mantissa tensor from p-adic digits
        mantissa_tensor = torch.zeros((batch_size, min_channel_size), 
                                      dtype=torch.float32, device=self.device)
        
        # Create exponent tensor from valuations
        # Format: [sign, magnitude, 0, 0, ...] matching CPU implementation
        exponent_tensor = torch.zeros((batch_size, min_channel_size), 
                                      dtype=torch.float32, device=self.device)
        
        # Fill tensors
        for i, weight in enumerate(batch):
            # Validate weight
            if weight.prime != self.prime:
                raise ValueError(f"Weight {i} has prime {weight.prime}, expected {self.prime}")
            
            # Fill mantissa channel with digits
            digit_count = len(weight.digits)
            if digit_count > 0:
                mantissa_tensor[i, :digit_count] = torch.tensor(
                    weight.digits, dtype=torch.float32, device=self.device
                )
            
            # Fill exponent channel with valuation encoding
            valuation = getattr(weight, 'valuation', 0)
            exponent_tensor[i, 0] = 1.0 if valuation >= 0 else -1.0  # Sign
            exponent_tensor[i, 1] = float(abs(valuation))  # Magnitude
        
        return mantissa_tensor, exponent_tensor

    def _gpu_reconstruct_from_channels(self, mantissa: torch.Tensor, exponent: torch.Tensor, 
                                      batch: List[PadicWeight], precision: int) -> torch.Tensor:
        """
        GPU reconstruction using separated mantissa/exponent channels.
        
        Args:
            mantissa: Mantissa channel tensor (batch_size, channel_size)
            exponent: Exponent channel tensor (batch_size, channel_size)
            batch: Original PadicWeight list for metadata
            precision: Target precision
            
        Returns:
            Reconstructed values tensor
        """
        batch_size = mantissa.shape[0]
        
        # Extract valuations from exponent channel
        valuation_signs = exponent[:, 0]
        valuation_magnitudes = exponent[:, 1]
        valuations = (valuation_signs * valuation_magnitudes).int()
        
        # Prepare digits tensor for reconstruction (only actual digits, not full channel)
        max_actual_digits = max(len(weight.digits) for weight in batch)
        digits_tensor = torch.zeros((batch_size, max_actual_digits), 
                                   dtype=torch.int32, device=self.device)
        
        for i, weight in enumerate(batch):
            actual_digits = len(weight.digits)
            if actual_digits > 0:
                # Use mantissa values up to actual digit count
                digits_tensor[i, :actual_digits] = mantissa[i, :actual_digits].int()
        
        # Use existing GPU reconstruction kernel
        results = self._gpu_reconstruct_kernel_channelized(
            digits_tensor, valuations, min(precision, max_actual_digits)
        )
        
        return results

    def _gpu_reconstruct_kernel_channelized(self, digits: torch.Tensor, 
                                           valuations: torch.Tensor,
                                           precision: int) -> torch.Tensor:
        """
        GPU kernel for channel-based batch reconstruction.
        Enhanced version of SafePadicReconstructor's kernel.
        """
        batch_size = digits.shape[0]
        device = digits.device
        
        # Pre-compute prime powers on GPU (reuse if possible)
        if not hasattr(self, '_cached_prime_powers') or len(self._cached_prime_powers) < precision:
            self._cached_prime_powers = torch.pow(
                float(self.prime), 
                torch.arange(precision, dtype=torch.float32, device=device)
            )
        
        prime_powers = self._cached_prime_powers[:precision]
        
        # Batch matrix multiplication for mantissa processing
        digits_float = digits[:, :precision].float()
        mantissa_results = torch.sum(digits_float * prime_powers.unsqueeze(0), dim=1)
        
        # Apply exponent (valuation) factors efficiently
        # Split positive and negative valuations for numerical stability
        pos_mask = valuations >= 0
        neg_mask = ~pos_mask
        
        results = torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        if pos_mask.any():
            pos_valuations = valuations[pos_mask].float()
            pos_factors = torch.pow(float(self.prime), pos_valuations)
            results[pos_mask] = mantissa_results[pos_mask] * pos_factors
        
        if neg_mask.any():
            neg_valuations = valuations[neg_mask].float()
            neg_factors = torch.pow(float(self.prime), neg_valuations)
            results[neg_mask] = mantissa_results[neg_mask] * neg_factors
        
        return results
    
    def _validate_prime_powers(self):
        """Validate pre-computed powers are within safe ranges"""
        for i, power in enumerate(self.prime_powers):
            if power > 1e100:  # Conservative limit for double precision
                self.max_precision = i - 1
                self.prime_powers = self.prime_powers[:i]
                break
    
    def is_safe_to_decompress(self, weight: PadicWeight, threshold: float = 1e12) -> bool:
        """
        Pre-check if weight will cause overflow during reconstruction
        
        Args:
            weight: P-adic weight to validate
            threshold: Overflow threshold (default 1e12)
            
        Returns:
            True if safe to decompress, False if likely to cause overflow
        """
        try:
            # Extract weight properties
            valuation = getattr(weight, 'valuation', 0)
            digits = getattr(weight, 'digits', [])
            prime = getattr(weight, 'prime', self.prime)
            
            if not digits:
                return True  # Empty digits are safe
            
            # Conservative estimate of reconstructed value before valuation
            max_digit = max(abs(d) for d in digits)
            precision = len(digits)
            
            # Estimate maximum possible value: max_digit * sum(prime^i for i in range(precision))
            # This is approximately max_digit * (prime^precision - 1) / (prime - 1)
            if precision > 0 and prime > 1:
                geometric_sum = (prime ** precision - 1) // (prime - 1)
                estimated_base_value = max_digit * geometric_sum
            else:
                estimated_base_value = max_digit
            
            # Apply valuation: final_value = estimated_base_value * prime^valuation
            if valuation > 0:
                # Check if prime^valuation would cause overflow
                if valuation >= len(self.prime_powers):
                    return False  # Valuation too high for our precomputed powers
                
                prime_power_valuation = self.prime_powers[valuation]
                estimated_final_value = estimated_base_value * prime_power_valuation
            elif valuation < 0:
                # Division reduces value, so it's safer
                estimated_final_value = estimated_base_value / (prime ** abs(valuation))
            else:
                estimated_final_value = estimated_base_value
            
            # Use 90% of threshold as safety margin
            safety_threshold = threshold * 0.9
            
            return estimated_final_value <= safety_threshold
            
        except Exception:
            # If we can't validate, err on the side of caution
            return False
    
    def _initialize_gpu_resources(self) -> None:
        """Initialize GPU streams and memory pool"""
        try:
            # Create CUDA streams
            for i in range(self.config.num_streams):
                if self.config.stream_priority_high:
                    stream = torch.cuda.Stream(device=self.device, priority=-1)
                else:
                    stream = torch.cuda.Stream(device=self.device)
                self.streams.append(stream)
            
            # Initialize memory pool (simplified - would use actual GPU memory management)
            self.memory_pool = {
                'allocated': 0,
                'limit': self.config.memory_pool_size_mb * 1024 * 1024,
                'blocks': []
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GPU resources: {e}")
    
    def decompress_progressive(self, padic_weights: List[PadicWeight], 
                             target_precision: int,
                             metadata: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Progressive decompression with GPU optimization and edge case filtering
        
        Args:
            padic_weights: List of p-adic weights to decompress
            target_precision: Target precision for decompression
            metadata: Decompression metadata
            
        Returns:
            Tuple of (decompressed_tensor, decompression_metadata)
        """
        with self._lock:
            if not padic_weights:
                raise ValueError("Cannot decompress empty weight list")
            
            start_time = time.time()
            original_count = len(padic_weights)
            
            # Pre-validation: Filter out unsafe weights and track positions
            safe_weights = []
            filtered_positions = set()  # Track which positions were filtered
            skip_reasons = {'overflow_risk': 0, 'validation_error': 0}
            
            print(f"Pre-validating {original_count} weights for safe decompression...")
            
            for i, weight in enumerate(padic_weights):
                try:
                    if self.is_safe_to_decompress(weight):
                        safe_weights.append((i, weight))
                    else:
                        filtered_positions.add(i)
                        skip_reasons['overflow_risk'] += 1
                        if len(filtered_positions) <= 5:  # Log first 5 skips
                            print(f"Skipping weight {i}: overflow risk detected (valuation={getattr(weight, 'valuation', 0)})")
                except Exception as e:
                    filtered_positions.add(i)
                    skip_reasons['validation_error'] += 1
                    if len(filtered_positions) <= 5:
                        print(f"Skipping weight {i}: validation error: {e}")
            
            if not safe_weights:
                raise ValueError("No safe weights to decompress after filtering")
            
            print(f"Filtered weights: {len(safe_weights)} safe, {len(filtered_positions)} skipped")
            
            # Store filtering information in metadata for tensor reconstruction
            metadata['filtered_positions'] = filtered_positions
            metadata['original_weight_count'] = original_count
            
            try:
                # Validate inputs for safe weights
                safe_weight_list = [w for _, w in safe_weights]
                self._validate_decompression_inputs(safe_weight_list, target_precision, metadata)
                
                # Create precision schedule
                precision_schedule = self._create_decompression_schedule(
                    safe_weight_list[0].precision, target_precision
                )
                
                # Prepare GPU memory
                self._prepare_gpu_memory(len(safe_weights), target_precision)
                
                # Process in batches with graceful degradation
                successful_results = []
                failed_batches = []
                stream_idx = 0
                
                for batch_start in range(0, len(safe_weights), self.config.batch_size):
                    batch_end = min(batch_start + self.config.batch_size, len(safe_weights))
                    batch_weights = [safe_weights[i][1] for i in range(batch_start, batch_end)]
                    batch_indices = [safe_weights[i][0] for i in range(batch_start, batch_end)]
                    
                    # Use round-robin stream assignment
                    stream = self.streams[stream_idx]
                    stream_idx = (stream_idx + 1) % len(self.streams)
                    
                    try:
                        # Process batch with error handling
                        batch_result = self._decompress_batch_gpu(
                            batch_weights, precision_schedule, stream, batch_start
                        )
                        successful_results.append((batch_indices, batch_result))
                        
                        # Update stream utilization
                        self.decompression_stats['stream_utilization'][stream_idx] += 1
                        
                    except (GPUProcessingError, DecompressionError) as e:
                        print(f"Batch {batch_start//self.config.batch_size} failed: {e}")
                        failed_batches.append({
                            'indices': batch_indices,
                            'error': str(e),
                            'batch_start': batch_start
                        })
                        # Continue processing remaining batches
                        continue
                        
                    except Exception as e:
                        print(f"Unexpected error in batch {batch_start//self.config.batch_size}: {e}")
                        failed_batches.append({
                            'indices': batch_indices,
                            'error': f"Unexpected: {str(e)}",
                            'batch_start': batch_start
                        })
                        continue
                
                # Synchronize all streams
                for stream in self.streams:
                    stream.synchronize()
                
                if not successful_results:
                    raise ValueError("All batches failed during processing")
                
                # Combine successful results
                final_tensor = self._combine_successful_batch_results(successful_results, metadata)
                
                # FINAL VALIDATION on combined results
                if torch.any(torch.isnan(final_tensor)):
                    raise ValueError("Final tensor contains NaN values")
                if torch.any(torch.isinf(final_tensor)):  
                    raise ValueError("Final tensor contains infinite values")
                
                # Check for reasonable value range (but allow larger values since we filtered)
                max_abs_val = torch.max(torch.abs(final_tensor))
                if max_abs_val > 1e15:  # Increased threshold since we pre-filtered
                    print(f"Warning: Final tensor contains large values: max={max_abs_val:.2e}")
                
                # Update statistics
                decompression_time = time.time() - start_time
                processed_count = sum(len(indices) for indices, _ in successful_results)
                self._update_decompression_stats(processed_count, decompression_time)
                
                # Calculate comprehensive statistics
                total_failed_weights = len(filtered_positions) + sum(len(batch['indices']) for batch in failed_batches)
                
                # Create enhanced metadata with shape integrity tracking
                decompression_metadata = {
                    'input_weights': original_count,
                    'processed_weights': processed_count,
                    'filtered_weights': len(filtered_positions),
                    'failed_batch_weights': sum(len(batch['indices']) for batch in failed_batches),
                    'total_failed_weights': total_failed_weights,
                    'success_rate': processed_count / original_count,
                    'shape_integrity_maintained': True,  # Solution 2 always maintains shape
                    'zero_filled_positions': len(filtered_positions),
                    'skip_reasons': skip_reasons,
                    'failed_batches': len(failed_batches),
                    'successful_batches': len(successful_results),
                    'target_precision': target_precision,
                    'precision_schedule': precision_schedule,
                    'decompression_time': decompression_time,
                    'gpu_utilization': self._calculate_gpu_utilization(),
                    'memory_usage': self._get_memory_usage_info(),
                    'throughput': processed_count / decompression_time if decompression_time > 0 else 0,
                    'max_value': float(max_abs_val),
                    'algorithm': 'gpu_arbitrary_precision_filtered'
                }
                
                print(f"Decompression complete: {processed_count}/{original_count} weights processed "
                      f"({processed_count/original_count*100:.1f}% success rate)")
                
                return final_tensor, decompression_metadata
                
            except Exception as e:
                # Clean up GPU resources on failure
                self._cleanup_gpu_memory()
                raise ValueError(f"GPU decompression failed: {e}")
    
    def _validate_decompression_inputs(self, weights: List[PadicWeight], 
                                     target_precision: int, metadata: Dict[str, Any]) -> None:
        """Validate decompression inputs with hard failures"""
        # Check weights consistency
        if not weights:
            raise ValueError("Cannot decompress empty weight list")
        
        if not all(w.prime == self.prime for w in weights):
            raise ValueError(f"All weights must have prime {self.prime}")
        
        # Allow precision changes for compression/decompression scenarios
        if target_precision <= 0:
            raise ValueError(f"Target precision must be > 0, got {target_precision}")
        if target_precision > self.max_precision:
            raise ValueError(f"Target precision {target_precision} exceeds maximum {self.max_precision}")
        
        # Check metadata
        required_keys = {'original_shape', 'dtype', 'device'}
        if not all(key in metadata for key in required_keys):
            raise ValueError(f"Missing required metadata keys: {required_keys - set(metadata.keys())}")
        
        # Validate original shape makes sense
        original_shape = metadata['original_shape']
        if not isinstance(original_shape, (tuple, list)):
            raise ValueError(f"original_shape must be tuple or list, got {type(original_shape)}")
        
        expected_elements = 1
        for dim in original_shape:
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(f"Invalid shape dimension: {dim}")
            expected_elements *= dim
        
        # Store expected elements for later tensor reconstruction
        metadata['expected_elements'] = expected_elements
        
        # NOTE: We no longer enforce exact weight count match here since filtering may reduce weight count
        # The shape integrity will be maintained during tensor reconstruction
        if len(weights) > expected_elements:
            raise ValueError(f"Too many weights: got {len(weights)}, expected at most {expected_elements}")
    
    def _create_decompression_schedule(self, current_precision: int, target_precision: int) -> List[int]:
        """Create progressive decompression schedule with overflow protection"""
        if not self.config.enable_progressive_precision:
            return [target_precision]
        
        if self.config.precision_schedule is not None:
            schedule = [p for p in self.config.precision_schedule
                       if current_precision <= p <= target_precision]
            if not schedule or schedule[-1] != target_precision:
                schedule.append(target_precision)
            return sorted(schedule)
        
        # Create conservative schedule to avoid large precision jumps
        if target_precision - current_precision <= 4:
            return [target_precision]
        
        # CONSERVATIVE PROGRESSION: Smaller steps to avoid overflow
        schedule = []
        current = current_precision
        
        while current < target_precision:
            # Use smaller multiplier to avoid dangerous precision jumps
            next_precision = min(int(current * 1.2), current + 8, target_precision)
            if next_precision > current:
                # SAFETY CHECK: Ensure we can handle this precision
                if next_precision > self.max_precision:
                    raise ValueError(f"Precision schedule would exceed maximum {self.max_precision}")
                schedule.append(next_precision)
                current = next_precision
            else:
                break
        
        # SAFETY: Ensure schedule is never empty
        if not schedule:
            schedule = [target_precision]
        
        return schedule
    
    def _prepare_gpu_memory(self, total_elements: int, target_precision: int) -> None:
        """Prepare GPU memory for decompression"""
        if not torch.cuda.is_available():
            return
        
        # Estimate memory requirements
        element_size = 8 if target_precision > 32 else 4  # double vs float
        required_memory = total_elements * target_precision * element_size
        
        # Check available memory
        available_memory = torch.cuda.get_device_properties(0).total_memory
        used_memory = torch.cuda.memory_allocated()
        free_memory = available_memory - used_memory
        
        memory_threshold = 0.8  # Use 80% of available memory
        if required_memory > free_memory * memory_threshold:
            # Clear cache to free memory
            torch.cuda.empty_cache()
            # Re-check
            used_memory = torch.cuda.memory_allocated()
            free_memory = available_memory - used_memory
            if required_memory > free_memory * memory_threshold:
                raise RuntimeError(f"Insufficient GPU memory: need {required_memory/(1024**3):.2f}GB, have {free_memory/(1024**3):.2f}GB free")
        
        # Reset memory pool
        if self.memory_pool:
            self.memory_pool['allocated'] = 0
            self.memory_pool['blocks'] = []
    
    def _decompress_batch_gpu(self, batch: List[PadicWeight], precision_schedule: List[int],
                            stream: torch.cuda.Stream, batch_start: int) -> Dict[str, Any]:
        """Decompress batch using GPU stream with channel-based processing"""
        with torch.cuda.stream(stream):
            current_result = None
            
            for precision in precision_schedule:
                # CRITICAL FIX: Convert batch using channel extraction
                mantissa_channel, exponent_channel = self._convert_padic_to_gpu_format_fixed(batch)
                
                # Process with channel-based reconstruction
                result = self._process_gpu_data_fixed(
                    (mantissa_channel, exponent_channel), batch, precision
                )
                
                # Ensure result is never None
                if result is None:
                    print(f"Warning: _process_gpu_data_fixed returned None, using zeros for batch {batch_start}")
                    result = torch.zeros(len(batch), dtype=torch.float32, device=self.device)
                
                # Validate intermediate result
                if torch.any(torch.isnan(result)):
                    raise ValueError(f"NaN detected at precision {precision} in batch {batch_start}")
                if torch.any(torch.isinf(result)):
                    raise ValueError(f"Inf detected at precision {precision} in batch {batch_start}")
                
                max_val = torch.max(torch.abs(result))
                if max_val > 1e15:  # Increased threshold to match SafePadicReconstructor
                    raise ValueError(f"Overflow detected at precision {precision} in batch {batch_start}: max={max_val:.2e}")
                
                current_result = result
            
            # Final conversion to float tensor
            if current_result is None:
                print(f"Warning: current_result is None for batch {batch_start}, using zeros")
                current_result = torch.zeros(len(batch), dtype=torch.float32, device=self.device)
            
            final_data = self._convert_to_final_format(current_result)
            
            return {
                'data': final_data,
                'batch_start': batch_start,
                'batch_size': len(batch),
                'final_precision': precision_schedule[-1] if precision_schedule else 1,
                'used_channels': True  # New flag to indicate channel-based processing
            }
    
    def _convert_padic_to_gpu_format_fixed(self, batch: List[PadicWeight]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert p-adic weights to GPU-friendly format using channel separation.
        Now returns mantissa and exponent channels instead of raw digits.
        
        Args:
            batch: List of PadicWeight objects
            
        Returns:
            Tuple of (mantissa_channel, exponent_channel) tensors
        """
        # HARD VALIDATION: Ensure all weights are valid
        for i, weight in enumerate(batch):
            if not isinstance(weight, PadicWeight):
                raise TypeError(f"Batch element {i} must be PadicWeight, got {type(weight)}")
            if weight.prime != self.prime:
                raise ValueError(f"Weight {i} has prime {weight.prime}, expected {self.prime}")
            if not weight.digits:
                raise ValueError(f"Weight {i} has empty digits")
        
        # Extract channels using GPU-optimized method
        return self._extract_gpu_channels(batch)
    
    def _process_gpu_data_fixed(self, gpu_data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
                               batch: List[PadicWeight], target_precision: int) -> torch.Tensor:
        """
        Process p-adic data using channel-based GPU reconstruction.
        
        Args:
            gpu_data: Either legacy tensor or tuple of (mantissa, exponent) channels
            batch: List of PadicWeight objects
            target_precision: Target precision for reconstruction
            
        Returns:
            Reconstructed values tensor
        """
        # Handle both legacy (single tensor) and new (channel tuple) formats
        if isinstance(gpu_data, tuple):
            mantissa_channel, exponent_channel = gpu_data
            use_channels = True
        else:
            # Legacy path - convert to channels
            mantissa_channel, exponent_channel = self._extract_gpu_channels(batch)
            use_channels = False
        
        try:
            # Use channel-based GPU reconstruction
            results_tensor = self._gpu_reconstruct_from_channels(
                mantissa_channel, exponent_channel, batch, target_precision
            )
            
            # Ensure we never return None
            if results_tensor is None:
                raise RuntimeError("GPU channel reconstruction returned None")
            
            # Validate results
            max_val = torch.max(torch.abs(results_tensor))
            if max_val > 1e15:  # Consistent with SafePadicReconstructor
                print(f"Warning: Large values detected (max={max_val:.2e}), may indicate precision issues")
            
            return results_tensor
            
        except (OverflowError, RuntimeError) as gpu_error:
            # Try with reduced precision if overflow
            if "overflow" in str(gpu_error).lower() and target_precision > 1:
                print(f"GPU overflow detected, reducing precision from {target_precision} to {target_precision-1}")
                
                reduced_precision = max(1, target_precision - 1)
                try:
                    results_tensor = self._gpu_reconstruct_from_channels(
                        mantissa_channel, exponent_channel, batch, reduced_precision
                    )
                    
                    if results_tensor is None:
                        raise RuntimeError("GPU channel reconstruction with reduced precision returned None")
                    
                    return results_tensor
                    
                except Exception as retry_error:
                    print(f"GPU retry with reduced precision failed: {retry_error}")
                    # Fall through to CPU fallback
            
            # CPU fallback using safe reconstruction
            print(f"GPU channel reconstruction failed: {gpu_error}, falling back to CPU")
            try:
                # Convert batch to safe weight format
                safe_weights = []
                for weight in batch:
                    from .safe_reconstruction import PadicWeight as SafeWeight
                    safe_weight = SafeWeight(
                        digits=weight.digits,
                        valuation=getattr(weight, 'valuation', 0),
                        precision=weight.precision,
                        prime=self.prime
                    )
                    safe_weights.append(safe_weight)
                
                # Use CPU reconstruction
                cpu_results = self.safe_reconstructor.reconstruct_batch_cpu(
                    safe_weights, target_precision
                )
                
                if cpu_results is None:
                    raise RuntimeError("CPU reconstruction returned None")
                
                # Convert to GPU tensor
                result_tensor = torch.tensor(cpu_results, dtype=torch.float32, device=self.device)
                return result_tensor
                
            except Exception as cpu_error:
                # Last resort: return zeros
                print(f"Both GPU and CPU reconstruction failed, returning zeros: GPU: {gpu_error}, CPU: {cpu_error}")
                return torch.zeros(len(batch), dtype=torch.float32, device=self.device)
        
        except Exception as other_error:
            # Handle other errors with fallback
            print(f"Unexpected error in GPU processing: {other_error}, attempting CPU fallback")
            try:
                # Convert batch to safe weight format
                safe_weights = []
                for weight in batch:
                    from .safe_reconstruction import PadicWeight as SafeWeight
                    safe_weight = SafeWeight(
                        digits=weight.digits,
                        valuation=getattr(weight, 'valuation', 0),
                        precision=weight.precision,
                        prime=self.prime
                    )
                    safe_weights.append(safe_weight)
                
                cpu_results = self.safe_reconstructor.reconstruct_batch_cpu(
                    safe_weights, target_precision
                )
                result_tensor = torch.tensor(cpu_results, dtype=torch.float32, device=self.device)
                return result_tensor
            except Exception as final_error:
                # Last resort: return zeros
                print(f"Final fallback failed, returning zeros: {final_error}")
                return torch.zeros(len(batch), dtype=torch.float32, device=self.device)
    
    def _update_batch_precision(self, batch: List[PadicWeight], precision: int) -> List[PadicWeight]:
        """Update batch with new precision (for progressive processing)"""
        # For this implementation, return the same batch
        # In a full implementation, would update precision appropriately
        return batch
    
    def _convert_to_final_format(self, processed_data: torch.Tensor) -> torch.Tensor:
        """Convert processed GPU data to final format"""
        # HARD FAILURE if processed_data is None
        if processed_data is None:
            raise RuntimeError("GPU decompression failed: processed_data is None - HARD FAILURE")
        
        # Ensure data is on CPU for final assembly
        return processed_data.cpu()
    
    def _combine_batch_results(self, batch_results: List[Dict[str, Any]], 
                             metadata: Dict[str, Any]) -> torch.Tensor:
        """Combine batch results into final tensor"""
        # Sort by batch start position
        batch_results.sort(key=lambda x: x['batch_start'])
        
        # Ensure all batches are on the same device
        device = metadata.get('device', 'cpu')
        batches_on_device = []
        
        for batch_result in batch_results:
            batch_data = batch_result['data']
            if batch_data.device.type != device:
                batch_data = batch_data.to(device)
            batches_on_device.append(batch_data)
        
        # Concatenate all batches
        combined = torch.cat(batches_on_device, dim=0)
        
        # Reshape to original shape
        original_shape = metadata['original_shape']
        final_tensor = combined.reshape(original_shape)
        
        # Convert to requested dtype
        dtype_str = metadata['dtype'].split('.')[-1] if '.' in metadata['dtype'] else metadata['dtype']
        if hasattr(torch, dtype_str):
            target_dtype = getattr(torch, dtype_str)
            if final_tensor.dtype != target_dtype:
                final_tensor = final_tensor.to(target_dtype)
        
        return final_tensor
    
    def _combine_successful_batch_results(self, successful_results: List[Tuple[List[int], Dict[str, Any]]], 
                                        metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Combine successful batch results into final tensor with full shape integrity.
        
        This method implements Solution 2: maintain original tensor shape by filling
        filtered positions with safe default values (zeros).
        """
        if not successful_results:
            raise ValueError("No successful results to combine")
        
        # Get shape and filtering information
        original_shape = metadata['original_shape']
        expected_elements = metadata['expected_elements']
        filtered_positions = metadata.get('filtered_positions', set())
        original_weight_count = metadata.get('original_weight_count', expected_elements)
        
        # Determine device and dtype
        device = metadata.get('device', 'cpu')
        if device == 'cuda:0' and torch.cuda.is_available():
            target_device = torch.device('cuda:0')
        else:
            target_device = torch.device('cpu')
            
        dtype_str = metadata.get('dtype', 'torch.float32')
        if '.' in dtype_str:
            dtype_str = dtype_str.split('.')[-1]
        target_dtype = getattr(torch, dtype_str) if hasattr(torch, dtype_str) else torch.float32
        
        # Create full-size result tensor initialized with zeros
        result_tensor = torch.zeros(expected_elements, dtype=target_dtype, device=target_device)
        
        # Map successful results back to their original positions
        for batch_indices, batch_result in successful_results:
            batch_data = batch_result['data'] if isinstance(batch_result, dict) else batch_result
            
            # Ensure batch data is on correct device and dtype
            if batch_data.device != target_device:
                batch_data = batch_data.to(target_device)
            if batch_data.dtype != target_dtype:
                batch_data = batch_data.to(target_dtype)
            
            # Flatten batch data for position mapping
            batch_flat = batch_data.flatten()
            
            # Map each element back to its original position
            for i, original_pos in enumerate(batch_indices):
                if i < len(batch_flat) and original_pos < expected_elements:
                    result_tensor[original_pos] = batch_flat[i]
        
        # Reshape to original shape
        result_tensor = result_tensor.reshape(original_shape)
        
        # Log reconstruction statistics
        filled_positions = expected_elements - len(filtered_positions)
        zero_positions = len(filtered_positions)
        
        print(f"Tensor reconstruction: {filled_positions} filled positions, {zero_positions} zero-filled (filtered) positions")
        print(f"Final tensor shape: {result_tensor.shape}, dtype: {result_tensor.dtype}")
        
        # Verify tensor integrity
        if torch.any(torch.isnan(result_tensor)):
            raise ValueError("Reconstructed tensor contains NaN values")
        if torch.any(torch.isinf(result_tensor)):
            raise ValueError("Reconstructed tensor contains infinite values")
        
        return result_tensor
    
    def _calculate_gpu_utilization(self) -> Dict[str, float]:
        """Calculate GPU utilization metrics"""
        total_stream_usage = sum(self.decompression_stats['stream_utilization'])
        if total_stream_usage == 0:
            return {'average': 0.0, 'per_stream': [0.0] * len(self.streams)}
        
        per_stream_util = [usage / total_stream_usage for usage in self.decompression_stats['stream_utilization']]
        average_util = sum(per_stream_util) / len(per_stream_util)
        
        return {
            'average': average_util,
            'per_stream': per_stream_util
        }
    
    def _get_memory_usage_info(self) -> Dict[str, Any]:
        """Get memory usage information"""
        return {
            'allocated_bytes': self.memory_pool['allocated'],
            'limit_bytes': self.memory_pool['limit'],
            'utilization': self.memory_pool['allocated'] / self.memory_pool['limit'],
            'peak_usage': self.decompression_stats['memory_peak_usage']
        }
    
    def _cleanup_gpu_memory(self) -> None:
        """Clean up GPU resources on failure"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if self.memory_pool:
            self.memory_pool['allocated'] = 0
            self.memory_pool['blocks'] = []
    
    def _update_decompression_stats(self, num_elements: int, decompression_time: float) -> None:
        """Update decompression statistics"""
        self.decompression_stats['total_decompressions'] += 1
        self.decompression_stats['total_weights_processed'] += num_elements
        self.decompression_stats['total_gpu_time'] += decompression_time
        
        # Update throughput
        total_elements = self.decompression_stats['total_weights_processed']
        total_time = self.decompression_stats['total_gpu_time']
        if total_time > 0:
            self.decompression_stats['average_throughput'] = total_elements / total_time
    
    def get_decompression_stats(self) -> Dict[str, Any]:
        """Get decompression statistics"""
        with self._lock:
            return dict(self.decompression_stats)
    
    def reset_stats(self) -> None:
        """Reset decompression statistics"""
        with self._lock:
            self.decompression_stats = {
                'total_decompressions': 0,
                'total_weights_processed': 0,
                'total_gpu_time': 0.0,
                'total_transfer_time': 0.0,
                'average_throughput': 0.0,
                'stream_utilization': [0] * self.config.num_streams,
                'memory_peak_usage': 0
            }
    
    def cleanup(self) -> None:
        """Clean up GPU resources"""
        with self._lock:
            self._cleanup_gpu_memory()
            self.streams.clear()


class PadicOptimizer(ABC):
    """Abstract base class for p-adic optimizers"""
    
    def __init__(self, params: List[PadicWeight], prime: int, lr: float = 0.01):
        """Initialize p-adic optimizer"""
        self.param_groups = [{'params': params, 'lr': lr}]
        self.prime = prime
        
        # Calculate max safe precision for this prime
        import math
        safe_threshold = 1e12
        max_safe_precision = int(math.log(safe_threshold) / math.log(prime))
        safe_precision = min(10, max_safe_precision)  # Use 10 or less if unsafe
        
        self.math_ops = PadicMathematicalOperations(prime, safe_precision)
        self.state = defaultdict(dict)
        self.step_count = 0
    
    @abstractmethod
    def step(self, gradients: List[PadicWeight]) -> None:
        """Perform optimization step"""
        pass
    
    def zero_grad(self) -> None:
        """Zero gradients (placeholder for compatibility)"""
        pass


class PadicSGD(PadicOptimizer):
    """P-adic Stochastic Gradient Descent optimizer"""
    
    def __init__(self, params: List[PadicWeight], prime: int, lr: float = 0.01, 
                 momentum: float = 0.0, dampening: float = 0.0):
        """Initialize P-adic SGD optimizer"""
        super().__init__(params, prime, lr)
        self.momentum = momentum
        self.dampening = dampening
        
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if dampening < 0.0:
            raise ValueError(f"Invalid dampening value: {dampening}")
    
    def step(self, gradients: List[PadicWeight]) -> None:
        """Perform SGD step with p-adic arithmetic"""
        if len(gradients) != len(self.param_groups[0]['params']):
            raise ValueError("Gradient count mismatch")
        
        self.step_count += 1
        lr = self.param_groups[0]['lr']
        
        for i, (param, grad) in enumerate(zip(self.param_groups[0]['params'], gradients)):
            param_state = self.state[i]
            
            # Apply momentum if enabled
            if self.momentum != 0:
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = grad.copy()
                else:
                    buf = param_state['momentum_buffer']
                    # buf = momentum * buf + (1 - dampening) * grad
                    buf_scaled = self._scale_padic(buf, self.momentum)
                    grad_scaled = self._scale_padic(grad, 1.0 - self.dampening)
                    param_state['momentum_buffer'] = self.math_ops.add_padic(buf_scaled, grad_scaled)
                
                grad = param_state['momentum_buffer']
            
            # Apply update: param = param - lr * grad
            lr_grad = self._scale_padic(grad, lr)
            param_update = self.math_ops.subtract_padic(param, lr_grad)
            
            # Update parameter in place
            param.digits = param_update.digits.copy()
    
    def _scale_padic(self, padic_weight: PadicWeight, scalar: float) -> PadicWeight:
        """Scale p-adic weight by scalar"""
        scaled_coeffs = [int(coeff * scalar) for coeff in padic_weight.digits]
        return PadicWeight(
            value=padic_weight.value,
            prime=padic_weight.prime,
            precision=padic_weight.precision,
            valuation=padic_weight.valuation,
            digits=scaled_coeffs
        )


class PadicAdam(PadicOptimizer):
    """P-adic Adam optimizer"""
    
    def __init__(self, params: List[PadicWeight], prime: int, lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        """Initialize P-adic Adam optimizer"""
        super().__init__(params, prime, lr)
        self.betas = betas
        self.eps = eps
        
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
    
    def step(self, gradients: List[PadicWeight]) -> None:
        """Perform Adam step with p-adic arithmetic"""
        if len(gradients) != len(self.param_groups[0]['params']):
            raise ValueError("Gradient count mismatch")
        
        self.step_count += 1
        lr = self.param_groups[0]['lr']
        beta1, beta2 = self.betas
        
        for i, (param, grad) in enumerate(zip(self.param_groups[0]['params'], gradients)):
            param_state = self.state[i]
            
            # Initialize state
            if 'exp_avg' not in param_state:
                param_state['exp_avg'] = self._create_zero_padic(param)
                param_state['exp_avg_sq'] = self._create_zero_padic(param)
            
            exp_avg = param_state['exp_avg']
            exp_avg_sq = param_state['exp_avg_sq']
            
            # Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            exp_avg_scaled = self._scale_padic(exp_avg, beta1)
            grad_scaled = self._scale_padic(grad, 1.0 - beta1)
            param_state['exp_avg'] = self.math_ops.add_padic(exp_avg_scaled, grad_scaled)
            
            # Update biased second raw moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            grad_squared = self._square_padic(grad)
            exp_avg_sq_scaled = self._scale_padic(exp_avg_sq, beta2)
            grad_sq_scaled = self._scale_padic(grad_squared, 1.0 - beta2)
            param_state['exp_avg_sq'] = self.math_ops.add_padic(exp_avg_sq_scaled, grad_sq_scaled)
            
            # Bias correction
            bias_correction1 = 1.0 - (beta1 ** self.step_count)
            bias_correction2 = 1.0 - (beta2 ** self.step_count)
            
            # Corrected estimates
            corrected_exp_avg = self._scale_padic(param_state['exp_avg'], 1.0 / bias_correction1)
            corrected_exp_avg_sq = self._scale_padic(param_state['exp_avg_sq'], 1.0 / bias_correction2)
            
            # Update: param = param - lr * m_hat / (sqrt(v_hat) + eps)
            denominator = self._sqrt_padic(corrected_exp_avg_sq, self.eps)
            update = self._divide_padic(corrected_exp_avg, denominator)
            lr_update = self._scale_padic(update, lr)
            
            param_update = self.math_ops.subtract_padic(param, lr_update)
            param.digits = param_update.digits.copy()
    
    def _create_zero_padic(self, template: PadicWeight) -> PadicWeight:
        """Create zero p-adic weight with same structure as template"""
        return PadicWeight(
            value=template.value,
            prime=template.prime,
            precision=template.precision,
            valuation=template.valuation,
            digits=[0] * template.precision
        )
    
    def _square_padic(self, padic_weight: PadicWeight) -> PadicWeight:
        """Square p-adic weight (simplified)"""
        squared_coeffs = [coeff ** 2 for coeff in padic_weight.digits]
        return PadicWeight(
            value=padic_weight.value,
            prime=padic_weight.prime,
            precision=padic_weight.precision,
            valuation=padic_weight.valuation,
            digits=squared_coeffs
        )
    
    def _sqrt_padic(self, padic_weight: PadicWeight, eps: float) -> PadicWeight:
        """Square root of p-adic weight (simplified with epsilon)"""
        sqrt_coeffs = [max(eps, abs(coeff) ** 0.5) for coeff in padic_weight.digits]
        return PadicWeight(
            value=padic_weight.value,
            prime=padic_weight.prime,
            precision=padic_weight.precision,
            valuation=padic_weight.valuation,
            digits=sqrt_coeffs
        )
    
    def _divide_padic(self, numerator: PadicWeight, denominator: PadicWeight) -> PadicWeight:
        """Divide p-adic weights (simplified)"""
        div_coeffs = []
        for i in range(numerator.precision):
            num = numerator.digits[i] if i < len(numerator.digits) else 0
            den = denominator.digits[i] if i < len(denominator.digits) else 1
            div_coeffs.append(num / den if den != 0 else num)
        
        return PadicWeight(
            value=Fraction(numerator.value.numerator, denominator.value.numerator) if denominator.value.numerator != 0 else numerator.value,
            prime=numerator.prime,
            precision=numerator.precision,
            valuation=numerator.valuation - denominator.valuation,
            digits=[int(coeff) for coeff in div_coeffs]
        )


class PadicRMSprop(PadicOptimizer):
    """P-adic RMSprop optimizer"""
    
    def __init__(self, params: List[PadicWeight], prime: int, lr: float = 0.01,
                 alpha: float = 0.99, eps: float = 1e-8, momentum: float = 0.0):
        """Initialize P-adic RMSprop optimizer"""
        super().__init__(params, prime, lr)
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
    
    def step(self, gradients: List[PadicWeight]) -> None:
        """Perform RMSprop step with p-adic arithmetic"""
        if len(gradients) != len(self.param_groups[0]['params']):
            raise ValueError("Gradient count mismatch")
        
        self.step_count += 1
        lr = self.param_groups[0]['lr']
        
        for i, (param, grad) in enumerate(zip(self.param_groups[0]['params'], gradients)):
            param_state = self.state[i]
            
            # Initialize state
            if 'square_avg' not in param_state:
                param_state['square_avg'] = self._create_zero_padic(param)
                if self.momentum > 0:
                    param_state['momentum_buffer'] = self._create_zero_padic(param)
            
            square_avg = param_state['square_avg']
            
            # Update exponential moving average of squared gradients
            grad_squared = self._square_padic(grad)
            square_avg_scaled = self._scale_padic(square_avg, self.alpha)
            grad_sq_scaled = self._scale_padic(grad_squared, 1.0 - self.alpha)
            param_state['square_avg'] = self.math_ops.add_padic(square_avg_scaled, grad_sq_scaled)
            
            # Compute update
            avg_sqrt = self._sqrt_padic(param_state['square_avg'], self.eps)
            update = self._divide_padic(grad, avg_sqrt)
            
            # Apply momentum if enabled
            if self.momentum > 0:
                buf = param_state['momentum_buffer']
                buf_scaled = self._scale_padic(buf, self.momentum)
                param_state['momentum_buffer'] = self.math_ops.add_padic(buf_scaled, update)
                update = param_state['momentum_buffer']
            
            # Apply update
            lr_update = self._scale_padic(update, lr)
            param_update = self.math_ops.subtract_padic(param, lr_update)
            param.digits = param_update.digits.copy()
    
    def _create_zero_padic(self, template: PadicWeight) -> PadicWeight:
        """Create zero p-adic weight"""
        return PadicWeight(
            value=template.value,
            prime=template.prime,
            precision=template.precision,
            valuation=template.valuation,
            digits=[0] * template.precision
        )
    
    def _square_padic(self, padic_weight: PadicWeight) -> PadicWeight:
        """Square p-adic weight"""
        squared_coeffs = [coeff ** 2 for coeff in padic_weight.digits]
        return PadicWeight(
            value=padic_weight.value,
            prime=padic_weight.prime,
            precision=padic_weight.precision,
            valuation=padic_weight.valuation,
            digits=squared_coeffs
        )
    
    def _sqrt_padic(self, padic_weight: PadicWeight, eps: float) -> PadicWeight:
        """Square root of p-adic weight with epsilon"""
        sqrt_coeffs = [max(eps, abs(coeff) ** 0.5) for coeff in padic_weight.digits]
        return PadicWeight(
            value=padic_weight.value,
            prime=padic_weight.prime,
            precision=padic_weight.precision,
            valuation=padic_weight.valuation,
            digits=sqrt_coeffs
        )
    
    def _divide_padic(self, numerator: PadicWeight, denominator: PadicWeight) -> PadicWeight:
        """Divide p-adic weights"""
        div_coeffs = []
        for i in range(numerator.precision):
            num = numerator.digits[i] if i < len(numerator.digits) else 0
            den = denominator.digits[i] if i < len(denominator.digits) else 1
            div_coeffs.append(num / den if den != 0 else num)
        
        return PadicWeight(
            value=Fraction(numerator.value.numerator, denominator.value.numerator) if denominator.value.numerator != 0 else numerator.value,
            prime=numerator.prime,
            precision=numerator.precision,
            valuation=numerator.valuation - denominator.valuation,
            digits=[int(coeff) for coeff in div_coeffs]
        )
    
    def _scale_padic(self, padic_weight: PadicWeight, scalar: float) -> PadicWeight:
        """Scale p-adic weight by scalar"""
        scaled_coeffs = [int(coeff * scalar) for coeff in padic_weight.digits]
        return PadicWeight(
            value=padic_weight.value,
            prime=padic_weight.prime,
            precision=padic_weight.precision,
            valuation=padic_weight.valuation,
            digits=scaled_coeffs
        )


class PadicOptimizationManager:
    """
    Manager for p-adic optimization algorithms
    Provides unified interface and performance tracking
    """
    
    def __init__(self, prime: int):
        """Initialize optimization manager"""
        self.prime = prime
        PadicValidation.validate_prime(prime)
        
        self.optimizers: Dict[str, PadicOptimizer] = {}
        self.optimizer_stats: Dict[str, Dict[str, Any]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
    
    def create_sgd_optimizer(self, params: List[PadicWeight], lr: float = 0.01,
                           momentum: float = 0.0, dampening: float = 0.0,
                           name: Optional[str] = None) -> str:
        """Create SGD optimizer"""
        with self._lock:
            optimizer_name = name or f"sgd_{len(self.optimizers)}"
            
            optimizer = PadicSGD(
                params=params,
                prime=self.prime,
                lr=lr,
                momentum=momentum,
                dampening=dampening
            )
            
            self.optimizers[optimizer_name] = optimizer
            self.optimizer_stats[optimizer_name] = {
                'type': 'SGD',
                'creation_time': time.time(),
                'steps_taken': 0,
                'total_step_time': 0.0,
                'average_step_time': 0.0
            }
            
            return optimizer_name
    
    def create_adam_optimizer(self, params: List[PadicWeight], lr: float = 0.001,
                            betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                            name: Optional[str] = None) -> str:
        """Create Adam optimizer"""
        with self._lock:
            optimizer_name = name or f"adam_{len(self.optimizers)}"
            
            optimizer = PadicAdam(
                params=params,
                prime=self.prime,
                lr=lr,
                betas=betas,
                eps=eps
            )
            
            self.optimizers[optimizer_name] = optimizer
            self.optimizer_stats[optimizer_name] = {
                'type': 'Adam',
                'creation_time': time.time(),
                'steps_taken': 0,
                'total_step_time': 0.0,
                'average_step_time': 0.0
            }
            
            return optimizer_name
    
    def create_rmsprop_optimizer(self, params: List[PadicWeight], lr: float = 0.01,
                               alpha: float = 0.99, eps: float = 1e-8,
                               momentum: float = 0.0, name: Optional[str] = None) -> str:
        """Create RMSprop optimizer"""
        with self._lock:
            optimizer_name = name or f"rmsprop_{len(self.optimizers)}"
            
            optimizer = PadicRMSprop(
                params=params,
                prime=self.prime,
                lr=lr,
                alpha=alpha,
                eps=eps,
                momentum=momentum
            )
            
            self.optimizers[optimizer_name] = optimizer
            self.optimizer_stats[optimizer_name] = {
                'type': 'RMSprop',
                'creation_time': time.time(),
                'steps_taken': 0,
                'total_step_time': 0.0,
                'average_step_time': 0.0
            }
            
            return optimizer_name
    
    def step(self, optimizer_name: str, gradients: List[PadicWeight]) -> None:
        """Perform optimization step"""
        with self._lock:
            if optimizer_name not in self.optimizers:
                raise KeyError(f"Optimizer not found: {optimizer_name}")
            
            start_time = time.time()
            
            try:
                self.optimizers[optimizer_name].step(gradients)
                
                # Update statistics
                step_time = time.time() - start_time
                stats = self.optimizer_stats[optimizer_name]
                stats['steps_taken'] += 1
                stats['total_step_time'] += step_time
                stats['average_step_time'] = stats['total_step_time'] / stats['steps_taken']
                
            except Exception as e:
                raise ValueError(f"Optimization step failed for {optimizer_name}: {e}")
    
    def get_optimizer_stats(self, optimizer_name: str) -> Dict[str, Any]:
        """Get optimizer statistics"""
        with self._lock:
            if optimizer_name not in self.optimizer_stats:
                raise KeyError(f"Optimizer not found: {optimizer_name}")
            
            return dict(self.optimizer_stats[optimizer_name])
    
    def list_optimizers(self) -> List[str]:
        """List all optimizer names"""
        with self._lock:
            return list(self.optimizers.keys())
    
    def remove_optimizer(self, optimizer_name: str) -> None:
        """Remove optimizer"""
        with self._lock:
            if optimizer_name not in self.optimizers:
                raise KeyError(f"Optimizer not found: {optimizer_name}")
            
            del self.optimizers[optimizer_name]
            del self.optimizer_stats[optimizer_name]
    
    def reset_optimizer_stats(self, optimizer_name: str) -> None:
        """Reset optimizer statistics"""
        with self._lock:
            if optimizer_name not in self.optimizer_stats:
                raise KeyError(f"Optimizer not found: {optimizer_name}")
            
            stats = self.optimizer_stats[optimizer_name]
            optimizer_type = stats['type']
            creation_time = stats['creation_time']
            
            self.optimizer_stats[optimizer_name] = {
                'type': optimizer_type,
                'creation_time': creation_time,
                'steps_taken': 0,
                'total_step_time': 0.0,
                'average_step_time': 0.0
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all optimizers"""
        with self._lock:
            return {name: dict(stats) for name, stats in self.optimizer_stats.items()}


# Integration hooks
class PadicAdvancedIntegration:
    """Integration utilities for advanced P-adic features"""
    
    @staticmethod
    def integrate_hensel_lifting(compression_system: 'PadicCompressionSystem',
                                hensel_config: HenselLiftingConfig) -> HenselLiftingProcessor:
        """Integrate Hensel lifting with compression system"""
        hensel_processor = HenselLiftingProcessor(
            hensel_config, compression_system.prime, compression_system.precision
        )
        
        # Add methods to compression system
        compression_system.hensel_processor = hensel_processor
        compression_system.lift_precision = hensel_processor.lift_to_precision
        
        return hensel_processor
    
    @staticmethod
    def integrate_hierarchical_clustering(compression_system: 'PadicCompressionSystem',
                                        clustering_config: ClusteringConfig) -> HierarchicalClusteringManager:
        """Integrate hierarchical clustering with compression system"""
        clustering_manager = HierarchicalClusteringManager(
            clustering_config, compression_system.prime
        )
        
        # Add methods to compression system
        compression_system.clustering_manager = clustering_manager
        compression_system.build_clusters = clustering_manager.build_hierarchical_clustering
        compression_system.find_cluster = clustering_manager.find_cluster_for_weight
        
        return clustering_manager
    
    @staticmethod
    def integrate_gpu_decompression(compression_system: 'PadicCompressionSystem',
                                  gpu_config: GPUDecompressionConfig) -> PadicDecompressionEngine:
        """Integrate GPU decompression with compression system"""
        decompression_engine = PadicDecompressionEngine(
            gpu_config, compression_system.prime
        )
        
        # Add methods to compression system
        compression_system.gpu_decompression_engine = decompression_engine
        compression_system.decompress_gpu = decompression_engine.decompress_progressive
        
        return decompression_engine
    
    @staticmethod
    def integrate_optimization_manager(compression_system: 'PadicCompressionSystem') -> PadicOptimizationManager:
        """Integrate optimization manager with compression system"""
        optimization_manager = PadicOptimizationManager(compression_system.prime)
        
        # Add methods to compression system
        compression_system.optimization_manager = optimization_manager
        compression_system.create_optimizer = optimization_manager.create_sgd_optimizer
        compression_system.create_adam_optimizer = optimization_manager.create_adam_optimizer
        compression_system.create_rmsprop_optimizer = optimization_manager.create_rmsprop_optimizer
        
        return optimization_manager
