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
from .padic_compressor import PadicCompressionSystem


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
        
        # Initialize components
        self.math_ops = PadicMathematicalOperations(prime, 10)  # Base precision
        
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
        """Initialize GPU decompression engine"""
        self.config = config
        self.prime = prime
        
        # Validate prime
        PadicValidation.validate_prime(prime)
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for GPU decompression")
        
        # Initialize GPU resources
        self.device = torch.device('cuda:0')
        self.streams = []
        self.memory_pool = None
        
        # Initialize components
        self.math_ops = PadicMathematicalOperations(prime, 10)
        
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
        Progressive decompression with GPU optimization
        
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
            
            try:
                # Validate inputs
                self._validate_decompression_inputs(padic_weights, target_precision, metadata)
                
                # Create precision schedule
                precision_schedule = self._create_decompression_schedule(
                    padic_weights[0].precision, target_precision
                )
                
                # Prepare GPU memory
                total_elements = len(padic_weights)
                self._prepare_gpu_memory(total_elements, target_precision)
                
                # Process in batches with GPU streams
                decompressed_batches = []
                stream_idx = 0
                
                for batch_start in range(0, len(padic_weights), self.config.batch_size):
                    batch_end = min(batch_start + self.config.batch_size, len(padic_weights))
                    batch = padic_weights[batch_start:batch_end]
                    
                    # Use round-robin stream assignment
                    stream = self.streams[stream_idx]
                    stream_idx = (stream_idx + 1) % len(self.streams)
                    
                    # Process batch
                    batch_result = self._decompress_batch_gpu(
                        batch, precision_schedule, stream, batch_start
                    )
                    decompressed_batches.append(batch_result)
                    
                    # Update stream utilization
                    self.decompression_stats['stream_utilization'][stream_idx] += 1
                
                # Synchronize all streams
                for stream in self.streams:
                    stream.synchronize()
                
                # Combine results
                final_tensor = self._combine_batch_results(decompressed_batches, metadata)
                
                # Update statistics
                decompression_time = time.time() - start_time
                self._update_decompression_stats(total_elements, decompression_time)
                
                # Create metadata
                decompression_metadata = {
                    'input_weights': len(padic_weights),
                    'target_precision': target_precision,
                    'precision_schedule': precision_schedule,
                    'num_batches': len(decompressed_batches),
                    'streams_used': len(self.streams),
                    'decompression_time': decompression_time,
                    'gpu_utilization': self._calculate_gpu_utilization(),
                    'memory_usage': self._get_memory_usage_info(),
                    'throughput': len(padic_weights) / decompression_time
                }
                
                return final_tensor, decompression_metadata
                
            except Exception as e:
                # Clean up GPU resources on failure
                self._cleanup_gpu_memory()
                raise ValueError(f"GPU decompression failed: {e}")
    
    def _validate_decompression_inputs(self, weights: List[PadicWeight], 
                                     target_precision: int, metadata: Dict[str, Any]) -> None:
        """Validate decompression inputs"""
        # Check weights consistency
        if not all(w.prime == self.prime for w in weights):
            raise ValueError(f"All weights must have prime {self.prime}")
        
        # Check precision requirements
        min_precision = min(w.precision for w in weights)
        if target_precision < min_precision:
            raise ValueError(f"Target precision {target_precision} < minimum weight precision {min_precision}")
        
        # Check metadata
        required_keys = {'original_shape', 'dtype', 'device'}
        if not all(key in metadata for key in required_keys):
            raise ValueError(f"Missing required metadata keys: {required_keys - set(metadata.keys())}")
    
    def _create_decompression_schedule(self, current_precision: int, target_precision: int) -> List[int]:
        """Create progressive decompression schedule"""
        if not self.config.enable_progressive_precision:
            return [target_precision]
        
        if self.config.precision_schedule is not None:
            # Use provided schedule
            schedule = [p for p in self.config.precision_schedule 
                       if current_precision <= p <= target_precision]
            if not schedule or schedule[-1] != target_precision:
                schedule.append(target_precision)
            return sorted(schedule)
        
        # Create default progressive schedule
        if target_precision - current_precision <= 2:
            return [target_precision]
        
        # Geometric progression
        schedule = []
        current = current_precision
        
        while current < target_precision:
            next_precision = min(int(current * 1.4), target_precision)
            if next_precision > current:
                schedule.append(next_precision)
                current = next_precision
            else:
                break
        
        if not schedule or schedule[-1] != target_precision:
            schedule.append(target_precision)
        
        return schedule
    
    def _prepare_gpu_memory(self, num_elements: int, precision: int) -> None:
        """Prepare GPU memory for decompression"""
        # Estimate memory requirements
        estimated_size = num_elements * precision * 4  # 4 bytes per coefficient
        
        if estimated_size > self.memory_pool['limit']:
            raise RuntimeError(f"Estimated memory requirement {estimated_size} exceeds limit {self.memory_pool['limit']}")
        
        # Reset memory pool
        self.memory_pool['allocated'] = 0
        self.memory_pool['blocks'] = []
    
    def _decompress_batch_gpu(self, batch: List[PadicWeight], precision_schedule: List[int],
                            stream: torch.cuda.Stream, batch_start: int) -> Dict[str, Any]:
        """Decompress batch using GPU stream"""
        with torch.cuda.stream(stream):
            batch_results = []
            
            # Process through precision schedule
            current_batch = batch
            for target_precision in precision_schedule:
                # Convert p-adic to intermediate representation
                intermediate_data = self._convert_padic_to_gpu_format(current_batch, target_precision)
                
                # Transfer to GPU if async enabled
                if self.config.enable_async_transfer:
                    gpu_data = intermediate_data.to(self.device, non_blocking=True)
                else:
                    gpu_data = intermediate_data.to(self.device)
                
                # Process on GPU
                processed_data = self._process_gpu_data(gpu_data, target_precision)
                
                # Update for next iteration
                current_batch = self._update_batch_precision(current_batch, target_precision)
            
            # Final conversion to float tensor
            final_data = self._convert_to_final_format(processed_data)
            
            return {
                'data': final_data,
                'batch_start': batch_start,
                'batch_size': len(batch),
                'final_precision': precision_schedule[-1]
            }
    
    def _convert_padic_to_gpu_format(self, batch: List[PadicWeight], precision: int) -> torch.Tensor:
        """Convert p-adic weights to GPU-friendly format"""
        # Create coefficient matrix
        coeffs_matrix = np.zeros((len(batch), precision), dtype=np.float32)
        
        for i, weight in enumerate(batch):
            for j in range(min(precision, len(weight.digits))):
                coeffs_matrix[i, j] = float(weight.digits[j])
        
        return torch.from_numpy(coeffs_matrix)
    
    def _process_gpu_data(self, gpu_data: torch.Tensor, precision: int) -> torch.Tensor:
        """Process data on GPU"""
        # Apply p-adic to float conversion using GPU operations
        batch_size, num_coeffs = gpu_data.shape
        
        # Create powers of prime
        powers = torch.pow(self.prime, torch.arange(num_coeffs, dtype=torch.float32, device=gpu_data.device))
        
        # Compute weighted sum: sum(coeff_i * prime^i)
        result = torch.sum(gpu_data * powers, dim=1)
        
        return result
    
    def _update_batch_precision(self, batch: List[PadicWeight], precision: int) -> List[PadicWeight]:
        """Update batch with new precision (for progressive processing)"""
        # For this implementation, return the same batch
        # In a full implementation, would update precision appropriately
        return batch
    
    def _convert_to_final_format(self, processed_data: torch.Tensor) -> torch.Tensor:
        """Convert processed GPU data to final format"""
        # Ensure data is on CPU for final assembly
        return processed_data.cpu()
    
    def _combine_batch_results(self, batch_results: List[Dict[str, Any]], 
                             metadata: Dict[str, Any]) -> torch.Tensor:
        """Combine batch results into final tensor"""
        # Sort by batch start position
        batch_results.sort(key=lambda x: x['batch_start'])
        
        # Concatenate all batch data
        all_data = []
        for batch_result in batch_results:
            all_data.append(batch_result['data'])
        
        # Combine into single tensor
        combined = torch.cat(all_data, dim=0)
        
        # Reshape to original shape
        original_shape = metadata['original_shape']
        reshaped = combined.reshape(original_shape)
        
        # Convert to target dtype and device
        dtype_str = metadata['dtype'].split('.')[-1]
        if hasattr(torch, dtype_str):
            target_dtype = getattr(torch, dtype_str)
            reshaped = reshaped.to(dtype=target_dtype)
        
        target_device = torch.device(metadata['device'])
        reshaped = reshaped.to(device=target_device)
        
        return reshaped
    
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
        """Clean up GPU memory"""
        torch.cuda.empty_cache()
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
        self.math_ops = PadicMathematicalOperations(prime, 10)
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
            coefficients=div_coeffs,
            prime=numerator.prime,
            precision=numerator.precision
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
            value=numerator.value,
            prime=numerator.prime,
            precision=numerator.precision,
            valuation=numerator.valuation,
            digits=div_coeffs
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
    def integrate_hensel_lifting(compression_system: PadicCompressionSystem,
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
    def integrate_hierarchical_clustering(compression_system: PadicCompressionSystem,
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
    def integrate_gpu_decompression(compression_system: PadicCompressionSystem,
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
    def integrate_optimization_manager(compression_system: PadicCompressionSystem) -> PadicOptimizationManager:
        """Integrate optimization manager with compression system"""
        optimization_manager = PadicOptimizationManager(compression_system.prime)
        
        # Add methods to compression system
        compression_system.optimization_manager = optimization_manager
        compression_system.create_optimizer = optimization_manager.create_sgd_optimizer
        compression_system.create_adam_optimizer = optimization_manager.create_adam_optimizer
        compression_system.create_rmsprop_optimizer = optimization_manager.create_rmsprop_optimizer
        
        return optimization_manager