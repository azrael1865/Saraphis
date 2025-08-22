"""
Ultrametric Tree Implementation for P-adic Compression System
Provides O(log n) LCA queries using binary lifting
NO FALLBACKS - HARD FAILURES ONLY
"""

import math
import logging
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import deque

# Import existing p-adic structures
from padic_encoder import PadicWeight, PadicMathematicalOperations
# Forward declaration to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .hybrid_clustering import HybridClusterNode, HybridPadicWeight


@dataclass
class UltrametricTreeNode:
    """
    Node in ultrametric tree with binary lifting support
    Stores ancestors at powers of 2 for O(log n) LCA queries
    """
    node_id: str
    data: Any  # Can be PadicWeight, HybridPadicWeight, or cluster data
    parent: Optional['UltrametricTreeNode'] = None
    children: List['UltrametricTreeNode'] = field(default_factory=list)
    depth: int = 0
    p_adic_valuation: float = 0.0
    
    # Binary lifting: ancestors[i] = 2^i-th ancestor
    ancestors: List[Optional['UltrametricTreeNode']] = field(default_factory=list)
    
    # Ultrametric distance to parent
    parent_distance: float = 0.0
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate node initialization"""
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("Node ID must be non-empty string")
        if not isinstance(self.depth, int) or self.depth < 0:
            raise ValueError(f"Depth must be non-negative int, got {self.depth}")
        if not isinstance(self.p_adic_valuation, (int, float)):
            raise ValueError(f"P-adic valuation must be numeric, got {type(self.p_adic_valuation)}")
        if not isinstance(self.parent_distance, (int, float)) or self.parent_distance < 0:
            raise ValueError(f"Parent distance must be non-negative, got {self.parent_distance}")
        if not isinstance(self.children, list):
            raise TypeError(f"Children must be list, got {type(self.children)}")
        if not isinstance(self.ancestors, list):
            raise TypeError(f"Ancestors must be list, got {type(self.ancestors)}")
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """Check if this is the root node"""
        return self.parent is None
    
    def get_ancestor_at_depth(self, target_depth: int) -> Optional['UltrametricTreeNode']:
        """Get ancestor at specific depth using binary lifting"""
        if target_depth < 0 or target_depth > self.depth:
            return None
        
        current = self
        depth_diff = self.depth - target_depth
        
        # Use binary representation of depth_diff for jumping
        jump_idx = 0
        while depth_diff > 0 and current is not None:
            if depth_diff & 1:  # Check if bit is set
                if jump_idx < len(current.ancestors) and current.ancestors[jump_idx] is not None:
                    current = current.ancestors[jump_idx]
                else:
                    # Fallback to linear traversal if binary lifting not available
                    for _ in range(1 << jump_idx):
                        if current.parent is None:
                            break
                        current = current.parent
            depth_diff >>= 1
            jump_idx += 1
        
        return current


class UltrametricTree:
    """
    Ultrametric tree with O(log n) LCA queries using binary lifting
    Maintains ultrametric property: d(x,z) ≤ max(d(x,y), d(y,z))
    """
    
    def __init__(self, prime: int, precision: int = 4):
        """Initialize ultrametric tree"""
        if not isinstance(prime, int) or prime < 2:
            raise ValueError(f"Prime must be int >= 2, got {prime}")
        if not isinstance(precision, int) or precision < 1:
            raise ValueError(f"Precision must be int >= 1, got {precision}")
        
        self.prime = prime
        self.precision = precision
        self.root: Optional[UltrametricTreeNode] = None
        self.nodes: Dict[str, UltrametricTreeNode] = {}
        self.max_depth = 0
        self.logger = logging.getLogger('UltrametricTree')
        
        # Binary lifting preprocessing
        self.lca_preprocessed = False
        self.log_max_depth = 0
        
        # Initialize p-adic operations
        self.math_ops = PadicMathematicalOperations(prime, precision)
    
    def build_tree(self, cluster_root: Any) -> UltrametricTreeNode:
        """
        Build ultrametric tree from HybridClusterNode hierarchy
        
        Args:
            cluster_root: Root of cluster hierarchy (HybridClusterNode)
            
        Returns:
            Root of ultrametric tree
        """
        # Runtime type check - avoid importing HybridClusterNode
        if not hasattr(cluster_root, 'node_id') or not hasattr(cluster_root, 'children'):
            raise TypeError(f"Expected HybridClusterNode-like object, got {type(cluster_root)}")
        
        # Clear existing tree
        self.nodes.clear()
        self.max_depth = 0
        self.lca_preprocessed = False
        
        # Build tree recursively
        self.root = self._build_node_recursive(cluster_root, parent=None, depth=0)
        
        # Preprocess for LCA queries
        self.preprocess_lca()
        
        self.logger.info(f"Built ultrametric tree with {len(self.nodes)} nodes, max depth {self.max_depth}")
        
        return self.root
    
    def _build_node_recursive(self, cluster_node: Any, 
                            parent: Optional[UltrametricTreeNode], 
                            depth: int) -> UltrametricTreeNode:
        """Recursively build ultrametric tree nodes from cluster nodes"""
        # Create tree node
        tree_node = UltrametricTreeNode(
            node_id=cluster_node.node_id,
            data=cluster_node,
            parent=parent,
            depth=depth,
            metadata={
                'cluster_size': cluster_node.cluster_size,
                'is_leaf': cluster_node.is_leaf(),
                'intra_cluster_distance': cluster_node.intra_cluster_distance
            }
        )
        
        # Calculate p-adic valuation based on depth
        # Deeper nodes have higher valuation (closer to root in ultrametric sense)
        tree_node.p_adic_valuation = self._compute_node_valuation(cluster_node, depth)
        
        # Calculate ultrametric distance to parent
        if parent is not None:
            tree_node.parent_distance = self.prime ** (-depth)
        
        # Store node
        self.nodes[tree_node.node_id] = tree_node
        self.max_depth = max(self.max_depth, depth)
        
        # Process children
        for child_cluster in cluster_node.children:
            child_tree_node = self._build_node_recursive(child_cluster, tree_node, depth + 1)
            tree_node.children.append(child_tree_node)
        
        return tree_node
    
    def _compute_node_valuation(self, cluster_node: Any, depth: int) -> float:
        """Compute p-adic valuation for a node"""
        # Base valuation on depth (root has valuation 0)
        base_valuation = depth
        
        # Adjust based on cluster properties
        if cluster_node.is_leaf():
            # Leaf nodes get additional valuation based on their weight count
            base_valuation += math.log(cluster_node.cluster_size + 1) / math.log(self.prime)
        
        # Consider intra-cluster distance
        if cluster_node.intra_cluster_distance > 0:
            # Higher intra-cluster distance means lower valuation (more spread out)
            base_valuation -= math.log(1 + cluster_node.intra_cluster_distance) / math.log(self.prime)
        
        return max(0, base_valuation)
    
    def preprocess_lca(self) -> None:
        """
        Preprocess tree for O(log n) LCA queries using binary lifting
        Stores ancestors at powers of 2 for each node
        """
        if self.root is None:
            raise RuntimeError("Cannot preprocess LCA: tree not built")
        
        # Calculate log of max depth for binary lifting
        self.log_max_depth = math.ceil(math.log2(self.max_depth + 1)) if self.max_depth > 0 else 0
        
        # Initialize ancestors for all nodes
        for node in self.nodes.values():
            node.ancestors = [None] * (self.log_max_depth + 1)
            
            # First ancestor is the parent
            if node.parent is not None:
                node.ancestors[0] = node.parent
                
                # Fill in ancestors at powers of 2
                for i in range(1, self.log_max_depth + 1):
                    prev_ancestor = node.ancestors[i - 1]
                    if prev_ancestor is not None and i - 1 < len(prev_ancestor.ancestors):
                        # The 2^i-th ancestor is the 2^(i-1)-th ancestor of the 2^(i-1)-th ancestor
                        node.ancestors[i] = prev_ancestor.ancestors[i - 1]
        
        self.lca_preprocessed = True
        self.logger.info(f"LCA preprocessing complete: log_max_depth={self.log_max_depth}")
    
    def find_lca(self, node1: UltrametricTreeNode, node2: UltrametricTreeNode) -> UltrametricTreeNode:
        """
        Find lowest common ancestor in O(log n) time using binary lifting
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            Lowest common ancestor node
        """
        if not self.lca_preprocessed:
            raise RuntimeError("LCA not preprocessed. Call preprocess_lca() first")
        
        if not isinstance(node1, UltrametricTreeNode):
            raise TypeError(f"node1 must be UltrametricTreeNode, got {type(node1)}")
        if not isinstance(node2, UltrametricTreeNode):
            raise TypeError(f"node2 must be UltrametricTreeNode, got {type(node2)}")
        
        # Handle same node
        if node1.node_id == node2.node_id:
            return node1
        
        # Make sure node1 is at the same or deeper level than node2
        if node1.depth < node2.depth:
            node1, node2 = node2, node1
        
        # Bring node1 up to the same level as node2
        depth_diff = node1.depth - node2.depth
        node1 = self._jump_up(node1, depth_diff)
        
        # If they're the same after leveling, we found LCA
        if node1.node_id == node2.node_id:
            return node1
        
        # Binary search for LCA
        for i in range(self.log_max_depth, -1, -1):
            if i < len(node1.ancestors) and i < len(node2.ancestors):
                ancestor1 = node1.ancestors[i]
                ancestor2 = node2.ancestors[i]
                
                # If ancestors at 2^i are different, jump to them
                if (ancestor1 is not None and ancestor2 is not None and 
                    ancestor1.node_id != ancestor2.node_id):
                    node1 = ancestor1
                    node2 = ancestor2
        
        # The LCA is the parent of the current nodes
        if node1.parent is None:
            raise RuntimeError("LCA not found: tree structure corrupted")
        
        return node1.parent
    
    def _jump_up(self, node: UltrametricTreeNode, distance: int) -> UltrametricTreeNode:
        """Jump up the tree by specified distance using binary lifting"""
        if distance == 0:
            return node
        
        current = node
        jump_idx = 0
        
        while distance > 0 and current is not None:
            if distance & 1:  # Check if bit is set
                if jump_idx < len(current.ancestors) and current.ancestors[jump_idx] is not None:
                    current = current.ancestors[jump_idx]
                else:
                    # Should not happen if tree is properly preprocessed
                    raise RuntimeError(f"Binary lifting failed at jump_idx={jump_idx}")
            distance >>= 1
            jump_idx += 1
        
        if current is None:
            raise RuntimeError("Jump exceeded tree height")
        
        return current
    
    def ultrametric_distance(self, node1: UltrametricTreeNode, node2: UltrametricTreeNode) -> float:
        """
        Compute ultrametric distance using LCA
        Distance is prime^(-depth of LCA)
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            Ultrametric distance
        """
        if not isinstance(node1, UltrametricTreeNode):
            raise TypeError(f"node1 must be UltrametricTreeNode, got {type(node1)}")
        if not isinstance(node2, UltrametricTreeNode):
            raise TypeError(f"node2 must be UltrametricTreeNode, got {type(node2)}")
        
        # Same node has distance 0
        if node1.node_id == node2.node_id:
            return 0.0
        
        # Find LCA
        lca = self.find_lca(node1, node2)
        
        # Distance is based on depth of LCA
        # Deeper LCA means closer nodes
        distance = self.prime ** (-lca.depth)
        
        # Validate ultrametric property
        if distance <= 0 or math.isnan(distance) or math.isinf(distance):
            raise ValueError(f"Invalid ultrametric distance: {distance}")
        
        return distance
    
    def p_adic_valuation(self, value: int) -> int:
        """
        Enhanced p-adic valuation: vₚ(x) = max{k : pᵏ divides x}
        
        Args:
            value: Integer value to compute valuation for
            
        Returns:
            P-adic valuation
        """
        if not isinstance(value, int):
            raise TypeError(f"Value must be int, got {type(value)}")
        
        if value == 0:
            return float('inf')
        
        valuation = 0
        value = abs(value)
        
        while value % self.prime == 0:
            value //= self.prime
            valuation += 1
        
        return valuation
    
    def validate_ultrametric_property(self) -> bool:
        """
        Validate that the tree satisfies the ultrametric property
        For all triplets (x, y, z): d(x,z) ≤ max(d(x,y), d(y,z))
        
        Returns:
            True if ultrametric property is satisfied
        """
        if len(self.nodes) < 3:
            return True  # Trivially satisfied for < 3 nodes
        
        # Sample validation for efficiency (checking all triplets would be O(n³))
        nodes_list = list(self.nodes.values())
        num_samples = min(100, len(nodes_list))
        
        import random
        random.seed(42)  # Fixed seed for reproducibility
        
        for _ in range(num_samples):
            # Select random triplet
            triplet = random.sample(nodes_list, 3)
            x, y, z = triplet
            
            # Compute distances
            d_xy = self.ultrametric_distance(x, y)
            d_xz = self.ultrametric_distance(x, z)
            d_yz = self.ultrametric_distance(y, z)
            
            # Check ultrametric inequality
            tolerance = 1e-10
            if d_xz > max(d_xy, d_yz) + tolerance:
                self.logger.error(
                    f"Ultrametric property violated: "
                    f"d({x.node_id},{z.node_id})={d_xz:.6e} > "
                    f"max(d({x.node_id},{y.node_id})={d_xy:.6e}, "
                    f"d({y.node_id},{z.node_id})={d_yz:.6e})"
                )
                return False
        
        return True
    
    def get_path_to_root(self, node: UltrametricTreeNode) -> List[UltrametricTreeNode]:
        """Get path from node to root"""
        if not isinstance(node, UltrametricTreeNode):
            raise TypeError(f"Expected UltrametricTreeNode, got {type(node)}")
        
        path = []
        current = node
        
        while current is not None:
            path.append(current)
            current = current.parent
        
        return path
    
    def get_subtree_nodes(self, node: UltrametricTreeNode) -> List[UltrametricTreeNode]:
        """Get all nodes in subtree rooted at given node"""
        if not isinstance(node, UltrametricTreeNode):
            raise TypeError(f"Expected UltrametricTreeNode, got {type(node)}")
        
        nodes = []
        queue = deque([node])
        
        while queue:
            current = queue.popleft()
            nodes.append(current)
            queue.extend(current.children)
        
        return nodes
    
    def compute_tree_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive tree statistics"""
        if self.root is None:
            raise RuntimeError("Tree not built")
        
        leaf_nodes = [n for n in self.nodes.values() if n.is_leaf()]
        internal_nodes = [n for n in self.nodes.values() if not n.is_leaf()]
        
        # Compute average depths
        avg_leaf_depth = sum(n.depth for n in leaf_nodes) / len(leaf_nodes) if leaf_nodes else 0
        avg_internal_depth = sum(n.depth for n in internal_nodes) / len(internal_nodes) if internal_nodes else 0
        
        # Compute branching factor statistics
        branching_factors = [len(n.children) for n in internal_nodes]
        avg_branching = sum(branching_factors) / len(branching_factors) if branching_factors else 0
        max_branching = max(branching_factors) if branching_factors else 0
        
        # Compute valuation statistics
        valuations = [n.p_adic_valuation for n in self.nodes.values()]
        avg_valuation = sum(valuations) / len(valuations) if valuations else 0
        
        return {
            'total_nodes': len(self.nodes),
            'leaf_nodes': len(leaf_nodes),
            'internal_nodes': len(internal_nodes),
            'max_depth': self.max_depth,
            'average_leaf_depth': avg_leaf_depth,
            'average_internal_depth': avg_internal_depth,
            'average_branching_factor': avg_branching,
            'max_branching_factor': max_branching,
            'average_valuation': avg_valuation,
            'lca_preprocessed': self.lca_preprocessed,
            'memory_overhead_bytes': self._estimate_memory_overhead()
        }
    
    def _estimate_memory_overhead(self) -> int:
        """Estimate memory overhead of binary lifting structure"""
        # Each node stores log_max_depth ancestor pointers
        # Assume 8 bytes per pointer
        bytes_per_pointer = 8
        total_pointers = len(self.nodes) * (self.log_max_depth + 1)
        return total_pointers * bytes_per_pointer
    
    def find_node_by_id(self, node_id: str) -> Optional[UltrametricTreeNode]:
        """Find node by ID"""
        return self.nodes.get(node_id)
    
    def clear(self) -> None:
        """Clear the tree"""
        self.root = None
        self.nodes.clear()
        self.max_depth = 0
        self.lca_preprocessed = False
        self.log_max_depth = 0


def find_lca_binary_lifting(tree: UltrametricTree, node1_id: str, node2_id: str) -> Optional[UltrametricTreeNode]:
    """
    Convenience function for O(log n) LCA using binary lifting
    
    Args:
        tree: Ultrametric tree
        node1_id: ID of first node
        node2_id: ID of second node
        
    Returns:
        LCA node or None if not found
    """
    node1 = tree.find_node_by_id(node1_id)
    node2 = tree.find_node_by_id(node2_id)
    
    if node1 is None or node2 is None:
        return None
    
    return tree.find_lca(node1, node2)


def ultrametric_distance(tree: UltrametricTree, node1_id: str, node2_id: str, prime: int) -> float:
    """
    Compute ultrametric distance using LCA
    
    Args:
        tree: Ultrametric tree
        node1_id: ID of first node
        node2_id: ID of second node
        prime: Prime number for p-adic system
        
    Returns:
        Ultrametric distance
    """
    node1 = tree.find_node_by_id(node1_id)
    node2 = tree.find_node_by_id(node2_id)
    
    if node1 is None or node2 is None:
        raise ValueError(f"Nodes not found: {node1_id}, {node2_id}")
    
    return tree.ultrametric_distance(node1, node2)


def p_adic_valuation(value: int, prime: int) -> int:
    """
    Compute p-adic valuation
    
    Args:
        value: Integer value
        prime: Prime number
        
    Returns:
        P-adic valuation
    """
    if not isinstance(value, int):
        raise TypeError(f"Value must be int, got {type(value)}")
    if not isinstance(prime, int) or prime < 2:
        raise ValueError(f"Prime must be int >= 2, got {prime}")
    
    if value == 0:
        return float('inf')
    
    k = 0
    value = abs(value)
    while value % prime == 0:
        value //= prime
        k += 1
    
    return k