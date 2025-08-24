"""
Ultrametric Tree Data Structure
Efficient hierarchical data structure for p-adic and ultrametric spaces
"""

from typing import Any, Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import math

@dataclass
class UltrametricNode:
    """Node in an ultrametric tree"""
    key: Any
    value: Any
    distance: float = 0.0
    children: Dict[Any, 'UltrametricNode'] = field(default_factory=dict)
    parent: Optional['UltrametricNode'] = None
    
class UltrametricTree:
    """
    Ultrametric tree for efficient nearest neighbor searches in ultrametric spaces.
    
    In ultrametric spaces, the strong triangle inequality holds:
    d(x,z) <= max(d(x,y), d(y,z))
    
    This allows for efficient hierarchical organization of data.
    """
    
    def __init__(self, metric_func=None):
        """
        Initialize ultrametric tree.
        
        Args:
            metric_func: Function to compute ultrametric distance between keys
        """
        self.root = UltrametricNode(key=None, value=None)
        self.metric_func = metric_func or self._default_metric
        self._size = 0
        self._node_map = {}  # Quick lookup by key
    
    def _default_metric(self, x, y):
        """Default ultrametric: discrete metric"""
        return 0.0 if x == y else 1.0
    
    def insert(self, key: Any, value: Any) -> None:
        """
        Insert a key-value pair into the tree.
        
        Args:
            key: Key to insert
            value: Associated value
        """
        if key in self._node_map:
            # Update existing node
            self._node_map[key].value = value
            return
        
        # Create new node
        new_node = UltrametricNode(key=key, value=value)
        
        if self._size == 0:
            # First node becomes child of root
            self.root.children[key] = new_node
            new_node.parent = self.root
            new_node.distance = 0.0
        else:
            # Find best insertion point
            best_parent = self._find_best_parent(key)
            distance = self.metric_func(key, best_parent.key) if best_parent.key is not None else 0.0
            
            best_parent.children[key] = new_node
            new_node.parent = best_parent
            new_node.distance = distance
        
        self._node_map[key] = new_node
        self._size += 1
    
    def _find_best_parent(self, key: Any) -> UltrametricNode:
        """
        Find the best parent node for inserting a new key.
        
        Uses the ultrametric property to maintain tree structure.
        """
        if not self.root.children:
            return self.root
        
        # Find node with minimum distance
        min_dist = float('inf')
        best_node = self.root
        
        for node_key, node in self._node_map.items():
            dist = self.metric_func(key, node_key)
            if dist < min_dist:
                min_dist = dist
                best_node = node
        
        return best_node
    
    def search(self, key: Any) -> Optional[Any]:
        """
        Search for a key in the tree.
        
        Args:
            key: Key to search for
            
        Returns:
            Associated value if found, None otherwise
        """
        node = self._node_map.get(key)
        return node.value if node else None
    
    def delete(self, key: Any) -> bool:
        """
        Delete a key from the tree.
        
        Args:
            key: Key to delete
            
        Returns:
            True if deleted, False if not found
        """
        if key not in self._node_map:
            return False
        
        node = self._node_map[key]
        
        # If node has children, promote them to parent
        if node.children:
            parent = node.parent
            for child_key, child in node.children.items():
                child.parent = parent
                if parent:
                    parent.children[child_key] = child
        
        # Remove from parent's children
        if node.parent:
            del node.parent.children[key]
        
        # Remove from node map
        del self._node_map[key]
        self._size -= 1
        
        return True
    
    def nearest_neighbor(self, key: Any, k: int = 1) -> List[Tuple[Any, Any, float]]:
        """
        Find k nearest neighbors to a given key.
        
        Args:
            key: Query key
            k: Number of neighbors to find
            
        Returns:
            List of (key, value, distance) tuples
        """
        if self._size == 0:
            return []
        
        # Compute distances to all nodes
        distances = []
        for node_key, node in self._node_map.items():
            if node_key != key:
                dist = self.metric_func(key, node_key)
                distances.append((node_key, node.value, dist))
        
        # Sort by distance and return top k
        distances.sort(key=lambda x: x[2])
        return distances[:k]
    
    def range_search(self, key: Any, radius: float) -> List[Tuple[Any, Any, float]]:
        """
        Find all nodes within a given radius of a key.
        
        Args:
            key: Query key
            radius: Search radius
            
        Returns:
            List of (key, value, distance) tuples
        """
        results = []
        
        for node_key, node in self._node_map.items():
            if node_key != key:
                dist = self.metric_func(key, node_key)
                if dist <= radius:
                    results.append((node_key, node.value, dist))
        
        return results
    
    def __len__(self) -> int:
        """Return the number of nodes in the tree"""
        return self._size
    
    def __contains__(self, key: Any) -> bool:
        """Check if a key exists in the tree"""
        return key in self._node_map
    
    def items(self) -> List[Tuple[Any, Any]]:
        """Return all key-value pairs in the tree"""
        return [(key, node.value) for key, node in self._node_map.items()]
    
    def keys(self) -> List[Any]:
        """Return all keys in the tree"""
        return list(self._node_map.keys())
    
    def values(self) -> List[Any]:
        """Return all values in the tree"""
        return [node.value for node in self._node_map.values()]
    
    def clear(self) -> None:
        """Clear all nodes from the tree"""
        self.root = UltrametricNode(key=None, value=None)
        self._node_map.clear()
        self._size = 0
    
    def height(self) -> int:
        """Compute the height of the tree"""
        if self._size == 0:
            return 0
        
        def _node_height(node: UltrametricNode) -> int:
            if not node.children:
                return 0
            return 1 + max(_node_height(child) for child in node.children.values())
        
        return _node_height(self.root)
    
    def is_ultrametric(self) -> bool:
        """
        Verify that the tree satisfies the ultrametric property.
        
        Returns:
            True if tree structure is ultrametric, False otherwise
        """
        # Check strong triangle inequality for all triples
        keys = list(self._node_map.keys())
        
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                for k in range(j + 1, len(keys)):
                    d_ij = self.metric_func(keys[i], keys[j])
                    d_jk = self.metric_func(keys[j], keys[k])
                    d_ik = self.metric_func(keys[i], keys[k])
                    
                    # Strong triangle inequality: d(i,k) <= max(d(i,j), d(j,k))
                    if d_ik > max(d_ij, d_jk) + 1e-10:  # Small tolerance for floating point
                        return False
        
        return True

__all__ = ['UltrametricTree', 'UltrametricNode']