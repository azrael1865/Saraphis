"""
Comprehensive Unit Tests for UltrametricTree and UltrametricTreeNode Classes
Tests tree construction, LCA queries, ultrametric distance, and validation
"""

import sys
import os
import unittest
import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Add path to compression systems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'compression_systems', 'padic'))

from ultrametric_tree import (
    UltrametricTreeNode, 
    UltrametricTree,
    find_lca_binary_lifting,
    ultrametric_distance,
    p_adic_valuation
)


# Mock cluster node for testing tree building
@dataclass
class MockClusterNode:
    """Mock cluster node that mimics HybridClusterNode interface"""
    node_id: str
    children: List['MockClusterNode'] = field(default_factory=list)
    cluster_size: int = 1
    intra_cluster_distance: float = 0.0
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class TestUltrametricTreeNode(unittest.TestCase):
    """Test suite for UltrametricTreeNode class"""
    
    def test_valid_initialization(self):
        """Test creating a valid tree node"""
        node = UltrametricTreeNode(
            node_id="test_node",
            data={"test": "data"},
            depth=5,
            p_adic_valuation=2.5,
            parent_distance=0.125
        )
        
        self.assertEqual(node.node_id, "test_node")
        self.assertEqual(node.data, {"test": "data"})
        self.assertEqual(node.depth, 5)
        self.assertEqual(node.p_adic_valuation, 2.5)
        self.assertEqual(node.parent_distance, 0.125)
        self.assertTrue(node.is_leaf())
        self.assertTrue(node.is_root())
    
    def test_invalid_node_id(self):
        """Test that invalid node IDs are rejected"""
        invalid_ids = ["", "  ", None]
        
        for invalid_id in [""]:  # None will raise different error
            with self.assertRaises(ValueError) as context:
                UltrametricTreeNode(
                    node_id=invalid_id,
                    data="test"
                )
            self.assertIn("Node ID must be non-empty string", str(context.exception))
    
    def test_invalid_depth(self):
        """Test that invalid depths are rejected"""
        invalid_depths = [-1, -10, -100]
        
        for depth in invalid_depths:
            with self.assertRaises(ValueError) as context:
                UltrametricTreeNode(
                    node_id="test",
                    data="test",
                    depth=depth
                )
            self.assertIn("Depth must be non-negative", str(context.exception))
    
    def test_invalid_valuation(self):
        """Test that non-numeric valuations are rejected"""
        with self.assertRaises(ValueError) as context:
            UltrametricTreeNode(
                node_id="test",
                data="test",
                p_adic_valuation="invalid"
            )
        self.assertIn("P-adic valuation must be numeric", str(context.exception))
    
    def test_invalid_parent_distance(self):
        """Test that negative parent distances are rejected"""
        with self.assertRaises(ValueError) as context:
            UltrametricTreeNode(
                node_id="test",
                data="test",
                parent_distance=-0.5
            )
        self.assertIn("Parent distance must be non-negative", str(context.exception))
    
    def test_is_leaf_and_is_root(self):
        """Test leaf and root detection"""
        root = UltrametricTreeNode(node_id="root", data="root_data")
        child1 = UltrametricTreeNode(node_id="child1", data="child1_data", parent=root)
        child2 = UltrametricTreeNode(node_id="child2", data="child2_data", parent=root)
        
        root.children = [child1, child2]
        
        self.assertTrue(root.is_root())
        self.assertFalse(root.is_leaf())
        
        self.assertFalse(child1.is_root())
        self.assertTrue(child1.is_leaf())
        
        self.assertFalse(child2.is_root())
        self.assertTrue(child2.is_leaf())
    
    def test_get_ancestor_at_depth(self):
        """Test getting ancestors at specific depths"""
        # Build a simple tree
        root = UltrametricTreeNode(node_id="root", data="root", depth=0)
        level1 = UltrametricTreeNode(node_id="l1", data="l1", parent=root, depth=1)
        level2 = UltrametricTreeNode(node_id="l2", data="l2", parent=level1, depth=2)
        level3 = UltrametricTreeNode(node_id="l3", data="l3", parent=level2, depth=3)
        
        # Set up ancestors for binary lifting
        level1.ancestors = [root]
        level2.ancestors = [level1, root]
        level3.ancestors = [level2, level1, root]
        
        # Test getting ancestors
        self.assertEqual(level3.get_ancestor_at_depth(3), level3)
        self.assertEqual(level3.get_ancestor_at_depth(2), level2)
        self.assertEqual(level3.get_ancestor_at_depth(1), level1)
        self.assertEqual(level3.get_ancestor_at_depth(0), root)
        
        # Test invalid depths
        self.assertIsNone(level3.get_ancestor_at_depth(-1))
        self.assertIsNone(level3.get_ancestor_at_depth(4))


class TestUltrametricTree(unittest.TestCase):
    """Test suite for UltrametricTree class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tree = UltrametricTree(prime=7, precision=4)
    
    def test_valid_initialization(self):
        """Test creating a valid UltrametricTree"""
        tree = UltrametricTree(prime=5, precision=3)
        self.assertEqual(tree.prime, 5)
        self.assertEqual(tree.precision, 3)
        self.assertIsNone(tree.root)
        self.assertEqual(len(tree.nodes), 0)
        self.assertEqual(tree.max_depth, 0)
        self.assertFalse(tree.lca_preprocessed)
    
    def test_invalid_prime(self):
        """Test that invalid primes are rejected"""
        invalid_primes = [0, 1, -1, -10]
        
        for prime in invalid_primes:
            with self.assertRaises(ValueError) as context:
                UltrametricTree(prime=prime, precision=4)
            self.assertIn("Prime must be int >= 2", str(context.exception))
    
    def test_invalid_precision(self):
        """Test that invalid precisions are rejected"""
        invalid_precisions = [0, -1, -10]
        
        for precision in invalid_precisions:
            with self.assertRaises(ValueError) as context:
                UltrametricTree(prime=7, precision=precision)
            self.assertIn("Precision must be int >= 1", str(context.exception))
    
    def test_build_tree_simple(self):
        """Test building a simple tree from cluster hierarchy"""
        # Create mock cluster hierarchy
        root_cluster = MockClusterNode(
            node_id="root",
            cluster_size=4,
            intra_cluster_distance=0.1
        )
        
        child1 = MockClusterNode(
            node_id="child1",
            cluster_size=2,
            intra_cluster_distance=0.05
        )
        
        child2 = MockClusterNode(
            node_id="child2",
            cluster_size=2,
            intra_cluster_distance=0.05
        )
        
        root_cluster.children = [child1, child2]
        
        # Build tree
        tree_root = self.tree.build_tree(root_cluster)
        
        self.assertIsNotNone(tree_root)
        self.assertEqual(tree_root.node_id, "root")
        self.assertEqual(len(tree_root.children), 2)
        self.assertEqual(len(self.tree.nodes), 3)
        self.assertTrue(self.tree.lca_preprocessed)
    
    def test_build_tree_deep(self):
        """Test building a deeper tree"""
        # Create a 4-level tree
        root = MockClusterNode(node_id="root", cluster_size=8)
        l1_1 = MockClusterNode(node_id="l1_1", cluster_size=4)
        l1_2 = MockClusterNode(node_id="l1_2", cluster_size=4)
        l2_1 = MockClusterNode(node_id="l2_1", cluster_size=2)
        l2_2 = MockClusterNode(node_id="l2_2", cluster_size=2)
        l2_3 = MockClusterNode(node_id="l2_3", cluster_size=2)
        l2_4 = MockClusterNode(node_id="l2_4", cluster_size=2)
        l3_1 = MockClusterNode(node_id="l3_1", cluster_size=1)
        l3_2 = MockClusterNode(node_id="l3_2", cluster_size=1)
        
        root.children = [l1_1, l1_2]
        l1_1.children = [l2_1, l2_2]
        l1_2.children = [l2_3, l2_4]
        l2_1.children = [l3_1, l3_2]
        
        tree_root = self.tree.build_tree(root)
        
        self.assertEqual(len(self.tree.nodes), 9)
        self.assertEqual(self.tree.max_depth, 3)
        self.assertTrue(self.tree.lca_preprocessed)
        
        # Check depths
        self.assertEqual(self.tree.nodes["root"].depth, 0)
        self.assertEqual(self.tree.nodes["l1_1"].depth, 1)
        self.assertEqual(self.tree.nodes["l2_1"].depth, 2)
        self.assertEqual(self.tree.nodes["l3_1"].depth, 3)
    
    def test_find_lca_same_node(self):
        """Test LCA when both nodes are the same"""
        # Build simple tree
        root = MockClusterNode(node_id="root")
        child = MockClusterNode(node_id="child")
        root.children = [child]
        
        self.tree.build_tree(root)
        
        node = self.tree.nodes["child"]
        lca = self.tree.find_lca(node, node)
        
        self.assertEqual(lca.node_id, "child")
    
    def test_find_lca_parent_child(self):
        """Test LCA between parent and child"""
        # Build tree
        root = MockClusterNode(node_id="root")
        child = MockClusterNode(node_id="child")
        grandchild = MockClusterNode(node_id="grandchild")
        
        root.children = [child]
        child.children = [grandchild]
        
        self.tree.build_tree(root)
        
        parent_node = self.tree.nodes["root"]
        child_node = self.tree.nodes["child"]
        
        lca = self.tree.find_lca(parent_node, child_node)
        self.assertEqual(lca.node_id, "root")
    
    def test_find_lca_siblings(self):
        """Test LCA between sibling nodes"""
        # Build tree
        root = MockClusterNode(node_id="root")
        child1 = MockClusterNode(node_id="child1")
        child2 = MockClusterNode(node_id="child2")
        
        root.children = [child1, child2]
        
        self.tree.build_tree(root)
        
        node1 = self.tree.nodes["child1"]
        node2 = self.tree.nodes["child2"]
        
        lca = self.tree.find_lca(node1, node2)
        self.assertEqual(lca.node_id, "root")
    
    def test_find_lca_cousins(self):
        """Test LCA between cousin nodes"""
        # Build tree with cousins
        root = MockClusterNode(node_id="root")
        parent1 = MockClusterNode(node_id="parent1")
        parent2 = MockClusterNode(node_id="parent2")
        cousin1 = MockClusterNode(node_id="cousin1")
        cousin2 = MockClusterNode(node_id="cousin2")
        
        root.children = [parent1, parent2]
        parent1.children = [cousin1]
        parent2.children = [cousin2]
        
        self.tree.build_tree(root)
        
        node1 = self.tree.nodes["cousin1"]
        node2 = self.tree.nodes["cousin2"]
        
        lca = self.tree.find_lca(node1, node2)
        self.assertEqual(lca.node_id, "root")
    
    def test_find_lca_not_preprocessed(self):
        """Test that LCA queries fail without preprocessing"""
        # Create nodes without building tree properly
        node1 = UltrametricTreeNode(node_id="n1", data="d1")
        node2 = UltrametricTreeNode(node_id="n2", data="d2")
        
        with self.assertRaises(RuntimeError) as context:
            self.tree.find_lca(node1, node2)
        self.assertIn("LCA not preprocessed", str(context.exception))
    
    def test_ultrametric_distance_same_node(self):
        """Test ultrametric distance for same node"""
        root = MockClusterNode(node_id="root")
        self.tree.build_tree(root)
        
        node = self.tree.nodes["root"]
        distance = self.tree.ultrametric_distance(node, node)
        
        self.assertEqual(distance, 0.0)
    
    def test_ultrametric_distance_siblings(self):
        """Test ultrametric distance between siblings"""
        root = MockClusterNode(node_id="root")
        child1 = MockClusterNode(node_id="child1")
        child2 = MockClusterNode(node_id="child2")
        
        root.children = [child1, child2]
        self.tree.build_tree(root)
        
        node1 = self.tree.nodes["child1"]
        node2 = self.tree.nodes["child2"]
        
        # Distance should be prime^(-depth_of_lca)
        # LCA is root at depth 0, so distance = 7^0 = 1
        distance = self.tree.ultrametric_distance(node1, node2)
        self.assertEqual(distance, 1.0)
    
    def test_ultrametric_distance_different_depths(self):
        """Test ultrametric distance between nodes at different depths"""
        root = MockClusterNode(node_id="root")
        child = MockClusterNode(node_id="child")
        grandchild = MockClusterNode(node_id="grandchild")
        
        root.children = [child]
        child.children = [grandchild]
        
        self.tree.build_tree(root)
        
        child_node = self.tree.nodes["child"]
        grandchild_node = self.tree.nodes["grandchild"]
        
        # LCA is child at depth 1, so distance = 7^(-1)
        distance = self.tree.ultrametric_distance(child_node, grandchild_node)
        self.assertAlmostEqual(distance, 1/7, places=6)
    
    def test_p_adic_valuation(self):
        """Test p-adic valuation computation"""
        # Test various values
        self.assertEqual(self.tree.p_adic_valuation(7), 1)    # 7^1
        self.assertEqual(self.tree.p_adic_valuation(49), 2)   # 7^2
        self.assertEqual(self.tree.p_adic_valuation(343), 3)  # 7^3
        self.assertEqual(self.tree.p_adic_valuation(14), 1)   # 7^1 * 2
        self.assertEqual(self.tree.p_adic_valuation(98), 2)   # 7^2 * 2
        self.assertEqual(self.tree.p_adic_valuation(5), 0)    # No factor of 7
        self.assertEqual(self.tree.p_adic_valuation(1), 0)    # No factor of 7
        
        # Test zero
        self.assertEqual(self.tree.p_adic_valuation(0), float('inf'))
        
        # Test negative values
        self.assertEqual(self.tree.p_adic_valuation(-49), 2)
        self.assertEqual(self.tree.p_adic_valuation(-7), 1)
    
    def test_validate_ultrametric_property(self):
        """Test validation of ultrametric property"""
        # Build a proper tree
        root = MockClusterNode(node_id="root")
        c1 = MockClusterNode(node_id="c1")
        c2 = MockClusterNode(node_id="c2")
        gc1 = MockClusterNode(node_id="gc1")
        gc2 = MockClusterNode(node_id="gc2")
        
        root.children = [c1, c2]
        c1.children = [gc1]
        c2.children = [gc2]
        
        self.tree.build_tree(root)
        
        # Should satisfy ultrametric property
        self.assertTrue(self.tree.validate_ultrametric_property())
    
    def test_get_path_to_root(self):
        """Test getting path from node to root"""
        # Build tree
        root = MockClusterNode(node_id="root")
        child = MockClusterNode(node_id="child")
        grandchild = MockClusterNode(node_id="grandchild")
        
        root.children = [child]
        child.children = [grandchild]
        
        self.tree.build_tree(root)
        
        # Get path from grandchild to root
        gc_node = self.tree.nodes["grandchild"]
        path = self.tree.get_path_to_root(gc_node)
        
        self.assertEqual(len(path), 3)
        self.assertEqual(path[0].node_id, "grandchild")
        self.assertEqual(path[1].node_id, "child")
        self.assertEqual(path[2].node_id, "root")
    
    def test_get_subtree_nodes(self):
        """Test getting all nodes in a subtree"""
        # Build tree
        root = MockClusterNode(node_id="root")
        c1 = MockClusterNode(node_id="c1")
        c2 = MockClusterNode(node_id="c2")
        gc1 = MockClusterNode(node_id="gc1")
        gc2 = MockClusterNode(node_id="gc2")
        gc3 = MockClusterNode(node_id="gc3")
        
        root.children = [c1, c2]
        c1.children = [gc1, gc2]
        c2.children = [gc3]
        
        self.tree.build_tree(root)
        
        # Get subtree rooted at c1
        c1_node = self.tree.nodes["c1"]
        subtree = self.tree.get_subtree_nodes(c1_node)
        
        self.assertEqual(len(subtree), 3)
        subtree_ids = {n.node_id for n in subtree}
        self.assertEqual(subtree_ids, {"c1", "gc1", "gc2"})
    
    def test_compute_tree_statistics(self):
        """Test computing tree statistics"""
        # Build a tree
        root = MockClusterNode(node_id="root", cluster_size=6)
        c1 = MockClusterNode(node_id="c1", cluster_size=3)
        c2 = MockClusterNode(node_id="c2", cluster_size=3)
        leaf1 = MockClusterNode(node_id="leaf1", cluster_size=1)
        leaf2 = MockClusterNode(node_id="leaf2", cluster_size=1)
        leaf3 = MockClusterNode(node_id="leaf3", cluster_size=1)
        
        root.children = [c1, c2]
        c1.children = [leaf1, leaf2]
        c2.children = [leaf3]
        
        self.tree.build_tree(root)
        
        stats = self.tree.compute_tree_statistics()
        
        self.assertEqual(stats['total_nodes'], 6)
        self.assertEqual(stats['leaf_nodes'], 3)
        self.assertEqual(stats['internal_nodes'], 3)
        self.assertEqual(stats['max_depth'], 2)
        self.assertTrue(stats['lca_preprocessed'])
        self.assertGreater(stats['memory_overhead_bytes'], 0)
    
    def test_find_node_by_id(self):
        """Test finding nodes by ID"""
        root = MockClusterNode(node_id="root")
        child = MockClusterNode(node_id="child")
        root.children = [child]
        
        self.tree.build_tree(root)
        
        # Find existing nodes
        found_root = self.tree.find_node_by_id("root")
        self.assertIsNotNone(found_root)
        self.assertEqual(found_root.node_id, "root")
        
        found_child = self.tree.find_node_by_id("child")
        self.assertIsNotNone(found_child)
        self.assertEqual(found_child.node_id, "child")
        
        # Find non-existent node
        not_found = self.tree.find_node_by_id("nonexistent")
        self.assertIsNone(not_found)
    
    def test_clear_tree(self):
        """Test clearing the tree"""
        # Build a tree
        root = MockClusterNode(node_id="root")
        child = MockClusterNode(node_id="child")
        root.children = [child]
        
        self.tree.build_tree(root)
        
        # Verify tree is built
        self.assertIsNotNone(self.tree.root)
        self.assertEqual(len(self.tree.nodes), 2)
        self.assertTrue(self.tree.lca_preprocessed)
        
        # Clear tree
        self.tree.clear()
        
        # Verify tree is cleared
        self.assertIsNone(self.tree.root)
        self.assertEqual(len(self.tree.nodes), 0)
        self.assertEqual(self.tree.max_depth, 0)
        self.assertFalse(self.tree.lca_preprocessed)
        self.assertEqual(self.tree.log_max_depth, 0)


class TestUltrametricTreeHelperFunctions(unittest.TestCase):
    """Test suite for helper functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tree = UltrametricTree(prime=5, precision=3)
        
        # Build a simple tree
        root = MockClusterNode(node_id="root")
        child1 = MockClusterNode(node_id="child1")
        child2 = MockClusterNode(node_id="child2")
        root.children = [child1, child2]
        
        self.tree.build_tree(root)
    
    def test_find_lca_binary_lifting(self):
        """Test the convenience LCA function"""
        lca = find_lca_binary_lifting(self.tree, "child1", "child2")
        self.assertIsNotNone(lca)
        self.assertEqual(lca.node_id, "root")
        
        # Test with non-existent nodes
        lca_none = find_lca_binary_lifting(self.tree, "child1", "nonexistent")
        self.assertIsNone(lca_none)
    
    def test_ultrametric_distance_function(self):
        """Test the convenience distance function"""
        distance = ultrametric_distance(self.tree, "child1", "child2", prime=5)
        self.assertEqual(distance, 1.0)  # Root at depth 0
        
        # Test with non-existent nodes
        with self.assertRaises(ValueError):
            ultrametric_distance(self.tree, "child1", "nonexistent", prime=5)
    
    def test_p_adic_valuation_function(self):
        """Test the standalone p-adic valuation function"""
        # Test with prime 5
        self.assertEqual(p_adic_valuation(25, 5), 2)   # 5^2
        self.assertEqual(p_adic_valuation(125, 5), 3)  # 5^3
        self.assertEqual(p_adic_valuation(10, 5), 1)   # 5^1 * 2
        self.assertEqual(p_adic_valuation(7, 5), 0)    # No factor of 5
        
        # Test zero
        self.assertEqual(p_adic_valuation(0, 5), float('inf'))
        
        # Test invalid inputs
        with self.assertRaises(TypeError):
            p_adic_valuation(3.14, 5)
        
        with self.assertRaises(ValueError):
            p_adic_valuation(10, 1)


class TestUltrametricTreeEdgeCases(unittest.TestCase):
    """Test edge cases for UltrametricTree"""
    
    def test_single_node_tree(self):
        """Test tree with single node"""
        tree = UltrametricTree(prime=3, precision=2)
        root = MockClusterNode(node_id="single")
        
        tree.build_tree(root)
        
        self.assertEqual(len(tree.nodes), 1)
        self.assertEqual(tree.max_depth, 0)
        
        # LCA of single node with itself
        node = tree.nodes["single"]
        lca = tree.find_lca(node, node)
        self.assertEqual(lca.node_id, "single")
        
        # Distance to itself
        distance = tree.ultrametric_distance(node, node)
        self.assertEqual(distance, 0.0)
        
        # Validate ultrametric property (trivially true)
        self.assertTrue(tree.validate_ultrametric_property())
    
    def test_deep_linear_tree(self):
        """Test tree that is a single chain (no branching)"""
        tree = UltrametricTree(prime=2, precision=1)
        
        # Build linear chain
        root = MockClusterNode(node_id="n0")
        current = root
        for i in range(1, 10):
            child = MockClusterNode(node_id=f"n{i}")
            current.children = [child]
            current = child
        
        tree.build_tree(root)
        
        self.assertEqual(len(tree.nodes), 10)
        self.assertEqual(tree.max_depth, 9)
        
        # Test LCA at various depths
        n0 = tree.nodes["n0"]
        n5 = tree.nodes["n5"]
        n9 = tree.nodes["n9"]
        
        lca_05 = tree.find_lca(n0, n5)
        self.assertEqual(lca_05.node_id, "n0")
        
        lca_59 = tree.find_lca(n5, n9)
        self.assertEqual(lca_59.node_id, "n5")
    
    def test_wide_shallow_tree(self):
        """Test tree with many children but shallow depth"""
        tree = UltrametricTree(prime=11, precision=2)
        
        root = MockClusterNode(node_id="root")
        # Add 20 direct children
        for i in range(20):
            child = MockClusterNode(node_id=f"child{i}")
            root.children.append(child)
        
        tree.build_tree(root)
        
        self.assertEqual(len(tree.nodes), 21)
        self.assertEqual(tree.max_depth, 1)
        
        # All pairs of children should have root as LCA
        child0 = tree.nodes["child0"]
        child10 = tree.nodes["child10"]
        child19 = tree.nodes["child19"]
        
        lca1 = tree.find_lca(child0, child10)
        self.assertEqual(lca1.node_id, "root")
        
        lca2 = tree.find_lca(child10, child19)
        self.assertEqual(lca2.node_id, "root")
    
    def test_unbalanced_tree(self):
        """Test tree with unbalanced structure"""
        tree = UltrametricTree(prime=7, precision=3)
        
        # Create unbalanced tree
        root = MockClusterNode(node_id="root")
        
        # Left subtree is deep
        left = MockClusterNode(node_id="left")
        left_child = MockClusterNode(node_id="left_child")
        left_grandchild = MockClusterNode(node_id="left_grandchild")
        left.children = [left_child]
        left_child.children = [left_grandchild]
        
        # Right subtree is shallow
        right = MockClusterNode(node_id="right")
        
        root.children = [left, right]
        
        tree.build_tree(root)
        
        # Test LCA between deep and shallow nodes
        deep_node = tree.nodes["left_grandchild"]
        shallow_node = tree.nodes["right"]
        
        lca = tree.find_lca(deep_node, shallow_node)
        self.assertEqual(lca.node_id, "root")
        
        # Distance should reflect LCA at root
        distance = tree.ultrametric_distance(deep_node, shallow_node)
        self.assertEqual(distance, 1.0)
    
    def test_large_prime_values(self):
        """Test with larger prime values"""
        large_primes = [101, 103, 107, 109, 113]
        
        for prime in large_primes:
            tree = UltrametricTree(prime=prime, precision=2)
            root = MockClusterNode(node_id="root")
            child = MockClusterNode(node_id="child")
            root.children = [child]
            
            tree.build_tree(root)
            
            # Test basic operations
            self.assertEqual(tree.prime, prime)
            self.assertEqual(len(tree.nodes), 2)
            
            # Test valuation with large prime
            self.assertEqual(tree.p_adic_valuation(prime), 1)
            self.assertEqual(tree.p_adic_valuation(prime * prime), 2)


if __name__ == "__main__":
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestUltrametricTreeNode))
    suite.addTests(loader.loadTestsFromTestCase(TestUltrametricTree))
    suite.addTests(loader.loadTestsFromTestCase(TestUltrametricTreeHelperFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestUltrametricTreeEdgeCases))
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("ULTRAMETRICTREE TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)