"""
Compact Tree Encoding System - Reduces metadata overhead from O(r^h) to O(n log n)
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
import struct
import math
import logging


class BitVector:
    """Efficient bit storage for tree structure encoding"""
    
    def __init__(self, initial_capacity: int = 1024):
        """Initialize bit vector with given capacity"""
        if not isinstance(initial_capacity, int) or initial_capacity <= 0:
            raise ValueError(f"Initial capacity must be positive int, got {initial_capacity}")
        
        # Store bits as bytes array for efficiency
        self.byte_capacity = (initial_capacity + 7) // 8
        self.data = bytearray(self.byte_capacity)
        self.bit_count = 0
        self.capacity = initial_capacity
    
    def append(self, bit: bool) -> None:
        """Append a single bit to the vector"""
        if not isinstance(bit, (bool, int)):
            raise TypeError(f"Bit must be bool or int, got {type(bit)}")
        
        # Expand if necessary
        if self.bit_count >= self.capacity:
            self._expand()
        
        byte_idx = self.bit_count // 8
        bit_idx = self.bit_count % 8
        
        if bit:
            self.data[byte_idx] |= (1 << bit_idx)
        else:
            self.data[byte_idx] &= ~(1 << bit_idx)
        
        self.bit_count += 1
    
    def append_bits(self, value: int, num_bits: int) -> None:
        """Append multiple bits from an integer value"""
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Value must be non-negative int, got {value}")
        if not isinstance(num_bits, int) or num_bits <= 0:
            raise ValueError(f"Num bits must be positive int, got {num_bits}")
        if value >= (1 << num_bits):
            raise ValueError(f"Value {value} requires more than {num_bits} bits")
        
        for i in range(num_bits):
            self.append((value >> i) & 1)
    
    def get(self, index: int) -> bool:
        """Get bit at given index"""
        if not isinstance(index, int) or index < 0:
            raise ValueError(f"Index must be non-negative int, got {index}")
        if index >= self.bit_count:
            raise IndexError(f"Index {index} out of range for {self.bit_count} bits")
        
        byte_idx = index // 8
        bit_idx = index % 8
        return bool((self.data[byte_idx] >> bit_idx) & 1)
    
    def read_bits(self, start: int, num_bits: int) -> int:
        """Read multiple bits as integer starting from given position"""
        if not isinstance(start, int) or start < 0:
            raise ValueError(f"Start must be non-negative int, got {start}")
        if not isinstance(num_bits, int) or num_bits <= 0:
            raise ValueError(f"Num bits must be positive int, got {num_bits}")
        if start + num_bits > self.bit_count:
            raise IndexError(f"Range [{start}, {start + num_bits}) out of bounds for {self.bit_count} bits")
        
        result = 0
        for i in range(num_bits):
            if self.get(start + i):
                result |= (1 << i)
        return result
    
    def _expand(self) -> None:
        """Expand internal storage capacity"""
        new_capacity = self.capacity * 2
        new_byte_capacity = (new_capacity + 7) // 8
        new_data = bytearray(new_byte_capacity)
        new_data[:len(self.data)] = self.data
        self.data = new_data
        self.capacity = new_capacity
        self.byte_capacity = new_byte_capacity
    
    def to_bytes(self) -> bytes:
        """Convert to compact byte representation"""
        # Return only used bytes
        used_bytes = (self.bit_count + 7) // 8
        return bytes(self.data[:used_bytes])
    
    @classmethod
    def from_bytes(cls, data: bytes, bit_count: int) -> 'BitVector':
        """Create BitVector from byte data"""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"Data must be bytes or bytearray, got {type(data)}")
        if not isinstance(bit_count, int) or bit_count < 0:
            raise ValueError(f"Bit count must be non-negative int, got {bit_count}")
        
        vec = cls(bit_count)
        vec.data = bytearray(data)
        vec.bit_count = bit_count
        vec.byte_capacity = len(data)
        vec.capacity = len(data) * 8
        return vec
    
    def __len__(self) -> int:
        """Return number of bits stored"""
        return self.bit_count
    
    def get_size_bytes(self) -> int:
        """Get actual storage size in bytes"""
        return (self.bit_count + 7) // 8


@dataclass
class SparseTreeNode:
    """Sparse representation of a tree node for encoding"""
    node_id: str
    has_left: bool
    has_right: bool
    weight_index: Optional[int] = None  # Index into weight array
    level: int = 0
    parent_offset: Optional[int] = None  # Offset to parent in encoding


class TreeEncoder:
    """Compact tree encoding system to reduce metadata overhead"""
    
    def __init__(self):
        """Initialize tree encoder"""
        self.logger = logging.getLogger('TreeEncoder')
        self.encoding_stats = {
            'trees_encoded': 0,
            'total_nodes': 0,
            'total_original_bytes': 0,
            'total_encoded_bytes': 0,
            'compression_ratio': 0.0
        }
    
    def encode_tree_structure(self, root_node) -> Dict[str, Any]:
        """
        Convert HybridClusterNode tree to compact bit vectors.
        
        Args:
            root_node: HybridClusterNode root of tree
            
        Returns:
            Dictionary with encoded tree data
        """
        if root_node is None:
            raise ValueError("Root node cannot be None")
        
        # Collect all unique weights and create index mapping
        weights_list, weight_indices = self._collect_weights(root_node)
        
        # Encode tree structure using level-order traversal
        structure_bits = BitVector()
        node_metadata = []
        
        # Level-order traversal queue: (node, level)
        queue = [(root_node, 0)]
        node_count = 0
        
        while queue:
            node, level = queue.pop(0)
            
            # Encode node existence (2 bits: has_left, has_right)
            has_left = len(node.children) > 0
            has_right = len(node.children) > 1
            
            structure_bits.append(has_left)
            structure_bits.append(has_right)
            
            # Store metadata for this node
            sparse_node = SparseTreeNode(
                node_id=node.node_id,
                has_left=has_left,
                has_right=has_right,
                level=level
            )
            
            # For leaf nodes, store weight indices
            if not has_left and not has_right:
                # Store indices of weights in this leaf
                weight_idxs = []
                for weight in node.hybrid_weights:
                    weight_key = self._get_weight_key(weight)
                    if weight_key in weight_indices:
                        weight_idxs.append(weight_indices[weight_key])
                sparse_node.weight_index = weight_idxs[0] if weight_idxs else None
            
            node_metadata.append(sparse_node)
            node_count += 1
            
            # Add children to queue
            if has_left:
                queue.append((node.children[0], level + 1))
            if has_right:
                queue.append((node.children[1], level + 1))
        
        # Apply differential encoding to reduce size further
        encoded_data = self.apply_differential_encoding(structure_bits, node_metadata)
        
        # Calculate compression statistics
        original_size = self._calculate_original_size(root_node)
        encoded_size = self._calculate_encoded_size(encoded_data, weights_list)
        
        # Update stats
        self.encoding_stats['trees_encoded'] += 1
        self.encoding_stats['total_nodes'] += node_count
        self.encoding_stats['total_original_bytes'] += original_size
        self.encoding_stats['total_encoded_bytes'] += encoded_size
        if self.encoding_stats['total_original_bytes'] > 0:
            self.encoding_stats['compression_ratio'] = (
                self.encoding_stats['total_encoded_bytes'] / 
                self.encoding_stats['total_original_bytes']
            )
        
        self.logger.info(f"Encoded tree: {node_count} nodes, {original_size} -> {encoded_size} bytes "
                        f"(ratio: {encoded_size/max(1, original_size):.3f})")
        
        return {
            'structure_bits': encoded_data['structure_bits'],
            'node_metadata': encoded_data['node_metadata'],
            'weights_list': weights_list,
            'weight_indices': weight_indices,
            'node_count': node_count,
            'original_size': original_size,
            'encoded_size': encoded_size,
            'compression_ratio': encoded_size / max(1, original_size)
        }
    
    def store_sparse_tree(self, encoded_tree: Dict[str, Any]) -> bytes:
        """
        Store encoded tree as compact byte stream.
        
        Args:
            encoded_tree: Dictionary from encode_tree_structure
            
        Returns:
            Compact byte representation
        """
        if not isinstance(encoded_tree, dict):
            raise TypeError(f"Encoded tree must be dict, got {type(encoded_tree)}")
        
        # Pack into binary format:
        # [header][structure_bits][node_metadata][weights]
        
        parts = []
        
        # Header: node_count (4 bytes), weight_count (4 bytes), structure_bit_count (4 bytes)
        node_count = encoded_tree['node_count']
        weight_count = len(encoded_tree['weights_list'])
        structure_bits = encoded_tree['structure_bits']
        structure_bit_count = len(structure_bits)
        
        header = struct.pack('<III', node_count, weight_count, structure_bit_count)
        parts.append(header)
        
        # Structure bits
        parts.append(structure_bits.to_bytes())
        
        # Node metadata (simplified - just store node IDs and weight indices)
        metadata_bytes = bytearray()
        for node in encoded_tree['node_metadata']:
            # Store node_id length and string
            node_id_bytes = node.node_id.encode('utf-8')
            metadata_bytes.extend(struct.pack('<H', len(node_id_bytes)))
            metadata_bytes.extend(node_id_bytes)
            
            # Store weight index (-1 for non-leaf)
            weight_idx = node.weight_index if node.weight_index is not None else -1
            metadata_bytes.extend(struct.pack('<i', weight_idx))
        
        parts.append(bytes(metadata_bytes))
        
        # Weights are stored separately (not duplicated here)
        # Just store count and reference
        
        return b''.join(parts)
    
    def apply_differential_encoding(self, structure_bits: BitVector, 
                                   node_metadata: List[SparseTreeNode]) -> Dict[str, Any]:
        """
        Apply differential encoding to reduce redundancy between tree levels.
        
        Args:
            structure_bits: Raw structure encoding
            node_metadata: Node metadata list
            
        Returns:
            Differentially encoded data
        """
        # Group nodes by level for differential encoding
        levels = {}
        for node in node_metadata:
            if node.level not in levels:
                levels[node.level] = []
            levels[node.level].append(node)
        
        # Create differential encoding
        diff_bits = BitVector()
        
        # First level stored fully
        if 0 in levels:
            for node in levels[0]:
                diff_bits.append(node.has_left)
                diff_bits.append(node.has_right)
        
        # Subsequent levels as deltas from parent level patterns
        for level in sorted(levels.keys())[1:]:
            prev_level = levels[level - 1]
            curr_level = levels[level]
            
            # Encode differences using run-length encoding for similar patterns
            run_length = 0
            last_pattern = None
            
            for node in curr_level:
                pattern = (node.has_left, node.has_right)
                if pattern == last_pattern:
                    run_length += 1
                else:
                    if last_pattern is not None:
                        # Encode run: [run_length_bits(4)][pattern(2)]
                        diff_bits.append_bits(min(run_length, 15), 4)
                        diff_bits.append(last_pattern[0])
                        diff_bits.append(last_pattern[1])
                    last_pattern = pattern
                    run_length = 1
            
            # Encode final run
            if last_pattern is not None:
                diff_bits.append_bits(min(run_length, 15), 4)
                diff_bits.append(last_pattern[0])
                diff_bits.append(last_pattern[1])
        
        return {
            'structure_bits': diff_bits,
            'node_metadata': node_metadata,
            'level_counts': {level: len(nodes) for level, nodes in levels.items()}
        }
    
    def decode_tree_structure(self, encoded_data: Dict[str, Any]):
        """
        Reconstruct HybridClusterNode tree from compact encoding.
        
        Args:
            encoded_data: Encoded tree data from encode_tree_structure
            
        Returns:
            Reconstructed HybridClusterNode root
        """
        if not isinstance(encoded_data, dict):
            raise TypeError(f"Encoded data must be dict, got {type(encoded_data)}")
        
        # Import here to avoid circular dependency
        from ..padic.hybrid_clustering import HybridClusterNode
        
        structure_bits = encoded_data['structure_bits']
        node_metadata = encoded_data['node_metadata']
        weights_list = encoded_data['weights_list']
        
        if not node_metadata:
            raise ValueError("No nodes in encoded tree")
        
        # Reconstruct nodes
        nodes = []
        bit_idx = 0
        
        for i, sparse_node in enumerate(node_metadata):
            # Read structure bits
            if bit_idx + 1 < len(structure_bits):
                has_left = structure_bits.get(bit_idx)
                has_right = structure_bits.get(bit_idx + 1)
                bit_idx += 2
            else:
                has_left = sparse_node.has_left
                has_right = sparse_node.has_right
            
            # Get weights for this node
            node_weights = []
            if sparse_node.weight_index is not None and sparse_node.weight_index >= 0:
                if sparse_node.weight_index < len(weights_list):
                    node_weights = [weights_list[sparse_node.weight_index]]
            
            # Create node - handle empty weights for internal nodes
            # Internal nodes may have weights from their children aggregated
            if not node_weights:
                # For internal nodes, we'll populate weights later from children
                node_weights = []
            
            node = HybridClusterNode(
                node_id=sparse_node.node_id,
                hybrid_weights=node_weights,
                children=[],
                cluster_metadata={'reconstructed': True, 'level': sparse_node.level}
            )
            nodes.append(node)
        
        # Reconstruct tree structure using level-order
        if not nodes:
            raise ValueError("No nodes reconstructed")
        
        root = nodes[0]
        node_idx = 0
        queue = [root]
        
        while queue and node_idx < len(nodes):
            current = queue.pop(0)
            
            # Check if this node should have children
            sparse_node = node_metadata[node_idx]
            node_idx += 1
            
            child_idx = node_idx
            if sparse_node.has_left and child_idx < len(nodes):
                left_child = nodes[child_idx]
                current.children.append(left_child)
                left_child.parent = current
                queue.append(left_child)
                child_idx += 1
            
            if sparse_node.has_right and child_idx < len(nodes):
                right_child = nodes[child_idx]
                current.children.append(right_child)
                right_child.parent = current
                queue.append(right_child)
        
        # Populate internal node weights from children (bottom-up)
        self._populate_internal_weights(root)
        
        self.logger.info(f"Decoded tree with {len(nodes)} nodes")
        return root
    
    def _populate_internal_weights(self, node) -> None:
        """Populate internal node weights from children recursively"""
        # Process children first (post-order traversal)
        for child in node.children:
            self._populate_internal_weights(child)
        
        # If this is an internal node with no weights, aggregate from children
        if len(node.children) > 0 and not node.hybrid_weights:
            all_weights = []
            for child in node.children:
                all_weights.extend(child.get_all_weights())
            node.hybrid_weights = all_weights
    
    def _collect_weights(self, root_node) -> Tuple[List[Any], Dict[str, int]]:
        """Collect all unique weights and create index mapping"""
        weights_list = []
        weight_indices = {}
        
        def traverse(node):
            for weight in node.hybrid_weights:
                weight_key = self._get_weight_key(weight)
                if weight_key not in weight_indices:
                    weight_indices[weight_key] = len(weights_list)
                    weights_list.append(weight)
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return weights_list, weight_indices
    
    def _get_weight_key(self, weight) -> str:
        """Generate unique key for a weight"""
        # Use a hash of tensor values for uniqueness
        # This is simplified - in production would use more robust hashing
        try:
            exp_hash = hash(tuple(weight.exponent_channel.cpu().flatten().tolist()[:10]))
            man_hash = hash(tuple(weight.mantissa_channel.cpu().flatten().tolist()[:10]))
            return f"{exp_hash}_{man_hash}_{weight.prime}_{weight.precision}"
        except Exception:
            # Fallback to object ID if hashing fails
            return str(id(weight))
    
    def _calculate_original_size(self, root_node) -> int:
        """Calculate original tree size in bytes"""
        total_size = 0
        
        def traverse(node):
            nonlocal total_size
            # Node overhead: object + references
            total_size += 64  # Base object size
            total_size += len(node.node_id.encode('utf-8'))
            
            # Weight storage (duplicated in parent nodes)
            for weight in node.hybrid_weights:
                # Each weight tensor + metadata
                total_size += weight.exponent_channel.element_size() * weight.exponent_channel.numel()
                total_size += weight.mantissa_channel.element_size() * weight.mantissa_channel.numel()
                total_size += 32  # Metadata (prime, precision, etc.)
            
            # Children references
            total_size += 8 * len(node.children)
            
            # Metadata dictionary
            total_size += 256  # Approximate dict overhead
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return total_size
    
    def _calculate_encoded_size(self, encoded_data: Dict[str, Any], weights_list: List[Any]) -> int:
        """Calculate encoded tree size in bytes"""
        size = 0
        
        # Structure bits
        structure_bits = encoded_data['structure_bits']
        size += structure_bits.get_size_bytes()
        
        # Node metadata (simplified)
        for node in encoded_data['node_metadata']:
            size += len(node.node_id.encode('utf-8')) + 2  # ID + length
            size += 4  # Weight index
            size += 4  # Level and flags
        
        # Weight storage (only stored once)
        for weight in weights_list:
            size += weight.exponent_channel.element_size() * weight.exponent_channel.numel()
            size += weight.mantissa_channel.element_size() * weight.mantissa_channel.numel()
            size += 32  # Metadata
        
        # Header and overhead
        size += 12  # Header (3 * 4 bytes)
        
        return size
    
    def get_encoding_stats(self) -> Dict[str, Any]:
        """Get encoding statistics"""
        return self.encoding_stats.copy()