"""
Metadata Compression Module - Efficient encoding for hierarchical structures
NO FALLBACKS - HARD FAILURES ONLY
"""

from .tree_encoder import (
    BitVector,
    TreeEncoder,
    SparseTreeNode
)

__all__ = [
    'BitVector',
    'TreeEncoder', 
    'SparseTreeNode'
]