"""
Entropy Coding Module for P-adic Compression System

Provides Huffman and Arithmetic coding implementations for further compression
of p-adic digit sequences.
"""

from .huffman_arithmetic import (
    HuffmanNode,
    HuffmanEncoder,
    ArithmeticEncoder,
    HybridEncoder,
    CompressionMetrics,
    validate_reconstruction,
    run_compression_tests
)

__all__ = [
    'HuffmanNode',
    'HuffmanEncoder',
    'ArithmeticEncoder',
    'HybridEncoder',
    'CompressionMetrics',
    'validate_reconstruction',
    'run_compression_tests'
]