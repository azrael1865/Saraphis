"""
GPU Memory Management System Module
Production-ready implementation with NO FALLBACKS
"""

from .gpu_memory_core import (
    GPUMemoryOptimizer,
    GPUMemoryManager,
    MemoryPool,
    StreamManager,
    MemoryOptimizer,
    GPUMemoryBlock,
    MemoryBlock,
    MemoryState,
    CUDAStream,
    GPUOptimizationResult,
    KernelConfig,
    GPUMemoryIntegration
)

__all__ = [
    'GPUMemoryOptimizer',
    'GPUMemoryManager',
    'MemoryPool',
    'StreamManager',
    'MemoryOptimizer',
    'GPUMemoryBlock',
    'MemoryBlock',
    'MemoryState',
    'CUDAStream',
    'GPUOptimizationResult',
    'KernelConfig',
    'GPUMemoryIntegration'
]

# Module version
__version__ = '1.0.0'

# Module metadata
__author__ = 'Independent Core Development Team'
__description__ = 'GPU memory management system for independent_core compression'
__status__ = 'Production'