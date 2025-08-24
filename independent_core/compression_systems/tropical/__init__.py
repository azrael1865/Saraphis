"""
Tropical mathematics module for neural network compression.
Implements max-plus algebra operations for piecewise linear approximations.
"""

from .tropical_core import (
    # Constants
    TROPICAL_ZERO,
    TROPICAL_EPSILON,
    
    # Core classes
    TropicalNumber,
    TropicalValidation,
    TropicalMathematicalOperations,
    TropicalGradientTracker,
    
    # Utility functions
    is_tropical_zero,
    to_tropical_safe,
    from_tropical_safe,
    tropical_distance,
)

__all__ = [
    # Constants
    'TROPICAL_ZERO',
    'TROPICAL_EPSILON',
    
    # Core classes
    'TropicalNumber',
    'TropicalValidation', 
    'TropicalMathematicalOperations',
    'TropicalGradientTracker',
    
    # Utility functions
    'is_tropical_zero',
    'to_tropical_safe',
    'from_tropical_safe',
    'tropical_distance',
]