"""
Saraphis Analytical Compression Systems Framework

Defines interfaces and structure for compression algorithms.
Fail-loud architecture - no fallbacks.
"""

# Import tensor decomposition components
from .tensor_decomposition import (
    HOSVDDecomposer,
    TensorRingDecomposer,
    AdvancedTensorRankOptimizer,
    TensorGPUAccelerator
)

# Import P-adic compression components
from .padic import (
    PadicCompressionSystem,
    PadicGradientCompressor,
    PadicAdvancedIntegration
)

# Import sheaf compression components
from .sheaf import (
    SheafCompressionSystem,
    SheafAdvancedIntegration,
    SheafSystemOrchestrator
)

__all__ = [
    # Tensor decomposition
    'HOSVDDecomposer',
    'TensorRingDecomposer',
    'AdvancedTensorRankOptimizer',
    'TensorGPUAccelerator',
    
    # P-adic compression
    'PadicCompressionSystem',
    'PadicGradientCompressor',
    'PadicAdvancedIntegration',
    
    # Sheaf compression
    'SheafCompressionSystem',
    'SheafAdvancedIntegration',
    'SheafSystemOrchestrator'
]

__version__ = '1.0.0'