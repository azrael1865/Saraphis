"""
Saraphis Analytical Compression Systems Framework

Defines interfaces and structure for compression algorithms.
Fail-loud architecture - no fallbacks.
"""

# Import tensor decomposition components
# Temporarily disabled due to missing gpytorch dependency
# from .tensor_decomposition import (
#     HOSVDDecomposer,
#     TensorRingDecomposer,
#     AdvancedTensorRankOptimizer,
#     TensorGPUAccelerator
# )

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

# Import system integration coordinator
from .system_integration_coordinator import (
    SystemIntegrationCoordinator,
    SystemConfiguration,
    OptimizationStrategy,
    CompressionRequest,
    CompressionResult,
    create_compression_system,
    load_compression_system
)

# Create alias for integration test compatibility
MasterSystemCoordinator = SystemIntegrationCoordinator

# Import GPU memory management components
from .gpu_memory.smart_pool import SmartPool
from .gpu_memory.auto_swap_manager import AutoSwapManager as AutoSwap

__all__ = [
    # Tensor decomposition - temporarily disabled
    # 'HOSVDDecomposer',
    # 'TensorRingDecomposer',
    # 'AdvancedTensorRankOptimizer',
    # 'TensorGPUAccelerator',
    
    # P-adic compression
    'PadicCompressionSystem',
    'PadicGradientCompressor',
    'PadicAdvancedIntegration',
    
    # Sheaf compression
    'SheafCompressionSystem',
    'SheafAdvancedIntegration',
    'SheafSystemOrchestrator',
    
    # System integration
    'SystemIntegrationCoordinator',
    'MasterSystemCoordinator',
    'SystemConfiguration',
    'OptimizationStrategy',
    'CompressionRequest',
    'CompressionResult',
    'create_compression_system',
    'load_compression_system',
    
    # GPU memory management
    'SmartPool',
    'AutoSwap'
]

__version__ = '2.0.0'  # Updated for complete system integration