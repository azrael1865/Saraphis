"""
P-adic compression system for neural network weights.
Hard failure mode - no fallbacks, all errors throw immediately.
"""

from .padic_encoder import PadicWeight, PadicValidation, PadicMathematicalOperations
from .padic_compressor import PadicCompressionSystem
from .padic_gradient import PadicGradientCompressor
from .padic_advanced import (
    HenselLiftingConfig,
    HenselLiftingProcessor,
    ClusteringConfig,
    ClusterNode,
    HierarchicalClusteringManager,
    GPUDecompressionConfig,
    PadicDecompressionEngine,
    PadicOptimizer,
    PadicSGD,
    PadicAdam,
    PadicRMSprop,
    PadicOptimizationManager,
    PadicAdvancedIntegration
)
from .padic_integration import (
    PadicIntegrationConfig,
    PadicGACIntegration,
    PadicBrainIntegration,
    PadicTrainingIntegration,
    PadicSystemOrchestrator,
    initialize_padic_integration,
    get_orchestrator,
    shutdown_padic_integration
)
from .padic_service_layer import (
    PadicServiceInterface,
    PadicServiceManager,
    PadicServiceValidator,
    PadicServiceMetrics,
    PadicServiceConfig,
    PadicServiceMethod
)
from .padic_service_config import PadicServiceConfiguration
from .hybrid_padic_structures import (
    HybridPadicWeight,
    HybridPadicValidator,
    HybridPadicConverter,
    HybridPadicManager
)
from .hybrid_padic_compressor import (
    HybridPadicCompressionSystem,
    HybridPadicIntegrationManager
)
from .hybrid_padic_gpu_ops import (
    GPUOperationConfig,
    HybridPadicGPUOps,
    HybridPadicGPUOptimizer,
    HybridPadicGPUManager
)
from .memory_pressure_handler import (
    MemoryPressureHandler,
    PressureHandlerConfig,
    ProcessingMode,
    MemoryState,
    MemoryMetrics,
    PerformanceMetrics,
    integrate_memory_pressure_handler
)

__all__ = [
    'PadicWeight',
    'PadicValidation', 
    'PadicMathematicalOperations',
    'PadicCompressionSystem',
    'PadicGradientCompressor',
    'HenselLiftingConfig',
    'HenselLiftingProcessor',
    'ClusteringConfig',
    'ClusterNode',
    'HierarchicalClusteringManager',
    'GPUDecompressionConfig',
    'PadicDecompressionEngine',
    'PadicOptimizer',
    'PadicSGD',
    'PadicAdam',
    'PadicRMSprop',
    'PadicOptimizationManager',
    'PadicAdvancedIntegration',
    'PadicIntegrationConfig',
    'PadicGACIntegration',
    'PadicBrainIntegration',
    'PadicTrainingIntegration',
    'PadicSystemOrchestrator',
    'initialize_padic_integration',
    'get_orchestrator',
    'shutdown_padic_integration',
    'PadicServiceInterface',
    'PadicServiceManager',
    'PadicServiceValidator',
    'PadicServiceMetrics',
    'PadicServiceConfig',
    'PadicServiceMethod',
    'PadicServiceConfiguration',
    'HybridPadicWeight',
    'HybridPadicValidator',
    'HybridPadicConverter',
    'HybridPadicManager',
    'HybridPadicCompressionSystem',
    'HybridPadicIntegrationManager',
    'GPUOperationConfig',
    'HybridPadicGPUOps',
    'HybridPadicGPUOptimizer',
    'HybridPadicGPUManager',
    'MemoryPressureHandler',
    'PressureHandlerConfig',
    'ProcessingMode',
    'MemoryState',
    'MemoryMetrics',
    'PerformanceMetrics',
    'integrate_memory_pressure_handler'
]