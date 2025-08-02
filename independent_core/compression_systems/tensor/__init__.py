"""
Tensor Decomposition Compression System Module
Production-ready implementation with NO FALLBACKS
"""

from .tensor_core import (
    TensorCompressionSystem,
    TuckerDecomposer,
    CPDecomposer,
    TensorTrainDecomposer,
    TensorValidator,
    DecompositionType,
    TensorDecomposition,
    TensorRankOptimizer,
    TensorDecompositionIntegration
)

from .tensor_service_layer import (
    TensorServiceInterface,
    TensorServiceRegistry,
    TensorServiceOrchestrator,
    TensorServiceIntegration,
    TensorServiceValidation,
    TensorServiceMetrics,
    TensorServiceMiddleware,
    TensorServiceConfiguration,
    TensorServiceHealthMonitor,
    TensorServiceCache,
    TensorServiceSecurity,
    TensorServiceLoadBalancer,
    TensorServiceLogger,
    TensorServiceRateLimiter,
    TensorServiceRetryManager,
    ServiceStatus,
    CircuitBreakerState,
    ServiceEndpoint,
    ServiceContract,
    ServiceHealth
)

__all__ = [
    # Core tensor classes
    'TensorCompressionSystem',
    'TuckerDecomposer',
    'CPDecomposer', 
    'TensorTrainDecomposer',
    'TensorValidator',
    'DecompositionType',
    'TensorDecomposition',
    'TensorRankOptimizer',
    'TensorDecompositionIntegration',
    # Service layer classes
    'TensorServiceInterface',
    'TensorServiceRegistry',
    'TensorServiceOrchestrator',
    'TensorServiceIntegration',
    'TensorServiceValidation',
    'TensorServiceMetrics',
    'TensorServiceMiddleware',
    'TensorServiceConfiguration',
    'TensorServiceHealthMonitor',
    'TensorServiceCache',
    'TensorServiceSecurity',
    'TensorServiceLoadBalancer',
    'TensorServiceLogger',
    'TensorServiceRateLimiter',
    'TensorServiceRetryManager',
    # Service supporting classes
    'ServiceStatus',
    'CircuitBreakerState',
    'ServiceEndpoint',
    'ServiceContract',
    'ServiceHealth'
]

# Module version
__version__ = '1.0.0'

# Module metadata
__author__ = 'Independent Core Development Team'
__description__ = 'Tensor decomposition based compression system for independent_core'
__status__ = 'Production'