"""
Sheaf Theory Compression System Module
Production-ready implementation with NO FALLBACKS
"""

from .sheaf_core import (
    SheafCompressionSystem,
    CellularSheaf,
    RestrictionMap,
    SheafValidation
)
from .sheaf_advanced import (
    CellularSheafBuilder,
    RestrictionMapProcessor,
    SheafCohomologyCalculator,
    SheafReconstructionEngine,
    SheafAdvancedIntegration
)
from .sheaf_integration import (
    SheafBrainIntegration,
    SheafDomainIntegration,
    SheafTrainingIntegration,
    SheafSystemOrchestrator,
    create_integrated_sheaf_system,
    validate_sheaf_integration
)
from .sheaf_service_core import (
    SheafServiceInterface,
    SheafServiceRegistry,
    SheafServiceOrchestrator,
    SheafServiceManager,
    ServiceOperationType,
    ServiceStatus,
    ServiceRequest,
    ServiceResponse,
    LoadBalancingStrategy,
    create_sheaf_service_system
)
from .sheaf_service_integration import (
    SheafServiceIntegration,
    SheafServiceValidation,
    SheafServiceMetrics
)

__all__ = [
    'SheafCompressionSystem',
    'CellularSheaf',
    'RestrictionMap',
    'SheafValidation',
    'CellularSheafBuilder',
    'RestrictionMapProcessor',
    'SheafCohomologyCalculator',
    'SheafReconstructionEngine',
    'SheafAdvancedIntegration',
    'SheafBrainIntegration',
    'SheafDomainIntegration',
    'SheafTrainingIntegration',
    'SheafSystemOrchestrator',
    'create_integrated_sheaf_system',
    'validate_sheaf_integration',
    'SheafServiceInterface',
    'SheafServiceRegistry',
    'SheafServiceOrchestrator',
    'SheafServiceManager',
    'ServiceOperationType',
    'ServiceStatus',
    'ServiceRequest',
    'ServiceResponse',
    'LoadBalancingStrategy',
    'create_sheaf_service_system',
    'SheafServiceIntegration',
    'SheafServiceValidation',
    'SheafServiceMetrics'
]

# Module version
__version__ = '1.0.0'

# Module metadata
__author__ = 'Independent Core Development Team'
__description__ = 'Sheaf Theory based compression system for independent_core'
__status__ = 'Production'