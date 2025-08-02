"""
Service Interface Definitions Module
Production-ready implementation with NO FALLBACKS
"""

from .service_interfaces_core import (
    ServiceRequest,
    ServiceResponse,
    ServiceStatus,
    ServiceHealth,
    ServiceValidation,
    ServiceMetrics,
    ServiceRegistry,
    CompressionServiceInterface,
    ServiceInterfaceIntegration
)

__all__ = [
    'ServiceRequest',
    'ServiceResponse',
    'ServiceStatus',
    'ServiceHealth',
    'ServiceValidation',
    'ServiceMetrics',
    'ServiceRegistry',
    'CompressionServiceInterface',
    'ServiceInterfaceIntegration'
]

# Module version
__version__ = '1.0.0'

# Module metadata
__author__ = 'Independent Core Development Team'
__description__ = 'Service interface definitions for independent_core compression systems'
__status__ = 'Production'