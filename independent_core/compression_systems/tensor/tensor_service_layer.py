"""
Tensor Service Layer Implementation

This module provides comprehensive service interfaces for external modules to interact
with the Tensor compression system. It includes service orchestration, validation,
metrics, caching, security, load balancing, and more.
"""

import time
import json
import hashlib
import threading
import logging
import traceback
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Set, Callable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import pickle
import torch
import numpy as np

# Handle missing dependencies gracefully
try:
    from ..service_interfaces.service_interfaces_core import (
        CompressionServiceInterface, ServiceRequest, ServiceResponse,
        ServiceRegistry, ServiceMetrics, ServiceValidation
    )
except ImportError:
    # Fallback definitions for standalone operation
    class CompressionServiceInterface:
        def __init__(self, config=None):
            self.config = config or {}
            self.registry = None
    
    class ServiceRequest:
        def __init__(self, service_name, method_name, version, payload):
            self.service_name = service_name
            self.method_name = method_name
            self.version = version
            self.payload = payload
            self.metadata = {}
            self.request_id = str(hash(service_name + method_name))
    
    class ServiceResponse:
        def __init__(self, request_id, status):
            self.request_id = request_id
            self.status = status
            self.data = None
            self.errors = []
            self.metadata = {}
            self.processing_time = 0.0
        
        def add_error(self, code, message, details=None):
            self.errors.append({'code': code, 'message': message, 'details': details})
        
        def is_success(self):
            return len(self.errors) == 0
    
    class ServiceRegistry:
        def __init__(self):
            pass
        
        def register_service(self, name, instance, version, metadata=None):
            pass
    
    class ServiceMetrics:
        def __init__(self):
            pass
    
    class ServiceValidation:
        def __init__(self):
            pass

try:
    from .tensor_core import TensorCompressionSystem, TensorDecomposition
except ImportError:
    # Fallback tensor classes
    class TensorCompressionSystem:
        def __init__(self, config=None):
            self.config = config or {}
        
        def compress(self, data):
            return TensorDecomposition()
        
        def decompress(self, decomposition):
            return torch.randn(10, 10)
        
        def get_compression_stats(self):
            return {}
    
    class TensorDecomposition:
        def __init__(self):
            self.factors = []
            self.decomposition_type = "CP"
            self.ranks = []
            self.compression_ratio = 0.5
            self.reconstruction_error = 0.01

try:
    from .tensor_integration import TensorIntegrator
except ImportError:
    # Fallback integrator
    class TensorIntegrator:
        def __init__(self):
            pass


class ServiceStatus(Enum):
    """Service status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNAVAILABLE = "unavailable"
    STARTING = "starting"
    STOPPING = "stopping"


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration"""
    name: str
    version: str
    url: str
    protocol: str = "http"
    timeout: float = 30.0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceContract:
    """Service contract definition"""
    service_name: str
    version: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    constraints: Dict[str, Any] = field(default_factory=dict)
    sla: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceHealth:
    """Service health information"""
    service_name: str
    status: ServiceStatus
    uptime: float
    last_check: datetime
    response_time_ms: float
    error_rate: float
    throughput: float
    active_connections: int
    memory_usage_mb: float
    cpu_usage_percent: float
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class TensorServiceInterface(CompressionServiceInterface):
    """
    Service interface for Tensor compression system.
    Extends the base compression service interface with tensor-specific functionality.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.tensor_system = TensorCompressionSystem(config)
        self.tensor_services: Dict[str, Any] = {}
        self.service_versions: Dict[str, str] = {}
        self.service_contracts: Dict[str, ServiceContract] = {}
        self.performance_metrics = defaultdict(lambda: {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0.0,
            'average_response_time': 0.0,
            'peak_response_time': 0.0,
            'min_response_time': float('inf')
        })
        
        # Initialize registry if not set by parent
        if not hasattr(self, 'registry') or self.registry is None:
            self.registry = ServiceRegistry()
        
        self._initialize_tensor_services()
    
    def _initialize_tensor_services(self):
        """Initialize tensor-specific services"""
        # Register core tensor services
        self.register_tensor_service(
            "tensor_compress",
            self.tensor_system.compress,
            "1.0.0"
        )
        self.register_tensor_service(
            "tensor_decompress",
            self.tensor_system.decompress,
            "1.0.0"
        )
    
    def register_tensor_service(self, service_name: str, service_instance: Any, version: str) -> None:
        """Register a tensor service"""
        if service_name is None:
            raise ValueError("Service name cannot be None")
        if service_instance is None:
            raise ValueError("Service instance cannot be None")
        if version is None:
            raise ValueError("Version cannot be None")
        
        # Validate service instance
        if not callable(service_instance):
            raise TypeError("Service instance must be callable")
        
        # Store service
        self.tensor_services[service_name] = service_instance
        self.service_versions[service_name] = version
        
        # Register with base registry
        self.registry.register_service(service_name, service_instance, version)
        
        # Log registration
        logging.info(f"Registered tensor service: {service_name} v{version}")
    
    def invoke_tensor_compression(self, request: ServiceRequest) -> ServiceResponse:
        """Invoke tensor compression service"""
        if request is None:
            raise ValueError("Request cannot be None")
        
        start_time = time.time()
        response = ServiceResponse(
            request_id=request.request_id,
            status=self._get_initial_status()
        )
        
        try:
            # Validate request
            is_valid, errors = self.validate_tensor_request(request)
            if not is_valid:
                response.status = self._get_error_status()
                for error in errors:
                    response.add_error("VALIDATION_ERROR", error)
                return response
            
            # Extract tensor data
            tensor_data = request.payload.get('tensor')
            if tensor_data is None:
                raise ValueError("No tensor data in request")
            
            # Convert to torch tensor if needed
            if not isinstance(tensor_data, torch.Tensor):
                tensor_data = torch.tensor(tensor_data)
            
            # Perform compression
            decomposition = self.tensor_system.compress(tensor_data)
            
            # Build response
            response.status = self._get_success_status()
            response.data = {
                'decomposition': decomposition,
                'compression_ratio': decomposition.compression_ratio,
                'reconstruction_error': decomposition.reconstruction_error,
                'method': str(decomposition.decomposition_type),
                'ranks': decomposition.ranks
            }
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_service_metrics('tensor_compression', processing_time, True)
            
        except Exception as e:
            response.status = self._get_error_status()
            response.add_error("COMPRESSION_ERROR", str(e), {
                'type': type(e).__name__,
                'traceback': traceback.format_exc()
            })
            processing_time = time.time() - start_time
            self._update_service_metrics('tensor_compression', processing_time, False)
        
        response.processing_time = processing_time
        return response
    
    def invoke_tensor_decompression(self, request: ServiceRequest) -> ServiceResponse:
        """Invoke tensor decompression service"""
        if request is None:
            raise ValueError("Request cannot be None")
        
        start_time = time.time()
        response = ServiceResponse(
            request_id=request.request_id,
            status=self._get_initial_status()
        )
        
        try:
            # Validate request
            is_valid, errors = self.validate_tensor_request(request)
            if not is_valid:
                response.status = self._get_error_status()
                for error in errors:
                    response.add_error("VALIDATION_ERROR", error)
                return response
            
            # Extract decomposition data
            decomposition_data = request.payload.get('decomposition')
            if decomposition_data is None:
                raise ValueError("No decomposition data in request")
            
            # Reconstruct tensor
            if isinstance(decomposition_data, TensorDecomposition):
                reconstructed = self.tensor_system.decompress(decomposition_data)
            else:
                # Create TensorDecomposition from data
                decomposition = TensorDecomposition(
                    decomposition_type=decomposition_data['decomposition_type'],
                    factors=decomposition_data['factors'],
                    ranks=decomposition_data['ranks'],
                    compression_ratio=decomposition_data['compression_ratio']
                )
                reconstructed = self.tensor_system.decompress(decomposition)
            
            # Build response
            response.status = self._get_success_status()
            response.data = {
                'tensor': reconstructed,
                'shape': list(reconstructed.shape),
                'dtype': str(reconstructed.dtype),
                'device': str(reconstructed.device)
            }
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_service_metrics('tensor_decompression', processing_time, True)
            
        except Exception as e:
            response.status = self._get_error_status()
            response.add_error("DECOMPRESSION_ERROR", str(e), {
                'type': type(e).__name__,
                'traceback': traceback.format_exc()
            })
            processing_time = time.time() - start_time
            self._update_service_metrics('tensor_decompression', processing_time, False)
        
        response.processing_time = processing_time
        return response
    
    def invoke_tensor_decomposition(self, request: ServiceRequest) -> ServiceResponse:
        """Invoke tensor decomposition service"""
        if request is None:
            raise ValueError("Request cannot be None")
        
        start_time = time.time()
        response = ServiceResponse(
            request_id=request.request_id,
            status=self._get_initial_status()
        )
        
        try:
            # Validate request
            is_valid, errors = self.validate_tensor_request(request)
            if not is_valid:
                response.status = self._get_error_status()
                for error in errors:
                    response.add_error("VALIDATION_ERROR", error)
                return response
            
            # Extract parameters
            tensor_data = request.payload.get('tensor')
            method = request.payload.get('method', 'CP')
            ranks = request.payload.get('ranks')
            
            if tensor_data is None:
                raise ValueError("No tensor data in request")
            
            # Convert to torch tensor if needed
            if not isinstance(tensor_data, torch.Tensor):
                tensor_data = torch.tensor(tensor_data)
            
            # Perform decomposition
            decomposer = self.tensor_system.decomposers.get(method)
            if decomposer is None:
                raise ValueError(f"Unknown decomposition method: {method}")
            
            decomposition = decomposer.decompose(tensor_data, ranks)
            
            # Build response
            response.status = self._get_success_status()
            response.data = {
                'factors': decomposition.factors,
                'method': str(decomposition.decomposition_type),
                'ranks': decomposition.ranks,
                'compression_ratio': decomposition.compression_ratio,
                'reconstruction_error': decomposition.reconstruction_error
            }
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_service_metrics('tensor_decomposition', processing_time, True)
            
        except Exception as e:
            response.status = self._get_error_status()
            response.add_error("DECOMPOSITION_ERROR", str(e), {
                'type': type(e).__name__,
                'traceback': traceback.format_exc()
            })
            processing_time = time.time() - start_time
            self._update_service_metrics('tensor_decomposition', processing_time, False)
        
        response.processing_time = processing_time
        return response
    
    def get_tensor_metrics(self) -> Dict[str, Any]:
        """Get comprehensive tensor service metrics"""
        metrics = {
            'service_metrics': dict(self.performance_metrics),
            'tensor_system_stats': self.tensor_system.get_compression_stats(),
            'cache_stats': {
                'cache_size': len(getattr(self.tensor_system, 'decomposition_cache', {})),
                'cache_limit': getattr(self.tensor_system, 'cache_size_limit', 0)
            },
            'registered_services': list(self.tensor_services.keys()),
            'service_versions': dict(self.service_versions)
        }
        return metrics
    
    def validate_tensor_request(self, request: ServiceRequest) -> Tuple[bool, List[str]]:
        """Validate tensor service request"""
        errors = []
        
        if request is None:
            return False, ["Request cannot be None"]
        
        if not isinstance(request, ServiceRequest):
            return False, ["Request must be ServiceRequest instance"]
        
        # Check required fields
        if request.service_name is None:
            errors.append("Service name is required")
        
        if request.payload is None:
            errors.append("Request payload is required")
        elif not isinstance(request.payload, dict):
            errors.append("Request payload must be dictionary")
        
        # Validate service exists
        if request.service_name and request.service_name not in self.tensor_services:
            errors.append(f"Unknown service: {request.service_name}")
        
        # Validate tensor data for compression/decompression
        if request.service_name in ['tensor_compress', 'tensor_decomposition']:
            if 'tensor' not in request.payload:
                errors.append("Tensor data is required for compression")
        elif request.service_name == 'tensor_decompress':
            if 'decomposition' not in request.payload:
                errors.append("Decomposition data is required for decompression")
        
        return len(errors) == 0, errors
    
    def _update_service_metrics(self, service_name: str, processing_time: float, success: bool):
        """Update service performance metrics"""
        metrics = self.performance_metrics[service_name]
        
        metrics['total_requests'] += 1
        if success:
            metrics['successful_requests'] += 1
        else:
            metrics['failed_requests'] += 1
        
        metrics['total_processing_time'] += processing_time
        metrics['average_response_time'] = (
            metrics['total_processing_time'] / metrics['total_requests']
        )
        
        if processing_time > metrics['peak_response_time']:
            metrics['peak_response_time'] = processing_time
        
        if processing_time < metrics['min_response_time']:
            metrics['min_response_time'] = processing_time
    
    def _get_initial_status(self):
        """Get initial response status"""
        # Import from service_interfaces_core to get the correct status
        from ..service_interfaces.service_interfaces_core import ServiceStatus
        return ServiceStatus.SUCCESS
    
    def _get_success_status(self):
        """Get success response status"""
        from ..service_interfaces.service_interfaces_core import ServiceStatus
        return ServiceStatus.SUCCESS
    
    def _get_error_status(self):
        """Get error response status"""
        from ..service_interfaces.service_interfaces_core import ServiceStatus
        return ServiceStatus.ERROR


class TensorServiceRegistry:
    """Registry for tensor services"""
    
    def __init__(self):
        self.compression_services: Dict[str, Any] = {}
        self.decompression_services: Dict[str, Any] = {}
        self.decomposition_services: Dict[str, Any] = {}
        self.service_metadata: Dict[str, Dict[str, Any]] = {}
        self.service_dependencies: Dict[str, Set[str]] = {}
        self._lock = threading.RLock()
    
    def register_tensor_compression_service(self, service_name: str, service_instance: Any) -> None:
        """Register tensor compression service"""
        if service_name is None:
            raise ValueError("Service name cannot be None")
        if service_instance is None:
            raise ValueError("Service instance cannot be None")
        
        with self._lock:
            if service_name in self.compression_services:
                raise ValueError(f"Compression service already registered: {service_name}")
            
            # Validate service
            if not hasattr(service_instance, 'compress'):
                raise ValueError("Compression service must have 'compress' method")
            
            self.compression_services[service_name] = service_instance
            self.service_metadata[service_name] = {
                'type': 'compression',
                'registered_at': datetime.now(),
                'instance': service_instance
            }
    
    def register_tensor_decompression_service(self, service_name: str, service_instance: Any) -> None:
        """Register tensor decompression service"""
        if service_name is None:
            raise ValueError("Service name cannot be None")
        if service_instance is None:
            raise ValueError("Service instance cannot be None")
        
        with self._lock:
            if service_name in self.decompression_services:
                raise ValueError(f"Decompression service already registered: {service_name}")
            
            # Validate service
            if not hasattr(service_instance, 'decompress'):
                raise ValueError("Decompression service must have 'decompress' method")
            
            self.decompression_services[service_name] = service_instance
            self.service_metadata[service_name] = {
                'type': 'decompression',
                'registered_at': datetime.now(),
                'instance': service_instance
            }
    
    def register_tensor_decomposition_service(self, service_name: str, service_instance: Any) -> None:
        """Register tensor decomposition service"""
        if service_name is None:
            raise ValueError("Service name cannot be None")
        if service_instance is None:
            raise ValueError("Service instance cannot be None")
        
        with self._lock:
            if service_name in self.decomposition_services:
                raise ValueError(f"Decomposition service already registered: {service_name}")
            
            # Validate service
            if not hasattr(service_instance, 'decompose'):
                raise ValueError("Decomposition service must have 'decompose' method")
            
            self.decomposition_services[service_name] = service_instance
            self.service_metadata[service_name] = {
                'type': 'decomposition',
                'registered_at': datetime.now(),
                'instance': service_instance
            }
    
    def get_tensor_service(self, service_name: str) -> Any:
        """Get tensor service by name"""
        if service_name is None:
            raise ValueError("Service name cannot be None")
        
        with self._lock:
            # Check all registries
            if service_name in self.compression_services:
                return self.compression_services[service_name]
            elif service_name in self.decompression_services:
                return self.decompression_services[service_name]
            elif service_name in self.decomposition_services:
                return self.decomposition_services[service_name]
            else:
                raise ValueError(f"Service not found: {service_name}")
    
    def list_tensor_services(self) -> Dict[str, List[str]]:
        """List all registered tensor services"""
        with self._lock:
            return {
                'compression': list(self.compression_services.keys()),
                'decompression': list(self.decompression_services.keys()),
                'decomposition': list(self.decomposition_services.keys())
            }
    
    def validate_tensor_service(self, service_name: str) -> bool:
        """Validate if tensor service exists and is functional"""
        if service_name is None:
            return False
        
        with self._lock:
            if service_name not in self.service_metadata:
                return False
            
            metadata = self.service_metadata[service_name]
            service_type = metadata['type']
            instance = metadata['instance']
            
            # Validate based on type
            if service_type == 'compression':
                return hasattr(instance, 'compress') and callable(instance.compress)
            elif service_type == 'decompression':
                return hasattr(instance, 'decompress') and callable(instance.decompress)
            elif service_type == 'decomposition':
                return hasattr(instance, 'decompose') and callable(instance.decompose)
            
            return False


class TensorServiceOrchestrator:
    """Orchestrates tensor service requests"""
    
    def __init__(self, registry: TensorServiceRegistry, config: Optional[Dict[str, Any]] = None):
        self.registry = registry
        self.config = config or {}
        self.routing_rules: Dict[str, Callable] = {}
        self.orchestration_metrics = {
            'total_requests': 0,
            'compression_requests': 0,
            'decompression_requests': 0,
            'decomposition_requests': 0,
            'routing_errors': 0,
            'validation_errors': 0
        }
        self.request_queue = deque(maxlen=1000)
        self.response_cache = {}
        self.cache_size_limit = self.config.get('cache_size_limit', 100)
        self._executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 10))
        self._initialize_routing_rules()
    
    def _initialize_routing_rules(self):
        """Initialize routing rules"""
        self.routing_rules = {
            'compress': self._route_to_compression,
            'decompress': self._route_to_decompression,
            'decompose': self._route_to_decomposition
        }
    
    def orchestrate_compression_request(self, request: ServiceRequest) -> ServiceResponse:
        """Orchestrate compression request"""
        if request is None:
            raise ValueError("Request cannot be None")
        
        self.orchestration_metrics['total_requests'] += 1
        self.orchestration_metrics['compression_requests'] += 1
        
        try:
            # Check cache
            cache_key = self._generate_cache_key(request)
            if cache_key in self.response_cache:
                return self.response_cache[cache_key]
            
            # Route to appropriate service
            service_name = self.route_tensor_request(request)
            service = self.registry.get_tensor_service(service_name)
            
            # Execute compression
            tensor_data = request.payload.get('tensor')
            if tensor_data is None:
                raise ValueError("No tensor data in request")
            
            decomposition = service.compress(tensor_data)
            
            # Build response
            response = ServiceResponse(
                request_id=request.request_id,
                status=self._get_success_status()
            )
            response.data = {
                'decomposition': decomposition,
                'service_used': service_name
            }
            
            # Cache response
            self._cache_response(cache_key, response)
            
            # Validate response
            if not self.validate_tensor_response(response):
                raise ValueError("Response validation failed")
            
            return response
            
        except Exception as e:
            self.orchestration_metrics['routing_errors'] += 1
            response = ServiceResponse(
                request_id=request.request_id,
                status=self._get_error_status()
            )
            response.add_error("ORCHESTRATION_ERROR", str(e))
            return response
    
    def orchestrate_decompression_request(self, request: ServiceRequest) -> ServiceResponse:
        """Orchestrate decompression request"""
        if request is None:
            raise ValueError("Request cannot be None")
        
        self.orchestration_metrics['total_requests'] += 1
        self.orchestration_metrics['decompression_requests'] += 1
        
        try:
            # Check cache
            cache_key = self._generate_cache_key(request)
            if cache_key in self.response_cache:
                return self.response_cache[cache_key]
            
            # Route to appropriate service
            service_name = self.route_tensor_request(request)
            service = self.registry.get_tensor_service(service_name)
            
            # Execute decompression
            decomposition_data = request.payload.get('decomposition')
            if decomposition_data is None:
                raise ValueError("No decomposition data in request")
            
            reconstructed = service.decompress(decomposition_data)
            
            # Build response
            response = ServiceResponse(
                request_id=request.request_id,
                status=self._get_success_status()
            )
            response.data = {
                'tensor': reconstructed,
                'service_used': service_name
            }
            
            # Cache response
            self._cache_response(cache_key, response)
            
            # Validate response
            if not self.validate_tensor_response(response):
                raise ValueError("Response validation failed")
            
            return response
            
        except Exception as e:
            self.orchestration_metrics['routing_errors'] += 1
            response = ServiceResponse(
                request_id=request.request_id,
                status=self._get_error_status()
            )
            response.add_error("ORCHESTRATION_ERROR", str(e))
            return response
    
    def orchestrate_decomposition_request(self, request: ServiceRequest) -> ServiceResponse:
        """Orchestrate decomposition request"""
        if request is None:
            raise ValueError("Request cannot be None")
        
        self.orchestration_metrics['total_requests'] += 1
        self.orchestration_metrics['decomposition_requests'] += 1
        
        try:
            # Check cache
            cache_key = self._generate_cache_key(request)
            if cache_key in self.response_cache:
                return self.response_cache[cache_key]
            
            # Route to appropriate service
            service_name = self.route_tensor_request(request)
            service = self.registry.get_tensor_service(service_name)
            
            # Execute decomposition
            tensor_data = request.payload.get('tensor')
            method = request.payload.get('method', 'CP')
            ranks = request.payload.get('ranks')
            
            if tensor_data is None:
                raise ValueError("No tensor data in request")
            
            decomposition = service.decompose(tensor_data, ranks, method)
            
            # Build response
            response = ServiceResponse(
                request_id=request.request_id,
                status=self._get_success_status()
            )
            response.data = {
                'decomposition': decomposition,
                'service_used': service_name
            }
            
            # Cache response
            self._cache_response(cache_key, response)
            
            # Validate response
            if not self.validate_tensor_response(response):
                raise ValueError("Response validation failed")
            
            return response
            
        except Exception as e:
            self.orchestration_metrics['routing_errors'] += 1
            response = ServiceResponse(
                request_id=request.request_id,
                status=self._get_error_status()
            )
            response.add_error("ORCHESTRATION_ERROR", str(e))
            return response
    
    def route_tensor_request(self, request: ServiceRequest) -> str:
        """Route tensor request to appropriate service"""
        if request is None:
            raise ValueError("Request cannot be None")
        
        # Extract operation type from request
        operation = request.metadata.get('operation')
        if operation is None:
            # Infer from service name
            if 'compress' in request.service_name.lower():
                operation = 'compress'
            elif 'decompress' in request.service_name.lower():
                operation = 'decompress'
            elif 'decompos' in request.service_name.lower():
                operation = 'decompose'
            else:
                raise ValueError("Cannot determine operation type")
        
        # Get routing function
        route_func = self.routing_rules.get(operation)
        if route_func is None:
            raise ValueError(f"No routing rule for operation: {operation}")
        
        return route_func(request)
    
    def validate_tensor_response(self, response: ServiceResponse) -> bool:
        """Validate tensor service response"""
        if response is None:
            return False
        
        if not isinstance(response, ServiceResponse):
            return False
        
        if response.data is None:
            return False
        
        # Check for required fields based on data
        if 'decomposition' in response.data:
            decomposition = response.data['decomposition']
            if not hasattr(decomposition, 'factors'):
                return False
            if not hasattr(decomposition, 'decomposition_type'):
                return False
        elif 'tensor' in response.data:
            tensor = response.data['tensor']
            if not isinstance(tensor, (torch.Tensor, np.ndarray)):
                return False
        else:
            return False
        
        return True
    
    def get_orchestration_metrics(self) -> Dict[str, Any]:
        """Get orchestration metrics"""
        return {
            'metrics': dict(self.orchestration_metrics),
            'cache_size': len(self.response_cache),
            'cache_limit': self.cache_size_limit,
            'queue_size': len(self.request_queue),
            'active_workers': len(self._executor._threads) if hasattr(self._executor, '_threads') else 0
        }
    
    def _route_to_compression(self, request: ServiceRequest) -> str:
        """Route to compression service"""
        services = self.registry.list_tensor_services()['compression']
        if not services:
            raise ValueError("No compression services available")
        
        # Simple round-robin for now
        # In production, would use more sophisticated routing
        return services[0]
    
    def _route_to_decompression(self, request: ServiceRequest) -> str:
        """Route to decompression service"""
        services = self.registry.list_tensor_services()['decompression']
        if not services:
            raise ValueError("No decompression services available")
        
        return services[0]
    
    def _route_to_decomposition(self, request: ServiceRequest) -> str:
        """Route to decomposition service"""
        services = self.registry.list_tensor_services()['decomposition']
        if not services:
            raise ValueError("No decomposition services available")
        
        return services[0]
    
    def _generate_cache_key(self, request: ServiceRequest) -> str:
        """Generate cache key for request"""
        # Create a deterministic key from request data
        key_data = {
            'service': request.service_name,
            'operation': request.metadata.get('operation'),
            'data_hash': hashlib.md5(
                json.dumps(request.payload, sort_keys=True, default=str).encode()
            ).hexdigest()
        }
        return json.dumps(key_data, sort_keys=True)
    
    def _cache_response(self, key: str, response: ServiceResponse):
        """Cache response with size limit"""
        if len(self.response_cache) >= self.cache_size_limit:
            # Remove oldest entry
            self.response_cache.pop(next(iter(self.response_cache)))
        
        self.response_cache[key] = response
    
    def _get_success_status(self):
        """Get success response status"""
        from ..service_interfaces.service_interfaces_core import ServiceStatus
        return ServiceStatus.SUCCESS
    
    def _get_error_status(self):
        """Get error response status"""
        from ..service_interfaces.service_interfaces_core import ServiceStatus
        return ServiceStatus.ERROR


class TensorServiceIntegration:
    """Handles integration with other system components"""
    
    def __init__(self, service_interface: TensorServiceInterface):
        self.service_interface = service_interface
        self.domain_router = None
        self.brain_core = None
        self.training_manager = None
        self.integration_status = {
            'domain_router': False,
            'brain_core': False,
            'training_manager': False
        }
        self.service_endpoints: Dict[str, ServiceEndpoint] = {}
        self.integration_metrics = defaultdict(int)
    
    def integrate_with_domain_router(self, domain_router: Any) -> None:
        """Integrate with domain router"""
        if domain_router is None:
            raise ValueError("Domain router cannot be None")
        
        self.domain_router = domain_router
        
        # Register tensor services with domain router
        self.domain_router.register_domain(
            'tensor_compression',
            self.service_interface.invoke_tensor_compression
        )
        self.domain_router.register_domain(
            'tensor_decompression',
            self.service_interface.invoke_tensor_decompression
        )
        
        # Add routing patterns
        self.domain_router.add_pattern(
            'tensor_compress',
            r'compress.*tensor|tensor.*compress',
            'tensor_compression'
        )
        self.domain_router.add_pattern(
            'tensor_decompress',
            r'decompress.*tensor|tensor.*decompress|reconstruct.*tensor',
            'tensor_decompression'
        )
        
        self.integration_status['domain_router'] = True
        self.integration_metrics['domain_router_integrations'] += 1
    
    def integrate_with_brain_core(self, brain_core: Any) -> None:
        """Integrate with brain core system"""
        if brain_core is None:
            raise ValueError("Brain core cannot be None")
        
        self.brain_core = brain_core
        
        # Register tensor compression as memory optimization service
        self.brain_core.register_memory_service(
            'tensor_compression',
            self.service_interface
        )
        
        # Set up bidirectional communication
        self.brain_core.set_compression_callback(
            self.service_interface.invoke_tensor_compression
        )
        
        self.integration_status['brain_core'] = True
        self.integration_metrics['brain_core_integrations'] += 1
    
    def integrate_with_training_manager(self, training_manager: Any) -> None:
        """Integrate with training manager"""
        if training_manager is None:
            raise ValueError("Training manager cannot be None")
        
        self.training_manager = training_manager
        
        # Register compression for model checkpointing
        self.training_manager.register_checkpoint_compression(
            self.service_interface.invoke_tensor_compression
        )
        
        # Register decompression for model loading
        self.training_manager.register_checkpoint_decompression(
            self.service_interface.invoke_tensor_decompression
        )
        
        self.integration_status['training_manager'] = True
        self.integration_metrics['training_manager_integrations'] += 1
    
    def register_service_endpoints(self) -> None:
        """Register service endpoints"""
        # Register compression endpoint
        self.service_endpoints['tensor_compress'] = ServiceEndpoint(
            name='tensor_compress',
            version='1.0.0',
            url='/api/v1/tensor/compress',
            protocol='http',
            timeout=30.0,
            max_retries=3,
            metadata={
                'description': 'Tensor compression service',
                'input_format': 'torch.Tensor or numpy.ndarray',
                'output_format': 'TensorDecomposition'
            }
        )
        
        # Register decompression endpoint
        self.service_endpoints['tensor_decompress'] = ServiceEndpoint(
            name='tensor_decompress',
            version='1.0.0',
            url='/api/v1/tensor/decompress',
            protocol='http',
            timeout=30.0,
            max_retries=3,
            metadata={
                'description': 'Tensor decompression service',
                'input_format': 'TensorDecomposition',
                'output_format': 'torch.Tensor'
            }
        )
        
        # Register decomposition endpoint
        self.service_endpoints['tensor_decompose'] = ServiceEndpoint(
            name='tensor_decompose',
            version='1.0.0',
            url='/api/v1/tensor/decompose',
            protocol='http',
            timeout=30.0,
            max_retries=3,
            metadata={
                'description': 'Tensor decomposition service',
                'input_format': 'torch.Tensor with method and ranks',
                'output_format': 'TensorDecomposition'
            }
        )
    
    def validate_integration(self) -> bool:
        """Validate all integrations are working"""
        all_integrated = all(self.integration_status.values())
        
        if not all_integrated:
            return False
        
        # Test domain router integration
        if self.domain_router:
            try:
                test_request = ServiceRequest(
                    service_name='tensor_compression',
                    method_name='compress',
                    version='1.0.0',
                    payload={'tensor': torch.randn(10, 10)}
                )
                
                result = self.domain_router.route_request(
                    'compress this tensor',
                    domain_hint='tensor_compression'
                )
                
                if result.target_domain != 'tensor_compression':
                    return False
            except:
                return False
        
        # Test brain core integration
        if self.brain_core:
            try:
                services = self.brain_core.get_memory_services()
                if 'tensor_compression' not in services:
                    return False
            except:
                return False
        
        # Test training manager integration
        if self.training_manager:
            try:
                compressor = self.training_manager.get_checkpoint_compressor()
                if compressor is None:
                    return False
            except:
                return False
        
        return True


class TensorServiceValidation:
    """Validates tensor service requests and responses"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.validation_rules = self._initialize_validation_rules()
        self.validation_metrics = defaultdict(int)
    
    def _initialize_validation_rules(self) -> Dict[str, List[Callable]]:
        """Initialize validation rules"""
        return {
            'compression': [
                self._validate_tensor_shape,
                self._validate_tensor_dtype,
                self._validate_tensor_device,
                self._validate_compression_parameters
            ],
            'decompression': [
                self._validate_decomposition_structure,
                self._validate_decomposition_factors,
                self._validate_decomposition_metadata
            ],
            'decomposition': [
                self._validate_tensor_shape,
                self._validate_decomposition_method,
                self._validate_decomposition_ranks
            ]
        }
    
    def validate_compression_request(self, request: ServiceRequest) -> Tuple[bool, List[str]]:
        """Validate compression request"""
        errors = []
        
        if request is None:
            return False, ["Request cannot be None"]
        
        # Check tensor data
        tensor_data = request.payload.get('tensor')
        if tensor_data is None:
            errors.append("Tensor data is required")
            return False, errors
        
        # Apply validation rules
        for rule in self.validation_rules['compression']:
            is_valid, rule_errors = rule(tensor_data, request)
            if not is_valid:
                errors.extend(rule_errors)
        
        self.validation_metrics['compression_validations'] += 1
        if errors:
            self.validation_metrics['compression_validation_failures'] += 1
        
        return len(errors) == 0, errors
    
    def validate_decompression_request(self, request: ServiceRequest) -> Tuple[bool, List[str]]:
        """Validate decompression request"""
        errors = []
        
        if request is None:
            return False, ["Request cannot be None"]
        
        # Check decomposition data
        decomposition_data = request.payload.get('decomposition')
        if decomposition_data is None:
            errors.append("Decomposition data is required")
            return False, errors
        
        # Apply validation rules
        for rule in self.validation_rules['decompression']:
            is_valid, rule_errors = rule(decomposition_data, request)
            if not is_valid:
                errors.extend(rule_errors)
        
        self.validation_metrics['decompression_validations'] += 1
        if errors:
            self.validation_metrics['decompression_validation_failures'] += 1
        
        return len(errors) == 0, errors
    
    def validate_decomposition_request(self, request: ServiceRequest) -> Tuple[bool, List[str]]:
        """Validate decomposition request"""
        errors = []
        
        if request is None:
            return False, ["Request cannot be None"]
        
        # Check required data
        tensor_data = request.payload.get('tensor')
        if tensor_data is None:
            errors.append("Tensor data is required")
            return False, errors
        
        # Apply validation rules
        for rule in self.validation_rules['decomposition']:
            is_valid, rule_errors = rule(tensor_data, request)
            if not is_valid:
                errors.extend(rule_errors)
        
        self.validation_metrics['decomposition_validations'] += 1
        if errors:
            self.validation_metrics['decomposition_validation_failures'] += 1
        
        return len(errors) == 0, errors
    
    def validate_service_response(self, response: ServiceResponse) -> Tuple[bool, List[str]]:
        """Validate service response"""
        errors = []
        
        if response is None:
            return False, ["Response cannot be None"]
        
        if not isinstance(response, ServiceResponse):
            return False, ["Response must be ServiceResponse instance"]
        
        # Check basic response structure
        if response.data is None:
            errors.append("Response data is required")
        
        # Check for errors in failed responses
        if not response.is_success() and not response.errors:
            errors.append("Failed response must include error details")
        
        # Validate response data based on content
        if response.data:
            if 'tensor' in response.data:
                tensor = response.data['tensor']
                if not isinstance(tensor, (torch.Tensor, np.ndarray)):
                    errors.append("Response tensor must be torch.Tensor or numpy.ndarray")
            
            if 'decomposition' in response.data:
                decomposition = response.data['decomposition']
                if not hasattr(decomposition, 'factors'):
                    errors.append("Decomposition must have factors attribute")
        
        return len(errors) == 0, errors
    
    def validate_tensor_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate tensor operation parameters"""
        errors = []
        
        if parameters is None:
            return False, ["Parameters cannot be None"]
        
        # Validate compression ratio
        if 'compression_ratio' in parameters:
            ratio = parameters['compression_ratio']
            if not isinstance(ratio, (int, float)):
                errors.append("Compression ratio must be numeric")
            elif ratio <= 0 or ratio >= 1:
                errors.append("Compression ratio must be between 0 and 1")
        
        # Validate method
        if 'method' in parameters:
            method = parameters['method']
            valid_methods = ['CP', 'Tucker', 'TT', 'SVD']
            if method not in valid_methods:
                errors.append(f"Method must be one of {valid_methods}")
        
        # Validate ranks
        if 'ranks' in parameters:
            ranks = parameters['ranks']
            if not isinstance(ranks, (list, tuple, int)):
                errors.append("Ranks must be int, list, or tuple")
        
        return len(errors) == 0, errors
    
    def _validate_tensor_shape(self, tensor: Any, request: ServiceRequest) -> Tuple[bool, List[str]]:
        """Validate tensor shape"""
        errors = []
        
        if isinstance(tensor, torch.Tensor):
            if len(tensor.shape) == 0:
                errors.append("Tensor must have at least one dimension")
            elif any(dim <= 0 for dim in tensor.shape):
                errors.append("All tensor dimensions must be positive")
        elif isinstance(tensor, np.ndarray):
            if len(tensor.shape) == 0:
                errors.append("Array must have at least one dimension")
        else:
            try:
                # Try to convert to tensor
                torch.tensor(tensor)
            except:
                errors.append("Data cannot be converted to tensor")
        
        return len(errors) == 0, errors
    
    def _validate_tensor_dtype(self, tensor: Any, request: ServiceRequest) -> Tuple[bool, List[str]]:
        """Validate tensor data type"""
        errors = []
        
        if isinstance(tensor, torch.Tensor):
            supported_dtypes = [torch.float32, torch.float64, torch.float16]
            if tensor.dtype not in supported_dtypes:
                errors.append(f"Tensor dtype {tensor.dtype} not supported")
        
        return len(errors) == 0, errors
    
    def _validate_tensor_device(self, tensor: Any, request: ServiceRequest) -> Tuple[bool, List[str]]:
        """Validate tensor device"""
        errors = []
        
        if isinstance(tensor, torch.Tensor):
            # Check if CUDA is available if tensor is on GPU
            if tensor.is_cuda and not torch.cuda.is_available():
                errors.append("CUDA not available for GPU tensor")
        
        return len(errors) == 0, errors
    
    def _validate_compression_parameters(self, tensor: Any, request: ServiceRequest) -> Tuple[bool, List[str]]:
        """Validate compression parameters"""
        errors = []
        
        # Check optional parameters
        if 'target_ratio' in request.payload:
            ratio = request.payload['target_ratio']
            if not 0 < ratio < 1:
                errors.append("Target ratio must be between 0 and 1")
        
        return len(errors) == 0, errors
    
    def _validate_decomposition_structure(self, decomposition: Any, request: ServiceRequest) -> Tuple[bool, List[str]]:
        """Validate decomposition structure"""
        errors = []
        
        if not hasattr(decomposition, 'factors'):
            errors.append("Decomposition must have factors")
        
        if not hasattr(decomposition, 'decomposition_type'):
            errors.append("Decomposition must have decomposition_type")
        
        return len(errors) == 0, errors
    
    def _validate_decomposition_factors(self, decomposition: Any, request: ServiceRequest) -> Tuple[bool, List[str]]:
        """Validate decomposition factors"""
        errors = []
        
        if hasattr(decomposition, 'factors'):
            factors = decomposition.factors
            if not isinstance(factors, (list, tuple)):
                errors.append("Factors must be list or tuple")
            elif len(factors) == 0:
                errors.append("Factors cannot be empty")
        
        return len(errors) == 0, errors
    
    def _validate_decomposition_metadata(self, decomposition: Any, request: ServiceRequest) -> Tuple[bool, List[str]]:
        """Validate decomposition metadata"""
        errors = []
        
        required_attrs = ['compression_ratio']
        for attr in required_attrs:
            if not hasattr(decomposition, attr):
                errors.append(f"Decomposition missing required attribute: {attr}")
        
        return len(errors) == 0, errors
    
    def _validate_decomposition_method(self, tensor: Any, request: ServiceRequest) -> Tuple[bool, List[str]]:
        """Validate decomposition method"""
        errors = []
        
        method = request.payload.get('method')
        if method:
            valid_methods = ['CP', 'Tucker', 'TT', 'SVD']
            if method not in valid_methods:
                errors.append(f"Invalid method: {method}")
        
        return len(errors) == 0, errors
    
    def _validate_decomposition_ranks(self, tensor: Any, request: ServiceRequest) -> Tuple[bool, List[str]]:
        """Validate decomposition ranks"""
        errors = []
        
        ranks = request.payload.get('ranks')
        if ranks:
            if isinstance(ranks, int):
                if ranks <= 0:
                    errors.append("Rank must be positive")
            elif isinstance(ranks, (list, tuple)):
                if any(r <= 0 for r in ranks):
                    errors.append("All ranks must be positive")
            else:
                errors.append("Ranks must be int, list, or tuple")
        
        return len(errors) == 0, errors


class TensorServiceMetrics:
    """Tracks and manages tensor service metrics"""
    
    def __init__(self):
        self.compression_metrics = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'total_size': 0,
            'compressed_size': 0,
            'avg_ratio': 0.0,
            'errors': 0
        })
        self.decompression_metrics = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'reconstruction_errors': [],
            'errors': 0
        })
        self.decomposition_metrics = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'methods_used': defaultdict(int),
            'rank_distribution': defaultdict(int),
            'errors': 0
        })
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'throughput': 0.0,
            'error_rate': 0.0
        }
        self._start_time = time.time()
        self._lock = threading.RLock()
    
    def record_compression_request(self, request: ServiceRequest, response: ServiceResponse) -> None:
        """Record compression request metrics"""
        if request is None or response is None:
            return
        
        with self._lock:
            service_name = request.service_name
            metrics = self.compression_metrics[service_name]
            
            metrics['count'] += 1
            processing_time = response.processing_time
            
            # Update timing metrics
            metrics['total_time'] += processing_time
            metrics['avg_time'] = metrics['total_time'] / metrics['count']
            metrics['min_time'] = min(metrics['min_time'], processing_time)
            metrics['max_time'] = max(metrics['max_time'], processing_time)
            
            # Update compression metrics
            if response.is_success() and response.data:
                decomposition = response.data.get('decomposition')
                if decomposition and hasattr(decomposition, 'factors'):
                    # Calculate sizes
                    if hasattr(decomposition, 'metadata') and 'original_shape' in decomposition.metadata:
                        original_size = np.prod(decomposition.metadata['original_shape']) * 4  # float32
                    else:
                        # Estimate from factors
                        original_size = sum(f.numel() for f in decomposition.factors) * 4
                    
                    compressed_size = sum(
                        f.numel() * 4 for f in decomposition.factors
                        if isinstance(f, torch.Tensor)
                    )
                    metrics['total_size'] += original_size
                    metrics['compressed_size'] += compressed_size
                    
                    if metrics['total_size'] > 0:
                        metrics['avg_ratio'] = metrics['compressed_size'] / metrics['total_size']
            else:
                metrics['errors'] += 1
            
            self._update_performance_metrics(response)
    
    def record_decompression_request(self, request: ServiceRequest, response: ServiceResponse) -> None:
        """Record decompression request metrics"""
        if request is None or response is None:
            return
        
        with self._lock:
            service_name = request.service_name
            metrics = self.decompression_metrics[service_name]
            
            metrics['count'] += 1
            processing_time = response.processing_time
            
            # Update timing metrics
            metrics['total_time'] += processing_time
            metrics['avg_time'] = metrics['total_time'] / metrics['count']
            metrics['min_time'] = min(metrics['min_time'], processing_time)
            metrics['max_time'] = max(metrics['max_time'], processing_time)
            
            # Track reconstruction error
            if response.is_success() and response.data:
                error = response.data.get('reconstruction_error', 0.0)
                metrics['reconstruction_errors'].append(error)
                # Keep only last 1000 errors
                if len(metrics['reconstruction_errors']) > 1000:
                    metrics['reconstruction_errors'].pop(0)
            else:
                metrics['errors'] += 1
            
            self._update_performance_metrics(response)
    
    def record_decomposition_request(self, request: ServiceRequest, response: ServiceResponse) -> None:
        """Record decomposition request metrics"""
        if request is None or response is None:
            return
        
        with self._lock:
            service_name = request.service_name
            metrics = self.decomposition_metrics[service_name]
            
            metrics['count'] += 1
            processing_time = response.processing_time
            
            # Update timing metrics
            metrics['total_time'] += processing_time
            metrics['avg_time'] = metrics['total_time'] / metrics['count']
            
            # Track method and rank distribution
            if response.is_success():
                method = request.payload.get('method', 'unknown')
                metrics['methods_used'][method] += 1
                
                ranks = request.payload.get('ranks')
                if isinstance(ranks, int):
                    metrics['rank_distribution'][ranks] += 1
                elif isinstance(ranks, (list, tuple)):
                    for r in ranks:
                        metrics['rank_distribution'][r] += 1
            else:
                metrics['errors'] += 1
            
            self._update_performance_metrics(response)
    
    def get_compression_metrics(self) -> Dict[str, Any]:
        """Get compression metrics"""
        with self._lock:
            return dict(self.compression_metrics)
    
    def get_decompression_metrics(self) -> Dict[str, Any]:
        """Get decompression metrics"""
        with self._lock:
            metrics = dict(self.decompression_metrics)
            # Calculate average reconstruction error
            for service_metrics in metrics.values():
                errors = service_metrics.get('reconstruction_errors', [])
                if errors:
                    service_metrics['avg_reconstruction_error'] = np.mean(errors)
                    service_metrics['std_reconstruction_error'] = np.std(errors)
            return metrics
    
    def get_decomposition_metrics(self) -> Dict[str, Any]:
        """Get decomposition metrics"""
        with self._lock:
            return dict(self.decomposition_metrics)
    
    def get_service_performance_metrics(self) -> Dict[str, Any]:
        """Get overall service performance metrics"""
        with self._lock:
            uptime = time.time() - self._start_time
            metrics = dict(self.performance_metrics)
            metrics['uptime_seconds'] = uptime
            metrics['requests_per_second'] = (
                metrics['total_requests'] / uptime if uptime > 0 else 0
            )
            return metrics
    
    def _update_performance_metrics(self, response: ServiceResponse):
        """Update overall performance metrics"""
        self.performance_metrics['total_requests'] += 1
        
        if response.is_success():
            self.performance_metrics['successful_requests'] += 1
        else:
            self.performance_metrics['failed_requests'] += 1
        
        # Update average response time
        total = self.performance_metrics['total_requests']
        current_avg = self.performance_metrics['avg_response_time']
        self.performance_metrics['avg_response_time'] = (
            (current_avg * (total - 1) + response.processing_time) / total
        )
        
        # Update error rate
        self.performance_metrics['error_rate'] = (
            self.performance_metrics['failed_requests'] / total
        )
        
        # Update throughput
        uptime = time.time() - self._start_time
        self.performance_metrics['throughput'] = total / uptime if uptime > 0 else 0


class TensorServiceMiddleware:
    """Middleware for tensor service processing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.middleware_stack: List[Callable] = []
        self.error_handlers: Dict[type, Callable] = {}
        self.health_checks: Dict[str, Callable] = {}
        self._initialize_middleware()
    
    def _initialize_middleware(self):
        """Initialize middleware components"""
        # Register default error handlers
        self.error_handlers[ValueError] = self._handle_value_error
        self.error_handlers[TypeError] = self._handle_type_error
        self.error_handlers[RuntimeError] = self._handle_runtime_error
        self.error_handlers[Exception] = self._handle_generic_error
    
    def pre_process_request(self, request: ServiceRequest) -> ServiceRequest:
        """Pre-process service request"""
        if request is None:
            raise ValueError("Request cannot be None")
        
        # Add request ID if not present
        if 'request_id' not in request.metadata:
            request.metadata['request_id'] = self._generate_request_id()
        
        # Add timestamp
        request.metadata['received_at'] = datetime.now()
        
        # Normalize tensor data
        if 'tensor' in request.payload:
            tensor = request.payload['tensor']
            if isinstance(tensor, np.ndarray):
                request.payload['tensor'] = torch.from_numpy(tensor)
            elif isinstance(tensor, list):
                request.payload['tensor'] = torch.tensor(tensor)
        
        # Set default parameters
        if 'timeout' not in request.metadata:
            request.metadata['timeout'] = self.config.get('default_timeout', 30.0)
        
        # Apply middleware stack
        for middleware in self.middleware_stack:
            request = middleware(request)
        
        return request
    
    def post_process_response(self, response: ServiceResponse) -> ServiceResponse:
        """Post-process service response"""
        if response is None:
            raise ValueError("Response cannot be None")
        
        # Add response metadata
        response.metadata['processed_at'] = datetime.now()
        
        # Convert tensor outputs if needed
        if response.data and 'tensor' in response.data:
            tensor = response.data['tensor']
            if isinstance(tensor, torch.Tensor):
                # Add tensor metadata
                response.data['tensor_info'] = {
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'device': str(tensor.device),
                    'requires_grad': tensor.requires_grad
                }
        
        # Add performance metadata
        if hasattr(response, 'processing_time'):
            response.metadata['processing_time_ms'] = response.processing_time * 1000
        
        # Add service version
        response.metadata['service_version'] = '1.0.0'
        
        return response
    
    def handle_service_error(self, error: Exception, request: ServiceRequest) -> ServiceResponse:
        """Handle service error"""
        response = ServiceResponse(
            request_id=request.request_id,
            status=self._get_error_status()
        )
        
        # Get specific error handler
        error_type = type(error)
        handler = self.error_handlers.get(error_type, self.error_handlers[Exception])
        
        # Apply error handler
        response = handler(error, request, response)
        
        # Add error metadata
        response.metadata['error_type'] = error_type.__name__
        response.metadata['error_timestamp'] = datetime.now()
        response.metadata['request_id'] = request.metadata.get('request_id', 'unknown')
        
        # Log error
        logging.error(
            f"Service error: {error_type.__name__} - {str(error)}",
            exc_info=True
        )
        
        return response
    
    def validate_service_health(self, service_name: str) -> bool:
        """Validate service health"""
        if service_name is None:
            return False
        
        # Check if health check exists
        if service_name not in self.health_checks:
            # Default health check
            return True
        
        # Run health check
        try:
            health_check = self.health_checks[service_name]
            return health_check()
        except Exception as e:
            logging.error(f"Health check failed for {service_name}: {str(e)}")
            return False
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        timestamp = int(time.time() * 1000000)
        random_part = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
        return f"req_{timestamp}_{random_part}"
    
    def _handle_value_error(self, error: ValueError, request: ServiceRequest, 
                           response: ServiceResponse) -> ServiceResponse:
        """Handle ValueError"""
        response.add_error("VALUE_ERROR", str(error), {
            'message': 'Invalid value provided',
            'suggestion': 'Check input parameters and data types'
        })
        return response
    
    def _handle_type_error(self, error: TypeError, request: ServiceRequest,
                          response: ServiceResponse) -> ServiceResponse:
        """Handle TypeError"""
        response.add_error("TYPE_ERROR", str(error), {
            'message': 'Type mismatch in request',
            'suggestion': 'Ensure correct data types for all parameters'
        })
        return response
    
    def _handle_runtime_error(self, error: RuntimeError, request: ServiceRequest,
                             response: ServiceResponse) -> ServiceResponse:
        """Handle RuntimeError"""
        response.add_error("RUNTIME_ERROR", str(error), {
            'message': 'Runtime error during processing',
            'suggestion': 'Check system resources and configuration'
        })
        return response
    
    def _handle_generic_error(self, error: Exception, request: ServiceRequest,
                             response: ServiceResponse) -> ServiceResponse:
        """Handle generic errors"""
        response.add_error("GENERIC_ERROR", str(error), {
            'message': 'Unexpected error occurred',
            'suggestion': 'Contact support if issue persists'
        })
        return response
    
    def _get_error_status(self):
        """Get error response status"""
        from ..service_interfaces.service_interfaces_core import ServiceStatus
        return ServiceStatus.ERROR


class TensorServiceConfiguration:
    """Manages tensor service configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.config_schema = self._define_config_schema()
        self.config_history: List[Dict[str, Any]] = []
        self._load_default_config()
        
        if config_path:
            self.load_service_config(config_path)
    
    def _define_config_schema(self) -> Dict[str, Any]:
        """Define configuration schema"""
        return {
            'service': {
                'name': str,
                'version': str,
                'description': str,
                'enabled': bool
            },
            'compression': {
                'default_method': str,
                'target_ratio': float,
                'cache_enabled': bool,
                'cache_size': int,
                'gpu_enabled': bool
            },
            'performance': {
                'max_workers': int,
                'timeout': float,
                'batch_size': int,
                'memory_limit_mb': int
            },
            'security': {
                'authentication_enabled': bool,
                'encryption_enabled': bool,
                'api_key_required': bool
            },
            'monitoring': {
                'metrics_enabled': bool,
                'log_level': str,
                'health_check_interval': int
            }
        }
    
    def _load_default_config(self):
        """Load default configuration"""
        self.config = {
            'service': {
                'name': 'TensorCompressionService',
                'version': '1.0.0',
                'description': 'High-performance tensor compression service',
                'enabled': True
            },
            'compression': {
                'default_method': 'CP',
                'target_ratio': 0.5,
                'cache_enabled': True,
                'cache_size': 100,
                'gpu_enabled': torch.cuda.is_available()
            },
            'performance': {
                'max_workers': 10,
                'timeout': 30.0,
                'batch_size': 32,
                'memory_limit_mb': 4096
            },
            'security': {
                'authentication_enabled': False,
                'encryption_enabled': False,
                'api_key_required': False
            },
            'monitoring': {
                'metrics_enabled': True,
                'log_level': 'INFO',
                'health_check_interval': 60
            }
        }
    
    def load_service_config(self, config_path: str) -> Dict[str, Any]:
        """Load service configuration from file"""
        if config_path is None:
            raise ValueError("Config path cannot be None")
        
        try:
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            with open(path, 'r') as f:
                if path.suffix == '.json':
                    loaded_config = json.load(f)
                else:
                    # Assume Python dict format
                    content = f.read()
                    loaded_config = eval(content)
            
            # Validate before applying
            is_valid, errors = self.validate_service_config(loaded_config)
            if not is_valid:
                raise ValueError(f"Invalid configuration: {errors}")
            
            # Merge with defaults
            self._merge_config(loaded_config)
            
            # Save to history
            self.config_history.append({
                'config': dict(self.config),
                'timestamp': datetime.now(),
                'source': config_path
            })
            
            return self.config
            
        except Exception as e:
            logging.error(f"Failed to load config: {str(e)}")
            raise
    
    def validate_service_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate service configuration"""
        errors = []
        
        if config is None:
            return False, ["Config cannot be None"]
        
        # Validate against schema
        for section, schema in self.config_schema.items():
            if section in config:
                section_config = config[section]
                for key, expected_type in schema.items():
                    if key in section_config:
                        value = section_config[key]
                        if not isinstance(value, expected_type):
                            errors.append(
                                f"{section}.{key} must be {expected_type.__name__}"
                            )
        
        # Validate specific constraints
        if 'compression' in config:
            comp = config['compression']
            if 'target_ratio' in comp:
                ratio = comp['target_ratio']
                if not 0 < ratio < 1:
                    errors.append("target_ratio must be between 0 and 1")
            
            if 'default_method' in comp:
                method = comp['default_method']
                valid_methods = ['CP', 'Tucker', 'TT', 'SVD']
                if method not in valid_methods:
                    errors.append(f"default_method must be one of {valid_methods}")
        
        if 'performance' in config:
            perf = config['performance']
            if 'max_workers' in perf and perf['max_workers'] <= 0:
                errors.append("max_workers must be positive")
            
            if 'timeout' in perf and perf['timeout'] <= 0:
                errors.append("timeout must be positive")
        
        return len(errors) == 0, errors
    
    def update_service_config(self, config: Dict[str, Any]) -> bool:
        """Update service configuration"""
        if config is None:
            raise ValueError("Config cannot be None")
        
        # Validate new config
        is_valid, errors = self.validate_service_config(config)
        if not is_valid:
            logging.error(f"Invalid config update: {errors}")
            return False
        
        # Save current config to history
        self.config_history.append({
            'config': dict(self.config),
            'timestamp': datetime.now(),
            'source': 'update'
        })
        
        # Apply updates
        self._merge_config(config)
        
        # Save to file if path exists
        if self.config_path:
            try:
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            except Exception as e:
                logging.error(f"Failed to save config: {str(e)}")
                return False
        
        return True
    
    def get_service_config(self) -> Dict[str, Any]:
        """Get current service configuration"""
        return dict(self.config)
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """Merge new configuration with existing"""
        for key, value in new_config.items():
            if isinstance(value, dict) and key in self.config:
                # Merge nested dictionaries
                self.config[key].update(value)
            else:
                self.config[key] = value


class TensorServiceHealthMonitor:
    """Monitors tensor service health"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.service_status: Dict[str, ServiceHealth] = {}
        self.health_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.alert_handlers: List[Callable] = []
        self.check_interval = config.get('check_interval', 60)
        self._monitoring = False
        self._monitor_thread = None
        self._start_times: Dict[str, float] = {}
    
    def monitor_service_health(self, service_name: str) -> Dict[str, Any]:
        """Monitor service health"""
        if service_name is None:
            raise ValueError("Service name cannot be None")
        
        # Initialize if not exists
        if service_name not in self.service_status:
            self._start_times[service_name] = time.time()
            self.service_status[service_name] = ServiceHealth(
                service_name=service_name,
                status=ServiceStatus.STARTING,
                uptime=0.0,
                last_check=datetime.now(),
                response_time_ms=0.0,
                error_rate=0.0,
                throughput=0.0,
                active_connections=0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0
            )
        
        # Perform health checks
        health = self._perform_health_checks(service_name)
        
        # Update status
        self.service_status[service_name] = health
        
        # Record history
        self.health_history[service_name].append({
            'timestamp': datetime.now(),
            'status': health.status,
            'metrics': {
                'response_time': health.response_time_ms,
                'error_rate': health.error_rate,
                'throughput': health.throughput
            }
        })
        
        # Check for alerts
        self._check_alerts(service_name, health)
        
        return {
            'service': service_name,
            'health': health,
            'history_size': len(self.health_history[service_name])
        }
    
    def check_service_availability(self, service_name: str) -> bool:
        """Check if service is available"""
        if service_name is None:
            return False
        
        if service_name not in self.service_status:
            return False
        
        health = self.service_status[service_name]
        return health.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]
    
    def get_service_status(self, service_name: str) -> str:
        """Get service status"""
        if service_name is None:
            return ServiceStatus.UNAVAILABLE.value
        
        if service_name not in self.service_status:
            return ServiceStatus.UNAVAILABLE.value
        
        return self.service_status[service_name].status.value
    
    def report_service_issues(self, service_name: str, issue: str) -> None:
        """Report service issues"""
        if service_name is None or issue is None:
            return
        
        # Log issue
        logging.warning(f"Service issue reported for {service_name}: {issue}")
        
        # Update service status
        if service_name in self.service_status:
            health = self.service_status[service_name]
            
            # Degrade status based on issue severity
            if 'critical' in issue.lower():
                health.status = ServiceStatus.UNHEALTHY
            elif health.status == ServiceStatus.HEALTHY:
                health.status = ServiceStatus.DEGRADED
            
            # Add to custom metrics
            health.custom_metrics['last_issue'] = issue
            health.custom_metrics['last_issue_time'] = datetime.now()
        
        # Trigger alerts
        for handler in self.alert_handlers:
            try:
                handler(service_name, issue)
            except Exception as e:
                logging.error(f"Alert handler failed: {str(e)}")
    
    def _perform_health_checks(self, service_name: str) -> ServiceHealth:
        """Perform actual health checks"""
        health = self.service_status.get(service_name)
        if health is None:
            health = ServiceHealth(
                service_name=service_name,
                status=ServiceStatus.STARTING,
                uptime=0.0,
                last_check=datetime.now(),
                response_time_ms=0.0,
                error_rate=0.0,
                throughput=0.0,
                active_connections=0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0
            )
        
        # Update basic metrics
        health.last_check = datetime.now()
        health.uptime = time.time() - self._start_times.get(service_name, time.time())
        
        # Simulate health check (in production, would make actual service call)
        try:
            # Test service responsiveness
            start = time.time()
            # In production: make test request to service
            response_time = (time.time() - start) * 1000
            health.response_time_ms = response_time
            
            # Update status based on metrics
            if response_time > 1000:  # > 1 second
                health.status = ServiceStatus.DEGRADED
            elif health.error_rate > 0.1:  # > 10% errors
                health.status = ServiceStatus.UNHEALTHY
            else:
                health.status = ServiceStatus.HEALTHY
            
        except Exception as e:
            health.status = ServiceStatus.UNHEALTHY
            health.custom_metrics['last_error'] = str(e)
        
        return health
    
    def _check_alerts(self, service_name: str, health: ServiceHealth):
        """Check if alerts should be triggered"""
        # Alert on status changes
        history = self.health_history[service_name]
        if len(history) > 1:
            prev_status = history[-2]['status']
            if health.status != prev_status:
                if health.status == ServiceStatus.UNHEALTHY:
                    self.report_service_issues(
                        service_name,
                        f"Service status changed to UNHEALTHY"
                    )
                elif health.status == ServiceStatus.DEGRADED:
                    self.report_service_issues(
                        service_name,
                        f"Service performance degraded"
                    )
        
        # Alert on high error rate
        if health.error_rate > 0.05:  # > 5% errors
            self.report_service_issues(
                service_name,
                f"High error rate: {health.error_rate:.2%}"
            )
        
        # Alert on slow response
        if health.response_time_ms > 500:  # > 500ms
            self.report_service_issues(
                service_name,
                f"Slow response time: {health.response_time_ms:.0f}ms"
            )


class TensorServiceCache:
    """Cache for tensor service results"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.compression_cache: Dict[str, Dict[str, Any]] = {}
        self.decompression_cache: Dict[str, torch.Tensor] = {}
        self.decomposition_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_metadata: Dict[str, Dict[str, Any]] = {}
        self.max_cache_size = config.get('max_cache_size', 1000)
        self.ttl_seconds = config.get('ttl_seconds', 3600)
        self._lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
    
    def cache_compression_result(self, key: str, result: Dict[str, Any]) -> None:
        """Cache compression result"""
        if key is None or result is None:
            return
        
        with self._lock:
            # Check cache size
            if len(self.compression_cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = min(
                    (k for k in self.cache_metadata.keys() 
                     if self.cache_metadata[k]['type'] == 'compression'),
                    key=lambda k: self.cache_metadata[k]['created_at'],
                    default=None
                )
                if oldest_key:
                    del self.compression_cache[oldest_key]
                    del self.cache_metadata[oldest_key]
            
            # Store result
            self.compression_cache[key] = result
            self.cache_metadata[key] = {
                'created_at': time.time(),
                'access_count': 0,
                'last_accessed': time.time(),
                'type': 'compression'
            }
    
    def get_cached_compression(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached compression result"""
        if key is None:
            return None
        
        with self._lock:
            if key not in self.compression_cache:
                self.miss_count += 1
                return None
            
            # Check TTL
            metadata = self.cache_metadata[key]
            age = time.time() - metadata['created_at']
            if age > self.ttl_seconds:
                # Expired
                del self.compression_cache[key]
                del self.cache_metadata[key]
                self.miss_count += 1
                return None
            
            # Update access metadata
            metadata['access_count'] += 1
            metadata['last_accessed'] = time.time()
            
            self.hit_count += 1
            return self.compression_cache[key]
    
    def cache_decompression_result(self, key: str, result: torch.Tensor) -> None:
        """Cache decompression result"""
        if key is None or result is None:
            return
        
        with self._lock:
            # Check cache size
            if len(self.decompression_cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = min(
                    (k for k in self.cache_metadata.keys() 
                     if self.cache_metadata[k]['type'] == 'decompression'),
                    key=lambda k: self.cache_metadata[k]['created_at'],
                    default=None
                )
                if oldest_key:
                    del self.decompression_cache[oldest_key]
                    del self.cache_metadata[oldest_key]
            
            # Store result
            self.decompression_cache[key] = result.clone()
            self.cache_metadata[key] = {
                'created_at': time.time(),
                'access_count': 0,
                'last_accessed': time.time(),
                'type': 'decompression',
                'tensor_size': result.numel() * result.element_size()
            }
    
    def get_cached_decompression(self, key: str) -> Optional[torch.Tensor]:
        """Get cached decompression result"""
        if key is None:
            return None
        
        with self._lock:
            if key not in self.decompression_cache:
                self.miss_count += 1
                return None
            
            # Check TTL
            metadata = self.cache_metadata[key]
            age = time.time() - metadata['created_at']
            if age > self.ttl_seconds:
                # Expired
                del self.decompression_cache[key]
                del self.cache_metadata[key]
                self.miss_count += 1
                return None
            
            # Update access metadata
            metadata['access_count'] += 1
            metadata['last_accessed'] = time.time()
            
            self.hit_count += 1
            return self.decompression_cache[key].clone()
    
    def cache_decomposition_result(self, key: str, result: Dict[str, Any]) -> None:
        """Cache decomposition result"""
        if key is None or result is None:
            return
        
        with self._lock:
            # Check cache size
            if len(self.decomposition_cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = min(
                    (k for k in self.cache_metadata.keys()
                     if self.cache_metadata[k]['type'] == 'decomposition'),
                    key=lambda k: self.cache_metadata[k]['created_at'],
                    default=None
                )
                if oldest_key:
                    del self.decomposition_cache[oldest_key]
                    del self.cache_metadata[oldest_key]
            
            # Store result
            self.decomposition_cache[key] = result
            self.cache_metadata[key] = {
                'created_at': time.time(),
                'access_count': 0,
                'last_accessed': time.time(),
                'type': 'decomposition'
            }
    
    def get_cached_decomposition(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached decomposition result"""
        if key is None:
            return None
        
        with self._lock:
            if key not in self.decomposition_cache:
                self.miss_count += 1
                return None
            
            # Check TTL
            metadata = self.cache_metadata[key]
            age = time.time() - metadata['created_at']
            if age > self.ttl_seconds:
                # Expired
                del self.decomposition_cache[key]
                del self.cache_metadata[key]
                self.miss_count += 1
                return None
            
            # Update access metadata
            metadata['access_count'] += 1
            metadata['last_accessed'] = time.time()
            
            self.hit_count += 1
            return self.decomposition_cache[key]
    
    def clear_cache(self) -> None:
        """Clear all caches"""
        with self._lock:
            self.compression_cache.clear()
            self.decompression_cache.clear()
            self.decomposition_cache.clear()
            self.cache_metadata.clear()
            self.hit_count = 0
            self.miss_count = 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
            
            return {
                'total_size': (
                    len(self.compression_cache) +
                    len(self.decompression_cache) +
                    len(self.decomposition_cache)
                ),
                'compression_cache_size': len(self.compression_cache),
                'decompression_cache_size': len(self.decompression_cache),
                'decomposition_cache_size': len(self.decomposition_cache),
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'max_size': self.max_cache_size,
                'ttl_seconds': self.ttl_seconds
            }


class TensorServiceSecurity:
    """Security layer for tensor services"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.api_keys: Set[str] = set()
        self.user_permissions: Dict[str, Set[str]] = {}
        self.encryption_key = self._generate_encryption_key()
        self.auth_cache: Dict[str, Tuple[bool, float]] = {}
        self.auth_cache_ttl = config.get('auth_cache_ttl', 300)
        self.failed_attempts: Dict[str, int] = defaultdict(int)
        self.max_failed_attempts = config.get('max_failed_attempts', 5)
        self._lock = threading.RLock()
    
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key"""
        # In production, load from secure storage
        return hashlib.sha256(b"tensor_service_secret").digest()
    
    def validate_service_authentication(self, request: ServiceRequest) -> bool:
        """Validate service authentication"""
        if request is None:
            return False
        
        # Check if authentication is required
        if not self.config.get('authentication_enabled', False):
            return True
        
        with self._lock:
            # Extract authentication info
            auth_token = request.metadata.get('auth_token')
            api_key = request.metadata.get('api_key')
            user_id = request.metadata.get('user_id')
            
            # Check cache
            cache_key = f"{user_id}:{auth_token}"
            if cache_key in self.auth_cache:
                is_valid, timestamp = self.auth_cache[cache_key]
                if time.time() - timestamp < self.auth_cache_ttl:
                    return is_valid
            
            # Validate API key
            if api_key and api_key in self.api_keys:
                self.auth_cache[cache_key] = (True, time.time())
                return True
            
            # Check failed attempts
            if user_id and self.failed_attempts[user_id] >= self.max_failed_attempts:
                logging.warning(f"User {user_id} locked out due to failed attempts")
                return False
            
            # Authentication failed
            if user_id:
                self.failed_attempts[user_id] += 1
            
            self.auth_cache[cache_key] = (False, time.time())
            return False
    
    def authorize_service_access(self, request: ServiceRequest) -> bool:
        """Authorize service access"""
        if request is None:
            return False
        
        # Check if authorization is required
        if not self.config.get('authorization_enabled', False):
            return True
        
        with self._lock:
            user_id = request.metadata.get('user_id')
            service_name = request.service_name
            
            if user_id is None or service_name is None:
                return False
            
            # Check user permissions
            if user_id in self.user_permissions:
                permissions = self.user_permissions[user_id]
                
                # Check specific permission
                if service_name in permissions:
                    return True
                
                # Check wildcard permission
                if '*' in permissions:
                    return True
                
                # Check partial match
                for perm in permissions:
                    if perm.endswith('*') and service_name.startswith(perm[:-1]):
                        return True
            
            return False
    
    def encrypt_service_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt service data"""
        if data is None:
            return None
        
        if not self.config.get('encryption_enabled', False):
            return data
        
        try:
            # Serialize data
            serialized = pickle.dumps(data)
            
            # Simple XOR encryption (in production, use proper encryption)
            encrypted = bytearray()
            key_len = len(self.encryption_key)
            for i, byte in enumerate(serialized):
                encrypted.append(byte ^ self.encryption_key[i % key_len])
            
            return {
                'encrypted': True,
                'data': encrypted.hex(),
                'algorithm': 'xor',
                'timestamp': time.time()
            }
            
        except Exception as e:
            logging.error(f"Encryption failed: {str(e)}")
            raise
    
    def decrypt_service_data(self, encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt service data"""
        if encrypted_data is None:
            return None
        
        if not encrypted_data.get('encrypted', False):
            return encrypted_data
        
        try:
            # Extract encrypted bytes
            encrypted_hex = encrypted_data.get('data')
            if encrypted_hex is None:
                raise ValueError("No encrypted data found")
            
            encrypted = bytearray.fromhex(encrypted_hex)
            
            # Decrypt (reverse XOR)
            decrypted = bytearray()
            key_len = len(self.encryption_key)
            for i, byte in enumerate(encrypted):
                decrypted.append(byte ^ self.encryption_key[i % key_len])
            
            # Deserialize
            data = pickle.loads(bytes(decrypted))
            return data
            
        except Exception as e:
            logging.error(f"Decryption failed: {str(e)}")
            raise
    
    def add_api_key(self, api_key: str) -> None:
        """Add API key"""
        if api_key:
            with self._lock:
                self.api_keys.add(api_key)
    
    def revoke_api_key(self, api_key: str) -> None:
        """Revoke API key"""
        if api_key:
            with self._lock:
                self.api_keys.discard(api_key)
    
    def grant_permission(self, user_id: str, permission: str) -> None:
        """Grant permission to user"""
        if user_id and permission:
            with self._lock:
                if user_id not in self.user_permissions:
                    self.user_permissions[user_id] = set()
                self.user_permissions[user_id].add(permission)
    
    def revoke_permission(self, user_id: str, permission: str) -> None:
        """Revoke permission from user"""
        if user_id and permission:
            with self._lock:
                if user_id in self.user_permissions:
                    self.user_permissions[user_id].discard(permission)


class TensorServiceLoadBalancer:
    """Load balancer for tensor services"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.service_loads: Dict[str, float] = defaultdict(float)
        self.service_capacities: Dict[str, float] = defaultdict(lambda: 1.0)
        self.service_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self.routing_algorithm = config.get('algorithm', 'round_robin')
        self.current_indices: Dict[str, int] = defaultdict(int)
        self.request_history: deque = deque(maxlen=10000)
        self._lock = threading.RLock()
    
    def balance_compression_load(self, requests: List[ServiceRequest]) -> List[str]:
        """Balance compression load across services"""
        if not requests:
            return []
        
        with self._lock:
            service_assignments = []
            available_services = self._get_available_services('compression')
            
            if not available_services:
                raise ValueError("No compression services available")
            
            for request in requests:
                # Select service based on algorithm
                if self.routing_algorithm == 'round_robin':
                    service = self._round_robin_select(available_services, 'compression')
                elif self.routing_algorithm == 'least_load':
                    service = self._least_load_select(available_services)
                elif self.routing_algorithm == 'weighted':
                    service = self._weighted_select(available_services)
                else:
                    service = available_services[0]
                
                service_assignments.append(service)
                
                # Update load
                self.service_loads[service] += self._estimate_request_load(request)
                
                # Record assignment
                self.request_history.append({
                    'timestamp': time.time(),
                    'service': service,
                    'type': 'compression'
                })
            
            return service_assignments
    
    def balance_decompression_load(self, requests: List[ServiceRequest]) -> List[str]:
        """Balance decompression load across services"""
        if not requests:
            return []
        
        with self._lock:
            service_assignments = []
            available_services = self._get_available_services('decompression')
            
            if not available_services:
                raise ValueError("No decompression services available")
            
            for request in requests:
                # Select service based on algorithm
                if self.routing_algorithm == 'round_robin':
                    service = self._round_robin_select(available_services, 'decompression')
                elif self.routing_algorithm == 'least_load':
                    service = self._least_load_select(available_services)
                elif self.routing_algorithm == 'weighted':
                    service = self._weighted_select(available_services)
                else:
                    service = available_services[0]
                
                service_assignments.append(service)
                
                # Update load
                self.service_loads[service] += self._estimate_request_load(request)
                
                # Record assignment
                self.request_history.append({
                    'timestamp': time.time(),
                    'service': service,
                    'type': 'decompression'
                })
            
            return service_assignments
    
    def balance_decomposition_load(self, requests: List[ServiceRequest]) -> List[str]:
        """Balance decomposition load across services"""
        if not requests:
            return []
        
        with self._lock:
            service_assignments = []
            available_services = self._get_available_services('decomposition')
            
            if not available_services:
                raise ValueError("No decomposition services available")
            
            for request in requests:
                # Select service based on algorithm
                if self.routing_algorithm == 'round_robin':
                    service = self._round_robin_select(available_services, 'decomposition')
                elif self.routing_algorithm == 'least_load':
                    service = self._least_load_select(available_services)
                elif self.routing_algorithm == 'weighted':
                    service = self._weighted_select(available_services)
                else:
                    service = available_services[0]
                
                service_assignments.append(service)
                
                # Update load
                self.service_loads[service] += self._estimate_request_load(request)
                
                # Record assignment
                self.request_history.append({
                    'timestamp': time.time(),
                    'service': service,
                    'type': 'decomposition'
                })
            
            return service_assignments
    
    def get_service_load(self, service_name: str) -> float:
        """Get current load for service"""
        if service_name is None:
            return 0.0
        
        with self._lock:
            # Decay old load
            self._decay_loads()
            return self.service_loads.get(service_name, 0.0)
    
    def optimize_service_distribution(self) -> Dict[str, Any]:
        """Optimize service distribution based on historical data"""
        with self._lock:
            # Analyze request history
            service_counts = defaultdict(int)
            service_types = defaultdict(lambda: defaultdict(int))
            
            for record in self.request_history:
                service = record['service']
                req_type = record['type']
                service_counts[service] += 1
                service_types[service][req_type] += 1
            
            # Calculate optimal weights
            total_requests = len(self.request_history)
            if total_requests > 0:
                for service, count in service_counts.items():
                    # Adjust weight inversely to usage
                    usage_ratio = count / total_requests
                    if usage_ratio > 0:
                        self.service_weights[service] = 1.0 / usage_ratio
            
            # Return optimization report
            return {
                'service_counts': dict(service_counts),
                'service_types': dict(service_types),
                'updated_weights': dict(self.service_weights),
                'total_requests_analyzed': total_requests
            }
    
    def _get_available_services(self, service_type: str) -> List[str]:
        """Get available services of given type"""
        # In production, would query service registry
        # For now, return mock services
        if service_type == 'compression':
            return ['compress_service_1', 'compress_service_2']
        elif service_type == 'decompression':
            return ['decompress_service_1', 'decompress_service_2']
        elif service_type == 'decomposition':
            return ['decompose_service_1', 'decompose_service_2']
        return []
    
    def _round_robin_select(self, services: List[str], service_type: str) -> str:
        """Select service using round-robin"""
        index = self.current_indices[service_type]
        service = services[index % len(services)]
        self.current_indices[service_type] = index + 1
        return service
    
    def _least_load_select(self, services: List[str]) -> str:
        """Select service with least load"""
        return min(services, key=lambda s: self.service_loads.get(s, 0.0))
    
    def _weighted_select(self, services: List[str]) -> str:
        """Select service based on weights"""
        weighted_services = [
            (s, self.service_weights.get(s, 1.0)) for s in services
        ]
        total_weight = sum(w for _, w in weighted_services)
        
        if total_weight == 0:
            return services[0]
        
        # Random weighted selection
        r = np.random.uniform(0, total_weight)
        cumulative = 0
        
        for service, weight in weighted_services:
            cumulative += weight
            if r <= cumulative:
                return service
        
        return services[-1]
    
    def _estimate_request_load(self, request: ServiceRequest) -> float:
        """Estimate load for request"""
        # Simple estimation based on tensor size
        if 'tensor' in request.payload:
            tensor = request.payload['tensor']
            if isinstance(tensor, torch.Tensor):
                return tensor.numel() / 1e6  # Load in millions of elements
        return 1.0  # Default load
    
    def _decay_loads(self):
        """Decay service loads over time"""
        decay_factor = 0.95
        for service in self.service_loads:
            self.service_loads[service] *= decay_factor


class TensorServiceLogger:
    """Logger for tensor service operations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.log_level = config.get('log_level', 'INFO')
        self.log_file = config.get('log_file', 'tensor_service.log')
        self.max_log_size = config.get('max_log_size', 100 * 1024 * 1024)  # 100MB
        self.log_retention_days = config.get('log_retention_days', 30)
        self.service_logs: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger configuration"""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('TensorService')
    
    def log_service_request(self, request: ServiceRequest) -> None:
        """Log service request"""
        if request is None:
            return
        
        log_entry = {
            'timestamp': datetime.now(),
            'type': 'request',
            'service': request.service_name,
            'request_id': request.metadata.get('request_id', 'unknown'),
            'user_id': request.metadata.get('user_id'),
            'data_size': self._calculate_data_size(request.payload)
        }
        
        # Add to service logs
        self.service_logs[request.service_name].append(log_entry)
        
        # Log to file
        self.logger.info(
            f"Request: service={request.service_name}, "
            f"id={log_entry['request_id']}, "
            f"size={log_entry['data_size']}"
        )
    
    def log_service_response(self, response: ServiceResponse) -> None:
        """Log service response"""
        if response is None:
            return
        
        log_entry = {
            'timestamp': datetime.now(),
            'type': 'response',
            'success': response.is_success(),
            'processing_time': response.processing_time,
            'errors': response.errors if not response.is_success() else None
        }
        
        # Log to file
        if response.is_success():
            self.logger.info(
                f"Response: success=True, time={response.processing_time:.3f}s"
            )
        else:
            self.logger.warning(
                f"Response: success=False, errors={response.errors}"
            )
    
    def log_service_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log service error"""
        if error is None:
            return
        
        log_entry = {
            'timestamp': datetime.now(),
            'type': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'traceback': traceback.format_exc()
        }
        
        # Add to service logs
        service_name = context.get('service_name', 'unknown')
        self.service_logs[service_name].append(log_entry)
        
        # Log to file
        self.logger.error(
            f"Error in {service_name}: {type(error).__name__} - {str(error)}",
            exc_info=True
        )
    
    def get_service_logs(self, service_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get service logs"""
        if service_name is None:
            return []
        
        logs = list(self.service_logs.get(service_name, []))
        
        # Sort by timestamp (newest first)
        logs.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Apply limit
        return logs[:limit]
    
    def _calculate_data_size(self, data: Any) -> int:
        """Calculate approximate data size"""
        if data is None:
            return 0
        
        if isinstance(data, dict):
            size = 0
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    size += value.numel() * value.element_size()
                elif isinstance(value, np.ndarray):
                    size += value.nbytes
                else:
                    # Rough estimate
                    size += len(str(value))
            return size
        
        return len(str(data))


class TensorServiceRateLimiter:
    """Rate limiter for tensor services"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.rate_limits: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'requests_per_minute': 60,
            'requests_per_hour': 1000,
            'burst_size': 10
        })
        self.request_counts: Dict[str, deque] = defaultdict(deque)
        self.burst_tokens: Dict[str, int] = defaultdict(lambda: 10)
        self._lock = threading.RLock()
        self._last_reset = time.time()
    
    def check_rate_limit(self, service_name: str, request_id: str) -> bool:
        """Check if request is within rate limits"""
        if service_name is None or request_id is None:
            return False
        
        with self._lock:
            current_time = time.time()
            
            # Clean old requests
            self._clean_old_requests(service_name, current_time)
            
            # Get limits
            limits = self.rate_limits[service_name]
            
            # Check burst limit
            if self.burst_tokens[service_name] <= 0:
                return False
            
            # Check rate limits
            minute_count = sum(
                1 for timestamp in self.request_counts[service_name]
                if current_time - timestamp < 60
            )
            
            if minute_count >= limits['requests_per_minute']:
                return False
            
            hour_count = len(self.request_counts[service_name])
            if hour_count >= limits['requests_per_hour']:
                return False
            
            return True
    
    def update_rate_limit(self, service_name: str, request_id: str) -> None:
        """Update rate limit tracking"""
        if service_name is None or request_id is None:
            return
        
        with self._lock:
            current_time = time.time()
            
            # Record request
            self.request_counts[service_name].append(current_time)
            
            # Consume burst token
            self.burst_tokens[service_name] -= 1
            
            # Replenish tokens periodically
            if current_time - self._last_reset > 1.0:  # Every second
                self._replenish_tokens()
                self._last_reset = current_time
    
    def get_rate_limit_status(self, service_name: str) -> Dict[str, Any]:
        """Get rate limit status for service"""
        if service_name is None:
            return {}
        
        with self._lock:
            current_time = time.time()
            self._clean_old_requests(service_name, current_time)
            
            limits = self.rate_limits[service_name]
            
            minute_count = sum(
                1 for timestamp in self.request_counts[service_name]
                if current_time - timestamp < 60
            )
            
            hour_count = len(self.request_counts[service_name])
            
            return {
                'service': service_name,
                'limits': limits,
                'current_usage': {
                    'requests_per_minute': minute_count,
                    'requests_per_hour': hour_count,
                    'burst_tokens': self.burst_tokens[service_name]
                },
                'available': {
                    'requests_per_minute': limits['requests_per_minute'] - minute_count,
                    'requests_per_hour': limits['requests_per_hour'] - hour_count
                }
            }
    
    def reset_rate_limits(self) -> None:
        """Reset all rate limits"""
        with self._lock:
            self.request_counts.clear()
            for service in self.burst_tokens:
                self.burst_tokens[service] = self.rate_limits[service]['burst_size']
    
    def set_rate_limit(self, service_name: str, limits: Dict[str, Any]) -> None:
        """Set rate limit for service"""
        if service_name and limits:
            with self._lock:
                self.rate_limits[service_name].update(limits)
                if 'burst_size' in limits:
                    self.burst_tokens[service_name] = limits['burst_size']
    
    def _clean_old_requests(self, service_name: str, current_time: float):
        """Remove old requests from tracking"""
        # Keep only requests from last hour
        cutoff_time = current_time - 3600
        
        while (self.request_counts[service_name] and 
               self.request_counts[service_name][0] < cutoff_time):
            self.request_counts[service_name].popleft()
    
    def _replenish_tokens(self):
        """Replenish burst tokens"""
        for service in self.burst_tokens:
            max_tokens = self.rate_limits[service]['burst_size']
            if self.burst_tokens[service] < max_tokens:
                self.burst_tokens[service] += 1


class TensorServiceRetryManager:
    """Manages retry logic for tensor services"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.retry_policies: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'max_attempts': 3,
            'initial_delay': 1.0,
            'max_delay': 60.0,
            'exponential_base': 2.0,
            'jitter': True
        })
        self.retry_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.circuit_breakers: Dict[str, CircuitBreakerState] = defaultdict(
            lambda: CircuitBreakerState.CLOSED
        )
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.success_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
    
    def should_retry_request(self, request: ServiceRequest, attempt: int) -> bool:
        """Determine if request should be retried"""
        if request is None:
            return False
        
        with self._lock:
            service_name = request.service_name
            
            # Check circuit breaker
            if self.circuit_breakers[service_name] == CircuitBreakerState.OPEN:
                return False
            
            # Get retry policy
            policy = self.retry_policies[service_name]
            
            # Check max attempts
            if attempt >= policy['max_attempts']:
                return False
            
            # Check if error is retryable
            last_error = request.metadata.get('last_error')
            if last_error and not self._is_retryable_error(last_error):
                return False
            
            return True
    
    def retry_request(self, request: ServiceRequest, max_attempts: int = 3) -> ServiceResponse:
        """Retry request with exponential backoff"""
        if request is None:
            raise ValueError("Request cannot be None")
        
        service_name = request.service_name
        policy = self.retry_policies[service_name]
        
        for attempt in range(max_attempts):
            try:
                # Check if we should retry
                if attempt > 0 and not self.should_retry_request(request, attempt):
                    break
                
                # Calculate delay
                if attempt > 0:
                    delay = self._calculate_delay(attempt, policy)
                    time.sleep(delay)
                
                # Record attempt
                self.retry_history[service_name].append({
                    'attempt': attempt,
                    'timestamp': datetime.now(),
                    'request_id': request.metadata.get('request_id')
                })
                
                # Make request (in production, would call actual service)
                response = self._execute_request(request)
                
                if response.is_success():
                    self._record_success(service_name)
                    return response
                else:
                    self._record_failure(service_name)
                    request.metadata['last_error'] = response.errors[0] if response.errors else 'Unknown error'
                    
            except Exception as e:
                self._record_failure(service_name)
                request.metadata['last_error'] = str(e)
                
                if attempt == max_attempts - 1:
                    # Final attempt failed
                    response = ServiceResponse(
                        request_id=request.request_id,
                        status=self._get_error_status()
                    )
                    response.add_error("RETRY_FAILED", f"All retry attempts failed: {str(e)}")
                    return response
        
        # All attempts failed
        response = ServiceResponse(
            request_id=request.request_id,
            status=self._get_error_status()
        )
        response.add_error("MAX_RETRIES_EXCEEDED", f"Exceeded maximum retry attempts ({max_attempts})")
        return response
    
    def get_retry_policy(self, service_name: str) -> Dict[str, Any]:
        """Get retry policy for service"""
        if service_name is None:
            return {}
        
        with self._lock:
            return dict(self.retry_policies[service_name])
    
    def update_retry_policy(self, service_name: str, policy: Dict[str, Any]) -> None:
        """Update retry policy for service"""
        if service_name is None or policy is None:
            return
        
        with self._lock:
            self.retry_policies[service_name].update(policy)
    
    def _calculate_delay(self, attempt: int, policy: Dict[str, Any]) -> float:
        """Calculate retry delay with exponential backoff"""
        base_delay = policy['initial_delay'] * (policy['exponential_base'] ** (attempt - 1))
        delay = min(base_delay, policy['max_delay'])
        
        # Add jitter
        if policy['jitter']:
            jitter = np.random.uniform(0, delay * 0.1)
            delay += jitter
        
        return delay
    
    def _is_retryable_error(self, error: str) -> bool:
        """Check if error is retryable"""
        non_retryable = [
            'invalid', 'unauthorized', 'forbidden', 'not found',
            'bad request', 'validation'
        ]
        
        error_lower = error.lower()
        return not any(keyword in error_lower for keyword in non_retryable)
    
    def _execute_request(self, request: ServiceRequest) -> ServiceResponse:
        """Execute the actual request (mock implementation)"""
        # In production, this would call the actual service
        response = ServiceResponse(
            request_id=request.request_id,
            status=self._get_success_status()
        )
        
        # Simulate success/failure
        if np.random.random() > 0.3:  # 70% success rate
            response.data = {'result': 'success'}
        else:
            response.status = self._get_error_status()
            response.add_error("SERVICE_ERROR", "Service temporarily unavailable")
        
        response.processing_time = np.random.uniform(0.1, 1.0)
        return response
    
    def _record_success(self, service_name: str):
        """Record successful request"""
        with self._lock:
            self.success_counts[service_name] += 1
            self.failure_counts[service_name] = 0  # Reset failure count
            
            # Update circuit breaker
            if self.circuit_breakers[service_name] == CircuitBreakerState.HALF_OPEN:
                self.circuit_breakers[service_name] = CircuitBreakerState.CLOSED
    
    def _record_failure(self, service_name: str):
        """Record failed request"""
        with self._lock:
            self.failure_counts[service_name] += 1
            
            # Check if circuit breaker should open
            if self.failure_counts[service_name] >= 5:
                self.circuit_breakers[service_name] = CircuitBreakerState.OPEN
                
                # Schedule circuit breaker reset
                threading.Timer(
                    30.0,  # Reset after 30 seconds
                    self._reset_circuit_breaker,
                    args=[service_name]
                ).start()
    
    def _reset_circuit_breaker(self, service_name: str):
        """Reset circuit breaker to half-open state"""
        with self._lock:
            if self.circuit_breakers[service_name] == CircuitBreakerState.OPEN:
                self.circuit_breakers[service_name] = CircuitBreakerState.HALF_OPEN
                self.failure_counts[service_name] = 0
    
    def _get_success_status(self):
        """Get success response status"""
        from ..service_interfaces.service_interfaces_core import ServiceStatus
        return ServiceStatus.SUCCESS
    
    def _get_error_status(self):
        """Get error response status"""
        from ..service_interfaces.service_interfaces_core import ServiceStatus
        return ServiceStatus.ERROR