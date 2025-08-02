"""
Service Interface Definitions for Compression Systems
NO FALLBACKS - HARD FAILURES ONLY
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from datetime import datetime
from enum import Enum
import time
import json
import hashlib
import uuid
from abc import ABC, abstractmethod


class ServiceStatus(Enum):
    """Service status enumeration"""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    INVALID = "invalid"
    UNAUTHORIZED = "unauthorized"
    UNAVAILABLE = "unavailable"


class ServiceHealth(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceRequest:
    """Standardized service request format"""
    service_name: str
    method_name: str
    version: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    auth_token: Optional[str] = None
    timeout: Optional[float] = None
    priority: int = 0
    
    def __post_init__(self):
        """Validate request on initialization"""
        if not self.service_name or not isinstance(self.service_name, str):
            raise ValueError("Service name must be non-empty string")
        if not self.method_name or not isinstance(self.method_name, str):
            raise ValueError("Method name must be non-empty string")
        if not self.version or not isinstance(self.version, str):
            raise ValueError("Version must be non-empty string")
        if self.payload is None:
            raise ValueError("Payload cannot be None")
        if not isinstance(self.payload, dict):
            raise ValueError("Payload must be dictionary")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary"""
        return {
            'service_name': self.service_name,
            'method_name': self.method_name,
            'version': self.version,
            'payload': self.payload,
            'metadata': self.metadata,
            'request_id': self.request_id,
            'timestamp': self.timestamp.isoformat(),
            'auth_token': self.auth_token,
            'timeout': self.timeout,
            'priority': self.priority
        }
    
    def get_signature(self) -> str:
        """Generate request signature for caching/tracking"""
        content = f"{self.service_name}:{self.method_name}:{self.version}:{json.dumps(self.payload, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class ServiceResponse:
    """Standardized service response format"""
    request_id: str
    status: ServiceStatus
    data: Optional[Dict[str, Any]] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    processing_time: Optional[float] = None
    service_version: Optional[str] = None
    
    def __post_init__(self):
        """Validate response on initialization"""
        if not self.request_id or not isinstance(self.request_id, str):
            raise ValueError("Request ID must be non-empty string")
        if not isinstance(self.status, ServiceStatus):
            raise ValueError("Status must be ServiceStatus enum")
        if self.data is not None and not isinstance(self.data, dict):
            raise ValueError("Data must be dictionary or None")
        if not isinstance(self.errors, list):
            raise ValueError("Errors must be list")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return {
            'request_id': self.request_id,
            'status': self.status.value,
            'data': self.data,
            'errors': self.errors,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'processing_time': self.processing_time,
            'service_version': self.service_version
        }
    
    def is_success(self) -> bool:
        """Check if response indicates success"""
        return self.status == ServiceStatus.SUCCESS
    
    def add_error(self, error_code: str, error_message: str, error_details: Optional[Dict[str, Any]] = None) -> None:
        """Add error to response"""
        if not error_code or not isinstance(error_code, str):
            raise ValueError("Error code must be non-empty string")
        if not error_message or not isinstance(error_message, str):
            raise ValueError("Error message must be non-empty string")
        
        error = {
            'code': error_code,
            'message': error_message,
            'timestamp': datetime.utcnow().isoformat()
        }
        if error_details:
            error['details'] = error_details
        
        self.errors.append(error)


class ServiceValidation:
    """Service request/response validation"""
    
    def __init__(self):
        self.validation_rules = {}
        self.contract_definitions = {}
        self.version_compatibility = {}
        self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> None:
        """Initialize default validation rules"""
        self.validation_rules = {
            'request': {
                'required_fields': ['service_name', 'method_name', 'version', 'payload'],
                'field_types': {
                    'service_name': str,
                    'method_name': str,
                    'version': str,
                    'payload': dict,
                    'metadata': dict,
                    'request_id': str,
                    'priority': int
                },
                'version_pattern': r'^\d+\.\d+\.\d+$'
            },
            'response': {
                'required_fields': ['request_id', 'status'],
                'field_types': {
                    'request_id': str,
                    'status': str,
                    'data': (dict, type(None)),
                    'errors': list,
                    'metadata': dict
                }
            }
        }
    
    def validate_request(self, request: ServiceRequest) -> Tuple[bool, List[str]]:
        """Validate service request"""
        if not isinstance(request, ServiceRequest):
            raise TypeError("Request must be ServiceRequest instance")
        
        errors = []
        
        # Validate required fields
        for field in self.validation_rules['request']['required_fields']:
            if not hasattr(request, field) or getattr(request, field) is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate version format
        import re
        if not re.match(self.validation_rules['request']['version_pattern'], request.version):
            errors.append(f"Invalid version format: {request.version}")
        
        # Validate payload structure if contract exists
        contract_key = f"{request.service_name}.{request.method_name}"
        if contract_key in self.contract_definitions:
            contract_errors = self._validate_against_contract(
                request.payload,
                self.contract_definitions[contract_key]
            )
            errors.extend(contract_errors)
        
        return len(errors) == 0, errors
    
    def validate_response(self, response: ServiceResponse) -> Tuple[bool, List[str]]:
        """Validate service response"""
        if not isinstance(response, ServiceResponse):
            raise TypeError("Response must be ServiceResponse instance")
        
        errors = []
        
        # Validate required fields
        for field in self.validation_rules['response']['required_fields']:
            if not hasattr(response, field) or getattr(response, field) is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate status
        if not isinstance(response.status, ServiceStatus):
            errors.append("Invalid status type")
        
        # Validate errors structure
        for idx, error in enumerate(response.errors):
            if not isinstance(error, dict):
                errors.append(f"Error at index {idx} must be dictionary")
            elif 'code' not in error or 'message' not in error:
                errors.append(f"Error at index {idx} missing required fields")
        
        return len(errors) == 0, errors
    
    def register_service_contract(self, service_name: str, method_name: str, 
                                contract: Dict[str, Any]) -> None:
        """Register service contract for validation"""
        if not service_name or not isinstance(service_name, str):
            raise ValueError("Service name must be non-empty string")
        if not method_name or not isinstance(method_name, str):
            raise ValueError("Method name must be non-empty string")
        if contract is None or not isinstance(contract, dict):
            raise ValueError("Contract must be non-empty dictionary")
        
        contract_key = f"{service_name}.{method_name}"
        self.contract_definitions[contract_key] = contract
    
    def _validate_against_contract(self, data: Dict[str, Any], 
                                 contract: Dict[str, Any]) -> List[str]:
        """Validate data against contract definition"""
        errors = []
        
        # Check required fields
        if 'required' in contract:
            for field in contract['required']:
                if field not in data:
                    errors.append(f"Missing required field: {field}")
        
        # Check field types
        if 'properties' in contract:
            for field, schema in contract['properties'].items():
                if field in data:
                    if 'type' in schema:
                        expected_type = schema['type']
                        if not self._check_type(data[field], expected_type):
                            errors.append(f"Field {field} has incorrect type")
                    
                    # Additional validations
                    if 'minLength' in schema and isinstance(data[field], str):
                        if len(data[field]) < schema['minLength']:
                            errors.append(f"Field {field} too short")
                    
                    if 'minimum' in schema and isinstance(data[field], (int, float)):
                        if data[field] < schema['minimum']:
                            errors.append(f"Field {field} below minimum value")
        
        return errors
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        type_mapping = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'object': dict,
            'array': list
        }
        
        if expected_type not in type_mapping:
            raise ValueError(f"Unknown type: {expected_type}")
        
        return isinstance(value, type_mapping[expected_type])
    
    def validate_version_compatibility(self, requested_version: str, 
                                     available_version: str) -> bool:
        """Check version compatibility"""
        if not requested_version or not isinstance(requested_version, str):
            raise ValueError("Requested version must be non-empty string")
        if not available_version or not isinstance(available_version, str):
            raise ValueError("Available version must be non-empty string")
        
        # Parse versions
        req_parts = [int(x) for x in requested_version.split('.')]
        avail_parts = [int(x) for x in available_version.split('.')]
        
        # Check major version compatibility
        if req_parts[0] != avail_parts[0]:
            return False
        
        # Check minor version (available must be >= requested)
        if avail_parts[1] < req_parts[1]:
            return False
        
        return True


class ServiceMetrics:
    """Service metrics tracking"""
    
    def __init__(self):
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        self.method_metrics = {}
        self.error_types = {}
        self.last_request_time = None
        self.last_error_time = None
    
    def record_request(self, request: ServiceRequest, response: ServiceResponse) -> None:
        """Record metrics for request/response pair"""
        if not isinstance(request, ServiceRequest):
            raise TypeError("Request must be ServiceRequest instance")
        if not isinstance(response, ServiceResponse):
            raise TypeError("Response must be ServiceResponse instance")
        
        self.request_count += 1
        self.last_request_time = datetime.utcnow()
        
        # Record success/error
        if response.is_success():
            self.success_count += 1
        else:
            self.error_count += 1
            self.last_error_time = datetime.utcnow()
            
            # Track error types
            for error in response.errors:
                error_code = error.get('code', 'unknown')
                self.error_types[error_code] = self.error_types.get(error_code, 0) + 1
        
        # Record processing time
        if response.processing_time:
            self.total_processing_time += response.processing_time
        
        # Record method-specific metrics
        method_key = f"{request.service_name}.{request.method_name}"
        if method_key not in self.method_metrics:
            self.method_metrics[method_key] = {
                'count': 0,
                'success': 0,
                'error': 0,
                'total_time': 0.0
            }
        
        self.method_metrics[method_key]['count'] += 1
        if response.is_success():
            self.method_metrics[method_key]['success'] += 1
        else:
            self.method_metrics[method_key]['error'] += 1
        
        if response.processing_time:
            self.method_metrics[method_key]['total_time'] += response.processing_time
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        avg_processing_time = (
            self.total_processing_time / self.request_count 
            if self.request_count > 0 else 0.0
        )
        
        success_rate = (
            self.success_count / self.request_count
            if self.request_count > 0 else 0.0
        )
        
        return {
            'total_requests': self.request_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': success_rate,
            'average_processing_time': avg_processing_time,
            'method_metrics': self.method_metrics,
            'error_types': self.error_types,
            'last_request_time': self.last_request_time.isoformat() if self.last_request_time else None,
            'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None
        }


class ServiceRegistry:
    """Service registration and discovery"""
    
    def __init__(self):
        self.services = {}
        self.service_metadata = {}
        self.service_health = {}
        self.service_versions = {}
    
    def register_service(self, service_name: str, service_instance: Any,
                       version: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a service"""
        if not service_name or not isinstance(service_name, str):
            raise ValueError("Service name must be non-empty string")
        if service_instance is None:
            raise ValueError("Service instance cannot be None")
        if not version or not isinstance(version, str):
            raise ValueError("Version must be non-empty string")
        
        # Validate version format
        import re
        if not re.match(r'^\d+\.\d+\.\d+$', version):
            raise ValueError(f"Invalid version format: {version}")
        
        # Store service
        if service_name not in self.services:
            self.services[service_name] = {}
        
        self.services[service_name][version] = service_instance
        
        # Store metadata
        if service_name not in self.service_metadata:
            self.service_metadata[service_name] = {}
        
        self.service_metadata[service_name][version] = metadata or {}
        self.service_metadata[service_name][version]['registered_at'] = datetime.utcnow().isoformat()
        
        # Initialize health status
        if service_name not in self.service_health:
            self.service_health[service_name] = {}
        
        self.service_health[service_name][version] = ServiceHealth.HEALTHY
        
        # Track versions
        if service_name not in self.service_versions:
            self.service_versions[service_name] = []
        
        if version not in self.service_versions[service_name]:
            self.service_versions[service_name].append(version)
            self.service_versions[service_name].sort(key=lambda v: [int(x) for x in v.split('.')])
    
    def unregister_service(self, service_name: str, version: Optional[str] = None) -> None:
        """Unregister a service"""
        if not service_name or not isinstance(service_name, str):
            raise ValueError("Service name must be non-empty string")
        
        if service_name not in self.services:
            raise ValueError(f"Service not found: {service_name}")
        
        if version:
            # Unregister specific version
            if version not in self.services[service_name]:
                raise ValueError(f"Version not found: {version}")
            
            del self.services[service_name][version]
            
            if service_name in self.service_metadata and version in self.service_metadata[service_name]:
                del self.service_metadata[service_name][version]
            
            if service_name in self.service_health and version in self.service_health[service_name]:
                del self.service_health[service_name][version]
            
            if service_name in self.service_versions and version in self.service_versions[service_name]:
                self.service_versions[service_name].remove(version)
            
            # Clean up empty entries
            if not self.services[service_name]:
                del self.services[service_name]
                if service_name in self.service_metadata:
                    del self.service_metadata[service_name]
                if service_name in self.service_health:
                    del self.service_health[service_name]
                if service_name in self.service_versions:
                    del self.service_versions[service_name]
        else:
            # Unregister all versions
            del self.services[service_name]
            
            if service_name in self.service_metadata:
                del self.service_metadata[service_name]
            
            if service_name in self.service_health:
                del self.service_health[service_name]
            
            if service_name in self.service_versions:
                del self.service_versions[service_name]
    
    def discover_service(self, service_name: str, version: Optional[str] = None) -> Any:
        """Discover and return service instance"""
        if not service_name or not isinstance(service_name, str):
            raise ValueError("Service name must be non-empty string")
        
        if service_name not in self.services:
            raise ValueError(f"Service not found: {service_name}")
        
        if version:
            # Get specific version
            if version not in self.services[service_name]:
                raise ValueError(f"Version not found: {version}")
            
            return self.services[service_name][version]
        else:
            # Get latest version
            if service_name not in self.service_versions or not self.service_versions[service_name]:
                raise ValueError(f"No versions available for service: {service_name}")
            
            latest_version = self.service_versions[service_name][-1]
            return self.services[service_name][latest_version]
    
    def list_services(self) -> Dict[str, List[str]]:
        """List all registered services and versions"""
        return dict(self.service_versions)
    
    def get_service_info(self, service_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get service information"""
        if not service_name or not isinstance(service_name, str):
            raise ValueError("Service name must be non-empty string")
        
        if service_name not in self.services:
            raise ValueError(f"Service not found: {service_name}")
        
        if version:
            if version not in self.services[service_name]:
                raise ValueError(f"Version not found: {version}")
            
            return {
                'name': service_name,
                'version': version,
                'metadata': self.service_metadata.get(service_name, {}).get(version, {}),
                'health': self.service_health.get(service_name, {}).get(version, ServiceHealth.UNKNOWN).value
            }
        else:
            # Return info for all versions
            info = {
                'name': service_name,
                'versions': {}
            }
            
            for ver in self.service_versions.get(service_name, []):
                info['versions'][ver] = {
                    'metadata': self.service_metadata.get(service_name, {}).get(ver, {}),
                    'health': self.service_health.get(service_name, {}).get(ver, ServiceHealth.UNKNOWN).value
                }
            
            return info
    
    def update_service_health(self, service_name: str, version: str, health: ServiceHealth) -> None:
        """Update service health status"""
        if not service_name or not isinstance(service_name, str):
            raise ValueError("Service name must be non-empty string")
        if not version or not isinstance(version, str):
            raise ValueError("Version must be non-empty string")
        if not isinstance(health, ServiceHealth):
            raise ValueError("Health must be ServiceHealth enum")
        
        if service_name not in self.services:
            raise ValueError(f"Service not found: {service_name}")
        
        if version not in self.services[service_name]:
            raise ValueError(f"Version not found: {version}")
        
        self.service_health[service_name][version] = health


class CompressionServiceInterface:
    """Main compression service interface"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.registry = ServiceRegistry()
        self.validation = ServiceValidation()
        self.metrics = ServiceMetrics()
        self.service_handlers = {}
        self.middleware_stack = []
        self.request_interceptors = []
        self.response_interceptors = []
        self._initialize_interface()
    
    def _initialize_interface(self) -> None:
        """Initialize service interface"""
        # Register built-in middleware
        self.add_middleware(self._validation_middleware)
        self.add_middleware(self._metrics_middleware)
        self.add_middleware(self._health_check_middleware)
    
    def register_service(self, service_name: str, service_instance: Any,
                       version: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a compression service"""
        # Register in registry
        self.registry.register_service(service_name, service_instance, version, metadata)
        
        # Extract and register service methods
        service_key = f"{service_name}:{version}"
        self.service_handlers[service_key] = {}
        
        # Inspect service instance for methods
        for method_name in dir(service_instance):
            if not method_name.startswith('_'):
                method = getattr(service_instance, method_name)
                if callable(method):
                    self.service_handlers[service_key][method_name] = method
    
    def unregister_service(self, service_name: str, version: Optional[str] = None) -> None:
        """Unregister a compression service"""
        # Unregister from registry
        self.registry.unregister_service(service_name, version)
        
        # Remove service handlers
        if version:
            service_key = f"{service_name}:{version}"
            if service_key in self.service_handlers:
                del self.service_handlers[service_key]
        else:
            # Remove all versions
            keys_to_remove = [k for k in self.service_handlers.keys() if k.startswith(f"{service_name}:")]
            for key in keys_to_remove:
                del self.service_handlers[key]
    
    def invoke_service(self, request: ServiceRequest) -> ServiceResponse:
        """Invoke a compression service"""
        start_time = time.time()
        
        try:
            # Apply request interceptors
            for interceptor in self.request_interceptors:
                request = interceptor(request)
            
            # Process through middleware stack
            response = self._process_middleware(request, 0)
            
            # Apply response interceptors
            for interceptor in self.response_interceptors:
                response = interceptor(response)
            
            # Set processing time
            response.processing_time = time.time() - start_time
            
            return response
            
        except Exception as e:
            # Create error response
            response = ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.ERROR,
                processing_time=time.time() - start_time
            )
            response.add_error(
                error_code="SERVICE_INVOCATION_ERROR",
                error_message=str(e),
                error_details={'exception_type': type(e).__name__}
            )
            return response
    
    def _process_middleware(self, request: ServiceRequest, index: int) -> ServiceResponse:
        """Process request through middleware stack"""
        if index >= len(self.middleware_stack):
            # End of middleware stack, invoke actual service
            return self._invoke_service_handler(request)
        
        # Call current middleware with next function
        middleware = self.middleware_stack[index]
        next_func = lambda req: self._process_middleware(req, index + 1)
        return middleware(request, next_func)
    
    def _invoke_service_handler(self, request: ServiceRequest) -> ServiceResponse:
        """Invoke the actual service handler"""
        # Find service handler
        service_key = f"{request.service_name}:{request.version}"
        
        if service_key not in self.service_handlers:
            # Try to find compatible version
            compatible_version = self._find_compatible_version(request.service_name, request.version)
            if compatible_version:
                service_key = f"{request.service_name}:{compatible_version}"
            else:
                response = ServiceResponse(
                    request_id=request.request_id,
                    status=ServiceStatus.UNAVAILABLE
                )
                response.add_error(
                    error_code="SERVICE_NOT_FOUND",
                    error_message=f"Service not found: {request.service_name}:{request.version}"
                )
                return response
        
        if request.method_name not in self.service_handlers[service_key]:
            response = ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.INVALID
            )
            response.add_error(
                error_code="METHOD_NOT_FOUND",
                error_message=f"Method not found: {request.method_name}"
            )
            return response
        
        # Invoke handler
        try:
            handler = self.service_handlers[service_key][request.method_name]
            result = handler(**request.payload)
            
            # Create success response
            response = ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.SUCCESS,
                data=result if isinstance(result, dict) else {'result': result},
                service_version=service_key.split(':')[1]
            )
            
            return response
            
        except Exception as e:
            response = ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.ERROR
            )
            response.add_error(
                error_code="HANDLER_ERROR",
                error_message=str(e),
                error_details={'exception_type': type(e).__name__}
            )
            return response
    
    def _find_compatible_version(self, service_name: str, requested_version: str) -> Optional[str]:
        """Find compatible service version"""
        available_versions = self.registry.service_versions.get(service_name, [])
        
        for version in reversed(available_versions):
            if self.validation.validate_version_compatibility(requested_version, version):
                return version
        
        return None
    
    def add_middleware(self, middleware: Callable) -> None:
        """Add middleware to processing stack"""
        if middleware is None:
            raise ValueError("Middleware cannot be None")
        if not callable(middleware):
            raise ValueError("Middleware must be callable")
        
        self.middleware_stack.append(middleware)
    
    def add_request_interceptor(self, interceptor: Callable) -> None:
        """Add request interceptor"""
        if interceptor is None:
            raise ValueError("Interceptor cannot be None")
        if not callable(interceptor):
            raise ValueError("Interceptor must be callable")
        
        self.request_interceptors.append(interceptor)
    
    def add_response_interceptor(self, interceptor: Callable) -> None:
        """Add response interceptor"""
        if interceptor is None:
            raise ValueError("Interceptor cannot be None")
        if not callable(interceptor):
            raise ValueError("Interceptor must be callable")
        
        self.response_interceptors.append(interceptor)
    
    def _validation_middleware(self, request: ServiceRequest, next_func: Callable) -> ServiceResponse:
        """Validation middleware"""
        # Validate request
        is_valid, errors = self.validation.validate_request(request)
        
        if not is_valid:
            response = ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.INVALID
            )
            for error in errors:
                response.add_error(
                    error_code="VALIDATION_ERROR",
                    error_message=error
                )
            return response
        
        # Process request
        response = next_func(request)
        
        # Validate response
        is_valid, errors = self.validation.validate_response(response)
        
        if not is_valid:
            # Log validation errors but don't fail the response
            response.metadata['validation_errors'] = errors
        
        return response
    
    def _metrics_middleware(self, request: ServiceRequest, next_func: Callable) -> ServiceResponse:
        """Metrics collection middleware"""
        # Process request
        response = next_func(request)
        
        # Record metrics
        self.metrics.record_request(request, response)
        
        return response
    
    def _health_check_middleware(self, request: ServiceRequest, next_func: Callable) -> ServiceResponse:
        """Health check middleware"""
        # Check if this is a health check request
        if request.method_name == "_health_check":
            service_info = self.registry.get_service_info(request.service_name, request.version)
            
            response = ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.SUCCESS,
                data={
                    'service_info': service_info,
                    'metrics': self.metrics.get_metrics_summary()
                }
            )
            return response
        
        # Normal request processing
        return next_func(request)
    
    def get_service_health(self, service_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get service health information"""
        if not service_name or not isinstance(service_name, str):
            raise ValueError("Service name must be non-empty string")
        
        service_info = self.registry.get_service_info(service_name, version)
        
        # Add metrics data
        metrics_summary = self.metrics.get_metrics_summary()
        
        # Filter metrics for this service
        service_metrics = {}
        for method_key, method_data in metrics_summary['method_metrics'].items():
            if method_key.startswith(f"{service_name}."):
                service_metrics[method_key] = method_data
        
        return {
            'service': service_info,
            'metrics': service_metrics,
            'overall_health': self._calculate_health_score(service_metrics)
        }
    
    def _calculate_health_score(self, service_metrics: Dict[str, Any]) -> str:
        """Calculate overall health score from metrics"""
        if not service_metrics:
            return ServiceHealth.UNKNOWN.value
        
        total_requests = sum(m['count'] for m in service_metrics.values())
        total_errors = sum(m['error'] for m in service_metrics.values())
        
        if total_requests == 0:
            return ServiceHealth.UNKNOWN.value
        
        error_rate = total_errors / total_requests
        
        if error_rate < 0.01:  # Less than 1% errors
            return ServiceHealth.HEALTHY.value
        elif error_rate < 0.05:  # Less than 5% errors
            return ServiceHealth.DEGRADED.value
        else:
            return ServiceHealth.UNHEALTHY.value
    
    def register_service_contract(self, service_name: str, method_name: str,
                                contract: Dict[str, Any]) -> None:
        """Register service contract for validation"""
        self.validation.register_service_contract(service_name, method_name, contract)
    
    def list_services(self) -> Dict[str, List[str]]:
        """List all registered services"""
        return self.registry.list_services()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get overall metrics summary"""
        return self.metrics.get_metrics_summary()


# Integration hooks for existing systems
class ServiceInterfaceIntegration:
    """Integration utilities for existing systems"""
    
    @staticmethod
    def register_with_brain_core(brain_core: Any, service_interface: CompressionServiceInterface) -> None:
        """Register service interface with BrainCore"""
        brain_core.register_service_interface(service_interface)
        brain_core.service_interface = service_interface
    
    @staticmethod
    def integrate_with_compression_systems(compression_system: Any, service_interface: CompressionServiceInterface) -> None:
        """Integrate with compression systems"""
        # Register compression system as service
        service_name = getattr(compression_system, '__class__').__name__.lower()
        version = "1.0.0"
        
        service_interface.register_service(
            service_name=service_name,
            service_instance=compression_system,
            version=version,
            metadata={
                'type': 'compression_system',
                'integrated_at': datetime.utcnow().isoformat()
            }
        )
        
        # Add service interface to compression system
        compression_system.service_interface = service_interface
        compression_system.create_service_request = lambda method, payload: ServiceRequest(
            service_name=service_name,
            method_name=method,
            version=version,
            payload=payload
        )
        compression_system.invoke_service = service_interface.invoke_service
    
    @staticmethod
    def integrate_with_api_layer(api_layer: Any, service_interface: CompressionServiceInterface) -> None:
        """Integrate with API layer"""
        # Add service interface to API layer
        api_layer.service_interface = service_interface
        
        # Create API endpoints for service operations
        api_layer.service_endpoints = {
            'invoke': lambda req: service_interface.invoke_service(req),
            'list_services': lambda: service_interface.list_services(),
            'get_health': lambda name, ver=None: service_interface.get_service_health(name, ver),
            'get_metrics': lambda: service_interface.get_metrics_summary()
        }