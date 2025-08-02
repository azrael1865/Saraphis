"""
Sheaf Service Middleware & Configuration Module

This module implements middleware components and configuration management for the Sheaf service layer,
including request/response processing, health monitoring, and caching capabilities.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import threading
import time
import json
import uuid
import hashlib
from pathlib import Path
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
import traceback
from enum import Enum
import asyncio
import pickle
import zlib
import base64
from cryptography.fernet import Fernet

# Import Sheaf service core
from .sheaf_service_core import (
    SheafServiceInterface, SheafServiceRegistry, SheafServiceOrchestrator,
    ServiceRequest, ServiceResponse, ServiceStatus, ServiceOperationType
)

# Import service interfaces
from ..service_interfaces.service_interfaces_core import (
    CompressionServiceInterface, ServiceValidation, ServiceMetrics,
    ServiceHealth
)

# Import Sheaf compression system
from .sheaf_core import SheafCompressionSystem, CellularSheaf
from .sheaf_advanced import (
    CellularSheafBuilder, RestrictionMapProcessor, 
    SheafCohomologyCalculator, SheafReconstructionEngine
)
from .sheaf_integration import (
    SheafBrainIntegration, SheafDomainIntegration, 
    SheafTrainingIntegration, SheafSystemOrchestrator
)

# Import GPU memory management
from ..gpu_memory.gpu_memory_core import GPUMemoryManager, StreamManager, MemoryOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MiddlewareType(Enum):
    """Types of middleware processing"""
    REQUEST_VALIDATION = "request_validation"
    REQUEST_TRANSFORMATION = "request_transformation"
    RESPONSE_TRANSFORMATION = "response_transformation"
    PERFORMANCE_MONITORING = "performance_monitoring"
    SECURITY_VALIDATION = "security_validation"
    RATE_LIMITING = "rate_limiting"
    CIRCUIT_BREAKING = "circuit_breaking"
    LOGGING = "logging"
    CACHING = "caching"
    ERROR_HANDLING = "error_handling"


@dataclass
class MiddlewareContext:
    """Context passed through middleware chain"""
    request: ServiceRequest
    response: Optional[ServiceResponse] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    middleware_chain: List[str] = field(default_factory=list)
    cache_key: Optional[str] = None
    skip_remaining: bool = False


class SheafServiceMiddleware:
    """
    Middleware component for Sheaf service layer.
    
    Provides request/response processing, validation, and transformation
    capabilities for the Sheaf compression service.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize middleware with configuration"""
        self.config = config or {}
        self.middleware_chain: List[Tuple[MiddlewareType, Callable]] = []
        self.request_validators: List[Callable] = []
        self.response_transformers: List[Callable] = []
        self.performance_monitors: List[Callable] = []
        self.error_handlers: Dict[type, Callable] = {}
        self.rate_limiter = self._create_rate_limiter()
        self.circuit_breaker = self._create_circuit_breaker()
        self.metrics = ServiceMetrics()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._lock = threading.RLock()
        self._initialize_middleware()
    
    def _initialize_middleware(self) -> None:
        """Initialize default middleware components"""
        try:
            # Add default middleware in order
            self.add_middleware(MiddlewareType.LOGGING, self._logging_middleware)
            self.add_middleware(MiddlewareType.REQUEST_VALIDATION, self._validation_middleware)
            self.add_middleware(MiddlewareType.RATE_LIMITING, self._rate_limiting_middleware)
            self.add_middleware(MiddlewareType.CIRCUIT_BREAKING, self._circuit_breaking_middleware)
            self.add_middleware(MiddlewareType.PERFORMANCE_MONITORING, self._performance_middleware)
            self.add_middleware(MiddlewareType.CACHING, self._caching_middleware)
            self.add_middleware(MiddlewareType.ERROR_HANDLING, self._error_handling_middleware)
            
            # Initialize request validators
            self._initialize_validators()
            
            # Initialize performance monitors
            self._initialize_performance_monitors()
            
            self.logger.info("Middleware initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize middleware: {e}")
            raise RuntimeError(f"Middleware initialization failed: {e}")
    
    def _initialize_validators(self) -> None:
        """Initialize request validators"""
        # Add Sheaf-specific validators
        self.add_request_validator(self._validate_sheaf_data_format)
        self.add_request_validator(self._validate_sheaf_parameters)
        self.add_request_validator(self._validate_resource_limits)
        self.add_request_validator(self._validate_security_constraints)
    
    def _initialize_performance_monitors(self) -> None:
        """Initialize performance monitoring"""
        # Add performance monitors
        self.add_performance_monitor(self._monitor_request_latency)
        self.add_performance_monitor(self._monitor_memory_usage)
        self.add_performance_monitor(self._monitor_gpu_utilization)
        self.add_performance_monitor(self._monitor_cache_effectiveness)
    
    def _create_rate_limiter(self) -> Dict[str, Any]:
        """Create rate limiting configuration"""
        return {
            'requests_per_second': self.config.get('rate_limit_rps', 100),
            'burst_size': self.config.get('rate_limit_burst', 150),
            'window_size': self.config.get('rate_limit_window', 60),
            'client_limits': defaultdict(lambda: deque(maxlen=1000)),
            'global_limit': deque(maxlen=10000),
            'last_cleanup': time.time()
        }
    
    def _create_circuit_breaker(self) -> Dict[str, Any]:
        """Create circuit breaker configuration"""
        return {
            'failure_threshold': self.config.get('circuit_breaker_threshold', 5),
            'recovery_timeout': self.config.get('circuit_breaker_timeout', 60),
            'half_open_max_calls': self.config.get('circuit_breaker_half_open', 3),
            'states': defaultdict(lambda: 'CLOSED'),
            'failure_counts': defaultdict(int),
            'last_failure_times': defaultdict(float),
            'half_open_calls': defaultdict(int)
        }
    
    def add_middleware(self, middleware_type: MiddlewareType, handler: Callable) -> None:
        """Add middleware to the processing chain"""
        if not callable(handler):
            raise ValueError(f"Middleware handler must be callable: {handler}")
        
        with self._lock:
            self.middleware_chain.append((middleware_type, handler))
            self.logger.info(f"Added middleware: {middleware_type.value}")
    
    def add_request_validator(self, validator: Callable) -> None:
        """Add request validator"""
        if not callable(validator):
            raise ValueError(f"Validator must be callable: {validator}")
        
        with self._lock:
            self.request_validators.append(validator)
    
    def add_response_transformer(self, transformer: Callable) -> None:
        """Add response transformer"""
        if not callable(transformer):
            raise ValueError(f"Transformer must be callable: {transformer}")
        
        with self._lock:
            self.response_transformers.append(transformer)
    
    def add_performance_monitor(self, monitor: Callable) -> None:
        """Add performance monitor"""
        if not callable(monitor):
            raise ValueError(f"Monitor must be callable: {monitor}")
        
        with self._lock:
            self.performance_monitors.append(monitor)
    
    def add_error_handler(self, error_type: type, handler: Callable) -> None:
        """Add error handler for specific error type"""
        if not isinstance(error_type, type):
            raise ValueError(f"Error type must be a type: {error_type}")
        if not callable(handler):
            raise ValueError(f"Handler must be callable: {handler}")
        
        with self._lock:
            self.error_handlers[error_type] = handler
    
    def process_request(self, request: ServiceRequest, next_handler: Callable) -> ServiceResponse:
        """Process request through middleware chain"""
        context = MiddlewareContext(request=request)
        
        try:
            # Process through middleware chain
            for middleware_type, handler in self.middleware_chain:
                if context.skip_remaining:
                    break
                
                context.middleware_chain.append(middleware_type.value)
                context = handler(context)
                
                # Check if response was set early
                if context.response is not None:
                    break
            
            # If no response yet, call the next handler
            if context.response is None:
                context.response = next_handler(context.request)
            
            # Apply response transformers
            for transformer in self.response_transformers:
                context.response = transformer(context.response, context)
            
            return context.response
            
        except Exception as e:
            self.logger.error(f"Middleware processing failed: {e}")
            return self._handle_middleware_error(e, context)
    
    def _logging_middleware(self, context: MiddlewareContext) -> MiddlewareContext:
        """Log request details"""
        try:
            self.logger.info(f"Processing request: {context.request.request_id}")
            self.logger.debug(f"Request details: operation={context.request.operation_type}")
            context.metadata['logged'] = True
        except Exception as e:
            context.errors.append(f"Logging failed: {e}")
        
        return context
    
    def _validation_middleware(self, context: MiddlewareContext) -> MiddlewareContext:
        """Validate request using registered validators"""
        try:
            for validator in self.request_validators:
                is_valid, error = validator(context.request)
                if not is_valid:
                    context.errors.append(error)
            
            if context.errors:
                context.response = ServiceResponse(
                    request_id=context.request.request_id,
                    status=ServiceStatus.FAILED,
                    error_message=f"Validation failed: {', '.join(context.errors)}"
                )
                context.skip_remaining = True
        except Exception as e:
            self.logger.error(f"Validation middleware failed: {e}")
            context.errors.append(f"Validation error: {e}")
        
        return context
    
    def _rate_limiting_middleware(self, context: MiddlewareContext) -> MiddlewareContext:
        """Apply rate limiting"""
        try:
            client_id = context.request.metadata.get('client_id', 'default')
            current_time = time.time()
            
            # Clean up old entries periodically
            if current_time - self.rate_limiter['last_cleanup'] > 60:
                self._cleanup_rate_limiter()
            
            # Check client-specific limit
            client_times = self.rate_limiter['client_limits'][client_id]
            recent_requests = sum(1 for t in client_times if current_time - t < self.rate_limiter['window_size'])
            
            if recent_requests >= self.rate_limiter['requests_per_second']:
                context.response = ServiceResponse(
                    request_id=context.request.request_id,
                    status=ServiceStatus.FAILED,
                    error_message="Rate limit exceeded"
                )
                context.skip_remaining = True
            else:
                client_times.append(current_time)
                self.rate_limiter['global_limit'].append(current_time)
        except Exception as e:
            self.logger.error(f"Rate limiting failed: {e}")
            # Don't block on rate limiting errors
        
        return context
    
    def _circuit_breaking_middleware(self, context: MiddlewareContext) -> MiddlewareContext:
        """Apply circuit breaking pattern"""
        try:
            service_key = f"sheaf_service:{context.request.operation_type}"
            state = self.circuit_breaker['states'][service_key]
            current_time = time.time()
            
            if state == 'OPEN':
                # Check if recovery timeout has passed
                last_failure = self.circuit_breaker['last_failure_times'][service_key]
                if current_time - last_failure > self.circuit_breaker['recovery_timeout']:
                    self.circuit_breaker['states'][service_key] = 'HALF_OPEN'
                    self.circuit_breaker['half_open_calls'][service_key] = 0
                else:
                    context.response = ServiceResponse(
                        request_id=context.request.request_id,
                        status=ServiceStatus.FAILED,
                        error_message="Service circuit breaker is open"
                    )
                    context.skip_remaining = True
            
            elif state == 'HALF_OPEN':
                # Allow limited calls in half-open state
                if self.circuit_breaker['half_open_calls'][service_key] >= self.circuit_breaker['half_open_max_calls']:
                    context.response = ServiceResponse(
                        request_id=context.request.request_id,
                        status=ServiceStatus.FAILED,
                        error_message="Service circuit breaker is in half-open state"
                    )
                    context.skip_remaining = True
                else:
                    self.circuit_breaker['half_open_calls'][service_key] += 1
        except Exception as e:
            self.logger.error(f"Circuit breaking failed: {e}")
            # Don't block on circuit breaker errors
        
        return context
    
    def _performance_middleware(self, context: MiddlewareContext) -> MiddlewareContext:
        """Monitor performance metrics"""
        try:
            # Run performance monitors
            for monitor in self.performance_monitors:
                monitor(context)
            
            # Record basic metrics
            context.metadata['performance_monitored'] = True
        except Exception as e:
            self.logger.error(f"Performance monitoring failed: {e}")
            context.errors.append(f"Performance monitoring error: {e}")
        
        return context
    
    def _caching_middleware(self, context: MiddlewareContext) -> MiddlewareContext:
        """Check cache for existing results"""
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(context.request)
            context.cache_key = cache_key
            context.metadata['cache_key'] = cache_key
            
            # Cache checking is handled by SheafServiceCache
            context.metadata['cache_checked'] = True
        except Exception as e:
            self.logger.error(f"Caching middleware failed: {e}")
            # Don't block on caching errors
        
        return context
    
    def _error_handling_middleware(self, context: MiddlewareContext) -> MiddlewareContext:
        """Handle errors in the request"""
        try:
            if context.errors:
                self.logger.warning(f"Request {context.request.request_id} has errors: {context.errors}")
                
                # Update circuit breaker if needed
                if context.response and context.response.status in [ServiceStatus.FAILED, ServiceStatus.ERROR]:
                    self._update_circuit_breaker(context)
        except Exception as e:
            self.logger.error(f"Error handling middleware failed: {e}")
        
        return context
    
    def _validate_sheaf_data_format(self, request: ServiceRequest) -> Tuple[bool, Optional[str]]:
        """Validate Sheaf-specific data format"""
        try:
            data = request.data
            if data is None:
                return False, "Data cannot be None"
            
            if not isinstance(data, (torch.Tensor, np.ndarray, dict)):
                return False, f"Invalid data type: {type(data)}"
            
            if isinstance(data, torch.Tensor):
                if data.numel() == 0:
                    return False, "Empty tensor provided"
                if not data.is_contiguous():
                    return False, "Tensor must be contiguous"
            
            return True, None
        except Exception as e:
            return False, f"Data format validation failed: {e}"
    
    def _validate_sheaf_parameters(self, request: ServiceRequest) -> Tuple[bool, Optional[str]]:
        """Validate Sheaf compression parameters"""
        try:
            params = request.metadata.get('parameters', {})
            
            # Validate cohomology dimension
            if 'cohomology_dim' in params:
                dim = params['cohomology_dim']
                if not isinstance(dim, int) or dim < 0 or dim > 10:
                    return False, f"Invalid cohomology dimension: {dim}"
            
            # Validate compression ratio
            if 'target_ratio' in params:
                ratio = params['target_ratio']
                if not isinstance(ratio, (int, float)) or ratio <= 0 or ratio > 1:
                    return False, f"Invalid compression ratio: {ratio}"
            
            # Validate sheaf structure
            if 'sheaf_structure' in params:
                structure = params['sheaf_structure']
                if not isinstance(structure, dict):
                    return False, "Sheaf structure must be a dictionary"
            
            return True, None
        except Exception as e:
            return False, f"Parameter validation failed: {e}"
    
    def _validate_resource_limits(self, request: ServiceRequest) -> Tuple[bool, Optional[str]]:
        """Validate resource usage limits"""
        try:
            data = request.data
            if isinstance(data, torch.Tensor):
                # Check memory limits
                memory_bytes = data.element_size() * data.numel()
                max_memory = self.config.get('max_memory_bytes', 1e9)  # 1GB default
                
                if memory_bytes > max_memory:
                    return False, f"Data size {memory_bytes} exceeds limit {max_memory}"
            
            # Check dimension limits
            if hasattr(data, 'shape'):
                max_dim = self.config.get('max_dimensions', 10000)
                if any(d > max_dim for d in data.shape):
                    return False, f"Data dimensions exceed limit {max_dim}"
            
            return True, None
        except Exception as e:
            return False, f"Resource validation failed: {e}"
    
    def _validate_security_constraints(self, request: ServiceRequest) -> Tuple[bool, Optional[str]]:
        """Validate security constraints"""
        try:
            # Check authentication
            if self.config.get('require_auth', False):
                auth_token = request.metadata.get('auth_token')
                if not auth_token:
                    return False, "Authentication required"
                
                # Validate token (simplified for this implementation)
                if not self._validate_auth_token(auth_token):
                    return False, "Invalid authentication token"
            
            # Check allowed operations
            allowed_ops = self.config.get('allowed_operations', [])
            if allowed_ops and request.operation_type not in allowed_ops:
                return False, f"Operation {request.operation_type} not allowed"
            
            return True, None
        except Exception as e:
            return False, f"Security validation failed: {e}"
    
    def _validate_auth_token(self, token: str) -> bool:
        """Validate authentication token"""
        try:
            # Simplified token validation
            expected_prefix = self.config.get('token_prefix', 'Bearer ')
            return token.startswith(expected_prefix) and len(token) > len(expected_prefix)
        except Exception:
            return False
    
    def _monitor_request_latency(self, context: MiddlewareContext) -> None:
        """Monitor request latency"""
        try:
            current_time = time.time()
            latency = current_time - context.start_time
            
            # Record latency
            service_key = f"sheaf_service:{context.request.operation_type}"
            if hasattr(self.metrics, 'response_times'):
                if not hasattr(self.metrics.response_times, service_key):
                    self.metrics.response_times[service_key] = []
                self.metrics.response_times[service_key].append(latency)
            
            # Alert on high latency
            latency_threshold = self.config.get('latency_threshold', 1.0)
            if latency > latency_threshold:
                self.logger.warning(f"High latency detected: {latency:.3f}s for {service_key}")
        except Exception as e:
            self.logger.error(f"Latency monitoring failed: {e}")
    
    def _monitor_memory_usage(self, context: MiddlewareContext) -> None:
        """Monitor memory usage"""
        try:
            if hasattr(context.request.data, 'element_size'):
                memory_bytes = context.request.data.element_size() * context.request.data.numel()
                context.metadata['memory_usage'] = memory_bytes
                
                # Track memory metrics
                self.metrics.record_metric('memory_usage', memory_bytes)
        except Exception as e:
            self.logger.error(f"Memory monitoring failed: {e}")
    
    def _monitor_gpu_utilization(self, context: MiddlewareContext) -> None:
        """Monitor GPU utilization"""
        try:
            if torch.cuda.is_available():
                # Get GPU memory stats
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                
                context.metadata['gpu_allocated'] = allocated
                context.metadata['gpu_reserved'] = reserved
                
                # Track GPU metrics
                self.metrics.record_metric('gpu_allocated', allocated)
                self.metrics.record_metric('gpu_reserved', reserved)
        except Exception as e:
            self.logger.error(f"GPU monitoring failed: {e}")
    
    def _monitor_cache_effectiveness(self, context: MiddlewareContext) -> None:
        """Monitor cache hit rates"""
        try:
            cache_hit = context.metadata.get('cache_hit', False)
            cache_key = context.metadata.get('cache_key')
            
            if cache_key:
                self.metrics.record_metric('cache_requests', 1)
                if cache_hit:
                    self.metrics.record_metric('cache_hits', 1)
        except Exception as e:
            self.logger.error(f"Cache monitoring failed: {e}")
    
    def _generate_cache_key(self, request: ServiceRequest) -> str:
        """Generate cache key for request"""
        try:
            # Create key components
            key_parts = [
                "sheaf_service",
                request.operation_type.value if hasattr(request.operation_type, 'value') else str(request.operation_type),
                "1.0.0"
            ]
            
            # Add data hash if available
            if hasattr(request.data, 'numpy'):
                data_hash = hashlib.sha256(request.data.numpy().tobytes()).hexdigest()[:16]
                key_parts.append(data_hash)
            elif isinstance(request.data, np.ndarray):
                data_hash = hashlib.sha256(request.data.tobytes()).hexdigest()[:16]
                key_parts.append(data_hash)
            
            # Add metadata hash if available  
            if request.metadata:
                param_str = json.dumps(request.metadata, sort_keys=True)
                param_hash = hashlib.sha256(param_str.encode()).hexdigest()[:16]
                key_parts.append(param_hash)
            
            return ':'.join(key_parts)
        except Exception as e:
            self.logger.error(f"Cache key generation failed: {e}")
            return f"error:{uuid.uuid4()}"
    
    def _cleanup_rate_limiter(self) -> None:
        """Clean up old rate limiter entries"""
        try:
            current_time = time.time()
            window_size = self.rate_limiter['window_size']
            
            # Clean up client limits
            for client_id, times in list(self.rate_limiter['client_limits'].items()):
                # Remove old entries
                self.rate_limiter['client_limits'][client_id] = deque(
                    (t for t in times if current_time - t < window_size),
                    maxlen=times.maxlen
                )
                
                # Remove empty clients
                if not self.rate_limiter['client_limits'][client_id]:
                    del self.rate_limiter['client_limits'][client_id]
            
            self.rate_limiter['last_cleanup'] = current_time
        except Exception as e:
            self.logger.error(f"Rate limiter cleanup failed: {e}")
    
    def _update_circuit_breaker(self, context: MiddlewareContext) -> None:
        """Update circuit breaker state based on response"""
        try:
            service_key = f"sheaf_service:{context.request.operation_type}"
            current_time = time.time()
            
            if context.response.status == ServiceStatus.FAILED:
                # Increment failure count
                self.circuit_breaker['failure_counts'][service_key] += 1
                self.circuit_breaker['last_failure_times'][service_key] = current_time
                
                # Check if threshold exceeded
                if self.circuit_breaker['failure_counts'][service_key] >= self.circuit_breaker['failure_threshold']:
                    self.circuit_breaker['states'][service_key] = 'OPEN'
                    self.logger.warning(f"Circuit breaker opened for {service_key}")
            else:
                # Success in half-open state
                if self.circuit_breaker['states'][service_key] == 'HALF_OPEN':
                    self.circuit_breaker['states'][service_key] = 'CLOSED'
                    self.circuit_breaker['failure_counts'][service_key] = 0
                    self.logger.info(f"Circuit breaker closed for {service_key}")
        except Exception as e:
            self.logger.error(f"Circuit breaker update failed: {e}")
    
    def _handle_middleware_error(self, error: Exception, context: MiddlewareContext) -> ServiceResponse:
        """Handle middleware processing errors"""
        error_type = type(error)
        
        # Check for specific error handler
        if error_type in self.error_handlers:
            try:
                return self.error_handlers[error_type](error, context)
            except Exception as e:
                self.logger.error(f"Error handler failed: {e}")
        
        # Default error response
        return ServiceResponse(
            request_id=context.request.request_id,
            status=ServiceStatus.FAILED,
            error_message=f"Middleware error: {str(error)}"
        )
    
    def get_middleware_stats(self) -> Dict[str, Any]:
        """Get middleware statistics"""
        try:
            return {
                'middleware_count': len(self.middleware_chain),
                'validator_count': len(self.request_validators),
                'transformer_count': len(self.response_transformers),
                'monitor_count': len(self.performance_monitors),
                'error_handler_count': len(self.error_handlers),
                'rate_limiter_stats': {
                    'active_clients': len(self.rate_limiter['client_limits']),
                    'global_requests': len(self.rate_limiter['global_limit'])
                },
                'circuit_breaker_stats': {
                    'open_circuits': sum(1 for s in self.circuit_breaker['states'].values() if s == 'OPEN'),
                    'half_open_circuits': sum(1 for s in self.circuit_breaker['states'].values() if s == 'HALF_OPEN')
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to get middleware stats: {e}")
            return {}


class ConfigurationSource(Enum):
    """Configuration source types"""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    REMOTE = "remote"
    DEFAULT = "default"


@dataclass
class ConfigurationVersion:
    """Configuration version information"""
    version: str
    timestamp: datetime
    source: ConfigurationSource
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SheafServiceConfiguration:
    """
    Configuration management for Sheaf service layer.
    
    Provides configuration loading, validation, hot-reloading,
    persistence, and encryption capabilities.
    """
    
    def __init__(self, config_path: Optional[Path] = None, encryption_key: Optional[str] = None):
        """Initialize configuration manager"""
        self.config_path = config_path or Path("sheaf_config.json")
        self.encryption_key = encryption_key
        self.current_config: Dict[str, Any] = {}
        self.config_versions: List[ConfigurationVersion] = []
        self.config_validators: List[Callable] = []
        self.config_transformers: List[Callable] = []
        self.hot_reload_enabled = False
        self.reload_interval = 60  # seconds
        self.reload_thread: Optional[threading.Thread] = None
        self.cipher_suite: Optional[Fernet] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._lock = threading.RLock()
        self._initialize_configuration()
    
    def _initialize_configuration(self) -> None:
        """Initialize configuration system"""
        try:
            # Initialize encryption if key provided
            if self.encryption_key:
                self.cipher_suite = Fernet(self.encryption_key.encode())
            
            # Load initial configuration
            self.load_configuration()
            
            # Initialize validators
            self._initialize_validators()
            
            # Initialize transformers
            self._initialize_transformers()
            
            self.logger.info("Configuration system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize configuration: {e}")
            raise RuntimeError(f"Configuration initialization failed: {e}")
    
    def _initialize_validators(self) -> None:
        """Initialize configuration validators"""
        self.add_validator(self._validate_required_fields)
        self.add_validator(self._validate_data_types)
        self.add_validator(self._validate_value_ranges)
        self.add_validator(self._validate_dependencies)
    
    def _initialize_transformers(self) -> None:
        """Initialize configuration transformers"""
        self.add_transformer(self._transform_environment_variables)
        self.add_transformer(self._transform_relative_paths)
        self.add_transformer(self._transform_computed_values)
    
    def load_configuration(self, source: Optional[ConfigurationSource] = None) -> Dict[str, Any]:
        """Load configuration from specified source"""
        source = source or ConfigurationSource.FILE
        
        try:
            with self._lock:
                if source == ConfigurationSource.FILE:
                    config = self._load_from_file()
                elif source == ConfigurationSource.ENVIRONMENT:
                    config = self._load_from_environment()
                elif source == ConfigurationSource.DATABASE:
                    config = self._load_from_database()
                elif source == ConfigurationSource.REMOTE:
                    config = self._load_from_remote()
                else:
                    config = self._load_defaults()
                
                # Validate configuration
                self._validate_configuration(config)
                
                # Transform configuration
                config = self._transform_configuration(config)
                
                # Update current configuration
                self.current_config = config
                
                # Record version
                self._record_version(config, source)
                
                self.logger.info(f"Configuration loaded from {source.value}")
                return config
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise RuntimeError(f"Configuration loading failed: {e}")
    
    def _load_from_file(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.config_path.exists():
            self.logger.warning(f"Configuration file not found: {self.config_path}")
            return self._load_defaults()
        
        try:
            with open(self.config_path, 'r') as f:
                content = f.read()
                
                # Decrypt if needed
                if self.cipher_suite and content.startswith('gAAAAA'):
                    content = self.cipher_suite.decrypt(content.encode()).decode()
                
                return json.loads(content)
        except Exception as e:
            self.logger.error(f"Failed to load configuration file: {e}")
            raise
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config = {}
        
        # Define environment variable mappings
        env_mappings = {
            'SHEAF_MAX_WORKERS': ('max_workers', int),
            'SHEAF_RATE_LIMIT': ('rate_limit_rps', int),
            'SHEAF_CACHE_SIZE': ('cache_size_mb', int),
            'SHEAF_GPU_ENABLED': ('gpu_enabled', lambda x: x.lower() == 'true'),
            'SHEAF_LOG_LEVEL': ('log_level', str),
            'SHEAF_COHOMOLOGY_DIM': ('default_cohomology_dim', int),
            'SHEAF_COMPRESSION_RATIO': ('default_compression_ratio', float)
        }
        
        import os
        for env_var, (config_key, converter) in env_mappings.items():
            if env_var in os.environ:
                try:
                    config[config_key] = converter(os.environ[env_var])
                except Exception as e:
                    self.logger.warning(f"Failed to parse {env_var}: {e}")
        
        return config
    
    def _load_from_database(self) -> Dict[str, Any]:
        """Load configuration from database"""
        # This would connect to a database and load configuration
        # For now, return defaults
        self.logger.warning("Database configuration loading not implemented")
        return self._load_defaults()
    
    def _load_from_remote(self) -> Dict[str, Any]:
        """Load configuration from remote service"""
        # This would fetch configuration from a remote service
        # For now, return defaults
        self.logger.warning("Remote configuration loading not implemented")
        return self._load_defaults()
    
    def _load_defaults(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'service': {
                'name': 'SheafCompressionService',
                'version': '1.0.0',
                'description': 'Cellular Sheaf-based compression service'
            },
            'performance': {
                'max_workers': 10,
                'thread_pool_size': 20,
                'async_enabled': True,
                'batch_size': 32
            },
            'limits': {
                'max_memory_bytes': 1e9,  # 1GB
                'max_dimensions': 10000,
                'max_request_size': 100e6,  # 100MB
                'rate_limit_rps': 100,
                'rate_limit_burst': 150
            },
            'sheaf': {
                'default_cohomology_dim': 3,
                'default_compression_ratio': 0.5,
                'use_gpu': True,
                'cache_enabled': True,
                'cache_size_mb': 1024
            },
            'security': {
                'require_auth': False,
                'token_prefix': 'Bearer ',
                'allowed_operations': []
            },
            'monitoring': {
                'metrics_enabled': True,
                'health_check_interval': 30,
                'log_level': 'INFO',
                'alert_thresholds': {
                    'latency_ms': 1000,
                    'error_rate': 0.05,
                    'memory_usage': 0.9
                }
            }
        }
    
    def save_configuration(self, config: Optional[Dict[str, Any]] = None, encrypt: bool = True) -> None:
        """Save configuration to file"""
        config = config or self.current_config
        
        try:
            with self._lock:
                # Convert to JSON
                content = json.dumps(config, indent=2)
                
                # Encrypt if requested and key available
                if encrypt and self.cipher_suite:
                    content = self.cipher_suite.encrypt(content.encode()).decode()
                
                # Write to file
                with open(self.config_path, 'w') as f:
                    f.write(content)
                
                self.logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise RuntimeError(f"Configuration save failed: {e}")
    
    def update_configuration(self, updates: Dict[str, Any], validate: bool = True) -> Dict[str, Any]:
        """Update configuration with new values"""
        try:
            with self._lock:
                # Deep merge updates into current config
                new_config = self._deep_merge(self.current_config.copy(), updates)
                
                # Validate if requested
                if validate:
                    self._validate_configuration(new_config)
                
                # Transform configuration
                new_config = self._transform_configuration(new_config)
                
                # Update current configuration
                old_config = self.current_config
                self.current_config = new_config
                
                # Record version
                self._record_version(new_config, ConfigurationSource.DEFAULT)
                
                # Notify listeners
                self._notify_configuration_change(old_config, new_config)
                
                self.logger.info("Configuration updated")
                return new_config
                
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            raise RuntimeError(f"Configuration update failed: {e}")
    
    def _deep_merge(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    def add_validator(self, validator: Callable) -> None:
        """Add configuration validator"""
        if not callable(validator):
            raise ValueError(f"Validator must be callable: {validator}")
        
        with self._lock:
            self.config_validators.append(validator)
    
    def add_transformer(self, transformer: Callable) -> None:
        """Add configuration transformer"""
        if not callable(transformer):
            raise ValueError(f"Transformer must be callable: {transformer}")
        
        with self._lock:
            self.config_transformers.append(transformer)
    
    def _validate_configuration(self, config: Dict[str, Any]) -> None:
        """Validate configuration using registered validators"""
        errors = []
        
        for validator in self.config_validators:
            try:
                is_valid, error = validator(config)
                if not is_valid:
                    errors.append(error)
            except Exception as e:
                errors.append(f"Validator error: {e}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
    
    def _transform_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform configuration using registered transformers"""
        for transformer in self.config_transformers:
            try:
                config = transformer(config)
            except Exception as e:
                self.logger.warning(f"Transformer failed: {e}")
        
        return config
    
    def _validate_required_fields(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate required configuration fields"""
        required_fields = [
            'service.name',
            'service.version',
            'performance.max_workers',
            'limits.max_memory_bytes',
            'sheaf.default_cohomology_dim'
        ]
        
        for field_path in required_fields:
            value = self._get_nested_value(config, field_path)
            if value is None:
                return False, f"Required field missing: {field_path}"
        
        return True, None
    
    def _validate_data_types(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate configuration data types"""
        type_constraints = {
            'performance.max_workers': int,
            'performance.batch_size': int,
            'limits.max_memory_bytes': (int, float),
            'limits.rate_limit_rps': int,
            'sheaf.default_compression_ratio': (int, float),
            'sheaf.use_gpu': bool,
            'security.require_auth': bool
        }
        
        for field_path, expected_type in type_constraints.items():
            value = self._get_nested_value(config, field_path)
            if value is not None and not isinstance(value, expected_type):
                return False, f"Invalid type for {field_path}: expected {expected_type}, got {type(value)}"
        
        return True, None
    
    def _validate_value_ranges(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate configuration value ranges"""
        range_constraints = {
            'performance.max_workers': (1, 1000),
            'performance.batch_size': (1, 10000),
            'limits.rate_limit_rps': (1, 10000),
            'sheaf.default_cohomology_dim': (0, 10),
            'sheaf.default_compression_ratio': (0.001, 1.0),
            'monitoring.health_check_interval': (1, 3600)
        }
        
        for field_path, (min_val, max_val) in range_constraints.items():
            value = self._get_nested_value(config, field_path)
            if value is not None and not (min_val <= value <= max_val):
                return False, f"Value out of range for {field_path}: {value} not in [{min_val}, {max_val}]"
        
        return True, None
    
    def _validate_dependencies(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate configuration dependencies"""
        # Example: GPU-related settings require GPU to be enabled
        use_gpu = self._get_nested_value(config, 'sheaf.use_gpu')
        if not use_gpu:
            gpu_settings = ['sheaf.gpu_memory_fraction', 'sheaf.gpu_device_id']
            for setting in gpu_settings:
                if self._get_nested_value(config, setting) is not None:
                    return False, f"{setting} requires sheaf.use_gpu to be enabled"
        
        return True, None
    
    def _transform_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform environment variable placeholders"""
        import os
        
        def replace_env_vars(obj):
            if isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                env_var = obj[2:-1]
                return os.environ.get(env_var, obj)
            elif isinstance(obj, dict):
                return {k: replace_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_env_vars(item) for item in obj]
            return obj
        
        return replace_env_vars(config)
    
    def _transform_relative_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform relative paths to absolute"""
        path_fields = ['cache.directory', 'logs.directory', 'data.directory']
        
        for field_path in path_fields:
            value = self._get_nested_value(config, field_path)
            if value and isinstance(value, str) and not Path(value).is_absolute():
                absolute_path = Path(value).absolute()
                self._set_nested_value(config, field_path, str(absolute_path))
        
        return config
    
    def _transform_computed_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform computed configuration values"""
        # Example: Calculate cache size based on available memory
        if 'sheaf.cache_size_mb' not in config.get('sheaf', {}):
            max_memory = config.get('limits', {}).get('max_memory_bytes', 1e9)
            cache_size_mb = int(max_memory * 0.1 / 1e6)  # 10% of max memory
            self._set_nested_value(config, 'sheaf.cache_size_mb', cache_size_mb)
        
        return config
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get nested configuration value by path"""
        keys = path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set nested configuration value by path"""
        keys = path.split('.')
        target = config
        
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        target[keys[-1]] = value
    
    def _record_version(self, config: Dict[str, Any], source: ConfigurationSource) -> None:
        """Record configuration version"""
        try:
            # Calculate checksum
            config_str = json.dumps(config, sort_keys=True)
            checksum = hashlib.sha256(config_str.encode()).hexdigest()
            
            # Create version record
            version = ConfigurationVersion(
                version=f"{len(self.config_versions) + 1}.0.0",
                timestamp=datetime.now(),
                source=source,
                checksum=checksum,
                metadata={'size': len(config_str)}
            )
            
            self.config_versions.append(version)
            
            # Keep only recent versions
            max_versions = 100
            if len(self.config_versions) > max_versions:
                self.config_versions = self.config_versions[-max_versions:]
        except Exception as e:
            self.logger.error(f"Failed to record version: {e}")
    
    def _notify_configuration_change(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        """Notify listeners of configuration changes"""
        try:
            # Calculate changes
            changes = self._calculate_changes(old_config, new_config)
            
            if changes:
                self.logger.info(f"Configuration changed: {changes}")
                
                # Here you would notify registered listeners
                # For now, just log the changes
        except Exception as e:
            self.logger.error(f"Failed to notify configuration change: {e}")
    
    def _calculate_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> List[str]:
        """Calculate differences between configurations"""
        changes = []
        
        def compare_dicts(old, new, path=''):
            for key in set(old.keys()) | set(new.keys()):
                current_path = f"{path}.{key}" if path else key
                
                if key not in old:
                    changes.append(f"Added: {current_path}")
                elif key not in new:
                    changes.append(f"Removed: {current_path}")
                elif old[key] != new[key]:
                    if isinstance(old[key], dict) and isinstance(new[key], dict):
                        compare_dicts(old[key], new[key], current_path)
                    else:
                        changes.append(f"Modified: {current_path}")
        
        compare_dicts(old_config, new_config)
        return changes
    
    def enable_hot_reload(self, interval: int = 60) -> None:
        """Enable configuration hot-reloading"""
        if self.hot_reload_enabled:
            self.logger.warning("Hot reload already enabled")
            return
        
        self.reload_interval = interval
        self.hot_reload_enabled = True
        
        # Start reload thread
        self.reload_thread = threading.Thread(target=self._hot_reload_loop, daemon=True)
        self.reload_thread.start()
        
        self.logger.info(f"Hot reload enabled with interval {interval}s")
    
    def disable_hot_reload(self) -> None:
        """Disable configuration hot-reloading"""
        self.hot_reload_enabled = False
        
        if self.reload_thread:
            self.reload_thread.join(timeout=5)
            self.reload_thread = None
        
        self.logger.info("Hot reload disabled")
    
    def _hot_reload_loop(self) -> None:
        """Hot reload loop"""
        last_checksum = None
        
        while self.hot_reload_enabled:
            try:
                # Check if file has changed
                if self.config_path.exists():
                    with open(self.config_path, 'rb') as f:
                        content = f.read()
                        checksum = hashlib.sha256(content).hexdigest()
                    
                    if checksum != last_checksum:
                        self.logger.info("Configuration file changed, reloading...")
                        self.load_configuration()
                        last_checksum = checksum
                
                # Sleep for interval
                time.sleep(self.reload_interval)
                
            except Exception as e:
                self.logger.error(f"Hot reload error: {e}")
                time.sleep(self.reload_interval)
    
    def get_configuration(self, path: Optional[str] = None) -> Any:
        """Get configuration value by path"""
        with self._lock:
            if path:
                return self._get_nested_value(self.current_config, path)
            return self.current_config.copy()
    
    def get_configuration_history(self) -> List[ConfigurationVersion]:
        """Get configuration version history"""
        with self._lock:
            return self.config_versions.copy()


class HealthStatus(Enum):
    """Health check status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Health check result"""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)
    checks: Dict[str, bool] = field(default_factory=dict)


class SheafServiceHealthMonitor:
    """
    Health monitoring for Sheaf service layer.
    
    Provides health checks, metrics collection, alerting,
    and reporting capabilities.
    """
    
    def __init__(self, service: SheafServiceInterface, config: Optional[Dict[str, Any]] = None):
        """Initialize health monitor"""
        self.service = service
        self.config = config or {}
        self.health_checks: Dict[str, Callable] = {}
        self.health_results: Dict[str, HealthCheckResult] = {}
        self.health_history: deque = deque(maxlen=1000)
        self.metrics_collectors: List[Callable] = []
        self.alert_handlers: List[Callable] = []
        self.monitor_thread: Optional[threading.Thread] = None
        self.is_monitoring = False
        self.check_interval = self.config.get('health_check_interval', 30)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._lock = threading.RLock()
        self._initialize_health_monitor()
    
    def _initialize_health_monitor(self) -> None:
        """Initialize health monitoring system"""
        try:
            # Register default health checks
            self.register_health_check('service_status', self._check_service_status)
            self.register_health_check('memory_usage', self._check_memory_usage)
            self.register_health_check('gpu_status', self._check_gpu_status)
            self.register_health_check('cache_health', self._check_cache_health)
            self.register_health_check('compression_health', self._check_compression_health)
            self.register_health_check('thread_pool', self._check_thread_pool)
            self.register_health_check('error_rate', self._check_error_rate)
            self.register_health_check('latency', self._check_latency)
            
            # Register metrics collectors
            self._initialize_metrics_collectors()
            
            # Register alert handlers
            self._initialize_alert_handlers()
            
            self.logger.info("Health monitor initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize health monitor: {e}")
            raise RuntimeError(f"Health monitor initialization failed: {e}")
    
    def _initialize_metrics_collectors(self) -> None:
        """Initialize metrics collectors"""
        self.add_metrics_collector(self._collect_service_metrics)
        self.add_metrics_collector(self._collect_compression_metrics)
        self.add_metrics_collector(self._collect_resource_metrics)
        self.add_metrics_collector(self._collect_performance_metrics)
    
    def _initialize_alert_handlers(self) -> None:
        """Initialize alert handlers"""
        self.add_alert_handler(self._log_alert)
        if self.config.get('email_alerts_enabled'):
            self.add_alert_handler(self._email_alert)
        if self.config.get('webhook_alerts_enabled'):
            self.add_alert_handler(self._webhook_alert)
    
    def register_health_check(self, name: str, check_function: Callable) -> None:
        """Register a health check"""
        if not callable(check_function):
            raise ValueError(f"Health check must be callable: {check_function}")
        
        with self._lock:
            self.health_checks[name] = check_function
            self.logger.info(f"Registered health check: {name}")
    
    def add_metrics_collector(self, collector: Callable) -> None:
        """Add metrics collector"""
        if not callable(collector):
            raise ValueError(f"Metrics collector must be callable: {collector}")
        
        with self._lock:
            self.metrics_collectors.append(collector)
    
    def add_alert_handler(self, handler: Callable) -> None:
        """Add alert handler"""
        if not callable(handler):
            raise ValueError(f"Alert handler must be callable: {handler}")
        
        with self._lock:
            self.alert_handlers.append(handler)
    
    def start_monitoring(self) -> None:
        """Start health monitoring"""
        if self.is_monitoring:
            self.logger.warning("Health monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring"""
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            self.monitor_thread = None
        
        self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Run health checks
                self.run_health_checks()
                
                # Collect metrics
                self.collect_metrics()
                
                # Check for alerts
                self.check_alerts()
                
                # Sleep for interval
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.check_interval)
    
    def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}
        
        with self._lock:
            for name, check_function in self.health_checks.items():
                try:
                    result = check_function()
                    results[name] = result
                    self.health_results[name] = result
                    
                    # Add to history
                    self.health_history.append((datetime.now(), name, result))
                    
                except Exception as e:
                    self.logger.error(f"Health check '{name}' failed: {e}")
                    results[name] = HealthCheckResult(
                        component=name,
                        status=HealthStatus.UNKNOWN,
                        message=f"Check failed: {str(e)}",
                        timestamp=datetime.now()
                    )
        
        return results
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect all metrics"""
        metrics = {}
        
        for collector in self.metrics_collectors:
            try:
                collector_metrics = collector()
                metrics.update(collector_metrics)
            except Exception as e:
                self.logger.error(f"Metrics collector failed: {e}")
        
        return metrics
    
    def check_alerts(self) -> None:
        """Check for alert conditions"""
        try:
            # Check health status
            overall_status = self.get_overall_health()
            
            if overall_status == HealthStatus.UNHEALTHY:
                self._trigger_alert('critical', 'Service is unhealthy', self.health_results)
            elif overall_status == HealthStatus.DEGRADED:
                self._trigger_alert('warning', 'Service is degraded', self.health_results)
            
            # Check specific thresholds
            self._check_threshold_alerts()
            
        except Exception as e:
            self.logger.error(f"Alert checking failed: {e}")
    
    def _check_threshold_alerts(self) -> None:
        """Check for threshold-based alerts"""
        thresholds = self.config.get('alert_thresholds', {})
        
        # Check error rate
        if 'error_rate' in thresholds:
            error_rate = self._calculate_error_rate()
            if error_rate > thresholds['error_rate']:
                self._trigger_alert('warning', f'High error rate: {error_rate:.2%}', {'error_rate': error_rate})
        
        # Check latency
        if 'latency_ms' in thresholds:
            avg_latency = self._calculate_average_latency()
            if avg_latency > thresholds['latency_ms']:
                self._trigger_alert('warning', f'High latency: {avg_latency:.0f}ms', {'latency': avg_latency})
        
        # Check memory usage
        if 'memory_usage' in thresholds:
            memory_usage = self._calculate_memory_usage()
            if memory_usage > thresholds['memory_usage']:
                self._trigger_alert('warning', f'High memory usage: {memory_usage:.1%}', {'memory_usage': memory_usage})
    
    def _trigger_alert(self, severity: str, message: str, context: Dict[str, Any]) -> None:
        """Trigger an alert"""
        alert = {
            'severity': severity,
            'message': message,
            'timestamp': datetime.now(),
            'context': context
        }
        
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")
    
    def _check_service_status(self) -> HealthCheckResult:
        """Check service status"""
        try:
            status = self.service.status
            
            if status == ServiceStatus.ACTIVE:
                return HealthCheckResult(
                    component='service_status',
                    status=HealthStatus.HEALTHY,
                    message='Service is active',
                    timestamp=datetime.now(),
                    checks={'service_active': True}
                )
            elif status == ServiceStatus.STARTING:
                return HealthCheckResult(
                    component='service_status',
                    status=HealthStatus.DEGRADED,
                    message='Service is starting',
                    timestamp=datetime.now(),
                    checks={'service_active': False}
                )
            else:
                return HealthCheckResult(
                    component='service_status',
                    status=HealthStatus.UNHEALTHY,
                    message=f'Service status: {status}',
                    timestamp=datetime.now(),
                    checks={'service_active': False}
                )
        except Exception as e:
            return HealthCheckResult(
                component='service_status',
                status=HealthStatus.UNKNOWN,
                message=f'Status check failed: {e}',
                timestamp=datetime.now()
            )
    
    def _check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Define thresholds
            warning_threshold = 70
            critical_threshold = 90
            
            if memory_percent < warning_threshold:
                status = HealthStatus.HEALTHY
                message = f'Memory usage: {memory_percent:.1f}%'
            elif memory_percent < critical_threshold:
                status = HealthStatus.DEGRADED
                message = f'High memory usage: {memory_percent:.1f}%'
            else:
                status = HealthStatus.UNHEALTHY
                message = f'Critical memory usage: {memory_percent:.1f}%'
            
            return HealthCheckResult(
                component='memory_usage',
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics={
                    'memory_bytes': memory_info.rss,
                    'memory_percent': memory_percent
                },
                checks={
                    'memory_ok': memory_percent < warning_threshold
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component='memory_usage',
                status=HealthStatus.UNKNOWN,
                message=f'Memory check failed: {e}',
                timestamp=datetime.now()
            )
    
    def _check_gpu_status(self) -> HealthCheckResult:
        """Check GPU status"""
        try:
            if not torch.cuda.is_available():
                return HealthCheckResult(
                    component='gpu_status',
                    status=HealthStatus.HEALTHY,
                    message='GPU not available/required',
                    timestamp=datetime.now(),
                    checks={'gpu_available': False}
                )
            
            # Check GPU memory
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            
            usage_percent = (allocated / total) * 100
            
            if usage_percent < 70:
                status = HealthStatus.HEALTHY
                message = f'GPU memory usage: {usage_percent:.1f}%'
            elif usage_percent < 90:
                status = HealthStatus.DEGRADED
                message = f'High GPU memory usage: {usage_percent:.1f}%'
            else:
                status = HealthStatus.UNHEALTHY
                message = f'Critical GPU memory usage: {usage_percent:.1f}%'
            
            return HealthCheckResult(
                component='gpu_status',
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics={
                    'gpu_allocated': allocated,
                    'gpu_reserved': reserved,
                    'gpu_total': total,
                    'gpu_usage_percent': usage_percent
                },
                checks={
                    'gpu_available': True,
                    'gpu_memory_ok': usage_percent < 70
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component='gpu_status',
                status=HealthStatus.UNKNOWN,
                message=f'GPU check failed: {e}',
                timestamp=datetime.now()
            )
    
    def _check_cache_health(self) -> HealthCheckResult:
        """Check cache health"""
        try:
            # Get cache stats from service
            cache_stats = getattr(self.service, 'get_cache_stats', lambda: {})()
            
            if not cache_stats:
                return HealthCheckResult(
                    component='cache_health',
                    status=HealthStatus.HEALTHY,
                    message='Cache not enabled',
                    timestamp=datetime.now(),
                    checks={'cache_enabled': False}
                )
            
            hit_rate = cache_stats.get('hit_rate', 0)
            size = cache_stats.get('size', 0)
            max_size = cache_stats.get('max_size', 1)
            
            usage_percent = (size / max_size) * 100 if max_size > 0 else 0
            
            # Check health based on hit rate and usage
            if hit_rate > 0.5 and usage_percent < 90:
                status = HealthStatus.HEALTHY
                message = f'Cache healthy: {hit_rate:.1%} hit rate'
            elif hit_rate > 0.3 or usage_percent < 95:
                status = HealthStatus.DEGRADED
                message = f'Cache degraded: {hit_rate:.1%} hit rate, {usage_percent:.0f}% full'
            else:
                status = HealthStatus.UNHEALTHY
                message = f'Cache unhealthy: {hit_rate:.1%} hit rate, {usage_percent:.0f}% full'
            
            return HealthCheckResult(
                component='cache_health',
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics={
                    'cache_hit_rate': hit_rate,
                    'cache_size': size,
                    'cache_max_size': max_size,
                    'cache_usage_percent': usage_percent
                },
                checks={
                    'cache_enabled': True,
                    'cache_effective': hit_rate > 0.3
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component='cache_health',
                status=HealthStatus.UNKNOWN,
                message=f'Cache check failed: {e}',
                timestamp=datetime.now()
            )
    
    def _check_compression_health(self) -> HealthCheckResult:
        """Check compression system health"""
        try:
            # Get compression stats
            compression_stats = getattr(self.service, 'compression_system', None)
            if not compression_stats:
                return HealthCheckResult(
                    component='compression_health',
                    status=HealthStatus.HEALTHY,
                    message='Compression system not initialized',
                    timestamp=datetime.now()
                )
            
            stats = getattr(compression_stats, 'stats', {})
            
            total_compressions = stats.get('total_compressions', 0)
            total_decompressions = stats.get('total_decompressions', 0)
            avg_ratio = stats.get('avg_compression_ratio', 0)
            avg_error = stats.get('avg_reconstruction_error', 0)
            
            # Define health criteria
            if total_compressions == 0:
                status = HealthStatus.HEALTHY
                message = 'No compressions performed yet'
            elif avg_error < 0.01 and avg_ratio > 0.3:
                status = HealthStatus.HEALTHY
                message = f'Compression healthy: {avg_ratio:.1%} ratio, {avg_error:.3f} error'
            elif avg_error < 0.05 and avg_ratio > 0.2:
                status = HealthStatus.DEGRADED
                message = f'Compression degraded: {avg_ratio:.1%} ratio, {avg_error:.3f} error'
            else:
                status = HealthStatus.UNHEALTHY
                message = f'Compression unhealthy: {avg_ratio:.1%} ratio, {avg_error:.3f} error'
            
            return HealthCheckResult(
                component='compression_health',
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics={
                    'total_compressions': total_compressions,
                    'total_decompressions': total_decompressions,
                    'avg_compression_ratio': avg_ratio,
                    'avg_reconstruction_error': avg_error
                },
                checks={
                    'compression_working': total_compressions > 0 or status == HealthStatus.HEALTHY,
                    'compression_accurate': avg_error < 0.05
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component='compression_health',
                status=HealthStatus.UNKNOWN,
                message=f'Compression check failed: {e}',
                timestamp=datetime.now()
            )
    
    def _check_thread_pool(self) -> HealthCheckResult:
        """Check thread pool health"""
        try:
            # Check if service has executor
            if hasattr(self.service, 'executor'):
                executor = self.service.executor
                # This is a simplified check - real implementation would need more details
                status = HealthStatus.HEALTHY
                message = 'Thread pool healthy'
            else:
                status = HealthStatus.HEALTHY
                message = 'No thread pool configured'
            
            return HealthCheckResult(
                component='thread_pool',
                status=status,
                message=message,
                timestamp=datetime.now(),
                checks={'thread_pool_ok': True}
            )
        except Exception as e:
            return HealthCheckResult(
                component='thread_pool',
                status=HealthStatus.UNKNOWN,
                message=f'Thread pool check failed: {e}',
                timestamp=datetime.now()
            )
    
    def _check_error_rate(self) -> HealthCheckResult:
        """Check error rate"""
        try:
            error_rate = self._calculate_error_rate()
            
            if error_rate < 0.01:
                status = HealthStatus.HEALTHY
                message = f'Low error rate: {error_rate:.2%}'
            elif error_rate < 0.05:
                status = HealthStatus.DEGRADED
                message = f'Elevated error rate: {error_rate:.2%}'
            else:
                status = HealthStatus.UNHEALTHY
                message = f'High error rate: {error_rate:.2%}'
            
            return HealthCheckResult(
                component='error_rate',
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics={'error_rate': error_rate},
                checks={'error_rate_ok': error_rate < 0.01}
            )
        except Exception as e:
            return HealthCheckResult(
                component='error_rate',
                status=HealthStatus.UNKNOWN,
                message=f'Error rate check failed: {e}',
                timestamp=datetime.now()
            )
    
    def _check_latency(self) -> HealthCheckResult:
        """Check service latency"""
        try:
            avg_latency = self._calculate_average_latency()
            
            if avg_latency < 100:
                status = HealthStatus.HEALTHY
                message = f'Low latency: {avg_latency:.0f}ms'
            elif avg_latency < 500:
                status = HealthStatus.DEGRADED
                message = f'Elevated latency: {avg_latency:.0f}ms'
            else:
                status = HealthStatus.UNHEALTHY
                message = f'High latency: {avg_latency:.0f}ms'
            
            return HealthCheckResult(
                component='latency',
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics={'avg_latency_ms': avg_latency},
                checks={'latency_ok': avg_latency < 100}
            )
        except Exception as e:
            return HealthCheckResult(
                component='latency',
                status=HealthStatus.UNKNOWN,
                message=f'Latency check failed: {e}',
                timestamp=datetime.now()
            )
    
    def _collect_service_metrics(self) -> Dict[str, Any]:
        """Collect service metrics"""
        try:
            metrics = self.service.metrics
            return {
                'service_uptime': time.time() - getattr(metrics, 'uptime_start', time.time()),
                'total_requests': getattr(metrics, 'total_requests', 0),
                'successful_requests': getattr(metrics, 'successful_requests', 0),
                'failed_requests': getattr(metrics, 'failed_requests', 0)
            }
        except Exception as e:
            self.logger.error(f"Service metrics collection failed: {e}")
            return {}
    
    def _collect_compression_metrics(self) -> Dict[str, Any]:
        """Collect compression metrics"""
        try:
            compression_system = getattr(self.service, 'compression_system', None)
            if not compression_system:
                return {}
            
            stats = getattr(compression_system, 'stats', {})
            return {
                'compression_total': stats.get('total_compressions', 0),
                'decompression_total': stats.get('total_decompressions', 0),
                'compression_ratio': stats.get('avg_compression_ratio', 0),
                'reconstruction_error': stats.get('avg_reconstruction_error', 0)
            }
        except Exception as e:
            self.logger.error(f"Compression metrics collection failed: {e}")
            return {}
    
    def _collect_resource_metrics(self) -> Dict[str, Any]:
        """Collect resource metrics"""
        try:
            metrics = {}
            
            # CPU metrics
            import psutil
            metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            metrics['memory_percent'] = psutil.virtual_memory().percent
            
            # GPU metrics
            if torch.cuda.is_available():
                metrics['gpu_memory_allocated'] = torch.cuda.memory_allocated()
                metrics['gpu_memory_reserved'] = torch.cuda.memory_reserved()
            
            return metrics
        except Exception as e:
            self.logger.error(f"Resource metrics collection failed: {e}")
            return {}
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics"""
        try:
            # Calculate various performance metrics
            return {
                'avg_latency_ms': self._calculate_average_latency(),
                'error_rate': self._calculate_error_rate(),
                'throughput_rps': self._calculate_throughput()
            }
        except Exception as e:
            self.logger.error(f"Performance metrics collection failed: {e}")
            return {}
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate"""
        try:
            metrics = self.service.metrics
            total = getattr(metrics, 'total_requests', 0)
            if total == 0:
                return 0.0
            return getattr(metrics, 'failed_requests', 0) / total
        except Exception:
            return 0.0
    
    def _calculate_average_latency(self) -> float:
        """Calculate average latency in milliseconds"""
        try:
            response_times = []
            metrics = self.service.metrics
            if hasattr(metrics, 'response_times'):
                for times in metrics.response_times.values():
                    response_times.extend(times)
            
            if not response_times:
                return 0.0
            
            return sum(response_times) / len(response_times) * 1000
        except Exception:
            return 0.0
    
    def _calculate_memory_usage(self) -> float:
        """Calculate memory usage percentage"""
        try:
            import psutil
            return psutil.virtual_memory().percent / 100
        except Exception:
            return 0.0
    
    def _calculate_throughput(self) -> float:
        """Calculate throughput in requests per second"""
        try:
            metrics = self.service.metrics
            uptime = time.time() - getattr(metrics, 'uptime_start', time.time())
            if uptime == 0:
                return 0.0
            return getattr(metrics, 'total_requests', 0) / uptime
        except Exception:
            return 0.0
    
    def _log_alert(self, alert: Dict[str, Any]) -> None:
        """Log alert"""
        severity = alert['severity']
        message = alert['message']
        
        if severity == 'critical':
            self.logger.critical(f"ALERT: {message}")
        elif severity == 'warning':
            self.logger.warning(f"ALERT: {message}")
        else:
            self.logger.info(f"ALERT: {message}")
    
    def _email_alert(self, alert: Dict[str, Any]) -> None:
        """Send email alert (placeholder)"""
        # This would implement email sending
        self.logger.info(f"Email alert would be sent: {alert['message']}")
    
    def _webhook_alert(self, alert: Dict[str, Any]) -> None:
        """Send webhook alert (placeholder)"""
        # This would implement webhook posting
        self.logger.info(f"Webhook alert would be sent: {alert['message']}")
    
    def get_health_status(self, component: Optional[str] = None) -> Union[HealthCheckResult, Dict[str, HealthCheckResult]]:
        """Get health status for component(s)"""
        with self._lock:
            if component:
                return self.health_results.get(component)
            return self.health_results.copy()
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall health status"""
        with self._lock:
            if not self.health_results:
                return HealthStatus.UNKNOWN
            
            statuses = [result.status for result in self.health_results.values()]
            
            if any(s == HealthStatus.UNHEALTHY for s in statuses):
                return HealthStatus.UNHEALTHY
            elif any(s == HealthStatus.DEGRADED for s in statuses):
                return HealthStatus.DEGRADED
            elif all(s == HealthStatus.HEALTHY for s in statuses):
                return HealthStatus.HEALTHY
            else:
                return HealthStatus.UNKNOWN
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        try:
            overall_health = self.get_overall_health()
            health_results = self.get_health_status()
            metrics = self.collect_metrics()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': overall_health.value,
                'components': {
                    name: {
                        'status': result.status.value,
                        'message': result.message,
                        'checks': result.checks,
                        'metrics': result.metrics
                    }
                    for name, result in health_results.items()
                },
                'metrics': metrics,
                'uptime': time.time() - getattr(self.service.metrics, 'uptime_start', time.time())
            }
        except Exception as e:
            self.logger.error(f"Failed to generate health report: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': HealthStatus.UNKNOWN.value,
                'error': str(e)
            }


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    size_bytes: int
    timestamp: datetime
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    ttl: Optional[int] = None
    compression_ratio: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live
    SIZE = "size"  # Size-based


class SheafServiceCache:
    """
    Caching system for Sheaf compression results.
    
    Provides result caching, invalidation, statistics,
    persistence, and optimization capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize cache"""
        self.config = config or {}
        self.max_size_bytes = self.config.get('max_size_mb', 1024) * 1024 * 1024
        self.eviction_policy = EvictionPolicy(self.config.get('eviction_policy', 'lru'))
        self.ttl_seconds = self.config.get('ttl_seconds', 3600)
        self.compression_threshold = self.config.get('compression_threshold', 1024)
        self.cache: Dict[str, CacheEntry] = {}
        self.access_queue: deque = deque(maxlen=10000)
        self.size_bytes = 0
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'invalidations': 0
        }
        self.persistence_enabled = self.config.get('persistence_enabled', False)
        self.persistence_path = Path(self.config.get('persistence_path', 'sheaf_cache.pkl'))
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._lock = threading.RLock()
        self._initialize_cache()
    
    def _initialize_cache(self) -> None:
        """Initialize cache system"""
        try:
            # Load persisted cache if enabled
            if self.persistence_enabled and self.persistence_path.exists():
                self._load_cache()
            
            # Start cleanup thread
            self._start_cleanup_thread()
            
            self.logger.info(f"Cache initialized with {len(self.cache)} entries")
        except Exception as e:
            self.logger.error(f"Failed to initialize cache: {e}")
            raise RuntimeError(f"Cache initialization failed: {e}")
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread"""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(60)  # Run every minute
                    self._cleanup_expired()
                except Exception as e:
                    self.logger.error(f"Cleanup thread error: {e}")
        
        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self.cache:
                self.stats['misses'] += 1
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if entry.ttl and (datetime.now() - entry.timestamp).total_seconds() > entry.ttl:
                self._remove_entry(key)
                self.stats['misses'] += 1
                return None
            
            # Update access info
            entry.access_count += 1
            entry.last_access = datetime.now()
            self.access_queue.append((datetime.now(), key))
            
            self.stats['hits'] += 1
            
            # Decompress if needed
            value = entry.value
            if entry.metadata.get('compressed', False):
                value = self._decompress_value(value)
            
            return value
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Put value in cache"""
        try:
            with self._lock:
                # Calculate size
                size_bytes = self._calculate_size(value)
                
                # Check if we need to evict
                while self.size_bytes + size_bytes > self.max_size_bytes:
                    if not self._evict_entry():
                        self.logger.warning("Cannot evict more entries")
                        return False
                
                # Compress if needed
                compressed = False
                compression_ratio = 1.0
                if size_bytes > self.compression_threshold:
                    compressed_value, compressed_size = self._compress_value(value)
                    if compressed_size < size_bytes * 0.9:  # Only use if >10% savings
                        value = compressed_value
                        compression_ratio = compressed_size / size_bytes
                        size_bytes = compressed_size
                        compressed = True
                
                # Create entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    size_bytes=size_bytes,
                    timestamp=datetime.now(),
                    ttl=ttl or self.ttl_seconds,
                    compression_ratio=compression_ratio,
                    metadata=metadata or {}
                )
                
                entry.metadata['compressed'] = compressed
                
                # Remove old entry if exists
                if key in self.cache:
                    self._remove_entry(key)
                
                # Add new entry
                self.cache[key] = entry
                self.size_bytes += size_bytes
                self.access_queue.append((datetime.now(), key))
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to put value in cache: {e}")
            return False
    
    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry"""
        with self._lock:
            if key in self.cache:
                self._remove_entry(key)
                self.stats['invalidations'] += 1
                return True
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate entries matching pattern"""
        import re
        regex = re.compile(pattern)
        
        with self._lock:
            keys_to_remove = [k for k in self.cache.keys() if regex.match(k)]
            
            for key in keys_to_remove:
                self._remove_entry(key)
                self.stats['invalidations'] += 1
            
            return len(keys_to_remove)
    
    def clear(self) -> None:
        """Clear entire cache"""
        with self._lock:
            self.cache.clear()
            self.access_queue.clear()
            self.size_bytes = 0
            self.logger.info("Cache cleared")
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache"""
        if key in self.cache:
            entry = self.cache[key]
            self.size_bytes -= entry.size_bytes
            del self.cache[key]
    
    def _evict_entry(self) -> bool:
        """Evict entry based on policy"""
        if not self.cache:
            return False
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Find least recently used
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k].last_access)
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Find least frequently used
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k].access_count)
        elif self.eviction_policy == EvictionPolicy.FIFO:
            # Find oldest entry
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k].timestamp)
        elif self.eviction_policy == EvictionPolicy.SIZE:
            # Evict largest entry
            oldest_key = max(self.cache.keys(), 
                           key=lambda k: self.cache[k].size_bytes)
        else:  # TTL or default
            # Find entry closest to expiration
            now = datetime.now()
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: (self.cache[k].ttl or float('inf')) - 
                                       (now - self.cache[k].timestamp).total_seconds())
        
        self._remove_entry(oldest_key)
        self.stats['evictions'] += 1
        return True
    
    def _cleanup_expired(self) -> None:
        """Clean up expired entries"""
        try:
            with self._lock:
                now = datetime.now()
                expired_keys = []
                
                for key, entry in self.cache.items():
                    if entry.ttl and (now - entry.timestamp).total_seconds() > entry.ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    self._remove_entry(key)
                
                if expired_keys:
                    self.logger.info(f"Cleaned up {len(expired_keys)} expired entries")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes"""
        try:
            if isinstance(value, torch.Tensor):
                return value.element_size() * value.numel()
            elif isinstance(value, np.ndarray):
                return value.nbytes
            elif isinstance(value, (str, bytes)):
                return len(value)
            else:
                # Serialize to estimate size
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default size
    
    def _compress_value(self, value: Any) -> Tuple[bytes, int]:
        """Compress value for storage"""
        try:
            # Serialize value
            serialized = pickle.dumps(value)
            
            # Compress
            compressed = zlib.compress(serialized, level=6)
            
            return compressed, len(compressed)
        except Exception as e:
            self.logger.error(f"Compression failed: {e}")
            raise
    
    def _decompress_value(self, compressed: bytes) -> Any:
        """Decompress value from storage"""
        try:
            # Decompress
            decompressed = zlib.decompress(compressed)
            
            # Deserialize
            return pickle.loads(decompressed)
        except Exception as e:
            self.logger.error(f"Decompression failed: {e}")
            raise
    
    def save_cache(self) -> None:
        """Save cache to disk"""
        if not self.persistence_enabled:
            return
        
        try:
            with self._lock:
                cache_data = {
                    'cache': self.cache,
                    'stats': self.stats,
                    'size_bytes': self.size_bytes
                }
                
                with open(self.persistence_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                
                self.logger.info(f"Cache saved to {self.persistence_path}")
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")
    
    def _load_cache(self) -> None:
        """Load cache from disk"""
        try:
            with open(self.persistence_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.cache = cache_data.get('cache', {})
            self.stats = cache_data.get('stats', self.stats)
            self.size_bytes = cache_data.get('size_bytes', 0)
            
            # Clean up expired entries
            self._cleanup_expired()
            
            self.logger.info(f"Cache loaded from {self.persistence_path}")
        except Exception as e:
            self.logger.error(f"Failed to load cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'size_bytes': self.size_bytes,
                'max_size_bytes': self.max_size_bytes,
                'utilization': self.size_bytes / self.max_size_bytes,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': hit_rate,
                'evictions': self.stats['evictions'],
                'invalidations': self.stats['invalidations'],
                'eviction_policy': self.eviction_policy.value,
                'ttl_seconds': self.ttl_seconds,
                'compression_enabled': self.compression_threshold > 0,
                'persistence_enabled': self.persistence_enabled
            }
    
    def optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache performance"""
        try:
            with self._lock:
                optimization_report = {
                    'before': self.get_stats(),
                    'actions': []
                }
                
                # Remove expired entries
                self._cleanup_expired()
                optimization_report['actions'].append('Cleaned expired entries')
                
                # Analyze access patterns
                access_analysis = self._analyze_access_patterns()
                
                # Adjust eviction policy based on patterns
                if access_analysis['temporal_locality'] > 0.7:
                    self.eviction_policy = EvictionPolicy.LRU
                    optimization_report['actions'].append('Set eviction policy to LRU')
                elif access_analysis['frequency_skew'] > 0.7:
                    self.eviction_policy = EvictionPolicy.LFU
                    optimization_report['actions'].append('Set eviction policy to LFU')
                
                # Compress large uncompressed entries
                compressed_count = self._compress_large_entries()
                if compressed_count > 0:
                    optimization_report['actions'].append(f'Compressed {compressed_count} entries')
                
                # Save if persistence enabled
                if self.persistence_enabled:
                    self.save_cache()
                    optimization_report['actions'].append('Saved cache to disk')
                
                optimization_report['after'] = self.get_stats()
                
                return optimization_report
                
        except Exception as e:
            self.logger.error(f"Cache optimization failed: {e}")
            return {'error': str(e)}
    
    def _analyze_access_patterns(self) -> Dict[str, float]:
        """Analyze cache access patterns"""
        try:
            if len(self.access_queue) < 100:
                return {
                    'temporal_locality': 0.5,
                    'frequency_skew': 0.5
                }
            
            # Analyze temporal locality
            recent_accesses = list(self.access_queue)[-1000:]
            unique_keys = len(set(k for _, k in recent_accesses))
            temporal_locality = 1.0 - (unique_keys / len(recent_accesses))
            
            # Analyze frequency skew
            access_counts = defaultdict(int)
            for _, key in recent_accesses:
                access_counts[key] += 1
            
            sorted_counts = sorted(access_counts.values(), reverse=True)
            top_20_percent = int(len(sorted_counts) * 0.2)
            top_20_accesses = sum(sorted_counts[:top_20_percent])
            total_accesses = sum(sorted_counts)
            frequency_skew = top_20_accesses / total_accesses if total_accesses > 0 else 0
            
            return {
                'temporal_locality': temporal_locality,
                'frequency_skew': frequency_skew
            }
        except Exception as e:
            self.logger.error(f"Access pattern analysis failed: {e}")
            return {
                'temporal_locality': 0.5,
                'frequency_skew': 0.5
            }
    
    def _compress_large_entries(self) -> int:
        """Compress large uncompressed entries"""
        compressed_count = 0
        
        try:
            for key, entry in list(self.cache.items()):
                if (not entry.metadata.get('compressed', False) and 
                    entry.size_bytes > self.compression_threshold):
                    
                    # Get original value
                    value = entry.value
                    
                    # Try to compress
                    compressed_value, compressed_size = self._compress_value(value)
                    
                    if compressed_size < entry.size_bytes * 0.9:
                        # Update entry
                        self.size_bytes -= entry.size_bytes
                        entry.value = compressed_value
                        entry.size_bytes = compressed_size
                        entry.compression_ratio = compressed_size / entry.size_bytes
                        entry.metadata['compressed'] = True
                        self.size_bytes += compressed_size
                        compressed_count += 1
            
            return compressed_count
        except Exception as e:
            self.logger.error(f"Entry compression failed: {e}")
            return compressed_count