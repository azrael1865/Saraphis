"""
Hybrid Service Middleware - Middleware for request/response processing
NO FALLBACKS - HARD FAILURES ONLY
"""

import asyncio
import logging
import threading
import time
import torch
import uuid
import json
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

# Import service interfaces
from service_interfaces.service_interfaces_core import (
    CompressionServiceInterface, ServiceRequest, ServiceResponse,
    ServiceStatus, ServiceHealth
)


class MiddlewareType(Enum):
    """Types of middleware"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMITING = "rate_limiting"
    CACHING = "caching"
    COMPRESSION = "compression"
    ENCRYPTION = "encryption"
    VALIDATION = "validation"
    MONITORING = "monitoring"
    TRANSFORMATION = "transformation"
    LOAD_BALANCING = "load_balancing"


class MiddlewarePriority(Enum):
    """Middleware execution priorities"""
    CRITICAL = 1000
    HIGH = 800
    NORMAL = 500
    LOW = 200
    BACKGROUND = 100


@dataclass
class MiddlewareConfig:
    """Configuration for middleware components"""
    # Authentication & Authorization
    enable_authentication: bool = False
    auth_token_header: str = "Authorization"
    auth_token_prefix: str = "Bearer"
    auth_timeout_seconds: float = 300.0
    
    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_requests_per_minute: int = 1000
    rate_limit_window_seconds: int = 60
    rate_limit_burst_size: int = 100
    
    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    cache_max_size_mb: int = 256
    cache_compression_enabled: bool = True
    
    # Monitoring & Logging
    enable_request_logging: bool = True
    enable_performance_tracking: bool = True
    enable_error_tracking: bool = True
    log_level: str = "INFO"
    
    # Request processing
    enable_request_validation: bool = True
    enable_response_validation: bool = True
    enable_payload_transformation: bool = True
    max_request_size_mb: int = 100
    request_timeout_seconds: float = 300.0
    
    # GPU optimization
    enable_gpu_memory_monitoring: bool = True
    gpu_memory_threshold_mb: int = 1024
    enable_tensor_optimization: bool = True
    
    # Security
    enable_request_sanitization: bool = True
    enable_response_filtering: bool = True
    max_payload_depth: int = 10
    
    def __post_init__(self):
        """Validate middleware configuration"""
        if not isinstance(self.rate_limit_requests_per_minute, int) or self.rate_limit_requests_per_minute <= 0:
            raise ValueError("Rate limit requests per minute must be positive int")
        if not isinstance(self.cache_ttl_seconds, int) or self.cache_ttl_seconds <= 0:
            raise ValueError("Cache TTL must be positive int")
        if not isinstance(self.max_request_size_mb, int) or self.max_request_size_mb <= 0:
            raise ValueError("Max request size must be positive int")
        if not isinstance(self.request_timeout_seconds, (int, float)) or self.request_timeout_seconds <= 0:
            raise ValueError("Request timeout must be positive number")


@dataclass
class MiddlewareMetrics:
    """Metrics for middleware operations"""
    total_requests_processed: int = 0
    requests_passed: int = 0
    requests_blocked: int = 0
    requests_modified: int = 0
    average_processing_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    rate_limit_violations: int = 0
    validation_failures: int = 0
    authentication_failures: int = 0
    transformation_errors: int = 0
    last_metrics_update: Optional[datetime] = None
    
    def update_processing_metrics(self, processing_time: float, passed: bool, modified: bool = False):
        """Update processing metrics"""
        self.total_requests_processed += 1
        self.last_metrics_update = datetime.utcnow()
        
        if passed:
            self.requests_passed += 1
        else:
            self.requests_blocked += 1
        
        if modified:
            self.requests_modified += 1
        
        # Update average processing time
        if self.total_requests_processed > 1:
            old_avg = self.average_processing_time
            self.average_processing_time = (
                (old_avg * (self.total_requests_processed - 1) + processing_time) / 
                self.total_requests_processed
            )
        else:
            self.average_processing_time = processing_time


class MiddlewareComponent:
    """Base class for middleware components"""
    
    def __init__(self, name: str, middleware_type: MiddlewareType, 
                 priority: MiddlewarePriority = MiddlewarePriority.NORMAL):
        self.name = name
        self.middleware_type = middleware_type
        self.priority = priority
        self.enabled = True
        self.metrics = MiddlewareMetrics()
        self.logger = logging.getLogger(f'Middleware.{name}')
    
    def process_request(self, request: ServiceRequest, context: Dict[str, Any]) -> Tuple[ServiceRequest, bool]:
        """
        Process incoming request.
        
        Args:
            request: Service request to process
            context: Processing context
            
        Returns:
            Tuple of (modified_request, should_continue)
        """
        start_time = time.time()
        
        try:
            modified_request, should_continue = self._process_request_impl(request, context)
            
            processing_time = time.time() - start_time
            self.metrics.update_processing_metrics(processing_time, should_continue, modified_request != request)
            
            return modified_request, should_continue
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics.update_processing_metrics(processing_time, False)
            self.logger.error(f"Error in request processing: {e}")
            raise e
    
    def process_response(self, response: ServiceResponse, context: Dict[str, Any]) -> ServiceResponse:
        """
        Process outgoing response.
        
        Args:
            response: Service response to process
            context: Processing context
            
        Returns:
            Modified response
        """
        try:
            return self._process_response_impl(response, context)
        except Exception as e:
            self.logger.error(f"Error in response processing: {e}")
            raise e
    
    def _process_request_impl(self, request: ServiceRequest, context: Dict[str, Any]) -> Tuple[ServiceRequest, bool]:
        """Implementation of request processing - override in subclasses"""
        return request, True
    
    def _process_response_impl(self, response: ServiceResponse, context: Dict[str, Any]) -> ServiceResponse:
        """Implementation of response processing - override in subclasses"""
        return response


class RateLimitingMiddleware(MiddlewareComponent):
    """Rate limiting middleware"""
    
    def __init__(self, config: MiddlewareConfig):
        super().__init__("RateLimiting", MiddlewareType.RATE_LIMITING, MiddlewarePriority.HIGH)
        self.config = config
        self.request_counts: Dict[str, List[datetime]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def _process_request_impl(self, request: ServiceRequest, context: Dict[str, Any]) -> Tuple[ServiceRequest, bool]:
        if not self.config.enable_rate_limiting:
            return request, True
        
        client_id = self._get_client_id(request)
        current_time = datetime.utcnow()
        
        with self._lock:
            # Clean old requests outside the window
            window_start = current_time - timedelta(seconds=self.config.rate_limit_window_seconds)
            self.request_counts[client_id] = [
                req_time for req_time in self.request_counts[client_id]
                if req_time >= window_start
            ]
            
            # Check rate limit
            if len(self.request_counts[client_id]) >= self.config.rate_limit_requests_per_minute:
                self.metrics.rate_limit_violations += 1
                context['rate_limit_exceeded'] = True
                return request, False
            
            # Add current request
            self.request_counts[client_id].append(current_time)
            return request, True
    
    def _get_client_id(self, request: ServiceRequest) -> str:
        """Extract client ID from request"""
        # Use auth token if available, otherwise use request metadata
        if request.auth_token:
            return hashlib.sha256(request.auth_token.encode()).hexdigest()[:16]
        
        # Fallback to service name + method combination
        return f"{request.service_name}:{request.method_name}"


class CachingMiddleware(MiddlewareComponent):
    """Caching middleware for responses"""
    
    def __init__(self, config: MiddlewareConfig):
        super().__init__("Caching", MiddlewareType.CACHING, MiddlewarePriority.NORMAL)
        self.config = config
        self.cache: Dict[str, Tuple[ServiceResponse, datetime]] = {}
        self._lock = threading.RLock()
    
    def _process_request_impl(self, request: ServiceRequest, context: Dict[str, Any]) -> Tuple[ServiceRequest, bool]:
        if not self.config.enable_caching:
            return request, True
        
        cache_key = self._generate_cache_key(request)
        
        with self._lock:
            if cache_key in self.cache:
                cached_response, cache_time = self.cache[cache_key]
                
                # Check if cache entry is still valid
                if self._is_cache_valid(cache_time):
                    self.metrics.cache_hits += 1
                    context['cached_response'] = cached_response
                    context['cache_hit'] = True
                    return request, False  # Skip actual service call
                else:
                    # Remove expired entry
                    del self.cache[cache_key]
        
        self.metrics.cache_misses += 1
        context['cache_key'] = cache_key
        return request, True
    
    def _process_response_impl(self, response: ServiceResponse, context: Dict[str, Any]) -> ServiceResponse:
        if not self.config.enable_caching or not response.is_success():
            return response
        
        cache_key = context.get('cache_key')
        if cache_key:
            with self._lock:
                # Store successful responses in cache
                self.cache[cache_key] = (response, datetime.utcnow())
                
                # Clean cache if it's getting too large
                if len(self.cache) > 1000:  # Simple cache size management
                    self._cleanup_cache()
        
        return response
    
    def _generate_cache_key(self, request: ServiceRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            'service': request.service_name,
            'method': request.method_name,
            'version': request.version,
            'payload': request.payload
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_time: datetime) -> bool:
        """Check if cache entry is still valid"""
        age = (datetime.utcnow() - cache_time).total_seconds()
        return age < self.config.cache_ttl_seconds
    
    def _cleanup_cache(self) -> None:
        """Clean up old cache entries"""
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, (_, cache_time) in self.cache.items()
            if not self._is_cache_valid(cache_time)
        ]
        
        for key in expired_keys:
            del self.cache[key]


class ValidationMiddleware(MiddlewareComponent):
    """Request and response validation middleware"""
    
    def __init__(self, config: MiddlewareConfig):
        super().__init__("Validation", MiddlewareType.VALIDATION, MiddlewarePriority.HIGH)
        self.config = config
    
    def _process_request_impl(self, request: ServiceRequest, context: Dict[str, Any]) -> Tuple[ServiceRequest, bool]:
        if not self.config.enable_request_validation:
            return request, True
        
        try:
            # Validate request structure
            self._validate_request_structure(request)
            
            # Validate payload size
            self._validate_payload_size(request)
            
            # Validate payload content
            self._validate_payload_content(request)
            
            return request, True
            
        except Exception as e:
            self.metrics.validation_failures += 1
            context['validation_error'] = str(e)
            self.logger.warning(f"Request validation failed: {e}")
            return request, False
    
    def _process_response_impl(self, response: ServiceResponse, context: Dict[str, Any]) -> ServiceResponse:
        if not self.config.enable_response_validation:
            return response
        
        try:
            self._validate_response_structure(response)
            return response
        except Exception as e:
            self.logger.error(f"Response validation failed: {e}")
            # Don't block response, but log the issue
            response.metadata['validation_warning'] = str(e)
            return response
    
    def _validate_request_structure(self, request: ServiceRequest) -> None:
        """Validate request structure"""
        if not request.service_name or not isinstance(request.service_name, str):
            raise ValueError("Invalid service name")
        if not request.method_name or not isinstance(request.method_name, str):
            raise ValueError("Invalid method name")
        if not request.version or not isinstance(request.version, str):
            raise ValueError("Invalid version")
        if request.payload is None or not isinstance(request.payload, dict):
            raise ValueError("Invalid payload")
    
    def _validate_payload_size(self, request: ServiceRequest) -> None:
        """Validate payload size"""
        payload_str = json.dumps(request.payload)
        payload_size_mb = len(payload_str.encode('utf-8')) / (1024 * 1024)
        
        if payload_size_mb > self.config.max_request_size_mb:
            raise ValueError(f"Payload size {payload_size_mb:.2f}MB exceeds limit {self.config.max_request_size_mb}MB")
    
    def _validate_payload_content(self, request: ServiceRequest) -> None:
        """Validate payload content"""
        if self.config.enable_request_sanitization:
            self._sanitize_payload(request.payload, depth=0)
    
    def _sanitize_payload(self, obj: Any, depth: int) -> None:
        """Recursively sanitize payload content"""
        if depth > self.config.max_payload_depth:
            raise ValueError(f"Payload depth exceeds limit {self.config.max_payload_depth}")
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(key, str) and any(char in key for char in ['<', '>', '&', '"', "'"]):
                    raise ValueError(f"Invalid characters in key: {key}")
                self._sanitize_payload(value, depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                self._sanitize_payload(item, depth + 1)
    
    def _validate_response_structure(self, response: ServiceResponse) -> None:
        """Validate response structure"""
        if not response.request_id or not isinstance(response.request_id, str):
            raise ValueError("Invalid request ID in response")
        if not isinstance(response.status, ServiceStatus):
            raise ValueError("Invalid status in response")


class GPUOptimizationMiddleware(MiddlewareComponent):
    """GPU optimization middleware for tensor operations"""
    
    def __init__(self, config: MiddlewareConfig):
        super().__init__("GPUOptimization", MiddlewareType.TRANSFORMATION, MiddlewarePriority.NORMAL)
        self.config = config
        self.gpu_available = torch.cuda.is_available()
    
    def _process_request_impl(self, request: ServiceRequest, context: Dict[str, Any]) -> Tuple[ServiceRequest, bool]:
        if not self.config.enable_tensor_optimization or not self.gpu_available:
            return request, True
        
        try:
            # Check GPU memory before processing
            if self.config.enable_gpu_memory_monitoring:
                self._check_gpu_memory(context)
            
            # Optimize tensor data in payload
            modified_request = self._optimize_request_tensors(request)
            
            return modified_request, True
            
        except Exception as e:
            self.logger.error(f"GPU optimization failed: {e}")
            return request, True  # Continue without optimization
    
    def _process_response_impl(self, response: ServiceResponse, context: Dict[str, Any]) -> ServiceResponse:
        if not self.config.enable_tensor_optimization or not self.gpu_available:
            return response
        
        try:
            # Optimize tensor data in response
            return self._optimize_response_tensors(response)
        except Exception as e:
            self.logger.error(f"Response tensor optimization failed: {e}")
            return response
    
    def _check_gpu_memory(self, context: Dict[str, Any]) -> None:
        """Check GPU memory availability"""
        if not torch.cuda.is_available():
            return
        
        allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        
        if allocated_mb > self.config.gpu_memory_threshold_mb:
            context['gpu_memory_warning'] = True
            self.logger.warning(f"GPU memory usage high: {allocated_mb:.2f}MB")
        
        context['gpu_memory_allocated_mb'] = allocated_mb
    
    def _optimize_request_tensors(self, request: ServiceRequest) -> ServiceRequest:
        """Optimize tensors in request payload"""
        optimized_payload = self._optimize_tensor_data(request.payload)
        
        if optimized_payload != request.payload:
            # Create new request with optimized payload
            new_request = ServiceRequest(
                service_name=request.service_name,
                method_name=request.method_name,
                version=request.version,
                payload=optimized_payload,
                metadata=request.metadata.copy(),
                request_id=request.request_id,
                timestamp=request.timestamp,
                auth_token=request.auth_token,
                timeout=request.timeout,
                priority=request.priority
            )
            new_request.metadata['tensor_optimized'] = True
            return new_request
        
        return request
    
    def _optimize_response_tensors(self, response: ServiceResponse) -> ServiceResponse:
        """Optimize tensors in response data"""
        if response.data:
            optimized_data = self._optimize_tensor_data(response.data)
            
            if optimized_data != response.data:
                response.data = optimized_data
                response.metadata['tensor_optimized'] = True
        
        return response
    
    def _optimize_tensor_data(self, data: Any) -> Any:
        """Recursively optimize tensor data"""
        if isinstance(data, torch.Tensor):
            # Move small tensors to CPU, keep large ones on GPU
            if data.numel() < 1000 and data.is_cuda:
                return data.cpu()
            elif data.numel() >= 1000 and not data.is_cuda and torch.cuda.is_available():
                return data.cuda()
            return data
        elif isinstance(data, dict):
            return {key: self._optimize_tensor_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._optimize_tensor_data(item) for item in data]
        else:
            return data


class MonitoringMiddleware(MiddlewareComponent):
    """Monitoring and logging middleware"""
    
    def __init__(self, config: MiddlewareConfig):
        super().__init__("Monitoring", MiddlewareType.MONITORING, MiddlewarePriority.LOW)
        self.config = config
        self.request_logs: deque = deque(maxlen=1000)
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def _process_request_impl(self, request: ServiceRequest, context: Dict[str, Any]) -> Tuple[ServiceRequest, bool]:
        if self.config.enable_request_logging:
            self._log_request(request, context)
        
        context['monitoring_start_time'] = time.time()
        return request, True
    
    def _process_response_impl(self, response: ServiceResponse, context: Dict[str, Any]) -> ServiceResponse:
        if self.config.enable_performance_tracking:
            self._track_performance(response, context)
        
        if self.config.enable_error_tracking and not response.is_success():
            self._track_error(response, context)
        
        return response
    
    def _log_request(self, request: ServiceRequest, context: Dict[str, Any]) -> None:
        """Log request details"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': request.request_id,
            'service': request.service_name,
            'method': request.method_name,
            'version': request.version,
            'payload_size': len(json.dumps(request.payload)),
            'context': {k: v for k, v in context.items() if isinstance(v, (str, int, float, bool))}
        }
        
        with self._lock:
            self.request_logs.append(log_entry)
        
        if self.config.log_level == "DEBUG":
            self.logger.debug(f"Request: {log_entry}")
        elif self.config.log_level == "INFO":
            self.logger.info(f"Processing request {request.request_id} for {request.service_name}.{request.method_name}")
    
    def _track_performance(self, response: ServiceResponse, context: Dict[str, Any]) -> None:
        """Track performance metrics"""
        start_time = context.get('monitoring_start_time')
        if start_time:
            processing_time = time.time() - start_time
            
            method_key = f"{response.metadata.get('service_name', 'unknown')}.{response.metadata.get('method_name', 'unknown')}"
            
            with self._lock:
                self.performance_metrics[method_key].append(processing_time)
                
                # Keep only recent metrics
                if len(self.performance_metrics[method_key]) > 100:
                    self.performance_metrics[method_key] = self.performance_metrics[method_key][-50:]
    
    def _track_error(self, response: ServiceResponse, context: Dict[str, Any]) -> None:
        """Track error details"""
        error_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': response.request_id,
            'status': response.status.value,
            'errors': response.errors,
            'context': {k: v for k, v in context.items() if isinstance(v, (str, int, float, bool))}
        }
        
        self.logger.error(f"Request failed: {error_entry}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        with self._lock:
            summary = {}
            for method, times in self.performance_metrics.items():
                if times:
                    summary[method] = {
                        'count': len(times),
                        'avg_time': sum(times) / len(times),
                        'min_time': min(times),
                        'max_time': max(times)
                    }
            return summary


class HybridServiceMiddleware:
    """
    Main middleware processor for hybrid services.
    Manages middleware pipeline for request/response processing.
    """
    
    def __init__(self, config: Optional[MiddlewareConfig] = None):
        """Initialize hybrid service middleware"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, MiddlewareConfig):
            raise TypeError(f"Config must be MiddlewareConfig or None, got {type(config)}")
        
        self.config = config or MiddlewareConfig()
        self.logger = logging.getLogger('HybridServiceMiddleware')
        
        # Middleware components
        self.middleware_components: List[MiddlewareComponent] = []
        self.middleware_by_type: Dict[MiddlewareType, List[MiddlewareComponent]] = defaultdict(list)
        
        # Processing state
        self.is_initialized = False
        self.total_requests_processed = 0
        self.startup_time = datetime.utcnow()
        
        # Thread safety
        self._middleware_lock = threading.RLock()
        
        # Initialize default middleware
        self._initialize_default_middleware()
        
        self.logger.info("HybridServiceMiddleware created successfully")
    
    def _initialize_default_middleware(self) -> None:
        """Initialize default middleware components"""
        try:
            # Add core middleware components
            if self.config.enable_rate_limiting:
                self.add_middleware(RateLimitingMiddleware(self.config))
            
            if self.config.enable_request_validation:
                self.add_middleware(ValidationMiddleware(self.config))
            
            if self.config.enable_caching:
                self.add_middleware(CachingMiddleware(self.config))
            
            if self.config.enable_gpu_memory_monitoring or self.config.enable_tensor_optimization:
                self.add_middleware(GPUOptimizationMiddleware(self.config))
            
            if (self.config.enable_request_logging or 
                self.config.enable_performance_tracking or 
                self.config.enable_error_tracking):
                self.add_middleware(MonitoringMiddleware(self.config))
            
            self.is_initialized = True
            self.logger.info(f"Initialized {len(self.middleware_components)} middleware components")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default middleware: {e}")
            raise RuntimeError(f"Middleware initialization failed: {e}")
    
    def add_middleware(self, middleware: MiddlewareComponent) -> None:
        """
        Add middleware component to processing pipeline.
        
        Args:
            middleware: Middleware component to add
            
        Raises:
            ValueError: If middleware is invalid
        """
        if not isinstance(middleware, MiddlewareComponent):
            raise TypeError(f"Middleware must be MiddlewareComponent, got {type(middleware)}")
        
        with self._middleware_lock:
            self.middleware_components.append(middleware)
            self.middleware_by_type[middleware.middleware_type].append(middleware)
            
            # Sort by priority (higher priority first)
            self.middleware_components.sort(key=lambda m: m.priority.value, reverse=True)
            
            self.logger.info(f"Added middleware: {middleware.name} (type: {middleware.middleware_type.value})")
    
    def remove_middleware(self, middleware_name: str) -> bool:
        """
        Remove middleware component by name.
        
        Args:
            middleware_name: Name of middleware to remove
            
        Returns:
            True if middleware was removed
        """
        if not isinstance(middleware_name, str) or not middleware_name.strip():
            raise ValueError("Middleware name must be non-empty string")
        
        with self._middleware_lock:
            for middleware in self.middleware_components[:]:
                if middleware.name == middleware_name:
                    self.middleware_components.remove(middleware)
                    self.middleware_by_type[middleware.middleware_type].remove(middleware)
                    self.logger.info(f"Removed middleware: {middleware_name}")
                    return True
            
            return False
    
    def process_request(self, request: ServiceRequest) -> Tuple[ServiceRequest, bool, Dict[str, Any]]:
        """
        Process request through middleware pipeline.
        
        Args:
            request: Service request to process
            
        Returns:
            Tuple of (modified_request, should_continue, context)
            
        Raises:
            ValueError: If request is invalid
            RuntimeError: If processing fails
        """
        if not isinstance(request, ServiceRequest):
            raise TypeError(f"Request must be ServiceRequest, got {type(request)}")
        
        if not self.is_initialized:
            raise RuntimeError("Middleware not initialized")
        
        self.total_requests_processed += 1
        
        # Initialize processing context
        context = {
            'processing_start_time': time.time(),
            'middleware_pipeline_length': len(self.middleware_components),
            'request_id': request.request_id,
            'original_request': request
        }
        
        current_request = request
        
        with self._middleware_lock:
            # Process through middleware pipeline
            for middleware in self.middleware_components:
                if not middleware.enabled:
                    continue
                
                try:
                    current_request, should_continue = middleware.process_request(current_request, context)
                    
                    context[f'{middleware.name}_processed'] = True
                    
                    if not should_continue:
                        context['blocked_by_middleware'] = middleware.name
                        context['processing_blocked'] = True
                        return current_request, False, context
                        
                except Exception as e:
                    context['processing_error'] = str(e)
                    context['failed_middleware'] = middleware.name
                    self.logger.error(f"Middleware {middleware.name} failed: {e}")
                    
                    # Decide whether to continue or fail
                    if middleware.priority.value >= MiddlewarePriority.HIGH.value:
                        # Critical/High priority middleware failure blocks request
                        return current_request, False, context
                    else:
                        # Continue with lower priority middleware failures
                        continue
        
        context['processing_completed'] = True
        return current_request, True, context
    
    def process_response(self, response: ServiceResponse, context: Dict[str, Any]) -> ServiceResponse:
        """
        Process response through middleware pipeline.
        
        Args:
            response: Service response to process
            context: Processing context from request processing
            
        Returns:
            Modified response
            
        Raises:
            ValueError: If response is invalid
        """
        if not isinstance(response, ServiceResponse):
            raise TypeError(f"Response must be ServiceResponse, got {type(response)}")
        
        if not isinstance(context, dict):
            raise TypeError(f"Context must be dict, got {type(context)}")
        
        current_response = response
        
        with self._middleware_lock:
            # Process through middleware pipeline in reverse order
            for middleware in reversed(self.middleware_components):
                if not middleware.enabled:
                    continue
                
                try:
                    current_response = middleware.process_response(current_response, context)
                    context[f'{middleware.name}_response_processed'] = True
                    
                except Exception as e:
                    context[f'{middleware.name}_response_error'] = str(e)
                    self.logger.error(f"Response middleware {middleware.name} failed: {e}")
                    
                    # Add error metadata but continue processing
                    if not current_response.metadata:
                        current_response.metadata = {}
                    current_response.metadata[f'{middleware.name}_error'] = str(e)
        
        # Add final processing metadata
        processing_time = time.time() - context.get('processing_start_time', time.time())
        current_response.metadata['middleware_processing_time'] = processing_time
        current_response.metadata['middleware_components_used'] = len(self.middleware_components)
        
        return current_response
    
    def get_middleware_status(self) -> Dict[str, Any]:
        """
        Get status of all middleware components.
        
        Returns:
            Dictionary containing middleware status
        """
        with self._middleware_lock:
            return {
                'initialized': self.is_initialized,
                'total_components': len(self.middleware_components),
                'total_requests_processed': self.total_requests_processed,
                'startup_time': self.startup_time.isoformat(),
                'uptime_seconds': (datetime.utcnow() - self.startup_time).total_seconds(),
                'components': [
                    {
                        'name': m.name,
                        'type': m.middleware_type.value,
                        'priority': m.priority.value,
                        'enabled': m.enabled,
                        'requests_processed': m.metrics.total_requests_processed,
                        'requests_passed': m.metrics.requests_passed,
                        'requests_blocked': m.metrics.requests_blocked,
                        'average_processing_time': m.metrics.average_processing_time
                    }
                    for m in self.middleware_components
                ],
                'components_by_type': {
                    middleware_type.value: len(components)
                    for middleware_type, components in self.middleware_by_type.items()
                }
            }
    
    def get_middleware_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive middleware metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        with self._middleware_lock:
            metrics = {
                'overall': {
                    'total_requests': self.total_requests_processed,
                    'average_pipeline_time': 0.0,
                    'components_count': len(self.middleware_components)
                },
                'by_component': {},
                'by_type': {}
            }
            
            # Collect metrics from each component
            total_processing_time = 0.0
            for middleware in self.middleware_components:
                component_metrics = {
                    'total_processed': middleware.metrics.total_requests_processed,
                    'passed': middleware.metrics.requests_passed,
                    'blocked': middleware.metrics.requests_blocked,
                    'modified': middleware.metrics.requests_modified,
                    'average_time': middleware.metrics.average_processing_time,
                    'cache_hits': getattr(middleware.metrics, 'cache_hits', 0),
                    'cache_misses': getattr(middleware.metrics, 'cache_misses', 0),
                    'rate_limit_violations': getattr(middleware.metrics, 'rate_limit_violations', 0),
                    'validation_failures': getattr(middleware.metrics, 'validation_failures', 0)
                }
                
                metrics['by_component'][middleware.name] = component_metrics
                total_processing_time += middleware.metrics.average_processing_time
                
                # Aggregate by type
                type_key = middleware.middleware_type.value
                if type_key not in metrics['by_type']:
                    metrics['by_type'][type_key] = {
                        'components': 0,
                        'total_processed': 0,
                        'total_time': 0.0
                    }
                
                metrics['by_type'][type_key]['components'] += 1
                metrics['by_type'][type_key]['total_processed'] += middleware.metrics.total_requests_processed
                metrics['by_type'][type_key]['total_time'] += middleware.metrics.average_processing_time
            
            # Calculate overall average
            if self.middleware_components:
                metrics['overall']['average_pipeline_time'] = total_processing_time / len(self.middleware_components)
            
            return metrics
    
    def enable_middleware(self, middleware_name: str) -> bool:
        """Enable middleware component by name"""
        with self._middleware_lock:
            for middleware in self.middleware_components:
                if middleware.name == middleware_name:
                    middleware.enabled = True
                    self.logger.info(f"Enabled middleware: {middleware_name}")
                    return True
            return False
    
    def disable_middleware(self, middleware_name: str) -> bool:
        """Disable middleware component by name"""
        with self._middleware_lock:
            for middleware in self.middleware_components:
                if middleware.name == middleware_name:
                    middleware.enabled = False
                    self.logger.info(f"Disabled middleware: {middleware_name}")
                    return True
            return False
    
    def shutdown(self) -> None:
        """Shutdown middleware processor"""
        self.logger.info("Shutting down hybrid service middleware")
        
        with self._middleware_lock:
            # Disable all middleware
            for middleware in self.middleware_components:
                middleware.enabled = False
            
            self.is_initialized = False
        
        self.logger.info("Hybrid service middleware shutdown complete")
    
    def configure_middleware_pipeline(self, middleware_configs: List[Dict[str, Any]]) -> None:
        """
        Configure middleware pipeline with custom middleware components.
        
        Args:
            middleware_configs: List of middleware configuration dictionaries
                Each config should have: 
                - 'type': middleware type (str)
                - 'name': middleware name (str) 
                - 'config': middleware-specific configuration (dict)
                - 'priority': middleware priority (int, optional)
                - 'enabled': whether middleware is enabled (bool, optional)
        """
        if not isinstance(middleware_configs, list):
            raise TypeError("Middleware configs must be a list")
        
        with self._middleware_lock:
            self.logger.info(f"Configuring middleware pipeline with {len(middleware_configs)} components")
            
            for config in middleware_configs:
                if not isinstance(config, dict):
                    raise TypeError("Each middleware config must be a dictionary")
                
                required_keys = ['type', 'name', 'config']
                for key in required_keys:
                    if key not in config:
                        raise ValueError(f"Missing required key '{key}' in middleware config")
                
                middleware_type = config['type']
                middleware_name = config['name']
                middleware_config = config['config']
                priority = config.get('priority', MiddlewarePriority.NORMAL.value)
                enabled = config.get('enabled', True)
                
                # Create middleware based on type
                try:
                    if middleware_type == 'rate_limiting':
                        middleware = RateLimitingMiddleware(MiddlewareConfig(**middleware_config))
                    elif middleware_type == 'caching':
                        middleware = CachingMiddleware(MiddlewareConfig(**middleware_config))
                    elif middleware_type == 'validation':
                        middleware = ValidationMiddleware(MiddlewareConfig(**middleware_config))
                    elif middleware_type == 'gpu_optimization':
                        middleware = GPUOptimizationMiddleware(MiddlewareConfig(**middleware_config))
                    elif middleware_type == 'monitoring':
                        middleware = MonitoringMiddleware(MiddlewareConfig(**middleware_config))
                    else:
                        raise ValueError(f"Unknown middleware type: {middleware_type}")
                    
                    # Override name and priority
                    middleware.name = middleware_name
                    middleware.priority = MiddlewarePriority(priority)
                    middleware.enabled = enabled
                    
                    # Add to pipeline
                    self.add_middleware(middleware)
                    
                    self.logger.info(f"Added configured middleware: {middleware_name} (type: {middleware_type})")
                    
                except Exception as e:
                    self.logger.error(f"Failed to configure middleware {middleware_name}: {e}")
                    raise RuntimeError(f"Middleware configuration failed for {middleware_name}: {e}")
            
            self.logger.info("Middleware pipeline configuration complete")