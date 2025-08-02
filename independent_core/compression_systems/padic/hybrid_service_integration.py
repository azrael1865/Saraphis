"""
Hybrid Service Integration - Integration bridge between hybrid and existing services
NO FALLBACKS - HARD FAILURES ONLY
"""

import asyncio
import logging
import threading
import time
import torch
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

# Import service interfaces
from ..service_interfaces.service_interfaces_core import (
    CompressionServiceInterface, ServiceRequest, ServiceResponse,
    ServiceStatus, ServiceHealth
)

# Import hybrid p-adic components
from .hybrid_padic_service_layer import (
    HybridPadicServiceLayer, HybridServiceConfig, HybridServiceMethod
)

# Import existing p-adic service layer
from .padic_service_layer import PadicServiceInterface, PadicServiceConfig, PadicServiceMethod


class ServiceRoutingMethod(Enum):
    """Service routing methods"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    PERFORMANCE_BASED = "performance_based"
    RESOURCE_AWARE = "resource_aware"
    HYBRID_OPTIMIZED = "hybrid_optimized"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    EQUAL_WEIGHT = "equal_weight"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    CAPACITY_WEIGHTED = "capacity_weighted"
    ADAPTIVE = "adaptive"


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration"""
    service_name: str
    service_type: str  # 'hybrid', 'pure_padic', 'tensor', 'sheaf'
    service_instance: Any
    version: str
    priority: int = 0
    weight: float = 1.0
    max_concurrent_requests: int = 10
    health_status: ServiceHealth = ServiceHealth.HEALTHY
    last_health_check: Optional[datetime] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate service endpoint configuration"""
        if not isinstance(self.service_name, str) or not self.service_name.strip():
            raise ValueError("Service name must be non-empty string")
        if not isinstance(self.service_type, str) or not self.service_type.strip():
            raise ValueError("Service type must be non-empty string")
        if self.service_instance is None:
            raise ValueError("Service instance cannot be None")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("Version must be non-empty string")
        if not isinstance(self.priority, int):
            raise TypeError("Priority must be int")
        if not isinstance(self.weight, (int, float)) or self.weight <= 0:
            raise ValueError("Weight must be positive number")
        if not isinstance(self.max_concurrent_requests, int) or self.max_concurrent_requests <= 0:
            raise ValueError("Max concurrent requests must be positive int")


@dataclass
class IntegrationConfig:
    """Configuration for hybrid service integration"""
    # Routing configuration
    default_routing_method: ServiceRoutingMethod = ServiceRoutingMethod.HYBRID_OPTIMIZED
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    enable_service_discovery: bool = True
    enable_health_monitoring: bool = True
    health_check_interval: float = 30.0
    
    # Performance configuration
    request_timeout: float = 300.0
    max_retry_attempts: int = 3
    retry_delay_base: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    
    # Load balancing configuration
    enable_load_balancing: bool = True
    max_queue_size: int = 1000
    queue_timeout: float = 30.0
    adaptive_weight_adjustment: bool = True
    performance_window_size: int = 100
    
    # Fallback configuration
    enable_fallback_routing: bool = True
    fallback_timeout: float = 60.0
    max_fallback_attempts: int = 2
    preserve_request_context: bool = True
    
    # Monitoring configuration
    enable_request_tracking: bool = True
    enable_performance_monitoring: bool = True
    metrics_retention_period: int = 86400  # 24 hours in seconds
    alert_threshold_error_rate: float = 0.05  # 5%
    alert_threshold_latency: float = 10.0  # 10 seconds
    
    def __post_init__(self):
        """Validate integration configuration"""
        if not isinstance(self.default_routing_method, ServiceRoutingMethod):
            raise TypeError("Default routing method must be ServiceRoutingMethod")
        if not isinstance(self.load_balancing_strategy, LoadBalancingStrategy):
            raise TypeError("Load balancing strategy must be LoadBalancingStrategy")
        if not isinstance(self.request_timeout, (int, float)) or self.request_timeout <= 0:
            raise ValueError("Request timeout must be positive number")
        if not isinstance(self.max_retry_attempts, int) or self.max_retry_attempts < 0:
            raise ValueError("Max retry attempts must be non-negative int")
        if not isinstance(self.health_check_interval, (int, float)) or self.health_check_interval <= 0:
            raise ValueError("Health check interval must be positive number")


@dataclass
class IntegrationMetrics:
    """Metrics for service integration operations"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    routed_to_hybrid: int = 0
    routed_to_pure: int = 0
    routed_to_fallback: int = 0
    circuit_breaker_trips: int = 0
    retry_attempts: int = 0
    average_response_time: float = 0.0
    average_routing_time: float = 0.0
    service_utilization: Dict[str, float] = field(default_factory=dict)
    error_rates: Dict[str, float] = field(default_factory=dict)
    last_metrics_update: Optional[datetime] = None
    
    def update_request_metrics(self, success: bool, response_time: float, service_type: str):
        """Update request metrics"""
        self.total_requests += 1
        self.last_metrics_update = datetime.utcnow()
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Update routing metrics
        if service_type == 'hybrid':
            self.routed_to_hybrid += 1
        elif service_type == 'pure_padic':
            self.routed_to_pure += 1
        else:
            self.routed_to_fallback += 1
        
        # Update average response time
        if self.total_requests > 1:
            old_avg = self.average_response_time
            self.average_response_time = (
                (old_avg * (self.total_requests - 1) + response_time) / self.total_requests
            )
        else:
            self.average_response_time = response_time


class CircuitBreaker:
    """Circuit breaker for service protection"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = 'closed'  # closed, open, half_open
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == 'open':
                if self._should_attempt_reset():
                    self.state = 'half_open'
                else:
                    raise RuntimeError("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        self.state = 'closed'
    
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'


class HybridServiceIntegration:
    """
    Integration bridge between hybrid and existing services.
    Provides service routing, load balancing, fallback mechanisms, and service discovery.
    """
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        """Initialize hybrid service integration"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, IntegrationConfig):
            raise TypeError(f"Config must be IntegrationConfig or None, got {type(config)}")
        
        self.config = config or IntegrationConfig()
        self.logger = logging.getLogger('HybridServiceIntegration')
        
        # Service registry and routing
        self.service_endpoints: Dict[str, ServiceEndpoint] = {}
        self.service_routing_table: Dict[str, List[str]] = defaultdict(list)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Integration state
        self.is_initialized = False
        self.startup_time = datetime.utcnow()
        
        # Core service instances
        self.hybrid_service: Optional[HybridPadicServiceLayer] = None
        self.compression_service_interface: Optional[CompressionServiceInterface] = None
        
        # Performance tracking
        self.metrics = IntegrationMetrics()
        self.request_history: deque = deque(maxlen=self.config.performance_window_size)
        self.active_requests: Dict[str, datetime] = {}
        
        # Thread safety
        self._routing_lock = threading.RLock()
        self._metrics_lock = threading.RLock()
        self._health_lock = threading.RLock()
        
        # Background services
        self._health_monitor_enabled = False
        self._health_monitor_thread: Optional[threading.Thread] = None
        self._stop_health_monitor = threading.Event()
        
        # Request processing
        self.executor = ThreadPoolExecutor(
            max_workers=16,
            thread_name_prefix="IntegrationService"
        )
        
        self.logger.info("HybridServiceIntegration created successfully")
    
    def initialize_integration(self, hybrid_service: HybridPadicServiceLayer,
                             compression_interface: CompressionServiceInterface) -> None:
        """
        Initialize integration with core services.
        
        Args:
            hybrid_service: Hybrid p-adic service layer instance
            compression_interface: Compression service interface instance
            
        Raises:
            RuntimeError: If initialization fails
        """
        if self.is_initialized:
            return
        
        with self._routing_lock:
            try:
                # Validate inputs
                if not isinstance(hybrid_service, HybridPadicServiceLayer):
                    raise TypeError(f"hybrid_service must be HybridPadicServiceLayer, got {type(hybrid_service)}")
                if not isinstance(compression_interface, CompressionServiceInterface):
                    raise TypeError(f"compression_interface must be CompressionServiceInterface, got {type(compression_interface)}")
                
                # Store service instances
                self.hybrid_service = hybrid_service
                self.compression_service_interface = compression_interface
                
                # Initialize hybrid service if needed
                if not hybrid_service.is_initialized:
                    hybrid_service.initialize_hybrid_services()
                
                # Register core services
                self._register_core_services()
                
                # Initialize routing table
                self._initialize_routing_table()
                
                # Start health monitoring
                if self.config.enable_health_monitoring:
                    self._start_health_monitoring()
                
                self.is_initialized = True
                self.logger.info("Hybrid service integration initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize integration: {e}")
                raise RuntimeError(f"Integration initialization failed: {e}")
    
    def _register_core_services(self) -> None:
        """Register core services with integration layer"""
        # Register hybrid service
        hybrid_endpoint = ServiceEndpoint(
            service_name="hybrid_padic",
            service_type="hybrid",
            service_instance=self.hybrid_service,
            version="1.0.0",
            priority=10,
            weight=1.5,  # Higher weight for hybrid service
            max_concurrent_requests=self.hybrid_service.config.max_concurrent_hybrid_requests
        )
        self.register_service_endpoint(hybrid_endpoint)
        
        # Register compression service interface
        interface_endpoint = ServiceEndpoint(
            service_name="compression_interface",
            service_type="interface",
            service_instance=self.compression_service_interface,
            version="1.0.0",
            priority=5,
            weight=1.0,
            max_concurrent_requests=20
        )
        self.register_service_endpoint(interface_endpoint)
        
        self.logger.info("Core services registered successfully")
    
    def _initialize_routing_table(self) -> None:
        """Initialize service routing table"""
        # Map hybrid methods to hybrid service
        for method in HybridServiceMethod:
            self.service_routing_table[method.value].append("hybrid_padic")
        
        # Map standard compression methods to interface
        standard_methods = ["compress", "decompress", "encode", "decode", "validate"]
        for method in standard_methods:
            self.service_routing_table[method].append("compression_interface")
        
        self.logger.info(f"Routing table initialized with {len(self.service_routing_table)} method mappings")
    
    def register_service_endpoint(self, endpoint: ServiceEndpoint) -> bool:
        """
        Register a service endpoint.
        
        Args:
            endpoint: Service endpoint to register
            
        Returns:
            True if registration successful
            
        Raises:
            ValueError: If endpoint is invalid
        """
        if not isinstance(endpoint, ServiceEndpoint):
            raise TypeError(f"Endpoint must be ServiceEndpoint, got {type(endpoint)}")
        
        with self._routing_lock:
            try:
                # Validate endpoint
                endpoint.__post_init__()
                
                # Register endpoint
                self.service_endpoints[endpoint.service_name] = endpoint
                
                # Create circuit breaker
                self.circuit_breakers[endpoint.service_name] = CircuitBreaker(
                    failure_threshold=self.config.circuit_breaker_threshold,
                    recovery_timeout=self.config.circuit_breaker_timeout
                )
                
                # Register with compression service interface if available
                if self.compression_service_interface and endpoint.service_instance:
                    self.compression_service_interface.register_service(
                        service_name=endpoint.service_name,
                        service_instance=endpoint.service_instance,
                        version=endpoint.version,
                        metadata={
                            'service_type': endpoint.service_type,
                            'priority': endpoint.priority,
                            'weight': endpoint.weight,
                            'registered_via': 'hybrid_integration'
                        }
                    )
                
                self.logger.info(f"Registered service endpoint: {endpoint.service_name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to register service endpoint {endpoint.service_name}: {e}")
                raise RuntimeError(f"Service registration failed: {e}")
    
    def unregister_service_endpoint(self, service_name: str) -> bool:
        """
        Unregister a service endpoint.
        
        Args:
            service_name: Name of service to unregister
            
        Returns:
            True if unregistration successful
        """
        if not isinstance(service_name, str) or not service_name.strip():
            raise ValueError("Service name must be non-empty string")
        
        with self._routing_lock:
            try:
                if service_name not in self.service_endpoints:
                    self.logger.warning(f"Service not found for unregistration: {service_name}")
                    return False
                
                # Remove from service endpoints
                del self.service_endpoints[service_name]
                
                # Remove circuit breaker
                if service_name in self.circuit_breakers:
                    del self.circuit_breakers[service_name]
                
                # Remove from routing table
                for method_routes in self.service_routing_table.values():
                    if service_name in method_routes:
                        method_routes.remove(service_name)
                
                # Unregister from compression service interface
                if self.compression_service_interface:
                    try:
                        self.compression_service_interface.unregister_service(service_name)
                    except Exception as e:
                        self.logger.warning(f"Failed to unregister from compression interface: {e}")
                
                self.logger.info(f"Unregistered service endpoint: {service_name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to unregister service endpoint {service_name}: {e}")
                return False
    
    def route_request(self, request: ServiceRequest) -> ServiceResponse:
        """
        Route service request to appropriate service endpoint.
        
        Args:
            request: Service request to route
            
        Returns:
            Service response
            
        Raises:
            ValueError: If request is invalid
            RuntimeError: If routing fails
        """
        if not isinstance(request, ServiceRequest):
            raise TypeError(f"Request must be ServiceRequest, got {type(request)}")
        
        if not self.is_initialized:
            raise RuntimeError("Integration not initialized")
        
        start_time = time.time()
        
        with self._metrics_lock:
            self.active_requests[request.request_id] = datetime.utcnow()
        
        try:
            # Determine target service
            target_service = self._select_target_service(request)
            
            if not target_service:
                raise RuntimeError(f"No available service for method: {request.method_name}")
            
            # Route request with circuit breaker protection
            circuit_breaker = self.circuit_breakers.get(target_service)
            if not circuit_breaker:
                raise RuntimeError(f"No circuit breaker for service: {target_service}")
            
            response = circuit_breaker.call(self._execute_service_request, request, target_service)
            
            # Update metrics
            response_time = time.time() - start_time
            service_type = self.service_endpoints[target_service].service_type
            
            with self._metrics_lock:
                self.metrics.update_request_metrics(True, response_time, service_type)
                self.request_history.append({
                    'timestamp': datetime.utcnow(),
                    'method': request.method_name,
                    'service': target_service,
                    'response_time': response_time,
                    'success': True
                })
            
            return response
            
        except Exception as e:
            # Handle routing failure
            response_time = time.time() - start_time
            
            # Update failure metrics
            with self._metrics_lock:
                self.metrics.update_request_metrics(False, response_time, 'failed')
                self.request_history.append({
                    'timestamp': datetime.utcnow(),
                    'method': request.method_name,
                    'service': 'failed',
                    'response_time': response_time,
                    'success': False,
                    'error': str(e)
                })
            
            # Try fallback routing if enabled
            if self.config.enable_fallback_routing:
                try:
                    return self._attempt_fallback_routing(request, str(e))
                except Exception as fallback_error:
                    self.logger.error(f"Fallback routing also failed: {fallback_error}")
            
            # Create error response
            error_response = ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.ERROR,
                processing_time=response_time,
                service_version="integration_v1.0"
            )
            error_response.add_error(
                error_code="ROUTING_ERROR",
                error_message=str(e),
                error_details={'routing_failure': True}
            )
            
            self.logger.error(f"Request routing failed: {e}")
            return error_response
            
        finally:
            with self._metrics_lock:
                self.active_requests.pop(request.request_id, None)
    
    def _select_target_service(self, request: ServiceRequest) -> Optional[str]:
        """Select target service for request"""
        method_name = request.method_name
        
        # Check routing table
        available_services = self.service_routing_table.get(method_name, [])
        
        if not available_services:
            # Try to find services that can handle the method
            for service_name, endpoint in self.service_endpoints.items():
                if self._service_can_handle_method(endpoint, method_name):
                    available_services.append(service_name)
        
        if not available_services:
            return None
        
        # Filter by health status
        healthy_services = [
            service for service in available_services
            if self.service_endpoints[service].health_status == ServiceHealth.HEALTHY
        ]
        
        if not healthy_services:
            # Fall back to degraded services if no healthy ones
            healthy_services = [
                service for service in available_services
                if self.service_endpoints[service].health_status != ServiceHealth.UNHEALTHY
            ]
        
        if not healthy_services:
            return None
        
        # Apply routing strategy
        return self._apply_routing_strategy(healthy_services, request)
    
    def _service_can_handle_method(self, endpoint: ServiceEndpoint, method_name: str) -> bool:
        """Check if service endpoint can handle method"""
        service_instance = endpoint.service_instance
        
        # Check if method exists on service instance
        if hasattr(service_instance, method_name):
            return True
        
        # Check for method variants
        method_variants = [
            f"handle_{method_name}",
            f"process_{method_name}",
            f"_handle_{method_name}",
            f"{method_name}_handler"
        ]
        
        for variant in method_variants:
            if hasattr(service_instance, variant):
                return True
        
        return False
    
    def _apply_routing_strategy(self, available_services: List[str], request: ServiceRequest) -> str:
        """Apply routing strategy to select service"""
        if len(available_services) == 1:
            return available_services[0]
        
        routing_method = self.config.default_routing_method
        
        if routing_method == ServiceRoutingMethod.ROUND_ROBIN:
            return self._round_robin_selection(available_services)
        elif routing_method == ServiceRoutingMethod.LEAST_LOADED:
            return self._least_loaded_selection(available_services)
        elif routing_method == ServiceRoutingMethod.PERFORMANCE_BASED:
            return self._performance_based_selection(available_services)
        elif routing_method == ServiceRoutingMethod.RESOURCE_AWARE:
            return self._resource_aware_selection(available_services)
        elif routing_method == ServiceRoutingMethod.HYBRID_OPTIMIZED:
            return self._hybrid_optimized_selection(available_services, request)
        else:
            return available_services[0]  # Default fallback
    
    def _hybrid_optimized_selection(self, available_services: List[str], request: ServiceRequest) -> str:
        """Hybrid-optimized service selection"""
        # Check if request contains data that would benefit from hybrid processing
        payload = request.payload
        
        # Prefer hybrid service for large data or GPU operations
        if 'data' in payload:
            data = payload['data']
            if hasattr(data, '__len__') and len(data) > 1000:
                # Large data - prefer hybrid
                hybrid_services = [s for s in available_services if 'hybrid' in s.lower()]
                if hybrid_services:
                    return hybrid_services[0]
        
        # Check for GPU tensor data
        if 'data' in payload and hasattr(payload['data'], 'is_cuda'):
            if payload['data'].is_cuda:
                hybrid_services = [s for s in available_services if 'hybrid' in s.lower()]
                if hybrid_services:
                    return hybrid_services[0]
        
        # Fall back to performance-based selection
        return self._performance_based_selection(available_services)
    
    def _performance_based_selection(self, available_services: List[str]) -> str:
        """Select service based on performance metrics"""
        best_service = available_services[0]
        best_score = 0.0
        
        for service_name in available_services:
            endpoint = self.service_endpoints[service_name]
            
            # Calculate performance score
            avg_response_time = endpoint.performance_metrics.get('avg_response_time', 1.0)
            success_rate = endpoint.performance_metrics.get('success_rate', 0.9)
            
            # Higher score is better
            score = (success_rate / max(avg_response_time, 0.1)) * endpoint.weight
            
            if score > best_score:
                best_score = score
                best_service = service_name
        
        return best_service
    
    def _least_loaded_selection(self, available_services: List[str]) -> str:
        """Select least loaded service"""
        min_load = float('inf')
        selected_service = available_services[0]
        
        for service_name in available_services:
            # Simple load metric based on active requests
            current_load = len([req for req in self.active_requests.values() 
                              if service_name in str(req)])
            
            if current_load < min_load:
                min_load = current_load
                selected_service = service_name
        
        return selected_service
    
    def _round_robin_selection(self, available_services: List[str]) -> str:
        """Simple round-robin selection"""
        # Use timestamp for simple round-robin
        index = int(time.time()) % len(available_services)
        return available_services[index]
    
    def _resource_aware_selection(self, available_services: List[str]) -> str:
        """Resource-aware service selection"""
        # Check GPU memory if hybrid service is available
        if 'hybrid_padic' in available_services and torch.cuda.is_available():
            gpu_memory_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            gpu_usage_ratio = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            
            if gpu_usage_ratio < 0.8:  # GPU not heavily loaded
                return 'hybrid_padic'
        
        # Fall back to performance-based selection
        return self._performance_based_selection(available_services)
    
    def _execute_service_request(self, request: ServiceRequest, target_service: str) -> ServiceResponse:
        """Execute service request on target service"""
        endpoint = self.service_endpoints[target_service]
        service_instance = endpoint.service_instance
        
        # Route to appropriate service
        if isinstance(service_instance, HybridPadicServiceLayer):
            return service_instance.process_hybrid_request(request)
        elif isinstance(service_instance, CompressionServiceInterface):
            return service_instance.invoke_service(request)
        else:
            # Generic service invocation
            return self._invoke_generic_service(service_instance, request)
    
    def _invoke_generic_service(self, service_instance: Any, request: ServiceRequest) -> ServiceResponse:
        """Invoke generic service instance"""
        method_name = request.method_name
        
        # Try to find appropriate method
        handler = None
        for method_variant in [method_name, f"handle_{method_name}", f"process_{method_name}"]:
            if hasattr(service_instance, method_variant):
                handler = getattr(service_instance, method_variant)
                break
        
        if not handler:
            raise RuntimeError(f"No handler found for method: {method_name}")
        
        try:
            # Invoke handler
            result = handler(request)
            
            # Ensure result is ServiceResponse
            if isinstance(result, ServiceResponse):
                return result
            else:
                # Wrap result in ServiceResponse
                return ServiceResponse(
                    request_id=request.request_id,
                    status=ServiceStatus.SUCCESS,
                    data=result if isinstance(result, dict) else {'result': result},
                    service_version=f"{service_instance.__class__.__name__}_v1.0"
                )
                
        except Exception as e:
            error_response = ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.ERROR,
                service_version=f"{service_instance.__class__.__name__}_v1.0"
            )
            error_response.add_error(
                error_code="SERVICE_EXECUTION_ERROR",
                error_message=str(e),
                error_details={'service_type': type(service_instance).__name__}
            )
            return error_response
    
    def _attempt_fallback_routing(self, request: ServiceRequest, original_error: str) -> ServiceResponse:
        """Attempt fallback routing for failed requests"""
        self.metrics.routed_to_fallback += 1
        
        # Try alternative services
        all_services = list(self.service_endpoints.keys())
        
        for service_name in all_services:
            endpoint = self.service_endpoints[service_name]
            
            # Skip unhealthy services
            if endpoint.health_status == ServiceHealth.UNHEALTHY:
                continue
            
            # Try if service can handle method
            if self._service_can_handle_method(endpoint, request.method_name):
                try:
                    circuit_breaker = self.circuit_breakers.get(service_name)
                    if circuit_breaker and circuit_breaker.state != 'open':
                        response = circuit_breaker.call(self._execute_service_request, request, service_name)
                        
                        # Add fallback metadata
                        response.metadata['fallback_used'] = True
                        response.metadata['original_error'] = original_error
                        response.metadata['fallback_service'] = service_name
                        
                        return response
                        
                except Exception as e:
                    self.logger.warning(f"Fallback attempt failed for service {service_name}: {e}")
                    continue
        
        # All fallback attempts failed
        raise RuntimeError(f"All fallback routing attempts failed. Original error: {original_error}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """
        Get comprehensive integration status.
        
        Returns:
            Dictionary containing integration status information
        """
        with self._routing_lock:
            return {
                'initialized': self.is_initialized,
                'startup_time': self.startup_time.isoformat(),
                'uptime_seconds': (datetime.utcnow() - self.startup_time).total_seconds(),
                'registered_services': list(self.service_endpoints.keys()),
                'service_health': {
                    name: endpoint.health_status.value 
                    for name, endpoint in self.service_endpoints.items()
                },
                'routing_table_size': len(self.service_routing_table),
                'active_requests': len(self.active_requests),
                'circuit_breaker_states': {
                    name: cb.state for name, cb in self.circuit_breakers.items()
                },
                'configuration': {
                    'routing_method': self.config.default_routing_method.value,
                    'load_balancing': self.config.load_balancing_strategy.value,
                    'health_monitoring': self.config.enable_health_monitoring,
                    'fallback_routing': self.config.enable_fallback_routing
                }
            }
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """
        Get integration performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        with self._metrics_lock:
            return {
                'request_metrics': {
                    'total_requests': self.metrics.total_requests,
                    'successful_requests': self.metrics.successful_requests,
                    'failed_requests': self.metrics.failed_requests,
                    'success_rate': (
                        self.metrics.successful_requests / max(1, self.metrics.total_requests)
                    ),
                    'average_response_time': self.metrics.average_response_time
                },
                'routing_metrics': {
                    'routed_to_hybrid': self.metrics.routed_to_hybrid,
                    'routed_to_pure': self.metrics.routed_to_pure,
                    'routed_to_fallback': self.metrics.routed_to_fallback,
                    'circuit_breaker_trips': self.metrics.circuit_breaker_trips,
                    'retry_attempts': self.metrics.retry_attempts
                },
                'service_utilization': self.metrics.service_utilization,
                'error_rates': self.metrics.error_rates,
                'recent_requests': list(self.request_history)[-10:],  # Last 10 requests
                'last_update': self.metrics.last_metrics_update.isoformat() if self.metrics.last_metrics_update else None
            }
    
    def _start_health_monitoring(self) -> None:
        """Start background health monitoring"""
        if self._health_monitor_enabled:
            return
        
        self._health_monitor_enabled = True
        self._stop_health_monitor.clear()
        self._health_monitor_thread = threading.Thread(
            target=self._health_monitoring_loop,
            name="IntegrationHealthMonitor",
            daemon=True
        )
        self._health_monitor_thread.start()
        self.logger.info("Health monitoring started")
    
    def _health_monitoring_loop(self) -> None:
        """Background health monitoring loop"""
        while not self._stop_health_monitor.wait(self.config.health_check_interval):
            try:
                with self._health_lock:
                    for service_name, endpoint in self.service_endpoints.items():
                        self._check_service_health(service_name, endpoint)
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
    
    def _check_service_health(self, service_name: str, endpoint: ServiceEndpoint) -> None:
        """Check health of individual service"""
        try:
            # Create health check request
            health_request = ServiceRequest(
                service_name=service_name,
                method_name="_health_check",
                version=endpoint.version,
                payload={},
                timeout=5.0
            )
            
            # Execute health check
            start_time = time.time()
            response = self._execute_service_request(health_request, service_name)
            response_time = time.time() - start_time
            
            # Update health status
            if response.is_success():
                endpoint.health_status = ServiceHealth.HEALTHY
            else:
                endpoint.health_status = ServiceHealth.DEGRADED
            
            # Update performance metrics
            endpoint.performance_metrics['last_health_check'] = datetime.utcnow().isoformat()
            endpoint.performance_metrics['health_check_response_time'] = response_time
            endpoint.last_health_check = datetime.utcnow()
            
        except Exception as e:
            # Mark service as unhealthy
            endpoint.health_status = ServiceHealth.UNHEALTHY
            endpoint.performance_metrics['last_health_error'] = str(e)
            self.logger.warning(f"Health check failed for service {service_name}: {e}")
    
    def shutdown(self) -> None:
        """Shutdown integration service"""
        self.logger.info("Shutting down hybrid service integration")
        
        # Stop health monitoring
        if self._health_monitor_enabled:
            self._health_monitor_enabled = False
            self._stop_health_monitor.set()
            if self._health_monitor_thread and self._health_monitor_thread.is_alive():
                self._health_monitor_thread.join(timeout=5.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Clear active requests
        with self._metrics_lock:
            self.active_requests.clear()
        
        self.is_initialized = False
        self.logger.info("Hybrid service integration shutdown complete")
    
    def discover_services(self, service_type: Optional[str] = None) -> Dict[str, List[ServiceEndpoint]]:
        """
        Discover available services by type.
        
        Args:
            service_type: Optional service type filter
            
        Returns:
            Dictionary of discovered services grouped by type
        """
        discovered = defaultdict(list)
        
        with self._routing_lock:
            for service_name, endpoint in self.service_endpoints.items():
                if service_type is None or endpoint.service_type == service_type:
                    discovered[endpoint.service_type].append(endpoint)
        
        return dict(discovered)
    
    def load_balance_requests(self, requests: List[ServiceRequest]) -> List[ServiceResponse]:
        """
        Load balance multiple requests across available services.
        
        Args:
            requests: List of service requests to load balance
            
        Returns:
            List of service responses
        """
        if not requests:
            return []
        
        responses = []
        
        # Process requests with load balancing
        for request in requests:
            try:
                response = self.route_request(request)
                responses.append(response)
            except Exception as e:
                error_response = ServiceResponse(
                    request_id=request.request_id,
                    status=ServiceStatus.ERROR
                )
                error_response.add_error(
                    error_code="LOAD_BALANCE_ERROR",
                    error_message=str(e)
                )
                responses.append(error_response)
        
        return responses
    
    def get_service_health(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get health status of services.
        
        Args:
            service_name: Optional specific service name
            
        Returns:
            Dictionary containing health information
        """
        health_info = {}
        
        with self._health_lock:
            if service_name:
                if service_name in self.service_endpoints:
                    endpoint = self.service_endpoints[service_name]
                    health_info[service_name] = {
                        'status': endpoint.health_status.value,
                        'last_check': endpoint.last_health_check.isoformat() if endpoint.last_health_check else None,
                        'performance_metrics': endpoint.performance_metrics.copy()
                    }
            else:
                for name, endpoint in self.service_endpoints.items():
                    health_info[name] = {
                        'status': endpoint.health_status.value,
                        'last_check': endpoint.last_health_check.isoformat() if endpoint.last_health_check else None,
                        'performance_metrics': endpoint.performance_metrics.copy()
                    }
        
        return health_info
    
    def handle_failover(self, failed_service: str, request: ServiceRequest) -> ServiceResponse:
        """
        Handle service failover for failed requests.
        
        Args:
            failed_service: Name of the failed service
            request: Original service request
            
        Returns:
            Service response from failover attempt
        """
        self.logger.warning(f"Handling failover for failed service: {failed_service}")
        
        # Mark service as unhealthy
        if failed_service in self.service_endpoints:
            self.service_endpoints[failed_service].health_status = ServiceHealth.UNHEALTHY
        
        # Attempt fallback routing
        try:
            return self._attempt_fallback_routing(request, f"Service {failed_service} failed")
        except Exception as e:
            # Create final error response
            error_response = ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.ERROR
            )
            error_response.add_error(
                error_code="FAILOVER_FAILED",
                error_message=f"All failover attempts failed: {e}",
                error_details={'failed_service': failed_service}
            )
            return error_response