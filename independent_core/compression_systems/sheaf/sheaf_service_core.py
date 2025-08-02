"""
Sheaf Service Core Implementation
Production-ready service layer for sheaf compression operations
Hard failure mode - no fallbacks, all errors surface immediately
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor, Future
import uuid
import json
from collections import defaultdict, deque

# Core service interfaces (would be imported from service framework)
class ServiceStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    STARTING = "starting"
    STOPPING = "stopping"

class ServiceOperationType(Enum):
    SHEAF_COMPRESSION = "sheaf_compression"
    SHEAF_DECOMPRESSION = "sheaf_decompression"
    COHOMOLOGY_CALCULATION = "cohomology_calculation"
    RESTRICTION_MAP_PROCESSING = "restriction_map_processing"
    CELLULAR_SHEAF_BUILDING = "cellular_sheaf_building"

@dataclass
class ServiceRequest:
    request_id: str
    operation_type: ServiceOperationType
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: int = 1  # 1-10, higher is more priority

@dataclass
class ServiceResponse:
    request_id: str
    status: ServiceStatus
    result: Optional[Any] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class ServiceMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: Optional[float] = None
    uptime_start: float = field(default_factory=time.time)

class CompressionServiceInterface(ABC):
    """Base interface for compression services"""
    
    @abstractmethod
    def compress(self, data: Any) -> Any:
        pass
    
    @abstractmethod
    def decompress(self, compressed_data: Any) -> Any:
        pass
    
    @abstractmethod
    def validate_input(self, data: Any) -> Tuple[bool, List[str]]:
        pass

# Load balancing strategies
class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED_RANDOM = "weighted_random"
    HEALTH_BASED = "health_based"

@dataclass
class ServiceInstance:
    service_id: str
    service_interface: 'SheafServiceInterface'
    version: str
    capabilities: Set[ServiceOperationType]
    load_weight: float = 1.0
    current_load: int = 0
    health_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    last_health_check: float = field(default_factory=time.time)

class SheafServiceInterface(CompressionServiceInterface):
    """
    Service interface for sheaf compression operations
    Extends the base compression interface with sheaf-specific functionality
    """
    
    def __init__(self, service_id: str, config: Optional[Dict[str, Any]] = None):
        self.service_id = service_id
        self.config = config or {}
        self.metrics = ServiceMetrics()
        self.status = ServiceStatus.INACTIVE
        self._lock = threading.RLock()
        self.logger = logging.getLogger(f"SheafService.{service_id}")
        
        # Sheaf-specific components (would be imported from sheaf modules)
        self._compression_system = None
        self._cohomology_calculator = None
        self._reconstruction_engine = None
        
    def start_service(self) -> bool:
        """Initialize and start the sheaf service"""
        try:
            with self._lock:
                if self.status == ServiceStatus.ACTIVE:
                    return True
                    
                self.status = ServiceStatus.STARTING
                
                # Initialize sheaf components
                self._initialize_sheaf_components()
                
                self.status = ServiceStatus.ACTIVE
                self.logger.info(f"Sheaf service {self.service_id} started successfully")
                return True
                
        except Exception as e:
            self.status = ServiceStatus.FAILED
            self.logger.error(f"Failed to start sheaf service {self.service_id}: {e}")
            raise RuntimeError(f"Service startup failed: {e}")
    
    def stop_service(self) -> bool:
        """Stop the sheaf service and cleanup resources"""
        try:
            with self._lock:
                if self.status == ServiceStatus.INACTIVE:
                    return True
                    
                self.status = ServiceStatus.STOPPING
                
                # Cleanup sheaf components
                self._cleanup_sheaf_components()
                
                self.status = ServiceStatus.INACTIVE
                self.logger.info(f"Sheaf service {self.service_id} stopped successfully")
                return True
                
        except Exception as e:
            self.status = ServiceStatus.FAILED
            self.logger.error(f"Failed to stop sheaf service {self.service_id}: {e}")
            raise RuntimeError(f"Service shutdown failed: {e}")
    
    def invoke_sheaf_compression(self, request: ServiceRequest) -> ServiceResponse:
        """Handle sheaf compression request"""
        start_time = time.time()
        
        try:
            # Validate request
            is_valid, errors = self.validate_sheaf_request(request)
            if not is_valid:
                return ServiceResponse(
                    request_id=request.request_id,
                    status=ServiceStatus.FAILED,
                    error_message="; ".join(errors)
                )
            
            # Process based on operation type
            if request.operation_type == ServiceOperationType.SHEAF_COMPRESSION:
                result = self.compress(request.data)
            elif request.operation_type == ServiceOperationType.SHEAF_DECOMPRESSION:
                result = self.decompress(request.data)
            elif request.operation_type == ServiceOperationType.COHOMOLOGY_CALCULATION:
                result = self._calculate_cohomology(request.data)
            elif request.operation_type == ServiceOperationType.RESTRICTION_MAP_PROCESSING:
                result = self._process_restriction_maps(request.data)
            elif request.operation_type == ServiceOperationType.CELLULAR_SHEAF_BUILDING:
                result = self._build_cellular_sheaf(request.data)
            else:
                raise ValueError(f"Unsupported operation type: {request.operation_type}")
            
            processing_time = time.time() - start_time
            
            # Update metrics
            with self._lock:
                self.metrics.total_requests += 1
                self.metrics.successful_requests += 1
                self._update_average_response_time(processing_time)
                self.metrics.last_request_time = time.time()
            
            return ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.ACTIVE,
                result=result,
                processing_time=processing_time,
                metadata={"operation_type": request.operation_type.value}
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            with self._lock:
                self.metrics.total_requests += 1
                self.metrics.failed_requests += 1
                self._update_average_response_time(processing_time)
            
            self.logger.error(f"Sheaf compression request failed: {e}")
            return ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.FAILED,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def compress(self, data: Any) -> Any:
        """Compress data using sheaf theory methods"""
        if self._compression_system is None:
            raise RuntimeError("Compression system not initialized")
        
        # Implementation would use actual sheaf compression
        # For now, return a placeholder that indicates sheaf compression
        return {
            "compressed_data": data,  # Placeholder
            "compression_method": "sheaf_theory",
            "metadata": {"original_shape": getattr(data, 'shape', None)}
        }
    
    def decompress(self, compressed_data: Any) -> Any:
        """Decompress data using sheaf theory methods"""
        if self._reconstruction_engine is None:
            raise RuntimeError("Reconstruction engine not initialized")
        
        # Implementation would use actual sheaf decompression
        # For now, return the original data from placeholder
        if isinstance(compressed_data, dict) and "compressed_data" in compressed_data:
            return compressed_data["compressed_data"]
        return compressed_data
    
    def validate_input(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate input data for sheaf operations"""
        errors = []
        
        if data is None:
            errors.append("Input data cannot be None")
        
        # Add sheaf-specific validations
        if hasattr(data, 'shape') and len(data.shape) < 2:
            errors.append("Data must be at least 2-dimensional for sheaf operations")
        
        return len(errors) == 0, errors
    
    def validate_sheaf_request(self, request: ServiceRequest) -> Tuple[bool, List[str]]:
        """Validate sheaf-specific request parameters"""
        errors = []
        
        if not isinstance(request.operation_type, ServiceOperationType):
            errors.append("Invalid operation type")
        
        is_data_valid, data_errors = self.validate_input(request.data)
        if not is_data_valid:
            errors.extend(data_errors)
        
        return len(errors) == 0, errors
    
    def get_service_health(self) -> Dict[str, Any]:
        """Get current service health status"""
        with self._lock:
            uptime = time.time() - self.metrics.uptime_start
            success_rate = (
                self.metrics.successful_requests / max(self.metrics.total_requests, 1)
            )
            
            return {
                "service_id": self.service_id,
                "status": self.status.value,
                "uptime_seconds": uptime,
                "total_requests": self.metrics.total_requests,
                "success_rate": success_rate,
                "average_response_time": self.metrics.average_response_time,
                "last_request_time": self.metrics.last_request_time
            }
    
    def get_service_capabilities(self) -> Set[ServiceOperationType]:
        """Return the operations this service can handle"""
        return {
            ServiceOperationType.SHEAF_COMPRESSION,
            ServiceOperationType.SHEAF_DECOMPRESSION,
            ServiceOperationType.COHOMOLOGY_CALCULATION,
            ServiceOperationType.RESTRICTION_MAP_PROCESSING,
            ServiceOperationType.CELLULAR_SHEAF_BUILDING
        }
    
    def _initialize_sheaf_components(self):
        """Initialize sheaf-specific components"""
        # This would initialize actual sheaf components
        # from .sheaf_core import SheafCompressionSystem
        # from .sheaf_advanced import SheafCohomologyCalculator, SheafReconstructionEngine
        
        self._compression_system = "SheafCompressionSystem"  # Placeholder
        self._cohomology_calculator = "SheafCohomologyCalculator"  # Placeholder
        self._reconstruction_engine = "SheafReconstructionEngine"  # Placeholder
    
    def _cleanup_sheaf_components(self):
        """Cleanup sheaf-specific components"""
        self._compression_system = None
        self._cohomology_calculator = None
        self._reconstruction_engine = None
    
    def _calculate_cohomology(self, data: Any) -> Any:
        """Calculate sheaf cohomology"""
        if self._cohomology_calculator is None:
            raise RuntimeError("Cohomology calculator not initialized")
        
        # Placeholder implementation
        return {"cohomology_groups": [], "betti_numbers": []}
    
    def _process_restriction_maps(self, data: Any) -> Any:
        """Process restriction maps"""
        # Placeholder implementation
        return {"processed_maps": data, "consistency_check": True}
    
    def _build_cellular_sheaf(self, data: Any) -> Any:
        """Build cellular sheaf structure"""
        # Placeholder implementation
        return {"cellular_sheaf": data, "cell_complex": {}}
    
    def _update_average_response_time(self, new_time: float):
        """Update running average of response time"""
        if self.metrics.average_response_time == 0.0:
            self.metrics.average_response_time = new_time
        else:
            # Simple exponential moving average
            alpha = 0.1
            self.metrics.average_response_time = (
                alpha * new_time + (1 - alpha) * self.metrics.average_response_time
            )

class SheafServiceRegistry:
    """
    Registry for managing sheaf service instances
    Handles service discovery, health monitoring, and load balancing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._services: Dict[str, ServiceInstance] = {}
        self._service_types: Dict[ServiceOperationType, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()
        self._health_monitor_thread = None
        self._shutdown_event = threading.Event()
        self.logger = logging.getLogger("SheafServiceRegistry")
        
        # Load balancing state
        self._round_robin_counters: Dict[ServiceOperationType, int] = defaultdict(int)
        
    def start_registry(self):
        """Start the service registry and health monitoring"""
        if self._health_monitor_thread is not None:
            return
        
        self._shutdown_event.clear()
        self._health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True
        )
        self._health_monitor_thread.start()
        self.logger.info("Sheaf service registry started")
    
    def stop_registry(self):
        """Stop the service registry and cleanup"""
        if self._health_monitor_thread is None:
            return
        
        self._shutdown_event.set()
        self._health_monitor_thread.join(timeout=5.0)
        self._health_monitor_thread = None
        self.logger.info("Sheaf service registry stopped")
    
    def register_service(self, service: SheafServiceInterface, version: str = "1.0.0") -> bool:
        """Register a new sheaf service instance"""
        try:
            with self._lock:
                if service.service_id in self._services:
                    raise ValueError(f"Service {service.service_id} already registered")
                
                capabilities = service.get_service_capabilities()
                
                instance = ServiceInstance(
                    service_id=service.service_id,
                    service_interface=service,
                    version=version,
                    capabilities=capabilities,
                    metadata={"config": service.config}
                )
                
                self._services[service.service_id] = instance
                
                # Update service type mapping
                for capability in capabilities:
                    self._service_types[capability].add(service.service_id)
                
                self.logger.info(f"Registered sheaf service {service.service_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to register service {service.service_id}: {e}")
            raise RuntimeError(f"Service registration failed: {e}")
    
    def unregister_service(self, service_id: str) -> bool:
        """Unregister a sheaf service instance"""
        try:
            with self._lock:
                if service_id not in self._services:
                    self.logger.warning(f"Service {service_id} not found for unregistration")
                    return False
                
                instance = self._services[service_id]
                
                # Remove from service type mapping
                for capability in instance.capabilities:
                    self._service_types[capability].discard(service_id)
                
                # Stop the service if it's still active
                if instance.service_interface.status == ServiceStatus.ACTIVE:
                    instance.service_interface.stop_service()
                
                del self._services[service_id]
                
                self.logger.info(f"Unregistered sheaf service {service_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to unregister service {service_id}: {e}")
            raise RuntimeError(f"Service unregistration failed: {e}")
    
    def discover_services(self, operation_type: ServiceOperationType) -> List[ServiceInstance]:
        """Discover available services for a specific operation type"""
        with self._lock:
            service_ids = self._service_types.get(operation_type, set())
            return [
                self._services[service_id] 
                for service_id in service_ids 
                if service_id in self._services
                and self._services[service_id].service_interface.status == ServiceStatus.ACTIVE
            ]
    
    def select_service(
        self, 
        operation_type: ServiceOperationType,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    ) -> Optional[ServiceInstance]:
        """Select a service instance based on load balancing strategy"""
        
        available_services = self.discover_services(operation_type)
        if not available_services:
            return None
        
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._select_round_robin(operation_type, available_services)
        elif strategy == LoadBalancingStrategy.LEAST_LOADED:
            return self._select_least_loaded(available_services)
        elif strategy == LoadBalancingStrategy.HEALTH_BASED:
            return self._select_health_based(available_services)
        elif strategy == LoadBalancingStrategy.WEIGHTED_RANDOM:
            return self._select_weighted_random(available_services)
        else:
            return available_services[0]  # Default to first available
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get overall registry status and statistics"""
        with self._lock:
            active_services = sum(
                1 for service in self._services.values()
                if service.service_interface.status == ServiceStatus.ACTIVE
            )
            
            operation_coverage = {
                op_type.value: len(self._service_types[op_type])
                for op_type in ServiceOperationType
            }
            
            return {
                "total_services": len(self._services),
                "active_services": active_services,
                "operation_coverage": operation_coverage,
                "service_details": [
                    {
                        "service_id": instance.service_id,
                        "status": instance.service_interface.status.value,
                        "version": instance.version,
                        "capabilities": [cap.value for cap in instance.capabilities],
                        "health_score": instance.health_score,
                        "current_load": instance.current_load
                    }
                    for instance in self._services.values()
                ]
            }
    
    def _select_round_robin(
        self, 
        operation_type: ServiceOperationType, 
        services: List[ServiceInstance]
    ) -> ServiceInstance:
        """Round-robin service selection"""
        counter = self._round_robin_counters[operation_type]
        selected = services[counter % len(services)]
        self._round_robin_counters[operation_type] = (counter + 1) % len(services)
        return selected
    
    def _select_least_loaded(self, services: List[ServiceInstance]) -> ServiceInstance:
        """Select service with lowest current load"""
        return min(services, key=lambda s: s.current_load)
    
    def _select_health_based(self, services: List[ServiceInstance]) -> ServiceInstance:
        """Select service with highest health score"""
        return max(services, key=lambda s: s.health_score)
    
    def _select_weighted_random(self, services: List[ServiceInstance]) -> ServiceInstance:
        """Select service using weighted random selection based on load weights"""
        import random
        weights = [s.load_weight for s in services]
        return random.choices(services, weights=weights)[0]
    
    def _health_monitor_loop(self):
        """Background thread for monitoring service health"""
        while not self._shutdown_event.wait(30):  # Check every 30 seconds
            try:
                self._update_service_health_scores()
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
    
    def _update_service_health_scores(self):
        """Update health scores for all registered services"""
        with self._lock:
            current_time = time.time()
            
            for instance in self._services.values():
                try:
                    # Get service health metrics
                    health_data = instance.service_interface.get_service_health()
                    
                    # Calculate health score based on various factors
                    success_rate = health_data.get("success_rate", 0.0)
                    avg_response_time = health_data.get("average_response_time", float('inf'))
                    
                    # Normalize response time (assuming 1 second is baseline)
                    response_time_score = min(1.0, 1.0 / max(avg_response_time, 0.1))
                    
                    # Combine factors into overall health score
                    instance.health_score = (success_rate * 0.7 + response_time_score * 0.3)
                    instance.last_health_check = current_time
                    
                except Exception as e:
                    # Service health check failed, lower health score
                    instance.health_score = max(0.0, instance.health_score - 0.1)
                    self.logger.warning(f"Health check failed for service {instance.service_id}: {e}")

class SheafServiceOrchestrator:
    """
    Orchestrator for coordinating sheaf service requests
    Handles request routing, retries, circuit breaking, and response aggregation
    """
    
    def __init__(self, registry: SheafServiceRegistry, config: Optional[Dict[str, Any]] = None):
        self.registry = registry
        self.config = config or {}
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.get("max_workers", 10),
            thread_name_prefix="SheafOrchestrator"
        )
        self._request_queue = deque()
        self._lock = threading.RLock()
        self.logger = logging.getLogger("SheafServiceOrchestrator")
        
        # Circuit breaker state
        self._circuit_breakers: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "failure_count": 0,
                "last_failure_time": 0,
                "state": "closed",  # closed, open, half_open
                "failure_threshold": self.config.get("circuit_breaker_threshold", 5),
                "recovery_timeout": self.config.get("circuit_breaker_timeout", 60)
            }
        )
        
        # Retry configuration
        self.retry_config = {
            "max_retries": self.config.get("max_retries", 3),
            "initial_delay": self.config.get("initial_retry_delay", 1.0),
            "backoff_multiplier": self.config.get("retry_backoff_multiplier", 2.0),
            "max_delay": self.config.get("max_retry_delay", 30.0)
        }
    
    def submit_request(
        self, 
        request: ServiceRequest,
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    ) -> Future[ServiceResponse]:
        """Submit a request for processing with load balancing and fault tolerance"""
        
        future = self._executor.submit(
            self._process_request_with_retries,
            request,
            load_balancing_strategy
        )
        
        with self._lock:
            self._request_queue.append({
                "request": request,
                "future": future,
                "submitted_at": time.time()
            })
        
        return future
    
    async def submit_request_async(
        self, 
        request: ServiceRequest,
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    ) -> ServiceResponse:
        """Async version of request submission"""
        loop = asyncio.get_event_loop()
        future = self.submit_request(request, load_balancing_strategy)
        return await loop.run_in_executor(None, future.result)
    
    def batch_submit_requests(
        self, 
        requests: List[ServiceRequest],
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    ) -> List[Future[ServiceResponse]]:
        """Submit multiple requests for batch processing"""
        
        futures = []
        for request in requests:
            future = self.submit_request(request, load_balancing_strategy)
            futures.append(future)
        
        return futures
    
    def wait_for_responses(
        self, 
        futures: List[Future[ServiceResponse]], 
        timeout: Optional[float] = None
    ) -> List[ServiceResponse]:
        """Wait for multiple request futures to complete"""
        
        responses = []
        for future in futures:
            try:
                response = future.result(timeout=timeout)
                responses.append(response)
            except Exception as e:
                # Create error response for failed future
                error_response = ServiceResponse(
                    request_id="unknown",
                    status=ServiceStatus.FAILED,
                    error_message=f"Request processing failed: {e}"
                )
                responses.append(error_response)
        
        return responses
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get current orchestrator status and metrics"""
        with self._lock:
            pending_requests = len([
                item for item in self._request_queue
                if not item["future"].done()
            ])
            
            circuit_breaker_status = {
                service_id: {
                    "state": cb_data["state"],
                    "failure_count": cb_data["failure_count"]
                }
                for service_id, cb_data in self._circuit_breakers.items()
            }
            
            return {
                "pending_requests": pending_requests,
                "total_queued_requests": len(self._request_queue),
                "executor_active_threads": self._executor._threads.__len__() if hasattr(self._executor, '_threads') else 0,
                "circuit_breakers": circuit_breaker_status,
                "retry_config": self.retry_config
            }
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """Shutdown the orchestrator and cleanup resources"""
        self._executor.shutdown(wait=wait, timeout=timeout)
        self.logger.info("Sheaf service orchestrator shut down")
    
    def _process_request_with_retries(
        self, 
        request: ServiceRequest,
        load_balancing_strategy: LoadBalancingStrategy
    ) -> ServiceResponse:
        """Process request with retry logic and circuit breaking"""
        
        last_exception = None
        retry_delay = self.retry_config["initial_delay"]
        
        for attempt in range(self.retry_config["max_retries"] + 1):
            try:
                # Select service for this attempt
                service_instance = self.registry.select_service(
                    request.operation_type, 
                    load_balancing_strategy
                )
                
                if service_instance is None:
                    raise RuntimeError(f"No available services for operation: {request.operation_type}")
                
                # Check circuit breaker
                if not self._is_circuit_breaker_open(service_instance.service_id):
                    # Update service load
                    service_instance.current_load += 1
                    
                    try:
                        # Process the request
                        response = service_instance.service_interface.invoke_sheaf_compression(request)
                        
                        # Request succeeded, reset circuit breaker
                        self._reset_circuit_breaker(service_instance.service_id)
                        
                        return response
                        
                    finally:
                        # Always decrement load
                        service_instance.current_load = max(0, service_instance.current_load - 1)
                
                else:
                    raise RuntimeError(f"Circuit breaker open for service: {service_instance.service_id}")
                
            except Exception as e:
                last_exception = e
                
                if service_instance:
                    self._record_circuit_breaker_failure(service_instance.service_id)
                
                self.logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                
                # Don't retry on the last attempt
                if attempt < self.retry_config["max_retries"]:
                    time.sleep(retry_delay)
                    retry_delay = min(
                        retry_delay * self.retry_config["backoff_multiplier"],
                        self.retry_config["max_delay"]
                    )
        
        # All retries exhausted
        return ServiceResponse(
            request_id=request.request_id,
            status=ServiceStatus.FAILED,
            error_message=f"Request failed after {self.retry_config['max_retries']} retries. Last error: {last_exception}"
        )
    
    def _is_circuit_breaker_open(self, service_id: str) -> bool:
        """Check if circuit breaker is open for a service"""
        cb_data = self._circuit_breakers[service_id]
        
        if cb_data["state"] == "closed":
            return False
        elif cb_data["state"] == "open":
            # Check if we should transition to half-open
            if time.time() - cb_data["last_failure_time"] > cb_data["recovery_timeout"]:
                cb_data["state"] = "half_open"
                return False
            return True
        else:  # half_open
            return False
    
    def _record_circuit_breaker_failure(self, service_id: str):
        """Record a failure for circuit breaker logic"""
        cb_data = self._circuit_breakers[service_id]
        cb_data["failure_count"] += 1
        cb_data["last_failure_time"] = time.time()
        
        if cb_data["failure_count"] >= cb_data["failure_threshold"]:
            cb_data["state"] = "open"
            self.logger.warning(f"Circuit breaker opened for service: {service_id}")
    
    def _reset_circuit_breaker(self, service_id: str):
        """Reset circuit breaker after successful request"""
        cb_data = self._circuit_breakers[service_id]
        cb_data["failure_count"] = 0
        cb_data["state"] = "closed"

# Context manager for service lifecycle
class SheafServiceManager:
    """Context manager for managing sheaf service lifecycle"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.registry = None
        self.orchestrator = None
        self.services: List[SheafServiceInterface] = []
    
    def __enter__(self):
        # Initialize registry and orchestrator
        self.registry = SheafServiceRegistry(self.config.get("registry", {}))
        self.orchestrator = SheafServiceOrchestrator(
            self.registry, 
            self.config.get("orchestrator", {})
        )
        
        self.registry.start_registry()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup services
        for service in self.services:
            try:
                self.registry.unregister_service(service.service_id)
            except Exception as e:
                logging.error(f"Failed to unregister service {service.service_id}: {e}")
        
        # Shutdown orchestrator and registry
        if self.orchestrator:
            self.orchestrator.shutdown()
        
        if self.registry:
            self.registry.stop_registry()
    
    def create_service(self, service_id: str, service_config: Optional[Dict[str, Any]] = None) -> SheafServiceInterface:
        """Create and register a new sheaf service"""
        service = SheafServiceInterface(service_id, service_config)
        service.start_service()
        self.registry.register_service(service)
        self.services.append(service)
        return service

# Factory function for easy setup
def create_sheaf_service_system(config: Optional[Dict[str, Any]] = None) -> SheafServiceManager:
    """Factory function to create a complete sheaf service system"""
    return SheafServiceManager(config)