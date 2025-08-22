"""
Hybrid P-adic Service Layer Implementation
Service layer for hybrid p-adic compression operations with GPU acceleration.
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
from service_interfaces.service_interfaces_core import (
    CompressionServiceInterface, ServiceRequest, ServiceResponse,
    ServiceStatus, ServiceHealth
)

# Import hybrid p-adic components
from hybrid_padic_structures import (
    HybridPadicWeight, HybridPadicValidator, HybridPadicManager
)
from hybrid_padic_compressor import (
    HybridPadicCompressionSystem, HybridPadicIntegrationManager
)

# Import existing p-adic service layer
from padic_service_layer import PadicServiceInterface, PadicServiceConfig, PadicServiceMethod


class HybridServiceMethod(Enum):
    """Enumeration of hybrid p-adic service methods"""
    HYBRID_COMPRESS = "hybrid_compress"
    HYBRID_DECOMPRESS = "hybrid_decompress"
    HYBRID_OPTIMIZE = "hybrid_optimize"
    HYBRID_VALIDATE = "hybrid_validate"
    HYBRID_BATCH_COMPRESS = "hybrid_batch_compress"
    HYBRID_BATCH_DECOMPRESS = "hybrid_batch_decompress"
    CONVERT_TO_HYBRID = "convert_to_hybrid"
    CONVERT_FROM_HYBRID = "convert_from_hybrid"
    GPU_MEMORY_STATUS = "gpu_memory_status"
    HYBRID_PERFORMANCE_METRICS = "hybrid_performance_metrics"
    HYBRID_HEALTH_CHECK = "hybrid_health_check"
    CONFIGURE_HYBRID = "configure_hybrid"


@dataclass
class HybridServiceConfig:
    """Configuration for hybrid p-adic service layer"""
    # Hybrid-specific configuration
    enable_hybrid_compression: bool = True
    hybrid_threshold: int = 1000  # Elements threshold for hybrid compression
    force_hybrid: bool = False
    enable_dynamic_switching: bool = True
    gpu_memory_limit_mb: int = 14336
    max_concurrent_hybrid_requests: int = 50
    
    # Performance configuration
    hybrid_timeout_seconds: float = 600.0
    hybrid_batch_size: int = 1000
    enable_hybrid_caching: bool = True
    hybrid_cache_size_mb: int = 2048
    
    # Quality configuration
    validate_reconstruction: bool = True
    max_reconstruction_error: float = 1e-6
    preserve_ultrametric: bool = True
    
    # Monitoring configuration
    enable_hybrid_metrics: bool = True
    hybrid_metrics_buffer_size: int = 5000
    hybrid_health_check_interval: float = 30.0
    
    # GPU configuration
    enable_gpu_acceleration: bool = True
    gpu_memory_fraction: float = 0.7
    cuda_stream_count: int = 4
    enable_memory_pooling: bool = True
    
    # Fallback configuration
    enable_pure_fallback: bool = True
    fallback_threshold_mb: int = 100
    max_fallback_attempts: int = 2
    
    def __post_init__(self):
        """Validate hybrid service configuration"""
        if not isinstance(self.enable_hybrid_compression, bool):
            raise TypeError("enable_hybrid_compression must be bool")
        if not isinstance(self.hybrid_threshold, int) or self.hybrid_threshold <= 0:
            raise ValueError("hybrid_threshold must be positive int")
        if not isinstance(self.gpu_memory_limit_mb, int) or self.gpu_memory_limit_mb <= 0:
            raise ValueError("gpu_memory_limit_mb must be positive int")
        if not isinstance(self.hybrid_timeout_seconds, (int, float)) or self.hybrid_timeout_seconds <= 0:
            raise ValueError("hybrid_timeout_seconds must be positive number")
        if not isinstance(self.max_reconstruction_error, (int, float)) or self.max_reconstruction_error <= 0:
            raise ValueError("max_reconstruction_error must be positive number")


@dataclass
class HybridServiceMetrics:
    """Metrics for hybrid p-adic service operations"""
    total_hybrid_requests: int = 0
    successful_hybrid_requests: int = 0
    failed_hybrid_requests: int = 0
    fallback_to_pure_count: int = 0
    average_hybrid_processing_time: float = 0.0
    average_pure_fallback_time: float = 0.0
    gpu_memory_peak_usage_mb: float = 0.0
    gpu_memory_average_usage_mb: float = 0.0
    hybrid_compression_ratio_average: float = 0.0
    reconstruction_error_average: float = 0.0
    service_uptime_seconds: float = 0.0
    last_health_check: Optional[datetime] = None
    
    def update_request_metrics(self, success: bool, processing_time: float):
        """Update request metrics"""
        self.total_hybrid_requests += 1
        if success:
            self.successful_hybrid_requests += 1
        else:
            self.failed_hybrid_requests += 1
        
        # Update average processing time
        if self.total_hybrid_requests > 1:
            old_avg = self.average_hybrid_processing_time
            self.average_hybrid_processing_time = (
                (old_avg * (self.total_hybrid_requests - 1) + processing_time) / 
                self.total_hybrid_requests
            )
        else:
            self.average_hybrid_processing_time = processing_time


class HybridPadicServiceLayer:
    """
    Service layer for hybrid p-adic compression operations.
    Provides hybrid-specific service methods with GPU acceleration and integration with existing service infrastructure.
    """
    
    def __init__(self, config: Optional[HybridServiceConfig] = None):
        """Initialize hybrid p-adic service layer"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, HybridServiceConfig):
            raise TypeError(f"Config must be HybridServiceConfig or None, got {type(config)}")
        
        self.config = config or HybridServiceConfig()
        self.logger = logging.getLogger('HybridPadicServiceLayer')
        
        # Check GPU availability
        if self.config.enable_gpu_acceleration and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available but GPU acceleration is enabled")
        
        # Service state
        self.is_initialized = False
        self.service_health = ServiceHealth.UNKNOWN
        self.startup_time = datetime.utcnow()
        
        # Hybrid p-adic components
        self.hybrid_compression_system: Optional[HybridPadicCompressionSystem] = None
        self.hybrid_integration_manager: Optional[HybridPadicIntegrationManager] = None
        self.hybrid_manager: Optional[HybridPadicManager] = None
        
        # Service infrastructure
        self.service_interface: Optional[CompressionServiceInterface] = None
        self.pure_service_interface: Optional[PadicServiceInterface] = None
        
        # Performance tracking
        self.metrics = HybridServiceMetrics()
        self.performance_history: deque = deque(maxlen=self.config.hybrid_metrics_buffer_size)
        self.active_requests: Dict[str, datetime] = {}
        
        # Thread safety
        self._service_lock = threading.RLock()
        self._metrics_lock = threading.RLock()
        self._request_lock = threading.RLock()
        
        # Background services
        self._health_monitor_enabled = False
        self._health_monitor_thread: Optional[threading.Thread] = None
        self._stop_health_monitor = threading.Event()
        
        # Request processing
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_hybrid_requests,
            thread_name_prefix="HybridService"
        )
        
        self.logger.info("HybridPadicServiceLayer created successfully")
    
    def initialize_hybrid_services(self) -> None:
        """
        Initialize hybrid p-adic services and components.
        
        Raises:
            RuntimeError: If initialization fails
        """
        if self.is_initialized:
            return
        
        with self._service_lock:
            try:
                # Initialize hybrid compression system
                hybrid_config = {
                    'prime': 5,  # Default prime
                    'precision': 4,  # Default precision
                    'chunk_size': 1000,
                    'gpu_memory_limit_mb': self.config.gpu_memory_limit_mb,
                    'enable_hybrid': True,
                    'hybrid_threshold': self.config.hybrid_threshold,
                    'force_hybrid': self.config.force_hybrid,
                    'enable_dynamic_switching': self.config.enable_dynamic_switching,
                    'validate_reconstruction': self.config.validate_reconstruction,
                    'max_reconstruction_error': self.config.max_reconstruction_error,
                    'preserve_ultrametric': self.config.preserve_ultrametric,
                    'gpu_config': {
                        'memory_limit_mb': self.config.gpu_memory_limit_mb,
                        'memory_fraction': self.config.gpu_memory_fraction,
                        'stream_count': self.config.cuda_stream_count,
                        'enable_memory_pooling': self.config.enable_memory_pooling
                    }
                }
                
                self.hybrid_compression_system = HybridPadicCompressionSystem(hybrid_config)
                
                # Initialize integration manager
                self.hybrid_integration_manager = HybridPadicIntegrationManager()
                self.hybrid_integration_manager.initialize_systems(hybrid_config)
                
                # Initialize hybrid manager
                self.hybrid_manager = HybridPadicManager(hybrid_config)
                
                # Initialize service interface
                from ..service_interfaces.service_interfaces_core import CompressionServiceInterface
                self.service_interface = CompressionServiceInterface()
                
                # Register hybrid services
                self._register_hybrid_services()
                
                # Start health monitoring
                self._start_health_monitoring()
                
                self.is_initialized = True
                self.service_health = ServiceHealth.HEALTHY
                
                self.logger.info("Hybrid p-adic services initialized successfully")
                
            except Exception as e:
                self.service_health = ServiceHealth.UNHEALTHY
                self.logger.error(f"Failed to initialize hybrid services: {e}")
                raise RuntimeError(f"Hybrid service initialization failed: {e}")
    
    def _register_hybrid_services(self) -> None:
        """Register hybrid service methods with service interface"""
        if not self.service_interface:
            raise RuntimeError("Service interface not initialized")
        
        # Register each hybrid service method
        for method in HybridServiceMethod:
            method_handler = getattr(self, f"_handle_{method.value}", None)
            if method_handler:
                self.service_interface.register_service_method(
                    f"hybrid_padic.{method.value}",
                    method_handler
                )
        
        self.logger.info(f"Registered {len(HybridServiceMethod)} hybrid service methods")
    
    def register_hybrid_service(self, service_name: str, service_instance: Any) -> bool:
        """
        Register a hybrid service instance.
        
        Args:
            service_name: Name of the service to register
            service_instance: Service instance to register
            
        Returns:
            True if registration successful
            
        Raises:
            ValueError: If service_name or service_instance is invalid
            RuntimeError: If service interface not initialized
        """
        if not isinstance(service_name, str):
            raise TypeError(f"Service name must be str, got {type(service_name)}")
        if not service_name.strip():
            raise ValueError("Service name cannot be empty")
        if service_instance is None:
            raise ValueError("Service instance cannot be None")
        
        if not self.service_interface:
            raise RuntimeError("Service interface not initialized")
        
        with self._service_lock:
            try:
                success = self.service_interface.register_service(service_name, service_instance)
                if success:
                    self.logger.info(f"Registered hybrid service: {service_name}")
                else:
                    self.logger.warning(f"Failed to register hybrid service: {service_name}")
                return success
                
            except Exception as e:
                self.logger.error(f"Error registering hybrid service {service_name}: {e}")
                raise RuntimeError(f"Service registration failed: {e}")
    
    def process_hybrid_request(self, request: ServiceRequest) -> ServiceResponse:
        """
        Process a hybrid p-adic service request.
        
        Args:
            request: Service request to process
            
        Returns:
            Service response
            
        Raises:
            ValueError: If request is invalid
            RuntimeError: If service not initialized
        """
        if not isinstance(request, ServiceRequest):
            raise TypeError(f"Request must be ServiceRequest, got {type(request)}")
        
        if not self.is_initialized:
            raise RuntimeError("Hybrid service not initialized")
        
        start_time = time.time()
        
        with self._request_lock:
            self.active_requests[request.request_id] = datetime.utcnow()
        
        try:
            # Validate request
            self._validate_hybrid_request(request)
            
            # Route request to appropriate handler
            method_name = request.method_name
            if not method_name.startswith("hybrid_"):
                method_name = f"hybrid_{method_name}"
            
            handler_name = f"_handle_{method_name}"
            handler = getattr(self, handler_name, None)
            
            if not handler:
                raise ValueError(f"Unsupported hybrid method: {method_name}")
            
            # Process request
            result_data = handler(request)
            
            # Create successful response
            response = ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.SUCCESS,
                data=result_data,
                processing_time=time.time() - start_time,
                service_version="hybrid_padic_v1.0"
            )
            
            # Update metrics
            with self._metrics_lock:
                self.metrics.update_request_metrics(True, response.processing_time)
                self.performance_history.append({
                    'timestamp': datetime.utcnow(),
                    'method': method_name,
                    'processing_time': response.processing_time,
                    'success': True
                })
            
            return response
            
        except Exception as e:
            # Create error response
            response = ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.ERROR,
                errors=[{
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }],
                processing_time=time.time() - start_time,
                service_version="hybrid_padic_v1.0"
            )
            
            # Update metrics
            with self._metrics_lock:
                self.metrics.update_request_metrics(False, response.processing_time)
                self.performance_history.append({
                    'timestamp': datetime.utcnow(),
                    'method': request.method_name,
                    'processing_time': response.processing_time,
                    'success': False,
                    'error': str(e)
                })
            
            self.logger.error(f"Error processing hybrid request {request.request_id}: {e}")
            return response
            
        finally:
            with self._request_lock:
                self.active_requests.pop(request.request_id, None)
    
    def _validate_hybrid_request(self, request: ServiceRequest) -> None:
        """Validate hybrid service request"""
        if not request.payload:
            raise ValueError("Request payload cannot be empty")
        
        # Check for required fields based on method
        method = request.method_name
        if method in ["hybrid_compress", "convert_to_hybrid"]:
            if 'data' not in request.payload:
                raise ValueError("'data' field required for compression")
        elif method in ["hybrid_decompress", "convert_from_hybrid"]:
            if 'compressed_data' not in request.payload:
                raise ValueError("'compressed_data' field required for decompression")
    
    def _handle_hybrid_compress(self, request: ServiceRequest) -> Dict[str, Any]:
        """Handle hybrid compression request"""
        data = request.payload.get('data')
        if data is None:
            raise ValueError("Data cannot be None")
        
        # Convert data to tensor if needed
        if not isinstance(data, torch.Tensor):
            if isinstance(data, (list, tuple)):
                data = torch.tensor(data, dtype=torch.float32)
            else:
                raise TypeError(f"Data must be tensor or list, got {type(data)}")
        
        # Perform hybrid compression
        compressed = self.hybrid_compression_system.compress(data)
        
        return {
            'compressed_data': compressed,
            'compression_type': compressed.get('compression_type', 'hybrid'),
            'original_shape': compressed.get('original_shape'),
            'compression_ratio': compressed.get('compression_ratio', 0.0),
            'compression_time': compressed.get('compression_time', 0.0)
        }
    
    def _handle_hybrid_decompress(self, request: ServiceRequest) -> Dict[str, Any]:
        """Handle hybrid decompression request"""
        compressed_data = request.payload.get('compressed_data')
        if compressed_data is None:
            raise ValueError("Compressed data cannot be None")
        
        # Perform hybrid decompression
        decompressed = self.hybrid_compression_system.decompress(compressed_data)
        
        return {
            'decompressed_data': decompressed.tolist(),  # Convert tensor to list for JSON
            'data_shape': list(decompressed.shape),
            'data_type': str(decompressed.dtype),
            'decompression_time': time.time()  # Would be tracked in actual implementation
        }
    
    def _handle_hybrid_batch_compress(self, request: ServiceRequest) -> Dict[str, Any]:
        """Handle batch hybrid compression request"""
        data_list = request.payload.get('data_list')
        if not data_list:
            raise ValueError("Data list cannot be empty")
        
        if len(data_list) > self.config.hybrid_batch_size:
            raise ValueError(f"Batch size {len(data_list)} exceeds limit {self.config.hybrid_batch_size}")
        
        results = []
        for i, data in enumerate(data_list):
            try:
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data, dtype=torch.float32)
                
                compressed = self.hybrid_compression_system.compress(data)
                results.append({
                    'index': i,
                    'success': True,
                    'compressed_data': compressed,
                    'compression_type': compressed.get('compression_type')
                })
                
            except Exception as e:
                results.append({
                    'index': i,
                    'success': False,
                    'error': str(e)
                })
        
        return {
            'batch_results': results,
            'total_items': len(data_list),
            'successful_items': sum(1 for r in results if r['success']),
            'failed_items': sum(1 for r in results if not r['success'])
        }
    
    def _handle_gpu_memory_status(self, request: ServiceRequest) -> Dict[str, Any]:
        """Handle GPU memory status request"""
        if not torch.cuda.is_available():
            return {
                'cuda_available': False,
                'message': 'CUDA not available'
            }
        
        memory_info = {}
        for device_id in range(torch.cuda.device_count()):
            with torch.cuda.device(device_id):
                allocated = torch.cuda.memory_allocated()
                cached = torch.cuda.memory_reserved()
                total = torch.cuda.get_device_properties(device_id).total_memory
                
                memory_info[f'device_{device_id}'] = {
                    'allocated_mb': allocated / 1024 / 1024,
                    'cached_mb': cached / 1024 / 1024,
                    'total_mb': total / 1024 / 1024,
                    'utilization_percent': (allocated / total) * 100
                }
        
        return {
            'cuda_available': True,
            'device_count': torch.cuda.device_count(),
            'memory_info': memory_info
        }
    
    def _handle_hybrid_performance_metrics(self, request: ServiceRequest) -> Dict[str, Any]:
        """Handle performance metrics request"""
        with self._metrics_lock:
            return {
                'service_metrics': {
                    'total_requests': self.metrics.total_hybrid_requests,
                    'successful_requests': self.metrics.successful_hybrid_requests,
                    'failed_requests': self.metrics.failed_hybrid_requests,
                    'success_rate': (
                        self.metrics.successful_hybrid_requests / max(1, self.metrics.total_hybrid_requests)
                    ),
                    'average_processing_time': self.metrics.average_hybrid_processing_time,
                    'fallback_count': self.metrics.fallback_to_pure_count,
                    'uptime_seconds': (datetime.utcnow() - self.startup_time).total_seconds()
                },
                'gpu_metrics': {
                    'peak_usage_mb': self.metrics.gpu_memory_peak_usage_mb,
                    'average_usage_mb': self.metrics.gpu_memory_average_usage_mb
                },
                'compression_metrics': {
                    'average_compression_ratio': self.metrics.hybrid_compression_ratio_average,
                    'average_reconstruction_error': self.metrics.reconstruction_error_average
                },
                'recent_performance': list(self.performance_history)[-10:]  # Last 10 requests
            }
    
    def _handle_hybrid_health_check(self, request: ServiceRequest) -> Dict[str, Any]:
        """Handle health check request"""
        health_status = self.get_hybrid_service_status()
        
        return {
            'service_health': health_status['health'].value,
            'is_initialized': health_status['initialized'],
            'components_status': health_status['components'],
            'last_health_check': health_status['last_check'].isoformat() if health_status['last_check'] else None,
            'uptime_seconds': (datetime.utcnow() - self.startup_time).total_seconds(),
            'active_requests': len(self.active_requests),
            'gpu_available': torch.cuda.is_available() if self.config.enable_gpu_acceleration else False
        }
    
    def get_hybrid_service_status(self) -> Dict[str, Any]:
        """
        Get comprehensive hybrid service status.
        
        Returns:
            Dictionary containing service status information
        """
        with self._service_lock:
            components_status = {
                'hybrid_compression_system': self.hybrid_compression_system is not None,
                'hybrid_integration_manager': self.hybrid_integration_manager is not None,
                'hybrid_manager': self.hybrid_manager is not None,
                'service_interface': self.service_interface is not None
            }
            
            # Determine overall health
            if all(components_status.values()) and self.is_initialized:
                health = ServiceHealth.HEALTHY
            elif any(components_status.values()):
                health = ServiceHealth.DEGRADED
            else:
                health = ServiceHealth.UNHEALTHY
            
            return {
                'health': health,
                'initialized': self.is_initialized,
                'components': components_status,
                'last_check': self.metrics.last_health_check,
                'configuration': {
                    'hybrid_enabled': self.config.enable_hybrid_compression,
                    'gpu_enabled': self.config.enable_gpu_acceleration,
                    'fallback_enabled': self.config.enable_pure_fallback
                }
            }
    
    def validate_hybrid_service_config(self) -> Dict[str, Any]:
        """
        Validate hybrid service configuration.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check GPU availability if GPU acceleration is enabled
        if self.config.enable_gpu_acceleration and not torch.cuda.is_available():
            validation_results['errors'].append("GPU acceleration enabled but CUDA not available")
            validation_results['valid'] = False
        
        # Check memory limits
        if self.config.gpu_memory_limit_mb > 8192:  # > 8GB
            validation_results['warnings'].append("GPU memory limit very high, may cause OOM")
        
        # Check batch size
        if self.config.hybrid_batch_size > 100:
            validation_results['warnings'].append("Large batch size may cause memory issues")
        
        # Check timeout settings
        if self.config.hybrid_timeout_seconds < 60:
            validation_results['warnings'].append("Short timeout may cause request failures")
        
        # Performance recommendations
        if self.config.enable_hybrid_compression and not self.config.enable_gpu_acceleration:
            validation_results['recommendations'].append("Enable GPU acceleration for better hybrid performance")
        
        if not self.config.enable_hybrid_caching:
            validation_results['recommendations'].append("Enable caching for improved performance")
        
        return validation_results
    
    def _start_health_monitoring(self) -> None:
        """Start background health monitoring"""
        if self._health_monitor_enabled:
            return
        
        self._health_monitor_enabled = True
        self._stop_health_monitor.clear()
        self._health_monitor_thread = threading.Thread(
            target=self._health_monitoring_loop,
            name="HybridServiceHealthMonitor",
            daemon=True
        )
        self._health_monitor_thread.start()
        self.logger.info("Health monitoring started")
    
    def _health_monitoring_loop(self) -> None:
        """Background health monitoring loop"""
        while not self._stop_health_monitor.wait(self.config.hybrid_health_check_interval):
            try:
                # Update health metrics
                with self._metrics_lock:
                    self.metrics.last_health_check = datetime.utcnow()
                    self.metrics.service_uptime_seconds = (
                        datetime.utcnow() - self.startup_time
                    ).total_seconds()
                
                # Check GPU memory if enabled
                if self.config.enable_gpu_acceleration and torch.cuda.is_available():
                    current_usage = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                    if current_usage > self.metrics.gpu_memory_peak_usage_mb:
                        self.metrics.gpu_memory_peak_usage_mb = current_usage
                    
                    # Update average (simple moving average)
                    if self.metrics.gpu_memory_average_usage_mb == 0:
                        self.metrics.gpu_memory_average_usage_mb = current_usage
                    else:
                        self.metrics.gpu_memory_average_usage_mb = (
                            self.metrics.gpu_memory_average_usage_mb * 0.9 + current_usage * 0.1
                        )
                
                # Update service health based on recent performance
                self._update_service_health()
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
    
    def _update_service_health(self) -> None:
        """Update service health based on performance metrics"""
        if not self.is_initialized:
            self.service_health = ServiceHealth.UNHEALTHY
            return
        
        # Check recent failure rate
        recent_requests = list(self.performance_history)[-10:]  # Last 10 requests
        if recent_requests:
            failure_rate = sum(1 for req in recent_requests if not req['success']) / len(recent_requests)
            if failure_rate > 0.5:  # > 50% failure rate
                self.service_health = ServiceHealth.DEGRADED
            elif failure_rate > 0.8:  # > 80% failure rate
                self.service_health = ServiceHealth.UNHEALTHY
            else:
                self.service_health = ServiceHealth.HEALTHY
        else:
            self.service_health = ServiceHealth.HEALTHY
    
    def shutdown(self) -> None:
        """Shutdown hybrid service layer"""
        self.logger.info("Shutting down hybrid service layer")
        
        # Stop health monitoring
        if self._health_monitor_enabled:
            self._health_monitor_enabled = False
            self._stop_health_monitor.set()
            if self._health_monitor_thread and self._health_monitor_thread.is_alive():
                self._health_monitor_thread.join(timeout=5.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Clear active requests
        with self._request_lock:
            self.active_requests.clear()
        
        self.is_initialized = False
        self.service_health = ServiceHealth.UNHEALTHY
        
        self.logger.info("Hybrid service layer shutdown complete")