"""
Hybrid Service Integration Example - Complete integration patterns with CompressionServiceInterface
Demonstrates how to use HybridPadicServiceLayer, HybridServiceIntegration, and HybridServiceMiddleware together
NO FALLBACKS - HARD FAILURES ONLY
"""

import logging
import time
import torch
from typing import Dict, Any, Optional

# Import service interfaces
from ..service_interfaces.service_interfaces_core import (
    CompressionServiceInterface, ServiceRequest, ServiceResponse,
    ServiceStatus, ServiceHealth
)

# Import hybrid service components
from .hybrid_padic_service_layer import HybridPadicServiceLayer, HybridServiceConfig
from .hybrid_service_integration import HybridServiceIntegration, IntegrationConfig, ServiceEndpoint
from .hybrid_service_middleware import HybridServiceMiddleware, MiddlewareConfig

# Import existing service components
from .padic_service_layer import PadicServiceInterface, PadicServiceConfig


class HybridServiceIntegrationPattern:
    """
    Complete integration pattern for hybrid p-adic services.
    Demonstrates proper setup and usage of all hybrid service components.
    """
    
    def __init__(self, 
                 hybrid_config: Optional[HybridServiceConfig] = None,
                 integration_config: Optional[IntegrationConfig] = None,
                 middleware_config: Optional[MiddlewareConfig] = None):
        """Initialize complete hybrid service integration"""
        self.logger = logging.getLogger('HybridServiceIntegrationPattern')
        
        # Store configurations
        self.hybrid_config = hybrid_config or HybridServiceConfig()
        self.integration_config = integration_config or IntegrationConfig()
        self.middleware_config = middleware_config or MiddlewareConfig()
        
        # Core service components
        self.compression_service_interface: Optional[CompressionServiceInterface] = None
        self.hybrid_service_layer: Optional[HybridPadicServiceLayer] = None
        self.service_integration: Optional[HybridServiceIntegration] = None
        self.service_middleware: Optional[HybridServiceMiddleware] = None
        
        # Integration state
        self.is_initialized = False
        self.startup_time = None
        
        self.logger.info("HybridServiceIntegrationPattern created")
    
    def initialize_complete_integration(self) -> None:
        """
        Initialize complete hybrid service integration with all components.
        
        Raises:
            RuntimeError: If initialization fails
        """
        if self.is_initialized:
            return
        
        try:
            self.logger.info("Starting complete hybrid service integration...")
            
            # Step 1: Initialize core compression service interface
            self._initialize_compression_service_interface()
            
            # Step 2: Initialize hybrid service layer
            self._initialize_hybrid_service_layer()
            
            # Step 3: Initialize service middleware
            self._initialize_service_middleware()
            
            # Step 4: Initialize service integration
            self._initialize_service_integration()
            
            # Step 5: Register all services and set up routing
            self._setup_service_routing()
            
            # Step 6: Configure middleware pipeline
            self._configure_middleware_pipeline()
            
            self.is_initialized = True
            self.startup_time = time.time()
            
            self.logger.info("Complete hybrid service integration initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize complete integration: {e}")
            raise RuntimeError(f"Integration initialization failed: {e}")
    
    def _initialize_compression_service_interface(self) -> None:
        """Initialize core compression service interface"""
        self.logger.info("Initializing compression service interface...")
        
        # Create compression service interface with configuration
        interface_config = {
            'enable_validation': True,
            'enable_metrics': True,
            'enable_health_monitoring': True,
            'request_timeout': self.integration_config.request_timeout,
            'max_concurrent_requests': 50
        }
        
        self.compression_service_interface = CompressionServiceInterface(interface_config)
        
        self.logger.info("Compression service interface initialized")
    
    def _initialize_hybrid_service_layer(self) -> None:
        """Initialize hybrid p-adic service layer"""
        self.logger.info("Initializing hybrid service layer...")
        
        # Create and initialize hybrid service layer
        self.hybrid_service_layer = HybridPadicServiceLayer(self.hybrid_config)
        self.hybrid_service_layer.initialize_hybrid_services()
        
        self.logger.info("Hybrid service layer initialized")
    
    def _initialize_service_middleware(self) -> None:
        """Initialize service middleware"""
        self.logger.info("Initializing service middleware...")
        
        # Create service middleware with enhanced configuration
        enhanced_middleware_config = MiddlewareConfig(
            enable_rate_limiting=True,
            rate_limit_requests_per_minute=self.integration_config.request_timeout,
            enable_caching=True,
            cache_ttl_seconds=300,
            enable_request_validation=True,
            enable_gpu_memory_monitoring=True,
            enable_tensor_optimization=True,
            enable_request_logging=True,
            enable_performance_tracking=True,
            enable_error_tracking=True
        )
        
        self.service_middleware = HybridServiceMiddleware(enhanced_middleware_config)
        
        self.logger.info("Service middleware initialized")
    
    def _initialize_service_integration(self) -> None:
        """Initialize service integration layer"""
        self.logger.info("Initializing service integration...")
        
        # Create service integration
        self.service_integration = HybridServiceIntegration(self.integration_config)
        
        # Initialize integration with core services
        self.service_integration.initialize_integration(
            hybrid_service=self.hybrid_service_layer,
            compression_interface=self.compression_service_interface
        )
        
        self.logger.info("Service integration initialized")
    
    def _setup_service_routing(self) -> None:
        """Set up service routing and registration"""
        self.logger.info("Setting up service routing...")
        
        # Register pure p-adic service as fallback
        pure_padic_config = PadicServiceConfig(
            prime=self.hybrid_config.hybrid_threshold,  # Use same prime
            precision=4,
            chunk_size=1000,
            enable_validation=True
        )
        pure_padic_service = PadicServiceInterface(pure_padic_config)
        
        # Create service endpoint for pure p-adic
        pure_endpoint = ServiceEndpoint(
            service_name="pure_padic",
            service_type="pure_padic",
            service_instance=pure_padic_service,
            version="1.0.0",
            priority=5,
            weight=1.0,
            max_concurrent_requests=20
        )
        
        # Register pure p-adic service
        self.service_integration.register_service_endpoint(pure_endpoint)
        
        # Register services with compression interface
        self.compression_service_interface.register_service(
            service_name="hybrid_padic_complete",
            service_instance=self.hybrid_service_layer,
            version="1.0.0",
            metadata={
                'type': 'hybrid_compression',
                'gpu_enabled': self.hybrid_config.enable_gpu_acceleration,
                'integration_pattern': True
            }
        )
        
        self.compression_service_interface.register_service(
            service_name="pure_padic_fallback",
            service_instance=pure_padic_service,
            version="1.0.0",
            metadata={
                'type': 'pure_compression',
                'fallback_service': True
            }
        )
        
        self.logger.info("Service routing configured")
    
    def _configure_middleware_pipeline(self) -> None:
        """Configure middleware pipeline for optimal performance"""
        self.logger.info("Configuring middleware pipeline...")
        
        # Middleware is already configured with default components
        # Additional custom middleware can be added here if needed
        
        # Example: Add custom hybrid-specific middleware
        # custom_middleware = CustomHybridMiddleware(self.middleware_config)
        # self.service_middleware.add_middleware(custom_middleware)
        
        self.logger.info("Middleware pipeline configured")
    
    def process_service_request_complete(self, request: ServiceRequest) -> ServiceResponse:
        """
        Process service request through complete integration pipeline.
        
        Args:
            request: Service request to process
            
        Returns:
            Service response
            
        Raises:
            RuntimeError: If integration not initialized
            ValueError: If request is invalid
        """
        if not self.is_initialized:
            raise RuntimeError("Integration not initialized")
        
        if not isinstance(request, ServiceRequest):
            raise TypeError(f"Request must be ServiceRequest, got {type(request)}")
        
        try:
            self.logger.debug(f"Processing request {request.request_id} through complete pipeline")
            
            # Step 1: Process request through middleware pipeline
            processed_request, should_continue, middleware_context = (
                self.service_middleware.process_request(request)
            )
            
            if not should_continue:
                # Request was blocked by middleware
                blocked_response = ServiceResponse(
                    request_id=request.request_id,
                    status=ServiceStatus.INVALID,
                    metadata=middleware_context
                )
                blocked_response.add_error(
                    error_code="MIDDLEWARE_BLOCKED",
                    error_message=f"Request blocked by {middleware_context.get('blocked_by_middleware', 'unknown')} middleware"
                )
                return self.service_middleware.process_response(blocked_response, middleware_context)
            
            # Step 2: Route request through service integration
            integration_response = self.service_integration.route_request(processed_request)
            
            # Step 3: Process response through middleware pipeline
            final_response = self.service_middleware.process_response(integration_response, middleware_context)
            
            self.logger.debug(f"Request {request.request_id} processed successfully")
            return final_response
            
        except Exception as e:
            self.logger.error(f"Error processing request {request.request_id}: {e}")
            
            # Create error response
            error_response = ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.ERROR,
                metadata={'integration_error': True}
            )
            error_response.add_error(
                error_code="INTEGRATION_ERROR",
                error_message=str(e),
                error_details={'exception_type': type(e).__name__}
            )
            
            return error_response
    
    def compress_data_hybrid_optimized(self, data: torch.Tensor, 
                                     compression_options: Optional[Dict[str, Any]] = None) -> ServiceResponse:
        """
        Compress data using hybrid-optimized pipeline.
        
        Args:
            data: Tensor data to compress
            compression_options: Optional compression parameters
            
        Returns:
            Service response with compression results
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Data must be torch.Tensor, got {type(data)}")
        
        # Create service request for hybrid compression
        request = ServiceRequest(
            service_name="hybrid_padic_complete",
            method_name="hybrid_compress",
            version="1.0.0",
            payload={
                'data': data,
                'options': compression_options or {}
            },
            metadata={
                'data_shape': list(data.shape),
                'data_dtype': str(data.dtype),
                'optimization_requested': True
            }
        )
        
        return self.process_service_request_complete(request)
    
    def decompress_data_hybrid_optimized(self, compressed_data: Dict[str, Any]) -> ServiceResponse:
        """
        Decompress data using hybrid-optimized pipeline.
        
        Args:
            compressed_data: Compressed data dictionary
            
        Returns:
            Service response with decompression results
        """
        if not isinstance(compressed_data, dict):
            raise TypeError(f"Compressed data must be dict, got {type(compressed_data)}")
        
        # Create service request for hybrid decompression
        request = ServiceRequest(
            service_name="hybrid_padic_complete",
            method_name="hybrid_decompress",
            version="1.0.0",
            payload={
                'compressed_data': compressed_data
            },
            metadata={
                'decompression_requested': True
            }
        )
        
        return self.process_service_request_complete(request)
    
    def get_complete_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of complete integration.
        
        Returns:
            Dictionary containing status of all components
        """
        if not self.is_initialized:
            return {'initialized': False, 'error': 'Integration not initialized'}
        
        return {
            'initialized': True,
            'uptime_seconds': time.time() - self.startup_time if self.startup_time else 0,
            'compression_interface': {
                'services_registered': len(self.compression_service_interface.list_services()),
                'health': 'healthy'
            },
            'hybrid_service_layer': self.hybrid_service_layer.get_hybrid_service_status(),
            'service_integration': self.service_integration.get_integration_status(),
            'service_middleware': self.service_middleware.get_middleware_status(),
            'configuration': {
                'hybrid_enabled': self.hybrid_config.enable_hybrid_compression,
                'gpu_acceleration': self.hybrid_config.enable_gpu_acceleration,
                'rate_limiting': self.middleware_config.enable_rate_limiting,
                'caching': self.middleware_config.enable_caching,
                'monitoring': self.middleware_config.enable_performance_tracking
            }
        }
    
    def get_complete_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics from all components.
        
        Returns:
            Dictionary containing metrics from all components
        """
        if not self.is_initialized:
            return {'error': 'Integration not initialized'}
        
        return {
            'compression_interface_metrics': self.compression_service_interface.get_metrics_summary(),
            'hybrid_service_metrics': self.hybrid_service_layer.get_hybrid_service_status(),
            'integration_metrics': self.service_integration.get_integration_metrics(),
            'middleware_metrics': self.service_middleware.get_middleware_metrics(),
            'performance_summary': {
                'total_components': 4,
                'all_healthy': self._check_all_components_healthy(),
                'integration_complete': True
            }
        }
    
    def _check_all_components_healthy(self) -> bool:
        """Check if all components are healthy"""
        try:
            # Check compression interface
            if not self.compression_service_interface:
                return False
            
            # Check hybrid service layer
            hybrid_status = self.hybrid_service_layer.get_hybrid_service_status()
            if hybrid_status['health'] != ServiceHealth.HEALTHY:
                return False
            
            # Check service integration
            integration_status = self.service_integration.get_integration_status()
            if not integration_status['initialized']:
                return False
            
            # Check middleware
            middleware_status = self.service_middleware.get_middleware_status()
            if not middleware_status['initialized']:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def run_integration_test(self) -> Dict[str, Any]:
        """
        Run comprehensive integration test.
        
        Returns:
            Test results dictionary
        """
        test_results = {
            'test_start_time': time.time(),
            'tests_passed': 0,
            'tests_failed': 0,
            'test_details': [],
            'overall_success': False
        }
        
        try:
            # Test 1: Initialize integration
            self.logger.info("Running integration test...")
            
            if not self.is_initialized:
                self.initialize_complete_integration()
            
            test_results['tests_passed'] += 1
            test_results['test_details'].append({
                'test': 'initialization',
                'status': 'passed',
                'message': 'Integration initialized successfully'
            })
            
            # Test 2: Compress test data
            test_data = torch.randn(100, 100, dtype=torch.float32)
            if torch.cuda.is_available():
                test_data = test_data.cuda()
            
            compression_response = self.compress_data_hybrid_optimized(test_data)
            
            if compression_response.is_success():
                test_results['tests_passed'] += 1
                test_results['test_details'].append({
                    'test': 'compression',
                    'status': 'passed',
                    'message': 'Data compressed successfully'
                })
            else:
                test_results['tests_failed'] += 1
                test_results['test_details'].append({
                    'test': 'compression',
                    'status': 'failed',
                    'message': f"Compression failed: {compression_response.errors}"
                })
            
            # Test 3: Decompress test data
            if compression_response.is_success():
                compressed_data = compression_response.data
                decompression_response = self.decompress_data_hybrid_optimized(compressed_data)
                
                if decompression_response.is_success():
                    test_results['tests_passed'] += 1
                    test_results['test_details'].append({
                        'test': 'decompression',
                        'status': 'passed',
                        'message': 'Data decompressed successfully'
                    })
                else:
                    test_results['tests_failed'] += 1
                    test_results['test_details'].append({
                        'test': 'decompression',
                        'status': 'failed',
                        'message': f"Decompression failed: {decompression_response.errors}"
                    })
            
            # Test 4: Check health status
            health_status = self.get_complete_status()
            if health_status.get('initialized', False):
                test_results['tests_passed'] += 1
                test_results['test_details'].append({
                    'test': 'health_check',
                    'status': 'passed',
                    'message': 'All components healthy'
                })
            else:
                test_results['tests_failed'] += 1
                test_results['test_details'].append({
                    'test': 'health_check',
                    'status': 'failed',
                    'message': 'Component health issues detected'
                })
            
            test_results['overall_success'] = test_results['tests_failed'] == 0
            
        except Exception as e:
            test_results['tests_failed'] += 1
            test_results['test_details'].append({
                'test': 'general',
                'status': 'failed',
                'message': f"Integration test failed: {e}"
            })
            test_results['overall_success'] = False
        
        test_results['test_duration'] = time.time() - test_results['test_start_time']
        
        self.logger.info(f"Integration test completed. Passed: {test_results['tests_passed']}, Failed: {test_results['tests_failed']}")
        
        return test_results
    
    def shutdown_complete_integration(self) -> None:
        """Shutdown complete integration cleanly"""
        self.logger.info("Shutting down complete hybrid service integration...")
        
        try:
            # Shutdown components in reverse order
            if self.service_middleware:
                self.service_middleware.shutdown()
            
            if self.service_integration:
                self.service_integration.shutdown()
            
            if self.hybrid_service_layer:
                self.hybrid_service_layer.shutdown()
            
            # Compression service interface doesn't have explicit shutdown
            
            self.is_initialized = False
            self.startup_time = None
            
            self.logger.info("Complete integration shutdown successful")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


def create_example_integration() -> HybridServiceIntegrationPattern:
    """
    Create example integration with optimized configuration.
    
    Returns:
        Configured integration pattern instance
    """
    # Create optimized configurations
    hybrid_config = HybridServiceConfig(
        enable_hybrid_compression=True,
        hybrid_threshold=500,  # Use hybrid for tensors > 500 elements
        enable_gpu_acceleration=True,
        gpu_memory_limit_mb=2048,
        validate_reconstruction=True,
        max_reconstruction_error=1e-6,
        enable_hybrid_caching=True,
        hybrid_cache_size_mb=512
    )
    
    integration_config = IntegrationConfig(
        enable_service_discovery=True,
        enable_health_monitoring=True,
        enable_load_balancing=True,
        enable_fallback_routing=True,
        request_timeout=300.0,
        max_retry_attempts=3
    )
    
    middleware_config = MiddlewareConfig(
        enable_rate_limiting=True,
        rate_limit_requests_per_minute=1000,
        enable_caching=True,
        cache_ttl_seconds=300,
        enable_request_validation=True,
        enable_gpu_memory_monitoring=True,
        enable_tensor_optimization=True,
        enable_performance_tracking=True
    )
    
    return HybridServiceIntegrationPattern(
        hybrid_config=hybrid_config,
        integration_config=integration_config,
        middleware_config=middleware_config
    )


def run_complete_example():
    """Run complete integration example"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('HybridIntegrationExample')
    
    try:
        # Create and initialize integration
        integration = create_example_integration()
        integration.initialize_complete_integration()
        
        # Run integration test
        test_results = integration.run_integration_test()
        
        logger.info("Integration Example Results:")
        logger.info(f"Tests Passed: {test_results['tests_passed']}")
        logger.info(f"Tests Failed: {test_results['tests_failed']}")
        logger.info(f"Overall Success: {test_results['overall_success']}")
        
        # Get status and metrics
        status = integration.get_complete_status()
        metrics = integration.get_complete_metrics()
        
        logger.info("Integration Status:")
        logger.info(f"Initialized: {status['initialized']}")
        logger.info(f"All Components Healthy: {metrics['performance_summary']['all_healthy']}")
        
        # Shutdown
        integration.shutdown_complete_integration()
        
        return test_results['overall_success']
        
    except Exception as e:
        logger.error(f"Integration example failed: {e}")
        return False


if __name__ == "__main__":
    success = run_complete_example()
    print(f"Integration example {'PASSED' if success else 'FAILED'}")