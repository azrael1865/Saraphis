"""
External API for compression services.
Zero-tolerance error handling - everything throws immediately.
"""

from typing import Any, Dict, List, Optional
import asyncio
import torch
from .compression_service import (
    CompressionServiceRegistry, 
    CompressionRequest, 
    CompressionResponse
)


class CompressionAPI:
    """Public API for external modules to access compression services."""
    
    def __init__(self, registry: CompressionServiceRegistry, config: Dict[str, Any]):
        if not isinstance(registry, CompressionServiceRegistry):
            raise TypeError("Registry must be CompressionServiceRegistry instance")
        
        if not isinstance(config, dict):
            raise TypeError("Config must be dict")
        
        required_config = ['max_concurrent_requests', 'request_timeout_seconds']
        for key in required_config:
            if key not in config:
                raise KeyError(f"Missing required config: {key}")
        
        self.registry = registry
        self.config = config
        self._active_requests: Dict[str, asyncio.Task] = {}
        self._rate_limits: Dict[str, int] = {}
        
    async def request_compression(self, request_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process compression request from external module.
        
        Throws immediately on any validation failure or processing error.
        """
        # Validate request structure
        required_fields = ['module', 'data_type', 'compression_type', 'data']
        for field in required_fields:
            if field not in request_dict:
                raise KeyError(f"Missing required field: {field}")
        
        # Create structured request
        request = CompressionRequest(
            module_name=request_dict['module'],
            data_type=request_dict['data_type'],
            compression_type=request_dict['compression_type'],
            data=request_dict['data'],
            config=request_dict.get('config', {})
        )
        
        # Check concurrent request limits
        if len(self._active_requests) >= self.config['max_concurrent_requests']:
            raise RuntimeError(f"Too many concurrent requests: {len(self._active_requests)}")
        
        # Check rate limits for module
        self._enforce_rate_limit(request.module_name)
        
        # Route to appropriate service
        service_name = self._resolve_service_name(request.compression_type, request.data_type)
        
        # Process request with timeout
        task = asyncio.create_task(
            self._process_compression_request(service_name, request)
        )
        
        self._active_requests[request.request_id] = task
        
        try:
            response = await asyncio.wait_for(
                task, 
                timeout=self.config['request_timeout_seconds']
            )
            return response.to_dict()
            
        except asyncio.TimeoutError:
            task.cancel()
            raise RuntimeError(f"Request {request.request_id} timed out")
            
        finally:
            if request.request_id in self._active_requests:
                del self._active_requests[request.request_id]
    
    async def request_decompression(self, request_dict: Dict[str, Any]) -> torch.Tensor:
        """
        Process decompression request.
        
        Throws immediately on any validation failure.
        """
        required_fields = ['request_id', 'compressed_data', 'service_name']
        for field in required_fields:
            if field not in request_dict:
                raise KeyError(f"Missing required field: {field}")
        
        service_name = request_dict['service_name']
        service = self.registry.get_service(service_name)  # Throws if not found
        
        compressed_data = request_dict['compressed_data']
        if not isinstance(compressed_data, dict):
            raise TypeError("Compressed data must be dict")
        
        return await service.decompress(compressed_data)
    
    def list_available_services(self) -> List[Dict[str, Any]]:
        """List all available compression services with metadata."""
        services = []
        for service_name in self.registry.list_services():
            service = self.registry.get_service(service_name)
            metadata = self.registry._service_metadata[service_name]
            
            services.append({
                'name': service_name,
                'service_id': service.service_id,
                'supported_data_types': metadata['supported_data_types'],
                'algorithm_type': metadata['algorithm_type'],
                'version': metadata['version'],
                'metrics': service.get_service_metrics()
            })
        
        return services
    
    def get_services_for_data_type(self, data_type: str) -> List[str]:
        """Get compatible services for data type."""
        return self.registry.get_services_for_data_type(data_type)
    
    def get_api_statistics(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        registry_stats = self.registry.get_registry_stats()
        
        return {
            'active_requests': len(self._active_requests),
            'max_concurrent_requests': self.config['max_concurrent_requests'],
            'rate_limits': dict(self._rate_limits),
            **registry_stats
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all services."""
        health_results = {}
        
        for service_name in self.registry.list_services():
            try:
                service = self.registry.get_service(service_name)
                metrics = service.get_service_metrics()
                
                health_results[service_name] = {
                    'status': 'healthy',
                    'metrics': metrics
                }
                
            except Exception as e:
                health_results[service_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        overall_healthy = all(
            result['status'] == 'healthy' 
            for result in health_results.values()
        )
        
        return {
            'overall_status': 'healthy' if overall_healthy else 'unhealthy',
            'services': health_results,
            'total_services': len(health_results),
            'healthy_services': sum(
                1 for result in health_results.values() 
                if result['status'] == 'healthy'
            )
        }
    
    def _resolve_service_name(self, compression_type: str, data_type: str) -> str:
        """Resolve compression type to actual service name."""
        if compression_type == 'auto':
            # Get best service for data type
            compatible_services = self.registry.get_services_for_data_type(data_type)
            if not compatible_services:
                raise ValueError(f"No services available for data type: {data_type}")
            return compatible_services[0]  # Return first compatible service
        
        # Direct service name mapping
        service_mapping = {
            'padic': 'padic_compressor',
            'sheaf': 'sheaf_compressor', 
            'tensor': 'tensor_compressor',
            'gpu_memory': 'gpu_memory_compressor'
        }
        
        if compression_type not in service_mapping:
            raise ValueError(f"Unknown compression type: {compression_type}")
        
        service_name = service_mapping[compression_type]
        
        # Verify service exists and supports data type
        service = self.registry.get_service(service_name)  # Throws if not found
        if not service.supports_data_type(data_type):
            raise ValueError(f"Service {service_name} does not support data type: {data_type}")
        
        return service_name
    
    def _enforce_rate_limit(self, module_name: str) -> None:
        """Enforce rate limiting per module."""
        max_requests_per_module = self.config.get('max_requests_per_module', 100)
        
        current_requests = self._rate_limits.get(module_name, 0)
        if current_requests >= max_requests_per_module:
            raise RuntimeError(f"Rate limit exceeded for module: {module_name}")
        
        self._rate_limits[module_name] = current_requests + 1
    
    async def _process_compression_request(self, service_name: str, 
                                         request: CompressionRequest) -> CompressionResponse:
        """Process compression request through service."""
        import time
        
        start_time = time.time()
        
        compressed_data = await self.registry.compress_with_service(
            service_name=service_name,
            data=request.data,
            request_config=request.config
        )
        
        processing_time = time.time() - start_time
        
        return CompressionResponse(
            request_id=request.request_id,
            compressed_data=compressed_data,
            service_name=service_name,
            processing_time=processing_time
        )