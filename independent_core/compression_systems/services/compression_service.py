"""
Compression service interfaces for external module access.
Strict interface contracts - no lenient behavior.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import asyncio
import torch
import uuid
from datetime import datetime


class CompressionService(ABC):
    """Base interface for compression services. Must implement all methods."""
    
    def __init__(self, service_id: str, config: Dict[str, Any]):
        if not service_id:
            raise ValueError("Service ID cannot be empty")
        if not isinstance(config, dict):
            raise TypeError(f"Config must be dict, got {type(config)}")
        
        self.service_id = service_id
        self.config = config
        self.is_initialized = False
        self._validate_service_config()
    
    @abstractmethod
    def _validate_service_config(self) -> None:
        """Validate service-specific configuration."""
        pass
    
    @abstractmethod
    async def compress(self, data: torch.Tensor, 
                      request_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compress data asynchronously. Must throw on failure."""
        pass
    
    @abstractmethod
    async def decompress(self, compressed_data: Dict[str, Any]) -> torch.Tensor:
        """Decompress data asynchronously. Must throw on failure."""
        pass
    
    @abstractmethod
    def get_service_metrics(self) -> Dict[str, Any]:
        """Return current service performance metrics."""
        pass
    
    @abstractmethod
    def supports_data_type(self, data_type: str) -> bool:
        """Check if service supports given data type."""
        pass


class CompressionServiceRegistry:
    """Registry for compression services. Strict registration."""
    
    def __init__(self):
        self._services: Dict[str, CompressionService] = {}
        self._service_metadata: Dict[str, Dict[str, Any]] = {}
        self._request_history: List[Dict[str, Any]] = []
        
    def register_service(self, name: str, service: CompressionService,
                        metadata: Dict[str, Any]) -> None:
        """Register compression service. Throws on conflicts."""
        if not name:
            raise ValueError("Service name cannot be empty")
        
        if name in self._services:
            raise ValueError(f"Service '{name}' already registered")
        
        if not isinstance(service, CompressionService):
            raise TypeError(f"Service must inherit from CompressionService")
        
        required_metadata = ['supported_data_types', 'algorithm_type', 'version']
        for key in required_metadata:
            if key not in metadata:
                raise KeyError(f"Missing required metadata: {key}")
        
        self._services[name] = service
        self._service_metadata[name] = metadata
    
    def unregister_service(self, name: str) -> None:
        """Unregister service. Throws if not found."""
        if name not in self._services:
            raise KeyError(f"Service '{name}' not found")
        
        del self._services[name]
        del self._service_metadata[name]
    
    def get_service(self, name: str) -> CompressionService:
        """Get service by name. Throws if not found."""
        if name not in self._services:
            raise KeyError(f"Service '{name}' not registered")
        
        return self._services[name]
    
    def list_services(self) -> List[str]:
        """List all registered service names."""
        return list(self._services.keys())
    
    def get_services_for_data_type(self, data_type: str) -> List[str]:
        """Get services that support specific data type."""
        if not data_type:
            raise ValueError("Data type cannot be empty")
        
        compatible_services = []
        for name, service in self._services.items():
            if service.supports_data_type(data_type):
                compatible_services.append(name)
        
        if not compatible_services:
            raise ValueError(f"No services support data type: {data_type}")
        
        return compatible_services
    
    async def compress_with_service(self, service_name: str, data: torch.Tensor,
                                   request_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Route compression request to service."""
        service = self.get_service(service_name)  # Throws if not found
        
        request_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            result = await service.compress(data, request_config)
            result['request_id'] = request_id
            result['service_name'] = service_name
            result['timestamp'] = start_time.isoformat()
            
            # Log request
            self._request_history.append({
                'request_id': request_id,
                'service_name': service_name,
                'status': 'success',
                'timestamp': start_time.isoformat(),
                'data_size': data.numel()
            })
            
            return result
            
        except Exception as e:
            # Log failure
            self._request_history.append({
                'request_id': request_id,
                'service_name': service_name,
                'status': 'failed',
                'error': str(e),
                'timestamp': start_time.isoformat(),
                'data_size': data.numel()
            })
            raise  # Re-raise original exception
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_requests = len(self._request_history)
        successful_requests = sum(1 for req in self._request_history if req['status'] == 'success')
        
        return {
            'total_services': len(self._services),
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'success_rate': successful_requests / total_requests if total_requests > 0 else 0,
            'registered_services': list(self._services.keys())
        }


class CompressionRequest:
    """Structured compression request. Strict validation."""
    
    def __init__(self, module_name: str, data_type: str, 
                 compression_type: str, data: torch.Tensor,
                 config: Optional[Dict[str, Any]] = None):
        
        if not module_name:
            raise ValueError("Module name cannot be empty")
        if not data_type:
            raise ValueError("Data type cannot be empty")
        if not compression_type:
            raise ValueError("Compression type cannot be empty")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Data must be torch.Tensor, got {type(data)}")
        
        self.module_name = module_name
        self.data_type = data_type
        self.compression_type = compression_type
        self.data = data
        self.config = config or {}
        self.request_id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary."""
        return {
            'request_id': self.request_id,
            'module_name': self.module_name,
            'data_type': self.data_type,
            'compression_type': self.compression_type,
            'config': self.config,
            'timestamp': self.timestamp.isoformat(),
            'data_shape': list(self.data.shape),
            'data_dtype': str(self.data.dtype)
        }


class CompressionResponse:
    """Structured compression response. Strict format."""
    
    def __init__(self, request_id: str, compressed_data: Dict[str, Any],
                 service_name: str, processing_time: float):
        
        if not request_id:
            raise ValueError("Request ID cannot be empty")
        if not isinstance(compressed_data, dict):
            raise TypeError("Compressed data must be dict")
        if not service_name:
            raise ValueError("Service name cannot be empty")
        if processing_time < 0:
            raise ValueError("Processing time cannot be negative")
        
        self.request_id = request_id
        self.compressed_data = compressed_data
        self.service_name = service_name
        self.processing_time = processing_time
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            'request_id': self.request_id,
            'service_name': self.service_name,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp.isoformat(),
            'compressed_data': self.compressed_data
        }