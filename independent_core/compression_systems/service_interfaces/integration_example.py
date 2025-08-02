"""
Service Interface Integration Example
Demonstrates integration with compression systems, BrainCore, and API layer
"""

import time
from typing import Dict, Any
from datetime import datetime

# Import core components
from independent_core.brain_core import BrainCore
from independent_core.compression_systems.service_interfaces import (
    ServiceRequest,
    ServiceResponse,
    ServiceStatus,
    CompressionServiceInterface,
    ServiceInterfaceIntegration
)


def integrate_service_interface_with_brain_core():
    """Demonstrate integration of Service Interface with BrainCore"""
    
    print("=== Service Interface Integration Example ===")
    
    # Initialize BrainCore
    brain_config = {
        'enable_service_interface': True,
        'enable_service_discovery': True,
        'service_timeout': 30.0
    }
    brain = BrainCore(config=brain_config)
    
    # Initialize Service Interface
    service_config = {
        'enable_validation': True,
        'enable_metrics': True,
        'enable_health_monitoring': True,
        'middleware_timeout': 10.0
    }
    service_interface = CompressionServiceInterface(config=service_config)
    
    # Register Service Interface with BrainCore
    ServiceInterfaceIntegration.register_with_brain_core(brain, service_interface)
    
    print(f"Initialized Service Interface with BrainCore")
    
    # Example 1: Register mock compression services
    print(f"\n1. Service Registration:")
    
    # Mock P-adic compression service
    class MockPadicCompressor:
        def compress(self, data: bytes, base: int = 7) -> Dict[str, Any]:
            """Mock P-adic compression"""
            compressed_size = len(data) // 2  # Mock 50% compression
            return {
                'compressed_data': f"padic_compressed_{len(data)}_bytes",
                'compression_ratio': len(data) / compressed_size,
                'base': base,
                'original_size': len(data),
                'compressed_size': compressed_size
            }
        
        def decompress(self, compressed_data: str, original_size: int) -> bytes:
            """Mock P-adic decompression"""
            return b"x" * original_size
    
    # Mock Sheaf compression service
    class MockSheafCompressor:
        def compress(self, tensor_data: list, cohomology_degree: int = 1) -> Dict[str, Any]:
            """Mock Sheaf compression"""
            original_elements = len(tensor_data)
            compressed_elements = original_elements // 3  # Mock 33% compression
            return {
                'sheaf_structure': f"sheaf_compressed_{original_elements}_elements",
                'cohomology_degree': cohomology_degree,
                'compression_ratio': original_elements / compressed_elements,
                'global_sections': compressed_elements
            }
        
        def reconstruct(self, sheaf_structure: str, target_size: int) -> list:
            """Mock Sheaf reconstruction"""
            return [f"reconstructed_{i}" for i in range(target_size)]
    
    # Register services
    padic_service = MockPadicCompressor()
    sheaf_service = MockSheafCompressor()
    
    service_interface.register_service(
        service_name="padic_compressor",
        service_instance=padic_service,
        version="1.0.0",
        metadata={"type": "padic", "description": "P-adic number compression"}
    )
    
    service_interface.register_service(
        service_name="sheaf_compressor", 
        service_instance=sheaf_service,
        version="1.0.0",
        metadata={"type": "sheaf", "description": "Sheaf theory compression"}
    )
    
    print(f"   Registered services: {list(service_interface.list_services().keys())}")
    
    # Example 2: Service invocation
    print(f"\n2. Service Invocation:")
    
    # P-adic compression request
    padic_request = ServiceRequest(
        service_name="padic_compressor",
        method_name="compress",
        version="1.0.0",
        payload={
            "data": b"Hello, World! This is test data for compression.",
            "base": 7
        },
        metadata={"client": "example_client"}
    )
    
    padic_response = service_interface.invoke_service(padic_request)
    print(f"   P-adic compression: {padic_response.status.value}")
    if padic_response.is_success():
        print(f"     Compression ratio: {padic_response.data['compression_ratio']:.2f}")
        print(f"     Processing time: {padic_response.processing_time:.4f}s")
    
    # Sheaf compression request
    sheaf_request = ServiceRequest(
        service_name="sheaf_compressor",
        method_name="compress", 
        version="1.0.0",
        payload={
            "tensor_data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "cohomology_degree": 2
        },
        metadata={"tensor_type": "float32"}
    )
    
    sheaf_response = service_interface.invoke_service(sheaf_request)
    print(f"   Sheaf compression: {sheaf_response.status.value}")
    if sheaf_response.is_success():
        print(f"     Global sections: {sheaf_response.data['global_sections']}")
        print(f"     Processing time: {sheaf_response.processing_time:.4f}s")
    
    # Example 3: Service contracts and validation
    print(f"\n3. Service Contract Validation:")
    
    # Register contract for P-adic compression
    padic_contract = {
        "required": ["data", "base"],
        "properties": {
            "data": {"type": "string"},
            "base": {"type": "integer", "minimum": 2}
        }
    }
    
    service_interface.register_service_contract(
        "padic_compressor", "compress", padic_contract
    )
    
    # Test valid request
    valid_request = ServiceRequest(
        service_name="padic_compressor",
        method_name="compress",
        version="1.0.0",
        payload={"data": b"test data", "base": 5}
    )
    
    valid_response = service_interface.invoke_service(valid_request)
    print(f"   Valid request: {valid_response.status.value}")
    
    # Test invalid request (missing required field)
    invalid_request = ServiceRequest(
        service_name="padic_compressor",
        method_name="compress",
        version="1.0.0",
        payload={"data": b"test data"}  # Missing 'base'
    )
    
    invalid_response = service_interface.invoke_service(invalid_request)
    print(f"   Invalid request: {invalid_response.status.value}")
    if invalid_response.errors:
        print(f"     Error: {invalid_response.errors[0]['message']}")
    
    # Example 4: Health monitoring
    print(f"\n4. Health Monitoring:")
    
    # Get health for specific service
    padic_health = service_interface.get_service_health("padic_compressor", "1.0.0")
    print(f"   P-adic service health: {padic_health['overall_health']}")
    print(f"   Service info: {padic_health['service']['metadata']}")
    
    # Get overall metrics
    metrics = service_interface.get_metrics_summary()
    print(f"   Total requests: {metrics['total_requests']}")
    print(f"   Success rate: {metrics['success_rate']:.2%}")
    print(f"   Average processing time: {metrics['average_processing_time']:.4f}s")
    
    # Example 5: Middleware demonstration
    print(f"\n5. Custom Middleware:")
    
    # Add custom logging middleware
    def logging_middleware(request: ServiceRequest, next_func):
        print(f"     [LOG] Processing {request.service_name}.{request.method_name}")
        start_time = time.time()
        response = next_func(request)
        end_time = time.time()
        print(f"     [LOG] Completed in {end_time - start_time:.4f}s with status {response.status.value}")
        return response
    
    service_interface.add_middleware(logging_middleware)
    
    # Test with middleware
    middleware_request = ServiceRequest(
        service_name="sheaf_compressor",
        method_name="compress",
        version="1.0.0",
        payload={"tensor_data": [1, 2, 3, 4], "cohomology_degree": 1}
    )
    
    middleware_response = service_interface.invoke_service(middleware_request)
    print(f"   Request with middleware: {middleware_response.status.value}")
    
    # Example 6: Request/Response interceptors
    print(f"\n6. Request/Response Interceptors:")
    
    # Add request interceptor for authentication
    def auth_interceptor(request: ServiceRequest) -> ServiceRequest:
        request.metadata['authenticated'] = True
        request.metadata['auth_timestamp'] = datetime.utcnow().isoformat()
        return request
    
    # Add response interceptor for caching headers
    def cache_interceptor(response: ServiceResponse) -> ServiceResponse:
        response.metadata['cache_policy'] = 'max-age=300'
        response.metadata['cache_timestamp'] = datetime.utcnow().isoformat()
        return response
    
    service_interface.add_request_interceptor(auth_interceptor)
    service_interface.add_response_interceptor(cache_interceptor)
    
    # Test with interceptors
    interceptor_request = ServiceRequest(
        service_name="padic_compressor",
        method_name="compress",
        version="1.0.0",
        payload={"data": b"interceptor test", "base": 3}
    )
    
    interceptor_response = service_interface.invoke_service(interceptor_request)
    print(f"   Request with interceptors: {interceptor_response.status.value}")
    print(f"   Auth metadata: {interceptor_response.metadata.get('cache_policy')}")
    
    # Example 7: Error handling
    print(f"\n7. Error Handling (NO FALLBACKS):")
    
    # Test non-existent service
    try:
        nonexistent_request = ServiceRequest(
            service_name="nonexistent_service",
            method_name="test",
            version="1.0.0",
            payload={}
        )
        error_response = service_interface.invoke_service(nonexistent_request)
        print(f"   Non-existent service: {error_response.status.value}")
        print(f"   Error: {error_response.errors[0]['message']}")
    except Exception as e:
        print(f"   ✓ Correctly raised exception: {e}")
    
    # Test invalid version format
    try:
        invalid_version_request = ServiceRequest(
            service_name="padic_compressor",
            method_name="compress",
            version="invalid.version",
            payload={"data": b"test", "base": 7}
        )
    except ValueError as e:
        print(f"   ✓ Invalid version caught: {e}")
    
    # Test service method error
    class ErrorService:
        def failing_method(self):
            raise RuntimeError("Intentional service error")
    
    service_interface.register_service(
        service_name="error_service",
        service_instance=ErrorService(),
        version="1.0.0"
    )
    
    error_method_request = ServiceRequest(
        service_name="error_service",
        method_name="failing_method",
        version="1.0.0",
        payload={}
    )
    
    error_method_response = service_interface.invoke_service(error_method_request)
    print(f"   Method error: {error_method_response.status.value}")
    print(f"   Error details: {error_method_response.errors[0]['details']}")
    
    print(f"\nService Interface successfully integrated with BrainCore!")
    return brain, service_interface


def demonstrate_compression_system_integration():
    """Demonstrate integration with compression systems"""
    
    print(f"\n=== Compression System Integration ===")
    
    # Mock compression systems
    class MockTensorCompressor:
        def __init__(self):
            self.service_interface = None
            
        def compress_tensor(self, tensor: list, method: str = "tucker") -> Dict[str, Any]:
            return {
                'compressed_tensor': f"tensor_compressed_{len(tensor)}_{method}",
                'method': method,
                'compression_ratio': 0.3,
                'factors': len(tensor) // 3
            }
    
    class MockGPUMemoryManager:
        def __init__(self):
            self.service_interface = None
            
        def allocate_memory(self, size: int, device_id: int = 0) -> Dict[str, Any]:
            return {
                'memory_block': f"gpu_block_{size}_{device_id}",
                'allocated_size': size,
                'device_id': device_id,
                'allocation_time': time.time()
            }
    
    # Create service interface
    service_interface = CompressionServiceInterface()
    
    # Create compression systems
    tensor_compressor = MockTensorCompressor()
    gpu_manager = MockGPUMemoryManager()
    
    # Integrate systems
    ServiceInterfaceIntegration.integrate_with_compression_systems(
        tensor_compressor, service_interface
    )
    ServiceInterfaceIntegration.integrate_with_compression_systems(
        gpu_manager, service_interface
    )
    
    print("Testing integrated compression systems:")
    
    # Test tensor compression service
    try:
        tensor_request = tensor_compressor.create_service_request(
            method="compress_tensor",
            payload={"tensor": [1, 2, 3, 4, 5, 6], "method": "cp"}
        )
        
        tensor_response = tensor_compressor.invoke_service(tensor_request)
        print(f"  ✓ Tensor compression: {tensor_response.status.value}")
        if tensor_response.is_success():
            print(f"    Method: {tensor_response.data['method']}")
            print(f"    Compression ratio: {tensor_response.data['compression_ratio']}")
        
    except Exception as e:
        print(f"  ✗ Tensor compression error: {e}")
    
    # Test GPU memory service
    try:
        gpu_request = gpu_manager.create_service_request(
            method="allocate_memory",
            payload={"size": 1024*1024, "device_id": 0}
        )
        
        gpu_response = gpu_manager.invoke_service(gpu_request)
        print(f"  ✓ GPU memory allocation: {gpu_response.status.value}")
        if gpu_response.is_success():
            print(f"    Allocated size: {gpu_response.data['allocated_size']} bytes")
            print(f"    Device ID: {gpu_response.data['device_id']}")
        
    except Exception as e:
        print(f"  ✗ GPU memory error: {e}")
    
    # List all integrated services
    services = service_interface.list_services()
    print(f"  Integrated services: {list(services.keys())}")
    
    print("Compression system integration successful!")


def demonstrate_api_layer_integration():
    """Demonstrate integration with API layer"""
    
    print(f"\n=== API Layer Integration ===")
    
    # Mock API layer
    class MockAPILayer:
        def __init__(self):
            self.service_interface = None
            self.service_endpoints = {}
        
        def handle_request(self, endpoint: str, *args, **kwargs):
            """Handle API request through service interface"""
            if endpoint not in self.service_endpoints:
                return {"error": f"Endpoint not found: {endpoint}"}
            
            try:
                return self.service_endpoints[endpoint](*args, **kwargs)
            except Exception as e:
                return {"error": str(e)}
    
    # Create components
    service_interface = CompressionServiceInterface()
    api_layer = MockAPILayer()
    
    # Register a test service
    class TestCompressionService:
        def process_data(self, data: str, algorithm: str = "default") -> Dict[str, Any]:
            return {
                'processed_data': f"processed_{data}_{algorithm}",
                'algorithm': algorithm,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    service_interface.register_service(
        service_name="test_compression",
        service_instance=TestCompressionService(),
        version="1.0.0"
    )
    
    # Integrate API layer
    ServiceInterfaceIntegration.integrate_with_api_layer(api_layer, service_interface)
    
    print("Testing API layer integration:")
    
    # Test service invocation through API
    test_request = ServiceRequest(
        service_name="test_compression",
        method_name="process_data",
        version="1.0.0",
        payload={"data": "test_input", "algorithm": "advanced"}
    )
    
    api_response = api_layer.handle_request("invoke", test_request)
    print(f"  ✓ API service invocation: {api_response.status.value}")
    
    # Test service listing through API
    services_list = api_layer.handle_request("list_services")
    print(f"  ✓ API service listing: {list(services_list.keys())}")
    
    # Test health check through API
    health_info = api_layer.handle_request("get_health", "test_compression", "1.0.0")
    print(f"  ✓ API health check: {health_info['overall_health']}")
    
    # Test metrics through API
    metrics_info = api_layer.handle_request("get_metrics")
    print(f"  ✓ API metrics: {metrics_info['total_requests']} requests")
    
    print("API layer integration successful!")


def demonstrate_advanced_features():
    """Demonstrate advanced service interface features"""
    
    print(f"\n=== Advanced Features ===")
    
    service_interface = CompressionServiceInterface()
    
    # Example 1: Version compatibility
    print("1. Version Compatibility:")
    
    class VersionedService:
        def v1_method(self):
            return {"version": "1.0.0", "features": ["basic"]}
        
        def v2_method(self):
            return {"version": "2.0.0", "features": ["basic", "advanced"]}
    
    # Register multiple versions
    service_v1 = VersionedService()
    service_v2 = VersionedService()
    
    service_interface.register_service("versioned_service", service_v1, "1.0.0")
    service_interface.register_service("versioned_service", service_v2, "2.0.0")
    
    # Test version compatibility
    v1_request = ServiceRequest(
        service_name="versioned_service",
        method_name="v1_method",
        version="1.0.0",
        payload={}
    )
    
    v2_request = ServiceRequest(
        service_name="versioned_service", 
        method_name="v2_method",
        version="2.0.0",
        payload={}
    )
    
    v1_response = service_interface.invoke_service(v1_request)
    v2_response = service_interface.invoke_service(v2_request)
    
    print(f"  Version 1.0.0: {v1_response.data['features'] if v1_response.is_success() else 'Error'}")
    print(f"  Version 2.0.0: {v2_response.data['features'] if v2_response.is_success() else 'Error'}")
    
    # Example 2: Service health monitoring
    print(f"\n2. Service Health Monitoring:")
    
    # Simulate service with varying health
    class HealthMonitoredService:
        def __init__(self):
            self.call_count = 0
        
        def monitored_method(self):
            self.call_count += 1
            if self.call_count % 5 == 0:  # Fail every 5th call
                raise RuntimeError("Simulated service failure")
            return {"call_count": self.call_count, "status": "success"}
    
    health_service = HealthMonitoredService()
    service_interface.register_service("health_monitored", health_service, "1.0.0")
    
    # Make several calls to generate metrics
    for i in range(7):
        health_request = ServiceRequest(
            service_name="health_monitored",
            method_name="monitored_method",
            version="1.0.0",
            payload={}
        )
        response = service_interface.invoke_service(health_request)
    
    # Check health
    health_info = service_interface.get_service_health("health_monitored", "1.0.0")
    print(f"  Service health: {health_info['overall_health']}")
    print(f"  Total calls: {sum(m['count'] for m in health_info['metrics'].values())}")
    print(f"  Error rate: {sum(m['error'] for m in health_info['metrics'].values()) / sum(m['count'] for m in health_info['metrics'].values()):.2%}")
    
    print("Advanced features demonstration complete!")


if __name__ == "__main__":
    # Run integration demonstrations
    brain, service_interface = integrate_service_interface_with_brain_core()
    
    # Run compression system integration
    demonstrate_compression_system_integration()
    
    # Run API layer integration
    demonstrate_api_layer_integration()
    
    # Run advanced features
    demonstrate_advanced_features()
    
    print(f"\n=== All Service Interface Integration Tests Complete ===")