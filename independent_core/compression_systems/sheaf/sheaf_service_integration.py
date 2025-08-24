"""
Sheaf Service Integration and Validation Components

This module provides service integration, validation, and metrics
for the Sheaf compression system within the Saraphis framework.
"""

import time
import logging
from typing import Any, Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import threading
import numpy as np

try:
    from compression_systems.service_interfaces.service_interfaces_core import (
        CompressionServiceInterface, ServiceRequest, ServiceResponse, ServiceMetrics
    )
except ImportError:
    # Define minimal stubs for testing
    CompressionServiceInterface = None
    ServiceRequest = None
    ServiceResponse = None
    ServiceMetrics = None

logger = logging.getLogger(__name__)


class SheafServiceIntegration:
    """Manages integration of Sheaf services with the broader system."""
    
    def __init__(self, sheaf_system: Any, config: Optional[Dict[str, Any]] = None):
        if sheaf_system is None:
            raise ValueError("Sheaf system cannot be None")
            
        self.sheaf_system = sheaf_system
        self.config = config or {}
        
        # Integration state
        self.domain_router = None
        self.brain_core = None
        self.training_manager = None
        self.service_endpoints: Dict[str, Any] = {}
        self.integration_status: Dict[str, bool] = {
            'domain_router': False,
            'brain_core': False,
            'training_manager': False,
            'service_endpoints': False
        }
        
        # Service registry
        self.registered_services: Set[str] = set()
        self.service_handlers: Dict[str, Any] = {}
        
        # Thread safety
        self._integration_lock = threading.RLock()
        
        logger.info("Initialized SheafServiceIntegration")
    
    def integrate_with_domain_router(self, domain_router: Any) -> None:
        """Integrate with the domain router for cross-domain communication."""
        if domain_router is None:
            raise ValueError("Domain router cannot be None")
            
        with self._integration_lock:
            # Validate domain router interface
            required_methods = ['register_domain', 'route_request', 'get_domain_status']
            for method in required_methods:
                if not hasattr(domain_router, method):
                    raise TypeError(f"Domain router missing required method: {method}")
            
            self.domain_router = domain_router
            
            # Register sheaf domain with router
            domain_config = {
                'domain_name': 'sheaf_compression',
                'domain_type': 'compression',
                'supported_operations': [
                    'compress', 'decompress', 'compute_cohomology',
                    'apply_restriction_map', 'compute_sheaf_morphism'
                ],
                'performance_characteristics': {
                    'latency_ms': self.config.get('expected_latency_ms', 10),
                    'throughput_mbps': self.config.get('expected_throughput_mbps', 1000),
                    'memory_footprint_mb': self.config.get('memory_footprint_mb', 512)
                }
            }
            
            self.domain_router.register_domain('sheaf_compression', domain_config)
            
            # Set up request routing
            self.domain_router.set_handler('sheaf_compression', self._handle_routed_request)
            
            self.integration_status['domain_router'] = True
            logger.info("Successfully integrated with domain router")
    
    def integrate_with_brain_core(self, brain_core: Any) -> None:
        """Integrate with the brain core for neural enhancement."""
        if brain_core is None:
            raise ValueError("Brain core cannot be None")
            
        with self._integration_lock:
            # Validate brain core interface
            required_methods = ['register_predictor', 'get_prediction', 'update_model']
            for method in required_methods:
                if not hasattr(brain_core, method):
                    raise TypeError(f"Brain core missing required method: {method}")
            
            self.brain_core = brain_core
            
            # Register sheaf predictors
            predictor_configs = [
                {
                    'name': 'sheaf_compression_predictor',
                    'input_shape': (None,),  # Variable length input
                    'output_shape': (None,),  # Variable length output
                    'model_type': 'adaptive',
                    'update_frequency': 100
                },
                {
                    'name': 'cohomology_predictor',
                    'input_shape': (None, None),  # Variable matrix input
                    'output_shape': (None,),  # Cohomology dimension vector
                    'model_type': 'topological',
                    'update_frequency': 50
                }
            ]
            
            for config in predictor_configs:
                self.brain_core.register_predictor(config['name'], config)
            
            # Set up prediction callbacks
            self.brain_core.set_callback('sheaf_compression_predictor', 
                                       self._handle_compression_prediction)
            self.brain_core.set_callback('cohomology_predictor',
                                       self._handle_cohomology_prediction)
            
            self.integration_status['brain_core'] = True
            logger.info("Successfully integrated with brain core")
    
    def integrate_with_training_manager(self, training_manager: Any) -> None:
        """Integrate with the training manager for model updates."""
        if training_manager is None:
            raise ValueError("Training manager cannot be None")
            
        with self._integration_lock:
            # Validate training manager interface
            required_methods = ['register_trainable', 'schedule_training', 'get_training_status']
            for method in required_methods:
                if not hasattr(training_manager, method):
                    raise TypeError(f"Training manager missing required method: {method}")
            
            self.training_manager = training_manager
            
            # Register trainable components
            trainable_components = [
                {
                    'component_id': 'sheaf_compression_model',
                    'component_type': 'compression',
                    'training_data_requirements': {
                        'min_samples': 1000,
                        'data_format': 'structured',
                        'features_required': ['input_data', 'compressed_output', 'compression_ratio']
                    },
                    'update_callback': self._update_compression_model
                },
                {
                    'component_id': 'cohomology_calculator',
                    'component_type': 'topological',
                    'training_data_requirements': {
                        'min_samples': 500,
                        'data_format': 'topological',
                        'features_required': ['sheaf_structure', 'cohomology_groups']
                    },
                    'update_callback': self._update_cohomology_calculator
                }
            ]
            
            for component in trainable_components:
                self.training_manager.register_trainable(
                    component['component_id'], component
                )
            
            # Schedule initial training
            self.training_manager.schedule_training('sheaf_compression_model', 
                                                  priority='high')
            self.training_manager.schedule_training('cohomology_calculator',
                                                  priority='medium')
            
            self.integration_status['training_manager'] = True
            logger.info("Successfully integrated with training manager")
    
    def register_service_endpoints(self) -> None:
        """Register all sheaf service endpoints."""
        with self._integration_lock:
            # Define service endpoints
            endpoints = {
                'sheaf.compress': {
                    'handler': self._handle_compress_request,
                    'input_validation': 'strict',
                    'output_format': 'compressed_sheaf',
                    'performance_tier': 'high'
                },
                'sheaf.decompress': {
                    'handler': self._handle_decompress_request,
                    'input_validation': 'strict',
                    'output_format': 'original_data',
                    'performance_tier': 'high'
                },
                'sheaf.compute_cohomology': {
                    'handler': self._handle_cohomology_request,
                    'input_validation': 'mathematical',
                    'output_format': 'cohomology_groups',
                    'performance_tier': 'compute_intensive'
                },
                'sheaf.apply_restriction': {
                    'handler': self._handle_restriction_request,
                    'input_validation': 'topological',
                    'output_format': 'restricted_sheaf',
                    'performance_tier': 'medium'
                },
                'sheaf.compute_morphism': {
                    'handler': self._handle_morphism_request,
                    'input_validation': 'categorical',
                    'output_format': 'sheaf_morphism',
                    'performance_tier': 'compute_intensive'
                }
            }
            
            # Register each endpoint
            for endpoint_name, endpoint_config in endpoints.items():
                self.service_endpoints[endpoint_name] = endpoint_config
                self.registered_services.add(endpoint_name)
                self.service_handlers[endpoint_name] = endpoint_config['handler']
                
                logger.debug(f"Registered endpoint: {endpoint_name}")
            
            self.integration_status['service_endpoints'] = True
            logger.info(f"Registered {len(endpoints)} service endpoints")
    
    def validate_integration(self) -> bool:
        """Validate that all integrations are properly configured."""
        with self._integration_lock:
            # Check all integration points
            all_integrated = all(self.integration_status.values())
            
            if not all_integrated:
                missing = [k for k, v in self.integration_status.items() if not v]
                logger.warning(f"Missing integrations: {missing}")
                return False
            
            # Validate domain router connection
            if self.domain_router:
                try:
                    status = self.domain_router.get_domain_status('sheaf_compression')
                    if status.get('status') != 'active':
                        logger.error("Domain router integration not active")
                        return False
                except Exception as e:
                    logger.error(f"Domain router validation failed: {e}")
                    return False
            
            # Validate brain core connection
            if self.brain_core:
                try:
                    # Test prediction
                    test_input = np.random.randn(10)
                    prediction = self.brain_core.get_prediction(
                        'sheaf_compression_predictor', test_input
                    )
                    if prediction is None:
                        logger.error("Brain core prediction test failed")
                        return False
                except Exception as e:
                    logger.error(f"Brain core validation failed: {e}")
                    return False
            
            # Validate training manager connection
            if self.training_manager:
                try:
                    status = self.training_manager.get_training_status(
                        'sheaf_compression_model'
                    )
                    if status is None:
                        logger.error("Training manager status check failed")
                        return False
                except Exception as e:
                    logger.error(f"Training manager validation failed: {e}")
                    return False
            
            # Validate service endpoints
            if not self.service_endpoints:
                logger.error("No service endpoints registered")
                return False
            
            logger.info("All integrations validated successfully")
            return True
    
    # Private handler methods
    def _handle_routed_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle requests routed from domain router."""
        operation = request.get('operation')
        if operation not in self.service_handlers:
            raise ValueError(f"Unknown operation: {operation}")
        
        handler = self.service_handlers[operation]
        return handler(request)
    
    def _handle_compress_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle compression requests."""
        data = request.get('data')
        if data is None:
            raise ValueError("No data provided for compression")
        
        params = request.get('parameters', {})
        compressed = self.sheaf_system.compress(data, **params)
        
        return {
            'status': 'success',
            'compressed_data': compressed,
            'compression_ratio': len(data) / len(compressed) if compressed else 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def _handle_decompress_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle decompression requests."""
        compressed_data = request.get('compressed_data')
        if compressed_data is None:
            raise ValueError("No compressed data provided")
        
        params = request.get('parameters', {})
        decompressed = self.sheaf_system.decompress(compressed_data, **params)
        
        return {
            'status': 'success',
            'data': decompressed,
            'timestamp': datetime.now().isoformat()
        }
    
    def _handle_cohomology_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cohomology computation requests."""
        sheaf_data = request.get('sheaf_data')
        if sheaf_data is None:
            raise ValueError("No sheaf data provided")
        
        degree = request.get('degree', 0)
        cohomology = self.sheaf_system.compute_cohomology(sheaf_data, degree)
        
        return {
            'status': 'success',
            'cohomology_groups': cohomology,
            'degree': degree,
            'timestamp': datetime.now().isoformat()
        }
    
    def _handle_restriction_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle restriction map requests."""
        sheaf_data = request.get('sheaf_data')
        subset = request.get('subset')
        
        if sheaf_data is None or subset is None:
            raise ValueError("Missing sheaf data or subset specification")
        
        restricted = self.sheaf_system.apply_restriction(sheaf_data, subset)
        
        return {
            'status': 'success',
            'restricted_sheaf': restricted,
            'subset': subset,
            'timestamp': datetime.now().isoformat()
        }
    
    def _handle_morphism_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sheaf morphism computation requests."""
        source_sheaf = request.get('source_sheaf')
        target_sheaf = request.get('target_sheaf')
        
        if source_sheaf is None or target_sheaf is None:
            raise ValueError("Missing source or target sheaf")
        
        morphism = self.sheaf_system.compute_morphism(source_sheaf, target_sheaf)
        
        return {
            'status': 'success',
            'morphism': morphism,
            'timestamp': datetime.now().isoformat()
        }
    
    def _handle_compression_prediction(self, input_data: Any) -> Any:
        """Handle compression predictions from brain core."""
        # Use brain core predictions to optimize compression
        return self.sheaf_system.predict_compression_ratio(input_data)
    
    def _handle_cohomology_prediction(self, sheaf_structure: Any) -> Any:
        """Handle cohomology predictions from brain core."""
        # Use brain core to predict cohomology dimensions
        return self.sheaf_system.predict_cohomology_dimension(sheaf_structure)
    
    def _update_compression_model(self, training_data: Dict[str, Any]) -> None:
        """Update compression model based on training data."""
        self.sheaf_system.update_compression_model(training_data)
    
    def _update_cohomology_calculator(self, training_data: Dict[str, Any]) -> None:
        """Update cohomology calculator based on training data."""
        self.sheaf_system.update_cohomology_calculator(training_data)


class SheafServiceValidation:
    """Validates sheaf service requests and responses."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Validation rules
        self.compression_rules = {
            'max_input_size': self.config.get('max_input_size', 100 * 1024 * 1024),  # 100MB
            'min_input_size': self.config.get('min_input_size', 1),
            'supported_formats': {'binary', 'text', 'structured', 'tensor'},
            'required_fields': {'data', 'format'}
        }
        
        self.decompression_rules = {
            'required_fields': {'compressed_data', 'compression_metadata'},
            'max_compressed_size': self.config.get('max_compressed_size', 50 * 1024 * 1024)
        }
        
        self.cohomology_rules = {
            'required_fields': {'sheaf_data', 'degree'},
            'max_degree': self.config.get('max_cohomology_degree', 10),
            'min_degree': 0
        }
        
        logger.info("Initialized SheafServiceValidation")
    
    def validate_compression_request(self, request: ServiceRequest) -> Tuple[bool, List[str]]:
        """Validate a compression request."""
        errors = []
        
        if request is None:
            return False, ["Request is None"]
        
        # Check required fields
        for field in self.compression_rules['required_fields']:
            if not hasattr(request, field) or getattr(request, field) is None:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        # Validate data size
        data = getattr(request, 'data', None)
        if data is not None:
            data_size = len(data) if hasattr(data, '__len__') else 0
            
            if data_size > self.compression_rules['max_input_size']:
                errors.append(f"Data size {data_size} exceeds maximum {self.compression_rules['max_input_size']}")
            
            if data_size < self.compression_rules['min_input_size']:
                errors.append(f"Data size {data_size} below minimum {self.compression_rules['min_input_size']}")
        
        # Validate format
        format_type = getattr(request, 'format', None)
        if format_type and format_type not in self.compression_rules['supported_formats']:
            errors.append(f"Unsupported format: {format_type}")
        
        # Validate parameters
        params = getattr(request, 'parameters', {})
        if params:
            param_errors = self._validate_compression_parameters(params)
            errors.extend(param_errors)
        
        return len(errors) == 0, errors
    
    def validate_decompression_request(self, request: ServiceRequest) -> Tuple[bool, List[str]]:
        """Validate a decompression request."""
        errors = []
        
        if request is None:
            return False, ["Request is None"]
        
        # Check required fields
        for field in self.decompression_rules['required_fields']:
            if not hasattr(request, field) or getattr(request, field) is None:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        # Validate compressed data size
        compressed_data = getattr(request, 'compressed_data', None)
        if compressed_data is not None:
            data_size = len(compressed_data) if hasattr(compressed_data, '__len__') else 0
            
            if data_size > self.decompression_rules['max_compressed_size']:
                errors.append(f"Compressed data size {data_size} exceeds maximum")
        
        # Validate metadata
        metadata = getattr(request, 'compression_metadata', None)
        if metadata:
            if not isinstance(metadata, dict):
                errors.append("Compression metadata must be a dictionary")
            else:
                required_meta = ['compression_algorithm', 'original_size', 'compressed_size']
                for field in required_meta:
                    if field not in metadata:
                        errors.append(f"Missing metadata field: {field}")
        
        return len(errors) == 0, errors
    
    def validate_cohomology_request(self, request: ServiceRequest) -> Tuple[bool, List[str]]:
        """Validate a cohomology computation request."""
        errors = []
        
        if request is None:
            return False, ["Request is None"]
        
        # Check required fields
        for field in self.cohomology_rules['required_fields']:
            if not hasattr(request, field) or getattr(request, field) is None:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        # Validate degree
        degree = getattr(request, 'degree', None)
        if degree is not None:
            if not isinstance(degree, int):
                errors.append("Degree must be an integer")
            elif degree < self.cohomology_rules['min_degree']:
                errors.append(f"Degree {degree} below minimum {self.cohomology_rules['min_degree']}")
            elif degree > self.cohomology_rules['max_degree']:
                errors.append(f"Degree {degree} exceeds maximum {self.cohomology_rules['max_degree']}")
        
        # Validate sheaf data structure
        sheaf_data = getattr(request, 'sheaf_data', None)
        if sheaf_data:
            sheaf_errors = self._validate_sheaf_structure(sheaf_data)
            errors.extend(sheaf_errors)
        
        return len(errors) == 0, errors
    
    def validate_service_response(self, response: ServiceResponse) -> Tuple[bool, List[str]]:
        """Validate a service response."""
        errors = []
        
        if response is None:
            return False, ["Response is None"]
        
        # Check basic response structure
        if not hasattr(response, 'status'):
            errors.append("Response missing status field")
        elif response.status not in ['success', 'error', 'partial']:
            errors.append(f"Invalid response status: {response.status}")
        
        # For error responses, check error details
        if hasattr(response, 'status') and response.status == 'error':
            if not hasattr(response, 'error_message'):
                errors.append("Error response missing error_message")
            if not hasattr(response, 'error_code'):
                errors.append("Error response missing error_code")
        
        # For success responses, check data
        if hasattr(response, 'status') and response.status == 'success':
            if not hasattr(response, 'data') and not hasattr(response, 'result'):
                errors.append("Success response missing data or result")
        
        # Check processing metadata
        if hasattr(response, 'processing_time'):
            if not isinstance(response.processing_time, (int, float)):
                errors.append("Processing time must be numeric")
            elif response.processing_time < 0:
                errors.append("Processing time cannot be negative")
        
        return len(errors) == 0, errors
    
    def validate_sheaf_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate sheaf-specific parameters."""
        errors = []
        
        if not isinstance(parameters, dict):
            return False, ["Parameters must be a dictionary"]
        
        # Validate cohomological dimension
        if 'cohomological_dimension' in parameters:
            dim = parameters['cohomological_dimension']
            if not isinstance(dim, int) or dim < 0:
                errors.append("Cohomological dimension must be non-negative integer")
        
        # Validate sheaf type
        if 'sheaf_type' in parameters:
            valid_types = {'constant', 'constructible', 'perverse', 'coherent'}
            if parameters['sheaf_type'] not in valid_types:
                errors.append(f"Invalid sheaf type: {parameters['sheaf_type']}")
        
        # Validate restriction maps
        if 'restriction_maps' in parameters:
            maps = parameters['restriction_maps']
            if not isinstance(maps, dict):
                errors.append("Restriction maps must be a dictionary")
            else:
                for key, value in maps.items():
                    if not isinstance(key, str):
                        errors.append(f"Restriction map key must be string: {key}")
                    if not callable(value) and not isinstance(value, dict):
                        errors.append(f"Restriction map value must be callable or dict: {key}")
        
        # Validate compression level
        if 'compression_level' in parameters:
            level = parameters['compression_level']
            if not isinstance(level, (int, float)) or level < 0 or level > 1:
                errors.append("Compression level must be between 0 and 1")
        
        # Validate topological parameters
        if 'topology' in parameters:
            topo = parameters['topology']
            if not isinstance(topo, dict):
                errors.append("Topology must be a dictionary")
            else:
                if 'open_sets' in topo and not isinstance(topo['open_sets'], list):
                    errors.append("Open sets must be a list")
                if 'base' in topo and not isinstance(topo['base'], list):
                    errors.append("Base must be a list")
        
        return len(errors) == 0, errors
    
    # Private validation helpers
    def _validate_compression_parameters(self, params: Dict[str, Any]) -> List[str]:
        """Validate compression-specific parameters."""
        errors = []
        
        # Validate chunk size
        if 'chunk_size' in params:
            chunk_size = params['chunk_size']
            if not isinstance(chunk_size, int) or chunk_size <= 0:
                errors.append("Chunk size must be positive integer")
        
        # Validate algorithm
        if 'algorithm' in params:
            valid_algorithms = {'sheaf_standard', 'sheaf_fast', 'sheaf_maximum'}
            if params['algorithm'] not in valid_algorithms:
                errors.append(f"Invalid compression algorithm: {params['algorithm']}")
        
        return errors
    
    def _validate_sheaf_structure(self, sheaf_data: Any) -> List[str]:
        """Validate sheaf data structure."""
        errors = []
        
        if not isinstance(sheaf_data, dict):
            errors.append("Sheaf data must be a dictionary")
            return errors
        
        # Check required sheaf components
        required_components = ['sections', 'topology', 'restriction_maps']
        for component in required_components:
            if component not in sheaf_data:
                errors.append(f"Missing sheaf component: {component}")
        
        # Validate sections
        if 'sections' in sheaf_data:
            sections = sheaf_data['sections']
            if not isinstance(sections, dict):
                errors.append("Sections must be a dictionary")
        
        # Validate topology
        if 'topology' in sheaf_data:
            topology = sheaf_data['topology']
            if not isinstance(topology, dict):
                errors.append("Topology must be a dictionary")
            elif 'open_sets' not in topology:
                errors.append("Topology missing open_sets")
        
        return errors


class SheafServiceMetrics:
    """Tracks metrics for sheaf service operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Metrics storage
        self.compression_metrics = defaultdict(lambda: {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_bytes_input': 0,
            'total_bytes_output': 0,
            'total_processing_time': 0.0,
            'compression_ratios': [],
            'error_types': defaultdict(int)
        })
        
        self.decompression_metrics = defaultdict(lambda: {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_bytes_input': 0,
            'total_bytes_output': 0,
            'total_processing_time': 0.0,
            'error_types': defaultdict(int)
        })
        
        self.cohomology_metrics = defaultdict(lambda: {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0.0,
            'degree_distribution': defaultdict(int),
            'dimension_distribution': defaultdict(int),
            'error_types': defaultdict(int)
        })
        
        # Time-based metrics
        self.hourly_metrics = defaultdict(lambda: {
            'compression': {'requests': 0, 'bytes': 0},
            'decompression': {'requests': 0, 'bytes': 0},
            'cohomology': {'requests': 0, 'computations': 0}
        })
        
        # Performance tracking
        self.performance_history = {
            'compression_latency': [],
            'decompression_latency': [],
            'cohomology_latency': []
        }
        
        # Thread safety
        self._metrics_lock = threading.RLock()
        
        logger.info("Initialized SheafServiceMetrics")
    
    def record_compression_request(self, request: ServiceRequest, response: ServiceResponse) -> None:
        """Record metrics for a compression request."""
        with self._metrics_lock:
            # Determine time bucket
            hour_bucket = datetime.now().strftime("%Y-%m-%d-%H")
            
            # Extract metrics
            input_size = len(request.data) if hasattr(request, 'data') else 0
            output_size = len(response.data) if hasattr(response, 'data') else 0
            processing_time = getattr(response, 'processing_time', 0.0)
            
            # Update compression metrics
            metrics = self.compression_metrics[hour_bucket]
            metrics['total_requests'] += 1
            
            if response.status == 'success':
                metrics['successful_requests'] += 1
                metrics['total_bytes_input'] += input_size
                metrics['total_bytes_output'] += output_size
                metrics['total_processing_time'] += processing_time
                
                # Calculate compression ratio
                if input_size > 0:
                    compression_ratio = input_size / output_size if output_size > 0 else float('inf')
                    metrics['compression_ratios'].append(compression_ratio)
                
                # Update performance history
                self.performance_history['compression_latency'].append(processing_time)
                if len(self.performance_history['compression_latency']) > 1000:
                    self.performance_history['compression_latency'].pop(0)
            else:
                metrics['failed_requests'] += 1
                error_type = getattr(response, 'error_code', 'unknown')
                metrics['error_types'][error_type] += 1
            
            # Update hourly metrics
            self.hourly_metrics[hour_bucket]['compression']['requests'] += 1
            self.hourly_metrics[hour_bucket]['compression']['bytes'] += input_size
    
    def record_decompression_request(self, request: ServiceRequest, response: ServiceResponse) -> None:
        """Record metrics for a decompression request."""
        with self._metrics_lock:
            # Determine time bucket
            hour_bucket = datetime.now().strftime("%Y-%m-%d-%H")
            
            # Extract metrics
            input_size = len(request.compressed_data) if hasattr(request, 'compressed_data') else 0
            output_size = len(response.data) if hasattr(response, 'data') else 0
            processing_time = getattr(response, 'processing_time', 0.0)
            
            # Update decompression metrics
            metrics = self.decompression_metrics[hour_bucket]
            metrics['total_requests'] += 1
            
            if response.status == 'success':
                metrics['successful_requests'] += 1
                metrics['total_bytes_input'] += input_size
                metrics['total_bytes_output'] += output_size
                metrics['total_processing_time'] += processing_time
                
                # Update performance history
                self.performance_history['decompression_latency'].append(processing_time)
                if len(self.performance_history['decompression_latency']) > 1000:
                    self.performance_history['decompression_latency'].pop(0)
            else:
                metrics['failed_requests'] += 1
                error_type = getattr(response, 'error_code', 'unknown')
                metrics['error_types'][error_type] += 1
            
            # Update hourly metrics
            self.hourly_metrics[hour_bucket]['decompression']['requests'] += 1
            self.hourly_metrics[hour_bucket]['decompression']['bytes'] += output_size
    
    def record_cohomology_request(self, request: ServiceRequest, response: ServiceResponse) -> None:
        """Record metrics for a cohomology computation request."""
        with self._metrics_lock:
            # Determine time bucket
            hour_bucket = datetime.now().strftime("%Y-%m-%d-%H")
            
            # Extract metrics
            degree = getattr(request, 'degree', 0)
            processing_time = getattr(response, 'processing_time', 0.0)
            
            # Update cohomology metrics
            metrics = self.cohomology_metrics[hour_bucket]
            metrics['total_requests'] += 1
            
            if response.status == 'success':
                metrics['successful_requests'] += 1
                metrics['total_processing_time'] += processing_time
                metrics['degree_distribution'][degree] += 1
                
                # Record dimension if available
                if hasattr(response, 'cohomology_groups'):
                    dimension = len(response.cohomology_groups)
                    metrics['dimension_distribution'][dimension] += 1
                
                # Update performance history
                self.performance_history['cohomology_latency'].append(processing_time)
                if len(self.performance_history['cohomology_latency']) > 1000:
                    self.performance_history['cohomology_latency'].pop(0)
            else:
                metrics['failed_requests'] += 1
                error_type = getattr(response, 'error_code', 'unknown')
                metrics['error_types'][error_type] += 1
            
            # Update hourly metrics
            self.hourly_metrics[hour_bucket]['cohomology']['requests'] += 1
            self.hourly_metrics[hour_bucket]['cohomology']['computations'] += 1
    
    def get_compression_metrics(self) -> Dict[str, Any]:
        """Get aggregated compression metrics."""
        with self._metrics_lock:
            # Aggregate metrics across all time buckets
            total_metrics = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'total_bytes_compressed': 0,
                'total_bytes_output': 0,
                'average_compression_ratio': 0.0,
                'average_processing_time': 0.0,
                'success_rate': 0.0,
                'error_distribution': {}
            }
            
            all_ratios = []
            all_processing_times = []
            error_totals = defaultdict(int)
            
            for bucket_metrics in self.compression_metrics.values():
                total_metrics['total_requests'] += bucket_metrics['total_requests']
                total_metrics['successful_requests'] += bucket_metrics['successful_requests']
                total_metrics['failed_requests'] += bucket_metrics['failed_requests']
                total_metrics['total_bytes_compressed'] += bucket_metrics['total_bytes_input']
                total_metrics['total_bytes_output'] += bucket_metrics['total_bytes_output']
                
                all_ratios.extend(bucket_metrics['compression_ratios'])
                if bucket_metrics['successful_requests'] > 0:
                    avg_time = bucket_metrics['total_processing_time'] / bucket_metrics['successful_requests']
                    all_processing_times.append(avg_time)
                
                for error_type, count in bucket_metrics['error_types'].items():
                    error_totals[error_type] += count
            
            # Calculate averages
            if all_ratios:
                total_metrics['average_compression_ratio'] = sum(all_ratios) / len(all_ratios)
            
            if all_processing_times:
                total_metrics['average_processing_time'] = sum(all_processing_times) / len(all_processing_times)
            
            if total_metrics['total_requests'] > 0:
                total_metrics['success_rate'] = (
                    total_metrics['successful_requests'] / total_metrics['total_requests']
                )
            
            total_metrics['error_distribution'] = dict(error_totals)
            
            # Add performance percentiles
            if self.performance_history['compression_latency']:
                latencies = sorted(self.performance_history['compression_latency'])
                total_metrics['latency_p50'] = latencies[len(latencies) // 2]
                total_metrics['latency_p95'] = latencies[int(len(latencies) * 0.95)]
                total_metrics['latency_p99'] = latencies[int(len(latencies) * 0.99)]
            
            return total_metrics
    
    def get_decompression_metrics(self) -> Dict[str, Any]:
        """Get aggregated decompression metrics."""
        with self._metrics_lock:
            # Aggregate metrics across all time buckets
            total_metrics = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'total_bytes_input': 0,
                'total_bytes_decompressed': 0,
                'average_processing_time': 0.0,
                'success_rate': 0.0,
                'error_distribution': {}
            }
            
            all_processing_times = []
            error_totals = defaultdict(int)
            
            for bucket_metrics in self.decompression_metrics.values():
                total_metrics['total_requests'] += bucket_metrics['total_requests']
                total_metrics['successful_requests'] += bucket_metrics['successful_requests']
                total_metrics['failed_requests'] += bucket_metrics['failed_requests']
                total_metrics['total_bytes_input'] += bucket_metrics['total_bytes_input']
                total_metrics['total_bytes_decompressed'] += bucket_metrics['total_bytes_output']
                
                if bucket_metrics['successful_requests'] > 0:
                    avg_time = bucket_metrics['total_processing_time'] / bucket_metrics['successful_requests']
                    all_processing_times.append(avg_time)
                
                for error_type, count in bucket_metrics['error_types'].items():
                    error_totals[error_type] += count
            
            # Calculate averages
            if all_processing_times:
                total_metrics['average_processing_time'] = sum(all_processing_times) / len(all_processing_times)
            
            if total_metrics['total_requests'] > 0:
                total_metrics['success_rate'] = (
                    total_metrics['successful_requests'] / total_metrics['total_requests']
                )
            
            total_metrics['error_distribution'] = dict(error_totals)
            
            # Add performance percentiles
            if self.performance_history['decompression_latency']:
                latencies = sorted(self.performance_history['decompression_latency'])
                total_metrics['latency_p50'] = latencies[len(latencies) // 2]
                total_metrics['latency_p95'] = latencies[int(len(latencies) * 0.95)]
                total_metrics['latency_p99'] = latencies[int(len(latencies) * 0.99)]
            
            return total_metrics
    
    def get_cohomology_metrics(self) -> Dict[str, Any]:
        """Get aggregated cohomology computation metrics."""
        with self._metrics_lock:
            # Aggregate metrics across all time buckets
            total_metrics = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'average_processing_time': 0.0,
                'success_rate': 0.0,
                'degree_distribution': {},
                'dimension_distribution': {},
                'error_distribution': {}
            }
            
            all_processing_times = []
            degree_totals = defaultdict(int)
            dimension_totals = defaultdict(int)
            error_totals = defaultdict(int)
            
            for bucket_metrics in self.cohomology_metrics.values():
                total_metrics['total_requests'] += bucket_metrics['total_requests']
                total_metrics['successful_requests'] += bucket_metrics['successful_requests']
                total_metrics['failed_requests'] += bucket_metrics['failed_requests']
                
                if bucket_metrics['successful_requests'] > 0:
                    avg_time = bucket_metrics['total_processing_time'] / bucket_metrics['successful_requests']
                    all_processing_times.append(avg_time)
                
                for degree, count in bucket_metrics['degree_distribution'].items():
                    degree_totals[degree] += count
                
                for dimension, count in bucket_metrics['dimension_distribution'].items():
                    dimension_totals[dimension] += count
                
                for error_type, count in bucket_metrics['error_types'].items():
                    error_totals[error_type] += count
            
            # Calculate averages
            if all_processing_times:
                total_metrics['average_processing_time'] = sum(all_processing_times) / len(all_processing_times)
            
            if total_metrics['total_requests'] > 0:
                total_metrics['success_rate'] = (
                    total_metrics['successful_requests'] / total_metrics['total_requests']
                )
            
            total_metrics['degree_distribution'] = dict(degree_totals)
            total_metrics['dimension_distribution'] = dict(dimension_totals)
            total_metrics['error_distribution'] = dict(error_totals)
            
            # Add performance percentiles
            if self.performance_history['cohomology_latency']:
                latencies = sorted(self.performance_history['cohomology_latency'])
                total_metrics['latency_p50'] = latencies[len(latencies) // 2]
                total_metrics['latency_p95'] = latencies[int(len(latencies) * 0.95)]
                total_metrics['latency_p99'] = latencies[int(len(latencies) * 0.99)]
            
            return total_metrics
    
    def get_hourly_metrics(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get hourly metrics for the last N hours."""
        with self._metrics_lock:
            from datetime import datetime, timedelta
            
            current_time = datetime.now()
            hourly_data = {}
            
            for i in range(hours_back):
                hour = current_time - timedelta(hours=i)
                hour_key = hour.strftime("%Y-%m-%d-%H")
                
                if hour_key in self.hourly_metrics:
                    hourly_data[hour_key] = self.hourly_metrics[hour_key]
                else:
                    hourly_data[hour_key] = {
                        'compression': {'requests': 0, 'bytes': 0},
                        'decompression': {'requests': 0, 'bytes': 0},
                        'cohomology': {'requests': 0, 'computations': 0}
                    }
            
            return hourly_data
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        with self._metrics_lock:
            summary = {}
            
            for operation_type in ['compression', 'decompression', 'cohomology']:
                latency_key = f'{operation_type}_latency'
                if latency_key in self.performance_history:
                    latencies = self.performance_history[latency_key]
                    if latencies:
                        sorted_latencies = sorted(latencies)
                        summary[operation_type] = {
                            'avg_latency_ms': sum(latencies) / len(latencies) * 1000,
                            'min_latency_ms': min(latencies) * 1000,
                            'max_latency_ms': max(latencies) * 1000,
                            'p50_latency_ms': sorted_latencies[len(sorted_latencies) // 2] * 1000,
                            'p95_latency_ms': sorted_latencies[int(len(sorted_latencies) * 0.95)] * 1000,
                            'p99_latency_ms': sorted_latencies[int(len(sorted_latencies) * 0.99)] * 1000,
                            'sample_count': len(latencies)
                        }
                    else:
                        summary[operation_type] = {
                            'avg_latency_ms': 0,
                            'min_latency_ms': 0,
                            'max_latency_ms': 0,
                            'p50_latency_ms': 0,
                            'p95_latency_ms': 0,
                            'p99_latency_ms': 0,
                            'sample_count': 0
                        }
            
            return summary