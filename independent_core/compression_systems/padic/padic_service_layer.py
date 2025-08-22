"""
P-Adic Service Layer Implementation

This module provides service interfaces for external modules to interact with 
the P-Adic compression system, including service lifecycle management, validation,
and performance tracking.

Author: Saraphis Team
Date: 2024
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
from pathlib import Path
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
import traceback
from enum import Enum

# Import P-Adic compression system
from padic_compressor import PadicCompressionSystem
from padic_encoder import PadicWeight, PadicMathematicalOperations
from padic_advanced import (
    HenselLiftingProcessor, HierarchicalClusteringManager,
    PadicDecompressionEngine, PadicOptimizationManager
)
from padic_integration import (
    PadicGACIntegration, PadicBrainIntegration,
    PadicTrainingIntegration, PadicSystemOrchestrator
)

# Import service interfaces
from service_interfaces.service_interfaces_core import (
    CompressionServiceInterface, ServiceRequest, ServiceResponse,
    ServiceValidation, ServiceMetrics, ServiceRegistry, ServiceStatus
)

# Import GPU memory management
from gpu_memory.gpu_memory_core import GPUMemoryManager, StreamManager, MemoryOptimizer

# Configure logging
logger = logging.getLogger(__name__)


class PadicServiceMethod(Enum):
    """Enumeration of P-Adic service methods"""
    COMPRESS = "compress"
    DECOMPRESS = "decompress"
    OPTIMIZE = "optimize"
    VALIDATE = "validate"
    ANALYZE = "analyze"
    BATCH_COMPRESS = "batch_compress"
    BATCH_DECOMPRESS = "batch_decompress"
    GET_METRICS = "get_metrics"
    HEALTH_CHECK = "health_check"
    CONFIGURE = "configure"


@dataclass
class PadicServiceConfig:
    """Configuration for P-Adic service layer"""
    max_batch_size: int = 100
    timeout_seconds: float = 300.0
    max_concurrent_requests: int = 10
    enable_caching: bool = True
    cache_size_mb: int = 1024
    enable_gpu_acceleration: bool = True
    gpu_memory_fraction: float = 0.8
    compression_threshold: float = 0.1
    reconstruction_error_threshold: float = 1e-6
    enable_metrics_collection: bool = True
    metrics_buffer_size: int = 10000
    health_check_interval: float = 60.0
    auto_recovery_enabled: bool = True
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 1.0


class PadicServiceInterface(CompressionServiceInterface):
    """
    Service interface for P-Adic compression system.
    
    Extends CompressionServiceInterface to provide P-Adic specific
    service methods for external module integration.
    """
    
    def __init__(self, 
                 padic_system: PadicCompressionSystem,
                 config: Optional[PadicServiceConfig] = None):
        """
        Initialize P-Adic service interface.
        
        Args:
            padic_system: P-Adic compression system instance
            config: Service configuration
        """
        super().__init__(config.__dict__ if config else {})
        
        self.padic_system = padic_system
        self.service_config = config or PadicServiceConfig()
        
        # Initialize thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(
            max_workers=self.service_config.max_concurrent_requests
        )
        
        # Initialize cache
        self.cache_lock = threading.RLock()
        self.request_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
        # Initialize GPU manager if enabled
        self.gpu_manager = None
        if self.service_config.enable_gpu_acceleration:
            try:
                self.gpu_manager = GPUMemoryManager({
                    'memory_fraction': self.service_config.gpu_memory_fraction
                })
            except Exception as e:
                logger.warning(f"Failed to initialize GPU manager: {e}")
        
        # Register P-Adic specific methods
        self._register_service_methods()
        
        logger.info("P-Adic service interface initialized")
    
    def _register_service_methods(self) -> None:
        """Register P-Adic specific service methods."""
        service_methods = {
            PadicServiceMethod.COMPRESS.value: self._handle_compress,
            PadicServiceMethod.DECOMPRESS.value: self._handle_decompress,
            PadicServiceMethod.OPTIMIZE.value: self._handle_optimize,
            PadicServiceMethod.VALIDATE.value: self._handle_validate,
            PadicServiceMethod.ANALYZE.value: self._handle_analyze,
            PadicServiceMethod.BATCH_COMPRESS.value: self._handle_batch_compress,
            PadicServiceMethod.BATCH_DECOMPRESS.value: self._handle_batch_decompress,
            PadicServiceMethod.GET_METRICS.value: self._handle_get_metrics,
            PadicServiceMethod.HEALTH_CHECK.value: self._handle_health_check,
            PadicServiceMethod.CONFIGURE.value: self._handle_configure
        }
        
        for method_name, handler in service_methods.items():
            self.service_registry.register_handler(
                service_name="padic_compression",
                method_name=method_name,
                handler=handler,
                version="1.0.0"
            )
    
    def _handle_compress(self, request: ServiceRequest) -> ServiceResponse:
        """
        Handle compression request.
        
        Args:
            request: Service request containing compression parameters
            
        Returns:
            Service response with compressed data
        """
        try:
            # Extract payload
            payload = request.payload
            data = payload.get('data')
            compression_params = payload.get('params', {})
            
            if data is None:
                raise ValueError("Data is required for compression")
            
            # Convert data to tensor if needed
            if isinstance(data, list):
                data = torch.tensor(data, dtype=torch.float32)
            elif isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            elif not isinstance(data, torch.Tensor):
                raise TypeError(f"Unsupported data type: {type(data)}")
            
            # Check cache
            cache_key = self._generate_cache_key(data, compression_params)
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                return ServiceResponse(
                    request_id=request.request_id,
                    status=ServiceStatus.SUCCESS,
                    data=cached_result,
                    metadata={'cache_hit': True}
                )
            
            # Perform compression
            start_time = time.time()
            compressed_data = self.padic_system.compress(data, **compression_params)
            compression_time = time.time() - start_time
            
            # Prepare response data
            response_data = {
                'compressed': compressed_data,
                'compression_time': compression_time,
                'original_shape': list(data.shape),
                'original_dtype': str(data.dtype),
                'compression_ratio': getattr(compressed_data, 'compression_ratio', 0.0),
                'reconstruction_error': getattr(compressed_data, 'reconstruction_error', 0.0)
            }
            
            # Cache result
            self._cache_result(cache_key, response_data)
            
            return ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.SUCCESS,
                data=response_data,
                processing_time=compression_time
            )
            
        except Exception as e:
            logger.error(f"Compression failed: {e}", exc_info=True)
            return ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.ERROR,
                errors=[{
                    'code': 'COMPRESSION_ERROR',
                    'message': str(e),
                    'traceback': traceback.format_exc()
                }]
            )
    
    def _handle_decompress(self, request: ServiceRequest) -> ServiceResponse:
        """
        Handle decompression request.
        
        Args:
            request: Service request containing compressed data
            
        Returns:
            Service response with decompressed data
        """
        try:
            # Extract payload
            payload = request.payload
            compressed_data = payload.get('compressed_data')
            
            if compressed_data is None:
                raise ValueError("Compressed data is required for decompression")
            
            # Perform decompression
            start_time = time.time()
            decompressed_data = self.padic_system.decompress(compressed_data)
            decompression_time = time.time() - start_time
            
            # Convert tensor to list for JSON serialization
            if isinstance(decompressed_data, torch.Tensor):
                decompressed_list = decompressed_data.tolist()
            else:
                decompressed_list = decompressed_data
            
            # Prepare response
            response_data = {
                'decompressed': decompressed_list,
                'decompression_time': decompression_time,
                'shape': list(decompressed_data.shape) if hasattr(decompressed_data, 'shape') else None,
                'dtype': str(decompressed_data.dtype) if hasattr(decompressed_data, 'dtype') else None
            }
            
            return ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.SUCCESS,
                data=response_data,
                processing_time=decompression_time
            )
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}", exc_info=True)
            return ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.ERROR,
                errors=[{
                    'code': 'DECOMPRESSION_ERROR',
                    'message': str(e),
                    'traceback': traceback.format_exc()
                }]
            )
    
    def _handle_optimize(self, request: ServiceRequest) -> ServiceResponse:
        """
        Handle optimization request for P-Adic compression.
        
        Args:
            request: Service request containing optimization parameters
            
        Returns:
            Service response with optimization results
        """
        try:
            # Extract payload
            payload = request.payload
            data = payload.get('data')
            target_ratio = payload.get('target_ratio', 0.1)
            max_error = payload.get('max_error', 1e-6)
            optimization_method = payload.get('method', 'gradient_descent')
            
            if data is None:
                raise ValueError("Data is required for optimization")
            
            # Convert data to tensor
            if isinstance(data, list):
                data = torch.tensor(data, dtype=torch.float32)
            elif isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            
            # Initialize optimizer
            optimizer = PadicOptimizationManager({
                'target_ratio': target_ratio,
                'max_error': max_error,
                'method': optimization_method
            })
            
            # Perform optimization
            start_time = time.time()
            optimal_params = optimizer.optimize_compression_parameters(
                data, 
                self.padic_system
            )
            optimization_time = time.time() - start_time
            
            # Apply optimal parameters and compress
            self.padic_system.update_parameters(optimal_params)
            compressed_data = self.padic_system.compress(data)
            
            # Prepare response
            response_data = {
                'optimal_parameters': optimal_params,
                'compression_ratio': getattr(compressed_data, 'compression_ratio', 0.0),
                'reconstruction_error': getattr(compressed_data, 'reconstruction_error', 0.0),
                'optimization_time': optimization_time,
                'iterations': getattr(optimizer, 'iteration_count', 0)
            }
            
            return ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.SUCCESS,
                data=response_data,
                processing_time=optimization_time
            )
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}", exc_info=True)
            return ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.ERROR,
                errors=[{
                    'code': 'OPTIMIZATION_ERROR',
                    'message': str(e),
                    'traceback': traceback.format_exc()
                }]
            )
    
    def _handle_validate(self, request: ServiceRequest) -> ServiceResponse:
        """
        Handle validation request for P-Adic compression.
        
        Args:
            request: Service request containing validation parameters
            
        Returns:
            Service response with validation results
        """
        try:
            # Extract payload
            payload = request.payload
            compressed_data = payload.get('compressed_data')
            original_data = payload.get('original_data')
            validation_criteria = payload.get('criteria', {})
            
            if compressed_data is None:
                raise ValueError("Compressed data is required for validation")
            
            # Perform validation
            validation_results = {
                'is_valid': True,
                'checks': {},
                'warnings': [],
                'errors': []
            }
            
            # Check compression format
            if not isinstance(compressed_data, dict):
                validation_results['is_valid'] = False
                validation_results['errors'].append("Invalid compression format")
            else:
                # Validate required fields
                required_fields = ['padic_weights', 'metadata']
                for field in required_fields:
                    if field in compressed_data:
                        validation_results['checks'][f'{field}_present'] = True
                    else:
                        validation_results['checks'][f'{field}_present'] = False
                        validation_results['is_valid'] = False
                        validation_results['errors'].append(f"Missing required field: {field}")
                
                # Validate P-Adic weights structure
                if 'padic_weights' in compressed_data:
                    padic_weights = compressed_data['padic_weights']
                    if isinstance(padic_weights, dict):
                        validation_results['checks']['weights_structure'] = True
                        
                        # Check individual weights
                        for key, weight in padic_weights.items():
                            if not hasattr(weight, 'valuation') or not hasattr(weight, 'digits'):
                                validation_results['warnings'].append(
                                    f"Weight {key} missing required attributes"
                                )
                    else:
                        validation_results['checks']['weights_structure'] = False
                        validation_results['is_valid'] = False
                        validation_results['errors'].append("Invalid P-Adic weights structure")
            
            # Validate against original data if provided
            if original_data is not None:
                try:
                    # Convert original data to tensor
                    if isinstance(original_data, list):
                        original_tensor = torch.tensor(original_data, dtype=torch.float32)
                    elif isinstance(original_data, np.ndarray):
                        original_tensor = torch.from_numpy(original_data).float()
                    else:
                        original_tensor = original_data
                    
                    # Decompress and compare
                    decompressed = self.padic_system.decompress(compressed_data)
                    
                    # Calculate reconstruction error
                    error = torch.mean((original_tensor - decompressed) ** 2).item()
                    max_error = validation_criteria.get('max_error', 1e-6)
                    
                    validation_results['checks']['reconstruction_error'] = error
                    validation_results['checks']['error_within_threshold'] = error <= max_error
                    
                    if error > max_error:
                        validation_results['warnings'].append(
                            f"Reconstruction error {error} exceeds threshold {max_error}"
                        )
                        
                except Exception as e:
                    validation_results['warnings'].append(
                        f"Could not validate against original data: {e}"
                    )
            
            return ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.SUCCESS,
                data=validation_results
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {e}", exc_info=True)
            return ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.ERROR,
                errors=[{
                    'code': 'VALIDATION_ERROR',
                    'message': str(e),
                    'traceback': traceback.format_exc()
                }]
            )
    
    def _handle_analyze(self, request: ServiceRequest) -> ServiceResponse:
        """
        Handle analysis request for P-Adic compression.
        
        Args:
            request: Service request containing analysis parameters
            
        Returns:
            Service response with analysis results
        """
        try:
            # Extract payload
            payload = request.payload
            data = payload.get('data')
            analysis_type = payload.get('type', 'comprehensive')
            
            if data is None:
                raise ValueError("Data is required for analysis")
            
            # Convert data to tensor
            if isinstance(data, list):
                data = torch.tensor(data, dtype=torch.float32)
            elif isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            
            # Perform analysis
            analysis_results = {
                'data_characteristics': self._analyze_data_characteristics(data),
                'compression_potential': self._analyze_compression_potential(data),
                'optimal_parameters': self._analyze_optimal_parameters(data)
            }
            
            if analysis_type == 'comprehensive':
                # Perform test compression
                test_results = []
                original_p = getattr(self.padic_system, 'p', 5)
                
                for p in [2, 3, 5, 7, 11]:
                    try:
                        self.padic_system.p = p
                        compressed = self.padic_system.compress(data)
                        test_results.append({
                            'p': p,
                            'compression_ratio': getattr(compressed, 'compression_ratio', 0.0),
                            'reconstruction_error': getattr(compressed, 'reconstruction_error', 0.0)
                        })
                    except Exception as e:
                        test_results.append({
                            'p': p,
                            'error': str(e)
                        })
                
                # Restore original p
                self.padic_system.p = original_p
                
                analysis_results['test_compressions'] = test_results
                
                # Find best parameters
                valid_results = [r for r in test_results if 'error' not in r]
                if valid_results:
                    best_result = min(valid_results, 
                                    key=lambda x: x['reconstruction_error'] + x['compression_ratio'])
                    analysis_results['recommended_p'] = best_result['p']
            
            return ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.SUCCESS,
                data=analysis_results
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.ERROR,
                errors=[{
                    'code': 'ANALYSIS_ERROR',
                    'message': str(e),
                    'traceback': traceback.format_exc()
                }]
            )
    
    def _handle_batch_compress(self, request: ServiceRequest) -> ServiceResponse:
        """
        Handle batch compression request.
        
        Args:
            request: Service request containing batch data
            
        Returns:
            Service response with batch compression results
        """
        try:
            # Extract payload
            payload = request.payload
            batch_data = payload.get('batch_data', [])
            compression_params = payload.get('params', {})
            parallel_processing = payload.get('parallel', True)
            
            if not batch_data:
                raise ValueError("Batch data is required")
            
            if len(batch_data) > self.service_config.max_batch_size:
                raise ValueError(
                    f"Batch size {len(batch_data)} exceeds maximum {self.service_config.max_batch_size}"
                )
            
            # Process batch
            results = []
            errors = []
            
            if parallel_processing and self.executor:
                # Parallel processing
                futures = []
                for i, data_item in enumerate(batch_data):
                    future = self.executor.submit(
                        self._compress_single_item,
                        data_item,
                        compression_params,
                        i
                    )
                    futures.append(future)
                
                # Collect results
                for future in futures:
                    try:
                        result = future.result(timeout=self.service_config.timeout_seconds)
                        results.append(result)
                    except Exception as e:
                        errors.append({
                            'index': len(results),
                            'error': str(e)
                        })
                        results.append(None)
            else:
                # Sequential processing
                for i, data_item in enumerate(batch_data):
                    try:
                        result = self._compress_single_item(
                            data_item,
                            compression_params,
                            i
                        )
                        results.append(result)
                    except Exception as e:
                        errors.append({
                            'index': i,
                            'error': str(e)
                        })
                        results.append(None)
            
            # Prepare response
            response_data = {
                'results': results,
                'total_processed': len(batch_data),
                'successful': len([r for r in results if r is not None]),
                'failed': len(errors),
                'errors': errors
            }
            
            status = ServiceStatus.SUCCESS if not errors else ServiceStatus.PARTIAL
            
            return ServiceResponse(
                request_id=request.request_id,
                status=status,
                data=response_data
            )
            
        except Exception as e:
            logger.error(f"Batch compression failed: {e}", exc_info=True)
            return ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.ERROR,
                errors=[{
                    'code': 'BATCH_COMPRESSION_ERROR',
                    'message': str(e),
                    'traceback': traceback.format_exc()
                }]
            )
    
    def _handle_batch_decompress(self, request: ServiceRequest) -> ServiceResponse:
        """
        Handle batch decompression request.
        
        Args:
            request: Service request containing batch compressed data
            
        Returns:
            Service response with batch decompression results
        """
        try:
            # Extract payload
            payload = request.payload
            batch_compressed = payload.get('batch_compressed', [])
            parallel_processing = payload.get('parallel', True)
            
            if not batch_compressed:
                raise ValueError("Batch compressed data is required")
            
            # Process batch
            results = []
            errors = []
            
            if parallel_processing and self.executor:
                # Parallel processing
                futures = []
                for i, compressed_item in enumerate(batch_compressed):
                    future = self.executor.submit(
                        self._decompress_single_item,
                        compressed_item,
                        i
                    )
                    futures.append(future)
                
                # Collect results
                for future in futures:
                    try:
                        result = future.result(timeout=self.service_config.timeout_seconds)
                        results.append(result)
                    except Exception as e:
                        errors.append({
                            'index': len(results),
                            'error': str(e)
                        })
                        results.append(None)
            else:
                # Sequential processing
                for i, compressed_item in enumerate(batch_compressed):
                    try:
                        result = self._decompress_single_item(compressed_item, i)
                        results.append(result)
                    except Exception as e:
                        errors.append({
                            'index': i,
                            'error': str(e)
                        })
                        results.append(None)
            
            # Prepare response
            response_data = {
                'results': results,
                'total_processed': len(batch_compressed),
                'successful': len([r for r in results if r is not None]),
                'failed': len(errors),
                'errors': errors
            }
            
            status = ServiceStatus.SUCCESS if not errors else ServiceStatus.PARTIAL
            
            return ServiceResponse(
                request_id=request.request_id,
                status=status,
                data=response_data
            )
            
        except Exception as e:
            logger.error(f"Batch decompression failed: {e}", exc_info=True)
            return ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.ERROR,
                errors=[{
                    'code': 'BATCH_DECOMPRESSION_ERROR',
                    'message': str(e),
                    'traceback': traceback.format_exc()
                }]
            )
    
    def _handle_get_metrics(self, request: ServiceRequest) -> ServiceResponse:
        """
        Handle get metrics request.
        
        Args:
            request: Service request for metrics
            
        Returns:
            Service response with metrics data
        """
        try:
            # Extract parameters
            payload = request.payload
            metric_type = payload.get('type', 'all')
            time_range = payload.get('time_range', 'last_hour')
            
            # Collect metrics
            metrics_data = {
                'compression_stats': getattr(self.padic_system, 'stats', {}),
                'cache_stats': self.cache_stats,
                'service_metrics': getattr(self, 'metrics', {})
            }
            
            if metric_type != 'all':
                # Filter specific metric type
                if metric_type in metrics_data:
                    metrics_data = {metric_type: metrics_data[metric_type]}
                else:
                    raise ValueError(f"Unknown metric type: {metric_type}")
            
            return ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.SUCCESS,
                data=metrics_data
            )
            
        except Exception as e:
            logger.error(f"Get metrics failed: {e}", exc_info=True)
            return ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.ERROR,
                errors=[{
                    'code': 'METRICS_ERROR',
                    'message': str(e),
                    'traceback': traceback.format_exc()
                }]
            )
    
    def _handle_health_check(self, request: ServiceRequest) -> ServiceResponse:
        """
        Handle health check request.
        
        Args:
            request: Service request for health check
            
        Returns:
            Service response with health status
        """
        try:
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'components': {}
            }
            
            # Check P-Adic system
            try:
                test_data = torch.randn(10, 10)
                compressed = self.padic_system.compress(test_data)
                decompressed = self.padic_system.decompress(compressed)
                
                health_status['components']['padic_system'] = {
                    'status': 'healthy',
                    'test_compression': 'passed'
                }
            except Exception as e:
                health_status['components']['padic_system'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_status['status'] = 'degraded'
            
            # Check GPU if enabled
            if self.gpu_manager:
                try:
                    gpu_status = self.gpu_manager.get_status()
                    health_status['components']['gpu'] = {
                        'status': 'healthy' if gpu_status['available'] else 'unavailable',
                        'memory_used': gpu_status.get('memory_used', 0),
                        'memory_total': gpu_status.get('memory_total', 0)
                    }
                except Exception as e:
                    health_status['components']['gpu'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Check thread pool
            health_status['components']['thread_pool'] = {
                'status': 'healthy',
                'active_threads': len(self.executor._threads) if self.executor and hasattr(self.executor, '_threads') else 0,
                'max_threads': self.service_config.max_concurrent_requests
            }
            
            # Check cache
            health_status['components']['cache'] = {
                'status': 'healthy',
                'size': len(self.request_cache),
                'hit_rate': (self.cache_stats['hits'] / 
                           max(1, self.cache_stats['hits'] + self.cache_stats['misses']))
            }
            
            return ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.SUCCESS,
                data=health_status
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.ERROR,
                errors=[{
                    'code': 'HEALTH_CHECK_ERROR',
                    'message': str(e),
                    'traceback': traceback.format_exc()
                }]
            )
    
    def _handle_configure(self, request: ServiceRequest) -> ServiceResponse:
        """
        Handle configuration update request.
        
        Args:
            request: Service request with configuration updates
            
        Returns:
            Service response with configuration status
        """
        try:
            # Extract configuration updates
            payload = request.payload
            config_updates = payload.get('config', {})
            
            if not config_updates:
                raise ValueError("Configuration updates required")
            
            # Apply updates
            applied_updates = {}
            rejected_updates = {}
            
            for key, value in config_updates.items():
                if hasattr(self.service_config, key):
                    try:
                        # Validate configuration value
                        if key == 'max_batch_size' and value <= 0:
                            rejected_updates[key] = "Value must be positive"
                        elif key == 'timeout_seconds' and value <= 0:
                            rejected_updates[key] = "Value must be positive"
                        elif key == 'gpu_memory_fraction' and not (0 < value <= 1):
                            rejected_updates[key] = "Value must be between 0 and 1"
                        else:
                            setattr(self.service_config, key, value)
                            applied_updates[key] = value
                            
                            # Apply specific configurations
                            if key == 'max_concurrent_requests':
                                # Recreate thread pool with new size
                                old_executor = self.executor
                                self.executor = ThreadPoolExecutor(max_workers=value)
                                old_executor.shutdown(wait=False)
                                
                    except Exception as e:
                        rejected_updates[key] = str(e)
                else:
                    rejected_updates[key] = "Unknown configuration parameter"
            
            # Update P-Adic system configuration if provided
            if 'padic_config' in config_updates:
                padic_config = config_updates['padic_config']
                for key, value in padic_config.items():
                    if hasattr(self.padic_system, key):
                        setattr(self.padic_system, key, value)
                        applied_updates[f'padic.{key}'] = value
            
            response_data = {
                'applied': applied_updates,
                'rejected': rejected_updates,
                'current_config': {
                    key: getattr(self.service_config, key)
                    for key in dir(self.service_config)
                    if not key.startswith('_')
                }
            }
            
            return ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.SUCCESS,
                data=response_data
            )
            
        except Exception as e:
            logger.error(f"Configuration update failed: {e}", exc_info=True)
            return ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.ERROR,
                errors=[{
                    'code': 'CONFIGURATION_ERROR',
                    'message': str(e),
                    'traceback': traceback.format_exc()
                }]
            )
    
    def _compress_single_item(self, 
                            data: Any, 
                            params: Dict[str, Any], 
                            index: int) -> Optional[Dict[str, Any]]:
        """Compress a single data item."""
        try:
            # Convert to tensor
            if isinstance(data, list):
                tensor_data = torch.tensor(data, dtype=torch.float32)
            elif isinstance(data, np.ndarray):
                tensor_data = torch.from_numpy(data).float()
            else:
                tensor_data = data
            
            # Compress
            compressed = self.padic_system.compress(tensor_data, **params)
            
            return {
                'index': index,
                'compressed': compressed,
                'compression_ratio': getattr(compressed, 'compression_ratio', 0.0),
                'reconstruction_error': getattr(compressed, 'reconstruction_error', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Failed to compress item {index}: {e}")
            raise
    
    def _decompress_single_item(self, 
                              compressed: Dict[str, Any], 
                              index: int) -> Optional[Dict[str, Any]]:
        """Decompress a single compressed item."""
        try:
            # Decompress
            decompressed = self.padic_system.decompress(compressed)
            
            # Convert to list
            if isinstance(decompressed, torch.Tensor):
                decompressed_list = decompressed.tolist()
            else:
                decompressed_list = decompressed
            
            return {
                'index': index,
                'decompressed': decompressed_list,
                'shape': list(decompressed.shape) if hasattr(decompressed, 'shape') else None
            }
            
        except Exception as e:
            logger.error(f"Failed to decompress item {index}: {e}")
            raise
    
    def _analyze_data_characteristics(self, data: torch.Tensor) -> Dict[str, Any]:
        """Analyze characteristics of input data."""
        return {
            'shape': list(data.shape),
            'dtype': str(data.dtype),
            'size_bytes': data.element_size() * data.nelement(),
            'statistics': {
                'mean': float(data.mean()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'sparsity': float((data == 0).sum() / data.nelement())
            },
            'distribution': {
                'skewness': float(self._compute_skewness(data)),
                'kurtosis': float(self._compute_kurtosis(data))
            }
        }
    
    def _analyze_compression_potential(self, data: torch.Tensor) -> Dict[str, Any]:
        """Analyze compression potential of data."""
        # Compute entropy
        data_flat = data.flatten()
        hist, _ = np.histogram(data_flat.cpu().numpy(), bins=100)
        hist = hist + 1e-10  # Avoid log(0)
        probs = hist / hist.sum()
        entropy = -np.sum(probs * np.log2(probs))
        
        # Compute redundancy measures
        fft_data = torch.fft.fft(data_flat)
        spectral_energy = torch.abs(fft_data) ** 2
        energy_concentration = torch.sum(spectral_energy[:len(spectral_energy)//10]) / torch.sum(spectral_energy)
        
        return {
            'entropy': float(entropy),
            'normalized_entropy': float(entropy / np.log2(len(hist))),
            'energy_concentration': float(energy_concentration),
            'compressibility_score': float(1.0 - entropy / np.log2(len(hist)) + energy_concentration) / 2
        }
    
    def _analyze_optimal_parameters(self, data: torch.Tensor) -> Dict[str, Any]:
        """Analyze optimal P-Adic parameters for data."""
        # Test different p values
        p_values = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        results = []
        
        for p in p_values:
            # Estimate compression efficiency
            # Higher p gives more digits but potentially better representation
            digit_efficiency = np.log(p) / p
            
            # Estimate based on data characteristics
            data_range = float(data.max() - data.min())
            precision_needed = int(np.log(data_range) / np.log(p)) + 1
            
            results.append({
                'p': p,
                'digit_efficiency': digit_efficiency,
                'precision_needed': precision_needed,
                'estimated_compression': 1.0 / precision_needed
            })
        
        # Find optimal p
        optimal = max(results, key=lambda x: x['digit_efficiency'] * x['estimated_compression'])
        
        return {
            'recommended_p': optimal['p'],
            'recommended_precision': optimal['precision_needed'],
            'analysis_results': results
        }
    
    def _compute_skewness(self, data: torch.Tensor) -> float:
        """Compute skewness of data distribution."""
        mean = data.mean()
        std = data.std()
        if std == 0:
            return 0.0
        return ((data - mean) ** 3).mean() / (std ** 3)
    
    def _compute_kurtosis(self, data: torch.Tensor) -> float:
        """Compute kurtosis of data distribution."""
        mean = data.mean()
        std = data.std()
        if std == 0:
            return 0.0
        return ((data - mean) ** 4).mean() / (std ** 4) - 3
    
    def _generate_cache_key(self, data: torch.Tensor, params: Dict[str, Any]) -> str:
        """Generate cache key for request."""
        # Create hash from data shape and parameters
        data_hash = hash((data.shape, data.dtype, float(data.sum())))
        params_hash = hash(frozenset(params.items()))
        return f"padic_{data_hash}_{params_hash}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available."""
        if not self.service_config.enable_caching:
            return None
        
        with self.cache_lock:
            if cache_key in self.request_cache:
                self.cache_stats['hits'] += 1
                return self.request_cache[cache_key].copy()
            else:
                self.cache_stats['misses'] += 1
                return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache compression result."""
        if not self.service_config.enable_caching:
            return
        
        with self.cache_lock:
            # Check cache size limit
            cache_size_bytes = sum(
                self._estimate_size(v) for v in self.request_cache.values()
            )
            
            if cache_size_bytes > self.service_config.cache_size_mb * 1024 * 1024:
                # Evict oldest entries
                evict_count = len(self.request_cache) // 4
                for _ in range(evict_count):
                    if self.request_cache:
                        self.request_cache.pop(next(iter(self.request_cache)))
                        self.cache_stats['evictions'] += 1
            
            self.request_cache[cache_key] = result.copy()
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object."""
        if isinstance(obj, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v) 
                      for k, v in obj.items())
        elif isinstance(obj, (list, tuple)):
            return sum(self._estimate_size(item) for item in obj)
        elif isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement()
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        else:
            return 64  # Default estimate
    
    def shutdown(self) -> None:
        """Shutdown service interface."""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            
            if self.gpu_manager:
                self.gpu_manager.cleanup()
            
            logger.info("P-Adic service interface shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


class PadicServiceManager:
    """
    Manages P-Adic service lifecycle including registration, 
    discovery, health monitoring, and performance tracking.
    """
    
    def __init__(self, 
                 service_registry: ServiceRegistry,
                 config: Optional[PadicServiceConfig] = None):
        """
        Initialize P-Adic service manager.
        
        Args:
            service_registry: Service registry instance
            config: Service configuration
        """
        self.service_registry = service_registry
        self.config = config or PadicServiceConfig()
        
        # Service instances
        self.services = {}
        self.service_lock = threading.RLock()
        
        # Health monitoring
        self.health_monitor = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True
        )
        self.monitoring_active = False
        
        # Performance tracking
        self.performance_stats = defaultdict(lambda: {
            'request_count': 0,
            'total_time': 0.0,
            'error_count': 0,
            'last_request': None
        })
        
        # Recovery mechanisms
        self.recovery_attempts = defaultdict(int)
        self.blacklisted_services = set()
        
        logger.info("P-Adic service manager initialized")
    
    def register_service(self, 
                        service_id: str, 
                        service_interface: PadicServiceInterface,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a P-Adic service instance.
        
        Args:
            service_id: Unique service identifier
            service_interface: P-Adic service interface instance
            metadata: Additional service metadata
            
        Returns:
            True if registration successful
        """
        try:
            with self.service_lock:
                if service_id in self.services:
                    logger.warning(f"Service {service_id} already registered")
                    return False
                
                # Register with service registry
                registration_info = {
                    'service_id': service_id,
                    'service_type': 'padic_compression',
                    'version': '1.0.0',
                    'endpoint': f'padic://{service_id}',
                    'metadata': metadata or {},
                    'registered_at': datetime.now().isoformat()
                }
                
                self.service_registry.register_service(
                    service_name='padic_compression',
                    service_info=registration_info
                )
                
                # Store service instance
                self.services[service_id] = {
                    'interface': service_interface,
                    'info': registration_info,
                    'status': 'active',
                    'last_health_check': datetime.now()
                }
                
                logger.info(f"Registered P-Adic service: {service_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register service {service_id}: {e}")
            return False
    
    def discover_services(self, 
                         criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Discover available P-Adic services.
        
        Args:
            criteria: Discovery criteria
            
        Returns:
            List of discovered services
        """
        try:
            with self.service_lock:
                discovered = []
                
                for service_id, service_data in self.services.items():
                    if service_id in self.blacklisted_services:
                        continue
                    
                    # Apply criteria filters
                    if criteria:
                        match = True
                        for key, value in criteria.items():
                            if key in service_data['info']:
                                if service_data['info'][key] != value:
                                    match = False
                                    break
                            elif key in service_data['info'].get('metadata', {}):
                                if service_data['info']['metadata'][key] != value:
                                    match = False
                                    break
                        
                        if not match:
                            continue
                    
                    # Include service in results
                    discovered.append({
                        'service_id': service_id,
                        'info': service_data['info'],
                        'status': service_data['status'],
                        'performance': self.performance_stats[service_id]
                    })
                
                return discovered
                
        except Exception as e:
            logger.error(f"Service discovery failed: {e}")
            return []
    
    def get_service(self, service_id: str) -> Optional[PadicServiceInterface]:
        """
        Get service interface by ID.
        
        Args:
            service_id: Service identifier
            
        Returns:
            Service interface or None
        """
        with self.service_lock:
            service_data = self.services.get(service_id)
            if service_data and service_data['status'] == 'active':
                return service_data['interface']
            return None
    
    def invoke_service(self, 
                      service_id: str, 
                      request: ServiceRequest) -> ServiceResponse:
        """
        Invoke a service with request.
        
        Args:
            service_id: Service identifier
            request: Service request
            
        Returns:
            Service response
        """
        start_time = time.time()
        
        try:
            # Get service interface
            service_interface = self.get_service(service_id)
            if not service_interface:
                return ServiceResponse(
                    request_id=request.request_id,
                    status=ServiceStatus.ERROR,
                    errors=[{
                        'code': 'SERVICE_NOT_FOUND',
                        'message': f"Service {service_id} not found or inactive"
                    }]
                )
            
            # Invoke service
            response = service_interface.invoke_service(request)
            
            # Update performance stats
            elapsed_time = time.time() - start_time
            with self.service_lock:
                stats = self.performance_stats[service_id]
                stats['request_count'] += 1
                stats['total_time'] += elapsed_time
                stats['last_request'] = datetime.now()
                
                if response.status == ServiceStatus.ERROR:
                    stats['error_count'] += 1
            
            return response
            
        except Exception as e:
            logger.error(f"Service invocation failed for {service_id}: {e}")
            
            # Update error stats
            with self.service_lock:
                self.performance_stats[service_id]['error_count'] += 1
            
            return ServiceResponse(
                request_id=request.request_id,
                status=ServiceStatus.ERROR,
                errors=[{
                    'code': 'INVOCATION_ERROR',
                    'message': str(e),
                    'traceback': traceback.format_exc()
                }]
            )
    
    def start_monitoring(self) -> None:
        """Start health monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.health_monitor.start()
            logger.info("Started P-Adic service health monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.monitoring_active = False
        logger.info("Stopped P-Adic service health monitoring")
    
    def _health_monitor_loop(self) -> None:
        """Health monitoring loop."""
        while self.monitoring_active:
            try:
                with self.service_lock:
                    for service_id, service_data in list(self.services.items()):
                        if service_id in self.blacklisted_services:
                            continue
                        
                        # Check service health
                        if self._check_service_health(service_id, service_data):
                            service_data['status'] = 'active'
                            service_data['last_health_check'] = datetime.now()
                            self.recovery_attempts[service_id] = 0
                        else:
                            service_data['status'] = 'unhealthy'
                            
                            # Attempt recovery
                            if self.config.auto_recovery_enabled:
                                self._attempt_recovery(service_id, service_data)
                
                # Sleep until next check
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(5)  # Brief pause on error
    
    def _check_service_health(self, 
                            service_id: str, 
                            service_data: Dict[str, Any]) -> bool:
        """Check health of a service."""
        try:
            # Create health check request
            health_request = ServiceRequest(
                service_name='padic_compression',
                method_name=PadicServiceMethod.HEALTH_CHECK.value,
                version='1.0.0',
                payload={}
            )
            
            # Invoke health check
            interface = service_data['interface']
            response = interface._handle_health_check(health_request)
            
            return response.status == ServiceStatus.SUCCESS
            
        except Exception as e:
            logger.error(f"Health check failed for {service_id}: {e}")
            return False
    
    def _attempt_recovery(self, 
                         service_id: str, 
                         service_data: Dict[str, Any]) -> None:
        """Attempt to recover unhealthy service."""
        try:
            self.recovery_attempts[service_id] += 1
            
            if self.recovery_attempts[service_id] > self.config.max_retry_attempts:
                # Blacklist service after max attempts
                self.blacklisted_services.add(service_id)
                logger.error(f"Service {service_id} blacklisted after {self.config.max_retry_attempts} recovery attempts")
                return
            
            logger.info(f"Attempting recovery for service {service_id} (attempt {self.recovery_attempts[service_id]})")
            
            # Get service interface
            interface = service_data['interface']
            
            # Attempt to reinitialize
            if hasattr(interface, 'padic_system'):
                # Test with simple operation
                test_data = torch.randn(10, 10)
                test_result = interface.padic_system.compress(test_data)
                interface.padic_system.decompress(test_result)
                
                # Clear cache
                with interface.cache_lock:
                    interface.request_cache.clear()
                    interface.cache_stats = {
                        'hits': 0,
                        'misses': 0,
                        'evictions': 0
                    }
                
                # If we get here, recovery succeeded
                service_data['status'] = 'active'
                logger.info(f"Service {service_id} recovered successfully")
                
        except Exception as e:
            logger.error(f"Recovery failed for service {service_id}: {e}")
            time.sleep(self.config.retry_delay_seconds)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report for all services."""
        with self.service_lock:
            report = {
                'timestamp': datetime.now().isoformat(),
                'services': {}
            }
            
            for service_id, stats in self.performance_stats.items():
                avg_time = (stats['total_time'] / stats['request_count'] 
                           if stats['request_count'] > 0 else 0)
                error_rate = (stats['error_count'] / stats['request_count'] 
                             if stats['request_count'] > 0 else 0)
                
                report['services'][service_id] = {
                    'total_requests': stats['request_count'],
                    'total_errors': stats['error_count'],
                    'average_time': avg_time,
                    'error_rate': error_rate,
                    'last_request': stats['last_request'].isoformat() if stats['last_request'] else None,
                    'status': self.services[service_id]['status'] if service_id in self.services else 'unknown'
                }
            
            return report
    
    def shutdown(self) -> None:
        """Shutdown service manager."""
        try:
            # Stop monitoring
            self.stop_monitoring()
            
            # Shutdown all services
            with self.service_lock:
                for service_id, service_data in self.services.items():
                    try:
                        interface = service_data['interface']
                        if hasattr(interface, 'shutdown'):
                            interface.shutdown()
                    except Exception as e:
                        logger.error(f"Error shutting down service {service_id}: {e}")
                
                # Clear services
                self.services.clear()
            
            logger.info("P-Adic service manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during service manager shutdown: {e}")


class PadicServiceError(Exception):
    """Base exception for P-Adic service errors"""
    pass


class PadicServiceValidationError(PadicServiceError):
    """Exception for P-Adic service validation errors"""
    pass


class PadicServiceValidation:
    """
    Validation layer for P-Adic service requests and responses.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize P-Adic service validation.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Validation rules
        self.request_validators = self._initialize_request_validators()
        self.response_validators = self._initialize_response_validators()
        self.parameter_validators = self._initialize_parameter_validators()
        
        # Validation cache
        self.validation_cache = {}
        self.cache_size = self.config.get('cache_size', 1000)
        
        # Metrics
        self.validation_metrics = defaultdict(lambda: {
            'total_validations': 0,
            'passed': 0,
            'failed': 0,
            'cached': 0
        })
        
        logger.info("P-Adic service validation initialized")
    
    def validate_request(self, request: ServiceRequest) -> Tuple[bool, Optional[List[str]]]:
        """
        Validate a P-Adic service request.
        
        Args:
            request: Service request to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        validator_type = 'request'
        
        try:
            # Check cache
            cache_key = self._generate_validation_cache_key(request)
            if cache_key in self.validation_cache:
                self.validation_metrics[validator_type]['cached'] += 1
                return self.validation_cache[cache_key]
            
            # Validate request structure
            struct_errors = self._validate_request_structure(request)
            errors.extend(struct_errors)
            
            # Validate method
            if request.method_name in self.request_validators:
                method_errors = self.request_validators[request.method_name](request)
                errors.extend(method_errors)
            else:
                errors.append(f"Unknown method: {request.method_name}")
            
            # Validate parameters
            param_errors = self._validate_parameters(request)
            errors.extend(param_errors)
            
            # Update metrics
            is_valid = len(errors) == 0
            self.validation_metrics[validator_type]['total_validations'] += 1
            if is_valid:
                self.validation_metrics[validator_type]['passed'] += 1
            else:
                self.validation_metrics[validator_type]['failed'] += 1
            
            # Cache result
            result = (is_valid, errors if errors else None)
            self._cache_validation_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Request validation error: {str(e)}")
            return False, [f"Validation error: {str(e)}"]
    
    def validate_response(self, response: ServiceResponse) -> Tuple[bool, Optional[List[str]]]:
        """
        Validate a P-Adic service response.
        
        Args:
            response: Service response to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        validator_type = 'response'
        
        try:
            # Validate response structure
            struct_errors = self._validate_response_structure(response)
            errors.extend(struct_errors)
            
            # Validate response data based on status
            if response.status == ServiceStatus.SUCCESS:
                data_errors = self._validate_response_data(response)
                errors.extend(data_errors)
            
            # Update metrics
            is_valid = len(errors) == 0
            self.validation_metrics[validator_type]['total_validations'] += 1
            if is_valid:
                self.validation_metrics[validator_type]['passed'] += 1
            else:
                self.validation_metrics[validator_type]['failed'] += 1
            
            return is_valid, errors if errors else None
            
        except Exception as e:
            logger.error(f"Response validation error: {str(e)}")
            return False, [f"Validation error: {str(e)}"]
    
    def validate_padic_parameters(self, p: int, precision: int,
                                 compression_ratio: float) -> Tuple[bool, Optional[List[str]]]:
        """
        Validate P-Adic specific parameters.
        
        Args:
            p: Prime number for P-Adic representation
            precision: Precision bits
            compression_ratio: Target compression ratio
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Validate p (must be prime)
        if not self._is_prime(p):
            errors.append(f"p={p} is not a prime number")
        elif p < 2:
            errors.append(f"p={p} must be >= 2")
        elif p > 97:  # Largest prime < 100
            errors.append(f"p={p} is too large (max 97)")
        
        # Validate precision
        if precision < 1:
            errors.append(f"precision={precision} must be >= 1")
        elif precision > 256:
            errors.append(f"precision={precision} exceeds maximum (256)")
        
        # Validate compression ratio
        if compression_ratio <= 0:
            errors.append(f"compression_ratio={compression_ratio} must be > 0")
        elif compression_ratio >= 1:
            errors.append(f"compression_ratio={compression_ratio} must be < 1")
        
        return len(errors) == 0, errors if errors else None
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """
        Get validation metrics.
        
        Returns:
            Validation metrics dictionary
        """
        return dict(self.validation_metrics)
    
    def _initialize_request_validators(self) -> Dict[str, Callable]:
        """Initialize request validators for each method."""
        return {
            'compress': self._validate_compress_request,
            'decompress': self._validate_decompress_request,
            'batch_compress': self._validate_batch_compress_request,
            'batch_decompress': self._validate_batch_decompress_request,
            'optimize': self._validate_optimize_request,
            'analyze_compression': self._validate_analyze_request
        }
    
    def _initialize_response_validators(self) -> Dict[str, Callable]:
        """Initialize response validators."""
        return {
            'compressed_data': self._validate_compressed_data,
            'decompressed_data': self._validate_decompressed_data,
            'optimization_result': self._validate_optimization_result
        }
    
    def _initialize_parameter_validators(self) -> Dict[str, Any]:
        """Initialize parameter validation rules."""
        return {
            'p': {'type': int, 'min': 2, 'max': 97, 'validator': self._is_prime},
            'precision': {'type': int, 'min': 1, 'max': 256},
            'compression_ratio': {'type': float, 'min': 0.001, 'max': 0.999},
            'max_iterations': {'type': int, 'min': 1, 'max': 10000},
            'tolerance': {'type': float, 'min': 1e-10, 'max': 1.0}
        }
    
    def _validate_request_structure(self, request: ServiceRequest) -> List[str]:
        """Validate basic request structure."""
        errors = []
        
        if not request.service_name:
            errors.append("Service name is required")
        elif request.service_name != 'padic':
            errors.append(f"Invalid service name: {request.service_name}")
        
        if not request.method_name:
            errors.append("Method name is required")
        
        if not request.payload:
            errors.append("Payload is required")
        
        return errors
    
    def _validate_compress_request(self, request: ServiceRequest) -> List[str]:
        """Validate compression request."""
        errors = []
        payload = request.payload
        
        if 'data' not in payload:
            errors.append("Data is required for compression")
        else:
            data = payload['data']
            if not isinstance(data, (torch.Tensor, list, np.ndarray)):
                errors.append("Data must be tensor, list, or numpy array")
        
        # Validate optional parameters
        if 'p' in payload:
            p_valid, p_errors = self.validate_padic_parameters(
                payload['p'],
                payload.get('precision', 32),
                payload.get('compression_ratio', 0.1)
            )
            if not p_valid:
                errors.extend(p_errors)
        
        return errors
    
    def _validate_decompress_request(self, request: ServiceRequest) -> List[str]:
        """Validate decompression request."""
        errors = []
        payload = request.payload
        
        if 'compressed_data' not in payload:
            errors.append("Compressed data is required for decompression")
        else:
            compressed = payload['compressed_data']
            if not isinstance(compressed, dict):
                errors.append("Compressed data must be a dictionary")
            elif 'padic_weights' not in compressed:
                errors.append("Compressed data must contain padic_weights")
        
        return errors
    
    def _validate_batch_compress_request(self, request: ServiceRequest) -> List[str]:
        """Validate batch compression request."""
        errors = []
        payload = request.payload
        
        if 'batch_data' not in payload:
            errors.append("Batch data is required")
        else:
            batch = payload['batch_data']
            if not isinstance(batch, list):
                errors.append("Batch data must be a list")
            elif len(batch) == 0:
                errors.append("Batch data cannot be empty")
            elif len(batch) > self.config.get('max_batch_size', 100):
                errors.append(f"Batch size exceeds maximum ({self.config.get('max_batch_size', 100)})")
        
        return errors
    
    def _validate_batch_decompress_request(self, request: ServiceRequest) -> List[str]:
        """Validate batch decompression request."""
        errors = []
        payload = request.payload
        
        if 'compressed_batch' not in payload:
            errors.append("Compressed batch is required")
        else:
            batch = payload['compressed_batch']
            if not isinstance(batch, list):
                errors.append("Compressed batch must be a list")
            elif len(batch) == 0:
                errors.append("Compressed batch cannot be empty")
        
        return errors
    
    def _validate_optimize_request(self, request: ServiceRequest) -> List[str]:
        """Validate optimization request."""
        errors = []
        payload = request.payload
        
        if 'data' not in payload:
            errors.append("Data is required for optimization")
        
        # Validate optimization parameters
        if 'target_ratio' in payload:
            ratio = payload['target_ratio']
            if not isinstance(ratio, (int, float)):
                errors.append("Target ratio must be numeric")
            elif ratio <= 0 or ratio >= 1:
                errors.append("Target ratio must be between 0 and 1")
        
        return errors
    
    def _validate_analyze_request(self, request: ServiceRequest) -> List[str]:
        """Validate analysis request."""
        errors = []
        payload = request.payload
        
        if 'data' not in payload:
            errors.append("Data is required for analysis")
        
        return errors
    
    def _validate_response_structure(self, response: ServiceResponse) -> List[str]:
        """Validate basic response structure."""
        errors = []
        
        if not response.request_id:
            errors.append("Response must have request_id")
        
        if response.status not in ServiceStatus:
            errors.append(f"Invalid response status: {response.status}")
        
        return errors
    
    def _validate_response_data(self, response: ServiceResponse) -> List[str]:
        """Validate response data."""
        errors = []
        
        if not response.data:
            errors.append("Successful response must contain data")
            return errors
        
        # Validate based on response type
        if 'compressed_data' in response.data:
            comp_errors = self._validate_compressed_data(response.data['compressed_data'])
            errors.extend(comp_errors)
        
        if 'decompressed_data' in response.data:
            decomp_errors = self._validate_decompressed_data(response.data['decompressed_data'])
            errors.extend(decomp_errors)
        
        return errors
    
    def _validate_compressed_data(self, compressed_data: Dict[str, Any]) -> List[str]:
        """Validate compressed data structure."""
        errors = []
        
        required_fields = ['padic_weights', 'compression_ratio', 'metadata']
        for field in required_fields:
            if field not in compressed_data:
                errors.append(f"Compressed data missing required field: {field}")
        
        if 'padic_weights' in compressed_data:
            weights = compressed_data['padic_weights']
            if not isinstance(weights, dict):
                errors.append("P-Adic weights must be a dictionary")
        
        return errors
    
    def _validate_decompressed_data(self, decompressed_data: Any) -> List[str]:
        """Validate decompressed data."""
        errors = []
        
        if not isinstance(decompressed_data, (list, np.ndarray)):
            errors.append("Decompressed data must be list or array")
        
        return errors
    
    def _validate_optimization_result(self, result: Dict[str, Any]) -> List[str]:
        """Validate optimization result."""
        errors = []
        
        required_fields = ['optimal_p', 'optimal_precision', 'achieved_ratio']
        for field in required_fields:
            if field not in result:
                errors.append(f"Optimization result missing field: {field}")
        
        return errors
    
    def _validate_parameters(self, request: ServiceRequest) -> List[str]:
        """Validate request parameters against rules."""
        errors = []
        
        for param_name, rules in self.parameter_validators.items():
            if param_name in request.payload:
                value = request.payload[param_name]
                
                # Type check
                if not isinstance(value, rules['type']):
                    errors.append(f"{param_name} must be {rules['type'].__name__}")
                    continue
                
                # Range check
                if 'min' in rules and value < rules['min']:
                    errors.append(f"{param_name}={value} below minimum {rules['min']}")
                if 'max' in rules and value > rules['max']:
                    errors.append(f"{param_name}={value} above maximum {rules['max']}")
                
                # Custom validator
                if 'validator' in rules and not rules['validator'](value):
                    errors.append(f"{param_name}={value} failed validation")
        
        return errors
    
    def _is_prime(self, n: int) -> bool:
        """Check if a number is prime."""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def _generate_validation_cache_key(self, request: ServiceRequest) -> str:
        """Generate cache key for validation result."""
        key_data = {
            'method': request.method_name,
            'payload_keys': sorted(request.payload.keys()),
            'payload_types': {k: type(v).__name__ for k, v in request.payload.items()}
        }
        return json.dumps(key_data, sort_keys=True)
    
    def _cache_validation_result(self, cache_key: str, result: Tuple[bool, Optional[List[str]]]) -> None:
        """Cache validation result."""
        if len(self.validation_cache) >= self.cache_size:
            # Remove oldest entry
            self.validation_cache.pop(next(iter(self.validation_cache)))
        
        self.validation_cache[cache_key] = result


class PadicServiceMetrics:
    """
    Tracks P-Adic service performance metrics including compression
    ratios, decompression speeds, error rates, and resource usage.
    """
    
    def __init__(self, config: Optional[PadicServiceConfig] = None):
        """
        Initialize P-Adic service metrics.
        
        Args:
            config: Service configuration
        """
        self.config = config or PadicServiceConfig()
        
        # Metrics storage
        self.metrics_lock = threading.RLock()
        self.compression_metrics = deque(maxlen=self.config.metrics_buffer_size)
        self.decompression_metrics = deque(maxlen=self.config.metrics_buffer_size)
        self.error_metrics = deque(maxlen=self.config.metrics_buffer_size)
        self.resource_metrics = deque(maxlen=self.config.metrics_buffer_size)
        
        # Aggregated stats
        self.stats = {
            'total_compressions': 0,
            'total_decompressions': 0,
            'total_errors': 0,
            'avg_compression_ratio': 0.0,
            'avg_compression_time': 0.0,
            'avg_decompression_time': 0.0,
            'avg_reconstruction_error': 0.0,
            'peak_memory_usage': 0,
            'total_bytes_compressed': 0,
            'total_bytes_decompressed': 0
        }
        
        # Time-based buckets for aggregation
        self.time_buckets = {
            'minute': deque(maxlen=60),
            'hour': deque(maxlen=60),
            'day': deque(maxlen=24)
        }
        
        # Start metrics aggregation thread
        self.aggregation_thread = threading.Thread(
            target=self._aggregation_loop,
            daemon=True
        )
        self.aggregation_active = False
        
        logger.info("P-Adic service metrics initialized")
    
    def record_compression(self, 
                         data_size: int,
                         compressed_size: int,
                         compression_time: float,
                         reconstruction_error: float,
                         parameters: Dict[str, Any]) -> None:
        """
        Record compression metrics.
        
        Args:
            data_size: Original data size in bytes
            compressed_size: Compressed data size in bytes
            compression_time: Time taken for compression
            reconstruction_error: Reconstruction error after decompression
            parameters: Compression parameters used
        """
        try:
            metric = {
                'timestamp': datetime.now(),
                'data_size': data_size,
                'compressed_size': compressed_size,
                'compression_ratio': compressed_size / max(1, data_size),
                'compression_time': compression_time,
                'reconstruction_error': reconstruction_error,
                'parameters': parameters.copy()
            }
            
            with self.metrics_lock:
                self.compression_metrics.append(metric)
                
                # Update stats
                self.stats['total_compressions'] += 1
                self.stats['total_bytes_compressed'] += data_size
                
                # Update running averages
                self._update_compression_averages()
            
        except Exception as e:
            logger.error(f"Failed to record compression metrics: {e}")
    
    def record_decompression(self,
                           compressed_size: int,
                           decompressed_size: int,
                           decompression_time: float) -> None:
        """
        Record decompression metrics.
        
        Args:
            compressed_size: Compressed data size in bytes
            decompressed_size: Decompressed data size in bytes
            decompression_time: Time taken for decompression
        """
        try:
            metric = {
                'timestamp': datetime.now(),
                'compressed_size': compressed_size,
                'decompressed_size': decompressed_size,
                'decompression_time': decompression_time,
                'decompression_speed': decompressed_size / max(0.001, decompression_time)
            }
            
            with self.metrics_lock:
                self.decompression_metrics.append(metric)
                
                # Update stats
                self.stats['total_decompressions'] += 1
                self.stats['total_bytes_decompressed'] += decompressed_size
                
                # Update running averages
                self._update_decompression_averages()
            
        except Exception as e:
            logger.error(f"Failed to record decompression metrics: {e}")
    
    def record_error(self,
                    error_type: str,
                    error_message: str,
                    operation: str,
                    context: Optional[Dict[str, Any]] = None) -> None:
        """
        Record error metrics.
        
        Args:
            error_type: Type of error
            error_message: Error message
            operation: Operation that failed
            context: Additional error context
        """
        try:
            metric = {
                'timestamp': datetime.now(),
                'error_type': error_type,
                'error_message': error_message,
                'operation': operation,
                'context': context or {}
            }
            
            with self.metrics_lock:
                self.error_metrics.append(metric)
                self.stats['total_errors'] += 1
            
        except Exception as e:
            logger.error(f"Failed to record error metrics: {e}")
    
    def record_resource_usage(self,
                            cpu_percent: float,
                            memory_mb: int,
                            gpu_memory_mb: Optional[int] = None,
                            thread_count: int = 0) -> None:
        """
        Record resource usage metrics.
        
        Args:
            cpu_percent: CPU usage percentage
            memory_mb: Memory usage in MB
            gpu_memory_mb: GPU memory usage in MB (if applicable)
            thread_count: Number of active threads
        """
        try:
            metric = {
                'timestamp': datetime.now(),
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'gpu_memory_mb': gpu_memory_mb,
                'thread_count': thread_count
            }
            
            with self.metrics_lock:
                self.resource_metrics.append(metric)
                
                # Update peak memory
                total_memory = memory_mb + (gpu_memory_mb or 0)
                if total_memory > self.stats['peak_memory_usage']:
                    self.stats['peak_memory_usage'] = total_memory
            
        except Exception as e:
            logger.error(f"Failed to record resource metrics: {e}")
    
    def get_metrics(self, time_range: str = 'all') -> Dict[str, Any]:
        """
        Get metrics for specified time range.
        
        Args:
            time_range: Time range for metrics
            
        Returns:
            Metrics dictionary
        """
        try:
            with self.metrics_lock:
                # Get time filter
                if time_range == 'all':
                    time_filter = None
                elif time_range == 'last_minute':
                    time_filter = datetime.now() - timedelta(minutes=1)
                elif time_range == 'last_hour':
                    time_filter = datetime.now() - timedelta(hours=1)
                elif time_range == 'last_day':
                    time_filter = datetime.now() - timedelta(days=1)
                elif time_range == 'last_week':
                    time_filter = datetime.now() - timedelta(weeks=1)
                else:
                    time_filter = None
                
                # Filter metrics
                compression_filtered = self._filter_by_time(
                    self.compression_metrics, time_filter
                )
                decompression_filtered = self._filter_by_time(
                    self.decompression_metrics, time_filter
                )
                error_filtered = self._filter_by_time(
                    self.error_metrics, time_filter
                )
                resource_filtered = self._filter_by_time(
                    self.resource_metrics, time_filter
                )
                
                # Calculate aggregates
                metrics = {
                    'time_range': time_range,
                    'summary': self._calculate_summary(
                        compression_filtered,
                        decompression_filtered,
                        error_filtered
                    ),
                    'compression': self._aggregate_compression_metrics(compression_filtered),
                    'decompression': self._aggregate_decompression_metrics(decompression_filtered),
                    'errors': self._aggregate_error_metrics(error_filtered),
                    'resources': self._aggregate_resource_metrics(resource_filtered),
                    'overall_stats': self.stats.copy()
                }
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {}
    
    def start_aggregation(self) -> None:
        """Start metrics aggregation."""
        if not self.aggregation_active:
            self.aggregation_active = True
            self.aggregation_thread.start()
            logger.info("Started metrics aggregation")
    
    def stop_aggregation(self) -> None:
        """Stop metrics aggregation."""
        self.aggregation_active = False
        logger.info("Stopped metrics aggregation")
    
    def _aggregation_loop(self) -> None:
        """Aggregation loop for time-based metrics."""
        last_minute = datetime.now()
        last_hour = datetime.now()
        
        while self.aggregation_active:
            try:
                current_time = datetime.now()
                
                # Minute aggregation
                if (current_time - last_minute).total_seconds() >= 60:
                    self._aggregate_minute()
                    last_minute = current_time
                
                # Hour aggregation
                if (current_time - last_hour).total_seconds() >= 3600:
                    self._aggregate_hour()
                    last_hour = current_time
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Aggregation error: {e}")
                time.sleep(10)
    
    def _aggregate_minute(self) -> None:
        """Aggregate metrics for the last minute."""
        try:
            with self.metrics_lock:
                minute_ago = datetime.now() - timedelta(minutes=1)
                
                # Get metrics from last minute
                compression_minute = self._filter_by_time(
                    self.compression_metrics, minute_ago
                )
                
                if compression_minute:
                    minute_stats = {
                        'timestamp': datetime.now(),
                        'compressions': len(compression_minute),
                        'avg_ratio': sum(m['compression_ratio'] for m in compression_minute) / len(compression_minute),
                        'avg_time': sum(m['compression_time'] for m in compression_minute) / len(compression_minute)
                    }
                    
                    self.time_buckets['minute'].append(minute_stats)
                    
        except Exception as e:
            logger.error(f"Minute aggregation error: {e}")
    
    def _aggregate_hour(self) -> None:
        """Aggregate metrics for the last hour."""
        try:
            with self.metrics_lock:
                # Aggregate minute buckets into hour bucket
                if self.time_buckets['minute']:
                    total_compressions = sum(m['compressions'] for m in self.time_buckets['minute'])
                    if total_compressions > 0:
                        hour_stats = {
                            'timestamp': datetime.now(),
                            'total_compressions': total_compressions,
                            'avg_ratio': sum(m['avg_ratio'] * m['compressions'] for m in self.time_buckets['minute']) / total_compressions,
                            'avg_time': sum(m['avg_time'] * m['compressions'] for m in self.time_buckets['minute']) / total_compressions
                        }
                        
                        self.time_buckets['hour'].append(hour_stats)
                    
        except Exception as e:
            logger.error(f"Hour aggregation error: {e}")
    
    def _update_compression_averages(self) -> None:
        """Update running compression averages."""
        if self.compression_metrics:
            recent = list(self.compression_metrics)[-100:]  # Last 100 samples
            
            self.stats['avg_compression_ratio'] = sum(
                m['compression_ratio'] for m in recent
            ) / len(recent)
            
            self.stats['avg_compression_time'] = sum(
                m['compression_time'] for m in recent
            ) / len(recent)
            
            self.stats['avg_reconstruction_error'] = sum(
                m['reconstruction_error'] for m in recent
            ) / len(recent)
    
    def _update_decompression_averages(self) -> None:
        """Update running decompression averages."""
        if self.decompression_metrics:
            recent = list(self.decompression_metrics)[-100:]  # Last 100 samples
            
            self.stats['avg_decompression_time'] = sum(
                m['decompression_time'] for m in recent
            ) / len(recent)
    
    def _filter_by_time(self, 
                       metrics: deque, 
                       time_filter: Optional[datetime]) -> List[Dict[str, Any]]:
        """Filter metrics by time."""
        if time_filter is None:
            return list(metrics)
        
        return [m for m in metrics if m['timestamp'] >= time_filter]
    
    def _calculate_summary(self,
                         compression_metrics: List[Dict[str, Any]],
                         decompression_metrics: List[Dict[str, Any]],
                         error_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics."""
        total_operations = len(compression_metrics) + len(decompression_metrics)
        
        return {
            'total_operations': total_operations,
            'total_compressions': len(compression_metrics),
            'total_decompressions': len(decompression_metrics),
            'total_errors': len(error_metrics),
            'error_rate': len(error_metrics) / max(1, total_operations),
            'success_rate': 1.0 - (len(error_metrics) / max(1, total_operations))
        }
    
    def _aggregate_compression_metrics(self, 
                                     metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate compression metrics."""
        if not metrics:
            return {}
        
        return {
            'count': len(metrics),
            'avg_compression_ratio': sum(m['compression_ratio'] for m in metrics) / len(metrics),
            'avg_compression_time': sum(m['compression_time'] for m in metrics) / len(metrics),
            'avg_reconstruction_error': sum(m['reconstruction_error'] for m in metrics) / len(metrics),
            'total_bytes_compressed': sum(m['data_size'] for m in metrics),
            'total_bytes_output': sum(m['compressed_size'] for m in metrics)
        }
    
    def _aggregate_decompression_metrics(self, 
                                       metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate decompression metrics."""
        if not metrics:
            return {}
        
        return {
            'count': len(metrics),
            'avg_decompression_time': sum(m['decompression_time'] for m in metrics) / len(metrics),
            'avg_decompression_speed': sum(m['decompression_speed'] for m in metrics) / len(metrics),
            'total_bytes_decompressed': sum(m['decompressed_size'] for m in metrics)
        }
    
    def _aggregate_error_metrics(self, 
                               metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate error metrics."""
        if not metrics:
            return {}
        
        # Group by error type
        error_types = defaultdict(int)
        operation_errors = defaultdict(int)
        
        for metric in metrics:
            error_types[metric['error_type']] += 1
            operation_errors[metric['operation']] += 1
        
        return {
            'count': len(metrics),
            'by_type': dict(error_types),
            'by_operation': dict(operation_errors),
            'recent_errors': [
                {
                    'timestamp': m['timestamp'].isoformat(),
                    'type': m['error_type'],
                    'operation': m['operation'],
                    'message': m['error_message']
                }
                for m in metrics[-10:]  # Last 10 errors
            ]
        }
    
    def _aggregate_resource_metrics(self, 
                                  metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate resource metrics."""
        if not metrics:
            return {}
        
        cpu_values = [m['cpu_percent'] for m in metrics]
        memory_values = [m['memory_mb'] for m in metrics]
        gpu_values = [m['gpu_memory_mb'] for m in metrics if m['gpu_memory_mb'] is not None]
        
        result = {
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory': {
                'avg': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values)
            }
        }
        
        if gpu_values:
            result['gpu_memory'] = {
                'avg': sum(gpu_values) / len(gpu_values),
                'max': max(gpu_values),
                'min': min(gpu_values)
            }
        
        return result


# Module exports
__all__ = [
    'PadicServiceInterface',
    'PadicServiceManager',
    'PadicServiceValidator',
    'PadicServiceMetrics',
    'PadicServiceConfig',
    'PadicServiceMethod'
]