"""
Integration of P-adic compression with GAC GradientCompressionComponent.
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
import time

from ..padic.padic_gradient import PadicGradientCompressor
from ..services.compression_service import CompressionService
from ...gac_system.gac_components import GradientCompressionComponent


class PadicGradientCompressionComponent(GradientCompressionComponent):
    """Enhanced GAC component with P-adic compression capabilities"""
    
    def __init__(self, component_id: str = "padic_gradient_compressor", config: Dict[str, Any] = None):
        """Initialize P-adic enhanced gradient compression component"""
        if config is None:
            config = {}
        
        # Initialize base GAC component
        super().__init__(component_id, config)
        
        # P-adic specific configuration
        padic_config = self._extract_padic_config(config)
        
        # Initialize P-adic gradient compressor
        try:
            self.padic_compressor = PadicGradientCompressor(padic_config)
        except Exception as e:
            raise ValueError(f"Failed to initialize P-adic compressor: {e}")
        
        # Integration configuration
        self.use_padic_compression = config.get('use_padic_compression', True)
        self.fallback_to_topk = config.get('fallback_to_topk', False)  # Should be False for fail-loud
        self.hybrid_mode = config.get('hybrid_mode', False)
        self.padic_threshold = config.get('padic_threshold', 1e-6)
        
        # Validate configuration
        self._validate_integration_config()
        
        # Performance tracking
        self.padic_stats = {
            'padic_compressions': 0,
            'topk_compressions': 0,
            'hybrid_compressions': 0,
            'compression_failures': 0
        }
    
    def _extract_padic_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract P-adic specific configuration"""
        padic_config = {}
        
        # Required P-adic parameters
        required_padic_params = ['prime', 'precision', 'chunk_size', 'gpu_memory_limit_mb']
        for param in required_padic_params:
            if param in config:
                padic_config[param] = config[param]
            else:
                # Set defaults - but fail loud if critical params missing
                if param == 'prime':
                    padic_config[param] = 2
                elif param == 'precision':
                    padic_config[param] = 32
                elif param == 'chunk_size':
                    padic_config[param] = 1024
                elif param == 'gpu_memory_limit_mb':
                    raise KeyError(f"Critical P-adic parameter missing: {param}")
        
        # Optional P-adic parameters
        optional_params = [
            'preserve_ultrametric', 'adaptive_precision', 'error_feedback',
            'gradient_threshold', 'preserve_gradient_norm', 'max_reconstruction_error'
        ]
        
        for param in optional_params:
            if param in config:
                padic_config[param] = config[param]
        
        return padic_config
    
    def _validate_integration_config(self) -> None:
        """Validate integration-specific configuration"""
        if not isinstance(self.use_padic_compression, bool):
            raise TypeError(f"use_padic_compression must be bool, got {type(self.use_padic_compression)}")
        
        if not isinstance(self.fallback_to_topk, bool):
            raise TypeError(f"fallback_to_topk must be bool, got {type(self.fallback_to_topk)}")
        
        if self.fallback_to_topk:
            raise ValueError("fallback_to_topk must be False for fail-loud operation")
        
        if not isinstance(self.hybrid_mode, bool):
            raise TypeError(f"hybrid_mode must be bool, got {type(self.hybrid_mode)}")
        
        if not isinstance(self.padic_threshold, (int, float)):
            raise TypeError(f"padic_threshold must be numeric, got {type(self.padic_threshold)}")
        
        if self.padic_threshold <= 0:
            raise ValueError(f"padic_threshold must be > 0, got {self.padic_threshold}")
    
    async def process_gradient(self, gradient: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """Process gradient using P-adic compression or hybrid approach"""
        start_time = time.time()
        
        try:
            # Validate input
            if not isinstance(gradient, torch.Tensor):
                raise TypeError(f"Gradient must be torch.Tensor, got {type(gradient)}")
            
            if gradient.numel() == 0:
                raise ValueError("Gradient tensor is empty")
            
            # Choose compression method
            if self.hybrid_mode:
                return await self._process_hybrid_compression(gradient, context)
            elif self.use_padic_compression:
                return await self._process_padic_compression(gradient, context)
            else:
                # Use original top-k compression
                return await super().process_gradient(gradient, context)
                
        except Exception as e:
            self.padic_stats['compression_failures'] += 1
            self.metrics.error_count += 1
            # Fail loud - re-raise the exception
            raise RuntimeError(f"P-adic gradient compression failed: {e}")
    
    async def _process_padic_compression(self, gradient: torch.Tensor, 
                                       context: Dict[str, Any]) -> torch.Tensor:
        """Process gradient using P-adic compression"""
        try:
            # Compress using P-adic method
            compressed = self.padic_compressor.compress_gradient(gradient, context)
            
            # Decompress immediately for gradient update
            processed_gradient = self.padic_compressor.decompress_gradient(compressed)
            
            # Update statistics
            self.padic_stats['padic_compressions'] += 1
            
            # Emit GAC event
            self.emit_event("GRADIENT_UPDATE", {
                "compression_method": "padic",
                "original_norm": torch.norm(gradient).item(),
                "processed_norm": torch.norm(processed_gradient).item(),
                "compression_ratio": compressed.get('compression_ratio', 0),
                "padic_prime": self.padic_compressor.padic_system.prime,
                "padic_precision": self.padic_compressor.padic_system.precision
            })
            
            return processed_gradient
            
        except Exception as e:
            raise RuntimeError(f"P-adic compression failed: {e}")
    
    async def _process_hybrid_compression(self, gradient: torch.Tensor,
                                        context: Dict[str, Any]) -> torch.Tensor:
        """Process gradient using hybrid P-adic + top-k compression"""
        try:
            gradient_norm = torch.norm(gradient).item()
            
            # Use P-adic for large gradients, top-k for small ones
            if gradient_norm >= self.padic_threshold:
                processed_gradient = await self._process_padic_compression(gradient, context)
                compression_method = "padic"
            else:
                processed_gradient = await super().process_gradient(gradient, context)
                compression_method = "topk"
                self.padic_stats['topk_compressions'] += 1
            
            self.padic_stats['hybrid_compressions'] += 1
            
            # Emit hybrid event
            self.emit_event("GRADIENT_UPDATE", {
                "compression_method": compression_method,
                "hybrid_mode": True,
                "gradient_norm": gradient_norm,
                "threshold": self.padic_threshold,
                "original_norm": torch.norm(gradient).item(),
                "processed_norm": torch.norm(processed_gradient).item()
            })
            
            return processed_gradient
            
        except Exception as e:
            raise RuntimeError(f"Hybrid compression failed: {e}")
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get component information including P-adic statistics"""
        # Get base component info
        base_info = super().get_component_info()
        
        # Add P-adic specific information
        padic_info = {
            "padic_enabled": self.use_padic_compression,
            "hybrid_mode": self.hybrid_mode,
            "padic_threshold": self.padic_threshold,
            "padic_prime": self.padic_compressor.padic_system.prime,
            "padic_precision": self.padic_compressor.padic_system.precision,
            "padic_statistics": dict(self.padic_stats),
            "padic_memory_usage": self.padic_compressor.get_memory_usage(),
            "padic_gradient_stats": self.padic_compressor.get_gradient_stats()
        }
        
        # Combine information
        base_info.update(padic_info)
        base_info["component_type"] = "padic_gradient_compressor"
        
        return base_info
    
    def get_padic_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed P-adic performance metrics"""
        total_compressions = (
            self.padic_stats['padic_compressions'] + 
            self.padic_stats['topk_compressions']
        )
        
        if total_compressions == 0:
            return {
                "total_compressions": 0,
                "padic_usage_rate": 0.0,
                "topk_usage_rate": 0.0,
                "failure_rate": 0.0
            }
        
        return {
            "total_compressions": total_compressions,
            "padic_compressions": self.padic_stats['padic_compressions'],
            "topk_compressions": self.padic_stats['topk_compressions'],
            "hybrid_compressions": self.padic_stats['hybrid_compressions'],
            "compression_failures": self.padic_stats['compression_failures'],
            "padic_usage_rate": self.padic_stats['padic_compressions'] / total_compressions,
            "topk_usage_rate": self.padic_stats['topk_compressions'] / total_compressions,
            "failure_rate": self.padic_stats['compression_failures'] / 
                           (total_compressions + self.padic_stats['compression_failures']),
            "system_performance": self.padic_compressor.padic_system.get_performance_stats(),
            "memory_efficiency": self.padic_compressor.get_memory_usage()
        }
    
    def reset_padic_stats(self) -> None:
        """Reset P-adic compression statistics"""
        self.padic_stats = {
            'padic_compressions': 0,
            'topk_compressions': 0,
            'hybrid_compressions': 0,
            'compression_failures': 0
        }
        
        # Reset underlying P-adic stats
        self.padic_compressor.reset_gradient_stats()
        self.padic_compressor.padic_system.reset_performance_stats()


class PadicCompressionService(CompressionService):
    """Compression service wrapper for P-adic compression"""
    
    def __init__(self, service_id: str = "padic_compression", config: Dict[str, Any] = None):
        """Initialize P-adic compression service"""
        if config is None:
            config = {}
        
        super().__init__(service_id, config)
        
        # Initialize P-adic compressor
        self.padic_compressor = PadicGradientCompressor(config)
        
        # Service configuration
        self.supported_data_types = config.get('supported_data_types', ['gradient', 'weight', 'activation'])
        
    def _validate_service_config(self) -> None:
        """Validate service-specific configuration"""
        if not isinstance(self.supported_data_types, list):
            raise TypeError(f"supported_data_types must be list, got {type(self.supported_data_types)}")
        
        if not self.supported_data_types:
            raise ValueError("supported_data_types cannot be empty")
        
        for data_type in self.supported_data_types:
            if not isinstance(data_type, str):
                raise TypeError(f"Data type must be str, got {type(data_type)}")
    
    async def compress(self, data: torch.Tensor, 
                      request_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compress data using P-adic compression"""
        if request_config is None:
            request_config = {}
        
        try:
            # Use P-adic gradient compressor
            compressed = self.padic_compressor.compress_gradient(data, request_config)
            
            # Add service metadata
            compressed['service_id'] = self.service_id
            compressed['compression_algorithm'] = 'padic'
            compressed['service_timestamp'] = time.time()
            
            return compressed
            
        except Exception as e:
            raise RuntimeError(f"P-adic compression service failed: {e}")
    
    async def decompress(self, compressed_data: Dict[str, Any]) -> torch.Tensor:
        """Decompress P-adic compressed data"""
        try:
            # Validate service metadata
            if compressed_data.get('service_id') != self.service_id:
                raise ValueError(f"Service ID mismatch: {compressed_data.get('service_id')} != {self.service_id}")
            
            if compressed_data.get('compression_algorithm') != 'padic':
                raise ValueError(f"Algorithm mismatch: {compressed_data.get('compression_algorithm')} != padic")
            
            # Decompress using P-adic compressor
            return self.padic_compressor.decompress_gradient(compressed_data)
            
        except Exception as e:
            raise RuntimeError(f"P-adic decompression service failed: {e}")
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics"""
        return {
            'service_id': self.service_id,
            'supported_data_types': self.supported_data_types,
            'padic_metrics': self.padic_compressor.get_gradient_stats(),
            'memory_usage': self.padic_compressor.get_memory_usage(),
            'performance_stats': self.padic_compressor.padic_system.get_performance_stats()
        }
    
    def supports_data_type(self, data_type: str) -> bool:
        """Check if service supports given data type"""
        if not isinstance(data_type, str):
            raise TypeError(f"Data type must be str, got {type(data_type)}")
        
        return data_type in self.supported_data_types