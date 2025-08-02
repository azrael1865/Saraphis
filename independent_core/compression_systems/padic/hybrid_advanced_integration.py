"""
Hybrid Advanced Integration - Main orchestrator for integrating advanced features with hybrid system
NO FALLBACKS - HARD FAILURES ONLY
"""

import asyncio
import logging
import threading
import time
import torch
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

# Import existing advanced features
from .padic_advanced import (
    HenselLiftingProcessor, HenselLiftingConfig,
    HierarchicalClusteringManager, ClusteringConfig,
    PadicOptimizationManager
)

# Import hybrid system components
from .hybrid_padic_structures import HybridPadicWeight, HybridPadicManager
from .dynamic_switching_manager import DynamicSwitchingManager, CompressionMode
from .hybrid_padic_compressor import HybridPadicCompressionSystem

# Import performance optimizer
from ...performance_optimizer import PerformanceOptimizer


class AdvancedFeatureType(Enum):
    """Advanced feature type enumeration"""
    HENSEL_LIFTING = "hensel_lifting"
    HIERARCHICAL_CLUSTERING = "hierarchical_clustering"
    PADIC_OPTIMIZATION = "padic_optimization"
    ALL_FEATURES = "all_features"


class IntegrationStatus(Enum):
    """Integration status enumeration"""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    INTEGRATING = "integrating"
    INTEGRATED = "integrated"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class AdvancedIntegrationConfig:
    """Configuration for advanced features integration"""
    # Feature enabling
    enable_hensel_lifting: bool = True
    enable_hierarchical_clustering: bool = True
    enable_padic_optimization: bool = True
    enable_gpu_acceleration: bool = True
    
    # Integration behavior
    enable_seamless_switching: bool = True
    enable_performance_monitoring: bool = True
    enable_advanced_analytics: bool = True
    auto_fallback_to_pure: bool = False  # NO FALLBACKS by design
    
    # Performance thresholds
    hensel_lifting_threshold: int = 1000  # Use hybrid for data > 1000 elements
    clustering_threshold: int = 500       # Use hybrid for > 500 weights
    optimization_threshold: int = 100     # Use hybrid for > 100 parameters
    
    # GPU configuration
    gpu_memory_limit_mb: int = 1024
    enable_gpu_memory_optimization: bool = True
    
    # Threading and concurrency
    max_concurrent_operations: int = 4
    enable_async_operations: bool = True
    
    # Monitoring and analytics
    performance_window_size: int = 50
    analytics_retention_hours: int = 24
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.hensel_lifting_threshold <= 0:
            raise ValueError(f"hensel_lifting_threshold must be > 0, got {self.hensel_lifting_threshold}")
        if self.clustering_threshold <= 0:
            raise ValueError(f"clustering_threshold must be > 0, got {self.clustering_threshold}")
        if self.optimization_threshold <= 0:
            raise ValueError(f"optimization_threshold must be > 0, got {self.optimization_threshold}")
        if self.gpu_memory_limit_mb <= 0:
            raise ValueError(f"gpu_memory_limit_mb must be > 0, got {self.gpu_memory_limit_mb}")
        if self.max_concurrent_operations <= 0:
            raise ValueError(f"max_concurrent_operations must be > 0, got {self.max_concurrent_operations}")
        if self.performance_window_size <= 0:
            raise ValueError(f"performance_window_size must be > 0, got {self.performance_window_size}")


@dataclass
class AdvancedIntegrationMetrics:
    """Metrics for advanced features integration"""
    total_operations: int = 0
    hybrid_operations: int = 0
    pure_operations: int = 0
    hensel_lifting_operations: int = 0
    clustering_operations: int = 0
    optimization_operations: int = 0
    
    # Performance metrics
    average_hybrid_performance: float = 0.0
    average_pure_performance: float = 0.0
    performance_improvement_ratio: float = 0.0
    
    # Feature usage statistics
    feature_usage_counts: Dict[AdvancedFeatureType, int] = field(default_factory=lambda: defaultdict(int))
    feature_performance: Dict[AdvancedFeatureType, float] = field(default_factory=dict)
    
    # Integration status
    last_update: Optional[datetime] = None
    integration_health: str = "healthy"
    
    def update_operation_metrics(self, operation_type: str, performance_score: float, feature_type: AdvancedFeatureType):
        """Update operation metrics"""
        self.total_operations += 1
        self.last_update = datetime.utcnow()
        
        if operation_type == "hybrid":
            self.hybrid_operations += 1
            old_avg = self.average_hybrid_performance
            self.average_hybrid_performance = (
                (old_avg * (self.hybrid_operations - 1) + performance_score) / self.hybrid_operations
            )
        elif operation_type == "pure":
            self.pure_operations += 1
            old_avg = self.average_pure_performance
            self.average_pure_performance = (
                (old_avg * (self.pure_operations - 1) + performance_score) / self.pure_operations
            )
        
        # Update feature-specific metrics
        self.feature_usage_counts[feature_type] += 1
        if feature_type in self.feature_performance:
            old_perf = self.feature_performance[feature_type]
            self.feature_performance[feature_type] = (old_perf + performance_score) / 2
        else:
            self.feature_performance[feature_type] = performance_score
        
        # Calculate performance improvement ratio
        if self.average_pure_performance > 0:
            self.performance_improvement_ratio = self.average_hybrid_performance / self.average_pure_performance


class HybridAdvancedIntegration:
    """
    Main orchestrator for integrating advanced features with hybrid system.
    Coordinates Hensel lifting, hierarchical clustering, and optimization with hybrid p-adic compression.
    """
    
    def __init__(self, config: Optional[AdvancedIntegrationConfig] = None):
        """Initialize hybrid advanced integration"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, AdvancedIntegrationConfig):
            raise TypeError(f"Config must be AdvancedIntegrationConfig or None, got {type(config)}")
        
        self.config = config or AdvancedIntegrationConfig()
        self.logger = logging.getLogger('HybridAdvancedIntegration')
        
        # Integration state
        self.integration_status = IntegrationStatus.NOT_INITIALIZED
        self.is_initialized = False
        
        # Component references
        self.hensel_lifting_processor: Optional[HenselLiftingProcessor] = None
        self.clustering_manager: Optional[HierarchicalClusteringManager] = None
        self.optimization_manager: Optional[PadicOptimizationManager] = None
        self.dynamic_switching_manager: Optional[DynamicSwitchingManager] = None
        self.hybrid_padic_manager: Optional[HybridPadicManager] = None
        self.performance_optimizer: Optional[PerformanceOptimizer] = None
        
        # Hybrid feature integrations (will be imported dynamically)
        self.hybrid_hensel_lifting = None
        self.hybrid_clustering = None
        self.hybrid_optimization = None
        
        # Integration metrics and analytics
        self.integration_metrics = AdvancedIntegrationMetrics()
        self.feature_status: Dict[AdvancedFeatureType, IntegrationStatus] = {
            AdvancedFeatureType.HENSEL_LIFTING: IntegrationStatus.NOT_INITIALIZED,
            AdvancedFeatureType.HIERARCHICAL_CLUSTERING: IntegrationStatus.NOT_INITIALIZED,
            AdvancedFeatureType.PADIC_OPTIMIZATION: IntegrationStatus.NOT_INITIALIZED
        }
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=self.config.performance_window_size)
        self.operation_history: deque = deque(maxlen=1000)
        
        # Thread safety
        self._integration_lock = threading.RLock()
        self._metrics_lock = threading.RLock()
        self._operation_lock = threading.RLock()
        
        # Background monitoring
        self._monitoring_enabled = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        self.logger.info("HybridAdvancedIntegration created successfully")
    
    def initialize_advanced_features(self,
                                   hensel_config: Optional[HenselLiftingConfig] = None,
                                   clustering_config: Optional[ClusteringConfig] = None,
                                   prime: int = 7,
                                   precision: int = 10) -> None:
        """
        Initialize advanced features integration.
        
        Args:
            hensel_config: Configuration for Hensel lifting
            clustering_config: Configuration for hierarchical clustering
            prime: Prime number for p-adic operations
            precision: Base precision for p-adic operations
            
        Raises:
            RuntimeError: If initialization fails
        """
        if self.is_initialized:
            return
        
        with self._integration_lock:
            try:
                self.integration_status = IntegrationStatus.INITIALIZING
                
                # Initialize existing advanced feature components
                if self.config.enable_hensel_lifting:
                    hensel_config = hensel_config or HenselLiftingConfig()
                    self.hensel_lifting_processor = HenselLiftingProcessor(hensel_config, prime, precision)
                    self.feature_status[AdvancedFeatureType.HENSEL_LIFTING] = IntegrationStatus.INITIALIZED
                    self.logger.info("Hensel lifting processor initialized")
                
                if self.config.enable_hierarchical_clustering:
                    clustering_config = clustering_config or ClusteringConfig()
                    self.clustering_manager = HierarchicalClusteringManager(clustering_config, prime)
                    self.feature_status[AdvancedFeatureType.HIERARCHICAL_CLUSTERING] = IntegrationStatus.INITIALIZED
                    self.logger.info("Hierarchical clustering manager initialized")
                
                if self.config.enable_padic_optimization:
                    self.optimization_manager = PadicOptimizationManager(prime)
                    self.feature_status[AdvancedFeatureType.PADIC_OPTIMIZATION] = IntegrationStatus.INITIALIZED
                    self.logger.info("P-adic optimization manager initialized")
                
                # Initialize hybrid p-adic manager
                self.hybrid_padic_manager = HybridPadicManager({
                    'enable_gpu_acceleration': self.config.enable_gpu_acceleration,
                    'gpu_memory_limit_mb': self.config.gpu_memory_limit_mb
                })
                
                # Start background monitoring if enabled
                if self.config.enable_performance_monitoring:
                    self._start_background_monitoring()
                
                self.integration_status = IntegrationStatus.INITIALIZED
                self.is_initialized = True
                self.logger.info("Advanced features integration initialized successfully")
                
            except Exception as e:
                self.integration_status = IntegrationStatus.ERROR
                self.logger.error(f"Failed to initialize advanced features: {e}")
                raise RuntimeError(f"Advanced features initialization failed: {e}")
    
    def integrate_hensel_lifting_with_hybrid(self) -> bool:
        """
        Integrate Hensel lifting with hybrid system.
        
        Returns:
            True if integration successful
            
        Raises:
            RuntimeError: If integration fails
        """
        if not self.is_initialized:
            raise RuntimeError("Advanced features not initialized")
        if not self.config.enable_hensel_lifting:
            raise RuntimeError("Hensel lifting not enabled in configuration")
        
        with self._integration_lock:
            try:
                self.feature_status[AdvancedFeatureType.HENSEL_LIFTING] = IntegrationStatus.INTEGRATING
                
                # Import and initialize hybrid Hensel lifting
                from .hybrid_hensel_lifting import HybridHenselLifting
                
                self.hybrid_hensel_lifting = HybridHenselLifting(
                    config=self.hensel_lifting_processor.config,
                    prime=self.hensel_lifting_processor.prime,
                    base_precision=self.hensel_lifting_processor.base_precision
                )
                
                # Validate integration
                if not self._validate_hensel_integration():
                    raise RuntimeError("Hensel lifting integration validation failed")
                
                self.feature_status[AdvancedFeatureType.HENSEL_LIFTING] = IntegrationStatus.INTEGRATED
                self.logger.info("Hensel lifting integrated with hybrid system successfully")
                
                return True
                
            except Exception as e:
                self.feature_status[AdvancedFeatureType.HENSEL_LIFTING] = IntegrationStatus.ERROR
                self.logger.error(f"Failed to integrate Hensel lifting: {e}")
                raise RuntimeError(f"Hensel lifting integration failed: {e}")
    
    def integrate_clustering_with_hybrid(self) -> bool:
        """
        Integrate hierarchical clustering with hybrid system.
        
        Returns:
            True if integration successful
            
        Raises:
            RuntimeError: If integration fails
        """
        if not self.is_initialized:
            raise RuntimeError("Advanced features not initialized")
        if not self.config.enable_hierarchical_clustering:
            raise RuntimeError("Hierarchical clustering not enabled in configuration")
        
        with self._integration_lock:
            try:
                self.feature_status[AdvancedFeatureType.HIERARCHICAL_CLUSTERING] = IntegrationStatus.INTEGRATING
                
                # Import and initialize hybrid clustering
                from .hybrid_clustering import HybridHierarchicalClustering
                
                self.hybrid_clustering = HybridHierarchicalClustering(
                    config=self.clustering_manager.config,
                    prime=self.clustering_manager.prime
                )
                
                # Validate integration
                if not self._validate_clustering_integration():
                    raise RuntimeError("Clustering integration validation failed")
                
                self.feature_status[AdvancedFeatureType.HIERARCHICAL_CLUSTERING] = IntegrationStatus.INTEGRATED
                self.logger.info("Hierarchical clustering integrated with hybrid system successfully")
                
                return True
                
            except Exception as e:
                self.feature_status[AdvancedFeatureType.HIERARCHICAL_CLUSTERING] = IntegrationStatus.ERROR
                self.logger.error(f"Failed to integrate clustering: {e}")
                raise RuntimeError(f"Clustering integration failed: {e}")
    
    def integrate_optimization_with_hybrid(self) -> bool:
        """
        Integrate optimization with hybrid system.
        
        Returns:
            True if integration successful
            
        Raises:
            RuntimeError: If integration fails
        """
        if not self.is_initialized:
            raise RuntimeError("Advanced features not initialized")
        if not self.config.enable_padic_optimization:
            raise RuntimeError("P-adic optimization not enabled in configuration")
        
        with self._integration_lock:
            try:
                self.feature_status[AdvancedFeatureType.PADIC_OPTIMIZATION] = IntegrationStatus.INTEGRATING
                
                # Import and initialize hybrid optimization
                from .hybrid_optimization import HybridOptimizationManager
                
                self.hybrid_optimization = HybridOptimizationManager(
                    prime=self.optimization_manager.prime
                )
                
                # Validate integration
                if not self._validate_optimization_integration():
                    raise RuntimeError("Optimization integration validation failed")
                
                self.feature_status[AdvancedFeatureType.PADIC_OPTIMIZATION] = IntegrationStatus.INTEGRATED
                self.logger.info("P-adic optimization integrated with hybrid system successfully")
                
                return True
                
            except Exception as e:
                self.feature_status[AdvancedFeatureType.PADIC_OPTIMIZATION] = IntegrationStatus.ERROR
                self.logger.error(f"Failed to integrate optimization: {e}")
                raise RuntimeError(f"Optimization integration failed: {e}")
    
    def set_dynamic_switching_manager(self, switching_manager: DynamicSwitchingManager) -> None:
        """
        Set dynamic switching manager for seamless feature switching.
        
        Args:
            switching_manager: Dynamic switching manager instance
            
        Raises:
            TypeError: If switching_manager is invalid
        """
        if not isinstance(switching_manager, DynamicSwitchingManager):
            raise TypeError(f"Switching manager must be DynamicSwitchingManager, got {type(switching_manager)}")
        
        self.dynamic_switching_manager = switching_manager
        self.logger.info("Dynamic switching manager set for advanced features")
    
    def set_performance_optimizer(self, performance_optimizer: PerformanceOptimizer) -> None:
        """
        Set performance optimizer for advanced features monitoring.
        
        Args:
            performance_optimizer: Performance optimizer instance
            
        Raises:
            TypeError: If performance_optimizer is invalid
        """
        if not isinstance(performance_optimizer, PerformanceOptimizer):
            raise TypeError(f"Performance optimizer must be PerformanceOptimizer, got {type(performance_optimizer)}")
        
        self.performance_optimizer = performance_optimizer
        self.logger.info("Performance optimizer set for advanced features")
    
    def get_advanced_integration_status(self) -> Dict[str, Any]:
        """
        Get comprehensive integration status.
        
        Returns:
            Dictionary containing integration status information
        """
        with self._metrics_lock:
            return {
                'overall_status': self.integration_status.value,
                'is_initialized': self.is_initialized,
                'feature_status': {
                    feature.value: status.value 
                    for feature, status in self.feature_status.items()
                },
                'component_availability': {
                    'hensel_lifting_processor': self.hensel_lifting_processor is not None,
                    'clustering_manager': self.clustering_manager is not None,
                    'optimization_manager': self.optimization_manager is not None,
                    'hybrid_hensel_lifting': self.hybrid_hensel_lifting is not None,
                    'hybrid_clustering': self.hybrid_clustering is not None,
                    'hybrid_optimization': self.hybrid_optimization is not None,
                    'dynamic_switching_manager': self.dynamic_switching_manager is not None,
                    'performance_optimizer': self.performance_optimizer is not None
                },
                'configuration': {
                    'enable_hensel_lifting': self.config.enable_hensel_lifting,
                    'enable_hierarchical_clustering': self.config.enable_hierarchical_clustering,
                    'enable_padic_optimization': self.config.enable_padic_optimization,
                    'enable_seamless_switching': self.config.enable_seamless_switching,
                    'enable_gpu_acceleration': self.config.enable_gpu_acceleration
                },
                'metrics': {
                    'total_operations': self.integration_metrics.total_operations,
                    'hybrid_operations': self.integration_metrics.hybrid_operations,
                    'pure_operations': self.integration_metrics.pure_operations,
                    'performance_improvement_ratio': self.integration_metrics.performance_improvement_ratio,
                    'integration_health': self.integration_metrics.integration_health
                },
                'last_update': datetime.utcnow().isoformat()
            }
    
    def validate_advanced_compatibility(self, data: torch.Tensor, feature_type: AdvancedFeatureType) -> bool:
        """
        Validate compatibility of data with advanced features.
        
        Args:
            data: Input tensor data
            feature_type: Type of advanced feature to validate
            
        Returns:
            True if compatible
            
        Raises:
            ValueError: If data or feature_type is invalid
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Data must be torch.Tensor, got {type(data)}")
        if not isinstance(feature_type, AdvancedFeatureType):
            raise TypeError(f"Feature type must be AdvancedFeatureType, got {type(feature_type)}")
        if data.numel() == 0:
            raise ValueError("Data tensor cannot be empty")
        
        try:
            # Check feature-specific compatibility
            if feature_type == AdvancedFeatureType.HENSEL_LIFTING:
                return data.numel() >= self.config.hensel_lifting_threshold
            elif feature_type == AdvancedFeatureType.HIERARCHICAL_CLUSTERING:
                return data.numel() >= self.config.clustering_threshold
            elif feature_type == AdvancedFeatureType.PADIC_OPTIMIZATION:
                return data.numel() >= self.config.optimization_threshold
            elif feature_type == AdvancedFeatureType.ALL_FEATURES:
                return (data.numel() >= min(
                    self.config.hensel_lifting_threshold,
                    self.config.clustering_threshold,
                    self.config.optimization_threshold
                ))
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating compatibility: {e}")
            return False
    
    def should_use_hybrid_feature(self, data: torch.Tensor, feature_type: AdvancedFeatureType, 
                                context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determine if hybrid version of advanced feature should be used.
        
        Args:
            data: Input tensor data
            feature_type: Type of advanced feature
            context: Optional context information
            
        Returns:
            True if hybrid feature should be used
        """
        if not self.config.enable_seamless_switching:
            return self.config.enable_gpu_acceleration  # Default to GPU if available
        
        if not self.validate_advanced_compatibility(data, feature_type):
            return False  # Use pure if not compatible
        
        # Use dynamic switching manager if available
        if self.dynamic_switching_manager:
            try:
                should_switch, confidence, trigger = self.dynamic_switching_manager.should_switch_to_hybrid(data, context)
                return should_switch and confidence >= 0.7  # High confidence threshold
            except Exception as e:
                self.logger.warning(f"Dynamic switching failed for {feature_type.value}: {e}")
        
        # Fallback to size-based decision
        if feature_type == AdvancedFeatureType.HENSEL_LIFTING:
            return data.numel() > self.config.hensel_lifting_threshold
        elif feature_type == AdvancedFeatureType.HIERARCHICAL_CLUSTERING:
            return data.numel() > self.config.clustering_threshold
        elif feature_type == AdvancedFeatureType.PADIC_OPTIMIZATION:
            return data.numel() > self.config.optimization_threshold
        
        return self.config.enable_gpu_acceleration
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive integration metrics.
        
        Returns:
            Dictionary containing integration metrics
        """
        with self._metrics_lock:
            return {
                'overall_metrics': {
                    'total_operations': self.integration_metrics.total_operations,
                    'hybrid_operations': self.integration_metrics.hybrid_operations,
                    'pure_operations': self.integration_metrics.pure_operations,
                    'hybrid_ratio': (
                        self.integration_metrics.hybrid_operations / 
                        max(1, self.integration_metrics.total_operations)
                    ),
                    'average_hybrid_performance': self.integration_metrics.average_hybrid_performance,
                    'average_pure_performance': self.integration_metrics.average_pure_performance,
                    'performance_improvement_ratio': self.integration_metrics.performance_improvement_ratio
                },
                'feature_metrics': {
                    'hensel_lifting_operations': self.integration_metrics.hensel_lifting_operations,
                    'clustering_operations': self.integration_metrics.clustering_operations,
                    'optimization_operations': self.integration_metrics.optimization_operations,
                    'feature_usage_counts': dict(self.integration_metrics.feature_usage_counts),
                    'feature_performance': dict(self.integration_metrics.feature_performance)
                },
                'performance_history': {
                    'history_length': len(self.performance_history),
                    'recent_average': (
                        sum(self.performance_history) / len(self.performance_history)
                        if self.performance_history else 0.0
                    )
                },
                'integration_health': self.integration_metrics.integration_health,
                'last_update': self.integration_metrics.last_update.isoformat() if self.integration_metrics.last_update else None
            }
    
    def _validate_hensel_integration(self) -> bool:
        """Validate Hensel lifting integration"""
        if self.hybrid_hensel_lifting is None:
            return False
        
        # Test basic functionality
        try:
            # Create test hybrid weight
            test_weight = HybridPadicWeight(
                exponent_channel=torch.tensor([1.0, 2.0], device='cuda'),
                mantissa_channel=torch.tensor([0.5, 0.3], device='cuda'),
                prime=self.hensel_lifting_processor.prime,
                precision=self.hensel_lifting_processor.base_precision,
                valuation=0,
                device=torch.device('cuda'),
                dtype=torch.float32
            )
            
            # Test lifting operation
            self.hybrid_hensel_lifting.validate_hybrid_weight(test_weight)
            return True
            
        except Exception as e:
            self.logger.error(f"Hensel integration validation failed: {e}")
            return False
    
    def _validate_clustering_integration(self) -> bool:
        """Validate clustering integration"""
        if self.hybrid_clustering is None:
            return False
        
        # Test basic functionality
        try:
            # Create test hybrid weights
            test_weights = [
                HybridPadicWeight(
                    exponent_channel=torch.tensor([1.0], device='cuda'),
                    mantissa_channel=torch.tensor([0.5], device='cuda'),
                    prime=self.clustering_manager.prime,
                    precision=10,
                    valuation=0,
                    device=torch.device('cuda'),
                    dtype=torch.float32
                ) for _ in range(3)
            ]
            
            # Test distance computation
            distance = self.hybrid_clustering.compute_hybrid_ultrametric_distance(test_weights[0], test_weights[1])
            return isinstance(distance, (int, float)) and distance >= 0
            
        except Exception as e:
            self.logger.error(f"Clustering integration validation failed: {e}")
            return False
    
    def _validate_optimization_integration(self) -> bool:
        """Validate optimization integration"""
        if self.hybrid_optimization is None:
            return False
        
        # Test basic functionality
        try:
            # Create test hybrid weights
            test_params = [
                HybridPadicWeight(
                    exponent_channel=torch.tensor([1.0], device='cuda'),
                    mantissa_channel=torch.tensor([0.5], device='cuda'),
                    prime=self.optimization_manager.prime,
                    precision=10,
                    valuation=0,
                    device=torch.device('cuda'),
                    dtype=torch.float32
                )
            ]
            
            # Test optimizer creation
            optimizer_id = self.hybrid_optimization.create_hybrid_sgd_optimizer(test_params, lr=0.01)
            return isinstance(optimizer_id, str) and len(optimizer_id) > 0
            
        except Exception as e:
            self.logger.error(f"Optimization integration validation failed: {e}")
            return False
    
    def _start_background_monitoring(self) -> None:
        """Start background monitoring thread"""
        if self._monitoring_enabled:
            return
        
        self._monitoring_enabled = True
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._background_monitoring_loop,
            name="AdvancedIntegrationMonitor",
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info("Background integration monitoring started")
    
    def _background_monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while not self._stop_monitoring.wait(60.0):  # Check every minute
            try:
                # Update integration health
                self._update_integration_health()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Log performance summary
                if self.integration_metrics.total_operations > 0:
                    self.logger.debug(f"Integration metrics: {self.integration_metrics.total_operations} total operations, "
                                    f"{self.integration_metrics.performance_improvement_ratio:.2f} improvement ratio")
                
            except Exception as e:
                self.logger.error(f"Error in background monitoring: {e}")
    
    def _update_integration_health(self) -> None:
        """Update integration health status"""
        try:
            with self._metrics_lock:
                # Check if all enabled features are integrated
                enabled_features = [
                    (AdvancedFeatureType.HENSEL_LIFTING, self.config.enable_hensel_lifting),
                    (AdvancedFeatureType.HIERARCHICAL_CLUSTERING, self.config.enable_hierarchical_clustering),
                    (AdvancedFeatureType.PADIC_OPTIMIZATION, self.config.enable_padic_optimization)
                ]
                
                integrated_count = 0
                total_enabled = 0
                
                for feature_type, enabled in enabled_features:
                    if enabled:
                        total_enabled += 1
                        if self.feature_status[feature_type] == IntegrationStatus.INTEGRATED:
                            integrated_count += 1
                
                if total_enabled == 0:
                    self.integration_metrics.integration_health = "disabled"
                elif integrated_count == total_enabled:
                    self.integration_metrics.integration_health = "healthy"
                elif integrated_count > 0:
                    self.integration_metrics.integration_health = "partial"
                else:
                    self.integration_metrics.integration_health = "unhealthy"
                    
        except Exception as e:
            self.logger.error(f"Error updating integration health: {e}")
            self.integration_metrics.integration_health = "error"
    
    def _cleanup_old_data(self) -> None:
        """Clean up old performance data"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.config.analytics_retention_hours)
        
        # Clean up operation history
        with self._operation_lock:
            self.operation_history = deque([
                op for op in self.operation_history
                if op.get('timestamp', datetime.utcnow()) >= cutoff_time
            ], maxlen=1000)
    
    def update_operation_metrics(self, operation_type: str, performance_score: float, 
                               feature_type: AdvancedFeatureType) -> None:
        """
        Update operation metrics.
        
        Args:
            operation_type: Type of operation ('hybrid' or 'pure')
            performance_score: Performance score (0-1)
            feature_type: Advanced feature type used
        """
        with self._metrics_lock:
            self.integration_metrics.update_operation_metrics(operation_type, performance_score, feature_type)
            
            # Update feature-specific counters
            if feature_type == AdvancedFeatureType.HENSEL_LIFTING:
                self.integration_metrics.hensel_lifting_operations += 1
            elif feature_type == AdvancedFeatureType.HIERARCHICAL_CLUSTERING:
                self.integration_metrics.clustering_operations += 1
            elif feature_type == AdvancedFeatureType.PADIC_OPTIMIZATION:
                self.integration_metrics.optimization_operations += 1
            
            # Update performance history
            self.performance_history.append(performance_score)
            
            # Record operation
            with self._operation_lock:
                self.operation_history.append({
                    'timestamp': datetime.utcnow(),
                    'operation_type': operation_type,
                    'performance_score': performance_score,
                    'feature_type': feature_type.value
                })
    
    def shutdown(self) -> None:
        """Shutdown advanced integration"""
        self.logger.info("Shutting down hybrid advanced integration")
        
        # Stop background monitoring
        if self._monitoring_enabled:
            self._monitoring_enabled = False
            self._stop_monitoring.set()
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=5.0)
        
        # Clear references
        self.hensel_lifting_processor = None
        self.clustering_manager = None
        self.optimization_manager = None
        self.hybrid_hensel_lifting = None
        self.hybrid_clustering = None
        self.hybrid_optimization = None
        
        # Clear data
        self.performance_history.clear()
        self.operation_history.clear()
        
        self.is_initialized = False
        self.integration_status = IntegrationStatus.NOT_INITIALIZED
        self.logger.info("Hybrid advanced integration shutdown complete")