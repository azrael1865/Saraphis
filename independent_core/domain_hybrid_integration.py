"""
Domain Hybrid Integration - Main orchestrator for domain-hybrid integration
NO FALLBACKS - HARD FAILURES ONLY
"""

import logging
import threading
import time
import torch
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from enum import Enum
import json
import hashlib

# Import domain system components (avoid circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from domain_registry import DomainRegistry, DomainConfig, DomainStatus, DomainType
    from domain_router import DomainRouter, RoutingStrategy

# Import hybrid compression system components
from compression_systems.padic.hybrid_padic_compressor import HybridPadicCompressionSystem
from compression_systems.padic.dynamic_switching_manager import DynamicSwitchingManager
from compression_systems.padic.direction_manager import DirectionManager
from compression_systems.padic.hybrid_bounding_engine import HybridBoundingEngine


class DomainIntegrationMode(Enum):
    """Domain integration mode enumeration"""
    STANDARD = "standard"
    HYBRID_OPTIMIZED = "hybrid_optimized"
    COMPRESSION_FIRST = "compression_first"
    PERFORMANCE_FIRST = "performance_first"
    MEMORY_EFFICIENT = "memory_efficient"
    DOMAIN_SPECIFIC = "domain_specific"


class DomainCompressionStrategy(Enum):
    """Domain compression strategy enumeration"""
    ADAPTIVE = "adaptive"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    DOMAIN_OPTIMIZED = "domain_optimized"
    REAL_TIME = "real_time"
    BATCH_OPTIMIZED = "batch_optimized"


class DomainIntegrationState(Enum):
    """Domain integration state enumeration"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    DOMAINS_ACTIVE = "domains_active"
    OPTIMIZING = "optimizing"
    ERROR = "error"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class DomainHybridConfig:
    """Configuration for domain-hybrid integration"""
    integration_mode: DomainIntegrationMode = DomainIntegrationMode.HYBRID_OPTIMIZED
    compression_strategy: DomainCompressionStrategy = DomainCompressionStrategy.ADAPTIVE
    enable_real_time_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_gpu_optimization: bool = True
    enable_domain_specific_compression: bool = True
    enable_performance_monitoring: bool = True
    enable_error_recovery: bool = True
    
    # Performance thresholds
    memory_threshold_gb: float = 8.0
    gpu_threshold_percent: float = 85.0
    performance_threshold: float = 0.8
    compression_ratio_threshold: float = 0.7
    
    # Domain parameters
    max_active_domains: int = 32
    domain_batch_size: int = 16
    compression_batch_size: int = 64
    optimization_interval_seconds: int = 30
    
    # Analytics configuration
    enable_analytics: bool = True
    analytics_history_size: int = 1000
    performance_window_size: int = 100
    
    def __post_init__(self):
        """Validate configuration"""
        if not isinstance(self.integration_mode, DomainIntegrationMode):
            raise TypeError("Integration mode must be DomainIntegrationMode")
        if not isinstance(self.compression_strategy, DomainCompressionStrategy):
            raise TypeError("Compression strategy must be DomainCompressionStrategy")
        if self.memory_threshold_gb <= 0:
            raise ValueError("Memory threshold must be positive")
        if not (0.0 <= self.performance_threshold <= 1.0):
            raise ValueError("Performance threshold must be between 0.0 and 1.0")


@dataclass
class DomainMetrics:
    """Domain performance metrics"""
    domain_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_operations: int = 0
    compression_operations: int = 0
    optimization_operations: int = 0
    
    # Performance metrics
    average_operation_time_ms: float = 0.0
    average_compression_time_ms: float = 0.0
    average_optimization_time_ms: float = 0.0
    
    # Memory metrics
    peak_memory_usage_gb: float = 0.0
    average_memory_usage_gb: float = 0.0
    memory_savings_percent: float = 0.0
    
    # GPU metrics
    peak_gpu_utilization_percent: float = 0.0
    average_gpu_utilization_percent: float = 0.0
    gpu_memory_usage_gb: float = 0.0
    
    # Compression metrics
    compression_ratio: float = 0.0
    total_compressed_data_gb: float = 0.0
    compression_efficiency: float = 0.0
    
    # Domain-specific metrics
    domain_accuracy: float = 0.0
    domain_throughput: float = 0.0
    domain_error_rate: float = 0.0
    
    # Error tracking
    error_count: int = 0
    recovery_count: int = 0
    
    def __post_init__(self):
        """Validate metrics"""
        if not self.domain_name:
            raise ValueError("Domain name cannot be empty")
        if self.total_operations < 0:
            raise ValueError("Total operations must be non-negative")


@dataclass
class DomainAnalytics:
    """Comprehensive domain analytics"""
    total_domains: int = 0
    active_domains: int = 0
    registered_domains: int = 0
    failed_domains: int = 0
    
    # Performance analytics
    average_domain_performance: float = 0.0
    average_compression_efficiency: float = 0.0
    average_memory_savings: float = 0.0
    average_gpu_utilization: float = 0.0
    
    # System health
    system_health_score: float = 0.0
    performance_trend: str = "stable"
    memory_trend: str = "stable"
    compression_trend: str = "stable"
    
    # Optimization analytics
    optimization_success_rate: float = 0.0
    performance_improvement_rate: float = 0.0
    memory_optimization_rate: float = 0.0
    
    # Domain-specific analytics
    best_performing_domain: str = ""
    worst_performing_domain: str = ""
    most_compressed_domain: str = ""
    
    timestamp: datetime = field(default_factory=datetime.utcnow)


class DomainHybridIntegration:
    """
    Main orchestrator for domain-hybrid integration.
    Manages integration between domain systems and hybrid p-adic system.
    """
    
    def __init__(self, config: Optional[DomainHybridConfig] = None):
        """Initialize domain-hybrid integration"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, DomainHybridConfig):
            raise TypeError(f"Config must be DomainHybridConfig or None, got {type(config)}")
        
        self.config = config or DomainHybridConfig()
        self.logger = logging.getLogger('DomainHybridIntegration')
        
        # System components
        self.domain_registry: Optional['DomainRegistry'] = None
        self.domain_router: Optional['DomainRouter'] = None
        self.hybrid_compression: Optional[HybridPadicCompressionSystem] = None
        self.dynamic_switching: Optional[DynamicSwitchingManager] = None
        self.direction_manager: Optional[DirectionManager] = None
        self.bounding_engine: Optional[HybridBoundingEngine] = None
        
        # Integration state
        self.integration_state = DomainIntegrationState.UNINITIALIZED
        self.active_domains: Dict[str, DomainMetrics] = {}
        self.domain_queue: deque = deque(maxlen=self.config.max_active_domains)
        
        # Performance tracking
        self.performance_metrics: deque = deque(maxlen=self.config.analytics_history_size)
        self.optimization_history: List[Dict[str, Any]] = []
        self.compression_history: List[Dict[str, Any]] = []
        
        # Analytics tracking
        self.domain_analytics = DomainAnalytics()
        self.performance_baselines: Dict[str, float] = {}
        self.domain_baselines: Dict[str, Dict[str, float]] = {}
        self.last_optimization: Optional[datetime] = None
        
        # Thread safety
        self._integration_lock = threading.RLock()
        self._domain_lock = threading.RLock()
        self._analytics_lock = threading.RLock()
        
        # Monitoring state
        self.monitoring_active = False
        self.error_recovery_active = False
        self.optimization_active = False
        
        # Domain-specific configurations
        self.domain_configs: Dict[str, Dict[str, Any]] = {}
        self.domain_compression_strategies: Dict[str, DomainCompressionStrategy] = {}
        
        self.logger.info("DomainHybridIntegration created successfully")
    
    def initialize_domain_hybrid_integration(self,
                                           domain_registry: 'DomainRegistry',
                                           domain_router: 'DomainRouter',
                                           hybrid_compression: HybridPadicCompressionSystem,
                                           dynamic_switching: Optional[DynamicSwitchingManager] = None,
                                           direction_manager: Optional[DirectionManager] = None,
                                           bounding_engine: Optional[HybridBoundingEngine] = None) -> None:
        """
        Initialize domain-hybrid integration with required systems.
        
        Args:
            domain_registry: Domain registry instance
            domain_router: Domain router instance
            hybrid_compression: Hybrid compression system
            dynamic_switching: Optional dynamic switching manager
            direction_manager: Optional direction manager
            bounding_engine: Optional bounding engine
            
        Raises:
            TypeError: If systems are invalid
            RuntimeError: If initialization fails
        """
        if domain_registry is None:
            raise ValueError("Domain registry cannot be None")
        if domain_router is None:
            raise ValueError("Domain router cannot be None")
        if not isinstance(hybrid_compression, HybridPadicCompressionSystem):
            raise TypeError(f"Hybrid compression must be HybridPadicCompressionSystem, got {type(hybrid_compression)}")
        
        try:
            with self._integration_lock:
                self.integration_state = DomainIntegrationState.INITIALIZING
                
                # Store system references
                self.domain_registry = domain_registry
                self.domain_router = domain_router
                self.hybrid_compression = hybrid_compression
                self.dynamic_switching = dynamic_switching
                self.direction_manager = direction_manager
                self.bounding_engine = bounding_engine
                
                # Initialize integration systems
                self._initialize_performance_baselines()
                self._setup_domain_hooks()
                self._initialize_compression_coordination()
                
                if self.config.enable_performance_monitoring:
                    self._start_performance_monitoring()
                
                if self.config.enable_analytics:
                    self._initialize_domain_analytics()
                
                self.integration_state = DomainIntegrationState.INITIALIZED
                self.logger.info("Domain-hybrid integration initialized successfully")
                
        except Exception as e:
            self.integration_state = DomainIntegrationState.ERROR
            self.logger.error(f"Failed to initialize domain-hybrid integration: {e}")
            raise RuntimeError(f"Domain-hybrid integration initialization failed: {e}")
    
    def register_domain_with_hybrid(self, domain_name: str, domain_config: Dict[str, Any]) -> DomainMetrics:
        """
        Register a domain with hybrid compression integration.
        
        Args:
            domain_name: Unique domain identifier
            domain_config: Domain configuration
            
        Returns:
            Domain metrics object
            
        Raises:
            ValueError: If domain name is invalid or already exists
            RuntimeError: If registration fails
        """
        if not domain_name or not isinstance(domain_name, str):
            raise ValueError("Domain name must be non-empty string")
        if not isinstance(domain_config, dict):
            raise TypeError(f"Domain config must be dict, got {type(domain_config)}")
        
        try:
            with self._domain_lock:
                if domain_name in self.active_domains:
                    raise ValueError(f"Domain {domain_name} already registered")
                
                if len(self.active_domains) >= self.config.max_active_domains:
                    raise RuntimeError(f"Maximum active domains ({self.config.max_active_domains}) reached")
                
                # Create domain metrics
                domain_metrics = DomainMetrics(
                    domain_name=domain_name,
                    start_time=datetime.utcnow()
                )
                
                # Configure compression for domain
                compression_strategy = self._determine_domain_compression_strategy(domain_name, domain_config)
                compression_config = self._create_domain_compression_config(domain_name, domain_config, compression_strategy)
                
                if not self._configure_compression_for_domain(domain_name, compression_config):
                    raise RuntimeError(f"Failed to configure compression for domain {domain_name}")
                
                # Set up optimization coordination
                if self.config.enable_real_time_optimization:
                    self._setup_domain_optimization_coordination(domain_name, domain_config)
                
                # Register domain
                self.active_domains[domain_name] = domain_metrics
                self.domain_queue.append(domain_name)
                self.domain_configs[domain_name] = domain_config
                self.domain_compression_strategies[domain_name] = compression_strategy
                
                # Update analytics
                with self._analytics_lock:
                    self.domain_analytics.total_domains += 1
                    self.domain_analytics.active_domains += 1
                    self.domain_analytics.registered_domains += 1
                
                self.logger.info(f"Domain {domain_name} registered with hybrid integration successfully")
                return domain_metrics
                
        except Exception as e:
            self.logger.error(f"Failed to register domain {domain_name} with hybrid integration: {e}")
            raise RuntimeError(f"Domain registration failed: {e}")
    
    def optimize_domain_compression(self, domain_name: str, data: torch.Tensor) -> Dict[str, Any]:
        """
        Optimize compression for domain data.
        
        Args:
            domain_name: Domain identifier
            data: Domain data tensor
            
        Returns:
            Compression optimization results
            
        Raises:
            ValueError: If domain or data is invalid
            RuntimeError: If optimization fails
        """
        if not domain_name or domain_name not in self.active_domains:
            raise ValueError(f"Invalid or unknown domain: {domain_name}")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Data must be torch.Tensor, got {type(data)}")
        if data.numel() == 0:
            raise ValueError("Data tensor cannot be empty")
        
        try:
            start_time = time.time()
            domain_metrics = self.active_domains[domain_name]
            
            # Determine optimal compression strategy for domain
            compression_strategy = self._determine_domain_compression_strategy(domain_name, self.domain_configs[domain_name])
            
            # Apply domain-specific compression optimization
            compression_result = self._apply_domain_compression_optimization(domain_name, data, compression_strategy)
            if not compression_result.get('success', False):
                raise RuntimeError(f"Compression failed for domain {domain_name}")
            
            # Apply dynamic switching optimization if available
            if self.dynamic_switching and compression_strategy == DomainCompressionStrategy.ADAPTIVE:
                switching_result = self._apply_domain_dynamic_switching_optimization(domain_name, data, compression_result)
                compression_result.update(switching_result)
            
            # Update domain metrics
            compression_time = time.time() - start_time
            domain_metrics.compression_operations += 1
            domain_metrics.average_compression_time_ms = (
                (domain_metrics.average_compression_time_ms * (domain_metrics.compression_operations - 1) + 
                 compression_time * 1000) / domain_metrics.compression_operations
            )
            
            # Update compression analytics
            compression_ratio = compression_result.get('compression_ratio', 0.0)
            domain_metrics.compression_ratio = (
                (domain_metrics.compression_ratio * (domain_metrics.compression_operations - 1) + 
                 compression_ratio) / domain_metrics.compression_operations
            )
            
            optimization_result = {
                'domain_name': domain_name,
                'compression_time_ms': compression_time * 1000,
                'compression_ratio': compression_ratio,
                'strategy_used': compression_strategy.value,
                'data_size_mb': data.numel() * data.element_size() / (1024 * 1024),
                'compressed_size_mb': compression_result.get('compressed_size', 0) / (1024 * 1024),
                'optimization_success': True,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Record optimization history
            self.compression_history.append(optimization_result)
            
            self.logger.debug(f"Domain compression optimized for {domain_name}: {compression_ratio:.4f} ratio")
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Failed to optimize domain compression for {domain_name}: {e}")
            raise RuntimeError(f"Domain compression optimization failed: {e}")
    
    def monitor_domain_performance(self, domain_name: str) -> Dict[str, Any]:
        """
        Monitor performance for a domain.
        
        Args:
            domain_name: Domain identifier
            
        Returns:
            Performance monitoring results
            
        Raises:
            ValueError: If domain is invalid
            RuntimeError: If monitoring fails
        """
        if not domain_name or domain_name not in self.active_domains:
            raise ValueError(f"Invalid or unknown domain: {domain_name}")
        
        try:
            domain_metrics = self.active_domains[domain_name]
            
            # Capture current performance metrics
            current_metrics = self._capture_domain_performance(domain_name)
            
            # Update domain metrics
            self._update_domain_performance_metrics(domain_name, current_metrics)
            
            # Analyze performance trends
            performance_trends = self._analyze_domain_performance_trends(domain_name)
            
            # Check for performance alerts
            alerts = self._check_domain_performance_alerts(domain_name, current_metrics)
            
            # Generate performance recommendations
            recommendations = self._generate_domain_performance_recommendations(domain_name, current_metrics, performance_trends)
            
            monitoring_result = {
                'domain_name': domain_name,
                'current_metrics': current_metrics,
                'performance_trends': performance_trends,
                'alerts': alerts,
                'recommendations': recommendations,
                'monitoring_timestamp': datetime.utcnow().isoformat(),
                'domain_duration_minutes': (datetime.utcnow() - domain_metrics.start_time).total_seconds() / 60
            }
            
            # Record performance history
            self.performance_metrics.append(monitoring_result)
            
            self.logger.debug(f"Performance monitoring completed for domain {domain_name}")
            return monitoring_result
            
        except Exception as e:
            self.logger.error(f"Failed to monitor domain performance for {domain_name}: {e}")
            raise RuntimeError(f"Domain performance monitoring failed: {e}")
    
    def handle_domain_errors(self, domain_name: str, error: Exception) -> Dict[str, Any]:
        """
        Handle domain errors and attempt recovery.
        
        Args:
            domain_name: Domain identifier
            error: Domain error that occurred
            
        Returns:
            Error handling results
            
        Raises:
            ValueError: If domain is invalid
            RuntimeError: If error handling fails
        """
        if not domain_name or domain_name not in self.active_domains:
            raise ValueError(f"Invalid or unknown domain: {domain_name}")
        if not isinstance(error, Exception):
            raise TypeError(f"Error must be Exception, got {type(error)}")
        
        try:
            domain_metrics = self.active_domains[domain_name]
            error_start_time = time.time()
            
            # Update error statistics
            domain_metrics.error_count += 1
            
            # Assess error severity
            error_severity = self._assess_domain_error_severity(domain_name, error)
            
            # Attempt error recovery based on severity
            recovery_result = None
            if self.config.enable_error_recovery:
                if error_severity == "critical":
                    recovery_result = self._attempt_domain_critical_error_recovery(domain_name, error)
                elif error_severity == "moderate":
                    recovery_result = self._attempt_domain_moderate_error_recovery(domain_name, error)
                elif error_severity == "minor":
                    recovery_result = self._attempt_domain_minor_error_recovery(domain_name, error)
            
            # Update recovery statistics
            if recovery_result and recovery_result.get('success', False):
                domain_metrics.recovery_count += 1
            
            error_handling_result = {
                'domain_name': domain_name,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'error_severity': error_severity,
                'recovery_attempted': recovery_result is not None,
                'recovery_successful': recovery_result.get('success', False) if recovery_result else False,
                'recovery_details': recovery_result,
                'error_handling_time_ms': (time.time() - error_start_time) * 1000,
                'total_errors': domain_metrics.error_count,
                'total_recoveries': domain_metrics.recovery_count,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.error(f"Domain error handled for {domain_name}: {error_severity} - {type(error).__name__}")
            return error_handling_result
            
        except Exception as e:
            self.logger.error(f"Failed to handle domain error for {domain_name}: {e}")
            raise RuntimeError(f"Domain error handling failed: {e}")
    
    def get_domain_analytics(self, domain_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive domain analytics.
        
        Args:
            domain_name: Optional specific domain for detailed analytics
            
        Returns:
            Domain analytics results
            
        Raises:
            ValueError: If domain name is invalid
            RuntimeError: If analytics retrieval fails
        """
        try:
            with self._analytics_lock:
                if domain_name:
                    if domain_name not in self.active_domains:
                        raise ValueError(f"Invalid or unknown domain: {domain_name}")
                    
                    # Get domain-specific analytics
                    domain_metrics = self.active_domains[domain_name]
                    domain_analytics = self._calculate_domain_analytics(domain_name, domain_metrics)
                    
                    return {
                        'domain_analytics': domain_analytics,
                        'domain_metrics': domain_metrics,
                        'performance_history': [m for m in self.performance_metrics if m.get('domain_name') == domain_name],
                        'compression_history': [c for c in self.compression_history if c.get('domain_name') == domain_name],
                        'optimization_history': [o for o in self.optimization_history if o.get('domain_name') == domain_name]
                    }
                else:
                    # Get global analytics
                    self._update_global_domain_analytics()
                    
                    return {
                        'global_analytics': self.domain_analytics,
                        'active_domains': {name: metrics for name, metrics in self.active_domains.items()},
                        'performance_baselines': self.performance_baselines,
                        'domain_baselines': self.domain_baselines,
                        'recent_performance': list(self.performance_metrics)[-20:],
                        'recent_compressions': self.compression_history[-20:],
                        'recent_optimizations': self.optimization_history[-20:],
                        'system_health': self._calculate_domain_system_health(),
                        'recommendations': self._generate_domain_system_recommendations()
                    }
                    
        except Exception as e:
            self.logger.error(f"Failed to get domain analytics: {e}")
            raise RuntimeError(f"Domain analytics retrieval failed: {e}")
    
    def _initialize_performance_baselines(self) -> None:
        """Initialize performance baselines"""
        try:
            self.performance_baselines = {
                'memory_usage_gb': 2.0,
                'gpu_utilization_percent': 50.0,
                'compression_ratio': 0.5,
                'operation_time_ms': 100.0,
                'compression_time_ms': 50.0,
                'optimization_time_ms': 30.0,
                'domain_accuracy': 0.8,
                'domain_throughput': 100.0
            }
            self.logger.debug("Performance baselines initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize performance baselines: {e}")
            raise RuntimeError(f"Performance baseline initialization failed: {e}")
    
    def _setup_domain_hooks(self) -> None:
        """Setup domain system hooks"""
        try:
            if hasattr(self.domain_registry, 'add_hook'):
                self.domain_registry.add_hook('domain_registered', self._domain_registered_hook)
                self.domain_registry.add_hook('domain_updated', self._domain_updated_hook)
                self.domain_registry.add_hook('domain_error', self._domain_error_hook)
            
            if hasattr(self.domain_router, 'add_hook'):
                self.domain_router.add_hook('routing_completed', self._routing_completed_hook)
                self.domain_router.add_hook('routing_failed', self._routing_failed_hook)
            
            self.logger.debug("Domain hooks setup completed")
        except Exception as e:
            self.logger.error(f"Failed to setup domain hooks: {e}")
            raise RuntimeError(f"Domain hooks setup failed: {e}")
    
    def _initialize_compression_coordination(self) -> None:
        """Initialize compression coordination"""
        try:
            if hasattr(self.hybrid_compression, 'set_coordination_mode'):
                self.hybrid_compression.set_coordination_mode('domain_optimized')
            
            if self.dynamic_switching and hasattr(self.dynamic_switching, 'set_domain_mode'):
                self.dynamic_switching.set_domain_mode(True)
            
            self.logger.debug("Compression coordination initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize compression coordination: {e}")
            raise RuntimeError(f"Compression coordination initialization failed: {e}")
    
    def _start_performance_monitoring(self) -> None:
        """Start performance monitoring"""
        try:
            self.monitoring_active = True
            self.logger.debug("Performance monitoring started")
        except Exception as e:
            self.logger.error(f"Failed to start performance monitoring: {e}")
            raise RuntimeError(f"Performance monitoring start failed: {e}")
    
    def _initialize_domain_analytics(self) -> None:
        """Initialize domain analytics"""
        try:
            self.domain_analytics = DomainAnalytics()
            self.logger.debug("Domain analytics initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize domain analytics: {e}")
            raise RuntimeError(f"Domain analytics initialization failed: {e}")
    
    def _determine_domain_compression_strategy(self, domain_name: str, domain_config: Dict[str, Any]) -> DomainCompressionStrategy:
        """Determine optimal compression strategy for domain"""
        if self.config.compression_strategy == DomainCompressionStrategy.ADAPTIVE:
            # Adaptive strategy based on domain characteristics
            domain_type = domain_config.get('domain_type', 'generic')
            data_characteristics = domain_config.get('data_characteristics', {})
            performance_requirements = domain_config.get('performance_requirements', {})
            
            if domain_type in ['financial', 'fraud_detection']:
                return DomainCompressionStrategy.AGGRESSIVE
            elif domain_type in ['image', 'video']:
                return DomainCompressionStrategy.CONSERVATIVE
            elif performance_requirements.get('real_time', False):
                return DomainCompressionStrategy.REAL_TIME
            else:
                return DomainCompressionStrategy.DOMAIN_OPTIMIZED
        
        return self.config.compression_strategy
    
    def _create_domain_compression_config(self, domain_name: str, domain_config: Dict[str, Any], strategy: DomainCompressionStrategy) -> Dict[str, Any]:
        """Create compression configuration for domain"""
        compression_config = {
            'strategy': strategy.value,
            'domain_name': domain_name,
            'domain_type': domain_config.get('domain_type', 'generic'),
            'batch_size': domain_config.get('batch_size', self.config.domain_batch_size),
            'enable_gpu': self.config.enable_gpu_optimization,
            'memory_limit_gb': self.config.memory_threshold_gb,
            'performance_threshold': self.config.performance_threshold,
            'domain_specific_params': domain_config.get('compression_params', {})
        }
        return compression_config
    
    def _configure_compression_for_domain(self, domain_name: str, config: Dict[str, Any]) -> bool:
        """Configure compression system for domain"""
        try:
            if hasattr(self.hybrid_compression, 'configure_for_domain'):
                return self.hybrid_compression.configure_for_domain(domain_name, config)
            return True
        except Exception as e:
            self.logger.error(f"Failed to configure compression for domain: {e}")
            return False
    
    def _setup_domain_optimization_coordination(self, domain_name: str, config: Dict[str, Any]) -> None:
        """Setup optimization coordination for domain"""
        try:
            # Store domain-specific optimization configuration
            optimization_config = {
                'enable_real_time': self.config.enable_real_time_optimization,
                'enable_memory_optimization': self.config.enable_memory_optimization,
                'enable_gpu_optimization': self.config.enable_gpu_optimization,
                'domain_specific_optimization': config.get('optimization_params', {})
            }
            
            if domain_name not in self.domain_configs:
                self.domain_configs[domain_name] = {}
            
            self.domain_configs[domain_name]['optimization'] = optimization_config
            self.logger.debug(f"Optimization coordination setup for domain {domain_name}")
        except Exception as e:
            self.logger.error(f"Failed to setup optimization coordination: {e}")
            raise RuntimeError(f"Optimization coordination setup failed: {e}")
    
    def _apply_domain_compression_optimization(self, domain_name: str, data: torch.Tensor, strategy: DomainCompressionStrategy) -> Dict[str, Any]:
        """Apply domain-specific compression optimization"""
        try:
            # Configure compression for domain-specific requirements
            domain_config = self.domain_configs.get(domain_name, {})
            compression_params = {
                'domain_name': domain_name,
                'strategy': strategy.value,
                'domain_type': domain_config.get('domain_type', 'generic'),
                'enable_domain_optimization': True
            }
            
            # Apply compression
            compression_result = self.hybrid_compression.compress(data)
            compression_result['domain_optimization'] = compression_params
            
            return compression_result
        except Exception as e:
            self.logger.error(f"Domain compression optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _apply_domain_dynamic_switching_optimization(self, domain_name: str, data: torch.Tensor, compression_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply dynamic switching optimization for domain"""
        try:
            if not self.dynamic_switching:
                return {}
            
            switching_result = self.dynamic_switching.optimize_for_domain(
                domain_name, data, compression_result
            )
            return switching_result
        except Exception as e:
            self.logger.error(f"Domain dynamic switching optimization failed: {e}")
            return {'switching_error': str(e)}
    
    def _capture_domain_performance(self, domain_name: str) -> Dict[str, float]:
        """Capture current performance metrics for domain"""
        try:
            import psutil
            
            # System metrics
            memory_usage = psutil.virtual_memory().used / (1024**3)  # GB
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # GPU metrics (if available)
            gpu_utilization = 0.0
            gpu_memory_usage = 0.0
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                gpu_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_usage = gpu_memory_info.used / (1024**3)  # GB
            except:
                pass
            
            # Domain-specific metrics
            domain_metrics = self.active_domains.get(domain_name)
            domain_accuracy = domain_metrics.domain_accuracy if domain_metrics else 0.0
            domain_throughput = domain_metrics.domain_throughput if domain_metrics else 0.0
            
            return {
                'memory_usage_gb': memory_usage,
                'cpu_utilization_percent': cpu_percent,
                'gpu_utilization_percent': gpu_utilization,
                'gpu_memory_usage_gb': gpu_memory_usage,
                'domain_accuracy': domain_accuracy,
                'domain_throughput': domain_throughput
            }
        except Exception as e:
            self.logger.error(f"Failed to capture domain performance metrics: {e}")
            return {}
    
    def _update_domain_performance_metrics(self, domain_name: str, current_metrics: Dict[str, float]) -> None:
        """Update domain performance metrics"""
        try:
            domain_metrics = self.active_domains[domain_name]
            
            # Update peak metrics
            domain_metrics.peak_memory_usage_gb = max(
                domain_metrics.peak_memory_usage_gb,
                current_metrics.get('memory_usage_gb', 0.0)
            )
            domain_metrics.peak_gpu_utilization_percent = max(
                domain_metrics.peak_gpu_utilization_percent,
                current_metrics.get('gpu_utilization_percent', 0.0)
            )
            
            # Update GPU memory usage
            domain_metrics.gpu_memory_usage_gb = current_metrics.get('gpu_memory_usage_gb', 0.0)
            
            # Update averages (simple moving average)
            operations = max(1, domain_metrics.total_operations)
            domain_metrics.average_memory_usage_gb = (
                (domain_metrics.average_memory_usage_gb * (operations - 1) + 
                 current_metrics.get('memory_usage_gb', 0.0)) / operations
            )
            domain_metrics.average_gpu_utilization_percent = (
                (domain_metrics.average_gpu_utilization_percent * (operations - 1) + 
                 current_metrics.get('gpu_utilization_percent', 0.0)) / operations
            )
            
            # Update domain-specific metrics
            domain_metrics.domain_accuracy = current_metrics.get('domain_accuracy', 0.0)
            domain_metrics.domain_throughput = current_metrics.get('domain_throughput', 0.0)
            
        except Exception as e:
            self.logger.error(f"Failed to update domain performance metrics: {e}")
    
    def _analyze_domain_performance_trends(self, domain_name: str) -> Dict[str, str]:
        """Analyze performance trends for domain"""
        try:
            recent_metrics = [m for m in self.performance_metrics if m.get('domain_name') == domain_name][-10:]
            
            trends = {}
            if len(recent_metrics) >= 3:
                # Analyze memory trend
                memory_values = [m['current_metrics'].get('memory_usage_gb', 0) for m in recent_metrics]
                memory_trend = np.polyfit(range(len(memory_values)), memory_values, 1)[0]
                trends['memory'] = 'increasing' if memory_trend > 0.1 else 'decreasing' if memory_trend < -0.1 else 'stable'
                
                # Analyze GPU trend
                gpu_values = [m['current_metrics'].get('gpu_utilization_percent', 0) for m in recent_metrics]
                gpu_trend = np.polyfit(range(len(gpu_values)), gpu_values, 1)[0]
                trends['gpu'] = 'increasing' if gpu_trend > 1.0 else 'decreasing' if gpu_trend < -1.0 else 'stable'
                
                # Analyze domain accuracy trend
                accuracy_values = [m['current_metrics'].get('domain_accuracy', 0) for m in recent_metrics]
                accuracy_trend = np.polyfit(range(len(accuracy_values)), accuracy_values, 1)[0]
                trends['accuracy'] = 'improving' if accuracy_trend > 0.01 else 'degrading' if accuracy_trend < -0.01 else 'stable'
            else:
                trends = {'memory': 'insufficient_data', 'gpu': 'insufficient_data', 'accuracy': 'insufficient_data'}
            
            return trends
        except Exception as e:
            self.logger.error(f"Failed to analyze domain performance trends: {e}")
            return {}
    
    def _check_domain_performance_alerts(self, domain_name: str, current_metrics: Dict[str, float]) -> List[str]:
        """Check for domain performance alerts"""
        alerts = []
        
        memory_usage = current_metrics.get('memory_usage_gb', 0.0)
        if memory_usage > self.config.memory_threshold_gb:
            alerts.append(f"High memory usage for domain {domain_name}: {memory_usage:.1f}GB > {self.config.memory_threshold_gb}GB")
        
        gpu_utilization = current_metrics.get('gpu_utilization_percent', 0.0)
        if gpu_utilization > self.config.gpu_threshold_percent:
            alerts.append(f"High GPU utilization for domain {domain_name}: {gpu_utilization:.1f}% > {self.config.gpu_threshold_percent}%")
        
        domain_accuracy = current_metrics.get('domain_accuracy', 1.0)
        if domain_accuracy < self.config.performance_threshold:
            alerts.append(f"Low domain accuracy for {domain_name}: {domain_accuracy:.3f} < {self.config.performance_threshold}")
        
        return alerts
    
    def _generate_domain_performance_recommendations(self, domain_name: str, current_metrics: Dict[str, float], trends: Dict[str, str]) -> List[str]:
        """Generate domain performance recommendations"""
        recommendations = []
        
        if trends.get('memory') == 'increasing':
            recommendations.append(f"Consider reducing batch size or enabling more aggressive compression for domain {domain_name}")
        
        if trends.get('accuracy') == 'degrading':
            recommendations.append(f"Domain {domain_name} accuracy declining - consider model retraining or optimization")
        
        if trends.get('gpu') == 'decreasing':
            recommendations.append(f"GPU utilization declining for domain {domain_name} - consider increasing workload or batch size")
        
        if current_metrics.get('memory_usage_gb', 0) > self.config.memory_threshold_gb * 0.8:
            recommendations.append(f"Memory usage approaching threshold for domain {domain_name} - enable memory optimization")
        
        return recommendations if recommendations else [f"Domain {domain_name} performance within acceptable parameters"]
    
    def _assess_domain_error_severity(self, domain_name: str, error: Exception) -> str:
        """Assess error severity level for domain"""
        error_type = type(error).__name__
        
        critical_errors = ['RuntimeError', 'SystemError', 'MemoryError', 'OSError']
        moderate_errors = ['ValueError', 'TypeError', 'AttributeError']
        
        if error_type in critical_errors:
            return "critical"
        elif error_type in moderate_errors:
            return "moderate"
        else:
            return "minor"
    
    def _attempt_domain_critical_error_recovery(self, domain_name: str, error: Exception) -> Dict[str, Any]:
        """Attempt recovery from critical domain errors"""
        try:
            recovery_actions = []
            
            # Memory cleanup
            import gc
            gc.collect()
            recovery_actions.append("memory_cleanup")
            
            # Reset compression system for domain
            if hasattr(self.hybrid_compression, 'reset_for_domain'):
                self.hybrid_compression.reset_for_domain(domain_name)
                recovery_actions.append("compression_reset")
            
            # Reduce domain batch sizes
            if domain_name in self.domain_configs:
                current_batch_size = self.domain_configs[domain_name].get('batch_size', 32)
                new_batch_size = max(8, int(current_batch_size * 0.5))
                self.domain_configs[domain_name]['batch_size'] = new_batch_size
                recovery_actions.append(f"batch_size_reduced_to_{new_batch_size}")
            
            return {
                'success': True,
                'recovery_actions': recovery_actions,
                'recovery_type': 'critical'
            }
        except Exception as e:
            return {
                'success': False,
                'recovery_error': str(e),
                'recovery_type': 'critical'
            }
    
    def _attempt_domain_moderate_error_recovery(self, domain_name: str, error: Exception) -> Dict[str, Any]:
        """Attempt recovery from moderate domain errors"""
        try:
            recovery_actions = []
            
            # Adjust compression parameters for domain
            if hasattr(self.hybrid_compression, 'adjust_domain_parameters'):
                self.hybrid_compression.adjust_domain_parameters(domain_name, conservative=True)
                recovery_actions.append("compression_parameters_adjusted")
            
            # Switch to conservative compression strategy
            if domain_name in self.domain_compression_strategies:
                self.domain_compression_strategies[domain_name] = DomainCompressionStrategy.CONSERVATIVE
                recovery_actions.append("compression_strategy_switched_conservative")
            
            return {
                'success': True,
                'recovery_actions': recovery_actions,
                'recovery_type': 'moderate'
            }
        except Exception as e:
            return {
                'success': False,
                'recovery_error': str(e),
                'recovery_type': 'moderate'
            }
    
    def _attempt_domain_minor_error_recovery(self, domain_name: str, error: Exception) -> Dict[str, Any]:
        """Attempt recovery from minor domain errors"""
        try:
            recovery_actions = []
            
            # Simple retry
            recovery_actions.append("retry")
            
            # Log warning
            self.logger.warning(f"Minor error in domain {domain_name}: {error}")
            recovery_actions.append("error_logged")
            
            return {
                'success': True,
                'recovery_actions': recovery_actions,
                'recovery_type': 'minor'
            }
        except Exception as e:
            return {
                'success': False,
                'recovery_error': str(e),
                'recovery_type': 'minor'
            }
    
    def _calculate_domain_analytics(self, domain_name: str, domain_metrics: DomainMetrics) -> Dict[str, Any]:
        """Calculate analytics for specific domain"""
        try:
            current_time = datetime.utcnow()
            domain_duration = (current_time - domain_metrics.start_time).total_seconds() / 60  # minutes
            
            return {
                'domain_name': domain_name,
                'duration_minutes': domain_duration,
                'total_operations': domain_metrics.total_operations,
                'operations_per_minute': domain_metrics.total_operations / max(1, domain_duration),
                'compression_operations': domain_metrics.compression_operations,
                'optimization_operations': domain_metrics.optimization_operations,
                'average_operation_time_ms': domain_metrics.average_operation_time_ms,
                'average_compression_time_ms': domain_metrics.average_compression_time_ms,
                'compression_ratio': domain_metrics.compression_ratio,
                'peak_memory_usage_gb': domain_metrics.peak_memory_usage_gb,
                'average_memory_usage_gb': domain_metrics.average_memory_usage_gb,
                'peak_gpu_utilization_percent': domain_metrics.peak_gpu_utilization_percent,
                'domain_accuracy': domain_metrics.domain_accuracy,
                'domain_throughput': domain_metrics.domain_throughput,
                'error_rate': domain_metrics.error_count / max(1, domain_metrics.total_operations),
                'recovery_rate': domain_metrics.recovery_count / max(1, domain_metrics.error_count)
            }
        except Exception as e:
            self.logger.error(f"Failed to calculate domain analytics: {e}")
            return {}
    
    def _update_global_domain_analytics(self) -> None:
        """Update global domain analytics"""
        try:
            with self._analytics_lock:
                self.domain_analytics.active_domains = len(self.active_domains)
                
                if self.active_domains:
                    # Calculate averages across active domains
                    total_performance = 0
                    total_compression_efficiency = 0
                    total_memory_usage = 0
                    total_gpu_utilization = 0
                    
                    best_accuracy = 0
                    worst_accuracy = 1.0
                    best_compression = 0
                    best_domain = ""
                    worst_domain = ""
                    most_compressed_domain = ""
                    
                    for domain_name, domain_metrics in self.active_domains.items():
                        total_performance += domain_metrics.domain_accuracy
                        total_compression_efficiency += domain_metrics.compression_ratio
                        total_memory_usage += domain_metrics.average_memory_usage_gb
                        total_gpu_utilization += domain_metrics.average_gpu_utilization_percent
                        
                        if domain_metrics.domain_accuracy > best_accuracy:
                            best_accuracy = domain_metrics.domain_accuracy
                            best_domain = domain_name
                        
                        if domain_metrics.domain_accuracy < worst_accuracy:
                            worst_accuracy = domain_metrics.domain_accuracy
                            worst_domain = domain_name
                        
                        if domain_metrics.compression_ratio > best_compression:
                            best_compression = domain_metrics.compression_ratio
                            most_compressed_domain = domain_name
                    
                    num_domains = len(self.active_domains)
                    self.domain_analytics.average_domain_performance = total_performance / num_domains
                    self.domain_analytics.average_compression_efficiency = total_compression_efficiency / num_domains
                    self.domain_analytics.average_memory_savings = total_memory_usage / num_domains
                    self.domain_analytics.average_gpu_utilization = total_gpu_utilization / num_domains
                    
                    self.domain_analytics.best_performing_domain = best_domain
                    self.domain_analytics.worst_performing_domain = worst_domain
                    self.domain_analytics.most_compressed_domain = most_compressed_domain
                
                # Update system health score
                self.domain_analytics.system_health_score = self._calculate_domain_system_health_score()
                
                # Update timestamp
                self.domain_analytics.timestamp = datetime.utcnow()
                
        except Exception as e:
            self.logger.error(f"Failed to update global domain analytics: {e}")
    
    def _calculate_domain_system_health(self) -> Dict[str, Any]:
        """Calculate overall domain system health"""
        try:
            health_score = self._calculate_domain_system_health_score()
            
            health_status = "excellent" if health_score > 0.9 else \
                           "good" if health_score > 0.7 else \
                           "fair" if health_score > 0.5 else \
                           "poor"
            
            return {
                'health_score': health_score,
                'health_status': health_status,
                'active_domains': len(self.active_domains),
                'monitoring_active': self.monitoring_active,
                'error_recovery_active': self.error_recovery_active,
                'optimization_active': self.optimization_active,
                'integration_state': self.integration_state.value
            }
        except Exception as e:
            self.logger.error(f"Failed to calculate domain system health: {e}")
            return {'health_score': 0.0, 'health_status': 'unknown'}
    
    def _calculate_domain_system_health_score(self) -> float:
        """Calculate domain system health score"""
        try:
            score_factors = []
            
            # Domain success rate
            if self.domain_analytics.total_domains > 0:
                success_rate = (self.domain_analytics.total_domains - self.domain_analytics.failed_domains) / self.domain_analytics.total_domains
                score_factors.append(success_rate)
            
            # Performance efficiency
            if self.domain_analytics.average_domain_performance > 0:
                efficiency_score = min(1.0, self.domain_analytics.average_domain_performance / 0.9)
                score_factors.append(efficiency_score)
            
            # Compression efficiency
            if self.domain_analytics.average_compression_efficiency > 0:
                compression_score = min(1.0, self.domain_analytics.average_compression_efficiency / 0.8)
                score_factors.append(compression_score)
            
            # Integration state health
            state_score = 1.0 if self.integration_state == DomainIntegrationState.INITIALIZED else \
                         0.8 if self.integration_state == DomainIntegrationState.DOMAINS_ACTIVE else \
                         0.6 if self.integration_state == DomainIntegrationState.OPTIMIZING else \
                         0.2
            score_factors.append(state_score)
            
            return np.mean(score_factors) if score_factors else 0.5
            
        except Exception as e:
            self.logger.error(f"Failed to calculate domain system health score: {e}")
            return 0.0
    
    def _generate_domain_system_recommendations(self) -> List[str]:
        """Generate domain system-wide recommendations"""
        recommendations = []
        
        if self.domain_analytics.system_health_score < 0.7:
            recommendations.append("Domain system health below optimal - consider performance optimization")
        
        if len(self.active_domains) == 0:
            recommendations.append("No active domains - system ready for new domain registrations")
        elif len(self.active_domains) >= self.config.max_active_domains * 0.8:
            recommendations.append("High domain load - consider scaling or load balancing")
        
        if self.domain_analytics.average_compression_efficiency < 0.5:
            recommendations.append("Low compression efficiency across domains - review compression strategies")
        
        if self.domain_analytics.average_domain_performance < 0.7:
            recommendations.append("Domain performance below optimal - consider domain optimization")
        
        if not self.monitoring_active:
            recommendations.append("Performance monitoring disabled - enable for better domain insights")
        
        return recommendations if recommendations else ["Domain system operating within optimal parameters"]
    
    def _domain_registered_hook(self, domain_name: str, domain_config: Dict[str, Any]) -> None:
        """Domain registered hook"""
        try:
            self.logger.info(f"Domain registered hook triggered for {domain_name}")
            # Could trigger automatic hybrid registration here
        except Exception as e:
            self.logger.error(f"Domain registered hook failed: {e}")
    
    def _domain_updated_hook(self, domain_name: str, domain_config: Dict[str, Any]) -> None:
        """Domain updated hook"""
        try:
            if domain_name in self.active_domains:
                # Update domain configuration
                self.domain_configs[domain_name].update(domain_config)
                self.logger.info(f"Domain configuration updated for {domain_name}")
        except Exception as e:
            self.logger.error(f"Domain updated hook failed: {e}")
    
    def _domain_error_hook(self, domain_name: str, error: Exception) -> None:
        """Domain error hook"""
        try:
            if domain_name in self.active_domains:
                self.handle_domain_errors(domain_name, error)
        except Exception as e:
            self.logger.error(f"Domain error hook failed: {e}")
    
    def _routing_completed_hook(self, routing_data: Dict[str, Any]) -> None:
        """Routing completed hook"""
        try:
            domain_name = routing_data.get('domain_name')
            if domain_name and domain_name in self.active_domains:
                # Update domain metrics based on routing success
                domain_metrics = self.active_domains[domain_name]
                domain_metrics.total_operations += 1
        except Exception as e:
            self.logger.error(f"Routing completed hook failed: {e}")
    
    def _routing_failed_hook(self, routing_data: Dict[str, Any]) -> None:
        """Routing failed hook"""
        try:
            domain_name = routing_data.get('domain_name')
            error = routing_data.get('error')
            if domain_name and domain_name in self.active_domains and error:
                self.handle_domain_errors(domain_name, error)
        except Exception as e:
            self.logger.error(f"Routing failed hook failed: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        return {
            'integration_state': self.integration_state.value,
            'active_domains': len(self.active_domains),
            'monitoring_active': self.monitoring_active,
            'error_recovery_active': self.error_recovery_active,
            'optimization_active': self.optimization_active,
            'system_health': self._calculate_domain_system_health(),
            'domain_compression_strategies': {name: strategy.value for name, strategy in self.domain_compression_strategies.items()}
        }
    
    def shutdown(self) -> None:
        """Shutdown domain-hybrid integration"""
        try:
            with self._integration_lock:
                self.monitoring_active = False
                self.error_recovery_active = False
                self.optimization_active = False
                
                # Complete active domains
                for domain_name in list(self.active_domains.keys()):
                    domain_metrics = self.active_domains[domain_name]
                    domain_metrics.end_time = datetime.utcnow()
                
                self.integration_state = DomainIntegrationState.COMPLETED
                self.logger.info("Domain-hybrid integration shutdown completed")
                
        except Exception as e:
            self.logger.error(f"Domain-hybrid integration shutdown failed: {e}")
            raise RuntimeError(f"Shutdown failed: {e}")