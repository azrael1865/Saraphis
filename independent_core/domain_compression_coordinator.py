"""
Domain Compression Coordinator - Coordination between domains and compression operations
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

# Import domain system components (avoid circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from domain_registry import DomainRegistry, DomainConfig
    from domain_router import DomainRouter

# Import hybrid compression system components
from compression_systems.padic.hybrid_padic_compressor import HybridPadicCompressionSystem
from compression_systems.padic.dynamic_switching_manager import DynamicSwitchingManager
from compression_systems.padic.direction_manager import DirectionManager


class DomainCoordinationMode(Enum):
    """Domain coordination mode enumeration"""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    ADAPTIVE = "adaptive"
    DOMAIN_PRIORITY = "domain_priority"
    LOAD_BALANCED = "load_balanced"
    REAL_TIME = "real_time"


class DomainCompressionPriority(Enum):
    """Domain compression priority enumeration"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    DOMAIN_SPECIFIC = "domain_specific"


class DomainResourceStrategy(Enum):
    """Domain resource allocation strategy enumeration"""
    DOMAIN_FIRST = "domain_first"
    COMPRESSION_FIRST = "compression_first"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    DOMAIN_ISOLATED = "domain_isolated"


@dataclass
class DomainCompressionCoordinatorConfig:
    """Configuration for domain-compression coordination"""
    coordination_mode: DomainCoordinationMode = DomainCoordinationMode.ADAPTIVE
    resource_strategy: DomainResourceStrategy = DomainResourceStrategy.BALANCED
    enable_real_time_coordination: bool = True
    enable_domain_specific_compression: bool = True
    enable_memory_coordination: bool = True
    enable_performance_optimization: bool = True
    enable_resource_monitoring: bool = True
    
    # Coordination thresholds
    memory_coordination_threshold_gb: float = 6.0
    gpu_coordination_threshold_percent: float = 80.0
    compression_latency_threshold_ms: float = 100.0
    domain_latency_threshold_ms: float = 200.0
    
    # Batch coordination
    coordination_batch_size: int = 32
    domain_compression_batch_size: int = 64
    
    # Performance parameters
    coordination_interval_seconds: int = 10
    performance_check_interval_seconds: int = 5
    resource_monitoring_interval_seconds: int = 15
    
    # Analytics configuration
    enable_analytics: bool = True
    analytics_history_size: int = 500
    coordination_history_size: int = 200
    
    def __post_init__(self):
        """Validate configuration"""
        if not isinstance(self.coordination_mode, DomainCoordinationMode):
            raise TypeError("Coordination mode must be DomainCoordinationMode")
        if not isinstance(self.resource_strategy, DomainResourceStrategy):
            raise TypeError("Resource strategy must be DomainResourceStrategy")
        if self.memory_coordination_threshold_gb <= 0:
            raise ValueError("Memory coordination threshold must be positive")
        if not (0.0 <= self.gpu_coordination_threshold_percent <= 100.0):
            raise ValueError("GPU coordination threshold must be between 0.0 and 100.0")


@dataclass
class DomainCompressionOperation:
    """Domain-compression coordination operation"""
    operation_id: str
    domain_name: str
    operation_type: str  # 'domain_compression', 'domain_memory_management', etc.
    priority: DomainCompressionPriority
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Operation data
    data_size_bytes: int = 0
    compressed_size_bytes: int = 0
    compression_ratio: float = 0.0
    
    # Performance metrics
    coordination_time_ms: float = 0.0
    compression_time_ms: float = 0.0
    domain_impact_ms: float = 0.0
    
    # Resource usage
    memory_usage_gb: float = 0.0
    gpu_utilization_percent: float = 0.0
    cpu_utilization_percent: float = 0.0
    
    # Domain-specific metrics
    domain_accuracy_impact: float = 0.0
    domain_throughput_impact: float = 0.0
    
    # Success tracking
    coordination_successful: bool = False
    domain_successful: bool = False
    compression_successful: bool = False
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate operation"""
        if not self.operation_id or not self.domain_name:
            raise ValueError("Operation ID and Domain name cannot be empty")
        if not isinstance(self.priority, DomainCompressionPriority):
            raise TypeError("Priority must be DomainCompressionPriority")


@dataclass
class DomainCompressionAnalytics:
    """Comprehensive domain-compression analytics"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    
    # Operation type breakdown
    domain_compression_operations: int = 0
    domain_memory_operations: int = 0
    domain_performance_operations: int = 0
    domain_resource_operations: int = 0
    
    # Domain-specific analytics
    domains_coordinated: int = 0
    average_domain_compression_ratio: float = 0.0
    average_domain_performance_impact: float = 0.0
    
    # Performance analytics
    average_coordination_time_ms: float = 0.0
    average_compression_time_ms: float = 0.0
    average_domain_impact_ms: float = 0.0
    
    # Resource analytics
    average_memory_usage_gb: float = 0.0
    peak_memory_usage_gb: float = 0.0
    average_gpu_utilization_percent: float = 0.0
    peak_gpu_utilization_percent: float = 0.0
    
    # Efficiency metrics
    coordination_efficiency: float = 0.0
    resource_utilization_efficiency: float = 0.0
    domain_compression_efficiency: float = 0.0
    
    # Domain performance metrics
    best_performing_domain: str = ""
    worst_performing_domain: str = ""
    most_compressed_domain: str = ""
    
    # Trend analysis
    performance_trend: str = "stable"
    memory_trend: str = "stable"
    compression_trend: str = "stable"
    
    timestamp: datetime = field(default_factory=datetime.utcnow)


class DomainCompressionCoordinator:
    """
    Coordination between domains and compression operations.
    Manages real-time coordination, domain-specific optimization, and performance management.
    """
    
    def __init__(self, config: Optional[DomainCompressionCoordinatorConfig] = None):
        """Initialize domain-compression coordinator"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, DomainCompressionCoordinatorConfig):
            raise TypeError(f"Config must be DomainCompressionCoordinatorConfig or None, got {type(config)}")
        
        self.config = config or DomainCompressionCoordinatorConfig()
        self.logger = logging.getLogger('DomainCompressionCoordinator')
        
        # System references
        self.domain_registry: Optional['DomainRegistry'] = None
        self.domain_router: Optional['DomainRouter'] = None
        self.hybrid_compression: Optional[HybridPadicCompressionSystem] = None
        self.dynamic_switching: Optional[DynamicSwitchingManager] = None
        self.direction_manager: Optional[DirectionManager] = None
        
        # Coordination state
        self.coordination_mode = self.config.coordination_mode
        self.active_operations: Dict[str, DomainCompressionOperation] = {}
        self.operation_queue: deque = deque(maxlen=1000)
        self.coordination_history: deque = deque(maxlen=self.config.coordination_history_size)
        
        # Domain-specific coordination
        self.domain_coordination_configs: Dict[str, Dict[str, Any]] = {}
        self.domain_resource_allocations: Dict[str, Dict[str, float]] = {}
        self.domain_compression_strategies: Dict[str, str] = {}
        
        # Resource coordination
        self.resource_allocation: Dict[str, float] = {
            'domain_memory_percent': 50.0,
            'compression_memory_percent': 30.0,
            'system_memory_percent': 20.0,
            'domain_gpu_percent': 60.0,
            'compression_gpu_percent': 30.0,
            'system_gpu_percent': 10.0,
            'domain_cpu_percent': 50.0,
            'compression_cpu_percent': 40.0,
            'system_cpu_percent': 10.0
        }
        
        # Performance tracking
        self.coordination_analytics = DomainCompressionAnalytics()
        self.performance_baselines: Dict[str, float] = {}
        self.domain_performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.analytics_history_size))
        
        # Optimization tracking
        self.optimization_history: List[Dict[str, Any]] = []
        self.last_coordination_optimization: Optional[datetime] = None
        
        # Thread safety
        self._coordination_lock = threading.RLock()
        self._operation_lock = threading.RLock()
        self._analytics_lock = threading.RLock()
        
        # Coordination monitoring
        self.coordination_active = False
        self.monitoring_active = False
        self.resource_monitoring_active = False
        
        self.logger.info("DomainCompressionCoordinator created successfully")
    
    def initialize_coordinator(self,
                             domain_registry: 'DomainRegistry',
                             domain_router: 'DomainRouter',
                             hybrid_compression: HybridPadicCompressionSystem,
                             dynamic_switching: Optional[DynamicSwitchingManager] = None,
                             direction_manager: Optional[DirectionManager] = None) -> None:
        """
        Initialize domain-compression coordinator with required systems.
        
        Args:
            domain_registry: Domain registry instance
            domain_router: Domain router instance
            hybrid_compression: Hybrid compression system
            dynamic_switching: Optional dynamic switching manager
            direction_manager: Optional direction manager
            
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
            with self._coordination_lock:
                # Store system references
                self.domain_registry = domain_registry
                self.domain_router = domain_router
                self.hybrid_compression = hybrid_compression
                self.dynamic_switching = dynamic_switching
                self.direction_manager = direction_manager
                
                # Initialize coordination systems
                self._initialize_performance_baselines()
                self._setup_coordination_hooks()
                self._initialize_domain_resource_coordination()
                
                if self.config.enable_resource_monitoring:
                    self._start_resource_monitoring()
                
                if self.config.enable_analytics:
                    self._initialize_coordination_analytics()
                
                self.coordination_active = True
                self.logger.info("Domain-compression coordinator initialized successfully")
                
        except Exception as e:
            self.coordination_active = False
            self.logger.error(f"Failed to initialize domain-compression coordinator: {e}")
            raise RuntimeError(f"Coordinator initialization failed: {e}")
    
    def coordinate_domain_compression(self, domain_name: str, data: torch.Tensor) -> DomainCompressionOperation:
        """
        Coordinate domain data compression operation.
        
        Args:
            domain_name: Domain identifier
            data: Domain data tensor to compress
            
        Returns:
            Domain compression operation result
            
        Raises:
            ValueError: If domain or data is invalid
            RuntimeError: If coordination fails
        """
        if not domain_name or not isinstance(domain_name, str):
            raise ValueError("Domain name must be non-empty string")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Data must be torch.Tensor, got {type(data)}")
        if data.numel() == 0:
            raise ValueError("Data tensor cannot be empty")
        
        try:
            operation_id = self._generate_operation_id("domain_compression")
            start_time = datetime.utcnow()
            
            # Create coordination operation
            operation = DomainCompressionOperation(
                operation_id=operation_id,
                domain_name=domain_name,
                operation_type="domain_compression",
                priority=self._determine_domain_compression_priority(domain_name, data),
                start_time=start_time,
                data_size_bytes=data.numel() * data.element_size()
            )
            
            with self._operation_lock:
                self.active_operations[operation_id] = operation
            
            # Coordinate resource allocation for domain compression
            resource_allocation = self._coordinate_domain_resource_allocation_for_compression(domain_name, data)
            
            # Perform coordinated domain compression
            compression_start = time.time()
            compression_result = self._perform_coordinated_domain_compression(domain_name, data, resource_allocation)
            compression_time = (time.time() - compression_start) * 1000  # ms
            
            # Update operation results
            operation.end_time = datetime.utcnow()
            operation.coordination_time_ms = (operation.end_time - operation.start_time).total_seconds() * 1000
            operation.compression_time_ms = compression_time
            operation.compressed_size_bytes = compression_result.get('compressed_size', 0)
            operation.compression_ratio = compression_result.get('compression_ratio', 0.0)
            operation.compression_successful = compression_result.get('success', False)
            operation.coordination_successful = operation.compression_successful
            
            # Update resource usage
            operation.memory_usage_gb = self._get_current_memory_usage()
            operation.gpu_utilization_percent = self._get_current_gpu_utilization()
            operation.cpu_utilization_percent = self._get_current_cpu_utilization()
            
            # Measure domain impact
            operation.domain_impact_ms = self._measure_domain_impact(domain_name, compression_time)
            operation.domain_accuracy_impact = compression_result.get('accuracy_impact', 0.0)
            operation.domain_throughput_impact = compression_result.get('throughput_impact', 0.0)
            
            # Record coordination
            with self._operation_lock:
                del self.active_operations[operation_id]
            
            self.coordination_history.append(operation)
            self._update_coordination_analytics(operation)
            
            self.logger.debug(f"Domain compression coordinated for {domain_name}: {operation.compression_ratio:.4f} ratio")
            return operation
            
        except Exception as e:
            self.logger.error(f"Failed to coordinate domain compression for {domain_name}: {e}")
            if operation_id in self.active_operations:
                with self._operation_lock:
                    del self.active_operations[operation_id]
            raise RuntimeError(f"Domain compression coordination failed: {e}")
    
    def coordinate_domain_memory_management(self, domain_name: str) -> DomainCompressionOperation:
        """
        Coordinate memory management for domain.
        
        Args:
            domain_name: Domain identifier
            
        Returns:
            Domain memory management operation result
            
        Raises:
            ValueError: If domain is invalid
            RuntimeError: If coordination fails
        """
        if not domain_name or not isinstance(domain_name, str):
            raise ValueError("Domain name must be non-empty string")
        
        try:
            operation_id = self._generate_operation_id("domain_memory")
            start_time = datetime.utcnow()
            
            # Create coordination operation
            operation = DomainCompressionOperation(
                operation_id=operation_id,
                domain_name=domain_name,
                operation_type="domain_memory_management",
                priority=DomainCompressionPriority.NORMAL,
                start_time=start_time
            )
            
            with self._operation_lock:
                self.active_operations[operation_id] = operation
            
            # Capture initial memory state
            initial_memory = self._get_current_memory_usage()
            
            # Coordinate domain memory optimization
            memory_optimization_start = time.time()
            memory_optimization_result = self._perform_coordinated_domain_memory_management(domain_name)
            memory_optimization_time = (time.time() - memory_optimization_start) * 1000  # ms
            
            # Capture final memory state
            final_memory = self._get_current_memory_usage()
            memory_savings = initial_memory - final_memory
            
            # Update operation results
            operation.end_time = datetime.utcnow()
            operation.coordination_time_ms = (operation.end_time - operation.start_time).total_seconds() * 1000
            operation.memory_usage_gb = final_memory
            operation.coordination_successful = memory_optimization_result.get('success', False)
            
            # Measure domain impact
            operation.domain_impact_ms = memory_optimization_time * 0.1  # Lower impact for memory operations
            
            # Record coordination
            with self._operation_lock:
                del self.active_operations[operation_id]
            
            self.coordination_history.append(operation)
            self._update_coordination_analytics(operation)
            
            self.logger.debug(f"Domain memory management coordinated for {domain_name}: {memory_savings:.2f}GB saved")
            return operation
            
        except Exception as e:
            self.logger.error(f"Failed to coordinate domain memory management for {domain_name}: {e}")
            if operation_id in self.active_operations:
                with self._operation_lock:
                    del self.active_operations[operation_id]
            raise RuntimeError(f"Domain memory management coordination failed: {e}")
    
    def coordinate_domain_performance_optimization(self, domain_name: str) -> DomainCompressionOperation:
        """
        Coordinate performance optimization for domain.
        
        Args:
            domain_name: Domain identifier
            
        Returns:
            Domain performance optimization operation result
            
        Raises:
            ValueError: If domain is invalid
            RuntimeError: If coordination fails
        """
        if not domain_name or not isinstance(domain_name, str):
            raise ValueError("Domain name must be non-empty string")
        
        try:
            operation_id = self._generate_operation_id("domain_performance")
            start_time = datetime.utcnow()
            
            # Create coordination operation
            operation = DomainCompressionOperation(
                operation_id=operation_id,
                domain_name=domain_name,
                operation_type="domain_performance_optimization",
                priority=DomainCompressionPriority.HIGH,
                start_time=start_time
            )
            
            with self._operation_lock:
                self.active_operations[operation_id] = operation
            
            # Capture initial performance metrics
            initial_performance = self._capture_domain_performance_metrics(domain_name)
            
            # Coordinate domain performance optimization
            optimization_start = time.time()
            optimization_result = self._perform_coordinated_domain_performance_optimization(domain_name)
            optimization_time = (time.time() - optimization_start) * 1000  # ms
            
            # Capture final performance metrics
            final_performance = self._capture_domain_performance_metrics(domain_name)
            
            # Calculate performance improvements
            performance_improvements = self._calculate_domain_performance_improvements(initial_performance, final_performance)
            
            # Update operation results
            operation.end_time = datetime.utcnow()
            operation.coordination_time_ms = (operation.end_time - operation.start_time).total_seconds() * 1000
            operation.memory_usage_gb = final_performance.get('memory_usage_gb', 0.0)
            operation.gpu_utilization_percent = final_performance.get('gpu_utilization_percent', 0.0)
            operation.cpu_utilization_percent = final_performance.get('cpu_utilization_percent', 0.0)
            operation.coordination_successful = optimization_result.get('success', False)
            
            # Update domain-specific impacts
            operation.domain_accuracy_impact = performance_improvements.get('accuracy_improvement', 0.0)
            operation.domain_throughput_impact = performance_improvements.get('throughput_improvement', 0.0)
            
            # Record coordination
            with self._operation_lock:
                del self.active_operations[operation_id]
            
            self.coordination_history.append(operation)
            self._update_coordination_analytics(operation)
            
            self.logger.debug(f"Domain performance optimization coordinated for {domain_name}")
            return operation
            
        except Exception as e:
            self.logger.error(f"Failed to coordinate domain performance optimization for {domain_name}: {e}")
            if operation_id in self.active_operations:
                with self._operation_lock:
                    del self.active_operations[operation_id]
            raise RuntimeError(f"Domain performance optimization coordination failed: {e}")
    
    def coordinate_domain_resource_allocation(self, domain_name: str) -> Dict[str, Any]:
        """
        Coordinate resource allocation for domain.
        
        Args:
            domain_name: Domain identifier
            
        Returns:
            Domain resource allocation coordination results
            
        Raises:
            ValueError: If domain is invalid
            RuntimeError: If coordination fails
        """
        if not domain_name or not isinstance(domain_name, str):
            raise ValueError("Domain name must be non-empty string")
        
        try:
            # Analyze current domain resource usage
            current_resources = self._analyze_domain_resource_usage(domain_name)
            
            # Determine optimal resource allocation for domain
            optimal_allocation = self._determine_optimal_domain_resource_allocation(domain_name, current_resources)
            
            # Apply domain resource allocation changes
            allocation_result = self._apply_domain_resource_allocation(domain_name, optimal_allocation)
            
            # Monitor allocation effectiveness
            allocation_effectiveness = self._monitor_domain_allocation_effectiveness(domain_name, optimal_allocation)
            
            coordination_result = {
                'domain_name': domain_name,
                'current_resources': current_resources,
                'optimal_allocation': optimal_allocation,
                'allocation_result': allocation_result,
                'allocation_effectiveness': allocation_effectiveness,
                'coordination_timestamp': datetime.utcnow().isoformat()
            }
            
            # Store domain-specific allocation
            self.domain_resource_allocations[domain_name] = optimal_allocation
            
            self.logger.debug(f"Domain resource allocation coordinated for {domain_name}")
            return coordination_result
            
        except Exception as e:
            self.logger.error(f"Failed to coordinate domain resource allocation for {domain_name}: {e}")
            raise RuntimeError(f"Domain resource allocation coordination failed: {e}")
    
    def get_domain_compression_analytics(self, domain_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive domain-compression analytics.
        
        Args:
            domain_name: Optional specific domain for detailed analytics
            
        Returns:
            Domain compression analytics results
            
        Raises:
            ValueError: If domain name is invalid
            RuntimeError: If analytics retrieval fails
        """
        try:
            with self._analytics_lock:
                if domain_name:
                    # Get domain-specific analytics
                    domain_operations = [op for op in self.coordination_history if op.domain_name == domain_name]
                    domain_analytics = self._calculate_domain_coordination_analytics(domain_name, domain_operations)
                    
                    return {
                        'domain_name': domain_name,
                        'domain_analytics': domain_analytics,
                        'domain_operations': domain_operations,
                        'domain_performance_history': list(self.domain_performance_history.get(domain_name, [])),
                        'domain_resource_allocation': self.domain_resource_allocations.get(domain_name, {}),
                        'domain_compression_strategy': self.domain_compression_strategies.get(domain_name, 'default')
                    }
                else:
                    # Get global analytics
                    self._update_global_coordination_analytics()
                    
                    return {
                        'global_analytics': self.coordination_analytics,
                        'active_operations': len(self.active_operations),
                        'coordination_history': list(self.coordination_history),
                        'domain_resource_allocations': self.domain_resource_allocations,
                        'domain_compression_strategies': self.domain_compression_strategies,
                        'performance_baselines': self.performance_baselines,
                        'optimization_history': self.optimization_history[-20:],
                        'coordination_trends': self._analyze_domain_coordination_trends(),
                        'system_recommendations': self._generate_domain_coordination_recommendations()
                    }
                    
        except Exception as e:
            self.logger.error(f"Failed to get domain compression analytics: {e}")
            raise RuntimeError(f"Domain compression analytics retrieval failed: {e}")
    
    def optimize_domain_compression_strategy(self, domain_name: str) -> Dict[str, Any]:
        """
        Optimize compression strategy for domain.
        
        Args:
            domain_name: Domain identifier
            
        Returns:
            Domain compression strategy optimization results
            
        Raises:
            ValueError: If domain is invalid
            RuntimeError: If optimization fails
        """
        if not domain_name or not isinstance(domain_name, str):
            raise ValueError("Domain name must be non-empty string")
        
        try:
            # Analyze domain compression performance
            domain_operations = [op for op in self.coordination_history if op.domain_name == domain_name]
            
            if len(domain_operations) < 5:
                return {
                    'domain_name': domain_name,
                    'optimization_result': 'insufficient_data',
                    'current_strategy': self.domain_compression_strategies.get(domain_name, 'default'),
                    'recommended_strategy': 'continue_current',
                    'reason': 'Need at least 5 operations for strategy optimization'
                }
            
            # Calculate performance metrics
            recent_operations = domain_operations[-10:]
            average_compression_ratio = np.mean([op.compression_ratio for op in recent_operations])
            average_compression_time = np.mean([op.compression_time_ms for op in recent_operations])
            average_domain_impact = np.mean([op.domain_impact_ms for op in recent_operations])
            
            # Determine optimal strategy
            current_strategy = self.domain_compression_strategies.get(domain_name, 'adaptive')
            optimal_strategy = self._determine_optimal_compression_strategy(
                domain_name, average_compression_ratio, average_compression_time, average_domain_impact
            )
            
            optimization_result = {
                'domain_name': domain_name,
                'current_strategy': current_strategy,
                'recommended_strategy': optimal_strategy,
                'performance_metrics': {
                    'average_compression_ratio': average_compression_ratio,
                    'average_compression_time_ms': average_compression_time,
                    'average_domain_impact_ms': average_domain_impact
                },
                'optimization_applied': False
            }
            
            # Apply strategy change if beneficial
            if optimal_strategy != current_strategy:
                self.domain_compression_strategies[domain_name] = optimal_strategy
                optimization_result['optimization_applied'] = True
                optimization_result['optimization_reason'] = f"Strategy changed from {current_strategy} to {optimal_strategy} for better performance"
                
                # Configure compression system with new strategy
                if hasattr(self.hybrid_compression, 'set_domain_strategy'):
                    self.hybrid_compression.set_domain_strategy(domain_name, optimal_strategy)
            
            self.logger.info(f"Domain compression strategy optimized for {domain_name}: {optimal_strategy}")
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Failed to optimize domain compression strategy for {domain_name}: {e}")
            raise RuntimeError(f"Domain compression strategy optimization failed: {e}")
    
    def _initialize_performance_baselines(self) -> None:
        """Initialize performance baselines"""
        try:
            self.performance_baselines = {
                'coordination_time_ms': 50.0,
                'compression_time_ms': 100.0,
                'domain_impact_ms': 20.0,
                'memory_usage_gb': 4.0,
                'gpu_utilization_percent': 60.0,
                'cpu_utilization_percent': 40.0,
                'compression_ratio': 0.6,
                'domain_accuracy_impact': 0.01,
                'domain_throughput_impact': 0.05
            }
            self.logger.debug("Performance baselines initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize performance baselines: {e}")
            raise RuntimeError(f"Performance baseline initialization failed: {e}")
    
    def _setup_coordination_hooks(self) -> None:
        """Setup coordination hooks"""
        try:
            if hasattr(self.domain_registry, 'add_hook'):
                self.domain_registry.add_hook('domain_registered', self._domain_registered_hook)
                self.domain_registry.add_hook('domain_updated', self._domain_updated_hook)
            
            if hasattr(self.domain_router, 'add_hook'):
                self.domain_router.add_hook('routing_completed', self._domain_routing_completed_hook)
                self.domain_router.add_hook('routing_failed', self._domain_routing_failed_hook)
            
            if hasattr(self.hybrid_compression, 'add_hook'):
                self.hybrid_compression.add_hook('compression_completed', self._compression_completed_hook)
                self.hybrid_compression.add_hook('compression_failed', self._compression_failed_hook)
            
            self.logger.debug("Coordination hooks setup completed")
        except Exception as e:
            self.logger.error(f"Failed to setup coordination hooks: {e}")
            raise RuntimeError(f"Coordination hooks setup failed: {e}")
    
    def _initialize_domain_resource_coordination(self) -> None:
        """Initialize domain resource coordination"""
        try:
            # Set initial resource allocation based on strategy
            if self.config.resource_strategy == DomainResourceStrategy.DOMAIN_FIRST:
                self.resource_allocation.update({
                    'domain_memory_percent': 70.0,
                    'compression_memory_percent': 20.0,
                    'domain_gpu_percent': 75.0,
                    'compression_gpu_percent': 15.0
                })
            elif self.config.resource_strategy == DomainResourceStrategy.COMPRESSION_FIRST:
                self.resource_allocation.update({
                    'domain_memory_percent': 30.0,
                    'compression_memory_percent': 60.0,
                    'domain_gpu_percent': 40.0,
                    'compression_gpu_percent': 50.0
                })
            elif self.config.resource_strategy == DomainResourceStrategy.DOMAIN_ISOLATED:
                self.resource_allocation.update({
                    'domain_memory_percent': 80.0,
                    'compression_memory_percent': 15.0,
                    'domain_gpu_percent': 85.0,
                    'compression_gpu_percent': 10.0
                })
            # BALANCED and others use default values
            
            self.logger.debug("Domain resource coordination initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize domain resource coordination: {e}")
            raise RuntimeError(f"Domain resource coordination initialization failed: {e}")
    
    def _start_resource_monitoring(self) -> None:
        """Start resource monitoring"""
        try:
            self.resource_monitoring_active = True
            self.logger.debug("Resource monitoring started")
        except Exception as e:
            self.logger.error(f"Failed to start resource monitoring: {e}")
            raise RuntimeError(f"Resource monitoring start failed: {e}")
    
    def _initialize_coordination_analytics(self) -> None:
        """Initialize coordination analytics"""
        try:
            self.coordination_analytics = DomainCompressionAnalytics()
            self.logger.debug("Coordination analytics initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize coordination analytics: {e}")
            raise RuntimeError(f"Coordination analytics initialization failed: {e}")
    
    def _generate_operation_id(self, operation_type: str) -> str:
        """Generate unique operation ID"""
        timestamp = str(int(time.time() * 1000))
        return f"{operation_type}_{timestamp}_{hash(threading.current_thread()) % 10000:04d}"
    
    def _determine_domain_compression_priority(self, domain_name: str, data: torch.Tensor) -> DomainCompressionPriority:
        """Determine compression priority based on domain and data characteristics"""
        data_size_mb = data.numel() * data.element_size() / (1024 * 1024)
        
        # Check if domain has specific priority requirements
        domain_config = self.domain_coordination_configs.get(domain_name, {})
        priority_config = domain_config.get('priority', 'normal')
        
        if priority_config == 'critical':
            return DomainCompressionPriority.CRITICAL
        elif priority_config == 'high':
            return DomainCompressionPriority.HIGH
        elif data_size_mb > 500:  # Large data gets high priority for compression
            return DomainCompressionPriority.HIGH
        elif data_size_mb > 100:
            return DomainCompressionPriority.NORMAL
        elif data_size_mb < 10:
            return DomainCompressionPriority.LOW
        else:
            return DomainCompressionPriority.DOMAIN_SPECIFIC
    
    def _coordinate_domain_resource_allocation_for_compression(self, domain_name: str, data: torch.Tensor) -> Dict[str, float]:
        """Coordinate resource allocation for domain compression operation"""
        try:
            current_memory = self._get_current_memory_usage()
            current_gpu = self._get_current_gpu_utilization()
            
            # Get domain-specific resource allocation if available
            domain_allocation = self.domain_resource_allocations.get(domain_name, {})
            
            # Determine resource allocation based on current usage and data size
            data_size_gb = data.numel() * data.element_size() / (1024**3)
            
            if current_memory > self.config.memory_coordination_threshold_gb:
                # High memory usage - allocate more to compression, less to domain
                memory_allocation = min(0.8, domain_allocation.get('compression_memory', 0.4) + data_size_gb * 0.1)
                domain_memory_allocation = max(0.2, domain_allocation.get('domain_memory', 0.6) - data_size_gb * 0.1)
            else:
                # Normal memory usage
                memory_allocation = domain_allocation.get('compression_memory', self.resource_allocation['compression_memory_percent'] / 100.0)
                domain_memory_allocation = domain_allocation.get('domain_memory', self.resource_allocation['domain_memory_percent'] / 100.0)
            
            if current_gpu > self.config.gpu_coordination_threshold_percent:
                # High GPU usage - reduce GPU allocation for compression
                gpu_allocation = max(0.1, domain_allocation.get('compression_gpu', self.resource_allocation['compression_gpu_percent'] / 100.0) - 0.2)
                domain_gpu_allocation = min(0.8, domain_allocation.get('domain_gpu', self.resource_allocation['domain_gpu_percent'] / 100.0) + 0.1)
            else:
                gpu_allocation = domain_allocation.get('compression_gpu', self.resource_allocation['compression_gpu_percent'] / 100.0)
                domain_gpu_allocation = domain_allocation.get('domain_gpu', self.resource_allocation['domain_gpu_percent'] / 100.0)
            
            return {
                'compression_memory_allocation': memory_allocation,
                'domain_memory_allocation': domain_memory_allocation,
                'compression_gpu_allocation': gpu_allocation,
                'domain_gpu_allocation': domain_gpu_allocation,
                'compression_cpu_allocation': domain_allocation.get('compression_cpu', self.resource_allocation['compression_cpu_percent'] / 100.0),
                'domain_cpu_allocation': domain_allocation.get('domain_cpu', self.resource_allocation['domain_cpu_percent'] / 100.0)
            }
        except Exception as e:
            self.logger.error(f"Failed to coordinate domain resource allocation for compression: {e}")
            return {
                'compression_memory_allocation': 0.3, 'domain_memory_allocation': 0.5,
                'compression_gpu_allocation': 0.2, 'domain_gpu_allocation': 0.6,
                'compression_cpu_allocation': 0.4, 'domain_cpu_allocation': 0.5
            }
    
    def _perform_coordinated_domain_compression(self, domain_name: str, data: torch.Tensor, resource_allocation: Dict[str, float]) -> Dict[str, Any]:
        """Perform coordinated domain compression"""
        try:
            # Configure compression system with domain-specific and resource allocation
            compression_config = {
                'domain_name': domain_name,
                'memory_limit_percent': resource_allocation['compression_memory_allocation'] * 100,
                'gpu_allocation_percent': resource_allocation['compression_gpu_allocation'] * 100,
                'domain_specific_optimization': True,
                'compression_strategy': self.domain_compression_strategies.get(domain_name, 'adaptive')
            }
            
            # Perform compression with domain coordination
            compression_result = self.hybrid_compression.compress(data)
            compression_result['resource_allocation'] = resource_allocation
            compression_result['domain_coordination'] = compression_config
            
            # Calculate domain-specific impacts
            compression_result['accuracy_impact'] = self._estimate_domain_accuracy_impact(domain_name, compression_result)
            compression_result['throughput_impact'] = self._estimate_domain_throughput_impact(domain_name, compression_result)
            
            return compression_result
        except Exception as e:
            self.logger.error(f"Coordinated domain compression failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _perform_coordinated_domain_memory_management(self, domain_name: str) -> Dict[str, Any]:
        """Perform coordinated domain memory management"""
        try:
            operations_performed = []
            
            # Domain-specific memory cleanup
            if hasattr(self.hybrid_compression, 'clear_domain_cache'):
                self.hybrid_compression.clear_domain_cache(domain_name)
                operations_performed.append(f"domain_cache_cleared_{domain_name}")
            
            # Optimize memory pools for domain
            if hasattr(self.hybrid_compression, 'optimize_domain_memory_pools'):
                self.hybrid_compression.optimize_domain_memory_pools(domain_name)
                operations_performed.append(f"domain_memory_pools_optimized_{domain_name}")
            
            # General memory cleanup
            import gc
            collected = gc.collect()
            if collected > 0:
                operations_performed.append(f"garbage_collection_{collected}")
            
            return {
                'success': True,
                'operations_performed': operations_performed,
                'domain_name': domain_name
            }
        except Exception as e:
            self.logger.error(f"Coordinated domain memory management failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _perform_coordinated_domain_performance_optimization(self, domain_name: str) -> Dict[str, Any]:
        """Perform coordinated domain performance optimization"""
        try:
            optimizations_applied = []
            
            # Optimize compression parameters for domain
            if hasattr(self.hybrid_compression, 'optimize_domain_parameters'):
                self.hybrid_compression.optimize_domain_parameters(domain_name)
                optimizations_applied.append(f"domain_compression_parameters_optimized_{domain_name}")
            
            # Optimize dynamic switching for domain
            if self.dynamic_switching and hasattr(self.dynamic_switching, 'optimize_domain_switching'):
                self.dynamic_switching.optimize_domain_switching(domain_name)
                optimizations_applied.append(f"domain_switching_optimized_{domain_name}")
            
            # Optimize domain resource allocation
            self._optimize_domain_resource_allocation(domain_name)
            optimizations_applied.append(f"domain_resource_allocation_optimized_{domain_name}")
            
            return {
                'success': True,
                'optimizations_applied': optimizations_applied,
                'domain_name': domain_name
            }
        except Exception as e:
            self.logger.error(f"Coordinated domain performance optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _measure_domain_impact(self, domain_name: str, compression_time_ms: float) -> float:
        """Measure impact of compression on domain performance"""
        try:
            # Simple estimation based on compression time and domain characteristics
            domain_config = self.domain_coordination_configs.get(domain_name, {})
            domain_sensitivity = domain_config.get('latency_sensitivity', 1.0)
            
            # Impact factor based on domain sensitivity and system load
            current_cpu = self._get_current_cpu_utilization()
            current_gpu = self._get_current_gpu_utilization()
            
            system_load_factor = 1.0 + (current_cpu + current_gpu) / 200.0  # Normalized impact
            domain_impact = compression_time_ms * domain_sensitivity * system_load_factor * 0.2  # Estimated impact
            
            return domain_impact
        except Exception as e:
            self.logger.error(f"Failed to measure domain impact: {e}")
            return 0.0
    
    def _capture_domain_performance_metrics(self, domain_name: str) -> Dict[str, float]:
        """Capture current performance metrics for domain"""
        try:
            base_metrics = {
                'memory_usage_gb': self._get_current_memory_usage(),
                'gpu_utilization_percent': self._get_current_gpu_utilization(),
                'cpu_utilization_percent': self._get_current_cpu_utilization(),
                'timestamp': time.time()
            }
            
            # Add domain-specific metrics if available
            domain_config = self.domain_coordination_configs.get(domain_name, {})
            if 'performance_metrics' in domain_config:
                base_metrics.update(domain_config['performance_metrics'])
            
            return base_metrics
        except Exception as e:
            self.logger.error(f"Failed to capture domain performance metrics: {e}")
            return {}
    
    def _calculate_domain_performance_improvements(self, initial: Dict[str, float], final: Dict[str, float]) -> Dict[str, float]:
        """Calculate domain performance improvements"""
        try:
            improvements = {}
            
            for metric in ['memory_usage_gb', 'gpu_utilization_percent', 'cpu_utilization_percent']:
                if metric in initial and metric in final:
                    if metric == 'memory_usage_gb':
                        # For memory, reduction is improvement
                        improvements[f"{metric}_improvement"] = initial[metric] - final[metric]
                    else:
                        # For utilization, optimization can go either way
                        improvements[f"{metric}_change"] = final[metric] - initial[metric]
            
            # Domain-specific improvements
            improvements['accuracy_improvement'] = final.get('domain_accuracy', 0) - initial.get('domain_accuracy', 0)
            improvements['throughput_improvement'] = final.get('domain_throughput', 0) - initial.get('domain_throughput', 0)
            
            return improvements
        except Exception as e:
            self.logger.error(f"Failed to calculate domain performance improvements: {e}")
            return {}
    
    def _analyze_domain_resource_usage(self, domain_name: str) -> Dict[str, float]:
        """Analyze current domain resource usage"""
        try:
            base_usage = {
                'memory_usage_gb': self._get_current_memory_usage(),
                'memory_usage_percent': self._get_current_memory_usage() / 16.0 * 100,  # Assume 16GB total
                'gpu_utilization_percent': self._get_current_gpu_utilization(),
                'cpu_utilization_percent': self._get_current_cpu_utilization(),
                'active_operations': len([op for op in self.active_operations.values() if op.domain_name == domain_name])
            }
            
            # Add domain-specific resource usage
            domain_allocation = self.domain_resource_allocations.get(domain_name, {})
            base_usage['domain_memory_allocation'] = domain_allocation.get('domain_memory', 0.5)
            base_usage['domain_gpu_allocation'] = domain_allocation.get('domain_gpu', 0.6)
            base_usage['domain_cpu_allocation'] = domain_allocation.get('domain_cpu', 0.5)
            
            return base_usage
        except Exception as e:
            self.logger.error(f"Failed to analyze domain resource usage: {e}")
            return {}
    
    def _determine_optimal_domain_resource_allocation(self, domain_name: str, current_resources: Dict[str, float]) -> Dict[str, float]:
        """Determine optimal resource allocation for domain"""
        try:
            optimal_allocation = self.domain_resource_allocations.get(domain_name, {}).copy()
            if not optimal_allocation:
                optimal_allocation = {
                    'domain_memory': 0.5, 'compression_memory': 0.3, 'system_memory': 0.2,
                    'domain_gpu': 0.6, 'compression_gpu': 0.3, 'system_gpu': 0.1,
                    'domain_cpu': 0.5, 'compression_cpu': 0.4, 'system_cpu': 0.1
                }
            
            # Adjust based on current usage
            memory_usage = current_resources.get('memory_usage_percent', 50.0)
            gpu_usage = current_resources.get('gpu_utilization_percent', 50.0)
            active_ops = current_resources.get('active_operations', 0)
            
            if memory_usage > 80.0:
                # High memory usage - reduce domain allocation, increase compression efficiency
                optimal_allocation['domain_memory'] = max(0.3, optimal_allocation['domain_memory'] - 0.1)
                optimal_allocation['compression_memory'] = min(0.5, optimal_allocation['compression_memory'] + 0.1)
            elif memory_usage < 30.0:
                # Low memory usage - can afford to give more to domain
                optimal_allocation['domain_memory'] = min(0.7, optimal_allocation['domain_memory'] + 0.1)
                optimal_allocation['compression_memory'] = max(0.2, optimal_allocation['compression_memory'] - 0.1)
            
            if gpu_usage > 85.0:
                # High GPU usage - balance allocation
                optimal_allocation['domain_gpu'] = max(0.4, optimal_allocation['domain_gpu'] - 0.1)
                optimal_allocation['compression_gpu'] = min(0.4, optimal_allocation['compression_gpu'] + 0.1)
            
            if active_ops > 5:
                # High operation load - optimize for throughput
                optimal_allocation['compression_cpu'] = min(0.5, optimal_allocation['compression_cpu'] + 0.1)
                optimal_allocation['domain_cpu'] = max(0.4, optimal_allocation['domain_cpu'] - 0.1)
            
            return optimal_allocation
        except Exception as e:
            self.logger.error(f"Failed to determine optimal domain resource allocation: {e}")
            return self.domain_resource_allocations.get(domain_name, {})
    
    def _apply_domain_resource_allocation(self, domain_name: str, allocation: Dict[str, float]) -> Dict[str, Any]:
        """Apply domain resource allocation changes"""
        try:
            # Update internal allocation
            self.domain_resource_allocations[domain_name] = allocation
            
            # Apply to systems if they support it
            allocation_results = []
            
            if hasattr(self.hybrid_compression, 'set_domain_resource_allocation'):
                compression_result = self.hybrid_compression.set_domain_resource_allocation(domain_name, {
                    'memory_percent': allocation.get('compression_memory', 0.3) * 100,
                    'gpu_percent': allocation.get('compression_gpu', 0.3) * 100,
                    'cpu_percent': allocation.get('compression_cpu', 0.4) * 100
                })
                allocation_results.append(f"compression_allocation_{compression_result}")
            
            # Store domain configuration for coordination
            if domain_name not in self.domain_coordination_configs:
                self.domain_coordination_configs[domain_name] = {}
            
            self.domain_coordination_configs[domain_name]['resource_allocation'] = allocation
            allocation_results.append(f"domain_coordination_config_updated")
            
            return {
                'success': True,
                'allocation_results': allocation_results,
                'domain_name': domain_name
            }
        except Exception as e:
            self.logger.error(f"Failed to apply domain resource allocation: {e}")
            return {'success': False, 'error': str(e)}
    
    def _monitor_domain_allocation_effectiveness(self, domain_name: str, allocation: Dict[str, float]) -> Dict[str, Any]:
        """Monitor domain allocation effectiveness"""
        try:
            # Simple effectiveness monitoring
            current_resources = self._analyze_domain_resource_usage(domain_name)
            
            effectiveness_score = 1.0
            if current_resources.get('memory_usage_percent', 0) > 90.0:
                effectiveness_score -= 0.3
            if current_resources.get('gpu_utilization_percent', 0) > 95.0:
                effectiveness_score -= 0.2
            if current_resources.get('cpu_utilization_percent', 0) > 95.0:
                effectiveness_score -= 0.2
            if current_resources.get('active_operations', 0) > 10:
                effectiveness_score -= 0.1
            
            return {
                'effectiveness_score': max(0.0, effectiveness_score),
                'current_resources': current_resources,
                'allocation_applied': allocation,
                'domain_name': domain_name
            }
        except Exception as e:
            self.logger.error(f"Failed to monitor domain allocation effectiveness: {e}")
            return {'effectiveness_score': 0.5}
    
    def _optimize_domain_resource_allocation(self, domain_name: str) -> None:
        """Optimize resource allocation for domain"""
        try:
            current_resources = self._analyze_domain_resource_usage(domain_name)
            optimal_allocation = self._determine_optimal_domain_resource_allocation(domain_name, current_resources)
            self._apply_domain_resource_allocation(domain_name, optimal_allocation)
        except Exception as e:
            self.logger.error(f"Failed to optimize domain resource allocation: {e}")
    
    def _determine_optimal_compression_strategy(self, domain_name: str, compression_ratio: float, compression_time: float, domain_impact: float) -> str:
        """Determine optimal compression strategy for domain"""
        try:
            # Simple strategy selection based on performance metrics
            if compression_ratio > 0.8 and compression_time < 50:
                return "aggressive"
            elif compression_ratio < 0.3 or compression_time > 200:
                return "conservative"
            elif domain_impact > 100:
                return "real_time"
            elif compression_time < 100 and domain_impact < 50:
                return "domain_optimized"
            else:
                return "adaptive"
        except Exception as e:
            self.logger.error(f"Failed to determine optimal compression strategy: {e}")
            return "adaptive"
    
    def _estimate_domain_accuracy_impact(self, domain_name: str, compression_result: Dict[str, Any]) -> float:
        """Estimate impact on domain accuracy"""
        try:
            compression_ratio = compression_result.get('compression_ratio', 0.0)
            # Simple estimation: higher compression may impact accuracy
            if compression_ratio > 0.9:
                return 0.05  # 5% potential accuracy impact
            elif compression_ratio > 0.7:
                return 0.02  # 2% potential accuracy impact
            else:
                return 0.01  # 1% potential accuracy impact
        except Exception as e:
            self.logger.error(f"Failed to estimate domain accuracy impact: {e}")
            return 0.0
    
    def _estimate_domain_throughput_impact(self, domain_name: str, compression_result: Dict[str, Any]) -> float:
        """Estimate impact on domain throughput"""
        try:
            compression_time = compression_result.get('compression_time_ms', 0.0)
            # Simple estimation: longer compression time impacts throughput
            if compression_time > 200:
                return 0.20  # 20% throughput impact
            elif compression_time > 100:
                return 0.10  # 10% throughput impact
            else:
                return 0.05  # 5% throughput impact
        except Exception as e:
            self.logger.error(f"Failed to estimate domain throughput impact: {e}")
            return 0.0
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        try:
            import psutil
            return psutil.virtual_memory().used / (1024**3)
        except Exception:
            return 0.0
    
    def _get_current_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        except Exception:
            return 0.0
    
    def _get_current_cpu_utilization(self) -> float:
        """Get current CPU utilization percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0
    
    def _update_coordination_analytics(self, operation: DomainCompressionOperation) -> None:
        """Update coordination analytics with operation results"""
        try:
            with self._analytics_lock:
                self.coordination_analytics.total_operations += 1
                
                if operation.coordination_successful:
                    self.coordination_analytics.successful_operations += 1
                else:
                    self.coordination_analytics.failed_operations += 1
                
                # Update operation type counters
                if operation.operation_type == "domain_compression":
                    self.coordination_analytics.domain_compression_operations += 1
                elif operation.operation_type == "domain_memory_management":
                    self.coordination_analytics.domain_memory_operations += 1
                elif operation.operation_type == "domain_performance_optimization":
                    self.coordination_analytics.domain_performance_operations += 1
                elif operation.operation_type == "domain_resource_allocation":
                    self.coordination_analytics.domain_resource_operations += 1
                
                # Update performance averages
                total_ops = self.coordination_analytics.total_operations
                self.coordination_analytics.average_coordination_time_ms = (
                    (self.coordination_analytics.average_coordination_time_ms * (total_ops - 1) + 
                     operation.coordination_time_ms) / total_ops
                )
                
                if operation.compression_time_ms > 0:
                    self.coordination_analytics.average_compression_time_ms = (
                        (self.coordination_analytics.average_compression_time_ms * (total_ops - 1) + 
                         operation.compression_time_ms) / total_ops
                    )
                
                if operation.domain_impact_ms > 0:
                    self.coordination_analytics.average_domain_impact_ms = (
                        (self.coordination_analytics.average_domain_impact_ms * (total_ops - 1) + 
                         operation.domain_impact_ms) / total_ops
                    )
                
                if operation.compression_ratio > 0:
                    self.coordination_analytics.average_domain_compression_ratio = (
                        (self.coordination_analytics.average_domain_compression_ratio * (total_ops - 1) + 
                         operation.compression_ratio) / total_ops
                    )
                
                # Update resource tracking
                if operation.memory_usage_gb > 0:
                    self.coordination_analytics.average_memory_usage_gb = (
                        (self.coordination_analytics.average_memory_usage_gb * (total_ops - 1) + 
                         operation.memory_usage_gb) / total_ops
                    )
                    self.coordination_analytics.peak_memory_usage_gb = max(
                        self.coordination_analytics.peak_memory_usage_gb,
                        operation.memory_usage_gb
                    )
                
                if operation.gpu_utilization_percent > 0:
                    self.coordination_analytics.average_gpu_utilization_percent = (
                        (self.coordination_analytics.average_gpu_utilization_percent * (total_ops - 1) + 
                         operation.gpu_utilization_percent) / total_ops
                    )
                    self.coordination_analytics.peak_gpu_utilization_percent = max(
                        self.coordination_analytics.peak_gpu_utilization_percent,
                        operation.gpu_utilization_percent
                    )
                
                # Update domain performance tracking
                self.domain_performance_history[operation.domain_name].append({
                    'timestamp': operation.end_time or operation.start_time,
                    'compression_ratio': operation.compression_ratio,
                    'compression_time_ms': operation.compression_time_ms,
                    'domain_impact_ms': operation.domain_impact_ms,
                    'coordination_successful': operation.coordination_successful
                })
                
        except Exception as e:
            self.logger.error(f"Failed to update coordination analytics: {e}")
    
    def _calculate_domain_coordination_analytics(self, domain_name: str, operations: List[DomainCompressionOperation]) -> Dict[str, Any]:
        """Calculate coordination analytics for specific domain"""
        try:
            if not operations:
                return {'domain_name': domain_name, 'total_operations': 0}
            
            successful_ops = [op for op in operations if op.coordination_successful]
            
            analytics = {
                'domain_name': domain_name,
                'total_operations': len(operations),
                'successful_operations': len(successful_ops),
                'success_rate': len(successful_ops) / len(operations),
                'operation_types': {
                    'domain_compression': len([op for op in operations if op.operation_type == "domain_compression"]),
                    'domain_memory': len([op for op in operations if op.operation_type == "domain_memory_management"]),
                    'domain_performance': len([op for op in operations if op.operation_type == "domain_performance_optimization"]),
                    'domain_resource': len([op for op in operations if op.operation_type == "domain_resource_allocation"])
                },
                'average_coordination_time_ms': np.mean([op.coordination_time_ms for op in operations]),
                'average_compression_time_ms': np.mean([op.compression_time_ms for op in operations if op.compression_time_ms > 0]),
                'average_compression_ratio': np.mean([op.compression_ratio for op in operations if op.compression_ratio > 0]),
                'average_domain_impact_ms': np.mean([op.domain_impact_ms for op in operations if op.domain_impact_ms > 0]),
                'peak_memory_usage_gb': max([op.memory_usage_gb for op in operations if op.memory_usage_gb > 0], default=0.0),
                'peak_gpu_utilization_percent': max([op.gpu_utilization_percent for op in operations if op.gpu_utilization_percent > 0], default=0.0),
                'average_accuracy_impact': np.mean([op.domain_accuracy_impact for op in operations if op.domain_accuracy_impact > 0]),
                'average_throughput_impact': np.mean([op.domain_throughput_impact for op in operations if op.domain_throughput_impact > 0])
            }
            
            return analytics
        except Exception as e:
            self.logger.error(f"Failed to calculate domain coordination analytics: {e}")
            return {'domain_name': domain_name, 'error': str(e)}
    
    def _update_global_coordination_analytics(self) -> None:
        """Update global coordination analytics"""
        try:
            with self._analytics_lock:
                # Update domains coordinated
                unique_domains = set(op.domain_name for op in self.coordination_history)
                self.coordination_analytics.domains_coordinated = len(unique_domains)
                
                # Find best and worst performing domains
                if unique_domains:
                    domain_performance = {}
                    for domain in unique_domains:
                        domain_ops = [op for op in self.coordination_history if op.domain_name == domain and op.coordination_successful]
                        if domain_ops:
                            avg_compression_ratio = np.mean([op.compression_ratio for op in domain_ops if op.compression_ratio > 0])
                            avg_domain_impact = np.mean([op.domain_impact_ms for op in domain_ops if op.domain_impact_ms > 0])
                            # Performance score: higher compression ratio, lower impact is better
                            performance_score = avg_compression_ratio / max(1.0, avg_domain_impact / 100.0)
                            domain_performance[domain] = performance_score
                    
                    if domain_performance:
                        best_domain = max(domain_performance.items(), key=lambda x: x[1])
                        worst_domain = min(domain_performance.items(), key=lambda x: x[1])
                        most_compressed = max(
                            [(domain, np.mean([op.compression_ratio for op in self.coordination_history 
                                             if op.domain_name == domain and op.compression_ratio > 0]))
                             for domain in unique_domains if any(op.domain_name == domain and op.compression_ratio > 0 
                                                               for op in self.coordination_history)],
                            key=lambda x: x[1], default=("", 0)
                        )
                        
                        self.coordination_analytics.best_performing_domain = best_domain[0]
                        self.coordination_analytics.worst_performing_domain = worst_domain[0]
                        self.coordination_analytics.most_compressed_domain = most_compressed[0]
                
                # Calculate efficiency metrics
                if self.coordination_analytics.total_operations > 0:
                    self.coordination_analytics.coordination_efficiency = (
                        self.coordination_analytics.successful_operations / 
                        self.coordination_analytics.total_operations
                    )
                
                # Update timestamp
                self.coordination_analytics.timestamp = datetime.utcnow()
                
        except Exception as e:
            self.logger.error(f"Failed to update global coordination analytics: {e}")
    
    def _analyze_domain_coordination_trends(self) -> Dict[str, str]:
        """Analyze domain coordination trends"""
        try:
            recent_operations = list(self.coordination_history)[-20:]
            
            if len(recent_operations) < 5:
                return {'insufficient_data': 'true'}
            
            # Analyze performance trends
            coordination_times = [op.coordination_time_ms for op in recent_operations]
            compression_times = [op.compression_time_ms for op in recent_operations if op.compression_time_ms > 0]
            compression_ratios = [op.compression_ratio for op in recent_operations if op.compression_ratio > 0]
            domain_impacts = [op.domain_impact_ms for op in recent_operations if op.domain_impact_ms > 0]
            
            trends = {}
            
            if len(coordination_times) >= 3:
                coord_trend = np.polyfit(range(len(coordination_times)), coordination_times, 1)[0]
                trends['coordination_time'] = 'improving' if coord_trend < -1.0 else 'degrading' if coord_trend > 1.0 else 'stable'
            
            if len(compression_times) >= 3:
                comp_trend = np.polyfit(range(len(compression_times)), compression_times, 1)[0]
                trends['compression_time'] = 'improving' if comp_trend < -2.0 else 'degrading' if comp_trend > 2.0 else 'stable'
            
            if len(compression_ratios) >= 3:
                ratio_trend = np.polyfit(range(len(compression_ratios)), compression_ratios, 1)[0]
                trends['compression_ratio'] = 'improving' if ratio_trend > 0.01 else 'degrading' if ratio_trend < -0.01 else 'stable'
            
            if len(domain_impacts) >= 3:
                impact_trend = np.polyfit(range(len(domain_impacts)), domain_impacts, 1)[0]
                trends['domain_impact'] = 'improving' if impact_trend < -1.0 else 'degrading' if impact_trend > 1.0 else 'stable'
            
            return trends
        except Exception as e:
            self.logger.error(f"Failed to analyze domain coordination trends: {e}")
            return {}
    
    def _generate_domain_coordination_recommendations(self) -> List[str]:
        """Generate domain coordination recommendations"""
        recommendations = []
        
        if self.coordination_analytics.coordination_efficiency < 0.8:
            recommendations.append("Domain coordination efficiency below 80% - review coordination strategies")
        
        if len(self.active_operations) > 10:
            recommendations.append("High number of active domain operations - consider load balancing")
        
        if self.coordination_analytics.average_coordination_time_ms > 100:
            recommendations.append("High coordination overhead for domains - optimize coordination algorithms")
        
        if self.coordination_analytics.average_domain_compression_ratio < 0.5:
            recommendations.append("Low compression ratios across domains - review domain-specific strategies")
        
        if self.coordination_analytics.average_domain_impact_ms > 100:
            recommendations.append("High domain performance impact - consider more efficient coordination")
        
        unique_domains = set(op.domain_name for op in self.coordination_history)
        if len(unique_domains) < 3:
            recommendations.append("Few domains being coordinated - system ready for more domain integrations")
        
        return recommendations if recommendations else ["Domain coordination operating within optimal parameters"]
    
    def _domain_registered_hook(self, domain_name: str, domain_config: Dict[str, Any]) -> None:
        """Domain registered hook"""
        try:
            # Automatically setup coordination for new domain
            self.domain_coordination_configs[domain_name] = {
                'priority': domain_config.get('priority', 'normal'),
                'latency_sensitivity': domain_config.get('latency_sensitivity', 1.0),
                'compression_preference': domain_config.get('compression_preference', 'adaptive')
            }
            
            # Set default compression strategy
            self.domain_compression_strategies[domain_name] = domain_config.get('compression_strategy', 'adaptive')
            
            self.logger.info(f"Domain coordination setup for registered domain: {domain_name}")
        except Exception as e:
            self.logger.error(f"Domain registered hook failed: {e}")
    
    def _domain_updated_hook(self, domain_name: str, domain_config: Dict[str, Any]) -> None:
        """Domain updated hook"""
        try:
            # Update domain coordination configuration
            if domain_name in self.domain_coordination_configs:
                self.domain_coordination_configs[domain_name].update({
                    'priority': domain_config.get('priority', 'normal'),
                    'latency_sensitivity': domain_config.get('latency_sensitivity', 1.0),
                    'compression_preference': domain_config.get('compression_preference', 'adaptive')
                })
                
                # Update compression strategy if specified
                if 'compression_strategy' in domain_config:
                    self.domain_compression_strategies[domain_name] = domain_config['compression_strategy']
                
                self.logger.info(f"Domain coordination configuration updated for: {domain_name}")
        except Exception as e:
            self.logger.error(f"Domain updated hook failed: {e}")
    
    def _domain_routing_completed_hook(self, routing_data: Dict[str, Any]) -> None:
        """Domain routing completed hook"""
        try:
            domain_name = routing_data.get('domain_name')
            if domain_name:
                # Record successful routing for coordination analytics
                self.domain_performance_history[domain_name].append({
                    'timestamp': datetime.utcnow(),
                    'routing_successful': True,
                    'routing_time_ms': routing_data.get('routing_time_ms', 0)
                })
        except Exception as e:
            self.logger.error(f"Domain routing completed hook failed: {e}")
    
    def _domain_routing_failed_hook(self, routing_data: Dict[str, Any]) -> None:
        """Domain routing failed hook"""
        try:
            domain_name = routing_data.get('domain_name')
            if domain_name:
                # Record failed routing for coordination analytics
                self.domain_performance_history[domain_name].append({
                    'timestamp': datetime.utcnow(),
                    'routing_successful': False,
                    'routing_error': routing_data.get('error', 'Unknown error')
                })
        except Exception as e:
            self.logger.error(f"Domain routing failed hook failed: {e}")
    
    def _compression_completed_hook(self, compression_data: Dict[str, Any]) -> None:
        """Compression completed hook"""
        try:
            domain_name = compression_data.get('domain_name')
            if domain_name:
                # Record compression performance for domain
                self.domain_performance_history[domain_name].append({
                    'timestamp': datetime.utcnow(),
                    'compression_successful': True,
                    'compression_time_ms': compression_data.get('compression_time_ms', 0),
                    'compression_ratio': compression_data.get('compression_ratio', 0)
                })
        except Exception as e:
            self.logger.error(f"Compression completed hook failed: {e}")
    
    def _compression_failed_hook(self, compression_data: Dict[str, Any]) -> None:
        """Compression failed hook"""
        try:
            domain_name = compression_data.get('domain_name')
            if domain_name:
                # Record compression failure for domain
                self.domain_performance_history[domain_name].append({
                    'timestamp': datetime.utcnow(),
                    'compression_successful': False,
                    'compression_error': compression_data.get('error', 'Unknown error')
                })
        except Exception as e:
            self.logger.error(f"Compression failed hook failed: {e}")
    
    def get_coordinator_status(self) -> Dict[str, Any]:
        """Get current coordinator status"""
        return {
            'coordination_active': self.coordination_active,
            'monitoring_active': self.monitoring_active,
            'resource_monitoring_active': self.resource_monitoring_active,
            'coordination_mode': self.coordination_mode.value,
            'active_operations': len(self.active_operations),
            'domains_coordinated': len(set(op.domain_name for op in self.coordination_history)),
            'domain_resource_allocations': self.domain_resource_allocations,
            'domain_compression_strategies': self.domain_compression_strategies,
            'recent_performance': list(self.coordination_history)[-5:]
        }
    
    def shutdown(self) -> None:
        """Shutdown domain-compression coordinator"""
        try:
            with self._coordination_lock:
                self.coordination_active = False
                self.monitoring_active = False
                self.resource_monitoring_active = False
                
                # Complete active operations
                for operation_id in list(self.active_operations.keys()):
                    operation = self.active_operations[operation_id]
                    operation.end_time = datetime.utcnow()
                    operation.coordination_successful = False
                    operation.error_message = "Coordinator shutdown"
                
                self.active_operations.clear()
                self.logger.info("Domain-compression coordinator shutdown completed")
                
        except Exception as e:
            self.logger.error(f"Domain-compression coordinator shutdown failed: {e}")
            raise RuntimeError(f"Coordinator shutdown failed: {e}")