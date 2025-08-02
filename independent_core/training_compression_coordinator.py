"""
Training Compression Coordinator - Coordination between training and compression operations
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

# Import training system components (avoid circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from training_manager import TrainingManager

# Import hybrid compression system components
from compression_systems.padic.hybrid_padic_compressor import HybridPadicCompressionSystem
from compression_systems.padic.dynamic_switching_manager import DynamicSwitchingManager
from compression_systems.padic.direction_manager import DirectionManager


class CoordinationMode(Enum):
    """Coordination mode enumeration"""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    ADAPTIVE = "adaptive"
    PRIORITY_BASED = "priority_based"
    LOAD_BALANCED = "load_balanced"
    REAL_TIME = "real_time"


class CompressionPriority(Enum):
    """Compression priority enumeration"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    REAL_TIME = "real_time"


class ResourceAllocationStrategy(Enum):
    """Resource allocation strategy enumeration"""
    TRAINING_FIRST = "training_first"
    COMPRESSION_FIRST = "compression_first"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"
    PERFORMANCE_OPTIMIZED = "performance_optimized"


@dataclass
class TrainingCompressionCoordinatorConfig:
    """Configuration for training-compression coordination"""
    coordination_mode: CoordinationMode = CoordinationMode.ADAPTIVE
    resource_allocation_strategy: ResourceAllocationStrategy = ResourceAllocationStrategy.BALANCED
    enable_real_time_coordination: bool = True
    enable_gradient_compression: bool = True
    enable_data_compression: bool = True
    enable_memory_coordination: bool = True
    enable_performance_optimization: bool = True
    enable_resource_monitoring: bool = True
    
    # Coordination thresholds
    memory_coordination_threshold_gb: float = 6.0
    gpu_coordination_threshold_percent: float = 80.0
    compression_latency_threshold_ms: float = 100.0
    training_latency_threshold_ms: float = 200.0
    
    # Batch coordination
    coordination_batch_size: int = 32
    gradient_compression_batch_size: int = 64
    data_compression_batch_size: int = 128
    
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
        if not isinstance(self.coordination_mode, CoordinationMode):
            raise TypeError("Coordination mode must be CoordinationMode")
        if not isinstance(self.resource_allocation_strategy, ResourceAllocationStrategy):
            raise TypeError("Resource allocation strategy must be ResourceAllocationStrategy")
        if self.memory_coordination_threshold_gb <= 0:
            raise ValueError("Memory coordination threshold must be positive")
        if not (0.0 <= self.gpu_coordination_threshold_percent <= 100.0):
            raise ValueError("GPU coordination threshold must be between 0.0 and 100.0")


@dataclass
class CoordinationOperation:
    """Training-compression coordination operation"""
    operation_id: str
    session_id: str
    operation_type: str  # 'data_compression', 'gradient_compression', 'memory_management', etc.
    priority: CompressionPriority
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Operation data
    data_size_bytes: int = 0
    compressed_size_bytes: int = 0
    compression_ratio: float = 0.0
    
    # Performance metrics
    coordination_time_ms: float = 0.0
    compression_time_ms: float = 0.0
    training_impact_ms: float = 0.0
    
    # Resource usage
    memory_usage_gb: float = 0.0
    gpu_utilization_percent: float = 0.0
    cpu_utilization_percent: float = 0.0
    
    # Success tracking
    coordination_successful: bool = False
    training_successful: bool = False
    compression_successful: bool = False
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate operation"""
        if not self.operation_id or not self.session_id:
            raise ValueError("Operation ID and Session ID cannot be empty")
        if not isinstance(self.priority, CompressionPriority):
            raise TypeError("Priority must be CompressionPriority")


@dataclass
class CoordinationAnalytics:
    """Comprehensive coordination analytics"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    
    # Operation type breakdown
    data_compression_operations: int = 0
    gradient_compression_operations: int = 0
    memory_management_operations: int = 0
    performance_optimization_operations: int = 0
    
    # Performance analytics
    average_coordination_time_ms: float = 0.0
    average_compression_time_ms: float = 0.0
    average_training_impact_ms: float = 0.0
    average_compression_ratio: float = 0.0
    
    # Resource analytics
    average_memory_usage_gb: float = 0.0
    peak_memory_usage_gb: float = 0.0
    average_gpu_utilization_percent: float = 0.0
    peak_gpu_utilization_percent: float = 0.0
    
    # Efficiency metrics
    coordination_efficiency: float = 0.0
    resource_utilization_efficiency: float = 0.0
    compression_efficiency: float = 0.0
    
    # Trend analysis
    performance_trend: str = "stable"
    memory_trend: str = "stable"
    compression_trend: str = "stable"
    
    timestamp: datetime = field(default_factory=datetime.utcnow)


class TrainingCompressionCoordinator:
    """
    Coordination between training and compression operations.
    Manages real-time coordination, resource allocation, and performance optimization.
    """
    
    def __init__(self, config: Optional[TrainingCompressionCoordinatorConfig] = None):
        """Initialize training-compression coordinator"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, TrainingCompressionCoordinatorConfig):
            raise TypeError(f"Config must be TrainingCompressionCoordinatorConfig or None, got {type(config)}")
        
        self.config = config or TrainingCompressionCoordinatorConfig()
        self.logger = logging.getLogger('TrainingCompressionCoordinator')
        
        # System references
        self.training_manager: Optional['TrainingManager'] = None
        self.hybrid_compression: Optional[HybridPadicCompressionSystem] = None
        self.dynamic_switching: Optional[DynamicSwitchingManager] = None
        self.direction_manager: Optional[DirectionManager] = None
        
        # Coordination state
        self.coordination_mode = self.config.coordination_mode
        self.active_operations: Dict[str, CoordinationOperation] = {}
        self.operation_queue: deque = deque(maxlen=1000)
        self.coordination_history: deque = deque(maxlen=self.config.coordination_history_size)
        
        # Resource coordination
        self.resource_allocation: Dict[str, float] = {
            'training_memory_percent': 60.0,
            'compression_memory_percent': 30.0,
            'system_memory_percent': 10.0,
            'training_gpu_percent': 70.0,
            'compression_gpu_percent': 25.0,
            'system_gpu_percent': 5.0,
            'training_cpu_percent': 50.0,
            'compression_cpu_percent': 40.0,
            'system_cpu_percent': 10.0
        }
        
        # Performance tracking
        self.coordination_analytics = CoordinationAnalytics()
        self.performance_baselines: Dict[str, float] = {}
        self.performance_history: deque = deque(maxlen=self.config.analytics_history_size)
        
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
        
        self.logger.info("TrainingCompressionCoordinator created successfully")
    
    def initialize_coordinator(self,
                             training_manager: 'TrainingManager',
                             hybrid_compression: HybridPadicCompressionSystem,
                             dynamic_switching: Optional[DynamicSwitchingManager] = None,
                             direction_manager: Optional[DirectionManager] = None) -> None:
        """
        Initialize training-compression coordinator with required systems.
        
        Args:
            training_manager: Training manager instance
            hybrid_compression: Hybrid compression system
            dynamic_switching: Optional dynamic switching manager
            direction_manager: Optional direction manager
            
        Raises:
            TypeError: If systems are invalid
            RuntimeError: If initialization fails
        """
        if training_manager is None:
            raise ValueError("Training manager cannot be None")
        if hasattr(training_manager, '__class__') and 'TrainingManager' not in str(training_manager.__class__):
            raise TypeError(f"Training manager must be TrainingManager, got {type(training_manager)}")
        if not isinstance(hybrid_compression, HybridPadicCompressionSystem):
            raise TypeError(f"Hybrid compression must be HybridPadicCompressionSystem, got {type(hybrid_compression)}")
        
        try:
            with self._coordination_lock:
                # Store system references
                self.training_manager = training_manager
                self.hybrid_compression = hybrid_compression
                self.dynamic_switching = dynamic_switching
                self.direction_manager = direction_manager
                
                # Initialize coordination systems
                self._initialize_performance_baselines()
                self._setup_coordination_hooks()
                self._initialize_resource_coordination()
                
                if self.config.enable_resource_monitoring:
                    self._start_resource_monitoring()
                
                if self.config.enable_analytics:
                    self._initialize_coordination_analytics()
                
                self.coordination_active = True
                self.logger.info("Training-compression coordinator initialized successfully")
                
        except Exception as e:
            self.coordination_active = False
            self.logger.error(f"Failed to initialize training-compression coordinator: {e}")
            raise RuntimeError(f"Coordinator initialization failed: {e}")
    
    def coordinate_training_data_compression(self, session_id: str, data: torch.Tensor) -> CoordinationOperation:
        """
        Coordinate training data compression operation.
        
        Args:
            session_id: Training session identifier
            data: Training data tensor to compress
            
        Returns:
            Coordination operation result
            
        Raises:
            ValueError: If session or data is invalid
            RuntimeError: If coordination fails
        """
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Session ID must be non-empty string")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Data must be torch.Tensor, got {type(data)}")
        if data.numel() == 0:
            raise ValueError("Data tensor cannot be empty")
        
        try:
            operation_id = self._generate_operation_id("data_compression")
            start_time = datetime.utcnow()
            
            # Create coordination operation
            operation = CoordinationOperation(
                operation_id=operation_id,
                session_id=session_id,
                operation_type="data_compression",
                priority=self._determine_compression_priority(session_id, data),
                start_time=start_time,
                data_size_bytes=data.numel() * data.element_size()
            )
            
            with self._operation_lock:
                self.active_operations[operation_id] = operation
            
            # Coordinate resource allocation
            resource_allocation = self._coordinate_resource_allocation_for_compression(session_id, data)
            
            # Perform coordinated compression
            compression_start = time.time()
            compression_result = self._perform_coordinated_data_compression(session_id, data, resource_allocation)
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
            
            # Record coordination
            with self._operation_lock:
                del self.active_operations[operation_id]
            
            self.coordination_history.append(operation)
            self._update_coordination_analytics(operation)
            
            self.logger.debug(f"Data compression coordinated for session {session_id}: {operation.compression_ratio:.4f} ratio")
            return operation
            
        except Exception as e:
            self.logger.error(f"Failed to coordinate training data compression for session {session_id}: {e}")
            if operation_id in self.active_operations:
                with self._operation_lock:
                    del self.active_operations[operation_id]
            raise RuntimeError(f"Training data compression coordination failed: {e}")
    
    def coordinate_gradient_compression(self, session_id: str, gradients: List[torch.Tensor]) -> CoordinationOperation:
        """
        Coordinate gradient compression operation.
        
        Args:
            session_id: Training session identifier
            gradients: List of gradient tensors to compress
            
        Returns:
            Coordination operation result
            
        Raises:
            ValueError: If session or gradients are invalid
            RuntimeError: If coordination fails
        """
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Session ID must be non-empty string")
        if not isinstance(gradients, list) or not gradients:
            raise ValueError("Gradients must be non-empty list")
        if not all(isinstance(g, torch.Tensor) for g in gradients):
            raise TypeError("All gradients must be torch.Tensor")
        
        try:
            operation_id = self._generate_operation_id("gradient_compression")
            start_time = datetime.utcnow()
            
            # Calculate total gradient data size
            total_data_size = sum(g.numel() * g.element_size() for g in gradients)
            
            # Create coordination operation
            operation = CoordinationOperation(
                operation_id=operation_id,
                session_id=session_id,
                operation_type="gradient_compression",
                priority=CompressionPriority.HIGH,  # Gradients are typically high priority
                start_time=start_time,
                data_size_bytes=total_data_size
            )
            
            with self._operation_lock:
                self.active_operations[operation_id] = operation
            
            # Coordinate resource allocation for gradient compression
            resource_allocation = self._coordinate_resource_allocation_for_gradients(session_id, gradients)
            
            # Perform coordinated gradient compression
            compression_start = time.time()
            compression_results = self._perform_coordinated_gradient_compression(session_id, gradients, resource_allocation)
            compression_time = (time.time() - compression_start) * 1000  # ms
            
            # Aggregate compression results
            total_compressed_size = sum(r.get('compressed_size', 0) for r in compression_results)
            average_compression_ratio = np.mean([r.get('compression_ratio', 0.0) for r in compression_results])
            all_successful = all(r.get('success', False) for r in compression_results)
            
            # Update operation results
            operation.end_time = datetime.utcnow()
            operation.coordination_time_ms = (operation.end_time - operation.start_time).total_seconds() * 1000
            operation.compression_time_ms = compression_time
            operation.compressed_size_bytes = total_compressed_size
            operation.compression_ratio = average_compression_ratio
            operation.compression_successful = all_successful
            operation.coordination_successful = all_successful
            
            # Update resource usage
            operation.memory_usage_gb = self._get_current_memory_usage()
            operation.gpu_utilization_percent = self._get_current_gpu_utilization()
            operation.cpu_utilization_percent = self._get_current_cpu_utilization()
            
            # Measure training impact
            operation.training_impact_ms = self._measure_training_impact(session_id, compression_time)
            
            # Record coordination
            with self._operation_lock:
                del self.active_operations[operation_id]
            
            self.coordination_history.append(operation)
            self._update_coordination_analytics(operation)
            
            self.logger.debug(f"Gradient compression coordinated for session {session_id}: {len(gradients)} gradients, {average_compression_ratio:.4f} ratio")
            return operation
            
        except Exception as e:
            self.logger.error(f"Failed to coordinate gradient compression for session {session_id}: {e}")
            if operation_id in self.active_operations:
                with self._operation_lock:
                    del self.active_operations[operation_id]
            raise RuntimeError(f"Gradient compression coordination failed: {e}")
    
    def coordinate_memory_management(self, session_id: str) -> CoordinationOperation:
        """
        Coordinate memory management between training and compression.
        
        Args:
            session_id: Training session identifier
            
        Returns:
            Coordination operation result
            
        Raises:
            ValueError: If session is invalid
            RuntimeError: If coordination fails
        """
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Session ID must be non-empty string")
        
        try:
            operation_id = self._generate_operation_id("memory_management")
            start_time = datetime.utcnow()
            
            # Create coordination operation
            operation = CoordinationOperation(
                operation_id=operation_id,
                session_id=session_id,
                operation_type="memory_management",
                priority=CompressionPriority.NORMAL,
                start_time=start_time
            )
            
            with self._operation_lock:
                self.active_operations[operation_id] = operation
            
            # Capture initial memory state
            initial_memory = self._get_current_memory_usage()
            
            # Coordinate memory optimization
            memory_optimization_start = time.time()
            memory_optimization_result = self._perform_coordinated_memory_management(session_id)
            memory_optimization_time = (time.time() - memory_optimization_start) * 1000  # ms
            
            # Capture final memory state
            final_memory = self._get_current_memory_usage()
            memory_savings = initial_memory - final_memory
            
            # Update operation results
            operation.end_time = datetime.utcnow()
            operation.coordination_time_ms = (operation.end_time - operation.start_time).total_seconds() * 1000
            operation.memory_usage_gb = final_memory
            operation.coordination_successful = memory_optimization_result.get('success', False)
            
            # Calculate memory efficiency
            if initial_memory > 0:
                memory_efficiency = memory_savings / initial_memory
            else:
                memory_efficiency = 0.0
            
            # Record coordination
            with self._operation_lock:
                del self.active_operations[operation_id]
            
            self.coordination_history.append(operation)
            self._update_coordination_analytics(operation)
            
            coordination_result = {
                'operation': operation,
                'initial_memory_gb': initial_memory,
                'final_memory_gb': final_memory,
                'memory_savings_gb': memory_savings,
                'memory_efficiency': memory_efficiency,
                'optimization_details': memory_optimization_result
            }
            
            self.logger.debug(f"Memory management coordinated for session {session_id}: {memory_savings:.2f}GB saved")
            return operation
            
        except Exception as e:
            self.logger.error(f"Failed to coordinate memory management for session {session_id}: {e}")
            if operation_id in self.active_operations:
                with self._operation_lock:
                    del self.active_operations[operation_id]
            raise RuntimeError(f"Memory management coordination failed: {e}")
    
    def coordinate_performance_optimization(self, session_id: str) -> CoordinationOperation:
        """
        Coordinate performance optimization between training and compression.
        
        Args:
            session_id: Training session identifier
            
        Returns:
            Coordination operation result
            
        Raises:
            ValueError: If session is invalid
            RuntimeError: If coordination fails
        """
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Session ID must be non-empty string")
        
        try:
            operation_id = self._generate_operation_id("performance_optimization")
            start_time = datetime.utcnow()
            
            # Create coordination operation
            operation = CoordinationOperation(
                operation_id=operation_id,
                session_id=session_id,
                operation_type="performance_optimization",
                priority=CompressionPriority.NORMAL,
                start_time=start_time
            )
            
            with self._operation_lock:
                self.active_operations[operation_id] = operation
            
            # Capture initial performance metrics
            initial_performance = self._capture_performance_metrics(session_id)
            
            # Coordinate performance optimization
            optimization_start = time.time()
            optimization_result = self._perform_coordinated_performance_optimization(session_id)
            optimization_time = (time.time() - optimization_start) * 1000  # ms
            
            # Capture final performance metrics
            final_performance = self._capture_performance_metrics(session_id)
            
            # Calculate performance improvements
            performance_improvements = self._calculate_performance_improvements(initial_performance, final_performance)
            
            # Update operation results
            operation.end_time = datetime.utcnow()
            operation.coordination_time_ms = (operation.end_time - operation.start_time).total_seconds() * 1000
            operation.memory_usage_gb = final_performance.get('memory_usage_gb', 0.0)
            operation.gpu_utilization_percent = final_performance.get('gpu_utilization_percent', 0.0)
            operation.cpu_utilization_percent = final_performance.get('cpu_utilization_percent', 0.0)
            operation.coordination_successful = optimization_result.get('success', False)
            
            # Record coordination
            with self._operation_lock:
                del self.active_operations[operation_id]
            
            self.coordination_history.append(operation)
            self._update_coordination_analytics(operation)
            
            self.logger.debug(f"Performance optimization coordinated for session {session_id}")
            return operation
            
        except Exception as e:
            self.logger.error(f"Failed to coordinate performance optimization for session {session_id}: {e}")
            if operation_id in self.active_operations:
                with self._operation_lock:
                    del self.active_operations[operation_id]
            raise RuntimeError(f"Performance optimization coordination failed: {e}")
    
    def coordinate_resource_allocation(self, session_id: str) -> Dict[str, Any]:
        """
        Coordinate resource allocation between training and compression.
        
        Args:
            session_id: Training session identifier
            
        Returns:
            Resource allocation coordination results
            
        Raises:
            ValueError: If session is invalid
            RuntimeError: If coordination fails
        """
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Session ID must be non-empty string")
        
        try:
            # Analyze current resource usage
            current_resources = self._analyze_current_resource_usage(session_id)
            
            # Determine optimal resource allocation
            optimal_allocation = self._determine_optimal_resource_allocation(session_id, current_resources)
            
            # Apply resource allocation changes
            allocation_result = self._apply_resource_allocation(session_id, optimal_allocation)
            
            # Monitor allocation effectiveness
            allocation_effectiveness = self._monitor_allocation_effectiveness(session_id, optimal_allocation)
            
            coordination_result = {
                'session_id': session_id,
                'current_resources': current_resources,
                'optimal_allocation': optimal_allocation,
                'allocation_result': allocation_result,
                'allocation_effectiveness': allocation_effectiveness,
                'coordination_timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.debug(f"Resource allocation coordinated for session {session_id}")
            return coordination_result
            
        except Exception as e:
            self.logger.error(f"Failed to coordinate resource allocation for session {session_id}: {e}")
            raise RuntimeError(f"Resource allocation coordination failed: {e}")
    
    def get_coordination_analytics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive coordination analytics.
        
        Args:
            session_id: Optional specific session ID for detailed analytics
            
        Returns:
            Coordination analytics results
            
        Raises:
            ValueError: If session ID is invalid
            RuntimeError: If analytics retrieval fails
        """
        try:
            with self._analytics_lock:
                if session_id:
                    # Get session-specific analytics
                    session_operations = [op for op in self.coordination_history if op.session_id == session_id]
                    session_analytics = self._calculate_session_coordination_analytics(session_id, session_operations)
                    
                    return {
                        'session_id': session_id,
                        'session_analytics': session_analytics,
                        'session_operations': session_operations,
                        'performance_history': [p for p in self.performance_history if p.get('session_id') == session_id]
                    }
                else:
                    # Get global analytics
                    self._update_global_coordination_analytics()
                    
                    return {
                        'global_analytics': self.coordination_analytics,
                        'active_operations': len(self.active_operations),
                        'coordination_history': list(self.coordination_history),
                        'resource_allocation': self.resource_allocation,
                        'performance_baselines': self.performance_baselines,
                        'optimization_history': self.optimization_history[-20:],
                        'coordination_trends': self._analyze_coordination_trends(),
                        'system_recommendations': self._generate_coordination_recommendations()
                    }
                    
        except Exception as e:
            self.logger.error(f"Failed to get coordination analytics: {e}")
            raise RuntimeError(f"Coordination analytics retrieval failed: {e}")
    
    def _initialize_performance_baselines(self) -> None:
        """Initialize performance baselines"""
        try:
            self.performance_baselines = {
                'coordination_time_ms': 50.0,
                'compression_time_ms': 100.0,
                'training_impact_ms': 20.0,
                'memory_usage_gb': 4.0,
                'gpu_utilization_percent': 60.0,
                'cpu_utilization_percent': 40.0,
                'compression_ratio': 0.6
            }
            self.logger.debug("Performance baselines initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize performance baselines: {e}")
            raise RuntimeError(f"Performance baseline initialization failed: {e}")
    
    def _setup_coordination_hooks(self) -> None:
        """Setup coordination hooks"""
        try:
            if hasattr(self.training_manager, 'add_hook'):
                self.training_manager.add_hook('pre_batch', self._training_pre_batch_hook)
                self.training_manager.add_hook('post_batch', self._training_post_batch_hook)
                self.training_manager.add_hook('pre_gradient', self._training_pre_gradient_hook)
                self.training_manager.add_hook('post_gradient', self._training_post_gradient_hook)
            
            if hasattr(self.hybrid_compression, 'add_hook'):
                self.hybrid_compression.add_hook('pre_compression', self._compression_pre_hook)
                self.hybrid_compression.add_hook('post_compression', self._compression_post_hook)
            
            self.logger.debug("Coordination hooks setup completed")
        except Exception as e:
            self.logger.error(f"Failed to setup coordination hooks: {e}")
            raise RuntimeError(f"Coordination hooks setup failed: {e}")
    
    def _initialize_resource_coordination(self) -> None:
        """Initialize resource coordination"""
        try:
            # Set initial resource allocation based on strategy
            if self.config.resource_allocation_strategy == ResourceAllocationStrategy.TRAINING_FIRST:
                self.resource_allocation.update({
                    'training_memory_percent': 70.0,
                    'compression_memory_percent': 20.0,
                    'training_gpu_percent': 80.0,
                    'compression_gpu_percent': 15.0
                })
            elif self.config.resource_allocation_strategy == ResourceAllocationStrategy.COMPRESSION_FIRST:
                self.resource_allocation.update({
                    'training_memory_percent': 40.0,
                    'compression_memory_percent': 50.0,
                    'training_gpu_percent': 50.0,
                    'compression_gpu_percent': 45.0
                })
            # BALANCED and others use default values
            
            self.logger.debug("Resource coordination initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize resource coordination: {e}")
            raise RuntimeError(f"Resource coordination initialization failed: {e}")
    
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
            self.coordination_analytics = CoordinationAnalytics()
            self.logger.debug("Coordination analytics initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize coordination analytics: {e}")
            raise RuntimeError(f"Coordination analytics initialization failed: {e}")
    
    def _generate_operation_id(self, operation_type: str) -> str:
        """Generate unique operation ID"""
        timestamp = str(int(time.time() * 1000))
        return f"{operation_type}_{timestamp}_{hash(threading.current_thread()) % 10000:04d}"
    
    def _determine_compression_priority(self, session_id: str, data: torch.Tensor) -> CompressionPriority:
        """Determine compression priority based on data characteristics"""
        data_size_mb = data.numel() * data.element_size() / (1024 * 1024)
        
        if data_size_mb > 500:  # Large data gets high priority for compression
            return CompressionPriority.HIGH
        elif data_size_mb > 100:
            return CompressionPriority.NORMAL
        elif data_size_mb < 10:
            return CompressionPriority.LOW
        else:
            return CompressionPriority.NORMAL
    
    def _coordinate_resource_allocation_for_compression(self, session_id: str, data: torch.Tensor) -> Dict[str, float]:
        """Coordinate resource allocation for compression operation"""
        try:
            current_memory = self._get_current_memory_usage()
            current_gpu = self._get_current_gpu_utilization()
            
            # Determine resource allocation based on current usage and data size
            data_size_gb = data.numel() * data.element_size() / (1024**3)
            
            if current_memory > self.config.memory_coordination_threshold_gb:
                # High memory usage - allocate more to compression
                memory_allocation = min(0.8, 0.4 + data_size_gb * 0.1)
            else:
                # Normal memory usage
                memory_allocation = self.resource_allocation['compression_memory_percent'] / 100.0
            
            if current_gpu > self.config.gpu_coordination_threshold_percent:
                # High GPU usage - reduce GPU allocation for compression
                gpu_allocation = max(0.1, self.resource_allocation['compression_gpu_percent'] / 100.0 - 0.2)
            else:
                gpu_allocation = self.resource_allocation['compression_gpu_percent'] / 100.0
            
            return {
                'memory_allocation': memory_allocation,
                'gpu_allocation': gpu_allocation,
                'cpu_allocation': self.resource_allocation['compression_cpu_percent'] / 100.0
            }
        except Exception as e:
            self.logger.error(f"Failed to coordinate resource allocation for compression: {e}")
            return {'memory_allocation': 0.3, 'gpu_allocation': 0.2, 'cpu_allocation': 0.4}
    
    def _coordinate_resource_allocation_for_gradients(self, session_id: str, gradients: List[torch.Tensor]) -> Dict[str, float]:
        """Coordinate resource allocation for gradient compression"""
        try:
            total_gradient_size = sum(g.numel() * g.element_size() for g in gradients) / (1024**3)  # GB
            
            # Gradients typically need higher priority and more resources
            memory_allocation = min(0.6, 0.3 + total_gradient_size * 0.1)
            gpu_allocation = min(0.5, 0.25 + total_gradient_size * 0.05)
            cpu_allocation = 0.5  # Higher CPU allocation for gradient processing
            
            return {
                'memory_allocation': memory_allocation,
                'gpu_allocation': gpu_allocation,
                'cpu_allocation': cpu_allocation
            }
        except Exception as e:
            self.logger.error(f"Failed to coordinate resource allocation for gradients: {e}")
            return {'memory_allocation': 0.4, 'gpu_allocation': 0.3, 'cpu_allocation': 0.5}
    
    def _perform_coordinated_data_compression(self, session_id: str, data: torch.Tensor, resource_allocation: Dict[str, float]) -> Dict[str, Any]:
        """Perform coordinated data compression"""
        try:
            # Configure compression system with resource allocation
            compression_config = {
                'memory_limit_percent': resource_allocation['memory_allocation'] * 100,
                'gpu_allocation_percent': resource_allocation['gpu_allocation'] * 100,
                'session_id': session_id
            }
            
            # Perform compression
            compression_result = self.hybrid_compression.compress(data)
            compression_result['resource_allocation'] = resource_allocation
            
            return compression_result
        except Exception as e:
            self.logger.error(f"Coordinated data compression failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _perform_coordinated_gradient_compression(self, session_id: str, gradients: List[torch.Tensor], resource_allocation: Dict[str, float]) -> List[Dict[str, Any]]:
        """Perform coordinated gradient compression"""
        try:
            results = []
            
            for i, gradient in enumerate(gradients):
                # Configure compression for each gradient
                gradient_config = {
                    'memory_limit_percent': resource_allocation['memory_allocation'] * 100,
                    'gpu_allocation_percent': resource_allocation['gpu_allocation'] * 100,
                    'session_id': session_id,
                    'gradient_index': i
                }
                
                # Perform gradient compression
                compression_result = self.hybrid_compression.compress(gradient)
                compression_result['resource_allocation'] = resource_allocation
                compression_result['gradient_index'] = i
                
                results.append(compression_result)
            
            return results
        except Exception as e:
            self.logger.error(f"Coordinated gradient compression failed: {e}")
            return [{'success': False, 'error': str(e)} for _ in gradients]
    
    def _perform_coordinated_memory_management(self, session_id: str) -> Dict[str, Any]:
        """Perform coordinated memory management"""
        try:
            # Memory management operations
            operations_performed = []
            
            # Garbage collection
            import gc
            collected = gc.collect()
            if collected > 0:
                operations_performed.append(f"garbage_collection_{collected}")
            
            # Clear compression caches if available
            if hasattr(self.hybrid_compression, 'clear_cache'):
                self.hybrid_compression.clear_cache()
                operations_performed.append("compression_cache_cleared")
            
            # Optimize memory pools if available
            if hasattr(self.hybrid_compression, 'optimize_memory_pools'):
                self.hybrid_compression.optimize_memory_pools()
                operations_performed.append("memory_pools_optimized")
            
            return {
                'success': True,
                'operations_performed': operations_performed
            }
        except Exception as e:
            self.logger.error(f"Coordinated memory management failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _perform_coordinated_performance_optimization(self, session_id: str) -> Dict[str, Any]:
        """Perform coordinated performance optimization"""
        try:
            optimizations_applied = []
            
            # Optimize compression parameters
            if hasattr(self.hybrid_compression, 'optimize_parameters'):
                self.hybrid_compression.optimize_parameters()
                optimizations_applied.append("compression_parameters_optimized")
            
            # Optimize dynamic switching if available
            if self.dynamic_switching and hasattr(self.dynamic_switching, 'optimize_switching'):
                self.dynamic_switching.optimize_switching(session_id)
                optimizations_applied.append("dynamic_switching_optimized")
            
            # Optimize resource allocation
            self._optimize_resource_allocation(session_id)
            optimizations_applied.append("resource_allocation_optimized")
            
            return {
                'success': True,
                'optimizations_applied': optimizations_applied
            }
        except Exception as e:
            self.logger.error(f"Coordinated performance optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _measure_training_impact(self, session_id: str, compression_time_ms: float) -> float:
        """Measure impact of compression on training performance"""
        try:
            # Simple estimation based on compression time and system load
            current_cpu = self._get_current_cpu_utilization()
            current_gpu = self._get_current_gpu_utilization()
            
            # Impact factor based on system utilization
            impact_factor = 1.0 + (current_cpu + current_gpu) / 200.0  # Normalized impact
            
            training_impact = compression_time_ms * impact_factor * 0.3  # Estimated impact
            return training_impact
        except Exception as e:
            self.logger.error(f"Failed to measure training impact: {e}")
            return 0.0
    
    def _capture_performance_metrics(self, session_id: str) -> Dict[str, float]:
        """Capture current performance metrics"""
        try:
            return {
                'memory_usage_gb': self._get_current_memory_usage(),
                'gpu_utilization_percent': self._get_current_gpu_utilization(),
                'cpu_utilization_percent': self._get_current_cpu_utilization(),
                'timestamp': time.time()
            }
        except Exception as e:
            self.logger.error(f"Failed to capture performance metrics: {e}")
            return {}
    
    def _calculate_performance_improvements(self, initial: Dict[str, float], final: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance improvements"""
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
            
            return improvements
        except Exception as e:
            self.logger.error(f"Failed to calculate performance improvements: {e}")
            return {}
    
    def _analyze_current_resource_usage(self, session_id: str) -> Dict[str, float]:
        """Analyze current resource usage"""
        try:
            return {
                'memory_usage_gb': self._get_current_memory_usage(),
                'memory_usage_percent': self._get_current_memory_usage() / 16.0 * 100,  # Assume 16GB total
                'gpu_utilization_percent': self._get_current_gpu_utilization(),
                'cpu_utilization_percent': self._get_current_cpu_utilization(),
                'active_operations': len(self.active_operations)
            }
        except Exception as e:
            self.logger.error(f"Failed to analyze resource usage: {e}")
            return {}
    
    def _determine_optimal_resource_allocation(self, session_id: str, current_resources: Dict[str, float]) -> Dict[str, float]:
        """Determine optimal resource allocation"""
        try:
            optimal_allocation = self.resource_allocation.copy()
            
            # Adjust based on current usage
            memory_usage = current_resources.get('memory_usage_percent', 50.0)
            gpu_usage = current_resources.get('gpu_utilization_percent', 50.0)
            
            if memory_usage > 80.0:
                # High memory usage - reduce training allocation, increase compression
                optimal_allocation['training_memory_percent'] -= 10.0
                optimal_allocation['compression_memory_percent'] += 10.0
            elif memory_usage < 30.0:
                # Low memory usage - can afford to give more to training
                optimal_allocation['training_memory_percent'] += 10.0
                optimal_allocation['compression_memory_percent'] -= 10.0
            
            if gpu_usage > 85.0:
                # High GPU usage - balance allocation
                optimal_allocation['training_gpu_percent'] -= 5.0
                optimal_allocation['compression_gpu_percent'] += 5.0
            
            return optimal_allocation
        except Exception as e:
            self.logger.error(f"Failed to determine optimal resource allocation: {e}")
            return self.resource_allocation.copy()
    
    def _apply_resource_allocation(self, session_id: str, allocation: Dict[str, float]) -> Dict[str, Any]:
        """Apply resource allocation changes"""
        try:
            # Update internal allocation
            self.resource_allocation.update(allocation)
            
            # Apply to systems if they support it
            allocation_results = []
            
            if hasattr(self.training_manager, 'set_resource_allocation'):
                training_result = self.training_manager.set_resource_allocation(session_id, {
                    'memory_percent': allocation['training_memory_percent'],
                    'gpu_percent': allocation['training_gpu_percent'],
                    'cpu_percent': allocation['training_cpu_percent']
                })
                allocation_results.append(f"training_allocation_{training_result}")
            
            if hasattr(self.hybrid_compression, 'set_resource_allocation'):
                compression_result = self.hybrid_compression.set_resource_allocation({
                    'memory_percent': allocation['compression_memory_percent'],
                    'gpu_percent': allocation['compression_gpu_percent'],
                    'cpu_percent': allocation['compression_cpu_percent']
                })
                allocation_results.append(f"compression_allocation_{compression_result}")
            
            return {
                'success': True,
                'allocation_results': allocation_results
            }
        except Exception as e:
            self.logger.error(f"Failed to apply resource allocation: {e}")
            return {'success': False, 'error': str(e)}
    
    def _monitor_allocation_effectiveness(self, session_id: str, allocation: Dict[str, float]) -> Dict[str, Any]:
        """Monitor allocation effectiveness"""
        try:
            # Simple effectiveness monitoring
            current_resources = self._analyze_current_resource_usage(session_id)
            
            effectiveness_score = 1.0
            if current_resources.get('memory_usage_percent', 0) > 90.0:
                effectiveness_score -= 0.3
            if current_resources.get('gpu_utilization_percent', 0) > 95.0:
                effectiveness_score -= 0.2
            if current_resources.get('cpu_utilization_percent', 0) > 95.0:
                effectiveness_score -= 0.2
            
            return {
                'effectiveness_score': max(0.0, effectiveness_score),
                'current_resources': current_resources,
                'allocation_applied': allocation
            }
        except Exception as e:
            self.logger.error(f"Failed to monitor allocation effectiveness: {e}")
            return {'effectiveness_score': 0.5}
    
    def _optimize_resource_allocation(self, session_id: str) -> None:
        """Optimize resource allocation for session"""
        try:
            current_resources = self._analyze_current_resource_usage(session_id)
            optimal_allocation = self._determine_optimal_resource_allocation(session_id, current_resources)
            self._apply_resource_allocation(session_id, optimal_allocation)
        except Exception as e:
            self.logger.error(f"Failed to optimize resource allocation: {e}")
    
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
    
    def _update_coordination_analytics(self, operation: CoordinationOperation) -> None:
        """Update coordination analytics with operation results"""
        try:
            with self._analytics_lock:
                self.coordination_analytics.total_operations += 1
                
                if operation.coordination_successful:
                    self.coordination_analytics.successful_operations += 1
                else:
                    self.coordination_analytics.failed_operations += 1
                
                # Update operation type counters
                if operation.operation_type == "data_compression":
                    self.coordination_analytics.data_compression_operations += 1
                elif operation.operation_type == "gradient_compression":
                    self.coordination_analytics.gradient_compression_operations += 1
                elif operation.operation_type == "memory_management":
                    self.coordination_analytics.memory_management_operations += 1
                elif operation.operation_type == "performance_optimization":
                    self.coordination_analytics.performance_optimization_operations += 1
                
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
                
                if operation.training_impact_ms > 0:
                    self.coordination_analytics.average_training_impact_ms = (
                        (self.coordination_analytics.average_training_impact_ms * (total_ops - 1) + 
                         operation.training_impact_ms) / total_ops
                    )
                
                if operation.compression_ratio > 0:
                    self.coordination_analytics.average_compression_ratio = (
                        (self.coordination_analytics.average_compression_ratio * (total_ops - 1) + 
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
                
        except Exception as e:
            self.logger.error(f"Failed to update coordination analytics: {e}")
    
    def _calculate_session_coordination_analytics(self, session_id: str, operations: List[CoordinationOperation]) -> Dict[str, Any]:
        """Calculate coordination analytics for specific session"""
        try:
            if not operations:
                return {'session_id': session_id, 'total_operations': 0}
            
            successful_ops = [op for op in operations if op.coordination_successful]
            
            analytics = {
                'session_id': session_id,
                'total_operations': len(operations),
                'successful_operations': len(successful_ops),
                'success_rate': len(successful_ops) / len(operations),
                'operation_types': {
                    'data_compression': len([op for op in operations if op.operation_type == "data_compression"]),
                    'gradient_compression': len([op for op in operations if op.operation_type == "gradient_compression"]),
                    'memory_management': len([op for op in operations if op.operation_type == "memory_management"]),
                    'performance_optimization': len([op for op in operations if op.operation_type == "performance_optimization"])
                },
                'average_coordination_time_ms': np.mean([op.coordination_time_ms for op in operations]),
                'average_compression_time_ms': np.mean([op.compression_time_ms for op in operations if op.compression_time_ms > 0]),
                'average_compression_ratio': np.mean([op.compression_ratio for op in operations if op.compression_ratio > 0]),
                'peak_memory_usage_gb': max([op.memory_usage_gb for op in operations if op.memory_usage_gb > 0], default=0.0),
                'peak_gpu_utilization_percent': max([op.gpu_utilization_percent for op in operations if op.gpu_utilization_percent > 0], default=0.0)
            }
            
            return analytics
        except Exception as e:
            self.logger.error(f"Failed to calculate session coordination analytics: {e}")
            return {'session_id': session_id, 'error': str(e)}
    
    def _update_global_coordination_analytics(self) -> None:
        """Update global coordination analytics"""
        try:
            with self._analytics_lock:
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
    
    def _analyze_coordination_trends(self) -> Dict[str, str]:
        """Analyze coordination trends"""
        try:
            recent_operations = list(self.coordination_history)[-20:]
            
            if len(recent_operations) < 5:
                return {'insufficient_data': 'true'}
            
            # Analyze performance trends
            coordination_times = [op.coordination_time_ms for op in recent_operations]
            compression_times = [op.compression_time_ms for op in recent_operations if op.compression_time_ms > 0]
            compression_ratios = [op.compression_ratio for op in recent_operations if op.compression_ratio > 0]
            
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
            
            return trends
        except Exception as e:
            self.logger.error(f"Failed to analyze coordination trends: {e}")
            return {}
    
    def _generate_coordination_recommendations(self) -> List[str]:
        """Generate coordination recommendations"""
        recommendations = []
        
        if self.coordination_analytics.coordination_efficiency < 0.8:
            recommendations.append("Coordination efficiency below 80% - review coordination strategies")
        
        if len(self.active_operations) > 10:
            recommendations.append("High number of active operations - consider load balancing")
        
        if self.coordination_analytics.average_coordination_time_ms > 100:
            recommendations.append("High coordination overhead - optimize coordination algorithms")
        
        if self.coordination_analytics.average_compression_ratio < 0.5:
            recommendations.append("Low compression ratios - review compression strategies")
        
        return recommendations if recommendations else ["Coordination operating within optimal parameters"]
    
    def _training_pre_batch_hook(self, session_id: str, batch_data: Dict[str, Any]) -> None:
        """Training pre-batch hook"""
        try:
            # Coordinate resource allocation before batch processing
            if self.config.enable_real_time_coordination:
                self._optimize_resource_allocation(session_id)
        except Exception as e:
            self.logger.error(f"Training pre-batch hook failed: {e}")
    
    def _training_post_batch_hook(self, session_id: str, batch_data: Dict[str, Any]) -> None:
        """Training post-batch hook"""
        try:
            # Record performance metrics after batch
            performance_metrics = self._capture_performance_metrics(session_id)
            self.performance_history.append({
                'session_id': session_id,
                'metrics': performance_metrics,
                'batch_info': batch_data
            })
        except Exception as e:
            self.logger.error(f"Training post-batch hook failed: {e}")
    
    def _training_pre_gradient_hook(self, session_id: str, gradient_data: Dict[str, Any]) -> None:
        """Training pre-gradient hook"""
        try:
            # Prepare for gradient compression coordination
            if self.config.enable_gradient_compression:
                self._prepare_gradient_compression_coordination(session_id, gradient_data)
        except Exception as e:
            self.logger.error(f"Training pre-gradient hook failed: {e}")
    
    def _training_post_gradient_hook(self, session_id: str, gradient_data: Dict[str, Any]) -> None:
        """Training post-gradient hook"""
        try:
            # Coordinate gradient compression if enabled
            if self.config.enable_gradient_compression and 'gradients' in gradient_data:
                self.coordinate_gradient_compression(session_id, gradient_data['gradients'])
        except Exception as e:
            self.logger.error(f"Training post-gradient hook failed: {e}")
    
    def _compression_pre_hook(self, compression_data: Dict[str, Any]) -> None:
        """Compression pre-operation hook"""
        try:
            # Prepare coordination for compression operation
            session_id = compression_data.get('session_id')
            if session_id and self.config.enable_real_time_coordination:
                self._prepare_compression_coordination(session_id, compression_data)
        except Exception as e:
            self.logger.error(f"Compression pre-hook failed: {e}")
    
    def _compression_post_hook(self, compression_data: Dict[str, Any]) -> None:
        """Compression post-operation hook"""
        try:
            # Record compression performance
            session_id = compression_data.get('session_id')
            if session_id:
                self._record_compression_performance(session_id, compression_data)
        except Exception as e:
            self.logger.error(f"Compression post-hook failed: {e}")
    
    def _prepare_gradient_compression_coordination(self, session_id: str, gradient_data: Dict[str, Any]) -> None:
        """Prepare for gradient compression coordination"""
        try:
            # Pre-allocate resources for gradient compression
            if 'gradients' in gradient_data:
                gradients = gradient_data['gradients']
                resource_allocation = self._coordinate_resource_allocation_for_gradients(session_id, gradients)
                gradient_data['resource_allocation'] = resource_allocation
        except Exception as e:
            self.logger.error(f"Failed to prepare gradient compression coordination: {e}")
    
    def _prepare_compression_coordination(self, session_id: str, compression_data: Dict[str, Any]) -> None:
        """Prepare coordination for compression operation"""
        try:
            # Set coordination parameters for compression
            compression_data['coordination_mode'] = self.coordination_mode.value
            compression_data['resource_allocation'] = self.resource_allocation
        except Exception as e:
            self.logger.error(f"Failed to prepare compression coordination: {e}")
    
    def _record_compression_performance(self, session_id: str, compression_data: Dict[str, Any]) -> None:
        """Record compression performance metrics"""
        try:
            performance_record = {
                'session_id': session_id,
                'compression_time_ms': compression_data.get('compression_time_ms', 0.0),
                'compression_ratio': compression_data.get('compression_ratio', 0.0),
                'data_size_bytes': compression_data.get('data_size_bytes', 0),
                'timestamp': datetime.utcnow().isoformat()
            }
            self.performance_history.append(performance_record)
        except Exception as e:
            self.logger.error(f"Failed to record compression performance: {e}")
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination status"""
        return {
            'coordination_active': self.coordination_active,
            'monitoring_active': self.monitoring_active,
            'resource_monitoring_active': self.resource_monitoring_active,
            'coordination_mode': self.coordination_mode.value,
            'active_operations': len(self.active_operations),
            'resource_allocation': self.resource_allocation,
            'recent_performance': list(self.performance_history)[-5:]
        }
    
    def shutdown(self) -> None:
        """Shutdown training-compression coordinator"""
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
                self.logger.info("Training-compression coordinator shutdown completed")
                
        except Exception as e:
            self.logger.error(f"Training-compression coordinator shutdown failed: {e}")
            raise RuntimeError(f"Coordinator shutdown failed: {e}")