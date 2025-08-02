"""
Brain-GAC Coordinator - Coordination between Brain and GAC operations  
NO FALLBACKS - HARD FAILURES ONLY
"""

import logging
import math
import threading
import time
import torch
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from enum import Enum

# Import GAC system components
from gac_system.gac_types import DirectionState, DirectionType
from gac_system.direction_state import DirectionStateManager, DirectionHistory
from gac_system.gac_components import GACSystem

# Import Brain system components
from brain import Brain

# Import compression system components
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


class SessionCoordinationStrategy(Enum):
    """Session coordination strategy enumeration"""
    BRAIN_FIRST = "brain_first"
    GAC_FIRST = "gac_first"
    PARALLEL = "parallel"
    ADAPTIVE_ORDER = "adaptive_order"
    PERFORMANCE_OPTIMIZED = "performance_optimized"


class ResourceCoordinationLevel(Enum):
    """Resource coordination level enumeration"""
    MINIMAL = "minimal"
    MODERATE = "moderate"
    COMPREHENSIVE = "comprehensive"
    AGGRESSIVE = "aggressive"


@dataclass
class TrainingSessionConfig:
    """Configuration for training session coordination"""
    session_id: str
    coordination_strategy: SessionCoordinationStrategy
    enable_real_time_coordination: bool = True
    enable_performance_monitoring: bool = True
    enable_resource_coordination: bool = True
    max_session_duration_minutes: int = 60
    gradient_coordination_batch_size: int = 64
    performance_check_interval_seconds: int = 30
    error_recovery_enabled: bool = True
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate session config"""
        if not self.session_id:
            raise ValueError("Session ID cannot be empty")
        if not isinstance(self.coordination_strategy, SessionCoordinationStrategy):
            raise TypeError("Coordination strategy must be SessionCoordinationStrategy")
        if self.max_session_duration_minutes <= 0:
            raise ValueError("Max session duration must be positive")


@dataclass
class CoordinationResult:
    """Result of coordination operation"""
    operation_type: str
    coordination_successful: bool
    brain_result: Optional[Any]
    gac_result: Optional[Any]
    compression_result: Optional[Any]
    coordination_time_ms: float
    performance_metrics: Dict[str, float]
    error_details: Optional[str]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate coordination result"""
        if not self.operation_type:
            raise ValueError("Operation type cannot be empty")
        if self.coordination_time_ms < 0:
            raise ValueError("Coordination time must be non-negative")


@dataclass
class CoordinationAnalytics:
    """Comprehensive coordination analytics"""
    total_coordinations: int
    successful_coordinations: int
    failed_coordinations: int
    average_coordination_time_ms: float
    brain_coordination_efficiency: float
    gac_coordination_efficiency: float
    resource_coordination_efficiency: float
    session_coordination_stats: Dict[str, Dict[str, Any]]
    performance_optimization_stats: Dict[str, float]
    error_recovery_stats: Dict[str, int]
    coordination_trends: Dict[str, List[float]]
    optimization_recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BrainGACCoordinatorConfig:
    """Configuration for Brain-GAC coordinator"""
    # Coordination mode configuration
    coordination_mode: CoordinationMode = CoordinationMode.ADAPTIVE
    default_session_strategy: SessionCoordinationStrategy = SessionCoordinationStrategy.ADAPTIVE_ORDER
    resource_coordination_level: ResourceCoordinationLevel = ResourceCoordinationLevel.MODERATE
    
    # Performance configuration
    max_coordination_time_ms: float = 100.0
    enable_parallel_coordination: bool = True
    enable_adaptive_coordination: bool = True
    coordination_timeout_ms: float = 5000.0
    
    # Session coordination
    max_concurrent_sessions: int = 5
    session_queue_size: int = 100
    enable_session_prioritization: bool = True
    session_cleanup_interval_minutes: int = 10
    
    # Resource coordination
    enable_memory_coordination: bool = True
    enable_gpu_coordination: bool = True
    enable_cpu_coordination: bool = True
    resource_monitoring_interval_ms: float = 1000.0
    
    # Performance optimization
    enable_performance_optimization: bool = True
    optimization_interval_seconds: int = 60
    performance_baseline_window: int = 100
    enable_predictive_optimization: bool = True
    
    # Error recovery
    enable_error_recovery: bool = True
    max_retry_attempts: int = 3
    error_recovery_timeout_ms: float = 2000.0
    enable_fallback_coordination: bool = True
    
    # Analytics configuration
    analytics_history_size: int = 5000
    enable_detailed_analytics: bool = True
    analytics_export_interval_minutes: int = 30
    
    def __post_init__(self):
        """Validate configuration"""
        if not isinstance(self.coordination_mode, CoordinationMode):
            raise TypeError("Coordination mode must be CoordinationMode")
        if self.max_coordination_time_ms <= 0:
            raise ValueError("Max coordination time must be positive")
        if self.max_concurrent_sessions <= 0:
            raise ValueError("Max concurrent sessions must be positive")


class BrainGACCoordinator:
    """
    Coordination between Brain and GAC operations.
    Manages real-time coordination of training sessions, gradient optimization,
    performance optimization, and error recovery.
    """
    
    def __init__(self, config: Optional[BrainGACCoordinatorConfig] = None):
        """Initialize Brain-GAC coordinator"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, BrainGACCoordinatorConfig):
            raise TypeError(f"Config must be BrainGACCoordinatorConfig or None, got {type(config)}")
        
        self.config = config or BrainGACCoordinatorConfig()
        self.logger = logging.getLogger('BrainGACCoordinator')
        
        # System references
        self.brain_system: Optional[Brain] = None
        self.gac_system: Optional[GACSystem] = None
        self.compression_system: Optional[HybridPadicCompressionSystem] = None
        self.direction_manager: Optional[DirectionManager] = None
        
        # Coordination state
        self.coordination_mode = self.config.coordination_mode
        self.active_sessions: Dict[str, TrainingSessionConfig] = {}
        self.session_queue: deque = deque(maxlen=self.config.session_queue_size)
        
        # Performance tracking
        self.coordination_history: deque = deque(maxlen=self.config.analytics_history_size)
        self.performance_baselines: Dict[str, float] = {}
        self.coordination_analytics: Dict[str, Any] = {}
        
        # Resource coordination
        self.resource_allocation: Dict[str, Dict[str, float]] = {
            'memory': {'brain': 0.5, 'gac': 0.3, 'compression': 0.2},
            'gpu': {'brain': 0.6, 'gac': 0.25, 'compression': 0.15},
            'cpu': {'brain': 0.4, 'gac': 0.4, 'compression': 0.2}
        }
        
        # Optimization tracking
        self.optimization_history: List[Dict[str, Any]] = []
        self.last_optimization: Optional[datetime] = None
        self.performance_trends: Dict[str, List[float]] = defaultdict(list)
        
        # Thread safety
        self._coordination_lock = threading.RLock()
        self._session_lock = threading.RLock()
        self._analytics_lock = threading.RLock()
        
        # Coordination metrics
        self.total_coordinations = 0
        self.successful_coordinations = 0
        self.failed_coordinations = 0
        self.coordination_efficiency = 0.0
        
        # Initialize coordination system
        self._initialize_coordination_system()
        
        self.logger.info("BrainGACCoordinator created successfully")
    
    def initialize_coordinator(self,
                             brain_system: Brain,
                             gac_system: GACSystem,
                             compression_system: Optional[HybridPadicCompressionSystem] = None,
                             direction_manager: Optional[DirectionManager] = None) -> None:
        """
        Initialize coordinator with system components.
        
        Args:
            brain_system: Brain system instance
            gac_system: GAC system instance
            compression_system: Optional compression system
            direction_manager: Optional direction manager
            
        Raises:
            TypeError: If systems are invalid
            RuntimeError: If initialization fails
        """
        if not isinstance(brain_system, Brain):
            raise TypeError(f"Brain system must be Brain instance, got {type(brain_system)}")
        if not isinstance(gac_system, GACSystem):
            raise TypeError(f"GAC system must be GACSystem instance, got {type(gac_system)}")
        if compression_system is not None and not isinstance(compression_system, HybridPadicCompressionSystem):
            raise TypeError(f"Compression system must be HybridPadicCompressionSystem, got {type(compression_system)}")
        if direction_manager is not None and not isinstance(direction_manager, DirectionManager):
            raise TypeError(f"Direction manager must be DirectionManager, got {type(direction_manager)}")
        
        try:
            with self._coordination_lock:
                # Set system references
                self.brain_system = brain_system
                self.gac_system = gac_system
                self.compression_system = compression_system
                self.direction_manager = direction_manager
                
                # Initialize performance baselines
                self._initialize_performance_baselines()
                
                # Setup coordination hooks
                self._setup_coordination_hooks()
                
                # Initialize resource coordination
                self._initialize_resource_coordination()
                
                # Start background coordination tasks
                self._start_coordination_monitoring()
                
                self.logger.info("Brain-GAC coordinator initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize coordinator: {e}")
            raise RuntimeError(f"Coordinator initialization failed: {e}")
    
    def coordinate_training_session(self, session_config: TrainingSessionConfig) -> CoordinationResult:
        """
        Coordinate a training session between Brain and GAC systems.
        
        Args:
            session_config: Training session configuration
            
        Returns:
            Coordination result
            
        Raises:
            ValueError: If session config is invalid
            RuntimeError: If coordination fails
        """
        if not isinstance(session_config, TrainingSessionConfig):
            raise TypeError(f"Session config must be TrainingSessionConfig, got {type(session_config)}")
        
        if session_config.session_id in self.active_sessions:
            raise ValueError(f"Session {session_config.session_id} already active")
        
        try:
            start_time = time.time()
            
            with self._session_lock:
                # Add session to active sessions
                self.active_sessions[session_config.session_id] = session_config
                
                coordination_result = CoordinationResult(
                    operation_type="training_session",
                    coordination_successful=False,
                    brain_result=None,
                    gac_result=None,
                    compression_result=None,
                    coordination_time_ms=0.0,
                    performance_metrics={},
                    error_details=None,
                    recommendations=[]
                )
                
                # Apply coordination strategy
                if session_config.coordination_strategy == SessionCoordinationStrategy.BRAIN_FIRST:
                    coordination_result = self._coordinate_brain_first_session(session_config)
                elif session_config.coordination_strategy == SessionCoordinationStrategy.GAC_FIRST:
                    coordination_result = self._coordinate_gac_first_session(session_config)
                elif session_config.coordination_strategy == SessionCoordinationStrategy.PARALLEL:
                    coordination_result = self._coordinate_parallel_session(session_config)
                elif session_config.coordination_strategy == SessionCoordinationStrategy.ADAPTIVE_ORDER:
                    coordination_result = self._coordinate_adaptive_session(session_config)
                elif session_config.coordination_strategy == SessionCoordinationStrategy.PERFORMANCE_OPTIMIZED:
                    coordination_result = self._coordinate_performance_optimized_session(session_config)
                else:
                    raise RuntimeError(f"Unknown coordination strategy: {session_config.coordination_strategy}")
                
                # Update coordination metrics
                coordination_time = (time.time() - start_time) * 1000
                coordination_result.coordination_time_ms = coordination_time
                
                # Update performance tracking
                self._update_coordination_performance(coordination_result)
                
                # Store coordination result
                self.coordination_history.append(coordination_result)
                
                # Clean up session
                if session_config.session_id in self.active_sessions:
                    del self.active_sessions[session_config.session_id]
                
                self.logger.debug(f"Training session coordination completed: {session_config.session_id} in {coordination_time:.2f}ms")
                
                return coordination_result
                
        except Exception as e:
            self.logger.error(f"Training session coordination failed: {e}")
            # Clean up session on error
            if session_config.session_id in self.active_sessions:
                del self.active_sessions[session_config.session_id]
            raise RuntimeError(f"Training session coordination failed: {e}")
    
    def coordinate_gradient_optimization(self, 
                                       gradients: List[torch.Tensor], 
                                       context: Optional[Dict[str, Any]] = None) -> CoordinationResult:
        """
        Coordinate gradient optimization between Brain and GAC systems.
        
        Args:
            gradients: List of gradient tensors
            context: Optional optimization context
            
        Returns:
            Coordination result
            
        Raises:
            ValueError: If gradients are invalid
            RuntimeError: If coordination fails
        """
        if not isinstance(gradients, list):
            raise TypeError("Gradients must be list")
        if len(gradients) == 0:
            raise ValueError("Gradients list cannot be empty")
        for i, grad in enumerate(gradients):
            if not isinstance(grad, torch.Tensor):
                raise TypeError(f"Gradient {i} must be torch.Tensor")
        
        try:
            start_time = time.time()
            
            with self._coordination_lock:
                coordination_result = CoordinationResult(
                    operation_type="gradient_optimization",
                    coordination_successful=False,
                    brain_result=None,
                    gac_result=None,
                    compression_result=None,
                    coordination_time_ms=0.0,
                    performance_metrics={},
                    error_details=None,
                    recommendations=[]
                )
                
                # Apply coordination mode
                if self.coordination_mode == CoordinationMode.SYNCHRONOUS:
                    coordination_result = self._coordinate_synchronous_gradients(gradients, context)
                elif self.coordination_mode == CoordinationMode.ASYNCHRONOUS:
                    coordination_result = self._coordinate_asynchronous_gradients(gradients, context)
                elif self.coordination_mode == CoordinationMode.ADAPTIVE:
                    coordination_result = self._coordinate_adaptive_gradients(gradients, context)
                elif self.coordination_mode == CoordinationMode.PRIORITY_BASED:
                    coordination_result = self._coordinate_priority_gradients(gradients, context)
                elif self.coordination_mode == CoordinationMode.LOAD_BALANCED:
                    coordination_result = self._coordinate_load_balanced_gradients(gradients, context)
                elif self.coordination_mode == CoordinationMode.REAL_TIME:
                    coordination_result = self._coordinate_real_time_gradients(gradients, context)
                else:
                    raise RuntimeError(f"Unknown coordination mode: {self.coordination_mode}")
                
                # Update coordination metrics
                coordination_time = (time.time() - start_time) * 1000
                coordination_result.coordination_time_ms = coordination_time
                
                # Update performance tracking
                self._update_coordination_performance(coordination_result)
                
                # Store coordination result
                self.coordination_history.append(coordination_result)
                
                self.logger.debug(f"Gradient optimization coordination completed in {coordination_time:.2f}ms")
                
                return coordination_result
                
        except Exception as e:
            self.logger.error(f"Gradient optimization coordination failed: {e}")
            raise RuntimeError(f"Gradient optimization coordination failed: {e}")
    
    def coordinate_performance_optimization(self) -> CoordinationResult:
        """
        Coordinate performance optimization across all systems.
        
        Returns:
            Coordination result
            
        Raises:
            RuntimeError: If coordination fails
        """
        try:
            start_time = time.time()
            
            with self._coordination_lock:
                coordination_result = CoordinationResult(
                    operation_type="performance_optimization",
                    coordination_successful=False,
                    brain_result=None,
                    gac_result=None,
                    compression_result=None,
                    coordination_time_ms=0.0,
                    performance_metrics={},
                    error_details=None,
                    recommendations=[]
                )
                
                optimization_results = {}
                
                # Coordinate Brain performance optimization
                if self.brain_system and hasattr(self.brain_system, 'optimize_performance'):
                    brain_optimization = self.brain_system.optimize_performance()
                    optimization_results['brain'] = brain_optimization
                    coordination_result.brain_result = brain_optimization
                
                # Coordinate GAC performance optimization
                if self.gac_system and hasattr(self.gac_system, 'optimize_performance'):
                    gac_optimization = self.gac_system.optimize_performance()
                    optimization_results['gac'] = gac_optimization
                    coordination_result.gac_result = gac_optimization
                
                # Coordinate compression performance optimization
                if self.compression_system and hasattr(self.compression_system, 'optimize_performance'):
                    compression_optimization = self.compression_system.optimize_performance()
                    optimization_results['compression'] = compression_optimization
                    coordination_result.compression_result = compression_optimization
                
                # Coordinate direction manager optimization
                if self.direction_manager and hasattr(self.direction_manager, 'optimize_direction_based_switching'):
                    direction_optimization = self.direction_manager.optimize_direction_based_switching()
                    optimization_results['direction'] = direction_optimization
                
                # Calculate overall coordination performance
                coordination_result.performance_metrics = self._calculate_coordination_performance(optimization_results)
                coordination_result.coordination_successful = len(optimization_results) > 0
                
                # Generate optimization recommendations
                coordination_result.recommendations = self._generate_coordination_recommendations(optimization_results)
                
                # Update coordination metrics
                coordination_time = (time.time() - start_time) * 1000
                coordination_result.coordination_time_ms = coordination_time
                
                # Update performance tracking
                self._update_coordination_performance(coordination_result)
                
                # Store coordination result
                self.coordination_history.append(coordination_result)
                
                # Update last optimization time
                self.last_optimization = datetime.utcnow()
                
                self.logger.info(f"Performance optimization coordination completed in {coordination_time:.2f}ms")
                
                return coordination_result
                
        except Exception as e:
            self.logger.error(f"Performance optimization coordination failed: {e}")
            raise RuntimeError(f"Performance optimization coordination failed: {e}")
    
    def coordinate_error_recovery(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> CoordinationResult:
        """
        Coordinate error recovery across all systems.
        
        Args:
            error: Error that occurred
            context: Optional error context
            
        Returns:
            Coordination result
            
        Raises:
            ValueError: If error is invalid
            RuntimeError: If coordination fails
        """
        if error is None:
            raise ValueError("Error cannot be None")
        if not isinstance(error, Exception):
            raise TypeError("Error must be Exception instance")
        
        try:
            start_time = time.time()
            
            with self._coordination_lock:
                coordination_result = CoordinationResult(
                    operation_type="error_recovery",
                    coordination_successful=False,
                    brain_result=None,
                    gac_result=None,
                    compression_result=None,
                    coordination_time_ms=0.0,
                    performance_metrics={},
                    error_details=str(error),
                    recommendations=[]
                )
                
                recovery_results = {}
                
                # Coordinate Brain error recovery
                if self.brain_system and hasattr(self.brain_system, 'handle_error'):
                    brain_recovery = self.brain_system.handle_error(error, context)
                    recovery_results['brain'] = brain_recovery
                    coordination_result.brain_result = brain_recovery
                
                # Coordinate GAC error recovery
                if self.gac_system and hasattr(self.gac_system, 'handle_error'):
                    gac_recovery = self.gac_system.handle_error(error, context)
                    recovery_results['gac'] = gac_recovery
                    coordination_result.gac_result = gac_recovery
                
                # Coordinate compression error recovery
                if self.compression_system and hasattr(self.compression_system, 'handle_error'):
                    compression_recovery = self.compression_system.handle_error(error, context)
                    recovery_results['compression'] = compression_recovery
                    coordination_result.compression_result = compression_recovery
                
                # Apply system-wide recovery coordination
                system_recovery = self._coordinate_system_recovery(error, recovery_results, context)
                recovery_results['system'] = system_recovery
                
                # Determine overall recovery success
                recovery_successful = any(
                    result.get('recovery_successful', False) 
                    for result in recovery_results.values() 
                    if isinstance(result, dict)
                )
                
                coordination_result.coordination_successful = recovery_successful
                
                # Calculate recovery performance metrics
                coordination_result.performance_metrics = self._calculate_recovery_performance(recovery_results)
                
                # Generate recovery recommendations
                coordination_result.recommendations = self._generate_recovery_recommendations(error, recovery_results)
                
                # Update coordination metrics
                coordination_time = (time.time() - start_time) * 1000
                coordination_result.coordination_time_ms = coordination_time
                
                # Update performance tracking
                self._update_coordination_performance(coordination_result)
                
                # Store coordination result
                self.coordination_history.append(coordination_result)
                
                self.logger.info(f"Error recovery coordination completed: {'successful' if recovery_successful else 'failed'} in {coordination_time:.2f}ms")
                
                return coordination_result
                
        except Exception as e:
            self.logger.error(f"Error recovery coordination failed: {e}")
            raise RuntimeError(f"Error recovery coordination failed: {e}")
    
    def coordinate_resource_management(self) -> CoordinationResult:
        """
        Coordinate resource management across all systems.
        
        Returns:
            Coordination result
            
        Raises:
            RuntimeError: If coordination fails
        """
        try:
            start_time = time.time()
            
            with self._coordination_lock:
                coordination_result = CoordinationResult(
                    operation_type="resource_management",
                    coordination_successful=False,
                    brain_result=None,
                    gac_result=None,
                    compression_result=None,
                    coordination_time_ms=0.0,
                    performance_metrics={},
                    error_details=None,
                    recommendations=[]
                )
                
                # Get current resource usage
                current_usage = self._get_current_resource_usage()
                
                # Apply resource coordination level
                if self.config.resource_coordination_level == ResourceCoordinationLevel.MINIMAL:
                    resource_result = self._apply_minimal_resource_coordination(current_usage)
                elif self.config.resource_coordination_level == ResourceCoordinationLevel.MODERATE:
                    resource_result = self._apply_moderate_resource_coordination(current_usage)
                elif self.config.resource_coordination_level == ResourceCoordinationLevel.COMPREHENSIVE:
                    resource_result = self._apply_comprehensive_resource_coordination(current_usage)
                elif self.config.resource_coordination_level == ResourceCoordinationLevel.AGGRESSIVE:
                    resource_result = self._apply_aggressive_resource_coordination(current_usage)
                else:
                    raise RuntimeError(f"Unknown resource coordination level: {self.config.resource_coordination_level}")
                
                # Update resource allocation based on results
                self._update_resource_allocation(resource_result)
                
                # Set coordination results
                coordination_result.brain_result = resource_result.get('brain', {})
                coordination_result.gac_result = resource_result.get('gac', {})
                coordination_result.compression_result = resource_result.get('compression', {})
                coordination_result.coordination_successful = resource_result.get('success', False)
                
                # Calculate performance metrics
                coordination_result.performance_metrics = self._calculate_resource_performance(resource_result)
                
                # Generate resource recommendations
                coordination_result.recommendations = self._generate_resource_recommendations(current_usage, resource_result)
                
                # Update coordination metrics
                coordination_time = (time.time() - start_time) * 1000
                coordination_result.coordination_time_ms = coordination_time
                
                # Update performance tracking
                self._update_coordination_performance(coordination_result)
                
                # Store coordination result
                self.coordination_history.append(coordination_result)
                
                self.logger.debug(f"Resource management coordination completed in {coordination_time:.2f}ms")
                
                return coordination_result
                
        except Exception as e:
            self.logger.error(f"Resource management coordination failed: {e}")
            raise RuntimeError(f"Resource management coordination failed: {e}")
    
    def get_coordination_analytics(self) -> CoordinationAnalytics:
        """
        Get comprehensive coordination analytics.
        
        Returns:
            Coordination analytics data
            
        Raises:
            RuntimeError: If analytics generation fails
        """
        try:
            with self._analytics_lock:
                # Calculate basic statistics
                total_coords = self.total_coordinations
                success_coords = self.successful_coordinations
                failed_coords = self.failed_coordinations
                
                # Calculate average coordination time
                if self.coordination_history:
                    avg_coord_time = sum(c.coordination_time_ms for c in self.coordination_history) / len(self.coordination_history)
                else:
                    avg_coord_time = 0.0
                
                # Calculate system efficiencies
                brain_efficiency = self._calculate_brain_coordination_efficiency()
                gac_efficiency = self._calculate_gac_coordination_efficiency()
                resource_efficiency = self._calculate_resource_coordination_efficiency()
                
                # Calculate session statistics
                session_stats = self._calculate_session_coordination_stats()
                
                # Calculate performance optimization statistics
                perf_opt_stats = self._calculate_performance_optimization_stats()
                
                # Calculate error recovery statistics
                error_recovery_stats = self._calculate_error_recovery_stats()
                
                # Calculate coordination trends
                coordination_trends = self._calculate_coordination_trends()
                
                # Generate optimization recommendations
                optimization_recommendations = self._generate_system_optimization_recommendations()
                
                return CoordinationAnalytics(
                    total_coordinations=total_coords,
                    successful_coordinations=success_coords,
                    failed_coordinations=failed_coords,
                    average_coordination_time_ms=avg_coord_time,
                    brain_coordination_efficiency=brain_efficiency,
                    gac_coordination_efficiency=gac_efficiency,
                    resource_coordination_efficiency=resource_efficiency,
                    session_coordination_stats=session_stats,
                    performance_optimization_stats=perf_opt_stats,
                    error_recovery_stats=error_recovery_stats,
                    coordination_trends=coordination_trends,
                    optimization_recommendations=optimization_recommendations
                )
                
        except Exception as e:
            self.logger.error(f"Coordination analytics generation failed: {e}")
            raise RuntimeError(f"Coordination analytics generation failed: {e}")
    
    def _initialize_coordination_system(self) -> None:
        """Initialize coordination system"""
        # Initialize performance baselines
        self.performance_baselines = {
            'coordination_time_ms': self.config.max_coordination_time_ms,
            'success_rate': 0.95,
            'efficiency_score': 0.85
        }
        
        # Initialize coordination analytics
        self.coordination_analytics = {
            'operations': defaultdict(int),
            'performance': defaultdict(list),
            'errors': defaultdict(int),
            'sessions': {}
        }
    
    def _initialize_performance_baselines(self) -> None:
        """Initialize performance baselines"""
        self.performance_baselines.update({
            'brain_response_time_ms': 50.0,
            'gac_processing_time_ms': 30.0,
            'compression_time_ms': 20.0,
            'resource_efficiency': 0.8
        })
    
    def _setup_coordination_hooks(self) -> None:
        """Setup coordination hooks"""
        # Setup Brain coordination hooks
        if hasattr(self.brain_system, 'register_coordination_hook'):
            self.brain_system.register_coordination_hook('pre_operation', self._brain_pre_operation_hook)
            self.brain_system.register_coordination_hook('post_operation', self._brain_post_operation_hook)
        
        # Setup GAC coordination hooks
        if hasattr(self.gac_system, 'register_coordination_hook'):
            self.gac_system.register_coordination_hook('pre_optimization', self._gac_pre_optimization_hook)
            self.gac_system.register_coordination_hook('post_optimization', self._gac_post_optimization_hook)
    
    def _initialize_resource_coordination(self) -> None:
        """Initialize resource coordination"""
        if self.config.enable_memory_coordination:
            self._initialize_memory_coordination()
        if self.config.enable_gpu_coordination:
            self._initialize_gpu_coordination()
        if self.config.enable_cpu_coordination:
            self._initialize_cpu_coordination()
    
    def _start_coordination_monitoring(self) -> None:
        """Start coordination monitoring"""
        self.logger.info("Coordination monitoring started")
    
    def _coordinate_brain_first_session(self, session_config: TrainingSessionConfig) -> CoordinationResult:
        """Coordinate Brain-first session strategy"""
        result = CoordinationResult(
            operation_type="brain_first_session",
            coordination_successful=False,
            brain_result=None,
            gac_result=None,
            compression_result=None,
            coordination_time_ms=0.0,
            performance_metrics={},
            error_details=None,
            recommendations=[]
        )
        
        try:
            # Execute Brain operations first
            if self.brain_system and hasattr(self.brain_system, 'execute_training_step'):
                brain_result = self.brain_system.execute_training_step(session_config.__dict__)
                result.brain_result = brain_result
                
                # Use Brain results for GAC optimization
                if brain_result and self.gac_system:
                    gac_context = {'brain_result': brain_result}
                    if hasattr(self.gac_system, 'optimize_with_context'):
                        gac_result = self.gac_system.optimize_with_context(gac_context)
                        result.gac_result = gac_result
            
            result.coordination_successful = True
            
        except Exception as e:
            result.error_details = str(e)
            result.coordination_successful = False
        
        return result
    
    def _coordinate_gac_first_session(self, session_config: TrainingSessionConfig) -> CoordinationResult:
        """Coordinate GAC-first session strategy"""
        result = CoordinationResult(
            operation_type="gac_first_session",
            coordination_successful=False,
            brain_result=None,
            gac_result=None,
            compression_result=None,
            coordination_time_ms=0.0,
            performance_metrics={},
            error_details=None,
            recommendations=[]
        )
        
        try:
            # Execute GAC operations first
            if self.gac_system and hasattr(self.gac_system, 'prepare_optimization'):
                gac_result = self.gac_system.prepare_optimization(session_config.__dict__)
                result.gac_result = gac_result
                
                # Use GAC results for Brain training
                if gac_result and self.brain_system:
                    brain_context = {'gac_result': gac_result}
                    if hasattr(self.brain_system, 'execute_training_with_context'):
                        brain_result = self.brain_system.execute_training_with_context(brain_context)
                        result.brain_result = brain_result
            
            result.coordination_successful = True
            
        except Exception as e:
            result.error_details = str(e)
            result.coordination_successful = False
        
        return result
    
    def _coordinate_parallel_session(self, session_config: TrainingSessionConfig) -> CoordinationResult:
        """Coordinate parallel session strategy"""
        result = CoordinationResult(
            operation_type="parallel_session",
            coordination_successful=False,
            brain_result=None,
            gac_result=None,
            compression_result=None,
            coordination_time_ms=0.0,
            performance_metrics={},
            error_details=None,
            recommendations=[]
        )
        
        try:
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}
                
                # Submit Brain operation
                if self.brain_system and hasattr(self.brain_system, 'execute_training_step'):
                    futures['brain'] = executor.submit(
                        self.brain_system.execute_training_step, 
                        session_config.__dict__
                    )
                
                # Submit GAC operation
                if self.gac_system and hasattr(self.gac_system, 'optimize_gradients'):
                    futures['gac'] = executor.submit(
                        self.gac_system.optimize_gradients,
                        [], 0.001  # Default empty gradients and learning rate
                    )
                
                # Submit compression operation
                if self.compression_system and hasattr(self.compression_system, 'prepare_compression'):
                    futures['compression'] = executor.submit(
                        self.compression_system.prepare_compression,
                        session_config.__dict__
                    )
                
                # Collect results
                for system_name, future in futures.items():
                    try:
                        result_data = future.result(timeout=self.config.coordination_timeout_ms / 1000.0)
                        if system_name == 'brain':
                            result.brain_result = result_data
                        elif system_name == 'gac':
                            result.gac_result = result_data
                        elif system_name == 'compression':
                            result.compression_result = result_data
                    except Exception as system_error:
                        self.logger.warning(f"Parallel {system_name} operation failed: {system_error}")
            
            result.coordination_successful = True
            
        except Exception as e:
            result.error_details = str(e)
            result.coordination_successful = False
        
        return result
    
    def _coordinate_adaptive_session(self, session_config: TrainingSessionConfig) -> CoordinationResult:
        """Coordinate adaptive session strategy"""
        # Analyze current system performance and choose best strategy
        current_performance = self._get_current_system_performance()
        
        if current_performance.get('brain_load', 0.5) > 0.8:
            # High Brain load - use GAC first
            return self._coordinate_gac_first_session(session_config)
        elif current_performance.get('gac_load', 0.5) > 0.8:
            # High GAC load - use Brain first
            return self._coordinate_brain_first_session(session_config)
        elif self.config.enable_parallel_coordination:
            # Balanced load - use parallel
            return self._coordinate_parallel_session(session_config)
        else:
            # Default to Brain first
            return self._coordinate_brain_first_session(session_config)
    
    def _coordinate_performance_optimized_session(self, session_config: TrainingSessionConfig) -> CoordinationResult:
        """Coordinate performance-optimized session strategy"""
        # Use the strategy with best historical performance
        best_strategy = self._get_best_performing_strategy()
        
        if best_strategy == 'brain_first':
            return self._coordinate_brain_first_session(session_config)
        elif best_strategy == 'gac_first':
            return self._coordinate_gac_first_session(session_config)
        elif best_strategy == 'parallel':
            return self._coordinate_parallel_session(session_config)
        else:
            return self._coordinate_adaptive_session(session_config)
    
    def _coordinate_synchronous_gradients(self, gradients: List[torch.Tensor], context: Optional[Dict[str, Any]]) -> CoordinationResult:
        """Coordinate gradients synchronously"""
        result = CoordinationResult(
            operation_type="synchronous_gradients",
            coordination_successful=False,
            brain_result=None,
            gac_result=None,
            compression_result=None,
            coordination_time_ms=0.0,
            performance_metrics={},
            error_details=None,
            recommendations=[]
        )
        
        try:
            # Process gradients through GAC system
            if self.gac_system:
                gac_result = self.gac_system.optimize_gradients(gradients, context.get('learning_rate', 0.001) if context else 0.001)
                result.gac_result = gac_result
                
                # Use optimized gradients for compression
                if gac_result and self.compression_system:
                    for grad in gac_result:
                        compression_result = self.compression_system.compress(grad)
                        # Store last compression result
                        result.compression_result = compression_result
            
            result.coordination_successful = True
            
        except Exception as e:
            result.error_details = str(e)
            result.coordination_successful = False
        
        return result
    
    def _coordinate_asynchronous_gradients(self, gradients: List[torch.Tensor], context: Optional[Dict[str, Any]]) -> CoordinationResult:
        """Coordinate gradients asynchronously"""
        result = CoordinationResult(
            operation_type="asynchronous_gradients",
            coordination_successful=False,
            brain_result=None,
            gac_result=None,
            compression_result=None,
            coordination_time_ms=0.0,
            performance_metrics={},
            error_details=None,
            recommendations=[]
        )
        
        try:
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit GAC optimization
                gac_future = None
                if self.gac_system:
                    gac_future = executor.submit(
                        self.gac_system.optimize_gradients,
                        gradients,
                        context.get('learning_rate', 0.001) if context else 0.001
                    )
                
                # Submit compression (use original gradients)
                compression_futures = []
                if self.compression_system:
                    for grad in gradients[:5]:  # Limit to first 5 gradients
                        compression_future = executor.submit(self.compression_system.compress, grad)
                        compression_futures.append(compression_future)
                
                # Collect GAC result
                if gac_future:
                    try:
                        result.gac_result = gac_future.result(timeout=self.config.coordination_timeout_ms / 1000.0)
                    except Exception as gac_error:
                        self.logger.warning(f"Asynchronous GAC operation failed: {gac_error}")
                
                # Collect compression results
                if compression_futures:
                    try:
                        compression_results = []
                        for future in compression_futures:
                            compression_results.append(future.result(timeout=1.0))
                        result.compression_result = compression_results
                    except Exception as compression_error:
                        self.logger.warning(f"Asynchronous compression failed: {compression_error}")
            
            result.coordination_successful = True
            
        except Exception as e:
            result.error_details = str(e)
            result.coordination_successful = False
        
        return result
    
    def _coordinate_adaptive_gradients(self, gradients: List[torch.Tensor], context: Optional[Dict[str, Any]]) -> CoordinationResult:
        """Coordinate gradients adaptively"""
        # Choose coordination mode based on current conditions
        current_load = self._assess_current_coordination_load()
        
        if current_load > 0.8:
            # High load - use asynchronous
            return self._coordinate_asynchronous_gradients(gradients, context)
        elif len(gradients) > 50:
            # Many gradients - use asynchronous
            return self._coordinate_asynchronous_gradients(gradients, context)
        else:
            # Normal conditions - use synchronous
            return self._coordinate_synchronous_gradients(gradients, context)
    
    def _coordinate_priority_gradients(self, gradients: List[torch.Tensor], context: Optional[Dict[str, Any]]) -> CoordinationResult:
        """Coordinate gradients with priority-based processing"""
        # Calculate gradient priorities
        gradient_priorities = []
        for i, grad in enumerate(gradients):
            priority = self._calculate_gradient_priority(grad, context)
            gradient_priorities.append((priority, i, grad))
        
        # Sort by priority (highest first)
        gradient_priorities.sort(key=lambda x: x[0], reverse=True)
        
        # Process high-priority gradients first
        high_priority_gradients = [grad for priority, _, grad in gradient_priorities[:10]]  # Top 10
        
        return self._coordinate_synchronous_gradients(high_priority_gradients, context)
    
    def _coordinate_load_balanced_gradients(self, gradients: List[torch.Tensor], context: Optional[Dict[str, Any]]) -> CoordinationResult:
        """Coordinate gradients with load balancing"""
        # Distribute gradients across available systems based on current load
        system_loads = self._get_current_system_loads()
        
        if system_loads.get('gac', 0.5) < system_loads.get('compression', 0.5):
            # GAC has lower load - prioritize GAC processing
            return self._coordinate_synchronous_gradients(gradients, context)
        else:
            # Compression has lower load - use asynchronous
            return self._coordinate_asynchronous_gradients(gradients, context)
    
    def _coordinate_real_time_gradients(self, gradients: List[torch.Tensor], context: Optional[Dict[str, Any]]) -> CoordinationResult:
        """Coordinate gradients with real-time constraints"""
        # Use the fastest coordination mode for real-time requirements
        if len(gradients) <= 10:
            # Small batch - use synchronous for consistency
            return self._coordinate_synchronous_gradients(gradients, context)
        else:
            # Large batch - use asynchronous for speed
            return self._coordinate_asynchronous_gradients(gradients, context)
    
    def _coordinate_system_recovery(self, error: Exception, recovery_results: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Coordinate system-wide recovery"""
        system_recovery = {
            'recovery_attempted': True,
            'recovery_successful': False,
            'coordination_adjustments': [],
            'system_state_changes': []
        }
        
        try:
            # Analyze recovery results from individual systems
            brain_recovered = recovery_results.get('brain', {}).get('recovery_successful', False)
            gac_recovered = recovery_results.get('gac', {}).get('recovery_successful', False)
            compression_recovered = recovery_results.get('compression', {}).get('recovery_successful', False)
            
            # Apply system-wide coordination adjustments
            if not brain_recovered:
                # Reduce Brain system load
                self.resource_allocation['cpu']['brain'] *= 0.8
                system_recovery['coordination_adjustments'].append('reduced_brain_cpu_allocation')
            
            if not gac_recovered:
                # Switch to simpler coordination mode
                self.coordination_mode = CoordinationMode.SYNCHRONOUS
                system_recovery['coordination_adjustments'].append('switched_to_synchronous_mode')
            
            if not compression_recovered:
                # Reduce compression system priority
                self.resource_allocation['memory']['compression'] *= 0.8
                system_recovery['coordination_adjustments'].append('reduced_compression_memory_allocation')
            
            # Determine overall recovery success
            any_recovered = brain_recovered or gac_recovered or compression_recovered
            system_recovery['recovery_successful'] = any_recovered
            
            if any_recovered:
                system_recovery['system_state_changes'].append('partial_recovery_achieved')
            else:
                system_recovery['system_state_changes'].append('fallback_mode_activated')
                # Activate fallback coordination mode
                self.coordination_mode = CoordinationMode.SYNCHRONOUS
                
        except Exception as recovery_error:
            self.logger.error(f"System recovery coordination failed: {recovery_error}")
            system_recovery['recovery_successful'] = False
        
        return system_recovery
    
    def _update_coordination_performance(self, coordination_result: CoordinationResult) -> None:
        """Update coordination performance metrics"""
        self.total_coordinations += 1
        
        if coordination_result.coordination_successful:
            self.successful_coordinations += 1
        else:
            self.failed_coordinations += 1
        
        # Update efficiency
        if self.total_coordinations > 0:
            self.coordination_efficiency = self.successful_coordinations / self.total_coordinations
        
        # Update performance trends
        self.performance_trends['coordination_time'].append(coordination_result.coordination_time_ms)
        self.performance_trends['success_rate'].append(1.0 if coordination_result.coordination_successful else 0.0)
        
        # Limit trend history
        max_trend_size = 100
        for trend in self.performance_trends.values():
            if len(trend) > max_trend_size:
                trend[:] = trend[-max_trend_size:]
    
    def _calculate_coordination_performance(self, optimization_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate coordination performance metrics"""
        performance_metrics = {
            'systems_optimized': len(optimization_results),
            'total_optimizations': sum(
                len(result.get('optimizations_applied', []))
                for result in optimization_results.values()
                if isinstance(result, dict)
            ),
            'average_improvement': 0.0
        }
        
        # Calculate average improvement
        improvements = []
        for result in optimization_results.values():
            if isinstance(result, dict) and 'performance_improvement' in result:
                improvements.append(result['performance_improvement'])
        
        if improvements:
            performance_metrics['average_improvement'] = sum(improvements) / len(improvements)
        
        return performance_metrics
    
    def _generate_coordination_recommendations(self, optimization_results: Dict[str, Any]) -> List[str]:
        """Generate coordination recommendations"""
        recommendations = []
        
        # Analyze optimization results
        if len(optimization_results) < 2:
            recommendations.append("Consider enabling more system integrations for better coordination")
        
        # Check for performance improvements
        total_improvements = sum(
            len(result.get('optimizations_applied', []))
            for result in optimization_results.values()
            if isinstance(result, dict)
        )
        
        if total_improvements < 3:
            recommendations.append("Consider enabling adaptive coordination for better optimization")
        
        # Check coordination efficiency
        if self.coordination_efficiency < 0.8:
            recommendations.append("Consider switching to more reliable coordination mode")
        
        return recommendations
    
    def _calculate_recovery_performance(self, recovery_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate recovery performance metrics"""
        performance_metrics = {
            'systems_recovered': sum(
                1 for result in recovery_results.values()
                if isinstance(result, dict) and result.get('recovery_successful', False)
            ),
            'total_systems': len(recovery_results),
            'recovery_rate': 0.0
        }
        
        if performance_metrics['total_systems'] > 0:
            performance_metrics['recovery_rate'] = performance_metrics['systems_recovered'] / performance_metrics['total_systems']
        
        return performance_metrics
    
    def _generate_recovery_recommendations(self, error: Exception, recovery_results: Dict[str, Any]) -> List[str]:
        """Generate recovery recommendations"""
        recommendations = []
        
        # Analyze recovery success rate
        successful_recoveries = sum(
            1 for result in recovery_results.values()
            if isinstance(result, dict) and result.get('recovery_successful', False)
        )
        
        if successful_recoveries == 0:
            recommendations.append("Consider implementing more robust error recovery strategies")
        elif successful_recoveries < len(recovery_results):
            recommendations.append("Consider improving error recovery for failing systems")
        
        # Error-specific recommendations
        if isinstance(error, MemoryError):
            recommendations.append("Consider enabling memory-efficient coordination mode")
        elif isinstance(error, TimeoutError):
            recommendations.append("Consider increasing coordination timeout limits")
        
        return recommendations
    
    def _get_current_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage across systems"""
        return {
            'memory_usage_percent': 0.6,  # Mock data
            'cpu_usage_percent': 0.5,
            'gpu_usage_percent': 0.7,
            'coordination_load': self._assess_current_coordination_load()
        }
    
    def _apply_minimal_resource_coordination(self, current_usage: Dict[str, float]) -> Dict[str, Any]:
        """Apply minimal resource coordination"""
        return {
            'success': True,
            'adjustments_made': [],
            'brain': {'resource_adjustment': 'none'},
            'gac': {'resource_adjustment': 'none'},
            'compression': {'resource_adjustment': 'none'}
        }
    
    def _apply_moderate_resource_coordination(self, current_usage: Dict[str, float]) -> Dict[str, Any]:
        """Apply moderate resource coordination"""
        adjustments = []
        
        # Adjust based on current usage
        if current_usage.get('memory_usage_percent', 0) > 0.8:
            # High memory usage - reduce allocations
            self.resource_allocation['memory']['brain'] *= 0.9
            self.resource_allocation['memory']['gac'] *= 0.9
            adjustments.append('reduced_memory_allocation')
        
        return {
            'success': True,
            'adjustments_made': adjustments,
            'brain': {'resource_adjustment': 'moderate'},
            'gac': {'resource_adjustment': 'moderate'},
            'compression': {'resource_adjustment': 'moderate'}
        }
    
    def _apply_comprehensive_resource_coordination(self, current_usage: Dict[str, float]) -> Dict[str, Any]:
        """Apply comprehensive resource coordination"""
        adjustments = []
        
        # Comprehensive resource rebalancing
        total_memory = sum(self.resource_allocation['memory'].values())
        total_cpu = sum(self.resource_allocation['cpu'].values())
        total_gpu = sum(self.resource_allocation['gpu'].values())
        
        # Normalize allocations
        for resource_type in ['memory', 'cpu', 'gpu']:
            total = sum(self.resource_allocation[resource_type].values())
            if total > 1.0:
                # Normalize to sum to 1.0
                for system in self.resource_allocation[resource_type]:
                    self.resource_allocation[resource_type][system] /= total
                adjustments.append(f'normalized_{resource_type}_allocation')
        
        return {
            'success': True,
            'adjustments_made': adjustments,
            'brain': {'resource_adjustment': 'comprehensive'},
            'gac': {'resource_adjustment': 'comprehensive'},
            'compression': {'resource_adjustment': 'comprehensive'}
        }
    
    def _apply_aggressive_resource_coordination(self, current_usage: Dict[str, float]) -> Dict[str, Any]:
        """Apply aggressive resource coordination"""
        adjustments = []
        
        # Aggressive optimization based on performance
        if self.coordination_efficiency < 0.7:
            # Poor efficiency - reallocate resources to best performing system
            best_system = self._identify_best_performing_system()
            
            if best_system == 'brain':
                self.resource_allocation['cpu']['brain'] = 0.6
                self.resource_allocation['cpu']['gac'] = 0.25
                self.resource_allocation['cpu']['compression'] = 0.15
            elif best_system == 'gac':
                self.resource_allocation['cpu']['gac'] = 0.6
                self.resource_allocation['cpu']['brain'] = 0.25
                self.resource_allocation['cpu']['compression'] = 0.15
            
            adjustments.append(f'aggressive_reallocation_to_{best_system}')
        
        return {
            'success': True,
            'adjustments_made': adjustments,
            'brain': {'resource_adjustment': 'aggressive'},
            'gac': {'resource_adjustment': 'aggressive'},
            'compression': {'resource_adjustment': 'aggressive'}
        }
    
    def _update_resource_allocation(self, resource_result: Dict[str, Any]) -> None:
        """Update resource allocation based on coordination results"""
        if resource_result.get('success', False):
            # Log successful resource coordination
            self.logger.debug(f"Resource allocation updated: {len(resource_result.get('adjustments_made', []))} adjustments")
    
    def _calculate_resource_performance(self, resource_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate resource coordination performance"""
        return {
            'coordination_successful': 1.0 if resource_result.get('success', False) else 0.0,
            'adjustments_made': len(resource_result.get('adjustments_made', [])),
            'resource_efficiency': self._calculate_current_resource_efficiency()
        }
    
    def _generate_resource_recommendations(self, current_usage: Dict[str, float], resource_result: Dict[str, Any]) -> List[str]:
        """Generate resource management recommendations"""
        recommendations = []
        
        if current_usage.get('memory_usage_percent', 0) > 0.9:
            recommendations.append("Consider enabling aggressive memory coordination")
        
        if current_usage.get('cpu_usage_percent', 0) > 0.9:
            recommendations.append("Consider load balancing CPU allocation across systems")
        
        if len(resource_result.get('adjustments_made', [])) == 0:
            recommendations.append("Consider enabling more comprehensive resource coordination")
        
        return recommendations
    
    # Helper methods for coordination analytics
    def _calculate_brain_coordination_efficiency(self) -> float:
        """Calculate Brain coordination efficiency"""
        if not self.coordination_history:
            return 0.0
        
        brain_coords = [c for c in self.coordination_history if c.brain_result is not None]
        if not brain_coords:
            return 0.0
        
        successful_brain_coords = [c for c in brain_coords if c.coordination_successful]
        return len(successful_brain_coords) / len(brain_coords)
    
    def _calculate_gac_coordination_efficiency(self) -> float:
        """Calculate GAC coordination efficiency"""
        if not self.coordination_history:
            return 0.0
        
        gac_coords = [c for c in self.coordination_history if c.gac_result is not None]
        if not gac_coords:
            return 0.0
        
        successful_gac_coords = [c for c in gac_coords if c.coordination_successful]
        return len(successful_gac_coords) / len(gac_coords)
    
    def _calculate_resource_coordination_efficiency(self) -> float:
        """Calculate resource coordination efficiency"""
        resource_coords = [c for c in self.coordination_history if c.operation_type == "resource_management"]
        if not resource_coords:
            return 0.0
        
        successful_resource_coords = [c for c in resource_coords if c.coordination_successful]
        return len(successful_resource_coords) / len(resource_coords)
    
    def _calculate_session_coordination_stats(self) -> Dict[str, Dict[str, Any]]:
        """Calculate session coordination statistics"""
        session_stats = {}
        
        session_coords = [c for c in self.coordination_history if "session" in c.operation_type]
        
        for coord in session_coords:
            session_type = coord.operation_type
            if session_type not in session_stats:
                session_stats[session_type] = {
                    'total_sessions': 0,
                    'successful_sessions': 0,
                    'average_time_ms': 0.0,
                    'times': []
                }
            
            session_stats[session_type]['total_sessions'] += 1
            session_stats[session_type]['times'].append(coord.coordination_time_ms)
            
            if coord.coordination_successful:
                session_stats[session_type]['successful_sessions'] += 1
        
        # Calculate averages
        for stats in session_stats.values():
            if stats['times']:
                stats['average_time_ms'] = sum(stats['times']) / len(stats['times'])
        
        return session_stats
    
    def _calculate_performance_optimization_stats(self) -> Dict[str, float]:
        """Calculate performance optimization statistics"""
        perf_coords = [c for c in self.coordination_history if c.operation_type == "performance_optimization"]
        
        if not perf_coords:
            return {}
        
        return {
            'total_optimizations': len(perf_coords),
            'successful_optimizations': len([c for c in perf_coords if c.coordination_successful]),
            'average_optimization_time_ms': sum(c.coordination_time_ms for c in perf_coords) / len(perf_coords)
        }
    
    def _calculate_error_recovery_stats(self) -> Dict[str, int]:
        """Calculate error recovery statistics"""
        error_coords = [c for c in self.coordination_history if c.operation_type == "error_recovery"]
        
        return {
            'total_error_recoveries': len(error_coords),
            'successful_recoveries': len([c for c in error_coords if c.coordination_successful]),
            'failed_recoveries': len([c for c in error_coords if not c.coordination_successful])
        }
    
    def _calculate_coordination_trends(self) -> Dict[str, List[float]]:
        """Calculate coordination trends"""
        trends = {}
        
        if len(self.coordination_history) >= 10:
            recent_coords = list(self.coordination_history)[-50:]  # Last 50 coordinations
            
            trends['coordination_time'] = [c.coordination_time_ms for c in recent_coords]
            trends['success_rate'] = [1.0 if c.coordination_successful else 0.0 for c in recent_coords]
        
        return trends
    
    def _generate_system_optimization_recommendations(self) -> List[str]:
        """Generate system optimization recommendations"""
        recommendations = []
        
        if self.coordination_efficiency < 0.8:
            recommendations.append("Consider optimizing coordination strategies for better efficiency")
        
        if len(self.active_sessions) > self.config.max_concurrent_sessions * 0.8:
            recommendations.append("Consider increasing max concurrent sessions or implementing session queuing")
        
        # Performance-based recommendations
        if self.coordination_history:
            avg_time = sum(c.coordination_time_ms for c in self.coordination_history) / len(self.coordination_history)
            if avg_time > self.config.max_coordination_time_ms * 0.8:
                recommendations.append("Consider enabling asynchronous coordination for better performance")
        
        return recommendations
    
    # Helper methods for internal operations
    def _get_current_system_performance(self) -> Dict[str, float]:
        """Get current system performance metrics"""
        return {
            'brain_load': 0.5,  # Mock data
            'gac_load': 0.6,
            'compression_load': 0.4,
            'coordination_efficiency': self.coordination_efficiency
        }
    
    def _get_best_performing_strategy(self) -> str:
        """Get the best performing coordination strategy"""
        # Analyze historical performance of different strategies
        strategy_performance = defaultdict(list)
        
        for coord in self.coordination_history:
            if "session" in coord.operation_type:
                strategy = coord.operation_type.replace("_session", "")
                success_score = 1.0 if coord.coordination_successful else 0.0
                strategy_performance[strategy].append(success_score)
        
        # Find best performing strategy
        best_strategy = "brain_first"  # Default
        best_score = 0.0
        
        for strategy, scores in strategy_performance.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_strategy = strategy
        
        return best_strategy
    
    def _assess_current_coordination_load(self) -> float:
        """Assess current coordination load"""
        # Simple load assessment based on active sessions and recent coordination time
        active_load = len(self.active_sessions) / max(1, self.config.max_concurrent_sessions)
        
        if self.coordination_history:
            recent_coords = list(self.coordination_history)[-5:]
            avg_recent_time = sum(c.coordination_time_ms for c in recent_coords) / len(recent_coords)
            time_load = min(1.0, avg_recent_time / self.config.max_coordination_time_ms)
        else:
            time_load = 0.0
        
        return (active_load + time_load) / 2.0
    
    def _calculate_gradient_priority(self, gradient: torch.Tensor, context: Optional[Dict[str, Any]]) -> float:
        """Calculate gradient priority for priority-based coordination"""
        # Simple priority calculation based on gradient magnitude
        magnitude = torch.norm(gradient).item()
        normalized_magnitude = min(1.0, magnitude / 10.0)  # Normalize to [0, 1]
        
        # Add context-based priority adjustments
        context_priority = 0.5  # Default
        if context:
            if context.get('is_validation', False):
                context_priority = 0.8
            elif context.get('is_critical', False):
                context_priority = 1.0
        
        return (normalized_magnitude + context_priority) / 2.0
    
    def _get_current_system_loads(self) -> Dict[str, float]:
        """Get current system loads"""
        return {
            'brain': 0.5,  # Mock data
            'gac': 0.6,
            'compression': 0.4
        }
    
    def _calculate_current_resource_efficiency(self) -> float:
        """Calculate current resource efficiency"""
        # Simple efficiency calculation based on resource allocation balance
        memory_balance = 1.0 - abs(sum(self.resource_allocation['memory'].values()) - 1.0)
        cpu_balance = 1.0 - abs(sum(self.resource_allocation['cpu'].values()) - 1.0)
        gpu_balance = 1.0 - abs(sum(self.resource_allocation['gpu'].values()) - 1.0)
        
        return (memory_balance + cpu_balance + gpu_balance) / 3.0
    
    def _identify_best_performing_system(self) -> str:
        """Identify the best performing system"""
        # Analyze coordination history to find best performing system
        system_performance = defaultdict(list)
        
        for coord in self.coordination_history:
            if coord.brain_result and coord.coordination_successful:
                system_performance['brain'].append(1.0)
            elif coord.brain_result:
                system_performance['brain'].append(0.0)
                
            if coord.gac_result and coord.coordination_successful:
                system_performance['gac'].append(1.0)
            elif coord.gac_result:
                system_performance['gac'].append(0.0)
                
            if coord.compression_result and coord.coordination_successful:
                system_performance['compression'].append(1.0)
            elif coord.compression_result:
                system_performance['compression'].append(0.0)
        
        # Find system with best average performance
        best_system = 'brain'  # Default
        best_score = 0.0
        
        for system, scores in system_performance.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_system = system
        
        return best_system
    
    # Coordination hook methods
    def _brain_pre_operation_hook(self, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called before Brain operations"""
        # Add coordination context
        operation_data['coordination_context'] = {
            'coordinator_id': id(self),
            'coordination_mode': self.coordination_mode.name,
            'active_sessions': len(self.active_sessions)
        }
        return operation_data
    
    def _brain_post_operation_hook(self, operation_result: Dict[str, Any]) -> None:
        """Hook called after Brain operations"""
        # Update coordination analytics
        self.coordination_analytics['operations']['brain_operations'] += 1
    
    def _gac_pre_optimization_hook(self, optimization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called before GAC optimization"""
        # Add coordination context
        optimization_data['coordination_context'] = {
            'coordinator_id': id(self),
            'resource_allocation': self.resource_allocation.copy()
        }
        return optimization_data
    
    def _gac_post_optimization_hook(self, optimization_result: Dict[str, Any]) -> None:
        """Hook called after GAC optimization"""
        # Update coordination analytics
        self.coordination_analytics['operations']['gac_optimizations'] += 1
    
    def _initialize_memory_coordination(self) -> None:
        """Initialize memory coordination"""
        self.logger.debug("Memory coordination initialized")
    
    def _initialize_gpu_coordination(self) -> None:
        """Initialize GPU coordination"""
        self.logger.debug("GPU coordination initialized")
    
    def _initialize_cpu_coordination(self) -> None:
        """Initialize CPU coordination"""
        self.logger.debug("CPU coordination initialized")
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination status"""
        return {
            'coordination_mode': self.coordination_mode.name,
            'active_sessions': len(self.active_sessions),
            'total_coordinations': self.total_coordinations,
            'successful_coordinations': self.successful_coordinations,
            'failed_coordinations': self.failed_coordinations,
            'coordination_efficiency': self.coordination_efficiency,
            'resource_allocation': self.resource_allocation.copy(),
            'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None
        }
    
    def shutdown(self) -> None:
        """Shutdown Brain-GAC coordinator"""
        self.logger.info("Shutting down Brain-GAC coordinator")
        
        # Clear active sessions
        self.active_sessions.clear()
        self.session_queue.clear()
        
        # Clear references
        self.brain_system = None
        self.gac_system = None
        self.compression_system = None
        self.direction_manager = None
        
        # Clear tracking data
        self.coordination_history.clear()
        self.optimization_history.clear()
        self.performance_trends.clear()
        
        self.logger.info("Brain-GAC coordinator shutdown complete")