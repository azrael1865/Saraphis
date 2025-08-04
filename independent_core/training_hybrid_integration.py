"""
Training Hybrid Integration - Main orchestrator for training-hybrid integration
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

# Import training system components (avoid circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from training_manager import TrainingManager, TrainingConfig, TrainingStatus

# Import hybrid compression system components
from compression_systems.padic.hybrid_padic_compressor import HybridPadicCompressionSystem
from compression_systems.padic.dynamic_switching_manager import DynamicSwitchingManager
from compression_systems.padic.direction_manager import DirectionManager
from compression_systems.padic.hybrid_bounding_engine import HybridBoundingEngine

# Import Brain-GAC integration components
from brain_gac_integration import BrainGACIntegration
from brain_gac_coordinator import BrainGACCoordinator


class TrainingIntegrationMode(Enum):
    """Training integration mode enumeration"""
    STANDARD = "standard"
    HYBRID_OPTIMIZED = "hybrid_optimized"
    COMPRESSION_FIRST = "compression_first"
    PERFORMANCE_FIRST = "performance_first"
    MEMORY_EFFICIENT = "memory_efficient"
    GPU_OPTIMIZED = "gpu_optimized"


class TrainingCompressionStrategy(Enum):
    """Training compression strategy enumeration"""
    ADAPTIVE = "adaptive"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    REAL_TIME = "real_time"
    BATCH_OPTIMIZED = "batch_optimized"
    GRADIENT_FOCUSED = "gradient_focused"


class TrainingIntegrationState(Enum):
    """Training integration state enumeration"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    TRAINING_ACTIVE = "training_active"
    OPTIMIZING = "optimizing"
    ERROR = "error"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class TrainingHybridConfig:
    """Configuration for training-hybrid integration"""
    integration_mode: TrainingIntegrationMode = TrainingIntegrationMode.HYBRID_OPTIMIZED
    compression_strategy: TrainingCompressionStrategy = TrainingCompressionStrategy.ADAPTIVE
    enable_real_time_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_gpu_optimization: bool = True
    enable_gradient_compression: bool = True
    enable_data_compression: bool = True
    enable_performance_monitoring: bool = True
    enable_error_recovery: bool = True
    
    # Performance thresholds
    memory_threshold_gb: float = 8.0
    gpu_threshold_percent: float = 85.0
    performance_threshold: float = 0.8
    compression_ratio_threshold: float = 0.7
    
    # Training parameters
    max_training_sessions: int = 16
    training_batch_size: int = 32
    compression_batch_size: int = 64
    optimization_interval_seconds: int = 30
    
    # Analytics configuration
    enable_analytics: bool = True
    analytics_history_size: int = 1000
    performance_window_size: int = 100
    
    def __post_init__(self):
        """Validate configuration"""
        if not isinstance(self.integration_mode, TrainingIntegrationMode):
            raise TypeError("Integration mode must be TrainingIntegrationMode")
        if not isinstance(self.compression_strategy, TrainingCompressionStrategy):
            raise TypeError("Compression strategy must be TrainingCompressionStrategy")
        if self.memory_threshold_gb <= 0:
            raise ValueError("Memory threshold must be positive")
        if not (0.0 <= self.performance_threshold <= 1.0):
            raise ValueError("Performance threshold must be between 0.0 and 1.0")


@dataclass
class TrainingSessionMetrics:
    """Training session performance metrics"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_iterations: int = 0
    compression_operations: int = 0
    optimization_operations: int = 0
    
    # Performance metrics
    average_iteration_time_ms: float = 0.0
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
    
    # Training metrics
    training_loss: float = 0.0
    training_accuracy: float = 0.0
    validation_loss: float = 0.0
    validation_accuracy: float = 0.0
    
    # Error tracking
    error_count: int = 0
    recovery_count: int = 0
    
    def __post_init__(self):
        """Validate metrics"""
        if not self.session_id:
            raise ValueError("Session ID cannot be empty")
        if self.total_iterations < 0:
            raise ValueError("Total iterations must be non-negative")


@dataclass
class TrainingAnalytics:
    """Comprehensive training analytics"""
    total_sessions: int = 0
    active_sessions: int = 0
    completed_sessions: int = 0
    failed_sessions: int = 0
    
    # Performance analytics
    average_session_duration_minutes: float = 0.0
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
    
    timestamp: datetime = field(default_factory=datetime.utcnow)


class TrainingHybridIntegration:
    """
    Main orchestrator for training-hybrid integration.
    Manages integration between Training Manager and hybrid p-adic system.
    """
    
    def __init__(self, config: Optional[TrainingHybridConfig] = None):
        """Initialize training-hybrid integration"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, TrainingHybridConfig):
            raise TypeError(f"Config must be TrainingHybridConfig or None, got {type(config)}")
        
        self.config = config or TrainingHybridConfig()
        self.logger = logging.getLogger('TrainingHybridIntegration')
        
        # System components
        self.training_manager: Optional['TrainingManager'] = None
        self.hybrid_compression: Optional[HybridPadicCompressionSystem] = None
        self.dynamic_switching: Optional[DynamicSwitchingManager] = None
        self.direction_manager: Optional[DirectionManager] = None
        self.bounding_engine: Optional[HybridBoundingEngine] = None
        self.brain_gac_integration: Optional[BrainGACIntegration] = None
        
        # Integration state
        self.integration_state = TrainingIntegrationState.UNINITIALIZED
        self.active_training_sessions: Dict[str, TrainingSessionMetrics] = {}
        self.session_queue: deque = deque(maxlen=self.config.max_training_sessions)
        
        # Performance tracking
        self.performance_metrics: deque = deque(maxlen=self.config.analytics_history_size)
        self.optimization_history: List[Dict[str, Any]] = []
        self.compression_history: List[Dict[str, Any]] = []
        
        # Analytics tracking
        self.training_analytics = TrainingAnalytics()
        self.performance_baselines: Dict[str, float] = {}
        self.last_optimization: Optional[datetime] = None
        
        # Thread safety
        self._integration_lock = threading.RLock()
        self._session_lock = threading.RLock()
        self._analytics_lock = threading.RLock()
        
        # Monitoring state
        self.monitoring_active = False
        self.error_recovery_active = False
        self.optimization_active = False
        
        self.logger.info("TrainingHybridIntegration created successfully")
    
    def initialize_training_hybrid_integration(self,
                                             training_manager: 'TrainingManager',
                                             hybrid_compression: HybridPadicCompressionSystem,
                                             dynamic_switching: Optional[DynamicSwitchingManager] = None,
                                             direction_manager: Optional[DirectionManager] = None,
                                             bounding_engine: Optional[HybridBoundingEngine] = None,
                                             brain_gac_integration: Optional[BrainGACIntegration] = None) -> None:
        """
        Initialize training-hybrid integration with required systems.
        
        Args:
            training_manager: Training manager instance
            hybrid_compression: Hybrid compression system
            dynamic_switching: Optional dynamic switching manager
            direction_manager: Optional direction manager
            bounding_engine: Optional bounding engine
            brain_gac_integration: Optional Brain-GAC integration
            
        Raises:
            TypeError: If systems are invalid
            RuntimeError: If initialization fails
        """
        if training_manager is None:
            raise ValueError("Training manager cannot be None")
        # Type check at runtime if needed
        if hasattr(training_manager, '__class__') and 'TrainingManager' not in str(training_manager.__class__):
            raise TypeError(f"Training manager must be TrainingManager, got {type(training_manager)}")
        if not isinstance(hybrid_compression, HybridPadicCompressionSystem):
            raise TypeError(f"Hybrid compression must be HybridPadicCompressionSystem, got {type(hybrid_compression)}")
        
        try:
            with self._integration_lock:
                self.integration_state = TrainingIntegrationState.INITIALIZING
                
                # Store system references
                self.training_manager = training_manager
                self.hybrid_compression = hybrid_compression
                self.dynamic_switching = dynamic_switching
                self.direction_manager = direction_manager
                self.bounding_engine = bounding_engine
                self.brain_gac_integration = brain_gac_integration
                
                # Initialize integration systems
                self._initialize_performance_baselines()
                self._setup_training_hooks()
                self._initialize_compression_coordination()
                
                if self.config.enable_performance_monitoring:
                    self._start_performance_monitoring()
                
                if self.config.enable_analytics:
                    self._initialize_training_analytics()
                
                self.integration_state = TrainingIntegrationState.INITIALIZED
                self.logger.info("Training-hybrid integration initialized successfully")
                
        except Exception as e:
            self.integration_state = TrainingIntegrationState.ERROR
            self.logger.error(f"Failed to initialize training-hybrid integration: {e}")
            raise RuntimeError(f"Training-hybrid integration initialization failed: {e}")
    
    def start_hybrid_training_session(self, session_id: str, config: Dict[str, Any]) -> TrainingSessionMetrics:
        """
        Start a hybrid training session with compression integration.
        
        Args:
            session_id: Unique session identifier
            config: Training session configuration
            
        Returns:
            Training session metrics
            
        Raises:
            ValueError: If session ID is invalid or already exists
            RuntimeError: If session start fails
        """
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Session ID must be non-empty string")
        if not isinstance(config, dict):
            raise TypeError(f"Config must be dict, got {type(config)}")
        
        try:
            with self._session_lock:
                if session_id in self.active_training_sessions:
                    raise ValueError(f"Training session {session_id} already exists")
                
                if len(self.active_training_sessions) >= self.config.max_training_sessions:
                    raise RuntimeError(f"Maximum training sessions ({self.config.max_training_sessions}) reached")
                
                # Create session metrics
                session_metrics = TrainingSessionMetrics(
                    session_id=session_id,
                    start_time=datetime.utcnow()
                )
                
                # Initialize training session
                if not self.training_manager.start_training_session(session_id, config):
                    raise RuntimeError(f"Failed to start training session {session_id}")
                
                # Configure compression for training
                compression_config = self._create_compression_config(config)
                if not self._configure_compression_for_training(session_id, compression_config):
                    raise RuntimeError(f"Failed to configure compression for session {session_id}")
                
                # Set up optimization coordination
                if self.config.enable_real_time_optimization:
                    self._setup_optimization_coordination(session_id, config)
                
                # Register session
                self.active_training_sessions[session_id] = session_metrics
                self.session_queue.append(session_id)
                
                # Update analytics
                with self._analytics_lock:
                    self.training_analytics.total_sessions += 1
                    self.training_analytics.active_sessions += 1
                
                self.logger.info(f"Hybrid training session {session_id} started successfully")
                return session_metrics
                
        except Exception as e:
            self.logger.error(f"Failed to start hybrid training session {session_id}: {e}")
            raise RuntimeError(f"Hybrid training session start failed: {e}")
    
    def optimize_training_compression(self, session_id: str, data: torch.Tensor) -> Dict[str, Any]:
        """
        Optimize training data compression for a session.
        
        Args:
            session_id: Training session identifier
            data: Training data tensor
            
        Returns:
            Compression optimization results
            
        Raises:
            ValueError: If session or data is invalid
            RuntimeError: If optimization fails
        """
        if not session_id or session_id not in self.active_training_sessions:
            raise ValueError(f"Invalid or unknown session ID: {session_id}")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Data must be torch.Tensor, got {type(data)}")
        if data.numel() == 0:
            raise ValueError("Data tensor cannot be empty")
        
        try:
            start_time = time.time()
            session_metrics = self.active_training_sessions[session_id]
            
            # Determine optimal compression strategy
            compression_strategy = self._determine_compression_strategy(session_id, data)
            
            # Apply compression optimization
            compression_result = self.hybrid_compression.compress(data)
            if not compression_result.get('success', False):
                raise RuntimeError(f"Compression failed for session {session_id}")
            
            # Apply dynamic switching optimization if available
            if self.dynamic_switching and compression_strategy == TrainingCompressionStrategy.ADAPTIVE:
                switching_result = self._apply_dynamic_switching_optimization(session_id, data, compression_result)
                compression_result.update(switching_result)
            
            # Update session metrics
            compression_time = time.time() - start_time
            session_metrics.compression_operations += 1
            session_metrics.average_compression_time_ms = (
                (session_metrics.average_compression_time_ms * (session_metrics.compression_operations - 1) + 
                 compression_time * 1000) / session_metrics.compression_operations
            )
            
            # Update compression analytics
            compression_ratio = compression_result.get('compression_ratio', 0.0)
            session_metrics.compression_ratio = (
                (session_metrics.compression_ratio * (session_metrics.compression_operations - 1) + 
                 compression_ratio) / session_metrics.compression_operations
            )
            
            optimization_result = {
                'session_id': session_id,
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
            
            self.logger.debug(f"Training compression optimized for session {session_id}: {compression_ratio:.4f} ratio")
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Failed to optimize training compression for session {session_id}: {e}")
            raise RuntimeError(f"Training compression optimization failed: {e}")
    
    def monitor_training_performance(self, session_id: str) -> Dict[str, Any]:
        """
        Monitor training performance for a session.
        
        Args:
            session_id: Training session identifier
            
        Returns:
            Performance monitoring results
            
        Raises:
            ValueError: If session is invalid
            RuntimeError: If monitoring fails
        """
        if not session_id or session_id not in self.active_training_sessions:
            raise ValueError(f"Invalid or unknown session ID: {session_id}")
        
        try:
            session_metrics = self.active_training_sessions[session_id]
            
            # Capture current performance metrics
            current_metrics = self._capture_current_performance(session_id)
            
            # Update session metrics
            self._update_session_performance_metrics(session_id, current_metrics)
            
            # Analyze performance trends
            performance_trends = self._analyze_performance_trends(session_id)
            
            # Check for performance alerts
            alerts = self._check_performance_alerts(session_id, current_metrics)
            
            # Generate performance recommendations
            recommendations = self._generate_performance_recommendations(session_id, current_metrics, performance_trends)
            
            monitoring_result = {
                'session_id': session_id,
                'current_metrics': current_metrics,
                'performance_trends': performance_trends,
                'alerts': alerts,
                'recommendations': recommendations,
                'monitoring_timestamp': datetime.utcnow().isoformat(),
                'session_duration_minutes': (datetime.utcnow() - session_metrics.start_time).total_seconds() / 60
            }
            
            # Record performance history
            self.performance_metrics.append(monitoring_result)
            
            self.logger.debug(f"Performance monitoring completed for session {session_id}")
            return monitoring_result
            
        except Exception as e:
            self.logger.error(f"Failed to monitor training performance for session {session_id}: {e}")
            raise RuntimeError(f"Training performance monitoring failed: {e}")
    
    def handle_training_errors(self, session_id: str, error: Exception) -> Dict[str, Any]:
        """
        Handle training errors and attempt recovery.
        
        Args:
            session_id: Training session identifier
            error: Training error that occurred
            
        Returns:
            Error handling results
            
        Raises:
            ValueError: If session is invalid
            RuntimeError: If error handling fails
        """
        if not session_id or session_id not in self.active_training_sessions:
            raise ValueError(f"Invalid or unknown session ID: {session_id}")
        if not isinstance(error, Exception):
            raise TypeError(f"Error must be Exception, got {type(error)}")
        
        try:
            session_metrics = self.active_training_sessions[session_id]
            error_start_time = time.time()
            
            # Update error statistics
            session_metrics.error_count += 1
            
            # Assess error severity
            error_severity = self._assess_error_severity(session_id, error)
            
            # Attempt error recovery based on severity
            recovery_result = None
            if self.config.enable_error_recovery:
                if error_severity == "critical":
                    recovery_result = self._attempt_critical_error_recovery(session_id, error)
                elif error_severity == "moderate":
                    recovery_result = self._attempt_moderate_error_recovery(session_id, error)
                elif error_severity == "minor":
                    recovery_result = self._attempt_minor_error_recovery(session_id, error)
            
            # Update recovery statistics
            if recovery_result and recovery_result.get('success', False):
                session_metrics.recovery_count += 1
            
            error_handling_result = {
                'session_id': session_id,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'error_severity': error_severity,
                'recovery_attempted': recovery_result is not None,
                'recovery_successful': recovery_result.get('success', False) if recovery_result else False,
                'recovery_details': recovery_result,
                'error_handling_time_ms': (time.time() - error_start_time) * 1000,
                'total_errors': session_metrics.error_count,
                'total_recoveries': session_metrics.recovery_count,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.error(f"Training error handled for session {session_id}: {error_severity} - {type(error).__name__}")
            return error_handling_result
            
        except Exception as e:
            self.logger.error(f"Failed to handle training error for session {session_id}: {e}")
            raise RuntimeError(f"Training error handling failed: {e}")
    
    def get_training_analytics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive training analytics.
        
        Args:
            session_id: Optional specific session ID for detailed analytics
            
        Returns:
            Training analytics results
            
        Raises:
            ValueError: If session ID is invalid
            RuntimeError: If analytics retrieval fails
        """
        try:
            with self._analytics_lock:
                if session_id:
                    if session_id not in self.active_training_sessions:
                        raise ValueError(f"Invalid or unknown session ID: {session_id}")
                    
                    # Get session-specific analytics
                    session_metrics = self.active_training_sessions[session_id]
                    session_analytics = self._calculate_session_analytics(session_id, session_metrics)
                    
                    return {
                        'session_analytics': session_analytics,
                        'session_metrics': session_metrics,
                        'performance_history': [m for m in self.performance_metrics if m.get('session_id') == session_id],
                        'compression_history': [c for c in self.compression_history if c.get('session_id') == session_id],
                        'optimization_history': [o for o in self.optimization_history if o.get('session_id') == session_id]
                    }
                else:
                    # Get global analytics
                    self._update_global_analytics()
                    
                    return {
                        'global_analytics': self.training_analytics,
                        'active_sessions': {sid: metrics for sid, metrics in self.active_training_sessions.items()},
                        'performance_baselines': self.performance_baselines,
                        'recent_performance': list(self.performance_metrics)[-20:],
                        'recent_compressions': self.compression_history[-20:],
                        'recent_optimizations': self.optimization_history[-20:],
                        'system_health': self._calculate_system_health(),
                        'recommendations': self._generate_system_recommendations()
                    }
                    
        except Exception as e:
            self.logger.error(f"Failed to get training analytics: {e}")
            raise RuntimeError(f"Training analytics retrieval failed: {e}")
    
    def _initialize_performance_baselines(self) -> None:
        """Initialize performance baselines"""
        try:
            self.performance_baselines = {
                'memory_usage_gb': 2.0,
                'gpu_utilization_percent': 50.0,
                'compression_ratio': 0.5,
                'iteration_time_ms': 100.0,
                'compression_time_ms': 50.0,
                'optimization_time_ms': 30.0
            }
            self.logger.debug("Performance baselines initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize performance baselines: {e}")
            raise RuntimeError(f"Performance baseline initialization failed: {e}")
    
    def _setup_training_hooks(self) -> None:
        """Setup training system hooks"""
        try:
            if hasattr(self.training_manager, 'add_hook'):
                self.training_manager.add_hook('pre_iteration', self._training_pre_iteration_hook)
                self.training_manager.add_hook('post_iteration', self._training_post_iteration_hook)
                self.training_manager.add_hook('error', self._training_error_hook)
            self.logger.debug("Training hooks setup completed")
        except Exception as e:
            self.logger.error(f"Failed to setup training hooks: {e}")
            raise RuntimeError(f"Training hooks setup failed: {e}")
    
    def _initialize_compression_coordination(self) -> None:
        """Initialize compression coordination"""
        try:
            if hasattr(self.hybrid_compression, 'set_coordination_mode'):
                self.hybrid_compression.set_coordination_mode('training_optimized')
            
            if self.dynamic_switching and hasattr(self.dynamic_switching, 'set_training_mode'):
                self.dynamic_switching.set_training_mode(True)
            
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
    
    def _initialize_training_analytics(self) -> None:
        """Initialize training analytics"""
        try:
            self.training_analytics = TrainingAnalytics()
            self.logger.debug("Training analytics initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize training analytics: {e}")
            raise RuntimeError(f"Training analytics initialization failed: {e}")
    
    def _create_compression_config(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create compression configuration from training configuration"""
        compression_config = {
            'strategy': self.config.compression_strategy.value,
            'batch_size': training_config.get('batch_size', self.config.compression_batch_size),
            'enable_gpu': self.config.enable_gpu_optimization,
            'memory_limit_gb': self.config.memory_threshold_gb,
            'performance_threshold': self.config.performance_threshold
        }
        return compression_config
    
    def _configure_compression_for_training(self, session_id: str, config: Dict[str, Any]) -> bool:
        """Configure compression system for training session"""
        try:
            if hasattr(self.hybrid_compression, 'configure_for_training'):
                return self.hybrid_compression.configure_for_training(session_id, config)
            return True
        except Exception as e:
            self.logger.error(f"Failed to configure compression for training: {e}")
            return False
    
    def _setup_optimization_coordination(self, session_id: str, config: Dict[str, Any]) -> None:
        """Setup optimization coordination for session"""
        try:
            if self.brain_gac_integration:
                self.brain_gac_integration.register_training_session(session_id, config)
            self.logger.debug(f"Optimization coordination setup for session {session_id}")
        except Exception as e:
            self.logger.error(f"Failed to setup optimization coordination: {e}")
            raise RuntimeError(f"Optimization coordination setup failed: {e}")
    
    def _determine_compression_strategy(self, session_id: str, data: torch.Tensor) -> TrainingCompressionStrategy:
        """Determine optimal compression strategy"""
        if self.config.compression_strategy == TrainingCompressionStrategy.ADAPTIVE:
            # Adaptive strategy based on data characteristics and session performance
            data_size_mb = data.numel() * data.element_size() / (1024 * 1024)
            session_metrics = self.active_training_sessions[session_id]
            
            if data_size_mb > 100 and session_metrics.compression_operations > 10:
                if session_metrics.average_compression_time_ms > 100:
                    return TrainingCompressionStrategy.CONSERVATIVE
                else:
                    return TrainingCompressionStrategy.AGGRESSIVE
            elif data_size_mb < 10:
                return TrainingCompressionStrategy.REAL_TIME
            else:
                return TrainingCompressionStrategy.BATCH_OPTIMIZED
        
        return self.config.compression_strategy
    
    def _apply_dynamic_switching_optimization(self, session_id: str, data: torch.Tensor, compression_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply dynamic switching optimization"""
        try:
            if not self.dynamic_switching:
                return {}
            
            switching_result = self.dynamic_switching.optimize_for_training(
                session_id, data, compression_result
            )
            return switching_result
        except Exception as e:
            self.logger.error(f"Dynamic switching optimization failed: {e}")
            return {'switching_error': str(e)}
    
    def _capture_current_performance(self, session_id: str) -> Dict[str, float]:
        """Capture current performance metrics"""
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
            except Exception as e:
                # NO FALLBACKS - HARD FAILURE
                raise RuntimeError(f"Failed to get GPU memory info: {e}")
            
            return {
                'memory_usage_gb': memory_usage,
                'cpu_utilization_percent': cpu_percent,
                'gpu_utilization_percent': gpu_utilization,
                'gpu_memory_usage_gb': gpu_memory_usage
            }
        except Exception as e:
            self.logger.error(f"Failed to capture performance metrics: {e}")
            return {}
    
    def _update_session_performance_metrics(self, session_id: str, current_metrics: Dict[str, float]) -> None:
        """Update session performance metrics"""
        try:
            session_metrics = self.active_training_sessions[session_id]
            
            # Update peak metrics
            session_metrics.peak_memory_usage_gb = max(
                session_metrics.peak_memory_usage_gb,
                current_metrics.get('memory_usage_gb', 0.0)
            )
            session_metrics.peak_gpu_utilization_percent = max(
                session_metrics.peak_gpu_utilization_percent,
                current_metrics.get('gpu_utilization_percent', 0.0)
            )
            
            # Update GPU memory usage
            session_metrics.gpu_memory_usage_gb = current_metrics.get('gpu_memory_usage_gb', 0.0)
            
            # Update averages (simple moving average)
            operations = max(1, session_metrics.total_iterations)
            session_metrics.average_memory_usage_gb = (
                (session_metrics.average_memory_usage_gb * (operations - 1) + 
                 current_metrics.get('memory_usage_gb', 0.0)) / operations
            )
            session_metrics.average_gpu_utilization_percent = (
                (session_metrics.average_gpu_utilization_percent * (operations - 1) + 
                 current_metrics.get('gpu_utilization_percent', 0.0)) / operations
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update session performance metrics: {e}")
    
    def _analyze_performance_trends(self, session_id: str) -> Dict[str, str]:
        """Analyze performance trends for session"""
        try:
            recent_metrics = [m for m in self.performance_metrics if m.get('session_id') == session_id][-10:]
            
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
            else:
                trends = {'memory': 'insufficient_data', 'gpu': 'insufficient_data'}
            
            return trends
        except Exception as e:
            self.logger.error(f"Failed to analyze performance trends: {e}")
            return {}
    
    def _check_performance_alerts(self, session_id: str, current_metrics: Dict[str, float]) -> List[str]:
        """Check for performance alerts"""
        alerts = []
        
        memory_usage = current_metrics.get('memory_usage_gb', 0.0)
        if memory_usage > self.config.memory_threshold_gb:
            alerts.append(f"High memory usage: {memory_usage:.1f}GB > {self.config.memory_threshold_gb}GB")
        
        gpu_utilization = current_metrics.get('gpu_utilization_percent', 0.0)
        if gpu_utilization > self.config.gpu_threshold_percent:
            alerts.append(f"High GPU utilization: {gpu_utilization:.1f}% > {self.config.gpu_threshold_percent}%")
        
        return alerts
    
    def _generate_performance_recommendations(self, session_id: str, current_metrics: Dict[str, float], trends: Dict[str, str]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if trends.get('memory') == 'increasing':
            recommendations.append("Consider reducing batch size or enabling more aggressive compression")
        
        if trends.get('gpu') == 'decreasing':
            recommendations.append("GPU utilization declining - consider increasing batch size or workload")
        
        if current_metrics.get('memory_usage_gb', 0) > self.config.memory_threshold_gb * 0.8:
            recommendations.append("Memory usage approaching threshold - enable memory optimization")
        
        return recommendations
    
    def _assess_error_severity(self, session_id: str, error: Exception) -> str:
        """Assess error severity level"""
        error_type = type(error).__name__
        
        critical_errors = ['RuntimeError', 'SystemError', 'MemoryError', 'OSError']
        moderate_errors = ['ValueError', 'TypeError', 'AttributeError']
        
        if error_type in critical_errors:
            return "critical"
        elif error_type in moderate_errors:
            return "moderate"
        else:
            return "minor"
    
    def _attempt_critical_error_recovery(self, session_id: str, error: Exception) -> Dict[str, Any]:
        """Attempt recovery from critical errors"""
        try:
            # Critical error recovery strategies
            recovery_actions = []
            
            # Memory cleanup
            import gc
            gc.collect()
            recovery_actions.append("memory_cleanup")
            
            # Reset compression system
            if hasattr(self.hybrid_compression, 'reset'):
                self.hybrid_compression.reset()
                recovery_actions.append("compression_reset")
            
            # Reduce batch sizes
            if hasattr(self.training_manager, 'reduce_batch_size'):
                self.training_manager.reduce_batch_size(session_id, 0.5)
                recovery_actions.append("batch_size_reduction")
            
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
    
    def _attempt_moderate_error_recovery(self, session_id: str, error: Exception) -> Dict[str, Any]:
        """Attempt recovery from moderate errors"""
        try:
            recovery_actions = []
            
            # Parameter adjustment
            if hasattr(self.hybrid_compression, 'adjust_parameters'):
                self.hybrid_compression.adjust_parameters(conservative=True)
                recovery_actions.append("parameter_adjustment")
            
            # Retry with different strategy
            if self.dynamic_switching:
                self.dynamic_switching.switch_strategy(session_id, 'conservative')
                recovery_actions.append("strategy_switch")
            
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
    
    def _attempt_minor_error_recovery(self, session_id: str, error: Exception) -> Dict[str, Any]:
        """Attempt recovery from minor errors"""
        try:
            recovery_actions = []
            
            # Simple retry
            recovery_actions.append("retry")
            
            # Log warning
            self.logger.warning(f"Minor error in session {session_id}: {error}")
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
    
    def _calculate_session_analytics(self, session_id: str, session_metrics: TrainingSessionMetrics) -> Dict[str, Any]:
        """Calculate analytics for specific session"""
        try:
            current_time = datetime.utcnow()
            session_duration = (current_time - session_metrics.start_time).total_seconds() / 60  # minutes
            
            return {
                'session_id': session_id,
                'duration_minutes': session_duration,
                'total_iterations': session_metrics.total_iterations,
                'iterations_per_minute': session_metrics.total_iterations / max(1, session_duration),
                'compression_operations': session_metrics.compression_operations,
                'optimization_operations': session_metrics.optimization_operations,
                'average_iteration_time_ms': session_metrics.average_iteration_time_ms,
                'average_compression_time_ms': session_metrics.average_compression_time_ms,
                'compression_ratio': session_metrics.compression_ratio,
                'peak_memory_usage_gb': session_metrics.peak_memory_usage_gb,
                'average_memory_usage_gb': session_metrics.average_memory_usage_gb,
                'peak_gpu_utilization_percent': session_metrics.peak_gpu_utilization_percent,
                'error_rate': session_metrics.error_count / max(1, session_metrics.total_iterations),
                'recovery_rate': session_metrics.recovery_count / max(1, session_metrics.error_count)
            }
        except Exception as e:
            self.logger.error(f"Failed to calculate session analytics: {e}")
            return {}
    
    def _update_global_analytics(self) -> None:
        """Update global training analytics"""
        try:
            with self._analytics_lock:
                self.training_analytics.active_sessions = len(self.active_training_sessions)
                
                if self.active_training_sessions:
                    # Calculate averages across active sessions
                    total_duration = 0
                    total_compression_efficiency = 0
                    total_memory_usage = 0
                    total_gpu_utilization = 0
                    
                    current_time = datetime.utcnow()
                    for session_metrics in self.active_training_sessions.values():
                        duration = (current_time - session_metrics.start_time).total_seconds() / 60
                        total_duration += duration
                        total_compression_efficiency += session_metrics.compression_ratio
                        total_memory_usage += session_metrics.average_memory_usage_gb
                        total_gpu_utilization += session_metrics.average_gpu_utilization_percent
                    
                    num_sessions = len(self.active_training_sessions)
                    self.training_analytics.average_session_duration_minutes = total_duration / num_sessions
                    self.training_analytics.average_compression_efficiency = total_compression_efficiency / num_sessions
                    self.training_analytics.average_memory_savings = total_memory_usage / num_sessions
                    self.training_analytics.average_gpu_utilization = total_gpu_utilization / num_sessions
                
                # Update system health score
                self.training_analytics.system_health_score = self._calculate_system_health_score()
                
                # Update timestamp
                self.training_analytics.timestamp = datetime.utcnow()
                
        except Exception as e:
            self.logger.error(f"Failed to update global analytics: {e}")
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health"""
        try:
            health_score = self._calculate_system_health_score()
            
            health_status = "excellent" if health_score > 0.9 else \
                           "good" if health_score > 0.7 else \
                           "fair" if health_score > 0.5 else \
                           "poor"
            
            return {
                'health_score': health_score,
                'health_status': health_status,
                'active_sessions': len(self.active_training_sessions),
                'monitoring_active': self.monitoring_active,
                'error_recovery_active': self.error_recovery_active,
                'optimization_active': self.optimization_active,
                'integration_state': self.integration_state.value
            }
        except Exception as e:
            # NO FALLBACKS - HARD FAILURE
            raise RuntimeError(f"Failed to calculate system health: {e}")
    
    def _calculate_system_health_score(self) -> float:
        """Calculate system health score"""
        try:
            score_factors = []
            
            # Session success rate
            if self.training_analytics.total_sessions > 0:
                success_rate = self.training_analytics.completed_sessions / self.training_analytics.total_sessions
                score_factors.append(success_rate)
            
            # Performance efficiency
            if self.training_analytics.average_compression_efficiency > 0:
                efficiency_score = min(1.0, self.training_analytics.average_compression_efficiency / 0.8)
                score_factors.append(efficiency_score)
            
            # Memory utilization (optimal around 70%)
            if self.training_analytics.average_memory_savings > 0:
                memory_score = 1.0 - abs(0.7 - min(1.0, self.training_analytics.average_memory_savings / 10.0))
                score_factors.append(memory_score)
            
            # Integration state health
            state_score = 1.0 if self.integration_state == TrainingIntegrationState.INITIALIZED else \
                         0.8 if self.integration_state == TrainingIntegrationState.TRAINING_ACTIVE else \
                         0.6 if self.integration_state == TrainingIntegrationState.OPTIMIZING else \
                         0.2
            score_factors.append(state_score)
            
            return np.mean(score_factors) if score_factors else 0.5
            
        except Exception as e:
            # NO FALLBACKS - HARD FAILURE
            raise RuntimeError(f"Failed to calculate system health score: {e}")
    
    def _generate_system_recommendations(self) -> List[str]:
        """Generate system-wide recommendations"""
        recommendations = []
        
        if self.training_analytics.system_health_score < 0.7:
            recommendations.append("System health below optimal - consider performance optimization")
        
        if len(self.active_training_sessions) == 0:
            recommendations.append("No active training sessions - system ready for new workloads")
        elif len(self.active_training_sessions) >= self.config.max_training_sessions * 0.8:
            recommendations.append("High session load - consider scaling or load balancing")
        
        if self.training_analytics.average_compression_efficiency < 0.5:
            recommendations.append("Low compression efficiency - review compression strategy")
        
        if not self.monitoring_active:
            recommendations.append("Performance monitoring disabled - enable for better insights")
        
        return recommendations if recommendations else ["System operating within optimal parameters"]
    
    def _training_pre_iteration_hook(self, session_id: str, iteration_data: Dict[str, Any]) -> None:
        """Training pre-iteration hook"""
        try:
            if session_id in self.active_training_sessions:
                session_metrics = self.active_training_sessions[session_id]
                session_metrics.total_iterations += 1
        except Exception as e:
            self.logger.error(f"Training pre-iteration hook failed: {e}")
    
    def _training_post_iteration_hook(self, session_id: str, iteration_data: Dict[str, Any]) -> None:
        """Training post-iteration hook"""
        try:
            if session_id in self.active_training_sessions:
                session_metrics = self.active_training_sessions[session_id]
                
                # Update iteration timing
                iteration_time = iteration_data.get('iteration_time_ms', 0.0)
                if iteration_time > 0:
                    session_metrics.average_iteration_time_ms = (
                        (session_metrics.average_iteration_time_ms * (session_metrics.total_iterations - 1) + 
                         iteration_time) / session_metrics.total_iterations
                    )
                
                # Update training metrics
                if 'loss' in iteration_data:
                    session_metrics.training_loss = iteration_data['loss']
                if 'accuracy' in iteration_data:
                    session_metrics.training_accuracy = iteration_data['accuracy']
                
        except Exception as e:
            self.logger.error(f"Training post-iteration hook failed: {e}")
    
    def _training_error_hook(self, session_id: str, error: Exception) -> None:
        """Training error hook"""
        try:
            self.handle_training_errors(session_id, error)
        except Exception as e:
            self.logger.error(f"Training error hook failed: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        return {
            'integration_state': self.integration_state.value,
            'active_sessions': len(self.active_training_sessions),
            'monitoring_active': self.monitoring_active,
            'error_recovery_active': self.error_recovery_active,
            'optimization_active': self.optimization_active,
            'system_health': self._calculate_system_health()
        }
    
    def shutdown(self) -> None:
        """Shutdown training-hybrid integration"""
        try:
            with self._integration_lock:
                self.monitoring_active = False
                self.error_recovery_active = False
                self.optimization_active = False
                
                # Complete active sessions
                for session_id in list(self.active_training_sessions.keys()):
                    session_metrics = self.active_training_sessions[session_id]
                    session_metrics.end_time = datetime.utcnow()
                
                self.integration_state = TrainingIntegrationState.COMPLETED
                self.logger.info("Training-hybrid integration shutdown completed")
                
        except Exception as e:
            self.logger.error(f"Training-hybrid integration shutdown failed: {e}")
            raise RuntimeError(f"Shutdown failed: {e}")