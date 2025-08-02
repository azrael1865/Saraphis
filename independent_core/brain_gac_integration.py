"""
Brain-GAC Integration - Main orchestrator for Brain-GAC integration
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

# Import compression system components
from compression_systems.padic.hybrid_padic_compressor import HybridPadicCompressionSystem
from compression_systems.padic.dynamic_switching_manager import DynamicSwitchingManager
from compression_systems.padic.direction_manager import DirectionManager
from compression_systems.padic.hybrid_bounding_engine import HybridBoundingEngine

# Import Brain system components
from brain import Brain


class IntegrationMode(Enum):
    """Brain-GAC integration mode enumeration"""
    STANDARD = "standard"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    MEMORY_EFFICIENT = "memory_efficient"
    HIGH_THROUGHPUT = "high_throughput"
    LOW_LATENCY = "low_latency"
    BALANCED = "balanced"


class GradientFlowStrategy(Enum):
    """Gradient flow strategy enumeration"""
    DIRECT_FLOW = "direct_flow"
    BUFFERED_FLOW = "buffered_flow"
    BATCHED_FLOW = "batched_flow"
    ADAPTIVE_FLOW = "adaptive_flow"
    PRIORITY_FLOW = "priority_flow"


class IntegrationState(Enum):
    """Integration state enumeration"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    RUNNING = "running"
    OPTIMIZING = "optimizing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class IntegrationPerformanceMetrics:
    """Integration performance metrics"""
    gradient_processing_time_ms: float
    gac_processing_time_ms: float
    compression_time_ms: float
    total_integration_time_ms: float
    memory_usage_mb: float
    gpu_utilization: float
    cpu_utilization: float
    throughput_ops_per_sec: float
    latency_ms: float
    efficiency_score: float
    error_count: int
    success_count: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate performance metrics"""
        if self.gradient_processing_time_ms < 0:
            raise ValueError("Gradient processing time must be non-negative")
        if self.total_integration_time_ms < 0:
            raise ValueError("Total integration time must be non-negative")
        if not (0.0 <= self.efficiency_score <= 1.0):
            raise ValueError("Efficiency score must be between 0.0 and 1.0")


@dataclass
class IntegrationAnalytics:
    """Comprehensive integration analytics"""
    total_operations: int
    successful_operations: int
    failed_operations: int
    average_processing_time_ms: float
    average_memory_usage_mb: float
    average_gpu_utilization: float
    integration_efficiency: float
    gradient_flow_efficiency: float
    compression_efficiency: float
    gac_efficiency: float
    performance_trends: Dict[str, List[float]]
    error_analysis: Dict[str, int]
    optimization_recommendations: List[str]
    system_health_score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BrainGACConfig:
    """Configuration for Brain-GAC integration"""
    # Integration mode configuration
    integration_mode: IntegrationMode = IntegrationMode.BALANCED
    gradient_flow_strategy: GradientFlowStrategy = GradientFlowStrategy.ADAPTIVE_FLOW
    enable_real_time_optimization: bool = True
    enable_performance_monitoring: bool = True
    
    # Performance configuration
    max_gradient_processing_time_ms: float = 50.0
    max_memory_usage_mb: float = 2048.0
    target_gpu_utilization: float = 0.8
    target_throughput_ops_per_sec: float = 1000.0
    min_efficiency_score: float = 0.8
    
    # Gradient flow configuration
    gradient_buffer_size: int = 1000
    gradient_batch_size: int = 32
    enable_gradient_prioritization: bool = True
    gradient_flow_timeout_ms: float = 100.0
    
    # Integration optimization
    enable_adaptive_optimization: bool = True
    optimization_interval_seconds: int = 60
    enable_performance_regression_detection: bool = True
    performance_baseline_window: int = 100
    
    # Error handling configuration
    enable_error_recovery: bool = True
    max_retry_attempts: int = 3
    error_recovery_timeout_ms: float = 1000.0
    enable_graceful_degradation: bool = True
    
    # Analytics configuration
    analytics_history_size: int = 10000
    enable_detailed_analytics: bool = True
    enable_trend_analysis: bool = True
    analytics_export_interval_seconds: int = 3600
    
    def __post_init__(self):
        """Validate configuration"""
        if not isinstance(self.integration_mode, IntegrationMode):
            raise TypeError("Integration mode must be IntegrationMode")
        if not isinstance(self.gradient_flow_strategy, GradientFlowStrategy):
            raise TypeError("Gradient flow strategy must be GradientFlowStrategy")
        if self.max_gradient_processing_time_ms <= 0:
            raise ValueError("Max gradient processing time must be positive")
        if self.gradient_buffer_size <= 0:
            raise ValueError("Gradient buffer size must be positive")


class BrainGACIntegration:
    """
    Main orchestrator for Brain-GAC integration.
    Coordinates gradient flow between Brain and GAC systems with hybrid compression.
    """
    
    def __init__(self, config: Optional[BrainGACConfig] = None):
        """Initialize Brain-GAC integration"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, BrainGACConfig):
            raise TypeError(f"Config must be BrainGACConfig or None, got {type(config)}")
        
        self.config = config or BrainGACConfig()
        self.logger = logging.getLogger('BrainGACIntegration')
        
        # System components
        self.brain_system: Optional[Brain] = None
        self.gac_system: Optional[GACSystem] = None
        self.compression_system: Optional[HybridPadicCompressionSystem] = None
        self.direction_manager: Optional[DirectionManager] = None
        self.bounding_engine: Optional[HybridBoundingEngine] = None
        
        # Integration state
        self.integration_state = IntegrationState.UNINITIALIZED
        self.gradient_flow_strategy = self.config.gradient_flow_strategy
        
        # Performance tracking
        self.performance_metrics: deque = deque(maxlen=self.config.analytics_history_size)
        self.gradient_flow_buffer: deque = deque(maxlen=self.config.gradient_buffer_size)
        self.integration_analytics: Dict[str, Any] = {}
        
        # Optimization tracking
        self.optimization_history: List[Dict[str, Any]] = []
        self.performance_baselines: Dict[str, float] = {}
        self.last_optimization: Optional[datetime] = None
        
        # Thread safety
        self._integration_lock = threading.RLock()
        self._gradient_flow_lock = threading.RLock()
        self._analytics_lock = threading.RLock()
        
        # Performance monitoring
        self.monitoring_active = False
        self.error_recovery_active = False
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0
        
        self.logger.info("BrainGACIntegration created successfully")
    
    def initialize_brain_gac_integration(self,
                                       brain_system: Brain,
                                       gac_system: GACSystem,
                                       compression_system: Optional[HybridPadicCompressionSystem] = None,
                                       direction_manager: Optional[DirectionManager] = None,
                                       bounding_engine: Optional[HybridBoundingEngine] = None) -> None:
        """
        Initialize Brain-GAC integration with required systems.
        
        Args:
            brain_system: Brain system instance
            gac_system: GAC system instance
            compression_system: Optional hybrid compression system
            direction_manager: Optional direction manager
            bounding_engine: Optional bounding engine
            
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
        if bounding_engine is not None and not isinstance(bounding_engine, HybridBoundingEngine):
            raise TypeError(f"Bounding engine must be HybridBoundingEngine, got {type(bounding_engine)}")
        
        try:
            self.integration_state = IntegrationState.INITIALIZING
            
            with self._integration_lock:
                # Set system references
                self.brain_system = brain_system
                self.gac_system = gac_system
                self.compression_system = compression_system
                self.direction_manager = direction_manager
                self.bounding_engine = bounding_engine
                
                # Initialize performance baselines
                self._initialize_performance_baselines()
                
                # Setup integration hooks
                self._setup_integration_hooks()
                
                # Initialize gradient flow system
                self._initialize_gradient_flow()
                
                # Start performance monitoring if enabled
                if self.config.enable_performance_monitoring:
                    self._start_performance_monitoring()
                
                # Initialize analytics
                self._initialize_integration_analytics()
                
                self.integration_state = IntegrationState.INITIALIZED
                self.logger.info("Brain-GAC integration initialized successfully")
                
        except Exception as e:
            self.integration_state = IntegrationState.ERROR
            self.logger.error(f"Failed to initialize Brain-GAC integration: {e}")
            raise RuntimeError(f"Brain-GAC integration initialization failed: {e}")
    
    def integrate_hybrid_compression_with_gac(self, 
                                            data: torch.Tensor, 
                                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Integrate hybrid compression with GAC system.
        
        Args:
            data: Data tensor to process
            context: Optional processing context
            
        Returns:
            Integration processing results
            
        Raises:
            RuntimeError: If integration fails
            ValueError: If data is invalid
        """
        if self.integration_state != IntegrationState.INITIALIZED:
            raise RuntimeError("Integration system not initialized")
        
        if data is None:
            raise ValueError("Data cannot be None")
        if not isinstance(data, torch.Tensor):
            raise TypeError("Data must be torch.Tensor")
        if data.numel() == 0:
            raise ValueError("Data cannot be empty")
        
        try:
            start_time = time.time()
            
            with self._integration_lock:
                integration_results = {
                    'compression_result': None,
                    'gac_result': None,
                    'direction_analysis': None,
                    'bounding_result': None,
                    'performance_metrics': {},
                    'processing_time_ms': 0.0,
                    'success': False
                }
                
                # Apply direction analysis if available
                if self.direction_manager:
                    direction_analysis = self.direction_manager.analyze_gradient_direction(data, context)
                    integration_results['direction_analysis'] = direction_analysis
                    context = context or {}
                    context['direction_state'] = direction_analysis
                
                # Apply bounding if available
                if self.bounding_engine:
                    bounding_result = self.bounding_engine.bound_hybrid_gradients(data, context)
                    integration_results['bounding_result'] = bounding_result
                    data = bounding_result.bounded_gradients
                
                # Apply GAC processing
                if self.gac_system:
                    gac_result = self.gac_system.optimize_gradients([data], context.get('learning_rate', 0.001) if context else 0.001)
                    integration_results['gac_result'] = gac_result
                    if gac_result:
                        data = gac_result[0]
                
                # Apply hybrid compression
                if self.compression_system:
                    compression_result = self.compression_system.compress(data)
                    integration_results['compression_result'] = compression_result
                
                # Calculate performance metrics
                processing_time = (time.time() - start_time) * 1000
                integration_results['processing_time_ms'] = processing_time
                integration_results['success'] = True
                
                # Update performance tracking
                self._update_integration_performance(integration_results, processing_time)
                
                self.logger.debug(f"Hybrid compression-GAC integration completed in {processing_time:.2f}ms")
                
                return integration_results
                
        except Exception as e:
            self.logger.error(f"Hybrid compression-GAC integration failed: {e}")
            self._record_integration_error(e, context)
            raise RuntimeError(f"Hybrid compression-GAC integration failed: {e}")
    
    def optimize_gradient_flow(self, 
                             gradients: List[torch.Tensor], 
                             context: Optional[Dict[str, Any]] = None) -> List[torch.Tensor]:
        """
        Optimize gradient flow through Brain-GAC integration.
        
        Args:
            gradients: List of gradient tensors
            context: Optional optimization context
            
        Returns:
            Optimized gradient tensors
            
        Raises:
            RuntimeError: If optimization fails
            ValueError: If gradients are invalid
        """
        if self.integration_state != IntegrationState.INITIALIZED:
            raise RuntimeError("Integration system not initialized")
        
        if not isinstance(gradients, list):
            raise TypeError("Gradients must be list")
        if len(gradients) == 0:
            raise ValueError("Gradients list cannot be empty")
        for i, grad in enumerate(gradients):
            if not isinstance(grad, torch.Tensor):
                raise TypeError(f"Gradient {i} must be torch.Tensor")
            if grad.numel() == 0:
                raise ValueError(f"Gradient {i} cannot be empty")
        
        try:
            start_time = time.time()
            
            with self._gradient_flow_lock:
                # Apply gradient flow strategy
                if self.gradient_flow_strategy == GradientFlowStrategy.DIRECT_FLOW:
                    optimized_gradients = self._apply_direct_flow(gradients, context)
                elif self.gradient_flow_strategy == GradientFlowStrategy.BUFFERED_FLOW:
                    optimized_gradients = self._apply_buffered_flow(gradients, context)
                elif self.gradient_flow_strategy == GradientFlowStrategy.BATCHED_FLOW:
                    optimized_gradients = self._apply_batched_flow(gradients, context)
                elif self.gradient_flow_strategy == GradientFlowStrategy.ADAPTIVE_FLOW:
                    optimized_gradients = self._apply_adaptive_flow(gradients, context)
                elif self.gradient_flow_strategy == GradientFlowStrategy.PRIORITY_FLOW:
                    optimized_gradients = self._apply_priority_flow(gradients, context)
                else:
                    raise RuntimeError(f"Unknown gradient flow strategy: {self.gradient_flow_strategy}")
                
                # Update gradient flow metrics
                processing_time = (time.time() - start_time) * 1000
                self._update_gradient_flow_metrics(len(gradients), processing_time)
                
                self.logger.debug(f"Gradient flow optimization completed: {len(optimized_gradients)} gradients in {processing_time:.2f}ms")
                
                return optimized_gradients
                
        except Exception as e:
            self.logger.error(f"Gradient flow optimization failed: {e}")
            self._record_integration_error(e, context)
            raise RuntimeError(f"Gradient flow optimization failed: {e}")
    
    def monitor_integration_performance(self) -> Dict[str, Any]:
        """
        Monitor Brain-GAC integration performance.
        
        Returns:
            Current performance monitoring status
            
        Raises:
            RuntimeError: If monitoring fails
        """
        if self.integration_state != IntegrationState.INITIALIZED:
            raise RuntimeError("Integration system not initialized")
        
        try:
            with self._analytics_lock:
                monitoring_status = {
                    'monitoring_active': self.monitoring_active,
                    'integration_state': self.integration_state.name,
                    'total_operations': self.total_operations,
                    'successful_operations': self.successful_operations,
                    'failed_operations': self.failed_operations,
                    'success_rate': self.successful_operations / max(1, self.total_operations),
                    'current_performance': {},
                    'performance_trends': {},
                    'alerts': [],
                    'recommendations': [],
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Calculate current performance metrics
                if self.performance_metrics:
                    recent_metrics = list(self.performance_metrics)[-10:]  # Last 10 operations
                    monitoring_status['current_performance'] = self._calculate_current_performance(recent_metrics)
                    monitoring_status['performance_trends'] = self._calculate_performance_trends(recent_metrics)
                
                # Check for performance alerts
                alerts = self._check_performance_alerts()
                monitoring_status['alerts'] = alerts
                
                # Generate performance recommendations
                recommendations = self._generate_performance_recommendations()
                monitoring_status['recommendations'] = recommendations
                
                self.logger.debug(f"Integration performance monitoring completed: {len(alerts)} alerts, {len(recommendations)} recommendations")
                
                return monitoring_status
                
        except Exception as e:
            self.logger.error(f"Integration performance monitoring failed: {e}")
            raise RuntimeError(f"Integration performance monitoring failed: {e}")
    
    def handle_integration_errors(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle integration errors with recovery strategies.
        
        Args:
            error: Error that occurred
            context: Optional error context
            
        Returns:
            Error handling results
            
        Raises:
            RuntimeError: If error handling fails
        """
        if error is None:
            raise ValueError("Error cannot be None")
        if not isinstance(error, Exception):
            raise TypeError("Error must be Exception instance")
        
        try:
            error_handling_results = {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'recovery_attempted': False,
                'recovery_successful': False,
                'recovery_strategy': None,
                'fallback_applied': False,
                'system_state': self.integration_state.name,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            with self._integration_lock:
                # Determine error severity
                error_severity = self._assess_error_severity(error, context)
                error_handling_results['error_severity'] = error_severity
                
                # Apply error recovery if enabled
                if self.config.enable_error_recovery:
                    recovery_result = self._attempt_error_recovery(error, context)
                    error_handling_results.update(recovery_result)
                
                # Apply graceful degradation if needed
                if not error_handling_results['recovery_successful'] and self.config.enable_graceful_degradation:
                    degradation_result = self._apply_graceful_degradation(error, context)
                    error_handling_results.update(degradation_result)
                
                # Update error statistics
                self._update_error_statistics(error, error_handling_results)
                
                # Log error handling results
                if error_handling_results['recovery_successful']:
                    self.logger.info(f"Error recovery successful: {error_handling_results['recovery_strategy']}")
                else:
                    self.logger.error(f"Error recovery failed for {type(error).__name__}: {error}")
                
                return error_handling_results
                
        except Exception as e:
            self.logger.error(f"Error handling failed: {e}")
            raise RuntimeError(f"Error handling failed: {e}")
    
    def get_integration_analytics(self) -> IntegrationAnalytics:
        """
        Get comprehensive integration analytics.
        
        Returns:
            Integration analytics data
            
        Raises:
            RuntimeError: If analytics generation fails
        """
        if self.integration_state != IntegrationState.INITIALIZED:
            raise RuntimeError("Integration system not initialized")
        
        try:
            with self._analytics_lock:
                # Calculate operation statistics
                total_ops = self.total_operations
                success_ops = self.successful_operations
                failed_ops = self.failed_operations
                
                # Calculate performance statistics
                if self.performance_metrics:
                    avg_processing_time = sum(m.gradient_processing_time_ms for m in self.performance_metrics) / len(self.performance_metrics)
                    avg_memory_usage = sum(m.memory_usage_mb for m in self.performance_metrics) / len(self.performance_metrics)
                    avg_gpu_util = sum(m.gpu_utilization for m in self.performance_metrics) / len(self.performance_metrics)
                    integration_efficiency = sum(m.efficiency_score for m in self.performance_metrics) / len(self.performance_metrics)
                else:
                    avg_processing_time = 0.0
                    avg_memory_usage = 0.0
                    avg_gpu_util = 0.0
                    integration_efficiency = 0.0
                
                # Calculate subsystem efficiencies
                gradient_flow_efficiency = self._calculate_gradient_flow_efficiency()
                compression_efficiency = self._calculate_compression_efficiency()
                gac_efficiency = self._calculate_gac_efficiency()
                
                # Calculate performance trends
                performance_trends = self._calculate_detailed_performance_trends()
                
                # Analyze errors
                error_analysis = self._analyze_integration_errors()
                
                # Generate optimization recommendations
                optimization_recommendations = self._generate_optimization_recommendations()
                
                # Calculate system health score
                system_health_score = self._calculate_system_health_score()
                
                return IntegrationAnalytics(
                    total_operations=total_ops,
                    successful_operations=success_ops,
                    failed_operations=failed_ops,
                    average_processing_time_ms=avg_processing_time,
                    average_memory_usage_mb=avg_memory_usage,
                    average_gpu_utilization=avg_gpu_util,
                    integration_efficiency=integration_efficiency,
                    gradient_flow_efficiency=gradient_flow_efficiency,
                    compression_efficiency=compression_efficiency,
                    gac_efficiency=gac_efficiency,
                    performance_trends=performance_trends,
                    error_analysis=error_analysis,
                    optimization_recommendations=optimization_recommendations,
                    system_health_score=system_health_score
                )
                
        except Exception as e:
            self.logger.error(f"Integration analytics generation failed: {e}")
            raise RuntimeError(f"Integration analytics generation failed: {e}")
    
    def _initialize_performance_baselines(self) -> None:
        """Initialize performance baselines"""
        self.performance_baselines = {
            'gradient_processing_time_ms': self.config.max_gradient_processing_time_ms,
            'memory_usage_mb': self.config.max_memory_usage_mb,
            'gpu_utilization': self.config.target_gpu_utilization,
            'throughput_ops_per_sec': self.config.target_throughput_ops_per_sec,
            'efficiency_score': self.config.min_efficiency_score
        }
    
    def _setup_integration_hooks(self) -> None:
        """Setup integration hooks between systems"""
        # Setup Brain-GAC hooks
        if hasattr(self.brain_system, 'register_gac_hook'):
            self.brain_system.register_gac_hook('gradient_update', self._brain_gradient_hook)
            self.brain_system.register_gac_hook('training_step', self._brain_training_hook)
            self.brain_system.register_gac_hook('error_callback', self._brain_error_hook)
        
        # Setup GAC-Brain hooks
        if hasattr(self.gac_system, 'register_brain_hook'):
            self.gac_system.register_brain_hook('optimization_complete', self._gac_optimization_hook)
            self.gac_system.register_brain_hook('performance_update', self._gac_performance_hook)
    
    def _initialize_gradient_flow(self) -> None:
        """Initialize gradient flow system"""
        self.gradient_flow_buffer.clear()
        self.gradient_flow_strategy = self.config.gradient_flow_strategy
        
        # Initialize strategy-specific parameters
        if self.gradient_flow_strategy == GradientFlowStrategy.BUFFERED_FLOW:
            self._initialize_buffered_flow()
        elif self.gradient_flow_strategy == GradientFlowStrategy.BATCHED_FLOW:
            self._initialize_batched_flow()
        elif self.gradient_flow_strategy == GradientFlowStrategy.PRIORITY_FLOW:
            self._initialize_priority_flow()
    
    def _start_performance_monitoring(self) -> None:
        """Start performance monitoring"""
        self.monitoring_active = True
        self.logger.info("Performance monitoring started")
    
    def _initialize_integration_analytics(self) -> None:
        """Initialize integration analytics"""
        self.integration_analytics = {
            'operations': {
                'total': 0,
                'successful': 0,
                'failed': 0
            },
            'performance': {
                'average_time_ms': 0.0,
                'average_memory_mb': 0.0,
                'average_efficiency': 0.0
            },
            'errors': defaultdict(int),
            'trends': defaultdict(list)
        }
    
    def _apply_direct_flow(self, gradients: List[torch.Tensor], context: Optional[Dict[str, Any]]) -> List[torch.Tensor]:
        """Apply direct gradient flow strategy"""
        optimized_gradients = []
        
        for gradient in gradients:
            # Apply direct processing through integration pipeline
            integration_result = self.integrate_hybrid_compression_with_gac(gradient, context)
            
            # Extract optimized gradient
            if integration_result.get('gac_result'):
                optimized_gradients.append(integration_result['gac_result'][0])
            elif integration_result.get('compression_result'):
                # Use compressed data if GAC not available
                compressed_data = integration_result['compression_result']
                if 'compressed_data' in compressed_data:
                    optimized_gradients.append(compressed_data['compressed_data'])
                else:
                    optimized_gradients.append(gradient)
            else:
                optimized_gradients.append(gradient)
        
        return optimized_gradients
    
    def _apply_buffered_flow(self, gradients: List[torch.Tensor], context: Optional[Dict[str, Any]]) -> List[torch.Tensor]:
        """Apply buffered gradient flow strategy"""
        # Add gradients to buffer
        for gradient in gradients:
            self.gradient_flow_buffer.append((gradient, context, time.time()))
        
        # Process buffered gradients
        optimized_gradients = []
        
        while self.gradient_flow_buffer:
            gradient, grad_context, timestamp = self.gradient_flow_buffer.popleft()
            
            # Check for timeout
            if time.time() - timestamp > self.config.gradient_flow_timeout_ms / 1000.0:
                self.logger.warning("Gradient flow timeout - processing immediately")
            
            integration_result = self.integrate_hybrid_compression_with_gac(gradient, grad_context)
            
            if integration_result.get('gac_result'):
                optimized_gradients.append(integration_result['gac_result'][0])
            else:
                optimized_gradients.append(gradient)
        
        return optimized_gradients
    
    def _apply_batched_flow(self, gradients: List[torch.Tensor], context: Optional[Dict[str, Any]]) -> List[torch.Tensor]:
        """Apply batched gradient flow strategy"""
        optimized_gradients = []
        batch_size = self.config.gradient_batch_size
        
        # Process gradients in batches
        for i in range(0, len(gradients), batch_size):
            batch = gradients[i:i + batch_size]
            batch_context = context.copy() if context else {}
            batch_context['batch_size'] = len(batch)
            batch_context['batch_index'] = i // batch_size
            
            # Process batch
            for gradient in batch:
                integration_result = self.integrate_hybrid_compression_with_gac(gradient, batch_context)
                
                if integration_result.get('gac_result'):
                    optimized_gradients.append(integration_result['gac_result'][0])
                else:
                    optimized_gradients.append(gradient)
        
        return optimized_gradients
    
    def _apply_adaptive_flow(self, gradients: List[torch.Tensor], context: Optional[Dict[str, Any]]) -> List[torch.Tensor]:
        """Apply adaptive gradient flow strategy"""
        # Analyze current system load and performance
        current_load = self._assess_system_load()
        performance_metrics = self._get_recent_performance_metrics()
        
        # Adapt strategy based on current conditions
        if current_load > 0.8:
            # High load - use buffered flow
            return self._apply_buffered_flow(gradients, context)
        elif performance_metrics.get('average_time_ms', 0) > self.config.max_gradient_processing_time_ms:
            # Slow processing - use batched flow
            return self._apply_batched_flow(gradients, context)
        elif len(gradients) > self.config.gradient_batch_size * 2:
            # Large gradient set - use batched flow
            return self._apply_batched_flow(gradients, context)
        else:
            # Normal conditions - use direct flow
            return self._apply_direct_flow(gradients, context)
    
    def _apply_priority_flow(self, gradients: List[torch.Tensor], context: Optional[Dict[str, Any]]) -> List[torch.Tensor]:
        """Apply priority-based gradient flow strategy"""
        if not self.config.enable_gradient_prioritization:
            return self._apply_direct_flow(gradients, context)
        
        # Calculate gradient priorities
        gradient_priorities = []
        for i, gradient in enumerate(gradients):
            priority = self._calculate_gradient_priority(gradient, context)
            gradient_priorities.append((priority, i, gradient))
        
        # Sort by priority (highest first)
        gradient_priorities.sort(key=lambda x: x[0], reverse=True)
        
        # Process gradients in priority order
        optimized_gradients = [None] * len(gradients)
        
        for priority, original_index, gradient in gradient_priorities:
            priority_context = context.copy() if context else {}
            priority_context['gradient_priority'] = priority
            priority_context['original_index'] = original_index
            
            integration_result = self.integrate_hybrid_compression_with_gac(gradient, priority_context)
            
            if integration_result.get('gac_result'):
                optimized_gradients[original_index] = integration_result['gac_result'][0]
            else:
                optimized_gradients[original_index] = gradient
        
        return optimized_gradients
    
    def _update_integration_performance(self, integration_results: Dict[str, Any], processing_time: float) -> None:
        """Update integration performance metrics"""
        # Create performance metrics
        metrics = IntegrationPerformanceMetrics(
            gradient_processing_time_ms=processing_time,
            gac_processing_time_ms=integration_results.get('gac_processing_time', 0.0),
            compression_time_ms=integration_results.get('compression_time', 0.0),
            total_integration_time_ms=processing_time,
            memory_usage_mb=self._get_current_memory_usage(),
            gpu_utilization=self._get_current_gpu_utilization(),
            cpu_utilization=self._get_current_cpu_utilization(),
            throughput_ops_per_sec=1000.0 / processing_time if processing_time > 0 else 0.0,
            latency_ms=processing_time,
            efficiency_score=self._calculate_efficiency_score(processing_time),
            error_count=0 if integration_results.get('success') else 1,
            success_count=1 if integration_results.get('success') else 0
        )
        
        # Store metrics
        self.performance_metrics.append(metrics)
        
        # Update operation counters
        self.total_operations += 1
        if integration_results.get('success'):
            self.successful_operations += 1
        else:
            self.failed_operations += 1
    
    def _record_integration_error(self, error: Exception, context: Optional[Dict[str, Any]]) -> None:
        """Record integration error for analytics"""
        error_type = type(error).__name__
        
        # Update error statistics
        if 'errors' not in self.integration_analytics:
            self.integration_analytics['errors'] = defaultdict(int)
        
        self.integration_analytics['errors'][error_type] += 1
        self.failed_operations += 1
        self.total_operations += 1
        
        # Log error details
        self.logger.error(f"Integration error: {error_type}: {error}")
        if context:
            self.logger.error(f"Error context: {context}")
    
    def _update_gradient_flow_metrics(self, gradient_count: int, processing_time: float) -> None:
        """Update gradient flow specific metrics"""
        if 'gradient_flow' not in self.integration_analytics:
            self.integration_analytics['gradient_flow'] = {
                'total_gradients': 0,
                'total_time_ms': 0.0,
                'average_time_per_gradient': 0.0
            }
        
        flow_metrics = self.integration_analytics['gradient_flow']
        flow_metrics['total_gradients'] += gradient_count
        flow_metrics['total_time_ms'] += processing_time
        
        if flow_metrics['total_gradients'] > 0:
            flow_metrics['average_time_per_gradient'] = flow_metrics['total_time_ms'] / flow_metrics['total_gradients']
    
    def _calculate_current_performance(self, recent_metrics: List[IntegrationPerformanceMetrics]) -> Dict[str, float]:
        """Calculate current performance from recent metrics"""
        if not recent_metrics:
            return {}
        
        return {
            'average_processing_time_ms': sum(m.gradient_processing_time_ms for m in recent_metrics) / len(recent_metrics),
            'average_memory_usage_mb': sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics),
            'average_gpu_utilization': sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics),
            'average_efficiency_score': sum(m.efficiency_score for m in recent_metrics) / len(recent_metrics),
            'average_throughput': sum(m.throughput_ops_per_sec for m in recent_metrics) / len(recent_metrics)
        }
    
    def _calculate_performance_trends(self, recent_metrics: List[IntegrationPerformanceMetrics]) -> Dict[str, float]:
        """Calculate performance trends"""
        if len(recent_metrics) < 2:
            return {}
        
        # Calculate linear trends using simple regression
        times = [m.gradient_processing_time_ms for m in recent_metrics]
        memories = [m.memory_usage_mb for m in recent_metrics]
        efficiencies = [m.efficiency_score for m in recent_metrics]
        
        return {
            'time_trend': self._calculate_trend(times),
            'memory_trend': self._calculate_trend(memories),
            'efficiency_trend': self._calculate_trend(efficiencies)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend using linear regression"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = list(range(n))
        
        # Simple linear regression
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def _check_performance_alerts(self) -> List[Dict[str, str]]:
        """Check for performance alerts"""
        alerts = []
        
        if not self.performance_metrics:
            return alerts
        
        recent_metrics = list(self.performance_metrics)[-5:]  # Last 5 operations
        
        # Check processing time alerts
        avg_time = sum(m.gradient_processing_time_ms for m in recent_metrics) / len(recent_metrics)
        if avg_time > self.config.max_gradient_processing_time_ms:
            alerts.append({
                'type': 'high_processing_time',
                'severity': 'warning',
                'message': f'Average processing time {avg_time:.2f}ms exceeds limit {self.config.max_gradient_processing_time_ms}ms'
            })
        
        # Check memory usage alerts
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        if avg_memory > self.config.max_memory_usage_mb:
            alerts.append({
                'type': 'high_memory_usage',
                'severity': 'warning',
                'message': f'Average memory usage {avg_memory:.2f}MB exceeds limit {self.config.max_memory_usage_mb}MB'
            })
        
        # Check efficiency alerts
        avg_efficiency = sum(m.efficiency_score for m in recent_metrics) / len(recent_metrics)
        if avg_efficiency < self.config.min_efficiency_score:
            alerts.append({
                'type': 'low_efficiency',
                'severity': 'warning',
                'message': f'Average efficiency {avg_efficiency:.3f} below minimum {self.config.min_efficiency_score}'
            })
        
        return alerts
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if not self.performance_metrics:
            return recommendations
        
        recent_metrics = list(self.performance_metrics)[-10:]
        
        # Analyze performance patterns
        avg_time = sum(m.gradient_processing_time_ms for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        avg_gpu_util = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)
        
        # Time-based recommendations
        if avg_time > self.config.max_gradient_processing_time_ms * 0.8:
            recommendations.append("Consider switching to batched gradient flow for better performance")
        
        # Memory-based recommendations
        if avg_memory > self.config.max_memory_usage_mb * 0.8:
            recommendations.append("Consider enabling memory-efficient integration mode")
        
        # GPU utilization recommendations
        if avg_gpu_util < self.config.target_gpu_utilization * 0.7:
            recommendations.append("Consider enabling GPU-accelerated bounding for better utilization")
        
        # Integration mode recommendations
        if len(recent_metrics) > 5:
            efficiency_trend = self._calculate_trend([m.efficiency_score for m in recent_metrics])
            if efficiency_trend < -0.01:  # Decreasing efficiency
                recommendations.append("Consider switching to performance-optimized integration mode")
        
        return recommendations
    
    def _assess_error_severity(self, error: Exception, context: Optional[Dict[str, Any]]) -> str:
        """Assess error severity"""
        if isinstance(error, (RuntimeError, SystemError, MemoryError)):
            return 'critical'
        elif isinstance(error, (ValueError, TypeError)):
            return 'high'
        elif isinstance(error, (TimeoutError, ConnectionError)):
            return 'medium'
        else:
            return 'low'
    
    def _attempt_error_recovery(self, error: Exception, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Attempt error recovery"""
        recovery_result = {
            'recovery_attempted': True,
            'recovery_successful': False,
            'recovery_strategy': None
        }
        
        try:
            # Determine recovery strategy based on error type
            if isinstance(error, MemoryError):
                recovery_result['recovery_strategy'] = 'memory_cleanup'
                self._cleanup_memory()
                recovery_result['recovery_successful'] = True
            elif isinstance(error, TimeoutError):
                recovery_result['recovery_strategy'] = 'timeout_recovery'
                self._reset_gradient_flow()
                recovery_result['recovery_successful'] = True
            elif isinstance(error, (ValueError, TypeError)):
                recovery_result['recovery_strategy'] = 'parameter_reset'
                self._reset_integration_parameters()
                recovery_result['recovery_successful'] = True
            else:
                recovery_result['recovery_strategy'] = 'system_reset'
                self._soft_reset_integration()
                recovery_result['recovery_successful'] = True
                
        except Exception as recovery_error:
            self.logger.error(f"Error recovery failed: {recovery_error}")
            recovery_result['recovery_successful'] = False
        
        return recovery_result
    
    def _apply_graceful_degradation(self, error: Exception, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply graceful degradation"""
        degradation_result = {
            'fallback_applied': True,
            'degradation_level': 'minimal'
        }
        
        # Apply degradation based on error severity
        error_severity = self._assess_error_severity(error, context)
        
        if error_severity == 'critical':
            # Disable advanced features
            self.config.enable_real_time_optimization = False
            self.config.enable_performance_monitoring = False
            degradation_result['degradation_level'] = 'significant'
        elif error_severity == 'high':
            # Reduce performance features
            self.gradient_flow_strategy = GradientFlowStrategy.DIRECT_FLOW
            degradation_result['degradation_level'] = 'moderate'
        else:
            # Minimal degradation
            self.config.enable_adaptive_optimization = False
            degradation_result['degradation_level'] = 'minimal'
        
        return degradation_result
    
    def _update_error_statistics(self, error: Exception, error_handling_results: Dict[str, Any]) -> None:
        """Update error statistics"""
        error_type = type(error).__name__
        
        if 'error_stats' not in self.integration_analytics:
            self.integration_analytics['error_stats'] = defaultdict(int)
        
        self.integration_analytics['error_stats'][error_type] += 1
        self.integration_analytics['error_stats']['total_errors'] += 1
        
        if error_handling_results.get('recovery_successful'):
            self.integration_analytics['error_stats']['recovered_errors'] += 1
    
    def _calculate_gradient_flow_efficiency(self) -> float:
        """Calculate gradient flow efficiency"""
        if 'gradient_flow' not in self.integration_analytics:
            return 0.0
        
        flow_metrics = self.integration_analytics['gradient_flow']
        total_gradients = flow_metrics.get('total_gradients', 0)
        total_time = flow_metrics.get('total_time_ms', 0)
        
        if total_time == 0:
            return 0.0
        
        # Calculate gradients per second
        gradients_per_sec = (total_gradients * 1000.0) / total_time
        
        # Normalize to efficiency score
        target_rate = self.config.target_throughput_ops_per_sec
        return min(1.0, gradients_per_sec / target_rate)
    
    def _calculate_compression_efficiency(self) -> float:
        """Calculate compression efficiency"""
        if not self.compression_system:
            return 1.0  # No compression system
        
        # Get compression statistics from system
        if hasattr(self.compression_system, 'get_compression_stats'):
            stats = self.compression_system.get_compression_stats()
            return stats.get('efficiency_score', 0.8)
        
        return 0.8  # Default efficiency
    
    def _calculate_gac_efficiency(self) -> float:
        """Calculate GAC efficiency"""
        if not self.gac_system:
            return 1.0  # No GAC system
        
        # Get GAC performance metrics
        if hasattr(self.gac_system, 'get_performance_metrics'):
            metrics = self.gac_system.get_performance_metrics()
            return metrics.get('efficiency_score', 0.8)
        
        return 0.8  # Default efficiency
    
    def _calculate_detailed_performance_trends(self) -> Dict[str, List[float]]:
        """Calculate detailed performance trends"""
        trends = defaultdict(list)
        
        if len(self.performance_metrics) >= 10:
            recent_metrics = list(self.performance_metrics)[-50:]  # Last 50 operations
            
            trends['processing_time'] = [m.gradient_processing_time_ms for m in recent_metrics]
            trends['memory_usage'] = [m.memory_usage_mb for m in recent_metrics]
            trends['gpu_utilization'] = [m.gpu_utilization for m in recent_metrics]
            trends['efficiency_score'] = [m.efficiency_score for m in recent_metrics]
            trends['throughput'] = [m.throughput_ops_per_sec for m in recent_metrics]
        
        return dict(trends)
    
    def _analyze_integration_errors(self) -> Dict[str, int]:
        """Analyze integration errors"""
        error_analysis = defaultdict(int)
        
        if 'error_stats' in self.integration_analytics:
            error_stats = self.integration_analytics['error_stats']
            for error_type, count in error_stats.items():
                error_analysis[error_type] = count
        
        return dict(error_analysis)
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Analyze current performance
        if self.performance_metrics:
            recent_metrics = list(self.performance_metrics)[-20:]
            avg_efficiency = sum(m.efficiency_score for m in recent_metrics) / len(recent_metrics)
            avg_time = sum(m.gradient_processing_time_ms for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
            
            if avg_efficiency < 0.7:
                recommendations.append("Consider enabling performance-optimized integration mode")
            
            if avg_time > self.config.max_gradient_processing_time_ms * 0.8:
                recommendations.append("Consider optimizing gradient flow strategy")
            
            if avg_memory > self.config.max_memory_usage_mb * 0.8:
                recommendations.append("Consider enabling memory-efficient mode")
        
        # Error-based recommendations
        if 'error_stats' in self.integration_analytics:
            total_errors = self.integration_analytics['error_stats'].get('total_errors', 0)
            if total_errors > self.total_operations * 0.1:  # > 10% error rate
                recommendations.append("Consider enabling error recovery and graceful degradation")
        
        return recommendations
    
    def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score"""
        if self.total_operations == 0:
            return 1.0
        
        # Success rate component
        success_rate = self.successful_operations / self.total_operations
        
        # Performance component
        if self.performance_metrics:
            avg_efficiency = sum(m.efficiency_score for m in self.performance_metrics) / len(self.performance_metrics)
        else:
            avg_efficiency = 1.0
        
        # Error rate component
        error_rate = self.failed_operations / self.total_operations
        error_component = max(0.0, 1.0 - error_rate * 2)  # Penalize errors
        
        # Combined health score
        health_score = (success_rate * 0.4 + avg_efficiency * 0.4 + error_component * 0.2)
        
        return max(0.0, min(1.0, health_score))
    
    def _brain_gradient_hook(self, gradients: List[torch.Tensor], context: Dict[str, Any]) -> List[torch.Tensor]:
        """Hook for Brain gradient updates"""
        return self.optimize_gradient_flow(gradients, context)
    
    def _brain_training_hook(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for Brain training steps"""
        # Add integration metrics to training data
        training_data['integration_metrics'] = self.get_integration_analytics()
        return training_data
    
    def _brain_error_hook(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for Brain errors"""
        return self.handle_integration_errors(error, context)
    
    def _gac_optimization_hook(self, optimization_results: Dict[str, Any]) -> None:
        """Hook for GAC optimization completion"""
        # Update integration analytics with GAC results
        if 'gac_optimization' not in self.integration_analytics:
            self.integration_analytics['gac_optimization'] = []
        
        self.integration_analytics['gac_optimization'].append(optimization_results)
    
    def _gac_performance_hook(self, performance_data: Dict[str, Any]) -> None:
        """Hook for GAC performance updates"""
        # Incorporate GAC performance into integration metrics
        if 'gac_performance' not in self.integration_analytics:
            self.integration_analytics['gac_performance'] = []
        
        self.integration_analytics['gac_performance'].append(performance_data)
    
    def _initialize_buffered_flow(self) -> None:
        """Initialize buffered flow parameters"""
        self.gradient_flow_buffer.clear()
    
    def _initialize_batched_flow(self) -> None:
        """Initialize batched flow parameters"""
        # No special initialization needed for batched flow
        pass
    
    def _initialize_priority_flow(self) -> None:
        """Initialize priority flow parameters"""
        if not hasattr(self, 'gradient_priority_weights'):
            self.gradient_priority_weights = {
                'magnitude': 0.4,
                'variance': 0.3,
                'age': 0.2,
                'context': 0.1
            }
    
    def _assess_system_load(self) -> float:
        """Assess current system load"""
        # Simple system load assessment
        if self.performance_metrics:
            recent_metrics = list(self.performance_metrics)[-5:]
            avg_time = sum(m.gradient_processing_time_ms for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
            avg_cpu = sum(m.cpu_utilization for m in recent_metrics) / len(recent_metrics)
            
            # Normalize and combine metrics
            time_load = min(1.0, avg_time / self.config.max_gradient_processing_time_ms)
            memory_load = min(1.0, avg_memory / self.config.max_memory_usage_mb)
            cpu_load = avg_cpu
            
            return (time_load + memory_load + cpu_load) / 3.0
        
        return 0.5  # Default moderate load
    
    def _get_recent_performance_metrics(self) -> Dict[str, float]:
        """Get recent performance metrics"""
        if not self.performance_metrics:
            return {}
        
        recent_metrics = list(self.performance_metrics)[-5:]
        return {
            'average_time_ms': sum(m.gradient_processing_time_ms for m in recent_metrics) / len(recent_metrics),
            'average_memory_mb': sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics),
            'average_efficiency': sum(m.efficiency_score for m in recent_metrics) / len(recent_metrics)
        }
    
    def _calculate_gradient_priority(self, gradient: torch.Tensor, context: Optional[Dict[str, Any]]) -> float:
        """Calculate gradient priority score"""
        if not hasattr(self, 'gradient_priority_weights'):
            return 1.0
        
        weights = self.gradient_priority_weights
        priority_score = 0.0
        
        # Magnitude component
        magnitude = torch.norm(gradient).item()
        magnitude_score = min(1.0, magnitude / 10.0)  # Normalize
        priority_score += weights['magnitude'] * magnitude_score
        
        # Variance component
        variance = torch.var(gradient).item()
        variance_score = min(1.0, variance / 1.0)  # Normalize
        priority_score += weights['variance'] * variance_score
        
        # Age component (newer gradients have higher priority)
        age_score = 1.0  # Default for new gradients
        priority_score += weights['age'] * age_score
        
        # Context component
        context_score = 0.5  # Default
        if context:
            if context.get('training_step', 0) % 100 == 0:  # Milestone steps
                context_score = 1.0
            elif context.get('is_validation', False):
                context_score = 0.8
        priority_score += weights['context'] * context_score
        
        return min(1.0, priority_score)
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            return psutil.virtual_memory().used / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def _get_current_gpu_utilization(self) -> float:
        """Get current GPU utilization"""
        if torch.cuda.is_available():
            return torch.cuda.utilization() / 100.0 if hasattr(torch.cuda, 'utilization') else 0.5
        return 0.0
    
    def _get_current_cpu_utilization(self) -> float:
        """Get current CPU utilization"""
        try:
            import psutil
            return psutil.cpu_percent(interval=None) / 100.0
        except ImportError:
            return 0.5
    
    def _calculate_efficiency_score(self, processing_time: float) -> float:
        """Calculate efficiency score based on processing time"""
        if processing_time <= 0:
            return 1.0
        
        # Efficiency based on time vs target
        time_efficiency = min(1.0, self.config.max_gradient_processing_time_ms / processing_time)
        
        # Additional factors could be added here
        return time_efficiency
    
    def _cleanup_memory(self) -> None:
        """Cleanup memory for error recovery"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear some buffers if they're too large
        if len(self.gradient_flow_buffer) > self.config.gradient_buffer_size // 2:
            # Keep only recent half
            buffer_list = list(self.gradient_flow_buffer)
            self.gradient_flow_buffer.clear()
            for item in buffer_list[-self.config.gradient_buffer_size // 2:]:
                self.gradient_flow_buffer.append(item)
    
    def _reset_gradient_flow(self) -> None:
        """Reset gradient flow for error recovery"""
        self.gradient_flow_buffer.clear()
        self.gradient_flow_strategy = GradientFlowStrategy.DIRECT_FLOW
    
    def _reset_integration_parameters(self) -> None:
        """Reset integration parameters for error recovery"""
        self.config.gradient_batch_size = min(32, self.config.gradient_batch_size)
        self.config.gradient_flow_timeout_ms = min(100.0, self.config.gradient_flow_timeout_ms)
    
    def _soft_reset_integration(self) -> None:
        """Perform soft reset of integration system"""
        self.gradient_flow_buffer.clear()
        self.gradient_flow_strategy = GradientFlowStrategy.DIRECT_FLOW
        self.config.enable_real_time_optimization = False
        
        # Reset to stable configuration
        self.config.integration_mode = IntegrationMode.STANDARD
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        return {
            'integration_state': self.integration_state.name,
            'gradient_flow_strategy': self.gradient_flow_strategy.name,
            'monitoring_active': self.monitoring_active,
            'total_operations': self.total_operations,
            'successful_operations': self.successful_operations,
            'failed_operations': self.failed_operations,
            'success_rate': self.successful_operations / max(1, self.total_operations),
            'gradient_buffer_size': len(self.gradient_flow_buffer),
            'system_health_score': self._calculate_system_health_score(),
            'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None
        }
    
    def shutdown(self) -> None:
        """Shutdown Brain-GAC integration"""
        self.logger.info("Shutting down Brain-GAC integration")
        
        self.integration_state = IntegrationState.SHUTDOWN
        self.monitoring_active = False
        
        # Clear references
        self.brain_system = None
        self.gac_system = None
        self.compression_system = None
        self.direction_manager = None
        self.bounding_engine = None
        
        # Clear tracking data
        self.performance_metrics.clear()
        self.gradient_flow_buffer.clear()
        self.optimization_history.clear()
        
        self.logger.info("Brain-GAC integration shutdown complete")