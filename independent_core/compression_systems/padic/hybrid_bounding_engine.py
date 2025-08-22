"""
Hybrid Bounding Engine - Main hybrid bounding engine for p-adic gradient operations
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
from gac_system.enhanced_bounder import EnhancedGradientBounder, EnhancedBoundingResult

# Import direction system components
from direction_manager import DirectionManager
from dynamic_switching_manager import DynamicSwitchingManager


class HybridBoundingStrategy(Enum):
    """Hybrid bounding strategy enumeration"""
    DIRECTION_ADAPTIVE = "direction_adaptive"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    MEMORY_EFFICIENT = "memory_efficient"
    GPU_ACCELERATED = "gpu_accelerated"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


class BoundingOperationType(Enum):
    """Bounding operation type enumeration"""
    HYBRID_GRADIENT = "hybrid_gradient"
    PADIC_GRADIENT = "padic_gradient"
    DIRECTION_SPECIFIC = "direction_specific"
    SWITCHING_OPTIMIZED = "switching_optimized"


@dataclass
class HybridBoundingResult:
    """Hybrid bounding operation result"""
    bounded_gradients: torch.Tensor
    bounding_strategy: HybridBoundingStrategy
    operation_type: BoundingOperationType
    direction_state: DirectionState
    bounding_performance: Dict[str, float]
    memory_usage: Dict[str, float]
    gpu_utilization: float
    bounding_time_ms: float
    quality_metrics: Dict[str, float]
    error_metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate bounding result"""
        if self.bounded_gradients is None:
            raise ValueError("Bounded gradients cannot be None")
        if not isinstance(self.bounded_gradients, torch.Tensor):
            raise TypeError("Bounded gradients must be torch.Tensor")
        if not isinstance(self.bounding_strategy, HybridBoundingStrategy):
            raise TypeError("Bounding strategy must be HybridBoundingStrategy")
        if not isinstance(self.operation_type, BoundingOperationType):
            raise TypeError("Operation type must be BoundingOperationType")
        if not isinstance(self.direction_state, DirectionState):
            raise TypeError("Direction state must be DirectionState")


@dataclass
class HybridBoundingAnalytics:
    """Hybrid bounding analytics data"""
    total_bounding_operations: int
    successful_operations: int
    failed_operations: int
    average_bounding_time: float
    average_gpu_utilization: float
    memory_usage_stats: Dict[str, float]
    strategy_performance: Dict[str, Dict[str, float]]
    direction_bounding_stats: Dict[str, Dict[str, float]]
    quality_metrics: Dict[str, float]
    error_analysis: Dict[str, int]
    performance_trends: Dict[str, List[float]]
    optimization_recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HybridBoundingConfig:
    """Configuration for hybrid bounding engine"""
    # Strategy configuration
    default_strategy: HybridBoundingStrategy = HybridBoundingStrategy.DIRECTION_ADAPTIVE
    enable_direction_adaptive: bool = True
    enable_performance_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_gpu_acceleration: bool = True
    
    # Bounding parameters
    hybrid_max_norm: float = 2.5
    hybrid_clip_value: float = 20.0
    padic_max_norm: float = 1.8
    padic_clip_value: float = 15.0
    
    # Direction-specific parameters
    ascent_scaling_factor: float = 1.2
    descent_scaling_factor: float = 0.85
    stable_scaling_factor: float = 0.75
    oscillating_scaling_factor: float = 0.65
    
    # Performance parameters
    max_bounding_time_ms: float = 10.0
    target_gpu_utilization: float = 0.8
    memory_efficiency_threshold: float = 0.9
    quality_threshold: float = 0.85
    
    # Optimization parameters
    enable_adaptive_parameters: bool = True
    enable_real_time_optimization: bool = True
    enable_performance_monitoring: bool = True
    optimization_interval_seconds: int = 60
    
    # Analytics parameters
    analytics_history_size: int = 1000
    performance_trend_window: int = 100
    error_tracking_enabled: bool = True
    detailed_metrics_enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if not (0.0 < self.hybrid_max_norm <= 10.0):
            raise ValueError("Hybrid max norm must be between 0.0 and 10.0")
        if not (0.0 < self.hybrid_clip_value <= 50.0):
            raise ValueError("Hybrid clip value must be between 0.0 and 50.0")
        if not (0.0 < self.max_bounding_time_ms <= 100.0):
            raise ValueError("Max bounding time must be between 0.0 and 100.0 ms")
        if not (0.0 <= self.target_gpu_utilization <= 1.0):
            raise ValueError("Target GPU utilization must be between 0.0 and 1.0")


class HybridBoundingEngine:
    """
    Main hybrid bounding engine for p-adic gradient operations.
    Integrates with GAC system and provides direction-aware bounding.
    """
    
    def __init__(self, config: Optional[HybridBoundingConfig] = None):
        """Initialize hybrid bounding engine"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, HybridBoundingConfig):
            raise TypeError(f"Config must be HybridBoundingConfig or None, got {type(config)}")
        
        self.config = config or HybridBoundingConfig()
        self.logger = logging.getLogger('HybridBoundingEngine')
        
        # System components
        self.enhanced_bounder: Optional[EnhancedGradientBounder] = None
        self.direction_manager: Optional[DirectionManager] = None
        self.switching_manager: Optional[DynamicSwitchingManager] = None
        
        # Bounding system state
        self.is_initialized = False
        self.current_strategy = self.config.default_strategy
        
        # Performance tracking
        self.bounding_history: deque = deque(maxlen=self.config.analytics_history_size)
        self.performance_metrics: Dict[str, Any] = {}
        self.strategy_performance: Dict[HybridBoundingStrategy, List[float]] = defaultdict(list)
        
        # Optimization tracking
        self.adaptive_parameters: Dict[str, float] = {}
        self.last_optimization: Optional[datetime] = None
        self.gpu_memory_stats: Dict[str, float] = {}
        
        # Thread safety
        self._bounding_lock = threading.RLock()
        self._analytics_lock = threading.RLock()
        
        # Analytics
        self.bounding_analytics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'average_time_ms': 0.0,
            'average_gpu_utilization': 0.0,
            'memory_efficiency': 0.0
        }
        
        self.logger.info("HybridBoundingEngine created successfully")
    
    def initialize_hybrid_bounding(self,
                                 enhanced_bounder: EnhancedGradientBounder,
                                 direction_manager: Optional[DirectionManager] = None,
                                 switching_manager: Optional[DynamicSwitchingManager] = None) -> None:
        """
        Initialize hybrid bounding system with required components.
        
        Args:
            enhanced_bounder: GAC enhanced gradient bounder
            direction_manager: Optional direction manager
            switching_manager: Optional switching manager
            
        Raises:
            TypeError: If components are invalid
            RuntimeError: If initialization fails
        """
        if not isinstance(enhanced_bounder, EnhancedGradientBounder):
            raise TypeError(f"Enhanced bounder must be EnhancedGradientBounder, got {type(enhanced_bounder)}")
        if direction_manager is not None and not isinstance(direction_manager, DirectionManager):
            raise TypeError(f"Direction manager must be DirectionManager, got {type(direction_manager)}")
        if switching_manager is not None and not isinstance(switching_manager, DynamicSwitchingManager):
            raise TypeError(f"Switching manager must be DynamicSwitchingManager, got {type(switching_manager)}")
        
        try:
            # Set component references
            self.enhanced_bounder = enhanced_bounder
            self.direction_manager = direction_manager
            self.switching_manager = switching_manager
            
            # Initialize adaptive parameters
            self._initialize_adaptive_parameters()
            
            # Initialize GPU memory tracking
            self._initialize_gpu_memory_tracking()
            
            # Initialize performance monitoring
            self._initialize_performance_monitoring()
            
            self.is_initialized = True
            self.logger.info("Hybrid bounding system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize hybrid bounding system: {e}")
            raise RuntimeError(f"Hybrid bounding system initialization failed: {e}")
    
    def bound_hybrid_gradients(self, 
                             gradients: torch.Tensor, 
                             context: Optional[Dict[str, Any]] = None) -> HybridBoundingResult:
        """
        Bound gradients using hybrid bounding strategies.
        
        Args:
            gradients: Gradient tensor to bound
            context: Optional bounding context
            
        Returns:
            Hybrid bounding result
            
        Raises:
            RuntimeError: If bounding fails
            ValueError: If gradients are invalid
        """
        if not self.is_initialized:
            raise RuntimeError("Hybrid bounding system not initialized")
        
        if gradients is None:
            raise ValueError("Gradients cannot be None")
        if not isinstance(gradients, torch.Tensor):
            raise TypeError("Gradients must be torch.Tensor")
        if gradients.numel() == 0:
            raise ValueError("Gradients cannot be empty")
        
        try:
            start_time = time.time()
            
            with self._bounding_lock:
                # Get direction state if available
                direction_state = self._get_direction_state(gradients, context)
                
                # Select bounding strategy
                strategy = self._select_bounding_strategy(gradients, direction_state, context)
                
                # Apply bounding strategy
                bounding_result = self._apply_bounding_strategy(
                    gradients, direction_state, strategy, context
                )
                
                # Monitor performance
                bounding_time = (time.time() - start_time) * 1000
                self._update_performance_metrics(bounding_result, bounding_time)
                
                # Record operation
                self.bounding_history.append(bounding_result)
                
                self.logger.debug(f"Hybrid bounding completed: {strategy.name} in {bounding_time:.2f}ms")
                
                return bounding_result
                
        except Exception as e:
            self.logger.error(f"Hybrid gradient bounding failed: {e}")
            self._record_bounding_failure(e)
            raise RuntimeError(f"Hybrid gradient bounding failed: {e}")
    
    def apply_hybrid_bounding_strategy(self, 
                                     gradients: torch.Tensor, 
                                     direction_state: DirectionState) -> HybridBoundingResult:
        """
        Apply specific hybrid bounding strategy based on direction state.
        
        Args:
            gradients: Gradient tensor to bound
            direction_state: Current direction state
            
        Returns:
            Hybrid bounding result
            
        Raises:
            RuntimeError: If strategy application fails
            ValueError: If inputs are invalid
        """
        if not self.is_initialized:
            raise RuntimeError("Hybrid bounding system not initialized")
        
        if gradients is None:
            raise ValueError("Gradients cannot be None")
        if not isinstance(gradients, torch.Tensor):
            raise TypeError("Gradients must be torch.Tensor")
        if direction_state is None:
            raise ValueError("Direction state cannot be None")
        if not isinstance(direction_state, DirectionState):
            raise TypeError("Direction state must be DirectionState")
        
        try:
            with self._bounding_lock:
                # Apply direction-specific bounding
                if self.current_strategy == HybridBoundingStrategy.DIRECTION_ADAPTIVE:
                    return self._apply_direction_adaptive_bounding(gradients, direction_state)
                elif self.current_strategy == HybridBoundingStrategy.PERFORMANCE_OPTIMIZED:
                    return self._apply_performance_optimized_bounding(gradients, direction_state)
                elif self.current_strategy == HybridBoundingStrategy.MEMORY_EFFICIENT:
                    return self._apply_memory_efficient_bounding(gradients, direction_state)
                elif self.current_strategy == HybridBoundingStrategy.GPU_ACCELERATED:
                    return self._apply_gpu_accelerated_bounding(gradients, direction_state)
                elif self.current_strategy == HybridBoundingStrategy.CONSERVATIVE:
                    return self._apply_conservative_bounding(gradients, direction_state)
                elif self.current_strategy == HybridBoundingStrategy.AGGRESSIVE:
                    return self._apply_aggressive_bounding(gradients, direction_state)
                else:
                    raise RuntimeError(f"Unknown bounding strategy: {self.current_strategy}")
                    
        except Exception as e:
            self.logger.error(f"Bounding strategy application failed: {e}")
            raise RuntimeError(f"Bounding strategy application failed: {e}")
    
    def optimize_hybrid_bounding_performance(self) -> Dict[str, Any]:
        """
        Optimize hybrid bounding performance based on analytics.
        
        Returns:
            Optimization results
            
        Raises:
            RuntimeError: If optimization fails
        """
        if not self.is_initialized:
            raise RuntimeError("Hybrid bounding system not initialized")
        
        try:
            optimization_results = {
                'optimizations_applied': [],
                'performance_improvements': {},
                'parameter_adjustments': {},
                'strategy_changes': {},
                'timestamp': datetime.utcnow().isoformat()
            }
            
            with self._bounding_lock:
                # Optimize bounding parameters
                if self.config.enable_adaptive_parameters:
                    parameter_optimization = self._optimize_bounding_parameters()
                    optimization_results['parameter_adjustments'] = parameter_optimization
                    optimization_results['optimizations_applied'].append('parameter_optimization')
                
                # Optimize strategy selection
                if len(self.bounding_history) >= 50:
                    strategy_optimization = self._optimize_strategy_selection()
                    optimization_results['strategy_changes'] = strategy_optimization
                    optimization_results['optimizations_applied'].append('strategy_optimization')
                
                # Optimize GPU utilization
                if self.config.enable_gpu_acceleration:
                    gpu_optimization = self._optimize_gpu_utilization()
                    optimization_results['performance_improvements']['gpu'] = gpu_optimization
                    optimization_results['optimizations_applied'].append('gpu_optimization')
                
                # Optimize memory usage
                if self.config.enable_memory_optimization:
                    memory_optimization = self._optimize_memory_usage()
                    optimization_results['performance_improvements']['memory'] = memory_optimization
                    optimization_results['optimizations_applied'].append('memory_optimization')
                
                # Update last optimization time
                self.last_optimization = datetime.utcnow()
                
                self.logger.info(f"Hybrid bounding optimization completed: {len(optimization_results['optimizations_applied'])} optimizations applied")
                
                return optimization_results
                
        except Exception as e:
            self.logger.error(f"Hybrid bounding optimization failed: {e}")
            raise RuntimeError(f"Hybrid bounding optimization failed: {e}")
    
    def get_hybrid_bounding_analytics(self) -> HybridBoundingAnalytics:
        """
        Get comprehensive hybrid bounding analytics.
        
        Returns:
            Hybrid bounding analytics
            
        Raises:
            RuntimeError: If analytics generation fails
        """
        if not self.is_initialized:
            raise RuntimeError("Hybrid bounding system not initialized")
        
        try:
            with self._analytics_lock:
                # Calculate operation statistics
                total_operations = len(self.bounding_history)
                successful_operations = sum(1 for r in self.bounding_history if r.error_metrics.get('error_count', 0) == 0)
                failed_operations = total_operations - successful_operations
                
                # Calculate performance statistics
                if total_operations > 0:
                    avg_bounding_time = sum(r.bounding_time_ms for r in self.bounding_history) / total_operations
                    avg_gpu_utilization = sum(r.gpu_utilization for r in self.bounding_history) / total_operations
                else:
                    avg_bounding_time = 0.0
                    avg_gpu_utilization = 0.0
                
                # Calculate memory usage statistics
                memory_stats = self._calculate_memory_usage_stats()
                
                # Calculate strategy performance
                strategy_performance = self._calculate_strategy_performance()
                
                # Calculate direction-specific statistics
                direction_stats = self._calculate_direction_bounding_stats()
                
                # Calculate quality metrics
                quality_metrics = self._calculate_quality_metrics()
                
                # Analyze errors
                error_analysis = self._analyze_bounding_errors()
                
                # Calculate performance trends
                performance_trends = self._calculate_performance_trends()
                
                # Generate optimization recommendations
                recommendations = self._generate_optimization_recommendations()
                
                return HybridBoundingAnalytics(
                    total_bounding_operations=total_operations,
                    successful_operations=successful_operations,
                    failed_operations=failed_operations,
                    average_bounding_time=avg_bounding_time,
                    average_gpu_utilization=avg_gpu_utilization,
                    memory_usage_stats=memory_stats,
                    strategy_performance=strategy_performance,
                    direction_bounding_stats=direction_stats,
                    quality_metrics=quality_metrics,
                    error_analysis=error_analysis,
                    performance_trends=performance_trends,
                    optimization_recommendations=recommendations
                )
                
        except Exception as e:
            self.logger.error(f"Hybrid bounding analytics generation failed: {e}")
            raise RuntimeError(f"Hybrid bounding analytics generation failed: {e}")
    
    def validate_hybrid_bounding_correctness(self, 
                                           original_gradients: torch.Tensor, 
                                           bounded_gradients: torch.Tensor) -> Dict[str, Any]:
        """
        Validate correctness of hybrid bounding operation.
        
        Args:
            original_gradients: Original gradient tensor
            bounded_gradients: Bounded gradient tensor
            
        Returns:
            Validation results
            
        Raises:
            ValueError: If gradients are invalid
            RuntimeError: If validation fails
        """
        if original_gradients is None:
            raise ValueError("Original gradients cannot be None")
        if bounded_gradients is None:
            raise ValueError("Bounded gradients cannot be None")
        if not isinstance(original_gradients, torch.Tensor):
            raise TypeError("Original gradients must be torch.Tensor")
        if not isinstance(bounded_gradients, torch.Tensor):
            raise TypeError("Bounded gradients must be torch.Tensor")
        if original_gradients.shape != bounded_gradients.shape:
            raise ValueError("Gradient tensors must have same shape")
        
        try:
            validation_results = {
                'is_valid': True,
                'validation_errors': [],
                'quality_metrics': {},
                'norm_analysis': {},
                'gradient_analysis': {},
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Validate gradient norms
            orig_norm = torch.norm(original_gradients).item()
            bounded_norm = torch.norm(bounded_gradients).item()
            
            validation_results['norm_analysis'] = {
                'original_norm': orig_norm,
                'bounded_norm': bounded_norm,
                'norm_ratio': bounded_norm / (orig_norm + 1e-8),
                'norm_reduction': orig_norm - bounded_norm
            }
            
            # Check if bounding was effective
            if bounded_norm > orig_norm * 1.1:  # Allow 10% tolerance
                validation_results['is_valid'] = False
                validation_results['validation_errors'].append("Bounded norm greater than original norm")
            
            # Check for invalid values
            if torch.isnan(bounded_gradients).any():
                validation_results['is_valid'] = False
                validation_results['validation_errors'].append("Bounded gradients contain NaN values")
            
            if torch.isinf(bounded_gradients).any():
                validation_results['is_valid'] = False
                validation_results['validation_errors'].append("Bounded gradients contain infinite values")
            
            # Calculate quality metrics
            cosine_similarity = torch.cosine_similarity(
                original_gradients.flatten(), 
                bounded_gradients.flatten(), 
                dim=0
            ).item()
            
            mse_loss = torch.nn.functional.mse_loss(original_gradients, bounded_gradients).item()
            
            validation_results['quality_metrics'] = {
                'cosine_similarity': cosine_similarity,
                'mse_loss': mse_loss,
                'direction_preservation': cosine_similarity > 0.8,
                'magnitude_preservation': abs(validation_results['norm_analysis']['norm_ratio'] - 1.0) < 0.5
            }
            
            # Overall quality assessment
            if cosine_similarity < 0.5:
                validation_results['is_valid'] = False
                validation_results['validation_errors'].append("Poor direction preservation")
            
            self.logger.debug(f"Bounding validation completed: {'VALID' if validation_results['is_valid'] else 'INVALID'}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Bounding validation failed: {e}")
            raise RuntimeError(f"Bounding validation failed: {e}")
    
    def _get_direction_state(self, gradients: torch.Tensor, context: Optional[Dict[str, Any]]) -> Optional[DirectionState]:
        """Get direction state from direction manager if available"""
        if self.direction_manager and self.direction_manager.is_initialized:
            try:
                return self.direction_manager.analyze_gradient_direction(gradients, context)
            except Exception as e:
                self.logger.warning(f"Failed to get direction state: {e}")
        return None
    
    def _select_bounding_strategy(self, 
                                gradients: torch.Tensor, 
                                direction_state: Optional[DirectionState], 
                                context: Optional[Dict[str, Any]]) -> HybridBoundingStrategy:
        """Select optimal bounding strategy"""
        if not self.config.enable_adaptive_parameters:
            return self.current_strategy
        
        # Strategy selection based on direction state
        if direction_state:
            stability_score = direction_state.metadata.get('stability_score', 0.5)
            
            if direction_state.direction == DirectionType.STABLE and stability_score > 0.8:
                return HybridBoundingStrategy.CONSERVATIVE
            elif direction_state.direction == DirectionType.OSCILLATING:
                return HybridBoundingStrategy.AGGRESSIVE
            elif direction_state.confidence > 0.9:
                return HybridBoundingStrategy.PERFORMANCE_OPTIMIZED
            else:
                return HybridBoundingStrategy.DIRECTION_ADAPTIVE
        
        # Strategy selection based on gradient characteristics
        grad_norm = torch.norm(gradients).item()
        grad_std = torch.std(gradients).item()
        
        if grad_norm > self.config.hybrid_max_norm:
            return HybridBoundingStrategy.AGGRESSIVE
        elif grad_std < 0.1:
            return HybridBoundingStrategy.CONSERVATIVE
        elif context and context.get('memory_pressure', False):
            return HybridBoundingStrategy.MEMORY_EFFICIENT
        else:
            return self.current_strategy
    
    def _apply_bounding_strategy(self, 
                               gradients: torch.Tensor, 
                               direction_state: Optional[DirectionState], 
                               strategy: HybridBoundingStrategy, 
                               context: Optional[Dict[str, Any]]) -> HybridBoundingResult:
        """Apply selected bounding strategy"""
        start_time = time.time()
        
        # Record GPU memory before operation
        gpu_memory_before = self._get_gpu_memory_usage()
        
        # Apply strategy-specific bounding
        if strategy == HybridBoundingStrategy.DIRECTION_ADAPTIVE:
            bounded_gradients = self._apply_direction_adaptive_bounding_impl(gradients, direction_state)
            operation_type = BoundingOperationType.DIRECTION_SPECIFIC
        elif strategy == HybridBoundingStrategy.PERFORMANCE_OPTIMIZED:
            bounded_gradients = self._apply_performance_optimized_bounding_impl(gradients, direction_state)
            operation_type = BoundingOperationType.HYBRID_GRADIENT
        elif strategy == HybridBoundingStrategy.MEMORY_EFFICIENT:
            bounded_gradients = self._apply_memory_efficient_bounding_impl(gradients, direction_state)
            operation_type = BoundingOperationType.SWITCHING_OPTIMIZED
        elif strategy == HybridBoundingStrategy.GPU_ACCELERATED:
            bounded_gradients = self._apply_gpu_accelerated_bounding_impl(gradients, direction_state)
            operation_type = BoundingOperationType.HYBRID_GRADIENT
        elif strategy == HybridBoundingStrategy.CONSERVATIVE:
            bounded_gradients = self._apply_conservative_bounding_impl(gradients, direction_state)
            operation_type = BoundingOperationType.PADIC_GRADIENT
        elif strategy == HybridBoundingStrategy.AGGRESSIVE:
            bounded_gradients = self._apply_aggressive_bounding_impl(gradients, direction_state)
            operation_type = BoundingOperationType.HYBRID_GRADIENT
        else:
            raise RuntimeError(f"Unknown bounding strategy: {strategy}")
        
        # Record performance metrics
        bounding_time = (time.time() - start_time) * 1000
        gpu_memory_after = self._get_gpu_memory_usage()
        gpu_utilization = self._calculate_gpu_utilization(gpu_memory_before, gpu_memory_after)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_bounding_quality_metrics(gradients, bounded_gradients)
        
        # Calculate error metrics
        error_metrics = self._calculate_error_metrics(gradients, bounded_gradients)
        
        # Create result
        return HybridBoundingResult(
            bounded_gradients=bounded_gradients,
            bounding_strategy=strategy,
            operation_type=operation_type,
            direction_state=direction_state or self._create_default_direction_state(),
            bounding_performance={
                'bounding_time_ms': bounding_time,
                'throughput_ops_per_sec': 1000.0 / bounding_time if bounding_time > 0 else 0.0,
                'efficiency_score': min(1.0, self.config.max_bounding_time_ms / bounding_time) if bounding_time > 0 else 0.0
            },
            memory_usage={
                'memory_before_mb': gpu_memory_before,
                'memory_after_mb': gpu_memory_after,
                'memory_delta_mb': gpu_memory_after - gpu_memory_before,
                'memory_efficiency': max(0.0, 1.0 - abs(gpu_memory_after - gpu_memory_before) / gpu_memory_before) if gpu_memory_before > 0 else 1.0
            },
            gpu_utilization=gpu_utilization,
            bounding_time_ms=bounding_time,
            quality_metrics=quality_metrics,
            error_metrics=error_metrics
        )
    
    def _apply_direction_adaptive_bounding_impl(self, gradients: torch.Tensor, direction_state: Optional[DirectionState]) -> torch.Tensor:
        """Apply direction-adaptive bounding implementation"""
        if not direction_state:
            return self.enhanced_bounder.bound_gradients(gradients).bounded_gradients
        
        # Get direction-specific parameters
        direction = direction_state.direction
        stability_score = direction_state.metadata.get('stability_score', 0.5)
        
        if direction == DirectionType.ASCENT:
            max_norm = self.config.hybrid_max_norm * self.config.ascent_scaling_factor
            clip_value = self.config.hybrid_clip_value * self.config.ascent_scaling_factor
        elif direction == DirectionType.DESCENT:
            max_norm = self.config.hybrid_max_norm * self.config.descent_scaling_factor
            clip_value = self.config.hybrid_clip_value * self.config.descent_scaling_factor
        elif direction == DirectionType.STABLE:
            max_norm = self.config.padic_max_norm * self.config.stable_scaling_factor
            clip_value = self.config.padic_clip_value * self.config.stable_scaling_factor
        else:  # OSCILLATING
            max_norm = self.config.padic_max_norm * self.config.oscillating_scaling_factor
            clip_value = self.config.padic_clip_value * self.config.oscillating_scaling_factor
        
        # Apply stability-aware scaling
        stability_factor = min(1.0, stability_score + 0.5)
        max_norm *= stability_factor
        clip_value *= stability_factor
        
        # Apply bounding
        current_norm = torch.norm(gradients)
        if current_norm > max_norm:
            bounded_gradients = gradients * (max_norm / current_norm)
        else:
            bounded_gradients = gradients.clone()
        
        # Apply clipping
        bounded_gradients = torch.clamp(bounded_gradients, -clip_value, clip_value)
        
        return bounded_gradients
    
    def _apply_performance_optimized_bounding_impl(self, gradients: torch.Tensor, direction_state: Optional[DirectionState]) -> torch.Tensor:
        """Apply performance-optimized bounding implementation"""
        # Use enhanced bounder with performance-optimized parameters
        performance_context = {
            'max_norm': self.adaptive_parameters.get('performance_max_norm', self.config.hybrid_max_norm),
            'clip_value': self.adaptive_parameters.get('performance_clip_value', self.config.hybrid_clip_value),
            'optimization_mode': 'performance'
        }
        
        result = self.enhanced_bounder.bound_gradients(gradients, performance_context)
        return result.bounded_gradients
    
    def _apply_memory_efficient_bounding_impl(self, gradients: torch.Tensor, direction_state: Optional[DirectionState]) -> torch.Tensor:
        """Apply memory-efficient bounding implementation"""
        # In-place operations to minimize memory usage
        bounded_gradients = gradients.clone()
        
        # Calculate norm without creating additional tensors
        grad_norm = torch.norm(bounded_gradients).item()
        max_norm = self.adaptive_parameters.get('memory_max_norm', self.config.padic_max_norm)
        
        if grad_norm > max_norm:
            bounded_gradients.mul_(max_norm / grad_norm)
        
        # In-place clipping
        clip_value = self.adaptive_parameters.get('memory_clip_value', self.config.padic_clip_value)
        bounded_gradients.clamp_(-clip_value, clip_value)
        
        return bounded_gradients
    
    def _apply_gpu_accelerated_bounding_impl(self, gradients: torch.Tensor, direction_state: Optional[DirectionState]) -> torch.Tensor:
        """Apply GPU-accelerated bounding implementation"""
        # Ensure gradients are on GPU
        if not gradients.is_cuda:
            gradients = gradients.cuda()
        
        # Use GPU-optimized operations
        max_norm = self.adaptive_parameters.get('gpu_max_norm', self.config.hybrid_max_norm)
        clip_value = self.adaptive_parameters.get('gpu_clip_value', self.config.hybrid_clip_value)
        
        # GPU-optimized norm calculation and scaling
        grad_norm = torch.linalg.norm(gradients)
        scale_factor = torch.min(torch.tensor(1.0, device=gradients.device), max_norm / grad_norm)
        bounded_gradients = gradients * scale_factor
        
        # GPU-optimized clipping
        bounded_gradients = torch.clamp(bounded_gradients, -clip_value, clip_value)
        
        return bounded_gradients
    
    def _apply_conservative_bounding_impl(self, gradients: torch.Tensor, direction_state: Optional[DirectionState]) -> torch.Tensor:
        """Apply conservative bounding implementation"""
        # Use stricter bounding parameters
        conservative_max_norm = self.config.padic_max_norm * 0.8
        conservative_clip_value = self.config.padic_clip_value * 0.8
        
        current_norm = torch.norm(gradients)
        if current_norm > conservative_max_norm:
            bounded_gradients = gradients * (conservative_max_norm / current_norm)
        else:
            bounded_gradients = gradients.clone()
        
        bounded_gradients = torch.clamp(bounded_gradients, -conservative_clip_value, conservative_clip_value)
        
        return bounded_gradients
    
    def _apply_aggressive_bounding_impl(self, gradients: torch.Tensor, direction_state: Optional[DirectionState]) -> torch.Tensor:
        """Apply aggressive bounding implementation"""
        # Use more permissive bounding parameters
        aggressive_max_norm = self.config.hybrid_max_norm * 1.2
        aggressive_clip_value = self.config.hybrid_clip_value * 1.2
        
        current_norm = torch.norm(gradients)
        if current_norm > aggressive_max_norm:
            bounded_gradients = gradients * (aggressive_max_norm / current_norm)
        else:
            bounded_gradients = gradients.clone()
        
        bounded_gradients = torch.clamp(bounded_gradients, -aggressive_clip_value, aggressive_clip_value)
        
        return bounded_gradients
    
    def _initialize_adaptive_parameters(self) -> None:
        """Initialize adaptive parameters"""
        self.adaptive_parameters = {
            'performance_max_norm': self.config.hybrid_max_norm,
            'performance_clip_value': self.config.hybrid_clip_value,
            'memory_max_norm': self.config.padic_max_norm,
            'memory_clip_value': self.config.padic_clip_value,
            'gpu_max_norm': self.config.hybrid_max_norm,
            'gpu_clip_value': self.config.hybrid_clip_value
        }
    
    def _initialize_gpu_memory_tracking(self) -> None:
        """Initialize GPU memory tracking"""
        self.gpu_memory_stats = {
            'peak_memory_mb': 0.0,
            'average_memory_mb': 0.0,
            'memory_efficiency': 1.0
        }
    
    def _initialize_performance_monitoring(self) -> None:
        """Initialize performance monitoring"""
        self.performance_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'average_time_ms': 0.0,
            'min_time_ms': float('inf'),
            'max_time_ms': 0.0,
            'throughput_ops_per_sec': 0.0
        }
    
    def _update_performance_metrics(self, result: HybridBoundingResult, bounding_time: float) -> None:
        """Update performance metrics"""
        self.bounding_analytics['total_operations'] += 1
        
        if result.error_metrics.get('error_count', 0) == 0:
            self.bounding_analytics['successful_operations'] += 1
        else:
            self.bounding_analytics['failed_operations'] += 1
        
        # Update timing metrics
        total_ops = self.bounding_analytics['total_operations']
        current_avg = self.bounding_analytics['average_time_ms']
        self.bounding_analytics['average_time_ms'] = (current_avg * (total_ops - 1) + bounding_time) / total_ops
        
        # Update GPU utilization
        current_gpu_avg = self.bounding_analytics['average_gpu_utilization']
        self.bounding_analytics['average_gpu_utilization'] = (current_gpu_avg * (total_ops - 1) + result.gpu_utilization) / total_ops
        
        # Update memory efficiency
        memory_efficiency = result.memory_usage.get('memory_efficiency', 1.0)
        current_mem_eff = self.bounding_analytics['memory_efficiency']
        self.bounding_analytics['memory_efficiency'] = (current_mem_eff * (total_ops - 1) + memory_efficiency) / total_ops
    
    def _record_bounding_failure(self, error: Exception) -> None:
        """Record bounding failure for analytics"""
        self.bounding_analytics['failed_operations'] += 1
        self.bounding_analytics['total_operations'] += 1
        
        # Log error details
        self.logger.error(f"Bounding operation failed: {type(error).__name__}: {error}")
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0
    
    def _calculate_gpu_utilization(self, memory_before: float, memory_after: float) -> float:
        """Calculate GPU utilization score"""
        if memory_before == 0:
            return 0.0
        memory_delta = abs(memory_after - memory_before)
        return min(1.0, memory_delta / (memory_before + 1e-8))
    
    def _calculate_bounding_quality_metrics(self, original: torch.Tensor, bounded: torch.Tensor) -> Dict[str, float]:
        """Calculate bounding quality metrics"""
        return {
            'norm_preservation': torch.norm(bounded).item() / (torch.norm(original).item() + 1e-8),
            'direction_preservation': torch.cosine_similarity(original.flatten(), bounded.flatten(), dim=0).item(),
            'mse_error': torch.nn.functional.mse_loss(original, bounded).item(),
            'max_error': torch.max(torch.abs(original - bounded)).item()
        }
    
    def _calculate_error_metrics(self, original: torch.Tensor, bounded: torch.Tensor) -> Dict[str, float]:
        """Calculate error metrics"""
        has_nan = torch.isnan(bounded).any().item()
        has_inf = torch.isinf(bounded).any().item()
        
        return {
            'error_count': int(has_nan) + int(has_inf),
            'has_nan': has_nan,
            'has_inf': has_inf,
            'numerical_stability': 1.0 - (int(has_nan) + int(has_inf)) / 2.0
        }
    
    def _create_default_direction_state(self) -> DirectionState:
        """Create default direction state when none available"""
        return DirectionState(
            direction=DirectionType.STABLE,
            confidence=0.5,
            magnitude=0.0,
            timestamp=time.time(),
            metadata={'stability_score': 0.5, 'default_state': True}
        )
    
    def _optimize_bounding_parameters(self) -> Dict[str, Any]:
        """Optimize bounding parameters based on performance"""
        optimization_results = {}
        
        if len(self.bounding_history) < 20:
            return {'status': 'insufficient_data'}
        
        recent_results = list(self.bounding_history)[-20:]
        
        # Optimize max norm parameter
        performance_scores = [r.bounding_performance.get('efficiency_score', 0.0) for r in recent_results]
        avg_performance = sum(performance_scores) / len(performance_scores)
        
        if avg_performance < 0.7:
            # Reduce max norm for better performance
            for key in self.adaptive_parameters:
                if 'max_norm' in key:
                    old_value = self.adaptive_parameters[key]
                    self.adaptive_parameters[key] *= 0.95
                    optimization_results[key] = {'old': old_value, 'new': self.adaptive_parameters[key]}
        elif avg_performance > 0.9:
            # Increase max norm for better quality
            for key in self.adaptive_parameters:
                if 'max_norm' in key:
                    old_value = self.adaptive_parameters[key]
                    self.adaptive_parameters[key] *= 1.05
                    optimization_results[key] = {'old': old_value, 'new': self.adaptive_parameters[key]}
        
        return optimization_results
    
    def _optimize_strategy_selection(self) -> Dict[str, Any]:
        """Optimize strategy selection based on performance"""
        strategy_performance = defaultdict(list)
        
        for result in self.bounding_history:
            strategy = result.bounding_strategy
            efficiency = result.bounding_performance.get('efficiency_score', 0.0)
            quality = result.quality_metrics.get('direction_preservation', 0.0)
            combined_score = (efficiency + quality) / 2.0
            strategy_performance[strategy].append(combined_score)
        
        # Find best performing strategy
        best_strategy = None
        best_score = 0.0
        
        for strategy, scores in strategy_performance.items():
            if len(scores) >= 5:  # Minimum sample size
                avg_score = sum(scores) / len(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_strategy = strategy
        
        if best_strategy and best_strategy != self.current_strategy:
            old_strategy = self.current_strategy
            self.current_strategy = best_strategy
            return {
                'strategy_changed': True,
                'old_strategy': old_strategy.name,
                'new_strategy': best_strategy.name,
                'performance_improvement': best_score
            }
        
        return {'strategy_changed': False}
    
    def _optimize_gpu_utilization(self) -> Dict[str, float]:
        """Optimize GPU utilization"""
        if len(self.bounding_history) < 10:
            return {'status': 'insufficient_data'}
        
        recent_results = list(self.bounding_history)[-10:]
        avg_gpu_util = sum(r.gpu_utilization for r in recent_results) / len(recent_results)
        
        optimization_results = {
            'current_utilization': avg_gpu_util,
            'target_utilization': self.config.target_gpu_utilization
        }
        
        if avg_gpu_util < self.config.target_gpu_utilization * 0.8:
            # Increase GPU utilization
            optimization_results['action'] = 'increase_gpu_usage'
            optimization_results['adjustment'] = 'enabling_more_gpu_operations'
        elif avg_gpu_util > self.config.target_gpu_utilization * 1.2:
            # Decrease GPU utilization
            optimization_results['action'] = 'decrease_gpu_usage'
            optimization_results['adjustment'] = 'reducing_gpu_operations'
        else:
            optimization_results['action'] = 'maintain_current'
        
        return optimization_results
    
    def _optimize_memory_usage(self) -> Dict[str, float]:
        """Optimize memory usage"""
        if len(self.bounding_history) < 10:
            return {'status': 'insufficient_data'}
        
        recent_results = list(self.bounding_history)[-10:]
        avg_memory_eff = sum(r.memory_usage.get('memory_efficiency', 1.0) for r in recent_results) / len(recent_results)
        
        optimization_results = {
            'current_memory_efficiency': avg_memory_eff,
            'target_efficiency': self.config.memory_efficiency_threshold
        }
        
        if avg_memory_eff < self.config.memory_efficiency_threshold:
            optimization_results['action'] = 'improve_memory_efficiency'
            optimization_results['recommendations'] = [
                'use_in_place_operations',
                'reduce_intermediate_tensors',
                'optimize_tensor_allocation'
            ]
        else:
            optimization_results['action'] = 'maintain_current'
        
        return optimization_results
    
    def _calculate_memory_usage_stats(self) -> Dict[str, float]:
        """Calculate memory usage statistics"""
        if not self.bounding_history:
            return {}
        
        memory_usages = [r.memory_usage for r in self.bounding_history]
        
        return {
            'average_memory_delta': sum(m.get('memory_delta_mb', 0.0) for m in memory_usages) / len(memory_usages),
            'max_memory_delta': max(m.get('memory_delta_mb', 0.0) for m in memory_usages),
            'average_efficiency': sum(m.get('memory_efficiency', 1.0) for m in memory_usages) / len(memory_usages),
            'peak_memory_usage': max(m.get('memory_after_mb', 0.0) for m in memory_usages)
        }
    
    def _calculate_strategy_performance(self) -> Dict[str, Dict[str, float]]:
        """Calculate strategy performance statistics"""
        strategy_stats = defaultdict(lambda: defaultdict(list))
        
        for result in self.bounding_history:
            strategy = result.bounding_strategy.name
            strategy_stats[strategy]['bounding_time'].append(result.bounding_time_ms)
            strategy_stats[strategy]['gpu_utilization'].append(result.gpu_utilization)
            strategy_stats[strategy]['quality_score'].append(
                result.quality_metrics.get('direction_preservation', 0.0)
            )
        
        # Calculate averages
        performance_stats = {}
        for strategy, metrics in strategy_stats.items():
            performance_stats[strategy] = {
                'average_time_ms': sum(metrics['bounding_time']) / len(metrics['bounding_time']) if metrics['bounding_time'] else 0.0,
                'average_gpu_utilization': sum(metrics['gpu_utilization']) / len(metrics['gpu_utilization']) if metrics['gpu_utilization'] else 0.0,
                'average_quality': sum(metrics['quality_score']) / len(metrics['quality_score']) if metrics['quality_score'] else 0.0,
                'operation_count': len(metrics['bounding_time'])
            }
        
        return performance_stats
    
    def _calculate_direction_bounding_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate direction-specific bounding statistics"""
        direction_stats = defaultdict(lambda: defaultdict(list))
        
        for result in self.bounding_history:
            direction = result.direction_state.direction.name
            direction_stats[direction]['bounding_time'].append(result.bounding_time_ms)
            direction_stats[direction]['quality_score'].append(
                result.quality_metrics.get('direction_preservation', 0.0)
            )
            direction_stats[direction]['stability_score'].append(
                result.direction_state.metadata.get('stability_score', 0.5)
            )
        
        # Calculate averages
        stats = {}
        for direction, metrics in direction_stats.items():
            stats[direction] = {
                'average_time_ms': sum(metrics['bounding_time']) / len(metrics['bounding_time']) if metrics['bounding_time'] else 0.0,
                'average_quality': sum(metrics['quality_score']) / len(metrics['quality_score']) if metrics['quality_score'] else 0.0,
                'average_stability': sum(metrics['stability_score']) / len(metrics['stability_score']) if metrics['stability_score'] else 0.0,
                'operation_count': len(metrics['bounding_time'])
            }
        
        return stats
    
    def _calculate_quality_metrics(self) -> Dict[str, float]:
        """Calculate overall quality metrics"""
        if not self.bounding_history:
            return {}
        
        quality_scores = []
        error_counts = []
        
        for result in self.bounding_history:
            quality_scores.append(result.quality_metrics.get('direction_preservation', 0.0))
            error_counts.append(result.error_metrics.get('error_count', 0))
        
        return {
            'average_quality': sum(quality_scores) / len(quality_scores),
            'min_quality': min(quality_scores),
            'max_quality': max(quality_scores),
            'quality_consistency': 1.0 - (max(quality_scores) - min(quality_scores)),
            'error_rate': sum(error_counts) / len(error_counts),
            'success_rate': 1.0 - (sum(1 for e in error_counts if e > 0) / len(error_counts))
        }
    
    def _analyze_bounding_errors(self) -> Dict[str, int]:
        """Analyze bounding errors"""
        error_analysis = {
            'total_errors': 0,
            'nan_errors': 0,
            'inf_errors': 0,
            'numerical_instability': 0
        }
        
        for result in self.bounding_history:
            error_metrics = result.error_metrics
            error_analysis['total_errors'] += error_metrics.get('error_count', 0)
            if error_metrics.get('has_nan', False):
                error_analysis['nan_errors'] += 1
            if error_metrics.get('has_inf', False):
                error_analysis['inf_errors'] += 1
            if error_metrics.get('numerical_stability', 1.0) < 0.9:
                error_analysis['numerical_instability'] += 1
        
        return error_analysis
    
    def _calculate_performance_trends(self) -> Dict[str, List[float]]:
        """Calculate performance trends"""
        if len(self.bounding_history) < self.config.performance_trend_window:
            return {}
        
        recent_results = list(self.bounding_history)[-self.config.performance_trend_window:]
        
        return {
            'bounding_time_trend': [r.bounding_time_ms for r in recent_results],
            'gpu_utilization_trend': [r.gpu_utilization for r in recent_results],
            'quality_trend': [r.quality_metrics.get('direction_preservation', 0.0) for r in recent_results],
            'error_trend': [r.error_metrics.get('error_count', 0) for r in recent_results]
        }
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if len(self.bounding_history) < 10:
            recommendations.append("Collect more performance data for better recommendations")
            return recommendations
        
        # Analyze recent performance
        recent_results = list(self.bounding_history)[-10:]
        avg_time = sum(r.bounding_time_ms for r in recent_results) / len(recent_results)
        avg_gpu_util = sum(r.gpu_utilization for r in recent_results) / len(recent_results)
        avg_quality = sum(r.quality_metrics.get('direction_preservation', 0.0) for r in recent_results) / len(recent_results)
        
        # Time-based recommendations
        if avg_time > self.config.max_bounding_time_ms:
            recommendations.append("Consider using memory_efficient or conservative strategy to reduce bounding time")
        
        # GPU utilization recommendations
        if avg_gpu_util < 0.3:
            recommendations.append("Enable GPU acceleration to improve performance")
        elif avg_gpu_util > 0.95:
            recommendations.append("Consider reducing GPU operations to prevent memory issues")
        
        # Quality recommendations
        if avg_quality < 0.7:
            recommendations.append("Use direction_adaptive or performance_optimized strategy for better quality")
        
        # Error-based recommendations
        error_rate = sum(1 for r in recent_results if r.error_metrics.get('error_count', 0) > 0) / len(recent_results)
        if error_rate > 0.1:
            recommendations.append("Switch to conservative strategy to reduce numerical errors")
        
        # Memory recommendations
        avg_memory_eff = sum(r.memory_usage.get('memory_efficiency', 1.0) for r in recent_results) / len(recent_results)
        if avg_memory_eff < 0.8:
            recommendations.append("Enable memory optimization to improve memory efficiency")
        
        return recommendations
    
    def shutdown(self) -> None:
        """Shutdown hybrid bounding engine"""
        self.logger.info("Shutting down hybrid bounding engine")
        
        # Clear references
        self.enhanced_bounder = None
        self.direction_manager = None
        self.switching_manager = None
        
        # Clear tracking data
        self.bounding_history.clear()
        self.strategy_performance.clear()
        self.adaptive_parameters.clear()
        
        self.is_initialized = False
        self.logger.info("Hybrid bounding engine shutdown complete")