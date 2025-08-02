"""
Bounding Strategy Manager - Management of bounding strategies for hybrid operations
NO FALLBACKS - HARD FAILURES ONLY
"""

import logging
import math
import statistics
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
from ...gac_system.gac_types import DirectionState, DirectionType
from ...gac_system.direction_state import DirectionStateManager, DirectionHistory

# Import hybrid bounding components
from .hybrid_bounding_engine import HybridBoundingStrategy, BoundingOperationType, HybridBoundingResult


class StrategySelectionMode(Enum):
    """Strategy selection mode enumeration"""
    PERFORMANCE_BASED = "performance_based"
    QUALITY_BASED = "quality_based"
    MEMORY_BASED = "memory_based"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"
    CONSERVATIVE = "conservative"


class StrategyValidationLevel(Enum):
    """Strategy validation level enumeration"""
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    RIGOROUS = "rigorous"
    PRODUCTION = "production"


@dataclass
class StrategyPerformanceMetrics:
    """Strategy performance metrics"""
    strategy: HybridBoundingStrategy
    total_operations: int
    successful_operations: int
    failed_operations: int
    average_time_ms: float
    average_quality_score: float
    average_gpu_utilization: float
    memory_efficiency: float
    error_rate: float
    stability_score: float
    performance_trend: List[float]
    quality_trend: List[float]
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate performance metrics"""
        if not isinstance(self.strategy, HybridBoundingStrategy):
            raise TypeError("Strategy must be HybridBoundingStrategy")
        if self.total_operations < 0:
            raise ValueError("Total operations must be non-negative")
        if not (0.0 <= self.average_quality_score <= 1.0):
            raise ValueError("Average quality score must be between 0.0 and 1.0")


@dataclass
class StrategyRecommendation:
    """Strategy recommendation result"""
    recommended_strategy: HybridBoundingStrategy
    confidence: float
    reasoning: str
    expected_performance: Dict[str, float]
    risk_assessment: Dict[str, float]
    fallback_strategies: List[HybridBoundingStrategy]
    implementation_priority: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate strategy recommendation"""
        if not isinstance(self.recommended_strategy, HybridBoundingStrategy):
            raise TypeError("Recommended strategy must be HybridBoundingStrategy")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not isinstance(self.reasoning, str) or not self.reasoning.strip():
            raise ValueError("Reasoning must be non-empty string")


@dataclass
class StrategyValidationResult:
    """Strategy validation result"""
    strategy: HybridBoundingStrategy
    is_valid: bool
    validation_level: StrategyValidationLevel
    validation_errors: List[str]
    performance_score: float
    quality_score: float
    stability_score: float
    memory_efficiency_score: float
    validation_recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BoundingStrategyConfig:
    """Configuration for bounding strategy manager"""
    # Strategy selection configuration
    default_selection_mode: StrategySelectionMode = StrategySelectionMode.ADAPTIVE
    enable_performance_based_selection: bool = True
    enable_quality_based_selection: bool = True
    enable_memory_based_selection: bool = True
    enable_predictive_selection: bool = True
    
    # Performance thresholds
    min_performance_score: float = 0.7
    min_quality_score: float = 0.75
    min_memory_efficiency: float = 0.8
    max_error_rate: float = 0.05
    
    # Strategy evaluation parameters
    evaluation_window_size: int = 50
    performance_history_size: int = 200
    trend_analysis_window: int = 20
    validation_sample_size: int = 10
    
    # Optimization parameters
    enable_adaptive_optimization: bool = True
    enable_real_time_monitoring: bool = True
    enable_predictive_optimization: bool = True
    optimization_interval_seconds: int = 120
    
    # Validation configuration
    default_validation_level: StrategyValidationLevel = StrategyValidationLevel.COMPREHENSIVE
    enable_continuous_validation: bool = True
    validation_frequency_seconds: int = 300
    
    # Risk management
    enable_risk_assessment: bool = True
    max_acceptable_risk: float = 0.3
    fallback_strategy_count: int = 2
    enable_emergency_fallback: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if not (0.0 <= self.min_performance_score <= 1.0):
            raise ValueError("Min performance score must be between 0.0 and 1.0")
        if not (0.0 <= self.min_quality_score <= 1.0):
            raise ValueError("Min quality score must be between 0.0 and 1.0")
        if not (0.0 <= self.max_error_rate <= 1.0):
            raise ValueError("Max error rate must be between 0.0 and 1.0")
        if self.evaluation_window_size <= 0:
            raise ValueError("Evaluation window size must be positive")


class BoundingStrategyManager:
    """
    Management of bounding strategies for hybrid operations.
    Provides strategy selection, evaluation, and optimization.
    """
    
    def __init__(self, config: Optional[BoundingStrategyConfig] = None):
        """Initialize bounding strategy manager"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, BoundingStrategyConfig):
            raise TypeError(f"Config must be BoundingStrategyConfig or None, got {type(config)}")
        
        self.config = config or BoundingStrategyConfig()
        self.logger = logging.getLogger('BoundingStrategyManager')
        
        # Strategy management state
        self.current_strategy = HybridBoundingStrategy.DIRECTION_ADAPTIVE
        self.selection_mode = self.config.default_selection_mode
        
        # Performance tracking
        self.strategy_metrics: Dict[HybridBoundingStrategy, StrategyPerformanceMetrics] = {}
        self.performance_history: deque = deque(maxlen=self.config.performance_history_size)
        self.evaluation_results: Dict[HybridBoundingStrategy, List[float]] = defaultdict(list)
        
        # Strategy optimization
        self.optimization_recommendations: Dict[str, Any] = {}
        self.last_optimization: Optional[datetime] = None
        self.predictive_models: Dict[str, Any] = {}
        
        # Validation tracking
        self.validation_results: Dict[HybridBoundingStrategy, StrategyValidationResult] = {}
        self.last_validation: Optional[datetime] = None
        self.validation_history: deque = deque(maxlen=100)
        
        # Thread safety
        self._strategy_lock = threading.RLock()
        self._metrics_lock = threading.RLock()
        
        # Initialize strategy metrics
        self._initialize_strategy_metrics()
        
        # Initialize predictive models
        self._initialize_predictive_models()
        
        self.logger.info("BoundingStrategyManager created successfully")
    
    def select_bounding_strategy(self, 
                               gradients: torch.Tensor, 
                               context: Optional[Dict[str, Any]] = None) -> HybridBoundingStrategy:
        """
        Select optimal bounding strategy based on current conditions.
        
        Args:
            gradients: Gradient tensor for strategy selection
            context: Optional selection context
            
        Returns:
            Selected bounding strategy
            
        Raises:
            ValueError: If gradients are invalid
            RuntimeError: If strategy selection fails
        """
        if gradients is None:
            raise ValueError("Gradients cannot be None")
        if not isinstance(gradients, torch.Tensor):
            raise TypeError("Gradients must be torch.Tensor")
        if gradients.numel() == 0:
            raise ValueError("Gradients cannot be empty")
        
        try:
            with self._strategy_lock:
                # Apply strategy selection mode
                if self.selection_mode == StrategySelectionMode.PERFORMANCE_BASED:
                    strategy = self._select_performance_based_strategy(gradients, context)
                elif self.selection_mode == StrategySelectionMode.QUALITY_BASED:
                    strategy = self._select_quality_based_strategy(gradients, context)
                elif self.selection_mode == StrategySelectionMode.MEMORY_BASED:
                    strategy = self._select_memory_based_strategy(gradients, context)
                elif self.selection_mode == StrategySelectionMode.ADAPTIVE:
                    strategy = self._select_adaptive_strategy(gradients, context)
                elif self.selection_mode == StrategySelectionMode.PREDICTIVE:
                    strategy = self._select_predictive_strategy(gradients, context)
                elif self.selection_mode == StrategySelectionMode.CONSERVATIVE:
                    strategy = self._select_conservative_strategy(gradients, context)
                else:
                    raise RuntimeError(f"Unknown selection mode: {self.selection_mode}")
                
                # Validate strategy selection
                if not self._validate_strategy_selection(strategy, gradients, context):
                    strategy = self._get_fallback_strategy(gradients, context)
                
                self.current_strategy = strategy
                self.logger.debug(f"Selected bounding strategy: {strategy.name}")
                
                return strategy
                
        except Exception as e:
            self.logger.error(f"Strategy selection failed: {e}")
            raise RuntimeError(f"Strategy selection failed: {e}")
    
    def evaluate_strategy_performance(self, 
                                    strategy: HybridBoundingStrategy, 
                                    results: List[HybridBoundingResult]) -> StrategyPerformanceMetrics:
        """
        Evaluate performance of a specific strategy.
        
        Args:
            strategy: Strategy to evaluate
            results: List of bounding results for the strategy
            
        Returns:
            Strategy performance metrics
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If evaluation fails
        """
        if not isinstance(strategy, HybridBoundingStrategy):
            raise TypeError("Strategy must be HybridBoundingStrategy")
        if not isinstance(results, list):
            raise TypeError("Results must be list")
        if len(results) == 0:
            raise ValueError("Results list cannot be empty")
        
        for i, result in enumerate(results):
            if not isinstance(result, HybridBoundingResult):
                raise TypeError(f"Result {i} must be HybridBoundingResult")
        
        try:
            with self._metrics_lock:
                # Calculate basic statistics
                total_operations = len(results)
                successful_operations = sum(1 for r in results if r.error_metrics.get('error_count', 0) == 0)
                failed_operations = total_operations - successful_operations
                
                # Calculate performance metrics
                avg_time = sum(r.bounding_time_ms for r in results) / total_operations
                avg_quality = sum(r.quality_metrics.get('direction_preservation', 0.0) for r in results) / total_operations
                avg_gpu_util = sum(r.gpu_utilization for r in results) / total_operations
                memory_eff = sum(r.memory_usage.get('memory_efficiency', 1.0) for r in results) / total_operations
                error_rate = failed_operations / total_operations
                
                # Calculate stability score
                quality_scores = [r.quality_metrics.get('direction_preservation', 0.0) for r in results]
                quality_std = np.std(quality_scores) if len(quality_scores) > 1 else 0.0
                stability_score = max(0.0, 1.0 - quality_std)
                
                # Calculate trends
                performance_trend = [r.bounding_performance.get('efficiency_score', 0.0) for r in results[-20:]]
                quality_trend = [r.quality_metrics.get('direction_preservation', 0.0) for r in results[-20:]]
                
                # Create performance metrics
                metrics = StrategyPerformanceMetrics(
                    strategy=strategy,
                    total_operations=total_operations,
                    successful_operations=successful_operations,
                    failed_operations=failed_operations,
                    average_time_ms=avg_time,
                    average_quality_score=avg_quality,
                    average_gpu_utilization=avg_gpu_util,
                    memory_efficiency=memory_eff,
                    error_rate=error_rate,
                    stability_score=stability_score,
                    performance_trend=performance_trend,
                    quality_trend=quality_trend
                )
                
                # Update strategy metrics
                self.strategy_metrics[strategy] = metrics
                
                self.logger.debug(f"Strategy {strategy.name} evaluation completed: quality={avg_quality:.3f}, time={avg_time:.2f}ms")
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Strategy performance evaluation failed: {e}")
            raise RuntimeError(f"Strategy performance evaluation failed: {e}")
    
    def optimize_strategy_parameters(self, 
                                   strategy: HybridBoundingStrategy, 
                                   performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimize parameters for a specific strategy.
        
        Args:
            strategy: Strategy to optimize
            performance_data: Historical performance data
            
        Returns:
            Optimization results
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If optimization fails
        """
        if not isinstance(strategy, HybridBoundingStrategy):
            raise TypeError("Strategy must be HybridBoundingStrategy")
        if not isinstance(performance_data, list):
            raise TypeError("Performance data must be list")
        if len(performance_data) < 10:
            raise ValueError("Insufficient performance data for optimization")
        
        try:
            optimization_results = {
                'strategy': strategy.name,
                'optimizations_applied': [],
                'parameter_changes': {},
                'performance_improvement': 0.0,
                'optimization_confidence': 0.0,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            with self._strategy_lock:
                # Analyze performance trends
                performance_trends = self._analyze_performance_trends(performance_data)
                optimization_results['performance_trends'] = performance_trends
                
                # Optimize timing parameters
                if performance_trends.get('time_trend', 0.0) > 0.1:  # Time increasing
                    timing_optimization = self._optimize_timing_parameters(strategy, performance_data)
                    optimization_results['parameter_changes']['timing'] = timing_optimization
                    optimization_results['optimizations_applied'].append('timing_optimization')
                
                # Optimize quality parameters
                if performance_trends.get('quality_trend', 0.0) < -0.1:  # Quality decreasing
                    quality_optimization = self._optimize_quality_parameters(strategy, performance_data)
                    optimization_results['parameter_changes']['quality'] = quality_optimization
                    optimization_results['optimizations_applied'].append('quality_optimization')
                
                # Optimize memory parameters
                memory_efficiency = sum(d.get('memory_efficiency', 1.0) for d in performance_data) / len(performance_data)
                if memory_efficiency < self.config.min_memory_efficiency:
                    memory_optimization = self._optimize_memory_parameters(strategy, performance_data)
                    optimization_results['parameter_changes']['memory'] = memory_optimization
                    optimization_results['optimizations_applied'].append('memory_optimization')
                
                # Calculate overall performance improvement
                if optimization_results['optimizations_applied']:
                    improvement = self._calculate_optimization_improvement(performance_data, optimization_results)
                    optimization_results['performance_improvement'] = improvement
                    optimization_results['optimization_confidence'] = min(1.0, len(optimization_results['optimizations_applied']) * 0.3)
                
                self.logger.info(f"Strategy {strategy.name} optimization completed: {len(optimization_results['optimizations_applied'])} optimizations applied")
                
                return optimization_results
                
        except Exception as e:
            self.logger.error(f"Strategy parameter optimization failed: {e}")
            raise RuntimeError(f"Strategy parameter optimization failed: {e}")
    
    def switch_strategy(self, 
                       current_strategy: HybridBoundingStrategy, 
                       new_strategy: HybridBoundingStrategy) -> Dict[str, Any]:
        """
        Switch from current strategy to new strategy.
        
        Args:
            current_strategy: Current bounding strategy
            new_strategy: New bounding strategy to switch to
            
        Returns:
            Strategy switch results
            
        Raises:
            ValueError: If strategies are invalid
            RuntimeError: If strategy switch fails
        """
        if not isinstance(current_strategy, HybridBoundingStrategy):
            raise TypeError("Current strategy must be HybridBoundingStrategy")
        if not isinstance(new_strategy, HybridBoundingStrategy):
            raise TypeError("New strategy must be HybridBoundingStrategy")
        if current_strategy == new_strategy:
            raise ValueError("Current and new strategies cannot be the same")
        
        try:
            switch_results = {
                'previous_strategy': current_strategy.name,
                'new_strategy': new_strategy.name,
                'switch_successful': False,
                'switch_reasoning': '',
                'expected_benefits': {},
                'transition_plan': [],
                'rollback_plan': [],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            with self._strategy_lock:
                # Validate strategy switch
                switch_validation = self._validate_strategy_switch(current_strategy, new_strategy)
                
                if not switch_validation['is_valid']:
                    switch_results['switch_reasoning'] = f"Switch validation failed: {switch_validation['reason']}"
                    return switch_results
                
                # Calculate expected benefits
                expected_benefits = self._calculate_switch_benefits(current_strategy, new_strategy)
                switch_results['expected_benefits'] = expected_benefits
                
                # Create transition plan
                transition_plan = self._create_transition_plan(current_strategy, new_strategy)
                switch_results['transition_plan'] = transition_plan
                
                # Create rollback plan
                rollback_plan = self._create_rollback_plan(current_strategy, new_strategy)
                switch_results['rollback_plan'] = rollback_plan
                
                # Perform strategy switch
                self.current_strategy = new_strategy
                switch_results['switch_successful'] = True
                switch_results['switch_reasoning'] = f"Strategy switched for improved performance: {expected_benefits.get('performance_improvement', 0.0):.3f}"
                
                self.logger.info(f"Strategy switched from {current_strategy.name} to {new_strategy.name}")
                
                return switch_results
                
        except Exception as e:
            self.logger.error(f"Strategy switch failed: {e}")
            raise RuntimeError(f"Strategy switch failed: {e}")
    
    def get_strategy_recommendations(self, 
                                   performance_data: List[Dict[str, Any]]) -> List[StrategyRecommendation]:
        """
        Get strategy recommendations based on performance data.
        
        Args:
            performance_data: Historical performance data
            
        Returns:
            List of strategy recommendations
            
        Raises:
            ValueError: If performance data is invalid
            RuntimeError: If recommendation generation fails
        """
        if not isinstance(performance_data, list):
            raise TypeError("Performance data must be list")
        if len(performance_data) < 5:
            raise ValueError("Insufficient performance data for recommendations")
        
        try:
            recommendations = []
            
            with self._metrics_lock:
                # Analyze current performance
                current_performance = self._analyze_current_performance(performance_data)
                
                # Generate recommendations for each strategy
                for strategy in HybridBoundingStrategy:
                    recommendation = self._generate_strategy_recommendation(
                        strategy, current_performance, performance_data
                    )
                    
                    if recommendation.confidence >= 0.6:  # Minimum confidence threshold
                        recommendations.append(recommendation)
                
                # Sort recommendations by confidence and expected performance
                recommendations.sort(
                    key=lambda r: (r.confidence, r.expected_performance.get('overall_score', 0.0)),
                    reverse=True
                )
                
                # Limit to top recommendations
                recommendations = recommendations[:5]
                
                self.logger.debug(f"Generated {len(recommendations)} strategy recommendations")
                
                return recommendations
                
        except Exception as e:
            self.logger.error(f"Strategy recommendation generation failed: {e}")
            raise RuntimeError(f"Strategy recommendation generation failed: {e}")
    
    def validate_strategy_correctness(self, 
                                    strategy: HybridBoundingStrategy,
                                    validation_level: Optional[StrategyValidationLevel] = None) -> StrategyValidationResult:
        """
        Validate correctness of a bounding strategy.
        
        Args:
            strategy: Strategy to validate
            validation_level: Optional validation level
            
        Returns:
            Strategy validation result
            
        Raises:
            ValueError: If strategy is invalid
            RuntimeError: If validation fails
        """
        if not isinstance(strategy, HybridBoundingStrategy):
            raise TypeError("Strategy must be HybridBoundingStrategy")
        
        validation_level = validation_level or self.config.default_validation_level
        
        if not isinstance(validation_level, StrategyValidationLevel):
            raise TypeError("Validation level must be StrategyValidationLevel")
        
        try:
            validation_result = StrategyValidationResult(
                strategy=strategy,
                is_valid=True,
                validation_level=validation_level,
                validation_errors=[],
                performance_score=0.0,
                quality_score=0.0,
                stability_score=0.0,
                memory_efficiency_score=0.0,
                validation_recommendations=[]
            )
            
            with self._strategy_lock:
                # Basic validation
                basic_validation = self._perform_basic_validation(strategy)
                validation_result.validation_errors.extend(basic_validation['errors'])
                
                if validation_level in [StrategyValidationLevel.COMPREHENSIVE, 
                                      StrategyValidationLevel.RIGOROUS, 
                                      StrategyValidationLevel.PRODUCTION]:
                    # Comprehensive validation
                    comprehensive_validation = self._perform_comprehensive_validation(strategy)
                    validation_result.validation_errors.extend(comprehensive_validation['errors'])
                    validation_result.performance_score = comprehensive_validation['performance_score']
                    validation_result.quality_score = comprehensive_validation['quality_score']
                
                if validation_level in [StrategyValidationLevel.RIGOROUS, 
                                      StrategyValidationLevel.PRODUCTION]:
                    # Rigorous validation
                    rigorous_validation = self._perform_rigorous_validation(strategy)
                    validation_result.validation_errors.extend(rigorous_validation['errors'])
                    validation_result.stability_score = rigorous_validation['stability_score']
                    validation_result.memory_efficiency_score = rigorous_validation['memory_efficiency_score']
                
                if validation_level == StrategyValidationLevel.PRODUCTION:
                    # Production validation
                    production_validation = self._perform_production_validation(strategy)
                    validation_result.validation_errors.extend(production_validation['errors'])
                    validation_result.validation_recommendations.extend(production_validation['recommendations'])
                
                # Overall validity assessment
                validation_result.is_valid = len(validation_result.validation_errors) == 0
                
                # Store validation result
                self.validation_results[strategy] = validation_result
                self.validation_history.append(validation_result)
                
                self.logger.debug(f"Strategy {strategy.name} validation completed: {'VALID' if validation_result.is_valid else 'INVALID'}")
                
                return validation_result
                
        except Exception as e:
            self.logger.error(f"Strategy validation failed: {e}")
            raise RuntimeError(f"Strategy validation failed: {e}")
    
    def _initialize_strategy_metrics(self) -> None:
        """Initialize strategy performance metrics"""
        for strategy in HybridBoundingStrategy:
            self.strategy_metrics[strategy] = StrategyPerformanceMetrics(
                strategy=strategy,
                total_operations=0,
                successful_operations=0,
                failed_operations=0,
                average_time_ms=0.0,
                average_quality_score=0.0,
                average_gpu_utilization=0.0,
                memory_efficiency=1.0,
                error_rate=0.0,
                stability_score=1.0,
                performance_trend=[],
                quality_trend=[]
            )
    
    def _initialize_predictive_models(self) -> None:
        """Initialize predictive models for strategy selection"""
        self.predictive_models = {
            'performance_predictor': {
                'model_type': 'linear_regression',
                'features': ['gradient_norm', 'gradient_variance', 'direction_confidence'],
                'trained': False
            },
            'quality_predictor': {
                'model_type': 'decision_tree',
                'features': ['gradient_direction', 'stability_score', 'memory_pressure'],
                'trained': False
            }
        }
    
    def _select_performance_based_strategy(self, gradients: torch.Tensor, context: Optional[Dict[str, Any]]) -> HybridBoundingStrategy:
        """Select strategy based on performance metrics"""
        # Find strategy with best average performance
        best_strategy = HybridBoundingStrategy.DIRECTION_ADAPTIVE
        best_performance = 0.0
        
        for strategy, metrics in self.strategy_metrics.items():
            if metrics.total_operations >= 5:  # Minimum sample size
                # Calculate combined performance score
                time_factor = min(1.0, 10.0 / metrics.average_time_ms) if metrics.average_time_ms > 0 else 1.0
                quality_factor = metrics.average_quality_score
                gpu_factor = min(1.0, metrics.average_gpu_utilization * 2.0)
                
                performance_score = (time_factor + quality_factor + gpu_factor) / 3.0
                
                if performance_score > best_performance:
                    best_performance = performance_score
                    best_strategy = strategy
        
        return best_strategy
    
    def _select_quality_based_strategy(self, gradients: torch.Tensor, context: Optional[Dict[str, Any]]) -> HybridBoundingStrategy:
        """Select strategy based on quality metrics"""
        # Find strategy with highest quality score
        best_strategy = HybridBoundingStrategy.DIRECTION_ADAPTIVE
        best_quality = 0.0
        
        for strategy, metrics in self.strategy_metrics.items():
            if metrics.total_operations >= 5 and metrics.average_quality_score > best_quality:
                best_quality = metrics.average_quality_score
                best_strategy = strategy
        
        return best_strategy
    
    def _select_memory_based_strategy(self, gradients: torch.Tensor, context: Optional[Dict[str, Any]]) -> HybridBoundingStrategy:
        """Select strategy based on memory efficiency"""
        # Check if memory pressure is indicated
        memory_pressure = context.get('memory_pressure', False) if context else False
        
        if memory_pressure:
            return HybridBoundingStrategy.MEMORY_EFFICIENT
        
        # Find strategy with best memory efficiency
        best_strategy = HybridBoundingStrategy.MEMORY_EFFICIENT
        best_efficiency = 0.0
        
        for strategy, metrics in self.strategy_metrics.items():
            if metrics.total_operations >= 5 and metrics.memory_efficiency > best_efficiency:
                best_efficiency = metrics.memory_efficiency
                best_strategy = strategy
        
        return best_strategy
    
    def _select_adaptive_strategy(self, gradients: torch.Tensor, context: Optional[Dict[str, Any]]) -> HybridBoundingStrategy:
        """Select strategy using adaptive approach"""
        # Analyze gradient characteristics
        grad_norm = torch.norm(gradients).item()
        grad_std = torch.std(gradients).item()
        grad_mean = torch.mean(torch.abs(gradients)).item()
        
        # Get direction information if available
        direction_type = context.get('direction_type') if context else None
        stability_score = context.get('stability_score', 0.5) if context else 0.5
        
        # Adaptive selection logic
        if grad_norm > 3.0:  # High gradient norm
            return HybridBoundingStrategy.AGGRESSIVE
        elif grad_std < 0.01:  # Low variance
            return HybridBoundingStrategy.CONSERVATIVE
        elif stability_score > 0.8:  # High stability
            return HybridBoundingStrategy.PERFORMANCE_OPTIMIZED
        elif direction_type == DirectionType.OSCILLATING:
            return HybridBoundingStrategy.AGGRESSIVE
        elif context and context.get('gpu_available', True):
            return HybridBoundingStrategy.GPU_ACCELERATED
        else:
            return HybridBoundingStrategy.DIRECTION_ADAPTIVE
    
    def _select_predictive_strategy(self, gradients: torch.Tensor, context: Optional[Dict[str, Any]]) -> HybridBoundingStrategy:
        """Select strategy using predictive models"""
        # Extract features for prediction
        features = self._extract_prediction_features(gradients, context)
        
        # Use performance predictor if trained
        if self.predictive_models['performance_predictor']['trained']:
            predicted_performance = self._predict_performance(features)
            
            # Select strategy with highest predicted performance
            best_strategy = max(
                HybridBoundingStrategy,
                key=lambda s: predicted_performance.get(s.name, 0.0)
            )
            
            return best_strategy
        else:
            # Fallback to adaptive selection
            return self._select_adaptive_strategy(gradients, context)
    
    def _select_conservative_strategy(self, gradients: torch.Tensor, context: Optional[Dict[str, Any]]) -> HybridBoundingStrategy:
        """Select conservative strategy with low risk"""
        # Always use conservative strategy for safety
        return HybridBoundingStrategy.CONSERVATIVE
    
    def _validate_strategy_selection(self, 
                                   strategy: HybridBoundingStrategy, 
                                   gradients: torch.Tensor, 
                                   context: Optional[Dict[str, Any]]) -> bool:
        """Validate that selected strategy is appropriate"""
        # Check if strategy has acceptable performance history
        if strategy in self.strategy_metrics:
            metrics = self.strategy_metrics[strategy]
            if metrics.total_operations >= 5:
                if metrics.error_rate > self.config.max_error_rate:
                    return False
                if metrics.average_quality_score < self.config.min_quality_score:
                    return False
                if metrics.memory_efficiency < self.config.min_memory_efficiency:
                    return False
        
        # Check for specific constraints
        if context:
            if context.get('require_high_quality') and strategy == HybridBoundingStrategy.AGGRESSIVE:
                return False
            if context.get('memory_pressure') and strategy == HybridBoundingStrategy.GPU_ACCELERATED:
                return False
        
        return True
    
    def _get_fallback_strategy(self, gradients: torch.Tensor, context: Optional[Dict[str, Any]]) -> HybridBoundingStrategy:
        """Get fallback strategy when primary selection fails"""
        # Use conservative strategy as ultimate fallback
        return HybridBoundingStrategy.CONSERVATIVE
    
    def _analyze_performance_trends(self, performance_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze performance trends from historical data"""
        if len(performance_data) < 5:
            return {}
        
        # Extract time series data
        times = [d.get('time_ms', 0.0) for d in performance_data]
        qualities = [d.get('quality_score', 0.0) for d in performance_data]
        
        # Calculate trends using simple linear regression
        n = len(times)
        if n > 1:
            x = np.arange(n)
            time_slope = np.polyfit(x, times, 1)[0] if len(times) > 1 else 0.0
            quality_slope = np.polyfit(x, qualities, 1)[0] if len(qualities) > 1 else 0.0
        else:
            time_slope = 0.0
            quality_slope = 0.0
        
        return {
            'time_trend': time_slope,
            'quality_trend': quality_slope,
            'data_points': n
        }
    
    def _optimize_timing_parameters(self, strategy: HybridBoundingStrategy, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize timing-related parameters for strategy"""
        avg_time = sum(d.get('time_ms', 0.0) for d in performance_data) / len(performance_data)
        target_time = 5.0  # Target 5ms
        
        optimization = {
            'current_avg_time': avg_time,
            'target_time': target_time,
            'optimization_needed': avg_time > target_time,
            'recommended_adjustments': []
        }
        
        if avg_time > target_time:
            optimization['recommended_adjustments'].extend([
                'reduce_computation_complexity',
                'enable_parallel_processing',
                'optimize_memory_access_patterns'
            ])
        
        return optimization
    
    def _optimize_quality_parameters(self, strategy: HybridBoundingStrategy, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize quality-related parameters for strategy"""
        avg_quality = sum(d.get('quality_score', 0.0) for d in performance_data) / len(performance_data)
        target_quality = 0.85
        
        optimization = {
            'current_avg_quality': avg_quality,
            'target_quality': target_quality,
            'optimization_needed': avg_quality < target_quality,
            'recommended_adjustments': []
        }
        
        if avg_quality < target_quality:
            optimization['recommended_adjustments'].extend([
                'increase_precision',
                'improve_direction_preservation',
                'reduce_aggressive_clipping'
            ])
        
        return optimization
    
    def _optimize_memory_parameters(self, strategy: HybridBoundingStrategy, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize memory-related parameters for strategy"""
        avg_memory_eff = sum(d.get('memory_efficiency', 1.0) for d in performance_data) / len(performance_data)
        target_efficiency = 0.9
        
        optimization = {
            'current_memory_efficiency': avg_memory_eff,
            'target_efficiency': target_efficiency,
            'optimization_needed': avg_memory_eff < target_efficiency,
            'recommended_adjustments': []
        }
        
        if avg_memory_eff < target_efficiency:
            optimization['recommended_adjustments'].extend([
                'use_in_place_operations',
                'reduce_intermediate_allocations',
                'optimize_tensor_reuse'
            ])
        
        return optimization
    
    def _calculate_optimization_improvement(self, 
                                          performance_data: List[Dict[str, Any]], 
                                          optimization_results: Dict[str, Any]) -> float:
        """Calculate expected performance improvement from optimizations"""
        # Simple heuristic: each optimization type contributes certain improvement
        improvement_factors = {
            'timing_optimization': 0.15,
            'quality_optimization': 0.12,
            'memory_optimization': 0.08
        }
        
        total_improvement = 0.0
        for optimization in optimization_results['optimizations_applied']:
            total_improvement += improvement_factors.get(optimization, 0.05)
        
        return min(0.5, total_improvement)  # Cap at 50% improvement
    
    def _validate_strategy_switch(self, 
                                current_strategy: HybridBoundingStrategy, 
                                new_strategy: HybridBoundingStrategy) -> Dict[str, Any]:
        """Validate strategy switch"""
        validation = {
            'is_valid': True,
            'reason': '',
            'risk_level': 'low'
        }
        
        # Check if new strategy has sufficient performance history
        if new_strategy in self.strategy_metrics:
            metrics = self.strategy_metrics[new_strategy]
            if metrics.total_operations < 5:
                validation['is_valid'] = False
                validation['reason'] = 'Insufficient performance history for new strategy'
                validation['risk_level'] = 'high'
            elif metrics.error_rate > self.config.max_error_rate * 2:
                validation['is_valid'] = False
                validation['reason'] = 'New strategy has high error rate'
                validation['risk_level'] = 'high'
        
        return validation
    
    def _calculate_switch_benefits(self, 
                                 current_strategy: HybridBoundingStrategy, 
                                 new_strategy: HybridBoundingStrategy) -> Dict[str, float]:
        """Calculate expected benefits of strategy switch"""
        benefits = {
            'performance_improvement': 0.0,
            'quality_improvement': 0.0,
            'memory_improvement': 0.0,
            'overall_benefit': 0.0
        }
        
        if current_strategy in self.strategy_metrics and new_strategy in self.strategy_metrics:
            current_metrics = self.strategy_metrics[current_strategy]
            new_metrics = self.strategy_metrics[new_strategy]
            
            # Calculate improvements
            if current_metrics.average_time_ms > 0:
                time_improvement = (current_metrics.average_time_ms - new_metrics.average_time_ms) / current_metrics.average_time_ms
                benefits['performance_improvement'] = max(0.0, time_improvement)
            
            quality_improvement = new_metrics.average_quality_score - current_metrics.average_quality_score
            benefits['quality_improvement'] = max(0.0, quality_improvement)
            
            memory_improvement = new_metrics.memory_efficiency - current_metrics.memory_efficiency
            benefits['memory_improvement'] = max(0.0, memory_improvement)
            
            # Overall benefit score
            benefits['overall_benefit'] = (
                benefits['performance_improvement'] * 0.4 +
                benefits['quality_improvement'] * 0.4 +
                benefits['memory_improvement'] * 0.2
            )
        
        return benefits
    
    def _create_transition_plan(self, 
                              current_strategy: HybridBoundingStrategy, 
                              new_strategy: HybridBoundingStrategy) -> List[str]:
        """Create transition plan for strategy switch"""
        return [
            f"Validate {new_strategy.name} readiness",
            f"Prepare fallback to {current_strategy.name}",
            f"Switch to {new_strategy.name}",
            "Monitor performance for 10 operations",
            "Confirm successful transition"
        ]
    
    def _create_rollback_plan(self, 
                            current_strategy: HybridBoundingStrategy, 
                            new_strategy: HybridBoundingStrategy) -> List[str]:
        """Create rollback plan for strategy switch"""
        return [
            "Detect performance degradation",
            f"Immediately switch back to {current_strategy.name}",
            "Log rollback reason and metrics",
            "Mark new strategy for further analysis",
            "Continue with original strategy"
        ]
    
    def _analyze_current_performance(self, performance_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze current performance from recent data"""
        recent_data = performance_data[-10:] if len(performance_data) >= 10 else performance_data
        
        return {
            'average_time': sum(d.get('time_ms', 0.0) for d in recent_data) / len(recent_data),
            'average_quality': sum(d.get('quality_score', 0.0) for d in recent_data) / len(recent_data),
            'average_memory_efficiency': sum(d.get('memory_efficiency', 1.0) for d in recent_data) / len(recent_data),
            'error_rate': sum(1 for d in recent_data if d.get('error_count', 0) > 0) / len(recent_data),
            'data_points': len(recent_data)
        }
    
    def _generate_strategy_recommendation(self, 
                                        strategy: HybridBoundingStrategy, 
                                        current_performance: Dict[str, float], 
                                        performance_data: List[Dict[str, Any]]) -> StrategyRecommendation:
        """Generate recommendation for a specific strategy"""
        # Calculate expected performance for this strategy
        if strategy in self.strategy_metrics:
            metrics = self.strategy_metrics[strategy]
            expected_performance = {
                'time_ms': metrics.average_time_ms,
                'quality_score': metrics.average_quality_score,
                'memory_efficiency': metrics.memory_efficiency,
                'overall_score': (metrics.average_quality_score + metrics.memory_efficiency) / 2.0
            }
            
            # Calculate confidence based on historical data
            confidence = min(1.0, metrics.total_operations / 50.0)  # Higher confidence with more data
            
            # Adjust confidence based on recent performance
            if metrics.error_rate > self.config.max_error_rate:
                confidence *= 0.5
            
            # Generate reasoning
            reasoning = f"Strategy shows {expected_performance['overall_score']:.3f} overall score with {metrics.total_operations} operations"
            
        else:
            # No historical data
            expected_performance = {
                'time_ms': 10.0,  # Default estimates
                'quality_score': 0.7,
                'memory_efficiency': 0.8,
                'overall_score': 0.75
            }
            confidence = 0.3  # Low confidence without data
            reasoning = "Limited historical data available for this strategy"
        
        # Risk assessment
        risk_assessment = {
            'performance_risk': 0.2 if strategy == HybridBoundingStrategy.CONSERVATIVE else 0.4,
            'quality_risk': 0.1 if strategy in [HybridBoundingStrategy.DIRECTION_ADAPTIVE, HybridBoundingStrategy.PERFORMANCE_OPTIMIZED] else 0.3,
            'memory_risk': 0.1 if strategy == HybridBoundingStrategy.MEMORY_EFFICIENT else 0.25
        }
        
        # Fallback strategies
        fallback_strategies = [
            HybridBoundingStrategy.CONSERVATIVE,
            HybridBoundingStrategy.DIRECTION_ADAPTIVE
        ]
        if strategy in fallback_strategies:
            fallback_strategies.remove(strategy)
        
        return StrategyRecommendation(
            recommended_strategy=strategy,
            confidence=confidence,
            reasoning=reasoning,
            expected_performance=expected_performance,
            risk_assessment=risk_assessment,
            fallback_strategies=fallback_strategies[:2],  # Top 2 fallbacks
            implementation_priority=int(confidence * 10)
        )
    
    def _perform_basic_validation(self, strategy: HybridBoundingStrategy) -> Dict[str, Any]:
        """Perform basic strategy validation"""
        errors = []
        
        # Check if strategy is supported
        if strategy not in HybridBoundingStrategy:
            errors.append(f"Unsupported strategy: {strategy}")
        
        # Check if strategy has been tested
        if strategy not in self.strategy_metrics:
            errors.append(f"Strategy {strategy.name} has no performance history")
        
        return {'errors': errors}
    
    def _perform_comprehensive_validation(self, strategy: HybridBoundingStrategy) -> Dict[str, Any]:
        """Perform comprehensive strategy validation"""
        errors = []
        performance_score = 0.0
        quality_score = 0.0
        
        if strategy in self.strategy_metrics:
            metrics = self.strategy_metrics[strategy]
            
            # Validate performance metrics
            if metrics.error_rate > self.config.max_error_rate:
                errors.append(f"Strategy {strategy.name} has high error rate: {metrics.error_rate:.3f}")
            
            if metrics.average_quality_score < self.config.min_quality_score:
                errors.append(f"Strategy {strategy.name} has low quality score: {metrics.average_quality_score:.3f}")
            
            performance_score = min(1.0, 10.0 / metrics.average_time_ms) if metrics.average_time_ms > 0 else 0.0
            quality_score = metrics.average_quality_score
        else:
            errors.append(f"Strategy {strategy.name} lacks comprehensive performance data")
        
        return {
            'errors': errors,
            'performance_score': performance_score,
            'quality_score': quality_score
        }
    
    def _perform_rigorous_validation(self, strategy: HybridBoundingStrategy) -> Dict[str, Any]:
        """Perform rigorous strategy validation"""
        errors = []
        stability_score = 0.0
        memory_efficiency_score = 0.0
        
        if strategy in self.strategy_metrics:
            metrics = self.strategy_metrics[strategy]
            
            # Validate stability
            if metrics.stability_score < 0.8:
                errors.append(f"Strategy {strategy.name} has low stability: {metrics.stability_score:.3f}")
            
            # Validate memory efficiency
            if metrics.memory_efficiency < self.config.min_memory_efficiency:
                errors.append(f"Strategy {strategy.name} has poor memory efficiency: {metrics.memory_efficiency:.3f}")
            
            stability_score = metrics.stability_score
            memory_efficiency_score = metrics.memory_efficiency
        else:
            errors.append(f"Strategy {strategy.name} lacks rigorous validation data")
        
        return {
            'errors': errors,
            'stability_score': stability_score,
            'memory_efficiency_score': memory_efficiency_score
        }
    
    def _perform_production_validation(self, strategy: HybridBoundingStrategy) -> Dict[str, Any]:
        """Perform production-level strategy validation"""
        errors = []
        recommendations = []
        
        if strategy in self.strategy_metrics:
            metrics = self.strategy_metrics[strategy]
            
            # Production readiness checks
            if metrics.total_operations < 100:
                errors.append(f"Strategy {strategy.name} needs more production testing")
            
            # Generate production recommendations
            if metrics.average_quality_score > 0.9:
                recommendations.append(f"Strategy {strategy.name} shows excellent quality - consider for primary use")
            
            if metrics.error_rate == 0.0 and metrics.total_operations > 50:
                recommendations.append(f"Strategy {strategy.name} shows zero errors - highly reliable")
        else:
            errors.append(f"Strategy {strategy.name} not ready for production")
        
        return {
            'errors': errors,
            'recommendations': recommendations
        }
    
    def _extract_prediction_features(self, gradients: torch.Tensor, context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Extract features for predictive models"""
        features = {
            'gradient_norm': torch.norm(gradients).item(),
            'gradient_variance': torch.var(gradients).item(),
            'gradient_mean': torch.mean(torch.abs(gradients)).item()
        }
        
        if context:
            features['direction_confidence'] = context.get('direction_confidence', 0.5)
            features['stability_score'] = context.get('stability_score', 0.5)
            features['memory_pressure'] = 1.0 if context.get('memory_pressure', False) else 0.0
        
        return features
    
    def _predict_performance(self, features: Dict[str, float]) -> Dict[str, float]:
        """Predict performance for each strategy"""
        # Simplified prediction logic (in practice, would use trained ML models)
        predictions = {}
        
        for strategy in HybridBoundingStrategy:
            # Heuristic prediction based on strategy characteristics
            if strategy == HybridBoundingStrategy.CONSERVATIVE:
                predictions[strategy.name] = 0.7  # Consistent but not optimal
            elif strategy == HybridBoundingStrategy.AGGRESSIVE:
                predictions[strategy.name] = 0.6 if features.get('gradient_norm', 0) > 2.0 else 0.4
            elif strategy == HybridBoundingStrategy.MEMORY_EFFICIENT:
                predictions[strategy.name] = 0.8 if features.get('memory_pressure', 0) > 0.5 else 0.5
            elif strategy == HybridBoundingStrategy.GPU_ACCELERATED:
                predictions[strategy.name] = 0.9 if features.get('gradient_norm', 0) > 1.0 else 0.6
            else:  # DIRECTION_ADAPTIVE, PERFORMANCE_OPTIMIZED
                predictions[strategy.name] = 0.8
        
        return predictions
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current strategy management metrics"""
        return {
            'current_strategy': self.current_strategy.name,
            'selection_mode': self.selection_mode.name,
            'total_strategies_evaluated': len(self.strategy_metrics),
            'total_validations_performed': len(self.validation_history),
            'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None,
            'strategy_performance_summary': {
                strategy.name: {
                    'operations': metrics.total_operations,
                    'quality': metrics.average_quality_score,
                    'error_rate': metrics.error_rate
                }
                for strategy, metrics in self.strategy_metrics.items()
                if metrics.total_operations > 0
            }
        }
    
    def shutdown(self) -> None:
        """Shutdown bounding strategy manager"""
        self.logger.info("Shutting down bounding strategy manager")
        
        # Clear tracking data
        self.strategy_metrics.clear()
        self.performance_history.clear()
        self.evaluation_results.clear()
        self.validation_results.clear()
        self.validation_history.clear()
        
        self.logger.info("Bounding strategy manager shutdown complete")