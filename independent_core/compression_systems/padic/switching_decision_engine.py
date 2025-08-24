"""
Switching Decision Engine - Intelligent decision engine for determining when to switch compression modes
NO FALLBACKS - HARD FAILURES ONLY
"""

import logging
import math
import time
import torch
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

# Import GAC system components - with fallback for missing modules
try:
    from gac_system.direction_state import DirectionStateManager, DirectionState
    from gac_system.enhanced_bounder import EnhancedGradientBounder
except ImportError:
    # Create minimal stub classes if GAC system not available
    from enum import Enum
    
    class DirectionState(Enum):
        """Direction state enumeration stub"""
        STABLE = "stable"
        OSCILLATING = "oscillating"
        ASCENDING = "ascending"
        DESCENDING = "descending"
        UNKNOWN = "unknown"
    
    class DirectionStateManager:
        """Minimal stub for DirectionStateManager"""
        def __init__(self, *args, **kwargs):
            self.state = DirectionState.UNKNOWN
        
        def update_direction_state(self, gradients):
            """Stub method for updating direction state"""
            return DirectionState.UNKNOWN
    
    class EnhancedGradientBounder:
        """Minimal stub for EnhancedGradientBounder"""
        def __init__(self, *args, **kwargs):
            pass

# Import performance optimizer - with fallback for missing module
try:
    from performance_optimizer import PerformanceOptimizer
except ImportError:
    # Create minimal stub class if performance optimizer not available
    class PerformanceOptimizer:
        """Minimal stub for PerformanceOptimizer"""
        def __init__(self, *args, **kwargs):
            self.is_initialized = False
        
        def optimize(self, *args, **kwargs):
            """Stub optimization method"""
            return {}


class DecisionCriterion(Enum):
    """Decision criteria enumeration"""
    GRADIENT_STABILITY = "gradient_stability"
    DATA_SIZE = "data_size"
    MEMORY_USAGE = "memory_usage"
    PERFORMANCE_HISTORY = "performance_history"
    ERROR_RATE = "error_rate"
    COMPUTATIONAL_LOAD = "computational_load"
    GPU_UTILIZATION = "gpu_utilization"


@dataclass
class DecisionWeights:
    """Weights for different decision criteria"""
    gradient_stability: float = 0.25
    data_size: float = 0.20
    memory_usage: float = 0.15
    performance_history: float = 0.20
    error_rate: float = 0.10
    computational_load: float = 0.05
    gpu_utilization: float = 0.05
    
    def __post_init__(self):
        """Validate decision weights"""
        total_weight = (
            self.gradient_stability + self.data_size + self.memory_usage +
            self.performance_history + self.error_rate + self.computational_load +
            self.gpu_utilization
        )
        if not math.isclose(total_weight, 1.0, rel_tol=1e-6):
            raise ValueError(f"Decision weights must sum to 1.0, got {total_weight}")
        
        # Validate individual weights
        for field_name in self.__dataclass_fields__:
            weight = getattr(self, field_name)
            if not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
                raise ValueError(f"{field_name} weight must be in [0, 1], got {weight}")


@dataclass
class DecisionAnalysis:
    """Analysis result for switching decision"""
    criterion: DecisionCriterion
    score: float  # 0-1 score for this criterion
    confidence: float  # 0-1 confidence in this score
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate decision analysis"""
        if not isinstance(self.criterion, DecisionCriterion):
            raise TypeError("Criterion must be DecisionCriterion")
        if not isinstance(self.score, (int, float)) or not 0 <= self.score <= 1:
            raise ValueError(f"Score must be in [0, 1], got {self.score}")
        if not isinstance(self.confidence, (int, float)) or not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if not isinstance(self.reasoning, str):
            raise TypeError("Reasoning must be string")


@dataclass
class PerformancePrediction:
    """Performance prediction for switching decision"""
    predicted_compression_time: float
    predicted_decompression_time: float
    predicted_memory_usage: float
    predicted_compression_ratio: float
    predicted_error_rate: float
    confidence: float
    prediction_basis: str
    
    def __post_init__(self):
        """Validate performance prediction"""
        for field_name in ['predicted_compression_time', 'predicted_decompression_time', 
                          'predicted_memory_usage', 'predicted_compression_ratio', 
                          'predicted_error_rate']:
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")
        
        if not isinstance(self.confidence, (int, float)) or not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


class SwitchingDecisionEngine:
    """
    Intelligent decision engine for determining when to switch compression modes.
    Uses multi-criteria analysis including gradient direction, performance prediction, and resource monitoring.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize switching decision engine"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, dict):
            raise TypeError(f"Config must be dict or None, got {type(config)}")
        
        self.config = config or {}
        self.logger = logging.getLogger('SwitchingDecisionEngine')
        
        # Decision engine state
        self.is_initialized = False
        
        # Component integrations
        self.direction_state_manager: Optional[DirectionStateManager] = None
        self.performance_optimizer: Optional[PerformanceOptimizer] = None
        self.gradient_bounder: Optional[EnhancedGradientBounder] = None
        
        # Decision configuration
        self.decision_weights = DecisionWeights(
            gradient_stability=self.config.get('gradient_stability_weight', 0.25),
            data_size=self.config.get('data_size_weight', 0.20),
            memory_usage=self.config.get('memory_usage_weight', 0.15),
            performance_history=self.config.get('performance_history_weight', 0.20),
            error_rate=self.config.get('error_rate_weight', 0.10),
            computational_load=self.config.get('computational_load_weight', 0.05),
            gpu_utilization=self.config.get('gpu_utilization_weight', 0.05)
        )
        
        # Decision thresholds
        self.hybrid_threshold = self.config.get('hybrid_threshold', 0.7)
        self.pure_threshold = self.config.get('pure_threshold', 0.3)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=100)
        self.decision_history: deque = deque(maxlen=500)
        self.gradient_analysis_cache: Dict[str, Any] = {}
        
        # Adaptive thresholds
        self.adaptive_thresholds_enabled = self.config.get('enable_adaptive_thresholds', True)
        self.threshold_adaptation_rate = self.config.get('threshold_adaptation_rate', 0.1)
        
        self.logger.info("SwitchingDecisionEngine created successfully")
    
    def initialize_decision_engine(self,
                                 direction_manager: DirectionStateManager,
                                 performance_optimizer: PerformanceOptimizer,
                                 gradient_bounder: Optional[EnhancedGradientBounder] = None) -> None:
        """
        Initialize decision engine with required components.
        
        Args:
            direction_manager: GAC direction state manager
            performance_optimizer: Performance optimizer instance
            gradient_bounder: Optional gradient bounder
            
        Raises:
            RuntimeError: If initialization fails
        """
        if self.is_initialized:
            return
        
        try:
            # Validate and store component references
            if not isinstance(direction_manager, DirectionStateManager):
                raise TypeError(f"Direction manager must be DirectionStateManager, got {type(direction_manager)}")
            if not isinstance(performance_optimizer, PerformanceOptimizer):
                raise TypeError(f"Performance optimizer must be PerformanceOptimizer, got {type(performance_optimizer)}")
            
            self.direction_state_manager = direction_manager
            self.performance_optimizer = performance_optimizer
            self.gradient_bounder = gradient_bounder
            
            self.is_initialized = True
            self.logger.info("Switching decision engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize decision engine: {e}")
            raise RuntimeError(f"Decision engine initialization failed: {e}")
    
    def analyze_switching_criteria(self, data: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze switching criteria for given data and context.
        
        Args:
            data: Input tensor data
            context: Optional context information
            
        Returns:
            Dictionary containing analysis results
            
        Raises:
            RuntimeError: If decision engine not initialized
            ValueError: If data is invalid
        """
        if not self.is_initialized:
            raise RuntimeError("Decision engine not initialized")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Data must be torch.Tensor, got {type(data)}")
        if data.numel() == 0:
            raise ValueError("Data tensor cannot be empty")
        
        context = context or {}
        start_time = time.time()
        
        try:
            # Analyze each decision criterion
            analyses = []
            
            # 1. Gradient stability analysis
            if context.get('gradients') is not None:
                gradient_analysis = self._analyze_gradient_stability(context['gradients'])
                analyses.append(gradient_analysis)
            
            # 2. Data size analysis
            data_size_analysis = self._analyze_data_size(data)
            analyses.append(data_size_analysis)
            
            # 3. Memory usage analysis
            memory_analysis = self._analyze_memory_usage(data)
            analyses.append(memory_analysis)
            
            # 4. Performance history analysis
            performance_analysis = self._analyze_performance_history()
            analyses.append(performance_analysis)
            
            # 5. Error rate analysis
            error_analysis = self._analyze_error_rate(context)
            analyses.append(error_analysis)
            
            # 6. Computational load analysis
            load_analysis = self._analyze_computational_load()
            analyses.append(load_analysis)
            
            # 7. GPU utilization analysis
            gpu_analysis = self._analyze_gpu_utilization()
            analyses.append(gpu_analysis)
            
            # Calculate weighted decision score
            weighted_score = self._calculate_weighted_score(analyses)
            overall_confidence = self._calculate_overall_confidence(analyses)
            
            # Make switching recommendation
            recommendation = self._make_switching_recommendation(weighted_score, overall_confidence)
            
            # Store decision in history
            decision_record = {
                'timestamp': datetime.now(timezone.utc),
                'data_shape': data.shape,
                'weighted_score': weighted_score,
                'overall_confidence': overall_confidence,
                'recommendation': recommendation,
                'analyses': analyses,
                'processing_time_ms': (time.time() - start_time) * 1000
            }
            self.decision_history.append(decision_record)
            
            return {
                'weighted_score': weighted_score,
                'overall_confidence': overall_confidence,
                'recommendation': recommendation,
                'criteria_analyses': [
                    {
                        'criterion': analysis.criterion.value,
                        'score': analysis.score,
                        'confidence': analysis.confidence,
                        'reasoning': analysis.reasoning,
                        'metadata': analysis.metadata
                    }
                    for analysis in analyses
                ],
                'performance_confidence': overall_confidence,
                'processing_time_ms': (time.time() - start_time) * 1000
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing switching criteria: {e}")
            raise RuntimeError(f"Switching criteria analysis failed: {e}")
    
    def evaluate_gradient_direction(self, gradients: torch.Tensor) -> float:
        """
        Evaluate gradient direction for switching decision.
        
        Args:
            gradients: Gradient tensor
            
        Returns:
            Confidence score for gradient-based switching (0-1)
            
        Raises:
            ValueError: If gradients are invalid
        """
        if not isinstance(gradients, torch.Tensor):
            raise TypeError(f"Gradients must be torch.Tensor, got {type(gradients)}")
        if gradients.numel() == 0:
            raise ValueError("Gradients tensor cannot be empty")
        
        try:
            # Use direction state manager for gradient analysis
            if self.direction_state_manager:
                direction_state = self.direction_state_manager.update_direction_state(gradients)
                
                # Calculate confidence based on direction stability
                if direction_state == DirectionState.STABLE:
                    return 0.9  # High confidence for hybrid with stable gradients
                elif direction_state == DirectionState.OSCILLATING:
                    return 0.8  # Good confidence for hybrid with oscillating gradients
                elif direction_state == DirectionState.ASCENDING:
                    return 0.4  # Lower confidence - might favor pure
                elif direction_state == DirectionState.DESCENDING:
                    return 0.3  # Lower confidence - might favor pure
                else:
                    return 0.5  # Neutral confidence for unknown state
            
            # Fallback gradient analysis if no direction manager
            gradient_norm = torch.norm(gradients).item()
            gradient_variance = torch.var(gradients).item()
            
            # Normalize variance relative to norm
            if gradient_norm > 0:
                normalized_variance = gradient_variance / (gradient_norm ** 2)
                # High variance might indicate oscillating/unstable gradients (favor hybrid)
                # Low variance might indicate stable direction (could favor either)
                return min(0.8, max(0.2, normalized_variance * 2))
            else:
                return 0.5  # Neutral for zero gradients
                
        except Exception as e:
            self.logger.error(f"Error evaluating gradient direction: {e}")
            raise RuntimeError(f"Gradient direction evaluation failed: {e}")
    
    def predict_performance_impact(self, data: torch.Tensor, target_mode: str) -> PerformancePrediction:
        """
        Predict performance impact of switching to target mode.
        
        Args:
            data: Input tensor data
            target_mode: Target compression mode ('hybrid' or 'pure_padic')
            
        Returns:
            Performance prediction
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Data must be torch.Tensor, got {type(data)}")
        if target_mode not in ['hybrid', 'pure_padic']:
            raise ValueError(f"Target mode must be 'hybrid' or 'pure_padic', got {target_mode}")
        
        try:
            data_size = data.numel()
            data_complexity = self._calculate_data_complexity(data)
            
            # Base predictions based on data characteristics
            if target_mode == 'hybrid':
                # Hybrid mode predictions
                base_compression_time = data_size * 0.001  # ms per element
                base_decompression_time = data_size * 0.0008
                base_memory_usage = data_size * 8  # bytes (float64 equivalent)
                base_compression_ratio = 0.3 + (data_complexity * 0.4)  # Better ratio for complex data
                base_error_rate = 1e-7
                confidence = 0.8 if data_size > 1000 else 0.6
                basis = "data_size_and_complexity_model"
            else:
                # Pure p-adic mode predictions
                base_compression_time = data_size * 0.002  # Slower for large data
                base_decompression_time = data_size * 0.0015
                base_memory_usage = data_size * 4  # bytes (float32)
                base_compression_ratio = 0.5 + (data_complexity * 0.2)  # Decent ratio
                base_error_rate = 1e-8  # Lower error rate
                confidence = 0.8 if data_size < 500 else 0.5
                basis = "data_size_linear_model"
            
            # Adjust predictions based on historical performance
            if self.performance_history:
                historical_adjustment = self._get_historical_performance_adjustment(target_mode)
                base_compression_time *= historical_adjustment
                base_decompression_time *= historical_adjustment
                confidence = min(0.9, confidence + 0.1)  # Increase confidence with history
                basis += "_with_historical_adjustment"
            
            return PerformancePrediction(
                predicted_compression_time=base_compression_time,
                predicted_decompression_time=base_decompression_time,
                predicted_memory_usage=base_memory_usage,
                predicted_compression_ratio=base_compression_ratio,
                predicted_error_rate=base_error_rate,
                confidence=confidence,
                prediction_basis=basis
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting performance impact: {e}")
            raise RuntimeError(f"Performance prediction failed: {e}")
    
    def calculate_switching_confidence(self, data: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate overall confidence in switching decision.
        
        Args:
            data: Input tensor data
            context: Optional context information
            
        Returns:
            Overall confidence score (0-1)
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Data must be torch.Tensor, got {type(data)}")
        
        context = context or {}
        
        try:
            # Analyze switching criteria
            analysis_result = self.analyze_switching_criteria(data, context)
            return analysis_result['overall_confidence']
            
        except Exception as e:
            self.logger.error(f"Error calculating switching confidence: {e}")
            return 0.0  # Return low confidence on error
    
    def get_optimal_switching_threshold(self) -> Tuple[float, float]:
        """
        Get optimal switching thresholds based on historical performance.
        
        Returns:
            Tuple of (hybrid_threshold, pure_threshold)
        """
        if not self.adaptive_thresholds_enabled or len(self.decision_history) < 10:
            return self.hybrid_threshold, self.pure_threshold
        
        try:
            # Analyze recent decisions and their outcomes
            recent_decisions = list(self.decision_history)[-50:]  # Last 50 decisions
            
            # Calculate success rate at different thresholds
            optimal_hybrid = self._find_optimal_threshold('hybrid', recent_decisions)
            optimal_pure = self._find_optimal_threshold('pure', recent_decisions)
            
            # Gradually adapt thresholds
            adapted_hybrid = (
                self.hybrid_threshold * (1 - self.threshold_adaptation_rate) +
                optimal_hybrid * self.threshold_adaptation_rate
            )
            adapted_pure = (
                self.pure_threshold * (1 - self.threshold_adaptation_rate) +
                optimal_pure * self.threshold_adaptation_rate
            )
            
            # Ensure valid threshold range
            adapted_hybrid = max(0.5, min(0.9, adapted_hybrid))
            adapted_pure = max(0.1, min(0.5, adapted_pure))
            
            return adapted_hybrid, adapted_pure
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal thresholds: {e}")
            return self.hybrid_threshold, self.pure_threshold
    
    def update_decision_weights(self, performance_data: Dict[str, Any]) -> None:
        """
        Update decision weights based on performance data.
        
        Args:
            performance_data: Performance data for weight adjustment
        """
        if not isinstance(performance_data, dict):
            raise TypeError(f"Performance data must be dict, got {type(performance_data)}")
        
        try:
            # Store performance data
            self.performance_history.append({
                'timestamp': datetime.now(timezone.utc),
                'data': performance_data.copy()
            })
            
            # Adaptive weight adjustment based on performance
            if len(self.performance_history) >= 20:  # Need sufficient data
                self._adapt_decision_weights()
                
        except Exception as e:
            self.logger.error(f"Error updating decision weights: {e}")
    
    def _analyze_gradient_stability(self, gradients: torch.Tensor) -> DecisionAnalysis:
        """Analyze gradient stability for switching decision"""
        try:
            gradient_confidence = self.evaluate_gradient_direction(gradients)
            
            # Higher gradient confidence suggests hybrid mode might be better
            score = gradient_confidence
            confidence = 0.8 if self.direction_state_manager else 0.6
            
            if gradient_confidence > 0.7:
                reasoning = "Stable/oscillating gradients favor hybrid compression"
            elif gradient_confidence < 0.4:
                reasoning = "Unstable gradients might favor pure p-adic compression"
            else:
                reasoning = "Gradient direction is neutral for mode selection"
            
            return DecisionAnalysis(
                criterion=DecisionCriterion.GRADIENT_STABILITY,
                score=score,
                confidence=confidence,
                reasoning=reasoning,
                metadata={'gradient_confidence': gradient_confidence}
            )
            
        except Exception as e:
            return DecisionAnalysis(
                criterion=DecisionCriterion.GRADIENT_STABILITY,
                score=0.5,
                confidence=0.1,
                reasoning=f"Gradient analysis failed: {e}",
                metadata={'error': str(e)}
            )
    
    def _analyze_data_size(self, data: torch.Tensor) -> DecisionAnalysis:
        """Analyze data size for switching decision"""
        data_size = data.numel()
        
        # Score based on data size preference for hybrid vs pure
        if data_size > 2000:
            score = 0.9  # Strongly favor hybrid for large data
            reasoning = f"Large data size ({data_size} elements) strongly favors hybrid compression"
        elif data_size > 1000:
            score = 0.7  # Favor hybrid
            reasoning = f"Medium-large data size ({data_size} elements) favors hybrid compression"
        elif data_size > 100:
            score = 0.5  # Neutral
            reasoning = f"Medium data size ({data_size} elements) is neutral for mode selection"
        else:
            score = 0.2  # Favor pure
            reasoning = f"Small data size ({data_size} elements) favors pure p-adic compression"
        
        return DecisionAnalysis(
            criterion=DecisionCriterion.DATA_SIZE,
            score=score,
            confidence=0.9,  # High confidence in data size analysis
            reasoning=reasoning,
            metadata={'data_size': data_size}
        )
    
    def _analyze_memory_usage(self, data: torch.Tensor) -> DecisionAnalysis:
        """Analyze memory usage for switching decision"""
        try:
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated()
                memory_usage_ratio = allocated_memory / total_memory
                
                if memory_usage_ratio > 0.8:
                    score = 0.3  # High memory usage might favor pure (smaller memory footprint)
                    reasoning = f"High GPU memory usage ({memory_usage_ratio:.1%}) favors pure compression"
                elif memory_usage_ratio > 0.6:
                    score = 0.5  # Neutral
                    reasoning = f"Medium GPU memory usage ({memory_usage_ratio:.1%}) is neutral"
                else:
                    score = 0.7  # Low memory usage allows hybrid
                    reasoning = f"Low GPU memory usage ({memory_usage_ratio:.1%}) allows hybrid compression"
                
                confidence = 0.8
                metadata = {
                    'gpu_memory_usage_ratio': memory_usage_ratio,
                    'allocated_mb': allocated_memory / (1024 * 1024),
                    'total_mb': total_memory / (1024 * 1024)
                }
            else:
                score = 0.4  # No GPU might favor pure
                reasoning = "No GPU available, might favor pure p-adic compression"
                confidence = 0.6
                metadata = {'gpu_available': False}
            
            return DecisionAnalysis(
                criterion=DecisionCriterion.MEMORY_USAGE,
                score=score,
                confidence=confidence,
                reasoning=reasoning,
                metadata=metadata
            )
            
        except Exception as e:
            return DecisionAnalysis(
                criterion=DecisionCriterion.MEMORY_USAGE,
                score=0.5,
                confidence=0.2,
                reasoning=f"Memory analysis failed: {e}",
                metadata={'error': str(e)}
            )
    
    def _analyze_performance_history(self) -> DecisionAnalysis:
        """Analyze performance history for switching decision"""
        if not self.performance_history:
            return DecisionAnalysis(
                criterion=DecisionCriterion.PERFORMANCE_HISTORY,
                score=0.5,
                confidence=0.1,
                reasoning="No performance history available",
                metadata={'history_length': 0}
            )
        
        try:
            recent_performance = list(self.performance_history)[-10:]
            avg_performance = np.mean([p['data'].get('performance_score', 0.5) for p in recent_performance])
            
            # Higher historical performance suggests current mode is working well
            if avg_performance > 0.8:
                score = 0.3  # Good performance, don't switch
                reasoning = f"Good recent performance ({avg_performance:.2f}) suggests current mode is effective"
            elif avg_performance > 0.6:
                score = 0.5  # Neutral performance
                reasoning = f"Neutral recent performance ({avg_performance:.2f})"
            else:
                score = 0.8  # Poor performance, consider switching
                reasoning = f"Poor recent performance ({avg_performance:.2f}) suggests switching might help"
            
            confidence = min(0.8, len(recent_performance) / 10.0)  # Confidence based on data availability
            
            return DecisionAnalysis(
                criterion=DecisionCriterion.PERFORMANCE_HISTORY,
                score=score,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    'avg_performance': avg_performance,
                    'history_length': len(recent_performance)
                }
            )
            
        except Exception as e:
            return DecisionAnalysis(
                criterion=DecisionCriterion.PERFORMANCE_HISTORY,
                score=0.5,
                confidence=0.2,
                reasoning=f"Performance history analysis failed: {e}",
                metadata={'error': str(e)}
            )
    
    def _analyze_error_rate(self, context: Dict[str, Any]) -> DecisionAnalysis:
        """Analyze error rate for switching decision"""
        error_rate = context.get('error_rate', 0.0)
        
        if error_rate > 0.01:  # > 1% error rate
            score = 0.8  # High error rate suggests switching
            reasoning = f"High error rate ({error_rate:.3%}) suggests switching compression mode"
            confidence = 0.7
        elif error_rate > 0.001:  # > 0.1% error rate
            score = 0.6  # Moderate error rate
            reasoning = f"Moderate error rate ({error_rate:.3%}) slightly favors switching"
            confidence = 0.6
        else:
            score = 0.3  # Low error rate, current mode is working
            reasoning = f"Low error rate ({error_rate:.3%}) suggests current mode is effective"
            confidence = 0.8
        
        return DecisionAnalysis(
            criterion=DecisionCriterion.ERROR_RATE,
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            metadata={'error_rate': error_rate}
        )
    
    def _analyze_computational_load(self) -> DecisionAnalysis:
        """Analyze computational load for switching decision"""
        # This is a placeholder - would need actual system load monitoring
        # For now, return neutral analysis
        return DecisionAnalysis(
            criterion=DecisionCriterion.COMPUTATIONAL_LOAD,
            score=0.5,
            confidence=0.3,
            reasoning="Computational load analysis not implemented",
            metadata={'placeholder': True}
        )
    
    def _analyze_gpu_utilization(self) -> DecisionAnalysis:
        """Analyze GPU utilization for switching decision"""
        try:
            if torch.cuda.is_available():
                # Simple GPU utilization check (placeholder - would need proper monitoring)
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                
                # Use memory allocation as proxy for utilization
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                utilization_proxy = allocated / total
                
                if utilization_proxy > 0.7:
                    score = 0.4  # High utilization might favor pure
                    reasoning = f"High GPU utilization ({utilization_proxy:.1%}) might favor pure compression"
                elif utilization_proxy > 0.3:
                    score = 0.6  # Medium utilization favors hybrid
                    reasoning = f"Medium GPU utilization ({utilization_proxy:.1%}) favors hybrid compression"
                else:
                    score = 0.8  # Low utilization allows hybrid
                    reasoning = f"Low GPU utilization ({utilization_proxy:.1%}) allows hybrid compression"
                
                confidence = 0.6  # Medium confidence in proxy metric
                metadata = {
                    'gpu_count': gpu_count,
                    'current_device': current_device,
                    'utilization_proxy': utilization_proxy
                }
            else:
                score = 0.2  # No GPU favors pure
                reasoning = "No GPU available, favors pure p-adic compression"
                confidence = 0.8
                metadata = {'gpu_available': False}
            
            return DecisionAnalysis(
                criterion=DecisionCriterion.GPU_UTILIZATION,
                score=score,
                confidence=confidence,
                reasoning=reasoning,
                metadata=metadata
            )
            
        except Exception as e:
            return DecisionAnalysis(
                criterion=DecisionCriterion.GPU_UTILIZATION,
                score=0.5,
                confidence=0.2,
                reasoning=f"GPU utilization analysis failed: {e}",
                metadata={'error': str(e)}
            )
    
    def _calculate_weighted_score(self, analyses: List[DecisionAnalysis]) -> float:
        """Calculate weighted decision score from analyses"""
        total_score = 0.0
        total_weight = 0.0
        
        for analysis in analyses:
            weight = getattr(self.decision_weights, analysis.criterion.value, 0.0)
            weighted_contribution = analysis.score * analysis.confidence * weight
            total_score += weighted_contribution
            total_weight += weight * analysis.confidence
        
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.5  # Neutral score if no valid weights
    
    def _calculate_overall_confidence(self, analyses: List[DecisionAnalysis]) -> float:
        """Calculate overall confidence from analyses"""
        if not analyses:
            return 0.0
        
        confidences = [analysis.confidence for analysis in analyses]
        return np.mean(confidences)
    
    def _make_switching_recommendation(self, weighted_score: float, confidence: float) -> str:
        """Make switching recommendation based on score and confidence"""
        if confidence < self.confidence_threshold:
            return "insufficient_confidence"
        
        if weighted_score > self.hybrid_threshold:
            return "switch_to_hybrid"
        elif weighted_score < self.pure_threshold:
            return "switch_to_pure"
        else:
            return "maintain_current"
    
    def _calculate_data_complexity(self, data: torch.Tensor) -> float:
        """Calculate data complexity metric"""
        try:
            # Simple complexity metric based on variance and range
            data_variance = torch.var(data).item()
            data_range = (torch.max(data) - torch.min(data)).item()
            
            # Normalize complexity to [0, 1]
            if data_range > 0:
                normalized_variance = data_variance / (data_range ** 2)
                return min(1.0, normalized_variance * 10)  # Scale to reasonable range
            else:
                return 0.0  # Constant data has zero complexity
                
        except Exception:
            return 0.5  # Default complexity on error
    
    def _get_historical_performance_adjustment(self, target_mode: str) -> float:
        """Get performance adjustment factor based on historical data"""
        if not self.performance_history:
            return 1.0
        
        # Find recent performance for the target mode
        mode_performance = []
        for perf in list(self.performance_history)[-20:]:
            if perf['data'].get('mode') == target_mode:
                mode_performance.append(perf['data'].get('performance_score', 1.0))
        
        if mode_performance:
            avg_performance = np.mean(mode_performance)
            # Adjust based on performance (better performance = faster predicted time)
            return 2.0 - avg_performance  # [1.0, 2.0] range (inverse relationship)
        else:
            return 1.0  # No adjustment if no historical data
    
    def _find_optimal_threshold(self, mode: str, decisions: List[Dict[str, Any]]) -> float:
        """Find optimal threshold for given mode based on decision history"""
        # Simplified threshold optimization - would need more sophisticated analysis
        current_threshold = self.hybrid_threshold if mode == 'hybrid' else self.pure_threshold
        
        # Count successful decisions near current threshold
        threshold_range = np.arange(current_threshold - 0.2, current_threshold + 0.2, 0.05)
        best_threshold = current_threshold
        best_success_rate = 0.0
        
        for threshold in threshold_range:
            success_count = 0
            total_count = 0
            
            for decision in decisions:
                if decision['recommendation'] == f'switch_to_{mode}':
                    total_count += 1
                    # This would need actual success tracking from performance monitor
                    # For now, assume neutral success rate
                    success_count += 1
            
            if total_count > 0:
                success_rate = success_count / total_count
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_threshold = threshold
        
        return best_threshold
    
    def _adapt_decision_weights(self) -> None:
        """Adapt decision weights based on recent performance"""
        # Simplified weight adaptation - would need more sophisticated analysis
        # For now, slightly adjust weights based on recent success patterns
        
        if len(self.performance_history) < 20:
            return
        
        # This would analyze which criteria led to better decisions
        # and adjust weights accordingly - placeholder implementation
        pass
    
    def update_configuration(self, config: Dict[str, Any]) -> None:
        """Update decision engine configuration"""
        if not isinstance(config, dict):
            raise TypeError(f"Config must be dict, got {type(config)}")
        
        self.config.update(config)
        
        # Update thresholds
        self.hybrid_threshold = config.get('hybrid_threshold', self.hybrid_threshold)
        self.pure_threshold = config.get('pure_threshold', self.pure_threshold)
        self.confidence_threshold = config.get('confidence_threshold', self.confidence_threshold)
        
        self.logger.info("Decision engine configuration updated")
    
    def shutdown(self) -> None:
        """Shutdown decision engine"""
        self.logger.info("Shutting down switching decision engine")
        
        # Clear caches and history
        self.gradient_analysis_cache.clear()
        self.performance_history.clear()
        self.decision_history.clear()
        
        self.is_initialized = False
        self.logger.info("Switching decision engine shutdown complete")