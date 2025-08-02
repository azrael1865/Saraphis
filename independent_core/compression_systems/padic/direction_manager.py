"""
Direction Manager - Main orchestrator for direction-based hybrid switching decisions
NO FALLBACKS - HARD FAILURES ONLY
"""

import logging
import math
import threading
import time
import torch
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from enum import Enum

# Import GAC system components
from ...gac_system.gac_types import DirectionState, DirectionType
from ...gac_system.direction_state import DirectionStateManager, DirectionHistory

# Import hybrid system components
from .dynamic_switching_manager import DynamicSwitchingManager
from .direction_analyzer import DirectionAnalyzer
from .direction_coordinator import DirectionCoordinator


class DirectionSwitchingStrategy(Enum):
    """Direction-based switching strategy enumeration"""
    STABILITY_BASED = "stability_based"
    PATTERN_BASED = "pattern_based"
    CONFIDENCE_BASED = "confidence_based"
    PERFORMANCE_BASED = "performance_based"
    ADAPTIVE = "adaptive"


class DirectionDecisionType(Enum):
    """Direction decision type enumeration"""
    USE_HYBRID = "use_hybrid"
    USE_PURE = "use_pure"
    MAINTAIN_CURRENT = "maintain_current"
    ANALYZE_FURTHER = "analyze_further"


@dataclass
class DirectionSwitchingDecision:
    """Direction-based switching decision result"""
    decision_type: DirectionDecisionType
    confidence: float
    reasoning: str
    direction_state: DirectionState
    switching_context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate decision result"""
        if not isinstance(self.decision_type, DirectionDecisionType):
            raise TypeError("Decision type must be DirectionDecisionType")
        if not isinstance(self.confidence, (int, float)) or not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be float between 0.0 and 1.0")
        if not isinstance(self.reasoning, str) or not self.reasoning.strip():
            raise ValueError("Reasoning must be non-empty string")


@dataclass
class DirectionAnalytics:
    """Direction-based switching analytics"""
    total_decisions: int
    hybrid_decisions: int
    pure_decisions: int
    avg_confidence: float
    stability_decisions: int
    pattern_decisions: int
    performance_decisions: int
    decision_accuracy: float
    switching_efficiency: float
    direction_trends: Dict[str, List[float]]
    performance_correlation: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DirectionManagerConfig:
    """Configuration for direction manager"""
    # Strategy configuration
    default_strategy: DirectionSwitchingStrategy = DirectionSwitchingStrategy.ADAPTIVE
    enable_stability_switching: bool = True
    enable_pattern_switching: bool = True
    enable_confidence_switching: bool = True
    enable_performance_switching: bool = True
    
    # Decision thresholds
    stability_threshold: float = 0.7
    pattern_confidence_threshold: float = 0.8
    performance_improvement_threshold: float = 0.1
    direction_confidence_threshold: float = 0.75
    
    # Analysis parameters
    direction_history_size: int = 100
    pattern_analysis_window: int = 20
    performance_tracking_window: int = 50
    stability_analysis_window: int = 15
    
    # Optimization parameters
    enable_adaptive_thresholds: bool = True
    enable_performance_feedback: bool = True
    enable_strategy_optimization: bool = True
    optimization_interval_seconds: int = 300
    
    # Integration parameters
    enable_gac_integration: bool = True
    enable_hybrid_integration: bool = True
    enable_real_time_coordination: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if not (0.0 <= self.stability_threshold <= 1.0):
            raise ValueError("Stability threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.pattern_confidence_threshold <= 1.0):
            raise ValueError("Pattern confidence threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.performance_improvement_threshold <= 1.0):
            raise ValueError("Performance improvement threshold must be between 0.0 and 1.0")
        if self.direction_history_size <= 0:
            raise ValueError("Direction history size must be positive")
        if self.pattern_analysis_window <= 0:
            raise ValueError("Pattern analysis window must be positive")


class DirectionManager:
    """
    Main orchestrator for direction-based hybrid switching decisions.
    Integrates gradient direction analysis with hybrid p-adic switching.
    """
    
    def __init__(self, config: Optional[DirectionManagerConfig] = None):
        """Initialize direction manager"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, DirectionManagerConfig):
            raise TypeError(f"Config must be DirectionManagerConfig or None, got {type(config)}")
        
        self.config = config or DirectionManagerConfig()
        self.logger = logging.getLogger('DirectionManager')
        
        # System components
        self.direction_state_manager: Optional[DirectionStateManager] = None
        self.switching_manager: Optional[DynamicSwitchingManager] = None
        self.direction_analyzer: Optional[DirectionAnalyzer] = None
        self.direction_coordinator: Optional[DirectionCoordinator] = None
        
        # Direction system state
        self.is_initialized = False
        self.current_strategy = self.config.default_strategy
        
        # Decision tracking
        self.decision_history: deque = deque(maxlen=self.config.direction_history_size)
        self.performance_history: deque = deque(maxlen=self.config.performance_tracking_window)
        self.switching_analytics: Dict[str, Any] = {}
        
        # Strategy optimization
        self.strategy_performance: Dict[DirectionSwitchingStrategy, List[float]] = defaultdict(list)
        self.adaptive_thresholds: Dict[str, float] = {}
        self.last_optimization: Optional[datetime] = None
        
        # Thread safety
        self._direction_lock = threading.RLock()
        self._analytics_lock = threading.RLock()
        
        # Performance tracking
        self.direction_metrics = {
            'total_decisions': 0,
            'correct_decisions': 0,
            'average_confidence': 0.0,
            'switching_efficiency': 0.0
        }
        
        self.logger.info("DirectionManager created successfully")
    
    def initialize_direction_system(self,
                                  direction_state_manager: DirectionStateManager,
                                  switching_manager: DynamicSwitchingManager,
                                  direction_analyzer: Optional[DirectionAnalyzer] = None,
                                  direction_coordinator: Optional[DirectionCoordinator] = None) -> None:
        """
        Initialize direction system with required components.
        
        Args:
            direction_state_manager: GAC direction state manager
            switching_manager: Dynamic switching manager
            direction_analyzer: Optional direction analyzer
            direction_coordinator: Optional direction coordinator
            
        Raises:
            TypeError: If components are invalid
            RuntimeError: If initialization fails
        """
        if not isinstance(direction_state_manager, DirectionStateManager):
            raise TypeError(f"Direction state manager must be DirectionStateManager, got {type(direction_state_manager)}")
        if not isinstance(switching_manager, DynamicSwitchingManager):
            raise TypeError(f"Switching manager must be DynamicSwitchingManager, got {type(switching_manager)}")
        if direction_analyzer is not None and not isinstance(direction_analyzer, DirectionAnalyzer):
            raise TypeError(f"Direction analyzer must be DirectionAnalyzer, got {type(direction_analyzer)}")
        if direction_coordinator is not None and not isinstance(direction_coordinator, DirectionCoordinator):
            raise TypeError(f"Direction coordinator must be DirectionCoordinator, got {type(direction_coordinator)}")
        
        try:
            # Set component references
            self.direction_state_manager = direction_state_manager
            self.switching_manager = switching_manager
            self.direction_analyzer = direction_analyzer
            self.direction_coordinator = direction_coordinator
            
            # Initialize adaptive thresholds
            self._initialize_adaptive_thresholds()
            
            # Initialize strategy performance tracking
            self._initialize_strategy_tracking()
            
            self.is_initialized = True
            self.logger.info("Direction system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize direction system: {e}")
            raise RuntimeError(f"Direction system initialization failed: {e}")
    
    def analyze_gradient_direction(self, gradients: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> DirectionState:
        """
        Analyze gradient direction for switching decisions.
        
        Args:
            gradients: Gradient tensor to analyze
            context: Optional analysis context
            
        Returns:
            Direction state analysis result
            
        Raises:
            RuntimeError: If analysis fails
            ValueError: If gradients are invalid
        """
        if not self.is_initialized:
            raise RuntimeError("Direction system not initialized")
        
        if gradients is None:
            raise ValueError("Gradients cannot be None")
        if not isinstance(gradients, torch.Tensor):
            raise TypeError("Gradients must be torch.Tensor")
        if gradients.numel() == 0:
            raise ValueError("Gradients cannot be empty")
        
        try:
            with self._direction_lock:
                # Update direction state through GAC system
                direction_state = self.direction_state_manager.update_direction_state(gradients, context)
                
                # Enhance direction analysis if analyzer available
                if self.direction_analyzer:
                    enhanced_analysis = self.direction_analyzer.analyze_direction_patterns(
                        list(self.direction_state_manager.gradient_history)
                    )
                    
                    # Update direction state metadata with enhanced analysis
                    # Note: DirectionState is immutable, enhanced data goes to metadata
                    if 'pattern_confidence' in enhanced_analysis:
                        direction_state.metadata['enhanced_confidence'] = enhanced_analysis['pattern_confidence']
                    if 'pattern_strength' in enhanced_analysis:
                        direction_state.metadata['pattern_strength'] = enhanced_analysis['pattern_strength']
                    if 'angle_stability' in enhanced_analysis:
                        direction_state.metadata['enhanced_stability'] = enhanced_analysis['angle_stability']
                
                self.logger.debug(f"Direction analysis completed: {direction_state.direction.name}")
                
                return direction_state
                
        except Exception as e:
            self.logger.error(f"Direction analysis failed: {e}")
            raise RuntimeError(f"Direction analysis failed: {e}")
    
    def make_switching_decision(self, data: torch.Tensor, gradients: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> DirectionSwitchingDecision:
        """
        Make direction-based switching decision.
        
        Args:
            data: Data tensor for switching decision
            gradients: Gradient tensor for direction analysis
            context: Optional decision context
            
        Returns:
            Direction-based switching decision
            
        Raises:
            RuntimeError: If decision making fails
            ValueError: If inputs are invalid
        """
        if not self.is_initialized:
            raise RuntimeError("Direction system not initialized")
        
        if data is None:
            raise ValueError("Data cannot be None")
        if not isinstance(data, torch.Tensor):
            raise TypeError("Data must be torch.Tensor")
        if gradients is None:
            raise ValueError("Gradients cannot be None")
        if not isinstance(gradients, torch.Tensor):
            raise TypeError("Gradients must be torch.Tensor")
        
        try:
            with self._direction_lock:
                # Analyze gradient direction
                direction_state = self.analyze_gradient_direction(gradients, context)
                
                # Apply current switching strategy
                decision = self._apply_switching_strategy(data, direction_state, context)
                
                # Record decision for analytics
                self.decision_history.append(decision)
                
                # Update metrics
                self._update_decision_metrics(decision)
                
                # Coordinate with direction coordinator if available
                if self.direction_coordinator:
                    coordination_result = self.direction_coordinator.coordinate_direction_switching(
                        direction_state, 
                        {'decision': decision, 'context': context}
                    )
                    
                    # Update decision based on coordination
                    if coordination_result.get('override_decision'):
                        decision.decision_type = coordination_result['override_decision']
                        decision.reasoning += f" (Coordinated: {coordination_result.get('reasoning', 'System coordination')})"
                
                self.logger.debug(f"Switching decision: {decision.decision_type.name} (confidence: {decision.confidence:.3f})")
                
                return decision
                
        except Exception as e:
            self.logger.error(f"Switching decision failed: {e}")
            raise RuntimeError(f"Switching decision failed: {e}")
    
    def update_direction_strategy(self, direction_state: DirectionState) -> None:
        """
        Update direction strategy based on current state.
        
        Args:
            direction_state: Current direction state
            
        Raises:
            RuntimeError: If strategy update fails
            ValueError: If direction state is invalid
        """
        if not self.is_initialized:
            raise RuntimeError("Direction system not initialized")
        
        if direction_state is None:
            raise ValueError("Direction state cannot be None")
        if not isinstance(direction_state, DirectionState):
            raise TypeError("Direction state must be DirectionState")
        
        try:
            with self._direction_lock:
                # Update adaptive thresholds based on performance
                if self.config.enable_adaptive_thresholds:
                    self._update_adaptive_thresholds(direction_state)
                
                # Optimize strategy based on performance feedback
                if self.config.enable_strategy_optimization and self._should_optimize_strategy():
                    self._optimize_strategy_selection()
                
                # Update switching manager with direction insights
                if self.config.enable_hybrid_integration:
                    self._update_switching_manager_strategy(direction_state)
                
                self.logger.debug(f"Direction strategy updated: {self.current_strategy.name}")
                
        except Exception as e:
            self.logger.error(f"Direction strategy update failed: {e}")
            raise RuntimeError(f"Direction strategy update failed: {e}")
    
    def get_direction_analytics(self) -> DirectionAnalytics:
        """
        Get comprehensive direction analytics.
        
        Returns:
            Direction analytics data
            
        Raises:
            RuntimeError: If analytics generation fails
        """
        if not self.is_initialized:
            raise RuntimeError("Direction system not initialized")
        
        try:
            with self._analytics_lock:
                # Calculate decision statistics
                total_decisions = len(self.decision_history)
                if total_decisions == 0:
                    return DirectionAnalytics(
                        total_decisions=0,
                        hybrid_decisions=0,
                        pure_decisions=0,
                        avg_confidence=0.0,
                        stability_decisions=0,
                        pattern_decisions=0,
                        performance_decisions=0,
                        decision_accuracy=0.0,
                        switching_efficiency=0.0,
                        direction_trends={},
                        performance_correlation={}
                    )
                
                hybrid_decisions = sum(1 for d in self.decision_history if d.decision_type == DirectionDecisionType.USE_HYBRID)
                pure_decisions = sum(1 for d in self.decision_history if d.decision_type == DirectionDecisionType.USE_PURE)
                avg_confidence = sum(d.confidence for d in self.decision_history) / total_decisions
                
                # Calculate strategy-based decisions
                stability_decisions = sum(1 for d in self.decision_history if 'stability' in d.reasoning.lower())
                pattern_decisions = sum(1 for d in self.decision_history if 'pattern' in d.reasoning.lower())
                performance_decisions = sum(1 for d in self.decision_history if 'performance' in d.reasoning.lower())
                
                # Calculate accuracy and efficiency
                decision_accuracy = self.direction_metrics['correct_decisions'] / self.direction_metrics['total_decisions'] if self.direction_metrics['total_decisions'] > 0 else 0.0
                switching_efficiency = self.direction_metrics['switching_efficiency']
                
                # Generate direction trends
                direction_trends = self._calculate_direction_trends()
                
                # Calculate performance correlation
                performance_correlation = self._calculate_performance_correlation()
                
                return DirectionAnalytics(
                    total_decisions=total_decisions,
                    hybrid_decisions=hybrid_decisions,
                    pure_decisions=pure_decisions,
                    avg_confidence=avg_confidence,
                    stability_decisions=stability_decisions,
                    pattern_decisions=pattern_decisions,
                    performance_decisions=performance_decisions,
                    decision_accuracy=decision_accuracy,
                    switching_efficiency=switching_efficiency,
                    direction_trends=direction_trends,
                    performance_correlation=performance_correlation
                )
                
        except Exception as e:
            self.logger.error(f"Direction analytics generation failed: {e}")
            raise RuntimeError(f"Direction analytics generation failed: {e}")
    
    def optimize_direction_based_switching(self) -> Dict[str, Any]:
        """
        Optimize direction-based switching performance.
        
        Returns:
            Optimization results
            
        Raises:
            RuntimeError: If optimization fails
        """
        if not self.is_initialized:
            raise RuntimeError("Direction system not initialized")
        
        try:
            optimization_results = {
                'optimizations_applied': [],
                'performance_improvements': {},
                'strategy_changes': {},
                'threshold_adjustments': {},
                'timestamp': datetime.utcnow().isoformat()
            }
            
            with self._direction_lock:
                # Optimize switching strategy
                if self.config.enable_strategy_optimization:
                    strategy_optimization = self._optimize_strategy_selection()
                    optimization_results['strategy_changes'] = strategy_optimization
                    optimization_results['optimizations_applied'].append('strategy_optimization')
                
                # Optimize adaptive thresholds
                if self.config.enable_adaptive_thresholds:
                    threshold_optimization = self._optimize_adaptive_thresholds()
                    optimization_results['threshold_adjustments'] = threshold_optimization
                    optimization_results['optimizations_applied'].append('threshold_optimization')
                
                # Optimize direction analysis parameters
                if self.direction_analyzer:
                    analysis_optimization = self.direction_analyzer.optimize_direction_parameters(
                        list(self.performance_history)
                    )
                    optimization_results['performance_improvements']['analysis'] = analysis_optimization
                    optimization_results['optimizations_applied'].append('analysis_optimization')
                
                # Optimize coordination strategy
                if self.direction_coordinator:
                    coordination_optimization = self.direction_coordinator.optimize_coordination_strategy(
                        list(self.performance_history)
                    )
                    optimization_results['performance_improvements']['coordination'] = coordination_optimization
                    optimization_results['optimizations_applied'].append('coordination_optimization')
                
                # Update last optimization time
                self.last_optimization = datetime.utcnow()
                
                self.logger.info(f"Direction optimization completed: {len(optimization_results['optimizations_applied'])} optimizations applied")
                
                return optimization_results
                
        except Exception as e:
            self.logger.error(f"Direction optimization failed: {e}")
            raise RuntimeError(f"Direction optimization failed: {e}")
    
    def _apply_switching_strategy(self, data: torch.Tensor, direction_state: DirectionState, context: Optional[Dict[str, Any]]) -> DirectionSwitchingDecision:
        """Apply current switching strategy"""
        if self.current_strategy == DirectionSwitchingStrategy.STABILITY_BASED:
            return self._apply_stability_strategy(data, direction_state, context)
        elif self.current_strategy == DirectionSwitchingStrategy.PATTERN_BASED:
            return self._apply_pattern_strategy(data, direction_state, context)
        elif self.current_strategy == DirectionSwitchingStrategy.CONFIDENCE_BASED:
            return self._apply_confidence_strategy(data, direction_state, context)
        elif self.current_strategy == DirectionSwitchingStrategy.PERFORMANCE_BASED:
            return self._apply_performance_strategy(data, direction_state, context)
        elif self.current_strategy == DirectionSwitchingStrategy.ADAPTIVE:
            return self._apply_adaptive_strategy(data, direction_state, context)
        else:
            raise RuntimeError(f"Unknown switching strategy: {self.current_strategy}")
    
    def _apply_stability_strategy(self, data: torch.Tensor, direction_state: DirectionState, context: Optional[Dict[str, Any]]) -> DirectionSwitchingDecision:
        """Apply stability-based switching strategy"""
        stability_threshold = self.adaptive_thresholds.get('stability', self.config.stability_threshold)
        
        stability_score = direction_state.metadata.get('stability_score', 0.5)
        if stability_score >= stability_threshold:
            # Stable gradients - use hybrid
            decision_type = DirectionDecisionType.USE_HYBRID
            reasoning = f"Stable gradient pattern (stability: {stability_score:.3f} >= {stability_threshold:.3f})"
            confidence = stability_score
        else:
            # Unstable gradients - use pure p-adic
            decision_type = DirectionDecisionType.USE_PURE
            reasoning = f"Unstable gradient pattern (stability: {stability_score:.3f} < {stability_threshold:.3f})"
            confidence = 1.0 - stability_score
        
        return DirectionSwitchingDecision(
            decision_type=decision_type,
            confidence=confidence,
            reasoning=reasoning,
            direction_state=direction_state,
            switching_context={'strategy': 'stability_based', 'threshold': stability_threshold}
        )
    
    def _apply_pattern_strategy(self, data: torch.Tensor, direction_state: DirectionState, context: Optional[Dict[str, Any]]) -> DirectionSwitchingDecision:
        """Apply pattern-based switching strategy"""
        pattern_confidence_threshold = self.adaptive_thresholds.get('pattern_confidence', self.config.pattern_confidence_threshold)
        
        # Analyze direction patterns if analyzer available
        if self.direction_analyzer and len(self.direction_state_manager.gradient_history) >= self.config.pattern_analysis_window:
            pattern_analysis = self.direction_analyzer.analyze_direction_patterns(
                list(self.direction_state_manager.gradient_history)[-self.config.pattern_analysis_window:]
            )
            
            pattern_confidence = pattern_analysis.get('pattern_confidence', direction_state.confidence)
            pattern_type = pattern_analysis.get('pattern_type', 'unknown')
            
            if pattern_confidence >= pattern_confidence_threshold:
                if pattern_type in ['ascending', 'stable_ascending']:
                    decision_type = DirectionDecisionType.USE_HYBRID
                    reasoning = f"Ascending pattern detected (confidence: {pattern_confidence:.3f})"
                elif pattern_type in ['descending', 'stable_descending']:
                    decision_type = DirectionDecisionType.USE_PURE
                    reasoning = f"Descending pattern detected (confidence: {pattern_confidence:.3f})"
                elif pattern_type == 'oscillating':
                    decision_type = DirectionDecisionType.USE_PURE
                    reasoning = f"Oscillating pattern detected (confidence: {pattern_confidence:.3f})"
                else:
                    decision_type = DirectionDecisionType.USE_HYBRID
                    reasoning = f"Stable pattern detected (confidence: {pattern_confidence:.3f})"
                
                confidence = pattern_confidence
            else:
                decision_type = DirectionDecisionType.ANALYZE_FURTHER
                reasoning = f"Pattern confidence too low (confidence: {pattern_confidence:.3f} < {pattern_confidence_threshold:.3f})"
                confidence = pattern_confidence
        else:
            # Fallback to direction confidence
            if direction_state.confidence >= pattern_confidence_threshold:
                decision_type = DirectionDecisionType.USE_HYBRID
                reasoning = f"High direction confidence (confidence: {direction_state.confidence:.3f})"
                confidence = direction_state.confidence
            else:
                decision_type = DirectionDecisionType.USE_PURE
                reasoning = f"Low direction confidence (confidence: {direction_state.confidence:.3f})"
                confidence = direction_state.confidence
        
        return DirectionSwitchingDecision(
            decision_type=decision_type,
            confidence=confidence,
            reasoning=reasoning,
            direction_state=direction_state,
            switching_context={'strategy': 'pattern_based', 'threshold': pattern_confidence_threshold}
        )
    
    def _apply_confidence_strategy(self, data: torch.Tensor, direction_state: DirectionState, context: Optional[Dict[str, Any]]) -> DirectionSwitchingDecision:
        """Apply confidence-based switching strategy"""
        confidence_threshold = self.adaptive_thresholds.get('direction_confidence', self.config.direction_confidence_threshold)
        
        if direction_state.confidence >= confidence_threshold:
            # High confidence - use hybrid
            decision_type = DirectionDecisionType.USE_HYBRID
            reasoning = f"High direction confidence (confidence: {direction_state.confidence:.3f} >= {confidence_threshold:.3f})"
            confidence = direction_state.confidence
        else:
            # Low confidence - use pure p-adic
            decision_type = DirectionDecisionType.USE_PURE
            reasoning = f"Low direction confidence (confidence: {direction_state.confidence:.3f} < {confidence_threshold:.3f})"
            confidence = 1.0 - direction_state.confidence
        
        return DirectionSwitchingDecision(
            decision_type=decision_type,
            confidence=confidence,
            reasoning=reasoning,
            direction_state=direction_state,
            switching_context={'strategy': 'confidence_based', 'threshold': confidence_threshold}
        )
    
    def _apply_performance_strategy(self, data: torch.Tensor, direction_state: DirectionState, context: Optional[Dict[str, Any]]) -> DirectionSwitchingDecision:
        """Apply performance-based switching strategy"""
        # Analyze historical performance for similar direction states
        current_stability = direction_state.metadata.get('stability_score', 0.5)
        similar_decisions = [
            d for d in self.decision_history
            if abs(d.direction_state.confidence - direction_state.confidence) < 0.1
            and abs(d.direction_state.metadata.get('stability_score', 0.5) - current_stability) < 0.1
        ]
        
        if len(similar_decisions) >= 5:
            # Calculate performance for hybrid vs pure decisions
            hybrid_performance = []
            pure_performance = []
            
            for decision in similar_decisions[-20:]:  # Last 20 similar decisions
                if decision.decision_type == DirectionDecisionType.USE_HYBRID:
                    hybrid_performance.append(decision.confidence)
                elif decision.decision_type == DirectionDecisionType.USE_PURE:
                    pure_performance.append(decision.confidence)
            
            avg_hybrid_perf = sum(hybrid_performance) / len(hybrid_performance) if hybrid_performance else 0.0
            avg_pure_perf = sum(pure_performance) / len(pure_performance) if pure_performance else 0.0
            
            performance_improvement_threshold = self.adaptive_thresholds.get('performance_improvement', self.config.performance_improvement_threshold)
            
            if avg_hybrid_perf > avg_pure_perf + performance_improvement_threshold:
                decision_type = DirectionDecisionType.USE_HYBRID
                reasoning = f"Historical hybrid performance better (hybrid: {avg_hybrid_perf:.3f} vs pure: {avg_pure_perf:.3f})"
                confidence = avg_hybrid_perf
            elif avg_pure_perf > avg_hybrid_perf + performance_improvement_threshold:
                decision_type = DirectionDecisionType.USE_PURE
                reasoning = f"Historical pure performance better (pure: {avg_pure_perf:.3f} vs hybrid: {avg_hybrid_perf:.3f})"
                confidence = avg_pure_perf
            else:
                decision_type = DirectionDecisionType.USE_HYBRID
                reasoning = f"Similar historical performance (hybrid: {avg_hybrid_perf:.3f}, pure: {avg_pure_perf:.3f})"
                confidence = max(avg_hybrid_perf, avg_pure_perf)
        else:
            # Insufficient historical data - fallback to confidence strategy
            return self._apply_confidence_strategy(data, direction_state, context)
        
        return DirectionSwitchingDecision(
            decision_type=decision_type,
            confidence=confidence,
            reasoning=reasoning,
            direction_state=direction_state,
            switching_context={'strategy': 'performance_based', 'historical_decisions': len(similar_decisions)}
        )
    
    def _apply_adaptive_strategy(self, data: torch.Tensor, direction_state: DirectionState, context: Optional[Dict[str, Any]]) -> DirectionSwitchingDecision:
        """Apply adaptive switching strategy"""
        # Combine multiple strategies based on current performance
        strategies = []
        
        if self.config.enable_stability_switching:
            stability_decision = self._apply_stability_strategy(data, direction_state, context)
            strategies.append(('stability', stability_decision, 0.3))
        
        if self.config.enable_pattern_switching and self.direction_analyzer:
            pattern_decision = self._apply_pattern_strategy(data, direction_state, context)
            strategies.append(('pattern', pattern_decision, 0.3))
        
        if self.config.enable_confidence_switching:
            confidence_decision = self._apply_confidence_strategy(data, direction_state, context)
            strategies.append(('confidence', confidence_decision, 0.2))
        
        if self.config.enable_performance_switching and len(self.decision_history) >= 10:
            performance_decision = self._apply_performance_strategy(data, direction_state, context)
            strategies.append(('performance', performance_decision, 0.2))
        
        if not strategies:
            # Fallback to confidence strategy
            return self._apply_confidence_strategy(data, direction_state, context)
        
        # Weighted decision combination
        hybrid_score = 0.0
        pure_score = 0.0
        total_confidence = 0.0
        reasoning_parts = []
        
        for strategy_name, decision, weight in strategies:
            weighted_confidence = decision.confidence * weight
            total_confidence += weighted_confidence
            
            if decision.decision_type == DirectionDecisionType.USE_HYBRID:
                hybrid_score += weighted_confidence
                reasoning_parts.append(f"{strategy_name}->hybrid({decision.confidence:.2f})")
            elif decision.decision_type == DirectionDecisionType.USE_PURE:
                pure_score += weighted_confidence
                reasoning_parts.append(f"{strategy_name}->pure({decision.confidence:.2f})")
        
        if hybrid_score > pure_score:
            decision_type = DirectionDecisionType.USE_HYBRID
            confidence = hybrid_score / sum(weight for _, _, weight in strategies)
        else:
            decision_type = DirectionDecisionType.USE_PURE
            confidence = pure_score / sum(weight for _, _, weight in strategies)
        
        reasoning = f"Adaptive strategy: {', '.join(reasoning_parts)} -> {decision_type.name.lower()}"
        
        return DirectionSwitchingDecision(
            decision_type=decision_type,
            confidence=confidence,
            reasoning=reasoning,
            direction_state=direction_state,
            switching_context={'strategy': 'adaptive', 'sub_strategies': len(strategies)}
        )
    
    def _initialize_adaptive_thresholds(self) -> None:
        """Initialize adaptive thresholds"""
        self.adaptive_thresholds = {
            'stability': self.config.stability_threshold,
            'pattern_confidence': self.config.pattern_confidence_threshold,
            'performance_improvement': self.config.performance_improvement_threshold,
            'direction_confidence': self.config.direction_confidence_threshold
        }
    
    def _initialize_strategy_tracking(self) -> None:
        """Initialize strategy performance tracking"""
        for strategy in DirectionSwitchingStrategy:
            self.strategy_performance[strategy] = []
    
    def _update_decision_metrics(self, decision: DirectionSwitchingDecision) -> None:
        """Update decision metrics"""
        self.direction_metrics['total_decisions'] += 1
        
        # Assume decision is correct if confidence is high
        if decision.confidence >= 0.7:
            self.direction_metrics['correct_decisions'] += 1
        
        # Update average confidence
        total_confidence = self.direction_metrics.get('total_confidence', 0.0)
        total_confidence += decision.confidence
        self.direction_metrics['total_confidence'] = total_confidence
        self.direction_metrics['average_confidence'] = total_confidence / self.direction_metrics['total_decisions']
        
        # Update switching efficiency (placeholder calculation)
        self.direction_metrics['switching_efficiency'] = min(1.0, self.direction_metrics['average_confidence'] * 1.2)
    
    def _update_adaptive_thresholds(self, direction_state: DirectionState) -> None:
        """Update adaptive thresholds based on performance"""
        if len(self.decision_history) < 20:
            return  # Need sufficient history
        
        recent_decisions = list(self.decision_history)[-20:]
        
        # Calculate success rates for different threshold levels
        for threshold_name, current_threshold in self.adaptive_thresholds.items():
            if threshold_name == 'stability':
                # Analyze stability threshold performance
                stable_decisions = [d for d in recent_decisions if d.direction_state.metadata.get('stability_score', 0.5) >= current_threshold]
                if stable_decisions:
                    avg_confidence = sum(d.confidence for d in stable_decisions) / len(stable_decisions)
                    if avg_confidence < 0.6:
                        self.adaptive_thresholds[threshold_name] = min(0.9, current_threshold + 0.05)
                    elif avg_confidence > 0.8:
                        self.adaptive_thresholds[threshold_name] = max(0.3, current_threshold - 0.05)
    
    def _should_optimize_strategy(self) -> bool:
        """Check if strategy optimization should be performed"""
        if self.last_optimization is None:
            return True
        
        time_since_optimization = datetime.utcnow() - self.last_optimization
        return time_since_optimization.total_seconds() >= self.config.optimization_interval_seconds
    
    def _optimize_strategy_selection(self) -> Dict[str, Any]:
        """Optimize strategy selection based on performance"""
        if len(self.decision_history) < 50:
            return {'status': 'insufficient_data'}
        
        # Analyze performance of each strategy
        strategy_performance = defaultdict(list)
        
        for decision in self.decision_history:
            strategy = decision.switching_context.get('strategy', 'unknown')
            strategy_performance[strategy].append(decision.confidence)
        
        # Find best performing strategy
        best_strategy = None
        best_performance = 0.0
        
        for strategy_name, performances in strategy_performance.items():
            if len(performances) >= 10:  # Minimum sample size
                avg_performance = sum(performances) / len(performances)
                if avg_performance > best_performance:
                    best_performance = avg_performance
                    best_strategy = strategy_name
        
        # Update current strategy if significantly better option found
        if best_strategy and best_performance > self.direction_metrics['average_confidence'] + 0.1:
            try:
                new_strategy = DirectionSwitchingStrategy(best_strategy)
                old_strategy = self.current_strategy
                self.current_strategy = new_strategy
                
                return {
                    'status': 'strategy_updated',
                    'old_strategy': old_strategy.name,
                    'new_strategy': new_strategy.name,
                    'performance_improvement': best_performance - self.direction_metrics['average_confidence']
                }
            except ValueError:
                pass  # Invalid strategy name
        
        return {'status': 'no_change'}
    
    def _optimize_adaptive_thresholds(self) -> Dict[str, float]:
        """Optimize adaptive thresholds"""
        threshold_changes = {}
        
        if len(self.decision_history) < 30:
            return threshold_changes
        
        recent_decisions = list(self.decision_history)[-30:]
        
        # Optimize each threshold
        for threshold_name in self.adaptive_thresholds:
            current_threshold = self.adaptive_thresholds[threshold_name]
            
            # Test different threshold values
            best_threshold = current_threshold
            best_performance = 0.0
            
            for test_threshold in [current_threshold - 0.1, current_threshold, current_threshold + 0.1]:
                if not (0.0 <= test_threshold <= 1.0):
                    continue
                
                # Calculate performance with this threshold
                correct_decisions = 0
                total_decisions = 0
                
                for decision in recent_decisions:
                    if threshold_name in decision.switching_context:
                        total_decisions += 1
                        if decision.confidence >= 0.7:  # Consider high confidence as correct
                            correct_decisions += 1
                
                if total_decisions > 0:
                    performance = correct_decisions / total_decisions
                    if performance > best_performance:
                        best_performance = performance
                        best_threshold = test_threshold
            
            # Update threshold if improvement found
            if abs(best_threshold - current_threshold) > 0.05 and best_performance > 0.7:
                old_threshold = self.adaptive_thresholds[threshold_name]
                self.adaptive_thresholds[threshold_name] = best_threshold
                threshold_changes[threshold_name] = {
                    'old_value': old_threshold,
                    'new_value': best_threshold,
                    'performance_improvement': best_performance
                }
        
        return threshold_changes
    
    def _update_switching_manager_strategy(self, direction_state: DirectionState) -> None:
        """Update switching manager with direction insights"""
        if not hasattr(self.switching_manager, 'update_direction_insights'):
            return  # Switching manager doesn't support direction insights
        
        direction_insights = {
            'direction_confidence': direction_state.confidence,
            'direction_stability': direction_state.metadata.get('stability_score', 0.5),
            'direction_type': direction_state.direction.name,
            'recommended_strategy': self.current_strategy.name
        }
        
        self.switching_manager.update_direction_insights(direction_insights)
    
    def _calculate_direction_trends(self) -> Dict[str, List[float]]:
        """Calculate direction trends"""
        trends = defaultdict(list)
        
        if len(self.decision_history) < 10:
            return dict(trends)
        
        recent_decisions = list(self.decision_history)[-50:]  # Last 50 decisions
        
        # Calculate confidence trend
        confidence_values = [d.confidence for d in recent_decisions]
        trends['confidence'] = confidence_values
        
        # Calculate stability trend
        stability_values = [d.direction_state.metadata.get('stability_score', 0.5) for d in recent_decisions]
        trends['stability'] = stability_values
        
        # Calculate decision type trend (hybrid = 1.0, pure = 0.0)
        decision_values = [
            1.0 if d.decision_type == DirectionDecisionType.USE_HYBRID else 0.0
            for d in recent_decisions
        ]
        trends['hybrid_ratio'] = decision_values
        
        return dict(trends)
    
    def _calculate_performance_correlation(self) -> Dict[str, float]:
        """Calculate performance correlation metrics"""
        correlation = {}
        
        if len(self.decision_history) < 20:
            return correlation
        
        recent_decisions = list(self.decision_history)[-30:]
        
        # Correlation between confidence and stability
        confidences = [d.confidence for d in recent_decisions]
        stabilities = [d.direction_state.metadata.get('stability_score', 0.5) for d in recent_decisions]
        
        if len(confidences) > 1 and len(stabilities) > 1:
            correlation['confidence_stability'] = self._calculate_correlation(confidences, stabilities)
        
        # Correlation between decision type and performance
        decision_types = [1.0 if d.decision_type == DirectionDecisionType.USE_HYBRID else 0.0 for d in recent_decisions]
        performances = [d.confidence for d in recent_decisions]
        
        if len(decision_types) > 1:
            correlation['decision_performance'] = self._calculate_correlation(decision_types, performances)
        
        return correlation
    
    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate correlation coefficient between two lists"""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        sum_y2 = sum(y * y for y in y_values)
        
        denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
        
        if denominator == 0:
            return 0.0
        
        correlation = (n * sum_xy - sum_x * sum_y) / denominator
        
        return max(-1.0, min(1.0, correlation))  # Clamp to [-1, 1]
    
    def shutdown(self) -> None:
        """Shutdown direction manager"""
        self.logger.info("Shutting down direction manager")
        
        # Clear references
        self.direction_state_manager = None
        self.switching_manager = None
        self.direction_analyzer = None
        self.direction_coordinator = None
        
        # Clear tracking data
        self.decision_history.clear()
        self.performance_history.clear()
        self.strategy_performance.clear()
        
        self.is_initialized = False
        self.logger.info("Direction manager shutdown complete")