"""
Direction Coordinator - Coordination between direction analysis and switching systems
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
from gac_system.gac_types import DirectionState, DirectionType
from gac_system.direction_state import DirectionHistory

# Import direction analysis components
from .direction_analyzer import DirectionAnalyzer, DirectionPatternAnalysis, DirectionTrendAnalysis


class CoordinationStrategy(Enum):
    """Coordination strategy enumeration"""
    REACTIVE = "reactive"
    PROACTIVE = "proactive"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"
    CONSERVATIVE = "conservative"


class CoordinationMode(Enum):
    """Coordination mode enumeration"""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    SEMI_AUTOMATIC = "semi_automatic"
    EMERGENCY = "emergency"


class CoordinationPriority(Enum):
    """Coordination priority enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class CoordinationDecision:
    """Coordination decision result"""
    decision_id: str
    coordination_strategy: CoordinationStrategy
    override_decision: Optional[str]
    confidence: float
    reasoning: str
    coordination_actions: List[str]
    performance_impact: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate coordination decision"""
        if not isinstance(self.decision_id, str) or not self.decision_id.strip():
            raise ValueError("Decision ID must be non-empty string")
        if not isinstance(self.coordination_strategy, CoordinationStrategy):
            raise TypeError("Coordination strategy must be CoordinationStrategy")
        if not isinstance(self.confidence, (int, float)) or not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be float between 0.0 and 1.0")
        if not isinstance(self.reasoning, str) or not self.reasoning.strip():
            raise ValueError("Reasoning must be non-empty string")


@dataclass
class CoordinationPolicy:
    """Coordination policy definition"""
    policy_id: str
    policy_name: str
    strategy: CoordinationStrategy
    priority: CoordinationPriority
    conditions: Dict[str, Any]
    actions: List[str]
    enabled: bool = True
    
    def __post_init__(self):
        """Validate coordination policy"""
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError("Policy ID must be non-empty string")
        if not isinstance(self.strategy, CoordinationStrategy):
            raise TypeError("Strategy must be CoordinationStrategy")
        if not isinstance(self.priority, CoordinationPriority):
            raise TypeError("Priority must be CoordinationPriority")


@dataclass
class CoordinationPerformanceMetrics:
    """Coordination performance metrics"""
    total_coordinations: int
    successful_coordinations: int
    override_rate: float
    average_coordination_time_ms: float
    performance_improvement: float
    error_rate: float
    coordination_efficiency: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CoordinationConfig:
    """Configuration for direction coordinator"""
    # Coordination strategy
    default_strategy: CoordinationStrategy = CoordinationStrategy.ADAPTIVE
    coordination_mode: CoordinationMode = CoordinationMode.AUTOMATIC
    enable_proactive_coordination: bool = True
    enable_predictive_coordination: bool = True
    
    # Performance thresholds
    performance_improvement_threshold: float = 0.05
    override_confidence_threshold: float = 0.8
    coordination_timeout_ms: float = 100.0
    error_tolerance_rate: float = 0.05
    
    # Policy management
    enable_dynamic_policies: bool = True
    policy_update_interval_seconds: int = 600
    policy_performance_window: int = 100
    
    # Coordination optimization
    enable_coordination_optimization: bool = True
    optimization_interval_seconds: int = 300
    performance_tracking_window: int = 50
    
    # Integration parameters
    enable_gac_coordination: bool = True
    enable_hybrid_coordination: bool = True
    enable_analyzer_coordination: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if not (0.0 <= self.performance_improvement_threshold <= 1.0):
            raise ValueError("Performance improvement threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.override_confidence_threshold <= 1.0):
            raise ValueError("Override confidence threshold must be between 0.0 and 1.0")
        if self.coordination_timeout_ms <= 0:
            raise ValueError("Coordination timeout must be positive")
        if not (0.0 <= self.error_tolerance_rate <= 1.0):
            raise ValueError("Error tolerance rate must be between 0.0 and 1.0")


class DirectionCoordinator:
    """
    Coordination between direction analysis and switching systems.
    Provides real-time coordination and adaptive switching policies.
    """
    
    def __init__(self, config: Optional[CoordinationConfig] = None):
        """Initialize direction coordinator"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, CoordinationConfig):
            raise TypeError(f"Config must be CoordinationConfig or None, got {type(config)}")
        
        self.config = config or CoordinationConfig()
        self.logger = logging.getLogger('DirectionCoordinator')
        
        # Coordination state
        self.current_strategy = self.config.default_strategy
        self.coordination_mode = self.config.coordination_mode
        
        # Coordination policies
        self.coordination_policies: Dict[str, CoordinationPolicy] = {}
        self.active_policies: List[str] = []
        
        # Performance tracking
        self.coordination_history: deque = deque(maxlen=self.config.performance_tracking_window)
        self.performance_metrics = CoordinationPerformanceMetrics(
            total_coordinations=0,
            successful_coordinations=0,
            override_rate=0.0,
            average_coordination_time_ms=0.0,
            performance_improvement=0.0,
            error_rate=0.0,
            coordination_efficiency=0.0
        )
        
        # Strategy optimization
        self.strategy_performance: Dict[CoordinationStrategy, List[float]] = defaultdict(list)
        self.policy_performance: Dict[str, List[float]] = defaultdict(list)
        self.last_optimization: Optional[datetime] = None
        
        # Thread safety
        self._coordination_lock = threading.RLock()
        self._policy_lock = threading.RLock()
        self._metrics_lock = threading.RLock()
        
        # Coordination analytics
        self.coordination_analytics = {
            'total_decisions': 0,
            'override_decisions': 0,
            'proactive_actions': 0,
            'predictive_actions': 0,
            'emergency_overrides': 0,
            'average_response_time_ms': 0.0
        }
        
        # Initialize default policies
        self._initialize_default_policies()
        
        self.logger.info("DirectionCoordinator created successfully")
    
    def coordinate_direction_switching(self, direction_state: DirectionState, switching_context: Dict[str, Any]) -> CoordinationDecision:
        """
        Coordinate direction analysis with switching decisions.
        
        Args:
            direction_state: Current direction state
            switching_context: Switching decision context
            
        Returns:
            Coordination decision
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If coordination fails
        """
        if direction_state is None:
            raise ValueError("Direction state cannot be None")
        if not isinstance(direction_state, DirectionState):
            raise TypeError("Direction state must be DirectionState")
        if not isinstance(switching_context, dict):
            raise TypeError("Switching context must be dict")
        
        try:
            start_time = time.time()
            decision_id = f"coord_{int(time.time() * 1000)}"
            
            with self._coordination_lock:
                # Apply current coordination strategy
                coordination_decision = self._apply_coordination_strategy(
                    direction_state, switching_context, decision_id
                )
                
                # Check for policy overrides
                policy_override = self._check_policy_overrides(
                    direction_state, switching_context, coordination_decision
                )
                
                if policy_override:
                    coordination_decision = policy_override
                
                # Record coordination for analytics
                coordination_time = (time.time() - start_time) * 1000
                self._record_coordination(coordination_decision, coordination_time)
                
                # Update performance metrics
                self._update_coordination_metrics(coordination_decision, coordination_time)
                
                self.logger.debug(f"Coordination completed: {coordination_decision.coordination_strategy.name} (confidence: {coordination_decision.confidence:.3f})")
                
                return coordination_decision
                
        except Exception as e:
            self.logger.error(f"Direction switching coordination failed: {e}")
            raise RuntimeError(f"Direction switching coordination failed: {e}")
    
    def update_switching_policy(self, direction_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update switching policy based on direction analysis.
        
        Args:
            direction_analysis: Direction analysis results
            
        Returns:
            Policy update results
            
        Raises:
            ValueError: If direction analysis is invalid
            RuntimeError: If policy update fails
        """
        if not isinstance(direction_analysis, dict):
            raise TypeError("Direction analysis must be dict")
        if not direction_analysis:
            raise ValueError("Direction analysis cannot be empty")
        
        try:
            with self._policy_lock:
                update_results = {
                    'policies_updated': [],
                    'policies_created': [],
                    'policies_disabled': [],
                    'performance_improvements': {},
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Analyze direction patterns for policy updates
                if 'pattern_type' in direction_analysis:
                    pattern_updates = self._update_pattern_policies(direction_analysis)
                    update_results['policies_updated'].extend(pattern_updates)
                
                # Update stability-based policies
                if 'stability_score' in direction_analysis:
                    stability_updates = self._update_stability_policies(direction_analysis)
                    update_results['policies_updated'].extend(stability_updates)
                
                # Update performance-based policies
                if 'performance_correlation' in direction_analysis:
                    performance_updates = self._update_performance_policies(direction_analysis)
                    update_results['policies_updated'].extend(performance_updates)
                
                # Create adaptive policies if needed
                if self.config.enable_dynamic_policies:
                    adaptive_policies = self._create_adaptive_policies(direction_analysis)
                    update_results['policies_created'].extend(adaptive_policies)
                
                # Disable underperforming policies
                disabled_policies = self._disable_underperforming_policies()
                update_results['policies_disabled'].extend(disabled_policies)
                
                # Calculate performance improvements
                update_results['performance_improvements'] = self._calculate_policy_improvements()
                
                self.logger.info(f"Policy update completed: {len(update_results['policies_updated'])} updated, {len(update_results['policies_created'])} created")
                
                return update_results
                
        except Exception as e:
            self.logger.error(f"Switching policy update failed: {e}")
            raise RuntimeError(f"Switching policy update failed: {e}")
    
    def monitor_direction_switching_performance(self) -> Dict[str, Any]:
        """
        Monitor direction switching performance.
        
        Returns:
            Performance monitoring results
            
        Raises:
            RuntimeError: If monitoring fails
        """
        try:
            with self._metrics_lock:
                monitoring_results = {
                    'current_performance': self._get_current_performance(),
                    'performance_trends': self._analyze_performance_trends(),
                    'coordination_efficiency': self._calculate_coordination_efficiency(),
                    'strategy_performance': self._analyze_strategy_performance(),
                    'policy_effectiveness': self._analyze_policy_effectiveness(),
                    'recommendations': self._generate_performance_recommendations(),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Check for performance degradation
                performance_alerts = self._check_performance_alerts(monitoring_results)
                if performance_alerts:
                    monitoring_results['alerts'] = performance_alerts
                
                self.logger.debug(f"Performance monitoring completed: efficiency={monitoring_results['coordination_efficiency']:.3f}")
                
                return monitoring_results
                
        except Exception as e:
            self.logger.error(f"Performance monitoring failed: {e}")
            raise RuntimeError(f"Performance monitoring failed: {e}")
    
    def handle_direction_switching_errors(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle direction switching errors.
        
        Args:
            error: Error that occurred
            context: Error context
            
        Returns:
            Error handling results
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If error handling fails
        """
        if error is None:
            raise ValueError("Error cannot be None")
        if not isinstance(error, Exception):
            raise TypeError("Error must be Exception")
        if not isinstance(context, dict):
            raise TypeError("Context must be dict")
        
        try:
            with self._coordination_lock:
                error_handling_results = {
                    'error_type': type(error).__name__,
                    'error_message': str(error),
                    'recovery_actions': [],
                    'fallback_strategy': None,
                    'coordination_mode_change': None,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Determine error severity
                error_severity = self._assess_error_severity(error, context)
                error_handling_results['error_severity'] = error_severity
                
                # Apply error recovery strategy
                if error_severity == 'critical':
                    recovery_actions = self._apply_critical_error_recovery(error, context)
                elif error_severity == 'high':
                    recovery_actions = self._apply_high_severity_recovery(error, context)
                else:
                    recovery_actions = self._apply_standard_recovery(error, context)
                
                error_handling_results['recovery_actions'] = recovery_actions
                
                # Switch to emergency mode if needed
                if error_severity == 'critical':
                    self.coordination_mode = CoordinationMode.EMERGENCY
                    error_handling_results['coordination_mode_change'] = 'emergency'
                    self.coordination_analytics['emergency_overrides'] += 1
                
                # Update error metrics
                self._update_error_metrics(error, error_severity)
                
                self.logger.warning(f"Error handled: {error_severity} severity, {len(recovery_actions)} recovery actions")
                
                return error_handling_results
                
        except Exception as e:
            self.logger.error(f"Error handling failed: {e}")
            raise RuntimeError(f"Error handling failed: {e}")
    
    def optimize_coordination_strategy(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimize coordination strategy based on performance.
        
        Args:
            performance_data: Performance data for optimization
            
        Returns:
            Optimization results
            
        Raises:
            ValueError: If performance data is invalid
            RuntimeError: If optimization fails
        """
        if not isinstance(performance_data, list):
            raise TypeError("Performance data must be list")
        if len(performance_data) < 10:
            raise ValueError("Insufficient performance data for optimization")
        
        try:
            optimization_results = {
                'strategy_changes': {},
                'policy_optimizations': {},
                'performance_improvements': {},
                'configuration_updates': {},
                'timestamp': datetime.utcnow().isoformat()
            }
            
            with self._coordination_lock:
                # Optimize coordination strategy
                strategy_optimization = self._optimize_strategy_selection(performance_data)
                optimization_results['strategy_changes'] = strategy_optimization
                
                # Optimize coordination policies
                policy_optimization = self._optimize_coordination_policies(performance_data)
                optimization_results['policy_optimizations'] = policy_optimization
                
                # Optimize configuration parameters
                config_optimization = self._optimize_configuration_parameters(performance_data)
                optimization_results['configuration_updates'] = config_optimization
                
                # Calculate performance improvements
                performance_improvements = self._calculate_optimization_improvements(
                    strategy_optimization, policy_optimization, config_optimization
                )
                optimization_results['performance_improvements'] = performance_improvements
                
                # Update last optimization time
                self.last_optimization = datetime.utcnow()
                
                self.logger.info(f"Coordination optimization completed: {sum(optimization_results['performance_improvements'].values()):.3f} total improvement")
                
                return optimization_results
                
        except Exception as e:
            self.logger.error(f"Coordination strategy optimization failed: {e}")
            raise RuntimeError(f"Coordination strategy optimization failed: {e}")
    
    def get_coordination_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive coordination analytics.
        
        Returns:
            Coordination analytics data
            
        Raises:
            RuntimeError: If analytics generation fails
        """
        try:
            with self._metrics_lock:
                analytics = {
                    'performance_metrics': {
                        'total_coordinations': self.performance_metrics.total_coordinations,
                        'successful_coordinations': self.performance_metrics.successful_coordinations,
                        'success_rate': self.performance_metrics.successful_coordinations / max(1, self.performance_metrics.total_coordinations),
                        'override_rate': self.performance_metrics.override_rate,
                        'average_coordination_time_ms': self.performance_metrics.average_coordination_time_ms,
                        'performance_improvement': self.performance_metrics.performance_improvement,
                        'error_rate': self.performance_metrics.error_rate,
                        'coordination_efficiency': self.performance_metrics.coordination_efficiency
                    },
                    'coordination_analytics': self.coordination_analytics.copy(),
                    'strategy_performance': {
                        strategy.name: {
                            'average_performance': statistics.mean(performances) if performances else 0.0,
                            'usage_count': len(performances),
                            'performance_trend': self._calculate_trend(performances[-10:]) if len(performances) >= 10 else 'insufficient_data'
                        }
                        for strategy, performances in self.strategy_performance.items()
                    },
                    'policy_summary': {
                        'total_policies': len(self.coordination_policies),
                        'active_policies': len(self.active_policies),
                        'policy_effectiveness': self._calculate_policy_effectiveness()
                    },
                    'recent_performance': self._get_recent_performance_summary(),
                    'coordination_trends': self._get_coordination_trends(),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                return analytics
                
        except Exception as e:
            self.logger.error(f"Coordination analytics generation failed: {e}")
            raise RuntimeError(f"Coordination analytics generation failed: {e}")
    
    def _apply_coordination_strategy(self, direction_state: DirectionState, context: Dict[str, Any], decision_id: str) -> CoordinationDecision:
        """Apply current coordination strategy"""
        if self.current_strategy == CoordinationStrategy.REACTIVE:
            return self._apply_reactive_strategy(direction_state, context, decision_id)
        elif self.current_strategy == CoordinationStrategy.PROACTIVE:
            return self._apply_proactive_strategy(direction_state, context, decision_id)
        elif self.current_strategy == CoordinationStrategy.PREDICTIVE:
            return self._apply_predictive_strategy(direction_state, context, decision_id)
        elif self.current_strategy == CoordinationStrategy.ADAPTIVE:
            return self._apply_adaptive_strategy(direction_state, context, decision_id)
        elif self.current_strategy == CoordinationStrategy.CONSERVATIVE:
            return self._apply_conservative_strategy(direction_state, context, decision_id)
        else:
            raise RuntimeError(f"Unknown coordination strategy: {self.current_strategy}")
    
    def _apply_reactive_strategy(self, direction_state: DirectionState, context: Dict[str, Any], decision_id: str) -> CoordinationDecision:
        """Apply reactive coordination strategy"""
        # React only when significant changes detected
        confidence_change = abs(direction_state.confidence - 0.5)
        stability_change = abs(direction_state.metadata.get('stability_score', 0.5) - 0.5)
        
        if confidence_change > 0.3 or stability_change > 0.3:
            # Significant change detected - coordinate
            coordination_actions = ['adjust_switching_threshold', 'update_confidence_weighting']
            override_decision = None
            reasoning = f"Reactive coordination triggered by significant changes (confidence: {confidence_change:.3f}, stability: {stability_change:.3f})"
            confidence = max(confidence_change, stability_change)
        else:
            # No significant changes - minimal coordination
            coordination_actions = []
            override_decision = None
            reasoning = "No significant changes detected - maintaining current strategy"
            confidence = 0.5
        
        return CoordinationDecision(
            decision_id=decision_id,
            coordination_strategy=CoordinationStrategy.REACTIVE,
            override_decision=override_decision,
            confidence=confidence,
            reasoning=reasoning,
            coordination_actions=coordination_actions,
            performance_impact={'expected_improvement': confidence * 0.1}
        )
    
    def _apply_proactive_strategy(self, direction_state: DirectionState, context: Dict[str, Any], decision_id: str) -> CoordinationDecision:
        """Apply proactive coordination strategy"""
        # Proactively adjust based on direction trends
        coordination_actions = ['optimize_switching_parameters', 'adjust_thresholds', 'update_policies']
        
        # Determine if override is needed
        override_decision = None
        if direction_state.confidence < 0.4 and direction_state.metadata.get('stability_score', 0.5) < 0.4:
            override_decision = 'use_pure'  # Conservative fallback
        elif direction_state.confidence > 0.8 and direction_state.metadata.get('stability_score', 0.5) > 0.7:
            override_decision = 'use_hybrid'  # Aggressive optimization
        
        reasoning = f"Proactive coordination based on direction metrics (confidence: {direction_state.confidence:.3f}, stability: {direction_state.metadata.get('stability_score', 0.5):.3f})"
        confidence = (direction_state.confidence + direction_state.metadata.get('stability_score', 0.5)) / 2.0
        
        # Update analytics
        self.coordination_analytics['proactive_actions'] += 1
        
        return CoordinationDecision(
            decision_id=decision_id,
            coordination_strategy=CoordinationStrategy.PROACTIVE,
            override_decision=override_decision,
            confidence=confidence,
            reasoning=reasoning,
            coordination_actions=coordination_actions,
            performance_impact={'expected_improvement': confidence * 0.15}
        )
    
    def _apply_predictive_strategy(self, direction_state: DirectionState, context: Dict[str, Any], decision_id: str) -> CoordinationDecision:
        """Apply predictive coordination strategy"""
        # Predict future performance and coordinate accordingly
        predicted_performance = self._predict_future_performance(direction_state, context)
        
        coordination_actions = ['predictive_threshold_adjustment', 'anticipatory_policy_update']
        
        # Determine override based on prediction
        override_decision = None
        if predicted_performance < 0.3:
            override_decision = 'use_pure'  # Predicted poor performance with hybrid
            coordination_actions.append('emergency_fallback_preparation')
        elif predicted_performance > 0.8:
            override_decision = 'use_hybrid'  # Predicted excellent performance with hybrid
        
        reasoning = f"Predictive coordination based on performance prediction: {predicted_performance:.3f}"
        confidence = predicted_performance
        
        # Update analytics
        self.coordination_analytics['predictive_actions'] += 1
        
        return CoordinationDecision(
            decision_id=decision_id,
            coordination_strategy=CoordinationStrategy.PREDICTIVE,
            override_decision=override_decision,
            confidence=confidence,
            reasoning=reasoning,
            coordination_actions=coordination_actions,
            performance_impact={'expected_improvement': predicted_performance * 0.20}
        )
    
    def _apply_adaptive_strategy(self, direction_state: DirectionState, context: Dict[str, Any], decision_id: str) -> CoordinationDecision:
        """Apply adaptive coordination strategy"""
        # Adapt based on recent performance and current conditions
        recent_performance = self._get_recent_performance_score()
        
        # Combine multiple coordination approaches
        reactive_weight = 0.3
        proactive_weight = 0.4
        predictive_weight = 0.3
        
        # Get sub-strategy decisions
        reactive_decision = self._apply_reactive_strategy(direction_state, context, f"{decision_id}_reactive")
        proactive_decision = self._apply_proactive_strategy(direction_state, context, f"{decision_id}_proactive")
        predictive_decision = self._apply_predictive_strategy(direction_state, context, f"{decision_id}_predictive")
        
        # Combine decisions
        combined_confidence = (
            reactive_decision.confidence * reactive_weight +
            proactive_decision.confidence * proactive_weight +
            predictive_decision.confidence * predictive_weight
        )
        
        # Combine actions
        coordination_actions = list(set(
            reactive_decision.coordination_actions +
            proactive_decision.coordination_actions +
            predictive_decision.coordination_actions
        ))
        
        # Choose override based on highest confidence
        override_decisions = [
            (reactive_decision.override_decision, reactive_decision.confidence * reactive_weight),
            (proactive_decision.override_decision, proactive_decision.confidence * proactive_weight),
            (predictive_decision.override_decision, predictive_decision.confidence * predictive_weight)
        ]
        
        override_decision = max(override_decisions, key=lambda x: x[1] if x[0] else 0)[0]
        
        reasoning = f"Adaptive coordination combining reactive ({reactive_decision.confidence:.2f}), proactive ({proactive_decision.confidence:.2f}), predictive ({predictive_decision.confidence:.2f})"
        
        return CoordinationDecision(
            decision_id=decision_id,
            coordination_strategy=CoordinationStrategy.ADAPTIVE,
            override_decision=override_decision,
            confidence=combined_confidence,
            reasoning=reasoning,
            coordination_actions=coordination_actions,
            performance_impact={'expected_improvement': combined_confidence * 0.18}
        )
    
    def _apply_conservative_strategy(self, direction_state: DirectionState, context: Dict[str, Any], decision_id: str) -> CoordinationDecision:
        """Apply conservative coordination strategy"""
        # Conservative approach - minimal changes, high confidence required for overrides
        coordination_actions = []
        override_decision = None
        
        # Only override with very high confidence
        if direction_state.confidence > 0.9 and direction_state.metadata.get('stability_score', 0.5) > 0.85:
            override_decision = 'use_hybrid'
            coordination_actions = ['conservative_hybrid_optimization']
            confidence = min(direction_state.confidence, direction_state.metadata.get('stability_score', 0.5))
        elif direction_state.confidence < 0.2 and direction_state.metadata.get('stability_score', 0.5) < 0.3:
            override_decision = 'use_pure'
            coordination_actions = ['conservative_pure_fallback']
            confidence = 1.0 - max(direction_state.confidence, direction_state.metadata.get('stability_score', 0.5))
        else:
            # No override - maintain current approach
            confidence = 0.5
        
        reasoning = f"Conservative coordination with high confidence thresholds (confidence: {direction_state.confidence:.3f}, stability: {direction_state.metadata.get('stability_score', 0.5):.3f})"
        
        return CoordinationDecision(
            decision_id=decision_id,
            coordination_strategy=CoordinationStrategy.CONSERVATIVE,
            override_decision=override_decision,
            confidence=confidence,
            reasoning=reasoning,
            coordination_actions=coordination_actions,
            performance_impact={'expected_improvement': confidence * 0.08}
        )
    
    def _check_policy_overrides(self, direction_state: DirectionState, context: Dict[str, Any], base_decision: CoordinationDecision) -> Optional[CoordinationDecision]:
        """Check for policy-based overrides"""
        for policy_id in self.active_policies:
            policy = self.coordination_policies.get(policy_id)
            if not policy or not policy.enabled:
                continue
            
            # Check if policy conditions are met
            if self._evaluate_policy_conditions(policy, direction_state, context):
                # Policy conditions met - apply override
                override_decision = CoordinationDecision(
                    decision_id=f"{base_decision.decision_id}_policy_{policy_id}",
                    coordination_strategy=policy.strategy,
                    override_decision=policy.actions[0] if policy.actions else None,
                    confidence=base_decision.confidence * 0.9,  # Slight confidence reduction for policy override
                    reasoning=f"Policy override: {policy.policy_name} - {base_decision.reasoning}",
                    coordination_actions=base_decision.coordination_actions + policy.actions,
                    performance_impact=base_decision.performance_impact
                )
                
                return override_decision
        
        return None
    
    def _evaluate_policy_conditions(self, policy: CoordinationPolicy, direction_state: DirectionState, context: Dict[str, Any]) -> bool:
        """Evaluate if policy conditions are met"""
        conditions = policy.conditions
        
        # Check confidence conditions
        if 'min_confidence' in conditions:
            if direction_state.confidence < conditions['min_confidence']:
                return False
        
        if 'max_confidence' in conditions:
            if direction_state.confidence > conditions['max_confidence']:
                return False
        
        # Check stability conditions
        if 'min_stability' in conditions:
            if direction_state.metadata.get('stability_score', 0.5) < conditions['min_stability']:
                return False
        
        if 'max_stability' in conditions:
            if direction_state.metadata.get('stability_score', 0.5) > conditions['max_stability']:
                return False
        
        # Check context conditions
        if 'required_context' in conditions:
            for key, value in conditions['required_context'].items():
                if context.get(key) != value:
                    return False
        
        return True
    
    def _initialize_default_policies(self) -> None:
        """Initialize default coordination policies"""
        # High confidence hybrid policy
        self.coordination_policies['high_confidence_hybrid'] = CoordinationPolicy(
            policy_id='high_confidence_hybrid',
            policy_name='High Confidence Hybrid',
            strategy=CoordinationStrategy.PROACTIVE,
            priority=CoordinationPriority.HIGH,
            conditions={'min_confidence': 0.8, 'min_stability': 0.7},
            actions=['use_hybrid', 'optimize_hybrid_parameters']
        )
        
        # Low confidence pure fallback policy
        self.coordination_policies['low_confidence_pure'] = CoordinationPolicy(
            policy_id='low_confidence_pure',
            policy_name='Low Confidence Pure Fallback',
            strategy=CoordinationStrategy.CONSERVATIVE,
            priority=CoordinationPriority.MEDIUM,
            conditions={'max_confidence': 0.3, 'max_stability': 0.4},
            actions=['use_pure', 'increase_analysis_window']
        )
        
        # Emergency stability policy
        self.coordination_policies['emergency_stability'] = CoordinationPolicy(
            policy_id='emergency_stability',
            policy_name='Emergency Stability',
            strategy=CoordinationStrategy.CONSERVATIVE,
            priority=CoordinationPriority.CRITICAL,
            conditions={'max_stability': 0.2},
            actions=['use_pure', 'emergency_stabilization']
        )
        
        # Activate default policies
        self.active_policies = ['high_confidence_hybrid', 'low_confidence_pure', 'emergency_stability']
    
    def _record_coordination(self, decision: CoordinationDecision, coordination_time_ms: float) -> None:
        """Record coordination for analytics"""
        coordination_record = {
            'decision_id': decision.decision_id,
            'strategy': decision.coordination_strategy.name,
            'override': decision.override_decision is not None,
            'confidence': decision.confidence,
            'coordination_time_ms': coordination_time_ms,
            'actions_count': len(decision.coordination_actions),
            'timestamp': decision.timestamp
        }
        
        self.coordination_history.append(coordination_record)
    
    def _update_coordination_metrics(self, decision: CoordinationDecision, coordination_time_ms: float) -> None:
        """Update coordination performance metrics"""
        self.performance_metrics.total_coordinations += 1
        
        # Consider coordination successful if confidence is high
        if decision.confidence >= 0.7:
            self.performance_metrics.successful_coordinations += 1
        
        # Update override rate
        if decision.override_decision:
            self.coordination_analytics['override_decisions'] += 1
        
        total_overrides = self.coordination_analytics['override_decisions']
        self.performance_metrics.override_rate = total_overrides / self.performance_metrics.total_coordinations
        
        # Update average coordination time
        total_time = self.performance_metrics.average_coordination_time_ms * (self.performance_metrics.total_coordinations - 1)
        total_time += coordination_time_ms
        self.performance_metrics.average_coordination_time_ms = total_time / self.performance_metrics.total_coordinations
        
        # Update strategy performance
        self.strategy_performance[decision.coordination_strategy].append(decision.confidence)
        
        # Update coordination efficiency
        self.performance_metrics.coordination_efficiency = self._calculate_coordination_efficiency()
    
    def _update_error_metrics(self, error: Exception, severity: str) -> None:
        """Update error metrics"""
        total_errors = getattr(self, '_total_errors', 0) + 1
        setattr(self, '_total_errors', total_errors)
        
        if self.performance_metrics.total_coordinations > 0:
            self.performance_metrics.error_rate = total_errors / self.performance_metrics.total_coordinations
    
    def _predict_future_performance(self, direction_state: DirectionState, context: Dict[str, Any]) -> float:
        """Predict future performance based on current state"""
        # Simple prediction based on current metrics and trends
        base_performance = (direction_state.confidence + direction_state.metadata.get('stability_score', 0.5)) / 2.0
        
        # Adjust based on recent performance trend
        recent_performance = self._get_recent_performance_score()
        trend_adjustment = (recent_performance - 0.5) * 0.2
        
        # Combine factors
        predicted_performance = base_performance + trend_adjustment
        
        return max(0.0, min(1.0, predicted_performance))
    
    def _get_recent_performance_score(self) -> float:
        """Get recent performance score"""
        if len(self.coordination_history) < 5:
            return 0.5  # Neutral if insufficient data
        
        recent_coordinations = list(self.coordination_history)[-10:]
        avg_confidence = sum(coord['confidence'] for coord in recent_coordinations) / len(recent_coordinations)
        
        return avg_confidence
    
    def _calculate_coordination_efficiency(self) -> float:
        """Calculate coordination efficiency"""
        if self.performance_metrics.total_coordinations == 0:
            return 0.0
        
        success_rate = self.performance_metrics.successful_coordinations / self.performance_metrics.total_coordinations
        time_efficiency = min(1.0, 50.0 / max(1.0, self.performance_metrics.average_coordination_time_ms))
        error_penalty = 1.0 - self.performance_metrics.error_rate
        
        efficiency = (success_rate * 0.5 + time_efficiency * 0.3 + error_penalty * 0.2)
        
        return max(0.0, min(1.0, efficiency))
    
    def _get_current_performance(self) -> Dict[str, float]:
        """Get current performance metrics"""
        return {
            'coordination_efficiency': self.performance_metrics.coordination_efficiency,
            'success_rate': self.performance_metrics.successful_coordinations / max(1, self.performance_metrics.total_coordinations),
            'override_rate': self.performance_metrics.override_rate,
            'error_rate': self.performance_metrics.error_rate,
            'average_response_time_ms': self.performance_metrics.average_coordination_time_ms
        }
    
    def _analyze_performance_trends(self) -> Dict[str, str]:
        """Analyze performance trends"""
        if len(self.coordination_history) < 10:
            return {'trend': 'insufficient_data'}
        
        recent_performance = [coord['confidence'] for coord in list(self.coordination_history)[-10:]]
        older_performance = [coord['confidence'] for coord in list(self.coordination_history)[-20:-10]] if len(self.coordination_history) >= 20 else []
        
        if not older_performance:
            return {'trend': 'insufficient_historical_data'}
        
        recent_avg = sum(recent_performance) / len(recent_performance)
        older_avg = sum(older_performance) / len(older_performance)
        
        if recent_avg > older_avg + 0.05:
            trend = 'improving'
        elif recent_avg < older_avg - 0.05:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'recent_average': recent_avg,
            'older_average': older_avg,
            'improvement': recent_avg - older_avg
        }
    
    def _analyze_strategy_performance(self) -> Dict[str, Any]:
        """Analyze performance by strategy"""
        strategy_analysis = {}
        
        for strategy, performances in self.strategy_performance.items():
            if len(performances) >= 3:
                strategy_analysis[strategy.name] = {
                    'average_performance': sum(performances) / len(performances),
                    'usage_count': len(performances),
                    'recent_trend': self._calculate_trend(performances[-5:]) if len(performances) >= 5 else 'insufficient_data'
                }
        
        return strategy_analysis
    
    def _analyze_policy_effectiveness(self) -> Dict[str, float]:
        """Analyze policy effectiveness"""
        policy_effectiveness = {}
        
        for policy_id, performances in self.policy_performance.items():
            if len(performances) >= 3:
                effectiveness = sum(performances) / len(performances)
                policy_effectiveness[policy_id] = effectiveness
        
        return policy_effectiveness
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Check coordination efficiency
        if self.performance_metrics.coordination_efficiency < 0.7:
            recommendations.append("Coordination efficiency is low - consider optimizing strategy selection")
        
        # Check error rate
        if self.performance_metrics.error_rate > self.config.error_tolerance_rate:
            recommendations.append("Error rate exceeds tolerance - review error handling policies")
        
        # Check response time
        if self.performance_metrics.average_coordination_time_ms > self.config.coordination_timeout_ms:
            recommendations.append("Average coordination time exceeds timeout - optimize coordination algorithms")
        
        return recommendations
    
    def _check_performance_alerts(self, monitoring_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for performance alerts"""
        alerts = []
        
        current_performance = monitoring_results['current_performance']
        
        # Check for critical performance degradation
        if current_performance['coordination_efficiency'] < 0.5:
            alerts.append({
                'type': 'critical_efficiency_degradation',
                'message': f"Coordination efficiency critically low: {current_performance['coordination_efficiency']:.3f}",
                'severity': 'critical'
            })
        
        # Check for high error rate
        if current_performance['error_rate'] > 0.1:
            alerts.append({
                'type': 'high_error_rate',
                'message': f"Error rate too high: {current_performance['error_rate']:.3f}",
                'severity': 'high'
            })
        
        return alerts
    
    def _assess_error_severity(self, error: Exception, context: Dict[str, Any]) -> str:
        """Assess error severity"""
        error_type = type(error).__name__
        
        # Critical errors
        if error_type in ['RuntimeError', 'SystemError', 'MemoryError']:
            return 'critical'
        
        # High severity errors
        if error_type in ['ValueError', 'TypeError', 'AttributeError']:
            return 'high'
        
        # Medium severity errors
        if error_type in ['KeyError', 'IndexError']:
            return 'medium'
        
        # Default to low severity
        return 'low'
    
    def _apply_critical_error_recovery(self, error: Exception, context: Dict[str, Any]) -> List[str]:
        """Apply critical error recovery actions"""
        return [
            'switch_to_conservative_strategy',
            'disable_problematic_policies',
            'increase_error_monitoring',
            'activate_emergency_mode'
        ]
    
    def _apply_high_severity_recovery(self, error: Exception, context: Dict[str, Any]) -> List[str]:
        """Apply high severity error recovery actions"""
        return [
            'adjust_coordination_parameters',
            'increase_validation_checks',
            'reduce_coordination_frequency'
        ]
    
    def _apply_standard_recovery(self, error: Exception, context: Dict[str, Any]) -> List[str]:
        """Apply standard error recovery actions"""
        return [
            'log_error_details',
            'continue_with_fallback',
            'monitor_error_recurrence'
        ]
    
    def _update_pattern_policies(self, direction_analysis: Dict[str, Any]) -> List[str]:
        """Update pattern-based policies"""
        # Placeholder implementation
        return ['pattern_policy_updated']
    
    def _update_stability_policies(self, direction_analysis: Dict[str, Any]) -> List[str]:
        """Update stability-based policies"""
        # Placeholder implementation
        return ['stability_policy_updated']
    
    def _update_performance_policies(self, direction_analysis: Dict[str, Any]) -> List[str]:
        """Update performance-based policies"""
        # Placeholder implementation
        return ['performance_policy_updated']
    
    def _create_adaptive_policies(self, direction_analysis: Dict[str, Any]) -> List[str]:
        """Create adaptive policies"""
        # Placeholder implementation
        return ['adaptive_policy_created']
    
    def _disable_underperforming_policies(self) -> List[str]:
        """Disable underperforming policies"""
        disabled = []
        
        for policy_id, performances in self.policy_performance.items():
            if len(performances) >= 5 and sum(performances) / len(performances) < 0.3:
                if policy_id in self.coordination_policies:
                    self.coordination_policies[policy_id].enabled = False
                    disabled.append(policy_id)
        
        return disabled
    
    def _calculate_policy_improvements(self) -> Dict[str, float]:
        """Calculate policy performance improvements"""
        improvements = {}
        
        for policy_id, performances in self.policy_performance.items():
            if len(performances) >= 10:
                recent_avg = sum(performances[-5:]) / 5
                older_avg = sum(performances[-10:-5]) / 5
                improvement = recent_avg - older_avg
                improvements[policy_id] = improvement
        
        return improvements
    
    def _optimize_strategy_selection(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize strategy selection"""
        # Analyze performance by strategy
        strategy_performance = defaultdict(list)
        
        for data in performance_data:
            strategy = data.get('strategy')
            performance = data.get('performance', 0.5)
            if strategy:
                strategy_performance[strategy].append(performance)
        
        # Find best performing strategy
        best_strategy = None
        best_performance = 0.0
        
        for strategy_name, performances in strategy_performance.items():
            if len(performances) >= 5:
                avg_performance = sum(performances) / len(performances)
                if avg_performance > best_performance:
                    best_performance = avg_performance
                    best_strategy = strategy_name
        
        # Update strategy if significantly better
        if best_strategy and best_performance > self._get_recent_performance_score() + 0.1:
            try:
                new_strategy = CoordinationStrategy(best_strategy)
                old_strategy = self.current_strategy
                self.current_strategy = new_strategy
                
                return {
                    'strategy_changed': True,
                    'old_strategy': old_strategy.name,
                    'new_strategy': new_strategy.name,
                    'performance_improvement': best_performance - self._get_recent_performance_score()
                }
            except ValueError:
                pass  # Invalid strategy name
        
        return {'strategy_changed': False}
    
    def _optimize_coordination_policies(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize coordination policies"""
        # Placeholder implementation
        return {'policies_optimized': 0}
    
    def _optimize_configuration_parameters(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize configuration parameters"""
        # Placeholder implementation
        return {'parameters_optimized': 0}
    
    def _calculate_optimization_improvements(self, strategy_opt: Dict, policy_opt: Dict, config_opt: Dict) -> Dict[str, float]:
        """Calculate optimization improvements"""
        improvements = {}
        
        if strategy_opt.get('strategy_changed'):
            improvements['strategy'] = strategy_opt.get('performance_improvement', 0.0)
        
        improvements['policy'] = policy_opt.get('improvement', 0.0)
        improvements['configuration'] = config_opt.get('improvement', 0.0)
        
        return improvements
    
    def _calculate_policy_effectiveness(self) -> float:
        """Calculate overall policy effectiveness"""
        if not self.policy_performance:
            return 0.5
        
        total_effectiveness = 0.0
        total_policies = 0
        
        for performances in self.policy_performance.values():
            if len(performances) >= 3:
                effectiveness = sum(performances) / len(performances)
                total_effectiveness += effectiveness
                total_policies += 1
        
        return total_effectiveness / max(1, total_policies)
    
    def _get_recent_performance_summary(self) -> Dict[str, Any]:
        """Get recent performance summary"""
        if len(self.coordination_history) < 5:
            return {'status': 'insufficient_data'}
        
        recent_records = list(self.coordination_history)[-10:]
        
        return {
            'average_confidence': sum(r['confidence'] for r in recent_records) / len(recent_records),
            'average_response_time_ms': sum(r['coordination_time_ms'] for r in recent_records) / len(recent_records),
            'override_rate': sum(1 for r in recent_records if r['override']) / len(recent_records),
            'strategy_distribution': self._calculate_strategy_distribution(recent_records)
        }
    
    def _get_coordination_trends(self) -> Dict[str, Any]:
        """Get coordination trends"""
        if len(self.coordination_history) < 10:
            return {'trend': 'insufficient_data'}
        
        # Calculate confidence trend
        confidences = [r['confidence'] for r in self.coordination_history]
        confidence_trend = self._calculate_trend(confidences[-10:])
        
        # Calculate response time trend
        response_times = [r['coordination_time_ms'] for r in self.coordination_history]
        response_time_trend = self._calculate_trend(response_times[-10:])
        
        return {
            'confidence_trend': confidence_trend,
            'response_time_trend': response_time_trend,
            'coordination_frequency': len(self.coordination_history) / max(1, (datetime.utcnow() - list(self.coordination_history)[0]['timestamp']).total_seconds() / 60)  # per minute
        }
    
    def _calculate_strategy_distribution(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate strategy distribution"""
        strategy_counts = defaultdict(int)
        
        for record in records:
            strategy_counts[record['strategy']] += 1
        
        total = len(records)
        return {strategy: count / total for strategy, count in strategy_counts.items()}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from values"""
        if len(values) < 3:
            return 'insufficient_data'
        
        # Simple trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg + 0.05:
            return 'improving'
        elif second_avg < first_avg - 0.05:
            return 'declining'
        else:
            return 'stable'
    
    def shutdown(self) -> None:
        """Shutdown direction coordinator"""
        self.logger.info("Shutting down direction coordinator")
        
        # Clear policies and tracking data
        self.coordination_policies.clear()
        self.active_policies.clear()
        self.coordination_history.clear()
        self.strategy_performance.clear()
        self.policy_performance.clear()
        
        self.logger.info("Direction coordinator shutdown complete")