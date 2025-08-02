"""
Brain Decision Engine - Advanced Decision Making System
Handles complex decision processes, multi-criteria analysis, and adaptive decision strategies
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock, Lock
import threading
from collections import defaultdict, deque
import json
import traceback
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    BINARY = "binary"
    MULTI_CHOICE = "multi_choice"
    CONTINUOUS = "continuous"
    RANKING = "ranking"
    OPTIMIZATION = "optimization"

class DecisionStrategy(Enum):
    UTILITY_MAXIMIZATION = "utility_maximization"
    RISK_MINIMIZATION = "risk_minimization"
    BALANCED = "balanced"
    CONSENSUS = "consensus"
    ADAPTIVE = "adaptive"

class DecisionConfidence(Enum):
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.9

class CriteriaType(Enum):
    BENEFIT = "benefit"  # Higher is better
    COST = "cost"       # Lower is better
    TARGET = "target"   # Closer to target is better

@dataclass
class DecisionCriteria:
    name: str
    weight: float
    criteria_type: CriteriaType
    target_value: Optional[float] = None
    min_acceptable: Optional[float] = None
    max_acceptable: Optional[float] = None
    importance: float = 1.0
    dynamic_weight: bool = False

@dataclass
class DecisionAlternative:
    alternative_id: str
    name: str
    description: str
    attributes: Dict[str, float]
    constraints: Dict[str, Any] = field(default_factory=dict)
    feasible: bool = True
    estimated_outcome: Optional[Dict[str, float]] = None

@dataclass
class DecisionContext:
    context_id: str
    decision_type: DecisionType
    strategy: DecisionStrategy
    criteria: List[DecisionCriteria]
    alternatives: List[DecisionAlternative]
    constraints: Dict[str, Any] = field(default_factory=dict)
    urgency: float = 0.5  # 0 = no urgency, 1 = maximum urgency
    risk_tolerance: float = 0.5  # 0 = risk averse, 1 = risk seeking
    time_limit: Optional[float] = None
    required_confidence: float = 0.7
    stakeholders: List[str] = field(default_factory=list)
    historical_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DecisionResult:
    decision_id: str
    context_id: str
    selected_alternative: Optional[str] = None
    ranking: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    reasoning: str = ""
    criteria_analysis: Dict[str, Dict[str, float]] = field(default_factory=dict)
    sensitivity_analysis: Dict[str, float] = field(default_factory=dict)
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class DecisionMethod(ABC):
    """Abstract base class for decision methods"""
    
    @abstractmethod
    def decide(self, context: DecisionContext) -> DecisionResult:
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        pass

class WeightedSumMethod(DecisionMethod):
    """Simple weighted sum decision method"""
    
    def decide(self, context: DecisionContext) -> DecisionResult:
        scores = {}
        criteria_analysis = {}
        
        for alternative in context.alternatives:
            if not alternative.feasible:
                continue
            
            total_score = 0.0
            alt_criteria_scores = {}
            
            for criteria in context.criteria:
                if criteria.name in alternative.attributes:
                    value = alternative.attributes[criteria.name]
                    
                    # Normalize value based on criteria type
                    if criteria.criteria_type == CriteriaType.BENEFIT:
                        normalized = value
                    elif criteria.criteria_type == CriteriaType.COST:
                        normalized = -value
                    else:  # TARGET
                        target = criteria.target_value or 0
                        normalized = -abs(value - target)
                    
                    weighted_score = normalized * criteria.weight
                    total_score += weighted_score
                    alt_criteria_scores[criteria.name] = weighted_score
            
            scores[alternative.alternative_id] = total_score
            criteria_analysis[alternative.alternative_id] = alt_criteria_scores
        
        # Select best alternative
        if scores:
            selected = max(scores.keys(), key=lambda x: scores[x])
            ranking = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            confidence = self._calculate_confidence(scores)
        else:
            selected = None
            ranking = []
            confidence = 0.0
        
        return DecisionResult(
            decision_id=f"ws_{int(time.time())}",
            context_id=context.context_id,
            selected_alternative=selected,
            ranking=ranking,
            scores=scores,
            confidence=confidence,
            reasoning="Weighted sum method: Selected alternative with highest weighted score",
            criteria_analysis=criteria_analysis
        )
    
    def _calculate_confidence(self, scores: Dict[str, float]) -> float:
        if len(scores) < 2:
            return 0.8
        
        sorted_scores = sorted(scores.values(), reverse=True)
        if sorted_scores[0] == sorted_scores[1]:
            return 0.3  # Tie, low confidence
        
        # Confidence based on gap between top two alternatives
        gap = (sorted_scores[0] - sorted_scores[1]) / max(abs(sorted_scores[0]), 1e-6)
        return min(0.95, 0.5 + gap * 0.5)
    
    def get_method_name(self) -> str:
        return "Weighted Sum"

class TOPSISMethod(DecisionMethod):
    """TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) method"""
    
    def decide(self, context: DecisionContext) -> DecisionResult:
        feasible_alternatives = [alt for alt in context.alternatives if alt.feasible]
        if not feasible_alternatives:
            return DecisionResult(
                decision_id=f"topsis_{int(time.time())}",
                context_id=context.context_id,
                confidence=0.0,
                reasoning="No feasible alternatives available"
            )
        
        # Build decision matrix
        criteria_names = [c.name for c in context.criteria]
        matrix = []
        
        for alt in feasible_alternatives:
            row = []
            for criteria in context.criteria:
                value = alt.attributes.get(criteria.name, 0.0)
                row.append(value)
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        # Normalize matrix
        normalized_matrix = self._normalize_matrix(matrix)
        
        # Apply weights
        weights = np.array([c.weight for c in context.criteria])
        weighted_matrix = normalized_matrix * weights
        
        # Determine ideal and negative-ideal solutions
        ideal_solution = []
        negative_ideal_solution = []
        
        for i, criteria in enumerate(context.criteria):
            if criteria.criteria_type == CriteriaType.BENEFIT:
                ideal_solution.append(np.max(weighted_matrix[:, i]))
                negative_ideal_solution.append(np.min(weighted_matrix[:, i]))
            else:  # COST or TARGET
                ideal_solution.append(np.min(weighted_matrix[:, i]))
                negative_ideal_solution.append(np.max(weighted_matrix[:, i]))
        
        ideal_solution = np.array(ideal_solution)
        negative_ideal_solution = np.array(negative_ideal_solution)
        
        # Calculate distances
        distances_to_ideal = []
        distances_to_negative_ideal = []
        
        for row in weighted_matrix:
            dist_ideal = np.sqrt(np.sum((row - ideal_solution) ** 2))
            dist_negative = np.sqrt(np.sum((row - negative_ideal_solution) ** 2))
            distances_to_ideal.append(dist_ideal)
            distances_to_negative_ideal.append(dist_negative)
        
        # Calculate TOPSIS scores
        scores = {}
        for i, alt in enumerate(feasible_alternatives):
            if distances_to_ideal[i] + distances_to_negative_ideal[i] == 0:
                score = 0.5
            else:
                score = distances_to_negative_ideal[i] / (distances_to_ideal[i] + distances_to_negative_ideal[i])
            scores[alt.alternative_id] = score
        
        # Select best alternative
        selected = max(scores.keys(), key=lambda x: scores[x])
        ranking = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        confidence = self._calculate_topsis_confidence(scores)
        
        return DecisionResult(
            decision_id=f"topsis_{int(time.time())}",
            context_id=context.context_id,
            selected_alternative=selected,
            ranking=ranking,
            scores=scores,
            confidence=confidence,
            reasoning="TOPSIS method: Selected alternative closest to ideal solution",
            metadata={'method': 'TOPSIS'}
        )
    
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize decision matrix using vector normalization"""
        normalized = np.zeros_like(matrix)
        for j in range(matrix.shape[1]):
            column_norm = np.sqrt(np.sum(matrix[:, j] ** 2))
            if column_norm > 0:
                normalized[:, j] = matrix[:, j] / column_norm
            else:
                normalized[:, j] = matrix[:, j]
        return normalized
    
    def _calculate_topsis_confidence(self, scores: Dict[str, float]) -> float:
        if len(scores) < 2:
            return 0.8
        
        sorted_scores = sorted(scores.values(), reverse=True)
        gap = sorted_scores[0] - sorted_scores[1]
        return min(0.95, 0.6 + gap * 0.4)
    
    def get_method_name(self) -> str:
        return "TOPSIS"

class AHPMethod(DecisionMethod):
    """Analytic Hierarchy Process method"""
    
    def decide(self, context: DecisionContext) -> DecisionResult:
        # Simplified AHP implementation
        # In a full implementation, this would use pairwise comparison matrices
        
        feasible_alternatives = [alt for alt in context.alternatives if alt.feasible]
        if not feasible_alternatives:
            return DecisionResult(
                decision_id=f"ahp_{int(time.time())}",
                context_id=context.context_id,
                confidence=0.0,
                reasoning="No feasible alternatives available"
            )
        
        scores = {}
        criteria_analysis = {}
        
        # Calculate scores using hierarchical structure
        for alternative in feasible_alternatives:
            total_score = 0.0
            alt_criteria_scores = {}
            
            for criteria in context.criteria:
                if criteria.name in alternative.attributes:
                    value = alternative.attributes[criteria.name]
                    
                    # Normalize and apply importance weighting
                    normalized_value = self._normalize_for_ahp(value, criteria)
                    weighted_score = normalized_value * criteria.weight * criteria.importance
                    
                    total_score += weighted_score
                    alt_criteria_scores[criteria.name] = weighted_score
            
            scores[alternative.alternative_id] = total_score
            criteria_analysis[alternative.alternative_id] = alt_criteria_scores
        
        # Select best alternative
        selected = max(scores.keys(), key=lambda x: scores[x])
        ranking = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        confidence = self._calculate_ahp_confidence(scores, context)
        
        return DecisionResult(
            decision_id=f"ahp_{int(time.time())}",
            context_id=context.context_id,
            selected_alternative=selected,
            ranking=ranking,
            scores=scores,
            confidence=confidence,
            reasoning="AHP method: Selected alternative using hierarchical decision structure",
            criteria_analysis=criteria_analysis,
            metadata={'method': 'AHP'}
        )
    
    def _normalize_for_ahp(self, value: float, criteria: DecisionCriteria) -> float:
        """Normalize value for AHP method"""
        if criteria.criteria_type == CriteriaType.BENEFIT:
            return value
        elif criteria.criteria_type == CriteriaType.COST:
            return 1.0 / (1.0 + value) if value > 0 else 1.0
        else:  # TARGET
            target = criteria.target_value or 0
            distance = abs(value - target)
            return 1.0 / (1.0 + distance)
    
    def _calculate_ahp_confidence(self, scores: Dict[str, float], context: DecisionContext) -> float:
        if len(scores) < 2:
            return 0.7
        
        # Consider criteria consistency and score separation
        sorted_scores = sorted(scores.values(), reverse=True)
        gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.5
        
        # Factor in number of criteria (more criteria = potentially higher confidence)
        criteria_factor = min(1.0, len(context.criteria) / 10.0)
        
        base_confidence = 0.5 + gap * 0.3 + criteria_factor * 0.2
        return min(0.95, base_confidence)
    
    def get_method_name(self) -> str:
        return "AHP"

class BrainDecisionEngine:
    def __init__(self, brain_instance=None, config: Optional[Dict] = None):
        self.brain = brain_instance
        self.config = config or {}
        
        # Core state management
        self._lock = RLock()
        self._decision_history: Dict[str, DecisionResult] = {}
        self._active_decisions: Dict[str, DecisionContext] = {}
        
        # Decision methods
        self._methods: Dict[str, DecisionMethod] = {
            'weighted_sum': WeightedSumMethod(),
            'topsis': TOPSISMethod(),
            'ahp': AHPMethod()
        }
        
        # Strategy mapping
        self._strategy_method_map = {
            DecisionStrategy.UTILITY_MAXIMIZATION: 'weighted_sum',
            DecisionStrategy.RISK_MINIMIZATION: 'topsis',
            DecisionStrategy.BALANCED: 'topsis',
            DecisionStrategy.CONSENSUS: 'ahp',
            DecisionStrategy.ADAPTIVE: 'adaptive'
        }
        
        # Learning components
        self._learning_enabled = self.config.get('learning_enabled', True)
        self._decision_outcomes: Dict[str, Dict[str, Any]] = {}
        self._performance_metrics: Dict[str, float] = {}
        
        # Multi-criteria analysis
        self._criteria_weights_history: Dict[str, List[float]] = defaultdict(list)
        self._alternative_performance_history: Dict[str, List[float]] = defaultdict(list)
        
        # Consensus building
        self._stakeholder_preferences: Dict[str, Dict[str, float]] = {}
        self._consensus_threshold = self.config.get('consensus_threshold', 0.8)
        
        # Risk analysis
        self._risk_models: Dict[str, Callable] = {}
        self._uncertainty_quantification = self.config.get('uncertainty_quantification', True)
        
        # Performance tracking
        self._decision_count = 0
        self._successful_decisions = 0
        self._average_confidence = 0.0
        self._average_execution_time = 0.0
        
        logger.info("BrainDecisionEngine initialized")
    
    def process(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for decision processing"""
        operation = parameters.get('operation', 'make_decision')
        
        if operation == 'make_decision':
            return self._make_decision(parameters)
        elif operation == 'analyze_alternatives':
            return self._analyze_alternatives(parameters)
        elif operation == 'sensitivity_analysis':
            return self._perform_sensitivity_analysis(parameters)
        elif operation == 'consensus_building':
            return self._build_consensus(parameters)
        elif operation == 'update_preferences':
            return self._update_stakeholder_preferences(parameters)
        elif operation == 'get_decision_history':
            return self._get_decision_history(parameters)
        else:
            logger.warning(f"Unknown operation: {operation}")
            return {"error": f"Unknown operation: {operation}"}
    
    def _make_decision(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Make a decision based on provided context"""
        try:
            start_time = time.time()
            
            # Parse decision context
            context = self._parse_decision_context(parameters)
            
            # Validate context
            if not self._validate_context(context):
                return {"error": "Invalid decision context"}
            
            # Store active decision
            with self._lock:
                self._active_decisions[context.context_id] = context
            
            # Select decision method
            method = self._select_decision_method(context)
            
            # Make decision
            result = method.decide(context)
            result.execution_time = time.time() - start_time
            
            # Post-process result
            self._post_process_result(result, context)
            
            # Store result
            with self._lock:
                self._decision_history[result.decision_id] = result
                if context.context_id in self._active_decisions:
                    del self._active_decisions[context.context_id]
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            # Learn from decision if enabled
            if self._learning_enabled:
                self._learn_from_decision(result, context)
            
            logger.info(f"Decision {result.decision_id} completed in {result.execution_time:.3f}s")
            
            return {
                "decision_id": result.decision_id,
                "selected_alternative": result.selected_alternative,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "ranking": result.ranking,
                "scores": result.scores,
                "execution_time": result.execution_time,
                "method_used": method.get_method_name()
            }
            
        except Exception as e:
            logger.error(f"Decision making failed: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def _parse_decision_context(self, parameters: Dict[str, Any]) -> DecisionContext:
        """Parse parameters into DecisionContext"""
        # Extract basic information
        context_id = parameters.get('context_id', f"ctx_{int(time.time())}")
        decision_type = DecisionType(parameters.get('decision_type', 'multi_choice'))
        strategy = DecisionStrategy(parameters.get('strategy', 'balanced'))
        
        # Parse criteria
        criteria = []
        criteria_data = parameters.get('criteria', [])
        for crit_data in criteria_data:
            criteria.append(DecisionCriteria(
                name=crit_data['name'],
                weight=crit_data.get('weight', 1.0),
                criteria_type=CriteriaType(crit_data.get('type', 'benefit')),
                target_value=crit_data.get('target_value'),
                min_acceptable=crit_data.get('min_acceptable'),
                max_acceptable=crit_data.get('max_acceptable'),
                importance=crit_data.get('importance', 1.0),
                dynamic_weight=crit_data.get('dynamic_weight', False)
            ))
        
        # Parse alternatives
        alternatives = []
        alternatives_data = parameters.get('alternatives', [])
        for alt_data in alternatives_data:
            alternatives.append(DecisionAlternative(
                alternative_id=alt_data['id'],
                name=alt_data.get('name', alt_data['id']),
                description=alt_data.get('description', ''),
                attributes=alt_data.get('attributes', {}),
                constraints=alt_data.get('constraints', {}),
                feasible=alt_data.get('feasible', True),
                estimated_outcome=alt_data.get('estimated_outcome')
            ))
        
        return DecisionContext(
            context_id=context_id,
            decision_type=decision_type,
            strategy=strategy,
            criteria=criteria,
            alternatives=alternatives,
            constraints=parameters.get('constraints', {}),
            urgency=parameters.get('urgency', 0.5),
            risk_tolerance=parameters.get('risk_tolerance', 0.5),
            time_limit=parameters.get('time_limit'),
            required_confidence=parameters.get('required_confidence', 0.7),
            stakeholders=parameters.get('stakeholders', []),
            historical_data=parameters.get('historical_data', {})
        )
    
    def _validate_context(self, context: DecisionContext) -> bool:
        """Validate decision context"""
        if not context.criteria:
            logger.error("No criteria provided")
            return False
        
        if not context.alternatives:
            logger.error("No alternatives provided")
            return False
        
        # Check that all alternatives have required criteria values
        required_criteria = {c.name for c in context.criteria}
        for alt in context.alternatives:
            if alt.feasible:
                missing_criteria = required_criteria - set(alt.attributes.keys())
                if missing_criteria:
                    logger.warning(f"Alternative {alt.alternative_id} missing criteria: {missing_criteria}")
        
        # Validate weights sum to reasonable value
        total_weight = sum(c.weight for c in context.criteria)
        if total_weight <= 0:
            logger.error("Total criteria weights must be positive")
            return False
        
        return True
    
    def _select_decision_method(self, context: DecisionContext) -> DecisionMethod:
        """Select appropriate decision method based on context"""
        if context.strategy == DecisionStrategy.ADAPTIVE:
            # Adaptive selection based on context characteristics
            if len(context.alternatives) <= 3:
                method_name = 'weighted_sum'
            elif len(context.criteria) > 5:
                method_name = 'topsis'
            else:
                method_name = 'ahp'
        else:
            method_name = self._strategy_method_map.get(context.strategy, 'weighted_sum')
        
        return self._methods[method_name]
    
    def _post_process_result(self, result: DecisionResult, context: DecisionContext):
        """Post-process decision result"""
        # Perform sensitivity analysis
        result.sensitivity_analysis = self._quick_sensitivity_analysis(result, context)
        
        # Perform risk assessment
        result.risk_assessment = self._assess_decision_risk(result, context)
        
        # Enhance reasoning
        result.reasoning = self._enhance_reasoning(result, context)
    
    def _quick_sensitivity_analysis(self, result: DecisionResult, context: DecisionContext) -> Dict[str, float]:
        """Perform quick sensitivity analysis"""
        sensitivity = {}
        
        if not result.selected_alternative or len(context.criteria) < 2:
            return sensitivity
        
        # Test weight changes for each criteria
        for criteria in context.criteria:
            original_weight = criteria.weight
            
            # Test 10% increase in weight
            criteria.weight *= 1.1
            temp_method = self._select_decision_method(context)
            temp_result = temp_method.decide(context)
            
            # Check if decision changes
            if temp_result.selected_alternative != result.selected_alternative:
                sensitivity[criteria.name] = 0.9  # High sensitivity
            else:
                # Measure score change
                if (result.selected_alternative in result.scores and
                    result.selected_alternative in temp_result.scores):
                    score_change = abs(temp_result.scores[result.selected_alternative] - 
                                     result.scores[result.selected_alternative])
                    sensitivity[criteria.name] = min(1.0, score_change)
                else:
                    sensitivity[criteria.name] = 0.1
            
            # Restore original weight
            criteria.weight = original_weight
        
        return sensitivity
    
    def _assess_decision_risk(self, result: DecisionResult, context: DecisionContext) -> Dict[str, float]:
        """Assess risk associated with the decision"""
        risk_assessment = {}
        
        if not result.selected_alternative:
            return {"overall_risk": 1.0}
        
        # Risk factors
        risk_assessment['confidence_risk'] = 1.0 - result.confidence
        risk_assessment['alternative_count_risk'] = min(1.0, 1.0 / len(context.alternatives))
        
        # Score spread risk (low spread = high risk)
        if len(result.scores) > 1:
            scores = list(result.scores.values())
            score_std = np.std(scores)
            score_range = max(scores) - min(scores)
            risk_assessment['score_spread_risk'] = 1.0 / (1.0 + score_range)
        else:
            risk_assessment['score_spread_risk'] = 0.5
        
        # Urgency risk
        risk_assessment['urgency_risk'] = context.urgency
        
        # Calculate overall risk
        weights = [0.4, 0.2, 0.2, 0.2]  # Weights for different risk factors
        overall_risk = sum(w * r for w, r in zip(weights, risk_assessment.values()))
        risk_assessment['overall_risk'] = overall_risk
        
        return risk_assessment
    
    def _enhance_reasoning(self, result: DecisionResult, context: DecisionContext) -> str:
        """Enhance the reasoning explanation"""
        reasoning_parts = [result.reasoning]
        
        if result.selected_alternative:
            selected_alt = next((alt for alt in context.alternatives 
                               if alt.alternative_id == result.selected_alternative), None)
            
            if selected_alt:
                reasoning_parts.append(f"Selected alternative: {selected_alt.name}")
                
                # Highlight key strengths
                if result.criteria_analysis and result.selected_alternative in result.criteria_analysis:
                    criteria_scores = result.criteria_analysis[result.selected_alternative]
                    best_criteria = max(criteria_scores.keys(), key=lambda x: criteria_scores[x])
                    reasoning_parts.append(f"Strongest criterion: {best_criteria}")
        
        # Add confidence interpretation
        if result.confidence >= 0.8:
            reasoning_parts.append("High confidence decision.")
        elif result.confidence >= 0.6:
            reasoning_parts.append("Moderate confidence decision.")
        else:
            reasoning_parts.append("Low confidence decision - consider gathering more information.")
        
        # Add risk warning if necessary
        if 'overall_risk' in result.risk_assessment and result.risk_assessment['overall_risk'] > 0.7:
            reasoning_parts.append("WARNING: High risk decision - careful monitoring recommended.")
        
        return " ".join(reasoning_parts)
    
    def _analyze_alternatives(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze alternatives without making a final decision"""
        try:
            context = self._parse_decision_context(parameters)
            
            if not self._validate_context(context):
                return {"error": "Invalid decision context"}
            
            analysis = {}
            
            # Analyze each alternative across all criteria
            for alternative in context.alternatives:
                if not alternative.feasible:
                    continue
                
                alt_analysis = {
                    'name': alternative.name,
                    'description': alternative.description,
                    'feasible': alternative.feasible,
                    'criteria_scores': {},
                    'strengths': [],
                    'weaknesses': [],
                    'overall_assessment': ''
                }
                
                # Score against each criteria
                for criteria in context.criteria:
                    if criteria.name in alternative.attributes:
                        value = alternative.attributes[criteria.name]
                        
                        # Normalize score
                        if criteria.criteria_type == CriteriaType.BENEFIT:
                            normalized_score = value
                            performance = "High" if value > 0.7 else "Medium" if value > 0.4 else "Low"
                        elif criteria.criteria_type == CriteriaType.COST:
                            normalized_score = 1.0 / (1.0 + value)
                            performance = "Low" if value < 0.3 else "Medium" if value < 0.6 else "High"
                        else:  # TARGET
                            target = criteria.target_value or 0.5
                            distance = abs(value - target)
                            normalized_score = 1.0 / (1.0 + distance)
                            performance = "Excellent" if distance < 0.1 else "Good" if distance < 0.3 else "Poor"
                        
                        alt_analysis['criteria_scores'][criteria.name] = {
                            'raw_value': value,
                            'normalized_score': normalized_score,
                            'performance': performance,
                            'weight': criteria.weight
                        }
                        
                        # Identify strengths and weaknesses
                        if normalized_score > 0.7:
                            alt_analysis['strengths'].append(criteria.name)
                        elif normalized_score < 0.3:
                            alt_analysis['weaknesses'].append(criteria.name)
                
                # Generate overall assessment
                strength_count = len(alt_analysis['strengths'])
                weakness_count = len(alt_analysis['weaknesses'])
                
                if strength_count > weakness_count:
                    alt_analysis['overall_assessment'] = "Strong candidate"
                elif weakness_count > strength_count:
                    alt_analysis['overall_assessment'] = "Weak candidate"
                else:
                    alt_analysis['overall_assessment'] = "Balanced candidate"
                
                analysis[alternative.alternative_id] = alt_analysis
            
            return {
                "analysis": analysis,
                "summary": self._generate_analysis_summary(analysis, context)
            }
            
        except Exception as e:
            logger.error(f"Alternative analysis failed: {e}")
            return {"error": str(e)}
    
    def _generate_analysis_summary(self, analysis: Dict[str, Any], context: DecisionContext) -> Dict[str, Any]:
        """Generate summary of alternative analysis"""
        summary = {
            'total_alternatives': len(analysis),
            'feasible_alternatives': len([a for a in analysis.values() if a['feasible']]),
            'criteria_count': len(context.criteria),
            'top_performers': {},
            'recommendations': []
        }
        
        # Identify top performers for each criterion
        for criteria in context.criteria:
            best_alt = None
            best_score = -float('inf')
            
            for alt_id, alt_analysis in analysis.items():
                if criteria.name in alt_analysis['criteria_scores']:
                    score = alt_analysis['criteria_scores'][criteria.name]['normalized_score']
                    if score > best_score:
                        best_score = score
                        best_alt = alt_id
            
            if best_alt:
                summary['top_performers'][criteria.name] = {
                    'alternative': best_alt,
                    'score': best_score
                }
        
        # Generate recommendations
        strong_candidates = [alt_id for alt_id, alt_data in analysis.items() 
                           if alt_data['overall_assessment'] == "Strong candidate"]
        
        if strong_candidates:
            summary['recommendations'].append(f"Consider focusing on: {', '.join(strong_candidates)}")
        
        weak_candidates = [alt_id for alt_id, alt_data in analysis.items() 
                          if alt_data['overall_assessment'] == "Weak candidate"]
        
        if weak_candidates:
            summary['recommendations'].append(f"May want to exclude: {', '.join(weak_candidates)}")
        
        return summary
    
    def _perform_sensitivity_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive sensitivity analysis"""
        try:
            decision_id = parameters.get('decision_id')
            if not decision_id or decision_id not in self._decision_history:
                return {"error": "Decision not found"}
            
            result = self._decision_history[decision_id]
            
            # Reconstruct context (this would need to be stored in practice)
            # For now, return the quick sensitivity analysis from the result
            
            return {
                "decision_id": decision_id,
                "sensitivity_analysis": result.sensitivity_analysis,
                "interpretation": self._interpret_sensitivity_analysis(result.sensitivity_analysis)
            }
            
        except Exception as e:
            logger.error(f"Sensitivity analysis failed: {e}")
            return {"error": str(e)}
    
    def _interpret_sensitivity_analysis(self, sensitivity: Dict[str, float]) -> Dict[str, Any]:
        """Interpret sensitivity analysis results"""
        interpretation = {
            'high_sensitivity_criteria': [],
            'low_sensitivity_criteria': [],
            'stability_assessment': '',
            'recommendations': []
        }
        
        for criteria, sensitivity_value in sensitivity.items():
            if sensitivity_value > 0.7:
                interpretation['high_sensitivity_criteria'].append(criteria)
            elif sensitivity_value < 0.3:
                interpretation['low_sensitivity_criteria'].append(criteria)
        
        # Generate stability assessment
        avg_sensitivity = sum(sensitivity.values()) / len(sensitivity) if sensitivity else 0
        
        if avg_sensitivity > 0.7:
            interpretation['stability_assessment'] = "Unstable - decision highly sensitive to weight changes"
            interpretation['recommendations'].append("Consider gathering more data or refining criteria weights")
        elif avg_sensitivity > 0.4:
            interpretation['stability_assessment'] = "Moderately stable - some sensitivity to weight changes"
            interpretation['recommendations'].append("Monitor key criteria closely")
        else:
            interpretation['stability_assessment'] = "Stable - decision robust to weight changes"
            interpretation['recommendations'].append("Decision appears reliable")
        
        return interpretation
    
    def _build_consensus(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Build consensus among stakeholders"""
        try:
            stakeholders = parameters.get('stakeholders', [])
            stakeholder_preferences = parameters.get('preferences', {})
            
            if not stakeholders or not stakeholder_preferences:
                return {"error": "Stakeholders and preferences required for consensus building"}
            
            # Store stakeholder preferences
            for stakeholder in stakeholders:
                if stakeholder in stakeholder_preferences:
                    self._stakeholder_preferences[stakeholder] = stakeholder_preferences[stakeholder]
            
            # Calculate consensus metrics
            consensus_metrics = self._calculate_consensus_metrics(stakeholder_preferences)
            
            # Generate consensus recommendations
            recommendations = self._generate_consensus_recommendations(consensus_metrics)
            
            return {
                "consensus_metrics": consensus_metrics,
                "recommendations": recommendations,
                "consensus_achieved": consensus_metrics['overall_consensus'] >= self._consensus_threshold
            }
            
        except Exception as e:
            logger.error(f"Consensus building failed: {e}")
            return {"error": str(e)}
    
    def _calculate_consensus_metrics(self, stakeholder_preferences: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate consensus metrics"""
        metrics = {
            'stakeholder_count': len(stakeholder_preferences),
            'criteria_consensus': {},
            'alternative_consensus': {},
            'overall_consensus': 0.0
        }
        
        # Calculate consensus for each criterion
        all_criteria = set()
        for prefs in stakeholder_preferences.values():
            all_criteria.update(prefs.get('criteria_weights', {}).keys())
        
        for criteria in all_criteria:
            weights = []
            for prefs in stakeholder_preferences.values():
                if criteria in prefs.get('criteria_weights', {}):
                    weights.append(prefs['criteria_weights'][criteria])
            
            if weights:
                weight_std = np.std(weights)
                consensus_score = 1.0 / (1.0 + weight_std)
                metrics['criteria_consensus'][criteria] = consensus_score
        
        # Calculate overall consensus
        if metrics['criteria_consensus']:
            metrics['overall_consensus'] = np.mean(list(metrics['criteria_consensus'].values()))
        
        return metrics
    
    def _generate_consensus_recommendations(self, consensus_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations for building consensus"""
        recommendations = []
        
        overall_consensus = consensus_metrics['overall_consensus']
        
        if overall_consensus >= self._consensus_threshold:
            recommendations.append("Strong consensus achieved - proceed with decision")
        elif overall_consensus >= 0.6:
            recommendations.append("Moderate consensus - consider minor adjustments")
        else:
            recommendations.append("Low consensus - significant discussion needed")
        
        # Identify problematic criteria
        low_consensus_criteria = [
            criteria for criteria, score in consensus_metrics['criteria_consensus'].items()
            if score < 0.5
        ]
        
        if low_consensus_criteria:
            recommendations.append(f"Focus discussion on: {', '.join(low_consensus_criteria)}")
        
        return recommendations
    
    def _update_stakeholder_preferences(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Update stakeholder preferences"""
        stakeholder = parameters.get('stakeholder')
        preferences = parameters.get('preferences', {})
        
        if not stakeholder:
            return {"error": "Stakeholder identifier required"}
        
        self._stakeholder_preferences[stakeholder] = preferences
        
        return {
            "stakeholder": stakeholder,
            "preferences_updated": True,
            "total_stakeholders": len(self._stakeholder_preferences)
        }
    
    def _get_decision_history(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get decision history and analytics"""
        limit = parameters.get('limit', 10)
        include_details = parameters.get('include_details', False)
        
        # Get recent decisions
        recent_decisions = list(self._decision_history.values())[-limit:]
        
        if include_details:
            decision_list = [
                {
                    'decision_id': result.decision_id,
                    'context_id': result.context_id,
                    'selected_alternative': result.selected_alternative,
                    'confidence': result.confidence,
                    'execution_time': result.execution_time,
                    'timestamp': result.timestamp,
                    'reasoning': result.reasoning
                }
                for result in recent_decisions
            ]
        else:
            decision_list = [
                {
                    'decision_id': result.decision_id,
                    'selected_alternative': result.selected_alternative,
                    'confidence': result.confidence,
                    'timestamp': result.timestamp
                }
                for result in recent_decisions
            ]
        
        # Calculate performance statistics
        if recent_decisions:
            avg_confidence = sum(r.confidence for r in recent_decisions) / len(recent_decisions)
            avg_execution_time = sum(r.execution_time for r in recent_decisions) / len(recent_decisions)
        else:
            avg_confidence = 0.0
            avg_execution_time = 0.0
        
        return {
            "decisions": decision_list,
            "statistics": {
                "total_decisions": len(self._decision_history),
                "recent_decisions": len(recent_decisions),
                "average_confidence": avg_confidence,
                "average_execution_time": avg_execution_time,
                "success_rate": self._successful_decisions / max(1, self._decision_count)
            }
        }
    
    def _learn_from_decision(self, result: DecisionResult, context: DecisionContext):
        """Learn from decision outcomes for future improvement"""
        # Store decision outcome for learning
        self._decision_outcomes[result.decision_id] = {
            'context': context,
            'result': result,
            'timestamp': time.time()
        }
        
        # Update criteria weight learning
        if result.selected_alternative and result.confidence > 0.7:
            for criteria in context.criteria:
                self._criteria_weights_history[criteria.name].append(criteria.weight)
        
        # Update performance tracking
        self._decision_count += 1
        if result.confidence >= context.required_confidence:
            self._successful_decisions += 1
    
    def _update_performance_metrics(self, result: DecisionResult):
        """Update performance metrics"""
        self._average_confidence = (
            (self._average_confidence * self._decision_count + result.confidence) / 
            (self._decision_count + 1)
        )
        
        self._average_execution_time = (
            (self._average_execution_time * self._decision_count + result.execution_time) /
            (self._decision_count + 1)
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            'decision_count': self._decision_count,
            'successful_decisions': self._successful_decisions,
            'success_rate': self._successful_decisions / max(1, self._decision_count),
            'average_confidence': self._average_confidence,
            'average_execution_time': self._average_execution_time,
            'active_decisions': len(self._active_decisions),
            'learning_enabled': self._learning_enabled,
            'available_methods': list(self._methods.keys()),
            'stakeholder_count': len(self._stakeholder_preferences)
        }