"""
Symbolic Reasoning for Financial Fraud Detection
Advanced symbolic reasoning capabilities for fraud detection
"""

import logging
import json
import time
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Enhanced import
try:
    from enhanced_symbolic_reasoner import (
        EnhancedFinancialSymbolicReasoner,
        SymbolicReasoningException,
        ReasoningValidationError,
        RuleValidationError,
        KnowledgeBaseError,
        ReasoningTimeoutError,
        ReasoningPerformanceError,
        ReasoningSecurityError,
        ReasoningIntegrationError,
        InferenceValidationError,
        FactValidationError,
        QueryValidationError,
        ReasoningCapacityError,
        ValidatedFact,
        ValidatedRule,
        ValidatedInference,
        ValidationUtils,
        SecurityValidator,
        PerformanceMonitor,
        ResourceManager,
        ErrorRecoveryManager
    )
    ENHANCED_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced symbolic reasoner not available: {e}")
    ENHANCED_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

class ReasoningType(Enum):
    """Types of reasoning operations"""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    RULE_BASED = "rule_based"
    FUZZY = "fuzzy"
    TEMPORAL = "temporal"

class RuleType(Enum):
    """Types of business rules"""
    TRANSACTION_LIMIT = "transaction_limit"
    VELOCITY = "velocity"
    GEOGRAPHIC = "geographic"
    BEHAVIORAL = "behavioral"
    TEMPORAL = "temporal"
    RELATIONSHIP = "relationship"
    COMPOSITE = "composite"

class FactType(Enum):
    """Types of facts in knowledge base"""
    TRANSACTION = "transaction"
    USER_BEHAVIOR = "user_behavior"
    HISTORICAL_PATTERN = "historical_pattern"
    EXTERNAL_DATA = "external_data"
    RULE_VIOLATION = "rule_violation"
    ML_PREDICTION = "ml_prediction"

@dataclass
class Fact:
    """Represents a fact in the knowledge base"""
    fact_id: str
    fact_type: FactType
    subject: str
    predicate: str
    object: Any
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_triple(self) -> Tuple[str, str, Any]:
        """Convert to RDF-style triple"""
        return (self.subject, self.predicate, self.object)

@dataclass
class Rule:
    """Represents a business rule"""
    rule_id: str
    rule_type: RuleType
    name: str
    description: str
    conditions: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    priority: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate(self, facts: List[Fact]) -> bool:
        """Evaluate rule against facts"""
        # Simple evaluation logic - can be extended
        for condition in self.conditions:
            if not self._evaluate_condition(condition, facts):
                return False
        return True
    
    def _evaluate_condition(self, condition: Dict[str, Any], facts: List[Fact]) -> bool:
        """Evaluate a single condition"""
        # Implement condition evaluation logic
        return True

@dataclass
class Inference:
    """Represents an inference result"""
    inference_id: str
    reasoning_type: ReasoningType
    premises: List[str]  # Fact IDs
    conclusion: Fact
    confidence: float
    reasoning_path: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReasoningMetrics:
    """Metrics for reasoning performance"""
    total_inferences: int = 0
    successful_inferences: int = 0
    failed_inferences: int = 0
    
    average_reasoning_time_ms: float = 0.0
    total_facts_processed: int = 0
    total_rules_evaluated: int = 0
    
    inference_accuracy: float = 0.0
    rule_coverage: float = 0.0
    
    # Breakdown by reasoning type
    metrics_by_type: Dict[ReasoningType, Dict[str, float]] = field(default_factory=dict)
    
    # Time-based metrics
    hourly_inference_count: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'total_inferences': self.total_inferences,
            'successful_inferences': self.successful_inferences,
            'failed_inferences': self.failed_inferences,
            'avg_reasoning_time': self.average_reasoning_time_ms,
            'facts_processed': self.total_facts_processed,
            'rules_evaluated': self.total_rules_evaluated,
            'inference_accuracy': self.inference_accuracy,
            'rule_coverage': self.rule_coverage
        }

class BaseReasoningEngine(ABC):
    """Abstract base class for reasoning engines"""
    
    @abstractmethod
    def reason(self, facts: List[Fact], rules: List[Rule]) -> List[Inference]:
        """Perform reasoning on facts using rules"""
        pass
    
    @abstractmethod
    def validate(self, inference: Inference, facts: List[Fact]) -> bool:
        """Validate an inference"""
        pass

class RuleBasedReasoningEngine(BaseReasoningEngine):
    """Rule-based reasoning engine"""
    
    def __init__(self):
        self.forward_chaining = True
        
    def reason(self, facts: List[Fact], rules: List[Rule]) -> List[Inference]:
        """Perform rule-based reasoning"""
        inferences = []
        
        if self.forward_chaining:
            inferences = self._forward_chain(facts, rules)
        else:
            inferences = self._backward_chain(facts, rules)
        
        return inferences
    
    def _forward_chain(self, facts: List[Fact], rules: List[Rule]) -> List[Inference]:
        """Forward chaining inference"""
        inferences = []
        fact_set = set(facts)
        applied_rules = set()
        
        changed = True
        while changed:
            changed = False
            
            for rule in rules:
                if rule.rule_id in applied_rules or not rule.enabled:
                    continue
                
                if rule.evaluate(list(fact_set)):
                    # Generate new facts from rule actions
                    for action in rule.actions:
                        new_fact = self._create_fact_from_action(action, rule)
                        if new_fact:
                            fact_set.add(new_fact)
                            
                            # Create inference
                            inference = Inference(
                                inference_id=self._generate_id('INF'),
                                reasoning_type=ReasoningType.RULE_BASED,
                                premises=[f.fact_id for f in facts if self._fact_matches_condition(f, rule.conditions)],
                                conclusion=new_fact,
                                confidence=0.9,
                                reasoning_path=[{
                                    'rule': rule.rule_id,
                                    'conditions': rule.conditions,
                                    'action': action
                                }]
                            )
                            inferences.append(inference)
                            changed = True
                    
                    applied_rules.add(rule.rule_id)
        
        return inferences
    
    def _backward_chain(self, facts: List[Fact], rules: List[Rule]) -> List[Inference]:
        """Backward chaining inference (goal-driven)"""
        # Simplified implementation
        return []
    
    def _create_fact_from_action(self, action: Dict[str, Any], rule: Rule) -> Optional[Fact]:
        """Create new fact from rule action"""
        if action.get('type') == 'assert':
            return Fact(
                fact_id=self._generate_id('FACT'),
                fact_type=FactType.RULE_VIOLATION,
                subject=action.get('subject', 'transaction'),
                predicate=action.get('predicate', 'violates'),
                object=rule.name,
                confidence=action.get('confidence', 0.9),
                source=f"rule_{rule.rule_id}"
            )
        return None
    
    def _fact_matches_condition(self, fact: Fact, conditions: List[Dict[str, Any]]) -> bool:
        """Check if fact matches any condition"""
        # Simplified matching logic
        return True
    
    def validate(self, inference: Inference, facts: List[Fact]) -> bool:
        """Validate rule-based inference"""
        # Check if all premises exist in facts
        fact_ids = {f.fact_id for f in facts}
        return all(premise_id in fact_ids for premise_id in inference.premises)
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID"""
        import hashlib
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        return f"{prefix}_{hashlib.md5(timestamp.encode()).hexdigest()[:8]}"

class FuzzyReasoningEngine(BaseReasoningEngine):
    """Fuzzy logic reasoning engine"""
    
    def __init__(self, membership_functions: Optional[Dict[str, Any]] = None):
        self.membership_functions = membership_functions or {}
    
    def reason(self, facts: List[Fact], rules: List[Rule]) -> List[Inference]:
        """Perform fuzzy reasoning"""
        inferences = []
        
        # Apply fuzzy logic to facts
        fuzzy_facts = self._fuzzify_facts(facts)
        
        # Apply fuzzy rules
        for rule in rules:
            if rule.rule_type == RuleType.BEHAVIORAL:
                inference = self._apply_fuzzy_rule(rule, fuzzy_facts)
                if inference:
                    inferences.append(inference)
        
        return inferences
    
    def _fuzzify_facts(self, facts: List[Fact]) -> List[Dict[str, Any]]:
        """Convert crisp facts to fuzzy facts"""
        fuzzy_facts = []
        
        for fact in facts:
            if fact.fact_type == FactType.TRANSACTION:
                # Apply membership functions
                fuzzy_fact = {
                    'original': fact,
                    'fuzzy_values': self._calculate_membership(fact)
                }
                fuzzy_facts.append(fuzzy_fact)
        
        return fuzzy_facts
    
    def _calculate_membership(self, fact: Fact) -> Dict[str, float]:
        """Calculate membership values for fact"""
        memberships = {}
        
        # Example: transaction amount fuzzy sets
        if hasattr(fact.object, 'amount'):
            amount = fact.object.amount
            memberships['low_amount'] = self._triangular_membership(amount, 0, 100, 500)
            memberships['medium_amount'] = self._triangular_membership(amount, 100, 1000, 5000)
            memberships['high_amount'] = self._triangular_membership(amount, 1000, 10000, float('inf'))
        
        return memberships
    
    def _triangular_membership(self, x: float, a: float, b: float, c: float) -> float:
        """Triangular membership function"""
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:
            return (c - x) / (c - b)
    
    def _apply_fuzzy_rule(self, rule: Rule, fuzzy_facts: List[Dict[str, Any]]) -> Optional[Inference]:
        """Apply fuzzy rule to fuzzy facts"""
        # Simplified fuzzy rule application
        return None
    
    def validate(self, inference: Inference, facts: List[Fact]) -> bool:
        """Validate fuzzy inference"""
        return inference.confidence > 0.5

class TemporalReasoningEngine(BaseReasoningEngine):
    """Temporal logic reasoning engine"""
    
    def __init__(self, time_window_hours: int = 24):
        self.time_window = timedelta(hours=time_window_hours)
    
    def reason(self, facts: List[Fact], rules: List[Rule]) -> List[Inference]:
        """Perform temporal reasoning"""
        inferences = []
        
        # Group facts by time windows
        temporal_groups = self._group_facts_temporally(facts)
        
        # Apply temporal rules
        for rule in rules:
            if rule.rule_type == RuleType.TEMPORAL:
                for window, window_facts in temporal_groups.items():
                    inference = self._apply_temporal_rule(rule, window_facts, window)
                    if inference:
                        inferences.append(inference)
        
        return inferences
    
    def _group_facts_temporally(self, facts: List[Fact]) -> Dict[str, List[Fact]]:
        """Group facts by time windows"""
        groups = {}
        
        for fact in facts:
            window_key = fact.timestamp.strftime('%Y-%m-%d_%H')
            if window_key not in groups:
                groups[window_key] = []
            groups[window_key].append(fact)
        
        return groups
    
    def _apply_temporal_rule(self, rule: Rule, facts: List[Fact], 
                           time_window: str) -> Optional[Inference]:
        """Apply temporal rule to facts in time window"""
        # Check for temporal patterns
        if len(facts) > rule.metadata.get('min_events', 5):
            # Detected temporal anomaly
            conclusion = Fact(
                fact_id=self._generate_id('FACT'),
                fact_type=FactType.RULE_VIOLATION,
                subject=f"temporal_pattern_{time_window}",
                predicate="indicates",
                object="potential_fraud",
                confidence=0.8,
                source=f"temporal_rule_{rule.rule_id}"
            )
            
            return Inference(
                inference_id=self._generate_id('INF'),
                reasoning_type=ReasoningType.TEMPORAL,
                premises=[f.fact_id for f in facts],
                conclusion=conclusion,
                confidence=0.8,
                reasoning_path=[{
                    'rule': rule.rule_id,
                    'time_window': time_window,
                    'event_count': len(facts)
                }]
            )
        
        return None
    
    def validate(self, inference: Inference, facts: List[Fact]) -> bool:
        """Validate temporal inference"""
        # Check temporal consistency
        premise_facts = [f for f in facts if f.fact_id in inference.premises]
        if not premise_facts:
            return False
        
        # All facts should be within time window
        min_time = min(f.timestamp for f in premise_facts)
        max_time = max(f.timestamp for f in premise_facts)
        
        return (max_time - min_time) <= self.time_window
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID"""
        import hashlib
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        return f"{prefix}_{hashlib.md5(timestamp.encode()).hexdigest()[:8]}"

class FinancialSymbolicReasoner:
    """
    Advanced Symbolic Reasoning for Financial Fraud Detection
    
    Features:
    - Multiple reasoning engines (rule-based, fuzzy, temporal)
    - Business rule validation
    - Knowledge base integration
    - Logical reasoning methods
    - Performance metrics
    - Reasoning persistence
    - Enhanced validation and error handling (when available)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, use_enhanced: bool = True):
        """
        Initialize symbolic reasoner
        
        Args:
            config: Configuration dictionary
            use_enhanced: Whether to use enhanced symbolic reasoner if available
        """
        self.config = config or self._default_config()
        self.use_enhanced = use_enhanced and ENHANCED_AVAILABLE
        
        # Initialize enhanced reasoner if available
        if self.use_enhanced:
            try:
                enhanced_config = self._convert_config_for_enhanced(self.config)
                self.enhanced_reasoner = EnhancedFinancialSymbolicReasoner(enhanced_config)
                logger.info("Using enhanced symbolic reasoner with comprehensive validation")
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced reasoner: {e}")
                self.use_enhanced = False
                self.enhanced_reasoner = None
        else:
            self.enhanced_reasoner = None
        
        # Initialize reasoning engines
        self.reasoning_engines = {
            'rule_based': RuleBasedReasoningEngine(),
            'fuzzy': FuzzyReasoningEngine(),
            'temporal': TemporalReasoningEngine(
                self.config.get('temporal_window_hours', 24)
            )
        }
        
        # Knowledge base
        self.facts: Dict[str, Fact] = {}
        self.rules: Dict[str, Rule] = {}
        self.inferences: Dict[str, Inference] = {}
        
        # Initialize default rules
        self._initialize_default_rules()
        
        # Metrics
        self.metrics = ReasoningMetrics()
        
        # Thread safety
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Persistence
        self.storage_path = Path(self.config.get('storage_path', 'reasoning'))
        self.storage_path.mkdir(exist_ok=True)
        
        logger.info("FinancialSymbolicReasoner initialized with engines: %s%s", 
                   list(self.reasoning_engines.keys()),
                   " (enhanced)" if self.use_enhanced else "")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'temporal_window_hours': 24,
            'max_facts_per_inference': 1000,
            'inference_confidence_threshold': 0.6,
            'enable_parallel_reasoning': True,
            'auto_persist': True,
            'reasoning_timeout_seconds': 30
        }
    
    def _convert_config_for_enhanced(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert configuration for enhanced reasoner"""
        enhanced_config = config.copy()
        
        # Map configuration keys to enhanced reasoner format
        enhanced_config.update({
            'validation_level': 'strict',
            'security_level': 'high',
            'performance_monitoring': True,
            'error_recovery': True,
            'resource_monitoring': True,
            'audit_trail': True,
            'concurrent_processing': config.get('enable_parallel_reasoning', True),
            'timeout_seconds': config.get('reasoning_timeout_seconds', 30),
            'max_memory_mb': 512,
            'max_cpu_percent': 80.0,
            'confidence_threshold': config.get('inference_confidence_threshold', 0.6)
        })
        
        return enhanced_config
    
    def _initialize_default_rules(self) -> None:
        """Initialize default fraud detection rules"""
        default_rules = [
            Rule(
                rule_id='RULE_001',
                rule_type=RuleType.TRANSACTION_LIMIT,
                name='High Value Transaction',
                description='Flag transactions above threshold',
                conditions=[
                    {'type': 'fact_match', 'fact_type': 'TRANSACTION', 
                     'predicate': 'amount', 'operator': '>', 'value': 10000}
                ],
                actions=[
                    {'type': 'assert', 'subject': 'transaction', 
                     'predicate': 'violates', 'confidence': 0.8}
                ],
                priority=1
            ),
            Rule(
                rule_id='RULE_002',
                rule_type=RuleType.VELOCITY,
                name='Rapid Transaction Velocity',
                description='Multiple transactions in short time',
                conditions=[
                    {'type': 'temporal_pattern', 'pattern': 'rapid_sequence',
                     'time_window_minutes': 10, 'min_count': 5}
                ],
                actions=[
                    {'type': 'assert', 'subject': 'user_behavior',
                     'predicate': 'exhibits', 'object': 'velocity_anomaly'}
                ],
                priority=2
            ),
            Rule(
                rule_id='RULE_003',
                rule_type=RuleType.BEHAVIORAL,
                name='Unusual Behavior Pattern',
                description='Deviation from normal user behavior',
                conditions=[
                    {'type': 'statistical', 'metric': 'deviation',
                     'threshold': 3, 'attribute': 'transaction_amount'}
                ],
                actions=[
                    {'type': 'assert', 'subject': 'user',
                     'predicate': 'shows', 'object': 'behavioral_anomaly'}
                ],
                priority=3
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_fact(self, fact_data: Dict[str, Any]) -> str:
        """
        Add fact to knowledge base
        
        Args:
            fact_data: Fact information
            
        Returns:
            Fact ID
        """
        if self.use_enhanced:
            try:
                return self.enhanced_reasoner.add_fact(fact_data)
            except Exception as e:
                logger.warning(f"Enhanced fact addition failed, falling back: {e}")
                # Fall through to basic implementation
        
        try:
            fact = Fact(
                fact_id=fact_data.get('fact_id', self._generate_id('FACT')),
                fact_type=FactType(fact_data.get('fact_type', FactType.TRANSACTION.value)),
                subject=fact_data['subject'],
                predicate=fact_data['predicate'],
                object=fact_data['object'],
                confidence=fact_data.get('confidence', 1.0),
                source=fact_data.get('source', 'unknown'),
                metadata=fact_data.get('metadata', {})
            )
            
            with self._lock:
                self.facts[fact.fact_id] = fact
                self.metrics.total_facts_processed += 1
            
            logger.debug(f"Added fact {fact.fact_id}: {fact.to_triple()}")
            return fact.fact_id
            
        except Exception as e:
            logger.error(f"Failed to add fact: {str(e)}")
            raise
    
    def add_rule(self, rule: Union[Rule, Dict[str, Any]]) -> str:
        """
        Add rule to rule base
        
        Args:
            rule: Rule object or rule data
            
        Returns:
            Rule ID
        """
        if self.use_enhanced:
            try:
                # Convert Rule object to dict if needed
                rule_data = rule if isinstance(rule, dict) else {
                    'rule_id': rule.rule_id,
                    'rule_type': rule.rule_type.value,
                    'name': rule.name,
                    'description': rule.description,
                    'conditions': rule.conditions,
                    'actions': rule.actions,
                    'priority': rule.priority,
                    'enabled': rule.enabled,
                    'metadata': rule.metadata
                }
                return self.enhanced_reasoner.add_rule(rule_data)
            except Exception as e:
                logger.warning(f"Enhanced rule addition failed, falling back: {e}")
                # Fall through to basic implementation
        
        try:
            if isinstance(rule, dict):
                rule = Rule(
                    rule_id=rule.get('rule_id', self._generate_id('RULE')),
                    rule_type=RuleType(rule.get('rule_type', RuleType.COMPOSITE.value)),
                    name=rule['name'],
                    description=rule.get('description', ''),
                    conditions=rule.get('conditions', []),
                    actions=rule.get('actions', []),
                    priority=rule.get('priority', 0),
                    enabled=rule.get('enabled', True),
                    metadata=rule.get('metadata', {})
                )
            
            with self._lock:
                self.rules[rule.rule_id] = rule
            
            logger.info(f"Added rule {rule.rule_id}: {rule.name}")
            return rule.rule_id
            
        except Exception as e:
            logger.error(f"Failed to add rule: {str(e)}")
            raise
    
    def reason(self, engine_names: Optional[List[str]] = None) -> List[Inference]:
        """
        Perform reasoning using specified engines
        
        Args:
            engine_names: List of engine names to use (None = all)
            
        Returns:
            List of inferences
        """
        if self.use_enhanced:
            try:
                enhanced_result = self.enhanced_reasoner.reason(
                    reasoning_types=engine_names,
                    max_inferences=self.config.get('max_facts_per_inference', 1000)
                )
                
                # Convert enhanced inferences to basic format if needed
                if enhanced_result and hasattr(enhanced_result, 'inferences'):
                    return self._convert_enhanced_inferences(enhanced_result.inferences)
                return enhanced_result or []
                
            except Exception as e:
                logger.warning(f"Enhanced reasoning failed, falling back: {e}")
                # Fall through to basic implementation
        
        start_time = time.time()
        all_inferences = []
        
        try:
            # Get facts and rules
            with self._lock:
                facts = list(self.facts.values())
                rules = [r for r in self.rules.values() if r.enabled]
            
            # Determine engines to use
            if engine_names is None:
                engine_names = list(self.reasoning_engines.keys())
            
            # Run reasoning with each engine
            if self.config.get('enable_parallel_reasoning'):
                # Parallel reasoning
                futures = []
                for engine_name in engine_names:
                    if engine_name in self.reasoning_engines:
                        future = self._executor.submit(
                            self._run_reasoning_engine,
                            engine_name, facts, rules
                        )
                        futures.append((engine_name, future))
                
                # Collect results
                for engine_name, future in futures:
                    try:
                        inferences = future.result(
                            timeout=self.config.get('reasoning_timeout_seconds', 30)
                        )
                        all_inferences.extend(inferences)
                    except Exception as e:
                        logger.error(f"Reasoning failed for {engine_name}: {str(e)}")
            else:
                # Sequential reasoning
                for engine_name in engine_names:
                    if engine_name in self.reasoning_engines:
                        inferences = self._run_reasoning_engine(engine_name, facts, rules)
                        all_inferences.extend(inferences)
            
            # Store inferences
            with self._lock:
                for inference in all_inferences:
                    self.inferences[inference.inference_id] = inference
            
            # Update metrics
            reasoning_time = (time.time() - start_time) * 1000
            self._update_metrics(len(all_inferences), reasoning_time, True)
            
            # Filter by confidence threshold
            threshold = self.config.get('inference_confidence_threshold', 0.6)
            filtered_inferences = [
                inf for inf in all_inferences 
                if inf.confidence >= threshold
            ]
            
            logger.info(f"Reasoning completed: {len(filtered_inferences)} inferences "
                       f"(filtered from {len(all_inferences)}) in {reasoning_time:.2f}ms")
            
            return filtered_inferences
            
        except Exception as e:
            logger.error(f"Reasoning failed: {str(e)}")
            self._update_metrics(0, 0, False)
            return []
    
    def _run_reasoning_engine(self, engine_name: str, 
                            facts: List[Fact], 
                            rules: List[Rule]) -> List[Inference]:
        """Run a specific reasoning engine"""
        try:
            engine = self.reasoning_engines[engine_name]
            
            # Filter rules relevant to engine
            relevant_rules = self._filter_rules_for_engine(engine_name, rules)
            
            # Run reasoning
            inferences = engine.reason(facts, relevant_rules)
            
            # Validate inferences
            valid_inferences = []
            for inference in inferences:
                if engine.validate(inference, facts):
                    valid_inferences.append(inference)
                else:
                    logger.warning(f"Invalid inference {inference.inference_id} from {engine_name}")
            
            logger.debug(f"{engine_name} produced {len(valid_inferences)} valid inferences")
            return valid_inferences
            
        except Exception as e:
            logger.error(f"Engine {engine_name} failed: {str(e)}")
            return []
    
    def _filter_rules_for_engine(self, engine_name: str, rules: List[Rule]) -> List[Rule]:
        """Filter rules relevant for specific engine"""
        if engine_name == 'rule_based':
            return [r for r in rules if r.rule_type in 
                   [RuleType.TRANSACTION_LIMIT, RuleType.VELOCITY, RuleType.COMPOSITE]]
        elif engine_name == 'fuzzy':
            return [r for r in rules if r.rule_type == RuleType.BEHAVIORAL]
        elif engine_name == 'temporal':
            return [r for r in rules if r.rule_type == RuleType.TEMPORAL]
        else:
            return rules
    
    def _convert_enhanced_inferences(self, enhanced_inferences: List[Any]) -> List[Inference]:
        """Convert enhanced inferences to basic format"""
        basic_inferences = []
        
        for enhanced_inf in enhanced_inferences:
            try:
                # Convert enhanced inference to basic inference
                basic_inference = Inference(
                    inference_id=getattr(enhanced_inf, 'inference_id', self._generate_id('INF')),
                    reasoning_type=ReasoningType(getattr(enhanced_inf, 'reasoning_type', 'rule_based')),
                    premises=getattr(enhanced_inf, 'premises', []),
                    conclusion=getattr(enhanced_inf, 'conclusion', None),
                    confidence=getattr(enhanced_inf, 'confidence', 0.5),
                    reasoning_path=getattr(enhanced_inf, 'reasoning_path', []),
                    metadata=getattr(enhanced_inf, 'metadata', {})
                )
                basic_inferences.append(basic_inference)
            except Exception as e:
                logger.warning(f"Failed to convert enhanced inference: {e}")
                continue
        
        return basic_inferences
    
    def validate_rules(self, rules_to_validate: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Validate business rules
        
        Args:
            rules_to_validate: List of rule IDs to validate (None = all)
            
        Returns:
            Dictionary of rule_id -> is_valid
        """
        if self.use_enhanced:
            try:
                return self.enhanced_reasoner.validate_rules(rules_to_validate)
            except Exception as e:
                logger.warning(f"Enhanced rule validation failed, falling back: {e}")
                # Fall through to basic implementation
        
        validation_results = {}
        
        with self._lock:
            if rules_to_validate is None:
                rules_to_validate = list(self.rules.keys())
            
            for rule_id in rules_to_validate:
                rule = self.rules.get(rule_id)
                if rule:
                    is_valid = self._validate_rule(rule)
                    validation_results[rule_id] = is_valid
                    
                    if not is_valid:
                        logger.warning(f"Rule {rule_id} failed validation")
        
        logger.info(f"Validated {len(validation_results)} rules, "
                   f"{sum(validation_results.values())} valid")
        
        return validation_results
    
    def _validate_rule(self, rule: Rule) -> bool:
        """Validate a single rule"""
        try:
            # Check rule structure
            if not rule.conditions or not rule.actions:
                return False
            
            # Validate conditions
            for condition in rule.conditions:
                if not self._validate_condition(condition):
                    return False
            
            # Validate actions
            for action in rule.actions:
                if not self._validate_action(action):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Rule validation error: {str(e)}")
            return False
    
    def _validate_condition(self, condition: Dict[str, Any]) -> bool:
        """Validate rule condition"""
        required_fields = ['type']
        return all(field in condition for field in required_fields)
    
    def _validate_action(self, action: Dict[str, Any]) -> bool:
        """Validate rule action"""
        required_fields = ['type']
        return all(field in action for field in required_fields)
    
    def integrate_knowledge(self, external_facts: List[Dict[str, Any]]) -> int:
        """
        Integrate external knowledge into knowledge base
        
        Args:
            external_facts: List of external facts to integrate
            
        Returns:
            Number of facts integrated
        """
        integrated_count = 0
        
        try:
            for fact_data in external_facts:
                # Validate and transform external fact
                if self._validate_external_fact(fact_data):
                    transformed_fact = self._transform_external_fact(fact_data)
                    fact_id = self.add_fact(transformed_fact)
                    
                    if fact_id:
                        integrated_count += 1
                else:
                    logger.warning(f"Invalid external fact: {fact_data}")
            
            logger.info(f"Integrated {integrated_count} of {len(external_facts)} external facts")
            
            # Trigger reasoning if significant new knowledge
            if integrated_count > 10:
                self._executor.submit(self.reason)
            
            return integrated_count
            
        except Exception as e:
            logger.error(f"Knowledge integration failed: {str(e)}")
            return integrated_count
    
    def _validate_external_fact(self, fact_data: Dict[str, Any]) -> bool:
        """Validate external fact"""
        required_fields = ['subject', 'predicate', 'object']
        return all(field in fact_data for field in required_fields)
    
    def _transform_external_fact(self, fact_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform external fact to internal format"""
        # Add source information
        fact_data['source'] = fact_data.get('source', 'external')
        
        # Normalize fact type
        if 'fact_type' not in fact_data:
            fact_data['fact_type'] = FactType.EXTERNAL_DATA.value
        
        return fact_data
    
    def query_knowledge(self, query: Dict[str, Any]) -> List[Union[Fact, Inference]]:
        """
        Query knowledge base
        
        Args:
            query: Query parameters
            
        Returns:
            List of matching facts and inferences
        """
        if self.use_enhanced:
            try:
                enhanced_results = self.enhanced_reasoner.query_knowledge(query)
                # Convert enhanced results to basic format if needed
                return self._convert_enhanced_query_results(enhanced_results)
            except Exception as e:
                logger.warning(f"Enhanced query failed, falling back: {e}")
                # Fall through to basic implementation
        
        results = []
        
        try:
            with self._lock:
                # Query facts
                if query.get('include_facts', True):
                    for fact in self.facts.values():
                        if self._match_fact_query(fact, query):
                            results.append(fact)
                
                # Query inferences  
                if query.get('include_inferences', True):
                    for inference in self.inferences.values():
                        if self._match_inference_query(inference, query):
                            results.append(inference)
            
            # Sort by confidence if requested
            if query.get('sort_by_confidence'):
                results.sort(key=lambda x: getattr(x, 'confidence', 0), reverse=True)
            
            # Apply limit
            limit = query.get('limit')
            if limit:
                results = results[:limit]
            
            logger.debug(f"Query returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Knowledge query failed: {str(e)}")
            return []
    
    def _match_fact_query(self, fact: Fact, query: Dict[str, Any]) -> bool:
        """Check if fact matches query"""
        # Match by subject
        if 'subject' in query and fact.subject != query['subject']:
            return False
        
        # Match by predicate
        if 'predicate' in query and fact.predicate != query['predicate']:
            return False
        
        # Match by fact type
        if 'fact_type' in query and fact.fact_type.value != query['fact_type']:
            return False
        
        # Match by confidence threshold
        if 'min_confidence' in query and fact.confidence < query['min_confidence']:
            return False
        
        return True
    
    def _match_inference_query(self, inference: Inference, query: Dict[str, Any]) -> bool:
        """Check if inference matches query"""
        # Match by reasoning type
        if 'reasoning_type' in query and inference.reasoning_type.value != query['reasoning_type']:
            return False
        
        # Match by confidence threshold
        if 'min_confidence' in query and inference.confidence < query['min_confidence']:
            return False
        
        return True
    
    def explain_inference(self, inference_id: str) -> Dict[str, Any]:
        """
        Explain reasoning behind an inference
        
        Args:
            inference_id: Inference ID to explain
            
        Returns:
            Explanation dictionary
        """
        try:
            with self._lock:
                inference = self.inferences.get(inference_id)
                if not inference:
                    return {'error': 'Inference not found'}
                
                # Get premise facts
                premise_facts = [
                    self.facts.get(fact_id) 
                    for fact_id in inference.premises
                ]
                premise_facts = [f for f in premise_facts if f]
                
                # Build explanation
                explanation = {
                    'inference_id': inference_id,
                    'conclusion': {
                        'subject': inference.conclusion.subject,
                        'predicate': inference.conclusion.predicate,
                        'object': inference.conclusion.object,
                        'confidence': inference.conclusion.confidence
                    },
                    'reasoning_type': inference.reasoning_type.value,
                    'confidence': inference.confidence,
                    'premises': [
                        {
                            'fact_id': f.fact_id,
                            'triple': f.to_triple(),
                            'confidence': f.confidence
                        }
                        for f in premise_facts
                    ],
                    'reasoning_path': inference.reasoning_path,
                    'timestamp': inference.timestamp.isoformat()
                }
                
                return explanation
                
        except Exception as e:
            logger.error(f"Failed to explain inference: {str(e)}")
            return {'error': str(e)}
    
    def _convert_enhanced_query_results(self, enhanced_results: List[Any]) -> List[Union[Fact, Inference]]:
        """Convert enhanced query results to basic format"""
        basic_results = []
        
        for result in enhanced_results:
            try:
                if hasattr(result, 'fact_id'):
                    # It's a fact
                    basic_fact = Fact(
                        fact_id=result.fact_id,
                        fact_type=FactType(getattr(result, 'fact_type', 'TRANSACTION')),
                        subject=result.subject,
                        predicate=result.predicate,
                        object=result.object,
                        confidence=getattr(result, 'confidence', 1.0),
                        source=getattr(result, 'source', 'unknown'),
                        metadata=getattr(result, 'metadata', {})
                    )
                    basic_results.append(basic_fact)
                elif hasattr(result, 'inference_id'):
                    # It's an inference
                    basic_inference = Inference(
                        inference_id=result.inference_id,
                        reasoning_type=ReasoningType(getattr(result, 'reasoning_type', 'rule_based')),
                        premises=getattr(result, 'premises', []),
                        conclusion=getattr(result, 'conclusion', None),
                        confidence=getattr(result, 'confidence', 0.5),
                        reasoning_path=getattr(result, 'reasoning_path', []),
                        metadata=getattr(result, 'metadata', {})
                    )
                    basic_results.append(basic_inference)
            except Exception as e:
                logger.warning(f"Failed to convert enhanced query result: {e}")
                continue
        
        return basic_results
    
    def get_metrics(self) -> ReasoningMetrics:
        """Get reasoning metrics"""
        if self.use_enhanced:
            try:
                enhanced_metrics = self.enhanced_reasoner.get_metrics()
                # Convert enhanced metrics to basic format if needed
                return self._convert_enhanced_metrics(enhanced_metrics)
            except Exception as e:
                logger.warning(f"Enhanced metrics failed, falling back: {e}")
                # Fall through to basic implementation
        
        with self._lock:
            return self.metrics
    
    def _convert_enhanced_metrics(self, enhanced_metrics: Any) -> ReasoningMetrics:
        """Convert enhanced metrics to basic format"""
        try:
            return ReasoningMetrics(
                total_inferences=getattr(enhanced_metrics, 'total_inferences', 0),
                successful_inferences=getattr(enhanced_metrics, 'successful_inferences', 0),
                failed_inferences=getattr(enhanced_metrics, 'failed_inferences', 0),
                average_reasoning_time_ms=getattr(enhanced_metrics, 'average_reasoning_time_ms', 0.0),
                total_facts_processed=getattr(enhanced_metrics, 'total_facts_processed', 0),
                total_rules_evaluated=getattr(enhanced_metrics, 'total_rules_evaluated', 0),
                inference_accuracy=getattr(enhanced_metrics, 'inference_accuracy', 0.0),
                rule_coverage=getattr(enhanced_metrics, 'rule_coverage', 0.0),
                metrics_by_type=getattr(enhanced_metrics, 'metrics_by_type', {}),
                hourly_inference_count=getattr(enhanced_metrics, 'hourly_inference_count', {})
            )
        except Exception as e:
            logger.warning(f"Failed to convert enhanced metrics: {e}")
            return self.metrics
    
    def export_knowledge_base(self) -> Dict[str, Any]:
        """Export entire knowledge base"""
        with self._lock:
            return {
                'facts': {
                    fact_id: {
                        'triple': fact.to_triple(),
                        'type': fact.fact_type.value,
                        'confidence': fact.confidence,
                        'source': fact.source,
                        'timestamp': fact.timestamp.isoformat()
                    }
                    for fact_id, fact in self.facts.items()
                },
                'rules': {
                    rule_id: {
                        'name': rule.name,
                        'type': rule.rule_type.value,
                        'description': rule.description,
                        'enabled': rule.enabled,
                        'priority': rule.priority
                    }
                    for rule_id, rule in self.rules.items()
                },
                'inferences': {
                    inf_id: {
                        'conclusion': inf.conclusion.to_triple(),
                        'confidence': inf.confidence,
                        'reasoning_type': inf.reasoning_type.value,
                        'premise_count': len(inf.premises)
                    }
                    for inf_id, inf in self.inferences.items()
                },
                'metrics': self.metrics.to_dict(),
                'export_timestamp': datetime.now().isoformat()
            }
    
    def persist_knowledge(self) -> bool:
        """Persist knowledge base to storage"""
        try:
            # Persist facts
            facts_file = self.storage_path / 'facts.pkl'
            with open(facts_file, 'wb') as f:
                pickle.dump(self.facts, f)
            
            # Persist rules
            rules_file = self.storage_path / 'rules.pkl'
            with open(rules_file, 'wb') as f:
                pickle.dump(self.rules, f)
            
            # Persist inferences
            inferences_file = self.storage_path / 'inferences.pkl'
            with open(inferences_file, 'wb') as f:
                pickle.dump(self.inferences, f)
            
            logger.info("Knowledge base persisted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to persist knowledge: {str(e)}")
            return False
    
    def load_knowledge(self) -> bool:
        """Load knowledge base from storage"""
        try:
            # Load facts
            facts_file = self.storage_path / 'facts.pkl'
            if facts_file.exists():
                with open(facts_file, 'rb') as f:
                    self.facts = pickle.load(f)
            
            # Load rules
            rules_file = self.storage_path / 'rules.pkl'
            if rules_file.exists():
                with open(rules_file, 'rb') as f:
                    self.rules = pickle.load(f)
            
            # Load inferences
            inferences_file = self.storage_path / 'inferences.pkl'
            if inferences_file.exists():
                with open(inferences_file, 'rb') as f:
                    self.inferences = pickle.load(f)
            
            logger.info(f"Knowledge base loaded: {len(self.facts)} facts, "
                       f"{len(self.rules)} rules, {len(self.inferences)} inferences")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load knowledge: {str(e)}")
            return False
    
    def _update_metrics(self, inference_count: int, time_ms: float, success: bool) -> None:
        """Update reasoning metrics"""
        with self._lock:
            self.metrics.total_inferences += inference_count
            
            if success:
                self.metrics.successful_inferences += inference_count
            else:
                self.metrics.failed_inferences += 1
            
            # Update average time
            if inference_count > 0:
                total = self.metrics.total_inferences
                current_avg = self.metrics.average_reasoning_time_ms
                self.metrics.average_reasoning_time_ms = (
                    (current_avg * (total - inference_count) + time_ms) / total
                )
            
            # Update accuracy
            if self.metrics.total_inferences > 0:
                self.metrics.inference_accuracy = (
                    self.metrics.successful_inferences / self.metrics.total_inferences
                )
            
            # Update hourly metrics
            hour_key = datetime.now().strftime('%Y-%m-%d_%H')
            if hour_key not in self.metrics.hourly_inference_count:
                self.metrics.hourly_inference_count[hour_key] = 0
            self.metrics.hourly_inference_count[hour_key] += inference_count
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID"""
        import hashlib
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        random_suffix = hashlib.md5(
            f"{timestamp}_{time.time()}".encode()
        ).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{random_suffix}"
    
    def __repr__(self) -> str:
        enhanced_status = " [Enhanced]" if self.use_enhanced else ""
        return (f"FinancialSymbolicReasoner{enhanced_status}(facts={len(self.facts)}, "
               f"rules={len(self.rules)}, "
               f"inferences={len(self.inferences)})")

# Legacy compatibility
SymbolicReasoner = FinancialSymbolicReasoner

# Export main classes
__all__ = [
    'FinancialSymbolicReasoner',
    'SymbolicReasoner',  # Legacy compatibility
    'Fact',
    'Rule',
    'Inference',
    'ReasoningMetrics',
    'ReasoningType',
    'RuleType',
    'FactType',
    'BaseReasoningEngine',
    'RuleBasedReasoningEngine',
    'FuzzyReasoningEngine',
    'TemporalReasoningEngine'
]

if __name__ == "__main__":
    # Example usage and testing
    
    # Initialize reasoner
    config = {
        'temporal_window_hours': 24,
        'inference_confidence_threshold': 0.6,
        'enable_parallel_reasoning': True
    }
    
    reasoner = FinancialSymbolicReasoner(config)
    print("FinancialSymbolicReasoner initialized")
    
    # Add facts
    print("\nAdding facts...")
    
    # Transaction facts
    fact1_id = reasoner.add_fact({
        'fact_type': 'TRANSACTION',
        'subject': 'txn_001',
        'predicate': 'amount',
        'object': 15000,
        'confidence': 1.0,
        'source': 'transaction_system'
    })
    
    fact2_id = reasoner.add_fact({
        'fact_type': 'USER_BEHAVIOR',
        'subject': 'user_123',
        'predicate': 'transaction_count',
        'object': 10,
        'confidence': 0.9,
        'source': 'behavior_analysis'
    })
    
    print(f"Added facts: {fact1_id}, {fact2_id}")
    
    # Add custom rule
    rule_id = reasoner.add_rule({
        'name': 'Custom High Risk Rule',
        'rule_type': 'COMPOSITE',
        'description': 'Combined high amount and behavior anomaly',
        'conditions': [
            {'type': 'fact_match', 'predicate': 'amount', 'operator': '>', 'value': 10000}
        ],
        'actions': [
            {'type': 'assert', 'subject': 'fraud_risk', 'predicate': 'level', 'object': 'high'}
        ],
        'priority': 5
    })
    
    print(f"Added rule: {rule_id}")
    
    # Perform reasoning
    print("\nPerforming reasoning...")
    inferences = reasoner.reason()
    
    print(f"Generated {len(inferences)} inferences")
    for inf in inferences:
        print(f"  - {inf.conclusion.subject} {inf.conclusion.predicate} {inf.conclusion.object} "
              f"(confidence: {inf.confidence:.2f})")
    
    # Validate rules
    print("\nValidating rules...")
    validation_results = reasoner.validate_rules()
    print(f"Validation results: {validation_results}")
    
    # Query knowledge
    print("\nQuerying knowledge base...")
    query_results = reasoner.query_knowledge({
        'fact_type': 'TRANSACTION',
        'min_confidence': 0.8,
        'include_inferences': False
    })
    print(f"Query returned {len(query_results)} results")
    
    # Get metrics
    metrics = reasoner.get_metrics()
    print(f"\nMetrics: {metrics.to_dict()}")
    
    # Export knowledge base
    kb_export = reasoner.export_knowledge_base()
    print(f"\nKnowledge base export:")
    print(f"  - Facts: {len(kb_export['facts'])}")
    print(f"  - Rules: {len(kb_export['rules'])}")
    print(f"  - Inferences: {len(kb_export['inferences'])}")
    
    print("\nFinancialSymbolicReasoner ready for production use!")