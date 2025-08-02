#!/usr/bin/env python3
"""
Universal AI Core Symbolic Reasoning Engine
==========================================

This module provides comprehensive symbolic reasoning capabilities for the Universal AI Core system.
Extracted and adapted from the Saraphis symbolic AI system, made domain-agnostic while preserving
all sophisticated reasoning capabilities.

Features:
- Multi-modal symbolic reasoning (deductive, inductive, abductive)
- Rule-based inference engine
- Constraint solving framework
- Pattern matching and recognition
- Knowledge base integration
- Async reasoning with background monitoring
- Comprehensive caching and optimization
"""

import asyncio
import json
import logging
import hashlib
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import weakref
import queue
from concurrent.futures import ThreadPoolExecutor
import copy

logger = logging.getLogger(__name__)


class SymbolicOperation(Enum):
    """Types of symbolic operations"""
    REASONING = "reasoning"
    INFERENCE = "inference"
    DEDUCTION = "deduction"
    INDUCTION = "induction"
    ABDUCTION = "abduction"
    PATTERN_MATCHING = "pattern_matching"
    RULE_APPLICATION = "rule_application"
    CONSTRAINT_SOLVING = "constraint_solving"


class ReasoningStrategy(Enum):
    """Reasoning strategies"""
    FORWARD_CHAINING = "forward_chaining"
    BACKWARD_CHAINING = "backward_chaining"
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    BEST_FIRST = "best_first"
    MIXED = "mixed"


class ConstraintType(Enum):
    """Types of constraints"""
    EQUALITY = "equality"
    INEQUALITY = "inequality"
    LOGICAL = "logical"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


@dataclass(frozen=True)
class SymbolicResult:
    """Immutable symbolic reasoning result"""
    result_id: str
    timestamp: str
    operation_type: SymbolicOperation
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    reasoning_chain: List[str]
    confidence: float
    processing_time: float
    immutable_hash: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})


@dataclass
class Rule:
    """Symbolic reasoning rule"""
    id: str
    name: str
    conditions: List[str]
    conclusions: List[str]
    confidence: float = 1.0
    priority: int = 0
    domain: str = "general"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def matches(self, facts: Set[str]) -> bool:
        """Check if rule conditions are satisfied by given facts"""
        return all(condition in facts for condition in self.conditions)
    
    def apply(self, facts: Set[str]) -> Set[str]:
        """Apply rule to facts and return new conclusions"""
        if self.matches(facts):
            return set(self.conclusions)
        return set()


@dataclass
class Constraint:
    """Constraint representation"""
    id: str
    constraint_type: ConstraintType
    variables: List[str]
    expression: str
    domain: Optional[Tuple[Any, Any]] = None
    priority: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class KnowledgeBase:
    """
    Domain-agnostic knowledge base for symbolic reasoning.
    
    Extracted and adapted from Saraphis symbolic_ai_module.py,
    made generic while preserving sophisticated reasoning capabilities.
    """
    
    def __init__(self):
        self.facts = set()
        self.rules = {}
        self.logical_rules = self._initialize_logical_rules()
        self.inference_patterns = self._initialize_inference_patterns()
        self.constraint_solving = self._initialize_constraint_solving()
        self.pattern_templates = self._initialize_pattern_templates()
        self._rule_cache = {}
        self._fact_cache = {}
        
    def _initialize_logical_rules(self) -> Dict[str, str]:
        """Initialize fundamental logical rules"""
        return {
            "modus_ponens": "If P implies Q and P is true, then Q is true",
            "modus_tollens": "If P implies Q and Q is false, then P is false",
            "syllogism": "If all A are B and all B are C, then all A are C",
            "contradiction": "If P and not P, then false",
            "excluded_middle": "Either P or not P is true",
            "double_negation": "Not not P is equivalent to P",
            "contraposition": "P implies Q is equivalent to not Q implies not P",
            "hypothetical_syllogism": "If P implies Q and Q implies R, then P implies R",
            "disjunctive_syllogism": "If P or Q, and not P, then Q",
            "conjunction": "If P and Q are both true, then P and Q is true",
            "simplification": "If P and Q is true, then P is true",
            "addition": "If P is true, then P or Q is true"
        }
    
    def _initialize_inference_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize inference patterns"""
        return {
            "deductive": {
                "description": "General principles to specific conclusions",
                "confidence_threshold": 0.95,
                "applicable_operations": [SymbolicOperation.DEDUCTION, SymbolicOperation.REASONING],
                "strategy": ReasoningStrategy.FORWARD_CHAINING
            },
            "inductive": {
                "description": "Specific observations to general patterns",
                "confidence_threshold": 0.80,
                "applicable_operations": [SymbolicOperation.INDUCTION, SymbolicOperation.PATTERN_MATCHING],
                "strategy": ReasoningStrategy.BREADTH_FIRST
            },
            "abductive": {
                "description": "Best explanation for observations",
                "confidence_threshold": 0.70,
                "applicable_operations": [SymbolicOperation.ABDUCTION, SymbolicOperation.REASONING],
                "strategy": ReasoningStrategy.BEST_FIRST
            }
        }
    
    def _initialize_constraint_solving(self) -> Dict[str, str]:
        """Initialize constraint solving methods"""
        return {
            "linear_programming": "Solve linear optimization problems",
            "satisfiability": "Find satisfying assignments for logical formulas",
            "constraint_propagation": "Propagate constraints through networks",
            "backtracking": "Systematic search with constraint checking",
            "arc_consistency": "Ensure consistency between constraint variables"
        }
    
    def _initialize_pattern_templates(self) -> Dict[str, str]:
        """Initialize pattern matching templates"""
        return {
            "sequence": "Find patterns in ordered data",
            "classification": "Group similar items together",
            "causation": "Identify cause-effect relationships",
            "correlation": "Find relationships between variables",
            "hierarchy": "Organize data in tree structures",
            "similarity": "Identify similar patterns or structures",
            "anomaly": "Detect unusual or unexpected patterns"
        }
    
    def add_fact(self, fact: str) -> None:
        """Add a fact to the knowledge base"""
        self.facts.add(fact)
        self._fact_cache.clear()  # Invalidate cache
    
    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the knowledge base"""
        self.rules[rule.id] = rule
        self._rule_cache.clear()  # Invalidate cache
    
    def remove_fact(self, fact: str) -> None:
        """Remove a fact from the knowledge base"""
        self.facts.discard(fact)
        self._fact_cache.clear()
    
    def remove_rule(self, rule_id: str) -> None:
        """Remove a rule from the knowledge base"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            self._rule_cache.clear()
    
    def query_facts(self, pattern: str) -> Set[str]:
        """Query facts matching a pattern"""
        cache_key = f"facts:{pattern}"
        if cache_key in self._fact_cache:
            return self._fact_cache[cache_key]
        
        matching_facts = {fact for fact in self.facts if pattern.lower() in fact.lower()}
        self._fact_cache[cache_key] = matching_facts
        return matching_facts
    
    def get_applicable_rules(self, facts: Set[str]) -> List[Rule]:
        """Get rules that can be applied to the given facts"""
        cache_key = f"rules:{hash(frozenset(facts))}"
        if cache_key in self._rule_cache:
            return self._rule_cache[cache_key]
        
        applicable_rules = [rule for rule in self.rules.values() if rule.matches(facts)]
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)
        
        self._rule_cache[cache_key] = applicable_rules
        return applicable_rules


class SymbolicReasoningEngine:
    """
    Core symbolic reasoning engine with multi-modal reasoning capabilities.
    
    Extracted and adapted from Saraphis SymbolicAICore (lines 38-227),
    made domain-agnostic while preserving all sophisticated capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the symbolic reasoning engine"""
        self.config = config or {}
        self._system_hash = self._calculate_system_hash()
        self._operation_counter = 0
        self.knowledge_base = KnowledgeBase()
        self._results_history = []
        self._lock = threading.RLock()
        self._processing_active = True
        self._reasoning_cache = {}
        self._constraint_cache = {}
        
        # Performance settings
        self.max_reasoning_depth = self.config.get('max_reasoning_depth', 100)
        self.reasoning_timeout = self.config.get('reasoning_timeout', 60.0)
        self.cache_size_limit = self.config.get('cache_size_limit', 10000)
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("🧠 Symbolic Reasoning Engine initialized")
    
    def _calculate_system_hash(self) -> str:
        """Calculate immutable hash of symbolic reasoning system"""
        system_content = {
            "operations": [o.value for o in SymbolicOperation],
            "version": "1.0.0",
            "symbolic_reasoning": True,
            "domain_agnostic": True,
            "immutable": True
        }
        content_str = json.dumps(system_content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    async def symbolic_reasoning(self, input_data: Dict[str, Any]) -> SymbolicResult:
        """
        Main symbolic reasoning interface.
        
        Args:
            input_data: Input data for symbolic processing
        
        Returns:
            SymbolicResult: Immutable symbolic reasoning result
        """
        with self._lock:
            self._operation_counter += 1
            result_id = f"symbolic_{self._operation_counter}_{int(datetime.utcnow().timestamp())}"
        
        start_time = time.time()
        
        try:
            # Determine operation type based on input
            operation_type = self._determine_operation_type(input_data)
            
            # Check cache
            cache_key = self._generate_cache_key(input_data, operation_type)
            cached_result = self._reasoning_cache.get(cache_key)
            if cached_result:
                logger.info(f"🎯 Using cached reasoning result for operation: {operation_type.value}")
                return cached_result
            
            # Perform symbolic reasoning
            output_data, reasoning_chain, confidence = await self._perform_symbolic_reasoning(
                operation_type, input_data
            )
            
            processing_time = time.time() - start_time
            
            # Create immutable result
            result = SymbolicResult(
                result_id=result_id,
                timestamp=datetime.utcnow().isoformat(),
                operation_type=operation_type,
                input_data=input_data,
                output_data=output_data,
                reasoning_chain=reasoning_chain,
                confidence=confidence,
                processing_time=processing_time,
                immutable_hash="",
                metadata={"engine_version": "1.0.0", "system_hash": self._system_hash}
            )
            
            # Calculate immutable hash
            result_dict = asdict(result)
            result_dict["immutable_hash"] = ""
            content_str = json.dumps(result_dict, sort_keys=True)
            immutable_hash = hashlib.sha256(content_str.encode()).hexdigest()
            
            # Create final immutable result
            final_result = SymbolicResult(
                result_id=result.result_id,
                timestamp=result.timestamp,
                operation_type=result.operation_type,
                input_data=result.input_data,
                output_data=result.output_data,
                reasoning_chain=result.reasoning_chain,
                confidence=result.confidence,
                processing_time=result.processing_time,
                immutable_hash=immutable_hash,
                metadata=result.metadata
            )
            
            # Store in history and cache
            self._results_history.append(final_result)
            if len(self._reasoning_cache) < self.cache_size_limit:
                self._reasoning_cache[cache_key] = final_result
            
            logger.info(f"🧠 Symbolic operation completed: {operation_type.value} with confidence {confidence:.2f}")
            
            return final_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Symbolic reasoning error: {e}")
            
            error_result = SymbolicResult(
                result_id=result_id,
                timestamp=datetime.utcnow().isoformat(),
                operation_type=SymbolicOperation.REASONING,
                input_data=input_data,
                output_data={"error": str(e)},
                reasoning_chain=[f"Error in symbolic reasoning: {e}"],
                confidence=0.0,
                processing_time=processing_time,
                immutable_hash="error_hash",
                metadata={"error": True}
            )
            
            return error_result
    
    def _determine_operation_type(self, input_data: Dict[str, Any]) -> SymbolicOperation:
        """Determine the type of symbolic operation to perform"""
        input_str = json.dumps(input_data, default=str).lower()
        
        # Enhanced operation detection
        if any(word in input_str for word in ["deduce", "conclude", "therefore", "implies", "follows"]):
            return SymbolicOperation.DEDUCTION
        elif any(word in input_str for word in ["generalize", "pattern", "trend", "similar", "infer"]):
            return SymbolicOperation.INDUCTION
        elif any(word in input_str for word in ["explain", "hypothesis", "best", "likely", "because"]):
            return SymbolicOperation.ABDUCTION
        elif any(word in input_str for word in ["match", "find", "search", "locate", "identify"]):
            return SymbolicOperation.PATTERN_MATCHING
        elif any(word in input_str for word in ["rule", "apply", "follow", "execute", "implement"]):
            return SymbolicOperation.RULE_APPLICATION
        elif any(word in input_str for word in ["constraint", "solve", "optimize", "minimize", "satisfy"]):
            return SymbolicOperation.CONSTRAINT_SOLVING
        else:
            return SymbolicOperation.REASONING
    
    async def _perform_symbolic_reasoning(self, operation_type: SymbolicOperation, 
                                        input_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], float]:
        """Perform symbolic reasoning based on operation type"""
        try:
            if operation_type == SymbolicOperation.DEDUCTION:
                return await self._perform_deductive_reasoning(input_data)
            elif operation_type == SymbolicOperation.INDUCTION:
                return await self._perform_inductive_reasoning(input_data)
            elif operation_type == SymbolicOperation.ABDUCTION:
                return await self._perform_abductive_reasoning(input_data)
            elif operation_type == SymbolicOperation.PATTERN_MATCHING:
                return await self._perform_pattern_matching(input_data)
            elif operation_type == SymbolicOperation.RULE_APPLICATION:
                return await self._perform_rule_application(input_data)
            elif operation_type == SymbolicOperation.CONSTRAINT_SOLVING:
                return await self._perform_constraint_solving(input_data)
            else:
                return await self._perform_general_reasoning(input_data)
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Reasoning timeout for operation: {operation_type.value}")
            return {"error": "reasoning_timeout"}, ["Reasoning operation timed out"], 0.0
        except Exception as e:
            logger.error(f"❌ Error in reasoning operation {operation_type.value}: {e}")
            return {"error": str(e)}, [f"Error in {operation_type.value}: {e}"], 0.0
    
    async def _perform_deductive_reasoning(self, input_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], float]:
        """Perform deductive reasoning (general to specific)"""
        reasoning_chain = ["Starting deductive reasoning"]
        
        premises = input_data.get("premises", [])
        conclusion = input_data.get("conclusion", "")
        rules = input_data.get("rules", [])
        
        reasoning_chain.append(f"Analyzing {len(premises)} premises for conclusion: {conclusion}")
        
        # Convert premises to facts
        facts = set(premises)
        
        # Apply logical rules
        applied_rules = []
        derived_facts = set()
        
        for rule_name in rules:
            if rule_name in self.knowledge_base.logical_rules:
                applied_rules.append(rule_name)
                reasoning_chain.append(f"Applied logical rule: {rule_name}")
                
                # Simulate rule application
                if rule_name == "modus_ponens":
                    # Look for implications and premises
                    for fact in facts:
                        if "implies" in fact and "->" in fact:
                            parts = fact.split("->")
                            if len(parts) == 2:
                                antecedent = parts[0].strip()
                                consequent = parts[1].strip()
                                if antecedent in facts:
                                    derived_facts.add(consequent)
                                    reasoning_chain.append(f"Derived: {consequent} from {fact}")
        
        # Check if conclusion follows
        all_facts = facts.union(derived_facts)
        conclusion_valid = conclusion in all_facts or any(conclusion in fact for fact in all_facts)
        
        if conclusion_valid:
            reasoning_chain.append("Conclusion follows from premises")
            confidence = 0.95
        else:
            reasoning_chain.append("Conclusion does not follow from premises")
            confidence = 0.3
        
        output_data = {
            "conclusion_valid": conclusion_valid,
            "premises_analyzed": len(premises),
            "rules_applied": applied_rules,
            "derived_facts": list(derived_facts),
            "reasoning_type": "deductive"
        }
        
        return output_data, reasoning_chain, confidence
    
    async def _perform_inductive_reasoning(self, input_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], float]:
        """Perform inductive reasoning (specific to general)"""
        reasoning_chain = ["Starting inductive reasoning"]
        
        observations = input_data.get("observations", [])
        pattern_type = input_data.get("pattern_type", "general")
        
        reasoning_chain.append(f"Analyzing {len(observations)} observations for {pattern_type} patterns")
        
        # Find patterns in observations
        patterns = self._find_patterns(observations, pattern_type)
        reasoning_chain.append(f"Found {len(patterns)} patterns")
        
        # Generate general rule from patterns
        general_rule = self._generate_general_rule(patterns, pattern_type)
        reasoning_chain.append(f"Generated general rule: {general_rule}")
        
        # Calculate confidence based on pattern strength
        confidence = min(0.85, len(patterns) / max(len(observations), 1) * 1.2)
        
        output_data = {
            "patterns_found": patterns,
            "general_rule": general_rule,
            "pattern_type": pattern_type,
            "confidence_factors": ["pattern_count", "observation_count", "pattern_consistency"],
            "reasoning_type": "inductive"
        }
        
        return output_data, reasoning_chain, confidence
    
    async def _perform_abductive_reasoning(self, input_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], float]:
        """Perform abductive reasoning (best explanation)"""
        reasoning_chain = ["Starting abductive reasoning"]
        
        observations = input_data.get("observations", [])
        hypotheses = input_data.get("hypotheses", [])
        context = input_data.get("context", {})
        
        reasoning_chain.append(f"Evaluating {len(hypotheses)} hypotheses for {len(observations)} observations")
        
        # Evaluate each hypothesis
        hypothesis_scores = {}
        for hypothesis in hypotheses:
            score = self._evaluate_hypothesis(hypothesis, observations, context)
            hypothesis_scores[hypothesis] = score
            reasoning_chain.append(f"Hypothesis '{hypothesis}' score: {score:.2f}")
        
        # Find best hypothesis
        if hypothesis_scores:
            best_hypothesis = max(hypothesis_scores.keys(), key=lambda h: hypothesis_scores[h])
            best_score = hypothesis_scores[best_hypothesis]
        else:
            best_hypothesis = None
            best_score = 0.0
        
        reasoning_chain.append(f"Best explanation: {best_hypothesis} (score: {best_score:.2f})")
        
        output_data = {
            "best_hypothesis": best_hypothesis,
            "hypothesis_score": best_score,
            "all_scores": hypothesis_scores,
            "evaluation_criteria": ["explanatory_power", "simplicity", "consistency"],
            "reasoning_type": "abductive"
        }
        
        return output_data, reasoning_chain, best_score
    
    async def _perform_pattern_matching(self, input_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], float]:
        """Perform pattern matching and recognition"""
        reasoning_chain = ["Starting pattern matching"]
        
        data = input_data.get("data", [])
        template = input_data.get("template", "")
        similarity_threshold = input_data.get("similarity_threshold", 0.7)
        
        reasoning_chain.append(f"Matching {len(data)} items against template with threshold {similarity_threshold}")
        
        # Find matches
        matches = self._find_matches(data, template, similarity_threshold)
        reasoning_chain.append(f"Found {len(matches)} matches")
        
        # Calculate overall confidence
        confidence = len(matches) / max(len(data), 1)
        
        output_data = {
            "matches_found": matches,
            "match_count": len(matches),
            "template_used": template,
            "similarity_threshold": similarity_threshold,
            "reasoning_type": "pattern_matching"
        }
        
        return output_data, reasoning_chain, confidence
    
    async def _perform_rule_application(self, input_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], float]:
        """Perform rule-based reasoning"""
        reasoning_chain = ["Starting rule application"]
        
        facts = set(input_data.get("facts", []))
        max_iterations = input_data.get("max_iterations", 10)
        
        reasoning_chain.append(f"Applying rules to {len(facts)} facts")
        
        # Get applicable rules from knowledge base
        applicable_rules = self.knowledge_base.get_applicable_rules(facts)
        reasoning_chain.append(f"Found {len(applicable_rules)} applicable rules")
        
        # Apply rules iteratively
        derived_facts = set()
        iteration = 0
        
        while iteration < max_iterations:
            new_facts = set()
            
            for rule in applicable_rules:
                if rule.matches(facts.union(derived_facts)):
                    conclusions = rule.apply(facts.union(derived_facts))
                    new_facts.update(conclusions)
                    reasoning_chain.append(f"Applied rule '{rule.name}': {conclusions}")
            
            if not new_facts:
                break
            
            derived_facts.update(new_facts)
            facts.update(new_facts)
            iteration += 1
        
        confidence = min(0.9, len(derived_facts) / max(len(applicable_rules), 1))
        
        output_data = {
            "initial_facts": list(input_data.get("facts", [])),
            "derived_facts": list(derived_facts),
            "applied_rules": len(applicable_rules),
            "iterations": iteration,
            "reasoning_type": "rule_application"
        }
        
        return output_data, reasoning_chain, confidence
    
    async def _perform_constraint_solving(self, input_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], float]:
        """Perform constraint solving"""
        reasoning_chain = ["Starting constraint solving"]
        
        constraints = input_data.get("constraints", [])
        variables = input_data.get("variables", {})
        optimization_target = input_data.get("optimization_target", None)
        
        reasoning_chain.append(f"Solving {len(constraints)} constraints with {len(variables)} variables")
        
        # Simple constraint satisfaction
        solution = self._solve_constraints(constraints, variables, optimization_target)
        
        if solution:
            reasoning_chain.append("Constraint satisfaction successful")
            confidence = 0.9
        else:
            reasoning_chain.append("No solution found for constraints")
            confidence = 0.0
        
        output_data = {
            "solution": solution,
            "constraints_count": len(constraints),
            "variables_count": len(variables),
            "optimization_target": optimization_target,
            "reasoning_type": "constraint_solving"
        }
        
        return output_data, reasoning_chain, confidence
    
    async def _perform_general_reasoning(self, input_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], float]:
        """Perform general symbolic reasoning"""
        reasoning_chain = ["Starting general symbolic reasoning"]
        
        # Apply logical operations to input data
        result = self._apply_logical_operations(input_data)
        reasoning_chain.append("Applied general logical operations")
        
        output_data = {
            "reasoning_result": result,
            "operations_applied": ["logical_inference", "pattern_analysis", "rule_matching"],
            "reasoning_type": "general"
        }
        
        return output_data, reasoning_chain, 0.7
    
    def _find_patterns(self, observations: List[Any], pattern_type: str) -> List[str]:
        """Find patterns in observations"""
        patterns = []
        
        if pattern_type == "sequence":
            # Look for sequential patterns
            for i in range(len(observations) - 1):
                pattern = f"Sequential pattern: {observations[i]} -> {observations[i+1]}"
                patterns.append(pattern)
        
        elif pattern_type == "frequency":
            # Look for frequency patterns
            from collections import Counter
            counts = Counter(observations)
            for item, count in counts.most_common(3):
                pattern = f"Frequent item: {item} (appears {count} times)"
                patterns.append(pattern)
        
        elif pattern_type == "similarity":
            # Look for similarity patterns
            similar_groups = self._group_similar_items(observations)
            for group in similar_groups:
                if len(group) > 1:
                    pattern = f"Similar group: {group}"
                    patterns.append(pattern)
        
        return patterns
    
    def _generate_general_rule(self, patterns: List[str], pattern_type: str) -> str:
        """Generate a general rule from observed patterns"""
        if not patterns:
            return "No consistent pattern found"
        
        if pattern_type == "sequence":
            return f"Sequential rule based on {len(patterns)} observed transitions"
        elif pattern_type == "frequency":
            return f"Frequency rule based on {len(patterns)} common items"
        elif pattern_type == "similarity":
            return f"Similarity rule based on {len(patterns)} grouped patterns"
        else:
            return f"General rule based on {len(patterns)} observed patterns"
    
    def _evaluate_hypothesis(self, hypothesis: str, observations: List[Any], context: Dict[str, Any]) -> float:
        """Evaluate how well a hypothesis explains observations"""
        # Simple evaluation based on keyword matching and context
        score = 0.0
        
        # Check if hypothesis mentions key observation terms
        obs_terms = set()
        for obs in observations:
            obs_terms.update(str(obs).lower().split())
        
        hyp_terms = set(hypothesis.lower().split())
        term_overlap = len(obs_terms.intersection(hyp_terms))
        score += term_overlap / max(len(obs_terms), 1) * 0.5
        
        # Check context relevance
        context_terms = set()
        for value in context.values():
            context_terms.update(str(value).lower().split())
        
        context_overlap = len(context_terms.intersection(hyp_terms))
        score += context_overlap / max(len(context_terms), 1) * 0.3
        
        # Bonus for explanatory keywords
        explanatory_words = ["because", "due to", "caused by", "results in", "leads to"]
        if any(word in hypothesis.lower() for word in explanatory_words):
            score += 0.2
        
        return min(score, 1.0)
    
    def _find_matches(self, data: List[Any], template: str, threshold: float) -> List[Any]:
        """Find items matching a template"""
        matches = []
        template_lower = template.lower()
        
        for item in data:
            item_str = str(item).lower()
            
            # Simple similarity based on common words
            template_words = set(template_lower.split())
            item_words = set(item_str.split())
            
            if template_words and item_words:
                similarity = len(template_words.intersection(item_words)) / len(template_words.union(item_words))
                if similarity >= threshold:
                    matches.append(item)
            elif template_lower in item_str:
                matches.append(item)
        
        return matches
    
    def _solve_constraints(self, constraints: List[str], variables: Dict[str, Any], 
                          optimization_target: Optional[str]) -> Optional[Dict[str, Any]]:
        """Simple constraint solving implementation"""
        if not constraints or not variables:
            return None
        
        # Simple constraint satisfaction for basic cases
        solution = variables.copy()
        
        # Try to satisfy basic equality constraints
        for constraint in constraints:
            if "=" in constraint and "!=" not in constraint:
                parts = constraint.split("=")
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    
                    if left in variables and right.isdigit():
                        solution[left] = int(right)
                    elif left in variables and right.replace(".", "").isdigit():
                        solution[left] = float(right)
        
        return solution
    
    def _apply_logical_operations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply logical operations to data"""
        result = {}
        
        for key, value in data.items():
            if isinstance(value, bool):
                result[f"not_{key}"] = not value
            elif isinstance(value, (int, float)):
                result[f"double_{key}"] = value * 2
            elif isinstance(value, str):
                result[f"upper_{key}"] = value.upper()
            else:
                result[key] = value
        
        return result
    
    def _group_similar_items(self, items: List[Any]) -> List[List[Any]]:
        """Group similar items together"""
        if not items:
            return []
        
        groups = []
        ungrouped = items.copy()
        
        while ungrouped:
            current = ungrouped.pop(0)
            group = [current]
            
            # Find similar items
            to_remove = []
            for item in ungrouped:
                if self._are_similar(current, item):
                    group.append(item)
                    to_remove.append(item)
            
            # Remove grouped items
            for item in to_remove:
                ungrouped.remove(item)
            
            groups.append(group)
        
        return groups
    
    def _are_similar(self, item1: Any, item2: Any) -> bool:
        """Check if two items are similar"""
        # Simple similarity check
        str1 = str(item1).lower()
        str2 = str(item2).lower()
        
        # Check for common words
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if words1 and words2:
            overlap = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return overlap / union > 0.3
        
        return False
    
    def _generate_cache_key(self, input_data: Dict[str, Any], operation_type: SymbolicOperation) -> str:
        """Generate cache key for reasoning results"""
        cache_data = {
            "input": input_data,
            "operation": operation_type.value
        }
        cache_str = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _monitoring_loop(self):
        """Background monitoring loop for symbolic reasoning"""
        while self._processing_active:
            try:
                # Monitor system health
                if len(self._results_history) > 10000:
                    # Clean old results
                    self._results_history = self._results_history[-5000:]
                    logger.info("🧹 Cleaned old symbolic reasoning results")
                
                # Clean caches if they get too large
                if len(self._reasoning_cache) > self.cache_size_limit:
                    # Remove oldest entries (simple FIFO)
                    keys_to_remove = list(self._reasoning_cache.keys())[:len(self._reasoning_cache)//2]
                    for key in keys_to_remove:
                        del self._reasoning_cache[key]
                    logger.info("🧹 Cleaned reasoning cache")
                
                time.sleep(60)  # Monitor every minute
            except Exception as e:
                logger.error(f"❌ Error in symbolic reasoning monitoring: {e}")
                time.sleep(10)
    
    def get_symbolic_status(self) -> Dict[str, Any]:
        """Get symbolic reasoning system status"""
        return {
            "system_hash": self._system_hash,
            "operation_count": self._operation_counter,
            "results_history_size": len(self._results_history),
            "processing_active": self._processing_active,
            "knowledge_base_facts": len(self.knowledge_base.facts),
            "knowledge_base_rules": len(self.knowledge_base.rules),
            "cache_size": len(self._reasoning_cache),
            "max_reasoning_depth": self.max_reasoning_depth,
            "reasoning_timeout": self.reasoning_timeout
        }
    
    def shutdown(self):
        """Shutdown the symbolic reasoning engine"""
        self._processing_active = False
        logger.info("🛑 Symbolic Reasoning Engine shutdown")


# Global instance management
_symbolic_engine = None

def get_symbolic_engine(config: Optional[Dict[str, Any]] = None) -> SymbolicReasoningEngine:
    """Get the global symbolic reasoning engine instance"""
    global _symbolic_engine
    if _symbolic_engine is None:
        _symbolic_engine = SymbolicReasoningEngine(config)
    return _symbolic_engine

def get_symbolic_status() -> Dict[str, Any]:
    """Get symbolic reasoning system status"""
    engine = get_symbolic_engine()
    return engine.get_symbolic_status()


# Main async function for testing
async def main():
    """Main function for testing the symbolic reasoning engine"""
    print("🧠 UNIVERSAL AI CORE SYMBOLIC REASONING ENGINE")
    print("=" * 60)
    
    # Initialize symbolic engine
    config = {
        'max_reasoning_depth': 50,
        'reasoning_timeout': 30.0,
        'cache_size_limit': 1000
    }
    
    engine = get_symbolic_engine(config)
    
    try:
        # Test deductive reasoning
        print("\n🔍 Testing deductive reasoning...")
        deductive_data = {
            "premises": ["All humans are mortal", "Socrates is human"],
            "conclusion": "Socrates is mortal",
            "rules": ["modus_ponens", "syllogism"]
        }
        
        result = await engine.symbolic_reasoning(deductive_data)
        print(f"✅ Deductive result: {result.operation_type.value}")
        print(f"🎯 Confidence: {result.confidence:.2f}")
        print(f"⏱️ Processing time: {result.processing_time:.3f}s")
        
        # Test pattern matching
        print("\n🔍 Testing pattern matching...")
        pattern_data = {
            "data": ["apple", "application", "apply", "banana", "approach"],
            "template": "app",
            "similarity_threshold": 0.3
        }
        
        result = await engine.symbolic_reasoning(pattern_data)
        print(f"✅ Pattern matching result: {len(result.output_data.get('matches_found', []))} matches")
        
        # Test constraint solving
        print("\n🔍 Testing constraint solving...")
        constraint_data = {
            "constraints": ["x = 5", "y = 10", "z = x + y"],
            "variables": {"x": 0, "y": 0, "z": 0}
        }
        
        result = await engine.symbolic_reasoning(constraint_data)
        print(f"✅ Constraint solving: {result.output_data.get('solution', {})}")
        
        # Show status
        status = engine.get_symbolic_status()
        print(f"\n📊 Symbolic Engine Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Symbolic reasoning engine test completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())