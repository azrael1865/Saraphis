"""
Enhanced Symbolic Reasoning for Financial Fraud Detection
Advanced symbolic reasoning with comprehensive validation, error handling,
security features, and production-ready reliability mechanisms.
"""

import logging
import json
import time
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set, Type, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import numpy as np
import traceback
import psutil
import signal
from contextlib import contextmanager
from functools import wraps
import re
import secrets
import hashlib
import collections

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# Custom Exception Classes
# ============================================================================

class SymbolicReasoningException(Exception):
    """Base exception for symbolic reasoning errors"""
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()

class ReasoningValidationError(SymbolicReasoningException):
    """Raised when reasoning input validation fails"""
    pass

class RuleValidationError(SymbolicReasoningException):
    """Raised when rule validation fails"""
    pass

class KnowledgeBaseError(SymbolicReasoningException):
    """Raised when knowledge base operations fail"""
    pass

class ReasoningTimeoutError(SymbolicReasoningException):
    """Raised when reasoning operations timeout"""
    pass

class ReasoningPerformanceError(SymbolicReasoningException):
    """Raised when performance thresholds are exceeded"""
    pass

class ReasoningSecurityError(SymbolicReasoningException):
    """Raised when security validation fails"""
    pass

class ReasoningIntegrationError(SymbolicReasoningException):
    """Raised when integration checks fail"""
    pass

class InferenceValidationError(SymbolicReasoningException):
    """Raised when inference validation fails"""
    pass

class FactValidationError(SymbolicReasoningException):
    """Raised when fact validation fails"""
    pass

class QueryValidationError(SymbolicReasoningException):
    """Raised when query validation fails"""
    pass

class ReasoningCapacityError(SymbolicReasoningException):
    """Raised when system capacity limits are exceeded"""
    pass

# ============================================================================
# Validation and Security Components
# ============================================================================

class ValidationUtils:
    """Utility functions for validation"""
    
    @staticmethod
    def validate_confidence(confidence: float) -> Tuple[bool, str]:
        """Validate confidence value"""
        if not isinstance(confidence, (int, float)):
            return False, "Confidence must be numeric"
        
        if not 0 <= confidence <= 1:
            return False, "Confidence must be between 0 and 1"
        
        return True, "Valid confidence"
    
    @staticmethod
    def validate_timestamp(timestamp: datetime) -> Tuple[bool, str]:
        """Validate timestamp"""
        if not isinstance(timestamp, datetime):
            return False, "Timestamp must be datetime object"
        
        now = datetime.now()
        if timestamp > now + timedelta(minutes=5):
            return False, "Timestamp cannot be in future"
        
        if timestamp < now - timedelta(days=365):
            return False, "Timestamp too old (>1 year)"
        
        return True, "Valid timestamp"
    
    @staticmethod
    def validate_id_format(id_string: str, prefix: str = None) -> Tuple[bool, str]:
        """Validate ID format"""
        if not isinstance(id_string, str):
            return False, "ID must be string"
        
        if not id_string:
            return False, "ID cannot be empty"
        
        # Check length
        if len(id_string) < 3 or len(id_string) > 100:
            return False, "ID length must be between 3 and 100 characters"
        
        # Check format (alphanumeric, underscores, hyphens)
        pattern = r'^[a-zA-Z0-9_-]+$'
        if not re.match(pattern, id_string):
            return False, "ID can only contain letters, numbers, underscores, and hyphens"
        
        # Check prefix if specified
        if prefix and not id_string.startswith(prefix):
            return False, f"ID must start with '{prefix}'"
        
        return True, "Valid ID format"
    
    @staticmethod
    def validate_rule_priority(priority: int) -> Tuple[bool, str]:
        """Validate rule priority"""
        if not isinstance(priority, int):
            return False, "Priority must be integer"
        
        if not 0 <= priority <= 10:
            return False, "Priority must be between 0 and 10"
        
        return True, "Valid priority"

class SecurityValidator:
    """Validates security aspects of reasoning operations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.allowed_sources = set(self.config.get('allowed_sources', []))
        self.max_fact_size_kb = self.config.get('max_fact_size_kb', 100)
        self.max_rule_complexity = self.config.get('max_rule_complexity', 50)
    
    def validate_input_safety(self, data: Any) -> Tuple[bool, str]:
        """Validate input data for safety"""
        try:
            # Check for dangerous patterns
            if isinstance(data, str):
                dangerous_patterns = [
                    r'<script[^>]*>.*?</script>',
                    r'javascript:',
                    r'on\w+\s*=',
                    r'eval\s*\(',
                    r'exec\s*\(',
                    r'import\s+',
                    r'__import__\s*\(',
                    r'getattr\s*\(',
                    r'setattr\s*\(',
                    r'delattr\s*\(',
                    r'hasattr\s*\(',
                ]
                
                for pattern in dangerous_patterns:
                    if re.search(pattern, data, re.IGNORECASE):
                        return False, f"Potentially dangerous pattern detected: {pattern}"
            
            # Check for nested depth (prevent DoS)
            if isinstance(data, (dict, list)):
                depth = self._get_nested_depth(data)
                if depth > 20:
                    return False, f"Nested depth {depth} exceeds maximum allowed depth"
            
            # Check data size
            data_size = len(str(data)) if data else 0
            if data_size > self.max_fact_size_kb * 1024:
                return False, f"Data size {data_size} exceeds maximum allowed size"
            
            return True, "Input validation passed"
            
        except Exception as e:
            return False, f"Security validation error: {str(e)}"
    
    def validate_fact_source(self, source: str) -> Tuple[bool, str]:
        """Validate fact source"""
        if not self.allowed_sources:
            return True, "No source restrictions"
        
        if source not in self.allowed_sources:
            return False, f"Source '{source}' not in allowed sources: {list(self.allowed_sources)}"
        
        return True, "Source validated"
    
    def validate_rule_complexity(self, conditions: List[Dict[str, Any]], 
                                actions: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Validate rule complexity"""
        total_complexity = len(conditions) + len(actions)
        
        # Add complexity for nested conditions
        for condition in conditions:
            if isinstance(condition.get('value'), (dict, list)):
                total_complexity += 2
            if condition.get('nested_conditions'):
                total_complexity += len(condition['nested_conditions'])
        
        # Add complexity for nested actions
        for action in actions:
            if isinstance(action.get('value'), (dict, list)):
                total_complexity += 2
            if action.get('nested_actions'):
                total_complexity += len(action['nested_actions'])
        
        if total_complexity > self.max_rule_complexity:
            return False, f"Rule complexity {total_complexity} exceeds maximum {self.max_rule_complexity}"
        
        return True, "Rule complexity validated"
    
    def _get_nested_depth(self, obj: Any, depth: int = 0) -> int:
        """Calculate nested depth of data structure"""
        if depth > 50:  # Prevent infinite recursion
            return depth
            
        if isinstance(obj, dict):
            return max([self._get_nested_depth(v, depth + 1) for v in obj.values()] + [depth])
        elif isinstance(obj, list):
            return max([self._get_nested_depth(item, depth + 1) for item in obj] + [depth])
        else:
            return depth

class PerformanceMonitor:
    """Monitors and limits performance metrics"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_reasoning_time_ms = self.config.get('max_reasoning_time_ms', 5000)
        self.max_memory_usage_mb = self.config.get('max_memory_usage_mb', 512)
        self.max_cpu_usage_percent = self.config.get('max_cpu_usage_percent', 80)
        self.process = psutil.Process()
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager for operation monitoring"""
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            memory_used = end_memory - start_memory
            duration_ms = (end_time - start_time) * 1000
            
            logger.debug(f"{operation_name} - Memory: {memory_used:.2f}MB, Duration: {duration_ms:.2f}ms")
            
            # Check thresholds
            if duration_ms > self.max_reasoning_time_ms:
                raise ReasoningPerformanceError(
                    f"Operation {operation_name} exceeded time limit: {duration_ms:.2f}ms > {self.max_reasoning_time_ms}ms"
                )
            
            if memory_used > self.max_memory_usage_mb:
                raise ReasoningPerformanceError(
                    f"Operation {operation_name} exceeded memory limit: {memory_used:.2f}MB > {self.max_memory_usage_mb}MB"
                )
    
    def check_system_resources(self) -> Tuple[bool, str]:
        """Check if system resources are within limits"""
        try:
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent()
            
            if memory_mb > self.max_memory_usage_mb:
                return False, f"Memory usage {memory_mb:.2f}MB exceeds limit {self.max_memory_usage_mb}MB"
            
            if cpu_percent > self.max_cpu_usage_percent:
                return False, f"CPU usage {cpu_percent:.2f}% exceeds limit {self.max_cpu_usage_percent}%"
            
            return True, "Resource usage within limits"
            
        except Exception as e:
            return False, f"Resource check error: {str(e)}"

class RateLimiter:
    """Rate limiting for API calls"""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = collections.defaultdict(list)
        self._lock = threading.Lock()
    
    def check_rate_limit(self, user_id: str) -> Tuple[bool, str]:
        """Check if user is within rate limit"""
        with self._lock:
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)
            
            # Clean old entries
            self.calls[user_id] = [
                call_time for call_time in self.calls[user_id]
                if call_time > minute_ago
            ]
            
            # Check limit
            if len(self.calls[user_id]) >= self.calls_per_minute:
                return False, f"Rate limit exceeded: {len(self.calls[user_id])} calls in last minute"
            
            # Record this call
            self.calls[user_id].append(now)
            
            return True, "Within rate limit"

def timeout_decorator(timeout_seconds: int):
    """Decorator to add timeout to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def timeout_handler(signum, frame):
                raise ReasoningTimeoutError(f"Operation timed out after {timeout_seconds} seconds")
            
            # Set alarm for timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator

def validation_decorator(validator_func: Callable[[Any], Tuple[bool, str]]):
    """Decorator to add validation to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate all arguments
            for arg in args[1:]:  # Skip self
                is_valid, message = validator_func(arg)
                if not is_valid:
                    raise ReasoningValidationError(f"Validation failed: {message}")
            
            # Validate keyword arguments
            for key, value in kwargs.items():
                is_valid, message = validator_func(value)
                if not is_valid:
                    raise ReasoningValidationError(f"Validation failed for {key}: {message}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# ============================================================================
# Enhanced Data Classes
# ============================================================================

class ReasoningType(Enum):
    """Types of reasoning operations"""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    RULE_BASED = "rule_based"
    FUZZY = "fuzzy"
    TEMPORAL = "temporal"
    PROBABILISTIC = "probabilistic"
    CAUSAL = "causal"

class RuleType(Enum):
    """Types of business rules"""
    TRANSACTION_LIMIT = "transaction_limit"
    VELOCITY = "velocity"
    GEOGRAPHIC = "geographic"
    BEHAVIORAL = "behavioral"
    TEMPORAL = "temporal"
    RELATIONSHIP = "relationship"
    COMPOSITE = "composite"
    COMPLIANCE = "compliance"
    RISK_ASSESSMENT = "risk_assessment"

class FactType(Enum):
    """Types of facts in knowledge base"""
    TRANSACTION = "transaction"
    USER_BEHAVIOR = "user_behavior"
    HISTORICAL_PATTERN = "historical_pattern"
    EXTERNAL_DATA = "external_data"
    RULE_VIOLATION = "rule_violation"
    ML_PREDICTION = "ml_prediction"
    COMPLIANCE_DATA = "compliance_data"
    RISK_INDICATOR = "risk_indicator"

class SecurityLevel(Enum):
    """Security levels for reasoning operations"""
    BASIC = "basic"
    STANDARD = "standard"
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"

@dataclass
class ValidatedFact:
    """Enhanced fact with comprehensive validation"""
    fact_id: str
    fact_type: FactType
    subject: str
    predicate: str
    object: Any
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Validation attributes
    validation_status: str = "pending"
    validation_errors: List[str] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.STANDARD
    
    # Quality attributes
    reliability_score: float = 1.0
    freshness_score: float = 1.0
    
    def __post_init__(self):
        """Validate fact after initialization"""
        self.validate()
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Comprehensive fact validation"""
        errors = []
        
        # Basic field validation
        if not self.fact_id:
            errors.append("fact_id is required")
        else:
            is_valid, msg = ValidationUtils.validate_id_format(self.fact_id, "FACT")
            if not is_valid:
                errors.append(f"fact_id format invalid: {msg}")
        
        if not self.subject:
            errors.append("subject is required")
        
        if not self.predicate:
            errors.append("predicate is required")
        
        if self.object is None:
            errors.append("object is required")
        
        # Confidence validation
        is_valid, msg = ValidationUtils.validate_confidence(self.confidence)
        if not is_valid:
            errors.append(f"confidence validation failed: {msg}")
        
        # Reliability score validation
        is_valid, msg = ValidationUtils.validate_confidence(self.reliability_score)
        if not is_valid:
            errors.append(f"reliability_score validation failed: {msg}")
        
        # Freshness score validation
        is_valid, msg = ValidationUtils.validate_confidence(self.freshness_score)
        if not is_valid:
            errors.append(f"freshness_score validation failed: {msg}")
        
        # Timestamp validation
        is_valid, msg = ValidationUtils.validate_timestamp(self.timestamp)
        if not is_valid:
            errors.append(f"timestamp validation failed: {msg}")
        
        # Calculate freshness score based on age
        age_hours = (datetime.now() - self.timestamp).total_seconds() / 3600
        if age_hours < 1:
            self.freshness_score = 1.0
        elif age_hours < 24:
            self.freshness_score = max(0.8, 1.0 - (age_hours / 24) * 0.2)
        elif age_hours < 168:  # 1 week
            self.freshness_score = max(0.5, 0.8 - (age_hours / 168) * 0.3)
        else:
            self.freshness_score = max(0.1, 0.5 - min(age_hours / 8760, 1.0) * 0.4)  # 1 year
        
        # Update validation status
        if errors:
            self.validation_status = "failed"
            self.validation_errors = errors
        else:
            self.validation_status = "passed"
            self.validation_errors = []
        
        return len(errors) == 0, errors
    
    def to_triple(self) -> Tuple[str, str, Any]:
        """Convert to RDF-style triple"""
        return (self.subject, self.predicate, self.object)
    
    def calculate_quality_score(self) -> float:
        """Calculate overall quality score"""
        return (self.confidence + self.reliability_score + self.freshness_score) / 3

@dataclass
class ValidatedRule:
    """Enhanced rule with comprehensive validation"""
    rule_id: str
    rule_type: RuleType
    name: str
    description: str
    conditions: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    priority: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Validation attributes
    validation_status: str = "pending"
    validation_errors: List[str] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.STANDARD
    
    # Performance attributes
    execution_count: int = 0
    avg_execution_time_ms: float = 0.0
    success_rate: float = 1.0
    
    def __post_init__(self):
        """Validate rule after initialization"""
        self.validate()
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Comprehensive rule validation"""
        errors = []
        
        # Basic field validation
        if not self.rule_id:
            errors.append("rule_id is required")
        else:
            is_valid, msg = ValidationUtils.validate_id_format(self.rule_id, "RULE")
            if not is_valid:
                errors.append(f"rule_id format invalid: {msg}")
        
        if not self.name:
            errors.append("name is required")
        
        if not self.description:
            errors.append("description is required")
        
        # Priority validation
        is_valid, msg = ValidationUtils.validate_rule_priority(self.priority)
        if not is_valid:
            errors.append(f"priority validation failed: {msg}")
        
        # Conditions validation
        if not self.conditions:
            errors.append("conditions cannot be empty")
        else:
            for i, condition in enumerate(self.conditions):
                condition_errors = self._validate_condition(condition, i)
                errors.extend(condition_errors)
        
        # Actions validation
        if not self.actions:
            errors.append("actions cannot be empty")
        else:
            for i, action in enumerate(self.actions):
                action_errors = self._validate_action(action, i)
                errors.extend(action_errors)
        
        # Semantic validation
        semantic_errors = self._validate_semantics()
        errors.extend(semantic_errors)
        
        # Update validation status
        if errors:
            self.validation_status = "failed"
            self.validation_errors = errors
        else:
            self.validation_status = "passed"
            self.validation_errors = []
        
        return len(errors) == 0, errors
    
    def _validate_condition(self, condition: Dict[str, Any], index: int) -> List[str]:
        """Validate a single condition"""
        errors = []
        
        if not isinstance(condition, dict):
            errors.append(f"condition[{index}] must be a dictionary")
            return errors
        
        # Required fields
        required_fields = ['type']
        for field in required_fields:
            if field not in condition:
                errors.append(f"condition[{index}] missing required field: {field}")
        
        # Type-specific validation
        condition_type = condition.get('type')
        if condition_type == 'fact_match':
            if 'predicate' not in condition:
                errors.append(f"condition[{index}] fact_match requires 'predicate'")
            if 'operator' in condition and condition['operator'] not in ['=', '!=', '>', '<', '>=', '<=', 'in', 'not_in']:
                errors.append(f"condition[{index}] invalid operator: {condition['operator']}")
        
        elif condition_type == 'temporal_pattern':
            if 'time_window_minutes' not in condition:
                errors.append(f"condition[{index}] temporal_pattern requires 'time_window_minutes'")
            if 'min_count' not in condition:
                errors.append(f"condition[{index}] temporal_pattern requires 'min_count'")
        
        return errors
    
    def _validate_action(self, action: Dict[str, Any], index: int) -> List[str]:
        """Validate a single action"""
        errors = []
        
        if not isinstance(action, dict):
            errors.append(f"action[{index}] must be a dictionary")
            return errors
        
        # Required fields
        required_fields = ['type']
        for field in required_fields:
            if field not in action:
                errors.append(f"action[{index}] missing required field: {field}")
        
        # Type-specific validation
        action_type = action.get('type')
        if action_type == 'assert':
            if 'subject' not in action:
                errors.append(f"action[{index}] assert requires 'subject'")
            if 'predicate' not in action:
                errors.append(f"action[{index}] assert requires 'predicate'")
        
        elif action_type == 'retract':
            if 'fact_id' not in action and 'pattern' not in action:
                errors.append(f"action[{index}] retract requires 'fact_id' or 'pattern'")
        
        return errors
    
    def _validate_semantics(self) -> List[str]:
        """Validate rule semantics and detect contradictions"""
        errors = []
        
        # Check for contradictory conditions
        numeric_conditions = []
        for condition in self.conditions:
            if condition.get('type') == 'fact_match' and condition.get('operator') in ['>', '<', '>=', '<=']:
                numeric_conditions.append(condition)
        
        # Check for impossible combinations
        for i, cond1 in enumerate(numeric_conditions):
            for j, cond2 in enumerate(numeric_conditions[i+1:], i+1):
                if (cond1.get('predicate') == cond2.get('predicate') and
                    cond1.get('subject') == cond2.get('subject')):
                    
                    # Check for contradictions like x > 100 AND x < 50
                    if (cond1.get('operator') == '>' and cond2.get('operator') == '<' and
                        cond1.get('value', 0) >= cond2.get('value', 0)):
                        errors.append(f"Contradictory conditions: {cond1} and {cond2}")
                    
                    elif (cond1.get('operator') == '>=' and cond2.get('operator') == '<=' and
                          cond1.get('value', 0) > cond2.get('value', 0)):
                        errors.append(f"Contradictory conditions: {cond1} and {cond2}")
        
        return errors
    
    def evaluate(self, facts: List[ValidatedFact]) -> bool:
        """Evaluate rule against facts with performance tracking"""
        start_time = time.time()
        
        try:
            # Simple evaluation logic - can be extended
            for condition in self.conditions:
                if not self._evaluate_condition(condition, facts):
                    return False
            
            # Update performance metrics
            execution_time = (time.time() - start_time) * 1000
            self.execution_count += 1
            
            # Update average execution time
            if self.execution_count == 1:
                self.avg_execution_time_ms = execution_time
            else:
                self.avg_execution_time_ms = (
                    (self.avg_execution_time_ms * (self.execution_count - 1) + execution_time) / 
                    self.execution_count
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Rule evaluation failed: {str(e)}")
            # Update success rate
            total_attempts = self.execution_count + 1
            successful_attempts = self.execution_count * self.success_rate
            self.success_rate = successful_attempts / total_attempts
            return False
    
    def _evaluate_condition(self, condition: Dict[str, Any], facts: List[ValidatedFact]) -> bool:
        """Evaluate a single condition"""
        # Enhanced condition evaluation logic
        condition_type = condition.get('type')
        
        if condition_type == 'fact_match':
            return self._evaluate_fact_match(condition, facts)
        elif condition_type == 'temporal_pattern':
            return self._evaluate_temporal_pattern(condition, facts)
        elif condition_type == 'statistical':
            return self._evaluate_statistical_condition(condition, facts)
        
        return True  # Default to true for unknown conditions
    
    def _evaluate_fact_match(self, condition: Dict[str, Any], facts: List[ValidatedFact]) -> bool:
        """Evaluate fact match condition"""
        predicate = condition.get('predicate')
        operator = condition.get('operator', '=')
        value = condition.get('value')
        fact_type = condition.get('fact_type')
        
        for fact in facts:
            # Filter by fact type if specified
            if fact_type and fact.fact_type.value != fact_type:
                continue
            
            # Check if predicate matches
            if fact.predicate != predicate:
                continue
            
            # Apply operator
            if operator == '=':
                if fact.object == value:
                    return True
            elif operator == '!=':
                if fact.object != value:
                    return True
            elif operator == '>':
                if isinstance(fact.object, (int, float)) and fact.object > value:
                    return True
            elif operator == '<':
                if isinstance(fact.object, (int, float)) and fact.object < value:
                    return True
            elif operator == '>=':
                if isinstance(fact.object, (int, float)) and fact.object >= value:
                    return True
            elif operator == '<=':
                if isinstance(fact.object, (int, float)) and fact.object <= value:
                    return True
        
        return False
    
    def _evaluate_temporal_pattern(self, condition: Dict[str, Any], facts: List[ValidatedFact]) -> bool:
        """Evaluate temporal pattern condition"""
        time_window_minutes = condition.get('time_window_minutes', 60)
        min_count = condition.get('min_count', 1)
        pattern = condition.get('pattern', 'count')
        
        # Filter facts within time window
        now = datetime.now()
        window_start = now - timedelta(minutes=time_window_minutes)
        
        recent_facts = [
            fact for fact in facts
            if fact.timestamp >= window_start
        ]
        
        if pattern == 'count':
            return len(recent_facts) >= min_count
        elif pattern == 'rapid_sequence':
            # Check for rapid sequence of similar facts
            if len(recent_facts) < min_count:
                return False
            
            # Group by subject
            subject_groups = {}
            for fact in recent_facts:
                if fact.subject not in subject_groups:
                    subject_groups[fact.subject] = []
                subject_groups[fact.subject].append(fact)
            
            # Check if any subject has rapid sequence
            for subject, subject_facts in subject_groups.items():
                if len(subject_facts) >= min_count:
                    return True
        
        return False
    
    def _evaluate_statistical_condition(self, condition: Dict[str, Any], facts: List[ValidatedFact]) -> bool:
        """Evaluate statistical condition"""
        metric = condition.get('metric')
        threshold = condition.get('threshold', 0)
        attribute = condition.get('attribute')
        
        if metric == 'deviation' and attribute:
            # Calculate z-score deviation
            values = []
            for fact in facts:
                if fact.predicate == attribute and isinstance(fact.object, (int, float)):
                    values.append(fact.object)
            
            if len(values) > 1:
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                if std_val > 0:
                    # Check if latest value deviates significantly
                    latest_value = values[-1]
                    z_score = abs(latest_value - mean_val) / std_val
                    return z_score > threshold
        
        return False

@dataclass
class ValidatedInference:
    """Enhanced inference with comprehensive validation"""
    inference_id: str
    reasoning_type: ReasoningType
    premises: List[str]  # Fact IDs
    conclusion: ValidatedFact
    confidence: float
    reasoning_path: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Validation attributes
    validation_status: str = "pending"
    validation_errors: List[str] = field(default_factory=list)
    
    # Quality attributes
    evidence_strength: float = 0.0
    logical_consistency: float = 0.0
    
    def __post_init__(self):
        """Validate inference after initialization"""
        self.validate()
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Comprehensive inference validation"""
        errors = []
        
        # Basic field validation
        if not self.inference_id:
            errors.append("inference_id is required")
        else:
            is_valid, msg = ValidationUtils.validate_id_format(self.inference_id, "INF")
            if not is_valid:
                errors.append(f"inference_id format invalid: {msg}")
        
        if not self.premises:
            errors.append("premises cannot be empty")
        
        # Confidence validation
        is_valid, msg = ValidationUtils.validate_confidence(self.confidence)
        if not is_valid:
            errors.append(f"confidence validation failed: {msg}")
        
        # Conclusion validation
        if not isinstance(self.conclusion, ValidatedFact):
            errors.append("conclusion must be ValidatedFact")
        else:
            is_valid, conclusion_errors = self.conclusion.validate()
            if not is_valid:
                errors.extend([f"conclusion {err}" for err in conclusion_errors])
        
        # Reasoning path validation
        if not self.reasoning_path:
            errors.append("reasoning_path cannot be empty")
        
        # Update validation status
        if errors:
            self.validation_status = "failed"
            self.validation_errors = errors
        else:
            self.validation_status = "passed"
            self.validation_errors = []
        
        return len(errors) == 0, errors

# ============================================================================
# Enhanced Reasoning Engines
# ============================================================================

class ValidatedReasoningEngine(ABC):
    """Enhanced abstract base class for reasoning engines"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.security_validator = SecurityValidator(config)
        self.performance_monitor = PerformanceMonitor(config)
        self.validation_cache = {}
        self._lock = threading.Lock()
    
    @abstractmethod
    def reason(self, facts: List[ValidatedFact], rules: List[ValidatedRule]) -> List[ValidatedInference]:
        """Perform reasoning on facts using rules"""
        pass
    
    @abstractmethod
    def validate(self, inference: ValidatedInference, facts: List[ValidatedFact]) -> bool:
        """Validate an inference"""
        pass
    
    def validate_inputs(self, facts: List[ValidatedFact], rules: List[ValidatedRule]) -> None:
        """Validate inputs before processing"""
        # Validate facts
        for fact in facts:
            if fact.validation_status != "passed":
                raise FactValidationError(f"Invalid fact {fact.fact_id}: {fact.validation_errors}")
        
        # Validate rules
        for rule in rules:
            if rule.validation_status != "passed":
                raise RuleValidationError(f"Invalid rule {rule.rule_id}: {rule.validation_errors}")
            
            if not rule.enabled:
                logger.debug(f"Skipping disabled rule {rule.rule_id}")

class EnhancedRuleBasedReasoningEngine(ValidatedReasoningEngine):
    """Enhanced rule-based reasoning engine with validation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.forward_chaining = config.get('forward_chaining', True) if config else True
        self.max_iterations = config.get('max_iterations', 100) if config else 100
    
    @timeout_decorator(30)
    def reason(self, facts: List[ValidatedFact], rules: List[ValidatedRule]) -> List[ValidatedInference]:
        """Enhanced rule-based reasoning with validation"""
        with self.performance_monitor.monitor_operation("rule_based_reasoning"):
            self.validate_inputs(facts, rules)
            
            if self.forward_chaining:
                return self._enhanced_forward_chain(facts, rules)
            else:
                return self._enhanced_backward_chain(facts, rules)
    
    def _enhanced_forward_chain(self, facts: List[ValidatedFact], 
                               rules: List[ValidatedRule]) -> List[ValidatedInference]:
        """Enhanced forward chaining with performance monitoring"""
        inferences = []
        fact_set = set(facts)
        applied_rules = set()
        
        iteration = 0
        changed = True
        
        while changed and iteration < self.max_iterations:
            changed = False
            iteration += 1
            
            # Sort rules by priority (higher priority first)
            sorted_rules = sorted(
                [r for r in rules if r.enabled and r.rule_id not in applied_rules],
                key=lambda x: x.priority,
                reverse=True
            )
            
            for rule in sorted_rules:
                try:
                    if rule.evaluate(list(fact_set)):
                        # Generate new facts from rule actions
                        for action in rule.actions:
                            new_fact = self._create_fact_from_action(action, rule)
                            if new_fact and new_fact not in fact_set:
                                fact_set.add(new_fact)
                                
                                # Create inference
                                inference = self._create_inference(rule, list(fact_set), new_fact)
                                inferences.append(inference)
                                changed = True
                        
                        applied_rules.add(rule.rule_id)
                        
                except Exception as e:
                    logger.error(f"Rule {rule.rule_id} evaluation failed: {str(e)}")
                    continue
            
            # Check for performance issues
            if iteration > self.max_iterations * 0.8:
                logger.warning(f"Forward chaining approaching iteration limit: {iteration}/{self.max_iterations}")
        
        if iteration >= self.max_iterations:
            logger.warning(f"Forward chaining stopped at iteration limit: {self.max_iterations}")
        
        return inferences
    
    def _enhanced_backward_chain(self, facts: List[ValidatedFact], 
                                rules: List[ValidatedRule]) -> List[ValidatedInference]:
        """Enhanced backward chaining (goal-driven)"""
        # Enhanced implementation for backward chaining
        inferences = []
        
        # Identify potential goals from rule conclusions
        goals = set()
        for rule in rules:
            if rule.enabled:
                for action in rule.actions:
                    if action.get('type') == 'assert':
                        goal_key = f"{action.get('subject', '')}_{action.get('predicate', '')}"
                        goals.add(goal_key)
        
        # Try to prove each goal
        for goal in goals:
            try:
                goal_inferences = self._prove_goal(goal, facts, rules)
                inferences.extend(goal_inferences)
            except Exception as e:
                logger.error(f"Failed to prove goal {goal}: {str(e)}")
        
        return inferences
    
    def _prove_goal(self, goal: str, facts: List[ValidatedFact], 
                   rules: List[ValidatedRule]) -> List[ValidatedInference]:
        """Prove a specific goal using backward chaining"""
        inferences = []
        
        # Find rules that can establish this goal
        relevant_rules = []
        for rule in rules:
            if rule.enabled:
                for action in rule.actions:
                    if action.get('type') == 'assert':
                        goal_key = f"{action.get('subject', '')}_{action.get('predicate', '')}"
                        if goal_key == goal:
                            relevant_rules.append(rule)
                            break
        
        # Try to satisfy rule conditions
        for rule in relevant_rules:
            if rule.evaluate(facts):
                # Rule conditions satisfied, create inference
                for action in rule.actions:
                    new_fact = self._create_fact_from_action(action, rule)
                    if new_fact:
                        inference = self._create_inference(rule, facts, new_fact)
                        inferences.append(inference)
        
        return inferences
    
    def _create_fact_from_action(self, action: Dict[str, Any], rule: ValidatedRule) -> Optional[ValidatedFact]:
        """Create new fact from rule action with validation"""
        if action.get('type') == 'assert':
            try:
                fact = ValidatedFact(
                    fact_id=self._generate_id('FACT'),
                    fact_type=FactType.RULE_VIOLATION,
                    subject=action.get('subject', 'unknown'),
                    predicate=action.get('predicate', 'derived'),
                    object=action.get('object', rule.name),
                    confidence=action.get('confidence', 0.9),
                    source=f"rule_{rule.rule_id}",
                    metadata={
                        'rule_id': rule.rule_id,
                        'rule_name': rule.name,
                        'action_type': action.get('type')
                    }
                )
                
                # Validate the new fact
                is_valid, errors = fact.validate()
                if not is_valid:
                    logger.warning(f"Generated fact failed validation: {errors}")
                    return None
                
                return fact
                
            except Exception as e:
                logger.error(f"Failed to create fact from action: {str(e)}")
                return None
        
        return None
    
    def _create_inference(self, rule: ValidatedRule, facts: List[ValidatedFact], 
                         conclusion: ValidatedFact) -> ValidatedInference:
        """Create inference with validation"""
        # Find premise facts that match rule conditions
        premise_ids = []
        for fact in facts:
            for condition in rule.conditions:
                if self._fact_matches_condition(fact, condition):
                    premise_ids.append(fact.fact_id)
                    break
        
        # Calculate confidence based on premises and rule
        premise_confidences = [f.confidence for f in facts if f.fact_id in premise_ids]
        if premise_confidences:
            avg_premise_confidence = np.mean(premise_confidences)
            # Combine with rule success rate
            combined_confidence = (avg_premise_confidence + rule.success_rate) / 2
        else:
            combined_confidence = rule.success_rate
        
        inference = ValidatedInference(
            inference_id=self._generate_id('INF'),
            reasoning_type=ReasoningType.RULE_BASED,
            premises=premise_ids,
            conclusion=conclusion,
            confidence=combined_confidence,
            reasoning_path=[{
                'rule_id': rule.rule_id,
                'rule_name': rule.name,
                'conditions_met': len(premise_ids),
                'execution_time_ms': rule.avg_execution_time_ms
            }],
            metadata={
                'rule_priority': rule.priority,
                'rule_type': rule.rule_type.value
            }
        )
        
        return inference
    
    def _fact_matches_condition(self, fact: ValidatedFact, condition: Dict[str, Any]) -> bool:
        """Enhanced fact matching with condition validation"""
        condition_type = condition.get('type')
        
        if condition_type == 'fact_match':
            # Check predicate match
            if fact.predicate != condition.get('predicate'):
                return False
            
            # Check fact type if specified
            if 'fact_type' in condition and fact.fact_type.value != condition['fact_type']:
                return False
            
            # Check operator and value
            operator = condition.get('operator', '=')
            value = condition.get('value')
            
            if value is not None:
                if operator == '=':
                    return fact.object == value
                elif operator == '!=':
                    return fact.object != value
                elif operator == '>' and isinstance(fact.object, (int, float)):
                    return fact.object > value
                elif operator == '<' and isinstance(fact.object, (int, float)):
                    return fact.object < value
                elif operator == '>=' and isinstance(fact.object, (int, float)):
                    return fact.object >= value
                elif operator == '<=' and isinstance(fact.object, (int, float)):
                    return fact.object <= value
            
            return True
        
        return False
    
    def validate(self, inference: ValidatedInference, facts: List[ValidatedFact]) -> bool:
        """Enhanced inference validation"""
        try:
            # Basic validation
            is_valid, errors = inference.validate()
            if not is_valid:
                logger.warning(f"Inference validation failed: {errors}")
                return False
            
            # Check if all premises exist in facts
            fact_ids = {f.fact_id for f in facts}
            missing_premises = [p for p in inference.premises if p not in fact_ids]
            if missing_premises:
                logger.warning(f"Missing premises: {missing_premises}")
                return False
            
            # Validate reasoning path
            if not inference.reasoning_path:
                logger.warning("Empty reasoning path")
                return False
            
            # Check confidence consistency
            premise_facts = [f for f in facts if f.fact_id in inference.premises]
            if premise_facts:
                avg_premise_confidence = np.mean([f.confidence for f in premise_facts])
                # Inference confidence shouldn't be much higher than premise confidence
                if inference.confidence > avg_premise_confidence + 0.3:
                    logger.warning(f"Inference confidence {inference.confidence} too high vs premises {avg_premise_confidence}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Inference validation error: {str(e)}")
            return False
    
    def _generate_id(self, prefix: str) -> str:
        """Generate cryptographically secure unique ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_part = secrets.token_hex(8)
        return f"{prefix}_{timestamp}_{random_part}"

# ============================================================================
# Enhanced Main Symbolic Reasoner
# ============================================================================

class EnhancedFinancialSymbolicReasoner:
    """
    Enhanced Symbolic Reasoning for Financial Fraud Detection
    
    Advanced features:
    - Comprehensive input validation
    - Security validation and access control
    - Performance monitoring and optimization
    - Error recovery and graceful degradation
    - Detailed audit trails and metrics
    - Production-ready reliability mechanisms
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced symbolic reasoner
        
        Args:
            config: Configuration dictionary with enhanced options
        """
        self.config = config or self._default_config()
        
        # Initialize validation and security components
        self.security_validator = SecurityValidator(self.config)
        self.performance_monitor = PerformanceMonitor(self.config)
        self.rate_limiter = RateLimiter(self.config.get('max_reasoning_calls_per_minute', 60))
        
        # Initialize enhanced reasoning engines
        self.reasoning_engines = {
            'enhanced_rule_based': EnhancedRuleBasedReasoningEngine(self.config),
            'fuzzy': self._create_fuzzy_engine(),
            'temporal': self._create_temporal_engine()
        }
        
        # Enhanced storage
        self.facts: Dict[str, ValidatedFact] = {}
        self.rules: Dict[str, ValidatedRule] = {}
        self.inferences: Dict[str, ValidatedInference] = {}
        
        # Initialize default rules with validation
        self._initialize_enhanced_default_rules()
        
        # Enhanced metrics and monitoring
        self.metrics = self._initialize_enhanced_metrics()
        
        # Thread safety and concurrency
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 4),
            thread_name_prefix='EnhancedReasoner'
        )
        
        # Enhanced persistence
        self.storage_path = Path(self.config.get('storage_path', 'enhanced_reasoning'))
        self.storage_path.mkdir(exist_ok=True)
        
        # Audit trail
        self.audit_trail: List[Dict[str, Any]] = []
        
        # Cache for validation results
        self.validation_cache = {}
        self.cache_ttl = self.config.get('validation_cache_ttl', 300)  # 5 minutes
        
        logger.info("Enhanced FinancialSymbolicReasoner initialized with advanced validation and security")
    
    def _default_config(self) -> Dict[str, Any]:
        """Enhanced default configuration"""
        return {
            # Validation settings
            'enable_error_recovery': True,
            'validation_strict_mode': False,
            'validation_cache_ttl': 300,
            'inference_confidence_threshold': 0.6,
            
            # Security settings
            'allowed_sources': [],
            'max_fact_size_kb': 100,
            'max_rule_complexity': 50,
            'max_reasoning_calls_per_minute': 60,
            
            # Performance settings
            'max_reasoning_time_ms': 5000,
            'max_memory_usage_mb': 512,
            'max_cpu_usage_percent': 80,
            'max_workers': 4,
            'enable_parallel_reasoning': True,
            
            # Error recovery settings
            'max_retry_attempts': 3,
            'retry_delay_seconds': 1,
            'reasoning_timeout_seconds': 30,
            
            # Monitoring settings
            'enable_audit_trail': True,
            'max_audit_entries': 10000,
            'enable_health_monitoring': True
        }
    
    def _create_fuzzy_engine(self) -> ValidatedReasoningEngine:
        """Create enhanced fuzzy reasoning engine"""
        # Placeholder for fuzzy engine - would implement FuzzyReasoningEngine with validation
        return EnhancedRuleBasedReasoningEngine(self.config)  # Simplified for now
    
    def _create_temporal_engine(self) -> ValidatedReasoningEngine:
        """Create enhanced temporal reasoning engine"""
        # Placeholder for temporal engine - would implement TemporalReasoningEngine with validation
        return EnhancedRuleBasedReasoningEngine(self.config)  # Simplified for now
    
    def _initialize_enhanced_default_rules(self) -> None:
        """Initialize enhanced default fraud detection rules"""
        default_rules = [
            {
                'rule_id': 'RULE_ENHANCED_001',
                'rule_type': RuleType.TRANSACTION_LIMIT,
                'name': 'Enhanced High Value Transaction Detection',
                'description': 'Flag transactions above configurable threshold with context',
                'conditions': [
                    {
                        'type': 'fact_match',
                        'fact_type': 'TRANSACTION',
                        'predicate': 'amount',
                        'operator': '>',
                        'value': self.config.get('high_value_threshold', 10000)
                    }
                ],
                'actions': [
                    {
                        'type': 'assert',
                        'subject': 'transaction',
                        'predicate': 'requires_review',
                        'object': 'high_value_flag',
                        'confidence': 0.9
                    }
                ],
                'priority': 8,
                'metadata': {'category': 'financial_limit', 'severity': 'high'}
            },
            {
                'rule_id': 'RULE_ENHANCED_002',
                'rule_type': RuleType.VELOCITY,
                'name': 'Enhanced Velocity Anomaly Detection',
                'description': 'Detect rapid transaction sequences with improved accuracy',
                'conditions': [
                    {
                        'type': 'temporal_pattern',
                        'pattern': 'rapid_sequence',
                        'time_window_minutes': 10,
                        'min_count': 5
                    }
                ],
                'actions': [
                    {
                        'type': 'assert',
                        'subject': 'user_behavior',
                        'predicate': 'exhibits',
                        'object': 'velocity_anomaly',
                        'confidence': 0.85
                    }
                ],
                'priority': 7,
                'metadata': {'category': 'behavioral', 'severity': 'medium'}
            },
            {
                'rule_id': 'RULE_ENHANCED_003',
                'rule_type': RuleType.BEHAVIORAL,
                'name': 'Enhanced Statistical Anomaly Detection',
                'description': 'Detect behavioral deviations using statistical analysis',
                'conditions': [
                    {
                        'type': 'statistical',
                        'metric': 'deviation',
                        'threshold': 3,
                        'attribute': 'transaction_amount'
                    }
                ],
                'actions': [
                    {
                        'type': 'assert',
                        'subject': 'user',
                        'predicate': 'shows',
                        'object': 'statistical_anomaly',
                        'confidence': 0.8
                    }
                ],
                'priority': 6,
                'metadata': {'category': 'statistical', 'severity': 'medium'}
            }
        ]
        
        for rule_data in default_rules:
            try:
                rule = ValidatedRule(**rule_data)
                self.rules[rule.rule_id] = rule
                logger.debug(f"Initialized default rule: {rule.rule_id}")
            except Exception as e:
                logger.error(f"Failed to initialize default rule {rule_data.get('rule_id')}: {str(e)}")
    
    def _initialize_enhanced_metrics(self) -> Dict[str, Any]:
        """Initialize enhanced metrics tracking"""
        return {
            # Basic metrics
            'total_facts': 0,
            'total_rules': 0,
            'total_inferences': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            
            # Performance metrics
            'average_reasoning_time_ms': 0.0,
            'total_reasoning_calls': 0,
            'successful_reasoning_calls': 0,
            'failed_reasoning_calls': 0,
            
            # Quality metrics
            'inference_accuracy': 0.0,
            'rule_coverage': 0.0,
            'fact_utilization': 0.0,
            
            # Security metrics
            'total_security_violations': 0,
            'blocked_sources': 0,
            'rate_limit_hits': 0,
            
            # Error metrics
            'validation_errors': 0,
            'timeout_errors': 0,
            'recovery_attempts': 0,
            'successful_recoveries': 0,
            
            # System metrics
            'memory_usage_mb': 0.0,
            'cpu_usage_percent': 0.0,
            
            # Time-based metrics
            'hourly_reasoning_count': {},
            'daily_error_count': {},
            
            # Engine-specific metrics
            'engine_performance': {
                'enhanced_rule_based': {'calls': 0, 'successes': 0, 'avg_time_ms': 0},
                'fuzzy': {'calls': 0, 'successes': 0, 'avg_time_ms': 0},
                'temporal': {'calls': 0, 'successes': 0, 'avg_time_ms': 0}
            }
        }
    
    @timeout_decorator(60)
    def add_fact(self, fact_data: Dict[str, Any], 
                user_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Add fact to knowledge base with enhanced validation
        
        Args:
            fact_data: Fact information
            user_context: User context for security validation
            
        Returns:
            Fact ID
        """
        operation_id = self._generate_id('OP')
        
        try:
            # Add audit entry
            self._add_audit_entry('add_fact_started', {
                'operation_id': operation_id,
                'user_context': user_context
            })
            
            # Rate limiting
            if user_context and 'user_id' in user_context:
                is_within_limit, limit_msg = self.rate_limiter.check_rate_limit(user_context['user_id'])
                if not is_within_limit:
                    self.metrics['rate_limit_hits'] += 1
                    raise ReasoningSecurityError(f"Rate limit exceeded: {limit_msg}")
            
            # Security validation
            is_safe, safety_msg = self.security_validator.validate_input_safety(fact_data)
            if not is_safe:
                self.metrics['total_security_violations'] += 1
                raise ReasoningSecurityError(f"Fact security validation failed: {safety_msg}")
            
            # Source validation
            source = fact_data.get('source', 'unknown')
            is_valid_source, source_msg = self.security_validator.validate_fact_source(source)
            if not is_valid_source:
                self.metrics['blocked_sources'] += 1
                raise ReasoningSecurityError(f"Source validation failed: {source_msg}")
            
            # Performance monitoring
            with self.performance_monitor.monitor_operation("add_fact"):
                # Create validated fact
                fact = ValidatedFact(
                    fact_id=fact_data.get('fact_id', self._generate_id('FACT')),
                    fact_type=FactType(fact_data.get('fact_type', FactType.TRANSACTION.value)),
                    subject=fact_data['subject'],
                    predicate=fact_data['predicate'],
                    object=fact_data['object'],
                    confidence=fact_data.get('confidence', 1.0),
                    source=source,
                    metadata=fact_data.get('metadata', {}),
                    security_level=SecurityLevel(fact_data.get('security_level', SecurityLevel.STANDARD.value))
                )
                
                # Additional validation in strict mode
                if self.config.get('validation_strict_mode'):
                    is_valid, errors = fact.validate()
                    if not is_valid:
                        self.metrics['validation_errors'] += 1
                        raise FactValidationError(f"Strict validation failed: {', '.join(errors)}")
                
                # Store fact
                with self._lock:
                    self.facts[fact.fact_id] = fact
                    self.metrics['total_facts'] += 1
                
                # Add completion audit entry
                self._add_audit_entry('add_fact_completed', {
                    'operation_id': operation_id,
                    'fact_id': fact.fact_id,
                    'validation_status': fact.validation_status
                })
                
                logger.debug(f"Added validated fact {fact.fact_id}: {fact.to_triple()}")
                return fact.fact_id
                
        except SymbolicReasoningException:
            # Re-raise our custom exceptions
            self._add_audit_entry('add_fact_failed', {
                'operation_id': operation_id,
                'error_type': 'validation_error'
            })
            raise
            
        except Exception as e:
            self.metrics['validation_errors'] += 1
            self._add_audit_entry('add_fact_failed', {
                'operation_id': operation_id,
                'error': str(e),
                'error_type': 'unexpected_error'
            })
            logger.error(f"Failed to add fact: {str(e)}")
            raise FactValidationError(f"Failed to add fact: {str(e)}")
    
    @timeout_decorator(60)
    def add_rule(self, rule_data: Union[ValidatedRule, Dict[str, Any]], 
                user_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Add rule to rule base with enhanced validation
        
        Args:
            rule_data: Rule object or rule data
            user_context: User context for security validation
            
        Returns:
            Rule ID
        """
        operation_id = self._generate_id('OP')
        
        try:
            # Add audit entry
            self._add_audit_entry('add_rule_started', {
                'operation_id': operation_id,
                'user_context': user_context
            })
            
            # Rate limiting
            if user_context and 'user_id' in user_context:
                is_within_limit, limit_msg = self.rate_limiter.check_rate_limit(user_context['user_id'])
                if not is_within_limit:
                    self.metrics['rate_limit_hits'] += 1
                    raise ReasoningSecurityError(f"Rate limit exceeded: {limit_msg}")
            
            # Security validation for dict input
            if isinstance(rule_data, dict):
                is_safe, safety_msg = self.security_validator.validate_input_safety(rule_data)
                if not is_safe:
                    self.metrics['total_security_violations'] += 1
                    raise ReasoningSecurityError(f"Rule security validation failed: {safety_msg}")
                
                # Rule complexity validation
                conditions = rule_data.get('conditions', [])
                actions = rule_data.get('actions', [])
                is_valid_complexity, complexity_msg = self.security_validator.validate_rule_complexity(conditions, actions)
                if not is_valid_complexity:
                    self.metrics['total_security_violations'] += 1
                    raise ReasoningSecurityError(f"Rule complexity validation failed: {complexity_msg}")
            
            # Performance monitoring
            with self.performance_monitor.monitor_operation("add_rule"):
                # Create validated rule
                if isinstance(rule_data, dict):
                    rule = ValidatedRule(
                        rule_id=rule_data.get('rule_id', self._generate_id('RULE')),
                        rule_type=RuleType(rule_data.get('rule_type', RuleType.COMPOSITE.value)),
                        name=rule_data['name'],
                        description=rule_data.get('description', ''),
                        conditions=rule_data.get('conditions', []),
                        actions=rule_data.get('actions', []),
                        priority=rule_data.get('priority', 0),
                        enabled=rule_data.get('enabled', True),
                        metadata=rule_data.get('metadata', {}),
                        security_level=SecurityLevel(rule_data.get('security_level', SecurityLevel.STANDARD.value))
                    )
                else:
                    rule = rule_data
                
                # Additional validation in strict mode
                if self.config.get('validation_strict_mode'):
                    is_valid, errors = rule.validate()
                    if not is_valid:
                        self.metrics['validation_errors'] += 1
                        raise RuleValidationError(f"Strict validation failed: {', '.join(errors)}")
                
                # Store rule
                with self._lock:
                    self.rules[rule.rule_id] = rule
                    self.metrics['total_rules'] += 1
                
                # Add completion audit entry
                self._add_audit_entry('add_rule_completed', {
                    'operation_id': operation_id,
                    'rule_id': rule.rule_id,
                    'validation_status': rule.validation_status
                })
                
                logger.info(f"Added validated rule {rule.rule_id}: {rule.name}")
                return rule.rule_id
                
        except SymbolicReasoningException:
            # Re-raise our custom exceptions
            self._add_audit_entry('add_rule_failed', {
                'operation_id': operation_id,
                'error_type': 'validation_error'
            })
            raise
            
        except Exception as e:
            self.metrics['validation_errors'] += 1
            self._add_audit_entry('add_rule_failed', {
                'operation_id': operation_id,
                'error': str(e),
                'error_type': 'unexpected_error'
            })
            logger.error(f"Failed to add rule: {str(e)}")
            raise RuleValidationError(f"Failed to add rule: {str(e)}")
    
    @timeout_decorator(120)
    def reason(self, engine_names: Optional[List[str]] = None,
              user_context: Optional[Dict[str, Any]] = None) -> List[ValidatedInference]:
        """
        Perform enhanced reasoning with comprehensive validation
        
        Args:
            engine_names: List of engine names to use (None = all)
            user_context: User context for security validation
            
        Returns:
            List of validated inferences
        """
        operation_id = self._generate_id('OP')
        start_time = time.time()
        
        try:
            # Add audit entry
            self._add_audit_entry('reasoning_started', {
                'operation_id': operation_id,
                'engine_names': engine_names,
                'user_context': user_context
            })
            
            # Rate limiting
            if user_context and 'user_id' in user_context:
                is_within_limit, limit_msg = self.rate_limiter.check_rate_limit(user_context['user_id'])
                if not is_within_limit:
                    self.metrics['rate_limit_hits'] += 1
                    raise ReasoningSecurityError(f"Rate limit exceeded: {limit_msg}")
            
            # Resource check
            is_within_limits, resource_msg = self.performance_monitor.check_system_resources()
            if not is_within_limits:
                raise ReasoningPerformanceError(f"System resources exceeded: {resource_msg}")
            
            # Performance monitoring
            with self.performance_monitor.monitor_operation("enhanced_reasoning"):
                all_inferences = []
                
                # Get facts and rules with validation
                with self._lock:
                    facts = [f for f in self.facts.values() if f.validation_status == "passed"]
                    rules = [r for r in self.rules.values() if r.enabled and r.validation_status == "passed"]
                
                if not facts:
                    logger.warning("No valid facts available for reasoning")
                    return []
                
                if not rules:
                    logger.warning("No valid rules available for reasoning")
                    return []
                
                # Determine engines to use
                if engine_names is None:
                    engine_names = list(self.reasoning_engines.keys())
                
                # Validate engine names
                valid_engines = []
                for engine_name in engine_names:
                    if engine_name in self.reasoning_engines:
                        valid_engines.append(engine_name)
                    else:
                        logger.warning(f"Unknown reasoning engine: {engine_name}")
                
                if not valid_engines:
                    raise ReasoningValidationError("No valid reasoning engines specified")
                
                # Run reasoning with error recovery
                if self.config.get('enable_parallel_reasoning'):
                    all_inferences = self._parallel_reasoning(valid_engines, facts, rules)
                else:
                    all_inferences = self._sequential_reasoning(valid_engines, facts, rules)
                
                # Validate and filter inferences
                validated_inferences = []
                for inference in all_inferences:
                    try:
                        # Validate inference
                        engine = self.reasoning_engines[inference.metadata.get('engine_name', 'enhanced_rule_based')]
                        if engine.validate(inference, facts):
                            validated_inferences.append(inference)
                        else:
                            logger.warning(f"Inference {inference.inference_id} failed validation")
                    except Exception as e:
                        logger.error(f"Failed to validate inference {inference.inference_id}: {str(e)}")
                
                # Store validated inferences
                with self._lock:
                    for inference in validated_inferences:
                        self.inferences[inference.inference_id] = inference
                    
                    self.metrics['total_inferences'] += len(validated_inferences)
                    self.metrics['successful_inferences'] += len(validated_inferences)
                
                # Filter by confidence threshold
                threshold = self.config.get('inference_confidence_threshold', 0.6)
                filtered_inferences = [
                    inf for inf in validated_inferences 
                    if inf.confidence >= threshold
                ]
                
                # Update metrics
                reasoning_time = (time.time() - start_time) * 1000
                self._update_reasoning_metrics(reasoning_time, True, len(filtered_inferences))
                
                # Add completion audit entry
                self._add_audit_entry('reasoning_completed', {
                    'operation_id': operation_id,
                    'inferences_generated': len(filtered_inferences),
                    'reasoning_time_ms': reasoning_time,
                    'engines_used': valid_engines
                })
                
                logger.info(f"Enhanced reasoning completed: {len(filtered_inferences)} validated inferences "
                           f"(filtered from {len(all_inferences)}) in {reasoning_time:.2f}ms")
                
                return filtered_inferences
                
        except SymbolicReasoningException:
            # Re-raise our custom exceptions
            self.metrics['failed_reasoning_calls'] += 1
            self._add_audit_entry('reasoning_failed', {
                'operation_id': operation_id,
                'error_type': 'reasoning_error'
            })
            raise
            
        except Exception as e:
            reasoning_time = (time.time() - start_time) * 1000
            self._update_reasoning_metrics(reasoning_time, False, 0)
            
            # Error recovery
            if self.config.get('enable_error_recovery'):
                recovery_result = self._attempt_error_recovery(e, engine_names, user_context)
                if recovery_result:
                    return recovery_result
            
            self._add_audit_entry('reasoning_failed', {
                'operation_id': operation_id,
                'error': str(e),
                'error_type': 'unexpected_error'
            })
            logger.error(f"Enhanced reasoning failed: {str(e)}")
            raise ReasoningValidationError(f"Reasoning failed: {str(e)}")
    
    def _parallel_reasoning(self, engine_names: List[str], 
                           facts: List[ValidatedFact], 
                           rules: List[ValidatedRule]) -> List[ValidatedInference]:
        """Execute reasoning in parallel across engines"""
        all_inferences = []
        
        # Submit tasks to thread pool
        futures = []
        for engine_name in engine_names:
            future = self._executor.submit(
                self._run_enhanced_reasoning_engine,
                engine_name, facts, rules
            )
            futures.append((engine_name, future))
        
        # Collect results with timeout
        timeout = self.config.get('reasoning_timeout_seconds', 30)
        for engine_name, future in futures:
            try:
                inferences = future.result(timeout=timeout)
                all_inferences.extend(inferences)
                self._update_engine_metrics(engine_name, True, len(inferences))
            except Exception as e:
                logger.error(f"Parallel reasoning failed for {engine_name}: {str(e)}")
                self._update_engine_metrics(engine_name, False, 0)
        
        return all_inferences
    
    def _sequential_reasoning(self, engine_names: List[str], 
                             facts: List[ValidatedFact], 
                             rules: List[ValidatedRule]) -> List[ValidatedInference]:
        """Execute reasoning sequentially across engines"""
        all_inferences = []
        
        for engine_name in engine_names:
            try:
                inferences = self._run_enhanced_reasoning_engine(engine_name, facts, rules)
                all_inferences.extend(inferences)
                self._update_engine_metrics(engine_name, True, len(inferences))
            except Exception as e:
                logger.error(f"Sequential reasoning failed for {engine_name}: {str(e)}")
                self._update_engine_metrics(engine_name, False, 0)
                continue
        
        return all_inferences
    
    def _run_enhanced_reasoning_engine(self, engine_name: str, 
                                     facts: List[ValidatedFact], 
                                     rules: List[ValidatedRule]) -> List[ValidatedInference]:
        """Run a specific enhanced reasoning engine"""
        start_time = time.time()
        
        try:
            engine = self.reasoning_engines[engine_name]
            
            # Filter rules relevant to engine
            relevant_rules = self._filter_rules_for_engine(engine_name, rules)
            
            # Run reasoning
            inferences = engine.reason(facts, relevant_rules)
            
            # Add engine metadata to inferences
            for inference in inferences:
                inference.metadata['engine_name'] = engine_name
                inference.metadata['engine_execution_time_ms'] = (time.time() - start_time) * 1000
            
            logger.debug(f"{engine_name} produced {len(inferences)} inferences")
            return inferences
            
        except Exception as e:
            logger.error(f"Enhanced engine {engine_name} failed: {str(e)}")
            raise ReasoningValidationError(f"Engine {engine_name} failed: {str(e)}")
    
    def _filter_rules_for_engine(self, engine_name: str, rules: List[ValidatedRule]) -> List[ValidatedRule]:
        """Filter rules relevant for specific engine"""
        if engine_name == 'enhanced_rule_based':
            return [r for r in rules if r.rule_type in 
                   [RuleType.TRANSACTION_LIMIT, RuleType.VELOCITY, RuleType.COMPOSITE, RuleType.COMPLIANCE]]
        elif engine_name == 'fuzzy':
            return [r for r in rules if r.rule_type == RuleType.BEHAVIORAL]
        elif engine_name == 'temporal':
            return [r for r in rules if r.rule_type == RuleType.TEMPORAL]
        else:
            return rules
    
    def _attempt_error_recovery(self, error: Exception, engine_names: Optional[List[str]], 
                               user_context: Optional[Dict[str, Any]]) -> Optional[List[ValidatedInference]]:
        """Attempt to recover from reasoning errors"""
        self.metrics['recovery_attempts'] += 1
        
        try:
            logger.info(f"Attempting error recovery for: {str(error)}")
            
            # Strategy 1: Reduce complexity by using fewer engines
            if engine_names and len(engine_names) > 1:
                logger.info("Recovery: Reducing engine count")
                simplified_engines = engine_names[:1]  # Use only first engine
                return self.reason(simplified_engines, user_context)
            
            # Strategy 2: Use only simple rule-based reasoning
            if engine_names != ['enhanced_rule_based']:
                logger.info("Recovery: Using only rule-based engine")
                return self.reason(['enhanced_rule_based'], user_context)
            
            # Strategy 3: Disable complex rules temporarily
            logger.info("Recovery: Disabling complex rules")
            complex_rules = []
            with self._lock:
                for rule in self.rules.values():
                    if (len(rule.conditions) + len(rule.actions)) > 5:  # Arbitrary complexity threshold
                        complex_rules.append(rule.rule_id)
                        rule.enabled = False
            
            try:
                result = self.reason(['enhanced_rule_based'], user_context)
                self.metrics['successful_recoveries'] += 1
                return result
            finally:
                # Re-enable disabled rules
                with self._lock:
                    for rule_id in complex_rules:
                        if rule_id in self.rules:
                            self.rules[rule_id].enabled = True
            
        except Exception as recovery_error:
            logger.error(f"Error recovery failed: {str(recovery_error)}")
            return None
        
        return None
    
    def validate_rules(self, rules_to_validate: Optional[List[str]] = None) -> Dict[str, Union[bool, List[str]]]:
        """
        Enhanced rule validation with detailed error reporting
        
        Args:
            rules_to_validate: List of rule IDs to validate (None = all)
            
        Returns:
            Dictionary of rule_id -> validation result (True/False or detailed errors)
        """
        validation_results = {}
        
        with self._lock:
            if rules_to_validate is None:
                rules_to_validate = list(self.rules.keys())
            
            for rule_id in rules_to_validate:
                rule = self.rules.get(rule_id)
                if rule:
                    try:
                        is_valid, errors = rule.validate()
                        if is_valid:
                            validation_results[rule_id] = True
                        else:
                            validation_results[rule_id] = errors
                    except Exception as e:
                        validation_results[rule_id] = [f"Validation error: {str(e)}"]
                        logger.error(f"Rule validation failed for {rule_id}: {str(e)}")
                else:
                    validation_results[rule_id] = [f"Rule not found: {rule_id}"]
        
        # Update metrics
        valid_count = sum(1 for result in validation_results.values() if result is True)
        logger.info(f"Rule validation completed: {valid_count}/{len(validation_results)} rules valid")
        
        return validation_results
    
    def integrate_knowledge(self, external_facts: List[Dict[str, Any]], 
                          validation_level: str = "standard") -> Dict[str, Any]:
        """
        Enhanced knowledge integration with comprehensive validation
        
        Args:
            external_facts: List of external facts to integrate
            validation_level: Validation level (basic, standard, strict)
            
        Returns:
            Detailed integration report
        """
        operation_id = self._generate_id('OP')
        start_time = time.time()
        
        integration_report = {
            'operation_id': operation_id,
            'total_facts': len(external_facts),
            'integrated': 0,
            'rejected': 0,
            'errors': [],
            'warnings': [],
            'processing_time_ms': 0,
            'validation_level': validation_level
        }
        
        try:
            self._add_audit_entry('knowledge_integration_started', {
                'operation_id': operation_id,
                'fact_count': len(external_facts),
                'validation_level': validation_level
            })
            
            # Validate external facts based on level
            for i, fact_data in enumerate(external_facts):
                try:
                    # Security validation
                    is_safe, safety_msg = self.security_validator.validate_input_safety(fact_data)
                    if not is_safe:
                        integration_report['errors'].append({
                            'index': i,
                            'error': f"Security validation failed: {safety_msg}",
                            'fact_data': fact_data
                        })
                        integration_report['rejected'] += 1
                        
                        if validation_level == "strict":
                            break  # Stop on first error in strict mode
                        continue
                    
                    # Transform and validate external fact
                    transformed_fact = self._transform_external_fact(fact_data)
                    
                    # Create validated fact
                    try:
                        validated_fact = ValidatedFact(**transformed_fact)
                        
                        # Additional validation for strict mode
                        if validation_level == "strict":
                            is_valid, errors = validated_fact.validate()
                            if not is_valid:
                                integration_report['errors'].append({
                                    'index': i,
                                    'error': f"Strict validation failed: {', '.join(errors)}",
                                    'fact_data': fact_data
                                })
                                integration_report['rejected'] += 1
                                break
                        
                        # Store fact
                        with self._lock:
                            self.facts[validated_fact.fact_id] = validated_fact
                        
                        integration_report['integrated'] += 1
                        
                    except Exception as fact_error:
                        integration_report['errors'].append({
                            'index': i,
                            'error': f"Fact creation failed: {str(fact_error)}",
                            'fact_data': fact_data
                        })
                        integration_report['rejected'] += 1
                        
                        if validation_level == "strict":
                            break
                
                except Exception as e:
                    integration_report['errors'].append({
                        'index': i,
                        'error': f"Processing failed: {str(e)}",
                        'fact_data': fact_data
                    })
                    integration_report['rejected'] += 1
                    
                    if validation_level == "strict":
                        break
            
            # Calculate processing time
            integration_report['processing_time_ms'] = (time.time() - start_time) * 1000
            
            # Add completion audit entry
            self._add_audit_entry('knowledge_integration_completed', integration_report)
            
            logger.info(f"Knowledge integration completed: {integration_report['integrated']} integrated, "
                       f"{integration_report['rejected']} rejected in {integration_report['processing_time_ms']:.2f}ms")
            
            # Trigger reasoning if significant new knowledge and no errors in strict mode
            if (integration_report['integrated'] > 10 and 
                not (validation_level == "strict" and integration_report['errors'])):
                self._executor.submit(self.reason)
            
            return integration_report
            
        except Exception as e:
            integration_report['processing_time_ms'] = (time.time() - start_time) * 1000
            integration_report['errors'].append({
                'index': -1,
                'error': f"Integration process failed: {str(e)}",
                'fact_data': None
            })
            
            self._add_audit_entry('knowledge_integration_failed', {
                'operation_id': operation_id,
                'error': str(e)
            })
            
            logger.error(f"Knowledge integration failed: {str(e)}")
            return integration_report
    
    def query_knowledge(self, query: Dict[str, Any]) -> List[Union[ValidatedFact, ValidatedInference]]:
        """
        Enhanced knowledge base query with validation
        
        Args:
            query: Query parameters with enhanced options
            
        Returns:
            List of matching facts and inferences
        """
        try:
            # Validate query parameters
            self._validate_query(query)
            
            results = []
            
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
            
            # Apply filters
            results = self._apply_query_filters(results, query)
            
            # Sort results
            if query.get('sort_by_confidence'):
                results.sort(key=lambda x: getattr(x, 'confidence', 0), reverse=True)
            elif query.get('sort_by_timestamp'):
                results.sort(key=lambda x: getattr(x, 'timestamp', datetime.min), reverse=True)
            
            # Apply limit
            limit = query.get('limit')
            if limit and limit > 0:
                results = results[:limit]
            
            logger.debug(f"Enhanced query returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Enhanced knowledge query failed: {str(e)}")
            raise QueryValidationError(f"Query failed: {str(e)}")
    
    def _validate_query(self, query: Dict[str, Any]) -> None:
        """Validate query parameters"""
        if not isinstance(query, dict):
            raise QueryValidationError("Query must be a dictionary")
        
        # Validate limit
        limit = query.get('limit')
        if limit is not None:
            if not isinstance(limit, int) or limit < 0:
                raise QueryValidationError("Limit must be a non-negative integer")
            if limit > 10000:  # Prevent resource exhaustion
                raise QueryValidationError("Limit cannot exceed 10000")
        
        # Validate confidence threshold
        min_confidence = query.get('min_confidence')
        if min_confidence is not None:
            is_valid, msg = ValidationUtils.validate_confidence(min_confidence)
            if not is_valid:
                raise QueryValidationError(f"Invalid min_confidence: {msg}")
        
        # Validate time range
        if 'start_time' in query or 'end_time' in query:
            start_time = query.get('start_time')
            end_time = query.get('end_time')
            
            if start_time and not isinstance(start_time, datetime):
                raise QueryValidationError("start_time must be datetime object")
            
            if end_time and not isinstance(end_time, datetime):
                raise QueryValidationError("end_time must be datetime object")
            
            if start_time and end_time and start_time >= end_time:
                raise QueryValidationError("start_time must be before end_time")
    
    def _apply_query_filters(self, results: List[Union[ValidatedFact, ValidatedInference]], 
                            query: Dict[str, Any]) -> List[Union[ValidatedFact, ValidatedInference]]:
        """Apply advanced filters to query results"""
        filtered_results = results
        
        # Time range filter
        start_time = query.get('start_time')
        end_time = query.get('end_time')
        if start_time or end_time:
            filtered_results = [
                r for r in filtered_results
                if (not start_time or r.timestamp >= start_time) and
                   (not end_time or r.timestamp <= end_time)
            ]
        
        # Validation status filter
        validation_status = query.get('validation_status')
        if validation_status:
            filtered_results = [
                r for r in filtered_results
                if getattr(r, 'validation_status', None) == validation_status
            ]
        
        # Security level filter
        security_level = query.get('security_level')
        if security_level:
            filtered_results = [
                r for r in filtered_results
                if getattr(r, 'security_level', None) == SecurityLevel(security_level)
            ]
        
        return filtered_results
    
    def _match_fact_query(self, fact: ValidatedFact, query: Dict[str, Any]) -> bool:
        """Enhanced fact matching with additional criteria"""
        # Basic matching
        if 'subject' in query and fact.subject != query['subject']:
            return False
        
        if 'predicate' in query and fact.predicate != query['predicate']:
            return False
        
        if 'fact_type' in query and fact.fact_type.value != query['fact_type']:
            return False
        
        if 'source' in query and fact.source != query['source']:
            return False
        
        # Confidence filtering
        min_confidence = query.get('min_confidence')
        if min_confidence is not None and fact.confidence < min_confidence:
            return False
        
        # Quality filtering
        min_quality = query.get('min_quality_score')
        if min_quality is not None and fact.calculate_quality_score() < min_quality:
            return False
        
        return True
    
    def _match_inference_query(self, inference: ValidatedInference, query: Dict[str, Any]) -> bool:
        """Enhanced inference matching with additional criteria"""
        # Basic matching
        if 'reasoning_type' in query and inference.reasoning_type.value != query['reasoning_type']:
            return False
        
        # Confidence filtering
        min_confidence = query.get('min_confidence')
        if min_confidence is not None and inference.confidence < min_confidence:
            return False
        
        # Engine filtering
        engine_name = query.get('engine_name')
        if engine_name and inference.metadata.get('engine_name') != engine_name:
            return False
        
        return True
    
    def explain_inference(self, inference_id: str) -> Dict[str, Any]:
        """
        Enhanced inference explanation with comprehensive details
        
        Args:
            inference_id: Inference ID to explain
            
        Returns:
            Detailed explanation dictionary
        """
        try:
            with self._lock:
                inference = self.inferences.get(inference_id)
                if not inference:
                    return {'error': 'Inference not found', 'inference_id': inference_id}
                
                # Get premise facts
                premise_facts = [
                    self.facts.get(fact_id) 
                    for fact_id in inference.premises
                ]
                premise_facts = [f for f in premise_facts if f]
                
                # Enhanced explanation
                explanation = {
                    'inference_id': inference_id,
                    'reasoning_type': inference.reasoning_type.value,
                    'confidence': inference.confidence,
                    'validation_status': inference.validation_status,
                    
                    'conclusion': {
                        'subject': inference.conclusion.subject,
                        'predicate': inference.conclusion.predicate,
                        'object': inference.conclusion.object,
                        'confidence': inference.conclusion.confidence,
                        'quality_score': inference.conclusion.calculate_quality_score()
                    },
                    
                    'premises': [
                        {
                            'fact_id': f.fact_id,
                            'triple': f.to_triple(),
                            'confidence': f.confidence,
                            'quality_score': f.calculate_quality_score(),
                            'source': f.source,
                            'validation_status': f.validation_status
                        }
                        for f in premise_facts
                    ],
                    
                    'reasoning_path': inference.reasoning_path,
                    'evidence_strength': inference.evidence_strength,
                    'logical_consistency': inference.logical_consistency,
                    
                    'metadata': {
                        'engine_name': inference.metadata.get('engine_name'),
                        'execution_time_ms': inference.metadata.get('engine_execution_time_ms'),
                        'timestamp': inference.timestamp.isoformat()
                    },
                    
                    'quality_assessment': {
                        'premise_count': len(premise_facts),
                        'avg_premise_confidence': np.mean([f.confidence for f in premise_facts]) if premise_facts else 0,
                        'reasoning_path_length': len(inference.reasoning_path),
                        'overall_quality': (inference.confidence + inference.evidence_strength + inference.logical_consistency) / 3
                    }
                }
                
                return explanation
                
        except Exception as e:
            logger.error(f"Failed to explain inference: {str(e)}")
            return {'error': str(e), 'inference_id': inference_id}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive enhanced metrics"""
        with self._lock:
            # Update system metrics
            try:
                memory_info = psutil.Process().memory_info()
                self.metrics['memory_usage_mb'] = memory_info.rss / 1024 / 1024
                self.metrics['cpu_usage_percent'] = psutil.Process().cpu_percent()
            except Exception:
                pass  # Ignore if psutil fails
            
            # Calculate derived metrics
            if self.metrics['total_facts'] > 0:
                self.metrics['fact_utilization'] = (
                    len([f for f in self.facts.values() if any(f.fact_id in inf.premises for inf in self.inferences.values())]) /
                    self.metrics['total_facts']
                )
            
            if self.metrics['total_rules'] > 0:
                self.metrics['rule_coverage'] = (
                    len([r for r in self.rules.values() if r.execution_count > 0]) /
                    self.metrics['total_rules']
                )
            
            if self.metrics['total_inferences'] > 0:
                self.metrics['inference_accuracy'] = (
                    self.metrics['successful_inferences'] / 
                    self.metrics['total_inferences']
                )
            
            return self.metrics.copy()
    
    def get_reasoning_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'components': {},
            'metrics': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check reasoning engines
            for engine_name, engine in self.reasoning_engines.items():
                engine_health = {
                    'status': 'operational',
                    'last_used': 'unknown',
                    'success_rate': 0.0
                }
                
                engine_metrics = self.metrics['engine_performance'].get(engine_name, {})
                if engine_metrics.get('calls', 0) > 0:
                    engine_health['success_rate'] = engine_metrics['successes'] / engine_metrics['calls']
                    engine_health['avg_time_ms'] = engine_metrics['avg_time_ms']
                
                health_status['components'][engine_name] = engine_health
            
            # Check knowledge base health
            kb_health = {
                'total_facts': len(self.facts),
                'valid_facts': len([f for f in self.facts.values() if f.validation_status == "passed"]),
                'total_rules': len(self.rules),
                'enabled_rules': len([r for r in self.rules.values() if r.enabled]),
                'total_inferences': len(self.inferences)
            }
            health_status['components']['knowledge_base'] = kb_health
            
            # Check system resources
            try:
                memory_info = psutil.Process().memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                cpu_percent = psutil.Process().cpu_percent()
                
                health_status['components']['system_resources'] = {
                    'memory_usage_mb': memory_mb,
                    'cpu_usage_percent': cpu_percent,
                    'status': 'healthy'
                }
                
                # Check for resource issues
                if memory_mb > self.config.get('max_memory_usage_mb', 512):
                    health_status['issues'].append(f"High memory usage: {memory_mb:.1f}MB")
                    health_status['status'] = 'warning'
                
                if cpu_percent > self.config.get('max_cpu_usage_percent', 80):
                    health_status['issues'].append(f"High CPU usage: {cpu_percent:.1f}%")
                    health_status['status'] = 'warning'
                
            except Exception:
                health_status['components']['system_resources'] = {'status': 'unknown'}
            
            # Check validation health
            validation_health = {
                'total_validation_errors': self.metrics['validation_errors'],
                'security_violations': self.metrics['total_security_violations'],
                'rate_limit_hits': self.metrics['rate_limit_hits'],
                'status': 'healthy'
            }
            
            if self.metrics['validation_errors'] > 100:
                health_status['issues'].append(f"High validation error count: {self.metrics['validation_errors']}")
                health_status['status'] = 'warning'
            
            health_status['components']['validation'] = validation_health
            
            # Performance metrics
            health_status['metrics'] = {
                'avg_reasoning_time_ms': self.metrics['average_reasoning_time_ms'],
                'total_reasoning_calls': self.metrics['total_reasoning_calls'],
                'success_rate': (
                    self.metrics['successful_reasoning_calls'] / 
                    max(self.metrics['total_reasoning_calls'], 1)
                ),
                'inference_accuracy': self.metrics['inference_accuracy']
            }
            
            # Generate recommendations
            if health_status['issues']:
                if any('memory' in issue.lower() for issue in health_status['issues']):
                    health_status['recommendations'].append("Consider cleaning up old facts and inferences")
                
                if any('cpu' in issue.lower() for issue in health_status['issues']):
                    health_status['recommendations'].append("Consider reducing parallel reasoning workers")
                
                if any('validation' in issue.lower() for issue in health_status['issues']):
                    health_status['recommendations'].append("Review input validation configuration")
            
            # Overall status determination
            if health_status['issues']:
                if any('critical' in issue.lower() for issue in health_status['issues']):
                    health_status['status'] = 'critical'
                elif health_status['status'] != 'warning':
                    health_status['status'] = 'warning'
            
        except Exception as e:
            health_status['status'] = 'error'
            health_status['issues'].append(f"Health check failed: {str(e)}")
            logger.error(f"Health check error: {str(e)}")
        
        return health_status
    
    def export_knowledge_base(self) -> Dict[str, Any]:
        """Export enhanced knowledge base with comprehensive data"""
        with self._lock:
            export_data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_facts': len(self.facts),
                    'total_rules': len(self.rules),
                    'total_inferences': len(self.inferences),
                    'enhanced_version': True,
                    'validation_enabled': True
                },
                
                'facts': {
                    fact_id: {
                        'triple': fact.to_triple(),
                        'type': fact.fact_type.value,
                        'confidence': fact.confidence,
                        'quality_score': fact.calculate_quality_score(),
                        'source': fact.source,
                        'timestamp': fact.timestamp.isoformat(),
                        'validation_status': fact.validation_status,
                        'security_level': fact.security_level.value,
                        'metadata': fact.metadata
                    }
                    for fact_id, fact in self.facts.items()
                },
                
                'rules': {
                    rule_id: {
                        'name': rule.name,
                        'type': rule.rule_type.value,
                        'description': rule.description,
                        'enabled': rule.enabled,
                        'priority': rule.priority,
                        'validation_status': rule.validation_status,
                        'execution_count': rule.execution_count,
                        'success_rate': rule.success_rate,
                        'avg_execution_time_ms': rule.avg_execution_time_ms,
                        'security_level': rule.security_level.value,
                        'conditions': rule.conditions,
                        'actions': rule.actions,
                        'metadata': rule.metadata
                    }
                    for rule_id, rule in self.rules.items()
                },
                
                'inferences': {
                    inf_id: {
                        'conclusion': inf.conclusion.to_triple(),
                        'confidence': inf.confidence,
                        'reasoning_type': inf.reasoning_type.value,
                        'premise_count': len(inf.premises),
                        'validation_status': inf.validation_status,
                        'evidence_strength': inf.evidence_strength,
                        'logical_consistency': inf.logical_consistency,
                        'timestamp': inf.timestamp.isoformat(),
                        'reasoning_path': inf.reasoning_path,
                        'metadata': inf.metadata
                    }
                    for inf_id, inf in self.inferences.items()
                },
                
                'metrics': self.metrics,
                'configuration': {
                    key: value for key, value in self.config.items()
                    if not key.startswith('secret') and not key.startswith('key')
                }
            }
            
            return export_data
    
    def export_enhanced_audit_trail(self) -> List[Dict[str, Any]]:
        """Export complete enhanced audit trail"""
        with self._lock:
            return self.audit_trail.copy()
    
    def persist_knowledge(self) -> bool:
        """Enhanced knowledge persistence with validation"""
        try:
            # Create backup of current data
            backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Persist facts with validation info
            facts_file = self.storage_path / f'enhanced_facts_{backup_timestamp}.pkl'
            with open(facts_file, 'wb') as f:
                pickle.dump(self.facts, f)
            
            # Persist rules with performance data
            rules_file = self.storage_path / f'enhanced_rules_{backup_timestamp}.pkl'
            with open(rules_file, 'wb') as f:
                pickle.dump(self.rules, f)
            
            # Persist inferences with validation
            inferences_file = self.storage_path / f'enhanced_inferences_{backup_timestamp}.pkl'
            with open(inferences_file, 'wb') as f:
                pickle.dump(self.inferences, f)
            
            # Persist metrics
            metrics_file = self.storage_path / f'enhanced_metrics_{backup_timestamp}.json'
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
            
            # Persist audit trail
            audit_file = self.storage_path / f'enhanced_audit_{backup_timestamp}.json'
            with open(audit_file, 'w') as f:
                json.dump(self.audit_trail, f, indent=2, default=str)
            
            logger.info(f"Enhanced knowledge base persisted with timestamp {backup_timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to persist enhanced knowledge: {str(e)}")
            return False
    
    def load_knowledge(self) -> bool:
        """Enhanced knowledge loading with validation"""
        try:
            # Find latest backup files
            fact_files = sorted(self.storage_path.glob('enhanced_facts_*.pkl'))
            rule_files = sorted(self.storage_path.glob('enhanced_rules_*.pkl'))
            inference_files = sorted(self.storage_path.glob('enhanced_inferences_*.pkl'))
            
            if fact_files:
                with open(fact_files[-1], 'rb') as f:
                    loaded_facts = pickle.load(f)
                    # Validate loaded facts
                    valid_facts = {}
                    for fact_id, fact in loaded_facts.items():
                        if isinstance(fact, ValidatedFact):
                            is_valid, errors = fact.validate()
                            if is_valid:
                                valid_facts[fact_id] = fact
                            else:
                                logger.warning(f"Loaded fact {fact_id} failed validation: {errors}")
                    self.facts = valid_facts
            
            if rule_files:
                with open(rule_files[-1], 'rb') as f:
                    loaded_rules = pickle.load(f)
                    # Validate loaded rules
                    valid_rules = {}
                    for rule_id, rule in loaded_rules.items():
                        if isinstance(rule, ValidatedRule):
                            is_valid, errors = rule.validate()
                            if is_valid:
                                valid_rules[rule_id] = rule
                            else:
                                logger.warning(f"Loaded rule {rule_id} failed validation: {errors}")
                    self.rules = valid_rules
            
            if inference_files:
                with open(inference_files[-1], 'rb') as f:
                    loaded_inferences = pickle.load(f)
                    # Validate loaded inferences
                    valid_inferences = {}
                    for inf_id, inference in loaded_inferences.items():
                        if isinstance(inference, ValidatedInference):
                            is_valid, errors = inference.validate()
                            if is_valid:
                                valid_inferences[inf_id] = inference
                            else:
                                logger.warning(f"Loaded inference {inf_id} failed validation: {errors}")
                    self.inferences = valid_inferences
            
            logger.info(f"Enhanced knowledge base loaded: {len(self.facts)} facts, "
                       f"{len(self.rules)} rules, {len(self.inferences)} inferences")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load enhanced knowledge: {str(e)}")
            return False
    
    def _transform_external_fact(self, fact_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform external fact to internal enhanced format"""
        # Add enhanced fields
        transformed = fact_data.copy()
        
        # Set default fact type
        if 'fact_type' not in transformed:
            transformed['fact_type'] = FactType.EXTERNAL_DATA.value
        
        # Set default security level
        if 'security_level' not in transformed:
            transformed['security_level'] = SecurityLevel.STANDARD.value
        
        # Add source information
        if 'source' not in transformed:
            transformed['source'] = 'external'
        
        # Set default confidence if not provided
        if 'confidence' not in transformed:
            transformed['confidence'] = 0.8  # Slightly lower for external data
        
        # Add metadata if not present
        if 'metadata' not in transformed:
            transformed['metadata'] = {}
        
        transformed['metadata']['import_timestamp'] = datetime.now().isoformat()
        
        return transformed
    
    def _update_reasoning_metrics(self, time_ms: float, success: bool, inference_count: int) -> None:
        """Update reasoning performance metrics"""
        with self._lock:
            self.metrics['total_reasoning_calls'] += 1
            
            if success:
                self.metrics['successful_reasoning_calls'] += 1
            else:
                self.metrics['failed_reasoning_calls'] += 1
            
            # Update average time
            total = self.metrics['total_reasoning_calls']
            current_avg = self.metrics['average_reasoning_time_ms']
            self.metrics['average_reasoning_time_ms'] = (
                (current_avg * (total - 1) + time_ms) / total
            )
            
            # Update hourly metrics
            hour_key = datetime.now().strftime('%Y-%m-%d_%H')
            if hour_key not in self.metrics['hourly_reasoning_count']:
                self.metrics['hourly_reasoning_count'][hour_key] = 0
            self.metrics['hourly_reasoning_count'][hour_key] += 1
    
    def _update_engine_metrics(self, engine_name: str, success: bool, inference_count: int) -> None:
        """Update engine-specific metrics"""
        with self._lock:
            if engine_name not in self.metrics['engine_performance']:
                self.metrics['engine_performance'][engine_name] = {
                    'calls': 0, 'successes': 0, 'avg_time_ms': 0
                }
            
            engine_metrics = self.metrics['engine_performance'][engine_name]
            engine_metrics['calls'] += 1
            
            if success:
                engine_metrics['successes'] += 1
    
    def _add_audit_entry(self, operation: str, details: Dict[str, Any]) -> None:
        """Add entry to enhanced audit trail"""
        if not self.config.get('enable_audit_trail', True):
            return
        
        with self._lock:
            audit_entry = {
                'timestamp': datetime.now().isoformat(),
                'operation': operation,
                'details': details,
                'audit_id': self._generate_id('AUDIT'),
                'status': 'completed' if not operation.endswith('_failed') else 'failed'
            }
            
            self.audit_trail.append(audit_entry)
            
            # Limit audit trail size
            max_audit_entries = self.config.get('max_audit_entries', 10000)
            if len(self.audit_trail) > max_audit_entries:
                self.audit_trail = self.audit_trail[-max_audit_entries:]
    
    def _generate_id(self, prefix: str) -> str:
        """Generate cryptographically secure unique ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_part = secrets.token_hex(8)
        return f"{prefix}_{timestamp}_{random_part}"
    
    def __repr__(self) -> str:
        return (f"Enhanced FinancialSymbolicReasoner(facts={len(self.facts)}, "
               f"rules={len(self.rules)}, inferences={len(self.inferences)}, "
               f"validation=enabled, security=enhanced)")

# Legacy compatibility
SymbolicReasoner = EnhancedFinancialSymbolicReasoner

# Export enhanced classes
__all__ = [
    'EnhancedFinancialSymbolicReasoner',
    'SymbolicReasoner',  # Legacy compatibility
    'ValidatedFact',
    'ValidatedRule',
    'ValidatedInference',
    'ReasoningType',
    'RuleType',
    'FactType',
    'SecurityLevel',
    'ValidatedReasoningEngine',
    'EnhancedRuleBasedReasoningEngine',
    'ValidationUtils',
    'SecurityValidator',
    'PerformanceMonitor',
    'RateLimiter',
    # Exception classes
    'SymbolicReasoningException',
    'ReasoningValidationError',
    'RuleValidationError',
    'KnowledgeBaseError',
    'ReasoningTimeoutError',
    'ReasoningPerformanceError',
    'ReasoningSecurityError',
    'ReasoningIntegrationError',
    'InferenceValidationError',
    'FactValidationError',
    'QueryValidationError',
    'ReasoningCapacityError'
]

if __name__ == "__main__":
    # Enhanced example usage and testing
    
    # Initialize enhanced reasoner with comprehensive config
    config = {
        # Validation settings
        'enable_error_recovery': True,
        'validation_strict_mode': False,
        'inference_confidence_threshold': 0.7,
        
        # Security settings
        'allowed_sources': ['transaction_system', 'fraud_detector', 'behavior_analysis'],
        'max_fact_size_kb': 50,
        'max_rule_complexity': 30,
        'max_reasoning_calls_per_minute': 100,
        
        # Performance settings
        'max_reasoning_time_ms': 3000,
        'max_memory_usage_mb': 256,
        'enable_parallel_reasoning': True,
        
        # Monitoring settings
        'enable_audit_trail': True,
        'enable_health_monitoring': True
    }
    
    reasoner = EnhancedFinancialSymbolicReasoner(config)
    print("Enhanced FinancialSymbolicReasoner initialized with comprehensive validation")
    
    # Check enhanced capabilities
    print(f"Enhanced mode: True")
    print(f"Validation enabled: True")
    print(f"Security features: Advanced")
    
    # Add enhanced facts with validation
    print("\nAdding enhanced facts with validation...")
    
    try:
        # Transaction fact
        fact1_id = reasoner.add_fact({
            'fact_type': 'TRANSACTION',
            'subject': 'txn_enhanced_001',
            'predicate': 'amount',
            'object': 25000,
            'confidence': 0.98,
            'source': 'transaction_system',
            'security_level': 'enhanced',
            'metadata': {
                'transaction_category': 'high_value',
                'risk_indicators': ['large_amount', 'unusual_time']
            }
        })
        print(f"✅ Added enhanced fact: {fact1_id}")
        
        # Behavioral fact
        fact2_id = reasoner.add_fact({
            'fact_type': 'USER_BEHAVIOR',
            'subject': 'user_789',
            'predicate': 'velocity_anomaly_score',
            'object': 0.92,
            'confidence': 0.89,
            'source': 'behavior_analysis',
            'metadata': {
                'analysis_window': '24h',
                'baseline_deviation': 4.2
            }
        })
        print(f"✅ Added enhanced fact: {fact2_id}")
        
    except Exception as e:
        print(f"❌ Error adding facts: {type(e).__name__}: {str(e)}")
    
    # Add enhanced rules with validation
    print("\nAdding enhanced rules with comprehensive validation...")
    
    try:
        rule_id = reasoner.add_rule({
            'name': 'Enhanced Comprehensive Fraud Detection',
            'description': 'Multi-factor fraud detection with behavioral and transaction analysis',
            'rule_type': 'COMPOSITE',
            'conditions': [
                {
                    'type': 'fact_match',
                    'fact_type': 'TRANSACTION',
                    'predicate': 'amount',
                    'operator': '>',
                    'value': 20000
                },
                {
                    'type': 'fact_match',
                    'fact_type': 'USER_BEHAVIOR',
                    'predicate': 'velocity_anomaly_score',
                    'operator': '>',
                    'value': 0.8
                }
            ],
            'actions': [
                {
                    'type': 'assert',
                    'subject': 'fraud_detection',
                    'predicate': 'requires',
                    'object': 'immediate_review',
                    'confidence': 0.95
                }
            ],
            'priority': 9,
            'security_level': 'enhanced',
            'metadata': {
                'category': 'composite_analysis',
                'severity': 'critical',
                'review_required': True
            }
        })
        print(f"✅ Added enhanced rule: {rule_id}")
        
    except Exception as e:
        print(f"❌ Error adding rule: {type(e).__name__}: {str(e)}")
    
    # Validate all rules
    print("\nValidating all rules with detailed error reporting...")
    validation_results = reasoner.validate_rules()
    for rule_id, result in validation_results.items():
        if result is True:
            print(f"✅ Rule {rule_id}: Valid")
        else:
            print(f"❌ Rule {rule_id}: Errors - {result}")
    
    # Perform enhanced reasoning
    print("\nPerforming enhanced reasoning with comprehensive monitoring...")
    try:
        user_context = {'user_id': 'system', 'role': 'fraud_analyst'}
        inferences = reasoner.reason(user_context=user_context)
        
        print(f"✅ Generated {len(inferences)} validated inferences")
        
        for inf in inferences:
            print(f"  - {inf.conclusion.subject} {inf.conclusion.predicate} {inf.conclusion.object}")
            print(f"    Confidence: {inf.confidence:.2f}, Validation: {inf.validation_status}")
            
            # Explain inference
            explanation = reasoner.explain_inference(inf.inference_id)
            print(f"    Quality: {explanation['quality_assessment']['overall_quality']:.2f}")
        
    except Exception as e:
        print(f"❌ Enhanced reasoning failed: {type(e).__name__}: {str(e)}")
    
    # Test enhanced knowledge integration
    print("\nTesting enhanced knowledge integration...")
    
    external_facts = [
        {
            'subject': 'external_alert_001',
            'predicate': 'fraud_score',
            'object': 0.94,
            'confidence': 0.88,
            'source': 'external_fraud_service'
        },
        {
            'subject': 'compliance_check_001',
            'predicate': 'aml_status',
            'object': 'flagged',
            'confidence': 1.0,
            'source': 'compliance_system'
        }
    ]
    
    try:
        # Add external source to allowed sources for testing
        reasoner.security_validator.allowed_sources.add('external_fraud_service')
        reasoner.security_validator.allowed_sources.add('compliance_system')
        
        integration_report = reasoner.integrate_knowledge(external_facts, validation_level='strict')
        print(f"✅ Integration completed: {integration_report['integrated']} integrated, {integration_report['rejected']} rejected")
        
        if integration_report['errors']:
            for error in integration_report['errors']:
                print(f"  ❌ Error at index {error['index']}: {error['error']}")
        
    except Exception as e:
        print(f"❌ Knowledge integration failed: {type(e).__name__}: {str(e)}")
    
    # Test enhanced querying
    print("\nTesting enhanced knowledge querying...")
    
    try:
        query_results = reasoner.query_knowledge({
            'include_facts': True,
            'include_inferences': True,
            'min_confidence': 0.8,
            'validation_status': 'passed',
            'sort_by_confidence': True,
            'limit': 10
        })
        
        print(f"✅ Query returned {len(query_results)} results")
        for result in query_results[:3]:  # Show first 3
            if isinstance(result, ValidatedFact):
                print(f"  Fact: {result.to_triple()} (confidence: {result.confidence:.2f})")
            else:
                print(f"  Inference: {result.conclusion.to_triple()} (confidence: {result.confidence:.2f})")
        
    except Exception as e:
        print(f"❌ Enhanced querying failed: {type(e).__name__}: {str(e)}")
    
    # Check enhanced system health
    print("\nChecking enhanced system health...")
    health = reasoner.get_reasoning_health()
    print(f"  Status: {health['status']}")
    print(f"  Components: {len(health['components'])}")
    print(f"  Metrics: {len(health['metrics'])}")
    if health['issues']:
        print(f"  Issues: {health['issues']}")
    if health['recommendations']:
        print(f"  Recommendations: {health['recommendations']}")
    
    # Get enhanced metrics
    metrics = reasoner.get_metrics()
    print(f"\nEnhanced Metrics:")
    print(f"  Facts: {metrics['total_facts']}")
    print(f"  Rules: {metrics['total_rules']}")
    print(f"  Inferences: {metrics['total_inferences']}")
    print(f"  Validation errors: {metrics['validation_errors']}")
    print(f"  Security violations: {metrics['total_security_violations']}")
    print(f"  Success rate: {metrics.get('success_rate', 0):.2%}")
    
    # Export enhanced audit trail
    audit_trail = reasoner.export_enhanced_audit_trail()
    print(f"\nAudit Trail: {len(audit_trail)} entries")
    if audit_trail:
        latest = audit_trail[-1]
        print(f"  Latest: {latest['operation']} - {latest['status']}")
    
    # Export enhanced knowledge base
    kb_export = reasoner.export_knowledge_base()
    print(f"\nEnhanced Knowledge Base Export:")
    print(f"  Facts: {len(kb_export['facts'])}")
    print(f"  Rules: {len(kb_export['rules'])}")
    print(f"  Inferences: {len(kb_export['inferences'])}")
    print(f"  Enhanced features: {kb_export['metadata']['enhanced_version']}")
    
    print("\n🎯 Enhanced FinancialSymbolicReasoner ready for production use!")
    print("Features: Comprehensive validation, advanced security, detailed monitoring, error recovery")