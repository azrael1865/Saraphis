"""
Domain Router for Universal AI Core Brain.
Intelligent routing system that routes requests to appropriate domains based on input characteristics.
"""

import logging
import threading
import re
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
from collections import defaultdict, deque


class RoutingStrategy(Enum):
    """Available routing strategies."""
    PATTERN_MATCHING = "pattern_matching"
    HEURISTIC_BASED = "heuristic_based"
    ML_BASED = "ml_based"
    HYBRID = "hybrid"
    CONFIDENCE_WEIGHTED = "confidence_weighted"


class RoutingConfidence(Enum):
    """Confidence levels for routing decisions."""
    VERY_HIGH = "very_high"    # 0.9+
    HIGH = "high"              # 0.7-0.9
    MEDIUM = "medium"          # 0.5-0.7
    LOW = "low"                # 0.3-0.5
    VERY_LOW = "very_low"      # <0.3


@dataclass
class RoutingResult:
    """Result of domain routing decision."""
    target_domain: str
    confidence_score: float
    confidence_level: RoutingConfidence
    reasoning: str
    alternative_domains: List[Tuple[str, float]] = field(default_factory=list)
    routing_strategy: RoutingStrategy = RoutingStrategy.PATTERN_MATCHING
    routing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'target_domain': self.target_domain,
            'confidence_score': self.confidence_score,
            'confidence_level': self.confidence_level.value,
            'reasoning': self.reasoning,
            'alternative_domains': self.alternative_domains,
            'routing_strategy': self.routing_strategy.value,
            'routing_time_ms': self.routing_time_ms,
            'metadata': self.metadata
        }


@dataclass
class DomainPattern:
    """Pattern definition for domain routing."""
    domain_name: str
    patterns: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    regex_patterns: List[str] = field(default_factory=list)
    input_types: List[str] = field(default_factory=list)
    priority: int = 5  # 1-10, higher is more priority
    confidence_boost: float = 0.0  # Additional confidence for this domain
    
    def matches_input(self, input_data: str, input_type: str = "text") -> Tuple[bool, float]:
        """Check if input matches this domain pattern."""
        matches = 0
        total_checks = 0
        
        input_lower = input_data.lower() if isinstance(input_data, str) else str(input_data).lower()
        
        # Check keywords
        if self.keywords:
            total_checks += len(self.keywords)
            for keyword in self.keywords:
                if keyword.lower() in input_lower:
                    matches += 1
        
        # Check simple patterns
        if self.patterns:
            total_checks += len(self.patterns)
            for pattern in self.patterns:
                if pattern.lower() in input_lower:
                    matches += 1
        
        # Check regex patterns
        if self.regex_patterns:
            total_checks += len(self.regex_patterns)
            for regex_pattern in self.regex_patterns:
                try:
                    if re.search(regex_pattern, input_data, re.IGNORECASE):
                        matches += 1
                except re.error:
                    continue
        
        # Check input types
        if self.input_types:
            total_checks += 1
            if input_type in self.input_types:
                matches += 1
        
        if total_checks == 0:
            return False, 0.0
        
        base_confidence = matches / total_checks
        final_confidence = min(1.0, base_confidence + self.confidence_boost)
        
        return matches > 0, final_confidence


@dataclass
class RoutingMetrics:
    """Metrics for routing performance tracking."""
    total_routes: int = 0
    successful_routes: int = 0
    failed_routes: int = 0
    average_routing_time_ms: float = 0.0
    domain_usage_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    confidence_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    strategy_usage: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_reset: datetime = field(default_factory=datetime.now)
    
    def record_route(self, domain: str, confidence: float, strategy: RoutingStrategy, 
                    routing_time_ms: float, success: bool = True):
        """Record a routing decision."""
        self.total_routes += 1
        if success:
            self.successful_routes += 1
        else:
            self.failed_routes += 1
        
        self.domain_usage_counts[domain] += 1
        self.strategy_usage[strategy.value] += 1
        
        # Update average routing time
        self.average_routing_time_ms = (
            (self.average_routing_time_ms * (self.total_routes - 1) + routing_time_ms) 
            / self.total_routes
        )
        
        # Track confidence distribution
        if confidence >= 0.9:
            self.confidence_distribution["very_high"] += 1
        elif confidence >= 0.7:
            self.confidence_distribution["high"] += 1
        elif confidence >= 0.5:
            self.confidence_distribution["medium"] += 1
        elif confidence >= 0.3:
            self.confidence_distribution["low"] += 1
        else:
            self.confidence_distribution["very_low"] += 1


class DomainRouter:
    """
    Intelligent domain routing system that determines which domain should handle incoming requests.
    Supports multiple routing strategies and provides detailed routing decisions.
    """
    
    def __init__(self, domain_registry=None, config: Optional[Dict[str, Any]] = None):
        """Initialize domain router with registry and configuration."""
        self.domain_registry = domain_registry
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Routing configuration
        self.default_strategy = RoutingStrategy(
            self.config.get('default_strategy', RoutingStrategy.HYBRID.value)
        )
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.enable_fallback = self.config.get('enable_fallback', True)
        self.fallback_domain = self.config.get('fallback_domain', 'general')
        
        # Domain patterns for pattern-based routing
        self.domain_patterns: Dict[str, DomainPattern] = {}
        self.pattern_cache: Dict[str, RoutingResult] = {}
        self.cache_max_size = self.config.get('cache_max_size', 1000)
        
        # Routing history and metrics
        self.routing_history: deque = deque(maxlen=self.config.get('history_size', 1000))
        self.routing_metrics = RoutingMetrics()
        
        # Machine learning features (placeholder for future ML implementation)
        self.ml_model = None
        self.feature_extractors = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize default patterns
        self._initialize_default_patterns()
        
        self.logger.info(f"DomainRouter initialized with strategy: {self.default_strategy.value}")
    
    def _initialize_default_patterns(self):
        """Initialize default domain patterns for common domains."""
        # General/default domain patterns
        self.add_domain_pattern(DomainPattern(
            domain_name="general",
            keywords=["general", "basic", "simple", "help", "info"],
            patterns=["what is", "how to", "tell me about"],
            input_types=["text", "query"],
            priority=1
        ))
        
        # Mathematical domain patterns
        self.add_domain_pattern(DomainPattern(
            domain_name="mathematics",
            keywords=["math", "calculate", "equation", "formula", "solve", "algebra", "geometry"],
            patterns=["calculate", "solve for", "find the", "what is the result"],
            regex_patterns=[r'\d+[\+\-\*/]\d+', r'x\s*=', r'f\(x\)', r'\d+\^\d+'],
            input_types=["math", "equation", "calculation"],
            priority=8,
            confidence_boost=0.1
        ))
        
        # Scientific domain patterns
        self.add_domain_pattern(DomainPattern(
            domain_name="science",
            keywords=["chemistry", "physics", "biology", "molecule", "atom", "reaction", "experiment"],
            patterns=["chemical formula", "molecular structure", "scientific method"],
            regex_patterns=[r'[A-Z][a-z]?\d*', r'pH\s*\d+', r'\d+\s*mol'],
            input_types=["scientific", "chemical", "molecular"],
            priority=8,
            confidence_boost=0.15
        ))
        
        # Programming domain patterns
        self.add_domain_pattern(DomainPattern(
            domain_name="programming",
            keywords=["code", "programming", "function", "algorithm", "debug", "compile", "syntax"],
            patterns=["write a function", "debug this code", "implement algorithm"],
            regex_patterns=[r'def\s+\w+', r'class\s+\w+', r'import\s+\w+', r'#include'],
            input_types=["code", "programming", "algorithm"],
            priority=7,
            confidence_boost=0.1
        ))
        
        # Language/text domain patterns
        self.add_domain_pattern(DomainPattern(
            domain_name="language",
            keywords=["translate", "grammar", "language", "meaning", "definition", "synonym"],
            patterns=["what does", "translate to", "meaning of"],
            input_types=["text", "language", "translation"],
            priority=6
        ))
    
    def add_domain_pattern(self, pattern: DomainPattern) -> bool:
        """Add a domain pattern for routing decisions."""
        with self._lock:
            try:
                self.domain_patterns[pattern.domain_name] = pattern
                self.logger.debug(f"Added pattern for domain: {pattern.domain_name}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to add domain pattern: {e}")
                return False
    
    def remove_domain_pattern(self, domain_name: str) -> bool:
        """Remove domain pattern."""
        with self._lock:
            if domain_name in self.domain_patterns:
                del self.domain_patterns[domain_name]
                self.logger.debug(f"Removed pattern for domain: {domain_name}")
                return True
            return False
    
    def route_request(self, input_data: Any, domain_hint: Optional[str] = None,
                     strategy: Optional[RoutingStrategy] = None,
                     input_type: str = "text") -> RoutingResult:
        """
        Route request to appropriate domain with comprehensive analysis.
        
        Args:
            input_data: Input data to be routed
            domain_hint: Optional hint for target domain
            strategy: Optional routing strategy to use
            input_type: Type of input data
            
        Returns:
            RoutingResult with routing decision and metadata
        """
        start_time = time.time()
        
        with self._lock:
            try:
                # Use provided strategy or default
                routing_strategy = strategy or self.default_strategy
                
                # Convert input to string for processing
                input_str = str(input_data) if not isinstance(input_data, str) else input_data
                
                # Check cache first
                cache_key = f"{hash(input_str)}_{routing_strategy.value}_{domain_hint}"
                if cache_key in self.pattern_cache:
                    cached_result = self.pattern_cache[cache_key]
                    cached_result.routing_time_ms = (time.time() - start_time) * 1000
                    return cached_result
                
                # Route based on strategy
                if routing_strategy == RoutingStrategy.PATTERN_MATCHING:
                    result = self._route_by_patterns(input_str, input_type, domain_hint)
                elif routing_strategy == RoutingStrategy.HEURISTIC_BASED:
                    result = self._route_by_heuristics(input_str, input_type, domain_hint)
                elif routing_strategy == RoutingStrategy.ML_BASED:
                    result = self._route_by_ml(input_str, input_type, domain_hint)
                elif routing_strategy == RoutingStrategy.HYBRID:
                    result = self._route_hybrid(input_str, input_type, domain_hint)
                elif routing_strategy == RoutingStrategy.CONFIDENCE_WEIGHTED:
                    result = self._route_confidence_weighted(input_str, input_type, domain_hint)
                else:
                    result = self._route_by_patterns(input_str, input_type, domain_hint)
                
                # Set routing metadata
                result.routing_strategy = routing_strategy
                result.routing_time_ms = (time.time() - start_time) * 1000
                
                # Apply confidence threshold and fallback
                if result.confidence_score < self.confidence_threshold and self.enable_fallback:
                    original_domain = result.target_domain
                    result.target_domain = self.fallback_domain
                    result.reasoning += f" (Fallback from {original_domain} due to low confidence)"
                    result.metadata['fallback_applied'] = True
                    result.metadata['original_domain'] = original_domain
                
                # Set confidence level
                result.confidence_level = self._get_confidence_level(result.confidence_score)
                
                # Cache result
                if len(self.pattern_cache) < self.cache_max_size:
                    self.pattern_cache[cache_key] = result
                
                # Record metrics
                self.routing_metrics.record_route(
                    result.target_domain,
                    result.confidence_score,
                    routing_strategy,
                    result.routing_time_ms
                )
                
                # Add to history
                self.routing_history.append({
                    'timestamp': datetime.now(),
                    'input_preview': input_str[:100] + "..." if len(input_str) > 100 else input_str,
                    'result': result.to_dict()
                })
                
                self.logger.debug(f"Routed to domain: {result.target_domain} "
                                f"(confidence: {result.confidence_score:.3f})")
                
                return result
                
            except Exception as e:
                self.logger.error(f"Routing failed: {e}")
                # Return fallback result
                return RoutingResult(
                    target_domain=self.fallback_domain,
                    confidence_score=0.1,
                    confidence_level=RoutingConfidence.VERY_LOW,
                    reasoning=f"Routing error, using fallback: {str(e)}",
                    routing_strategy=routing_strategy,
                    routing_time_ms=(time.time() - start_time) * 1000,
                    metadata={'error': str(e)}
                )
    
    def _route_by_patterns(self, input_str: str, input_type: str, 
                          domain_hint: Optional[str]) -> RoutingResult:
        """Route using pattern matching."""
        best_domain = self.fallback_domain
        best_confidence = 0.0
        reasoning = "Pattern matching: "
        alternatives = []
        
        # Check domain hint first
        if domain_hint and domain_hint in self.domain_patterns:
            pattern = self.domain_patterns[domain_hint]
            matches, confidence = pattern.matches_input(input_str, input_type)
            if matches:
                return RoutingResult(
                    target_domain=domain_hint,
                    confidence_score=min(1.0, confidence * 1.2),  # Boost for hint
                    reasoning=f"Domain hint matched with confidence {confidence:.3f}",
                    alternative_domains=[]
                )
        
        # Check all domain patterns
        for domain_name, pattern in self.domain_patterns.items():
            matches, confidence = pattern.matches_input(input_str, input_type)
            
            if matches:
                # Apply priority weighting
                weighted_confidence = confidence * (pattern.priority / 10.0)
                alternatives.append((domain_name, confidence))
                
                if weighted_confidence > best_confidence:
                    best_confidence = weighted_confidence
                    best_domain = domain_name
                    reasoning = f"Best pattern match for {domain_name} "
        
        # Sort alternatives by confidence
        alternatives.sort(key=lambda x: x[1], reverse=True)
        alternatives = alternatives[:5]  # Keep top 5
        
        if best_confidence == 0.0:
            reasoning += "No pattern matches found, using fallback"
        else:
            reasoning += f"with confidence {best_confidence:.3f}"
        
        return RoutingResult(
            target_domain=best_domain,
            confidence_score=best_confidence,
            reasoning=reasoning,
            alternative_domains=alternatives
        )
    
    def _route_by_heuristics(self, input_str: str, input_type: str,
                           domain_hint: Optional[str]) -> RoutingResult:
        """Route using heuristic analysis."""
        # Start with pattern matching as base
        pattern_result = self._route_by_patterns(input_str, input_type, domain_hint)
        
        # Apply heuristic boosters
        heuristic_confidence = pattern_result.confidence_score
        reasoning = "Heuristic analysis: "
        
        # Length-based heuristics
        if len(input_str) > 500:
            # Long inputs might need specialized domains
            if "code" in input_str.lower() or any(x in input_str for x in ["def ", "class ", "import "]):
                heuristic_confidence += 0.2
                reasoning += "Long code detected. "
            elif any(x in input_str.lower() for x in ["molecule", "chemical", "reaction"]):
                heuristic_confidence += 0.15
                reasoning += "Scientific content detected. "
        
        # Complexity heuristics
        if any(char in input_str for char in "∫∑∂∆αβγδε"):
            heuristic_confidence += 0.25
            reasoning += "Mathematical symbols detected. "
        
        # Question type heuristics
        if input_str.strip().endswith('?'):
            if input_str.lower().startswith(('what', 'how', 'why', 'when', 'where')):
                reasoning += "Question pattern detected. "
                heuristic_confidence = max(heuristic_confidence, 0.6)
        
        return RoutingResult(
            target_domain=pattern_result.target_domain,
            confidence_score=min(1.0, heuristic_confidence),
            reasoning=reasoning + pattern_result.reasoning,
            alternative_domains=pattern_result.alternative_domains
        )
    
    def _route_by_ml(self, input_str: str, input_type: str,
                    domain_hint: Optional[str]) -> RoutingResult:
        """Route using machine learning (placeholder for future implementation)."""
        # For now, fall back to heuristic routing with ML placeholder
        heuristic_result = self._route_by_heuristics(input_str, input_type, domain_hint)
        
        # Placeholder for ML model predictions
        if self.ml_model:
            # Future: Use trained model to predict domain
            pass
        
        heuristic_result.reasoning = "ML-based routing (fallback to heuristics): " + heuristic_result.reasoning
        return heuristic_result
    
    def _route_hybrid(self, input_str: str, input_type: str,
                     domain_hint: Optional[str]) -> RoutingResult:
        """Route using hybrid approach combining multiple strategies."""
        # Get results from different strategies
        pattern_result = self._route_by_patterns(input_str, input_type, domain_hint)
        heuristic_result = self._route_by_heuristics(input_str, input_type, domain_hint)
        
        # Combine results with weighted average
        pattern_weight = 0.6
        heuristic_weight = 0.4
        
        if pattern_result.target_domain == heuristic_result.target_domain:
            # Both agree - high confidence
            combined_confidence = (
                pattern_result.confidence_score * pattern_weight +
                heuristic_result.confidence_score * heuristic_weight
            )
            combined_confidence = min(1.0, combined_confidence * 1.1)  # Agreement boost
            
            return RoutingResult(
                target_domain=pattern_result.target_domain,
                confidence_score=combined_confidence,
                reasoning=f"Hybrid consensus: {pattern_result.target_domain}",
                alternative_domains=pattern_result.alternative_domains
            )
        else:
            # Disagreement - choose higher confidence
            if pattern_result.confidence_score > heuristic_result.confidence_score:
                chosen_result = pattern_result
                alternative_domain = heuristic_result.target_domain
            else:
                chosen_result = heuristic_result
                alternative_domain = pattern_result.target_domain
            
            # Reduce confidence due to disagreement
            chosen_result.confidence_score *= 0.8
            chosen_result.reasoning = f"Hybrid choice (disagreement): {chosen_result.reasoning}"
            chosen_result.alternative_domains.append((alternative_domain, 0.5))
            
            return chosen_result
    
    def _route_confidence_weighted(self, input_str: str, input_type: str,
                                 domain_hint: Optional[str]) -> RoutingResult:
        """Route using confidence-weighted ensemble of strategies."""
        results = []
        
        # Get results from all available strategies
        results.append(self._route_by_patterns(input_str, input_type, domain_hint))
        results.append(self._route_by_heuristics(input_str, input_type, domain_hint))
        
        # Weight by confidence and calculate ensemble
        total_weight = sum(r.confidence_score for r in results)
        
        if total_weight == 0:
            return results[0]  # Return first if all have zero confidence
        
        domain_scores = defaultdict(float)
        reasoning_parts = []
        
        for result in results:
            weight = result.confidence_score / total_weight
            domain_scores[result.target_domain] += weight
            reasoning_parts.append(f"{result.target_domain}({weight:.2f})")
        
        # Choose domain with highest weighted score
        best_domain = max(domain_scores.items(), key=lambda x: x[1])
        
        return RoutingResult(
            target_domain=best_domain[0],
            confidence_score=best_domain[1],
            reasoning=f"Confidence-weighted ensemble: {', '.join(reasoning_parts)}",
            alternative_domains=[(k, v) for k, v in sorted(domain_scores.items(), 
                                                          key=lambda x: x[1], reverse=True)[1:]]
        )
    
    def _get_confidence_level(self, confidence_score: float) -> RoutingConfidence:
        """Convert confidence score to confidence level enum."""
        if confidence_score >= 0.9:
            return RoutingConfidence.VERY_HIGH
        elif confidence_score >= 0.7:
            return RoutingConfidence.HIGH
        elif confidence_score >= 0.5:
            return RoutingConfidence.MEDIUM
        elif confidence_score >= 0.3:
            return RoutingConfidence.LOW
        else:
            return RoutingConfidence.VERY_LOW
    
    def detect_domain(self, input_data: Any, input_type: str = "text") -> str:
        """Auto-detect domain from input data."""
        result = self.route_request(input_data, input_type=input_type)
        return result.target_domain
    
    def validate_routing(self, domain_name: str, input_data: Any,
                        expected_confidence: float = 0.5) -> bool:
        """Validate that input should be routed to specified domain."""
        result = self.route_request(input_data)
        return (result.target_domain == domain_name and 
                result.confidence_score >= expected_confidence)
    
    def get_routing_confidence(self, input_data: Any, domain_name: str) -> float:
        """Get confidence score for routing input to specific domain."""
        if domain_name not in self.domain_patterns:
            return 0.0
        
        input_str = str(input_data) if not isinstance(input_data, str) else input_data
        pattern = self.domain_patterns[domain_name]
        matches, confidence = pattern.matches_input(input_str)
        
        return confidence if matches else 0.0
    
    def get_available_domains(self) -> List[str]:
        """Get list of all available domains for routing."""
        with self._lock:
            domains = list(self.domain_patterns.keys())
            if self.domain_registry:
                registered_domains = self.domain_registry.list_domains()
                domains.extend([d for d in registered_domains if d not in domains])
            return sorted(set(domains))
    
    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get comprehensive routing metrics."""
        with self._lock:
            return {
                'total_routes': self.routing_metrics.total_routes,
                'successful_routes': self.routing_metrics.successful_routes,
                'failed_routes': self.routing_metrics.failed_routes,
                'success_rate': (
                    self.routing_metrics.successful_routes / self.routing_metrics.total_routes
                    if self.routing_metrics.total_routes > 0 else 0.0
                ),
                'average_routing_time_ms': self.routing_metrics.average_routing_time_ms,
                'domain_usage_counts': dict(self.routing_metrics.domain_usage_counts),
                'confidence_distribution': dict(self.routing_metrics.confidence_distribution),
                'strategy_usage': dict(self.routing_metrics.strategy_usage),
                'cache_size': len(self.pattern_cache),
                'history_size': len(self.routing_history)
            }
    
    def get_routing_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent routing history."""
        with self._lock:
            return list(self.routing_history)[-limit:]
    
    def clear_cache(self) -> None:
        """Clear routing cache."""
        with self._lock:
            self.pattern_cache.clear()
            self.logger.debug("Routing cache cleared")
    
    def reset_metrics(self) -> None:
        """Reset routing metrics."""
        with self._lock:
            self.routing_metrics = RoutingMetrics()
            self.logger.debug("Routing metrics reset")
    
    def export_patterns(self, filepath: Path) -> bool:
        """Export domain patterns to file."""
        try:
            patterns_data = {}
            for domain_name, pattern in self.domain_patterns.items():
                patterns_data[domain_name] = {
                    'patterns': pattern.patterns,
                    'keywords': pattern.keywords,
                    'regex_patterns': pattern.regex_patterns,
                    'input_types': pattern.input_types,
                    'priority': pattern.priority,
                    'confidence_boost': pattern.confidence_boost
                }
            
            with open(filepath, 'w') as f:
                json.dump(patterns_data, f, indent=2)
            
            self.logger.info(f"Exported {len(patterns_data)} domain patterns to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export patterns: {e}")
            return False
    
    def import_patterns(self, filepath: Path) -> bool:
        """Import domain patterns from file."""
        try:
            with open(filepath, 'r') as f:
                patterns_data = json.load(f)
            
            imported_count = 0
            for domain_name, pattern_config in patterns_data.items():
                pattern = DomainPattern(
                    domain_name=domain_name,
                    patterns=pattern_config.get('patterns', []),
                    keywords=pattern_config.get('keywords', []),
                    regex_patterns=pattern_config.get('regex_patterns', []),
                    input_types=pattern_config.get('input_types', []),
                    priority=pattern_config.get('priority', 5),
                    confidence_boost=pattern_config.get('confidence_boost', 0.0)
                )
                
                if self.add_domain_pattern(pattern):
                    imported_count += 1
            
            self.logger.info(f"Imported {imported_count} domain patterns from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import patterns: {e}")
            return False
    
    def optimize_routing(self) -> Dict[str, Any]:
        """Analyze routing performance and suggest optimizations."""
        with self._lock:
            metrics = self.get_routing_metrics()
            suggestions = []
            
            # Analyze cache hit rate
            if self.routing_metrics.total_routes > 100:
                cache_efficiency = len(self.pattern_cache) / self.routing_metrics.total_routes
                if cache_efficiency < 0.1:
                    suggestions.append("Consider increasing cache size for better performance")
            
            # Analyze confidence distribution
            low_confidence_rate = (
                metrics['confidence_distribution'].get('low', 0) +
                metrics['confidence_distribution'].get('very_low', 0)
            ) / max(self.routing_metrics.total_routes, 1)
            
            if low_confidence_rate > 0.3:
                suggestions.append("High rate of low-confidence routing - consider refining domain patterns")
            
            # Analyze domain usage balance
            domain_counts = list(metrics['domain_usage_counts'].values())
            if domain_counts:
                max_usage = max(domain_counts)
                min_usage = min(domain_counts)
                if max_usage > min_usage * 10:
                    suggestions.append("Unbalanced domain usage - some domains may need better patterns")
            
            return {
                'current_metrics': metrics,
                'optimization_suggestions': suggestions,
                'cache_efficiency': len(self.pattern_cache) / max(self.routing_metrics.total_routes, 1),
                'low_confidence_rate': low_confidence_rate
            }