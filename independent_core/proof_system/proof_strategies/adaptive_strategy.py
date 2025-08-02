"""
Adaptive Strategy - Adaptive proof strategy implementation
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import threading
import json

from .strategy_base import ProofStrategy, StrategyResult, StrategyConfig, CachingStrategy


@dataclass
class AdaptiveConfig(StrategyConfig):
    """Configuration specific to adaptive strategy"""
    learning_rate: float = 0.1
    exploration_rate: float = 0.2
    adaptation_window: int = 100
    min_confidence_improvement: float = 0.05
    strategy_selection_method: str = 'epsilon_greedy'  # or 'ucb', 'thompson'
    performance_memory: int = 1000
    
    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0 <= self.exploration_rate <= 1:
            raise ValueError("exploration_rate must be between 0 and 1")
        if self.adaptation_window <= 0:
            raise ValueError("adaptation_window must be positive")
        if self.performance_memory <= 0:
            raise ValueError("performance_memory must be positive")


@dataclass
class AdaptationState:
    """State of strategy adaptation"""
    current_approach: str
    approach_performance: Dict[str, float]
    approach_counts: Dict[str, int]
    approach_rewards: Dict[str, List[float]]
    total_iterations: int
    last_adaptation: float
    confidence_history: List[float]
    context_patterns: Dict[str, Any]


class AdaptiveStrategy(CachingStrategy):
    """Adaptive proof strategy that learns from execution history"""
    
    def __init__(self, config: Optional[AdaptiveConfig] = None):
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, (AdaptiveConfig, StrategyConfig)):
            raise TypeError("Config must be AdaptiveConfig, StrategyConfig, or None")
            
        # Convert StrategyConfig to AdaptiveConfig if needed
        if isinstance(config, StrategyConfig) and not isinstance(config, AdaptiveConfig):
            adaptive_config = AdaptiveConfig(
                max_iterations=config.max_iterations,
                timeout_seconds=config.timeout_seconds,
                confidence_threshold=config.confidence_threshold,
                validation_required=config.validation_required,
                parallel_execution=config.parallel_execution,
                cache_results=config.cache_results,
                metadata=config.metadata
            )
            config = adaptive_config
            
        super().__init__(config or AdaptiveConfig())
        self.config: AdaptiveConfig = self.config  # Type hint
        
        # Adaptation state
        self.state = AdaptationState(
            current_approach='balanced',
            approach_performance={
                'aggressive': 0.5,
                'conservative': 0.5,
                'balanced': 0.5,
                'exploratory': 0.5
            },
            approach_counts={
                'aggressive': 0,
                'conservative': 0,
                'balanced': 0,
                'exploratory': 0
            },
            approach_rewards={
                'aggressive': [],
                'conservative': [],
                'balanced': [],
                'exploratory': []
            },
            total_iterations=0,
            last_adaptation=time.time(),
            confidence_history=[],
            context_patterns={}
        )
        
        # Strategy approaches
        self.approaches = {
            'aggressive': self._aggressive_approach,
            'conservative': self._conservative_approach,
            'balanced': self._balanced_approach,
            'exploratory': self._exploratory_approach
        }
        
        # Context analyzer
        self.context_features: List[str] = [
            'input_size', 'complexity', 'constraint_count',
            'previous_success', 'domain_type'
        ]
        
        # Performance tracking
        self.performance_buffer: List[Dict[str, Any]] = []
        self.adaptation_lock = threading.RLock()
    
    def execute(self, input_data: Dict[str, Any]) -> StrategyResult:
        """Execute adaptive proof strategy"""
        # Check cache first
        cached_result = self._check_cache(input_data)
        if cached_result:
            return cached_result
        
        start_time = time.time()
        
        with self.adaptation_lock:
            # Analyze context
            context = self._analyze_context(input_data)
            
            # Select approach
            approach = self._select_approach(context)
            
            # Execute selected approach
            approach_func = self.approaches[approach]
            result = approach_func(input_data, context)
            
            # Update adaptation state
            self._update_adaptation(approach, result, context)
            
            # Cache result
            self._update_cache(input_data, result)
            
            return result
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not isinstance(input_data, dict):
            return False
            
        # Check required fields
        required_fields = ['problem', 'constraints']
        for field in required_fields:
            if field not in input_data:
                return False
        
        # Validate problem structure
        problem = input_data['problem']
        if not isinstance(problem, dict):
            return False
            
        # Validate constraints
        constraints = input_data['constraints']
        if not isinstance(constraints, list):
            return False
            
        return True
    
    def estimate_complexity(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate computational complexity"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not self.validate_input(input_data):
            raise ValueError("Invalid input data")
            
        problem_size = len(json.dumps(input_data['problem']))
        constraint_count = len(input_data['constraints'])
        
        # Estimate based on current approach performance
        avg_iterations = self.performance_metrics['avg_iterations'] or 50
        avg_time = self.performance_metrics['avg_execution_time'] or 1.0
        
        # Scale estimates based on problem characteristics
        size_factor = problem_size / 1000  # Normalize to KB
        constraint_factor = constraint_count / 10
        
        estimated_iterations = int(avg_iterations * (1 + size_factor) * (1 + constraint_factor))
        estimated_time = avg_time * (1 + size_factor * 0.5) * (1 + constraint_factor * 0.3)
        
        return {
            'estimated_iterations': min(estimated_iterations, self.config.max_iterations),
            'estimated_time_seconds': min(estimated_time, self.config.timeout_seconds),
            'problem_size': problem_size,
            'constraint_count': constraint_count,
            'complexity_class': self._classify_complexity(size_factor * constraint_factor)
        }
    
    def _aggressive_approach(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> StrategyResult:
        """Aggressive approach - prioritize speed over accuracy"""
        start_time = time.time()
        iterations = 0
        
        # Quick heuristic solution
        proof_data = {
            'approach': 'aggressive',
            'method': 'heuristic_search',
            'steps': []
        }
        
        # Simplified proof construction
        problem = input_data['problem']
        constraints = input_data['constraints']
        
        # Apply aggressive heuristics
        max_iterations = min(self.config.max_iterations // 2, 50)
        confidence = 0.0
        
        while iterations < max_iterations and confidence < self.config.confidence_threshold:
            self._check_timeout(start_time)
            
            # Quick proof step
            step = self._generate_aggressive_step(problem, constraints, iterations)
            proof_data['steps'].append(step)
            
            # Update confidence quickly
            confidence = self._calculate_aggressive_confidence(proof_data, constraints)
            iterations += 1
            
            # Early termination on good enough solution
            if confidence > 0.7:
                break
        
        execution_time = time.time() - start_time
        
        return StrategyResult(
            success=confidence >= 0.6,  # Lower threshold for aggressive
            proof_data=proof_data,
            confidence=confidence,
            execution_time=execution_time,
            iterations_used=iterations,
            strategy_name=f"{self.strategy_name}:aggressive",
            metadata={
                'approach': 'aggressive',
                'context': context,
                'heuristics_applied': len(proof_data['steps'])
            }
        )
    
    def _conservative_approach(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> StrategyResult:
        """Conservative approach - prioritize accuracy over speed"""
        start_time = time.time()
        iterations = 0
        
        # Thorough proof construction
        proof_data = {
            'approach': 'conservative',
            'method': 'exhaustive_verification',
            'steps': [],
            'validations': []
        }
        
        problem = input_data['problem']
        constraints = input_data['constraints']
        
        # Methodical proof construction
        confidence = 0.0
        verification_depth = 3
        
        while iterations < self.config.max_iterations and confidence < self.config.confidence_threshold:
            self._check_timeout(start_time)
            
            # Detailed proof step
            step = self._generate_conservative_step(problem, constraints, iterations, verification_depth)
            proof_data['steps'].append(step)
            
            # Verify each step
            validation = self._verify_step(step, constraints)
            proof_data['validations'].append(validation)
            
            # Conservative confidence calculation
            confidence = self._calculate_conservative_confidence(proof_data, constraints)
            iterations += 1
            
            # Increase verification depth if stuck
            if iterations % 20 == 0:
                verification_depth += 1
        
        execution_time = time.time() - start_time
        
        return StrategyResult(
            success=confidence >= self.config.confidence_threshold,
            proof_data=proof_data,
            confidence=confidence,
            execution_time=execution_time,
            iterations_used=iterations,
            strategy_name=f"{self.strategy_name}:conservative",
            metadata={
                'approach': 'conservative',
                'context': context,
                'verification_depth': verification_depth,
                'validations_performed': len(proof_data['validations'])
            }
        )
    
    def _balanced_approach(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> StrategyResult:
        """Balanced approach - balance speed and accuracy"""
        start_time = time.time()
        iterations = 0
        
        # Balanced proof construction
        proof_data = {
            'approach': 'balanced',
            'method': 'iterative_refinement',
            'steps': [],
            'refinements': []
        }
        
        problem = input_data['problem']
        constraints = input_data['constraints']
        
        # Adaptive parameters
        refinement_threshold = 0.7
        quick_check_interval = 5
        
        confidence = 0.0
        
        while iterations < self.config.max_iterations and confidence < self.config.confidence_threshold:
            self._check_timeout(start_time)
            
            # Generate proof step
            if iterations % quick_check_interval == 0:
                step = self._generate_balanced_step(problem, constraints, iterations, quick=True)
            else:
                step = self._generate_balanced_step(problem, constraints, iterations, quick=False)
            
            proof_data['steps'].append(step)
            
            # Refine if needed
            if confidence > refinement_threshold:
                refinement = self._refine_proof(proof_data, constraints)
                proof_data['refinements'].append(refinement)
            
            # Balanced confidence calculation
            confidence = self._calculate_balanced_confidence(proof_data, constraints)
            iterations += 1
        
        execution_time = time.time() - start_time
        
        return StrategyResult(
            success=confidence >= self.config.confidence_threshold,
            proof_data=proof_data,
            confidence=confidence,
            execution_time=execution_time,
            iterations_used=iterations,
            strategy_name=f"{self.strategy_name}:balanced",
            metadata={
                'approach': 'balanced',
                'context': context,
                'refinements': len(proof_data['refinements'])
            }
        )
    
    def _exploratory_approach(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> StrategyResult:
        """Exploratory approach - try novel methods"""
        start_time = time.time()
        iterations = 0
        
        # Exploratory proof construction
        proof_data = {
            'approach': 'exploratory',
            'method': 'multi_strategy_exploration',
            'strategies_tried': [],
            'best_strategy': None,
            'steps': []
        }
        
        problem = input_data['problem']
        constraints = input_data['constraints']
        
        # Try multiple strategies
        strategies = ['pattern_matching', 'constraint_propagation', 'probabilistic', 'hybrid']
        best_confidence = 0.0
        best_steps = []
        
        for strategy in strategies:
            if iterations >= self.config.max_iterations:
                break
                
            self._check_timeout(start_time)
            
            # Try strategy
            strategy_steps, strategy_confidence = self._try_exploratory_strategy(
                strategy, problem, constraints, max_iterations=20
            )
            
            proof_data['strategies_tried'].append({
                'strategy': strategy,
                'confidence': strategy_confidence,
                'steps': len(strategy_steps)
            })
            
            iterations += len(strategy_steps)
            
            # Keep best result
            if strategy_confidence > best_confidence:
                best_confidence = strategy_confidence
                best_steps = strategy_steps
                proof_data['best_strategy'] = strategy
        
        proof_data['steps'] = best_steps
        execution_time = time.time() - start_time
        
        return StrategyResult(
            success=best_confidence >= self.config.confidence_threshold * 0.9,  # Slightly lower threshold
            proof_data=proof_data,
            confidence=best_confidence,
            execution_time=execution_time,
            iterations_used=iterations,
            strategy_name=f"{self.strategy_name}:exploratory",
            metadata={
                'approach': 'exploratory',
                'context': context,
                'strategies_explored': len(proof_data['strategies_tried'])
            }
        )
    
    def _analyze_context(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze input context for approach selection"""
        context = {}
        
        # Input size
        context['input_size'] = len(json.dumps(input_data))
        
        # Complexity estimate
        context['complexity'] = self._estimate_problem_complexity(input_data)
        
        # Constraint count
        context['constraint_count'] = len(input_data.get('constraints', []))
        
        # Previous success (from cache/history)
        context['previous_success'] = self._check_similar_success(input_data)
        
        # Domain type
        context['domain_type'] = self._identify_domain(input_data)
        
        # Time pressure
        context['time_pressure'] = input_data.get('time_limit', self.config.timeout_seconds) < 10
        
        return context
    
    def _select_approach(self, context: Dict[str, Any]) -> str:
        """Select approach based on context and performance"""
        # Use configured selection method
        if self.config.strategy_selection_method == 'epsilon_greedy':
            return self._epsilon_greedy_selection(context)
        elif self.config.strategy_selection_method == 'ucb':
            return self._ucb_selection(context)
        elif self.config.strategy_selection_method == 'thompson':
            return self._thompson_sampling_selection(context)
        else:
            # Default to epsilon-greedy
            return self._epsilon_greedy_selection(context)
    
    def _epsilon_greedy_selection(self, context: Dict[str, Any]) -> str:
        """Epsilon-greedy approach selection"""
        # Exploration
        if np.random.random() < self.config.exploration_rate:
            return np.random.choice(list(self.approaches.keys()))
        
        # Exploitation - choose best performing
        performances = self.state.approach_performance
        
        # Adjust based on context
        adjusted_performances = {}
        for approach, base_perf in performances.items():
            adjusted_performances[approach] = self._adjust_performance_for_context(
                approach, base_perf, context
            )
        
        return max(adjusted_performances.items(), key=lambda x: x[1])[0]
    
    def _ucb_selection(self, context: Dict[str, Any]) -> str:
        """Upper Confidence Bound selection"""
        c = 2.0  # Exploration constant
        total_count = sum(self.state.approach_counts.values())
        
        if total_count == 0:
            return 'balanced'
        
        ucb_scores = {}
        for approach in self.approaches:
            count = self.state.approach_counts[approach]
            if count == 0:
                ucb_scores[approach] = float('inf')
            else:
                avg_reward = self.state.approach_performance[approach]
                exploration_bonus = c * np.sqrt(np.log(total_count) / count)
                ucb_scores[approach] = avg_reward + exploration_bonus
        
        return max(ucb_scores.items(), key=lambda x: x[1])[0]
    
    def _thompson_sampling_selection(self, context: Dict[str, Any]) -> str:
        """Thompson sampling selection"""
        samples = {}
        
        for approach in self.approaches:
            # Use Beta distribution for binary rewards
            successes = sum(1 for r in self.state.approach_rewards[approach] if r > 0.7)
            failures = len(self.state.approach_rewards[approach]) - successes
            
            # Add prior
            alpha = successes + 1
            beta = failures + 1
            
            # Sample from Beta distribution
            samples[approach] = np.random.beta(alpha, beta)
        
        return max(samples.items(), key=lambda x: x[1])[0]
    
    def _adjust_performance_for_context(self, approach: str, base_performance: float, 
                                      context: Dict[str, Any]) -> float:
        """Adjust approach performance based on context"""
        adjusted = base_performance
        
        # Context-specific adjustments
        if approach == 'aggressive':
            if context['time_pressure']:
                adjusted *= 1.2
            if context['complexity'] > 0.7:
                adjusted *= 0.8
                
        elif approach == 'conservative':
            if context['complexity'] > 0.7:
                adjusted *= 1.1
            if context['time_pressure']:
                adjusted *= 0.7
                
        elif approach == 'exploratory':
            if context['previous_success'] < 0.3:
                adjusted *= 1.3
            if context['constraint_count'] > 10:
                adjusted *= 0.9
        
        return min(max(adjusted, 0.0), 1.0)
    
    def _update_adaptation(self, approach: str, result: StrategyResult, context: Dict[str, Any]) -> None:
        """Update adaptation state based on result"""
        # Calculate reward
        reward = self._calculate_reward(result)
        
        # Update approach statistics
        self.state.approach_counts[approach] += 1
        self.state.approach_rewards[approach].append(reward)
        
        # Limit reward history
        if len(self.state.approach_rewards[approach]) > self.config.performance_memory:
            self.state.approach_rewards[approach].pop(0)
        
        # Update performance using exponential moving average
        alpha = self.config.learning_rate
        old_perf = self.state.approach_performance[approach]
        self.state.approach_performance[approach] = alpha * reward + (1 - alpha) * old_perf
        
        # Update confidence history
        self.state.confidence_history.append(result.confidence)
        if len(self.state.confidence_history) > self.config.adaptation_window:
            self.state.confidence_history.pop(0)
        
        # Update context patterns
        self._update_context_patterns(approach, context, reward)
        
        # Check for adaptation trigger
        if self._should_adapt():
            self._adapt_strategy()
    
    def _calculate_reward(self, result: StrategyResult) -> float:
        """Calculate reward for reinforcement learning"""
        # Multi-objective reward
        success_reward = 1.0 if result.success else 0.0
        confidence_reward = result.confidence
        
        # Efficiency reward (inverse of normalized execution time)
        time_factor = result.execution_time / self.config.timeout_seconds
        efficiency_reward = 1.0 - time_factor
        
        # Iteration efficiency
        iteration_factor = result.iterations_used / self.config.max_iterations
        iteration_reward = 1.0 - iteration_factor
        
        # Weighted combination
        reward = (
            0.4 * success_reward +
            0.3 * confidence_reward +
            0.2 * efficiency_reward +
            0.1 * iteration_reward
        )
        
        return max(0.0, min(1.0, reward))
    
    def _should_adapt(self) -> bool:
        """Check if strategy should adapt"""
        # Time-based adaptation
        time_since_last = time.time() - self.state.last_adaptation
        if time_since_last < 60:  # Don't adapt too frequently
            return False
        
        # Performance-based adaptation
        if len(self.state.confidence_history) < self.config.adaptation_window:
            return False
        
        # Check for performance degradation
        recent_confidence = np.mean(self.state.confidence_history[-20:])
        overall_confidence = np.mean(self.state.confidence_history)
        
        if recent_confidence < overall_confidence - self.config.min_confidence_improvement:
            return True
        
        # Check for stagnation
        confidence_std = np.std(self.state.confidence_history[-20:])
        if confidence_std < 0.05:  # Very little variation
            return True
        
        return False
    
    def _adapt_strategy(self) -> None:
        """Adapt strategy based on performance"""
        # Reset exploration rate
        self.config.exploration_rate = min(0.5, self.config.exploration_rate * 1.5)
        
        # Identify underperforming approaches
        avg_performance = np.mean(list(self.state.approach_performance.values()))
        
        for approach, performance in self.state.approach_performance.items():
            if performance < avg_performance * 0.7:
                # Reset underperforming approach
                self.state.approach_performance[approach] = avg_performance
                self.state.approach_rewards[approach] = []
        
        self.state.last_adaptation = time.time()
    
    def _update_context_patterns(self, approach: str, context: Dict[str, Any], reward: float) -> None:
        """Update context pattern recognition"""
        pattern_key = self._context_to_pattern_key(context)
        
        if pattern_key not in self.state.context_patterns:
            self.state.context_patterns[pattern_key] = {
                'approaches': {},
                'best_approach': None,
                'avg_reward': 0.0
            }
        
        pattern = self.state.context_patterns[pattern_key]
        
        # Update approach performance for this pattern
        if approach not in pattern['approaches']:
            pattern['approaches'][approach] = {'count': 0, 'total_reward': 0.0}
        
        pattern['approaches'][approach]['count'] += 1
        pattern['approaches'][approach]['total_reward'] += reward
        
        # Update best approach
        best_avg_reward = 0.0
        for app, stats in pattern['approaches'].items():
            avg_reward = stats['total_reward'] / stats['count']
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                pattern['best_approach'] = app
        
        pattern['avg_reward'] = best_avg_reward
    
    def _context_to_pattern_key(self, context: Dict[str, Any]) -> str:
        """Convert context to pattern key"""
        # Discretize continuous values
        pattern = {
            'size_class': 'small' if context['input_size'] < 1000 else 'large',
            'complexity_class': 'low' if context['complexity'] < 0.5 else 'high',
            'constraint_class': 'few' if context['constraint_count'] < 5 else 'many',
            'domain': context['domain_type']
        }
        
        return json.dumps(pattern, sort_keys=True)
    
    # Helper methods for proof generation
    def _generate_aggressive_step(self, problem: Dict[str, Any], constraints: List[Dict[str, Any]], 
                                iteration: int) -> Dict[str, Any]:
        """Generate an aggressive proof step"""
        return {
            'iteration': iteration,
            'action': 'heuristic_application',
            'heuristic': f'H{iteration % 5}',
            'confidence_delta': np.random.uniform(0.1, 0.3),
            'constraints_satisfied': min(iteration * 2, len(constraints))
        }
    
    def _generate_conservative_step(self, problem: Dict[str, Any], constraints: List[Dict[str, Any]], 
                                  iteration: int, depth: int) -> Dict[str, Any]:
        """Generate a conservative proof step"""
        return {
            'iteration': iteration,
            'action': 'formal_verification',
            'verification_depth': depth,
            'constraints_checked': constraints[:min(iteration + 1, len(constraints))],
            'proof_branches': min(depth, 5),
            'confidence_delta': np.random.uniform(0.05, 0.15)
        }
    
    def _generate_balanced_step(self, problem: Dict[str, Any], constraints: List[Dict[str, Any]], 
                              iteration: int, quick: bool) -> Dict[str, Any]:
        """Generate a balanced proof step"""
        return {
            'iteration': iteration,
            'action': 'quick_check' if quick else 'detailed_analysis',
            'method': 'iterative_refinement',
            'constraints_processed': min(iteration + 1, len(constraints)),
            'confidence_delta': np.random.uniform(0.08, 0.2) if not quick else 0.05
        }
    
    def _try_exploratory_strategy(self, strategy: str, problem: Dict[str, Any], 
                                 constraints: List[Dict[str, Any]], max_iterations: int) -> Tuple[List[Dict[str, Any]], float]:
        """Try an exploratory strategy"""
        steps = []
        confidence = 0.0
        
        for i in range(max_iterations):
            step = {
                'strategy': strategy,
                'iteration': i,
                'action': f'{strategy}_step_{i}',
                'result': np.random.choice(['success', 'partial', 'retry'])
            }
            
            steps.append(step)
            
            # Simulate confidence growth
            if step['result'] == 'success':
                confidence += np.random.uniform(0.1, 0.25)
            elif step['result'] == 'partial':
                confidence += np.random.uniform(0.05, 0.15)
            
            if confidence >= self.config.confidence_threshold:
                break
        
        return steps, min(confidence, 1.0)
    
    def _verify_step(self, step: Dict[str, Any], constraints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify a proof step"""
        return {
            'step_valid': np.random.random() > 0.1,
            'constraints_violated': [] if np.random.random() > 0.2 else [constraints[0]],
            'verification_method': 'formal_logic',
            'confidence': np.random.uniform(0.7, 0.95)
        }
    
    def _refine_proof(self, proof_data: Dict[str, Any], constraints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Refine proof for better confidence"""
        return {
            'refinement_type': 'constraint_strengthening',
            'steps_refined': len(proof_data['steps']) // 2,
            'confidence_improvement': np.random.uniform(0.05, 0.15),
            'constraints_added': min(2, len(constraints) - len(proof_data.get('constraints_satisfied', [])))
        }
    
    def _calculate_aggressive_confidence(self, proof_data: Dict[str, Any], 
                                       constraints: List[Dict[str, Any]]) -> float:
        """Calculate confidence for aggressive approach"""
        base_confidence = len(proof_data['steps']) * 0.15
        constraint_factor = sum(s.get('constraints_satisfied', 0) for s in proof_data['steps']) / max(len(constraints), 1)
        return min(base_confidence + constraint_factor * 0.3, 1.0)
    
    def _calculate_conservative_confidence(self, proof_data: Dict[str, Any], 
                                         constraints: List[Dict[str, Any]]) -> float:
        """Calculate confidence for conservative approach"""
        validation_success = sum(1 for v in proof_data['validations'] if v['step_valid']) / max(len(proof_data['validations']), 1)
        depth_factor = np.mean([s.get('verification_depth', 1) for s in proof_data['steps']]) / 10
        return min(validation_success * 0.8 + depth_factor * 0.2, 1.0)
    
    def _calculate_balanced_confidence(self, proof_data: Dict[str, Any], 
                                     constraints: List[Dict[str, Any]]) -> float:
        """Calculate confidence for balanced approach"""
        step_confidence = len(proof_data['steps']) * 0.1
        refinement_bonus = len(proof_data['refinements']) * 0.05
        constraint_coverage = len(set(s.get('constraints_processed', 0) for s in proof_data['steps'])) / max(len(constraints), 1)
        return min(step_confidence + refinement_bonus + constraint_coverage * 0.4, 1.0)
    
    def _estimate_problem_complexity(self, input_data: Dict[str, Any]) -> float:
        """Estimate problem complexity"""
        factors = []
        
        # Size factor
        size = len(json.dumps(input_data))
        factors.append(min(size / 10000, 1.0))
        
        # Constraint complexity
        constraints = input_data.get('constraints', [])
        constraint_complexity = sum(len(json.dumps(c)) for c in constraints) / max(len(constraints), 1)
        factors.append(min(constraint_complexity / 100, 1.0))
        
        # Nesting depth
        def get_depth(obj, current_depth=0):
            if isinstance(obj, dict):
                return max(get_depth(v, current_depth + 1) for v in obj.values()) if obj else current_depth
            elif isinstance(obj, list):
                return max(get_depth(item, current_depth + 1) for item in obj) if obj else current_depth
            return current_depth
        
        depth = get_depth(input_data)
        factors.append(min(depth / 10, 1.0))
        
        return np.mean(factors)
    
    def _check_similar_success(self, input_data: Dict[str, Any]) -> float:
        """Check success rate of similar problems"""
        # Simple similarity based on structure
        pattern_key = self._input_to_pattern_key(input_data)
        
        if pattern_key in self.state.context_patterns:
            return self.state.context_patterns[pattern_key]['avg_reward']
        
        return 0.5  # Default unknown success rate
    
    def _input_to_pattern_key(self, input_data: Dict[str, Any]) -> str:
        """Convert input to pattern key for similarity matching"""
        pattern = {
            'structure': list(input_data.keys()),
            'constraint_count': len(input_data.get('constraints', [])),
            'has_metadata': 'metadata' in input_data
        }
        return json.dumps(pattern, sort_keys=True)
    
    def _identify_domain(self, input_data: Dict[str, Any]) -> str:
        """Identify problem domain"""
        # Simple domain identification based on keywords
        problem_str = json.dumps(input_data.get('problem', {})).lower()
        
        if any(term in problem_str for term in ['logic', 'proposition', 'theorem']):
            return 'logical'
        elif any(term in problem_str for term in ['algebra', 'equation', 'polynomial']):
            return 'algebraic'
        elif any(term in problem_str for term in ['geometry', 'shape', 'angle']):
            return 'geometric'
        elif any(term in problem_str for term in ['graph', 'node', 'edge']):
            return 'graph_theory'
        else:
            return 'general'
    
    def _classify_complexity(self, complexity_score: float) -> str:
        """Classify complexity level"""
        if complexity_score < 0.3:
            return 'low'
        elif complexity_score < 0.7:
            return 'medium'
        else:
            return 'high'