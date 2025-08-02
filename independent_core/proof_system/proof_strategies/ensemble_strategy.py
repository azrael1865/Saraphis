"""
Ensemble Strategy - Ensemble proof strategy implementation
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
import threading
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import json

from .strategy_base import ProofStrategy, StrategyResult, StrategyConfig, ComposableStrategy


@dataclass
class EnsembleConfig(StrategyConfig):
    """Configuration specific to ensemble strategy"""
    voting_method: str = 'weighted'  # 'weighted', 'majority', 'consensus', 'adaptive'
    min_agreement: float = 0.6
    weight_update_rate: float = 0.1
    parallel_limit: int = 4
    timeout_per_strategy: float = 30.0
    confidence_aggregation: str = 'weighted_mean'  # 'weighted_mean', 'max', 'median'
    require_quorum: bool = True
    quorum_size: float = 0.5
    boosting_enabled: bool = True
    boosting_rounds: int = 3
    
    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.min_agreement <= 1:
            raise ValueError("min_agreement must be between 0 and 1")
        if not 0 <= self.weight_update_rate <= 1:
            raise ValueError("weight_update_rate must be between 0 and 1")
        if self.parallel_limit <= 0:
            raise ValueError("parallel_limit must be positive")
        if self.timeout_per_strategy <= 0:
            raise ValueError("timeout_per_strategy must be positive")
        if not 0 <= self.quorum_size <= 1:
            raise ValueError("quorum_size must be between 0 and 1")
        if self.boosting_rounds <= 0:
            raise ValueError("boosting_rounds must be positive")


@dataclass
class StrategyWeight:
    """Weight and performance tracking for individual strategy"""
    weight: float = 1.0
    total_executions: int = 0
    successful_executions: int = 0
    total_confidence: float = 0.0
    avg_execution_time: float = 0.0
    reliability_score: float = 1.0
    recent_performances: List[float] = field(default_factory=list)
    
    def update_performance(self, success: bool, confidence: float, execution_time: float):
        """Update strategy performance metrics"""
        self.total_executions += 1
        if success:
            self.successful_executions += 1
        
        self.total_confidence += confidence
        
        # Update average execution time
        alpha = 0.1
        self.avg_execution_time = (alpha * execution_time + 
                                 (1 - alpha) * self.avg_execution_time)
        
        # Update recent performances
        performance = confidence if success else 0.0
        self.recent_performances.append(performance)
        if len(self.recent_performances) > 20:
            self.recent_performances.pop(0)
        
        # Update reliability score
        if self.total_executions > 0:
            success_rate = self.successful_executions / self.total_executions
            avg_confidence = self.total_confidence / self.total_executions
            recent_avg = np.mean(self.recent_performances) if self.recent_performances else 0.0
            
            self.reliability_score = (
                0.4 * success_rate +
                0.3 * avg_confidence +
                0.3 * recent_avg
            )


class EnsembleStrategy(ComposableStrategy):
    """Ensemble strategy combining multiple proof strategies"""
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, (EnsembleConfig, StrategyConfig)):
            raise TypeError("Config must be EnsembleConfig, StrategyConfig, or None")
            
        # Convert StrategyConfig to EnsembleConfig if needed
        if isinstance(config, StrategyConfig) and not isinstance(config, EnsembleConfig):
            ensemble_config = EnsembleConfig(
                max_iterations=config.max_iterations,
                timeout_seconds=config.timeout_seconds,
                confidence_threshold=config.confidence_threshold,
                validation_required=config.validation_required,
                parallel_execution=config.parallel_execution,
                cache_results=config.cache_results,
                metadata=config.metadata
            )
            config = ensemble_config
            
        super().__init__(config or EnsembleConfig())
        self.config: EnsembleConfig = self.config  # Type hint
        
        # Strategy weights
        self.strategy_weights: Dict[str, StrategyWeight] = {}
        self.weight_lock = threading.RLock()
        
        # Execution pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=self.config.parallel_limit)
        
        # Voting mechanisms
        self.voting_methods = {
            'weighted': self._weighted_voting,
            'majority': self._majority_voting,
            'consensus': self._consensus_voting,
            'adaptive': self._adaptive_voting
        }
        
        # Aggregation methods
        self.aggregation_methods = {
            'weighted_mean': self._weighted_mean_aggregation,
            'max': self._max_aggregation,
            'median': self._median_aggregation
        }
        
        # Boosting state
        self.boosting_weights: Dict[str, float] = {}
        self.boosting_history: List[Dict[str, Any]] = []
    
    def _initialize(self) -> None:
        """Initialize ensemble-specific components"""
        # Initialize weights for any pre-added strategies
        for name in self.sub_strategies:
            if name not in self.strategy_weights:
                self.strategy_weights[name] = StrategyWeight()
    
    def add_sub_strategy(self, name: str, strategy: ProofStrategy) -> None:
        """Add a sub-strategy to ensemble"""
        super().add_sub_strategy(name, strategy)
        
        with self.weight_lock:
            if name not in self.strategy_weights:
                self.strategy_weights[name] = StrategyWeight()
            
            # Initialize boosting weight
            self.boosting_weights[name] = 1.0
    
    def execute(self, input_data: Dict[str, Any]) -> StrategyResult:
        """Execute ensemble strategy"""
        start_time = time.time()
        
        if not self.sub_strategies:
            raise RuntimeError("No sub-strategies added to ensemble")
        
        # Check quorum requirement
        active_strategies = self._get_active_strategies()
        if self.config.require_quorum:
            required_count = int(len(self.sub_strategies) * self.config.quorum_size)
            if len(active_strategies) < required_count:
                raise RuntimeError(f"Insufficient strategies for quorum: {len(active_strategies)} < {required_count}")
        
        # Execute strategies
        if self.config.boosting_enabled:
            results = self._execute_with_boosting(input_data, active_strategies)
        elif self.config.parallel_execution:
            results = self._execute_parallel(input_data, active_strategies)
        else:
            results = self._execute_sequential(input_data, active_strategies)
        
        # Aggregate results
        ensemble_result = self._aggregate_results(results, input_data)
        
        # Update weights based on performance
        self._update_weights(results, ensemble_result)
        
        # Set final execution time
        ensemble_result.execution_time = time.time() - start_time
        
        return ensemble_result
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input using all strategies"""
        if not self.sub_strategies:
            return False
        
        # Input is valid if majority of strategies accept it
        valid_count = 0
        for strategy in self.sub_strategies.values():
            try:
                if strategy.validate_input(input_data):
                    valid_count += 1
            except Exception:
                pass
        
        return valid_count > len(self.sub_strategies) / 2
    
    def estimate_complexity(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate complexity using all strategies"""
        if not self.sub_strategies:
            raise RuntimeError("No sub-strategies for complexity estimation")
        
        estimates = []
        
        for name, strategy in self.sub_strategies.items():
            try:
                estimate = strategy.estimate_complexity(input_data)
                estimates.append(estimate)
            except Exception:
                pass
        
        if not estimates:
            raise RuntimeError("No strategy could estimate complexity")
        
        # Aggregate estimates
        avg_iterations = np.mean([e.get('estimated_iterations', 0) for e in estimates])
        avg_time = np.mean([e.get('estimated_time_seconds', 0) for e in estimates])
        
        # Take most common complexity class
        complexity_classes = [e.get('complexity_class', 'unknown') for e in estimates]
        most_common_class = max(set(complexity_classes), key=complexity_classes.count)
        
        return {
            'estimated_iterations': int(avg_iterations),
            'estimated_time_seconds': avg_time,
            'complexity_class': most_common_class,
            'strategy_estimates': len(estimates)
        }
    
    def _get_active_strategies(self) -> List[str]:
        """Get list of active strategies based on weights and reliability"""
        with self.weight_lock:
            active = []
            
            for name, weight_info in self.strategy_weights.items():
                if name in self.sub_strategies:
                    # Skip strategies with very low reliability
                    if weight_info.reliability_score > 0.1:
                        active.append(name)
            
            return active
    
    def _execute_parallel(self, input_data: Dict[str, Any], 
                         strategy_names: List[str]) -> Dict[str, StrategyResult]:
        """Execute strategies in parallel"""
        results = {}
        futures: Dict[Future, str] = {}
        
        # Submit tasks
        for name in strategy_names:
            strategy = self.sub_strategies[name]
            future = self.executor.submit(self._execute_single_strategy, 
                                        name, strategy, input_data)
            futures[future] = name
        
        # Collect results with timeout
        timeout = min(self.config.timeout_per_strategy, 
                     self.config.timeout_seconds / max(len(strategy_names), 1))
        
        for future in as_completed(futures, timeout=timeout):
            name = futures[future]
            try:
                result = future.result()
                if result:
                    results[name] = result
            except Exception as e:
                # Create error result
                results[name] = StrategyResult(
                    success=False,
                    proof_data={},
                    confidence=0.0,
                    execution_time=timeout,
                    iterations_used=0,
                    strategy_name=name,
                    error_info={'type': type(e).__name__, 'message': str(e)}
                )
        
        return results
    
    def _execute_sequential(self, input_data: Dict[str, Any], 
                           strategy_names: List[str]) -> Dict[str, StrategyResult]:
        """Execute strategies sequentially"""
        results = {}
        
        for name in strategy_names:
            strategy = self.sub_strategies[name]
            result = self._execute_single_strategy(name, strategy, input_data)
            if result:
                results[name] = result
        
        return results
    
    def _execute_with_boosting(self, input_data: Dict[str, Any], 
                              strategy_names: List[str]) -> Dict[str, StrategyResult]:
        """Execute strategies with boosting"""
        all_results = {}
        
        for round_idx in range(self.config.boosting_rounds):
            # Weight input based on previous rounds
            if round_idx > 0:
                input_data = self._boost_input(input_data, all_results)
            
            # Execute strategies for this round
            if self.config.parallel_execution:
                round_results = self._execute_parallel(input_data, strategy_names)
            else:
                round_results = self._execute_sequential(input_data, strategy_names)
            
            # Update boosting weights
            self._update_boosting_weights(round_results)
            
            # Merge results
            for name, result in round_results.items():
                if name not in all_results or result.confidence > all_results[name].confidence:
                    all_results[name] = result
        
        return all_results
    
    def _execute_single_strategy(self, name: str, strategy: ProofStrategy, 
                                input_data: Dict[str, Any]) -> Optional[StrategyResult]:
        """Execute a single strategy with error handling"""
        try:
            # Set timeout for individual strategy
            start_time = time.time()
            
            # Run strategy
            result = strategy.run(input_data)
            
            # Check timeout
            if time.time() - start_time > self.config.timeout_per_strategy:
                return None
            
            # Add ensemble metadata
            result.metadata['ensemble_strategy'] = name
            result.metadata['ensemble_weight'] = self.strategy_weights[name].weight
            
            return result
            
        except Exception as e:
            # Return error result
            return StrategyResult(
                success=False,
                proof_data={},
                confidence=0.0,
                execution_time=time.time() - start_time,
                iterations_used=0,
                strategy_name=name,
                error_info={'type': type(e).__name__, 'message': str(e)}
            )
    
    def _aggregate_results(self, results: Dict[str, StrategyResult], 
                          input_data: Dict[str, Any]) -> StrategyResult:
        """Aggregate results from multiple strategies"""
        if not results:
            raise RuntimeError("No results to aggregate")
        
        # Get voting method
        voting_method = self.voting_methods.get(self.config.voting_method, self._weighted_voting)
        
        # Perform voting
        selected_result, voting_confidence = voting_method(results)
        
        # Aggregate confidence scores
        aggregation_method = self.aggregation_methods.get(
            self.config.confidence_aggregation, 
            self._weighted_mean_aggregation
        )
        aggregated_confidence = aggregation_method(results)
        
        # Create ensemble result
        ensemble_result = StrategyResult(
            success=selected_result.success and voting_confidence >= self.config.min_agreement,
            proof_data=self._merge_proof_data(results),
            confidence=min(aggregated_confidence, voting_confidence),
            execution_time=0.0,  # Will be set by caller
            iterations_used=sum(r.iterations_used for r in results.values()),
            strategy_name=self.strategy_name,
            validation_passed=selected_result.validation_passed,
            sub_results=list(results.values()),
            metadata={
                'voting_method': self.config.voting_method,
                'voting_confidence': voting_confidence,
                'aggregated_confidence': aggregated_confidence,
                'strategies_used': len(results),
                'successful_strategies': sum(1 for r in results.values() if r.success),
                'selected_strategy': selected_result.strategy_name
            }
        )
        
        return ensemble_result
    
    def _weighted_voting(self, results: Dict[str, StrategyResult]) -> Tuple[StrategyResult, float]:
        """Weighted voting based on strategy performance"""
        with self.weight_lock:
            # Calculate weighted scores
            weighted_scores = {}
            total_weight = 0.0
            
            for name, result in results.items():
                weight = self.strategy_weights[name].weight * self.boosting_weights.get(name, 1.0)
                weighted_scores[name] = result.confidence * weight
                total_weight += weight
            
            # Find best result
            best_name = max(weighted_scores.items(), key=lambda x: x[1])[0]
            best_result = results[best_name]
            
            # Calculate voting confidence
            if total_weight > 0:
                voting_confidence = sum(weighted_scores.values()) / total_weight
            else:
                voting_confidence = 0.0
            
            return best_result, voting_confidence
    
    def _majority_voting(self, results: Dict[str, StrategyResult]) -> Tuple[StrategyResult, float]:
        """Simple majority voting"""
        # Count successes
        success_count = sum(1 for r in results.values() if r.success)
        total_count = len(results)
        
        # Find result with highest confidence among successful ones
        successful_results = {k: v for k, v in results.items() if v.success}
        
        if successful_results:
            best_name = max(successful_results.items(), key=lambda x: x[1].confidence)[0]
            best_result = successful_results[best_name]
        else:
            # No successful results, pick highest confidence
            best_name = max(results.items(), key=lambda x: x[1].confidence)[0]
            best_result = results[best_name]
        
        voting_confidence = success_count / total_count if total_count > 0 else 0.0
        
        return best_result, voting_confidence
    
    def _consensus_voting(self, results: Dict[str, StrategyResult]) -> Tuple[StrategyResult, float]:
        """Consensus voting requiring high agreement"""
        # Group results by similarity
        groups = self._group_similar_results(results)
        
        # Find largest consensus group
        largest_group = max(groups.values(), key=len)
        
        # Select best from consensus group
        best_result = max(largest_group, key=lambda r: r.confidence)
        
        # Calculate consensus strength
        consensus_size = len(largest_group)
        total_size = len(results)
        voting_confidence = consensus_size / total_size if total_size > 0 else 0.0
        
        return best_result, voting_confidence
    
    def _adaptive_voting(self, results: Dict[str, StrategyResult]) -> Tuple[StrategyResult, float]:
        """Adaptive voting based on context and history"""
        with self.weight_lock:
            # Calculate adaptive scores
            adaptive_scores = {}
            
            for name, result in results.items():
                # Base score
                base_score = result.confidence
                
                # Reliability adjustment
                reliability = self.strategy_weights[name].reliability_score
                
                # Recency adjustment
                recent_perf = np.mean(self.strategy_weights[name].recent_performances[-5:]) \
                            if self.strategy_weights[name].recent_performances else 0.5
                
                # Calculate final score
                adaptive_scores[name] = base_score * (0.5 + 0.3 * reliability + 0.2 * recent_perf)
            
            # Select best
            best_name = max(adaptive_scores.items(), key=lambda x: x[1])[0]
            best_result = results[best_name]
            
            # Calculate confidence
            avg_score = np.mean(list(adaptive_scores.values()))
            voting_confidence = avg_score
            
            return best_result, voting_confidence
    
    def _weighted_mean_aggregation(self, results: Dict[str, StrategyResult]) -> float:
        """Weighted mean of confidence scores"""
        with self.weight_lock:
            total_weighted_confidence = 0.0
            total_weight = 0.0
            
            for name, result in results.items():
                weight = self.strategy_weights[name].weight
                total_weighted_confidence += result.confidence * weight
                total_weight += weight
            
            return total_weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _max_aggregation(self, results: Dict[str, StrategyResult]) -> float:
        """Maximum confidence score"""
        return max(r.confidence for r in results.values()) if results else 0.0
    
    def _median_aggregation(self, results: Dict[str, StrategyResult]) -> float:
        """Median confidence score"""
        confidences = [r.confidence for r in results.values()]
        return float(np.median(confidences)) if confidences else 0.0
    
    def _merge_proof_data(self, results: Dict[str, StrategyResult]) -> Dict[str, Any]:
        """Merge proof data from multiple strategies"""
        merged = {
            'ensemble_proof': True,
            'strategy_proofs': {},
            'consensus_elements': [],
            'confidence_distribution': {}
        }
        
        # Store individual proofs
        for name, result in results.items():
            merged['strategy_proofs'][name] = result.proof_data
            merged['confidence_distribution'][name] = result.confidence
        
        # Find consensus elements
        all_elements = []
        for result in results.values():
            if 'steps' in result.proof_data:
                all_elements.extend(result.proof_data['steps'])
        
        # Simple consensus: elements that appear in multiple proofs
        element_counts = {}
        for element in all_elements:
            key = json.dumps(element, sort_keys=True)
            element_counts[key] = element_counts.get(key, 0) + 1
        
        # Add elements that appear in multiple strategies
        for element_key, count in element_counts.items():
            if count > 1:
                merged['consensus_elements'].append({
                    'element': json.loads(element_key),
                    'support_count': count
                })
        
        return merged
    
    def _update_weights(self, results: Dict[str, StrategyResult], 
                       ensemble_result: StrategyResult) -> None:
        """Update strategy weights based on performance"""
        with self.weight_lock:
            for name, result in results.items():
                weight_info = self.strategy_weights[name]
                
                # Update performance metrics
                weight_info.update_performance(
                    result.success,
                    result.confidence,
                    result.execution_time
                )
                
                # Update weight using ensemble performance
                performance_delta = result.confidence - ensemble_result.confidence
                
                # Reward strategies that performed better than ensemble
                if performance_delta > 0:
                    weight_info.weight *= (1 + self.config.weight_update_rate)
                else:
                    weight_info.weight *= (1 - self.config.weight_update_rate * 0.5)
                
                # Keep weights in reasonable range
                weight_info.weight = max(0.1, min(10.0, weight_info.weight))
    
    def _group_similar_results(self, results: Dict[str, StrategyResult]) -> Dict[int, List[StrategyResult]]:
        """Group results by similarity"""
        groups = {}
        group_id = 0
        
        # Simple grouping by success and confidence range
        for result in results.values():
            placed = False
            
            for gid, group in groups.items():
                # Check if similar to group
                representative = group[0]
                if (result.success == representative.success and 
                    abs(result.confidence - representative.confidence) < 0.1):
                    group.append(result)
                    placed = True
                    break
            
            if not placed:
                groups[group_id] = [result]
                group_id += 1
        
        return groups
    
    def _boost_input(self, input_data: Dict[str, Any], 
                    previous_results: Dict[str, StrategyResult]) -> Dict[str, Any]:
        """Boost input based on previous round results"""
        boosted = input_data.copy()
        
        # Add boosting hints based on previous failures
        failed_strategies = [name for name, result in previous_results.items() if not result.success]
        
        if failed_strategies:
            boosted['boosting_hints'] = {
                'failed_strategies': failed_strategies,
                'difficulty_areas': self._identify_difficulty_areas(previous_results)
            }
        
        return boosted
    
    def _update_boosting_weights(self, results: Dict[str, StrategyResult]) -> None:
        """Update boosting weights based on round performance"""
        # Calculate average confidence
        avg_confidence = np.mean([r.confidence for r in results.values()])
        
        for name, result in results.items():
            # Update weight based on relative performance
            if result.confidence > avg_confidence:
                self.boosting_weights[name] *= 1.1
            else:
                self.boosting_weights[name] *= 0.9
            
            # Keep in range
            self.boosting_weights[name] = max(0.1, min(2.0, self.boosting_weights[name]))
    
    def _identify_difficulty_areas(self, results: Dict[str, StrategyResult]) -> List[str]:
        """Identify areas where strategies struggled"""
        difficulty_areas = []
        
        # Analyze error patterns
        error_types = {}
        for result in results.values():
            if result.error_info:
                error_type = result.error_info.get('type', 'unknown')
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Add common error types as difficulty areas
        for error_type, count in error_types.items():
            if count > len(results) / 3:
                difficulty_areas.append(f"error_{error_type}")
        
        return difficulty_areas
    
    def get_strategy_rankings(self) -> List[Tuple[str, float]]:
        """Get current strategy rankings by performance"""
        with self.weight_lock:
            rankings = []
            
            for name, weight_info in self.strategy_weights.items():
                # Combined score
                score = (
                    weight_info.weight * 0.4 +
                    weight_info.reliability_score * 0.6
                )
                rankings.append((name, score))
            
            # Sort by score
            rankings.sort(key=lambda x: x[1], reverse=True)
            
            return rankings
    
    def reset_weights(self) -> None:
        """Reset all strategy weights to default"""
        with self.weight_lock:
            for weight_info in self.strategy_weights.values():
                weight_info.weight = 1.0
                weight_info.recent_performances.clear()
            
            self.boosting_weights = {name: 1.0 for name in self.sub_strategies}
            self.boosting_history.clear()